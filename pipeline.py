#
# Copyright 2024 Two Sigma Open Source, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from collections import defaultdict
import ray
import pickle
import os
import ast

from prompts.prompt_col_to_dataset_sem_type import col_to_dataset_sem_type_prompt
from prompts.prompt_sem_types_to_general_sem_type import sem_types_to_general_sem_type_prompt
from prompts.prompt_cross_type_cast import cross_type_cast_prompt

from prompt_utils import get_model, count_tokens_in_request
from code_parsing import extract_class_and_mapping_dicts_from_str, run_per_col, build_general_types, \
    add_cross_type_casts
from graph_construction import build_leaves, merge_common_names_across_products, get_root_nodes, \
    get_matches_per_gen_type, pickle_graph
from util import get_col_summary, df_reader, df_reader_v2
from ray_cmds import code_compiling_func, Params, KeyValueStore


def get_avg_num_tokens_in_col_to_dataset_prompt(data_df):
    """
    Get Average Number of Tokens for the Col -> T-FST Prompt

    :param data_df: input dataframe
    :return: avg token count
    """
    p_tokens = []
    for ix, row in data_df.iterrows():
        prompt = col_to_dataset_sem_type_prompt(row.data_product + '/' + row.file_name, row.str_col_summary)
        p_tokens.append(count_tokens_in_request(get_model(), prompt))
    print(f'Avg number of tokens in prompt: {np.array(p_tokens).mean()}')


def get_avg_num_tokens_pfst_to_gfst_prompt(g, root_to_pfsts):
    """
    Get Average Number of TOkens for the (P-FST/T-FST -> G-FST)

    :param g: networkx dag of graph (which is a tree) up until G-FST generation
    :param root_to_pfsts: mapping from G-FST name to matching P-FST/T-FST names
    :return:
    """
    p_tokens = []
    for top_name, matching_roots in root_to_pfsts.items():
        ex_prompt = sem_types_to_general_sem_type_prompt([g.nodes[n]['str_class_def'] for n in matching_roots])
        p_tokens.append(count_tokens_in_request(get_model(), ex_prompt))
    print(f'Avg number of tokens in prompt: {np.array(p_tokens).mean()}')

def get_kv_store_actor():
    """
    To Cache GPT calls, we use the KV Store class, which is bound to a Ray Actor for parallelization.

    :return: kv store actor
    """
    kv_store_pickle_name = 'kv_store.pickle'
    if not os.path.exists(kv_store_pickle_name):
        pickle.dump({}, open(kv_store_pickle_name, 'wb'))

    return KeyValueStore.remote(kv_store_pickle_name, clear_cache=False)

class CreateGraph(metaclass=ABCMeta):
    def __init__(self, data_type):
        self.table_dir = f'assets/{data_type}/tables'
        self.dataset_df_name = f'assets/{data_type}/{data_type}_dataset_data_df.csv'

        self.tfst_data_name = f'assets/{data_type}/{data_type}_tfst_extraction.csv'
        self.tfst_llm_params = None

        self.gfst_data_name = f'assets/{data_type}/{data_type}_gfst_extraction.csv'
        self.gfst_llm_params = None
        self.force_gen_types = False

        self.crosstypecast_data_name = f'assets/{data_type}/{data_type}_crosstypecasts_extraction.csv'
        self.crosstypecast_llm_params = None
        self.use_close_matches_cross_type_cast = False

    @abstractmethod
    def get_input_data(self, use_presaved):
        """
        Input DataFrame Creation

        :param use_presaved: use cached dataframe stored as CSV
        :return: dataframe where each row corresponds to a given table in the universe
        """
        pass

    def get_tfsts_from_llm(self, data_df, use_presaved):
        """
        Gets T-FSTS generated by the LLM, either from cache or using parallelized API call.

        :param data_df: input dataframe
        :param use_presaved: use cached GPT result stored as CSV
        :return: list of strings where each string corresponds to the T-FSTs for a given table
        """
        if use_presaved:
            return list([item[0] for item in pd.read_csv(self.tfst_data_name).values])
        else:
            results = ray.get(
                [
                    code_compiling_func.remote(
                        f'{ix}/{len(data_df)}',
                        (data_df.loc[ix].data_product + '/' + data_df.loc[ix].file_name,
                         data_df.loc[ix].str_col_summary),
                        col_to_dataset_sem_type_prompt,
                        self.tfst_llm_params,
                        kv_store_actor=get_kv_store_actor(),
                        return_non_compiling_code=True
                    ) for ix in data_df.index
                ]
            )
            return results

    def get_gfsts_from_llm(self, g, root_to_matches, use_presaved):
        """
        Get G-FSTs generated by the LLM, either from cache or using parallelized API call.

        :param g: networkx graph (which is a tree at this point) from Col -> T-FSTs -> P-FSTS
        :param root_to_matches: mapping between G-FST name and matching P-FSTs
        :param use_presaved: use cached GPT result stored as CSV
        :return: list of strings where each corresponds to a G-FST for a group of P-FSTs
        """
        if use_presaved:
            return [list(row) for row in pd.read_csv(self.gfst_data_name).values]
        else:
            results = ray.get(
                [
                    code_compiling_func.remote(
                        f'{gfst}: {ix}/{len(root_to_matches)}',
                        [[g.nodes[n]['str_class_def'] for n in matching_roots]],
                        sem_types_to_general_sem_type_prompt,
                        self.gfst_llm_params,
                        return_non_compiling_code=True,
                        kv_store_actor=get_kv_store_actor()
                    )
                    for ix, (gfst, matching_roots) in enumerate(root_to_matches.items())
                ]
            )
            gfst_and_definition = [[root, result] for root, result in zip(root_to_matches.keys(), results)]
            return gfst_and_definition

    def get_cross_type_casts_from_llm(self, g, gfst_to_matches, use_presaved):
        """
        Get cross_type_cast functions generated by the LLM, either from cache or parallelized API call.

        :param g: networkx graph (which is still a tree) from Col -> T-FST -> G-FST
        :param gfst_to_matches: mapping from G-FST to other G-FST classes that are close in vector space.
        :param use_presaved: use cached GPT result stored as CSV
        :return: list of strings where each corresponds to a set of cross_type_cast functions for a G-FST.
        """
        if use_presaved:
            return [list(row) for row in pd.read_csv(self.crosstypecast_data_name).values]
        else:
            cross_type_casts = ray.get(
                [
                    code_compiling_func.remote(
                        f'{gfst}: {ix}/{len(gfst_to_matches)}',
                        [
                            g.nodes[gfst]['str_class_def'],
                            [g.nodes[n]['str_class_def'] for n in matching_gfsts]
                        ],
                        cross_type_cast_prompt,
                        self.crosstypecast_llm_params,
                        return_non_compiling_code=True,
                        kv_store_actor=get_kv_store_actor()
                    )
                    for ix, (gfst, matching_gfsts) in enumerate(gfst_to_matches.items())
                ]
            )
            gfst_and_cross_type_casts = [
                [gfst, cross_type_casts_per_gfst] for gfst, cross_type_casts_per_gfst in
                zip(gfst_to_matches.keys(), cross_type_casts)
            ]
            return gfst_and_cross_type_casts

    def main(self, graph_name, use_presaved=True):
        """
        FST and Graph Generation

        :param graph_name: name of generated graph to be pickle'd at the end.
        :param use_presaved: use cached CSV for generation instead of API calls.
        :return: generated ontology as a networkx graph
        """
        if use_presaved:
            for item in [self.tfst_data_name, self.gfst_data_name, self.crosstypecast_data_name]:
                assert os.path.exists(item), f'{item} doesnt exist!'

        data_df = self.get_input_data(use_presaved)

        tfsts = self.get_tfsts_from_llm(data_df, use_presaved)
        data_df.loc[:, 'tfsts'] = tfsts
        non_compile_ixs = []
        for ix, row in data_df.iterrows():
            try:
                ast.parse(row.tfsts)
            except Exception as e:
                non_compile_ixs.append(ix)

        data_df = data_df.loc[~data_df.index.isin(non_compile_ixs)]
        extract_class_and_mapping_dicts_from_str(data_df, 'tfsts')
        unrolled_df = run_per_col(data_df, self.table_dir)

        g = build_leaves(unrolled_df, self.table_dir)
        merge_common_names_across_products(g)
        root_nodes = get_root_nodes(g)

        for root_node in root_nodes:
            _, _, _, c_name = root_node.split(':')
            assert (c_name.islower()) and (' ' not in c_name) and ('_ ' not in c_name), root_node

        root_to_matches = defaultdict(set)
        for root_node in root_nodes:
            _, _, _, name = root_node.split(':')
            root_to_matches[name].add(root_node)

        gfsts = self.get_gfsts_from_llm(g, root_to_matches, use_presaved)

        build_general_types(g, gfsts, root_to_matches, force_gen_type_name=self.force_gen_types)
        gfst_to_matches = get_matches_per_gen_type(g)

        cross_type_casts = self.get_cross_type_casts_from_llm(g, gfst_to_matches, use_presaved)
        add_cross_type_casts(g, gfst_to_matches, [matches for (root, matches) in cross_type_casts], use_close_matches=self.use_close_matches_cross_type_cast)

        pickle_graph(g, graph_name)
        return g


class HarvardGraphCreator(CreateGraph):
    def __init__(self):
        super().__init__('harvard')

        self.tfst_llm_params = Params()

        self.gfst_llm_params = Params(MAX_TOKENS=4096, USE_CACHE=True)
        self.force_gen_types = True

        self.crosstypecast_llm_params = Params(MAX_TOKENS=4096, USE_CACHE=False)
        self.use_close_matches_cross_type_cast = True

    def get_input_data(self, use_presaved):
        if use_presaved:
            data_df = pd.read_csv(self.dataset_df_name)
            return data_df
        else:
            csv_dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.table_dir, x)),
                                   os.listdir(self.table_dir)))

            info = {}
            for csv_dir in csv_dirs:
                d_files = []
                csv_files = []
                top_csv_dir = os.path.join(self.table_dir, csv_dir)
                for f_name in os.listdir(top_csv_dir):
                    if 'dictionary' in f_name.lower():
                        d_files.append(f_name)
                    else:
                        if f_name.endswith('.csv') and os.access(os.path.join(os.path.join(top_csv_dir, f_name)),
                                                                 os.R_OK):
                            csv_files.append(f_name)
                info[csv_dir] = {'data_dictionary': d_files, 'csvs': csv_files}

            df = pd.DataFrame.from_dict(info, orient='index')

            col_summaries = []
            for dp, row in df.iterrows():
                for f_name in row.csvs:
                    t = df_reader_v2(os.path.join(self.table_dir, dp, f_name), max_rows=1e5)
                    with open(os.path.join(self.table_dir, dp, f_name.replace('.csv', '.txt')), 'r') as r:
                        first_line = r.readline()
                        d_name, d_descrip = first_line.split('|^|')
                    col_count = 0
                    for col in t.columns:
                        if t[col].isna().all():
                            col_count += 1

                    col_summaries.append(
                        [dp, f_name, get_col_summary(t), t.iloc[:5].copy(), col_count, len(t.columns), d_name,
                         d_descrip])

            data_df = pd.DataFrame(col_summaries, columns=['data_product', 'file_name', 'str_col_summary', 'partial_df',
                                                           'num_full_na_cols', 'num_cols', 'd_name', 'd_description'],
                                   dtype=object)
            return data_df

class KaggleGraphCreator(CreateGraph):
    def __init__(self):
        super().__init__('kaggle')
        self.tfst_llm_params = Params(MAX_TOKENS=5000, USE_CACHE=True, USE_LARGE=True)

        self.gfst_llm_params = Params(MAX_TOKENS=4096, USE_CACHE=True)

        self.crosstypecast_llm_params = Params(MAX_TOKENS=4096, USE_CACHE=True)

    def get_input_data(self, use_presaved):
        if use_presaved:
            data_df = pd.read_csv(self.dataset_df_name)
            return data_df
        else:
            csv_dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.table_dir, x)), os.listdir(self.table_dir)))

            info = {}
            for csv_dir in csv_dirs:
                d_files = []
                csv_files = []
                top_csv_dir = os.path.join(self.table_dir, csv_dir)
                for f_name in os.listdir(top_csv_dir):
                    if 'dictionary' in f_name.lower():
                        d_files.append(f_name)
                    else:
                        if f_name.endswith('.csv') and os.access(os.path.join(os.path.join(top_csv_dir, f_name)),
                                                                 os.R_OK):
                            csv_files.append(f_name)
                info[csv_dir] = {'data_dictionary': d_files, 'csvs': csv_files}

            df = pd.DataFrame.from_dict(info, orient='index')

            col_summaries = []
            errors = []
            for dp, row in df.iterrows():
                for f_name in row.csvs:
                    t = df_reader(os.path.join(self.table_dir, dp, f_name), max_rows=1e5)
                    col_summaries.append([dp, f_name, get_col_summary(t), t.iloc[:5].copy()])

            data_df = pd.DataFrame(col_summaries,
                                   columns=['data_product', 'file_name', 'str_col_summary', 'partial_df'], dtype=object)
            return data_df
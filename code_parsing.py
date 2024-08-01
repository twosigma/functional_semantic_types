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
import numpy as np

from semantic_type_base_classes import gen_base_class_file, BASE_CLASSES
from semantic_type_base_classes import create_string_representation_of_imports_for_datasets, \
    create_string_representation_of_imports_for_general_types, \
    create_string_representation_of_imports_for_cross_type_cast

gen_base_class_file()
from semantic_type_base_classes_gen import *

import pandas as pd
import os
from graph_construction import NodeType, alone_context
from prompt_utils import fix_code
import re
from util import df_reader

from collections import defaultdict
import difflib
from unidecode import unidecode
import tqdm


def process_class_name(c_name):
    """
    Basic String parsing on class name.

    :param c_name: string name of a generated class
    :return: new class name
    """
    new_name = ''.join(ch for ch in c_name if ch.isalnum()).lower()
    new_name = re.sub(r'id$', 'identifier', new_name)
    new_name = re.sub(r'desc$', 'description', new_name)
    new_name = re.sub(r'pct$', 'percent', new_name)
    new_name = re.sub(r'percentage$', 'percent', new_name)
    return new_name


def extract_class_and_mapping_dicts_from_str(data_df, sem_type_string_col_name):
    """
    Extract T-FSTs and the mapping from Col -> T-FST from LLM response, stored in a column called "sem_type_string_col_name"

    :param data_df: input dataframe
    :param sem_type_string_col_name: name of column containing LLM response
    :return: dataframe enriched with columns "passes_ast" - if the generated code compiles, "class_dict" - class_name -> T-FST, "mapping_dict" - Col -> T-FST name
    """
    import ast

    data_df.loc[:, 'passes_ast'] = False
    data_df.loc[:, 'class_dict'] = None
    data_df.loc[:, 'mapping_dict'] = None

    FORBIDDEN_CLASS_NAMES = {
        'datetime': 'datetimeclass',
        'round': 'roundclass',
        'yield': 'yieldclass'
    }
    for ix, row in data_df.loc[~data_df[sem_type_string_col_name].isna()].iterrows():
        sem_type_big_string = row[sem_type_string_col_name]
        module = ast.parse(sem_type_big_string)
        classes = {}
        imports = []
        other_inheritor = {}
        mapping = None

        forbidden_class_replace = True
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                node.name = process_class_name(node.name)
                if node.name in FORBIDDEN_CLASS_NAMES:
                    node.name = FORBIDDEN_CLASS_NAMES[node.name]
                    forbidden_class_replace = True

                for func_def in node.body:
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == '__init__':
                        func_def.args.args.append(ast.arg(arg='*args', annotation=None))
                        func_def.args.args.append(ast.arg(arg='**kwargs', annotation=None))

                c_name = node.name
                c_body = ast.unparse(node)
                classes[c_name] = c_body

                if node.bases:
                    base_classes = [base.id for base in node.bases]
                    assert len(base_classes) == 1
                    base_class_name = base_classes[0]
                    if not base_class_name in BASE_CLASSES.keys():
                        base_class_name = process_class_name(base_class_name)
                        if base_class_name in FORBIDDEN_CLASS_NAMES:
                            base_class_name = FORBIDDEN_CLASS_NAMES[base_class_name]
                        other_inheritor[c_name] = base_class_name
            elif isinstance(node, ast.Assign):
                mappings = list(filter(lambda x: isinstance(x, ast.Name) and (x.id.lower() == 'mapping'), node.targets))
                if len(mappings) > 0:
                    mapping_node = node.value
                    assert isinstance(mapping_node, ast.Dict)
                    pop_ixs = []

                    for idx in range(len(mapping_node.keys)):
                        if isinstance(mapping_node.keys[idx], ast.Constant):
                            mapping_node.keys[idx].value = mapping_node.keys[idx].value.strip(
                                ' ')  # sometimes the columns have spaces at the end, sometimes they dont. either way we gotta get rid of them.

                    for idx in range(len(mapping_node.values)):
                        if isinstance(mapping_node.values[idx], ast.Constant):
                            pop_ixs.append(idx)
                            continue

                        if isinstance(mapping_node.values[idx], ast.Call):
                            mapping_node.values[idx] = ast.Name(id=mapping_node.values[
                                idx].func.id)  # sometimes gpt will create a mapping between col_name -> obj, rather than col_name -> class

                        if (mapping_node.values[idx].id in FORBIDDEN_CLASS_NAMES) and forbidden_class_replace:
                            mapping_node.values[idx].id = FORBIDDEN_CLASS_NAMES[mapping_node.values[idx].id]
                        mapping_node.values[idx].id = process_class_name(mapping_node.values[idx].id)
                        if mapping_node.values[idx].id not in classes:
                            pop_ixs.append(idx)  # we want to assert the

                    keys = [mapping_node.keys[ix] for ix in range(len(mapping_node.values)) if ix not in pop_ixs]
                    values = [mapping_node.values[ix] for ix in range(len(mapping_node.values)) if ix not in pop_ixs]

                    mapping_node.keys = keys
                    mapping_node.values = values

                    mapping = ast.unparse(node.value)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))  # need to add imports to the top of the class

        if (mapping is None):
            print('Error w/ mapping: ', ix)

        if len(classes) == 0:
            print('Error w/ classes: ', ix)

        str_classes = {}
        for k, v in classes.items():
            str_classes[k] = create_string_representation_of_imports_for_datasets() + '\n' + '\n'.join(imports) + '\n'
            if k in other_inheritor:
                str_classes[k] += classes[other_inheritor[k]] + '\n' + v
            else:
                str_classes[k] += v

        if len(classes) > 0 and (mapping is not None):
            data_df.at[ix, 'passes_ast'] = True
            data_df.at[ix, 'class_dict'] = str_classes
            data_df.at[ix, 'mapping_dict'] = mapping


def _run_per_col(ix, row, prefix):
    class_obj_map = {}
    col_mapping = {}
    for class_name, string_class_def in row.class_dict.items():
        exec(string_class_def, globals())
        class_obj_map[class_name] = eval(f'{class_name}()')
    col_mapping = eval(row.mapping_dict)

    data_table_head = df_reader(os.path.join(f"{prefix}/{row['data_product']}/{row['file_name']}"), max_rows=1e3)
    kaggle_run_data = []
    for col_name, sem_type_class in col_mapping.items():
        if sem_type_class is None:
            print(col_name, ' is None')
            continue

        sem_type_class_name = sem_type_class.__name__
        if (sem_type_class_name in class_obj_map):
            if col_name in data_table_head.columns:
                col_values = data_table_head[col_name].values
                kaggle_run_data.append([
                    ix,
                    row['data_product'],
                    row.file_name,
                    col_name,
                    col_values,
                    sem_type_class_name,
                    row.class_dict[sem_type_class_name],
                    class_obj_map[sem_type_class_name],
                ])

            else:
                print(col_name, ' not in original table ', ix, data_table_head.columns)
        else:
            print(f'{sem_type_class_name} not in class_obj_map', ix)
    return kaggle_run_data


def run_per_col(data_df, prefix='kaggle_datasets'):
    """
    Unrolls the input dataframe into a new dataframe, where before, each row corresponded to a data-table, but the result
    is a dataframe where each row corresponds to a T-FST.
    """
    kaggle_run_data = []
    for ix in tqdm.tqdm(data_df.loc[data_df.passes_ast].index):
        row = data_df.loc[ix]
        kag_row_data = _run_per_col(ix, row, prefix)
        kaggle_run_data += kag_row_data
    results_df = pd.DataFrame(kaggle_run_data,
                              columns=['df_ix', 'data_product', 'file_name', 'col_name', 'raw_col_values', 'class_name',
                                       'str_class_def', 'obj_class_def'])
    return results_df


def build_general_types(g, name_and_enriched_type_list, general_to_sub_type_map, force_gen_type_name=False):
    """
    Adds G-FSTs to graph given LLM response.

    :param g: networkx graph (tree at this point) containing Col -> T-FST -> P-FST
    :param name_and_enriched_type_list: list of lists, where each sub-list is a G-FST name, and its class (from LLM)
    :param general_to_sub_type_map: mapping from G-FST name to list of P-FST names
    :param force_gen_type_name: sometimes the G-FST class name doesn't match the one specified in general_to_sub_type_map. This boolean forces a renaming.
    :return:
    """
    import ast

    # human corrections
    correction = {
        'stocksereies': 'stockseries',
        'deliverblepercentage': 'deliverablepercentage',
        'stockpercentdeliverble': 'stockpercentdeliverable',
        'availablity': 'availability',
        'sociaeconomicstatus': 'socioeconomicstatus',
    }

    FORBIDDEN_CLASS_NAMES = {
        'round': 'roundclass',
    }
    d_to_create = defaultdict(set)
    for ix, (general_type_name, str_class_def) in enumerate(name_and_enriched_type_list):
        str_class_def = str_class_def.replace('@property', '')
        module = ast.parse(str_class_def)
        classes = {}
        imports = []
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                if node.name in FORBIDDEN_CLASS_NAMES:
                    node.name = FORBIDDEN_CLASS_NAMES[node.name]

                if not force_gen_type_name:
                    c_name = node.name
                else:
                    node.name = general_type_name
                    c_name = general_type_name

                for func_def in node.body:
                    if isinstance(func_def, ast.FunctionDef) and func_def.name == '__init__':
                        func_def.args.args.append(ast.arg(arg='*args', annotation=None))
                        func_def.args.args.append(ast.arg(arg='**kwargs', annotation=None))
                c_body = ast.unparse(node)
                classes[c_name] = c_body
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))  # need to add imports to the top of the class

        if general_type_name in correction:
            actual_class_name = correction[general_type_name]
        else:
            actual_class_name = general_type_name

        string_class_def = create_string_representation_of_imports_for_general_types() + '\n' + '\n'.join(
            imports) + '\n' + classes[actual_class_name]
        obj = alone_context(string_class_def, actual_class_name)

        dst = f'TYPE:_:_:{actual_class_name}'
        if dst not in g.nodes():
            g.add_node(
                dst,
                node_type=NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE,
                str_class_def=string_class_def,
                obj_class_def=obj
            )
        for src in general_to_sub_type_map[general_type_name]:
            g.add_edge(src, dst)


def add_cross_type_casts(g, matches_per_gen_type, cross_type_cast_string_list, use_close_matches=False):
    """
    Adds cross_type_casts to graph given LLM response.

    :param g: networkx graph (tree at this point) from Col -> T-FST -> P-FST -> G-FST
    :param matches_per_gen_type: mapping from G-FST name to other G-FST names that are close in vector space.
    :param cross_type_cast_string_list: LLM response containing cross_type_cast functions per G-FST name. (same length as matches_per_gen_type, and same order)
    :param use_close_matches: sometimes the generated names don't match the graph, so we use close matches
    """
    import ast

    remove_edges = []
    for e in g.edges():
        if 'cross_type_cast' in g.edges[e]:
            remove_edges.append(e)

    if len(remove_edges) > 0:
        print(f'Removing {len(remove_edges)} edges')
        g.remove_edges_from(remove_edges)

    forbidden_name = 'TYPE:_:_:agregaçãodaseleiçõesparecerista3'
    for gen_type, cross_type_cast_string in zip(matches_per_gen_type.keys(), cross_type_cast_string_list):
        if (gen_type in [forbidden_name, unidecode(forbidden_name)]) or (
                cross_type_cast_string in [np.nan, float('NaN'), None]):
            continue

        try:
            module = ast.parse(cross_type_cast_string.strip("\n`"))
        except Exception as e:
            try:
                fixed_code = fix_code(cross_type_cast_string[0], use_gpt=False)
            except Exception as e1:
                raise Exception(gen_type, cross_type_cast_string)
            module = ast.parse(fixed_code)

        # Find the __init__ method and add *args, **kwargs to its arguments
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef):
                has_return = False
                for body in node.body:
                    if isinstance(body, ast.Return):
                        if isinstance(body.value, ast.Constant) and (body.value.value == None):
                            continue
                        has_return = True

                if has_return:
                    relevant_substring = node.name[len('cross_type_cast_between_'):]
                    src, target = relevant_substring.split('_and_')

                    src = ''.join(src.split('_'))
                    target = ''.join(target.split('_'))
                    node.name = f'cross_type_cast_between_{src}_and_{target}'

                    if ('id' in src) and ('id' in target):
                        continue

                    root_node = f'TYPE:_:_:{src}'
                    target_node = f'TYPE:_:_:{target}'
                    if (target_node not in g.nodes()) and use_close_matches:
                        new_target_node = difflib.get_close_matches(target_node, matches_per_gen_type[root_node])[0]
                        if target_node != new_target_node:
                            print(f'**Using {new_target_node} instead of {target_node} ')
                        target_node = new_target_node

                        if root_node != gen_type:
                            print(f'**Using {gen_type} instead of {root_node} ')

                        root_node = unidecode(gen_type)
                        get_name = lambda x: x.split(':')[-1]

                        node.name = f'cross_type_cast_between_{get_name(root_node)}_and_{get_name(target_node)}'
                    assert target_node in g.nodes(), (root_node, target_node)
                    assert root_node in g.nodes(), (root_node, gen_type)

                    g.add_edge(root_node, target_node)
                    g.edges[(root_node, target_node)][
                        'cross_type_cast'] = create_string_representation_of_imports_for_cross_type_cast() + '\n' + ast.unparse(
                        node)

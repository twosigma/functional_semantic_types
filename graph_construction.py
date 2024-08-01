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
from semantic_type_base_classes import gen_base_class_file

gen_base_class_file()
from semantic_type_base_classes_gen import *

import networkx as nx
from enum import Enum
from collections import defaultdict
import numpy as np
import tqdm
import itertools
import os
from sentence_transformers import SentenceTransformer
import pickle


class NodeType(Enum):
    COLUMN = 1
    DATA_SET_SEMANTIC_TYPE = 2
    DATA_PRODUCT_SEMANTIC_TYPE = 3
    GENERAL_ENRICHED_SEMANTIC_TYPE = 4


def build_leaves(results_df, data_dir):
    """
    Builds the leaves of the graph, aka Col -> T-FST edges

    :param results_df: dataframe where each row corresponds to Col -> T-FST
    :param data_dir: directory where raw tables are stored
    :return: networkx graph
    """
    g = nx.DiGraph()

    for ix, row in results_df.iterrows():
        data_product = row['data_product']
        file_name = row['file_name'].replace('.csv', '')

        src = f'COL:{data_product}:{file_name}:{row.col_name}'
        dst = f'TYPE:{data_product}:{file_name}:{row.class_name}'
        assert src not in g.nodes(), g.nodes[src]
        if dst in g.nodes():
            assert g.nodes[dst]['file_name'] == file_name, (ix, src, dst)

        g.add_node(src, node_type=NodeType.COLUMN, col_values=row.raw_col_values)

        if dst not in g:
            g.add_node(
                dst,
                node_type=NodeType.DATA_SET_SEMANTIC_TYPE,
                str_class_def=row.str_class_def,
                obj_class_def=row.obj_class_def,
                data_dir=data_dir,
                dp=data_product,
                file_name=file_name,
            )

        g.add_edge(src, dst)

    return g


def merge_common_names_across_products(g):
    """
    Uses max throughput heuristic to merge T-FSTs with identical names into a P-FSTs

    :param g: networkx graph with Col -> T-FST
    :return: networkx graph with Col -> T-FST -> P-FST
    """
    all_dataset_specific_types = [n for n, data in g.nodes(data=True) if
                                  data['node_type'].value == NodeType.DATA_SET_SEMANTIC_TYPE.value]
    top_level_to_matches = defaultdict(lambda: defaultdict(set))
    for node_name in all_dataset_specific_types:
        _, top_level, bottom_level, name = node_name.split(':')
        top_level_to_matches[top_level][name].add(node_name)

    # here in the merge step we iterate over all data products and group semantic types by their name, then we add edges between
    # the group and the name to create a "DataProduct" Semantic Type that spans all the data-set specific Semantic Types
    for top_level, sem_type_names in top_level_to_matches.items():
        for sem_type_name in sem_type_names:
            matching_node_names = top_level_to_matches[top_level][sem_type_name]
            if len(matching_node_names) == 1:
                continue

            dst = f'TYPE:{top_level}:*:{sem_type_name}'
            g.add_node(
                dst,
                node_type=NodeType.DATA_PRODUCT_SEMANTIC_TYPE
            )

            for src in matching_node_names:
                g.add_edge(src, dst)  # left to right edge from matching data-set specific type -> data product type

    root_nodes = get_root_nodes(g)
    cross_data_product_types = {
        root_node: list(g.predecessors(root_node)) for root_node in root_nodes if
        g.nodes[root_node]['node_type'].value == NodeType.DATA_PRODUCT_SEMANTIC_TYPE.value
    }
    d = {}
    print('Performing matrix cast() calculations...')
    for cross_data_product_type, matching_sub_types in tqdm.tqdm(cross_data_product_types.items()):
        matrix = np.zeros((len(matching_sub_types), len(matching_sub_types), 2))

        sub_type_to_col_vals = defaultdict(set)
        for matching_sub_type in matching_sub_types:
            for pred in g.predecessors(matching_sub_type):
                assert g.nodes[pred]['node_type'].value == NodeType.COLUMN.value
                col_vals = g.nodes[pred]['col_values']
                sub_type_to_col_vals[matching_sub_type] = sub_type_to_col_vals[matching_sub_type].union(col_vals)

        for ix in range(0, len(matching_sub_types)):
            matching_sub_type = matching_sub_types[ix]
            matching_sub_type_obj = g.nodes[matching_sub_type]['obj_class_def']
            for ix_2 in range(0, len(matching_sub_types)):
                all_col_values = sub_type_to_col_vals[matching_sub_types[ix_2]]
                for val in all_col_values:
                    try:
                        new_val = matching_sub_type_obj.cast(val)
                        matrix[ix, ix_2, 0] += 1  # + (0.5 if new_val != val else 0)
                        matrix[ix, ix_2, 1] += int(new_val != val)
                    except Exception as e:
                        pass

        d[cross_data_product_type] = matrix

    d_2 = {}
    print('Performing max() cast() selection')
    for cross_data_product_type, matching_sub_types in tqdm.tqdm(cross_data_product_types.items()):
        matrix = d[cross_data_product_type]
        unique_rows = defaultdict(set)
        for ix in range(len(matrix)):
            unique_rows[tuple(matrix[ix, :, 0])].add(ix)

        max_ix = None
        max_val = -1
        for row_hash, matching_ixs in unique_rows.items():
            summed = sum(row_hash)
            if summed > max_val:
                max_val = summed
                max_change_ix = -1
                for ix in matching_ixs:
                    if matrix[ix, :, 1].sum() > max_change_ix:
                        max_change_ix = ix
                max_ix = max_change_ix
                # ran_ix = list(matching_ixs)[0] # random.choice(list(matching_ixs)) # TODO: This should be replaced with a GPT call to perform the agglomeration
            # if summed > max_val:
            # max_ix = ran_ix

        d_2[cross_data_product_type] = max_ix

    for cross_data_product_type, matching_sub_types in cross_data_product_types.items():
        max_ix = d_2[cross_data_product_type]
        results_per_match = d[cross_data_product_type][max_ix]

        max_matching_sub_type = matching_sub_types[max_ix]
        g.nodes[cross_data_product_type]['str_class_def'] = g.nodes[max_matching_sub_type]['str_class_def']
        g.nodes[cross_data_product_type]['obj_class_def'] = g.nodes[max_matching_sub_type]['obj_class_def']

        for src, cast_passes in zip(matching_sub_types, results_per_match):
            g.edges[(src, cross_data_product_type)]['cast_passes'] = cast_passes

    return g


def get_matches_per_gen_type(g, max_neighbors=20):
    """
    Uses embedding model to find other G-FST classes similar to each G-FST, in vector space. We vectorize each class.

    :param g: networkx graph
    :param max_neighbors: K in K-Nearest Neighbors
    :return: mapping from G-FST (string) -> Matching G-FSTs (list[str])
    """
    gen_types = get_nodes_by_node_type(g, NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    strings = []
    for i in tqdm.tqdm(range(0, len(gen_types))):
        gen_type = gen_types[i]
        c_name = gen_types[i].split(":")[-1]
        obj = alone_context(g.nodes[gen_type]['str_class_def'], c_name)
        variables = vars(obj)
        string = f"Name: {c_name}. Description: {variables['description']}. Format: {variables['format']}."
        strings.append(string)

    vectors = model.encode(strings)

    matches_per_gen_type = {}
    for ix in tqdm.tqdm(range(len(vectors))):
        vec = vectors[ix]
        dist = np.sqrt(((vectors - vec) ** 2).sum(axis=1))
        interesting_ixs = np.argsort(dist)
        interesting_ixs = interesting_ixs[interesting_ixs != ix][:max_neighbors]
        """
        str_prompt = cross_type_cast_semantic_type_prompt(
            g.nodes[root_nodes[ix]]['str_class_def'],
            [g.nodes[root_nodes[ix_2]]['str_class_def'] for ix_2 in interesting_ixs]
        )
        """
        matches_per_gen_type[gen_types[ix]] = [gen_types[gen_ix] for gen_ix in interesting_ixs]

    return matches_per_gen_type


def get_root_nodes(graph):
    """
    Get Nodes with out-degree = 0

    :param graph: nx.DiGraph
    :return: list[str] of node names
    """
    return [n for n in graph.nodes() if (graph.out_degree(n) == 0) and (graph.nodes[n]['node_type'].value >= 2)]


def get_nodes_by_node_type(graph, n_type):
    """
    Get all Nodes with NodeType == n_type

    :param graph: nx.DiGraph
    :param n_type: NodeType
    :return: list[str] of node names
    """
    return [n for n in graph.nodes() if graph.nodes[n]['node_type'].value == n_type.value]


def get_nodes_by_substring_match(graph, sub):
    """
    Get all Nodes with "sub" in their name

    :param graph: nx.DiGraph
    :param sub: str substring
    :return: list[str] of node names
    """
    matches = []
    for n in graph.nodes():
        if sub in n:
            matches.append(n)
    return matches


def predecessors_filtered(g, n):
    """
    Get all predecessors of a node, filtering out connections that use a cross-type-cast

    :param g: nx.DiGraph
    :param n: source node
    :return: list of predecessors
    """
    return list(filter(lambda x: 'cross_type_cast' not in g.edges[(x, n)], g.predecessors(n)))


def dfs(g, n, reverse=True):
    """
    Perform DFS on graph
    """
    visited = set()

    def dfs_helper(n_2):
        if n_2 in visited:
            return

        visited.add(n_2)
        if reverse:
            for pred in predecessors_filtered(g, n_2):
                dfs_helper(pred)
        else:
            for succ in g.successors(n_2):
                dfs_helper(succ)

    dfs_helper(n)
    return visited


def get_downstream_columns(g, root_node):
    """
    Get nodes of type NodeType.Column from any given node in the graph
    """
    cols = []
    visited = set()

    def dfs_helper(n):
        if n in visited:
            return

        preds = predecessors_filtered(g, n)
        visited.add(n)

        if len(preds) == 0:
            return

        for pred in preds:
            dfs_helper(pred)

    dfs_helper(root_node)
    for n in visited:
        if g.nodes[n]['node_type'].value == NodeType.COLUMN.value:
            cols.append(n)
    return sorted(cols)


def get_downstream_columns_and_their_unique_values(g, root_node):
    """
    For each table we start with, we store the values of their columns at g.nodes[n]['col_values'] where n is of type
    NodeType.Column. For a given node in the graph, we look backwards to find the union of all column values.
    """
    unique_set = set()
    values = []
    for col in get_downstream_columns(g, root_node):
        for val in g.nodes[col]['col_values']:
            if val in unique_set:
                continue

            unique_set.add(val)
            values.append(val)
    return values


def get_raw_table_and_columns(g, src, dp, table, reader):
    """
    Given a graph, source node, data product, and table name, we retrieve the matching data.

    :param g: nx.DiGraph
    :param src: source node
    :param dp: data product
    :param table: data table
    :param reader: function to read in a directory and output a dataframe
    :return: np.array of columnar data
    """
    ds_types = get_nodes_by_node_type(g, NodeType.DATA_SET_SEMANTIC_TYPE)

    relevant_ds_types_for_table = []
    for ds_type in ds_types:
        if (g.nodes[ds_type]['dp'] == dp) and (
                (table.strip('.csv') == g.nodes[ds_type]['file_name']) or (
                table.replace('.csv', '') == g.nodes[ds_type]['file_name'])
        ):
            relevant_ds_types_for_table.append(ds_type)

    if len(relevant_ds_types_for_table) == 0:
        print(f'Warning: {dp}/{table} doesnt have any nodes in the graph')

    directory = src + '/' + dp + '/' + table + '.csv'
    all_cols = reader(directory, max_rows=1).columns
    typed_cols = set(itertools.chain(*[get_downstream_columns(g, ds_type) for ds_type in relevant_ds_types_for_table]))
    results = []

    typed_col_names = set()
    for col in typed_cols:
        col_name = col.split(':')[-1]
        typed_col_names.add(col_name)
        results.append([col_name, col, list(g.successors(col))[0]])

    for col_name in all_cols:
        if col_name not in typed_col_names:
            results.append([col_name, '', ''])

    return np.array(results)

def pickle_graph(g, graph_name):
    """
    Pickle the graph, save using copy
    """
    new_g = nx.DiGraph()

    for n, d in g.nodes(data=True):
        new_g.add_node(n, **{k: v for k, v in d.items() if k != 'obj_class_def'})

    for src, dst, d in g.edges(data=True):
        if 'cross_type_cast' in d:
            new_g.add_edge(src, dst, **d)
        else:
            new_g.add_edge(src, dst)

    pickle.dump(new_g, open(graph_name, 'wb'))


def unpickle_graph(graph_name, no_obj_creation=False):
    """
    Unpickle the graph, and load instantioted objects of each class definition stored in the graph
    """
    g = pickle.load(open(graph_name, 'rb'))
    for n, d in g.nodes(data=True):
        if d['node_type'].value >= NodeType.DATA_SET_SEMANTIC_TYPE.value:
            if not no_obj_creation:
                g.nodes[n]['obj_class_def'] = alone_context(d['str_class_def'], n.split(':')[-1])
    return g


def alone_context(str_class_def, class_name):
    """
    Instantiates a class
    """
    exec(str_class_def, locals())
    return eval(f'{class_name}()')


def alone_context_2(str_func_def, func_name):
    """
    Instantiates a class
    """
    exec(str_func_def, locals())
    return eval(f'{func_name}')

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
from graph_construction import NodeType, get_nodes_by_node_type, alone_context, get_downstream_columns, predecessors_filtered, alone_context_2, get_downstream_columns_and_their_unique_values
import tqdm
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def printTree(g, root, markerStr="+- ", levelMarkers=[]):
    emptyStr = " "*len(markerStr)
    connectionStr = "|" + emptyStr[:-1]
    level = len(levelMarkers)   # recursion level
    mapper = lambda draw: connectionStr if draw else emptyStr
    markers = "".join(map(mapper, levelMarkers[:-1]))
    markers += markerStr if level > 0 else ""
    print(f"{markers}{root}")
    # After root has been printed, recurse down (depth-first) the child nodes.
    predecessors = list(predecessors_filtered(g, root))
    for i, child in enumerate(predecessors):
        # The last child will not need connection markers on the current level 
        # (see example above)
        isLast = i == len(predecessors) - 1
        printTree(g, child, markerStr, [*levelMarkers, not isLast])
    
import numpy as np
    
def _analyze_column_to_dataset_cast_func_behavior(obj):
    module = ast.parse(obj)
    multi_body_func = False
    single_line_func_behavior = None
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef):
            for func_def in node.body:
                if func_def.name == 'cast':
                    if len(func_def.body) == 1:
                        return_line = func_def.body[0]
                        if isinstance(return_line, ast.If) or isinstance(return_line, ast.Try):
                            single_line_func_behavior = 'if or try clause'
                            break
                        #print(ast.dump(return_line, indent=4))
                        if isinstance(return_line.value, ast.Name):
                            single_line_func_behavior = 'no-op func'
                        elif isinstance(return_line.value, ast.Call):
                            sub_func = return_line.value.func
                            if isinstance(sub_func, ast.Attribute):
                                single_line_func_behavior = f'attribute_call: {sub_func.attr}'
                            else:
                                single_line_func_behavior = f'func_call: {sub_func.id}'
                        else:
                            single_line_func_behavior = 'explicit_nan_checker'
                    else:
                        multi_body_func = True
                        
    return [multi_body_func, single_line_func_behavior]

def analyze_cast_func_behavior(g, n_type):
    results = []
    assert n_type in [NodeType.DATA_SET_SEMANTIC_TYPE, NodeType.DATA_PRODUCT_SEMANTIC_TYPE]
    ds_types = get_nodes_by_node_type(g, n_type)
    for ds_type in tqdm.tqdm(ds_types):
        results.append([ds_type, *_analyze_column_to_dataset_cast_func_behavior(g.nodes[ds_type]['str_class_def'])])
        
    cast_df = pd.DataFrame(results, columns=['ds_type', 'multibody_func', 'single_line_behavior'])
    return cast_df
        
def col_to_dataset_analysis(g):
    ds_types = get_nodes_by_node_type(g, NodeType.DATA_SET_SEMANTIC_TYPE)
    results = []

    for ds_type in tqdm.tqdm(ds_types):
        c_name = ds_type.split(':')[-1]
        obj = alone_context(g.nodes[ds_type]['str_class_def'], c_name)
        #all_cols = get_downstream_columns(g, ds_type)
        #all_vals = list(itertools.chain(*[g.nodes[col]['col_values'] for col in all_cols]))
        all_vals = get_downstream_columns_and_their_unique_values(g, ds_type)
        for val in all_vals:
            add_to_results_ds_or_dp_type(results, obj, val, [ds_type])
            
                
    results_df = pd.DataFrame(
        results,
        columns=['ds_type', 'input_val_is_null', 'passed', 'changed', 'original_val', 'new_val', 'exception_type', 'str_exception']
    )
    results_df['unique_id'] = [ix for ix in range(len(results_df))]
    return results_df

def col_to_dp_analysis(g, col_to_dataset_results_df):
    dps = get_nodes_by_node_type(g, NodeType.DATA_PRODUCT_SEMANTIC_TYPE)

    results = []
    for dp in tqdm.tqdm(dps):
        dp_preds = list(g.predecessors(dp))
        #sub_df = col_to_dataset_results_df.loc[col_to_dataset_results_df.ds_type.isin(dp_preds)]
        sub_df = col_to_dataset_results_df.loc[col_to_dataset_results_df.ds_type.isin(dp_preds)].drop_duplicates('original_val')
        c_name = dp.split(':')[-1]
        obj = alone_context(g.nodes[dp]['str_class_def'], c_name)
        for ix, row in sub_df.loc[sub_df.ds_type.isin(dp_preds)].iterrows():
            add_to_results_ds_or_dp_type(results, obj, row.original_val, [dp, row.ds_type])
    col_to_dp_results_df = pd.DataFrame(
        results,
        columns=['dp', 'ds_type', 'input_val_is_null', 'passed', 'changed', 'original_val', 'new_val', 'exception_type', 'str_exception']
    )
    col_to_dp_results_df.loc[:, 'unique_id'] = [ix for ix in range(len(col_to_dp_results_df))]

    return col_to_dp_results_df

import warnings
def ds_or_gp_to_gen_analysis(g, col_to_dataset_results_df, col_to_dp_results_df):
    gen_types = get_nodes_by_node_type(g, NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE)
    results = []
    for gen_type in tqdm.tqdm(gen_types):
        gen_type_preds = list(predecessors_filtered(g, gen_type))
        all_preds_df = []
        for pred in gen_type_preds:
            sub_df = None
            if g.nodes[pred]['node_type'].value == NodeType.DATA_PRODUCT_SEMANTIC_TYPE.value:
                sub_df = col_to_dp_results_df.loc[(col_to_dp_results_df.dp == pred) & col_to_dp_results_df.passed]
            else:
                sub_df = col_to_dataset_results_df.loc[(col_to_dataset_results_df.ds_type == pred) & col_to_dataset_results_df.passed]
                if len(sub_df) == 0:
                    print(gen_type, pred, 'no valid passes')
                    continue
                with warnings.catch_warnings(record=True) as w:
                    sub_df.loc[:, 'dp'] = None
            all_preds_df.append(sub_df)
            
        if len(all_preds_df) == 0:
            print(gen_type, 'no valid preds')
            continue
        #all_preds_df = pd.concat(all_preds_df, axis=0)
        all_preds_df = pd.concat(all_preds_df, axis=0).drop_duplicates('new_val')
        c_name = gen_type.split(':')[-1]
        obj = alone_context(g.nodes[gen_type]['str_class_def'], c_name)
        for ix, row in all_preds_df.iterrows():
            add_to_results_gen_type(results, obj, row.new_val, [gen_type, row.dp, row.ds_type])
    
    col_to_gen_type_df = pd.DataFrame(
        results,
        columns=['gen_type', 'dp', 'ds_type', 'input_val_is_null', 'passed', 'changed', 'validated', 'original_val', 'new_val', 'super_cast_exception_type', 'super_cast_str_exception', 'validate_exception_type', 'validate_str_exception']
    )
    col_to_gen_type_df.loc[:, 'unique_id'] = [ix for ix in range(len(col_to_gen_type_df))]
    return col_to_gen_type_df


def cross_type_cast_analysis(g, col_to_gen_type_df):
    gen_types = get_nodes_by_node_type(g, NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE)
    results = []
    for gen_type in tqdm.tqdm(gen_types):
        sub_df = col_to_gen_type_df.loc[(col_to_gen_type_df.gen_type == gen_type) & col_to_gen_type_df.validated]
        for succ in g.successors(gen_type):
            cast_func = g.edges[(gen_type, succ)]['cross_type_cast']
            func_name = f"cross_type_cast_between_{gen_type.split(':')[-1]}_and_{succ.split(':')[-1]}"
            func = alone_context_2(cast_func, func_name)

            second_obj_c_name = succ.split(':')[-1]
            second_obj = alone_context(g.nodes[succ]['str_class_def'], second_obj_c_name)
            for ix, row in sub_df.iterrows():
                original_val = row.new_val
                add_to_results_cross_type_cast(results, func, second_obj, original_val, [gen_type, succ, row.dp, row.ds_type, row.input_val_is_null])
                
    cross_type_casts_df = pd.DataFrame(
        results,
        columns=['src_gen_type', 'dst_gen_type', 'dp', 'ds_type', 'input_val_is_null', 'passed', 'changed', 'validated', 'original_val', 'new_val', 'cross_type_cast_exception_type', 'cross_type_cast_exception', 'validate_dst_exception_type', 'validate_dst_str_exception']
    )
    cross_type_casts_df.loc[:, 'unique_id'] = [ix for ix in range(len(cross_type_casts_df))]
    return cross_type_casts_df
    

def add_to_results_ds_or_dp_type(results, obj, val, extra_info):
    null_status = False
    if isinstance(val, list) or isinstance(val, set) or isinstance(val, np.ndarray):
        if pd.isna(val).all():
            null_status = True
    elif pd.isna(val):
        null_status = True
    input_val = val

    try:
        new = obj.cast(input_val)
        changed = new != input_val
        results.append([*extra_info, null_status, True, changed, input_val, new, None, None])
    except Exception as e:
        if str(e) in [
            'cannot use a string pattern on a bytes-like object',
            'strptime() argument 1 must be str, not float'
        ]:
            add_to_results_ds_or_dp_type(results, obj, str(val), extra_info)
        else:
            results.append([*extra_info, null_status, False, False, input_val, None, str(type(e).__name__), str(e)])
        
from collections.abc import Iterable

def add_to_results_gen_type(results, obj, val, extra_info):
    null_status = False
    if isinstance(val, list) or isinstance(val, set) or isinstance(val, np.ndarray):
        if pd.isna(val).all():
            null_status = True
    elif pd.isna(val):
        null_status = True

    input_val = val
    try:
        new = obj.super_cast(input_val)
        changed = new != input_val
        try:
            validated = obj.validate(input_val)
            if validated is None:
                validated = True
            results.append([*extra_info, null_status, True, changed, validated, input_val, new, None, None, None, None])
        except Exception as e:
            results.append([*extra_info, null_status, True, changed, False, input_val, new, None, None, str(type(e).__name__), str(e)])
    except Exception as e:
        string_e = str(e)
        if (string_e in [
            'cannot use a string pattern on a bytes-like object',
        ]) or ('strptime() argument 1 must be str' in string_e):
            add_to_results_gen_type(results, obj, str(val), extra_info)
        else:
            results.append([*extra_info, null_status, False, False, False, val, None, str(type(e).__name__), str(e), None, None])

        
def add_to_results_cross_type_cast(results, func, second_obj, original_val, extra_info):
    try:
        new_val = func(original_val)
        changed = new_val != original_val
        if isinstance(changed, pd.Series) or isinstance(changed, np.ndarray):
            changed = changed.all()
        elif changed not in [True, False] != bool:
            raise Exception(f'Handle: {new_val}, {original_val}, {changed}')
            
        try:
            validated = second_obj.validate(new_val)
            if validated is None:
                validated = True
            results.append([*extra_info, True, changed, validated, original_val, new_val, None, None, None, None])
        except Exception as e:
            results.append([*extra_info, True, False, False, original_val, None, None, None, str(type(e).__name__), str(e)])
    except Exception as e:
        if str(e).startswith('Handle:'):
            raise e
            
        results.append([*extra_info, False, False, False, original_val, None, str(type(e).__name__), str(e), None, None])  
        
def get_gen_type_usage_df(g, gen_types = None):
    gen_types_downstream_map = {}
    if gen_types is None:
        gen_types = get_nodes_by_node_type(g, NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE)
    else:
        for n in gen_types:
            assert g.nodes[n]['node_type'].value == NodeType.GENERAL_ENRICHED_SEMANTIC_TYPE.value
        
    for n in gen_types:
        down_cols = get_downstream_columns(g, n)
        num_cols = len(set(down_cols))

        across_dps = defaultdict(int)
        across_tables = defaultdict(int)
        for col in down_cols:
            splitted = col.split(':')
            _, dp, file_name, _ = splitted

            across_dps[dp] +=1
            across_tables[f'{dp}/{file_name}'] += 1

        gen_types_downstream_map[n] = {'down_cols': down_cols, 'across_dps': across_dps, 'across_tables': across_tables}   
        
    gen_type_rank_df = pd.DataFrame(gen_types_downstream_map).T
    gen_type_rank_df.index = [thing.split(':')[-1] for thing in gen_type_rank_df.index]
    gen_type_rank_df['num_down_cols'] = gen_type_rank_df.down_cols.map(len)
    gen_type_rank_df['num_across_dps'] = gen_type_rank_df.across_dps.map(len)
    gen_type_rank_df['num_across_tables'] = gen_type_rank_df.across_tables.map(len)
    return gen_type_rank_df

def show_gen_type_usage(g, gen_type_rank_df = None):
    if gen_type_rank_df is None:
        gen_type_rank_df = get_gen_type_usage_df(g)
    sorted_down_cols = gen_type_rank_df.sort_values('num_across_dps', ascending=False)
    fig, ax = plt.subplots(figsize=(30, 10))
    sorted_down_cols[['num_down_cols', 'num_across_dps', 'num_across_tables']].iloc[:100].plot.bar(ax=ax, fontsize=20)
    
    
import os
import json
from graph_construction import get_raw_table_and_columns

def get_dp_human_eval_results(g, src_dir, reader):
    dt_nodes = get_nodes_by_node_type(g, NodeType.DATA_SET_SEMANTIC_TYPE)
    agg_results = []

    for dp in os.listdir(src_dir):
        if dp == 'interesting_results.txt':
            interesting_results = True
            continue

        new_dir = os.path.join(src_dir, dp)
        for f_name in os.listdir(new_dir):
            if f_name.endswith('_results.json'):
                table_name = f_name.replace('_results.json', '')

                results = get_raw_table_and_columns(g, src_dir, dp, table_name, reader)
                all_cols = results[:, 0]

                with open(f'{new_dir}/{f_name}') as f:
                    results_json = json.load(f)
                    filtered_json = {k:v for k,v in results_json.items() if 'unnamed' not in k}
                    all_cols = set([col for col in all_cols if 'unnamed' not in col])
                    assert set(filtered_json.keys()) == all_cols, (f'{new_dir}/{f_name}', set(filtered_json.keys()), all_cols)

                positive_set = set([k for k,v in filtered_json.items() if v[0]])
                type_preds = set(results[np.char.str_len(results[:, 1]) > 0, 0])
                
                for col, col_type_str, ds_type_str, in results:
                    if col not in all_cols:
                        continue
                    human_prediction = 'IS FST' if (col in positive_set) else 'NOT FST'
                    gpt_prediction = 'IS FST' if (col in type_preds) else 'NOT FST'
                    
                    agg_results.append(
                        [
                            f"{dp}/{table_name}",
                            col,
                            col_type_str,
                            ds_type_str,
                            human_prediction,
                            gpt_prediction,
                            filtered_json[col][1],
                        ]
                    )
    df = pd.DataFrame(agg_results, columns=['id', 'col_name', 'col_node', 'ds_node', 'human_prediction', 'gpt_prediction', 'scope'])
    df['unique_id'] = range(len(df))
    return df

def get_cross_human_eval_results(g, src_dir, results_obj_suffix = '_results.json'):
    agg_results = []
    for gen_type_json in os.listdir(src_dir):
        if gen_type_json == 'matches_per_gen.pickle':
            continue
        
        if not gen_type_json.endswith(results_obj_suffix):
            continue 
            
        src_name = gen_type_json.replace(results_obj_suffix, '')
        src_type = f"TYPE:_:_:{src_name}"
        assert src_type in g.nodes()
        gen_type_results = json.load(open(os.path.join(src_dir, gen_type_json), 'r'))
        o_df = pd.DataFrame.from_dict(gen_type_results, orient='index', columns=['relation', 'same_entity', 'relation_type', 'reasoning', 'logic', 'interesting'])
        all_neighbors = set(o_df.index)

        for edge_type in ['direct', 'indirect']:
            true_neighbors = set(o_df.loc[o_df.relation_type.isin(['Bidirectional', 'Uni Src->Dst'])].index)

            if edge_type == 'direct':
                pred_neighbors = set([succ for succ in g.successors(src_type) if 'cross_type_cast' in g.edges[(src_type, succ)]])
            else:
                pred_neighbors = set(nx.dfs_tree(g, src_type, depth_limit=2).nodes()).intersection(all_neighbors) - {src_type}            
                
            for neighbor in all_neighbors:
                human_prediction = 'CASTABLE' if (neighbor in true_neighbors) else 'UNCASTABLE'
                gpt_prediction = 'CASTABLE' if (neighbor in pred_neighbors) else 'UNCASTABLE'
                agg_results.append([src_name, neighbor, edge_type, human_prediction, gpt_prediction, o_df.loc[neighbor, 'relation_type'], o_df.loc[neighbor, 'same_entity'], o_df.loc[neighbor, 'reasoning'], o_df.loc[neighbor, 'logic']])
    cross_df = pd.DataFrame(agg_results, columns=['src', 'dst', 'edge_type', 'human_prediction', 'gpt_prediction', 'relation_type', 'identical', 'reasoning', 'logic'])
    cross_df['unique_id'] = range(len(cross_df))
    return cross_df
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
import pandas as pd
import numpy as np
from jinja2 import Environment, BaseLoader

def get_top_5_most_freq(values):
    """
    Get top-5 most frequent values.
    """
    series = pd.Series(list(values), dtype='object')
    return list(series.value_counts().sort_values(ascending=False).iloc[:5].index)
    
def get_col_summary(df, data_dict=None):
    """
    Get Column Summary of a table.

    :param df: data-table in the universe.
    :param data_dict: data dictionary mapping Column -> Description.
    :return: string column summary
    """
    inferred = df.infer_objects()
    column_summary = ''
    if data_dict is None:
        data_dict = {}
    for col in inferred.columns:
        description = data_dict.get(col, None)
        description = '' if description is None else description
        num_na = np.count_nonzero(inferred[col].isna())  
        if pd.api.types.is_bool_dtype(inferred[col].dtype) or (not pd.api.types.is_numeric_dtype(inferred[col].dtype)) or (inferred[col].isna().all()):
            try:
                average_size = inferred.loc[~inferred[col].isna(), col].astype(str).str.len().mean()
            except Exception as e:
                print(inferred.loc[~inferred[col].isna(), col])
                raise e
            if average_size > 40:
                continue
            try:
                num_unique = len(inferred[col].unique())
                values = [val[:300] if isinstance(val, str) else val for val in inferred[col].values]
                top_5_most_freq = get_top_5_most_freq(values)
                first_five = values[:5]
                column_summary += f'-col:{col}\n' + '\n'.join([
                    f'\t*{k}:{v}' for k,v in [('description', description), ('num_unique', num_unique), ('top_5_most_freq', top_5_most_freq), ('first_five', first_five), ('num_na', num_na)]
                ]) + '\n'
            except Exception as e:
                pass 
        else:
            mean = round(inferred[col].mean(), 3)
            minimum = inferred[col].min()
            maximum = inferred[col].max()
            std = round(inferred[col].std(), 3)
            quantiles = inferred[col].quantile([0.25, 0.5, 0.75])
            q1 = round(quantiles[0.25], 3)
            q2 = round(quantiles[0.5], 3)
            q3 = round(quantiles[0.75], 3)
            first_five = list(inferred[col].values[:5])
            column_summary += f'-col:{col}\n' + '\n'.join([
                f'\t*{k}:{v}' for k,v in [('description', description), ('mean', mean), ('std', std), ('min', minimum), ('0.25', q1), ('0.5', q2), ('0.75', q3), ('max', maximum), ('first_five', first_five), ('num_na', num_na)]
            ]) + '\n'
    
    return column_summary

def col_prep(col):
    """
    Normalization of column name for summary
    """
    col = col.lower()
    col = col.replace('%', 'percent')
    return ''.join(ch for ch in col if ch.isalnum()).strip()

def df_reader(directory, max_rows):
    """
    Reads in a dataframe from a directory, with maximum row specification. Also only loads first 30 columns. Used for Kaggle.
    """
    df = pd.read_csv(directory, nrows=max_rows)
    df.columns = [col_prep(col) for col in df.columns]
    sub_df = df[df.columns[:30]]
    return sub_df.loc[:,~sub_df.columns.duplicated()]

def is_good_name(col):
    for bad_name in ['tsdatasnowflake', 'unnamed']:
        if bad_name in col:
            return False
    return True

def df_reader_v2(directory, max_rows):
    """
    Reads in a dataframe from a directory, with maximum row specification and column filtering. Also only loads first 30 columns. Used for Harvard.
    """
    df = pd.read_csv(directory, nrows=max_rows)
    df.columns = [col_prep(col) for col in df.columns]
    sub_df = df[df.columns[:30]]
    col_mask = [is_good_name(col) for col in sub_df.columns]
    sub_df = sub_df.loc[:, sub_df.columns[col_mask]]
    return sub_df.loc[:,~sub_df.columns.duplicated()]    
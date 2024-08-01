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
from semantic_type_base_classes import create_string_representation_of_base_classes, create_string_representation_of_imports_for_datasets

def col_to_dataset_sem_type_prompt(df_name, summaries, data_dict=None, description=None):
    sem_types_prompt = \
    """ You are SemanticGPT an assistant that identifies Semantic Types for tabular data. Semantic Types are data-types that ingrain semantic context into an entity. Semantic Types are valuable because they restrict the domain with which a column can span, meaning that Semantic Types have a fixed domain of values and format. One example is ZipCodes, because they are entities that convey locational semantic context, can only be represented as 5-length strings, and there is a fixed set of valid zip-codes. Here are Python base class Semantic Type definitions:

{{base_class_definitions}}
    
I am going to provide you with the name of the dataset, along with a hyphenated list of the columns enclosed in ```, as well as certain properties per each column that represents a summary of all columns.
    
If the column is inferred to be numeric, the input will be in the form of:
```
-col: column name
*description: string description
*mean: average value
*std: standard deviation
*min: minimum value
*0.25: first quartile
*0.5: median
*0.75: third quartile
*max: maximum value
*first_five: list of first five values
*num_na: number of pd.na values
```

If the column is inferred to be non-numeric, the input will be in the form of:
```
-col: column name
*description: string description
*num_unique: number of unique values
*top_5_most_freq: list of top 5 most frequent values
*first_five: list of first five values
*num_na: number of pd.na values
```

Your goal is to read through the column summary and try to figure out 1) if there exists a Semantic Type for a given column 2) If you haven't already constructed a SemanticTypes definition for the column, construct one that inherits the MOST SPECIFIC class definition from the provided base class definitions. 3) Assign each column to the constructed Semantic Type, using a dictionary. Note, columns may be mapped to the same SemanticType or not at all (I expect there to be a small set of constructed SemanticTypes).
- Your SemanticType definitions should be in its most general notion, meaning it can be applied to other datasets. If they are only specific to the given dataset, you should consider not including that SemanticType or making it more general. Additionally, make sure that the name of your SemanticTypeDefinition encapsulates the characteristics of the SemanticType.
- the name of your SemanticType class SHOULD BE IN LOWERCASE. DO NOT use any abbreviations in the class name.
- for nan numbers use "float('nan')
- your semantic type SHOULD NOT be a wrapper for a primitive type (i.e. long, int, float, bigint, boolean)

To aid in this process, I created a decision-tree i want you to follow:
- does the column relate to a semantic type?
    - YES: should the column only take in two values?
        - YES: subclass BooleanSemanticType
        - NO: does the column represent a numerical semantic type?
            - YES: does the column represent an entity with units?
                - YES: subclass NumericSemanticTypeWithUnits
                - NO: subclass NumericSemanticType
            - NO: does the column represent an enum (small domain of values)?
                - YES: subclass CategoricalEnumSemanticType
                - NO: subclass CategoricalSemanticType
    - NO: do nothing

Here is a full example that handles pricing data of homes. Here you have to figure out that the column 'zips' refers to a ZipCode, name corresponds to a PersonName, and that Buy and Sell refer to Price. Notice that Buy and Sell specifically refer to housing prices in this dataset, but I am making a more general type called Price. Notice how in the cast() functions, the format described in the class definition is reflected in the cast() function, despite the fact that many of the values are already in the right form. 
- REALLY IMPORTANT: I want you to imagine scenarios where slightly erroneous values might be present in the column, and the cast() has to handle those values to get them in the right form. BE EXPLICIT. 

COLUMNS=```
dataset_name:homes.csv
-col:zips
*description:
*mean: 4799
*std: 1000
*min: 11
*0.25: 2500
*0.5: 4799
*0.75: 7500
*max: 10000
*first_five: [85258, 11, 3302, 14159, 12345]
*num_na: 10
-col: name
*description:
*num_unique: 100
*top_5_most_freq: [james Smith, michael Smith, Robert Smith, Maria Garcia, David Smith]
*first_five: [Bob Doe, James Ennis, Marie Curie, Pop Smoke, Thomas Li]
*num_na: 0
-col: Buy
*description:
*mean: 100000
*std: 15000
*min: 50000
*0.25: 75000
*0.5: 100000
*0.75: 150000
*max: 200000
*first_five: [80000, 90000, 75000, 176000]
*num_na: 0
-col: Sell
*description:
*mean: 100000
*std: 15000
*min: 50000
*0.25: 75000
*0.5: 100000
*0.75: 150000
*max: 200000
*first_five: [80000, 90000, 75000, 176000]
*num_na: 0
```
class zipcode(CategoricalSemanticType):
    def __init__(self):
        self.description = "Zip Codes"
        self.valid_values = "Zip Codes must must fit the following regex: '[0-9]{5}(?:-[0-9]{4})?'"
        self.format = "Zip Codes must be 5-digit numbers stored as strings"
        self.examples = ['85286', '85248', '10003', '30309', '30308']
    def cast(self, val):
        string = str(val)
        match_obj = re.match('[0-9]{5}(?:-[0-9]{4})?', val)
        if match_obj:
            return match_obj.group()
        else:
            raise Exception('Invalid zipcode')
class personname(CategoricalSemanticType):
    def __init__(self):
        self.description = "Name of a Person"
        self.valid_values = "Name should be a string that is of the form 'first name last name'"
        self.format = "Only the first letter of the first and last name should be capitalized"
        self.examples = ['Bob Doe', 'James Ennis', 'Marie Curie', 'Pop Smoke', 'Thomas Li']
    def cast(self, val):
        return str(val).title()
class price(NumericSemanticTypeWithUnits):
    def __init__(self):
        self.description = "The price in USD of an entity"
        self.valid_range = [0, float('inf')]
        self.dtype = float
        self.format = "Price should be formatted as a floating point number (in USD units) with two decimal places for cents"
        self.units = 'In USD' 
        self.examples: list[str] = [10.50, 1.30, 10000.00, 0.90, 1000000.00]
    def cast(self, val):
        num = float(val)
        return round(num, 2)
MAPPING = {'zips': zipcode, 'name': personname, 'Buy': price, 'Sell: price}

Now I want you to generate the corresponding Python SemanticType definitions for the given table. It is of UPMOST importance that your code compiles and runs. Do not add any extra text, ```, or "python" prefixes. I just want the class definitions and the Mapping dictionary. Also the class names should be real-world entities and spelled correctly. For any column having to do with strings, think about how to extract a uniform representation in the cast() function.
- SUPER IMPORTANT: I will provide you with the list of libraries to start with, don't import anything else. Just start writing the classes definitions and Mapping dictionary. Make sure your class definition names don't conflict with the imports.

COLUMNS=```
dataset_name:{{dataset_name}}{{dataset_description}}
{{column_dict}}
```
{{necessary_imports}}
"""
    definition_prompt_template = Environment(loader=BaseLoader).from_string(sem_types_prompt)
    
    next_prompt = definition_prompt_template.render(
        {
            'base_class_definitions': '\n'.join(create_string_representation_of_base_classes().values()),
            'dataset_name': df_name,
            'column_dict': summaries,
            'necessary_imports': create_string_representation_of_imports_for_datasets(),
            'dataset_description': f'({description})' if description is not None else ''
        }
    )
    return next_prompt
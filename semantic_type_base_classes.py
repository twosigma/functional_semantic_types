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
"""This is a dictionary containing a JSON representation of the base classes we use during T-FST Generation"""

BASE_CLASSES = {
    'NumericSemanticType': {
        'description': [str, '', 'short sentence representing the canonical characteristics of this SemanticType'],
        'valid_range': [list, [], "two-length list representing the lower and upper bound of this SemanticType. If either bound is open, use float('inf')"],
        'dtype': [object, None, 'primitive data-type associated with the type (float, int, etc.)'],
        'format': [str, '', 'short sentence that describes the canonical format of this semantic type'],
        'examples': [list, [], '5-length list with 5 examples of the canonical format of this SemanticType, stored as floats'],
        'cast': [None, None, 'Cast value to the canonical format']
    },
    'NumericSemanticTypeWithUnits': {
        'SUPER_CLASS': 'NumericSemanticType',
        'description': [str, '', 'short sentence representing the canonical characteristics of this SemanticType'],
        'valid_range': [list, [], "two-length list representing the lower and upper bound of this SemanticType. If either bound is open, use float('inf')"],
        'dtype': [object, None, 'primitive data-type associated with the type (float, int, etc.)'],
        'format': [str, '', 'short sentence that describes the canonical format of this semantic type'],
        'unit': [str, '', 'short sentence that describes the units and why it was selected'],
        'examples': [list, [], '5-length list with 5 examples of the canonical format of this SemanticType, stored as floats'],
        'cast': [None, None, 'Cast value to the canonical format']
    },
    'CategoricalSemanticType': {
        'description': [str, '', 'short sentence representing the canonical characteristics of this SemanticType'],
        'valid_values': [str, '', "short sentence describing the domain of values that this Categorical variable should take"],
        'format': [str, '', 'short sentence that describes the canonical format of this semantic type'],
        'examples': [list, [], '5-length list with 5 examples of the canonical format of this SemanticType, stored as strings'],
        'cast': [None, None, 'Cast value from to the canonical format']
    },
    'CategoricalEnumSemanticType': {
        'SUPER_CLASS': 'CategoricalSemanticType',
        'description': [str, '', 'short sentence representing the canonical characteristics of this SemanticType'],
        'valid_values': [str, '', "short sentence describing the finite domain of values that this Categorical variable should take"],
        'format': [str, '', 'short sentence that describes the canonical format of this semantic type'],
        'examples': [list, [], '5-length list with 5 examples of the canonical format of this SemanticType, stored as strings'],
        'cast': [None, None, 'Cast value to the canonical format']
    },
    'BooleanSemanticType': {
        'valid_values': [list, [], '2-len list of valid values'],
        'cast': [None, None, 'Cast value from column to a value defined in self.valid_values']
    }
}

"""This a serialized representation of the G-FST base class we use during G-FST Generation"""
GENERAL_SEMANTIC_TYPES = {
    'GeneralSemanticType': {
        'description': [str, '', 'Short string description of what the Type Represents'],
        'format': [str, '', 'Short string description of a single, canonical representation that spans the inputs'],
        'examples': [list, [], '5-length list with 5 examples of the canonical format of this SemanticType, stored as floats'],
        'super_cast': [None, None, 'From any class, convert the expected output of its cast() method, to the format described in self.format'],
        'validate': [None, None, 'Generate validation code that performs sanity-checks on the casted data, i.e. range checking, comparison to average value, etc.\n\t\tcasted_val = self.super_cast(val)']
    },
}

def create_string_representation_of_base_classes():
    """
    Returns a mapping from class_name to string-serialized representation of the base T-FST classes.
    """
    all_accum = {}
    for c_name in BASE_CLASSES:
        string = BASE_CLASSES[c_name]
        all_accum[c_name] = create_string_from_json(c_name, string)
    return all_accum

def create_string_representation_of_general_classes():
    """
    Returns a mapping from class_name to string-serialized representation of the base G-FST classes.
    """
    all_accum = {}
    for c_name in GENERAL_SEMANTIC_TYPES:
        string = GENERAL_SEMANTIC_TYPES[c_name]
        all_accum[c_name] = create_string_from_json(c_name, string)
    return all_accum

def create_string_representation_of_imports_for_datasets():
    """
    Returns a string containing imports needed for T-FSTs
    """
    imports = [
        'import numpy as np',
        'import pandas as pd',
        'from datetime import datetime',
        'import math',
        'import pycountry',
        'import re',
        'from countryinfo import CountryInfo',
        'from semantic_type_base_classes_gen import ' + ','.join([k for k in BASE_CLASSES])
    ]
    return '\n'.join(imports)


def create_string_representation_of_imports_for_general_types():
    """
    Returns a string containing imports needed for G-FSTs
    """
    imports = [
        'import numpy as np',
        'import pandas as pd',
        'from datetime import datetime',
        'import math',
        'import pycountry',
        'import re',
        'from countryinfo import CountryInfo',
        'from semantic_type_base_classes_gen import ' + ','.join([k for k in GENERAL_SEMANTIC_TYPES])
    ]
    return '\n'.join(imports)

def create_string_representation_of_imports_for_cross_type_cast():
    """
    Returns a string containing imports needed for cross_type_casts.
    """
    imports = [
        'import numpy as np',
        'import pandas as pd',
        'from datetime import datetime',
        'import math',
        'import pycountry',
        'import re',
        'from countryinfo import CountryInfo',
    ]
    return '\n'.join(imports)

def create_string_from_json(c_name, class_dict):
    """
    Creates a String Class Definition from a JSON specification.
    """
    accum = f'class {c_name}'
    if 'SUPER_CLASS' in class_dict:
        accum += f"({class_dict['SUPER_CLASS']}):"
    else:
        accum += ':'
    accum += '\n\tdef __init__(self):'
    for k,v in class_dict.items():
        if k == 'SUPER_CLASS':
            continue
        if v[0] is None:
            continue
            
        if isinstance(v, list) and (len(v) == 3):
            accum += f'\n\t\tself.{str(k)}: {v[0].__name__} = {repr(v[1])} # {v[2]}'
        else:
            accum += f'\n\t\tself.{str(k)}= {v}'
    for k,v in class_dict.items():
        if v[0] is not None:
            continue
        accum += f'\n\tdef {k}(self, val): # {v[2]}'
        accum += f'\n\t\tpass'
    return accum

def gen_base_class_file():
    """
    For the string-serialized classes, they get materialized in a generated file.
    """
    with open('semantic_type_base_classes_gen.py', 'w') as f:
        accum = ''
        for k,v in create_string_representation_of_base_classes().items():
            accum += v + '\n'
            
        for k,v in create_string_representation_of_general_classes().items():
            accum += v + '\n'
        f.write(accum)
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
from semantic_type_base_classes import create_string_representation_of_general_classes, create_string_representation_of_imports_for_cross_type_cast

def cross_type_cast_prompt(source_str_class_def, target_str_class_defs):
    sem_types_prompt = \
    """You are CastGPT, an agent that will help me convert between two Semantic Type Class Definitions. These class definitions have a super_cast() method, which converts a value to the class's canonical format, and a validate() method which sanity-checks the result of the validate() method. Given a root class definition and {{len_targets}} target class definitions, I want you to generate at MOST {{len_targets}} cross_type_cast_functions. For example, given root class a and target class b you will generate a method called cross_type_cast_between_a_and_b(val). This is how it will be used:

```
casted_a_val = a().super_cast(val)
casted_b_val = cross_type_cast_between_a_and_b(casted_val)
b().validate(casted_b_val) # should return True if cross_type_cast works correctly.
```

There are two main challenges here. The first is that you need to figure out if class a and class b represent the same type of information, and whether the result of a().super_cast(val) can be casted to the form/function described by b().super_cast(val). If that is possible, then you need to generate the right python mapping code to perform the conversion of a().super_cast(val) to the format of b(). Here is an example where a cross_type that works:
- a=Zipcode, b=City, explanation= both are locational entities, so they are castable.

Here are two examples that DO NOT make sense:
- a=PersonName, b=PersonHeartRate, explanation= there is no way to map a PersonName -> PersonHeartRate, because many people can have the same heart rate.
- a=HotelName, b=Location, explanation= hotels have locations all around the US so theres no way to make a mapping.

The full form of the function is defined as followed. For each (a,b) pairing that is valid (maximum {{len_targets}}), I want you to generate:

```
def cross-cross_type_cast_between_a_and_b(val):
    reason = '' # Tell me why the below code successfully maps a value of the form a().format to b().format. Look at the output of a().super_cast(val) and b().super_cast(va) to help.
    pass
```

Here is a full example where I am given one source type (weightkg) and three target classes (weightlbs, weightgrams, waittime). Notice I don't generate a cross_type_cast_between_weightlbs_and_waittime() because weightkg is a measure of mass, while waittime is a measure of time`:

SOURCE=```
class weightkg(GeneralSemanticType):
	def __init__(self):
		self.description = 'Weight in kilograms'
		self.format = 'float with 2 sigfigs'
	
	def super_cast(self, val):
		return round(float(val), 2)
	
	def validate(self, val):
		casted_val = self.super_cast(val)
		assert casted_val >= 0 and casted_val <= float('inf')
```
TARGETS = ```
class weightlbs(GeneralSemanticType):
	def __init__(self):
		self.description = 'Weight in pounds'
		self.format = 'float with 2 sigfigs'
	
	def super_cast(self, val):
		return round(float(val), 2)
	
	def validate(self, val):
		casted_val = self.super_cast(val)
		assert casted_val >= 0 and casted_val <= float('inf')

class weightgrams(GeneralSemanticType):
	def __init__(self):
		self.description = 'Weight in grams'
		self.format = 'float with 2 sigfigs'
	
	def super_cast(self, val):
		return round(float(val), 2)
	
	def validate(self, val):
		casted_val = self.super_cast(val)
		assert casted_val >= 0 and casted_val <= float('inf')

class waittime(GeneralSemanticType):
	def __init__(self):
		self.description = 'Wait time in hours'
		self.format = 'integer'
	
	def super_cast(self, val):
		return int(val)
	
	def validate(self, val):
		casted_val = self.super_cast(val)
		assert casted_val >= 0 and casted_val <= float('inf')
```
FUNCTIONS = ```
def cross_type_cast_between_weightkg_and_weightlbs(val):
    reason='weightkg and weightlbs both represent the real-world entity, weight. The map between the two is the metric conversion between kg and lbs as seen below.'
    return val*2.20462
def cross_type_cast_between_weightkg_and_weightg(val):
    str_a = 'weightkg and weightg both represent the real-world entity, weight. The map between the two is the metric conversion between kg and g as seen below.'
    return val*1000
```

Here is a partial example that DOES WORK. Here you have a type called countryname (that represents the name of a country) and you want to convert to a type called latlong (that represents a tuple of a latitude and longitude). This will require you to use the CountryInfo library in order to make a valid conversion, but in general, you need to figure out the right libraries to use and avoid incorrect usages:

FUNCTIONS = ```
def cross_type_cast_between_countryname_and_latlong(val):
    return tuple(CountryInfo(val).capital_latlng())
```

Here is a partial example that DOES NOT WORK. Here you have a type called hairtype (as adjectives) and another type called personname. The cross_type_cast() doesnt work because there is no way to get someone's name from a hair description, its nonsensical because many people with the same name can have the same hair type. It also the result would never pass b().validate() nor does it match b().format.

FUNCTIONS = ```
def cross_type_cast_between_hairtype_and_personname(val):
    reason='We can convert from a random hairtype to a personname by simply adding the suffix " Person" to the hairtype, which would make it a name of a person.' # this is invalid
    return val + " Person"
```

Here is a partial example that DOES NOT WORK. Here you have a type called windspeed (in m/s) and another type called error. The cross_type_cast() doesnt work because even though both are floating point numbers, the two relate to completely different measures and have no relation. Avoid casting things just because they are the same primitive, think deep about the entities that the two belong to, and whether those have semantic relations.

FUNCTIONS = ```
def cross_type_cast_between_windspeed_and_error(val):
    reason='Windspeed and error are both floating point statistical measures that related to a real-world entity. They can be casted because they have the same format and validation checks.' # this is invalid
    return val
```

Now I want you to try on the following examples. Like the example just generate the cross_type_cast() functions according to the template I gave you, don't give me anything else but code!
- IMPORTANT: if type types are not cross-type-castable, DO NOT generate an empty cross_type_cast() function, just skip it.
- Also, if you need bizzare mapping code, DO NOT generate a cross_type_cast() function.
- I want you to be EXTREMELY conservative with your conversions. There shouldnt be a lot of converstions that work, because only
small numbers of entities actually represent the same type of information.
- SUPER IMPORTANT: I will provide you with the list of libraries to start with, don't import anything else. Just start writing the cross_type_cast_functions.

SOURCE=```
{{src_class_def}}
```
TARGETS=```
{{target_class_defs}}
```
FUNCTIONS = ```
{{necessary_imports}}
"""
    definition_prompt_template = Environment(loader=BaseLoader).from_string(sem_types_prompt)
    
    next_prompt = definition_prompt_template.render(
        {
            'len_targets': len(target_str_class_defs),
            'src_class_def': source_str_class_def,
            'target_class_defs': '\n\n'.join([str_class.replace('from semantic_type_base_classes_gen import *\n', '').strip() for str_class in target_str_class_defs]),
            'necessary_imports': create_string_representation_of_imports_for_cross_type_cast()
        }
    )
    return next_prompt
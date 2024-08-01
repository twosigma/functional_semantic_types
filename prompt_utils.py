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
from difflib import get_close_matches
import random
import re
from typing import Union

from jinja2 import Environment, BaseLoader


class BaseLMProvider:
    def __init__(self):
        pass

    def templated_prompt(self, prompt_template, input_fields, params):
        """
        Executes call to GPT using prompt, input_fields, and API parameters

        :param prompt_template: jinja template
        :param input_fields: dict of k/v pairs to fill template
        :param params: ray_cmds.Params
        :return: {'texts': [...] # API Response}
        """
        pass


def get_lm_provider(model_name):
    """
    Returns a BaseLMProvider

    :param model_name: name of model for API call
    :return: subclass of BaseLMProvider
    """
    raise NotImplementedError('lm provider class not implemented')


def count_tokens_in_request(lm_provider, request):
    """
    Count tokens in prompt based on model
    """
    raise NotImplementedError('token counter not implemented')


def executor(lm_provider, prompt, params={}):
    """
    Executes a prompt using a LM Provider

    :param lm_provider: BaseLMProvider
    :param prompt: string prompt as jinja template string
    :param params: k/v pairs for the prompt
    :return: tuple(list[str] # list of responses, str # prompt filled with params if specified)
    """
    d = lm_provider.templated_prompt(
        prompt_template=prompt,
        input_fields={},
        params=params
    )
    return d['texts'], prompt


LM_MODEL = 'gpt-4-8k-0613'


def get_model():
    return get_lm_provider(LM_MODEL)


LM_LONG_MODEL = 'gpt-4-32k-0613'


def get_long_model():
    return get_lm_provider(LM_LONG_MODEL)


def fix_code_prompt(string):
    """
    Returns a prompt to fix a non-compiling code response from the LLM

    :param string: non-compiling code response as a string
    :return: string prompt
    """
    code, error = string
    prompt_str = \
        """You have generated some python code for me, but its not working when i run it through the exec() function. I will give you the
code (enclosed in +++) and the error (enclosed in +++) and its your job to give me back the corrected code without any errors. Assume
that the classes: CategoricalSemanticType, NumericSemanticTypeWithUnits, NumericSemanticType, BooleanSemanticType, GenericSemanticType have already been defined.
- IMPORTANT: make the MINIMAL number of edits necessary to fix the code

Here is an example of what I want you to do:

CODE=+++
class serialnumber(NumericSemanticType):
    def __init__(self):
        self.description = "Serial Numbers"
        self.valid_range = [1, float('inf')]
        self.dtype = int
        self.format = "Serial numbers should be positive integers"
        self.examples = [1, 2, 3, 4, 5]
```

```python
MAPPING = {'Sno': serialnumber}
+++
ERROR=+++
invalid syntax (<string>, line 38)
+++
FIXED=+++
class serialnumber(NumericSemanticType):
    def __init__(self):
        self.description = "Serial Numbers"
        self.valid_range = [1, float('inf')]
        self.dtype = int
        self.format = "Serial numbers should be positive integers"
        self.examples = [1, 2, 3, 4, 5]
MAPPING = {'Sno': serialnumber}
+++
    
Now I want you to do the same here:
CODE=+++
{{code}}
+++
ERROR=+++
{{error}}
+++
FIXED=
"""
    definition_prompt_template = Environment(loader=BaseLoader).from_string(prompt_str)

    next_prompt = definition_prompt_template.render(
        {
            'code': code,
            'error': error,
        }
    )

    return next_prompt


def test_exec(string):
    """
    Tests execution of code string

    :param string: code string
    :return: boolean if it compiles
    """
    try:
        exec(string)
        return True
    except Exception as e:
        return str(e)


def quick_doctor(code):
    """
    Quickly applies regex fixes to the string to get it to compile
    """
    code = code.replace("n't", 'nt')
    exec_output = test_exec(code)
    if not isinstance(exec_output, str):
        return code
    else:
        ret = re.search(r"```([\s\S]*)```", code)
        if ret is None:
            return None
        extract = ret.group(1)
        exec_output = test_exec(extract)
        if not isinstance(exec_output, str):
            return extract
        return None


def fix_code(code_and_error, use_gpt=True):
    """
    Executes code fixing
    """
    lm_provider = get_model()
    code, _ = code_and_error
    if not isinstance(test_exec(code), str):
        return code

    quick_fix = quick_doctor(code)
    if quick_fix is not None:
        return quick_fix

    ret = re.search(r"```python(?:\n)?([\s\S]*)(?:```|\n)", code)
    if ret is None:
        if not use_gpt:
            return code_and_error
        else:
            str_prompt = fix_code_prompt(code_and_error)
            fixed_code, _ = executor(lm_provider, str_prompt)
            return format_code_output(fixed_code[0])
    else:
        potensh_code = ret.group(1)
        exec_output = test_exec(potensh_code)
        if not isinstance(exec_output, str):
            return potensh_code
        else:
            if not use_gpt:
                return code_and_error
            else:
                code_and_error = [potensh_code, exec_output]
                str_prompt = fix_code_prompt(code_and_error)
                fixed_code, _ = executor(lm_provider, str_prompt)
                return format_code_output(fixed_code[0])


def format_code_output(string):
    """
    Some basic string stripping to fix code string compilation
    """
    return string.strip('`python"+')

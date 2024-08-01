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
import ray
from prompt_utils import get_model, get_long_model, executor, format_code_output, fix_code

from semantic_type_base_classes import gen_base_class_file
gen_base_class_file()
from semantic_type_base_classes_gen import *

from dataclasses import dataclass
import dataclasses
import pickle


def test_exec(string):
    """
    Tests whether string compiles

    :param string: code string
    :return: if it compiles (bool)
    """
    try:
        exec(string)
        return True
    except Exception as e:
        return str(e)


@dataclass
class Params:
    MAX_RETRIES: int = 2
    MAX_TOKENS: int = 2048
    BATCH_SIZE: int = 1
    USE_LARGE: bool = False
    USE_CACHE: bool = True


def freeze_data_class(dc):
    """
    Freeze DataClass as tuple for KV Storage

    :param dc: of type Params
    :return: tuple of GPT Parameters
    """
    return dataclasses.astuple(dc)

def _code_compiling_func(status, str_prompt, gpt_params, return_non_compiling_code=False):
    """
    Executes a GPT call, with retries if the data doesn't compile
    """
    print(f'Working on {status}')
    if gpt_params.USE_LARGE:
        lm_provider = get_long_model()
    else:
        lm_provider = get_model()

    error_mssg = None
    for i in range(gpt_params.MAX_RETRIES):
        try:
            output_list, _ = executor(lm_provider, str_prompt, {
                "provider_args": {"max_tokens": gpt_params.MAX_TOKENS, "n": gpt_params.BATCH_SIZE}})
            error_outputs = []
            for output in output_list:
                prep_str = format_code_output(output)
                exec_output = test_exec(prep_str)
                if not isinstance(exec_output, str):
                    return prep_str
                else:
                    error_outputs.append([prep_str, exec_output])

            print(f'Nothing worked on {status}, trying to fix the existing code')
            potentially_fixed = fix_code(error_outputs[0])
            if not isinstance(test_exec(potentially_fixed), str):
                return potentially_fixed
            else:
                print(f'Nothing worked on {status}')
                if return_non_compiling_code:
                    return error_outputs
                return None

        except Exception as e:
            error_mssg = str(e)
            print(f'Error on {i}-th try for {status}: ', error_mssg)
            if 'context_length_exceeded' in error_mssg:
                break
    print(f'FAILED on: {status}')
    return None


@ray.remote(max_calls=1)
def code_compiling_func(status, input_args, prompt_func, gpt_params, kv_store_actor=None,
                        return_non_compiling_code=False):
    """
    Used to make API calls that return code

    :param status: status string
    :param input_args: parameters for prompt_func
    :param prompt_func: function that takes in "input_args" and generates a prompt
    :param gpt_params: of type Params(), and specifies parameters for GPT
    :param kv_store_actor: actor Ray remote ref for retrieving cached GPT calls
    :param return_non_compiling_code: whether we return code that doesn't compile
    :return: string with compiling code
    """
    str_prompt = prompt_func(*input_args)

    if gpt_params.USE_CACHE and (kv_store_actor is not None):
        result = ray.get(kv_store_actor.get_value.remote((str_prompt, freeze_data_class(gpt_params))))
        if result is not None:
            print(f'On {status}: using cached result!')
            return result

    result = _code_compiling_func(status, str_prompt, gpt_params, return_non_compiling_code)
    if kv_store_actor is not None:
        ray.get(kv_store_actor.set_value.remote((str_prompt, freeze_data_class(gpt_params)), result))
    return result


@ray.remote
class KeyValueStore:
    def __init__(self, kv_store_pickle_path, clear_cache=False):
        self.path = kv_store_pickle_path
        if not clear_cache:
            with open(self.path, 'rb') as f:
                self.store = pickle.load(f)
        else:
            print('Clearing Cache!')
            self.store = {}

    def get_value(self, key):
        """
        Retrieve from KV Store Cache

        :param key: Key is a tuple of type (prompt, freeze_data_class(Params))
        :return: String Prompt
        """
        if key in self.store:
            return self.store[key]
        else:
            return None

    def set_value(self, key, value):
        """
        Sets KV Store Cache

        :param key: Key is a tuple of type (prompt, freeze_data_class(Params))
        :param value: String Prompt
        :return:
        """
        self.store[key] = value
        with open(self.path, 'wb') as f:
            pickle.dump(self.store, f)

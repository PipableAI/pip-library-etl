import inspect
from typing import List

from pip_library_etl.models import Function
from pip_library_etl.models.plan import Plan
from pip_library_etl.pip_base_class import PipBaseClass

INFERENCE_URL = "https://playground.pipable.ai/infer"


class PipPlanner(PipBaseClass):
    """
    Class for python executable plans.
    """

    def __init__(
        self,
        model_key: str = "PipableAI/base-model-test-planning-data",
        device: str = "cuda",
        url: str = INFERENCE_URL,
    ):
        super().__init__(model_key, device, url)
        self.functions: List[Function] = []

    def add_function(self, signature: str, docs: str, name: str):
        """
        A function to add a new function to the planner functions list if it is not already present.

        Args:
            signature (str): The signature of the function.
            docs (str): Documentation of the function.
            name (str): The name of the function.

        Returns:
            None
        """
        try:
            func = Function(signature=signature, docs=docs, name=name)
            if func not in self.functions:
                self.functions.append(func)
        except Exception as e:
            raise Exception(f"Unable to add function {name} with error {e}.")

    def register_functions(self, functions: List[callable]):
        """
        Registers a list of callable functions with the planner.

        Args:
            functions (List[callable]): A list of callable functions to be registered.

        Raises:
            Exception: If there is an error while registering a function. The exception message will include the name of the function and the error message.

        Returns:
            None
        """
        for function in functions:
            try:
                signature = str(inspect.signature(function))
                docs = function.__doc__
                name = function.__name__
                self.add_function(signature, docs, name)
            except Exception as e:
                print(f"Unable to register function {function} with error {e}.")

    def generate_plan(
        self, question: str, instructions: str = "", max_new_tokens: int = 900
    ) -> Plan:
        """
        Generates a plan based on the given question and instructions.

        Args:
            question (str): The question for which a plan is to be generated.
            instructions (str, optional): Additional instructions for generating the plan. Defaults to "".
            max_new_tokens (int, optional): The maximum number of new tokens to be generated. Defaults to 1360.

        Returns:
            Plan: The generated plan.

        Raises:
            None

        Description:
            This function generates a plan based on the given question and instructions. It first generates a prompt
            using the `_generate_prompt` method. Then, it queries the model using the `_query_model` method and
            retrieves the response. The response is processed to remove 'None' and extract the JSON section. Finally,
            the JSON section is parsed to create a `Plan` object and returned.

        """
        try:
            prompt = self._generate_prompt(question, instructions)
            response = self._query_model(prompt, max_new_tokens)
            response = response.replace("None", "null")
            response = response.split("<json>")[1].split("</json>")[0]
        except Exception as e:
            raise ValueError(f"Unable to generate the plan with error: {e}") from e
        try:
            plan = Plan.model_validate_json(response)
        except Exception as e:
            raise ValueError(
                f"Unable to parse the response: {response} with error: {e}"
            ) from e
        return plan

    def _generate_prompt(self, question: str, instructions: str = "") -> str:
        func_info = [
            f"""--name:{function.name}\n--annotations:{function.signature}\n--doc:{function.docs}\n\n"""
            for function in self.functions
        ]
        prompt = f"""
<functions>
{func_info}
</functions>
<json_structure>
{{
  "tasks": [
    {{
      "task_id": 1,
      "function_name": "some_library.some_function",
      "parameters": [
        {{
        "name":"name of this parameter according to annotations.",
        "value":"value to be passed for this parameter",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why this value should be utilized.",
        }},
        {{
        "name":"name of this parameter according to annotations..",
        "value":"value to be passed for this parameter.",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why the above mentioned value should be utilized for this parameter.",
        }}
      ],
      "outputs": ["variable_1"],
      "description": "some description",
    }},
    {{
      "task_id": 2,
      "function_name": "some_library_2.some_random_function",
      "parameters": [
        {{,
        "name":"name of this parameter according to annotations.",
        "value":"value to be passed for this parameter",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why this value should be utilized.",
        }},
        {{
        "name":"self",
        "value":"variable name to be passed for this parameter self.",
        "dtype":"type annotation of the self parameter",
        "description": "An explanation of why the cariable should be used for this self parameter.",
        }},
        {{,
        "name":"name of this parameter according to annotations.",
        "value":"value to be passed for this parameter",
        "dtype":"type annotation of the variable",
        "description": "An explanation of why this value should be utilized.",
        }},
      ],
      "outputs": ["variable_2"],
      "description": "some description",
    }}
  ]
}}
</json_structure>
<instructions>
1. Include self as parameter name in the parameters list if mentioned on the function's annotation for some task.
2. Use self as parameter name in the parameters list if mentioned on the function's annotation for some task.
3. Try to name outputs as variable1 , variable2 and so on in chronological order.
4. Give attention to the type annotation of the parameter while filling values for it.
{instructions}
</instructions>
<question>
Given the above functions,
- Do not give the parameters in json which have null values and default values of the function, only give the sequencial function calls with parameters to execute the below question:
{question}
</question>
<json>
"""
        return prompt

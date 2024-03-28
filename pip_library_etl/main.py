import importlib
import inspect
import json
from typing import Any

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


class PipEtl:
    """
    Class for generating documentation and SQL queries using a Pipable model.
    """

    def __init__(
        self,
        model_key="PipableAI/pip-library-etl-1.3b",
        device="cuda",
        url="http://52.165.83.89:3100/infer",
    ):
        self.device = device
        self.model_key = model_key
        self.model = None
        self.tokenizer = None
        self.url = None
        if self.device == "cloud":
            self.url = url
        else:
            self._load_model()

    def query_model(self, prompt: str, max_new_tokens: int) -> str:
        if self.device == "cloud":
            payload = {
                "model_name": "PipableAI/pip-library-etl-1.3b",
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
            response = requests.request(
                method="POST", url=self.url, data=payload, timeout=120
            )
            if response.status_code == 200:
                return json.loads(response.text)["response"]
            else:
                raise Exception(f"Error generating response using {self.url}.")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_key).to(
                self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_key)

    def generate_docstring(self, code: str) -> str:
        """
        Generate a docstring for Python code using a local GPU-based model loaded from Hugging Face.

        Args:
        - code (str): The Python code for which a docstring needs to be generated.

        Returns:
        - str: The generated docstring for the input code.

        Note:
        This function loads a GPU-based language model from Hugging Face and utilizes it to analyze the provided code,
        generating a corresponding docstring.

        """
        try:
            prompt = f"""
           <example_response>
            --code:def divide_by_two(x: float) -> float: return x / 2
            --question:Document the python code above giving function description ,parameters and return type and example on how to call the function
            --doc:
            Description: This function divides a given number by 2.
            Parameters:
            - x (float): The input value to be divided by 2.
            Returns:
            - float: The result of x divided by 2.
            Example:
            divide_by_two(1.0)
            </example_response>
            <function_code>{code}</function_code>
            <instructions> 
            1. In the examples while calling function use the name mentioned after `def ` in the above function_code.
            2. In the generated docs use valid python type hints as per PEP 484.
            </instructions>
            <question>Document the python code above giving function description ,parameters and return type and example on how to call the function</question>
            <doc>"""
            res = self.query_model(prompt, 450)
            doc = res.split("<doc>")[-1].split("</doc>")[0]
            doc = (
                doc.replace("<p>", "")
                .replace("</p>", "")
                .replace("<function_description>", "")
                .replace("</function_description>", "")
            )
            return doc
        except Exception as e:
            message = f"Unable to generate the docs using model with error: {e}"
            raise ValueError(message) from e

    def generate_module_docstrings(self, module: Any, module_name: str) -> dict:
        """
        Generate documentation for all methods and functions in a given module.

        Args:
        - module (Any): The module or package to inspect.
        - module_name (str): The name of the module or package.

        Returns:
        - dict: A dictionary mapping method/function names to their corresponding generated docstrings.
        """
        complete_docs = {}

        # Replace 'get_all_methods_and_functions' with your actual implementation
        code_data = self._get_all_methods_and_functions(module, module_name)

        try:
            for function, code in code_data.items():
                print(f"Generating docs for {function}:")

                try:
                    doc = self.generate_docstring(code)
                except ValueError as e:
                    print(e)

                complete_docs[function] = doc

                print(f"Doc for {function}:\n{doc}\n")

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: Returning the latest complete_docs.")
            return complete_docs
        else:
            return complete_docs

    def generate_sql(
        self, schema: str, question: str, instructions: str = None, examples: str = None
    ) -> str:
        """
        Generate SQL queries based on the provided schema and question.

        Args:
            schema (str): The schema for the SQL query.
            question (str): The question related to the SQL query.
            instructions (str, optional): Additional instructions for generating the SQL query. Defaults to None.
            examples (str, optional): An examples for generating the SQL query. Defaults to None.

        Returns:
            str: The generated SQL query.

        Raises:
            ValueError: If unable to generate the SQL query using the model.

        """
        try:
            prompt = "Generate simple SQL queries from the schema mentioned for the following questions."

            if instructions:
                prompt += f"\n<instructions>{instructions}</instructions>"

            if examples:
                prompt += f"\n<example>{examples}</example>"

            prompt += f"""
            <schema>{schema}</schema>
            <question>{question}</question>
            <sql>"""
            res = self.query_model(prompt, 300)
            sql_section = res.split("<sql>")[1].split("</sql>")[0]

            sql_section = sql_section.replace("<p>", "").replace("</p>", "")

            return sql_section

        except Exception as e:
            message = f"Unable to generate the SQL query using model with error: {e}"
            raise ValueError(message) from e

    def _get_all_methods_and_functions(self, module: Any, module_name: str):
        """
        Retrieve methods and functions along with their source code from a module or package.

        Args:
        - module (Any): The module or package to inspect.
        - module_name (str): The name of the module or package.

        Returns:
        - dict: A dictionary mapping method/function names to their corresponding source code.

        Note:
        This function recursively explores the module or package and its submodules to extract
        methods and functions along with their source code.

        """
        function_to_code_data = {}
        already_done = {}

        def _helper_function(module_or_class: Any, path: str):
            try:
                for name, obj in inspect.getmembers(module_or_class):
                    if name.startswith("_"):
                        continue
                    complete_path = path + "." + name
                    if inspect.isclass(obj) or inspect.ismodule(obj):
                        if type(obj).__name__ == "module":
                            try:
                                importlib.import_module(complete_path)
                            except ModuleNotFoundError:
                                if name in already_done or not name.startswith(
                                    f"{module_name}."
                                ):
                                    continue
                                already_done[name] = 1
                        _helper_function(obj, path + "." + name)
                    elif inspect.ismethod(obj) or inspect.isfunction(obj):
                        function_to_code_data[complete_path] = str(
                            inspect.getsource(obj)
                        )
            except TypeError as e:
                print(f"Unable to extract code for {path} with Error: {e}")

        _helper_function(module, module_name)
        return function_to_code_data

    def generate_function_call(self, docstring: str, question: str):
        """
        Generates a function call in Python language based on a given question.

        Args:
            docstring (str): The documentation string template for the function.
            question (str): The question prompting the function call generation.

        Returns:
            str: The Python function call generated based on the question and the provided docstring template.
        """
        prompt = f"""
        Give a function call in python langugae for the following question:
        <example_response>
        --doc:
        Description: This function logs a curl command in debug mode.
        Parameters:
        - method (str): The HTTP method to use for the request.
        - url (str): The URL to send the request to.
        - data (dict, optional): The data to send in the request. Defaults to None.
        - headers (dict, optional): The headers to send with the request. Defaults to None.
        - level (int, optional): The log level to use for this log message. Defaults to logging.DEBUG.
        Returns:
        - None
        Example:
        log_curl_debug('GET', 'https://example.com')
        --question: log a curl PUT request for url https://web.io/
        --function_call: log_curl_debug(method='PUT', url = 'https://web.io')
        </example_response>
        <doc>
        {docstring}
        </doc>
        <instruction>
        1. Strictly use named parameters mentioned in the doc to generate function calls.
        2. Only return the response as python parsable string version of function call.
        3. mention the 'self' parameter if required.
        </instruction>
        <question>
        {question}
        </question>
        <function_call>
        """
        res = self.query_model(prompt, 200)
        result = res.split("<function_call>")[1].split("</function_call>")[0]
        return result

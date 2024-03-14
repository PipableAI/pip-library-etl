import importlib
import inspect
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_KEY = "PipableAI/pip-code-to-doc-1.3b"

model = AutoModelForCausalLM.from_pretrained(MODEL_KEY).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY)


def generate_docstring_from_pip_model(code: str) -> str:
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
        prompt = f"""<code>{code}</code>
        <question>Document the code above</question>
        <doc>"""
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=300)
        doc = (
            tokenizer.decode(outputs[0], skip_special_tokens=True)
            .split("<doc>")[-1]
            .split("</doc>")[0]
        )
        doc = doc.replace("<p>", "").replace("</p>", "")
        return doc
    except Exception as e:
        message = f"Unable to generate the docs using model with error: {e}"
        raise ValueError(message) from e


def get_all_methods_and_functions(module: Any, module_name: str):
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
                    function_to_code_data[complete_path] = str(inspect.getsource(obj))
        except TypeError as e:
            print(
                f"Unable to extract code for {path} with Error: {e}"
            )

    _helper_function(module, module_name)
    return function_to_code_data


def generate_module_docs(module: Any, module_name: str) -> dict:
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
    code_data = get_all_methods_and_functions(module, module_name)

    try:
        for function, code in code_data.items():
            print(f"Generating docs for {function}:")

            try:
                doc = generate_docstring_from_pip_model(code)
            except ValueError as e:
                print(e)

            complete_docs[function] = doc

            print(f"Doc for {function}:\n{doc}\n")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: Returning the latest complete_docs.")
        return complete_docs
    else:
        return complete_docs

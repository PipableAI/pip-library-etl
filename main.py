import importlib
import inspect
from typing import Any

already_done = {}


def code_to_docstring(code: str) -> str:
    print(code)


# str(inspect.getsource(obj)),
def get_methods(module_or_class: Any, path: str, all_functions: dict):
    try:
        for name, obj in inspect.getmembers(module_or_class):
            if name.startswith("_"):
                continue
            complete_path = path + "." + name
            if inspect.isclass(obj) or inspect.ismodule(obj):
                if type(obj).__name__ == "module":
                    try:
                        module_or_class = importlib.import_module(complete_path)
                    except ModuleNotFoundError:
                        if name in already_done or not name.startswith(f"{NAME}."):
                            continue
                        already_done[name] = 1
                get_methods(obj, path + "." + name, all_functions)
            elif inspect.ismethod(obj) or inspect.isfunction(obj):
                all_functions[complete_path] = str(inspect.getsource(obj))
    except Exception as e:
        print(e)


import pandas

NAME = "pandas"
get_methods(pandas, NAME, {})

# print(all_functions)

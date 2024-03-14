# pip_library_parser

`pip_library_parser` is a Python package designed to facilitate the automatic generation of docstrings for functions and methods within a specified module. It leverages [pip-code-to-doc-1.3b](https://huggingface.co/PipableAI/pip-code-to-doc-1.3b) language model to analyze Python code and produce informative docstrings.

## Installation

```bash
pip3 install git+https://github.com/PipableAI/pip-library-parser
```

## Usage

### Generate Docstrings for Functions and Methods

```python
from pip_library_parser import CodeToDocGenerator

# Replace 'your_module' and 'YourModule' with the actual module and module name
module_name = 'your_module'
module = __import__(module_name)

# Instantiate the CodeToDocGenerator
generator = CodeToDocGenerator()

# Generate docstrings for the module's functions and methods
docs = generator.generate_module_docs(module, module_name)

# 'docs' now contains a dictionary mapping function/method names to their generated docstrings
```

### Example: Generate Docstring for a Single Code snippet

```python
from pip_library_parser import CodeToDocGenerator

# Instantiate the CodeToDocGenerator
generator = CodeToDocGenerator()

code_snippet = """
def example_function(x):
    return x * 2
"""

docstring = generator.generate_docstring_from_pip_model(code_snippet)
print("Generated Docstring:")
print(docstring)
```

### Changing Model and Device

The `CodeToDocGenerator` class allows you to change the huggingface pip model and device while initializing the object. By default, it uses the model key `PipableAI/pip-code-to-doc-1.3b` and the device `cuda`. You can specify different models and devices by providing arguments during initialization. (Make sure the prompt of the new model is same as that of `PipableAI/pip-code-to-doc-1.3b`)

```python
# Example: Instantiate CodeToDocGenerator with a different model and device
generator = CodeToDocGenerator(model_key="your_custom_model", device="cpu")
```

## How It Works

- `generate_docstring_from_pip_model`: Utilizes a GPU-based language model from Hugging Face to analyze Python code and generate corresponding docstrings.
- `get_all_methods_and_functions`: Recursively explores a module or package, extracting methods and functions along with their source code.
- `generate_module_docs`: Generates documentation for all methods and functions in a specified module using the above functionalities.

## Dependencies

- `transformers` from Hugging Face

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
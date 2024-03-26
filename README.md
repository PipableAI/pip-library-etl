# pip_library_etl

`pip_library_etl`is a Python package aimed at simplifying the process of generating docstrings for functions and methods within a designated module, as well as generating SQL queries for a specified schema. It harnesses the power of the [PipableAI/pip-library-etl-1.3b](https://huggingface.co/PipableAI/pip-library-etl-1.3b) language model to do all the tasks.

For more examples: [notebook](https://colab.research.google.com/drive/17PyMU_3QN9LROy7x-jmaema0cuLRzBvc?usp=sharing)

## Installation

```bash
pip3 install git+https://github.com/PipableAI/pip-library-etl.git
```

## Usage

### Generate Docstrings for Functions and Methods

```python
from pip_library_etl import PipEtl

# Replace 'your_module' and 'YourModule' with the actual module and module name
module_name = 'your_module'
module = __import__(module_name)

# Instantiate the PipEtl
generator = PipEtl()

# Generate docstrings for the module's functions and methods
docs = generator.generate_module_docs(module, module_name)

# 'docs' now contains a dictionary mapping function/method names to their generated docstrings
```

### Example: Generate Docstring for a Single Code snippet

```python
from pip_library_etl import PipEtl

# Instantiate the PipEtl
generator = PipEtl()

code_snippet = """
def example_function(x):
    return x * 2
"""

docstring = generator.generate_docstring(code_snippet)
print("Generated Docstring:")
print(docstring)
```

### Example: Generate SQL queries
```python

instructions = """
1. In department table, column Budget_in_Billions is in billions, so 1 will represent 1 billion
"""

schema = f"""
<schema>CREATE TABLE department (Department_ID number,
  Name text,
  Creation text,
  Ranking number,
  Budget_in_Billions number,
  Num_Employees number);

  CREATE TABLE head (head_ID number,
  name text,
  born_state text,
  age number);

  CREATE TABLE management (department_ID number,
  head_ID number,
  temporary_acting text);</schema>
"""

question = "What are the names of the heads who are born outside the California state ?"

generator = PipEtl()

query = generator.generate_sql(schema=schema, question=question, instructions=instructions)
print("Generated SQL:")
print(query)
```
### Changing Model and Device

The `PipEtl` class allows you to change the huggingface pip model and device while initializing the object. By default, it uses the model key `PipableAI/pip-library-etl-1.3b` and the device `cuda`. You can specify different models and devices by providing arguments during initialization. (Make sure the prompt of the new model is same as that of `PipableAI/pip-library-etl-1.3b`)

```python
# Example: Instantiate PipEtl with a different model and device
generator = PipEtl(model_key="your_custom_model", device="cpu")
```

## How It Works

- `generate_docstring`: Utilizes a GPU-based language model to analyze Python code and generate corresponding docstrings.
- `generate_module_docstrings`: Generates documentation for all methods and functions in a specified module.
- `generate_sql`: Generate SQL queries based on the provided schema and question.

## Dependencies

- `transformers` from Hugging Face

## Contributing

Feel free to contribute to the project by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
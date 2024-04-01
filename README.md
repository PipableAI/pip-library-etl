# pip_library_etl

`pip_library_etl` is a Python package designed to streamline the creation of docstrings for functions and methods within specified modules, while also providing the capability to effortlessly generate SQL queries tailored to a designated schema. With its intuitive functionality, users can seamlessly generate function calls from natural language input, existing function signatures, or craft new ones using the integrated model. This versatile tool not only enhances documentation generation but also empowers users to compose sophisticated SQL queries with the finesse expected from cutting-edge Language Model Models (LLMs). Elevate your development and data management workflows with the comprehensive capabilities of pip_library_etl.

It harnesses the power of the [PipableAI/pip-library-etl-1.3b](https://huggingface.co/PipableAI/pip-library-etl-1.3b) language model to do all the tasks.

For more examples: [notebook](https://colab.research.google.com/drive/17PyMU_3QN9LROy7x-jmaema0cuLRzBvc?usp=sharing)

## Installation

```bash
pip3 install git+https://github.com/PipableAI/pip-library-etl.git
```

## Usage


### NOTE

If you want to try this model without using your GPU, we have hosted the model on our end.
To run the library using the playground hosted model, initialize the generator in the following way:

```python
generator = PipEtl(device="cloud")
```

If you want to use your own GPU of the local machine (at least 10-12 GB VRAM):

```python
generator = PipEtl(device="cuda")
```

If you want to infer on the CPU of the local machine:

```python
generator = PipEtl(device="cpu")
```

### Example: Function Calling
```python
docstring = """
Function Name: make_get_req
Description: This function is used to make a GET request.
Parameters:
- path (str): The path of the URL to be requested.
- data (dict): The data to be sent in the body of the request.
- flags (dict): The flags to be sent in the request.
- params (dict): The parameters to be sent in the request.
- headers (dict): The headers to be sent in the request.
- not_json_response (bool): OPTIONAL: If set to True, the function will return the raw response content instead of trying to parse it as JSON.
- trailing (str): OPTIONAL: For wrapping slash symbol in the end of string.
- absolute (bool): OPTIONAL: If set to True, the function will not prefix the URL with the base URL.
- advanced_mode (bool): OPTIONAL: If set to True, the function will return the raw response instead of trying to parse it as JSON.
Returns:
- Union[str, dict, list, None]: The response content as a string, a dictionary, a list, or None if the response was not successful.
"""

question = """
Make a GET request for the URL parameter using variable_2. For the params parameter, use 'weight' as one of the keys with variable_3 as its value, and 'width' as another key with a value of 10. For the data parameter, use variable_1. Prefix the URL with the base URL, and ensure the response is in raw format.
"""

function_call = generator.generate_function_call(docstring=docstring, question=question)

print(function_call)
```

```python
code = """
def _query_model(prompt: str, max_new_tokens: int) -> str:
    if device == "cloud":
        payload = {
            "model_name": "PipableAI/pip-library-etl-1.3b",
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        response = requests.request(
            method="POST", url=url, data=payload, timeout=120
        )
        if response.status_code == 200:
            return json.loads(response.text)["response"]
        else:
            raise Exception(f"Error generating response using url.")
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
"""

question = """
I want to query model with prompt = "What is 2 + 2", and use 200 as maximum token limit.
"""

function_call = generator.generate_function_call(code=code, question=question)

print(function_call)
```


### Generate Docstrings for Functions and Methods

```python
from pip_library_etl import PipEtl

# Instantiate the PipEtl
generator = PipEtl()

# Replace 'your_module' and 'YourModule' with the actual module and module name
module_name = 'your_module'
module = __import__(module_name)

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
<schema>
CREATE TABLE department (
  Department_ID number,         -- Unique identifier for the department
  Name text,                     -- Name of the department
  Creation text,                 -- Date of creation or establishment
  Ranking number,                -- Ranking of the department
  Budget_in_Billions number,     -- Budget of the department in billions
  Num_Employees number          -- Number of employees in the department
);

CREATE TABLE head (
  head_ID number,                -- Unique identifier for the head
  name text,                     -- Name of the head
  born_state text,               -- State where the head was born
  age number                     -- Age of the head
);

CREATE TABLE management (
  department_ID number,          -- Foreign key referencing Department_ID in department table
  head_ID number,                -- Foreign key referencing head_ID in head table
  temporary_acting text          -- Indicates if the head is temporarily acting
);
</schema>
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

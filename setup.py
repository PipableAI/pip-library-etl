from setuptools import find_packages, setup

with open("README.md", "r", encoding= 'UTF-8') as f:
    long_description = f.read()
    
setup(
    name='pip_library_parser',
    version='0.1.0',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'transformers',
    ],
)
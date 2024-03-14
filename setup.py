from setuptools import find_packages, setup

from version import __version__

with open("README.md", "r", encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="pip_library_parser",
    version=__version__,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "transformers",
    ],
)

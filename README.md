# WrapperBox

This repository is written like a Python module, to ensure that relative imports
work as expected, please change to the root of this project's directory, and
(preferably in a virtual environment) do 

```bash
pip install -e .
```

# Versioning

To ensure that pickling is as expected, please ensure that the 
installed sklearn distribution is version 1.1.1.

For the walrus operator to work, your python version must be at least 3.8

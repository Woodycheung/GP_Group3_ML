# GDP and Visitor Analysis Project(Group 3)

### Author
```
CHAN Chung Hang  22061759S
CHEUNG Ho Bun  22056983S
POON Wing Fung 22056100S
YEUNG Ka Wai 22049550S
```

## Purpose
This Group Project analyzes the relationship between the number 
of visitors and the Hong Kong Gross Domestic Product (GDP) 
across different countries. 
It aims to determine if there is a significant correlation 
between visitors and economic performance, 
utilizing machine learning models for a thorough investigation.

## Dependencies
This project uses Poetry for dependency management. It requires:

- Python 3.8 or higher
- Poetry

The `pyproject.toml` file in the repository lists all the necessary Python packages.

## Environment Setup

### For macOS:

1. Install [Homebrew](https://brew.sh/) if it's not already installed.
   ```sh
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python using Homebrew.
   ```
   brew install python
   ```
3. Install Poetry via Homebrew.
   ```
   brew install poetry
   ```
4. Install the project dependencies using Poetry.
   ```
   poetry install
   ```
### For Windows:
1. Install [Python](https://www.python.org/downloads/). Ensure you select the option to Add Python to PATH during installation.
2. Install Poetry using PowerShell.
   ```
   (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
   ```
3. Install the project dependencies using Poetry.
   ```
   poetry install
   ```
### Running the Analysis
To run the analysis, use Poetry to handle the environment and dependencies:
For macOS and Windows:
```
poetry run python gdpwithdifferenctcountry.py
```
This will activate the virtual environment and run the analysis script.

### Output
The scripts will generate visualizations and model predictions, 
comparing the number of visitors to GDP figures and 
storing the output in specified directories.

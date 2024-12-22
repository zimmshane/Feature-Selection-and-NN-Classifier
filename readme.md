# Feature Selection with Nearest Neighbor Classifier

A Python implementation of feature selection algorithms using k-Nearest Neighbor classification. This project implements three different search strategies for finding optimal feature subsets: Forward Selection, Backward Elimination, and Simulated Annealing.

## Features

- Three feature selection algorithms:
  1. Forward Selection
  2. Backward Elimination
  3. Simulated Annealing
- k-Nearest Neighbor classification
- Multiple data normalization options
- Leave-one-out cross-validation
- Support for custom datasets
- Built-in test datasets including Titanic dataset

## Requirements

- Python 3.x
- NumPy
- pathlib
- argparse

## Installation

1. Clone this repository
2. Ensure you have the required dependencies installed:

```bash
pip install numpy
```

## Usage

The program can be run from the command line with various arguments:

```bash
python main.py [options]
```

### Command Line Arguments

- `--customdata`, `-d`: Path to a custom dataset file
- `--testdata`: Choose from provided test datasets [`bigdata`, `smalldata`, `titanic`]
- `--search`, `-s`: Select feature search method [`forward`, `backward`, `simulated-annealing`]
- `--debug`: Enable debug logging (default: `False`)
- `--NN`, `-k`: Set k value for k-nearest neighbor (default: `3`)
- `--normalization`, `-norm`: Choose normalization method [`min-max`, `std-normal`, `numpy`, `none`]

### Example Commands

```bash
# Run with default settings (forward selection on titanic dataset)
python main.py

# Run backward elimination on small dataset with k=5
python main.py --search backward --testdata smalldata --NN 5

# Use custom dataset with simulated annealing
python main.py -d path/to/dataset.txt -s simulated-annealing

# Run with different normalization method
python main.py --normalization std-normal
```

## Data Format

Input data is parsed using numpy's `loadtxt` function.
Input data should be formatted as a text file with:
- First column: Binary labels (`0` or `1`)
- Subsequent columns: Feature values
- Space-separated values
- One instance per line

### Example Data Format
Your input dataset should be a `.txt` and should look something like this.
```
1    0.1   0.2   0.3
0    0.4   0.5   0.6
1    0.7   0.8   0.9
```

## Algorithms

### Feature Selection Methods

1. **Forward Selection** [`forward`]: Starts with no features and iteratively adds the most beneficial features
2. **Backward Elimination**[`backward`]: Starts with all features and iteratively removes the least beneficial features
3. **Simulated Annealing**[`simulated-annealing`]: Uses probabilistic approach to search feature space, potentially escaping local optima

### Normalization Options

- `min-max`: Scales features to range [`0`,`1`]
- `std-normal`: Standardizes features to a mean of 0 and standard deviation of 1.
- `numpy`: Uses NumPy's default normalization
- `none`: No normalization applied

## License

MIT License

## Contributors

Equal Contributions to this project came from [Lindsay Adams](https://github.com/lindsayadams2552)

## Acknowledgments

This project was developed as part of CS-170 at UCR.

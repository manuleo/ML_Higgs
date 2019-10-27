# EPFL ML Project 1

Detecting the Higgs Boson. Project about binary classification on a data-set which come directly from CERN.


#### Prerequisites

1. Python 3
2. PIP
3. Numpy


#### Usage instructions

1. Go to the project's directory

2. Copy there the **train.csv** and **test.csv** files

3. Run one of the following commands in CMD/bash:

  ```
  python3 run.py
  ```
  Run the implementation without performing cross validation. The hyper-parameters and degrees used for the Ridge Regression are    hardcoded for simplicity.
  ```
  python3 run.py -cross
  ```
  Run the implementation with cross validation which computes the hyper-parameters and degree for each jet. **Be careful: this function requires more than 1 hour to run**
  
### Folder content

* `run.py` - This .py is used to run the pipeline of the model used to achieve the prediction in AICROWD. It uses Ridge Regression with  optimal lambdas and degrees. 
* `helpers.py` - Helpers functions: this .py includes generic helpers from labs, helpers for the run.py (preprocessing, cross validation...) and specific helpers for the project (e.g. load_csv(..)).
* `implementations.py` - Implementations of the 6 mandatory algorithms.
* `implementations_helpers.py` - Helper functions for the algorithms in implementations.py
* `report.pdf` - The report containg all the steps we followed to reach the final submission: abstract, introduction, model and methods, results and final summary.
* `images`- This folder contains all the images we analyzed to decide which columns drop and which modify with cubic root during the preprocessing phase.
### Authors
Manuel Leone, Gabriele Macchi, Marco Vicentini

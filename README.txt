Implementation of Collaborative Filtering in recommender system

Implement recommender system with technique of collaborative filtering, and add some improvement of tranditional recommender system. Features added are regularization of user bias and item bias, temporal parameters, make use of cosine distance to calculate similarity, and pick top k nearest neighbor to do collaborative filtering.
Usage

Run:

> python3 recommenderSys.py

Notice:

Python Version: 3.5.1

Required modules: Scipy, Numpy, Pandas, Math, Scikitlearn

Dataset

ml-100k, we've included and hardcoded the dataset path into the code, you can test with it directly, download it from grouplens, you can download it from Movielens

File tree

Python files:
├── recommenderSys.py               # Main python file including training and testing.
├── draw.py 						# Used to draw figures of test results
├── draw_neighbor.py 				# Used to draw figures of test results of user-user and movie-movie test sets


Jupyter Notebook files:
├── RecommSys.ipynb               # Ipython notebook file used for tests, not complete

Txt files:
├── recordToFindDamp.txt            # Test results to find a proper damp parameter
├── recordToFindKValue.txt          # Test results to find proper amount of neighbor to take into account
├── testsOnmultipleTrainingTestDataset # Test results on different dataset

Dataset folder:
├── ml-100k/						# Dataset, part of original ml-100k/, only contains u1,u2, if you would like to test on complete datasets, please download datasets from Movielens. We've attached the link in the bottom of report.pdf.


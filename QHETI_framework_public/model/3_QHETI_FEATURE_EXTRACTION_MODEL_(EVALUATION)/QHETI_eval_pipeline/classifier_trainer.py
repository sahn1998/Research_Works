"""
ClassifierTrainer Module

This module defines a class to perform hyperparameter tuning and training
for several conventional ML models using GridSearchCV.
"""

import psutil
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import tree, neighbors, linear_model, gaussian_process
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

class ClassifierTrainer:
    """
    Trains conventional classifiers with GridSearchCV based on the specified model type.
    """
    def __init__(self, model_type):
        """
        Args:
            model_type (str): One of ['NaiveBayes', 'DecisionTree', 'LogisticRegression', 'K-NN', 'SVM'].
        """
        self.model_type = model_type

    def train(self, X_train, Y_train):
        """
        Performs grid search and returns the best fitted estimator.

        Args:
            X_train (np.ndarray): Training features.
            Y_train (np.ndarray): Training labels.

        Returns:
            sklearn.base.BaseEstimator: Trained model with best hyperparameters.
        """
                
        if self.model_type == "NaiveBayes":
            print("Setting up Naive Bayes GridSearchCV")
            parameters = {
                "kernel": [
                    ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"),
                    1.0 * RBF(1.0),
                    DotProduct() + WhiteKernel(),
                ]
            }
            model = gaussian_process.GaussianProcessClassifier(random_state=42)

        elif self.model_type == "DecisionTree":
            print("Setting up Decision Tree GridSearchCV")
            parameters = {
                "max_depth": range(2, 23),
                "criterion": ["entropy", "gini"]
            }
            model = tree.DecisionTreeClassifier()

        elif self.model_type == "LogisticRegression":
            print("Setting up Logistic Regression GridSearchCV")
            parameters = [
                {
                    "penalty": ["l2"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["lbfgs"]
                },
                {
                    "penalty": ["l1"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear"]
                }
            ]
            model = linear_model.LogisticRegression()

        elif self.model_type == "K-NN":
            print("Setting up KNN GridSearchCV")
            parameters = {
                "n_neighbors": [3, 5, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            }
            model = neighbors.KNeighborsClassifier()

        elif self.model_type == "SVM":
            print("Setting up SVM GridSearchCV")
            parameters = [
                {
                    "kernel": ["linear"],
                    "C": [1, 10, 100]
                },
                {
                    "kernel": ["rbf"],
                    "C": [1, 10],
                    "gamma": ["scale", "auto"]
                },
                {
                    "kernel": ["poly"],
                    "C": [1],
                    "degree": [3, 4],
                    "coef0": [0.0, 0.5],
                    "gamma": ["scale"]
                },
                {
                    "kernel": ["sigmoid"],
                    "C": [1],
                    "coef0": [0.0, 0.5],
                    "gamma": ["scale"]
                }
            ]
            model = SVC()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=4, verbose=0)
        print(f"Fitting {self.model_type} GridSearchCV")
        
        grid_search.fit(X_train, Y_train.ravel())
        print(f"Best {self.model_type} parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

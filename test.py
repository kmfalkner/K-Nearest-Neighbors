import unittest
import numpy as np
import os
import inspect
from knn import KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

def generate_data():
    X_train, y_train = make_classification(n_samples=1000, n_features=100, n_informative=90, n_redundant=10, n_clusters_per_class=1, random_state=None)
    X_test, y_test = make_classification(n_samples=200, n_features=100, n_informative=90, n_redundant=10, n_clusters_per_class=1, random_state=None)
    return X_train, y_train, X_test, y_test

class TestKNN(unittest.TestCase):

    def setUp(self):
        # Generate 5 different datasets
        self.datasets = [generate_data() for _ in range(5)]
        self.ks = [3, 5]  # Two different k values
        
    def run_knn_test(self, X_train, y_train, X_test, y_test, k):
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        # Using sklearn's KNeighborsClassifier for comparison
        knn_sklearn = KNeighborsClassifier(n_neighbors=k)
        knn_sklearn.fit(X_train, y_train)
        expected_predictions = knn_sklearn.predict(X_test)
        
        np.testing.assert_array_equal(predictions, expected_predictions)
        print(f"Test with k={k} and dataset passed.")

    def test_knn_datasets(self):
        for i, (X_train, y_train, X_test, y_test) in enumerate(self.datasets):
            for k in self.ks:
                with self.subTest(i=i, k=k):
                    self.run_knn_test(X_train, y_train, X_test, y_test, k)
                    print(f"Dataset {i+1}, k={k} test passed.")


if __name__ == '__main__':
    unittest.main()

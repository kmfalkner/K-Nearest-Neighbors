

# HW1—K-Nearest Neighbors (KNN)

## Background 

You are working at a biotech startup called GenomicGenius. The company focuses on classifying genetic sequences to help in identifying various species. You, along with your co-worker Alex, an expert in bioinformatics, are tasked with building a robust classification system using machine learning techniques.

Your first assignment is to implement the k-Nearest Neighbors (kNN) algorithm, a simple yet powerful method for classification. You need to classify DNA sequences based on their k nearest neighbors in a feature space. The goal is to correctly classify the sequences to their respective species using the provided "DNA" features (100-dimensional).

## Task Description

In this homework, you will implement the k-Nearest Neighbors (kNN) algorithm. The task is to classify DNA sequences that belongs to two categories, where each sequence is represented as a NumPy array. You will implement the kNN algorithm in Python, test it using provided test cases, and ensure it performs well on unseen data.

You will be provided with a dataset of "DNA" sequences, each labeled with its respective species. The dataset will be split into a training set and a test set. Your implementation should be able to:

1. "Train" the KNN classifier using the training set.
2. Predict the labels for the test set.
3. Evaluate the performance of your classifier.

"Training" a nearest neighbor algorithm consists merely of loading the data into a an accessible data structure or database of some sort, such that you can retrieve labelled data items and compute their distance from the current input.  If the amount of data is large, this can be quite complicated, since you don't want to compare the input to EVERY previously classified instance.  In your implementation, however, the data will consist of 100 features (dimensions); and there will be only 1000 items in the "training" set.  Hence comparing new inputs to all previously classified items will not be a problem in terms of computational resources (memory or time).  (See section 18.8.1 of Russell and Nerving, 3rd edition, for further discussion of kNN and this issue.)

### Dataset

The data set which you will use to "train" your system will be automatically generated for you. We have provided the dataset generate scripts for you. You don't need to worry about dataset.

### Requirements

Your implementation should handle the following requirements:

- **Distance Metric**: Use Euclidean distance to measure the similarity between sequences.
- **K Value**: The number of neighbors to consider (k) will be provided as an input.
- **Binary Classification**: Each sequence belongs to one of two species.
- **NumPy Arrays**: Sequences will be represented as NumPy arrays 

## Example

Here is an example of how the KNN classifier should work:

```python
import numpy as np
from knn import KNN

# Example sequences (as NumPy arrays)
train_sequences = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]])
train_labels = np.array([0, 1, 0, 1])

test_sequences = np.array([[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]])
test_labels = np.array([0, 1])

# Create a KNN classifier with k=3
knn = KNN(k=3)

# Train the classifier
knn.fit(train_sequences, train_labels)

# Predict the labels for the test set
predicted_labels = knn.predict(test_sequences)

```

## Deliverables

You are required to submit the following file:

`knn.py`: This file will contain your implementation of the KNN algorithm.



### Implementation Details

- **Fit Method**: Your KNN class should have a `fit` method that takes in the training sequences and their corresponding labels.

- **Predict Method**: Your KNN class should have a `predict` method that takes in the test sequences and returns the predicted labels.

  

**Note**: The hidden test cases in `test_hide.py` will be different from the published test cases but similar in nature. Ensure that your implementation is robust and can handle different scenarios.

Good luck, and happy coding!

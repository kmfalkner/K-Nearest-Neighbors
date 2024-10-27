import numpy as np
import math

class KNN: 
    def __init__(self, k):
        """
        Initializes the KNN classifier with a specified number of neighbors (k).
        
        Parameters:
        k (int): The number of neighbors to consider for classification.
        """
        self.k = k
        self.train_sequences = None
        self.train_labels = None

    def fit(self, train_sequences, train_labels):
        """
        Fits the KNN classifier using the training data.
        
        Parameters:
        train_sequences (np.ndarray): The training sequences represented as a NumPy array of TNF features.
        train_labels (np.ndarray): The labels corresponding to the training sequences.
        """
        self.train_sequences = train_sequences
        self.train_labels = train_labels
        #print(train_sequences.ndim)



    def predict(self, test_sequences):
        """
        Predicts the labels for the test sequences.
        
        Parameters:
        test_sequences (np.ndarray): The test sequences represented as a NumPy array of TNF features.
        
        Returns:
        np.ndarray: The predicted labels for the test sequences.
        """
        temp = len(self.train_sequences[0])
        predicted = []
        for test in test_sequences:
            distances = []
            j = 0
            for set in self.train_sequences:
                i = 0
                sum = 0
                while i < temp:
                    sum = sum + ((set[i] - test[i]) ** 2)
                    i = i + 1
                distances.append([math.sqrt(sum), self.train_labels[j]])
                j = j + 1
                distances.sort()
            testset = distances[0:self.k]
            zerocounter = 0
            onecounter = 0
            for m in testset:
                if m[1] == 0:
                    zerocounter = zerocounter + 1
                else:
                    onecounter = onecounter + 1
            if onecounter > zerocounter:
                predicted.append(1)
            else:
                predicted.append(0)
        return predicted
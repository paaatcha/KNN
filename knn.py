'''
Author: Andre Pacheco
E-mail: pacheco.comp@gmail.com
This code computes the K-nearst neighbors using the Euclidean distance. If you find some bug, please e-mail me

Input:      
      in_train: a [number of samples x number of features] matrix with the training data
      out_train: a [number of samples x labels] matrix with the labels of the training data
      in_test: a [number of test samples x number of features] matrix with the testing data
      k: the number of K, of course

Output:
      labels: the number of neighbors for each label for the in_test array

'''
import numpy as np

# Euclidean distance
def distance (x,y):
      return np.sqrt(np.power(x-y,2).sum(axis=1))

def knn (in_train, out_train, in_test, k):
      in_train = np.asarray (in_train)
      out_train = np.asarray (out_train)
      in_test = np.asarray (in_test)
      
      size_in_train = in_train.shape
      size_out_train = out_train.shape
      size_in_test = in_test.shape


      #The labels array that will be returned
      labels = np.zeros ([size_in_test[0], size_out_train[1]])

      for i in range(size_in_test[0]):
           # Computing the distance from the sample test to the training set
           rpt_test = np.tile (in_test[i,:], (size_in_train[0], 1))
           dists = distance (rpt_test,in_train)

           # Sorting the distances and getting the k nearest neighbors
           index_sort = np.argsort (dists)
           pos_labels = index_sort[:k]
           closeness = out_train [pos_labels]

           # The final label will be the highest value in the row
           labels[i] = closeness.sum(axis=0)


      return labels









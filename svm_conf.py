import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
from sklearn import metrics

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC

#Load the dataset
dataset = datasets.fetch_openml('mnist_784') # can change to 'Fashion-MNIST'

#dataset = load_digits()
#print("Number of Samples: %d" %len(dataset.target))
#print("Output Categories: %s" %dataset.target_names)
features = dataset.data
print("Feature Vectors: %s" %features)
labels = dataset.target
print("Labels: %s" %labels)

trainIdx = np.random.rand(len(labels)) < 0.8
features_train = features[trainIdx]
labels_train = labels[trainIdx]
features_test = features[~trainIdx]
labels_test = labels[~trainIdx]
print("Number of training samples: ",features_train.shape[0])
print("Number of test samples: ",features_test.shape[0])
print("Feature vector dimensionality: ",features_train.shape[1])

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(features_train)
features_train = scaling.transform(features_train)
features_test = scaling.transform(features_test)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(features_train)
features_train = scaling.transform(features_train)
features_test = scaling.transform(features_test)

from sklearn.svm import SVC # "Support Vector Classifier"
svm = SVC(kernel='linear', probability=True) # can change kernel to 'rbf'
svm.fit(features_train, labels_train)
# predict the output labels of test data
labels_pred = svm.predict(features_test)
# print classification metrics 
print(metrics.classification_report(labels_test, labels_pred))
# print confusion matrix
print(metrics.confusion_matrix(labels_test, labels_pred))

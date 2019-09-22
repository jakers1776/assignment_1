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

# import modules
from sklearn.ensemble import GradientBoostingClassifier
# initiate the classifier
gbc_clf = GradientBoostingClassifier()
#gbc_clf = GradientBoostingClassifier(max_depth=5) # can change max_depth parameter
# fit the classifier model with training data
gbc_clf.fit(features_train, labels_train)

# predict the output labels of test data
labels_pred = gbc_clf.predict(features_test)
# print classification metrics 
print(metrics.classification_report(labels_test, labels_pred))
# print confusion matrix
print(metrics.confusion_matrix(labels_test, labels_pred))

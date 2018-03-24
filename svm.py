import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.svm import SVC


# load the histograms and the labels
histograms = np.load(os.path.join('colour_histograms','histograms.dat'))
labels = np.load(os.path.join('colour_histograms','labels.dat'))

# generate a random permutation of our data samples
data_size = len(labels)
perm = np.random.permutation(data_size)

# shuffle the data
histograms = histograms[perm]
labels = labels[perm]

# cast the histograms to floats
histograms = histograms.astype(np.float64)

# split into training and testing data
training_percentage = 0.8

histograms_train = histograms[:int(training_percentage * data_size)]
labels_train = histograms[:int(training_percentage * data_size)]
histograms_test = histograms[int(training_percentage * data_size):]
labels_test = histograms[int(training_percentage * data_size):]



lin_clf = SVC(kernel='linear')
lin_clf.fit(histograms_train, labels_train)
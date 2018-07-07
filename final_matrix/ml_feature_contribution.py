import socket
import sys
import csv
import xlwt
import time
import pandas as pd
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import svm
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFE
# ML parameters
gamma=0.1

# retrieving data...
mydata=pd.read_excel("x==.xls")
mydata1=mydata.iloc[:,:37]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data1=matrix(mydata1).transpose()[0].getA()[0][:, np.newaxis]
for i in range(35):
    col1=matrix(mydata1).transpose()[i+1].getA()[0][:, np.newaxis]
    data1=np.hstack((data1,col1))

X=data1
Xt=matrix(mydata1).transpose()[36].getA()[0][:, np.newaxis]
y=Xt
plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])
"""
# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')

# #############################################################################
"""
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
        color='navy', edgecolor='black')
"""
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='c',
        edgecolor='black')

"""
plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()
print(np.shape(X))
print(np.shape(Xt))
#print(np.shape(Y))
"""
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.3)

y_rbf = svr_rbf.fit(X, Xt).predict(Y)
print(y_rbf)

plt.plot(y_rbf,'ro',color='r',label='rbf')

plt.plot(Xt,'ro',color='g',label='orignal')

plt.grid()
plt.legend()
plt.show()
"""

# Load the digits dataset

X = X
y = Xt

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)
"""
# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
for making rfe
svc = SVC(C=1, kernel="linear")
    rfe = RFE(estimator=svc, n_features_to_select=300, step=0.1)
    rfe.fit(all_training, training_labels)
print ('coefficients',rfe.estimator_.coef_)
"""
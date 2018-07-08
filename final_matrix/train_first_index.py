import pandas as pd
import numpy as np
from numpy import matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xlwt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use("ggplot")
y=[]

mydata=pd.read_excel('firstindex==.xls')
feat=mydata.iloc[:,0:36]
feat.as_matrix()
label=mydata.iloc[:,36]
label.as_matrix()

x = np.array(feat)
y = np.array(label)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
model1 =KNeighborsClassifier(n_neighbors=3)

#clf=svm.SVC(kernel='linear',C=1)
#clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

model1.fit(x_train,y_train.ravel())
filename = 'finalized_model1.sav'
pickle.dump(model1, open(filename, 'wb'))

predictions=model1.predict(x_test)
accuracy=accuracy_score(y_test,predictions)

#print(y_test,predictions)
#print ("Accuracy = ", accuracy)

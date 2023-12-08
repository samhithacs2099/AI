import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

import os
# import the data set
for dirname, _, filenames in os.walk('/kaggle/input/gtsrb-german-traffic-sign'):
    for filename in filenames:
        os.path.join(dirname, filename)
print(dirname)

# Read the data set
meta_data = pd.read_csv('../input/gtsrb-german-traffic-sign/Meta.csv')
# no of rows and column
meta_shape = meta_data.shape
# rows
no_classes = meta_shape[0]

import cv2
train_data=[]
train_labels=[]
side = 20
channels = 3

# Resize each image and append it to train_data
# append the integer class labels to  train_labels
for c in range(no_classes) :
    path = "../input/gtsrb-german-traffic-sign/Train/{0}/".format(c)
    files = os.listdir(path)
    for file in files:
        train_image = cv2.imread(path+file)
        image_resized = cv2.resize(train_image, (side, side), interpolation = cv2.INTER_AREA)
        train_data.append(np.array(image_resized))
        train_labels.append(c)

# convert all the images to pixels
data = np.array(train_data)
data = data.reshape((data.shape[0], 20*20*3))
# convert the data from interger pixels to float point pixel values
data_scaled = data.astype(float)/255
# convert the interger class labels to binary using label encoder
labels = np.array(train_labels)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)
# data and labels are split into training and validation sets
# X-train is training data, Y-train is training labels
#x-val is validation data and y_val is validation labels
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_scaled, labels, test_size=0.75, random_state=42)

#Test and train with KNeighbors algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
knn = accuracy_score(y_val, y_pred)
print(accuracy_score(y_val, y_pred))

# Test and train RandomForest algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred2 = model.predict(X_val)
rfc = accuracy_score(y_val, y_pred2)
print(accuracy_score(y_val, y_pred2))

# Test and train SVM algorithm
from sklearn.svm import SVC 
model_svm = SVC(kernel='linear') 
model_svm.fit(X_train,y_train)
y_pred1 = model.predict(X_val)
svm = accuracy_score(y_val, y_pred1)
print(accuracy_score(y_val, y_pred1))

# Test and train Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model= LogisticRegression()
model.fit(X_train,y_train)
y_pred3 = model.predict(X_val)
lr = accuracy_score(y_val, y_pred3)
print(accuracy_score(y_val, y_pred3))

# plot a bar graph based on obtained accuracy scores
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['Randomforest Classifier', 'KNN', 'SVM', 'Logistic Regression']
accuracy = [rfc,knn,svm,lr]
ax.bar(labels,accuracy)
plt.show()
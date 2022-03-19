import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append("pylibs/")

from classify import dataset, modelling

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

# if len(sys.argv) < 2:
#     print("No dataset provided, usage: python knn.py [path/to/Dataset_ESC-50]")
#     sys.exit(1)

path = "./ESC-50/"

dataset_path = dir_path(path)

"Parameters"
K = 10 # Number of neighbours
sel_class = [12,14,40,41,49] # Select only some classes for the classification
#dataset_path = os.path.join(os.path.dirname(os.getcwd()), dataset_path)


classnames, allpath_mat = dataset.gen_allpath(dataset_path)
allpath_sel = np.array([allpath_mat[idx,:] for idx in sel_class])
print(allpath_sel)
classnames = np.array([classnames[idx] for idx in sel_class])
class_ids = np.arange(len(sel_class))
class_ids = np.repeat(class_ids, 40)
data_path = allpath_sel.reshape(-1)

print('The selected classes are {}'.format(classnames))

# ----------------------------------------------------------------------------------
"""Creation of the dataset"""
myds = dataset.SoundDS(class_ids, data_path, allpath_mat)

"Random split of 70:30 between training and validation"
train_pct = 0.7

H,W = myds[0][0].shape
featveclen = H*W # number of items in a feature vector
num_items = len(myds) # number of sounds in the dataset
num_sounds = len(myds.allpath_mat[0,:]) # number of sounds in each class
num_classes = int(num_items/num_sounds) # number of classes
num_train = round(num_sounds * train_pct) # number of sounds among num_sounds for training

data_aug_factor = 1
class_ids = np.arange(len(sel_class))
class_ids_aug = np.repeat(class_ids, num_sounds*data_aug_factor)

"Compute the matrixed dataset, this takes some seconds, but you can then reload it by commenting this loop and decommenting the np.load below"
X = np.zeros((data_aug_factor*num_classes*num_sounds, featveclen))
for s in range(data_aug_factor):
    for i, featvec in enumerate(myds):
        plt.imshow(
        featvec[0],
        aspect="auto",
        cmap='jet',
        origin='lower',
        )
        plt.show()
        X[s*num_classes*num_sounds+i,:] = featvec[0].reshape(-1)/np.amax(featvec[0])

# np.save("feature_matrix_2D.npy", X)

X = np.load("feature_matrix_2D.npy")

"Labels"
y = class_ids_aug
# ----------------------------------------------------------------------------------

"""Model definition and training"""

rf = modelling.RandomForestModel()

"Shuffle then split the dataset into training and testing subsets"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y) # random_state=1



rf.train(X_train, y_train)
randomForestTheoric = RandomForestClassifier(n_estimators=500,min_samples_split=4,min_samples_leaf=1)

randomForestTheoric.fit(X_train,y_train)

y_pred = randomForestTheoric.predict(X_test)
print("score=", randomForestTheoric.score(X_test,y_test))

import seaborn as sb
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test,y_pred)
ax = sb.heatmap(conf, annot=True)

plt.show()
# save the trained model 
filename = 'RandomForest.pickle'
rf.save(filename)

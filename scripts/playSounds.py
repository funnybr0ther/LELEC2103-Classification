import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import matplotlib.pyplot as plt
import os, sys
import soundfile as sf
import playsound as ps

import numpy as np
import sounddevice as sd

fs = 44100
t = np.linspace(0,1,fs)
data = 0.1*np.sin(2*np.pi*440*t)
sd.play(data, fs,blocking=True)

t = np.linspace(0,5,fs*5)
testdata = 0*np.sin(2*np.pi*440*t)
sys.path.append("../pylibs")

from classify import dataset, modelling

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if len(sys.argv) < 2:
    print("No dataset provided, usage: python knn.py [path/to/Dataset_ESC-50]")
    sys.exit(1)

dataset_path = dir_path(sys.argv[1])

"Parameters"
K = 10 # Number of neighbours
sel_class = [12,14,40,41,49] # Select only some classes for the classification
#dataset_path = os.path.join(os.path.dirname(os.getcwd()), dataset_path)

class_names = ["Crackling fire", "Chirping birds", "Helicopter", "Chainsaw", "Hand saw"]

classnames, allpath_mat = dataset.gen_allpath(dataset_path)
allpath_sel = np.array([allpath_mat[idx,:] for idx in sel_class])

for i in range(allpath_sel.shape[0]):
    print("Now playing " + class_names[i+4] + " class")
    for j in range(allpath_sel.shape[1]):
        print("Playing sound " + allpath_sel[i+4][j])
        # sd.play(testdata, fs, blocking=True)
        ps.playsound(allpath_sel[i+4][j])
        sd.play(data, fs,blocking=True)
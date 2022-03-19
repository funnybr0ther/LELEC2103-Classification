import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = pd.read_csv("measurements.csv")

labels = ['Crackling fire', 'Chirping birds', 'Helicopter', 'Chainsaw', 'Hand saw']
sb.heatmap(confusion_matrix(data.iloc[:,1],data.iloc[:,0]),annot=True,xticklabels=labels,yticklabels=labels)
plt.show()
print(accuracy_score(data.iloc[:,1],data.iloc[:,0]))
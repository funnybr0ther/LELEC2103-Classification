from ast import arg
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

measurements = "measurements/"

data = pd.read_csv("authLogsCopy.txt",sep=" ",header=None)
playsoundsdata = pd.read_csv("playSoundsLogs.txt", sep=",", header=None)
time_melvecs = data[data.columns[3]] + " " + data[data.columns[4]]
time_melvecs = pd.to_datetime(time_melvecs)
time_playsound = pd.to_datetime(playsoundsdata[playsoundsdata.columns[1]])

parser = argparse.ArgumentParser()
parser.add_argument('--index', action="store", type=int,
help='bill')

args = parser.parse_args()

measurementIndex = args.index

melvec = np.loadtxt(measurements + str(measurementIndex) + ".txt")
print("Loaded melvec " + str(measurementIndex))
time_melvec = time_melvecs.iloc[measurementIndex]
print("Taken at time " + str(time_melvec))
possibleIndex = np.searchsorted(time_playsound,time_melvec)
print("Plausible file:" + str(playsoundsdata[0].iloc[possibleIndex]))

plt.imshow(melvec, interpolation='nearest', origin="lower",aspect="auto")
plt.colorbar()
plt.show()
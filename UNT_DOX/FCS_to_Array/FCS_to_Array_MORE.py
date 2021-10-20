##### Prepare files for deep learning
# Derived from Hu lab. Starting out with flow cytometry data from Chen lab 
# comparing untreated cardiomyocytes versus those treated with Doxorubicin.
# Will use to eventually end up with a data object that can 
# be used for fitting the model using the CNN.

##### Import modules
import numpy as np
import scipy as sp
import pandas as pd
import rpy2 as rp
import os 
import pickle
import FlowCytometryTools as FCT
from collections import Counter

##### Load data
fns = list()
expr_list = list()
cytof_files = pd.read_csv("metaData_MORE.csv") # Before organizing data
for i in range(len(cytof_files.name)):
    fns.append(cytof_files.name[i])
for i in range(len(fns)):
    id_i = "sample%i" %i
    datafile_i = "FCS_Files/" + fns[i]
    data_i = FCT.FCMeasurement(id_i, datafile = datafile_i)
    expr_list.append(data_i)

##### Get common markers
markers = []
for i in range(len(expr_list)):
    markers.extend(expr_list[i].channel_names)

markers = Counter(markers)
markers = [k for k, c in markers.items() if c == 88]
print(markers)

##### Change expr_list to only include the desired markers
for i in range(len(expr_list)):
    t1 = expr_list[i]
    expr_list[i] = t1.data.loc[:,markers]
    expr_list[i] = expr_list[i].sample(n = 10000, replace = True) # sample 10000 cells

##### Transform and format into numpy array
def arcsinh(x):
    return(np.arcsinh(x/5))

coln = expr_list[0].columns.drop("Time")
for i in range(len(expr_list)):
    t1 = expr_list[i].drop(columns="Time")
    t1 = t1.apply(arcsinh)
    t1 = t1.values
    shape1 = list(t1.shape)+[1]
    t1 = t1.reshape(shape1)
    expr_list[i] = t1

expr_list = np.stack(expr_list)
print("The dimenstion of the data is: ", expr_list.shape)

##### Finally putting it into allData and saving file
allData = {"cytof_files" : cytof_files, 
            "expr_list" : expr_list,
            "marker_names" : markers}

with open("allData_MORE.obj", "wb") as f:
    pickle.dump(allData, f)
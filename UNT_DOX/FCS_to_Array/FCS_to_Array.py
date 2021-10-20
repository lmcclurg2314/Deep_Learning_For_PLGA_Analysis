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
# from collections import Counter

##### Load data
fns = list()
expr_list = list()
cytof_files_0 = pd.read_csv("metaData_0.csv") # Before organizing data
for i in range(len(cytof_files_0.name)):
    fns.append(cytof_files_0.name[i])
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
markers = [k for k, c in markers.items() if c == 7]
print(markers)

##### Manually change the markers based on RFE from logistic regression
markers = ['FSC-A', 'SSC-A', 'SSC-H', 'SSC-W', 'FITC-B-525/50-A', 
           'PI-B-630/40-A', 'PerPCy55-B-695/40-A', 'PacBlue-V-450/50-A']

for i in range(0,len(expr_list)):
    t1 = expr_list[i]
    expr_list[i] = t1.data.loc[:,markers]

##### Organizing data
# This is necessary to split data into training, validation, and testing
# Need to do this because of the limited data

# split_dataframe takes a dataframe and returns a list of dataframes with n
# number of dataframes that each have an equal number of data
def split_dataframe(df, n):
    df_len = len(df)
    if (df_len%n != 0): 
        raise ValueError("dataframe length not divisible by n")
    result = list()
    for i in range(n):
        step = df_len//n
        df_new = df[i*step:(i+1)*step]
        result.append(df_new)
    return result

def map_split_dataframe(dfs, n):
    result = list()
    for i in range(len(dfs)):
        result.append(split_dataframe(dfs[i], n))
    return result

# In this case, there are 50,000 cells per sample. Want 10,000 cells per sample
# Thus, n = 50,000 / 10,000 = 5
expr_list = map_split_dataframe(expr_list, 5)

# After mapping, expr_list is a list of dataframe lists
# Flattening results in a dataframe list
def flatten(l):
    result = list()
    for i in range(len(l)):
        curr_list = l[i]
        for j in range(len(curr_list)):
            result.append(curr_list[j])
    return result

expr_list = flatten(expr_list)

##### Transform and format into numpy array
def arcsinh(x):
    return(np.arcsinh(x/5))

#coln = expr_list[0].columns.drop("Time")
for i in range(len(expr_list)):
    t1 = expr_list[i]
    # above line used to be "t1 = expr_list[i].drop(columns="Time")""
    t1 = t1.apply(arcsinh)
    t1 = t1.values
    shape1 = list(t1.shape)+[1]
    t1 = t1.reshape(shape1)
    expr_list[i] = t1

expr_list = np.stack(expr_list)
print("The dimenstion of the data is: ", expr_list.shape)

##### Finally putting it into allData and saving file
cytof_files_1 = pd.read_csv("metaData_1.csv") # After organizing data
allData = {"cytof_files" : cytof_files_1, 
            "expr_list" : expr_list,
            "marker_names" : markers}

with open("allData_8_markers.obj", "wb") as f:
    pickle.dump(allData, f)

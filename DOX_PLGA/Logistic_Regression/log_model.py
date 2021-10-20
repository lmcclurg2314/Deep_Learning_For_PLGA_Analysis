# Importing modules
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm 
import sklearn as sk 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE

# Loading data frame using pickle
allData = pickle.load( open( "allData_MORE.obj", "rb" ) )
cytoData = allData["expr_list"]
metaData = allData["cytof_files"]
markerNames = allData["marker_names"]
markerNames.remove("Time")
markerNames.append("PLGA_Ab")
y = metaData.PLGA_Ab.values

# Making dataframe from cytoData
list_all = list() # should be a (88000, 18) list where 18th feature is PLGA_Ab
for i in range(len(cytoData)): # 88 samples
    for j in range(len(cytoData[0])): # 10000 cells
        list_cell = list()
        for k in range (len(cytoData[0][0])): # 17 features
            list_cell.append(cytoData[i][j][k][0])
        list_cell.append(y[i])
        list_all.append(list_cell)
dataframe_all = pd.DataFrame(np.array(list_all), columns = markerNames)

# Curating data as needed for logistic regression function
X = dataframe_all.loc[:, dataframe_all.columns != "PLGA_Ab"]
y = dataframe_all.loc[:, dataframe_all.columns == "PLGA_Ab"]

# Using recursive feature elimination to determine which parameters to use
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

# From the above results, we know to use the following markers (RFE)
cols =  ['SSC-A', 'SSC-H', 'SSC-W', 'FITC-B-525/50-A', 'PI-B-630/40-A',
         'BV605-610/20-A', 'PE-YG-585/15-A', 'PE-Cy7-YG-780/60-A']
X = X[cols]
y = y["PLGA_Ab"] # y remains the same

# Implementing logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

# Model fitting
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, 
y.values.ravel(), test_size = 0.3, random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Printing the accuracy of the model
y_pred = logreg.predict(X_test)
print("Accuracy of model is:{:.2f}".format(logreg.score(X_test, y_test)))

# Printing confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Showing precision, recall, and F-measure
print(classification_report(y_test, y_pred))
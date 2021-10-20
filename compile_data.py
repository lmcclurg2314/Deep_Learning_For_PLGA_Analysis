##### Introduction
# This python file simply takes all the data from the 44 samples and makes a
# a data object called data_all_132.obj (132 because there are 3X44 samples)

##### Importing pickle module
import pickle

##### Loading data
allDataUntDox = pickle.load(open("UNT_DOX/Simple_Stats/allData_MORE_no_arcsinh.obj", "rb"))
cytoData0 = allDataUntDox["expr_list"]
markerNames = allDataUntDox["marker_names"]
allDataDoxPlga = pickle.load(open("DOX_PLGA/Simple_Stats/allData_MORE_no_arcsinh.obj", "rb"))
cytoData1 = allDataDoxPlga["expr_list"]

##### Parsing out data
# Making list for untreated cells
unt_list = list() # should be a (440000, 17)
for i in range(44):
    for j in range(len(cytoData0[0])): # 10000 cells
        list_cell = list()
        for k in range (len(cytoData0[0][0])): # 17 features
            list_cell.append(cytoData0[i][j][k][0])
        unt_list.append(list_cell)

# Making list for DOX cells
dox_list = list() # should be a (440000, 17)
for i in range(44,88):
    for j in range(len(cytoData0[0])): # 10000 cells
        list_cell = list()
        for k in range (len(cytoData0[0][0])): # 17 features
            list_cell.append(cytoData0[i][j][k][0])
        dox_list.append(list_cell)

# Making list for DOX_PLGA cells
plga_list = list() # should be a (440000, 17)
for i in range(44):
    for j in range(len(cytoData1[0])): # 10000 cells
        list_cell = list()
        for k in range (len(cytoData1[0][0])): # 17 features
            list_cell.append(cytoData1[i][j][k][0])
        plga_list.append(list_cell)

##### Putting it all together and saving it as data_all_132.list
data_all_132_list = [None] * 3
data_all_132_list[0] = unt_list
data_all_132_list[1] = dox_list
data_all_132_list[2] = plga_list
data_all_132 = {"data_list" : data_all_132_list, "marker_names" : markerNames}

with open("data_all_132.obj", "wb") as f:
    pickle.dump(data_all_132, f)
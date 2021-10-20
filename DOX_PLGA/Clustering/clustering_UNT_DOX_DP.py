##### Importing modules
import pandas as pd
import numpy as np
import pickle 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ##### Loading data
# allData_0 = pickle.load( open( "Data/allData_MORE.obj", "rb" ) ) # This has the data for DOX and DOX-PLGA
# cytoData_0 = allData_0["expr_list"]
# markerNames_0 = allData_0["marker_names"]
# if "Time" in markerNames_0: markerNames_0.remove("Time")

# allData_1 = pickle.load( open( "Data/allData_MORE_with_UNT.obj", "rb" ) ) # This has the data for UNT
# cytoData_1 = allData_1["expr_list"]
# markerNames_1 = allData_1["marker_names"]
# if "Time" in markerNames_1: markerNames_1.remove("Time")

# ##### Rearranging data to make a dataframe
# # dimensions of cytoData are (88, 10000, 17, 1), we want (1320000, 17)
# # note that we're clustering all samples (44 UNT, 44 DOX, and 44 DOX-PLGA)
# list_all = list() 
# for i in range(88): 
#     for j in range(len(cytoData_0[0])):
#         list_cell = list()
#         for k in range (len(cytoData_0[0][0])):
#             list_cell.append(cytoData_0[i][j][k][0])
#         list_all.append(list_cell)
# for i in range(44): 
#     for j in range(len(cytoData_1[0])):
#         list_cell = list()
#         for k in range (len(cytoData_1[0][0])):
#             list_cell.append(cytoData_1[i][j][k][0])
#         list_all.append(list_cell)
# df = pd.DataFrame(data = np.array(list_all), columns = markerNames_0)

# with open("Data/df_ALL_with_UNT.arr", "wb") as f:
#     pickle.dump(df, f)

##### Looking at df info
df = pickle.load( open( "Data/df_ALL_with_UNT.arr", "rb" ) )

##### Rearranging columns to match order in poster (9/22/21)
cols_1 = ["FSC-A", "SSC-A", "SSC-H", "SSC-W", "FITC-B-525/50-A", "BUV563-A",
          "PI-B-630/40-A", "BV605-610/20-A", "BV711-V-760/50-A",
          "APC-Cy7-R-780/60-A", "PE-Cy7-YG-780/60-A", "PerPCy55-B-695/40-A",
          "PacBlue-V-450/50-A", "PE CY5-YG-680/42-A", "BUV395-A", 
          "APC-R-670/14-A", "PE-YG-585/15-A"]
df = df[cols_1]

# ##### Finding number of clusters
X=df
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
# # inertia = []
# # for i in range(1,11):
# #     kmeans = KMeans(
# #         n_clusters=i, init="k-means++",
# #         n_init=10,
# #         tol=1e-04, random_state=42
# #     )
# #     kmeans.fit(X)
# #     inertia.append(kmeans.inertia_)
# # fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
# # fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
# #                   yaxis={'title':'Inertia'})
# # fig.write_image("Results/elbow_ALL_with_UNT.png")

##### Creating clusters
#From the code above, we found that number of clusters is 2
kmeans = KMeans(
        n_clusters=3, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
kmeans.fit(X)
clusters=pd.DataFrame(X,columns=df.columns)
with open("Data/clusters_ALL_with_UNT_1.df", "wb") as f:
    pickle.dump(clusters, f)
clusters['label']=kmeans.labels_
polar=clusters.groupby("label").mean().reset_index()
polar=pd.melt(polar,id_vars=["label"])
fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
fig4.write_image("Results/clusters_ALL_with_UNT_1.png")

##### Showing how many cells are in each cluster
pie=clusters.groupby('label').size().reset_index()
pie.columns=['label','value']
fig_pie = px.pie(pie,values='value',names='label',color=['blue','red','green'])
fig_pie.write_image("Results/pie_chart_ALL_with_UNT_1.png")

##### Importing modules
import pandas as pd
import numpy as np
import pickle 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

##### Loading data
# allData = pickle.load( open( "Data/allData_MORE.obj", "rb" ) )
# cytoData = allData["expr_list"]
# markerNames = allData["marker_names"]
# if "Time" in markerNames: markerNames.remove("Time")

##### Rearranging data to make a dataframe
# dimensions of cytoData are (88, 10000, 17, 1), we want (880000, 17)
# note that we're clustering all samples (44 UNT and 44 DOX)
# list_all = list() 
# for i in range(88): 
#     for j in range(len(cytoData[0])):
#         list_cell = list()
#         for k in range (len(cytoData[0][0])):
#             list_cell.append(cytoData[i][j][k][0])
#         list_all.append(list_cell)
# df = pd.DataFrame(data = np.array(list_all), columns = markerNames)

# with open("Data/df_ALL.arr", "wb") as f:
#     pickle.dump(df, f)

##### Looking at df info
# df = pickle.load( open( "Data/df_ALL.arr", "rb" ) )
# print(df.info())

##### Finding number of clusters
# X=df
# scaler = MinMaxScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# inertia = []
# for i in range(1,11):
#     kmeans = KMeans(
#         n_clusters=i, init="k-means++",
#         n_init=10,
#         tol=1e-04, random_state=42
#     )
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)
# fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
# fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
#                   yaxis={'title':'Inertia'})
# fig.write_image("Results/elbow_ALL.png")

##### Creating clusters
# From the code above, we found that number of clusters is 2
# kmeans = KMeans(
#         n_clusters=3, init="k-means++",
#         n_init=10,
#         tol=1e-04, random_state=42
#     )
# kmeans.fit(X)
# clusters=pd.DataFrame(X,columns=df.columns)
# with open("Data/clusters_ALL.df", "wb") as f:
#     pickle.dump(clusters, f)
# clusters['label']=kmeans.labels_
# polar=clusters.groupby("label").mean().reset_index()
# polar=pd.melt(polar,id_vars=["label"])
# fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
# fig4.write_image("Results/clusters_ALL.png")

##### Showing how many cells are in each cluster
# pie=clusters.groupby('label').size().reset_index()
# pie.columns=['label','value']
# fig_pie = px.pie(pie,values='value',names='label',color=['blue','red','green'])
# fig_pie.write_image("Results/pie_chart_ALL.png")

##### Analyzing clusters
# df = pickle.load( open( "Data/df_ALL.arr", "rb" ) )
# df["ID"] = [i for i in range(880000)]
# with open("Data/df_ALL_IDs.arr", "wb") as f:
#     pickle.dump(df, f)

clusters = pickle.load( open( "Data/clusters_ALL.df", "rb" ) )
df_IDs = pickle.load(open("Data/df_ALL_IDs.arr", "rb"))

clusters_list = clusters.values.tolist()
df_IDs_list = df_IDs.values.tolist()

cluster_ID = list()
for clustered_cell in clusters_list:
    for cell_ID in df_IDs_list:
        cell = cell_ID[:-1]
        if (clustered_cell == cell):
            cluster_ID.append(cell_ID[-1])
            break
clusters["ID"] = cluster_ID 
print(clusters)
with open("Data/clusters_ALL_IDs.df", "wb") as f:
    pickle.dump(clusters, f)


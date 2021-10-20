#
### Introduction #####
# Code derived from Hu Lab
# Used for comparing untreated samples and DOX samples

##### Step 1: import functions #####

from keras import backend as K
import pickle
import pandas as pd
import numpy as np
from numpy.random import seed; seed(111)
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf; tf.random.set_random_seed(111)
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D, AveragePooling2D, Input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from scipy.stats import ttest_ind
from IPython.display import Image
import pydotplus

# ##### Step 2: load data #####
        
# load data
allData = pickle.load( open( "Data/allData_MORE.obj", "rb" ) )
metaData = allData["cytof_files"]
cytoData = allData["expr_list"]
markerNames = allData["marker_names"]
if ("Time" in markerNames): markerNames.remove("Time")

# changing metadata and markerNames type
# when setting up with FCS_to_Array, they were numpy arrays. change to pandas dataframe and series
markerNames = pd.Series(markerNames)

# inspect the data
print("\nmetaData: ")
print(metaData,"\n")

print("Dimensions of cytoData: ",cytoData.shape,"\n")
print("Names of the 17 markers: \n",markerNames.values)

#### Step 3: split train, validation and test######
# 60 samples of training, 18 samples of validation, and 10 samples of testing
# Each sample has 10,000 cells

y = metaData.DOX_Ab.values
x = cytoData

train_id = metaData.study_accession=="study_train"
valid_id = metaData.study_accession=="study_valid"
test_id = metaData.study_accession =="study_test"

x_train = x[train_id]; y_train = y[train_id]
x_valid = x[valid_id]; y_valid = y[valid_id]
x_test = x[test_id]; y_test = y[test_id]

##### Step 4: define model #####
# model by Hu lab (input, first convolution, second convolution, pooling, dense, and output layers)

# input
model_input = Input(shape=x_train[0].shape)

# first convolution layer
model_output = Conv2D(3, kernel_size=(1, x_train.shape[2]),
                 activation=None)(model_input)
model_output = BatchNormalization()(model_output)
model_output = Activation("relu")(model_output)

# sceond convolution layer
model_output = Conv2D(3, (1, 1), activation=None)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation("relu")(model_output)

# pooling layer
model_output = AveragePooling2D(pool_size=(x_train.shape[1], 1))(model_output)
model_output = Flatten()(model_output)

# Dense layer
model_output = Dense(3, activation=None)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation("relu")(model_output)

# output layer
model_output = Dense(1, activation=None)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation("sigmoid")(model_output)


##### Step 5: Fit model #####
# using Adam algorithm

# specify input and output
model = Model(inputs=[model_input],
              outputs=model_output)

# define loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# save the best performing model
checkpointer = ModelCheckpoint(filepath='Result/saved_weights_MORE_2.hdf5', 
                               monitor='val_loss', verbose=0, save_best_only=True)

# model training
model.fit([x_train], y_train,
          batch_size=32,
          epochs=500, 
          verbose=1,
          callbacks=[checkpointer],
          validation_data=([x_valid], y_valid))

# ##### Step 6: plot train and validation loss #####
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig("Result/loss_graph_MORE_2.png")
plt.clf()

##### Step 7: test the final model #####

# load final model
final_model = load_model("Result/saved_weights_MORE_2.hdf5")

# generate ROC and AUC
y_scores = final_model.predict([x_test])
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC = {0:.2f}'.format(roc_auc))
plt.savefig("Result/AUCROC_MORE_2.png")

##### Step 8: Interpret the deep learning model. #####

# warning: may take a long time (around 30 mins) to run

# Calculate the impact of each cell on the model output
dY = np.zeros([x_test.shape[0],x_test.shape[1]])
s1 = np.random.randint(0,(x_test.shape[1]-1),int(x_test.shape[1]*0.05))
final_model = load_model('Result/saved_weights_MORE_2.hdf5')

for i in range(x_test.shape[0]):
    pred_i = final_model.predict([x_test[[i],:,:,:]])
    for j in range(x_test.shape[1]):
        t1 = x_test[[i],:,:,:].copy()
        t1[:,s1,:,:] = t1[:,j,:,:]
        pred_j = final_model.predict([t1])
        dY[i,j] = pred_j-pred_i

# reformat dY
x_test2 = x_test.reshape((x_test.shape[0]*x_test.shape[1],17))
dY = dY.reshape([x_test.shape[0]*x_test.shape[1]])

with open("dY_MORE_2.arr", "wb") as f:
    pickle.dump(dY, f)

# Build decision tree to identify cell subset with high dY
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(x_test2, dY)

# Plot the decision tree
dot_data = StringIO()
export_graphviz(regr_1, out_file=dot_data, 
                feature_names= markerNames,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("Result/decisionTree_MORE_2.png")

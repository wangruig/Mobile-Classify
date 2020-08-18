#coding=utf-8
import numpy as np
import pandas as pd
import time
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
start_time=time.time()
path=r'C:\Users\wangrg\Desktop\kaggle_ML\Mobile price'
train_data=pd.read_csv(path+r'\\'+'train.csv')
test_data=pd.read_csv(path+r'\\'+'test.csv')
print(train_data.info())

# print(train_data.corr(method='pearson',min_periods=1)['price_range'].sort_values(ascending=False))
# preprocessParams=preprocessing.StandardScaler().fit(train_data)
# train_data_normalized=preprocessParams.transform(train_data)
# test_data_normalized=preprocessParams.transform(test_data)
# # sns.heatmap(train_data.corr(),annot=True)
# # import panda

# features=['ram','battery_power','px_width','px_height','int_memory','sc_w','pc']

# KNN=KNeighborsClassifier(n_neighbors=3,algorithm='auto')
# kf=KFold(n_splits=20,random_state=1).split(train_data_normalized)
# predictions=[]
# test_predictions3=[]
# for train,test in kf:
#     train_features=train_data_normalized[features].iloc[train,:]
#     train_target=train_data_normalized['price_range'].iloc[train]
#     KNN.fit(train_features,train_target)
#     test_predictions=KNN.predict(train_data_normalized[features].iloc[test,:])
#     predictions.append(test_predictions)

# predictions=np.concatenate(predictions,axis=0)
# print(predictions)
# accuracy = len(predictions[predictions == train_data_normalized['price_range']])/len(predictions)
# print(accuracy)

# test_predictions2=KNN.predict(test_data_normalized[features].iloc[:,:])
# test_predictions3.append(test_predictions2)
# # print(test_predictions3)
# # print(test_data[features].iloc[:,:])
# # print(test_predictions)
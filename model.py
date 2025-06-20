# prepare a cluster of customers to predict the purchase power based on thier income and spending score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

# loading the dataset into dataframe
df = pd.read_csv("Mall_Customers1.csv")
data_head = df.head()
#print(data_head)
#print(df.info())
check_null_values = df.isnull().sum(axis=0)
#print(check_null_values)

x=df[["Annual Income (k$)","Spending Score (1-100)"]]

wcss_list =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=1)#(random state)I don't want my rows to be suffled
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_) # which is trying to find out the wcss 

#visualize the plot
# plt.plot(range(1,11),wcss_list)
# plt.title("elbow method graph")
# plt.xlabel("wcss list")
# plt.ylabel("number of cluster")
# plt.show()

model = KMeans(n_clusters=6,init="k-means++",random_state =1)
y_predict = model.fit_predict(x)

print(y_predict)
#converting a dataframe x into a numpy array
x_array = x.values

plt.scatter(x_array[y_predict == 0, 0], x_array[y_predict == 0, 1], s=100, color="Green", label="Cluster 0")
plt.scatter(x_array[y_predict == 1, 0], x_array[y_predict == 1, 1], s=100, color="Red", label="Cluster 1")
plt.scatter(x_array[y_predict == 2, 0], x_array[y_predict == 2, 1], s=100, color="Yellow", label="Cluster 2")
plt.scatter(x_array[y_predict == 3, 0], x_array[y_predict == 3, 1], s=100, color="Blue", label="Cluster 3")
plt.scatter(x_array[y_predict == 4, 0], x_array[y_predict == 4, 1], s=100, color="Pink", label="Cluster 4")
plt.scatter(x_array[y_predict == 5, 0], x_array[y_predict == 5, 1], s=100, color="purple", label="Cluster 5")


plt.title("customer segmentation graph")
plt.xlabel("spending score")
plt.ylabel("annual income")
plt.show()

joblib.dump(model,"model.pkl")
print("model has been saved")
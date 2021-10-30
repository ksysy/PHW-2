
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans as CLARANS
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import copy

import matplotlib.pyplot as plt
import numpy as np


# Change CLARANS result to ScikitLearn result
def clarans_label_converter(labels):
  total_len = 0
  for k in range(0, len(labels)):
    total_len += len(labels[k])

  outList = np.empty((total_len), dtype=int)
  cluster_number = 0
  for k in range(0, len(labels)):
    for l in range(0, len(labels[k])):
      outList[labels[k][l]] = cluster_number
    cluster_number += 1
  return outList



# PHW#2 function
# Problem:
#  - This function will try combinations of the various models automatically.
#  - this fucntion will find best hyperpatameter k value for k-means, EM, DBSCAN, MeanShift and CLARANS which using the best silhouette score.
#  - This function let us know what scaler, model, and hyperparameter has the best silhouette score.
#  - This function was documented by pydoc.
def findBestClusterOptions(
    X:DataFrame,
    scalers=[None, StandardScaler()],
    models=[
        KMeans(n_clusters = 2), # n_clusters = k
        GaussianMixture(), # n_components = k
        DBSCAN(eps=0.5, min_samples=5)
    ],
    cluster_k = [3],
):
    """
    Parameters
    ----------
    - `X`: pandas.DataFrame
      - training dataset.
    - `scalers`: array
      - Scaler functions to scale data. This can be modified by user.
      - `None, StandardScaler()` as default
      - This parameter is compatible with `StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler`.
    - `models`: array
      - Model functions to clustering data. This can be modified by user.
      - KMeans, GaussianMixture, DBSCAN(eps=0.5, min_samples=5) as default with hyperparameters.
      - This parameter is compatible with `KMeans, GaussianMixture, DBSCAN, CLARANS, MeanShift`.
    - `cluster_k`: array
      - Number of cluster. Default value is [3].
      - This can be modified by user.

    Returns
    ----------
    - `best_params_`: dictionary
      - `best_scaler_`: Scaler what has best silhouette score.
      - `best_model_`: Model what has best silhouette score.
      - `best_k_`: Best number of clusters
    - `best_score_`: double
      - Represent the silhouette score of the `best_params`.

    Examples
    ----------
    result = findBestOptions(
      df, 
      models=[
        CLARANS(data=df.to_numpy(), number_clusters=1, numlocal=2, maxneighbor=3),
        GaussianMixture(),
        KMeans(),
        DBSCAN(eps=0.5, min_samples=5),
        MeanShift(bandwidth=bandwidth)
      ],
      scalers=[None,], #StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler() 
      cluster_k = range(2,11)
    )

    # Extract results
    labels = result['labels_']
    best_score = result['best_score_']
    result = result['best_params_']
    best_scaler = result['best_scaler_']
    best_model = result['best_model_']
    best_k = result['best_k_']

    # Print the result of best option
    print("\nBest Scaler: ", end="")
    print(best_scaler)
    print("Best Model: ", end="")
    print(best_model)
    print("Score: ", end="")
    print(best_score)
    print("labels: ", end="")
    print(labels)
    print("k: ", end="")
    print(best_k)
    """

    # Initialize variables
    maxScore = -1.0
    best_scaler = None
    best_model = None
    labels_ = None
    best_k_ = None

    curr_case = 1
    total_case = len(scalers) * len(models) * len(cluster_k)

    # Find best scaler
    for n in range(0, len(scalers)):
        if(scalers[n] != None): 
          X = scalers[n].fit_transform(X)
        
        # Find best model
        for m in range(0, len(models)):

            # Scan once for DBSCAN
            isScaned = False
            
            # Find best k value of CV
            for i in range(0, len(cluster_k)):
                print("Progressing: (",end="")
                print(curr_case,end="/")
                print(total_case,end=")\n")
                curr_case += 1

                # model fitting
                models[m].n_clusters = cluster_k[i]       # for k-Means
                models[m].n_components = cluster_k[i]     # for Gaussian Mixture
                
                
                labels = None
                # calculate silhouette score
                if type(models[m]) == type(CLARANS(X,1,0,0)) :                  
                  models[m] = copy.deepcopy(CLARANS(
                    data=df.to_numpy(), 
                    number_clusters=cluster_k[i],   # CLARANS cluster number setting
                    numlocal=models[m].__dict__['_clarans__numlocal'], 
                    maxneighbor=models[m].__dict__['_clarans__maxneighbor']
                  ))
                  models[m].process()
                  clarans_label = models[m].get_clusters()
                  labels = clarans_label_converter(labels=clarans_label)
                  
                  score_result = silhouette_score(X, labels)
                  
                elif type(models[m]) == type(DBSCAN()) or  type(models[m]) == type(MeanShift()) :
                  if isScaned == True:
                    continue
                  
                  isScaned = True
                  labels = models[m].fit_predict(X)
                  
                  # when cluster nuber is just 1, skip scoring
                  gen_cluster_k = len(pd.DataFrame(labels).drop_duplicates().to_numpy().flatten())
                  if gen_cluster_k <= 1:
                    continue                  
                  score_result = silhouette_score(X, labels)

                else:
                  labels = models[m].fit_predict(X)
                  score_result = silhouette_score(X, labels)



                # if mean value of scores are bigger than max variable,
                # update new options(model, scaler, k) to best options
                if maxScore < score_result:
                    maxScore = score_result
                    best_scaler = copy.deepcopy(scalers[n])
                    best_model = copy.deepcopy(models[m])
                    best_k_ = cluster_k[i]
                    # Calculated by DBSCAN
                    if type(best_model) == type(DBSCAN()) or type(best_model) == type(MeanShift()) : best_k_ = gen_cluster_k 
                    labels_ = copy.deepcopy(labels)

    
    # Return value with dictionary type
    return {
        'best_params_': {
            'best_scaler_': best_scaler,
            'best_model_' : best_model,
            'best_k_': best_k_,
        },
        'best_score_': maxScore,
        'labels_': labels_
    }




## Visualization by latitude and longitude
def lat_lang_plot(df1, df2):

  # Cluster 0
  lats0 = df1.iloc[:,0].to_numpy()
  lons0 = df1.iloc[:,1].to_numpy()
  median_income0 =  df1.iloc[:,7].to_numpy()
  median_house_value0 =  df1.iloc[:,8].to_numpy()

  # Cluster 1
  lats1 = df2.iloc[:,0].to_numpy()
  lons1 = df2.iloc[:,1].to_numpy()
  median_income1 =  df2.iloc[:,7].to_numpy()
  median_house_value1 =  df2.iloc[:,8].to_numpy()


  # median_income
  median_income0_max = df1.iloc[:,7].to_numpy().max()
  median_income1_max = df2.iloc[:,7].to_numpy().max()

  # median_house_value
  median_house_value0_max = df1.iloc[:,8].to_numpy().max()
  median_house_value1_max = df2.iloc[:,8].to_numpy().max()


  plt.figure(figsize=(13, 10))

  # Draw line for purity check
  plt.plot([-124, -114], [35.773, 35.773], "r")

  # Plot scatter with different size and color refer by median_income(size) and median_house_value(color transparency)
  for i in range(len(lats0)):
    dotsize = median_income0[i] / median_income0_max
    colorFade = median_house_value0[i] / median_house_value0_max
    plt.plot(lats0[i], lons0[i], "o", markersize=12 * dotsize, markerfacecolor=[1,0,0,colorFade], markeredgewidth=0.2, markeredgecolor=[1,0.5,0.5],)

  for i in range(len(lats1)):
    dotsize = median_income1[i] / median_income1_max
    colorFade = median_house_value1[i] / median_house_value1_max
    plt.plot(lats1[i], lons1[i], "o", markersize=12 * dotsize, markerfacecolor=[0,0,1,colorFade], markeredgewidth=0.2, markeredgecolor=[0.5,0.5,1],)

  # Plot data
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('Scatter plot in location (median_income, median_house_value)')

  plt.show()



# Data Visualization by ocean_proximity with using latitude, longitude
def lat_lang_op_plot(X, x, y, hue):
  sns.set_style('whitegrid')
  sns.relplot(x=x, y=y, hue=hue, data=X, kind='scatter')
  plt.show()



# Purity check for latitude and longitude
def purity_check(X, y_pred):

# Make arbitrary target dataset to calculate score.
# Seperating line to Northern and Souther california -> 35.773
  y = np.array([], dtype=int)
  for i in range(0, len(X)):
    if X.iloc[i, 1] > 35.773:
      y = np.append(y, [1])
    else:
      y = np.append(y, [0])
  return purity_socre(y, y_pred)

# Scoring function through purity check formula
def purity_socre(y_true, y_pred):
  # compute contingency matrix (also called confusion matrix)
  contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
  return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)



def median_income_ocean_proximity_plot(clusters, cluster_info):
  for i in range(0, len(cluster_info)):
    plt.scatter(clusters[i]['median_income'], clusters[i]['ocean_proximity'], marker='o', s=5)


  plt.title("median_income <=> ocean_proximity")
  legend = clusters[0].columns[10:15]
  plt.legend(legend)
  plt.xlabel('median_income')
  plt.ylabel('ocean_proximity')

  plt.show()



def plot_elbow(X):
# Elbow method for KMeans
  scores = []
  for i in range(2, 11):
    model = KMeans(n_clusters=i)
    model.fit(X)
    scores.append(model.inertia_)

  # scores plot
  plt.plot(range(2,11), scores)
  plt.show()



# Systematic Sampling to row reduction
def systematic_sampling(X, hop=5):
  """
  # Systematic Sampling
   - Reduction rows through systematic sampling
  
  # Examples
   - Total rows: 20430
   - If hop=2 => 50% size compare to origin / 10215 rows
   - If hop=3 => 33% size compare to origin / 6810 rows
   - If hop=5 => 20% size compare to origin / 4086 rows
   - If hop=10 => 10% size compare to origin / 2043 rows
  """
  dft = pd.DataFrame(columns=X.columns)
  countSystematic = 0

 
  for i in range(0, len(X), hop):
    dft = dft.append(X.iloc[i,:])
    countSystematic += 1
  return dft





### Source code
# ===================================================================================

# Import dataset
df = pd.read_csv("./housing.csv")



## Preprocessing =======================
# Drop useless feature

# 전체 데이터 중, total_bedrooms -> 207/20640 (1.003%)
# describe() 로 확인 시, 전체 데이터에 큰 영향을 주지 않는다고 판단
# 의미 없는 데이터 -> 삭제

df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)



## Handle categorical data  =======================
# Labeling for categorical data (ocean_proximity) 
# Select one of LabelEncoder and OneHotEncoder

# OneHotEncoder
# print("Encoder: OneHotEncoder")
# onehot_encoder = OneHotEncoder()
# result = onehot_encoder.fit_transform(df['ocean_proximity'].to_numpy().reshape(-1,1))
# result = result.toarray()
# labelDf = DataFrame(result)
# for i in range(len(labelDf.columns)):
#   df[onehot_encoder.categories_[0][i]] = labelDf.iloc[:,i]
# df.drop(["ocean_proximity"], axis=1, inplace=True)

# LabelEncoder
print("Encoder: LabelEncoder")
label_encoder = LabelEncoder()
result = label_encoder.fit_transform(df['ocean_proximity'])
df['ocean_proximity'] = result



## Feature combination  =======================

# Save origin dataframe to join clustered data after clustering
dft = copy.deepcopy(df)

# Select features to drop
drop_target = [
  # "longitude", 
  # "latitude", 
  "housing_median_age", 
  "total_rooms",
  "total_bedrooms", 
  "population",
  "households", 
  # "median_income",
  "median_house_value",

  # For Label Encoder ...
  # "ocean_proximity",

  # For Onehot Encoder...
  # "<1H OCEAN",
  # "INLAND",
  # "ISLAND",
  # "NEAR BAY",
  # 'NEAR OCEAN',
]

df.drop(drop_target, axis=1, inplace=True)

print("Columns: ", end="")
print(df.columns)



## Systematic sampling  =======================
# df = systematic_sampling(df)



## Find Best model and options  =======================
# Run findBestOptions()

# Estimate bandwidth for MeanShift
bandwidth = estimate_bandwidth(df, quantile=0.1, n_samples=len(df))

result = findBestClusterOptions(
  df, 
  models=[
    # CLARANS(data=df.to_numpy(), number_clusters=1, numlocal=2, maxneighbor=3), # Force apply number_clusters = cluster_k(below) and ignore number_clusters=1
    # GaussianMixture(), # n_components = k
    KMeans(), # n_clusters = k
    # DBSCAN(eps=0.5, min_samples=5),
    # MeanShift(bandwidth=bandwidth)
  ],
  scalers=[None, StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler(),],
  cluster_k = range(3,5),
)

# Extract results
labels = result['labels_']
best_score = result['best_score_']
result = result['best_params_']
best_scaler = result['best_scaler_']
best_model = result['best_model_']
best_k = result['best_k_']

# Print the result of best option
print("\nBest Scaler: ", end="")
print(best_scaler)
print("Best Model: ", end="")
print(best_model)
print("Score: ", end="")
print(best_score)
print("labels: ", end="")
print(labels)
print("k: ", end="")
print(best_k)

# Purity check
# This can only check purity of cluster which clustered by latitude and longitude (k=2)
print("Pruity score: ", end="")
print(purity_check(dft, labels))

print("")



## Analyze cluster =======================
# Extrace cluster numbers 
cluster_info = pd.DataFrame(labels).drop_duplicates().to_numpy().flatten()

# Make dataframe for each cluster
clusters_df = []
for i in range(0, len(cluster_info)):
  clusters_df.append(pd.DataFrame(columns=dft.columns))

for i in range(0, len(labels)):
  clusters_df[labels[i]] = clusters_df[labels[i]].append(dft.iloc[i, :])

# Print describe() to analyze clusters
print("Cluster Info:", cluster_info)
for i in range(0, len(clusters_df)):
  print("Cluster", cluster_info[i])
  print(clusters_df[i].describe())
  clusters_df[i].to_csv("cluster %d.csv" %i)
  clusters_df[i].describe().to_csv("desc cluster %d.csv" %i)
  print("\n")



## Visualization =======================
plot_elbow(df)



# Data Visualization with
#  - latitude
#  - longitude
#  - Ocean Proximity
# lat_lang_op_plot(df, x='longitude', y='latitude', hue='ocean_proximity')


# Clustering with
#  - latitude
#  - longitude
# lat_lang_plot(clusters_df[0], clusters_df[1])


# Clustering with
#  - median_income
#  - ocean_proximity
median_income_ocean_proximity_plot(clusters_df, cluster_info)
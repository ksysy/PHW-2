# Programming Homework #2
 - Auto ML for clustering

## findBestClusterOptions
- Goal: This function will try combinations of the various models automatically.
   - this fucntion will find best hyperpatameter k value for k-means, EM, DBSCAN, MeanShift and CLARANS which using the best silhouette score.
   - This function let us know what scaler, model, and hyperparameter has the best silhouette score.
   - This function was documented by pydoc.

### Parameters
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


### Returns
- `best_params_`: dictionary
    - `best_scaler_`: Scaler what has best silhouette score.
    - `best_model_`: Model what has best silhouette score.
    - `best_k_`: Best number of clusters
- `best_score_`: double
    - Represent the silhouette score of the `best_params`.


### Example

``` python
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
```

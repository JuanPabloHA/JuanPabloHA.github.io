---
title: "End-to-end data science project using Scikit-learn"
date: 2021-02-16
tags: [python, scikit-learn, data science]
header:
  image: "/images/silas-kohler-msaNukYYPpM-unsplash.jpg"
  excerpt: "python, scikit-learn, data science"
  mathjax: "true"
---

## Seoul Bike-sharing End-to-end Machine Learning Project

In this notebook, I will work through an example of a hypothetical data science project. We will be following the steps that are part of the CRISP-DM methodology:

1. Problem description
2. Data acquisition
3. Exploratory data analysis
4. Data preparation
5. Model selection and training
6. Fine-tuning of the model
7. Conclusion

We will skip the Launch and Monitoring steps. 

## 1. Problem description

The Seoul bike sharing demand data set is hosted in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The data set contains the count of the number of bikes rented at each hour in the Seoul bike-sharing system and information regarding weather conditions. 

The final product will consist of a model that predicts the number of bicycles rented in any given day based on the hour and other weather-related variables such as rainfall and humidity. The system's predictions are used to guarantee that available bikes will meet the demand for the service.

## 2. Data acquisition

As mentioned before, the dataset used for this example is hosted by UCI in their Machine Learning Repository. Since we are using a new version of pandas, we can pass the URL into to "read_csv" function, and it will download the dataset for us. The "encoding" argument in the read_csv function is necessary since the data set contains **non-ASCII** characters.


```python
%config Completer.use_jedi = False # Was having issues with aoutocomplete

import pandas as pd
import numpy as np

BIKE_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"

bike_df = pd.read_csv(BIKE_DATA_URL, encoding= 'unicode_escape')
```

### 2.1. A quick look at the data

In the table below, we can observe how each row represents a different day. The fourteen various attributes include the date, number of bikes rented on a given day and wheater-related information.


```python
bike_df[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Rented Bike Count</th>
      <th>Hour</th>
      <th>Temperature(°C)</th>
      <th>Humidity(%)</th>
      <th>Wind speed (m/s)</th>
      <th>Visibility (10m)</th>
      <th>Dew point temperature(°C)</th>
      <th>Solar Radiation (MJ/m2)</th>
      <th>Rainfall(mm)</th>
      <th>Snowfall (cm)</th>
      <th>Seasons</th>
      <th>Holiday</th>
      <th>Functioning Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01/12/2017</td>
      <td>254</td>
      <td>0</td>
      <td>-5.2</td>
      <td>37</td>
      <td>2.2</td>
      <td>2000</td>
      <td>-17.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>No Holiday</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01/12/2017</td>
      <td>204</td>
      <td>1</td>
      <td>-5.5</td>
      <td>38</td>
      <td>0.8</td>
      <td>2000</td>
      <td>-17.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>No Holiday</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01/12/2017</td>
      <td>173</td>
      <td>2</td>
      <td>-6.0</td>
      <td>39</td>
      <td>1.0</td>
      <td>2000</td>
      <td>-17.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>No Holiday</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01/12/2017</td>
      <td>107</td>
      <td>3</td>
      <td>-6.2</td>
      <td>40</td>
      <td>0.9</td>
      <td>2000</td>
      <td>-17.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>No Holiday</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01/12/2017</td>
      <td>78</td>
      <td>4</td>
      <td>-6.0</td>
      <td>36</td>
      <td>2.3</td>
      <td>2000</td>
      <td>-18.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Winter</td>
      <td>No Holiday</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
names = "date,bike_count,hour,temperature,humidity,wind_speed,visibility,dew_point,solar_radiation,rainfall,snowfall,seasons,holiday,functioning_day"
colnames = names.split(",")
bike_df.columns = colnames 
```

Using the info() method, we get a brief description of the data, in particular, we can observe that the dataset has 8760 rows, and none of the attributes presents missing values. Another observation is that the *Date*, *Seasons*, *Holiday* and *Functioning day* attributes are type object.


```python
bike_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8760 entries, 0 to 8759
    Data columns (total 14 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   date             8760 non-null   object 
     1   bike_count       8760 non-null   int64  
     2   hour             8760 non-null   int64  
     3   temperature      8760 non-null   float64
     4   humidity         8760 non-null   int64  
     5   wind_speed       8760 non-null   float64
     6   visibility       8760 non-null   int64  
     7   dew_point        8760 non-null   float64
     8   solar_radiation  8760 non-null   float64
     9   rainfall         8760 non-null   float64
     10  snowfall         8760 non-null   float64
     11  seasons          8760 non-null   object 
     12  holiday          8760 non-null   object 
     13  functioning_day  8760 non-null   object 
    dtypes: float64(6), int64(4), object(4)
    memory usage: 958.2+ KB


Another way of exploring the dataset is by looking at the value counts for the various attributes; in particular, we want to know more about the *Hour* attribute. In the table below, we have that for each value of the *Hour* we have the same number of observations, this is telling us that the *Rented bike count* is organised by day and hour of the day.


```python
bike_df['hour'].value_counts()
```




    0     365
    8     365
    15    365
    7     365
    22    365
    14    365
    6     365
    21    365
    13    365
    5     365
    20    365
    12    365
    4     365
    19    365
    11    365
    3     365
    18    365
    10    365
    2     365
    17    365
    9     365
    1     365
    16    365
    23    365
    Name: hour, dtype: int64



The *describe()* method returns a summary of the numerical values.


```python
bike_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bike_count</th>
      <th>hour</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>wind_speed</th>
      <th>visibility</th>
      <th>dew_point</th>
      <th>solar_radiation</th>
      <th>rainfall</th>
      <th>snowfall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
      <td>8760.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>704.602055</td>
      <td>11.500000</td>
      <td>12.882922</td>
      <td>58.226256</td>
      <td>1.724909</td>
      <td>1436.825799</td>
      <td>4.073813</td>
      <td>0.569111</td>
      <td>0.148687</td>
      <td>0.075068</td>
    </tr>
    <tr>
      <th>std</th>
      <td>644.997468</td>
      <td>6.922582</td>
      <td>11.944825</td>
      <td>20.362413</td>
      <td>1.036300</td>
      <td>608.298712</td>
      <td>13.060369</td>
      <td>0.868746</td>
      <td>1.128193</td>
      <td>0.436746</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-17.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>-30.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>191.000000</td>
      <td>5.750000</td>
      <td>3.500000</td>
      <td>42.000000</td>
      <td>0.900000</td>
      <td>940.000000</td>
      <td>-4.700000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>504.500000</td>
      <td>11.500000</td>
      <td>13.700000</td>
      <td>57.000000</td>
      <td>1.500000</td>
      <td>1698.000000</td>
      <td>5.100000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1065.250000</td>
      <td>17.250000</td>
      <td>22.500000</td>
      <td>74.000000</td>
      <td>2.300000</td>
      <td>2000.000000</td>
      <td>14.800000</td>
      <td>0.930000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3556.000000</td>
      <td>23.000000</td>
      <td>39.400000</td>
      <td>98.000000</td>
      <td>7.400000</td>
      <td>2000.000000</td>
      <td>27.200000</td>
      <td>3.520000</td>
      <td>35.000000</td>
      <td>8.800000</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we want to observe the numerical attributes distribution; We achieve this using a histogram. Some of the features that we can notice in hour histograms are:

1. Some of the histograms are tail heavy.
2. All the attributes have different scales.

Both findings tell us that we will require to normalise the attributes further down the project.


```python
%matplotlib inline
import matplotlib.pyplot as plt

bike_df[bike_df.columns.difference(['hour'])].hist(bins=50, figsize=(10,10))
plt.show()
```


<img src="{{site.url}}/images/End-to-end_files/End-to-end_15_0.png" style="display: block; margin: auto;" />   
    


### 2.2 Create a test set

Before we proceed to the EDA and further analyse the data, we need to create a test set, put it aside and don't look at it, the reason for this is because we want to avoid incurring in any *data snooping bias*. For this exercise, we will set aside 20% of our data set. To split the dataset the best and more convenient option is to use the *train_test_split()* function part of Scikit-Learn. It provides various useful features, such as defining a random state and generating balanced samples. The result is that we have a train set with 7008 rows and a test set with 1752 rows. In the following sections, we will be working only with the train set, and we will save the test set to validate our results by the end of the project.


```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(bike_df, test_size=0.2, random_state=42)
```


```python
len(train_set), len(test_set)
```




    (7008, 1752)



## 3. EDA

Our first approach to data exploration is going to be through data visualisation. Looking at the results, it seems that the most promising attribute to predict the bike count is the temperature. This result holds up when we look at the coefficient of correlation results presented in the following table.


```python
from pandas.plotting import scatter_matrix

scatter_matrix(train_set, figsize=(15,15))
plt.show()
```


<img src="{{site.url}}/images/End-to-end_files/End-to-end_21_0.png" style="display: block; margin: auto;" />    
    


### 3.2. Correlation matrix


```python
corr_matrix = train_set.corr()
corr_matrix["bike_count"].sort_values(ascending=True)
```




    humidity          -0.202004
    snowfall          -0.141440
    rainfall          -0.123586
    wind_speed         0.121388
    visibility         0.204672
    solar_radiation    0.258930
    dew_point          0.377737
    hour               0.418294
    temperature        0.537088
    bike_count         1.000000
    Name: bike_count, dtype: float64



## 4. Data Preparation

Instead of manually preparing the data to be used in machine learning algorithms, we will use different Scikit-Learn's classes. The transformations that we will use can be divided into two groups:

1. Categorical data.
2. Numeric data.


```python
bike = train_set.drop("bike_count", axis=1)
bike_count_labels = train_set["bike_count"].copy()
```

### 4.1. Categorical data

As observed before, the dataset contains attributes of type object. These attributes include various values, each representing a different category. In this example, we will use *one-hot encoding* to represent different categories.


```python
bike_cat = bike[["hour", "seasons", "holiday", "functioning_day"]]

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
bike_cat_encoded = cat_encoder.fit_transform(bike_cat)

cat_encoder.categories_
```




    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23]),
     array(['Autumn', 'Spring', 'Summer', 'Winter'], dtype=object),
     array(['Holiday', 'No Holiday'], dtype=object),
     array(['No', 'Yes'], dtype=object)]



### 4.2. Feature scaling

ML algorithms don't perform well when the numerical input attributes have different scales. To solve this issue, we will use a scaling known as *standardisation*. The process consists of first subtracting the mean value and then dividing by the standard deviation. 


```python
bike_num = bike.drop(["date", "hour", "seasons", "holiday", "functioning_day"], axis = 1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])
```

### 4.3. Full pipeline

Finally, we can use Scikit-Learn's *ColumnTransformer* class to execute all the required transformations at once.


```python
from sklearn.compose import ColumnTransformer

num_attribs = list(bike_num)
cat_attribs = list(bike_cat)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

bike_prepared = full_pipeline.fit_transform(bike)
```

## 5. Model selection and training

First, we have to select an evaluation metric to determine the performance of different models. We use the **RMSE** (root-mean-square error)

### 5.1. Linear regression model: 

Considering that we are tackling a regression problem, a good start is to fit a linear regression model. The RMSE for the training set using a linear model, gave us a value of **375.05**. The result itself is not bad; however, we still need to evaluate how good the "fit" is. When we look at the distribution of the bike count labels vs the model's predictions, we observe that our predictions do not fit the data well.  


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(bike_prepared, bike_count_labels)

from sklearn.metrics import mean_squared_error

bike_count_predictions = lin_reg.predict(bike_prepared)

lin_mse = mean_squared_error(bike_count_labels, bike_count_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    375.0516741408278




```python
plt.hist(bike_count_labels, bins=100)
plt.hist(bike_count_predictions, bins=100)
plt.show()
```


<img src="{{site.url}}/images/End-to-end_files/End-to-end_34_0.png" style="display: block; margin: auto;" />    
    


### 5.2. Decision Tree regressor

We may want to try a decision tree method since it is useful in finding nonlinear relationships in the data. When observing the results, we get that the RMSE is 0, which is a clear indication of overfitting.


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(bike_prepared, bike_count_labels)

tree_bike_preds = tree_reg.predict(bike_prepared)

tree_mse = mean_squared_error(bike_count_labels, tree_bike_preds)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



### 5.3. Cross-Validation

Using cross-validation, we obtain a model that is no longer overfitting the data (RMSE 325), which is a sign of improvement. In the following step, we will attempt to reduce the **RMSE** even further.


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, bike_prepared, bike_count_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```


```python
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
display_scores(tree_rmse_scores)
```

    Scores:  [312.48952279 327.03865154 330.53729225 312.05661498 353.79473727
     302.75621893 346.98723907 318.70136814 319.13351662 326.80624885]
    Mean:  325.03014104485845
    Standard deviation:  14.978531885807717


### 5.4. Random forest & Cross-Validation

Looking at the results, we observe significant improvements compared to previous approaches (RMSE, not 0.0 and less than 375 and 325). However, by studying the difference between the validation sets RMSE scores and the training set, we can still find clues that indicate overfitting of the data (90 much less than 242).


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(bike_prepared, bike_count_labels)

forest_bike_pred = forest_reg.predict(bike_prepared)

forest_mse = mean_squared_error(bike_count_labels, forest_bike_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    89.62892975159357




```python
scores = cross_val_score(forest_reg, bike_prepared, bike_count_labels,
                         scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)
```

    Scores:  [236.20268952 260.93777771 230.62275149 226.14701357 273.06526787
     231.31115863 248.3352994  223.15298577 250.25856952 244.78357333]
    Mean:  242.48170868206284
    Standard deviation:  15.235770987343177



```python
plt.hist(bike_count_labels, bins=100)
plt.hist(forest_bike_pred, bins = 100)
plt.show()
```


<img src="{{site.url}}/images/End-to-end_files/End-to-end_43_0.png" style="display: block; margin: auto;" />    



## 6. Fine-tuning of the model

### 6.1. Grid search

Now that we have found a model that looks promising (random forest) it is time to find the best set of hyperparameters for our model. In this case, we can use various alternatives, including **grid search** and **randomized search**. For this example, we will be using the former.

Looking at the results for the grid search, we improved the **RMSE** and got a value of 237.53. when we observe the feature importance, we get the attributes associated with the hour are the least important; however, since we want to predict the demand by the hour, we will not remove them. 


```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[40, 45, 50], 'max_features':[15, 16, 17, 18, 19, 20]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(bike_prepared, bike_count_labels)

grid_search.best_params_
```




    {'max_features': 20, 'n_estimators': 50}




```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    240.72088103712125 {'max_features': 15, 'n_estimators': 40}
    241.56268780114797 {'max_features': 15, 'n_estimators': 45}
    239.4167959635331 {'max_features': 15, 'n_estimators': 50}
    240.8534059119795 {'max_features': 16, 'n_estimators': 40}
    241.1778691983095 {'max_features': 16, 'n_estimators': 45}
    240.1210789782961 {'max_features': 16, 'n_estimators': 50}
    241.01047399910235 {'max_features': 17, 'n_estimators': 40}
    239.97517499642979 {'max_features': 17, 'n_estimators': 45}
    239.16082046598981 {'max_features': 17, 'n_estimators': 50}
    239.83144053694588 {'max_features': 18, 'n_estimators': 40}
    239.61695149480622 {'max_features': 18, 'n_estimators': 45}
    239.92087501102722 {'max_features': 18, 'n_estimators': 50}
    240.27069775846425 {'max_features': 19, 'n_estimators': 40}
    242.36403625877026 {'max_features': 19, 'n_estimators': 45}
    239.0213192886211 {'max_features': 19, 'n_estimators': 50}
    242.25908800023421 {'max_features': 20, 'n_estimators': 40}
    240.5824740636072 {'max_features': 20, 'n_estimators': 45}
    238.38489444027496 {'max_features': 20, 'n_estimators': 50}


### 6.2. Feature importance


```python
feature_importances = grid_search.best_estimator_.feature_importances_

cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.24925523435485936, 'temperature'),
     (0.11892072895820302, 'humidity'),
     (0.06868069921827176, 18),
     (0.06009045183113895, 'solar_radiation'),
     (0.055163448744503345, 'dew_point'),
     (0.0327626219972338, 19),
     (0.028420059613568065, 21),
     (0.027625856729633743, 'rainfall'),
     (0.02708803371132146, 20),
     (0.02464708472882059, 'wind_speed'),
     (0.024150065562431888, 'visibility'),
     (0.021775564006770304, 17),
     (0.020669664758574534, 22),
     (0.017360559924392078, 8),
     (0.0129605654461304, 4),
     (0.010173976545120967, 5),
     (0.008343711223862616, 23),
     (0.0062534912228521785, 3),
     (0.006118249221910772, 16),
     (0.003802775082124766, 2),
     (0.0036395654953332007, 7),
     (0.0035982975070102136, 0),
     (0.003232067441822936, 6),
     (0.002643936093809061, 10),
     (0.0020803645999142406, 1),
     (0.0017573523372114424, 15),
     (0.0010371033137319388, 11),
     (0.0009688555368639241, 9),
     (0.0005199503824870223, 'snowfall'),
     (0.0004299681395659959, 14),
     (0.0003555204550505528, 13),
     (0.0003067776133455992, 12)]



### 6.3. Evaluate results on the test set

The final step is to evaluate the performance of our system on the test set. The process consists of transforming the test data using our full pipeline and then make predictions using our final model. The **final RMSE** is 244.62 which is not too far from our train results. 


```python
final_model = grid_search.best_estimator_

X_test = test_set.drop("bike_count", axis=1)
y_test = test_set["bike_count"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
```




    244.42099470392225




```python
plt.hist(y_test, bins=100)
plt.hist(final_predictions, bins=100)
plt.show()
```


<img src="{{site.url}}/images/End-to-end_files/End-to-end_51_0.png" style="display: block; margin: auto;" />
    



```python
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```




    array([228.46763229, 259.39503908])



## 7. Conclussion 

We presented all the necessary steps to build a system that makes predictions based on observed attributes. We also showed various ways to improve the system by evaluating the results. Further steps should include launch, monitoring and maintenance of the system.


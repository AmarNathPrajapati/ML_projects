import pandas as pd
housing = pd.read_csv('data.csv')





housing.describe()




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))




from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\n Rows in test set: {len(test_set)}")


# # Stratify Sampling



from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    stratify_train_set = housing.loc[train_index]
    stratify_test_set = housing.loc[test_index]




stratify_test_set['CHAS'].value_counts()




stratify_train_set['CHAS'].value_counts()



housing = stratify_train_set.copy()


# # Co-relation Cofficient



corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# # Plotting Co-relations graph 



from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM',"ZN",'LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))




housing.plot(kind='scatter',x='RM',y="MEDV",alpha=0.8)




housing.head()


# # Combining The Attributes



housing['TaxPerRM'] = housing['TAX']/housing['RM']




housing.head()




corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)




housing.plot(kind='scatter',x='TaxPerRM',y="MEDV",alpha=0.8)


# # Handling Missing Values



housing = stratify_train_set.drop("MEDV",axis=1)
housing_labels = stratify_train_set["MEDV"].copy()




from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


#


imputer.statistics_




x = imputer.transform(housing)
housing_tr = pd.DataFrame(x,columns = housing.columns)




housing_tr.describe()


# # Creating Pipeline



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)





housing_num_tr
housing_num_tr.shape


# # Selecting Model for JAUNPUR Real Estates

# # Training with LinearRegression



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_num_tr,housing_labels)



some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)



prepared_data[0]




list(some_labels)


# # Evaluating the model



from sklearn.metrics import mean_squared_error
import numpy as np
housing_predictions = model.predict(housing_num_tr)




# mse = mean_squared_error(housing_labels,housing_predictions)
# rmse = np.sqrt(mse)
# rmse
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
def print_scores(scores):
    print("Scores are: ",scores)
    print("Mean: ",scores.mean())
    print("std_deviation: ",scores.std())
print_scores(rmse_scores)


# # Training with Decision Tree Regression



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)
rmse


# Above result is showing zero mean sqaured error, it means our model overfitted on this data
# we have to resolve the problem of overfitting.
# Because I want that my model should learn trande not noise of data

# # Using  Cross validaton Technique (kfold group data point)



from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores




print_scores(rmse_scores)


# # Training The model with Random forest Regressor



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
housing_predictions = model.predict(housing_num_tr)
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
print_scores(rmse_scores)


# # Launching The model using Sklearn Joblib



from joblib import dump,load
dump(model,"Jaunpur.joblib")


# # Model Testing




x_test = stratify_test_set.drop("MEDV",axis=1)
y_test = stratify_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse




print(final_predictions,list(y_test))







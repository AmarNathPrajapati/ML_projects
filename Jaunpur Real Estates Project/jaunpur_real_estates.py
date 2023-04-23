import pandas as pd
import pickle
housing = pd.read_csv('data.csv')

from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)

# # Stratify Sampling

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    stratify_train_set = housing.loc[train_index]
    stratify_test_set = housing.loc[test_index]

housing = stratify_train_set.copy()


# # Handling Missing Values
from sklearn.impute import SimpleImputer

housing = stratify_train_set.drop("MEDV",axis=1)
housing_labels = stratify_train_set["MEDV"].copy()


# # Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)


# # Selecting Model for JAUNPUR Real Estates

def print_scores(scores):
    print("Scores are: ",scores)
    print("Mean: ",scores.mean())
    print("std_deviation: ",scores.std())


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


# # Model Testing

x_test = stratify_test_set.drop("MEDV",axis=1)
y_test = stratify_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)


print(final_predictions,list(y_test))

# Saving the mode using pickle
pickle.dump(model,open('real_estates.pkl','wb'))






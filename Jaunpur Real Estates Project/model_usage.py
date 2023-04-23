from joblib import dump,load
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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


# model = load('Jaunpur.joblib')
model = pickle.load(open('real_estates.pkl','rb'))

arr = np.array([[0.03578,20,3.33,0,0.4429,7.82,64.5,4.6947,5,216,14.9,387.31,3.76]])
print(arr)

prepared_data = my_pipeline.transform(arr)
print(prepared_data)


a=model.predict(prepared_data)
print(round(a[0],3))






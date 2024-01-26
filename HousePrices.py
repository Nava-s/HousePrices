from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector

#import the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#check for the number of rows and columns
np.shape(train)

#create an object with the columns with missing data
missing = train.columns[train.count() < len(train)]
missing_test = test.columns[test.count() < len(test)]

#clean the quantitative data with mean substitution
for i in missing:
    if train[i].dtypes == 'float64' or train[i].dtypes == 'int64':
        mean = np.mean(train[i])
        train[i] = train[i].apply(lambda x: mean if pd.isna(x) == True else x)

for i in missing_test:
    if test[i].dtypes == 'float64' or test[i].dtypes == 'int64':
        mean = np.mean(test[i])
        test[i] = test[i].apply(lambda x: mean if pd.isna(x) == True else x)

#check how many NaN for other columns 
for i in missing:
  print(i," - ",np.sum(pd.isna(train[i])))

for i in missing_test:
  print(i," - ",np.sum(pd.isna(test[i])))

#Drop the features with too many NaN
train = train.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1)
test = test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1)

#recreate an object with the columns with missing data after dropping
missing = train.columns[train.count() < len(train)]
missing_test = test.columns[test.count() < len(test)]

#Substitute the Qualitative values with the most frequent value (mode)
for i in missing:
    mode = train[i].mode()[0]
    train[i] = train[i].apply(lambda x: mode if pd.isna(x) == True else x)   

for i in missing_test:
    mode = test[i].mode()[0]
    test[i] = test[i].apply(lambda x: mode if pd.isna(x) == True else x)   

#create a df with quntiative data
qt_index = []
ql_index = []
qt_index_test = []
ql_index_test = []

col = train.columns
col_test = test.columns

for i in col:
    if train[i].dtypes == 'int64' or train[i].dtypes == 'float64':
        qt_index.append(i)
    else:
        ql_index.append(i)

for i in col_test:
    if test[i].dtypes == 'int64' or test[i].dtypes == 'float64':
        qt_index_test.append(i)
    else:
        ql_index_test.append(i)

qt_index = qt_index[:-1]
qt_index = qt_index[1:]

qt_index_test = qt_index_test[1:]


# Unisci i dati di addestramento e di test
combined = pd.concat([train[ql_index], test[ql_index_test]])

# Applica get_dummies ai dati combinati
one_hot_combined = pd.get_dummies(combined, drop_first=True)

# Dividi nuovamente i dati in addestramento e test
one_hot = one_hot_combined.iloc[:len(train)]
one_hot_test = one_hot_combined.iloc[len(train):]




lr = LinearRegression()
scaler = StandardScaler()
train[qt_index] = scaler.fit_transform(train[qt_index])
test[qt_index_test] = scaler.fit_transform(test[qt_index_test])
train = train.drop(ql_index, axis = 1)
test = test.drop(ql_index_test, axis = 1)
train = pd.concat([train, one_hot], axis = 1)
test = pd.concat([test,one_hot_test], axis = 1)
X = train[train.columns]
X_submission = test[test.columns]
X = X.drop(['SalePrice','Id'], axis = 1)
X_submission = X_submission.drop(['Id'], axis = 1)
y = train['SalePrice']

#train test split
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=123)

#Introducing SBS 
sfs = SequentialFeatureSelector(lr,
                                n_features_to_select=100)

sfs.fit(X,y)
features = sfs.get_feature_names_out()

X_train = X_train[features]
X_test = X_test[features]
X_submission = X_submission[features]

lr.fit(X_train,y_train)
predictions = lr.predict(X_train)
train_score = lr.score(X_train,y_train)
test_score = lr.score(X_test,y_test)
print(f'train score is {train_score:.2f} \n test score is {test_score:.2f}')
X = X[features]
lr.fit(X,y)
y_predicted_sub = lr.predict(X_submission)

submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_predicted_sub})
submission_df.to_csv('submission.csv', index=False,sep = ',')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


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

scaler = StandardScaler()
scaler.fit(train[qt_index])
train[qt_index] = scaler.transform(train[qt_index])
test[qt_index_test] = scaler.transform(test[qt_index_test])

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

# Definisci il modello della rete neurale
model = Sequential([
    Dense(64, activation='relu', input_shape=(232,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

# Compila il modello
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',  # Mean Squared Error per il problema di regressione
              metrics=['mse'])  # Possiamo anche monitorare l'MSE durante l'addestramento

# Addestra il modello
history = model.fit(X_train, y_train, epochs=100, batch_size=None, validation_split=0.2, verbose=1)

# Valuta il modello
train_loss, train_mse = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f'Train MSE: {train_mse:.2f}')
print(f'Test MSE: {test_mse:.2f}')

# Eseguire le predizioni sul dataset di test
predictions_train_nn = model.predict(X_train)
predictions_test_nn = model.predict(X_test)

# Calcola R^2 per il set di addestramento
r2_train_nn = 1 - (np.sum(np.square(y_train.values - predictions_train_nn.flatten())) / np.sum(np.square(y_train.values - np.mean(y_train))))
print(f'R^2 sul set di addestramento: {r2_train_nn:.2f}')

# Calcola R^2 per il set di test
r2_test_nn = 1 - (np.sum(np.square(y_test.values - predictions_test_nn.flatten())) / np.sum(np.square(y_test.values - np.mean(y_test))))
print(f'R^2 sul set di test: {r2_test_nn:.2f}')

# Esegui le predizioni sul dataset di sottomissione X_submission
y_predicted_sub = model.predict(X_submission)

# Appiattisci la matrice 2D in un array 1D
y_predicted_sub_flat = y_predicted_sub.flatten()

# Crea un DataFrame per le predizioni di sottomissione
submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_predicted_sub_flat})

# Salva il DataFrame come file CSV di sottomissione
submission_df.to_csv('submission.csv', index=False, sep=',')

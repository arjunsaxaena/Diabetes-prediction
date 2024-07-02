import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import  roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import shap

df=pd.read_csv('D:/Downloads/diabetes_prediction_dataset.csv')
df.diabetes.value_counts() #DATA IS HIGHLY IMBALANCED
#%%
#DATA VISUALIZATION(BASIC PLOTS)

sns.relplot(data=df,x="age", y="bmi", hue="diabetes", style="diabetes")
sns.pairplot(df)
sns.displot(df, x="age", hue="diabetes")
sns.catplot(data=df, x="smoking_history", y="blood_glucose_level", hue="diabetes")
sns.catplot(data=df, x="gender", y="age", hue="diabetes")
sns.barplot(data=df, x="heart_disease",y="gender", hue="diabetes")
sns.displot(df, x="HbA1c_level", hue="diabetes")
sns.displot(df, x="blood_glucose_level", hue="diabetes")
#%%OVERSAMPLING DATA

X = df.drop('diabetes', axis=1) #INPUT DATA
y = df['diabetes']  #TARGET VARIABLE
categorical_columns = [col for col in df.columns if df[col].dtype == 'object'] #DETECTING CATEGORICAL COLUMNS

#LABEL ENCODING CATEGORICAL COLUMNS IN X
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])
    
#SCALING DATA
scaler = StandardScaler()
X = scaler.fit_transform(X)


# OVERSAMPLING USING SMOTE SINCE THE DATASET IS HIGHLY IMBALANCED
sm = SMOTE(random_state=78)
X, y = sm.fit_resample(X, y)
X, y = shuffle(X, y, random_state=42)

print("Shape of X after SMOTE: ", X.shape)
print('\nBalance of positive and negative classes (%):')
y.value_counts(normalize=True) * 100
print(y.value_counts())

#SPLITTING DATA INTO TEST AND TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
#IMPLEMENTING LOGISTIC REGRESSION
logreg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
start_time = time.time()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
end_time = time.time() 
training_time_logreg = end_time - start_time 


#ACCURACY
print('Training Time Logreg:', training_time_logreg)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,logreg.predict(X_train)))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#%%
#IMPLEMENTING SVM MODEL
svm = SVC(kernel='rbf', C=1, random_state=42)

start_time = time.time()
svm.fit(X_train, y_train)
end_time = time.time() 
training_time_svm = end_time - start_time 


#ACCURACY
print('Training Time SVM:', training_time_svm)
print('Test accuracy:', accuracy_score(y_test, svm.predict(X_test)))
print("Train accuracy:", accuracy_score(y_train,svm.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

#%%
rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time() 
training_time_rf = end_time - start_time 

y_pred = rf_model.predict(X_test)

#ACCURACY
print('Training Time SVM:', training_time_rf)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,rf_model.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict(X_test).ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR NN MODEL')
plt.legend(loc="lower right")
plt.show()

#%%
#IMPLEMENTING MODEL
gnb = GaussianNB()

start_time = time.time()
gnb.fit(X_train, y_train)
end_time = time.time() 
training_time_gnb = end_time - start_time 

#ACCURACY
print('Training Time GNB:', training_time_gnb)
print('Test accuracy:', accuracy_score(y_test,gnb.predict(X_test)))
print("Train accuracy:", accuracy_score(y_train,gnb.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, gnb.predict(X_test)))
print('Confusion Matrix:')
print(confusion_matrix(y_test, gnb.predict(X_test)))
#%%
#IMPLEMENTING NN MODEL
NN_model = tf.keras.Sequential()

NN_model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
for i in range(4):          
    NN_model.add(tf.keras.layers.Dense(units=16, activation='relu'))
NN_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

NN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = NN_model.fit(
    X_train, y_train,
    epochs=50, batch_size=32, 
    validation_split=0.2,
    #callbacks=[early_stopping]
)

y_pred = (NN_model.predict(X_test) > 0.5).astype("int32")
end_time = time.time() 
training_time_NN = end_time - start_time  


#ACCURACY
print('Training Time NN:', training_time_NN)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,(NN_model.predict(X_train) > 0.5).astype("int32")))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, NN_model.predict(X_test).ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR NN MODEL')
plt.legend(loc="lower right")
plt.show()

#%%
#USING LGBM WITH GRIDSEARCH OPTIMIZATION
params = {
        'max_depth': [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        'feature_fraction': [0.5,0.6,0.7,0.8,0.9,1.0]
        }
LGBM_model = lgb.LGBMClassifier(random_state = 1)
LGBM_model = GridSearchCV(LGBM_model,params,scoring='roc_auc',verbose=0,n_jobs=-1)

start_time = time.time()
LGBM_model.fit(X_train,y_train)
end_time = time.time() 
training_time_lgbm = end_time - start_time  


#ACCURACY
print('Training Time:', training_time_lgbm)
print('Test accuracy:', accuracy_score(y_test, LGBM_model.predict(X_test)))
print('Train accuracy:', accuracy_score(y_train, LGBM_model.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, LGBM_model.predict(X_test)))
print('Confusion Matrix:')
print(confusion_matrix(y_test, LGBM_model.predict(X_test)))


#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, LGBM_model.predict(X_test).ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR LGBM MODEL')
plt.legend(loc="lower right")
plt.show()

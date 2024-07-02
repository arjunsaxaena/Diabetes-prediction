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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

df=pd.read_csv('D:/Downloads/diabetes_prediction_dataset.csv')

df.diabetes.value_counts() #DATA IS HIGHLY IMBALANCED
df.isnull().sum()
df.columns
df[df['gender']=='Other'].count()
df['hypertension'].unique()
df['smoking_history'].unique()
df['bmi'].nunique()
df.info()
df.describe()

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

#%%
#OVERSAMPLING DATA

X = df.drop('diabetes', axis=1) #INPUT DATA
y = df['diabetes']  #TARGET VARIABLE
categorical_columns = [col for col in df.columns if df[col].dtype == 'object'] #DETECTING CATEGORICAL COLUMNS

#LABEL ENCODING CATEGORICAL COLUMNS IN X
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])
    
#SCALING DATA
sc = StandardScaler()
X = sc.fit_transform(X)


# OVERSAMPLING USING SMOTE SINCE THE DATASET IS HIGHLY IMBALANCED
sm = SMOTE(random_state=78)
X, y = sm.fit_resample(X, y)
X, y = shuffle(X, y, random_state=42)

print("Shape of X after SMOTE: ", X.shape)
print('\nBalance of positive and negative classes (%):')
y.value_counts(normalize=True) * 100
print(y.value_counts())
'''
1    91500
0    91500
'''

#SPLITTING DATA INTO TEST AND TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
#IMPLEMENTING LOGISTIC REGRESSION

logreg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=10000)

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
logreg.fit(X_train, y_train)
end_time = time.time() 
training_time_logreg = end_time - start_time 

#PREDICTING X_test
y_pred = logreg.predict(X_test)

#ACCURACY
print('Training Time Logreg:', training_time_logreg)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,logreg.predict(X_train)))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

'''
Training Time Logreg: 0.06501054763793945
Test accuracy: 0.8886885245901639
Train accuracy: 0.8857377049180328
'''

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR Logistic Regression MODEL')
plt.legend(loc="lower right")
plt.show()

#CLEARLY NOT THE BEST MODEL 0.89 AUROC SCORE

#%%
#IMPLEMENTING SVM MODEL

svm = SVC(kernel='rbf', C=1, random_state=42)

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
svm.fit(X_train, y_train)
end_time = time.time() 
training_time_svm = end_time - start_time 

#PREDICTING X_test
y_pred = svm.predict(X_test)

#ACCURACY
print('Training Time SVM:', training_time_svm)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train, svm.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

'''
Training Time SVM: 174.2328975200653
Test accuracy: 0.9066302367941712
Train accuracy: 0.9062295081967213
'''

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, svm.predict(X_test).ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR SVM MODEL')
plt.legend(loc="lower right")
plt.show()

#%%
#IMPLEMENTING RANDOM FOREST MODEL

rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time() 
training_time_rf = end_time - start_time 

#PREDICTING X_test
y_pred = rf_model.predict(X_test)

#ACCURACY
print('Training Time SVM:', training_time_rf)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,rf_model.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

'''
Training Time SVM: 8.058397769927979
Test accuracy: 0.9764663023679417
Train accuracy: 0.9996096799375488
'''

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR RF MODEL')
plt.legend(loc="lower right")
plt.show()

#BEST MODEL SO FAR 0.98 AUROC SCORE

#%%
#IMPLEMENTING NB MODEL

gnb = GaussianNB()

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
gnb.fit(X_train, y_train)
end_time = time.time() 
training_time_gnb = end_time - start_time 

#PREDICTING X_test
y_pred = gnb.predict(X_test)

#ACCURACY
print('Training Time GNB:', training_time_gnb)
print('Test accuracy:', accuracy_score(y_test,y_pred))
print("Train accuracy:", accuracy_score(y_train,gnb.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

'''
Training Time GNB: 0.03300666809082031
Test accuracy: 0.8402732240437158
Train accuracy: 0.8394145199063232
'''

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC FOR NB MODEL')
plt.legend(loc="lower right")
plt.show()

#WORST MODEL SO FAR 0.84 auroc score

#%%
#IMPLEMENTING NN MODEL

NN_model = tf.keras.Sequential()
NN_model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
for i in range(4):          
    NN_model.add(tf.keras.layers.Dense(units=16, activation='relu'))
NN_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
NN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
history = NN_model.fit(
    X_train, y_train,
    epochs=50, batch_size=16, 
    validation_split=0.2,
    #callbacks=[early_stopping]
)
end_time = time.time() 
training_time_NN = end_time - start_time  

#PREDICTING X_test
y_pred = (NN_model.predict(X_test) > 0.5).astype("int32")

#ACCURACY
print('Training Time NN:', training_time_NN)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print("Train accuracy:", accuracy_score(y_train,(NN_model.predict(X_train) > 0.5).astype("int32")))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

'''
Training Time NN: 392.4646990299225
Test accuracy: 0.9181420765027323
Train accuracy: 0.9198594847775176
'''

#FEATURE IMPORTANCE USING SHAP
explainer = shap.Explainer(NN_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, max_display=8)

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

#0.98 AUROC SCORE

#%%
#IMPLEMENTING LGBM WITH GRIDSEARCH OPTIMIZATION

params = {
        'max_depth': [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        'feature_fraction': [0.5,0.6,0.7,0.8,0.9,1.0]
        } #0.6

LGBM_model = lgb.LGBMClassifier(random_state = 1)
LGBM_model = GridSearchCV(LGBM_model,params,scoring='roc_auc',verbose=0,n_jobs=-1)

#FITTING THE MODEL AND CAPTURING TIME TAKEN BY THE MODEL
start_time = time.time()
LGBM_model.fit(X_train,y_train)
end_time = time.time() 
training_time_lgbm = end_time - start_time  

#PREDICTING X_test
y_pred = (LGBM_model.predict(X_test))


#ACCURACY
print('Training Time:', training_time_lgbm)
print('Test accuracy:', accuracy_score(y_test, y_pred))
print('Train accuracy:', accuracy_score(y_train, LGBM_model.predict(X_train)))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

'''
Training Time: 170.11738801002502
Test accuracy: 0.9793078324225866
Train accuracy: 0.980967993754879
'''

#AUROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred.ravel())
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

#AUROC SCORE OF 0.98


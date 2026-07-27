import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

url="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df=pd.read_csv(url)

print("DATASET INFO : \n",df.info())
print("CLASS DISTRIBUTION : \n",df['class'].value_counts())

X=df.drop(columns=['class'])
Y=df['class']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


kf=KFold(n_splits=5,shuffle=True,random_state=42)

rf_model=RandomForestClassifier(random_state=42)
scores_kfold=cross_val_score(rf_model,X_train,Y_train,cv=kf,scoring='accuracy')

print(f"K-Fold cross validation scores : {scores_kfold}")
print(f"Mean Accuracy K-Fold : {scores_kfold.mean():.2f}")

# Stratified K-Fold

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

scores_stratified=cross_val_score(rf_model,X_train,Y_train,cv=skf,scoring='accuracy')

print(f"\n Stratiefied K-Fold cross validation scores : {scores_stratified}")
print(f"Mean Accuracy Strartified K-Fold : {scores_stratified.mean():.2f}")
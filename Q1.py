from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data=load_breast_cancer()
X,Y=data.data,data.target

print("Features : ",data.feature_names)
print("Classes : ",data.target_names)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model_default=RandomForestClassifier(random_state=42)
model_default.fit(X_train,Y_train)

y_pred_default=model_default.predict(X_test)

accuracy_default=accuracy_score(Y_test,y_pred_default)

print("\n Default Model Accuracy Score : ",accuracy_default)
print("Default Model Classification Report : ",classification_report(Y_test,y_pred_default))


# For tuned accuracy score and classification report using parameters

rf_tuned=RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf_tuned.fit(X_train,Y_train)
y_pred_tuned=rf_tuned.predict(X_test)

accuracy_tuned=accuracy_score(Y_test,y_pred_tuned)

print("\n Tuned Model Accuracy Score : ",accuracy_tuned)
print("Tuned Model Classification Report : ",classification_report(Y_test,y_pred_tuned))
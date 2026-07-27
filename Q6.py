from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np


data=load_iris()
X,Y=data.data,data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

param_grid={
    'n_estimators':[50,100,150],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2]
}

grid_search=GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train,Y_train)

best_params=grid_search.best_params_
best_score=grid_search.best_score_

best_model=grid_search.best_estimator_

y_pred_grid=grid_search.predict(X_test)

accuracy_grid=accuracy_score(Y_test,y_pred_grid)

print(f"Test Accuracy (Grid Search): {accuracy_grid:.2f}")
print(f"\n Classification Report :\n {classification_report(Y_test,y_pred_grid)}")


# Random Search Working


param_dist={
    'C':np.logspace(-3,3,10),
    'kernel':['linear','rbf','poly','sigmoid'],
    'gamma':['scale','auto']
}

random_search=RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)


random_search.fit(X_train,Y_train)

best_params_random=random_search.best_params_
best_score=random_search.best_score_

best_model_random=random_search.best_estimator_

y_pred_random=random_search.predict(X_test)

accuracy_random=accuracy_score(Y_test,y_pred_grid)

print(f"\n Test Accuracy (Random Search): {accuracy_random:.2f}")
print(f"\n Classification Report :\n {classification_report(Y_test,y_pred_random)}")


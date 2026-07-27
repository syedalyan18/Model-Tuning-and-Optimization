from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV


data=load_iris()
X,Y=data.data,data.target

print("Features : ",data.feature_names)
print("Classes : ",data.target_names)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

param_grid={
    'n_estimators':[50,100,150],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10]
}

grid_search = GridSearchCV (
    estimator = RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train,Y_train)

grid_best_model=grid_search.best_estimator_
y_pred_grid=grid_best_model.predict(X_test)
accuracy_grid=accuracy_score(Y_test,y_pred_grid)

print("Best Hyperparamters (Grid Search): ",grid_best_model)
print(f"Grid Search Accuracy Score : {accuracy_grid:.4f}")




param_dist= {
    'n_estimators':np.arange(50,200,10),
    'max_depth':[None ,5,10,15],
    'min_samples_split':[2,5,10,20]
}

random_search = RandomizedSearchCV (
    estimator = RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train,Y_train)

random_best_model=random_search.best_estimator_
y_pred_random=random_best_model.predict(X_test)
accuracy_random=accuracy_score(Y_test,y_pred_grid)

print("\n\n Best Hyperparamters (Random Search): ",random_search.best_params_)
print(f"Random Search Accuracy Score : {accuracy_random:.4f}")
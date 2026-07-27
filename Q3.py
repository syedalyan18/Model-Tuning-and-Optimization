from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna 


data=load_breast_cancer()
X,Y=data.data,data.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

print("Training data shape : ",X_train.shape)
print("Testing data shape : ",X_test.shape)

baseline_model=XGBClassifier(eval_metric='logloss',random_state=42)
baseline_model.fit(X_train,Y_train)

baseline_pred=baseline_model.predict(X_test)

accuracy_baseline=accuracy_score(Y_test,baseline_pred)
print(f"XGboost model accuracy : {accuracy_baseline:.4f}")


def Objective(trial):
    params= {
        'n_estimators':trial.suggest_int('n_estimators',50,500),
        'max_depth':trial.suggest_int('max_depth',3,100),
        'learning_rate':trial.suggest_float('learning_rate',0.01,0.3),
        'subsample':trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
        'gamma':trial.suggest_float('gamma',0,5),
        'reg_alpha':trial.suggest_float('reg_alpha',0,10),
        'reg_lambda':trial.suggest_float('reg_lambda',0,10),
    }

    model=XGBClassifier(eval_metric='logloss',random_state=42,**params)
    model.fit(X_train,Y_train)

    y_pred=model.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)
    return accuracy


study=optuna.create_study(direction="maximize")
study.optimize(Objective, n_trials=50)
 

print("Best Hyperparameters : ",study.best_params)
print(f"Best accuracy : {study.best_value:.4f}")


param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.6,0,8,1.0]
}


grid_search = GridSearchCV (
    estimator = XGBClassifier(eval_metric='logloss',random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train,Y_train)

print("\n\n Grid Search Best Parameters : ",grid_search.best_params_)
print(f"Grid Search Best Accuracy : {grid_search.best_score_:.4f}")

param_dist={
    'n_estimators':[50,100,200,300,400],
    'max_depth':[3,5,7,9],
    'learning_rate':[0.01,0.05,0.1,0.2],
    'subsample':[0.6,0,7,0.8,0.9,1.0],
    'colsample_bytree':[0.6,0,7,0.8,0.9,1.0]
}


random_search = RandomizedSearchCV (
    estimator = XGBClassifier(eval_metric='logloss',random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42
)


random_search.fit(X_train,Y_train)

random_best_model=random_search.best_estimator_
y_pred_random=random_best_model.predict(X_test)
accuracy_random=accuracy_score(Y_test,y_pred_random)

print("\n\n Best Hyperparamters (Random Search): ",random_search.best_params_)
print(f"Random Search Accuracy Score : {accuracy_random:.4f}")
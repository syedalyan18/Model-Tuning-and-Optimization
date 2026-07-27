from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd

california=fetch_california_housing()
X,Y=california.data,california.target
feature_names=california.feature_names

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

print("Feature Names : \n ",feature_names)
print('\n Sample Data : \n',pd.DataFrame(X,Y,columns=feature_names).head())

# LINEAR MODEL

lr_model=LinearRegression()
lr_model.fit(X_train,Y_train)

y_pred=lr_model.predict(X_test)

mse_lr=mean_squared_error(Y_test,y_pred)
print(f"Linear Regression MSE (No Regularization) : {mse_lr:.2f}")
print("Coefficients of Linear Regression: ",lr_model.coef_)


# RIDGE MODEL

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,Y_train)

Y_pred_ridge=ridge_model.predict(X_test)
mse_ridge=mean_squared_error(Y_test,Y_pred_ridge)

print(f"\n Ridge Regression MSE (No Regularization) : {mse_ridge:.2f}")
print("Coefficients of Ridge Model : ",ridge_model.coef_)

# LASSO MODEL

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,Y_train)

Y_pred_lasso=lasso_model.predict(X_test)
mse_lassoe=mean_squared_error(Y_test,Y_pred_ridge)

print(f"\n Lasso Regression MSE (No Regularization) : {mse_lassoe:.2f}")
print("Coefficients of Lasso Model : ",lasso_model.coef_)

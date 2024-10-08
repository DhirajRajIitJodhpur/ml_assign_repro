import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score

salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
salary.columns
y = salary['Salary']
X = salary[['Experience Years']]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

mlflow.set_experiment("NEW_EXPERIMENT")
with mlflow.start_run() as run:
    model_lr = LinearRegression()
    model_lr.fit(X_train,y_train)

    y_pred = model_lr.predict(X_test)

    mse_lr = mean_absolute_error(y_test,y_pred)
    r2_lr = r2_score(y_test, y_pred)
    print(f'Mean Squared Error LR: {mse_lr}')
    print(f'R-squared LR: {r2_lr}')
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse_lr", mse_lr)
    mlflow.log_metric("r2_lr", r2_lr)
    mlflow.sklearn.log_model(model_lr, "Linear Regression")
    mlflow.register_model(f"runs:/{run.info.run_id}/linear_regression_model", "LinearRegressionModel")

    print(f"Run URL: {mlflow.active_run().info.artifact_uri}")


with mlflow.start_run() as run:
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train,y_train)

    y_pred = model_rf.predict(X_test)

    mse_rf = mean_absolute_error(y_test,y_pred)
    r2_rf = r2_score(y_test, y_pred)
    print(f'Mean Squared Error LR: {mse_rf}')
    print(f'R-squared LR: {r2_rf}')
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("mse_rf", mse_rf)
    mlflow.log_metric("r2_rf", r2_rf)
    mlflow.sklearn.log_model(model_rf, "RandomForestRegressor")
    mlflow.register_model(f"runs:/{run.info.run_id}/RandomForest_model", "RandomForestModel")

    print(f"Run URL: {mlflow.active_run().info.artifact_uri}")
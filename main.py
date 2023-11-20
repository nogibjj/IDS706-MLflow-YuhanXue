"""runs a simple machine learning expirement"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    """runs a basic linear regression model
    and logs it with mlflow"""
    df = pd.read_csv("diabetes.csv", delimiter=",")

    features = df[
        [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
    ]
    target = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    clf = LinearRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("data_path", "diabetes.csv")

        mlflow.log_metric("accuracy", mse)

        mlflow.sklearn.log_model(clf, "mlruns/0")


if __name__ == "__main__":
    main()

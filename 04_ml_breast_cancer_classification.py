import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main():
    np.random.seed(42)

    # load the dataset
    data = pd.read_csv("data/breast_cancer.csv")
    data = data.dropna().sample(frac=1.0, random_state=42)
    print(data.head())
    print(data.info())
    x = data.drop(columns=["diagnosis"]).values
    # convert the target variable to binary (0 for benign, 1 for malignant)
    data["diagnosis"] = data["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
    y = data["diagnosis"].values
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    x_train, x_test = x[:-100], x[-100:]
    y_train, y_test = y[:-100], y[-100:]
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    # standardize the features
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # train the model (sklearn's SGDRegressor)
    model1 = SGDClassifier(
        loss="log_loss",
        penalty=None,
        learning_rate="constant",
        eta0=0.001,
        max_iter=10_000,
        tol=None,
        shuffle=False,
        average=True,
    )
    model1.fit(x_train, y_train, coef_init=[0.0] * x_train.shape[1], intercept_init=0.0)
    sgd_acc = calculate_accuracy(y_test, model1.predict(x_test))
    model1_coef_str = ", ".join(f"{ci:.4f}" for ci in model1.coef_[0])
    print(
        f"Sklearn SGDRegressor parameters: a = [{model1_coef_str}], b = {model1.intercept_[0]:.4f}"
    )

    # compare with sklearn's LinearRegression
    model2 = LogisticRegression()
    model2.fit(x_train, y_train)
    lr_acc = calculate_accuracy(y_test, model2.predict(x_test))
    model2_coef_str = ", ".join(f"{ci:.4f}" for ci in model2.coef_[0])
    print(
        f"Sklearn LinearRegression parameters: a = [{model2_coef_str}], b = {model2.intercept_[0]:.4f}"
    )

    print(f"Sklearn SGDRegressor accuracy: {sgd_acc:.4f}")
    print(f"Sklearn LinearRegression accuracy: {lr_acc:.4f}")


if __name__ == "__main__":
    main()

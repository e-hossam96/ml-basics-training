import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor, LinearRegression


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    loss = np.mean((y_true - y_pred) ** 2, axis=0)
    return float(loss)


def main():
    np.random.seed(42)

    # load the dataset
    data = pd.read_csv("data/housing.csv")
    data = data.dropna().sample(frac=1.0, random_state=42)
    print(data.head())
    print(data.info())
    x = data.drop(columns=["median_house_value"]).values
    y = data["median_house_value"].values
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    x_train, x_test = x[:-1000], x[-1000:]
    y_train, y_test = y[:-1000], y[-1000:]
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    # standardize the features
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # train the model (sklearn's SGDRegressor)
    model1 = SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.001,
        max_iter=10_000,
        tol=None,
        shuffle=False,
        average=True,
    )
    model1.fit(x_train, y_train, coef_init=[0.0] * x_train.shape[1], intercept_init=0.0)
    sgd_test_loss = compute_loss(y_test, model1.predict(x_test))
    model1_coef_str = ", ".join(f"{ci:.4f}" for ci in model1.coef_)
    print(
        f"Sklearn SGDRegressor parameters: a = [{model1_coef_str}], b = {model1.intercept_[0]:.4f}"
    )

    # compare with sklearn's LinearRegression
    model2 = LinearRegression()
    model2.fit(x_train, y_train)
    lr_test_loss = compute_loss(y_test, model2.predict(x_test))
    model2_coef_str = ", ".join(f"{ci:.4f}" for ci in model2.coef_)
    print(
        f"Sklearn LinearRegression parameters: a = [{model2_coef_str}], b = {model2.intercept_:.4f}"
    )

    print(f"Sklearn SGDRegressor test loss (MSE): {sgd_test_loss:.4f}")
    print(f"Sklearn LinearRegression test loss (MSE): {lr_test_loss:.4f}")


if __name__ == "__main__":
    main()

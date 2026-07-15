# one feature ml model
import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    loss = np.mean((y_true - y_pred) ** 2, axis=0)
    return float(loss)


def predict(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    return x @ a + b


def train(
    x: np.ndarray,
    y_true: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
) -> tuple[np.ndarray, float]:
    a = np.array([0.0] * x.shape[1])
    b = 0.0
    n_samples = x.shape[0]

    for _ in range(n_iterations):
        y_pred = predict(x, a, b)

        # Compute gradients
        da = (y_pred - y_true) @ x
        da = (2 / n_samples) * da
        db = np.sum(y_pred - y_true, axis=0)
        db = (2 / n_samples) * float(db)

        # Update parameters
        a -= learning_rate * da
        b -= learning_rate * db

    return a, b


def main():
    np.random.seed(42)
    w_true = np.array([0.5] * 8)
    b_true = 1.0
    # true function is y = a_true * x + b_true (nonlinear)

    # generate some sample data
    x = np.random.uniform(low=-1.0, high=1.0, size=(1000, 8))
    y = x @ w_true + b_true + np.random.uniform(low=0.0, high=0.2, size=1000)
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    x_train, x_test = x[:800], x[800:]
    y_train, y_test = y[:800], y[800:]

    # train the model (linear model fit to nonlinear data - expect underfitting)
    a, b = train(x_train, y_train, learning_rate=0.001, n_iterations=10_000)
    test_loss = compute_loss(y_test, predict(x_test, a, b))
    w_true_str = ", ".join(f"{wi:.4f}" for wi in w_true)
    print(f"True model parameters: w = [{w_true_str}], b = {b_true:.4f}")
    a_str = ", ".join(f"{ai:.4f}" for ai in a)
    print(f"Full batch GD model parameters: w = [{a_str}], b = {b:.4f}")

    # compare with sklearn's SGDRegressor
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

    print(f"Full batch GD test loss (MSE): {test_loss:.4f}")
    print(f"Sklearn SGDRegressor test loss (MSE): {sgd_test_loss:.4f}")
    print(f"Sklearn LinearRegression test loss (MSE): {lr_test_loss:.4f}")


if __name__ == "__main__":
    main()

# one feature ml model
import random
from sklearn.linear_model import SGDRegressor, LinearRegression


def compute_loss(y_true: list[float], y_pred: list[float]) -> float:
    n_samples = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n_samples)) / n_samples


def predict(x: list[float], a: float, b: float) -> list[float]:
    return [a * x[i] + b for i in range(len(x))]


def train(
    x: list[float],
    y_true: list[float],
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
) -> tuple[float, float]:
    a = 0.0
    b = 0.0
    n_samples = len(x)

    for _ in range(n_iterations):
        y_pred = predict(x, a, b)

        # Compute gradients
        da = (2 / n_samples) * sum(
            x[i] * (y_pred[i] - y_true[i]) for i in range(n_samples)
        )
        db = (2 / n_samples) * sum(y_pred[i] - y_true[i] for i in range(n_samples))

        # Update parameters
        a -= learning_rate * da
        b -= learning_rate * db

    return a, b


def main():
    random.seed(42)
    a_true = 0.5
    b_true = 1.0
    # true function is y = a_true * x + b_true (nonlinear)

    # generate some sample data
    x = [random.uniform(-1.0, 1.0) for _ in range(1000)]
    y = [a_true * x[i] + b_true + random.uniform(0.0, 0.2) for i in range(1000)]

    x_train, x_test = x[:800], x[800:]
    y_train, y_test = y[:800], y[800:]

    # train the model (linear model fit to nonlinear data - expect underfitting)
    a, b = train(x_train, y_train, learning_rate=0.001, n_iterations=10_000)
    test_loss = compute_loss(y_test, predict(x_test, a, b))
    print(f"True model parameters: a = {a_true:.4f}, b = {b_true:.4f}")
    print(f"Full batch GD model parameters: a = {a:.4f}, b = {b:.4f}")

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
    model1.fit([[xi] for xi in x_train], y_train, coef_init=[0.0], intercept_init=0.0)
    sgd_test_loss = compute_loss(y_test, list(model1.predict([[xi] for xi in x_test])))
    print(
        f"Sklearn SGDRegressor parameters: a = {model1.coef_[0]:.4f}, b = {model1.intercept_[0]:.4f}"
    )

    # compare with sklearn's LinearRegression
    model2 = LinearRegression()
    model2.fit([[xi] for xi in x_train], y_train)
    lr_test_loss = compute_loss(y_test, list(model2.predict([[xi] for xi in x_test])))
    print(
        f"Sklearn LinearRegression parameters: a = {model2.coef_[0]:.4f}, b = {model2.intercept_:.4f}"
    )

    print(f"Full batch GD test loss (MSE): {test_loss:.4f}")
    print(f"Sklearn SGDRegressor test loss (MSE): {sgd_test_loss:.4f}")
    print(f"Sklearn LinearRegression test loss (MSE): {lr_test_loss:.4f}")


if __name__ == "__main__":
    main()

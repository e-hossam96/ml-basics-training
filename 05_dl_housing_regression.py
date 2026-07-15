import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class HousingRegressionModel(torch.nn.Module):
    def __init__(self, num_features: int) -> None:
        super(HousingRegressionModel, self).__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=20),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=20, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neural_net(x)


def train_batch(
    model: HousingRegressionModel,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_batch(
    model: HousingRegressionModel,
    criterion: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
    return loss.item()


def main():
    # read the dataset
    data = pd.read_csv("data/housing.csv")
    data = data.dropna().sample(frac=1.0, random_state=42)
    X = data.drop(columns=["median_house_value"]).values
    y = data["median_house_value"].values.reshape(-1, 1) / 100_000
    X_train, X_test = X[:-1000], X[-1000:]
    y_train, y_test = y[:-1000], y[-1000:]

    # standardize features and target using training-set statistics
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # create the model, loss function, and optimizer
    num_features = X_train.shape[1]
    model = HousingRegressionModel(num_features=num_features)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # training loop
    batch_size = 32
    num_epochs = 800
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            x_batch = X_train_tensor[i : i + batch_size]
            y_batch = y_train_tensor[i : i + batch_size]
            train_loss = train_batch(model, optimizer, criterion, x_batch, y_batch)
            total_train_loss += train_loss * x_batch.size(0)
        total_train_loss = total_train_loss / X_train_tensor.size(0)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:4d}/{num_epochs:4d}], Train Loss: {total_train_loss:.4f}"
            )

    # evaluate on the test set
    total_test_loss = 0.0
    for i in range(0, X_test_tensor.size(0), batch_size):
        x_batch = X_test_tensor[i : i + batch_size]
        y_batch = y_test_tensor[i : i + batch_size]
        test_loss = eval_batch(model, criterion, x_batch, y_batch)
        total_test_loss += test_loss * x_batch.size(0)
    total_test_loss = total_test_loss / X_test_tensor.size(0)
    print(f"Test Loss: {total_test_loss:.4f}")


if __name__ == "__main__":
    main()

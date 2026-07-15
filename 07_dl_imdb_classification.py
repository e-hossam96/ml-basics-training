import json
import torch
import pandas as pd
from collections import Counter


class IMDBSentimentClassificationModel(torch.nn.Module):
    def __init__(
        self, vocab_size: int, padding_idx: int, embedding_dim: int = 1024
    ) -> None:
        super(IMDBSentimentClassificationModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 2),
            torch.nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim),
            torch.nn.Linear(in_features=embedding_dim, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x.long()).mean(dim=1)
        prob = self.linear_layers(embeddings)
        return prob


def train_batch(
    model: IMDBSentimentClassificationModel,
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
    model: IMDBSentimentClassificationModel,
    criterion: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
    return loss.item()


def train_tokenizer(texts: list[str], vocab_size: int = 4096) -> dict[str, int]:
    word_counts = Counter(" ".join(texts).lower().split())
    most_common_words = word_counts.most_common(vocab_size - 2)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}

    # add special tokens for unknown words and padding
    vocab["<UNK>"] = len(vocab) - 2
    vocab["<PAD>"] = len(vocab) - 1

    return vocab


def tokenize(text: str, vocab: dict[str, int], max_num_words: int = 256) -> list[int]:
    tokens = []
    for word in text.strip().lower().split()[:max_num_words]:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(vocab["<UNK>"])

    # pad the tokens with <UNK> if the number of tokens is less than max_num_words
    while len(tokens) < max_num_words:
        tokens.append(vocab["<PAD>"])

    return tokens


def main():
    # read the dataset
    data = pd.read_csv("data/imdb.csv")
    data = data.dropna().sample(frac=1.0, random_state=42)
    data["sentiment"] = data["sentiment"].apply(lambda x: 1 if x == "P" else 0)
    X = data["review"].values
    y = data["sentiment"].values.reshape(-1, 1)
    X_train, X_test = X[:-5000], X[-5000:]
    y_train, y_test = y[:-5000], y[-5000:]

    # train tokenizer on the training data
    vocab_size = 8192
    vocab = train_tokenizer(X_train, vocab_size=vocab_size)

    # tokenize the text data
    X_train = [tokenize(text, vocab) for text in X_train]
    X_test = [tokenize(text, vocab) for text in X_test]

    # convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # create the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMDBSentimentClassificationModel(
        vocab_size=vocab_size, padding_idx=vocab["<PAD>"]
    ).to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4
    )

    # training loop
    batch_size = 64
    num_epochs = 128
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            x_batch = X_train_tensor[i : i + batch_size].to(device)
            y_batch = y_train_tensor[i : i + batch_size].to(device)
            train_loss = train_batch(model, optimizer, criterion, x_batch, y_batch)
            total_train_loss += train_loss * x_batch.size(0)
        total_train_loss = total_train_loss / X_train_tensor.size(0)
        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch + 1:4d}/{num_epochs:4d}], Train Loss: {total_train_loss:.4f}"
            )

    # evaluate on the test set
    total_test_loss = 0.0
    for i in range(0, X_test_tensor.size(0), batch_size):
        x_batch = X_test_tensor[i : i + batch_size].to(device)
        y_batch = y_test_tensor[i : i + batch_size].to(device)
        test_loss = eval_batch(model, criterion, x_batch, y_batch)
        total_test_loss += test_loss * x_batch.size(0)
    total_test_loss = total_test_loss / X_test_tensor.size(0)
    print(f"Test Loss: {total_test_loss:.4f}")

    # calculate accuracy on the test set
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_labels = (y_pred >= 0.5).float()
        accuracy = (y_pred_labels == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "models/imdb_classifier.pth")
    with open("models/imdb_classifier_params.json", "w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "padding_idx": vocab["<PAD>"],
                "unk_idx": vocab["<UNK>"],
            },
            f,
        )
    with open("models/imdb_classifier_vocab.json", "w") as f:
        json.dump(vocab, f)

    print("Model saved to: 'models/imdb_classifier.pth'")
    print("Model params saved to: 'models/imdb_classifier_params.json'")
    print("Model vocab saved to: 'models/imdb_classifier_vocab.json'")


if __name__ == "__main__":
    main()

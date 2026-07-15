import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel


class IMDBSentimentClassificationModel(torch.nn.Module):
    def __init__(
        self, vocab_size: int, padding_idx: int, embedding_dim: int = 128
    ) -> None:
        super(IMDBSentimentClassificationModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x.long()).mean(dim=1)
        prob = self.linear_layers(embeddings)
        return prob


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


with open("models/imdb_classifier_params.json") as f:
    params = json.load(f)
with open("models/imdb_classifier_vocab.json") as f:
    vocab = json.load(f)

model = IMDBSentimentClassificationModel(
    vocab_size=params["vocab_size"], padding_idx=params["padding_idx"]
)
model.load_state_dict(torch.load("models/imdb_classifier.pth", map_location="cpu"))
model.eval()

app = FastAPI()


class Review(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str


@app.post("/predict", response_model=Prediction)
def predict(review: Review) -> Prediction:
    tokens = torch.tensor([tokenize(review.text, vocab)], dtype=torch.float32)
    with torch.no_grad():
        prob = model(tokens).item()
    return Prediction(label="Positive" if prob >= 0.5 else "Negative")

import os
import torch
import pandas as pd
import pytorch_lightning as pl
from transformers import BertTokenizer

# Define your PyTorch Lightning model class


class SimpleSentimentModel(pl.LightningModule):
    def __init__(self, vocab_size, pad_index, embedding_dim=100, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc1 = torch.nn.Linear(embedding_dim, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        x = x.to(self.device)  # Ensure inputs are on the same device as model
        embedded = self.embedding(x).mean(dim=1)
        hidden = torch.nn.functional.relu(self.fc1(embedded))
        return self.fc2(hidden)

    def configure_optimizers(self):
        # Define optimizers and learning rate scheduler here
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return optimizer


def predict_sentiment(text, model, vocab, tokenizer, sentiment_labels):
    tokenized_text = [vocab.get(token, vocab['<unk>'])
                      for token in tokenizer.tokenize(text)]
    input_tensor = torch.tensor(
        [tokenized_text], dtype=torch.long).to(model.device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
    return sentiment_labels[predicted_index]  # Return the sentiment label


def load_model_from_checkpoint(checkpoint_path, vocab_size, pad_index):
    model = SimpleSentimentModel.load_from_checkpoint(
        checkpoint_path,
        vocab_size=vocab_size,
        pad_index=pad_index
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def load_vocab(vocab_file):
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"The file {vocab_file} does not exist.")
    with open(vocab_file, 'r') as file:
        vocab = {line.strip(): idx for idx, line in enumerate(file.readlines())}
    return vocab


def main(input_text, model_version, root_path):
    vocab_file = 'vocab.txt'
    vocab = load_vocab(vocab_file)
    vocab_size = len(vocab)
    pad_index = vocab['<pad>']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Define the valid sentiment labels
    sentiment_labels = ['neutral', 'negative', 'positive']

    checkpoint_directory = os.path.join(
        root_path, model_version, "checkpoints")
    checkpoint_files = [f for f in os.listdir(
        checkpoint_directory) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_files[0])

    model = load_model_from_checkpoint(checkpoint_path, vocab_size, pad_index)

    sentiment = predict_sentiment(
        input_text, model, vocab, tokenizer, sentiment_labels)
    print(f"Predicted sentiment: {sentiment}")



"""
=====================================================================
---------------------- Settings Section -----------------------------
=====================================================================
"""

if __name__ == "__main__":
    MODEL_VERSION = 'version_1'
    ROOT_PATH = '/home/marco/Repos/learning/sentimaent_sample/lightning_logs'
    INPUT_TEXT = "i am sick"

    main(INPUT_TEXT, MODEL_VERSION, ROOT_PATH)

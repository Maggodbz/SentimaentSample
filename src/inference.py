import os
import torch
import pytorch_lightning as pl
from transformers import BertTokenizer
import pandas as pd

# Define your PyTorch Lightning model class


class SimpleSentimentModel(pl.LightningModule):
    def __init__(self, vocab_size, pad_index, embedding_dim=100, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
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


def predict_sentiment(text, model, vocab, tokenizer):
    tokenized_text = [vocab.get(token, vocab['<unk>']) for token in tokenizer.tokenize(text)]
    input_tensor = torch.tensor([tokenized_text], dtype=torch.long).to(model.device)  # Move input tensor to the same device as model
    with torch.no_grad():
        output = model(input_tensor)
        predicted_sentiment = torch.argmax(output, dim=1).item()
    return predicted_sentiment


# Function to load model from checkpoint


def load_model_from_checkpoint(checkpoint_path, vocab_size, pad_index):
    model = SimpleSentimentModel.load_from_checkpoint(
        checkpoint_path,
        vocab_size=vocab_size,
        pad_index=pad_index
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to appropriate device
    model.eval()
    return model



def load_vocab(vocab_file):
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"The file {vocab_file} does not exist.")
    with open(vocab_file, 'r') as file:
        vocab = {line.strip(): idx for idx, line in enumerate(file.readlines())}
    return vocab


# Load the vocabulary
vocab = load_vocab('vocab.txt')  # Adjust this to your actual vocab file path
vocab_size = len(vocab)
pad_index = vocab['<pad>']

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model from checkpoint
checkpoint_path = "/home/marco/Repos/learning/school_presentation/lightning_logs/version_1/checkpoints/epoch=9-step=8590.ckpt"
model = load_model_from_checkpoint(checkpoint_path, vocab_size, pad_index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move model to appropriate device


# Example usage
input_text = "nothing happend today."
sentiment = predict_sentiment(input_text, model, vocab, tokenizer)
# This will print the index of the sentiment
print(f"Predicted sentiment: {sentiment}")

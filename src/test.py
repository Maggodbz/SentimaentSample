from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import os
import torch
import pytorch_lightning as pl
from transformers import BertTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define your PyTorch Lightning model class


class SentimentDataset(Dataset):
    def __init__(self, dataframe, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self._yield_tokens(
            dataframe['text']), specials=["<unk>", "<pad>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.max_length = max_length
        self.pad_idx = self.vocab['<pad>']

        # Convert sentiment to categorical codes
        self.labels = dataframe['sentiment'].astype('category').cat.codes

    def _yield_tokens(self, data):
        for text in data:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.labels[idx]  # Directly use pre-converted label
        tokenized_text = self.vocab(self.tokenizer(text))
        if len(tokenized_text) > self.max_length:
            tokenized_text = tokenized_text[:self.max_length]
        else:
            tokenized_text += [self.pad_idx] * \
                (self.max_length - len(tokenized_text))
        return torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def get_vocab(self):
        return self.vocab


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


def predict_sentiment(text, model, vocab, tokenizer):
    tokenized_text = [vocab.get(token, vocab['<unk>'])
                      for token in tokenizer.tokenize(text)]
    input_tensor = torch.tensor([tokenized_text], dtype=torch.long).to(
        model.device)  # Move input tensor to the same device as model
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


# Assuming the classes are already defined/imported above as mentioned
# SentimentDataset, SimpleSentimentModel, load_model_from_checkpoint, load_vocab


def evaluate_model(data_path, model, vocab, tokenizer, batch_size=16):
    # Load the data
    df = pd.read_csv(data_path)
    dataset = SentimentDataset(df, max_length=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(model.device)  # Move to the same device as model
            outputs = model(texts)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy


# Load the vocabulary and model as done previously
vocab = load_vocab('vocab.txt')  # Adjust this to your actual vocab file path
vocab_size = len(vocab)
pad_index = vocab['<pad>']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = load_model_from_checkpoint('/home/marco/Repos/learning/school_presentation/lightning_logs/version_1/checkpoints/epoch=9-step=8590.ckpt', vocab_size, pad_index)

# Set the path to your CSV file
csv_path = '/home/marco/Repos/learning/school_presentation/cleaned_test_indexed.csv'

# Evaluate the model
accuracy = evaluate_model(csv_path, model, vocab, tokenizer)
print(f"Model Accuracy: {accuracy}")

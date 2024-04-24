import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score, recall_score, f1_score
import os
from transformers import BertTokenizer


"""
=====================================================================
---------------------- Code Section --------------------------------
=====================================================================
"""


class SentimentDataset(Dataset):
    """Dataset for loading and processing text for sentiment analysis."""
    def __init__(self, dataframe, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self._yield_tokens(dataframe['text']), specials=["<unk>", "<pad>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.max_length = max_length
        self.pad_idx = self.vocab['<pad>']
        self.labels = dataframe['sentiment'].astype('category').cat.codes

    def _yield_tokens(self, data):
        for text in data:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.labels[idx]
        tokenized_text = self.vocab(self.tokenizer(text))
        if len(tokenized_text) > self.max_length:
            tokenized_text = tokenized_text[:self.max_length]
        else:
            tokenized_text += [self.pad_idx] * (self.max_length - len(tokenized_text))
        return torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SimpleSentimentModel(pl.LightningModule):
    """A simple sentiment analysis model with embedding and linear layers."""
    def __init__(self, vocab_size, pad_index, embedding_dim=100, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc1 = torch.nn.Linear(embedding_dim, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        x = x.to(self.device)
        embedded = self.embedding(x).mean(dim=1)
        hidden = torch.nn.functional.relu(self.fc1(embedded))
        return self.fc2(hidden)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return optimizer



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

def evaluate_model(data_path, model, vocab, tokenizer, batch_size=16):
    df = pd.read_csv(data_path)
    dataset = SentimentDataset(df, max_length=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(model.device)
            outputs = model(texts)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    return accuracy, recall, f1


def main(data_path, checkpoint_directory, vocab_file):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = load_vocab(vocab_file)
    vocab_size = len(vocab)
    pad_index = vocab['<pad>']

    results = []

    for checkpoint_path, version in find_checkpoints(checkpoint_directory):
        model = load_model_from_checkpoint(checkpoint_path, vocab_size, pad_index)
        accuracy, recall, f1 = evaluate_model(data_path, model, vocab, tokenizer)
        results.append((version, accuracy, recall, f1))

    print("Model | Accuracy | Recall | F1 Score")
    for version, accuracy, recall, f1 in results:
        print(f"{version} | {accuracy:.2f} | {recall:.2f} | {f1:.2f}")

def load_vocab(vocab_file):
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"The file {vocab_file} does not exist.")
    with open(vocab_file, 'r') as file:
        vocab = {line.strip(): idx for idx, line in enumerate(file.readlines())}
    return vocab

def find_checkpoints(directory):
    for model_dir in os.listdir(directory):
        checkpoint_dir = os.path.join(directory, model_dir, 'checkpoints')
        if os.path.isdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.ckpt'):
                    yield os.path.join(checkpoint_dir, file), model_dir


"""
=====================================================================
---------------------- Settings Section -----------------------------
=====================================================================
"""

if __name__ == "__main__":
    DATA_PATH = '/home/marco/Repos/learning/school_presentation/data/Processed/test.csv'
    CHECKPOINT_DIRECTORY = '/home/marco/Repos/learning/school_presentation/lightning_logs'
    VOCAB_FILE = 'vocab.txt'
    
    main(DATA_PATH, CHECKPOINT_DIRECTORY, VOCAB_FILE)


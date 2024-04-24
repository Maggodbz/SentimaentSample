import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

"""
=====================================================================
---------------------- Code Section --------------------------------
=====================================================================
"""


class SentimentDataset(Dataset):
    """Handles loading and tokenizing text data for sentiment analysis."""

    def __init__(self, dataframe, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self._yield_tokens(
            dataframe['text']), specials=["<unk>", "<pad>"])
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
            tokenized_text += [self.pad_idx] * \
                (self.max_length - len(tokenized_text))
        return torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def get_vocab(self):
        return self.vocab


class SimpleSentimentModel(pl.LightningModule):
    """A simple model for text sentiment classification."""

    def __init__(self, vocab_size, vocab, embedding_dim=100, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.fc1 = torch.nn.Linear(embedding_dim, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        hidden = torch.nn.functional.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)


def save_vocab(vocab, file_path):
    """Saves the vocabulary to a file."""
    with open(file_path, 'w') as file:
        for token, index in vocab.get_stoi().items():
            file.write(f'{token}\n')


def main(data_root_dir, csv_name, batch_size, max_epochs, learning_rate):
    """Main function that setups and runs the training process."""
    train_df = pd.read_csv(data_root_dir + csv_name)
    # Use only the necessary columns
    train_df = train_df[['text', 'sentiment']]

    train_dataset = SentimentDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    vocab = train_dataset.get_vocab()
    model = SimpleSentimentModel(len(vocab), vocab)
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader)

    save_vocab(vocab, 'vocab.txt')


"""
=====================================================================
---------------------- Settings Section -----------------------------
=====================================================================
"""
if __name__ == "__main__":
    # Folder in which the data is stored
    DATA_ROOT_DIR = 'sentimaent_sample/data/Processed/'
    # Name of the CSV file containing the training data
    CSV_NAME = 'train.csv'
    # Decides how many ('text', 'sentiment') pairs are predicted at once
    # before the model is updated
    BATCH_SIZE = 1
    # How many times the model sees the training data
    MAX_EPOCHS = 1
    # Determines how fast the model learns from the data
    LEARNING_RATE = 0.002

    main(DATA_ROOT_DIR, CSV_NAME, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE)

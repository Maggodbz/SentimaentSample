import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Load and preprocess data
class SentimentDataset(Dataset):
    def __init__(self, dataframe, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self._yield_tokens(dataframe['text']), specials=["<unk>", "<pad>"])
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
            tokenized_text += [self.pad_idx] * (self.max_length - len(tokenized_text))
        return torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def get_vocab(self):
        return self.vocab

# PyTorch Lightning Module
class SimpleSentimentModel(pl.LightningModule):
    def __init__(self, vocab_size, vocab, embedding_dim=100, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.fc1 = torch.nn.Linear(embedding_dim, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)  # Average embedding vector
        hidden = torch.nn.functional.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Load CSV
data_root_dir = 'data/Processed/'
csv_name = 'train.csv'
train_df = pd.read_csv(data_root_dir + csv_name)
train_df = train_df[['text', 'sentiment']]  # Ensure to use only the necessary columns

# Setup data
train_dataset = SentimentDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model training
vocab = train_dataset.get_vocab()
model = SimpleSentimentModel(len(vocab), vocab)
trainer = Trainer(max_epochs=10)
# trainer.fit(model, train_loader)

# Saving vocabulary to a text file
def save_vocab(vocab, file_path):
    with open(file_path, 'w') as file:
        for token, index in vocab.get_stoi().items():
            file.write(f'{token}\n')

save_vocab(vocab, 'vocab.txt')
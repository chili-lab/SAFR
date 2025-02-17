import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.datasets import SentimentDataset
from data.read import read_csv_data
from model.model import CustomTransformer

class SentimentAnalyzer:
    def __init__(self, file_path, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.file_path = file_path
        self.device = device
        self.config = config

        # Set max_len based on the selected dataset
        if self.config.get("dataset") == "imdb":
            self.max_len = 256
        elif self.config.get("dataset") == "sst2":
            self.max_len = 128
        else:
            self.max_len = 128

        # Default configurations
        self.batch_size = 32
        self.n_classes = 2
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 1
        self.d_ff = 1024
        self.dropout = 0.1
        self.mask_hidden_dim = 1024
        self.lr = 0.0001
        self.num_epochs = 100
        self.patience = 3
        self.importance_lambda = self.config.get("importance_lambda", 1)
        self.interaction_lambda = self.config.get("interaction_lambda", 1)
        self.model_save_path = "model.pth"
        self.vocab = None

    def build_vocab(self, sentences, min_freq=2):
        word_freq = {}
        for sentence in sentences:
            for word in sentence.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        return vocab

    def run(self):
        print(self.file_path)
        x_train, y_train, x_dev, y_dev, x_test, y_test = read_csv_data(self.file_path, self.config.get("dataset"))
        train_loader, dev_loader, test_loader = self.create_data_loaders(x_train, y_train, x_dev, y_dev, x_test, y_test)
        model = self.train_model(train_loader, dev_loader, test_loader)
        return model

    def create_data_loaders(self, x_train, y_train, x_dev, y_dev, x_test, y_test):
        self.vocab = self.build_vocab(x_train)
        train_dataset = SentimentDataset(x_train, y_train, self.vocab, self.max_len)
        dev_dataset   = SentimentDataset(x_dev, y_dev, self.vocab, self.max_len)
        test_dataset  = SentimentDataset(x_test, y_test, self.vocab, self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_loader   = DataLoader(dev_dataset, batch_size=self.batch_size)
        test_loader  = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model = CustomTransformer(
            vocab_size=len(self.vocab),
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_length=self.max_len,
            num_classes=self.n_classes,
            dropout=self.dropout,
            mask_hidden_dim=self.mask_hidden_dim
        ).to(self.device)
        self.model.set_reverse_vocab(self.vocab)
        return train_loader, dev_loader, test_loader

    def train_model(self, train_loader, dev_loader, test_loader):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(self.device)
        best_dev_accuracy = 0.0
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_ce_loss = 0
            train_importance_loss = 0
            train_interaction_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs, flag='train')
                ce_loss = criterion(outputs, labels)
                importance_loss = model.polyse_loss
                interaction_loss = model.inter_loss
                loss = ce_loss + self.importance_lambda * importance_loss + self.interaction_lambda * interaction_loss
                loss.backward()
                optimizer.step()

                train_ce_loss += ce_loss.item()
                train_importance_loss += importance_loss.item()
                train_interaction_loss += interaction_loss.item()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = train_correct / train_total
            dev_accuracy = self.evaluate(model, dev_loader)
            
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        test_accuracy = self.evaluate(model, test_loader)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        torch.save(model.state_dict(), self.model_save_path)
        return model

    def evaluate(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs, flag='eval')
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_len):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        indices = [self.vocab.get(word, self.vocab.get('<UNK>')) for word in sentence.split()]
        if len(indices) < self.max_len:
            indices += [self.vocab.get('<PAD>')] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices), torch.tensor(label)

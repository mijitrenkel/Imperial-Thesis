import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EPS_PRICE = 1e-8
FEATURE_COLS = ["log_moneyness", "inst_vol", "tau_norm", "strike_frac", "r"]

class PathDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        blobs = []
        for _, sub in df.groupby("path_id", sort=False):
            sub = sub.sort_values("step")
            X = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
            delta = sub["delta"].to_numpy(dtype=np.float32)
            blobs.append((X, delta))
        self.blobs = blobs

    def __len__(self):  return len(self.blobs)
    def __getitem__(self, i): return self.blobs[i]

def collate_batch(batch):
    lengths = [len(x) for x, _ in batch]
    order= np.argsort(-np.array(lengths))
    batch = [batch[i] for i in order]
    lengths = [lengths[i] for i in order]

    Xs, Ds = zip(*batch)
    X_pad = nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x in Xs], batch_first=True)
    D_pad= nn.utils.rnn.pad_sequence([torch.from_numpy(d) for d in Ds], batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return X_pad, D_pad.unsqueeze(-1), lengths

class LSTMMultiTask(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=n_layers,batch_first=True, dropout=dropout)
        self.head_delta = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(),batch_first=True, enforce_sorted=True)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        delta_hat = self.head_delta(out)
        return delta_hat
def _masked_mse(pred, true, lengths):
    B, T, _ = pred.shape
    mask = (torch.arange(T, device=pred.device)[None, :] < lengths[:, None]).float()
    mask = mask.unsqueeze(-1)
    return ((pred - true) ** 2 * mask).sum() / mask.sum()
def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for X, D, L in loader:
        X, D, L = X.to(device), D.to(device), L.to(device)
        opt.zero_grad()
        D_hat = model(X, L)
        loss = _masked_mse(D_hat, D, L)
        loss.backward(); opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for X, D, L in loader:
        X, D, L = X.to(device), D.to(device), L.to(device)
        D_hat = model(X, L)
        total += _masked_mse(D_hat, D, L).item() * X.size(0)
    return total / len(loader.dataset)
__all__ = ["FEATURE_COLS", "PathDataset", "LSTMMultiTask",
           "train_epoch", "eval_epoch", "collate_batch"]


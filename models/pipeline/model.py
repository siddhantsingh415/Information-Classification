"""
The Graph Attention Network classifier, plus training / fine-tuning /
evaluation / checkpointing utilities.

Edit this file to change: the model architecture (`FakeNewsGAT`), the
cold-start training loop (`train_gat`), or the warm-start fine-tuning loop
used by the batched training loop (`finetune_gat`).
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from .config import FEATURE_DIM


class FakeNewsGAT(torch.nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, n_heads=4, dropout=0.4):
        super().__init__()
        self.dropout    = dropout
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        self.conv1 = GATv2Conv(hidden_dim,     hidden_dim,     heads=n_heads, concat=True,  dropout=dropout)
        self.lin1  = torch.nn.Linear(hidden_dim * n_heads, hidden_dim)
        self.bn1   = torch.nn.BatchNorm1d(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim,     hidden_dim * 2, heads=n_heads, concat=True,  dropout=dropout)
        self.lin2  = torch.nn.Linear(hidden_dim * 2 * n_heads, hidden_dim * 2)
        self.bn2   = torch.nn.BatchNorm1d(hidden_dim * 2)

        self.conv3 = GATv2Conv(hidden_dim * 2, hidden_dim * 2, heads=n_heads, concat=True,  dropout=dropout)
        self.lin3  = torch.nn.Linear(hidden_dim * 2 * n_heads, hidden_dim * 2)
        self.bn3   = torch.nn.BatchNorm1d(hidden_dim * 2)

        self.conv4 = GATv2Conv(hidden_dim * 2, hidden_dim,     heads=n_heads, concat=False, dropout=dropout)
        self.bn4   = torch.nn.BatchNorm1d(hidden_dim)

        pool_dim = (hidden_dim + hidden_dim*2 + hidden_dim*2 + hidden_dim) * 2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, 256), torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 64),  torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        x    = F.relu(self.input_proj(x))
        x    = F.dropout(x, p=self.dropout, training=self.training)
        x1   = self.bn1(F.relu(self.lin1(self.conv1(x,  edge_index)))) + x   # residual: hidden → hidden
        x2   = self.bn2(F.relu(self.lin2(self.conv2(x1, edge_index))))        # no residual: dim doubles
        x3   = self.bn3(F.relu(self.lin3(self.conv3(x2, edge_index)))) + x2  # residual: hidden*2 → hidden*2
        x4   = self.bn4(F.relu(self.conv4(x3, edge_index))) + x3[:, :x1.shape[1]]  # residual: hidden*2 → hidden
        xjk  = torch.cat([x1, x2, x3, x4], dim=1)
        xp   = torch.cat([global_mean_pool(xjk, batch), global_max_pool(xjk, batch)], dim=1)
        return self.mlp(xp)


def train_gat(train_data, val_data, epochs=150, lr=5e-4, patience=20):
    model     = FakeNewsGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    t_loader  = DataLoader(train_data, batch_size=16, shuffle=True)
    v_loader  = DataLoader(val_data,   batch_size=16)

    best_val, no_improve, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for b in t_loader:
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.batch).squeeze()
            loss = criterion(out, b.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for b in v_loader:
                v_loss += criterion(
                    model(b.x, b.edge_index, b.batch).squeeze(), b.y.squeeze()
                ).item()

        avg_t, avg_v = t_loss / len(t_loader), v_loss / len(v_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}  train={avg_t:.4f}  val={avg_v:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}')

        if avg_v < best_val:
            best_val, no_improve = avg_v, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(best_state)
    return model


def finetune_gat(
    model,
    train_data,
    val_data,
    epochs:     int   = 30,
    lr:         float = 1e-4,
    patience:   int   = 8,
    batch_size: int   = 16,
):
    """
    Fine-tune an existing FakeNewsGAT on new data (warm start).

    Differs from train_gat in three ways: no re-initialisation (continues
    from the model's current weights), a lower default learning rate, and it
    returns the loss dict alongside the model instead of just printing it.

    Returns the updated model and a loss dict {'train_loss': ..., 'val_loss': ...}.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    t_loader  = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    v_loader  = DataLoader(val_data,   batch_size=batch_size)

    best_val, no_improve, best_state = float('inf'), 0, None
    avg_t = avg_v = 0.0

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for b in t_loader:
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.batch).squeeze()
            loss = criterion(out, b.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for b in v_loader:
                v_loss += criterion(
                    model(b.x, b.edge_index, b.batch).squeeze(),
                    b.y.squeeze()
                ).item()

        avg_t = t_loss / max(len(t_loader), 1)
        avg_v = v_loss / max(len(v_loader), 1)

        if avg_v < best_val:
            best_val, no_improve = avg_v, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, {'train_loss': avg_t, 'val_loss': best_val}


def evaluate_model(model, dataset):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in DataLoader(dataset, batch_size=16):
            probs = torch.sigmoid(model(b.x, b.edge_index, b.batch).squeeze())
            preds.extend(probs.tolist())
            labels.extend(b.y.squeeze().tolist())
    binary = [1 if p > 0.5 else 0 for p in preds]
    return {
        'accuracy': accuracy_score(labels, binary),
        'f1':       f1_score(labels, binary, zero_division=0),
        'auc':      roc_auc_score(labels, preds)
    }, preds


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Saved to {path}')


def load_model(path):
    m = FakeNewsGAT()
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m

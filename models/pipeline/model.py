"""
The Graph Attention Network classifier, plus training / fine-tuning /
evaluation / checkpointing utilities.

Edit this file to change: the model architecture (`FakeNewsGAT`), the
cold-start training loop (`train_gat`), or the warm-start fine-tuning loop
used by the batched training loop (`finetune_gat`).

Patch history:
  v2: `BCEWithLogitsLoss` weighted by `pos_weight = n_negative / n_positive`
      to counter AVeriTeC's ~2.6:1 class imbalance (was causing an
      always-predict-majority-class collapse).
  v3: two follow-up fixes, made after v2's full pos_weight combined with
      widespread evidence-retrieval failures (many training graphs reduced
      to a single, nearly-featureless claim node) to produce the mirror-image
      failure -- the model converged to always predicting the *minority*
      class, since with little real signal to learn from, that was the
      loss-minimizing strategy under a strong pos_weight, and early stopping
      couldn't tell the difference because it picked checkpoints by
      validation *loss*, not by how well the model actually discriminated:
        1. `pos_weight` is now damped: `ratio ** pos_weight_damping`
           (default damping=0.5, i.e. sqrt of the full inverse-frequency
           ratio) rather than the full ratio, so it corrects for imbalance
           without being strong enough to make a majority-negative
           prediction pattern this costly to avoid.
        2. Checkpoint selection in both `train_gat` and `finetune_gat` now
           tracks validation F1 (higher is better) instead of validation
           loss (lower is better) -- a weighted loss can be "minimized" by a
           degenerate constant-output solution that a discrimination metric
           like F1 would immediately expose.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

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


def _pos_weight_from(data, damping: float = 0.5) -> torch.Tensor:
    """
    BCEWithLogitsLoss pos_weight, computed from whatever dataset is passed
    in. `damping` interpolates between no correction (0.0) and the full
    inverse-frequency ratio n_negative/n_positive (1.0) via `ratio **
    damping` -- default 0.5 is the square root of the full ratio. See the
    v3 note in the module docstring for why the full ratio overcorrected.
    """
    labels = torch.cat([d.y for d in data]).squeeze()
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0:
        return torch.tensor([1.0])
    ratio = n_neg / n_pos
    return torch.tensor([ratio ** damping])


def _eval_epoch(model, loader, criterion):
    """Runs one pass over `loader` in eval mode, returning (avg_loss, f1)."""
    model.eval()
    v_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for b in loader:
            out = model(b.x, b.edge_index, b.batch).view(-1)
            y   = b.y.view(-1)
            v_loss += criterion(out, y).item()
            preds.extend((torch.sigmoid(out) > 0.5).long().tolist())
            labels.extend(y.long().tolist())
    avg_loss = v_loss / max(len(loader), 1)
    val_f1   = f1_score(labels, preds, zero_division=0)
    return avg_loss, val_f1


def train_gat(train_data, val_data, epochs=150, lr=5e-4, patience=20, pos_weight_damping=0.5):
    model      = FakeNewsGAT()
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_weight = _pos_weight_from(train_data, damping=pos_weight_damping)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    t_loader   = DataLoader(train_data, batch_size=16, shuffle=True)
    v_loader   = DataLoader(val_data,   batch_size=16)

    best_f1, no_improve, best_state = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for b in t_loader:
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.batch).view(-1)
            loss = criterion(out, b.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        avg_v, val_f1 = _eval_epoch(model, v_loader, criterion)
        avg_t = t_loss / len(t_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:3d}  train_loss={avg_t:.4f}  val_loss={avg_v:.4f}  '
                  f'val_f1={val_f1:.3f}  lr={scheduler.get_last_lr()[0]:.2e}')

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} (best val_f1={best_f1:.3f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune_gat(
    model,
    train_data,
    val_data,
    epochs:             int   = 30,
    lr:                 float = 1e-4,
    patience:           int   = 8,
    batch_size:         int   = 16,
    pos_weight_damping: float = 0.5,
):
    """
    Fine-tune an existing FakeNewsGAT on new data (warm start).

    Differs from train_gat in three ways: no re-initialisation (continues
    from the model's current weights), a lower default learning rate, and it
    returns the loss/F1 dict alongside the model instead of just printing it.

    Returns the updated model and a dict {'train_loss', 'val_loss', 'val_f1'}
    from the best (highest val_f1) checkpoint found.
    """
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_weight = _pos_weight_from(train_data, damping=pos_weight_damping)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    t_loader   = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    v_loader   = DataLoader(val_data,   batch_size=batch_size)

    best_f1, no_improve, best_state = -1.0, 0, None
    avg_t, best_val_loss = 0.0, float('inf')

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for b in t_loader:
            optimizer.zero_grad()
            out  = model(b.x, b.edge_index, b.batch).view(-1)
            loss = criterion(out, b.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        avg_v, val_f1 = _eval_epoch(model, v_loader, criterion)
        avg_t = t_loss / max(len(t_loader), 1)

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_loss = avg_v
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {'train_loss': avg_t, 'val_loss': best_val_loss, 'val_f1': best_f1}


def evaluate_model(model, dataset):
    from sklearn.metrics import accuracy_score, roc_auc_score
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in DataLoader(dataset, batch_size=16):
            probs = torch.sigmoid(model(b.x, b.edge_index, b.batch).view(-1))
            preds.extend(probs.tolist())
            labels.extend(b.y.view(-1).tolist())
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

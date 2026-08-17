"""
Batched build-and-train loop: interleaves graph construction
(dataset_builder.build_batch) with cold-start / warm-start model training
(model.train_gat / model.finetune_gat), evaluates against a fixed global
validation pool after each batch, and updates the source-credibility DB
(credibility.bulk_update_from_prediction) as it goes.

This is the "models4" streaming alternative to the classic
build_dataset -> train_gat -> evaluate_model flow: training starts after the
first batch instead of waiting for the whole dataset to be built.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as _tts
import torch
from torch_geometric.loader import DataLoader as _DL

from .model import train_gat, finetune_gat, evaluate_model, save_model
from .dataset_builder import build_batch, retry_failed_evidence
from .credibility import bulk_update_from_prediction


def batched_train_loop(
    df,
    batch_size:    int   = 50,
    val_pool_size: int   = 60,
    cold_epochs:   int   = 80,
    warm_epochs:   int   = 30,
    val_split:     float = 0.2,
    retry:         bool  = True,
    use_nli:       bool  = True,
    fetch_full_text: bool = True,
    save_prefix:   str   = 'fake_news_gat_v4',
    text_col:      str   = 'text',
    title_col:     str   = 'title',
    label_col:     str   = 'label_binary',
):
    """
    Iterative build-and-train loop with per-batch metric tracking.

    fetch_full_text controls whether each retrieved doc's full article page
    is scraped (see retrieval.py) — set False to skip those extra network
    round-trips for a faster run at the cost of thinner, snippet-only
    evidence.

    Returns the final model and batch_history (list of per-batch metric dicts).
    """
    # ── 1. Reserve a fixed global validation pool ─────────────────────────────
    pool_frac = val_pool_size / len(df)
    pool_df, work_df = _tts(df, test_size=(1 - pool_frac),
                            stratify=df[label_col], random_state=42)
    pool_df = pool_df.iloc[:val_pool_size].reset_index(drop=True)
    work_df = work_df.reset_index(drop=True)

    n_batches = (len(work_df) + batch_size - 1) // batch_size
    print(f'Global val pool : {len(pool_df)} articles')
    print(f'Working set     : {len(work_df)} articles -> ~{n_batches} batches of {batch_size}')

    # Build val-pool graphs once upfront (they never change)
    print('\n[init] Building global validation pool...')
    val_pool_pyg, _, val_pool_fails, _ = build_batch(
        pool_df, text_col, title_col, label_col,
        use_nli=use_nli, fetch_full_text=fetch_full_text
    )
    if retry and val_pool_fails:
        extra, _, _ = retry_failed_evidence(val_pool_fails, use_nli=use_nli,
                                            fetch_full_text=fetch_full_text)
        val_pool_pyg.extend(extra)
    print(f'[init] Val pool ready: {len(val_pool_pyg)} graphs\n')

    # ── 2. Batch loop ─────────────────────────────────────────────────────────
    model         = None
    batch_history = []
    all_pyg       = []
    all_scored    = []

    for b_idx in range(n_batches):
        start    = b_idx * batch_size
        end      = min(start + batch_size, len(work_df))
        batch_df = work_df.iloc[start:end]

        print('=' * 60)
        print(f'BATCH {b_idx+1}/{n_batches}  (rows {start}-{end-1}, n={len(batch_df)})')
        print('=' * 60)

        # 2a. Build graphs for this batch
        b_pyg, b_scored, b_failed, _ = build_batch(
            batch_df, text_col, title_col, label_col,
            use_nli=use_nli, fetch_full_text=fetch_full_text
        )

        # 2b. Retry failed evidence
        if retry and b_failed:
            print(f'  Retrying {len(b_failed)} failed articles...')
            r_pyg, r_scored, _ = retry_failed_evidence(b_failed, use_nli=use_nli,
                                                        fetch_full_text=fetch_full_text)
            b_pyg.extend(r_pyg)
            b_scored.extend(r_scored)

        all_pyg.extend(b_pyg)
        all_scored.extend(b_scored)

        if len(b_pyg) < 4:
            print(f'  Batch too small ({len(b_pyg)} graphs) — skipping training step.')
            continue

        # 2c. Batch-local train / val split (for early stopping only)
        n_val   = max(1, int(len(b_pyg) * val_split))
        b_val   = b_pyg[:n_val]
        b_train = b_pyg[n_val:]
        if not b_train:
            b_train = b_val   # edge case: tiny batch

        # 2d. Train or fine-tune
        if model is None:
            print(f'  Cold start: {cold_epochs} epochs...')
            model = train_gat(b_train, b_val, epochs=cold_epochs, patience=15)
        else:
            print(f'  Warm start: {warm_epochs} epochs (lr=1e-4)...')
            model, ft = finetune_gat(model, b_train, b_val,
                                     epochs=warm_epochs, patience=8)
            print(f'  Fine-tune: train_loss={ft["train_loss"]:.4f}  '
                  f'val_loss={ft["val_loss"]:.4f}')

        # 2e. Evaluate on the fixed global val pool
        if val_pool_pyg:
            metrics, _ = evaluate_model(model, val_pool_pyg)
            print(f'  [global val] acc={metrics["accuracy"]:.3f}  '
                  f'f1={metrics["f1"]:.3f}  auc={metrics["auc"]:.3f}')
        else:
            metrics = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.0}

        batch_history.append({
            'batch':        b_idx + 1,
            'n_graphs':     len(b_pyg),
            'total_graphs': len(all_pyg),
            **metrics,
        })

        # 2f. Update credibility DB with this batch's predictions
        n_db = 0
        model.eval()
        with torch.no_grad():
            for graph, scored in zip(b_pyg, b_scored):
                if not scored:
                    continue
                for btch in _DL([graph], batch_size=1):
                    prob = torch.sigmoid(
                        model(btch.x, btch.edge_index, btch.batch).squeeze()
                    ).item()
                    conf = abs(prob - 0.5) * 2.0
                    n_db += bulk_update_from_prediction(
                        scored, prob, conf, confidence_threshold=0.1
                    )
        print(f'  Credibility DB: {n_db} entries updated.')

        # 2g. Save checkpoint
        ckpt = f'{save_prefix}_batch{b_idx+1:03d}.pt'
        save_model(model, ckpt)

    # ── 3. Plot learning curve ────────────────────────────────────────────────
    if batch_history:
        _plot_batch_history(batch_history, save_prefix)

    print('\n[done] Batched training complete.')
    return model, batch_history


def _plot_batch_history(history, prefix):
    """Save a 2-panel learning-curve figure: metrics + dataset growth."""
    batches = [h['batch']        for h in history]
    accs    = [h['accuracy']     for h in history]
    f1s     = [h['f1']           for h in history]
    aucs    = [h['auc']          for h in history]
    totals  = [h['total_graphs'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(batches, accs, marker='o', label='Accuracy')
    ax1.plot(batches, f1s,  marker='s', label='F1')
    ax1.plot(batches, aucs, marker='^', label='AUC-ROC')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Score (global val pool)')
    ax1.set_title('Model Performance per Batch')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(batches, totals, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Cumulative graphs built')
    ax2.set_title('Dataset Growth per Batch')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = f'{prefix}_learning_curve.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[plot] Learning curve saved -> {out}')


# ── Recommended config for a 500-article sample ──────────────────────────────
#
#   from pipeline.training_loop import batched_train_loop
#
#   final_model, history = batched_train_loop(
#       df,
#       batch_size      = 50,
#       val_pool_size   = 60,
#       cold_epochs     = 80,
#       warm_epochs     = 30,
#       retry           = True,
#       fetch_full_text = True,   # set False for a quicker, snippet-only run
#   )

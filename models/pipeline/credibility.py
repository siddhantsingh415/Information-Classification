"""
Per-domain source credibility database (SQLite), backed by a Beta(alpha, beta)
distribution per domain. Each domain starts at Beta(2, 2) (score = 0.5 — no
information) and gets nudged by two kinds of signals:

  - 'model' signals: whenever the GAT classifies an article with high
    confidence, every domain that supplied evidence for that article gets a
    small nudge (MODEL_WEIGHT).
  - 'user' signals: explicit user feedback gets a full-strength nudge
    (USER_WEIGHT). Nothing in this pipeline emits 'user' signals yet — that's
    a hook for a future feedback UI.
"""
import sqlite3
from contextlib import contextmanager
from urllib.parse import urlparse
from datetime import datetime, timezone

DB_PATH      = 'source_credibility_v3.db'
MODEL_WEIGHT = 0.3
USER_WEIGHT  = 1.0


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                domain        TEXT PRIMARY KEY,
                alpha         REAL DEFAULT 2.0,
                beta          REAL DEFAULT 2.0,
                model_updates INTEGER DEFAULT 0,
                user_updates  INTEGER DEFAULT 0,
                last_updated  TEXT
            )''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS credibility_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                domain      TEXT,
                signal_type TEXT,
                signal      TEXT,
                confidence  REAL,
                timestamp   TEXT
            )''')


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.replace('www.', '')


def get_credibility(domain: str) -> float:
    with get_db() as conn:
        row = conn.execute(
            'SELECT alpha, beta FROM sources WHERE domain = ?', (domain,)
        ).fetchone()
    return (row[0] / (row[0] + row[1])) if row else 0.5


def update_credibility(domain: str, signal: str, signal_type: str, confidence: float = 1.0):
    assert signal in ('real', 'fake') and signal_type in ('model', 'user')
    weight = (MODEL_WEIGHT if signal_type == 'model' else USER_WEIGHT) * confidence
    now    = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            'INSERT OR IGNORE INTO sources (domain,alpha,beta,last_updated) VALUES(?,2.0,2.0,?)',
            (domain, now)
        )
        col        = 'alpha' if signal == 'real' else 'beta'
        update_col = 'model_updates' if signal_type == 'model' else 'user_updates'
        conn.execute(
            f'UPDATE sources SET {col}={col}+?,{update_col}={update_col}+1,last_updated=? WHERE domain=?',
            (weight, now, domain)
        )
        conn.execute(
            'INSERT INTO credibility_log(domain,signal_type,signal,confidence,timestamp) VALUES(?,?,?,?,?)',
            (domain, signal_type, signal, confidence, now)
        )


def bulk_update_from_prediction(scored_docs, prediction: float,
                                 model_confidence: float,
                                 confidence_threshold: float = 0.1) -> int:
    if model_confidence < confidence_threshold:
        return 0
    signal = 'fake' if prediction > 0.5 else 'real'
    n = 0
    for doc, doc_score in scored_docs:
        url = doc.get('link', '')
        if url:
            update_credibility(extract_domain(url), signal, 'model',
                               confidence=model_confidence * doc_score)
            n += 1
    return n


def print_credibility_leaderboard(limit: int = 25):
    """Print the most-updated domains, ranked by total signal count, with
    their current credibility score."""
    with get_db() as conn:
        rows = conn.execute(
            'SELECT domain, alpha, beta, model_updates, user_updates '
            'FROM sources ORDER BY model_updates + user_updates DESC LIMIT ?',
            (limit,)
        ).fetchall()

    print(f'{"Domain":<40} {"Score":>6} {"Signals":>8} {"Model":>7} {"User":>6}')
    print('-' * 70)
    for domain, alpha, beta, m, u in rows:
        score = alpha / (alpha + beta)
        n     = int(alpha + beta - 4)
        print(f'{domain:<40} {score:>6.3f} {n:>8} {m:>7} {u:>6}')


# Create the DB file/tables as soon as this module is imported, so every
# other module can assume the DB is ready.
init_db()

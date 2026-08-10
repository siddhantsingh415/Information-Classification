"""
Zero-shot sentence-role classification (claim / evidence / analysis / background)
via NLI entailment. Every sentence added to the argumentation graph — from the
article itself or from retrieved evidence — gets one of these four role labels.
"""
import numpy as np

from .config import NLI_MODEL, NLI_ENTAILMENT, ROLE_NAMES

# Defining hypotheses for each role — written to activate the NLI model's
# entailment signal when the premise (sentence) matches the role description.
ROLE_HYPOTHESES = [
    "This sentence makes a specific factual assertion or claim.",         # ROLE_CLAIM
    "This sentence presents data, quotes, or cited supporting evidence.",  # ROLE_EVIDENCE
    "This sentence provides interpretation, opinion, or analysis.",       # ROLE_ANALYSIS
    "This sentence provides background context or general information.",  # ROLE_BACKGROUND
]


def classify_sentence_roles(sentences: list[str]) -> list[int]:
    """
    Classify each sentence into one of 4 roles using zero-shot NLI.

    For each sentence we run NLI against all 4 hypotheses in a single
    batch call. The hypothesis with the highest entailment score wins.

    Batching all sentences x all hypotheses in one predict() call is
    much faster than calling predict() once per sentence.
    """
    if not sentences:
        return []

    # Build all (sentence, hypothesis) pairs
    pairs = [
        (sent, hyp)
        for sent in sentences
        for hyp in ROLE_HYPOTHESES
    ]

    # Batch NLI inference
    logits    = NLI_MODEL.predict(pairs)                          # (n_sent * 4, 3)
    probs     = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ent_probs = probs[:, NLI_ENTAILMENT]                          # entailment probability only

    # Reshape to (n_sentences, 4) and argmax per sentence
    ent_matrix = ent_probs.reshape(len(sentences), len(ROLE_HYPOTHESES))
    roles      = np.argmax(ent_matrix, axis=1).tolist()
    return roles


if __name__ == '__main__':
    # Quick sanity check: `python -m pipeline.sentence_roles`
    test_sents = [
        "President Trump signed an executive order on Friday.",
        "According to documents obtained by the Times, the memo was dated March 3.",
        "This represents a fundamental shift in American foreign policy.",
        "The conflict between the two nations began in 1948."
    ]
    roles = classify_sentence_roles(test_sents)
    for s, r in zip(test_sents, roles):
        print(f'[{ROLE_NAMES[r]:<12}] {s}')

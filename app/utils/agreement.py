"""
Inter-annotator agreement utilities.
- cohen_kappa: 2-annotator agreement
- fleiss_kappa: multi-annotator agreement
"""
from collections import Counter
from typing import Sequence


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Compute Cohen's Kappa for two annotators."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0

    n = len(labels_a)
    categories = list(set(labels_a) | set(labels_b))
    # observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # expected agreement
    pe = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """
    Compute Fleiss' Kappa for multiple annotators.
    ratings: list of items, each item is a list of labels from each annotator.
    """
    if not ratings:
        return 0.0

    n_items = len(ratings)
    all_labels = list(set(label for row in ratings for label in row))
    k = len(all_labels)
    n_annotators = len(ratings[0])

    if n_annotators < 2 or k < 2:
        return 1.0

    # Build count matrix
    counts = [[row.count(label) for label in all_labels] for row in ratings]

    # P_i for each subject
    p_i = []
    for row in counts:
        total = sum(row)
        if total <= 1:
            p_i.append(1.0)
        else:
            p_i.append((sum(c * (c - 1) for c in row)) / (total * (total - 1)))

    p_bar = sum(p_i) / n_items

    # p_j: proportion of all assignments to category j
    total_ratings = n_items * n_annotators
    p_j = [sum(row[j] for row in counts) / total_ratings for j in range(k)]

    p_e = sum(pj ** 2 for pj in p_j)

    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def majority_label(labels: Sequence[str]) -> tuple[str | None, float]:
    """Return (majority_label, proportion)."""
    if not labels:
        return None, 0.0
    counter = Counter(labels)
    top_label, top_count = counter.most_common(1)[0]
    return top_label, top_count / len(labels)

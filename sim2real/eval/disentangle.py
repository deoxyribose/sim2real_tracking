"""Latent-disentanglement probes.

- `silhouette_zwhat`: silhouette score on z_what when clustered by GT slot identity.
- `linear_probe_zwhere_from_zwhat`: a sanity test — try to predict GT z_where from predicted
  z_what. Should be POOR if the latents are decomposed correctly (z_what carries appearance,
  z_where carries geometry).
"""

from __future__ import annotations

import numpy as np


def silhouette_zwhat(pred_zwhat, gt_pres, gt_ids):
    """`pred_zwhat`: (T, N, Zw); `gt_pres`: (T, N); `gt_ids`: (T, N) integer labels.

    Aggregates ALL alive frames into one set and computes a single silhouette score against the
    GT identity labels. This is necessary because per-frame slot counts are typically small
    (silhouette needs n_samples > n_labels).
    """
    from sklearn.metrics import silhouette_score

    feats_all = []
    ids_all = []
    for t in range(pred_zwhat.shape[0]):
        pres = np.asarray(gt_pres[t]).astype(bool)
        if not pres.any():
            continue
        feats_all.append(np.asarray(pred_zwhat[t])[pres])
        ids_all.append(np.asarray(gt_ids[t])[pres])
    if not feats_all:
        return float("nan")
    feats = np.concatenate(feats_all, axis=0)
    ids = np.concatenate(ids_all, axis=0)
    n_labels = len(set(ids.tolist()))
    if feats.shape[0] < 2 or n_labels < 2 or n_labels >= feats.shape[0]:
        return float("nan")
    return float(silhouette_score(feats, ids))


def linear_probe_zwhere_from_zwhat(pred_zwhat, gt_zwhere, gt_pres):
    """Linear regression z_what → z_where, returns R² on a held-out split.

    Lower R² is better for decomposition (we want z_what to NOT encode z_where).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    feats = np.asarray(pred_zwhat).reshape(-1, pred_zwhat.shape[-1])
    targets = np.asarray(gt_zwhere).reshape(-1, gt_zwhere.shape[-1])
    pres = np.asarray(gt_pres).reshape(-1).astype(bool)
    feats = feats[pres]
    targets = targets[pres]
    if feats.shape[0] < 20:
        return float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(feats, targets, test_size=0.3, random_state=0)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    return float(model.score(X_te, y_te))

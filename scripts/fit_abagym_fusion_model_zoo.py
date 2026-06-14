#!/usr/bin/env python3
"""Leave-family-out fusion model zoo for AbAgym AIRank features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


def parse_feature(spec: str) -> tuple[str, str, str]:
    name, rest = spec.split("=", 1)
    path, column = rest.rsplit(":", 1)
    return name, path, column


def second_order_numpy(x: np.ndarray) -> np.ndarray:
    parts = [x, x * x]
    interactions = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            interactions.append((x[:, i] * x[:, j])[:, None])
    if interactions:
        parts.append(np.concatenate(interactions, axis=1))
    return np.concatenate(parts, axis=1)


def make_model(name: str, monotonic: list[int]):
    alphas = np.logspace(-4, 4, 17)
    if name == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if name == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75],
                alphas=np.logspace(-4, 1, 20),
                max_iter=20000,
                cv=5,
                random_state=0,
            ),
        )
    if name == "spline_ridge":
        return make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="continue"),
            RidgeCV(alphas=alphas),
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=12,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        )
    if name == "monotonic_hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=12,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            monotonic_cst=monotonic,
            random_state=0,
        )
    raise ValueError(f"unknown model {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--feature", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", default="rank_target", choices=["DMS_score", "rank_target"])
    ap.add_argument(
        "--models",
        default="ridge,elasticnet,spline_ridge,hgb,monotonic_hgb,isotonic_ridge",
        help="Comma-separated model names.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["sample_id", "DMS_name", "DMS_on", "DMS_score", "rank_target", "holdout_group"]
    df = pd.read_csv(args.base, low_memory=False)[keep].copy()
    feature_cols = []
    for spec in args.feature:
        name, path, column = parse_feature(spec)
        feat = pd.read_csv(path, low_memory=False)[["sample_id", column]].rename(columns={column: name})
        df = df.merge(feat, on="sample_id", validate="one_to_one")
        feature_cols.append(name)
    df = df.dropna(subset=feature_cols + [args.target]).copy()

    # All four branch scores are oriented so larger values should mean higher mutation priority.
    monotonic = [1 for _ in feature_cols]
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    summaries = {}
    for model_name in requested:
        pred_df = df.copy()
        pred_df["prediction"] = np.nan
        fold_summaries = []
        for holdout in sorted(df["holdout_group"].astype(str).unique()):
            train = df["holdout_group"].astype(str) != holdout
            test = ~train
            x_train = df.loc[train, feature_cols].to_numpy(dtype=np.float32)
            y_train = df.loc[train, args.target].to_numpy(dtype=np.float32)
            x_test = df.loc[test, feature_cols].to_numpy(dtype=np.float32)

            if model_name == "isotonic_ridge":
                ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 17)))
                ridge.fit(x_train, y_train)
                train_score = ridge.predict(x_train)
                test_score = ridge.predict(x_test)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(train_score, y_train)
                pred = iso.predict(test_score)
                fold_summaries.append({"holdout_group": holdout, "ridge_alpha": float(ridge.named_steps["ridgecv"].alpha_)})
            else:
                model = make_model(model_name, monotonic)
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                payload = {"holdout_group": holdout}
                final = model
                if hasattr(model, "named_steps"):
                    final = list(model.named_steps.values())[-1]
                if hasattr(final, "alpha_"):
                    payload["alpha"] = float(final.alpha_)
                if hasattr(final, "l1_ratio_"):
                    payload["l1_ratio"] = float(final.l1_ratio_)
                fold_summaries.append(payload)
            pred_df.loc[test, "prediction"] = pred
            print(f"{model_name} holdout={holdout} train={int(train.sum())} test={int(test.sum())}", flush=True)

        if pred_df["prediction"].isna().any():
            raise RuntimeError(f"missing predictions for {model_name}")
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_df[keep + feature_cols + ["prediction"]].to_csv(model_dir / "predictions.csv", index=False)
        summaries[model_name] = {
            "model": model_name,
            "records": int(len(pred_df)),
            "target": args.target,
            "features": feature_cols,
            "fold_models": fold_summaries,
        }
        (model_dir / "summary.json").write_text(json.dumps(summaries[model_name], indent=2) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"records": int(len(df)), "models": requested, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

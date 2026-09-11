"""Hybrid rule + statistical models, evaluated honestly.

Motivation from the measured results: on topic the hand-written gazetteer is not
far behind the trained model *and gets different rows right* -- it is strongest
exactly where the trained model is weakest, on categories with unambiguous
vocabulary like ``accident`` and ``education``. Two models with complementary
errors are the textbook case for an ensemble, and the complementarity is
quantified below (an "oracle ceiling": the accuracy a perfect chooser between the
two would reach).

Two combination strategies are compared, deliberately in order of increasing
capacity, because the training split is only ~670 rows:

1. **Probability blend** -- ``a * P_model + (1 - a) * P_rules``. One parameter.
2. **Stacking** -- a logistic-regression meta-learner over both models'
   probability vectors. It can learn *per-class* trust, which is the point: it
   can believe the gazetteer about ``accident`` and ignore it about ``society``.

Both are scored by cross-validation on the training split only; the held-out set
is touched once per strategy at the end. Stacking is evaluated with
``StackingClassifier``, whose internal cross-fitting means the meta-learner never
sees a base model's prediction on a row that base model was trained on -- without
that, stacking reports a badly optimistic number.

The base model is built from train.py's TUNED entry, so it is byte-for-byte the
pipeline that serves requests. A blend weight fitted against any other
configuration would be the wrong weight for the deployed model.

Usage::

    python ensemble.py
    python ensemble.py --task topic
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.ensemble import StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from app.nlp.features import build_vectorizer  # noqa: E402
from train import TUNED, build_classifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
OUTPUT = Path(__file__).resolve().parent / "models" / "ensemble.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------
def prepare(texts: list[str]) -> np.ndarray:
    """Raw texts as an indexable array.

    No pre-tokenisation here, deliberately. An earlier version tokenised once up
    front and fed the vectorisers pre-split text, which made the 144-point grid
    search in tune.py affordable. But the weight this script fits is only valid
    for the model it is fitted against, so this script uses **exactly** the
    production pipeline -- build_vectorizer and build_classifier with train.py's
    tuned per-task settings. A blend weight fitted against a differently
    configured base model would be the wrong weight for the deployed one.
    """
    return np.array(texts, dtype=object)


def build_tfidf(task: str) -> Pipeline:
    """The production pipeline for ``task``, identically configured.

    Every knob comes from train.py's TUNED entry -- algorithm, penalty, class
    weight and feature thresholds -- because a blend weight is only valid for
    the model it was fitted against.
    """
    tuned = TUNED.get(task, {})
    return Pipeline(
        [
            ("features", build_vectorizer(**tuned.get("features", {}))),
            (
                "classifier",
                build_classifier(
                    tuned.get("algorithm", "logreg"),
                    0,
                    tuned.get("classifier_c", 4.0),
                    tuned.get("class_weight", "balanced"),
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------
# Rule-based backends as scikit-learn estimators
# --------------------------------------------------------------------------
class RuleClassifier(ClassifierMixin, BaseEstimator):
    """Adapts a rule-based backend to the scikit-learn estimator interface.

    ``fit`` learns nothing -- it only records the class order, which sklearn
    requires. That is the honest shape for these backends: they are fixed
    functions of the text, so they leak nothing across folds and can safely sit
    inside a cross-validated stack.
    """

    def __init__(self, task: str = "topic") -> None:
        self.task = task

    def _backend(self):
        if self.task == "topic":
            from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend

            return GazetteerTopicBackend()
        from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend

        return LexiconSentimentBackend()

    def fit(self, X, y):  # noqa: N803 - sklearn's parameter name
        self.classes_ = np.array(sorted(set(y)))
        self._impl = self._backend()
        return self

    def predict_proba(self, X):  # noqa: N803
        predictions = []
        for text in X:
            # Raw text: the backend applies the same cleaning and tokenisation
            # it uses in production, so nothing here can drift from it.
            prediction = self._impl.predict(str(text))
            predictions.append(
                [float(prediction.probabilities.get(c, 0.0)) for c in self.classes_]
            )
        matrix = np.array(predictions, dtype=float)
        # Renormalise: restricting to classes seen in training can drop mass.
        totals = matrix.sum(axis=1, keepdims=True)
        uniform = 1.0 / len(self.classes_)
        return np.where(totals > 0, matrix / np.maximum(totals, 1e-12), uniform)

    def predict(self, X):  # noqa: N803
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# Temperatures tried for the rule-based distribution before blending. The
# gazetteer's probabilities are a *share of matched evidence*, not calibrated
# probabilities, so they are much flatter than the logistic model's -- averaging
# the two raw is like averaging a confident vote with an abstention. Raising the
# distribution to a power > 1 sharpens it, < 1 flattens it; 1.0 is the identity,
# so the original blend stays in the search space and can still win.
TEMPERATURES = (0.5, 1.0, 2.0, 3.0, 5.0)


def sharpen(matrix: np.ndarray, temperature: float) -> np.ndarray:
    """Raise a probability matrix to ``temperature`` and renormalise rows."""
    if temperature == 1.0:
        return matrix
    powered = np.power(np.maximum(matrix, 1e-12), temperature)
    return powered / powered.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Strategy 1: probability blend
# --------------------------------------------------------------------------
def evaluate_blend(x_train, y_train, x_test, y_test, task: str) -> dict:
    """Weighted average of the two probability vectors.

    The weight is chosen on out-of-fold training predictions, so the held-out
    set plays no part in selecting it.
    """
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    base = build_tfidf(task)
    rules = RuleClassifier(task=task)

    # Out-of-fold model probabilities: every row scored by a model that did not
    # see it. Blending on in-fold probabilities would pick a weight of ~1.0,
    # because a fitted model is near-perfect on its own training rows.
    oof = cross_val_predict(
        base, x_train, y_train, cv=splitter, method="predict_proba", n_jobs=1
    )
    order = list(np.array(sorted(set(y_train))))
    rule_train = rules.fit(x_train, y_train).predict_proba(x_train)

    truth = np.array([order.index(label) for label in y_train])
    curve = []
    # Best (alpha, temperature) overall, and best with temperature pinned to the
    # identity. Both are reported: the two-parameter variant is only worth its
    # extra knob if it beats the one-parameter one by more than noise.
    best = {1.0: (1.0, -1.0), "any": (1.0, 1.0, -1.0)}
    for temperature in TEMPERATURES:
        sharpened = sharpen(rule_train, temperature)
        for alpha in [round(0.05 * i, 2) for i in range(21)]:
            blended = alpha * oof + (1 - alpha) * sharpened
            score = float(
                f1_score(
                    truth, blended.argmax(axis=1), average="macro", zero_division=0
                )
            )
            curve.append(
                {
                    "alpha": alpha,
                    "temperature": temperature,
                    "cv_f1_macro": round(score, 4),
                }
            )
            if score > best["any"][2]:
                best["any"] = (alpha, temperature, score)
            if temperature == 1.0 and score > best[1.0][1]:
                best[1.0] = (alpha, score)

    plain_alpha, plain_cv = best[1.0]
    tuned_alpha, tuned_temperature, tuned_cv = best["any"]

    base.fit(x_train, y_train)
    model_test = base.predict_proba(x_test)
    rule_test = rules.predict_proba(x_test)

    def held_out(alpha: float, temperature: float) -> tuple[float, float, list[str]]:
        blended = alpha * model_test + (1 - alpha) * sharpen(rule_test, temperature)
        predicted = [order[i] for i in blended.argmax(axis=1)]
        return (
            float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            float(accuracy_score(y_test, predicted)),
            predicted,
        )

    plain_f1, plain_accuracy, _ = held_out(plain_alpha, 1.0)
    tuned_f1, tuned_accuracy, _ = held_out(tuned_alpha, tuned_temperature)

    print(f"  blend        : alpha {plain_alpha:.2f}  CV macro-F1 {plain_cv:.4f}")
    print(
        "  weight curve: "
        + "  ".join(
            f"{e['alpha']:.1f}={e['cv_f1_macro']:.3f}"
            for e in curve
            if e["temperature"] == 1.0 and e["alpha"] % 0.2 < 0.01
        )
    )
    print(
        f"  + temperature: alpha {tuned_alpha:.2f} temp {tuned_temperature:g}  "
        f"CV {tuned_cv:.4f} ({tuned_cv - plain_cv:+.4f})  "
        f"held-out F1 {tuned_f1:.4f} ({tuned_f1 - plain_f1:+.4f})"
    )

    return {
        "strategy": "probability blend",
        "alpha": plain_alpha,
        "cv_f1_macro": plain_cv,
        "held_out_f1_macro": plain_f1,
        "held_out_accuracy": plain_accuracy,
        # Kept as a recorded negative result: sharpening the gazetteer
        # distribution moves CV and held-out in opposite directions, so the
        # extra parameter is not justified. Production uses alpha alone.
        "with_temperature": {
            "alpha": tuned_alpha,
            "temperature": tuned_temperature,
            "cv_f1_macro": tuned_cv,
            "held_out_f1_macro": tuned_f1,
            "held_out_accuracy": tuned_accuracy,
        },
        "weight_curve": curve,
    }


# --------------------------------------------------------------------------
# Strategy 2: stacking
# --------------------------------------------------------------------------
def evaluate_stack(x_train, y_train, x_test, y_test, task: str) -> dict:
    """Logistic-regression meta-learner over both models' probabilities."""
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    stack = StackingClassifier(
        estimators=[
            ("tfidf", build_tfidf(task)),
            ("rules", RuleClassifier(task=task)),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        stack_method="predict_proba",
        cv=splitter,
        passthrough=False,
        n_jobs=1,
    )

    # Nested CV: the outer split measures, the inner cv= cross-fits the stack.
    began = time.perf_counter()
    scores = cross_val_score(
        stack, x_train, y_train, cv=splitter, scoring="f1_macro", n_jobs=1
    )
    print(
        f"  stacking     : CV macro-F1 {scores.mean():.4f} "
        f"(+/- {scores.std():.4f})  [{time.perf_counter() - began:.0f}s]"
    )

    stack.fit(x_train, y_train)
    y_pred = list(stack.predict(x_test))

    return {
        "strategy": "stacking (LogReg meta-learner)",
        "cv_f1_macro": float(scores.mean()),
        "cv_std": float(scores.std()),
        "held_out_f1_macro": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "held_out_accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(
            y_test, y_pred, zero_division=0, digits=3, output_dict=True
        ),
        "_predictions": y_pred,
    }


# --------------------------------------------------------------------------
def run_task(task: str, texts: list[str], labels: list[str]) -> dict:
    print(f"\n{'=' * 76}\n{task.upper()} - rule + statistical ensemble\n{'=' * 76}")

    started = time.perf_counter()
    features = prepare(texts)
    print(f"tokenised {len(texts)} documents in {time.perf_counter() - started:.0f}s")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    classes = sorted(set(labels))
    print(f"train / test : {len(x_train)} / {len(x_test)}   classes: {len(classes)}\n")

    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # Both components alone, for a like-for-like baseline.
    base = build_tfidf(task)
    model_cv = cross_val_score(
        base, x_train, y_train, cv=splitter, scoring="f1_macro", n_jobs=1
    )
    base.fit(x_train, y_train)
    model_pred = list(base.predict(x_test))
    print(f"  TF-IDF alone : CV macro-F1 {model_cv.mean():.4f}")

    rules = RuleClassifier(task=task).fit(x_train, y_train)
    rule_pred = list(rules.predict(x_test))
    rule_cv = f1_score(
        y_train, list(rules.predict(x_train)), average="macro", zero_division=0
    )
    print(f"  rules alone  : train macro-F1 {rule_cv:.4f} (nothing fitted)")

    # How complementary are they? This is the number that decides whether an
    # ensemble can help at all.
    both_wrong = sum(
        1
        for truth, a, b in zip(y_test, model_pred, rule_pred, strict=True)
        if a != truth and b != truth
    )
    rules_rescue = sum(
        1
        for truth, a, b in zip(y_test, model_pred, rule_pred, strict=True)
        if a != truth and b == truth
    )
    model_rescue = sum(
        1
        for truth, a, b in zip(y_test, model_pred, rule_pred, strict=True)
        if a == truth and b != truth
    )
    ceiling = 1 - both_wrong / len(y_test)
    print(
        f"\n  complementarity on held-out ({len(y_test)} rows):\n"
        f"    only rules correct : {rules_rescue}\n"
        f"    only model correct : {model_rescue}\n"
        f"    both wrong         : {both_wrong}\n"
        f"    oracle ceiling     : {ceiling:.4f} accuracy "
        f"(the best any combination of these two could reach)\n"
    )

    blend = evaluate_blend(x_train, y_train, x_test, y_test, task)
    stack = evaluate_stack(x_train, y_train, x_test, y_test, task)

    baseline = {
        "strategy": "TF-IDF alone",
        "cv_f1_macro": float(model_cv.mean()),
        "held_out_f1_macro": float(
            f1_score(y_test, model_pred, average="macro", zero_division=0)
        ),
        "held_out_accuracy": float(accuracy_score(y_test, model_pred)),
    }

    candidates = [baseline, blend, stack]
    print("\n  summary (selection by CV; held-out shown for the record)")
    print(f"    {'strategy':34} {'CV F1':>8} {'held-out F1':>12} {'held-out acc':>13}")
    for entry in candidates:
        print(
            f"    {entry['strategy']:34} {entry['cv_f1_macro']:>8.4f} "
            f"{entry['held_out_f1_macro']:>12.4f} {entry['held_out_accuracy']:>13.4f}"
        )

    winner = max(candidates, key=lambda e: e["cv_f1_macro"])
    print(f"\n  CV winner    : {winner['strategy']}")

    predictions = stack.pop("_predictions")
    if winner is stack:
        print("\n  per-class on held-out:")
        print(classification_report(y_test, predictions, zero_division=0, digits=3))

    return {
        "oracle_ceiling_accuracy": float(ceiling),
        "complementarity": {
            "only_rules_correct": rules_rescue,
            "only_model_correct": model_rescue,
            "both_wrong": both_wrong,
            "held_out_rows": len(y_test),
        },
        "cv_winner": winner["strategy"],
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["topic", "sentiment", "both"], default="both")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    texts = [f"{r['title']} {r['content']}" for r in rows]

    tasks = ["topic", "sentiment"] if args.task == "both" else [args.task]
    results = {}
    for task in tasks:
        results[task] = run_task(task, texts, [r[task] for r in rows])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

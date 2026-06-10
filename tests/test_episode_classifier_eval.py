"""
Stage-1 episode-classifier evaluation against a hand-labeled set.

Reads `tests/data/episode_classifier_labeled.jsonl`, runs each labeled
(user, assistant) pair through the REAL stage-1 extractor (which spawns
`claude --print`), compares the predicted `has_episode` boolean to the
hand-labeled `expected_has_episode`, computes precision/recall/F1, and
writes `tests/artifacts/episode_classifier_eval.json` for review.

v1 design: the test does NOT assert a hard threshold. Reviewers
read the JSON artifact on the PR and set the threshold based on the
first run. Working target: precision >= 0.7, recall >= 0.6 (weighted
toward precision because false positives burn a stage-2 subprocess).

Skip behavior: this test is opt-in via the `RUN_CLASSIFIER_EVAL=1`
env var. It is NOT run by `make test` because (a) it requires the
live `claude` binary, (b) one full pass takes 5-15 minutes, and (c)
each pass costs real API tokens. Run on demand via:

    RUN_CLASSIFIER_EVAL=1 pytest tests/test_episode_classifier_eval.py -s

The produced artifact (`tests/artifacts/episode_classifier_eval.json`)
gets attached to the PR for reviewer threshold-setting.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from kai.config import Config
from kai.memory_extraction import _build_extraction_payload, _run_extractor

_LABELS = Path(__file__).parent / "data" / "episode_classifier_labeled.jsonl"
_ARTIFACT = Path(__file__).parent / "artifacts" / "episode_classifier_eval.json"

_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    webhook_secret="test-secret",
)


def _eval_config() -> Config:
    """Stage-1 config tuned for the eval pass: extraction enabled,
    consolidation off (the labeled exchanges have no prior facts to
    consolidate against). The timeout matches the production-tuned
    value so the eval reflects realistic latency."""
    return replace(
        _BASE_CONFIG,
        memory_enabled=True,
        memory_extraction_enabled=True,
        memory_extraction_timeout_s=60,
        memory_consolidation_candidates_n=0,
    )


@pytest.mark.skipif(
    os.environ.get("RUN_CLASSIFIER_EVAL") != "1",
    reason="Opt-in. Set RUN_CLASSIFIER_EVAL=1 to run the live-Haiku eval (5-15 min, costs API tokens).",
)
@pytest.mark.asyncio
async def test_episode_classifier_precision_recall():
    """Run every labeled exchange through stage 1 and compute the
    confusion matrix for `has_episode`. Writes a JSON artifact under
    tests/artifacts/ for the PR reviewer.

    The test does NOT assert a hard threshold. v1 leaves threshold
    selection to the reviewer based on the first run's numbers."""
    assert _LABELS.exists(), f"missing labeled set at {_LABELS}"

    labeled: list[dict] = []
    with _LABELS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labeled.append(json.loads(line))

    assert len(labeled) >= 30, f"need at least 30 labeled exchanges, got {len(labeled)}"

    config = _eval_config()
    results: list[dict] = []
    tp = fp = tn = fn = 0
    failures = 0

    for item in labeled:
        user = item["user"]
        assistant = item["assistant"]
        expected: bool = item["expected_has_episode"]
        item_id = item.get("id", "?")

        # Pass an empty candidate list explicitly to mirror the
        # production call shape with memory_consolidation_candidates_n=0
        # (the kill-switch branch in extract_and_store passes []). The
        # function's `candidates` arg defaults to None and treats None
        # and [] identically, so this is cosmetic, not a behavior fix -
        # but it makes the eval's correspondence to the production code
        # path one less inferential step for a future reader.
        payload = _build_extraction_payload(user, assistant, [])
        start = time.monotonic()
        try:
            result = await _run_extractor(
                payload,
                config,
                candidate_ids=set(),
                candidate_metadata={},
                user_id=f"eval-{item_id}",
                effective_backend="claude",
            )
            predicted: bool = result.has_episode
            error: str | None = None
        except Exception as e:
            # An extraction failure is recorded as an error and skipped
            # from the confusion matrix - it's neither a TP/FP/TN/FN nor
            # a model misclassification. Surfaces in the artifact under
            # `errors` so the reviewer can spot if a high error count is
            # masking the headline numbers.
            predicted = False
            error = f"{type(e).__name__}: {e}"
            failures += 1

        duration_ms = int((time.monotonic() - start) * 1000)

        results.append(
            {
                "id": item_id,
                "expected": expected,
                "predicted": predicted,
                "duration_ms": duration_ms,
                "error": error,
                "user_preview": user[:120],
                "assistant_preview": assistant[:120],
            }
        )

        if error is not None:
            continue
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    artifact = {
        "n_total": len(labeled),
        "n_errors": failures,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_item": results,
    }

    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")

    # Soft visibility for `pytest -s` runs. Spec v1 deliberately does
    # not assert a threshold; reviewer reads the artifact.
    print("\n=== Episode classifier eval ===")
    print(f"n_total={len(labeled)} n_errors={failures}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print(f"artifact: {_ARTIFACT}")

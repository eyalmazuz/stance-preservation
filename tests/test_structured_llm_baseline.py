import pytest

from scripts.structured_llm_baseline import (
    DocumentJudgment,
    PairJudgment,
    StanceDistribution,
    score_judgment,
    validate_document_judgment,
)


def pair_judgment(
    pair_id: int,
    source: tuple[float, float, float],
    summary: tuple[float, float, float],
    topics_match: bool = True,
) -> PairJudgment:
    labels = ("Against", "Neutral", "Favor")
    source_label = labels[max(range(3), key=lambda index: source[index])]
    summary_label = labels[max(range(3), key=lambda index: summary[index])]
    return PairJudgment(
        pair_id=pair_id,
        source_topic="policy",
        summary_topic="policy" if topics_match else "different target",
        topics_match=topics_match,
        comparison_topic="policy" if topics_match else "NONE",
        source_stance=source_label,
        summary_stance=summary_label,
        source_distribution=StanceDistribution(against=source[0], neutral=source[1], favor=source[2]),
        summary_distribution=StanceDistribution(against=summary[0], neutral=summary[1], favor=summary[2]),
        source_evidence="source evidence",
        summary_evidence="summary evidence",
    )


def test_identical_distributions_have_maximum_emd_preservation() -> None:
    judgment = DocumentJudgment(pairs=[pair_judgment(0, (0.1, 0.2, 0.7), (0.1, 0.2, 0.7))])

    scores = score_judgment(judgment)

    assert scores["structured_llm_emd_preds"] == pytest.approx(2.0)
    assert scores["structured_llm_argmax_preds"] == pytest.approx(1.0)
    assert scores["structured_llm_topic_coverage"] == pytest.approx(1.0)


def test_full_reversal_has_minimum_emd_preservation() -> None:
    judgment = DocumentJudgment(pairs=[pair_judgment(0, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))])

    scores = score_judgment(judgment)

    assert scores["structured_llm_emd_preds"] == pytest.approx(0.0)
    assert scores["structured_llm_argmax_preds"] == pytest.approx(0.0)


def test_topic_mismatches_are_excluded_and_coverage_is_reported() -> None:
    judgment = DocumentJudgment(
        pairs=[
            pair_judgment(0, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            pair_judgment(1, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), topics_match=False),
        ]
    )

    scores = score_judgment(judgment)

    assert scores["structured_llm_emd_preds"] == pytest.approx(2.0)
    assert scores["structured_llm_argmax_preds"] == pytest.approx(1.0)
    assert scores["structured_llm_topic_coverage"] == pytest.approx(0.5)


def test_validation_rejects_missing_pair_ids() -> None:
    judgment = DocumentJudgment(pairs=[pair_judgment(1, (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))])

    with pytest.raises(ValueError, match="Expected pair IDs"):
        validate_document_judgment(judgment, expected_pairs=1)

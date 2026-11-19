from argparse import Namespace

import pytest

from main import prediction_column_name


@pytest.mark.parametrize(
    ("model", "emd_method", "expected"),
    [
        ("emd", "emd", "emd_preds"),
        ("emd", "js", "js_preds"),
        ("emd", "kl", "kl_preds"),
        ("emd", "argmax_ordinal", "argmax_ordinal_preds"),
        ("emd", "argmax_exact", "argmax_exact_preds"),
        ("emd", "euclidean", "euclidean_preds"),
        ("emd", "itakura", "itakura_preds"),
    ],
)
def test_emd_variants_have_distinct_prediction_columns(model: str, emd_method: str, expected: str) -> None:
    args = Namespace(
        model=model,
        emd_score_method=emd_method,
        nli_score_method="preservation",
        use_hebrew_morph_normalization=False,
    )

    assert prediction_column_name(args) == expected


def test_other_specialized_prediction_columns_are_preserved() -> None:
    nli_args = Namespace(
        model="nli",
        emd_score_method="emd",
        nli_score_method="preservation",
        use_hebrew_morph_normalization=False,
    )
    morph_args = Namespace(
        model="rouge1",
        emd_score_method="emd",
        nli_score_method="preservation",
        use_hebrew_morph_normalization=True,
    )

    assert prediction_column_name(nli_args) == "nli_preservation_preds"
    assert prediction_column_name(morph_args) == "rouge1_hebrew_morph_preds"

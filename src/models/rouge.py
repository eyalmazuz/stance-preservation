from rouge_score import rouge_scorer

from src.utils.text_utils import hebrew_morph_normalize


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


class RougeScorer:
    def __init__(self, type_: str, normalize_hebrew: bool = False) -> None:
        self.type = type_
        self.normalize_hebrew = normalize_hebrew
        self.scorer = rouge_scorer.RougeScorer(
            [self.type],
            use_stemmer=False,
            tokenizer=WhitespaceTokenizer() if self.normalize_hebrew else None,
        )

    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        rouge_score: float = 0.0
        if isinstance(hypotheses, list) and isinstance(references, list):
            for hypothesis, reference in zip(hypotheses, references):
                rouge_score += self.scorer.score(
                    self.prepare_text(hypothesis["text"]),
                    self.prepare_text(reference["text"]),
                )[self.type].fmeasure
            rouge_score /= len(hypotheses)
        elif isinstance(hypotheses, str) and isinstance(references, str):
            rouge_score = self.scorer.score(self.prepare_text(hypotheses), self.prepare_text(references))[
                self.type
            ].fmeasure
        else:
            raise ValueError("Invalid Data.")

        return rouge_score

    def prepare_text(self, text: str) -> str:
        if self.normalize_hebrew:
            return hebrew_morph_normalize(text)
        return text

from rouge_score import rouge_scorer


class RougeScorer:
    def __init__(self, type_: str) -> None:
        self.type = type_
        self.scorer = rouge_scorer.RougeScorer(
            [self.type],
            use_stemmer=False,
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
                rouge_score += self.scorer.score(hypothesis["text"], reference["text"])[self.type].fmeasure
            rouge_score /= len(hypotheses)
        elif isinstance(hypotheses, str) and isinstance(references, str):
            rouge_score = self.scorer.score(hypotheses, references)[self.type].fmeasure
        else:
            raise ValueError("Invalid Data.")

        return rouge_score

from sacrebleu import sentence_bleu


class BleuScorer:
    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        bleu_score: float = 0.0
        if isinstance(hypotheses, list) and isinstance(references, list):
            for hypothesis, reference in zip(hypotheses, references):
                bleu_score += sentence_bleu(hypothesis["text"], [reference["text"]]).score
            bleu_score /= len(hypotheses)
        elif isinstance(hypotheses, str) and isinstance(references, str):
            bleu_score = sentence_bleu(hypotheses, [references]).score
        else:
            raise ValueError("Invalid Data.")

        return bleu_score

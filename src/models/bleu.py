from sacrebleu import sentence_bleu

from src.utils.text_utils import hebrew_morph_normalize


class BleuScorer:
    def __init__(self, topic_filter: str | None = None, normalize_hebrew: bool = False) -> None:
        self.topic_filter = topic_filter
        self.normalize_hebrew = normalize_hebrew

    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        bleu_score: float = 0.0
        if isinstance(hypotheses, list) and isinstance(references, list):
            scored_pairs = 0
            for hypothesis, reference in zip(hypotheses, references):
                if not self.should_score_pair(hypothesis, reference):
                    continue
                bleu_score += sentence_bleu(
                    self.prepare_text(hypothesis["text"]),
                    [self.prepare_text(reference["text"])],
                ).score
                scored_pairs += 1
            if scored_pairs == 0:
                return 0.0
            bleu_score /= scored_pairs
        elif isinstance(hypotheses, str) and isinstance(references, str):
            bleu_score = sentence_bleu(self.prepare_text(hypotheses), [self.prepare_text(references)]).score
        else:
            raise ValueError("Invalid Data.")

        return bleu_score

    def prepare_text(self, text: str) -> str:
        if self.normalize_hebrew:
            return hebrew_morph_normalize(text)
        return text

    def should_score_pair(self, hypothesis: dict[str, str], reference: dict[str, str]) -> bool:
        if self.topic_filter is None:
            return True
        topics_match = hypothesis.get("topic") == reference.get("topic")
        if self.topic_filter == "match":
            return topics_match
        if self.topic_filter == "mismatch":
            return not topics_match
        raise ValueError(f"Invalid topic filter: {self.topic_filter}")

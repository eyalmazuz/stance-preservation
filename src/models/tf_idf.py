import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfIdfScorer:
    def __init__(
        self,
    ) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))

    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        if isinstance(hypotheses, list) and isinstance(references, list) and len(hypotheses) != len(references):
            raise ValueError("Sentence-level TF-IDF expects aligned lists of equal length.")
        tfidf_score: float = 0.0
        if isinstance(hypotheses, str) and isinstance(references, str):
            hypotheses_text: list[str] = [hypotheses]
            references_text: list[str] = [references]
        elif isinstance(hypotheses, list) and isinstance(references, list):
            hypotheses_text: list[str] = [hyp["text"] for hyp in hypotheses]
            references_text: list[str] = [ref["text"] for ref in references]
        self.fit(hypotheses_text + references_text)
        hypo_vec = self.vectorizer.transform(hypotheses_text)
        ref_vec = self.vectorizer.transform(references_text)
        sim_score = cosine_similarity(hypo_vec, ref_vec)
        tfidf_score = np.diag(sim_score).mean()

        return tfidf_score

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

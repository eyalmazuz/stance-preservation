import os

import numpy as np

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv is not found falling back to environ.")


class EmbeddingScorer:
    def __init__(
        self,
        model: str = "text-embedding-3-large",
    ) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG"),
            project=os.environ.get("OPENAI_PROJECT"),
        )
        self.model = model

    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        if isinstance(hypotheses, list) and isinstance(references, list) and len(hypotheses) != len(references):
            raise ValueError("Sentence-level TF-IDF expects aligned lists of equal length.")
        emb_score: float = 0.0
        if isinstance(hypotheses, str) and isinstance(references, str):
            hypotheses_text: list[str] = [hypotheses]
            references_text: list[str] = [references]
        elif isinstance(hypotheses, list) and isinstance(references, list):
            hypotheses_text: list[str] = [hyp["text"] for hyp in hypotheses]
            references_text: list[str] = [ref["text"] for ref in references]
        hypo_emb = self.get_embedding(hypotheses_text)
        ref_emb = self.get_embedding(references_text)
        sim_score = cosine_similarity(hypo_emb, ref_emb)
        emb_score = np.diag(sim_score).mean()

        return emb_score

    def get_embedding(self, texts: str | list[str], model="text-embedding-3-small"):
        if isinstance(texts, str):
            texts = [texts]
        texts = [text.replace("\n", " ")[:8191] for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [res.embedding for res in response.data]
        return np.array(embeddings)

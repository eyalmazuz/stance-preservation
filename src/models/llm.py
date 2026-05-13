import os

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("dotenv is not found falling back to environ.")


class ScoreResponse(BaseModel):
    score: int = Field(ge=0, le=10)


class LLMScorer:
    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        prompt: str = "",
        max_workers: int = 1,
    ) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG"),
            project=os.environ.get("OPENAI_PROJECT"),
        )
        self.model = model
        self.prompt = self.load_prompt(prompt)
        self.max_workers = max_workers

    def score(
        self,
        hypotheses: str | list[dict[str, str]],
        references: str | list[dict[str, str]],
        **kwargs: dict,
    ) -> float:
        if isinstance(hypotheses, list) and isinstance(references, list) and len(hypotheses) != len(references):
            raise ValueError("Sentence-level TF-IDF expects aligned lists of equal length.")
        llm_score: float = 0.0
        if isinstance(hypotheses, str) and isinstance(references, str):
            hypotheses_text: list[str] = [hypotheses]
            references_text: list[str] = [references]
        elif isinstance(hypotheses, list) and isinstance(references, list):
            hypotheses_text: list[str] = [hyp["text"] for hyp in hypotheses]
            references_text: list[str] = [ref["text"] for ref in references]
        else:
            raise TypeError("LLM scoring expects both inputs to be strings or both inputs to be sentence lists.")
        sentence_pairs = list(zip(hypotheses_text, references_text, strict=True))
        if not sentence_pairs:
            return 0.0
        if self.max_workers <= 1 or len(sentence_pairs) <= 1:
            for hyp, ref in sentence_pairs:
                llm_score += self.score_sentence_pair((hyp, ref))
        else:
            workers = min(self.max_workers, len(sentence_pairs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                llm_score = sum(executor.map(self.score_sentence_pair, sentence_pairs))
        return llm_score / len(sentence_pairs)

    def score_sentence_pair(self, pair: tuple[str, str]) -> float:
        hyp, ref = pair
        return self.generate(summary=hyp, article=ref, model=self.model) / 10

    def generate(self, summary: str, article: str, model="text-embedding-3-small"):
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that judges the quality of "
                        "preserving the stance in text summaries."
                    ),
                },
                {
                    "role": "user",
                    "content": self.build_prompt(summary=summary, article=article),
                },
            ],
            text_format=ScoreResponse,
        )

        if response.output_parsed is not None:
            return response.output_parsed.score
        else:
            return 0

    def build_prompt(self, summary: str, article: str) -> str:
        return self.prompt.format(article=article, summary=summary)

    @staticmethod
    def load_prompt(prompt: str) -> str:
        prompt_path = Path(prompt)
        if prompt_path.is_file():
            return prompt_path.read_text(encoding="utf-8")
        return prompt

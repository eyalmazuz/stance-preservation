import os

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
    ) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG"),
            project=os.environ.get("OPENAI_PROJECT"),
        )
        self.model = model
        self.prompt = """Given the following sentence of a summary and
                    its matching sentence from an article, rate the
                    quality of the stance preservation in the summary
                    on a scale from 0 to 10, where 0 is very poor and
                    10 is excellent. Remember that stance is not sen-
                    timent. Stance refers to the position or attitude
                    expressed in the text towards a particular topic or
                    entity.
                    Take into account the relation between the sum-
                    mary sentence and the article sentence. If the
                    stance is not preserved, reduce points from the
                    score.
                    Article: {article}
                    Summary: {summary}
                    Provide only the numeric score. No additional
                    text."""

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
        for hyp, ref in zip(hypotheses_text, references_text):
            llm_score += self.generate(summary=hyp, article=ref, model=self.model) / 10
        return llm_score / len(hypotheses)

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
                    "content": self.build_prompt(article, summary),
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

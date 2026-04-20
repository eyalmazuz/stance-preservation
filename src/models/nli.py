from transformers import pipeline


class NLIScorer:
    def __init__(self, model_name: str, aggregate: str, language: str = "he") -> None:
        self.model_name = model_name
        self.aggregate = aggregate
        # self.model = AutoModel.from_pretrained(self.model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = pipeline("zero-shot-classification", model=self.model_name, truncation=True)
        self.language = language
        self.stance_to_score = {"Favor": +1.0, "Against": -1.0, "Neutral": 0.0}

    def score(self, hypotheses: str | list[dict[str, str]], references: str | list[dict[str, str]], **kwargs) -> float:
        shift_score: float = 0.0

        if isinstance(hypotheses, str) and isinstance(references, str):
            hypotheses: list[str] = [hypotheses]
            references: list[str] = [references]

        for hyp, ref in zip(hypotheses, references):
            hyp_text, hyp_topic = self.get_data(hyp)
            ref_text, ref_topic = self.get_data(ref)

            hyp_labels, hyp_scores = self.predict(hyp_text, hyp_topic)
            hyp_score = self.score_dist_to_expected(hyp_labels, hyp_scores)

            ref_labels, ref_scores = self.predict(ref_text, ref_topic)
            ref_score = self.score_dist_to_expected(ref_labels, ref_scores)

            shift_score += hyp_score - ref_score

        return shift_score / len(hypotheses)

    def predict(self, text: str, topic: str):
        res = self.pipeline(
            text,
            candidate_labels=["Favor", "Against", "Neutral"],
            hypothesis_template=self.get_template(topic, self.language),
            multi_label=False,
        )

        labels = list(res["labels"])
        scores = [float(x) for x in res["scores"]]

        return labels, scores

    def score_dist_to_expected(
        self,
        labels: list[str],
        scores: list[float],
    ) -> float:
        total = float(sum(scores)) or 1.0
        s = 0.0
        for lab, sc in zip(labels, scores):
            s += self.stance_to_score.get(lab, 0.0) * float(sc)
        v = s / total
        return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)

    def get_data(self, data) -> tuple[str, str]:
        if isinstance(data, dict):
            text = data["text"] if self.aggregate == "sentence" else data["full_text"]
            topic = data["topic"]
        else:
            text = data
            topic = ""

        return text, topic

    def get_template(self, topic: str, language: str = "he") -> str:
        if language == "en":
            hyp = f"This text expresses a {{}} stance toward '{topic or 'the topic'}'."
        elif language == "he":
            hyp = f"הטקסט הבא מביע עמדה {{}} כלפי '{topic or 'הנושא'}'."
        else:
            raise ValueError(f"Invalid language option: {language}.")

        return hyp

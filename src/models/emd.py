import numpy as np
import ot
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from torch.distributions import Categorical
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from src.utils.data_utils import split_into_sentences
from src.utils.prompt_utils import get_emd_prompt
from src.utils.text_utils import topics_match_soft


class EMDScorer:
    TASK = "Retrieve semantically similar text."

    def __init__(
        self,
        matching_model_name: str,
        topic_model_name: str,
        stance_model_name: str,
        aggregate: str,
        language: str = "he",
        entropy_threshold: float = 0.0,
        use_topic_filtering: bool = False,
        use_soft_topic_filtering: bool = False,
        use_dist_topic_score: bool = False,
        use_weighted_emd: bool = False,
    ) -> None:
        self.matching_model = self.get_matching_model(matching_model_name)
        self.matching_model_name = matching_model_name
        self.topic_model, self.topic_tokenizer = self.get_topic_model(topic_model_name)
        self.stance_model, self.stance_tokenizer = self.get_stance_model(stance_model_name)
        self.aggregate = aggregate
        self.language = language
        self.entropy_threshold = entropy_threshold if entropy_threshold != 0.0 else float("inf")
        self.canonical_labels = ["Against", "Neutral", "Favor"]
        self.use_topic_filtering = use_topic_filtering
        self.use_soft_topic_filtering = use_soft_topic_filtering
        self.use_dist_topic_score = use_dist_topic_score
        self.use_weighted_emd = use_weighted_emd
        self.stance_value = {"Against": -1, "Neutral": 0, "Favor": 1}
        self.C = np.array(
            [
                [abs(self.stance_value[a] - self.stance_value[b]) for b in self.canonical_labels]
                for a in self.canonical_labels
            ],
            dtype=np.float64,
        )

    def score(self, hypotheses: str | list[dict[str, str]], references: str | list[dict[str, str]], **kwargs) -> float:
        emd_score: float = 0.0

        if isinstance(hypotheses, str) and isinstance(references, str):
            hyp_sentences: list[str] = split_into_sentences(hypotheses)
            ref_sentences: list[str] = split_into_sentences(references)
            matched_pairs = self.get_matching_pairs(hyp_sentences, ref_sentences)
            full_hyp = hypotheses
            full_ref = references
        elif isinstance(hypotheses, list) and isinstance(references, list):
            matched_pairs = [(hyp["text"], ref["text"], 1.0) for hyp, ref in zip(hypotheses, references, strict=True)]
            full_hyp = hypotheses[0]["full_text"]
            full_ref = references[0]["full_text"]

        kept = 0
        for hyp_sentence, ref_sentence, sim in tqdm(matched_pairs, leave=False):
            hyp_topic = self.get_topic(full_hyp, hyp_sentence)
            ref_topic = self.get_topic(full_ref, ref_sentence)

            hyp_stance_probs = self.get_stance(hyp_sentence, hyp_topic)
            ref_stance_probs = self.get_stance(ref_sentence, ref_topic)

            if (
                ((hyp_topic != ref_topic) and self.use_topic_filtering)
                or (not topics_match_soft(hyp_topic, ref_topic) and self.use_soft_topic_filtering)
                or (Categorical(hyp_stance_probs).entropy() > self.entropy_threshold)
                or (Categorical(ref_stance_probs).entropy() > self.entropy_threshold)
            ):
                continue

            stance_emd = ot.emd2(
                ref_stance_probs.numpy().astype(np.float64),
                hyp_stance_probs.numpy().astype(np.float64),
                np.array(self.C).astype(np.float64),
            )
            if self.use_dist_topic_score:
                topic_similarity = (
                    (self.encode_text([hyp_topic]) @ self.encode_text([ref_topic]).T).squeeze().cpu().item()
                )
                emd_score += stance_emd + 0.5 * (1 - topic_similarity)
            elif self.use_weighted_emd:
                emd_score += stance_emd * sim
            else:
                emd_score += stance_emd
            kept += 1 if not self.use_weighted_emd else sim

        if kept == 0:
            return 2.0
        return emd_score / kept

    def get_matching_model(self, model_name: str):
        model = SentenceTransformer(model_name)
        return model

    def get_topic_model(self, model_name: str):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            quantization_config=quant_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def get_stance_model(self, model_name: str):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def get_matching_pairs(
        self, hyp_sentences: list[str], ref_sentences: list[str]) -> list[tuple[str, str, float]]:

        hyp_embeddings = self.encode_text(hyp_sentences, is_query=True)
        ref_embeddings = self.encode_text(ref_sentences, is_query=False)

        scores = hyp_embeddings @ ref_embeddings.T
        best_sims, best_cols = scores.max(axis=1)
        matched_pairs = [(hyp_sentences[i], ref_sentences[j], best_sims[i].item()) for i, j in enumerate(best_cols)]

        return matched_pairs

    def encode_text(self, texts: list[str], is_query: bool = False):
        if is_query:
            match self.matching_model_name:
                case "intfloat/multilingual-e5-large-instruct":
                    texts_instruct: list[str] = [
                        EMDScorer.get_detailed_instruct(EMDScorer.TASK, sentence) for sentence in texts
                    ]
                    return self.matching_model.encode(texts_instruct, convert_to_tensor=True, normalize_embeddings=True)
                case "microsoft/harrier-oss-v1-0.6b":
                    return self.matching_model.encode(texts, prompt_name="sts_query", convert_to_tensor=True)
                case _:
                    return self.matching_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

        return self.matching_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def get_topic(self, full_text: str, sentence: str) -> str:
        prompt = get_emd_prompt(self.language).format(context=full_text, sentence=sentence)

        inputs = self.topic_tokenizer(prompt.strip(), return_tensors="pt", padding=True).to(self.topic_model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.topic_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=False,
                max_new_tokens=10,
                pad_token_id=self.topic_tokenizer.eos_token_id,
            )

        topic_tokens = outputs[0][prompt_length:]
        topic = self.topic_tokenizer.decode(topic_tokens, skip_special_tokens=True).strip()
        return topic.split("\n")[0]

    def get_stance(self, sentence: str, topic: str):
        if self.language == "he":
            combined_input = f"{sentence} [SEP] {topic}"
            inputs = self.stance_tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.stance_model(**inputs)

            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()

        elif self.language == "en":  # English
            pass

        return probs

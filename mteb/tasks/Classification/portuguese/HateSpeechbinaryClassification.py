from mteb import MTEB
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from sentence_transformers import SentenceTransformer


class HateSpeechbinaryClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "HateSpeechbinaryClassification",
            "hf_hub_name": "projetomemoreba/mteb_Portuguese_Hate_Speech_binary",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": [ "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "217d8bd202909938bbc921a0495fe362aaeae6a9",
        }

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[HateSpeechbinaryClassification()])
evaluation.run(model)
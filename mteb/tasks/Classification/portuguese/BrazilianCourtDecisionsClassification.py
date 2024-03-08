from mteb import MTEB
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from sentence_transformers import SentenceTransformer


class BrazilianCourtDecisionsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "BrazilianCourtDecisionsClassification",
            "hf_hub_name": "projetomemoreba/mteb_brazilian_court_decisions",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "ae5cdd58be9e246773486042e743abc6832906c8",
        }

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[BrazilianCourtDecisionsClassification()])
evaluation.run(model)
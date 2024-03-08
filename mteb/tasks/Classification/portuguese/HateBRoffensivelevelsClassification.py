from mteb import MTEB
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from sentence_transformers import SentenceTransformer


class HateBRoffensivelevelsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "HateBRoffensivelevelsClassification",
            "hf_hub_name": "projetomemoreba/mteb_HateBR_offensive_levels",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": [ "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "49e1b6c489aae11bd10052a7e317d7048e2fcfd9",
        }

model = SentenceTransformer("average_word_embeddings_komninos")
evaluation = MTEB(tasks=[HateBRoffensivelevelsClassification()])
evaluation.run(model)
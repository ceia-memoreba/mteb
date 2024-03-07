import datasets
from ...abstasks import AbsTaskClassification


class BrazilianCourtDecisionsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "BrazilianCourtDecisionsClassification",
            "hf_hub_name": "projetomemoreba/mteb_brazilian_court_decisions",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ['pt'],
            "main_score": "accuracy",
            "revision": "ae5cdd58be9e246773486042e743abc6832906c8",
        }


    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return


        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        self.dataset_transform()
        self.data_loaded = True


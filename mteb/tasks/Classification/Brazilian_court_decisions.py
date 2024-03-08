from ...abstasks import AbsTaskClassification

class Brazilian_court_decisions(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "Brazilian_court_decisions",
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
            "revision": "d0864a96e7a85092b37a99c50055a7a86771ec78",
        }
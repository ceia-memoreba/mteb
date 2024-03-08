from ...abstasks import AbsTaskClassification


class BrazilianCourtDecisionsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "BrazilianCourtDecisionsClassification",
            "hf_hub_name": "projetomemoreba/mteb_brazilian_court_decisions",
            "description": "Amazon Polarity Classification Dataset.",
            "reference": "https://dl.acm.org/doi/10.1145/2507157.2507163",
            "category": "p2p",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "accuracy",
            "revision": "ae5cdd58be9e246773486042e743abc6832906c8",
        }
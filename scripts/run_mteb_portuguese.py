"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging

import logging
from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

TASK_LIST_CLASSIFICATION = [
    "BrazilianCourtDecisionsClassification"
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
)

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)
evaluation = MTEB(tasks=[ "BrazilianCourtDecisionsClassification"])
evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"])


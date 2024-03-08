"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
    "BrazilianCourtDecisionsClassification"
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
)

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
    evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=eval_splits)


from enum import StrEnum


class ModelType(StrEnum):
    Embedding = "Embedding"
    SequenceClassification = "SequenceClassification"
    TokenClassification = "TokenClassification"
    SentenceEmbedding = "SentenceEmbedding"
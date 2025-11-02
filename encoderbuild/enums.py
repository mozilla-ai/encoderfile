from enum import StrEnum

class ModelType(StrEnum):
    EMBEDDING = "embedding"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"

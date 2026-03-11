from enum import StrEnum


class ModelType(StrEnum):
    Embedding = "embedding"
    SequenceClassification = "sequence_classification"
    TokenClassification = "token_classification"
    SentenceEmbedding = "sentence_embedding"


class TokenizerTruncationSide(StrEnum):
    Left = "left"
    Right = "right"


class TokenizerTruncationStrategy(StrEnum):
    LongestFirst = "longest_first"
    OnlyFirst = "only_first"
    OnlySecond = "only_second"


class TokenizerPadStrategy(StrEnum):
    BatchLongest = "batch_longest"
    Fixed = "fixed"

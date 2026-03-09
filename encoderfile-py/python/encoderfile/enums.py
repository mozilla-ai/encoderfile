from enum import StrEnum


class ModelType(StrEnum):
    Embedding = "Embedding"
    SequenceClassification = "SequenceClassification"
    TokenClassification = "TokenClassification"
    SentenceEmbedding = "SentenceEmbedding"


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

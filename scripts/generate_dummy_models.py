"""Create dummy models."""

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizerFast,
    BertModel,
    PretrainedConfig,
    PreTrainedModel,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from optimum.onnxruntime import (
    ORTModel,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from pathlib import Path
import random
import numpy as np
import torch

# i'm not playing games with randomness today -RB
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TOKENIZER: BertTokenizerFast = AutoTokenizer.from_pretrained("bert-base-uncased")

BERT_CONFIG = BertConfig(
    vocab_size=TOKENIZER.vocab_size,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
)

BERT_CONFIG_WITH_LABELS = BertConfig(
    vocab_size=TOKENIZER.vocab_size,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    num_labels=2,
    id2label={0: "good", 1: "bad"},
    label2id={"good": 0, "bad": 1},
)


def create_bert_model(
    name: str,
    model_cls: type[PreTrainedModel],
    ort_cls: type[ORTModel],
    config: PretrainedConfig = BERT_CONFIG,
):
    model_path = Path(f"models/{name}")
    model_path.mkdir(parents=True, exist_ok=True)

    # create base model and save
    model = model_cls(config)

    model.save_pretrained(model_path)

    # save tokenizer
    TOKENIZER.save_pretrained(model_path)

    # load as ORT and export
    ort = ort_cls.from_pretrained(model_path, export=True)
    ort.save_pretrained(model_path)


create_bert_model(
    "embedding",
    BertModel,
    ORTModel,
)

create_bert_model(
    "sequence_classification",
    BertForSequenceClassification,
    ORTModelForSequenceClassification,
    config=BERT_CONFIG_WITH_LABELS,
)

create_bert_model(
    "token_classification",
    BertForTokenClassification,
    ORTModelForTokenClassification,
    config=BERT_CONFIG_WITH_LABELS,
)

create_bert_model(
    "sentence_embedding",
    BertModel,
    ORTModel,
)


def assert_dir_size_under(path, max_mb):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    max_bytes = max_mb * 1024 * 1024

    if total_bytes > max_bytes:
        raise RuntimeError(
            f"Directory {path} is {total_bytes / (1024**2):.2f} MB (limit: {max_mb} MB)"
        )


# throw error if total dir size is over 32 MB
assert_dir_size_under(Path("models"), 32)

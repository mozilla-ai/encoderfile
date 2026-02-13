"""Download models for testing."""

import os
from transformers import AutoTokenizer, AutoConfig
from optimum.onnxruntime import (
    ORTModel,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from create_dummy_model import (
    DummySequenceConfig,
    DummyTokenConfig,
    DUMMY_SEQUENCE_ENCODER,
    DUMMY_TOKEN_ENCODER,
)

MODELS_DIR = "models/"


def download_export_models(
    model_id: str,
    save_name: str,
    ort_cls: type[ORTModel],
    export: bool = True,
):
    save_dir = os.path.join(MODELS_DIR, save_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)

    model = ort_cls.from_pretrained(model_id, export=export)
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    # save embedding model
    download_export_models("microsoft/MiniLM-L12-H384-uncased", "embedding", ORTModel)

    # save sequence classification model
    download_export_models(
        "tabularisai/multilingual-sentiment-analysis",
        "sequence_classification",
        ORTModelForSequenceClassification,
    )

    # save token classification model
    download_export_models(
        "mozilla-ai/tiny-pii-electra-small",
        "token_classification",
        ORTModelForTokenClassification,
    )

    AutoConfig.register(DUMMY_SEQUENCE_ENCODER, DummySequenceConfig)
    AutoConfig.register(DUMMY_TOKEN_ENCODER, DummyTokenConfig)
    # save dummy models
    download_export_models(
        "mozilla-ai/test-dummy-sequence-encoder",
        "dummy_sequence_classifier",
        ORTModelForSequenceClassification,
        export=False,
    )
    download_export_models(
        "mozilla-ai/test-dummy-token-encoder",
        "dummy_token_classifier",
        ORTModelForTokenClassification,
        export=False,
    )

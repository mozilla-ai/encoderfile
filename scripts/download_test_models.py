"""Download models for testing."""

import os
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModel

MODELS_DIR = "models/"


def download_export_models(
    model_id: str,
    save_name: str,
    transformers_cls: type[PreTrainedModel],
    ort_cls: type[ORTModel],
):
    save_dir = os.path.join(MODELS_DIR, save_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_dir)

    model = transformers_cls.from_pretrained(model_id)
    model.save_pretrained(save_dir)

    model = ort_cls.from_pretrained(model_id, export=True)
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    # save embedding model
    download_export_models(
        "microsoft/MiniLM-L12-H384-uncased", "embedding", AutoModel, ORTModel
    )

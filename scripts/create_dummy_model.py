"""
Creates dummy PyTorch models with fixed logit distribution for CI/testing.

The models output logits where:
- First class has the highest value
- Second class has a slightly lower value
- Remaining classes have -4.0 values
"""

from pathlib import Path
import click
import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    ElectraConfig,
)
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    BaseModelOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import register_tasks_manager_onnx
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.onnx.model_configs import COMMON_TEXT_TASKS


DUMMY_SEQUENCE_CLASSIFIER = "mozilla-ai/test-dummy-sequence-classifier"
DUMMY_TOKEN_CLASSIFIER = "mozilla-ai/test-dummy-token-classifier"
DUMMY_SEQUENCE_EMBEDDINGS = "mozilla-ai/test-dummy-sequence-embeddings"
DUMMY_TOKEN_EMBEDDINGS = "mozilla-ai/test-dummy-token-embeddings"

SEQUENCE_CLASSIFIER_OUTPUT_DIR = "./models/dummy_electra_sequence_classifier"
TOKEN_CLASSIFIER_OUTPUT_DIR = "./models/dummy_electra_token_classifier"
SEQUENCE_EMBEDDINGS_OUTPUT_DIR = "./models/dummy_electra_sequence_embeddings"
TOKEN_EMBEDDINGS_OUTPUT_DIR = "./models/dummy_electra_token_embeddings"


class DummyConfig(ElectraConfig):
    """Dummy configuration similar to BERT configuration."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class DummyTokenEmbedConfig(DummyConfig):
    """Dummy configuration similar to BERT configuration."""

    model_type = DUMMY_TOKEN_EMBEDDINGS

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class DummySequenceEmbedConfig(DummyConfig):
    """Dummy configuration similar to BERT configuration."""

    model_type = DUMMY_SEQUENCE_EMBEDDINGS

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class DummyTokenClassConfig(DummyConfig):
    """Dummy configuration similar to BERT configuration."""

    model_type = DUMMY_TOKEN_CLASSIFIER

    def __init__(
        self,
        num_labels=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class DummySequenceClassConfig(DummyConfig):
    """Dummy configuration similar to BERT configuration."""

    model_type = DUMMY_SEQUENCE_CLASSIFIER

    def __init__(
        self,
        num_labels=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class DummySequenceCommon(PreTrainedModel):
    """Dummy sequence embeddings that outputs fixed logits per sequence."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Create logit template with fixed values
        # First class: 2.0, Second class: 1.0, Rest: -4.0
        logits_template = torch.full((1, config.num_labels), -4.0)
        logits_template[0, 0] = 2.0
        if config.num_labels > 1:
            logits_template[0, 1] = 1.0

        self.logits_template = logits_template
        self.register_buffer(
            "dummy", torch.tensor(0)
        )  # Dummy buffer to avoid empty model warning

    def get_loss_logits(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        """Forward pass that returns fixed logits for each sequence."""

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get batch size from input_ids
        # Force use of attention mask so optimum does not drop it later
        batch_size, _ = (torch.mul(input_ids, attention_mask)).shape

        # Expand template logits to match batch size
        # Each sequence gets the same fixed logit values from the template
        logits = self.logits_template.expand(batch_size, -1).clone()

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return loss, logits


class DummySequenceEmbeddings(DummySequenceCommon):
    """Dummy sequence embeddings that outputs fixed logits per sequence."""

    config_class = DummySequenceEmbedConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        _, logits = self.get_loss_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        return BaseModelOutput(
            last_hidden_state=logits,
            hidden_states=None,
            attentions=None,
        )


class DummySequenceClassifier(DummySequenceCommon):
    """Dummy sequence classifier that outputs fixed logits per sequence."""

    config_class = DummySequenceClassConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        loss, logits = self.get_loss_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class DummyTokenCommon(PreTrainedModel):
    """Dummy token embeddings that outputs fixed logits for each token."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Create logit template with fixed values
        # First class: 2.0, Second class: 1.0, Rest: -4.0
        logits_template = torch.full((1, 1, config.num_labels), -4.0)
        logits_template[0, 0, 0] = 2.0
        if config.num_labels > 1:
            logits_template[0, 0, 1] = 1.0

        # Register as buffer so it moves with the model to GPU/CPU
        self.logits_template = logits_template
        self.register_buffer(
            "dummy", torch.tensor(0)
        )  # Dummy buffer to avoid empty model warning

    def get_loss_logits(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        """Forward pass that returns fixed logits for each token."""

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get batch size and sequence length from input_ids
        # Force use of attention mask so optimum does not drop it later
        batch_size, seq_length = (torch.mul(input_ids, attention_mask)).shape

        # Expand logits template for batch and sequence length
        logits = self.logits_template.expand(batch_size, seq_length, -1).clone()

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return loss, logits


class DummyTokenEmbeddings(DummyTokenCommon):
    """Dummy token embeddings that outputs fixed logits for each token."""

    config_class = DummyTokenEmbedConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        _, logits = self.get_loss_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        return BaseModelOutput(
            last_hidden_state=logits,
            hidden_states=None,
            attentions=None,
        )


class DummyTokenClassifier(DummyTokenCommon):
    """Dummy token classifier that outputs fixed logits per token."""

    config_class = DummyTokenClassConfig

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        loss, logits = self.get_loss_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def _create_and_save_model(
    model_class,
    config_class,
    tokenizer_name: str,
    output_dir: str,
    num_labels: Optional[int],
    model_name: str,
    auto_model_class,
    task: str,
):
    """
    Common helper to create, save, register, and export a dummy model to ONNX.

    Args:
        model_class: The model class to instantiate
        tokenizer_name: HuggingFace model ID for tokenizer
        output_dir: Directory to save the model
        num_labels: Number of classification labels
        model_name: Model identifier (e.g., DUMMY_SEQUENCE_ENCODER)
        auto_model_class: The AutoModel class for the task (e.g., AutoModelForSequenceClassification)
        task: The ONNX task name (e.g., "text-classification" or "token-classification")
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = config_class(num_labels=num_labels)

    print(f"Config: {config}")

    # Create and save model
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model = model_class(config)
    model.save_pretrained(output_dir)

    print(f"✓ Dummy {model_class.__name__} saved to {output_dir}")

    # Register model with transformers
    AutoConfig.register(model_name, config_class)
    auto_model_class.register(config_class, model_class)

    # Register ONNX configuration
    @register_tasks_manager_onnx(model_name, *COMMON_TEXT_TASKS)
    class DummyOnnxConfig(BertOnnxConfig):
        pass

    onnx_path = Path(output_dir) / Path("model.onnx")
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        "onnx",
        model,
        library_name="transformers",
        task=task,
    )
    onnx_config = onnx_config_constructor(model.config)
    _onnx_inputs, _onnx_outputs = export(model, onnx_config, onnx_path)
    print(f"✓ ONNX model exported to {onnx_path}")

    return model, tokenizer


def create_and_save_dummy_sequence_classifier(
    tokenizer_name: str,
    output_dir: str,
    num_labels: int,
):
    """
    Create and save a dummy sequence classification model.

    Args:
        tokenizer_name: HuggingFace model ID for tokenizer
        output_dir: Directory to save the model
        num_labels: Number of classification labels
    """
    return _create_and_save_model(
        model_class=DummySequenceClassifier,
        config_class=DummySequenceClassConfig,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        num_labels=num_labels,
        model_name=DUMMY_SEQUENCE_CLASSIFIER,
        auto_model_class=AutoModelForSequenceClassification,
        task="text-classification",
    )


def create_and_save_dummy_sequence_embeddings(
    tokenizer_name: str,
    output_dir: str,
    num_labels: int,
):
    """
    Create and save a dummy sequence embeddings model.

    Args:
        tokenizer_name: HuggingFace model ID for tokenizer
        output_dir: Directory to save the model
        num_labels: Number of classification labels
    """
    return _create_and_save_model(
        model_class=DummySequenceEmbeddings,
        config_class=DummySequenceEmbedConfig,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        num_labels=None,
        model_name=DUMMY_SEQUENCE_EMBEDDINGS,
        auto_model_class=AutoModel,
        task="feature-extraction",
    )


def create_and_save_dummy_token_classifier(
    tokenizer_name: str,
    output_dir: str,
    num_labels: int,
):
    """
    Create and save a dummy token classification model.

    Args:
        tokenizer_name: HuggingFace model ID for tokenizer
        output_dir: Directory to save the model
        num_labels: Number of classification labels
    """
    return _create_and_save_model(
        model_class=DummyTokenClassifier,
        config_class=DummyTokenClassConfig,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        num_labels=num_labels,
        model_name=DUMMY_TOKEN_CLASSIFIER,
        auto_model_class=AutoModelForTokenClassification,
        task="token-classification",
    )


def create_and_save_dummy_token_embeddings(
    tokenizer_name: str,
    output_dir: str,
    num_labels: int,
):
    """
    Create and save a dummy token embeddings model.

    Args:
        tokenizer_name: HuggingFace model ID for tokenizer
        output_dir: Directory to save the model
        num_labels: Number of classification labels
    """
    return _create_and_save_model(
        model_class=DummyTokenEmbeddings,
        config_class=DummyTokenEmbedConfig,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        num_labels=None,
        model_name=DUMMY_TOKEN_EMBEDDINGS,
        auto_model_class=AutoModel,
        task="feature-extraction",
    )


def load_dummy_sequence_classifier(model_dir: str, num_labels: int):
    """Load a dummy sequence classification model from directory."""
    AutoConfig.register(DUMMY_SEQUENCE_CLASSIFIER, DummySequenceClassConfig)
    AutoModelForSequenceClassification.register(
        DummySequenceClassConfig, DummySequenceClassifier
    )
    config = AutoConfig.from_pretrained(model_dir, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_dummy_sequence_embeddings(model_dir: str):
    """Load a dummy sequence classification model from directory."""
    AutoConfig.register(DUMMY_SEQUENCE_EMBEDDINGS, DummySequenceEmbedConfig)
    AutoModel.register(DummySequenceEmbedConfig, DummySequenceEmbeddings)
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_dummy_token_classifier(model_dir: str, num_labels: int):
    """Load a dummy token classification model from directory."""
    AutoConfig.register(DUMMY_TOKEN_CLASSIFIER, DummyTokenClassConfig)
    AutoModelForTokenClassification.register(
        DummyTokenClassConfig, DummyTokenClassifier
    )
    config = AutoConfig.from_pretrained(model_dir, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_dummy_token_embeddings(model_dir: str):
    """Load a dummy token embeddings model from directory."""
    AutoConfig.register(DUMMY_TOKEN_EMBEDDINGS, DummyTokenEmbedConfig)
    AutoModel.register(DummyTokenEmbedConfig, DummyTokenEmbeddings)
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def test_dummy_sequence_classifier(output_dir: str):
    """Test the dummy sequence classification model with sample inputs."""

    print("\n" + "=" * 50)
    print("Testing Dummy Sequence Classifier")
    print("=" * 50)

    # Create and save model (includes ONNX export)
    model, tokenizer = load_dummy_sequence_classifier(output_dir, num_labels=5)
    # Test with sample input
    text = "Hello, my dog is cute"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\nInput text: '{text}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits values: {logits}")

    predicted_class = logits.argmax(dim=-1).item()
    print(f"Predicted class: {predicted_class}")

    # Test with batch
    texts = ["Hello, my dog is cute", "I love this model"]
    inputs_batch = tokenizer(texts, return_tensors="pt", padding=True)

    print(f"\nBatch input shape: {inputs_batch['input_ids'].shape}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.logits

    print(f"Batch output logits:\n{logits_batch}")

    # Test loss computation
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        outputs_with_loss = model(**inputs_batch, labels=labels)
        loss = outputs_with_loss.loss

    print(f"\nLoss: {loss.item():.4f}")

    print("\n✓ Sequence classifier tests passed!")


def test_dummy_sequence_embeddings(output_dir: str):
    """Test the dummy sequence embeddings model with sample inputs."""

    print("\n" + "=" * 50)
    print("Testing Dummy Sequence Embeddings")
    print("=" * 50)

    # Create and save model (includes ONNX export)
    model, tokenizer = load_dummy_sequence_embeddings(output_dir)
    # Test with sample input
    text = "Hello, my dog is cute"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\nInput text: '{text}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.last_hidden_state

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits values: {logits}")

    # Test with batch
    texts = ["Hello, my dog is cute", "I love this model"]
    inputs_batch = tokenizer(texts, return_tensors="pt", padding=True)

    print(f"\nBatch input shape: {inputs_batch['input_ids'].shape}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.last_hidden_state

    print(f"Batch output logits:\n{logits_batch}")

    print("\n✓ Sequence embeddings tests passed!")


def test_dummy_token_classifier(output_dir: str):
    """Test the dummy token classification model with sample inputs."""

    print("\n" + "=" * 50)
    print("Testing Dummy Token Classifier")
    print("=" * 50)

    # Create and save model (includes ONNX export)
    model, tokenizer = load_dummy_token_classifier(output_dir, num_labels=5)

    # Test with sample input
    text = "Hello, my dog is cute"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\nInput text: '{text}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits first token: {logits[0, 0, :]}")

    predicted_classes = logits.argmax(dim=-1)
    print(f"Predicted classes per token: {predicted_classes}")

    # Test with batch
    texts = ["Hello, my dog is cute", "I love this model"]
    inputs_batch = tokenizer(texts, return_tensors="pt", padding=True)

    print(f"\nBatch input shape: {inputs_batch['input_ids'].shape}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.logits

    print(f"Batch output logits shape: {logits_batch.shape}")
    print(f"First sample, first token logits: {logits_batch[0, 0, :]}")

    # Test loss computation
    # Create dummy labels matching sequence length
    seq_len = inputs_batch["input_ids"].shape[1]
    labels = torch.zeros((2, seq_len), dtype=torch.long)
    labels[0, :3] = 1
    labels[1, :2] = 2

    with torch.no_grad():
        outputs_with_loss = model(**inputs_batch, labels=labels)
        loss = outputs_with_loss.loss

    print(f"\nLoss: {loss.item():.4f}")

    print("\n✓ Token classifier tests passed!")


def test_dummy_token_embeddings(output_dir: str):
    """Test the dummy token embeddings model with sample inputs."""

    print("\n" + "=" * 50)
    print("Testing Dummy Token Embeddings")
    print("=" * 50)

    # Create and save model (includes ONNX export)
    model, tokenizer = load_dummy_token_embeddings(output_dir)

    # Test with sample input
    text = "Hello, my dog is cute"
    inputs = tokenizer(text, return_tensors="pt")

    print(f"\nInput text: '{text}'")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.last_hidden_state

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Logits first token: {logits[0, 0, :]}")

    # Test with batch
    texts = ["Hello, my dog is cute", "I love this model"]
    inputs_batch = tokenizer(texts, return_tensors="pt", padding=True)

    print(f"\nBatch input shape: {inputs_batch['input_ids'].shape}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.last_hidden_state

    print(f"Batch output logits shape: {logits_batch.shape}")
    print(f"First sample, first token logits: {logits_batch[0, 0, :]}")

    # Test loss computation
    # Create dummy labels matching sequence length
    seq_len = inputs_batch["input_ids"].shape[1]
    labels = torch.zeros((2, seq_len), dtype=torch.long)
    labels[0, :3] = 1
    labels[1, :2] = 2

    print("\n✓ Token embeddings tests passed!")


# CLI Options
VALID_MODEL_TYPES = [
    "sequence_classifier",
    "token_classifier",
    "sequence_embeddings",
    "token_embeddings",
]

# Reusable option decorator for model class
model_type_option = click.option(
    "--model_type",
    "-m",
    type=click.Choice(VALID_MODEL_TYPES, case_sensitive=False),
    default="sequence_classifier",
    help="Type of model to generate/test: token or sequence",
)


@click.group()
def app():
    """Create and test dummy PyTorch models for CI/testing."""
    pass


@app.command()
@model_type_option
def generate(model_type: str):
    """Generate a dummy model and save it to disk."""
    if model_type == "sequence_classifier":
        create_and_save_dummy_sequence_classifier(
            tokenizer_name="google/electra-small-discriminator",
            output_dir=SEQUENCE_CLASSIFIER_OUTPUT_DIR,
            num_labels=7,
        )
    elif model_type == "token_classifier":
        create_and_save_dummy_token_classifier(
            tokenizer_name="google/electra-small-discriminator",
            output_dir=TOKEN_CLASSIFIER_OUTPUT_DIR,
            num_labels=7,
        )
    elif model_type == "sequence_embeddings":
        create_and_save_dummy_sequence_embeddings(
            tokenizer_name="google/electra-small-discriminator",
            output_dir=SEQUENCE_EMBEDDINGS_OUTPUT_DIR,
            num_labels=7,
        )
    elif model_type == "token_embeddings":
        create_and_save_dummy_token_embeddings(
            tokenizer_name="google/electra-small-discriminator",
            output_dir=TOKEN_EMBEDDINGS_OUTPUT_DIR,
            num_labels=7,
        )


@app.command()
@model_type_option
def test(model_type: str):
    """Test a dummy model from disk."""
    if model_type == "sequence_classifier":
        output_dir = SEQUENCE_CLASSIFIER_OUTPUT_DIR
        test_dummy_sequence_classifier(output_dir)
    elif model_type == "token_classifier":
        output_dir = TOKEN_CLASSIFIER_OUTPUT_DIR
        test_dummy_token_classifier(output_dir)
    elif model_type == "sequence_embeddings":
        output_dir = SEQUENCE_EMBEDDINGS_OUTPUT_DIR
        test_dummy_sequence_embeddings(output_dir)
    elif model_type == "token_embeddings":
        output_dir = TOKEN_EMBEDDINGS_OUTPUT_DIR
        test_dummy_token_embeddings(output_dir)


if __name__ == "__main__":
    app()

from onnx import ModelProto
import onnx

from .enums import ModelType

def validate_embedding_model(model: ModelProto):
    """Validate embedding model."""

    # must output last hidden state
    if all(outp.name != "last_hidden_state" for outp in model.graph.output):
        raise ValueError("Model must return last_hidden_state")

    # get last_hidden_state
    lhs = next(
        filter(
            lambda m: m.name == "last_hidden_state",
            model.graph.output
            )
        )

    # make sure returns matrix of rank 3
    if len(lhs.type.tensor_type.shape.dim) != 3:
        raise ValueError("Model must return tensor of shape [batch_size, sequence_length, hidden_state_size]")

def validate_sequence_classification_model(model: ModelProto):
    """Validate sequence classification model."""

    # must output logits
    if all(outp.name != "logits" for outp in model.graph.output):
        raise ValueError("Model must return logits")
    
    # get logits
    logits = next(
        filter(
            lambda m: m.name == "logits",
            model.graph.output
        )
    )

    # must return matrix of rank 2
    if len(logits.type.tensor_type.shape.dim) != 2:
        raise ValueError("Model must return tensor of shape [batch_size, n_labels]")

def validate_token_classification_model(model: ModelProto):
    """Validate token classification model."""
    # must output logits
    if all(outp.name != "logits" for outp in model.graph.output):
        raise ValueError("Model must return logits")
    
    # get logits
    logits = next(
        filter(
            lambda m: m.name == "logits",
            model.graph.output
        )
    )

    # must return matrix of rank 3
    if len(logits.type.tensor_type.shape.dim) != 3:
        raise ValueError("Model must return tensor of shape [batch_size, n_tokens, n_labels]")

def validate_model(model_weights_path: str, model_type: ModelType):
    model = onnx.load(model_weights_path)

    match model_type:
        case ModelType.EMBEDDING:
            validate_embedding_model(model)
        case ModelType.SEQUENCE_CLASSIFICATION:
            validate_sequence_classification_model(model)
        case ModelType.TOKEN_CLASSIFICATION:
            validate_token_classification_model(model)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

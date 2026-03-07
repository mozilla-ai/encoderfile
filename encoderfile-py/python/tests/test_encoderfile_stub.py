import pytest
from encoderfile import (
    EncoderfileBuilder,
    ModelConfig,
    EncoderfileConfig,
    InspectInfo,
    inspect,
    ModelType,
)
from conftest import asset_path, load_yaml_asset


def test_encoderfilebuilder_from_configpath_returns_builder():
    config_path = asset_path("test_config.yml")
    builder = EncoderfileBuilder.from_configpath(config_path)
    assert isinstance(builder, EncoderfileBuilder)


def test_encoderfilebuilder_from_config_fails():
    config = {
      "name": "my-model-2",
      "path": "models/token_classification",
      "model_type": "whatever",
      "output_path": "./test-model.encoderfile",
      "transform": """
        --- Applies a softmax across token classification logits.
        --- Each token classification is normalized independently.
        --- 
        --- Args:
        ---   arr (Tensor): A tensor of shape [batch_size, n_tokens, n_labels].
        ---                 The softmax is applied along the third axis (n_labels).
        ---
        --- Returns:
        ---   Tensor: The input tensor with softmax-normalized embeddings.
        ---@param arr Tensor
        ---@return Tensor
        function Postprocess(arr)
            return arr:softmax(3)
        end
        """
    }
    with pytest.raises(RuntimeError)as exc_info:
        builder = EncoderfileBuilder.from_config(**config)
    assert "Invalid model type" in str(exc_info.value)

def test_encoderfilebuilder_from_config_returns_builder():
    config = {
      "name": "my-model-2",
      "path": "models/token_classification",
      "model_type": ModelType.TokenClassification,
      "output_path": "./test-model.encoderfile",
      "transform": """
        --- Applies a softmax across token classification logits.
        --- Each token classification is normalized independently.
        --- 
        --- Args:
        ---   arr (Tensor): A tensor of shape [batch_size, n_tokens, n_labels].
        ---                 The softmax is applied along the third axis (n_labels).
        ---
        --- Returns:
        ---   Tensor: The input tensor with softmax-normalized embeddings.
        ---@param arr Tensor
        ---@return Tensor
        function Postprocess(arr)
            return arr:softmax(3)
        end
        """
    }
    builder = EncoderfileBuilder.from_config(**config)
    assert isinstance(builder, EncoderfileBuilder)

@pytest.mark.parametrize("config_filename", ["test_config.yml", "test_config_lua.yml"])
def test_encoderfilebuilder_build_runs(config_filename):
    config_path = asset_path(config_filename)
    config_info = load_yaml_asset(config_filename)
    builder = EncoderfileBuilder.from_configpath(config_path)
    # Should not raise
    builder.build(working_dir=None, version=None, no_download=True)
    result = inspect(config_info["encoderfile"]["output_path"])
    assert isinstance(result, InspectInfo)
    assert isinstance(result.model_config, ModelConfig)
    assert isinstance(result.encoderfile_config, EncoderfileConfig)
    print(result.model_config)
    print(result.encoderfile_config)
    assert result.encoderfile_config.name == config_info["encoderfile"]["name"]
    assert (
        result.encoderfile_config.transform.strip()
        == config_info["encoderfile"].get("transform").strip()
    )
    assert result.encoderfile_config.lua_libs == config_info["encoderfile"].get(
        "lua_libs"
    )

def test_encoderfilebuilder_build_from_config():
    name = "model-built-from-config"
    config = {
      "name": name,
      "path": "models/token_classification",
      "model_type": ModelType.TokenClassification,
      "output_path": f"./{name}.encoderfile",
      "transform": """
        --- No docs
        ---
        --- Args:
        ---   arr (Tensor): A tensor of shape [batch_size, n_tokens, n_labels].
        ---                 The softmax is applied along the third axis (n_labels).
        ---
        --- Returns:
        ---   Tensor: The input tensor with softmax-normalized embeddings.
        ---@param arr Tensor
        ---@return Tensor
        function Postprocess(arr)
            return arr:softmax(3)
        end
        """
    }
    builder = EncoderfileBuilder.from_config(**config)
    # Should not raise
    builder.build(working_dir=None, version=None, no_download=True)
    result = inspect(config["output_path"])
    assert isinstance(result, InspectInfo)
    assert isinstance(result.model_config, ModelConfig)
    assert isinstance(result.encoderfile_config, EncoderfileConfig)
    print(result.model_config)
    print(result.encoderfile_config)
    assert result.encoderfile_config.name == config["name"]
    assert (
        result.encoderfile_config.transform.strip()
        == config.get("transform").strip()
    )
    assert result.encoderfile_config.lua_libs == config.get("lua_libs")

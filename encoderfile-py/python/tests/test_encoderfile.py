import pytest
from encoderfile import (
    EncoderfileBuilder,
    ModelConfig,
    EncoderfileConfig,
    InspectInfo,
    inspect,
    ModelType,
    TokenizerBuildConfig,
    TargetSpec,
)
from pathlib import Path
from conftest import asset_path, load_yaml_asset, load_json


def test_encoderfilebuilder_from_configpath_returns_builder():
    config_path = asset_path("test_config.yml")
    builder = EncoderfileBuilder.from_config(config_path)
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
        """,
    }
    with pytest.raises(RuntimeError) as exc_info:
        EncoderfileBuilder.from_dict(**config)
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
        """,
    }
    builder = EncoderfileBuilder.from_dict(**config)
    assert isinstance(builder, EncoderfileBuilder)


@pytest.mark.parametrize("config_filename", ["test_config.yml", "test_config_lua.yml"])
def test_encoderfilebuilder_build_runs(config_filename):
    config_path = asset_path(config_filename)
    config_info = load_yaml_asset(config_filename)
    builder = EncoderfileBuilder.from_config(config_path)
    # Should not raise
    builder.build(workdir=None, version=None, no_download=True)
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


def test_encoderfilebuilder_build_from_dict(tmp_path):
    name = "model-built-from-config"
    config = {
        "name": name,
        "path": "models/token_classification",
        "model_type": ModelType.TokenClassification,
        "output_path": str(tmp_path / f"{name}.encoderfile"),
        "tokenizer": TokenizerBuildConfig.new(
            pad_strategy="batch_longest",
        ),
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
        """,
    }
    model_info = load_json(Path(config["path"]) / "config.json")
    print(model_info)
    builder = EncoderfileBuilder.from_dict(**config)
    # Should not raise
    builder.build(workdir=None, version=None, no_download=True)
    result = inspect(config["output_path"])
    assert isinstance(result, InspectInfo)
    assert isinstance(result.model_config, ModelConfig)
    assert isinstance(result.encoderfile_config, EncoderfileConfig)
    assert result.encoderfile_config.name == config["name"]
    assert (
        result.encoderfile_config.transform.strip() == config.get("transform").strip()
    )
    assert result.encoderfile_config.lua_libs == config.get("lua_libs")
    assert result.model_config.num_labels == model_info.get(
        "num_labels"
    ) or result.model_config.num_labels == len(model_info["id2label"])
    # FIXME ints are getting interpreted as... ints (or at least not as strings)
    # assert result.model_config.id2label == model_info["id2label"]
    # assert result.model_config.label2id == model_info["label2id"]
    assert result.model_config.model_type == model_info.get("model_type")


def test_encoderfilebuilder_build_all_from_dict(tmp_path):
    name = "model-built-from-config-all"
    config = {
        "name": name,
        "version": "0.2.0",
        "path": "models/token_classification",
        "model_type": ModelType.TokenClassification,
        "output_path": str(tmp_path / f"{name}.encoderfile"),
        "cache_dir": "./cache",
        "base_binary_path": "./target/debug/encoderfile-runtime",
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
        """,
        "lua_libs": ["table", "math"],
        "tokenizer": TokenizerBuildConfig.new(
            pad_strategy="batch_longest",
            truncation_side="right",
            truncation_strategy="longest_first",
            max_length=512,
            stride=0,
        ),
        "validate_transform": False,
        "target": "x86_64-unknown-linux-musl",
    }
    builder = EncoderfileBuilder.from_dict(**config)
    builder.build(workdir=None, version=None, no_download=True)
    result = inspect(config["output_path"])
    assert isinstance(result, InspectInfo)
    assert isinstance(result.model_config, ModelConfig)
    assert isinstance(result.encoderfile_config, EncoderfileConfig)
    assert result.encoderfile_config.name == config["name"]
    assert result.encoderfile_config.version == config["version"]
    assert result.encoderfile_config.model_type == config["model_type"].value
    assert (
        result.encoderfile_config.transform.strip() == config.get("transform").strip()
    )
    assert result.encoderfile_config.lua_libs == config.get("lua_libs")


def test_encoderfilebuilder_wrong_arch(tmp_path):
    name = "model-built-from-config"
    config = {
        "name": name,
        "path": "models/token_classification",
        "model_type": ModelType.TokenClassification,
        "output_path": str(tmp_path / f"{name}.encoderfile"),
        "tokenizer": TokenizerBuildConfig.new(
            pad_strategy="batch_longest",
        ),
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
        """,
        "target": "nonsense!",
    }
    builder = EncoderfileBuilder.from_dict(**config)
    with pytest.raises(RuntimeError) as exc_info:
        builder.build(workdir=None, version=None, no_download=True)
    assert (
        f"Error building encoderfile: invalid or unsupported target triple `{config['target']}`"
        in str(exc_info.value)
    )


def test_tokenizer_build_config_from_dict():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": 512,
        "stride": 0,
    }
    tokenizer_config = TokenizerBuildConfig.new(**config)
    assert tokenizer_config.pad_strategy == config["pad_strategy"]
    assert tokenizer_config.truncation_side == config["truncation_side"]
    assert tokenizer_config.truncation_strategy == config["truncation_strategy"]
    assert tokenizer_config.max_length == config["max_length"]
    assert tokenizer_config.stride == config["stride"]


def test_tokenizer_wrong_config_pad_strategy():
    config = {
        "pad_strategy": "nonsense!",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": 512,
        "stride": 0,
    }
    with pytest.raises(RuntimeError) as exc_info:
        TokenizerBuildConfig.new(**config)
    assert f"Invalid pad strategy: {config['pad_strategy']}" in str(exc_info.value)


def test_tokenizer_wrong_config_truncation_side():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "nonsense!",
        "truncation_strategy": "longest_first",
        "max_length": 512,
        "stride": 0,
    }
    with pytest.raises(RuntimeError) as exc_info:
        TokenizerBuildConfig.new(**config)
    assert f"Invalid truncation side: {config['truncation_side']}" in str(
        exc_info.value
    )


def test_tokenizer_wrong_config_truncation_strategy():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "nonsense!",
        "max_length": 512,
        "stride": 0,
    }
    with pytest.raises(RuntimeError) as exc_info:
        TokenizerBuildConfig.new(**config)
    assert f"Invalid truncation strategy: {config['truncation_strategy']}" in str(
        exc_info.value
    )


def test_tokenizer_wrong_config_max_length():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": -1,
        "stride": 0,
    }
    with pytest.raises(OverflowError):
        TokenizerBuildConfig.new(**config)


def test_tokenizer_wrong_type_max_length():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": "nonsense!",
        "stride": 0,
    }
    with pytest.raises(TypeError) as exc_info:
        TokenizerBuildConfig.new(**config)
    assert "'str' object cannot be interpreted as an integer" in str(exc_info.value)


def test_tokenizer_wrong_config_stride():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": 512,
        "stride": -1,
    }
    with pytest.raises(OverflowError):
        TokenizerBuildConfig.new(**config)


def test_tokenizer_wrong_type_stride():
    config = {
        "pad_strategy": "batch_longest",
        "truncation_side": "right",
        "truncation_strategy": "longest_first",
        "max_length": 512,
        "stride": "nonsense!",
    }
    with pytest.raises(TypeError) as exc_info:
        TokenizerBuildConfig.new(**config)
    assert "'str' object cannot be interpreted as an integer" in str(exc_info.value)


def test_tokenizer_config_optional_fields():
    config = {
        "pad_strategy": None,
        "truncation_side": None,
        "truncation_strategy": None,
        "max_length": None,
        "stride": None,
    }
    tokenizer_config = TokenizerBuildConfig.new(**config)
    assert tokenizer_config.pad_strategy is None
    assert tokenizer_config.truncation_side is None
    assert tokenizer_config.truncation_strategy is None
    assert tokenizer_config.max_length is None
    assert tokenizer_config.stride is None


def test_tokenizer_config_all_optional_fields():
    config = {}
    tokenizer_config = TokenizerBuildConfig.new(**config)
    assert tokenizer_config.pad_strategy is None
    assert tokenizer_config.truncation_side is None
    assert tokenizer_config.truncation_strategy is None
    assert tokenizer_config.max_length is None
    assert tokenizer_config.stride is None


def test_tokenizer_config_partial_optional_fields():
    config = {
        "pad_strategy": "batch_longest",
        "max_length": 512,
    }
    tokenizer_config = TokenizerBuildConfig.new(**config)
    assert tokenizer_config.pad_strategy == config["pad_strategy"]
    assert tokenizer_config.truncation_side is None
    assert tokenizer_config.truncation_strategy is None
    assert tokenizer_config.max_length == config["max_length"]
    assert tokenizer_config.stride is None


def test_parse_target_spec_valid_1():
    spec_str = "x86_64-unknown-linux-musl"
    target_spec = TargetSpec.parse(spec_str)
    assert target_spec.arch == "x86_64"
    assert target_spec.os == "linux"
    assert target_spec.abi == "musl"


def test_parse_target_spec_valid_2():
    spec_str = "aarch64-apple-darwin"
    target_spec = TargetSpec.parse(spec_str)
    assert target_spec.arch == "aarch64"
    assert target_spec.os == "darwin"
    assert target_spec.abi == "gnu"


def test_parse_target_spec_valid_3():
    spec_str = "x86_64-pc-windows-msvc"
    target_spec = TargetSpec.parse(spec_str)
    assert target_spec.arch == "x86_64"
    assert target_spec.os == "windows"
    assert target_spec.abi == "msvc"


def test_parse_target_spec_invalid_format():
    spec_str = "nonsense!"
    with pytest.raises(ValueError) as exc_info:
        TargetSpec.parse(spec_str)
    assert (
        "Failed to parse target spec: invalid or unsupported target triple `nonsense!`"
        in str(exc_info.value)
    )


def test_parse_target_spec_unsupported_arch():
    spec_str = "riscv64-unknown-linux-musl"
    with pytest.raises(ValueError) as exc_info:
        TargetSpec.parse(spec_str)
    assert "Failed to parse target spec: unsupported architecture `riscv64`" in str(
        exc_info.value
    )

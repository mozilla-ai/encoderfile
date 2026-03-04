import pytest
from encoderfile import (
    EncoderfileBuilder,
    ModelConfig,
    EncoderfileConfig,
    InspectInfo,
    inspect,
)
from conftest import asset_path, load_yaml_asset


def test_encoderfilebuilder_from_config_returns_builder():
    config_path = asset_path("test_config.yml")
    builder = EncoderfileBuilder.from_config(config_path)
    assert isinstance(builder, EncoderfileBuilder)


@pytest.mark.parametrize("config_filename", ["test_config.yml", "test_config_lua.yml"])
def test_encoderfilebuilder_build_runs(config_filename):
    config_path = asset_path(config_filename)
    config_info = load_yaml_asset(config_filename)
    builder = EncoderfileBuilder.from_config(config_path)
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

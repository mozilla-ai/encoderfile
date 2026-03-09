from typing import Optional, final
from .enums import ModelType

@final
class EncoderfileBuilder:
    @staticmethod
    def from_config(
        config: str,
    ) -> "EncoderfileBuilder": ...
    @staticmethod
    def from_dict(
        *,
        name: str,
        version: Optional[str] = "0.1.0",
        model_type: ModelType,
        path: str,
        output_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        base_binary_path: Optional[str] = None,
        transform: Optional[str] = None,
        lua_libs: Optional[list[str]] = None,
        tokenizer: Optional[TokenizerBuildConfig] = None,
        validate_transform: bool = True,
        target: Optional[str] = None,
    ) -> "EncoderfileBuilder": ...
    def build(
        self,
        workdir: Optional[str] = None,
        version: Optional[str] = None,
        no_download: bool = False,
    ): ...

@final
class TokenizerBuildConfig:
    @staticmethod
    def new(
        *,
        pad_strategy: Optional[str] = None,
        truncation_side: Optional[str] = None,
        truncation_strategy: Optional[str] = None,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> TokenizerBuildConfig: ...

    pad_strategy: Optional[str]
    truncation_side: Optional[str]
    truncation_strategy: Optional[str]
    max_length: Optional[int]
    stride: Optional[int]

@final
class ModelConfig:
    model_type: str
    num_labels: Optional[int]
    id2label: Optional[dict[int, str]]
    label2id: Optional[dict[str, int]]

@final
class EncoderfileConfig:
    name: str
    version: str
    model_type: str
    transform: Optional[str]
    lua_libs: Optional[list[str]]

@final
class InspectInfo:
    model_config: ModelConfig
    encoderfile_config: EncoderfileConfig

def inspect(path: str) -> InspectInfo: ...

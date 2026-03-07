from typing import Optional, final
from .enums import ModelType


@final
class EncoderfileBuilder:
    @classmethod
    def from_configpath(
        cls,
        config: str,
    ) -> "EncoderfileBuilder": ...

    @classmethod
    def from_config(
        cls,
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
        # not yet supported
        # tokenizer: Optional[str],
        validate_transform: bool = True,
        target: Optional[str] = None,
    ) -> "EncoderfileBuilder": ...

    def build(
        self,
        working_dir: Optional[str] = None,
        version: Optional[str] = None,
        no_download: bool = False,
    ): ...

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

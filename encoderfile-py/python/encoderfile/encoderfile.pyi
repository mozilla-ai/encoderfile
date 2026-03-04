from typing import Optional, final

@final
class EncoderfileBuilder:
    @classmethod
    def from_config(
        cls,
        config: str,
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

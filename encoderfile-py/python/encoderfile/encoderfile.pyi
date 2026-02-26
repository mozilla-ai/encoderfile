from typing import Optional

class EncoderfileBuilder:
    @classmethod
    def from_config(
        cls,
        config: str,
        output_path: Optional[str] = None,
        base_binary_path: Optional[str] = None,
        platform: Optional[str] = None,
        version: Optional[str] = None,
        no_download: bool = False,
        directory: Optional[str] = None,
    ) -> "EncoderfileBuilder": ...
    def build(self, cache_dir: Optional[str] = None): ...

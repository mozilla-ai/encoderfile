from typing import Optional, Literal

from .enums import ModelType, TokenizerTruncationSide, TokenizerTruncationStrategy
from ._core import (
    TargetSpec,
    EncoderfileBuilder,
    TokenizerBuildConfig,
    Fixed,
    BatchLongest,
)


def build_from_config(
    config_path: str, workdir: Optional[str] = None, no_download: bool = False
):
    """
    Build an encoderfile binary from a YAML configuration file.

    A convenience wrapper around :class:`EncoderfileBuilder` for the common
    case where build settings are defined in a config file. For programmatic
    configuration, use :func:`build` or :class:`EncoderfileBuilder` directly.

    Args:
        config_path: Path to the YAML build configuration file.
        workdir: Temporary working directory for intermediate build files.
            Defaults to a system temp directory.
        no_download: When ``True``, disables downloading the base binary
            from the network. A local base binary must be available via
            ``base_binary_path`` in the config or the cache.

    Raises:
        FileNotFoundError: If ``config_path`` or any referenced model files
            do not exist.
        ValueError: If the configuration file is invalid or the model is
            incompatible.
    """
    builder = EncoderfileBuilder.from_config(config_path)

    builder.build(workdir=workdir, no_download=no_download)


def build(
    *,
    name: str,
    version: Optional[str] = None,
    model_type: ModelType | str,
    path: str,
    output_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    base_binary_path: Optional[str] = None,
    transform_str: Optional[str] = None,
    transform_path: Optional[str] = None,
    lua_libs: Optional[list[str]] = None,
    tokenizer_pad_to: Optional[Literal["batch_longest"] | int] = None,
    tokenizer_truncation_side: Optional[str | TokenizerTruncationSide] = None,
    tokenizer_truncation_strategy: Optional[str | TokenizerTruncationStrategy] = None,
    tokenizer_max_length: Optional[int] = None,
    tokenizer_stride: Optional[int] = None,
    validate_transform: bool = True,
    target: Optional[str | TargetSpec] = None,
    workdir: Optional[str] = None,
    no_download: bool = False,
):
    """
    Build an encoderfile binary from explicit configuration.

    A convenience wrapper around :class:`EncoderfileBuilder` with a flat
    argument structure that avoids importing supporting classes for common
    use cases. For full control over the build process, use
    :class:`EncoderfileBuilder` directly.

    Args:
        name: Model identifier used in API responses and as the default
            output filename when ``output_path`` is not set.
        version: Model version string. Defaults to ``"0.1.0"``.
        model_type: Architecture of the model. Accepts a :class:`ModelType`
            enum value or a plain string (e.g. ``"sequence_classification"``).
        path: Path to a directory containing ``model.onnx``,
            ``tokenizer.json``, and ``config.json``.
        output_path: Destination path for the compiled binary. Defaults
            to ``./<name>.encoderfile`` in the current directory.
        cache_dir: Directory used for caching intermediate build artifacts.
            Defaults to the system cache directory.
        base_binary_path: Path to a local pre-built base binary. When
            provided, skips downloading the base binary from the network.
        transform_str: Inline Lua post-processing script applied to model
            logits before returning results. Mutually exclusive with
            ``transform_path``.
        transform_path: Path to a Lua post-processing script. The file
            contents are read and embedded at build time. Mutually
            exclusive with ``transform_str``.
        lua_libs: Additional Lua library paths made available to the
            transform script at runtime.
        tokenizer_pad_to: Padding strategy. Pass ``"batch_longest"`` to
            pad to the longest sequence in each batch, or an ``int`` to
            pad all sequences to a fixed length.
        tokenizer_truncation_side: Side from which to truncate long
            sequences (e.g. ``"left"`` or ``"right"``).
        tokenizer_truncation_strategy: Algorithm used to select which
            tokens to remove when truncating (e.g. ``"longest_first"``).
        tokenizer_max_length: Maximum number of tokens per sequence.
            Sequences longer than this value are truncated.
        tokenizer_stride: Number of overlapping tokens between consecutive
            chunks when a sequence is split due to ``tokenizer_max_length``.
        validate_transform: Whether to perform a dry-run validation of the
            transform script before building. Defaults to ``True``.
        target: Target platform triple for cross-compilation, supplied
            either as a ``"<arch>-<os>-<abi>"`` string or a
            :class:`TargetSpec` instance. Defaults to the host machine's
            architecture.
        workdir: Temporary working directory for intermediate build files.
            Defaults to a system temp directory.
        no_download: When ``True``, disables downloading the base binary
            from the network. A local base binary must be available via
            ``base_binary_path`` or the cache.

    Raises:
        FileNotFoundError: If ``path`` or any referenced model files do
            not exist.
        ValueError: If ``transform_str`` and ``transform_path`` are both
            provided, if ``tokenizer_pad_to`` is invalid, or if the model
            is incompatible.
    """

    # transform
    if transform_str is not None and transform_path is not None:
        raise ValueError("Only one of transform_str, transform_path is allowed")
    elif transform_path is not None:
        with open(transform_path) as f:
            transform = f.read()
    else:
        transform = transform_str  # either a string or None, both valid

    # tokenizer build config
    if all(
        i is None
        for i in [
            tokenizer_pad_to,
            tokenizer_truncation_side,
            tokenizer_truncation_strategy,
            tokenizer_max_length,
            tokenizer_stride,
        ]
    ):
        tokenizer = None
    else:
        if tokenizer_pad_to is None:
            pad_strategy = None
        elif isinstance(tokenizer_pad_to, int):
            pad_strategy = Fixed(n=tokenizer_pad_to)
        elif tokenizer_pad_to == "batch_longest":
            pad_strategy = BatchLongest()
        else:
            raise ValueError(
                'tokenizer_pad_to must be either an int for fixed padding or `"batch_longest"` for dynamic padding'
            )

        tokenizer = TokenizerBuildConfig(
            pad_strategy=pad_strategy,
            truncation_side=tokenizer_truncation_side,
            truncation_strategy=tokenizer_truncation_strategy,
            max_length=tokenizer_max_length,
            stride=tokenizer_stride,
        )

    builder = EncoderfileBuilder(
        name=name,
        version=version,
        model_type=model_type,
        path=path,
        output_path=output_path,
        cache_dir=cache_dir,
        base_binary_path=base_binary_path,
        transform=transform,
        lua_libs=lua_libs,
        tokenizer=tokenizer,
        validate_transform=validate_transform,
        target=target,
    )

    builder.build(workdir=workdir, no_download=no_download)

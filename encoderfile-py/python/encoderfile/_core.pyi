from typing import Optional, final

from .enums import ModelType

@final
class TargetSpec:
    """
    Represents a compilation target platform for building encoderfile binaries.

    Attributes:
        arch: CPU architecture (e.g., ``"aarch64"``, ``"x86_64"``).
        os: Operating system (e.g., ``"apple"``, ``"unknown-linux"``).
        abi: ABI/environment suffix (e.g., ``"darwin"``, ``"gnu"``).
    """

    arch: str
    os: str
    abi: str
    def __new__(cls, spec: str):
        """
        Parse a target triple string into a ``TargetSpec``.

        Args:
            spec: A Rust-style target triple such as ``"aarch64-apple-darwin"``
                or ``"x86_64-unknown-linux-gnu"``. Equivalent to Cargo's
                ``--target`` flag.
        """
        ...

@final
class EncoderfileBuilder:
    """
    Builds a self-contained encoderfile binary from an ONNX model.

    The builder validates model files, embeds ONNX weights, tokenizer
    configuration, and model metadata into a pre-built base binary, then
    writes the result to the configured output path.

    Typical usage::

        builder = EncoderfileBuilder(
            name="sentiment-analyzer",
            model_type=ModelType.SequenceClassification,
            path="./models/distilbert-sst2",
            output_path="./sentiment-analyzer.encoderfile",
        )
        builder.build()

    Alternatively, load configuration from a YAML file::

        builder = EncoderfileBuilder.from_config("sentiment-config.yml")
        builder.build()
    """

    @staticmethod
    def from_config(
        config_path: str,
    ) -> "EncoderfileBuilder":
        """
        Create an ``EncoderfileBuilder`` from a YAML configuration file.

        The YAML file must contain an ``encoderfile`` top-level key whose
        value matches the ``EncoderfileConfig`` schema.  Example::

            encoderfile:
              name: sentiment-analyzer
              version: "1.0.0"
              path: ./models/distilbert-sst2
              model_type: sequence_classification
              output_path: ./build/sentiment-analyzer.encoderfile

        Args:
            config_path: Path to the YAML build configuration file.

        Returns:
            A configured ``EncoderfileBuilder`` instance.

        Raises:
            ValueError: If the configuration file is missing required fields
                or contains invalid values.
            FileNotFoundError: If ``config_path`` does not exist.
        """
        ...
    def __new__(
        cls,
        *,
        name: str,
        version: Optional[str] = None,
        model_type: ModelType | str,
        path: str,
        output_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        base_binary_path: Optional[str] = None,
        transform: Optional[str] = None,
        lua_libs: Optional[list[str]] = None,
        tokenizer: Optional[TokenizerBuildConfig] = None,
        validate_transform: bool = True,
        target: Optional[str | TargetSpec] = None,
    ) -> "EncoderfileBuilder":
        """
        Create an ``EncoderfileBuilder`` with explicit configuration.

        Args:
            name: Model identifier used in API responses and as the default
                output filename when ``output_path`` is not set.
            version: Model version string. Defaults to ``"0.1.0"``.
            model_type: Architecture of the model. Determines how inference
                outputs are structured (embeddings, sequence labels, or token
                labels).
            path: Path to a directory containing ``model.onnx``,
                ``tokenizer.json``, and ``config.json``, or an explicit
                mapping of individual file paths.
            output_path: Destination path for the compiled binary. Defaults
                to ``./<name>.encoderfile`` in the current directory.
            cache_dir: Directory used for caching intermediate build
                artifacts. Defaults to the system cache directory.
            base_binary_path: Path to a local pre-built base binary. When
                provided, skips downloading the base binary from the network.
            transform: Lua post-processing script applied to model logits
                before returning results. May be an inline Lua string or a
                file path.  Example::

                    "function Postprocess(logits) return logits:lp_normalize(2.0, 2.0) end"

            lua_libs: Additional Lua library paths made available to the
                transform script at runtime.
            tokenizer: Tokenizer padding and truncation settings. Any values provided here will
                override any settings found in `tokenizer_config.json` or `tokenizer.json`.
                When ``None``, the tokenizer uses its default configuration.
            validate_transform: Whether to perform a dry-run validation of
                the transform script before building. Defaults to ``True``.
            target: Target platform triple for cross-compilation, supplied
                either as a ``"<arch>-<os>-<abi>"`` string or a
                ``TargetSpec`` instance. Defaults to the host machine's
                architecture.

        Returns:
            A configured ``EncoderfileBuilder`` instance ready to call
            :meth:`build`.
        """
        ...

    def build(
        self,
        workdir: Optional[str] = None,
        runtime_version: Optional[str] = None,
        no_download: bool = False,
    ):
        """
        Compile and write the encoderfile binary.

        Performs the following steps:

        1. Validates model files (``model.onnx``, ``tokenizer.json``,
           ``config.json``).
        2. Validates the ONNX model structure and compatibility.
        3. Optionally validates the Lua transform with a dry run.
        4. Embeds all assets into the base binary.
        5. Writes the finished binary to ``output_path``.

        Args:
            workdir: Temporary working directory for intermediate build
                files. Defaults to a system temp directory.
            runtime_version: Override the encoderfile runtime version to embed.
                Takes precedence over the version set on the builder.
            no_download: When ``True``, disables downloading the base
                binary from the network. A local base binary must be
                available via ``base_binary_path`` or the cache.

        Raises:
            FileNotFoundError: If required model files are missing.
            ValueError: If the ONNX model structure is incompatible or
                the transform script fails validation.
            RuntimeError: If the binary cannot be written to
                ``output_path``.
        """
        ...

@final
class BatchLongest:
    """
    Pad all sequences in a batch to the length of the longest sequence.

    Use this as the ``pad_strategy`` on :class:`TokenizerBuildConfig` when
    you want dynamic, batch-relative padding rather than a fixed sequence
    length.
    """

    pass

@final
class Fixed:
    """
    Pad all sequences to a fixed sequence length.

    Attributes:
        n: The fixed number of tokens every sequence will be padded or
           truncated to.
    """

    n: int

    def __new__(cls, *, n: int) -> "Fixed": ...

@final
class TokenizerBuildConfig:
    """
    Tokenizer padding and truncation settings embedded at build time.

    These settings are baked into the encoderfile binary and applied
    consistently at inference time without requiring runtime configuration.

    Attributes:
        pad_strategy: How sequences are padded.  ``BatchLongest`` pads to
            the longest sequence in each batch; ``Fixed(n=N)`` pads every
            sequence to exactly ``N`` tokens.  ``None`` uses the
            tokenizer's default padding behaviour.
        truncation_side: Which side to truncate from when a sequence
            exceeds ``max_length``. Typically ``"left"`` or ``"right"``.
        truncation_strategy: Strategy used when truncating sequences that
            exceed ``max_length`` (e.g. ``"longest_first"``).
        max_length: Maximum number of tokens per sequence. Sequences
            longer than this value are truncated.
        stride: Number of overlapping tokens between consecutive chunks
            when a sequence is split due to ``max_length``.
    """

    pad_strategy: Optional[BatchLongest | Fixed]
    truncation_side: Optional[str]
    truncation_strategy: Optional[str]
    max_length: Optional[int]
    stride: Optional[int]

    def __new__(
        cls,
        *,
        pad_strategy: Optional[BatchLongest | Fixed] = None,
        truncation_side: Optional[str] = None,
        truncation_strategy: Optional[str] = None,
        max_length: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> "TokenizerBuildConfig":
        """
        Args:
            pad_strategy: Padding strategy. Pass ``BatchLongest()`` for
                dynamic batch padding or ``Fixed(n=512)`` for a fixed
                sequence length.
            truncation_side: Side from which to truncate long sequences.
            truncation_strategy: Algorithm used to select which tokens to
                remove when truncating.
            max_length: Hard cap on sequence length in tokens.
            stride: Token overlap between sequence chunks produced by
                sliding-window truncation.
        """
        ...

@final
class ModelConfig:
    """
    Model architecture metadata extracted from the embedded ``config.json``.

    Attributes:
        model_type: Architecture identifier as it appears in the HuggingFace
            config (e.g. ``"bert"``, ``"distilbert"``).
        num_labels: Number of output labels for classification models.
            ``None`` for embedding models.
        id2label: Mapping from integer label index to label string
            (e.g. ``{0: "NEGATIVE", 1: "POSITIVE"}``). ``None`` when not
            present in the model config.
        label2id: Reverse mapping from label string to integer index.
            ``None`` when not present in the model config.
    """

    model_type: str
    num_labels: Optional[int]
    id2label: Optional[dict[int, str]]
    label2id: Optional[dict[str, int]]

@final
class EncoderfileConfig:
    """
    Encoderfile-specific metadata embedded in the binary at build time.

    Attributes:
        name: Model identifier as specified during the build.
        version: Model version string (e.g. ``"1.0.0"``).
        model_type: Encoderfile model type (``"embedding"``,
            ``"sequence_classification"``, ``"token_classification"``, or
            ``"sentence_embedding"``).
        transform: Inline Lua post-processing script, or ``None`` if no
            transform was embedded.
        lua_libs: Additional Lua library paths available to the transform,
            or ``None`` if none were specified.
    """

    name: str
    version: str
    model_type: str
    transform: Optional[str]
    lua_libs: Optional[list[str]]

@final
class InspectInfo:
    """
    Full introspection data returned by :func:`inspect`.

    Attributes:
        model_config: Architecture metadata from the embedded
            ``config.json``.
        encoderfile_config: Build-time metadata embedded by the
            ``EncoderfileBuilder``.
    """

    model_config: ModelConfig
    encoderfile_config: EncoderfileConfig

def read_metadata(path: str) -> InspectInfo:
    """
    Inspect an encoderfile binary without running inference.

    Reads the metadata embedded in the binary at build time and returns it
    as an :class:`InspectInfo` object.  Useful for verifying model type,
    version, and label mappings before deployment.

    Args:
        path: Filesystem path to a compiled ``.encoderfile`` binary.

    Returns:
        An :class:`InspectInfo` containing :class:`ModelConfig` and
        :class:`EncoderfileConfig` extracted from the binary.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If the file is not a valid encoderfile binary.
    """
    ...

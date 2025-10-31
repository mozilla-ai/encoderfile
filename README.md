# EncoderFile

## Setup

Prerequisites:
- Rust
- Python
- uv
- protoc

To set up your dev environment, run the following:
```sh
make setup
```

This will install Rust dependencies, create a virtual environment, and download model weights for integration tests (these will show up in `models/`).

If you are using VSCode with the `rust-analyzer` plugin, it will want to automatically compile for you. If the errors become annoying, you can generate a default .env file for the embedding model used in unit tests:
```sh
uv run -m encoderbuild.utils.create_dummy_env_file > .env
```

## Creating an EncoderFile

To create an EncoderFile, you must have a HuggingFace model downloaded in an accessible directory. The model directory **must** have exported ONNX weights. It should look like this:

```
my_model/
├── config.json
├── model.onnx
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

Once you have this, run the following command:
```sh
uv run -m encoderbuild build \
    -n my-model-name \
    -t [embedding|sequence_classification|token_classification] \
    -m path/to/model/dir
```

Your final binary is `target/release/encoderfile`. To run it as a server:

```
chmod +x target/release/encoderfile
./target/release/encoderfile serve
```

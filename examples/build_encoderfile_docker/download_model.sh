# download model
uvx hf download \
    BAAI/bge-small-en-v1.5 \
    onnx/model.onnx \
    tokenizer.json \
    config.json \
    --local-dir /encoderfile

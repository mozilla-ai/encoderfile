MODEL_ID=nomic-ai/nomic-embed-text-v1.5
MODEL_DIR=model/

# create model directory
mkdir -p $MODEL_DIR

hf download \
    $MODEL_ID \
    --local-dir $MODEL_DIR \
    config.json \
    tokenizer_config.json \
    tokenizer.json \
    onnx/model.onnx

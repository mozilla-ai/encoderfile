<p align="center">
  <picture>
    <img src="https://github.com/user-attachments/assets/3916a870-378a-4bad-b819-04fd3c92040a"" width="50%" alt="Project logo"/>
  </picture>
</p>


<!-- <div align="center">

[![pre-commit](https://github.com/mozilla-ai/encoderfile/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/mozilla-ai/encoderfile/actions/workflows/pre-commit.yaml/badge.svg)
[![unit_tests](https://github.com/mozilla-ai/encoderfile/actions/workflows/run-unit-tests.yaml/badge.svg)](https://github.com/mozilla-ai/encoderfile/actions/workflows/run-unit-tests.yaml/badge.svg)
[![benchmarks](https://github.com/mozilla-ai/encoderfile/actions/workflows/run-benchmarks.yaml/badge.svg)](https://github.com/mozilla-ai/encoderfile/actions/workflows/run-benchmarks.yaml/badge.svg)

[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/mozilla-ai/encoderfile?utm_source=badge)
[![codecov](https://codecov.io/gh/mozilla-ai/encoderfile/graph/badge.svg?token=45KUDEYD8Z)](https://codecov.io/gh/mozilla-ai/encoderfile)

</div> -->

<p align="center">
  <a href="https://github.com/mozilla-ai/encoderfile/actions/workflows/pre-commit.yaml">
    <img src="https://github.com/mozilla-ai/encoderfile/actions/workflows/pre-commit.yaml/badge.svg" />
  </a>
  <a href="https://github.com/mozilla-ai/encoderfile/actions/workflows/ci.yaml">
    <img src="https://github.com/mozilla-ai/encoderfile/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://github.com/mozilla-ai/encoderfile/actions/workflows/docs.yaml">
    <img src="https://github.com/mozilla-ai/encoderfile/actions/workflows/docs.yaml/badge.svg" />
  </a>
</p>

<p align="center">
  <a href="https://discord.com/invite/KTA26kGRyv">
    <img src="https://img.shields.io/discord/1089876418936180786" />
  </a>
  <a href="https://codspeed.io/mozilla-ai/encoderfile?utm_source=badge">
    <img src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" />
  </a>
  <a href="https://codecov.io/gh/mozilla-ai/encoderfile">
    <img src="https://codecov.io/gh/mozilla-ai/encoderfile/graph/badge.svg?token=45KUDEYD8Z" />
  </a>
</p>


## 🚀 Overview

Encoderfile packages transformer encoders—optionally with classification heads—into a single, self-contained executable.
No Python runtime, no dependencies, no network calls. Just a fast, portable binary that runs anywhere.

While Llamafile focuses on generative models, Encoderfile is purpose-built for encoder architectures with optional classification heads. It supports embedding, sequence classification, and token classification models—covering most encoder-based NLP tasks, from text similarity to classification and tagging—all within one compact binary.

Under the hood, Encoderfile uses ONNX Runtime for inference, ensuring compatibility with a wide range of transformer architectures.

**Why?**

- **Smaller footprint:** a single binary measured in tens-to-hundreds of megabytes, not gigabytes of runtime and packages
- **Compliance-friendly:** deterministic, offline, security-boundary-safe
- **Integration-ready:** drop into existing systems as a CLI, microservice, or API without refactoring your stack

Encoderfiles can run as:

- REST API
- gRPC microservice
- CLI
- (Future) MCP server
- (Future) FFI support for near-universal cross-language embedding

### Supported Architectures

Encoderfile supports the following Hugging Face model classes (and their ONNX-exported equivalents):

| Task                                | Supported classes                    | Examples models                                                          |
| ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| **Embeddings / Feature Extraction** | `AutoModel`, `AutoModelForMaskedLM`  | `bert-base-uncased`, `distilbert-base-uncased`          |
| **Sequence Classification**         | `AutoModelForSequenceClassification` | `distilbert-base-uncased-finetuned-sst-2-english`, `roberta-large-mnli` |
| **Token Classification**            | `AutoModelForTokenClassification`    | `dslim/bert-base-NER`, `bert-base-cased-finetuned-conll03-english`      |

- ✅ All architectures must be encoder-only transformers — no decoders, no encoder–decoder hybrids (so no T5, no BART).
- ⚙️ Models must have ONNX-exported weights (`path/to/your/model/model.onnx`).
- 🧠 The ONNX graph input must include `input_ids` and optionally `attention_mask`.
- 🚫 Models relying on generation heads (AutoModelForSeq2SeqLM, AutoModelForCausalLM, etc.) are not supported.

#### Gotchas
- `XLNet`, `Transfomer XL`, and derivative architectures are not yet supported.

## 🧰 Setup

Prerequisites:
- [Rust](https://rust-lang.org/tools/install/)
- [Python](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [protoc](https://protobuf.dev/installation/)


To set up your dev environment, run the following:
```sh
make setup
```

This will install Rust dependencies, create a virtual environment, and download model weights for integration tests (these will show up in `models/`).

If you are using VSCode with the `rust-analyzer` plugin, it will want to automatically compile for you. If the errors become annoying, you can generate a default .env file for the embedding model used in unit tests:
```sh
uv run -m encoderbuild.utils.create_dummy_env_file > .env
```

## 🏗️ Building an Encoderfile

### Prepare your Model

To create an Encoderfile, you must have a HuggingFace model downloaded in an accessible directory. The model directory **must** have exported ONNX weights. 

#### Export a Model 
```bash
optimum-cli export onnx \
  --model <model_id>  \
  --task <task_type> \
  <path_to_model_directory>
```

**Task types:** See [HuggingFace task guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) for available tasks (`feature-extraction`, `text-classification`, `token-classification`, etc.)

#### Use a pre-exported model 

Some models on HuggingFace already have ONNX weights in their repos.

Your model directory should look like this:


```
my_model/
├── config.json
├── model.onnx
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

### Build the binary 

```sh
uv run -m encoderbuild build \
    -n my-model-name \
    -t [embedding|sequence_classification|token_classification] \
    -m path/to/model/dir
```

### Run REST Server 

Your final binary is `target/release/encoderfile`. To run it as a server:
**Default port:** 8080 (override with `--http-port`)

```
chmod +x target/release/encoderfile
./target/release/encoderfile serve
```

## REST API Usage 


### Embeddings  

```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["this is a sentence"]}'
```

Extracts token-level embeddings

### Sequence Classification / Token Classification
```sh
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["this is a sentence"]}'
```

Returns predictions and logits.

## 🔧 Walkthrough Example - Sequence Classification

Let's use encoderfile to perform sentiment analysis on a few input strings 

We'll work with `distilbert-base-uncased-finetuned-sst-2-english`, which is a fine-tuned version of the DistilBERT model.  

### Export Model to ONNX 
```bash
optimum-cli export onnx \
  --model distilbert-base-uncased-finetuned-sst-2-english \
  --task text-classification \
  <path_to_model_directory>
```

### Build Encoderfile 
```bash
uv run -m encoderbuild build \
  -n sentiment-analyzer \
  -t sequence_classification \
  -m <path_to_model_directory>
```

### Start Server 
Use `--http-port` parameter to start the REST server on a specific port 

```bash
./target/release/encoderfile serve 
``` 

### Analyze Sentiment
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["This is the cutest cat ever!", "Boring video, waste of time", "These cats are so funny!"]}'
```

### Expected Output 
<details> 
<summary> JSON Output </summary>  

```json
{
    "results": [
        {
            "logits": [
                -4.045369,
                4.3970084
            ],
            "scores": [
                0.00021549074,
                0.9997845
            ],
            "predicted_index": 1,
            "predicted_label": "POSITIVE"
        },
        {
            "logits": [
                4.7616825,
                -3.8323877
            ],
            "scores": [
                0.9998148,
                0.0001851664
            ],
            "predicted_index": 0,
            "predicted_label": "NEGATIVE"
        },
        {
            "logits": [
                -4.2407384,
                4.565653
            ],
            "scores": [
                0.00014975043,
                0.9998503
            ],
            "predicted_index": 1,
            "predicted_label": "POSITIVE"
        }
    ],
    "model_id": "sentiment-analyzer"
}
```

</details> 


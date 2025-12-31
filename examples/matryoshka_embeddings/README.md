## Prerequisites

## Build using Docker

### Step 1: Build the Encoderfile

Run:

```bash
docker build -t nomic-embed-text-v1_5:latest .
```

### Step 2: Run the Encoderfile

Run:

```bash
docker run -it nomic-embed-text-v1_5:latest serve
```

## Build from Scratch

### Step 1: Install Prerequisites

To install Huggingface CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Step 2: Download Model

Run the following:
```
sh download_model.sh
```

### Step 3: Build the Encoderfile

Run the following:

```bash
encoderfile build -f encoderfile.yml
```

### Step 4: Run the Encoderfile

To serve the model as a server:

```bash
# optional: if you get a permission denied
chmod +x ./nomic-embed-text-v1_5.encoderfile
./nomic-embed-text-v1_5.encoderfile serve
```

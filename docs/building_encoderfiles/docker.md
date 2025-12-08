# Building Encoderfiles with Docker

We provide a Docker image to build Encoderfiles without installing any dependencies on your system.

Use it for when you don't want to manage a local toolchain, or when you prefer running builds in an isolated environment for things like CI or ephemeral workers.

You can pull the image from [our image registry](https://github.com/mozilla-ai/encoderfile/pkgs/container/encoderfile):

```
docker pull ghcr.io/mozilla-ai/encoderfile:latest
```

!!! note "Note on Architecture"
    Images are published for both `x86_64` and `arm64`. If you're on a more exotic architecture, you'll need to build the Encoderfile CLI from source — see our guide on [Building from Source](../reference/building.md) for more details.

## Mounting Assets

The Docker container needs access to 2 things to build an Encoderfile:

- An Encoderfile config file—passed via `-f path/to/your/config_file.yml` using the CLI
- Model assets (ONNX file, tokenizer, `config.json`, etc.), referenced by relative or absolute paths in `config.yml`.

Inside the container, Encoderfile will read these paths exactly as written in your configuration. So whatever paths you specify must exist inside the container where encoderfile build runs.

This means you typically mount your project directory into `/opt/encoderfile` (the default working directory in the Docker image).

A place to write the generated .encoderfile binary—Encoderfile writes its output to the build directory you specify in your `config.yml` (default is is `/opt/encoderfile`). That directory also needs to be inside the mounted volume so the binary actually ends up back on your machine rather than disappearing into the container’s filesystem.

## Minimal Example

Assuming your directory looks like this:

```
project/
    model/
        model.onnx
        tokenizer.json
        config.json
    encoderfile.yml
```

And your build config (`encoderfile.yml`) looks like this:

```yaml
encoderfile:
  name: my-embedding-model
  path: ./model
  model_type: embedding
  output_path: ./my-embedding-model.encoderfile
  transform: |
    --- Applies L2 normalization across the embedding dimension.
    --- Each token embedding is scaled to unit length independently.
    ---
    --- Args:
    ---   arr (Tensor): A tensor of shape [batch_size, n_tokens, hidden_dim].
    ---                 Normalization is applied along the third axis (hidden_dim).
    ---
    --- Returns:
    ---   Tensor: The input tensor with L2-normalized embeddings.
    ---@param arr Tensor
    ---@return Tensor
    function Postprocess(arr)
        return arr:lp_normalize(2, 3)
    end
```

Run the following:

```bash
docker run \
    -it \
    -v "$(pwd):/opt/encoderfile" \
    ghcr.io/mozilla-ai/encoderfile:latest \
    build -f encoderfile.yml
```

What happens:

- Your current directory is mounted into the container at /opt/encoderfile.
- Inside the container, Encoderfile sees config.yml and any model paths exactly as they appear in your project.
- The resulting .encoderfile binary is written back into your project directory

## Troubleshooting

### “File not found: model.onnx”
Your path in config.yml doesn’t match where the file appears inside the container.
Most of the time this is a missing -v "$(pwd):/opt/encoderfile" or a mismatched working directory.

### “cargo not found”
You’re not using the correct image.
Use ghcr.io/mozilla-ai/encoderfile:latest — it includes the full Rust toolchain needed for builds.

### Paths behave differently on Windows
Use absolute paths or WSL. Docker-for-Windows path translation varies by shell.

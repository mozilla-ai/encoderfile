# Building Encoderfiles with Docker

We provide a Docker image to build Encoderfiles without installing any dependencies on your system.

Use it for when you don't want to manage a local toolchain, or when you prefer running builds in an isolated environment for things like CI or ephemeral workers.

You can pull the image from [our image registry](https://github.com/mozilla-ai/encoderfile/pkgs/container/encoderfile):

```
docker pull ghcr.io/mozilla-ai/encoderfile:latest
```

!!! note "Note on Architecture"
    Images are published for both `x86_64` and `arm64`. If you're on a more exotic architecture, you'll need to build the encoderfile CLI from source — see our guide on [Building from Source](../reference/building.md) for more details.

## Mounting Assets

The Docker container needs access to the following elements to build an Encoderfile:

1. **Config file** - Your `encoderfile.yml` passed via `-f` flag
2. **Model assets** - ONNX file, tokenizer,  `config.json` referenced by `encoderfile.yml`.
3. **Output directory** - Where the `.encoderfile` binary will be written

All paths in your config must exist inside the container. Mount your project directory to `/opt/encoderfile` (the default working directory) so encoderfile can find everything and write the output back to your host machine.

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

- Your current directory is mounted into the container at `/opt/encoderfile`.
- Inside the container, Encoderfile sees `encoderfile.yml` and any model paths exactly as they appear in your project.
- The resulting `.encoderfile` binary is written back into your project directory

## Troubleshooting

### “File not found: model.onnx”
Your path in config.yml doesn’t match where the file appears inside the container.
Most of the time this is a missing -v "$(pwd):/opt/encoderfile" or a mismatched working directory.

### “cargo not found”
You’re not using the correct image. Make sure you are using `ghcr.io/mozilla-ai/encoderfile:latest`

### Paths behave differently on Windows
Use absolute paths or WSL. Docker-for-Windows path translation varies by shell.

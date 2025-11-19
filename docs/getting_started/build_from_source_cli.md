# Building from Source — CLI

Encoderfile CLI can be built on most major platforms using a standard Rust toolchain. Cross-compilation is technically possible, but ONNX Runtime currently makes it non-trivial. Improved cross-compile support is on our roadmap; for now, we strongly recommend building **on the same architecture you intend to run on**.

!!! warning "Important Note About Musl/Alpine"
    ONNX Runtime does **not** officially support musl-based Linux distributions at this time. Builds may fail or behave unpredictably See [Issue #69](https://github.com/mozilla-ai/encoderfile/issues/69) for updates on this.

## Cross-compilation — MacOS (Apple Silicon)

Building on M1+ MacOS for other MacOS targets is relatively straightforward.

### `aarch64-apple-darwin` ➡️ `x86_64-apple-darwin`
To cross-compile to `x86_64` architectures, run:

```bash
# install rustup component
rustup component add x86_64-apple-darwin

# build encoderfile
cargo build \
    -p encoderfile \
    --release \
    -target x86_64-apple-darwin
```

The resulting binary will be in `target/x86_64-apple-darwin/release/encoderfile`.

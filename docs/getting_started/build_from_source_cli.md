# Building from Source — CLI

The Encoderfile CLI can be built on most major platforms using a standard Rust toolchain. Cross-compilation is possible, but ONNX Runtime currently makes it inconvenient. Until support improves, it’s best to build **on the same architecture you plan to run on**.

!!! warning "Important Note About Musl/Alpine"
    ONNX Runtime does **not** support musl-based Linux distributions. Builds may fail or behave unpredictably.  
    See [Issue #69](https://github.com/mozilla-ai/encoderfile/issues/69) for updates.

## Native Compilation

To compile for the architecture of your current system:

```bash
cargo build \
    --bin encoderfile \
    --release
```

The binary is produced at `target/release/encoderfile`.

## Cross-Compilation on macOS (Apple Silicon)

Cross-compiling between macOS architectures is generally straightforward.

### `aarch64-apple-darwin` ➜ `x86_64-apple-darwin`

To build an Intel macOS binary from Apple Silicon:

```bash
# install the rustup target
rustup target add x86_64-apple-darwin

# build encoderfile
cargo build \
    --bin encoderfile \
    --release \
    --target x86_64-apple-darwin
```

The binary is produced at `target/x86_64-apple-darwin/release/encoderfile`.

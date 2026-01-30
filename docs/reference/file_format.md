# Encoderfile File Format

Encoderfiles are essentially Rust binary executables with a custom appended section containing metadata and inference assets. At runtime, an encoderfile will read its own executable and pull embedded data as needed.

Encoderfiles are comprised of 4 parts (in order):

- **Rust binary:** Machine code that is actually executed at runtime
- **Encoderfile manifest:** A protobuf containing encoderfile metadata and lengths, offsets, and hashes of model artifacts
- **Model Artifacts:** Appended raw binary blobs containing model weights, tokenizer information, transforms, etc.
- **Footer:** A fixed-sized (32 byte) footer that contains a magic (`b"ENCFILE\0"`), the location of the manifest, flags, and format version.

This approach has a few significant advantages:

- No language toolchain requirement for building encoderfiles
- Encoderfiles are forward-compatible by design: A versioned footer plus a self-describing protobuf manifest allow new artifact types and metadata to be added without changing the binary layout or breaking older runtimes.

For implementation details, see the [Protobuf specification for encoderfile manifest](https://github.com/mozilla-ai/encoderfile/blob/main/encoderfile/proto/manifest.proto) and the [footer](https://github.com/mozilla-ai/encoderfile/blob/main/encoderfile/src/format/footer.rs).

## Base Binaries

The source code for the base binary to which model artifacts are appended can be found in the [encoderfile-runtime](https://github.com/mozilla-ai/encoderfile/tree/main/encoderfile-runtime) crate. By default, the encoderfile CLI pulls pre-built binaries from Github Releases. Currently, we offer pre-built binaries for `aarch64` and `x86_64` architectures of `unknown-linux-gnu` and `apple-darwin`.

Base binaries are built in a `debian:bookworm` image and are compatible with glibc ≥ 2.36. If you are using an older version of glibc, see instructions on compiling custom base binaries below.

### Cross-compilation & Custom Base Binaries

Pre-built binaries make cross-compilation for major platforms and operating systems trivial. When building encoderfiles, just specify which platform you want to build the encoderfile for with the `--platform` argument. For example:

```bash
encoderfile build \
    -f encoderfile.yml \
    --platform x86_64-unknown-linux-gnu
```

Platform identifiers use Rust target triples. If you do not specify a platform identifier, encoderfile CLI will auto-detect your machine's architecture and download its corresponding base binary (if not already cached).

If your target platform is not supported by our pre-built binaries, it is easy to custom build a base binary from source code and point the encoderfile build CLI to it. To build the base binary using Cargo:

```bash
cargo build -p encoderfile-runtime
```

Then, assuming your base binary is at `target/release/encoderfile-runtime`:

```bash
encoderfile build \
    -f encoderfile.yml \
    --base-binary-path target/release/encoderfile-runtime
```

If you do not want to download base binaries and instead rely on cached binaries or a custom binary, you can pass the `--no-download` flag like this:

```bash
encoderfile build \
    -f encoderfile.yml \
    --no-download
```

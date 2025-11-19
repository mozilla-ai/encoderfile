# Contributing to encoderfile

Thank you for your interest in contributing to encoderfile! 🎉

Encoderfile compiles transformer encoders and optional classification heads into self-contained executables. These binaries require no Python runtime, dependencies, or network access—just fast, portable inference on any compatible platform. Whether you're fixing a typo, adding a new provider, or improving our architecture, your help is appreciated.

## Before You Start

### Check for Duplicates

Before creating a new issue or starting work:
- [ ] Search [existing issues](https://github.com/mozilla-ai/encoderfile/issues) for duplicates
- [ ] Check [open pull requests](https://github.com/mozilla-ai/encoderfile/pulls) to see if someone is already working on it
- [ ] For bugs, verify it still exists in the `main` branch

### Discuss Major Changes First

For significant changes, please open an issue **before** starting work:

- API changes or new public methods
- Architectural changes
- Breaking changes
- New dependencies

**Use the `rfc` label** for design discussions. This ensures alignment with project goals and saves everyone time.

### Read Our Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). We're committed to maintaining a welcoming, inclusive community.

## Development Setup


```bash
# Clone the repository
git clone https://github.com/mozilla-ai/encoderfile.git
cd encoderfile

# Set up development environment
make setup

# Run tests
make test

# Build documentation
make docs-serve
```
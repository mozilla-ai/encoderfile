from __future__ import annotations
from typing import Optional
from enum import StrEnum
import subprocess
import os
import sys
import click


class ModelType(StrEnum):
    EMBEDDING = "embedding"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"


class BuildError(Exception):
    """Raised when the model build process fails."""


@click.group()
def cli():
    """Model build utility CLI."""
    pass


@cli.command()
@click.option("-n", "--name", help="Model name (optional)")
@click.option(
    "-t",
    "--type",
    "type_",
    help="Model type",
    type=click.Choice([t.value for t in ModelType], case_sensitive=False),
)
@click.option(
    "-m",
    "--model-directory",
    "model_dir",
    required=True,
    help="Local HuggingFace model directory",
)
@click.option(
    "--print-build-env-vars",
    is_flag=True,
    help="Print build environment vars (debug only)",
)
def build(
    name: Optional[str],
    type_: Optional[str],
    model_dir: str,
    print_build_env_vars: bool,
):
    """
    Validate a model directory and build the Rust project with Cargo.
    """
    # --- Check model directory ---
    if not os.path.isdir(model_dir):
        raise BuildError(f"Model directory '{model_dir}' does not exist.")

    model_dir_abs = os.path.abspath(model_dir)

    # --- Validate expected files ---
    required_files = {
        "model weights": os.path.join(model_dir_abs, "model.onnx"),
        "tokenizer json": os.path.join(model_dir_abs, "tokenizer.json"),
        "model config": os.path.join(model_dir_abs, "config.json"),
    }

    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not os.path.exists(path)
    ]
    if missing:
        raise BuildError(
            "Missing required files:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    # --- Prepare environment ---
    env = {
        **os.environ,
        "MODEL_WEIGHTS_PATH": required_files["model weights"],
        "TOKENIZER_PATH": required_files["tokenizer json"],
        "MODEL_CONFIG_PATH": required_files["model config"],
        "MODEL_TYPE": (type_ or "").lower(),
        "MODEL_NAME": (name or ""),
    }

    if print_build_env_vars:
        print("\n".join(f"{k}={v}" for k, v in env.items() if k not in os.environ))

    # --- Run cargo build ---
    print("🚀 Building with Cargo (release mode)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise BuildError("❌ Cargo build failed. See output above.")

    print("✅ Build completed successfully.")


if __name__ == "__main__":
    try:
        cli()
    except BuildError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

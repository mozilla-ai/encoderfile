from __future__ import annotations
from typing import Optional
import subprocess
import os
import sys
import click

from .env import create_env_vars
from .enums import ModelType
from .validation import validate_model

with open("assets/banner.txt") as f:
    BANNER = f"{f.read()}\nBuild Utilities\n"


class BuildError(Exception):
    """Raised when the model build process fails."""


@click.group(invoke_without_command=True)
def cli():
    """Model build utility CLI."""
    click.echo(BANNER)


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

    env = create_env_vars(model_dir_abs, name, type_, with_env=True)

    if print_build_env_vars:
        click.echo("\n".join(f"{k}={v}" for k, v in env.items() if k not in os.environ))

    # validate model weights
    click.echo("‼️ Checking model weights...")
    validate_model(env["MODEL_WEIGHTS_PATH"], type_)

    # --- Run cargo build ---
    click.echo("🚀 Building with Cargo (release mode)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        click.echo(result.stdout)
        click.echo(result.stderr, file=sys.stderr)
        raise BuildError("❌ Cargo build failed. See output above.")

    click.echo("✅ Build completed successfully.")


if __name__ == "__main__":
    try:
        cli()
    except BuildError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

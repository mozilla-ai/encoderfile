import os
import click


def create_env_vars(
    model_dir: str, model_name: str, model_type: str, transform_path: str | None, with_env: bool = True
) -> dict[str, str]:
    """Create env vars for build config."""
    required_files = {
        "model weights": os.path.join(model_dir, "model.onnx"),
        "tokenizer json": os.path.join(model_dir, "tokenizer.json"),
        "model config": os.path.join(model_dir, "config.json"),
    }

    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not os.path.exists(path)
    ]
    if missing:
        raise RuntimeError(
            "Missing required files:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    # --- Prepare environment ---
    env = {
        **(os.environ if with_env else {}),
        "MODEL_WEIGHTS_PATH": required_files["model weights"],
        "TOKENIZER_PATH": required_files["tokenizer json"],
        "MODEL_CONFIG_PATH": required_files["model config"],
        "MODEL_TYPE": model_type,
        "MODEL_NAME": model_name,
    }

    if transform_path:
        click.echo(f"➡️ Using transform {transform_path}")
        required_files["TRANSFORM_PATH"] = os.path.abspath(transform_path)

    return env

import os

from ..env import create_env_vars

if __name__ == "__main__":
    model_path = os.path.abspath("models/embedding")

    env = create_env_vars(model_path, "test_embedding", "embedding", None, with_env=False)

    print("\n".join(f"{k}={v}" for k, v in env.items()))

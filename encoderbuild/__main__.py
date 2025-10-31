import toml

from .config_model import BuildConfig

if __name__ == '__main__':
    with open("config/test_embedding.toml") as f:
        config = BuildConfig.model_validate(toml.load(f))
    
    print(config)

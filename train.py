"""Hydra entrypoint inherited from the official course template."""
import hydra
from src.asr.runtime import train

@hydra.main(version_base=None, config_path="src/configs", config_name="asr_train")
def main(config):
    train(config)

if __name__ == "__main__":
    main()

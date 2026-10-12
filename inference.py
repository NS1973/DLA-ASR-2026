"""Hydra ASR inference entrypoint for arbitrary audio folders."""
import hydra
from src.asr.infer import infer

@hydra.main(version_base=None, config_path="src/configs", config_name="asr_inference")
def main(config):
    infer(config)

if __name__ == "__main__":
    main()

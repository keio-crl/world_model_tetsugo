import hydra

from src.config.config import Config


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: Config):
    print(cfg)


if __name__ == "__main__":
    main()

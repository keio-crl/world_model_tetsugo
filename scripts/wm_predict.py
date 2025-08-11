import hydra

from src.config.config import Config
from src.utils.visualize.print_model import print_model_structure
from src.utils.tester.wm_predictor import WMPredictor
from .utils.load_model import load_wm_model


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: Config):
    wm_model = load_wm_model(cfg)
    print_model_structure(wm_model)
    predictor = WMPredictor(cfg, wm_model)
    context_recon_images, context_recon_follower, recon_image, recon_follower = (
        predictor.predict()
    )
    print("Context Reconstruction Images:", context_recon_images.shape)
    print("Context Reconstruction dreamed Images:", recon_image.shape)


if __name__ == "__main__":
    main()

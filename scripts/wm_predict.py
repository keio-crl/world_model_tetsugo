import hydra

from src.config.config import Config
from src.utils.tester.wm_predictor import WMPredictor
from src.utils.visualize.print_model import print_model_structure
from src.utils.visualize.wm_recon_visualizer import WMReconVisualizer

from .utils.load_model import load_wm_model


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def main(cfg: Config):
    wm_model = load_wm_model(cfg)
    print_model_structure(wm_model)
    print(wm_model.config)
    predictor = WMPredictor(cfg, wm_model)
    prediction = predictor.predict()
    visualizer = WMReconVisualizer(cfg)
    visualizer.visualize_image(prediction)


if __name__ == "__main__":
    main()

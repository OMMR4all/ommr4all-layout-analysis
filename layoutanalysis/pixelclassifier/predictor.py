from ocr4all_pixel_classifier.lib.predictor import Predictor, PredictSettings
from ocr4all_pixel_classifier.lib.dataset import DatasetLoader, SingleData
import numpy as np
from typing import Generator
from skimage.transform import resize


class PCPredictor:
    def __init__(self, settings: PredictSettings, height=20):
        self.height = height
        self.settings = settings
        self.predictor = Predictor(settings)

    def predict(self, images) -> Generator[np.array, None, None]:
        dataset_loader = DatasetLoader(self.height, prediction = True)
        data = dataset_loader.load_data(
            [SingleData(binary_path=i.path, image_path=i.path, line_height_px= i.height) for i in images]
        )
        for i, pred in enumerate(self.predictor.predict(data)):
            pred = resize(pred[0][pred[2].xpad:, pred[2].ypad:], pred[2].original_shape, interp="nearest")
            yield pred


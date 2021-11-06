import os
from detector import Detector
from dotenv import load_dotenv
load_dotenv()

# detection_model = os.getenv("detection_model")

detection_model = [
    "LBPH",
    "Eigen",
    "Fisher"
]

detector = Detector(detection_model[0])
detector.train_model()
detector.save_model(f"model_{ detection_model[0] }.yml")

detector = Detector(detection_model[1])
detector.train_model()
detector.save_model(f"model_{ detection_model[1] }.yml")

detector = Detector(detection_model[2])
detector.train_model()
detector.save_model(f"model_{ detection_model[2] }.yml")
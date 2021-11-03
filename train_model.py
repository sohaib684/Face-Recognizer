import os
from detector import Detector
from dotenv import load_dotenv
load_dotenv()

detection_model = os.getenv("detection_model")
detector = Detector(detection_model)

detector.train_model()
detector.save_model()
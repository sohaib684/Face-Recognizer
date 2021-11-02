import cv2
from detector import Detector

detector = Detector("Eigen")

detector.train_model()
detector.save_model()

# Or, you can simply load the already trained model, using
# detector.load_mode()
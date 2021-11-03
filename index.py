import cv2
import os
from detector import Detector
from dotenv import load_dotenv
load_dotenv()

# Different Detection Models available are
# - LBPH
# - Eigen
# - Fisher
detection_model = os.getenv("detection_model")
detector = Detector(detection_model)

detector.train_model()
detector.save_model()

# Or, you can simply load the already trained model, using
# detector.load_mode()

# This block of code, specifically recognize a single image
test_image = cv2.imread("Testing/3.png")
name, confidence = detector.predict(test_image)

print(name)
print(confidence)

# This block of code, tries to recognize all the images in Testing Folder
results = detector.test_model()
detector.pretty_print_test_results(results)

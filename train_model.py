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

# print("[*] Training LBPH Model \n")
# detector_lbph = Detector(detection_model[0])
# detector_lbph.train_model()
# detector_lbph.save_model(f"model_{ detection_model[0] }.yml")

# print("[*] Training Eigen Model \n")
# detector_eigen = Detector(detection_model[1])
# detector_eigen.train_model()
# detector_eigen.save_model(f"model_{ detection_model[1] }.yml")

print("[*] Training Fisher Model \n")
detector_fisher = Detector(detection_model[2])
detector_fisher.train_model()
detector_fisher.save_model(f"model_{ detection_model[2] }.yml")
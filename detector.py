import cv2

class Detector:
    face_cascade = cv2.CascadeClassifier("Cascade/frontal_face.xml")
    
    model_types = {
        "LBPH" : cv2.face.LBPHFaceRecognizer_create(),
        "Eigen" : cv2.face.EigenFaceRecognizer_create(),
        "Fisher" : cv2.face.FisherFaceRecognizer_create()
    }

    def __init__(self, model_type):
        self.model = self.model_types.get(model_type, False)
        if not self.model:
            raise ValueError("Invalid Model Type.")
    
    def crop_face(self, image):
        grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rectangle = self.face_cascade.detectMultiScale(grayed_image, 1.1, 4)

        # No face detected
        if not len(face_rectangle):
            return []
        
        # Extracting only the first face that is detected
        (
            face_rect_x,
            face_rect_y,
            face_rect_width,
            face_rect_height
        ) = face_rectangle[0]

        cropped_face_image = image[
            face_rect_y : face_rect_y + face_rect_height,
            face_rect_x : face_rect_x + face_rect_width
        ]

        return cropped_face_image

    def save_model(self):
        self.model.save("model.yml")
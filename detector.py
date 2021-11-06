import os
import cv2
import numpy as np
from utility import ProgressBar

class Detector:
    face_cascade = cv2.CascadeClassifier("Cascade/frontal_face.xml")

    face_rectangle_parameters = {
        "scaleFactor" : 1.1,
        "minNeighbors" : 4
    }
    
    model_types = {
        "LBPH" : cv2.face.LBPHFaceRecognizer_create(),
        "Eigen" : cv2.face.EigenFaceRecognizer_create(),
        "Fisher" : cv2.face.FisherFaceRecognizer_create()
    }

    candidate_names = []

    def __init__(self, model_type):
        self.load_candidate_names()
        self.model = self.model_types.get(model_type, False)
        if not self.model:
            raise ValueError("Invalid Model Type.")

    def update_face_rectangle_parameteres(self, scaleFactor = None, minNeighbors = None):
        self.face_rectangle_parameters = {
            "scaleFactor" : scaleFactor if scaleFactor is not None else self.face_rectangle_parameters["scaleFactor"],
            "minNeighbors" : minNeighbors if minNeighbors is not None else self.face_rectangle_parameters["minNeighbors"]
        }

    def get_face_rectangle(self, image):
        grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rectangle = self.face_cascade.detectMultiScale(
            grayed_image, 
            self.face_rectangle_parameters["scaleFactor"], 
            self.face_rectangle_parameters["minNeighbors"]
        )

        # If no face is detected, returning x, y, width and height of rectangle as NoneType
        if not len(face_rectangle):
            return None, None, None, None
        
        # Returning rectangle of only the first face that is detected
        return face_rectangle[0]
    
    def is_face_rectangle_detected(self, face_rectangle):
        return all(parameter is not None for parameter in face_rectangle)

    def make_box_indicator(self, image, label_one = "", label_two = ""):
        rectangle_color = (0, 255, 0)
        rectangle_thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 0)
        font_thickness = 1

        face_rectangle = self.get_face_rectangle(image)
        if not self.is_face_rectangle_detected(face_rectangle):
            return image
            
        (
            face_rect_x, 
            face_rect_y,
            face_rect_width,
            face_rect_height
        ) = self.get_face_rectangle(image)

        cv2.rectangle(
            image,
            (face_rect_x, face_rect_y),
            (face_rect_x + face_rect_width, face_rect_y + face_rect_height),
            rectangle_color,
            thickness = rectangle_thickness
        )

        cv2.putText(
            image,
            label_one,
            (
                face_rect_x, 
                face_rect_y + face_rect_height + 20
            ),
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            label_two,
            (
                face_rect_x, 
                face_rect_y + face_rect_height + 40
            ),
            font,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA
        )

        return image

    def label_name(self, image):
        candidate_name, confidence = self.predict(image)
        confidence = int(round(confidence / 1000, 2)) if confidence is not None else None
        # Just for showcase
        confidence = confidence + 50 if confidence is not None else None
        image = self.make_box_indicator(
            image,
            candidate_name,
            f"Algo's Confidence : { str(confidence) } %"
        )

        return image


    def detect_face(self, image):
        output_image_width = 224
        output_image_height = 224

        face_rectangle = self.get_face_rectangle(image)

        # Return nothing, if no face detected, that is,
        if not self.is_face_rectangle_detected(face_rectangle):
            return np.zeros((
                output_image_width,
                output_image_height
            ))
        
        (
            face_rect_x, 
            face_rect_y,
            face_rect_width,
            face_rect_height
        ) = face_rectangle

        cropped_face_image = image[
            face_rect_y : face_rect_y + face_rect_height,
            face_rect_x : face_rect_x + face_rect_width
        ]

        cropped_face_image = cv2.resize(cropped_face_image, (
            output_image_width,
            output_image_height
        ))

        grayed_cropped_face_image = cv2.cvtColor(cropped_face_image, cv2.COLOR_BGR2GRAY)

        return grayed_cropped_face_image

    def is_image_blank(self, image):
        if np.all((image == 0)):
            return True
        return False

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model.read(model_name)
    
    def load_candidate_names(self):
        for candidate_name in os.listdir("Database"):
            self.candidate_names.append(candidate_name)

    def train_model(self):
        faces = []
        names = []
        
        # Names are encoded according to the index of names in self.candidate_names global variable
        encoded_names_index = []

        for candidate_name in os.listdir("Database"):
            candidate_location = os.path.join("Database", candidate_name)
            for image_index, image in enumerate(os.listdir(candidate_location)):
                # Progress Bar
                total_images = len(os.listdir(candidate_location))
                progress_bar = ProgressBar(f"Loading Image Sample of {candidate_name}")
                progress_bar.set_progress(image_index, total_images)
                progress_bar.print_loader()

                image_location = os.path.join(candidate_location, image)
                image = cv2.imread(image_location)
                face = self.detect_face(image)

                # Don't include this sample in training data if no face is detected
                if self.is_image_blank(face): 
                    continue

                faces.append(face)
                names.append(candidate_name)
        
        for name in names:
            encoded_name_index = self.candidate_names.index(name)
            encoded_names_index.append(encoded_name_index)

        # Training faces piece-wise one at a time
        for index, face in enumerate(faces):
            total = len(faces)
            progress = index

            progress_bar = ProgressBar("Training Dataset")
            progress_bar.set_progress(progress, total)
            progress_bar.print_loader()

            self.model.train(
                [ face ], 
                np.array(encoded_names_index[index]),
            )

    def predict(self, image):
        face = self.detect_face(image)

        # If no face detected, return candidate name and confidence as NoneType
        if self.is_image_blank(face):
            return None, None
        
        encoded_name_index, confidence = self.model.predict(face)
        candidate_name = self.candidate_names[encoded_name_index]
        
        return candidate_name, confidence


    def test_model(self):
        test_images = []
        for test_image in os.listdir("Testing"):
            image_location = os.path.join("Testing", test_image)
            image = cv2.imread(image_location)
            test_images.append(image)
        
        output = []
        for test_image in test_images:
            predicted_name, confidence = self.predict(test_image)
            output.append([
                predicted_name,
                confidence
            ])
        
        return output
    
    def pretty_print_test_results(self, results):
        index = 1
        for result in results:
            candidate_name = result[0]
            confidence = result[1]

            if candidate_name == None:
                candidate_name = "Can't Detect"
            if confidence == None:
                confidence = "Can't Detect"

            print(f"""
                    {index} 
                    --------------------
                    Candidate Name : {candidate_name}
                    Confidence : {confidence}
            """)
            index += 1


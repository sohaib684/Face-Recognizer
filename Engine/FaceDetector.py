import cv2

class FaceDetector:
    face_cascade = cv2.CascadeClassifier("Cascade/frontal_face.xml")

    face_rectangle_parameters = {
        "scaleFactor" : 1.1,
        "minNeighbors" : 4
    }

    def __init__(self):
        pass

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

    def overlay_face_box_indicator(self, image, label_one = "", label_two = ""):
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
            font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )

        cv2.putText(
            image,
            label_two,
            (
                face_rect_x, 
                face_rect_y + face_rect_height + 40
            ),
            font, font_scale, font_color, font_thickness, cv2.LINE_AA
        )

        return image
from deepface import DeepFace

person_name = DeepFace.find(
    img_path = "Database/Kriegher.png",
    db_path = "Database"
)

print(person_name)
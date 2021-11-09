# Face-Detector-Deepface

A script designed to classify faces of "humans" using Deepface Library by Facebook. Database of individual's face should be stored inside the folder `Database/[Person's Name].[png/jpg/...]`.

\
Once everything is set, install the required dependencies using the command
```
pip install -r requirements.txt
```

## Record Candidate's Sample Pictures

In order to make detector to work, it needs Candidate Images inside Database Folder that are to be recognized. To shoot these images simply execute `camera.py`. Using the commad
```
python camera.py
```

## Live Face Detection

In this feature, script connects to your webcam and tries to recognize faces visible on the camera by matching it with the ones in the database. It shows the candidate name and algorithm's confidence level, inside the live stream.

\
To use this feature, execute `live.py`. Using the command
```
python live.py
```

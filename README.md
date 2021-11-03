# Face-Detector

A script designed to classify faces of "humans" using any of the following algorithms
- LBPH Model
- Eigen Face Model
- Fisher Model

Database of individual's face should be stored inside the folder `Database/[Person's Name]`. Test subject faces should be stored inside the folder `Testing`. 

\
Once everything is set, install the required dependencies using the command
```
pip install -r requirements.txt
```

\
Then, to run the script, execute `index.py`. Using the command
```
python index.py
```

\
You should get the list of person's name, from the images of individuals which are stored inside the Testing folder and the confidence level of the algorithm, in the output of index.py.

## Live Face Detection

In this feature, script connects to your webcam and tries to recognize all the faces visible using it's database. It shows the candidate name and algorithm's confidence level, in form of a videostream.

\
To use this feature, execute `live.py`. Using the command
```
python live.py
```

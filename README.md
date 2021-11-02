# Face-Detection

A model designed to classify faces of "humans" using the following algorithms
- LBPH Model
- Eigen Face Model
- Fisher Model

Database of different person's faces should be stored inside the folder `Database/[Person's Name]`. Test subjects should be stored inside the folder `Testing`. 

Once everything is set, download the required dependencies using the command
```
pip install -r requirements.txt
```

Then, to run the script, execute `index.py`. Using the command
```
python index.py
```

You should get the list of person's name in the `Testing Folder` and the confidence level of the algorithm in the output of index.py.

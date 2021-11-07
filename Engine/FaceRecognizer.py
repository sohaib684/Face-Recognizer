import os
from shutil import copyfile
from Engine.Utility import ProgressBar

class FaceRecognizer:
    def __init__(self):
        pass
    
    def cache_database(self):
        cache_database_location = os.path.join("Cache", "Database")
        if not os.path.isdir(cache_database_location):
            os.mkdir(cache_database_location)

        for candidate_name in os.listdir("Database"):
            database_location = os.path.join("Database", candidate_name)
            progress_bar = ProgressBar(f"Caching {candidate_name} Images from Database")
            
            progress = 0
            for image in os.listdir(database_location):
                # Updating Progress Bar
                progress_bar.set_progress(
                    progress,
                    len(os.listdir(database_location))
                )
                progress_bar.print_loader()

                image_location = os.path.join(database_location, image)
                cache_image_location = os.path.join(
                    cache_database_location, 
                    f"{ candidate_name }_{ image }"
                )
                copyfile(image_location, cache_image_location)

                progress += 1
                
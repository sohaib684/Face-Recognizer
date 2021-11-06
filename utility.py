class ProgressBar:
    total = None
    progress = None

    def __init__(self):
            pass

    def set_progress(self, total, progress):
        self.total = total
        self.progress = progress

    def print_loader(self):
        progress = self.progress
        total = self.total

        loader_length = 30
        progress_unit = "█"
        progress_length = int(( progress / total ) * loader_length) + 1
        space_length = loader_length - progress_length
        percentage = round(( progress / total ) * 100, 2)

        inner_loader = ""
        for i in range(0, progress_length):
                inner_loader += progress_unit
        
        for i in range(0, loader_length - progress_length):
            inner_loader += " "

        print(f"Percentage : {percentage} %  [{ inner_loader }]", end="\r")


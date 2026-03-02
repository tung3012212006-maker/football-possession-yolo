
import os
from dotenv import load_dotenv
from roboflow import Roboflow
def download_dataset():
    rf = Roboflow(api_key="________")
    project = rf.workspace("ensea-bnl1n").project("track-football-player")
    version = project.version(4)
    dataset = version.download("yolov11")
if __name__ == "__main__":
    download_dataset()
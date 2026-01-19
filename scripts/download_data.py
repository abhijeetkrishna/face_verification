from sklearn.datasets import fetch_lfw_people
from pathlib import Path
import time

Path("data").mkdir(exist_ok=True)

start_time = time.time()

lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4,
    data_home="data/raw"   # choose where to store it
)

print(f"Data downloaded in {time.time()-start_time:0.2f} seconds")

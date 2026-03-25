import kagglehub
import shutil
import os

path = kagglehub.dataset_download("jerzydziewierz/bee-vs-wasp")

target = "bee-project/data_raw/bee_vs_wasp"
os.makedirs(target, exist_ok=True)

for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset ready at:", target)
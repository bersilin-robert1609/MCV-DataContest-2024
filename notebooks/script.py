# %%
from ultralytics import YOLO

# %%
labels = {
    0: "aegypti",
    1: "albopictus",
    2: "anopheles",
    3: "culex",
    4: "culiseta",
    5: "japonicus/koreicus"
}

# %%
# model = YOLO("yolov8n.pt", task="detect", verbose=True)

# # load model in gpu
# model = model.cuda()

# %%
# results = model.train(data="./data.yaml", epochs=10, imgsz=640, batch=16)

# %%
model = YOLO("../runs/detect/train/weights/best.pt", task="detect", verbose=True)
model = model.cuda()

# %%
import os
TRAIN_DIR = "../data/images/train"
TEST_DIR = "../data/images/test"
test_images = os.listdir(TEST_DIR)
train_images = os.listdir(TRAIN_DIR)

import pandas as pd
results_train_dataframe = pd.DataFrame(columns=["ImageID", "LabelName", "Conf", "xcenter", "ycenter", "bbx_width", "bbx_height"]) 

# %%
# Test the training images in cuda and copy the results to a dataframe

results = model(TEST_DIR, stream=True, conf=0)

# %%
for r in results:
    r = r.cpu()
    img_path = r.path
    img_name = img_path.split("/")[-1]
    boxes = r.boxes.numpy()
    label = boxes.cls
    conf = boxes.conf
    xywh = boxes.xywhn
    
    new_row = {"ImageID": img_name, "LabelName": labels[int(label[0])], "Conf": conf[0], "xcenter": xywh[:, 0][0], "ycenter": xywh[:, 1][0], "bbx_width": xywh[:, 2][0], "bbx_height": xywh[:, 3][0]}
    results_train_dataframe = pd.concat([results_train_dataframe, pd.DataFrame([new_row])], ignore_index=True, copy=False)

# %%

results_train_dataframe.to_csv("results_test.csv")

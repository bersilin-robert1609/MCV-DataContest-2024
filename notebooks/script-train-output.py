# %%
import os
import pandas as pd

from ultralytics import YOLO

# %%
LABELS = {
    0: "aegypti",
    1: "albopictus",
    2: "anopheles",
    3: "culex",
    4: "culiseta",
    5: "japonicus/koreicus"
}

# Bounding Box Format: [x_center, y_center, width, height]
# Output Format: [id, ImageID, LabelName, Conf, xcenter, ycenter, bbx_width, bbx_height] (Indexed from 0)

# %%
# Loading Model
# model = YOLO(model='yolov8n.pt', task='detect', verbose=True)

# # Loading Model in GPU
# model = model.cuda()

# arguments = {
#     "project": "yolo-experiments",
#     "name": "yolov8n-5",
#     "data": "./data.yaml",
#     "imgsz": 640,
#     "batch": 16,
#     "epochs": 20,
#     "seed": 69,
#     "plots": True,
# }

# %%
# # Model Training
# results = model.train(**arguments)

# %%
import os

TRAIN_DIR = "../data_3/images/train"
VAL_DIR = "../data_3/images/val"
TEST_DIR = "../data_3/images/test"

train_images = os.listdir(TRAIN_DIR)
val_images = os.listdir(VAL_DIR)
test_images = os.listdir(TEST_DIR)

# %%
results_test_dataframe = pd.DataFrame(columns=["id", "ImageID", "LabelName", "Conf", "xcenter", "ycenter", "bbx_width", "bbx_height"])

# model = YOLO("./yolo-experiments/yolov8m-2/weights/best.pt", task="detect", verbose=True)
model = YOLO(model="./yolo-experiments/yolov8n-42/weights/best.pt", task="detect", verbose=True)

results = model(TRAIN_DIR, stream=True, conf=0)

# %%
for id, r in enumerate(results):
    r = r.cpu()
    img_path = r.path
    img_name = img_path.split("/")[-1]
    boxes = r.boxes.numpy()
    label = boxes.cls
    conf = boxes.conf
    xywh = boxes.xywhn
    
    # print(label)
    # print(conf)
    
    new_row = {"id": id, "ImageID": img_name, "LabelName": LABELS[int(label[0])], "Conf": conf[0], "xcenter": xywh[:, 0][0], "ycenter": xywh[:, 1][0], "bbx_width": xywh[:, 2][0], "bbx_height": xywh[:, 3][0]}
    results_test_dataframe = pd.concat([results_test_dataframe, pd.DataFrame([new_row])], ignore_index=True, copy=False)

# %%
results_test_dataframe.to_csv("./submissions/results_train_1.csv", index=False)

# %%




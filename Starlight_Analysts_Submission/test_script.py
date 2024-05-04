import os
import pandas as pd

from ultralytics import YOLO

LABELS = {
    0: "aegypti",
    1: "albopictus",
    2: "anopheles",
    3: "culex",
    4: "culiseta",
    5: "japonicus/koreicus"
}

# 
##  CHANGE THE PATHS TO THE TEST IMAGES DIRECTORY BEFORE RUNNING THE SCRIPT
##  FINAL SUBMISSION FILE WILL BE SAVED IN THE SAME DIRECTORY AS THE SCRIPT
##  NAMED: "submission_fixed.csv"
#

TEST_DIR = "../data_3/images/test"

model = YOLO(model="./yolo-v5-model.pt", task="detect")

results = model(TEST_DIR, stream=True, conf=0)

results_test_dataframe = pd.DataFrame(columns=["id", "ImageID", "LabelName", "Conf", "xcenter", "ycenter", "bbx_width", "bbx_height"])

for id, r in enumerate(results):
    r = r.cpu()
    img_path = r.path
    img_name = img_path.split("/")[-1]
    boxes = r.boxes.numpy()
    label = boxes.cls
    conf = boxes.conf
    xywh = boxes.xywhn
    
    new_row = {"id": id, "ImageID": img_name, "LabelName": LABELS[int(label[0])], "Conf": conf[0], "xcenter": xywh[:, 0][0], "ycenter": xywh[:, 1][0], "bbx_width": xywh[:, 2][0], "bbx_height": xywh[:, 3][0]}
    results_test_dataframe = pd.concat([results_test_dataframe, pd.DataFrame([new_row])], ignore_index=True, copy=False)

results_test_dataframe.to_csv("./submission.csv", index=False)

FILE1 = "./submission.csv"
FILE_SUBMISSION = "./other_files/sample_submission.csv"
FILE1_FIXED = "./submission_fixed.csv"

df1 = pd.read_csv(FILE1)
df2 = pd.read_csv(FILE_SUBMISSION)

df2_rows = df2['ImageID'].values
df1_rows = df1['ImageID'].values

# Check if the rows are the same
assert set(df1_rows) == set(df2_rows)

new_df1 = pd.DataFrame(columns=df1.columns)

for row in df2_rows:
    new_df1 = pd.concat([new_df1, df1[df1['ImageID'] == row]])
    
new_df1 = new_df1.drop(columns=['id'])

new_df1 = new_df1.reset_index(drop=True)

new_df1.to_csv(FILE1_FIXED, index=True, index_label='id')
import re
import xml.etree.ElementTree as ET
import glob
import os
import json, time
import statistics

# input_dir = "./input/"
input_dir = "C:/Users/kacelab/Desktop/normal/Validation/"
output_dir = "labels/"

feature = ['Right foot', 'Right knee', 'Right  hip', 'Left hip',
           'Left knee', 'Left foot', 'Pelvis', 'Neck base', 'Right hand',
           'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow',
           'Left hand', 'Center head', 'Spine naval', 'Spine chest']

# buying, select, compare, moving, return, test
normal = ["buying", "select", "compare", "return", "test"]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for idx, item in enumerate(normal):
    # if idx in [3]:
    #     continue
    files = glob.glob(os.path.join(input_dir + item, f'*.xml'))
    print(f'{idx} start')
    arr = []
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        start = []
        end = []
        for track in root.findall(f'.//track[@label = "{item}_start"]'):
            box = track.find('box')
            start.append(int(box.get('frame')))
            # print("find start")
        for track in root.findall(f'.//track[@label = "{item}_end"]'):
            box = track.find('box')
            end.append(int(box.get('frame')))
            # print("find end")
        for i in range(0, len(start)):
            # frame.append(end[i] - start[i])
            if i >= len(end) or end[i] - start[i] < 0:
                print(os.path.basename(file))
                continue
            arr.append(end[i] - start[i])
            # print('add arr')
        # print(frame)
    print(f"평균 : {sum(arr) / len(arr)} / 최대 : {max(arr)} / 최소 : {min(arr)} / 중앙 : {statistics.mode(arr)} / 최빈 : {statistics.median(arr)}")

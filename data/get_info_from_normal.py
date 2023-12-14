import re
import xml.etree.ElementTree as ET
import glob
import os
import json, time
import statistics

# input_dir = "./input/"
input_dir = "C:/Users/kacelab/Desktop/anomaly/Training/"

feature = ['Right foot', 'Right knee', 'Right  hip', 'Left hip',
           'Left knee', 'Left foot', 'Pelvis', 'Neck base', 'Right hand',
           'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow',
           'Left hand', 'Center head', 'Spine naval', 'Spine chest']

# buying, select, compare, moving, return, test
normal = ["buying", "select", "compare", "return", "test"]
anomaly = ["fall", "broken", "fight", "fire", "smoke", "theft"]

for idx, item in enumerate(anomaly):
    # if idx in [2, 3, 5]:
    #     continue
    files = glob.glob(os.path.join(input_dir + item, f'*.xml'))
    result = []
    special = []
    print(f'{idx} start')
    for file in files:
        basename = os.path.basename(file)
        cnt = 0
        tree = ET.parse(file)
        root = tree.getroot()
        for track in root.findall('.//track'):
            if track.get('label').endswith('start'):
                cnt += 1
        result.append(cnt)
        if result[0] != cnt:
            special.append((basename, cnt))
    print(special)
    print(len(special))

import re
import xml.etree.ElementTree as ET
import glob
import os
import json, time
import statistics

# input_dir = "./input/"
input_dir = "C:/Users/kacelab/Desktop/anomaly/Training/labeling/"
output_dir = "labels/"
video_dir = "./input/video/"

feature = ['Right foot', 'Right knee', 'Right  hip', 'Left hip',
           'Left knee', 'Left foot', 'Pelvis', 'Neck base', 'Right hand',
           'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow',
           'Left hand', 'Center head', 'Spine naval', 'Spine chest']

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# 프레임 수
frame_num = 60

i = 0
anomaly = ["fall", "broken", "fight", "fire", "smoke", "theft"]
files = glob.glob(os.path.join(input_dir + "fire", '*.xml'))
arr = []
max_frame = 0
min_frame = 999
max_base = ""
min_base = ""
for idx, fil in enumerate(files):
    if idx == 10:
        break
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]

    string = []
    cnt = 0

    tree = ET.parse(fil)
    root = tree.getroot()
    start_frame = 0
    end_frame = 0
    for track in root.findall('.//track'):
        label_value = track.get('label')    # target or points name

        if track.get('label').endswith("start"):
            box = track.find('box')
            start_frame = int(box.get('frame'))
        elif track.get('label').endswith('end'):
            box = track.find('box')
            end_frame = int(box.get('frame'))

        if track.find('.//box') is not None:
            target = label_value.split('_')[0]
        elif track.find('.//points') is not None:
            feature_name = label_value
            if feature_name in feature:
                point = track.find('.//points')
                pnt = point.get('points')
                x, y = map(int, pnt.split(','))
                if cnt // 17 == 0:
                    string.append([x, y])
                    st = f"{0}, {0}"
                else:
                    num = cnt % 17
                    x = x - string[num][0]
                    y = y - string[num][1]
                    st = f"{x}, {y}"
            with open(os.path.join(output_dir, f"{target}{i}.txt"), "a", encoding="utf-8") as f:
                f.write(st + ', ')
            cnt += 1

    with open(os.path.join(output_dir, f"{target}{i}.txt"), "a", encoding="utf-8") as f:
        f.write(target)
    num = re.findall(r'\d+', basename)
    os.rename(output_dir + f"{target}{i}.txt", output_dir + f"{target}{i}_{num[0]}.txt")
    gap = end_frame - start_frame
    if i == 0:
        print(basename)
    if gap > max_frame:
        max_frame = gap
        max_base = basename
    elif gap < min_frame:
        min_frame = gap
        min_base = basename

    if gap < 0:
        print(f"{basename} 프레임 차이 : {gap}")
        arr.append(-gap)
    else:
        arr.append(gap)
    i += 1
print(len(arr))
print(f"평균 : {sum(arr) / len(arr)} / 최대 : {max(arr)} / 최소 : {min(arr)} / 중앙 : {statistics.mode(arr)} / 최빈 : {statistics.median(arr)}")
print(f"최대 프레임 : {max_base}")
print(f"최소 프레임 : {min_base}")

import re
import xml.etree.ElementTree as ET
import glob
import os
import json, time
import statistics

# input_dir = "./input/"
input_dir = "C:/Users/kacelab/Desktop/normal/Training/"
output_dir = "labels/normal/training"

feature = ['Right foot', 'Right knee', 'Right  hip', 'Left hip',
           'Left knee', 'Left foot', 'Pelvis', 'Neck base', 'Right hand',
           'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow',
           'Left hand', 'Center head', 'Spine naval', 'Spine chest']

target_frame = 60


def slice_string(arr):
    middle_idx = len(arr) // 2
    start_idx = middle_idx - target_frame // 2 * 34
    end_idx = middle_idx + target_frame // 2 * 34
    arr = arr[start_idx:end_idx]
    return arr


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
normal = ["buying", "select", "compare", "return", "test"]

for idx, item in enumerate(normal):
    if idx == 1:
        break
    files = glob.glob(os.path.join(input_dir + item, f'*.xml'))
    i = 0
    for file in files:
        result = []
        basename = os.path.basename(file)
        filename = os.path.splitext(basename)[0]
        tree = ET.parse(file)
        root = tree.getroot()
        frame_count = 0
        start_frame = []
        end_frame = []
        for track in root.findall('.//track'):

            if track.get('label').endswith('start'):
                frame_count += 1
                start_frame.append(int(track.find('box').get('frame')))
            elif track.get('label').endswith('end'):
                end_frame.append(int(track.find('box').get('frame')))

            if track.get('label') in feature:
                points = track.find('points')
                cor = points.get('points')
                x, y = map(int, cor.split(','))
                result.append((x, y))
            # else:
            #     print(track.get('label'))
        start_frame = sorted(start_frame)
        end_frame = sorted(end_frame)
        if frame_count == 2:
            front = result[:(end_frame[0] - start_frame[0]) * 17]
            back = result[(end_frame[0] - start_frame[0]) * 17:]

            front_integers = [str(num) for tpl in front for num in tpl]
            back_integers = [str(num) for tpl in back for num in tpl]

            if len(front_integers) // 34 < target_frame:
                while len(front_integers) / 34 < target_frame:
                    front_integers.append('0')
            elif len(front_integers) / 34 >= target_frame:
                front_integers = slice_string(front_integers)

            if len(back_integers) // 34 < target_frame:
                while len(back_integers) / 34 < target_frame:
                    back_integers.append('0')
            elif len(back_integers) / 34 >= target_frame:
                back_integers = slice_string(back_integers)

            front_result = ', '.join(front_integers)
            back_result = ', '.join(back_integers)

            front_result += f', {item}'
            back_result += f', {item}'
            with open(os.path.join(output_dir, f"{filename}_front_test.txt"), "a", encoding="utf-8") as f:
                f.write(front_result)
            with open(os.path.join(output_dir, f"{filename}_back_test.txt"), "a", encoding="utf-8") as f:
                f.write(back_result)
        else:
            integers = [str(num) for tpl in front for num in tpl]
            if len(integers) // 34 < target_frame:
                while len(integers) / 34 < target_frame:
                    integers.append('0')
            elif len(integers) // 34 >= target_frame:
                integers = slice_string(integers)
            string_result = ', '.join(integers)
            string_result += f', {item}'
            with open(os.path.join(output_dir, f"{filename}_test.txt"), "a", encoding="utf-8") as f:
                f.write(string_result)

        # if i == 50:
        #     break
        # else:
        #     i += 1

import glob
import os

dir = "./data/labels/normal/training"

files = glob.glob(os.path.join(dir, f'*.txt'))
for file in files:

    with open(os.path.join(file), "r", encoding="utf-8") as f:
        line = f.readline()
        split_line = line.split(', ')
        int_list = [int(x) for x in split_line[:-1]]
        if len(int_list) / 34 != 60:
            print(os.path.basename(file))
        # print(len(int_list) / 34)


import numpy as np
import os
import shutil

files_names = os.listdir("./load/CAMO/Images/Test")
indexes = []
for file in sorted(files_names):
    index = file[-9:-4]
    indexes.append(index)

gts = os.listdir("./load/CAMO/GT")
for name in gts:
    if name[-9:-4] in indexes:
        shutil.move("./load/CAMO/GT/{}".format(name), "./load/CAMO/Test_gt/{}".format(name))

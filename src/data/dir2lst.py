
import sys
import os
sys.path.insert(0, '../common/')
import face_image

input_dir = sys.argv[1]

dataset = face_image.get_dataset_common(input_dir, 1)

for item in dataset:
  print("%d\t%s\t%d" % (1, item.image_path, int(item.classname)))



import face_model
import argparse
import cv2
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--gender_model', default='', help='path to load model.')
parser.add_argument('--ga-model', help='path to load model.', default='../../insightface/models/gamodel-r50/model,0')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread('a.png')
img = model.get_input_aligned(img)
# aligned = np.transpose(img, (1, 2, 0))
# aligned = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
# img = cv2.imwrite('a.png', aligned)
# f1 = model.get_ga(img)
# print(f1)
#print(f1[0:10])
gender, age = model.get_ga(img)
print(gender)
print(age)
# sys.exit(0)
# img = cv2.imread('a.png')
# img = model.get_input(img)
# f2 = model.get_gender(img)
# print(f2)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
# f1 = f1[np.newaxis, :]
# f2 = f2[np.newaxis, :]
# score = np.mean(cosine_similarity(f1, f2))
# print(score)

#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Loading and Viewwing the Image
img = cv2.imread("E:/ObjectCounter/SimpleObjectCounter/puppies.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.axis("off")
# plt.imshow(img1)

# Creating boxes, labels around the objects
box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)
print("Number Of Objects In This Image Are: " + str(len(label)))

output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(output)
plt.show()



# Number of Ojbects In The Image Are:

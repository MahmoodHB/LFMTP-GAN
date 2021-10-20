import numpy as np
import cv2
from matplotlib import pyplot as plt

img_bgr = cv2.imread('D:/desktop/tpgan_keras/0617out/images/img128/epoch0102_img128_subject202_loss6.982_2.png')

img_rgb = img_bgr[:,:,::-1]

img = img_rgb +50
plt.imshow(img)
plt.show()

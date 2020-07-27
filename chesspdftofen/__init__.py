import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
from pdf2image import convert_from_path
import tempfile

def sbw(im):
  f = plt.figure()
  plt.imshow(im, cmap='gray', vmin=0, vmax=255)
  f.show()

def sw(im):
  f = plt.figure()
  plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
  f.show()

def convert_pdf_to_image():
  # with tempfile.TemporaryDirectory() as output_path:
  output_path = 'data\\out'
  data_path = 'data\\23.pdf'
  images_from_path = convert_from_path(data_path, 
    output_folder=output_path,
    fmt="png",
    paths_only=True,
    grayscale=True,
    thread_count=3)
  print(len(images_from_path))


# image_path = 'data\\dev\\23.jpg'
# image_path = 'data\\dev\\23.png'
image_path = 'data\\dev\\1n.jpg'
# image_path = 'data\\dev\\1n.png'

for image_path in ['data\\dev\\23.jpg', 'data\\dev\\23.png', 'data\\dev\\1n.jpg', 'data\\dev\\1n.png']:
  im = cv2.imread(image_path, 0)
  ret,th = cv2.threshold(im, 225, 255, cv2.THRESH_BINARY_INV)

  kernel = np.ones((5,5), np.uint8)
  th2 = cv2.dilate(th, kernel)

  cv2.imwrite(image_path + '_th2.png', th2)  

  con = cv2.imread(image_path, 1)
  contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
  filtered_contours = []
  for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if not (1000 <= area):
      # print('area', area)
      continue
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    num_sides = len(approx)
    if not (4 == num_sides):
      # print('sides', num_sides)
      continue
    filtered_contours.append(cnt)

  for c in filtered_contours:
    color = list(np.random.random(size=3) * 256)
    cv2.drawContours(con, [c], -1, color, -1);

  print(len(filtered_contours))
  cv2.imwrite(image_path + '_con.png', con)

def run():
  pass
  # process_jpeg()
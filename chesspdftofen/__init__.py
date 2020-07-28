import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pdf2image import convert_from_path
import tempfile

from segment_boards import segment_boards

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
  boards = segment_boards(im)

plt.show()
def run():
  pass
  # process_jpeg()
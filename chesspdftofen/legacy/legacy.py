import tempfile
from pdf2image import convert_from_path
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def sbw(im):
  f = plt.figure()
  plt.imshow(im, cmap='gray', vmin=0, vmax=255)
  f.show()

def sw(im):
  f = plt.figure()
  plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
  f.show()

def convert_pdf_to_jpeg():
  # with tempfile.TemporaryDirectory() as output_path:
  output_path = 'data\\out'
  data_path = 'data\\23.pdf'
  images_from_path = convert_from_path(data_path, 
    output_folder=output_path,
    fmt="jpeg",
    paths_only=True,
    grayscale=True,
    thread_count=3)
  print(len(images_from_path))

# def process_jpeg():
image_path = 'data\\out\\74a7ca44-f34e-4223-bd8d-416b4374f515-165.jpg'
image_path = 'data\\out\\8ef0055b-d48b-4585-bf30-2a60ab18ad2f-172.png'

image_path = 'data\\out\\620f4fbf-395d-4469-8fe4-c183420057fb-0016.jpg'

image_path = 'data\\dev\\23.jpg'
image_path = 'data\\dev\\23.png'
image_path = 'data\\dev\\1n.jpg'
image_path = 'data\\dev\\1n.png'

im = cv2.imread(image_path, 0)

# imgaussfilt
sigma = 2
ksize = 2*math.ceil(2*sigma)+1
blur = cv2.GaussianBlur(im, (ksize, ksize), 
  sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

ret,th = cv2.threshold(im, 225, 255, cv2.THRESH_BINARY_INV)

sbw(th)

kernel = np.ones((3,3), np.uint8)
dst = cv2.dilate(dst, kernel)
sbw(dst)


# get rid of board edges
kernel = np.ones((5,5), np.uint8)
th2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
sbw(th2)



# adaptthresh
ret,th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

neighborhood_size = 2*np.floor(np.asarray(blur.shape)/16)+1
# pick one dimension (not sure)
blockSize = int(max(neighborhood_size[0], neighborhood_size[1]))
T = cv2.adaptiveThreshold(blur, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
  thresholdType=cv2.THRESH_BINARY_INV, blockSize=blockSize, C=0)




dst = cv2.Canny(im, 50, 200, None, 3)
# cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# retval, corners = cv2.findChessboardCorners(im, patternSize=(7, 7))
cdstP = cv2.imread(image_path, 1)
linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=50, lines=None, minLineLength=50, maxLineGap=6)
if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10:
        pass
      cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

sw(cdstP)



contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours.sort(key=lambda cnt: cv2.contourArea(cnt), reverse=True)
filtered_contours = []
for i, cnt in enumerate(contours):
  area = cv2.contourArea(cnt)
  if not (500 <= area):
    print('area', area)
    continue
  epsilon = 0.01*cv2.arcLength(cnt,True)
  approx = cv2.approxPolyDP(cnt,epsilon,True)
  is_convex = cv2.isContourConvex(cnt)
  num_sides = len(approx)
  if not is_convex or not (4 <= num_sides <= 6):
    print('sides', 'convex', num_sides, is_convex)
    continue
  filtered_contours.append(cnt)

con = cv2.imread(image_path, 1)
cv2.drawContours(con, [filtered_contours[-1]], -1, (0,255,0), -1)
for c in filtered_contours:
  color = list(np.random.random(size=3) * 256)
  cv2.drawContours(con, [c], -1, color, -1);

sw(con)
areas = list(map(lambda cnt: cv2.contourArea(cnt), contours))



dst = cv2.cornerHarris(im,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
harris = cv2.imread(image_path, 1)
# Threshold for an optimal value, it may vary depending on the image.
harris[dst>0.01*dst.max()]=[0,0,255]


cdstP = cv2.imread(image_path, 1)
linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=50, lines=None, minLineLength=100, maxLineGap=6)
if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10:
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

sw(cdstP)


def run():
  pass
  # process_jpeg()


im = cv2.imread(image_path, 0)
ret,th = cv2.threshold(im, 225, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5), np.uint8)
th2 = cv2.dilate(th, kernel)

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
  # check for square
  if not (4 == num_sides):
    # print('sides', num_sides)
    continue
  x, y, w, h = cv2.boundingRect(cnt)
  h_max, w_max = im.shape
  if abs(w - h) > 25:
    # print('x,y,w,h', x, y, w, h)
    continue
  potential_board = im[max(y - 25, 0):min(y + h + 25, h_max), max(x - 25, 0):min(x + w + 25, w_max)] 
  filtered_contours.append(cnt)
  break


for c in filtered_contours:
  color = list(np.random.random(size=3) * 256)
  cv2.drawContours(con, [c], -1, color, -1);

print(len(filtered_contours))

cv2.drawContours(con, [filtered_contours[0]], -1, (0, 255, 0), -1);



dst = cv2.cornerHarris(potential_board,6,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
harris = cv2.cvtColor(potential_board, cv2.COLOR_GRAY2RGB)
# Threshold for an optimal value, it may vary depending on the image.
harris[dst>0.01*dst.max()]=[0,0,255]


dst = cv2.Canny(potential_board, 50, 200, None, 3)
dst = th2[max(y - 25, 0):min(y + h + 25, h_max), max(x - 25, 0):min(x + w + 25, w_max)]
cdstP = cv2.cvtColor(potential_board, cv2.COLOR_GRAY2RGB)
linesP = cv2.HoughLinesP(lap, rho=1, theta=np.pi / 180, threshold=50, lines=None, minLineLength=50, maxLineGap=20)
i = 0
if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10:
      # pass
        color = list(np.random.random(size=3) * 256)
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)
        color[-1] = 0
        cv2.circle(cdstP, (l[0], l[1]), 5, color, -1)
        cv2.circle(cdstP, (l[2], l[3]), 5, color, -1)
        i += 1

sw(cdstP)

lines = cv2.HoughLines(dst, rho=1, theta = np.pi / 180, threshold=0)


_lines, width, prec, nfa  = cv2.LineSegmentDetector.detect(potential_board)

dst = th2[max(y - 25, 0):min(y + h + 25, h_max), max(x - 25, 0):min(x + w + 25, w_max)]

sobelX = cv2.Sobel(potential_board, cv2.CV_64F, 1, 0)
sobelX = cv2.Sobel(dst, cv2.CV_64F, 1, 0)


image = th2[max(y - 25, 0):min(y + h + 25, h_max), max(x - 25, 0):min(x + w + 25, w_max)]
lap = cv2.Laplacian(dst, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
sbw(lap)

# Compute gradients along the X and Y axis, respectively
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

# The sobelX and sobelY images are now of the floating
# point data type -- we need to take care when converting
# back to an 8-bit unsigned integer that we do not miss
# any images due to clipping values outside the range
# of [0, 255]. First, we take the absolute value of the
# graident magnitude images, THEN we convert them back
# to 8-bit unsigned integers
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# We can combine our Sobel gradient images using our
# bitwise OR
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Show our Sobel images
sbw(sobelX)
sbw(sobelY)
sbw(sobelCombined)


cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
linesP = cv2.HoughLinesP(lap, rho=1, theta=np.pi / 180, threshold=50, lines=None, minLineLength=50, maxLineGap=6)
if linesP is not None:
    for i in range(0, len(linesP)):
      l = linesP[i][0]
      # if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10:
      cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

sw(cdstP)
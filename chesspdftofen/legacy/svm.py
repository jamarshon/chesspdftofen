### des1
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
blockSize = (64, 64) #
blockStride = (32, 32) #
cellSize = (16, 16) #

# blockSize = (16, 16) #
# blockStride = (8, 8) #
# cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (125, 125))
    x = hog.compute(im, padding=(3,3))
#     x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print('confusion_matrix', confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)


### des2
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
blockSize = (64, 64) #
blockStride = (32, 32) #
cellSize = (16, 16) #

# blockSize = (16, 16) #
# blockStride = (8, 8) #
# cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (128, 128))
#     x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)


### des3
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
blockSize = (64, 64) #
blockStride = (32, 32) #
cellSize = (16, 16) #

# blockSize = (16, 16) #
# blockStride = (8, 8) #
# cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (128, 128))
#     x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)

### des4
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
# blockSize = (64, 64) #
# blockStride = (32, 32) #
# cellSize = (16, 16) #

blockSize = (16, 16) #
blockStride = (8, 8) #
cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (128, 128))
#     x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)


### des5
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
# blockSize = (64, 64) #
# blockStride = (32, 32) #
# cellSize = (16, 16) #

blockSize = (16, 16) #
blockStride = (8, 8) #
cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (128, 128))
#     x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)

### des6
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
blockSize = (64, 64) #
blockStride = (32, 32) #
cellSize = (16, 16) #

# blockSize = (16, 16) #
# blockStride = (8, 8) #
# cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (128, 128))
#     x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)

### des7
%%time
folder = '..\\data\\out\\1n_final'
piecesnames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
label_len = len(piecesnames)
confusion_matrix = np.zeros((label_len, label_len), dtype=np.int32)

winSize = (128, 128) #
# winSize = (96, 96) #
blockSize = (64, 64) #
blockStride = (32, 32) #
cellSize = (16, 16) #

# blockSize = (16, 16) #
# blockStride = (8, 8) #
# cellSize = (8, 8) #
nbins = 9 #
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True #

hog = cv2.HOGDescriptor(
  winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
  L2HysThreshold, gammaCorrection, nlevels, signedGradients
)

k = 0
for i, piece in enumerate(sorted(os.listdir(folder))):
  folder2 = os.path.join(folder, piece)
  if os.path.isfile(folder2):
    continue
  for j, filename in enumerate(sorted(os.listdir(folder2))):
    fullname = os.path.join(folder2, filename)
    im = cv2.imread(fullname, 0)
    im = cv2.resize(im, (125, 125))
    x = hog.compute(im, padding=(3,3))
    x = hog.compute(im)
    y = int(linear.predict(x.T))
    
    confusion_matrix[i][y] += 1
    if k % 1000 == 0:
      print(k)
    k += 1

expected = 26816
total = np.sum(confusion_matrix)
print(confusion_matrix)
print('accuracy', np.trace(confusion_matrix) / total)
print('total', total)

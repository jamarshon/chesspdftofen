# Chess PDF to FEN
Annotates PDF with comments containing FEN of detected chessboards. Runs completely locally as an offline process with possible GPU support and is open source

## Usage
```python
import chesspdftofen

for status in chesspdftofen.run('path/to/input.pdf', 'path/to/output.pdf'):
  print(status)
```
<p align="center">
  <img src="https://github.com/jamarshon/chesspdftofen/blob/master/readme/output.png" alt="Example output" width="600px">
</p>

### API
```python
def run(file_path, output_file_path, num_threads=4, num_pages_to_print=10, build_training_set=False):
```
Parameter | Description
------------ | -------------
**file_path** (str) | Path to the input pdf file
**output_path** (str) | Path for the output file name
**num_threads** (int, optional) | Number of threads to use (recommended less than 4)
**num_pages_to_print** (int, optional) | Number of pages to process before printing progress
**build_training_set** (bool, optional) | Used to create training data, should be False otherwise

## Setup
### Requirements
Python 3, conda, pip

### Install Dependencies
Before installing this package, there are a few dependencies that are needed to be run in terminal
```bash
# Install pytorch, command differs based on operating system, gpu, etc. so 
# visit https://pytorch.org/get-started/locally/ for complete instructions
conda install pytorch torchvision cpuonly -c pytorch
# Install poppler which allows pdf manipulation
conda install -c conda-forge poppler
```

### Install Package
```bash
pip install chesspdftofen
```

## Algorithm
### Board Detection
Pages of the pdf are converted to images which are thresholded and then square areas of sufficent area that are surrounded by white pixels are identified. These potential boards have a Laplacian filter applied to get edges which are given to the Hough transform. The lines have their intersections calculated and these intersections are clustered into 81 points using k-means. Comparing the 81 intersections with the expected location of intersections give the probability about whether this area is a board.  
<img src="https://github.com/jamarshon/chesspdftofen/blob/master/readme/segment.png">
### Piece Detection
Once a board is detected, it is split into 8 x 8 or 64 tensors of size (64, 64). They are used as input to the CNN with the following architecture.
<img src="https://github.com/jamarshon/chesspdftofen/blob/master/readme/cnn.png">
CNN will classify each cell and this is used to generate the FEN.

## Disclaimer
The segmentation and classification model was trained on a certain style of chessboard and book so it may have inaccuracies on applications that differ drastically. To further improve the model, file a GitHub issue with the name of the book so that future iterations may possibly consider the given variants.


import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
import os
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader
import re
import tempfile
# import tkinter as tk
# from tkinter import Frame, Text, Label
import torch
import torchvision

from .cnn import get_model
from .segment_boards import segment_boards
from .pdf_helper import create_annotation, add_annotation_to_page

def sbw(im):  
  f = plt.figure()
  plt.imshow(im, cmap='gray', vmin=0, vmax=255)
  f.show()

def sw(im):
  f = plt.figure()
  plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
  f.show()

def run_gui():
  pass
  # window = tk.Tk()
  # window.title('Chess PDF to FEN')
  # window.state('zoomed')


  # main_container = Frame(master=window, background="grey")
  # main_container.pack(side="top", fill="both", expand=True)

  # top_frame = Frame(main_container, background="green")
  # bottom_frame = Frame(main_container, background="yellow")

  # top_frame.pack(side="top", fill="x", expand=False)
  # bottom_frame.pack(side="bottom", fill="both", expand=True)
  
  # welcome_lbl = tk.Label(master=top_frame, text="Welcome to Chess PDF to FEN")
  # welcome_lbl.config(font=("Consolas ", 24))
  # welcome_lbl.grid(row=0, column=1)
  # welcome_lbl.pack(side="top")

  # frm_pdf_select = tk.Frame(master=bottom_frame)
  # frm_pdf_select.pack(side="top", fill="x")
  # lbl_pdf_select = tk.Label(master=frm_pdf_select, text="Select a PDF file")
  # lbl_pdf_select.config(font=("Consolas ", 16))
  # lbl_pdf_select.pack(side="left")
  # window.mainloop()

def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

# piecenames = ['BlackBishop', 'BlackKing', 'BlackKnight', 'BlackPawn', 'BlackQueen', 'BlackRook', 'BlackSpace', 'WhiteBishop', 'WhiteKing', 'WhiteKnight', 'WhitePawn', 'WhiteQueen', 'WhiteRook', 'WhiteSpace']
piecenames = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'em', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr', 'em']

def get_fen_str(predicted):
  with io.StringIO() as s:
    s.write('https://lichess.org/analysis/')
    for row in range(8):
      empty = 0
      for cell in range(8):
        c = piecenames[predicted[row*8 + cell]]
        if c[0] in ('w', 'b'):
          if empty > 0:
            s.write(str(empty))
            empty = 0
          s.write(c[1].upper() if c[0] == 'w' else c[1].lower())
        else:
          empty += 1

      if empty > 0:
        s.write(str(empty))
      s.write('/')
    # Move one position back to overwrite last '/'
    s.seek(s.tell() - 1)
    # If you do not have the additional information choose what to put
    # s.write(' w KQkq - 0 1')
    s.write('%20w%20KQkq%20-%200%201')
    return s.getvalue()

def run(file_path, output_file_path, num_threads=4, num_pages_to_print=50):
  r"""
  file_path           (str): Path to the pdf file
  output_path         (str): Path for the output file name
  num_threads         (int, optional): Number of threads to used (recommended less than 4)
  num_pages_to_print  (int, optional): Number of pages to process before printing progress

  Example usage
  chesspdftofen.run('data/yasser.pdf', 'data/yasser2.pdf')  
  """
  net = get_model()

  transform = torchvision.transforms.Compose([
    # torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((8*64, 8*64)),
    # GaussianSmoothing([0, 5]),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, ), (0.5, ))
  ])

  pdf_input = PdfFileReader(open(file_path, 'rb'), strict=False)
  pdf_output = PdfFileWriter()
  pdf_output.appendPagesFromReader(pdf_input)

  num_pages = pdf_output.getNumPages()
  assert num_pages > 0
  page1 = pdf_output.getPage(0)
  _, _, pdf_w, pdf_h = page1.mediaBox

  with torch.no_grad():
    with tempfile.TemporaryDirectory() as output_path:
      im_paths = convert_from_path(
        file_path, 
        output_folder=output_path,
        fmt="jpg",
        paths_only=True,
        grayscale=True,
        thread_count=num_threads)

      print('Converting %s' % (file_path,))
      num_pages = len(im_paths)
      
      for i, im_path in enumerate(im_paths):
        if i % num_pages_to_print == 0:
          print('Completed %d / %d' % (i, num_pages))

        if i > 50:
          break

        im_path_base, im_path_ext = os.path.splitext(im_path)
        page_num = re.match('.*?([0-9]+)$', im_path_base)
        assert page_num is not None
        page_num = int(page_num.group(1)) - 1

        page_im = cv2.imread(im_path, 0)
        page_im_h, page_im_w = page_im.shape
        boards = segment_boards(page_im)

        for board in boards:
          ymin, ymax, xmin, xmax, _, _ = board
          board_im = page_im[ymin:ymax, xmin:xmax]

          im = Image.fromarray(board_im)
          # im = pil_loader(path)
          im = transform(im)
          dim = 64
          tensors = [im[:, dim*k: dim*(k+1), dim*j: dim*(j+1)] for k in range(8) for j in range(8)]
          images = torch.stack(tensors)

          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          
          fen_str  = get_fen_str(predicted)

          annotation = create_annotation(xmax / page_im_w * pdf_w, (1 - ymin / page_im_h) * pdf_h, {
            'author': '',
            'contents': fen_str
          })
          add_annotation_to_page(annotation, pdf_output.getPage(page_num), pdf_output)
  pdf_output.write(open(output_file_path, 'wb'))

if __name__ == "__main__":
  run()

__version__ = '0.5.0'
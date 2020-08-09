from distutils.core import setup
setup(
  name = 'chesspdftofen',
  packages = ['chesspdftofen', 'chesspdftofen.data'],
  package_data = {
    'chesspdftofen': ['data/*.dat']
  },
  version = '0.5.0',
  license='MIT',
  description = 'Annotates PDF with comments including FEN. Runs completely locally and is open source',
  url = 'https://github.com/jamarshon/chesspdftofen',
  # download_url = 'https://github.com/jamarshon/chesspdftofen/archive/0.1.tar.gz',
  keywords = ['chess', 'computervision'],
  install_requires=[
          'importlib_resources',
          'matplotlib',
          'numpy',
          'opencv-python',
          # 'opencv-contrib-python',
          'pdf2image',
          'PyPDF2',
          'pillow',
          # 'scipy',
          # 'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
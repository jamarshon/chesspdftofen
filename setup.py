from distutils.core import setup
setup(
  name = 'chesspdftofen',
  packages = ['chesspdftofen'],
  version = '0.1.1',
  license='MIT',
  description = 'Annotates PDF with comments including FEN. Runs completely locally and is open source',
  author = 'jamarshon',
  author_email = 'ultralisk27@gmail.com',
  url = 'https://github.com/jamarshon/chesspdftofen',
  # download_url = 'https://github.com/jamarshon/chesspdftofen/archive/0.1.tar.gz',
  keywords = ['chess', 'computervision'],
  install_requires=[
          'numpy',
          'opencv-python',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
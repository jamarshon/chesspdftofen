from distutils.core import setup
setup(
  name = 'chesspdftofen',
  packages = ['chesspdftofen'],
  version = '0.1',
  license='MIT',
  description = 'Annotates PDF with comments including FEN. Runs completely locally and is open source',
  author = 'jamarshon',
  url = 'https://github.com/jamarshon/chesspdftofen',
  download_url = 'https://github.com/jamarshon/chesspdftofen/archive/v_01.tar.gz',
  keywords = ['chess', 'computervision'],
  install_requires=[            # I get to this in a second
          'numpy',
          'opencv-python',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)
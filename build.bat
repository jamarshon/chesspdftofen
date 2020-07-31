pip uninstall chesspdftofen -y
rm dist/*
python setup.py sdist
cd dist
pip install chesspdftofen-0.1.1.tar.gz
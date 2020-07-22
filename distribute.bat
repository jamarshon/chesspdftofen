rm dist/*
python setup.py sdist
twine upload --skip-existing dist/*
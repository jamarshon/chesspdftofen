## Miscellaneous Notes for Contributors and Developers
```bash
# Create a conda environment (optional)
conda create --name chesspdftofen python=3.8 --yes
# Activate environment. For Windows its just "activate chesspdftofen"
conda activate chesspdftofen

# Adding to requirements.txt
pip freeze > requirements.txt
# Creating distribution
python setup.py sdist
# Uploading files
twine upload --skip-existing dist/*

# Deleting release
git push --delete origin 0.2

# run file
python -m chesspdftofen.__init__
```
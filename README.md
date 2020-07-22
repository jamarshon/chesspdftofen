# Chess PDF to FEN
Annotates PDF with comments including FEN. Runs completely locally and is open source

## Setup
```bash
# Install the package (only needs to be done once)
pip install chesspdftofen
# Run the package/GUI
python -c "import chesspdftofen; chesspdftofen.run()"

``` 

## Miscellaneous Developer Notes
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
```

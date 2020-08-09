# Chess PDF to FEN
Annotates PDF with comments including FEN. Runs completely locally and is open source

## Setup
```bash
# Install the package and dependencies (only needs to be done once)
# Install pytorch, command differs based on operating system, gpu, etc. so 
# see https://pytorch.org/get-started/locally/
conda install pytorch torchvision cpuonly -c pytorch
conda install -c conda-forge poppler
pip install chesspdftofen

# Run the package/GUI
python -c "import chesspdftofen; chesspdftofen.run()"

```

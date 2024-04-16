# Video detection task
Using model:
- __Faster R-CNN__
## Local development
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv/Source/activate

# Install torch with cuda
pip install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# Install other dependencies
pip install -r requirements.txt
# Run example
python task.py --input inference/input/crowd.mp4 --device cuda --thresh 0.7 --imgsize 512
```

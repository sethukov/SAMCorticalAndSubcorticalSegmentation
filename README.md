Brain segmentation using Meta's Segment Anything Model. 

# Create an environment called deep
conda activate deep 

# Installed packages
 conda install anaconda::opencv

# Upgraded packages
conda upgrade setuptools

# Installed SAM
git clone git@github.com:facebookresearch/segment-anything.git
pip install -e .

# Running instructions
python -i sam_testing.py

# Resources 
https://docs.ultralytics.com/models/sam-2/#core-capabilities
https://github.com/facebookresearch/segment-anything?tab=readme-ov-file
https://ai.meta.com/blog/segment-anything-model-3/

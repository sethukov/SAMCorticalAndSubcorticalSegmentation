# Code for prediction using the pre-trained model 

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



# 1. Load model and set device (e.g., "cuda" or "cpu")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 2. Configure the automatic mask generator
mask_generator = SamAutomaticMaskGenerator(model=sam)

# 3. Load and process image (OpenCV loads in BGR, convert to RGB)
image = cv2.imread('/media/sethu/STORAGE10TB/PROJECTS_DEEPLEARNING/CorticalAndSubcorticalSegmentation/h1mri.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 4. Generate masks
masks = mask_generator.generate(image)

print(f"Generated {len(masks)} masks")

# Visualize output mask
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('sam_output.png')
plt.show() 


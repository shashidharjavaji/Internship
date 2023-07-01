import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_score

# Load the predicted and ground truth segmentations
pred_img = nib.load('final_segmentation.nii.gz')
true_img = nib.load('ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00002-000/BraTS-PED-00002-000-seg.nii.gz')

# Get the data arrays from the images
pred_data = pred_img.get_fdata()
true_data = true_img.get_fdata()

# Ensure the data arrays have the same shape
assert pred_data.shape == true_data.shape

# Define the number of classes in your segmentation
num_classes = 4  # Modify this if needed

# Initialize an array to hold the DSC for each class
dsc_per_class = np.zeros(num_classes)

# Calculate the DSC for each class
for i in range(num_classes):
    # Create binary masks for class i in the predicted and ground truth data
    pred_mask = pred_data == i
    true_mask = true_data == i
    
    # Calculate the DSC for class i
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    dsc_per_class[i] = 2. * intersection / union

# Print the DSC for each class
for i, dsc in enumerate(dsc_per_class):
    print(f'DSC for class {i}: {dsc:.4f}')


import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import surfer
from surfer import Brain, project_volume_data
from mayavi import mlab


def make_surface_mask(img, thresh=None, hemi='lh', use_nans=True):
    """Convert ROI mask to surface array"""
    surface_data = project_volume_data(img, 'lh', subject_id='fsaverage')

    if use_nans:
        mask = np.where(surface_data > thresh, 1, np.nan)
    else:
        mask = np.where(surface_data > thresh, 1, 0)

    return mask


def make_random_weights(mask):
    """Transform values within mask to random numbers"""
    rand_values = np.random.rand(*mask.shape)
    return mask * rand_values

def take_screenshot(img):
    """
    Exception handling for bizarre pysurfer/mayavi issue where a ValueError
    is raised in the first attempt, but works just fine re-running the command
    """
    try:
        arr = img.screenshot()
    except ValueError:
        arr = img.screenshot()
    return arr


if __name__ == '__main__':

    mask_image = 'figures/auditory_cortex_mask.png'

    roi = make_surface_mask('l_auditory_cortex.nii.gz', 4)
    roi[np.isnan(roi)] = -11

    mask_brain = Brain('fsaverage', 'lh', "pial_semi_inflated", background="white",
              cortex=("binary", -4, 8, False), size=(1000, 600))
    mask_brain.add_data(roi, thresh=-10, min=0, max=1, colormap='Purples',
               colorbar=False, alpha=.5)

    mlab.view(distance=300)
    image_array = take_screenshot(mask_brain)
    mask_brain.close()

    plt.imshow(image_array, rasterized=True)
    plt.axis('off')

    plt.show()




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


def plot_roi_mask(filename, threshold, hemi='lh'):
    """Create a colour mask for ROI"""

    roi = make_surface_mask(filename, threshold, hemi)

    brain = Brain('fsaverage', hemi, "pial", background="white",
                  cortex='low_contrast', size=(1000, 600))
    # replace nans so that they can be thresholded off when adding data
    roi[np.isnan(roi)] = -11
    brain.add_data(roi, thresh=-10, min=0, max=1, colormap='tab10',
                   colorbar=False, alpha=.8)
    return brain


def plot_roi_weights(filename, threshold, hemi='lh'):

    roi = make_surface_mask(filename, threshold, hemi)

    # transform values within mask into random numbers
    roi = roi * np.random.uniform(-1, 1, size=roi.shape)

    brain = Brain('fsaverage', hemi, "inflated", background="white",
                  cortex='low_contrast', size=(1000, 600))
    # replace nans so that they can be thresholded off when adding data
    roi[np.isnan(roi)] = -11
    brain.add_data(roi, thresh=-10, min=-1, max=1, colormap='bwr',
                   colorbar=False)
    return brain


def take_screenshot(img):
    """
    Exception handling for bizarre pysurfer/mayavi issue where a ValueError
    is raised in the first attempt, but works just fine re-running the command
    """
    try:
        arr = img.screenshot()
    except ValueError:
        arr = img.screenshot()

    img.close()
    return arr


if __name__ == '__main__':

    file_name = 'l_inf_temp_ant.nii.gz'
    roi_mask = plot_roi_mask(file_name, 5, 'lh')
    mlab.view(distance=300)
    arr = take_screenshot(roi_mask)

    plt.figure(1)
    plt.imshow(arr[50:, :], rasterized=True)
    plt.axis('off')

    plt.show()


    roi_weights = plot_roi_weights(file_name, 5, 'lh')
    mlab.view(158, 96, 60, [-34, 17, -54])
    arr = take_screenshot(roi_weights)

    plt.figure(2)
    plt.imshow(arr[50:, :], rasterized=True)
    plt.axis('off')

    plt.show()



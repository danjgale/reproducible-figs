
import numpy as np
import nibabel as nib
import surfer
from surfer import Brain, project_volume_data
import matplotlib.pyplot as plt


def make_random_weights(img, thresh=None):
    """Convert binary mask image to a mock decoding map"""

    # extract a binary mask
    surface_data = project_volume_data(img, 'rh', subject_id='fsaverage')
    surface_data = np.where(surface_data > thresh, 1, np.nan)

    # set mask to random values
    rand_values = np.random.rand(*surface_data.shape)
    return surface_data * rand_values


if __name__ == '__main__':


    roi = make_random_weights('l_aSTS.nii.gz', 4)
    roi[np.isnan(roi)] = -11

    b = Brain('fsaverage', 'rh', "inflated", background="white",
              cortex=("binary", -4, 8, False), size=(1000, 600))
    b.add_data(roi, thresh=-10, min=0, max=1, colormap='viridis',
               colorbar=False)


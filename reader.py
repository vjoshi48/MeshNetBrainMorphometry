from typing import List, Optional

import nibabel as nib
import numpy as np

from catalyst.contrib.data.reader import IReader

from freesurfer_stats import CorticalParcellationStats

class NiftiReader(IReader):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from
    a `csv` dataset.
    """

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        image_name = str(element[self.input_key])
        img = nib.load(image_name)
        output = {self.output_key: img}
        return output

class NiftiGrayMatterReader(IReader):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images from
    a `csv` dataset.
    """

    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        rootpath: Optional[str] = None,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        gray_matter = element[self.input_key]
        gray_matter = gray_matter.split(',')

        stats_r_path = gray_matter[0][2:-1]
        stats_l_path = gray_matter[1][2:-2]

        #using volume path for right and left hemispheres to read in data for each hemisphere
        stats_r = CorticalParcellationStats.read(stats_r_path)
        stats_l = CorticalParcellationStats.read(stats_l_path)

        #creating measurement dataframes and renaming them
        df_r = stats_r.structural_measurements[['gray_matter_volume_mm^3']]
        df_l = stats_l.structural_measurements[['gray_matter_volume_mm^3']]

        array_l = df_l.values
        array_r = df_r.values

        array_both = np.concatenate((array_l, array_r), axis=0)

        output = {self.output_key: array_both}
        return output

class NiftiFixedVolumeReader(NiftiReader):
    """
    Nifti reader abstraction for NeuroImaging. Reads nifti images
    from a `csv` dataset.
    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        rootpath: str = None,
        volume_shape: List = None,
    ):
        """
        Args:
            input_key (str): key to use from annotation dict
            output_key (str): key to use to store the result
            rootpath (str): path to images dataset root directory
                (so your can use relative paths in annotations)
            coords (list): crop coordinaties
        """
        super().__init__(input_key, output_key or input_key)
        self.rootpath = rootpath
        if volume_shape is None:
            volume_shape = [256, 256, 256]
        self.volume_shape = volume_shape

    def __call__(self, element):
        """Reads a row from your annotations dict with filename and
        transfer it to an image
        Args:
            element: elem in your dataset.
        Returns:
            np.ndarray: Image
        """
        image_name = str(element[self.input_key])
        img = nib.load(image_name)
        img = img.get_fdata()
        img = (img - img.min()) / (img.max() - img.min())
        new_img = np.zeros(self.volume_shape)
        new_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        output = {self.output_key: new_img}
        return output

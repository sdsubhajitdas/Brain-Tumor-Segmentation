import torch
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
from skimage.transform import resize
import os


class TumorDataset(Dataset):
    """ Returns a TumorDataset class class object which represents our tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, root_dir, transform=None):
        """ Constructor for our TumorDataset class.
        Parameters:
            root_dir(str): Directory with all the images.
            transform(callable, optional): Optional transform to be applied
                on a sample.

        Returns: None
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """ Overridden method from inheritted class so that
        len(self) returns the size of the dataset. 
        """
        error_msg = 'Part of dataset is missing!\nNumber of tumor and mask images are not same.'
        total_files = len(os.listdir(self.root_dir))

        assert (total_files % 2 == 0), error_msg
        return total_files//2

    def __getitem__(self, index):
        """ Overridden method from inheritted class to support
        indexing of dataset such that datset[I] can be used
        to get Ith sample.
        Parameters:
            index(int): Index of the dataset sample

        Return:
            sample(dict): Contains the index, image, mask ndarray.(Grayscaled & Normalized)
                         'index': Index of the image.
                         'image': Contains the tumor image numpy array.
                         'mask' : Contains the mask image numpy array.
        """
        image_name = os.path.join(self.root_dir, str(index)+'.png')
        mask_name = os.path.join(self.root_dir, str(index)+'_mask.png')

        image = io.imread(image_name, as_gray=True)
        mask = io.imread(mask_name, as_gray=True)

        # Not all images are resized.
        # But if any image dimension is off then it will be resized.
        if image.shape != (512, 512):
            image = resize(image, (512, 512))
        if mask.shape != (512, 512):
            mask = resize(mask, (512, 512))

        image = torch.Tensor(image).view((1, 512, 512))
        mask = torch.Tensor(mask).view((1, 512, 512))

        # Pytorch transformations applied if any.
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        sample = {'index': int(index),'image': image, 'mask': mask}

        return sample

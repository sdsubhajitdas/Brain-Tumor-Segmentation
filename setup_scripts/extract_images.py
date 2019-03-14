import h5py
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm
import os


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters: 
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def get_image_data(filename, path):
    """ Reads the mat image file and returns the image & mask array.
    Parameters:
        filename(str): Name of the file without the extension.
        path(str): Path where the filename is located.

    Returns:
        data(dict): A dictionary with the image & mask numpy array.
                    'image': The numpy array for image.
                    'mask' : The numpy array for the above image mask.
    """
    path = os.path.join(path, filename+'.mat')
    file = h5py.File(path, 'r')
    data = dict()
    data['image'] = np.array(file.get('cjdata/image'))
    data['mask'] = np.array(file.get('cjdata/tumorMask'))
    return data


def save_image_data(filename, path, data):
    """ Saves the image & mask array in png format.
    Parameters:
        filename(str): Name of the file without the extension.
        path(str): Path where the filename is to be saved.
        data(dict): A dictionary with the image & mask numpy array.
                    'image': The numpy array for image.
                    'mask' : The numpy array for the above image mask.

    Returns: None
    """
    path_image = os.path.join(path, filename+'.png')
    path_mask = os.path.join(path, filename+'_mask.png')
    mpimg.imsave(path_image, data['image'], cmap='gray', format='png')
    mpimg.imsave(path_mask, data['mask'], cmap='gray', format='png')


def main():
    # Total number of images
    total_images = 3064

    # Dataset paths
    data_read_path = os.path.join('dataset', 'mat_dataset')
    data_save_path = os.path.join('dataset', 'png_dataset')

    clear_screen()

    # Make if folder is missing.
    make_folder(data_save_path)

    print(f'Starting to save images in {data_save_path}')

    for filename in tqdm(range(1, total_images+1)):
        filename = str(filename)
        data = get_image_data(filename, data_read_path)
        save_image_data(str(int(filename)-1), data_save_path, data)


if __name__ == "__main__":
    main()

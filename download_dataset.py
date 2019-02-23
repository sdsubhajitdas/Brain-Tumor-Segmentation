import requests
from tqdm import tqdm
import os
import argparse


def check_if_file_exits(file):
    """ Checks if the file specified is downloaded or not.
    Parameters:
        file(str): Name of the file to be checked.

    Returns: None
    """
    extension = file[-3:]
    file = file[:-4] + '_done.'+extension
    return True if os.path.isfile(file) else False


def download_file(url, path):
    """ Download the file in url to the path specified.
    Parameters:
        url(str): URL of the file to be downloaded.
        path(str): Destination where the downloaded file will be saved.

    Returns: None
    """
    # Check if file already exists.
    if check_if_file_exits(path):
        print(f'Already existing file {path}')
        return

    # Deleting the partial downloaded file.
    if os.path.isfile(path):
        print(f'Deleted existing partial file {path}')
        os.remove(path)

    response = requests.get(url, stream=True)
    handle = open(path, "wb")
    with open(path, "wb") as handle:
        chunk_size = 1024
        total_size = round(int(response.headers['Content-Length']), 3)
        pbar = tqdm(unit="B", total=total_size)
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)
                pbar.update(len(chunk))

    # Marking the file as downloaded.
    extension = path[-3:]
    os.rename(path, path[:-4]+'_done.'+extension)


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters: 
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def main():
    # URL of the dataset used.
    dataset_urls = ['https://ndownloader.figshare.com/files/3381290',
                    'https://ndownloader.figshare.com/files/3381296',
                    'https://ndownloader.figshare.com/files/3381293',
                    'https://ndownloader.figshare.com/files/3381302']

    # URL of dataset README
    dataset_readme = 'https://ndownloader.figshare.com/files/7953679'

    target_folder = 'dataset'
    dataset_part = 1
    dataset_file_name = f'brain_tumor_dataset_part_'

    clear_screen()
    make_folder(target_folder)

    print(f'\n\tDownloading dataset README.txt')
    download_file(dataset_readme, os.path.join(target_folder, 'README.TXT'))

    print('\n\tStarting download process\n')
    for url in dataset_urls:
        try:
            path = os.path.join(
                target_folder, f'{dataset_file_name}{dataset_part}.zip')
            print(f'\t\tDownloading :  {path}')
            download_file(url, path)
            dataset_part += 1
        except KeyboardInterrupt:
            print('\t\t\n\nDownload stopped')
            break


if __name__ == "__main__":
    main()

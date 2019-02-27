from zipfile import ZipFile
import os


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def unzip_file(source_name, destination):
    """ Unizips a zip file and stores the contents in destination folder.
    Parameters:
        source_name(str): Full path of the source path
        destination(str): Full folder path where contents of source_name will be stored.

    Returns: None
    """
    with ZipFile(source_name, 'r') as zipfile:
        # extracting all the files
        print(f'\tExtracting files of {source_name}')
        zipfile.extractall(destination)
        print(f'\tDone with {source_name}')


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters: 
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def main():
    # Clears the screen.
    clear_screen()

    # File names in a list.
    file_names = [
        f'brain_tumor_dataset_part_{i}_done.zip' for i in range(1, 5)]

    # Destination folder to store files.
    destination = os.path.join('dataset', 'mat_dataset')
    # Make the destination folder.
    make_folder(os.path.join('dataset', 'mat_dataset'))

    for file in file_names:
        path = os.path.join('dataset', file)
        unzip_file(path, destination)


if __name__ == "__main__":
    main()

import argparse
import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

from bts.model import DynamicUNet
from bts.classifier import BrainTumorClassifier


def get_arguments():
    """Returns the command line arguments as a dict"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False, type=str,
                        help='Single input file name.')
    parser.add_argument('--dir', required=False,
                        type=str, help='Directory name with input images')
    parser.add_argument('--ofp', required=False,
                        type=str, help='Single output file path with name.Use this if using "file" flag.')
    parser.add_argument('--odp', required=False,
                        type=str, help='Directory path for output images.Use this if using "dir" flag.')
    args = parser.parse_args()
    args = {'file': args.file, 'folder': args.dir,
            'ofp': args.ofp, 'odp': args.odp}
    return args


class Api:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def call(self, file, folder, ofp, odp):
        """Method saves the predicted image by taking different parameters."""
        if file != None and folder != None:
            print('"folder" flag and "file" flag cant be used together')
            return

        model = self._load_model()
        save_path = None
        # For a single file
        if file != None:
            image = self._get_file(file)
            output = self._get_model_output(image, model)

            name, extension = file.split('.')
            save_path = name+'_predicted'+'.'+extension
            if ofp:
                save_path = os.path.join(ofp,save_path)
                
            self._save_image(output, save_path)
            print(f'Output Image Saved At {save_path}')

        elif folder != None:
            image_list = os.listdir(folder)
            for file in image_list:
                file_name = os.path.join(folder, file)
                image = self._get_file(file_name)
                output = self._get_model_output(image, model)

                name, extension = file.split('.')
                save_path = name+'_predicted'+'.'+extension

                save_path = os.path.join(
                    odp, save_path) if odp else os.path.join(folder, save_path)
                self._save_image(output, save_path)
                print(f'Output Image Saved At {save_path}')

    def _load_model(self):
        """Load the saved model and return it."""
        filter_list = [16, 32, 64, 128, 256]

        model = DynamicUNet(filter_list).to(self.device)
        classifier = BrainTumorClassifier(model, self.device)
        model_path = os.path.join(
            'saved_models', 'UNet-[16, 32, 64, 128, 256].pt')
        classifier.restore_model(model_path)
        print(
            f'Saved model at location "{model_path}" loaded on {self.device}')
        return model

    def _get_model_output(self, image, model):
        """Returns the saved model output"""
        image = image.view((-1, 1, 512, 512)).to(self.device)
        output = model(image).detach().cpu()
        output = (output > 0.5)
        output = output.numpy()
        output = np.resize((output * 255), (512, 512))
        return output

    def _save_image(self, image, path):
        """Save the image to storage specified by path"""
        image = Image.fromarray(np.uint8(image), 'L')
        image.save(path)

    def _get_file(self, file_name):
        """Load the image by taking file name as input"""
        default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])

        image = default_transformation(Image.open(file_name))
        return TF.to_tensor(image)


if __name__ == "__main__":
    args = get_arguments()
    api = Api()
    api.call(**args)

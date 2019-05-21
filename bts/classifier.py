import torch
import bts.loss as loss
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np

from datetime import datetime
from time import time


class BrainTumorClassifier():
    """ Returns a BrainTumorClassifier class object which represents our 
    optimizer for our network.
    """

    def __init__(self, model, device):
        """ Constructor for our BrainTumorClassifier class.
        Parameters:
            model(DynamicUNet): UNet model to be trained.
            device(torch.device): Device currently used for all computations.

        Returns: 
            None
        """
        self.model = model
        self.device = device
        self.criterion = loss.BCEDiceLoss(self.device).to(device)
        self.log_path = datetime.now().strftime("%I-%M-%S_%p_on_%B_%d,_%Y")

    def train(self, epochs, trainloader, mini_batch=None, learning_rate=0.001, save_best=None, plot_image=None):
        """ Train the model using Adam Optimizer.
        Parameters:
            epochs(int): Number of epochs for the training session.
            trainloader(torch.utils.data.Dataloader): Training data
                        loader for the optimizer.
            mini_batch(int): Used to print logs for epoch batches.
                            If None then batch logs won't be printed.
                            Default: None
            learning_rate(float): Learning rate for optimizer.
                                  Default: 0.001
            save_best(str): Path to save the best model. At the end of 
                            the training the epoch with losest loss will
                            be saved. If None then model won't be saved.
                            Default: None
            plot_image(list): Plot some samples in Tensorboard while training.
                          Visualization of model training progress.If None
                          then nothing will be done.
                          Default: None

        Returns:
            history(dict): Contains information about training session.
                            'train_loss': List of loss at every epoch
        """
        # Tensorboard Writter
        self.tb_writer = SummaryWriter(log_dir=f'logs/{self.log_path}')
        # Training session history data.
        history = {'train_loss': list()}
        # For save best feature. Initial loss taken a very high value.
        last_loss = 1000
        # Optimizer used for training process. Adam Optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Reducing LR on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.85, patience=2, verbose=True)
        print('Starting Training Process')
        # Epoch Loop
        for epoch in range(epochs):
            start_time = time()
            # Training a single epoch
            epoch_loss = self._train_epoch(trainloader, mini_batch)
            # Collecting all epoch loss values for future visualization.
            history['train_loss'].append(epoch_loss)
            # Logging to Tensorboard
            self.tb_writer.add_scalar('Train Loss', epoch_loss, epoch)
            self.tb_writer.add_scalar(
                'Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
            # Reduce LR On Plateau
            self.scheduler.step(epoch_loss)

            # Plotting some sample output on TensorBoard for visualization purpose.
            if plot_image:
                self.model.eval()
                self._plot_image(epoch, plot_image)
                self.model.train()

            time_taken = time()-start_time
            # Training Logs printed.
            print(f'Epoch: {epoch+1:03d},  ', end='')
            print(f'Loss:{epoch_loss:.7f},  ', end='')
            print(f'Time:{time_taken:.2f}secs', end='')

            # Save the best model with lowest epoch loss feature.
            if save_best != None and last_loss > epoch_loss:
                self.save_model(save_best)
                last_loss = epoch_loss
                print(f'\tSaved at loss: {epoch_loss:.10f}')
            else:
                print()
        return history

    def save_model(self, path):
        """ Saves the currently used model to the path specified.
        Follows the best method recommended by Pytorch
        Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
        Parameters:
            path(str): The file location where the model needs to be saved.
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)

    def restore_model(self, path):
        """ Loads the saved model and restores it to the "model" object.
        Loads the model based on device used for computation.(CPU/GPU) 
        Follows the best method recommended by Pytorch
        Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
        Parameters:
            path(str): The file location where the model is saved.
        Returns:
            None
        """
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)

    def test(self, testloader, threshold=0.5):
        """ To test the performance of model on testing dataset.
        Parameters:
            testloader(torch.utils.data.Dataloader): Testing data
                        loader for the optimizer.
            threshold(float): Threshold value after which value will be part 
                              of output.
                              Default: 0.5

        Returns:
            mean_val_score(float): The mean Sørensen–Dice Coefficient for the 
                                    whole test dataset.
        """
        # Putting the model to evaluation mode
        self.model.eval()
        # Getting test data indices for dataloading
        test_data_indexes = testloader.sampler.indices[:]
        # Total testing data used.
        data_len = len(test_data_indexes)
        # Score after testing on dataset.
        mean_val_score = 0

        # Error checking to set testloader batch size to 1.
        batch_size = testloader.batch_size
        if batch_size != 1:
            raise Exception("Set batch size to 1 for testing purpose")
        # Converting to iterator to get data in loops.
        testloader = iter(testloader)
        # Running the loop until no more data is left to test.
        while len(test_data_indexes) != 0:
            # Getting a data sample.
            data = testloader.next()
            # Getting the data index
            index = int(data['index'])
            # Removing the data index from total data indices
            # to indicate this data score has been included.
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue
            # Data prepared to be given as input to model.
            image = data['image'].view((1, 1, 512, 512)).to(self.device)
            mask = data['mask']

            # Predicted output from the input sample.
            mask_pred = self.model(image).cpu()
            # Threshold elimination.
            mask_pred = (mask_pred > threshold)
            mask_pred = mask_pred.numpy()
            
            mask = np.resize(mask, (1, 512, 512))
            mask_pred = np.resize(mask_pred, (1, 512, 512))
            
            # Calculating the dice score for original and 
            # constructed image mask.
            mean_val_score += self._dice_coefficient(mask_pred, mask)

        # Calculating the mean score for the whole test dataset.
        mean_val_score = mean_val_score / data_len
        # Putting the model back to training mode.
        self.model.train()
        return mean_val_score

    def predict(self, data, threshold=0.5):
        """ Calculate the output mask on a single input data.
        Parameters:
            data(dict): Contains the index, image, mask torch.Tensor.
                        'index': Index of the image.
                        'image': Contains the tumor image torch.Tensor.
                        'mask' : Contains the mask image torch.Tensor.
            threshold(float): Threshold value after which value will be part of output.
                                Default: 0.5

        Returns:
            image(numpy.ndarray): 512x512 Original brain scanned image.
            mask(numpy.ndarray): 512x512 Original mask of scanned image.
            output(numpy.ndarray): 512x512 Generated mask of scanned image.
            score(float): Sørensen–Dice Coefficient for mask and output.
                            Calculates how similar are the two images.
        """
        self.model.eval()
        image = data['image'].numpy()
        mask = data['mask'].numpy()

        image_tensor = torch.Tensor(data['image'])
        image_tensor = image_tensor.view((-1, 1, 512, 512)).to(self.device)
        output = self.model(image_tensor).detach().cpu()
        output = (output > threshold)
        output = output.numpy()

        image = np.resize(image, (512, 512))
        mask = np.resize(mask, (512, 512))
        output = np.resize(output, (512, 512))
        score = self._dice_coefficient(output, mask)
        return image, mask, output, score

    def _train_epoch(self, trainloader, mini_batch):
        """ Training each epoch.
        Parameters:
            trainloader(torch.utils.data.Dataloader): Training data
                        loader for the optimizer.
            mini_batch(int): Used to print logs for epoch batches.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """
        epoch_loss, batch_loss, batch_iteration = 0, 0, 0
        for batch, data in enumerate(trainloader):
            # Keeping track how many iteration is happening.
            batch_iteration += 1
            # Loading data to device used.
            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)
            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()
            # Calculation predicted output using forward pass.
            output = self.model(image)
            # Calculating the loss value.
            loss_value = self.criterion(output, mask)
            # Computing the gradients.
            loss_value.backward()
            # Optimizing the network parameters.
            self.optimizer.step()
            # Updating the running training loss
            epoch_loss += loss_value.item()
            batch_loss += loss_value.item()

            # Printing batch logs if any.
            if mini_batch:
                if (batch+1) % mini_batch == 0:
                    batch_loss = batch_loss / \
                        (mini_batch*trainloader.batch_size)
                    print(
                        f'    Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0

        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss

    def _plot_image(self, epoch, sample):
        """
        Parameters:
            epoch(int): Running epoch number used to plot on Tensorboard
            sample(list): Sample inputs used to visualize the progress of
                          training over epochs.
        Returns:
            None
        """
        inputs = list()
        mask = list()

        # Inputs seperated.
        for data in sample:
            inputs.append(data['image'])
        # Inputs stacked together in a single batch
        inputs = torch.stack(inputs).to(self.device)
        # Outputs gained from model after passing input.
        outputs = self.model(inputs).detach().cpu()
        # Adding the outputs to Tensorboard for visualization.
        for index in range(len(sample)):
            self.tb_writer.add_image(
                str(sample[index]['index']), outputs[index], epoch)
        # Deleting the samples from GPU memory to save space.
        del inputs

    def _dice_coefficient(self, predicted, target):
        """Calculates the Sørensen–Dice Coefficient for a
        single sample.
        Parameters:
            predicted(numpy.ndarray): Predicted single output of the network.
                                    Shape - (Channel,Height,Width)
            target(numpy.ndarray): Actual required single output for the network
                                    Shape - (Channel,Height,Width)

        Returns:
            coefficient(float): Dice coefficient for the input sample.
                                        1 represents high similarity and
                                        0 represents low similarity.
        """
        smooth = 1
        product = np.multiply(predicted, target)
        intersection = np.sum(product)
        coefficient = (2*intersection + smooth) / \
            (np.sum(predicted) + np.sum(target) + smooth)
        return coefficient

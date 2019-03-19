import torch
import bts.loss as loss
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np

from datetime import datetime
from time import time


class BrainTumorClassifier():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = loss.BCEDiceLoss(self.device).to(device)
        self.log_path = datetime.now().strftime("%I-%M-%S_%p_on_%B_%d,_%Y")
        self.tb_writer = SummaryWriter(log_dir=f'logs/{self.log_path}')

    def _train_epoch(self, trainloader, mini_batch):
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
        inputs = list()
        mask = list()

        for data in sample:
            inputs.append(data['image'])

        inputs = torch.stack(inputs).to(self.device)
        outputs = self.model(inputs).detach().cpu()

        for index in range(len(sample)):
            self.tb_writer.add_image(
                str(sample[index]['index']), outputs[index], epoch)
        del inputs

    def train(self, epochs, trainloader, mini_batch=None, learning_rate=0.001, plot_image=None):
        history = {'train_loss': list()}
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print('Starting Training Process')
        for epoch in range(epochs):
            start_time = time()
            # Training a single epoch
            epoch_loss = self._train_epoch(trainloader, mini_batch)
            # Collecting all epoch loss values for future prediction
            history['train_loss'].append(epoch_loss)
            # Logging to Tensorboard
            self.tb_writer.add_scalar('Train Loss', epoch_loss, epoch)

            if plot_image:
                self.model.eval()
                self._plot_image(epoch, plot_image)
                self.model.train()

            time_taken = time()-start_time
            print(f'Epoch: {epoch+1:03d},  ', end='')
            print(f'Loss:{epoch_loss:.7f},  ', end='')
            print(f'Time:{time_taken:.2f}secs')
        return history

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def restore_model(self, path):
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)

    def test(self, testloader, threshold=0.5):
        self.model.eval()
        test_data_indexes = testloader.sampler.indices[:]
        data_len = len(test_data_indexes)
        try_iteration = 100
        mean_val_score = 0

        batch_size = testloader.batch_size
        if batch_size != 1:
            raise Exception("Set batch size to 1 for testing purpose")

        testloader = iter(testloader)

        while len(test_data_indexes) != 0:
            data = testloader.next()
            index = int(data['index'])
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue
            image = data['image'].view((1, 1, 512, 512)).to(self.device)
            mask = data['mask']

            mask_pred = self.model(image).cpu()
            mask_pred = (mask_pred > threshold)
            mask_pred = mask_pred.numpy()

            mask = np.resize(mask, (1, 512, 512))
            mask_pred = np.resize(mask_pred, (1, 512, 512))

            mean_val_score += self._dice_coefficient(mask_pred, mask)

        mean_val_score = mean_val_score / data_len
        self.model.train()
        return mean_val_score

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

    def predict(self, data, threshold=0.5):
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

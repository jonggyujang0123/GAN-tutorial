"""
Mnist Data loader, as given in Mnist tutorial
"""
import imageio
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, TensorDataset, Dataset


class MnistDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        transform_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        if config.data_mode == "download":
            dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform_train)
            dataset_train, dataset_val = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(1))
            dataset_test = datasets.MNIST('./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)
            self.train_loader = torch.utils.data.DataLoader(dataset_train,
                                                            batch_size=self.config.batch_size,
                                                            shuffle=True,
                                                            num_workers=self.config.data_loader_workers,
                                                            pin_memory=self.config.pin_memory)
            self.val_loader = torch.utils.data.DataLoader(dataset_val,
                                                        batch_size=self.config.test_batch_size,
                                                        shuffle=True,
                                                        num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory)
            self.test_loader = torch.utils.data.DataLoader(dataset_test,
                                                        batch_size=self.config.test_batch_size,
                                                        shuffle=True,
                                                        num_workers=self.config.data_loader_workers,
                                                        pin_memory=self.config.pin_memory)
        elif config.data_mode == "imgs":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'result.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass


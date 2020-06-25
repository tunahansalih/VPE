import os

import torch
from sacred import Experiment
from sacred.observers import MongoObserver
from torch import nn

from models import VAEIdsia

ex = Experiment(name="Variational Prototyping Encoder")
ex.observers.append(MongoObserver())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(self):
        # SACRED: we don't need any parameters here, they're in the config and the functions get a @ex.capture handle
        # later
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.loss_fn = self.make_loss_function()
        self.train_dataset, self.validation_dataset, self.test_dataset = self.get_datasets()
        self.train_loader, self.validation_dataset, self.test_loader = self.get_dataloaders()

    @ex.capture
    def make_model(self, arch):
        if arch is 'vaeIdsiaStn':
            model = VAEIdsia(nc=3, input_size=64, latent_variable_size=300, cnn_chn=[100, 150, 250],
                             param1=[200, 300, 200], param2=None,
                             param3=[150, 150, 150]).to(device)  # idsianet cnn_chn=[100,150,250] latent = 300
            print('Use vae+Idsianet (stn1 + stn3) with random initialization!')

        if arch is 'vaeIdsia':
            model = VAEIdsia(nc=3, input_size=64, latent_variable_size=300, cnn_chn=[100, 150, 250], param1=None,
                             param2=None, param3=None).to(device)  # idsianet cnn_chn=[100,150,250] latent = 300
            print('Use vae+Idsianet (without stns) with random initialization!')
        return model

    @ex.capture
    def make_optimizer(self, learning_rate):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def make_loss_function(self):
        reconstruction_function = nn.BCELoss(reduction="sum")

        def loss_function(recon_x, x, mu, log_var):
            BCE = reconstruction_function(recon_x, x)
            KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            return BCE + KLD

        return loss_function

    def get_datasets(self):
        pass

    def get_dataloaders(self):
        pass


@ex.config
def get_config():
    """
    Where you would normally do something like:
    parser = argparse.ArgumentParser(...)
    parser.add_argument(...)
    ...
    Now you need to store all your parameters in a function called get_config().
    Put the @ex.config handle above it to ensure that Sacred knows this is the config function it needs to look at.
    """

    seed: int = 42  # Random seed
    arch: str = 'vaeIdsiaStn'  # network type: vaeIdsia, vaeIdsiaStn
    dataset: str = 'belga2flickr'  # dataset to use [gtsrb, belga2flickr, belga2toplogo])
    # for gtsrb2TT100K scenario, use main_train_test.py
    example_dir: str = 'example_list'  # training scenario
    resume: str = None  # Resume training from previously saved model
    #
    epochs: int = 1000  # Training epochs
    learning_rate: float = 1e-4  # Learning rate
    batch_size: int = 128  # Batch size
    #
    img_width: int = 64  # resized image width
    img_height: int = 64  # resized image height
    workers: int = 4  # Data loader workers
    save_epoch: int = 100  # Saving frequency
    #
    input_directory = os.path.join(f"db")
    output_log_directory = os.path.join(f"log/{dataset}_log")
    output_image_directory = os.path.join(f"log/{dataset}_image")




@ex.main
def main(_experiment):
    if not os.path.isdir(_experiment["output_image_directory"]):
        os.mkdir(_experiment["output_image_directory"])
    if not os.path.isdir(_experiment["output_log_directory"]):
        os.mkdir(_experiment["output_log_directory"])

    best_acc = 0
    best_acc_val = 0

    for e in range(1, _experiment["epochs"] + 1):
        val_trigger = False
        train(e)
        temp_acc_val = validation(e, best_acc_val)
        if temp_acc_val > best_acc_val:
            best_acc_val = temp_acc_val
            val_trigger = True  # force test function to save log when validation performance is updated
        best_acc = test(e, best_acc, val_trigger)


if __name__ == '__main__':
    ex.run_commandline()

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from argparse import Namespace
from functools import partial
import tempfile

from ray import train, tune
from ray.tune import Trainable
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from model.isabel import *
from model.generator import Generator
from model.vgg19 import VGG19


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_data(args):
    train_dataset = IsabelDataset(
        root=args.root_dir_train,
        param_file=args.param_file_train,
        train=True,
        test=False,
        transform=transforms.Compose([Normalize(), ToTensor()])
    )

    test_dataset = IsabelDataset(
        root=args.root_dir_test,
        param_file=args.param_file_test,
        train=False,
        test=True,
        transform=transforms.Compose([Normalize(), ToTensor()])
    )

    return train_dataset, test_dataset


def test_accuracy(args, g_model, test_loader, device="cpu"):
    norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(device)
    norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(device)
    vgg = VGG19(layer='relu1_2').eval()
    if args.data_parallel and torch.cuda.device_count() > 1:
        vgg = nn.DataParallel(vgg)
    vgg.to(device)
    mse_criterion = nn.MSELoss(reduction='mean')

    g_model.eval()
    batch_test_losses = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample["image"].to(device)
            vparams = sample["vparams"].to(device)
            fake_image = g_model(vparams)
            test_loss = 0.
            test_loss += mse_criterion(image, fake_image) * len(image) / len(test_loader.dataset)

            if args.use_vgg_loss:
                image1 = ((image + 1.) * .5 - norm_mean) / norm_std
                fake_image1 = ((fake_image + 1.) * .5 - norm_mean) / norm_std
                features = vgg(image1)
                fake_features = vgg(fake_image1)
                perc_loss = args.vgg_loss_weight * mse_criterion(features, fake_features)
                test_loss += perc_loss * len(image) / len(test_loader.dataset)

            batch_test_losses.append(test_loss.item())
    epoch_test_loss = sum(batch_test_losses[-len(test_loader):])

    return epoch_test_loss


def load_config():
    config = {
        "_comment": "This json file provides parameter configurations.",
        "no_cuda": False,
        "data_parallel": True,
        "seed": 1,
        "root_dir_train": "/users/mtech/vivekg/actv_img_regr/data/Isabel_pressure_volume_images/4/train/",
        "root_dir_test": "/users/mtech/vivekg/actv_img_regr/data/Isabel_pressure_volume_images/4/test/",
        "param_file_train": "isabel_pr_viewparams_train.csv",
        "param_file_test": "isabel_pr_viewparams_test.csv",
        "root_out_dir": "/users/mtech/vivekg/actv_img_regr/outputs/Isabel_mse_active_random/",
        "resume": 0,
        "chkpt": "",
        "dvp": 3,
        "dvpe": 512,
        "ch": 64,
        "sn": 0,
        "use_mse_loss": 1,
        "use_vgg_loss": 1,
        "use_gan_loss": 0,
        "gan_loss_weight": 0.01,
        "vgg_loss_weight": 1.0,
        "mse_loss_weight": 1.0,
        "lr": 1e-3,
        "d_lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "batch_size": 10,
        "start_epoch": 0,
        "epochs": 10,
        "log_every": 1,
        "check_every": 50,
        "no_active_learning": False,
        "data_gen_script": "./gen_img4.py",
        "raw_inp_file": "../data/Isabel_pressure_raw/Pf25.binLE.raw_corrected_2_subsampled.vti",
        "varname": "ImageScalars",
        "num_new_samples": 100,
        "query_strategy": "VGG",
        "sampling_budget": 8000
    }

    return Namespace(**config)


class TrainModel(Trainable):
    def setup(self, config):
        self.args = load_config()
        self.config = config

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.g_model = Generator(dvp=self.args.dvp, dvpe=self.args.dvpe, ch=self.args.ch)
        self.g_model.apply(weights_init)

        if self.args.data_parallel and torch.cuda.device_count() > 1:
            self.g_model = nn.DataParallel(self.g_model)
        self.g_model.to(self.device)

        if self.args.use_vgg_loss:
            self.norm_mean = torch.tensor([.485, .456, .406]).view(-1, 1, 1).to(self.device)
            self.norm_std = torch.tensor([.229, .224, .225]).view(-1, 1, 1).to(self.device)
            self.vgg = VGG19(layer='relu1_2').eval().to(self.device)

        self.mse_criterion = nn.MSELoss(reduction='mean')
        self.batch_train_losses, self.train_losses, self.batch_val_losses, self.val_losses = [], [], [], []

        self.g_optimizer = optim.Adam(self.g_model.parameters(), lr=config["lr"],
                                      betas=(self.args.beta1, self.args.beta2))

        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = torch.load(fp)
                self.args.start_epoch = checkpoint_state["epoch"]
                self.g_model.load_state_dict(checkpoint_state["g_model_state_dict"], strict=False)
                self.g_optimizer.load_state_dict(checkpoint_state["g_optimizer_state_dict"])
        else:
            self.args.start_epoch = 0

        train_dataset, _ = load_data(self.args)
        test_abs = int(len(train_dataset) * 0.8)
        train_subset, val_subset = random_split(train_dataset, [test_abs, len(train_dataset) - test_abs])

        kwargs = {"num_workers": 8, "pin_memory": True} if self.args.cuda else {}
        self.args.batch_size = config["batch_size"]
        self.train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        self.val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=True, **kwargs)

    def step(self):
        self.g_model.train()
        for i, sample in enumerate(self.train_loader):
            loss = 0.0
            image = sample["image"].to(self.device)
            vparams = sample["vparams"].to(self.device)
            self.g_optimizer.zero_grad()
            fake_image = self.g_model(vparams)

            if self.args.use_mse_loss:
                mse_loss = self.args.mse_loss_weight * self.mse_criterion(image, fake_image)
                loss += mse_loss * len(image) / len(self.train_loader.dataset)

            if self.args.use_vgg_loss:
                image1 = ((image + 1.) * .5 - self.norm_mean) / self.norm_std
                fake_image1 = ((fake_image + 1.) * .5 - self.norm_mean) / self.norm_std
                features = self.vgg(image1)
                fake_features = self.vgg(fake_image1)
                perc_loss = self.args.vgg_loss_weight * self.mse_criterion(features, fake_features)
                loss += perc_loss * len(image) / len(self.train_loader.dataset)

            loss.backward()
            self.g_optimizer.step()

            self.batch_train_losses.append(loss.item())

        epoch_loss = sum(self.batch_train_losses[-len(self.train_loader):])
        self.train_losses.append(epoch_loss)

        self.g_model.eval()
        for i, sample in enumerate(self.val_loader):
            with torch.no_grad():
                image = sample["image"].to(self.device)
                vparams = sample["vparams"].to(self.device)
                fake_image = self.g_model(vparams)
                val_loss = 0.
                val_loss += self.mse_criterion(image, fake_image) * len(image) / len(self.val_loader.dataset)

                if self.args.use_vgg_loss:
                    image1 = ((image + 1.) * .5 - self.norm_mean) / self.norm_std
                    fake_image1 = ((fake_image + 1.) * .5 - self.norm_mean) / self.norm_std
                    features = self.vgg(image1)
                    fake_features = self.vgg(fake_image1)
                    perc_loss = self.args.vgg_loss_weight * self.mse_criterion(features, fake_features)
                    val_loss += perc_loss * len(image) / len(self.val_loader.dataset)

                self.batch_val_losses.append(val_loss.item())
        epoch_val_loss = sum(self.batch_val_losses[-len(self.val_loader):])
        self.val_losses.append(epoch_val_loss)

        checkpoint_data = {
            "epoch": self.iteration,
            "g_model_state_dict": self.g_model.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(loss=epoch_val_loss, checkpoint=checkpoint)

    def save_checkpoint(self, checkpoint_dir):
        data_path = Path(checkpoint_dir) / "data.pkl"
        checkpoint_data = {
            "epoch": self.iteration,
            "g_model_state_dict": self.g_model.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
        }
        with open(data_path, "wb") as fp:
            torch.save(checkpoint_data, fp)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            checkpoint_data = torch.load(fp)
        self.iteration = checkpoint_data["epoch"]
        self.g_model.load_state_dict(checkpoint_data["g_model_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint_data["g_optimizer_state_dict"])


num_samples = 100
max_num_epochs = 10
gpus_per_trial = 0.25

args = load_config()
args.epochs = max_num_epochs
train_dataset, test_dataset = load_data(args)
config = {
    "lr": tune.loguniform(1e-4, 1e-3),
    "batch_size": tune.choice([16]),
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=4,
    reduction_factor=2,
)
result = tune.run(
    TrainModel,
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=9,
    scheduler=scheduler,
)

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

g_model = Generator(dvp=3, dvpe=512, ch=64)

args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if gpus_per_trial > 1:
        g_model = nn.DataParallel(g_model)
g_model.to(device)

best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
with best_checkpoint.as_directory() as checkpoint_dir:
    data_path = Path(checkpoint_dir) / "data.pkl"
    with open(data_path, "rb") as fp:
        best_checkpoint_data = pickle.load(fp)

    g_model.load_state_dict(best_checkpoint_data["g_model_state_dict"])
    test_acc = test_accuracy(args, g_model, test_loader, device)
    print("Best trial test set accuracy: {}".format(test_acc))

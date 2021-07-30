from typing import Tuple, List
from pathlib import Path
from collections import namedtuple
from argparse import ArgumentParser

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.transforms import RandomRotation
from torchmetrics import SSIM
import pytorch_lightning as pl
from inpainting_transformer_base import InpaintingTransformer, img_to_window_patches


def main(args):
    # DataModule for dataset related prcess like-
    #  scanning images from root folder, spliting them into train+val,
    #  creating train/val/test Dataset object with necessary augmentations.
    dm = MVTechDataModule(args)
    dm.prepare_data()
    dm.setup(stage='fit')

    # Inpainting Transformer model
    model = InpTrans(args)

    # Trainer object for training
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)


class InpTrans(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        assert args.img_width == args.img_height, 'Expected square image shape.'
        self.L2_loss = nn.MSELoss()
        self.ssim = SSIM()
        self.last_epoch = self.current_epoch
        self.model = InpaintingTransformer(
            img_size=args.img_width,  # assuming width and height same
            num_channels=args.img_channel,
            patch_size=args.patch_size,
            window_size=args.window_size,
            embed_dim=args.embed_dim,
            positional_mapping=args.positional_mapping,
            n_heads=args.num_heads,
            depth=args.block_depth,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=args.qkv_bias,
            p=args.dropout_probability  # no dropout used in paper
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InpTrans")
        parser.add_argument('--patch_size', type=int, default=16)
        parser.add_argument('--window_size', type=int, default=7)
        parser.add_argument('--embed_dim', type=int, default=512)
        parser.add_argument('--positional_mapping', type=str, default='local')
        parser.add_argument('--block_depth', type=int, default=13)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--qkv_bias', type=bool, default=True)
        parser.add_argument('--mlp_ratio', type=int, default=4)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--dropout_probability', type=float, default=0)
        return parent_parser  # parent_parser IMPORTANT!!!

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.args.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, Y_pred, Y, mode="train"):
        l2_loss = self.L2_loss(Y_pred, Y)
        ssim_loss = 1 - self.ssim(Y_pred, Y)
        loss = l2_loss + (0.01 * ssim_loss)
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_L2_loss', l2_loss)
        self.log(f'{mode}_SSIM_loss', ssim_loss)
        return loss

    def training_step(self, batch, batch_idx):
        patches_info, labels = batch
        X = patches_info['neighbor_patchs']
        Y = patches_info['inpaint_patch']
        X_positions = patches_info['neighbor_positions']
        Y_position = patches_info['inpaint_position']

        Y_pred = self.model(X, X_positions, Y_position)
        loss = self._calculate_loss(Y_pred, Y, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        patches_info, labels = batch
        X = patches_info['neighbor_patchs']
        Y = patches_info['inpaint_patch']
        X_positions = patches_info['neighbor_positions']
        Y_position = patches_info['inpaint_position']

        Y_pred = self.model(X, X_positions, Y_position)
        loss = self._calculate_loss(Y_pred, Y, mode='val')

        # save 10 images for logging
        if  self.current_epoch - self.last_epoch > 0:
            self.last_epoch += 1
            grid = utils.make_grid(
                torch.cat([Y[:10], Y_pred[:10]], dim=0),
                nrow=10,
                normalize=False
            )
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
        return loss


class MVTechDataModule(pl.LightningDataModule):

    def __init__(self, args, seed=1):
        super().__init__()
        pl.seed_everything(seed)  # to reproduce same result
        self.args = args

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomApply(nn.ModuleList([
                # transforms.Grayscale(),
                transforms.RandomResizedCrop(320, scale=(.1, 1.0))]),
                p=0.3),
            transforms.RandomRotation(degrees=10),

            transforms.Resize([args.img_height, args.img_width]),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize([args.img_height, args.img_width]),
            transforms.ToTensor()
        ])

    def prepare_data(self):
        # scan image files
        print(f'Scanning in {self.args.img_root}...')
        root_dir = Path(self.args.img_root)
        self.train_img_paths = list(
            (root_dir / 'train' / 'good').rglob('*.png'))
        self.test_img_paths = list((root_dir / 'test').rglob('*.png'))

        # Assign label
        # In train folder, all are good. We copy 'good' string for all.
        self.train_img_labels = ['good'] * len(self.train_img_paths)
        # But, we need detailed label for test folder.
        # In MVTech dataset, parent folder is the image class name
        self.test_img_labels = [p.parent.name for p in self.test_img_paths]
        # all unique class name found in the dataset
        self.classes = []
        self.classes.append('good')
        self.classes.extend(
            [c for c in set(self.test_img_labels) if c not in self.classes])

        print(f'\tFound {len(self.train_img_paths)} training images.')
        print(f'\tFound {len(self.test_img_paths)} test images.')

    def setup(self, stage):
        if stage == 'fit':
            # We will use train set for training and validation.
            # here validation helps to check image reconstraction quality.
            # make pair for each path with it's label
            path_label_pairs = list(
                zip(self.train_img_paths, self.train_img_labels)
            )
            # calculate train and validation count
            total_count = len(path_label_pairs)
            new_train_count = int(self.args.train_ratio*total_count)
            new_val_count = total_count - new_train_count
            print(
                f'Train split: {new_train_count}, Val split: {new_val_count}')
            # Random split into new train and validation set
            train_split_pairs, val_split_pairs = random_split(
                path_label_pairs,
                [new_train_count, new_val_count]
            )
            # random_split() returns dataset.Subset object containing indices.
            # But we do not need the indices. (path, label) pair list is enough.
            # (list object is particularly helpful in the following step)
            train_split_pairs = list(train_split_pairs)
            val_split_pairs = list(val_split_pairs)

            # In the paper, authors sampled 600 windows from each image.
            # To produce that, we will repeat each image for 600 times.
            # Inside dataloader, a random window will be selected each time,
            # Thus we will get 600 random windows from each image.
            train_split_pairs = train_split_pairs * 600  # list repeating

            # Create dataset object
            self.train_dataset = MVTechDataset(
                train_split_pairs,
                self.train_transform,
                self.args
            )
            self.val_dataset = MVTechDataset(
                val_split_pairs,
                self.test_transform,
                self.args
            )

        if stage == 'test':
            # 'test' is very different than 'fit' stage.
            # This inference method consists of two parts.
            # Firs, need to reconstruct the whole image using all patchs.
            # So, to get the complete image, instead of random window position,
            # we need to generate window position sequentially first to last.
            # After that, need to compare with original image.
            raise NotImplemented()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )


class MVTechDataset(Dataset):
    def __init__(self,
                 path_label_pairs: List[Tuple[str, str]],
                 transform,
                 args,
                 seed=1):
        self.path_label_pairs = path_label_pairs
        self.transform = transform
        self.args = args
        # pl.seed_everything(seed)  # to reproduce same result

    def __len__(self):
        return len(self.path_label_pairs)

    def __getitem__(self, idx):
        img_path, img_label = self.path_label_pairs[idx]
        img = Image.open(img_path).convert('RGB')
        X = self.transform(img)
        # sampled random window from image
        K, L = self.args.patch_size, self.args.window_size
        C, H, W = X.shape
        N = H // K
        M = W // K
        with torch.no_grad():
            # (r, s) is the window's top-left patch coordinate
            r = torch.FloatTensor(1).uniform_(
                0, N-L-1).round().type(torch.long).item()
            s = torch.FloatTensor(1).uniform_(
                0, M-L-1).round().type(torch.long).item()
            # (t, u) is the inpainted patchs's coodinate
            t = torch.FloatTensor(1).uniform_(
                0, L-1).round().type(torch.long).item()
            u = torch.FloatTensor(1).uniform_(
                0, L-1).round().type(torch.long).item()

        patches_info: dict = img_to_window_patches(X,
                                                   K=K, L=L, r=r, s=s, t=t, u=u,
                                                   flatten_patch=True,
                                                   positional_mapping='local')

        return patches_info, img_label


if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # add built-in Trainer args
    parser.add_argument('--img_root', type=str, default='./dataset/wood/')
    parser.add_argument('--img_width', type=int, default=320)
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--img_channel', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.86)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser = InpTrans.add_model_specific_args(
        parser)  # add args defined in model
    args = parser.parse_args()
    print(args)
    main(args)

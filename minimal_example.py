from typing import List
from pathlib import Path

import argparse
import collections
from collections import OrderedDict

from brain_dataset import BrainDataset
from model import MeshNet, UNet, Branched, MeshNet_2_Task
#from nest_model import NesT
import nibabel as nib
import numpy as np
import pandas as pd
from reader import NiftiFixedVolumeReader, NiftiReader, NiftiGrayMatterReader
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

from catalyst import metrics
from catalyst.callbacks import CheckpointCallback
from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose
from catalyst.dl import Runner, DeviceEngine, DataParallelEngine
from catalyst.metrics.functional._segmentation import dice

from freesurfer_stats import CorticalParcellationStats

#import matplotlib.pyplot as plt
torch.manual_seed(1)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_loaders(
    worker_init_fn,
    random_state: int,
    volume_shape: List[int],
    subvolume_shape: List[int],
    train_subvolumes: int = 128,
    infer_subvolumes: int = 512,
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    batch_size: int = 1,
    num_workers: int = 32,
) -> dict:
    """Get Dataloaders"""
    datasets = {}
    open_fn = ReaderCompose(
        [
            NiftiFixedVolumeReader(input_key="images", output_key="images"),
            NiftiReader(input_key="segmentation", output_key="segmentation"),
            NiftiGrayMatterReader(input_key="gray_matter", output_key="gray_matter")
        ]
    )

    for mode, source in zip(
        ("train", "validation", "infer"),
        (in_csv_train, in_csv_valid, in_csv_infer),
    ):
        if mode == "infer":
            n_subvolumes = infer_subvolumes
        else:
            n_subvolumes = train_subvolumes

        if source is not None and len(source) > 0:
            dataset = BrainDataset(
                list_data=dataframe_to_list(pd.read_csv(source)),
                list_shape=volume_shape,
                list_sub_shape=subvolume_shape,
                open_fn=open_fn,
                n_subvolumes=n_subvolumes,
                mode=mode,
                input_key="images",
                segmentation_key="segmentation",
                gray_matter_key="gray_matter"
            )

        datasets[mode] = {"dataset": dataset}


    train_loader = DataLoader(
        dataset=datasets["train"]["dataset"],
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=32,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=datasets["validation"]["dataset"],
        shuffle=True,
        worker_init_fn=worker_init_fn,
        batch_size=batch_size,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=datasets["infer"]["dataset"],
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=32,
        pin_memory=True,
        drop_last=True,
    )
    train_loaders = collections.OrderedDict()
    infer_loaders = collections.OrderedDict()
    train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader)
    train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader)
    infer_loaders["infer"] = BatchPrefetchLoaderWrapper(test_loader)

    return train_loaders, infer_loaders


class CustomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, n_classes: int, parallel: bool):
        """Init."""
        super().__init__()
        self.n_classes = n_classes
        self.parallel = parallel

    def get_engine(self):
        """Gets engine for multi or single gpu case"""
        if self.parallel:
            engine = DataParallelEngine()

        else:
            engine = DeviceEngine()

        return engine

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        """
        Predicts a batch for an inference dataloader and returns the
        predictions as well as the corresponding slice indices
        """
        # model inference step
        batch = batch[0]

        return (
            self.model(batch["images"].float().to(self.device)),
            batch["coords"],
        )

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            #for key in ["loss", "macro_dice", "normalizedLoss", "normalizedGray"]
            for key in ["loss", "normalizedLoss"]
        }

    def handle_batch(self, batch):
        """
        Custom train/ val step that includes batch unpacking, training, and
        DICE metrics
        """
        # model train/valid step
        batch = batch[0]
        x, y1, y2 = batch["images"].float(), batch["segmentation"], batch['gray_matter']

        y2 = y2.float()

        y_hat1, y_hat2 = self.model(x)

        y_hat2 = y_hat2.float()

        y_hat2 = torch.reshape(y_hat2, y2.shape)
        
        #NOTE: removing segmentation loss to see how model performs with just this architecture
        #loss_segmentation = F.cross_entropy(y_hat1, y1)

        loss_gray_matter = F.mse_loss(y_hat2, y2)

        zeros_y2 = torch.zeros_like(y2)

        #loss_segmentation_normalized = loss_segmentation / torch.linalg.norm(y1.float())
        loss_gray_matter_normalized = F.mse_loss(y2, zeros_y2)
        loss_gray_matter_normalized = loss_gray_matter / loss_gray_matter_normalized

        #loss_normalized = (loss_gray_matter_normalized + loss_segmentation_normalized) / 2.0
        loss_normalized = (loss_gray_matter_normalized)

        one_hot_targets = (
            torch.nn.functional.one_hot(y1, self.n_classes)
            .permute(0, 4, 1, 2, 3)
        )

        #loss = ((loss_segmentation + loss_gray_matter) / 2.0)
        loss = loss_gray_matter
        
        if self.is_train_loader:
            self.engine.backward_loss(loss, self.model, self.optimizer)
            self.engine.optimizer_step(loss, self.model, self.optimizer)
            #TODO: see if this breaks the model below
           # scheduler.step()
            self.optimizer.zero_grad()

        #TODO: make sure this is correct below
        macro_dice = dice(F.softmax(y_hat1, dim=0), one_hot_targets, mode="macro")

        self.batch_metrics.update({"loss": loss, "normalizedLoss": loss_normalized})

        for key in ["loss", "normalizedLoss"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "normalizedLoss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def voxel_majority_predict_from_subvolumes(loader, n_classes):
    """
    # TODO: change function to allow for cuda or cpu based predictions with cuda
    # as default.
    Predicts Brain Segmentations given a dataloader class and a optional dict
    to contain the outputs. Returns a dict of brain segmentations.
    """
    subject_macro_dice = []
    subject_micro_dice = []
    subject_gray_matter_mse_loss_normalized = []
    prediction_n = 0

    segmentation = torch.zeros(
        tuple(np.insert(loader.dataset.generator.volume_shape, 0, n_classes)),
        dtype=torch.uint8).cpu()

    for inference in tqdm(runner.predict_loader(loader=loader)):
        #y1 is segmentation and y2 is gray matter
        y_hat1 = inference[0][0]
        y_hat2 = inference[0][1]

        #import pdb 
        #pdb.set_trace()

        coords = inference[1]
        _, predicted = torch.max(F.log_softmax(y_hat1, dim=1), 1)

        for j in range(predicted.shape[0]):

            c_j = coords[j][0]
            subj_id = prediction_n // loader.dataset.n_subvolumes
            labels = nib.load(loader.dataset.data[subj_id]['segmentation']).get_fdata()

            gray_matter = loader.dataset.data[subj_id]['gray_matter']

            gray_matter = gray_matter.split(',')

            stats_r_path = gray_matter[0][2:-1]
            stats_l_path = gray_matter[1][2:-2]

            #using volume path for right and left hemispheres to read in data for each hemisphere
            stats_r = CorticalParcellationStats.read(stats_r_path)
            stats_l = CorticalParcellationStats.read(stats_l_path)

            #creating measurement dataframes and renaming them
            df_r = stats_r.structural_measurements[['gray_matter_volume_mm^3']]
            df_l = stats_l.structural_measurements[['gray_matter_volume_mm^3']]

            array_l = df_l.values
            array_r = df_r.values

            y2 = np.concatenate((array_l, array_r), axis=0)

            y2 = y2.astype('float32')
            y_hat2 = y_hat2.float()
            y_hat2 = torch.reshape(y_hat2, y2.shape)
            y2 = torch.tensor(y2)
            y2 = y2.to('cuda')
            loss_gray_matter = F.mse_loss(y_hat2, y2)
            zeros_y2 = torch.zeros_like(y2)
            loss_gray_matter_normalized = F.mse_loss(y2, zeros_y2)
            loss_gray_matter_normalized = loss_gray_matter / loss_gray_matter_normalized

            subject_gray_matter_mse_loss_normalized.append(loss_gray_matter_normalized)

            for c in range(n_classes):
                segmentation[
                    c,
                    c_j[0, 0] : c_j[0, 1],
                    c_j[1, 0] : c_j[1, 1],
                    c_j[2, 0] : c_j[2, 1],
                ] += (predicted[j] == c).to('cpu')
            prediction_n += 1

            if (prediction_n // loader.dataset.n_subvolumes) > subj_id:
                seg = torch.max(segmentation, 0)[1]
                seg = torch.nn.functional.one_hot(
                    seg, args.n_classes).permute(0, 3, 1, 2)
                one_hot_label = torch.nn.functional.one_hot(
                    torch.from_numpy(labels).long(),
                                     args.n_classes).permute(0, 3, 1, 2)
                subject_macro_dice.append(dice(
                    seg,
                    one_hot_label,
                    mode='macro').item())

                subject_micro_dice.append(dice(
                    seg,
                    one_hot_label).detach().numpy())

                segmentation = torch.zeros(
                    tuple(np.insert(loader.dataset.generator.volume_shape, 0, n_classes)),
                    dtype=torch.uint8).cpu()

    macro_dice_df = pd.DataFrame({'macro_dice': subject_macro_dice})
    micro_dice_df = pd.DataFrame(np.stack(subject_micro_dice))

    return macro_dice_df, micro_dice_df, subject_gray_matter_mse_loss_normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 segmentation Training")

    parser.add_argument(
        "--train_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/SegmentationReqs/MultitaskData/train.csv",
        help="Path to list with brains for training",
    )
    parser.add_argument(
        "--validation_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/SegmentationReqs/MultitaskData/validate.csv",
        help="Path to list with brains for validation",
    )
    parser.add_argument(
        "--inference_path",
        metavar="PATH",
        default="/data/users2/vjoshi6/SegmentationReqs/MultitaskData/test.csv",

        help="Path to list with brains for inference",
    )
    parser.add_argument("--n_classes", default=104+68, type=int)
    parser.add_argument("--n_filters", default=None, type=int)
    parser.add_argument(
        "--train_subvolumes",
        default=128,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        "--infer_subvolumes",
        default=512,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        "--sv_w", default=38, type=int, metavar="N", help="Width of subvolumes"
    )
    parser.add_argument(
        "--sv_h",
        default=38,
        type=int,
        metavar="N",
        help="Height of subvolumes",
    )
    parser.add_argument(
        "--sv_d", default=38, type=int, metavar="N", help="Depth of subvolumes"
    )
    parser.add_argument("--model", default="meshnet")
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        metavar="N",
        help="dropout probability for meshnet",
    )
    parser.add_argument("--large", default=False)
    parser.add_argument("--parallel", default=False)
    parser.add_argument(
        "--n_epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--lr",
        default=0.001,
        type=float
    )

    parser.add_argument(
        "--logdir",
        default='logdir'
    )

    args = parser.parse_args()
    print("{}".format(args))

    volume_shape = [256, 256, 256]
    subvolume_shape = [args.sv_h, args.sv_w, args.sv_d]
    train_loaders, infer_loaders = get_loaders(
        worker_init_fn,
        0,
        volume_shape,
        subvolume_shape,
        args.train_subvolumes,
        args.infer_subvolumes,
        args.train_path,
        args.validation_path,
        args.inference_path,
    )

    if args.model == "meshnet":
        net = MeshNet(
            n_channels=1,
            n_classes=args.n_classes,
            large=args.large,
            #n_filters=args.n_filters,
            dropout_p=args.dropout,
        )
    elif args.model == 'meshnet2':
        net = MeshNet_2_Task(
            n_channels=1,
            n_classes=args.n_classes,
            n_estimates=68,
            large=args.large,
            #n_filters=args.n_filters,
            dropout_p=args.dropout,
        )
    else:
        net = UNet(n_channels=1, n_classes=args.n_classes)

    logdir = "logs/{}epochs{}lr".format(args.n_epochs, args.lr)

    if args.large:
        logdir += "_large"

    if args.dropout:
        logdir += "_dropout"


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.02,
        epochs=args.n_epochs,
        steps_per_epoch=len(train_loaders["train"]),
    )
    
    Path(logdir).mkdir(parents=True, exist_ok=True)

    runner = CustomRunner(n_classes=args.n_classes,
                          parallel=args.parallel)

    runner.model = net
    runner.engine = runner.get_engine()

    
    runner.train(
        model=net,
        optimizer=optimizer,
        loaders=train_loaders,
        num_epochs=args.n_epochs,
        scheduler=scheduler,
        callbacks=[CheckpointCallback(logdir=logdir)],
        logdir=logdir,
        verbose=True,
    )
    
    macro_dice_df, micro_dice_df, normalized_gray_matter = voxel_majority_predict_from_subvolumes(
        infer_loaders['infer'], args.n_classes
    )

    macro_dice_df.to_csv('{logdir}/macro_dice_results.csv'.format(logdir=logdir), index=False)
    micro_dice_df.to_csv('{logdir}/micro_dice_results.csv'.format(logdir=logdir), index=False)

    print("Macro and micro dice: \n{}\n{}".format(macro_dice_df, micro_dice_df))
    print("Macro dice mean: {}".format(macro_dice_df.mean()))

    mean_gray_matter = 0
    for value in normalized_gray_matter:
        mean_gray_matter += value.cpu().numpy()

    mean_gray_matter = mean_gray_matter / len(normalized_gray_matter)

    print("Normalized gray matter loss mean: {}".format(mean_gray_matter))
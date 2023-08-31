"""
Train model for benchmark tasks.
"""
from argparse import ArgumentParser, FileType
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from src.benchmark.utils import add_device_hparams, get_lat2d, add_yml_params, seed_everything
from src.benchmark.collect_data import get_data, get_checkpoint_path
from src.benchmark.models import ConvLSTMForecaster
from src.benchmark.graphics import plot_random_outputs_multi_ts
from src.benchmark.metrics import eval_loss, define_loss_fn, collect_outputs
from deepspeed.ops import adam
from typing import Any, Dict

import json
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import LightningModule, Trainer, loggers
from arch import ClimaXRainBench
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics import (
    mse,
    lat_weighted_mse_val,
    lat_weighted_nrmse,
    lat_weighted_rmse,
)
from utils.pos_embed import interpolate_pos_embed
from torchvision.transforms import transforms


class RainForecastModule(LightningModule):
    """Lightning module for rain forcasting with the ClimaXRainBench model.

    Args:
        net (ClimaXRainBench): ClimaXRainBench model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        hparams,
        train_set,
        valid_set,
        test_set,
        normalizer,
        collate,
        pretrained_path: str = "",
        lat2d=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.categories = hparams['categories']
        self.net = ClimaXRainBench(
                        default_vars=self.categories['input'],
                        out_vars=self.categories['output'],
        )
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

        self.trainset = train_set
        self.validset = valid_set
        self.testset = test_set
        self.collate = collate
        self.normalizer = normalizer
        self.pred_range = 0
        self.val_clim = None
        self.set_test_clim()
        self.set_denormalizer()
        self.lead_times = hparams['lead_times']
        self.multi_gpu = hparams['multi_gpu']
        self.lat, self.lon = hparams['latlon']
        self.test_step_outputs = []
        self.val_step_outputs = []
        self.version = hparams["version"]
        
        if lat2d is None:
            lat2d = get_lat2d(hparams['grid'], self.validset.dataset)
        self.weights_lat, self.loss = define_loss_fn(lat2d)
        self.lat = lat2d[0][:,0]


    def load_mae_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )
        
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
            
            if 'token_embeds' in k or 'head' in k: # initialize embedding from scratch
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                continue
                
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=False)
    
    def set_test_clim(self):
        # y_avg = torch.from_numpy(self.Y_train_all).squeeze(1).mean(0) # H, W
        # w_lat = np.cos(np.deg2rad(self.lat)) # (H,)
        # w_lat = w_lat / w_lat.mean()
        # w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y_avg.dtype, device=y_avg.device) # (H, 1)
        # self.test_clim = torch.abs(torch.mean(y_avg * w_lat))
        self.test_clim = 0

    def set_denormalizer(self):
        target_v = self.categories['output'][0]
        std = self.normalizer[target_v]['std']

        self.denormalizer = lambda x: torch.exp(x - 1) * std

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch

        loss_dict, _ = self.net.forward(x, y, lead_times, self.categories['input'], self.categories['output'], [mse], lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict['loss']

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch
        _, pred = self.net.forward(
            x,
            y,
            lead_times,
            self.categories['input'],
            self.categories['output'],
            metric=None,
            lat=self.lat,
        )

        results = eval_loss(pred, y, lead_times, self.loss, self.lead_times, phase='val', target_v=self.categories['output'][0], normalizer=self.normalizer)

        self.val_step_outputs.append(results)
        return results

    
        # all_loss_dicts = self.net.evaluate(
        #     x,
        #     y,
        #     lead_times,
        #     self.categories['input'],
        #     self.categories['output'],
        #     transform=self.denormalizer,
        #     metrics=[lat_weighted_mse_val, lat_weighted_rmse],
        #     lat=self.lat,
        #     clim=self.val_clim,
        #     log_postfix=None
        # )

        # loss_dict = {}
        # for d in all_loss_dicts:
        #     for k in d.keys():
        #         loss_dict[k] = d[k]

        # for var in loss_dict.keys():
        #     self.log(
        #         "val/" + var,
        #         loss_dict[var],
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=False,
        #         sync_dist=True,
        #     )
        # return loss_dict
    

    def on_validation_epoch_end(self):
        node_loss = collect_outputs(self.val_step_outputs, False)
        self.val_step_outputs.clear()  # free memory

        if isinstance(node_loss, list):
            node_loss = node_loss[0]
    
        all_losses = self.all_gather(node_loss)
        mean_losses = {k: float(torch.mean(x)) for k, x in all_losses.items()}

        # log mean losses
        for var in mean_losses.keys():
            self.log(
                "val/" + var,
                mean_losses[var],
                sync_dist=True
            )


    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times = batch
        _, pred = self.net.forward(
            x,
            y,
            lead_times,
            self.categories['input'],
            self.categories['output'],
            metric=None,
            lat=self.lat,
        )
        results = eval_loss(pred, y, lead_times, self.loss, self.lead_times, phase='test', target_v=self.categories['output'][0], normalizer=self.normalizer)

        self.test_step_outputs.append(results)
        return results

        # all_loss_dicts = self.net.evaluate(
        #     x,
        #     y,
        #     lead_times,
        #     self.categories['input'],
        #     self.categories['output'],
        #     transform=self.denormalizer,
        #     metrics=[self.loss],
        #     lat=self.lat,
        #     clim=self.test_clim,
        #     log_postfix=None
        # )

        # loss_dict = {}
        # for d in all_loss_dicts:
        #     for k in d.keys():
        #         loss_dict[k] = d[k]

        # return loss_dict

    def on_test_epoch_end(self) -> None:
        node_loss = collect_outputs(self.test_step_outputs, False)
        self.test_step_outputs.clear()  # free memory

        if isinstance(node_loss, list):
            node_loss = node_loss[0]
    
        all_losses = self.all_gather(node_loss)
        mean_losses = {k: float(torch.mean(x)) for k, x in all_losses.items()}

        # log mean losses
        for var in mean_losses.keys():
            self.log(
                "test/" + var,
                mean_losses[var],
                sync_dist=True
            )
        
        # Save evaluation results
        results_path = Path(f'./results/{self.version}_results.json')
        
        with open(results_path, 'w') as fp:
            json.dump(mean_losses, fp, indent=4)

        fp.close()
        




    
    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        # optimizer = torch.optim.AdamW(
        #     [
        #         {
        #             "params": decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": self.hparams.weight_decay,
        #         },
        #         {
        #             "params": no_decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": 0
        #         },
        #     ]
        # )

        optimizer = adam.FusedAdam(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0
                },
            ]
        )

        # optimizer = adam.DeepSpeedCPUAdam(
        #     [
        #         {
        #             "params": decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": self.hparams.weight_decay,
        #         },
        #         {
        #             "params": no_decay,
        #             "lr": self.hparams.lr,
        #             "betas": (self.hparams.beta_1, self.hparams.beta_2),
        #             "weight_decay": 0
        #         },
        #     ]
        # )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
class RegressionModel(LightningModule):
    """
    Regression Module
    """
    def __init__(self, hparams, train_set, valid_set, normalizer, collate, lat2d=None):
        super().__init__()
        hparams['relu'] = not hparams['no_relu']
        self.hparams = hparams
        self.lead_times = hparams['lead_times']
        self.normalizer = normalizer
        self.categories = hparams['categories']
        self.trainset = train_set
        self.validset = valid_set
        self.normalizer = normalizer
        self.collate = collate
        self.multi_gpu = hparams['multi_gpu']
        self.target_v = self.categories['output'][0]
        
        self.net = ConvLSTMForecaster(
                        in_channels=hparams['num_channels'],
                        output_shape=(hparams['out_channels'], *hparams['latlon']),
                        channels=(hparams['hidden_1'], hparams['hidden_2']),
                        last_ts=True,
                        last_relu=hparams['relu'])
        
        self.plot = self.hparams['plot']
        if self.plot:
            # define dictionary to hold column names in input and output: {var_name: (input_col_index, output_col_index)}
            self.idxs = {}
            for ind_y, v in enumerate(self.categories['output']):
                self.idxs[v] = (self.categories['input'].index(v), ind_y) if v in self.categories['input'] else (None, ind_y)
            for ind_x, v in enumerate(self.categories['input']):
                if v not in self.categories['output']:
                    self.idxs[v] = (ind_x, None)
        
        if lat2d is None:
            lat2d = get_lat2d(hparams['grid'], self.validset.dataset)
        self.weights_lat, self.loss = define_loss_fn(lat2d)
        self.lat2d = lat2d
    
    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_nb):
        inputs, output, lts = batch
        pred = self(inputs.contiguous())
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='train', target_v=self.target_v, normalizer=self.normalizer)
        return {'loss': results['train_loss'], 'log': results, 'progress_bar': results}

    def validation_step(self, batch, batch_idx):
        inputs, output, lts = batch
        pred = self(inputs)
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='val', target_v=self.target_v, normalizer=self.normalizer)
        return results

    def test_step(self, batch, batch_idx):
        inputs, output, lts = batch
        pred = self(inputs)
        results = eval_loss(pred, output, lts, self.loss, self.lead_times, phase='test', target_v=self.target_v, normalizer=self.normalizer)
        return results

    def plot_outputs_on_tensorboard(self):
        samples = []
        for lt in self.hparams['lead_times']:
            sample_lt = self.validset.get_sample_at(f'lead_time_{lt}', datetime(2018, 7, 12, 0).timestamp())
            sample_lt['__sample_modes__'] = f'lead_time_{lt}'
            samples.append([sample_lt])
        sample = self.collate(samples)
        sample_X, sample_y, _ = sample
        pred_y = self(sample_X.cuda()).cpu()
        grid = plot_random_outputs_multi_ts(sample_X, sample_y, pred_y, self.idxs, self.normalizer, self.categories['output'])
        self.logger.experiment.add_image('generated_images', grid, self.global_step)
        
    def validation_epoch_end(self, outputs):
        log_dict = collect_outputs(outputs, self.multi_gpu)
        results = {'log': log_dict, 'progress_bar': {'val_loss': log_dict['val_loss']}}
        results = {**results, **log_dict}

        if self.plot:
            self.plot_outputs_on_tensorboard()
        return results

    def test_epoch_end(self, outputs):
        log_dict = collect_outputs(outputs, self.multi_gpu)
        results = {'log': log_dict, 'progress_bar': {'test_loss': log_dict['test_loss']}}
        results = {**results, **log_dict}
        return results
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams['lr'])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'], collate_fn=self.collate, shuffle=False)

    @staticmethod
    def load_model(log_dir, **params):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param params: list of named arguments, used to update the model hyperparameters
        """
        # load hparams
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            hparams = yaml.load(fp, Loader=yaml.Loader)
            hparams.update(params)

        # load data
        hparams, loaderDict, normalizer, collate = get_data(hparams, tvt='train_valid_test')
        
        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        print(f'Loading model {model_path.parent.stem}')
        train_set = loaderDict['train']
        valid_set = loaderDict['valid']
        model = RegressionModel.load_from_checkpoint(str(model_path), hparams=hparams, \
            train_set=train_set, valid_set=valid_set, normalizer=normalizer, collate=collate)
        return model, hparams, loaderDict, normalizer, collate


def main(hparams):
    hparams = vars(hparams)
    hparams, loaderDict, normalizer, collate = get_data(hparams)
    
    # ------------------------
    # Model
    # ------------------------
    add_device_hparams(hparams)

    # define logger
    Path(hparams['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(hparams['log_path'], version=hparams['version'])
    logger.log_hyperparams(params=hparams)

    # use rainforecast module
    model = RainForecastModule(hparams, loaderDict['train'], loaderDict['valid'], loaderDict['test'], normalizer, collate, pretrained_path=hparams['load'])


    trainer = Trainer(
        accelerator='gpu',
        devices=hparams['gpus'],
        # devices=[0,2],
        logger=logger,
        max_epochs=hparams['max_epochs'],
        precision=16 if hparams['use_amp'] else 32,
        default_root_dir=hparams['log_path'],
        deterministic=True,
        strategy=hparams['strategy'],
        callbacks=[EarlyStopping('val/val_loss', patience=3)],
        accumulate_grad_batches=2,
    )
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model)

    # Evaluate the model
    trainer.test(model.cuda())

    # res = collect_outputs(model.test_step_outputs, False)
    # model.test_step_outputs.clear()  # free memory
    # print(res, type(res))

    # if isinstance(res, list):
    #     res = res[0]

    # Save evaluation results
    # results_path = Path(f'./results/{hparams["version"]}_results.json')
    
    # with open(results_path, 'w') as fp:
    #     json.dump(res, fp, indent=4)

    # fp.close()
    


def main_baselines(hparams):
    """
    execute calculation for persistence / climatology baselines
    """
    assert hparams.phase is not None
    from src.benchmark.baseline_data import get_persistence_data, get_climatology_data
    phase = hparams.phase
    hparams = vars(hparams)
    add_device_hparams(hparams)

    if hparams['persistence']:
        loaderDict, dataloader, target_v, lead_times = get_persistence_data(hparams)
    else:
        same_pred, loaderDict, dataloader, target_v, lead_times = get_climatology_data(hparams)

    # define loss
    lat2d = get_lat2d(hparams['grid'], loaderDict[phase].dataset)
    _, loss = define_loss_fn(lat2d)
    
    # collect data and iterate through
    outputs = []
    if hparams['persistence']:
        for inputs, output, lts in tqdm(dataloader):
            results = eval_loss(inputs, output, lts, loss, lead_times, phase="test", target_v=target_v)
            outputs.append(results)
    else:
        for inputs, output, lts in tqdm(dataloader):
            if len(inputs) < hparams['batch_size']:
                same_pred = same_pred[:len(inputs)]
            results = eval_loss(same_pred, output, lts, loss, lead_times)
            outputs.append(results)
    
    # collect results
    log_dict = collect_outputs(outputs, False)
        
    # log_dict = {v: float(log_dict[v].detach().cpu()) for v in log_dict}
    print(log_dict)

    # Save evaluation results
    results_path = Path(f'./results/{hparams["version"]}_results.json')
    with open(results_path, 'w') as fp:
        json.dump(log_dict, fp, indent=4)

    return log_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    # Data
    parser.add_argument("--persistence", action='store_true', help='Compute persistence baseline')
    parser.add_argument("--climatology", action='store_true', help='Compute climatology baseline')
    parser.add_argument("--sources", type=str, choices=['simsat_era', 'era16_3', 'simsat', 'era'], help='Input sources')
    parser.add_argument("--imerg", action='store_true', help='Predict precipitation from IMERG')
    parser.add_argument("--grid", type=float, default=5.625, choices=[5.625, 1.4], help='Data resolution')
    parser.add_argument("--sample_time_window", type=int, default=12, help="Duration of sample time window, in hours")
    parser.add_argument("--sample_freq", type=int, default=3, help="Data frequency within the sample time window, in hours")
    parser.add_argument("--forecast_time_window", type=int, default=120, help="Maximum lead time, in hours")
    parser.add_argument("--forecast_freq", type=int, default=24, help="Forecast frequency")
    parser.add_argument("--inc_time", action='store_true', help='Including hour/day/month in input')
    # 
    parser.add_argument('--config_file', default='./config.yml', type=FileType(mode='r'), help='Config file path')
    parser.add_argument('--data_paths', nargs='+', help='Paths for dill files')
    parser.add_argument('--norm_path', type=str, help='Path of json file storing  normalisation statistics')
    parser.add_argument('--log_path', type=str, help='Path of folder to log training and store model')

    # Model
    parser.add_argument("--hidden_1", type=int, default=384, help="No. of hidden units (lstm).")
    parser.add_argument("--hidden_2", type=int, default=32, help="No. of hidden units (fc).")
    parser.add_argument("--no_relu", action='store_true', help='Not using relu on last network layer')
    # Training
    parser.add_argument("--gpus", type=int, default=-1, help="Number of available GPUs")
    parser.add_argument('--use_amp', action='store_true', help='If true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="No. of epochs to train")
    parser.add_argument("--num_workers", type=int, default=8, help="No. of dataloader workers")
    parser.add_argument("--test", action='store_true', help='Evaluate trained model')
    parser.add_argument("--load", type=str, help='Path of checkpoint directory to load')
    parser.add_argument("--phase", type=str, default='test', choices=['test', 'valid'], help='Which dataset to test on.')
    parser.add_argument("--auto_lr", action='store_true', help='Auto select learning rate.')
    parser.add_argument("--auto_bsz", action='store_true', help='Auto select batch size.')
    parser.add_argument("--strategy", type=str, default='deepspeed_stage_2', help='Memory saving strategy.')
    # Monitoring
    parser.add_argument("--version", type=str, help='Version tag for tensorboard')
    parser.add_argument("--plot", action='store_true', help='Plot outputs on tensorboard')

    hparams = parser.parse_args()

    if hparams.config_file:
        add_yml_params(hparams)

    seed_everything(hparams.seed)
    
    if hparams.persistence or hparams.climatology:
        main_baselines(hparams)
    else:
        main(hparams)


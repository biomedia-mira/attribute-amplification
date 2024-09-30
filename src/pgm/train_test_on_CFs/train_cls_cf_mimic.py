import sys
import argparse
import copy
import torchvision
import torch
import os
sys.path.append('../..')
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
sys.path.append('..')
from flow_pgm import FlowPGM
from layers import TraceStorage_ELBO
from train_setup import setup_directories, setup_logging
from utils import EMA
from train_pgm import preprocess
import pandas as pd
from torch.utils.data import DataLoader
from mimic import MimicDataset_with_cfs, MimicDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_pgm import setup_dataloaders
import torch.nn as nn
from utils_pgm import update_stats, calculate_loss
import pyro

def norm(batch):
    for k, v in batch.items():
        if k in ['x', 'cf_x']:
            batch[k] = (batch[k].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] *2 -1 #[-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            try:
                batch[k] = batch[k].float().unsqueeze(-1)
            except:
                batch[k] = batch[k]
    return batch

def loginfo(title, logger, stats):
    logger.info(f'{title} | ' +
                ' - '.join(f'{k}: {v:.4f}' for k, v in stats.items()))

def inv_preprocess(pa):
    # Undo [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        if k =='age':
            pa[k] = (v + 1) / 2 * 100
    return pa


def vae_preprocess(args, pa):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(
        1, 1, *(args.input_res,)*2).float()
    return pa


def get_metrics(preds, targets):
    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze().cpu()
        # print(f'{k} | preds: {preds[k].shape} - targets: {targets[k].shape}')
    stats = {}
    for k in preds.keys():
        if k=="age":
            preds_k = (preds[k] + 1) / 2 *100  # [-1,1] -> [0,100]
            stats[k+'_mae'] = torch.mean(
                torch.abs(targets[k] - preds_k)).item() 
    return stats

class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

def sup_epoch(model, ema, dataloader, elbo_fn=None, optimizer=None, setup=None, is_train=True, use_data="real_cf", loss_norm="l2"):
    stats = {'loss': 0, 'n': 0}
    model.train(is_train)
    for _batch, _cf_batch, _ in tqdm(dataloader):
        batch = {k:None for k in _batch.keys()}
        # Merge batch and cf batch
        for k in _batch.keys():
            if use_data=="real_cf":
                try:
                    batch[k] = torch.cat((_batch[k], _cf_batch[k]), 0)
                except:
                    batch[k] = _batch[k] + _cf_batch[k]
            elif use_data=="real":
                batch[k] = _batch[k]
            elif use_data=="cf":
                batch[k] = _cf_batch[k]
        bs = len(batch['x'])

        # Calculate loss
        batch = preprocess(batch, device=_DEVICE)  

        with torch.set_grad_enabled(is_train):
            if setup == 'sup_aux':
                loss = elbo_fn.differentiable_loss(model.model_anticausal,
                               model.guide_pass, **batch) / bs
            elif setup == "sup_determ": # Train classifiers in deterministic way
                pred_batch = model.predict_unnorm(**batch)
                loss = calculate_loss(pred_batch=pred_batch, 
                                      target_batch=batch, 
                                      loss_norm=loss_norm,
                                      )
            else:
                NotImplementedError
        # Optimize model
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
        # Update stats
        stats['loss'] += loss.item() * bs
        stats['n'] += bs
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}

@torch.no_grad()
def eval_epoch(model, dataloader, use_data="real_cf"):
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}
    stats = {'loss': 0, 'n': 0}

    for _batch, _cf_batch, _ in tqdm(dataloader):
        batch = {k:None for k in _batch.keys()}
        # Merge batch and cf batch
        for k in _batch.keys():
            if use_data=="real_cf":
                try:
                    batch[k] = torch.cat((_batch[k], _cf_batch[k]), 0)
                except:
                    batch[k] = _batch[k] + _cf_batch[k]
            elif use_data=="real":
                batch[k] = _batch[k]
            elif use_data=="cf":
                batch[k] = _cf_batch[k]
            else:
                logger.info('No use_data matches')
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))
        # predict
        batch = preprocess(batch)
        out = model.predict(**batch)

        for k, v in out.items():
            preds[k].extend(v)

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k])

    stats = {}
    for i, k in enumerate(model.variables.keys()):
        if k in ['sex','finding']:
            assert targets[k].squeeze(-1).size()==preds[k].size(), f"{k} size doesn't match, targets {targets[k].squeeze(-1).size()} preds {preds[k].size()}"
            stats[k+'_acc'] = (targets[k].squeeze(-1) == torch.round(preds[k])).sum().item() / targets[k].shape[0]
            stats[k+'_rocauc'] = roc_auc_score(
                targets[k].squeeze(-1).numpy(), preds[k].numpy(), average='macro')
        elif k == 'age':
            assert targets[k].size()==preds[k].unsqueeze(1).size(), f"{k} size doesn't match, targets {targets[k].size()} preds {preds[k].size()}"
            stats[k] = torch.mean(torch.abs(targets[k] - preds[k].unsqueeze(1))).item() * 50
        elif k == 'race':
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            stats[k + "_rocauc"] = roc_auc_score(
                targets[k].numpy(),
                preds[k].numpy(),
                multi_class="ovr",
                average="macro",)
    return stats

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',
                        help='Experiment name.', type=str, default='')
    parser.add_argument('--data_dir',
                        help='Data directory to load form.', type=str, default='')
    parser.add_argument('--csv_dir',
                        help='CSV directory to load form.', type=str, default='')  
     # Specify train and eval data
    parser.add_argument('--use_dataset',
                        help="Which dataset to use.", type=str, default="mimic_cfs")
    parser.add_argument('--use_data',
                        help='Use cf or real or both.', type=str, default='')  
    parser.add_argument('--eval_data',
                        help='Use cf or real or both.', type=str, default='')   
    parser.add_argument('--which_cf',
                        help='Which cf to use', type=str, default='')        
    parser.add_argument('--dscm_dir',
                        help='Which dscm dir to use', type=str, default='') 
    parser.add_argument('--which_checkpoint',
                        help='Which checkpoint to use', type=str, default='')   
     # End            
    parser.add_argument('--load_path',
                        help='Path to load checkpoint.', type=str, default='')
    parser.add_argument('--setup',  # sup_pgm/sup_aux/sup_determ
                        help='training setup.', type=str, default='')
    parser.add_argument('--seed',
                        help='Set random seed.', type=int, default=7)
    parser.add_argument('--deterministic',
                        help='Toggle cudNN determinism.', action='store_true', default=False)
    parser.add_argument('--testing',
                        help='Test model.', action='store_true', default=False)
    parser.add_argument('--enc_net',
                        help='encoder network architecture.', type=str, default='cnn')
    parser.add_argument('--loss_norm',
                        help='Loss norm for age.', type=str, default='l1')
    # training
    parser.add_argument('--epochs',
                        help='Number of training epochs.', type=int, default=5000)
    parser.add_argument('--bs',
                        help='Batch size.', type=int, default=32)
    parser.add_argument('--lr',
                        help='Learning rate.', type=float, default=1e-4)
    parser.add_argument('--lr_warmup_steps',
                        help='lr warmup steps.', type=int, default=1)
    parser.add_argument('--wd',
                        help='Weight decay penalty.', type=float, default=0.1)
    parser.add_argument('--input_res',
                        help='Input image crop resolution.', type=int, default=224)
    parser.add_argument('--input_channels',
                        help='Input image num channels.', type=int, default=1)
    parser.add_argument('--pad',
                        help='Input padding.', type=int, default=9)
    parser.add_argument('--hflip',
                        help='Horizontal flip prob.', type=float, default=0.5)
    parser.add_argument('--eval_freq',
                        help='Num epochs per eval.', type=int, default=1)
    # model
    parser.add_argument('--widths',
                        help='Cond flow fc network width per layer.', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--parents_x',
                        help='Parents of x to load.', nargs='+', default=[])
    parser.add_argument('--alpha',
                        help='aux loss multiplier.', type=float, default=1e-3)
    parser.add_argument('--std_fixed',
                        help='Fix aux dist std value (0 is off).', type=float, default=0)
       
    args = parser.parse_known_args()[0]
    dataloaders = setup_dataloaders(args)

    # Set up model
    _DEVICE = "cuda:0"
    # Init model
    pyro.clear_param_store()
    model = FlowPGM(args)
    ema = EMA(model, beta=0.999)
    model.to(_DEVICE)
    ema.to(_DEVICE)

    # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Set up optimizer and scheduler
    scheduler=ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=10,
                mode='min',
                min_lr=1e-5,
                )
    
    # Train model
    args.save_dir = setup_directories(args, ckpt_dir=f'checkpoints/{args.dscm_dir}/{args.which_checkpoint}')
    args.best_loss = float('inf')
    logger = setup_logging(args)
    # print(f"args.use_data {args.use_data}")
    for k in sorted(vars(args)):
        logger.info(f'--{k}={vars(args)[k]}')

    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}:')
        # supervised training of PGM or aux models
        if args.setup == 'sup_aux' or args.setup=="sup_determ":
            stats = sup_epoch(
                model=model,
                ema=ema,
                dataloader=dataloaders['train'],
                elbo_fn=elbo_fn,
                optimizer=optimizer,
                setup=args.setup, 
                is_train=True, 
                use_data=args.use_data, 
                loss_norm=args.loss_norm,
            )
            if epoch % args.eval_freq == 0:
                valid_stats = sup_epoch(
                    model=ema.ema_model, 
                    ema=None,
                    dataloader=dataloaders['valid'],
                    elbo_fn=elbo_fn,
                    optimizer=None,
                    setup=args.setup,
                    is_train=False,
                    loss_norm=args.loss_norm,
                    use_data=args.eval_data,
                )
                steps = (epoch + 1) * len(dataloaders['train'])
                logger.info(
                            f'loss | train: {stats["loss"]:.4f}' +
                            f' - valid: {valid_stats["loss"]:.4f} - steps: {steps}')
            
        else:
            NotImplementedError

        if epoch % args.eval_freq == 0:
            metrics = eval_epoch(
                ema.ema_model, 
                dataloaders['valid'],
                use_data=args.eval_data,
                )
            logger.info(
                'valid | '+' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))

        if valid_stats['loss'] < args.best_loss:
            args.best_loss = valid_stats['loss']
            ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')
            torch.save({'epoch': epoch + 1,
                        'step': steps,
                        'best_loss': args.best_loss,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema.ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hparams': vars(args)}, ckpt_path)
            logger.info(f'Model saved: {ckpt_path}')

        # Save current checkpoint
        ckpt_path = os.path.join(args.save_dir, 'checkpoint_current.pt')
        torch.save({'epoch': epoch + 1,
                        'step': steps,
                        'best_loss': args.best_loss,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema.ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hparams': vars(args)}, ckpt_path)
        logger.info(f'Model saved: {ckpt_path}')

import os
import argparse
import torch

from vae import HVAE
from train_setup import *
from utils import seed_all, EMA
from trainer_rsna import trainer


def main(args):
    seed_all(args.seed, args.deterministic)
    
    # update hyperparams if resuming from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'\nLoading checkpoint: {args.resume}')
            ckpt = torch.load(args.resume, map_location='cuda:0')
            ckpt_args = {
                k: v for k, v in ckpt['hparams'].items() if k != 'resume'}
            if args.data_dir is not None:
                ckpt_args['data_dir'] = args.data_dir
            vars(args).update(ckpt_args)
        else:
            print(f'Checkpoint not found at: {args.resume}')

    # load data
    dataloaders = setup_dataloaders(args)
    
    # init model
    model = HVAE(args)
    ema = EMA(model, beta=args.ema_rate)

    # setup model save directory, logging and tensorboard summaries
    assert args.exp_name != '', 'No experiment name given.'
    args.save_dir = setup_directories(args)
    # writer = setup_tensorboard(args, model)
    writer = None
    logger = setup_logging(args)
    
    # setup optimizer
    optimizer, scheduler = setup_optimizer(args, model)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.start_epoch = ckpt['epoch']
            args.resume_step = ckpt['step']
            args.best_loss = ckpt['best_loss']
            model.load_state_dict(ckpt['model_state_dict'])
            ema.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k]=v.cuda()
        else:
            print('Checkpoint not found at: {}'.format(args.resume))
    else:
        args.start_epoch, args.resume_step = 0, 0
        args.best_loss = float('inf')

    # train
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        model.to(device)
        ema.to(device)

        trainer(args, model, ema, dataloaders,
                optimizer, scheduler, writer, logger)
    except:
        import traceback
        print(traceback.format_exc())
        # if input('Training interrupted, keep logs? [Y/n]: ') == 'n':
        #     if input(f'Deleting \'{args.save_dir}\', are you sure? [y/N]: ') == 'y':
        #         import shutil
        #         shutil.rmtree(args.save_dir)
        #         if not os.path.isdir(args.save_dir):
        #             print('Done.')


if __name__ == '__main__':
    from hps import add_arguments, setup_hparams
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    args.free_bits = 0.0
    main(args)

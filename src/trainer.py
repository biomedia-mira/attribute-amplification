import os
import copy
import imageio
import numpy as np
import torch
import torch.nn as nn
import gc
from tqdm import tqdm
from utils import *
from matplotlib import colors
import matplotlib.pyplot as plt
import copy


def trainer(args, model, ema, dataloaders, optimizer, scheduler, writer, logger):

    for k in sorted(vars(args)):
        logger.info(f'--{k}={vars(args)[k]}')
    logger.info(f'encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}')
    logger.info(f'decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}')
    logger.info(f'total params: {sum(p.numel() for p in model.parameters()):,}')

    def run_epoch(dataloader, training=True):
        model.train(training)
        model.zero_grad(set_to_none=True)
        stats = {k: 0 for k in ['elbo', 'nll', 'kl', 'n']}
        updates_skipped = 0

        mininterval = 300 if 'SLURM_JOB_ID' in os.environ else 0.1
        loader = tqdm(enumerate(dataloader), total=len(
            dataloader), mininterval=mininterval)

        for i, batch in loader:
            # preprocessing
            batch['x'] = batch['x'].cuda().float()  # [-1, 1], already preprocessed in Dataloader
            batch['pa'] = batch['pa'][..., None, None].repeat(
                1, 1, args.input_res, args.input_res).cuda().float()
            # print(f"In trainer x: {batch['x']}, pa: {batch['pa']}")
            if training:
                args.iter = i + (args.epoch-1) * len(dataloader)
                if args.beta_warmup_steps > 0:
                    args.beta = args.beta_target * linear_warmup(args.beta_warmup_steps)(args.iter + 1)
                # writer.add_scalar('train/beta_kl', args.beta, args.iter)

                out = model(batch['x'], batch['pa'], beta=args.beta)
                out['elbo'] = out['elbo'] / args.accu_steps
                out['elbo'].backward()

                if i % args.accu_steps == 0:  # gradient accumulation update
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    # writer.add_scalar('train/grad_norm', grad_norm, args.iter)
                    nll_nan = torch.isnan(out['nll']).sum()
                    kl_nan = torch.isnan(out['kl']).sum()

                    if grad_norm < args.grad_skip and nll_nan == 0 and kl_nan == 0:
                        optimizer.step()
                        scheduler.step()
                        ema.update()
                    else:
                        updates_skipped += 1
                        logger.info(
                            f'Updates skipped: {updates_skipped}'
                            + f' - grad_norm: {grad_norm:.3f}'
                            + f' - nll_nan: {nll_nan.item()} - kl_nan: {kl_nan.item()}'
                        )
                    model.zero_grad(set_to_none=True)

                    if args.iter % args.viz_freq == 0 or (args.iter in early_evals):
                        with torch.no_grad():
                            # logger.info('viz batch_sizes | ' +' - '.join(f'{k}: {v.size()}' for k,v in viz_batch.items()) )
                            write_images(args, ema.ema_model, viz_batch)
            else:
                with torch.no_grad():
                    out = ema.ema_model(batch['x'], batch['pa'], beta=args.beta)
            torch.cuda.empty_cache()
                    
            # update stats
            bs = batch['x'].shape[0]
            stats['n'] += bs  # samples seen counter
            stats['elbo'] += out['elbo'] * args.accu_steps * bs
            stats['nll'] += out['nll'] * bs
            stats['kl'] += out['kl'] * bs

            split = 'train' if training else 'valid'
            loader.set_description(
                f' => {split} | nelbo: {stats["elbo"] / stats["n"]:.3f}' 
                + f' - nll: {stats["nll"] / stats["n"]:.3f}'
                + f' - kl: {stats["kl"] / stats["n"]:.3f}'
                + f' - lr: {scheduler.get_last_lr()[0]:.6g}'
                + (f' - grad norm: {grad_norm:.2f}' if training else ""),
                refresh=False
            )
        return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}
    
    if args.beta_warmup_steps > 0:
        args.beta_target = copy.deepcopy(args.beta)

    viz_batch = next(iter(dataloaders['valid']))
    n = args.bs
    viz_batch['x'] = viz_batch['x'][:n].cuda().float()  # [-1,1]    
    viz_batch['pa'] = viz_batch['pa'][:n, :, None, None].repeat(
        1, 1, args.input_res, args.input_res).cuda().float()

    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])

    # Start training loop
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f'Epoch {args.epoch}:')
        
        stats = run_epoch(dataloaders['train'], training=True)

        # writer.add_scalar(f'nelbo/train', stats['elbo'], args.epoch)
        # writer.add_scalar(f'nll/train', stats['nll'], args.epoch)
        # writer.add_scalar(f'kl/train', stats['kl'], args.epoch)
        logger.info(
            f'=> train | nelbo: {stats["elbo"]:.4f}'
            + f' - nll: {stats["nll"]:.4f} - kl: {stats["kl"]:.4f}'
            + f' - steps: {args.iter}' 
        )

        if (args.epoch - 1) % args.eval_freq == 0:
            valid_stats = run_epoch(dataloaders['valid'], training=False)
            logger.info(
                f'=> valid | nelbo: {valid_stats["elbo"]:.4f}'
                + f' - nll: {valid_stats["nll"]:.4f} - kl: {valid_stats["kl"]:.4f}'
                + f' - steps: {args.iter}' 
            )

            if valid_stats['elbo'] < args.best_loss:
                args.best_loss = valid_stats['elbo']
                save_dict = {
                    'epoch': args.epoch,
                    'step': args.epoch * len(dataloaders['train']),
                    'best_loss': args.best_loss,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema.ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'hparams': vars(args)
                }
                ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')
                torch.save(save_dict, ckpt_path)
                logger.info(f'Model saved: {ckpt_path}')
        gc.collect() # Free memory; not sure if it will work
    return

def vae_preprocess(args, pa):
    pa = torch.cat([pa[k] for k in args.parents_x], dim=1)
    pa = pa[..., None, None].repeat(
        1, 1, *(args.input_res,)*2).cuda().float()
    return pa

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
        
def preprocess(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = batch['x'].float().cuda()  # [0,1]
        elif k in ['age']:
            batch[k] = batch[k].float().cuda()
        elif k in ['race']:
            batch[k] =batch[k].float().cuda()
#             print("batch[race]: ", batch[k])
        elif k in ['finding']:
            batch[k] = batch[k].float().cuda()
            # batch[k] = F.one_hot(batch[k], num_classes=2).squeeze().float().cuda()
        else:
            try:
                batch[k] = batch[k].float().cuda()
            except:
                batch[k] = batch[k]
    return batch

def plot(x, fig=None, ax=None, nrows=1, cmap='Greys_r', norm=None, cbar=False):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n*4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j 
            ax = [ax] if n == 1 else ax
            _x = x[i*n+j].squeeze()            
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
                # norm = colors.TwoSlopeNorm(vmin=_x.min(), vcenter=0., vmax=_x.max())
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])
    
    plt.tight_layout()

    if cbar:
        if fig:
            # fig.subplots_adjust(wspace=-0.525, hspace=0.3)
            fig.subplots_adjust(wspace=-0.3, hspace=0.3)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                cbar_ax = fig.add_axes([
                    ax[idx].get_position().x0, 
                    ax[idx].get_position().y0 - 0.02, 
                    ax[idx].get_position().width, 
                    0.01
                ])
                cbar = plt.colorbar(im[i*n+j], cax=cbar_ax, orientation="horizontal")#, ticks=mticker.MultipleLocator(25)) #, ticks=mticker.AutoLocator())
                _x = x[i*n+j].squeeze()

                d = 20
                _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                _vmin = -(_vmin - (_vmin % d))
                _vmax = _vmax - (_vmax % d)
                
                lt = [_vmin, 0, _vmax]

                if (np.abs(_vmin) - 0) > d: 
                    lt.insert(1, _vmin // 2)
                if (_vmax - 0) > d: 
                    lt.insert(-2, _vmax // 2)

                cbar.set_ticks(lt)
                cbar.outline.set_visible(False)
    return fig, ax

@torch.no_grad()
def plot_cf_rec(args, x, cf_x, pa, cf_pa, do, rec_loc):

    def undo_norm(pa):
        # reverse [-1,1] parent preprocessing back to original range
        for k, v in pa.items():
            if k =="age":
                pa[k] = (v + 1) / 2 *100 # [-1,1] -> [0,100]
        return pa

    do = undo_norm(do)
    pa = undo_norm(pa)
    cf_pa = undo_norm(cf_pa)

    fs = 15
    m, s = 6, 3
    n = 8
    fig, ax = plt.subplots(m, n, figsize=(n*s-2, m*s))
    x = (x[:n].detach().cpu() + 1) * 127.5
    _, _ = plot(x, ax=ax[0])

    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    rec_loc = (rec_loc[:n].detach().cpu() + 1) * 127.5
    _, _ = plot(rec_loc, ax=ax[1])
    _, _ = plot(cf_x, ax=ax[2])
    _, _ = plot(rec_loc - x, ax=ax[3], fig=fig, cmap='RdBu_r', cbar=True, norm=MidpointNormalize(midpoint=0))
    _, _ = plot(cf_x - x, ax=ax[4], fig=fig, cmap='RdBu_r', cbar=True, norm=MidpointNormalize(midpoint=0))
    _, _ = plot(cf_x-rec_loc, ax=ax[5], fig=fig, cmap='RdBu_r', cbar=True, norm=MidpointNormalize(midpoint=0))
    sex_categories = ['male', 'female']  # 0,1
    race_categories = ['White', 'Asian', 'Black'] # 0,1,2

    for j in range(n):
        msg = ''
        for i, (k, v) in enumerate(do.items()):
            if k == 'sex':
                vv = sex_categories[int(v[j].item())]
                kk = 's'
            elif k == 'age':
                vv = str(v[j].item())
                kk = 'a'
            elif k == 'race':
                vv = race_categories[int(torch.argmax(v[j], dim=-1))]
                kk = 'r'
            msg += kk + '{{=}}' + vv
            msg += ', ' if (i+1) < len(list(do.keys())) else ''
        
        s = str(sex_categories[int(pa['sex'][j].item())])
        r = str(race_categories[int(torch.argmax(pa['race'][j], dim=-1))])
        a = str(int(pa['age'][j].item()))

        ax[0,j].set_title(rf'$a{{=}}{a}, \ s{{=}}{s}, \ r{{=}}{r}$',
                          pad=8, fontsize=fs-5, multialignment='center', linespacing=1.5)
        ax[1, j].set_title('rec_loc')
        ax[2, j].set_title(rf'do(${msg}$)', fontsize=fs-2, pad=8)
        ax[3, j].set_title('rec_loc - x')
        ax[4, j].set_title('cf_loc - x',
                            pad=8, fontsize=fs-5, multialignment='center', linespacing=1.5 )
        ax[5, j].set_title('cf_loc - rec_loc')

    # plt.show()
    fig.savefig(os.path.join(args.save_dir, f'viz-{args.iter}.png'), bbox_inches='tight')
    
def write_images(args, model, batch):
    bs, c, h, w = batch['x'].shape
    batch = preprocess(batch)
    # reconstructions
    zs = model.abduct(x=batch['x'], parents=batch['pa'])
    pa = {k: v for k, v in batch.items() if k != 'x'}   

    # for k, v in pa.items():
    #     print(f"{k}: {type(v)}")
    _pa = vae_preprocess(args, {k: v.clone() for k, v in pa.items()})
    
    rec_loc, _ = model.forward_latents(zs, parents=_pa)
    # counterfactuals (focus on changing sex)
    cf_pa = copy.deepcopy(pa)
    cf_pa['sex'] = 1-cf_pa['sex']
    do = {'sex': cf_pa['sex']}
    _cf_pa = vae_preprocess(args, {k: v.clone() for k, v in cf_pa.items()})   
    cf_loc, _ = model.forward_latents(zs, parents=_cf_pa)
    # plot this figure
    plot_cf_rec(args, batch['x'], cf_loc, pa, cf_pa, do, rec_loc)
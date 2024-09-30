import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return -0.5 + p_logscale - q_logscale + 0.5 * (q_logscale.exp().pow(2) + (q_loc - p_loc).pow(2)) / p_logscale.exp().pow(2)


@torch.jit.script
def sample_gaussian(loc, logscale):
    return loc + torch.exp(logscale) * torch.randn_like(loc)


class Block(nn.Module):
    def __init__(self, in_width, bottleneck, out_width, kernel_size=3, residual=True, down_rate=None):
        super().__init__()
        self.d = down_rate
        self.residual = residual
        padding = 0 if kernel_size == 1 else 1
        
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_width, bottleneck, 1, 1),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding),
            nn.GELU(),
            nn.Conv2d(bottleneck, out_width, 1, 1)
        )

        # self.conv = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_width, bottleneck, kernel_size, 1, padding),
        #     nn.ReLU(),
        #     nn.Conv2d(bottleneck, out_width, kernel_size, 1, padding)
        # )

        # self.conv = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(in_width, bottleneck, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(bottleneck, bottleneck, kernel_size, 1, padding, groups=bottleneck),
        #     nn.ReLU(),
        #     nn.Conv2d(bottleneck, out_width, 1, 1)
        # )

        if self.residual and (self.d or in_width > out_width):
            self.width_proj = nn.Conv2d(in_width, out_width, 1, 1)

        self.normalise = lambda x: (x - x.mean(dim=(1, 2, 3), keepdim=True)) \
            / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-5)

    def forward(self, x):
        # x = self.normalise(x)
        out = self.conv(x)
        if self.residual:
            if x.shape[1] != out.shape[1]:
                x = self.width_proj(x)
            out = out + x
        if self.d:
            if isinstance(self.d, float):
                out = F.adaptive_avg_pool2d(out, int(out.shape[-1] / self.d))
            else:
                out = F.avg_pool2d(out, kernel_size=self.d, stride=self.d)
        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture
        stages = []
        for i, stage in enumerate(args.enc_arch.split(',')):
            start = stage.index('b') + 1
            end = stage.index('d') if 'd' in stage else None
            n_blocks = int(stage[start:end])
            
            if i == 0:  # define network stem
                if n_blocks == 0 and 'd' not in stage:
                    print('Using stride=2 conv encoder stem.')
                    self.stem = nn.Conv2d(
                        1, args.widths[1], kernel_size=7, stride=2, padding=3)
                    continue
                else:
                    self.stem = nn.Conv2d(
                        1, args.widths[0], kernel_size=7, stride=1, padding=3)

            stages += [(args.widths[i], None) for _ in range(n_blocks)]
            if 'd' in stage:  # downsampling block
                stages += [(args.widths[i+1], int(stage[stage.index('d') + 1]))]
        # build architecture
        blocks = []
        for i, (width, d) in enumerate(stages):
            prev_width = stages[max(0, i-1)][0]
            bottleneck = int(prev_width / args.bottleneck)
            blocks.append(Block(prev_width, bottleneck, width, down_rate=d))
        # scale weights of last conv layer in each block
        for b in blocks:
            b.conv[-1].weight.data *= np.sqrt(1 / len(blocks))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.stem(x)
        acts = {}
        for block in self.blocks:
            x = block(x)
            res = x.shape[2]
            ########## Remove this for 224 resolution ################
            if res % 2 and res > 1:  # pad if odd resolution
                x = F.pad(x, [0, 1, 0, 1])
            ########## Remove this for 224 resolution ################
            acts[x.size(-1)] = x
        return acts


class DecoderBlock(nn.Module):
    def __init__(self, args, in_width, out_width, resolution):
        super().__init__()
        bottleneck = int(in_width / args.bottleneck)
        self.res = resolution
        self.stochastic = (self.res <= args.z_max_res)
        self.z_dim = args.z_dim
        k = 3 if self.res > 2 else 1
        self.prior = Block(in_width, bottleneck, 2*self.z_dim + in_width,
                           kernel_size=k, residual=False)
        if self.stochastic:
            self.posterior = Block(2*in_width + args.context_dim, bottleneck, 2*self.z_dim,
                                   kernel_size=k, residual=False)
        self.z_proj = nn.Conv2d(self.z_dim + args.context_dim, in_width, 1)
        self.z_feat_proj = nn.Conv2d(self.z_dim + in_width, out_width, 1)
        self.conv = Block(in_width, bottleneck, out_width, kernel_size=k)

    def forward_prior(self, z, t=None):  # z_{i-1}, t=temperature
        z = self.prior(z)
        p_loc = z[:, :self.z_dim, ...]
        p_logscale = z[:, self.z_dim:2*self.z_dim, ...]
        p_features = z[:, 2*self.z_dim:, ...]
        if t is not None:
            p_logscale = p_logscale + torch.tensor(t).to(z.device).log()
        return p_loc, p_logscale, p_features

    def forward_posterior(self, z, pa, x):  # z_{i-1}, pa_x, x=d_i
        h = torch.cat([z, pa, x], dim=1)
        q_loc, q_logscale = self.posterior(h).chunk(2, dim=1)
        return q_loc, q_logscale


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # parse architecture
        stages = []
        for i, stage in enumerate(args.dec_arch.split(',')):
            res = int(stage.split('b')[0])
            n_blocks = int(stage[stage.index('b') + 1:])
            stages += [(res, args.widths[::-1][i]) for _ in range(n_blocks)]
        # build architecture
        self.blocks = []
        for i, (res, width) in enumerate(stages):
            next_width = stages[min(len(stages)-1, i+1)][1]
            self.blocks.append(DecoderBlock(args, width, next_width, res))
        self._scale_params()
        self.blocks = nn.ModuleList(self.blocks)
        # bias params
        self.all_res = list(np.unique([stages[i][0]
                            for i in range(len(stages))]))
        bias = []
        for i, res in enumerate(self.all_res):
            if res <= args.bias_max_res:
                bias.append(nn.Parameter(
                    torch.zeros(1, args.widths[::-1][i], res, res)
                ))
        self.bias = nn.ParameterList(bias)

    def _scale_params(self):
        # scale down weights by num blocks
        scale = np.sqrt(1 / len(self.blocks))
        for b in self.blocks:
            b.z_proj.weight.data *= scale
            b.z_feat_proj.weight.data *= scale
            b.conv.conv[-1].weight.data *= scale
            b.prior.conv[-1].weight.data *= 0.0

    def forward(self, parents, x=None, t=None, abduct_z=False, latents=[]):
        # learnt params for each resolution r
        bias = {r.shape[2]: r for r in self.bias}
        h = bias[1].repeat(parents.shape[0], 1, 1, 1)  # h_0
        z = h  # for prior z_0

        stats = []
 
        for i, block in enumerate(self.blocks):
            res = block.res  # current block resolution, e.g. 64x64
            pa = parents[..., :res, :res]  # select parents @ res

            if h.size(-1) < res:  # upsample previous layer output
                b = bias[res] if res in bias.keys() else 0  # broadcasting
                h = b + F.interpolate(h, scale_factor=res/h.shape[-1])
                z = b + F.interpolate(z, scale_factor=res/z.shape[-1])
            # prior p(z_i | z_{i-1})
            p_loc, p_logscale, p_features = block.forward_prior(z, t)
            if block.stochastic:
                if x:
                    # posterior q(z_i | z_{i-1}, pa_x, f(x))
                    q_loc, q_logscale = block.forward_posterior(h, pa, x[res])
                    z = sample_gaussian(q_loc, q_logscale)
                    stat = dict(kl=gaussian_kl(
                        q_loc, q_logscale, p_loc, p_logscale))
                    if abduct_z:
                        stat.update(dict(z=z))#.detach()))
                    stats.append(stat)
                else:  # fixed latent or sample prior
                    try:
                        z = latents[i]    
                    except:
                        z = sample_gaussian(p_loc, p_logscale)
            else:  # deterministic path
                z = p_loc

            h = h + p_features
            z_pa = torch.cat([z, pa], dim=1)
            # h_i = h_{i-1} + f(z_i, pa_x)
            h = h + block.z_proj(z_pa)
            h = block.conv(h)

            if (i+1) < len(self.blocks):  # if not last block
                # z independent of pa_x for next layer prior
                z = block.z_feat_proj(torch.cat([z, p_features], dim=1))
        return h, stats

    def forward_cls(self, parents, _abducted_layers=-1, x=None, t=None, abduct_z=False, latents=[]):
        # learnt params for each resolution r
        bias = {r.shape[2]: r for r in self.bias}
        h = bias[1].repeat(parents.shape[0], 1, 1, 1)  # h_0
        z = h  # for prior z_0
        stats = []
        for i, block in enumerate(self.blocks):
            res = block.res  # current block resolution, e.g. 64x64
            pa = parents[..., :res, :res]  # select parents @ res

            if h.size(-1) < res:  # upsample previous layer output
                b = bias[res] if res in bias.keys() else 0  # broadcasting
                h = b + F.interpolate(h, scale_factor=res/h.shape[-1])
                z = b + F.interpolate(z, scale_factor=res/z.shape[-1])

            # prior p(z_i | z_{i-1})
            p_loc, p_logscale, p_features = block.forward_prior(z, t)

            if block.stochastic:
                if x:
                    # posterior q(z_i | z_{i-1}, pa_x, f(x))
                    q_loc, q_logscale = block.forward_posterior(h, pa, x[res])
                    if i<_abducted_layers:
                        z = q_loc
                        # z = sample_gaussian(q_loc, q_logscale)
                    else:
                        z = p_loc
                    stat = dict(kl=gaussian_kl(
                        q_loc, q_logscale, p_loc, p_logscale))
                    if abduct_z:
                        stat.update(dict(z=z))#.detach()))
                    stats.append(stat)
                else:  # fixed latent or sample prior
                    try:
                        z = latents[i]    
                    except:
                        z = sample_gaussian(p_loc, p_logscale)
            else:  # deterministic path
                z = p_loc

            h = h + p_features
            z_pa = torch.cat([z, pa], dim=1)
            # h_i = h_{i-1} + f(z_i, pa_x)
            h = h + block.z_proj(z_pa)
            h = block.conv(h)

            if (i+1) < len(self.blocks):  # if not last block
                # z independent of pa_x for next layer prior
                z = block.z_feat_proj(torch.cat([z, p_features], dim=1))
        return h, stats


class DGaussNet(nn.Module):
    def __init__(self, args):
        super(DGaussNet, self).__init__()
        self.x_loc = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1)
        self.x_logscale = nn.Conv2d(
            args.widths[0], args.input_channels, kernel_size=1, stride=1)

        if args.std_init > 0:  # if std_init=0, we random init weights for diag cov
            nn.init.zeros_(self.x_logscale.weight)
            nn.init.constant_(self.x_logscale.bias, np.log(args.std_init))

            covariance = args.x_like.split('_')[0]
            if covariance == 'fixed':
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = False
            elif covariance == 'shared':
                self.x_logscale.weight.requires_grad = False
                self.x_logscale.bias.requires_grad = True
            elif covariance == 'diag':
                self.x_logscale.weight.requires_grad = True
                self.x_logscale.bias.requires_grad = True
            else:
                NotImplementedError(f'{args.x_like} not implemented.')

    def forward(self, h, t=None):
        loc, logscale = self.x_loc(h), self.x_logscale(h)
        if t is not None:
            logscale = logscale + torch.tensor(t).to(h.device).log()
        return loc, logscale

    def approx_cdf(self, x):
        return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def nll(self, h, x):
        loc, logscale = self.forward(h)
        centered_x = x - loc
        inv_stdv = torch.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min,
                        torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return -1. * log_probs.mean(dim=(1, 2, 3))

    def sample(self, h, return_loc=True, t=None):
        if return_loc:
            x, logscale = self.forward(h)
        else:
            loc, logscale = self.forward(h, t)
            x = loc + torch.exp(logscale) * torch.randn_like(loc)
        x = torch.clamp(x, min=-1., max=1.)
        # x = (x.permute(0, 2, 3, 1) + 1.0) * 127.5  # [-1,1] to [0,255], channels last
        # return x.detach().cpu().numpy().astype(np.uint8)
        return x, logscale.exp()


class HVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.free_bits = args.free_bits
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        x_dist = args.x_like.split('_')[1]
        if x_dist == 'dgauss':
            self.likelihood = DGaussNet(args)
        else:
            NotImplementedError(f'{args.x_like} not implemented.')

    def forward(self, x, parents, beta=1):
        acts = self.encoder(x)
        h, stats = self.decoder(parents=parents, x=acts)
        nll_pp = self.likelihood.nll(h, x)
        if self.free_bits > 0:
            free_bits = torch.tensor(self.free_bits).type_as(nll_pp)
            kl_pp = 0.0
            for stat in stats:
                kl_pp += torch.maximum(
                    free_bits, stat['kl'].sum(dim=(2, 3)).mean(dim=0)
                ).sum()
        else:
            kl_pp = torch.zeros_like(nll_pp)
            for stat in stats:
                kl_pp += stat['kl'].sum(dim=(1, 2, 3))
        kl_pp /= np.prod(x.shape[1:])  # per pixel
        elbo = nll_pp.mean() + beta * kl_pp.mean()  # negative elbo (free energy)
        return dict(elbo=elbo, nll=nll_pp.mean(), kl=kl_pp.mean())

    def forward_nll(self, x, parents, _abducted_layers=40):
        acts = self.encoder(x)
        h, _ = self.decoder.forward_cls(parents=parents, x=acts, _abducted_layers=_abducted_layers) 
        nll_pp = self.likelihood.nll(h, x)
        return nll_pp

    def sample(self, parents, return_loc=True, t=None):
        h, _ = self.decoder(parents=parents, t=t)
        return self.likelihood.sample(h, return_loc, t=t)

    def abduct(self, x, parents, t=None):
        acts = self.encoder(x)
        _, stats = self.decoder.forward(x=acts, parents=parents, abduct_z=True, t=t)
        return [stat['z'] for stat in stats]  # return latents z

    def forward_latents(self, latents, parents, t=None):
        h, _ = self.decoder.forward(latents=latents, parents=parents, t=t)
        return self.likelihood.sample(h, t=t)


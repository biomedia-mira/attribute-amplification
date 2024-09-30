import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from pyro.nn import DenseNN
from pyro.infer.reparam.transform import TransformReparam
from pyro.distributions.conditional import ConditionalTransformedDistribution

from layers import (
    MLP, CNN,  # fmt: skip
    ConditionalGumbelMax,
    ConditionalAffineTransform,
    ConditionalTransformedDistributionGumbelMax,
)


class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.scm(t)
        return samples

    def sample(self, n_samples=1, t=None):
        with pyro.plate("obs", n_samples):
            samples = self.model(t)  # model defined in parent class
        return samples

    def infer_exogeneous(self, obs):
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(self, obs, intervention, num_particles=1, detach=True, t=None):
        dag_variables = self.variables.keys()
        # assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size, t)

            if hasattr(self, "discrete_variables"):  # hack for MIMIC
                # Check if we should change "finding", i.e. if its parents and/or
                # itself are not intervened, then we use its observed value.
                # This is needed due to stochastic abduction of discrete variables
                if (
                    "age" not in intervention.keys()
                    and "finding" not in intervention.keys()
                ):
                    counterfactuals["finding"] = obs["finding"]

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class FlowPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            "finding": "binary",
            "age": "continuous",
        }
        # Discrete variables that are not root nodes
        self.discrete_variables = {"finding": "binary"}
        # define base distributions
        for k in ["a"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))
        # age spline flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
        # self.age_constraints = T.ComposeTransform([
        #     T.AffineTransform(loc=4.09541458484, scale=0.32548387126),
        #     T.ExpTransform()])
        self.age_flow = T.ComposeTransform(
            [
                self.age_flow_components,
                # self.age_constraints,
            ]
        )
        # Finding (conditional) via MLP, a -> f
        finding_net = DenseNN(1, [8, 16], param_dims=[2], nonlinearity=nn.Sigmoid())
        self.finding_transform_GumbelMax = ConditionalGumbelMax(
            context_nn=finding_net, event_dim=0
        )
        # log space for sex and race
        self.sex_logit = nn.Parameter(np.log(1 / 2) * torch.ones(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        if args.setup != "sup_pgm":
            for k in ["f"]:
                self.register_buffer(f"{k}_base_loc", torch.zeros(1))
                self.register_buffer(f"{k}_base_scale", torch.ones(1))

            from resnet import ResNet, ResNet18, CustomBlock

            shared_model = ResNet(
                CustomBlock,
                layers=[2, 2, 2, 2],
                widths=[64, 128, 256, 512],
                norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),
            )

            shared_model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            kwargs = {
                "in_shape": (args.input_channels, *(args.input_res,) * 2),
                "base_model": shared_model,
            }
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNet18(num_outputs=1, **kwargs)
            # q(r | x) ~ OneHotCategorical(f(x))
            self.encoder_r = ResNet18(num_outputs=3, **kwargs)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = ResNet18(num_outputs=1, **kwargs)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNet18(num_outputs=2, context_dim=1, **kwargs)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )
        else:
            if args.enc_net == "cnn":
                input_shape=(1,192,192)
                # q(s | x) ~ Bernoulli(f(x))
                self.encoder_s = CNN(input_shape, num_outputs=1)
                # q(r | x) ~ OneHotCategorical(logits=f(x))
                self.encoder_r = CNN(input_shape, num_outputs=3)
                # q(f | x) ~ Bernoulli(f(x))
                self.encoder_f = CNN(input_shape, num_outputs=1)
                # q(a | x, f) ~ Normal(mu(x), sigma(x))
                self.encoder_a = CNN(input_shape, num_outputs=1, context_dim=1)

    def model(self, t=None):
        pyro.module("ChestPGM", self)
        # p(s), sex dist
        ps = dist.Bernoulli(logits=self.sex_logit).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(a), age flow
        pa_base = dist.Normal(self.a_base_loc, self.a_base_scale).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)
        # age_ = self.age_constraints.inv(age)
        _ = self.age_flow_components  # register with pyro

        # p(r), race dist
        pr = dist.OneHotCategorical(logits=self.race_logits).to_event(1)
        race = pyro.sample("race", pr)

        # p(f | a), finding as OneHotCategorical conditioned on age
        # finding_dist_base = dist.Gumbel(self.f_base_loc, self.f_base_scale).to_event(1)
        finding_dist_base = dist.Gumbel(torch.zeros(1), torch.ones(1)).to_event(1)
        
        finding_dist = ConditionalTransformedDistributionGumbelMax(
            finding_dist_base, [self.finding_transform_GumbelMax]
        ).condition(age)
        finding = pyro.sample("finding", finding_dist)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "finding": finding,
        }

    def guide(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
                qr_x = dist.OneHotCategorical(probs=r_probs).to_event(1)
                pyro.sample("race", qr_x)
            # q(f | x)
            if obs["finding"] is None:
                f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
                qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
                obs["finding"] = pyro.sample("finding", qf_x)
            # q(a | x, f)
            if obs["age"] is None:
                a_loc, a_logscale = self.encoder_a(obs["x"], y=obs["finding"]).chunk(
                    2, dim=-1
                )
                qa_xf = dist.Normal(a_loc, self.f(a_logscale)).to_event(1)
                pyro.sample("age_aux", qa_xf)

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            qs_x = dist.Bernoulli(probs=s_prob).to_event(1)
            # with pyro.poutine.scale(scale=0.8):
            pyro.sample("sex_aux", qs_x, obs=obs["sex"])

            # q(r | x)
            r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
            qr_x = dist.OneHotCategorical(probs=r_probs)  # .to_event(1)
            # with pyro.poutine.scale(scale=0.5):
            pyro.sample("race_aux", qr_x, obs=obs["race"])

            # q(f | x)
            f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
            qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
            pyro.sample("finding_aux", qf_x, obs=obs["finding"])

            # q(a | x, f)
            a_loc, a_logscale = self.encoder_a(obs["x"], y=obs["finding"]).chunk(
                2, dim=-1
            )
            qa_xf = dist.Normal(a_loc, self.f(a_logscale)).to_event(1)
            # with pyro.poutine.scale(scale=2):
            pyro.sample("age_aux", qa_xf, obs=obs["age"])

    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(r | x)
        r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)
        # q(f | x)
        f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
        # q(a | x, f)
        a_loc, _ = self.encoder_a(obs["x"], y=obs["finding"]).chunk(2, dim=-1)
        
        return {
            "sex": s_prob,
            "race": r_probs,
            "finding": f_prob,
            "age": a_loc,
        }

    def predict_unnorm(self, **obs):
        # q(s | x)
        s_prob = self.encoder_s(obs["x"])
        # q(r | x)
        r_logits = self.encoder_r(obs["x"])
        # q(f | x)
        f_prob = self.encoder_f(obs["x"])
        # qf_x = dist.Bernoulli(probs=torch.sigmoid(f_prob)).to_event(1)
        # obs_finding = pyro.sample("finding", qf_x)
        # q(a | x, f)
        a_loc, _ = self.encoder_a(
            obs["x"],
            y=obs['finding'],
            # y=obs_finding,
        ).chunk(2, dim=-1)

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }
        
    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass


class LegacyChestPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "race": "categorical",
            "sex": "binary",
            "finding": "binary",
            "age": "continuous",
        }
        # Discrete variables that are not root nodes
        self.discrete_variables = {
            "finding": "binary",
        }

        # prior age
        for k in ["a"]:
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # age flow
        self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
        # self.age_constraints = T.ComposeTransform([
        #     T.AffineTransform(loc=4.09541458484, scale=0.32548387126),
        #     T.ExpTransform()])
        self.age_flow = T.ComposeTransform(
            [
                self.age_flow_components,
                # self.age_constraints,
            ]
        )

        # Finding (conditional) via MLP, a -> f
        finding_net = DenseNN(
            1, [8, 16], param_dims=[2], nonlinearity=nn.Sigmoid()
        )  # .cuda()
        self.finding_transform_GumbelMax = ConditionalGumbelMax(
            context_nn=finding_net, event_dim=0
        )
        # log space for sex and race
        self.sex_logit = nn.Parameter(torch.zeros(1))
        # self.sex_logit = pyro.param(torch.zeros(1))
        self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))

        input_shape = (args.input_channels, args.input_res, args.input_res)

        if args.enc_net == "cnn":
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = CNN(input_shape, num_outputs=1)
            # q(r | x) ~ OneHotCategorical(logits=f(x))
            self.encoder_r = CNN(input_shape, num_outputs=3)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = CNN(input_shape, num_outputs=1)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = CNN(input_shape, num_outputs=1, context_dim=1)

    def model(self, t=None):
        # p(s), sex dist
        ps = dist.Bernoulli(logits=self.sex_logit).to_event(1)
        sex = pyro.sample("sex", ps)

        # p(a), age flow
        pa_base = dist.Normal(self.a_base_loc, self.a_base_scale).to_event(1)
        pa = dist.TransformedDistribution(pa_base, self.age_flow)
        age = pyro.sample("age", pa)
        # age_ = self.age_constraints.inv(age)
        _ = self.age_flow_components  # register with pyro

        # p(r), race dist
        race_dist = dist.OneHotCategorical(logits=self.race_logits).to_event(0)
        race = pyro.sample("race", race_dist)

        # p(f | a), finding as OneHotCategorical conditioned on age
        finding_dist_base = dist.Gumbel(torch.zeros(1), torch.ones(1)).to_event(1)
        finding_dist = ConditionalTransformedDistributionGumbelMax(
            finding_dist_base, [self.finding_transform_GumbelMax]
        ).condition(age)
        finding = pyro.sample("finding", finding_dist)

        return {
            "sex": sex,
            "race": race,
            "age": age,
            "finding": finding,
        }

    def guide(self, **obs):
        # print([k for k, v in obs.items() if v is not None])
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            if obs["sex"] is None:
                s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
                s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))
            # q(r | x)
            if obs["race"] is None:
                r_logits = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
                r = pyro.sample(
                    "race", dist.OneHotCategorical(logits=r_logits).to_event(1)
                )
            # q(f | x)
            if obs["finding"] is None:
                f_prob = torch.sigmoid(self.encoder_ff(obs["x"]))
                f = pyro.sample("finding", dist.Bernoulli(probs=f_prob).to_event(1))
            # q(a | x, f)
            if obs["age"] is None:
                a_loc = self.encoder_a(obs["x"], y=obs["finding"])
                pyro.sample(
                    "age",
                    dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1),
                )

    def model_anticausal(self, **obs):
        # assumes all variables are observed, train classfiers
        pyro.module("ChestPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(s | x)
            s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
            s = pyro.sample("sex", dist.Bernoulli(probs=s_prob).to_event(1))

            # q(r | x)
            r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
            r = pyro.sample("race", dist.OneHotCategorical(probs=r_probs).to_event(1))

            # q(f | x)
            f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
            qf_x = dist.Bernoulli(probs=f_prob).to_event(1)
            obs["finding"] = pyro.sample("finding", qf_x)

            # q(a | x, f)
            a_loc = self.encoder_a(obs["x"], y=obs["finding"])
            pyro.sample("age", dist.Normal(a_loc, torch.ones_like(a_loc)).to_event(1))

    def predict(self, **obs):
        # q(s | x)
        s_prob = torch.sigmoid(self.encoder_s(obs["x"]))
        # q(r | x)
        r_probs = F.softmax(self.encoder_r(obs["x"]), dim=-1)  # .squeeze()
        # q(f | x)
        f_prob = torch.sigmoid(self.encoder_f(obs["x"]))
        # q(a | x, f)
        a_loc = self.encoder_a(obs["x"], y=obs["finding"])

        return {
            "sex": s_prob,
            "race": r_probs,
            "age": a_loc,
            "finding": f_prob,
        }

    def predict_unnorm(self, **obs):
        # q(s | x)
        s_prob = self.encoder_s(obs["x"])
        # q(r | x)
        r_logits = self.encoder_r(obs["x"])
        # q(f | x)
        f_prob = self.encoder_f(obs["x"])
        qf_x = dist.Bernoulli(probs=torch.sigmoid(f_prob)).to_event(1)
        obs_finding = pyro.sample("finding", qf_x)
        # q(a | x, f)
        a_loc = self.encoder_a(
            obs["x"],
            # y=obs['finding'],
            y=obs_finding,
        )

        return {
            "sex": s_prob,
            "race": r_logits,
            "age": a_loc,
            "finding": f_prob,
        }

    def svi_model(self, **obs):
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs):
        pass

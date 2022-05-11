#! /usr/bin/env python

import torch

class GPMolGAN:
    def __init__(self, generator, discriminator, gopt, dopt,
                 generator_steps=1, discriminator_steps=1,
                 gp_weight=10.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator_steps = generator_steps
        self.discriminator_steps = discriminator_steps
        self.gp_weight = gp_weight

        self.gopt = gopt
        self.dopt = dopt

    def train_step(self, adj_real, feat_real):
        loss_g, loss_d = 0.0, 0.0
        for _ in range(self.discriminator_steps):
            latent_vec = torch.randn(feat_real.shape[0], self.generator.nlatent, device="cuda:0")
            adj_fake, feat_fake = self.generator(latent_vec)
            logits_real = self.discriminator(adj_real, feat_real)
            logits_fake = self.discriminator(adj_fake, feat_fake)
            loss_discriminator = logits_fake.mean() - logits_real.mean()
            # gradient penalty
            alpha = torch.rand(feat_real.shape[0], device="cuda:0")
            alpha1 = alpha.reshape(-1, 1, 1, 1)
            alpha2 = alpha.reshape(-1, 1, 1)
            adj_interp = adj_real * alpha1 + adj_fake * (1.0 - alpha1)
            feat_interp = feat_real * alpha2 + feat_fake * (1.0 - alpha2)
            logits = self.discriminator(adj_interp, feat_interp).sum()
            adj_grad, feat_grad = torch.autograd.grad(logits,
                                                      [adj_interp, feat_interp],
                                                      create_graph=True,
                                                      retain_graph=True)
            adj_grad_penalty = (1 - adj_grad.norm(dim=1)) ** 2
            feat_grad_penalty = (1 - adj_grad.norm(dim=2)) ** 2
            loss_gp = adj_grad_penalty.mean() + feat_grad_penalty.mean()

            # total loss
            loss = loss_discriminator + self.gp_weight * loss_gp
            loss_d = loss.item()

            self.dopt.zero_grad()
            loss.backward()
            self.dopt.step()

        for _ in range(self.generator_steps):
            latent_vec = torch.randn(feat_real.shape[0], self.generator.nlatent, device="cuda:0")
            adj_fake, feat_fake = self.generator(latent_vec)
            logits_fake = self.discriminator(adj_fake, feat_fake)
            loss = -logits_fake.mean()
            loss_g = loss.item()

            self.gopt.zero_grad()
            loss.backward()
            self.gopt.step()
        print("\r", f"Discriminator loss: {loss_d:12.6f}, Generator loss: {loss_g:12.6f}", end="")
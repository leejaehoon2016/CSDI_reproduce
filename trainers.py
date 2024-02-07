import torch
import torch.nn as nn
import numpy as np

class DiffTrainer(nn.Module):
    def __init__(
        self,
        num_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        n_samples: int = 10,
    ):
        super().__init__()
        self.num_steps = num_steps
        beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_steps) ** 2
        self.alpha = 1 - beta
        self.alpha_bar = np.cumprod(self.alpha)
        self.beta = torch.from_numpy(beta).float().cuda()
        self.alpha = torch.from_numpy(self.alpha).float().cuda()
        self.alpha_bar = torch.from_numpy(self.alpha_bar).float().cuda()
        self.n_samples = n_samples

    def forward(self, model, batch):
        observed_data = batch["observed_data"].permute(0, 2, 1).float().cuda()
        observed_mask = batch["observed_mask"].permute(0, 2, 1).float().cuda()
        observed_tp = batch["timepoints"].float().cuda()
        gt_mask = batch["gt_mask"].permute(0, 2, 1).float().cuda()

        t = torch.randint(0, self.num_steps, [len(observed_data),]).cuda()
        current_alpha = self.alpha_bar[t].unsqueeze(1).unsqueeze(1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        cond_mask = self.get_randmask(observed_mask)
        cond_data = (cond_mask * observed_data)
        score = model(noisy_data, cond_data, cond_mask, observed_tp, t)

        target_mask = observed_mask - cond_mask
        residual = (noise - score) * target_mask
        num_eval = target_mask.sum(dim=[1,2])
        loss = (residual ** 2).sum(dim=[1,2]) / (num_eval + 1e-6)
        return loss.mean()

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def impute(self, model, batch):
        observed_data = batch["observed_data"].permute(0, 2, 1).float().cuda()
        observed_mask = batch["observed_mask"].permute(0, 2, 1).float().cuda()
        observed_tp = batch["timepoints"].float().cuda()
        cond_mask = batch["gt_mask"].permute(0, 2, 1).float().cuda()
        
        imputed_samples = []

        for i in range(self.n_samples):
            current_sample = torch.randn_like(observed_data)

            for tp in range(self.num_steps - 1, -1, -1):
                cond_data = (cond_mask * observed_data)
                noisy_target = ((1 - cond_mask) * current_sample)
                t = torch.ones(len(observed_data)).cuda().long() * tp
                score = model(noisy_target, cond_data, cond_mask, observed_tp, t)

                coeff1 = 1 / self.alpha[t] ** 0.5
                coeff2 = (1 - self.alpha[t]) / (1 - self.alpha_bar[t]) ** 0.5
                coeff1, coeff2 = coeff1.unsqueeze(1).unsqueeze(1), coeff2.unsqueeze(1).unsqueeze(1)
                current_sample = coeff1 * (current_sample - coeff2 * score)

                if tp > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha_bar[t - 1]) / (1.0 - self.alpha_bar[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma.unsqueeze(1).unsqueeze(1) * noise

            imputed_samples.append(current_sample)
        imputed_samples = torch.stack(imputed_samples, dim=1).median(dim=1).values
        
        target_mask = observed_mask - cond_mask
        residual = (observed_data - imputed_samples) * target_mask
        num_eval = target_mask.sum(dim=[1,2])
        mse_loss = (residual ** 2).sum(dim=[1,2]) / (num_eval + 1e-6)
        mae_loss = residual.abs().sum(dim=[1,2]) / (num_eval + 1e-6)
        return mse_loss, mae_loss

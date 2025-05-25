import time
import torch
import torch.nn as nn
import torch.distributions as tdist
from networks import AttentionNet, FeatureAttention
from gaussian_diffusion import GaussianDiffusion

class StochDiff(nn.Module):
    def __init__(self, x_dim, h_dim, cond_dim, n_heads, dim_head, diff_dim, diff_steps, device, n_layers, pred_type='data'):
        super(StochDiff, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.diff_steps = diff_steps
        self.device = device

        self.x_proj = FeatureAttention(x_dim, h_dim, n_heads, dim_head, 0.1)

        self.prior_mean = nn.Linear(h_dim, h_dim)
        self.prior_logvar = nn.Linear(h_dim, h_dim)

        self.enc_mean = nn.Linear(h_dim*2, h_dim)
        self.enc_logvar = nn.Linear(h_dim*2, h_dim)

        self.z_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        self.denoise_net = AttentionNet(
            x_dim=x_dim,
            h_dim=h_dim,
            cond_dim=cond_dim,
            diff_step_emb_dim=diff_dim,
            num_heads=4,
            dropout=0.1
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_net,
            input_size=x_dim,
            diff_steps=diff_steps,
            pred_type=pred_type
        )

        self.rnn = nn.LSTM(
            input_size=x_dim+cond_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, ts, kld_weight, rec_weight):

        batch_size = ts.shape[0]
        seq_len = ts.shape[1]
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]

        kld = 0
        diff =0
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(seq_len):
            hidden_t = h[-1]+c[-1]
            
            x_proj_t = self.x_proj(x[:, t, :])

            prior_mean_t = self.prior_mean(hidden_t)
            prior_logvar_t = self.prior_logvar(hidden_t)

            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_dist = tdist.Normal(enc_mean_t, (enc_logvar_t * 0.5).exp())
            enc_z_t = enc_dist.rsample()
            enc_emb = self.z_proj(enc_z_t)

            kld += self.kld_gaussian(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            diff += self.diffusion.log_prob(x[:, t, :], torch.cat([enc_emb, time_emb[:, t, :]], dim=1))

            _, (h, c)= self.rnn(torch.cat([x[:, t, :], enc_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h, c))

        loss = (kld*kld_weight + diff*rec_weight)/seq_len

        return loss, kld, diff
    
    def sample(self, ts, context_len, pred_length, sampling_timesteps=None, true_obs=False):
        batch_size = ts.shape[0]
        # seq_len = ts.shape[1]
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(context_len):
            prior_t = h[-1]+c[-1]
            x_proj_t = self.x_proj(x[:, t, :])
            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, prior_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, prior_t], dim=1))
            prior_dist = tdist.Normal(enc_mean_t, (enc_logvar_t * 0.5).exp())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)
            _, (h,c) = self.rnn(torch.cat([x[:, t, :], prior_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h,c))

        pred = torch.zeros(pred_length, batch_size, self.x_dim, device=self.device)
        for t in range(pred_length):
            prior_t = h[-1]+c[-1]
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            prior_dist = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)

            pred[t] = self.diffusion.sample(cond=torch.cat([prior_emb, time_emb[:, t+context_len, :]], dim=1), sampling_timesteps=sampling_timesteps)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t+context_len, :]], dim=1).unsqueeze(1), (h,c))
        
        return pred
    
    def generation(self, ts, sample_time=None):
        batch_size = ts.shape[0]
        seq_len = ts.shape[1]
        # x = ts[:, :, :self.x_dim] 
        time_emb = ts[:, :, self.x_dim:]
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        pred = torch.zeros(seq_len, batch_size, self.x_dim, device=self.device)
        for t in range(seq_len):
            prior_t = h[-1]+c[-1]
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            prior_dist = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)

            pred[t] = self.diffusion.sample(cond=torch.cat([prior_emb, time_emb[:, t, :]], dim=1), sampling_timesteps=sample_time)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h,c))
        
        return pred
    
    def kld_gaussian(self, mean_1, log_var_1, mean_2, log_var_2):
        kld = (log_var_2 - log_var_1)/2 + (log_var_1.exp() + (mean_1 - mean_2).pow(2)) / (2 * log_var_2.exp()) - 0.5
        return kld.mean()





class StochDiff_fore(nn.Module):
    def __init__(self, x_dim, h_dim, cond_dim, n_heads, dim_head, diff_dim, diff_steps, device, n_layers, pred_type='data'):
        super(StochDiff_fore, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.diff_steps = diff_steps
        self.device = device

        self.x_proj = FeatureAttention(x_dim, h_dim, n_heads, dim_head, 0.1)

        self.prior_mean = nn.Linear(h_dim, h_dim)
        self.prior_logvar = nn.Linear(h_dim, h_dim)

        self.enc_mean = nn.Linear(h_dim*2, h_dim)
        self.enc_logvar = nn.Linear(h_dim*2, h_dim)

        self.z_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        self.denoise_net = AttentionNet(
            x_dim=x_dim,
            h_dim=h_dim,
            cond_dim=cond_dim,
            diff_step_emb_dim=diff_dim,
            num_heads=4,
            dropout=0.1
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_net,
            input_size=x_dim,
            diff_steps=diff_steps,
            pred_type=pred_type
        )

        self.rnn = nn.LSTM(
            input_size=x_dim+cond_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, ts, kld_weight, rec_weight):

        batch_size = ts.shape[0]
        seq_len = ts.shape[1]//2
        pred_length = ts.shape[1] - seq_len
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]

        kld = 0
        diff = 0
        pred = 0
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(seq_len):
            hidden_t = h[-1]+c[-1]
            
            x_proj_t = self.x_proj(x[:, t, :])

            prior_mean_t = self.prior_mean(hidden_t)
            prior_logvar_t = self.prior_logvar(hidden_t)

            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_dist = tdist.Normal(enc_mean_t, (enc_logvar_t * 0.5).exp())
            enc_z_t = enc_dist.rsample()
            enc_emb = self.z_proj(enc_z_t)

            kld += self.kld_gaussian(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            diff += self.diffusion.log_prob(x[:, t, :], torch.cat([enc_emb, time_emb[:, t, :]], dim=1), fore=False)

            _, (h, c)= self.rnn(torch.cat([x[:, t, :], enc_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h, c))

        for t in range(pred_length):
            prior_t = h[-1]+c[-1]
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            prior_dist = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)
            pred += self.diffusion.log_prob(x[:, t, :], torch.cat([enc_emb, time_emb[:, t+seq_len, :]], dim=1), fore=True)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t+seq_len, :]], dim=1).unsqueeze(1), (h,c))
        
        loss = (kld*kld_weight + diff*rec_weight)/seq_len + (pred*rec_weight)/pred_length

        return loss, kld, diff, pred
    
    def sample(self, ts, context_len, pred_length, sampling_timesteps=None, true_obs=False):
        batch_size = ts.shape[0]
        # seq_len = ts.shape[1]
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(context_len):
            prior_t = h[-1]+c[-1]
            x_proj_t = self.x_proj(x[:, t, :])
            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, prior_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, prior_t], dim=1))
            prior_dist = tdist.Normal(enc_mean_t, (enc_logvar_t * 0.5).exp())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)
            _, (h,c) = self.rnn(torch.cat([x[:, t, :], prior_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h,c))

        pred = torch.zeros(pred_length, batch_size, self.x_dim, device=self.device)
        for t in range(pred_length):
            prior_t = h[-1]+c[-1]
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            prior_dist = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)

            pred[t] = self.diffusion.sample(cond=torch.cat([prior_emb, time_emb[:, t+context_len, :]], dim=1), sampling_timesteps=sampling_timesteps)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t+context_len, :]], dim=1).unsqueeze(1), (h,c))
        
        return pred
    
    def generation(self, ts, sample_time=None):
        batch_size = ts.shape[0]
        seq_len = ts.shape[1]
        # x = ts[:, :, :self.x_dim] 
        time_emb = ts[:, :, self.x_dim:]
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        pred = torch.zeros(seq_len, batch_size, self.x_dim, device=self.device)
        for t in range(seq_len):
            prior_t = h[-1]+c[-1]
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            prior_dist = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)

            pred[t] = self.diffusion.sample(cond=torch.cat([prior_emb, time_emb[:, t, :]], dim=1), sampling_timesteps=sample_time)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h,c))
        
        return pred
    
    def kld_gaussian(self, mean_1, log_var_1, mean_2, log_var_2):
        kld = (log_var_2 - log_var_1)/2 + (log_var_1.exp() + (mean_1 - mean_2).pow(2)) / (2 * log_var_2.exp()) - 0.5
        return kld.mean()
    



class RNNDiff(nn.Module):
    def __init__(self, x_dim, h_dim, cond_dim, n_heads, dim_head, diff_dim, diff_steps, device, n_layers, pred_type='data'):
        super(RNNDiff, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.diff_steps = diff_steps
        self.device = device

        self.x_proj = FeatureAttention(x_dim, h_dim, n_heads, dim_head, 0.1)

        self.enc_mean = nn.Linear(h_dim*2, h_dim)
        self.enc_logvar = nn.Linear(h_dim*2, h_dim)

        self.z_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        self.denoise_net = AttentionNet(
            x_dim=x_dim,
            h_dim=h_dim,
            cond_dim=cond_dim,
            diff_step_emb_dim=diff_dim,
            num_heads=4,
            dropout=0.1
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_net,
            input_size=x_dim,
            diff_steps=diff_steps,
            pred_type=pred_type
        )

        self.rnn = nn.LSTM(
            input_size=x_dim+cond_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, ts, kld_weight, rec_weight):

        batch_size = ts.shape[0]
        seq_len = ts.shape[1]
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]

        kld = 0
        diff =0
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(seq_len):
            hidden_t = h[-1]+c[-1]
            
            x_proj_t = self.x_proj(x[:, t, :])

            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, hidden_t], dim=1))
            enc_dist = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
            enc_z_t = enc_dist.rsample()
            enc_emb = self.z_proj(enc_z_t)

            kld += self.kld_gaussianstd(enc_mean_t, enc_logvar_t)
            diff += self.diffusion.log_prob(x[:, t, :], torch.cat([enc_emb, time_emb[:, t, :]], dim=1))

            _, (h, c)= self.rnn(torch.cat([x[:, t, :], enc_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h, c))

        loss = (kld*kld_weight + diff*rec_weight)/seq_len

        return loss, kld, diff
    
    def sample(self, ts, context_len, pred_length, sampling_timesteps=None, true_obs=False):
        batch_size = ts.shape[0]
        # seq_len = ts.shape[1]
        x = ts[:, :, :self.x_dim]
        time_emb = ts[:, :, self.x_dim:]
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        for t in range(context_len):
            prior_t = h[-1]+c[-1]
            x_proj_t = self.x_proj(x[:, t, :])
            enc_mean_t = self.enc_mean(torch.cat([x_proj_t, prior_t], dim=1))
            enc_logvar_t = self.enc_logvar(torch.cat([x_proj_t, prior_t], dim=1))
            prior_dist = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
            z_t = prior_dist.rsample()
            prior_emb = self.z_proj(z_t)
            _, (h,c) = self.rnn(torch.cat([x[:, t, :], prior_emb, time_emb[:, t, :]], dim=1).unsqueeze(1), (h,c))

        pred = torch.zeros(pred_length, batch_size, self.x_dim, device=self.device)
        for t in range(pred_length):
            prior_t = h[-1]+c[-1]
            z_t = torch.randn(batch_size, self.h_dim, device=self.device)
            prior_emb = self.z_proj(z_t)

            pred[t] = self.diffusion.sample(cond=torch.cat([prior_emb, time_emb[:, t+context_len, :]], dim=1), sampling_timesteps=sampling_timesteps)
            _, (h,c) = self.rnn(torch.cat([pred[t], prior_emb, time_emb[:, t+context_len, :]], dim=1).unsqueeze(1), (h,c))
        
        return pred
    
    def kld_gaussianstd(self, mean, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kld.mean()
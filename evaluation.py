import argparse
import logging
import torch
import numpy as np
from Stoch_Diff import StochDiff
from tools import IndividualScaler
from dataloader import load_dataset
from crps import quantile_loss
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_logger():
    # make_sure_path_exists(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    # file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    # file_handler.setFormatter(log_formatter)
    # logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
    

if __name__ == '__main__':
    logger = init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="ecochg", help="name of the dataset to use for training")
    parser.add_argument("--window_size", type=int, default=100, help="window size for training")
    parser.add_argument("--time_dim", type=int, default=4, help="feature dimension for time embedding")
    parser.add_argument("--pred_length", type=int, default=20, help="length of prediction")
    parser.add_argument("--h_dim", type=int, default=128, help="hidden dimension of the model")
    parser.add_argument("--diff_dim", type=int, default=128, help="diffusion dimension for the model")
    parser.add_argument("--n_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="dimension of attention head")
    parser.add_argument("--n_layers", type=int, default=1, help="number of layers for the model")
    parser.add_argument("--model_name", type=str, default="sd", help="name of the model to use for evaluation")
    parser.add_argument("--sample_time", type=int, default=None, help="sample time steps for diffusion model")
    parser.add_argument("--slice", type=int, default=1, help="slice of the dataset to use for evaluation")
    args = parser.parse_args()
    window_size = args.window_size
    pred_length = args.pred_length
    time_dim = args.time_dim
    sample_time = args.sample_time

    test_set = load_dataset(args.data_name, window_size, time_dim, pred_length, scale_sty='stand', training=False)
    # if args.data_name == 'weather':
    #     _, s1, s2, _ = np.linspace(0, len(test_set), 4).astype(int)
    #     if args.slice == 1:
    #         test_set = test_set[:s1]
    #     elif args.slice == 2:
    #         test_set = test_set[s1:s2]
    #     else:
    #         test_set = test_set[s2:]
    #     logger.info(f"Data loaded, {int(len(test_set)/128)} batches in total slice {args.slice}")
    # trainloader = DataLoader(train_set, batch_size=64, shuffle=False)
    # spec_test = []
    # idx = -1
    # for i in range(7):
    #     spec_test.append(test_set[idx])
    #     idx -= pred_length
    # tgt_sum = np.sum(np.sum(np.abs(spec_test), axis=0), axis=-1)
    testloader = DataLoader(test_set, batch_size=128, shuffle=False)

    x_dim = test_set[0].shape[-1] - time_dim
    h_dim = args.h_dim
    cond_dim = h_dim+time_dim
    model = StochDiff(
        x_dim=x_dim, 
        h_dim=h_dim, 
        cond_dim=cond_dim, 
        n_heads=args.n_heads, 
        dim_head=args.dim_head, 
        diff_dim=args.diff_dim, 
        diff_steps=100, 
        n_layers=args.n_layers, 
        device=device
    )
    model = model.to(device)

    model_state = torch.load(f'save_model/{args.model_name}.pt')
    new_state_dict = OrderedDict()
    for k, v in model_state.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    num_pred = 100
    mse_bl = 0
    crps_bl = []
    logger.info(f"Start evaluating {args.data_name} dataset, using model {args.model_name} setting {h_dim}_{window_size}_{pred_length}")
    quantiles = (np.arange(20)/20.0)[1:]
    tgt_sum = 0
    tgt_mean = np.array([])
    with torch.no_grad():
        # s_time = time()
        for bc, data in enumerate(testloader):
            bs = data.shape[0]
            scaler = IndividualScaler(style='stand')
            data = data.numpy()
            y_true = data[:,-pred_length:, :-time_dim]
            y_true_agg = np.sum(y_true, axis=-1)
            tgt_sum += np.sum(np.abs(y_true_agg))
            tgt_mean = np.append(tgt_mean, np.mean(np.abs(y_true_agg), axis=-1))
            scaler.fit(data[:, :-pred_length, :])
            data = scaler.transform(data)
            obs = torch.from_numpy(data).to(torch.float32).to(device)
            obs_rep = obs.repeat(num_pred, 1, 1)
            samp_bl = model.sample(obs_rep, window_size, pred_length, sampling_timesteps=sample_time, true_obs=False).detach().to('cpu').permute(1,0,2).numpy()
            samp_bl = samp_bl.reshape(num_pred, bs, pred_length, x_dim)
            pred_bl = np.array([scaler.inverse(samp_n) for samp_n in samp_bl])
            # np.save(f'{args.data_name}_forecasts_{bc}.npy', pred_bl)
            pred_agg = np.sum(pred_bl, axis=-1)
            y_eval = np.zeros(pred_agg.shape[1:])
            for ba in range(pred_agg.shape[1]):
                c = []
                for q in quantiles:
                    # y_pred_q = np.sum(pred_bl[:, ba, :, :], axis=-1)
                    y_pred_q = np.quantile(pred_agg[:, ba], q, axis=0)
                    c.append(quantile_loss(y_true_agg[ba], y_pred_q, q))
                crps_bl.append(np.array(c))
                for t in range(pred_agg.shape[2]):
                    y_bl = pred_agg[:, ba, t]
                    gmm = GaussianMixture(n_components=5)
                    label = gmm.fit_predict(np.expand_dims(y_bl, axis=-1))
                    largest_cluster_label = np.argmax(np.bincount(label))
                    largest_cluster_center = gmm.means_[largest_cluster_label]
                    y_eval[ba, t] = largest_cluster_center[0]
            # y_eval_agg = np.sum(y_eval, axis=-1)
            mse_bl += np.sum(np.average((y_true_agg-y_eval)**2, axis=-1))
            if (bc+1)%10 == 0 or bc == 0:
                logger.info(f"Batch {bc+1} done")
        crps = np.mean(np.sum(crps_bl,axis=0)/tgt_sum)
        nrmse = np.sqrt(mse_bl/len(test_set))/np.mean(tgt_mean)
            # if bc%10 == 0 or bc == 1:
            #     logger.info(f"Batch {bc} done")
    # test_abs = np.abs(test_set)[:, :, :-time_dim]
    logger.info(f'CRPS: {crps}, NRMSE: {nrmse}')
    # logger.info(f"CRPS with true ob: {crps_ob/(len(test_set)*pred_length)}, CRPS w.o true ob: {crps_bl/(len(test_set)*pred_length)}")
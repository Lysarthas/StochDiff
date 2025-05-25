import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Stoch_Diff import StochDiff
from torch.utils.data import DataLoader
from dataloader import load_dataset

# Set random seed for numpy and torch
# np.random.seed(0)
# torch.manual_seed(33)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(33)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="number of epochs to train the model")
    parser.add_argument("--data_name", type=str, default="exchange", help="name of the dataset to use for training")
    parser.add_argument("--window_size", type=int, default=100, help="window size for training")
    parser.add_argument("--time_dim", type=int, default=4, help="feature dimension for time embedding")
    parser.add_argument("--save_name", type=str, default='sd', help="name of the file to save model state")
    parser.add_argument("--load_name", type=str, default=None, help="name of the file to load model state")
    parser.add_argument("--scale", type=str, default='stand', help="type of scaler to use for data")
    parser.add_argument("--b_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--pred", type=str, default='data', help="type of prediction to use for diffusion model")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--h_dim", type=int, default=128, help="hidden dimension for the model")
    parser.add_argument("--n_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="dimension of attention head")
    parser.add_argument("--diff_dim", type=int, default=128, help="diffusion dimension for the model")
    parser.add_argument("--diff_steps", type=int, default=100, help="diffusion steps for the model")
    parser.add_argument("--n_layers", type=int, default=1, help="number of layers for the model")
    parser.add_argument("--kld_w", type=float, default=1.0, help="weight for kld loss")
    parser.add_argument("--rec_w", type=float, default=1.0, help="weight for reconstruction loss")
    args = parser.parse_args()

    logger = init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = args.data_name
    window_size = args.window_size
    time_dim = args.time_dim
    batch_size = args.b_size

    train_set = load_dataset(window_size, scale_sty=args.scale, training=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    x_dim = train_set[0].shape[-1] - time_dim
    h_dim = args.h_dim
    cond_dim = time_dim + h_dim
    model = StochDiff(
        x_dim=x_dim, 
        h_dim=h_dim, 
        cond_dim=cond_dim,
        n_heads=args.n_heads, 
        dim_head=args.dim_head, 
        diff_dim=args.diff_dim, 
        diff_steps=args.diff_steps, 
        pred_type=args.pred, 
        n_layers=args.n_layers, 
        device=device
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    if args.load_name is not None:
        model.load_state_dict(torch.load(args.load_name + '.pt'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=0.0001, verbose=False)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    logger.info(f"Experiment {data_name} training on {device}, {params} parameters, settings: {h_dim}_{args.n_heads}_{args.dim_head}_{args.diff_dim}_{args.n_layers}")
    for epoch in range(args.epoch):
        running_loss = 0.0
        running_kld = 0.0
        running_diff = 0.0
        for i, data in enumerate(train_loader):
            x = data.to(torch.float32).to(device)
            optimizer.zero_grad()
            loss, kld, diff = model(x, args.kld_w, args.rec_w)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            running_kld += kld.mean().item()
            running_diff += diff.mean().item()
        running_loss /= len(train_loader)
        running_kld /= len(train_loader)
        running_diff /= len(train_loader)
        scheduler.step(running_loss)
        lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or (epoch + 1) % 10 == 0:
            logger.info('Epoch %d loss: %.3f kld: %.3f rec: %.3f lr: %.4f' % (epoch + 1, running_loss, running_kld, running_diff, lr))

    if args.save_name is not None:
        torch.save(model.state_dict(), f'./{args.save_name}_{data_name}_{args.epoch}.pt')
    else:
        torch.save(model.state_dict(), f'./{data_name}_{args.epoch}_model.pt')

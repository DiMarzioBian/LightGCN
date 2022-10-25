from os.path import join
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter

from config import args, cprint
from utils import load_ckpt, save_ckpt
from dataloader import get_dataloader
from models import PureMF, LightGCN
from epoch import train, evaluate


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = get_dataloader()

    if args.model == 'lightgcn':
        model = LightGCN(dataset.get_sparse_graph()).to(args.cuda)
    else:
        assert args.model == 'mf'
        model = PureMF().to(args.cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience*2, gamma=args.lr_factor)

    # load models
    epoch_cur = 0
    if not args.load:
        try:
            epoch_cur, model, scheduler = load_ckpt(model, scheduler)
            cprint(f'Succeeded load model weights:')
        except FileNotFoundError:
            cprint(f'Failed loading model weights:')
    print(f'{args.path_ckpt}')

    # init tensorboard
    if args.tensorboard:
        writer = SummaryWriter(join(args.path_board, time.strftime('%m-%d-%Hh%Mm%Ss-') + '-' + args.comment))
    else:
        writer = None
        cprint('Disable tensorboard.')

    # training
    for epoch in range(epoch_cur, args.n_epoch):
        if epoch % 10 == 0:
            evaluate(dataset, model, epoch, writer, args.multicore)

        res = train(model, optimizer, dataset, epoch=epoch, writer=writer)
        print(f'Epoch {epoch+1}/{args.n_epoch}: {res}')
        save_ckpt(epoch_cur, model.state_dict(), scheduler.state_dict())

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()

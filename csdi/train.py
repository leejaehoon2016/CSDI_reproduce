from typing import Any, Dict, Tuple

import os
import logging
import time

from collections import defaultdict

from jsonargparse import ArgumentParser, ActionConfigFile

import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import models, trainers
from torch.utils import tensorboard

def main(args):
    # DataLoader
    trainloader, validloader, testloader = dataset.get_dataloader(
        seed=1,
        nfold=0,
        batch_size=16,
        missing_ratio=0.1,
    )

    # tb_logger = tensorboard.SummaryWriter(log_dir=args.logdir)

    model = models.csdi.CSDI(35).cuda()
    # model.load_state_dict(torch.load("tmp_model.pth"))
    print("load")
    trainer = trainers.DiffTrainer().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    p1 = int(0.75 * args.epoch)
    p2 = int(0.9 * args.epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    iteration = 0
    for epoch in range(args.epoch):
        model.train()
        trainer.train()
        for batch in trainloader:
            iteration += 1
            t = time.time()

            optimizer.zero_grad()
            loss = trainer(model, batch)
            loss.backward()
            optimizer.step()

            # tb_logger.add_scalar("loss", loss, global_step=iteration)

            print(f"[{iteration:6d} ({epoch+1:3d})] [loss {loss.item():.4f}] [time {time.time()-t:.4f}s]", end="\r")
        print(" " * len(f"[{iteration:6d} ({epoch+1:3d})] [loss {loss.item():.4f}] [time {time.time()-t:.4f}s]"), end="\r")
        lr_scheduler.step()

        model.eval()
        trainer.eval()
        # with torch.no_grad():
        #     val_loss_lst = []
        #     for val_batch in validloader:
        #         loss = trainer(model, val_batch)
        #         val_loss_lst.append(loss)
        #     val_loss = sum(val_loss_lst)
            
        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                mse_lst, mae_lst, num_eval_lst = [], [], []
                for test_its, test_batch in enumerate(testloader):
                    mse_loss, mae_loss, num_eval = trainer.impute(model, test_batch)
                    mse_lst.append(mse_loss)
                    mae_lst.append(mae_loss)
                    num_eval_lst.append(num_eval)
                    tmp_mae_loss = torch.cat(mae_lst, dim=0).sum() / torch.cat(num_eval_lst, dim=0).sum()
                    print(f"[{test_its}/{len(testloader)}] [{tmp_mae_loss:.4f}]", end = "\r")
                num_eval = torch.cat(num_eval_lst, dim=0).sum()
                mse_loss, mae_loss = torch.cat(mse_lst, dim=0).sum() / num_eval, torch.cat(mae_lst, dim=0).sum() / num_eval
                print(f"epoch: {epoch+1}, mse: {mse_loss:.4f}, mae: {mae_loss:.4f}" + " "*20)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default='tmp/tmp')
    parser.add_argument("--epoch", type=int, default=200)
    # parser.add_argument("--dataset", type=str, default="ShapeNet")
    # parser.add_argument("--datadir", type=str, default="~/data/")
    # parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--num_iterations", type=int, default=100000)
    # parser.add_argument("--resolution", type=int, default=2048)
    # parser.add_argument("--save_freq", type=int, default=1000)
    # parser.add_argument("--eval_freq", type=int, default=100)
    args = parser.parse_args()
    main(args)
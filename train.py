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

def main(args):
    # DataLoader
    trainloader, validloader, testloader = dataset.get_dataloader(
        seed=1,
        nfold=0,
        batch_size=16,
        missing_ratio=0.1,
    )

    model = models.csdi.CSDI(35).cuda()
    trainer = trainers.DiffTrainer().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    iteration = 0
    for epoch in range(args.epoch):
        for batch in trainloader:
            iteration += 1
            t = time.time()
            loss = trainer(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[{iteration:6d} ({epoch+1:3d})] [loss {loss.item():.4f}] [time {time.time()-t:.4f}s]", end="\r")
        print(" " * len(f"[{iteration:6d} ({epoch+1:3d})] [loss {loss.item():.4f}] [time {time.time()-t:.4f}s]"), end="\r")

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                mse_lst, mae_lst, num_eval_lst = [], [], []
                for test_its, test_batch in enumerate(testloader):
                    mse_loss, mae_loss, num_eval = trainer.impute(model, test_batch)
                    mse_lst.append(mse_loss)
                    mae_lst.append(mae_loss)
                    num_eval_lst.append(num_eval)
                    print(f"[{test_its}/{len(testloader)}]", end = "\r")
                num_eval = torch.cat(num_eval_lst, dim=0).sum()
                mse_loss, mae_loss = torch.cat(mse_lst, dim=0).sum() / num_eval, torch.cat(mae_lst, dim=0).sum() / num_eval
                print(f"epoch: {epoch+1}, mse: {mse_loss:.4f}, mae: {mae_loss:.4f}" + " "*20)


if __name__ == "__main__":
    parser = ArgumentParser()
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
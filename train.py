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
    
    model = models.csdi.CSDI(35)
    trainer = trainers.DiffTrainer()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(args.epoch):
        for batch in trainloader:
            loss = trainer(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

    # Model, Optimizer, and Trainer
    # model = eval(args.model.class_path)(**args.model.init_args)
    # trainer = eval(args.trainer.class_path)(**args.trainer.init_args)
    # optimizer = eval(args.optimizer.class_path)(model.parameters(), **args.optimizer.init_args)
    # scheduler = utils.LRScheduler(optimizer, args.num_iterations, **args.lr_scheduler)

    logger.info(f"> # of model parameters: {model.num_parameters()}")

    # Setup
    model, trainer, optimizer, trainloader, testloader = \
            accelerator.prepare(model, trainer, optimizer, trainloader, testloader)

    # Start Training
    t = time.time()
    for iteration, batch in enumerate(trainloader, start=1):
        assert batch["x"].shape == (args.batch_size // accelerator.num_processes, args.resolution, 3)
        model.train()
        trainer.train()

        outputs = trainer(model, batch, **train_stats)
        loss = outputs["loss"]

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if iteration % args.eval_freq == 0:
            model.eval()
            trainer.eval()
            with torch.no_grad():
                eval_metrics = defaultdict(list)
                for test_batch in testloader:
                    for k, v in trainer(model, test_batch, **test_stats).items():
                        eval_metrics[k].append(v)
                eval_metrics = { k: accelerator.gather(torch.cat(v, dim=0)) for k, v in eval_metrics.items() }
                eval_metrics = { k: v.mean() if k.startswith("eval_metrics") else v for k, v in eval_metrics.items() }
                outputs.update(eval_metrics)

        if accelerator.is_main_process:
            print(f"[{iteration:6d} / {args.num_iterations}] [loss {loss.item():.4f}] [time {time.time()-t:.4f}s]", end="\r")
            tb_logger.add_scalar("loss", loss, global_step=iteration)
            tb_logger.add_scalar("optimization/batch_size", batch["x"].shape[0], global_step=iteration)
            tb_logger.add_scalar("optimization/lr", optimizer.param_groups[0]["lr"], global_step=iteration)

            pc_config_dict = dict(material=dict(cls='PointsMaterial', size=0.01))
            for k, v in outputs.items():
                if k.startswith("meshes"):
                    tb_logger.add_mesh(k, vertices=v[:1], global_step=iteration, config_dict=pc_config_dict)

            for k, v in outputs.items():
                if k.startswith("metrics") or k.startswith("eval_metrics"):
                    tb_logger.add_scalar(k, v, global_step=iteration)

            if args.save_freq > 0 and iteration % args.save_freq == 0:
                torch.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "trainer": accelerator.unwrap_model(trainer).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": iteration,
                }, os.path.join(args.logdir, f"checkpoint_{iteration:06d}.pth"))

        t = time.time()

    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="ShapeNet")
    parser.add_argument("--datadir", type=str, default="~/data/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=100)
    args = parser.parse_args()
    main(args)
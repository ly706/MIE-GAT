import os
import logging
import torch
import shutil

def set_logging_config(logdir):
    """
    set logging configuration
    :param logdir: directory put logs
    :return: None
    """
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)],
                        force=True)


def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join(exp_name, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(exp_name, 'checkpoint.pth.tar'), os.path.join(exp_name, 'model_best.pth.tar'))

def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    """
    adjust learning rate after some iterations
    :param optimizers: the optimizers
    :param lr: learning rate
    :param iteration: current iteration
    :param dec_lr_step: decrease learning rate in how many step
    :return: None
    """
    new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

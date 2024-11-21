import os
import sys
from utils import save_checkpoint
import torch.nn as nn
from metric import *
from torch.nn import functional as F
from WarmUpLR import WarmupLR
from config.data_config import config
from tensorboardX import SummaryWriter
cf = config()


class Trainer(object):
    def __init__(self, enc_module, data_loader, log, arg, config, best_step, test_acc):

        self.arg = arg
        self.config = config

        self.train_writer = SummaryWriter(os.path.join(self.arg.log_dir, 'train_scalar'))
        self.val_writer = SummaryWriter(os.path.join(self.arg.log_dir, 'val_scalar'))
        self.test_writer = SummaryWriter(os.path.join(self.arg.log_dir, 'test_scalar'))

        # initialize variables
        self.device = torch.device(f'cuda:{self.arg.gpu_ids[0]}')

        self.enc_module = enc_module.to(self.device)

        if len(self.arg.gpu_ids) > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=self.arg.gpu_ids, dim=0)
            print('done!\n')

        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # criterion and optimizer
        self.enc_loss = nn.CrossEntropyLoss()

        # set optimizer
        self.optimizer = torch.optim.SGD(params=self.enc_module.parameters(),
                                         lr=self.config['lr'],
                                         momentum=0.9,
                                         weight_decay=self.config['weight_decay'],
                                         nesterov=False)

        # set lr scheduler
        scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.arg.epoch, verbose=True)
        self.scheduler = WarmupLR(scheduler_steplr, init_lr=1e-7, num_warmup=5, warmup_strategy='cos')

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.test_acc = test_acc

        self.num_no_improve = 0

    def train(self, epoch, stop_num):

        metric_train = Metric_auc_gat()
        train_loss = AverageMeter()
        lrs = []

        for iteration in range(self.global_step, epoch):
            # set current step
            self.global_step += 1
            # set as train mode
            self.enc_module.train()
            for batch_ndx, batch in enumerate(self.data_loader['train']):
                img = batch['img'].float().to(self.device)
                struct_fea = batch['struct_fea'].float().to(self.device)
                label = batch['label'].long().to(self.device)
                _, _, pred, slice_out, batch_attention = self.enc_module(img, struct_fea)
                slice_loss = self.enc_loss(slice_out, label)
                loss = self.enc_loss(pred, label) + 0.5 * slice_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metric_train.update(label.tolist(), F.softmax(pred, dim=-1)[:, 1].tolist())
                train_loss.update(round(loss.item(), 6))
                if (batch_ndx + 1) % 50 == 0:
                    self.log.info('Epoch [{}/{}], iteration {}, Loss: {:.6f}'.format(self.global_step, self.arg.epoch, batch_ndx + 1, train_loss.avg))
                    sys.stdout.flush()

            self.scheduler.step()

            lrs.append((self.global_step, self.optimizer.param_groups[0]['lr']))

            # log training info
            if self.global_step % self.arg.log_step == 0:
                self.log.info('Epoch {}, Loss: {:.6f}, acc:{:.6f}'.format(self.global_step, train_loss.avg,
                                                                          metric_train.get_acc()))

            # evaluation
            if self.global_step % self.config['interval'] == 0:
                self.train_writer.add_scalar('Loss', train_loss.avg, self.global_step)
                self.train_writer.add_scalar('Accuracy', metric_train.get_acc(), self.global_step)
                is_best = 0
                test_acc = self.eval(partition='val')[1]
                if test_acc > self.test_acc:
                    self.log.info("Validation acc increases from {:.6f} to {:.6f}".format(self.test_acc, test_acc))
                    is_best = 1
                    self.test_acc = test_acc
                    self.best_step = self.global_step
                    self.num_no_improve = 0
                else:
                    self.num_no_improve += 1
                    self.log.info("Validation acc does not increase from {:.6f}, num_epoch_no_improvement {}".format(self.test_acc, self.num_no_improve))

                # log evaluation info
                self.log.info('val_acc : {}         step : {} '.format(test_acc, self.global_step))
                self.log.info('val_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))
                self.log.info('------------------------------------')

                self.val_writer.add_scalar('best_acc', self.test_acc, self.global_step)

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, self.arg.checkpoint_dir)

            train_loss.reset()
            metric_train.reset()

            if self.num_no_improve == stop_num:
                print("Early Stopping")
                break

    def eval(self, partition='test', log_flag=True, workbook=None, fold=None, save_test_result=None):
        with torch.no_grad():
            # set as eval mode
            self.enc_module.eval()
            self.log.info("validating....")
            result = []
            metric_val = Metric_auc_gat()
            R = []
            val_loss = AverageMeter()
            attention_list = []

            data_embed_collect = []
            slice_embed_collect = []
            label_collect = []
            for current_iteration, batch in enumerate(self.data_loader[partition]):
                img = batch['img'].float().to(self.device)
                struct_fea = batch['struct_fea'].float().to(self.device)
                label = batch['label'].long().to(self.device)

                slic_64, embed_128, pred, slice_out, batch_attention = self.enc_module(img, struct_fea)
                attention_list.append(batch_attention)
                slice_loss = self.enc_loss(slice_out, label)
                loss = self.enc_loss(pred, label) + 0.5 * slice_loss

                metric_val.update(label.tolist(), F.softmax(pred, dim=-1)[:, 1].tolist())
                metric_val.update_id(batch['id'])
                val_loss.update(round(loss.item(), 6))

                slice_embed_collect.append(slic_64)
                data_embed_collect.append(embed_128)
                label_collect.append(label)

        auc = metric_val.get_auc()
        result.append(auc)
        acc = metric_val.get_acc()
        result.append(acc)
        result = result + metric_val.get_metric()

        R.append(metric_val.get_id())
        R.append(metric_val.get_true())
        R.append(metric_val.get_pre())
        R.append(metric_val.get_pre_round())

        if partition == 'test':
            # 创建一个worksheet
            worksheet = workbook.add_sheet("fold" + str(fold))

            worksheet.write(0, 0, label="编号")
            worksheet.write(0, 1, label="真实标签")
            worksheet.write(0, 2, label="预测概率")
            worksheet.write(0, 3, label="预测标签")

            for i in range(len(R[0])):
                worksheet.write(i + 1, 0, label=R[0][i])

            for i in range(1, len(R)):
                for j in range(len(R[i])):
                    worksheet.write(j + 1, i, label=R[i][j])

            # 保存
            workbook.save(os.path.join(save_test_result, 'prediction.xls'))

        if(partition=='val'):
            self.val_writer.add_scalar('Loss', val_loss.avg, self.global_step)
            self.val_writer.add_scalar('Accuracy', metric_val.get_acc(), self.global_step)
        else:
            self.test_writer.add_scalar('Loss', val_loss.avg, self.global_step)
            self.test_writer.add_scalar('Accuracy', metric_val.get_acc(), self.global_step)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            metrics_name = ["auc", "acc", "recall", "spec", "pre", "f1"]
            print_log = ''
            for name, value in zip(metrics_name, result):
                print_log += '{}:{} '.format(name, value)
            self.log.info(print_log)
            self.log.info("pre:      "+" ".join(map(str, metric_val.get_pre()[:10])))
            self.log.info("pre_true: "+" ".join(map(str, metric_val.pre_round[:10])))
            self.log.info("true:     "+" ".join(map(str, metric_val.get_true()[:10])))
            self.log.info("confusion_matrix:"+" ".join(map(str, metric_val.get_confusion_matrix())))
            self.log.info('---------------------------')
        val_loss.reset()
        metric_val.reset()

        return result

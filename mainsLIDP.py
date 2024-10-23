import datetime

import pandas
import xlwt

from utils2 import set_logging_config
from dataset import GATData
import os
import random
import logging
import argparse
import importlib
from trainergat import Trainer
from metric import *
from models.GATCNN5S import GATEmbedding
from torch.utils.data import WeightedRandomSampler
from config.data_config import config
from torch.utils.data import DataLoader
cf = config()

# MIE-GAT with attribute features on the LIDP dataset
def main(args_opt):
    config_file = args_opt.config
    config = importlib.machinery.SourceFileLoader("", config_file).load_module().config

    args_opt.exp_name = config['dataset_name']

    result_test = []
    args_opt.save_test_result = os.path.join(args_opt.path_root, args_opt.exp_name, 'split'+str(args_opt.split), 'test_result')
    workbook = xlwt.Workbook(encoding='utf-8')

    for fold in range(1, args_opt.cross_num+1):
        if args_opt.mode == 'train':
            args_opt.log_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'split' + str(args_opt.split),
                                            'fold' + str(fold),
                                            'logs/{}/'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        else:
            args_opt.log_dir = os.path.join(args_opt.save_test_result,
                                            'fold' + str(fold),
                                            'logs/{}/'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

        args_opt.checkpoint_dir = os.path.join(args_opt.path_root, args_opt.exp_name, 'split'+str(args_opt.split), 'fold'+str(fold),
                                        'checkpoints/')

        set_logging_config(args_opt.log_dir)

        logger = logging.getLogger('{}-fold{}'.format(args_opt.mode, fold))

        # Load the configuration params of the experiment
        logger.info('Launching experiment from: {}'.format(config_file))
        logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
        logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
        print()

        logger.info('-------------command line arguments-------------')
        logger.info(args_opt)
        print()
        logger.info('-------------configs-------------')
        logger.info(config)

        # set random seed
        np.random.seed(args_opt.seed)
        torch.manual_seed(args_opt.seed)
        torch.cuda.manual_seed_all(args_opt.seed)
        random.seed(args_opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        enc_module = GATEmbedding(mid_nodes=args_opt.mid_nodes,
                                  emb_size=config['slice_emb_size'],
                                  nfeat=config['feat_in'],
                                  nhid=config['hidden'],
                                  nclass=config['nclass'],
                                  dropout=config['gat_dropout'],
                                  nheads=config['nb_heads'],
                                  alpha=config['alpha'])

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in args_opt.gpu_ids]

        dataset = GATData
        dataset_train = dataset(log=logger, partition='train', fold=fold, split=args_opt.split)
        dataset_valid = dataset(log=logger, partition='val', fold=fold, split=args_opt.split)
        dataset_test = dataset(log=logger, partition='test', fold=fold, split=args_opt.split)

        weights_train = [cf.sample_weight if sam['label'] == 0 else 1 for sam in dataset_train]
        sampler_train = WeightedRandomSampler(weights_train, num_samples=cf.sample_count_train, replacement=True)
        weights_val = [cf.sample_weight if sam['label'] == 0 else 1 for sam in dataset_valid]
        sampler_val = WeightedRandomSampler(weights_val, num_samples=cf.sample_count_val, replacement=True)
        # sampler_val = None
        sampler_test = None
        shuffle_val = False
        shuffle_train = False
        shuffle_test = False

        train_loader = DataLoader(dataset_train, batch_size=args_opt.batch_size, shuffle=shuffle_train, sampler=sampler_train,
                                  num_workers=args_opt.num_workers)
        valid_loader = DataLoader(dataset_valid, batch_size=args_opt.batch_size, shuffle=shuffle_val, sampler=sampler_val,
                                num_workers=args_opt.num_workers)
        test_loader = DataLoader(dataset_test, batch_size=args_opt.batch_size, shuffle=shuffle_test, sampler=sampler_test,
                                  num_workers=args_opt.num_workers)

        data_loader = {'train': train_loader,
                       'val': valid_loader,
                       'test': test_loader}

        # create trainer
        trainer = Trainer(enc_module=enc_module,
                              data_loader=data_loader,
                              log=logger,
                              arg=args_opt,
                              config=config,
                              best_step=0,
                              test_acc=0)

        if args_opt.mode == 'eval' or args_opt.resume == True:
            assert os.path.exists(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar')), '指定模型文件未找到，请检查'
            logger.info('find a checkpoint, loading checkpoint from {}'.format(args_opt.checkpoint_dir))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar'), map_location=torch.device('cpu'))

            logger.info('best model pack loaded')
            trainer.best_step = best_checkpoint['iteration']
            trainer.global_step = best_checkpoint['iteration']
            trainer.test_acc = best_checkpoint['test_acc']
            try:
                trainer.enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            except:
                new_enc_module_state_dict = {f'module.{k}': v for k, v in best_checkpoint['enc_module_state_dict'].items()}
                trainer.enc_module.load_state_dict(new_enc_module_state_dict)
            trainer.optimizer.load_state_dict(best_checkpoint['optimizer'])
            logger.info('current best test accuracy is: {}, at step: {}'.format(trainer.test_acc, trainer.best_step))

        elif args_opt.mode == 'train':
            if not os.path.exists(args_opt.checkpoint_dir):
                os.makedirs(args_opt.checkpoint_dir)
                logger.info('no checkpoint for model: {}, make a new one at {}'.format(
                            args_opt.exp_name,
                            args_opt.checkpoint_dir))

        else:
            logger.info('请检查mode和resume')


        if args_opt.mode == 'train':
            logger.info('--------------start fold{} training------------'.format(fold))
            trainer.train(args_opt.epoch, args_opt.stop_num)
        elif args_opt.mode == 'eval':
            logger.info('--------------start fold{} testing------------'.format(fold))
            result_test.append(trainer.eval(partition='test', workbook=workbook, fold=fold,
                                            save_test_result=args_opt.save_test_result))
        else:
            print('select a mode')
            exit()

    if args_opt.mode == 'eval':
        metrics_list = ["auc", "acc", "recall", "spec", "pre", "f1"]
        avg_result = avgResult(result_test)
        metrics_6 = np.vstack((avg_result.reshape(1, 6), np.array(result_test).reshape(args_opt.cross_num, 6)))
        metrics_df = pandas.DataFrame(metrics_6, columns=metrics_list)
        save_path = os.path.join(args_opt.save_test_result, 'test_result.csv')
        metrics_df.to_csv(save_path)
        print("model test result:\nauc:{}\nacc:{}\nrecall:{}\nspec:{}\npre:{}\nf1:{}".format(*avg_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--BioLinkBERT', type=bool, default=False)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--gpu_ids', type=list, default=[0], help='number of gpu')
    parser.add_argument('--cross_num', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=222, help='random seed')
    parser.add_argument('--epoch', type=int, default=100, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='train batch size')
    parser.add_argument('--stop_num', type=int, default=50, help='train stop_num')
    parser.add_argument('--mid_nodes', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', 'model_config.py'),
                        help='config file with parameters of the experiment. '
                             'It is assumed that the config file is placed under the directory ./config')
    parser.add_argument('--path_root', type=str, default='/media/data1/Models_ly/classification/DSEGNN/3DCNN-new--1-SGD--1/Mid4/',
                        help='path that checkpoint and logs will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')
    parser.add_argument('--display_step', type=int, default=1, help='display training information in how many step')
    parser.add_argument('--log_step', type=int, default=1, help='log information in how many steps')

    args_opt = parser.parse_args()

    main(args_opt)

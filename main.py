import os
import sys
import pdb
import argparse
import time
from datetime import datetime
from pathlib import Path
import yaml
import pprint
import random
import numpy as np
import platform

from datasets.data_loader import return_dataset
from models import create_model
from trainer import Trainer
from utils.common_utils import (
    getLogger, set_seed, save_checkpoint_best_only, trim, 
    collect_results_for_analysis)

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_args_parser():
    """
    Generates a command-line argument parser for training and testing a model.
    This function sets up an argument parser with subparsers for 'train' and 'test' modes.
    It includes options for configuration file paths, loading pretrained weights, and saving models.
    The function also reads a YAML configuration file to add additional arguments dynamically.
    Returns:
        argparse.Namespace: Parsed command-line arguments with additional attributes
        based on the configuration file and other settings.
    """
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    # to the end of this function
    # If an added arg is not on the command line, the dynamically set default here is used.
    # Dynamically set default args are either from this code or from the yaml file.
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                              help="choose from { train | test }")
    train_parser.add_argument('--cfg', type=str, default="configs/volleyball.yml",
                              help="config file path")
    train_parser.add_argument("--load_pretrained", type=int, required=False,
                              help="whether to load pretrained weights for training")
    train_parser.add_argument("--checkpoint", type=str, required=False,
                              help="a path to model checkpoint file to load pretrained weights")
    train_parser.add_argument('--not_save_best_model', action='store_true', 
                              help="not save best model in this run")
    
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                             help="choose from { train | test }")
    test_parser.add_argument('--cfg', type=str, default="configs/volleyball.yml",
                             help="config file path")
    test_parser.add_argument("--checkpoint", type=str, required=False,
                             help="a path to model checkpoint file to load pretrained weights") 
    
    # 1
    # do parse, processing stuff on the commandline
    args = parser.parse_args()
    
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    
    # 1
    # do parse, appearing to be processing the command line
    # but actually processing the yaml config file 
    args = parser.parse_args()

    # 3
    # devices and threads, details
    args.dev = 'cuda:' + str(args.dev)
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1

    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)   
        
    return args


def training_main(args):
    """
    Main training function for the group activity recognition model.
    Args:
        args are passed around all over the place
        args (Namespace): Command line arguments containing configuration settings such as:
            - log_dir (str): Directory to save logs.
            - exp_name (str): Experiment name for logging.
            - mode (str): Mode of operation (e.g., train, test).
            - seed (int): Random seed for reproducibility.
            - dataset_name (str): Name of the dataset to be used.
            - batch_size (int): Number of samples per batch.
            - num_workers (int): Number of subprocesses to use for data loading.
            - use_group_activity_weights (bool): Whether to use group activity weights.
            - use_person_action_weights (bool): Whether to use person action weights.
            - load_pretrained (bool): Whether to load a pretrained model.
            - checkpoint (str): Path to the checkpoint file.
            - learning_rate (float): Learning rate for the optimizer.
            - weight_decay (float): Weight decay for the optimizer.
            - gpu (list): List of GPU device IDs to use.
            - num_epochs (int): Number of epochs to train the model.
            - not_save_best_model (bool): Flag to prevent saving the best model.
            - checkpoint_dir (str): Directory to save checkpoints.
    Returns:
        None: The function does not return any value but logs training and testing results.
    """
    
    start_time = time.time()
    curr_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    # 3
    # configure logging
    logfile_path = os.path.join(args.log_dir, args.exp_name + '-' + curr_time + 
                                    '-' +  args.mode + '.log')
    logger = getLogger(name=__name__, path=logfile_path)
    
    # 5
    # positive values are seed values explicitly set
    # negative values means that the seed is not set
    if args.seed > 0:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        set_seed(args.seed)
    
    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    logger.info("Logfile will be saved in {}".format(logfile_path))
    
    # next two blocks
    # load and prepare data for use iteratively
    # 4
    # load two datasets for traning and testing respectively
    train_dataset = return_dataset(args.dataset_name, args, train_model=True)
    logger.info('total number of clips is {} for training data'.format(train_dataset.__len__()))
    test_dataset = return_dataset(args.dataset_name, args, train_model=False)
    logger.info('total number of clips is {} for testing data'.format(test_dataset.__len__()))

    t = train_dataset.__getitem__(0)
    # 10
    # make loaders/iterators for the datasets
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, 
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers, 
                             pin_memory=True)
    # 1
    # model
    model = create_model(args, logger)
    
    # to 176
    # loss, part of training
    # This can be useful when you have imbalanced data, where some classes have more samples than others. By assigning higher weights to the minority classes, you can give them more importance during the training process.
    if args.use_group_activity_weights:
        criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.group_activities_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion.to(args.dev)

    if args.use_person_action_weights:
        criterion_person = torch.nn.CrossEntropyLoss(weight=train_dataset.person_actions_weights)
    else:
        criterion_person = torch.nn.CrossEntropyLoss()
    criterion_person.to(args.dev)
            
    # 8
    # about the model again
    # load pretrained weights   
    # essentially torch.load some file and then model.load_state_dict the parameters
    # there got to be torch.save and model.state_dict somewhere else     
    if args.load_pretrained:  
        checkpoint = torch.load(args.checkpoint)
        params = checkpoint['state_dict']
        params = trim(params)  # trim key name prefix, implementation detail
        model.module.load_state_dict(
            params, strict=False) if hasattr(model, 'module') else model.load_state_dict(params, strict=False)
        
        logger.info("Loaded checkpoint from {}".format(args.checkpoint))
    
    # 2
    # optimizer, how to update weights with gradients already calculated
    # also part of training strategy
    # idiom
    # what params to look at, how much to change, how much to penalize large ones
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 3
    # model again
    # wrapping and moving the model so that computation is done in parallel
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    
    # 2
    # putting all things about traning together and creating a trainer
    # args: universal configurations
    # two loaders: data
    # model: model
    # loss, differentiation, optimizer: traning
    trainer = Trainer(args, model, logger, criterion, criterion_person, optimizer, 
                      train_loader=train_loader, test_loader=test_loader)
    
    best_prec1 = 0
    best_prec3 = 0
    best_prec1_person = 0
    best_prec3_person = 0
    best_epoch = -1
    is_best = False 
    
    # for loop
    # epoch, readonly in body
    for epoch in range(1, args.num_epochs + 1):
     
        if args.gpu:
            torch.cuda.empty_cache()

        # 4
        # main logic of this for body for training
        # train one epoch, making a pass of the entire dataset
        # test the model on the validation set
        trainer.train(epoch)
        logger.info(f"Finished Training epoch-{epoch}")
        prec1, prec3, prec1_person, prec3_person, loss, results = trainer.test(epoch)
        logger.info(f"Finished Testing epoch-{epoch}")
        
        # comparison and if
        # on the training main level, saving is done by keeping track of a few metric variables
        # remember best and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec3 = prec3
            best_prec1_person = prec1_person
            best_prec3_person = prec3_person
            best_epoch = epoch

            collect_results_for_analysis(
                results, test_dataset.class2idx, dataset_name=args.dataset_name, 
                result_prefix=logfile_path.split('.log')[0] + '_result', person_action_predicted=True, logger=logger) 
            
            
            # save the model if it is the best so far
            if not args.not_save_best_model:
                save_checkpoint_best_only(
                    {'cfg': args,
                     'epoch': epoch,
                     'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                     'top1': prec1,
                     'top3': prec3,
                     'top1_person': prec1_person,
                     'top3_person': prec3_person,
                     'best_top1': best_prec1,
                     'best_top3': best_prec3,
                     'best_top1_person': best_prec1_person,
                     'best_top3_person': best_prec3_person,
                     'optimizer': optimizer.state_dict()
                    },  
                    dir=args.checkpoint_dir, 
                    name=args.exp_name + '-' + curr_time)
                 
        logger.info(f"Test Prec@1: {prec1:.3f} % / Prec@3: {prec3:.3f} % / Person Prec@1: {prec1_person:.3f} % / Person Prec@3: {prec3_person:.3f}    @epoch-{epoch}")

        logger.info(f"Best Test Prec@1: {best_prec1:.3f} % / Prec@3: {best_prec3:.3f} % / Person Prec@1: {best_prec1_person:.3f} % / Person Prec@3: {best_prec3_person:.3f}   @epoch-{best_epoch}")
   
        logger.info(f"Checkpoint of epoch-{best_epoch} is/was saved to {args.checkpoint_dir}.")
     
    logger.info("Done training in {} seconds.".format(time.time() - start_time))
    
    return


def testing_main(args):
    """
    Runs the main testing procedure for the model.
    Parameters:
        args (Namespace): Command line arguments containing configuration options such as:
            - checkpoint (str): Path to the model checkpoint file.
            - log_dir (str): Directory where log files will be saved.
            - mode (str): Mode of operation (e.g., 'test').
            - seed (int): Random seed for reproducibility.
            - dataset_name (str): Name of the dataset to be used for testing.
            - batch_size (int): Number of samples per batch.
            - num_workers (int): Number of subprocesses to use for data loading.
            - use_group_activity_weights (bool): Flag to use group activity weights.
            - use_person_action_weights (bool): Flag to use person action weights.
            - gpu (bool): Flag to indicate if GPU should be used.
            - dev (str): Device to run the model on (e.g., 'cuda' or 'cpu').
    Returns:
        None: This function does not return any value. It performs testing and logs results.
    """
    start_time = time.time()
    
    exp_logname = args.checkpoint.split('/')[-1].split('_model_best.pth')[0]
    
    logfile_path = os.path.join(args.log_dir, exp_logname + '-' +  args.mode + '.log')
    
    logger = getLogger(name=__name__, path=logfile_path)

    if args.seed > 0:
        set_seed(args.seed)
    else:
        args.seed = random.randint(0, 1000000)
        set_seed(args.seed)
        
    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    print("Logfile will be saved in {}".format(logfile_path))
    
    
    test_dataset = return_dataset(args.dataset_name, args, train_model=False)
    logger.info('total number of clips is {} for testing data'.format(test_dataset.__len__()))
    
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers, 
                              pin_memory=True)
     
    
    model = create_model(args, logger)
    
    
    if args.use_group_activity_weights:
        criterion = torch.nn.CrossEntropyLoss(weight=test_dataset.group_activities_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    criterion.to(args.dev)
 
    if args.use_person_action_weights:
        criterion_person = torch.nn.CrossEntropyLoss(weight=test_dataset.person_actions_weights)
    else:
        criterion_person = torch.nn.CrossEntropyLoss()
    criterion_person.to(args.dev)
    
    
    optimizer = None
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    params = checkpoint['state_dict']
    params = trim(params)
    model.module.load_state_dict(
        params, strict=False) if hasattr(
        model, 'module') else model.load_state_dict(params, strict=False)
        
    logger.info("Loaded checkpoint from {}".format(args.checkpoint))

    # Model Num_Params
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    
    
    # Trainer
    trainer = Trainer(args, model, logger, criterion, criterion_person, optimizer, 
                      test_loader=test_loader)
    
    # Going to start testing
    epoch = checkpoint['epoch']
        
    if args.gpu:
        torch.cuda.empty_cache()

          
    prec1, prec3, prec1_person, prec3_person, loss, results = trainer.test(epoch)
    
    
    logger.info(f"Finished Testing epoch-{epoch}")
    logger.info(f"=== Test Result ===  Prec@1: {prec1:.3f} % / Prec@3: {prec3:.3f} % / Person Prec@1: {prec1_person:.3f} % / Person Prec@3: {prec3_person:.3f}     @epoch-{epoch}")
    
    
        
    collect_results_for_analysis(
        results, test_dataset.class2idx, dataset_name=args.dataset_name, 
        result_prefix=logfile_path.split('.log')[0] + '_result', person_action_predicted=True, logger=logger)  
    
    
    logger.info("Done testing in {} seconds.".format(time.time() - start_time))
    
    return


if __name__ == '__main__':
    
    args = get_args_parser()
    
    if args.mode == "train":
        training_main(args)
    elif args.mode == "test":
        testing_main(args)
    else:
        print("Wrong command mode!")
        os._exit(0)

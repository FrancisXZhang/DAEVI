import os
import json
import argparse
import numpy as np
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from core.trainer_Unflow_DP import Trainer
from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-cnf', '--config', default='configs/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
parser.add_argument("-c", "--ckptpath", default='/release_model/notexistant/', type=str)
parser.add_argument("-cn", "--ckptnumber", default='/release_model/notexistant/', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank

    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch'
                                             )
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank'])))

    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}'.format(config['model'],
    
                                                                      os.path.basename(args.config).split('.')[0]))
    if torch.cuda.is_available(): 
        print('gpu available')
        print('cuda:{}'.format(config['local_rank']))
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
        print('device: {}'.format(config['device']))
    else: 
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(config['config'], config_path)
        print('[**] create folder {}'.format(config['save_dir']))
    
    trainer = Trainer(config, debug=args.exam)
    trainer.train()


if __name__ == "__main__":
    # loading configs
    config = json.load(open(args.config))
    print('cuda', torch.cuda.is_available())
    print('config: {}'.format(config))
    config['model'] = args.model
    config['config'] = args.config
    config['initialmodel'] = args.ckptpath #added by Rema for loading initializing model
    config['chosen_epoch'] = args.ckptnumber #added by Rema for loading initializing model

    # setting distributed configurations
    config['world_size'] = 1
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = True if config['world_size'] > 1 else False

    print('world size: {}'.format(config['world_size']))
    # multiple processes have been launched by openmpi 
    config['local_rank'] = get_local_rank()
    config['global_rank'] = get_global_rank()
    main_worker(-1, config)

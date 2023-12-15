import numpy as np
import argparse
import os
import sys
import subprocess
import shutil
import ipdb

from itertools import cycle

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from utils.torch import *
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed, print_log, mkdir_if_missing

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

def visualize_model(nusc, nusc_maps, data, prev_gt_motion_3D, gt_motion_3D, recon_motion_3D, sample_motion_3D, viz_dir):
    plt.clf()

    scene_token = nusc.field2token('scene', 'name', data['seq'])[0]
    scene_record = nusc.get('scene', scene_token)
    log_record = nusc.get('log', scene_record['log_token'])
    location = log_record['location']

    nusc_map = nusc_maps[location]

    flattened = torch.cat([torch.flatten(sample_motion_3D, 0, 2), \
                           torch.flatten(recon_motion_3D, 0, 1), \
                           torch.flatten(gt_motion_3D, 0, 1), \
                           torch.flatten(prev_gt_motion_3D, 0, 1)])
    
    min_x = torch.min(flattened[:, 0]).item() - 4
    min_y = torch.min(flattened[:, 1]).item() - 4
    max_x = torch.max(flattened[:, 0]).item() + 4
    max_y = torch.max(flattened[:, 1]).item() + 4

    patch = (min_x, min_y, max_x, max_y)

    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    # fig, ax = nusc_map.render_layers(['lane'], figsize=1, bitmap=bitmap)
    # my_patch = (300, 1000, 500, 1200)
    # fig, ax = nusc_maps['singalpore-onenorth'].render_map_patch(my_patch, nusc_maps['singapore-onenorth'].non_geometric_layers, figsize=(12, 12), bitmap=bitmap)
    # plt.savefig('test.png')

    # ipdb.set_trace()

    fig, ax = nusc_map.render_map_patch(patch, nusc_map.non_geometric_layers, figsize=(10, 10), bitmap=bitmap)

    cycol = cycle('bgrcmk')

    for i, gt_traj in enumerate(gt_motion_3D):
        color = next(cycol)
        prev_gt_traj = prev_gt_motion_3D[i]
        sample_trajs = sample_motion_3D[i]

        # ipdb.set_trace()
        sample_handle = None
        gt_handle = ax.scatter(torch.cat([prev_gt_traj[:, 0], gt_traj[:, 0]]).cpu(), torch.cat([prev_gt_traj[:, 1], gt_traj[:, 1]]).cpu(), label='ground_truth', alpha=1.0, c='black', zorder=3)
        for sample_traj in sample_trajs:
            sample_handle = ax.scatter(sample_traj[:, 0].cpu(), sample_traj[:, 1].cpu(), label='sample', alpha=0.3, c='blue', zorder=2)
            for j in range(len(sample_traj) - 1):
                ax.plot((sample_traj[j, 0].cpu(), sample_traj[j, 1].cpu()), (sample_traj[j+1, 0].cpu(), sample_traj[j+1, 1].cpu()), label='sample', alpha=0.3, c='blue', zorder=3)
        
    old_legend = ax.get_legend()
    
    ax.add_artist(old_legend)

    # ax.legend(handles=[gt_handle, sample_handle])

    ax.legend(handles=[sample_handle, gt_handle])
                
    plt.axis('off')

    dir_name = f'{viz_dir}/{data["seq"]}'
    mkdir_if_missing(dir_name)
    fname = f'{viz_dir}/{data["seq"]}/frame_{int(data["frame"]):06d}.png'

    plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    # ipdb.set_trace()

    return

def get_model_prediction(data, sample_k):
    model.set_data(data)
    recon_motion_3D, _ = model.inference(mode='recon', sample_num=sample_k)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D = sample_motion_3D.transpose(0, 1).contiguous()
    return recon_motion_3D, sample_motion_3D

def save_prediction(pred, data, suffix, save_dir, pre=False):
    pred_num = 0
    pred_arr = []
    fut_data, seq_name, frame, valid_id, pred_mask = data['fut_data'], data['seq'], data['frame'], data['valid_id'], data['pred_mask']
    pre_data = data['pre_data']
    # ipdb.set_trace()
    for i in range(len(valid_id)):    # number of agents
        identity = valid_id[i]
        if pred_mask is not None and pred_mask[i] != 1.0:
            continue
        
        if pre:
            for j in range(cfg.past_frames):
                cur_data = pre_data[j]
                if len(cur_data) > 0 and identity in cur_data[:, 1]:
                    data = cur_data[cur_data[:, 1] == identity].squeeze()
                else:
                    data = most_recent_data.copy()
                    data[0] = frame + j + 1
                data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
                most_recent_data = data.copy()
                pred_arr.append(data)
            continue
        
        """future frames"""
        for j in range(cfg.future_frames):
            cur_data = fut_data[j]
            if len(cur_data) > 0 and identity in cur_data[:, 1]:
                data = cur_data[cur_data[:, 1] == identity].squeeze()
            else:
                data = most_recent_data.copy()
                data[0] = frame + j + 1
            data[[13, 15]] = pred[i, j].cpu().numpy()   # [13, 15] corresponds to 2D pos
            most_recent_data = data.copy()
            pred_arr.append(data)
        pred_num += 1

    if len(pred_arr) > 0:
        pred_arr = np.vstack(pred_arr)
        indices = [0, 1, 13, 15]            # frame, ID, x, z (remove y which is the height)
        pred_arr = pred_arr[:, indices]
        # save results
        fname = f'{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt'
        mkdir_if_missing(fname)
        np.savetxt(fname, pred_arr, fmt="%.3f")
    return pred_num

def test_model(generator, save_dir, cfg):
    total_num_pred = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot='nuscenes_erica', verbose=True)
    
    nusc_maps = {} # boston-seaport singapore-hollandvillage singapore-queenstown
    nusc_maps['singapore-onenorth'] = NuScenesMap(dataroot='nuscenes_erica', map_name='singapore-onenorth')
    nusc_maps['boston-seaport'] = NuScenesMap(dataroot='nuscenes_erica', map_name='boston-seaport')
    nusc_maps['singapore-hollandvillage'] = NuScenesMap(dataroot='nuscenes_erica', map_name='singapore-hollandvillage')
    nusc_maps['singapore-queenstown'] = NuScenesMap(dataroot='nuscenes_erica', map_name='singapore-queenstown')
    while not generator.is_epoch_end():
        data = generator()
        # ipdb.set_trace()
        if data is None:
            continue
        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        prev_gt_motion_3D = torch.stack(data['pre_motion_3D'], dim=0).to(device) * cfg.traj_scale
        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        with torch.no_grad():
            recon_motion_3D, sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        recon_motion_3D, sample_motion_3D = recon_motion_3D * cfg.traj_scale, sample_motion_3D * cfg.traj_scale

        """save samples"""
        recon_dir = os.path.join(save_dir, 'recon'); mkdir_if_missing(recon_dir)
        sample_dir = os.path.join(save_dir, 'samples'); mkdir_if_missing(sample_dir)
        gt_dir = os.path.join(save_dir, 'gt'); mkdir_if_missing(gt_dir)
        prev_gt_dir = os.path.join(save_dir, 'pre_gt'); mkdir_if_missing(prev_gt_dir)
        for i in range(sample_motion_3D.shape[0]):
            save_prediction(sample_motion_3D[i], data, f'/sample_{i:03d}', sample_dir)
        save_prediction(recon_motion_3D, data, '', recon_dir)        # save recon
        save_prediction(prev_gt_motion_3D, data, '', prev_gt_dir, pre=True)
        num_pred = save_prediction(gt_motion_3D, data, '', gt_dir)              # save gt
        total_num_pred += num_pred

        viz_dir = f'{save_dir}/viz'


        # bitmap = BitMap(nusc_maps[].dataroot, nusc_map.map_name, 'basemap')
        # fig, ax = nusc_map.render_layers(['lane'], figsize=1, bitmap=bitmap)
        # my_patch = (300, 1000, 500, 1200)
        # fig, ax = nusc_maps['singalpore-onenorth'].render_map_patch(my_patch, nusc_maps['singapore-onenorth'].non_geometric_layers, figsize=(12, 12), bitmap=bitmap)
        # plt.savefig('test.png')

        visualize_model(nusc, nusc_maps, data, prev_gt_motion_3D, gt_motion_3D, recon_motion_3D, sample_motion_3D, viz_dir)

        # ipdb.set_trace()

    print_log(f'\n\n total_num_pred: {total_num_pred}', log)
    if cfg.dataset == 'nuscenes_pred':
        scene_num = {
            'train': 32186,
            'val': 8560,
            'test': 9041
        }
        # assert total_num_pred == scene_num[generator.split]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--cleanup', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg)
    if args.epochs is None:
        epochs = [cfg.get_last_epoch()]
    else:
        epochs = [int(x) for x in args.epochs.split(',')]

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')

    for epoch in epochs:
        prepare_seed(cfg.seed)
        """ model """
        if not args.cached:
            model_id = cfg.get('model_id', 'agentformer')
            model = model_dict[model_id](cfg)
            model.set_device(device)
            model.eval()
            if epoch > 0:
                cp_path = cfg.model_path % epoch
                print_log(f'loading model from checkpoint: {cp_path}', log, display=True)
                model_cp = torch.load(cp_path, map_location='cpu')
                model.load_state_dict(model_cp['model_dict'], strict=False)

        """ save results and compute metrics """
        data_splits = [args.data_eval]

        for split in data_splits:  
            generator = data_generator(cfg, log, split=split, phase='testing')
            save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{split}'; mkdir_if_missing(save_dir)
            eval_dir = f'{save_dir}/samples'
            if not args.cached:
                test_model(generator, save_dir, cfg)

            log_file = os.path.join(cfg.log_dir, 'log_eval.txt')
            cmd = f"python eval.py --dataset {cfg.dataset} --results_dir {eval_dir} --data {split} --log {log_file}"
            subprocess.run(cmd.split(' '))

            # remove eval folder to save disk space
            if args.cleanup:
                shutil.rmtree(save_dir)

            



            




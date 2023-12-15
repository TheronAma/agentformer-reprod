import numpy as np
import argparse
from utils.config import Config
import os
from matplotlib import pyplot as plt
from utils.utils import mkdir_if_missing

VIZ_LIMIT = 100

def process_file(fname):
    traj_pred = np.loadtxt(fname)
    agent_dict = {}
    
    for agent in set(traj_pred[:, 1]):
        filter_arr = traj_pred[:, 1] == agent
        agent_dict[agent] = traj_pred[filter_arr]
    
    return agent_dict
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    args = parser.parse_args()

    cfg = Config(args.cfg)

    epoch = cfg.get_last_epoch()

    save_dir = f'{cfg.result_dir}/epoch_{epoch:04d}/{"test"}'#  mkdir_if_missing(save_dir)
    recon_dir = os.path.join(save_dir, 'recon')
    sample_dir = os.path.join(save_dir, 'samples')
    gt_dir = os.path.join(save_dir, 'gt')

    recon_scenes = sorted(os.listdir(recon_dir))
    sample_scenes = sorted(os.listdir(sample_dir))
    gt_scenes = sorted(os.listdir(gt_dir))

    for scene in gt_scenes:
        scene_recon_dir = os.path.join(recon_dir, scene)
        scene_sample_dir = os.path.join(sample_dir, scene)
        scene_gt_dir = os.path.join(gt_dir, scene)

        scene_recon_files = sorted(os.listdir(scene_recon_dir))
        scene_sample_files = sorted(os.listdir(scene_sample_dir))
        scene_gt_files = sorted(os.listdir(scene_gt_dir))

        viz_dir = os.path.join(save_dir, 'viz', scene); mkdir_if_missing(viz_dir)

        
        
        for file in scene_gt_files:
            
            gt_fname = f'{scene_gt_dir}/{file}'
            gt_pred = process_file(gt_fname)

            recon_fname = f'{scene_recon_dir}/{file}'
            recon_pred = process_file(recon_fname) #np.loadtxt(recon_fname)

            sample_fname = f'{scene_sample_dir}/{file}'[:-4]
            sample_frame_files = sorted(os.listdir(sample_fname))

            sample_pred = []

            for sample in sample_frame_files:
                fname = f'{sample_fname}/{sample}'
                sample_pred.append(process_file(fname))

            for agent in gt_pred:
                plt.clf()

                samples = 0
                for sample_dict in sample_pred:
                    plt.scatter(sample_dict[agent][:, 2], sample_dict[agent][:, 3], label='samples', alpha=0.15, c='orange')
                    if samples >= VIZ_LIMIT:
                        break
                    samples += 1
                plt.scatter(gt_pred[agent][:, 2], gt_pred[agent][:, 3], label='ground_truth', alpha=1.0, c='blue')
                # plt.scatter(recon_pred[agent][:, 2], recon_pred[agent][:, 3], label='recon_traj', alpha=0.2)
                
                
                viz_fname = os.path.join(viz_dir, f'{file[:-4]}_{int(agent)}.png')
                plt.savefig(viz_fname)
        

        

            




        


    


    
import math
import numpy as np
import argparse
from utils.config import Config
import os
from matplotlib import pyplot as plt
from utils.utils import mkdir_if_missing
import matplotlib.animation as animation
import ipdb

def compute_ADE_marginal(pred_arr, gt_arr, return_sample_vals=False, return_ped_vals=False,
                         return_argmin=False, **kwargs):
    """
    about 4 times faster than orgiinal AgentFormer computation due to numpy vectorization
    pred_arr: (num_peds, samples, frames, 2)
    gt_arr: (num_peds, frames, 2)
    """
    # assert pred_arr.shape[1] == 20, pred_arr.shape
    pred_arr = np.array(pred_arr)
    gt_arr = np.array(gt_arr)
    diff = pred_arr - np.expand_dims(gt_arr, axis=1)  # num_peds x samples x frames x 2
    dist = np.linalg.norm(diff, axis=-1)  # num_peds x samples x frames
    ades_per_sample = dist.mean(axis=-1)  # num_peds x samples
    made_per_ped = ades_per_sample.min(axis=-1)  # num_peds
    avg_made = made_per_ped.mean(axis=-1)  # (1,)
    return_vals = [avg_made]
    if return_sample_vals:  # for each sample: the avg ped ADE
        return_vals.append(ades_per_sample.mean(axis=0))
    if return_ped_vals:  # the ADE of each ped-sample (n_ped, samples)
        return_vals.append(ades_per_sample)
    if return_argmin:  # for each ped: index of sample that is argmin
        return_vals.append(ades_per_sample.argmin(axis=-1))
    return return_vals[0] if len(return_vals) == 1 else return_vals

VIZ_LIMIT = 100

def process_file(fname):
    traj_pred = np.loadtxt(fname)
    agent_dict = {}
    
    for agent in set(traj_pred[:, 1]):
        filter_arr = traj_pred[:, 1] == agent
        agent_dict[agent] = np.array(traj_pred[filter_arr])
    
    return agent_dict
        

def vis_agent(ax, agent, agent_gt, sample_pred, frame):
    # compute min ADE
    # ipdb.set_trace()

    # temporarily just plot everything
    diff = len(agent_gt) - len(sample_pred[0][agent])

    min_ade = 10000000000000.0

    # for i in range(5):
    #     ade = compute_ADE_marginal(, np.reshape( agent_gt))

    if (frame >= diff):
        for i in range(5):
            ax.scatter(sample_pred[i][agent][frame - diff, 2], sample_pred[i][agent][frame - diff, 3], label='samples', alpha=0.15, c='blue')
    
    ax.scatter(agent_gt[frame, 2],  agent_gt[frame, 3], label='ground_truth', alpha=1.0, c='black')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--data_eval', default='test')
    args = parser.parse_args()

    cfg = Config(args.cfg)

    epoch = 90 # cfg.get_last_epoch()

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
            plt.clf()
            
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
            
            # visualizing

            num_agents = len(gt_pred)

            rows = int(math.sqrt(num_agents)) + 1
            cols = int(math.sqrt(num_agents)) + 1


            fig = plt.figure(figsize=(20, 20))

            stiched_subfig, traj_subfig = fig.subfigures(2, 1, height_ratios=[0.2, 1.0])

            axs = traj_subfig.subplots(rows, cols, sharex=True, sharey=True)

            def animate(frame):
                for i, agent in enumerate(gt_pred.keys()):
                    id_col = i % cols
                    id_row = i // cols
                    ax = axs[id_row][id_col]
                    vis_agent(ax, agent, gt_pred[agent], sample_pred, frame)
            
            anim = animation.FuncAnimation(fig, animate, range(len(list(gt_pred.values())[0])), interval = 200, repeat = True, repeat_delay = 2000)
            writer = animation.PillowWriter(fps = 5)
            
            viz_fname = os.path.join(viz_dir, f'{file[:-4]}.gif')
            # plt.savefig(viz_fname)
            anim.save(viz_fname, writer=writer)

            # for agent in gt_pred:
            #     plt.clf()
            #     fig = plt.gcf()

            #     samples = 0

            #     for sample_dict in sample_pred:
            #         plt.scatter(sample_dict[agent][:, 2], sample_dict[agent][:, 3], label='samples', alpha=0.0, c='orange')
            #         if samples >= VIZ_LIMIT:
            #             break
            #         samples += 1
                
            #     plt.scatter(gt_pred[agent][:, 2], gt_pred[agent][:, 3], label='ground_truth', alpha=0.0, c='black')
            #     plt.scatter(recon_pred[agent][:, 2], recon_pred[agent][:, 3], label='recon_traj', alpha=0.0)

            #     def animate(frame):

            #         diff = len(gt_pred[agent]) - len(sample_pred[0][agent])
            #         for sample_dict in sample_pred:
            #             if (frame >= diff):
            #                 plt.scatter(sample_dict[agent][frame - diff, 2], sample_dict[agent][frame - diff, 3], label='samples', alpha=0.15, c='blue')
                    
            #         plt.scatter(gt_pred[agent][frame, 2],  gt_pred[agent][frame, 3], label='ground_truth', alpha=1.0, c='black')




        

        

            




        


    


    
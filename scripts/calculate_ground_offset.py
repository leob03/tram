"""
Helper script to calculate ground offset for Blender import.
Run this OUTSIDE of Blender using your system Python.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import numpy as np
from glob import glob
from lib.models.smpl import SMPL
from lib.vis.traj import traj_filter

def calculate_ground_offset(seq_folder):
    """
    Calculate the ground offset exactly like visualization.py does.
    Returns the offset_y value (vertical offset in TRAM coordinates).
    """

    hps_folder = f'{seq_folder}/hps'
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))

    # Load camera data
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    world_cam_R = torch.tensor(pred_cam['world_cam_R'])
    world_cam_T = torch.tensor(pred_cam['world_cam_T'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL().to(device)

    world_cam_R = world_cam_R.to(device)
    world_cam_T = world_cam_T.to(device)

    lowest = []
    for i in range(len(hps_files)):
        hps_file = hps_files[i]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat'].to(device)
        pred_shape = pred_smpl['pred_shape'].to(device)
        pred_trans = pred_smpl['pred_trans'].to(device)
        frame = pred_smpl['frame']

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred = smpl(body_pose=pred_rotmat[:,1:],
                    global_orient=pred_rotmat[:,[0]],
                    betas=pred_shape,
                    transl=pred_trans.squeeze(),
                    pose2rot=False,
                    default_smpl=True)
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        cam_r = world_cam_R[frame]
        cam_t = world_cam_T[frame]

        pred_vert_w = torch.einsum('bij,bnj->bni', cam_r, pred_vert) + cam_t[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', cam_r, pred_j3d) + cam_t[:,None]
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w.cpu(), pred_j3d_w.cpu())

        # Calculate lowest point (Y coordinate in TRAM, which is vertical)
        lowest.append(pred_vert_w[:, :, 1].min())

    # Calculate offset (same as visualization.py line 76-77)
    offset_y = torch.min(torch.stack(lowest)).item()

    return offset_y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate ground offset for Blender import')
    parser.add_argument('--seq_folder', type=str,
                        default='/mnt/share/dev-lbringer/markerless_mocap/tram/results/Barnab',
                        help='Path to sequence folder')
    args = parser.parse_args()

    print("Calculating ground offset...")
    offset_y = calculate_ground_offset(args.seq_folder)

    print(f"\nGround offset (Y in TRAM coords): {offset_y:.6f}")
    print(f"\nTo use in Blender, set this value in test_camera_blender.py:")
    print(f"offset_y = {offset_y}")

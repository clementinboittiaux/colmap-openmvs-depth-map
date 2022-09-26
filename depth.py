import torch
import argparse
from utils import load_openmvs_ply, pytorch3d_cameras_from_colmap
from pathlib import Path


def render_depth_maps(colmap_model_dir: Path, ply_mesh_path: Path, device: str = 'cpu'):
    print('Loading PLY mesh...')
    mesh = load_openmvs_ply(ply_mesh_path, device=device)

    print('Computing PyTorch3D cameras from COLMAP model...')
    cameras = pytorch3d_cameras_from_colmap(colmap_model_dir, device=device)
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute depth maps.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--colmap-model-dir', required=True, type=Path, help='path to COLMAP model directory.')
    parser.add_argument('--ply-mesh-path', required=True, type=Path, help='path to OpenMVS ply mesh.')
    args = parser.parse_args()

    render_depth_maps(args.colmap_model_dir, args.ply_mesh_path)

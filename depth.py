import argparse
from pathlib import Path

import cv2
import torch
import tqdm
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings

from colmap.scripts.python.read_write_model import read_model
from utils import load_openmvs_ply, p3d_cam_from_colmap


def render_depth_maps(
        colmap_model_dir: Path,
        ply_mesh_path: Path,
        output_dir: Path,
        filter_normals: bool = False,
        device: str = 'cpu'):
    print('Loading PLY mesh...')
    mesh = load_openmvs_ply(ply_mesh_path, device=device)

    print('Loading COLMAP model...')
    colmap_cameras, colmap_images, _ = read_model(colmap_model_dir)

    print('Rendering depth maps...')
    for colmap_image in tqdm.tqdm(colmap_images.values()):
        colmap_camera = colmap_cameras[colmap_image.camera_id]
        if colmap_camera.model != 'PINHOLE':
            tqdm.tqdm.write(f'Ignoring {colmap_image.name}, {colmap_camera.model} camera model is not supported.')
        else:
            p3d_cam = p3d_cam_from_colmap(colmap_image, colmap_camera, device=device)
            rasterizer = MeshRasterizer(
                cameras=p3d_cam,
                raster_settings=RasterizationSettings(
                    image_size=(colmap_camera.height, colmap_camera.width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    perspective_correct=True
                )
            )
            fragments = rasterizer(mesh)

            depth_map = fragments.zbuf[0, :, :, 0].clone()
            depth_map[depth_map < 0] = 0

            if filter_normals:
                normals_i_valid, normals_j_valid = torch.where(fragments.pix_to_face[0, :, :, 0] >= 0)
                faces = fragments.pix_to_face[0, normals_i_valid, normals_j_valid, 0]
                normals = mesh.faces_normals_packed()[faces]
                normals = p3d_cam.R[0].T @ normals.T
                normals_valid = normals[2] < 0  # keep normals with z axis pointing towards the camera
                normals_i_valid, normals_j_valid = normals_i_valid[normals_valid], normals_j_valid[normals_valid]
                depth_map_filtered = torch.zeros_like(depth_map)
                depth_map_filtered[normals_i_valid, normals_j_valid] = depth_map[normals_i_valid, normals_j_valid]
                depth_map = depth_map_filtered

            cv2.imwrite(str((output_dir / f'depth_{colmap_image.name}').with_suffix('.tiff')), depth_map.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute depth maps.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--colmap-model-dir', required=True, type=Path, help='path to COLMAP model directory.')
    parser.add_argument('--ply-mesh-path', required=True, type=Path, help='path to OpenMVS ply mesh.')
    parser.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser.add_argument('--filter-normals', action='store_true',
                        help="don't render depth where normals are pointing in the wrong direction.")
    parser.add_argument('--device', type=str, default='cpu', help='device.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_depth_maps(
        args.colmap_model_dir,
        args.ply_mesh_path,
        args.output_dir,
        filter_normals=args.filter_normals,
        device=args.device
    )

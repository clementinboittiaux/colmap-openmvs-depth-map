import torch
import numpy as np
from pathlib import Path
from pytorch3d.io import load_ply
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from colmap.scripts.python.read_write_model import read_model


p3dworld_to_colworld = torch.tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
colcam_to_p3dcam = torch.tensor([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)


class SimpleShader(torch.nn.Module):
    def __init__(self, blend_params: BlendParams = None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get('blend_params', self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


def load_openmvs_ply(mesh_path: Path, device: str = 'cpu') -> Meshes:
    verts, faces = load_ply(mesh_path)
    verts = (p3dworld_to_colworld[:3, :3].T @ verts.T).T
    verts, faces = verts.unsqueeze(dim=0), faces.unsqueeze(dim=0)
    textures = TexturesVertex(verts_features=torch.zeros_like(verts) + 0.5)
    return Meshes(verts, faces, textures=textures).to(device)


def pytorch3d_cameras_from_colmap(colmap_model_dir: Path, device: str = 'cpu') -> tuple[list[str], PerspectiveCameras]:
    colmap_cameras, colmap_images, _ = read_model(colmap_model_dir)
    image_names, fs, cs, rs, ts, hws = [], [], [], [], [], []
    for colmap_image in colmap_images.values():
        colmap_camera = colmap_cameras[colmap_image.camera_id]
        if colmap_camera.model != 'PINHOLE':
            print(f'Ignoring image {colmap_image.name}: {colmap_camera.model} camera model.')
        else:
            fx, fy, u0, v0 = colmap_camera.params
            r = colmap_image.qvec2rotmat()
            t = colmap_image.tvec.reshape(3, 1)
            colworld_to_colcam = torch.tensor(np.vstack([
                np.hstack([r, t]),
                [0, 0, 0, 1]
            ]), dtype=torch.float32)
            p3dworld_to_p3dcam = colcam_to_p3dcam @ colworld_to_colcam @ p3dworld_to_colworld
            image_names.append(colmap_image.name)
            fs.append(torch.tensor([fx, fy], dtype=torch.float32))
            cs.append(torch.tensor([u0, v0], dtype=torch.float32))
            rs.append(p3dworld_to_p3dcam[:3, :3].T)
            ts.append(p3dworld_to_p3dcam[:3, 3])
            hws.append(torch.tensor([colmap_camera.height, colmap_camera.width]))
    cameras = PerspectiveCameras(
        focal_length=torch.stack(fs),
        principal_point=torch.stack(cs),
        R=torch.stack(rs),
        T=torch.stack(ts),
        image_size=torch.stack(hws),
        in_ndc=False,
        device=device
    )
    return image_names, cameras

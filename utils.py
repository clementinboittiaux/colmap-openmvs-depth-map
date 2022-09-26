from pathlib import Path

import numpy as np
import torch
from pytorch3d.io import load_ply
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures.meshes import Meshes

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

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
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


def p3d_cam_from_colmap(colmap_image, colmap_camera, device: str = 'cpu') -> PerspectiveCameras:
    assert colmap_camera.model == 'PINHOLE', \
        f'Camera {colmap_camera.id} model is {colmap_camera.model}, only PINHOLE model is supported.'
    fx, fy, cx, cy = colmap_camera.params
    r = colmap_image.qvec2rotmat()
    t = colmap_image.tvec.reshape(3, 1)
    colworld_to_colcam = torch.tensor(np.vstack([
        np.hstack([r, t]),
        [0, 0, 0, 1]
    ]), dtype=torch.float32)
    p3dworld_to_p3dcam = colcam_to_p3dcam @ colworld_to_colcam @ p3dworld_to_colworld
    p3d_cam = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32),
        R=p3dworld_to_p3dcam[:3, :3].T.unsqueeze(dim=0),
        T=p3dworld_to_p3dcam[:3, 3].unsqueeze(dim=0),
        image_size=torch.tensor([[colmap_camera.height, colmap_camera.width]]),
        in_ndc=False,
        device=device
    )
    return p3d_cam

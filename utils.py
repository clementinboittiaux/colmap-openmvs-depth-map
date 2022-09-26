import torch
from pytorch3d.io import load_ply
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


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
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get('blend_params', self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


def load_mesh(mesh_path, device='cpu'):
    verts, faces = load_ply(mesh_path)
    verts = (p3dworld_to_colworld[:3, :3].T @ verts.T).T
    verts, faces = verts.unsqueeze(dim=0), faces.unsqueeze(dim=0)
    textures = TexturesVertex(verts_features=torch.zeros_like(verts) + 0.5)
    return Meshes(verts, faces, textures=textures).to(device)

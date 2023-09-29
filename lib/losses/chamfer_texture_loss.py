

import torch
import time
import torch.nn.functional as F

def mask2proj(mask_4d, threshold=0.9):
    # NOTE to make it simple, use mask=torch.ones, threshold=0.9 to select all points
    """
    convert from mask into coordinates
    input [1,1,299,299]: torch
    outout [1,N,2]
    NOTE: plt.scatter(yy,-xx), that's why swap x,y and make y * -1 after scale
    """
    mask_2d = mask_4d[0,0]
    indices_2d = torch.where(mask_2d>threshold)
    indices = torch.stack([indices_2d[1],indices_2d[0]],-1)
    assert mask_4d.shape[2] == mask_4d.shape[3]
    scale = mask_4d.shape[3]/2.0
    coords = indices/scale -1
    coords[:,1]*=(-1) # indices from top to down (row 0 to row N), coords fron down to top [-1,1]
    return coords.unsqueeze(0)

def grid_sample_from_vtx(vtx, color_map):
    """
    grid sample from vtx
    the vtx can be form mask2proj() or get_vtx_color(), or projected from vtx_3d
    color_map can be target image, rendered image, or feature map of any size
    vtx: [B, N, 2]
    color_map: [B, C, H, W]
    """
    vtx_copy = vtx.clone()
    vtx_copy[:,:,1] *= (-1)

    clr_sampled = F.grid_sample(color_map,vtx_copy.unsqueeze(2), align_corners=True).squeeze(-1).permute(0, 2, 1)

    return clr_sampled


import torch.nn as nn
import torch.nn.functional as F
import torch
from volume_renderer import FiLMSiren, LinearLayer

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class StyleField(nn.Module):
    """
    follows the architecture of deformation field in D-NeRF
    multires=10
    ref: https://github.com/albertpumarola/D-NeRF/blob/main/run_dnerf_helpers.py
    """
    def __init__(self, depth=8, width=256, skips=[4], multires=10, condition_latent_dim=0):
        super().__init__()
        self.embed_fn, self.input_ch = get_embedder(multires, 3, i=0)
        self.condition_latent_dim = condition_latent_dim
        self.input_ch += condition_latent_dim
        self.depth = depth
        self.width = width
        self.skips = skips
        layers = [nn.Linear(self.input_ch, width)]
        for i in range(depth-1):
            if i in skips:
                in_channels += self.input_ch
            else:
                in_channels = width
            
            layers += [nn.Linear(in_channels, width)]
        layers += [nn.Linear(width, 3)]
        self.net = nn.ModuleList(layers)
    
    def forward(self, x, c=None, feat=None):
        """
        Note that input for StyleSDF   ([1, 64, 64, 24, 3]
        input for EG3D             ([B, N, 3]
        """
        embed_x = self.embed_fn(x) # ([1, 64, 64, 24, 3] -> [1, 64, 64, 24, 63]
        
        if c is not None:
            batch, features = c.shape
            if x.dim() == 5:
                _, d1, d2, d3, _ = x.shape
                c = c.view(batch, 1, 1, 1, features).expand(-1, d1, d2, d3, -1)
            elif x.dim() == 3:
                _, d1, _ = x.shape
                c = c.view(batch, 1, features).expand(-1, d1, -1)
            
            embed_x = torch.cat([embed_x, c], -1)

        if feat is not None:
            embed_x = torch.cat([embed_x, feat], -1)

        h = embed_x
        for i, layer in enumerate(self.net):
            
            if i == self.depth:
                h = self.net[i](h)
            else:
                h = self.net[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([embed_x, h], -1)
        dx = h
        
        return dx


class FiLMSiren_StyleField(nn.Module):
    """
    follow the StyleSDF and pi-GAN architecture
    input_ch: 3, all other condition are from style given by the mapping network (not same model as in G)
    """
    def __init__(self, D=4, W=256, style_dim=256, input_ch=3):
        
        super(FiLMSiren_StyleField, self).__init__()
        self.net = nn.ModuleList(
            [FiLMSiren(input_ch, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)] + \
            [LinearLayer(W, 3)])

    def forward(self, x, styles):
        if styles.dim() == 3:
            raise NotImplementedError("Not implemented for W+ yet")
        mlp_out = x.contiguous()
        for i in range(len(self.net)-1):
            if styles.dim() == 2:
                mlp_out = self.net[i](mlp_out, styles)
            else:
                mlp_out = self.net[i](mlp_out, styles[:,i])
        dx = self.net[-1](mlp_out)

        return dx
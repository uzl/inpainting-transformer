from typing import List
import torch
import torch.nn as nn


def img_to_window_patches(x, K, L, r, s, t, u, flatten_patch=True, positional_mapping='local'):
    """
    It will make K by K fixed patchs from an image. 
    Then it will randomly choose square grid of L by L patchs.
    One patch from L by L patches will be used as inpaint patch. (This one we want to produce from model)
    Remaining patchs will be feed as input of the model. 
    An example: 
        for image shape (3, 384, 384),
        the main output will be (others positional information is not shown here):
            - torch.Size([48, 768])
            - torch.Size([1, 3, 16, 16]])  # it is not flatten because it will be used as image.
            or, when flatten_patch=False
            - torch.Size([48, 3, 16, 16])
            - torch.Size([1, 3, 16, 16])

    Args:
        x: Image (Channel, Height, Width). 
        K (int): Patch size. 
            For example, if we want to want to make a 16*16 pixel patch,
            then k=16.
        L (int): Subgrid arm length. We we want to take 7*7 patchs from all M*N patches,
            then L=7
        r (int): top position of (r, s) pair of subgrid start patch
        s (int): right position of (r, s) pair of subgrid start patch
        t (int): Local top position of (t, u) pair of inpaint patch 
        u (int): Local top position of (t, u) pair of inpaint patch 
        
        For more details of (r, s) and (t, u) pairs, please see section-3.1 of the paper.
    """
    with torch.no_grad():
        C, H, W = x.shape
        assert (H % K == 0) and (
            W % K == 0), 'Expected image width and height are divisible by patch size.'
        N = H // K
        M = W // K

        # Reshape [C, H, W] --> [C, N, K, M, K]
        x = x.reshape(C, N, K, M, K)
        # Re-arrange axis [C, N, K, M, K] --> [N, M, C, K, K]
        #                 [0, 1, 2, 3, 4] --> [1, 3, 0, 2, 4]
        x = x.permute(1, 3, 0, 2, 4)

        # We will choose L*L sub-grid from M*N gird.
        # The coordinate(index) of top-left patch of L*L grid will be denoted as (r, s).
        # Uniform random integer will be generated for r, s coordinate(index)
        # slicing [N, M, C, K, K] --> [L, L, C, K, K]
        sub_x = x[r:r+L, s:s+L, :, ]

        # Positional encoding for above selected L*L patchs.
        # If flag is 'global', we will use global index where the top-left patch is 0
        # and follow left-to-right english writing style.
        if positional_mapping == 'local':
            sub_pos_idx = torch.arange(0, L*L, dtype=torch.long).reshape(L, L)
        else: # global
            all_pos_idx = torch.arange(0, N*M, dtype=torch.long).reshape(N, M)
            sub_pos_idx = all_pos_idx[r:r+L, s:s+L]
            
        # Flatten in L dimension
        # [L, L, C, K, K] --> [(L, L), C, K, K]
        # i.e.: [7, 7, C, K, K] --> [49, C, K, K]
        sub_x = sub_x.flatten(0, 1)
        sub_pos_idx = sub_pos_idx.flatten(0)  # i.e. [7, 7] -> [49]

        # we will take one patch from L*L patches which we will try to inpaint.
        # For example, if L=7, then we will randomly take 1 patch as mask from 49(7*7).
        # Remaining 48 will be feed into model. The goal is to predict that msked one.
        mask_idx = (u * L) + t
        # Choose target inpaint patch [1, C, K, K]
        # i.e.: [1, C, K, K]
        inpaint_patch = sub_x[mask_idx, :, :, :]
        inpaint_pos_idx = sub_pos_idx[mask_idx]
        # Separate remaining patches. These will be the conditioning neighbors.
        # i.e.: [48, C, K, K]
        neighbor_patchs = torch.cat([
            sub_x[0:mask_idx, :, :, :],
            sub_x[mask_idx+1:, :, :, :]
        ], 0)

        neighbor_pos_idxs = torch.cat([
            sub_pos_idx[:mask_idx],
            sub_pos_idx[mask_idx+1:]
        ], 0)

        if flatten_patch:
            # Flatten in K and C dimensions. For example-
            # in neighbor_patchs: [48, C, K, K] --> [48, (C, K, K)]
            neighbor_patchs = neighbor_patchs.flatten(1, 3)

    return {
        'neighbor_patchs': neighbor_patchs,
        'neighbor_positions': neighbor_pos_idxs,
        'inpaint_patch': inpaint_patch,
        'inpaint_position': inpaint_pos_idx,
    }


 
class Attention(nn.Module):
    # https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        # (n_samples, n_heads, head_dim, n_patches + 1)
        k_t = k.transpose(-2, -1)
        dp = (
            q @ k_t
        ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        # (n_samples, n_patches + 1, dim)
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    # https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
            x
        )  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    # https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class InpaintingTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        num_channels=3,
        patch_size=16,
        window_size=7,
        embed_dim=512,
        positional_mapping='local',
        n_heads=8,
        depth=13,
        mlp_ratio=4,
        qkv_bias=True,
        p=0,
        attn_p=0,
    ):
        assert img_size % patch_size == 0, 'Image size should be multiple of patch size.'
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_channels = num_channels
        self.positional_mapping = positional_mapping

        # There are two types of positional mapping in the paper
        if positional_mapping == 'local':
            num_patches = window_size**2
        else:  # for 'global'
            num_patches = (img_size // patch_size)**2

        super().__init__()

        # Project (K*K*C) dim patch into D dim embedding
        self.linear_projection = nn.Linear(
            num_channels * (patch_size**2),
            embed_dim
        )
        # Project D dim embedding into (K*K*C) dim patch.
        self.affine_projection = nn.Linear(
            embed_dim,
            num_channels * (patch_size**2)
        )
        # Learnable parameters for positional embedding.
        # There will be total M*N embedding,
        # but in forward pass, we will dynamically select (48+1) embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches, embed_dim)
        )
        # Learnable parameter which will be inserted on behalf of inpaint-patch.
        self.x_inpaint = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # Transformer block.
        # TODO: MFSA instead of MSA.
        #       Performance gain is small. Skipping for now.
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x_nbr, patch_positions, inpaint_position):
        """
        Let's say, if window size L=7, patch size K=16, channel=3,
                   (L*L)-1 --> 48;  (C*K*K) --> 768
        Args:
            x_nbr (tensor):            [B, 48, 768] dim float
            patch_positions (tensor):  [B, 48]      dim int
            inpaint_position (tensor): [B, 1]       dim int
        """
        if self.positional_mapping == 'local':
            max_position = torch.max(patch_positions.max(), inpaint_position.max())
            assert max_position < (self.window_size**2), \
                f'Expected positional mapping from 0 to {(self.window_size**2)-1}, but found {max_position}.'

        B = x_nbr.shape[0]  # batch_size
        # linear projection from (C*K*K) to D dimensional embedding
        x_nbr = self.linear_projection(x_nbr)
        # copy x_inpaint learnable parameter for all samples in the batch
        x_inp = self.x_inpaint.repeat(B, 1, 1)

        # dynamically choose positional embedding from all(N*M).
        # i.e.: (B, 48, 512)
        neighbors_pos_embedding = self.pos_embedding[patch_positions, :]
        # i.e.: (B,  1, 512)
        inpaint_pos_embedding = self.pos_embedding[inpaint_position, :]

        # Add positional-embedding with patch projection-embedding
        x_nbr = x_nbr + neighbors_pos_embedding
        x_inp = x_inp + inpaint_pos_embedding

        # Create (L.L) * D dimensional embedding for transformer block
        x = torch.cat([x_inp, x_nbr], dim=1)

        # Transformer block with U-net structure skip connection
        blk_len = len(self.blocks)
        half_len = blk_len // 2
        tmp_x_list = []
        for i in range(half_len):
            x = self.blocks[i](x)
            tmp_x_list.append(x)

        n = half_len
        # if number of block is odd
        if blk_len % 2 == 1:
            x = self.blocks[half_len](x)
            n += 1

        tmp_x_list.reverse()
        for i, tmp_x in enumerate(tmp_x_list):
            x = self.blocks[i+n](x + tmp_x)

        x = self.norm(x)

        # [B*(L*L)*D] dim to [B, 1, D] dim by averaging.
        x = torch.mean(x, dim=1)

        # Project back to patch dimension(flatten state)
        # [B, 1, D] --> [B, (K*K*C)]
        x = self.affine_projection(x)
        x = x.reshape(B, self.num_channels, self.patch_size, self.patch_size)
        return x


if __name__ == '__main__':
    B = 8
    config = {
        "img_size": 320,  # for wood,
        "num_channels": 3,
        "patch_size": 16,
        "window_size": 7,
        "embed_dim": 512,  
        "positional_mapping": 'local',  # 'local' or 'global'
        "depth": 13,  
        "n_heads": 8, 
        "qkv_bias": True,
        "mlp_ratio": 4,
    }

    model_custom = InpaintingTransformer(**config)
    model_custom.eval()

    x_nbr = torch.rand(B, 48, 3*config['patch_size']*config['patch_size'])
    nbr_pos = torch.unsqueeze(torch.arange(
        0, 48, dtype=torch.long), 0).repeat(B, 1)
    inp_pos = torch.unsqueeze(torch.tensor(
        [48], dtype=torch.long), 0).repeat(B, 1)

    print('x_nbr:', x_nbr.shape)
    print('nbr_pos:', nbr_pos.shape)
    print('inp_pos:', inp_pos.shape)

    res_c = model_custom(x_nbr, nbr_pos, inp_pos)

    print('Output Patch:', res_c.shape)

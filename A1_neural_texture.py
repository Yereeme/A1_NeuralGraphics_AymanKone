import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class FeatureGrid(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim
        # one learnable grid per resolution. shape is (1, feat_dim, R, R), which is torch's
        # standard "batch, channels, height, width" format, just with feat_dim numbers per
        # cell instead of colors. starts as small random noise, gets shaped by training later
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, R, R) * 0.01)
            for R in resolutions
        ])
        # each resolution gives feat_dim numbers, and we're gonna glue all resolutions
        # together at the end, so the total output length is feat_dim times how many grids we have
        self.out_dim = feat_dim * len(resolutions)

    def forward(self, uv):
        N = uv.shape[0]  # how many (u,v) points we got this call

        # torch's grid_sample function expects coordinates in [-1, 1], but ours are in [0, 1].
        # u*2-1 maps 0 to -1, 1 to 1, and 0.5 to 0, so it's just a straight rescale
        grid_coords = uv * 2 - 1
        # grid_sample wants shape (batch, height_out, width_out, 2). we don't have a real
        # image grid of output points, we just have a flat list of N points, so we fake it
        # as "1 batch, N rows, 1 column" which tricks it into treating each point separately
        grid_coords = grid_coords.view(1, N, 1, 2)

        features = []
        for grid in self.grids:
            # this does the exact same bilinear blend as P1 (find the 4 nearest grid cells,
            # weight them by how close the point is to each), torch just runs the math for us
            sampled = F.grid_sample(grid, grid_coords, mode='bilinear',
                                     padding_mode='border', align_corners=False)
            # grid_sample gives back shape (1, feat_dim, N, 1), this reshapes it into the
            # more useful (N, feat_dim), so we get one row of features per input point
            sampled = sampled.view(self.feat_dim, N).T
            features.append(sampled)

        # stick every resolution's features side by side into one long row per point
        return torch.cat(features, dim=1)

class ColorMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # straight from the assignment spec: 2 hidden layers of width 64 with relu,
        # then squish the final 3 numbers into a valid [0,1] rgb color with sigmoid
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class NeuralTexture(nn.Module):
    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim=2):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
        self.mlp = ColorMLP(self.grid.out_dim)

    def forward(self, uv):
        # sample the grid to get features for this point, then decode those features into a color
        return self.mlp(self.grid(uv))


def build_training_data(texture):
    H, W = texture.shape[0], texture.shape[1]

    # (i + 0.5) / W gives the CENTER of texel i instead of its edge. e.g. for W=4,
    # texel 0 covers [0, 0.25), so its center is at u=0.125, which is what this computes
    us = (np.arange(W) + 0.5) / W
    vs = (np.arange(H) + 0.5) / H

    # meshgrid pairs up every u with every v, so we end up with every (u,v) combo in the image
    uu, vv = np.meshgrid(us, vs)  # each comes out as shape (H, W)

    # flatten both into a single column and stack them side by side -> shape (H*W, 2)
    uv = np.stack([uu.flatten(), vv.flatten()], axis=1)
    # flatten the texture the same row-by-row order, so row i here is the true color for uv row i
    colors = texture.reshape(-1, 3)

    return torch.tensor(uv, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)

def train(texture, model=None, steps=2000, batch_size=16384, lr=1e-2):
    all_uv, all_colors = build_training_data(texture)
    N = all_uv.shape[0]

    if model is None:
        model = NeuralTexture()

    # adam is just a standard optimizer, lr=1e-2 is the step size, both numbers straight from the spec
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    psnr_history = []

    for step in range(steps):
        # instead of training on all H*W texels every step (slow), grab a random batch of
        # batch_size of them. over many steps we still see the whole image, just in pieces
        idx = torch.randint(0, N, (batch_size,))
        batch_uv = all_uv[idx]
        batch_colors = all_colors[idx]

        pred_colors = model(batch_uv)
        # mse: average of (predicted - real)^2 over every predicted number. squaring makes
        # every error positive and punishes big misses more than small ones
        loss = ((pred_colors - batch_colors) ** 2).mean()

        optimizer.zero_grad()   # clear gradients left over from the last step
        loss.backward()         # figure out how much each weight is to blame for this loss
        optimizer.step()        # nudge every weight a little in the direction that lowers the loss

        if step % 100 == 0 or step == steps - 1:
            # psnr formula from the assignment, simplified since our pixels are already in [0,1]
            # (normally it's 20*log10(MAX) - 10*log10(mse), but log10(1) is just 0)
            psnr = -10 * torch.log10(loss).item()
            psnr_history.append((step, psnr))
            print(f"step {step:4d}  loss {loss.item():.6f}  psnr {psnr:.2f}")

    return model, psnr_history

def count_params(model):
    # numel() = how many numbers are in a tensor. summing this over every parameter
    # gives the total count of learnable numbers in the whole model (grids + mlp)
    return sum(p.numel() for p in model.parameters())

architecture = {
    "small":  dict(resolutions=(64,), feat_dim=2),
    "medium": dict(resolutions=(16, 32, 64), feat_dim=2),
    "large": dict(resolutions=(16, 32, 64, 128), feat_dim=4),
}

def quantize_uint8(x):
    # this whole function is the quantization formula given in the assignment/lecture,
    # just typed into code. idea: instead of storing every float32 value exactly (32 bits
    # each), split the value's actual range into 256 equal buckets (8 bits) and just store
    # which bucket each value falls in
    lo = x.min()
    hi = x.max()
    scale = (hi - lo) / 255           # how wide each of the 256 buckets is
    q = torch.round((x - lo) / scale)  # which bucket number (0-255) this value lands in
    x_hat = lo + q * scale             # convert the bucket number back into an approximate value
    return q, lo, scale, x_hat

def quantize_model(model, quantize_mlp=False):
    # go through every stored grid and replace its real values with the rounded (quantized) version
    for grid in model.grid.grids:
        q, lo, scale, x_hat = quantize_uint8(grid.data)
        grid.data.copy_(x_hat)
    # mlp is tiny so the assignment says quantizing it is optional, leaving it off by default
    if quantize_mlp:
        for p in model.mlp.parameters():
            q, lo, scale, x_hat = quantize_uint8(p.data)
            p.data.copy_(x_hat)

def evaluate_psnr(model, texture):
    # same psnr math as in train(), but run on the FULL image at once instead of a random
    # batch, so before/after quantization numbers are actually comparing the same thing
    all_uv, all_colors = build_training_data(texture)
    with torch.no_grad():   # we're just measuring quality here, not training, so no need to track gradients
        pred = model(all_uv)
    mse = ((pred - all_colors) ** 2).mean().item()
    return -10 * np.log10(mse)

def save_reconstruction(model, texture, save_path):
    all_uv, all_colors = build_training_data(texture)
    H, W = texture.shape[0], texture.shape[1]
    with torch.no_grad():
        pred = model(all_uv)
    reconstructed = pred.numpy().reshape(H, W, 3)  # unflatten the model's output back into an image shape

    original_img = (texture * 255).astype(np.uint8)
    recon_img = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
    side_by_side = np.concatenate([original_img, recon_img], axis=1)  # glue original and recon side by side
    Image.fromarray(side_by_side).save(save_path)
    print(f"saved {save_path}")

def count_params_split(model):
    # same idea as count_params, but kept separate for grid vs mlp since only the grid
    # actually gets shrunk down to 1 byte per value after quantizing
    grid_params = sum(p.numel() for p in model.grid.parameters())
    mlp_params = sum(p.numel() for p in model.mlp.parameters())
    return grid_params, mlp_params


def run_p6():
    # trains all 3 architectures on all 3 given textures, 9 runs total
    results = {}

    for tex_name in ["gradient.png", "bricks.png", "clouds.png"]:
        img = Image.open(f"textures/{tex_name}")
        texture = np.asarray(img).astype(np.float32) / 255.0
        H, W = texture.shape[0], texture.shape[1]
        raw_bytes = H * W * 3  # normal storage: 1 byte per channel, 3 channels per pixel

        for arch_name, cfg in architecture.items():
            print(f"\n=== {tex_name} | {arch_name} ===")
            model = NeuralTexture(**cfg)
            model, psnr_history = train(texture, model=model)

            num_params = count_params(model)
            neural_bytes = num_params * 4  # float32 = 4 bytes per stored number
            ratio = raw_bytes / neural_bytes
            final_psnr = psnr_history[-1][1]

            results[(tex_name, arch_name)] = {
                "psnr_history": psnr_history,
                "final_psnr": final_psnr,
                "neural_bytes": neural_bytes,
                "ratio": ratio,
            }
            print(f"final PSNR {final_psnr:.2f}  size {neural_bytes}B  ratio {ratio:.2f}")
            save_reconstruction(model, texture, f"recon_p6_{tex_name.split('.')[0]}_{arch_name}.png")

    return results


def run_p7():
    # takes the Large architecture on clouds.png and quantizes it down to 8 bit
    img = Image.open("textures/clouds.png")
    texture = np.asarray(img).astype(np.float32) / 255.0
    H, W = texture.shape[0], texture.shape[1]
    raw_bytes = H * W * 3

    model = NeuralTexture(resolutions=(16, 32, 64, 128), feat_dim=4)  # Large
    model, psnr_history = train(texture, model=model)

    psnr_before = evaluate_psnr(model, texture)
    grid_params, mlp_params = count_params_split(model)
    bytes_before = (grid_params + mlp_params) * 4  # everything still float32 here
    ratio_before = raw_bytes / bytes_before
    save_reconstruction(model, texture, "recon_p7_before.png")

    quantize_model(model, quantize_mlp=False)  # mlp stays float32, assignment says that's fine

    psnr_after = evaluate_psnr(model, texture)
    bytes_after = grid_params * 1 + mlp_params * 4  # grid values now cost 1 byte instead of 4
    ratio_after = raw_bytes / bytes_after
    save_reconstruction(model, texture, "recon_p7_after.png")

    print(f"PSNR before: {psnr_before:.2f}   size {bytes_before}B   ratio {ratio_before:.2f}")
    print(f"PSNR after:  {psnr_after:.2f}   size {bytes_after}B   ratio {ratio_after:.2f}")


def run_p8():
    # same idea as run_p6 but on my own 3 textures instead of the provided ones
    results = {}

    for tex_name in ["gfp-wood-texture.jpg", "KatanaZero.png", "Grass_texture.jpg"]:
        img = Image.open(f"textures/{tex_name}").convert("RGB")
        texture = np.asarray(img).astype(np.float32) / 255.0
        H, W = texture.shape[0], texture.shape[1]
        raw_bytes = H * W * 3

        for arch_name, cfg in architecture.items():
            print(f"\n=== {tex_name} | {arch_name} ===")
            model = NeuralTexture(**cfg)
            model, psnr_history = train(texture, model=model)

            num_params = count_params(model)
            neural_bytes = num_params * 4
            ratio = raw_bytes / neural_bytes
            final_psnr = psnr_history[-1][1]

            results[(tex_name, arch_name)] = {"final_psnr": final_psnr, "neural_bytes": neural_bytes, "ratio": ratio}
            print(f"final PSNR {final_psnr:.2f}  size {neural_bytes}B  ratio {ratio:.2f}")
            save_reconstruction(model, texture, f"recon_p8_{tex_name.split('.')[0]}_{arch_name}.png")

    return results


if __name__ == "__main__":
    # comment/uncomment whichever ones you want to run
    run_p6()
    run_p7()
    run_p8()
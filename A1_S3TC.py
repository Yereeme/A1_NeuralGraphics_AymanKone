import numpy as np
from PIL import Image

img = Image.open("textures/gradient.png")
texture = np.asarray(img).astype(np.float32) / 255.0  # values now between 0 and 1

def rgb565_quantize(c):
    # rgb565 = a real hardware format that only gives you 5 bits red, 6 bits green, 5 bits blue
    # instead of 8 bits each. levels here is "how many steps are possible" per channel (2^bits - 1)
    levels = np.array([31, 63, 31])  # 5 bits R, 6 bits G, 5 bits B
    # scale c up to the levels range, round to nearest whole step, then scale back down to [0,1]
    # this is literally just snapping a color to the nearest color rgb565 can actually represent
    q = np.round(c * levels)
    return q / levels

def compress_block(block):
    # c0 and c1 are the 2 endpoint colors for this block. using the min and max color found
    # in the block means these 2 colors will cover the full range of what's actually in the block
    c0 = rgb565_quantize(block.min(axis=(0, 1)))
    c1 = rgb565_quantize(block.max(axis=(0, 1)))
    # c2 and c3 are 2 more colors sitting between c0 and c1, so we get 4 total to choose from.
    # (2*c0 + c1)/3 is just averaging 3 numbers where c0 counts twice, so it lands 1/3 of the way
    # from c0 toward c1. same idea for c3 but 2/3 of the way there
    c2 = (2 * c0 + c1) / 3
    c3 = (c0 + 2 * c1) / 3
    palette = [c0, c1, c2, c3]

    indices = np.zeros((4, 4), dtype=int)
    for row in range(4):
        for col in range(4):
            p = block[row, col]
            # squared distance between the real pixel color and each of our 4 palette options.
            # squaring just makes sure the distance is always positive and bigger gaps count more
            d0 = ((p - c0) ** 2).sum()
            d1 = ((p - c1) ** 2).sum()
            d2 = ((p - c2) ** 2).sum()
            d3 = ((p - c3) ** 2).sum()
            distances = np.array([d0, d1, d2, d3])
            # argmin just tells us which of the 4 distances was smallest, aka the closest color
            indices[row, col] = np.argmin(distances)

    return c0, c1, indices


def decode_block(c0, c1, indices):
    # rebuild the same 4 color palette from just c0 and c1 (c2 and c3 don't need to be
    # stored since we can always recompute them the same way)
    c2 = (2 * c0 + c1) / 3
    c3 = (c0 + 2 * c1) / 3
    palette = [c0, c1, c2, c3]

    reconstructed_block = np.zeros((4, 4, 3))
    for row in range(4):
        for col in range(4):
            # just look up whichever palette color this pixel's index points to
            reconstructed_block[row, col] = palette[indices[row, col]]

    return reconstructed_block


def get_indices_and_fracs(u, v, W, H):
    # u,v are 0 to 1, this stretches them out to actual pixel coordinates in the texture
    x = u * W
    y = v * H

    # i0,j0 = the texel to the top-left of our point (floor just chops off the decimal part)
    i0 = int(np.floor(x))
    j0 = int(np.floor(y))

    # clamp so we never try to read a texel that doesn't exist (happens when u or v = 1.0 exactly)
    i0 = min(i0, W - 1)
    j0 = min(j0, H - 1)

    # the other 3 texels we need are just 1 to the right, 1 down, and 1 diagonal from i0,j0
    j1 = min(j0 + 1, H - 1)
    i1 = min(i0 + 1, W - 1)

    # s,t = how far past i0,j0 our point actually landed, as a fraction between 0 and 1.
    # this is what decides how much weight each of the 4 texels gets when we blend them
    s = x - i0
    t = y - j0

    return i0, i1, j0, j1, s, t

def sample_texture(texture, u, v):
    H, W = texture.shape[0], texture.shape[1]
    i0, i1, j0, j1, s, t = get_indices_and_fracs(u, v, W, H)

    # grab the 4 texels surrounding our point
    c00 = texture[j0, i0]
    c10 = texture[j0, i1]
    c01 = texture[j1, i0]
    c11 = texture[j1, i1]

    # bilinear blend: each corner's weight depends on how close our point is to it.
    # (1-s)(1-t) is biggest when s and t are both near 0, meaning we're right next to c00, and so on
    return (1 - s) * (1 - t) * c00 + s * (1 - t) * c10 + (1 - s) * t * c01 + s * t * c11


def run_s3tc(path):
    img = Image.open(path)
    texture = np.asarray(img).astype(np.float32) / 255.0
    H, W = texture.shape[0], texture.shape[1]
    reconstructed_texture = np.zeros_like(texture)

    # go block by block over the whole image, compress then immediately decode it
    for by in range(0, H, 4):
        for bx in range(0, W, 4):
            block = texture[by:by + 4, bx:bx + 4]
            c0, c1, indices = compress_block(block)
            reconstructed_texture[by:by + 4, bx:bx + 4] = decode_block(c0, c1, indices)

    # mse = average squared difference between original and reconstructed, over every pixel/channel
    mse = ((texture - reconstructed_texture) ** 2).mean()
    # psnr turns that error into decibels, higher = closer to the original. this simplified formula
    # only works because our pixel values are normalized to [0,1] (max possible value is 1)
    psnr = -10 * np.log10(mse)

    num_blocks = (H // 4) * (W // 4)
    # each block stores: c0 (16 bits) + c1 (16 bits) + 16 indices at 2 bits each = 64 bits = 8 bytes
    s3tc_bytes = num_blocks * 8
    raw_bytes = H * W * 3  # normal storage: 1 byte per channel, 3 channels per pixel
    compression_ratio = raw_bytes / s3tc_bytes

    print(path, "PSNR:", psnr, "compression ratio:", compression_ratio)

    out = (np.clip(reconstructed_texture, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(path.replace(".png", "_s3tc.png"))

    return reconstructed_texture


recon_gradient = run_s3tc("textures/gradient.png")
recon_bricks = run_s3tc("textures/bricks.png")
recon_clouds = run_s3tc("textures/clouds.png")
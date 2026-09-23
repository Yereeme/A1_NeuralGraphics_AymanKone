# Neural Texture Compression (A1, 15-474/674 Neural Graphics)

Compresses a texture into a small learned representation (a multi-resolution
feature grid + a small MLP decoder) and compares it against classical S3TC
block compression.

## Setup

1. Create a virtual environment and activate it
2. `pip install torch numpy pillow`
3. Put texture files in a `textures/` folder next to the scripts:
   - `gradient.png`, `bricks.png`, `clouds.png` (provided by the course)
   - `gfp-wood-texture.jpg`, `Grass_texture.jpg` (sourced from Wikimedia Commons)
   - `KatanaZero.png` (screenshot/wallpaper from the game Katana Zero by Askiisoft,
     used here for educational compression testing only)

## Files

- `python A1_S3TC.py` — P1 (bilinear texture sampler) and P2 (S3TC baseline compression).
  Running it processes gradient/bricks/clouds and prints PSNR + compression ratio for each,
  and saves `_s3tc.png` reconstructions next to each original.
- `A1_neural_texture.py` — P3 through P8, the neural texture compression pipeline.
  Contains `run_p6()`, `run_p7()`, `run_p8()`, each runnable on its own.

## How to reproduce results
"""Subset of registration nonrigid/utils helpers required by detection denoising."""

import numpy as np
import torch


def spatial_taper(sig, Ly, Lx):
    """Compute a smooth 2D edge taper mask."""
    y = torch.arange(0, Ly, dtype=torch.double)
    x = torch.arange(0, Lx, dtype=torch.double)
    x = (x - x.mean()).abs()
    y = (y - y.mean()).abs()
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1.0 / (1.0 + torch.exp((y - mY) / sig))
    maskX = 1.0 / (1.0 + torch.exp((x - mX) / sig))
    return maskY[:, None] * maskX


def kernelD(xs, ys, sigL=0.85):
    """Gaussian interpolation kernel used for subpixel upsampling."""
    xs0, xs1 = torch.meshgrid(xs, xs, indexing="ij")
    ys0, ys1 = torch.meshgrid(ys, ys, indexing="ij")
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    return torch.exp(-(dxs**2 + dys**2) / (2 * sigL**2))


def kernelD2(xs, ys):
    """Row-normalized Gaussian kernel over a 2D grid."""
    ys, xs = torch.meshgrid(xs, ys, indexing="ij")
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = torch.exp(-((ys - ys.T) ** 2 + (xs - xs.T) ** 2))
    return R / torch.sum(R, axis=0)


def mat_upsample(lpad, subpixel=10, device=torch.device("cpu")):
    """Interpolation matrix for sub-pixel upsampling."""
    xs = torch.arange(-lpad, lpad + 1, device=device)
    xs_up = torch.arange(-lpad, lpad + 0.001, 1.0 / subpixel, device=device)
    kernel0 = kernelD(xs, xs)
    kernel_up = kernelD(xs, xs_up)
    Kmat = torch.linalg.solve(kernel0, kernel_up)
    return Kmat, len(xs_up)


def calculate_nblocks(L, block_size):
    """Return effective block size and number of blocks for one dimension."""
    return (L, 1) if block_size >= L else (block_size, int(np.ceil(1.5 * L / block_size)))


def make_blocks(Ly, Lx, block_size, lpad=3, subpixel=10):
    """Compute overlapping blocks and interpolation helpers."""
    block_size = (int(block_size[0]), int(block_size[1]))
    block_size_y, ny = calculate_nblocks(L=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(L=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    ystart = np.linspace(0, Ly - block_size[0], ny).astype("int")
    xstart = np.linspace(0, Lx - block_size[1], nx).astype("int")
    yblock = [
        np.array([ystart[iy], ystart[iy] + block_size[0]])
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        np.array([xstart[ix], xstart[ix] + block_size[1]])
        for _ in range(ny)
        for ix in range(nx)
    ]

    NRsm = kernelD2(xs=torch.arange(nx), ys=torch.arange(ny)).T.numpy()
    Kmat, nup = mat_upsample(lpad=lpad, subpixel=subpixel)
    return yblock, xblock, [ny, nx], block_size, NRsm, Kmat, nup

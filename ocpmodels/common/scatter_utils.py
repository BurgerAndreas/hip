import torch


def segment_coo(src: torch.Tensor, index: torch.Tensor, dim_size=None, reduce: str = "sum"):
    if reduce not in {"sum", "add"}:
        raise ValueError(f"Unsupported reduction for HIP EquiformerV2: {reduce}")

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0

    out = src.new_zeros(dim_size)
    return out.scatter_add_(0, index, src)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum"):
    if reduce not in {"sum", "add"}:
        raise ValueError(f"Unsupported reduction for HIP EquiformerV2: {reduce}")

    return torch.stack(
        [src[start:end].sum() for start, end in zip(indptr[:-1], indptr[1:])]
    )

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from refiners.foundationals import dinov2


def get_dinov2_representations(
    model: dinov2.ViT,
    dataloader: DataLoader[torch.Tensor],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Get DINOV2 representations required to compute DINOv2-FD.

    Args:
        model: The DINOv2 model to use.
        dataloader: A dataloader that returns batches of preprocessed images.
        dtype: The dtype to use for the representations. Use float64 for good precision.

    Returns:
        A tensor with shape (batch, embedding_dim).
    """
    r: list[torch.Tensor] = []
    for batch in dataloader:
        assert isinstance(batch, torch.Tensor)
        batch_size = batch.shape[0]
        assert batch.shape == (batch_size, 3, 224, 224)
        batch = batch.to(model.device)

        with torch.no_grad():
            pred = model(batch)[:, 0]  # only keep class embeddings

        assert isinstance(pred, torch.Tensor)
        assert pred.shape == (batch_size, model.embedding_dim)

        r.append(pred.to(dtype))

    return torch.cat(r)


def frechet_distance(reps_a: torch.Tensor, reps_b: torch.Tensor) -> float:
    """
    Compute the Fréchet distance between two sets of representations.

    Args:
        reps_a: First set of representations (typically the reference). Shape (batch, N).
        reps_a: Second set of representations (typically the test set). Shape (batch, N).
    """
    assert reps_a.dim() == 2 and reps_b.dim() == 2, "representations must have shape (batch, N)"
    assert reps_a.shape[1] == reps_b.shape[1], "representations must have the same dimension"

    mean_a = torch.mean(reps_a, dim=0)
    cov_a = torch.cov(reps_a.t())
    mean_b = torch.mean(reps_b, dim=0)
    cov_b = torch.cov(reps_b.t())

    # The trace of the square root of a matrix is the sum of the square roots of its eigenvalues.
    trace = (torch.linalg.eigvals(cov_a.mm(cov_b)) ** 0.5).real.sum()  # type: ignore
    assert isinstance(trace, torch.Tensor)

    score = ((mean_a - mean_b) ** 2).sum() + cov_a.trace() + cov_b.trace() - 2 * trace
    return score.item()


class DinoDataset(Dataset[torch.Tensor]):
    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.image_paths = sorted(path.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = self.image_paths[i]
        img = Image.open(path)  # type: ignore
        return dinov2.preprocess(img)


def dinov2_frechet_distance(
    dataset_a: Dataset[torch.Tensor] | str | Path,
    dataset_b: Dataset[torch.Tensor] | str | Path,
    model: dinov2.ViT,
    batch_size: int = 64,
    dtype: torch.dtype = torch.float64,
) -> float:
    """
    Compute DINOv2-based Fréchet Distance between two datasets.

    There may be small discrepancies with other implementations due to the fact that DINOv2 in Refiners
    uses the new style interpolation whereas DINOv2-FD historically uses the legacy implementation
    (see https://github.com/facebookresearch/dinov2/pull/378)

    Args:
        dataset_a: First dataset (typically the reference). Can also be a path to a directory of PNG images.
            If a dataset is passed, it must preprocess the data using `dinov2.preprocess`.
        dataset_b: Second dataset (typically the test set). See `dataset_a` for details. Size can be different.
        model: The DINOv2 model to use.
        batch_size: The batch size to use.
        dtype: The dtype to use for the representations. Use float64 for good precision.
    """

    if not isinstance(dataset_a, Dataset):
        dataset_a = DinoDataset(dataset_a)
    if not isinstance(dataset_b, Dataset):
        dataset_b = DinoDataset(dataset_b)

    dataloader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=False)
    dataloader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=False)

    reps_a = get_dinov2_representations(model, dataloader_a, dtype)
    reps_b = get_dinov2_representations(model, dataloader_b, dtype)

    return frechet_distance(reps_a, reps_b)

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length

    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    inputs = np.stack([dataset[i : i + context_length] for i in start_indices])

    targets = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices])

    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    targets_tensor = torch.from_numpy(targets).long().to(device)

    return inputs_tensor, targets_tensor

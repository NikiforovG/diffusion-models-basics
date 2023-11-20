from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):  # type:ignore
    def __init__(
        self,
        sfilename: str,
        lfilename: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None,
        null_context: bool = False,
    ) -> None:
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")
        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape

    # Return the number of images in the dataset
    def __len__(self) -> int:
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self) -> tuple[list[int], list[int]]:
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape

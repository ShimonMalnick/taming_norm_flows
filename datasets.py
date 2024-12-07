from typing import Union, Optional, List, Callable
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torch
import torchvision.datasets as vision_datasets
import random


class EqualProbSampler(Sampler):
    """
    Sampler that enables equals probability to all samples in cases of len(dataset) < batch_size.
    """
    def __init__(self, data_source, batch_size):
        assert len(data_source) < batch_size, f"This sampler is used only when len(data_source) < batch_size but" \
                                              f" got batch_size={batch_size}, len(data_source)={len(data_source)}"
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.reps = batch_size // len(data_source)
        self.sequential = [i % len(self.data_source) for i in range(len(self.data_source) * self.reps)]

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        remainder_part = torch.randint(low=0, high=len(self.data_source),
                                       size=(self.batch_size - len(self.sequential), )).tolist()
        indices = self.sequential + remainder_part
        random.shuffle(indices)
        return iter(indices)


class CelebAPartial(vision_datasets.CelebA):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 exclude_images: List[str] = None,
                 exclude_identities: List[int] = None,
                 exclude_indices: List[int] = None,
                 include_only_images: List[str] = None,
                 include_only_identities: List[int] = None,
                 include_only_indices: List[int] = None):
        super().__init__(root, split=split, target_type=target_type, transform=transform,
                         target_transform=target_transform, download=download)
        """
        Added arguments on top of CelebA dataset from torchvision, in order to exclude certain files according to 
        given args. expecting (possible) args, or including only certain files:
        1. exclude_images: List[str] - list of file_names (with original name from celeba) to exclude
        2. exclude_identities: List[int] - list of identities to exclude (from possible {1, 2, ..., 10177} identities)
        3. exclude_indices: List[int] - list of indices to exclude
        4. include_only_images: List[str] - list of file_names (with original name from celeba) to include in dataset
        5. include_only_identities: List[int] - list of identities to include in dataset
        6. include_only_indices: List[int] - list of indices to include in dataset
        """
        assert not ((exclude_images or exclude_identities) and (include_only_images or include_only_identities)), \
            "excluding and including are mutually exclusive"
        all_exclude_indices = []
        if exclude_images is not None:
            all_exclude_indices += self.__images2idx(exclude_images)
        if exclude_identities is not None:
            all_exclude_indices += self.__identities2idx(exclude_identities)
        if exclude_indices is not None:
            all_exclude_indices += exclude_indices

        if all_exclude_indices:
            self.filename = [self.filename[i] for i in range(len(self.filename)) if i not in all_exclude_indices]
            index_tensor = torch.ones(len(self.attr), dtype=bool)
            index_tensor[all_exclude_indices] = False
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

        include_indices = []
        if include_only_images is not None:
            include_indices += self.__images2idx(include_only_images)
        if include_only_identities is not None:
            include_indices += self.__identities2idx(include_only_identities)
        if include_only_indices is not None:
            include_indices += include_only_indices

        if include_indices:
            self.filename = [self.filename[i] for i in include_indices]
            index_tensor = torch.zeros(len(self.attr), dtype=bool)
            index_tensor[include_indices] = True
            self.attr = self.attr[index_tensor]
            self.identity = self.identity[index_tensor]
            self.bbox = self.bbox[index_tensor]
            self.landmarks_align = self.landmarks_align[index_tensor]

    def __images2idx(self, images_names) -> List[int]:
        res = []
        for path in images_names:
            assert path in self.filename, f"{path} is not in the dataset"
            res.append(self.filename.index(path))
        return res

    def __identities2idx(self, identities) -> List[int]:
        assert 'identity' in self.target_type, "identity is not in the target_type"
        res = []
        for identity in identities:
            assert 1 <= identity <= 10177, f"{identity} is not in the dataset"
            cur_indices = ((self.identity == identity).nonzero(as_tuple=True)[0]).tolist()
            res += cur_indices
        return res


class PathsDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


class RandomDataset(Dataset):
    def __init__(self, img_size, num_images, transform=None, uniform=False, clip=False):
        self.img_size = img_size
        self.num_images = num_images
        self.transform = transform
        self.uniform = uniform
        self.clip = clip

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.uniform:
            img = torch.rand(*self.img_size)
        else:
            img = torch.randn(*self.img_size)
        if self.transform:
            img = self.transform(img)
        if self.clip:
            img = torch.clamp(img, -0.5, 0.5)

        return img, 0

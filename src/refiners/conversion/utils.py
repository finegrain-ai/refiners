import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from warnings import warn

import requests
import torch
from huggingface_hub import (  # pyright: ignore[reportMissingTypeStubs]
    HfFileMetadata,
    get_hf_file_metadata,  # pyright: ignore[reportUnknownVariableType]
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    hf_hub_url,
)
from tqdm import tqdm

from refiners.fluxion.utils import load_from_safetensors, load_tensors, save_to_safetensors

AnyDict = dict[str, Any]
TensorDict = dict[str, torch.Tensor]


def download_file_url(url: str, destination: Path) -> None:
    """Download a file from a url to a destination."""
    logging.debug(f"Downloading {url} to {destination}")

    # get the size of the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    # create a progress bar
    bar = tqdm(
        desc=destination.name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    )

    # download the file
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 1000):
                size = f.write(chunk)
                bar.update(size)
        bar.close()


class Hub:
    """A class representing a weight on the Hub.

    Note:
        The Hub denotes a directory on the local machine where the weights are stored.
        The Hub may also correspond to a remote repository on the Hugging Face Hub.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        expected_sha256: str,
        revision: str = "main",
        download_url: str | None = None,
    ) -> None:
        """Initialize the HubPath.

        Args:
            repo_id: The repository identifier on the hub.
            filename: The filename of the file in the repository.
            revision: The revision of the file on the hf hub.
            expected_sha256: The sha256 hash of the file.
            download_url: The url to download the file from, if not from the huggingface hub.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self.expected_sha256 = expected_sha256.lower()
        self.override_download_url = download_url

    @staticmethod
    def hub_location():
        """Return the path to the local hub root directory."""
        return Path(os.getenv("REFINERS_HUB_PATH", "tests/weights"))

    @property
    def hf_url(self) -> str:
        """Return the url to the file on the hf hub."""
        assert self.override_download_url is None, f"{self.repo_id}/{self.filename} is not available on the hub"
        return hf_hub_url(
            repo_id=self.repo_id,
            filename=self.filename,
            revision=self.revision,
        )

    @property
    def hf_cache_path(self) -> Path:
        """Download the file from the hf hub and return its path in the local hf cache."""
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                revision=self.revision,
            ),
        )

    @property
    def hf_metadata(self) -> HfFileMetadata:
        """Return the metadata of the file on the hf hub."""
        return get_hf_file_metadata(self.hf_url)

    @property
    def hf_sha256_hash(self) -> str:
        """Return the sha256 hash of the file on the hf hub."""
        remote_hash = self.hf_metadata.etag
        assert remote_hash is not None
        assert len(remote_hash) == 64
        return remote_hash.lower()

    @property
    def local_path(self) -> Path:
        """Return the path to the file in the local hub."""
        return self.hub_location() / self.repo_id / self.filename

    @property
    def local_hash(self) -> str:
        """Return the sha256 hash of the file in the local hub."""
        assert self.local_path.is_file(), f"{self.local_path} does not exist"
        # TODO: use https://docs.python.org/3/library/hashlib.html#hashlib.file_digest when support python >= 3.11
        return sha256(self.local_path.read_bytes()).hexdigest().lower()

    def check_local_hash(self) -> bool:
        """Check if the sha256 hash of the file in the local hub is correct."""
        if self.expected_sha256 != self.local_hash:
            logging.warning(f"{self.local_path} local sha256 mismatch, {self.local_hash} != {self.expected_sha256}")
            return False
        else:
            logging.debug(f"{self.local_path} local sha256 is correct ({self.local_hash})")
            return True

    def check_remote_hash(self) -> bool:
        """Check if the sha256 hash of the file on the hf hub is correct."""
        if self.expected_sha256 != self.hf_sha256_hash:
            logging.warning(
                f"{self.local_path} remote sha256 mismatch, {self.hf_sha256_hash} != {self.expected_sha256}"
            )
            return False
        else:
            logging.debug(f"{self.local_path} remote sha256 is correct ({self.hf_sha256_hash})")
            return True

    def download(self) -> None:
        """Download the file from the hf hub or from the override download url."""
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.local_path.is_file():
            logging.warning(f"{self.local_path} already exists")
        elif self.override_download_url is not None:
            download_file_url(url=self.override_download_url, destination=self.local_path)
        else:
            # TODO: pas assez de message de log quand local_path existe pas et que ça vient du hf cache
            self.local_path.symlink_to(self.hf_cache_path)
        assert self.check_local_hash()


class WeightRecipe:
    """A class representing a recipe to convert weights from one format to another."""

    def __init__(
        self,
        key_map: dict[str, str],
        key_prefix: str = "",
        key_aliases: dict[str, str] = {},
        tensor_reshapes: dict[str, tuple[int, ...]] = {},
    ):
        """Initialize the weight recipe.

        Args:
            key_map: A dictionary mapping the keys of the original state dict to the converted state dict.
            key_prefix: A prefix to remove from the keys of the original state dict.
            key_aliases: A dictionary mapping the keys of the original state dict to their aliases.
            tensor_reshapes: A dictionary mapping the keys of the original state dict to their new shapes.
        """
        self.key_prefix = key_prefix
        self.key_map = key_map
        self.key_aliases = key_aliases
        self.tensor_reshapes = tensor_reshapes

    @staticmethod
    def flatten_state_dict(state_dict: AnyDict, sep: str = ".") -> AnyDict:
        """Flattens a nested dictionary into a dictionary with dot-separated keys.

        Args:
            state_dict: A nested dictionary.
            sep: The separator to use between keys when flattening.
        """

        def _flatten(current_dict: AnyDict, parent_key: str = "") -> AnyDict:
            items: AnyDict = {}
            for k, v in current_dict.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(_flatten(cast(AnyDict, v), new_key))
                else:
                    items[new_key] = v
            return items

        return _flatten(state_dict)

    def name_map_keys(self, state_dict: TensorDict) -> TensorDict:
        """Map the keys of the state dict according to the name map."""
        new_state_dict: TensorDict = {}
        for key, value in state_dict.items():
            # check for .weight or .bias suffixes
            suffix = next(s for s in (".weight", ".bias", "") if key.endswith(s))
            key = key.removesuffix(suffix)

            # remove key_prefix
            key = key.removeprefix(self.key_prefix)

            # check for key aliases
            source_key = self.key_aliases.get(key, key)

            # get target_key from key_map
            target_key = self.key_map.get(source_key)
            if target_key is None:
                continue  # ignore key if it doesn't exist in the key_map

            # add value to new_state_dict with the mapped key
            new_state_dict[target_key + suffix] = value

        return new_state_dict

    def reshape_tensors(self, state_dict: TensorDict) -> TensorDict:
        """Reshape tensors in the state dict according to tensor_reshapes."""
        new_state_dict = state_dict.copy()
        for key, value in state_dict.items():
            if key in self.tensor_reshapes:
                new_shape = self.tensor_reshapes[key]
                new_state_dict[key] = value.reshape(new_shape)
        return new_state_dict

    def translate_keys(self, state_dict: AnyDict, flatten_state_dict: bool = True) -> TensorDict:
        """Translate the keys of a state dict."""
        if flatten_state_dict:
            state_dict = self.flatten_state_dict(state_dict)

        state_dict = self.name_map_keys(state_dict)
        state_dict = self.reshape_tensors(state_dict)

        return state_dict


class Conversion:
    """Structure to link original and converted weights on the Hub."""

    def __init__(
        self,
        original: Hub,
        converted: Hub,
        recipe: WeightRecipe,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the weight structure.

        Args:
            original: A Hub object representing the original weight.
            converted: A Hub object representing the converted weight.
            recipe: A WeightRecipe object used to convert from the original to the converted weight.
            dtype: The dtype of the converted weights.
        """
        self.original = original
        self.converted = converted
        self.recipe = recipe
        self.dtype = dtype

    @staticmethod
    def load_state_dict(path: Path) -> AnyDict:
        """Load a state dict from a file."""
        if path.suffix == ".safetensors" or path.suffix == ".sft":
            return load_from_safetensors(path)
        else:
            return load_tensors(path)

    @staticmethod
    def filter_tensors_state_dict(state_dict: AnyDict) -> TensorDict:
        """Filter out non-tensor values and tensors with NaNs from a state dict."""
        new_state_dict: TensorDict = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                warn(f"Value for key {key} is not a tensor, filtering")
                continue
            if torch.isnan(value).sum().item() > 0:
                warn(f"Found NaNs in {key}, filtering")
                continue
            new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def change_dtype(state_dict: TensorDict, dtype: torch.dtype) -> TensorDict:
        """Change the dtype of the tensors in a state dict."""
        return {k: v.to(dtype=dtype) for k, v in state_dict.items()}

    def convert(self) -> None:
        """Convert the weights from the original to the converted weights.

        Note: The original weights are automatically downloaded if they are not already present.
        """
        logging.info(
            f"Converting {self.original.repo_id}/{self.original.filename} "
            f"to {self.converted.repo_id}/{self.converted.filename}"
        )

        # check if the converted file already exists
        if self.converted.local_path.is_file():
            logging.warning(f"{self.converted.local_path} already exists")
            if self.converted.check_local_hash():
                try:
                    assert self.converted.check_remote_hash()
                except requests.exceptions.HTTPError:
                    logging.error(f"{self.converted.local_path} couldn't verify remote hash")
                return

        # get the original state_dict
        self.original.download()

        # load the original state_dict
        original_state_dict = self.load_state_dict(self.original.local_path)
        original_state_dict = self.filter_tensors_state_dict(original_state_dict)

        # convert the state_dict
        converted_state_dict = self.recipe.translate_keys(original_state_dict)
        converted_state_dict = self.change_dtype(converted_state_dict, self.dtype)

        # save the converted state_dict
        self.converted.local_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_safetensors(self.converted.local_path, converted_state_dict)

        # check the converted state_dict
        assert self.converted.check_local_hash()
        try:
            assert self.converted.check_remote_hash()
        except requests.exceptions.HTTPError:
            logging.warning(f"{self.converted.local_path} couldn't verify remote hash")

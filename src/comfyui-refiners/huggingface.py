from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore


class HfHubDownload:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "repo_id": ("STRING", {}),
            },
            "optional": {
                "filename": ("STRING", {}),
                "revision": (
                    "STRING",
                    {
                        "default": "main",
                    },
                ),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("path",)
    DESCRIPTION = "Download file(s) from the HuggingFace Hub."
    CATEGORY = "Refiners/HuggingFace"
    FUNCTION = "download"

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
    ) -> tuple[Path]:
        """Download file(s) from the HuggingFace Hub.

        Args:
            repo_id: The HuggingFace repository ID.
            filename: The filename to download, if empty, the entire repository will be downloaded.
            revision: The git revision to download.

        Returns:
            The path to the downloaded file(s).
        """
        if filename == "":
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
            )
        else:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        return (Path(path),)


NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "HfHubDownload": HfHubDownload,
}

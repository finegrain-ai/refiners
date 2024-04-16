from typing import Any

import wandb
from git import Repo
from loguru import logger

from refiners.training_utils.callback import Callback, CallbackConfig
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.trainer import Trainer

AnyTrainer = Trainer[BaseConfig, Any]


class ForceCommitConfig(CallbackConfig):
    """Configuration of the ForceCommit callback.

    Attributes:
        check_changed: Whether to check if there are modified files.
        check_untracked: Whether to check if there are untracked files.
        upload_wandb_patch: Whether to upload the patch of the changes.
        search_parent_directories: Whether to search parent directories for the git repository.
        exclusions: List of files to exclude from the checks.
    """

    check_changed: bool = True
    check_untracked: bool = False
    upload_wandb_patch: bool = False
    search_parent_directories: bool = False
    exclusions: list[str] = []


class ForceCommit(Callback[AnyTrainer]):
    """Callback to force user to commit or stash changes before running the training.

    This callback assumes that the training is run from a git repository.
    """

    def __init__(self, config: ForceCommitConfig) -> None:
        """Initialize the callback.

        Args:
            config: Configuration of the callback.
        """
        self.check_changed = config.check_changed
        self.check_untracked = config.check_untracked
        self.upload_wandb_patch = config.upload_wandb_patch
        self.search_parent_directories = config.search_parent_directories
        self.exclusions = config.exclusions

    def on_init_begin(self, trainer: AnyTrainer) -> None:
        # get git repo and diff list
        repo = Repo(search_parent_directories=self.search_parent_directories)
        logger.info(f"Git repository: {repo.working_dir}")
        logger.info(f"Git branch: {repo.active_branch}")
        logger.info(f"Git commit: {repo.head.commit.hexsha}")
        diffs = repo.index.diff(other=None, create_patch=True)  # type: ignore

        # get list of modified files
        modified_files: list[str] = [item.a_path for item in diffs]  # type: ignore
        modified_files: set[str] = set(modified_files) - set(self.exclusions)
        logger.info(f"Modified files: {modified_files}")
        if self.check_changed and modified_files:
            raise RuntimeError(
                "There are modified files. Please commit or stash them before running the training.",
            )

        # get list of untracked files
        untracked_files = repo.untracked_files
        untracked_files = set(untracked_files) - set(self.exclusions)
        logger.info(f"Untracked files: {untracked_files}")
        if self.check_untracked and untracked_files:
            raise RuntimeError(
                "There are untracked files. Please add them to the repository before running the training.",
            )

        # create patch
        if self.upload_wandb_patch:
            patch = str(repo.git.diff()).replace("\n", "<br>")
            artifact = wandb.Artifact(name="git", type="metadata")
            artifact.add(name="patch", obj=wandb.Html(patch))
            wandb.log_artifact(artifact)  # type: ignore

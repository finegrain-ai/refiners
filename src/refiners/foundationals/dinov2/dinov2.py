import torch

from refiners.foundationals.dinov2.vit import ViT

# TODO: add preprocessing logic like
# https://github.com/facebookresearch/dinov2/blob/2302b6b/dinov2/data/transforms.py#L77


class DINOv2_small(ViT):
    """DINOv2 small model.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    for more details.

    Attributes:
        embedding_dim (int): 384
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 12
        num_heads (int): 6
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 small model.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=384,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=6,
            device=device,
            dtype=dtype,
        )


class DINOv2_base(ViT):
    """DINOv2 base model.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    for more details.

    Attributes:
        embedding_dim (int): 768
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 12
        num_heads (int): 12
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 base model.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=768,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=12,
            device=device,
            dtype=dtype,
        )


class DINOv2_large(ViT):
    """DINOv2 large model.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    for more details.

    Attributes:
        embedding_dim (int): 1024
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 24
        num_heads (int): 16
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 large model.

        Args:
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=1024,
            patch_size=14,
            image_size=518,
            num_layers=24,
            num_heads=16,
            device=device,
            dtype=dtype,
        )


# TODO: implement SwiGLU layer
# class DINOv2_giant2(ViT):
#     def __init__(
#         self,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> None:
#         super().__init__(
#             embedding_dim=1536,
#             patch_size=14,
#             image_size=518,
#             num_layers=40,
#             num_heads=24,
#             device=device,
#             dtype=dtype,
#         )


class DINOv2_small_reg(ViT):
    """DINOv2 small model with register.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    and [[arXiv:2309.16588] Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
    for more details.

    Attributes:
        embedding_dim (int): 384
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 12
        num_heads (int): 6
        num_registers (int): 4
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 small model with register.

        Args:
            device (torch.device | str | None): The PyTorch device to use.
            dtype (torch.dtype | None): The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=384,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=6,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


class DINOv2_base_reg(ViT):
    """DINOv2 base model with register.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    and [[arXiv:2309.16588] Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
    for more details.

    Attributes:
        embedding_dim (int): 768
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 12
        num_heads (int): 12
        num_registers (int): 4
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 base model with register.

        Args:
            device (torch.device | str | None): The PyTorch device to use.
            dtype (torch.dtype | None): The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=768,
            patch_size=14,
            image_size=518,
            num_layers=12,
            num_heads=12,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


class DINOv2_large_reg(ViT):
    """DINOv2 large model with register.

    See [[arXiv:2304.07193] DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
    and [[arXiv:2309.16588] Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
    for more details.

    Attributes:
        embedding_dim (int): 1024
        patch_size (int): 14
        image_size (int): 518
        num_layers (int): 24
        num_heads (int): 16
        num_registers (int): 4
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize DINOv2 large model with register.

        Args:
            device (torch.device | str | None): The PyTorch device to use.
            dtype (torch.dtype | None): The PyTorch data type to use.
        """
        super().__init__(
            embedding_dim=1024,
            patch_size=14,
            image_size=518,
            num_layers=24,
            num_heads=16,
            num_registers=4,
            device=device,
            dtype=dtype,
        )


# TODO: implement SwiGLU layer
# class DINOv2_giant2_reg(ViT):
#     def __init__(
#         self,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> None:
#         super().__init__(
#             embedding_dim=1536,
#             patch_size=14,
#             image_size=518,
#             num_layers=40,
#             num_heads=24,
#             num_registers=4,
#             device=device,
#             dtype=dtype,
#         )

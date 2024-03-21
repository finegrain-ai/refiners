import torch
from torch import device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters import Adapter
from refiners.fluxion.context import Contexts
from refiners.foundationals.segment_anything.image_encoder import SAMViT, TransformerLayer
from refiners.foundationals.segment_anything.mask_decoder import (
    MaskDecoderTokens,
    MaskPrediction,
    Predictions,
)
from refiners.foundationals.segment_anything.model import SegmentAnything


class CompressViTFeat(fl.Chain):
    def __init__(
        self,
        transformer_dim: int = 256,
        vit_dim: int = 1024,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext(context="hq_sam", key="early_vit_embedding"),
            fl.Permute(0, 3, 1, 2),
            fl.ConvTranspose2d(
                in_channels=vit_dim,
                out_channels=transformer_dim,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=transformer_dim, device=device, dtype=dtype),
            fl.GeLU(),
            fl.ConvTranspose2d(
                in_channels=transformer_dim,
                out_channels=transformer_dim // 8,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
        )


class EmbeddingEncoder(fl.Chain):
    def __init__(
        self,
        transformer_dim: int = 256,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext(context="mask_decoder", key="image_embedding"),
            fl.ConvTranspose2d(
                in_channels=transformer_dim,
                out_channels=transformer_dim // 4,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
            fl.LayerNorm2d(channels=transformer_dim // 4, device=device, dtype=dtype),
            fl.GeLU(),
            fl.ConvTranspose2d(
                in_channels=transformer_dim // 4,
                out_channels=transformer_dim // 8,
                kernel_size=2,
                stride=2,
                device=device,
                dtype=dtype,
            ),
        )


class HQFeatures(fl.Sum):
    def __init__(
        self,
        vit_dim: int = 1024,
        transformer_dim: int = 256,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            EmbeddingEncoder(transformer_dim, device, dtype),
            CompressViTFeat(transformer_dim, vit_dim, device, dtype),
        )


class EmbeddingMaskfeature(fl.Chain):
    def __init__(
        self,
        transformer_dim: int = 256,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.UseContext(context="mask_decoder", key="upscaled_dense_embedding"),
            fl.Reshape(-1, transformer_dim, transformer_dim),
            fl.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1, device=device, dtype=dtype),
            fl.LayerNorm2d(transformer_dim // 4, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1, device=device, dtype=dtype),
        )


class DenseEmbeddingUpscalingHQ(fl.Sum):
    def __init__(
        self,
        vit_dim: int = 1024,
        transformer_dim: int = 256,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            EmbeddingMaskfeature(transformer_dim, device, dtype),
            HQFeatures(vit_dim, transformer_dim, device, dtype),
        )


class HQTokenMLP(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 3,
        target_num_mask_tokens: int = 5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            fl.Slicing(dim=1, start=target_num_mask_tokens, end=target_num_mask_tokens + 1),  # HQ token
            fl.MultiLinear(
                input_dim=embedding_dim,
                output_dim=embedding_dim // 8,
                inner_dim=embedding_dim,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            ),
        )


class HQSAMMaskPrediction(fl.Matmul):
    def __init__(
        self,
        embedding_dim: int,
        vit_dim: int = 1024,
        target_num_mask_tokens: int = 5,
        num_layers: int = 3,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        super().__init__(
            HQTokenMLP(
                embedding_dim,
                num_layers=num_layers,
                target_num_mask_tokens=target_num_mask_tokens,
                device=device,
                dtype=dtype,
            ),
            fl.Chain(
                DenseEmbeddingUpscalingHQ(vit_dim=vit_dim, transformer_dim=256, device=device, dtype=dtype),
                fl.Flatten(start_dim=2),
            ),
        )


class MaskPredictionAdapter(fl.Concatenate, Adapter[MaskPrediction]):
    def __init__(
        self,
        target: MaskPrediction,
        vit_dim: int = 1024,
        target_num_mask_tokens: int = 5,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(
                target,
                fl.Chain(
                    HQSAMMaskPrediction(
                        embedding_dim=target.embedding_dim,
                        vit_dim=vit_dim,
                        target_num_mask_tokens=target_num_mask_tokens,
                        num_layers=3,
                        device=device,
                        dtype=dtype,
                    ),
                    fl.Reshape(-1, target.embedding_dim, target.embedding_dim),
                ),
                dim=1,
            )

    @property
    def hq_sam_mask_prediction(self) -> HQSAMMaskPrediction:
        return self.ensure_find(HQSAMMaskPrediction)


class MaskDecoderTokensExtender(fl.Concatenate, Adapter[MaskDecoderTokens]):
    """
    Add a new weight to the MaskDecoderTokens to store the new HQ token.
    """

    def __init__(
        self,
        target: MaskDecoderTokens,
    ) -> None:
        self._hq_token = [fl.Parameter(1, target.embedding_dim, device=target.device, dtype=target.dtype)]
        with self.setup_adapter(target):
            super().__init__(
                target,
                fl.Chain(
                    fl.UseContext(context="mask_decoder", key="image_embedding"),  # use Context to infer batch size
                    self.hq_token,
                ),
                dim=1,
            )

    @property
    def regular_tokens(self) -> fl.Parameter:
        return self.target.ensure_find(fl.Parameter)

    @property
    def hq_token(self) -> fl.Parameter:
        return self._hq_token[0]


class SAMViTAdapter(fl.Chain, Adapter[SAMViT]):
    """
    Add a context to the image encoder transformer to store its early ViT embedding
    (first intermediate embedding of the ViT).
    """

    def __init__(self, target: SAMViT) -> None:
        with self.setup_adapter(target):
            super().__init__(target)
        target_transformer_layer = self._find_target_transformer_layer()
        assert target_transformer_layer is not None
        self._transformer_layer = [target_transformer_layer]
        self._set_early_vit_embedding_context = [fl.SetContext("hq_sam", "early_vit_embedding")]

    @property
    def target_transformer_layer(self) -> TransformerLayer:
        return self._transformer_layer[0]

    @property
    def set_early_vit_embedding_context(self) -> fl.SetContext:
        return self._set_early_vit_embedding_context[0]

    def _find_target_transformer_layer(self) -> TransformerLayer | None:
        for transformer_layer in self.target.layers(TransformerLayer):
            if transformer_layer.window_size is None:
                return transformer_layer
        return None

    def inject(self: "SAMViTAdapter", parent: fl.Chain | None = None) -> "SAMViTAdapter":
        self.target_transformer_layer.append(self.set_early_vit_embedding_context)
        return super().inject(parent)

    def eject(self) -> None:
        self.target_transformer_layer.remove(self.set_early_vit_embedding_context)
        super().eject()


class PredictionsPostProc(fl.Module):
    def __init__(self, hq_mask_only: bool = False) -> None:
        super().__init__()
        self.hq_mask_only = hq_mask_only

    def forward(
        self, masks_predictions: torch.Tensor, iou_predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hq_sam_mask = masks_predictions[:, -1:, ...]

        # The official implementation of HQ-SAM has two outputs modes:
        # 1. HQ mask only
        # 2. HQ mask + base SAM mask, using HQ as a correction to the base SAM mask
        # Details can be found in the paper: https://arxiv.org/abs/2306.01567 (section 3.3)
        # Heuristics are provided by the authors here: https://github.com/SysCV/sam-hq/blob/3224888/demo/demo_hqsam_pip_example.py#L73-L75
        if self.hq_mask_only:
            return (hq_sam_mask, iou_predictions)

        base_sam_masks = masks_predictions[:, :-1, ...]
        assert base_sam_masks.shape[1] == 1
        return (hq_sam_mask + base_sam_masks, iou_predictions)


class HQSAMAdapter(fl.Chain, Adapter[SegmentAnything]):
    """Adapter for SAM introducing HQ features.

    See [[arXiv:2306.01567] Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) for details.
    """

    def init_context(self) -> Contexts:
        return {"hq_sam": {"early_vit_embedding": None}}

    def __init__(
        self,
        target: SegmentAnything,
        hq_mask_only: bool = False,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.vit_embedding_dim = target.image_encoder.embedding_dim
        self.target_num_mask_tokens = target.mask_decoder.num_multimask_outputs + 2

        with self.setup_adapter(target):
            super().__init__(target)

        if target.mask_decoder.multimask_output:
            raise NotImplementedError("Multi-mask mode is not supported in HQSAMAdapter.")

        mask_prediction = target.mask_decoder.ensure_find(MaskPrediction)

        self._mask_prediction_adapter = [
            MaskPredictionAdapter(
                mask_prediction, self.vit_embedding_dim, self.target_num_mask_tokens, target.device, target.dtype
            )
        ]
        self._image_encoder_adapter = [SAMViTAdapter(target.image_encoder)]
        self._predictions_post_proc = [PredictionsPostProc(hq_mask_only)]

        mask_decoder_tokens = target.mask_decoder.ensure_find(MaskDecoderTokens)
        self._mask_decoder_tokens_extender = [MaskDecoderTokensExtender(mask_decoder_tokens)]

        if weights is not None:
            hq_token_prefix = "MaskDecoderTokensExtender.hq_token."
            hq_token_state_dict: dict[str, torch.Tensor] = {
                k.removeprefix(hq_token_prefix): v for k, v in weights.items() if k.startswith(hq_token_prefix)
            }
            self.mask_decoder_tokens_extender.hq_token.load_state_dict(hq_token_state_dict)

            mask_pred_prefix = "Chain.HQSAMMaskPrediction."
            mask_pred_state_dict: dict[str, torch.Tensor] = {
                k.removeprefix(mask_pred_prefix): v for k, v in weights.items() if k.startswith(mask_pred_prefix)
            }
            self.mask_prediction_adapter.hq_sam_mask_prediction.load_state_dict(mask_pred_state_dict)

        self.to(device=target.device, dtype=target.dtype)

    @property
    def mask_decoder_tokens_extender(self) -> MaskDecoderTokensExtender:
        return self._mask_decoder_tokens_extender[0]

    @property
    def mask_prediction_adapter(self) -> MaskPredictionAdapter:
        return self._mask_prediction_adapter[0]

    @property
    def image_encoder_adapter(self) -> SAMViTAdapter:
        return self._image_encoder_adapter[0]

    @property
    def predictions_post_proc(self) -> PredictionsPostProc:
        return self._predictions_post_proc[0]

    @property
    def hq_mask_only(self) -> bool:
        return self.predictions_post_proc.hq_mask_only

    @hq_mask_only.setter
    def hq_mask_only(self, value: bool) -> None:
        self.predictions_post_proc.hq_mask_only = value

    def inject(self: "HQSAMAdapter", parent: fl.Chain | None = None) -> "HQSAMAdapter":
        self.mask_decoder_tokens_extender.inject()
        self.mask_prediction_adapter.inject()
        self.image_encoder_adapter.inject()
        self.target.mask_decoder.insert_after_type(Predictions, self.predictions_post_proc)
        return super().inject(parent)

    def eject(self) -> None:
        self.mask_decoder_tokens_extender.eject()
        self.mask_prediction_adapter.eject()
        self.image_encoder_adapter.eject()
        self.target.mask_decoder.remove(self.predictions_post_proc)
        super().eject()

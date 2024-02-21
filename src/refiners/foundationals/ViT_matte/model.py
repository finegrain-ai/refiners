from torch import device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.foundationals.vit_matte.decoder import DetailCapture
from refiners.foundationals.vit_matte.vit_backbone import ViT


class ViTMatteH(fl.Chain):
    def __init__(self, device: Device | str | None = None, dtype: DType | None = None):
        super().__init__(
            fl.SetContext(context="detail_capture", key="images"),
            ViT(
                embedding_dim=384,
                num_layers=12,
                num_heads=6,
                global_attention_indices=(2, 5, 8, 11),  # 2, 5, 8 11 for global attention
                device=device,
                dtype=dtype,
            ),
            DetailCapture(),
        )

    def init_context(self):
        return {"detail_capture": {"images": None}}

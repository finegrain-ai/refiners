
import torch
import refiners.fluxion.layers as fl
from refiners.foundationals.clip import CLIPImageEncoderWithoutProj


class LlavaMultiModalProjector(fl.Chain):
    "LLava multimodal vision projector"

    def __init__(self,
                 in_dim: int,
                 feedforward_dim: int,
                 out_dim: int,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None,
                 ) -> None:
        super().__init__(
            fl.Linear(in_features=in_dim,
                      out_features=feedforward_dim,
                      device=device,
                      dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=feedforward_dim,
                      out_features=out_dim,
                      device=device,
                      dtype=dtype),
        )


class LlavaMetaModel(fl.Chain):
    "Inspired from the the HuggingFace 'llava-hf/llava-1.5-7b-hf' model."

    def __init__(self,
                 image_size: int = 336,
                 embedding_dim: int = 1024,
                 patch_size: int = 32,
                 num_layers: int = 24,
                 num_attention_heads: int = 12,
                 feedforward_dim: int = 3072,
                 mm_feedforward_dim: int = 4096,
                 layer_norm_eps: float = 1e-5,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None,):

        self.vision_tower = CLIPImageEncoderWithoutProj(image_size=image_size,
                                                        embedding_dim=embedding_dim,
                                                        patch_size=patch_size,
                                                        num_layers=num_layers,
                                                        num_attention_heads=num_attention_heads,
                                                        feedforward_dim=feedforward_dim,
                                                        layer_norm_eps=layer_norm_eps,
                                                        device=device,
                                                        dtype=dtype)
        self.mm_projector = LlavaMultiModalProjector(in_dim=embedding_dim,
                                                     feedforward_dim=mm_feedforward_dim,
                                                     out_dim=embedding_dim)

    def prepare_inputs_labels_for_multimodal(self, **kwargs):
        raise NotImplementedError("Not implemented yet")

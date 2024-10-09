import torch

from refiners.conversion.utils import Conversion, Hub, WeightRecipe

recipe = WeightRecipe(
    key_map={
        "mask_decoder.compress_vit_feat.0": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.CompressViTFeat.ConvTranspose2d_1",
        "mask_decoder.embedding_encoder.0": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.EmbeddingEncoder.ConvTranspose2d_1",
        "mask_decoder.embedding_maskfeature.0": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.EmbeddingMaskfeature.Conv2d_1",
        "mask_decoder.compress_vit_feat.1": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.CompressViTFeat.LayerNorm2d",
        "mask_decoder.embedding_encoder.1": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.EmbeddingEncoder.LayerNorm2d",
        "mask_decoder.embedding_maskfeature.1": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.EmbeddingMaskfeature.LayerNorm2d",
        "mask_decoder.compress_vit_feat.3": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.CompressViTFeat.ConvTranspose2d_2",
        "mask_decoder.embedding_encoder.3": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.HQFeatures.EmbeddingEncoder.ConvTranspose2d_2",
        "mask_decoder.embedding_maskfeature.3": "Chain.HQSAMMaskPrediction.Chain.DenseEmbeddingUpscalingHQ.EmbeddingMaskfeature.Conv2d_2",
        "mask_decoder.hf_mlp.layers.0": "Chain.HQSAMMaskPrediction.HQTokenMLP.MultiLinear.Linear_1",
        "mask_decoder.hf_mlp.layers.1": "Chain.HQSAMMaskPrediction.HQTokenMLP.MultiLinear.Linear_2",
        "mask_decoder.hf_mlp.layers.2": "Chain.HQSAMMaskPrediction.HQTokenMLP.MultiLinear.Linear_3",
        "mask_decoder.hf_token": "MaskDecoderTokensExtender.hq_token",
    },
)

vit_h = Conversion(
    original=Hub(
        repo_id="lkeab/hq-sam",
        filename="sam_hq_vit_h.pth",
        expected_sha256="a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35",
    ),
    converted=Hub(
        repo_id="refiners/sam.hq.vit_h",
        filename="model.safetensors",
        expected_sha256="017630c780ff67673d71e91beaec8804f8b5ae3a9ea607456b4504562f96cc2f",
    ),
    recipe=recipe,
    dtype=torch.float32,
)

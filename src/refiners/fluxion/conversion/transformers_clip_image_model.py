import torch
from torch import nn

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.clip.image_encoder import CLIPImageEncoder


def setup_converter(source_path, subfolder, threshold) -> ModelConverter:
    from transformers import CLIPVisionModelWithProjection  # type: ignore

    source: nn.Module = CLIPVisionModelWithProjection.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=source_path, subfolder=subfolder
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    architecture: str = source.config.architectures[0]  # type: ignore
    num_channels: int = source.config.num_channels  # type: ignore
    embedding_dim: int = source.config.hidden_size  # type: ignore
    image_size: int = source.config.image_size  # type: ignore
    patch_size: int = source.config.patch_size  # type: ignore
    output_dim: int = source.config.projection_dim  # type: ignore
    num_layers: int = source.config.num_hidden_layers  # type: ignore
    num_attention_heads: int = source.config.num_attention_heads  # type: ignore
    feedforward_dim: int = source.config.intermediate_size  # type: ignore
    activation: str = source.config.hidden_act  # type: ignore
    layer_norm_eps: float = source.config.layer_norm_eps  # type: ignore

    assert architecture == "CLIPVisionModelWithProjection", f"Unsupported architecture: {architecture}"
    assert num_channels == 3, f"Expected 3 input channels, got {num_channels}"
    assert activation == "gelu", f"Unsupported activation: {activation}"

    target = CLIPImageEncoder(
        image_size=image_size,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        patch_size=patch_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        layer_norm_eps=layer_norm_eps,
    )

    x = torch.randn(1, 3, image_size, image_size)

    converter = ModelConverter(source_model=source, target_model=target, verbose=True)

    # Custom conversion logic since the class embedding (fl.Parameter layer) is not supported out-of-the-box by the
    # converter
    mapping = converter.map_state_dicts((x,))
    assert mapping is not None

    source_state_dict = source.state_dict()
    target_state_dict = target.state_dict()

    # Remove the class embedding from state dict since it was not mapped by the model converter
    class_embedding = target.ensure_find(fl.Parameter)
    class_embedding_key = next(
        (n for n, p in target.named_parameters() if id(p) == id(class_embedding.parameter)), None
    )
    assert class_embedding_key is not None
    assert class_embedding_key in target_state_dict
    del target_state_dict[class_embedding_key]

    converted_state_dict = converter._convert_state_dict(  # type: ignore[reportPrivateUsage]
        source_state_dict=source_state_dict, target_state_dict=target_state_dict, state_dict_mapping=mapping
    )
    target.load_state_dict(state_dict=converted_state_dict, strict=False)

    # Ad hoc post-conversion steps
    class_embedding.parameter = torch.nn.Parameter(
        source.vision_model.embeddings.class_embedding.clone()
    )  # type: ignore

    assert converter.compare_models((x,), threshold=threshold)

    return converter

from torch import nn

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer


def setup_transformers_clip_text_converter(source_path, subfolder, verbose=False) -> ModelConverter:
    from transformers import CLIPTextModelWithProjection  # type: ignore

    source: nn.Module = CLIPTextModelWithProjection.from_pretrained(  # type: ignore
        pretrained_model_name_or_path=source_path, subfolder=subfolder
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"
    architecture: str = source.config.architectures[0]  # type: ignore
    embedding_dim: int = source.config.hidden_size  # type: ignore
    projection_dim: int = source.config.projection_dim  # type: ignore
    num_layers: int = source.config.num_hidden_layers  # type: ignore
    num_attention_heads: int = source.config.num_attention_heads  # type: ignore
    feed_forward_dim: int = source.config.intermediate_size  # type: ignore
    use_quick_gelu: bool = source.config.hidden_act == "quick_gelu"  # type: ignore
    target = CLIPTextEncoder(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feed_forward_dim,
        use_quick_gelu=use_quick_gelu,
    )
    match architecture:
        case "CLIPTextModel":
            source.text_projection = fl.Identity()
        case "CLIPTextModelWithProjection":
            target.append(module=fl.Linear(in_features=embedding_dim, out_features=projection_dim, bias=False))
        case _:
            raise RuntimeError(f"Unsupported architecture: {architecture}")
    text = "What a nice cat you have there!"
    tokenizer = target.ensure_find(CLIPTokenizer)
    tokens = tokenizer(text)
    converter = ModelConverter(source_model=source, target_model=target, skip_output_check=True, verbose=verbose)
    if not converter.run(source_args=(tokens,), target_args=(text,)):
        raise RuntimeError("Model conversion failed")
    return converter

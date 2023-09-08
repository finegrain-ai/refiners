import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter


def test_cross_attention_adapter() -> None:
    base = fl.Chain(fl.Attention(embedding_dim=4))
    adapter = CrossAttentionAdapter(base.Attention).inject()

    assert list(base) == [adapter]

    adapter.eject()

    assert len(base) == 1
    assert isinstance(base[0], fl.Attention)

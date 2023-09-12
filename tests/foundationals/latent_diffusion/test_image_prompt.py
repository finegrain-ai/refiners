import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter, InjectionPoint


def test_cross_attention_adapter() -> None:
    base = fl.Chain(fl.Attention(embedding_dim=4))
    adapter = CrossAttentionAdapter(base.Attention).inject()

    assert list(base) == [adapter]
    assert len(list(adapter.layers(fl.Linear))) == 6
    assert len(list(base.layers(fl.Linear))) == 6

    injection_points = list(adapter.layers(InjectionPoint))
    assert len(injection_points) == 4
    for ip in injection_points:
        assert len(ip) == 1
        assert isinstance(ip[0], fl.Linear)

    adapter.eject()

    assert len(base) == 1
    assert isinstance(base[0], fl.Attention)
    assert len(list(adapter.layers(fl.Linear))) == 2
    assert len(list(base.layers(fl.Linear))) == 4

    injection_points = list(adapter.layers(InjectionPoint))
    assert len(injection_points) == 4
    for ip in injection_points:
        assert len(ip) == 0

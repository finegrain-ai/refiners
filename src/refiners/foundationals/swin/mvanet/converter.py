import re

from torch import Tensor


def convert_weights(official_state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    rm_list = [
        # Official weights contains useless keys
        # See https://github.com/qianyu-dlut/MVANet/issues/3#issuecomment-2105650425
        r"multifieldcrossatt.linear[56]",
        r"multifieldcrossatt.attention.5",
        r"dec_blk\d+\.linear[12]",
        r"dec_blk[1234]\.attention\.[4567]",
        # We don't need the sideout weights
        r"sideout\d+",
    ]
    state_dict = {k: v for k, v in official_state_dict.items() if not any(re.match(rm, k) for rm in rm_list)}

    keys_map: dict[str, str] = {}
    for k in state_dict.keys():
        v: str = k

        def rpfx(s: str, src: str, dst: str) -> str:
            if not s.startswith(src):
                return s
            return s.replace(src, dst, 1)

        # Swin Transformer backbone

        v = rpfx(v, "backbone.patch_embed.proj.", "SwinTransformer.PatchEmbedding.Conv2d.")
        v = rpfx(v, "backbone.patch_embed.norm.", "SwinTransformer.PatchEmbedding.LayerNorm.")

        if m := re.match(r"backbone\.layers\.(\d+)\.downsample\.(.*)", v):
            s = m.group(2).replace("reduction.", "Linear.").replace("norm.", "LayerNorm.")
            v = f"SwinTransformer.Chain_{int(m.group(1)) + 1}.PatchMerging.{s}"

        if m := re.match(r"backbone\.layers\.(\d+)\.blocks\.(\d+)\.(.*)", v):
            s = m.group(3)
            s = s.replace("norm1.", "Residual_1.LayerNorm.")
            s = s.replace("norm2.", "Residual_2.LayerNorm.")

            s = s.replace("attn.qkv.", "Residual_1.WindowAttention.Linear_1.")
            s = s.replace("attn.proj.", "Residual_1.WindowAttention.Linear_2.")
            s = s.replace("attn.relative_position", "Residual_1.WindowAttention.WindowSDPA.rpb.relative_position")

            s = s.replace("mlp.fc", "Residual_2.Linear_")
            v = ".".join(
                [
                    f"SwinTransformer.Chain_{int(m.group(1)) + 1}",
                    f"BasicLayer.SwinTransformerBlock_{int(m.group(2)) + 1}",
                    s,
                ]
            )

        if m := re.match(r"backbone\.norm(\d+)\.(.*)", v):
            v = f"SwinTransformer.Chain_{int(m.group(1)) + 1}.Passthrough.LayerNorm.{m.group(2)}"

        # MVANet

        def mclm(s: str, pfx_src: str, pfx_dst: str) -> str:
            pca = f"{pfx_dst}Residual.PatchwiseCrossAttention"
            s = rpfx(s, f"{pfx_src}linear1.", f"{pfx_dst}FeedForward_1.Linear_1.")
            s = rpfx(s, f"{pfx_src}linear2.", f"{pfx_dst}FeedForward_1.Linear_2.")
            s = rpfx(s, f"{pfx_src}linear3.", f"{pfx_dst}FeedForward_2.Linear_1.")
            s = rpfx(s, f"{pfx_src}linear4.", f"{pfx_dst}FeedForward_2.Linear_2.")
            s = rpfx(s, f"{pfx_src}norm1.", f"{pfx_dst}LayerNorm_1.")
            s = rpfx(s, f"{pfx_src}norm2.", f"{pfx_dst}LayerNorm_2.")
            s = rpfx(s, f"{pfx_src}attention.0.", f"{pfx_dst}GlobalAttention.Sum.Chain.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.1.", f"{pca}.Concatenate.Chain_1.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.2.", f"{pca}.Concatenate.Chain_2.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.3.", f"{pca}.Concatenate.Chain_3.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.4.", f"{pca}.Concatenate.Chain_4.MultiheadAttention.")
            return s

        def mcrm(s: str, pfx_src: str, pfx_dst: str) -> str:
            # Note: there are no linear{1,2}, see https://github.com/qianyu-dlut/MVANet/issues/3#issuecomment-2105650425
            tca = f"{pfx_dst}Parallel_3.TiledCrossAttention"
            pca = f"{tca}.Sum.Chain_2.PatchwiseCrossAttention"
            s = rpfx(s, f"{pfx_src}linear3.", f"{tca}.FeedForward.Linear_1.")
            s = rpfx(s, f"{pfx_src}linear4.", f"{tca}.FeedForward.Linear_2.")
            s = rpfx(s, f"{pfx_src}norm1.", f"{tca}.LayerNorm_1.")
            s = rpfx(s, f"{pfx_src}norm2.", f"{tca}.LayerNorm_2.")
            s = rpfx(s, f"{pfx_src}attention.0.", f"{pca}.Concatenate.Chain_1.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.1.", f"{pca}.Concatenate.Chain_2.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.2.", f"{pca}.Concatenate.Chain_3.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}attention.3.", f"{pca}.Concatenate.Chain_4.MultiheadAttention.")
            s = rpfx(s, f"{pfx_src}sal_conv.", f"{pfx_dst}Parallel_2.Multiply.Chain.Conv2d.")
            return s

        def cbr(s: str, pfx_src: str, pfx_dst: str, shift: int = 0) -> str:
            s = rpfx(s, f"{pfx_src}{shift}.", f"{pfx_dst}Conv2d.")
            s = rpfx(s, f"{pfx_src}{shift + 1}.", f"{pfx_dst}BatchNorm2d.")
            s = rpfx(s, f"{pfx_src}{shift + 2}.", f"{pfx_dst}PReLU.")
            return s

        def cbg(s: str, pfx_src: str, pfx_dst: str) -> str:
            s = rpfx(s, f"{pfx_src}0.", f"{pfx_dst}Conv2d.")
            s = rpfx(s, f"{pfx_src}1.", f"{pfx_dst}BatchNorm2d.")
            return s

        v = rpfx(v, "shallow.0.", "ComputeShallow.Conv2d.")

        v = cbr(v, "output1.", "Pyramid.Sum.Chain.CBR.")
        v = cbr(v, "output2.", "Pyramid.Sum.PyramidL2.Sum.Chain.CBR.")
        v = cbr(v, "output3.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.Chain.CBR.")
        v = cbr(v, "output4.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.PyramidL4.Sum.Chain.CBR.")
        v = cbr(v, "output5.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.PyramidL4.Sum.PyramidL5.CBR.")

        v = cbr(v, "conv1.", "Pyramid.CBR.")
        v = cbr(v, "conv2.", "Pyramid.Sum.PyramidL2.CBR.")
        v = cbr(v, "conv3.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.CBR.")
        v = cbr(v, "conv4.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.PyramidL4.CBR.")

        v = mclm(v, "multifieldcrossatt.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.PyramidL4.Sum.PyramidL5.MCLM.")

        v = mcrm(v, "dec_blk1.", "Pyramid.MCRM.")
        v = mcrm(v, "dec_blk2.", "Pyramid.Sum.PyramidL2.MCRM.")
        v = mcrm(v, "dec_blk3.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.MCRM.")
        v = mcrm(v, "dec_blk4.", "Pyramid.Sum.PyramidL2.Sum.PyramidL3.Sum.PyramidL4.MCRM.")

        v = cbr(v, "insmask_head.", "RearrangeMultiView.Chain.CBR_1.")
        v = cbr(v, "insmask_head.", "RearrangeMultiView.Chain.CBR_2.", shift=3)

        v = rpfx(v, "insmask_head.6.", "RearrangeMultiView.Chain.Conv2d.")

        v = cbg(v, "upsample1.", "ShallowUpscaler.Sum_2.Chain_1.CBG.")
        v = cbg(v, "upsample2.", "ShallowUpscaler.CBG.")

        v = rpfx(v, "output.0.", "Conv2d.")

        if v != k:
            keys_map[k] = v

    for key, new_key in keys_map.items():
        state_dict[new_key] = state_dict[key]
        state_dict.pop(key)

    return state_dict

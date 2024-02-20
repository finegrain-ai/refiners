from torch import device as Device, dtype as DType

import refiners.fluxion.layers as fl


class VitMatteBasicConv3x3(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        stride: int = 2,
        padding: int = 1,
        use_bias: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding

        super().__init__(
            fl.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=self.padding,
                use_bias=self.use_bias,
                device=device,
                dtype=dtype,
            ),
            fl.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            fl.ReLU(),
        )


class VitMatteConvStream(fl.Passthrough):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: list[int] = [48, 96, 192],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        self.conv_chans = [in_channels] + out_channels
        super().__init__(
            fl.UseContext(context="detail_capture", key="images"),
            fl.SetContext("detailed_features", f"{0}"),
            *[
                fl.Chain(
                    VitMatteBasicConv3x3(self.conv_chans[i], self.conv_chans[i + 1], device=device, dtype=dtype),
                    fl.SetContext("detailed_features", f"{i+1}"),
                )
                for i in range(len(self.conv_chans) - 1)
            ],
        )


class Fusion_Block(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_key: str,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_key = conv_key

        super().__init__(
            fl.Concatenate(
                fl.Chain(fl.UseContext(context="detailed_features", key=self.conv_key)),
                fl.Interpolate(factor=2, mode="bilinear"),
                dim=1,
            ),
            VitMatteBasicConv3x3(self.in_channels, self.out_channels, stride=1, padding=1),
        )


class Matting_Head(fl.Chain):
    def __init__(
        self,
        in_channels: int = 32,
        mid_channels: int = 16,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        super().__init__(
            fl.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1, use_bias=True),
            fl.BatchNorm2d(mid_channels),
            fl.ReLU(),
            fl.Conv2d(
                in_channels=self.mid_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
                device=device,
                dtype=dtype,
            ),
        )


class FusionBlocks(fl.Chain):
    pass


class Detail_Capture(fl.Chain):
    def __init__(
        self,
        in_channels: int = 384,
        img_channels: int = 4,
        conv_channels: list[int] = [4, 48, 96, 192],
        convstream_out: list[int] = [48, 96, 192],
        fusion_out: list[int] = [256, 128, 64, 32],
    ) -> None:
        assert len(fusion_out) == len(convstream_out) + 1
        self.img_channels = img_channels
        self.convstream_out = convstream_out
        self.fusion_channels = [in_channels] + fusion_out
        self.conv_channels = conv_channels

        super().__init__(
            VitMatteConvStream(in_channels=img_channels),
            FusionBlocks(
                Fusion_Block(
                    in_channels=self.fusion_channels[i] + self.conv_channels[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                    conv_key=f"{len(self.conv_channels)-i-1}",
                )
                for i in range(len(self.fusion_channels) - 1)
            ),
            Matting_Head(in_channels=fusion_out[-1]),
            fl.Sigmoid(),
        )

    def init_context(self):
        return {"detailed_features": {f"{i}": None for i in range(len(self.fusion_channels) - 1)}}

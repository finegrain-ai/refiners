from torch import Tensor, device as Device, dtype as DType

from refiners.fluxion.adapters.adapter import Adapter
from refiners.fluxion.adapters.lora import Lora, LoraAdapter
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import Chain, Conv2d, Multiply, Passthrough, Residual, SiLU, UseContext
from refiners.fluxion.layers.module import WeightedModule
from refiners.foundationals.latent_diffusion.range_adapter import RangeAdapter2d
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import ResidualAccumulator
from refiners.foundationals.latent_diffusion.stable_diffusion_xl.unet import SDXLUNet
from refiners.foundationals.latent_diffusion.unet import ResidualBlock


class ConditionEncoder(Chain):
    """Encode an image into a condition latent tensor.

    Receives:
        (Float[Tensor, "batch in_channels width height"]): The input image.

    Returns:
        (Float[Tensor, "batch out_channels latent_width latent_height"]): The condition latent tensor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 320,
        intermediate_channels: tuple[int, ...] = (16, 32, 96, 256),
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the ConditionEncoder.

        Args:
            in_channels: The number of channels of the image tensor.
            out_channels: The number of channels of the latent tensor to encode the condition into.
            intermediate_channels: The number of channels of the intermediate layers.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """

        super().__init__(
            Chain(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=intermediate_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                SiLU(),
            ),
            *(
                Chain(
                    Conv2d(
                        in_channels=intermediate_channels[i],
                        out_channels=intermediate_channels[i],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    SiLU(),
                    Conv2d(
                        in_channels=intermediate_channels[i],
                        out_channels=intermediate_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        device=device,
                        dtype=dtype,
                    ),
                    SiLU(),
                )
                for i in range(len(intermediate_channels) - 1)
            ),
            Conv2d(
                in_channels=intermediate_channels[-1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                device=device,
                dtype=dtype,
            ),
        )


class ZeroConvolution(Passthrough):
    """Transform and store the ControlLora's residuals in the context of the original UNet.

    Receives:
        (Float[Tensor, "batch in_channels width height"]): The input tensor to transform and store.

    Returns: Updates context:
        (Tensor): Add the residual to the nth residual of the target's UNet.
            (context="unet", key="residuals")
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual_index: int,
        scale: float = 1.0,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize the ZeroConvolution.

        Args:
            in_channels: The number of channels of the input tensor.
            out_channels: The number of channels of the output tensor/residual.
            residual_index: The index of the residual to store in the target's UNet.
            scale: The scale to multiply the residuals by.
            device: The PyTorch device to use.
            dtype: The PyTorch data type to use.
        """
        self._scale = scale

        super().__init__(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                device=device,
                dtype=dtype,
            ),
            Multiply(scale=scale),
            ResidualAccumulator(n=residual_index),
        )

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value
        self.ensure_find(Multiply).scale = value


class ControlLora(Passthrough):
    """ControlLora is a Half-UNet clone of the target UNet,
    patched with various `LoRA` layers, `ZeroConvolution` layers, and a `ConditionEncoder`.

    Like ControlNet, it injects residual tensors into the target UNet.
    See <https://github.com/HighCWu/control-lora-v2> for more details.

    Receives: Gets context:
        (Float[Tensor, "batch condition_channels width height"]): The input image.

    Returns: Sets context:
        (list[Tensor]): The residuals to be added to the target UNet's residuals.
            (context="unet", key="residuals")
    """

    def __init__(
        self,
        name: str,
        unet: SDXLUNet,
        scale: float = 1.0,
        condition_channels: int = 3,
    ) -> None:
        """Initialize the ControlLora.

        Args:
            name: The name of the ControlLora.
            unet: The target UNet.
            scale: The scale to multiply the residuals by.
            condition_channels: The number of channels of the input condition tensor.
        """
        self.name = name

        super().__init__(
            timestep_encoder := unet.layer("TimestepEncoder", Chain).structural_copy(),
            downblocks := unet.layer("DownBlocks", Chain).structural_copy(),
            middle_block := unet.layer("MiddleBlock", Chain).structural_copy(),
        )

        # modify the context_key of the copied TimestepEncoder to avoid conflicts
        timestep_encoder.context_key = f"timestep_embedding_control_lora_{name}"

        # modify the context_key of each RangeAdapter2d to avoid conflicts
        for range_adapter in self.layers(RangeAdapter2d):
            range_adapter.context_key = f"timestep_embedding_control_lora_{name}"

        # insert the ConditionEncoder in the first DownBlock
        first_downblock = downblocks.layer(0, Chain)
        out_channels = first_downblock.layer(0, Conv2d).out_channels
        first_downblock.append(
            Residual(
                UseContext(f"control_lora_{name}", f"condition"),
                ConditionEncoder(
                    in_channels=condition_channels,
                    out_channels=out_channels,
                    device=unet.device,
                    dtype=unet.dtype,
                ),
            )
        )

        # replace each ResidualAccumulator by a ZeroConvolution
        for residual_accumulator in self.layers(ResidualAccumulator):
            downblock = self.ensure_find_parent(residual_accumulator)

            first_layer = downblock[0]
            assert hasattr(first_layer, "out_channels"), f"{first_layer} has no out_channels attribute"

            block_channels = first_layer.out_channels
            assert isinstance(block_channels, int)

            downblock.replace(
                residual_accumulator,
                ZeroConvolution(
                    scale=scale,
                    residual_index=residual_accumulator.n,
                    in_channels=block_channels,
                    out_channels=block_channels,
                    device=unet.device,
                    dtype=unet.dtype,
                ),
            )

        # append a ZeroConvolution to middle_block
        middle_block_channels = middle_block.layer(0, ResidualBlock).out_channels
        middle_block.append(
            ZeroConvolution(
                scale=scale,
                residual_index=len(downblocks),
                in_channels=middle_block_channels,
                out_channels=middle_block_channels,
                device=unet.device,
                dtype=unet.dtype,
            )
        )

    @property
    def scale(self) -> float:
        """The scale of the residuals stored in the context."""
        zero_convolution_module = self.ensure_find(ZeroConvolution)
        return zero_convolution_module.scale

    @scale.setter
    def scale(self, value: float) -> None:
        for zero_convolution_module in self.layers(ZeroConvolution):
            zero_convolution_module.scale = value


class ControlLoraAdapter(Chain, Adapter[SDXLUNet]):
    """Adapter for [`ControlLora`][refiners.foundationals.latent_diffusion.stable_diffusion_xl.ControlLora].

    This adapter simply prepends a `ControlLora` model inside the target `SDXLUNet`.
    """

    def __init__(
        self,
        name: str,
        target: SDXLUNet,
        scale: float = 1.0,
        condition_channels: int = 3,
        weights: dict[str, Tensor] | None = None,
    ) -> None:
        with self.setup_adapter(target):
            self.name = name
            self._control_lora = [
                ControlLora(
                    name=name,
                    unet=target,
                    scale=scale,
                    condition_channels=condition_channels,
                ),
            ]

            super().__init__(target)

        if weights:
            self.load_weights(weights)

    @property
    def control_lora(self) -> ControlLora:
        """The ControlLora model."""
        return self._control_lora[0]

    def init_context(self) -> Contexts:
        return {
            f"control_lora_{self.name}": {
                "condition": None,
            }
        }

    def inject(self, parent: Chain | None = None) -> "ControlLoraAdapter":
        self.target.insert(index=0, module=self.control_lora)
        return super().inject(parent)

    def eject(self) -> None:
        self.target.remove(self.control_lora)
        return super().eject()

    def structural_copy(self) -> "ControlLoraAdapter":
        raise RuntimeError("ControlLoraAdapter cannot be copied, eject it first.")

    @property
    def scale(self) -> float:
        """The scale of the injected residuals."""
        return self.control_lora.scale

    @scale.setter
    def scale(self, value: float) -> None:
        self.control_lora.scale = value

    def set_condition(self, condition: Tensor) -> None:
        self.set_context(
            context=f"control_lora_{self.name}",
            value={"condition": condition},
        )

    def load_weights(
        self,
        state_dict: dict[str, Tensor],
    ) -> None:
        """Load the weights from the state_dict into the `ControlLora`.

        Args:
            state_dict: The state_dict containing the weights to load.
        """
        ControlLoraAdapter.load_lora_layers(self.name, state_dict, self.control_lora)
        ControlLoraAdapter.load_zero_convolution_layers(state_dict, self.control_lora)
        ControlLoraAdapter.load_condition_encoder(state_dict, self.control_lora)

    @staticmethod
    def load_lora_layers(
        name: str,
        state_dict: dict[str, Tensor],
        control_lora: ControlLora,
    ) -> None:
        """Load the [`LoRA`][refiners.fluxion.adapters.lora.Lora] layers from the state_dict into the `ControlLora`.

        Args:
            name: The name of the ControlLora.
            state_dict: The state_dict containing the LoRA layers to load.
            control_lora: The ControlLora to load the LoRA layers into.
        """
        # filter the LoraAdapters from the state_dict
        lora_weights = {
            key.removeprefix("ControlLora."): value for key, value in state_dict.items() if "ControlLora" in key
        }
        lora_weights = {f"{key}.weight": value for key, value in lora_weights.items()}

        # move the tensors to the device and dtype of the ControlLora
        lora_weights = {
            key: value.to(
                dtype=control_lora.dtype,
                device=control_lora.device,
            )
            for key, value in lora_weights.items()
        }

        # load every LoRA layers from the filtered state_dict
        loras = Lora.from_dict(name, state_dict=lora_weights)

        # attach the LoRA layers to the ControlLora
        adapters: list[LoraAdapter] = []
        for key, lora in loras.items():
            target = control_lora.layer(key.split("."), WeightedModule)
            assert lora.is_compatible(target)
            adapter = LoraAdapter(target, lora)
            adapters.append(adapter)

        for adapter in adapters:
            adapter.inject(control_lora)

    @staticmethod
    def load_zero_convolution_layers(
        state_dict: dict[str, Tensor],
        control_lora: ControlLora,
    ):
        """Load the `ZeroConvolution` layers from the state_dict into the `ControlLora`.

        Args:
            state_dict: The state_dict containing the ZeroConvolution layers to load.
            control_lora: The ControlLora to load the ZeroConvolution layers into.
        """
        zero_convolution_layers = list(control_lora.layers(ZeroConvolution))
        for i, zero_convolution_layer in enumerate(zero_convolution_layers):
            zero_convolution_state_dict = {
                key.removeprefix(f"ZeroConvolution_{i+1:02d}."): value
                for key, value in state_dict.items()
                if f"ZeroConvolution_{i+1:02d}" in key
            }
            zero_convolution_layer.load_state_dict(zero_convolution_state_dict)

    @staticmethod
    def load_condition_encoder(
        state_dict: dict[str, Tensor],
        control_lora: ControlLora,
    ):
        """Load the `ConditionEncoder`'s layers from the state_dict into the `ControlLora`.

        Args:
            state_dict: The state_dict containing the ConditionEncoder layers to load.
            control_lora: The ControlLora to load the ConditionEncoder layers into.
        """
        condition_encoder_layer = control_lora.ensure_find(ConditionEncoder)
        condition_encoder_state_dict = {
            key.removeprefix("ConditionEncoder."): value
            for key, value in state_dict.items()
            if "ConditionEncoder" in key
        }
        condition_encoder_layer.load_state_dict(condition_encoder_state_dict)

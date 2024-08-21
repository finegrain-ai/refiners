import torch
from PIL import Image
from torch.nn.init import zeros_ as zero_init

from refiners.fluxion import layers as fl
from refiners.fluxion.utils import no_grad
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion.solvers.solver import Solver
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder, StableDiffusion_1
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import DownBlocks, SD1UNet


class ICLight(StableDiffusion_1):
    """IC-Light is a Stable Diffusion model that can be used to relight a reference image.

    At initialization, the UNet will be patched to accept four additional input channels.
    Only the text-conditioned relighting model is supported for now.


    Example:
        ```py
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image

        from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad
        from refiners.foundationals.clip import CLIPTextEncoderL
        from refiners.foundationals.latent_diffusion.stable_diffusion_1 import SD1Autoencoder, SD1UNet
        from refiners.foundationals.latent_diffusion.stable_diffusion_1.ic_light import ICLight

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        no_grad().__enter__()
        manual_seed(42)

        sd = ICLight(
            patch_weights=load_from_safetensors(
                path=hf_hub_download(
                    repo_id="refiners/ic_light.sd1_5.fc",
                    filename="model.safetensors",
                ),
                device=device,
            ),
            unet=SD1UNet(in_channels=4, device=device, dtype=dtype).load_from_safetensors(
                tensors_path=hf_hub_download(
                    repo_id="refiners/realistic_vision.v5_1.sd1_5.unet",
                    filename="model.safetensors",
                )
            ),
            clip_text_encoder=CLIPTextEncoderL(device=device, dtype=dtype).load_from_safetensors(
                tensors_path=hf_hub_download(
                    repo_id="refiners/realistic_vision.v5_1.sd1_5.text_encoder",
                    filename="model.safetensors",
                )
            ),
            lda=SD1Autoencoder(device=device, dtype=dtype).load_from_safetensors(
                tensors_path=hf_hub_download(
                    repo_id="refiners/realistic_vision.v5_1.sd1_5.autoencoder",
                    filename="model.safetensors",
                )
            ),
            device=device,
            dtype=dtype,
        )

        prompt = "soft lighting, high-quality professional image"
        negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        clip_text_embedding = sd.compute_clip_text_embedding(text=prompt, negative_text=negative_prompt)

        image = Image.open("reference-image.png").resize((512, 512))
        sd.set_ic_light_condition(image)

        x = torch.randn(
            size=(1, 4, 64, 64),
            device=device,
            dtype=dtype,
        )

        for step in sd.steps:
            x = sd(
                x=x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                condition_scale=1.5,
            )
        predicted_image = sd.lda.latents_to_image(x)

        predicted_image.save("ic-light-output.png")
        ```
    """

    def __init__(
        self,
        patch_weights: dict[str, torch.Tensor],
        unet: SD1UNet,
        lda: SD1Autoencoder | None = None,
        clip_text_encoder: CLIPTextEncoderL | None = None,
        solver: Solver | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            unet=unet,
            lda=lda,
            clip_text_encoder=clip_text_encoder,
            solver=solver,
            device=device,
            dtype=dtype,
        )
        self._extend_conv_in()
        self._apply_patch(weights=patch_weights)

    @no_grad()
    def _extend_conv_in(self) -> None:
        """Extend to 8 the input channels of the first convolutional layer of the UNet."""
        down_blocks = self.unet.ensure_find(DownBlocks)
        first_block = down_blocks.layer(0, fl.Chain)
        conv_in = first_block.ensure_find(fl.Conv2d)
        new_conv_in = fl.Conv2d(
            in_channels=conv_in.in_channels + 4,
            out_channels=conv_in.out_channels,
            kernel_size=(conv_in.kernel_size[0], conv_in.kernel_size[1]),
            padding=(int(conv_in.padding[0]), int(conv_in.padding[1])),
            device=conv_in.device,
            dtype=conv_in.dtype,
        )
        zero_init(new_conv_in.weight)
        new_conv_in.bias = conv_in.bias
        new_conv_in.weight[:, :4, :, :] = conv_in.weight
        first_block.replace(old_module=conv_in, new_module=new_conv_in)

    def _apply_patch(self, weights: dict[str, torch.Tensor]) -> None:
        """Apply the weights patch to the UNet, modifying inplace the state dict."""
        current_state_dict = self.unet.state_dict()
        new_state_dict = {
            key: tensor + weights[key].to(tensor.device, tensor.dtype) for key, tensor in current_state_dict.items()
        }
        self.unet.load_state_dict(new_state_dict)

    @staticmethod
    def compute_gray_composite(
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        """Compute a grayscale composite of an image and a mask.

        IC-Light will recreate the image

        Args:
            image: The image to composite.
            mask: The mask to use for the composite.
        """
        assert mask.mode == "L", "Mask must be a grayscale image"
        assert image.size == mask.size, "Image and mask must have the same size"
        background = Image.new("RGB", image.size, (127, 127, 127))
        return Image.composite(image, background, mask)

    def set_ic_light_condition(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> None:
        """Set the IC light condition.

        Args:
            image: The reference image.
            mask: The mask to use for the reference image.

        If a mask is provided, it will be used to compute a grayscale composite of the image and the mask ; otherwise,
        the image will be used as is, but note that IC-Light requires a 127-valued gray background to work.
        """
        if mask is not None:
            image = self.compute_gray_composite(image=image, mask=mask)
        latents = self.lda.image_to_latents(image)
        self._ic_light_condition = latents

    def __call__(
        self,
        x: torch.Tensor,
        step: int,
        *,
        clip_text_embedding: torch.Tensor,
        condition_scale: float = 2.0,
    ) -> torch.Tensor:
        assert self._ic_light_condition is not None, "Reference image not set, use `set_ic_light_condition` first"
        x = torch.cat((x, self._ic_light_condition), dim=1)
        return super().__call__(
            x,
            step,
            clip_text_embedding=clip_text_embedding,
            condition_scale=condition_scale,
        )

from pathlib import Path

import torch
from PIL import Image

from refiners.fluxion.utils import image_to_tensor, no_grad, normalize, tensor_to_image
from refiners.foundationals.swin.mvanet import MVANet

BoundingBox = tuple[int, int, int, int]


class BoxSegmenter:
    def __init__(
        self,
        *,
        margin: float = 0.05,
        weights: Path | str | dict[str, torch.Tensor] | None = None,
        device: torch.device | str = "cpu",
    ):
        assert margin >= 0
        self.margin = margin

        self.device = torch.device(device)
        self.model = MVANet(device=self.device).eval()

        if weights is None:
            from huggingface_hub.file_download import hf_hub_download  # type: ignore[reportUnknownVariableType]

            weights = hf_hub_download(
                repo_id="finegrain/finegrain-box-segmenter",
                filename="model.safetensors",
                revision="v0.1",
            )

        if isinstance(weights, dict):
            self.model.load_state_dict(weights)
        else:
            self.model.load_from_safetensors(weights)

    def __call__(self, img: Image.Image, box_prompt: BoundingBox | None = None) -> Image.Image:
        return self.run(img, box_prompt)

    def add_margin(self, box: BoundingBox) -> BoundingBox:
        x0, y0, x1, y1 = box
        mx = int((x1 - x0) * self.margin)
        my = int((y1 - y0) * self.margin)
        return (x0 - mx, y0 - my, x1 + mx, y1 + my)

    @staticmethod
    def crop_pad(img: Image.Image, box: BoundingBox) -> Image.Image:
        img = img.convert("RGB")

        x0, y0, x1, y1 = box
        px0, py0, px1, py1 = (max(0, -x0), max(0, -y0), max(0, x1 - img.width), max(0, y1 - img.height))
        if (px0, py0, px1, py1) == (0, 0, 0, 0):
            return img.crop(box)

        padded = Image.new("RGB", (img.width + px0 + px1, img.height + py0 + py1))
        padded.paste(img, (px0, py0))
        return padded.crop((x0 + px0, y0 + py0, x1 + px0, y1 + py0))

    def predict(self, img: Image.Image) -> Image.Image:
        in_t = image_to_tensor(img.resize((1024, 1024), Image.Resampling.BILINEAR)).squeeze()
        in_t = normalize(in_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
        with no_grad():
            prediction: torch.Tensor = self.model(in_t.to(self.device)).sigmoid()
        return tensor_to_image(prediction).resize(img.size, Image.Resampling.BILINEAR)

    def run(self, img: Image.Image, box_prompt: BoundingBox | None = None) -> Image.Image:
        if box_prompt is None:
            box_prompt = (0, 0, img.width, img.height)

        box = self.add_margin(box_prompt)
        cropped = self.crop_pad(img, box)
        prediction = self.predict(cropped)

        out = Image.new("L", (img.width, img.height))
        out.paste(prediction, box)
        return out

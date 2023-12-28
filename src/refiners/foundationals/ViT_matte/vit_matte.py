import refiners.fluxion.layers as fl
from refiners.foundationals.ViT_matte.decoder import Detail_Capture
from refiners.foundationals.ViT_matte.vit_backbone import MViTH


class ViTMatte(fl.Chain):
    def __init__(self):
        super().__init__(fl.SetContext(context="detail_capture", key="images"), MViTH(), Detail_Capture())

    def init_context(self):
        return {"detail_capture": {"images": None}}

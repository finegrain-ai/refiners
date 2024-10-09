import logging

from refiners.conversion import (
    autoencoder_sd15,
    autoencoder_sdxl,
    clip_image_sd21,
    clip_text_sd15,
    clip_text_sdxl,
    controllora_sdxl,
    controlnet_sd15,
    dinov2,
    ella,
    hq_sam,
    ipadapter_sd15,
    ipadapter_sdxl,
    loras,
    mvanet,
    preprocessors,
    sam,
    t2iadapter_sd15,
    t2iadapter_sdxl,
    unet_sd15,
    unet_sdxl,
)


def main() -> None:
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # MVANet
    mvanet.mvanet.convert()
    mvanet.finegrain_v01.download()

    # loras (no conversion)
    loras.sd15_pokemon.download()
    loras.sdxl_dpo.download()
    loras.sdxl_scifi.download()
    loras.sdxl_pixelart.download()
    loras.sdxl_age_slider.download()
    loras.sdxl_cartoon_slider.download()
    loras.sdxl_eyesize_slider.download()

    # preprocessors
    preprocessors.informative_drawings.convert()

    # SD1.5 autoencoders
    autoencoder_sd15.runwayml.convert()
    autoencoder_sd15.stability_mse.convert()
    autoencoder_sd15.juggernaut_reborn.convert()
    autoencoder_sd15.juggernaut_aftermath.convert()
    autoencoder_sd15.realistic_stock_photo_v3.convert()
    autoencoder_sd15.realistic_vision_v5.convert()

    # SDXL autoencoders
    autoencoder_sdxl.stability.convert()
    autoencoder_sdxl.madebyollin_fp16fix.convert()
    autoencoder_sdxl.juggernautXL_v10.convert()

    # SD1.5 text encoders
    clip_text_sd15.runwayml.convert()
    clip_text_sd15.juggernaut_reborn.convert()
    clip_text_sd15.juggernaut_aftermath.convert()
    clip_text_sd15.realistic_stock_photo_v3.convert()
    clip_text_sd15.realistic_vision_v5.convert()

    # SD2.1 image encoders
    clip_image_sd21.unclip_21.convert()

    # SDXL text encoders
    clip_text_sdxl.stability.convert()
    clip_text_sdxl.juggernautXL_v10.convert()

    # SD1.5 unets
    unet_sd15.runwayml.convert()
    unet_sd15.runwayml_inpainting.convert()
    unet_sd15.juggernaut_reborn.convert()
    unet_sd15.juggernaut_aftermath.convert()
    unet_sd15.realistic_stock_photo_v3.convert()
    unet_sd15.realistic_vision_v5.convert()

    # SD1.5 IC-Light
    unet_sd15.ic_light_fc.convert()
    unet_sd15.ic_light_fcon.convert()
    unet_sd15.ic_light_fbc.convert()

    # SDXL unets
    unet_sdxl.stability.convert()
    unet_sdxl.juggernautXL_v10.convert()

    # SDXL LCM unet
    unet_sdxl.lcm.convert()

    # SDXL Lightning unet
    unet_sdxl.lightning_4step.convert()
    unet_sdxl.lightning_1step.convert()

    # SD1.5 controlnets
    controlnet_sd15.tile.convert()
    controlnet_sd15.canny.convert()
    controlnet_sd15.depth.convert()
    controlnet_sd15.normalbae.convert()
    controlnet_sd15.lineart.convert()
    controlnet_sd15.sam.convert()

    # SDXL Control LoRAs
    controllora_sdxl.canny.convert()
    controllora_sdxl.cpds.convert()

    # SD1.5 IP-Adapters
    ipadapter_sd15.base.convert()
    ipadapter_sd15.plus.convert()

    # SDXL IP-Adapters
    ipadapter_sdxl.base.convert()
    ipadapter_sdxl.plus.convert()

    # SD1.5 T2I-Adapters
    t2iadapter_sd15.depth.convert()

    # SDXL T2I-Adapters
    t2iadapter_sdxl.canny.convert()

    # ELLA adapters
    ella.sd15_t5xl.convert()

    # DINOv2
    dinov2.small.convert()
    dinov2.small_reg.convert()
    dinov2.base.convert()
    dinov2.base_reg.convert()
    dinov2.large.convert()
    dinov2.large_reg.convert()
    dinov2.giant.convert()
    dinov2.giant_reg.convert()

    # SAM
    sam.vit_h.convert()

    # SAM-HQ
    hq_sam.vit_h.convert()


if __name__ == "__main__":
    main()

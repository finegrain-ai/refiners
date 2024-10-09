from refiners.conversion.utils import Hub

sd15_pokemon = Hub(
    repo_id="pcuenq/pokemon-lora",
    filename="pytorch_lora_weights.bin",
    revision="31ae8fe6f588a78c02828e9b8d352dccd90f1a24",
    expected_sha256="f712fcfb6618da14d25a4f3e0c9460a878fc2417e2df95cdd683a73f71b50384",
)
sdxl_dpo = Hub(
    repo_id="radames/sdxl-DPO-LoRA",
    filename="pytorch_lora_weights.safetensors",
    revision="319a544fff501b3ed907df67e1db356bee364c9f",
    expected_sha256="aeb5ec4a7db6679ea8085f794db1abca92cfd8e4c667a1b301b2b8ecd5599a5d",
)
sdxl_scifi = Hub(
    repo_id="civitai/Ciro_Negrogni",
    filename="Sci-fi_Environments_sdxl.safetensors",
    expected_sha256="5a3f738c9f79c65c1fac1418b1fe593967b0c1bd24fdb27f120ef1685e815c8e",
    download_url="https://civitai.com/api/download/models/140624?type=Model&format=SafeTensor",
)
sdxl_pixelart = Hub(
    repo_id="civitai/NeriJS",
    filename="pixel-art-xl-v1.1.safetensors",
    expected_sha256="bbf3d8defbfb3fb71331545225c0cf50c74a748d2525f7c19ebb8f74445de274",
    download_url="https://civitai.com/api/download/models/135931?type=Model&format=SafeTensor",
)
sdxl_age_slider = Hub(
    repo_id="baulab/sliders",
    filename="age.pt",
    expected_sha256="8c1c096f7cc1109b4072cbc604c811a5f0ff034fc0f6dc7cf66a558550aa4890",
    download_url="https://sliders.baulab.info/weights/xl_sliders/age.pt",
)
sdxl_cartoon_slider = Hub(
    repo_id="baulab/sliders",
    filename="cartoon_style.pt",
    expected_sha256="e07c30e4f82f709a474ae11dc5108ac48f81b6996b937757c8dd198920ea9b4d",
    download_url="https://sliders.baulab.info/weights/xl_sliders/cartoon_style.pt",
)
sdxl_eyesize_slider = Hub(
    repo_id="baulab/sliders",
    filename="eyesize.pt",
    expected_sha256="8fdffa3e7788f4bd6be9a2fe3b91957b4f35999fc9fa19eabfb49f92fbf6650b",
    download_url="https://sliders.baulab.info/weights/xl_sliders/eyesize.pt",
)
sdxl_lcm = Hub(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    revision="a18548dd4956b174ec5b0d78d340c8dae0a129cd",
    expected_sha256="a764e6859b6e04047cd761c08ff0cee96413a8e004c9f07707530cd776b19141",
)
sdxl_lightning_4steps = Hub(
    repo_id="ByteDance/SDXL-Lightning",
    filename="sdxl_lightning_4step_lora.safetensors",
    revision="c9a24f48e1c025556787b0c58dd67a091ece2e44",
    expected_sha256="bf56cf2657efb15e465d81402ed481d1e11c4677e4bcce1bc11fe71ad8506b79",
)

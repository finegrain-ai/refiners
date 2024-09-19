<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_light.png">
  <img alt="Finegrain Refiners Library" width="352" height="128" style="max-width: 100%;">
</picture>

**The simplest way to train and run adapters on top of foundation models**

[**Manifesto**](https://refine.rs/home/why/) |
[**Docs**](https://refine.rs) |
[**Guides**](https://refine.rs/guides/adapting_sdxl/) |
[**Discussions**](https://github.com/finegrain-ai/refiners/discussions) |
[**Discord**](https://discord.gg/mCmjNUVV7d)

</div>

## Installation

1. Navigate to the root of your ComfyUI workspace.
2. Activate your python virtual environment.
3. Install the nodes using one of the following methods.

### Comfy Registry (recommended)

The nodes are published at https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners.

See https://docs.comfy.org/comfy-cli/getting-started to install the Comfy CLI.

To automagically install the nodes, run the following command:
```bash
comfy node registry-install comfyui-refiners
```

See https://docs.comfy.org/registry/overview for more information.

### Manual

To manually install the nodes, you may alternatively do the following:
```bash
curl -o comfyui-refiners.zip https://storage.googleapis.com/comfy-registry/finegrain/comfyui-refiners/1.0.3/node.tar.gz
unzip -d custom_nodes/comfyui-refiners comfyui-refiners.zip
pip install -r custom_nodes/comfyui-refiners/requirements.txt
rm comfyui-refiners.zip
```

## Example Workflows

### [Box Segmenter](assets/box_segmenter.json)

This simple workflow leverages GroundingDINO and our [BoxSegmenter](https://huggingface.co/finegrain/finegrain-box-segmenter) to extract objects from an image.
[![Box Segmenter Workflow](assets/box_segmenter.png)](assets/box_segmenter.json)

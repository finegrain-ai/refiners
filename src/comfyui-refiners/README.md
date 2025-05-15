<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/finegrain-ai/refiners/main/assets/logo_light.png">
  <img alt="Finegrain Refiners Library" width="352" height="128" style="max-width: 100%;">
</picture>

**The simplest way to train and run adapters on top of foundation models**

[**Manifesto**](https://finegrain-ai.github.io/refiners/home/why/) |
[**Docs**](https://finegrain-ai.github.io/refiners/) |
[**Guides**](https://finegrain-ai.github.io/refiners/guides/adapting_sdxl/) |
[**Discussions**](https://github.com/finegrain-ai/refiners/discussions) |
[**Discord**](https://discord.gg/a4w4jXJ6)

</div>

## Installation

The nodes are published at https://registry.comfy.org/publishers/finegrain/nodes/comfyui-refiners. See our [installation guide](https://finegrain-ai.github.io/refiners/guides/comfyui_refiners/) for more details.

## Example Workflows

### [Box Segmenter](assets/box_segmenter.json)

This simple workflow leverages GroundingDINO and our [BoxSegmenter](https://huggingface.co/finegrain/finegrain-box-segmenter) to extract objects from an image.
[![Box Segmenter Workflow](assets/box_segmenter.png)](assets/box_segmenter.json)

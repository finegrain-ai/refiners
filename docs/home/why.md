---
icon: material/lightbulb-on-outline
---

# Why Refiners?

## PyTorch: an imperative framework

PyTorch is a great framework to implement deep learning models, widely adopted in academia and industry around the globe. A core design principle of PyTorch is that users write *imperative* Python code that manipulates Tensors[^1]. This code can be organized in Modules, which are just Python classes whose constructors typically initialize parameters and load weights, and which implement a `forward` method that computes the forward pass. Dealing with reconstructing an inference graph, backpropagation and so on are left to the framework.

This approach works very well in general, as demonstrated by the popularity of PyTorch. However, the growing importance of the Adaptation pattern is challenging it.

## Adaptation: patching foundation models

Adaptation is the idea of *patching* existing powerful models to implement new capabilities. Those models are called foundation models; they are typically trained from scratch on amounts of data inaccessible to most individuals, small companies or research labs, and exhibit emergent properties. Examples of such models are LLMs (GPT, LLaMa, Mistral), image generation models (Stable Diffusion, Muse), vision models (BLIP-2, LLaVA 1.5, Fuyu-8B) but also models trained on more specific tasks such as embedding extraction (CLIP, DINOv2) or image segmentation (SAM).

Adaptation of foundation models can take many forms. One of the simplest but most powerful derives from fine-tuning: re-training a subset of the weights of the model on a specific task, then distributing only those weights. Add to this a trick to significantly reduce the size of the fine-tuned weights and you get LoRA[^2], which is probably the most well-known adaptation method. However, adaptation can go beyond that and change the shape of the model or its inputs.

## Imperative code is hard to patch cleanly

There are several approaches to patch the code of a foundation model implemented in typical PyTorch imperative style to support adaptation, including:

- Just duplicate the original code base and edit it in place unconditionally. This approach is often adopted by researchers today.
- Change the original code base to optionally support the adapter. This approach is often used by frameworks and libraries built on top of PyTorch and works well for a single adapter. However, as you start adding support for multiple adapters to the same foundation model the cyclomatic complexity explodes and the code becomes hard to maintain and error-prone. The end result is that adapters typically do not compose well.
- Change the original code to abstract adaptation by adding ad-hoc hooks everywhere. This approach has the advantage of keeping the foundation model independent from its adapter, but it makes the code extremely non-linear and hard to reason about - so-called "spaghetti code".

As believers in adaptation, none of those approaches was appealing to us, so we designed Refiners as a better option. Refiners is a micro-framework built on top of PyTorch which does away with its imperative style. In Refiners, models are implemented in a *declarative* way instead, which makes them by nature easier to manipulate and patch.

## What's next?

Now you know *why* we wrote a declarative framework, you can check out [*how*](/concepts/chain/). It's not that complicated, we promise!

[^1]: Paszke et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library.
[^2]: Hu et al., 2022. LoRA: Low-Rank Adaptation of Large Language Models.


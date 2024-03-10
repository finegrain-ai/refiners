---
icon: material/family-tree
---

# Chain


When we say models are implemented in a declarative way in Refiners, what this means in practice is they are implemented as Chains. [`Chain`][refiners.fluxion.layers.Chain] is a Python class to implement trees of modules. It is a subclass of Refiners' [`Module`][refiners.fluxion.layers.Module], which is in turn a subclass of PyTorch's `Module`. All inner nodes of a Chain are subclasses of `Chain`, and leaf nodes are subclasses of Refiners' `Module`.

## A first example

To give you an idea of how it looks, let us take a simple convolution network to classify MNIST as an example. First, let us define a few variables.

```py
img_res = 28
channels = 128
kernel_size = 3
hidden_layer_in = (((img_res - kernel_size + 1) // 2) ** 2) * channels
hidden_layer_out = 200
output_size = 10
```

Now, here is the model in PyTorch:


```py
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, channels, kernel_size)
        self.linear_1 = nn.Linear(hidden_layer_in, hidden_layer_out)
        self.maxpool = nn.MaxPool2d(2)
        self.linear_2 = nn.Linear(hidden_layer_out, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.linear_2(x)
        return nn.functional.softmax(x, dim=0)
```

And here is how we could implement the same model in Refiners:

```py
class BasicModel(fl.Chain):
    def __init__(self):
        super().__init__(
            fl.Conv2d(1, channels, kernel_size),
            fl.ReLU(),
            fl.MaxPool2d(2),
            fl.Flatten(start_dim=1),
            fl.Linear(hidden_layer_in, hidden_layer_out),
            fl.ReLU(),
            fl.Linear(hidden_layer_out, output_size),
            fl.Lambda(lambda x: torch.nn.functional.softmax(x, dim=0)),
        )
```

!!! note
    We often use the namespace `fl` which means `fluxion`, which is the name of the part of Refiners that implements basic layers.

As of writing, Refiners does not include a `Softmax` layer by default, but as you can see you can easily call arbitrary code using [`fl.Lambda`][refiners.fluxion.layers.Lambda]. Alternatively, if you just wanted to write `Softmax()`, you could implement it like this:

```py
class Softmax(fl.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=0)
```

!!! note
    Notice the type hints here. All of Refiners' codebase is typed, which makes it a pleasure to use if your downstream code is typed too.

## Inspecting and manipulating

Let us instantiate the `BasicModel` we just defined and inspect its representation in a Python REPL:

```
>>> m = BasicModel()
>>> m
(CHAIN) BasicModel()
    ├── Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
    ├── ReLU() #1
    ├── MaxPool2d(kernel_size=2, stride=2)
    ├── Flatten(start_dim=1)
    ├── Linear(in_features=21632, out_features=200, device=cpu, dtype=float32) #1
    ├── ReLU() #2
    ├── Linear(in_features=200, out_features=10, device=cpu, dtype=float32) #2
    └── Softmax()
```

The children of a `Chain` are stored in a dictionary and can be accessed by name or index. When layers of the same type appear in the Chain, distinct suffixed keys are automatically generated.


```
>>> m[0]
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
>>> m.Conv2d
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
>>> m[6]
Linear(in_features=200, out_features=10, device=cpu, dtype=float32)
>>> m.Linear_2
Linear(in_features=200, out_features=10, device=cpu, dtype=float32)
```

The Chain class includes several helpers to manipulate the tree. For instance, imagine I want to organize my model by wrapping each layer of the convnet in a subchain. Here is how I could do it:


```py
class ConvLayer(fl.Chain):
    pass

class HiddenLayer(fl.Chain):
    pass

class OutputLayer(fl.Chain):
    pass

m.insert(0, ConvLayer(m.pop(0), m.pop(0), m.pop(0)))
m.insert_after_type(ConvLayer, HiddenLayer(m.pop(1), m.pop(1), m.pop(1)))
m.append(OutputLayer(m.pop(2), m.pop(2)))
```

Did it work? Let's see:

```
>>> m
(CHAIN) BasicModel()
    ├── (CHAIN) ConvLayer()
    │   ├── Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
    │   ├── ReLU()
    │   └── MaxPool2d(kernel_size=2, stride=2)
    ├── (CHAIN) HiddenLayer()
    │   ├── Flatten(start_dim=1)
    │   ├── Linear(in_features=21632, out_features=200, device=cpu, dtype=float32)
    │   └── ReLU()
    └── (CHAIN) OutputLayer()
        ├── Linear(in_features=200, out_features=10, device=cpu, dtype=float32)
        └── Softmax()
```

!!! note
    Organizing models like this is actually a good idea, it makes them easier to understand and adapt.

## Accessing and iterating

There are also many ways to access or iterate nodes even if they are deep in the tree. Most of them are implemented using a powerful iterator named [`walk`][refiners.fluxion.layers.Chain.walk]. However, most of the time, you can use simpler helpers. For instance, to iterate all the modules in the tree that hold weights (the `Conv2d` and the `Linear`s), we can just do:

```py
for x in m.layers(fl.WeightedModule):
    print(x)
```

It prints:

```
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
Linear(in_features=21632, out_features=200, device=cpu, dtype=float32)
Linear(in_features=200, out_features=10, device=cpu, dtype=float32)
```

---
icon: material/family-tree
---

# Chain


When we say models are implemented in a declarative way in Refiners, what this means in practice is they are implemented as Chains. [`Chain`][refiners.fluxion.layers.Chain] is a Python class to implement trees of modules. It is a subclass of Refiners' [`Module`][refiners.fluxion.layers.Module], which is in turn a subclass of PyTorch's `Module`. All inner nodes of a Chain are subclasses of `Chain`, and leaf nodes are subclasses of Refiners' `Module`.

## A first example

To give you an idea of how it looks, let us take an example similar to the one from the PyTorch paper[^1]:

```py
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        self.linear_1 = nn.Linear(128, 40)
        self.linear_2 = nn.Linear(40, 10)

    def forward(self, x):
        t1 = self.conv(x)
        t2 = nn.functional.relu(t1)
        t3 = self.linear_1(t2)
        t4 = self.linear_2(t3)
        return nn.functional.softmax(t4)
```

Here is how we could implement the same model in Refiners:

```py
class BasicModel(fl.Chain):
    def __init__(self):
        super().__init__(
            fl.Conv2d(1, 128, 3),
            fl.ReLU(),
            fl.Linear(128, 40),
            fl.Linear(40, 10),
            fl.Lambda(torch.nn.functional.softmax),
        )
```

!!! note
    We often use the namespace `fl` which means `fluxion`, which is the name of the part of Refiners that implements basic layers.

As of writing, Refiners does not include a `Softmax` layer by default, but as you can see you can easily call arbitrary code using [`fl.Lambda`][refiners.fluxion.layers.Lambda]. Alternatively, if you just wanted to write `Softmax()`, you could implement it like this:

```py
class Softmax(fl.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x)
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
    ├── ReLU()
    ├── Linear(in_features=128, out_features=40, device=cpu, dtype=float32) #1
    ├── Linear(in_features=40, out_features=10, device=cpu, dtype=float32) #2
    └── Softmax()
```

The children of a `Chain` are stored in a dictionary and can be accessed by name or index. When layers of the same type appear in the Chain, distinct suffixed keys are automatically generated.


```
>>> m[0]
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
>>> m.Conv2d
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
>>> m[3]
Linear(in_features=40, out_features=10, device=cpu, dtype=float32)
>>> m.Linear_2
Linear(in_features=40, out_features=10, device=cpu, dtype=float32)
```

The Chain class includes several helpers to manipulate the tree. For instance, imagine I want to wrap the two `Linear`s in a subchain. Here is how I could do it:


```py
m.insert_after_type(fl.ReLU, fl.Chain(m.pop(2), m.pop(2)))
```

Did it work? Let's see:

```
>>> m
(CHAIN) BasicModel()
    ├── Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
    ├── ReLU()
    ├── (CHAIN)
    │   ├── Linear(in_features=128, out_features=40, device=cpu, dtype=float32) #1
    │   └── Linear(in_features=40, out_features=10, device=cpu, dtype=float32) #2
    └── Softmax()
```

## Accessing and iterating

There are also many ways to access or iterate nodes even if they are deep in the tree. Most of them are implemented using a powerful iterator named [`walk`][refiners.fluxion.layers.Chain.walk]. However, most of the time, you can use simpler helpers. For instance, to iterate all the modules in the tree that hold weights (the `Conv2d` and the `Linear`s), we can just do:

```py
for x in m.layers(fl.WeightedModule):
    print(x)
```

It prints:

```
Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), device=cpu, dtype=float32)
Linear(in_features=128, out_features=40, device=cpu, dtype=float32)
Linear(in_features=40, out_features=10, device=cpu, dtype=float32
```

[^1]: Paszke et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library.

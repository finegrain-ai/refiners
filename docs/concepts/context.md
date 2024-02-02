---
icon: material/comment-alert-outline
---

# Context

## Motivation: avoiding "props drilling"

Chains are a powerful tool to represent computational graphs, but they are not always convenient.

Many adapters add extra input to the model. For instance, ControlNet and T2i-Adapter require a guide (condition image), inpainting adapters require a mask, Latent Consistency Models use a condition scale embedding, other adapters may leverage time or context embeddings... Those inputs are often passed by the user in a high-level format (numbers, text...) and converted to embeddings by the model before being consumed in downstream layers.

Managing this extra input is inconvenient. Typically, you would add them to the inputs and outputs of each layer somehow. But if you add them as channels or concatenate them you get composability issues, and if you try to pass them as extra arguments you end up needing to deal with them in layers that should not be concerned with their presence.

The same kind of having to pass extra contextual information up and down a tree exists in other fields, and in particular in JavaScript frameworks that deal with a Virtual DOM such as React, where it is called "props drilling". To make it easier to manage, the [Context API](https://react.dev/learn/passing-data-deeply-with-context) was introduced, and we went with a similar idea in Refiners.

## A simple example

Here is an example of how contexts work:


```py
from refiners.fluxion.context import Contexts

class MyProvider(fl.Chain):
    def init_context(self) -> Contexts:
        return {"my context": {"my key": None}}

m = MyProvider(
    fl.Chain(
        fl.Sum(
            fl.UseContext("my context", "my key"),
            fl.Lambda(lambda: 2),
        ),
        fl.SetContext("my context", "my key"),
    ),
    fl.Chain(
        fl.UseContext("my context", "my key"),
        fl.Lambda(print),
    ),
)

m.set_context("my context", {"my key": 4})
m()  # prints 6
```

As you can see, to use the context, you define it by subclassing any `Chain` and defining `init_context`. You can set the context with the [`set_context`][refiners.fluxion.layers.Chain.set_context] method or the [`SetContext`][refiners.fluxion.layers.SetContext] layer, and you can access it anywhere down the provider's tree with [`UseContext`][refiners.fluxion.layers.UseContext].

## Simplifying complex models with Context

Another use of the context is simplifying complex models, in particular those with long-range nested skip connections.

To emulate this, let us consider this toy example with a structure somewhat similar to a U-Net:

```py
square = fl.Lambda(lambda x: x ** 2)
sqrt = fl.Lambda(lambda x: x ** 0.5)

m1 = fl.Chain(
    fl.Residual(
        square,
        fl.Residual(
            square,
            fl.Residual(
                square,
            ),
            sqrt,
        ),
        sqrt,
    ),
    sqrt,
)
```

You can see two problems here:

- nesting is increasing 1 lever with every residual, in a real case this would become unreadable;
- you could not isolate the part that computes the squares (similar to down blocks in a U-Net) from the part that computes the square roots (similar to up blocks in a U-Net).

Let us solve those two issues using the context:

```py
from refiners.fluxion.context import Contexts

class MyModel(fl.Chain):
    def init_context(self) -> Contexts:
        return {"mymodel": {"residuals": []}}

push_residual = fl.SetContext("mymodel", "residuals", callback=lambda l, x: l.append(x))

class ApplyResidual(fl.Sum):
    def __init__(self):
        super().__init__(
            fl.Identity(),
            fl.UseContext("mymodel", "residuals").compose(lambda x: x.pop()),
        )

squares = fl.Chain(x for _ in range(3) for x in (push_residual, square))
sqrts = fl.Chain(x for _ in range(3) for x in (ApplyResidual(), sqrt))
m2 = MyModel(squares, sqrts)
```

As you can see, despite `squares` and `sqrts` being completely independent chains, they can access the same context due to being nested under the same provider.

Does it work?

```
>>> m1(2.0)
2.5547711633552384
>>> m2(2.0)
2.5547711633552384
```

Yes!✨

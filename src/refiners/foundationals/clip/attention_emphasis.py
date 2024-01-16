import re

import torch
from torch import Tensor, tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder, TokenEncoder
from refiners.foundationals.clip.tokenizer import CLIPTokenizer

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
    * (abc) - increases attention to abc by a multiplier of 1.0
    * (abc:3.12) - increases attention to abc by a multiplier of 3.12
    * \\( - literal character '('
    * \\) - literal character ')'
    * \\ - literal character '\'
    * anything else - just text with a multiplier of 1.0

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important:1.1) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('\\(literal\\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary:1.1)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (house:1.3) on a (hill:0.5), sun, (sky:1.5).')
    [['a ', 1.0],
     ['house', 1.3],
     [' on a ', 1.0],
     ['hill', 0.5],
     [', sun, ', 1.0],
     ['sky', 1.5],
     ['.', 1.0]]
    """

    res: list[tuple[str, float]] = []
    round_brackets: list[int] = []

    def multiply_range(start_position: int, multiplier: float):
        for p in range(start_position, len(res)):
            res[p] = (res[p][0], res[p][1] * multiplier)

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append((text[1:], 1.0))
        elif text == "(":
            round_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            round_brackets.pop()
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(("BREAK", -1))
                res.append((part, 1.0))

    if len(res) == 0:
        res = [("", 1.0)]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i] = (res[i][0] + res[i + 1][0], res[i][1] + res[i + 1][1])
            res.pop(i + 1)
        else:
            i += 1

    return res


class TokenEmphasisExtender(fl.Chain, Adapter[CLIPTokenizer]):
    def parse_prompt(self, prompt: str) -> Tensor:
        parsed_attention = parse_prompt_attention(prompt)

        result: list[Tensor] = []
        for text, emphasis in parsed_attention:
            token_length = self.target.encode(text).shape[1]
            result.append(tensor([emphasis] * token_length))

        return torch.stack(result)

    def __init__(self, target: CLIPTokenizer) -> None:
        with self.setup_adapter(target):
            super().__init__(
                target,
                fl.Passthrough(
                    fl.Lambda(self.parse_prompt),
                    fl.SetContext(
                        context="prompt_emphasis",
                        key="matrix",
                    ),
                ),
            )


class PromptEmphasisTokenExtender(fl.Chain, Adapter[TokenEncoder]):
    def combine(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        z, token_weights = inputs

        original_mean = z.mean()
        z *= token_weights
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        return z

    def __init__(self, target: TokenEncoder) -> None:
        with self.setup_adapter(target):
            super().__init__(
                fl.Parallel(
                    target,
                    fl.UseContext(
                        context="prompt_emphasis",
                        key="matrix",
                    ),
                ),
                fl.Lambda(func=self.combine),
            )


class PromptEmphasisExtender(fl.Chain, Adapter[CLIPTextEncoder]):
    def __init__(self, target: CLIPTextEncoder) -> None:
        with self.setup_adapter(target):
            super().__init__(target)

        try:
            token_encoder, token_encoder_parent = next(target.walk(TokenEncoder))
            self._token_encoder_parent = [token_encoder_parent]

        except StopIteration:
            raise RuntimeError("TokenEncoder not found.")

        try:
            clip_tokenizer, clip_tokenizer_parent = next(target.walk(CLIPTokenizer))
            self._clip_tokenizer_parent = [clip_tokenizer_parent]
        except StopIteration:
            raise RuntimeError("Tokenizer not found.")

        self._embedding_extender = [PromptEmphasisTokenExtender(token_encoder)]
        self._token_extender = [TokenEmphasisExtender(clip_tokenizer)]

    @property
    def embedding_extender(self) -> PromptEmphasisTokenExtender:
        assert len(self._embedding_extender) == 1, "EmbeddingExtender not found."
        return self._embedding_extender[0]

    @property
    def token_extender(self) -> TokenEmphasisExtender:
        assert len(self._token_extender) == 1, "TokenExtender not found."
        return self._token_extender[0]

    @property
    def token_encoder_parent(self) -> fl.Chain:
        assert len(self._token_encoder_parent) == 1, "TokenEncoder parent not found."
        return self._token_encoder_parent[0]

    @property
    def clip_tokenizer_parent(self) -> fl.Chain:
        assert len(self._clip_tokenizer_parent) == 1, "Tokenizer parent not found."
        return self._clip_tokenizer_parent[0]

    def inject(self: "PromptEmphasisExtender", parent: fl.Chain | None = None) -> "PromptEmphasisExtender":
        self.token_extender.inject(self.clip_tokenizer_parent)
        self.embedding_extender.inject(self.token_encoder_parent)
        return super().inject(parent)

    def eject(self) -> None:
        self.token_extender.eject()
        self.embedding_extender.eject()
        super().eject()

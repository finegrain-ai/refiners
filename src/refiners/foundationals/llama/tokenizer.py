import gzip
import json
import re
from pathlib import Path
from typing import Dict, TypedDict

from numpy import round
from torch import Tensor, tensor

import refiners.fluxion.layers as fl


class TokenDict(TypedDict):
    id: int
    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool


class LlamaTokenizer(fl.Module):
    """
    Implement a Unigram Tokenizer based on a vocabulary file for Llama (duplicated from Fuyu
    since its the same one used in HF)
    """

    def __init__(
        self,
        vocabulary_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        if vocabulary_path is None:
            vocabulary_path = Path.home() / ".cache/refiners/fuyu8b/tokenizer.json.gz"
            if not vocabulary_path.exists():
                raise FileNotFoundError(f"Vocabulary file not found at {vocabulary_path}")

        with gzip.open(vocabulary_path, "rt", encoding="utf-8") as f:
            config = json.load(f)

        self.vocabulary_path = vocabulary_path

        # for normalization
        self.prepend_char: str = config["normalizer"]["normalizers"][0]["prepend"]
        self.replace_pattern: str = config["normalizer"]["normalizers"][1]["pattern"]["String"]
        self.replace_char: str = config["normalizer"]["normalizers"][1]["content"]

        # dict for tokenization
        self.token_to_log_proba: Dict[str, float] = {token: log_proba for token, log_proba in config["model"]["vocab"]}
        self.token_to_id: Dict[str, int] = {token: i for i, (token, _) in enumerate(config["model"]["vocab"])}
        self.id_to_token: Dict[int, str] = {i: token for token, i in self.token_to_id.items()}

        # special tokens
        self.unknown_token: TokenDict = config["added_tokens"][0]
        self.pad_token: TokenDict = self.unknown_token
        self.eos_token: TokenDict = config["added_tokens"][1]
        self.newline_model_token = "<0x0A>"  # \n token
        self.boa_token = "<0x04>"  # beginning of answer
        self.bos_token = "<s>"  # beginning of sentence
        self.speaker_token = "|SPEAKER|"
        self.newline_token = "|NEWLINE|"  # image new line

        # special text and tokens for position handling
        self.text_bbox_open = "<box>"
        self.text_bbox_close = "</box>"
        self.text_point_open = "<point>"
        self.text_point_close = "</point>"
        self.token_bbox_open = "<0x00>"  # <bbox>
        self.token_bbox_close = "<0x01>"  # </bbox>
        self.token_point_open = "<0x02>"  # <point>
        self.token_point_close = "<0x03>"  # </point>

        # normalization token ids
        self.prepend_char_id = self.token_to_id[self.prepend_char]
        # special token ids
        self.boa_token_id = self.token_to_id[self.boa_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.speaker_token_id = self.token_to_id[self.speaker_token]
        self.newline_token_id = self.token_to_id[self.newline_token]
        # special token ids for position handling
        self.token_bbox_open_id = self.token_to_id[self.token_bbox_open]
        self.token_bbox_close_id = self.token_to_id[self.token_bbox_close]
        self.token_point_open_id = self.token_to_id[self.token_point_open]
        self.token_point_close_id = self.token_to_id[self.token_point_close]

    def _calculate_best_segmentation(self, text: str) -> list[int]:
        """
        Calculates the best segmentation of the input text based on the maximum log probabilities.
        Receives:
            text (str): The input text to tokenize.
        Returns:
            list[int]: A list of token IDs representing the best segmentation of the input text.
        """
        N = len(text)
        dp = [float("-inf")] * (N + 1)
        backpointer = [-1] * (N + 1)
        dp[0] = 0

        for i in range(1, N + 1):
            for j in range(0, i):
                piece = text[j:i]
                if piece in self.token_to_log_proba:
                    prob = self.token_to_log_proba[piece] + dp[j]
                    if prob > dp[i]:
                        dp[i] = prob
                        backpointer[i] = j
                elif j == i - 1:  # Single character not in vocab, consider it as unk
                    prob = self.token_to_log_proba.get(self.unknown_token["content"], 0) + dp[j]
                    if prob > dp[i]:
                        dp[i] = prob
                        backpointer[i] = j

        tokens: list[int] = []
        i = N
        while i > 0:
            j = backpointer[i]
            token = text[j:i] if text[j:i] in self.token_to_id else self.unknown_token["content"]
            tokens.append(self.token_to_id.get(token, self.unknown_token["id"]))
            i = j

        tokens.reverse()
        return tokens

    def process_text(self, text: str) -> list[int]:
        """
        preprocess and tokenize text
        """
        normalized_text = (self.prepend_char + text).replace(self.replace_pattern, self.replace_char)
        normalized_text = normalized_text.replace("\n", "<0x0A>")
        tokens = self._calculate_best_segmentation(normalized_text)
        return tokens

    def process_points_coordinates(self, points_coordinates: str, scale_factor: float) -> list[int]:
        """
        preprocess and tokenize coordinates and points
        """
        split_points_coordinates = points_coordinates.split(",")
        split_points_coordinates = [
            float(point_coordinate.strip())
            for point_coordinate in split_points_coordinates
            if point_coordinate.strip() != ""
        ]
        assert len(split_points_coordinates) in [
            2,
            4,
        ], "A bounding box needs to contain 4 values, a point 2 and each value needs to be separated by a comma"

        split_points_coordinates = [
            str(round((point_coordinate / 2) * scale_factor).astype(int))
            for point_coordinate in split_points_coordinates
        ]

        tokens = (
            [self.token_to_id[self.token_bbox_open if len(split_points_coordinates) == 4 else self.token_point_open]]
            + [self.token_to_id[point_coordinate] for point_coordinate in split_points_coordinates]
            + [
                self.token_to_id[
                    self.token_bbox_close if len(split_points_coordinates) == 4 else self.token_point_close
                ]
            ]
        )

        return tokens

    def forward(self, text: str, scale_factor: float = 1) -> Tensor:
        """
        Encodes a string of text into a tensor of token IDs.
        This method applies text normalization, handles special tokens and then tokenizes the text using the best segmentation
        strategy based on unigram probabilities. The resulting tokens are converted into their corresponding
        token IDs and returned as a tensor.
        Receives:
            text (str): The text to encode.
            scale_factor (float): for eventually rescale coordinates of bbox or points in the text
        Returns:
            Tensor: A tensor containing the encoded token IDs.
        """

        text = text.replace(self.text_bbox_open, self.token_bbox_open)
        text = text.replace(self.text_bbox_close, self.token_bbox_close)
        text = text.replace(self.text_point_open, self.token_point_open)
        text = text.replace(self.text_point_close, self.token_point_close)

        regex_pattern = re.compile(
            f"({self.token_bbox_open}|{self.token_bbox_close}|{self.token_point_open}|{self.token_point_close})"
        )
        text_split = regex_pattern.split(text)

        tokens: list[int] = []

        for i, elem in enumerate(text_split):
            match elem:
                case "" | self.token_bbox_open | self.token_bbox_close | self.token_point_open | self.token_point_close:
                    continue

                case _ if i > 0 and text_split[i - 1] in [self.token_bbox_open, self.token_point_open]:
                    tokens += self.process_points_coordinates(elem, scale_factor=scale_factor)

                case _:
                    tokens += self.process_text(elem)

        return tensor(tokens).unsqueeze(dim=0)

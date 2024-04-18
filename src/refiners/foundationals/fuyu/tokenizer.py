import gzip
import json
import re
from pathlib import Path
from typing import List

from numpy import round
from torch import Tensor, tensor

import refiners.fluxion.layers as fl


class FuyuTokenizer(fl.Module):
    """
    Implement a Unigram Tokenizer based on a vocabulary file for Fuyu
    """
    def __init__(
            self, 
            vocabulary_path: str | Path = Path(__file__).resolve().parent / "tokenizer.json.gz",
        ):
        super().__init__()

        with gzip.open(vocabulary_path, 'rt', encoding='utf-8') as f:
            config = json.load(f)
        
        self.vocabulary_path=vocabulary_path
        #for normalization
        self.prepend_char = config['normalizer']['normalizers'][0]['prepend']
        self.replace_pattern = config['normalizer']['normalizers'][1]['pattern']['String']
        self.replace_char = config['normalizer']['normalizers'][1]['content']

        self.token_to_log_proba = {token: log_proba for token, log_proba in config['model']['vocab']}
        self.token_to_id = {token: i for i, (token, _) in enumerate(config['model']['vocab'])}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        #special tokens
        self.unknown_token = config["added_tokens"][0]
        self.pad_token = self.unknown_token
        self.eos_token = config["added_tokens"][1]
        self.newline_model_token = "<0x0A>" # \n token
        
        self.text_bbox_open = "<box>"
        self.text_bbox_close = "</box>"
        self.text_point_open = "<point>"
        self.text_point_close = "</point>"

        self.token_bbox_open = "<0x00>"  # <bbox>
        self.token_bbox_close = "<0x01>"  # </bbox>
        self.token_point_open = "<0x02>"  # <point>
        self.token_point_close = "<0x03>"  # </point>

        self.boa_token_id = self.token_to_id["<0x04>"] #beginning of answer
        self.bos_token_id = self.token_to_id["<s>"] #beginning of sentence
        self.speaker_token_id = self.token_to_id["|SPEAKER|"]
        self.newline_token_id = self.token_to_id["|NEWLINE|"] # image new line

    def _calculate_best_segmentation(self, text: str) -> List[int]:
        """
        Calculates the best segmentation of the input text based on the maximum log probabilities.

        Receives:
            text (str): The input text to tokenize.

        Returns:
            List[int]: A list of token IDs representing the best segmentation of the input text.
        """ 
        N = len(text)
        dp = [float('-inf')] * (N + 1)
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
                elif j == i-1:  # Single character not in vocab, consider it as unk
                    prob = self.token_to_log_proba.get(self.unknown_token['content'], 0) + dp[j]
                    if prob > dp[i]:
                        dp[i] = prob
                        backpointer[i] = j

        tokens = []
        i = N
        while i > 0:
            j = backpointer[i]
            token = text[j:i] if text[j:i] in self.token_to_id else self.unknown_token['content']
            tokens.append(self.token_to_id.get(token, self.unknown_token['id']))
            i = j

        # Append bos token
        tokens.reverse()
        return tokens
    
    def process_text(self, text):
        normalized_text = (self.prepend_char + text).replace(self.replace_pattern, self.replace_char)
        normalized_text = normalized_text.replace('\n', '<0x0A>')
        tokens = self._calculate_best_segmentation(normalized_text)
        return tokens
    
    def process_points_coordinates(self, points_coordinates, scale_factor):
        points_coordinates = points_coordinates.split(',')
        assert len(points_coordinates) in [2, 4], "A bounding box needs to contain 4 values, a point 2"

        points_coordinates = [float(point_coordinate.strip()) for point_coordinate in points_coordinates]
        for i in range(len(points_coordinates)):
            points_coordinates[i] = str(round((points_coordinates[i] / 2)*scale_factor).astype(int))

        tokens = [self.token_to_id[self.token_bbox_open if len(points_coordinates)==4 else self.token_point_open]] + \
            [self.token_to_id[point_coordinate] for point_coordinate in points_coordinates] + \
            [self.token_to_id[self.token_bbox_close if len(points_coordinates)==4 else self.token_point_close]]
        
        return tokens

    def encode(self, text: str, scale_factor: float = 1) -> Tensor:
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
        
        list_points_coordinates = []
        list_text = []

        for i, elem in enumerate(text_split):
            if len(elem)==0 or elem in [self.token_bbox_open, self.token_bbox_close, self.token_point_open, self.token_point_close]:
                continue
            elif i > 0 and text_split[i-1] in [self.token_bbox_open, self.token_point_open]:
                list_points_coordinates.append([elem, i])
            else:
                list_text.append([elem, i])
            
        tokens_text = []
        for text, i in list_text:
            tokens = self.process_text(text)
            tokens_text.append([tokens, i])
        
        tokens_points_coordinates = []
        for points_coordinates, i in list_points_coordinates:
            tokens = self.process_points_coordinates(points_coordinates, scale_factor)
            tokens_points_coordinates.append([tokens, i])

        tokens = tokens_text + tokens_points_coordinates
        tokens = sorted(tokens, key=lambda x: x[1])
        tokens = [token[0] for token in tokens]
        tokens = [token for list_token in tokens for token in list_token]
        return tensor(tokens).unsqueeze(dim=0)
    
    def forward(self, text: str, scale_factor: float = 1) -> Tensor:
        return self.encode(text, scale_factor)
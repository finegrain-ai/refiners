import gzip
import json
from pathlib import Path
from typing import List

from torch import Tensor, tensor

import refiners.fluxion.layers as fl


class FuyuTokenizer(fl.Module):
    """
    Implement a Unigram Tokenizer based on a vocabulary file
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
    
    def encode(self, text: str) -> Tensor:
        """
        Encodes a string of text into a tensor of token IDs.

        This method applies text normalization and then tokenizes the text using the best segmentation
        strategy based on unigram probabilities. The resulting tokens are converted into their corresponding
        token IDs and returned as a tensor.

        Receives:
            text (str): The text to encode.

        Returns:
            Tensor: A tensor containing the encoded token IDs.
        """
        normalized_text = (self.prepend_char + text).replace(self.replace_pattern, self.replace_char)
        normalized_text = normalized_text.replace('\n', '<0x0A>')
        tokens = self._calculate_best_segmentation(normalized_text)
        return tensor(tokens).unsqueeze(dim=0)
    
    def forward(self, text: str) -> Tensor:
        return self.encode(text)
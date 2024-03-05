import gzip
import json
from pathlib import Path
from typing import List

from torch import Tensor, cat, tensor

import refiners.fluxion.layers as fl
from refiners.fluxion import pad


class FuyuTokenizer(fl.Module):
    def __init__(
            self, 
            vocabulary_path: str,
        ):
        super().__init__()

        with gzip.open(vocabulary_path, 'rt', encoding='utf-8') as f:
            config = json.load(f)
        
        self.vocabulary_path=vocabulary_path

        #special tokens
        self.unknown_token = config['added_tokens'][0]
        self.pad_token = self.unknown_token
        # self.end_of_sentence_token = config['added_tokens'][1]
        self.eos_token = '<0x04>'
        self.bos_token = '<s>'
        self.speaker = "|SPEAKER|"
        self.newline = "|NEWLINE|"

        #for normalization
        self.prepend_char = config['normalizer']['normalizers'][0]['prepend']
        self.replace_pattern = config['normalizer']['normalizers'][1]['pattern']['String']
        self.replace_char = config['normalizer']['normalizers'][1]['content']

        self.token_to_log_proba = {token: log_proba for token, log_proba in config['model']['vocab']}
        self.token_to_id = {token: i for i, (token, _) in enumerate(config['model']['vocab'])}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def _calculate_best_segmentation(self, text: str) -> List[int]: 
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
            token = text[j:i] if text[j:i] in self.token_to_id else self.unk['content']
            tokens.append(self.token_to_id.get(token, self.unknown_token['id']))
            i = j

        # Append bos token
        tokens.append(self.token_to_id[self.bos_token])
        tokens.reverse()
        # Append eos token
        tokens.append(self.token_to_id[self.eos_token])
        return tokens
    
    def encode(self, text: str) -> Tensor:
        normalized_text = (self.prepend_char + text).replace(self.replace_pattern, self.replace_char)
        tokens = self._calculate_best_segmentation(normalized_text)
        return tensor(tokens).unsqueeze(dim=0)
    
    def forward(self, text: str | list[str]) -> Tensor:
        if isinstance(text, str):
            return self.encode(text)
        else:
            assert isinstance(text, list), f"Expected type `str` or `list[str]`, got {type(text)}"
            tokens = [self.encode(txt) for txt in text]
            sequence_len = max(sequence.shape[1] for sequence in tokens)
            padded_tokkens = [
                pad(x=sequence, pad=(sequence_len - sequence.shape[1], 0), value=self.token_to_id[self.bos_token]) 
                for sequence in tokens
                ]
            padded_tokkens = cat(padded_tokkens)

        return(padded_tokkens)

    def decode(self, tokens: list[int]) -> str:
        # Reverse mapping from token ID to token
        id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        decoded_tokens = []
        for token_id in tokens:
            # Skip special tokens like unknown, pad, or eos in the decoded output
            if token_id in [self.unknown_token['id'], self.pad_token['id'], self.token_to_id[self.eos_token], self.token_to_id[self.bos_token]]:
                continue
            token = id_to_token.get(token_id, self.unknown_token['content'])  # Use unk content if token_id is not found
            decoded_tokens.append(token)
        
        # Join all tokens to form the original string
        decoded_text = ''.join(decoded_tokens).replace(self.prepend_char, ' ').strip()
        
        return decoded_text
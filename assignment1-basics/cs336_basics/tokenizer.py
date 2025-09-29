from collections.abc import Iterable

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        self.inverse_vocab = {v: k for k, v in vocab.items()}

        self.special_token_ids = {}
        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")

            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token_str] = self.inverse_vocab[token_bytes]

        self.merge_priority = {}

        for i, (token1, token2) in enumerate(merges):
            self.merge_priority[(token1, token2)] = i

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        token_ids = []

        if self.special_tokens:
            special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            chunks = re.split(f"({special_pattern})", text)

        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_token_ids:
                token_ids.append(self.special_token_ids[chunk])
                continue

            for match in re.finditer(PAT, chunk):
                pre_token = match.group()
                pre_token_bytes = pre_token.encode("utf-8")

                token_bytes = self._apply_merges(pre_token_bytes)

                token_ids.extend(self._bytes_to_ids(token_bytes))

        return token_ids

    def _apply_merges(self, text_bytes: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in text_bytes]

        if len(tokens) < 2:
            return tokens

        while True:
            best_pair = None
            best_priority = float("inf")
            best_pos = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]

                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_pos = i

            if best_pair is None:
                break

            merged_token = best_pair[0] + best_pair[1]
            tokens = tokens[:best_pos] + [merged_token] + tokens[best_pos + 2 :]

        return tokens

    def _bytes_to_ids(self, token_bytes: list[bytes]) -> list[int]:
        return [self.inverse_vocab[token] for token in token_bytes]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""

        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)

        return byte_sequence.decode("utf-8", errors="replace")

    def encode_iterable(self, text_stream: Iterable[str]) -> Iterable[int]:
        for text in text_stream:
            yield from self.encode(text)

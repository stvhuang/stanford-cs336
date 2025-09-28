import os
from collections import Counter, defaultdict

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = []

    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    for special_token in special_tokens:
        special_token_bytes = special_token.encode("utf-8")
        vocab[next_token_id] = special_token_bytes
        next_token_id += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(f"({special_pattern})", text)
        text_chunks = [chunk for chunk in text_chunks if chunk and chunk not in special_tokens]

    else:
        text_chunks = [text]

    word_freqs = defaultdict(int)

    for chunk in text_chunks:
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            word = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            word_freqs[word] += 1

    merges = []

    for _ in range(num_merges):
        stats = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                stats[pair] += freq

        if not stats:
            break

        most_common_pair = max(stats, key=lambda x: (stats[x], x[0], x[1]))
        merges.append(most_common_pair)

        merged_token = most_common_pair[0] + most_common_pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1

        new_word_freqs = {}

        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue

            if most_common_pair[0] not in word:
                new_word_freqs[word] = freq
                continue

            new_word = []
            i = 0
            changed = False

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(merged_token)
                    i += 2
                    changed = True

                else:
                    new_word.append(word[i])
                    i += 1

            if changed:
                new_word_freqs[tuple(new_word)] = freq

            else:
                new_word_freqs[word] = freq

        word_freqs = new_word_freqs

    return vocab, merges

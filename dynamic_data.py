from __future__ import annotations

import glob
import gzip
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from sentencepiece import sentencepiece_model_pb2 as sp_model
from torch import Tensor


SPIECE_UNDERLINE = "\u2581"
UNKNOWN_TOKEN_ID = 0
FIRST_ENCODEABLE_TOKEN_ID = 3
TRIE_TERMINAL = -1


def count_matching_files(pattern: str) -> int:
    return len(sorted(glob.glob(pattern)))


def load_raw_text_file(path: Path) -> str:
    suffixes = path.suffixes
    if suffixes[-2:] == [".txt", ".gz"] or suffixes[-1:] == [".gz"]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    if suffixes[-2:] == [".txt", ".zst"] or suffixes[-1:] == [".zst"]:
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Reading {path} requires the optional 'zstandard' package, which is not installed."
            ) from exc
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                return reader.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def _piece_to_bytes(piece: str, piece_type: int) -> bytes | None:
    if piece_type in (
        sp_model.ModelProto.SentencePiece.UNKNOWN,
        sp_model.ModelProto.SentencePiece.CONTROL,
        sp_model.ModelProto.SentencePiece.UNUSED,
    ):
        return None
    if piece_type == sp_model.ModelProto.SentencePiece.BYTE:
        if not (piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6):
            raise ValueError(f"Unexpected byte piece format: {piece!r}")
        return bytes([int(piece[3:5], 16)])
    return piece.encode("utf-8")


class TrieTokenizer:
    """
    Longest-prefix-match tokenizer over the SentencePiece-normalized byte stream.

    This does not try to replicate SentencePiece's Viterbi search. It implements the
    DGT-v1 tokenizer rule directly: a hot-swappable trie over normalized bytes.
    """

    def __init__(self, vocabulary: dict[bytes, int], id_to_bytes: dict[int, bytes]):
        self._validate_vocabulary(vocabulary, id_to_bytes)
        self.vocabulary = dict(vocabulary)
        self.id_to_bytes = dict(id_to_bytes)
        self.trie = self._build_trie(self.vocabulary)

    @classmethod
    def from_sentencepiece_model(cls, model_path: str | Path) -> "TrieTokenizer":
        proto = sp_model.ModelProto()
        with open(model_path, "rb") as f:
            proto.ParseFromString(f.read())
        cls._validate_sentencepiece_config(proto, model_path)

        vocabulary: dict[bytes, int] = {}
        id_to_bytes: dict[int, bytes] = {}
        for token_id, piece_proto in enumerate(proto.pieces):
            piece_bytes = _piece_to_bytes(piece_proto.piece, piece_proto.type)
            if piece_bytes is None:
                continue
            if piece_bytes in vocabulary and vocabulary[piece_bytes] != token_id:
                raise ValueError(
                    f"Duplicate byte sequence {piece_bytes!r} in {model_path}: "
                    f"token_ids {vocabulary[piece_bytes]} and {token_id}"
                )
            vocabulary[piece_bytes] = token_id
            id_to_bytes[token_id] = piece_bytes
        return cls(vocabulary=vocabulary, id_to_bytes=id_to_bytes)

    @staticmethod
    def _validate_sentencepiece_config(proto: sp_model.ModelProto, model_path: str | Path) -> None:
        normalizer = proto.normalizer_spec
        trainer = proto.trainer_spec
        unsupported = []
        if normalizer.name != "identity":
            unsupported.append(f"normalizer.name={normalizer.name!r}")
        if not normalizer.add_dummy_prefix:
            unsupported.append("normalizer.add_dummy_prefix=False")
        if not normalizer.remove_extra_whitespaces:
            unsupported.append("normalizer.remove_extra_whitespaces=False")
        if not normalizer.escape_whitespaces:
            unsupported.append("normalizer.escape_whitespaces=False")
        if not trainer.byte_fallback:
            unsupported.append("trainer.byte_fallback=False")
        if not trainer.split_by_whitespace:
            unsupported.append("trainer.split_by_whitespace=False")
        if trainer.treat_whitespace_as_suffix:
            unsupported.append("trainer.treat_whitespace_as_suffix=True")
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(
                f"TrieTokenizer only supports the identity + byte_fallback SentencePiece "
                f"configuration used by Gravity Tokenizer models. Unsupported in {model_path}: {joined}"
            )

    @staticmethod
    def _validate_vocabulary(vocabulary: dict[bytes, int], id_to_bytes: dict[int, bytes]) -> None:
        for byte_value in range(256):
            token_id = vocabulary.get(bytes([byte_value]))
            if token_id is None:
                raise ValueError(f"Vocabulary is missing byte fallback for 0x{byte_value:02X}")
        if len(set(id_to_bytes)) != len(id_to_bytes):
            raise ValueError("Duplicate token ids in vocabulary")

    @staticmethod
    def _build_trie(vocabulary: dict[bytes, int]) -> dict[int, dict]:
        root: dict[int, dict] = {}
        for byte_seq, token_id in vocabulary.items():
            node = root
            for byte_value in byte_seq:
                node = node.setdefault(byte_value, {})
            node[TRIE_TERMINAL] = token_id
        return root

    @staticmethod
    def normalize_text(text: str) -> bytes:
        """
        Reproduce the SentencePiece normalization contract used by the static model:
        - identity normalizer
        - add_dummy_prefix
        - remove_extra_whitespaces
        - escape_whitespaces

        This treats ASCII spaces specially while leaving tabs/newlines intact, which
        matches the Gravity tokenizer models we inspected.
        """
        if not text:
            return b""

        out: list[str] = []
        saw_nonspace = False
        pending_space = False
        for ch in text:
            if ch == " ":
                if saw_nonspace:
                    pending_space = True
                continue
            if not saw_nonspace:
                out.append(SPIECE_UNDERLINE)
                saw_nonspace = True
            elif pending_space:
                out.append(SPIECE_UNDERLINE)
                pending_space = False
            out.append(ch)
        if not out:
            return b""
        return "".join(out).encode("utf-8")

    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        return self.encode_normalized_bytes(normalized)

    def encode_normalized_bytes(self, normalized: bytes) -> list[int]:
        tokens: list[int] = []
        i = 0
        while i < len(normalized):
            node = self.trie
            best_token_id = UNKNOWN_TOKEN_ID
            best_match_len = 0
            j = i
            while j < len(normalized):
                byte_value = normalized[j]
                next_node = node.get(byte_value)
                if next_node is None:
                    break
                node = next_node
                j += 1
                terminal_token_id = node.get(TRIE_TERMINAL)
                if terminal_token_id is not None:
                    best_token_id = terminal_token_id
                    best_match_len = j - i
            if best_match_len == 0:
                best_token_id = self.vocabulary[bytes([normalized[i]])]
                best_match_len = 1
            tokens.append(best_token_id)
            i += best_match_len
        return tokens

    def mutate(self, removals: list[int], additions: dict[bytes, int]) -> None:
        new_vocab = dict(self.vocabulary)
        new_id_to_bytes = dict(self.id_to_bytes)
        for token_id in removals:
            byte_seq = new_id_to_bytes.pop(token_id, None)
            if byte_seq is not None:
                new_vocab.pop(byte_seq, None)
        for byte_seq, token_id in additions.items():
            new_vocab[byte_seq] = token_id
            new_id_to_bytes[token_id] = byte_seq
        self._validate_vocabulary(new_vocab, new_id_to_bytes)
        self.vocabulary = new_vocab
        self.id_to_bytes = new_id_to_bytes
        self.trie = self._build_trie(self.vocabulary)


class SentencePieceTokenizerAdapter:
    """
    Exact raw-text encoder backed by SentencePiece.

    This is useful during Phase 1 substrate validation, where we want JIT tokenization
    to match the existing pretokenized shards before switching over to the trie rule.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return self.processor.encode(text)

    def flush(self) -> None:
        return None


def build_raw_text_tokenizer(kind: str, model_path: str | Path):
    normalized_kind = kind.strip().lower()
    if normalized_kind == "sentencepiece":
        return SentencePieceTokenizerAdapter(model_path)
    if normalized_kind == "trie":
        return TrieTokenizer.from_sentencepiece_model(model_path)
    raise ValueError(
        f"Unsupported RAW_TOKENIZER_KIND={kind!r}; expected 'sentencepiece' or 'trie'"
    )


class RawTextTokenStream:
    """
    Sequentially reads raw text shards, tokenizes each shard with the current trie,
    and serves a deterministic token stream.

    Phase 1 keeps this exact and simple by tokenizing one shard at a time in memory.
    """

    def __init__(self, pattern: str, tokenizer):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.tokenizer = tokenizer
        self.file_idx = 0
        self.tokens = self._load_tokens_for_file(self.files[0])
        self.pos = 0

    def _load_tokens_for_file(self, file: Path) -> Tensor:
        text = load_raw_text_file(file)
        token_ids = self.tokenizer.encode(text)
        return torch.from_numpy(np.asarray(token_ids, dtype=np.uint16))

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = self._load_tokens_for_file(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def flush(self) -> None:
        # Phase 1 conservative behavior: restart the current shard under the new vocabulary.
        self.tokens = self._load_tokens_for_file(self.files[self.file_idx])
        self.pos = 0


class DistributedJITTokenLoader:
    """
    Drop-in replacement for DistributedTokenLoader that tokenizes raw text shards on demand.
    """

    def __init__(self, pattern: str, tokenizer, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = RawTextTokenStream(pattern, tokenizer)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def flush(self) -> None:
        self.stream.flush()


def load_validation_tokens_from_raw(pattern: str, tokenizer, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = []
    for file in files:
        text = load_raw_text_file(file)
        token_ids = tokenizer.encode(text)
        tokens.append(torch.from_numpy(np.asarray(token_ids, dtype=np.uint16)))
    merged = torch.cat(tokens).contiguous()
    usable = ((merged.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return merged[: usable + 1]

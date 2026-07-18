"""English text frontend backed directly by the eSpeak-ng executable."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import TextConfig

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_IDS = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
_RECORD_SENTINEL = "zzbanhmirecordseparatorzz"
_ESPEAK_BATCH_SIZE = 256
_ESPEAK_FORMATTING_CHARACTERS = frozenset({"\u200d"})


def ipa_to_symbols(ipa: str) -> List[str]:
    """Convert eSpeak IPA output to model symbols.

    eSpeak ``--ipa=3`` inserts U+200D (ZERO WIDTH JOINER) inside diphthongs
    and affricates as a display hint. It has no acoustic duration and Piper's
    phoneme map does not assign it an id, so it must not become a MAS token.
    Word spaces and audible IPA modifiers (stress/length) remain intact.
    """
    return [
        symbol
        for symbol in ipa.strip()
        if symbol not in _ESPEAK_FORMATTING_CHARACTERS
    ]


def find_espeak() -> str:
    configured = os.environ.get("ESPEAK_NG_PATH")
    candidates = [
        configured,
        shutil.which("espeak-ng"),
        shutil.which("espeak"),
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "eSpeak-ng executable not found; install espeak-ng or set ESPEAK_NG_PATH"
    )


def espeak_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    version_line = (result.stdout or result.stderr).strip().splitlines()[0]
    # Windows appends an installation-specific data path. Excluding it keeps
    # cache/config fingerprints portable across Windows and Linux.
    return version_line.split("  Data at:", maxsplit=1)[0]


def phonemize_batch(
    texts: Sequence[str], config: TextConfig, executable: str | None = None
) -> Tuple[List[List[str]], Dict[str, str]]:
    """Phonemize a corpus in one eSpeak process while preserving record order."""
    if not texts:
        return [], {}
    if any("\n" in text or "\r" in text for text in texts):
        raise ValueError("metadata text cannot contain embedded newlines")
    executable = executable or find_espeak()
    sentinel_result = subprocess.run(
        [executable, "-q", "--ipa=3", "-v", config.language, _RECORD_SENTINEL],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sentinel_phonemes = " ".join(sentinel_result.stdout.splitlines()).strip()
    if not sentinel_phonemes:
        raise RuntimeError("eSpeak failed to phonemize the record sentinel")

    sequences: List[List[str]] = []
    for start in range(0, len(texts), _ESPEAK_BATCH_SIZE):
        sequences.extend(
            _phonemize_chunk(
                texts[start : start + _ESPEAK_BATCH_SIZE],
                config,
                executable,
                sentinel_phonemes,
            )
        )
    empty_indices = [index for index, sequence in enumerate(sequences) if not sequence]
    if empty_indices:
        raise RuntimeError(f"eSpeak produced empty phonemes at indices {empty_indices[:10]}")
    return sequences, {
        "backend": "espeak-ng-cli",
        "version": espeak_version(executable),
    }


def _phonemize_chunk(
    texts: Sequence[str],
    config: TextConfig,
    executable: str,
    sentinel_phonemes: str,
) -> List[List[str]]:
    """Phonemize a chunk, recursively isolating unusual eSpeak records."""
    # eSpeak emits a new output line for every clause, so commas and other
    # punctuation make line-count based matching invalid. A synthetic sentinel
    # normally gives an unambiguous record boundary.
    input_text = "\n".join(f"{text} {_RECORD_SENTINEL}" for text in texts) + "\n"
    result = subprocess.run(
        [executable, "-q", "--ipa=3", "-v", config.language, "--stdin"],
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    combined_output = " ".join(result.stdout.splitlines())
    parts = combined_output.split(sentinel_phonemes)
    if len(parts) == len(texts) + 1:
        return [ipa_to_symbols(part) for part in parts[:-1]]

    if len(texts) > 1:
        midpoint = len(texts) // 2
        return _phonemize_chunk(
            texts[:midpoint], config, executable, sentinel_phonemes
        ) + _phonemize_chunk(
            texts[midpoint:], config, executable, sentinel_phonemes
        )

    # Some punctuation/markup can cause eSpeak to consume the sentinel. For a
    # single isolated record, no boundary marker is needed.
    single = subprocess.run(
        [executable, "-q", "--ipa=3", "-v", config.language, texts[0]],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    phonemes = " ".join(single.stdout.splitlines()).strip()
    if not phonemes:
        raise RuntimeError(f"eSpeak produced no phonemes for text: {texts[0]!r}")
    return [ipa_to_symbols(phonemes)]


def build_vocabulary(sequences: Sequence[Sequence[str]]) -> Dict[str, int]:
    symbols = sorted({symbol for sequence in sequences for symbol in sequence})
    vocabulary = dict(SPECIAL_IDS)
    for symbol in symbols:
        if symbol not in vocabulary:
            vocabulary[symbol] = len(vocabulary)
    return vocabulary


def encode_phonemes(
    phonemes: Sequence[str], vocabulary: Mapping[str, int]
) -> List[int]:
    """Encode with Piper/VITS-style blank insertion after every phoneme."""
    ids = [int(vocabulary[BOS])]
    pad_id = int(vocabulary[PAD])
    unknown_id = int(vocabulary[UNK])
    for phoneme in phonemes:
        ids.append(int(vocabulary.get(phoneme, unknown_id)))
        ids.append(pad_id)
    ids.append(int(vocabulary[EOS]))
    return ids


def frontend_metadata(
    config: TextConfig, vocabulary: Mapping[str, int], backend: Mapping[str, str]
) -> Dict[str, Any]:
    canonical = json.dumps(
        {"config": config.__dict__, "vocabulary": vocabulary, "backend": backend},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "language": config.language,
        "phoneme_type": "espeak-ipa-codepoints",
        "backend": dict(backend),
        "num_symbols": len(vocabulary),
        "phoneme_id_map": {symbol: [value] for symbol, value in vocabulary.items()},
        "special_ids": dict(SPECIAL_IDS),
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

#!/usr/bin/env python3
"""Synthesize a custom sentence from a training checkpoint.

Usage:
    python3 -m piper_train.say \
        --checkpoint training_dir/lightning_logs/version_6/checkpoints/epoch=29-step=87060.ckpt \
        --text "Hello, this is a test." \
        --output test.wav
"""
import argparse
import logging

import torch
from piper_phonemize import phoneme_ids_espeak, phonemize_espeak

from .vits.lightning import VitsModel
from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav

_LOGGER = logging.getLogger("piper_train.say")


def text_to_phoneme_ids(text: str, language: str = "en-us"):
    sentences_phonemes = phonemize_espeak(text, language)
    phonemes = [p for sentence in sentences_phonemes for p in sentence]
    return phoneme_ids_espeak(phonemes)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="piper_train.say")
    parser.add_argument("--checkpoint", required=True, help="Path to model .ckpt")
    parser.add_argument("--text", required=True, help="Sentence to synthesize")
    parser.add_argument("--output", required=True, help="Path to write the .wav file")
    parser.add_argument("--language", default="en-us", help="espeak-ng voice/language")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.8)
    parser.add_argument("--speaker-id", type=int, default=None)
    args = parser.parse_args()

    phoneme_ids = text_to_phoneme_ids(args.text, args.language)
    _LOGGER.info("Text: %s", args.text)
    _LOGGER.info("Phoneme ids (%d): %s", len(phoneme_ids), phoneme_ids)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None, strict=False)
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()

    text = torch.LongTensor(phoneme_ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(phoneme_ids)])
    scales = [args.noise_scale, args.length_scale, args.noise_w]
    sid = torch.LongTensor([args.speaker_id]) if args.speaker_id is not None else None

    with torch.no_grad():
        audio = model(text, text_lengths, scales, sid=sid).detach().numpy()
    audio = audio_float_to_int16(audio)

    write_wav(args.output, args.sample_rate, audio)
    _LOGGER.info("Wrote %s (%.2f sec)", args.output, audio.shape[-1] / args.sample_rate)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Streamlit demo for the Banhmi-TTS "+All" (BigVGAN + VITS2 + F0) checkpoint.

Run from WSL (edgetts conda env), from any directory:

    conda activate edgetts
    streamlit run /home/dev/PitchFlowNet/webapp/app.py
"""
import io
import os
import pathlib
import sys
import time

# piper_train isn't pip-installed -- it only resolves as a module when
# src/python is on sys.path. `streamlit run` doesn't add the launch cwd
# (unlike `python -c`/`python -m`), so add it explicitly here.
_SRC_PYTHON = pathlib.Path(__file__).resolve().parent.parent / "src" / "python"
if str(_SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SRC_PYTHON))

import streamlit as st
import torch
from piper_phonemize import phoneme_ids_espeak, phonemize_espeak

from piper_train.vits.lightning import VitsModel
from piper_train.vits.utils import audio_float_to_int16
from piper_train.vits.wavfile import write as write_wav

# See piper_train/__main__.py: PyTorch >=2.6 defaults torch.load to
# weights_only=True, which rejects the PosixPath hparam in our checkpoints.
torch.serialization.add_safe_globals([pathlib.PosixPath])

DEFAULT_CHECKPOINT = (
    "/home/dev/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/"
    "checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt"
)
SAMPLE_RATE = 22050


@st.cache_resource(show_spinner="Loading Banhmi-TTS checkpoint...")
def load_model(checkpoint_path: str) -> VitsModel:
    model = VitsModel.load_from_checkpoint(
        checkpoint_path, dataset=None, strict=False, map_location="cpu"
    )
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    return model


def text_to_phoneme_ids(text: str, language: str = "en-us"):
    sentences_phonemes = phonemize_espeak(text, language)
    phonemes = [p for sentence in sentences_phonemes for p in sentence]
    return phoneme_ids_espeak(phonemes)


def synthesize(model: VitsModel, text: str, noise_scale, length_scale, noise_w):
    phoneme_ids = text_to_phoneme_ids(text)
    text_t = torch.LongTensor(phoneme_ids).unsqueeze(0)
    text_lengths = torch.LongTensor([len(phoneme_ids)])
    scales = [noise_scale, length_scale, noise_w]

    start = time.perf_counter()
    with torch.no_grad():
        audio = model(text_t, text_lengths, scales, sid=None).detach().numpy()
    infer_sec = time.perf_counter() - start

    audio = audio_float_to_int16(audio)
    duration_sec = audio.shape[-1] / SAMPLE_RATE
    rtf = infer_sec / duration_sec if duration_sec > 0 else 0.0

    buf = io.BytesIO()
    write_wav(buf, SAMPLE_RATE, audio)
    return buf.getvalue(), duration_sec, rtf, len(phoneme_ids)


st.set_page_config(page_title="Banhmi-TTS", page_icon="🥖")
st.title("Banhmi-TTS")
st.caption("Config \"+All\" (BigVGAN + VITS2 + F0), best checkpoint by val_loss_mel")

with st.sidebar:
    checkpoint_path = st.text_input("Checkpoint path", value=os.environ.get(
        "BANHMI_CHECKPOINT", DEFAULT_CHECKPOINT
    ))
    noise_scale = st.slider("Noise scale", 0.0, 1.5, 0.667, 0.01)
    length_scale = st.slider("Length scale (speed)", 0.5, 2.0, 1.0, 0.05)
    noise_w = st.slider("Noise W (duration variance)", 0.0, 1.5, 0.8, 0.01)

text = st.text_area("Text", value="Hello, this is a test of the Banhmi text to speech system.", height=120)

if st.button("Generate", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        model = load_model(checkpoint_path)
        with st.spinner("Synthesizing..."):
            audio_bytes, duration_sec, rtf, n_phonemes = synthesize(
                model, text, noise_scale, length_scale, noise_w
            )
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("Download WAV", audio_bytes, file_name="banhmi_tts.wav", mime="audio/wav")
        st.caption(f"{n_phonemes} phonemes · {duration_sec:.2f}s audio · RTF {rtf:.2f}")

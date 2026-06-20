# EdgeTTS

Capstone project: a modernized VITS/Piper-based text-to-speech model ("Piper-Modern"), targeting eventual Edge deployment.

Built on top of [Piper](https://github.com/rhasspy/piper) (MIT License, Copyright (c) 2022 Michael Hansen). Architecture changes vs. upstream Piper:
- Snake1d activation (BigVGAN-style) in the generator's residual blocks
- MultiResolutionDiscriminator (UnivNet-style)
- Transformer-in-Flow coupling layers (VITS2-style)
- DurationDiscriminator (VITS2-style)

See `src/python/piper_train/vits/` for the modified model code and `src/python/piper_train/say.py` for a custom-sentence inference script.

---

Original Piper README (upstream notice):

Development has moved: https://github.com/OHF-Voice/piper1-gpl

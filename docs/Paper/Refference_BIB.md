%% ===================== CORE ARCHITECTURE =====================
%% VITS
@inproceedings{bibvits,
  author    = "Kim, J. and Kong, J. and Son, J.",
  title     = "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech",
  booktitle = "Proceedings of the 38th International Conference on Machine Learning (ICML)",
  pages     = "5530--5540",
  year      = "2021",
  publisher = "PMLR",
  note      = "Preprint at \url{https://arxiv.org/abs/2106.06103}"
}

%% Piper
@misc{bibpiper,
  author = "Hansen, M.",
  title  = "Piper: A fast, local neural text-to-speech system",
  year   = "2023",
  note   = "GitHub repository, Open Home Foundation. \url{https://github.com/OHF-Voice/piper1-gpl}"
}

%% Glow-TTS (Monotonic Alignment Search)
@misc{bibglowtts,
  author = "Kim, J. and Kim, S. and Kong, J. and Yoon, S.",
  title  = "{Glow-TTS}: A generative flow for text-to-speech via monotonic alignment search",
  year   = "2020",
  note   = "Preprint at \url{https://arxiv.org/abs/2005.11129}"
}

%% HiFi-GAN
@inproceedings{bibhifigan,
  author    = "Kong, J. and Kim, J. and Bae, J.",
  title     = "{HiFi-GAN}: Generative adversarial networks for efficient and high fidelity speech synthesis",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  volume    = "33",
  pages     = "17022--17033",
  year      = "2020",
  note      = "Preprint at \url{https://arxiv.org/abs/2010.05646}"
}

%% VITS2
@inproceedings{bibvits2,
  author    = "Kong, J. and Park, J. and Kim, B. and Kim, J. and Kong, D. and Kim, S.",
  title     = "{VITS2}: Improving quality and efficiency of single-stage text-to-speech with adversarial learning and architecture design",
  booktitle = "Proceedings of Interspeech 2023",
  pages     = "4374--4378",
  year      = "2023",
  doi       = "10.21437/Interspeech.2023-534",
  note      = "Preprint at \url{https://arxiv.org/abs/2307.16430}"
}

%% BigVGAN (source of both Snake activation usage and the Multi-Resolution Discriminator)
@inproceedings{bibbigvgan,
  author    = "Lee, S. and Ping, W. and Ginsburg, B. and Catanzaro, B. and Yoon, S.",
  title     = "{BigVGAN}: A universal neural vocoder with large-scale training",
  booktitle = "The Eleventh International Conference on Learning Representations (ICLR)",
  year      = "2023",
  note      = "Preprint at \url{https://arxiv.org/abs/2206.04658}"
}

%% Snake periodic activation (underlying activation function adopted by BigVGAN)
@inproceedings{bibsnake,
  author    = "Ziyin, L. and Hartwig, T. and Ueda, M.",
  title     = "Neural networks fail to learn periodic functions and how to fix it",
  booktitle = "Advances in Neural Information Processing Systems (NeurIPS)",
  volume    = "33",
  pages     = "1583--1594",
  year      = "2020",
  note      = "Preprint at \url{https://arxiv.org/abs/2006.08195}"
}

%% FastPitch (source of the explicit per-phoneme F0/pitch predictor)
@inproceedings{bibfastpitch,
  author    = "{\L}a{\'n}cucki, A.",
  title     = "{FastPitch}: Parallel text-to-speech with pitch prediction",
  booktitle = "Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)",
  pages     = "6588--6592",
  year      = "2021",
  doi       = "10.1109/ICASSP39728.2021.9413889"
}

%% ===================== DATA / EVALUATION =====================

%% LJSpeech dataset
@misc{bibljspeech,
  author = "Ito, K. and Johnson, L.",
  title  = "The {LJ} Speech Dataset",
  year   = "2017",
  note   = "\url{https://keithito.com/LJ-Speech-Dataset/}"
}

%% Silero VAD
@misc{bibsilerovad,
  author = "{Silero Team}",
  title  = "{Silero VAD}: pre-trained enterprise-grade voice activity detector, number detector and language classifier",
  year   = "2024",
  note   = "GitHub repository. \url{https://github.com/snakers4/silero-vad}"
}

%% WORLD vocoder (DIO / StoneMask F0 extraction)
@article{bibworld,
  author  = "Morise, M. and Yokomori, F. and Ozawa, K.",
  title   = "{WORLD}: A vocoder-based high-quality speech synthesis system for real-time applications",
  journal = "IEICE Transactions on Information and Systems",
  volume  = "E99-D",
  number  = "7",
  pages   = "1877--1884",
  year    = "2016"
}

%% PESQ
@inproceedings{bibpesq,
  author    = "Rix, A. W. and Beerends, J. G. and Hollier, M. P. and Hekstra, A. P.",
  title     = "Perceptual evaluation of speech quality ({PESQ}) --- a new method for speech quality assessment of telephone networks and codecs",
  booktitle = "Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)",
  pages     = "749--752",
  year      = "2001"
}

%% STOI
@article{bibstoi,
  author  = "Taal, C. H. and Hendriks, R. C. and Heusdens, R. and Jensen, J.",
  title   = "An algorithm for intelligibility prediction of time-frequency weighted noisy speech",
  journal = "IEEE Transactions on Audio, Speech, and Language Processing",
  volume  = "19",
  number  = "7",
  pages   = "2125--2136",
  year    = "2011"
}

%% Whisper (WER evaluation)
@misc{bibwhisper,
  author = "Radford, A. and Kim, J. W. and Xu, T. and Brockman, G. and McLeavey, C. and Sutskever, I.",
  title  = "Robust speech recognition via large-scale weak supervision",
  year   = "2022",
  note   = "Preprint at \url{https://arxiv.org/abs/2212.04356}"
}

%% FastSpeech 2 (variance adaptor / pitch predictor concept)
@misc{bibfastspeech2,
  author = "Ren, Y. and Hu, C. and Tan, X. and Qin, T. and Zhao, S. and Zhao, Z. and Liu, T.-Y.",
  title  = "{FastSpeech 2}: Fast and high-quality end-to-end text to speech",
  year   = "2020",
  note   = "Preprint at \url{https://arxiv.org/abs/2006.04558}"
}

%% Tukey / IQR outlier method
@book{bibtukey,
  author    = "Tukey, J. W.",
  title     = "Exploratory Data Analysis",
  publisher = "Addison-Wesley",
  address   = "Reading, MA",
  year      = "1977"
}

%% UnivNet (source of the Multi-Resolution Spectrogram Discriminator, cited independently in Related Work)
@inproceedings{bibunivnet,
  author    = "Jang, W. and Lim, D. and Yoon, J. and Kim, B. and Kim, J.",
  title     = "{UnivNet}: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation",
  booktitle = "Proceedings of Interspeech 2021",
  pages     = "2207--2211",
  year      = "2021",
  doi       = "10.21437/Interspeech.2021-1016",
  note      = "Preprint at \url{https://arxiv.org/abs/2106.07889}"
}
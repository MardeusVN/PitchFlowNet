# REPORT 2: DATA TASKS
## Data Collection and Cleaning, Exploratory Data Analysis (EDA)
### Application: Text-to-Speech System Based on the VITS Architecture

---

## Abstract

This report presents the process of collecting, describing, cleaning, and exploratorily analyzing the dataset used to train a Text-to-Speech (TTS) model within the scope of this graduation project. The dataset used is a single-speaker speech corpus in the LJSpeech format, consisting of 13,100 (text, audio) pairs, with a total duration of approximately 23.92 hours. The report presents the theoretical basis for each data-processing step — including Voice Activity Detection, outlier removal via the Interquartile Range method, acoustic feature extraction (spectrogram, F0 pitch contour), and phonemic representation — and synthesizes quantitative and visual analysis results to assess the quality and suitability of the data for training a speech-generation model.

---

## 1. Data Descriptions

### 1.1 Sources

In deep-learning-based speech synthesis systems, training data plays a decisive role in the quality of the generated voice, since the model learns to map from the linguistic space (text/phonemes) to the acoustic space (waveform/spectrogram) entirely from observed sample pairs. For this reason, speech corpora used for TTS are typically recorded under controlled conditions (a soundproofed room, a single reader, text clearly segmented into sentences or clauses) in order to minimize noise and inconsistency across samples.

The dataset used in this project is organized according to the **LJSpeech format** — a data-organization standard widely used in TTS research, consisting of:

- A transcript mapping file, in which each line corresponds to a recording identifier and the text content that was read;
- A directory containing waveform audio files corresponding 1-to-1 with each line of text.

A defining characteristic of this corpus is that it is **single-speaker**: all recordings were made by a single reader, with text content drawn from non-fiction works. The choice of a single-speaker corpus is consistent with the goal of training one specific voice, as opposed to a multi-speaker model, which would require additional speaker-identity features (speaker embeddings).

### 1.2 Size and Format

The dataset under examination has the following scale and format:

| Attribute | Value |
|---|---|
| Text format | Delimited text, no header row, each line containing an identifier and sentence content |
| Audio format | Digitized waveform signal (PCM), mono channel, 16-bit resolution |
| Sample rate | 22,050 Hz |
| Number of utterances | 13,100 |
| Number of speakers | 1 |
| Total audio duration | Approximately 23.92 hours |
| Text volume | Approximately 1.4 MB |
| Audio volume | Approximately 3.6 GB |

After the preprocessing stage, the raw data is converted into an intermediate representation for training, comprising: (i) a corpus configuration descriptor file (containing sample-rate information, the phoneme-unit mapping table, and the number of speakers); (ii) a training-sample manifest file in JSON Lines format, with each line corresponding to a sample that has been quantized into feature fields; and (iii) pre-computed cached tensor files stored in binary form to avoid recomputation at every training epoch.

### 1.3 Features

The dataset's features can be categorized into two groups: **raw features** (obtained directly from the source) and **processed features** (extracted to serve model training).

**Raw feature group:**

| Feature | Data type | Description |
|---|---|---|
| Sample identifier | String | Key linking the text to the audio file |
| Transcript (text) | String (UTF-8) | Natural-language content to be synthesized |
| Audio signal | 16-bit integer array, time domain | Speech signal corresponding to the text |

**Processed feature group (direct model input):**

| Feature | Data type | Theoretical basis |
|---|---|---|
| Phoneme sequence | Sequence of integers (indices into the phoneme dictionary) | Represents text at the phonological level, helping the model generalize better than character-level representation, especially for languages with inconsistent reading rules |
| Normalized waveform | Array of real numbers in [-1, 1] | Signal after silence removal and rescaling to a common amplitude range |
| Spectrogram | 2D real-valued matrix (frequency × time) | Frequency-domain representation of the signal via the Short-Time Fourier Transform, serving as the training target for the audio-generation component |
| Pitch contour (F0) | Sequence of real numbers per time frame | Represents the fundamental frequency of the speech signal, reflecting prosody/intonation |
| Speaker ID | Integer (optional) | Used for multi-speaker models; carries no statistical meaning for the current single-speaker corpus |

---

## 2. Data Cleaning and Processing

### 2.1 Theoretical Basis and Processing Pipeline

Data cleaning in TTS tasks pursues three main objectives: (1) removing samples that are technically invalid (missing, corrupted, or empty files); (2) removing samples exhibiting a mismatch between text and audio (e.g., abnormal reading speed, suggestive of a text–audio pairing error); and (3) normalizing the signal and text into a unified representation space so the model can learn more effectively. The processing pipeline follows the steps below:

**Step 1 — File validity check.**
Each record is checked for the existence and size of its corresponding audio file. Records referencing a non-existent or zero-size file are removed from the dataset before subsequent processing steps, in order to avoid runtime errors and statistical distortion in later stages.

**Step 2 — Voice Activity Detection and silence trimming.**
A common challenge in recorded data is the presence of silence at the beginning and end of each recording, arising from the reader's reaction delay or the recording equipment. If not removed, such silences would distort statistics related to actual speaking duration (e.g., reading speed), while also wasting computational resources during training, since the model would have to process segments containing no phonetic information. To address this, the system applies a deep-learning-based voice activity detection model (specifically the Silero VAD architecture, based on a gated recurrent network), operating on the signal resampled to 16 kHz, to determine the boundaries between speech and non-speech segments. The portion of the signal lying outside the detected boundaries is discarded before the signal is resampled to the target rate used for training.

**Step 3 — Removal of samples with abnormal reading speed (outlier detection).**
The reading speed of a sample is defined as the ratio between the number of text characters (excluding punctuation, since punctuation marks are not vocalized and do not contribute to speaking duration) and the actual speaking duration (determined via VAD in Step 2). Statistically, the reading speed across samples from the same speaker is expected to follow a relatively narrow distribution; samples whose reading speed deviates too far from this distribution are typically indicative of text–audio pairing errors, segmentation errors, or recording noise. The **Interquartile Range (IQR)** method is applied to determine the removal threshold: letting Q1 and Q3 denote the first and third quartiles of the reading-speed distribution, IQR = Q3 − Q1, samples whose speed falls outside the interval [Q1 − k·IQR, Q3 + k·IQR] (with the coefficient k chosen empirically) are removed from the training set. This is a non-parametric outlier-detection method (making no assumption of a normal distribution), which is appropriate for duration/speed data that is typically skewed.

**Step 4 — Text normalization and phonemic representation.**
Text is normalized to a consistent case convention, then converted into a sequence of phonemic units via a grapheme-to-phoneme conversion tool. Representing text at the phoneme level, rather than the character level, helps the model avoid having to learn the irregular pronunciation rules of natural language (e.g., heteronyms, loanwords), thereby improving generalization. During this process, the system simultaneously tracks and records phonemic units that are not present in the predefined mapping dictionary (missing phonemes), to support quality review of the source text.

**Step 5 — Sample length capping.**
Samples whose phoneme sequence exceeds a maximum length threshold are excluded from training. This capping serves two purposes: reducing memory requirements when batching samples together — since memory consumption scales with the longest sequence in the batch — and removing abnormally long samples that may indicate data-segmentation errors.

**Step 6 — Cached-data integrity check.**
Since the computation of features (audio normalization, spectrogram, F0) is cached as binary files for reuse across training runs, these files must be periodically checked for integrity (e.g., due to interrupted write processes) and discarded if they cannot be read back, forcing the system to recompute from the raw data.

### 2.2 Challenges Encountered and Solutions

**Challenge: parsing text containing special characters.**
While reading the delimited text file, a challenge arose from the presence of double-quote characters within sentence content (e.g., quoted excerpts in the source text). When using a CSV parsing mechanism with default settings — in which the double-quote character is interpreted as a field-enclosing character (quote character) — an odd number of quote characters across the full text caused the parser to misinterpret field boundaries, leading to multiple consecutive lines being incorrectly merged into a single record. As a result, the number of records successfully read was significantly lower than the actual count, and the content of the merged records became inaccurate. The solution applied was to disable the field-enclosure handling mechanism (disabling quote-handling) when parsing this type of text, which restored the correct record count and content. This challenge demonstrates the importance of cross-validating the number of text records against the number of corresponding audio files as a basic but essential quality-control step.

**Challenge: measuring actual speaking duration.**
Directly using the audio file's duration (total number of samples divided by the sample rate) as a "speaking duration" metric leads to systematic bias, since it includes silence segments that carry no phonetic information. The solution was to separate two concepts: file duration and actual speech duration (determined via VAD), using only the latter for analyses related to reading speed.

**Challenge: frame misalignment between acoustic features.**
Different feature-extraction methods (the Fourier transform for the spectrogram, the DIO/StoneMask algorithms for the pitch contour) can produce slightly different numbers of time frames for the same signal segment, due to differences in how each algorithm handles boundary padding. If left unaddressed, this frame misalignment causes alignment errors when combining multiple features as model input. The solution was to apply a post-processing alignment step (trimming or padding using an edge-replication strategy) to ensure that all features of the same sample share the same number of time frames.

**Challenge: length heterogeneity among samples within a training batch.**
Due to the sequential nature of both text and audio, samples within the same training batch have varying lengths. The standard deep-learning solution is to zero-pad all samples in the batch to the length of the longest sample, while separately storing each sample's actual length so the model can disregard the padded portion during loss computation (loss masking).

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Objectives and Methodology

Exploratory data analysis (EDA) in the context of TTS model training aims to answer three core questions: (i) what is the distributional character of the data, and are there anomalies that require handling; (ii) are the assumptions of data consistency satisfied (e.g., the correlation between text length and audio duration); and (iii) how should training hyperparameters (batch size, filtering thresholds) be chosen based on the empirical characteristics of the data. The analysis was performed directly on the full set of 13,100 samples in the dataset.

### 3.2 Audio Duration Distribution

The duration of audio samples in the dataset exhibits the following statistics: minimum 1.11 seconds, maximum 10.10 seconds, mean 6.57 seconds, median 6.76 seconds, standard deviation 2.19 seconds.

| Duration range | Number of samples | Percentage |
|---|---|---|
| Under 2 seconds | 272 | 2.1% |
| 2–5 seconds | 3,041 | 23.2% |
| 5–8 seconds | 5,720 | 43.7% |
| 8–10 seconds | 3,877 | 29.6% |
| Over 10 seconds | 190 | 1.4% |

![Audio duration distribution](data_processing_assets/duration_hist.png)

*Figure 1. Histogram of the duration distribution across 13,100 audio samples.*

The results show that the duration distribution is concentrated in the 5–8 second range, with negligible skewness and no extreme outliers (e.g., samples with near-zero duration or durations exceeding several dozen seconds). This implies that the raw data had already been segmented at a relatively consistent sentence or clause granularity prior to release, which is a favorable precondition for model training without requiring re-segmentation.

### 3.3 Text Length Distribution

Text length (measured in number of characters) exhibits the following statistics: minimum 12 characters, maximum 187 characters, mean 99.9 characters, median 102 characters.

| Length range (characters) | Number of samples | Percentage |
|---|---|---|
| Under 50 | 1,147 | 8.8% |
| 50–100 | 5,012 | 38.3% |
| 100–150 | 6,189 | 47.2% |
| 150–200 | 752 | 5.7% |

![Text length distribution](data_processing_assets/text_length_hist.png)

*Figure 2. Histogram of text-length distribution (number of characters) across 13,100 samples.*

The text-length distribution is relatively symmetric, concentrated around 100–150 characters, corresponding to the length of a complete sentence or clause in natural-language text. No empty text samples or abnormally long samples were observed, once the effect of the parsing error discussed in Section 2.2 is excluded.

### 3.4 Correlation Between Text Length and Audio Duration

![Correlation between text length and audio duration](data_processing_assets/text_len_vs_duration.png)

*Figure 3. Scatter plot of text length versus audio duration on a subsample of 800 observations.*

The scatter plot shows a clear positive correlation trend between text length and audio duration, consistent with physical expectation: longer text requires more time to read aloud. The dispersion observed around the linear trend reflects natural variation in reading speed across segments (due to intonation, lexical complexity, or pauses within sentences). Data points that deviate substantially from the main trend — i.e., samples with an abnormal text-length-to-audio-duration ratio — are precisely the targets that the Interquartile-Range filtering method (Step 3, Section 2.1) is designed to detect and remove. This observation provides an intuitive quantitative basis that supports and confirms the validity of the reading-speed filtering criterion presented in the data-cleaning section.

### 3.5 Visual Analysis of an Individual Sample's Signal

![Waveform and spectrogram of a sample](data_processing_assets/example_waveform_spectrogram.png)

*Figure 4. Waveform (top) and time-frequency spectrogram (bottom) of a representative audio sample.*

Observation at the level of an individual sample clearly shows the presence of silence segments at the beginning and end of the recording — precisely confirming the necessity of the Voice Activity Detection step in the data-cleaning pipeline (Section 2.1, Step 2). The spectrogram shows energy concentrated mainly in the low-to-mid frequency range, characteristic of natural speech signals, with no broadband noise bands spread uniformly across the entire frequency spectrum (a common indicator of equipment or background noise), suggesting that the corpus's recording quality is relatively good.

### 3.6 Summary and Findings

The results of the exploratory data analysis support the following conclusions:

1. **Completeness and integrity**: within the scope examined, the dataset contains no missing or severely corrupted samples; the main challenge arose from a text-format issue (special characters in the CSV) rather than from the quality of the data content itself.
2. **Consistency**: both the audio-duration and text-length distributions are unimodal, free of extreme skew, and exhibit a physically reasonable correlation between the two data modalities (text and audio) — a necessary condition for a speech-generation model to learn a stable mapping.
3. **Suitability for training**: since the dataset is single-speaker, the system's multi-speaker handling mechanisms (speaker-ID assignment, balancing sample counts across speakers) have no statistical effect on the current data, but should be kept in mind when extending to multi-speaker corpora in future development stages.
4. **Limitations of the analysis**: the statistics presented are based on signal- and text-level features (duration, length) and do not yet include deeper analyses of phonemic distribution (e.g., frequency of occurrence of individual phonemic units) or pitch characteristics (the global F0 distribution) — these represent additional analytical directions that could be pursued in subsequent reports.

---

## Conclusion

This report has systematically presented the process of describing, cleaning, and exploratorily analyzing the dataset used to train a speech-synthesis model. Building on the theoretical foundations of speech-signal processing (voice activity detection, acoustic feature extraction) and descriptive-statistics/outlier-removal methods (the Interquartile Range), combined with quantitative and visual analysis results on real data, it can be concluded that the current dataset meets the basic quality requirements for model training, and that the challenges encountered during processing have been clearly diagnosed and addressed with corresponding solutions.

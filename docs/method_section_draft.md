# Method — Banhmi-TTS

---

## 3. Method

We build on VITS (Kim et al., 2021) as the base architecture and introduce three independently controlled extensions. Sections 3.1–3.3 describe the base model; Sections 3.4–3.6 describe each extension; Section 3.7 gives the unified training objective.

### 3.0. System Overview

**Training.** Given a phoneme sequence $c$ and its paired waveform $x$, the model runs two parallel encoders. The *text encoder* processes $c$ and produces a prior distribution $(\mu_p, \sigma_p)$ over the latent space together with a hidden state $\mathbf{x}$ used by the duration and F0 predictors. The *posterior encoder* processes the linear spectrogram of $x$ and produces an approximate posterior $(\mu_q, \sigma_q)$ from which a latent sample $z$ is drawn. A normalizing *flow* $f$ maps $z$ to the prior space, yielding $z_p = f(z)$; the KL divergence between the posterior and prior is then computed in this transformed space. Since the prior is defined at the phoneme level but $z_p$ lives at the frame level, the two must be aligned: *Monotonic Alignment Search* (MAS) finds the optimal monotonic phoneme-to-frame alignment $A^*$ by maximizing $\log \mathcal{N}(z_p; \mu_p, \sigma_p)$ over all valid alignments. $A^*$ serves two roles: it expands $(\mu_p, \sigma_p)$ to frame resolution for the KL loss, and it provides phoneme durations $\{d_i\}$ to supervise the *stochastic duration predictor* (SDP). The SDP receives the text hidden state $\mathbf{x}$ and observed durations and returns a duration negative log-likelihood $\mathcal{L}_\text{dur}$. Finally, a random 8,192-sample segment is sliced from $z$, decoded by the *HiFi-GAN decoder* to a waveform segment $\hat{y}$, and scored by the discriminators against the corresponding ground-truth segment $y$.

**Inference.** The posterior encoder and MAS are absent at inference. The text encoder produces $(\mu_p, \sigma_p, \mathbf{x})$ from the input phoneme sequence. The SDP samples stochastic durations from its learned prior, which determine a frame-length mask. A latent $z_p$ is sampled from $\mathcal{N}(\mu_p, \sigma_p)$ — where $(\mu_p, \sigma_p)$ have been expanded to frame resolution using the sampled durations — and mapped back through the inverse flow $f^{-1}$ to obtain $z$. The decoder then upsamples $z$ to the output waveform. When `use_f0 = true`, the F0 predictor additionally produces per-phoneme log-F0 predictions from $\mathbf{x}$; these are expanded to frame-level via the predicted durations and injected into the decoder as an additive conditioning signal before upsampling begins.

---

### 3.1. Variational Inference with Normalizing Flows

**Generative model.** Let $c$ denote the input phoneme sequence and $x$ the target waveform. We model the conditional distribution $p_\theta(x \mid c)$ by introducing a latent variable $z$:

$$p_\theta(x \mid c) = \int p_\theta(x \mid z)\, p_\theta(z \mid c)\, \mathrm{d}z$$

The prior $p_\theta(z \mid c)$ is an isotropic Gaussian whose parameters $(\mu_p, \sigma_p)$ are produced by a text encoder from the phoneme sequence. The decoder $p_\theta(x \mid z)$ is a HiFi-GAN–style upsampling network.

**Variational lower bound.** Because the marginal is intractable, we introduce an approximate posterior $q_\phi(z \mid x, c)$ and optimize the evidence lower bound (ELBO):

$$\log p_\theta(x \mid c) \geq \mathbb{E}_{q_\phi(z \mid x, c)}\!\left[\log p_\theta(x \mid z)\right] - D_{\mathrm{KL}}\!\left[q_\phi(z \mid x, c) \;\|\; p_\theta(z \mid c)\right]$$

The posterior $q_\phi(z \mid x, c)$ is parameterized by a WaveNet-based posterior encoder that takes the linear spectrogram of $x$ as input, producing $(\mu_q, \sigma_q)$ from which $z$ is reparameterized. The posterior encoder is used only during training.

**Normalizing flow.** To increase the expressiveness of the prior, a normalizing flow $f$ maps the posterior sample $z$ into the prior space, yielding $z_p = f(z)$. The KL term is then evaluated in the transformed space:

$$D_{\mathrm{KL}}\!\left[q_\phi \;\|\; p_\theta\right] = D_{\mathrm{KL}}\!\left[\mathcal{N}(\mu_q, \sigma_q^2) \;\|\; \mathcal{N}(\tilde{\mu}_p,\, \tilde{\sigma}_p^2)\right]$$

where $\tilde{\mu}_p = A^*\mu_p$ and $\tilde{\sigma}_p = A^*\sigma_p$ are the prior parameters expanded from phoneme space to frame space by the alignment matrix $A^*$ produced by MAS (Section 3.2). The flow $f$ consists of 4 affine coupling layers, each conditioned by a WaveNet stack (kernel 5, dilation 1, 4 residual layers). All coupling layers are volume-preserving (Jacobian determinant 1, mean-only affine transform), following Kim et al. (2021). When `use_vits2 = true`, each coupling layer's conditioner gains an additional self-attention pass before the WaveNet stack, as described in Section 3.5. At inference, $z_p$ is sampled from the prior $\mathcal{N}(\mu_p, \sigma_p^2)$ and passed through the inverse flow $f^{-1}$ to obtain $z$.

**Components.** The text encoder is a relative-position-encoding Transformer (hidden 192, filter 768, 6 layers, 2 heads, kernel 3, dropout 0.1) that outputs prior parameters $(\mu_p, \log\sigma_p)$ and hidden state $\mathbf{x}$. The posterior encoder is a dilated WaveNet stack (kernel 5, dilation 1, 16 layers, hidden 192) mapping the 513-bin linear spectrogram to $(\mu_q, \log\sigma_q)$. The decoder is described in Section 3.4.

---

### 3.2. Monotonic Alignment Search

Phoneme durations are not observed directly; instead, the alignment between phoneme positions and latent frames is estimated jointly with the model parameters using Monotonic Alignment Search (MAS; Kim et al., 2020). MAS finds the monotonic, non-skipping alignment $A^*$ that maximizes the log-likelihood of the posterior sample $z_p$ under the expanded prior:

$$A^* = \arg\max_{A \in \mathcal{A}} \sum_{j,i} A_{ji} \log \mathcal{N}\!\left(z_{p,j};\; \mu_{p,i},\, \sigma_{p,i}\right)$$

where $\mathcal{A}$ is the set of valid monotonic alignments (each frame assigned to exactly one phoneme, order preserved). This is solved exactly in $O(T_\text{frame} \times T_\text{phone})$ via dynamic programming. The resulting $A^*$ is detached from the computation graph, so no gradient flows back through the alignment step. The prior parameters are then expanded from phoneme space to frame space as $\tilde{\mu}_p = A^* \mu_p$ and $\tilde{\sigma}_p = A^* \sigma_p$, and the phoneme duration of token $i$ is $d_i = \sum_j A^*_{ji}$.

---

### 3.3. Stochastic Duration Predictor

The stochastic duration predictor (SDP) learns the distribution $p(d_i \mid \mathbf{x}_i)$ over phoneme durations using a normalizing flow, enabling diverse rhythm at inference via stochastic sampling rather than a single deterministic prediction.

**Architecture.** The SDP uses a coupling-based normalizing flow with 4 coupling layers. Unlike the main normalizing flow (Section 3.1), the SDP coupling layers use **rational-quadratic neural spline flows** (Durkan et al., 2019) instead of affine coupling, providing a more flexible and accurate density estimator for the skewed, discrete-like duration distribution. Each coupling layer's conditioner is a DDSConv network: 3 layers of dilated depth-wise separable convolutions (kernel 3, dilation $3^i$ for layer $i \in \{0,1,2\}$, GELU activation, LayerNorm, filter 192).

**Training.** During training, the SDP encodes the observed duration $d_i$ into a 2D latent via a second set of 4 spline coupling layers (the posterior flow, conditioned on both $\mathbf{x}_i$ and $d_i$). The SDP is trained by maximizing a variational lower bound on $\log p(d \mid \mathbf{x})$: the posterior flow encodes the observed duration into a latent whose log-likelihood under the generative flow, minus the variational correction, gives $\mathcal{L}_\text{dur}$.

**Inference.** A sample $z \sim \mathcal{N}(0, I)$ is passed through the inverse generative flow to obtain log-durations, from which integer durations are obtained by rounding.

---

### 3.4. BigVGAN Extensions: SnakeBeta Activation and Multi-Resolution Discriminator

**Decoder architecture.** The HiFi-GAN decoder (Kong et al., 2020) upsamples the latent $z$ to the output waveform through three transposed convolutions with rates $(8, 8, 4)$, starting from 256 channels and halving at each stage. After each upsample, a multi-receptive-field (MRF) block averages the outputs of three parallel ResBlock2 branches with kernel sizes $(3, 5, 7)$ and dilation groups $\{(1,2), (2,6), (3,12)\}$, accumulating multi-scale temporal context before the next upsample. A final $7 \times 1$ convolution with no bias produces the single-channel waveform through a tanh output.

**Motivation.** This decoder uses LeakyReLU activations, which are piecewise linear and cannot extrapolate periodic signals (Ziyin et al., 2020). Simultaneously, the MPD operates only in the time domain and is less sensitive to high-frequency spectral artifacts (Jang et al., 2021). We address both by integrating the two core components of BigVGAN (Lee et al., 2023).

**SnakeBeta activation.** We replace each LeakyReLU in the decoder with the SnakeBeta activation:

$$f_{\alpha,\beta}(x) = x + \frac{1}{\beta}\sin^2(\alpha x)$$

where $\alpha$ (frequency) and $\beta$ (magnitude scale) are independent learnable parameters per channel, both constrained positive via $|\cdot|$. The periodic $\sin^2$ term allows the activation to model oscillatory signals without requiring the network to synthesize periodicity from depth alone. SnakeBeta is inserted before each transposed convolution (upsampling stage) and replaces the final LeakyReLU before the output convolution.

**Multi-Period Discriminator (MPD).** Following Kim et al. (2021), the baseline discriminator consists of one scale sub-discriminator (DiscriminatorS, operating on raw waveforms) and five period sub-discriminators (DiscriminatorP, periods [2, 3, 5, 7, 11]), totalling 46.7 M parameters. This formulation is equivalent to a multi-period discriminator with periods [1, 2, 3, 5, 7, 11], where DiscriminatorS serves as the period-1 sub-discriminator.

**Multi-Resolution Discriminator (MRD).** We add a MultiResolutionDiscriminator alongside the existing MPD. The MRD consists of three DiscriminatorR sub-discriminators operating at STFT resolutions (FFT/hop/window) of $(512, 50, 240)$, $(1024, 120, 600)$, and $(2048, 240, 1200)$. Each DiscriminatorR computes the STFT magnitude spectrogram of the waveform and processes it as a 2D image through 5 Conv2d layers followed by one output convolution, using LeakyReLU activations and weight normalization. The MRD adds 280 K parameters and is present only during training; it adds zero cost to the deployed model since it shares no parameters with the generator.

---

### 3.5. VITS2 Extensions: Transformer Coupling, Duration Discriminator, and MAS Noise

We incorporate three targeted refinements from VITS2 (Kong et al., 2023), all gated by `use_vits2`.

**Transformer coupling layers.** In the base model, each coupling layer's conditioner is a WaveNet stack, whose local receptive field limits its ability to capture long-range phoneme dependencies. We augment the conditioner with a single self-attention layer (1-layer Transformer Encoder, 2 heads, hidden 192) applied before the WaveNet stack:

$$\mathbf{h} = \mathbf{h}_\text{pre} + \text{Attn}(\mathbf{h}_\text{pre})$$
$$\mathbf{h} = \text{WN}(\mathbf{h},\, \mathbf{x}_\text{mask})$$

The self-attention establishes global phoneme context before the WaveNet refines local structure. Invertibility is unaffected: the coupled half $x_0$ still passes through unchanged; only the function computing the affine shift from $x_0$ becomes more expressive. The coupling layers remain volume-preserving (`mean_only=True`).

**Duration discriminator.** The SDP's likelihood objective alone provides a weak training signal for duration naturalness. We add a DurationDiscriminator that scores each phoneme's predicted log-duration as real or fake, conditioned on the text encoder hidden state:

1. Two Conv1d layers encode the text hidden state $\mathbf{x}$ (kernel 3, hidden 192, LayerNorm, ReLU).
2. A 1×1 Conv1d projection encodes the duration input $d$.
3. Text and duration representations are concatenated and passed through two further Conv1d layers (kernel 3, LayerNorm, ReLU) that produce a per-phoneme real/fake score.

The discriminator scores real durations $d^\text{real} = \log(w + 10^{-6})$ (from MAS) against predicted durations $\hat{d} = \text{SDP}_\text{reverse}(\mathbf{x})$ and is discarded after training, adding zero inference cost.

**MAS noise annealing.** To prevent early-training alignment overfitting to a single path, we perturb the MAS score matrix $\mathbf{S}$ with adaptive Gaussian noise before the dynamic programming step:

$$\tilde{\mathbf{S}} = \mathbf{S} + \varepsilon, \quad \varepsilon \sim \mathcal{N}\!\left(0,\, \tau^2 \cdot \sigma(\mathbf{S})^2\right)$$

where $\sigma(\mathbf{S})$ is the standard deviation of the current batch's score matrix and the noise scale decays linearly as:

$$\tau = \max\!\left(0,\; 0.01 - t \times 2 \times 10^{-6}\right)$$

with $t$ the global training step. $\tau$ reaches zero at step 5,000 (approximately epoch 13), transitioning MAS from soft exploration in early training to hard Viterbi alignment thereafter.

**Speaker conditioning.** VITS2 adds speaker conditioning to the text encoder. In our single-speaker setup, `gin_channels = 0` so the speaker conditioning projection is not instantiated; this component is a no-op for all eight ablation configurations.

---

### 3.6. F0 Conditioning

**Motivation.** In the base model, F0 is encoded implicitly within the VAE latent $z$ together with timbre and all other acoustic properties. This implicit coupling can cause the model to average pitch contours across training utterances (prosody averaging), yielding monotonous synthesis. We introduce explicit per-phoneme F0 conditioning following FastPitch (Łańcucki et al., 2021), adapted from its feed-forward setting to the flow-based VITS decoder.

**F0 predictor.** A lightweight two-layer network regresses per-phoneme log-F0 from the text encoder hidden state $\mathbf{x}$:

$$\hat{f}_i = \text{proj}\!\left(\text{drop}\!\left(\text{LN}\!\left(\text{ReLU}\!\left(\text{Conv}_2(\cdot)\right)\right)\right)\right)$$

with hidden 256, kernel 3, dropout 0.5. The architecture mirrors the non-stochastic `DurationPredictor` (not the SDP described in Section 3.3) so that the same backbone receives the same input.

**Ground-truth supervision.** The per-phoneme F0 target $f_i^\text{gt}$ is derived from the frame-level F0 contour $\mathbf{f} \in \mathbb{R}^{T_\text{frame}}$ via the MAS alignment matrix $A^*$:

$$f_i^\text{gt} = \frac{\sum_j A^*_{ji}\, f_j}{d_i}, \quad d_i = \sum_j A^*_{ji}$$

This is the duration-weighted mean F0 within each phoneme's aligned frame span. Unvoiced frames have $f_j = 0$ by DIO convention and are excluded from the average by the division by $d_i$. The F0 loss is MSE in log space:

$$\mathcal{L}_{f0} = \frac{\sum_i \left(\hat{f}_i - \log\max(f_i^\text{gt}, 1)\right)^2 \cdot x_{\text{mask},i}}{\sum_i x_{\text{mask},i}}$$

**Decoder injection.** A zero-initialized Conv1d projection (`f0_cond`: $1 \to 256$ channels, kernel 1) maps a frame-level F0 signal (in Hz) into the decoder's channel space and adds it additively to the decoder input after the initial pre-convolution:

$$\mathbf{h}_\text{dec} = \text{Conv}_\text{pre}(z_\text{slice}) + f_\text{cond}(\mathbf{f}_\text{frame})$$

At *training* time, $\mathbf{f}_\text{frame}$ is the **ground-truth** frame-level F0 contour (in Hz, from the cached DIO output), sliced to the same 8,192-sample window as $z_\text{slice}$. The F0 predictor's output is used only to compute $\mathcal{L}_{f0}$ and does not condition the decoder during training. Zero initialization of `f0_cond` ensures that F0 conditioning starts as an exact no-op, so the baseline decoder converges before the additional signal is introduced.

At *inference* time, the F0 predictor produces per-phoneme log-F0 from $\mathbf{x}$; these are expanded to frame-level via the SDP-predicted alignment and exponentiated to Hz, matching the Hz-scale input that `f0_cond` was trained to receive.

---

### 3.7. Training Objective

The generator is optimized end-to-end via the following composite objective:

$$\mathcal{L}_\text{gen} = \mathcal{L}_\text{adv,MPD} + \mathcal{L}_\text{adv,MRD} + \mathcal{L}_\text{fm} + c_\text{mel}\,\mathcal{L}_\text{mel} + \mathcal{L}_\text{dur} + c_\text{kl}\,\mathcal{L}_\text{kl} + \mathcal{L}_\text{dur,gen} + \mathcal{L}_{f0}$$

where $c_\text{mel} = 45$ and $c_\text{kl} = 1.0$. The conditional terms are zeroed when the corresponding flag is inactive: $\mathcal{L}_\text{adv,MRD} = 0$ if `use_bigvgan = false`; $\mathcal{L}_\text{dur,gen} = 0$ if `use_vits2 = false`; $\mathcal{L}_{f0} = 0$ if `use_f0 = false`.

**Adversarial losses.** Waveform discrimination uses the least-squares GAN formulation (Mao et al., 2017). The MPD and MRD discriminators each minimize:

$$\mathcal{L}_\text{disc} = \mathbb{E}\!\left[(1 - D(y))^2\right] + \mathbb{E}\!\left[D(\hat{y})^2\right]$$

and the generator minimizes $\mathcal{L}_\text{adv} = \mathbb{E}[(1 - D(\hat{y}))^2]$, applied independently to MPD and MRD. The DurationDiscriminator uses the same formulation over predicted log-durations.

**Feature matching loss.** $\mathcal{L}_\text{fm}$ is the mean absolute difference between the real and generated intermediate feature maps of both the MPD and (when active) the MRD discriminators, encouraging the generator to produce activations close to those of real audio.

**Reconstruction loss.** $\mathcal{L}_\text{mel} = \|\text{Mel}(y) - \text{Mel}(\hat{y})\|_1$ is the L1 loss between 80-band mel spectrograms of the ground-truth segment $y$ and the generator output $\hat{y}$.

**Duration loss.** $\mathcal{L}_\text{dur}$ is the SDP negative ELBO (Section 3.3), normalized by the total phoneme count in the batch.

**KL loss.** $\mathcal{L}_\text{kl} = D_\text{KL}[q_\phi(z \mid x, c) \| p_\theta(z_p \mid c)]$ is the KL divergence between the approximate posterior and the flow-transformed prior, computed in the normalizing flow's output space.

**Windowed generator training.** Rather than decoding the full waveform per step, a random 8,192-sample segment (corresponding to 32 latent frames at hop length 256) is sliced from $z$ and decoded by the generator, and the same slice is extracted from the ground-truth waveform for discriminator training. This windowed approach (Kim et al., 2021) reduces memory cost and acts as data augmentation by exposing the generator to different positions within each utterance across epochs.

**Optimization.** Generator and all discriminators are each optimized with AdamW ($\text{lr} = 2 \times 10^{-4}$, $\beta_1 = 0.8$, $\beta_2 = 0.99$, $\varepsilon = 10^{-9}$, weight decay $\lambda = 0.01$), following Kim et al. (2021). The generator backward pass runs first, then the discriminator backward pass. Gradient norms are clipped to 1.0. Both learning rate schedulers follow an exponential decay $\gamma = 0.999875 = 0.999^{1/8}$ stepped once per epoch.

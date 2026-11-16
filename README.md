# Audio Tools
**Author:** Marco Toninato ([@rrudere](https://github.com/rrudere))  
**Portfolios:** [marcotoninato.com](https://marcotoninato.com) • [brokenglitch.com](https://brokenglitch.com)

A collection of lightweight, production-grade Python DSP tools designed for **AI voice dataset conditioning, podcast vocal post-production, and electronic music mastering consistency**.

---

## 1. audio_clean.py — Voice Pre-Processing & Dataset Conditioner
Engineered specifically for preparing raw speech recordings for **AI Voice Cloning (ElevenLabs / Local TTS)** and broadcast voiceovers.

* **Format Agnostic:** Reads `.wav`, `.mp3`, `.ogg`, `.flac`, `.aiff`, `.opus` and standardizes output to uncompressed 24-bit PCM WAV.
* **Mono Summing:** Converts multi-channel or stereo recordings to phase-coherent mono.
* **Infrasonic Rumble Filtration:** Applies a 4th-order Butterworth High-Pass Filter at **80 Hz** (24 dB/oct slope) to eliminate HVAC hum, mic-handling thumps, and proximity plosives.
* **Silence Stripping:** Trims leading and trailing silence below -45 dBFS using energy-based thresholding.
* **Broadcast Loudness Normalization:** Conforms integrated loudness to **-16 LUFS** (ITU-R BS.1770 / EBU R128) with a **-1.0 dBFS Peak Guard** ceiling to prevent inter-sample clipping on lossy distribution formats.

**Usage:**
`python audio_clean.py --input path/to/raw_audio --output path/to/cleaned --lufs -16.0`

---

## 2. ep_loudness_matcher.py — EP Drop & Spectral Cohesion Analyzer
Designed for mastering consistency across multi-track releases (specialized for high-energy music / **Clip to Zero** workflows targeting -7 to -9 LUFS).

* **Drop-Focused Analysis:** Instead of relying solely on Integrated LUFS (which gets biased by quiet intros/outros), it calculates maximum **Short-Term Loudness (3-second sliding window)** to pinpoint the real impact of drops and choruses.
* **Sub-to-Mid Frequency Ratio:** Analyzes acoustic energy distribution between Sub (20–90 Hz) and Presence (1.5–5.0 kHz) to detect **Fletcher-Munson bias** (e.g. tracks that meter high in LUFS due to excessive sub-bass but sound perceived as quiet).
* **Automated Gain Matching:** Calculates the exact trim in decibels (dB) required to level every track to the loudest drop of the release.

**Usage:**
`python ep_loudness_matcher.py --folder path/to/master_tracks`

---

## Quickstart

1. **Install dependencies:**  
`pip install -r requirements.txt`

2. **Run tools:**  
Place audio files into the `input/` folder and execute either script via terminal.

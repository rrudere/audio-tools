"""
Audio Clean & Pre-Processing Pipeline for Voice Cloning and Podcasts
Author: Marco Toninato (@rrudere)
Description: Batch DSP pipeline supporting MP3, OGG, WAV, FLAC, AIFF.
             Applies Mono conversion, 80Hz High-Pass Filter, silence trimming, 
             and -16 LUFS normalization with Peak Guard.
"""

import os
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    """Applica un filtro passa-alto Butterworth per eliminare il sub-rumble."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def trim_silence(audio: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
    """Taglia i silenzi all'inizio e alla fine della registrazione."""
    threshold = 10 ** (threshold_db / 20)
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio
    return audio[non_silent[0]:non_silent[-1]]


def process_audio(file_path: str, output_path: str, target_lufs: float = -16.0):
    """Esegue l'intera catena di elaborazione su un singolo file."""
    print(f"\n[Processing] -> {os.path.basename(file_path)}")
    
    try:
        # 1. Caricamento del file audio (supporta WAV, MP3, OGG, FLAC)
        data, sample_rate = sf.read(file_path)

        # 2. Conversione in Mono se il file e' stereo o multicanale
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            print("  [OK] Converted to Mono")

        # 3. Filtro Passa-Alto a 80Hz (rimozione sub-rumble e colpi d'aria)
        data = butter_highpass_filter(data, cutoff=80.0, fs=sample_rate)
        print("  [OK] Applied 80Hz High-Pass Filter")

        # 4. Rimozione dei silenzi a inizio e fine
        data = trim_silence(data, threshold_db=-45.0)
        print("  [OK] Trimmed leading/trailing silence")

        # 5. Normalizzazione del Loudness (LUFS standard)
        meter = pyln.Meter(sample_rate)
        current_loudness = meter.integrated_loudness(data)
        
        if not np.isinf(current_loudness):
            data = pyln.normalize.loudness(data, current_loudness, target_lufs)
            print(f"  [OK] Normalized loudness from {current_loudness:.1f} LUFS to {target_lufs:.1f} LUFS")

        # 6. Peak Guard: previene il clipping digitale sopra i -1.0 dBFS
        max_peak = np.max(np.abs(data))
        if max_peak > 0.89:  # -1.0 dBFS ceiling
            data = data * (0.89 / max_peak)
            print("  [OK] Peak Guard applied (-1.0 dBFS ceiling)")

        # 7. Esportazione del master pulito in WAV 24-bit studio standard
        sf.write(output_path, data, sample_rate, subtype='PCM_24')
        print(f"  [DONE] Saved as studio master -> {os.path.basename(output_path)}")

    except Exception as e:
        print(f"  [ERROR] Impossibile processare il file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Voice Pre-Processing Tool for Voice Cloning")
    parser.add_argument("--input", default="input", help="Cartella dei file audio")
    parser.add_argument("--output", default="output", help="Cartella di destinazione")
    parser.add_argument("--lufs", type=float, default=-16.0, help="Target LUFS (default: -16)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Formati audio supportati
    supported_formats = (".wav", ".mp3", ".ogg", ".flac", ".aif", ".aiff", ".opus")

    files = [f for f in os.listdir(args.input) if f.lower().endswith(supported_formats)]

    if not files:
        print(f"Nessun file audio trovato nella cartella '{args.input}/'.")
        print(f"Formati supportati: {', '.join(supported_formats)}")
        return

    print(f"=== Trovati {len(files)} file da elaborare ===")
    for filename in files:
        in_file = os.path.join(args.input, filename)
        base_name = os.path.splitext(filename)[0]
        out_file = os.path.join(args.output, f"cleaned_{base_name}.wav")
        process_audio(in_file, out_file, target_lufs=args.lufs)

    print("\n[SUCCESS] Batch completato con successo!")


if __name__ == "__main__":
    main()

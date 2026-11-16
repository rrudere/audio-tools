"""
EP Loudness & Spectral Matcher (CTZ / High-Energy Mastering)
Author: Marco Toninato (@rrudere)
Description: Analyzes entire EP folder for drop loudness consistency (Short-Term LUFS)
             and sub-to-mid balance according to ITU-R BS.1770 / EBU R128.
"""

import os
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 3) -> np.ndarray:
    """Filtro passa-banda Butterworth per isolare range di frequenze specifici."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def calculate_rms(audio: np.ndarray) -> float:
    """Calcola l'energia RMS evitando log di zero."""
    rms = np.sqrt(np.mean(audio**2))
    return max(rms, 1e-9)


def analyze_track(file_path: str, meter: pyln.Meter, fs_target: int = 44100):
    data, sample_rate = sf.read(file_path)

    # Downmix mono per l'analisi spettrale e meter
    if data.ndim > 1:
        mono_data = np.mean(data, axis=1)
    else:
        mono_data = data

    # 1. Integrated LUFS complessivo dell'intera traccia
    integrated_lufs = meter.integrated_loudness(mono_data)

    # 2. Ricerca del Drop piu' potente (Short-Term LUFS con finestra di 3 sec)
    window_sec = 3.0
    step_sec = 1.0
    win_samples = int(window_sec * sample_rate)
    step_samples = int(step_sec * sample_rate)

    short_terms = []
    # Scorre la traccia a blocchi di 3 secondi
    for start in range(0, len(mono_data) - win_samples, step_samples):
        block = mono_data[start:start + win_samples]
        try:
            val = meter.integrated_loudness(block)
            if not np.isinf(val):
                short_terms.append(val)
        except:
            pass

    max_short_term = max(short_terms) if short_terms else integrated_lufs

    # 3. Analisi bilanciamento Spettrale (Sub 20-90Hz vs Medi 1.5-5kHz)
    sub_band = bandpass_filter(mono_data, 20.0, 90.0, sample_rate)
    mid_band = bandpass_filter(mono_data, 1500.0, 5000.0, sample_rate)

    rms_sub = calculate_rms(sub_band)
    rms_mid = calculate_rms(mid_band)
    
    # Rapporto in dB tra Sub e Mid: valori alti indicano eccesso di sub
    sub_mid_ratio_db = 20 * np.log10(rms_sub / rms_mid)

    return {
        "filename": os.path.basename(file_path),
        "integrated": integrated_lufs,
        "drop_lufs": max_short_term,
        "sub_mid_ratio": sub_mid_ratio_db
    }


def main():
    parser = argparse.ArgumentParser(description="EP Loudness & Drop Matcher")
    parser.add_argument("--folder", default="input", help="Cartella contenente i mix/master dell'EP")
    args = parser.parse_args()

    supported = (".wav", ".mp3", ".flac", ".aif", ".aiff")
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(supported)]

    if not files:
        print(f"Nessun file trovato nella cartella '{args.folder}/'.")
        return

    print(f"\nAnalisi dell'EP in corso ({len(files)} tracce)... Attendere.\n")
    meter = pyln.Meter(44100) # meter standard EBU R128

    results = []
    for f in files:
        path = os.path.join(args.folder, f)
        res = analyze_track(path, meter)
        results.append(res)

    # Identifica il Drop piu' potente come Riferimento (Target dell'EP)
    target_drop = max(r["drop_lufs"] for r in results)

    print("=" * 95)
    print(f"{'TRACCIA':<28} | {'INTEGRATED':<11} | {'DROP (MAX ST)':<14} | {'SUB/MID BIAS':<13} | {'TRIM CONSIGLIATO'}")
    print("=" * 95)

    for r in results:
        diff = target_drop - r["drop_lufs"]
        
        # Spiegazione del bias energetico
        if r["sub_mid_ratio"] > 14.0:
            bias_note = "Sub Heavy (!)"
        elif r["sub_mid_ratio"] < 6.0:
            bias_note = "Mid/Sharp"
        else:
            bias_note = "Bilanciato"

        if abs(diff) < 0.2:
            trim_str = "-> TARGET (0.0 dB)"
        else:
            trim_str = f"{diff:+.1f} dB"

        name = (r['filename'][:25] + '..') if len(r['filename']) > 27 else r['filename']
        print(f"{name:<28} | {r['integrated']:>6.1f} LUFS | {r['drop_lufs']:>9.1f} LUFS | {bias_note:<13} | {trim_str}")

    print("=" * 95)
    print("\n[NOTE DI MASTERING PER IL CLIP TO ZERO]:")
    print("1. DROP (MAX ST): e' la vera intensita' percepita nel punto di massimo impatto (Drop/Ritornello).")
    print("2. SUB HEAVY (!): significa che il sub sta ingannando il meter; togli 1-2 dB di sub per far emergere i medi.")
    print("3. TRIM CONSIGLIATO: sono i dB che mancano al master bus per suonare violento quanto la traccia piu' alta.")


if __name__ == "__main__":
    main()
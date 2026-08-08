import os
import math
import numpy as np
import scipy.io.wavfile as wav

def calculate_rms(audio):
    """Oblicza wartość RMS (Root Mean Square) sygnału."""
    return np.sqrt(np.mean(audio ** 2))

def match_target_snr(clean_audio, noise_audio, target_snr):
    """
    Dopasowuje głośność szumu do czystego sygnału dla zadanego poziomu SNR.
    SNR = 10 * log10(RMS_clean^2 / RMS_noise^2)
    """
    rms_clean = calculate_rms(clean_audio)
    rms_noise = calculate_rms(noise_audio)
    
    # Obsługa ciszy w plikach wejściowych
    if rms_clean == 0 or rms_noise == 0:
        return noise_audio
        
    # Kalkulacja wymaganego RMS dla szumu
    # target_snr = 20 * log10(rms_clean / rms_noise_target)
    rms_noise_target = rms_clean / (10 ** (target_snr / 20.0))
    
    # Skalowanie szumu
    scaled_noise = noise_audio * (rms_noise_target / rms_noise)
    return scaled_noise

def mix_audio_with_snr(clean_path, noise_path, output_path, target_snr):
    """
    Wczytuje pliki, dopasowuje ich długość, miksuje dla zadanego SNR i zapisuje wynik.
    """
    # Wczytanie plików audio
    sr_clean, clean = wav.read(clean_path)
    sr_noise, noise = wav.read(noise_path)
    
    if sr_clean != sr_noise:
        raise ValueError(f"Częstotliwości próbkowania się nie zgadzają: {sr_clean}Hz vs {sr_noise}Hz")
    
    # Konwersja do float32 w zakresie [-1.0, 1.0] dla stabilności obliczeń RMS
    if clean.dtype == np.int16:
        clean = clean.astype(np.float32) / 32768.0
    if noise.dtype == np.int16:
        noise = noise.astype(np.float32) / 32768.0
        
    # Dopasowanie długości (szum musi być tak długi jak czysty sygnał)
    if len(noise) < len(clean):
        # Zapętlenie szumu, jeśli jest za krótki
        repeats = math.ceil(len(clean) / len(noise))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # Przycięcie szumu, jeśli jest za długi (losowy punkt startowy lub od początku)
        # Dla powtarzalności bierzemy od początku lub losowo:
        start_idx = np.random.randint(0, len(noise) - len(clean) + 1)
        noise = noise[start_idx:start_idx + len(clean)]
        
    # Skalowanie szumu do pożądanego poziomu SNR
    scaled_noise = match_target_snr(clean, noise, target_snr)
    
    # Miksowanie (sumowanie sygnałów)
    mixed = clean + scaled_noise
    
    # Zapobieganie clippingowi (przesterowaniu) poprzez opcjonalną normalizację całego miksu
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
        
    # Konwersja z powrotem do formatu int16 i zapis pliku
    mixed_int16 = (mixed * 32767.0).astype(np.int16)
    wav.write(output_path, sr_clean, mixed_int16)
    print(f"Zapisano miks SNR={target_snr}dB -> {output_path}")

if __name__ == "__main__":
    # Przykład użycia skryptu dla folderu z danymi
    clean_voice = "path_to_clean_voice.wav"
    noise_sound = "path_to_noise.wav"
    output_dir = "augmented_dataset"
    
    # Lista poziomów SNR do wygenerowania (w decybelach)
    # Wyższe SNR = cichszy szum (np. 20dB to lekki szum, 0dB to szum tak głośny jak mowa)
    snr_levels = [-5, 0, 5, 10, 15, 20]
    
    print("Przykładowa konfiguracja załadowana. Zastąp ścieżki realnymi plikami, aby uruchomić proces.")
    # Pętla generująca miksy (odkomentuj w swoim środowisku po podaniu właściwych ścieżek):
    # os.makedirs(output_dir, exist_ok=True)
    # for snr in snr_levels:
    #     out_name = f"mixed_snr_{snr}.wav"
    #     mix_audio_with_snr(clean_voice, noise_sound, os.path.join(output_dir, out_name), snr)

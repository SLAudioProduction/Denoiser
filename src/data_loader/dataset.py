import os
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioNoiseReductionDataset(Dataset):
    """
    Klasa Dataset dla PyTorcha obsługująca pary plików: czysty głos i zaszumiony głos.
    """
    def __init__(self, clean_dir, noisy_dir, segment_len_seconds=2.0, target_sr=16000):
        """
        Argumenty:
            clean_dir (str): Ścieżka do katalogu z czystym audio.
            noisy_dir (str): Ścieżka do katalogu z zaszumionym audio.
            segment_len_seconds (float): Docelowa długość fragmentu audio w sekundach dla modelu.
            target_sr (int): Docelowa częstotliwość próbkowania (np. 16000 Hz).
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.target_sr = target_sr
        self.segment_samples = int(segment_len_seconds * target_sr)
        
        # Pobranie list plików (zakładamy, że pliki w obu folderach nazywają się tak samo)
        self.file_names = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
        
        if len(self.file_names) == 0:
            raise RuntimeError(f"Nie znaleziono plików .wav w katalogu: {clean_dir}")

    def __len__(self):
        """Zwraca całkowitą liczbę próbek w zestawie danych."""
        return len(self.file_names)

    def _process_audio_length(self, audio):
        """
        Docina lub dopasowuje długość audio za pomocą wycinania/uzupełniania zerami (padding),
        aby uzyskać dokładnie stałą liczbę próbek (self.segment_samples).
        """
        # audio ma kształt [kanały, próbki], zazwyczaj [1, n] dla mono
        num_samples = audio.shape[1]
        
        if num_samples > self.segment_samples:
            # Jeśli audio jest za długie, wycinamy losowy fragment (dobra praktyka dla augmentacji)
            start_sample = torch.randint(0, num_samples - self.segment_samples + 1, (1,)).item()
            audio = audio[:, start_sample:start_sample + self.segment_samples]
        elif num_samples < self.segment_samples:
            # Jeśli audio jest za krótkie, uzupełniamy zerami na końcu (padding)
            pad_len = self.segment_samples - num_samples
            audio = torch.nn.functional.pad(audio, (0, pad_len))
            
        return audio

    def __getitem__(self, idx):
        """
        Pobiera jedną parę (zaszumiony_tensor, czysty_tensor) dla danego indeksu.
        """
        file_name = self.file_names[idx]
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)
        
        # 1. Wczytanie audio za pomocą torchaudio
        clean_audio, sr_clean = torchaudio.load(clean_path)
        noisy_audio, sr_noise = torchaudio.load(noisy_path)
        
        # 2. Resampling, jeśli częstotliwość próbkowania się nie zgadza
        if sr_clean != self.target_sr:
            clean_audio = torchaudio.functional.resample(clean_audio, sr_clean, self.target_sr)
        if sr_noise != self.target_sr:
            noisy_audio = torchaudio.functional.resample(noisy_audio, sr_noise, self.target_sr)
            
        # 3. Wymuszenie formatu mono (jeśli plik był stereo, bierzemy pierwszy kanał)
        if clean_audio.shape[0] > 1:
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            
        # 4. Dopasowanie do stałej długości (stała liczba próbek dla sieci)
        # Używamy tego samego losowego punktu startowego dla obu plików, aby były zsynchronizowane
        num_samples = clean_audio.shape[1]
        if num_samples > self.segment_samples:
            start_sample = torch.randint(0, num_samples - self.segment_samples + 1, (1,)).item()
            clean_audio = clean_audio[:, start_sample:start_sample + self.segment_samples]
            noisy_audio = noisy_audio[:, start_sample:start_sample + self.segment_samples]
        else:
            clean_audio = self._process_audio_length(clean_audio)
            noisy_audio = self._process_audio_length(noisy_audio)
            
        # Zwracamy parę: wejście do sieci (zaszumiony) oraz oczekiwany wynik (czysty)
        return noisy_audio, clean_audio

# --- PRZYKŁAD UŻYCIA W KODZIE TRENINGOWYM ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Przykładowe uruchomienie testowe (użyj swoich ścieżek z katalogu data/)
    try:
        dataset = AudioNoiseReductionDataset(
            clean_dir="data/processed/clean", 
            noisy_dir="data/processed/noisy",
            segment_len_seconds=2.0,
            target_sr=16000
        )
        
        # Dataloader automatycznie grupuje próbki w paczki (batches) i miesza dane (shuffle)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Pobranie jednej paczki danych
        for noisy_batch, clean_batch in dataloader:
            print(f"Kształt paczki zaszumionej: {noisy_batch.shape}")  # Oczekiwane: [4, 1, 32000]
            print(f"Kształt paczki czystej: {clean_batch.shape}")      # Oczekiwane: [4, 1, 32000]
            break
            
    except Exception as e:
        print(f"Wskazówka: Aby przetestować ten plik, stwórz najpierw katalogi testowe. Błąd: {e}")

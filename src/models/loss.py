import torch
import torch.nn as nn
import torchaudio

class SingleScaleSpectralLoss(nn.Module):
    """Oblicza błąd spektralny dla jednej, konkretnej wielkości okna STFT."""
    def __init__(self, n_fft, hop_length, win_length):
        super(SingleScaleSpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Rejestrujemy okno Hanna jako bufor, aby automatycznie przenosiło się na GPU z modelem
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x, y):
        # Usunięcie wymiaru kanału, jeśli audio ma kształt [Batch, 1, Samples] -> [Batch, Samples]
        x = x.squeeze(1)
        y = y.squeeze(1)

        # Obliczenie STFT (spektrogramu zespolonego)
        stft_x = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length, 
            window=self.window, return_complex=True
        )
        stft_y = torch.stft(
            y, self.n_fft, self.hop_length, self.win_length, 
            window=self.window, return_complex=True
        )

        # Wyciągnięcie amplitudy (modułu) spektrogramu z zabezpieczeniem przed log(0)
        mag_x = torch.abs(stft_x) + 1e-7
        mag_y = torch.abs(stft_y) + 1e-7

        # 1. Konwergencja spektralna (błąd względny macierzy)
        sc_loss = torch.norm(mag_y - mag_x, p="fro") / torch.norm(mag_y, p="fro")

        # 2. Błąd w skali logarytmicznej (bliższy ludzkiej percepcji głośności)
        log_mag_loss = torch.mean(torch.abs(torch.log(mag_y) - torch.log(mag_x)))

        return sc_loss + log_mag_loss

class MultiScaleSpectralLoss(nn.Module):
    """
    Agreguje błędy spektralne z wielu rozdzielczości czasowo-częstotliwościowych
    oraz opcjonalnie dodaje podstawowy błąd czasowy L1.
    """
    def __init__(self, fft_sizes=[2048, 1024, 512, 256], hop_sizes=[512, 256, 128, 64], win_sizes=[2048, 1024, 512, 256], alpha_l1=0.1):
        super(MultiScaleSpectralLoss, self).__init__()
        self.loss_layers = nn.ModuleList()
        self.alpha_l1 = alpha_l1
        self.l1_loss = nn.L1Loss()
        
        # Tworzenie warstw dla każdej skali FFT
        for n_fft, hop_len, win_len in zip(fft_sizes, hop_sizes, win_sizes):
            self.loss_layers.append(SingleScaleSpectralLoss(n_fft, hop_len, win_len))

    def forward(self, output, target):
        """
        Argumenty:
            output (Tensor): Sygnał wyjściowy z sieci (wyczyszczone audio) [B, 1, T]
            target (Tensor): Sygnał referencyjny (czysty głos) [B, 1, T]
        """
        spectral_loss = 0.0
        
        # Sumowanie błędów ze wszystkich zdefiniowanych skal okna
        for layer in self.loss_layers:
            spectral_loss += layer(output, target)
            
        # Średnia wartość błędu spektralnego
        spectral_loss /= len(self.loss_layers)
        
        # Dodanie tradycyjnego błędu czasowego L1 (pomaga zachować idealną synchronizację kształtu fali)
        time_loss = self.l1_loss(output, target)
        
        total_loss = spectral_loss + self.alpha_l1 * time_loss
        return total_loss

# --- TEST FUNKCJI STRATY ---
if __name__ == "__main__":
    # Inicjalizacja funkcji straty
    criterion = MultiScaleSpectralLoss()
    
    # Symulacja wyjścia z sieci oraz idealnego targetu (np. batch=2, kanał=1, 2 sekundy audio przy 16kHz)
    dummy_output = torch.randn(2, 1, 32000, requires_grad=True)
    dummy_target = torch.randn(2, 1, 32000)
    
    loss = criterion(dummy_output, dummy_target)
    print(f"Obliczona wartość straty (Loss): {loss.item():.4f}")
    
    # Test wstecznej propagacji (czy graf obliczeń działa poprawnie)
    loss.backward()
    print("Sukces: Wsteczna propagacja (backward pass) przebiegła pomyślnie!")

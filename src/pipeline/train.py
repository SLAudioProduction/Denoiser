import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

# Importy z naszych wcześniejszych modułów
from src.data_loader.dataset import AudioNoiseReductionDataset
from src.models.unet import WaveUNet
from src.models.loss import MultiScaleSpectralLoss

def train_model():
    # 1. Konfiguracja sprzętowa (GPU jeśli dostępne, w przeciwnym razie CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uruchamianie treningu na urządzeniu: {device}")

    # 2. Hiperparametry (w przyszłości przeniesione do config.yaml)
    epochs = 50
    batch_size = 16
    learning_rate = 0.0003
    clean_data_dir = "data/processed/clean"
    noisy_data_dir = "data/processed/noisy"
    checkpoint_dir = "checkpoints"
    
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Przygotowanie danych (Dataset i DataLoader)
    print("Inicjalizacja ładowania danych...")
    try:
        train_dataset = AudioNoiseReductionDataset(
            clean_dir=clean_data_dir,
            noisy_dir=noisy_data_dir,
            segment_len_seconds=2.0,
            target_sr=16000
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,        # Równoległe ładowanie danych przez CPU
            pin_memory=True      # Szybszy transfer danych na GPU
        )
    except Exception as e:
        print(f"Błąd ładowania danych: {e}")
        print("Wskazówka: Upewnij się, że foldery z danymi istnieją i zawierają pliki .wav")
        return

    # 4. Inicjalizacja modeli, funkcji straty i optymalizatora
    model = WaveUNet(num_levels=5, start_channels=24).to(device)
    criterion = MultiScaleSpectralLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    # 5. Główna pętla treningowa
    print("Rozpoczęcie treningu sieci...")
    for epoch in range(1, epochs + 1):
        model.train()  # Przełączenie modelu w tryb treningu
        running_loss = 0.0
        
        for batch_idx, (noisy_audio, clean_audio) in enumerate(train_loader):
            # Przeniesienie danych na odpowiednie urządzenie (CPU/GPU)
            noisy_audio = noisy_audio.to(device)
            clean_audio = clean_audio.to(device)
            
            # Zerowanie gradientów z poprzedniego kroku
            optimizer.zero_grad()
            
            # Krok w przód (Forward pass)
            denoised_audio = model(noisy_audio)
            
            # Obliczenie wartości błędu (Loss)
            loss = criterion(denoised_audio, clean_audio)
            
            # Krok w tył (Backward pass - obliczenie gradientów)
            loss.backward()
            
            # Aktualizacja wag modelu
            optimizer.step()
            
            running_loss += loss.item()
            
            # Logowanie postępu wewnątrz epoki
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoka [{epoch}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Bieżący Loss: {loss.item():.4f}")
                
        # Obliczenie średniej straty dla całej epoki
        epoch_loss = running_loss / len(train_loader)
        print(f"===> Koniec Epoki {epoch} | Średni Loss: {epoch_loss:.4f} <===")
        
        # 6. Zapisywanie najlepszego modelu (Checkpointing)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Zapisano nowy najlepszy model do: {checkpoint_path}")
            
        print("-" * 50)

    print("Trening zakończony sukcesem!")

if __name__ == "__main__":
    train_model()

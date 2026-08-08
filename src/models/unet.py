import torch
import torch.nn as nn

class DownBlock(nn.Module):
    """Blok kodujący (Encoder): Konwolucja + Aktywacja + Downsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        # Downsampling realizujemy za pomocą Decimator (co druga próbka)
        self.downsample = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # Zwracamy wersję przed downsamplingiem (dla Skip Connection) oraz po downsamplingu
        x_before_pool = self.leaky_relu(self.conv(x))
        x_after_pool = self.downsample(x_before_pool)
        return x_after_pool, x_before_pool

class UpBlock(nn.Module):
    """Blok dekodujący (Decoder): Upsampling + Konkatenacja ze Skip Connection + Konwolucja"""
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7):
        super(UpBlock, self).__init__()
        # Zwiększanie wymiaru czasowego x2 poprzez interpolację liniową (unikamy artifactów 'checkerboard')
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # in_channels uwzględnia połączenie cech z obecnej warstwy oraz ze Skip Connection (x2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        
        # Dopasowanie długości w razie drobnych różnic zaokrągleń przy down/upsamplingu
        if x.shape[-1] != skip_connection.shape[-1]:
            x = nn.functional.interpolate(x, size=skip_connection.shape[-1], mode='linear', align_corners=True)
            
        # Połączenie cech z decodera i encodera wzdłuż wymiaru kanałów (dim=1)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.leaky_relu(self.conv(x))
        return x

class WaveUNet(nn.Module):
    """Główna architektura Wave-U-Net dla redukcji szumów w audio (Time Domain)"""
    def __init__(self, num_levels=5, start_channels=24):
        super(WaveUNet, self).__init__()
        self.num_levels = num_levels
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        # 1. Budowa Encodera
        in_ch = 1  # Audio mono (1 kanał wejściowy)
        out_ch = start_channels
        for i in range(num_levels):
            self.down_blocks.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch * 2
            
        # 2. Blok centralny (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. Budowa Decodera
        for i in range(num_levels):
            # Kanały wejściowe decodera = kanały z poprzedniej warstwy + kanały ze skip connection
            skip_ch = in_ch
            next_in_ch = in_ch + skip_ch
            next_out_ch = in_ch // 2
            
            self.up_blocks.append(UpBlock(next_in_ch, next_out_ch))
            in_ch = next_out_ch
            
        # 4. Warstwa wyjściowa - rzutowanie z powrotem na 1 kanał audio
        self.final_conv = nn.Conv1d(in_ch, 1, kernel_size=1)
        # Tanh ogranicza sygnał wyjściowy do bezpiecznego zakresu amplitudy [-1.0, 1.0]
        self.output_act = nn.Tanh()

    def forward(self, x):
        # Lista do przechowywania aktywacji dla Skip Connections
        skip_connections = []
        
        # Krok w dół (Encoder)
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Krok w górę (Decoder) - idziemy od końca listy skip_connections
        for i in range(self.num_levels):
            skip = skip_connections[-(i + 1)]
            up_block = self.up_blocks[i]
            x = up_block(x, skip)
            
        # Generowanie końcowej fali dźwiękowej
        output = self.output_act(self.final_conv(x))
        return output

# --- TEST ARCHITEKTURY ---
if __name__ == "__main__":
    # Inicjalizacja modelu
    model = WaveUNet(num_levels=5, start_channels=24)
    
    # Symulacja paczki danych (Batch) z Dataloadera z poprzedniego kroku:
    # [Rozmiar batcha=4, Liczba kanałów=1, Liczba próbek = 2 sekundy * 16000Hz = 32000]
    dummy_input = torch.randn(4, 1, 32000)
    
    # Przepuszczenie danych przez sieć
    with torch.no_grad():
        dummy_output = model(dummy_input)
        
    print(f"Wejście sieci: {dummy_input.shape}")   # Oczekiwane: torch.Size([4, 1, 32000])
    print(f"Wyjście sieci: {dummy_output.shape}") # Oczekiwane: torch.Size([4, 1, 32000])
    print("Sukces: Wymiary macierzy na wyjściu są identyczne z wejściowymi!")

import numpy as np
import sounddevice as sd
import scipy.signal as signal
import zlib
from PIL import Image
import matplotlib.pyplot as plt

# --- IEEE Project Specs: 4-FSK ---
FS = 44100
FREQS = [1000, 1500, 2000, 2500] 
BIT_DURATION = 0.02             # 20ms per symbol (2 bits)
PREAMBLE = [1, 0, 1, 0, 1, 0, 1, 1] 

class SuperFastModem:
    def __init__(self):
        self.spb = int(FS * BIT_DURATION)
        self.stats = {"mags": [], "decisions": [], "true_bits": []}

    def _get_tone(self, freq, samples):
        t = np.linspace(0, samples/FS, samples, endpoint=False)
        return np.sin(2 * np.pi * freq * t)

    def encode_image(self, img_path):
        # 1. Image Processing
        img = Image.open(img_path).convert('L').resize((32, 32))
        raw_bytes = np.array(img).tobytes()
        compressed = zlib.compress(raw_bytes)
        
        # 2. Add Header & Convert to Bits
        payload = len(compressed).to_bytes(2, 'big') + compressed
        bits = []
        for b in payload:
            bits.extend([int(x) for x in format(b, '08b')])
        
        # Save for BER calculation
        self.stats["true_bits"] = bits
        
        # 3. Generate Audio
        audio = []
        for bit in PREAMBLE:
            audio.extend(self._get_tone(FREQS[3] if bit==1 else FREQS[0], self.spb))
            
        for i in range(0, len(bits), 2):
            dibit = bits[i:i+2]
            if len(dibit) < 2: dibit += [0]
            val = int("".join(map(str, dibit)), 2)
            audio.extend(self._get_tone(FREQS[val], self.spb))
            
        return np.array(audio, dtype=np.float32), compressed

    def decode_image(self, rec_sig):
        # 1. Normalize and Sync
        rec_sig = rec_sig - np.mean(rec_sig)
        rec_sig /= (np.max(np.abs(rec_sig)) + 1e-9)
        
        sync_ref = []
        for bit in PREAMBLE:
            sync_ref.extend(self._get_tone(FREQS[3] if bit==1 else FREQS[0], self.spb))
        
        corr = signal.correlate(rec_sig, sync_ref, mode='valid')
        start_idx = np.argmax(np.abs(corr)) + len(sync_ref)
        
        # 2. Extract and Track Quality
        bits = []
        confidence_scores = []
        
        for i in range(start_idx, len(rec_sig) - self.spb, self.spb):
            chunk = rec_sig[i : i + self.spb]
            t = np.linspace(0, len(chunk)/FS, len(chunk), endpoint=False)
            
            mags = []
            for f in FREQS:
                # Quadrature Detection
                s = np.sum(chunk * np.sin(2 * np.pi * f * t))
                c = np.sum(chunk * np.cos(2 * np.pi * f * t))
                mags.append(np.sqrt(s**2 + c**2))
            
            best_val = np.argmax(mags)
            
            # Calibration Metric: Margin of Victory
            # Ratio of winning frequency power vs the average of others
            others = [m for idx, m in enumerate(mags) if idx != best_val]
            margin = mags[best_val] / (np.mean(others) + 1e-9)
            confidence_scores.append(margin)
            
            bits.extend([int(x) for x in format(best_val, '02b')])

        self.stats["confidence"] = confidence_scores
        self.stats["decoded_bits"] = bits

        # 3. Reconstruct
        decoded_bytes = []
        for i in range(0, len(bits) - (len(bits)%8), 8):
            decoded_bytes.append(int("".join(map(str, bits[i:i+8])), 2))
        
        if len(decoded_bytes) < 2: return None
        data_len = int.from_bytes(bytes(decoded_bytes[:2]), 'big')
        return bytes(decoded_bytes[2:2+data_len])

    def plot_calibration(self):
        """Generates the Signal Quality and BER plots."""
        # Calculate BER
        true = self.stats["true_bits"]
        decoded = self.stats["decoded_bits"][:len(true)]
        errors = np.array(true) != np.array(decoded)
        ber = (np.sum(errors) / len(true)) * 100 if len(true) > 0 else 0

        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Signal Confidence (Quality)
        plt.subplot(2, 1, 1)
        plt.plot(self.stats["confidence"], color='blue', alpha=0.7, label="Signal Margin")
        plt.axhline(y=2.0, color='red', linestyle='--', label="Unreliable Threshold")
        plt.title(f"Hardware Calibration: Signal Confidence (Avg: {np.mean(self.stats['confidence']):.2f})")
        plt.ylabel("Signal-to-Noise Ratio")
        plt.legend()

        # Subplot 2: Error Distribution
        plt.subplot(2, 1, 2)
        plt.stem(errors, markerfmt=' ', linefmt='red', label="Bit Errors")
        plt.title(f"Bit Error Distribution (Total BER: {ber:.2f}%)")
        plt.xlabel("Bit Index")
        plt.ylabel("Error (1=Wrong)")
        plt.legend()

        plt.tight_layout()
        plt.show()
        return ber

# --- EXECUTION ---
modem = SuperFastModem()
IMAGE_PATH = "input2.jpeg" # Create a 32x32 image!

try:
    # 1. Transmit
    audio_signal, compressed_original = modem.encode_image(IMAGE_PATH)
    print(f"Sending... Duration: {len(audio_signal)/FS:.2f}s")
    rec = sd.playrec(np.concatenate([np.zeros(FS), audio_signal, np.zeros(FS)]), FS, channels=1)
    sd.wait()

    # 2. Decode
    received_data = modem.decode_image(rec.flatten())
    
    # 3. Calibrate & Display
    ber = modem.plot_calibration()
    
    if ber == 0:
        raw_img = zlib.decompress(received_data)
        final_img = Image.frombytes('L', (32, 32), raw_img)
        final_img.show()
        print("Success! BER is 0.00%. Calibration Optimal.")
    else:
        print(f"Calibration Failed: BER is {ber:.2f}%. Check volume/distance.")

except Exception as e:
    print(f"Error: {e}")

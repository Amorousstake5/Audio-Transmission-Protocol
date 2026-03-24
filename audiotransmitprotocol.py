# Version 7 of Audio Transmission Protocol with 4 - FSK transmission
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import zlib
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

# --- IEEE Project Specs: 4-FSK ---
FS = 44100
FREQS = [1000, 1500, 2000, 2500] 
BIT_DURATION = 0.02             # 20ms symbols
PREAMBLE = [1, 0, 1, 0, 1, 0, 1, 1] 

# Gray Code Mapping: 00, 01, 11, 10
GRAY_MAP = [0, 1, 3, 2] 
INV_GRAY_MAP = {v: k for k, v in enumerate(GRAY_MAP)}

class UnifiedIEEEModem:
    def __init__(self):
        self.spb = int(FS * BIT_DURATION)
        self.stats = {"true_bits": [], "decoded_bits": [], "confidence": []}

    def _get_tone(self, freq, samples):
        t = np.linspace(0, samples/FS, samples, endpoint=False)
        return np.sin(2 * np.pi * freq * t)

    def print_progress(self, iteration, total, prefix='', suffix='', length=30, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()
        if iteration == total: print()

    def encode(self, data_bytes):
        """Generic encoder for any byte stream (Text or Image)."""
        compressed = zlib.compress(data_bytes, level=9)
        payload = len(compressed).to_bytes(2, 'big') + compressed
        
        bits = []
        for b in payload:
            bits.extend([int(x) for x in format(b, '08b')])
        self.stats["true_bits"] = bits
        
        audio = []
        # Preamble
        for bit in PREAMBLE:
            audio.extend(self._get_tone(FREQS[3] if bit==1 else FREQS[0], self.spb))
        # Payload
        for i in range(0, len(bits), 2):
            val = int("".join(map(str, bits[i:i+2])), 2)
            freq_idx = GRAY_MAP[val]
            audio.extend(self._get_tone(FREQS[freq_idx], self.spb))
            
        return np.array(audio, dtype=np.float32), len(compressed)

    def decode(self, rec_sig):
        """Demodulator with real-time progress and SNR tracking."""
        rec_sig = rec_sig - np.mean(rec_sig)
        rec_sig /= (np.max(np.abs(rec_sig)) + 1e-9)
        
        sync_ref = []
        for bit in PREAMBLE:
            sync_ref.extend(self._get_tone(FREQS[3] if bit==1 else FREQS[0], self.spb))
        corr = signal.correlate(rec_sig, sync_ref, mode='valid')
        start_idx = np.argmax(np.abs(corr)) + len(sync_ref)
        
        bits = []
        confidence = []
        total_symbols = (len(rec_sig) - start_idx) // self.spb
        
        print("[*] Decoding Acoustic Stream...")
        for count, i in enumerate(range(start_idx, len(rec_sig) - self.spb, self.spb)):
            chunk = rec_sig[i : i + self.spb]
            t = np.linspace(0, len(chunk)/FS, len(chunk), endpoint=False)
            
            mags = [np.sqrt(np.sum(chunk * np.sin(2*np.pi*f*t))**2 + 
                            np.sum(chunk * np.cos(2*np.pi*f*t))**2) for f in FREQS]
            
            best_val = np.argmax(mags)
            actual_val = INV_GRAY_MAP[best_val]
            bits.extend([int(x) for x in format(actual_val, '02b')])
            
            # Confidence Calculation (SNR Proxy)
            others = [m for idx, m in enumerate(mags) if idx != best_val]
            confidence.append(mags[best_val] / (np.mean(others) + 1e-9))
            
            if count % 10 == 0:
                self.print_progress(count, total_symbols, prefix='Progress', suffix='Complete')

        self.print_progress(total_symbols, total_symbols, prefix='Progress', suffix='Complete')
        self.stats["decoded_bits"], self.stats["confidence"] = bits, confidence

        byte_data = []
        for i in range(0, len(bits)-(len(bits)%8), 8):
            byte_data.append(int("".join(map(str, bits[i:i+8])), 2))
        
        if len(byte_data) < 2: return None
        d_len = int.from_bytes(bytes(byte_data[:2]), 'big')
        return bytes(byte_data[2:2+d_len])

    def show_calibration(self):
        true = self.stats["true_bits"]
        decoded = self.stats["decoded_bits"][:len(true)]
        errors = np.array(true) != np.array(decoded)
        ber = (np.sum(errors) / len(true)) * 100 if len(true) > 0 else 0

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(self.stats["confidence"], color='teal', label="Confidence")
        plt.axhline(y=2.5, color='red', linestyle='--', label="Fail Limit")
        plt.title(f"Channel Quality (Avg Confidence: {np.mean(self.stats['confidence']):.2f})")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.stem(errors, markerfmt=' ', linefmt='red')
        plt.title(f"Bit Error Distribution (BER: {ber:.2f}%)")
        plt.tight_layout()
        plt.show()
        return ber

# --- Utilities ---
def get_synthesized_image():
    print("\n[Dummy Generator] 1: BW (Grayscale) | 2: Color (ARGB)")
    mode = input("Select: ")
    if mode == '1':
        val = int(input("Brightness (0-255): "))
        return Image.new('L', (32, 32), color=val)
    else:
        print("Enter ARGB values (0-255):")
        a, r, g, b = int(input("Alpha: ")), int(input("Red: ")), int(input("Green: ")), int(input("Blue: "))
        return Image.new('RGBA', (32, 32), color=(r, g, b, a))

# --- Main CLI ---
def main():
    modem = UnifiedIEEEModem()
    print("\n" + "="*40 + "\n IEEE UNIFIED 4-FSK MODEM 2026 \n" + "="*40)
    
    while True:
        mode_select = input("\nTransmit: [1] Text [2] Image [Q] Quit: ").upper()
        if mode_select == 'Q': break
        
        data_bytes = b""
        img_mode = None

        if mode_select == '1':
            msg = input("Enter Text: ")
            data_bytes = msg.encode('utf-8')
        elif mode_select == '2':
            src = input("  Source: [D] Dummy [F] File: ").upper()
            if src == 'D':
                img_obj = get_synthesized_image()
            else:
                path = input("  Path: ")
                if not os.path.exists(path): continue
                img_obj = Image.open(path).resize((32, 32))
            img_obj.show(title="Original")
            img_mode = img_obj.mode
            data_bytes = np.array(img_obj).tobytes()
        else: continue

        # Transmission
        audio, c_len = modem.encode(data_bytes)
        print(f"[*] Compressed: {c_len}B. Playing ({len(audio)/FS:.1f}s)...")
        rec = sd.playrec(np.concatenate([np.zeros(FS), audio, np.zeros(FS)]), FS, channels=1)
        sd.wait()
        
        # Decoding
        try:
            received_bytes = modem.decode(rec.flatten())
            ber = modem.show_calibration()
            
            if ber == 0:
                raw = zlib.decompress(received_bytes)
                if mode_select == '1':
                    print(f"\n>> Received Text: {raw.decode('utf-8')}")
                else:
                    final_img = Image.frombytes(img_mode, (32, 32), raw)
                    final_img.show(title="Received")
                    print("\n>> Image Reconstruction Success.")
            else:
                print(f"\n!! BER: {ber:.2f}%. Check hardware distance.")
        except Exception as e:
            print(f"\n!! Recovery Error: {e}")

if __name__ == "__main__":
    main()

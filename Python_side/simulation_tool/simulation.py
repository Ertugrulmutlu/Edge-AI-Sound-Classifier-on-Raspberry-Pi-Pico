import time, math, json
import numpy as np
import sounddevice as sd
import serial

# ====== CONFIG ======
COM_PORT   = "COM3"     # Pico USB-CDC port
BAUD       = 115200
SR         = 16000      # must match data_extraction.py
SNIPPET_S  = 1.5        # window length in seconds
HOP_S      = 0.5        # hop length in seconds
FRAME_S    = 0.025      # frame size in seconds
HOP_FRAME  = 0.010      # frame hop in seconds
ROLLOFF_P  = 0.85
BANDS      = [
    (300,600),(600,900),(900,1200),(1200,1500),
    (1500,1800),(1800,2200),(2200,2600),(2600,3000),
    (3000,3400),(3400,3800),(3800,4300),(4300,4800)
]
PRINT_LAST = 3
# =====================

# --- helpers (aligned with data_extraction.py) ---
def goertzel_pow(frame, sr, f):
    w = 2*math.pi*f/sr
    c = math.cos(w)
    s0 = s1 = s2 = 0.0
    for x in frame:
        s0 = x + 2*c*s1 - s2
        s2, s1 = s1, s0
    return s1*s1 + s2*s2 - 2*c*s1*s2

def band_energy(frame, sr, lo, hi, n=3):
    freqs = np.linspace(lo, hi, n)
    return float(np.mean([goertzel_pow(frame, sr, f) for f in freqs]))

def frame_stack(y, sr, frame_s=FRAME_S, hop_s=HOP_FRAME):
    n = int(frame_s*sr); h = int(hop_s*sr)
    pad = (-(len(y)-n) % h) if len(y) >= n else (n - len(y))
    y2 = np.pad(y, (0, pad), mode="constant")
    T = 1 + (len(y2)-n)//h
    out = np.empty((T, n), dtype=np.float32)
    i = 0
    for t in range(T):
        out[t] = y2[i:i+n]; i += h
    return out  # [T, n]

def extract_features_live(snippet, sr):
    """
    Matches data_extraction.py::per_snippet_features:
    - Peak normalize (0.707)
    - Pre-emphasis (a=0.95)
    - Framing (25ms / 10ms), Hanning window
    - FFT-based stats, ZCR, flatness
    - 12 Goertzel band energies (mean & std), each internally z-scored
    Order:
      [be_mean(12,zscore) , be_std(12,zscore) ,
       rms_mean, rms_std ,
       centroid_mean, centroid_std ,
       rolloff_mean, rolloff_std ,
       zcr_mean, zcr_std ,
       flatness_mean]
    Total 33 floats.
    """
    y = np.asarray(snippet, dtype=np.float32)
    if y.ndim > 1: y = y.mean(axis=1)

    # Peak normalize
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = np.clip(y/peak*0.707, -1, 1)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.95*y[:-1]).astype(np.float32)

    # Framing + window
    frames = frame_stack(y, sr)
    win = np.hanning(frames.shape[1]).astype(np.float32)
    frames_w = frames * win

    # RMS
    rms = np.sqrt(np.mean(frames_w**2, axis=1) + 1e-12)

    # FFT magnitude
    F = np.fft.rfft(frames_w, axis=1)
    mag = np.abs(F) + 1e-12
    freqs = np.fft.rfftfreq(frames_w.shape[1], 1/sr)

    # Centroid
    centroid = (mag*freqs).sum(axis=1) / (mag.sum(axis=1))

    # Rolloff
    target = 0.85 * mag.sum(axis=1)
    csum = np.cumsum(mag, axis=1)
    roll_idx = np.argmax(csum >= target[:, None], axis=1)
    rolloff = freqs[roll_idx]

    # Zero crossing rate
    sgn = np.sign(frames_w); sgn[sgn == 0] = 1
    zc = (np.diff(sgn, axis=1) != 0).mean(axis=1).astype(np.float32)

    # Spectral flatness
    flat = np.exp(np.mean(np.log(mag), axis=1)) / (mag.mean(axis=1) + 1e-12)

    # Goertzel bands
    be_list = []
    for (lo, hi) in BANDS:
        vals = np.array([band_energy(fr, sr, lo, hi, n=3) for fr in frames_w], dtype=np.float32)
        be_list.append(vals)
    be = np.stack(be_list, axis=1)  # [T, 12]

    # Pool -> single vector
    be_mean = be.mean(axis=0); be_std = be.std(axis=0) + 1e-8
    be_mean = (be_mean - be_mean.mean()) / (be_mean.std() + 1e-8)
    be_std  = (be_std  - be_std.mean())  / (be_std.std() + 1e-8)

    feats = []
    feats.extend(be_mean.tolist())                  
    feats.extend(be_std.tolist())                   
    feats.extend([float(rms.mean()), float(rms.std()+1e-8)])                 
    feats.extend([float(centroid.mean()), float(centroid.std()+1e-8)])       
    feats.extend([float(rolloff.mean()), float(rolloff.std()+1e-8)])         
    feats.extend([float(zc.mean()), float(zc.std()+1e-8)])                   
    feats.append(float(flat.mean()))                                         

    return np.array(feats, dtype=np.float32)  # (33,)

def main():
    feat_dim = 33
    snippet_len = int(SR * SNIPPET_S)
    hop_len     = int(SR * HOP_S)

    print(f"[INFO] SR={SR}, snippet={SNIPPET_S}s({snippet_len}), hop={HOP_S}s({hop_len}), feat_dim={feat_dim}")

    ser = serial.Serial(COM_PORT, BAUD, timeout=0.05)
    time.sleep(0.3)
    ser.write(b"PING\n"); time.sleep(0.1)
    rx = ser.read(2048).decode(errors="ignore")
    print("[OK] PONG received." if "PONG" in rx else "[WARN] No PONG, continuing.")

    ser.write(b"RESET\n")

    ring = np.zeros(snippet_len, dtype=np.float32)
    write_pos = 0
    filled = 0

    def audio_cb(indata, frames, time_info, status):
        nonlocal ring, write_pos, filled
        if status: print(f"[SD] {status}", flush=True)
        x = indata[:, 0].copy()
        n = len(x)
        end = write_pos + n
        if end <= len(ring):
            ring[write_pos:end] = x
        else:
            first = len(ring) - write_pos
            ring[write_pos:] = x[:first]
            ring[:end % len(ring)] = x[first:]
        write_pos = (write_pos + n) % len(ring)
        filled = min(len(ring), filled + n)

    stream = sd.InputStream(
        samplerate=SR, channels=1, dtype='float32',
        blocksize=hop_len, callback=audio_cb
    )

    print("[INFO] Listening from microphone. Press Ctrl+C to exit.")
    with stream:
        while True:
            if filled >= len(ring):
                if write_pos == 0:
                    snippet = ring.copy()
                else:
                    snippet = np.concatenate([ring[write_pos:], ring[:write_pos]])

                fv = extract_features_live(snippet, SR)
                line = ",".join(f"{float(v):.6f}" for v in fv.tolist()) + "\n"
                ser.write(line.encode())

                t_end = time.time() + 0.03
                buf = []
                while time.time() < t_end:
                    ln = ser.readline()
                    if ln:
                        buf.append(ln.decode(errors="ignore").strip())
                if buf:
                    for ln in buf[-PRINT_LAST:]:
                        print(ln)

            time.sleep(0.005)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] user exit")

import os, glob, json, math, re
import numpy as np, pandas as pd, soundfile as sf

# ==== CONFIG ==========================================================
PREP_DIRS = {
    "doorbell":    "./dataset/prep/doorbel",
    "smoke_alarm": "./dataset/prep/fire_alarm",
    "baby":        "./dataset/prep/baby_cry",
    "other":       "./dataset/prep/Negativ",
}
OUT_CSV = "./features/featuresv1.csv"
SR = 16000
FRAME_S = 0.025
HOP_S   = 0.010
BANDS = [
    (300,600),(600,900),(900,1200),(1200,1500),
    (1500,1800),(1800,2200),(2200,2600),(2600,3000),
    (3000,3400),(3400,3800),(3800,4300),(4300,4800)
]
# =====================================================================

def goertzel_pow(frame, sr, f):
    w = 2*math.pi*f/sr
    s0=s1=s2=0.0
    for x in frame:
        s0 = x + 2*math.cos(w)*s1 - s2
        s2, s1 = s1, s0
    return s1*s1 + s2*s2 - 2*math.cos(w)*s1*s2

def band_energy(frame, sr, lo, hi, n=3):
    freqs = np.linspace(lo, hi, n)
    return float(np.mean([goertzel_pow(frame, sr, f) for f in freqs]))

def frame_stack(y, sr, frame_s=FRAME_S, hop_s=HOP_S):
    n = int(frame_s*sr); h = int(hop_s*sr)
    pad = (-(len(y)-n) % h) if len(y) >= n else (n-len(y))
    y2 = np.pad(y, (0, pad), mode="constant")
    T = 1 + (len(y2)-n)//h
    out = np.empty((T, n), dtype=np.float32)
    i = 0
    for t in range(T):
        out[t] = y2[i:i+n]; i += h
    return out

def extract_source_from_filename(path):
    # expects "SRC__UUID.wav"; fallback: filename stem
    stem = os.path.splitext(os.path.basename(path))[0]
    if "__" in stem:
        return stem.split("__", 1)[0]
    return os.path.basename(os.path.dirname(path)) + "_" + stem[:6]

def per_snippet_features(path):
    y, _ = sf.read(path)
    if y.ndim>1: y = y.mean(axis=1)
    peak = np.max(np.abs(y)) + 1e-9
    y = np.clip(y/peak*0.707, -1, 1)
    y = np.append(y[0], y[1:]-0.95*y[:-1]).astype(np.float32)

    frames = frame_stack(y, SR)
    win = np.hanning(frames.shape[1]).astype(np.float32)
    frames_w = frames * win

    rms = np.sqrt(np.mean(frames_w**2, axis=1) + 1e-12)
    F = np.fft.rfft(frames_w, axis=1)
    mag = np.abs(F) + 1e-12
    freqs = np.fft.rfftfreq(frames_w.shape[1], 1/SR)
    centroid = (mag*freqs).sum(axis=1)/mag.sum(axis=1)

    target = 0.85*mag.sum(axis=1)
    csum = np.cumsum(mag, axis=1)
    roll_idx = np.argmax(csum >= target[:,None], axis=1)
    rolloff = freqs[roll_idx]

    sgn = np.sign(frames_w); sgn[sgn==0]=1
    zc = (np.diff(sgn, axis=1)!=0).mean(axis=1).astype(np.float32)

    flat = np.exp(np.mean(np.log(mag), axis=1)) / (mag.mean(axis=1)+1e-12)

    be = []
    for (lo,hi) in BANDS:
        vals = np.array([band_energy(fr, SR, lo, hi, n=3) for fr in frames_w], dtype=np.float32)
        be.append(vals)
    be = np.stack(be, axis=1)  # [T, nb]

    feats = []
    be_mean = be.mean(axis=0); be_std = be.std(axis=0)+1e-8
    be_mean = (be_mean - be_mean.mean()) / (be_mean.std()+1e-8)
    be_std  = (be_std  - be_std.mean())  / (be_std.std()+1e-8)
    feats.extend(be_mean.tolist()); feats.extend(be_std.tolist())
    feats.extend([float(rms.mean()), float(rms.std()+1e-8)])
    feats.extend([float(centroid.mean()), float(centroid.std()+1e-8)])
    feats.extend([float(rolloff.mean()), float(rolloff.std()+1e-8)])
    feats.extend([float(zc.mean()), float(zc.std()+1e-8)])
    feats.append(float(flat.mean()))

    return np.array(feats, dtype=np.float32)

def main():
    rows = []
    for label, d in PREP_DIRS.items():
        if not os.path.isdir(d): 
            print("[SKIP]", label, d); continue
        files = glob.glob(os.path.join(d, "*.wav"))
        for p in files:
            fv = per_snippet_features(p)
            src = extract_source_from_filename(p)
            rows.append({"path": p, "label": label, "source": src, "feat": json.dumps(fv.tolist())})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV} with {len(df)} items; cols={list(df.columns)}")

if __name__ == "__main__":
    main()

import os, glob, uuid, numpy as np, soundfile as sf, librosa, random, re

# ==== CONFIG ==========================================================
CLASSES = [
    ("./dataset/raw/door_bel",    "./dataset/prep/doorbel",    2.0, 0.25, -55.0, 240),
    ("./dataset/raw/fire_alarm", "./dataset/prep/fire_alarm", 2.0, 0.25, -55.0, 240),
    ("./dataset/raw/baby_cry",   "./dataset/prep/baby_cry",   1.5, 0.25, -50.0, 240),
    ("./dataset/raw/Negativ",    "./dataset/prep/Negativ",    2.0, 0.25, -55.0, 320),
]
SR = 16000
PEAK_NORM = 0.707
SEED = 42
# =====================================================================

random.seed(SEED)

def rms_db(x): 
    return 20*np.log10(np.sqrt(np.mean(x**2)+1e-12)+1e-12)

def norm_peak(y):
    p = np.max(np.abs(y)) + 1e-9
    return np.clip(y / p * PEAK_NORM, -1, 1)

def sanitize(name):
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r'[^A-Za-z0-9]+', '_', base)
    return base[:40] if base else "src"

def gather_files(d):
    files = []
    for ext in ("*.wav","*.mp3","*.mp4","*.m4a","*.aac"):
        files += glob.glob(os.path.join(d, ext))
    return files

def cut_file(path, snip_s, hop_s, rms_thr_db, out_dir):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = norm_peak(y)
    win = int(snip_s*SR); hop = int(hop_s*SR)
    saved = 0
    src = sanitize(path)
    for i in range(0, max(0, len(y)-win+1), hop):
        seg = y[i:i+win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win-len(seg)))
        if rms_db(seg) < rms_thr_db:
            continue
        name = f"{src}__{uuid.uuid4().hex[:8]}.wav"
        sf.write(os.path.join(out_dir, name), seg, SR, subtype="PCM_16")
        saved += 1
    return saved

def main():
    for in_dir, out_dir, snip_s, hop_s, rms_thr_db, max_snips in CLASSES:
        os.makedirs(out_dir, exist_ok=True)
        files = gather_files(in_dir)
        if not files:
            print(f"[SKIP] no files: {in_dir}"); continue

        saved = 0; total = 0
        segs = []
        for f in files:
            y, _ = librosa.load(f, sr=SR, mono=True)
            y = norm_peak(y)
            win = int(snip_s*SR); hop = int(hop_s*SR)
            src = sanitize(f)
            for i in range(0, max(0, len(y)-win+1), hop):
                seg = y[i:i+win]
                if len(seg) < win: seg = np.pad(seg, (0, win-len(seg)))
                if rms_db(seg) < rms_thr_db: continue
                segs.append((src, seg))
        total = len(segs)
        if total == 0:
            print(f"[WARN] zero candidates: {in_dir}"); continue
        if total > max_snips:
            idxs = np.linspace(0, total-1, num=max_snips, dtype=int)
            segs = [segs[i] for i in idxs]
        for src, seg in segs:
            name = f"{src}__{uuid.uuid4().hex[:8]}.wav"
            sf.write(os.path.join(out_dir, name), seg, SR, subtype="PCM_16")
            saved += 1
        print(f"[OK] {in_dir} -> {saved} ({total} cand) -> {out_dir}")

if __name__ == "__main__":
    import numpy as np
    main()

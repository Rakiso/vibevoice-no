#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from speechbrain.pretrained import EncoderClassifier


TARGET_SR = 16000  # ECAPA expects 16 kHz


def read_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)


def compute_embedding(classifier: EncoderClassifier, audio: np.ndarray, sr: int) -> torch.Tensor:
    audio16 = resample(audio, sr, TARGET_SR)
    tensor = torch.from_numpy(audio16).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(tensor)
    return emb.squeeze(0).squeeze(0).cpu()


def build_train_speaker_bank(jsonl_path: str, classifier: EncoderClassifier) -> Dict[str, torch.Tensor]:
    bank: Dict[str, List[torch.Tensor]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            spk = row.get("speaker") or row.get("id") or "unknown"
            audio, sr = read_audio_mono(row["audio"])
            emb = compute_embedding(classifier, audio, sr)
            bank.setdefault(spk, []).append(emb)
    # Average per speaker
    spk_mean: Dict[str, torch.Tensor] = {
        spk: torch.stack(embs, dim=0).mean(dim=0) for spk, embs in bank.items()
    }
    return spk_mean


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float((a * b).sum().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    # Support both --gen and legacy --candidate_wav
    p.add_argument("--gen", required=False)
    p.add_argument("--candidate_wav", required=False)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--scan_dir", type=str, default=None, help="If set, scan a folder of generations")
    args = p.parse_args()

    gen_path = args.gen or args.candidate_wav
    if not gen_path and not args.scan_dir:
        raise SystemExit("Provide --gen (or --candidate_wav) or --scan_dir")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    bank = build_train_speaker_bank(args.train_jsonl, classifier)

    def check_one(path: str) -> Tuple[float, str]:
        audio, sr = read_audio_mono(path)
        emb = compute_embedding(classifier, audio, sr)
        sims = [(cosine_similarity(emb, ref), spk) for spk, ref in bank.items()]
        sims.sort(reverse=True, key=lambda x: x[0])
        return sims[0][0], sims[0][1]

    flagged = False
    if args.scan_dir:
        for fname in os.listdir(args.scan_dir):
            if not fname.lower().endswith((".wav", ".flac")):
                continue
            path = os.path.join(args.scan_dir, fname)
            sim, spk = check_one(path)
            print(f"{fname}: max cosine={sim:.3f} vs speaker={spk}")
            if sim > args.threshold:
                flagged = True
    else:
        sim, spk = check_one(gen_path)
        print(f"{os.path.basename(gen_path)}: max cosine={sim:.3f} vs speaker={spk}")
        if sim > args.threshold:
            flagged = True

    if flagged:
        print("Warning: similarity above threshold. Potential voice memorization.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()



"""
Embedding endpoint'ini tek çağrıda doğrular.
============================================

build_dataset.py'yi 1000+ chunk ile çalıştırmadan önce şunları kontrol eder:
  - endpoint ayakta mı, kimlik doğrulama geçiyor mu
  - model adı sunucudakiyle uyuşuyor mu (/v1/models listelenir)
  - input_type zorunlu mu (asimetrik model mi)
  - dönen vektör boyutu kaç, normalize mi
  - query ve passage uzayları gerçekten farklı mı

Kullanım:
    set EMBED_BASE_URL=https://integrate.api.nvidia.com/v1
    set EMBED_API_KEY=nvapi-...
    python benchmark/check_endpoint.py

    python benchmark/check_endpoint.py --embed-base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import requests

OK, FAIL, WARN = "[OK]  ", "[HATA]", "[UYARI]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-base-url", default=os.getenv("EMBED_BASE_URL"))
    p.add_argument("--embed-api-key", default=os.getenv("EMBED_API_KEY"))
    p.add_argument("--embed-model", default="nvidia/nv-embedqa-e5-v5")
    args = p.parse_args()

    if not args.embed_base_url:
        sys.exit(
            "HATA: EMBED_BASE_URL yok.\n"
            "  NVIDIA barındırmalı : https://integrate.api.nvidia.com/v1\n"
            "  Lokal NIM konteyneri: http://localhost:8000/v1"
        )

    base = args.embed_base_url.rstrip("/")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if args.embed_api_key:
        headers["Authorization"] = f"Bearer {args.embed_api_key}"

    print(f"endpoint : {base}")
    print(f"model    : {args.embed_model}")
    print(f"api key  : {'var' if args.embed_api_key else 'YOK'}\n")

    # --- 1. Sunulan modeller ---
    try:
        r = requests.get(f"{base}/models", headers=headers, timeout=30)
        if r.ok:
            ids = [m["id"] for m in r.json().get("data", [])]
            print(f"{OK} /models erişilebilir ({len(ids)} model)")
            if ids and args.embed_model not in ids:
                print(f"{WARN} '{args.embed_model}' listede yok. Sunulanlar:")
                for i in ids[:15]:
                    print(f"         {i}")
            elif ids:
                print(f"{OK} model adı sunucuda mevcut")
        else:
            print(f"{WARN} /models -> HTTP {r.status_code} (bazı sunucular bu ucu açmaz)")
    except Exception as exc:  # noqa: BLE001
        print(f"{WARN} /models sorgulanamadı: {exc}")

    # --- 2. Embedding çağrıları ---
    def embed(texts: list[str], input_type: str | None):
        payload: dict = {
            "model": args.embed_model,
            "input": texts,
            "encoding_format": "float",
            "truncate": "END",
        }
        if input_type:
            payload["input_type"] = input_type
        r = requests.post(f"{base}/embeddings", json=payload, headers=headers, timeout=60)
        return r

    probe = ["Allergic rhinitis is triggered by pollen exposure."]

    r = embed(probe, "passage")
    if not r.ok:
        print(f"\n{FAIL} passage embedding basarisiz -> HTTP {r.status_code}")
        print(f"       {r.text[:400]}")
        sys.exit(1)
    v_pass = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    print(f"\n{OK} passage embedding calisti  dim={v_pass.shape[0]}")

    r = embed(["What triggers allergic rhinitis?"], "query")
    if not r.ok:
        print(f"{FAIL} query embedding basarisiz -> HTTP {r.status_code}\n       {r.text[:400]}")
        sys.exit(1)
    v_query = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)
    print(f"{OK} query embedding calisti    dim={v_query.shape[0]}")

    if v_pass.shape != v_query.shape:
        print(f"{FAIL} boyutlar uyusmuyor: {v_pass.shape} vs {v_query.shape}")
        sys.exit(1)

    # --- 3. input_type zorunlu mu? ---
    r = embed(probe, None)
    if r.ok:
        print(f"{OK} input_type opsiyonel (simetrik model olabilir)")
        print("       -> --no-input-type kullanabilirsin")
    else:
        print(f"{OK} input_type ZORUNLU (asimetrik model, HTTP {r.status_code})")
        print("       -> --no-input-type KULLANMA")

    # --- 4. Normalize mi, asimetri var mi? ---
    n_p, n_q = float(np.linalg.norm(v_pass)), float(np.linalg.norm(v_query))
    normed = abs(n_p - 1.0) < 1e-2 and abs(n_q - 1.0) < 1e-2
    print(f"{OK} vektör normu: passage={n_p:.4f} query={n_q:.4f} "
          f"({'zaten normalize' if normed else 'normalize DEGIL, script normalize edecek'})")

    same = embed(probe, "query")
    if same.ok:
        v_same = np.asarray(same.json()["data"][0]["embedding"], dtype=np.float32)
        cos = float(v_pass @ v_same / (np.linalg.norm(v_pass) * np.linalg.norm(v_same)))
        print(f"{OK} ayni metin passage vs query benzerligi: {cos:.4f}")
        if cos > 0.999:
            print(f"{WARN} input_type ciktiyi degistirmiyor — model simetrik olabilir")

    dim = v_pass.shape[0]
    print(f"\nSONUC: dim={dim}")
    print(f"  OpenSearch index 'dim' ayari {dim} olmali (app.py su an 1024).")
    print("\nHazirsan:")
    print(f"  python benchmark/build_dataset.py --embed-model {args.embed_model}")


if __name__ == "__main__":
    main()

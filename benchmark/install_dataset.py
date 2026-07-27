"""
Üretilen custom dataset'i VectorDBBench'in beklediği yere kurar.
=================================================================

VectorDBBench dataset dosyalarını şu yoldan okur:

    {DATASET_LOCAL_DIR}/{dataset_name.lower()}/{dir}/train.parquet

Bu script:
  1. benchmark/dataset/*.parquet dosyalarını o yola kopyalar
  2. vectordb_bench/custom/custom_case.json dosyasına case tanımını yazar

Kullanım:
    python benchmark/install_dataset.py
    python benchmark/install_dataset.py --name deneysel --dataset-dir deneysel_bge_m3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["train.parquet", "test.parquet", "neighbors.parquet"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=ROOT / "benchmark" / "dataset")
    p.add_argument("--name", default="deneysel", help="Dataset adı (klasör: küçük harf)")
    p.add_argument(
        "--dataset-dir",
        default=None,
        help="Alt klasör adı. Verilmezse embed modelinden türetilir "
        "(orn. deneysel_nv_embedqa_e5_v5).",
    )
    p.add_argument(
        "--dataset-local-dir",
        type=Path,
        default=None,
        help="VectorDBBench DATASET_LOCAL_DIR. Verilmezse paketten okunur.",
    )
    args = p.parse_args()

    # --- Girdileri doğrula ---
    meta_path = args.src / "dataset_meta.json"
    if not meta_path.exists():
        sys.exit(f"HATA: {meta_path} yok. Önce build_dataset.py çalıştır.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Klasör adını embed modelinden türet: farklı modellerin dataset'leri
    # birbirinin üstüne yazmasın.
    if not args.dataset_dir:
        slug = re.sub(r"[^a-z0-9]+", "_", meta["embed_model"].lower()).strip("_")
        args.dataset_dir = f"{args.name.lower()}_{slug}"

    missing = [f for f in REQUIRED if not (args.src / f).exists()]
    if missing:
        sys.exit(f"HATA: eksik dosya(lar): {', '.join(missing)}")

    # --- VectorDBBench'i bul ---
    try:
        import vectordb_bench
        from vectordb_bench import config
    except ImportError:
        sys.exit(
            "HATA: vectordb_bench kurulu değil.\n"
            "  pip install vectordb-bench[opensearch]"
        )

    dataset_local_dir = Path(args.dataset_local_dir or config.DATASET_LOCAL_DIR)
    target = dataset_local_dir / args.name.lower() / args.dataset_dir
    target.mkdir(parents=True, exist_ok=True)

    for f in REQUIRED:
        shutil.copy2(args.src / f, target / f)
        print(f"  kopyalandi  {f}  ->  {target / f}")

    # --- custom_case.json yaz ---
    case = {
        "name": f"{args.name} (Performance Case)",
        "description": (
            f"DeneyselRAG kendi PDF korpusu. "
            f"chunk={meta['chunk_size']}/{meta['chunk_overlap']}, "
            f"embed={meta['embed_model']} ({meta['embed_backend']}), "
            f"{meta['size']} vektör x {meta['dim']} boyut."
        ),
        "load_timeout": 36000,
        "optimize_timeout": 36000,
        "dataset_config": {
            "name": args.name,
            "dir": args.dataset_dir,
            "size": meta["size"],
            "dim": meta["dim"],
            "metric_type": meta["metric_type"],
            "file_count": 1,
            "use_shuffled": False,
            "with_gt": True,
            "train_name": "train",
            "test_name": "test",
            "gt_name": "neighbors",
            "train_id_name": "id",
            "train_col_name": "emb",
            "test_col_name": "emb",
            "gt_col_name": "neighbors_id",
        },
    }

    custom_config_path = Path(config.CUSTOM_CONFIG_DIR)
    custom_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Var olan diğer case'leri koru, aynı isimliyi güncelle.
    existing = []
    if custom_config_path.exists():
        try:
            existing = json.loads(custom_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing = [
        c
        for c in existing
        if c.get("dataset_config", {}).get("name") != args.name
        and c.get("case_type") != "streaming"
    ]
    existing.append(case)
    custom_config_path.write_text(
        json.dumps(existing, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Çalıştırma scriptleri bu iki alanı meta'dan okuyor — geri yaz.
    meta["dataset_name"] = args.name
    meta["dataset_dir"] = args.dataset_dir
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    pkg_dir = Path(vectordb_bench.__file__).parent
    print()
    print("KURULUM TAMAM")
    print(f"  vectordb_bench    : {pkg_dir}")
    print(f"  DATASET_LOCAL_DIR : {dataset_local_dir}")
    print(f"  dataset           : {target}")
    print(f"  custom_case.json  : {custom_config_path}")
    print()
    print(f"  size={meta['size']}  dim={meta['dim']}  metric={meta['metric_type']}  gt_k={meta['gt_k']}")
    print()
    print(f"UYARI: benchmark'i --k {meta['gt_k']} veya altiyla calistir "
          f"(ground truth {meta['gt_k']} komsu iceriyor).")
    print()
    print("Sıradaki adım:  benchmark/run_opensearch.sh   (veya .ps1)")


if __name__ == "__main__":
    main()

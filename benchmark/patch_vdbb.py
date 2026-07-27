"""
VectorDBBench'in OSSOpenSearch client'ındaki iki engeli düzeltir.
=================================================================

Bu yamalar upstream'de (zilliztech/VectorDBBench) mevcut olan davranıştan
kaynaklanıyor; lokal, self-signed sertifikalı bir OpenSearch'e bağlanmayı
imkânsız kılıyorlar.

YAMA 1 — metric_type_name geçirilmiyor
  backend/clients/oss_opensearch/cli.py, OSSOpenSearchIndexConfig'i kurarken
  metric_type_name alanını doldurmuyor (aws_opensearch/cli.py dolduruyor).
  Sonuç: config.py::parse_metric() içinde
      self.metric_type = MetricType[self.metric_type_name.upper()]
  satırı None üzerinde .upper() çağırıp AttributeError ile patlıyor.

YAMA 2 — SSL yalnızca port 443'te açılıyor
  backend/clients/oss_opensearch/config.py::to_dict() içinde
      use_ssl = self.port == 443
      verify_certs = use_ssl
  sabit. OpenSearch'ün varsayılan dev kurulumu 9200 portunda HTTPS konuşur ve
  sertifikası self-signed'dır. Bu haliyle ne bağlanılabilir ne de sertifika
  doğrulaması atlanabilir.
  Yama, davranışı ortam değişkenleriyle kontrol edilebilir yapar:
      OSS_OPENSEARCH_USE_SSL=true|false
      OSS_OPENSEARCH_VERIFY_CERTS=true|false
  Değişken tanımlı değilse eski davranış aynen korunur.

Kullanım:
    python benchmark/patch_vdbb.py           # yamaları uygula
    python benchmark/patch_vdbb.py --check   # sadece durumu göster
    python benchmark/patch_vdbb.py --revert  # .bak dosyalarından geri al
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARK = "# [DeneyselRAG-patch]"

PATCH1_OLD = """        db_case_config=OSSOpenSearchIndexConfig(
            number_of_shards=parameters["number_of_shards"],"""

PATCH1_NEW = f"""        db_case_config=OSSOpenSearchIndexConfig(
            metric_type_name=parameters["metric_type"],  {MARK}
            number_of_shards=parameters["number_of_shards"],"""

PATCH2_OLD = """    def to_dict(self) -> dict:
        use_ssl = self.port == 443"""

PATCH2_NEW = f"""    def to_dict(self) -> dict:
        import os as _os  {MARK}

        def _envbool(key, default):  {MARK}
            v = _os.getenv(key)  {MARK}
            if v is None:  {MARK}
                return default  {MARK}
            return v.strip().lower() in ("1", "true", "yes", "on")  {MARK}

        use_ssl = _envbool("OSS_OPENSEARCH_USE_SSL", self.port == 443)  {MARK}"""

PATCH3_OLD = """            "use_ssl": use_ssl,
            "http_compress": True,
            "verify_certs": use_ssl,"""

PATCH3_NEW = f"""            "use_ssl": use_ssl,
            "http_compress": True,
            "verify_certs": _envbool("OSS_OPENSEARCH_VERIFY_CERTS", use_ssl),  {MARK}"""


def locate() -> tuple[Path, Path]:
    try:
        import vectordb_bench
    except ImportError:
        sys.exit(
            "HATA: vectordb_bench kurulu degil.\n"
            "  pip install vectordb-bench[opensearch]"
        )
    pkg = Path(vectordb_bench.__file__).parent
    cli = pkg / "backend" / "clients" / "oss_opensearch" / "cli.py"
    cfg = pkg / "backend" / "clients" / "oss_opensearch" / "config.py"
    for f in (cli, cfg):
        if not f.exists():
            sys.exit(f"HATA: beklenen dosya yok: {f}")
    return cli, cfg


def apply_to(path: Path, pairs: list[tuple[str, str]]) -> int:
    text = path.read_text(encoding="utf-8")
    applied = 0
    for old, new in pairs:
        if new.split("\n")[0] in text and MARK in text and old not in text:
            continue  # zaten yamalı
        if old not in text:
            print(f"  ATLANDI  beklenen kod bulunamadi ({path.name}). "
                  f"VectorDBBench surumu degismis olabilir.")
            continue
        text = text.replace(old, new, 1)
        applied += 1
    if applied:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"  yedek     {backup.name}")
        path.write_text(text, encoding="utf-8")
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Sadece durumu goster")
    ap.add_argument("--revert", action="store_true", help=".bak dosyalarindan geri al")
    args = ap.parse_args()

    cli, cfg = locate()
    print(f"vectordb_bench: {cli.parent}\n")

    if args.revert:
        for f in (cli, cfg):
            bak = f.with_suffix(f.suffix + ".bak")
            if bak.exists():
                shutil.copy2(bak, f)
                bak.unlink()
                print(f"  geri alindi  {f.name}")
            else:
                print(f"  yedek yok    {f.name}")
        return

    if args.check:
        for f, label in ((cli, "YAMA 1 metric_type_name"), (cfg, "YAMA 2 SSL")):
            state = "UYGULANMIS" if MARK in f.read_text(encoding="utf-8") else "UYGULANMAMIS"
            print(f"  {label:28s} {state}")
        return

    n = apply_to(cli, [(PATCH1_OLD, PATCH1_NEW)])
    n += apply_to(cfg, [(PATCH2_OLD, PATCH2_NEW), (PATCH3_OLD, PATCH3_NEW)])

    print(f"\n{n} yama uygulandi." if n else "\nDegisiklik yok (muhtemelen zaten yamali).")
    print("\nHTTPS + self-signed sertifikali lokal OpenSearch icin:")
    print("  set OSS_OPENSEARCH_USE_SSL=true")
    print("  set OSS_OPENSEARCH_VERIFY_CERTS=false")


if __name__ == "__main__":
    main()

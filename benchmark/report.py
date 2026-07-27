"""
VectorDBBench sonuç JSON'ını okunabilir bir rapora çevirir.
===========================================================

VectorDBBench sonuçları ham JSON olarak yazar ve manşet `qps` değeri
EŞZAMANLI TEPE QPS'tir. Tek istemcili (seri) bir Qdrant koşusuyla
karşılaştırmak isteyenler için bu iki sayı farklı şeyleri ölçer; rapor
ikisini de ayrı bölümlerde gösterir.

Birimler: VectorDBBench gecikmeleri SANİYE cinsinden saklar; burada ms'ye
çevrilir.

Kullanım:
    python benchmark/report.py                        # en son sonucu bul
    python benchmark/report.py --result <dosya.json>
    python benchmark/report.py --all                  # tüm koşuları tablola
    python benchmark/report.py --markdown rapor.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "benchmark" / "dataset"


# --------------------------------------------------------------------------
def find_results(explicit: Path | None) -> list[Path]:
    if explicit:
        if not explicit.exists():
            sys.exit(f"HATA: {explicit} yok.")
        return [explicit]

    try:
        from vectordb_bench import config

        results_dir = Path(config.RESULTS_LOCAL_DIR)
    except ImportError:
        sys.exit("HATA: vectordb_bench kurulu degil; --result ile dosya yolu ver.")

    files = sorted(
        results_dir.rglob("result_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Depoyla gelen referans sonuçları ayıkla: yalnızca bizim koşularımız.
    files = [f for f in files if "leaderboard" not in f.name]
    if not files:
        sys.exit(f"HATA: {results_dir} altinda sonuc bulunamadi.")
    return files


def read_failure_reasons(limit: int = 3) -> list[str]:
    """VectorDBBench log'undan gerçek hata sebebini çeker.

    Sonuç JSON'ı yalnızca 'label: x' der; asıl istisna log'da kalır. Kullanıcıyı
    log dosyasını elle taramaya bırakmak yerine buradan yüzeye çıkar.
    """
    candidates = [
        Path("logs/vectordb_bench.log"),                 # cwd'ye göre
        ROOT / "benchmark" / "logs" / "vectordb_bench.log",
        ROOT / "logs" / "vectordb_bench.log",
    ]
    log_file = next((p for p in candidates if p.exists()), None)
    if log_file is None:
        return []

    reasons: list[str] = []
    try:
        # Log büyük olabilir; sondan oku.
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for line in reversed(lines):
        if "failed to run, reason=" in line:
            reason = line.split("failed to run, reason=", 1)[1].strip()
            reason = reason.split(" (interface.py")[0].strip()
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= limit:
                break
    return reasons


# Bilinen hata imzaları -> ne yapılmalı
KNOWN_FIXES = [
    (
        "'NoneType' object has no attribute 'upper'",
        "Yama 1 uygulanmamış (metric_type_name geçirilmiyor).\n"
        "      ÇÖZÜM:  python benchmark\\patch_vdbb.py",
    ),
    (
        "ConnectionError",
        "OpenSearch'e ulaşılamıyor. Şema/port kontrol et; HTTP ise -NoSsl kullan.",
    ),
    (
        "AuthenticationException",
        ".env içindeki OPENSEARCH_USER / OPENSEARCH_PASSWORD hatalı.",
    ),
    (
        "index_not_found",
        "Dataset kurulmamış olabilir:  python benchmark\\install_dataset.py",
    ),
    (
        "certificate verify failed",
        "Self-signed sertifika. OSS_OPENSEARCH_VERIFY_CERTS=false (yama 2 gerekli).",
    ),
]


def load_memory_delta() -> dict | None:
    before_p, after_p = DATASET_DIR / "mem_before.json", DATASET_DIR / "mem_after.json"
    if not (before_p.exists() and after_p.exists()):
        return None
    before = json.loads(before_p.read_text(encoding="utf-8"))
    after = json.loads(after_p.read_text(encoding="utf-8"))

    def delta(key: str) -> float | None:
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            return None
        return (a - b) / 1e6  # MB

    return {
        "jvm_heap": delta("jvm_heap_used_bytes"),
        "segments": delta("segments_memory_bytes"),
        "knn_graph": delta("knn_graph_memory_bytes"),
        "store": delta("store_size_bytes"),
        "knn_available": after.get("knn_stats_available", False),
    }


# --------------------------------------------------------------------------
def diagnose(case: dict, mem: dict | None) -> tuple[list[str], list[str]]:
    """(agir_hatalar, uyarilar) döner.

    VectorDBBench başarısız bir case için de sonuç dosyası yazar; metrikler
    sıfır kalır. Ağır hata varsa rapor basılmaz — sıfırları geçerli ölçüm gibi
    göstermek yanıltıcı olur. Uyarılar raporu engellemez.
    """
    hard: list[str] = []
    warn: list[str] = []
    m = case.get("metrics", {})

    if case.get("label") == "x":
        hard.append("VectorDBBench bu case'i BAŞARISIZ olarak işaretledi (label='x').")

    if not (m.get("conc_num_list") or m.get("conc_qps_list")):
        hard.append("Eşzamanlı arama sonucu yok — arama aşaması hiç çalışmamış.")

    if not m.get("load_duration"):
        hard.append("Yükleme süresi 0 — veri OpenSearch'e hiç yazılmamış.")

    if not m.get("recall") and not m.get("serial_latency_p99"):
        hard.append("Recall ve gecikme ölçümü yok — seri arama aşaması da çalışmamış.")

    # Bellek ölçümü eksikse bu tek başına koşuyu geçersiz kılmaz.
    if mem is None:
        warn.append("Bellek anlık görüntüsü yok — os_memstat.py çalıştırılmamış.")
    elif all(mem.get(k) is None for k in ("jvm_heap", "segments", "knn_graph", "store")):
        warn.append(
            "Bellek anlık görüntüleri boş — os_memstat.py OpenSearch'e bağlanamamış."
        )
        if hard:
            hard.append(
                "Bellek de çekilememiş; sorun büyük olasılıkla OpenSearch bağlantısı."
            )

    return hard, warn


def render_failure(problems: list[str], src: Path | None) -> str:
    W = 58
    L = ["=" * W, "  KOŞU BAŞARISIZ — rapor edilecek geçerli ölçüm yok", "=" * W, ""]
    if src:
        L.append(f"  Sonuç dosyası: {src}")
        L.append("")

    # Log'daki gerçek istisna, sıfır metriklerden çok daha bilgilendirici.
    reasons = read_failure_reasons()
    if reasons:
        L.append("  VectorDBBench log'undaki hata:")
        for r in reasons:
            L.append(f"    {r}")
        L.append("")
        for signature, fix in KNOWN_FIXES:
            if any(signature in r for r in reasons):
                L.append("  >>> BİLİNEN SORUN")
                for line in fix.splitlines():
                    L.append(f"      {line.strip() if line.startswith('      ') else line}")
                L.append("")
                break

    L.append("  Tespit edilenler:")
    for p in problems:
        L.append(f"    - {p}")
    L += [
        "",
        "-" * W,
        "  KONTROL LİSTESİ",
        "-" * W,
        "",
        "  1) OpenSearch ayakta ve erişilebilir mi?",
        "       curl -k -u admin:$env:OPENSEARCH_PASSWORD https://localhost:9200",
        "     Bağlantı reddedildiyse servis çalışmıyor demektir.",
        "",
        "  2) HTTP mu HTTPS mi konuşuyor?",
        "     Güvenlik eklentisi AÇIKSA https, KAPALIYSA http.",
        "     Yanlış şema 'connection reset' ya da sessiz hata verir.",
        "       https ise : .\\benchmark\\run_opensearch.ps1",
        "       http  ise : .\\benchmark\\run_opensearch.ps1 -NoSsl",
        "",
        "  3) Kimlik bilgileri doğru mu?",
        "     .env içindeki OPENSEARCH_USER / OPENSEARCH_PASSWORD.",
        "     401 alıyorsan şifre yanlış.",
        "",
        "  4) VectorDBBench yamaları uygulandı mı?",
        "       python benchmark\\patch_vdbb.py --check",
        "     Uygulanmadıysa metric_type_name hatası ile düşer.",
        "",
        "  5) Dataset yerine kuruldu mu?",
        "       python benchmark\\install_dataset.py",
        "",
        "  Bağlantıyı tek komutla test etmek için:",
        "       python benchmark\\os_memstat.py --out nul --label test",
        "",
        "  Benchmark'ın gerçek hata mesajı için koşuyu tekrar çalıştır ve",
        "  vectordbbench çıktısındaki traceback'e bak — rapor onu göremez.",
        "=" * W,
    ]
    return "\n".join(L)


def summarize(case: dict) -> dict:
    """Bir CaseResult'ı rapor alanlarına indirger."""
    m = case["metrics"]
    tc = case["task_config"]
    cc = tc.get("case_config", {})
    dbc = tc.get("db_case_config", {}) or {}

    conc_nums = m.get("conc_num_list") or []
    conc_qps = m.get("conc_qps_list") or []
    conc_avg = m.get("conc_latency_avg_list") or []
    conc_p95 = m.get("conc_latency_p95_list") or []
    conc_p99 = m.get("conc_latency_p99_list") or []

    def at(idx: int, lst: list) -> float | None:
        return lst[idx] if idx is not None and idx < len(lst) else None

    # Seri = concurrency 1
    serial_i = conc_nums.index(1) if 1 in conc_nums else None
    # Tepe = en yüksek QPS
    peak_i = conc_qps.index(max(conc_qps)) if conc_qps else None

    duration = (cc.get("concurrency_search_config") or {}).get("concurrency_duration")

    return {
        "db": tc.get("db"),
        "label": tc.get("db_label") or case.get("label"),
        "k": cc.get("k"),
        "engine": dbc.get("engine"),
        "M": dbc.get("M"),
        "ef_construction": dbc.get("efConstruction"),
        "ef_search": dbc.get("efSearch"),
        "metric": dbc.get("metric_type_name") or dbc.get("metric_type"),
        "recall": m.get("recall"),
        "ndcg": m.get("ndcg"),
        "insert_duration": m.get("insert_duration"),
        "optimize_duration": m.get("optimize_duration"),
        "load_duration": m.get("load_duration"),
        "serial_p95_ms": (m.get("serial_latency_p95") or 0) * 1000 or None,
        "serial_p99_ms": (m.get("serial_latency_p99") or 0) * 1000 or None,
        "duration_s": duration,
        "serial": None
        if serial_i is None
        else {
            "conc": 1,
            "qps": at(serial_i, conc_qps),
            "avg_ms": (at(serial_i, conc_avg) or 0) * 1000 or None,
            "p95_ms": (at(serial_i, conc_p95) or 0) * 1000 or None,
            "p99_ms": (at(serial_i, conc_p99) or 0) * 1000 or None,
        },
        "peak": None
        if peak_i is None
        else {
            "conc": conc_nums[peak_i] if peak_i < len(conc_nums) else None,
            "qps": at(peak_i, conc_qps),
            "avg_ms": (at(peak_i, conc_avg) or 0) * 1000 or None,
            "p95_ms": (at(peak_i, conc_p95) or 0) * 1000 or None,
            "p99_ms": (at(peak_i, conc_p99) or 0) * 1000 or None,
        },
    }


# --------------------------------------------------------------------------
def fmt(v: float | None, unit: str = "", digits: int = 2) -> str:
    return "-" if v is None else f"{v:,.{digits}f}{unit}"


def render(s: dict, meta: dict | None, mem: dict | None) -> str:
    L: list[str] = []
    add = L.append
    W = 58

    add("=" * W)
    add(f"  {str(s['db'] or 'DB').upper()} BENCHMARK SONUÇLARI")
    add("=" * W)
    if s.get("_src"):
        add(f"  Kaynak             : {Path(s['_src']).name}")

    if meta:
        add(f"  Dataset            : {meta['size']:,} vektör × {meta['dim']} boyut "
            f"({meta['metric_type']})")
        add(f"  Embedding          : {meta['embed_model']}")
        add(f"  Sorgu seti         : {meta['num_queries']} vektör")
    add(f"  Index              : HNSW/{s['engine']}  M={s['M']}  "
        f"ef_construction={s['ef_construction']}  ef_search={s['ef_search']}")
    add(f"  Arama              : k={s['k']}")
    add("")

    if s["serial"]:
        d = s["serial"]
        total = s["duration_s"]
        n_q = total * d["qps"] if (total and d["qps"]) else None
        add("-" * W)
        add("  SERİ  (tek istemci — Qdrant koşunla karşılaştırılabilir)")
        add("-" * W)
        add(f"  Toplam Süre        : {fmt(total)} saniye")
        add(f"  Sorgu Sayısı       : {fmt(n_q, digits=0)}")
        add(f"  Sorgu Hızı (QPS)   : {fmt(d['qps'])} sorgu/sn")
        add(f"  Ortalama Gecikme   : {fmt(d['avg_ms'])} ms")
        add(f"  P95 Gecikme        : {fmt(d['p95_ms'])} ms")
        add(f"  P99 Gecikme        : {fmt(d['p99_ms'])} ms")
        add("")

    if s["peak"] and s["peak"]["conc"] != 1:
        d = s["peak"]
        add("-" * W)
        add(f"  EŞZAMANLI TEPE  ({d['conc']} istemci)")
        add("-" * W)
        add(f"  Sorgu Hızı (QPS)   : {fmt(d['qps'])} sorgu/sn")
        add(f"  Ortalama Gecikme   : {fmt(d['avg_ms'])} ms")
        add(f"  P95 Gecikme        : {fmt(d['p95_ms'])} ms")
        add(f"  P99 Gecikme        : {fmt(d['p99_ms'])} ms")
        add("")

    add("-" * W)
    add("  DOĞRULUK")
    add("-" * W)
    add(f"  Recall@{s['k']:<12}: {fmt(s['recall'], digits=4)}")
    add(f"  NDCG@{s['k']:<14}: {fmt(s['ndcg'], digits=4)}")
    add(f"  Seri P95           : {fmt(s['serial_p95_ms'])} ms")
    add(f"  Seri P99           : {fmt(s['serial_p99_ms'])} ms")
    add("")

    add("-" * W)
    add("  YÜKLEME")
    add("-" * W)
    add(f"  Insert Süresi      : {fmt(s['insert_duration'])} saniye")
    add(f"  Optimize Süresi    : {fmt(s['optimize_duration'])} saniye")
    add(f"  Toplam             : {fmt(s['load_duration'])} saniye")
    add("")

    add("-" * W)
    add("  BELLEK  (OpenSearch sunucu tarafı)")
    add("-" * W)
    if mem is None:
        add("  Ölçüm yok — koşu öncesi/sonrası os_memstat.py çalıştırılmamış.")
    else:
        knn_note = "" if mem["knn_available"] else "   (kNN stats alınamadı)"
        add(f"  kNN Graph Belleği  : {fmt(mem['knn_graph'], ' MB')}{knn_note}")
        seg_note = "   (OpenSearch 2.x bu alanı kaldırdı)" if mem["segments"] == 0 else ""
        add(f"  Segment Belleği    : {fmt(mem['segments'], ' MB')}{seg_note}")
        add(f"  JVM Heap Farkı     : {fmt(mem['jvm_heap'], ' MB')}   (GC nedeniyle gürültülü)")
        add(f"  Disk (store) Artışı: {fmt(mem['store'], ' MB')}")
        add("")
        add("  Not: Vektör indeksinin gerçek bellek maliyeti kNN Graph satırıdır.")
        add("  Qdrant raporundaki 'Ek RAM' istemci RSS'i ise aynı şeyi ölçmüyor.")
    add("=" * W)
    return "\n".join(L)


def render_table(rows: list[dict]) -> str:
    hdr = (
        f"{'label':22} {'M':>3} {'efS':>5} {'recall':>7} "
        f"{'seri QPS':>9} {'seri p99':>9} {'tepe QPS':>9} {'tepe p99':>9}"
    )
    out = [hdr, "-" * len(hdr)]
    for s in rows:
        ser, pk = s.get("serial") or {}, s.get("peak") or {}
        out.append(
            f"{str(s['label'])[:22]:22} {str(s['M'] or '-'):>3} {str(s['ef_search'] or '-'):>5} "
            f"{fmt(s['recall'], digits=4):>7} "
            f"{fmt(ser.get('qps')):>9} {fmt(ser.get('p99_ms')):>9} "
            f"{fmt(pk.get('qps')):>9} {fmt(pk.get('p99_ms')):>9}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, default=None)
    p.add_argument("--all", action="store_true", help="Tüm koşuları tablo olarak")
    p.add_argument("--markdown", type=Path, default=None, help="Markdown dosyaya yaz")
    args = p.parse_args()

    meta_path = DATASET_DIR / "dataset_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    mem = load_memory_delta()

    files = find_results(args.result)
    if not args.all:
        files = files[:1]

    summaries: list[dict] = []
    first_case: dict | None = None
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for case in data.get("results", []):
            if first_case is None:
                first_case = case
            s = summarize(case)
            s["label"] = s["label"] or data.get("task_label")
            s["_src"] = f
            summaries.append(s)

    if not summaries:
        sys.exit("HATA: sonuç dosyalarında case bulunamadı.")

    # Başarısız koşuyu sıfırlarla dolu geçerli bir raporymuş gibi basma.
    warnings: list[str] = []
    if not args.all:
        hard, warnings = diagnose(first_case or {}, mem)
        if hard:
            text = render_failure(hard + warnings, files[0])
            print(text)
            if args.markdown:
                args.markdown.write_text(f"```\n{text}\n```\n", encoding="utf-8")
            sys.exit(1)

    if args.all and len(summaries) > 1:
        text = render_table(summaries)
    else:
        text = render(summaries[0], meta, mem)

    print(text)
    for w in warnings:
        print(f"UYARI: {w}")

    if args.markdown:
        args.markdown.write_text(f"```\n{text}\n```\n", encoding="utf-8")
        print(f"\n-> {args.markdown}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Demo presentasi Kelompok 1. Jalankan dari folder hybrid-ir/:
#   bash presentasi/demo.sh          # semua langkah
#   bash presentasi/demo.sh 2        # hanya langkah 2
# Keluaran setiap langkah juga disimpan ke presentasi/cadangan/langkah<N>.txt (rencana cadangan).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1            # model sudah di cache; jangan menghubungi internet saat presentasi
HIR=../.venv/bin/hybrid-ir
SF=configs/scifact_bart.yaml
NF=configs/nfcorpus_bart.yaml
OUT=presentasi/cadangan
mkdir -p "$OUT"

Q1185="The US health care system can save up to \$750 million if 7% of patients waiting for kidney transplants participate in the optimized national kidney paired donation program."
Q1024="Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes."

judul() { printf '\n==================== %s ====================\n' "$1"; }

langkah1() {  # Hybrid menolong (NFCorpus): semantic vs hybrid
  judul "Langkah 1: 'betel nuts' — semantic vs hybrid (NFCorpus)"
  $HIR search --config $NF "betel nuts" --mode semantic --top-k 3
  $HIR search --config $NF "betel nuts" --mode hybrid --top-k 3
}

langkah2() {  # Hybrid menolong (SciFact 1185): dokumen relevan naik dari #3 ke #1
  judul "Langkah 2: SciFact 1185 (kidney paired donation) — semantic vs hybrid"
  $HIR search --config $SF "$Q1185" --mode semantic --top-k 3
  $HIR search --config $SF "$Q1185" --mode hybrid --top-k 3
}

langkah3() {  # Hybrid kalah dari BM25 (SciFact 1024): istilah langka CTCF
  judul "Langkah 3: SciFact 1024 (CTCF anchor sites) — bm25 vs hybrid"
  $HIR search --config $SF "$Q1024" --mode bm25 --top-k 3
  $HIR search --config $SF "$Q1024" --mode hybrid --top-k 3
}

langkah4() {  # Hybrid kalah dari BM25 (NFCorpus 'kohlrabi')
  judul "Langkah 4: 'kohlrabi' — bm25 vs keyphrase vs hybrid (NFCorpus)"
  $HIR search --config $NF "kohlrabi" --mode bm25 --top-k 2
  $HIR search --config $NF "kohlrabi" --mode keyphrase --top-k 2 || true
  $HIR search --config $NF "kohlrabi" --mode hybrid --top-k 2
}

langkah5() {  # Kueri bebas: keyphrase kueri dibangkitkan BART-KPG lokal
  judul "Langkah 5: kueri bebas 'does coffee raise blood pressure' (BART lokal)"
  $HIR search --config $NF "does coffee raise blood pressure" --mode hybrid --top-k 3
}

jalankan() { "langkah$1" 2>&1 | grep -v -E "^INFO|Loading weights" | tee "$OUT/langkah$1.txt"; }

if [[ $# -ge 1 ]]; then
  jalankan "$1"
else
  for n in 1 2 3 4 5; do jalankan "$n"; done
fi

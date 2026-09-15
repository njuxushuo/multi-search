#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export JAVA_HOME="${JAVA_HOME:-/data0/hcy/maple/jre.X86_64_LINUX}"
export PATH="$JAVA_HOME/bin:$PATH"
INPUT_DIR="data/corpus_bm25_input"
INDEX_DIR="data/index/bm25"
mkdir -p "$INPUT_DIR" "$INDEX_DIR"
ln -sfn ../corpus/wiki-18.jsonl "$INPUT_DIR/wiki-18.jsonl"

if [[ -f "$INDEX_DIR/segments_1" || -f "$INDEX_DIR/segments_2" ]]; then
  echo "BM25 index already appears to exist at $INDEX_DIR"
  exit 0
fi

conda run --no-capture-output -n search-r1 \
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$INPUT_DIR" \
    --index "$INDEX_DIR" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${BM25_THREADS:-32}" \
    --storePositions --storeDocvectors \
  2>&1 | tee outputs/bm25_index_build.log

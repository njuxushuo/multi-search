#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export JAVA_HOME="${JAVA_HOME:-/data0/hcy/maple/jre.X86_64_LINUX}"
export PATH="$JAVA_HOME/bin:$PATH"
if [[ ! -f data/index/bm25/segments_1 ]]; then
  echo "BM25 index is not complete" >&2
  exit 2
fi
conda run --no-capture-output -n search-r1 \
  python third_party/Search-R1/search_r1/search/retrieval_server.py \
    --index_path data/index/bm25 \
    --corpus_path data/corpus/wiki-18.jsonl \
    --topk 3 \
    --retriever_name bm25

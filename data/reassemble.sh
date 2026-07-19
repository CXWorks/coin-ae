#!/bin/bash
# Reassemble the split sample test data before use.
# Run once from the repo root: bash data/reassemble.sh

set -e

echo "Reassembling coin_test.pkl.sample..."
cat data/coin_test.pkl.sample.part_* > data/coin_test.pkl.sample
echo "  -> data/coin_test.pkl.sample ($(du -sh data/coin_test.pkl.sample | cut -f1))"

# Sanity check: 4-tuple (file, text, spans, windows) entries under
# 'safe'/'unsafe' keys, same format as the full coin_test.pkl.
python3 - <<'EOF'
import pickle
with open("data/coin_test.pkl.sample", "rb") as f:
    d = pickle.load(f)
n_fns = sum(len(e[2]) for k in ("safe", "unsafe") for e in d[k])
assert all(len(e) == 4 for k in ("safe", "unsafe") for e in d[k])
print(f"  OK: {len(d['safe'])} safe + {len(d['unsafe'])} unsafe file entries "
      f"({n_fns} functions), 4-tuple format")
EOF

echo "Done. Use it e.g. with:"
echo "  bash scripts/1_smoke_test.sh data/coin_test.pkl.sample"

#!/bin/bash
# Reassemble the split/compressed datasets before use.
# Run once from the repo root: bash data/reassemble.sh
#
# Produces:
#   data/coin_test.pkl.sample  - 2,000-entry sample split (smoke testing)
#   data/coin_train.pkl        - full training split   (~700 MB)
#   data/coin_valid.pkl        - full validation split (~230 MB)
#   data/coin_test.pkl         - full test split       (~230 MB, E1/E2)

set -e

echo "Reassembling coin_test.pkl.sample..."
cat data/coin_test.pkl.sample.part_* > data/coin_test.pkl.sample
echo "  -> data/coin_test.pkl.sample ($(du -sh data/coin_test.pkl.sample | cut -f1))"

echo "Decompressing full datasets..."
cat data/coin_train.pkl.gz.part_* | gunzip > data/coin_train.pkl
echo "  -> data/coin_train.pkl ($(du -sh data/coin_train.pkl | cut -f1))"
gunzip -k -f data/coin_valid.pkl.gz
echo "  -> data/coin_valid.pkl ($(du -sh data/coin_valid.pkl | cut -f1))"
gunzip -k -f data/coin_test.pkl.gz
echo "  -> data/coin_test.pkl ($(du -sh data/coin_test.pkl | cut -f1))"

# Sanity check: 4-tuple (file, text, spans, windows) entries under
# 'safe'/'unsafe' keys.
python3 - <<'EOF'
import pickle
for name in ("coin_test.pkl.sample", "coin_test.pkl"):
    with open(f"data/{name}", "rb") as f:
        d = pickle.load(f)
    n_fns = sum(len(e[2]) for k in ("safe", "unsafe") for e in d[k])
    assert all(len(e) == 4 for k in ("safe", "unsafe") for e in d[k])
    print(f"  OK {name}: {len(d['safe'])} safe + {len(d['unsafe'])} unsafe "
          f"file entries ({n_fns} functions), 4-tuple format")
EOF

echo "Done. Use e.g.:"
echo "  bash scripts/1_smoke_test.sh data/coin_test.pkl.sample   # quick check"
echo "  bash scripts/1_smoke_test.sh data/coin_test.pkl          # E1"
echo "  bash scripts/2_full_eval.sh  data/coin_test.pkl          # E2"

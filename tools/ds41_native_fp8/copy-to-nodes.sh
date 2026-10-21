#!/bin/bash
# Prepares the native FP8 dense weights for a GPU validation on the nodes. Run ON THE PC (Git Bash) once the original
# checkpoint download is complete; it is never run automatically.
#   tools/ds41_native_fp8/copy-to-nodes.sh [SRC_DIR] [EXTRACT_DIR]
# 1. extract.py copies only the dense FP8 sites (attention wq_a/wq_b/wkv/wo_a/wo_b + shared expert w1/w3/w2, codes and
#    32 x 32 exponents) of the 40 layers: 40 files of 162,163,200 bytes + index, 6.49 GB (not the 280+ GB checkpoint);
# 2. verifies the extract byte for byte against the source (--check);
# 3. on each node: requires free space >= 2 x the extract, then copies it to ~/ai-cluster-lab/models/ds41-native-fp8
#    (scp to the first node, rsync from there at low priority; every file is checked by sha256 after the copy).
# Nothing is executed from the model directories; Python runs isolated (-I) from this tools directory.
set -euo pipefail
SRC=${1:-/d/models/ds41-orig}
OUT=${2:-/d/models/ds41-native-fp8}
PY=${PY:-/d/conda3/python}
NODES=${NODES:-"cluster1 cluster2"}
DEST=ai-cluster-lab/models/ds41-native-fp8
here=$(cd "$(dirname "$0")" && pwd)

"$PY" -I "$here/extract.py" "$SRC" "$OUT" --layers 0-39
"$PY" -I "$here/extract.py" "$SRC" "$OUT" --layers 0-39 --check
bytes=$(du -sb "$OUT" | cut -f1)
need=$((2 * bytes / 1024))
( cd "$OUT" && sha256sum dense-fp8-*.safetensors model.safetensors.index.json > SHA256SUMS )
first=${NODES%% *}
for h in $NODES; do
  free=$(ssh -o BatchMode=yes sergioc@$h "df -k --output=avail ~ | tail -1")
  echo "$h: free $((free / 1024 / 1024)) GiB, extract $((bytes / 1024 / 1024)) MiB"
  if [ "$free" -lt "$need" ]; then echo "ABORT: not enough free disk on $h"; exit 3; fi
done
# PC -> first node (scp), then first node -> the others over the node link (rsync at low priority).
ssh -o BatchMode=yes sergioc@$first "mkdir -p ~/$DEST"
scp -q "$OUT"/* sergioc@$first:$DEST/
for h in $NODES; do
  [ "$h" = "$first" ] || ssh -o BatchMode=yes sergioc@$first "nice -n 19 ionice -c3 rsync -a --partial ~/$DEST/ $h:$DEST/"
  ssh -o BatchMode=yes sergioc@$h "cd ~/$DEST && sha256sum -c --quiet SHA256SUMS && echo $h: extract verified"
done

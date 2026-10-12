#!/bin/bash
# One leg of a vllm-bench-serve comparison: install a spyre-inference ref (and the torch-spyre
# rev it pins) into the harness .venv, run every config of BENCH/serve-tests.yaml with
# `make perf-tests`, check the leg with collect.py validate, and restore the install. A
# committed ref runs from a throwaway worktree; the working copy is never touched.
#
# Usage: run_leg.sh --bench DIR --leg NAME --ref REF [--label TEXT] [--role ROLE] [--env K=V]...
#   --bench DIR   bench directory holding serve-tests.yaml; the leg writes to DIR/NAME/
#   --leg NAME    leg directory name, e.g. pr1043 or main
#   --ref REF     git ref or sha; "." for the working copy, uncommitted changes included
#   --label TEXT  leg name in the report (default: NAME)
#   --role ROLE   target (default) or baseline
#   --env K=V     env var for this leg, repeatable; SPYRE_KERNEL_CACHE=1 unless given. With
#                 SPYRE_KERNEL_CACHE=0, TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 and
#                 VLLM_DISABLE_COMPILE_CACHE=1 are also set unless given, so nothing is cached
# Writes DIR/NAME/{leg.json,env.txt,sync.log,run.log,results/}.
# Exit: 0 valid leg; 1 the run or the validity check failed; 2 bad usage or setup failure.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
BENCH="" LEG="" REF="" LABEL="" ROLE=target
ENVS=()
die() { echo "run_leg.sh: $1" >&2; exit "${2:-2}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH=$2; shift 2 ;;
    --leg) LEG=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --role) ROLE=$2; shift 2 ;;
    --env) [[ $2 == *=* ]] || die "--env needs KEY=VALUE, got $2"
           ENVS+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done
[ -n "$BENCH" ] && [ -n "$LEG" ] && [ -n "$REF" ] || die "--bench, --leg and --ref are required"
[[ $ROLE == target || $ROLE == baseline ]] || die "--role must be target or baseline"
BENCH=$(realpath "$BENCH")
[ -f "$BENCH/serve-tests.yaml" ] || die "no $BENCH/serve-tests.yaml"
OUT=$BENCH/$LEG
[ -e "$OUT" ] && die "$OUT exists; remove it or pick another --leg"
mkdir -p "$OUT"

set +u
# Exists only if scripts/install-pinned-rpms.sh was run on this pod.
[ -f "$HOME/spyre-libs/env.sh" ] && source "$HOME/spyre-libs/env.sh"
source "$REPO/.venv/bin/activate"
set -u
# The server runs `python -m` from the harness checkout, which would put that checkout's
# spyre_inference/ ahead of the leg's installed tree on sys.path.
export PYTHONSAFEPATH=1

WT=""
if [ "$REF" = "." ]; then
  SRC=$REPO
  SHA=$(git -C "$REPO" rev-parse HEAD)
  DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
else
  SHA=$(git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}") || die "unknown ref $REF"
  SRC=$(mktemp -d -t "bench-$LEG-XXXXXX")
  git -C "$REPO" worktree add --detach "$SRC" "$SHA" >"$OUT/worktree.log" 2>&1 \
    || die "git worktree add failed, see $OUT/worktree.log"
  WT=$SRC
  DIRTY=0
fi

restore() {
  if ! (cd "$REPO" && uv sync --frozen) >"$OUT/restore.log" 2>&1; then
    echo "run_leg.sh: restoring the install failed, see $OUT/restore.log" >&2
  fi
  [ -n "$WT" ] && git -C "$REPO" worktree remove --force "$WT"
}
trap restore EXIT
trap 'exit 130' INT TERM

# Records the leg; ERROR, when set, marks it failed before any config ran.
write_meta() {
  python3 - "$OUT/leg.json" "$@" <<'EOF'
import json, os, platform, sys, time
path, *envs = sys.argv[1:]
e = os.environ
json.dump({
    "name": e["LEG"], "label": e["LABEL"], "role": e["ROLE"], "ref": e["REF"],
    "sha": e["SHA"], "dirty_files": int(e["DIRTY"]), "src": e["SRC"],
    "torch_spyre_pinned": e.get("TS_PIN", ""), "torch_spyre": e.get("TS_REV", ""),
    "torch": e.get("LEG_TORCH", ""), "vllm": e.get("LEG_VLLM", ""),
    "rpms_source": e.get("RPMS_SOURCE", ""), "rpms": json.loads(e.get("RPMS", "{}") or "{}"),
    "env": dict(kv.split("=", 1) for kv in envs),
    "host": platform.node(), "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "error": e.get("ERROR", ""),
}, open(path, "w"), indent=2)
EOF
}
fail() { export ERROR=$1; write_meta "${ENVS[@]}"; die "$1"; }
export LEG LABEL=${LABEL:-$LEG} ROLE REF SHA DIRTY SRC

echo "run_leg.sh: $LEG = $SHA from $SRC"
UV_PROJECT_ENVIRONMENT="$REPO/.venv" uv sync --frozen --project "$SRC" >"$OUT/sync.log" 2>&1 \
  || fail "uv sync failed, see $OUT/sync.log"

cd "$REPO" || fail "cannot cd to $REPO"
# Importing the plugin logs to stdout, so keep only the last line.
PKG=$(python -c "import os, spyre_inference; print(os.path.realpath(spyre_inference.__file__))" | tail -n1) \
  || fail "cannot import spyre_inference"
case "$PKG" in
  "$(realpath "$SRC")"/*) ;;
  *) fail "spyre_inference imports from $PKG, not from $SRC" ;;
esac
export TS_PIN TS_REV LEG_TORCH LEG_VLLM RPMS_SOURCE RPMS
TS_PIN=$(python -c "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['tool']['uv']['sources']['torch-spyre']['rev'])" "$SRC/pyproject.toml")
TS_REV=$(python -c "import importlib.metadata as m, json; print(json.loads(m.distribution('torch-spyre').read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id', 'unknown'))")
[ "$TS_PIN" = "$TS_REV" ] || fail "torch-spyre $TS_REV is installed, but $REF pins $TS_PIN"
LEG_TORCH=$(python -c "import torch; print(torch.__version__)" | tail -n1)
LEG_VLLM=$(python -c "import importlib.metadata as m; print(m.version('vllm'))")
if [ -d "$HOME/spyre-libs" ]; then
  RPMS_SOURCE="[packages] of $REF:spyre-rpms.lock, installed in ~/spyre-libs"
  RPMS=$(python -c "import json, sys, tomllib; print(json.dumps({k: v['version'] for k, v in tomllib.load(open(sys.argv[1], 'rb'))['packages'].items()}))" "$SRC/spyre-rpms.lock")
else
  RPMS_SOURCE=/opt/ibm/spyre/components.txt
  RPMS=$(python -c "import json, sys; print(json.dumps(dict(l.strip().split(':', 1) for l in open(sys.argv[1]) if ':' in l)))" "$RPMS_SOURCE")
fi

# Adds K=V to ENVS unless an --env already gives K.
default_env() {
  local kv
  for kv in "${ENVS[@]}"; do [[ $kv == "${1%%=*}"=* ]] && return; done
  ENVS+=("$1")
}
default_env SPYRE_KERNEL_CACHE=1
if printf '%s\n' "${ENVS[@]}" | grep -qx 'SPYRE_KERNEL_CACHE=0'; then
  default_env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
  default_env VLLM_DISABLE_COMPILE_CACHE=1
fi
for kv in "${ENVS[@]}"; do export "${kv?}"; done
env | grep -E '^(SPYRE|VLLM|TORCH|TORCHINDUCTOR|AIU|FLEX|DT|COLL|SENTIENT|OMP)_' | sort >"$OUT/env.txt"
write_meta "${ENVS[@]}"

echo "run_leg.sh: running $(grep -c 'test_name:' "$BENCH/serve-tests.yaml") configs, log in $OUT/run.log"
make perf-tests BENCH_TYPES=serve BENCH_CONFIGS_DIR="$BENCH" RESULTS_DIR="$OUT/results" \
  >"$OUT/run.log" 2>&1
MAKE_RC=$?
[ $MAKE_RC = 0 ] || echo "run_leg.sh: make perf-tests exited $MAKE_RC" >&2

# `vllm bench serve` exits 0 even when most requests fail, so the leg is judged by its results.
python3 "$HERE/collect.py" validate "$BENCH" "$LEG" || exit 1
[ $MAKE_RC = 0 ] || exit 1

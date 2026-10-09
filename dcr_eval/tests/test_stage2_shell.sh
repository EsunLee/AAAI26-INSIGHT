#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d /tmp/dcr-stage2-shell.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/model" "$TMP/adapter" "$TMP/out"
for i in 0 1 2 3 4; do
    printf '{"messages":[{"role":"user","content":"sample %s"}]}\n' "$i" >> "$TMP/input.jsonl"
done

cat > "$TMP/fake-swift" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
dataset=""
result=""
while (( $# > 0 )); do
    case "$1" in
        --val_dataset) dataset="$2"; shift 2 ;;
        --result_path) result="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[[ -n "$dataset" && -n "$result" ]]
while IFS= read -r line; do
    [[ -n "$line" ]] && printf '{"response":"<answer>take cup</answer>"}\n'
done < "$dataset" > "$result"
EOF
chmod +x "$TMP/fake-swift"

common=(
    "SWIFT_BIN=$TMP/fake-swift"
    "PYTHON_BIN=$(command -v python3)"
    "MODEL=$TMP/model"
    "ADAPTER=$TMP/adapter"
    "DATASET=$TMP/input.jsonl"
    "OUTPUT_DIR=$TMP/out"
    "GPU=0"
    "SPLIT_TOTAL=2"
    "SPLIT_INDEX=0"
)

env "${common[@]}" bash "$ROOT/scripts/03_stage2_generate.sh"
[[ "$(wc -l < "$TMP/out/candidate_0_part_0.jsonl")" -eq 2 ]]

# 完整分片应直接复用。
env "${common[@]}" bash "$ROOT/scripts/03_stage2_generate.sh"

# 残缺分片应保留备份并重新生成。
: > "$TMP/out/candidate_0_part_0.jsonl"
env "${common[@]}" bash "$ROOT/scripts/03_stage2_generate.sh"
[[ "$(wc -l < "$TMP/out/candidate_0_part_0.jsonl")" -eq 2 ]]
compgen -G "$TMP/out/candidate_0_part_0.jsonl.incomplete.*" > /dev/null

# 行数相同但 JSON 损坏也不得复用。
printf '{broken}\n{broken}\n' > "$TMP/out/candidate_0_part_0.jsonl"
env "${common[@]}" bash "$ROOT/scripts/03_stage2_generate.sh"
"$(command -v python3)" "$ROOT/scripts/validate_generated_jsonl.py" \
    "$TMP/out/candidate_0_part_0.jsonl" 2

echo "test_stage2_shell: PASS"

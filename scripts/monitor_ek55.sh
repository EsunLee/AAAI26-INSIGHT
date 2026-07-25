#!/bin/bash
# EK55 下载监控
# Ctrl+C 退出，不影响后台下载

LOG=/tmp/ek55_download.log
DIR=/data/datasets/EPIC-KITCHENS
CSV=/home/amax/ldy/git/AAAI26-INSIGHT/scripts/epic-kitchens-download-scripts/data/md5.csv
START=$(date +%s)
TOTAL_GB=220

# 总 tar 数（从 md5.csv 只读一次）
TOTAL_TARS=$(grep -c "frames_rgb_flow/rgb" "$CSV" 2>/dev/null || echo "?")

prev=0
while true; do
  # 采集
  bytes=$(du -sb "$DIR" 2>/dev/null | awk '{print $1}')
  human=$(du -sh "$DIR" 2>/dev/null | awk '{print $1}')
  tars=$(find "$DIR" -name "*.tar" 2>/dev/null | wc -l)
  skip=$(grep -c "already downloaded" "$LOG" 2>/dev/null)
  downloaded=$((skip))
  remaining=$((TOTAL_TARS - downloaded))
  disk=$(df -h /data | awk 'NR==2{print $4}')
  errs=$(grep -c "Could not download" "$LOG" 2>/dev/null)

  # 速度
  speed="-"; kb=0
  [ "$prev" -gt 0 ] && [ "$bytes" -gt "$prev" ] && {
    kb=$(((bytes - prev) / 10240))
    [ "$kb" -ge 1024 ] && speed="$(awk "BEGIN{printf \"%.1f\",$kb/1024}") MB/s" || speed="$kb KB/s"
  }
  prev=$bytes

  # 百分比
  gb=$(awk "BEGIN{printf \"%.1f\",$bytes/1073741824}")
  pct=$(awk "BEGIN{printf \"%.1f\",$gb/$TOTAL_GB*100}")

  # 用时
  elapsed=$(($(date +%s) - START))
  [ $elapsed -gt 3600 ] && et="$((elapsed/3600))h$(((elapsed%3600)/60))m" || et="$((elapsed/60))m"

  # ETA
  eta="-"
  [ "$kb" -gt 0 ] && [ "$bytes" -gt 1073741824 ] && {
    rem=$(((TOTAL_GB*1073741824 - bytes) / (kb * 1024)))
    [ $rem -gt 86400 ] && eta="$((rem/86400))d$(((rem%86400)/3600))h" || \
    [ $rem -gt 3600 ] && eta="$((rem/3600))h$(((rem%3600)/60))m" || \
    eta="$((rem/60))m"
  }

  # === 渲染 ===
  printf "\033[2J\033[H"
  echo "=== EK55 下载监控  $(date '+%H:%M:%S') ==="
  echo ""

  # 横排参数
  printf "  大小 %s/%sGB(%s%%) | tar %s/%s(%s剩余) | 速度 %s | 用时 %s | 剩余 %s | 磁盘 %s\n" \
    "$human" "$TOTAL_GB" "$pct" "$downloaded" "$TOTAL_TARS" "$remaining" "$speed" "$et" "$eta" "$disk"
  [ "$errs" -gt 0 ] && printf "  ⚠ 错误: %s\n" "$errs"
  echo ""

  # 总体进度条
  bar_width=50
  filled=$(awk "BEGIN{printf \"%d\",$pct*$bar_width/100}")
  empty=$((bar_width - filled))
  printf "  总体 ["
  [ $filled -gt 0 ] && printf "%${filled}s" | tr ' ' '#'
  [ $empty -gt 0 ] && printf "%${empty}s" | tr ' ' '-'
  printf "] %s%%\n" "$pct"
  echo ""

  # 当前文件（从日志取）
  echo "--- 当前文件 ---"
  tail -1 "$LOG"

  echo ""
  echo "Ctrl+C 退出"
  sleep 10
done

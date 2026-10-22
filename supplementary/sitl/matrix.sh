#!/bin/bash
# 每组条件前重启 SITL（电池/位置复位），逐条执行并汇总
source /opt/ros/humble/setup.bash
cd "$HOME/paper_sitl"
OUT="$HOME/paper_sitl/runs3"
mkdir -p "$OUT"
run_one () {
  local path="$1"; local hover="$2"
  bash "$HOME/qgc_px4/bin/stop_px4_qgc.sh" >/dev/null 2>&1
  sleep 2
  bash "$HOME/qgc_px4/bin/start_px4_qgc.sh" >/dev/null 2>&1
  sleep 9
  ros2 daemon stop >/dev/null 2>&1; ros2 daemon start >/dev/null 2>&1
  sleep 2
  echo "########## $path hover=$hover ##########"
  timeout 170 python3 sitl_turns.py --path "$path" --hover "$hover" --alt 5 --tag m1 --outdir "$OUT" 2>&1 | tail -8
}
run_one S20 0
run_one Q20 0
run_one S40 0
run_one S20 10
run_one Q20 10
echo "=== ALL DONE ==="
ls -la "$OUT"
for f in "$OUT"/m1_*.json; do echo -n "$(basename $f): "; cat "$f" | tr -d '\n'; echo; done
exit 0

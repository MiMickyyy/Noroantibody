#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE=""
INTERVAL=30
TAIL_LINES=40
ONCE=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/monitor_phase.sh [options]

Options:
  --phase <name>      Phase name, e.g. phase0_smoke / phase1_coarse_pilot
  --interval <sec>    Refresh interval for loop mode (default: 30)
  --tail <n>          Tail lines per log file (default: 40)
  --once              Print one snapshot and exit
  -h, --help          Show this help
USAGE
}

auto_detect_phase() {
  local latest=""
  latest="$(find "${ROOT_DIR}" -maxdepth 2 -name phase_status.json -type f -print 2>/dev/null | xargs ls -1t 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    basename "$(dirname "${latest}")"
    return 0
  fi
  echo "phase0_smoke"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

print_title() {
  local title="$1"
  echo
  echo "========== ${title} =========="
}

show_gpu() {
  print_title "GPU"
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits || true
    echo
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  else
    echo "nvidia-smi not found."
  fi
}

show_processes() {
  print_title "Pipeline Processes"
  local pattern='run_pipeline.py|rfdiffusion|proteinmpnn|rf2'
  if have_cmd rg; then
    ps -eo pid,etime,pcpu,pmem,args | rg -i "${pattern}" | rg -v "monitor_phase.sh|rg -i" || true
  else
    ps -eo pid,etime,pcpu,pmem,args | grep -Ei "${pattern}" | grep -Ev "monitor_phase.sh|grep -Ei" || true
  fi
}

show_phase_status() {
  local phase_dir="$1"
  local status_json="${phase_dir}/phase_status.json"
  local manifest_csv="${phase_dir}/phase_manifest.csv"

  print_title "Phase Status"
  echo "phase_dir: ${phase_dir}"

  if [[ -f "${status_json}" ]]; then
    python3 - "${status_json}" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
completed = data.get("completed_combinations", [])
print(f"status_file: {path}")
print(f"completed_combinations: {len(completed)}")
for k in ["last_updated", "phase", "note"]:
    if k in data:
        print(f"{k}: {data[k]}")
PY
  else
    echo "status_file not found: ${status_json}"
  fi

  if [[ -f "${manifest_csv}" ]]; then
    python3 - "${manifest_csv}" <<'PY'
import csv, sys
from collections import Counter
path = sys.argv[1]
rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
print(f"manifest_file: {path}")
print(f"manifest_rows: {len(rows)}")
ctr = Counter(r.get("status","") for r in rows)
for k, v in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {k or '<empty>'}: {v}")
PY
  else
    echo "manifest_file not found: ${manifest_csv}"
  fi
}

show_recent_logs() {
  local log_dir="$1"
  print_title "Recent Logs"
  if [[ ! -d "${log_dir}" ]]; then
    echo "log_dir not found: ${log_dir}"
    return
  fi

  local files
  files="$(ls -1t "${log_dir}"/*.log 2>/dev/null | head -n 3 || true)"
  if [[ -z "${files}" ]]; then
    echo "No .log files under ${log_dir}"
    return
  fi

  while IFS= read -r f; do
    [[ -z "${f}" ]] && continue
    echo "-- ${f} (tail -n ${TAIL_LINES})"
    tail -n "${TAIL_LINES}" "${f}" || true
    echo
  done <<< "${files}"
}

show_error_scan() {
  local log_dir="$1"
  print_title "Error Scan"
  if [[ ! -d "${log_dir}" ]]; then
    echo "log_dir not found: ${log_dir}"
    return
  fi
  local pattern='Traceback|Exception|Error executing job|Non-positive determinant|Exit code: [1-9]'
  if have_cmd rg; then
    rg -n -S "${pattern}" "${log_dir}" | tail -n 30 || true
  else
    grep -RInE "${pattern}" "${log_dir}" | tail -n 30 || true
  fi
}

show_system() {
  print_title "System"
  echo "time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "root: ${ROOT_DIR}"
  echo "phase: ${PHASE}"
  echo "load: $(uptime)"
  df -h / | sed -n '1,2p'
}

snapshot() {
  local phase_dir="${ROOT_DIR}/${PHASE}"
  local log_dir="${ROOT_DIR}/logs/${PHASE}"
  clear || true
  show_system
  show_gpu
  show_processes
  show_phase_status "${phase_dir}"
  show_recent_logs "${log_dir}"
  show_error_scan "${log_dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:-30}"
      shift 2
      ;;
    --tail)
      TAIL_LINES="${2:-40}"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${PHASE}" ]]; then
  PHASE="$(auto_detect_phase)"
fi

if [[ "${ONCE}" -eq 1 ]]; then
  snapshot
  exit 0
fi

while true; do
  snapshot
  sleep "${INTERVAL}"
done

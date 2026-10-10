#!/usr/bin/env bash
# Usage: scripts/verify_deploy.sh [base-url]   (default: the Render service)
# Exercises the live API end to end. Free Render instances sleep: the first call can take ~60 s.
set -u
B="${1:-https://chesttriage-api.onrender.com}"
fail=0
check() { # name, expected-substring, actual
  if printf '%s' "$3" | grep -q -- "$2"; then echo "ok    $1"; else echo "FAIL  $1 (wanted '$2', got: ${3:0:160})"; fail=1; fi
}
check "health"            '"status":"ok"'           "$(curl -s -m 90 "$B/health")"
check "status honest"     '"runs_a_trained_model":false' "$(curl -s -m 30 "$B/status")"
check "worklist"          '"items"'                 "$(curl -s -m 30 "$B/studies")"
check "queue settings"    '"max_wait_minutes"'      "$(curl -s -m 30 "$B/queue/settings")"
check "threshold"         '"min_sensitivity_target"' "$(curl -s -m 30 "$B/threshold")"
check "preview"           '"sensitivity"'           "$(curl -s -m 30 -X POST -H 'Content-Type: application/json' -d '{"value":0.5}' "$B/threshold/preview")"
check "threshold needs confirm" '"detail"'          "$(curl -s -m 30 -X PUT -H 'Content-Type: application/json' -d '{"value":0.5}' "$B/threshold")"
check "queue race"        '"voxelsort"'             "$(curl -s -m 60 -X POST -H 'Content-Type: application/json' -d '{"arrivals_per_hour":20,"reads_per_hour":18,"urgent_share":0.15,"hours":8,"seed":1}' "$B/simulations/queue-race")"
check "sample size"       '"positives_needed"'      "$(curl -s -m 30 "$B/site-readiness/sample-size?target_sensitivity=0.9&expected_sensitivity=0.95")"
check "evidence"          '"provenance"'            "$(curl -s -m 30 "$B/evidence")"
check "audit"             '^\['                     "$(curl -s -m 30 "$B/audit")"
check "cors vercel"       'access-control-allow-origin: https://x.vercel.app' "$(curl -si -m 30 -H 'Origin: https://x.vercel.app' "$B/health" | tr 'A-Z' 'a-z' | sed 's/\r//')"
# upload a generated PNG and read it back
tmp=$(mktemp -d); python3 -c "
from PIL import Image; import numpy as np
x=np.linspace(-1,1,320)[None,:]; y=np.linspace(-1,1,400)[:,None]
Image.fromarray(np.clip(120+60*np.exp(-((x*3)**2))*(1-0.3*y),0,255).astype('uint8')).save('$tmp/s.png')" 2>/dev/null && {
  r=$(curl -s -m 60 -w ' HTTP%{http_code}' -F "file=@$tmp/s.png" -F clinician_urgent=false -F actor=verify "$B/studies")
  check "upload 202"     'HTTP202'  "$r"
  id=$(printf '%s' "$r" | sed -n 's/.*"study_id":"\([^"]*\)".*/\1/p'); sleep 3
  check "upload unsure, not normal" '"effective_priority":"unsure"' "$(curl -s -m 30 "$B/studies/$id")"
  check "image served"   'HTTP200'  "$(curl -s -m 30 -o /dev/null -w 'HTTP%{http_code}' "$B/studies/$id/image")"
}
rm -rf "$tmp"
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }

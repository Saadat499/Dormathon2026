#!/usr/bin/env bash
# Usage: scripts/verify_deploy.sh [base-url] [frontend-origin]
#   base-url          default https://chesttriage-api.onrender.com
#   frontend-origin   optional: the deployed frontend's origin, to check CORS for it
# Exercises the live API end to end: health, worklist, upload, image, override, audit, review,
# threshold guard, Queue Race, Site Readiness, OOD check, evidence.
# Free Render instances sleep after 15 idle minutes: the first call can take about a minute.
# It writes to the live queue (one upload, one override, one review) and names itself "verify_deploy".
set -u
B="${1:-https://chesttriage-api.onrender.com}"
ORIGIN="${2:-}"
fail=0
check() { # name, expected-substring, actual
  if printf '%s' "$3" | grep -q -- "$2"; then echo "ok    $1"; else echo "FAIL  $1 (wanted '$2', got: ${3:0:200})"; fail=1; fi
}
J='Content-Type: application/json'
H=$(curl -s -m 120 "$B/health")
check "health ok"              '"status":"ok"'                 "$H"
check "db reachable, WAL"      '"journal_mode":"wal"'          "$H"
echo "      model_mode: $(printf '%s' "$H" | sed -n 's/.*"model_mode":"\([^"]*\)".*/\1/p')   degraded: $(printf '%s' "$H" | sed -n 's/.*"degraded":\([a-z]*\),"version".*/\1/p')"
check "saved predictions real" '"real":true'                   "$H"
check "worklist"               '"items":\['                    "$(curl -s -m 30 "$B/studies")"
check "queue settings"         '"max_wait_minutes"'            "$(curl -s -m 30 "$B/queue/settings")"
check "threshold"              '"min_sensitivity_target"'      "$(curl -s -m 30 "$B/threshold")"
check "preview"                '"held_out_test"'               "$(curl -s -m 30 -X POST -H "$J" -d '{"value":0.4}' "$B/threshold/preview")"
check "threshold needs confirm" 'confirm must be true'         "$(curl -s -m 30 -X PUT -H "$J" -d '{"value":0.3}' "$B/threshold")"
check "threshold floor"        'below the minimum target'      "$(curl -s -m 30 -X PUT -H "$J" -d '{"value":0.9,"confirm":true}' "$B/threshold")"
check "queue race"             '"voxelsort":{"urgent_wait_min"' "$(curl -s -m 90 -X POST -H "$J" -d '{"arrivals_per_hour":20,"reads_per_hour":18,"urgent_share":0.15,"hours":8,"seed":1}' "$B/simulations/queue-race")"
check "sample size"            '"positives_needed":127'        "$(curl -s -m 30 "$B/site-readiness/sample-size?target_sensitivity=0.9&expected_sensitivity=0.95")"
E=$(curl -s -m 30 "$B/evidence")
check "evidence is real"       '"data":"real"'                 "$E"
check "evidence keys"          '"cross_dataset":\['            "$E"
check "audit"                  '^\['                           "$(curl -s -m 30 "$B/audit")"
check "audit csv"              '^id,timestamp,actor,action'    "$(curl -s -m 30 "$B/audit.csv")"
if [ -n "$ORIGIN" ]; then
  check "cors for frontend"    "access-control-allow-origin: $ORIGIN" "$(curl -si -m 30 -H "Origin: $ORIGIN" "$B/health" | tr 'A-Z' 'a-z' | sed 's/\r//')"
fi
check "cors refuses others"    'no-acao'  "$(curl -si -m 30 -H 'Origin: https://not-allowed.example' "$B/health" | tr 'A-Z' 'a-z' | grep -q 'access-control-allow-origin' && echo has-acao || echo no-acao)"

# Upload a seeded demo image (a public NIH image served by the API itself), then override, audit, review.
tmp=$(mktemp -d)
first=$(curl -s -m 30 "$B/studies" | sed -n 's/.*"study_id":"\(nih_[0-9_]*\)".*/\1/p' | head -1)
[ -z "$first" ] && first="nih_00015355_007"
cd "$tmp" && curl -s -m 60 -o scan.png "$B/studies/$first/image"   # relative paths: Git Bash on Windows does not convert paths inside -F
r=$(curl -s -m 120 -w ' HTTP%{http_code}' -F "file=@scan.png;type=image/png" -F clinician_urgent=false -F actor=verify_deploy "$B/studies")
check "upload 202 processing"  '"status":"processing".* HTTP202' "$r"
id=$(printf '%s' "$r" | sed -n 's/.*"study_id":"\([^"]*\)".*/\1/p')
s=""
for _ in $(seq 1 30); do s=$(curl -s -m 30 "$B/studies/$id"); printf '%s' "$s" | grep -q '"status":"processing"' || break; sleep 2; done
check "processing finished"    '"status":"\(awaiting_review\|needs_review\)"' "$s"
echo "      result: $(printf '%s' "$s" | sed -n 's/.*"priority":"\([^"]*\)".*"model_source":"\([^"]*\)".*/priority \1, model_source \2/p')"
check "never normal w/o model" 'ok' "$(printf '%s' "$s" | grep -q '"model_source":"none"' && { printf '%s' "$s" | grep -q '"priority":"unsure"' && echo ok || echo bad; } || echo ok)"
check "image served"           'HTTP200' "$(curl -s -m 30 -o /dev/null -w 'HTTP%{http_code}' "$B/studies/$id/image")"
check "ood check"              '"ood_flag":'                   "$(curl -s -m 120 -F "file=@scan.png;type=image/png" "$B/ood-check")"
check "override needs reason"  'reason is required'            "$(curl -s -m 30 -X POST -H "$J" -d '{"priority":"raised"}' "$B/studies/$id/override")"
check "override applied"       '"priority_reason":"coordinator_override"' "$(curl -s -m 30 -X POST -H "$J" -d '{"priority":"raised","reason":"deploy check","actor":"verify_deploy"}' "$B/studies/$id/override")"
check "override in audit"      '"action":"override"'           "$(curl -s -m 30 "$B/audit?q=$id")"
check "review"                 '"status":"reviewed"'           "$(curl -s -m 30 -X POST -H "$J" -d '{"action":"mark_reviewed","actor":"verify_deploy"}' "$B/studies/$id/review")"
job=$(curl -s -m 120 -F "files=@scan.png;type=image/png" -F actor=verify_deploy "$B/site-readiness" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
v=""
for _ in $(seq 1 30); do v=$(curl -s -m 30 "$B/site-readiness/$job"); printf '%s' "$v" | grep -q '"status":"done"' && break; sleep 2; done
check "site readiness honest"  '"verdict":"not_enough_evidence"' "$v"
cd / && rm -rf "$tmp"
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || { echo "SOME CHECKS FAILED"; exit 1; }

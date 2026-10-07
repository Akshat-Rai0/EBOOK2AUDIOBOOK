#!/usr/bin/env bash
# Run a small end-to-end VITS conversion and verify its resume behaviour.
#
# Prerequisites:
#   uv run castbook models download vits
#   ffprobe and sqlite3 must be available on PATH.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMP_DIR="$(mktemp -d /tmp/castbook_full_book_test.XXXXXX)"
PROJECT_NAME="full_book_test_$(basename "${TEMP_DIR}" | tr -cd '[:alnum:]_')"
PROJECT_DIR="${PROJECT_ROOT}/projects/${PROJECT_NAME}"
TEST_BOOK="${TEMP_DIR}/test_book.txt"
BEFORE_MTIMES="${TEMP_DIR}/before_mtimes.txt"
AFTER_MTIMES="${TEMP_DIR}/after_mtimes.txt"

cleanup() {
    rm -rf "${TEMP_DIR}" "${PROJECT_DIR}"
}
trap cleanup EXIT

fail() {
    printf '%bError: %s%b\n' "${RED}" "$1" "${RESET}" >&2
    exit 1
}

now() {
    python3 -c 'import time; print(time.perf_counter())'
}

mtime() {
    if stat -f %m "$1" >/dev/null 2>&1; then
        stat -f %m "$1"
    else
        stat -c %Y "$1"
    fi
}

format_bytes() {
    awk -v bytes="$1" 'BEGIN {
        split("B KiB MiB GiB TiB", units, " ");
        value = bytes + 0;
        index = 1;
        while (value >= 1024 && index < 5) { value /= 1024; index++; }
        printf "%.2f %s", value, units[index];
    }'
}

snapshot_done_segment_mtimes() {
    local output_file="$1"
    local audio_path

    : > "${output_file}"
    while IFS= read -r audio_path; do
        [ -n "${audio_path}" ] || continue
        [ -f "${audio_path}" ] || fail "DONE segment WAV is missing: ${audio_path}"
        printf '%s\t%s\n' "${audio_path}" "$(mtime "${audio_path}")" >> "${output_file}"
    done < <(sqlite3 "${STATE_DB}" \
        "SELECT audio_path FROM segments WHERE status = 'done' ORDER BY id;")

    [ -s "${output_file}" ] || fail "No DONE segment WAV files were recorded."
}

for command in uv ffprobe sqlite3 python3; do
    command -v "${command}" >/dev/null 2>&1 || fail "${command} is required but was not found on PATH."
done

cat > "${TEST_BOOK}" <<'BOOK'
Chapter 1: The Beginning

Mira set the lamp down on the table and looked across the room.
"You lied to me," she said quietly. "I did not," Tomas answered.
The silence stretched between them like a fog over still water.

Chapter 2: The Middle

The storm rolled in from the west, grey and cold.
Neither of them spoke for a long time after that.

Chapter 3: The End

By morning the rain had stopped and the roads were clear.
Mira left without saying goodbye.
BOOK

WORD_COUNT="$(wc -w < "${TEST_BOOK}" | tr -d '[:space:]')"
STATE_DB="${PROJECT_DIR}/state.sqlite"
# M4B filename is based on project name
M4B_FILE="${PROJECT_DIR}/output/${PROJECT_NAME}.m4b"

printf '%bFull-Book VITS Test%b\n==================\n' "${GREEN}" "${RESET}"
printf 'Test book: synthetic fixture (%s words)\nProject: %s\n\n' "${WORD_COUNT}" "${PROJECT_NAME}"

printf 'Step 1: ingesting the synthetic book...\n'
(
    cd "${PROJECT_ROOT}"
    uv run castbook ingest "${TEST_BOOK}" --project "${PROJECT_NAME}"
)

printf '\nStep 2: converting with VITS on CPU...\n'
START_TIME="$(now)"
(
    cd "${PROJECT_ROOT}"
    uv run castbook convert --project "${PROJECT_NAME}" --engine vits --device cpu
)
END_TIME="$(now)"
WALL_TIME="$(awk -v start="${START_TIME}" -v end="${END_TIME}" 'BEGIN { printf "%.2f", end - start }')"

[ -f "${M4B_FILE}" ] || fail "M4B output was not created: ${M4B_FILE}"
[ -f "${STATE_DB}" ] || fail "State database was not created: ${STATE_DB}"

printf '\nStep 3: reading M4B metadata with ffprobe...\n'
FFPROBE_JSON="$(ffprobe -v error -select_streams a:0 \
    -show_entries format=duration,bit_rate:stream=codec_name,profile,bit_rate,sample_rate \
    -of json "${M4B_FILE}")"
read -r AUDIO_DURATION AUDIO_CODEC AUDIO_PROFILE AUDIO_BITRATE AUDIO_SAMPLE_RATE <<EOF
$(printf '%s' "${FFPROBE_JSON}" | python3 -c '
import json
import sys

data = json.load(sys.stdin)
stream = data.get("streams", [{}])[0]
format_data = data.get("format", {})
print(
    format_data.get("duration", "unknown"),
    stream.get("codec_name", "unknown"),
    stream.get("profile", "unknown").replace(" ", "_"),
    stream.get("bit_rate", format_data.get("bit_rate", "unknown")),
    stream.get("sample_rate", "unknown"),
)
')
EOF

if [ "${AUDIO_DURATION}" = "unknown" ] || ! awk -v duration="${AUDIO_DURATION}" 'BEGIN { exit !(duration > 0) }'; then
    fail "ffprobe did not report a positive audio duration."
fi
RTF="$(awk -v wall="${WALL_TIME}" -v duration="${AUDIO_DURATION}" \
    'BEGIN { printf "%.2f", wall / duration }')"

# Pipeline versions that record synthesis metrics expose peak_rss in stage metadata.
PEAK_RSS_BYTES="$(sqlite3 "${STATE_DB}" \
    "SELECT json_extract(metadata, '$.peak_rss') FROM stages
     WHERE stage_name = 'synthesis' AND json_extract(metadata, '$.peak_rss') IS NOT NULL
     ORDER BY id DESC LIMIT 1;" 2>/dev/null || true)"
if [ -n "${PEAK_RSS_BYTES}" ] && awk -v value="${PEAK_RSS_BYTES}" 'BEGIN { exit !(value >= 0) }'; then
    PEAK_RSS="$(format_bytes "${PEAK_RSS_BYTES}")"
else
    PEAK_RSS='unavailable (this pipeline did not record synthesis peak_rss metadata)'
fi

printf '\nStep 4: verifying resume behaviour...\n'
snapshot_done_segment_mtimes "${BEFORE_MTIMES}"
(
    cd "${PROJECT_ROOT}"
    uv run castbook convert --project "${PROJECT_NAME}" --engine vits --device cpu
)
snapshot_done_segment_mtimes "${AFTER_MTIMES}"

if cmp -s "${BEFORE_MTIMES}" "${AFTER_MTIMES}"; then
    RESUME_RESULT='PASSED (all DONE segment WAV mtimes unchanged)'
else
    RESUME_RESULT='FAILED (one or more DONE segment WAV mtimes changed)'
fi

printf '\nFull-Book VITS Test Results\n==========================\n'
printf 'Test book: synthetic fixture (%s words)\n' "${WORD_COUNT}"
printf 'Wall time: %ss\n' "${WALL_TIME}"
printf 'Audio duration: %ss\n' "${AUDIO_DURATION}"
printf 'RTF: %s (conversion wall time / audio duration)\n' "${RTF}"
printf 'Peak RSS: %s\n' "${PEAK_RSS}"
printf 'M4B codec: %s (%s), %s bps, %s Hz\n' \
    "${AUDIO_CODEC}" "${AUDIO_PROFILE//_/ }" "${AUDIO_BITRATE}" "${AUDIO_SAMPLE_RATE}"
printf 'Resume test: %s\n' "${RESUME_RESULT}"

if [ "${RESUME_RESULT}" != 'PASSED (all DONE segment WAV mtimes unchanged)' ]; then
    exit 1
fi

printf '%b\nFull-book VITS test passed.%b\n' "${GREEN}" "${RESET}"

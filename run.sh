#!/usr/bin/env bash
#
# Start SmartDoc: the FastAPI backend and the Next.js frontend, together.
#
#   ./run.sh                 start both on the default ports
#   ./run.sh --backend-only  API only (useful for curl / Swagger work)
#   ./run.sh --ui-only       UI only, pointed at an already-running API
#
# Ctrl-C stops both. Ports and log locations are overridable by environment
# variable, so a second instance can be brought up alongside the first:
#
#   API_PORT=8001 UI_PORT=3001 ./run.sh
#
# The UI port is not freely choosable in practice: the API's CORS allowlist
# names the frontend's origin, so a non-default UI_PORT also needs
# CORS_ALLOW_ORIGINS updated in .env, or the browser will block every call.
#
# Python is invoked as `python -m <module>` rather than through the venv's
# console scripts. Those scripts hard-code an absolute interpreter path in
# their shebang, which breaks the moment the project directory is moved;
# `python -m` has no such dependency.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-3000}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/.logs}"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
WEB_DIR="$PROJECT_ROOT/web"

# Seconds to wait for /health before giving up. Cold starts load the Chroma
# collection from disk, so the first boot is slower than later ones.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"

START_BACKEND=1
START_UI=1
for arg in "$@"; do
    case "$arg" in
        --backend-only) START_UI=0 ;;
        --ui-only)      START_BACKEND=0 ;;
        -h|--help)      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Preflight. Fail with an actionable message rather than a stack trace.
# ---------------------------------------------------------------------------

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: no virtualenv at .venv/" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "ERROR: .env is missing. Copy the template and add your API key:" >&2
    echo "  cp .env.example .env" >&2
    exit 1
fi

if ! grep -qE '^OPENAI_API_KEY=.+' "$PROJECT_ROOT/.env"; then
    echo "ERROR: OPENAI_API_KEY is not set in .env." >&2
    echo "  Embedding and generation both need it; the app cannot answer without one." >&2
    exit 1
fi

port_in_use() { lsof -ti :"$1" >/dev/null 2>&1; }

if [ "$START_BACKEND" -eq 1 ] && port_in_use "$API_PORT"; then
    echo "ERROR: port $API_PORT is already in use." >&2
    echo "  Stop the process (lsof -ti :$API_PORT | xargs kill) or set API_PORT." >&2
    exit 1
fi

if [ "$START_UI" -eq 1 ] && port_in_use "$UI_PORT"; then
    echo "ERROR: port $UI_PORT is already in use." >&2
    echo "  Stop the process (lsof -ti :$UI_PORT | xargs kill) or set UI_PORT." >&2
    echo "  Note: a different UI_PORT also needs CORS_ALLOW_ORIGINS updated in .env." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
API_LOG="$LOG_DIR/backend.log"
UI_LOG="$LOG_DIR/frontend.log"

# The UI is a thin HTTP client; this is the only wiring it needs. Kept for the
# retired Streamlit client in app/legacy_v1/; the Next.js client reads
# NEXT_PUBLIC_API_BASE_URL instead, exported in the frontend block below.
export SMARTDOC_API_URL="http://${API_HOST}:${API_PORT}"

# ---------------------------------------------------------------------------
# Shutdown. Both children die with the script, however it exits.
# ---------------------------------------------------------------------------

# Plain variables rather than an array: macOS still ships bash 3.2, which has
# neither `wait -n` nor negative array subscripts.
BACKEND_PID=""
UI_PID=""

shutdown() {
    # Captured first: on an EXIT trap this is the script's real exit status,
    # which must be preserved rather than overwritten by the exit below.
    status=$?
    trap - INT TERM EXIT
    echo ""
    echo "Stopping SmartDoc ..."
    for pid in "$BACKEND_PID" "$UI_PID"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Give them a moment to close their listeners before the shell exits,
    # otherwise an immediate re-run trips the port check above.
    for pid in "$BACKEND_PID" "$UI_PID"; do
        if [ -n "$pid" ]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Stopped."
    # Exit explicitly. A signal trap returns to whatever it interrupted, which
    # here is the watch loop below -- that loop would then find the children
    # dead and report a crash we just caused ourselves. The trap is already
    # cleared above, so this cannot recurse.
    exit "$status"
}
trap shutdown INT TERM EXIT

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

if [ "$START_BACKEND" -eq 1 ]; then
    echo "Starting backend  -> http://${API_HOST}:${API_PORT}  (log: ${API_LOG#$PROJECT_ROOT/})"
    "$VENV_PYTHON" -m uvicorn backend.main:app \
        --host "$API_HOST" --port "$API_PORT" >"$API_LOG" 2>&1 &
    BACKEND_PID=$!

    printf "  waiting for /health "
    ready=0
    for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: the backend exited during startup. Last lines of $API_LOG:" >&2
            tail -n 20 "$API_LOG" >&2
            exit 1
        fi
        if curl -fsS "http://${API_HOST}:${API_PORT}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        printf "."
        sleep 1
    done
    echo ""

    if [ "$ready" -ne 1 ]; then
        echo "ERROR: backend did not become healthy within ${HEALTH_TIMEOUT}s." >&2
        tail -n 20 "$API_LOG" >&2
        exit 1
    fi

    # Surface the corpus size up front: "0 chunks" is the single most common
    # reason a working app still answers "I don't know".
    health=$(curl -fsS "http://${API_HOST}:${API_PORT}/health" 2>/dev/null || echo '{}')
    echo "  ready. $health"
    case "$health" in
        *'"indexed_chunks":0'*|*'"indexed_chunks":null'*)
            echo "  NOTE: the vector store is empty. Upload PDFs in the UI sidebar,"
            echo "        or run: .venv/bin/python -m backend.ingest"
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

if [ "$START_UI" -eq 1 ]; then
    if [ ! -d "$WEB_DIR/node_modules" ]; then
        echo "ERROR: $WEB_DIR/node_modules is missing." >&2
        echo "  Run: (cd web && npm install)" >&2
        exit 1
    fi

    # The browser reaches the API directly, so the client bundle needs the API's
    # address at build time. Passing it here keeps a non-default API_PORT working
    # without editing web/.env.local.
    export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://${API_HOST}:${API_PORT}}"

    echo "Starting frontend -> http://localhost:${UI_PORT}  (log: ${UI_LOG#$PROJECT_ROOT/})"
    echo "  API base for the browser: $NEXT_PUBLIC_API_BASE_URL"
    (cd "$WEB_DIR" && npm run dev -- --port "$UI_PORT") >"$UI_LOG" 2>&1 &
    UI_PID=$!

    printf "  waiting for the UI "
    ready=0
    # 60 rather than 30: a cold Next.js dev start compiles the route tree first.
    for _ in $(seq 1 60); do
        if ! kill -0 "$UI_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: Next.js exited during startup. Last lines of $UI_LOG:" >&2
            tail -n 20 "$UI_LOG" >&2
            exit 1
        fi
        if curl -fsS "http://localhost:${UI_PORT}/login" >/dev/null 2>&1; then
            ready=1
            break
        fi
        printf "."
        sleep 1
    done
    echo ""

    if [ "$ready" -ne 1 ]; then
        echo "ERROR: the UI did not start within 60s." >&2
        tail -n 20 "$UI_LOG" >&2
        exit 1
    fi
    echo "  ready."
fi

echo ""
echo "SmartDoc is running."
# Written as `if` rather than `[ ... ] && echo`: under `set -e` a false test at
# the end of an AND-list is itself a non-zero exit and would kill the script.
if [ "$START_UI" -eq 1 ]; then
    echo "  App          http://localhost:${UI_PORT}"
fi
if [ "$START_BACKEND" -eq 1 ]; then
    echo "  API docs     http://${API_HOST}:${API_PORT}/docs"
fi
echo ""
echo "Press Ctrl-C to stop."

# Block until a child dies or the user interrupts. `wait -n` would be the
# natural tool but does not exist in bash 3.2, so poll instead -- a one-second
# tick is invisible to a human and costs nothing.
while true; do
    if [ -n "$BACKEND_PID" ] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo ""
        echo "The backend exited. Last lines of $API_LOG:" >&2
        tail -n 20 "$API_LOG" >&2
        break
    fi
    if [ -n "$UI_PID" ] && ! kill -0 "$UI_PID" 2>/dev/null; then
        echo ""
        echo "The UI exited. Last lines of $UI_LOG:" >&2
        tail -n 20 "$UI_LOG" >&2
        break
    fi
    sleep 1
done

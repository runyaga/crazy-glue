#!/bin/bash
# Start haiku-rag MCP server
# Usage: ./bin/start-rag-mcp.sh [port]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

PORT="${1:-8001}"

echo "Starting haiku-rag MCP server on port $PORT..."
exec haiku-rag --config haiku-rag.yaml serve --db db/rag.lancedb --mcp --mcp-port "$PORT"

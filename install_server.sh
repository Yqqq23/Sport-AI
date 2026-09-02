#!/usr/bin/env bash

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASKETBALL_V2_ROOT="${REPO_ROOT}"

exec bash "${REPO_ROOT}/scripts/install_server.sh" "$@"

#!/usr/bin/env bash
# Run on the LOCAL computer (bash/WSL), after replacing these values.
set -euo pipefail

LOCAL_ARCHIVE='/mnt/e/Datasets/MSG-SEVIRI'
BRIDGE_USER='bridge_user'
BRIDGE_HOST='BRIDGE_IP'
BRIDGE_PORT=22
WORKSPACE_USER='isac'
WORKSPACE_HOST='WORKSPACE_IP_REACHABLE_FROM_BRIDGE'
WORKSPACE_PORT=22
REMOTE_ARCHIVE='/data1/deep_convection_data/MSG-SEVIRI/'

# ProxyJump forwards SSH through the bridge; no dataset is staged there.
# Both hosts must accept your SSH credentials. Keep host-key verification on.
# Trailing slash on the source copies its CONTENTS (2020/20200101.zarr, ...).
# Pass --dry-run on the first invocation to inspect the transfer.
# Add --no-inc-recursive for a fixed archive total after a full initial scan.
rsync -a --partial --partial-dir=.rsync-partial --info=progress2 \
  --exclude='.rsync-partial/' \
  -e "ssh -p ${WORKSPACE_PORT} -J ${BRIDGE_USER}@${BRIDGE_HOST}:${BRIDGE_PORT} -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  "$@" "${LOCAL_ARCHIVE}/" \
  "${WORKSPACE_USER}@${WORKSPACE_HOST}:${REMOTE_ARCHIVE}"

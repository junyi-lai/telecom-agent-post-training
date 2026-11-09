#!/usr/bin/env bash

load_training_profile() {
  local root="${1:?project root is required}"
  PROFILE_FILE="${PROFILE_FILE:-configs/profiles/96gb.env}"
  if [[ "$PROFILE_FILE" != /* ]]; then
    PROFILE_FILE="$root/$PROFILE_FILE"
  fi
  export PROFILE_FILE
  if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "Profile not found: $PROFILE_FILE" >&2
    return 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$PROFILE_FILE"
  set +a
}

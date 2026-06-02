#!/usr/bin/env bash
# Extract the changelog section for the current git tag from CHANGELOG.md.
# Outputs the section body (without the ## header line) to stdout.
set -euo pipefail

CHANGELOG="${1:-CHANGELOG.md}"
TAG="${GITHUB_REF_NAME:-$(git describe --tags --exact-match 2>/dev/null)}"

if [[ -z "$TAG" ]]; then
    echo "Error: no tag found (GITHUB_REF_NAME unset and git describe failed)" >&2
    exit 1
fi

# Extract lines between the header matching the tag and the next ## header.
awk -v tag="$TAG" '
    /^## \[/ {
        if (found) exit
        if (index($0, "[" tag "]")) { found=1; next }
    }
    found { print }
' "$CHANGELOG"

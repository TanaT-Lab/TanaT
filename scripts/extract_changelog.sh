#!/usr/bin/env bash
# Extract the changelog section for the current git tag from CHANGELOG.md.
# Outputs the section body (without the ## header line) to stdout.
#
# Usage: ./scripts/extract_changelog.sh [VERSION]
#   VERSION: version tag (e.g., v0.10.1)
#            Falls back to GITHUB_REF_NAME, then git describe.
#
set -euo pipefail

CHANGELOG="${1:-CHANGELOG.md}"
VERSION="${2:-${GITHUB_REF_NAME:-$(git describe --tags --exact-match 2>/dev/null || true)}}"

if [[ -z "$VERSION" ]]; then
    echo "Error: no tag found (GITHUB_REF_NAME unset and git describe failed)" >&2
    exit 1
fi

# Extract lines between the header matching the tag and the next ## header.
# Matches both "## [v0.10.1]" and "## v0.10.1" formats.
NOTES=$(awk -v tag="$VERSION" '
    /^## \[?[0-9]/ {
        if (found) exit
        if (index($0, "[" tag "]") || index($0, " " tag)) { found=1; next }
    }
    found { print }
' "$CHANGELOG")

# Trim leading/trailing blank lines
NOTES=$(echo "$NOTES" | sed -e '/./,/^$/!d')

if [[ -z "$NOTES" ]]; then
    echo "Release $VERSION"
else
    echo "$NOTES"
fi

#!/usr/bin/env bash
set -euo pipefail

# Help function
function show_help() {
    echo "Usage: $0 [DIRECTORY]"
    echo "Generates __init__.py files in all subdirectories of the specified directory"
    echo ""
    echo "Arguments:"
    echo "  DIRECTORY    The directory to process (default: src/tanat relative to script)"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show this help"
}

# Function to generate stubs
function gen_stub() {
    local dir_path="$1"
    local stub_path="$dir_path/__init__.py"
    
    # Skip __pycache__ directories and create stub only if it doesn't exist
    if [[ "${dir_path##*/}" != "__pycache__" && ! -e "$stub_path" ]]; then
        echo "Creating $stub_path..."
        cat > "$stub_path" <<'STUB'
#!/usr/bin/env python3
"""Package stub."""
STUB
    fi
}

# Export function to make it accessible in subshells
export -f gen_stub

# Argument processing
if [[ $# -eq 0 ]]; then
    # Default behavior: use src/tanat relative to script
    SELF=$(readlink -f "${BASH_SOURCE[0]}")
    DIR=${SELF%/*/*}
    TARGET_DIR="$DIR/src/tanat"
elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
else
    # Use the directory specified as parameter
    TARGET_DIR="$1"
fi

# Check that the directory exists
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

# Convert to absolute path to avoid issues
TARGET_DIR=$(readlink -f "$TARGET_DIR")

echo "Processing directory: $TARGET_DIR"

# Process all subdirectories and generate stubs
find "$TARGET_DIR" -type d -exec bash -c 'gen_stub "$0"' {} \;
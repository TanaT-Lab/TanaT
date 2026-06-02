#!/bin/bash

# Check if both arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./gtag.sh [version] [message]"
    echo "Example: ./gtag.sh v1.2.3 'Fix login bug'"
    exit 1
fi

VERSION=$1
MESSAGE=$2

echo "--- Creating tag: $VERSION ---"

# Create the annotated tag
git tag -a "$VERSION" -m "$MESSAGE"

# Check if the tag was created successfully
if [ $? -eq 0 ]; then
    echo "Tag created successfully locally."
    
    # Push the tag to origin
    echo "Pushing tag to remote (origin)..."
    git push origin "$VERSION"
    
    if [ $? -eq 0 ]; then
        echo "Done! Tag $VERSION has been pushed."
    else
        echo "Error: Failed to push the tag to remote."
        exit 1
    fi
else
    echo "Error: Failed to create tag (it might already exist)."
    exit 1
fi
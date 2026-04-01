#!/bin/bash

: '
GIT WORKTREE MANAGER
====================
Description:
    A utility script to streamline the creation and removal of git worktrees.
    Worktrees allow you to have multiple branches checked out simultaneously 
    in separate directories.

Usage:
    ./git-wt.sh add <branch_name>     : Creates a new branch and a worktree in ../<branch_name>
    ./git-wt.sh remove <branch_name>  : Removes the worktree located at ../<branch_name>

Arguments:
    add/remove : The action to perform.
    name       : The name of the branch/directory.
'

# Configuration
PARENT_DIR=".."
MAIN_BRANCH="main"

# Check for minimum arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 {add|remove} <name>"
    exit 1
fi

COMMAND=$1
NAME=$2

case "$COMMAND" in
    add)
        echo "Adding worktree: $NAME"
        # Command: git worktree add -b <name> ../<name> main
        git worktree add -b "$NAME" "$PARENT_DIR/$NAME" "$MAIN_BRANCH"
        ;;
    remove)
        echo "Removing worktree: $NAME"
        # Command: git worktree remove ../<name>
        if [ -d "$PARENT_DIR/$NAME" ]; then
            git worktree remove "$PARENT_DIR/$NAME"
        else
            echo "Error: Worktree directory '$PARENT_DIR/$NAME' not found."
        fi
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Please use 'add' or 'remove'."
        exit 1
        ;;
esac
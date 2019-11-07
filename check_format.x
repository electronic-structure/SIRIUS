#!/bin/sh

git diff-tree --no-commit-id --name-only -r HEAD | python3 check_format.py

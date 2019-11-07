#!/bin/sh

git diff-tree --no-commit-id --name-only -r HEAD | python check_format.py

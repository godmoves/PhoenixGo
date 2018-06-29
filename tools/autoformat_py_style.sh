#!/usr/bin/env bash

# cd to parent of script's directory (i.e. project root)
cd "${0%/*}"/..

find . -type f \( -name '*.py' \) \
    -not -path './.git/*' \
    -not -path './ckpt/*' \
    -not -path './etc/*' \
    -not -path './log/*' \
    -not -path './third_party/*' \
    -exec autopep8 -i {} +

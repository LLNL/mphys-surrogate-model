#!/bin/bash

# A simple script to build all sphinx docs quickly.
# Should be run in this directory
# Every module should have a corresponding `sphinx-apidoc` command

# Delete all .rst files except for index
#echo "Removing..."
rm -fv ./source/modules.rst
rm -fv ./source/src*.rst

# Clean HTML
make clean

# Autodoc Generate
sphinx-apidoc -f --separate -o source/ ../src/

# Make
make html
make latexpdf

# Open
if [[ "$(uname)" == "Darwin" ]]; then
  open ./build/html/index.html
else
  true
fi

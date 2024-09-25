#!/bin/bash
source ~/.bashrc
cp $1 $1.sys
tmp=$RESULT
echo "$tmp"
sed -i 's|RESULT|'"${RESULT}"'|g' $1.sys

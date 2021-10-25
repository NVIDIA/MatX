#!/bin/bash
LICENSEFILE=LICENSE
LICENSELEN=$(wc -l $LICENSEFILE | cut -f1 -d ' ')
FILES=$(find ./{include,test,examples}/ -type f \( -name "*.h" -o -name "*.cu" \))
 
for file in $FILES; do
  echo $file
  head -$LICENSELEN $file | diff $LICENSEFILE - || ( (cat $LICENSEFILE; echo; cat $file) > /tmp/file ; mv /tmp/file $file )
done

#!/bin/bash
shopt -s extglob
PDIR=proprietary
script=$(basename "$0")
todel=""
prop=""
dset=0

if [ $# -gt 2 ]; then
    echo "Too many arguments!"
    exit 1
fi

function help() {
    echo "Create a release of MatX"
    echo "$script -d dir"
    echo "If dir is specified only proprietary files from there will be included."
    echo "Not specifying dir means all proprietary files are included"
    echo "Specifying NONE means no proprietary files will be included"
}

while test $# -gt 0; do
  case "$1" in
    -h|--help)
        help
        exit 0
        ;;
    -d)
        dset=1  
        shift
        if test $# -gt 0; then
            prop=$1
            if [ "$prop" = "NONE" ]; then
                echo "No proprietary files will be included in this release"
                pfiles=$(find $PDIR)
            else
                echo "Stripping build to only include $prop files"
                pfiles=$(ls -d $PDIR/!(*"$1"))
            fi

            for d in $pfiles; do
                todel="$todel --exclude='$d'"
            done                 
        else
            prop="all"
            echo "No directory specified. Including all proprietary files!"
        fi
        ;;
    *)
        break
        ;;
  esac
done

if [ $dset -ne 1 ]; then
    echo "Error: Must pass -d flag"
    help
    exit 1
fi

# Drop untracked files
pfiles=$(git ls-files --others --exclude-standard)
for d in $pfiles; do
    todel="$todel --exclude='$d'"
done 

commit=$(git log --format="%h" -n 1)
filename="matx_${prop}_${commit}.tgz"
cmd="tar --exclude $script $todel --exclude build -zcf $filename --exclude-vcs-ignores --exclude-vcs *"
echo "Creating $prop release with name $filename"
echo "$cmd"
eval "$cmd"



#!/bin/bash

OPEN=NO
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -o|--open)
      OPEN=YES
      shift # past argument
      ;;
    -h|--help)
      HELP=YES
      shift # past argument
      ;;
  *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

if ! [ -z ${HELP+x} ]
then 
    cat 
fi <<-EOF
Usage: $0 [options]

    -h| --help           show help
    -o|--open            open docs in firefox after build
EOF

if [ -z ${HELP+x} ]
then
    ###### ACTUAL SCRIPT
    rm -rf _build
    make html
    ######

    if [[ $OPEN == "YES" ]] 
    then
        firefox _build/html/index.html
    fi
fi
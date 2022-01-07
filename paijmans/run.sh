#!/bin/bash
for mode in default modified; do
  for i in {0..6}; do
    cmd="./KMCKaiC"
    for kaib in woKaiB wKaiB; do
      FNAME="${mode}_${kaib}_p${i}"
      cat "${mode}_${kaib}_tmplt.par" | sed s/FILENAME/${FNAME}/ | sed s/STRTPHOS/${i}/ > "${FNAME}.par"
      cmd="${cmd} ${FNAME}.par"
    done
    echo ${cmd}
    while :
    do
      SECONDS=0
      $cmd
      if [ $SECONDS -gt 5 ]; then
        break
      fi
    done
  done
done

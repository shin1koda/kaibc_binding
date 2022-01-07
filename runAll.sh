#!/bin/bash
. ./installPaijmans.sh
. ./runPaijmans.sh
python3 ./figPaijmans.py
python3 ./model6.py
python3 ./model1_5.py

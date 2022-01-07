#!/bin/sh
wget https://sourceforge.net/projects/kmc-kaic/files/KMC_KaiC_Original.zip
unzip -d paijmans KMC_KaiC_Original.zip
patch -p1 -d paijmans < paijmans.patch
cd paijmans
make
cd ..

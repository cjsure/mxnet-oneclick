@echo off
echo "Generating lst"
python preprocess/im2rec.py platechars/chars platechars --recursive=True --list=True
echo "Generating rec"
python preprocess/im2rec.py platechars/chars platechars
echo "Generating synsetwords"
python preprocess/generatesynsetwords.py --datadir=platechars
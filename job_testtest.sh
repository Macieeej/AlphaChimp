#!/bin/bash
#SBATCH -J alphachimp
#SBATCH -o /home/b5bd/tc19422.b5bd/AlphaChimp/logs/alphachimp-%j.out
#SBATCH -e /home/b5bd/tc19422.b5bd/AlphaChimp/logs/alphachimp-%j.err

set -x
echo "START"
date
hostname
pwd

cd /home/b5bd/tc19422.b5bd/AlphaChimp || exit 1
echo "CD OK"

which python
python --version || exit 1

echo "LIST ACTION FILES"
ls -l data/ChimpACT_processed/annotations/action || exit 1

echo "SETUP OK"
sleep 30

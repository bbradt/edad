#!/bin/sh
SITES=( 2 3 4 5 6 7 8 9 10 )
MODES=("pooled" "edad" "dsgd" "noshare")
SPLITS=("class" "random")

for site in "${SITES[@]}"
do
sbatch -J EDAD-S$site -o logs/increase_sites$site.log -e logs/increase_sites$site.err experiments/runner.sh "--n-sites ${site}"
done


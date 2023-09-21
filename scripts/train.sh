set -e

export PROJECT_ROOT="/root/repo/Siim-segmentation"
# rememberr to run for fold 3,4,5 
for FOLD in {1..5}
do
PYTHONPATH="${PROJECT_ROOT}" \
python "${PROJECT_ROOT}"/src/run.py "${PROJECT_ROOT}"/config/config.json --fold=$FOLD
done

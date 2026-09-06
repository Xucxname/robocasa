#!/usr/bin/env bash

set -Eeuo pipefail

readonly DATA_PYTHON="/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python"
readonly CLEANER="/home/user/robocasa/robocasa/scripts/clean_sonic_dataset.py"
readonly LOG_FILE="/home/user/robocasa/datasets/kettle_cheesy_mode_aware_cleanup.log"

"${DATA_PYTHON}" -u "${CLEANER}" \
    /home/user/robocasa/datasets/robocasa_CloseElectricKettleLid_g1_3cam \
    /home/user/robocasa/datasets/robocasa_CheesyBread_g1_3cam \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

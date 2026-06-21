#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
bash Motion_Deblurring/eval_all_motiondeblur.sh

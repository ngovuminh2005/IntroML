#!/usr/bin/env bash
set -euo pipefail

PY="python"
CONFIG="Motion_Deblurring/Options/Deblurring_Restormer.yml"
LAUNCHER="none"
CONFIG_OUT="/tmp/restormer_train_muon_$$.yml"
AUTO_RESUME="false"

# Optimizer options:
#   Adam                         use LR, WEIGHT_DECAY, BETA1, BETA2
#   AdamW                        use LR, WEIGHT_DECAY, BETA1, BETA2
#   SingleDeviceMuonWithAuxAdam  use MUON_* for matrix-like params and MUON_AUX_* for bias/norm/scalar params
# Weight decay defaults are shared: MUON_WEIGHT_DECAY and MUON_AUX_WEIGHT_DECAY point to WEIGHT_DECAY below.
OPTIMIZER="SingleDeviceMuonWithAuxAdam"
LR="3e-4"
WEIGHT_DECAY="1e-4"
BETA1="0.9"
BETA2="0.999"

MUON_LR="3e-3"
MUON_MOMENTUM="0.95"
MUON_WEIGHT_DECAY="$WEIGHT_DECAY"
MUON_NESTEROV="true"
MUON_NS_STEPS="5"
MUON_AUX_LR="$LR"
MUON_AUX_WEIGHT_DECAY="$WEIGHT_DECAY"
MUON_AUX_BETA1="$BETA1"
MUON_AUX_BETA2="$BETA2"
MUON_AUX_EPS="1e-8"

export RESTORMER_AUTO_RESUME="$AUTO_RESUME"
export CONFIG OPTIMIZER LR WEIGHT_DECAY BETA1 BETA2
export MUON_LR MUON_MOMENTUM MUON_WEIGHT_DECAY MUON_NESTEROV MUON_NS_STEPS
export MUON_AUX_LR MUON_AUX_WEIGHT_DECAY MUON_AUX_BETA1 MUON_AUX_BETA2 MUON_AUX_EPS CONFIG_OUT


"$PY" - <<'RESTORMER_CFG_PY'
from pathlib import Path
import os

config = Path(os.environ['CONFIG'])
out = Path(os.environ['CONFIG_OUT'])
lines = config.read_text().splitlines()
try:
    start = next(i for i, line in enumerate(lines) if line.strip() == 'optim_g:')
except StopIteration as exc:
    raise SystemExit(f'optim_g block not found in {config}') from exc
end = start + 1
while end < len(lines):
    line = lines[end]
    if line and not line.startswith('    ') and line.startswith('  '):
        break
    if line and not line.startswith('  '):
        break
    end += 1

def bool_value(value):
    value = value.lower()
    if value == 'true':
        return 'true'
    if value == 'false':
        return 'false'
    raise SystemExit(f'Invalid MUON_NESTEROV={os.environ["MUON_NESTEROV"]}')

block = [
    '  optim_g:',
    f'    type: {os.environ["OPTIMIZER"]}',
    f'    lr: !!float {os.environ["LR"]}',
    f'    weight_decay: !!float {os.environ["WEIGHT_DECAY"]}',
    f'    betas: [{os.environ["BETA1"]}, {os.environ["BETA2"]}]',
    f'    muon_lr: !!float {os.environ["MUON_LR"]}',
    f'    muon_momentum: !!float {os.environ["MUON_MOMENTUM"]}',
    f'    muon_weight_decay: !!float {os.environ["MUON_WEIGHT_DECAY"]}',
    f'    muon_nesterov: {bool_value(os.environ["MUON_NESTEROV"])}',
    f'    muon_ns_steps: {os.environ["MUON_NS_STEPS"]}',
    f'    muon_aux_lr: !!float {os.environ["MUON_AUX_LR"]}',
    f'    muon_aux_weight_decay: !!float {os.environ["MUON_AUX_WEIGHT_DECAY"]}',
    f'    muon_aux_beta1: !!float {os.environ["MUON_AUX_BETA1"]}',
    f'    muon_aux_beta2: !!float {os.environ["MUON_AUX_BETA2"]}',
    f'    muon_aux_eps: !!float {os.environ["MUON_AUX_EPS"]}',
]
out.write_text('\n'.join(lines[:start] + block + lines[end:]) + '\n')
print(f'Wrote training config: {out}')
RESTORMER_CFG_PY

echo "Training Restormer with:"
echo "  config:       $CONFIG_OUT"
echo "  optimizer:    $OPTIMIZER"
echo "  lr:           $LR"
echo "  weight_decay: $WEIGHT_DECAY"
echo "  launcher:     $LAUNCHER"
echo "  auto_resume:  $AUTO_RESUME"

exec "$PY" -m basicsr.train -opt "$CONFIG_OUT" --launcher "$LAUNCHER"

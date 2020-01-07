#!/bin/bash
which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi
python -c "import torch"
if [ $? -eq "1" ]; then echo "torch failed to import. You need to install pytorch (no need for cuda)"; exit; fi
python -c "from numpy import ones; ones(1) @ ones(1)"
if [ $? -eq "1" ]; then echo "Python >=3.5 required"; exit; fi

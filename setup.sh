#!/bin/bash

# 1. Load System Modules
module load ffmpeg
module load cuda/12.1

# 2. Activate Python Environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/.venv/bin/activate"

# 3. Fix the "Rats" (BYU-Specific Pathing)
export FFMPEG_LIB=/apps/spack/root/opt/spack/linux-rhel9-haswell/gcc-13.2.0/ffmpeg-7.0.1-pzg5pllmqfjzz2ubrlm3jcxyyh7gtpyr/lib
export LD_LIBRARY_PATH=$FFMPEG_LIB:$LD_LIBRARY_PATH

# 4. FIPS Security Bypass
export LD_PRELOAD=/usr/lib64/libcrypto.so.1.1.1k:/usr/lib64/libssl.so.1.1.1k

# 5. Headless Rendering for Simulation
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# export HF_HUB_OFFLINE=1

echo "VLA-0 Environment Ready."
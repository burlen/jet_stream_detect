#!/bin/bash

source /usr/common/software/teca/3.0.0/bin/teca_env.sh
echo "+ source /usr/common/software/teca/3.0.0/bin/teca_env.sh"

set -o xtrace

data_dir=/project/projectdirs/m1517/cascade/teca_tutorial/sample_wind_data/
first=0
last=-1

python3 ./teca_jet_stream_sinuosity \
    --first_step=${first} --last_step=${last} \
    --input_regex=${data_dir}/cam5_1_amip_run2.cam2.h2.1980-12-'.*\.nc$' \
    --area_threshold 7.e6 --output_file=sinuosity.bin

python3 ./plot_sinuosity.py ./sinuosity.bin ./sinuosity.png


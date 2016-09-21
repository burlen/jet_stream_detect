# jet_stream_detect
place to share jet stream detection code with Jiabin

```bash
mpiexec -np 10 teca_max_wind_cc_jet_detect \
    --input_regex=/home/bloring/work/teca/jet/cam5_1_amip_run2'.*\.nc' \
    --output_file=test.csv -p
```

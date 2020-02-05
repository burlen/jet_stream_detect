from teca import *
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 3:
    sys.stderr.write('usage: plot_sinuosity.py [input table] [output image]\n')
    sys.exit(1)

tr = teca_table_reader.New()
tr.set_file_name(sys.argv[1])

dc = teca_dataset_capture.New()
dc.set_input_connection(tr.get_output_port())

dc.update()

tab = as_teca_table(dc.get_dataset())

md = tab.get_metadata()

time_units = md['time_units']
bounds = md['bounds']

sinuosity = tab.get_column('sinuosity').as_array()
hemisphere = tab.get_column('hemisphere').as_array()
time = tab.get_column('time').as_array()

nh = np.where(hemisphere == 0)[0]
sh = np.where(hemisphere == 1)[0]

num_cols = 2 if len(nh) > 0 and len(sh) > 0 else 1

if num_cols == 2:
    plt.subplot(1,2,1)

if len(nh) > 0:
    plt.plot(time[nh], sinuosity[nh], 'g-', linewidth=2)
    plt.xlabel('time', fontweight='bold')
    plt.ylabel('sinuosity', fontweight='bold')
    plt.title('Jet Stream Sinuosity (NH)\nRegion %s'%(str(bounds)), fontweight='bold')
    plt.grid(True)

if num_cols == 2:
    plt.subplot(1,2,2)

if len(sh) > 0:
    plt.plot(time[sh], sinuosity[sh], 'b-', linewidth=2)
    plt.xlabel('time (%s)'%(time_units), fontweight='bold')
    plt.ylabel('sinuosity', fontweight='bold')
    plt.title('Jet Stream Sinuosity (SH)\nRegion %s'%(str(bounds)), fontweight='bold')
    plt.grid(True)


plt.savefig(sys.argv[2], dpi=150)

plt.show()

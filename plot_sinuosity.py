from teca import *
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 3:
    sys.stderr.write('usage: plot_sinuosity.py [input table] [output image]\n')
    sys.exit(1)

# construct the pipeline to read and access the data
tr = teca_table_reader.New()
tr.set_file_name(sys.argv[1])

dc = teca_dataset_capture.New()
dc.set_input_connection(tr.get_output_port())

# run the pipeline
dc.update()

# get the data as a table and columns of data from the table
tab = as_teca_table(dc.get_dataset())

sinuosity = tab.get_column('sinuosity').as_array()
hemisphere = tab.get_column('hemisphere').as_array()
time = tab.get_column('time').as_array()
year = tab.get_column('year').as_array()
month = tab.get_column('month').as_array()
day = tab.get_column('day').as_array()

# get the metadata for plot annotations. units etc
md = tab.get_metadata()

time_units = md['time_units']
calendar = md['calendar']

bounds = md['bounds']

xv = md['x_coordinate_variable']
yv = md['y_coordinate_variable']
zv = md['z_coordinate_variable']

atts = md['attributes']
xv_units = atts[xv]['units']
yv_units = atts[yv]['units']
zv_units = atts[zv]['units']

# use min max to make the y axis in plots the same
# adda little space above and below the line
sin_min = np.min(sinuosity)
sin_min -= 0.02*sin_min

sin_max = np.max(sinuosity)
sin_max += 0.02*sin_max

# plit the data into northern and souther hemisphere
nh = np.where(hemisphere == 0)[0]
sh = np.where(hemisphere == 1)[0]

# if the we only have one or the other hemispsher then
# make only one plot
num_cols = 2 if len(nh) > 0 and len(sh) > 0 else 1

fig = plt.figure(figsize=(8.0 if num_cols == 2 else 6.0, 4.0))
if num_cols == 2:
    plt.subplot(1,2,1)

# plot the norther hemisphere
if len(nh) > 0:
    plt.plot(time[nh], sinuosity[nh], 'g-', linewidth=2)
    plt.ylim((sin_min, sin_max))
    plt.xlabel('time (%s, %s)'%(time_units, calendar))
    plt.ylabel('sinuosity')
    plt.title('Northern Hemisphere', fontsize=10)
    plt.grid(True)

if num_cols == 2:
    plt.subplot(1,2,2)

# plot the southern hemisphere
if len(sh) > 0:
    plt.plot(time[sh], sinuosity[sh], 'b-', linewidth=2)
    plt.ylim((sin_min, sin_max))
    plt.xlabel('time (%s)'%(time_units))
    plt.ylabel('sinuosity')
    plt.title('Southern Hemisphere.', fontsize=10)
    plt.grid(True)

# make title that describes the data
plt.suptitle('Sinuosity Index. %s/%s/%s - %s/%s/%s. %s %0.2f %s\n' \
    '%s (%0.2f - %0.2f) %s, %s (%0.2f - %0.2f) %s'%( \
    year[0], month[0], day[0], year[-1], month[-1], day[-1], \
    zv, bounds[4], zv_units, yv, bounds[2], bounds[3], yv_units, \
    xv, bounds[0], bounds[1], xv_units), fontsize=12, fontweight='normal')

plt.subplots_adjust(wspace=0.25, top=0.825)

# save the plot
plt.savefig(sys.argv[2], dpi=150)


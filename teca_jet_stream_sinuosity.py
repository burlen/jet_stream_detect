from teca import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

Re = 6371.0088 #km
pi = np.pi
pi_over_180 = pi/180.
pi_over_2 = pi/2.

def great_circle_vincenty(deg_lat_0, deg_lon_0, deg_lat_1, deg_lon_1, R=Re):

    lat_0 = deg_lat_0*pi_over_180
    lon_0 = deg_lon_0*pi_over_180

    lat_1 = deg_lat_1*pi_over_180
    lon_1 = deg_lon_1*pi_over_180

    # use Vincenty to calculate central angle
    # 7 trig funcs, 8 mul, 1 sqrt, 4 add
    dlon = lon_1 - lon_0
    sin_dlon = np.sin(dlon)
    cos_dlon = np.cos(dlon)

    cos_lat_0 = np.cos(lat_0)
    sin_lat_0 = np.sin(lat_0)

    cos_lat_1 = np.cos(lat_1)
    sin_lat_1 = np.sin(lat_1)

    a = cos_lat_1*sin_dlon
    b = cos_lat_0*sin_lat_1 - sin_lat_0*cos_lat_1*cos_dlon
    c = sin_lat_0*sin_lat_1
    d = cos_lat_0*cos_lat_1*cos_dlon

    num = np.sqrt( a*a + b*b )
    den = c + d

    sig = np.arctan2(num, den)

    return R*sig

def parallel_distance(lat, deg_lon_0, deg_lon_1, R = Re):
    """ Calculates the length of a parallel at the given latitude `lat' """

    lat_rad = pi_over_180 * lat
    dlon_rad = pi_over_180 * (deg_lon_1 - deg_lon_0)

    return dlon_rad * R * np.cos(lat_rad)



class teca_jet_stream_sinuosity(teca_python_algorithm):
    """ Calculate jet stream sinuosity for northern and
    southern hemisphere. Takes as input a table containing
    the topological spine of the jet stream """
    def __init__(self):
        self.out_file = 'sinuosity'
        self.plot = False
        self.interact = False
        self.dpi = 100
        self.out_file = ''
        self.verbose = False

    def set_out_file(self, out_file):
        """ set the prefix for debug images """
        self.out_file = out_file

    def set_interact(self, interact):
        """
        If true then plots are displayed in pop up window
        """
        self.interact = interact

    def set_dpi(self, dpi):
        """
        Set output image DPI
        """
        self.dpi = dpi

    def set_plot(self, plot):
        """
        If true then plots are generated
        """
        self.plot = plot

    def get_execute_callback(self):
        """
        returns the function that implements the execute
        phase of the TECA pipeline.
        """
        def execute(port, data_in, req):

            # get the input
            table_in = as_teca_table(data_in[0])

            # set up the output
            table_out = teca_table.New()
            table_out.copy_metadata(table_in)

            table_out.declare_columns(['step','time','gid', \
                'hemisphere', 'mean_lat', 'arc_len', 'direct_len', \
                'sinuosity'], ['l', 'd', 'l', 'i', 'd', 'd', 'd', 'd'])

            gid = 0

            step = table_in.get_column('step').as_array()
            time = table_in.get_column('time').as_array()
            comp = table_in.get_column('comp_id').as_array()
            lat = table_in.get_column('lat').as_array()
            lon = table_in.get_column('lon').as_array()

            # work time step by time step
            if self.plot:
                fig = plt.figure(figsize=(8.0, 4.0))

            step_0 = np.min(step)
            step_1 = np.max(step)

            for step_i in range(step_0, step_1+1):

                i = np.where(step == step_i)[0]

                time_i = time[i[0]]
                comp_i = comp[i]
                lat_i = lat[i]
                lon_i = lon[i]

                # calculate the mean latitude of the spine, and the most
                # easterly and westerly longitude. this will be used to
                # calculate the length along a parallel
                j = np.where(lat_i >= 0.)[0]
                mean_lat_nh = 0. if len(j) == 0 else np.mean(lat_i[j])
                lon_0_nh = 0. if len(j) == 0 else np.min(lon_i[j])
                lon_1_nh = 0. if len(j) == 0 else np.max(lon_i[j])

                j = np.where(lat_i < 0.)[0]
                mean_lat_sh = 0. if len(j) == 0 else np.mean(lat_i[j])
                lon_0_sh = 0. if len(j) == 0 else np.min(lon_i[j])
                lon_1_sh = 0. if len(j) == 0 else np.max(lon_i[j])

                # work feature by feature
                comp_0 = np.min(comp_i)
                comp_1 = np.max(comp_i)

                # calculate the arc length from each component
                sinuosity_nh = 0.
                sinuosity_sh = 0.
                arc_len_nh = 0.
                arc_len_sh = 0.
                for comp_j in range(comp_0, comp_1+1):

                    j = np.where(comp_i == comp_j)[0]

                    if len(j) == 0:
                        sys.stderr.write('step %d comp %d missing\n'%(step_i, comp_j))
                        continue

                    lat_ij = lat_i[j]
                    lon_ij = lon_i[j]

                    # compute the length along this path
                    arc_len = 0.
                    for k in range(0, len(lat_ij)-1):
                        arc_len += great_circle_vincenty(lat_ij[k], lon_ij[k], \
                             lat_ij[k+1], lon_ij[k+1])

                    # add this segment's length to either northern or southern
                    # hemisphere total
                    nh = True if lat_ij[0] > 0. else False
                    if nh:
                        arc_len_nh += arc_len
                    else:
                        arc_len_sh += arc_len

                    # add this segment to plot
                    if self.plot:
                        plt.plot(lon_ij, lat_ij, 'g' if nh else 'b', linewidth=2)


                # for northern hemishpere
                if arc_len_nh > 0.:
                    gid_i = 1000000*step_i + gid

                    # calculate sinuosity
                    direct_len_nh = parallel_distance(mean_lat_nh, \
                         lon_0_nh, lon_1_nh)

                    sinuosity_nh = arc_len_nh/direct_len_nh

                    # put it int the output dataset
                    table_out << step_i << time_i << gid_i << 0 << mean_lat_nh \
                        << arc_len_nh << direct_len_nh << sinuosity_nh

                    gid += 1

                # for southern hemisphere
                if arc_len_sh > 0.:
                    gid_i = 1000000*step_i + gid

                    # calculate sinuosity
                    direct_len_sh = parallel_distance(mean_lat_sh, \
                        lon_0_sh, lon_1_sh)

                    sinuosity_sh = arc_len_sh/direct_len_sh

                    table_out << step_i << time_i << gid_i << 1 << mean_lat_sh \
                        << arc_len_sh << direct_len_sh << sinuosity_sh

                    gid += 1

                # plot for the tutorial/demo
                if self.plot:
                    if arc_len_nh > 0.:
                        plt.plot([lon_0_nh, lon_1_nh], [mean_lat_nh, mean_lat_nh], 'y--o', \
                            linewidth=2, mec='y', mfc=(0.,0.,0.,0.), mew=2)

                    if arc_len_sh > 0.:
                        plt.plot([lon_0_sh, lon_1_sh], [mean_lat_sh, mean_lat_sh], 'c--o', \
                            linewidth=2, mec='c', mfc=(0.,0.,0.,0.), mew=2)

                    plt.axis('equal')
                    plt.grid(True)
                    plt.title('Sinuosity NH=%g SH=%g step %d'%(sinuosity_nh, sinuosity_sh, step_i))
                    plt.xlabel('deg lon')
                    plt.ylabel('deg lat')
                    plt.savefig('%s_sinuosity_%06d.png'%(self.out_file, step_i), dpi=self.dpi)
                    if self.interact:
                        plt.show()

            return table_out
        return execute


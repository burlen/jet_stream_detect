from teca import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

Re = 6371.0088 #km
pi = 3.14159265358979
pi_over_180 = pi/180.
pi_over_2 = pi/2.

def great_circle_distance_0(deg_lat_0, deg_lon_0, deg_lat_1, deg_lon_1, R=Re):

    lat_0 = deg_lat_0*pi_over_180
    lon_0 = deg_lon_0*pi_over_180

    lat_1 = deg_lat_1*pi_over_180
    lon_1 = deg_lon_1*pi_over_180

    # use dot product to calculate the central angle.
    # 9 trig func calls, 12 mul, 1 div, 4 add
    rho = R
    phi0 = pi_over_2 - lat_0
    the0 = lon_0
    rho_sin_phi0 = rho*np.sin(phi0)
    x0 = rho_sin_phi0*np.cos(the0)
    y0 = rho_sin_phi0*np.sin(the0)
    z0 = rho*np.cos(phi0)

    phi1 = pi_over_2 - lat_1
    the1 = lon_1
    rho_sin_phi1 = rho*np.sin(phi1)
    x1 = rho_sin_phi1*np.cos(the1)
    y1 = rho_sin_phi1*np.sin(the1)
    z1 = rho*np.cos(phi1)

    dp = x0*x1 + y0*y1 + z0*z1
    sig = np.arccos(dp/(rho*rho))

    return R*sig

def great_circle_distance_1(deg_lat_0, deg_lon_0, deg_lat_1, deg_lon_1, R=Re):

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

great_circle_distance = great_circle_distance_1





class teca_jet_stream_sinuosity(teca_python_algorithm):
    """ Calculate jet stream sinuosity for northern and
    southern hemisphere. Takes as input a table containing
    the topological spine of the jet stream """
    def __init__(self):
        self.label_variable = 'labels'
        self.scalar_variable = None
        self.out_file = 'skel_'
        self.num_ghosts = 16
        self.bounds = None
        self.level = 0.0
        self.plot = False
        self.interact = False
        self.dpi = 100
        self.verbose = False
        self.tex = None
        if get_teca_has_data():
            tex_file = '%s/earthmap4kgy.png'%(get_teca_data_root())
            self.tex = plt.imread(tex_file)

    def set_num_ghosts(self, n):
        """
        Set the maximum length of each segement in the skeleton.
        Lower values result in a higher resolution output.
        """
        self.num_ghosts = n

    def set_label_variable(self, var):
        """
        Set the name of the connected commponent labels
        """
        self.label_variable = var

    def set_scalar_variable(self, var):
        """
        Set the name of a scalar field to plot (optional)
        """
        self.scalar_variable = var

    def set_out_file(self, var):
        """
        Set the name of a scalar field to plot (optional)
        """
        self.out_file = var

    def set_bounds(self, bounds):
        """
        Set the lat lon bounding box that jet stream is detected over
        should be a list with the following order:
            [lon_0, lon_1, lat_0, lat_1]
        """
        self.bounds = bounds

    def set_level(self, level):
        """
        Set the pressure level where the jet stream is detected
        """
        self.level = level

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

            table_out.declare_columns(['step','time','gid', 'hemisphere', \
                'lat_0', 'lon_0', 'lat_1', 'lon_1', 'arc_len', 'direct_len', \
                'sinuosity'], ['l', 'd', 'l', 'i', 'd', 'd', 'd', 'd', 'd', \
                'd', 'd'])

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

                # the start and end point for each hemisphere and accumulate
                # the arc length as we go
                lat_s_nh = sys.float_info.max
                lon_s_nh = sys.float_info.max
                lat_e_nh = sys.float_info.min
                lon_e_nh = sys.float_info.min

                lat_s_sh = sys.float_info.max
                lon_s_sh = sys.float_info.max
                lat_e_sh = sys.float_info.min
                lon_e_sh = sys.float_info.min

                arc_len_nh = 0.
                arc_len_sh = 0.

                # work feature by feature
                comp_0 = np.min(comp_i)
                comp_1 = np.max(comp_i)

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
                        arc_len += great_circle_distance(lat_ij[k], lon_ij[k], \
                             lat_ij[k+1], lon_ij[k+1])

                    # update start, end and arc length for either souther or northern
                    # hemisphere
                    lat_s = lat_ij[0]
                    lon_s = lon_ij[0]

                    lat_e = lat_ij[-1]
                    lon_e = lon_ij[-1]

                    # find most easterly/westerly points, ties are broken by taking
                    # further northern/southern lat in the northern hemisphere and further
                    # southern/northern in the southern hemisphere.
                    nh = True if lat_s > 0. else False
                    if nh:
                        if (lon_s < lon_s_nh) or (np.isclose(lon_s, lon_s_nh) and \
                            (great_circle_distance(lat_s, lon_s, lat_e_nh, lon_e_nh) > \
                            great_circle_distance(lat_s_nh, lon_s_nh, lat_e_nh, lon_e_nh))):
                            lat_s_nh = lat_s
                            lon_s_nh = lon_s
                        if (lon_e > lon_e_nh) or (np.isclose(lon_e, lon_e_nh) and \
                            (great_circle_distance(lat_s_nh, lon_s_nh, lat_e, lon_e) > \
                            great_circle_distance(lat_s_nh, lon_s_nh, lat_e_nh, lon_e_nh))):
                            lat_e_nh = lat_e
                            lon_e_nh = lon_e

                        arc_len_nh += arc_len
                    else:
                        if (lon_s < lon_s_sh) or (np.isclose(lon_s, lon_s_sh) and \
                            (great_circle_distance(lat_s, lon_s, lat_e_sh, lon_e_sh) > \
                            great_circle_distance(lat_s_sh, lon_s_sh, lat_e_sh, lon_e_sh))):
                            lat_s_sh = lat_s
                            lon_s_sh = lon_s
                        if (lon_e > lon_e_sh) or (np.isclose(lon_e, lon_e_sh) and \
                            (great_circle_distance(lat_s_sh, lon_s_sh, lat_e, lon_e) > \
                            great_circle_distance(lat_s_sh, lon_s_sh, lat_e_sh, lon_e_sh))):
                            lat_e_sh = lat_e
                            lon_e_sh = lon_e

                        arc_len_sh += arc_len

                    if self.plot:
                        plt.plot(lon_ij, lat_ij, 'g' if nh else 'b', linewidth=2)


                # norther hemishpere
                gid_i = 1000000*step_i + gid

                if arc_len_nh > 0.:

                    direct_len_nh = great_circle_distance(lat_s_nh, lon_s_nh, lat_e_nh, lon_e_nh)
                    sinuosity_nh = arc_len_nh/direct_len_nh

                    table_out << step_i << time_i << gid_i << 0 << lat_s_nh << lon_s_nh \
                        << lat_e_nh << lat_e_nh << arc_len_nh << direct_len_nh << sinuosity_nh

                    #sys.stderr.write('s= %g, %g   e= %g, %g  d=%g\n'%(lat_s_nh, lon_s_nh, lat_e_nh, lon_e_nh, great_circle_distance(-lat_s_nh, lon_s_nh, -lat_e_nh, lon_e_nh, Re)))
                    #sys.stderr.write('NH, step=%d arc_len=%g direct_len=%g sinuosity=%g\n'%(step_i, arc_len_nh, direct_len_nh, sinuosity_nh))

                    gid += 1

                # suthern hemisphere
                if arc_len_sh > 0.:
                    gid_i = 1000000*step_i + gid

                    direct_len_sh = great_circle_distance(lat_s_sh, lon_s_sh, lat_e_sh, lon_e_sh)
                    sinuosity_sh = arc_len_sh/direct_len_sh

                    #sys.stderr.write('s= %g, %g   e= %g, %g  d=%g\n'%(lat_s_sh, lon_s_sh, lat_e_sh, lon_e_sh, great_circle_distance(lat_s_sh, lon_s_sh, lat_e_sh, lon_e_sh, Re)))
                    #sys.stderr.write('SH, step=%d arc_len=%g direct_len=%g sinuosity=%g\n'%(step_i, arc_len_sh, direct_len_sh, sinuosity_sh))

                    table_out << step_i << time_i << gid_i << 1 << lat_s_sh << lon_s_sh \
                        << lat_e_sh << lat_e_sh << arc_len_sh << direct_len_sh << sinuosity_sh

                    gid += 1

                # plot for the tutorial/demo
                if self.plot:
                    if arc_len_nh > 0.:
                        plt.plot([lon_s_nh, lon_e_nh], [lat_s_nh, lat_e_nh], 'y--o', \
                            linewidth=2, mec='y', mfc=(0.,0.,0.,0.), mew=2)

                    if arc_len_sh > 0.:
                        plt.plot([lon_s_sh, lon_e_sh], [lat_s_sh, lat_e_sh], 'c--o', \
                            linewidth=2, mec='c', mfc=(0.,0.,0.,0.), mew=2)

                    plt.axis('equal')
                    plt.grid(True)
                    #plt.xlim(min(lon_s_nh, lon_s_sh), max(lon_e_nh, lon_e_sh))
                    #plt.ylim(min(lat_s_nh, lat_s_sh), max(lat_e_nh, lat_e_sh))
                    plt.title('Sinuosity step %d'%(step_i))
                    plt.xlabel('deg lon')
                    plt.ylabel('deg lat')
                    if  self.interact:
                        plt.show()
                    plt.savefig('%s_sinuosity_%06d.png'%(self.out_file, step_i), dpi=self.dpi)

            return table_out
        return execute


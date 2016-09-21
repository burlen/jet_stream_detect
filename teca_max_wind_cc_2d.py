from teca import *
import numpy as np
import sys

class teca_max_wind_cc_2d:
    """
    A class for locating the jet stream using the maximum
    value of wind speed in a plane at each longitude.
    """
    @staticmethod
    def New():
        return teca_max_wind_cc_2d()

    def __init__(self):
        self.bounds = [0.0, 360.0, -90.0, 90.0]
        self.level = 0.0
        self.wind_speed_variable = 'wind_speed'
        self.label_variable = 'labels'
        self.plot = False
        self.interact = False
        self.dpi = 100
        self.impl = teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_request_callback( \
            teca_max_wind_cc_2d.get_request_callback(self))
        self.impl.set_execute_callback( \
            teca_max_wind_cc_2d.get_execute_callback(self))

    def __str__(self):
        return 'bounds=%s, level=%f, wind_speed_variable=%s'%( \
            str(self.bounds), self.level, self.wind_speed_variable)

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

    def set_wind_speed_variable(self, wind_var):
        """
        Set the name of the wind speed variable
        """
        self.wind_speed_variable = wind_var

    def set_label_variable(self, wind_var):
        """
        Set the name of the wind speed variable
        """
        self.label_variable = wind_var

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

    def set_input_connection(self, obj):
        """
        set the input
        """
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        """
        get the output
        """
        return self.impl.get_output_port()

    def update(self):
        """
        execute the pipeline from this algorithm up.
        """
        self.impl.update()

    @staticmethod
    def get_request_callback(state):
        """
        returns the function that implements the request
        phase of the TECA pipeline.

        wind_speed_var_name - name of variable containing wind speed
        extent - describes the subset of the data to load
        """
        def request(port, md_in, req_in):
            req = teca_metadata(req_in)
            req['arrays'] = [state.wind_speed_variable]
            req['bounds'] = state.bounds + [state.level]*2
            return [req]
        return request

    @staticmethod
    def get_execute_callback(state):
        """
        returns the function that implements the execute
        phase of the TECA pipeline.

        wind_speed_var_name - name of variable containing wind speed
        """
        def execute(port, data_in, req):
            sys.stderr.write('.')

            # set up for plotting
            if state.plot:
                import matplotlib.pyplot as plt
                import matplotlib.patches as plt_mp
                import matplotlib.image as plt_img

                plt.rcParams['figure.max_open_warning'] = 0

            # get the input as a mesh
            mesh = as_teca_cartesian_mesh(data_in[0])

            # prep the output
            table = teca_table.New()
            table.copy_metadata(mesh)

            table.declare_columns(['step','time','avg_lat', \
                'max_wind_speed'], ['ul','d','d','d'])

            # get metadata
            step = mesh.get_time_step()
            time = mesh.get_time()

            extent = mesh.get_extent()

            # get the dimensions of the data
            nlon = extent[1] - extent[0] + 1
            nlat = extent[3] - extent[2] + 1

            # get the coordinate arrays
            lon = mesh.get_x_coordinates().as_array()
            lat = mesh.get_y_coordinates().as_array()
            lev = mesh.get_z_coordinates().as_array()

            # get the wind speed values as an numpy array
            wind = mesh.get_point_arrays().get( \
                    state.wind_speed_variable).as_array()

            wind = wind.reshape([nlat, nlon])

            # get the labels
            labels = mesh.get_point_arrays().get( \
                state.label_variable).as_array()

            imlabels = labels.reshape([nlat, nlon])

            # look at each lable and see if it spans the
            # extent if it does it is a jet stream candidate
            ulabels = sorted(set(labels))
            ulabels.pop(0)

            n_candidates = 0
            candidate_label = []
            candidate_ids = []
            candidate_mask = []

            for l in ulabels:
                ids = np.where(imlabels == l)

                mini = np.min(ids[1])
                maxi = np.max(ids[1])

                if (mini == 0) and (maxi == (nlon-1)):
                    candidate_label.append(l)
                    candidate_ids.append(ids)

                    mask = np.zeros([nlat, nlon])
                    mask[ids] = 1.0
                    candidate_mask.append(mask)

                    n_candidates += 1

            fig = None
            imext = []
            if state.plot:
                # load the wind field
                imext = [lon[0], lon[-1], lat[0], lat[-1]]
                fig = plt.figure(figsize=(6.4, 4.8))
                plt.imshow(wind, aspect='auto', extent=imext, origin='lower')
                plt.hold(True)
                plt.xlabel('deg lon')
                plt.ylabel('deg lat')
                plt.title('max wind cc 2d\nn_candidates=%d step=%d'%(n_candidates, step))

            i = 0
            while i < n_candidates:
                # examine only wind in this candidate
                candidate_wind = wind * candidate_mask[i]
                l = candidate_label[i]

                # for each lon find lat where max wind occurs
                lat_ids = np.argmax(candidate_wind, axis=0)
                avg_lat = np.average(lat[lat_ids])
                max_id = np.unravel_index(candidate_wind.argmax(), wind.shape)
                max_val = np.max(candidate_wind[lat_ids, np.arange(nlon)])

                if state.plot:
                    plt.plot(lon, lat[lat_ids], '--', color='#ff00ba', linewidth=2)

                    plt.contour(candidate_mask[i], colors='w', \
                        extent=imext, origin='lower')

                table << mesh.get_time_step() << mesh.get_time() \
                    << float(avg_lat) << float(max_val)

                i += 1

            if state.plot:
                plt.savefig('max_wind_cc_2d_%06d.png'%(step), dpi=state.dpi)
                if state.interact:
                    plt.show()
                plt.close(fig)

            return table
        return execute

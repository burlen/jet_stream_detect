from teca import *
import numpy as np
import sys

class teca_max_wind_loc_2d(teca_python_algorithm):
    """
    A class for locating the jet stream using the maximum
    value of wind speed in a plane at each longitude.
    """

    def __init__(self):
        self.bounds = [0.0, 360.0, -90.0, 90.0]
        self.level = 0.0
        self.wind_speed_variable = 'wind_speed'
        self.plot = False
        self.interact = False
        self.dpi = 100

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

    def get_execute_callback(state):
	    """
	    returns the function that implements the execute
	    phase of the TECA pipeline.

	    wind_speed_var_name - name of variable containing wind speed
	    """
	    def execute(port, data_in, req):
                sys.stderr.write('.')
	        # get the input as a mesh
	        mesh = as_teca_cartesian_mesh(data_in[0])

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

	        # for each lon find lat where max wind occurs
	        lat_ids = np.argmax(wind, axis=0)
	        avg_lat = np.average(lat[lat_ids])
                max_id = np.unravel_index(wind.argmax(), wind.shape)
	        max_val = np.max(wind[lat_ids, np.arange(nlon)])

                if state.plot:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as plt_mp
                    import matplotlib.image as plt_img

                    plt.rcParams['figure.max_open_warning'] = 0

                    fig = plt.figure(figsize=(6.4, 4.8))
                    plt.imshow(wind, aspect='auto', \
                        extent=[lon[0], lon[-1], lat[0], lat[-1]], origin='lower')
	            plt.plot(lon, lat[lat_ids], '--', color='#ff00ba', linewidth=2)
                    plt.plot(lon[max_id[1]], lat[max_id[0]], 'gx')
                    plt.plot([lon[0], lon[-1]], [avg_lat]*2, 'k--')
                    plt.xlabel('deg lon')
                    plt.ylabel('deg lat')
                    plt.title('jet stream max wind method\n step %d'%(step))
                    plt.savefig('max_wind_loc_2d_%06d.png'%(step), dpi=state.dpi)
                    if state.interact:
                        plt.show()
                    plt.close(fig)

	        # put it into a table
	        table = teca_table.New()
	        table.copy_metadata(mesh)

	        table.declare_columns(['step','time','avg_lat', \
	            'max_wind_speed'], ['ul','d','d','d'])

	        table << mesh.get_time_step() << mesh.get_time() \
	            << float(avg_lat) << float(max_val)

	        return table
	    return execute

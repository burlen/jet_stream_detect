from teca import *
import numpy as np
from skimage.morphology import medial_axis as ski_medial_axis
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as pltcm
import sys



class node:
    def __init__(self):
        self.next = []
        self.prev = []




class cstruct:
    """ a class that behaves like a C structure """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        return str(self.__dict__)

class teca_topological_skeleton(teca_python_algorithm):
    """
    computes the topological skeletons of features defined by
    a set of integer labels in [1 ... N]
    """
    def __init__(self):
        self.label_variable = 'labels'
        self.scalar_variable = None
        self.out_file = 'skel_'
        self.max_segment_length = 16
        self.bounds = None
        self.level = 0.0
        self.plot = False
        self.interact = False
        self.dpi = 100
        self.verbose = False

    def set_max_segment_length(self, n):
        """
        Set the maximum length of each segement in the skeleton.
        Lower values result in a higher resolution output.
        """
        self.max_segment_length = n

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

    def get_report_callback_tmp(self):
        """
        returns the function that implements the request
        phase of the TECA pipeline.
        """
        def report(port, md_in):
            # get the bounds of the data if the user hasn't set any
            md = md_in[0]
            if self.bounds is None:
                self.bounds = md['bounds'][:4]
            return md
        return report

    def get_request_callback(self):
        """
        returns the function that implements the request
        phase of the TECA pipeline.
        """
        def request(port, md_in, req_in):
            # request the bounds of the regeion
            if self.bounds is None:
                md = md_in[0]
                bounds = list(md['bounds'][:4])
            else:
                bounds = self.bounds
            req = teca_metadata(req_in)
            req['bounds'] = bounds + [self.level]*2
            return [req]
        return request

    def get_execute_callback(self):
        """
        returns the function that implements the execute
        phase of the TECA pipeline.
        """
        def execute(port, data_in, req):
            sys.stderr.write('.')
            sys.stderr.flush()

            # set up for plotting
            if self.plot:
                import matplotlib.pyplot as plt
                import matplotlib.patches as plt_mp
                import matplotlib.image as plt_img

                plt.rcParams['figure.max_open_warning'] = 0

            # get the input as a mesh
            mesh = as_teca_cartesian_mesh(data_in[0])

            md = mesh.get_metadata()
            print(md)

            comp_ids = md['component_ids']

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

            # get the scalar field
            scalar = None
            if self.scalar_variable is not None:
                scalar = mesh.get_point_arrays().get(self.scalar_variable).as_array()
                scalar = scalar.reshape([nlat, nlon])

            # get the labels
            labels = mesh.get_point_arrays().get( \
                self.label_variable).as_array()

            imlabels = labels.reshape([nlat, nlon])

            # construct the skeletonization
            segs = self.skeletonize(imlabels, self.max_segment_length)

            # prep the output
            table = teca_table.New()
            table.copy_metadata(mesh)

            table.declare_columns( \
                ['step','time','gid', 'feature', 'p0_lat','p0_lon','p1_lat','p1_lon','num_paths_out'], \
                ['l',    'd',  'l',    'i',      'd',     'd',     'd',     'd',     'i'])

            i = 0
            for seg in segs:

                p0 = seg.start
                p1 = seg.end

                if self.verbose:
                    print('%d, %d, (%s -> %s), ((%g, %g) -> (%g, %g)), %d, %d'%( \
                        seg.gid, seg.sid, str(p0), str(p1), lat[p0[1]], lon[p0[0]], \
                        lat[p1[1]], lon[p1[0]], seg.dist, seg.nout))

                table << step << time << step*100000 + seg.gid*10000 + i \
                    << seg.gid << lat[p0[1]] << lon[p0[0]] << lat[p1[1]] << lon[p1[0]] \
                    << seg.nout

                i += 1

            # TODO -- move plotting code into another algorithm??
            fig = None
            imext = []
            if self.plot:
                # load the wind field
                imext = [lon[0], lon[-1], lat[0], lat[-1]]

                fig = plt.figure(figsize=(6.4, 4.8))

                cmap = plt.get_cmap('autumn')
                cmap.set_under('black')

                # color by scalar or connected component
                if scalar is None:
                    plt.imshow(imlabels, cmap=cmap, aspect='auto', extent=imext, origin='lower', vmin=0.5)
                else:
                    plt.imshow(scalar, aspect='auto', extent=imext, origin='lower', cmap=plt.get_cmap('gray'))

                #plt.imshow(imlabels, aspect='auto', extent=imext, origin='lower')
                plt.xlabel('deg lon')
                plt.ylabel('deg lat')
                plt.title('skeleton step=%d'%(step))

                cmap = pltcm.ScalarMappable(norm=pltcolors.Normalize(vmin=1, vmax=segs[-1].gid), \
                     cmap=plt.get_cmap('cool'))

                for seg in segs:
                    x = [lon[seg.start[0]], lon[seg.end[0]]]
                    y = [lat[seg.start[1]], lat[seg.end[1]]]
                    c = cmap.to_rgba(seg.gid)
                    plt.plot(x,y, '-', color=c, linewidth=2)

                plt.contour(imlabels, [0.125, 1.125, 2.125, 3.125, 4.125, 5.125, 6.125], colors='w', \
                    extent=imext, origin='lower')

                plt.savefig('%s_%06d.png'%(self.out_file, step), dpi=self.dpi)

                if self.interact:
                    plt.show()

                plt.close(fig)

            return table
        return execute

    @staticmethod
    def get_paths_out(segend, nx,ny, medax, vis, vis_flag=1):
        # find paths out of this point in the skeleton
        nout = 0
        pout = []
        for ii in [-1, 0, 1]:
            i = segend[0] + ii
            # skip out of bounds look ups
            if (i < 0) or (i >= nx):
                continue
            for jj in [-1, 0, 1]:
                j = segend[1] + jj
                # skip this one if already used, not on skeleton, out of bounds
                if (ii == 0 and jj == 0) or (j < 0) or (j >= ny) \
                    or (vis[j,i] == vis_flag) or (medax[j,i] == 0):
                    continue
                # this point leads out
                nout += 1
                pout.append((i,j))
        return nout,pout

    @staticmethod
    def skeletonize(binseg, maxdist=8):
        # compute the medial axis transform from a binary segmentation
        medax = ski_medial_axis(binseg)

        # mask identifying points we already used
        nx = medax.shape[1]
        ny = medax.shape[0]
        vis = np.zeros((ny, nx), 'i8')

        # get a list of i,j points on the skeleton. these
        # are so called segment candidates
        qq = np.where(medax == 1)
        ii = qq[1]
        jj = qq[0]

        # conmvert these from numpy's format to i,j tuples
        # conveneint here
        cand = []
        n = len(ii)
        q = 0
        while q < n:
            qq = (ii[q], jj[q])
            cand.append(qq)
            q += 1

        # visit each candidate and try to use it to create a
        # segment.
        gid = 0
        segs = []
        while len(cand):

            qq = cand.pop()
            i = qq[0]
            j = qq[1]
            if vis[j,i] == 1:
                continue
            vis[j,i] = 1
            gid += 1

            # start a new segment from this candidate
            # process this and all segments that branch
            # from it
            sid = 0
            asegs = [cstruct(start=qq, end=qq, dist=0, gid=gid, sid=0, nout=0)]
            while len(asegs):
                aseg = asegs.pop()

                while True:
                    # find paths out of this point in the skeleton
                    nout,pout = teca_topological_skeleton.get_paths_out(aseg.end, nx, ny, medax, vis, 1)

                    # save the number of paths out of the end point
                    aseg.nout = nout

                    if nout == 0:
                        # end of the segment, pass to output
                        # and goto processing the next segment
                        if aseg.dist > 0:
                            segs.append(aseg)
                        break
                    elif aseg.dist < maxdist and nout == 1:
                        # extend this segment by one point
                        p = pout.pop()
                        aseg.end = p
                        aseg.dist += 1
                        vis[p[1],p[0]] = 1
                    else:
                        # start a new segment for each point
                        for p in pout:
                            sid += 1
                            i = p[0]
                            j = p[1]
                            if vis[j,i] == 0:
                                vis[j,i] = 1
                                asegs.append(cstruct(start=aseg.end, end=p, \
                                    dist=1, gid=gid, sid=sid, nout=0))
                        # end of the segment, pass to output
                        # and goto processing the next segment
                        if aseg.dist > 0:
                            segs.append(aseg)
                        break

        return segs


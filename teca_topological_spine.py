from teca import *
from copy import deepcopy
import numpy as np
from skimage.morphology import medial_axis as ski_medial_axis
from skimage.measure import label as ski_label
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as pltcm
import sys
import math

Re = 6371.0088 #km
pi = 3.14159265358979
pi_over_180 = pi/180.
pi_over_2 = pi/2.

def great_circle_distance_0(deg_lat_0, deg_lon_0, deg_lat_1, deg_lon_1, R):

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

def great_circle_distance_1(deg_lat_0, deg_lon_0, deg_lat_1, deg_lon_1, R):

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



class priority_queue(object):
    """ A generic priority queue

        Elements should implement __lt__ and have an idx member
        that is updated with the element's current position in the
        internal queue structure
    """

    def __init__(self):
        self.n = 0
        self.data = [None]

    def __repr__(self):
        levs = math.log(self.n,2)
        lev = 0
        i = 1
        ostr = ''
        while i < self.n:
            nsp = int(((levs - lev + 1)**2)//2)
            ostr += ' '*nsp
            n = 2**lev
            m = min(self.n, i + n)
            while i < m:
                ostr += '%s'%(str(self.data[i])) + ' '*max(nsp,1)
                i += 1
            ostr += '\n'
            lev += 1
        return ostr

    def __len__(self):
        return self.n

    def exchg(self, j, k):
        tmp = self.data[j]
        self.data[j] = self.data[k]
        self.data[k] = tmp
        self.data[j].idx = j
        self.data[k].idx = k

    def comp(self, j, k):
        return self.data[j] < self.data[k]

    def swim(self, k):
        while (k > 1) and self.comp(k, k//2):
            self.exchg(k//2, k)
            k = k//2

    def sink(self, k):
        while (2*k <= self.n):
            j = 2*k;
            if (j < self.n) and self.comp(j+1, j):
                j += 1
            if (not self.comp(j, k)):
                break;
            self.exchg(j, k)
            k = j

    def push(self, val):
        self.n += 1
        val.idx = self.n
        self.data.append(val)
        self.swim(self.n)

    def pop(self):
        val = self.data[1]
        self.exchg(1, self.n)
        del self.data[self.n]
        self.n -= 1
        self.sink(1)
        return val

    def fix(self, k):
        if (k > 1) and self.comp(k, k//2):
            self.swim(k)
        else:
            self.swim(k)

    def size(self):
        return self.n


class nodes(object):
    """ container for manipulating medial axis as a graph
        it is assumed that fetaures are simply connected and
        that no feature crosses a periodic boundary
    """
    def __init__(self, scalar, lab, lat, lon):
        """
        """
        self.ext = [0, lab.shape[1]-1, 0, lab.shape[1]-1]
        self.lat = lat
        self.lon = lon
        self.vis = np.zeros(lab.shape, 'i8')
        self.scalar = scalar
        self.lab = lab
        self.nodes = []
        # construct graph nodes
        locs = np.where(lab > 0)
        i = locs[1]
        j = locs[0]
        q = 0
        while q < len(i):
            n = node(i[q],j[q])
            self.nodes.append(n)
            q += 1
        # add slots for begining and end points
        beg = node(sys.maxsize, sys.maxsize)
        end = node(-sys.maxsize, sys.maxsize)

    def path(self):

        p = []

        ok,e = self.trace_path()

        if not ok:

            raise RuntimeError( \
                'failed to locate a path (%d,%d) -> (%d,%d)'%( \
                beg.i,beg.j,end.i,end.j))

        n = e
        while True:
            p.insert(0, (n.i, n.j))
            if n.parent is None:
                break
            n = n.parent

        return p

    def trace_path(self):

        # get the start and end points
        b,e = self.get_targets()
        b.dist = 0.0

        self.beg = b
        self.end = e

        #print('beg=%s'%(str(b)))
        #print('end=%s'%(str(e)))

        # initialize the heap, and random access structure
        ran = {}
        pq = priority_queue()
        for n in self.nodes:
            pq.push(n)
            ran[(n.i,n.j)] = n

        # initialize the visited flags
        vis = np.zeros(self.lab.shape, 'i8')

        # visit each node until end is reached
        while len(pq):

            # get the node closest to the beg
            n = pq.pop()

            # check for successful termination
            if n.i == e.i and n.j == e.j:
                return 1,e

            # check for failure, i.e. no path
            if n.dist == sys.float_info.max:
                raise RuntimeError('No path - neighbors are infinitely far')
                return 0,e

            # mark this one visited
            vis[n.j,n.i] = 1

            # distance to the neighbors
            d = n.dist + 1

            # visit neighbors
            i0 = max(n.i-1, self.ext[0])
            i1 = min(self.ext[1], n.i+1)

            j0 = max(n.j-1, self.ext[2])
            j1 = min(self.ext[3], n.j+1)

            r = j0
            while r <= j1:
                q = i0
                while q <= i1:

                    # skip points on the path that we already visited
                    if self.lab[r,q] > 0 and vis[r,q] < 1:

                        # put q,r in the queue to visit next
                        pos = (q,r)
                        m = ran[pos]

                        # would the path from n to m be shorter than the current path to m?
                        if d < m.dist:
                            # update m
                            m.dist = d
                            m.parent = n
                            pq.fix(m.idx)

                    q += 1
                r += 1

        for n in self.nodes:
            sys.stderr.write('n = %s\n'%(str(n)))
        sys.stderr.write('Should not be here ------------------- b=%s e=%s\n'%(str(b), str(e)))
        raise RuntimeError('Should not be here!')
        return 0,e

    def get_targets(self):

        # input feature must not cross a periodic boundary.  i.e. all points in
        # the feature are directly reachable from all other points
        cand = {}
        for c in self.nodes:
            cand[(c.i,c.j)] = c

        vis = np.zeros(self.lab.shape, 'i8')

        beg = node(sys.maxsize, sys.maxsize)
        end = node(-sys.maxsize, sys.maxsize)

        while len(cand):

            # visit locations reachable not crossing a periodic boundary from
            # this node
            k = next(iter(cand))
            work = [cand[k]]

            while len(work):

                # grab a node
                n = work.pop()

                if vis[n.j,n.i] > 0:
                    # hanlde cycles
                    continue

                # remove it from the list of candidates
                del cand[(n.i,n.j)]
                vis[n.j,n.i] = 1

                # update beg, beg is the furthest east. in the case of ties
                # larger value of scalar field breaks it
                if (beg.i > n.i) or ((beg.i == n.i) and (self.scalar[n.j, n.i] > self.scalar[beg.j, beg.i])):
                    beg = n

                # update end. end is the furthest west. in the case of ties
                # larger value of scalar field breaks it
                if  (end.i < n.i) or ((end.i == n.i) and (self.scalar[n.j, n.i] > self.scalar[end.j, end.i])):
                    end = n

                # add un-visited neighbors to the queue
                i0 = max(n.i-1, self.ext[0])
                i1 = min(self.ext[1], n.i+1)

                j0 = max(n.j-1, self.ext[0])
                j1 = min(self.ext[1], n.j+1)

                r = j0
                while r <= j1:
                    q = i0
                    while q <= i1:

                        # skip points on the path that we already visited
                        if self.lab[r,q] > 0 and vis[r,q] < 1:

                            # put q,r in the queue to visit next
                            pos = (q,r)
                            n = cand[pos]
                            work.append(n)

                        q += 1
                    r += 1

            # verify that assumption has been satisfied
            if len(cand):
                plt.imshow(vis)
                plt.plot(beg.i, beg.j, 'ro')
                plt.plot(end.i, end.j, 'bo')
                for k,c in cand.items():
                    plt.plot(c.i, c.j, 'gx')
                plt.title('runtime error!')
                plt.show()
                raise RuntimeError('%d points are not directly reachable'%(len(cand)))

        return beg,end


class node(object):

    gid = 0
    @classmethod
    def next_gid(cls):
        i = cls.gid
        cls.gid += 1
        return i

    def __init__(self, i,j):
        self.gid = self.next_gid()
        self.idx = 0 # used to reach back into priority_queue

        self.parent = None
        self.dist = sys.float_info.max

        self.i = i
        self.j = j

    def __lt__(self, other):
        return self.dist < other.dist

    def __repr__(self):
        return '%d, %d, (%d, %d), %g'%(self.gid, self.idx, self.i, self.j, self.dist)










































class cstruct:
    """ a class that behaves like a C structure """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        return str(self.__dict__)

class teca_topological_spine(teca_python_algorithm):
    """
    computes the topological skeletons of features defined by
    a set of integer labels in [1 ... N]
    """
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

            # get the input as a mesh
            mesh = as_teca_cartesian_mesh(data_in[0])
            md = mesh.get_metadata()

            # get metadata
            step = mesh.get_time_step()
            time = mesh.get_time()

            extent = mesh.get_extent()
            whole_extent = mesh.get_whole_extent()

            per_bc = True if (mesh.get_periodic_in_x() and \
                (extent[0] == whole_extent[0]) and \
                    (extent[1] == whole_extent[1])) else False

            # get the dimensions of the data
            nlon = extent[1] - extent[0] + 1
            nlat = extent[3] - extent[2] + 1
            nlev = extent[5] - extent[4] + 1

            # get the coordinate arrays
            lon = mesh.get_x_coordinates().as_array()
            lat = mesh.get_y_coordinates().as_array()
            lev = mesh.get_z_coordinates().as_array()

            # get the connected component labels, make it numpy 2D
            labels = mesh.get_point_arrays().get(self.label_variable).as_array()
            imlabels = labels.reshape([nlat, nlon])

            # get the scalar field (used to resolve ties)
            scalar = mesh.get_point_arrays().get(self.scalar_variable).as_array()
            scalar = scalar.reshape([nlat, nlon])

            # add ghost zones to the input to prevent artifacts at the periodic boundary
            # clip the path and adjust the indices after. Ghost zones are generated
            # either by copy across periodic BC or extend by duplication of values at
            # the edge
            ng = min(nlon/2, self.num_ghosts)
            gimlabels = self.add_ghost_zones(nlat, nlon, ng, per_bc, imlabels)
            gscalar = self.add_ghost_zones(nlat, nlon, ng, per_bc, scalar)

            # do the medial axis segmentation on the region with ghost zones
            medax = ski_medial_axis(gimlabels)

            # because we can't yet handle features that cross a periodic boundary
            # re-compute connected component labeling w/o the periodic bc
            if per_bc:
                imrelab, numlabs = ski_label(gimlabels, return_num=True)
                comp_ids = range(1, numlabs+1)
            else:
                imrelab = gimlabels
                comp_ids = md['component_ids_masked']

            # get the component ids and work component by component to calculate
            # the spine
            spines = []
            for comp_id in comp_ids:

                if comp_id == 0:
                    continue

                # mask out all but the current component
                mmedax = np.where(np.logical_and((imrelab == comp_id), (medax > 0)), 1, 0)

                #import matplotlib.pyplot as plt
                #plt.imshow(np.where(imrelab == comp_id, 1, 0))
                #plt.title('input labels %d '%(comp_id))
                #plt.show()
                #plt.imshow(mmedax)
                #plt.title('input mmedax')
                #plt.show()

                # calculate the spine
                lgraph = nodes(gscalar, mmedax, lat, lon)
                path = lgraph.path()

                # split i,j tuples into i and j arrays
                path_i = np.array([i for i,j in path], 'int32')
                path_j = np.array([j for i,j in path], 'int32')

                # indicies that remove ghost zones
                valid_i = np.where(np.logical_and(path_i >= ng, path_i < ng+nlon))
                if len(valid_i[0]) == 0:
                    # skip, this path is only in the ghost cells
                    continue

                # copy and shift lom indices back into non-ghost index space
                path_i = path_i[valid_i] - ng
                path_j = path_j[valid_i]

                # add to the result
                spines.append((comp_id, (path_i, path_j)))

                #import matplotlib.pyplot as plt
                #cmap = plt.get_cmap('cool')
                #cmap.set_under((0.,0.,0.,0.))
                #cmap.set_over((0.,0.,1.,1.))
                #plt.imshow(mmedax,cmap=cmap,vmin=0.25,vmax=0.75)
                #plt.contour(np.where(imrelab == comp_id, 1, 0), [0.125], colors='k')
                #plt.plot([ng, ng], [0, nlat], 'g')
                #plt.plot([nlon+ng, nlon+ng], [0, nlat], 'g')
                #plt.plot(path_i + ng, path_j, 'r', linewidth=2)
                #plt.title('medax comp_id=%d'%(comp_id))
                #plt.show()

            if self.plot:
                # make figures for the tutorial/demo
                imlabelsu = mesh.get_point_arrays().get('labels').as_array()
                imlabelsu = imlabelsu.reshape([nlat, nlon])

                self.plot_figs(step, lat, lon, scalar, imlabels, \
                    imlabelsu, imrelab[:,ng:nlon+ng], comp_ids, \
                    medax[:,ng:nlon+ng], [p for i,p in spines])

            # build the output table
            table = teca_table.New()
            table.copy_metadata(mesh)

            # set up the columns
            table.declare_columns( \
                ['step','time','gid', 'cid', 'lat', 'lon'], \
                ['l',    'd',  'l',    'i',    'd', 'd'])

            # copy each row into the table
            for comp_id, path in spines:

                path_len = len(path[0])
                cid = comp_id
                gid = 100000*step + cid
                path_lon = lon[path[0]]
                path_lat = lat[path[1]]

                i = 0
                while i < path_len:
                    table << step << time << gid << cid << path_lat[i] << path_lon[i]
                    i += 1

            return table
        return execute

    def add_ghost_zones(self, nlat, nlon, ng, per_bc, data):
        """ add ghost zones to the input to prevent artifacts at the periodic boundary
        clip the path and adjust the indices after. Ghost zones are generated
        either by copy across periodic BC or mirror values at the edge """
        gdata = np.empty((nlat, nlon+2*ng), data.dtype)
        gdata[:,ng:nlon+ng:1] = data
        if per_bc:
            gdata[:,0:ng:1] = data[:,-ng:nlon:1]
            gdata[:,-ng:nlon+2*ng:1] = data[:,0:ng:1]
        else:
            gdata[:,0:ng:1] = (data[:,0:ng:1])[:,::-1]
            gdata[:,-ng:nlon+2*ng:1] = (data[:,-ng:nlon:1])[:,::-1]
        return gdata

    def plot_tex(self, lat, lon):
        """ plot the blue marble texture in the background """
        if self.tex is not None:
            imext = [lon[0], lon[-1], lat[0], lat[-1]]
            i0 = int(self.tex.shape[1]/360.0*imext[0])
            i1 = int(self.tex.shape[1]/360.0*imext[1])
            j0 = int(-((imext[3] + 90.0)/180.0 - 1.0)*self.tex.shape[0])
            j1 = int(-((imext[2] + 90.0)/180.0 - 1.0)*self.tex.shape[0])
            plt.imshow(self.tex[j0:j1, i0:i1], extent=imext, aspect='equal', zorder=1)

    def plot_figs(self, step, lat, lon, scalar, imlabels,
        imlabelsu, imrelab, comp_ids, medax, spines):
        """ make plots for tutorial/presentation """
        # set up for plotting
        import matplotlib.pyplot as plt
        import matplotlib.patches as plt_mp
        import matplotlib.image as plt_img

        comp_alph = 0.75

        plt.rcParams['figure.max_open_warning'] = 0

        imext = [lon[0], lon[-1], lat[0], lat[-1]]

        # plot scalar in
        fig = plt.figure(figsize=(8.0, 4.0))

        self.plot_tex(lat, lon)

        cmap = plt.get_cmap('magma')
        plt.imshow(scalar, aspect='equal', extent=imext, \
            origin='lower', cmap=cmap, zorder=2)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Scalar variable (%s) step=%d'%(self.scalar_variable, step))

        plt.savefig('%s_scalar_variable_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()
        plt.close(fig)

        # plot unfiltered connected components
        fig = plt.figure(figsize=(8.0, 4.0))
        self.plot_tex(lat, lon)

        cmap = plt.get_cmap('cool')
        cmap.set_under((0.,0.,0.,0.))

        plt.imshow(imlabelsu, cmap=cmap, aspect='equal', \
            extent=imext, origin='lower', vmin=0.5, alpha=comp_alph, \
            zorder=2)

        plt.contour(imlabelsu, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Labeled segmentation (unfiltered) step=%d'%(step))

        plt.savefig('%s_labeled_segmentation_unfilt_%06d.png'%( \
            self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()
        plt.close(fig)

        # plot filtered connected components
        fig = plt.figure(figsize=(8.0, 4.0))

        self.plot_tex(lat, lon)

        cmap = plt.get_cmap('cool')
        cmap.set_under((0.,0.,0.,0.))
        plt.imshow(imlabels, cmap=cmap, aspect='equal', \
            extent=imext, origin='lower', vmin=0.5, alpha=comp_alph, \
            zorder=2)

        plt.contour(imlabels, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Labeled segmentation (filtered) step=%d'%(step))

        plt.savefig('%s_labeled_segmentation_filt_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()
        plt.close(fig)


        # plot filtered connected components with medial axis transform
        fig = plt.figure(figsize=(8.0, 4.0))

        self.plot_tex(lat, lon)

        cmap = plt.get_cmap('gray')
        cmap.set_under((0.,0.,0.,0.))
        plt.imshow(np.where(imlabels > 0, 1., 0.), cmap=cmap, aspect='equal', \
            extent=imext, origin='lower', vmin=0.5, alpha=comp_alph, zorder=2)

        plt.contour(imlabels, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        cmap = pltcm.ScalarMappable(norm=pltcolors.Normalize(vmin=1, vmax=np.max(comp_ids)), \
             cmap=plt.get_cmap('cool'))

        for comp_id in comp_ids:

            if comp_id == 0:
                continue

            mapts = np.where(np.logical_and((imrelab == comp_id), (medax > 0)))

            c = cmap.to_rgba(comp_id)
            plt.plot(lon[mapts[1]], lat[mapts[0]], '.', color=c, \
                markersize=3, zorder=3)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Medial axis tranform. step=%d'%(step))

        plt.savefig('%s_medial_axis_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()
        plt.close(fig)


        # plot filtered connected components with medial axis transform and spine
        fig = plt.figure(figsize=(8.0, 4.0))

        self.plot_tex(lat, lon)

        cmap = plt.get_cmap('gray')
        cmap.set_under((0.,0.,0.,0.))
        plt.imshow(np.where(imlabels > 0, 1., 0.), cmap=cmap, aspect='equal', \
                 extent=imext, origin='lower', vmin=0.5, alpha=comp_alph, zorder=2)

        plt.contour(imlabels, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        cmap = pltcm.ScalarMappable(norm=pltcolors.Normalize(vmin=1, vmax=np.max(comp_ids)), \
             cmap=plt.get_cmap('cool'))

        for comp_id in comp_ids:
            mapts = np.where(np.logical_and((imrelab == comp_id), (medax > 0)))

            c = cmap.to_rgba(comp_id)

            plt.plot(lon[mapts[1]], lat[mapts[0]], '.', markersize=2, \
                color='k', alpha=0.035, zorder=3)

        comp_id = 1
        for path  in spines:

            c = cmap.to_rgba(comp_id)
            comp_id += 1

            path_i = lon[path[0]]
            path_j = lat[path[1]]

            plt.plot(path_i, path_j, color=c, linewidth=2, zorder=4)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Medial axis tranform and spine. step=%d'%(step))

        plt.savefig('%s_medial_axis_spine_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()
        plt.close(fig)

        # plot the figure
        fig = plt.figure(figsize=(8.0, 4.0))

        self.plot_tex(lat, lon)


        # color by scalar
        cmap = plt.get_cmap('magma')
        nc = cmap(np.linspace(0, 1, cmap.N))
        a = 2.0*np.linspace(0, 1, cmap.N)
        nc[:,-1] = np.where(a <= 1.0, a, 1.0)
        cmap = pltcolors.ListedColormap(nc)
        plt.imshow(scalar, aspect='equal', extent=imext, \
            origin='lower', cmap=cmap, zorder=2)

        plt.contour(imlabels, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        for path  in spines:

            path_i = lon[path[0]]
            path_j = lat[path[1]]

            plt.plot(path_i, path_j, \
                'g' if path_j[0] > 0. else 'b', linewidth=2, zorder=4)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('NH/SH Jet Stream Spines step=%d'%(step))

        plt.savefig('%s_spine_and_wind_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()

        plt.close(fig)















































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
    def get_shortest_path(cur, end, medax, vis, path):
        # note that this is a recursive implementation and is very slow in Python!
        # this would work well in C++ or one could convert to iterative

        # always add the point to the path
        path_out = deepcopy(path)
        path_out.append(cur)

        # visit each point only once
        vis[cur[1],cur[0]] = 1

        # we've reached the desired end-point, success, send the path back
        if cur == end:
            return 1, path_out

        # find all paths leading out of this point
        nx = medax.shape[1]
        ny = medax.shape[0]

        while 1:

            nout,pout = teca_topological_spine.get_paths_out(cur, nx,ny, medax, vis)

            # if there is exactly one path out, keep marching down that path
            # because recursion in Python is very costly
            if nout != 1:
                break

            cur = pout.pop()
            path_out.append(cur)
            vis[cur[1],cur[0]] = 1

            # we've reached the desired end-point, success, send the path back
            if cur == end:
                return 1, path_out


        # we reached a dead end that was not the desired end-point, fail
        if nout == 0:
            return 0, path_out

        # trace all paths out, keep track of the minimum length reaching the end-point
        len_min = sys.maxsize
        path_min = []

        while len(pout):

            # attempt trace from the new point to the desired end point
            nexpt = pout.pop()
            ok, path_tmp = teca_topological_spine.get_shortest_path( \
                nexpt, end, medax, vis, path_out)

            if ok:
                # path got us to the desired end point
                len_tmp = len(path_tmp)
                if len_tmp < len_min:
                    # it is a shorter path than anything we saw so far
                    len_min = len_tmp
                    path_min = path_tmp

        # no paths to the desired end point were found
        if len(path_min) == 0:
            return 0, path_out

        return 1, path_min

    @staticmethod
    def get_spine(lat, lon, binseg):

        # add ghost zones tot he input to prevent artifacts at the periodic boundary
        nx = binseg.shape[1]
        ny = binseg.shape[0]
        ng = 16

        gbinseg = np.empty((ny, nx+2*ng), binseg.dtype)
        gbinseg[:,ng:nx+ng:1] = binseg
        gbinseg[:,0:ng:1] = binseg[:,-ng:nx:1]
        gbinseg[:,-ng:nx+2*ng:1] = binseg[:,0:ng:1]

        plt.imshow(gbinseg)
        plt.title('binseg w/ ghosts')
        plt.show()

        # do the medial axis segmentation
        gmedax = ski_medial_axis(gbinseg)

        # remove ghost zones
        medax = np.empty((ny,nx), gmedax.dtype)
        medax[:,:] = gmedax[:,ng:nx+ng]

        plt.imshow(medax)
        plt.title('medax w/o ghosts')
        plt.show()

        # determine start and end point for feature
        # these are points with the lowest and highest lon values
        mat = np.where(medax == 1)

        # which medial axis j,i has the lowest and highest i?
        i0 = np.argmin(mat[1])
        i1 = np.argmax(mat[1])

        # get the i of these, that is the i of the start and end point
        ii0 = mat[1][i0]
        ii1 = mat[1][i1]

        # enumerate ties. note ties are largely a consequence of not
        # handling periodic boundaries in the medial axis calc
        cand_i0 = np.where(medax[:,ii0] == 1)[0]
        cand_i1 = np.where(medax[:,ii1] == 1)[0]

        # evaluate only the two closest start/end points
        min_r = sys.maxsize
        jj0 = 0
        jj1 = 0
        q0 = 0
        while q0 < len(cand_i0):
            q1 = 0
            while q1 < len(cand_i1):

                beg = (ii0, cand_i0[q0])
                end = (ii1, cand_i1[q1])

                r = np.sqrt( (end[0] - end[1])**2 + (end[1] - beg[1])**2)
                #r = great_circle_distance(lat[cand_i0[q0]], lon[ii0], lat[cand_i1[q1]], lon[ii1], Re)

                if r < min_r:
                    min_r = r
                    jj0 = cand_i0[q0]
                    jj1 = cand_i1[q1]

                q1 += 1
            q0 += 1


        # find the shortest path between beg and end
        beg = (ii0, jj0)
        end = (ii1, jj1)

        vis = np.zeros(medax.shape, 'i8')

        ok,path = teca_topological_spine.get_shortest_path(beg, end, medax, vis, [])

        #print('beg=%s end=%s len=%d path=%s'%(str(beg), str(end), len(path), str(path)))

        if ok == 0:
            raise RuntimeError('no path between %s and %s'%(str(beg),str(end)))

        return path

    @staticmethod
    def get_skeleton(imlabs, scalars):

        skel = np.zeros(imlabs.shape, 'i8')

        nx = imlabs.shape[1]
        ny = imlabs.shape[0]

        i = 0
        while i < nx:

            active = 0
            jmax = 0
            smax = sys.float_info.min

            j = 0
            while j < ny:

                lab = imlabs[j,i]
                sval = scalars[j,i]

                # check for a change of state
                if active == 0 and lab != 0:
                    # switching from inactive to active, start scanning for
                    # next max
                    active = 1
                    smax = sval
                    jmax = j

                elif active == 1:
                    if lab == 0:
                        # switch from active to inactive, mark point in skeleton
                        # where the current max scalar is
                        active = 0
                        skel[jmax,i] = lab
                    elif sval > smax:
                        smax = sval
                        jmax = j

                j += 1

            i += 1


        return skel




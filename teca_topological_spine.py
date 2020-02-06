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

class teca_topological_spine(teca_python_algorithm):
    """
    computes the topological skeletons of features defined by
    a set of integer labels in [1 ... N]
    """
    def __init__(self):
        self.label_variable = 'labels'
        self.scalar_variable = None
        self.out_file = 'topological_spine'
        self.num_ghosts = 16
        self.bounds = None
        self.level = 0.0
        self.plot = False
        self.interact = False
        self.dpi = 100
        self.verbose = False
        self.tex = None

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
        if self.plot and get_teca_has_data():
            tex_file = '%s/earthmap4k.png'%(get_teca_data_root())
            self.tex = plt.imread(tex_file)

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

            # add ghost zones to the input to prevent artifacts at the periodic
            # boundary clip the path and adjust the indices after. Ghost zones
            # are generated either by copy across periodic BC or extend by
            # duplication of values at the edge
            ng = min(nlon/2, self.num_ghosts)
            gimlabels = self.add_ghost_zones(nlat, nlon, ng, per_bc, imlabels)
            gscalar = self.add_ghost_zones(nlat, nlon, ng, per_bc, scalar)

            # do the medial axis segmentation on the region with ghost zones
            medax = ski_medial_axis(gimlabels)

            # re-compute connected component labeling. 1. features crossing a
            # periodic bc need to be split for the start end point search 2.
            # ghost zones can introduce new component labels. the start, end
            # point search assume simply connected axis
            imrelab, numlabs = ski_label(gimlabels, return_num=True)
            comp_ids = range(1, numlabs+1)

            # get the component ids and work component by component to
            # calculate the spine
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
                try:
                    graph = graph_nodes(gscalar, mmedax, lat, lon)
                    path = graph.shortest_path()
                except:
                    sys.stderr.write('Error detected in step %d\n'%(step))
                    raise

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

            # set up the columns, name and data type
            table.declare_columns( \
                ['step','time','gid', 'comp_id', 'lat', 'lon'], \
                ['l',    'd',  'l',   'i',       'd',   'd'])

            # copy each row into the table
            cid = 0
            for comp_id, path in spines:

                path_len = len(path[0])
                path_lon = lon[path[0]]
                path_lat = lat[path[1]]

                i = 0
                while i < path_len:
                    gid = 1000000000*step + 1000000*cid + i
                    table << step << time << gid << cid << path_lat[i] << path_lon[i]
                    i += 1

                cid += 1

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


        # plot scalar and segmentation
        fig = plt.figure(figsize=(8.0, 4.0))
        self.plot_tex(lat, lon)

        segscalar = np.where(imlabelsu > 0, scalar, imlabelsu)

        cmap = plt.get_cmap('magma')
        cmap.set_under((0.,0.,0.,0.))

        plt.contour(imlabelsu, [0.125], colors='k', \
           extent=imext, origin='lower', zorder=3)

        cmap = plt.get_cmap('magma')
        plt.imshow(segscalar, aspect='equal', extent=imext, \
            origin='lower', cmap=cmap, vmin=0.5, zorder=2)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('Segmented scalar variable (%s) step=%d'%(self.scalar_variable, step))

        plt.savefig('%s_scalar_variable_and_seg_%06d.png'%(self.out_file, step), dpi=self.dpi)

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

        plt.savefig('%s_labeled_segmentation_filt_%06d.png'%( \
            self.out_file, step), dpi=self.dpi)

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

        cmap = pltcm.ScalarMappable(norm=pltcolors.Normalize(vmin=1, \
            vmax=np.max(comp_ids)), cmap=plt.get_cmap('cool'))

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
        segscalar = np.where(imlabels > 0, scalar, imlabels)

        cmap = plt.get_cmap('magma')
        nc = cmap(np.linspace(0, 1, cmap.N))
        a = 2.0*np.linspace(0, 1, cmap.N)
        nc[:,-1] = np.where(a <= 1.0, a, 1.0)
        cmap = pltcolors.ListedColormap(nc)

        plt.imshow(segscalar, aspect='equal', extent=imext, \
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

        # plot just the spine
        fig = plt.figure(figsize=(8.0, 4.0))
        self.plot_tex(lat, lon)

        for path  in spines:
            path_i = lon[path[0]]
            path_j = lat[path[1]]
            plt.plot(path_i, path_j, \
                'g' if path_j[0] > 0. else 'b', linewidth=2, zorder=4)

        plt.xlabel('deg lon')
        plt.ylabel('deg lat')
        plt.title('NH/SH Jet Stream Spines step=%d'%(step))

        plt.savefig('%s_spine_%06d.png'%(self.out_file, step), dpi=self.dpi)

        if self.interact:
            plt.show()

        plt.close(fig)



# below is supporting code for implementing shortest path graph traversal

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


class graph_nodes(object):
    """ container for manipulating medial axis as a graph
        it is assumed that fetaures are simply connected and
        that no feature crosses a periodic boundary
    """
    def __init__(self, scalar, lab, lat, lon):
        """
        """
        self.ext = [0, lab.shape[1]-1, 0, lab.shape[0]-1]
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

    def shortest_path(self):

        p = []

        ok,e = self.trace_path()

        if not ok:

            raise RuntimeError( \
                'failed to locate a path (%d,%d) -> (%d,%d)'%( \
                beg.i,beg.j,end.i,end.j))

        # start at the end point, walk node parents back toward the start
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





















































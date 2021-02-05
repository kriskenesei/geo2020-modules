#########################################
## 3D-NWB DISSERTATION PROJECT SCRIPTS ##
##  KRISTOF KENESEI, STUDENT 5142334   ##
##    K.Kenesei@student.tudelft.nl     ##
#########################################

import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely import ops
from scipy.spatial import cKDTree
from laspy.file import File
import matplotlib

def conv_tovx(lstr):
    """Converts the input shapely LineString to a shapely
    MultiPoint (used in geopandas column generation when
    plotting vertices in addition to lines in .plot()).
    """
    return MultiPoint(lstr.coords)

def make_geom_idcol(lstr):
    """Returns a list containing the string 'nbw' as many times
    as the input LineString has vertices. This is used to create
    a column in the NWB GeoDataFrame where the program can
    indicate what the source each the vertex is (i.e. NWB
    or densification).
    """
    return list(np.full(len(lstr.coords), 'nwb'))

def calc_angle(jt, end0, end1):
    """Computes the angle between two 2D vectors that share a
    starting vertex "jt". It is used when determining which
    wegvak to continue on when an intersection is reached in
    the NBRS-generation algorithms.
    """
    v0 = np.array(end0) - np.array(jt)
    v1 = np.array(end1) - np.array(jt)
    v0, v1 = v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)
    return np.arccos(np.dot(v0, v1))

def ahn3_importer(fpath, classes = [2, 26]):
    """ Loads the AHN3 tile at the provided file path,
    and returns the 3D coordinates of the points with
    the specified classes as a numpy array.
    """
    in_file = File(fpath, mode = "r")
    in_np = np.vstack((in_file.raw_classification,
                       in_file.x, in_file.y, in_file.z)).transpose()
    return in_np[np.isin(in_np[:,0], classes)]

def filter_outliers(vxs):
    """ Performs 1D-smoothing on a series of input
    vertices that form a 3D line. Fits a quartic
    polynomial to obtain a model, and uses the data-
    model errors to identify outliers. It then uses
    numpy interpolation to find better values at
    these locations, and locations where the elevation
    was missing originally. Edits the input in-place.
    """
    zs = vxs[:,2]
    # zero out the entire NBRS if it has too many NaNs
    if len(zs[np.isnan(zs)]) / len(zs) > 0.75:
        vxs[:,2] = np.full(len(zs), np.NaN); return
    # find model polynomial and compute model values
    # exclude locations where the elevation is missing
    diffs = np.insert(np.diff(vxs[:,:2], axis = 0), 0, 0, axis = 0)
    # the x values of the 1D profile are
    # based on 2D vertex distances
    dsts = np.cumsum(np.sqrt((diffs ** 2).sum(axis = 1)))
    zs_notnan, dsts_notnan = zs[~np.isnan(zs)], dsts[~np.isnan(zs)]
    coeffs = np.polynomial.polynomial.polyfit(dsts_notnan, zs_notnan, 4)
    model_zs = np.polynomial.polynomial.polyval(dsts, coeffs)
    # compute the data-model errors and STD
    error = np.abs(zs - model_zs)
    std = np.std(error[~np.isnan(error)])
    # if STD is tiny, increase it artificially - we do not
    # wish to modify meaningful elevations just because
    # they are very slightly off relative to the model
    if std < 0.4: std = 0.4
    # create an index of vertices where the elevation is
    # either missing, or was identified as an outlier
    out_ix = (error > std) | np.isnan(error)
    # if any were found, replace them with interpolated values
    if len(out_ix) > 0:
        zs[out_ix] = np.interp(dsts[out_ix], dsts[~out_ix], zs[~out_ix])

class nbrs_manager:
    
    def __init__(self, fpath):
        """Imports NWB, or a cropped NWB "tile" upon
        initialisation. NWB is stored as a geopandas
        GeoDataFrame. The intended usage is as follows:
            1. Instantiate class with the path to the NWB tile.
            2. Invoke .generate_nbrs() with either algorithm
               to build Non-Branching Road Segments (NBRS).
            3. Optionally, invoke .densify() with a threshold
               in metres to also perform vertex densification.
               Highly recommended, as NWB's sampling is coarse.
            4. Invoke .estimate_elevations() with aligned AHN3
               tile to perform an estimation of NWB elevations.
            4. Optionally, plot either a single NBRS
               via .plot() or all NBRS via .plot_all()
            5. Optionally, write to disc using .write_all().
        """
        nwb = gpd.read_file(fpath); nwb['NBRS_ID'] = None
        self.nwb, self.nbrs, self.jte = nwb, {}, {}
        self.geoms = dict(zip(self.nwb['WVK_ID'],
                              self.nwb['geometry']))
        self.origins = dict(zip(self.nwb['WVK_ID'],
                                self.nwb['geometry'].apply(make_geom_idcol)))
        self.wvk_count, self.nbrs_count = len(self.nwb.index), 0
        self.nbrs_ids, self.nbrs_geoms, self.nbrs_revs = {}, {}, {}
    
    def is_empty(self):
        """Return True if NBRS have been generated,
        else return False.
        """
        if not self.nbrs_count: return True
        return False
    
    def has_preliminary_z(self):
        """Return True if preliminary elevations have
        been generated, else return False.
        """
        if 'geometry_simpleZ' in self.nwb.columns: return True
        return False
    
    def nbrs_count(self):
        """Return the number of NBRS that were generated.
        """
        return self.nbrs_count
    
    def get_ids(self):
        """Return a list of all available NBRS IDs.
        """
        if self.is_empty(): print('NBRS first need to be generated.')
        else: return list(range(self.nbrs_count))
    
    def get(self, nbrs_id):
        """Return NBRS with a particular ID as a GeoDataFrame.
        """
        if self.is_empty(): print('NBRS first need to be generated.')
        available_ids = self.get_ids()
        if nbrs_id not in available_ids:
            print("NBRS not found. Valid IDs are: ", available_ids)
        else: return self.nwb[self.nwb['NBRS_ID'] == nbrs_id].copy()
    
    def get_geom(self, nbrs_id):
        """Return the geometries of a given wegvak. Since the
        wegvakken in self.nbrs are stored in-order, this method
        also returns them in the correct order.
        """
        wvk_ids = self.nbrs[nbrs_id]
        if wvk_ids: return [self.geoms[wvk_id] for wvk_id in wvk_ids]
    
    def get_wvk_geom(self, wvk_id):
        """Return the geometry of wegvak with the provided wegvak ID.
        """
        geom = self.geoms.get(wvk_id)
        if geom: return geom
        else: print('Wegvak not found.')
    
    def set_wvk_geom(self, wvk_id, new_geom):
        """Set the geometry of wegvak with the provided wegvak ID.
        Both the hashed-storage geometry and the GeoDataFrame
        geometry are reset.
        """
        geom_col = self.nwb.geometry.name
        if wvk_id in self.geoms.keys():
            self.nwb.loc[self.nwb['WVK_ID'] == wvk_id, geom_col] = new_geom
            self.geoms[wvk_id] = new_geom
        else: print('Wegvak not found.')
        
    def set_geocol(self, new_colname):
        """Set the geometry column of the NWB GeoDataFrame to a new
        column. This is needed, so that the class behaves similar to
        the GeoDataFrame itself.
        """
        if new_colname in self.nwb.columns:
            try: self.nwb.set_geometry(new_colname, inplace = True)
            except: print('Unable to set new geometry column.')
        else: print('Attribute table column not found.')
    
    def plot(self, ix, show_vxs = False):
        """Plots a selected NBRS. Optionally, it can
        also show the vertices in the figure (useful
        when looking at the results of densification).     
        """
        chosen = self.get(ix)
        if chosen is not None:
            if show_vxs == False: chosen.plot(figsize = [4,6])
            else:
                chosen['geometry_vx'] = chosen.geometry.apply(conv_tovx)
                base = chosen.plot(figsize = [4,6])
                try: chosen.set_geometry('geometry_vx', inplace = True)
                except: print('Unable to set new geometry column.')
                chosen.plot(ax = base, color = 'red', markersize = 20)
        else: print('Wegvak not found.')
    
    def plot_all(self):
        """Plots all the NBRS that were generated.
        Uses random colours to help distinguish between
        NBRS that run close to each other.
        """
        if self.is_empty():
            print('WARNING: NBRS have not been generated yet.')
        cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
        self.nwb.plot(column = 'NBRS_ID', cmap = cmap)
            
    def write_all(self, fpath, to_drop = ['geometry']):
        """Writes all the NBRS that were generated.
        """
        if self.is_empty():
            print('WARNING: NBRS have not been generated yet.')
        to_write = self.nwb.copy()
        to_write.drop(to_drop, 1, inplace = True)
        to_write.to_file(fpath)
    
    def generate_nbrs(self, algorithm = 'geometric'):
        """Starts either NBRS generation algorithm. For the
        threshold, around 15-25m works well for all testing
        "tiles". Both algorithms work equally well for all
        testing tiles, see below for more details. In this
        release, both algorithms now use hashing for all
        operations, hence their performance does not differ
        significantly. The GeoDataFrame is queried/edited
        only where absolutely necessary.
        """
        if not self.is_empty(): print('NBRS have already been generated.')
        elif algorithm == 'geometric': self._generate_nbrs_geometric()
        elif algorithm == 'semantic': self._generate_nbrs_geometric()
        else: print('Unknown algorithm. Available: "geometric", "semantic".')
    
    def _generate_nbrs_geometric(self):
        """Performs the geometry-based NBRS-generation.
        No geometric overlapping of NBRS is enforced "formally"
        via intersection checks. Furthermore, the navigation
        of the topology and the selection of the best candidate
        for continuation in intersections also takes place
        exclusively via examining the geometry, semantic
        information is not used to any extent.
        """
        print('\nSTARTING NBRS GENERATION')
        # build a hashing-based navigation structure of
        # the NWB topology based on linking each intersection
        # location to the wegvakken that end/begin there
        for wvk_id, lstr in self.geoms.items():
            last_vx = tuple(np.round(lstr.coords[-1], 1))
            first_vx = tuple(np.round(lstr.coords[0], 1))
            end, beg = self.jte.get(last_vx), self.jte.get(first_vx)
            if not end: self.jte[last_vx] = [wvk_id]
            else: self.jte[last_vx].append(wvk_id)
            if not beg: self.jte[first_vx] = [wvk_id]
            else: self.jte[first_vx].append(wvk_id)
        # nucleate "growing" processes by picking wegvakken
        # that have not been use yet - growing first takes
        # place starting from the last vertex, then backwards
        # starting from the first vertex
        i = 0
        while len(self.nbrs_ids) < self.wvk_count:
            wvk_id = (self.geoms.keys() - self.nbrs_ids.keys()).pop()
            # start a new NBRS with the chosen wegvak
            self.nbrs[i], self.nbrs_ids[wvk_id] = [wvk_id], i
            lstr = self.geoms[wvk_id]
            self.nbrs_revs[i], self.nbrs_geoms[i] = [False], lstr
            self.nwb.loc[self.nwb['WVK_ID'] == wvk_id, 'NBRS_ID'] = i
            end_vx, beg_vx = lstr.coords[-2], lstr.coords[1]
            end_jt = tuple(np.round(lstr.coords[-1], 1))
            beg_jt = tuple(np.round(lstr.coords[0], 1)) 
            # the attachment of wegvakken is recursive,
            # only one function call is needed per direction
            self._join_wvk_geometric(wvk_id, end_vx, end_jt, 'end', i)
            self._join_wvk_geometric(wvk_id, beg_vx, beg_jt, 'beg', i)
            i += 1
        self.nbrs_count = i
        print('FINISHED NBRS GENERATION')
            
    def _join_wvk_geometric(self, prev_id, prev_vx, jt_vx, sense, nbrs_id):
        """Traverse NWB recursively, attaching wegvakken
        in one direction based on geometric rules.
        """
        nbrs_geom = self.nbrs_geoms[nbrs_id]
        # look for unused wegvakken starting from end
        # "intersection" of the last wegvak
        cont_ids = self.jte[jt_vx] - self.nbrs_ids.keys()
        cont = {k: v for k, v in self.geoms.items() if k in cont_ids}
        # compute relative angles of outgoing vectors from
        # the intersection - vectors are based on the first
        # edge of each wegvak connected to the intersection
        cont_vecs = {}
        for wvk_id, lstr in cont.items():
            if tuple(np.round(lstr.coords[0], 1)) == jt_vx:
                cont_vecs[wvk_id] = lstr.coords[1]
            else: cont_vecs[wvk_id] = lstr.coords[-2]
        angles = {calc_angle(jt_vx, prev_vx, vec): wvk_id
                  for wvk_id, vec in cont_vecs.items()}
        # this is an override for intersections that join
        # exactly three wegvakken - this is typical where
        # ramps join motorways, and is used to avoid
        # continuing ramp NBRS on motorways
        if len(cont_vecs) == 2:
            vecs = [vec for _, vec in cont_vecs.items()]
            if calc_angle(jt_vx, vecs[0], vecs[1]) > max(angles.keys()):
                return
        # work through the "stack" of candidate wegvakken
        # in the order of decreasing angle optimality -
        # when found a good one, add it to the NBRS and
        # make the next recursive call
        while angles:
            best_id = angles.pop(max(angles.keys()))
            next_lstr = cont[best_id]
            # perform the non-intersection test with the
            # pre-existing merged NBRS geometry
            if not next_lstr.crosses(nbrs_geom):
                snapped = ops.snap(next_lstr, nbrs_geom, 0.01)
                self.nbrs_geoms[nbrs_id] = ops.linemerge((snapped,
                                                          nbrs_geom))
                if sense == 'end': self.nbrs[nbrs_id].append(best_id)
                else: self.nbrs[nbrs_id].insert(0, best_id)
                self.nbrs_ids[best_id] = nbrs_id
                self.nwb.loc[self.nwb['WVK_ID'] == best_id,
                             'NBRS_ID'] = nbrs_id
                if tuple(np.round(next_lstr.coords[0], 1)) == jt_vx:
                    next_vx = next_lstr.coords[-2]
                    next_jt = tuple(np.round(next_lstr.coords[-1], 1))
                else:
                    next_vx = next_lstr.coords[1]
                    next_jt = tuple(np.round(next_lstr.coords[0], 1))
                self._join_wvk_geometric(best_id, next_vx,
                                         next_jt, sense, nbrs_id)
                break
    
    def _generate_nbrs_semantic(self):
        """Performs semantic information-based NBRS-generation.
        The selection of the best candidate wegvak in intersections
        is handled using the same geometric tools as in the other
        algorithm, but additional filtering is performed.
        Only roads with the same wegnummer are allowed to be in the
        same NBRS. Furthermore, the BST-code in a given NBRS also
        needs to be consistent, apart from 'PST' which always
        denotes the last wegvak that joins a ramp to a motorway.
        As such, they are always isolated in terms of their BST-
        codes and are therefore merged into the ramp-NBRS that
        they are spatially consistent with.
        """
        print('\nSTARTING NBRS GENERATION')
        # build a hashing-based navigation structure of
        # the NWB topology based on linking wegvak IDs
        # to the intersections IDs (JTE_ID) that are found
        # in the attribute table of NWB
        self.jte_end = dict(zip(self.nwb['WVK_ID'], self.nwb['JTE_ID_END']))
        self.jte_beg = dict(zip(self.nwb['WVK_ID'], self.nwb['JTE_ID_BEG']))
        for wvk_id in self.geoms.keys():
            end_id, beg_id = self.jte_end[wvk_id], self.jte_beg[wvk_id]
            end, beg = self.jte.get(end_id), self.jte.get(beg_id)
            if not end: self.jte[end_id] = [wvk_id]
            else: self.jte[end_id].append(wvk_id)
            if not beg: self.jte[beg_id] = [wvk_id]
            else: self.jte[beg_id].append(wvk_id)
        self.nbrs_bst, self.nbrs_wegnos = {}, {}
        self.bst_codes = dict(zip(self.nwb['WVK_ID'], self.nwb['BST_CODE']))
        self.wegnos = dict(zip(self.nwb['WVK_ID'], self.nwb['WEGNUMMER']))
        # nucleate "growing" processes by picking wegvakken
        # that have not been use yet - growing first takes
        # place starting from the last vertex, then backwards
        # starting from the first vertex
        i = 0
        while len(self.nbrs_ids) < self.wvk_count:
            # select the first unused non-PST wegvak to use to
            # "nucleate" a new NBRS with - no NBRS should start
            # with a ramp-end wegvak, as that could result in
            # ramps continuing in motorway wegvakken
            # this effectively replaces the geometry-based
            # override that is found in the other algorithm
            wvk_ids = [k for k in self.geoms.keys()
                       if not self.bst_codes[k] == 'PST']
            unused_ids = set(wvk_ids) - self.nbrs_ids.keys()
            if unused_ids: wvk_id = unused_ids.pop()
            else: wvk_id = (self.geoms.keys() - self.nbrs_ids.keys()).pop()
            # start a new NBRS with the chosen wegvak
            self.nbrs[i], self.nbrs_ids[wvk_id] = [wvk_id], i
            self.nbrs_bst[i] = self.bst_codes[wvk_id]
            self.nbrs_wegnos[i] = self.wegnos[wvk_id]
            lstr = self.geoms[wvk_id]
            self.nbrs_revs[i] = [False]
            self.nwb.loc[self.nwb['WVK_ID'] == wvk_id, 'NBRS_ID'] = i
            end_vx, beg_vx = lstr.coords[-2], lstr.coords[1]
            # the attachment of wegvakken is recursive,
            # only one function call is needed per direction
            self._join_wvk_semantic(wvk_id, end_vx,
                                    self.jte_end[wvk_id], 'end', i)
            self._join_wvk_semantic(wvk_id, beg_vx,
                                    self.jte_beg[wvk_id], 'beg', i)
            i += 1
        self.nbrs_count = i
        # delete two large navigation dictionaries that will
        # no longer be needed and are just taking up space
        del self.jte_end, self.jte_beg
        print('FINISHED NBRS GENERATION')
    
    def _join_wvk_semantic(self, prev_id, prev_vx, prev_jt, sense, nbrs_id):
        """Internal method that recursively adds suitable
        wegvakken to the NBRS that is being built.
        """
        # select unused wegvakken that have the same
        # wegnummer as the previous one, and which
        # join to the previous one in the normal sense,
        # i.e. where no reversal occurs
        cont_ids = [k for k in self.jte[prev_jt]
                    if self.wegnos[k] == self.nbrs_wegnos[nbrs_id]]
        cont_ids = set(cont_ids) - self.nbrs_ids.keys()
        cont = {k: v for k, v in self.geoms.items() if k in cont_ids}
        # compute relative angles of outgoing vectors from
        # the intersection - vectors are based on the first
        # edge of each wegvak connected to the intersection
        cont_vxs, cont_revs = {}, {}
        for wvk_id, lstr in cont.items():
            if self.jte_beg[wvk_id] == prev_jt:
                cont_vxs[wvk_id] = lstr.coords[:2]
                cont_revs[wvk_id] = False
            else: cont_vxs[wvk_id] = (lstr.coords[-1], lstr.coords[-2])
        angles = {calc_angle(vxs[0], prev_vx, vxs[1]): wvk_id
                  for wvk_id, vxs in cont_vxs.items()}
        # work through two "stacks" of candidate wegvakken
        # in the order of decreasing angle optimality
        target_bst, best_id = self.nbrs_bst.get(nbrs_id), None
        # if the NBRS already has an associated BST-code,
        # then first try finding a continuation that has
        # the same code
        if target_bst:
            bst_stack = angles.copy()
            while bst_stack:
                pot_id = bst_stack.pop(max(bst_stack.keys()))
                if self.bst_codes[pot_id] == target_bst:
                    best_id = pot_id; break
        # only if either the NBRS has no BST-code associated
        # with it yet or no such continuation could be found,
        # should a 'PST' continuation be considered
        if not best_id and target_bst != 'PST':
            while angles:
                pot_id = angles.pop(max(angles.keys()))
                if self.bst_codes[pot_id] == 'PST':
                    best_id = pot_id; break
        # when found a good candidate, add it to the NBRS and
        # make the next recursive call - note how the non-
        # intersection test with the pre-existing merged NBRS
        # geometry is not performed in this algorithm
        if best_id:
            if sense == 'end': self.nbrs[nbrs_id].append(best_id)
            else: self.nbrs[nbrs_id].insert(0, best_id)
            self.nbrs_ids[best_id] = nbrs_id
            self.nwb.loc[self.nwb['WVK_ID'] == best_id,
                         'NBRS_ID'] = nbrs_id
            next_lstr = cont[best_id]
            if self.jte_beg[best_id] == prev_jt:
                next_vx = next_lstr.coords[-2]
                next_jt = self.jte_end[best_id]
            else:
                next_vx = next_lstr.coords[1]
                next_jt = self.jte_beg[best_id]
            self._join_wvk_semantic(best_id, next_vx,
                                    next_jt, sense, nbrs_id)
            
    def densify(self, thres):
        """This method densifies the vertices of the wegvakken
        to respect an input distance threshold (in metres).
        Each edge in each wegvak is checked, and is split in two
        if necessary. The same is then done to the resulting
        halves until the specified threshold is respected.
        """
        print('\nSTARTING NBRS DENSIFICATION')
        # iterate all wegvakken
        for wvk_id in self.geoms.keys():
            wvk_geom, origins = self.geoms[wvk_id], self.origins[wvk_id]
            orig_coords = wvk_geom.coords
            orig_coord_count = len(orig_coords)
            # iterate the original edges of the selected wegvak
            i, j = 0, 0 
            while i < orig_coord_count - 1:
                vxs = MultiPoint(orig_coords[i:i+2])
                # start the recursive vertex densification
                # of the selected edge of the wegvak
                dense = self._densify_edge(LineString(vxs), thres)
                # add the new vertices to the list that
                # indicates origin, and mark that they
                # originate from densification
                diff = len(dense.coords) - 2
                origins = origins[:j+1] + ['added'] * diff + origins[j+1:]
                # special case: wegvak is a single edge
                if orig_coord_count == 2: wvk_geom = dense
                else:
                    # patch the densified edge into the original
                    # wegvak (by replacing original edge)
                    split = ops.split(wvk_geom, vxs)
                    # special case: first edge of wegvak
                    if not i: to_merge = MultiLineString([dense, split[1]])
                    # special case: last edge of wegvak
                    elif i == orig_coord_count - 2:
                        to_merge = MultiLineString([split[0], dense])
                    # general case
                    else: to_merge = MultiLineString([split[0],
                                                      dense, split[2]])
                    wvk_geom = ops.linemerge(to_merge)
                i, j = i + 1, i + diff + 1
            # set the wegvak geometries and origin indicators
            # of the NWB GeoDataFrame to the densified ones
            # in one bulk operation, as this is more efficient
            self.set_wvk_geom(wvk_id, wvk_geom)
            self.origins[wvk_id] = origins
        print('FINISHED NBRS DENSIFICATION')

    def _densify_edge(self, edge, thres):
        """Internal method that recursively densifies an edge.
        It splits the edge while opening new frames on the
        stack, and merges them when returning out of the frames.
        """
        dst = edge.length
        # if distance threshold is not respected: split
        if dst > thres:
            pt = ops.substring(edge, dst / 2, dst / 2)
            edge_0 = LineString([edge.coords[0], pt])
            edge_1 = LineString([pt, edge.coords[1]])
            # recursively split first half and second half
            edge_0 = self._densify_edge(edge_0, thres)
            edge_1 = self._densify_edge(edge_1, thres)
            # merge the two halves when recursion is done
            return ops.linemerge(MultiLineString([edge_0, edge_1]))
        # if distance threshold is respected: return edge
        else: return edge
        
    def estimate_elevations(self, fpath):
        if self.has_preliminary_z():
            print('Preliminary elevations have already been generated.')
            return
        print('\nSTARTING PRELIMINARY ELEVATION ESTIMATION')
        # import and subsample Lidar
        lidar = ahn3_importer(fpath)[::2]
        # build a list of all NBRS vertex counts and vertices -
        # per-NBRS lists are needed for smoothing, and a
        # completely flat list of all vertices is needed
        # for the initial KD-tree query
        self.nbrs_vxnos, self.nbrs_vxs, self.nbrs_revs = {}, {}, {}
        flat_vxs = []
        for nbrs_id in self.get_ids():
            nbrs_geom = self.get_geom(nbrs_id)
            # although the wegvak IDs are in the correct order
            # in NBRS lists in self.nbrs, the LineString
            # coordinates themselves are often reversed
            # randomly in NWB, meaning that we need to check for
            # reversals while we are flattening the lists
            # NOTE: although it would be much simpler to
            # do all this with the merged NBRS geometries,
            # the goal is to be able to edit the original
            # NWB wegvakken LineString objects -
            # unfortunately the merged NBRS geometries
            # also contain snapped vertices and reversed
            # wegvakken and tracking these throughout the
            # below procedure would be even more difficult
            # than starting from scratch...
            revs = np.full(len(nbrs_geom), False)
            # special case: NBRS with only one wegvak
            if len(nbrs_geom) == 1:
                self.nbrs_vxnos[nbrs_id] = [len(nbrs_geom[0].coords)]
                self.nbrs_vxs[nbrs_id] = list(nbrs_geom[0].coords)
                self.nbrs_revs[nbrs_id] = list(revs)
                flat_vxs += list(nbrs_geom[0].coords)
                continue
            # the starting direction needs to first be
            # established "manually"
            if (np.round(nbrs_geom[0].coords[0], 1)
                == np.round(nbrs_geom[1].coords[0], 1)).any():
                nbrs_vxs = nbrs_geom[0].coords[::-1]; revs[0] = True
            elif (np.round(nbrs_geom[0].coords[0], 1)
                == np.round(nbrs_geom[1].coords[-1], 1)).any():
                nbrs_vxs = nbrs_geom[0].coords[::-1]; revs[0] = True
            else: nbrs_vxs = list(nbrs_geom[0].coords)
            flat_vxs += nbrs_vxs
            # based on rectifying the first wegvak above,
            # rectify the order of all subsequent ones -
            # when building the flat vertex lists, we must
            # discard the first vertex of each wegvak, as
            # it matches last vertex of the previous...
            nbrs_vxnos = [len(nbrs_geom[0].coords)]
            for wvk_ix in range(len(nbrs_geom) - 1):
                if (np.round(nbrs_geom[wvk_ix + 1].coords[0], 1) ==
                    np.round(nbrs_vxs[-1], 1)).any():
                        wvk_vxs = list(nbrs_geom[wvk_ix + 1].coords[1:])
                else:
                    wvk_vxs = nbrs_geom[wvk_ix + 1].coords[::-1][1:]
                    revs[wvk_ix + 1] = True
                nbrs_vxnos.append(len(wvk_vxs))
                nbrs_vxs += wvk_vxs; flat_vxs += wvk_vxs
            self.nbrs_vxnos[nbrs_id] = nbrs_vxnos
            self.nbrs_vxs[nbrs_id] = nbrs_vxs
            self.nbrs_revs[nbrs_id] = list(revs)
        # build KD-trees from Lidar points and NBRS vertices
        nbrs_tree = cKDTree(flat_vxs)
        lidar_tree = cKDTree(np.array([lidar[:,1], lidar[:,2]]).transpose())
        # get all the Lidar points that are very close to NBRS vertices
        nbrs_ix = nbrs_tree.query_ball_tree(lidar_tree, r = 3)
        # link 3D Lidar points to 2D-projected Lidar points
        arr_xy = np.array([lidar[:,1], lidar[:,2]]).transpose()
        xy = [tuple(arr) for arr in arr_xy]
        link = dict(zip(xy, lidar[:,3]))
        # initialise new 3D geometry column and activate it
        self.nwb['geometry_simpleZ'] = None
        self.set_geocol('geometry_simpleZ')
        self.z_wasnan = {}
        # compute a rough z-value for each NBRS vertex,
        # perform the smoothing workflow when an NBRS is
        # completed, and add 3D geometries to GeoDataFrame
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            nbrs_vxs = self.nbrs_vxs[nbrs_id]
            # fetch the nearby Lidar points of each
            # NBRS vertex and compute is elevation
            # as the median of the elevations of
            # the Lidar points
            zs = []
            for _ in range(sum(nbrs_vxnos)):
                # since the KD-tree required a completely flat
                # list of all vertices, it is essential that
                # here the same order of processing is used
                # as the one that produced the flat list -
                # the vertices are popped one by one from it
                # like in stack processing
                pt_ix = nbrs_ix.pop(0)
                if pt_ix:
                    pt_xy = [xy[ix] for ix in pt_ix]
                    pt_z = [link[xy] for xy in pt_xy]
                    zs.append(np.median(pt_z))
                else: zs.append(np.NaN)
            # assemble NBRS 3D coordinates into an array
            nbrs_vxs_z = np.c_[nbrs_vxs, zs]
            # apply the elevation smoothing algorithm
            # to the 3D coordinate array
            filter_outliers(nbrs_vxs_z)
            # re-assemble the smoothed 3D coordinate arrays
            # into wegvak geometries and set the active wegvak
            # geometries of the class to these new geometries, and
            # also save indicator lists (per wegvak, like in
            # self.origins) that record where the rough elevation
            # estimation originally failed - which is in turn
            # indicative of the presence of bridges, tunnels, etc.
            nbrs_vxnos = np.array(nbrs_vxnos); nbrs_vxnos[1:] += 1
            starts = np.roll(np.cumsum(nbrs_vxnos - 1), 1); starts[0] = 0
            ends = np.cumsum(nbrs_vxnos - 1) + 1
            slices = [[nbrs_vxs_z[start:end], list(np.isnan(zs))]
                      for start, end in zip(starts, ends)]
            for wvk_id, sliced, rev in zip(self.nbrs[nbrs_id], slices,
                                           self.nbrs_revs[nbrs_id]):
                wvk_vxs = sliced[0]
                if rev: wvk_vxs = np.flip(wvk_vxs, axis = 0)
                self.set_wvk_geom(wvk_id, LineString(wvk_vxs))
                self.z_wasnan[wvk_id] = sliced[1]
        # for now, delete the per-NBRS vertex list - maybe later
        # it will be useful for something else too, but for now
        # it has no other use and it is very large
        del self.nbrs_vxs
        print('FINISHED PRELIMINARY ELEVATION ESTIMATION')

# testing config, only runs when script is not imported
if __name__ == '__main__':
    nwb_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_nwb.shp'
    ahn_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_2_26_clipped.las'
    out_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_simpleZ.shp'
    roads = nbrs_manager(nwb_fpath)
    roads.generate_nbrs()
    roads.densify(5)
    roads.estimate_elevations(ahn_fpath)
    roads.write_all(out_fpath)
    #roads.plot(1, True)
    roads.plot_all()
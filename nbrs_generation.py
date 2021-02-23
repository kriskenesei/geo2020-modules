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
    a column in the NWB GeoDataFrame where the software can
    indicate what the source each the vertex is (i.e. NWB
    or densification).
    """
    return list(np.full(len(lstr.coords), 'nwb'))

def get_geom_type(lstr):
    """Returns the shapely geometry type of the input geometry.
    This is used in GeoPandas column generation, in turn used
    to filter out buggy geometries in the input.
    """
    return lstr.geom_type

def calc_angle(jt, end0, end1):
    """Computes the angle between two vectors that share a
    starting vertex "jt". It is used when determining which
    wegvak to continue on when an intersection is reached in
    the NBRS-generation algorithms.
    """
    v0 = np.array(end0) - np.array(jt)
    v1 = np.array(end1) - np.array(jt)
    v0, v1 = v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)
    return np.arccos(np.dot(v0, v1))

def las_reader(fpath, classes = (2, 26)):
    """ Loads the AHN3 tile at the provided file path,
    and returns the 3D coordinates of the points with
    the specified classes as a numpy array.
    """
    with File(fpath, mode = "r") as in_file:
        in_np = np.vstack((in_file.raw_classification,
                           in_file.x, in_file.y, in_file.z)).transpose()
        header = in_file.header.copy()
    return in_np[np.isin(in_np[:,0], classes)], header

def las_writer(fpath, header, pts):
    """Writes the input points of format (x, y, z,
    NBRS_ID, ORIGIN) to disk in the LAS format, using
    the provided header (preferably that of the input).
    This is used to write the segmented point cloud to disk,
    to visualise the intermediate results.
    """
    with File(fpath, mode="w", header=header) as out_file:
        out_file.define_new_dimension(name = "NBRS_ID", data_type = 3,
                                      description = 'NBRS_ID')
        out_file.define_new_dimension(name = "ORIGIN", data_type = 3,
                                      description = 'ORIGIN')
        out_file.x = pts[:,0]
        out_file.y = pts[:,1]
        out_file.z = pts[:,2]
        out_file.ORIGIN = pts[:,3].astype(int)
        out_file.NBRS_ID = pts[:,4].astype(int)

def planefit_lsq(vxs):
    """Function to perform least-squares plane fitting on the
    input points of format (x, y, z). The fitted plane is
    returned as the 'd' parameter of the plane equation and
    a normal vector of the plane.
    """
    A = np.c_[vxs[:,:2], np.full(len(vxs), 1)]
    (a, b, c) = np.linalg.lstsq(A, vxs[:,2], rcond = None)[0]
    normal = (a, b, -1) / np.linalg.norm((a, b, -1))
    d = -np.array([0.0, 0.0, c]).dot(normal)
    return d, normal

def dist_topoint(pt, other_pt):
    """Returns the 2D or 3D distance of two points, depending
    on the dimensionality of the input points.
    """
    if len(pt) == 2: return np.sqrt((pt[0] - other_pt[0]) ** 2 +
                                    (pt[1] - other_pt[1]) ** 2)
    return np.sqrt((pt[0] - other_pt[0]) ** 2 +
                   (pt[1] - other_pt[1]) ** 2 +
                   (pt[2] - other_pt[2]) ** 2)

def dist_toplane(pt, d, normal):
    """Returns the distance of a point to a plane represented
    by the 'd' parameter of the plane equation, and a normal
    of the plane.
    """
    nom = normal[0] * pt[0] + normal[1] * pt[1] + normal[2] * pt[2] + d
    den = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    return abs(nom / den)

def orientation(vx0, vx1, pt):
    """Function to perform an Orientation test.
    
    Takes three points.
    The Orientation test is relative to the last parameter point.
    
    Returns the Orientation test result in the following format:
    -1 -- Point is clockwise from the line.
    0  -- Point is collinear with the line.
    1  -- Point is counter-clockwise from the line.
    """
    A = np.array([[vx0[0], vx0[1], 1],
                  [vx1[0], vx1[1], 1],
                  [pt[0], pt[1], 1]])
    det = np.linalg.det(A)
    if det < 0: return -1
    if det > 0: return 1
    return 0

def get_cross(vx0, vx1, vx2, len_cross, sense):
    """Function to return a "cross-section" at the shared
    vertex of two connected edges. The cross-section will
    have the mean slope of the two edges. Its length should
    be provided as a parameter. The "sense" parameter
    specifies which side of the edges should serve as a
    reference for the orientation of the cross-section.
    A workaround for vertical edges is included. The
    function is void for edges with inconsistent slope
    signs, as handling these would introduce an unnecessary
    amount of complexity into the code.
    """
    if vx1[0] == vx0[0]: vx1[0] += 0.00001
    slope = (vx1[1] - vx0[1]) / (vx1[0] - vx0[0])
    # if the second edge does not exist, the cross section is
    # still constructed based on the one existing edge
    if vx2:
        if vx2[0] == vx1[0]: vx2[0] += 0.00001
        slope1 = (vx2[1] - vx1[1]) / (vx2[0] - vx1[0])
        if np.sign(slope) != np.sign(slope1): return
        len0, len1 = dist_topoint(vx0, vx1), dist_topoint(vx0, vx1)
        slope = (slope * len0 + slope1 * len1) / (len0 + len1)
    dy = np.sqrt((len_cross / 2)**2 / (slope**2 + 1))
    dx = -slope * dy
    cross = ((vx1[0] + dx, vx1[1] + dy),
             (vx1[0] - dx, vx1[1] - dy))
    # orientation test ensures that output cross-sections
    # are always constructed with the right orientation,
    # so that they can be used to construct road "curbs"
    if orientation(vx0, vx1, cross[0]) != sense: return cross[::-1]
    return cross

def filter_outliers(vxs, degree = 5, only_detect = False,
                    thres = 1, prox = None):
    """ Performs 1D-smoothing (or alternatively, just outlier
    detection) on a series of input points that form a 3D line.
    Fits a polynomial f the desired degree to obtain a model,
    and uses the data-model errors to identify outliers.
    If smoothing is desired, it uses numpy interpolation to find
    better values at the locations of outliers, and at locations
    where elevation was missing originally. Edits the input in-place.
    The parameter "thres" configures the outlier detection
    threshold as the number of standard deviations where the cutoff
    should occur. The parameter "prox" is for internal purposes,
    it allows the fitting to take place only on a certain
    percentage of the input points centred around the middle.
    """
    zs = vxs[:,2].copy()
    if prox:
        mid, cut = len(zs) // 2, round(len(zs) * prox // 2)
        zs[:mid - cut] = np.NaN; zs[mid + cut + 1:] = np.NaN
    if ((not prox and len(zs[np.isnan(zs)]) / len(zs) > 0.75)
        or (prox and len(zs[~np.isnan(zs)]) < degree + 1)):
        if only_detect == False:
            # zero out the entire NBRS if it has too many NaNs
            vxs[:,2] = np.full(len(zs), np.NaN)
        return
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
    error = np.abs(vxs[:,2] - model_zs)
    std = np.std(error[~np.isnan(error)])
    if std < 0.4: std = 0.4
    # do not perform smoothing if only filtering is desired
    if only_detect: return error > thres * std
    # create an index of vertices where the elevation is
    # either missing, or was identified as an outlier
    out_ix = (error > thres) | np.isnan(error)
    # if any were found, replace them with interpolated values
    if len(out_ix) > 0:
        zs[out_ix] = np.interp(dsts[out_ix], dsts[~out_ix], zs[~out_ix])
        vxs[:,2] = zs


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
            5. Optionally, write NBRS geometry to disc by
               invoking .write_all().
            6. Invoke .segment_lidar() with aligned DTB tile
               to perform point cloud segmentation (pre-
               selection of AHN3 points that fall on the road
               surfaces belonging to specific NBRS).
               Post-segmentation, the sublocuds can be found
               in .nbrs_subclouds, which needs to be indexed
               with an NBRS ID to return a subcloud.
            7. Optionally, write to segmented point cloud to
               disk as a LAS file by invoking .write_subclouds().
            8. Invoke .estimate_edges() to construct crude edge
               estimates. Currently, both the edge estimates
               and the cross-sections are stored as NBRS-level
               geometries in .nbrs_edges and .nbrs_crosses,
               which need to be indexed with an NBRS ID to
               return edges/cross-sections.
            9. Optionally, write the edges and cross-sections
               to disk by invoking .write_edges().
        NOTE: the edges and cross-sections are stored as 3D
            geometries. This is for reference purposes only, as
            their planned use is to provide initial road edge
            estimates for the upcoming active contour optimisation
            feature, which is an intrinsically 2D procedure.
        Example calls are provided at the end of this script.
        """
        nwb = gpd.read_file(fpath); nwb['NBRS_ID'] = None
        nwb = nwb[~nwb['geometry'].isnull()]
        nwb['geometry_type'] = nwb['geometry'].apply(get_geom_type)
        nwb = nwb[nwb['geometry_type'] == 'LineString']
        self.nwb = nwb.drop(['geometry_type'], 1)
        self.nbrs, self.jte, self.nbrs_subclouds = {}, {}, {}
        self.geoms = dict(zip(self.nwb['WVK_ID'],
                              self.nwb['geometry']))
        self.origins = dict(zip(self.nwb['WVK_ID'],
                                self.nwb['geometry'].apply(make_geom_idcol)))
        self.wvk_count, self.nbrs_count = len(self.nwb.index), 0
        self.nbrs_ids, self.nbrs_geoms, self.nbrs_revs = {}, {}, {}
        self.nbrs_edges, self.nbrs_crosses = [], []
    
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
        self.nwb.drop(to_drop, 1).to_file(fpath)
        
    def write_subclouds(self, fpath):
        """Writes all the subclouds that were generated.
        """
        if not self.nbrs_subclouds:
            print('ERROR: Subclouds have not been generated yet.')
            return
        subclouds = []
        for key, item in self.nbrs_subclouds.items():
            subclouds.append(np.c_[item, np.full(len(item), key)])
        las_writer(fpath, self.las_header, np.concatenate(subclouds))
        
    def write_edges(self, fpath_edges, fpath_crosses):
        """Writes the crude edges (and cross-sections) that were
        generated. A file path for both the edges and the cross-
        sections needs to be provided.
        """
        if len(self.nbrs_edges) == 0:
            print('ERROR: Edges have not been generated yet.')
            return
        self.nbrs_edges.to_file(fpath_edges)
        self.nbrs_crosses.to_file(fpath_crosses)
    
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
                snapped = ops.snap(next_lstr, nbrs_geom, 0.1)
                merged = ops.linemerge((snapped, nbrs_geom))
                # check for invalid geometries - if found,
                # split the NBRS there - most likely it is
                # a roundabout that is not split into multiple
                # wegvakken, but is stored as a LineString
                if merged.geom_type != 'LineString': break
                # if passed all the checks, add to NBRS and
                # make the next recursive call
                self.nbrs_geoms[nbrs_id] = merged
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
        self.thres = thres
        print('\nSTARTING NBRS DENSIFICATION')
        # iterate all wegvakken
        for wvk_id in self.geoms.keys():
            wvk_geom, origins = self.geoms[wvk_id], self.origins[wvk_id]
            wvk_geom, origins = self._densify_lstr(wvk_geom, thres, origins)
            self.set_wvk_geom(wvk_id, wvk_geom)
            self.origins[wvk_id] = origins
        print('FINISHED NBRS DENSIFICATION')

    def _densify_lstr(self, lstr, thres, origins = None):
        orig_coords = lstr.coords; orig_len = len(orig_coords)
        i, j = 0, 0
        while i < orig_len - 1:
            vxs = orig_coords[i:i+2]
            # start the recursive vertex densification
            # of the selected edge of the wegvak
            dense = LineString(self._densify_edge(vxs, thres))
            # add the new vertices to the list that
            # indicates origin, and mark that they
            # originate from densification
            diff = len(dense.coords) - 2
            if origins:
                origins = origins[:j+1] + ['added'] * diff + origins[j+1:]
            # special case: wegvak is a single edge
            if orig_len == 2: lstr = dense
            else:
                # patch the densified edge into the original
                # wegvak (by replacing original edge)
                split = ops.split(lstr, MultiPoint(vxs))
                # special case: first edge of wegvak
                if not i: to_merge = MultiLineString([dense, split[1]])
                # special case: last edge of wegvak
                elif i == orig_len - 2:
                    to_merge = MultiLineString([split[0], dense])
                # general case
                elif len(split) != 3: return
                else: to_merge = MultiLineString([split[0],
                                                  dense, split[2]])
                lstr = ops.linemerge(to_merge)
            i, j = i + 1, i + diff + 1
        return lstr, origins

    def _densify_edge(self, edge, thres):
        """Internal method that recursively densifies an edge.
        It splits the edge while opening new frames on the
        stack, and merges them when returning out of the frames.
        New in this version: no shapely objects are used in
        the recursion to speed up processing.
        """
        dst = dist_topoint(edge[0], edge[1])
        # if distance threshold is not respected: split
        if dst > thres:
            pt = tuple(np.mean(edge, axis = 0))
            edge_0, edge_1 = (edge[0], pt), (pt, edge[1])
            # recursively split first half and second half
            edge_0 = self._densify_edge(edge_0, thres)
            edge_1 = self._densify_edge(edge_1, thres)
            # merge the two halves when recursion is done
            return edge_0[:-1] + edge_1
        # if distance threshold is respected: return edge
        else: return edge
        
    def estimate_elevations(self, fpath):
        if self.has_preliminary_z():
            print('Preliminary elevations have already been generated.')
            return
        print('\nSTARTING PRELIMINARY ELEVATION ESTIMATION')
        # import and subsample Lidar
        print('IMPORTING AHN3 TILE')
        self.ahn3, self.las_header = las_reader(fpath)
        self.ahn3 = self.ahn3[::3]
        # build a list of all NBRS vertex counts and vertices -
        # per-NBRS lists are needed for smoothing, and a
        # completely flat list of all vertices is needed
        # for the initial KD-tree query
        print('FLATTENING ALL NBRS')
        self.nbrs_vxnos, self.nbrs_revs, flat_vxs = {}, {}, []
        for nbrs_id in self.get_ids():
            nbrs_geom = self.get_geom(nbrs_id)
            # although the wegvak IDs are in the correct order
            # in NBRS lists in self.nbrs, the LineString
            # coordinates themselves are often reversed
            # randomly in NWB, meaning that we need to check for
            # reversals while we are flattening the lists
            # NOTE: although it would be much simpler to
            # do all this with the merged NBRS geometries,
            # the goal is to be able to edit the ORIGINAL
            # NWB wegvakken - unfortunately the geometries in
            # self.nbrs_geoms contain snapped vertices and auto-
            # reversed wegvakken, and tracking these throughout
            # the below procedure would be even more difficult
            # than starting from scratch...
            revs = np.full(len(nbrs_geom), False)
            # special case: NBRS with only one wegvak
            if len(nbrs_geom) == 1:
                self.nbrs_vxnos[nbrs_id] = [len(nbrs_geom[0].coords)]
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
            # based on rectifying the first wegvak above,
            # rectify the order of all subsequent ones -
            # when building the flat vertex lists, we must
            # discard the first vertex of each wegvak, as
            # it matches the last vertex of the previous...
            nbrs_vxnos = [len(nbrs_geom[0].coords)]
            for wvk_ix in range(len(nbrs_geom) - 1):
                if (np.round(nbrs_geom[wvk_ix + 1].coords[0], 1) ==
                    np.round(nbrs_vxs[-1], 1)).any():
                        wvk_vxs = list(nbrs_geom[wvk_ix + 1].coords[1:])
                else:
                    wvk_vxs = nbrs_geom[wvk_ix + 1].coords[::-1][1:]
                    revs[wvk_ix + 1] = True
                nbrs_vxnos.append(len(wvk_vxs)); nbrs_vxs += wvk_vxs
            self.nbrs_vxnos[nbrs_id] = nbrs_vxnos
            self.nbrs_revs[nbrs_id] = list(revs)
            flat_vxs += nbrs_vxs
        # build KD-trees from Lidar points and NBRS vertices
        print('BUILDING 2D KD-TREES')
        nbrs_tree = cKDTree(flat_vxs)
        lidar_tree = cKDTree(self.ahn3[:,1:3])
        # get all the Lidar points that are very close to NBRS vertices
        print('FINDING NEARBY LIDAR POINTS')
        nbrs_ix = nbrs_tree.query_ball_tree(lidar_tree, self.thres * 0.2)
        # link 3D Lidar points to 2D-projected Lidar points
        xy = [tuple(arr) for arr in self.ahn3[:,1:3]]
        link = dict(zip(xy, self.ahn3[:,3]))
        print('ESTIMATING PRELIMINARY ELEVATIONS')
        # initialise new 3D geometry column and activate it
        self.nwb['geometry_simpleZ'] = None
        self.set_geocol('geometry_simpleZ')
        # compute a rough z-value for each NBRS vertex,
        # perform the smoothing workflow when an NBRS is
        # completed, and add 3D geometries to GeoDataFrame
        self.flat_vxs, self.z_wasnan, first = [], {}, 0
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            # fetch the nearby Lidar points of each NBRS vertex
            # and compute its elevation as their median
            zs = []
            for pt_ix in nbrs_ix[first:last]:
                if pt_ix:
                    pt_xy = [xy[ix] for ix in pt_ix]
                    pt_z = [link[xy] for xy in pt_xy]
                    zs.append(np.median(pt_z))
                else: zs.append(np.NaN)
            # assemble NBRS 3D coordinates into an array
            nbrs_vxs = flat_vxs[first:last]
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
            self.flat_vxs += nbrs_vxs_z.tolist()
            slices = [[nbrs_vxs_z[start:end], list(np.isnan(zs))]
                      for start, end in zip(starts, ends)]
            z_wasnan = []
            for wvk_id, sliced, rev in zip(self.nbrs[nbrs_id], slices,
                                           self.nbrs_revs[nbrs_id]):
                if rev: sliced[0] = np.flip(sliced[0], axis = 0)
                self.set_wvk_geom(wvk_id, LineString(sliced[0]))
                z_wasnan += sliced[1]
            self.z_wasnan[nbrs_id] = z_wasnan
            first = last
        print('FINISHED PRELIMINARY ELEVATION ESTIMATION')

    def segment_lidar(self, fpath):
        """Performs DTB-assisted Lidar segmentation of the AHN3 tile.
        The steps of the procedure are as follows:
            1. Densify DTB lines and create a pseudo-point-
               cloud from their vertices.
            2. Build KD-trees with the NBRS vertices, AHN3 points
               and DTB points.
            3. Perform least-squares plane fitting on the AHN3
               points close to NBRS vertices. Save basic statstics
               of relative and absolute plane orientation to be
               able to detect when trend becomes unstable.
               In case of instability, if DTB exists and is stable
               locally, then adjust the plane to conform to it.
               If not, then try to use the previous (stable)
               plane, with a set number of maximum attempts.
               If the number of maximum attempts are exceeded,
               skip iterations until plane becomes stable again
               by itself, or DTB becomes avaialable. This may
               give rise to data gaps in rare cases.
            4. Save the points that were added to subclouds along
               with attributes to indicate which subcloud they are
               part of, and whether they came from AHN3 or DTB.
        NOTE: For the time being, no special workflow was implemented
        for cases where DTB-based assistance is unavailable. This may
        give rise to data gaps in rare cases, more frequently in the
        case of provincial roads (where DTB generally does not exist).
        A workflow is planned for implementation that will split NBRS
        at locations that mark the boundary of plane instability and
        unavailability of DTB, so that the rest of NBRS can still be
        fitted accurately by isolating no-data regions.
        """
        print('\nSTARTING POINT CLOUD SEGMENTATION')
        print('IMPORTING DTB')
        dtb, r, dtb_vxs = gpd.read_file(fpath), self.thres, []
        dtb = dtb[~dtb['geometry'].isnull()]
        if len(dtb) != 0:
            dtb['geometry_type'] = dtb['geometry'].apply(get_geom_type)
            dtb = dtb[dtb['geometry_type'] == 'LineString']
        self.dtb = dtb
        print('GENERATING DTB POINT CLOUD')
        if len(self.dtb) != 0:
            surf_cats = 'verflijn', 'verfstippellijn', 'blokmarkering'
            dtb_surf = self.dtb[self.dtb['OMSCHR'].isin(surf_cats)]
            for lstr in dtb_surf['geometry'].values:
                densified = self._densify_lstr(lstr, 0.2 * r)
                # in cases of broken geometries (for instance
                # linear rings stored as LineString objects)
                # None will be returned - handle this here
                if densified: dtb_vxs += densified[0].coords[:]
            dtb_vxs = np.array(dtb_vxs)
        print('BUILDING 3D KD-TREES')
        nbrs_tree = cKDTree(self.flat_vxs)
        if len(dtb_vxs) > 0: dtb_tree = cKDTree(dtb_vxs)
        lidar_tree = cKDTree(self.ahn3[:,1:])
        # perform the aggregated NBRS-AHN3 query
        nbrs_ix = nbrs_tree.query_ball_tree(lidar_tree, r)
        print('SEGMENTING POINT CLOUD')
        first = 0
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            planes, lgroups = [], []
            # fit a plane on each group of AHN3 points that
            # were found to close to NWB vertices
            for pt_ix in nbrs_ix[first:last]:
                lgroups.append(np.array([])); planes.append(())
                if len(pt_ix) > 10:
                    pts = self.ahn3[:,1:][pt_ix]; lgroups[-1] = pts
                    if len(pt_ix) > 20: planes[-1] = (planefit_lsq(pts))
            # the main segmentation algorithm starts below
            subcloud, subcloud_dtb, i = set(), set(), 0
            prev_plane, prev_dist_p, prev_med_z = None, None, None
            for vx, z_wasnan, lgroup, plane in zip(self.flat_vxs[first:last],
                                                   self.z_wasnan[nbrs_id],
                                                   lgroups, planes):
                # if no plane could be fitted or we already know there
                # was an AHN3 data gap from preliminary elevation
                # estimation, then immediately flag for assistance
                need_assist = not plane or z_wasnan
                if not need_assist:
                    # compute close-by AHN3 points' distances
                    # to fitted plane and compute basic descriptors
                    dists = np.array([dist_toplane(pt, *plane)
                                      for pt in lgroup])
                    # standard deviation of distances from plane
                    std = np.std(dists)
                    # distance of plane from the NBRS vertex
                    dist_p = dist_toplane(vx, *plane)
                    # median of elevation of points
                    med_z = np.median(lgroup[:,2])
                    # special case: if first vertex of NBRS, then
                    # initialise rolling descriptors
                    if not prev_dist_p:
                        prev_dist_p, prev_med_z = dist_p, med_z
                    # if minimum values are reached, then compute
                    # percentage change of descriptors relative
                    # to the previous iteration
                    dp, dz = 0, 0
                    if abs(dist_p - prev_dist_p) > 0.2:
                        dp = abs(1 - dist_p / prev_dist_p)
                    if abs(med_z - prev_med_z) > 0.2:
                        dz = abs(1 - med_z / prev_med_z)
                    # if instability in the plane fitting is supected,
                    # flag for assist
                    #if nbrs_id == 2: print(vx, std, dz, dp)
                    need_assist = std > 0.1 * r or (dz > 0.5 or dp > 0.5)
                if need_assist:
                    pts = np.array([])
                    if len(dtb_vxs) > 0:
                        # if a previous plane exists, move the location of
                        # the forhcoming DTB query closer to the plane
                        ctr = vx.copy()
                        if prev_plane: ctr[2] = prev_med_z
                        # perform individual DTB query
                        pt_ix = dtb_tree.query_ball_point(ctr, 0.4 * r)
                        pts = dtb_vxs[pt_ix]
                        # keep only those DTB points, which are relatively
                        # conformant with the previous plane, to avoid
                        # accidental inclusion of underlying/overlying
                        # road surface markings
                        if prev_plane:
                            dists = np.array([dist_toplane(pt, *prev_plane)
                                              for pt in pts])
                            pts = pts[dists < 0.1 * r]
                    # if DTB assist is possible...
                    if len(pts) > 2:
                        # refit plane onto DTB-defined surface
                        plane = planefit_lsq(pts)
                        # re-compute AHN3 point distances to plane
                        dists = np.array([dist_toplane(pt, *plane)
                                          for pt in lgroup])
                        lclose = lgroup[dists < 0.1 * r]
                        # if many are close to the new plane, then
                        # refit the plane once again to the surface
                        # defined by the AHN3 points that conformed
                        # well with the DTB-defined surface -
                        # this is necessary because DTB is not always
                        # perfectly conformant with the AHN3-defined
                        # road surface, I have observed up to 1-1.5 m
                        # deviations, e.g. at the Knoppunt Ridderkerk
                        if len(lclose) > 20:
                            plane = planefit_lsq(lclose)
                            prev_med_z = np.median(lclose[:,2])
                        else: prev_med_z = np.median(pts[:,2])
                        prev_plane = plane
                        prev_dist_p = dist_toplane(vx, *plane)
                        # also save DTB points when DTB assistance
                        # completed succesfully
                        for pt in pts: subcloud_dtb.add(tuple(pt))
                    # if DTB assist is not possible...
                    else:
                        # if plane-reverting assistance had already
                        # run 3 times consequently before, then
                        # "give up" and skip to next NBRS vertex
                        if i > 2 or not prev_plane: continue
                        # else, try reverting the plane   
                        plane = prev_plane
                        dists = np.array([dist_toplane(pt, *prev_plane)
                                          for pt in lgroup])
                        i += 1
                else:
                    # roll the plane and the descriptors
                    prev_plane = plane
                    prev_dist_p, prev_med_z = dist_p, med_z
                    i = 0
                # pre-select those AHN3 points, which were very close
                # to the fitted plane
                lclose = lgroup[dists < 0.05 * r]
                for pt in lclose: subcloud.add(tuple(pt))
            # compile NBRS's subcloud from the pre-selected AHN3 and
            # DTB points added in previous steps - each point may only
            # be added once *per NBRS*
            if subcloud:
                # perform a quick outlier filtering on the subcloud
                subcloud = np.array(list(subcloud))
                subcloud_tree = cKDTree(subcloud)
                subcloud_tree_len = subcloud_tree.n
                _, ixs = subcloud_tree.query(subcloud, 3,
                                             distance_upper_bound = 0.8,
                                             workers = -1)
                subcloud_mask = []
                for pt_ixs in ixs:
                    if len(pt_ixs[pt_ixs != subcloud_tree_len]) == 3:
                        subcloud_mask += [True]
                    else:
                        subcloud_mask += [False]
                subcloud = subcloud[subcloud_mask]
                subcloud = np.c_[subcloud, np.full(len(subcloud), 0)]
            else: subcloud = np.empty((0, 4))
            if subcloud_dtb:
                subcloud_dtb = np.c_[np.array(list(subcloud_dtb)),
                                     np.full(len(subcloud_dtb), 1)]
                subcloud = np.concatenate((subcloud, subcloud_dtb))
            self.nbrs_subclouds[nbrs_id] = subcloud
            first = last
        print('FINISHED POINT CLOUD SEGMENTATION')
        
    def estimate_edges(self):
        """Method to construct crude preliminary edges estimates
        based on NBRS's NWB locations and subclouds. Cross-sections
        are constructed in NWB vertices (and densified vertices) and
        the elevations of nearby subcloud points are used to
        transpose them into 3D (they are first densified to enable
        better point cloud sampling). Line fitting is then used to
        identify an edge point on both side of NWB in each cross-
        section. The first conformant point is chosen from the
        ends of the cross-sections, progressing inwards (towards
        NWB). If no such point could be found, the first cross-
        section point (again, progressing inwards) with an existing
        elevation is picked. In the absence of such a point, the
        cross-section is "deleted".
        """
        print('\nSTARTING PRELIMINARY ROAD EDGE ESTIMATION')
        edgegeoms_l, edgegeoms_r, crossgeoms, first = [], [], [], 0
        for nbrs_id in self.get_ids():
            # gather necessary NBRS NWB and subcloud data
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = self.flat_vxs[first:last]
            subcloud = self.nbrs_subclouds[nbrs_id]
            # special case: NBRS has empty subcloud
            if len(subcloud) == 0:
                edgegeoms_l += [LineString()]
                edgegeoms_r += [LineString()]
                crossgeoms += [MultiLineString()]
                first = last
                continue
            # build 2D KD-tree from subcloud and create hashed
            # link to the elevations of the tree's points
            lidar_tree = cKDTree(subcloud[:,:2])
            lidar_tree_len = lidar_tree.n
            xy = [tuple(arr) for arr in subcloud[:,:2]]
            link = dict(zip(xy, subcloud[:,2]))
            # generate the cross-sections
            vx0, vx1 = nbrs_vxs[0][:2], nbrs_vxs[1][:2]
            # special case: cross-section on first NBRS vertex
            cross_edges = [get_cross(vx1, vx0, None, self.thres, 1)]
            # general case
            for vx2 in nbrs_vxs[2:]:
                cross_edges.append(get_cross(vx0, vx1, vx2[:2],
                                             self.thres, -1))
                vx0, vx1 = vx1, vx2[:2]
            # special case: cross-section on last NBRS vertex
            cross_edges.append(get_cross(vx0, vx1, None,
                                         self.thres, -1))
            # densify the cross-sections
            cross_edges = [self._densify_edge(edge, 0.3)
                           for edge in cross_edges
                           if edge is not None]
            cross_vxs = np.concatenate(cross_edges)
            # fetch the Lidar points necessary to transpose
            # the cross-sections into 3D
            _, ixs = lidar_tree.query(cross_vxs, 10,
                                      distance_upper_bound = 0.2,
                                      workers = -1)
            # transpose each (densified) cross-section vertex
            # into 3D using the median of the elevations of
            # the selected close-by subcloud points
            cross_zs = []
            for pt_ixs in ixs:
                pt_xy = [xy[ix] if ix != lidar_tree_len
                         else (None, None) for ix in pt_ixs]
                pt_zs = [link[pt] if None not in pt
                         else None for pt in pt_xy]
                pt_zs = np.array(pt_zs, dtype = float)
                pt_zs = pt_zs[~np.isnan(pt_zs)]
                if len(pt_zs) == 0: cross_zs.append(None)
                else: cross_zs.append(np.median(pt_zs))
            cross_vxs = np.array(np.c_[cross_vxs, cross_zs], dtype = float)
            # estimate crude edge locations
            mid = len(cross_edges[0]) // 2
            edges_l, edges_r, crosses = [], [], []
            for vxs in np.split(cross_vxs, len(cross_edges)):
                outliers = filter_outliers(vxs, 1, True, 1, 0.15)
                if outliers is None: continue
                missing = np.isnan(vxs[:,2])
                edge_l, edge_r = None, None
                # try looking for the first valid cross-section
                # vertex (one with a non-outlier elevation),
                # progressing inwards
                for i in range(mid):
                    if not outliers[i] and not missing[i]:
                        edge_l = list(vxs[i]); break
                for i in range(1, mid + 1):
                    if not outliers[-i] and not missing[-i]:
                        edge_r = list(vxs[-i]); break
                # if insuccesful, try just getting the first one
                # that has an elevation
                if not edge_l:
                    for i in range(mid):
                        if not missing[i]:
                            edge_l = list(vxs[i]); break
                if not edge_r:
                    for i in range(1, mid + 1):
                        if not missing[-i]:
                            edge_r = list(vxs[-i]); break
                # if an edge point could be found on both sides
                # of the NWB, then extend the road edge with
                # them, and construct the cross-section geometry
                if edge_l and edge_r:
                    if not dist_topoint(edge_l, edge_r) < 2:
                        edges_l.append(edge_l)
                        edges_r.append(edge_r)
                        crosses += [LineString((edge_l, edge_r))]
            # construct the NBRS's edge and cross-section geometries
            edgegeoms_l += [LineString(edges_l)]
            edgegeoms_r += [LineString(edges_r)]
            crossgeoms += [MultiLineString(crosses)]
            first = last
        
        # create GeoDataFrames with all the resulting road
        # edge and cross-section geometries
        sides = ['left'] * self.nbrs_count + ['right'] * self.nbrs_count
        edge_dict = {'NBRS_ID': self.get_ids() + self.get_ids(),
                     'SIDE': sides, 'geometry': edgegeoms_l + edgegeoms_r}
        cross_dict = {'NBRS_ID': self.get_ids(), 'geometry': crossgeoms}
        self.nbrs_edges = gpd.GeoDataFrame(edge_dict, crs="EPSG:28992")
        self.nbrs_crosses = gpd.GeoDataFrame(cross_dict, crs="EPSG:28992")
        print('\nFINISHED PRELIMINARY ROAD EDGE ESTIMATION')
        

# testing configuration, only runs when script is not imported
if __name__ == '__main__':
    nwb_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_nwb_2.shp'
    dtb_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_dtb.shp'
    ahn_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_2_26_clipped.las'
    simpleZ_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_simpleZ.shp'
    subclouds_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_subclouds.las'
    edges_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_edges.shp'
    crosses_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_crosses.shp'
    roads = nbrs_manager(nwb_fpath)
    roads.generate_nbrs()
    roads.densify(10)
    roads.estimate_elevations(ahn_fpath)
    roads.write_all(simpleZ_fpath)
    roads.segment_lidar(dtb_fpath)
    roads.write_subclouds(subclouds_fpath)
    roads.estimate_edges()
    roads.write_edges(edges_fpath, crosses_fpath)
    #roads.plot(1, True)
    #roads.plot_all()
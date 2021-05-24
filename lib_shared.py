#########################################
## 3D-NWB DISSERTATION PROJECT SCRIPTS ##
##  KRISTOF KENESEI, STUDENT 5142334   ##
##    K.Kenesei@student.tudelft.nl     ##
#########################################


import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, LineString, MultiLineString
from shapely import ops
from laspy.file import File
import rasterio
from rasterio.transform import Affine
import startin


def calc_angle(v0, v1):
    """Computes the angle between two vectors.
    """
    v0, v1 = v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)
    return np.arccos(np.round(np.dot(v0, v1), 8))

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
        out_file.define_new_dimension(name = "ORIGIN", data_type = 3,
                                      description = 'ORIGIN')
        out_file.define_new_dimension(name = "NBRS_ID", data_type = 3,
                                      description = 'NBRS_ID')
        out_file.define_new_dimension(name = "PART_ID", data_type = 3,
                                      description = 'PART_ID')
        out_file.x = pts[:,0]
        out_file.y = pts[:,1]
        out_file.z = pts[:,2]
        out_file.ORIGIN = pts[:,3].astype(int)
        out_file.NBRS_ID = pts[:,4].astype(int)
        out_file.PART_ID = pts[:,5].astype(int)

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

def calc_M(vx, tr_vxs):
    """Function to compute the M parameter for the
    TIN-linear error propagation formula.
    """
    x0, y0 = tr_vxs[0][0], tr_vxs[0][1]
    x1, y1 = tr_vxs[1][0], tr_vxs[1][1]
    x2, y2 = tr_vxs[2][0], tr_vxs[2][1]
    L0 = x0 * y1 + x2 * y0 + x1 * y2
    L1 = x2 * y1 + x0 * y2 + x1 * y0
    L = L0 - L1
    a0, a1, a2 = (y1 - y2) / L, (y2 - y0) / L, (y0 - y1) / L
    b0, b1, b2 = (x2 - x1) / L, (x0 - x2) / L, (x1 - x0) / L
    c0 = (x1 * y2 - x2 * y1) / L
    c1 = (x2 * y0 - x0 * y2) / L
    c2 = (x0 * y1 - x1 * y0) / L
    tr_mx = np.array([[a0, a1, a2],
                      [b0, b1, b2],
                      [c0, c1, c2]])
    vx_vc = np.array([vx[0], vx[1], 1])
    m0, m1, m2 = np.dot(vx_vc, tr_mx)
    return m0 ** 2 + m1 ** 2 + m2 ** 2

def calc_angleComponents(tin, tr):
    """Function to compute the angle component parameters
    for the TIN-linear error propagation formula.
    """
    # the workflow is based on finding a few neighbouring
    # triangles, fitting a plane and computing the angles
    # relative to the plane - much like the workflow used
    # in my TIN growing iterations
    trs = [tin.incident_triangles_to_vertex(ix) for ix in tr]
    init_ixs = set(np.concatenate(trs).flatten())
    nbr_trs = [tin.incident_triangles_to_vertex(tix)
               for tix in list(init_ixs)]
    nbr_ixs = set(np.concatenate(nbr_trs).flatten()) | init_ixs
    r_vxs = [tin.get_point(tix) for tix in list(nbr_ixs)]
    d, norm = planefit_lsq(np.array(r_vxs))
    angle_x = np.arcsin(np.dot(norm, np.array([1, 0, 0])))
    angle_y = np.arcsin(np.dot(norm, np.array([0, 1, 0])))
    return abs(angle_x), abs(angle_y)

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

def densify_lstr(lstr, thres, origins = None):
    """Densifies an LineString.
    Calls densify_edge() for each edge.
    Also handles the generation of "origins", a
    variable that track which vertices came from
    densification, and which ones from NWB.
    """
    orig_coords = lstr.coords; orig_len = len(orig_coords)
    i, j = 0, 0
    while i < orig_len - 1:
        vxs = orig_coords[i:i+2]
        # start the recursive vertex densification
        # of the selected edge of the wegvak
        dense = LineString(densify_edge(vxs, thres))
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

def densify_edge(edge, thres):
    """Recursively densifies an edge.
    It splits the edge while opening new frames on the
    stack, and merges them when returning out of the frames.
    """
    dst = dist_topoint(edge[0], edge[1])
    # if distance threshold is not respected: split
    if dst > thres:
        pt = tuple(np.mean(edge, axis = 0))
        edge_0, edge_1 = (edge[0], pt), (pt, edge[1])
        # recursively split first half and second half
        edge_0 = densify_edge(edge_0, thres)
        edge_1 = densify_edge(edge_1, thres)
        # merge the two halves when recursion is done
        return edge_0[:-1] + edge_1
    # if distance threshold is respected: return edge
    else: return edge

def filter_outliers(vxs, deg = 6, only_detect = False,
                    thres = 1, prox = None, only_nan = False):
    """ Performs 1D-smoothing (or alternatively, just outlier
    detection) on a series of input points that form a 3D line.
    Fits a polynomial of the desired degree to obtain a model,
    and uses the data-model errors to identify outliers.
    If interpolation is desired, it uses numpy functions to find
    better values at the locations of outliers, and at locations
    where elevation was missing originally. Edits the input in-place.
    Parameters:
    - 'vxs': the input points as a 3 by 'n' array
    - 'deg': the degree of the polynomial to fit
    - 'only_detect': set to True if a boolean mask is desired
                     instead of in-place interpolation
    - 'thres': the number of standard deviation outside of
               which a point is to be considered an outlier
    - 'prox': ratio of input points to fit the model on - should
              be between 0 and 1, and bear in mind that the points
              farthest away from halfway through the profile will
              be discarded, not at the beginning/end of the profile
    """
    zs = vxs[:,2].copy()
    # if only points in the proximity of halfway trough the
    # profile are desired, perform some slicing operations
    # to restrict operations to this part
    if prox:
        mid, cut = len(zs) // 2, round(len(zs) * prox // 2)
        zs[:mid - cut] = np.NaN; zs[mid + cut + 1:] = np.NaN
    if ((not prox and len(zs[np.isnan(zs)]) / len(zs) > 0.75)
         or (prox and len(zs[~np.isnan(zs)]) < deg + 1)):
        if only_detect == False:
            # zero out the entire NBRS if it has too many NaNs
            vxs[:,2] = np.full(len(zs), np.NaN)
        return
    # find model polynomial and compute model values
    # exclude locations where the elevation is missing
    diffs = np.insert(np.diff(vxs[:,:2], axis = 0), 0, 0, axis = 0)
    # the x values of the 1D profile are based on 2D
    # vertex distances
    dsts = np.cumsum(np.sqrt((diffs ** 2).sum(axis = 1)))
    zs_notnan, dsts_notnan = zs[~np.isnan(zs)], dsts[~np.isnan(zs)]
    coeffs, _ = np.polynomial.polynomial.polyfit(dsts_notnan, zs_notnan,
                                                 deg, full = True)
    model_zs = np.polynomial.polynomial.polyval(dsts, coeffs)
    # compute the data-model errors and STD
    errors = np.abs(vxs[:,2] - model_zs); mask = errors.copy()
    if prox: mask[:mid - cut], mask[mid + cut + 1:] = np.NaN, np.NaN
    std = np.std(mask[~np.isnan(mask)])
    if prox and std < 0.05: std = 0.05
    elif std < 0.4: std = 0.4
    # do not perform smoothing if only filtering is desired
    if only_detect: return errors > thres * std
    # create an index of vertices where the elevation is
    # missing, or was identified as an outlier
    if not only_nan: out_ix = (errors > thres) | np.isnan(errors)
    else: out_ix = np.isnan(errors)
    # if any were found, re-fit model on inliers and
    # replace them with interpolated values
    if len(out_ix) > 0:
        zs[out_ix] = np.interp(dsts[out_ix], dsts[~out_ix], zs[~out_ix])
        vxs[:,2] = zs

def utility_tin(pts, bounds, max_dh, max_angle, r,
                pts_inserted = None, seeds = None):
    """Utility function that can construct a TIN from the
    segmented point cloud of an NBRS. It needs either the
    preliminary edge estimates or the optimised edges
    to work (it controls the seeding of the initial TIN,
    and then its extension). In addition to using it to
    construct a TIN from points within the NBRS edges,
    the nbrs_manager class also uses it to optionally
    extend it with points outside the edges. The procedure
    still needs edges to be inserted into the TIN,
    but we do not wish to keep these in the TIN. However,
    point removal in startin does not seem to work
    reliably, so in each iteration the TIN is constructed
    anew, and the function returns a list containing
    inserted (and meaningful) points rather than the
    startin-based TIN itself.
    Arguments:
    - 'pts': the candidate points
    - 'bounds': the coordinates of a polygon that contains
                all candidate points
    - 'max_dh': TIN insertion elevation threshold (see the
                relevant nbrs_manager docstring for more)
    - 'max_angle': TIN insertion angle threshold (see the
                   relevant nbrs_manager docstring for more)
    - 'pts_inserted': if extension is desired, then the
                      points that were already inserted
                      into the TIN, in the order in which
                      they were previously inserted
    - 'seeds': the geometry from where the TIN construction
               is seeded, Lidar points very close to these
               points are inserted unconditionally
    """
    # initialise KD-tree from input points, and TIN
    tree = cKDTree(pts); tree_len = tree.n
    tin = startin.DT()
    # if initial TIN (between NBRS edges) is being
    # built, then seed it by unconditionally
    # inserting points around the "skeleton" of
    # the polygon created from the NBRS edges
    if pts_inserted is None:
        pts_inserted = []
        _, nbr_ixs = tree.query(seeds, 50,
                        distance_upper_bound = 1,
                        workers = -1)
        stack = set(nbr_ixs.flatten()) - {tree_len}
        used, buffer = stack.copy(), []
        while stack:
            pt = pts[stack.pop()]
            buffer += [pt]
            tin.insert_one_pt(*pt)
            pts_inserted.append(pt)
    # if this is not the first round (and extension
    # of a pre-existing TIN is desired), then first
    # re-construct the TIN from the already-inserted
    # points, then seed using the edges (or buffered
    # edges) from the last iteration
    else:
        for pti in pts_inserted: tin.insert_one_pt(*pti)
        pre_tree = cKDTree(pts_inserted)
        _, pre_ixs = pre_tree.query(seeds)
        # the seeds are the edges from the previous
        # iteration, and they are extruded to 3D using
        # the already-inserted points here
        seeds_z = []
        for vx, bix in zip(seeds, pre_ixs):
            seeds_z.append((*vx[:2], pts_inserted[bix][2]))
        used, buffer = set(), seeds_z.copy()
    # insert the bounds into the TIN, keeping them at
    # a constant elevation of zero - keep track of their
    # TIN indices, so that they can be excluded from
    # elevation discrepancy computations later on
    bound_ixs = set()
    for bound_vx in bounds: bound_ixs.add(tin.insert_one_pt(*bound_vx))
    # the outer iteration is based on a buffer, which
    # contains all points inserted in the previous iteration
    # of the outer loop - in the first iteration it contains
    # the seed points
    while buffer:
        # candidate points are fetched from the
        # neighbourhood of buffer points
        _, nbr_ixs = tree.query(buffer, 50,
                                distance_upper_bound = r,
                                workers = -1)
        # the inner loop is a stack-based one
        # all neighbours of buffer points are considered for
        # insertion, the variable "used" records which
        # points were inserted and should not be
        # considered again
        stack, buffer = set(nbr_ixs.flatten()) - {tree_len} - used, []
        while stack:
            ix = stack.pop(); pt = pts[ix]
            # get the triangle the candidate is located in
            # because all iterations of this function insert
            # a boundary encompassing all candidate points,
            # this operation is guaranteed to always work
            tri = tin.locate(*pt[:2])
            # identify boundary points among the vertices
            # of the located triangle
            vx_ixs = np.array([tix for tix in tri if tix not in bound_ixs])
            vxs = np.array([tin.get_point(tix) for tix in vx_ixs])
            dh, grow = None, False
            # if the triangle has a boundary vertex among its vertices
            # then consider the operation a "growing" operation
            if len(vx_ixs) < 3: grow = True
            # else, consider it a "growing" operation only, if the
            # area or the circumference of the triangle indicates
            # that it probably does not belong to the road surface
            # (i.e. it has a large area or long circumference)
            else:
                cross = np.cross(vxs[1] - vxs[0], vxs[2] - vxs[0])
                area = dist_topoint([0, 0, 0], cross) / 2
                a = dist_topoint(vxs[0], vxs[1])
                b = dist_topoint(vxs[1], vxs[2])
                c = dist_topoint(vxs[2], vxs[0])
                if area > 50 or a + b + c > 20: grow = True
            # if this is not a "growing" operation, interpolate
            # in the TIN to get the elevation deviation
            if not grow: dh = abs(pt[2] - tin.interpolate_laplace(*pt[:2]))
            # if this is a "growing" operation, then compute the
            # elevation difference as the mean difference relative
            # to the elevations of the TIN vertices, excluding the
            # boundary vertex
            # NOTE: this is the only way the algorithm can grow
            # the TIN beyond the current road surface extents
            elif len(vx_ixs) == 2:
                trs0 = tin.incident_triangles_to_vertex(vx_ixs[0])
                trs1 = tin.incident_triangles_to_vertex(vx_ixs[1])
                init_ixs = set(np.concatenate((trs0, trs1)).flatten())
                nbr_trs = [tin.incident_triangles_to_vertex(tix)
                           for tix in list(init_ixs)]
                nbr_ixs = set(np.concatenate(nbr_trs).flatten()) | init_ixs
                nbr_ixs = nbr_ixs - bound_ixs
                r_vxs = [tin.get_point(tix) for tix in list(nbr_ixs)]
                r_vxs = np.array([tin.get_point(tix)
                                  for tix in list(nbr_ixs)])
                dh = dist_toplane(pt, *planefit_lsq(r_vxs))
            # perform the angle test - if the triangle had a boundary
            # vertex among its vertices, ignore that vertex for the
            # purposes of the angle test (it is at zero elevation)
            if dh and dh < max_dh:
                insert = True
                for vx in vxs:
                    dd = dist_topoint(pt[:2], vx[:2])
                    if not dd or abs(np.arctan(dh / dd)) > max_angle:
                        insert = False; break
                # if candidate passed both the elevation difference
                # and angle tests, then insert into TIN, add to the
                # buffer and mark as having been used already
                if insert:
                    used.add(ix); buffer += [pt]
                    tin.insert_one_pt(*pt); pts_inserted.append(pt)
    return pts_inserted

def write_geotiff(raster, origin, size, fpath):
    """Writes data in an n by n numpy array to disk as a
    GeoTIFF raster. The header is based on the raster array
    and a manual definition of the coordinate system and an
    affine transform.
    """
    transform = (Affine.translation(origin[0], origin[1])
                 * Affine.scale(size, size))
    with rasterio.Env():
        with rasterio.open(fpath, 'w', driver = 'GTiff',
                           height = raster.shape[0],
                           width = raster.shape[1],
                           count = 1,
                           dtype = rasterio.float32,
                           crs = 'EPSG:28992',
                           transform = transform
                           ) as out_file:
            out_file.write(raster.astype(rasterio.float32), 1)
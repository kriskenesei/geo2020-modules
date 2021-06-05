#########################################
## 3D-NWB DISSERTATION PROJECT SCRIPTS ##
##  KRISTOF KENESEI, STUDENT 5142334   ##
##    K.Kenesei@student.tudelft.nl     ##
#########################################


import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString, MultiLineString, Polygon
from shapely import ops
from scipy.spatial import cKDTree
from skimage.segmentation import active_contour
import matplotlib as mpl
import lib_shared
import startin


def conv_tovx(lstr):
    """Converts the input shapely LineString to a shapely
    MultiPoint (used in geopandas column generation when
    plotting vertices in addition to lines in .plot()).
    """
    return MultiPoint(lstr.coords)

def make_geom_idcol(lstr):
    """Returns a list containing the string 'nbw' as many times
    as the input LineString has vertices. This is used to initialise
    a column in the NWB GeoDataFrame where the software can
    indicate what the source each the vertex is.
    """
    return list(np.full(len(lstr.coords), 'nwb'))

def get_geom_type(lstr):
    """Returns the shapely geometry type of the input geometry.
    This is used in GeoPandas column generation, in turn used
    to filter out buggy geometries in the input.
    """
    return lstr.geom_type


class nbrs_manager:
    
    def __init__(self, fpath):
        """Imports NWB, or a cropped NWB "tile" upon
        initialisation. NWB is stored as a geopandas
        GeoDataFrame. The intended usage is as follows:
            1.  Instantiate class with the path to the NWB tile.
            2.  Invoke .generate_nbrs() with either algorithm
                to build Non-Branching Road Segments (NBRS).
            3.  Optionally, invoke .densify() with a threshold
                in metres to also perform vertex densification.
                Highly recommended, as NWB's sampling is coarse.
            4.  Invoke .estimate_elevations() with aligned AHN3
                tile to perform an estimation of NWB elevations.
            4.  Optionally, plot either a single NBRS
                via .plot() or all NBRS via .plot_all()
            5.  Optionally, write NBRS geometry to disc by
                invoking .write_all().
            6.  Invoke .segment_lidar() with aligned DTB tile
                to perform point cloud segmentation (pre-
                selection of AHN3 points that fall on the road
                surfaces belonging to specific NBRS).
                Post-segmentation, the sublocuds can be found
                in .nbrs_subclouds, which needs to be indexed
                with an NBRS ID to return a subcloud.
            7.  Optionally, write to segmented point cloud to
                disk as a LAS file by invoking .write_subclouds().
            8.  Invoke .estimate_edges() to construct crude edge
                estimates. Currently, both the edge estimates
                and the cross-sections are stored as NBRS-level
                geometries in .nbrs_edges and .nbrs_crosses,
                which need to be indexed with an NBRS ID to
                return edges/cross-sections.
            9.  Optionally, write the edges and cross-sections
                to disk by invoking .write_edges().
            10. Optionally, run .optimise_edges() to perform
                active contour optimisation of the edges. The
                optimised edges (or "contours") are stored in
                .nbrs_contours. Fine-tuning the parametrisation
                may be necessary.
            11. Active contour optimisation generates attractor
                maps (rasters) for each NBRS. These can be optionally
                written to dis using .write_maps(), writing to a
                dedicated folder is recommended because each
                NBRS map (and type of map, if applicable) is
                written to a separate file. The rasters are
                written in the GeoTIFF format.
            12. Optionally, the optimised edges can also be
                written to disk using .write_contours, this is
                written to a single shapefile as usual.
            13. Invoke .build_tins() to construct the TIN surface
                models for each NBRS. It can use either the
                preliminary edges or the optimised ones.
                Fine-tuning the parametrisation may be necessary.
            14. Optionally, the TINs can be written to disk as
                OBJ files, one for each NBRS. An algorithm
                is included to remove most of the big triangles
                that fill the convex hull of the model, and
                some of the sliver triangles that appear around
                the road surface edges. One TIN is written at a
                time, as the method is intended primarily for
                debugging use and is not particularly fast.
            15. Invoke .interpolate_elevations() to compute the
                final 3D-NWB elevations. These are interpolated
                in the TINs produced in step 13, and continuity
                across NBRS intersections is enforced.
            16. Use .write_all() to write the 3D-NWB geometries
                to disk. The output is identical in its structure
                to that which can be exported in step 5, but the
                elevations are much more accurate because of the
                additional processing steps.
            17. Optionally, use generate_theoreticalerrors() to
                propagate input accuracy values to the output.
                These are stored in .wvk_z_errors.
            18. Optionally, use .write_origins() and .write_errors()
                to write the wegvak vertex origins, local density
                values and computed TIN-linear error propagation
                results to disk as CSV logfiles.
        Example calls are provided at the end of this script.
        """
        nwb = gpd.read_file(fpath); nwb['NBRS_ID'] = None
        nwb = nwb[~nwb['geometry'].isnull()]
        nwb['geometry_type'] = nwb['geometry'].apply(get_geom_type)
        nwb = nwb[nwb['geometry_type'] == 'LineString']
        self.nwb, self.thres = nwb.drop(['geometry_type'], 1), None
        self.wvk_count, self.jte = len(self.nwb.index), {}
        self.wvk_geoms = dict(zip(self.nwb['WVK_ID'],
                                  self.nwb['geometry']))
        self.wvk_origins = dict(zip(self.nwb['WVK_ID'],
                                    self.nwb['geometry'].apply(
                                        make_geom_idcol)))
        self.nbrs_count, self.nbrs_ids, self.nbrs_wvkn = 0, {}, {}
        self.nbrs_revs, self.nbrs_geoms, self.nbrs_subclouds,  = {}, {}, {}
        self.nbrs_parts, self.nbrs_edges, self.nbrs_crosses = {}, [], []
        self.nbrs_extents, self.nbrs_ress, self.nbrs_origins = {}, {}, {}
        self.wvk_z_origins, self.nbrs_maps, self.mapsize = {}, {}, 0
        self.ahn3, self.las_header, self.flat_vxs = None, None, []
        self.dtb, self.nbrs_tins, self.nbrs_dtb_ref_sets = None, {}, {}
        self.nbrs_edges = gpd.GeoDataFrame()
        self.nbrs_crosses = gpd.GeoDataFrame()
        self.nbrs_contours = gpd.GeoDataFrame()
        self.nbrs_cmap = None
    
    def is_empty(self):
        """Return True if NBRS have been generated, else return False.
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
        """Return the NBRS having the provided NBRS ID as a GeoDataFrame.
        """
        if self.is_empty(): print('NBRS first need to be generated.')
        available_ids = self.get_ids()
        if nbrs_id not in available_ids:
            print("NBRS not found. Valid IDs are: ", available_ids)
        else: return self.nwb[self.nwb['NBRS_ID'] == nbrs_id].copy()
    
    def get_geom(self, nbrs_id):
        """Return the geometries of a given NBRS. Since the
        wegvakken in self.nbrs_wvkn are stored in-order, this method
        also returns them in the correct order.
        """
        wvk_ids = self.nbrs_wvkn[nbrs_id]
        if wvk_ids: return [self.wvk_geoms[wvk_id] for wvk_id in wvk_ids]
    
    def get_wvk_geom(self, wvk_id):
        """Return the geometry of wegvak with the provided wegvak ID.
        """
        geom = self.wvk_geoms.get(wvk_id)
        if geom: return geom
        else: print('Wegvak not found.')
    
    def set_wvk_geom(self, wvk_id, new_geom):
        """Set the geometry of the wegvak having the provided wegvak ID.
        Both the hashed-storage geometry and the GeoDataFrame
        geometry are reset.
        """
        geom_col = self.nwb.geometry.name
        if wvk_id in self.wvk_geoms.keys():
            self.nwb.loc[self.nwb['WVK_ID'] == wvk_id, geom_col] = new_geom
            self.wvk_geoms[wvk_id] = new_geom
        else: print('Wegvak not found.')
        
    def set_geocol(self, new_colname):
        """Set the geometry column of the NWB GeoDataFrame to a new
        column. This is implemented, so that the class behaves the
        same way as the GeoDataFrame itself does.
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
        NBRS that run close to each other. To visualise
        results in 3D, please write them to file and open
        then in a suitable 3D viewer.
        """
        if self.is_empty():
            print('WARNING: NBRS have not been generated yet.')
        if not self.nbrs_cmap:
            self.nbrs_cmap = mpl.colors.ListedColormap(np.random.rand(256,3))
        self.nwb.plot(column = 'NBRS_ID', cmap = self.nbrs_cmap)        
        
    def write_all(self, fpath, to_drop = [], sparse = False):
        """Writes road geometry currently stored in the object.
        Running it after preliminary elevation estimation
        but before the final interpolation writes intermediate
        results. Running it after the final interpolation
        writes the final 3D road network.
        """
        if self.is_empty():
            print('WARNING: NBRS have not been generated yet.')
        else: to_drop += ['geometry']
        if not self.is_empty() and sparse == True:
            old_geocol = self.nwb.geometry.name
            self.nwb['sparse'] = None; self.set_geocol('sparse')
            for nbrs_id in self.get_ids():
                for wvk_id in self.nbrs_wvkn[nbrs_id]:
                    vxs = np.array(self.get_wvk_geom(wvk_id).coords[:])
                    origins = np.array(self.wvk_origins[wvk_id])
                    mask = origins[origins == 'nwb']
                    geom = LineString(vxs[mask])
                    self.nwb.loc[self.nwb['WVK_ID'] == wvk_id,
                                 'sparse'] = geom
            to_drop += ['geometry_accurateZ']
            self.nwb.drop(to_drop, 1).to_file(fpath)
            self.nwb.drop(['sparse'], 1, inplace = True)
            self.set_geocol(old_geocol)
        else: self.nwb.drop(to_drop, 1).to_file(fpath)
        
    def write_subclouds(self, fpath):
        """Writes all the subclouds that were generated as a
        LAS file, with the added NBRS ID and part ID information.
        """
        if not self.nbrs_subclouds:
            print('Subclouds have not been generated yet.'); return
        subclouds_out = []
        for nbrs_id, subclouds in self.nbrs_subclouds.items():
            for part_id, subcloud in subclouds.items():
                subclouds_out.append(np.c_[subcloud,
                                           np.full(len(subcloud), nbrs_id),
                                           np.full(len(subcloud), part_id)])
        lib_shared.las_writer(fpath, self.las_header,
                              np.concatenate(subclouds_out))
        
    def write_edges(self, fpath_edges, fpath_crosses):
        """Writes the crude edges (and cross-sections) that were
        generated. A file path for both the edges and the cross-
        sections needs to be provided.
        """
        if self.nbrs_edges.empty:
            print('Edges have not been generated yet.'); return
        self.nbrs_edges.to_file(fpath_edges)
        self.nbrs_crosses.to_file(fpath_crosses)
        
    def write_maps(self, fpath):
        """Writes the attractor maps of all NBRS.
        An NBRS may consist of parts, in which case multiple
        files per NBRS will be written. The file path should
        lead to a directory, the file names inside the
        directory will be auto-generated. The files will
        be written in the GeoTIFF format.
        """
        if not self.nbrs_maps:
            print('Maps have not been generated yet.'); return
        for nbrs_id, parts in self.nbrs_maps.items():
            for part_id, img in parts.items():
                orig = self.nbrs_origins[nbrs_id][part_id]
                fname = '{}_{}_{}.tif'.format(fpath, nbrs_id, part_id)
                lib_shared.write_geotiff(img, orig, self.mapsize, fname)
                
    def write_contours(self, fpath):
        """Writes the road surface contours that were generated.
        """
        if self.nbrs_contours.empty:
            print('Contours have not been generated yet.'); return
        self.nbrs_contours.to_file(fpath)
    
    def write_tin(self, fpath, nbrs_id):
        """Writes a single road surface model TIN to disk.
        An NBRS may consist of parts, in which case multiple
        files will be written, tagged by the part ID. The file
        path should lead to a directory, the file names inside
        the directory will be auto-generated. The files will
        be written in the OBJ format, and most of the large,
        non-meaningful triangles will be filtered out.
        """
        if not self.nbrs_tins:
            print('TINs have not been generated yet.'); return
        tins = self.nbrs_tins.get(nbrs_id)
        if not tins:
            print('NBRS has no TINs.'); return
        for part_id in range(len(self.nbrs_parts[nbrs_id])):
            tin = tins.get(part_id)
            if not tin: continue
            out_pts = np.array(tin.all_vertices())
            trs, out_trs = np.array(tin.all_triangles()), []
            for tr in trs:
                vxs = np.array([out_pts[tix] for tix in tr])
                cross = np.cross(vxs[1] - vxs[0], vxs[2] - vxs[0])
                area = lib_shared.dist_topoint([0, 0, 0], cross) / 2
                a = lib_shared.dist_topoint(vxs[0], vxs[1])
                b = lib_shared.dist_topoint(vxs[1], vxs[2])
                c = lib_shared.dist_topoint(vxs[2], vxs[0])
                if area < 50 and a + b + c < 20: out_trs.append(tr)
            fname = '{}_{}_{}.obj'.format(fpath, nbrs_id, part_id)
            with open(fname, "w") as file_out:
                for pt in out_pts[1:]:
                    file_out.write('v {} {} {}\n'.format(*pt))
                for tr in out_trs:
                    file_out.write('f {} {} {}\n'.format(*tr))
    
    def write_tins(self, fpath):
        """Writes all the road surface model TINs to disk.
        An NBRS may consist of parts, in which case multiple
        files will be written, tagged by the part ID. The file
        path should lead to a directory, the file names inside
        the directory will be auto-generated. The files will
        be written in the OBJ format, and most of the large,
        non-meaningful triangles will be filtered out.
        """
        if not self.nbrs_tins:
            print('TINs have not been generated yet.'); return
        for nbrs_id, tins in self.nbrs_tins.items():
            for part_id in range(len(self.nbrs_parts[nbrs_id])):
                tin = tins.get(part_id)
                if not tin: continue
                out_pts = np.array(tin.all_vertices())
                trs, out_trs = np.array(tin.all_triangles()), []
                for tr in trs:
                    vxs = np.array([out_pts[tix] for tix in tr])
                    cross = np.cross(vxs[1] - vxs[0], vxs[2] - vxs[0])
                    area = lib_shared.dist_topoint([0, 0, 0], cross) / 2
                    a = lib_shared.dist_topoint(vxs[0], vxs[1])
                    b = lib_shared.dist_topoint(vxs[1], vxs[2])
                    c = lib_shared.dist_topoint(vxs[2], vxs[0])
                    if area < 50 and a + b + c < 20: out_trs.append(tr)
                fname = '{}_{}_{}.obj'.format(fpath, nbrs_id, part_id)
                with open(fname, "w") as file_out:
                    for pt in out_pts[1:]:
                        file_out.write('v {} {} {}\n'.format(*pt))
                    for tr in out_trs:
                        file_out.write('f {} {} {}\n'.format(*tr))
    
    def write_origins(self, fpath):
        """Writes a CSV logfile of all the vertex elevation origins.
        The structure is "WVK_ID,ORIGINS".
        """
        if not self.wvk_z_origins:
            print('Origins have not been established yet.'); return
        with open(fpath, "w") as file_out:
            file_out.write('WVK_ID,ORIGINS\n')
            for wvk_id, origins in self.wvk_z_origins.items():
                file_out.write('{},{}\n'.format(wvk_id, np.array(origins)))
                
    def write_errors(self, fpath):
        """Writes a CSV logfile of all the vertex sampling densities
        and errors. The structure is "WVK_ID,DENSITY,ERROR".
        """
        if not self.wvk_z_errors:
            print('Accuracy has not been computed yet.'); return
        with open(fpath, "w") as file_out:
            file_out.write('WVK_ID,DENSITY,ERROR_M\n')
            for wvk_id, error in self.wvk_z_errors.items():
                dsy = np.round(self.wvk_density[wvk_id], 3)
                file_out.write('{},{},{}\n'.format(wvk_id, dsy,
                                                   np.round(error, 3)))
        
    def generate_nbrs(self, algorithm = 'geometric'):
        """Starts either NBRS generation algorithm.
        The two algorithms are: 'geometric', 'semantic'.
        """
        if not self.is_empty(): print('NBRS have already been generated.')
        elif algorithm == 'geometric': self._generate_nbrs_geometric()
        elif algorithm == 'semantic': self._generate_nbrs_semantic()
        else: print('Unknown algorithm. Available: "geometric", "semantic".')
    
    def _generate_nbrs_geometric(self):
        """Performs the geometry-based NBRS-generation.
        Geometric non-overlapping of NBRS is enforced "formally"
        via intersection checks. Furthermore, the navigation
        of the topology and the selection of the best candidate
        for continuation in intersections also takes place
        exclusively via examining the geometry, semantic
        information is not used in any way.
        """
        print('\nSTARTING NBRS GENERATION')
        # build a hashing-based navigation structure of
        # the NWB topology based on linking each intersection
        # location to the wegvakken that end/begin there
        for wvk_id, lstr in self.wvk_geoms.items():
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
            wvk_id = (self.wvk_geoms.keys() - self.nbrs_ids.keys()).pop()
            # start a new NBRS with the chosen wegvak
            self.nbrs_wvkn[i], self.nbrs_ids[wvk_id] = [wvk_id], i
            lstr = self.wvk_geoms[wvk_id]
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
        """Internal method that recursively adds suitable
        wegvakken to the NBRS that is being built.
        """
        nbrs_geom = self.nbrs_geoms[nbrs_id]
        # look for unused wegvakken starting from end
        # "intersection" of the last wegvak
        cont_ids = self.jte[jt_vx] - self.nbrs_ids.keys()
        cont = {k: v for k, v in self.wvk_geoms.items() if k in cont_ids}
        # compute relative angles of outgoing vectors from
        # the intersection - vectors are based on the first
        # edge of each wegvak connected to the intersection
        cont_vecs = {}
        for wvk_id, lstr in cont.items():
            if tuple(np.round(lstr.coords[0], 1)) == jt_vx:
                cont_vecs[wvk_id] = lstr.coords[1]
            else: cont_vecs[wvk_id] = lstr.coords[-2]
        angles = {lib_shared.calc_angle(
                            np.array(prev_vx) - np.array(jt_vx),
                            np.array(vec) - np.array(jt_vx)): wvk_id
                  for wvk_id, vec in cont_vecs.items()}
        # this is an override for intersections that join
        # exactly three wegvakken - this is typical where
        # ramps join motorways, and is used to avoid
        # continuing ramp NBRS on motorways
        if len(cont_vecs) == 2:
            vecs = [vec for _, vec in cont_vecs.items()]
            if (lib_shared.calc_angle(np.array(vecs[0]) - np.array(jt_vx),
                                      np.array(vecs[1]) - np.array(jt_vx))
                > max(angles.keys())): return
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
                if sense == 'end': self.nbrs_wvkn[nbrs_id].append(best_id)
                else: self.nbrs_wvkn[nbrs_id].insert(0, best_id)
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
        Only roads with the same 'wegnummer' are allowed to be in the
        same NBRS. Furthermore, the 'BST' code in a given NBRS also
        needs to be consistent, apart from 'PST' which always
        denotes the last wegvak that joins a ramp to a motorway.
        As such, they are always isolated in terms of their 'BST' 
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
        for wvk_id in self.wvk_geoms.keys():
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
            wvk_ids = [k for k in self.wvk_geoms.keys()
                       if not self.bst_codes[k] == 'PST']
            unused_ids = set(wvk_ids) - self.nbrs_ids.keys()
            if unused_ids: wvk_id = unused_ids.pop()
            else: wvk_id = (self.wvk_geoms.keys()
                            - self.nbrs_ids.keys()).pop()
            # start a new NBRS with the chosen wegvak
            self.nbrs_wvkn[i], self.nbrs_ids[wvk_id] = [wvk_id], i
            self.nbrs_bst[i] = self.bst_codes[wvk_id]
            self.nbrs_wegnos[i] = self.wegnos[wvk_id]
            lstr = self.wvk_geoms[wvk_id]
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
        del (self.jte_end, self.jte_beg, self.nbrs_bst,
             self.nbrs_wegnos, self.bst_codes, self.wegnos)
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
        cont = {k: v for k, v in self.wvk_geoms.items() if k in cont_ids}
        # compute relative angles of outgoing vectors from
        # the intersection - vectors are based on the first
        # edge of each wegvak connected to the intersection
        cont_vxs, cont_revs = {}, {}
        for wvk_id, lstr in cont.items():
            if self.jte_beg[wvk_id] == prev_jt:
                cont_vxs[wvk_id] = lstr.coords[:2]
                cont_revs[wvk_id] = False
            else: cont_vxs[wvk_id] = (lstr.coords[-1], lstr.coords[-2])
        angles = {lib_shared.calc_angle(
                            np.array(prev_vx) - np.array(vxs[0]),
                            np.array(vxs[1]) - np.array(vxs[0])): wvk_id
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
            if sense == 'end': self.nbrs_wvkn[nbrs_id].append(best_id)
            else: self.nbrs_wvkn[nbrs_id].insert(0, best_id)
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
        """Densifies the vertices of the wegvakken to respect
        an input distance threshold (in metres). Each edge in
        each wegvak is checked, and is split in two if necessary.
        The same is then done to the resulting halves (recursively)
        until the specified threshold is respected.
        """
        print('\nSTARTING NBRS DENSIFICATION')
        if self.thres:
            print('WARNING: Densification has already taken place.')
        self.thres = thres
        # iterate all wegvakken
        for wvk_id in self.wvk_geoms.keys():
            wvk_geom = self.wvk_geoms[wvk_id]
            origins = self.wvk_origins[wvk_id]
            wvk_geom, origins = lib_shared.densify_lstr(wvk_geom,
                                                        thres, origins)
            self.set_wvk_geom(wvk_id, wvk_geom)
            self.wvk_origins[wvk_id] = origins
        print('FINISHED NBRS DENSIFICATION')
        
    def estimate_elevations(self, fpath, r = 1, thin = 2):
        """Produces preliminary elevations for the road network.
        Each NBRS vertex is associated with an elevation based on
        nearby Lidar points. Artefacts due to occlusion are then
        (mostly) eliminated by fittin high-degree polynomials.
        Where the polynomial fit indicates outliers, the values
        are re-set according to the model.
        """
        nbrs_ids = self.get_ids()
        if not nbrs_ids: return
        print('\nSTARTING PRELIMINARY ELEVATION ESTIMATION')
        # import and subsample Lidar
        print('IMPORTING AHN3 TILE')
        self.ahn3, self.las_header = lib_shared.las_reader(fpath)
        # apply thinning to the point cloud
        self.ahn3 = self.ahn3[::thin]
        # build a list of all NBRS vertex counts and vertices -
        # per-NBRS lists are needed for smoothing, and a
        # completely flat list of all vertices is needed
        # for the initial KD-tree query
        print('FLATTENING ALL NBRS')
        self.nbrs_vxnos, self.nbrs_revs, flat_vxs = {}, {}, []
        for nbrs_id in nbrs_ids:
            nbrs_geom = self.get_geom(nbrs_id)
            # although the wegvak IDs are in the correct order
            # in NBRS lists in self.nbrs_wvkn, the LineString
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
        nbrs_ix = nbrs_tree.query_ball_tree(lidar_tree, r)
        print('ESTIMATING PRELIMINARY ELEVATIONS')
        # initialise new 3D geometry column and activate it
        self.nwb['geometry_simpleZ'] = None
        self.set_geocol('geometry_simpleZ')
        # compute a rough z-value for each NBRS vertex,
        # perform the smoothing workflow when an NBRS is
        # completed, and add 3D geometries to GeoDataFrame
        first = 0
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            # fetch the nearby Lidar points of each NBRS vertex
            # and compute its elevation as their median
            zs = []
            for pt_ix in nbrs_ix[first:last]:
                if pt_ix:
                    pt_z = self.ahn3[pt_ix, 3]
                    zs.append(np.median(pt_z))
                else: zs.append(np.NaN)
            # assemble NBRS 3D coordinates into an array
            nbrs_vxs = flat_vxs[first:last]
            nbrs_vxs_z = np.c_[nbrs_vxs, zs]
            # apply the elevation smoothing algorithm
            # to the 3D coordinate array
            lib_shared.filter_outliers(nbrs_vxs_z, 8, False, 0.2)
            # re-assemble the smoothed 3D coordinate arrays
            # into wegvak geometries and set the active wegvak
            # geometries of the class to these new geometries, and
            # also save indicator lists (per wegvak, like in
            # self.wvk_origins) that record where the rough elevation
            # estimation originally failed - which is in turn
            # indicative of the presence of bridges, tunnels, etc.
            nbrs_vxnos = np.array(nbrs_vxnos); nbrs_vxnos[1:] += 1
            starts = np.roll(np.cumsum(nbrs_vxnos - 1), 1); starts[0] = 0
            ends = np.cumsum(nbrs_vxnos - 1) + 1
            self.flat_vxs += nbrs_vxs_z.tolist()
            nbrs_wvkn_z = [nbrs_vxs_z[start:end]
                           for start, end in zip(starts, ends)]
            for wvk_id, wvk_z, rev in zip(self.nbrs_wvkn[nbrs_id],
                                          nbrs_wvkn_z,
                                          self.nbrs_revs[nbrs_id]):
                if rev: wvk_z = np.flip(wvk_z, axis = 0)
                self.set_wvk_geom(wvk_id, LineString(wvk_z))
            first = last
        print('FINISHED PRELIMINARY ELEVATION ESTIMATION')

    def segment_lidar(self, fpath, r):
        """Performs DTB-assisted Lidar segmentation of the AHN3 tile.
        The steps of the procedure are as follows:
            1. Densify DTB lines and create a point cloud from
               their vertices.
            2. Build KD-trees with the NBRS vertices, AHN3 points
               and DTB points.
            3. Perform least-squares plane fitting on the AHN3
               points close to NBRS vertices, progressing along
               NBRS vertices. Based on observing changes in basic
               plane-fit-related statistics, detect when fits
               become unstable. In case of instability, if DTB
               exists and is useful locally, then use it to
               stabilise plane fits. If not, then try to use the
               previous (stable) plane. If this also fails, then
               continue normally (even if this means that the
               plane fit might now describe an overlying road).
               Pre-select the points that conform with the plane fits.
            4. Like in preliminary elevation estimation, re-associate
               NBRS vertices with elevations based on the elevation
               of close-by Lidar points, but this time use relevant
               pre-selected points only. Insert breakpoints to
               indicate the starting and ending points of no-data
               regions. Exclude pre-selected points belonging to
               plane fits in the no-data regions. This step
               splits NBRS into parts, each with its own ID.
            5. Assemble point clouds, one per NBRS part. Keep a record
               of which points came from AHN3, and which ones from DTB.
        """
        if not roads.flat_vxs:
            print('Elevations first need to be estimated.'); return
        print('\nSTARTING POINT CLOUD SEGMENTATION')
        print('IMPORTING DTB')
        dtb, dtb_vxs = gpd.read_file(fpath), []
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
                densified = lib_shared.densify_lstr(lstr, 0.2 * r)
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
        area = np.pi * r**2
        print('SEGMENTING POINT CLOUD')
        first = 0
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            vxs = self.flat_vxs[first:last]
            ixs = nbrs_ix[first:last]
            first = last
            planes, lgroups = [], []
            # fit a plane on each group of AHN3 points that
            # were found to close to NWB vertices
            for pt_ix in ixs:
                lgroups.append(np.array([])); planes.append(())
                if len(pt_ix) / area > 1:
                    pts = self.ahn3[:,1:][pt_ix]; lgroups[-1] = pts
                    if len(pt_ix) / area > 2:
                        planes[-1] = (lib_shared.planefit_lsq(pts))
            # the main segmentation algorithm starts below
            subclouds, subclouds_dtb, can_revert = [], [], True
            prev_plane, prev_dist_p, prev_med_z = None, None, None
            i = -1
            for vx, lgroup, plane, in zip(vxs, lgroups, planes):
                i += 1
                vx_subcloud = np.empty([0, 3])
                vx_subcloud_dtb = np.empty([0, 3])
                # if previous vertex was a failed plane reversion,
                # then add a breakpoint to the NBRS and reset rolling
                # descriptors, so that a new elevation level
                # can be established
                if not can_revert:
                    prev_plane, prev_dist_p, prev_med_z = None, None, None
                # if no plane could be fitted or we already know there
                # was an AHN3 data gap from preliminary elevation
                # estimation, then immediately flag for assistance
                need_assist = not plane
                if not need_assist:
                    # compute close-by AHN3 points' distances
                    # to fitted plane and compute basic descriptors
                    dists = np.array([lib_shared.dist_toplane(pt, *plane)
                                      for pt in lgroup])
                    # standard deviation of distances from plane
                    std = np.std(dists)
                    # distance of plane from the NBRS vertex
                    dist_p = lib_shared.dist_toplane(vx, *plane)
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
                    need_assist = std > 0.1 * r or (dz > 0.5 or dp > 0.5)
                if need_assist:
                    pts = np.array([])
                    if len(dtb_vxs) > 0:
                        # if the previous plane exists, move the location
                        # of the next DTB query closer to the plane
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
                            dists = np.array([lib_shared.dist_toplane(pt,
                                              *prev_plane) for pt in pts])
                            pts = pts[dists < 0.1 * r]
                    # if a DTB-assist is possible...
                    if len(pts) > 2:
                        # refit plane onto DTB-defined surface
                        plane = lib_shared.planefit_lsq(pts)
                        # requery DTB to include any points the
                        # conservative radius above missed
                        pt_ix = dtb_tree.query_ball_point(ctr, r)
                        pts = dtb_vxs[pt_ix]
                        dists = np.array([lib_shared.dist_toplane(pt, *plane)
                                          for pt in pts])
                        pts = pts[dists < 0.1 * r]
                        plane = lib_shared.planefit_lsq(pts)
                        # re-compute AHN3 point distances to plane
                        dists = np.array([lib_shared.dist_toplane(pt, *plane)
                                          for pt in lgroup])
                        lclose = lgroup[dists < 0.1 * r]
                        # if many are close to the new plane, then
                        # refit the plane once again to the surface
                        # defined by the AHN3 points that conformed
                        # well with the DTB-defined surface -
                        # this is necessary because DTB is not always
                        # perfectly conformant with the AHN3-defined
                        # road surface, with up to ~1m deviations
                        if len(lclose) / area > 2:
                            plane = lib_shared.planefit_lsq(lclose)
                            prev_med_z = np.median(lclose[:,2])
                        else: prev_med_z = np.median(pts[:,2])
                        prev_plane = plane
                        prev_dist_p = lib_shared.dist_toplane(vx, *plane)
                        # also save DTB points when DTB assistance
                        # completed succesfully
                        if len(pts) > 0: vx_subcloud_dtb = pts.copy()
                    # else if a DTB-assist is not possible...
                    else:
                        # if plane-reverting assistance had
                        # run in the previous iteration, then
                        # "give up" and skip to next NBRS vertex
                        if not can_revert or not prev_plane:
                            dists = None
                        # else, try reverting the plane (use previous)
                        elif len(lgroup) != 0:
                            lgroup = lgroup[(lgroup[:,2] - vx[2]) < 2]
                            dists = np.array([lib_shared.dist_toplane(pt,
                                              *prev_plane) for pt in lgroup])
                            can_revert = False
                else:
                    # roll the plane and the descriptors
                    prev_plane = plane
                    prev_dist_p, prev_med_z = dist_p, med_z
                    can_revert = True
                # pre-select those AHN3 points, which were very close
                # to the fitted plane, and save them -
                # a single "Lidar patch" (spherical group of points)
                # will be associated with each succesful plane fit,
                # (so, one patch per NBRS vertex)
                if dists is not None and len(lgroup) != 0:
                    pre = lgroup[dists < 0.05 * r]
                    if len(pre) > 0: vx_subcloud = pre
                subclouds.append(vx_subcloud)
                subclouds_dtb.append(vx_subcloud_dtb)
            # if neither AHN3 nor DTB could yield enough points, then
            # consider the process a failure and skip to the next NBRS
            if (len(np.concatenate(subclouds)) < 100
                and len(np.concatenate(subclouds_dtb)) < 50): 
                self.nbrs_subclouds[nbrs_id] = {}
                self.nbrs_dtb_ref_sets[nbrs_id] = {}
                continue
            # associate NBRS vertices with an elevation derived
            # from close-by Lidar points in the Lidar patches
            vxs_z = [[*vx[:2], np.median(np.concatenate([vx_sub,
                                                         vx_sub_dtb])[:,2])]
                     if len(vx_sub) > 10 or len(vx_sub_dtb) > 5
                     else [*vx[:2], None]
                     for vx, vx_sub, vx_sub_dtb
                     in zip(vxs, subclouds, subclouds_dtb)]
            mask = lib_shared.filter_outliers(np.array(vxs_z, dtype = float),
                                              12, True, 5)
            # post-process the outlier mask to also exclude patches
            # with too few points inside of them
            coverage = []
            for vx_sub, vx_sub_dtb, masked in zip(subclouds, subclouds_dtb,
                                                  mask):
                if not masked:
                    coverage += [len(vx_sub) / area > 1
                                 or len(vx_sub_dtb) > 10]
                else: coverage += [False]
            # further post-process the mask to eliminate very short
            # changes in coverage - the variable below controls the
            # minimum length needed to accept a change from non-data
            # to data region (or vice-versa)
            min_len = 3
            for i in range(len(coverage) - min_len):
                curr = coverage[i]
                if coverage[i + 1] != curr:
                    cnt = 1
                    for j in range(1, min_len):
                        if coverage[i + 1 + j] != curr: cnt += 1
                        else: break
                    if cnt < min_len:
                        for j in range(1, min_len + 1):
                            if coverage[i + j] != curr:
                                coverage[i + j] = curr
            # generate the final breakpoints based on the mask
            # generated above - the list contains pairs of indices,
            # each corresponding to the beginning and end of a region
            # with proper coverage
            breakpts, beg = [], None
            for curr, i in zip(coverage, range(len(coverage))):
                if curr:
                    if beg is None: beg = i
                    elif i == len(coverage) - 1:
                        breakpts.append([beg, i + 1])
                elif beg is not None:
                    breakpts.append([beg, i])
                    beg = None
            self.nbrs_parts[nbrs_id] = breakpts
            self.nbrs_dtb_ref_sets[nbrs_id] = {}
            self.nbrs_subclouds[nbrs_id] = {}
            # compile NBRS's subclouds (one per NBRS part) from the
            # pre-selected AHN3 and DTB points added in previous steps,
            # each point may only be added once per NBRS part
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                subcloud, subcloud_dtb = set(), set()
                sc_zip = zip(subclouds[part_ix[0]:part_ix[1]],
                             subclouds_dtb[part_ix[0]:part_ix[1]])
                for vx_sub, vx_sub_dtb in sc_zip:
                    for pt in vx_sub: subcloud.add(tuple(pt))
                    for pt in vx_sub_dtb: subcloud_dtb.add(tuple(pt))
                if subcloud:
                    # perform quick outlier filtering on the subcloud
                    subcloud = np.array(list(subcloud))
                    subcloud_tree = cKDTree(subcloud)
                    subcloud_tree_len = subcloud_tree.n
                    _, s_ixs = subcloud_tree.query(subcloud, 3,
                                                   distance_upper_bound = 0.8,
                                                   workers = -1)
                    subcloud_mask = []
                    for pt_ixs in s_ixs:
                        if len(pt_ixs[pt_ixs != subcloud_tree_len]) == 3:
                            subcloud_mask += [True]
                        else: subcloud_mask += [False]
                    subcloud = subcloud[subcloud_mask]
                    subcloud = np.c_[subcloud, np.full(len(subcloud), 0)]
                else: subcloud = np.empty((0, 4))
                # merge DTB patches also into the final subcloud, but
                # create class-level sets of added DTB vertices, so that
                # they can be easily recognised later on as such
                dtb_ref_set = set()
                if subcloud_dtb:
                    subcloud_dtb = np.c_[np.array(list(subcloud_dtb)),
                                         np.full(len(subcloud_dtb), 1)]
                    dtb_rounded = np.round(subcloud_dtb[:,:3], 1)
                    dtb_ref_set = set([tuple(pt) for pt in dtb_rounded])
                    subcloud = np.concatenate((subcloud, subcloud_dtb))
                self.nbrs_dtb_ref_sets[nbrs_id][part_id] = dtb_ref_set
                self.nbrs_subclouds[nbrs_id][part_id] = subcloud
        print('FINISHED POINT CLOUD SEGMENTATION')
        
    def estimate_edges(self, min_width, max_width, thres, perc_to_fit):
        """Method to construct crude preliminary edge-estimates
        based on NBRS's NWB locations and subclouds. Cross-sections
        are constructed on NWB vertices (including densified vertices)
        and the elevations of nearby subcloud points are used to
        transpose them into 3D (they are first densified to enable
        better point cloud sampling). Line-fitting is then used to
        identify an edge point on both sides of NWB onn each cross-
        section. The first conformant point is chosen from the
        ends of the cross-sections, progressing inwards (towards
        NWB). In the absence of such a point, the cross-section is
        not considered any further.
        The first two parameters control the minimum and maximum
        width of generated roads, but are hard limits but are
        enforced differently. The third parameter is a threshold
        that controls the number of standard deviations within
        which points are classed as surface points (during the
        line fitting step). The last parameter controls what
        part of the cross-section will be fitted with the line,
        as fitting the whole cross-section would corrupt the fit
        significantly. Reducing the value takes out points close
        to the ends of the cross-section, meaning that points close
        to the NBRS vertex will be fitted.
        """
        if not self.nbrs_subclouds:
            print('Point cloud first needs to be segmented.'); return
        print('\nSTARTING PRELIMINARY ROAD EDGE ESTIMATION')
        edge_dict = {'NBRS_ID': [], 'PART_ID': [],
                     'SIDE': [], 'geometry': []}
        cross_dict = {'NBRS_ID': [], 'PART_ID': [], 'geometry': []}
        first = 0
        for nbrs_id in self.get_ids():
            # gather necessary NBRS and subcloud data
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = self.flat_vxs[first:last]
            first = last
            subclouds = self.nbrs_subclouds[nbrs_id]
            # special case: NBRS has no subclouds
            if not subclouds: continue
            breakpts = self.nbrs_parts[nbrs_id]
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                part_vxs = nbrs_vxs[part_ix[0]:part_ix[1]]
                subcloud = subclouds[part_id][:,:3]
                if len(subcloud) == 0: continue
                # build 2D KD-tree from subcloud
                lidar_tree = cKDTree(subcloud[:,:2])
                lidar_tree_len = lidar_tree.n
                # generate the cross-sections
                vx0, vx1 = part_vxs[0][:2], part_vxs[1][:2]
                # special case: cross-section on first NBRS vertex
                cross_edges = [lib_shared.get_cross(vx1, vx0, None,
                                                    max_width, 1)]
                # general case
                for vx2 in part_vxs[2:]:
                    cross_edges.append(lib_shared.get_cross(vx0, vx1, vx2[:2],
                                                            max_width, -1))
                    vx0, vx1 = vx1, vx2[:2]
                # special case: cross-section on last NBRS vertex
                cross_edges.append(lib_shared.get_cross(vx0, vx1, None,
                                                        max_width, -1))
                # densify the cross-sections
                cross_edges = [lib_shared.densify_edge(edge, 0.1)
                               for edge in cross_edges
                               if edge is not None]
                cross_vxs = np.concatenate(cross_edges)
                # fetch the Lidar points necessary to transpose
                # the cross-sections into 3D
                _, ixs = lidar_tree.query(cross_vxs, 10,
                                          distance_upper_bound = 0.25,
                                          workers = -1)
                # transpose each (densified) cross-section vertex
                # into 3D using the median of the elevations of
                # the selected close-by subcloud points
                cross_zs = []
                for pt_ixs in ixs:
                    pts = subcloud[pt_ixs[pt_ixs != lidar_tree_len]]
                    if len(pts) == 0: cross_zs.append(np.NaN)
                    else: cross_zs.append(np.median(pts[:,2]))
                cross_vxs = np.c_[cross_vxs, cross_zs]
                # estimate crude edge locations
                mid, prev_widths, prev_zs = len(cross_edges[0]) // 2, [], []
                edges_l, edges_r, crosses, fails = [], [], [], 0
                for vxs in np.split(cross_vxs, len(cross_edges)):
                    outliers = lib_shared.filter_outliers(vxs, 1, True,
                                                          thres, perc_to_fit)
                    if outliers is None: continue
                    missing = np.isnan(vxs[:,2])
                    edge_l, edge_r = None, None
                    # try looking for the first valid cross-section
                    # vertex (one with a non-outlier elevation),
                    # progressing inwards
                    hits = 0
                    for i in range(mid):
                        if not outliers[i] and not missing[i]:
                            if hits == 2 or fails > 2:
                                edge_l = list(vxs[i]); break
                            hits += 1
                        else: hits = 0
                    hits = 0
                    for i in range(1, mid + 1):
                        if not outliers[-i] and not missing[-i]:
                            if hits == 2 or fails > 2:
                                edge_r = list(vxs[-i]); break
                            hits += 1
                        else: hits = 0
                    if not edge_l or not edge_r:
                        fails += 1; continue
                    # the below conditionals ensure that the shape of
                    # the road surface (as defined by the edges) is
                    # not too thin, not too angular and not too bumpy -
                    # the conditions are relaxed after 3 failures to
                    # prevent the algorithm from getting "stuck"
                    width = lib_shared.dist_topoint(edge_l, edge_r)
                    z = (edge_l[2] + edge_r[2]) / 2
                    if width < min_width:
                        fails += 1; continue
                    if fails > 2:
                        prev_widths = []
                    if len(prev_zs) < 5:
                        prev_zs += [(edge_l[2] + edge_r[2]) / 2]
                    elif abs(np.mean(prev_zs) - z) > 1.5:
                        fails += 1; continue
                    else:
                        prev_zs.pop(0)
                        prev_zs += [z]
                    if len(prev_widths) < 5:
                        prev_widths += [width]
                    elif abs(np.mean(prev_widths) - width) > 1:
                        fails += 1; continue
                    else:
                        prev_widths.pop(0)
                        prev_widths += [width]
                    # extend the road edge with new edge points,
                    # and construct the cross-section geometry
                    fails = 0
                    edges_l.append(edge_l)
                    edges_r.append(edge_r)
                    crosses += [LineString((edge_l, edge_r))]
                # construct the part's edge and cross-section geometries
                if len(crosses) > 3:
                    edge_dict['NBRS_ID'] += [nbrs_id] * 2
                    edge_dict['PART_ID'] += [part_id] * 2
                    edge_dict['SIDE'] += ['left', 'right']
                    edge_dict['geometry'] += [LineString(edges_l),
                                              LineString(edges_r)]
                    cross_dict['NBRS_ID'] += [nbrs_id]
                    cross_dict['PART_ID'] += [part_id]
                    cross_dict['geometry'] += [MultiLineString(crosses)]
        # create GeoDataFrames with all the resulting road edge and
        # cross-section geometries
        self.nbrs_edges = gpd.GeoDataFrame(edge_dict, crs="EPSG:28992")
        self.nbrs_crosses = gpd.GeoDataFrame(cross_dict, crs="EPSG:28992")
        print('FINISHED PRELIMINARY ROAD EDGE ESTIMATION')
            
    def optimise_edges(self, size, a, b, g, w_l, w_e, max_iter,
                       chosen = None):
        """Optionally, the preliminary edges can be optimised using
        a method known as "active contour optimisation". This code
        uses the implementation contained in the scikit-image package.
        First the subclouds belonging to each NBRS are rasterised,
        the resulting "attractor map" pixels contain information
        about how smooth the surface is locally. The preliminary
        edges are then iteratively moved towards the highest contrast
        in smoothness in the rasters. The resulting optimised edges
        or "contours" are stored in self.nbrs_contours.
        The parameter 'size' controls the attractor map cell size
        (in metres). All other parameters (except for the last) are
        active contour optimisation parameters, documentation about
        these can be found on scikit's website. The last parameter
        'chosen' allows the user to execute the code for a single NBRS
        by providing its ID, which may be useful for debug purposes.
        [Deprecated, but still usable.]
        """
        if self.nbrs_edges.empty:
            print('Preliminary edges first need to be generated.'); return
        print('\nSTARTING EDGE OPTIMISATION')
        self.mapsize, first = size, 0
        contours = {'NBRS_ID': [], 'PART_ID': [], 'geometry': []}
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = np.array(self.flat_vxs[first:last])
            first = last
            subclouds = self.nbrs_subclouds[nbrs_id]
            edges = self.nbrs_edges[self.nbrs_edges['NBRS_ID'] == nbrs_id]
            # special case: NBRS has no subclouds or edges
            if ((chosen is not None and nbrs_id != chosen)
                or (not subclouds or edges.empty)): continue
            self.nbrs_extents[nbrs_id] = {}
            self.nbrs_ress[nbrs_id] = {}
            self.nbrs_origins[nbrs_id] = {} 
            self.nbrs_maps[nbrs_id] = {}
            breakpts = self.nbrs_parts[nbrs_id]
            # one map is constructed per NBRS part
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                part_vxs = nbrs_vxs[part_ix[0]:part_ix[1]]
                subcloud = subclouds[part_id][:,:3]
                part_edges = edges[edges['PART_ID'] == part_id]
                # special case: part has no subclouds or edges
                if len(subclouds) == 0 or part_edges.empty: continue
                ahn_tree = cKDTree(subcloud[:,:2])
                ahn_tree_len = ahn_tree.n
                # the region of interest is defined as a polygon
                # buffered from the NBRS (the centreline)
                roi = LineString(part_vxs).buffer(self.thres + 2)
                roi_path = mpl.path.Path(roi.exterior.coords[:])
                nwb_tree = cKDTree(part_vxs[:,:2])
                # define the raster dimensions based on the
                # dimensions of the subcloud
                extents = ((min(subcloud[:,0]), max(subcloud[:,0])),
                           (min(subcloud[:,1]), max(subcloud[:,1])))
                res = (round((extents[0][1] - extents[0][0]) / size),
                       round((extents[1][1] - extents[1][0]) / size))
                orig = (sum(extents[0]) / 2 - (size / 2) * res[0],
                        sum(extents[1]) / 2 - (size / 2) * res[1])
                self.nbrs_extents[nbrs_id][part_id] = extents
                self.nbrs_ress[nbrs_id][part_id] = res
                self.nbrs_origins[nbrs_id][part_id] = orig
                # define a flattened array of pixel centre points
                ctr_xs = np.linspace(orig[0], orig[0] + res[0] * size,
                                     res[0]) + size / 2
                ctr_ys = np.linspace(orig[1], orig[1] + res[1] * size,
                                     res[1]) + size / 2
                pix_xs, pix_ys = np.meshgrid(ctr_xs, ctr_ys)
                ctrs = np.c_[pix_xs.ravel(), pix_ys.ravel()]
                # mask out all pixels that are not relevant to the
                # subcloud to reduce computational complexity
                mask = roi_path.contains_points(ctrs)
                # fetch the Lidar points close to pixel centres and
                # NWB vertices (including densified ones)
                _, pix_ixs = ahn_tree.query(ctrs[mask], 15,
                                            distance_upper_bound = 2 * size,
                                            workers = -1)
                _, nwb_ixs = ahn_tree.query(part_vxs[:,:2], 15,
                                            distance_upper_bound = 2 * size,
                                            workers = -1)
                pix_pts = [subcloud[pt_ixs[pt_ixs != ahn_tree_len]] for
                           pt_ixs in pix_ixs]
                nwb_pts = [subcloud[pt_ixs[pt_ixs != ahn_tree_len]] for
                           pt_ixs in nwb_ixs]
                # link pixels to the closest NWB vertex
                pix_nwb = np.array([nwb_tree.query(ctr)[1]
                                    for ctr in ctrs[mask]])
                # mask out pixels where there were very few points
                # in the neighbourhood
                pix_nwb_mask = [True if len(nwb_pts[ix]) > 2 else False
                                for ix in pix_nwb]
                pix_pts_mask = [True if len(pts) > 2 and valid_nwb else False
                                for pts, valid_nwb
                                in zip(pix_pts, pix_nwb_mask)]
                # only the code is left uncommented below that generates
                # a single normal vector (based on its Lidar neighbourhood) -
                # in this release the program only generates one
                # attractor map, which contains kernel-based normal
                # vector inner products - the rest of the maps are not
                # used, because they were all significantly poorer in
                # terms of how they described smoothness contrast
                """nwb_pts_mask = [True if len(pts) > 2 else False
                                for pts in nwb_pts]
                pix_nwb = pix_nwb[pix_nwb_mask]"""
                pix_pts = [pts for pts, get
                           in zip(pix_pts, pix_pts_mask) if get]
                """nwb_pts = [pts if get else None
                           for pts, get in zip(nwb_pts, nwb_pts_mask)]"""
                mask[mask == True] = np.array(pix_pts_mask)
                """pix_vars = np.array([np.var(pts[:,2]) for pts in pix_pts])
                nwb_vars = np.array([np.var(pts[:,2])
                                     if pts is not None else None
                                     for pts in nwb_pts])
                pix_nwb_vars = np.array([nwb_vars[ix] for ix in pix_nwb])
                pix_nwb_dvars = np.array([abs(pix_var - nwb_var)
                                          for pix_var, nwb_var
                                          in zip(pix_vars, pix_nwb_vars)])"""
                pix_norms = np.array([lib_shared.planefit_lsq(pts)[1]
                                      for pts in pix_pts])
                """nwb_norms = [lib_shared.planefit_lsq(pts)[1]
                             if pts is not None else None
                             for pts in nwb_pts]
                pix_nwb_norms = np.array([nwb_norms[ix] for ix in pix_nwb])
                pix_nwb_dots = np.array([np.dot(pix_norm, nwb_norm)
                                         for pix_norm, nwb_norm
                                         in zip(pix_norms, pix_nwb_norms)])"""
                # the code below is a single iteration that computes
                # the kernel-based normal vector inner products in the
                # raster - pairwise normal vector compinations in the
                # kernel pixels are considered, but only a random subset
                # are actually dotted together to increase performance
                temp = np.full((res[0] * res[1], 3), np.NaN)
                if temp[mask].shape != pix_norms.shape: continue
                temp[mask] = pix_norms
                pix_norms_raster = np.array(np.split(temp, res[1]))
                mask_raster = np.array(np.split(mask, res[1]))
                kernel_shape = 4, 4
                kernel_grid = np.meshgrid(range(kernel_shape[0]),
                                          range(kernel_shape[1]))
                kernel_template = np.dstack(kernel_grid).reshape(-1, 2)
                pix_kernel_dots = []
                for yi in range(res[1]):
                    for xi in range(res[0]):
                        if not mask_raster[yi, xi]: continue
                        kernel = np.c_[kernel_template[:,0] - 1 + xi,
                                       kernel_template[:,1] - 1 + yi]
                        kernel = kernel[(kernel[:,0] > 0)
                                        & (kernel[:,0] > 0)]
                        kernel = kernel[(kernel[:,0] < res[0])
                                        & (kernel[:,1] < res[1])]
                        norms = pix_norms_raster[(kernel[:,1], kernel[:,0])]
                        norms = norms[~np.isnan(norms[:,0])]
                        if len(norms) > 5:
                            pair_mg = np.meshgrid(range(len(norms)),
                                                  range(len(norms)))
                            pair_ixs = np.dstack(pair_mg).reshape(-1, 2)
                            pair_ixs = pair_ixs[::len(pair_ixs) // 20]
                            pair_ixs = pair_ixs[pair_ixs[:,0] != pair_ixs[:,1]]
                            pairs = norms[pair_ixs,]
                            dots = np.array([np.dot(v0, v1)
                                             for v0, v1 in pairs])
                        else: dots = np.array([])
                        dots = dots[~np.isnan(dots)]
                        if len(dots) > 0: pix_kernel_dots += [np.median(dots)]
                        else: pix_kernel_dots += [None]
                img = np.full(len(mask), np.NaN)
                img[mask] = pix_kernel_dots
                img = np.array(np.split(img, res[1]))
                self.nbrs_maps[nbrs_id][part_id] = img
                # the preliminary edges are converted to
                # use image coordinates rather than the real CRS
                edge_l = part_edges[part_edges['SIDE'] == 'left']
                edge_r = part_edges[part_edges['SIDE'] == 'right']
                edge_l = np.array(edge_l['geometry'].iloc[0].coords[:])
                edge_r = np.array(edge_r['geometry'].iloc[0].coords[:])
                edge_l = np.flip((edge_l[:,:2] - orig) / size, axis = 1)
                edge_r = np.flip((edge_r[:,:2] - orig) / size, axis = 1)
                # scikit does not have a concept of no-data pixels, so the
                # "neutral" value of 1 is used instead
                img_nonan = img.copy(); img_nonan[np.isnan(img_nonan)] = 1
                # the left and right NBRS edges are optimised separately
                snake_l = active_contour(img_nonan, edge_l, coordinates = 'rc',
                                         alpha = a, beta = b, gamma = g,
                                         w_line = w_l, w_edge = w_e,
                                         max_iterations = max_iter,
                                         boundary_condition = 'fixed')
                snake_r = active_contour(img_nonan, edge_r, coordinates = 'rc',
                                         alpha = a, beta = b, gamma = g,
                                         w_line = w_l, w_edge = w_e,
                                         max_iterations = max_iter,
                                         boundary_condition = 'fixed')
                # the optimised edges are converted back into the
                # real-life CRS, are combined into a single Polygon
                snake_l = np.flip(snake_l, axis = 1) * size + orig
                snake_r = np.flip(snake_r, axis = 1) * size + orig
                contour = np.concatenate((snake_l, np.flip(snake_r, axis = 0)))
                contours['NBRS_ID'] += [nbrs_id]
                contours['PART_ID'] += [part_id]
                contours['geometry'] += [Polygon(contour)]
        # the optimised edges are saved into a class-level GeoDataFrame
        self.nbrs_contours = gpd.GeoDataFrame(contours, crs = "EPSG:28992")
        print('FINISHED EDGE OPTIMISATION')

    def build_tin(self,
                  max_dh_int, max_angle_int, r_int,
                  max_dh_ext, max_angle_ext, r_ext,
                  ext_steps, ext_dist,
                  type_edges = 'optimised',
                  chosen = None):
        """The TINs describing the surfaces of NBRS roads can either
        be constructed from the optimised edges, or simply from the
        preliminary edges. This method makes preparations to call
        lib_shared.utility_tin() once to construct an initial,
        conservative surface for the NBRS and then multiple times
        to gradually extend the surface beyond the boundary defined
        by the NBRS edges. This is intended to make it possible to
        include parts of road surfaces that were not detected by
        preliminary edge construction / active contour optimisation,
        but it is optional and the user should be aware that it
        may cause the inclusion of off-road points depending on the
        local distribution of points, and the closeness of the
        centreline to the real-life edges of the road surface.
        The TIN is constructed using a workflow inspired by
        ground filtering algorithms. The parameters are:
        - 'max_dh': maximum elevation difference threshold when
                    inserting a point into the TIN, the difference is
                    measured between the TIN-projected elevation of
                    the candidate point, and its own elevation
        - 'max_angle': maximum angle threshold when inserting a point
                       into the TIN - the angle is measured between
                       the triangle into which the point would be
                       inserted, and the three lines between the point
                       and the three vertices of the triangle
        - '_int' and '_ext' at the end of these variables indicates
           whether they will be taken into account during the initial
           TIN construction step, or the incremental TIN extension
        - 'ext_steps': the number of times the extension boundary
                       should be grown (buffered), in turn the
                       maximum number of iterations that will be
                       attempted when extending the TIN
        - 'ext_dist': the distance (in metres) that the extension
                      boundary should be buffered by in each
                      extension iteration
        - 'type_edges': whether to use 'preliminary' or 'optimised'
                        edges as the basis for the initial TIN
                        construction step (before extension)
        - 'chosen': allows the user to specify a single NBRS ID and
                    run the code for that NBRS only - may be useful
                    for testing parametrisation, etc.
        """
        if type_edges == 'optimised' and self.nbrs_contours.empty:
            print('Contour optimisation first needs to be run.'); return
        if type_edges == 'preliminary' and self.nbrs_edges.empty:
            print('Preliminary edges first need to be generated.'); return
        print('\nSTARTING TIN CONSTRUCTION')
        for nbrs_id in self.get_ids():
            subclouds = self.nbrs_subclouds[nbrs_id]
            edges = self.nbrs_edges[self.nbrs_edges['NBRS_ID'] == nbrs_id]
            # special case: NBRS has no subclouds or edges
            if ((chosen is not None and nbrs_id != chosen)
                or (not subclouds or edges.empty)): continue
            self.nbrs_tins[nbrs_id] = {}
            breakpts = self.nbrs_parts[nbrs_id]
            # a separate TIN will be constructed for each NBRS part
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                subcloud = subclouds[part_id][:,:3]
                part_edges = edges[edges['PART_ID'] == part_id]
                # special case: part has no subclouds or edges
                if len(subcloud) == 0 or part_edges.empty: continue
                # create seed points for conditional TIN insertions as
                # an approximation of the "skeleton" of the surface
                # (based on the preliminary or optimised edges)    
                edge_l = part_edges[part_edges['SIDE'] == 'left']
                edge_r = part_edges[part_edges['SIDE'] == 'right']
                edge_l = np.array(edge_l['geometry'].iloc[0].coords[:])
                edge_r = np.array(edge_r['geometry'].iloc[0].coords[:])
                if type_edges == 'optimised':
                    ctr = self.nbrs_contours[
                            (self.nbrs_contours['NBRS_ID'] == nbrs_id)
                             & (self.nbrs_contours['PART_ID'] == part_id)]
                    if ctr.empty: continue
                    ctr = np.array(ctr.values[0][2].exterior.coords[:])
                else:
                    ctr = np.concatenate((edge_l,
                                          np.flip(edge_r, axis = 0)))[:,:2]
                int_seeds = [(p0 + p1) / 2 for p0, p1 in zip(edge_l, edge_r)]
                int_seeds = lib_shared.densify_lstr(LineString(int_seeds),
                                                    self.thres)[0].coords[:]
                # if extension is desired, then mask out those
                # Lidar points that will not be considered based
                # on the parametrisation
                if ext_steps:
                    max_extend = 1 + ext_dist * (ext_steps + 1)
                    max_bounds = LineString(int_seeds).buffer(max_extend)
                    all_coords = max_bounds.exterior.coords[:]
                    all_path = mpl.path.Path(all_coords)
                    all_mask = all_path.contains_points(subcloud[:,:2])
                    all_pts = subcloud[all_mask]
                else: all_pts = subcloud
                # select the Lidar points that will be needed for the
                # construction of the conservative surface -
                # this means all points within the NBRS edges
                int_path = mpl.path.Path(ctr)
                int_mask = int_path.contains_points(all_pts[:,:2])
                int_pts = [tuple(spt) for spt in all_pts[int_mask]]
                int_bounds = np.c_[ctr, np.zeros(len(ctr))]
                # construct the conservative (initial) surface
                # the actual conditional TIN construction code is
                # found in lib_shared.utility_tin()
                pts_inserted = lib_shared.utility_tin(int_pts, int_bounds,
                                                      max_dh_int, max_angle_int,
                                                      r_int, None, int_seeds)
                if not pts_inserted: continue
                # prepare for iterative extension of the TIN surface
                if ext_steps:
                    prev_bounds = LineString(int_seeds).buffer(1)
                    init_coords = prev_bounds.exterior.coords[:]
                    init_path = mpl.path.Path(init_coords)
                    init_mask = init_path.contains_points(all_pts[:,:2])
                    init_pts = set([tuple(spt) for spt in all_pts[init_mask]])
                    # keep track of which points should no longer be
                    # considered, either because they have already been
                    # tested in a previous iteration, or because they
                    # are already part of the TIN
                    excluded_pts = init_pts | set(pts_inserted)
                # the iteration below calls lib_shared.utility_tin()
                # each time, but with a different insertion parametrisation,
                # and with a boundary that is incrementally buffered from
                # the NWB centreline - the seeds are, in each iteration,
                # the points from the boundary used in the *previous*
                # iteration of the for loop
                for extend in np.arange(ext_dist,
                                        ext_dist * (ext_steps + 1),
                                        ext_dist):
                    ext_coords = prev_bounds.exterior.coords[:]
                    ext_seeds = np.c_[ext_coords,
                                      np.zeros(len(ext_coords))].tolist()
                    prev_bounds = prev_bounds.buffer(extend)
                    ext_coords = prev_bounds.exterior.coords[:]
                    ext_bounds = np.c_[ext_coords,
                                       np.zeros(len(ext_coords))].tolist()
                    ext_path = mpl.path.Path(ext_coords)
                    ext_mask = ext_path.contains_points(all_pts[:,:2])
                    buffer_pts = set([tuple(spt) for spt in all_pts[ext_mask]])
                    ext_pts = buffer_pts - excluded_pts 
                    if not ext_pts: continue
                    pts_inserted = lib_shared.utility_tin(list(ext_pts),
                                                          ext_bounds,
                                                          max_dh_ext,
                                                          max_angle_ext,
                                                          r_ext,
                                                          pts_inserted,
                                                          ext_seeds)
                    excluded_pts = excluded_pts | set(pts_inserted) | ext_pts
                # so far, the TIN has been represented by a list of
                # point insertion - generate the final TIN here and
                # save it to the class-level TIN dictionary
                tin = startin.DT()
                tin.insert(pts_inserted)
                self.nbrs_tins[nbrs_id][part_id] = tin
        print('FINISHED TIN CONSTRUCTION')

    def interpolate_elevations(self):
        """Interpolation algorithm to compute the final elevations of
        the NBRS centrelines. The final elevations are interpolated
        in the TINs that were generated for each NBRS. Since the TINs
        around NBRS end-points (real-life intersections) will be almost
        identical for each NBRS concerned, continuity is ensured by
        enforcing that each intersection is associated with a single
        elevation value. Keep in mind that this kind of 'snapping' may
        not work reliably if the Lidar cloud is thinned too aggressively.
        """
        if not self.nbrs_tins:
            print('TINs first need to be constructed.'); return
        print('\nSTARTING ELEVATION INTERPOLATION')
        # create and activate the new geometry column
        self.nwb['geometry_accurateZ'] = None
        self.set_geocol('geometry_accurateZ')
        self.vx_zs, self.vx_z_origins, first = {}, {}, 0
        for nbrs_id in self.get_ids():
            # initialise NBRS data
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = np.array(self.flat_vxs[first:last])[:,:2]
            nbrs_vxs_3d = np.c_[np.array(nbrs_vxs), np.zeros(len(nbrs_vxs))]
            first = last
            tins = self.nbrs_tins.get(nbrs_id)
            # special case: NBRS has no TINs
            if not tins: continue
            breakpts = self.nbrs_parts[nbrs_id]
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                part_vxs = nbrs_vxs[part_ix[0]:part_ix[1]]
                dtb_set = self.nbrs_dtb_ref_sets[nbrs_id][part_id]
                # fetch the NBRS part's TIN
                tin = tins.get(part_id)
                if tin is not None:
                    # for each NBRS part vertex, interpolate in the TIN
                    # and indicate whether elevation was influenced by
                    # DTB or not - also keep track of what elevation
                    # belongs to which horizontal position in each NBRS
                    # by using dictionaries
                    for vx in part_vxs:
                        xy_id = (*np.round(vx, 1), nbrs_id)
                        try: z = tin.interpolate_tin_linear(*np.round(vx, 1))
                        except: continue
                        tr, origin = tin.locate(*vx), 'ahn3'
                        tr_vxs = [tin.get_point(tix) for tix in tr]
                        for tr_vx in tr_vxs:
                            if tuple(np.round(tr_vx, 1)) in dtb_set:
                                origin = 'dtb'; break
                        self.vx_zs[xy_id] = z
                        self.vx_z_origins[xy_id] = origin
            # select the final elevation for each NBRS vertex -
            # if an earlier NBRS already has an elevation for a
            # certain lateral position and is reasonably close
            # vertically, then snap current vertex to it and
            # indicate that this has taken place semantically
            prev_z = None
            for vx, i in zip(nbrs_vxs, range(len(nbrs_vxs))):
                xy = tuple(np.round(vx, 1))
                z = self.vx_zs.get((*xy, nbrs_id))
                if z or prev_z:
                    for nbrs_idi in self.get_ids()[:nbrs_id]:
                        zi = self.vx_zs.get((*xy, nbrs_idi))
                        zi_o = self.vx_z_origins.get((*xy, nbrs_idi))
                        if zi and (z and (abs(zi - z) < 0.4)
                                   or (prev_z and abs(zi - prev_z)  < 0.2)):
                            kw = zi_o + '_snapped'
                            self.vx_z_origins[(*xy, nbrs_id)] = kw
                            z = zi; break
                nbrs_vxs_3d[i, 2] = z; prev_z = z
            # fill no-data regions with interpolated values using
            # polynomial fitting - do not touch existing values
            lib_shared.filter_outliers(nbrs_vxs_3d, 10, False, 1, None, True)
            nbrs_vx_zs, nbrs_z_origins = {}, {}
            for vx, i in zip(nbrs_vxs_3d, range(len(nbrs_vxs_3d))):
                xy = tuple(np.round(vx[:2], 1))
                nbrs_vx_zs[xy] = vx[2]
                zo = self.vx_z_origins.get((*xy, nbrs_id))
                if not zo: zo = 'polynomial fit'
                nbrs_z_origins[xy] = zo
            # we want to write the results in the same format as
            # the input, so we need to carry over the final elevations
            # into the original wegvakken of the input tile -
            # for each wegvak of the NBRS, fetch its final elevations
            # per vertex and update its class geometry at the end
            for wvk_id in self.nbrs_wvkn[nbrs_id]:
                wvk_vxs = np.array(self.get_wvk_geom(wvk_id).coords[:])[:,:2]
                wvk_vxs_3d = np.c_[wvk_vxs, np.zeros(len(wvk_vxs))]
                self.wvk_z_origins[wvk_id] = []
                for vx, i in zip(wvk_vxs, range(len(wvk_vxs))):
                    xy = tuple(np.round(vx, 1))
                    wvk_vxs_3d[i, 2] = nbrs_vx_zs[xy]
                    self.wvk_z_origins[wvk_id] += [nbrs_z_origins[xy]]
                self.set_wvk_geom(wvk_id, LineString(wvk_vxs_3d))
        print('FINISHED ELEVATION INTERPOLATION')

    def generate_errormodel(self, r):
        """Generates an empirical interpolation error model
        from all TINs by randomly sampling them using the
        jackknife method.
        [Deprecated, but still usable.]
        """
        # "step" below specifies how many samples should be selected -
        # the value of 10 below means that every 10th TIN vertex will
        # be included in the jackknife procedure, i.e. 10% of all vertices
        step, samples = 10, []
        for nbrs_id, tins in self.nbrs_tins.items():
            for part_id in range(len(self.nbrs_parts[nbrs_id])):
                orig_tin = tins.get(part_id)
                if orig_tin is None: continue
                # pick the 10% of the given TIN's vertices (the samples)
                vxs = np.array(orig_tin.all_vertices()[1:])
                smp_vxs = vxs[::step]
                # find close-by vertices for sampling density computation
                smp_tree = cKDTree(smp_vxs[:,:2])
                part_tree = cKDTree(vxs[:,:2])
                qry = smp_tree.query_ball_tree(part_tree, r)  
                tin = startin.DT(); tin.insert(vxs)
                # remove the samples one by one, interpolate at their
                # lateral positions and associate the observed deviation
                # with the local sampling density
                for i, j in zip(range(1, tin.number_of_vertices(), step),
                                range(tin.number_of_vertices() - 1)):
                    tin.remove(i); vx = smp_vxs[j]
                    tr = tin.locate(*vx[:2])
                    if not tr: continue
                    z = tin.interpolate_laplace(*vx[:2])
                    samples += [(len(qry[j]) / r ** 2, abs(vx[2] - z))]
                    tin.insert_one_pt(*vx)
        # create the error model from the sampling density and
        # observed error pairs, and save it into a class variable
        samples = np.array(samples)
        self.errormodel = np.polynomial.polynomial.polyfit(samples[:,0],
                                                           samples[:,1], 3)
    
    def generate_empiricalerrors(self, r):
        """Computes the interpolation error at each NWB vertex
        using the error model generated by .generate_errormodel().
        Creates a dictonary with the errors of the original
        NWB wegvakken in .wvk_z_errors.
        [Deprecated, but still usable.]
        """
        self.wvk_z_errors, first = {}, 0
        # perform the usual iteration over NBRS parts
        # and their associated TINs
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = np.array(self.flat_vxs[first:last])[:,:2]
            first = last
            tins = self.nbrs_tins.get(nbrs_id)
            if not tins: continue
            breakpts = self.nbrs_parts[nbrs_id]
            nbrs_z_errors = {}
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                part_vxs = nbrs_vxs[part_ix[0]:part_ix[1]]
                tin = tins.get(part_id)
                if tin is None: continue
                tin_vxs = np.array(tin.all_vertices())[1:]
                # find close-by vertices for sampling density computation
                tin_tree = cKDTree(tin_vxs[:,:2])
                part_tree = cKDTree(part_vxs[:,:2])
                part_qry = part_tree.query_ball_tree(tin_tree, r)    
                # use the error model and the sampling density values
                # to obtain an error values for each NBRS part vertex
                for vx, qry in zip(part_vxs, part_qry):
                    tr = tin.locate(*vx)
                    if not tr: continue
                    loc, model = len(qry) / r ** 2, self.errormodel
                    error = np.polynomial.polynomial.polyval(loc, model)
                    nbrs_z_errors[tuple(np.round(vx, 1))] = error
            # save the computed errors in a class variable (dictionary)
            for wvk_id in self.nbrs_wvkn[nbrs_id]:
                wvk_vxs = np.array(self.get_wvk_geom(wvk_id).coords[:])[:,:2]
                wvk_z_errors = []
                for vx, i in zip(wvk_vxs, range(len(wvk_vxs))):
                    xy = tuple(np.round(vx, 1))
                    wvk_z_errors += [nbrs_z_errors.get(xy)]
                self.wvk_z_errors[wvk_id] = np.array(wvk_z_errors,
                                                     dtype = float)
                
    def generate_theoreticalerrors(self, ahn_zse, ahn_hse,
                                   dtb_zse, dtb_hse,
                                   r, thres):
        """Computes the interpolation error at each NWB vertex
        using theoretical error propagation in TIN-linear
        interpolation. Creates a dictonary with the errors of
        the original NWB wegvakken in .wvk_z_errors. Takes the
        primary data (AHN3) vertical and horizontal accuracy,
        support data (DTB) vertical and horizontal accuracy,
        density computation radius and minimum density threshold.
        """
        self.wvk_z_errors, self.wvk_density, first = {}, {}, 0
        # perform the usual iteration over NBRS parts
        # and their associated TINs
        for nbrs_id in self.get_ids():
            nbrs_vxnos = self.nbrs_vxnos[nbrs_id]
            last = first + sum(nbrs_vxnos)
            nbrs_vxs = np.array(self.flat_vxs[first:last])[:,:2]
            first = last
            tins = self.nbrs_tins.get(nbrs_id)
            if not tins: continue
            breakpts = self.nbrs_parts[nbrs_id]
            nbrs_errParam, nbrs_density = {}, {}
            for part_ix, part_id in zip(breakpts, range(len(breakpts))):
                part_vxs = nbrs_vxs[part_ix[0]:part_ix[1]]
                tin = tins.get(part_id)
                if tin is None: continue
                tin_vxs = np.array(tin.all_vertices())[1:]
                # find close-by vertices for sampling density computation
                tin_tree = cKDTree(tin_vxs[:,:2])
                part_tree = cKDTree(part_vxs[:,:2])
                part_qry = part_tree.query_ball_tree(tin_tree, r)
                # use the theoretical formula and the local sampling
                # density to compute the parameters of the error
                # propagation formula for each vertex
                for vx, qry in zip(part_vxs, part_qry):
                    xy = tuple(np.round(vx, 1))
                    tr = tin.locate(*vx)
                    if not tr: continue
                    nbrs_density[xy] = len(qry) / r ** 2
                    if len(qry) >= thres:
                        tr_vxs = [tin.get_point(tix) for tix in tr]
                        M = lib_shared.calc_M(vx, tr_vxs)
                        xz, yz = lib_shared.calc_angleComponents(tin, tr)
                        nbrs_errParam[xy] = (M, xz, yz)
            # use the error propagation parameters and the input
            # accuracy to establish the error of each elevation estimate
            for wvk_id in self.nbrs_wvkn[nbrs_id]:
                wvk_vxs = np.array(self.get_wvk_geom(wvk_id).coords[:])[:,:2]
                wvk_origs = self.wvk_z_origins[wvk_id]
                wvk_z_errors, wvk_density = [], []
                for vx, i, orig in zip(wvk_vxs, range(len(wvk_vxs)),
                                       wvk_origs):
                    xy = tuple(np.round(vx, 1))
                    errParams, se = nbrs_errParam.get(xy), None
                    density = nbrs_density.get(xy)
                    # use the origin of each elevation estimate to
                    # decide how their error should be determined
                    if errParams and orig != 'polynomial fit':
                        M, angle_x, angle_y = errParams
                        xz, yz = np.tan(angle_x), np.tan(angle_y)
                        if orig in ['ahn3', 'ahn3_snapped']:
                            zvar, hvar = ahn_zse ** 2, ahn_hse ** 2
                        if orig == 'dtb':
                            zvar, hvar = dtb_zse ** 2, dtb_hse ** 2
                        se = np.sqrt(M * (zvar + hvar * (xz ** 2 + yz ** 2)))
                    wvk_z_errors += [se]
                    wvk_density += [density]
                # save both the density and the errors into class
                # variables (dictionaries)
                self.wvk_z_errors[wvk_id] = np.array(wvk_z_errors,
                                                     dtype = float)
                self.wvk_density[wvk_id] = np.array(wvk_density,
                                                    dtype = float)
                
        
# testing configuration, only runs when script is not imported
if __name__ == '__main__':
    
    # example file paths - they use the same naming convention
    # for convenicence, as the uploaded testing data
    nwb_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_nwb.shp'
    dtb_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_dtb.shp'
    ahn_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_2_26_clipped.las'
    simpleZ_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_simpleZ.shp'
    subclouds_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_subclouds.las'
    edges_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_edges.shp'
    crosses_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_crosses.shp'
    #maps_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_map'
    #conts_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_conts.shp'
    tin_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_tin'
    accurateZ_fpath = '[PATH_TO_DIRECTORY]//C_39CZ1_accurateZ.shp'
    
    roads = nbrs_manager(fpath = nwb_fpath)
    roads.generate_nbrs(algorithm = 'geometric')
    roads.densify(thres = 5)
    #roads.plot_all()
    roads.estimate_elevations(fpath = ahn_fpath, r = 1, thin = 2)
    roads.write_all(fpath = simpleZ_fpath)
    roads.segment_lidar(fpath = dtb_fpath, r = 10)
    roads.write_subclouds(fpath = subclouds_fpath)
    roads.estimate_edges(min_width = 3.5, max_width = 7,
                         thres = 1, perc_to_fit = 0.4)
    roads.write_edges(fpath_edges = edges_fpath,
                      fpath_crosses = crosses_fpath)
    # active contour optimisation can be achieved using the
    # code below - this parametrisation works with all testing data
    """roads.optimise_edges(size = 0.5,
                            a = 0, b = 0.01, g = 0.005,
                            w_l = 0.05, w_e = 1,
                            max_iter = 1000)
    roads.write_maps(fpath = maps_fpath)
    roads.write_contours(fpath = conts_fpath)"""
    # set type_edges to 'optimised' below to use the results of
    # active contour optimisation in place of preliminary edges
    roads.build_tin(max_dh_int = 0.1, max_angle_int = 0.12, r_int = 1,
                    max_dh_ext = 0.03, max_angle_ext = 0.04, r_ext = 0.8,
                    ext_steps = 5, ext_dist = 0.5,
                    type_edges = 'preliminary')
    roads.write_tins(fpath = tin_fpath)
    roads.interpolate_elevations()
    roads.write_all(fpath = accurateZ_fpath,
                    to_drop = ['geometry_simpleZ'])
    roads.write_origins(fpath = origins_fpath)
    roads.generate_theoreticalerrors(ahn_zse = 0.075, ahn_hse = 0.09,
                                     dtb_zse = 0.1, dtb_hse = 0.05,
                                     r = 3, thres = 3)
    roads.write_errors(errors_fpath)
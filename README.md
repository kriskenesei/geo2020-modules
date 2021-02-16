# Python Repository of GEO2020 Modules

**This Python repo contains the software modules I develop for my MSc dissertation at the TU Delft (GEO2020).**
The research proposal (and all forthcoming reports) are based in [my other GEO2020 repo](https://github.com/kriskenesei/geo2020-tex).

Less than two weeks have so far been spent on development. Currently the splitting of NWB into Non-Branching Line Segments (NBRS),
vertex densification, preliminary elevation estimation, elevation smoothing and point cloud segmentation are the features that work.
These have all been packaged into one "module" `nbrs_generation.py`. Most of the functionality resides in the class `nbrs_manager`.
The documentation of the module is currently "inline", i.e. a comprehensive set of docstrings and comments are provided in the code.

Sample files and testing outputs can be downloaded [from here](https://1drv.ms/u/s!AphjAMHVq92GmLYqZnv1XvBE596u_A?e=tEYfZI).
The files have been compressed into a 7-Zip file as they are large (not conventional Zip).
Please note that for the time being, only the testing datasets 'Apeldoornseweg' `32HZ2`, 'Knoppunt Ridderkerk' `37HN2`, 'Gorinchem' `38GZ1`, and 'Knoppunt Deil' `39CZ1` (described in Section 4.4 of the proposal) and the corresponding outputs are maintained and published at the URL above. More will be added later on in the project, when focus will shift to fine-tuning algorithms and performing quality assessment.  
Also note that some wegvakken/NBRS in the output may have all-NaN elevations and no Lidar reflections in their subclouds. This is due to problems with how the testing files were produced;
some extend beyond the boundaries of the relevant AHN3 tiles and can therefore not be associated with elevations. The issue is not related to the software implementation itself.
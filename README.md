# Python Repository of GEO2020 Modules

**This Python repo contains the software modules I develop for my MSc dissertation at the TU Delft (GEO2020).**
The research proposal (and all forthcoming reports) are based in [my other GEO2020 repo](https://github.com/kriskenesei/geo2020-tex).

Less than two weeks have so far been spent on development. Currently the splitting of NWB into Non-Branching Line Segments (NBRS),
vertex densification, _(very rough)_ preliminary elevation estimation, and elevation smoothing are the features that work.
These have all been packaged into one "module" `nbrs_generation.py`. Most of the functionality resides in the class `nbrs_manager`.
The documentation of the module is currently "inline", i.e. a comprehensive set of docstrings and comments are provided in the code.

Sample files and testing outputs can be downloaded [here](https://we.tl/t-R98GRlCrzm).
The files have been compressed into a 7-Zip file as they are large (not conventional Zip).
Note that the testing files 'Veluwe, 32FZ2' and 'Gorinchem, 38GZ1' (mentioned in Section 4.4 of the proposal) and the corresponding outputs 
are missing from the package, because these testing files were produced incorrectly. This will be fixed later.
Also note that some wegvakken in the output may have all-NaN elevations. This is also due to problems with how the testing files were produced;
some extend beyond the boundaries of the relevant AHN3 tiles and can therefore not be associated with elevations. This will also be fixed later.
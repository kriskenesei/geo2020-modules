# Python Repository of GEO2020 Modules

**This Python repo hosts the source code I am working on for my MSc dissertation at the TU Delft (GEO2020).**
The research proposal (and all reports) are based in [my other GEO2020 repo](https://github.com/kriskenesei/geo2020-tex).

Approximatly three months were spent on development so far. All planned features of the software have been implemented and tested, only minor modifications are to be expected until the conclusion of the project.

I packaged most functionality into one "module" `nbrs_generation.py`. Almost all the features reside in the class `nbrs_manager`, to provide an interface that is easy to get familiar with for potential re-users. A set of processing functions were factored out into the file `lib_shared.py` to reduce clutter in the main module.

The documentation of the software is "inline", i.e. a comprehensive set of docstrings and comments are provided in the code. For a top-level description of how the interface is meant to be used, please see the docstring of the `nbrs_manager` class.

Sample inputs and outputs can be downloaded [from here](https://1drv.ms/u/s!AphjAMHVq92GmLdoPSu5BWCpGbbMbQ?e=377RGg). They were all generated using the example code provided at the end of `nbrs_generation.py` (the code that runs if the file itself is run).
The files were compressed into a 7-Zip file (not conventional Zip), because of their considerable size. The size of the 7-Zip archive is about 2Gb. All inputs and outputs (including intermediate results) are included for all testing datasets listed in Section 4.4 of the P2 document (the project proposal). Active contour optimisation-based outputs (attractor maps, optimised contours as well as TINs and 3D-NWB files generated from them) are only included for the datasets `C_20BN1`, `C_37HN2` and `C_39CZ1`. Outputs tagged `_conts` at the end of their file names are active contour optimisation-based versions of their non-tagged counterparts. To visualise attractor maps, I recommend using minimum and maximum bounds of 0.998 and 1 respectively.

Note that some wegvakken/NBRS in the output may have all-NaN elevations and empty Lidar subclouds. This is due to issues with how the Lidar input files were produced; some NBRS extend beyond the boundaries of the relevant AHN3 tiles and can therefore not be associated with elevations. This issue bears no relevance to the software's effectiveness.

My focus will now shift to producing the P4 report and working on the accuracy assessment part of the project. It is possible that I will implement some of the accuracy assessment workflows in the `nbrs_manager` class, in which case I will update the repository once more to release these features too.

![Screenshot of preliminary edges and cross-sections in C_37HN2](readme_thumb.png?raw=true "Screenshot of preliminary edges and cross-sections in C_37HN2")
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Python repository of MSc research source code

**This Python repository hosts the final source code related to my MSc thesis at the TU Delft, titled "Constructing a digital 3D road network for The Netherlands".**
The research proposal and the thesis itself are based in [my LaTeX repository](https://github.com/kriskenesei/geo2020-tex).

All planned features of the software have been implemented and tested, and the project has been concluded. The final thesis is now available in the LaTeX repository. Only minor modifications or additions are expected to the code from now on.

I packaged most functionality into one "module" called `nbrs_generation.py`. Almost all the features reside in the class `nbrs_manager`, to provide an interface that is easy to get familiar with for potential re-users. A set of processing functions were factored out into `lib_shared.py` to reduce clutter in the main module.

The documentation of the software is "inline", i.e. a comprehensive set of docstrings and comments are provided in the code. For a top-level description of how the interface is meant to be used, please see the docstring of the `nbrs_manager` class.

Sample inputs and outputs can be downloaded [from here](https://1drv.ms/u/s!AphjAMHVq92GmLkeH3rdEv6iYszUog?e=KAkav2). They were all generated using the example code provided at the end of `nbrs_generation.py` (the code that runs if the file itself is run).
The files were compressed into a 7-Zip file (not conventional Zip), because of their considerable size. The size of the 7-Zip archive is about 2Gb. All inputs and outputs (including intermediate results) are included for all testing datasets listed in Section 4.1.2 of the P4 report. Output files based on active contour optimisation are only included for the datasets `C_20BN1`, `C_37HN2` and `C_39CZ1`. Outputs tagged `_conts` at the end of their file names are active contour optimisation-based versions of their normal counterparts. To visualise attractor maps, I recommend using minimum and maximum bounds of 0.998 and 1 respectively.

Note that some LineStrings in the output may have all-NaN elevations and empty Lidar subclouds. This is due to issues with how the Lidar input files were produced; some NBRS extend beyond the boundaries of the relevant AHN3 tiles and can therefore not be associated with elevations. This issue bears no relevance to the implementation's effectiveness.

![Screenshot of preliminary edges and cross-sections in C_37HN2](readme_thumb.png?raw=true "Screenshot of preliminary edges and cross-sections in C_37HN2")

This work is licensed under an
[MIT license](https://opensource.org/licenses/MIT).
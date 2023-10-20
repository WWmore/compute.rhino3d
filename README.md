# Rhino Compute Server + Hops Component in Grasshopper

[![Build status](https://ci.appveyor.com/api/projects/status/unmnwi57we5nvnfi/branch/master?svg=true)](https://ci.appveyor.com/project/mcneel/compute-rhino3d/branch/master)
[![Discourse users](https://img.shields.io/discourse/https/discourse.mcneel.com/users.svg)](https://discourse.mcneel.com/c/rhino-developer/compute-rhino3d/90)

![https://www.rhino3d.com/compute](https://www.rhino3d.com/en/7.420921340460724505/images/rhino-compute-new.svg)



## Publication

This repository contains the implementation associated with the paper "Designing Asymptotic Geodesic Hybrid Gridshells", which can be found [here](https://doi.org/10.1016/j.cad.2022.103378). 
Please cite the paper if you use this code in your project. 

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{AGG2022,
      author       = {Schling, Eike<sup>1</sup> and Wang, Hui<sup>1</sup> and Hoyer, Sebastian and Pottmann, Helmut},
      title        = {Designing Asymptotic Geodesic Hybrid Gridshells},
      journal      = {Computer-Aided Design},
      volume       = {152},
      pages        = {1--17},
      year         = {2022},
      doi          = {10.1016/j.cad.2022.103378},
      url          = {https://www.huiwang.me/projects/6_project/}
}</code></pre>
  </div>
</section>

Implementation source :snake: code has been released in [Github :octocat:](https://github.com/WWmore/AAG). Welcome!

## Link local or remote server of Python implementation by Hops
A user-friendly design tool has been realized by integrating the optimization into the CAD system Rhinoceros3D. Our plugin allows the user to define the main inputs to the optimization, the initial mesh, the web type, strip width and optimization parameters like the number of iterations and weights. The results, namely the optimized mesh,
strip boundaries in piecewise quintic Bezier form, ruling vectors, the developable strips as well as their developments are returned as Rhinoceros3D geometry objects. The optimization is implemented in CPython and called from Grasshopper, Rhinoceros’ parametric design extension, using the [Hops component](https://developer.rhino3d.com/guides/compute/hops-component/). This allows the user to offload the actual computation to a more powerful remote machine, if desired. 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="hops.png" title="example image" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>


## Contributions
If you think this repoistory is useful, welcome to cite the paper and give a :star:. 
Welcom to work together to realize a Grasshopper plugin available to the architectural community.




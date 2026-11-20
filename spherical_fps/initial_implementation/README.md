# Initial independent implementation

`compare_independent_fps.cpp` benchmarks SphericalFPS independently against scalar RadiusFPS. `run_independent_fps.sh` compiles into root `bin/` and saves logs in `results/`.

Median host totals: centroid spherical 76.849 ms vs RadiusFPS 27.654 ms; sensor spherical 77.613 ms vs RadiusFPS 27.487 ms. Exact 128/128 indices matched.

# cem-optical-ray-tracer

For 2D-rendering install `python-shapely`

Run examples using `python -m demos.2dscene` and `python -m demos.3dscene`

### Global parameters

The following globals can be set before raytacing. All numbers are assumed to be positive and finite, anything else is undefined.

* **EPS**: float - A small number used to look ahead and behind the ray. Preferably around 1e-9. A ray which hits a surface does so 1e-15 to 1e-14 units away from the real surface due to rounding errors, EPS is used to counteract that.
* **RCOUNT**: integer - The number of rays (must be set before calling `raytracer.setup`)
* **TSIZE**: integer - The maximum number of steps done by the rays (must be set before calling `raytracer.setup`)
* **STEPSIZE**: float - The step size used in numerical integration
* **BOUNDARY**: float[2, 2] - The 2D plot size, has no impact on the tracing. Ordered as [[X1 X2] [Y1 Y2]]
* **NUMDIFF**: float - The resolution used in numerical gradient
* **SOLVER**: f(y: S, f: (S) -> S, step_size: float) -> S where S: [ float[N, 3] float[N, 3] ] - The numerical integrator used, given as a function which takes y an array [r p] where r and p are matrices of row vectors representing ray position and ray momentum. A function f such that dy/dt = f(y). And a step_size which is the step size to be used. The function returns y at the next time step with regard to the step size.

### Global lens arrays

For each lens an entry is typically created in every one of these arrays, with a shared array index corresponding to the order in which the lenses were added. Properties are typically accessed by being given the lens array index within parametric refractive indices or gradients. 

* **LENS_SHAPE** LensShape[] - An array of the primary shape of each lens. Contains position, sphere size, and plane normal.
* **LENS_INDEX** (float|f(r: float[N,3], i: integer) -> float[N,1])[] - The refractive indices of each lens. A refractive index is either a float (homogenous) or a function which given the row vectors of each ray position, and the lens array index, returns a column vector of all refractive indices at the positions r. A lens may have a defined refractive index outside of its shape, and is encouraged have it so. Rays outside the lens will be unaffected.
* **LENS_GRADIENT** (f(r: float[N,3], i: integer) -> float[N,3])[] - The gradients of each lens in cartesian coordinates. Each gradient must be defined by a function which takes the row vectors of the ray positions, and the lens array index, and returns the gradient as row vectors for each ray.
* **ABSORBER** boolean[] - Whether the lens should stop rays. Absorbers allow the raytracer to halt early.
* **SUBTRACTORS** LensShape[][] - An array of arrays of the shapes which are to be subtracted from some primary shape of a lens. There must be as many arrays as there lenses but the arrays may be empty if a particular lens has no subtractors.

### Global ray vectors

These are typically created after calling `raytracer.setup`, and should not be accessed before. The first time step values are initialized to 0, these are expected to be set by the user before tracing through `raytracer.ray_r[0]` and `raytracer.ray_p[0]`. After a call to `raytracer.trace()` the remaining time steps are filled in, the ray vectors should typically only be read after this.

* **ray_r** float[TSIZE,N,3] - a position vector for each time step and ray.
* **ray_p** float[TSIZE,N,3] - a momentum vector for each time step and ray.

# 
# Example 2D scene showing how to build and add simple lenses
# As well as set up constants for tracing
#

import numpy as np
import matplotlib.pyplot as plt
import raytracer as rt
from raytracer import Lens

# This sets the number of rays
rt.RCOUNT = 100
# This sets the total steps the ray can take
# Should not be too large
rt.TSIZE = 500
rt.BOUNDARY = [
    [-100, 100],
    [-100, 100]
]

# The integrator for gradients (default RK4)
rt.SOLVER = rt.Integrator.VVERLET
# The integrator step size (default 1)
rt.STEPSIZE = 1
# The numerical gradient resolution
rt.NUMDIFF = 1e-3

# A simple spherical lens with radius 30 and index 1.5
sphere = Lens.new_spherical(30).with_ri(1.5)

# A function of the index, with r as an array of vectors from the lens to the rays
# and i as the array index of the lens
# Returns a (1,N) vector of the index for each ray
def linear_index(r, i):
    # Access the radius from lens shape array
    radius = rt.LENS_SHAPE[i].size
    dist = rt.vnorm(r, axis=-1)[:,None]

    return 2. - dist / radius

# A function has been assigned as the index
# The gradient will be calculated using raytracer.Gradient.NUMERIC
grad_sphere = Lens.new_spherical(30).with_ri(linear_index)

# Position the lens somewhere else
grad_sphere.at([-50,10,0])

# Subtract a plane with normal -x from our sphere
# An array of LensShapes are expected
sphere.subtract([
    Lens.new_plane([-1,0,0]).lens_shape
])

# The subtracted plane remains at 0
sphere.at([10,0,0])

# Add lenses to the raytracer
sphere.add()
grad_sphere.add()

# Once we've set constants and lenses we can call setup
rt.setup()

# The ray vectors are now initialized, and can be read or written
rt.ray_r[0,:,0] = rt.BOUNDARY[0][0]
rt.ray_r[0,:,1] = np.linspace(-50, 50, rt.RCOUNT)
rt.ray_p[0,:,0] = 1

# Running the ray tracer
rt.trace()

# The results are now accessible through ray_r and ray_p
# We can draw simpler 2D scenes using the included draw function
# Draw red translucent rays along with lenses
rt.draw(('r', 0.2),'-')

# We may trace again
sphere.add()
grad_sphere.add()

rt.RCOUNT = 20

rt.setup()

rt.ray_r[0,:,0] = rt.BOUNDARY[0][0]
rt.ray_r[0,:,1] = np.linspace(-10, 10, rt.RCOUNT)
rt.ray_p[0,:,0] = 1

rt.trace()

# Draw rays on the existing plot
rt.draw_rays(('b', 0.2))

plt.show()
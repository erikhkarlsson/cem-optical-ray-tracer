# 
# An example 3D scene with a 2 step ray tracing process
# utilizing heavy post processing from the ray positions
# and directions to draw a colored scene with light and shadows
#

import numpy as np
import scipy
import matplotlib.pyplot as plt
import raytracer as rt
from raytracer import Lens

# For simple scenes without gradients a small number is usually enough
rt.TSIZE = 6
rt.BOUNDARY = [
    [-100, 100],
    [-100, 100]
]

# Create lens objects, we'll trace the same scene 2 times
# so we save the lens in an array
scene = []
scene.extend([
    Lens.new_spherical(30).set_absorber(),
    Lens.new_plane([0,0,1]).at([0,0,-50]).set_absorber(),
    Lens.new_spherical(10).at([-60,-10,-5]).with_ri(1.5),
])

# Add all the lenses to raytracer buffer
for l in scene:
  l.add()

# This dictates a quad which the rays will intersect
port_size = 10
# A 100 x 100 pixel image
img_size = 100

# 1 ray per pixel
rt.RCOUNT = img_size*img_size

# Sets up the relevant numpy arrays and clears the buffer
rt.setup()

# This creates the y and z intersection points on our viewport quad for each ray
vy, vz = np.meshgrid(
    np.linspace(-port_size, port_size, img_size),
    np.linspace(-port_size, port_size, img_size),
)

# Map all the rays to intersection points on the view port quad, and place it x = -100
rt.ray_r[0,:,0] = -100
rt.ray_r[0,:,1] = vy.reshape(rt.RCOUNT)
rt.ray_r[0,:,2] = vz.reshape(rt.RCOUNT)

# Sets the intersection angles at the viewport, such that a 90 degree frustum is formed.
rt.ray_p[0,:,0] = port_size
rt.ray_p[0,:,1] = vy.reshape(rt.RCOUNT)
rt.ray_p[0,:,2] = vz.reshape(rt.RCOUNT)

rt.ray_p[0,:,:] /= rt.vnorm(rt.ray_p[0,:,:], axis=-1)[:,None]

# Trace the scene and update our ray_r and ray_p
rt.trace()

# Check the last known medium of the ray
# -1 is the vacuum
hits = rt.where_is_ray(rt.ray_r[-1])

ray_normals = np.nan_to_num(rt.sample_normal(rt.ray_r[-1]))
# X,Y,Z normals mapped from 0 to 1 with inverted X.
ball_color = (0.5 * ([1,1,1] + ray_normals @ [[-1,0,0],[0,1,0],[0,0,1]]))

# Checkerboard
floor_color = np.clip(
  np.abs(scipy.signal.square(np.pi * rt.ray_r[-1,:,:2] / 20) @ [[1],[1]]),
  0.5,
  1
) * np.ones(3)

# Color look up table
colors = np.array([
    np.ones((rt.RCOUNT, 3)) * (0.52,0.8,0.92),  # Sky
    ball_color,
    floor_color,
    np.ones((rt.RCOUNT, 3)),                    # Glass lens
], dtype=float)

ray_cols = colors[hits + 1,np.arange(hits.shape[0])]

# Add pretty shade and highlights
light_dir = np.array([[-2,1,1]]) / np.sqrt(6)
illumination = np.vecdot(light_dir, ray_normals)

specular = np.clip(1e-3 / (1 - illumination), 0, 1)

np.multiply(ray_cols, np.maximum(0.1, illumination)[:,None]**0.5, out=ray_cols, where= (hits != -1)[:,None])
ray_cols += specular[:,None]

# Trace shadows
residual_rays = hits != -1
residual_r = rt.ray_r[-1,residual_rays,:]
rt.RCOUNT = np.count_nonzero(residual_rays)

for l in scene:
  l.add()

rt.setup()

rt.ray_r[0] = residual_r
rt.ray_p[0] = light_dir

rt.trace()

direct_shadow = np.clip(np.where(
    rt.where_is_ray(rt.ray_r[-1]) == -1,
    np.vecdot(rt.ray_p[-1], light_dir),
    0.5
), 0.5, 1)

ray_cols[residual_rays,:] *= (direct_shadow)[:,None]

ray_cols = np.clip(ray_cols, 0, 1)
plt.imshow(ray_cols.reshape((img_size,img_size,3)), origin='lower')
plt.show()
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import lineStyles
import numpy as np
from numpy.linalg import vector_norm as vnorm
import matplotlib as mplt
import matplotlib.pyplot as plt
import shapely
import copy

EPS = 1e-9           # A small number
# Default values
RSIZE = 3            # Number of dimensions
RCOUNT = 1           # Number of rays
TSIZE = 100          # Number of timesteps
STEPSIZE = 1
# Size of 2D plot view
BOUNDARY = [         #
    [0, 100],
    [-50, 50]
]
# Numerical gradient size
NUMDIFF = 1e-3

LENS_SHAPE = np.empty(0)
# lens surface refractive index
LENS_INDEX = np.empty(0)
# lens refractive gradient
LENS_GRADIENT = np.empty(0)
ABSORBER = np.empty(0, dtype=bool)
SUBTRACTORS = []

SOLVER = None

shape_buffer = []
ri_buffer = []
grad_buffer = []
absorb_buffer = []
subtractor_buffer = []

ray_r = np.empty((0,0,0))
ray_p = np.empty((0,0,0))

class Integrator:
  # Runge Kutta 4 integrator
  # Estimates 4 slopes and optimally (magically) weights them
  def RK4(y, f, step_size=1):
    k1 = f(y)
    k2 = f(y + 0.5*k1*step_size)
    k3 = f(y + 0.5*k2*step_size)
    k4 = f(y + k3*step_size)

    return y + step_size*(k1 + 2*k2 + 2*k3 + k4) / 6

  # Velocity Verlet integrator
  # This requires the assumption that d²r/dt² = dp/dt = f
  # By inserting equation 2 into equation 1 in our Hamiltonian-Lagrangian
  # we get d²r/dt² = dp/dt = 0.5 * grad ( [n(r)]² )
  def VVERLET(y, f, step_size=1):
    r, p = y
    a = f(y)[1]
    new_r = r + p*step_size + 0.5 * a * step_size*step_size
    new_a = f(np.array([new_r, p]))[1]
    new_p = p + 0.5 * (a + new_a) * step_size

    return np.array([new_r, new_p])

  def EULER(y, f, step_size=1):
    return y + step_size * f(y)

class Gradient:
  # Approximate 3D gradient at a point
  # Uses 3 point central difference for 6 points in 3 axes around the ray
  # By using the central difference the sampling at the middle point dissappears
  # but the accuracy is still retained
  def numeric_gradient(r, i):
    ri = LENS_INDEX[i]
    if not callable(ri): return 0

    forward_ri = np.empty(r.shape)
    backward_ri = np.empty(r.shape)

    # Create cartesian offset vectors (x, y, z)
    offset = NUMDIFF
    offset_v = offset * np.identity(RSIZE)

    forward_ri[:,None,0] = ri(r + offset_v[0], i)
    forward_ri[:,None,1] = ri(r + offset_v[1], i)
    forward_ri[:,None,2] = ri(r + offset_v[2], i)

    backward_ri[:,None,0] = ri(r - offset_v[0], i)
    backward_ri[:,None,1] = ri(r - offset_v[1], i)
    backward_ri[:,None,2] = ri(r - offset_v[2], i)

    return 0.5 * (forward_ri**2 - backward_ri**2) / offset

  NUMERIC = numeric_gradient
  LUNEBURG = lambda r, i: -2*r / LENS_SHAPE[i].size**2
  EATON180 = lambda r, i: -2*r*LENS_SHAPE[i].size / vnorm(r, axis=-1)[:,None]**3
  MAXWELL = lambda r, i: -16*LENS_INDEX[i]*LENS_SHAPE[i].size**4 * r / (LENS_SHAPE[i].size*LENS_SHAPE[i].size + np.vecdot(r,r)[:,None])**3

  class Index:
    def lb(r, i):
      print(r, i)
      return np.sqrt(2 - np.vecdot(r,r)[:,None] / (LENS_SHAPE[i].size*LENS_SHAPE[i].size))

    LUNEBURG = lambda r, i: np.sqrt(2 - np.vecdot(r,r)[:,None] / (LENS_SHAPE[i].size*LENS_SHAPE[i].size))
    EATON = lambda r, i: np.sqrt(2 * LENS_SHAPE[i].size / vnorm(r, axis=-1)[:,None] - 1)
    MAXWELL = lambda r, i: 2 / (1 + np.vecdot(r,r)[:,None] / (LENS_SHAPE[i].size*LENS_SHAPE[i].size))

class LensShape:
  SPHERICAL = 0
  PLANE = 1

  ty = None
  pos = np.array([0, 0, 0])
  size = 50
  pnorm = np.array([0, 0, 0])

  def patch(self, c=None):
    match self.ty:
      case LensShape.SPHERICAL:
        return mplt.patches.Circle(
            self.pos,
            self.size,
            facecolor=c,
        )
      case LensShape.PLANE:
        return mplt.patches.Rectangle(
            self.pos - 0.5 * 5e3 * np.array([1,0,0]),
            5e3,
            0.5 * 5e3,
            rotation_point = (self.pos[0], self.pos[1]),
            angle = np.degrees(np.arctan2(self.pnorm[0], -self.pnorm[1])),
            facecolor=c,
        )

  def contains(self, p):
    v = p - self.pos
    match self.ty:
      case LensShape.SPHERICAL:
        return vnorm(v, axis=-1) < self.size
      case LensShape.PLANE:
        return np.linalg.vecdot(self.pnorm, v) < 0

    return False

  def project_to(self, p):
    match self.ty:
      case LensShape.SPHERICAL:
        return self.pos + normalize(p - self.pos) * self.size
      case LensShape.PLANE:
        return p - self.pnorm * np.vecdot(p - self.pos, self.pnorm)[:,None]

  def intersect(self, p, d):
    v = p - self.pos
    match self.ty:
      case LensShape.SPHERICAL:
        # By how much the ray misses the centre
        rejection = -v - d * np.vecdot(-v, d)[:, None]
        rejnorm = vnorm(rejection, axis=-1)

        # How far away is the ray from closest contact
        projnorm = np.vecdot(d, -v)
        # Distance between surface point and closest contact point
        recession = np.sqrt(np.maximum(self.size*self.size - rejnorm*rejnorm, 0))

        # Entrance and exit intersections
        tlow = projnorm - recession
        thigh = projnorm + recession

        # We eliminate intersections behind the ray,
        # as well as intersections so close that floating point error pins the ray
        # to the surface (tests show EPS should be greater than 1e-15)
        tlow[tlow <= EPS] = np.inf
        thigh[thigh <= EPS] = np.inf

        # If the closest approach is further away than lens radius we obviously
        # do not have an intersection. Otherwise, we pick the closest intersection.
        return np.where(
          rejnorm < self.size,
          np.minimum(tlow, thigh),
          np.inf
        )[:, None]

      case LensShape.PLANE:
        # depth of ray into plane
        z = -np.vecdot(v, self.pnorm)
        # The cosine of the angle to the plane is
        # the adjacent (ray depth z) over the hypotenuse (ray path t1).
        # Equivalent to the dot product of the ray direction and plane normal
        dz = np.vecdot(d, self.pnorm)
        t1 = np.inf * np.ones(z.shape)[:,None]
        np.divide(z,dz,out=t1[...,0],where= dz != 0)

        return np.where( t1 > EPS, t1, np.inf )

class Lens:
  lens_shape = None
  ri = 1
  gradient = None
  absorber = False
  subtractors = None

  # Start constructing a spherical lens
  def new_spherical(size):
    lens = Lens()
    lens.lens_shape = LensShape()
    lens.lens_shape.ty = LensShape.SPHERICAL
    lens.lens_shape.size = size
    lens.gradient = Gradient.NUMERIC
    lens.subtractors = []

    return lens

  # Start constructing a dielectric plane
  def new_plane(normal):
    lens = Lens()
    lens.lens_shape = LensShape()
    lens.lens_shape.ty = LensShape.PLANE
    lens.lens_shape.pnorm = normalize(np.array(normal))
    lens.gradient = Gradient.NUMERIC
    lens.subtractors = []

    return lens

  # Set position
  def at(self, position):
    self.lens_shape.pos = np.array(position)

    return self

  # Set refractive index
  def with_ri(self, ri):
    self.ri = ri

    return self

  # Set GRIN
  def with_gradient(self, f):
    self.gradient = f

    return self

  def set_absorber(self, absorber=True):
    self.absorber = absorber

    return self

  def subtract(self, subtractors):
    self.subtractors.extend(subtractors)

    return self

  # Add the lens to the environment
  def add(self):
    global shape_buffer, ri_buffer, grad_buffer, absorb_buffer, subtractor_buffer
    shape_buffer.append(self.lens_shape)
    ri_buffer.append(self.ri)
    grad_buffer.append(self.gradient)
    absorb_buffer.append(self.absorber)
    subtractor_buffer.append(self.subtractors)



# if the ray is inside lens i
def is_ray_in(ray_r, i):
  if i == -1: return False

  inside = LENS_SHAPE[i].contains(ray_r)
  for sub in SUBTRACTORS[i]:
    inside = inside > sub.contains(ray_r)

  return inside

# gives the array index of which shape the ray is inside of
def where_is_ray(ray_r):
  inside = -np.ones(ray_r.shape[0], dtype=np.int64)
  for i in range(len(LENS_SHAPE)):
    inside[is_ray_in(ray_r, i)] = i

  return inside

# Find the next, closest, intersection point ahead of the ray
# See "Slab Method"
def next_encounter(ray_r, ray_p):
  ray_d = normalize(ray_p)
  # t is our parameter which we wish to optimize, t comes from the parametrization
  # of our optical path. For simplicity, and in our case, we can let t represent
  # the straight distance which the ray travels.
  t = np.inf * np.ones((ray_r.shape[0], 1))
  for (i, s) in enumerate(LENS_SHAPE):
    intersect = s.intersect(ray_r, ray_d)
    xp = ray_r + ray_d * intersect

    for sub in SUBTRACTORS[i]:
      intersect = np.minimum( intersect, sub.intersect(ray_r, ray_d) )

    t = np.minimum(t, intersect)

  # Let 0 * infnity = 0, which is otherwise undefined
  q = np.zeros(ray_r.shape)
  np.multiply(t, ray_d, out=q, where= ray_d != 0)

  return ray_r + q

# refractive index at each ray position
def sample_ri(ray_r, whereabout=None):
  if whereabout is None:
    whereabout = where_is_ray(ray_r)

  ri = np.ones((ray_r.shape[0], 1))
  for i in range(len(LENS_INDEX)):
    if callable(LENS_INDEX[i]):
      ri[whereabout == i,:] = LENS_INDEX[i](ray_r - LENS_SHAPE[i].pos, i)[whereabout == i,:]
    else:
      ri[whereabout == i,:] = LENS_INDEX[i]

  return ri

def sample_normal(ray_r):
  normal_of_shape = lambda s, v: np.broadcast_to(s.pnorm, v.shape) if s.ty == LensShape.PLANE else normalize(v)
  normal = np.nan * np.empty(ray_r.shape)
  dist = np.inf * np.ones(ray_r.shape[0])

  for (i,s) in enumerate(LENS_SHAPE):
    surf_points = s.project_to(ray_r)
    t = vnorm(ray_r - surf_points, axis=-1)
    n = normal_of_shape(s, ray_r - s.pos)

    if len(SUBTRACTORS[i]) > 0:
      sub_surf_points = np.empty((len(SUBTRACTORS[i]),) + ray_r.shape)
      t_sub = np.empty((len(SUBTRACTORS[i]), ray_r.shape[0]))
      n_sub = np.empty((len(SUBTRACTORS[i]),) + ray_r.shape)

      # query points
      for (j,sub) in enumerate(SUBTRACTORS[i]):
        # query all points for all subtractors
        sub_surf_points[j] = sub.project_to(ray_r)
        # only consider point if the subtractor has subtracted that portion from the solid
        t_sub[j] = np.where(
            s.contains(sub_surf_points[j]),
            vnorm(ray_r - sub_surf_points[j], axis=-1),
            np.inf
        )

        n_sub[j] = -normal_of_shape(sub, ray_r - sub.pos)

      flattened_points = sub_surf_points.reshape((-1,RSIZE))

      # filter points
      for (j,sub) in enumerate(SUBTRACTORS[i]):
        # invalidate points on the original solid removed by the subtractor
        t[sub.contains(surf_points)] = np.inf

        # invalidate points between the original solid and the subtractor already subtracted by another subtractor
        invalid_points = sub.contains(flattened_points).reshape((-1,ray_r.shape[0]))
        # Do not self invalidate
        invalid_points[j] = False

        t_sub[invalid_points] = np.inf

      t_shape = np.concat((t[None], t_sub))
      n_shape = np.concat((n[None], n_sub))

      nearest_shape = np.argmin(t_shape, axis=0)
      t = np.min(t_shape, axis=0)
      n = nearest_shape[:,None].choose(n_shape)
      # ^^^ if this crashes and you have more than 64 shapes replace with
      # n_shape[nearest_shape,np.arange(nearest_shape.shape[0])]

    normal[dist >= t,:] = n[dist >= t,:]

    dist = np.minimum(dist, t)

  return normal

# lens gradient at each ray position
def sample_grin(ray_r):
  g = np.zeros(ray_r.shape)
  whereabout = where_is_ray(ray_r)
  for i in range(len(LENS_GRADIENT)):
    v = ray_r - LENS_SHAPE[i].pos
    g[whereabout == i,:] = np.broadcast_to((LENS_GRADIENT[i])(v,i), g.shape)[whereabout == i,:]

  return g


# calculates new ray after refraction
def snells(n1, n2, ray_r, ray_p, normal):
  ray_d = normalize(ray_p)
  sin1 = np.cross(normal, ray_d)
  sin2 = n1 * sin1 / n2

  # Coordinate along the surface
  sin2tangent = np.cross(normal, sin2)
  #tangent /= EPS + vnorm(tangent, axis=-1)[:, None]

  sin2 = vnorm(sin2, axis=-1)[:, None]
  cos2 = np.sqrt(np.maximum(1 - sin2*sin2, 0))

  refraction = n2 * (-sin2tangent - normal * cos2)
  reflection = ray_p - 2 * normal * np.vecdot(normal, ray_p)[:, None]

  # Total internal reflection occurs when (n1/n2)*sin1 > 1
  new_ray = np.where(
      sin2 <= 1,
      refraction,
      reflection
  )

  return new_ray

def normalize(v):
  v = np.array(v)
  nv = np.zeros(v.shape)
  norm = vnorm(v, axis=-1)[...,None]
  np.divide(v, norm, out=nv, where= norm!=0 )

  return nv

# Set globals
def setup():
  # set default solver
  global SOLVER
  if SOLVER is None:
    SOLVER = Integrator.RK4

  # load buffers into the evironment
  global LENS_SHAPE, LENS_INDEX, LENS_GRADIENT, ABSORBER, SUBTRACTORS
  LENS_SHAPE = np.array(shape_buffer)
  LENS_INDEX = np.array(ri_buffer)
  LENS_GRADIENT = np.array(grad_buffer)
  ABSORBER = np.array(absorb_buffer)
  SUBTRACTORS = list(subtractor_buffer)

  # clear buffers
  shape_buffer.clear()
  ri_buffer.clear()
  grad_buffer.clear()
  absorb_buffer.clear()
  subtractor_buffer.clear()

  global ray_r, ray_p
  ray_r = np.nan * np.ones( (TSIZE, RCOUNT, RSIZE) )
  ray_p = np.nan * np.ones( (TSIZE, RCOUNT, RSIZE) )
  ray_r[0] = 0
  ray_p[0] = 0

def trace():
  for t_step in range(TSIZE-1):
    last_ray_d = normalize(ray_p[t_step])

    # Find the next shape intersection point straight ahead
    intersection_points = (next_encounter(ray_r[t_step], ray_p[t_step]), ray_p[t_step])

    # Solve RK4 for state vector
    # The lambda expression is our system of equations (Hamiltonian-Lagranian)
    # dr/dt = p
    # dp/dt = 0.5 * grad ( [n(r)]^2 )
    integration_points = SOLVER(
      np.array([ ray_r[t_step], ray_p[t_step] ]),
      lambda y: np.array([ y[1], 0.5 * sample_grin(y[0]) ]),
      STEPSIZE
    )

    # Update rays with analytical solution for gradient free media
    # otherwise use the numerically integrated solution
    ray_r[t_step+1], ray_p[t_step+1] = np.where(
      vnorm(sample_grin(ray_r[t_step] + last_ray_d * EPS), axis=-1)[:,None] == 0,
      intersection_points,
      integration_points
    )

    # When a ray in gradient free media has no intersections
    # it will step to infinity.
    # Stop the ray-tracer if all rays are at infinity
    if np.logical_or(
        np.isinf(ray_r[t_step+1]).any(axis=-1),
        (ray_p[t_step+1] == 0).all(axis=-1)
    ).all():
      i = np.maximum(0, np.argmax( np.isnan(ray_r).any(axis=-1), axis=0 ) - 1)
      np.nan_to_num(ray_r, nan=ray_r[i,np.arange(i.shape[0])], posinf=np.inf, neginf=-np.inf, copy=False)
      np.nan_to_num(ray_p, nan=ray_p[i,np.arange(i.shape[0])], posinf=np.inf, neginf=-np.inf, copy=False)
      return t_step + 1

    ray_d = normalize(ray_p[t_step+1])

    # Sample a bit ahead and behind the ray,
    # just enough for them to be on different sides of a surface.
    # We do not know if ray_r[t_step+1] ended up behind or ahead of the surface.
    # Smaller EPS is greater, but ray_r[t_step+1] might end up further away
    # from the surface than EPS, depending on how long the jump was.
    ahead = ray_r[t_step+1] + ray_d * EPS
    behind = ray_r[t_step+1] - ray_d * EPS

    ahead_whereabout = where_is_ray(ahead)
    behind_whereabout = where_is_ray(behind)

    # start testing for snell's law
    ahead_ri = sample_ri(ahead, ahead_whereabout)
    behind_ri = sample_ri(behind, behind_whereabout)

    has_intersected = np.logical_and(
        ahead_ri[...,0] != behind_ri[...,0],
        ahead_whereabout != behind_whereabout
    )

    # All shapes do not have a well defined normal everywhere.
    # When 2 shapes intersect, the normal is of which shape's surface defines
    # the intersection. If both shapes define an intersection (a 2D point, or 3D edge)
    # then the normal is undefined, and our raytracer does not model the difraction.
    # If a shape has a higher priority (higher index) the shape defines the normal
    # at the intersect, but only if it has a well defined normal.
    normal = sample_normal(ray_r[t_step+1])
    normal[np.vecdot(ray_d, normal) >= 0,:] *= -1

    # apply snell's law if ray has intersected
    ray_p[t_step+1,has_intersected,:] = snells(
        behind_ri,
        ahead_ri,
        ray_r[t_step+1],
        ray_p[t_step+1],
        normal
    )[has_intersected,:]

    # Absoroption detection
    ray_r[t_step+1], ray_p[t_step+1] = np.where(
        np.logical_and(
            ahead_whereabout != -1,
            ABSORBER[ahead_whereabout]
        )[:,None],
        (ahead, np.broadcast_to(0, ahead.shape)),
        (ray_r[t_step+1], ray_p[t_step+1])
    )

  print("\u001b[1m\u001b[31m---!! RAYS NOT FULLY RESOLVED !! ---\x1b[0m\n\u001b[36m- Increase \u001b[4mTSIZE\x1b[0m (" + str(TSIZE) + ")")
  return TSIZE

def draw_rays(ray_color='r', ls='-', m=None):

  r = np.where(
    np.isinf(ray_r),
    1e10 * ray_p,
    ray_r
  )

  if m is not None:
    plt.scatter(ray_r[...,0], ray_r[...,1], marker=m)

  if ray_color is not str or ray_color not in mplt.colormaps:
    rays = np.moveaxis(r[:,:,:2], 1, 0)
    lc = mplt.collections.LineCollection(rays, linestyles=ls, colors=ray_color)
    return plt.gca().add_collection(lc)
  else:
    x = np.vstack((r[:-1,:,0].reshape(-1), r[1:,:,0].reshape(-1)))
    y = np.vstack((r[:-1,:,1].reshape(-1), r[1:,:,1].reshape(-1)))
    L = np.where(np.isinf(ray_r).any(axis=-1).all(axis=1))[0]

    if len(L) == 0:
      L = 0
    else:
      L = L[0]

    segments = np.moveaxis(np.dstack((x,y)), 1, 0)
    lc = mplt.collections.LineCollection(segments, linestyles=ls, cmap=ray_color)
    lc.set_array( np.linspace(0, 2, L*RCOUNT) )

    return plt.gca().add_collection(lc)

# Ray and lens plotting
def draw(ray_color='r', style='-', marker=None):
  patches = []
  for (i,s) in enumerate(LENS_SHAPE):
    fill_col = '#ccc'
    edge_col = 'none'
    if LENS_INDEX[i] == 1:
      fill_col = ('C0',0.05)
      edge_col = 'C0'
    if ABSORBER[i]:
      fill_col = '#000'

    p = s.patch(c=fill_col)
    p.set_ls(':')
    p.set_lw(1)
    p.set_ec(edge_col)
    patches.append(p)
    plt.gca().add_patch(p)

    if len(SUBTRACTORS[i]) > 0:
      bound = shapely.Polygon([
          [BOUNDARY[0][0], BOUNDARY[1][0]],
          [BOUNDARY[0][1], BOUNDARY[1][0]],
          [BOUNDARY[0][1], BOUNDARY[1][1]],
          [BOUNDARY[0][0], BOUNDARY[1][1]],
      ])

      geometry = [
          lambda x: shapely.Point(x.pos[:2]).buffer(x.size),
          lambda x: shapely.LineString([
              [x.pos[0] - 5000 * x.pnorm[1], x.pos[1] + 5000 * x.pnorm[0]],
              [x.pos[0] + 5000 * x.pnorm[1], x.pos[1] - 5000 * x.pnorm[0]],
          ]).buffer(-10000, single_sided=True)
      ]

      a = geometry[s.ty](s)
      b = None
      for sub in SUBTRACTORS[i]:
        sub_geometry = geometry[sub.ty](sub)
        b = b and b.union(sub_geometry) or sub_geometry

      clip = np.transpose(a.intersection(bound).difference(b).exterior.xy)

      clip_path = mplt.path.Path(clip)
      clip_patch = mplt.patches.PathPatch(clip_path, facecolor='none', lw=1, edgecolor=edge_col, linestyle=':', transform=plt.gca().transData)
      p.set_clip_path(clip_patch)
      plt.gca().add_patch(clip_patch)

  draw_rays(ray_color, style, marker)
  plt.gca().set_aspect('equal')
  plt.xlim(BOUNDARY[0])
  plt.ylim(BOUNDARY[1])
  plt.grid(True)

  return patches

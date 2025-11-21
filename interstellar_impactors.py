"""
Generates a synthetic population of interstellar meteoroids using the Probabilistic method 
(Marceta, D.: Synthetic population of interstellar objects in the Solar System, Astronomy and Computing, vol. 42, 2023)

Methodology:
    
1. We generated a population inside 1 au using two different kinematics (slow and fast) and
calculated the time needed for all objects to leave the 1 au sphere, which is about 78 days for
both kinematics. This means that any ISO within 1 au would impact the Earth (if at all)
within those 80 days. If the analyzed time is shorter, not all ISOs inside 1 au will have the
opportunity to impact.

2. We calculate the maximum distance from which an ISO can reach Earth in 80 days,
assuming a direct path toward the Sun and a speed of 100 km/s, which is about 4.6 au.


3. We generate a large population of ISOs inside 5.6 au and calculate the MOID for all of
them. If the MOID is smaller than Earth’s effective radius and the MOID position is reached
within 80 days, we consider those ISOs as potential impactors. If the MOID position is
reached after 80 days, we do not consider them as impactors, since that would introduce a
bias toward ISOs initially outside 1 au. The total number of generated ISOs is approximately
16 billion, of which about 30 thousand passed through the torus within 80 days.

4. For potential impactors, we propagate the orbit with a small time step (1 second) and
determine the time spent in the effective torus.


Input:
    rmin: inner radius of the heliocentric shell (au)
    rmax: outer radius of the heliocentric shell (au)
    n0: number-density of the ISOs in the interstellar space (unperturbed by the Sun's gravity)
    v_min: minimum allowed interstellar speed (m/s)
    v_max: maximu allowed interstellar speed (m/s) 
    u_Sun:  u-component of the Sun's velocity w.r.t. LSR (m/s) 
    v_Sun: v-component of the Sun's velocity w.r.t. LSR (m/s) 
    w_Sun: w-component of the Sun's velocity w.r.t. LSR (m/s) 
    sigma_vx: standard deviation of x-component of ISOs' velocities w.r.t. LSR (m/s)
    sigma_vy: standard deviation of y-component of ISOs' velocities w.r.t. LSR (m/s)
    sigma_vz: standard deviation of z-component of ISOs' velocities w.r.t. LSR (m/s)
    vd: vertex deviation (radians)
    va:  assymetric drift (m/s)
    R_reff:  refference radius of the Sun (m)
    speed_resolution:  resolution of magnitudes of interstellar velocities (for numerical integration and inverse interpolation)
    angle_resolution: resolution of galactic longitude (for numerical integration and inverse interpolation)
    dr: increament step for heliocentric distance used for numerical integration and inverse interpolation (au)
    
        
    
Output (synthetic samples of orbital elements):
    q - perihelion distance (au)
    e - eccentricity
    E - eccentric (hyperbolic) anomaliy (radians)
    inc - orbital inclination (radians])
    Omega - longitude of ascending node (radians)
    omega - argument of perihelion (radians) 
    v - interstellar speed (m/s)
    l - galactic longitude of the incoming direction (rad)
    b - galactic latitude of the incoming direction (rad)
"""
import numpy as np
from synthetic_population import synthetic_population, orb2cart, gal2ecl_cart
from auxiliary_functions import cart2orb, kepler, spherical_coor, ecl2eq_spherical, moid
from astropy.constants import au, R_earth, GM_sun, GM_earth
import sys

run_number = int(sys.argv[1]) # for parallelization



n0 = 100  # Interstellar number density (set to an extremely large value to ensure a sufficient number of objects within the torus, approximately 3.6e-8 au^-3)
required_number_of_impacts = int(750000) # the number of impact we require for every run
inflation_factor = 1000. # inflaction factor to collect more impacts

v_min = 1e3  # minimum allowed interstellar speed of ISOs
v_max = 1e5  # maximum allowed interstellar speed of ISOs


number_of_impacts = 0
step_final_coarse = 100
step_final_dense = int(1e5)


total_time = 78 * 86400 # one year


dr = total_time * v_max / au.value

output_torus_file = 'results/interstellar_impactors_' +  str(run_number) + '.txt'


progress_file = 'results/progress_' +  str(run_number) + '.txt'
# Earth
R = R_earth.value * inflation_factor  # Earth' radius (m) (inflated)

# torus
r_major = 1.  # major radius of the torus (au)
r_minor = R/au.value  # minor radius of the torus (au)

# we inflate the torus by addtional factor of sqrt(2) since this is the maximum ratio between effective and physical radius of the Earth
r_in = r_major - r_minor * np.sqrt(2)  # inner radius of the torus
r_out = r_major + r_minor * np.sqrt(2)  # outer radius of the torus

r_model = r_major + r_minor*np.sqrt(2) + dr # radius of the model sphere where we generate ISOs

v_earth = np.sqrt(GM_sun.value/au.value) # Earth's orbital velocity
v_esc_earth = np.sqrt(2*GM_earth.value/R_earth.value) # Earth's escape velocity

# Kinematics of the ISO population (see (Marceta, D.: Synthetic population of interstellar objects in the Solar System, Astronomy and Computing, vol. 42, 2023))
# Motion of the Sun wrt LSR
u_Sun=1e4    # u-component of the solar motion with respect to the LSR (m/s)
v_Sun=1.1e4  # w-component of the solar motion with respect to the LSR (m/s)
w_Sun=7e3    # w-component of the solar motion with respect to the LSR (m/s)

# dispersions of galactic velocity components of ISOs
sigma_vx=3.1e4 # dispersion of the u-component of ISOs with respect to the LSR when far from the Sun (m/s)
sigma_vy=2.3e4 # dispersion of the v-component of ISOs with respect to the LSR when far from the Sun (m/s)
sigma_vz=1.6e4 # dispersion of the w-component of ISOs with respect to the LSR when far from the Sun (m/s)


# Resulting population of impactors and torus passers
q_torus = [] #  perihelion distance
e_torus = [] #  eccentricity
E_initial_torus = [] #  eccentric (hyperbolic) anomaly
inc_torus = []  # inclination
Omega_torus = []  # longitude of the ascending node
omega_torus = []  # argument of perihelion
RA_torus = []  # RA of the radiant
DEC_torus = []  # DEC of the radiant
sun_lon_torus = [] # solar longitude at the point of impact
v_gc_torus = []  # geocentric speed at the point of impact
v_hc_torus = []  # helicentric speed at the point of impact
v_hc_inf_torus = []  # interstellar speed of ISOs
inc_g_torus = []  # inclination wrt galactic plane
r_eff_torus = []  # Earth's effective radius
moid_torus = []  # MOID
M_moid_torus = []  # mean anomaly of MOID
M_initial_torus = []  # initial mean anomaly
t_imp_torus = []  # time of impact
mm_torus = []  # mean motion of ISOs
IDs = []  # IDs of ISOs

# output file header
header = ("perihelion_distance(au), eccentricity, initial hyperbolic_anomaly, mean anomaly at impact, MOID, "
          "inclination(wrt_ecliptic), inclination(wrt_galactic_plane), "
          "longitude_of_ascending_node, argument_of_perihelion, "
          "v_helicentric_at_infinity, v_helicentric_at_1au, "
          "v_geocentric, effective radius, radiant_RA, radiant_DEC, sun_longitude")

population_number = 0  # we generate small population batches, and this is their counter
total_objects = 0  # total number of generated objects

while number_of_impacts < required_number_of_impacts:

    # generating synthetic population of ISOs
    q, e, E, inc, Omega, omega = synthetic_population(rm = r_model,  n0=n0, v_min=v_min, v_max=v_max, 
                                                                    u_Sun=u_Sun, v_Sun=v_Sun, w_Sun=w_Sun, 
                                                                    sigma_vx=sigma_vx, sigma_vy=sigma_vy, sigma_vz=sigma_vz, 
                                                                    vd=np.deg2rad(7), va=0, R_reff=696340000.,
                                                                    speed_resolution=100, angle_resolution=90, dr=0.1, 
                                                                    d_ref=1000, d=[], alpha=[])
    

    a = q / (1 - e) * au.value # semi-major axis
    
    r = a * (1 - e * np.cosh(E))  # helicentric distance
    
    total_objects += len(q)  
    
    filter_q = q < r_out # we remove all with perihelion distance larger than the outer radius of the torus
    
    q = q[filter_q]
    e = e[filter_q]
    E = E[filter_q]
    inc = inc[filter_q]
    Omega = Omega[filter_q]
    omega = omega[filter_q]
    a = a[filter_q]
    r = r[filter_q]
    
    filter_outbound = np.logical_not(np.logical_and(r/au.value > r_out, E > 0)) # we remove all ISOs which are beyond the torus and leaving solar system
    
    q = q[filter_outbound]
    e = e[filter_outbound]
    E = E[filter_outbound]
    inc = inc[filter_outbound]
    Omega = Omega[filter_outbound]
    omega = omega[filter_outbound]
    r = r[filter_outbound]
    a = a[filter_outbound]
    
    v = np.sqrt(GM_sun.value * (2 / r - 1 / a)) # initial helicentric speed
  
    r_max = 1 + r_minor * np.sqrt(2) + (v * total_time)/au.value # maximum heliocentric distance from which an ISO can reach Earth in analyzed time frame
    filter_r = r/au.value < r_max # we remove all ISOs which move toward the Sun but don't have enough time to reach the torus
    
    q = q[filter_r]
    e = e[filter_r]
    E = E[filter_r]
    inc = inc[filter_r]
    Omega = Omega[filter_r]
    omega = omega[filter_r]
    r = r[filter_r]
    a = a[filter_r]
    
    v_hc_inf = np.sqrt(-GM_sun.value / a)
    
    mm = np.sqrt(-GM_sun.value / a**3)
    
    M = e * np.sinh(E) - E
    
    # conversion from galactic to ecliptic reference frame (generator works in galactic frame)

    inc_e = np.zeros_like(q)
    Omega_e = np.zeros_like(q)
    omega_e = np.zeros_like(q)
    
    MOID = []
    M_MOID = []
    for i in range(len(q)):
        xg, yg, zg, vxg, vyg, vzg = orb2cart(omega[i], Omega[i], inc[i], e[i], a[i], E[i], GM_sun.value)  # Converting orbital elements to Cartesian coordinates (galactic reference frame) (au)
        # Since we only require the position here (not the velocity vector), it is simply a geometric transformation, "mu" is not used, and "a" can be given in au.
                                                           
        x, y, z = gal2ecl_cart(xg, yg, zg)  # Converting galactic Cartesian coordinates to ecliptic (au)
        vx, vy, vz = gal2ecl_cart(vxg, vyg, vzg)  # Converting galactic Cartesian coordinates to ecliptic (au)
           
        omega_e[i], Omega_e[i], inc_e[i] = cart2orb(x, y, z, vx, vy, vz, GM_sun.value)[:3]
        
    for i in range(len(q)):
        
        # we calculate moid for every ISO
        moid_t, m_moid_t = moid(1., omega_e[i], Omega_e[i], inc_e[i], e[i], q[i], r_in, r_out, step_final = step_final_coarse, step_final_big_q = step_final_coarse * 10)
        
        if moid_t < 2 * r_minor:
            moid_t, m_moid_t = moid(1., omega_e[i], Omega_e[i], inc_e[i], e[i], q[i], r_in, r_out, step_final = step_final_dense, step_final_big_q = step_final_dense * 10)
        
        if moid_t < 2 * r_minor / inflation_factor:
            moid_t, m_moid_t = moid(1., omega_e[i], Omega_e[i], inc_e[i], e[i], q[i], r_in, r_out, step_final = step_final_dense * 10, step_final_big_q = step_final_dense * 100)
        
            
        MOID.append(moid_t)
        M_MOID.append(m_moid_t)
        
        
    
    MOID = np.array(MOID)
    M_MOID = np.array(M_MOID)
    selection_moid = MOID < r_minor # we select only those whose MOID is smaller than the  inflated radius of the Earth
    
    q = q[selection_moid]
    e = e[selection_moid]
    E = E[selection_moid]
    inc_e = inc_e[selection_moid]
    inc = inc[selection_moid]
    Omega_e = Omega_e[selection_moid]
    omega_e = omega_e[selection_moid]
    MOID = MOID[selection_moid]
    M_MOID = M_MOID[selection_moid]
    M = M[selection_moid]
    mm = mm[selection_moid]

    a = q / (1 - e) * au.value
    t_impact = (M_MOID - M) / mm
    
    # we select only those which impact within the simulated time (78 days)
    selection_t = np.logical_and(t_impact > 0, t_impact < total_time)
    
    q = q[selection_t]
    e = e[selection_t]
    E = E[selection_t]
    inc_e = inc_e[selection_t]
    inc = inc[selection_t]
    Omega_e = Omega_e[selection_t]
    omega_e = omega_e[selection_t]
    MOID = MOID[selection_t]
    M_MOID = M_MOID[selection_t]
    M = M[selection_t]
    mm = mm[selection_t]
    t_impact = t_impact[selection_t]
    a = a[selection_t]
 
    # selection for R_eff
    for i in range(len(q)):
        Et = kepler(e[i], M_MOID[i])
        xt, yt, zt, vxt, vyt, vzt = orb2cart(omega_e[i], Omega_e[i], inc_e[i], e[i], a[i], Et, GM_sun.value)
        
        ISO_lon = np.arctan2(yt, xt)
    
        v_earth_x,  v_earth_y = -v_earth * np.sin(ISO_lon), v_earth * np.cos(ISO_lon)
    
        # ISO geocentric velocity components
        v_gc_x = vxt - v_earth_x
        v_gc_y = vyt - v_earth_y
        v_gc_z = vzt  # because Earth moves in ecliptic plane and doesn't have z-component of the velocity vector
    
        # ISO geocentric speed
        v_gc = np.sqrt(v_gc_x**2 + v_gc_y**2 + v_gc_z**2) # this is speed relative to the Earth

        r_minor_eff = r_minor * np.sqrt(1+v_esc_earth**2 / v_gc**2) # effective radius of the earth
          
        if (np.sqrt((xt/au.value)**2 + (yt/au.value)**2) - r_major)**2 + (zt/au.value)**2 < r_minor_eff**2: # inside torus!
            v_hc = np.sqrt(vxt**2 + vyt**2 + vzt**2)
            radiant_long, radiant_lat = spherical_coor(-v_gc_x, -v_gc_y, -v_gc_z)
            radiant_RA, radiant_DEC = ecl2eq_spherical(radiant_long, radiant_lat)
            sun_long_imp = (np.array(ISO_lon) + np.pi) % (2 * np.pi) # solar longitude at imapct
            sun_long_imp = (sun_long_imp + np.pi) % (2 * np.pi) - np.pi # wrapping to -pi, pi
            

            q_torus.append(q[i])
            e_torus.append(e[i])
            E_initial_torus.append(E[i])
            inc_torus.append(inc_e[i])
            Omega_torus.append(Omega_e[i])
            omega_torus.append(omega_e[i])
            RA_torus.append(radiant_RA)
            DEC_torus.append(radiant_DEC)
            sun_lon_torus.append(sun_long_imp)
            v_gc_torus.append(v_gc)
            v_hc_torus.append(v_hc)
            v_hc_inf_torus.append(v_hc_inf[i])
            inc_g_torus.append(inc[i])
            r_eff_torus.append(r_minor_eff)
            M_moid_torus.append(M_MOID[i])
            moid_torus.append(MOID[i])

    # writin data to output files
    np.savetxt(
        output_torus_file,
        np.column_stack((q_torus, e_torus, E_initial_torus, M_moid_torus, moid_torus,
                         inc_torus, inc_g_torus, Omega_torus, omega_torus,
                         v_hc_inf_torus, v_hc_torus,
                         v_gc_torus, r_eff_torus, RA_torus, DEC_torus,
                         sun_lon_torus)),
        header=header,
        fmt="%.9f",          # format brojeva (možeš menjati)
        delimiter=", "       # razdvajač između kolona
    )
     
    number_of_impacts = len(q_torus) # total number of impacts
    
    population_number += 1
    np.savetxt(progress_file, [f'population number = {population_number}, total number of object = {total_objects}, torus passers = {len(inc_torus)}'], fmt = '%s')
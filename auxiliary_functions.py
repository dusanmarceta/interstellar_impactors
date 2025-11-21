import numpy as np

def array_making(x2 = 172680, x3 = 172980, x4 = 345480,  n1 = 20, n2 = 600):
    

#    x2 = 2*86400-120
#    x3 = x2 + 300
#    x4 = x3 + 2*86400
#    
#    n1 = 20
#    n2 = 60
    
    #niz1 = np.logspace(np.log10(x2), np.log10(x1), n1)
    niz2 = np.linspace(x2, x3, n2)
    niz3 = np.logspace(np.log10(x3), np.log10(x4), n1)
    niz1 = niz3[-1] - niz3[::-1]
    
    niz = np.concatenate([niz1, niz2, niz3])
    
    return niz


def kepler(e, M, accuracy = 1e-6):
    # =============================================================================
    # solves Kepler equation using Newton-Raphson method
    # for elliptic and hyperbolic orbit depanding on eccentricity

    # Input:
    # e - eccentricity
    # M - mean anomaly (radians)
    # accuracy - accuracy for Newton-Raphson method (for example 1e-6)
    #
    # Output:
    # E [radians] - eccentric (hyperbolic) anomaly
    # =============================================================================
    if e > 1:  # hyperbolic orbit (GOODIN & ODELL, 1988)

        L = M / e
        g = 1 / e

        q = 2 * (1 - g)
        r = 3 * L
        s = (np.sqrt(r ** 2 + q ** 3) + r) ** (1 / 3)

        H00 = 2 * r / (s ** 2 + q + (q / s) ** 2)

        #    if np.abs(np.abs(M)-1)<0.01:
        if np.mod(np.abs(M), 0.5) < 0.01 or np.mod(np.abs(M), 0.5) > 0.49:  # numerical problem about this value
            E = (M * np.arcsinh(L) + H00) / (M + 1 + 0.03)  # initial estimate
        else:
            E = (M * np.arcsinh(L) + H00) / (M + 1)  # initial estimate

        delta = 1.0
        while abs(delta) > accuracy:
            f = M - e * np.sinh(E) + E
            f1 = -e * np.cosh(E) + 1
            delta = f / f1
            E = E - delta

    elif e < 1:  # elliptic orbit
        delta = 1.0
        E = M

        while abs(delta) > accuracy:
            f = E - e * np.sin(E) - M
            f1 = 1 - e * np.cos(E)
            delta = f / f1
            E = E - delta

    return E

def ecc2true(E, e):
    # =============================================================================
    # converts eccentric (or hyperbolic) anomaly to true anomaly
    # Input:
    # E [radians] - eccentric (or hyperbolic anomaly)
    # Output:
    # True anomaly [radians]
    # =============================================================================
    if e > 1:
        return 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(E / 2))
    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(E), np.cos(E) - e)
    
    
def true2ecc(f, e):
    # =============================================================================
    # converts true anomaly to eccentric (or hyperbolic) anomaly
    # Input:
    # f [radians] - true anomaly
    # Output:
    # eccentric (or hyperbolic anomaly) [radians]
    # =============================================================================
    if e > 1:
        return 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(f / 2))

    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(f), e + np.cos(f))

# ecliptic to equatorial
def ecl2eq_spherical(long, lat):
    # =============================================================================
    # converts ecliptic to equatorial coordinates
    # Input:
    # long,lat [rad] - ecliptic longitude, ecliptic latitude
    # Output:
    # alpha, delta [rad] - right ascension, declination
    # =============================================================================

    eps = 0.409093  # ecliptic obliquity

    delta = np.arcsin(np.sin(eps) * np.sin(long) * np.cos(lat) + np.cos(eps) * np.sin(lat))
    sinus = (np.cos(eps) * np.sin(long) * np.cos(lat) - np.sin(eps) * np.sin(lat)) / np.cos(delta)
    kosinus = np.cos(long) * np.cos(lat) / np.cos(delta)
    alpha = np.arctan2(sinus, kosinus)

    return alpha, delta

def orb2cart(o, O, inc, e, a, E, G):
    # =============================================================================
    # calculates cartesian coordinates (in meters) from orbital elements for
    # elliptic and hyperbolic orbit depanding on eccentricity
    #
    # input:
    # o - argument of perihelion (rad)
    # O - longitude of ascending node (rad)
    # inc - inclination (rad)
    # a - semimajor axis (meters) (periapsis distance if parabolic orbit)
    # e - eccentricity
    # E - eccentric anomaly (radians) (true anomaly if parabolic orbit)
    # Output:
    # x,y,z [meters] - cartesian coordinates
    # =============================================================================
    
    if e > 1:  # hyperbolic orbit
        r = a * (1 - e * np.cosh(E))  # heliocentric distance
        f = np.mod(ecc2true(E, e), 2 * np.pi)  # true anomaly
        xt = -np.sqrt(-a * G) / r * np.sinh(E)  # minus sign to chose appropriate branch of hyperbola
        yt = np.sqrt(-a * G * (e ** 2 - 1)) / r * np.cosh(E)

    elif e < 1:  # elliptic orbit
        r = a * (1 - e * np.cos(E))  # helicentric distance
        f = np.mod(ecc2true(E, e), 2 * np.pi)  # true anomaly
        xt = -np.sqrt(G * a) / r * np.sin(E)
        yt = np.sqrt(G * a * (1 - e ** 2)) / r * np.cos(E)



    # cartesian coordinates
    x = r * (np.cos(O) * np.cos(o + f) - np.sin(O) * np.cos(inc) * np.sin(o + f))
    y = r * (np.sin(O) * np.cos(o + f) + np.cos(O) * np.cos(inc) * np.sin(o + f))
    z = r * (np.sin(inc) * np.sin(o + f))

    # cartesian components (ecliptical coordinate system)
    vx = xt * (np.cos(o) * np.cos(O) - np.sin(o) * np.cos(inc) * np.sin(O)) \
         - yt * (np.sin(o) * np.cos(O) + np.cos(o) * np.cos(inc) * np.sin(O))

    vy = xt * (np.cos(o) * np.sin(O) + np.sin(o) * np.cos(inc) * np.cos(O)) \
         - yt * (np.sin(o) * np.sin(O) - np.cos(o) * np.cos(inc) * np.cos(O))

    vz = xt * np.sin(o) * np.sin(inc) + yt * np.cos(o) * np.sin(inc)


    return x, y, z, vx, vy, vz


def cart2orb(x, y, z, vx, vy, vz, G):
    # =============================================================================
    # calculates keplerian orbital elements from cartesian
    # position and velocity
    # Input:
    # x, y, z - [m] cartesian coordinates
    # vx, vy, vz - [m/s] cartesian velocities
    # G - central body gravity parameter (m**3/s**2)
    #
    # Output
    # o - argument of perihelion [rad]
    # O - longitude of ascending node [rad]
    # inc - inclination [rad]
    # a - semimajor axis [meters]
    # e - eccentricity
    # M - mean anomaly [radians]
    # =============================================================================
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])

    # momenat kolicine kretanja
    h = np.cross(r, v)

    # vektor ekscentriciteta
    e = np.cross(v, h) / G - r / np.linalg.norm(r)

    # vektor u pravcu uzlaznog cvora
    n = np.cross(np.array([0, 0, 1]), h)

    # prava anomalija
    nu = np.arccos(np.linalg.linalg.dot(e, r) / np.linalg.norm(e) / np.linalg.norm(r))

    if np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e) < 1.:
        nu = 2 * np.pi - nu
    elif np.linalg.linalg.dot(r, v) < 0 and np.linalg.norm(e) > 1.:
        nu = -nu

    # nagib
    ink = np.arccos(h[2] / np.linalg.norm(h))

    # longituda cvora
    O = np.arccos(n[0] / np.linalg.norm(n))
    if n[1] < 0:
        O = 2 * np.pi - O

    # argument pericentra
    o = np.arccos(np.linalg.linalg.dot(n, e) / np.linalg.norm(n) / np.linalg.norm(e))
    if e[2] < 0:
        o = 2 * np.pi - o

    # ekscentricitet
    e = np.linalg.norm(e)

    # ekscentricna anomalija
    E = true2ecc(nu, e)

    # srednja anomalija
    #    if e < 1:
    #        M = E - e * np.sin(E)
    #    else:
    #        M = e * np.sinh(E) - E

    # poluosa
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / G)

    return o, O, ink, e, a, E, nu


def spherical_coor(x, y, z):
    # =============================================================================
    # calculates spherical coordinates
    # Input
    # x,y,z - cartesian coordinates
    # Output:
    # long, lat [degrees] - spherical coordinates
    # =============================================================================
    # geocentric spherical coordinates
    long = np.mod(np.arctan2(y, x), 2 * np.pi)
    lat = np.arctan(z / np.sqrt(x ** 2 + y ** 2))

    # converts to degrees
    return long, lat



# galactic to ecliptic
def gal2ecl_spherical(l, b):
    # =============================================================================
    # converts ecliptic to galactic coordinates
    # Input:
    # l, b (radians) - galactic longitude, galactic latitude
    # Output:
    # long,lat (radians) - ecliptic longitude, ecliptic latitude
    # =============================================================================
    lg = 3.14177
    bg = 0.52011
    bk = 1.68302

    lat = np.arcsin(np.sin(bg) * np.sin(b) + np.cos(bg) * np.cos(b) * np.cos(bk - l))
    sinus = np.cos(b) * np.sin(bk - l) / np.cos(lat)
    kosinus = (np.cos(bg) * np.sin(b) - np.sin(bg) * np.cos(b) * np.cos(bk - l)) / np.cos(lat)
    long = lg + np.arctan2(sinus, kosinus)

    return long, lat


def gal2ecl_cart(x, y, z):
    # =============================================================================
    # converts  galactic coordinates
    # Input:
    # x,y,z - galactic
    # Output:
    # x,y,z - ecliptic
    # =============================================================================

    r = (x ** 2 + y ** 2 + z ** 2) ** (1 / 2)
    l = np.arctan2(y, x)
    b = np.arctan(z / (x ** 2 + y ** 2) ** (1 / 2))
    
    long, lat = gal2ecl_spherical(l, b)
    
    return r * np.cos(long) * np.cos(lat), r * np.sin(long) * np.cos(lat), r * np.sin(lat)

def orbit_plot(o, O, i, e, a, rmax, plane, nodes, apse, color, axis_max, ax):
    # =============================================================================
    # plots orbite in 3D given the orbital elements
    # plane=1 (plots orbital plane)
    # nodes=1 (plots line of nodes)
    # apse=1 (plots line of apsides)
    # rmax - maximum distance from the central body
    # color - color of the orbit, e.g. 'r' for red
    #
    # ax - projection which should be defined as
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ============================================================================

    if e > 1:
        teta0 = np.arccos(a * (1 - e ** 2) / e / rmax - 1 / e)
        teta1 = np.linspace(-teta0, teta0, 100)

    else:
        teta1 = np.linspace(0, 2 * np.pi, 100)

    r = a * (1 - e ** 2) / (1 + e * np.cos(teta1))

    x = r * (np.cos(O) * np.cos(o + teta1) - np.sin(O) * np.cos(i) * np.sin(o + teta1))
    y = r * (np.sin(O) * np.cos(o + teta1) + np.cos(O) * np.cos(i) * np.sin(o + teta1))
    z = r * (np.sin(i) * np.sin(o + teta1))

    # plotting the orbit
    ax.plot(x, y, z, color, linewidth=2)

    # rotation matrix
    RO = np.transpose(np.array([[np.cos(O), np.sin(O), 0], [-np.sin(O), np.cos(O), 0], [0, 0, 1]]))  # rotacija za O
    Ri = np.transpose(np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]]))  # rotacija za i
    Ro = np.transpose(np.array([[np.cos(o), np.sin(o), 0], [-np.sin(o), np.cos(o), 0], [0, 0, 1]]))  # rotacija za o
    R = RO.dot(Ri).dot(Ro)

    # plotting the orbital plane
    if plane == 1:

        duzina = min(np.abs(rmax / np.cos(O)), np.abs(rmax / np.sin(O)))
        duzina = max(np.abs(duzina * np.cos(o)), np.abs(duzina * np.sin(o)))
        xx, yy = np.meshgrid([-duzina, duzina], [-duzina, duzina])

        xp = np.zeros_like(xx)
        yp = np.zeros_like(xx)
        zp = np.zeros_like(xx)

        for i1 in range(0, len(xx)):
            for i2 in range(0, len(xx)):
                [xp[i1, i2], yp[i1, i2], zp[i1, i2]] = np.dot(R, np.transpose(np.array([xx[i1, i2], yy[i1, i2], 0])))

        ax.plot_surface(xp, yp, zp, color=color, linewidth=0, alpha=0.2)

    # plotting the line of apsides
    if apse == 1:
        [xa1, ya1, za1] = np.dot(R, np.transpose(np.array([rmax, 0, 0])))
        [xa2, ya2, za2] = np.dot(R, np.transpose(np.array([-rmax, 0, 0])))

        ax.plot([xa1, xa2], [ya1, ya2], [za1, za2], '--k')

    # plotting the line of nodes
    if nodes == 1:
        duzina = min(np.abs(rmax / np.cos(O)), np.abs(rmax / np.sin(O)))
        ax.plot([duzina * np.cos(O), -duzina * np.cos(O)], [duzina * np.sin(O), -duzina * np.sin(O)], [0, 0], 'k')
        
    axis_limits = np.array([-axis_max, axis_max])
    for xs in axis_limits:
        for ys in axis_limits:
            for zs in 0.6 * axis_limits:
                ax.scatter(xs, ys, zs, alpha=0)  # providne tačke
        
def moid(r1, o2, O2, i2, e2, q2, r_min, r_max, step_initial = 10, step_final = 100, step_final_big_q = 1000):
    # =============================================================================
    # Calculates minimum orbit intersection distance between elliptic and hyperbolic orbits
    # Input:
    # o1,O1,i1,e1,a1 [rad]- ellipse
    # o2,O2,i2,e2,a2 [rad]- hyperbola
    # limit - limiting step for division of orbit in order to find approximate solution of moid (rad)
    # r_max - maximum helicentric distance for ISO
    # Output:
    # moid [same as a1 and a2] - minimum orbit intersection dostance
    # E1_konacno, E2_konacno [radians] - eccentric (hyperbolic) corresponding to moid
    # =============================================================================
    
    a2 = q2 / (1 - e2)
    try:
        # pretvaranje u radijane
        
        
        if q2 < r_min:
            '''
            perihelion smaller than the inner torus radius
            '''
            E_out = np.arccosh(1 / e2 - r_max / e2 / a2)  # granicna anomalija kada je r=r_max
            E_in = np.arccosh(1 / e2 - r_min / e2 / a2)  # granicna anomalija kada je r=r_min
            
            korak = (E_out - E_in) / step_initial
            
            E1_min = 0
            E1_max = 2 * np.pi - korak
            
            limit = (E_out - E_in) / step_final
            
            '''
            Inbound
            '''
            
            E2_min = -E_out
            E2_max = -E_in
        
            while korak > limit:
        
                E1 = np.linspace(E1_min, E1_max, int((E1_max - E1_min) / korak) + 1)  # Planet
    
        
                E2 = np.linspace(E2_min, E2_max, int((E2_max - E2_min) / korak) + 1)  # ISO
                T2 = ecc2true(E2, e2)
    
                r2 = a2 * (1 - e2 * np.cosh(E2))
        
                x1 = r1 * np.cos(E1)
                y1 = r1 * np.sin(E1)
    
        
                x2 = r2 * (np.cos(O2) * np.cos(o2 + T2) - np.sin(O2) * np.cos(i2) * np.sin(o2 + T2))
                y2 = r2 * (np.sin(O2) * np.cos(o2 + T2) + np.cos(O2) * np.cos(i2) * np.sin(o2 + T2))
                z2 = r2 * np.sin(o2 + T2) * np.sin(i2)
    
                r = np.sqrt(
                    (x1[:, None] - x2[None, :])**2 +
                    (y1[:, None] - y2[None, :])**2 +
                    (z2[None, :])**2
                )
    
                ind1, ind2 = np.unravel_index(np.argmin(r), r.shape)
        
                # priblizna resenja
                E1_konacno = E1[ind1]
                E2_konacno = E2[ind2]
        
                E1_min = E1_konacno - korak
                E1_max = E1_konacno + korak
        
                E2_min = E2_konacno - korak
                E2_max = E2_konacno + korak
        
                korak = korak / 2
                
            MOID_inbound = np.min(r)
            E_inbound = E2_konacno
            
            
            '''
            Outbound
            '''
            korak = (E_out - E_in) / step_initial
            
            E1_min = 0
            E1_max = 2 * np.pi - korak
            E2_min = E_in
            E2_max = E_out
        
            while korak > limit:
        
                E1 = np.linspace(E1_min, E1_max, int((E1_max - E1_min) / korak) + 1)  # Planet
    
        
                E2 = np.linspace(E2_min, E2_max, int((E2_max - E2_min) / korak) + 1)  # ISO
                T2 = ecc2true(E2, e2)
    
                r2 = a2 * (1 - e2 * np.cosh(E2))
        
                x1 = r1 * np.cos(E1)
                y1 = r1 * np.sin(E1)
    
        
                x2 = r2 * (np.cos(O2) * np.cos(o2 + T2) - np.sin(O2) * np.cos(i2) * np.sin(o2 + T2))
                y2 = r2 * (np.sin(O2) * np.cos(o2 + T2) + np.cos(O2) * np.cos(i2) * np.sin(o2 + T2))
                z2 = r2 * np.sin(o2 + T2) * np.sin(i2)
    
                r = np.sqrt(
                    (x1[:, None] - x2[None, :])**2 +
                    (y1[:, None] - y2[None, :])**2 +
                    (z2[None, :])**2
                )
    
                ind1, ind2 = np.unravel_index(np.argmin(r), r.shape)
        
                # priblizna resenja
                E1_konacno = E1[ind1]
                E2_konacno = E2[ind2]
        
                E1_min = E1_konacno - korak
                E1_max = E1_konacno + korak
        
                E2_min = E2_konacno - korak
                E2_max = E2_konacno + korak
        
                korak = korak / 2
                
            MOID_outbound = np.min(r)
            E_outbound = E2_konacno
            
            if MOID_inbound < MOID_outbound:
                MOID = MOID_inbound
                E_MOID = E_inbound

            else:
                MOID = MOID_outbound
                E_MOID = E_outbound

            MOID = np.min([MOID_inbound, MOID_outbound])

        else:
            '''
            perihelion larger than the inner torus radius
            '''
            E_in_out = np.arccosh(1 / e2 - r_max / e2 / a2)  # granicna anomalija kada je r=r_max


            E2_min = -E_in_out
            E2_max = E_in_out
            
            korak = (E2_max - E2_min) / step_initial
            
            E1_min = 0
            E1_max = 2 * np.pi - korak
            
            limit = (E2_max - E2_min) / step_final_big_q
        
            while korak > limit:
        
                E1 = np.linspace(E1_min, E1_max, int((E1_max - E1_min) / korak) + 1)  # Planet
    
        
                E2 = np.linspace(E2_min, E2_max, int((E2_max - E2_min) / korak) + 1)  # ISO
                T2 = np.mod(ecc2true(E2, e2), 2 * np.pi)
    
                r2 = a2 * (1 - e2 * np.cosh(E2))
        
                x1 = r1 * np.cos(E1)
                y1 = r1 * np.sin(E1)
    
        
                x2 = r2 * (np.cos(O2) * np.cos(o2 + T2) - np.sin(O2) * np.cos(i2) * np.sin(o2 + T2))
                y2 = r2 * (np.sin(O2) * np.cos(o2 + T2) + np.cos(O2) * np.cos(i2) * np.sin(o2 + T2))
                z2 = r2 * np.sin(o2 + T2) * np.sin(i2)
    
                r = np.sqrt(
                    (x1[:, None] - x2[None, :])**2 +
                    (y1[:, None] - y2[None, :])**2 +
                    (z2[None, :])**2
                )
    
                ind1, ind2 = np.unravel_index(np.argmin(r), r.shape)
        
                # priblizna resenja
                E1_konacno = E1[ind1]
                E2_konacno = E2[ind2]
        
                E1_min = E1_konacno - korak
                E1_max = E1_konacno + korak
        
                E2_min = E2_konacno - korak
                E2_max = E2_konacno + korak
        
                korak = korak / 2
                
            MOID = np.min(r)
            E_MOID = E2_konacno
            
        M_MOID = e2 * np.sinh(E_MOID) - E_MOID

    except:
        
        MOID=0
        E1_konacno=0
        E2_konacno=0
        

    return MOID, M_MOID

def torus_plot(r_major, r_minor, color, axis_max, ax):
    # Mreža parametara
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Parametarske jednačine
    x = (r_major + r_minor*np.cos(v)) * np.cos(u)
    y = (r_major + r_minor*np.cos(v)) * np.sin(u)
    z = r_minor * np.sin(v)   # <-- ispravljeno, ranije si imao r_major

    # Površina torusa
    ax.plot_surface(x, y, z, color=color, alpha=0.5, edgecolor='none')

    # Dodaj "nevidljive" tačke u ćoškove kocke
    axis_limits = np.array([-axis_max, axis_max])
    for xs in axis_limits:
        for ys in axis_limits:
            for zs in 0.6 * axis_limits:
                ax.scatter(xs, ys, zs, alpha=0)  # providne tačke
    



# Build rotation matrix to align plane with XY plane
# z' axis = n
def rotation_matrix_from_vectors(vec1, vec2):
    """Rotate vec1 to vec2"""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

   
def torus_intersection(r_major, r_minor, inc, Omega, color, ax):
    # Normal vector of plane
    n = np.array([np.sin(inc)*np.sin(Omega),
              -np.sin(inc)*np.cos(Omega),
              np.cos(inc)])
    
    Rmat = rotation_matrix_from_vectors(n, np.array([0,0,1]))

    # Sample points in XY plane of rotated system
    u = np.linspace(-r_major-r_minor, r_major+r_minor, 500)
    v = np.linspace(-r_major-r_minor, r_major+r_minor, 500)
    U,V = np.meshgrid(u,v)
    Zp = np.zeros_like(U)
    
    # Transform back to 3D
    pts = np.vstack([U.flatten(), V.flatten(), Zp.flatten()])
    pts_rot = Rmat.T @ pts
    X,Y,Z = pts_rot.reshape(3,*U.shape)
    
    # Torus equation mask
    mask = np.abs((np.sqrt(X**2+Y**2)-r_major)**2 + Z**2 - r_minor**2) < 0.001

    # Torus surface
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 50)
    U,V = np.meshgrid(u,v)  
    
    # Intersection curve
    ax.scatter(X[mask], Y[mask], Z[mask], s=0.2, c=color)

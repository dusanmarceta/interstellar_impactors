import numpy as np

'''
This code only merges the outputs produced by the parallel processes. 
It also selects only the impacts within the chosen inflation factor. 
The simulation captured all impacts on the Earth inflated by a factor of 1000, 
but here we use only those corresponding to an inflation factor of 1 
(i.e., the real Earth).
'''

first_core = 0
number_of_cores = 80

q            = np.array([])
ecc          = np.array([])
H_init       = np.array([])
M_impact     = np.array([])
moid         = np.array([])
inc_e        = np.array([])
inc_g        = np.array([])
Omega        = np.array([])
omega        = np.array([])
v_inf        = np.array([])
v_hc         = np.array([])
v_gc         = np.array([])
R_eff        = np.array([])
RA_radiant   = np.array([])
DEC_radiant  = np.array([])
sun_longitude = np.array([])

        
number = 0
for run in range(first_core, first_core + number_of_cores):

    try:
        torus_file = 'results/interstellar_impactors_' + str(run) + '.txt' # output file with the results
        
        (q_t, ecc_t, H_init_t, M_impact_t, moid_t, inc_e_t, inc_g_t, 
         Omega_t, omega_t, v_inf_t, v_hc_t, v_gc_t, R_eff_t, 
         RA_radiant_t, DEC_radiant_t, sun_longitude_t) = np.loadtxt(
            torus_file, skiprows=1, delimiter=',', unpack=True
            
        )
        selection = moid_t < 1 * R_eff_t / 1000 # since the earth was inflated 1000 times here we select only those within 1 effective radius of the Earth
        
        q            = np.append(q, q_t[selection])
        ecc          = np.append(ecc, ecc_t[selection])
        H_init       = np.append(H_init, H_init_t[selection])
        M_impact     = np.append(M_impact, M_impact_t[selection])
        moid         = np.append(moid, moid_t[selection])
        inc_e        = np.append(inc_e, inc_e_t[selection])
        inc_g        = np.append(inc_g, inc_g_t[selection])
        Omega        = np.append(Omega, Omega_t[selection])
        omega        = np.append(omega, omega_t[selection])
        v_inf        = np.append(v_inf, v_inf_t[selection])
        v_hc         = np.append(v_hc, v_hc_t[selection])
        v_gc         = np.append(v_gc, v_gc_t[selection])
        R_eff        = np.append(R_eff, R_eff_t[selection])
        RA_radiant   = np.append(RA_radiant, RA_radiant_t[selection])
        DEC_radiant  = np.append(DEC_radiant, DEC_radiant_t[selection])
        sun_longitude = np.append(sun_longitude, sun_longitude_t[selection])

    except:
        pass

    # napravi matricu gde je svaka kolona jedan od tvojih nizova
    data_out = np.column_stack((
        q, ecc, H_init, M_impact, moid, inc_e, inc_g,
        Omega, omega, v_inf, v_hc, v_gc, R_eff,
        RA_radiant, DEC_radiant, sun_longitude
    ))
    
    # nazivi kolona (isti redosled kao kod čitanja)
    header = "q,ecc,H_init,M_impact,moid,inc_e,inc_g,Omega,omega,v_inf,v_hc,v_gc,R_eff,RA_radiant,DEC_radiant,sun_longitude"
    
    # upis u CSV
    np.savetxt("results/impactors_R=1.csv", data_out, delimiter=",", header=header, comments='') # merged output file
    

import numpy as np
import math
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def Berryman(N,Kp,asp,K,mu):
    """
    Input parameters:
    -----------------
    N        np.int, number of the inclusion's type
    Kp       np.array(N), contains volume fraction of the inclusions
    asp      np.array(N), contains aspect ratios of the inclusions
    K        np.array(N), contains bulk modulus of the inclusions
    mu       np.array(N), contains shear modulus of the inclusions
   
    Returns:
    --------
    k_sc     np.float, effective bulk modulus
    mu_sc    np.float, effective shear modulus
    """

    if (np.sum(Kp)>1) or (np.sum(Kp)<0.99):
        print('The sum of all volume fraction is not equal to 1')
        return None


    P = np.zeros(N)
    Q = np.zeros(N)
    
    k_v = np.sum(Kp*K)  
    mu_v = np.sum(Kp*mu)  

    
    k_sc = k_v                   
    mu_sc = mu_v
    
       
    k_new = 0                                   
    mu_new = 0
    tol = 0.01 # define tolerance
    deltak = abs(k_sc-k_new)
    deltamu = abs(mu_sc-mu_new)
    step = 0
    while (deltak > tol) and (deltamu > tol):
        for i in range (N):
            th = (asp[i]/(1.-asp[i]**2.)**(1.5))*(math.acos(asp[i])-asp[i]*(1.-asp[i]**2.)**(1./2.))
            f=(asp[i]**2./(1.-asp[i]**2.))*(3.*th - 2.)
            
            nu_sc = (3.*k_sc-2.*mu_sc)/2./(3.*k_sc+mu_sc)
            
            R = (1.-2.*nu_sc)/(2.*(1.-nu_sc))
            A = mu[i]/mu_sc-1.
            B = (K[i]/k_sc - mu[i]/mu_sc)/3.
            
            F1 = 1.0+A*( 1.5*(f+th) - R*(1.5*f+2.5*th-4.0/3.0) )
            F2 = 1.0+(A*(1.0+((3.0/2.0)*(f+th))-((1.0/2.0)*R*((3.0*f)+(5.0*th)))))+(B*(3.0-(4.0*R))) \
            +((1.0/2.0)*A*(A+(3.0*B))*(3.0-(4.0*R))*(f+th-(R*(f-th+(2.0*(th**2.0))))))
            T=3.0*F1/F2
            P[i]=T/3.0
            
            F3=1.0+A*(1.0-(f+1.5*th)+R*(f+th))
            F4=1.0+0.25*A*(f+3.0*th-R*(f-th))
            F5=A*(-f+R*(f+th-4.0/3.0)) + B*th*(3.0-4.0*R)
            F6=1.0+A*(1+f-R*(f+th)) + B*(1.0-th)*(3.0-4.0*R)
            F7=2.0+0.25*A*(3.0*f+9.0*th-R*(3.0*f+5*th)) + B*th*(3.0-4.0*R)
            F8=A*(1.0-2.0*R +0.5*f*(R-1.0) + th*0.5*(5.0*R-3.0)) +B*(1.0-th)*(3.0-4.0*R)
            F9=A*((R-1.0)*f-R*th)+B*th*(3.0-4.0*R)
            Q[i]=(2.0/F3+1.0/F4+(F4*F5+F6*F7-F8*F9)/(F2*F4))/5.0
    
       
        A1 = np.sum(Kp*K*P)
        B1 = np.sum(Kp*mu*Q)            
        A2 = np.sum(Kp*P)
        B2 = np.sum(Kp*Q)
            

        k_new = A1/A2
        mu_new = B1/B2
        deltak = abs(k_sc-k_new)
        deltamu = abs(mu_sc-mu_new)
        k_sc = k_new
        mu_sc = mu_new
        step = step+1
    return k_sc, mu_sc

def Berryman_F(N,Kp,asp,K,mu,f):
    """ 
    Input parameters:
    -----------------
    N        np.int, number of the inclusions type
    Kp       np.array(nparam), contains volume fraction of the inclusions, the last element should be volume fraction of the matrix 
    asp      np.array(nparam), contains aspect ratios of the inclusions, the last element should be aspect ratio of the matrix 
    K        np.array(nparam), contains bulk moduli of the inclusions, the last element should be bulk modulus of the matrix 
    mu       np.array(nparam), contains shear moduli of the inclusions, the last element should be shear modulus of the matrix 
    f        np.float, f-parameter reflecting connectivity of the fluid inclusion

    Returns:
    --------
    k_sc     np.float, effective bulk modulus
    mu_sc    np.float, effective shear modulus
    """
    
    if (np.sum(Kp)>1) or (np.sum(Kp)<0.99):
        print('The sum of all volume fraction is not equal to 1')
        return None

   
    P = np.zeros(N)
    Q = np.zeros(N)
    
    Km = K[N-1]
    mum = mu[N-1]
    
    Kfl = K[0]
    mufl = mu[0]
    
    
    k_sc = (1-f)*Km + f*Kfl                  
    mu_sc= (1-f)*mum + f*mufl
    #print(k_sc, mu_sc)
    
    for i in range (N):
        S_sc = (mu_sc*(9*k_sc+8*mu_sc))/(6*(k_sc+2*mu_sc))

        theta = (asp[i]/((1-asp[i]**2)**(3/2)))*(np.arccos(asp[i])-asp[i]*(1-asp[i]**2)**(1/2))
        f=(asp[i]**2/(1-asp[i]**2))*(3*theta-2)

        nu_sc = (3*k_sc-2*mu_sc)/(2*(3*k_sc+mu_sc))
        R = (1-2*nu_sc)/(2*(1-nu_sc))
        A = mu[i]/mu_sc-1
        B = (K[i]/k_sc - mu[i]/mu_sc)/3
        f1 = 1+A*(3*(f+theta)/2-R*(3*f/2+5*theta/2-4/3))
        f2 = 1+A*(1+3*(f+theta)/2 - (R/2)*(3*f+5*theta))+B*(3-4*R)+(A/2)*(A+3*B)*(3-4*R)*(f+theta-R*(f-theta+2*theta**2))
        f3 = 1+A*(1-(f+3*theta/2)+R*(f+theta))
        f4 = 1+(A/4)*(f+3*theta - R*(f-theta))
        f5 = A*(-f+R*(f+theta-4./3.))+B*theta*(3-4*R)
        f6 = 1+A*(1+f-R*(f+theta))+B*(1-theta)*(3-4*R)
        f7 = 2+(A/4)*(3*f+9*theta-R*(3*f+5*theta))+B*theta*(3-4*R)
        f8 = A*(1-2*R+(f/2)*(R-1)+(theta/2)*(5*R-3))+B*(1-theta)*(3-4*R)
        f9 = A*((R-1)*f-R*theta)+B*theta*(3-4*R)
        P[i] = (3*f1/f2)/3
        Q[i] = (2/f3+1/f4+(f4*f5+f6*f7-f8*f9)/(f2*f4))/5
        #print(P)
       
    A1 = np.sum(Kp*K*P)
    B1 = np.sum(Kp*mu*Q)            
    A2 = np.sum(Kp*P)
    B2 = np.sum(Kp*Q)
      

    k_sc = A1/A2
    mu_sc = B1/B2
        
    return k_sc, mu_sc

def DEM(Kp,asp,K_m,mu_m,K_i,mu_i):
    """
    Input parameters:
    -----------------
    Kp        np.float, volume fraction of the voids (pores)
    asp       np.float, aspect ratios of the pores
    K_m       np.float, bulk modulus of the matrix
    mu_m      np.float, shear modulus of the matrix
    K_i       np.float, bulk modulus of the inclusions (fluid)
    mu_i      np.float, shear modulus of the inclusions (fluid)

    Returns:
    --------
    K_eff     np.float, effective bulk modulus
    mu_eff    np.float, effective shear modulus
    """
    
    def my_fun(t,y,*args):
        k_sc, mu_sc = y
        th = (asp/(1.-asp**2.)**(1.5))*(math.acos(asp)-asp*(1.-asp**2.)**(1./2.))
        f=(asp**2./(1.-asp**2.))*(3.*th - 2.)
         
        nu_sc = (3.*k_sc-2.*mu_sc)/2./(3.*k_sc+mu_sc)
        
        R = (1.-2.*nu_sc)/(2.*(1.-nu_sc))
        A = mu_i/mu_sc-1.
        B = (K_i/k_sc - mu_i/mu_sc)/3.
        
        F1 = 1.0+A*( 1.5*(f+th) - R*(1.5*f+2.5*th-4.0/3.0) )
        F2 = 1.0+(A*(1.0+((3.0/2.0)*(f+th))-((1.0/2.0)*R*((3.0*f)+(5.0*th)))))+(B*(3.0-(4.0*R))) \
        +((1.0/2.0)*A*(A+(3.0*B))*(3.0-(4.0*R))*(f+th-(R*(f-th+(2.0*(th**2.0))))))
        T=3.0*F1/F2
        P=T/3.0
        
        F3=1.0+A*(1.0-(f+1.5*th)+R*(f+th))
        F4=1.0+0.25*A*(f+3.0*th-R*(f-th))
        F5=A*(-f+R*(f+th-4.0/3.0)) + B*th*(3.0-4.0*R)
        F6=1.0+A*(1+f-R*(f+th)) + B*(1.0-th)*(3.0-4.0*R)
        F7=2.0+0.25*A*(3.0*f+9.0*th-R*(3.0*f+5*th)) + B*th*(3.0-4.0*R)
        F8=A*(1.0-2.0*R +0.5*f*(R-1.0) + th*0.5*(5.0*R-3.0)) +B*(1.0-th)*(3.0-4.0*R)
        F9=A*((R-1.0)*f-R*th)+B*th*(3.0-4.0*R)
        Q =(2.0/F3+1.0/F4+(F4*F5+F6*F7-F8*F9)/(F2*F4))/5.0
        
        return [(K_i-k_sc)*P/(1-t), (mu_i-mu_sc)*Q/(1-t)]

    t_eval = np.linspace(0, Kp, 1000)
    res = solve_ivp(my_fun, (0, Kp), [K_m,mu_m], args = (asp, K_i, mu_i), method = 'DOP853', dense_output=True, t_eval = t_eval)
    
    K_eff = res.y.T[len(t_eval)-1,0]
    mu_eff = res.y.T[len(t_eval)-1,1]
    
    return K_eff,mu_eff

def Kuster_Toksoz_N (N,Kp,asp,K_m,mu_m,K_i,mu_i):
    """ 
    N-types of inclusions
    Input parameters:
    -----------------
    N        np.int, number of the inclusions type
    Kp       np.array(N), volume fraction of the embedded inclusions
    asp      np.array(N),  aspect ratios of the embedded inclusions
    K_m      np.float, bulk modulus of the matrix
    mu_m     np.float, shear modulus of the matrix
    K_i      np.array(N), bulk modulus of the inclusions (fluid)
    mu_i     np.array(N), shear modulus of the inclusions (fluid)
    

    Returns:
    --------
    K_kt     np.float, effective bulk modulus
    mu_kt    np.float, effective shear modulus
    """
    
    if (np.sum(Kp)>1):
        print('The sum of all inclusion volume fraction is greater than 1')
        return None

    
    P = np.zeros(N)
    Q = np.zeros(N)
    
    for i in range (N):
            th = (asp[i]/(1.-asp[i]**2.)**(1.5))*(math.acos(asp[i])-asp[i]*(1.-asp[i]**2.)**(1./2.))
            f=(asp[i]**2./(1.-asp[i]**2.))*(3.*th - 2.)
            
            nu_m = (3.*K_m-2.*mu_m)/2./(3.*K_m+mu_m)
            
            R = (1.-2.*nu_m)/(2.*(1.-nu_m))
            A = mu_i[i]/mu_m-1.
            B = (K_i[i]/K_m - mu_i[i]/mu_m)/3.
            
            F1 = 1.0+A*( 1.5*(f+th) - R*(1.5*f+2.5*th-4.0/3.0) )
            F2 = 1.0+(A*(1.0+((3.0/2.0)*(f+th))-((1.0/2.0)*R*((3.0*f)+(5.0*th)))))+(B*(3.0-(4.0*R))) \
            +((1.0/2.0)*A*(A+(3.0*B))*(3.0-(4.0*R))*(f+th-(R*(f-th+(2.0*(th**2.0))))))
            T=3.0*F1/F2
            P[i]=T/3.0
            
            F3=1.0+A*(1.0-(f+1.5*th)+R*(f+th))
            F4=1.0+0.25*A*(f+3.0*th-R*(f-th))
            F5=A*(-f+R*(f+th-4.0/3.0)) + B*th*(3.0-4.0*R)
            F6=1.0+A*(1+f-R*(f+th)) + B*(1.0-th)*(3.0-4.0*R)
            F7=2.0+0.25*A*(3.0*f+9.0*th-R*(3.0*f+5*th)) + B*th*(3.0-4.0*R)
            F8=A*(1.0-2.0*R +0.5*f*(R-1.0) + th*0.5*(5.0*R-3.0)) +B*(1.0-th)*(3.0-4.0*R)
            F9=A*((R-1.0)*f-R*th)+B*th*(3.0-4.0*R)
            Q[i]=(2.0/F3+1.0/F4+(F4*F5+F6*F7-F8*F9)/(F2*F4))/5.0
     

    A1 = np.sum(Kp*(K_i-K_m)*P)
    B1 = np.sum(Kp*(mu_i-mu_m)*Q)
    Sm = (mu_m*(9.*K_m+8.*mu_m))/(6.*(K_m+2.*mu_m))
    
    mu_kt = (mu_m**2+Sm*(mu_m+B1))/(mu_m+Sm-B1)
    K_kt = (K_m**2+(4./3.*mu_m)*(K_m+A1))/(K_m+4./3.*mu_m-A1)
    
    return K_kt, mu_kt

def double_porosity_model_DEM(K_1,mu_1,K_2, mu_2,Kp_pore,Kp_crack, asp_pore, asp_crack):
    
    """
    Input parameters:
    -----------------
    K_1        np.float, bulk modulus of the matrix
    mu_1       np.float, shear modulus of the matrix
    K_2        np.float, bulk modulus of the inclusions (fluid)
    mu_2       np.float, shear modulus of the inclusions (fluid)
    Kp_pore    np.float, volume fractions of pores (isometric pores)
    Kp_crack   np.float, volume fractions of cracks (non-isometric pores)
    asp_pore   np.float, aspect ratio of pores (isometric pores)
    asp_crack  np.float, aspect ratio of cracks (non-isometric pores)

    Returns:
    --------
    Keff     np.float, effective bulk modulus
    mueff    np.float, effective shear modulus
    """

    # add cracks
    Keff1, mueff1 = DEM(Kp_crack,asp_crack,K_1,mu_1,K_2,mu_2)
    # add pores
    Keff, mueff = DEM(Kp_pore,asp_pore,Keff1,mueff1,K_2,mu_2)
        
    return Keff, mueff 


def double_porosity_model_SC(K_1,mu_1,K_2, mu_2,Kp_pore,Kp_crack, asp_pore, asp_crack):

    """
    Input parameters:
    -----------------
    K_1        np.float, bulk modulus of the matrix
    mu_1       np.float, shear modulus of the matrix
    K_2        np.float, bulk modulus of the inclusions (fluid)
    mu_2       np.float, shear modulus of the inclusions (fluid)
    Kp_pore    np.float, volume fractions of pores (isometric pores)
    Kp_crack   np.float, volume fractions of cracks (non-isometric pores)
    asp_pore   np.float, aspect ratio of pores (isometric pores)
    asp_crack  np.float, aspect ratio of cracks (non-isometric pores)

    Returns:
    --------
    Keff     np.float, effective bulk modulus
    mueff    np.float, effective shear modulus
    """

    # add cracks
    Kp = np.array([Kp_crack,1-Kp_crack])
    asp = np.array([asp_crack,0.99])
    K = np.array([K_2, K_1])
    mu = np.array([mu_2,mu_1])
    nparam = 2
    Keff1, mueff1 = self_consistent_Berryman (Kp,asp,K,mu,nparam)
    # add pores
    Kp = np.array([Kp_pore,1-Kp_pore])
    asp = np.array([asp_pore,0.99])
    K = np.array([K_2, Keff1])
    mu = np.array([mu_2,mueff1])
    nparam = 2
    Keff, mueff =  self_consistent_Berryman (Kp,asp,K,mu,nparam)
    
    return Keff, mueff 


def crack_density(phi_crack, asp):
    """
    Input parameters:
    -----------------
    phi_crack       np.float, crack volume fraction (porosity)
    asp             np.float, crack aspect ratio

    Returns:
    --------
    crack density   np.float, crack density
    """
    if (phi_crack>=1):
        print ('Crack volume fraction porosity shoulb be in fractions')
        return None
   
    return 3.*phi_crack/(4.*np.pi*asp)


def elastic_parameters_Lambda(K, mu):
    """
    Input parameters:
    -----------------
    K              np.float, bulk modulus, GPa
    mu             np.float, shear modulus, GPa

    Returns:
    --------
    lamda          np.float, 1st Lame parameter, GPa
    """

    return K-(2./3.)*mu


Mineral_iso = {
    'Calcite': {'K':74.8, 'mu':30.6, 'rho': 2.71, 'Vp':6.53, 'Vs':3.36, 'nu':0.32, 'Ref':'Dandekar (1968)'},
    'Dolomite':{'K':76.4, 'mu':49.7,  'rho':2.87, 'Vp':7.05, 'Vs':4.16, 'nu':0.23, 'Ref':'Nur and Simmonds (1969b)' },      
    'Quatz' :  {'K':37.0, 'mu':44.0,  'rho':2.65, 'Vp':6.05, 'Vs':4.09, 'nu':0.08, 'Ref':'Carmichael(1989)'},
    'Pyrite' : {'K':147.4,'mu':132.5, 'rho':4.93, 'Vp':8.10, 'Vs':5.18, 'nu':0.15, 'Ref':'Simmons and Birch (1963)'},
    'Albite' : {'K':75.6, 'mu':25.6,  'rho':2.63, 'Vp':6.46, 'Vs':3.12, 'nu':0.35, 'Ref':'Woeber et al. (1963)'},
               }

Fluids = {
    'Brine':     {'K':2.5,   'mu':0.0, 'rho':1.04, 'Ref':''},
    'Heavy Oil': {'K':1.49,  'mu':0.0, 'rho':1.0,  'Ref':''},
    'Light Oil': {'K':1.07,  'mu':0.0, 'rho':0.87, 'Ref':''},
    'Gas':       {'K':0.006, 'mu':0.0, 'rho':0.5,  'Ref':''}
}

Kerogen = {'Vernik' : {'K':3.5, 'mu':1.75, 'rho': 1.25, 'Ref':'Vernik, 1994'},            
           'Blangy' : {'K':2.9, 'mu':2.70, 'rho': 1.3, 'Ref':'Blangy (1992),Carmichael(1989)'}
          }

Clays_iso = {
    'Illite_1':    {'K':52.3, 'mu':31.7, 'rho': 2.79, 'Vp':5.82, 'Vs':3.37, 'Ref':'Katahara, 1996'},
    'Illite_2':    {'K':60.2, 'mu':25.4, 'rho': 2.71, 'Vp':5.89, 'Vs':3.06, 'Ref':'Wang et al., 2001'},
    'Chlorite':    {'K':54.3, 'mu':30.2, 'rho': 2.69, 'Vp':5.93, 'Vs':3.35, 'Ref':''},
    'Kaolinite_1': {'K':55.5, 'mu':31.8, 'rho': 2.52, 'Vp':6.23, 'Vs':3.55, 'Ref':''},
    'Kaolinite_2': {'K':12.0, 'mu':6.0,  'rho': 2.59, 'Vp':2.78, 'Vs':1.52, 'Ref':'Vanorio et al., 2003;Prasad et al., 2002'},
    'Smectite_1':  {'K':7.0,  'mu':3.9,  'rho': 2.29, 'Vp':2.30, 'Vs':1.30, 'Ref':'Vanorio et al., 2003'},
    'Smectite_2':  {'K':9.3,  'mu':6.9,  'rho': 2.40, 'Vp':2.78, 'Vs':1.70, 'Ref':'Wang et al., 2001'},
    'Mixed_1':     {'K':21.4,  'mu':6.7,  'rho': 2.62, 'Vp':3.41, 'Vs':1.63,  
                   'Ref':'Han et al., 1986;Berge and Berryman, 1995',  'Composition':''},
    'Mixed_2':     {'K':21.4,  'mu':6.7,  'rho': 2.62, 'Vp':3.41, 'Vs':1.63, 'Ref':'Wang et al., 2001', 
                   'Composition':'Illite-smectite - 60/40'},
    'Mixed_3a':    {'K':8.1,  'mu':2.8,  'rho': 2.17, 'Vp':2.33, 'Vs':1.13, 'Ref':'Bayuk et al., 2007', 
                   'Composition':'Illite-smectite-kaolinite-chlorite'},
    'Mixed_3b':    {'K':16.0,  'mu':6.4, 'rho': 2.17, 'Vp':3.36, 'Vs':1.76, 'Ref':'Sayers, 2005', 
                   'Composition':'Illite-smectite-kaolinite-chlorite'},
}


Clays_ani = {
    'Illite':    {'C11':179.9, 'C33':55.0,  'C44': 11.7, 'C66':70.0, 'C13':14.5, 'Kself':52.3, 'muself':31.7 , 'Ref':'Alexandrov and Ryzhova, 1961'},
    'Kaolinite': {'C11':171.5, 'C33':52.6,  'C44': 14.8, 'C66':66.3, 'C13':27.1, 'Kself':55.5, 'muself':31.8 , 'Ref':'Alexandrov and Ryzhova, 1961'},
    'Chlorite':  {'C11':181.8, 'C33':106.8, 'C44': 11.4, 'C66':62.5, 'C13':20.3, 'Kself':54.3, 'muself':30.2 , 'Ref':'Alexandrov and Ryzhova, 1961'}
    }




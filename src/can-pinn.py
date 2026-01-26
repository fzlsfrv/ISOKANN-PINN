# specify a-PINN / n-PINN / can-PINN
def create_nn(scheme, ff, n_ffs, sigma, lmbda, n_nodes, acf, lr_int):
    # input layers -> split into (x, y, dx, dy)
    inputs = layers.Input(shape=(4,))
    x, y, dx, dy = layers.Lambda( lambda k: tf.split(k, num_or_size_splits=4, axis=1))(inputs)

    # features mapping
    initializer_ff = tf.keras.initializers.TruncatedNormal(stddev=sigma)  # features initializer
    
    if (ff == 'FF'):
        hidden_f0 = layers.Dense(n_ffs, activation='linear', use_bias=False, kernel_initializer=initializer_ff)(layers.Concatenate()([x, y]))
        hidden_sin, hidden_cos = tf.math.sin(2*tf.constant(pi)*hidden_f0), tf.math.cos(2*tf.constant(pi)*hidden_f0)
        hidden_ff = layers.Concatenate()([hidden_sin, hidden_cos])
        
    if (ff == 'SF') or (ff == 'SIREN'):
        hidden_f0 = layers.Dense(n_ffs*2, activation='linear', kernel_initializer=initializer_ff)(layers.Concatenate()([x, y]))
        hidden_ff = tf.math.sin(2*tf.constant(pi)*hidden_f0)

    if (ff == 'HF'):
        hidden_ff = layers.Dense(n_ffs*2, activation=acf, kernel_initializer=initializer_ff)(layers.Concatenate()([x, y]))

    # hidden layers
    if (ff == 'SIREN'):
        initializer = tf.keras.initializers.HeUniform()  # hidden layers initializer
        hidden_1 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_ff)
        hidden_2 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_1)
        hidden_l = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_2)
    else:
        initializer = tf.keras.initializers.GlorotUniform()  # hidden layers initializer
        hidden_1 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_ff)
        hidden_2 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_1)
        hidden_l = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_2)

    # split layers - u
    if (ff == 'SIREN'):
        hidden_u1 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_l)
        hidden_u2 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_u1)
        hidden_ul = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_u2)
    else:
        hidden_u1 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_l)
        hidden_u2 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_u1)
        hidden_ul = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_u2)

    # split layers - v
    if (ff == 'SIREN'):
        hidden_v1 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_l)
        hidden_v2 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_v1)
        hidden_vl = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_v2)
    else:
        hidden_v1 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_l)
        hidden_v2 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_v1)
        hidden_vl = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_v2)  
        
    # split layers - p
    if (ff == 'SIREN'):
        hidden_p1 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_l)
        hidden_p2 = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_p1)
        hidden_pl = layers.Dense(n_nodes, activation=tf.math.sin, kernel_initializer=initializer)(hidden_p2)
    else:
        hidden_p1 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_l)
        hidden_p2 = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_p1)
        hidden_pl = layers.Dense(n_nodes, activation=acf, kernel_initializer=initializer)(hidden_p2)          
        
    # output layers
    u = layers.Dense(1, use_bias=False, name="U")(hidden_ul)
    v = layers.Dense(1, use_bias=False, name="V")(hidden_vl)
    p = layers.Dense(1, use_bias=False, name="P")(hidden_pl)  
    
    # initiate model
    outputs = layers.Concatenate()([u, v, p]) 
    nn = models.Model(inputs=inputs, outputs=outputs)
    
    # axillary PDE outputs
    u_x, u_y = K.gradients(u, x)[0], K.gradients(u, y)[0]
    v_x, v_y = K.gradients(v, x)[0], K.gradients(v, y)[0]
    p_x, p_y = K.gradients(p, x)[0], K.gradients(p, y)[0]
    u_xx, u_yy = K.gradients(u_x, x)[0], K.gradients(u_y, y)[0]
    v_xx, v_yy = K.gradients(v_x, x)[0], K.gradients(v_y, y)[0]    

    # initial & boundary conditions:
    # Top    : u = 1 , v = 0
    # Left   : u = 0 , v = 0
    # Right  : u = 0 , v = 0
    # Bottom : u = 0 , v = 0
    _top, _bottom = tf.equal(y, y_u), tf.equal(y, y_l)
    _left, _right = tf.equal(x, x_l), tf.equal(x, x_u)
    _bc = tf.logical_or( tf.logical_or(_top, _bottom) , tf.logical_or(_left, _right) )

    u_top, v_top = tf.boolean_mask(u, _top), tf.boolean_mask(v, _top)
    bc_top = tf.compat.v1.losses.mean_squared_error(labels=tf.ones_like(u_top), predictions=u_top) + \
             tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(v_top), predictions=v_top)

    u_left, v_left = tf.boolean_mask(u, _left & ~_top), tf.boolean_mask(v, _left & ~_top)
    bc_left = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(u_left), predictions=u_left) + \
              tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(v_left), predictions=v_left)

    u_right, v_right = tf.boolean_mask(u, _right & ~_top), tf.boolean_mask(v, _right & ~_top)
    bc_right = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(u_right), predictions=u_right) + \
               tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(v_right), predictions=v_right)

    u_bottom, v_bottom = tf.boolean_mask(u, _bottom), tf.boolean_mask(v, _bottom)
    bc_bottom = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(u_bottom), predictions=u_bottom) + \
                tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(v_bottom), predictions=v_bottom)
    
    bc_mse = bc_top + bc_left + bc_right + bc_bottom
    
    # PDE (NS equation)
    # Continuity equation : u_x + v_y = 0
    # Momentum equation 1 : u_t + u*u_x + v*u_y = -(1/rho)*p_x + nu*(u_xx + u_yy)
    # Momentum equation 2 : v_t + u*v_x + v*v_y = -(1/rho)*p_y + nu*(v_xx + v_yy)
    
    # auto-differentian PDE (a-pde)
    a_residuals_continuity = u_x + v_y
    a_residuals_momentum_1 = u*u_x + v*u_y + p_x - 1.0/Re*(u_xx + u_yy)
    a_residuals_momentum_2 = u*v_x + v*v_y + p_y - 1.0/Re*(v_xx + v_yy)
    
    # exclude BC points 
    a_residuals_continuity = tf.boolean_mask(a_residuals_continuity, ~_bc)
    a_residuals_momentum_1 = tf.boolean_mask(a_residuals_momentum_1, ~_bc)
    a_residuals_momentum_2 = tf.boolean_mask(a_residuals_momentum_2, ~_bc)
    
    a_mse_continuity = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(a_residuals_continuity),
                                                              predictions=a_residuals_continuity)
    a_mse_momentum_1 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(a_residuals_momentum_1),
                                                              predictions=a_residuals_momentum_1)
    a_mse_momentum_2 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(a_residuals_momentum_2),
                                                              predictions=a_residuals_momentum_2)
    a_pde_mse = a_mse_continuity + a_mse_momentum_1 + a_mse_momentum_2
    a_pde_mse = a_pde_mse / lmbda     
    
    
    # numerical differentiation PDE (n-pde)
    # dx & dy get from input
    xE, xW = x + dx, x - dx
    yN, yS = y + dy, y - dy
    uvpE  = nn(tf.stack([xE, y, dx, dy], 1))
    uvpW  = nn(tf.stack([xW, y, dx, dy], 1))
    uvpN  = nn(tf.stack([x, yN, dx, dy], 1))
    uvpS  = nn(tf.stack([x, yS, dx, dy], 1))
    uE, vE, pE  = tf.split(uvpE, num_or_size_splits=3, axis=1)
    uW, vW, pW  = tf.split(uvpW, num_or_size_splits=3, axis=1)
    uN, vN, pN  = tf.split(uvpN, num_or_size_splits=3, axis=1)
    uS, vS, pS  = tf.split(uvpS, num_or_size_splits=3, axis=1)
    
    # second order
    xEE, xWW = x + 2.0*dx, x - 2.0*dx
    yNN, ySS = y + 2.0*dy, y - 2.0*dy    
    uvpEE = nn(tf.stack([xEE, y, dx, dy], 1))
    uvpWW = nn(tf.stack([xWW, y, dx, dy], 1))
    uvpNN = nn(tf.stack([x, yNN, dx, dy], 1))
    uvpSS = nn(tf.stack([x, ySS, dx, dy], 1))  
    uEE, vEE, _ = tf.split(uvpEE, num_or_size_splits=3, axis=1)
    uWW, vWW, _ = tf.split(uvpWW, num_or_size_splits=3, axis=1)
    uNN, vNN, _ = tf.split(uvpNN, num_or_size_splits=3, axis=1)
    uSS, vSS, _ = tf.split(uvpSS, num_or_size_splits=3, axis=1)
    
    uc_e, uc_w = 0.5*(uE + u), 0.5*(uW + u) 
    vc_n, vc_s = 0.5*(vN + v), 0.5*(vS + v)
    div = (uc_e - uc_w) /dx + (vc_n - vc_s) /dy
    
    # 2nd upwind
    Uem_uw2 = 1.5*u  - 0.5*uW
    Uep_uw2 = 1.5*uE - 0.5*uEE  
    Uwm_uw2 = 1.5*uW - 0.5*uWW
    Uwp_uw2 = 1.5*u  - 0.5*uE
    Ue_uw2 = tf.where(tf.greater_equal(uc_e, 0.0), Uem_uw2, Uep_uw2)
    Uw_uw2 = tf.where(tf.greater_equal(uc_w, 0.0), Uwm_uw2, Uwp_uw2)
        
    Unm_uw2 = 1.5*u  - 0.5*uS
    Unp_uw2 = 1.5*uN - 0.5*uNN    
    Usm_uw2 = 1.5*uS - 0.5*uSS
    Usp_uw2 = 1.5*u  - 0.5*uN
    Un_uw2 = tf.where(tf.greater_equal(vc_n, 0.0), Unm_uw2, Unp_uw2)
    Us_uw2 = tf.where(tf.greater_equal(vc_s, 0.0), Usm_uw2, Usp_uw2)

    Vem_uw2 = 1.5*v  - 0.5*vW
    Vep_uw2 = 1.5*vE - 0.5*vEE
    Vwm_uw2 = 1.5*vW - 0.5*vWW
    Vwp_uw2 = 1.5*v  - 0.5*vE
    Ve_uw2 = tf.where(tf.greater_equal(uc_e, 0.0), Vem_uw2, Vep_uw2)
    Vw_uw2 = tf.where(tf.greater_equal(uc_w, 0.0), Vwm_uw2, Vwp_uw2)
        
    Vnm_uw2 = 1.5*v  - 0.5*vS
    Vnp_uw2 = 1.5*vN - 0.5*vNN    
    Vsm_uw2 = 1.5*vS - 0.5*vSS
    Vsp_uw2 = 1.5*v  - 0.5*vN
    Vn_uw2 = tf.where(tf.greater_equal(vc_n, 0.0), Vnm_uw2, Vnp_uw2)
    Vs_uw2 = tf.where(tf.greater_equal(vc_s, 0.0), Vsm_uw2, Vsp_uw2)
        
    UUx_uw2 = (uc_e*Ue_uw2 - uc_w*Uw_uw2) /dx
    VUy_uw2 = (vc_n*Un_uw2 - vc_s*Us_uw2) /dy
    UVx_uw2 = (uc_e*Ve_uw2 - uc_w*Vw_uw2) /dx
    VVy_uw2 = (vc_n*Vn_uw2 - vc_s*Vs_uw2) /dy
    
    # 2nd central difference    
    Uxx_cd2 = (uE - 2.0*u + uW)/ (dx*dx) 
    Uyy_cd2 = (uN - 2.0*u + uS)/ (dy*dy) 
    Vxx_cd2 = (vE - 2.0*v + vW)/ (dx*dx) 
    Vyy_cd2 = (vN - 2.0*v + vS)/ (dy*dy) 

    pe_cd2 = (p + pE) /2.0 
    pw_cd2 = (pW + p) /2.0 
    pn_cd2 = (p + pN) /2.0 
    ps_cd2 = (pS + p) /2.0 
    
    Px_cd2 = (pe_cd2 - pw_cd2) /dx
    Py_cd2 = (pn_cd2 - ps_cd2) /dy
        
    n_residuals_continuity = div
    n_residuals_momentum_1 = UUx_uw2 + VUy_uw2 - 1.0/Re *(Uxx_cd2 + Uyy_cd2) - u*div + Px_cd2
    n_residuals_momentum_2 = UVx_uw2 + VVy_uw2 - 1.0/Re *(Vxx_cd2 + Vyy_cd2) - v*div + Py_cd2   
    
    # exclude BC points 
    n_residuals_continuity = tf.boolean_mask(n_residuals_continuity, ~_bc)
    n_residuals_momentum_1 = tf.boolean_mask(n_residuals_momentum_1, ~_bc)
    n_residuals_momentum_2 = tf.boolean_mask(n_residuals_momentum_2, ~_bc)
    
    n_mse_continuity = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(n_residuals_continuity),
                                                              predictions=n_residuals_continuity)
    n_mse_momentum_1 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(n_residuals_momentum_1),
                                                              predictions=n_residuals_momentum_1)
    n_mse_momentum_2 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(n_residuals_momentum_2),
                                                              predictions=n_residuals_momentum_2)
    n_pde_mse = n_mse_continuity + n_mse_momentum_1 + n_mse_momentum_2
    n_pde_mse = n_pde_mse / lmbda     
    
    
    # coupled automatic-numerical differentiation PDE (can-pde)
    uE_x, uW_x = K.gradients(uE, xE)[0], K.gradients(uW, xW)[0]
    uN_y, uS_y = K.gradients(uN, yN)[0], K.gradients(uS, yS)[0]
    
    vE_x, vW_x = K.gradients(vE, xE)[0], K.gradients(vW, xW)[0]
    vN_y, vS_y = K.gradients(vN, yN)[0], K.gradients(vS, yS)[0]   
    
    pE_x, pW_x = K.gradients(pE, xE)[0], K.gradients(pW, xW)[0]
    pN_y, pS_y = K.gradients(pN, yN)[0], K.gradients(pS, yS)[0]        
    
    # can 2nd upwind
    Uem_cuw2 = u  +  u_x*dx /2.0 #+ (uE_x - u_x)*dx /8.0
    Uep_cuw2 = uE - uE_x*dx /2.0 #+ (uE_x - u_x)*dx /8.0  
    Uwm_cuw2 = uW + uW_x*dx /2.0 #+ (u_x - uW_x)*dx /8.0
    Uwp_cuw2 = u  -  u_x*dx /2.0 #+ (u_x - uW_x)*dx /8.0
    Ue_cuw2 = tf.where(tf.greater_equal(uc_e, 0.0), Uem_cuw2, Uep_cuw2)
    Uw_cuw2 = tf.where(tf.greater_equal(uc_w, 0.0), Uwm_cuw2, Uwp_cuw2)    
    
    Unm_cuw2 = u  +  u_y*dy /2.0 #+ (uN_y - u_y)*dy /8.0
    Unp_cuw2 = uN - uN_y*dy /2.0 #+ (uN_y - u_y)*dy /8.0 
    Usm_cuw2 = uS + uS_y*dy /2.0 #+ (u_y - uS_y)*dy /8.0
    Usp_cuw2 = u  -  u_y*dy /2.0 #+ (u_y - uS_y)*dy /8.0
    Un_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Unm_cuw2, Unp_cuw2)
    Us_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Usm_cuw2, Usp_cuw2)

    Vem_cuw2 = v  +  v_x*dx /2.0 #+ (vE_x - v_x)*dx /8.0
    Vep_cuw2 = vE - vE_x*dx /2.0 #+ (vE_x - v_x)*dx /8.0
    Vwm_cuw2 = vW + vW_x*dx /2.0 #+ (v_x - vW_x)*dx /8.0
    Vwp_cuw2 = v  -  v_x*dx /2.0 #+ (v_x - vW_x)*dx /8.0
    Ve_cuw2 = tf.where(tf.greater_equal(uc_e, 0.0), Vem_cuw2, Vep_cuw2)
    Vw_cuw2 = tf.where(tf.greater_equal(uc_w, 0.0), Vwm_cuw2, Vwp_cuw2)
        
    Vnm_cuw2 = v  +  v_y*dy /2.0 #+ (vN_y - v_y)*dy /8.0
    Vnp_cuw2 = vN - vN_y*dy /2.0 #+ (vN_y - v_y)*dy /8.0
    Vsm_cuw2 = vS + vS_y*dy /2.0 #+ (v_y - vS_y)*dy /8.0
    Vsp_cuw2 = v  -  v_y*dy /2.0 #+ (v_y - vS_y)*dy /8.0
    Vn_cuw2 = tf.where(tf.greater_equal(vc_n, 0.0), Vnm_cuw2, Vnp_cuw2)
    Vs_cuw2 = tf.where(tf.greater_equal(vc_s, 0.0), Vsm_cuw2, Vsp_cuw2)    
    
    UUx_cuw2 = (uc_e*Ue_cuw2 - uc_w*Uw_cuw2) /dx
    VUy_cuw2 = (vc_n*Un_cuw2 - vc_s*Us_cuw2) /dy
    UVx_cuw2 = (uc_e*Ve_cuw2 - uc_w*Vw_cuw2) /dx
    VVy_cuw2 = (vc_n*Vn_cuw2 - vc_s*Vs_cuw2) /dy       
    
    # can 2nd central difference    
    pe_ccd2 = (p + pE) /2.0 - (pE_x - p_x)*dx /8.0
    pw_ccd2 = (pW + p) /2.0 - (p_x - pW_x)*dx /8.0
    pn_ccd2 = (p + pN) /2.0 - (pN_y - p_y)*dy /8.0
    ps_ccd2 = (pS + p) /2.0 - (p_y - pS_y)*dy /8.0
        
    Px_ccd2 = (pe_ccd2 - pw_ccd2) /dx
    Py_ccd2 = (pn_ccd2 - ps_ccd2) /dy    
    
    can_residuals_continuity = div
    can_residuals_momentum_1 = UUx_cuw2 + VUy_cuw2 - 1.0/Re *(Uxx_cd2 + Uyy_cd2) - u*div + Px_ccd2
    can_residuals_momentum_2 = UVx_cuw2 + VVy_cuw2 - 1.0/Re *(Vxx_cd2 + Vyy_cd2) - v*div + Py_ccd2

    # exclude BC points 
    can_residuals_continuity = tf.boolean_mask(can_residuals_continuity, ~_bc)
    can_residuals_momentum_1 = tf.boolean_mask(can_residuals_momentum_1, ~_bc)
    can_residuals_momentum_2 = tf.boolean_mask(can_residuals_momentum_2, ~_bc)
    
    can_mse_continuity = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(can_residuals_continuity),
                                                                predictions=can_residuals_continuity)
    can_mse_momentum_1 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(can_residuals_momentum_1),
                                                                predictions=can_residuals_momentum_1)
    can_mse_momentum_2 = tf.compat.v1.losses.mean_squared_error(labels=tf.zeros_like(can_residuals_momentum_2),
                                                                predictions=can_residuals_momentum_2)
    can_pde_mse = can_mse_continuity + can_mse_momentum_1 + can_mse_momentum_2
    can_pde_mse = can_pde_mse / lmbda  
    
    
    # which method to use for PDE loss computation? a-PDE or n-PDE or can-PDE
    if (scheme == 'a-pde'):
        pde_mse = a_pde_mse
    if (scheme == 'n-pde'):
        pde_mse = n_pde_mse
    if (scheme == 'can-pde'):
        pde_mse = can_pde_mse    

        
    # optimizer
    optimizer = tf.keras.optimizers.Adam(lr_int)

    # compile model with [?] loss
    nn.compile(loss = compute_physics_loss(pde_mse, bc_mse),
               optimizer = optimizer,
               metrics = [compute_u_loss(dx), compute_v_loss(dy),
                          compute_bc_loss(bc_mse), compute_pde_loss(pde_mse)])

    # pathway to NN inside variables
    insiders = [u, v, p, pde_mse, bc_mse]
    eval_ins = K.function([nn.input, K.learning_phase()], insiders)   # evaluation function

    return (nn, eval_ins)

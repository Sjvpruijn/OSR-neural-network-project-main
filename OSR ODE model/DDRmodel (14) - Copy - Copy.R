## LACDR model for DNA damage response (DDR)
## Created by Muriel Heldring and made available for BOO 2020
## Date: 02/04/2020
##
## This model contains 6 state variables, namely 
## - DNA damage (DD)
## - p53 (P53)
## - phosphorylated p53 (P53P)
## - MDM2 (MDM2)
## - p21 (P21)
## - BTG2 (BTG2)

# First load the deSolve package. If this gives an error, you probably need 
# to install it first with install.packages("deSolve")
library('deSolve')

### SECTION I: Build the model and assign the parameter values ###

# Function that defines the ddr model including the stress input function
ddr <- function(t, inistate, parameters) {
  with(as.list(c(inistate, parameters)), {

    dS = -time_constant1 * S
    dDD  = ks_dd - kd_dd * DD + S
    dP53  = ks_p53 + dephos * P53P - phos * P53 * DD - kd_p53 * P53 - kd_p53_by_mdm2 * MDM2 * P53
    dP53P  = phos * P53 * DD - dephos * P53P - kd_p53p * P53P - kd_p53p_by_mdm2 * MDM2 * P53P
    dMDM2 = ks_mdm2 + (ks_mdm2_by_p53p * P53P**n) / (Km_mdm2**n + P53P**n) - kd_mdm2 * MDM2
    dP21  = ks_p21 + (ks_p21_by_p53p * P53P**n) / (Km_p21**n + P53P**n) - kd_p21 * P21
    dBTG2  = ks_btg2 + (ks_btg2_by_p53p * P53P**n) / (Km_btg2**n + P53P**n) - kd_btg2 * BTG2
    
    list(c(dS, dDD, dP53, dP53P, dMDM2, dP21, dBTG2))
  })
}

# Make a function that defines the parameters and 
# runs the ddr model with those parameters
ddrdose <- function(stress, tspan){
  if(1){
    pars.nostim = c(
      n=4,
      
      kd_dd = 0.108945157293661,
      dephos = 0.000468314849407653,
      kd_p53p = 0.085615829117503,
      kd_p53 = 0.5,  #kd_p53 = 1.4870247593958,
      kd_p53_by_mdm2 = 0.00441226604226379,
      kd_p53p_by_mdm2 = 0.000596158383212953,
      ks_mdm2 = 0.0458064479399243,
      ks_mdm2_by_p53p = 0.0201354678741032,
      ks_p21 = 0.000253113187517087,
      ks_p21_by_p53p = 99.4038034299917,
      Km_mdm2 = 0.174281399509842,
      Km_p21 = 2.55706150753566,
      Km_btg2 = 0.19316893595186,
      ks_btg2 = 0.12653028855262,
      ks_btg2_by_p53p = 0.2002780765973,
      
      ks_dd = 0.23810892003848794,
      phos = 0.1704687850596265,
      ks_p53 = 0.12726460633028158,
      kd_mdm2 = 0.4129756286953499,
      kd_p21 = 0.0001465624392043179,
      kd_btg2 = 0.08602950866891562,
      
      time_constant1 = 0.05
    )
  }
  inistate = c(
    DD = 2.18558516921194,
    P53 = 0.0684542572776999,
    P53P = 0.153491198428309,
    MDM2 = 0.129232950303743,
    P21 = 10.5322935656957,
    BTG2 = 2.13431252095854
  )
  #
  tspan_pre <-  seq(0, 1000, by = 1)
  ic <-  ode(
    y = c(S = 0, inistate), # stress is equal to 0
    times = tspan_pre,
    func = ddr,
    parms = pars.nostim
  )

  out <-  ode(
    y = c(S = stress, ic[nrow(ic),3:ncol(ic)]), # stress is equal to stresslevel
    times = tspan,
    func = ddr,
    parms = pars.nostim
  )
  return(out)
}

# Set the scaling an offset parameters to rescale the model output
sf_p53 <- 3.28835423065223
sf_mdm2 <- 19.8760689209942
sf_p21 <- 0.448726751792301
sf_btg2 <- 0.309196156820059
offset_p53 <- -0.569545578831001
offset_mdm2 <- -2.36437614248354
offset_p21 <- -4.64881738836483
offset_btg2 <- -0.592105707826557

### SECTION II: Run the ddr model ###

# Select the time span for which you want to run the model
tspan <-  seq(0, 43, by = 1)

# Run the model and save the simulation in variable 'xx'
xx <- ddrdose(stress = 1, tspan)

# Rescale the model simulation
P53scaled <- sf_p53 * (data.frame(xx)$P53 + data.frame(xx)$P53P) + offset_p53
MDM2scaled <- sf_mdm2 * data.frame(xx)$MDM2 + offset_mdm2
P21scaled <- sf_p21 * data.frame(xx)$P21 + offset_p21
BTG2scaled <- sf_btg2 * data.frame(xx)$BTG2 + offset_btg2

### SECTION III: Plot the output ###

par(mfrow=c(2,3))

plot(tspan, data.frame(xx)$S, xlab="time (hours)", ylab="Stress (a.u.)", ylim = c(0,4))
plot(tspan, data.frame(xx)$DD, xlab="time (hours)", ylab="DNA damage (a.u.)")
plot(tspan, P53scaled, xlab="time (hours)", ylab="p53 (a.u.)")
plot(tspan, MDM2scaled, xlab="time (hours)", ylab="Mdm2 (a.u.)")
plot(tspan, P21scaled, xlab="time (hours)", ylab="p21 (a.u.)")
plot(tspan, BTG2scaled, xlab="time (hours)", ylab="Btg2 (a.u.)")

P53scaled

## LACDR model for oxidative stress response (OSR)
## Created by Isoude Kuijper and made available for BOO 2020
## Date: 06/04/2020
##
## This model contains 5 state variables, namely 
## - Nrf2 (Nrf2)
## - nuclear Nrf2 (nNrf2)
## - Gsh (Gsh)
## - Srxn1 (Srxn1)
## - reactive metabolites (RM)

# First load the deSolve package. If this gives an error, you probably need 
# to install it first with install.packages("deSolve")
library('deSolve')

### SECTION I: Build the model and assign the parameter values ###

# Function that defines the ddr model including the stress input function
osr <- function(t, inistate, parameters) {
  with(as.list(c(inistate, parameters)), {

    dS = -time_constant1 * S
    dNrf2 = exportNrf2 * nNrf2 - importNrf2 * Nrf2 + buildNrf2Base - (vMax * Nrf2) / (km * (1 + RM / kiRM) + Nrf2)
    dnNrf2 = importNrf2 * Nrf2 - exportNrf2 * nNrf2
    dGsh = buildGshBase + buildGsh * nNrf2**hill_Gsh / (K_Gsh**hill_Gsh + nNrf2**hill_Gsh) - conjFormToDegrGsh * conjForm * Gsh - conjForm * Gsh * RM
    dSrxn1 = buildSrxn1Base + buildSrxn1 * nNrf2**hill_Srxn1 / (K_Srxn1**hill_Srxn1 + nNrf2**hill_Srxn1)  - degradSrxn1 * Srxn1
    dRM = buildRM - conjForm * Gsh * RM - degradRM * RM + S
    
    list(c(dS, dNrf2, dnNrf2, dGsh, dSrxn1, dRM))
  })
}

# Make a function that defines the parameters and 
# runs the ddr model with those parameters
osrdose <- function(stress, tspan){
  if(1){
    pars.nostim = c(

      importNrf2 = 4.251955e-04,
      exportNrf2 = 4.672772e+00,
      buildNrf2Base = 2.297422e+02,
      vMax = 1.345040e+08,
      km = 3.816342e+02,
      kiRM = 1.338881e-06,
      buildGshBase = 5.530482e-02,
      buildGsh = 1.063481e+00,
      hill_Gsh = 6.499604e+00,
      K_Gsh = 4.580483e+00,
      conjForm = 1.240263e-04,
      buildSrxn1Base = 9.996659e-09,
      buildSrxn1 = 3.405193e+06,
      hill_Srxn1 = 6.003976e+00,
      K_Srxn1 = 4.833310e-01,
      conjFormToDegrGsh = 1.709325e+03,
      buildRM = 3.386074e-07,
      degradSrxn1 = 3.687709e-08,
      degradRM = 9.137437e-02,
      
      time_constant1 = 0.75
    )
  }
  inistate = c(
    Nrf2 = 2.455416e-03,
    nNrf2 = 2.234288e-07,
    Gsh = 2.608704e-01,
    Srxn1 = 2.710805e-01,
    RM = 3.704403e-06
  )
  #
  tspan_pre <-  seq(0, 1000, by = 1)
  ic <-  ode(
    y = c(S = 0, inistate), # stress is equal to 0
    times = tspan_pre,
    func = osr,
    parms = pars.nostim
  )

  out <-  ode(
    y = c(S = stress, ic[nrow(ic),3:ncol(ic)]), # stress is equal to stresslevel
    times = tspan,
    func = osr,
    parms = pars.nostim
  )
  return(out)
}

### SECTION II: Run the ddr model ###

# Select the time span for which you want to run the model
tspan <-  seq(0, 49, by = 1)

# Run the model and save the simulation in variable 'xx'
xx <- osrdose(stress = 4, tspan)

# Rescale the model simulation
Nrf2 <- data.frame(xx)$Nrf2
nNrf2 <- data.frame(xx)$nNrf2
Gsh <- data.frame(xx)$Gsh
Srxn1 <- data.frame(xx)$Srxn1
RM <- data.frame(xx)$RM

### SECTION III: Plot the output ###

par(mfrow=c(2,3))

plot(tspan, data.frame(xx)$S, xlab="time (hours)", ylab="Stress (a.u.)", ylim = c(0,1))
plot(tspan, Nrf2, xlab="time (hours)", ylab="Nrf2 (a.u.)")
plot(tspan, nNrf2, xlab="time (hours)", ylab="nNrf2 (a.u.)")
plot(tspan, Gsh, xlab="time (hours)", ylab="Gsh (a.u.)")
plot(tspan, Srxn1, xlab="time (hours)", ylab="Srxn1 (a.u.)")
plot(tspan, RM, xlab="time (hours)", ylab="RM (a.u.)")


# Display parameters
figsize_1graph = (11, 11)
figsize_1graph_xtend = (14.67, 11)
figsize_1graph_ytend = (11, 15)
figsize_2graph_rows = (11, 22)
figsize_3graph_rows = (11, 33)
figsize_2graph_cols = (22, 11)
figsize_6graph_cols = (11, 66)
figsize_6graph_2rows_3cols = (25, 15)
figsize_12graph_3rows_4cols = (20, 15)
fontsize = 18

# Physical constants
mars_sol = 88775  # durée d'un sol en secondes
cst_stefan = 5.67e-8  # S.I.

# Miscellaneous parameters
threshold = 1e-13

# Simulation parameter
ndynstep = 960  # nombres de step en un sol
ecritphy = 80  # on écrit les data tous les 80 step
ptimestep = (mars_sol/ndynstep) * ecritphy  # ptimestep est le temps entre deux sorties

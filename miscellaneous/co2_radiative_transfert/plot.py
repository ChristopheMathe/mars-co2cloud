import numpy as np
import matplotlib.pyplot as plt

list_filename = ['co2i4.rfi', 'co2_ref_index_IR.txt', 'co2_ref_index_testIR.txt', 'co2_ref_index.txt']
list_color = ['black', 'red', 'blue', 'green']
list_linestyle = ['solid', 'dashed', 'dotted', 'dashdot']
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(11,11))
fig.suptitle('CO$_2$ optical index')
ax[0].set_title('Real part')
ax[1].set_title('Imaginary part')


for f, filename in enumerate(list_filename):
	dataset = np.loadtxt(filename)
	ax[0].plot(dataset[:, 0], dataset[:, 1], ls=list_linestyle[f], color=list_color[f], label=filename)
	ax[1].plot(dataset[:, 0], dataset[:, 2], ls=list_linestyle[f], color=list_color[f], label=filename)

for axes in ax.reshape(-1):
	axes.set_xscale('log')
	axes.legend(loc=0)

plt.savefig('comparison_input_file.png', bbox_inches='tight')
plt.show()

import sys
sys.path.append("./../")
import matplotlib.pyplot as plt
from jax import random
import numpy as np
from nt_transfer import *
import numpy as onp
from nt_transfer.nn_models import model_dict
from nt_transfer.plot_tools import *
import matplotlib.ticker as ticker

gen_kwargs = dict(dataset_str =  'mnist',
                  model_str = 'cnn_lenet_caffe',
                  NN_DENSITY_LEVEL_LIST = [0.03], # the fraction of weight remainining
                  OPTIMIZER_STR = 'adam', # the optimizer
                  NUM_RUNS = 2, # two runs (note that in our paper, we use NUM_RUNS = 5)
                  NUM_EPOCHS  = 20, # number of epochs
                  BATCH_SIZE  = 64,  # batch size
                  STEP_SIZE = 5e-4,  # SGD step size
                  MASK_UPDATE_FREQ = 100, # mask update frequency
                  LAMBDA_KER_DIST = 1e-3, # the strength constant for NTK distance used in NTT loss function
                  LAMBDA_L2_REG = 1e-5, # the l2 regularization constant
                  SAVE_BOOL = True,
                  save_dir = './../ntt_results/')

model = nt_transfer_model(**gen_kwargs, instance_input_shape = [28, 28, 1])
_, _, nt_trans_vali_all_sparsities_all_runs = model.optimize()

axis_font = 18

fig = plt.figure(constrained_layout=True, figsize=(5, 6))

gs = fig.add_gridspec(10, 12)
ax1 = fig.add_subplot(gs[:5, :])

# three trials of target distance
ax1.plot(nt_trans_vali_all_sparsities_all_runs[0][0][:, 0], label='run 1');
ax1.plot(nt_trans_vali_all_sparsities_all_runs[0][1][:, 0], label='run 2');
ax1.legend(ncol=1)
ax1.set_title('NTK distance', fontsize=axis_font)
ax1.set_ylabel('Loss', fontsize=axis_font)
ax1 = simpleaxis(ax1)

ax2 = fig.add_subplot(gs[5:, :])
ax2.plot(nt_trans_vali_all_sparsities_all_runs[0][0][:, 1], label='run 1');
ax2.plot(nt_trans_vali_all_sparsities_all_runs[0][1][:, 1], label='run 2');
ax2.legend(ncol=1)
ax2.set_title('Target distance', fontsize=axis_font)
ax2.set_ylabel('Loss', fontsize=axis_font)
ax2.set_xlabel('Iteration number / 100', fontsize=axis_font)
ax2 = simpleaxis(ax2)

gen_kwargs_supervised = dict(
ntt_file_name = 'mnist_layerwise_prune_cnn_lenet_caffe', # the saved result in ntt_results file
dataset_str  = 'mnist', # dataset to use'
sup_density_list = [0.03],  # the density levels to achieve'
OPTIMIZER_STR = 'adam', # optimizer'
EXPLOITATION_NUM_EPOCHS = 50, #number of training epochs'
EXPLOITATION_BATCH_SIZE  = 64, # number of samples in a minibatch'
STEP_SIZE = 1e-3, # learning step-size'
REG = 1e-8, # l2 regularization constant'
EXPLOIT_TRAIN_DATASET_FRACTION = 0.1, # the fraction of training data used as validation data'
RECORD_ACC_FREQ = 100, # frequency for saving the training and testing result'
save_supervised_result_bool = True)



model = exploit_model(ntt_file_name = gen_kwargs_supervised['ntt_file_name'], ntt_saved_path = './../ntt_results/', supervised_result_path = './../supervised_results/')

sup_learning_results_ntt_init = model.supervised_optimization(wiring_str = 'trans',  ** gen_kwargs_supervised)

sup_learning_results_rand_init = model.supervised_optimization(wiring_str = 'rand',  ** gen_kwargs_supervised)

RECORD_ACC_FREQ = gen_kwargs_supervised['RECORD_ACC_FREQ']

fig, ax = plt.subplots(1, 2)

selected_density_list = ['0.03']

fig.subplots_adjust(left=.15, bottom=.3, right=.94, top=.75, wspace = 0.5)

plot_length = onp.arange(sup_learning_results_rand_init['train_results'][selected_density_list[0]].shape[1])

title_str = str(100 * round( 1 - onp.float32(selected_density_list[0]), 2)) + '\% sparse'

ax[0].plot( plot_length * RECORD_ACC_FREQ, np.mean(sup_learning_results_rand_init['train_results'][selected_density_list[0]], axis = 0), label = 'Rand. init', color = '#998ec3', linestyle = '-',lw = 1.5)
ax[0].plot( plot_length * RECORD_ACC_FREQ, np.mean(sup_learning_results_ntt_init['train_results'][selected_density_list[0]], axis = 0), ** gen_kwargs_ntt_student_plot)
ax[0].legend(loc='upper left', bbox_to_anchor= (0.3, 1.7), ncol= 2, columnspacing = 0.7, frameon=False, fontsize = 'large')

ax[0].grid(linestyle='-', linewidth='0.5')
ax[0].set_ylim([0.92, 1.0])
ax[0].set_yticks([0.95, 1.0])
ax[0].set_xlabel('Training iterations')

ax[0].set_ylabel('Train accuracy')
ax[0] = simpleaxis(ax[0])


ax[1].plot( plot_length * RECORD_ACC_FREQ, np.mean(sup_learning_results_rand_init['test_results'][selected_density_list[0]], axis = 0), label = 'Rand. init', color = '#998ec3', linestyle = '-',lw = 1.5)
ax[1].plot( plot_length * RECORD_ACC_FREQ, np.mean(sup_learning_results_ntt_init['test_results'][selected_density_list[0]], axis = 0), ** gen_kwargs_ntt_student_plot)
ax[1].set_ylabel('Test accuracy')

ax[1].set_ylim([0.92, 1.0])
ax[1].set_yticks([0.95, 1.0])

ax[1].set_xlabel('Training iterations')

ax[1].grid(linestyle='-', linewidth='0.5')
ax[1] = simpleaxis(ax[1])

fig.set_size_inches(two_fig_size['width'] ** 1.5, two_fig_size['height'] ** 3 )
from Core.dgllim import dGLLiM
from tools.context import HapkeGonio1468_50
from tools.experience import SecondLearning

exp = SecondLearning(HapkeGonio1468_50, partiel=(0, 1, 2, 3))
exp.load_data(regenere_data=True, with_noise=50, N=10000, method="sobol")
# X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
dGLLiM.dF_hook = exp.context.dF
# X, _ = exp.add_data_training(None,adding_method="sample_perY:9000",only_added=False,Nadd=132845)
gllim = exp.load_model(1000, mode="r", track_theta=False, init_local=500,
                       sigma_type="iso", gamma_type="full", gllim_cls=dGLLiM)

# exp.extend_training_parallel(gllim,Y=exp.context.get_observations(),X=None,threshold=None,nb_per_X=20000,clusters_per_X=12)
# Y ,X , gllims = exp.load_second_learning(100,None,20000,12,withX=False)


exp.mesures.plot_mesures(gllim)
#
X0 = exp.Xtest[20]
# X0 = np.array([0.83,17,0.58,0.2])
Y0 = exp.context.F(X0[None, :])
exp.mesures.plot_conditionnal_density(gllim, Y0, X0, sub_densities=4, with_modal=True, colorplot=True)

#
# #



#
#
#

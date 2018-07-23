"""Implements severals tools to manipulate GLLiM algorithm"""
import datetime
import logging

import h5py
import json
import os

import scipy.io
import numpy as np

class Archive():
    """Helps with saving and loading results"""

    BASE_PATH = "/scratch/WORK/"
    """Context folder path"""

    PATH_MESURES = os.path.join(BASE_PATH,"_MESURES")

    @classmethod
    def save_mesures(cls,mesures,categorie):
        savepath = os.path.join(cls.PATH_MESURES,categorie+"_mes.json")
        with open(savepath,"w",encoding="utf8") as f:
            json.dump(mesures, f, indent=2)
        logging.info(f"\tMeasures saved in {savepath}")

    @classmethod
    def load_mesures(cls,categorie):
        savepath = os.path.join(cls.PATH_MESURES,categorie+"_mes.json")
        if not os.path.exists(savepath):
            logging.warning("Aucun fichier de mesure pour la catégorie {}.".format(categorie))
            return []
        with open(savepath,encoding="utf8") as f:
            m = json.load(f)
        logging.info(f"\tMeasures loaded from {savepath}")
        return m

    @classmethod
    def save_evolution_clusters(cls, rnks, Xdensitys):
        """Uses HDF5 format to save clusters evolution"""
        path = os.path.join(cls.BASE_PATH, "evo_cluster.mat")
        with h5py.File(path, "w") as f:
            f.create_dataset("rnks", data=rnks)
            f.create_dataset("Xdensitys", data=Xdensitys)
        logging.info(f"Clusters evolution saved in {path}")

    @classmethod
    def load_evolution_clusters(cls):
        """Inverse function of save_evolution_clusters"""
        path = os.path.join(cls.BASE_PATH, "evo_cluster.mat")
        with h5py.File(path) as f:
            rnks = np.array(f["rnks"])
            Xdensitys = np.array(f["Xdensitys"])
        logging.info(f"Clusters evolution loaded from {path}")
        return rnks, Xdensitys

    @classmethod
    def save_evolution_1D(cls, cks, ckSs, Aks, bks):
        """Uses matlab format to save 1D learning evolution"""
        path = os.path.join(cls.BASE_PATH, "evo_1D.mat")
        scipy.io.savemat(path, {"cks": cks, "ckSs": ckSs, "Aks": Aks, "bks": bks})
        logging.info(f"1D evolution saved in {path}")

    @classmethod
    def load_evolution_1D(cls):
        """Inverse function of save_evolution_1D"""
        path = os.path.join(cls.BASE_PATH, "evo_1D.mat")
        d = scipy.io.loadmat(path)
        logging.info(f"1D evolution loaded from {path}")
        return d["cks"], d["ckSs"], d["Aks"], d["bks"]


    def __init__(self,experience):
        self.experience = experience
        self.verbose = experience.verbose
        name_context = experience.context.__class__.__name__
        self.directory = os.path.join(self.BASE_PATH,name_context)

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
            os.mkdir(os.path.join(self.directory,"data"))
            os.mkdir(os.path.join(self.directory,"model"))
            os.mkdir(os.path.join(self.directory,"figures"))
            os.mkdir(os.path.join(self.directory,"second_models"))

    def _data_name(self):
        exp = self.experience
        n = exp.with_noise and "noisy:" + str(exp.with_noise) or "notNoisy"
        p = exp.partiel and "partiel:" + str(exp.partiel) or "total"
        added = exp.adding_method and "added:{}".format(exp.adding_method) or "notAdded"
        s =  "{}_meth:{}_{}_{}_N:{}_Nadd:{}".format(added,exp.method,n,p,exp.N,exp.Nadd)
        return s

    def _suffixe(self):
        exp = self.experience
        c = exp.gllim_cls.__name__
        onlyadded = exp.only_added and "onlyAdded" or ""
        file = "{}_{}_K:{}_Lw:{}_multiinit:{}_initlocal:{}_Sc:{}_Gc:{}".format(onlyadded,c,exp.K,exp.Lw,
                                                                               exp.multi_init,exp.init_local, exp.sigma_type,exp.gamma_type)
        return file


    def get_path(self,mode,filecategorie=None,with_track=False,fig_extension=".png",filename=None):
        basedir = os.path.join(self.directory,mode)
        dataname = self._data_name()
        subdir = os.path.join(basedir,dataname)

        if mode == "data":
            return subdir

        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        filename = filename or self._suffixe()

        if filecategorie:
            filename = filecategorie + "_" + filename

        if with_track:
            filename += "__track"
        if mode == "figures":
            filename += fig_extension

        return os.path.join(subdir, filename)


    def load_data(self):
        path = self.get_path("data")
        d = scipy.io.loadmat(path)
        X, Y = d["X"], d["Y"]
        m = self.experience.only_added and "Additionnal data" or "Data"
        logging.info("\t{} loaded from {}".format(m, path))
        return X,Y


    def save_data(self,X,Y):
        path = self.get_path("data") + ".mat"
        scipy.io.savemat(path,{"X":X,"Y":Y})
        m = self.experience.only_added and "Additionnal data" or "Data"
        logging.info("\t{} saved in {}".format(m, path))


    def _save_data(self,data,savepath):
        with open(savepath,'w',encoding='utf8') as f:
            json.dump(data,f,indent=2)
        logging.info(f"\tModel parameters saved in {savepath}")

    def save_gllim(self,gllim,track_theta,training_time=None):
        """Saves current gllim parameters, in json format, to avoid model fitting computations.
        Warning : The shape of Sigma depends on sigma_type.
        If track_theta, assumes gllim saved theta during iterations, and saves the track.
        Store training_time (in sec) if given.
        """
        savepath = self.get_path("model")
        dic = dict(gllim.theta,datetime=datetime.datetime.now().strftime("%c"),
                   training_time=training_time)
        self._save_data(dic,savepath)
        if track_theta:
            filename = self.get_path("model",with_track=True)
            d = {"thetas": gllim.track, "LLs": gllim.loglikelihoods}
            with open(filename, 'w', encoding='utf8') as f:
                json.dump(d, f, indent=2)
            logging.info(f"\tModel parameters history save in {filename}")

    def load_gllim(self):
        """Load parameters of the model and returns it as dict"""
        filename = self.get_path("model")
        with open(filename,encoding='utf8') as f:
            d = json.load(f)
        logging.info("f\tModel parameters loaded from {filename}")
        return d

    def load_tracked_thetas(self):
        filename = self.get_path("model", with_track=True)
        with open(filename,encoding='utf8') as f:
            d = json.load(f)
        logging.info(f"\tParameters history loaded from {filename}")
        return d["thetas"], d["LLs"]

    def save_second_learned(self,gllims,Y,X):
        path = self.get_path("second_models")
        dt = datetime.datetime.now().strftime("%c")
        for i, g in enumerate(gllims):
            savepath = path + str(i)
            self._save_data(dict(g.theta,datetime=dt),savepath)
        d = {"Yadd": Y}
        if X is not None:
            assert len(X) == len(Y)
            d["Xadd"] = X
        scipy.io.savemat(path + "add", d)
        logging.info(f"\tParameters and additional data saved in {path}")

    def load_second_learned(self,withX):
        path = self.get_path("second_models")
        thetas = []
        exp = self.experience
        for i in range(exp.Nadd):
            filename = path + str(i)
            with open(filename,encoding='utf8') as f:
                d = json.load(f)
            thetas.append(d)

        d = scipy.io.loadmat(path  +  "add")
        Y = d["Yadd"]
        X = None
        if withX:
            X = d["Xadd"]
        logging.info(f"\tModel parameters and additional data loaded from {path}")
        return Y, X, thetas


    def save_resultat(self,dic):
        path  = os.path.join(self.directory,self._suffixe())
        scipy.io.savemat(path,dic)
        logging.info(f"Results saved in {path}")

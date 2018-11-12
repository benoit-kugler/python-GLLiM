"""Implements severals tools to manipulate GLLiM algorithm"""
import datetime
import logging
import zlib

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
        logging.debug(f"\tMeasures saved in {savepath}")

    @classmethod
    def load_mesures(cls,categorie):
        savepath = os.path.join(cls.PATH_MESURES,categorie+"_mes.json")
        if not os.path.exists(savepath):
            logging.warning("Aucun fichier de mesure pour la catégorie {}.".format(categorie))
            return []
        with open(savepath,encoding="utf8") as f:
            m = json.load(f)
        logging.debug(f"\tMeasures loaded from {savepath}")
        return m

    @classmethod
    def save_evolution_clusters(cls, rnks, Xdensitys):
        """Uses HDF5 format to save clusters evolution"""
        path = os.path.join(cls.BASE_PATH, "evo_cluster.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("rnks", data=rnks)
            f.create_dataset("Xdensitys", data=Xdensitys)
        logging.debug(f"Clusters evolution saved in {path}")

    @classmethod
    def load_evolution_clusters(cls):
        """Inverse function of save_evolution_clusters"""
        path = os.path.join(cls.BASE_PATH, "evo_cluster.hdf5")
        with h5py.File(path) as f:
            rnks = np.array(f["rnks"])
            Xdensitys = np.array(f["Xdensitys"])
        logging.debug(f"Clusters evolution loaded from {path}")
        return rnks, Xdensitys

    @classmethod
    def save_evolution_1D(cls, cks, ckSs, Aks, bks):
        """Uses matlab format to save 1D learning evolution"""
        path = os.path.join(cls.BASE_PATH, "evo_1D.mat")
        scipy.io.savemat(path, {"cks": cks, "ckSs": ckSs, "Aks": Aks, "bks": bks})
        logging.debug(f"1D evolution saved in {path}")

    @classmethod
    def load_evolution_1D(cls):
        """Inverse function of save_evolution_1D"""
        path = os.path.join(cls.BASE_PATH, "evo_1D.mat")
        d = scipy.io.loadmat(path)
        logging.debug(f"1D evolution loaded from {path}")
        return d["cks"], d["ckSs"], d["Aks"], d["bks"]

    @classmethod
    def save_evoKN(cls, dic):
        filename = "plusieursKN.mat"
        filename = os.path.join(Archive.BASE_PATH, filename)
        scipy.io.savemat(filename, dic)
        logging.debug(f"KN evolution measures saved in {filename}")

    @classmethod
    def load_evoKN(cls):
        filename = "plusieursKN.mat"
        filename = os.path.join(Archive.BASE_PATH, filename)
        return scipy.io.loadmat(filename)


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
        noise_tag = zlib.adler32(exp.with_noise.encode('utf8'))
        n = exp.with_noise and "noisy:" + str(noise_tag) or "notNoisy"
        p = exp.partiel and "partiel:" + str(exp.partiel) or "total"
        s = "meth:{}_{}_{}_N:{}".format(exp.generation_method, n, p, exp.N)
        return s

    def _suffixe(self):
        exp = self.experience
        c = exp.gllim_cls.__name__
        file = "{}_K:{}_Lw:{}_multiinit:{}_initlocal:{}_Sc:{}_Gc:{}".format(c, exp.K, exp.Lw,
                                                                            exp.multi_init, exp.init_local, exp.sigma_type, exp.gamma_type)
        return file

    def make_dir_if_need(self, subdir):
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        return subdir

    def _base_dir(self, mode):
        return os.path.join(self.directory, mode)

    def get_path(self,mode,filecategorie=None,with_track=False,fig_extension=".png",filename=None):
        """If filename is not None, use it instead of suffixe."""
        basedir = self._base_dir(mode)
        dataname = self._data_name()
        subdir = os.path.join(basedir,dataname)

        if mode == "data":
            self.make_dir_if_need(basedir)
            return subdir  # it's actuallaly the file path

        if mode == "second_models":
            subdir = os.path.join(subdir, str(self.experience.number))

        self.make_dir_if_need(subdir)

        if filename:
            return os.path.join(subdir, filename)

        filename = self._suffixe()

        if mode == "second_models":
            filename += f"sl:{self.experience.second_learning}"

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
        logging.debug("\tData loaded from {}".format(path))
        return X,Y


    def save_data(self,X,Y):
        path = self.get_path("data") + ".mat"
        scipy.io.savemat(path,{"X":X,"Y":Y})
        logging.debug("\tData saved in {}".format(path))


    def _save_data(self,data,savepath):
        with open(savepath,'w',encoding='utf8') as f:
            json.dump(data,f,indent=2)

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
        logging.debug(f"\tModel parameters saved in {savepath}")
        if track_theta:
            filename = self.get_path("model",with_track=True)
            d = {"thetas": gllim.track, "LLs": gllim.loglikelihoods}
            self._save_data(d, filename)
            logging.debug(f"\tModel parameters history save in {filename}")

    def load_gllim(self):
        """Load parameters of the model and returns it as dict"""
        filename = self.get_path("model")
        with open(filename,encoding='utf8') as f:
            d = json.load(f)
        logging.debug(f"\tModel parameters loaded from {filename}")
        return d

    def load_tracked_thetas(self):
        filename = self.get_path("model", with_track=True)
        with open(filename,encoding='utf8') as f:
            d = json.load(f)
        logging.debug(f"\tParameters history loaded from {filename}")
        return d["thetas"], d["LLs"]

    def save_data_second_learned(self, Y, X):
        path = self.get_path("second_models")
        d = {"Yadd": Y, "Nadd": len(Y)}
        if X is not None:
            assert len(X) == len(Y)
            d["Xadd"] = X
        scipy.io.savemat(path, d)
        logging.debug(f"\tAdditional data saved in {path}")

    def get_path_second_learned_models(self, N):
        path = self.get_path("second_models")
        return [path + "-" + str(i) for i in range(N)]

    def load_second_learned(self,withX):
        path = self.get_path("second_models")
        data = scipy.io.loadmat(path)
        thetas = []
        for i in range(data["Nadd"][0, 0]):
            filename = path + "-" + str(i)
            with open(filename,encoding='utf8') as f:
                d = json.load(f)
            thetas.append(d)

        Y = data["Yadd"]
        X = data["Xadd"] if withX else None
        logging.debug(f"\tModel parameters and additional data loaded from {path}")
        return Y, X, thetas


    def save_resultat(self,dic):
        path = os.path.join(self.directory, "RES_" + self._suffixe())
        scipy.io.savemat(path,dic)
        logging.debug(f"Results saved in {path}")

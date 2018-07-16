import csv
import os
import subprocess

import numpy as np
import scipy.io

from tools.context import abstractHapkeModel


class RtlsH2OContext(abstractHapkeModel):

    BASE_PATH = "../DATA/RTLS"
    RTLS_FILE = "spect_RTLS_weights_H2O_ice_17669.txt"

    _theta0 = 61.6
    _theta = [70.37, 62.98, 56.07, 49.68, 43.09, 3.94, 46.82, 52.34, 58.34, 64.80, 71.65]
    _phi = [48.529254,48.534201,48.556448,48.602086,48.596889,48.719586,131.41790,131.41432,131.41587,131.41992,131.38141]


    def _load_context_data(self):
        D = len(self._theta)
        self.geometries = np.array([[self._theta0] * D, self._theta, self._phi])[:,None,:]
        with open(os.path.join(self.BASE_PATH,self.RTLS_FILE), newline="") as f:
            r = csv.reader(f, delimiter=' ')
            r.__next__()
            poids = np.array(list(r),dtype=float)

        self.wave_lengths = poids[:,0]
        self.poids = poids

    def compute_observations(self):
        os.chdir(self.BASE_PATH)
        path = "__tmp.mat"
        path_out = self.__class__.__name__ + "_fGV.mat"
        scipy.io.savemat(path, {"geometries": self.geometries[:,0,:]})
        script = """ load('{path}'); 
        [f_V,f_G] = RTLS_fast(geometries(1,:),geometries(2,:),geometries(3,:)); 
        save('-mat-binary','{path_out}','f_G','f_V');
        """.format(path=path,path_out=path_out)
        subprocess.run(['octave-cli',"--eval",script],check=True)
        d = scipy.io.loadmat(path_out)
        f_G , f_V = d["f_G"] , d["f_V"]

        reff = self.poids[:,1:2] + self.poids[:,2:3] * f_V + self.poids[:,3:4] * f_G
        path_reff = self.__class__.__name__ + "_reff"
        scipy.io.savemat(path_reff + ".mat",{"REFF":reff})
        np.savetxt(path_reff + ".txt",reff)
        os.chdir("../../DEV")
        print("Observations calculées à partir du modèle RTLS.")

    def get_observations(self):
        path = os.path.join(self.BASE_PATH,self.__class__.__name__ + "_reff.mat")
        return scipy.io.loadmat(path)["REFF"]


class RtlsCO2Context(RtlsH2OContext):

    RTLS_FILE = "spect_RTLS_weights_CO2_ice_63B0.txt"

    _theta0 = 66
    _theta = [67.674637,60.986866,55.314976,48.840813,43.664577,11.408441,42.847710,49.346943,54.778233,61.158634,67.509018]
    _phi = [131.28603,130.94880,130.52049,130.05166,129.53958,33.219429,44.183444,44.759813,45.196033,45.619191,45.980522]


if __name__ == '__main__':
    h = RtlsH2OContext(None)
    h.compute_observations()
    h = RtlsCO2Context(None)
    h.compute_observations()
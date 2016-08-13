#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Reimplementation of aMD reweighting protocol """

import sys
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import factorial

__author__ = "Jérôme Eberhardt, Roland H Stote, and Annick Dejaegere"
__copyright__ = "Copyright 2016, Jérôme Eberhardt"
__credits__ = ["Jérôme Eberhardt", "Roland H Stote", "Annick Dejaegere"]

__lience__ = "MIT"
__maintainer__ = "Jérôme Eberhardt"
__email__ = "qksoneo@gmail.com"

class Reweight:

    def __init__(self, coord_file, weight_file=None):

        # Read coordinate and weight file
        self.frame_idx, self.coordinates = self.read_coord_file(coord_file)
        self.dv, self.timestep = self.read_dv_file(weight_file)

        assert self.coordinates.shape[0] == self.dv.shape[0], "Size of coordinates and dV are not equal !"
        print "Number of coordinates: %s" % self.coordinates.shape[0]

    def read_coord_file(self, coord_file, center=True):
        data = np.loadtxt(coord_file, dtype=np.float32())

        if data.shape[1] == 2:
            coord = data
            frame_idx = np.arange(0, data.shape[0])
        elif data.shape[1] == 3:
            coord = data[:,1:]
            frame_idx = data[:,0]
        else:
            print 'Error: Cannot read coordinates file!'
            sys.exit(1)

        if center:
            # Get the middle
            x_center = np.min(coord[:,0]) + (np.max(coord[:,0]) - np.min(coord[:,0])) / 2.
            y_center = np.min(coord[:,1]) + (np.max(coord[:,1]) - np.min(coord[:,1])) / 2.
            # Center data
            coord[:,0] -= x_center
            coord[:,1] -= y_center

        """
        http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
        Please note that the histogram does not follow the Cartesian convention where x values are 
        on the abscissa and y values on the ordinate axis. Rather, x is histogrammed along the 
        first dimension of the array (vertical), and y along the second dimension of the array (horizontal).
        """
        # Inverse X and Y coordinates
        return frame_idx, np.fliplr(coord)

    def read_dv_file(self, weight_file):
        data = np.loadtxt(weight_file, usecols=(0,1), dtype=np.float32())
        time, dv = data[:,0], data[:,1]
        return dv, time

    def get_dv_statistics(self):
        dV_avg = np.mean(self.dv)
        dV_std = np.std(self.dv)
        dV_range = np.max(self.dv) - np.min(self.dv)

        print 'dV all:  avg = %8.3f, std = %8.3f range = %8.3f' % (dV_avg, dV_std, dV_range)

    def assignbins2D(self, coordinates, bin_size):

        x_min, x_max = np.min(coordinates[:,0]), np.max(coordinates[:,0])
        y_min, y_max = np.min(coordinates[:,1]), np.max(coordinates[:,1])

        x_length = (x_max - x_min)
        y_length = (y_max - y_min)

        x_center = x_min + (x_length/2)
        y_center = y_min + (y_length/2)

        if x_length > y_length:
            x_limit = np.array([x_center-(x_length/2)-0.5, x_center+(x_length/2)+0.5])
            y_limit = np.array([y_center-(x_length/2)-0.5, y_center+(x_length/2)+0.5])
        else:
            x_limit = np.array([x_center-(y_length/2)-0.5, x_center+(y_length/2)+0.5])
            y_limit = np.array([y_center-(y_length/2)-0.5, y_center+(y_length/2)+0.5])

        x_bins = np.arange(float(x_limit[0]), (float(x_limit[1]) + bin_size), bin_size)
        y_bins = np.arange(float(y_limit[0]), (float(y_limit[1]) + bin_size), bin_size)

        return x_bins, y_bins

    def anharmonicity(self, data):
        """
        Compute anharmonicity
        """
        var = np.var(data) # Get data variance
        hist, edges = np.histogram(data, bins=50, normed=True)
        hist += 1e-18  # To avoid division by error in log
        dx = edges[1] - edges[0]

        S_dV = -1. * np.trapz(hist * np.log(hist), dx=dx)
        S_max = 0.5 * np.log(2. * np.pi * np.exp(1.0) * var + 1e-18)
        alpha = S_max - S_dV

        if np.isinf(alpha):
            return 100
        else:
            return alpha

    def get_position_minima(self, hist):
        """
        http://stackoverflow.com/questions/3584243/python-get-the-position-of-the-biggest-item-in-a-numpy-array
        """
        # Get position of the minima
        return np.unravel_index(np.nanargmin(hist), hist.shape)

    def potential_mean_force(self, bin_size, cutoff=10, temperature=298.15):

        beta = 1.0 / (0.001987 * np.float32(temperature))

        # Get coordinates of the histogram
        edges_x, edges_y = self.assignbins2D(self.coordinates, bin_size)

        # Compute a simple histogram
        histogram = np.histogram2d(x=self.coordinates[:,0], y=self.coordinates[:,1], bins=(edges_x, edges_y))[0]
        # Select only the bins with mor than <cutoff> structures
        histogram[histogram <= cutoff] = 0.

        # Get the total structures
        total_structures  = self.coordinates.shape[0]

        # Compute pmf
        pmf = histogram / np.float32(total_structures)
        pmf = -(1./beta) * np.log(pmf + 1E-18) # Avoid 0 in log function
        pmf[pmf == np.max(pmf)] = np.nan
        pmf = pmf - np.nanmin(pmf)

        # On recupère l'energie minimale
        pmf_min = -np.nanmax(pmf)
        # On récupère la position de ce minima
        indice = self.get_position_minima(pmf)

        print "PMF Min = %8.3f (kcal/mol) (position: %6.3f, %6.3f)" % (pmf_min, edges_x[indice[1]], edges_y[indice[0]])

        return {'pmf': pmf, 'histogram': histogram, 'edges': [edges_x, edges_y]}

    def maclaurin_expansion(self, bin_size, cutoff=10, temperature=298.15, ml_order=10):

        beta = 1.0 / (0.001987 * np.float32(temperature))

        # Get coordinates of the histogram
        edges_x, edges_y = self.assignbins2D(self.coordinates, bin_size)

        beta_dv = self.dv * beta
        weights = np.zeros(self.dv.shape[0], dtype=np.float32)

        for n in xrange(ml_order + 1):
            weights += (np.power(beta_dv, n) / float(factorial(n)))

        # Compute histogram with the weights
        pmf = np.histogram2d(x=self.coordinates[:,0], y=self.coordinates[:,1], bins=(edges_x, edges_y), weights=weights)[0]

        # Compute a simple histogram
        histogram = np.histogram2d(x=self.coordinates[:,0], y=self.coordinates[:,1], bins=(edges_x, edges_y))[0]
        # Select only the bins with mor than <cutoff> structures
        pmf[histogram <= cutoff] = 0.

        # Conversion en kcal/mol
        pmf += 1e-18  # To avoid division by zero in log function
        pmf = (0.001987 * temperature) * np.log(pmf) # Convert to free energy in Kcal/mol
        pmf = np.max(pmf) - pmf  # Zero value to lowest energy state
        pmf[pmf == np.max(pmf)] = np.nan # Inf/max energy to np.nan

        # On recupère l'energie minimale
        pmf_min = -np.nanmax(pmf)
        # On récupère la position de ce minima
        indice = self.get_position_minima(pmf)

        print "PMF Min = %8.3f (kcal/mol) (position: %6.3f, %6.3f)" % (pmf_min, edges_x[indice[1]], edges_y[indice[0]])

        return {'pmf': pmf, 'histogram': histogram, 'edges': [edges_x, edges_y]}

    def cumulant_expansion(self, bin_size, cutoff=10, temperature=298.15):

        # Definition de la constante beta
        beta = 1.0 / (0.001987 * np.float32(temperature))

        # On definit les bornes afin que le paysage energetique soit au centre
        edges_x, edges_y = self.assignbins2D(self.coordinates, bin_size)

        ## Compute a simple histogram
        histogram = np.histogram2d(x=self.coordinates[:,0], y=self.coordinates[:,1], bins=(edges_x, edges_y))[0]

        # Taille de l'histogramme
        nbinsX = histogram.shape[0]
        nbinsY = histogram.shape[1]

        # Matrice pour stocker tous les dV/structure par bins
        dV_mat = np.zeros(shape=(nbinsX, nbinsY, int(np.max(histogram)+5)), dtype=np.float32())
        # Histogramme pour savoir ou on rajoute +1 a chaque dV/structure
        histo_tmp = np.zeros(shape=(nbinsX, nbinsY), dtype=np.int32())

        total_structures = self.coordinates.shape[0]

        # On range chaque dV dans chaque bins
        for i in xrange(total_structures):
            x = int((self.coordinates[i,0] - edges_x[0]) / bin_size)
            y = int((self.coordinates[i,1] - edges_y[0]) / bin_size)

            dV_mat[x, y, histo_tmp[x,y]] = self.dv[i]
            histo_tmp[x, y] += 1

        # On rend la memoire
        del histo_tmp

        # Matrice pour stocker la moyenne, l'ecart-type de dV
        dV_avg = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())
        dV_std = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())

        # Matrice pour stocker la PMF (potential mean force)
        pmf = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())

        # Matrice pour stocker la reponderation 'cumulant expansion 2nd order'
        c1 = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())
        c2 = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())

        dV_avg_all = np.mean(self.dv)
        dV_std_all = np.std(self.dv)

        diff_tol_avg = 10
        diff_tol_std = 1

        # Pour chaque case de l'histogramme
        for x in xrange(nbinsX):
            for y in xrange(nbinsY):

                # Si le nombre de structure est superieur a la limite basse
                if histogram[x, y] >= cutoff:
                    # On recupere tous les dV
                    # Et on enleve tous les zeros qui trainent a la fin

                    tmp = np.trim_zeros(dV_mat[x, y], trim='b')

                    # On calcule la moyenne, l'ecart-type
                    dV_avg[x, y] = np.mean(tmp)
                    dV_std[x, y] = np.std(tmp)

                    pmf[x, y] = -(1./beta) * np.log(histogram[x, y])

                    #print x, y, np.abs(dV_avg[x, y] - dV_avg_all), np.abs(dV_std[x, y] - dV_std_all)

                    if np.abs(dV_avg[x, y] - dV_avg_all) > diff_tol_avg \
                       or np.abs(dV_std[x, y] - dV_std_all) > diff_tol_std :

                        dV_avg[x, y] = 0
                        dV_std[x, y] = 0

        c1 = (beta * dV_avg)
        c2 = (((1./2.) * (beta**2)) * (dV_std**2))

        pmf_c2 = pmf - ((1./beta) * c1) - ((1./beta) * c2)
        pmf_c2 = pmf - np.min(pmf)
        pmf_c2[pmf_c2 == np.max(pmf_c2)] = np.nan

        # On recupère l'energie minimale
        pmf_min = -np.nanmax(pmf_c2)
        # On récupère la position de ce minima
        indice = self.get_position_minima(pmf_c2)

        print "PMF Min = %8.3f (kcal/mol) (position: %6.3f, %6.3f)" % (pmf_min, edges_x[indice[1]], edges_y[indice[0]])

        return {'pmf': pmf_c2, 'histogram': histogram, 'edges': [edges_x, edges_y]}

    def run(self, method, bin_size, cutoff=10, temperature=298.15, ml_order=10):

        # Print basic dV statistics
        self.get_dv_statistics()

        if method == 'pmf':
            self.result = self.potential_mean_force(bin_size, cutoff, temperature)
        elif method == 'maclaurin':
            self.result = self.maclaurin_expansion(bin_size, cutoff, temperature, ml_order)
        elif method == 'cumulant':
            self.result = self.cumulant_expansion(bin_size, cutoff, temperature)

        # Dirty function to obtain all other histograms (dV_mean, dV_std, ANH)
        self.histograms(bin_size, cutoff, self.result['edges'])
        # Get energy for each structure
        self.result['pmf_s'] = self.info_per_structure(self.result['pmf'], bin_size, self.result['edges'])

    def histograms(self, bin_size, cutoff, edges):

        histogram = self.result['histogram']

        # Get size of histogram
        nbinsX = histogram.shape[0]
        nbinsY = histogram.shape[1]

        dv_histogram = np.zeros(shape=(nbinsX, nbinsY, int(np.max(histogram))), dtype=np.float32())
        # Histogramme pour savoir ou on rajoute +1 a chaque dV/structure
        histo_tmp = np.zeros(shape=(nbinsX, nbinsY), dtype=np.int32())

        # On range chaque dV et structure dans chaque bins
        for i in xrange(self.coordinates.shape[0]):
            x = np.int((self.coordinates[i, 0] - edges[0][0]) / bin_size)
            y = np.int((self.coordinates[i, 1] - edges[1][0]) / bin_size)

            dv_histogram[x, y, histo_tmp[x,y]] = self.dv[i]

            histo_tmp[x, y] += 1

        # On rend la memoire
        del histo_tmp

        # Matrice pour stocker la moyenne, l'ecart-type et l'anharmonicite de dV
        dv_avg = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())
        dv_std = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())
        dv_anh = np.zeros(shape=(nbinsX, nbinsY), dtype=np.float32())

        # Pour chaque case de l'histogramme
        for x in xrange(nbinsX):
            for y in xrange(nbinsY):

                # Si on a au moins une structure dans le bin
                if histogram[x, y] > cutoff:
                    # On recupere tous les dV
                    # Et on enleve tous les zeros qui trainent a la fin
                    tmp = np.trim_zeros(dv_histogram[x, y], trim='b')

                    dv_avg[x, y] = np.mean(tmp)
                    dv_std[x, y] = np.std(tmp)
                    dv_anh[x, y] = self.anharmonicity(tmp)

        dv_avg[dv_avg == 0.] = np.nan
        dv_std[dv_std == 0.] = np.nan
        dv_anh[dv_anh == 0.] = np.nan

        # Store result
        self.result['dv_avg'] = dv_avg
        self.result['dv_std'] = dv_std
        self.result['dv_anh'] = dv_anh

    def info_per_structure(self, hist, bin_size, edges):
        array = np.zeros(self.coordinates.shape[0], dtype=np.float32)

        # On recupere l'energie de chaque point
        for i in xrange(self.coordinates.shape[0]):
            x = int((self.coordinates[i, 0] - edges[0][0]) / bin_size)
            y = int((self.coordinates[i, 1] - edges[1][0]) / bin_size)

            array[i] = hist[x, y]

        return array

    def save(self):
        with h5py.File('reweight_data.hdf5', 'w') as w:
            w['pmf'] = self.result['pmf']
            w['histogram'] = self.result['histogram']
            w['dv_avg'] = self.result['dv_avg']
            w['dv_std'] = self.result['dv_std']
            w['dv_anh'] = self.result['dv_anh']

        with open('reweight.txt', 'w') as w:
            for i in xrange(0, self.coordinates.shape[0]):
                w.write('%010d %10.5f %10.5f %10.5f\n' % (self.frame_idx[i], self.coordinates[i][0], 
                                                          self.coordinates[i][1], self.result['pmf_s'][i]))

def plot_histogram(hist, edges, vmin=None, vmax=None, fig_name='free_energy.png', contour=True, interval=1):

    if vmin is None:
        vmin = np.nanmin(hist)

    if vmax is None:
        vmax = np.nanmax(hist)

    edges_x = edges[0]
    edges_y = edges[1]

    extent = [edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]]  

    # Il y aura un contour par 0.5 kcal/mol
    n = np.int(np.rint(np.abs(vmax / interval)))

    fig, ax = plt.subplots(figsize=(16, 12))

    if contour:
        # Create the contour
        X, Y = np.meshgrid(edges_x[0:-1] + (np.diff(edges_x) / 2.), edges_y[0:-1] + (np.diff(edges_y) / 2.))
        plt.contour(X, Y, hist, n, colors='black', extent=extent)

    # Create picture
    plt.imshow(hist, interpolation='none', origin='low', extent=extent, vmin=vmin, vmax=vmax)

    # Create colorbar
    cbar_ticks = np.linspace(vmin, vmax, n+1)
    cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'), aspect=15)
    cb.set_label('kcal/mol', size=20)
    cb.ax.tick_params(labelsize=15)
    cb.set_clim(vmin, vmax)

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    # We find xlim and ylim ...
    xlim = [np.nan, np.nan]
    ylim = [np.nan, np.nan]

    for i in xrange(0, hist.shape[0]):
        ix = np.where(np.isnan(hist[i,:])==False)[0]
        iy = np.where(np.isnan(hist[:,i])==False)[0]

        if ix.size:
            xlim = [np.nanmin([xlim[0], np.min(ix)]), np.nanmax([xlim[1], np.max(ix)])]

        if iy.size:
            ylim = [np.nanmin([ylim[0], np.min(iy)]), np.nanmax([ylim[1], np.max(iy)])]

    lim = [np.int(np.min([xlim[0], ylim[0]])), np.int(np.max([xlim[-1], ylim[-1]]))]

    plt.xlim(edges_x[lim[0]] - 0.1, edges_x[lim[1]] + 0.1)
    plt.ylim(edges_y[lim[0]] - 0.1, edges_y[lim[1]] + 0.1)

    plt.savefig(fig_name, dpi=300, bbox_inches='tight')

def cmdlineparse():
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument("-c", "--coord", dest="coord_file", required=True,
                        action="store", help="2D input file")
    parser.add_argument("-w", "--weight", dest="weight_file",
                        action="store", default=None, 
                        help="weight file")
    parser.add_argument('-m', "--method", dest="method",
                        action="store", default='maclaurin',
                        choices=['pmf', 'exponential', 'maclaurin',
                        'cumulant'], help="Job type reweighting method")
    parser.add_argument('-b', "--binsize", dest="bin_size",
                        type=float, action="store", default=1,
                        help="Bin size in X dimension")
    parser.add_argument("--cutoff", dest="cutoff",
                        action="store", default=0, type=int,
                        help="histogram cutoff")
    parser.add_argument("-t", "--temperature", dest="temperature",
                        action="store", default=300, type=float,
                        help="Temperature")
    parser.add_argument("--mlorder", dest="ml_order",
                        action="store", default=10, type=int,
                        help="Order of Maclaurin series")
    parser.add_argument("--emax", dest="emax",
                        action="store", default=None, type=float,
                        help="enery maximium on free energy plot")

    args = parser.parse_args()

    return args

def main():

    args = cmdlineparse()

    coord_file = args.coord_file
    weight_file = args.weight_file
    bin_size = args.bin_size
    cutoff = args.cutoff
    temperature = args.temperature
    method = args.method
    ml_order = args.ml_order
    emax = args.emax

    R = Reweight(coord_file, weight_file)
    R.run(method, bin_size, cutoff, temperature, ml_order)
    R.save()

    plot_histogram(R.result['pmf'], R.result['edges'], 0, emax, 'free_energy_%s.png' % method, interval=0.5)
    plot_histogram(R.result['dv_avg'], R.result['edges'], None, None, 'dv_avg.png', False, 20)
    plot_histogram(R.result['dv_std'], R.result['edges'], 0, 40, 'dv_std.png', False, 4)
    plot_histogram(R.result['dv_anh'], R.result['edges'], 0, 0.5, 'dv_anh.png', False, 0.1)

if __name__ == '__main__':
    main()
    sys.exit(0)
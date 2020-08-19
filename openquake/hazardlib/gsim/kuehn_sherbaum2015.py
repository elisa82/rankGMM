# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`Kuehn_Scherbaum2015`.
"""
import numpy as np
from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class Kuehn_Scherbaum2015(GMPE):
    """
    Implements GMPE developed by Kuehn, N.M., Scherbaum, F. Ground-motion prediction model building: a multilevel approach. Bull Earthquake Eng 13, 2481-2491 (2015). https://doi.org/10.1007/s10518-015-9732-3
    """
    #: Supported tectonic region type is 'active shallow crust' because the
    #: equations have been derived from data from Italian database ITACA, as
    #: explained in the 'Introduction'.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types are 
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
    ])

    #: Required site parameter is only Vs30
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude and rake (eq. 1).
    REQUIRES_RUPTURE_PARAMETERS = set(('rake', 'mag'))

    #: Required distance measure is RRup (eq. 1).
    REQUIRES_DISTANCES = set(('rjb', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.

        C = self.COEFFS[imt]

        imean = self._get_mean(C, rup, dists, sites)
        # Convert units to g,
        # but only for PGA and SA (not PGV):
        if imt.name in "SA PGA":
            mean = np.log (np.exp(imean) / g)
        else:
            # PGV:
            mean = np.log(np.exp(imean * 100)) #in cm/s2

        istddevs = self._get_stddevs(C,
                                     stddev_types,
                                     num_sites=len(sites.vs30))

        # Return stddevs in terms of natural log scaling
        stddevs = np.array(istddevs)
        # mean_LogNaturale = np.log((10 ** mean) * 1e-2 / g)
        return mean, stddevs

    def _get_mean(self, C, rup, dists, sites):
        """
        Returns the mean ground motion
        """

        return (C['a0']+self._compute_magnitude(rup.mag, C) +
            self._compute_distance(rup.mag, dists.rjb, C) +
            self._get_site_amplification_term(sites.vs30, C) +
            self._get_mechanism(rup, C))


    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in table 1.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                #Standard deviation values are already squared!
                sigma=np.sqrt(C['tau']+C['phiS2S']+C['phi'])
                stddevs.append(sigma + np.zeros(num_sites))
        return stddevs

    def _compute_distance(self, mag, rval, C):
        """
        """
        rval = np.sqrt(rval ** 2 + C['a5'] ** 2)
        return (C['a3'] + C['a4'] * mag) * np.log(rval)

    def _compute_magnitude(self, mag, C):
        """

        """
        return C['a1'] * mag + C['a2'] * mag**2

    def _get_site_amplification_term(self, vs30, C):
        """
        Returns the site amplification term 
        """
        return C["a6"] * np.log(vs30 / 760.)


    def _get_mechanism(self, rup, C):
        """
        Get fault type dummy variables
        """
        SN, SR = self._get_fault_type_dummy_variables(rup)

        return C['a7'] * SN + C['a8'] * SR

    def _get_fault_type_dummy_variables(self, rup):
        """
        Fault type (Strike-slip, Normal, Thrust/reverse) is
        derived from rake angle.
        Rakes angles within 30 of horizontal are strike-slip,
        angles from 30 to 150 are reverse, and angles from
        -30 to -150 are normal.
        """
        SN, SR = 0, 0
        if np.abs(rup.rake) <= 30.0 or (180.0 - np.abs(rup.rake)) <= 30.0:
            # strike-slip
            SN = 0
            SR = 0
        elif rup.rake > 30.0 and rup.rake < 150.0:
            # reverse
            SR = 1
        else:
            # normal
            SN = 1
        return SN, SR

    #: Coefficients from SA from Table 1
    #: Coefficients from PGA e PGV from Table 5

#    The Supplementary Material of the paper has the coefficients wrongly placed. The code reads the rigth order of the parameters
    COEFFS = CoeffsTable(sa_damping=5, table="""
IMT    a0    a1    a2    a7    a8    a3    a4    a5    a6    tau    phiS2S    phi
pgv    -8.103544286    2.031488955    -0.108740659    -0.170768025    0.141147777    -2.077082672    0.163449001    8.085078598    -0.635447851    0.158781517    0.137778092    0.272712922
pga    -2.154894158    2.015257331    -0.160029327    -0.141680182    0.128978526    -2.819451336    0.227083419    12.11797779    -0.448080444    0.142260742    0.166776247    0.248363823
0.01    -2.106234623    2.008657151    -0.159751696    -0.140936074    0.1294334    -2.822843267    0.227254962    12.11226358    -0.444585874    0.142010937    0.166826147    0.24879545
0.05    -0.23218482    1.613593246    -0.137575732    -0.105409742    0.139739886    -3.055078135    0.258125132    11.03932754    -0.388962065    0.154309478    0.17793925    0.255020037
0.1    0.906269209    1.774799878    -0.160509367    -0.083964755    0.098314869    -3.328138783    0.26895077    14.71816362    -0.31151214    0.170097841    0.197538    0.25443447
0.5    -7.147211495    2.792726481    -0.171648899    -0.173780066    0.189004668    -1.794978122    0.104218336    9.324267262    -0.658612794    0.164871072    0.162009571    0.31060856
1    -10.5030686    3.077918677    -0.168509665    -0.225361874    0.138333876    -1.435228616    0.077208746    6.347658056    -0.828496275    0.176046212    0.214756679    0.311751878
2    -12.2728393    3.001050741    -0.146140177    -0.205427181    0.194300877    -1.396872553    0.088858708    5.679541442    -0.811001574    0.229146612    0.233406294    0.34133091
3    -12.91798681    2.860041548    -0.121632349    -0.168824955    0.173715748    -1.388225987    0.086893945    6.467590661    -0.704824381    0.259219458    0.238382581    0.351472888
4    -13.39138107    2.772477116    -0.101485348    -0.144765801    0.133294332    -1.372714583    0.075991134    7.968254841    -0.645184287    0.276900328    0.209377495    0.32073698
    """)

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
Module exports :class:`Ambraseys1996`.
"""
import numpy as np
from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class Ambraseys1996(GMPE):
    """
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
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GREATER_OF_TWO_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, page 1904
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
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

        imean = self._compute_magnitude(rup, C) + self._compute_distance(rup, dists, C) + self._get_site_amplification(sites, C) 

        istddevs = self._get_stddevs(C,stddev_types,num_sites=len(sites.vs30))
        mean = np.log(10.0 ** (imean)) 
        # Return stddevs in terms of natural log scaling
        stddevs = np.log(10.0 ** np.array(istddevs))
        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            stddevs.append(C['sigma'] + np.zeros(num_sites))
        return stddevs

    def _compute_distance(self, rup, dists, C):
        """
        """
        r = np.sqrt(dists.rjb**2 + C['h0']**2)
        return (C['c4']*np.log10(r))

    def _compute_magnitude(self, rup, C):
        """
        """
        return C['c1']+(C['c2']*rup.mag) 

    def _get_site_amplification(self, sites, C):
        """
        """
        sa, ss = self._get_site_type_dummy_variables(sites)

        return (C['cA'] * sa) + (C['cS'] * ss)

    def _get_site_type_dummy_variables(self, sites):
        """
        """
        sa = np.zeros(len(sites.vs30))
        ss = np.zeros(len(sites.vs30))

        # Class Very soft soil;  Vs30 <= 180 m/s.
        idx = (sites.vs30 < 180.0)
        ss[idx] = 1.0
        # Class Soft soil; 180 m/s < Vs30 <= 360 m/s.
        idx = (sites.vs30 > 180.0) & (sites.vs30 <= 360.0)
        ss[idx] = 1.0
        # Class Stiff soil; 360 m/s < Vs30 <= 750 m/s.
        idx = (sites.vs30 > 360.0) & (sites.vs30 <= 750)
        sa[idx] = 1.0
        # Class A; Vs30 > 750 m/s.
        return sa, ss

    #: Coefficients from SA from Table 1
    #: Coefficients from PGA e PGV from Table 5

    COEFFS = CoeffsTable(sa_damping=5, table="""
    IMT        c1         c2    h0         c4    cA      cS     sigma 
    pga     -1.48    0.266      3.5     -0.922  0.117   0.124   0.25
    0.1     -0.84    0.219	4.5	-0.954	0.078	0.027	0.27
    0.11    -0.86    0.221	4.5	-0.945	0.098	0.036	0.27
    0.12    -0.87    0.231	4.7	-0.96	0.111	0.052	0.27
    0.13    -0.87    0.238	5.3	-0.981	0.131	0.068	0.27
    0.14    -0.94    0.244	4.9	-0.955	0.136	0.077	0.27
    0.15    -0.98    0.247	4.7	-0.938	0.143	0.085	0.27
    0.16    -1.05    0.252	4.4	-0.907	0.152	0.101	0.27
    0.17    -1.08    0.258	4.3	-0.896	0.14	0.102	0.27
    0.18    -1.13    0.268	4	-0.901	0.129	0.107	0.27
    0.19    -1.19    0.278	3.9	-0.907	0.133	0.13	0.28
    0.2    -1.21     0.284	4.2	-0.922	0.135	0.142	0.27
    0.22    -1.28    0.295	4.1	-0.911	0.12	0.143	0.28
    0.24    -1.37    0.308	3.9	-0.916	0.124	0.155	0.28
    0.26    -1.4     0.318	4.3	-0.942	0.134	0.163	0.28
    0.28    -1.46    0.326	4.4	-0.946	0.134	0.158	0.29
    0.3    -1.55     0.338	4.2	-0.933	0.133	0.148	0.3
    0.32    -1.63    0.349	4.2	-0.932	0.125	0.161	0.31
    0.34    -1.65    0.351	4.4	-0.939	0.118	0.163	0.31
    0.36    -1.69    0.354	4.5	-0.936	0.124	0.16	0.31
    0.38    -1.82    0.364	3.9	-0.9	0.132	0.164	0.31
    0.4    -1.94     0.377	3.6	-0.888	0.139	0.172	0.31
    0.42    -1.99    0.384	3.7	-0.897	0.147	0.18	0.32
    0.44    -2.05    0.393	3.9	-0.908	0.153	0.187	0.32
    0.46    -2.11    0.401	3.7	-0.911	0.149	0.191	0.32
    0.48    -2.17    0.41	3.5	-0.92	0.15	0.197	0.32
    0.5    -2.25     0.42	3.3	-0.913	0.147	0.201	0.32
    0.55    -2.38    0.434	3.1	-0.911	0.134	0.203	0.32
    0.6    -2.49     0.438	2.5	-0.881	0.124	0.212	0.32
    0.65    -2.58    0.451	2.8	-0.901	0.122	0.215	0.32
    0.7    -2.67     0.463	3.1	-0.914	0.116	0.214	0.33
    0.75    -2.75    0.477	3.5	-0.942	0.113	0.212	0.32
    0.8    -2.86     0.485	3.7	-0.925	0.127	0.218	0.32
    0.85    -2.93    0.492	3.9	-0.92	0.124	0.218	0.32
    0.9    -3.03     0.502	4	-0.92	0.124	0.225	0.32
    0.95    -3.1     0.503	4	-0.892	0.121	0.217	0.32
    1       -3.17    0.508	4.3	-0.885	0.128	0.219	0.32
    1.1    -3.3	     0.513	4	-0.857	0.123	0.206	0.32
    1.2    -3.38     0.513	3.6	-0.851	0.128	0.214	0.31
    1.3    -3.43     0.514	3.6	-0.848	0.115	0.2	0.31
    1.4    -3.52     0.522	3.4	-0.839	0.109	0.197	0.31
    1.5    -3.61     0.524	3	-0.817	0.109	0.204	0.31
    1.6    -3.68     0.52	2.5	-0.781	0.108	0.206	0.31
    1.7    -3.74     0.517	2.5	-0.759	0.105	0.206	0.31
    1.8    -3.79     0.514	2.4	-0.73	0.104	0.204	0.32
    1.9    -3.8      0.508	2.8	-0.724	0.103	0.194	0.32
    2      -3.79     0.503	3.2	-0.728	0.101	0.182	0.32
    """)

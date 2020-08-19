# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2019 GEM Foundation
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
Module exports :class:'ChenGalasso20192020'
"""
import numpy as np
from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA, RSD595


class ChenGalasso20192020(GMPE):


    #: Supported tectonic region type is 'active shallow crust' because the
    #: equations have been derived from data from Southern Europe, North
    #: Africa, and active areas of the Middle East, as explained in the
    # 'Introduction', page 195.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
    #: this GSIM can calculate. A set should contain classes from module
    #: :mod:`openquake.hazardlib.imt`.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA,
        RSD595
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    #: :attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`, see page 196.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see equation 2, page 199.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Required site parameter is only Vs30 (used to distinguish rock
    #: and stiff and soft soil).
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude and rake (eq. 1, page 199).
    REQUIRES_RUPTURE_PARAMETERS = set(('rake', 'mag'))

    #: Required distance measure is RRup (eq. 1, page 199).
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

        imean = (self._compute_magnitude(rup, C) +
                 self._compute_distance(rup, dists, imt, C) +
                 self._get_site_amplification(sites, imt, C) +
                 self._get_mechanism(sites, rup, imt, C))

        # Convert units to g,
        # but only for PGA and SA (not PGV):
        if imt.name in 'PGA SA':
            mean = np.log((10.0 ** (imean - 2.0)) / g)
        else:
            # PGV:
            mean = np.log(10.0 ** imean)

        # apply scaling factor for SA at 4 s
        if imt.name == 'SA' and imt.period == 4.0:
            mean /= 0.8

        istddevs = self._get_stddevs(
            C, stddev_types, num_sites=len(sites.vs30)
        )

        stddevs = np.log(10 ** np.array(istddevs))

        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in table 1, p. 200.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(C['SigmaTot'] + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(C['Sigma1'] + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(C['tau'] + np.zeros(num_sites))
        return stddevs

    def _compute_magnitude(self, rup, C):
        """
        Compute the first term of the equation described on p. 199:

        ``b1 + b2 * M + b3 * M**2``
        """
        return C['b1'] + (C['b2'] * rup.mag) + (C['b3'] * (rup.mag ** 2))

    def _compute_distance(self, rup, dists, imt, C):
        """
        Compute the second term of the equation described on p. 199:

        ``(b4 + b5 * M) * log(sqrt(Rjb ** 2 + b6 ** 2))``
        """
        return (((C['b4'] + C['b5'] * rup.mag)
                 * np.log10((np.sqrt(dists.rjb ** 2.0 + C['b6'] ** 2.0)))))

    def _get_site_amplification(self, sites, imt, C):
        """
        Compute the third term of the equation described on p. 199:

        ``b7 * Ss + b8 * Sa``
        """
        Ss, Sa = self._get_site_type_dummy_variables(sites)
        return (C['b7'] * Ss) + (C['b8'] * Sa)

    def _get_site_type_dummy_variables(self, sites):
        """
        Get site type dummy variables, ``Ss`` (for soft and stiff soil sites)
        and ``Sa`` (for rock sites).
        """
        Ss = np.zeros((len(sites.vs30),))
        Sa = np.zeros((len(sites.vs30),))
        # Soft soil; Vs30 < 360 m/s. Page 199.
        idxSs = (sites.vs30 < 360.0)
        # Stiff soil Class A; 360 m/s <= Vs30 <= 750 m/s. Page 199.
        idxSa = (sites.vs30 >= 360.0) & (sites.vs30 <= 750.0)
        Ss[idxSs] = 1
        Sa[idxSa] = 1
        return Ss, Sa

    def _get_mechanism(self, sites, rup, imt, C):
        """
        Compute the fourth term of the equation described on p. 199:

        ``b9 * Fn + b10 * Fr``
        """
        Fn, Fr = self._get_fault_type_dummy_variables(sites, rup, imt)
        return (C['b9'] * Fn) + (C['b10'] * Fr)

    def _get_fault_type_dummy_variables(self, sites, rup, imt):
        """
        Same classification of SadighEtAl1997. Akkar and Bommer 2010 is based
        on Akkar and Bommer 2007b; read Strong-Motion Dataset and Record
        Processing on p. 514 (Akkar and Bommer 2007b).
        """

        Fn, Fr = 0, 0
        if rup.rake >= -135 and rup.rake <= -45:
            # normal
            Fn = 1
        elif rup.rake >= 45 and rup.rake <= 135:
            # reverse
            Fr = 1
        return Fn, Fr




    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT      b1        b2       b3       b4       b5        b6        b7       b8       b9       b10      Sigma1   tau      SigmaTot
    pga      3.5240    0.247   -0.020   -3.936    0.3510    12.417    0.228    0.160   -0.060    0.080    0.247    0.370    0.445
    pgv      0.7420    0.188    0.015   -3.089    0.2860    8.5290    0.308    0.144   -0.021    0.037    0.261    0.301    0.398
    rsd595  -0.9180    0.051    0.020    1.399   -0.1170    9.4830    0.093    0.015    0.015   -0.010    0.070    0.181    0.194
    0.010    3.5440    0.244   -0.019   -3.943    0.3520    12.438    0.228    0.160   -0.060    0.080    0.247    0.370    0.445
    0.025    3.7700    0.191   -0.016   -3.995    0.3590    12.220    0.224    0.156   -0.059    0.082    0.248    0.372    0.447
    0.040    4.3400    0.099   -0.014   -4.198    0.3870    11.956    0.212    0.148   -0.054    0.091    0.249    0.385    0.458
    0.050    4.6680    0.048   -0.013   -4.303    0.3990    11.931    0.211    0.155   -0.055    0.096    0.243    0.401    0.469
    0.070    4.9750    0.034   -0.013   -4.401    0.4040    12.404    0.215    0.157   -0.070    0.092    0.237    0.420    0.482
    0.100    4.9410    0.099   -0.015   -4.345    0.3790    14.067    0.212    0.163   -0.088    0.090    0.244    0.430    0.495
    0.150    3.6670    0.445   -0.032   -3.867    0.2900    15.633    0.192    0.160   -0.087    0.099    0.248    0.416    0.484
    0.200    2.5840    0.687   -0.042   -3.454    0.2250    16.378    0.190    0.162   -0.088    0.094    0.251    0.394    0.467
    0.250    1.7100    0.793   -0.039   -3.011    0.1650    15.061    0.195    0.132   -0.076    0.083    0.260    0.366    0.449
    0.300    1.2140    0.808   -0.034   -2.748    0.1370    13.969    0.220    0.140   -0.076    0.059    0.257    0.357    0.440
    0.350    0.8670    0.802   -0.026   -2.538    0.1090    13.637    0.246    0.141   -0.071    0.048    0.255    0.346    0.430
    0.400    0.5730    0.786   -0.019   -2.387    0.0960    12.917    0.260    0.138   -0.068    0.042    0.258    0.337    0.424
    0.450    0.1700    0.834   -0.021   -2.274    0.0900    12.086    0.280    0.146   -0.068    0.031    0.261    0.333    0.423
    0.500   -0.1310    0.861   -0.020   -2.174    0.0810    11.509    0.293    0.149   -0.069    0.025    0.265    0.329    0.423
    0.600   -0.4810    0.838   -0.012   -2.020    0.0680    10.626    0.312    0.151   -0.053    0.015    0.269    0.324    0.421
    0.700   -0.6480    0.764   -0.002   -1.913    0.0660    9.4870    0.319    0.153   -0.038    0.010    0.276    0.316    0.420
    0.750   -0.8440    0.785   -0.002   -1.869    0.0630    9.2920    0.323    0.152   -0.032    0.006    0.278    0.314    0.420
    0.800   -0.8840    0.753    0.002   -1.850    0.0660    8.9900    0.326    0.151   -0.031   -0.001    0.281    0.312    0.420
    0.900   -1.2350    0.798    0.000   -1.786    0.0640    8.2380    0.331    0.145   -0.024   -0.007    0.286    0.310    0.422
    1.000   -1.3290    0.754    0.006   -1.753    0.0680    7.6600    0.343    0.144   -0.013   -0.006    0.291    0.307    0.423
    1.200   -1.6020    0.744    0.008   -1.720    0.0760    7.0430    0.355    0.143   -0.002   -0.017    0.298    0.305    0.426
    1.400   -1.8270    0.726    0.013   -1.670    0.0770    6.3930    0.356    0.137    0.004   -0.020    0.301    0.303    0.427
    1.600   -1.8690    0.684    0.016   -1.714    0.0910    6.0700    0.365    0.133    0.008   -0.022    0.304    0.300    0.427
    1.800   -1.7820    0.580    0.029   -1.692    0.0890    5.9030    0.358    0.129    0.011   -0.026    0.306    0.300    0.428
    2.000   -1.8870    0.572    0.030   -1.689    0.0910    5.8580    0.345    0.127    0.021   -0.020    0.308    0.299    0.430
    2.500   -2.1140    0.596    0.026   -1.785    0.1140    5.8730    0.324    0.115    0.042   -0.014    0.320    0.298    0.437
    3.000   -2.1130    0.531    0.032   -1.822    0.1220    6.1080    0.314    0.112    0.061   -0.017    0.330    0.298    0.445
    3.500   -2.1660    0.500    0.035   -1.843    0.1260    6.2750    0.304    0.101    0.081   -0.010    0.337    0.298    0.450
    4.000   -2.0880    0.438    0.039   -1.914    0.1410    6.3610    0.305    0.101    0.094    0.000    0.340    0.301    0.454
    """)


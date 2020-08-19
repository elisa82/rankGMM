# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2013-2020 GEM Foundation
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
Module exports :class:`Calvi_Andreotti2019`.
"""

import numpy as np
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import TC,TD,alfa,SaC,SdD
import math
from pprint import pprint


def ElyTable(table):
    rows = table.strip().splitlines()
    header = rows.pop(0).split()
    if not header[0].upper() == "EC8":
        raise ValueError('first column in a table must be IMT')
    coeff_names = header[1:]
    result = {}
    for row in rows:
        values = row.strip().split()
        ec8 = values[0]
        coeffs = dict(zip(coeff_names, map(float, values[1:])))
        result[ec8] = coeffs
    return result
    
class Calvi_Andreotti2019(GMPE):
    """
    Implements GMPE developed by Calvi, G. & Andreotti, Guido. (2019). 
    Effects of Local Soil, Magnitude and Distance on Empirical Response Spectra for Design. 
    Journal of Earthquake Engineering. 1-28. 10.1080/13632469.2019.1703847.
    range M [4.5:0.02:7.0]
    range R [5.0:0.5:80.0]
    """
    #: The supported tectonic region type is active shallow crust because
    #: the equations have been developed for "all seismically- active regions
    #: bordering the Mediterranean Sea and extending to the Middle East", see
    #: section 'A New Generation of European Ground-Motion Models', page 4.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: The supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        TC,
        TD,
        alfa,
	SaC,
	SdD
    ])

    #Sd in metri, Sa in g

    #: The supported intensity measure component is 'average horizontal'
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL #It's the envelope Rot100 max

    #: The supported standard deviations are total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: The required site parameter is ec8
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: The required rupture parameter is  magnitude
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: The required distance parameter is 'Joyner-Boore' distance.
    REQUIRES_DISTANCES = set(('rjb', ))

    def _compute_ec8_class(self, vs30):
        """
        Compute ec8 site class
        """
        vs30=vs30[0]
        if(vs30 >= 360) and (vs30 < 800):
            ec8 = 'B'
        elif(vs30 >= 180) and (vs30 < 360):
            ec8 = 'C'
        elif(vs30 >= 800):
            ec8 ='A'
        else:
            raise ValueError(vs30)

        return ec8

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
	"""

        ec8 = self._compute_ec8_class(sites.vs30)

        C_alfa=self.COEFFS_alfa[ec8]
        C_Tc=self.COEFFS_Tc[ec8]
        C_Td=self.COEFFS_Td[ec8]
        C_SaC=self.COEFFS_SaC[ec8]
        C_SdD=self.COEFFS_SdD[ec8]
		
        mean = self._compute_mean(C_alfa, C_Tc, C_Td, C_SaC, C_SdD, rup.mag, dists.rjb, imt)

        stddevs = self._get_stddevs(C_alfa, C_Tc, C_Td, C_SaC, C_SdD, stddev_types, rup.mag, imt, mean, num_sites=len(sites.vs30))

        return mean, stddevs

    def _get_stddevs(self, C_alfa, C_Tc, C_Td, C_SaC, C_SdD, stddev_types, mag, imt, mean, num_sites):
        """
        Return standard deviations 
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                if(isinstance(imt,alfa)):
                    stddevs.append(C_alfa['sig_alfa']+np.zeros(num_sites))
                if(isinstance(imt,TC) or isinstance(imt,TD)):
                    stddevs.append(0.+np.zeros(num_sites))
                if(isinstance(imt,SaC)):
                    stddevs.append((C_SaC['sa1']*mag+C_SaC['sa2'])*mean+np.zeros(num_sites))
                if(isinstance(imt,SdD)):
                    stddevs.append((C_SdD['sd1']*mag+C_SdD['sd2'])*mean+np.zeros(num_sites))

        return stddevs

    def _compute_mean(self, C_alfa, C_Tc, C_Td, C_SaC, C_SdD, mag, dist, imt):
		
        if(isinstance(imt,alfa)):
            mean=C_alfa['alfa0']-C_alfa['alfa1']*(mag-4.5)-C_alfa['alfa2']*(dist-5.0)
        elif(isinstance(imt,TC)):
            mean=(C_Tc['c0']-C_Tc['c1']*(7.0-mag))*(C_Tc['c0']-C_Tc['c2']*(80-dist))/C_Tc['c0'] 
        elif(isinstance(imt,TD)):
            mean=(C_Td['d0']-C_Td['d1']*(7.0-mag))*(C_Td['d0']-C_Td['d2']*(80-dist))/C_Td['d0']
        elif(isinstance(imt,SaC)):
            mean=(C_SaC['a0']+C_SaC['a1']*np.exp(-C_SaC['aM2']*(7.0-mag))*np.cos(math.pi/(2.0*C_SaC['aM3'])*(7.0-mag)))*(C_SaC['a2']+(C_SaC['a0']+C_SaC['a1']-C_SaC['a2'])*np.exp(-C_SaC['aR2']*(dist-5.0))*np.cos(math.pi/(2.0*C_SaC['aR3'])*(dist-5.0)))/(C_SaC['a0']+C_SaC['a1'])            
        elif(isinstance(imt,SdD)):
            mean=(C_SdD['d0']+C_SdD['d1']*np.exp(-C_SdD['dM2']*(7.0-mag))*np.cos(math.pi/(2.0*C_SdD['dM3'])*(7.0-mag)))*(C_SdD['d2']+(C_SdD['d0']+C_SdD['d1']-C_SdD['d2'])*np.exp(-C_SdD['dR2']*(dist-5.0))*np.cos(math.pi/(2.0*C_SdD['dR3'])*(dist-5.0)))/(C_SdD['d0']+C_SdD['d1']) 
        else:
            import ipdb; ipdb.set_trace()
            raise ValueError("Invalid imt for _compute_mean: %s" % repr(imt))

        return mean
    
    COEFFS_Tc = ElyTable(table="""\
    EC8 c0 c1 c2 
    A 0.36 0.0364 0.001 
    B 0.47 0.071 0.0022 
    C 0.42 0.051 0.0002
    """)
    
    COEFFS_SaC = ElyTable(table="""\
    EC8 a0 a1 a2 aM2 aR2 aM3 aR3 sa1 sa2
    A 0.16 1.07 0.0923 0.118 0.0868 2.5 75 -0.39 3.0
    B 0.36 2.17 0.165 0.52 0.077 2.5 75 -0.35 2.70
    C 0.32 2.157 0.30 0.983 0.0576 2.5 75 -0.34 2.60
    """)
    
    COEFFS_Td = ElyTable(table="""\
    EC8 d0 d1 d2 
    A 5.75 1.34 0.047 
    B 5.39 1.514 0.042
    C 5.00 1.435 0.024
    """)
    
    COEFFS_SdD = ElyTable(table="""\
    EC8 d0 d1 d2 dM2 dR2 dM3 dR3 sd1 sd2
    A 0.006 0.2029 0.0355 1.1484 0.021 2.5 75.0 -0.23 2.0
    B 0.019 0.622 0.1044 1.62 0.092 2.5 75.0 -0.41 3.2
    C 0.0434 1.42 0.1934 2.326 0.1439 2.5 75.0 -0.2 2.10
    """)
 
    COEFFS_alfa = ElyTable(table="""\
    EC8 alfa0  alfa1 alfa2 sig_alfa   
    A 2.57 -0.27 -0.0292 1.38
    B 2.57 -0.217 -0.0223 1.42
    C 2.92 -0.0753 -0.0134 1.38
    """)

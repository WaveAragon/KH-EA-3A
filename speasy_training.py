# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:42:09 2026

@author: grego
"""

import speasy as spz
ace_mag = spz.get_data('amda/mes_b_orbmso', "2011-6-17 00:46", "2011-6-17 01:30")
ace_mag.plot()
ace_mag.show()


from datetime import datetime

import matplotlib.pyplot as plt
import speasy as spz

amda_tree = spz.inventories.tree.amda

ace_mag = spz.get_data(amda_tree.Parameters.ACE.MFI.mes_b_orbmso,
                       datetime(2016, 6, 2), datetime(2016, 6, 5))
ace_mag.plot()
plt.tight_layout()
plt.show()



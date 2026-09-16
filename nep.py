from joblib import Parallel, delayed
import numpy as np
import fiber 
import laser
import csv
best_snr_waist = np.array([0.00010612, 0.00033878, 0.00070918, 0.0013551, 0.0031551, 0.00191224, 0.00374286, 0.0037551, 0.00341837, 0.00311837, 0.00316735, 0.0037551, 0.00372449])
# rings = [0, 1, 2, 4, 6, 7]
dat = np.array([])
bundle = fiber.FiberBundle(2e-3, 2.5e-5, 5e-5)
def find_min_pressures(w, i, j, waist):
    bundle = fiber.FiberBundle(2e-3, 2.5e-5, 5e-5)
    bundle.drop_rings((0, i))
    l = laser.Laser(waist = waist, power = 90000, res = 2500)
    copy_l = laser.Laser(xyz = l.dy(j), waist = l.waist, power = 90)
    w.writerows([[l.waist, bundle.diff_power(l), j], [l.waist, bundle.diff_power(copy_l), j]])
    del copy_l, bundle, l
with open("data/20250612/NEP.csv", "w", newline = "") as csvfile:
    writer = csv.writer(csvfile)
    for i, waist in enumerate(best_snr_waist):
        Parallel(n_jobs = -1, backend = "threading")(delayed(find_min_pressures)(writer, i, j, waist) for j in np.logspace(-7, -12, 100))
    csvfile.close()
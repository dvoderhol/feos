"""Where does delta become large? Mode decomposition of delta = -1/2 sum_n (e_n.f)^2/lambda_n, harmonic displacement
u* = H+ f, forces and E0 of the configurations with the most negative delta."""
import sys
import numpy as np
sys.path.insert(0, "/local/home/dvoderhol/feos/mlip-cdft")
from feos_mlip.flex import FlexCoupling, HarmonicFramework
from feos_mlip.vext import KB_EV
kT = KB_EV * 298
H = np.load("hessian_2x1x1.npz")["H"]; fw = HarmonicFramework(H)
lam, vec = np.linalg.eigh(0.5 * (H + H.T)); lam, vec = lam[3:], vec[:, 3:]
cp = FlexCoupling.load("coupling_CO2_MH_20kT.npz")
keep = cp.E0 < 0
f = cp.f[keep].reshape(int(keep.sum()), -1).astype(float); E0 = cp.E0[keep]
proj = f @ vec                                   # (n, 3M-3)
contrib = -0.5 * proj ** 2 / lam                 # delta per mode
delta = contrib.sum(1)
ustar = (f @ fw.Hp).reshape(len(f), -1, 3)
umax = np.linalg.norm(ustar, axis=2).max(1)
fmax = np.linalg.norm(f.reshape(len(f), -1, 3), axis=2).max(1)
print("eigenvalues (eV/A^2), lowest 10:", np.round(lam[:10], 4))
print("frequencies of the lowest modes not needed; share of delta from the lowest 5 / 20 modes (median over E0<0 configs):",
      np.median(contrib[:, :5].sum(1) / delta), np.median(contrib[:, :20].sum(1) / delta))
for q in (50, 90, 99, 99.9):
    print(f"percentile {q}: |delta| {np.percentile(-delta, q):.4f} eV, max|u*| {np.percentile(umax, q):.3f} A, max|f| {np.percentile(fmax, q):.3f} eV/A")
order = np.argsort(delta)[:8]
for k in order:
    top = np.argsort(contrib[k])[:3]
    print(f"E0 {E0[k]:+.3f} delta {delta[k]:+.3f} eV max|f| {fmax[k]:.2f} max|u*| {umax[k]:.2f} A; top modes {top} "
          f"(lambda {np.round(lam[top], 3)}) share {contrib[k, top].sum() / delta[k]:.2f}")
# correlation of large delta with deep binding
for lo, hi in [(-0.6, -0.3), (-0.3, -0.1), (-0.1, 0.0)]:
    m = (E0 >= lo) & (E0 < hi)
    print(f"E0 in [{lo},{hi}): n {m.sum()}, delta median {np.median(delta[m]):.4f}, 99th pct {np.percentile(delta[m], 1):.4f}, "
          f"max|u*| 99th {np.percentile(umax[m], 99):.3f} A")
np.save("diag_umax.npy", umax)

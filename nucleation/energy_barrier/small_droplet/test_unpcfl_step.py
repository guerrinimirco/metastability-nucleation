"""Smoke test for _find_Rc_Wc_step stability-aware phase selection.

Run: python3 test_unpcfl_step.py
Asserts (1) cold/strong-gap limit reproduces the old "CFL wins above Rx" rule
and (2) a finite-T point where unpaired is the stable phase above Rx flips the
barrier back to unpaired (lower, easier nucleation channel).
"""
import numpy as np
from observables import _find_Rc_Wc_step, _blend_phase
from barrier import switching_step, work_of_formation

sigma, Rx = 100.0, 1.0
S_func = lambda R: switching_step(R, Rx)
df = lambda Rc: -2.0 * sigma / Rc          # invert R_c = -2 sigma / Delta_f


def _old_rule(R_c_unp, R_c_cfl, Df_unp, Df_cfl):
    """The pre-fix logic: above Rx always CFL."""
    R_c = np.where(np.isfinite(R_c_unp) & (R_c_unp <= Rx),
                   R_c_unp, np.maximum(Rx, R_c_cfl))
    S = switching_step(R_c, Rx)
    Df = _blend_phase(S, Df_unp, Df_cfl)
    W = work_of_formation(R_c, Df, sigma, np.zeros_like(R_c))
    return R_c, W, S


def test_cold_limit_identical():
    # cfl more stable everywhere (R_c_cfl < R_c_unp): not-pinned / below / pinned
    Ru = np.array([4.0, 0.5, 4.0])
    Rc = np.array([2.0, 0.25, 0.6667])
    z = np.zeros_like(Ru)
    R_n, W_n, S_n = _find_Rc_Wc_step(Ru, Rc, Rx, S_func, df(Ru), df(Rc), z, z, sigma)
    R_o, W_o, S_o = _old_rule(Ru, Rc, df(Ru), df(Rc))
    assert np.allclose(R_n, R_o) and np.allclose(S_n, S_o) and np.allclose(W_n, W_o)


def test_highT_flips_to_unpaired():
    # unpaired more stable above Rx (R_c_unp < R_c_cfl)
    Ru = np.array([2.0]); Rc = np.array([4.0])
    z = np.zeros_like(Ru)
    R_n, W_n, S_n = _find_Rc_Wc_step(Ru, Rc, Rx, S_func, df(Ru), df(Rc), z, z, sigma)
    R_o, W_o, S_o = _old_rule(Ru, Rc, df(Ru), df(Rc))
    assert S_n[0] == 0.0 and S_o[0] == 1.0          # new=unpaired, old=CFL
    assert W_n[0] < W_o[0]                            # easier (correct) channel


if __name__ == "__main__":
    test_cold_limit_identical()
    test_highT_flips_to_unpaired()
    print("OK: cold limit identical, high-T flips to unpaired")

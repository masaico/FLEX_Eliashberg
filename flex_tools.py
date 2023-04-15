import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla

from scipy.fft import fftn, ifftn, fftshift
from scipy.optimize import bisect, brentq, newton
from scipy.special import expit

import sys
from tqdm import tqdm
from multiprocessing import cpu_count, Process, Manager
from dataclasses import dataclass


# =================================================================================

@dataclass
class parameter_manager:

    Nqx: int    # 波数空間の分割数
    Nqy: int     
    Nqz: int
    Nm: int     # 松原周波数の個数（偶数個）
    Norbit: int # 軌道の数
    Norbit_each_site: tuple[int]     # サイト当たりの軌道の数
    Nrvec: int  # 格子ベクトルの本数
    Nsite: int  # ユニットセルのサイト数

    temperature: float
    Nfilling: float # 電子数
    Uinner: float   # 電子間相互作用
    Uinter: float   # 軌道間相互作用
    Jhund: float    # フント結合
    Jpair: float    # ペアホッピング
    # V: float      # オフサイトクーロン相互作用（未実装）
    spin_freedom: int=2

    def __post_init__(self):
        # 波数ベクトル 0 と 2pi は同じなので、piを消す。
        #kx = np.linspace(0, 2*np.pi, self.Nqx+1)[:-1]
        #ky = np.linspace(0, 2*np.pi, self.Nqy+1)[:-1]
        #kz = np.linspace(0, 2*np.pi, self.Nqz+1)[:-1]
        # 範囲は -pi ~ pi (-pi と pi は同じなので pi を消す)
        self.band_filling = self.spin_freedom*self.Nfilling /self.Norbit    # =1 で half filled (spin_freedom=2の時)
        kx = np.linspace(-np.pi, np.pi, self.Nqx+1)[:-1] if self.Nqx > 1 else np.zeros(1)
        ky = np.linspace(-np.pi, np.pi, self.Nqy+1)[:-1] if self.Nqy > 1 else np.zeros(1)
        kz = np.linspace(-np.pi, np.pi, self.Nqz+1)[:-1] if self.Nqz > 1 else np.zeros(1)
        # shape: (Nqx,Nqy,Nqz)
        self.KX, self.KY, self.KZ = \
            np.meshgrid(*[kx, ky, kz], indexing='ij')
        self.Ne = self.band_filling *self.Nsite     # 電子数
        # fermion の松原周波数 1,3,5,...,2*Nm-1
        self.matsubara_half = \
            1j *np.pi *self.temperature *np.arange(1, 2*self.Nm, 2)

    def get_Energy(self):
        return self.KX, self.KY, self.KZ, self.Norbit

    def get_INTERACTION(self):
        return self.Uinner, self.Uinter, self.Jhund, self.Jpair, self.Norbit_each_site

    def calc_mu_non_interacting(self):
        return self.temperature, self.band_filling

    def calc_Green(self):
        return (self.matsubara_half,)

    def calc_Susceptibilities(self):
        return (self.temperature,)
    
    def calc_Self_Energy(self):
        return (self.temperature,)

    def calc_mu(self):
        return self.matsubara_half, self.temperature, self.band_filling

    def get_DELTA_init(self):
        return self.KX, self.KY, self.Nm, self.Norbit

    def calc_GAP(self):
        return (self.temperature,)


class mkmask:
    # 計算の高速化のために使用する。Nsite が 1 で、Uinter, Jhund, Jpair が 0 でないときには効果がない。
    # 3 軌道以上のデバッグは十分でないので、まずは小さいモデルでマスキングを使用しない場合と比較すること。
    #
    # ユニットセルのサイト数が1より多い場合、
    # 例えば Nsite=2, Norbit_each_site=(2,3), Norbit=5 の時、
    # 軌道 0と1 や、2と3 などは相互作用するが、軌道 0と2 や 1と4 などは相互作用しない（これらは異なるサイトに属する軌道だから）。
    # これを反映して、相互作用行列は
    #  /                                              \
    # |  /                   \                         |
    # | | 0000 0001 0010 0011 |                        |
    # | | 0100 0101 0110 0111 |            0           |
    # | | 1000 1001 1010 1011 |                        |
    # | | 1100 1101 1110 1111 |                        |
    # |  \                   /                         |
    # |                          /                  \  |
    # |                         | 2222 2223 ... 2244 | |
    # |                         | 2322 2323 ... 2344 | |
    # |            0            |  :    :        :   | |
    # |                         | 4422 4423 ... 4444 | |
    # |                          \                  /  |
    #  \                                              /
    # のような行列となる。
    #
    # ほかにも、Uinter, Jhund, Jpair が 0 であるとき、1行（列）すべてが 0 になるような行（列）が出てくるので、
    # 相互作用行列が (Norbit^2, Norbit^2) の要素を持っておく必要がない。

    def __init__(self, U_charge, U_spin):
        is_zero = np.logical_or(U_charge==0, U_spin==0)
        if len(is_zero.shape) > 2:
            is_zero = np.sum(is_zero, axis=(0,1,2)) > 0
        mask = np.ones_like(is_zero, dtype=bool)
        for i, is_zero_vec in enumerate(is_zero):
            if np.count_nonzero(is_zero_vec) == len(is_zero_vec):
                mask[i,:].fill(False)
                mask[:,i].fill(False)
        self.mask = mask
    
    def apply_mask(self, mat):
        tile_shape = list(mat.shape)
        ret_shape = list(mat.shape)
        ret_shape[-1] = ret_shape[-2] = np.count_nonzero(self.mask[0])
        N = len(ret_shape)
        if N > 2:
            tile_shape[-1] = tile_shape[-2] = 1
            mask = np.tile(self.mask, tile_shape)
            ret_mat = mat[mask].copy().reshape(ret_shape)
        else:
            ret_mat = mat[self.mask].copy().reshape(ret_shape)
        return ret_mat

    def restore(self, mat):
        ret_shape = list(mat.shape)
        ret_shape[-1] = ret_shape[-2] = self.mask.shape[0]
        ret_mat = np.zeros(ret_shape, dtype=mat.dtype)
        N = len(ret_shape)
        if N > 2:
            ret_shape[-1] = ret_shape[-2] = 1
            mask = np.tile(self.mask, ret_shape)
            ret_mat[mask] = mat.copy().ravel()
        else:
            ret_mat[self.mask] = mat.copy().ravel()
        return ret_mat

# =================================================================================

def get_ENERGY(df, KX, KY, KZ, Norbit):
    Nqx,Nqy,Nqz = KX.shape
    # <orb_1|H|orb_1>, <orb_1|H|orb_2>, ... , <orb_N|H|orb_N>
    # shape: (Norbit^2,Nrvec,3)
    rvec = np.split(np.array(df)[:,:3], Norbit**2)[0]
    # shape: (Nqx,Nqy,Nqz,Nrvec)
    phase =  KX[:,:,:,np.newaxis] *rvec[:,0] \
           + KY[:,:,:,np.newaxis] *rvec[:,1] \
           + KZ[:,:,:,np.newaxis] *rvec[:,2]

    orbit_pair = [(i,j) for i in range(Norbit) for j in range(Norbit)]
    hopping_split = np.split(np.array(df)[:,3], Norbit**2)
    # shape: (Nqx,Nqy,Nqz,Norbit,Norbit) 
    ENERGY = np.empty((Nqx,Nqy,Nqz,Norbit,Norbit), dtype=np.complex128)
    for hopping, (i,j) in zip(hopping_split, orbit_pair):
        if i <= j:
            ENERGY[:,:,:,i,j] = (np.exp(-1j*phase) *hopping).sum(axis=-1)
        else:
            ENERGY[:,:,:,i,j] = np.conjugate(ENERGY[:,:,:,j,i])
    
    if Norbit > 1:
        # Norbit × Norbit 行列の対角化を Nqx*Nqy*Nqz 回行っている。
        # shape: (Nqx,Nqy,Nqz,Norbit), (Nqx,Nqy,Nqz,Norbit,Norbit)
        ENERGY_DIAG, UNITARY = la.eigh(ENERGY)
    else:
        ENERGY_DIAG = ENERGY[:,:,:,:,0]
        UNITARY = np.ones_like(ENERGY)
    # ENERGY_DIAG が、相互作用のないバンドになっている。
    return ENERGY, np.real(ENERGY_DIAG), UNITARY
    

def get_INTERACTION(Uinner, Uinter, Jhund, Jpair, Norbit_each_site, Uweight_each_site=None):

    # 0,1,2,3: orbit / a,b: spin
    #
    #  0,a                 2,b
    #    \                 /
    #     ^               ^
    #      \_ _ _ _ _ _ _/
    #      /  U(1234ab)  \
    #     ^               ^
    #    /                 \
    #  1,a                 3,b

    # ハミルトニアン:  H_I = sum_(i_site) sum_(1234ab) U(1234ab) c+(0a) c(1a) c+(2b) c(3b)
    # U(1234ab)
    #    /
    #   | Uinner: 0=1=2=3, a!=b
    # = | Uinter: 0=1!=2=3 (a=b のとき、1<->3 の入れ替えも可(符号は反転))
    #   | Jhund:  0=2!=1=3 (a=b のとき、1<->3 の入れ替えも可(符号は反転))
    #   | Jpair:  0=3!=1=2 a!=b
    #    \

    # 2軌道の場合、軌道0,1として、 .reshape(4, 4) で、
    #  /                   \
    # | 0000 0001 0010 0011 |
    # | 0100 0101 0110 0111 |
    # | 1000 1001 1010 1011 |
    # | 1100 1101 1110 1111 |
    #  \                   /
    # の要素を持った行列を得る。

    Norbit = np.sum(Norbit_each_site)
    U_charge = np.zeros((Norbit,Norbit,Norbit,Norbit))
    U_spin = np.zeros((Norbit,Norbit,Norbit,Norbit))
    start = 0
    if Uweight_each_site is None:
        Uweight_each_site = np.ones(len(Norbit_each_site))
    for Norbit_site, Uweight in zip(Norbit_each_site, Uweight_each_site):
        U_uu = np.zeros((Norbit_site,Norbit_site,Norbit_site,Norbit_site))
        U_ud = np.zeros((Norbit_site,Norbit_site,Norbit_site,Norbit_site))
        if Norbit_site==1 and (Uinter!=0 or Jhund!=0 or Jpair!=0):
            print("Warning: Interactions between different orbitals (Uinter, Jhund, Jpair) are treated as '0'.")
        for i in range(Norbit_site):
            U_ud[i,i,i,i] +=  Uinner *Uweight
            for j in range(Norbit_site):
                if i != j:
                    U_uu[i,i,j,j] +=  Uinter *Uweight
                    U_uu[i,j,i,j] += -Uinter *Uweight
                    U_ud[i,i,j,j] +=  Uinter *Uweight
                    U_uu[i,j,i,j] +=  Jhund *Uweight
                    U_uu[i,i,j,j] += -Jhund *Uweight
                    U_ud[i,j,i,j] +=  Jhund *Uweight
                    U_ud[i,j,j,i] +=  Jpair *Uweight
        end = start + Norbit_site
        U_charge[start:end,start:end,start:end,start:end] = U_ud + U_uu
        U_spin[start:end,start:end,start:end,start:end] = U_ud - U_uu
        start = end
    U_charge = U_charge.reshape(Norbit**2,Norbit**2)
    U_spin = U_spin.reshape(Norbit**2,Norbit**2)
    # shape: (Norbit^2, Norbit^2)
    # 現在のところ、オフサイトクーロン相互作用は実装していないが、U_charge の shape が (Nqx,Nqy,Nqz,Norbit^2,Norbit^2) になってもほかのコードを変更せずに計算できるようになっている。
    return U_charge, U_spin
    
def calc_mu_non_interacting(ENERGY_DIAG, temperature, band_filling,
                            solver='brent', search_factor=1.2, tol=1e-8):
    # 探索範囲
    search_min = search_factor *np.min(ENERGY_DIAG)
    search_max = search_factor *np.max(ENERGY_DIAG)
    # スピンの自由度 (このコードでは 2 以外想定していない。スピンの上下に別々の軌道を対応させたい時はコードの書き換えが必要。
    #                たぶん band_filling=1 を full filled にして、spin_freedom=1 にすればよいが、確認はしていない。)
    spin_freedom = 2
    beta = 1/temperature
    # full filled で band_filling=2 となることを反映している
    solved_func = \
        lambda mu: spin_freedom*np.mean( expit( -beta*(ENERGY_DIAG-mu) ) ) - band_filling
    if solver == 'brent':
        # ブレント法（二分法 + 逆二次補間）
        mu = brentq(solved_func, search_min, search_max, xtol=tol)
    elif solver == 'bisect':
        # 二分法
        mu = bisect(solved_func, search_min, search_max, xtol=tol)
    else:
        print("Error: solver must be 'brent' or 'bisect'!")
        sys.exit()
    return mu

def calc_Green(SIGMA, ENERGY, mu, matsubara_half, return_half=False):
    Nm = len(matsubara_half)
    Nqx, Nqy, Nqz, Norbit, _ = ENERGY.shape
    
    # shape: (Nm,Norbit,Norbit)
    MATSUBARA_MU = (matsubara_half + mu)[:,np.newaxis,np.newaxis] *np.eye(Norbit)[np.newaxis]
    if SIGMA is None:
        # shape: (Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
        G_inv = - ENERGY[np.newaxis] + MATSUBARA_MU.reshape((Nm,1,1,1,Norbit,Norbit))
    else:
        # shape: (Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
        G_inv = - SIGMA - ENERGY[np.newaxis] + MATSUBARA_MU.reshape((Nm,1,1,1,Norbit,Norbit))

    if Norbit == 1:
        GREEN = 1./G_inv
    else:
        # Norbit × Norbit 行列の逆行列を 2*Nm*Nqx*Nqy*Nqz 回求めている。LU分解。
        GREEN = la.inv(G_inv)
    if return_half:
        # shape: (Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
        return GREEN
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
    return np.vstack((np.conjugate(GREEN).transpose(0,1,2,3,5,4)[::-1], GREEN))

def from_iwn_k_to_tau_r(arr, fft_workers=None):
    # shape: (2*Nm,Nqx,Nqy,Nqz, ... )   (... は自由)
    # マスキングを使用しない場合には、入力として
    # (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit), (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit,Norbit,Norbit)
    # を想定している。
    if fft_workers is None:
        fft_workers = cpu_count()
    return fftn(arr, axes=(0,1,2,3), workers=fft_workers)

def calc_chi0_tau_r(G_tau_r):
    Nm2, Nqx, Nqy, Nqz, Norbit, _ = G_tau_r.shape
    
    """
    CHI0_tau_r = np.empty((Nm2,Nqx,Nqy,Nqz,Norbit,Norbit,Norbit,Norbit), dtype=np.complex128)
    for k in range(Norbit):
        for l in range(Norbit):
            for m in range(Norbit):
                for n in range(Norbit):
                    CHI0_tau_r[:,:,:,:,k,l,m,n] = \
                        G_tau_r[:,:,:,:,k,m] *np.roll(G_tau_r, -1, axis=(0,1,2,3))[::-1,::-1,::-1,::-1,n,l]
    """
    
    # chi0_0123(tau,r) = G_31(tau,r) *G_02(-tau,-r)
    # 上のコードと同じことをしている
    CHI0_tau_r = G_tau_r.transpose(0,1,2,3,5,4)[:,:,:,:,np.newaxis,:,np.newaxis,:]\
                *np.roll(G_tau_r, -1, axis=(0,1,2,3))[::-1,::-1,::-1,::-1,:,np.newaxis,:,np.newaxis]
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit^2,Norbit^2)
    return CHI0_tau_r.reshape((Nm2,Nqx,Nqy,Nqz,Norbit**2,Norbit**2))

def calc_Susceptibilities(CHI0_tau_r, U_charge, U_spin, temperature, fft_workers=None):
    if fft_workers is None:
        fft_workers = cpu_count()
    # マスキングを使用していなければ、N_mask = Norbit^2
    # Norbit = 1 の時は常に N_mask = 1
    Nm2, Nqx, Nqy, Nqz, N_mask, _ = CHI0_tau_r.shape

    CHI0 = - ifftn(CHI0_tau_r, axes=(0,1,2,3), workers=fft_workers) *temperature /(Nqx*Nqy*Nqz)
    CHI0 = fftshift(CHI0, axes=(0,1,2,3))
    # 松原周波数が 0,1,...,Nm-1,-Nm,...,-1 の順になっているので、-Nm,...,-1,0,...,Nm-1 の順に戻す。
    # ボソンの演算子なので、松原周波数もボソンのものに対応していることに注意。
    # -2Nm, -2Nm+2, ... , -2, | 0, 2, ... , 2Nm-2 
    # | は配列の中心。

    # 電荷感受率、スピン感受率
    if N_mask == 1:   # Norbit==1
        CHI_charge = CHI0 /(1 + U_charge*CHI0)
        CHI_spin = CHI0 /(1 - U_spin*CHI0)
    else:
        CHI_charge = np.matmul( CHI0, la.inv(np.eye(N_mask) + np.matmul(U_charge, CHI0)) )
        CHI_spin = np.matmul( CHI0, la.inv(np.eye(N_mask) - np.matmul(U_spin, CHI0)) )
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit^2,Norbit^2)
    return CHI0, CHI_charge, CHI_spin


def calc_V_effective(U_charge, U_spin, CHI0, CHI_charge, CHI_spin):
    # V_Sigma = 3/2 *U_S chi_S U_S + 1/2 *U_C chi_C U_C - 1/4 *(U_S + U_C) chi0 (U_S + U_C) + 1/2 *(U_C - U_S)
    U_M = 0.5*(U_spin + U_charge)
    V_Sigma = \
          1.5*np.matmul(U_spin, np.matmul(CHI_spin, U_spin)) \
        + 0.5*np.matmul(U_charge, np.matmul(CHI_charge, U_charge)) \
        - np.matmul(U_M, np.matmul(CHI0, U_M)) \
        - 0.5*U_charge + 1.5*U_spin
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit^2,Norbit^2)
    return V_Sigma


def calc_Self_Energy(V_Sigma_tau_r, G_tau_r, temperature, fft_workers=None):
    if fft_workers is None:
        fft_workers = cpu_count()

    Nm2,Nqx,Nqy,Nqz,Norbit,_ = G_tau_r.shape
    Nm = Nm2//2
    shape_V = (Nm2,Nqx,Nqy,Nqz,Norbit,Norbit,Norbit,Norbit)
    V_Sigma_tau_r = V_Sigma_tau_r.reshape(shape_V)
    # 0,1,2,3: orbit
    # Sigma_02(iw_n,k) = T/N *sum_13 sum_{q,iv_n} V_0123(q,iv_n) *G_13(k-q,iw_n-iv_n)
    # Sigma_02(tau,r) = T/N *sum_13 V_0123(tau,r) *G_13(tau,r)
    SIGMA_tau_r = np.sum( V_Sigma_tau_r *G_tau_r[:,:,:,:,:,np.newaxis,:,np.newaxis], axis=(-2,-4) )
    
    # 松原振動数の順番
    # 1, 3, ... , 2Nm-1, -2Nm+1, ... , -1
    SIGMA = ifftn(SIGMA_tau_r, axes=(0,1,2,3), workers=fft_workers) *temperature /(Nqx*Nqy*Nqz)
    # 松原振動数は半分だけ残す。
    # 反対側は複素共役なので、グリーン関数の計算にはこれで十分。
    # shape: (Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
    #return SIGMA[:Nm]
    return fftshift(SIGMA[:Nm], axes=(1,2,3))

def eval_Self_Energy_err(SIGMA_new, SIGMA_prev):
    if SIGMA_prev is None:
        return np.inf
    return np.max( np.absolute(SIGMA_new - SIGMA_prev) /(np.absolute(SIGMA_prev) + 1e-8) )

def mix_Self_Energy(SIGMA_new, SIGMA_prev, mix_factor):
    if SIGMA_prev is None:
        return mix_factor *SIGMA_new
    return mix_factor *SIGMA_new + (1-mix_factor) *SIGMA_prev


def calc_mu(SIGMA, ENERGY, ENERGY_DIAG, matsubara_half, temperature, band_filling,
            mu_init=None, search_min=None, search_max=None, solver='newton', tol=1e-5):
    beta = 1./temperature
    Nm,Nqx,Nqy,Nqz,Norbit,_ = SIGMA.shape
    # スピンの自由度 (このコードでは 2 以外想定していない)
    spin_freedom = 2
    coef = spin_freedom*temperature /(Nqx*Nqy*Nqz*Norbit) *2   # *2 はグリーン関数が松原周波数を半分しか保持していないため。
    # n = 2T/N *sum_{k,n} G(iw_n,k) + 1
    # から化学ポテンシャルを補正する。(n は band filling)
    # ただし、G は iw_n の増加に対して収束が遅い (1/iw_n のスピード) ので、
    # non-interacting の場合に
    # 2/N *sum_k f(eps(k)) = 2T/N *sum_{k,n} G0(iw_n,k) + 1
    # であることを用いて、
    # n = 2*<f(eps)> + 2T/N *sum_{k,n} ( G(iw_n,k) - G0(iw_n,k) )
    # で補正を行っている。
    def solved_func(mu):
        a = spin_freedom*np.mean( expit( -beta*(ENERGY_DIAG-mu) ) )
        # 化学ポテンシャルの計算には、グリーン関数の要素は半分でいいので、return_half オプションを True にする。
        b = np.real(calc_Green(SIGMA,ENERGY,mu,matsubara_half, return_half=True)).sum(axis=(0,1,2,3))
        c = np.real(calc_Green(None,ENERGY,mu,matsubara_half, return_half=True)).sum(axis=(0,1,2,3))
        b = np.sum(b.diagonal()) *coef
        c = np.sum(c.diagonal()) *coef
        return a + b - c - band_filling
    if solver == 'newton':
        # 割線法（速い。mu_init が解と離れていると収束しないことがある。）
        if mu_init is None:
            print("Error: mu_init must not be None!")
            sys.exit()
        mu = newton(solved_func, mu_init, tol=tol, maxiter=25)
    elif solver == 'brent':
        # ブレント法（二分法 + 逆二次補間 なので安全。）
        if (search_max is None) or (search_min is None):
            print("Error: search_max and search_min must not be None!")
            sys.exit()
        mu = brentq(solved_func, search_min, search_max, xtol=tol)
    elif solver == 'bisect':
        # 二分法
        if (search_max is None) or (search_min is None):
            print("Error: search_max and search_min must not be None!")
            sys.exit()
        mu = bisect(solved_func, search_min, search_max, xtol=tol)
    else:
        print("Error: solver must be 'newton', 'brent' or 'bisect'!")
        sys.exit()
    return mu

def get_DELTA_band_init(KX, KY, Nm, Norbit, gap_symmetry='random', seed=0):
    Nqx,Nqy,Nqz = KX.shape
    shape = (2*Nm,Nqx,Nqy,Nqz)
    DELTA_init = np.empty(shape, dtype=np.complex128)
    # 今のところ、random 以外機能していない。指定してもよいが、目的のギャップ関数の形に収束しない。おそらく数値的な誤差のため。
    # Eliashberg_pipeline の symmetrize を ('s', 'dx2-y2', 'dxy') と指定するのがよい。
    # あるいは、Eliashberg_pipeline の k に 2以上の整数を指定すると、複数の固有関数を一度に計算するので、目視で区別するのもよい。
    if gap_symmetry == 'random':
        np.random.seed(seed)
        DELTA_init[:] = np.random.random((Nqx,Nqy,Nqz))
        DELTA_init[:] += np.roll(DELTA_init, -1, axis=(1,2,3))[:,::-1,::-1,::-1]
    elif gap_symmetry == 's':
        DELTA_init[:] = np.cos(KX + KY) + np.cos(KX - KY)
    elif gap_symmetry == 'dx2-y2':
        DELTA_init[:] = np.cos(KX) - np.cos(KY)
    elif gap_symmetry == 'dxy':
        DELTA_init[:] = np.sin(KX)*np.sin(KY)
    # デバッグ用なので、通常は使用しない。
    elif gap_symmetry == 'one':
        DELTA_init[:] = 1
    else:
        print("Error: gap_symmetry must be 'random', 's', 'dx2-y2' or 'dxy'!")
        sys.exit()
    DELTA_init = DELTA_init[:,:,:,:,np.newaxis,np.newaxis] *np.eye(Norbit)
    DELTA_init /= la.norm(DELTA_init)
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
    return DELTA_init

def from_band_to_orbit(arr, UNITARY):
    # shape: (Nqx,Nqy,Nqz,Norbit,Norbit)
    UNITARY_dag = np.conjugate(UNITARY).transpose(0,1,2,4,3)
    # shape: (...,Nqx,Nqy,Nqz,Norbit,Norbit)
    return np.matmul(UNITARY, np.matmul(arr, UNITARY_dag))

def from_orbit_to_band(arr, UNITARY):
    # shape: (Nqx,Nqy,Nqz,Norbit,Norbit)
    UNITARY_dag = np.conjugate(UNITARY).transpose(0,1,2,4,3)
    # shape: (...,Nqx,Nqy,Nqz,Norbit,Norbit)
    return np.matmul(UNITARY_dag, np.matmul(arr, UNITARY))


def calc_Gamma_effective(U_charge, U_spin, CHI_charge, CHI_spin, spin_symmetry='singlet'):
    if spin_symmetry == 'singlet':
        GAMMA =   1.5*np.matmul(U_spin, np.matmul(CHI_spin, U_spin)) \
                - 0.5*np.matmul(U_charge, np.matmul(CHI_charge, U_charge)) \
                + 0.5*(U_charge + U_spin)
    elif spin_symmetry == 'triplet':
        GAMMA = - 0.5*np.matmul(U_spin, np.matmul(CHI_spin, U_spin)) \
                - 0.5*np.matmul(U_charge, np.matmul(CHI_charge, U_charge)) \
                + 0.5*(U_charge - U_spin)
    else:
        print("symmetry must be 'singlet' or 'triplet'!")
        sys.exit()
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit^2,Norbit^2)
    return GAMMA

def calc_Anomalous_Green(DELTA, GREEN, GREEN_dag_minus=None):
    if GREEN_dag_minus is None:
        GREEN_dag_minus = np.roll(GREEN, -1, axis=(1,2,3))[::-1,::-1,::-1,::-1].transpose(0,1,2,3,5,4)
    # F(q) = G(q)Delta(q)G^dag(-q)
    F = np.matmul(GREEN, np.matmul(DELTA, GREEN_dag_minus))
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
    return F

def calc_GAP(GAMMA_tau_r, F_tau_r, temperature, fft_workers=None):
    if fft_workers is None:
        fft_workers = cpu_count()
    Nm2,Nqx,Nqy,Nqz,Norbit,_ = F_tau_r.shape
    shape_Gamma = (Nm2,Nqx,Nqy,Nqz,Norbit,Norbit,Norbit,Norbit)
    GAMMA_tau_r = GAMMA_tau_r.reshape(shape_Gamma)
    DELTA_tau_r = np.sum( GAMMA_tau_r *F_tau_r[:,:,:,:,:,np.newaxis,:,np.newaxis], axis=(-2,-4) )
    DELTA = - ifftn(DELTA_tau_r, axes=(0,1,2,3), workers=fft_workers) *temperature /(Nqx*Nqy*Nqz)
    # shape: (2*Nm,Nqx,Nqy,Nqz,Norbit,Norbit)
    #return fftshift(DELTA, axes=0)
    return fftshift(DELTA, axes=(0,1,2,3))


def pade_approx(arr, matsubara_half, epsilon, Nm_cutoff):
    # フェルミオンの松原周波数を想定している。
    Nm2 = arr.shape[0]
    Nm = len(matsubara_half)
    Neps = len(epsilon)
    matsubara_sample = matsubara_half[:Nm_cutoff,np.newaxis]
    epsilon = epsilon[:,np.newaxis]
    if Nm2 == 2*Nm:
        coef = arr[Nm:Nm+Nm_cutoff].copy()
    elif Nm2 == Nm:
        coef = arr[:Nm_cutoff].copy()
    else:
        print("Error: 1st dimension of input array's shape must be Nm or 2*Nm!")
        sys.exit()
    coef  = coef.reshape(Nm_cutoff,coef.size//Nm_cutoff)

    A = np.empty((2,Neps,coef.shape[-1]), dtype=np.complex128)
    A[-1,:,:].fill(0)
    A[ 0,:,:] = coef[[0]].copy()
    B = np.ones((2,Neps,coef.shape[-1]), dtype=np.complex128)
    for i in tqdm(range(1,Nm_cutoff)):
        # 次元を合わせるため、coef[[i-1]] などの [] は必要。
        coef[i:] = (coef[[i-1]] - coef[i:]) /( (matsubara_sample[i:] - matsubara_sample[[i-1]]) *coef[i:])
        coef[i:][np.absolute(coef[i:])<1e-10] = 1e-20 + 1e-20j
        # いらなくなった配列に上書きしていく。この計算が遅い。
        C = (epsilon - matsubara_sample[[i-1]]) *coef[[i]]
        A[i%2] *= C
        A[i%2] += A[i%2-1]
        B[i%2] *= C
        B[i%2] += B[i%2-1]
    
    ret_shape = list(arr.shape)
    ret_shape[0] = Neps
    ret_arr = A[i%2]/B[i%2]
    return ret_arr.reshape(ret_shape)

# Nqx, Nqy, Nqz が大きい時に有効
# Nqx = Nqy = 16, Nqz = 1 くらいでは使わないほうが早いかも。
def pade_parallel(arr, matsubara_half, epsilon, Nm_cutoff, workers=None):
    # フェルミオンの松原周波数を想定している。
    Nm2 = arr.shape[0]
    N_k_orbit = arr.size //Nm2
    if workers is None:
        workers = np.min([cpu_count(), N_k_orbit])
    Nm = len(matsubara_half)
    Neps = len(epsilon)
    matsubara_sample = matsubara_half[:Nm_cutoff,np.newaxis]
    epsilon = epsilon[:,np.newaxis]
    if Nm2 == 2*Nm:
        coef = arr[Nm:Nm+Nm_cutoff].copy()
    elif Nm2 == Nm:
        coef = arr[:Nm_cutoff].copy()
    else:
        print("Error: 1st dimension of input array's shape must be Nm or 2*Nm!")
        sys.exit()
    coef  = coef.reshape(Nm_cutoff,N_k_orbit)
    coef_list = np.array_split(coef, workers, axis=1)

    # 並列化したい部分
    def main_routine(coef, matsubara_sample, epsilon, process_num, shared_dict):
        N_sample = len(epsilon)
        Nm_cutoff, N_split = coef.shape
        A = np.empty((2,N_sample,N_split), dtype=np.complex128)
        A[-1,:,:].fill(0)
        A[ 0,:,:] = coef[[0]].copy()
        B = np.ones((2,N_sample,N_split), dtype=np.complex128)
        for i in range(1,Nm_cutoff):
            # 次元を合わせるため、coef[[i-1]] などの [] は必要。
            coef[i:] = (coef[[i-1]] - coef[i:]) /( (matsubara_sample[i:] - matsubara_sample[[i-1]]) *coef[i:])
            coef[i:][np.absolute(coef[i:])<1e-12] = 1e-20 + 1e-20j
            # いらなくなった配列に上書きしていく。
            C = (epsilon - matsubara_sample[[i-1]]) *coef[[i]]
            A[i%2] *= C
            A[i%2] += A[i%2-1]
            B[i%2] *= C
            B[i%2] += B[i%2-1]
        shared_dict[process_num] = A[i%2]/B[i%2]

    with Manager() as manager:
        shared_dict = manager.dict()
        p_list = []
        for i, coef in enumerate(coef_list):
            p_list.append( 
                Process(target=main_routine, args=(coef, matsubara_sample, epsilon, i, shared_dict))
                )
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        ret_shape = list(arr.shape)
        ret_shape[0] = Neps
        ret_arr = np.hstack([shared_dict[i] for i in range(workers)])
    return ret_arr.reshape(ret_shape)

# =================================================================================

# 並列数は自動で決定される。
# 並列数を制限したい場合は、fft_workers オプションを指定したうえで、numpy の並列数も手動で指定する必要がある。
# 参考: https://qiita.com/yymgt/items/b7f151ee99fb830ca64c

def RPA_pipeline(ENERGY, ENERGY_DIAG, U_charge, U_spin, mu_init, pm, mask=None):
    G0 = calc_Green(None, ENERGY, mu_init, *pm.calc_Green())
    G_tau_r = from_iwn_k_to_tau_r(G0)
    CHI0_tau_r = calc_chi0_tau_r(G_tau_r)
    if mask is not None:
        U_charge = mask.apply_mask(U_charge)
        U_spin = mask.apply_mask(U_spin)
        CHI0_tau_r = mask.apply_mask(CHI0_tau_r)
    CHI0, CHI_charge, CHI_spin = \
        calc_Susceptibilities(CHI0_tau_r, U_charge, U_spin, *pm.calc_Susceptibilities())
    V_Sigma = calc_V_effective(U_charge, U_spin, CHI0, CHI_charge, CHI_spin)
    V_Sigma_tau_r = from_iwn_k_to_tau_r(V_Sigma)
    if mask is not None:
        V_Sigma_tau_r = mask.restore(V_Sigma_tau_r)
    SIGMA = calc_Self_Energy(V_Sigma_tau_r, G_tau_r, *pm.calc_Self_Energy())
    mu = calc_mu(SIGMA, ENERGY, ENERGY_DIAG, *pm.calc_mu(), mu_init=mu_init)
    G = calc_Green(SIGMA, ENERGY, mu, *pm.calc_Green())
    return G0, CHI0, CHI_charge, CHI_spin, SIGMA, mu, G


# FLEX_pileline はあくまで一例。自分で定義しなおすとよい。
# 計算が進むにつれて mix_rate を変化させたり、探索範囲を変化させたりすると効率的な計算ができる。
def FLEX_pipeline(ENERGY, ENERGY_DIAG, U_charge, U_spin, mu_init, pm, mix_rate, N_flexiter=1000, tol_flex=1e-3, mask=None):
    mu = mu_init
    SIGMA_tmp = None
    if mask is not None:
        U_charge = mask.apply_mask(U_charge)
        U_spin = mask.apply_mask(U_spin)
    for i in range(N_flexiter):
        G = calc_Green(SIGMA_tmp, ENERGY, mu, *pm.calc_Green())
        G_tau_r = from_iwn_k_to_tau_r(G)
        CHI0_tau_r = calc_chi0_tau_r(G_tau_r)
        if mask is not None:
            CHI0_tau_r = mask.apply_mask(CHI0_tau_r)
        CHI0, CHI_charge, CHI_spin = \
            calc_Susceptibilities(CHI0_tau_r, U_charge, U_spin, *pm.calc_Susceptibilities())
        V_Sigma = calc_V_effective(U_charge, U_spin, CHI0, CHI_charge, CHI_spin)
        V_Sigma_tau_r = from_iwn_k_to_tau_r(V_Sigma)
        if mask is not None:
            V_Sigma_tau_r = mask.restore(V_Sigma_tau_r)
        SIGMA = calc_Self_Energy(V_Sigma_tau_r, G_tau_r, *pm.calc_Self_Energy())
        # 収束の判定
        err = eval_Self_Energy_err(SIGMA, SIGMA_tmp)
        if err < tol_flex:
            SIGMA = mix_Self_Energy(SIGMA, SIGMA_tmp, 0.5)
            mu = calc_mu(SIGMA, ENERGY, ENERGY_DIAG, *pm.calc_mu(), mu_init=mu)
            G = calc_Green(SIGMA, ENERGY, mu, *pm.calc_Green())
            return CHI0, CHI_charge, CHI_spin, SIGMA, mu, G
        else:
            print('err={:.4g}'.format(err))
            SIGMA_tmp = mix_Self_Energy(SIGMA, SIGMA_tmp, mix_rate)
            if i < 2/mix_rate:
                try:
                    mu = calc_mu(SIGMA_tmp, ENERGY, ENERGY_DIAG, *pm.calc_mu(), search_min=mu-1, search_max=mu+1, solver='brent')
                except:
                    mu = calc_mu(SIGMA_tmp, ENERGY, ENERGY_DIAG, *pm.calc_mu(), search_min=mu-2, search_max=mu+2, solver='brent')
            else:
                try:
                    mu = calc_mu(SIGMA_tmp, ENERGY, ENERGY_DIAG, *pm.calc_mu(), mu_init=mu)
                except:
                    mu = calc_mu(SIGMA_tmp, ENERGY, ENERGY_DIAG, *pm.calc_mu(), search_min=mu-1, search_max=mu+1, solver='brent')
    print('Waring: Self Energy is not converge!  err={:.4g}'.format(err))
    G = calc_Green(SIGMA_tmp, ENERGY, mu, *pm.calc_Green())
    return CHI0, CHI_charge, CHI_spin, SIGMA, mu, G

# 参考: https://triqs.github.io/tprf/latest/_modules/triqs_tprf/eliashberg.html
def Eliashberg_pipeline(GREEN, U_charge, U_spin, CHI_charge, CHI_spin, DELTA_init, pm, tol=1e-4, mask=None, k=1, symmetrize='auto'):
    # symmetrize='auto' / None 以外を指定するときは、k=1 で計算することを想定している。
    # symmetrize に関数を指定することで、DELTA の形を制限することができるが、使ったことはない。
    # 収束しない場合は、symmetrize=None を試してみる。
    if mask is not None:
        U_charge = mask.apply_mask(U_charge)
        U_spin = mask.apply_mask(U_spin)
    GAMMA = calc_Gamma_effective(U_charge, U_spin, CHI_charge, CHI_spin)
    GAMMA_tau_r = from_iwn_k_to_tau_r(GAMMA)
    if mask is not None:
        GAMMA_tau_r = mask.restore(GAMMA_tau_r)
    # G^dag(-q)
    GREEN_dag_minus = np.roll(GREEN, -1, axis=(1,2,3))[::-1,::-1,::-1,::-1].transpose(0,1,2,3,5,4)
    
    def matvec(DELTA):
        DELTA = DELTA.reshape(GREEN.shape)
        F = calc_Anomalous_Green(DELTA, GREEN, GREEN_dag_minus)
        F_tau_r = from_iwn_k_to_tau_r(F)
        DELTA = calc_GAP(GAMMA_tau_r, F_tau_r, *pm.calc_GAP())
        if symmetrize is not None:
            # スピン1重項を想定しているので、スピン3重項を考える場合には書き直す必要がある。
            if symmetrize in ('auto', 's', 'dxy', 'dx2-y2'):
                _,Nx,Ny,Nz,_,_ = DELTA.shape
                x, y, z = int(Nx>1), int(Ny>1), int(Nz>1)
                # 松原周波数に関して偶関数を仮定。奇周波数超伝導を考える場合には別のコードを書く必要がある。
                #DELTA[:,x:,y:,z:] += DELTA[:,x:,y:,z:][::-1,::-1,::-1,::-1]
                #DELTA[:,x:,y:,z:] *= 0.5
                #Nm = Nm2//2
                # 松原周波数に関して偶関数を仮定。奇周波数超伝導を考える場合には別のコードを書く必要がある。
                #DELTA[Nm:,x:,y:,z:] = np.conjugate(DELTA)[:Nm,x:,y:,z:][::-1,::-1,::-1,::-1]
                # これ以降は Nx, Ny が偶数で Nx=Ny の場合を想定している。この辺りはちゃんと確認していないので、計算結果をよく確認すること。
                if symmetrize == 's':
                    for i in range(1,Nx//2+1):
                        DELTA[:,:i,i] = DELTA[:,i,:i]
                    DELTA[:,1:Nx//2,:Ny//2:-1] = DELTA[:,:Nx//2:-1,1:Ny//2] = DELTA[:,:Nx//2:-1,:Ny//2:-1] = DELTA[:,1:Nx//2,1:Ny//2]
                elif symmetrize == 'dxy':
                    DELTA[:,1:Nx//2,:Ny//2:-1] = DELTA[:,:Nx//2:-1,1:Ny//2] = - DELTA[:,1:Nx//2,1:Ny//2]
                    DELTA[:,[0,Nx//2],:] = DELTA[:,:,[0,Nx//2]] = 0
                elif symmetrize == 'dx2-y2':
                    DELTA[:,0,0] = 0
                    for i in range(1,Nx//2+1):
                        DELTA[:,i,i] = 0
                        DELTA[:,:i,i] = -DELTA[:,i,:i]
                    DELTA[:,:Nx//2,:Ny//2:-1] = DELTA[:,:Nx//2,1:Ny//2]
                    DELTA[:,:Nx//2:-1,:Ny//2] = DELTA[:,1:Nx//2,:Ny//2]
                    DELTA[:,:Nx//2-1:-1,:Ny//2-1:-1] =  DELTA[:,1:Nx//2+1,1:Ny//2+1]
                else:   # symmetrize == 'auto'
                    pass
            else:
                DELTA = symmetrize(DELTA)
        return DELTA.ravel()
     
    N = len(GREEN.ravel())
    linop = spla.LinearOperator( matvec=matvec, dtype=np.complex128, shape=(N,N) )
    # 大きさ最大の固有値ではなく、代数的に最大な固有値を探す（正の固有値の中で最大）。
    eigvals, DELTAs = spla.eigsh(linop, k=k, which='LA', tol=tol, v0=DELTA_init.ravel())
    return eigvals, [DELTA.reshape(GREEN.shape) for DELTA in DELTAs.T]
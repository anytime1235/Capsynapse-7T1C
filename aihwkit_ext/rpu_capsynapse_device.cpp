/**
 * (C) Copyright 2020-2024 IBM. All Rights Reserved. Licensed under the MIT license.
 * Based on the AIHWKIT 1.1.0 device sources. Modifications (C) 2026 Jaeweon Park, MIT license.
 */

/* CapSynapseRPUDevice: CPU implementation (no write disturbance). See the header. */

#include "rpu_capsynapse_device.h"
#include "utility_functions.h"
#include <cmath>
#include <iostream>

namespace RPU {

template <typename T>
void CapSynapseRPUDevice<T>::populate(
    const CapSynapseRPUDeviceMetaParameter<T> &p, RealWorldRNG<T> *rng) {

  PulsedRPUDevice<T>::populate(p, rng); // will clone par
  auto &par = getPar();

  T b_mu = par.vov_median / par.vdd;
  T b_sd = par.vov_std / par.vdd;
  T k = par.vdd / par.vov_median; // alpha = U (k beta)^2

  for (int i = 0; i < this->d_size_; ++i) {
    for (int j = 0; j < this->x_size_; ++j) {
      T gmax = (T)1.0 + par.bound_std * rng->sampleGauss();
      T gmin = (T)-1.0 + par.bound_std * rng->sampleGauss();
      T R = gmax - gmin;
      // symmetric bounds around the per-cell reference (V_cap = 0)
      this->w_max_bound_[i][j] = R / (T)2.0;
      this->w_min_bound_[i][j] = -R / (T)2.0;

      T beta_p, beta_d, alpha_p, alpha_d, lam_p, lam_d;
      if (par.shared_transistor) {
        beta_p = beta_d = b_mu + b_sd * rng->sampleGauss();
        T u = (T)1.0 + par.w_rel_std * rng->sampleGauss();
        alpha_p = alpha_d = u * (k * beta_p) * (k * beta_p);
        lam_p = par.lam * ((T)1.0 + par.lam_rel_std * rng->sampleGauss());
        lam_d = par.lam * ((T)1.0 + par.lam_rel_std * rng->sampleGauss());
      } else {
        beta_p = b_mu + b_sd * rng->sampleGauss();
        beta_d = b_mu + b_sd * rng->sampleGauss();
        T up = (T)1.0 + par.w_rel_std * rng->sampleGauss();
        T ud = (T)1.0 + par.w_rel_std * rng->sampleGauss();
        alpha_p = up * (k * beta_p) * (k * beta_p);
        alpha_d = ud * (k * beta_d) * (k * beta_d);
        lam_p = lam_d = par.lam;
      }
      beta_p = MAX(beta_p, (T)1e-3);
      beta_d = MAX(beta_d, (T)1e-3);
      alpha_p = MAX(alpha_p, (T)1e-3);
      alpha_d = MAX(alpha_d, (T)1e-3);

      this->w_scale_up_[i][j] = R * alpha_p / par.step_median;
      this->w_scale_down_[i][j] = R * alpha_d / par.step_median;
      beta_pot_[i][j] = beta_p;
      beta_dep_[i][j] = beta_d;
      lam_pot_[i][j] = lam_p;
      lam_dep_[i][j] = lam_d;
    }
  }
}

template <typename T> void CapSynapseRPUDevice<T>::printDP(int x_count, int d_count) const {
  if (x_count < 0 || x_count > this->x_size_) {
    x_count = this->x_size_;
  }
  if (d_count < 0 || d_count > this->d_size_) {
    d_count = this->d_size_;
  }
  for (int i = 0; i < d_count; ++i) {
    for (int j = 0; j < x_count; ++j) {
      std::cout.precision(5);
      std::cout << i << "," << j << ": [" << this->w_min_bound_[i][j] << ","
                << this->w_max_bound_[i][j] << "] su=" << this->w_scale_up_[i][j]
                << " sd=" << this->w_scale_down_[i][j] << " beta=(" << beta_pot_[i][j] << ","
                << beta_dep_[i][j] << ") lam=(" << lam_pot_[i][j] << "," << lam_dep_[i][j] << ")"
                << std::endl;
    }
  }
}

namespace {
template <typename T>
inline void update_once(
    T &w,
    const int &sign,
    const T &sd,
    const T &su,
    const T &wmin,
    const T &wmax,
    const T &beta_p,
    const T &beta_d,
    const T &lam_p_eff,
    const T &lam_d_eff,
    const T &dw_min_std,
    RNG<T> *rng) {
  T R = wmax - wmin;
  T dw;
  if (sign > 0) { // depression
    dw = -capsynapse_dw_dep(w, wmin, R, sd, beta_d, lam_d_eff);
  } else { // potentiation
    dw = capsynapse_dw_pot(w, wmax, R, su, beta_p, lam_p_eff);
  }
  if (dw_min_std > (T)0.0) {
    dw *= ((T)1.0 + dw_min_std * rng->sampleGauss());
  }
  w += dw;
  w = MAX(w, wmin);
  w = MIN(w, wmax);
}
} // namespace

template <typename T>
void CapSynapseRPUDevice<T>::doSparseUpdate(
    T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {

  const auto &par = getPar();
  T *scale_down = this->w_scale_down_[i];
  T *scale_up = this->w_scale_up_[i];
  T *w = weights[i];
  T *min_bound = this->w_min_bound_[i];
  T *max_bound = this->w_max_bound_[i];
  T *bp = beta_pot_[i];
  T *bd = beta_dep_[i];
  T *lp = lam_pot_[i];
  T *ld = lam_dep_[i];
  T ls = par.lam_scale;

  PULSED_UPDATE_W_LOOP(update_once(
                           w[j], sign, scale_down[j], scale_up[j], min_bound[j], max_bound[j],
                           bp[j], bd[j], lp[j] * ls, ld[j] * ls, par.dw_min_std, rng););
}

template <typename T>
void CapSynapseRPUDevice<T>::doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {

  const auto &par = getPar();
  T *scale_down = this->w_scale_down_[0];
  T *scale_up = this->w_scale_up_[0];
  T *w = weights[0];
  T *min_bound = this->w_min_bound_[0];
  T *max_bound = this->w_max_bound_[0];
  T *bp = beta_pot_[0];
  T *bd = beta_dep_[0];
  T *lp = lam_pot_[0];
  T *ld = lam_dep_[0];
  T ls = par.lam_scale;

  PULSED_UPDATE_W_LOOP_DENSE(update_once(
                                 w[j], sign, scale_down[j], scale_up[j], min_bound[j],
                                 max_bound[j], bp[j], bd[j], lp[j] * ls, ld[j] * ls,
                                 par.dw_min_std, rng););
}

template class CapSynapseRPUDevice<float>;
#ifdef RPU_USE_DOUBLE
template class CapSynapseRPUDevice<double>;
#endif
#ifdef RPU_USE_FP16
template class CapSynapseRPUDevice<half_t>;
#endif

} // namespace RPU

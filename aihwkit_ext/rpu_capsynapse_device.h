/**
 * (C) Copyright 2020-2024 IBM. All Rights Reserved. Licensed under the MIT license.
 * Based on the AIHWKIT 1.1.0 device sources. Modifications (C) 2026 Jaeweon Park, MIT license.
 */

/**
 * CapSynapseRPUDevice: 6T1C / 7T1C capacitor-based synapse (Supplemental Note S4).
 *
 * w = G - G_ref with G_ref the per-cell midpoint (V_cap = 0); bounds [-R/2, R/2], R = G_max - G_min.
 *
 * Per-cell parameters (populate):
 *   G_max, G_min   ~ N(+-1, bound_std^2)
 *   beta           V_ov / V_DD ~ N(vov_median/vdd, (vov_std/vdd)^2)
 *   alpha          U (vdd*beta/vov_median)^2, U = 1 + w_rel_std*Z
 *   lam_pot/dep    7T1C: lam (1 + lam_rel_std*Z) each; 6T1C: lam
 *   6T1C (shared_transistor = false): beta and alpha drawn separately for pot (N2) and dep (N4)
 *   7T1C (shared_transistor = true):  one beta and alpha for both directions (N7)
 *
 * One pulse, su = R*alpha_pot/step_median, sd = R*alpha_dep/step_median:
 *   pot: G_b = wmax - R*beta_pot
 *        dw = su (1 + lam_pot*lam_scale*(G_b - w))   if w <= G_b   (saturation)
 *        dw = su (wmax - w) / (R*beta_pot)           otherwise     (linear)
 *   dep: G_b = wmin + R*beta_dep
 *        dw = sd (1 + lam_dep*lam_scale*(w - G_b))   if w >  G_b   (saturation)
 *        dw = sd (w - wmin) / (R*beta_dep)           otherwise     (linear)
 *
 * Write disturbance (CUDA only, hs_gamma > 0; Kang et al., Adv. Sci. 2025). Every cell has one
 * transistor selected at all times, and the node it settles on depends only on the row pulse r and
 * the phase: potentiation r ? CP : CN, depression r ? CN : CP. Each CP <-> CN transition decays the
 * weight, w <- (1 - hs_gamma) w.
 *   hs_pulse_scheduling  all potentiation of one image before any depression
 *   hs_ps_group          mat-vecs per image (H_out * W_out of a conv layer), used to count the two
 *                        transitions at each image boundary under pulse scheduling
 */

#pragma once

#include "rng.h"
#include "rpu_pulsed_device.h"
#include "utility_functions.h"
#include <utility>

namespace RPU {

template <typename T> class CapSynapseRPUDevice;

BUILD_PULSED_DEVICE_META_PARAMETER(
    CapSynapse,
    /*implements*/
    DeviceUpdateType::CapSynapse,

    /*parameter def*/
    bool shared_transistor = false;
    T vdd = (T)3.0;
    T vov_median = (T)0.35;
    T vov_std = (T)0.043;
    T lam = (T)0.0324;
    T lam_rel_std = (T)(1.0 / 30.0);
    // (G_b - w) in weight units -> volts: 1 weight unit = 4.5 uS = 1.5 V (lam in 1/V)
    T lam_scale = (T)1.5;
    T w_rel_std = (T)(1.0 / 30.0);
    T bound_std = (T)(0.15 / 4.5);
    T step_median = (T)1000.0;
    T hs_gamma = (T)0.0;
    bool hs_pulse_scheduling = true;
    T hs_ps_group = (T)0.0;
    ,
    /*print body*/
    ss << "\t shared_transistor:\t" << std::boolalpha << shared_transistor << std::endl;
    ss << "\t vdd:\t\t\t" << vdd << std::endl;
    ss << "\t vov_median:\t\t" << vov_median << std::endl;
    ss << "\t vov_std:\t\t" << vov_std << std::endl;
    ss << "\t lam:\t\t\t" << lam << std::endl;
    ss << "\t lam_rel_std:\t\t" << lam_rel_std << std::endl;
    ss << "\t lam_scale:\t\t" << lam_scale << std::endl;
    ss << "\t w_rel_std:\t\t" << w_rel_std << std::endl;
    ss << "\t bound_std:\t\t" << bound_std << std::endl;
    ss << "\t step_median:\t\t" << step_median << std::endl;
    ss << "\t hs_gamma:\t\t" << hs_gamma << std::endl;
    ss << "\t hs_pulse_scheduling:\t" << std::boolalpha << hs_pulse_scheduling << std::endl;
    ss << "\t hs_ps_group:\t\t" << hs_ps_group << std::endl;
    ,
    /* calc weight granularity body */
    return (T)2.0 / step_median;
    ,
    /*Add*/
    bool implementsWriteNoise() const override { return false; };
    void initialize() override {
      this->dw_min = (T)2.0 / step_median;
      this->w_max = (T)1.0;
      this->w_min = (T)-1.0;
      PulsedRPUDeviceMetaParameter<T>::initialize();
      if (step_median <= (T)0.0 || vov_median <= (T)0.0 || vdd <= (T)0.0) {
        RPU_FATAL("CapSynapse: step_median, vov_median and vdd need to be positive.");
      }
    };);

template <typename T> class CapSynapseRPUDevice : public PulsedRPUDevice<T> {

  BUILD_PULSED_DEVICE_CONSTRUCTORS(
      CapSynapseRPUDevice,
      /* ctor*/
      int x_sz = this->x_size_;
      int d_sz = this->d_size_;
      beta_pot_ = Array_2D_Get<T>(d_sz, x_sz);
      beta_dep_ = Array_2D_Get<T>(d_sz, x_sz);
      lam_pot_ = Array_2D_Get<T>(d_sz, x_sz);
      lam_dep_ = Array_2D_Get<T>(d_sz, x_sz);
      for (int j = 0; j < x_sz; ++j) {
        for (int i = 0; i < d_sz; ++i) {
          beta_pot_[i][j] = (T)0.0;
          beta_dep_[i][j] = (T)0.0;
          lam_pot_[i][j] = (T)0.0;
          lam_dep_[i][j] = (T)0.0;
        }
      },
      /* dtor*/
      Array_2D_Free<T>(beta_pot_);
      Array_2D_Free<T>(beta_dep_);
      Array_2D_Free<T>(lam_pot_);
      Array_2D_Free<T>(lam_dep_);
      ,
      /* copy */
      for (int j = 0; j < other.x_size_; ++j) {
        for (int i = 0; i < other.d_size_; ++i) {
          beta_pot_[i][j] = other.beta_pot_[i][j];
          beta_dep_[i][j] = other.beta_dep_[i][j];
          lam_pot_[i][j] = other.lam_pot_[i][j];
          lam_dep_[i][j] = other.lam_dep_[i][j];
        }
      },
      /* move assignment */
      beta_pot_ = other.beta_pot_;
      beta_dep_ = other.beta_dep_;
      lam_pot_ = other.lam_pot_;
      lam_dep_ = other.lam_dep_;
      other.beta_pot_ = nullptr;
      other.beta_dep_ = nullptr;
      other.lam_pot_ = nullptr;
      other.lam_dep_ = nullptr;
      ,
      /* swap*/
      swap(a.beta_pot_, b.beta_pot_);
      swap(a.beta_dep_, b.beta_dep_);
      swap(a.lam_pot_, b.lam_pot_);
      swap(a.lam_dep_, b.lam_dep_);
      ,
      /* dp names*/
      names.push_back(std::string("beta_pot"));
      names.push_back(std::string("beta_dep"));
      names.push_back(std::string("lam_pot"));
      names.push_back(std::string("lam_dep"));
      ,
      /* dp2vec body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;
      for (int i = 0; i < size; ++i) {
        data_ptrs[n_prev][i] = beta_pot_[0][i];
        data_ptrs[n_prev + 1][i] = beta_dep_[0][i];
        data_ptrs[n_prev + 2][i] = lam_pot_[0][i];
        data_ptrs[n_prev + 3][i] = lam_dep_[0][i];
      },
      /* vec2dp body*/
      int n_prev = (int)names.size();
      int size = this->x_size_ * this->d_size_;
      for (int i = 0; i < size; ++i) {
        beta_pot_[0][i] = data_ptrs[n_prev][i];
        beta_dep_[0][i] = data_ptrs[n_prev + 1][i];
        lam_pot_[0][i] = data_ptrs[n_prev + 2][i];
        lam_dep_[0][i] = data_ptrs[n_prev + 3][i];
      },
      /*invert copy DP */
      for (int j = 0; j < this->x_size_; ++j) {
        for (int i = 0; i < this->d_size_; ++i) {
          std::swap(beta_pot_[i][j], beta_dep_[i][j]);
          std::swap(lam_pot_[i][j], lam_dep_[i][j]);
        }
      }
  );

  void printDP(int x_count, int d_count) const override;

  inline T **getBetaPot() const { return beta_pot_; };
  inline T **getBetaDep() const { return beta_dep_; };
  inline T **getLamPot() const { return lam_pot_; };
  inline T **getLamDep() const { return lam_dep_; };

  void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng)
      override;
  void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) override;

private:
  T **beta_pot_ = nullptr;
  T **beta_dep_ = nullptr;
  T **lam_pot_ = nullptr;
  T **lam_dep_ = nullptr;
};

/* update step of one pulse (CPU device) */
template <typename T>
inline T capsynapse_dw_pot(T w, T wmax, T R, T su, T beta_pot, T lam_pot_eff) {
  T gb = wmax - R * beta_pot;
  return (w <= gb) ? su * ((T)1.0 + lam_pot_eff * (gb - w)) : su * (wmax - w) / (R * beta_pot);
}
template <typename T>
inline T capsynapse_dw_dep(T w, T wmin, T R, T sd, T beta_dep, T lam_dep_eff) {
  T gb = wmin + R * beta_dep;
  return (w > gb) ? sd * ((T)1.0 + lam_dep_eff * (w - gb)) : sd * (w - wmin) / (R * beta_dep);
}

} // namespace RPU

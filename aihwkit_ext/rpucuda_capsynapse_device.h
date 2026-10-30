/**
 * (C) Copyright 2020-2024 IBM. All Rights Reserved. Licensed under the MIT license.
 * Based on the AIHWKIT 1.1.0 device sources. Modifications (C) 2026 Jaeweon Park, MIT license.
 */

/**
 * CapSynapseRPUDeviceCuda: CUDA implementation.
 *
 * Parameters of the update kernels:
 *   par_4     (wmin, scale_down, wmax, scale_up)
 *   par_2     6T1C: (beta_pot, beta_dep); 7T1C: (beta, lam_pot * lam_scale)
 *   par_1     7T1C: lam_dep * lam_scale
 *   global    [lam * lam_scale, shared_transistor, hs_gamma, hs_pulse_scheduling, hs_ps_group]
 *   hs_state  per cell: 0 = no history, else 1 + prev + 2*first (previous row bit, and the row bit
 *             at the start of the current pulse-scheduling group)
 *   hs_pos_   mat-vecs processed so far
 */

#pragma once

#include "pwu_kernel_parameter_base.h"
#include "rpu_capsynapse_device.h"
#include "rpucuda_pulsed_device.h"
#include <memory>

namespace RPU {

template <typename T> class CapSynapseRPUDeviceCuda : public PulsedRPUDeviceCuda<T> {

public:
  BUILD_PULSED_DEVICE_CONSTRUCTORS_CUDA(
      CapSynapseRPUDeviceCuda,
      CapSynapseRPUDevice,
      /*ctor body*/
      dev_par2_ = RPU::make_unique<CudaArray<param_t>>(this->context_, 2 * this->size_);
      dev_par1_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
      dev_global_pars_ = RPU::make_unique<CudaArray<T>>(this->context_, 8);
      dev_hs_state_ = RPU::make_unique<CudaArray<T>>(this->context_, this->size_);
      dev_hs_state_->setConst((T)0.0);
      ,
      /*dtor body*/
      ,
      /*copy body*/
      dev_par2_->assign(*other.dev_par2_);
      dev_par1_->assign(*other.dev_par1_);
      dev_global_pars_->assign(*other.dev_global_pars_);
      dev_hs_state_->assign(*other.dev_hs_state_);
      hs_pos_ = other.hs_pos_;
      ,
      /*move assigment body*/
      dev_par2_ = std::move(other.dev_par2_);
      dev_par1_ = std::move(other.dev_par1_);
      dev_global_pars_ = std::move(other.dev_global_pars_);
      dev_hs_state_ = std::move(other.dev_hs_state_);
      hs_pos_ = other.hs_pos_;
      ,
      /*swap body*/
      swap(a.dev_par2_, b.dev_par2_);
      swap(a.dev_par1_, b.dev_par1_);
      swap(a.dev_global_pars_, b.dev_global_pars_);
      swap(a.dev_hs_state_, b.dev_hs_state_);
      swap(a.hs_pos_, b.hs_pos_);
      ,
      /*host copy from cpu (rpu_device). Parent device params are copyied automatically*/
      const auto &par = getPar();
      int d_size = this->d_size_;
      int x_size = this->x_size_;
      T **bp = rpu_device.getBetaPot();
      T **bd = rpu_device.getBetaDep();
      T **lp = rpu_device.getLamPot();
      T **ld = rpu_device.getLamDep();
      T ls = par.lam_scale;
      param_t *tmp2 = new param_t[2 * this->size_];
      T *tmp1 = new T[this->size_];
      for (int i = 0; i < d_size; ++i) {
        for (int j = 0; j < x_size; ++j) {
          int l_t = j * d_size + i;
          int kk = 2 * l_t;
          if (par.shared_transistor) {
            tmp2[kk] = (param_t)bp[i][j];
            tmp2[kk + 1] = (param_t)(lp[i][j] * ls);
            tmp1[l_t] = ld[i][j] * ls;
          } else {
            tmp2[kk] = (param_t)bp[i][j];
            tmp2[kk + 1] = (param_t)bd[i][j];
            tmp1[l_t] = (T)0.0;
          }
        }
      }
      dev_par2_->assign(tmp2);
      dev_par1_->assign(tmp1);
      T tmpg[8];
      for (int k = 0; k < 8; ++k) { tmpg[k] = (T)0.0; }
      tmpg[0] = par.lam * ls;
      tmpg[1] = par.shared_transistor ? (T)1.0 : (T)0.0;
      tmpg[2] = par.hs_gamma;
      tmpg[3] = par.hs_pulse_scheduling ? (T)1.0 : (T)0.0;
      tmpg[4] = par.hs_ps_group;
      dev_global_pars_->assign(tmpg);
      this->context_->synchronize();
      delete[] tmp2;
      delete[] tmp1;);

  pwukpvec_t<T> getUpdateKernels(
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans,
      const PulsedUpdateMetaParameter<T> &up) override;

  T *getGlobalParamsData() override { return dev_global_pars_->getData(); };
  param_t *get2ParamsData() override { return dev_par2_->getData(); };
  T *get1ParamsData() override {
    return getPar().shared_transistor ? dev_par1_->getData() : nullptr;
  };
  T *getHSStateData() { return dev_hs_state_->getData(); };
  void resetHSState() {
    dev_hs_state_->setConst((T)0.0);
    hs_pos_ = 0;
  };
  int getHSPos() const { return hs_pos_; };
  void advanceHSPos(int n) { hs_pos_ += n; };

private:
  std::unique_ptr<CudaArray<param_t>> dev_par2_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_par1_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_global_pars_ = nullptr;
  std::unique_ptr<CudaArray<T>> dev_hs_state_ = nullptr;
  int hs_pos_ = 0;
};

} // namespace RPU

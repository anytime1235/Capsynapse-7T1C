/**
 * (C) Copyright 2020-2024 IBM. All Rights Reserved. Licensed under the MIT license.
 * Based on the AIHWKIT 1.1.0 device sources. Modifications (C) 2026 Jaeweon Park, MIT license.
 */

/* CapSynapseRPUDeviceCuda: update functor and write-disturbance kernels. See rpu_capsynapse_device.h. */

#include "pwu_kernel_parameter.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpucuda_capsynapse_device.h"
#include <memory>

namespace RPU {

template <typename T> struct UpdateFunctorCapSynapse {

  __device__ __forceinline__ void operator()(
      T &apparent_weight,
      uint32_t n,
      uint32_t negative,
      const param4_t par_4,
      const param2_t par_2,
      T &persistent_weight, // par_1: lam_dep * lam_scale (7T1C), read only
      const T *global_pars,
      const int global_params_count,
      T noise_std_dw,
      curandState &local_state) {

    UNUSED(global_params_count);

    T &w = apparent_weight;
    T wmin = (T)par_4.x;
    T sd = (T)par_4.y;
    T wmax = (T)par_4.z;
    T su = (T)par_4.w;
    T R = wmax - wmin;

    bool shared = global_pars[1] > (T)0.5;
    T beta_p = (T)par_2.x;
    T beta_d = shared ? beta_p : (T)par_2.y;
    T lam_p = shared ? (T)par_2.y : global_pars[0];
    T lam_d = shared ? persistent_weight : global_pars[0];

    // negative > 0 means going up (potentiation)
    for (uint32_t k = 0; k < n; k++) {
      T dw;
      if (negative > 0) {
        T gb = wmax - R * beta_p;
        dw = (w <= gb) ? su * ((T)1.0 + lam_p * (gb - w)) : su * (wmax - w) / (R * beta_p);
      } else {
        T gb = wmin + R * beta_d;
        dw = (w > gb) ? sd * ((T)1.0 + lam_d * (w - gb)) : sd * (w - wmin) / (R * beta_d);
        dw = -dw;
      }
      if (noise_std_dw > (T)0.0) {
        T stoch_value = curand_normal(&local_state);
        dw *= ((T)1.0 + noise_std_dw * stoch_value);
      }
      w += dw;
      w = (w > wmax) ? wmax : w;
      w = (w < wmin) ? wmin : w;
    }
  }
};


/* Write-disturbance kernels: the pulse words are walked slot by slot in temporal order, instead of
 * popcount(x & d), so that the node-group transitions of each cell can be counted. */
namespace {
template <typename T>
__device__ __forceinline__ void capsyn_update_once(
    T &w, bool pot, const param4_t &par_4, T beta_p, T beta_d, T lam_p, T lam_d, T noise_std_dw,
    curandState &local_state) {
  T wmin = (T)par_4.x, sd = (T)par_4.y, wmax = (T)par_4.z, su = (T)par_4.w;
  T R = wmax - wmin;
  T dw;
  if (pot) {
    T gb = wmax - R * beta_p;
    dw = (w <= gb) ? su * ((T)1.0 + lam_p * (gb - w)) : su * (wmax - w) / (R * beta_p);
  } else {
    T gb = wmin + R * beta_d;
    dw = (w > gb) ? sd * ((T)1.0 + lam_d * (w - gb)) : sd * (w - wmin) / (R * beta_d);
    dw = -dw;
  }
  if (noise_std_dw > (T)0.0) {
    dw *= ((T)1.0 + noise_std_dw * curand_normal(&local_state));
  }
  w += dw;
  w = (w > wmax) ? wmax : w;
  w = (w < wmin) ? wmin : w;
}

/* one mat-vec: bl pulse slots of the row word xd and the column word dd (bit 0 = sign) */
template <typename T>
__device__ __forceinline__ void capsyn_process_word(
    T &w, T &st, uint32_t xd, uint32_t dd, uint32_t negm, int bl, const param4_t &par_4,
    T beta_p, T beta_d, T lam_p, T lam_d, T gamma, bool ps, bool group_start, T noise_std_dw,
    curandState &local_state) {
  /* With pulse scheduling the node group flips only when the row bit flips (once in each phase);
   * without it the phase alternates every slot (two flips per slot, minus one when the row bit
   * changes). Decay is applied before the update of the same slot. */
  T one_minus_g = (T)1.0 - gamma;
  int code = (int)st;
  uint32_t prev = (code > 0) ? (uint32_t)((code - 1) & 1) : 0u;
  uint32_t first = (code > 0) ? (uint32_t)(((code - 1) >> 1) & 1) : 0u;

  for (int b = 1; b <= bl; b++) {
    uint32_t xb = (xd >> b) & 1u;
    uint32_t db = (dd >> b) & 1u;

    int n_ev;
    if (code == 0) {
      n_ev = ps ? 0 : 1; /* first slot ever: no previous group under PS */
      first = xb;
    } else if (ps && group_start && b == 1) {
      /* group boundary: end of the pot block vs start of the dep block of the finished image,
       * and end of its dep block vs start of the pot block of the new image */
      n_ev = ((prev == first) ? 1 : 0) + ((prev == xb) ? 1 : 0);
      first = xb;
    } else {
      bool flip = prev != xb;
      n_ev = ps ? (flip ? 2 : 0) : (flip ? 1 : 2);
    }
    for (int e = 0; e < n_ev; e++) {
      w *= one_minus_g;
    }
    prev = xb;
    code = 1 + (int)prev + 2 * (int)first;

    if (xb && db) {
      bool pot = ((negm >> b) & 1u) != 0u;
      capsyn_update_once<T>(w, pot, par_4, beta_p, beta_d, lam_p, lam_d, noise_std_dw, local_state);
    }
  }
  st = (T)code;
}

template <typename T>
__device__ __forceinline__ void capsyn_unpack(
    const param4_t &par_4, const param2_t &par_2, T par_1, const T *gp, T &beta_p, T &beta_d,
    T &lam_p, T &lam_d) {
  bool shared = gp[1] > (T)0.5;
  beta_p = (T)par_2.x;
  beta_d = shared ? beta_p : (T)par_2.y;
  lam_p = shared ? (T)par_2.y : gp[0];
  lam_d = shared ? par_1 : gp[0];
}
} // namespace

/* m_batch == 1, 32-bit compressed format (bit 0 of word 0 = sign) */
template <typename T>
__global__ void kernelUpdateWCapSynapseSingle(
    T *weights,
    uint32_t *x_counts,
    int x_size,
    uint32_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
    T *params_1,
    T *global_params,
    int bl,
    T dw_min_std,
    curandState *random_states,
    T *hs_state,
    int pos_offset) {
  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int xsz = x_size, dsz = d_size, sz = xsz * dsz;
  const int total_threads = blockDim.x * gridDim.x;
  __shared__ T gp[8];
  if (threadIdx.x < 8) {
    gp[threadIdx.x] = global_params[threadIdx.x];
  }
  __syncthreads();
  const T gamma = gp[2];
  const bool ps = gp[3] > (T)0.5;
  const int ps_group = (int)gp[4];
  /* one mat-vec per launch: a group starts when the position is a multiple of ps_group */
  const bool group_start = (ps_group > 0) && (pos_offset % ps_group == 0);
  const bool use_par_1 = params_1 != nullptr;
  curandState local_state;
  if (dw_min_std > (T)0.0 && tid < sz) {
    local_state = random_states[tid];
  }
  for (int idx = tid; idx < sz; idx += total_threads) {
    int dIdx = idx % dsz;
    int xIdx = idx / dsz;
    T w = weights[idx];
    T st = hs_state[idx];
    param4_t par_4 = reinterpret_cast<param4_t *>(params)[idx];
    param2_t par_2 = reinterpret_cast<param2_t *>(params_2)[idx];
    T par_1 = use_par_1 ? params_1[idx] : (T)0.0;
    T beta_p, beta_d, lam_p, lam_d;
    capsyn_unpack<T>(par_4, par_2, par_1, gp, beta_p, beta_d, lam_p, lam_d);

    uint32_t x0 = x_counts[xIdx];
    uint32_t d0 = d_counts[dIdx];
    uint32_t negm = ((x0 & 1u) ^ (d0 & 1u)) ? 0xffffffffu : 0u;
    capsyn_process_word<T>(
        w, st, x0, d0, negm, bl, par_4, beta_p, beta_d, lam_p, lam_d, gamma, ps, group_start,
        dw_min_std, local_state);
    weights[idx] = w;
    hs_state[idx] = st;
  }
  if (dw_min_std > (T)0.0 && tid < sz) {
    random_states[tid] = local_state;
  }
}

/* batched version (compressed 32-bit format, bit 0 = sign of each mat-vec) */
template <typename T, typename count_t, bool x_trans, bool d_trans>
__global__ void kernelUpdateWCapSynapseBatch(
    T *weights,
    count_t *x_counts,
    int x_size,
    count_t *d_counts,
    int d_size,
    param_t *params,
    param_t *params_2,
    T *params_1,
    T *global_params,
    int bl,
    int m_batch,
    const T dw_min_std,
    curandState *random_states,
    T *hs_state,
    int pos_offset) {
  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int total_threads = blockDim.x * gridDim.x;
  const int xsz = x_size, dsz = d_size, sz = xsz * dsz;
  const int nK32 = 1;
  __shared__ T gp[8];
  if (threadIdx.x < 8) {
    gp[threadIdx.x] = global_params[threadIdx.x];
  }
  __syncthreads();
  const T gamma = gp[2];
  const bool ps = gp[3] > (T)0.5;
  const int ps_group = (int)gp[4];
  const bool use_par_1 = params_1 != nullptr;
  curandState local_state;
  if (dw_min_std > (T)0.0 && tid < sz) {
    local_state = random_states[tid];
  }
  const int x_count_offset = xsz * nK32;
  const int d_count_offset = dsz * nK32;

  for (int idx = tid; idx < sz; idx += total_threads) {
    int dIdx = idx % dsz;
    int xIdx = idx / dsz;
    T w = weights[idx];
    T st = hs_state[idx];
    param4_t par_4 = reinterpret_cast<param4_t *>(params)[idx];
    param2_t par_2 = reinterpret_cast<param2_t *>(params_2)[idx];
    T par_1 = use_par_1 ? params_1[idx] : (T)0.0;
    T beta_p, beta_d, lam_p, lam_d;
    capsyn_unpack<T>(par_4, par_2, par_1, gp, beta_p, beta_d, lam_p, lam_d);

    for (int i_batch = 0; i_batch < m_batch; i_batch++) {
      count_t *xptr = &x_counts[getIdxToLoad<x_trans>(i_batch, xIdx, xsz, m_batch, x_count_offset)];
      count_t *dptr = &d_counts[getIdxToLoad<d_trans>(i_batch, dIdx, dsz, m_batch, d_count_offset)];
      uint32_t x0 = (uint32_t)(*xptr);
      uint32_t d0 = (uint32_t)(*dptr);
      uint32_t negm = ((x0 & 1u) ^ (d0 & 1u)) ? 0xffffffffu : 0u;
      /* the mat-vecs of one image are contiguous (AnalogConv2d unfolds to B x L) */
      const bool group_start =
          (ps_group > 0) && ((pos_offset + i_batch) % ps_group == 0);
      capsyn_process_word<T>(
          w, st, x0, d0, negm, bl, par_4, beta_p, beta_d, lam_p, lam_d, gamma, ps, group_start,
          dw_min_std, local_state);
    }
    weights[idx] = w;
    hs_state[idx] = st;
  }
  if (dw_min_std > (T)0.0 && tid < sz) {
    random_states[tid] = local_state;
  }
}

/* kernel-parameter classes (DEFINE_PWU_KERNEL_* is #undef'd in pwu_kernel_parameter.h) */
template <typename T> class PWUKernelParameterCapSynSingle : public PWUKernelParameterBase<T> {
public:
  using PWUKernelParameterBase<T>::print;
  using PWUKernelParameterBase<T>::run;
  PWUKernelParameterCapSynSingle(
      CudaContextPtr construction_context,
      int x_size,
      int d_size,
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans_in,
      const PulsedUpdateMetaParameter<T> &up,
      std::string update_name)
      : PWUKernelParameterBase<T>(
            construction_context, x_size, d_size, m_batch, nK32, use_bo64, out_trans_in, up,
            update_name + "/CapSynSingle") {
    if ((m_batch > 1) || (use_bo64 > 0) || this->implicit_pulses || (nK32 != 1)) {
      this->valid = false; /* disturbance decoding needs the plain 32-bit format */
      return;
    }
    int n32 = (this->size / 32 + 1) * 32;
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, n32);
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(this->size, this->nthreads));
    this->nstates = this->nthreads * this->nblocks;
  }

  void run(
      cudaStream_t s,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      PulsedRPUDeviceCuda<T> *rpucuda_device,
      const PulsedUpdateMetaParameter<T> &up,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) override {
    UNUSED(m_batch);
    UNUSED(one_sided);
    UNUSED(cwo);
    auto *dev = dynamic_cast<CapSynapseRPUDeviceCuda<T> *>(rpucuda_device);
    if (dev == nullptr) {
      RPU_FATAL("CapSynSingle kernel needs a CapSynapseRPUDeviceCuda.");
    }
    kernelUpdateWCapSynapseSingle<T><<<this->nblocks, this->nthreads, this->shared_mem, s>>>(
        dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,
        d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,
        rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
        rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), up.desired_BL,
        rpucuda_device->getWeightGranularityNoise(), dev_states, dev->getHSStateData(),
        dev->getHSPos());
    dev->advanceHSPos(1); /* this kernel is the m_batch == 1 path */
  }
};

template <typename T> class PWUKernelParameterCapSynBatch : public PWUKernelParameterBase<T> {
public:
  using PWUKernelParameterBase<T>::print;
  using PWUKernelParameterBase<T>::run;
  PWUKernelParameterCapSynBatch(
      CudaContextPtr construction_context,
      int x_size,
      int d_size,
      int m_batch,
      int nK32,
      int use_bo64,
      bool out_trans_in,
      const PulsedUpdateMetaParameter<T> &up,
      std::string update_name)
      : PWUKernelParameterBase<T>(
            construction_context, x_size, d_size, m_batch, nK32, use_bo64, out_trans_in, up,
            update_name + "/CapSynBatch") {
    if (this->implicit_pulses || (use_bo64 > 0) || (nK32 != 1)) {
      this->valid = false; /* disturbance decoding needs the plain 32-bit format */
      return;
    }
    this->nthreads = MIN(RPU_THREADS_PER_BLOCK_UPDATE, this->size);
    this->nthreads = (this->nthreads + 31) / 32 * 32;
    this->nblocks =
        MIN(this->max_block_count, construction_context->getNBlocks(this->size, this->nthreads));
    this->nstates = this->nthreads * this->nblocks;
  }

#define CS_SWITCH_TRANS(COUNT_T, ARGS)                                                             \
  if (this->out_trans) {                                                                           \
    kernelUpdateWCapSynapseBatch<T, COUNT_T, true, true>                                           \
        <<<this->nblocks, this->nthreads, this->shared_mem, s>>> ARGS;                             \
  } else {                                                                                         \
    kernelUpdateWCapSynapseBatch<T, COUNT_T, false, false>                                         \
        <<<this->nblocks, this->nthreads, this->shared_mem, s>>> ARGS;                             \
  }

  void run(
      cudaStream_t s,
      T *dev_weights,
      int m_batch,
      const BitLineMaker<T> *blm,
      PulsedRPUDeviceCuda<T> *rpucuda_device,
      const PulsedUpdateMetaParameter<T> &up,
      curandState_t *dev_states,
      int one_sided = 0,
      uint32_t *x_counts_chunk = nullptr,
      uint32_t *d_counts_chunk = nullptr,
      const ChoppedWeightOutput<T> *cwo = nullptr) override {
    UNUSED(one_sided);
    UNUSED(cwo);
    auto *dev = dynamic_cast<CapSynapseRPUDeviceCuda<T> *>(rpucuda_device);
    if (dev == nullptr) {
      RPU_FATAL("CapSynBatch kernel needs a CapSynapseRPUDeviceCuda.");
    }
    CS_SWITCH_TRANS(
        uint32_t,
        (dev_weights, x_counts_chunk ? x_counts_chunk : blm->getXCountsData(), this->x_size,
         d_counts_chunk ? d_counts_chunk : blm->getDCountsData(), this->d_size,
         rpucuda_device->get4ParamsData(), rpucuda_device->get2ParamsData(),
         rpucuda_device->get1ParamsData(), rpucuda_device->getGlobalParamsData(), up.desired_BL,
         m_batch, rpucuda_device->getWeightGranularityNoise(), dev_states, dev->getHSStateData(),
         dev->getHSPos()));
    dev->advanceHSPos(m_batch);
  }
#undef CS_SWITCH_TRANS
};

#define ARGS                                                                                       \
  (this->context_, this->x_size_, this->d_size_, m_batch, nK32, use_bo64, out_trans, up,           \
   par.getName())

template <typename T>
pwukpvec_t<T> CapSynapseRPUDeviceCuda<T>::getUpdateKernels(
    int m_batch, int nK32, int use_bo64, bool out_trans, const PulsedUpdateMetaParameter<T> &up) {

  pwukpvec_t<T> v;
  const auto &par = getPar();
  if (par.hs_gamma > (T)0.0) {
    // write-disturbance kernels
    v.push_back(RPU::make_unique<PWUKernelParameterCapSynSingle<T>> ARGS);
    v.push_back(RPU::make_unique<PWUKernelParameterCapSynBatch<T>> ARGS);
    return v;
  }
  v.push_back(
      RPU::make_unique<PWUKernelParameterSingleFunctor<T, UpdateFunctorCapSynapse<T>, 8>> ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchFunctor<T, UpdateFunctorCapSynapse<T>, 8>> ARGS);
  v.push_back(
      RPU::make_unique<PWUKernelParameterBatchSharedFunctor<T, UpdateFunctorCapSynapse<T>, 8>>
          ARGS);
  return v;
}
#undef ARGS

template class CapSynapseRPUDeviceCuda<float>;
#ifdef RPU_USE_DOUBLE
template class CapSynapseRPUDeviceCuda<double>;
#endif
#ifdef RPU_USE_FP16
template class CapSynapseRPUDeviceCuda<half_t>;
#endif

} // namespace RPU

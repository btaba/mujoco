// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_PY_MUJOCO_MJX_CUDA_ENGINE_CORE_SMOOTH_H_
#define THIRD_PARTY_PY_MUJOCO_MJX_CUDA_ENGINE_CORE_SMOOTH_H_

#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/tensorflow/compiler/xla/ffi/api/ffi.h"

namespace mujoco::mjx::cuda {

xla::ffi::Error LaunchKernel_Kinematics(
    cudaStream_t stream, xla::ffi::Buffer<xla::ffi::DataType::F32> qpos0,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_jntadr,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_jntnum,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_parentid,
    xla::ffi::Buffer<xla::ffi::DataType::S32> body_mocapid,
    xla::ffi::Buffer<xla::ffi::DataType::F32> body_pos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> body_quat,
    xla::ffi::Buffer<xla::ffi::DataType::F32> body_ipos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> body_iquat,
    xla::ffi::Buffer<xla::ffi::DataType::S32> jnt_type,
    xla::ffi::Buffer<xla::ffi::DataType::S32> jnt_qposadr,
    xla::ffi::Buffer<xla::ffi::DataType::F32> jnt_axis,
    xla::ffi::Buffer<xla::ffi::DataType::F32> jnt_pos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_pos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> geom_quat,
    xla::ffi::Buffer<xla::ffi::DataType::F32> site_pos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> site_quat,
    xla::ffi::Buffer<xla::ffi::DataType::F32> qpos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> mocap_pos,
    xla::ffi::Buffer<xla::ffi::DataType::F32> mocap_quat,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xanchor,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xaxis,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xmat,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xpos,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xquat,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> xipos,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> ximat,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> geom_xpos,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> geom_xmat,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> site_xpos,
    xla::ffi::Result<xla::ffi::Buffer<xla::ffi::DataType::F32>> site_xmat);

}  // namespace mujoco::mjx::cuda

#endif  // THIRD_PARTY_PY_MUJOCO_MJX_CUDA_ENGINE_CORE_SMOOTH_H_
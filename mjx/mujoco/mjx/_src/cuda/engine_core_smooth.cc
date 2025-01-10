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

#include "mujoco/mjx/google/cuda/engine_core_smooth.h"

#include "gpus/cuda/include/driver_types.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "tensorflow/compiler/xla/ffi/api/c_api.h"
#include "tensorflow/compiler/xla/ffi/api/ffi.h"

namespace mujoco::mjx::cuda {

namespace ffi = xla::ffi;

// Note that the types of the lambda passed to .To() below match the template
// arguments to .Ctx(), .Attr() and .Arg().
static const auto *kKinematics =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // qpos0
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // body_jntadr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // body_jntnum
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // body_parentid
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // body_mocapid
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // body_pos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // body_quat
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // body_ipos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // body_iquat
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // jnt_type
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // jnt_qposadr
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // jnt_axis
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // jnt_pos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // geom_pos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // geom_quat
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // site_pos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // site_quat
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // qpos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // mocap_pos
        .Arg<ffi::Buffer<ffi::DataType::F32>>()    // mocap_quat
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xanchor
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xaxis
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xmat
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xpos
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xquat
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // xipos
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // ximat
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // geom_xpos
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // geom_xmat
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // site_xpos
        .Ret<ffi::Buffer<ffi::DataType::F32>>()    // site_xmat
        .To(LaunchKernel_Kinematics)
        .release();

XLA_FFI_Error* Kinematics(XLA_FFI_CallFrame* call_frame) {
  return kKinematics->Call(call_frame);
}

namespace {

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(_engine_core_smooth, m) {
  m.def("kinematics", []() { return EncapsulateFfiCall(Kinematics); });
}

}  // namespace

}  // namespace mujoco::mjx::cuda
set(kernel_declare_file
    ${PADDLE_BINARY_DIR}/paddle/phi/kernels/declarations.h.tmp
    CACHE INTERNAL "declarations.h file")
set(kernel_declare_file_final
    ${PADDLE_BINARY_DIR}/paddle/phi/kernels/declarations.h)
file(
  WRITE ${kernel_declare_file}
  "// Generated by the paddle/phi/kernels/CMakeLists.txt.  DO NOT EDIT!\n\n#pragma once\n\n"
)
file(APPEND ${kernel_declare_file}
     "#include \"paddle/phi/core/kernel_registry.h\"\n\n")
set(kernel_declare_file_prune
    ${PADDLE_BINARY_DIR}/paddle/phi/kernels/declarations.h.prune
    CACHE INTERNAL "declarations.h file")

# phi functors and functions called by kernels
add_subdirectory(funcs)

# kernel autotune
add_subdirectory(autotune)

copy_if_different(${kernel_declare_file} ${kernel_declare_file_final})

file(GLOB kernel_h "*.h" "selected_rows/*.h" "sparse/*.h" "strings/*.h")
file(GLOB kernel_impl_h "impl/*.h" "selected_rows/impl/*.h")
file(GLOB kernel_primitive_h "primitive/*.h")

# fusion ops would be included here
file(
  GLOB kernel_cu
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "gpu/*.cu"
  "gpu/*.cu.cc"
  "gpudnn/*.cu"
  "kps/*.cu"
  "legacy/kps/*.cu"
  "legacy/gpu/*.cu"
  "selected_rows/gpu/*.cu"
  "sparse/gpu/*.cu"
  "strings/gpu/*.cu"
  "fusion/gpu/*.cu")

if(APPLE OR WIN32)
  list(REMOVE_ITEM kernel_cu "fusion/gpu/fusion_group_kernel.cu")
endif()

if(NOT WITH_DGC)
  list(REMOVE_ITEM kernel_cu "gpu/dgc_kernel.cu")
endif()

if(DEFINED REDUCE_INFERENCE_LIB_SIZE)
  list(FILTER kernel_cu EXCLUDE REGEX ".*_grad_kernel\\.cc$")
  list(FILTER kernel_cu EXCLUDE REGEX ".*_grad_kernel\\.cu$")
endif()

if(WITH_CUTLASS)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/conv2d/generated_tmp"
    COMMAND ${PYTHON_EXECUTABLE} "conv2d_bias_act.py"
    COMMAND ${PYTHON_EXECUTABLE} "conv2d_bias_residual.py"
    COMMAND ${PYTHON_EXECUTABLE} "conv2d_depthwise_bias_act.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/conv2d")
  set(generated_tmp_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/conv2d/generated_tmp
  )
  set(generated_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/conv2d/generated)
  file(GLOB con2d_generated_files ${generated_tmp_dir}/*.cu)

  if(EXISTS ${generated_dir})
    foreach(gen_file ${con2d_generated_files})
      string(REPLACE "generated_tmp" "generated" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                              "${gen_file}" "${now_file}")
    endforeach()
    message("copy if different ${generated_dir}")
  else()
    foreach(gen_file ${con2d_generated_files})
      string(REPLACE "generated_tmp" "generated" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${gen_file}"
                              "${now_file}")
    endforeach()
    message("copy ${generated_dir}")
  endif()

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE}
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/generate_kernels.py
      --cuda_arch "${NVCC_ARCH_BIN}" --gen_dir "autogen_tmp"
    RESULT_VARIABLE memory_efficient_attention_gen_res)

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE}
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/generate_variable_forward_kernels.py
      --cuda_arch "${NVCC_ARCH_BIN}" --gen_dir "autogen_variable_tmp"
    RESULT_VARIABLE memory_efficient_attention_gen_res)

  if(NOT memory_efficient_attention_gen_res EQUAL 0)
    message(
      FATAL_ERROR
        "The memory efficient attention kernel generation errors with NVCC_ARCH_BIN=${NVCC_ARCH_BIN}"
    )
  endif()

  set(autogen_tmp_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_tmp
  )
  set(autogen_variable_tmp_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable_tmp
  )
  set(autogen_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen
  )
  set(autogen_variable_dir
      ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable
  )

  file(GLOB generated_files ${autogen_tmp_dir}/*.h ${autogen_tmp_dir}/impl/*.cu)

  file(GLOB variable_generated_files ${autogen_variable_tmp_dir}/*.h
       ${autogen_variable_tmp_dir}/impl/*.cu)

  if(EXISTS ${autogen_dir})
    foreach(gen_file ${generated_files})
      string(REPLACE "autogen_tmp" "autogen" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                              "${gen_file}" "${now_file}")
    endforeach()
    message("copy if different ${autogen_dir}")
  else()
    foreach(gen_file ${generated_files})
      string(REPLACE "autogen_tmp" "autogen" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${gen_file}"
                              "${now_file}")
    endforeach()
    message("copy ${autogen_dir}")
  endif()

  if(EXISTS ${autogen_variable_dir})
    foreach(gen_file ${variable_generated_files})
      string(REPLACE "autogen_variable_tmp" "autogen_variable" now_file
                     ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                              "${gen_file}" "${now_file}")
    endforeach()
    message("copy if different ${autogen_variable_dir}")
  else()
    foreach(gen_file ${variable_generated_files})
      string(REPLACE "autogen_variable_tmp" "autogen_variable" now_file
                     ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${gen_file}"
                              "${now_file}")
    endforeach()
    message("copy ${autogen_variable_dir}")
  endif()

  file(
    REMOVE_RECURSE
    ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_tmp
  )
  file(
    REMOVE_RECURSE
    ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable_tmp
  )

  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E make_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen_tmp"
    COMMAND ${PYTHON_EXECUTABLE} generic_mixed_gemm_kernelLauncher.py
            --cuda_arch "${NVCC_ARCH_BIN}"
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/cutlass_kernels/fpA_intB_gemm"
  )
  set(fpA_intB_gemm_autogen_tmp_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen_tmp
  )
  set(fpA_intB_gemm_autogen_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen
  )

  file(GLOB fpA_intB_gemm_autogen_files ${fpA_intB_gemm_autogen_tmp_dir}/*.h
       ${fpA_intB_gemm_autogen_tmp_dir}/*.cu)

  if(EXISTS ${fpA_intB_gemm_autogen_dir})
    foreach(gen_file ${fpA_intB_gemm_autogen_files})
      string(REPLACE "autogen_tmp" "autogen" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
                              "${gen_file}" "${now_file}")
    endforeach()
    message("copy if different ${fpA_intB_gemm_autogen_dir}")
  else()
    foreach(gen_file ${fpA_intB_gemm_autogen_files})
      string(REPLACE "autogen_tmp" "autogen" now_file ${gen_file})
      execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${gen_file}"
                              "${now_file}")
    endforeach()
    message("copy ${fpA_intB_gemm_autogen_dir}")
  endif()

  file(
    GLOB cutlass_cu
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "fusion/cutlass/conv2d/generated/*.cu"
    "fusion/cutlass/conv2d/*.cu"
    "fusion/cutlass/*.cu"
    "fusion/cutlass/memory_efficient_attention/autogen/impl/*.cu"
    "fusion/cutlass/memory_efficient_attention/autogen_variable/impl/*.cu"
    "fusion/cutlass/cutlass_kernels/fpA_intB_gemm/autogen/*.cu"
    "fusion/cutlass/cutlass_kernels/fpA_intB_gemm/*.cu")

  list(APPEND kernel_cu ${cutlass_cu})
endif()

if(NOT WITH_CUDNN_FRONTEND)
  list(
    REMOVE_ITEM
    kernel_cu
    "fusion/gpu/fused_scale_bias_relu_conv_bn_kernel.cu"
    "fusion/gpu/fused_scale_bias_add_relu_kernel.cu"
    "fusion/gpu/fused_dconv_drelu_dbn_kernel.cu"
    "fusion/gpu/fused_dot_product_attention_op.cu"
    "fusion/gpu/max_pool2d_v2_grad_kernel.cu"
    "fusion/gpu/max_pool2d_v2_kernel.cu")
endif()

set(cc_search_pattern
    "*.cc"
    "cpu/*.cc"
    "legacy/*.cc"
    "legacy/cpu/*.cc"
    "selected_rows/*.cc"
    "selected_rows/cpu/*.cc"
    "sparse/*.cc"
    "sparse/cpu/*.cc"
    "legacy/*.cc"
    "legacy/cpu/*.cc"
    "strings/*.cc"
    "strings/cpu/*.cc"
    "fusion/*.cc"
    "stride/*.cc"
    "fusion/cpu/*.cc")

if(WITH_MKLDNN)
  set(cc_search_pattern ${cc_search_pattern} "legacy/onednn/*.cc" "onednn/*.cc"
                        "fusion/onednn/*.cc")
endif()

if(WITH_CUSTOM_DEVICE)
  set(cc_search_pattern ${cc_search_pattern} "custom/*.cc")
endif()

file(
  GLOB kernel_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  ${cc_search_pattern})

if(DEFINED REDUCE_INFERENCE_LIB_SIZE)
  list(FILTER kernel_cc EXCLUDE REGEX ".*_grad_kernel\\.cc$")
endif()

if(NOT
   (WITH_AVX
    AND AVX512F_FOUND
    AND AVX512F_FLAG
    AND WITH_MKL))
  list(REMOVE_ITEM kernel_cc "fusion/cpu/self_dp_attention_kernel.cc")
endif()

file(
  GLOB kernel_xpu
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "xpu/*.cc" "legacy/xpu/*.cc" "selected_rows/xpu/*.cc" "fusion/xpu/*.cc"
  "sparse/xpu/*.cc")

if(WITH_GPU OR WITH_ROCM)
  collect_srcs(kernels_srcs SRCS ${kernel_cu})
  kernel_declare("${kernel_cu}")
endif()

if(WITH_XPU)
  if(WITH_XPU_KP)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kps/
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kps/)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/legacy/kps/
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/kps/)
    file(GLOB kernel_xpu_kps "${CMAKE_CURRENT_BINARY_DIR}/kps/*.cu")
    foreach(kernel ${kernel_xpu_kps})
      get_filename_component(name ${kernel} NAME_WE)
      file(RENAME ${kernel} "${CMAKE_CURRENT_BINARY_DIR}/kps/${name}.kps")
    endforeach()
    file(GLOB kernel_xpu_kps "${CMAKE_CURRENT_BINARY_DIR}/kps/*.kps")
    collect_generated_srcs(kernels_srcs SRCS ${kernel_xpu_kps})

    foreach(kernel ${kernel_cc})
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${kernel}
                     ${CMAKE_CURRENT_BINARY_DIR}/${kernel} COPYONLY)
    endforeach()
    file(GLOB_RECURSE kernel_xpu_cc "${CMAKE_CURRENT_BINARY_DIR}/*.cc")
    collect_generated_srcs(kernels_srcs SRCS ${kernel_xpu_cc})
    set(kernel_cc "")

  endif()
  collect_srcs(kernels_srcs SRCS ${kernel_xpu})
  kernel_declare("${kernel_xpu}")
  kernel_declare("${kernel_xpu_kps}")
  kernel_declare("${kernel_xpu_cc}")
endif()

collect_srcs(kernels_srcs SRCS ${kernel_cc})
kernel_declare("${kernel_cc}")

if(NOT "${KERNEL_LIST}" STREQUAL "")
  prune_declaration_h()
endif()

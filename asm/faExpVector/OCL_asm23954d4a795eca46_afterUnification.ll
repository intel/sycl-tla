; ------------------------------------------------
; OCL_asm23954d4a795eca46_afterUnification.ll
; LLVM major version: 16
; ------------------------------------------------
; ModuleID = '<origin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [3 x i64] }
%class.__generated_ = type { i8 addrspace(1)*, i64, %"class.sycl::_V1::range", i8 addrspace(1)*, i64, %"class.sycl::_V1::range" }
%"class.sycl::_V1::detail::RoundedRangeIDGenerator" = type <{ %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", i8, [7 x i8] }>
%"class.sycl::_V1::detail::RoundedRangeKernel" = type { %"class.sycl::_V1::range", %class._ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_EUlSA_E_ }
%class._ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_EUlSA_E_ = type { i8 addrspace(4)*, i64, %"class.sycl::_V1::range", i8 addrspace(4)*, i64, %"class.sycl::_V1::range" }
%"class.sycl::_V1::range.0" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
%class.__generated_.2 = type <{ i8 addrspace(1)*, i16, [6 x i8] }>
%class.__generated_.9 = type <{ i8 addrspace(1)*, i32, [4 x i8] }>
%class.__generated_.12 = type <{ i8 addrspace(1)*, i8, [7 x i8] }>
%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" = type <{ i64, float, float, i32, float, float, [4 x i8] }>
%"struct.cutlass::reference::device::detail::RandomUniformFunc" = type { %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params", %"class.oneapi::mkl::rng::device::uniform", %"class.oneapi::mkl::rng::device::philox4x32x10", [4 x i8] }
%"class.oneapi::mkl::rng::device::uniform" = type { %"class.oneapi::mkl::rng::device::detail::distribution_base" }
%"class.oneapi::mkl::rng::device::detail::distribution_base" = type { float, float }
%"class.oneapi::mkl::rng::device::philox4x32x10" = type { %"class.oneapi::mkl::rng::device::detail::engine_base" }
%"class.oneapi::mkl::rng::device::detail::engine_base" = type { %"struct.oneapi::mkl::rng::device::detail::engine_state" }
%"struct.oneapi::mkl::rng::device::detail::engine_state" = type { [2 x i32], [4 x i32], i32, [4 x i32] }
%"struct.cutlass::gemm::GemmCoord" = type { %"struct.cutlass::Coord" }
%"struct.cutlass::Coord" = type { [3 x i32] }
%"class.cutlass::__generated_TensorRef" = type { i8 addrspace(1)*, %"class.sycl::_V1::range.0" }
%structtype.0 = type { i64 }
%"struct.cutlass::bfloat16_t" = type { i16 }

@gVar = internal global [36 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@gVar.61 = internal global [24 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@0 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@1 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@2 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@3 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@4 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@5 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@6 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@7 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@8 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@9 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@10 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@11 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@12 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@llvm.used = appending global [2 x i8*] [i8* getelementptr inbounds ([36 x i8], [36 x i8]* @gVar, i32 0, i32 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @gVar.61, i32 0, i32 0)], section "llvm.metadata"

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE(%"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %0, %class.__generated_* byval(%class.__generated_) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i64 %const_reg_qword12, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i32 0
  %4 = extractelement <3 x i32> %globalSize, i32 1
  %5 = extractelement <3 x i32> %globalSize, i32 2
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %globalOffset, i32 1
  %8 = extractelement <3 x i32> %globalOffset, i32 2
  %9 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %12 = extractelement <8 x i32> %r0, i32 0
  %13 = extractelement <8 x i32> %r0, i32 1
  %14 = extractelement <8 x i32> %r0, i32 2
  %15 = extractelement <8 x i32> %r0, i32 3
  %16 = extractelement <8 x i32> %r0, i32 4
  %17 = extractelement <8 x i32> %r0, i32 5
  %18 = extractelement <8 x i32> %r0, i32 6
  %19 = extractelement <8 x i32> %r0, i32 7
  %20 = alloca %"class.sycl::_V1::detail::RoundedRangeIDGenerator", align 8, !spirv.Decorations !0
  %21 = alloca %"class.sycl::_V1::range", align 8, !spirv.Decorations !0
  %22 = alloca %"class.sycl::_V1::detail::RoundedRangeKernel", align 8, !spirv.Decorations !0
  %23 = bitcast %"class.sycl::_V1::detail::RoundedRangeKernel"* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 104, i8* nonnull %23)
  %_alloca.sroa.0.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 0, i32 0, i32 0, i64 0
  store i64 %const_reg_qword, i64* %_alloca.sroa.0.0..sroa_idx, align 8
  %_alloca.sroa.3.0..sroa_idx93 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 0, i32 0, i32 0, i64 1
  store i64 %const_reg_qword1, i64* %_alloca.sroa.3.0..sroa_idx93, align 8
  %_alloca.sroa.4.0..sroa_idx96 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 0, i32 0, i32 0, i64 2
  store i64 %const_reg_qword2, i64* %_alloca.sroa.4.0..sroa_idx96, align 8
  %_alloca81.sroa.0.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1
  %_alloca81.sroa.0.0..sroa_cast = bitcast %class._ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_EUlSA_E_* %_alloca81.sroa.0.0..sroa_idx to i64*
  store i64 %const_reg_qword3, i64* %_alloca81.sroa.0.0..sroa_cast, align 8
  %_alloca81.sroa.2.0..sroa_idx82 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 1
  store i64 %const_reg_qword4, i64* %_alloca81.sroa.2.0..sroa_idx82, align 8
  %_alloca81.sroa.3.0..sroa_idx83 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 2, i32 0, i32 0, i64 0
  store i64 %const_reg_qword5, i64* %_alloca81.sroa.3.0..sroa_idx83, align 8
  %_alloca81.sroa.4.0..sroa_idx84 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 2, i32 0, i32 0, i64 1
  store i64 %const_reg_qword6, i64* %_alloca81.sroa.4.0..sroa_idx84, align 8
  %_alloca81.sroa.5.0..sroa_idx85 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 2, i32 0, i32 0, i64 2
  store i64 %const_reg_qword7, i64* %_alloca81.sroa.5.0..sroa_idx85, align 8
  %_alloca81.sroa.6.0..sroa_idx86 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 3
  %_alloca81.sroa.6.0..sroa_cast = bitcast i8 addrspace(4)** %_alloca81.sroa.6.0..sroa_idx86 to i64*
  store i64 %const_reg_qword8, i64* %_alloca81.sroa.6.0..sroa_cast, align 8
  %_alloca81.sroa.7.0..sroa_idx87 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 4
  store i64 %const_reg_qword9, i64* %_alloca81.sroa.7.0..sroa_idx87, align 8
  %_alloca81.sroa.8.0..sroa_idx88 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 5, i32 0, i32 0, i64 0
  store i64 %const_reg_qword10, i64* %_alloca81.sroa.8.0..sroa_idx88, align 8
  %_alloca81.sroa.9.0..sroa_idx89 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 5, i32 0, i32 0, i64 1
  store i64 %const_reg_qword11, i64* %_alloca81.sroa.9.0..sroa_idx89, align 8
  %_alloca81.sroa.10.0..sroa_idx90 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 5, i32 0, i32 0, i64 2
  store i64 %const_reg_qword12, i64* %_alloca81.sroa.10.0..sroa_idx90, align 8
  %24 = zext i32 %19 to i64
  %25 = zext i32 %11 to i64
  %26 = mul nuw i64 %25, %24
  %27 = zext i16 %localIdZ to i64
  %28 = add nuw i64 %26, %27
  %29 = zext i32 %8 to i64
  %30 = add nuw i64 %28, %29
  %31 = zext i32 %18 to i64
  %32 = zext i32 %10 to i64
  %33 = mul nuw i64 %32, %31
  %34 = zext i16 %localIdY to i64
  %35 = add nuw i64 %33, %34
  %36 = zext i32 %7 to i64
  %37 = add nuw i64 %35, %36
  %38 = zext i32 %13 to i64
  %39 = zext i32 %9 to i64
  %40 = mul nuw i64 %39, %38
  %41 = zext i16 %localIdX to i64
  %42 = add nuw i64 %40, %41
  %43 = zext i32 %6 to i64
  %44 = add nuw i64 %42, %43
  %45 = zext i32 %5 to i64
  %46 = zext i32 %4 to i64
  %47 = zext i32 %3 to i64
  %48 = bitcast %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 104, i8* nonnull %48)
  %49 = bitcast %"class.sycl::_V1::range"* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %49)
  %50 = getelementptr inbounds %"class.sycl::_V1::range", %"class.sycl::_V1::range"* %21, i64 0, i32 0, i32 0, i64 0
  store i64 %30, i64* %50, align 8
  %51 = getelementptr inbounds %"class.sycl::_V1::range", %"class.sycl::_V1::range"* %21, i64 0, i32 0, i32 0, i64 1
  store i64 %37, i64* %51, align 8
  %52 = getelementptr inbounds %"class.sycl::_V1::range", %"class.sycl::_V1::range"* %21, i64 0, i32 0, i32 0, i64 2
  store i64 %44, i64* %52, align 8
  %53 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 0
  store i64 %30, i64* %53, align 8
  %54 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 1
  store i64 %37, i64* %54, align 8
  %55 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 2
  store i64 %44, i64* %55, align 8
  %56 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 1, i32 0, i32 0, i64 0
  store i64 %30, i64* %56, align 8
  %57 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 1, i32 0, i32 0, i64 1
  store i64 %37, i64* %57, align 8
  %58 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 1, i32 0, i32 0, i64 2
  store i64 %44, i64* %58, align 8
  %_alloca.sroa.0.0..sroa_idx91 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 2, i32 0, i32 0, i64 0
  store i64 %const_reg_qword, i64* %_alloca.sroa.0.0..sroa_idx91, align 8
  %_alloca.sroa.3.0..sroa_idx94 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 2, i32 0, i32 0, i64 1
  store i64 %const_reg_qword1, i64* %_alloca.sroa.3.0..sroa_idx94, align 8
  %_alloca.sroa.4.0..sroa_idx97 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 2, i32 0, i32 0, i64 2
  store i64 %const_reg_qword2, i64* %_alloca.sroa.4.0..sroa_idx97, align 8
  %59 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 3, i32 0, i32 0, i64 0
  store i64 %45, i64* %59, align 8
  %60 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 3, i32 0, i32 0, i64 1
  store i64 %46, i64* %60, align 8
  %61 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 3, i32 0, i32 0, i64 2
  store i64 %47, i64* %61, align 8
  br label %62

62:                                               ; preds = %66, %2
  %63 = phi i8 [ 0, %2 ], [ %73, %66 ]
  %64 = phi i32 [ 0, %2 ], [ %74, %66 ]
  %65 = icmp ult i32 %64, 3
  br i1 %65, label %66, label %75

66:                                               ; preds = %62
  %67 = zext i32 %64 to i64
  %68 = getelementptr inbounds %"class.sycl::_V1::range", %"class.sycl::_V1::range"* %21, i64 0, i32 0, i32 0, i64 %67
  %69 = load i64, i64* %68, align 8
  %70 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 0, i32 0, i32 0, i64 %67
  %71 = load i64, i64* %70, align 8
  %72 = icmp ult i64 %69, %71
  %73 = select i1 %72, i8 %63, i8 1
  %74 = add nuw nsw i32 %64, 1, !spirv.Decorations !833
  br label %62

75:                                               ; preds = %62
  %76 = bitcast %"class.sycl::_V1::range"* %21 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %76)
  %77 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 1
  %78 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 2
  %79 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 3
  %80 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 4
  %81 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 1
  %82 = load i8 addrspace(4)*, i8 addrspace(4)** %79, align 8
  %83 = load i64, i64* %80, align 8
  %84 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 5, i32 0, i32 0, i64 0
  %85 = load i64, i64* %84, align 8
  %86 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 0
  %87 = load i8 addrspace(4)*, i8 addrspace(4)** %86, align 8
  %88 = load i64, i64* %81, align 8
  %89 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeKernel", %"class.sycl::_V1::detail::RoundedRangeKernel"* %22, i64 0, i32 1, i32 2, i32 0, i32 0, i64 0
  %90 = load i64, i64* %89, align 8
  br label %91

91:                                               ; preds = %132, %75
  %92 = phi i8 [ %133, %132 ], [ %63, %75 ]
  %93 = icmp eq i8 %92, 0
  br i1 %93, label %94, label %134

94:                                               ; preds = %91
  %95 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 0
  %96 = load i64, i64* %95, align 8
  %97 = load i64, i64* %77, align 8
  %98 = load i64, i64* %78, align 8
  %99 = icmp ult i64 %96, 2147483648
  call void @llvm.assume(i1 %99)
  %100 = icmp ult i64 %97, 2147483648
  call void @llvm.assume(i1 %100)
  %101 = icmp ult i64 %98, 2147483648
  call void @llvm.assume(i1 %101)
  %102 = mul i64 %83, %98
  %103 = mul i64 %85, %97
  %104 = getelementptr i8, i8 addrspace(4)* %82, i64 %102
  %105 = getelementptr i8, i8 addrspace(4)* %104, i64 %103
  %106 = getelementptr i8, i8 addrspace(4)* %105, i64 %96
  %107 = addrspacecast i8 addrspace(4)* %106 to i8 addrspace(1)*
  %108 = load i8, i8 addrspace(1)* %107, align 1
  %109 = mul i64 %88, %98
  %110 = mul i64 %90, %97
  %111 = getelementptr i8, i8 addrspace(4)* %87, i64 %109
  %112 = getelementptr i8, i8 addrspace(4)* %111, i64 %110
  %113 = getelementptr i8, i8 addrspace(4)* %112, i64 %96
  %114 = addrspacecast i8 addrspace(4)* %113 to i8 addrspace(1)*
  store i8 %108, i8 addrspace(1)* %114, align 1
  br label %115

115:                                              ; preds = %128, %94
  %116 = phi i32 [ 0, %94 ], [ %131, %128 ]
  %117 = icmp ult i32 %116, 3
  br i1 %117, label %118, label %132

118:                                              ; preds = %115
  %119 = zext i32 %116 to i64
  %120 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 3, i32 0, i32 0, i64 %119
  %121 = load i64, i64* %120, align 8
  %122 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 0, i32 0, i32 0, i64 %119
  %123 = load i64, i64* %122, align 8
  %124 = add i64 %123, %121
  store i64 %124, i64* %122, align 8
  %125 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 2, i32 0, i32 0, i64 %119
  %126 = load i64, i64* %125, align 8
  %127 = icmp ult i64 %124, %126
  br i1 %127, label %132, label %128

128:                                              ; preds = %118
  %129 = getelementptr inbounds %"class.sycl::_V1::detail::RoundedRangeIDGenerator", %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20, i64 0, i32 1, i32 0, i32 0, i64 %119
  %130 = load i64, i64* %129, align 8
  store i64 %130, i64* %122, align 8
  %131 = add nuw nsw i32 %116, 1, !spirv.Decorations !833
  br label %115

132:                                              ; preds = %118, %115
  %133 = phi i8 [ 1, %115 ], [ 0, %118 ]
  br label %91

134:                                              ; preds = %91
  %135 = bitcast %"class.sycl::_V1::detail::RoundedRangeIDGenerator"* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 104, i8* nonnull %135)
  %136 = bitcast %"class.sycl::_V1::detail::RoundedRangeKernel"* %22 to i8*
  call void @llvm.lifetime.end.p0i8(i64 104, i8* nonnull %136)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail(i8 addrspace(1)* align 1 %0, i64 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i8 addrspace(1)* align 1 %3, i64 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i32 %bufferOffset, i32 %bufferOffset6, i32 %bindlessOffset, i32 %bindlessOffset7, i32 %bindlessOffset8) #0 {
  %7 = extractelement <3 x i32> %globalOffset, i32 0
  %8 = extractelement <3 x i32> %globalOffset, i32 1
  %9 = extractelement <3 x i32> %globalOffset, i32 2
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %12 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %13 = extractelement <8 x i32> %r0, i32 0
  %14 = extractelement <8 x i32> %r0, i32 1
  %15 = extractelement <8 x i32> %r0, i32 2
  %16 = extractelement <8 x i32> %r0, i32 3
  %17 = extractelement <8 x i32> %r0, i32 4
  %18 = extractelement <8 x i32> %r0, i32 5
  %19 = extractelement <8 x i32> %r0, i32 6
  %20 = extractelement <8 x i32> %r0, i32 7
  %21 = mul i32 %12, %20
  %22 = zext i16 %localIdZ to i32
  %23 = add i32 %21, %22
  %24 = add i32 %23, %9
  %25 = zext i32 %24 to i64
  %26 = mul i32 %11, %19
  %27 = zext i16 %localIdY to i32
  %28 = add i32 %26, %27
  %29 = add i32 %28, %8
  %30 = zext i32 %29 to i64
  %31 = mul i32 %10, %14
  %32 = zext i16 %localIdX to i32
  %33 = add i32 %31, %32
  %34 = add i32 %33, %7
  %35 = zext i32 %34 to i64
  %36 = icmp sgt i32 %24, -1
  call void @llvm.assume(i1 %36)
  %37 = icmp sgt i32 %29, -1
  call void @llvm.assume(i1 %37)
  %38 = icmp sgt i32 %34, -1
  call void @llvm.assume(i1 %38)
  %39 = mul i64 %35, %4
  %40 = mul i64 %30, %const_reg_qword3
  %41 = getelementptr i8, i8 addrspace(1)* %3, i64 %39
  %42 = getelementptr i8, i8 addrspace(1)* %41, i64 %40
  %43 = getelementptr i8, i8 addrspace(1)* %42, i64 %25
  %44 = load i8, i8 addrspace(1)* %43, align 1
  %45 = mul i64 %35, %1
  %46 = mul i64 %30, %const_reg_qword
  %47 = getelementptr i8, i8 addrspace(1)* %0, i64 %45
  %48 = getelementptr i8, i8 addrspace(1)* %47, i64 %46
  %49 = getelementptr i8, i8 addrspace(1)* %48, i64 %25
  store i8 %44, i8 addrspace(1)* %49, align 1
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.2* byval(%class.__generated_.2) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i16 %const_reg_word, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i32 0
  %4 = extractelement <3 x i32> %globalSize, i32 1
  %5 = extractelement <3 x i32> %globalSize, i32 2
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %globalOffset, i32 1
  %8 = extractelement <3 x i32> %globalOffset, i32 2
  %9 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %12 = extractelement <8 x i32> %r0, i32 0
  %13 = extractelement <8 x i32> %r0, i32 1
  %14 = extractelement <8 x i32> %r0, i32 2
  %15 = extractelement <8 x i32> %r0, i32 3
  %16 = extractelement <8 x i32> %r0, i32 4
  %17 = extractelement <8 x i32> %r0, i32 5
  %18 = extractelement <8 x i32> %r0, i32 6
  %19 = extractelement <8 x i32> %r0, i32 7
  %20 = inttoptr i64 %const_reg_qword1 to i16 addrspace(4)*
  %21 = zext i32 %13 to i64
  %22 = zext i32 %9 to i64
  %23 = mul nuw i64 %22, %21
  %24 = zext i16 %localIdX to i64
  %25 = add nuw i64 %23, %24
  %26 = zext i32 %6 to i64
  %27 = add nuw i64 %25, %26
  %28 = zext i32 %3 to i64
  %29 = icmp ult i64 %27, %const_reg_qword
  br label %30

30:                                               ; preds = %32, %2
  %31 = phi i64 [ %27, %2 ], [ %38, %32 ]
  %.in = phi i1 [ %29, %2 ], [ %37, %32 ]
  br i1 %.in, label %32, label %39

32:                                               ; preds = %30
  %33 = icmp ult i64 %31, 2147483648
  call void @llvm.assume(i1 %33)
  %34 = getelementptr inbounds i16, i16 addrspace(4)* %20, i64 %31
  %35 = addrspacecast i16 addrspace(4)* %34 to i16 addrspace(1)*
  store i16 %const_reg_word, i16 addrspace(1)* %35, align 2
  %36 = add nuw nsw i64 %31, %28
  %37 = icmp ult i64 %36, %const_reg_qword
  %38 = select i1 %37, i64 %36, i64 %27
  br label %30

39:                                               ; preds = %30
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_(i16 addrspace(1)* align 2 %0, i16 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %globalOffset, i32 1
  %5 = extractelement <3 x i32> %globalOffset, i32 2
  %6 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %8 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %9 = extractelement <8 x i32> %r0, i32 0
  %10 = extractelement <8 x i32> %r0, i32 1
  %11 = extractelement <8 x i32> %r0, i32 2
  %12 = extractelement <8 x i32> %r0, i32 3
  %13 = extractelement <8 x i32> %r0, i32 4
  %14 = extractelement <8 x i32> %r0, i32 5
  %15 = extractelement <8 x i32> %r0, i32 6
  %16 = extractelement <8 x i32> %r0, i32 7
  %17 = mul i32 %6, %10
  %18 = zext i16 %localIdX to i32
  %19 = add i32 %17, %18
  %20 = add i32 %19, %3
  %21 = zext i32 %20 to i64
  %22 = icmp sgt i32 %20, -1
  call void @llvm.assume(i1 %22)
  %23 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 %21
  store i16 %1, i16 addrspace(1)* %23, align 2
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.9* byval(%class.__generated_.9) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i32 %const_reg_dword, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i32 0
  %4 = extractelement <3 x i32> %globalSize, i32 1
  %5 = extractelement <3 x i32> %globalSize, i32 2
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %globalOffset, i32 1
  %8 = extractelement <3 x i32> %globalOffset, i32 2
  %9 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %12 = extractelement <8 x i32> %r0, i32 0
  %13 = extractelement <8 x i32> %r0, i32 1
  %14 = extractelement <8 x i32> %r0, i32 2
  %15 = extractelement <8 x i32> %r0, i32 3
  %16 = extractelement <8 x i32> %r0, i32 4
  %17 = extractelement <8 x i32> %r0, i32 5
  %18 = extractelement <8 x i32> %r0, i32 6
  %19 = extractelement <8 x i32> %r0, i32 7
  %20 = inttoptr i64 %const_reg_qword1 to i32 addrspace(4)*
  %21 = zext i32 %13 to i64
  %22 = zext i32 %9 to i64
  %23 = mul nuw i64 %22, %21
  %24 = zext i16 %localIdX to i64
  %25 = add nuw i64 %23, %24
  %26 = zext i32 %6 to i64
  %27 = add nuw i64 %25, %26
  %28 = zext i32 %3 to i64
  %29 = icmp ult i64 %27, %const_reg_qword
  br label %30

30:                                               ; preds = %32, %2
  %31 = phi i64 [ %27, %2 ], [ %38, %32 ]
  %.in = phi i1 [ %29, %2 ], [ %37, %32 ]
  br i1 %.in, label %32, label %39

32:                                               ; preds = %30
  %33 = icmp ult i64 %31, 2147483648
  call void @llvm.assume(i1 %33)
  %34 = getelementptr inbounds i32, i32 addrspace(4)* %20, i64 %31
  %35 = addrspacecast i32 addrspace(4)* %34 to i32 addrspace(1)*
  store i32 %const_reg_dword, i32 addrspace(1)* %35, align 4
  %36 = add nuw nsw i64 %31, %28
  %37 = icmp ult i64 %36, %const_reg_qword
  %38 = select i1 %37, i64 %36, i64 %27
  br label %30

39:                                               ; preds = %30
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_(i32 addrspace(1)* align 4 %0, i32 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %globalOffset, i32 1
  %5 = extractelement <3 x i32> %globalOffset, i32 2
  %6 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %8 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %9 = extractelement <8 x i32> %r0, i32 0
  %10 = extractelement <8 x i32> %r0, i32 1
  %11 = extractelement <8 x i32> %r0, i32 2
  %12 = extractelement <8 x i32> %r0, i32 3
  %13 = extractelement <8 x i32> %r0, i32 4
  %14 = extractelement <8 x i32> %r0, i32 5
  %15 = extractelement <8 x i32> %r0, i32 6
  %16 = extractelement <8 x i32> %r0, i32 7
  %17 = mul i32 %6, %10
  %18 = zext i16 %localIdX to i32
  %19 = add i32 %17, %18
  %20 = add i32 %19, %3
  %21 = zext i32 %20 to i64
  %22 = icmp sgt i32 %20, -1
  call void @llvm.assume(i1 %22)
  %23 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %21
  store i32 %1, i32 addrspace(1)* %23, align 4
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.12* byval(%class.__generated_.12) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i8 %const_reg_byte7, i8 %const_reg_byte8, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i32 0
  %4 = extractelement <3 x i32> %globalSize, i32 1
  %5 = extractelement <3 x i32> %globalSize, i32 2
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %globalOffset, i32 1
  %8 = extractelement <3 x i32> %globalOffset, i32 2
  %9 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %12 = extractelement <8 x i32> %r0, i32 0
  %13 = extractelement <8 x i32> %r0, i32 1
  %14 = extractelement <8 x i32> %r0, i32 2
  %15 = extractelement <8 x i32> %r0, i32 3
  %16 = extractelement <8 x i32> %r0, i32 4
  %17 = extractelement <8 x i32> %r0, i32 5
  %18 = extractelement <8 x i32> %r0, i32 6
  %19 = extractelement <8 x i32> %r0, i32 7
  %20 = inttoptr i64 %const_reg_qword1 to i8 addrspace(4)*
  %21 = zext i32 %13 to i64
  %22 = zext i32 %9 to i64
  %23 = mul nuw i64 %22, %21
  %24 = zext i16 %localIdX to i64
  %25 = add nuw i64 %23, %24
  %26 = zext i32 %6 to i64
  %27 = add nuw i64 %25, %26
  %28 = zext i32 %3 to i64
  %29 = icmp ult i64 %27, %const_reg_qword
  br label %30

30:                                               ; preds = %32, %2
  %31 = phi i64 [ %27, %2 ], [ %38, %32 ]
  %.in = phi i1 [ %29, %2 ], [ %37, %32 ]
  br i1 %.in, label %32, label %39

32:                                               ; preds = %30
  %33 = icmp ult i64 %31, 2147483648
  call void @llvm.assume(i1 %33)
  %34 = getelementptr inbounds i8, i8 addrspace(4)* %20, i64 %31
  %35 = addrspacecast i8 addrspace(4)* %34 to i8 addrspace(1)*
  store i8 %const_reg_byte, i8 addrspace(1)* %35, align 1
  %36 = add nuw nsw i64 %31, %28
  %37 = icmp ult i64 %36, %const_reg_qword
  %38 = select i1 %37, i64 %36, i64 %27
  br label %30

39:                                               ; preds = %30
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_(i8 addrspace(1)* align 1 %0, i8 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %globalOffset, i32 1
  %5 = extractelement <3 x i32> %globalOffset, i32 2
  %6 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %8 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %9 = extractelement <8 x i32> %r0, i32 0
  %10 = extractelement <8 x i32> %r0, i32 1
  %11 = extractelement <8 x i32> %r0, i32 2
  %12 = extractelement <8 x i32> %r0, i32 3
  %13 = extractelement <8 x i32> %r0, i32 4
  %14 = extractelement <8 x i32> %r0, i32 5
  %15 = extractelement <8 x i32> %r0, i32 6
  %16 = extractelement <8 x i32> %r0, i32 7
  %17 = mul i32 %6, %10
  %18 = zext i16 %localIdX to i32
  %19 = add i32 %17, %18
  %20 = add i32 %19, %3
  %21 = zext i32 %20 to i64
  %22 = icmp sgt i32 %20, -1
  call void @llvm.assume(i1 %22)
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %21
  store i8 %1, i8 addrspace(1)* %23, align 1
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE(i16 addrspace(1)* align 2 %0, i64 %1, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"* byval(%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params") align 8 %2, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, float %const_reg_fp32, float %const_reg_fp321, i32 %const_reg_dword, float %const_reg_fp322, float %const_reg_fp323, i8 %const_reg_byte, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset7) #0 {
  %4 = extractelement <3 x i32> %numWorkGroups, i32 0
  %5 = extractelement <3 x i32> %numWorkGroups, i32 1
  %6 = extractelement <3 x i32> %numWorkGroups, i32 2
  %7 = extractelement <3 x i32> %localSize, i32 0
  %8 = extractelement <3 x i32> %localSize, i32 1
  %9 = extractelement <3 x i32> %localSize, i32 2
  %10 = extractelement <8 x i32> %r0, i32 0
  %11 = extractelement <8 x i32> %r0, i32 1
  %12 = extractelement <8 x i32> %r0, i32 2
  %13 = extractelement <8 x i32> %r0, i32 3
  %14 = extractelement <8 x i32> %r0, i32 4
  %15 = extractelement <8 x i32> %r0, i32 5
  %16 = extractelement <8 x i32> %r0, i32 6
  %17 = extractelement <8 x i32> %r0, i32 7
  %18 = alloca [3 x i64], align 8, !spirv.Decorations !0
  %19 = alloca [2 x i64], align 8, !spirv.Decorations !0
  %20 = alloca %"struct.cutlass::reference::device::detail::RandomUniformFunc", align 8, !spirv.Decorations !0
  %21 = bitcast %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 88, i8* nonnull %21)
  %.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 0
  store i64 %const_reg_qword, i64* %.sroa.0.0..sroa_idx, align 8
  %.sroa.5.0..sroa_idx2 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 1
  store float %const_reg_fp32, float* %.sroa.5.0..sroa_idx2, align 8
  %.sroa.7.0..sroa_idx4 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 2
  store float %const_reg_fp321, float* %.sroa.7.0..sroa_idx4, align 4
  %.sroa.9.0..sroa_idx6 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 3
  store i32 %const_reg_dword, i32* %.sroa.9.0..sroa_idx6, align 8
  %.sroa.10.0..sroa_idx8 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 4
  store float %const_reg_fp322, float* %.sroa.10.0..sroa_idx8, align 4
  %.sroa.11.0..sroa_idx10 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 5
  store float %const_reg_fp323, float* %.sroa.11.0..sroa_idx10, align 8
  %22 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 1, i32 0, i32 0
  store float %const_reg_fp321, float* %22, align 8
  %23 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 1, i32 0, i32 1
  store float %const_reg_fp32, float* %23, align 4
  %24 = bitcast [2 x i64]* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %24)
  %25 = getelementptr inbounds [2 x i64], [2 x i64]* %19, i64 0, i64 0
  store i64 0, i64* %25, align 8
  %26 = getelementptr inbounds [2 x i64], [2 x i64]* %19, i64 0, i64 1
  %27 = zext i16 %localIdX to i64
  %28 = zext i32 %11 to i64
  %29 = icmp sgt i32 %11, -1
  call void @llvm.assume(i1 %29)
  %30 = zext i32 %7 to i64
  %31 = icmp sgt i32 %7, -1
  call void @llvm.assume(i1 %31)
  %32 = mul nuw nsw i64 %28, %30, !spirv.Decorations !833
  %33 = add nuw nsw i64 %32, %27, !spirv.Decorations !833
  %34 = and i64 %33, 4294967295
  store i64 %34, i64* %26, align 8
  %35 = trunc i64 %const_reg_qword to i32
  %36 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 0, i64 0
  store i32 %35, i32* %36, align 8
  %37 = lshr i64 %const_reg_qword, 32
  %38 = trunc i64 %37 to i32
  %39 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  store i32 %38, i32* %39, align 4
  %40 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1
  %41 = bitcast [4 x i32]* %40 to i8*
  %42 = getelementptr inbounds [36 x i8], [36 x i8]* @gVar, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(36) %41, i8* noundef nonnull align 8 dereferenceable(36) %42, i64 36, i1 false)
  %43 = bitcast [3 x i64]* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %43)
  %44 = bitcast [3 x i64]* %18 to i8*
  %45 = getelementptr inbounds [24 x i8], [24 x i8]* @gVar.61, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %44, i8* noundef nonnull align 8 dereferenceable(24) %45, i64 24, i1 false)
  br label %46

46:                                               ; preds = %46, %3
  %47 = phi i1 [ false, %46 ], [ true, %3 ]
  %48 = phi i64 [ 1, %46 ], [ 0, %3 ]
  %49 = phi i32 [ %56, %46 ], [ 0, %3 ]
  %50 = getelementptr inbounds [2 x i64], [2 x i64]* %19, i64 0, i64 %48
  %51 = load i64, i64* %50, align 8
  %52 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 %48
  store i64 %51, i64* %52, align 8
  %53 = icmp eq i64 %51, 0
  %54 = trunc i64 %48 to i32
  %55 = add nuw nsw i32 %54, 1, !spirv.Decorations !833
  %56 = select i1 %53, i32 %49, i32 %55
  br i1 %47, label %46, label %57

57:                                               ; preds = %46
  %58 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 1
  %59 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 2
  %60 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 3
  %61 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 2
  %62 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 1
  %63 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 2
  %64 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 3
  br label %NodeBlock

NodeBlock:                                        ; preds = %57
  %Pivot = icmp slt i32 %56, 1
  br i1 %Pivot, label %LeafBlock, label %LeafBlock35

LeafBlock35:                                      ; preds = %NodeBlock
  %SwitchLeaf36 = icmp eq i32 %56, 1
  br i1 %SwitchLeaf36, label %68, label %65

LeafBlock:                                        ; preds = %NodeBlock
  %SwitchLeaf = icmp eq i32 %56, 0
  br i1 %SwitchLeaf, label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit, label %65

65:                                               ; preds = %LeafBlock35, %LeafBlock
  %66 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 0
  %67 = load i64, i64* %66, align 8
  br label %73

68:                                               ; preds = %LeafBlock35
  %69 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 0
  %70 = load i64, i64* %69, align 8
  %71 = icmp eq i64 %70, 0
  br i1 %71, label %72, label %73

72:                                               ; preds = %68
  store i32 0, i32* %61, align 8
  br label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit

73:                                               ; preds = %68, %65
  %74 = phi i64 [ %67, %65 ], [ %70, %68 ]
  %75 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 0
  store i64 %74, i64* %75, align 8
  br label %76

76:                                               ; preds = %235, %73
  %77 = phi i64 [ 0, %73 ], [ %240, %235 ]
  %78 = phi i32 [ %56, %73 ], [ %236, %235 ]
  %79 = icmp sgt i32 %78, 0
  br i1 %79, label %235, label %80

80:                                               ; preds = %76
  %81 = trunc i64 %74 to i32
  %82 = and i32 %81, 3
  %83 = sub nuw nsw i32 4, %82, !spirv.Decorations !833
  store i32 %83, i32* %61, align 8
  %84 = load i64, i64* %75, align 8
  %85 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 1
  %86 = load i64, i64* %85, align 8
  %87 = and i64 %84, 4294967295
  %88 = mul nuw i64 %87, 3528531795, !spirv.Decorations !836
  %89 = and i64 %86, 4294967295
  %90 = mul nuw i64 %89, 3449720151, !spirv.Decorations !836
  %91 = xor i64 %84, %90
  %92 = lshr i64 %91, 32
  %93 = xor i64 %86, %88
  %94 = xor i64 %93, %const_reg_qword
  %95 = lshr i64 %94, 32
  %96 = add i32 %35, -1640531527
  %97 = add i32 %38, -1150833019
  %.masked = and i64 %const_reg_qword, 4294967295
  %98 = xor i64 %92, %.masked
  %99 = mul nuw i64 %98, 3528531795, !spirv.Decorations !836
  %100 = lshr i64 %99, 32
  %101 = mul nuw i64 %95, 3449720151, !spirv.Decorations !836
  %102 = lshr i64 %101, 32
  %103 = xor i64 %90, %102
  %104 = trunc i64 %103 to i32
  %105 = xor i32 %96, %104
  %106 = xor i64 %88, %100
  %107 = trunc i64 %106 to i32
  %108 = xor i32 %97, %107
  %109 = add i32 %35, 1013904242
  %110 = add i32 %38, 1993301258
  %111 = zext i32 %105 to i64
  %112 = mul nuw i64 %111, 3528531795, !spirv.Decorations !836
  %113 = lshr i64 %112, 32
  %114 = zext i32 %108 to i64
  %115 = mul nuw i64 %114, 3449720151, !spirv.Decorations !836
  %116 = lshr i64 %115, 32
  %117 = xor i64 %101, %116
  %118 = trunc i64 %117 to i32
  %119 = xor i32 %109, %118
  %120 = xor i64 %99, %113
  %121 = trunc i64 %120 to i32
  %122 = xor i32 %110, %121
  %123 = add i32 %35, -626627285
  %124 = add i32 %38, 842468239
  %125 = zext i32 %119 to i64
  %126 = mul nuw i64 %125, 3528531795, !spirv.Decorations !836
  %127 = lshr i64 %126, 32
  %128 = zext i32 %122 to i64
  %129 = mul nuw i64 %128, 3449720151, !spirv.Decorations !836
  %130 = lshr i64 %129, 32
  %131 = xor i64 %115, %130
  %132 = trunc i64 %131 to i32
  %133 = xor i32 %123, %132
  %134 = xor i64 %112, %127
  %135 = trunc i64 %134 to i32
  %136 = xor i32 %124, %135
  %137 = add i32 %35, 2027808484
  %138 = add i32 %38, -308364780
  %139 = zext i32 %133 to i64
  %140 = mul nuw i64 %139, 3528531795, !spirv.Decorations !836
  %141 = lshr i64 %140, 32
  %142 = zext i32 %136 to i64
  %143 = mul nuw i64 %142, 3449720151, !spirv.Decorations !836
  %144 = lshr i64 %143, 32
  %145 = xor i64 %129, %144
  %146 = trunc i64 %145 to i32
  %147 = xor i32 %137, %146
  %148 = xor i64 %126, %141
  %149 = trunc i64 %148 to i32
  %150 = xor i32 %138, %149
  %151 = add i32 %35, 387276957
  %152 = add i32 %38, -1459197799
  %153 = zext i32 %147 to i64
  %154 = mul nuw i64 %153, 3528531795, !spirv.Decorations !836
  %155 = lshr i64 %154, 32
  %156 = zext i32 %150 to i64
  %157 = mul nuw i64 %156, 3449720151, !spirv.Decorations !836
  %158 = lshr i64 %157, 32
  %159 = xor i64 %143, %158
  %160 = trunc i64 %159 to i32
  %161 = xor i32 %151, %160
  %162 = xor i64 %140, %155
  %163 = trunc i64 %162 to i32
  %164 = xor i32 %152, %163
  %165 = add i32 %35, -1253254570
  %166 = add i32 %38, 1684936478
  %167 = zext i32 %161 to i64
  %168 = mul nuw i64 %167, 3528531795, !spirv.Decorations !836
  %169 = lshr i64 %168, 32
  %170 = zext i32 %164 to i64
  %171 = mul nuw i64 %170, 3449720151, !spirv.Decorations !836
  %172 = lshr i64 %171, 32
  %173 = xor i64 %157, %172
  %174 = trunc i64 %173 to i32
  %175 = xor i32 %165, %174
  %176 = xor i64 %154, %169
  %177 = trunc i64 %176 to i32
  %178 = xor i32 %166, %177
  %179 = add i32 %35, 1401181199
  %180 = add i32 %38, 534103459
  %181 = zext i32 %175 to i64
  %182 = mul nuw i64 %181, 3528531795, !spirv.Decorations !836
  %183 = lshr i64 %182, 32
  %184 = zext i32 %178 to i64
  %185 = mul nuw i64 %184, 3449720151, !spirv.Decorations !836
  %186 = lshr i64 %185, 32
  %187 = xor i64 %171, %186
  %188 = trunc i64 %187 to i32
  %189 = xor i32 %179, %188
  %190 = xor i64 %168, %183
  %191 = trunc i64 %190 to i32
  %192 = xor i32 %180, %191
  %193 = add i32 %35, -239350328
  %194 = add i32 %38, -616729560
  %195 = zext i32 %189 to i64
  %196 = mul nuw i64 %195, 3528531795, !spirv.Decorations !836
  %197 = lshr i64 %196, 32
  %198 = zext i32 %192 to i64
  %199 = mul nuw i64 %198, 3449720151, !spirv.Decorations !836
  %200 = lshr i64 %199, 32
  %201 = xor i64 %185, %200
  %202 = trunc i64 %201 to i32
  %203 = xor i32 %193, %202
  %204 = xor i64 %182, %197
  %205 = trunc i64 %204 to i32
  %206 = xor i32 %194, %205
  %207 = add i32 %35, -1879881855
  %208 = add i32 %38, -1767562579
  %209 = zext i32 %203 to i64
  %210 = mul nuw i64 %209, 3528531795, !spirv.Decorations !836
  %211 = trunc i64 %210 to i32
  %212 = lshr i64 %210, 32
  %213 = zext i32 %206 to i64
  %214 = mul nuw i64 %213, 3449720151, !spirv.Decorations !836
  %215 = trunc i64 %214 to i32
  %216 = lshr i64 %214, 32
  %217 = xor i64 %199, %216
  %218 = trunc i64 %217 to i32
  %219 = xor i32 %207, %218
  %220 = xor i64 %196, %212
  %221 = trunc i64 %220 to i32
  %222 = xor i32 %208, %221
  %223 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 0
  store i32 %219, i32* %223, align 4
  store i32 %215, i32* %62, align 8
  store i32 %222, i32* %63, align 4
  store i32 %211, i32* %64, align 8
  %224 = add i64 %84, 1
  %225 = icmp eq i64 %224, 0
  %226 = zext i1 %225 to i64
  %227 = add i64 %86, %226
  %228 = trunc i64 %224 to i32
  %229 = getelementptr inbounds [4 x i32], [4 x i32]* %40, i64 0, i64 0
  store i32 %228, i32* %229, align 8
  %230 = lshr i64 %224, 32
  %231 = trunc i64 %230 to i32
  store i32 %231, i32* %58, align 4
  %232 = trunc i64 %227 to i32
  store i32 %232, i32* %59, align 8
  %233 = lshr i64 %227, 32
  %234 = trunc i64 %233 to i32
  store i32 %234, i32* %60, align 4
  br label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit

235:                                              ; preds = %76
  %236 = add nsw i32 %78, -1, !spirv.Decorations !837
  %237 = zext i32 %236 to i64
  %238 = getelementptr inbounds [3 x i64], [3 x i64]* %18, i64 0, i64 %237
  %239 = load i64, i64* %238, align 8
  %240 = shl i64 %239, 62
  %241 = lshr i64 %239, 2
  %242 = or i64 %241, %77
  store i64 %242, i64* %238, align 8
  br label %76

_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit: ; preds = %LeafBlock, %72, %80
  %243 = bitcast [3 x i64]* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %243)
  %244 = bitcast [2 x i64]* %19 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %244)
  %245 = zext i16 %localIdX to i64
  %246 = zext i32 %11 to i64
  %247 = icmp sgt i32 %11, -1
  call void @llvm.assume(i1 %247)
  %248 = zext i32 %7 to i64
  %249 = icmp sgt i32 %7, -1
  call void @llvm.assume(i1 %249)
  %250 = mul nuw nsw i64 %246, %248, !spirv.Decorations !833
  %251 = add nuw nsw i64 %250, %245, !spirv.Decorations !833
  %252 = and i64 %251, 4294967295
  %253 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2
  %254 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 3
  %255 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 4
  %256 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 0, i32 5
  br label %257

257:                                              ; preds = %782, %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit
  %258 = phi i64 [ %252, %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit ], [ %788, %782 ]
  %259 = icmp ult i64 %258, %1
  br i1 %259, label %260, label %789

260:                                              ; preds = %257
  %261 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 1, i32 0, i32 0
  %262 = load float, float* %261, align 8, !noalias !838
  %263 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 1, i32 0, i32 1
  %264 = load float, float* %263, align 4, !noalias !838
  %265 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 2
  %266 = load i32, i32* %265, align 8, !noalias !841
  %267 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3
  br label %268

268:                                              ; preds = %275, %260
  %269 = phi i32 [ undef, %260 ], [ %280, %275 ]
  %270 = phi i1 [ true, %260 ], [ false, %275 ]
  %271 = phi i1 [ false, %260 ], [ true, %275 ]
  %272 = phi i32 [ %266, %260 ], [ %276, %275 ]
  %273 = icmp ne i32 %272, 0
  %274 = and i1 %273, %270
  br i1 %274, label %275, label %281

275:                                              ; preds = %268
  %276 = add nsw i32 %272, -1, !spirv.Decorations !837
  %277 = sub nsw i32 4, %272, !spirv.Decorations !837
  %278 = sext i32 %277 to i64
  %279 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 %278
  %280 = load i32, i32* %279, align 4, !noalias !841
  br label %268

281:                                              ; preds = %268
  br i1 %271, label %282, label %456

282:                                              ; preds = %281
  %283 = icmp eq i32 %266, 0
  br i1 %283, label %286, label %284

284:                                              ; preds = %282
  %285 = add i32 %266, -1
  store i32 %285, i32* %265, align 8, !noalias !838
  br label %455

286:                                              ; preds = %282
  store i32 3, i32* %265, align 8, !noalias !838
  %287 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1
  %288 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 1
  %289 = load i32, i32* %288, align 4, !noalias !838
  %290 = getelementptr inbounds [4 x i32], [4 x i32]* %287, i64 0, i64 0
  %291 = load i32, i32* %290, align 8, !noalias !838
  %292 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 2
  %293 = load i32, i32* %292, align 8, !noalias !838
  %294 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 3
  %295 = load i32, i32* %294, align 4, !noalias !838
  %296 = getelementptr inbounds %"class.oneapi::mkl::rng::device::philox4x32x10", %"class.oneapi::mkl::rng::device::philox4x32x10"* %253, i64 0, i32 0, i32 0, i32 0, i64 0
  %297 = load i32, i32* %296, align 8, !noalias !838
  %298 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  %299 = load i32, i32* %298, align 4, !noalias !838
  %300 = zext i32 %291 to i64
  %301 = mul nuw i64 %300, 3528531795, !spirv.Decorations !836
  %302 = lshr i64 %301, 32
  %303 = trunc i64 %302 to i32
  %304 = zext i32 %293 to i64
  %305 = mul nuw i64 %304, 3449720151, !spirv.Decorations !836
  %306 = lshr i64 %305, 32
  %307 = trunc i64 %306 to i32
  %308 = xor i32 %289, %307
  %309 = xor i32 %308, %297
  %310 = xor i32 %295, %303
  %311 = xor i32 %310, %299
  %312 = add i32 %297, -1640531527
  %313 = add i32 %299, -1150833019
  %314 = zext i32 %309 to i64
  %315 = mul nuw i64 %314, 3528531795, !spirv.Decorations !836
  %316 = lshr i64 %315, 32
  %317 = zext i32 %311 to i64
  %318 = mul nuw i64 %317, 3449720151, !spirv.Decorations !836
  %319 = lshr i64 %318, 32
  %320 = xor i64 %305, %319
  %321 = trunc i64 %320 to i32
  %322 = xor i32 %312, %321
  %323 = xor i64 %301, %316
  %324 = trunc i64 %323 to i32
  %325 = xor i32 %313, %324
  %326 = add i32 %297, 1013904242
  %327 = add i32 %299, 1993301258
  %328 = zext i32 %322 to i64
  %329 = mul nuw i64 %328, 3528531795, !spirv.Decorations !836
  %330 = lshr i64 %329, 32
  %331 = zext i32 %325 to i64
  %332 = mul nuw i64 %331, 3449720151, !spirv.Decorations !836
  %333 = lshr i64 %332, 32
  %334 = xor i64 %318, %333
  %335 = trunc i64 %334 to i32
  %336 = xor i32 %326, %335
  %337 = xor i64 %315, %330
  %338 = trunc i64 %337 to i32
  %339 = xor i32 %327, %338
  %340 = add i32 %297, -626627285
  %341 = add i32 %299, 842468239
  %342 = zext i32 %336 to i64
  %343 = mul nuw i64 %342, 3528531795, !spirv.Decorations !836
  %344 = lshr i64 %343, 32
  %345 = zext i32 %339 to i64
  %346 = mul nuw i64 %345, 3449720151, !spirv.Decorations !836
  %347 = lshr i64 %346, 32
  %348 = xor i64 %332, %347
  %349 = trunc i64 %348 to i32
  %350 = xor i32 %340, %349
  %351 = xor i64 %329, %344
  %352 = trunc i64 %351 to i32
  %353 = xor i32 %341, %352
  %354 = add i32 %297, 2027808484
  %355 = add i32 %299, -308364780
  %356 = zext i32 %350 to i64
  %357 = mul nuw i64 %356, 3528531795, !spirv.Decorations !836
  %358 = lshr i64 %357, 32
  %359 = zext i32 %353 to i64
  %360 = mul nuw i64 %359, 3449720151, !spirv.Decorations !836
  %361 = lshr i64 %360, 32
  %362 = xor i64 %346, %361
  %363 = trunc i64 %362 to i32
  %364 = xor i32 %354, %363
  %365 = xor i64 %343, %358
  %366 = trunc i64 %365 to i32
  %367 = xor i32 %355, %366
  %368 = add i32 %297, 387276957
  %369 = add i32 %299, -1459197799
  %370 = zext i32 %364 to i64
  %371 = mul nuw i64 %370, 3528531795, !spirv.Decorations !836
  %372 = lshr i64 %371, 32
  %373 = zext i32 %367 to i64
  %374 = mul nuw i64 %373, 3449720151, !spirv.Decorations !836
  %375 = lshr i64 %374, 32
  %376 = xor i64 %360, %375
  %377 = trunc i64 %376 to i32
  %378 = xor i32 %368, %377
  %379 = xor i64 %357, %372
  %380 = trunc i64 %379 to i32
  %381 = xor i32 %369, %380
  %382 = add i32 %297, -1253254570
  %383 = add i32 %299, 1684936478
  %384 = zext i32 %378 to i64
  %385 = mul nuw i64 %384, 3528531795, !spirv.Decorations !836
  %386 = lshr i64 %385, 32
  %387 = zext i32 %381 to i64
  %388 = mul nuw i64 %387, 3449720151, !spirv.Decorations !836
  %389 = lshr i64 %388, 32
  %390 = xor i64 %374, %389
  %391 = trunc i64 %390 to i32
  %392 = xor i32 %382, %391
  %393 = xor i64 %371, %386
  %394 = trunc i64 %393 to i32
  %395 = xor i32 %383, %394
  %396 = add i32 %297, 1401181199
  %397 = add i32 %299, 534103459
  %398 = zext i32 %392 to i64
  %399 = mul nuw i64 %398, 3528531795, !spirv.Decorations !836
  %400 = lshr i64 %399, 32
  %401 = zext i32 %395 to i64
  %402 = mul nuw i64 %401, 3449720151, !spirv.Decorations !836
  %403 = lshr i64 %402, 32
  %404 = xor i64 %388, %403
  %405 = trunc i64 %404 to i32
  %406 = xor i32 %396, %405
  %407 = xor i64 %385, %400
  %408 = trunc i64 %407 to i32
  %409 = xor i32 %397, %408
  %410 = add i32 %297, -239350328
  %411 = add i32 %299, -616729560
  %412 = zext i32 %406 to i64
  %413 = mul nuw i64 %412, 3528531795, !spirv.Decorations !836
  %414 = lshr i64 %413, 32
  %415 = zext i32 %409 to i64
  %416 = mul nuw i64 %415, 3449720151, !spirv.Decorations !836
  %417 = lshr i64 %416, 32
  %418 = xor i64 %402, %417
  %419 = trunc i64 %418 to i32
  %420 = xor i32 %410, %419
  %421 = xor i64 %399, %414
  %422 = trunc i64 %421 to i32
  %423 = xor i32 %411, %422
  %424 = add i32 %297, -1879881855
  %425 = add i32 %299, -1767562579
  %426 = zext i32 %420 to i64
  %427 = mul nuw i64 %426, 3528531795, !spirv.Decorations !836
  %428 = trunc i64 %427 to i32
  %429 = lshr i64 %427, 32
  %430 = zext i32 %423 to i64
  %431 = mul nuw i64 %430, 3449720151, !spirv.Decorations !836
  %432 = trunc i64 %431 to i32
  %433 = lshr i64 %431, 32
  %434 = xor i64 %416, %433
  %435 = trunc i64 %434 to i32
  %436 = xor i32 %424, %435
  %437 = xor i64 %413, %429
  %438 = trunc i64 %437 to i32
  %439 = xor i32 %425, %438
  %440 = getelementptr inbounds [4 x i32], [4 x i32]* %267, i64 0, i64 0
  store i32 %436, i32* %440, align 4, !noalias !838
  %441 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 1
  store i32 %432, i32* %441, align 8, !noalias !838
  %442 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 2
  store i32 %439, i32* %442, align 4, !noalias !838
  %443 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 3
  store i32 %428, i32* %443, align 8, !noalias !838
  %444 = add i32 %291, 1
  %445 = getelementptr inbounds [4 x i32], [4 x i32]* %287, i64 0, i64 0
  store i32 %444, i32* %445, align 8, !noalias !838
  %446 = icmp eq i32 %444, 0
  br i1 %446, label %447, label %455

447:                                              ; preds = %286
  %448 = add i32 %289, 1
  store i32 %448, i32* %288, align 4, !noalias !838
  %449 = icmp eq i32 %448, 0
  br i1 %449, label %450, label %455

450:                                              ; preds = %447
  %451 = add i32 %293, 1
  store i32 %451, i32* %292, align 8, !noalias !838
  %452 = icmp eq i32 %451, 0
  br i1 %452, label %453, label %455

453:                                              ; preds = %450
  %454 = add i32 %295, 1
  store i32 %454, i32* %294, align 4, !noalias !838
  br label %455

455:                                              ; preds = %453, %450, %447, %286, %284
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

456:                                              ; preds = %281
  %457 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1
  %458 = getelementptr inbounds [4 x i32], [4 x i32]* %457, i64 0, i64 0
  %459 = load i32, i32* %458, align 8, !noalias !841
  %460 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 1
  %461 = load i32, i32* %460, align 4, !noalias !841
  %462 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 2
  %463 = load i32, i32* %462, align 8, !noalias !841
  %464 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 1, i64 3
  %465 = load i32, i32* %464, align 4, !noalias !841
  %466 = getelementptr inbounds %"class.oneapi::mkl::rng::device::philox4x32x10", %"class.oneapi::mkl::rng::device::philox4x32x10"* %253, i64 0, i32 0, i32 0, i32 0, i64 0
  %467 = load i32, i32* %466, align 8, !noalias !841
  %468 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  %469 = load i32, i32* %468, align 4, !noalias !841
  %470 = zext i32 %459 to i64
  %471 = mul nuw i64 %470, 3528531795, !spirv.Decorations !836
  %472 = lshr i64 %471, 32
  %473 = trunc i64 %472 to i32
  %474 = zext i32 %463 to i64
  %475 = mul nuw i64 %474, 3449720151, !spirv.Decorations !836
  %476 = lshr i64 %475, 32
  %477 = trunc i64 %476 to i32
  %478 = xor i32 %461, %477
  %479 = xor i32 %478, %467
  %480 = xor i32 %465, %473
  %481 = xor i32 %480, %469
  %482 = add i32 %467, -1640531527
  %483 = add i32 %469, -1150833019
  %484 = zext i32 %479 to i64
  %485 = mul nuw i64 %484, 3528531795, !spirv.Decorations !836
  %486 = lshr i64 %485, 32
  %487 = zext i32 %481 to i64
  %488 = mul nuw i64 %487, 3449720151, !spirv.Decorations !836
  %489 = lshr i64 %488, 32
  %490 = xor i64 %475, %489
  %491 = trunc i64 %490 to i32
  %492 = xor i32 %482, %491
  %493 = xor i64 %471, %486
  %494 = trunc i64 %493 to i32
  %495 = xor i32 %483, %494
  %496 = add i32 %467, 1013904242
  %497 = add i32 %469, 1993301258
  %498 = zext i32 %492 to i64
  %499 = mul nuw i64 %498, 3528531795, !spirv.Decorations !836
  %500 = lshr i64 %499, 32
  %501 = zext i32 %495 to i64
  %502 = mul nuw i64 %501, 3449720151, !spirv.Decorations !836
  %503 = lshr i64 %502, 32
  %504 = xor i64 %488, %503
  %505 = trunc i64 %504 to i32
  %506 = xor i32 %496, %505
  %507 = xor i64 %485, %500
  %508 = trunc i64 %507 to i32
  %509 = xor i32 %497, %508
  %510 = add i32 %467, -626627285
  %511 = add i32 %469, 842468239
  %512 = zext i32 %506 to i64
  %513 = mul nuw i64 %512, 3528531795, !spirv.Decorations !836
  %514 = lshr i64 %513, 32
  %515 = zext i32 %509 to i64
  %516 = mul nuw i64 %515, 3449720151, !spirv.Decorations !836
  %517 = lshr i64 %516, 32
  %518 = xor i64 %502, %517
  %519 = trunc i64 %518 to i32
  %520 = xor i32 %510, %519
  %521 = xor i64 %499, %514
  %522 = trunc i64 %521 to i32
  %523 = xor i32 %511, %522
  %524 = add i32 %467, 2027808484
  %525 = add i32 %469, -308364780
  %526 = zext i32 %520 to i64
  %527 = mul nuw i64 %526, 3528531795, !spirv.Decorations !836
  %528 = lshr i64 %527, 32
  %529 = zext i32 %523 to i64
  %530 = mul nuw i64 %529, 3449720151, !spirv.Decorations !836
  %531 = lshr i64 %530, 32
  %532 = xor i64 %516, %531
  %533 = trunc i64 %532 to i32
  %534 = xor i32 %524, %533
  %535 = xor i64 %513, %528
  %536 = trunc i64 %535 to i32
  %537 = xor i32 %525, %536
  %538 = add i32 %467, 387276957
  %539 = add i32 %469, -1459197799
  %540 = zext i32 %534 to i64
  %541 = mul nuw i64 %540, 3528531795, !spirv.Decorations !836
  %542 = lshr i64 %541, 32
  %543 = zext i32 %537 to i64
  %544 = mul nuw i64 %543, 3449720151, !spirv.Decorations !836
  %545 = lshr i64 %544, 32
  %546 = xor i64 %530, %545
  %547 = trunc i64 %546 to i32
  %548 = xor i32 %538, %547
  %549 = xor i64 %527, %542
  %550 = trunc i64 %549 to i32
  %551 = xor i32 %539, %550
  %552 = add i32 %467, -1253254570
  %553 = add i32 %469, 1684936478
  %554 = zext i32 %548 to i64
  %555 = mul nuw i64 %554, 3528531795, !spirv.Decorations !836
  %556 = lshr i64 %555, 32
  %557 = zext i32 %551 to i64
  %558 = mul nuw i64 %557, 3449720151, !spirv.Decorations !836
  %559 = lshr i64 %558, 32
  %560 = xor i64 %544, %559
  %561 = trunc i64 %560 to i32
  %562 = xor i32 %552, %561
  %563 = xor i64 %541, %556
  %564 = trunc i64 %563 to i32
  %565 = xor i32 %553, %564
  %566 = add i32 %467, 1401181199
  %567 = add i32 %469, 534103459
  %568 = zext i32 %562 to i64
  %569 = mul nuw i64 %568, 3528531795, !spirv.Decorations !836
  %570 = lshr i64 %569, 32
  %571 = zext i32 %565 to i64
  %572 = mul nuw i64 %571, 3449720151, !spirv.Decorations !836
  %573 = lshr i64 %572, 32
  %574 = xor i64 %558, %573
  %575 = trunc i64 %574 to i32
  %576 = xor i32 %566, %575
  %577 = xor i64 %555, %570
  %578 = trunc i64 %577 to i32
  %579 = xor i32 %567, %578
  %580 = add i32 %469, -616729560
  %581 = zext i32 %576 to i64
  %582 = mul nuw i64 %581, 3528531795, !spirv.Decorations !836
  %583 = lshr i64 %582, 32
  %584 = mul i32 %579, -845247145
  %585 = xor i64 %569, %583
  %586 = trunc i64 %585 to i32
  %587 = xor i32 %580, %586
  %588 = add i32 %467, -1879881855
  %589 = zext i32 %587 to i64
  %590 = mul nuw i64 %589, 3449720151, !spirv.Decorations !836
  %591 = lshr i64 %590, 32
  %592 = trunc i64 %591 to i32
  %593 = xor i32 %584, %592
  %594 = xor i32 %593, %588
  %595 = icmp eq i32 %266, 0
  br i1 %595, label %598, label %596

596:                                              ; preds = %456
  %597 = add i32 %266, -1
  store i32 %597, i32* %265, align 8, !noalias !838
  br label %754

598:                                              ; preds = %456
  store i32 3, i32* %265, align 8, !noalias !838
  %599 = zext i32 %459 to i64
  %600 = mul nuw i64 %599, 3528531795, !spirv.Decorations !836
  %601 = lshr i64 %600, 32
  %602 = trunc i64 %601 to i32
  %603 = zext i32 %463 to i64
  %604 = mul nuw i64 %603, 3449720151, !spirv.Decorations !836
  %605 = lshr i64 %604, 32
  %606 = trunc i64 %605 to i32
  %607 = xor i32 %461, %606
  %608 = xor i32 %607, %467
  %609 = xor i32 %465, %602
  %610 = xor i32 %609, %469
  %611 = add i32 %467, -1640531527
  %612 = add i32 %469, -1150833019
  %613 = zext i32 %608 to i64
  %614 = mul nuw i64 %613, 3528531795, !spirv.Decorations !836
  %615 = lshr i64 %614, 32
  %616 = zext i32 %610 to i64
  %617 = mul nuw i64 %616, 3449720151, !spirv.Decorations !836
  %618 = lshr i64 %617, 32
  %619 = xor i64 %604, %618
  %620 = trunc i64 %619 to i32
  %621 = xor i32 %611, %620
  %622 = xor i64 %600, %615
  %623 = trunc i64 %622 to i32
  %624 = xor i32 %612, %623
  %625 = add i32 %467, 1013904242
  %626 = add i32 %469, 1993301258
  %627 = zext i32 %621 to i64
  %628 = mul nuw i64 %627, 3528531795, !spirv.Decorations !836
  %629 = lshr i64 %628, 32
  %630 = zext i32 %624 to i64
  %631 = mul nuw i64 %630, 3449720151, !spirv.Decorations !836
  %632 = lshr i64 %631, 32
  %633 = xor i64 %617, %632
  %634 = trunc i64 %633 to i32
  %635 = xor i32 %625, %634
  %636 = xor i64 %614, %629
  %637 = trunc i64 %636 to i32
  %638 = xor i32 %626, %637
  %639 = add i32 %467, -626627285
  %640 = add i32 %469, 842468239
  %641 = zext i32 %635 to i64
  %642 = mul nuw i64 %641, 3528531795, !spirv.Decorations !836
  %643 = lshr i64 %642, 32
  %644 = zext i32 %638 to i64
  %645 = mul nuw i64 %644, 3449720151, !spirv.Decorations !836
  %646 = lshr i64 %645, 32
  %647 = xor i64 %631, %646
  %648 = trunc i64 %647 to i32
  %649 = xor i32 %639, %648
  %650 = xor i64 %628, %643
  %651 = trunc i64 %650 to i32
  %652 = xor i32 %640, %651
  %653 = add i32 %467, 2027808484
  %654 = add i32 %469, -308364780
  %655 = zext i32 %649 to i64
  %656 = mul nuw i64 %655, 3528531795, !spirv.Decorations !836
  %657 = lshr i64 %656, 32
  %658 = zext i32 %652 to i64
  %659 = mul nuw i64 %658, 3449720151, !spirv.Decorations !836
  %660 = lshr i64 %659, 32
  %661 = xor i64 %645, %660
  %662 = trunc i64 %661 to i32
  %663 = xor i32 %653, %662
  %664 = xor i64 %642, %657
  %665 = trunc i64 %664 to i32
  %666 = xor i32 %654, %665
  %667 = add i32 %467, 387276957
  %668 = add i32 %469, -1459197799
  %669 = zext i32 %663 to i64
  %670 = mul nuw i64 %669, 3528531795, !spirv.Decorations !836
  %671 = lshr i64 %670, 32
  %672 = zext i32 %666 to i64
  %673 = mul nuw i64 %672, 3449720151, !spirv.Decorations !836
  %674 = lshr i64 %673, 32
  %675 = xor i64 %659, %674
  %676 = trunc i64 %675 to i32
  %677 = xor i32 %667, %676
  %678 = xor i64 %656, %671
  %679 = trunc i64 %678 to i32
  %680 = xor i32 %668, %679
  %681 = add i32 %467, -1253254570
  %682 = add i32 %469, 1684936478
  %683 = zext i32 %677 to i64
  %684 = mul nuw i64 %683, 3528531795, !spirv.Decorations !836
  %685 = lshr i64 %684, 32
  %686 = zext i32 %680 to i64
  %687 = mul nuw i64 %686, 3449720151, !spirv.Decorations !836
  %688 = lshr i64 %687, 32
  %689 = xor i64 %673, %688
  %690 = trunc i64 %689 to i32
  %691 = xor i32 %681, %690
  %692 = xor i64 %670, %685
  %693 = trunc i64 %692 to i32
  %694 = xor i32 %682, %693
  %695 = add i32 %467, 1401181199
  %696 = add i32 %469, 534103459
  %697 = zext i32 %691 to i64
  %698 = mul nuw i64 %697, 3528531795, !spirv.Decorations !836
  %699 = lshr i64 %698, 32
  %700 = zext i32 %694 to i64
  %701 = mul nuw i64 %700, 3449720151, !spirv.Decorations !836
  %702 = lshr i64 %701, 32
  %703 = xor i64 %687, %702
  %704 = trunc i64 %703 to i32
  %705 = xor i32 %695, %704
  %706 = xor i64 %684, %699
  %707 = trunc i64 %706 to i32
  %708 = xor i32 %696, %707
  %709 = add i32 %467, -239350328
  %710 = add i32 %469, -616729560
  %711 = zext i32 %705 to i64
  %712 = mul nuw i64 %711, 3528531795, !spirv.Decorations !836
  %713 = lshr i64 %712, 32
  %714 = zext i32 %708 to i64
  %715 = mul nuw i64 %714, 3449720151, !spirv.Decorations !836
  %716 = lshr i64 %715, 32
  %717 = xor i64 %701, %716
  %718 = trunc i64 %717 to i32
  %719 = xor i32 %709, %718
  %720 = xor i64 %698, %713
  %721 = trunc i64 %720 to i32
  %722 = xor i32 %710, %721
  %723 = add i32 %467, -1879881855
  %724 = add i32 %469, -1767562579
  %725 = zext i32 %719 to i64
  %726 = mul nuw i64 %725, 3528531795, !spirv.Decorations !836
  %727 = trunc i64 %726 to i32
  %728 = lshr i64 %726, 32
  %729 = zext i32 %722 to i64
  %730 = mul nuw i64 %729, 3449720151, !spirv.Decorations !836
  %731 = trunc i64 %730 to i32
  %732 = lshr i64 %730, 32
  %733 = xor i64 %715, %732
  %734 = trunc i64 %733 to i32
  %735 = xor i32 %723, %734
  %736 = xor i64 %712, %728
  %737 = trunc i64 %736 to i32
  %738 = xor i32 %724, %737
  %739 = getelementptr inbounds [4 x i32], [4 x i32]* %267, i64 0, i64 0
  store i32 %735, i32* %739, align 4, !noalias !838
  %740 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 1
  store i32 %731, i32* %740, align 8, !noalias !838
  %741 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 2
  store i32 %738, i32* %741, align 4, !noalias !838
  %742 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20, i64 0, i32 2, i32 0, i32 0, i32 3, i64 3
  store i32 %727, i32* %742, align 8, !noalias !838
  %743 = add i32 %459, 1
  %744 = getelementptr inbounds [4 x i32], [4 x i32]* %457, i64 0, i64 0
  store i32 %743, i32* %744, align 8, !noalias !838
  %745 = icmp eq i32 %743, 0
  br i1 %745, label %746, label %754

746:                                              ; preds = %598
  %747 = add i32 %461, 1
  store i32 %747, i32* %460, align 4, !noalias !838
  %748 = icmp eq i32 %747, 0
  br i1 %748, label %749, label %754

749:                                              ; preds = %746
  %750 = add i32 %463, 1
  store i32 %750, i32* %462, align 8, !noalias !838
  %751 = icmp eq i32 %750, 0
  br i1 %751, label %752, label %754

752:                                              ; preds = %749
  %753 = add i32 %465, 1
  store i32 %753, i32* %464, align 4, !noalias !838
  br label %754

754:                                              ; preds = %752, %749, %746, %598, %596
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit: ; preds = %455, %754
  %755 = phi i32 [ %269, %455 ], [ %594, %754 ]
  %756 = fadd reassoc nsz arcp contract float %264, %262, !spirv.Decorations !844
  %757 = fmul reassoc nsz arcp contract float %756, 5.000000e-01
  %758 = fsub reassoc nsz arcp contract float %264, %262, !spirv.Decorations !844
  %759 = fmul reassoc nsz arcp contract float %758, 0x3DF0000000000000
  %760 = sitofp i32 %755 to float
  %761 = fmul reassoc nsz arcp contract float %759, %760, !spirv.Decorations !844
  %762 = fadd reassoc nsz arcp contract float %761, %757, !spirv.Decorations !844
  %763 = load i32, i32* %254, align 8, !noalias !838
  %764 = icmp sgt i32 %763, -1
  br i1 %764, label %765, label %781

765:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %766 = load float, float* %255, align 4, !noalias !838
  %767 = fmul reassoc nsz arcp contract float %762, %766, !spirv.Decorations !844
  %768 = fcmp olt float %767, 0.000000e+00
  %769 = select i1 %768, float 0xBFDFFFFFE0000000, float 0x3FDFFFFFE0000000
  %770 = fadd float %769, %767
  %771 = call float @llvm.trunc.f32(float %770)
  %772 = fptosi float %771 to i32
  %773 = sitofp i32 %772 to float
  %774 = load float, float* %256, align 8, !noalias !838
  %775 = fmul reassoc nsz arcp contract float %774, %773, !spirv.Decorations !844
  %776 = fptosi float %775 to i32
  %777 = sitofp i32 %776 to float
  %778 = bitcast float %777 to i32
  %779 = lshr i32 %778, 16
  %780 = trunc i32 %779 to i16
  br label %782

781:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %bf_cvt = call i16 @llvm.genx.GenISA.ftobf.i16.f32(float %762, i32 0)
  br label %782

782:                                              ; preds = %781, %765
  %783 = phi i16 [ %780, %765 ], [ %bf_cvt, %781 ]
  %784 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 %258
  store i16 %783, i16 addrspace(1)* %784, align 2
  %785 = icmp sgt i32 %7, -1
  call void @llvm.assume(i1 %785)
  %786 = icmp sgt i32 %4, -1
  call void @llvm.assume(i1 %786)
  %.narrow = mul i32 %7, %4
  %787 = zext i32 %.narrow to i64
  %788 = add i64 %258, %787
  br label %257

789:                                              ; preds = %257
  %790 = bitcast %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 88, i8* nonnull %790)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i32 0
  %15 = extractelement <3 x i32> %numWorkGroups, i32 1
  %16 = extractelement <3 x i32> %numWorkGroups, i32 2
  %17 = extractelement <3 x i32> %localSize, i32 0
  %18 = extractelement <3 x i32> %localSize, i32 1
  %19 = extractelement <3 x i32> %localSize, i32 2
  %20 = extractelement <8 x i32> %r0, i32 0
  %21 = extractelement <8 x i32> %r0, i32 1
  %22 = extractelement <8 x i32> %r0, i32 2
  %23 = extractelement <8 x i32> %r0, i32 3
  %24 = extractelement <8 x i32> %r0, i32 4
  %25 = extractelement <8 x i32> %r0, i32 5
  %26 = extractelement <8 x i32> %r0, i32 6
  %27 = extractelement <8 x i32> %r0, i32 7
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [4 x float]], align 4, !spirv.Decorations !846
  %35 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %36 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %37 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %38 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %39 = icmp sgt i32 %21, -1
  call void @llvm.assume(i1 %39)
  %40 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %40)
  %41 = mul i32 %21, %17
  %42 = zext i16 %localIdX to i32
  %43 = add i32 %41, %42
  %44 = shl i32 %43, 2
  %45 = icmp sgt i32 %26, -1
  call void @llvm.assume(i1 %45)
  %46 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %46)
  %47 = mul i32 %26, %18
  %48 = zext i16 %localIdY to i32
  %49 = add i32 %47, %48
  %50 = shl i32 %49, 2
  %51 = zext i32 %27 to i64
  %52 = icmp sgt i32 %27, -1
  call void @llvm.assume(i1 %52)
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !837
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !837
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !844
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !837
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !837
  %62 = getelementptr inbounds float, float addrspace(4)* %35, i64 %61
  %63 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 1
  %64 = bitcast %structtype.0* %32 to [2 x i32]*
  %65 = getelementptr inbounds %structtype.0, %structtype.0* %32, i64 0, i32 0
  %66 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 1
  %67 = bitcast %structtype.0* %33 to [2 x i32]*
  %68 = getelementptr inbounds %structtype.0, %structtype.0* %33, i64 0, i32 0
  %69 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 1
  %70 = bitcast %structtype.0* %31 to [2 x i32]*
  %71 = getelementptr inbounds %structtype.0, %structtype.0* %31, i64 0, i32 0
  br label %72

72:                                               ; preds = %249, %13
  %73 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %54, %13 ], [ %191, %249 ]
  %74 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %56, %13 ], [ %193, %249 ]
  %75 = phi float addrspace(4)* [ %60, %13 ], [ %250, %249 ]
  %76 = phi float addrspace(4)* [ %62, %13 ], [ %254, %249 ]
  %77 = phi i32 [ %27, %13 ], [ %256, %249 ]
  %78 = icmp slt i32 %77, %8
  br i1 %78, label %79, label %257

79:                                               ; preds = %72
  %80 = bitcast [4 x [4 x float]]* %34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %80)
  br label %81

81:                                               ; preds = %89, %79
  %82 = phi i32 [ 0, %79 ], [ %90, %89 ]
  %83 = icmp ult i32 %82, 4
  br i1 %83, label %84, label %.preheader2

.preheader2:                                      ; preds = %81
  br label %95

84:                                               ; preds = %81
  %85 = zext i32 %82 to i64
  br label %86

86:                                               ; preds = %91, %84
  %87 = phi i32 [ %94, %91 ], [ 0, %84 ]
  %88 = icmp ult i32 %87, 4
  br i1 %88, label %91, label %89

89:                                               ; preds = %86
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !833
  br label %81, !llvm.loop !848

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !833
  br label %86, !llvm.loop !850

95:                                               ; preds = %105, %.preheader2
  %96 = phi i32 [ %106, %105 ], [ 0, %.preheader2 ]
  %97 = icmp slt i32 %96, %const_reg_dword2
  br i1 %97, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %95
  br label %180

.preheader:                                       ; preds = %95
  br label %98

98:                                               ; preds = %110, %.preheader
  %99 = phi i32 [ %111, %110 ], [ 0, %.preheader ]
  %100 = icmp ult i32 %99, 4
  br i1 %100, label %101, label %105

101:                                              ; preds = %98
  %102 = or i32 %50, %99
  %103 = icmp slt i32 %102, %const_reg_dword1
  %104 = zext i32 %99 to i64
  br label %107

105:                                              ; preds = %98
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !833
  br label %95

107:                                              ; preds = %178, %101
  %108 = phi i32 [ %179, %178 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !833
  br label %98, !llvm.loop !851

112:                                              ; preds = %107
  %113 = or i32 %44, %108
  %114 = icmp slt i32 %113, %const_reg_dword
  %115 = select i1 %114, i1 %103, i1 false
  br i1 %115, label %116, label %178

116:                                              ; preds = %112
  %117 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %117)
  %118 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %118)
  %119 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 0
  store i32 %113, i32* %119, align 4, !noalias !852
  store i32 %96, i32* %63, align 4, !noalias !852
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !852
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !852
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !833
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !837
  %136 = ashr i64 %131, 32
  %137 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %135, i32 0
  %138 = getelementptr i16, i16 addrspace(4)* %137, i64 %136
  %139 = addrspacecast i16 addrspace(4)* %138 to i16 addrspace(1)*
  %140 = load i16, i16 addrspace(1)* %139, align 2
  %141 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %141)
  %142 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %142)
  %143 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %96, i32* %143, align 4, !noalias !855
  store i32 %102, i32* %66, align 4, !noalias !855
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !855
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !855
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !833
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = ashr i64 %155, 32
  %158 = mul nsw i64 %157, %const_reg_qword5, !spirv.Decorations !837
  %159 = shl i64 %155, 32
  %160 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %158
  %161 = ashr exact i64 %159, 31
  %162 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %160 to i8 addrspace(4)*
  %163 = getelementptr i8, i8 addrspace(4)* %162, i64 %161
  %164 = bitcast i8 addrspace(4)* %163 to i16 addrspace(4)*
  %165 = addrspacecast i16 addrspace(4)* %164 to i16 addrspace(1)*
  %166 = load i16, i16 addrspace(1)* %165, align 2
  %167 = zext i16 %140 to i32
  %168 = shl nuw i32 %167, 16, !spirv.Decorations !836
  %169 = bitcast i32 %168 to float
  %170 = zext i16 %166 to i32
  %171 = shl nuw i32 %170, 16, !spirv.Decorations !836
  %172 = bitcast i32 %171 to float
  %173 = zext i32 %108 to i64
  %174 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %173, i64 %104
  %175 = fmul reassoc nsz arcp contract float %169, %172, !spirv.Decorations !844
  %176 = load float, float* %174, align 4
  %177 = fadd reassoc nsz arcp contract float %175, %176, !spirv.Decorations !844
  store float %177, float* %174, align 4
  br label %178

178:                                              ; preds = %153, %112
  %179 = add nuw nsw i32 %108, 1, !spirv.Decorations !833
  br label %107, !llvm.loop !858

180:                                              ; preds = %197, %.preheader1
  %181 = phi i32 [ %198, %197 ], [ 0, %.preheader1 ]
  %182 = icmp ult i32 %181, 4
  br i1 %182, label %183, label %187

183:                                              ; preds = %180
  %184 = or i32 %50, %181
  %185 = icmp slt i32 %184, %const_reg_dword1
  %186 = zext i32 %181 to i64
  br label %194

187:                                              ; preds = %180
  %188 = zext i32 %16 to i64
  %189 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %189)
  %190 = mul nsw i64 %188, %9, !spirv.Decorations !837
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %190
  %192 = mul nsw i64 %188, %10, !spirv.Decorations !837
  %193 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %192
  br i1 %57, label %244, label %249

194:                                              ; preds = %242, %183
  %195 = phi i32 [ %243, %242 ], [ 0, %183 ]
  %196 = icmp ult i32 %195, 4
  br i1 %196, label %199, label %197

197:                                              ; preds = %194
  %198 = add nuw nsw i32 %181, 1, !spirv.Decorations !833
  br label %180, !llvm.loop !859

199:                                              ; preds = %194
  %200 = or i32 %44, %195
  %201 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %201)
  %202 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %202)
  %203 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 0
  store i32 %200, i32* %203, align 4, !noalias !860
  store i32 %184, i32* %69, align 4, !noalias !860
  br label %204

204:                                              ; preds = %207, %199
  %205 = phi i32 [ 0, %199 ], [ %212, %207 ]
  %206 = icmp ult i32 %205, 2
  br i1 %206, label %207, label %213

207:                                              ; preds = %204
  %208 = zext i32 %205 to i64
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %208
  %210 = load i32, i32* %209, align 4, !noalias !860
  %211 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %208
  store i32 %210, i32* %211, align 4, !alias.scope !860
  %212 = add nuw nsw i32 %205, 1, !spirv.Decorations !833
  br label %204

213:                                              ; preds = %204
  %214 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %214)
  %215 = load i64, i64* %71, align 8
  %216 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %216)
  %217 = icmp slt i32 %200, %const_reg_dword
  %218 = select i1 %217, i1 %185, i1 false
  br i1 %218, label %219, label %242

219:                                              ; preds = %213
  %220 = zext i32 %195 to i64
  %221 = shl i64 %215, 32
  %222 = ashr exact i64 %221, 32
  %223 = ashr i64 %215, 32
  %224 = mul nsw i64 %222, %const_reg_qword9, !spirv.Decorations !837
  %225 = add nsw i64 %224, %223, !spirv.Decorations !837
  %226 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %220, i64 %186
  %227 = load float, float* %226, align 4
  %228 = fmul reassoc nsz arcp contract float %227, %1, !spirv.Decorations !844
  br i1 %57, label %229, label %239

229:                                              ; preds = %219
  %230 = mul nsw i64 %222, %const_reg_qword7, !spirv.Decorations !837
  %231 = getelementptr float, float addrspace(4)* %75, i64 %230
  %232 = getelementptr float, float addrspace(4)* %231, i64 %223
  %233 = addrspacecast float addrspace(4)* %232 to float addrspace(1)*
  %234 = load float, float addrspace(1)* %233, align 4
  %235 = fmul reassoc nsz arcp contract float %234, %4, !spirv.Decorations !844
  %236 = fadd reassoc nsz arcp contract float %228, %235, !spirv.Decorations !844
  %237 = getelementptr inbounds float, float addrspace(4)* %76, i64 %225
  %238 = addrspacecast float addrspace(4)* %237 to float addrspace(1)*
  store float %236, float addrspace(1)* %238, align 4
  br label %242

239:                                              ; preds = %219
  %240 = getelementptr inbounds float, float addrspace(4)* %76, i64 %225
  %241 = addrspacecast float addrspace(4)* %240 to float addrspace(1)*
  store float %228, float addrspace(1)* %241, align 4
  br label %242

242:                                              ; preds = %239, %229, %213
  %243 = add nuw nsw i32 %195, 1, !spirv.Decorations !833
  br label %194, !llvm.loop !863

244:                                              ; preds = %187
  %245 = zext i32 %16 to i64
  %246 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %246)
  %247 = mul nsw i64 %245, %11, !spirv.Decorations !837
  %248 = getelementptr inbounds float, float addrspace(4)* %75, i64 %247
  br label %249

249:                                              ; preds = %244, %187
  %250 = phi float addrspace(4)* [ %248, %244 ], [ %75, %187 ]
  %251 = zext i32 %16 to i64
  %252 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %252)
  %253 = mul nsw i64 %251, %12, !spirv.Decorations !837
  %254 = getelementptr inbounds float, float addrspace(4)* %76, i64 %253
  %255 = bitcast [4 x [4 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %255)
  %256 = add i32 %77, %16
  br label %72

257:                                              ; preds = %72
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i32 0
  %15 = extractelement <3 x i32> %numWorkGroups, i32 1
  %16 = extractelement <3 x i32> %numWorkGroups, i32 2
  %17 = extractelement <3 x i32> %localSize, i32 0
  %18 = extractelement <3 x i32> %localSize, i32 1
  %19 = extractelement <3 x i32> %localSize, i32 2
  %20 = extractelement <8 x i32> %r0, i32 0
  %21 = extractelement <8 x i32> %r0, i32 1
  %22 = extractelement <8 x i32> %r0, i32 2
  %23 = extractelement <8 x i32> %r0, i32 3
  %24 = extractelement <8 x i32> %r0, i32 4
  %25 = extractelement <8 x i32> %r0, i32 5
  %26 = extractelement <8 x i32> %r0, i32 6
  %27 = extractelement <8 x i32> %r0, i32 7
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !846
  %35 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %36 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %37 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %38 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %39 = icmp sgt i32 %21, -1
  call void @llvm.assume(i1 %39)
  %40 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %40)
  %41 = mul i32 %21, %17
  %42 = zext i16 %localIdX to i32
  %43 = add i32 %41, %42
  %44 = shl i32 %43, 2
  %45 = icmp sgt i32 %26, -1
  call void @llvm.assume(i1 %45)
  %46 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %46)
  %47 = mul i32 %26, %18
  %48 = zext i16 %localIdY to i32
  %49 = add i32 %47, %48
  %50 = shl i32 %49, 4
  %51 = zext i32 %27 to i64
  %52 = icmp sgt i32 %27, -1
  call void @llvm.assume(i1 %52)
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !837
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !837
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !844
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !837
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !837
  %62 = getelementptr inbounds float, float addrspace(4)* %35, i64 %61
  %63 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 1
  %64 = bitcast %structtype.0* %33 to [2 x i32]*
  %65 = getelementptr inbounds %structtype.0, %structtype.0* %33, i64 0, i32 0
  %66 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 1
  %67 = bitcast %structtype.0* %32 to [2 x i32]*
  %68 = getelementptr inbounds %structtype.0, %structtype.0* %32, i64 0, i32 0
  %69 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 1
  %70 = bitcast %structtype.0* %31 to [2 x i32]*
  %71 = getelementptr inbounds %structtype.0, %structtype.0* %31, i64 0, i32 0
  br label %72

72:                                               ; preds = %249, %13
  %73 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %54, %13 ], [ %191, %249 ]
  %74 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %56, %13 ], [ %193, %249 ]
  %75 = phi float addrspace(4)* [ %60, %13 ], [ %250, %249 ]
  %76 = phi float addrspace(4)* [ %62, %13 ], [ %254, %249 ]
  %77 = phi i32 [ %27, %13 ], [ %256, %249 ]
  %78 = icmp slt i32 %77, %8
  br i1 %78, label %79, label %257

79:                                               ; preds = %72
  %80 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* nonnull %80)
  br label %81

81:                                               ; preds = %89, %79
  %82 = phi i32 [ 0, %79 ], [ %90, %89 ]
  %83 = icmp ult i32 %82, 16
  br i1 %83, label %84, label %.preheader2

.preheader2:                                      ; preds = %81
  br label %95

84:                                               ; preds = %81
  %85 = zext i32 %82 to i64
  br label %86

86:                                               ; preds = %91, %84
  %87 = phi i32 [ %94, %91 ], [ 0, %84 ]
  %88 = icmp ult i32 %87, 4
  br i1 %88, label %91, label %89

89:                                               ; preds = %86
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !833
  br label %81, !llvm.loop !864

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !833
  br label %86, !llvm.loop !865

95:                                               ; preds = %105, %.preheader2
  %96 = phi i32 [ %106, %105 ], [ 0, %.preheader2 ]
  %97 = icmp slt i32 %96, %const_reg_dword2
  br i1 %97, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %95
  br label %180

.preheader:                                       ; preds = %95
  br label %98

98:                                               ; preds = %110, %.preheader
  %99 = phi i32 [ %111, %110 ], [ 0, %.preheader ]
  %100 = icmp ult i32 %99, 16
  br i1 %100, label %101, label %105

101:                                              ; preds = %98
  %102 = or i32 %50, %99
  %103 = icmp slt i32 %102, %const_reg_dword1
  %104 = zext i32 %99 to i64
  br label %107

105:                                              ; preds = %98
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !833
  br label %95

107:                                              ; preds = %178, %101
  %108 = phi i32 [ %179, %178 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !833
  br label %98, !llvm.loop !866

112:                                              ; preds = %107
  %113 = or i32 %44, %108
  %114 = icmp slt i32 %113, %const_reg_dword
  %115 = select i1 %114, i1 %103, i1 false
  br i1 %115, label %116, label %178

116:                                              ; preds = %112
  %117 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %117)
  %118 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %118)
  %119 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 0
  store i32 %113, i32* %119, align 4, !noalias !867
  store i32 %96, i32* %63, align 4, !noalias !867
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !867
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !867
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !833
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !837
  %136 = ashr i64 %131, 32
  %137 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %135, i32 0
  %138 = getelementptr i16, i16 addrspace(4)* %137, i64 %136
  %139 = addrspacecast i16 addrspace(4)* %138 to i16 addrspace(1)*
  %140 = load i16, i16 addrspace(1)* %139, align 2
  %141 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %141)
  %142 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %142)
  %143 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 0
  store i32 %96, i32* %143, align 4, !noalias !870
  store i32 %102, i32* %66, align 4, !noalias !870
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !870
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !870
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !833
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = ashr i64 %155, 32
  %158 = mul nsw i64 %157, %const_reg_qword5, !spirv.Decorations !837
  %159 = shl i64 %155, 32
  %160 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %158
  %161 = ashr exact i64 %159, 31
  %162 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %160 to i8 addrspace(4)*
  %163 = getelementptr i8, i8 addrspace(4)* %162, i64 %161
  %164 = bitcast i8 addrspace(4)* %163 to i16 addrspace(4)*
  %165 = addrspacecast i16 addrspace(4)* %164 to i16 addrspace(1)*
  %166 = load i16, i16 addrspace(1)* %165, align 2
  %167 = zext i16 %140 to i32
  %168 = shl nuw i32 %167, 16, !spirv.Decorations !836
  %169 = bitcast i32 %168 to float
  %170 = zext i16 %166 to i32
  %171 = shl nuw i32 %170, 16, !spirv.Decorations !836
  %172 = bitcast i32 %171 to float
  %173 = zext i32 %108 to i64
  %174 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %173, i64 %104
  %175 = fmul reassoc nsz arcp contract float %169, %172, !spirv.Decorations !844
  %176 = load float, float* %174, align 4
  %177 = fadd reassoc nsz arcp contract float %175, %176, !spirv.Decorations !844
  store float %177, float* %174, align 4
  br label %178

178:                                              ; preds = %153, %112
  %179 = add nuw nsw i32 %108, 1, !spirv.Decorations !833
  br label %107, !llvm.loop !873

180:                                              ; preds = %197, %.preheader1
  %181 = phi i32 [ %198, %197 ], [ 0, %.preheader1 ]
  %182 = icmp ult i32 %181, 16
  br i1 %182, label %183, label %187

183:                                              ; preds = %180
  %184 = or i32 %50, %181
  %185 = icmp slt i32 %184, %const_reg_dword1
  %186 = zext i32 %181 to i64
  br label %194

187:                                              ; preds = %180
  %188 = zext i32 %16 to i64
  %189 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %189)
  %190 = mul nsw i64 %188, %9, !spirv.Decorations !837
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %190
  %192 = mul nsw i64 %188, %10, !spirv.Decorations !837
  %193 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %192
  br i1 %57, label %244, label %249

194:                                              ; preds = %242, %183
  %195 = phi i32 [ %243, %242 ], [ 0, %183 ]
  %196 = icmp ult i32 %195, 4
  br i1 %196, label %199, label %197

197:                                              ; preds = %194
  %198 = add nuw nsw i32 %181, 1, !spirv.Decorations !833
  br label %180, !llvm.loop !874

199:                                              ; preds = %194
  %200 = or i32 %44, %195
  %201 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %201)
  %202 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %202)
  %203 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %200, i32* %203, align 4, !noalias !875
  store i32 %184, i32* %69, align 4, !noalias !875
  br label %204

204:                                              ; preds = %207, %199
  %205 = phi i32 [ 0, %199 ], [ %212, %207 ]
  %206 = icmp ult i32 %205, 2
  br i1 %206, label %207, label %213

207:                                              ; preds = %204
  %208 = zext i32 %205 to i64
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %208
  %210 = load i32, i32* %209, align 4, !noalias !875
  %211 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %208
  store i32 %210, i32* %211, align 4, !alias.scope !875
  %212 = add nuw nsw i32 %205, 1, !spirv.Decorations !833
  br label %204

213:                                              ; preds = %204
  %214 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %214)
  %215 = load i64, i64* %71, align 8
  %216 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %216)
  %217 = icmp slt i32 %200, %const_reg_dword
  %218 = select i1 %217, i1 %185, i1 false
  br i1 %218, label %219, label %242

219:                                              ; preds = %213
  %220 = zext i32 %195 to i64
  %221 = shl i64 %215, 32
  %222 = ashr exact i64 %221, 32
  %223 = ashr i64 %215, 32
  %224 = mul nsw i64 %222, %const_reg_qword9, !spirv.Decorations !837
  %225 = add nsw i64 %224, %223, !spirv.Decorations !837
  %226 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %220, i64 %186
  %227 = load float, float* %226, align 4
  %228 = fmul reassoc nsz arcp contract float %227, %1, !spirv.Decorations !844
  br i1 %57, label %229, label %239

229:                                              ; preds = %219
  %230 = mul nsw i64 %222, %const_reg_qword7, !spirv.Decorations !837
  %231 = getelementptr float, float addrspace(4)* %75, i64 %230
  %232 = getelementptr float, float addrspace(4)* %231, i64 %223
  %233 = addrspacecast float addrspace(4)* %232 to float addrspace(1)*
  %234 = load float, float addrspace(1)* %233, align 4
  %235 = fmul reassoc nsz arcp contract float %234, %4, !spirv.Decorations !844
  %236 = fadd reassoc nsz arcp contract float %228, %235, !spirv.Decorations !844
  %237 = getelementptr inbounds float, float addrspace(4)* %76, i64 %225
  %238 = addrspacecast float addrspace(4)* %237 to float addrspace(1)*
  store float %236, float addrspace(1)* %238, align 4
  br label %242

239:                                              ; preds = %219
  %240 = getelementptr inbounds float, float addrspace(4)* %76, i64 %225
  %241 = addrspacecast float addrspace(4)* %240 to float addrspace(1)*
  store float %228, float addrspace(1)* %241, align 4
  br label %242

242:                                              ; preds = %239, %229, %213
  %243 = add nuw nsw i32 %195, 1, !spirv.Decorations !833
  br label %194, !llvm.loop !878

244:                                              ; preds = %187
  %245 = zext i32 %16 to i64
  %246 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %246)
  %247 = mul nsw i64 %245, %11, !spirv.Decorations !837
  %248 = getelementptr inbounds float, float addrspace(4)* %75, i64 %247
  br label %249

249:                                              ; preds = %244, %187
  %250 = phi float addrspace(4)* [ %248, %244 ], [ %75, %187 ]
  %251 = zext i32 %16 to i64
  %252 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %252)
  %253 = mul nsw i64 %251, %12, !spirv.Decorations !837
  %254 = getelementptr inbounds float, float addrspace(4)* %76, i64 %253
  %255 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %255)
  %256 = add i32 %77, %16
  br label %72

257:                                              ; preds = %72
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i32 0
  %15 = extractelement <3 x i32> %numWorkGroups, i32 1
  %16 = extractelement <3 x i32> %numWorkGroups, i32 2
  %17 = extractelement <3 x i32> %localSize, i32 0
  %18 = extractelement <3 x i32> %localSize, i32 1
  %19 = extractelement <3 x i32> %localSize, i32 2
  %20 = extractelement <8 x i32> %r0, i32 0
  %21 = extractelement <8 x i32> %r0, i32 1
  %22 = extractelement <8 x i32> %r0, i32 2
  %23 = extractelement <8 x i32> %r0, i32 3
  %24 = extractelement <8 x i32> %r0, i32 4
  %25 = extractelement <8 x i32> %r0, i32 5
  %26 = extractelement <8 x i32> %r0, i32 6
  %27 = extractelement <8 x i32> %r0, i32 7
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [4 x float]], align 4, !spirv.Decorations !846
  %35 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %36 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %37 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %38 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %39 = icmp sgt i32 %21, -1
  call void @llvm.assume(i1 %39)
  %40 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %40)
  %41 = mul i32 %21, %17
  %42 = zext i16 %localIdX to i32
  %43 = add i32 %41, %42
  %44 = shl i32 %43, 2
  %45 = icmp sgt i32 %26, -1
  call void @llvm.assume(i1 %45)
  %46 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %46)
  %47 = mul i32 %26, %18
  %48 = zext i16 %localIdY to i32
  %49 = add i32 %47, %48
  %50 = shl i32 %49, 2
  %51 = zext i32 %27 to i64
  %52 = icmp sgt i32 %27, -1
  call void @llvm.assume(i1 %52)
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !837
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !837
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !844
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !837
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !837
  %62 = getelementptr inbounds float, float addrspace(4)* %35, i64 %61
  %63 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 1
  %64 = bitcast %structtype.0* %33 to [2 x i32]*
  %65 = getelementptr inbounds %structtype.0, %structtype.0* %33, i64 0, i32 0
  %66 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 1
  %67 = bitcast %structtype.0* %32 to [2 x i32]*
  %68 = getelementptr inbounds %structtype.0, %structtype.0* %32, i64 0, i32 0
  %69 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 1
  %70 = bitcast %structtype.0* %31 to [2 x i32]*
  %71 = getelementptr inbounds %structtype.0, %structtype.0* %31, i64 0, i32 0
  br label %72

72:                                               ; preds = %247, %13
  %73 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %54, %13 ], [ %189, %247 ]
  %74 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %56, %13 ], [ %191, %247 ]
  %75 = phi float addrspace(4)* [ %60, %13 ], [ %248, %247 ]
  %76 = phi float addrspace(4)* [ %62, %13 ], [ %252, %247 ]
  %77 = phi i32 [ %27, %13 ], [ %254, %247 ]
  %78 = icmp slt i32 %77, %8
  br i1 %78, label %79, label %255

79:                                               ; preds = %72
  %80 = bitcast [4 x [4 x float]]* %34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %80)
  br label %81

81:                                               ; preds = %89, %79
  %82 = phi i32 [ 0, %79 ], [ %90, %89 ]
  %83 = icmp ult i32 %82, 4
  br i1 %83, label %84, label %.preheader2

.preheader2:                                      ; preds = %81
  br label %95

84:                                               ; preds = %81
  %85 = zext i32 %82 to i64
  br label %86

86:                                               ; preds = %91, %84
  %87 = phi i32 [ %94, %91 ], [ 0, %84 ]
  %88 = icmp ult i32 %87, 4
  br i1 %88, label %91, label %89

89:                                               ; preds = %86
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !833
  br label %81, !llvm.loop !879

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !833
  br label %86, !llvm.loop !880

95:                                               ; preds = %105, %.preheader2
  %96 = phi i32 [ %106, %105 ], [ 0, %.preheader2 ]
  %97 = icmp slt i32 %96, %const_reg_dword2
  br i1 %97, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %95
  br label %178

.preheader:                                       ; preds = %95
  br label %98

98:                                               ; preds = %110, %.preheader
  %99 = phi i32 [ %111, %110 ], [ 0, %.preheader ]
  %100 = icmp ult i32 %99, 4
  br i1 %100, label %101, label %105

101:                                              ; preds = %98
  %102 = or i32 %50, %99
  %103 = icmp slt i32 %102, %const_reg_dword1
  %104 = zext i32 %99 to i64
  br label %107

105:                                              ; preds = %98
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !833
  br label %95

107:                                              ; preds = %176, %101
  %108 = phi i32 [ %177, %176 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !833
  br label %98, !llvm.loop !881

112:                                              ; preds = %107
  %113 = or i32 %44, %108
  %114 = icmp slt i32 %113, %const_reg_dword
  %115 = select i1 %114, i1 %103, i1 false
  br i1 %115, label %116, label %176

116:                                              ; preds = %112
  %117 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %117)
  %118 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %118)
  %119 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 0
  store i32 %113, i32* %119, align 4, !noalias !882
  store i32 %96, i32* %63, align 4, !noalias !882
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !882
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !882
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !833
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !837
  %136 = ashr i64 %131, 32
  %137 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %135, i32 0
  %138 = getelementptr i16, i16 addrspace(4)* %137, i64 %136
  %139 = addrspacecast i16 addrspace(4)* %138 to i16 addrspace(1)*
  %140 = load i16, i16 addrspace(1)* %139, align 2
  %141 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %141)
  %142 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %142)
  %143 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 0
  store i32 %96, i32* %143, align 4, !noalias !885
  store i32 %102, i32* %66, align 4, !noalias !885
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !885
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !885
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !833
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = shl i64 %155, 32
  %158 = ashr exact i64 %157, 32
  %159 = mul nsw i64 %158, %const_reg_qword5, !spirv.Decorations !837
  %160 = ashr i64 %155, 32
  %161 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %159, i32 0
  %162 = getelementptr i16, i16 addrspace(4)* %161, i64 %160
  %163 = addrspacecast i16 addrspace(4)* %162 to i16 addrspace(1)*
  %164 = load i16, i16 addrspace(1)* %163, align 2
  %165 = zext i16 %140 to i32
  %166 = shl nuw i32 %165, 16, !spirv.Decorations !836
  %167 = bitcast i32 %166 to float
  %168 = zext i16 %164 to i32
  %169 = shl nuw i32 %168, 16, !spirv.Decorations !836
  %170 = bitcast i32 %169 to float
  %171 = zext i32 %108 to i64
  %172 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %171, i64 %104
  %173 = fmul reassoc nsz arcp contract float %167, %170, !spirv.Decorations !844
  %174 = load float, float* %172, align 4
  %175 = fadd reassoc nsz arcp contract float %173, %174, !spirv.Decorations !844
  store float %175, float* %172, align 4
  br label %176

176:                                              ; preds = %153, %112
  %177 = add nuw nsw i32 %108, 1, !spirv.Decorations !833
  br label %107, !llvm.loop !888

178:                                              ; preds = %195, %.preheader1
  %179 = phi i32 [ %196, %195 ], [ 0, %.preheader1 ]
  %180 = icmp ult i32 %179, 4
  br i1 %180, label %181, label %185

181:                                              ; preds = %178
  %182 = or i32 %50, %179
  %183 = icmp slt i32 %182, %const_reg_dword1
  %184 = zext i32 %179 to i64
  br label %192

185:                                              ; preds = %178
  %186 = zext i32 %16 to i64
  %187 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %187)
  %188 = mul nsw i64 %186, %9, !spirv.Decorations !837
  %189 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %188
  %190 = mul nsw i64 %186, %10, !spirv.Decorations !837
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %190
  br i1 %57, label %242, label %247

192:                                              ; preds = %240, %181
  %193 = phi i32 [ %241, %240 ], [ 0, %181 ]
  %194 = icmp ult i32 %193, 4
  br i1 %194, label %197, label %195

195:                                              ; preds = %192
  %196 = add nuw nsw i32 %179, 1, !spirv.Decorations !833
  br label %178, !llvm.loop !889

197:                                              ; preds = %192
  %198 = or i32 %44, %193
  %199 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %199)
  %200 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %200)
  %201 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %198, i32* %201, align 4, !noalias !890
  store i32 %182, i32* %69, align 4, !noalias !890
  br label %202

202:                                              ; preds = %205, %197
  %203 = phi i32 [ 0, %197 ], [ %210, %205 ]
  %204 = icmp ult i32 %203, 2
  br i1 %204, label %205, label %211

205:                                              ; preds = %202
  %206 = zext i32 %203 to i64
  %207 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %206
  %208 = load i32, i32* %207, align 4, !noalias !890
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %206
  store i32 %208, i32* %209, align 4, !alias.scope !890
  %210 = add nuw nsw i32 %203, 1, !spirv.Decorations !833
  br label %202

211:                                              ; preds = %202
  %212 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %212)
  %213 = load i64, i64* %71, align 8
  %214 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %214)
  %215 = icmp slt i32 %198, %const_reg_dword
  %216 = select i1 %215, i1 %183, i1 false
  br i1 %216, label %217, label %240

217:                                              ; preds = %211
  %218 = zext i32 %193 to i64
  %219 = shl i64 %213, 32
  %220 = ashr exact i64 %219, 32
  %221 = ashr i64 %213, 32
  %222 = mul nsw i64 %220, %const_reg_qword9, !spirv.Decorations !837
  %223 = add nsw i64 %222, %221, !spirv.Decorations !837
  %224 = getelementptr inbounds [4 x [4 x float]], [4 x [4 x float]]* %34, i64 0, i64 %218, i64 %184
  %225 = load float, float* %224, align 4
  %226 = fmul reassoc nsz arcp contract float %225, %1, !spirv.Decorations !844
  br i1 %57, label %227, label %237

227:                                              ; preds = %217
  %228 = mul nsw i64 %220, %const_reg_qword7, !spirv.Decorations !837
  %229 = getelementptr float, float addrspace(4)* %75, i64 %228
  %230 = getelementptr float, float addrspace(4)* %229, i64 %221
  %231 = addrspacecast float addrspace(4)* %230 to float addrspace(1)*
  %232 = load float, float addrspace(1)* %231, align 4
  %233 = fmul reassoc nsz arcp contract float %232, %4, !spirv.Decorations !844
  %234 = fadd reassoc nsz arcp contract float %226, %233, !spirv.Decorations !844
  %235 = getelementptr inbounds float, float addrspace(4)* %76, i64 %223
  %236 = addrspacecast float addrspace(4)* %235 to float addrspace(1)*
  store float %234, float addrspace(1)* %236, align 4
  br label %240

237:                                              ; preds = %217
  %238 = getelementptr inbounds float, float addrspace(4)* %76, i64 %223
  %239 = addrspacecast float addrspace(4)* %238 to float addrspace(1)*
  store float %226, float addrspace(1)* %239, align 4
  br label %240

240:                                              ; preds = %237, %227, %211
  %241 = add nuw nsw i32 %193, 1, !spirv.Decorations !833
  br label %192, !llvm.loop !893

242:                                              ; preds = %185
  %243 = zext i32 %16 to i64
  %244 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %244)
  %245 = mul nsw i64 %243, %11, !spirv.Decorations !837
  %246 = getelementptr inbounds float, float addrspace(4)* %75, i64 %245
  br label %247

247:                                              ; preds = %242, %185
  %248 = phi float addrspace(4)* [ %246, %242 ], [ %75, %185 ]
  %249 = zext i32 %16 to i64
  %250 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %250)
  %251 = mul nsw i64 %249, %12, !spirv.Decorations !837
  %252 = getelementptr inbounds float, float addrspace(4)* %76, i64 %251
  %253 = bitcast [4 x [4 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %253)
  %254 = add i32 %77, %16
  br label %72

255:                                              ; preds = %72
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i32 0
  %15 = extractelement <3 x i32> %numWorkGroups, i32 1
  %16 = extractelement <3 x i32> %numWorkGroups, i32 2
  %17 = extractelement <3 x i32> %localSize, i32 0
  %18 = extractelement <3 x i32> %localSize, i32 1
  %19 = extractelement <3 x i32> %localSize, i32 2
  %20 = extractelement <8 x i32> %r0, i32 0
  %21 = extractelement <8 x i32> %r0, i32 1
  %22 = extractelement <8 x i32> %r0, i32 2
  %23 = extractelement <8 x i32> %r0, i32 3
  %24 = extractelement <8 x i32> %r0, i32 4
  %25 = extractelement <8 x i32> %r0, i32 5
  %26 = extractelement <8 x i32> %r0, i32 6
  %27 = extractelement <8 x i32> %r0, i32 7
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !846
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !846
  %35 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %36 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %37 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %38 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %39 = icmp sgt i32 %21, -1
  call void @llvm.assume(i1 %39)
  %40 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %40)
  %41 = mul i32 %21, %17
  %42 = zext i16 %localIdX to i32
  %43 = add i32 %41, %42
  %44 = shl i32 %43, 2
  %45 = icmp sgt i32 %26, -1
  call void @llvm.assume(i1 %45)
  %46 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %46)
  %47 = mul i32 %26, %18
  %48 = zext i16 %localIdY to i32
  %49 = add i32 %47, %48
  %50 = shl i32 %49, 4
  %51 = zext i32 %27 to i64
  %52 = icmp sgt i32 %27, -1
  call void @llvm.assume(i1 %52)
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !837
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !837
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !844
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !837
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !837
  %62 = getelementptr inbounds float, float addrspace(4)* %35, i64 %61
  %63 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 1
  %64 = bitcast %structtype.0* %33 to [2 x i32]*
  %65 = getelementptr inbounds %structtype.0, %structtype.0* %33, i64 0, i32 0
  %66 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 1
  %67 = bitcast %structtype.0* %32 to [2 x i32]*
  %68 = getelementptr inbounds %structtype.0, %structtype.0* %32, i64 0, i32 0
  %69 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 1
  %70 = bitcast %structtype.0* %31 to [2 x i32]*
  %71 = getelementptr inbounds %structtype.0, %structtype.0* %31, i64 0, i32 0
  br label %72

72:                                               ; preds = %247, %13
  %73 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %54, %13 ], [ %189, %247 ]
  %74 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %56, %13 ], [ %191, %247 ]
  %75 = phi float addrspace(4)* [ %60, %13 ], [ %248, %247 ]
  %76 = phi float addrspace(4)* [ %62, %13 ], [ %252, %247 ]
  %77 = phi i32 [ %27, %13 ], [ %254, %247 ]
  %78 = icmp slt i32 %77, %8
  br i1 %78, label %79, label %255

79:                                               ; preds = %72
  %80 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* nonnull %80)
  br label %81

81:                                               ; preds = %89, %79
  %82 = phi i32 [ 0, %79 ], [ %90, %89 ]
  %83 = icmp ult i32 %82, 16
  br i1 %83, label %84, label %.preheader2

.preheader2:                                      ; preds = %81
  br label %95

84:                                               ; preds = %81
  %85 = zext i32 %82 to i64
  br label %86

86:                                               ; preds = %91, %84
  %87 = phi i32 [ %94, %91 ], [ 0, %84 ]
  %88 = icmp ult i32 %87, 4
  br i1 %88, label %91, label %89

89:                                               ; preds = %86
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !833
  br label %81, !llvm.loop !894

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !833
  br label %86, !llvm.loop !895

95:                                               ; preds = %105, %.preheader2
  %96 = phi i32 [ %106, %105 ], [ 0, %.preheader2 ]
  %97 = icmp slt i32 %96, %const_reg_dword2
  br i1 %97, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %95
  br label %178

.preheader:                                       ; preds = %95
  br label %98

98:                                               ; preds = %110, %.preheader
  %99 = phi i32 [ %111, %110 ], [ 0, %.preheader ]
  %100 = icmp ult i32 %99, 16
  br i1 %100, label %101, label %105

101:                                              ; preds = %98
  %102 = or i32 %50, %99
  %103 = icmp slt i32 %102, %const_reg_dword1
  %104 = zext i32 %99 to i64
  br label %107

105:                                              ; preds = %98
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !833
  br label %95

107:                                              ; preds = %176, %101
  %108 = phi i32 [ %177, %176 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !833
  br label %98, !llvm.loop !896

112:                                              ; preds = %107
  %113 = or i32 %44, %108
  %114 = icmp slt i32 %113, %const_reg_dword
  %115 = select i1 %114, i1 %103, i1 false
  br i1 %115, label %116, label %176

116:                                              ; preds = %112
  %117 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %117)
  %118 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %118)
  %119 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 0
  store i32 %113, i32* %119, align 4, !noalias !897
  store i32 %96, i32* %63, align 4, !noalias !897
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !897
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !897
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !833
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !837
  %136 = ashr i64 %131, 32
  %137 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %135, i32 0
  %138 = getelementptr i16, i16 addrspace(4)* %137, i64 %136
  %139 = addrspacecast i16 addrspace(4)* %138 to i16 addrspace(1)*
  %140 = load i16, i16 addrspace(1)* %139, align 2
  %141 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %141)
  %142 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %142)
  %143 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 0
  store i32 %96, i32* %143, align 4, !noalias !900
  store i32 %102, i32* %66, align 4, !noalias !900
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !900
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !900
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !833
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = shl i64 %155, 32
  %158 = ashr exact i64 %157, 32
  %159 = mul nsw i64 %158, %const_reg_qword5, !spirv.Decorations !837
  %160 = ashr i64 %155, 32
  %161 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %159, i32 0
  %162 = getelementptr i16, i16 addrspace(4)* %161, i64 %160
  %163 = addrspacecast i16 addrspace(4)* %162 to i16 addrspace(1)*
  %164 = load i16, i16 addrspace(1)* %163, align 2
  %165 = zext i16 %140 to i32
  %166 = shl nuw i32 %165, 16, !spirv.Decorations !836
  %167 = bitcast i32 %166 to float
  %168 = zext i16 %164 to i32
  %169 = shl nuw i32 %168, 16, !spirv.Decorations !836
  %170 = bitcast i32 %169 to float
  %171 = zext i32 %108 to i64
  %172 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %171, i64 %104
  %173 = fmul reassoc nsz arcp contract float %167, %170, !spirv.Decorations !844
  %174 = load float, float* %172, align 4
  %175 = fadd reassoc nsz arcp contract float %173, %174, !spirv.Decorations !844
  store float %175, float* %172, align 4
  br label %176

176:                                              ; preds = %153, %112
  %177 = add nuw nsw i32 %108, 1, !spirv.Decorations !833
  br label %107, !llvm.loop !903

178:                                              ; preds = %195, %.preheader1
  %179 = phi i32 [ %196, %195 ], [ 0, %.preheader1 ]
  %180 = icmp ult i32 %179, 16
  br i1 %180, label %181, label %185

181:                                              ; preds = %178
  %182 = or i32 %50, %179
  %183 = icmp slt i32 %182, %const_reg_dword1
  %184 = zext i32 %179 to i64
  br label %192

185:                                              ; preds = %178
  %186 = zext i32 %16 to i64
  %187 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %187)
  %188 = mul nsw i64 %186, %9, !spirv.Decorations !837
  %189 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %188
  %190 = mul nsw i64 %186, %10, !spirv.Decorations !837
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %190
  br i1 %57, label %242, label %247

192:                                              ; preds = %240, %181
  %193 = phi i32 [ %241, %240 ], [ 0, %181 ]
  %194 = icmp ult i32 %193, 4
  br i1 %194, label %197, label %195

195:                                              ; preds = %192
  %196 = add nuw nsw i32 %179, 1, !spirv.Decorations !833
  br label %178, !llvm.loop !904

197:                                              ; preds = %192
  %198 = or i32 %44, %193
  %199 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %199)
  %200 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %200)
  %201 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %198, i32* %201, align 4, !noalias !905
  store i32 %182, i32* %69, align 4, !noalias !905
  br label %202

202:                                              ; preds = %205, %197
  %203 = phi i32 [ 0, %197 ], [ %210, %205 ]
  %204 = icmp ult i32 %203, 2
  br i1 %204, label %205, label %211

205:                                              ; preds = %202
  %206 = zext i32 %203 to i64
  %207 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %206
  %208 = load i32, i32* %207, align 4, !noalias !905
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %206
  store i32 %208, i32* %209, align 4, !alias.scope !905
  %210 = add nuw nsw i32 %203, 1, !spirv.Decorations !833
  br label %202

211:                                              ; preds = %202
  %212 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %212)
  %213 = load i64, i64* %71, align 8
  %214 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %214)
  %215 = icmp slt i32 %198, %const_reg_dword
  %216 = select i1 %215, i1 %183, i1 false
  br i1 %216, label %217, label %240

217:                                              ; preds = %211
  %218 = zext i32 %193 to i64
  %219 = shl i64 %213, 32
  %220 = ashr exact i64 %219, 32
  %221 = ashr i64 %213, 32
  %222 = mul nsw i64 %220, %const_reg_qword9, !spirv.Decorations !837
  %223 = add nsw i64 %222, %221, !spirv.Decorations !837
  %224 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %218, i64 %184
  %225 = load float, float* %224, align 4
  %226 = fmul reassoc nsz arcp contract float %225, %1, !spirv.Decorations !844
  br i1 %57, label %227, label %237

227:                                              ; preds = %217
  %228 = mul nsw i64 %220, %const_reg_qword7, !spirv.Decorations !837
  %229 = getelementptr float, float addrspace(4)* %75, i64 %228
  %230 = getelementptr float, float addrspace(4)* %229, i64 %221
  %231 = addrspacecast float addrspace(4)* %230 to float addrspace(1)*
  %232 = load float, float addrspace(1)* %231, align 4
  %233 = fmul reassoc nsz arcp contract float %232, %4, !spirv.Decorations !844
  %234 = fadd reassoc nsz arcp contract float %226, %233, !spirv.Decorations !844
  %235 = getelementptr inbounds float, float addrspace(4)* %76, i64 %223
  %236 = addrspacecast float addrspace(4)* %235 to float addrspace(1)*
  store float %234, float addrspace(1)* %236, align 4
  br label %240

237:                                              ; preds = %217
  %238 = getelementptr inbounds float, float addrspace(4)* %76, i64 %223
  %239 = addrspacecast float addrspace(4)* %238 to float addrspace(1)*
  store float %226, float addrspace(1)* %239, align 4
  br label %240

240:                                              ; preds = %237, %227, %211
  %241 = add nuw nsw i32 %193, 1, !spirv.Decorations !833
  br label %192, !llvm.loop !908

242:                                              ; preds = %185
  %243 = zext i32 %16 to i64
  %244 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %244)
  %245 = mul nsw i64 %243, %11, !spirv.Decorations !837
  %246 = getelementptr inbounds float, float addrspace(4)* %75, i64 %245
  br label %247

247:                                              ; preds = %242, %185
  %248 = phi float addrspace(4)* [ %246, %242 ], [ %75, %185 ]
  %249 = zext i32 %16 to i64
  %250 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %250)
  %251 = mul nsw i64 %249, %12, !spirv.Decorations !837
  %252 = getelementptr inbounds float, float addrspace(4)* %76, i64 %251
  %253 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %253)
  %254 = add i32 %77, %16
  br label %72

255:                                              ; preds = %72
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func float @__builtin_IB_frnd_zi(float noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func signext i16 @__builtin_IB_ftobf_1(float noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_global_offset(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_num_groups(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_global_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #3

declare i32 @printf(i8 addrspace(2)*, ...)

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind willreturn memory(none)
declare i16 @llvm.genx.GenISA.ftobf.i16.f32(float, i32) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.trunc.f32(float) #6

define spir_kernel void @Intel_Symbol_Table_Void_Program() {
entry:
  ret void
}

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!spirv.MemoryModel = !{!2, !2}
!spirv.Source = !{!3, !3}
!spirv.Generator = !{!4, !4}
!igc.functions = !{!5, !45, !63, !79, !84, !88, !89, !95, !96, !117, !131, !132, !133, !134}
!IGCMetadata = !{!136}
!opencl.ocl.version = !{!830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830}
!opencl.spir.version = !{!830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830, !830}
!llvm.ident = !{!831, !831, !831, !831, !831, !831, !831, !831, !831, !831, !831, !831, !831}
!llvm.module.flags = !{!832}

!0 = !{!1}
!1 = !{i32 44, i32 8}
!2 = !{i32 2, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{i16 6, i16 14}
!5 = !{void (%"class.sycl::_V1::range"*, %class.__generated_*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE, !6}
!6 = !{!7, !8}
!7 = !{!"function_type", i32 0}
!8 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !21, !23, !25, !27, !28, !29, !31, !33, !35, !37, !39, !41, !43}
!9 = !{i32 0}
!10 = !{i32 2}
!11 = !{i32 5}
!12 = !{i32 7}
!13 = !{i32 8}
!14 = !{i32 9}
!15 = !{i32 10}
!16 = !{i32 11}
!17 = !{i32 13}
!18 = !{i32 17, !19, !20}
!19 = !{!"explicit_arg_num", i32 0}
!20 = !{!"struct_arg_offset", i32 0}
!21 = !{i32 17, !19, !22}
!22 = !{!"struct_arg_offset", i32 8}
!23 = !{i32 17, !19, !24}
!24 = !{!"struct_arg_offset", i32 16}
!25 = !{i32 17, !26, !20}
!26 = !{!"explicit_arg_num", i32 1}
!27 = !{i32 17, !26, !22}
!28 = !{i32 17, !26, !24}
!29 = !{i32 17, !26, !30}
!30 = !{!"struct_arg_offset", i32 24}
!31 = !{i32 17, !26, !32}
!32 = !{!"struct_arg_offset", i32 32}
!33 = !{i32 17, !26, !34}
!34 = !{!"struct_arg_offset", i32 40}
!35 = !{i32 17, !26, !36}
!36 = !{!"struct_arg_offset", i32 48}
!37 = !{i32 17, !26, !38}
!38 = !{!"struct_arg_offset", i32 56}
!39 = !{i32 17, !26, !40}
!40 = !{!"struct_arg_offset", i32 64}
!41 = !{i32 17, !26, !42}
!42 = !{!"struct_arg_offset", i32 72}
!43 = !{i32 59, !44}
!44 = !{!"explicit_arg_num", i32 9}
!45 = !{void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail, !46}
!46 = !{!7, !47}
!47 = !{!"implicit_arg_desc", !9, !10, !12, !13, !14, !15, !16, !17, !48, !50, !51, !52, !54, !55, !56, !57, !59, !60, !61}
!48 = !{i32 17, !49, !20}
!49 = !{!"explicit_arg_num", i32 2}
!50 = !{i32 17, !49, !22}
!51 = !{i32 17, !49, !24}
!52 = !{i32 17, !53, !20}
!53 = !{!"explicit_arg_num", i32 5}
!54 = !{i32 17, !53, !22}
!55 = !{i32 17, !53, !24}
!56 = !{i32 15, !19}
!57 = !{i32 15, !58}
!58 = !{!"explicit_arg_num", i32 3}
!59 = !{i32 59, !19}
!60 = !{i32 59, !58}
!61 = !{i32 59, !62}
!62 = !{!"explicit_arg_num", i32 12}
!63 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !64}
!64 = !{!7, !65}
!65 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !66, !67, !69, !71, !73, !75, !77, !43}
!66 = !{i32 19, !26, !22}
!67 = !{i32 20, !26, !68}
!68 = !{!"struct_arg_offset", i32 10}
!69 = !{i32 20, !26, !70}
!70 = !{!"struct_arg_offset", i32 11}
!71 = !{i32 20, !26, !72}
!72 = !{!"struct_arg_offset", i32 12}
!73 = !{i32 20, !26, !74}
!74 = !{!"struct_arg_offset", i32 13}
!75 = !{i32 20, !26, !76}
!76 = !{!"struct_arg_offset", i32 14}
!77 = !{i32 20, !26, !78}
!78 = !{!"struct_arg_offset", i32 15}
!79 = !{void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_, !80}
!80 = !{!7, !81}
!81 = !{!"implicit_arg_desc", !9, !10, !12, !13, !14, !15, !16, !56, !59, !82}
!82 = !{i32 59, !83}
!83 = !{!"explicit_arg_num", i32 8}
!84 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !85}
!85 = !{!7, !86}
!86 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !87, !71, !73, !75, !77, !43}
!87 = !{i32 18, !26, !22}
!88 = !{void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_, !80}
!89 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !90}
!90 = !{!7, !91}
!91 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !92, !93, !67, !69, !71, !73, !75, !77, !43}
!92 = !{i32 20, !26, !22}
!93 = !{i32 20, !26, !94}
!94 = !{!"struct_arg_offset", i32 9}
!95 = !{void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_, !80}
!96 = !{void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE, !97}
!97 = !{!7, !98}
!98 = !{!"implicit_arg_desc", !9, !10, !99, !100, !13, !14, !15, !16, !17, !48, !101, !102, !103, !104, !106, !107, !109, !111, !113, !56, !59, !115}
!99 = !{i32 4}
!100 = !{i32 6}
!101 = !{i32 16, !49, !22}
!102 = !{i32 16, !49, !72}
!103 = !{i32 18, !49, !24}
!104 = !{i32 16, !49, !105}
!105 = !{!"struct_arg_offset", i32 20}
!106 = !{i32 16, !49, !30}
!107 = !{i32 20, !49, !108}
!108 = !{!"struct_arg_offset", i32 28}
!109 = !{i32 20, !49, !110}
!110 = !{!"struct_arg_offset", i32 29}
!111 = !{i32 20, !49, !112}
!112 = !{!"struct_arg_offset", i32 30}
!113 = !{i32 20, !49, !114}
!114 = !{!"struct_arg_offset", i32 31}
!115 = !{i32 59, !116}
!116 = !{!"explicit_arg_num", i32 10}
!117 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE, !118}
!118 = !{!7, !119}
!119 = !{!"implicit_arg_desc", !9, !10, !99, !100, !13, !14, !15, !16, !17, !120, !121, !123, !48, !50, !124, !125, !52, !54, !126, !128, !129}
!120 = !{i32 18, !19, !20}
!121 = !{i32 18, !19, !122}
!122 = !{!"struct_arg_offset", i32 4}
!123 = !{i32 18, !19, !22}
!124 = !{i32 17, !58, !20}
!125 = !{i32 17, !58, !22}
!126 = !{i32 17, !127, !20}
!127 = !{!"explicit_arg_num", i32 6}
!128 = !{i32 17, !127, !22}
!129 = !{i32 59, !130}
!130 = !{!"explicit_arg_num", i32 20}
!131 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !118}
!132 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE, !118}
!133 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !118}
!134 = !{void ()* @Intel_Symbol_Table_Void_Program, !135}
!135 = !{!7}
!136 = !{!"ModuleMD", !137, !138, !276, !589, !620, !642, !643, !647, !650, !651, !652, !691, !716, !730, !731, !732, !749, !750, !751, !752, !756, !757, !765, !766, !767, !768, !769, !770, !771, !772, !773, !774, !775, !780, !782, !786, !787, !788, !789, !790, !791, !792, !793, !794, !795, !796, !797, !798, !799, !800, !801, !802, !803, !804, !805, !356, !806, !807, !808, !810, !812, !815, !816, !817, !819, !820, !821, !826, !827, !828, !829}
!137 = !{!"isPrecise", i1 false}
!138 = !{!"compOpt", !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252, !253, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !275}
!139 = !{!"DenormsAreZero", i1 false}
!140 = !{!"BFTFDenormsAreZero", i1 false}
!141 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!142 = !{!"OptDisable", i1 false}
!143 = !{!"MadEnable", i1 true}
!144 = !{!"NoSignedZeros", i1 false}
!145 = !{!"NoNaNs", i1 false}
!146 = !{!"FloatDenormMode16", !"FLOAT_DENORM_RETAIN"}
!147 = !{!"FloatDenormMode32", !"FLOAT_DENORM_RETAIN"}
!148 = !{!"FloatDenormMode64", !"FLOAT_DENORM_RETAIN"}
!149 = !{!"FloatDenormModeBFTF", !"FLOAT_DENORM_RETAIN"}
!150 = !{!"FloatRoundingMode", i32 0}
!151 = !{!"FloatCvtIntRoundingMode", i32 3}
!152 = !{!"LoadCacheDefault", i32 4}
!153 = !{!"StoreCacheDefault", i32 2}
!154 = !{!"VISAPreSchedRPThreshold", i32 0}
!155 = !{!"VISAPreSchedCtrl", i32 0}
!156 = !{!"SetLoopUnrollThreshold", i32 0}
!157 = !{!"UnsafeMathOptimizations", i1 false}
!158 = !{!"disableCustomUnsafeOpts", i1 false}
!159 = !{!"disableReducePow", i1 false}
!160 = !{!"disableSqrtOpt", i1 false}
!161 = !{!"FiniteMathOnly", i1 false}
!162 = !{!"FastRelaxedMath", i1 false}
!163 = !{!"DashGSpecified", i1 false}
!164 = !{!"FastCompilation", i1 false}
!165 = !{!"UseScratchSpacePrivateMemory", i1 true}
!166 = !{!"RelaxedBuiltins", i1 false}
!167 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!168 = !{!"GreaterThan2GBBufferRequired", i1 true}
!169 = !{!"GreaterThan4GBBufferRequired", i1 true}
!170 = !{!"DisableA64WA", i1 false}
!171 = !{!"ForceEnableA64WA", i1 false}
!172 = !{!"PushConstantsEnable", i1 true}
!173 = !{!"HasPositivePointerOffset", i1 false}
!174 = !{!"HasBufferOffsetArg", i1 true}
!175 = !{!"BufferOffsetArgOptional", i1 true}
!176 = !{!"replaceGlobalOffsetsByZero", i1 false}
!177 = !{!"forcePixelShaderSIMDMode", i32 0}
!178 = !{!"forceTotalGRFNum", i32 0}
!179 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!180 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!181 = !{!"UniformWGS", i1 false}
!182 = !{!"disableVertexComponentPacking", i1 false}
!183 = !{!"disablePartialVertexComponentPacking", i1 false}
!184 = !{!"PreferBindlessImages", i1 true}
!185 = !{!"UseBindlessMode", i1 true}
!186 = !{!"UseLegacyBindlessMode", i1 false}
!187 = !{!"disableMathRefactoring", i1 false}
!188 = !{!"atomicBranch", i1 false}
!189 = !{!"spillCompression", i1 false}
!190 = !{!"AllowLICM", i1 true}
!191 = !{!"DisableEarlyOut", i1 false}
!192 = !{!"ForceInt32DivRemEmu", i1 false}
!193 = !{!"ForceInt32DivRemEmuSP", i1 false}
!194 = !{!"DisableIntDivRemIncrementReduction", i1 false}
!195 = !{!"WaveIntrinsicUsed", i1 false}
!196 = !{!"DisableMultiPolyPS", i1 false}
!197 = !{!"NeedTexture3DLODWA", i1 false}
!198 = !{!"UseLivePrologueKernelForRaytracingDispatch", i1 false}
!199 = !{!"DisableFastestSingleCSSIMD", i1 false}
!200 = !{!"DisableFastestLinearScan", i1 false}
!201 = !{!"UseStatelessforPrivateMemory", i1 false}
!202 = !{!"EnableTakeGlobalAddress", i1 false}
!203 = !{!"IsLibraryCompilation", i1 false}
!204 = !{!"LibraryCompileSIMDSize", i32 0}
!205 = !{!"FastVISACompile", i1 false}
!206 = !{!"MatchSinCosPi", i1 false}
!207 = !{!"ExcludeIRFromZEBinary", i1 false}
!208 = !{!"EmitZeBinVISASections", i1 false}
!209 = !{!"FP64GenEmulationEnabled", i1 false}
!210 = !{!"FP64GenConvEmulationEnabled", i1 false}
!211 = !{!"allowDisableRematforCS", i1 false}
!212 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!213 = !{!"DisableCPSOmaskWA", i1 false}
!214 = !{!"DisableFastestGopt", i1 false}
!215 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!216 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!217 = !{!"DisableConstantCoalescing", i1 false}
!218 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!219 = !{!"WaEnableALTModeVisaWA", i1 false}
!220 = !{!"EnableLdStCombineforLoad", i1 false}
!221 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!222 = !{!"WaEnableAtomicWaveFusion", i1 false}
!223 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!224 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!225 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!226 = !{!"WaEnableAtomicWaveFusionPartial", i1 false}
!227 = !{!"WaEnableAtomicWaveFusionMoreDimensions", i1 false}
!228 = !{!"WaEnableAtomicWaveFusionLoop", i1 false}
!229 = !{!"WaEnableAtomicWaveFusionReturnValuePolicy", i32 0}
!230 = !{!"ForceCBThroughSampler3D", i1 false}
!231 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!232 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!233 = !{!"WaTypedAtomicBinToRawAtomicBin", i1 false}
!234 = !{!"WaRawAtomicBinToTypedAtomicBin", i1 false}
!235 = !{!"WaSampleLoadToTypedRead", i1 false}
!236 = !{!"EnableTypedBufferStoreToUntypedStore", i1 false}
!237 = !{!"WaZeroSLMBeforeUse", i1 false}
!238 = !{!"EnableEmitMoreMoviCases", i1 false}
!239 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!240 = !{!"EnableFastSampleD", i1 false}
!241 = !{!"ForceUniformBuffer", i1 false}
!242 = !{!"ForceUniformSurfaceSampler", i1 false}
!243 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!244 = !{!"NewSpillCostFunction", i1 false}
!245 = !{!"EnableVRT", i1 false}
!246 = !{!"ForceLargeGRFNum4RQ", i1 false}
!247 = !{!"Enable2xGRFRetry", i1 false}
!248 = !{!"Detect2xGRFCandidate", i1 false}
!249 = !{!"EnableURBWritesMerging", i1 true}
!250 = !{!"ForceCacheLineAlignedURBWriteMerging", i1 false}
!251 = !{!"DisableURBLayoutAlignmentToCacheLine", i1 false}
!252 = !{!"DisableEUFusion", i1 false}
!253 = !{!"DisableFDivToFMulInvOpt", i1 false}
!254 = !{!"initializePhiSampleSourceWA", i1 false}
!255 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!256 = !{!"DisableLoosenSimd32Occu", i1 false}
!257 = !{!"FastestS1Options", i32 0}
!258 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!259 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!260 = !{!"LscSamplerRouting", i32 0}
!261 = !{!"UseBarrierControlFlowOptimization", i1 false}
!262 = !{!"EnableDynamicRQManagement", i1 false}
!263 = !{!"WaDisablePayloadCoalescing", i1 false}
!264 = !{!"Quad8InputThreshold", i32 0}
!265 = !{!"UseResourceLoopUnrollNested", i1 false}
!266 = !{!"DisableLoopUnroll", i1 false}
!267 = !{!"ForcePushConstantMode", i32 0}
!268 = !{!"UseInstructionHoistingOptimization", i1 false}
!269 = !{!"DisableResourceLoopDestLifeTimeStart", i1 false}
!270 = !{!"ForceVRTGRFCeiling", i32 0}
!271 = !{!"DisableSamplerBackingByLSC", i32 0}
!272 = !{!"UseLinearScanRA", i1 false}
!273 = !{!"DisableConvertingAtomicIAddToIncDec", i1 false}
!274 = !{!"EnableInlinedCrossThreadData", i1 false}
!275 = !{!"ZeroInitRegistersBeforeExecution", i1 false}
!276 = !{!"FuncMD", !277, !278, !390, !391, !428, !429, !440, !441, !451, !452, !459, !460, !467, !468, !475, !476, !481, !482, !491, !492, !570, !571, !572, !573, !574, !575, !576, !577}
!277 = !{!"FuncMDMap[0]", void (%"class.sycl::_V1::range"*, %class.__generated_*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE}
!278 = !{!"FuncMDValue[0]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !314, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !374, !377, !380, !383, !386, !389}
!279 = !{!"localOffsets"}
!280 = !{!"workGroupWalkOrder", !281, !282, !283}
!281 = !{!"dim0", i32 0}
!282 = !{!"dim1", i32 1}
!283 = !{!"dim2", i32 2}
!284 = !{!"funcArgs"}
!285 = !{!"functionType", !"KernelFunction"}
!286 = !{!"inlineDynConstants"}
!287 = !{!"inlineDynRootConstant"}
!288 = !{!"inlineDynConstantDescTable"}
!289 = !{!"m_pInterestingConstants"}
!290 = !{!"rtInfo", !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307, !309, !310, !311, !312, !313}
!291 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!292 = !{!"isContinuation", i1 false}
!293 = !{!"isMonolithic", i1 false}
!294 = !{!"hasTraceRayPayload", i1 false}
!295 = !{!"hasHitAttributes", i1 false}
!296 = !{!"hasCallableData", i1 false}
!297 = !{!"ShaderStackSize", i32 0}
!298 = !{!"ShaderHash", i64 0}
!299 = !{!"ShaderName", !""}
!300 = !{!"ParentName", !""}
!301 = !{!"SlotNum", i1* null}
!302 = !{!"NOSSize", i32 0}
!303 = !{!"globalRootSignatureSize", i32 0}
!304 = !{!"Entries"}
!305 = !{!"SpillUnions"}
!306 = !{!"CustomHitAttrSizeInBytes", i32 0}
!307 = !{!"Types", !308}
!308 = !{!"FullFrameTys"}
!309 = !{!"Aliases"}
!310 = !{!"numSyncRTStacks", i32 0}
!311 = !{!"NumCoherenceHintBits", i32 0}
!312 = !{!"useSyncHWStack", i1 false}
!313 = !{!"OriginatingShaderName", !""}
!314 = !{!"resAllocMD", !315, !316, !317, !318, !347}
!315 = !{!"uavsNumType", i32 0}
!316 = !{!"srvsNumType", i32 0}
!317 = !{!"samplersNumType", i32 0}
!318 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346}
!319 = !{!"argAllocMDListVec[0]", !320, !321, !322}
!320 = !{!"type", i32 0}
!321 = !{!"extensionType", i32 -1}
!322 = !{!"indexType", i32 -1}
!323 = !{!"argAllocMDListVec[1]", !320, !321, !322}
!324 = !{!"argAllocMDListVec[2]", !320, !321, !322}
!325 = !{!"argAllocMDListVec[3]", !320, !321, !322}
!326 = !{!"argAllocMDListVec[4]", !320, !321, !322}
!327 = !{!"argAllocMDListVec[5]", !320, !321, !322}
!328 = !{!"argAllocMDListVec[6]", !320, !321, !322}
!329 = !{!"argAllocMDListVec[7]", !320, !321, !322}
!330 = !{!"argAllocMDListVec[8]", !320, !321, !322}
!331 = !{!"argAllocMDListVec[9]", !320, !321, !322}
!332 = !{!"argAllocMDListVec[10]", !320, !321, !322}
!333 = !{!"argAllocMDListVec[11]", !320, !321, !322}
!334 = !{!"argAllocMDListVec[12]", !320, !321, !322}
!335 = !{!"argAllocMDListVec[13]", !320, !321, !322}
!336 = !{!"argAllocMDListVec[14]", !320, !321, !322}
!337 = !{!"argAllocMDListVec[15]", !320, !321, !322}
!338 = !{!"argAllocMDListVec[16]", !320, !321, !322}
!339 = !{!"argAllocMDListVec[17]", !320, !321, !322}
!340 = !{!"argAllocMDListVec[18]", !320, !321, !322}
!341 = !{!"argAllocMDListVec[19]", !320, !321, !322}
!342 = !{!"argAllocMDListVec[20]", !320, !321, !322}
!343 = !{!"argAllocMDListVec[21]", !320, !321, !322}
!344 = !{!"argAllocMDListVec[22]", !320, !321, !322}
!345 = !{!"argAllocMDListVec[23]", !320, !321, !322}
!346 = !{!"argAllocMDListVec[24]", !320, !321, !322}
!347 = !{!"inlineSamplersMD"}
!348 = !{!"maxByteOffsets"}
!349 = !{!"IsInitializer", i1 false}
!350 = !{!"IsFinalizer", i1 false}
!351 = !{!"CompiledSubGroupsNumber", i32 0}
!352 = !{!"hasInlineVmeSamplers", i1 false}
!353 = !{!"localSize", i32 0}
!354 = !{!"localIDPresent", i1 false}
!355 = !{!"groupIDPresent", i1 false}
!356 = !{!"privateMemoryPerWI", i32 0}
!357 = !{!"prevFPOffset", i32 0}
!358 = !{!"globalIDPresent", i1 false}
!359 = !{!"hasSyncRTCalls", i1 false}
!360 = !{!"hasPrintfCalls", i1 false}
!361 = !{!"requireAssertBuffer", i1 false}
!362 = !{!"requireSyncBuffer", i1 false}
!363 = !{!"hasIndirectCalls", i1 false}
!364 = !{!"hasNonKernelArgLoad", i1 false}
!365 = !{!"hasNonKernelArgStore", i1 false}
!366 = !{!"hasNonKernelArgAtomic", i1 false}
!367 = !{!"UserAnnotations"}
!368 = !{!"m_OpenCLArgAddressSpaces", !369, !370}
!369 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 0}
!370 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!371 = !{!"m_OpenCLArgAccessQualifiers", !372, !373}
!372 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!373 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!374 = !{!"m_OpenCLArgTypes", !375, !376}
!375 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::range"}
!376 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_"}
!377 = !{!"m_OpenCLArgBaseTypes", !378, !379}
!378 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::range"}
!379 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_"}
!380 = !{!"m_OpenCLArgTypeQualifiers", !381, !382}
!381 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!382 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!383 = !{!"m_OpenCLArgNames", !384, !385}
!384 = !{!"m_OpenCLArgNamesVec[0]", !""}
!385 = !{!"m_OpenCLArgNamesVec[1]", !""}
!386 = !{!"m_OpenCLArgScalarAsPointers", !387, !388}
!387 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 14}
!388 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 19}
!389 = !{!"m_OptsToDisablePerFunc"}
!390 = !{!"FuncMDMap[1]", void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail}
!391 = !{!"FuncMDValue[1]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !314, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !392, !398, !403, !410, !417, !422, !427, !389}
!392 = !{!"m_OpenCLArgAddressSpaces", !393, !370, !394, !395, !396, !397}
!393 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!394 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!395 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!396 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 0}
!397 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!398 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !399, !400, !401, !402}
!399 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!400 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!401 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!402 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!403 = !{!"m_OpenCLArgTypes", !404, !405, !406, !407, !408, !409}
!404 = !{!"m_OpenCLArgTypesVec[0]", !"char*"}
!405 = !{!"m_OpenCLArgTypesVec[1]", !"long"}
!406 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!407 = !{!"m_OpenCLArgTypesVec[3]", !"char*"}
!408 = !{!"m_OpenCLArgTypesVec[4]", !"long"}
!409 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!410 = !{!"m_OpenCLArgBaseTypes", !411, !412, !413, !414, !415, !416}
!411 = !{!"m_OpenCLArgBaseTypesVec[0]", !"char*"}
!412 = !{!"m_OpenCLArgBaseTypesVec[1]", !"long"}
!413 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!414 = !{!"m_OpenCLArgBaseTypesVec[3]", !"char*"}
!415 = !{!"m_OpenCLArgBaseTypesVec[4]", !"long"}
!416 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!417 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !418, !419, !420, !421}
!418 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!419 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!420 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!421 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!422 = !{!"m_OpenCLArgNames", !384, !385, !423, !424, !425, !426}
!423 = !{!"m_OpenCLArgNamesVec[2]", !""}
!424 = !{!"m_OpenCLArgNamesVec[3]", !""}
!425 = !{!"m_OpenCLArgNamesVec[4]", !""}
!426 = !{!"m_OpenCLArgNamesVec[5]", !""}
!427 = !{!"m_OpenCLArgScalarAsPointers"}
!428 = !{!"FuncMDMap[2]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!429 = !{!"FuncMDValue[2]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !430, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !432, !435, !380, !383, !438, !389}
!430 = !{!"resAllocMD", !315, !316, !317, !431, !347}
!431 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342}
!432 = !{!"m_OpenCLArgTypes", !433, !434}
!433 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::range.0"}
!434 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.2"}
!435 = !{!"m_OpenCLArgBaseTypes", !436, !437}
!436 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::range.0"}
!437 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.2"}
!438 = !{!"m_OpenCLArgScalarAsPointers", !439}
!439 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 12}
!440 = !{!"FuncMDMap[3]", void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!441 = !{!"FuncMDValue[3]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !442, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !444, !371, !445, !448, !380, !383, !427, !389}
!442 = !{!"resAllocMD", !315, !316, !317, !443, !347}
!443 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333}
!444 = !{!"m_OpenCLArgAddressSpaces", !393, !370}
!445 = !{!"m_OpenCLArgTypes", !446, !447}
!446 = !{!"m_OpenCLArgTypesVec[0]", !"short*"}
!447 = !{!"m_OpenCLArgTypesVec[1]", !"ushort"}
!448 = !{!"m_OpenCLArgBaseTypes", !449, !450}
!449 = !{!"m_OpenCLArgBaseTypesVec[0]", !"short*"}
!450 = !{!"m_OpenCLArgBaseTypesVec[1]", !"ushort"}
!451 = !{!"FuncMDMap[4]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!452 = !{!"FuncMDValue[4]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !453, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !455, !457, !380, !383, !438, !389}
!453 = !{!"resAllocMD", !315, !316, !317, !454, !347}
!454 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340}
!455 = !{!"m_OpenCLArgTypes", !433, !456}
!456 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.9"}
!457 = !{!"m_OpenCLArgBaseTypes", !436, !458}
!458 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.9"}
!459 = !{!"FuncMDMap[5]", void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!460 = !{!"FuncMDValue[5]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !442, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !444, !371, !461, !464, !380, !383, !427, !389}
!461 = !{!"m_OpenCLArgTypes", !462, !463}
!462 = !{!"m_OpenCLArgTypesVec[0]", !"int*"}
!463 = !{!"m_OpenCLArgTypesVec[1]", !"int"}
!464 = !{!"m_OpenCLArgBaseTypes", !465, !466}
!465 = !{!"m_OpenCLArgBaseTypesVec[0]", !"int*"}
!466 = !{!"m_OpenCLArgBaseTypesVec[1]", !"int"}
!467 = !{!"FuncMDMap[6]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!468 = !{!"FuncMDValue[6]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !469, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !471, !473, !380, !383, !438, !389}
!469 = !{!"resAllocMD", !315, !316, !317, !470, !347}
!470 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343}
!471 = !{!"m_OpenCLArgTypes", !433, !472}
!472 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.12"}
!473 = !{!"m_OpenCLArgBaseTypes", !436, !474}
!474 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.12"}
!475 = !{!"FuncMDMap[7]", void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!476 = !{!"FuncMDValue[7]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !442, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !444, !371, !477, !479, !380, !383, !427, !389}
!477 = !{!"m_OpenCLArgTypes", !404, !478}
!478 = !{!"m_OpenCLArgTypesVec[1]", !"uchar"}
!479 = !{!"m_OpenCLArgBaseTypes", !411, !480}
!480 = !{!"m_OpenCLArgBaseTypesVec[1]", !"uchar"}
!481 = !{!"FuncMDMap[8]", void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE}
!482 = !{!"FuncMDValue[8]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !314, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !483, !484, !485, !487, !489, !490, !427, !389}
!483 = !{!"m_OpenCLArgAddressSpaces", !393, !370, !394}
!484 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !399}
!485 = !{!"m_OpenCLArgTypes", !446, !405, !486}
!486 = !{!"m_OpenCLArgTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!487 = !{!"m_OpenCLArgBaseTypes", !449, !412, !488}
!488 = !{!"m_OpenCLArgBaseTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!489 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !418}
!490 = !{!"m_OpenCLArgNames", !384, !385, !423}
!491 = !{!"FuncMDMap[9]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE}
!492 = !{!"FuncMDValue[9]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !493, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !504, !513, !521, !535, !549, !557, !565, !389}
!493 = !{!"resAllocMD", !315, !316, !317, !494, !347}
!494 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !495, !496, !497, !498, !499, !500, !501, !502, !503}
!495 = !{!"argAllocMDListVec[25]", !320, !321, !322}
!496 = !{!"argAllocMDListVec[26]", !320, !321, !322}
!497 = !{!"argAllocMDListVec[27]", !320, !321, !322}
!498 = !{!"argAllocMDListVec[28]", !320, !321, !322}
!499 = !{!"argAllocMDListVec[29]", !320, !321, !322}
!500 = !{!"argAllocMDListVec[30]", !320, !321, !322}
!501 = !{!"argAllocMDListVec[31]", !320, !321, !322}
!502 = !{!"argAllocMDListVec[32]", !320, !321, !322}
!503 = !{!"argAllocMDListVec[33]", !320, !321, !322}
!504 = !{!"m_OpenCLArgAddressSpaces", !369, !370, !394, !505, !396, !397, !506, !507, !508, !509, !510, !511, !512}
!505 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!506 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!507 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 0}
!508 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!509 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!510 = !{!"m_OpenCLArgAddressSpacesVec[10]", i32 0}
!511 = !{!"m_OpenCLArgAddressSpacesVec[11]", i32 0}
!512 = !{!"m_OpenCLArgAddressSpacesVec[12]", i32 0}
!513 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !399, !400, !401, !402, !514, !515, !516, !517, !518, !519, !520}
!514 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!515 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!516 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!517 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!518 = !{!"m_OpenCLArgAccessQualifiersVec[10]", !"none"}
!519 = !{!"m_OpenCLArgAccessQualifiersVec[11]", !"none"}
!520 = !{!"m_OpenCLArgAccessQualifiersVec[12]", !"none"}
!521 = !{!"m_OpenCLArgTypes", !522, !523, !524, !525, !526, !527, !528, !529, !530, !531, !532, !533, !534}
!522 = !{!"m_OpenCLArgTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!523 = !{!"m_OpenCLArgTypesVec[1]", !"float"}
!524 = !{!"m_OpenCLArgTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!525 = !{!"m_OpenCLArgTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!526 = !{!"m_OpenCLArgTypesVec[4]", !"float"}
!527 = !{!"m_OpenCLArgTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!528 = !{!"m_OpenCLArgTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!529 = !{!"m_OpenCLArgTypesVec[7]", !"float"}
!530 = !{!"m_OpenCLArgTypesVec[8]", !"int"}
!531 = !{!"m_OpenCLArgTypesVec[9]", !"long"}
!532 = !{!"m_OpenCLArgTypesVec[10]", !"long"}
!533 = !{!"m_OpenCLArgTypesVec[11]", !"long"}
!534 = !{!"m_OpenCLArgTypesVec[12]", !"long"}
!535 = !{!"m_OpenCLArgBaseTypes", !536, !537, !538, !539, !540, !541, !542, !543, !544, !545, !546, !547, !548}
!536 = !{!"m_OpenCLArgBaseTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!537 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float"}
!538 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!539 = !{!"m_OpenCLArgBaseTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!540 = !{!"m_OpenCLArgBaseTypesVec[4]", !"float"}
!541 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!542 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!543 = !{!"m_OpenCLArgBaseTypesVec[7]", !"float"}
!544 = !{!"m_OpenCLArgBaseTypesVec[8]", !"int"}
!545 = !{!"m_OpenCLArgBaseTypesVec[9]", !"long"}
!546 = !{!"m_OpenCLArgBaseTypesVec[10]", !"long"}
!547 = !{!"m_OpenCLArgBaseTypesVec[11]", !"long"}
!548 = !{!"m_OpenCLArgBaseTypesVec[12]", !"long"}
!549 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !418, !419, !420, !421, !550, !551, !552, !553, !554, !555, !556}
!550 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!551 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!552 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!553 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!554 = !{!"m_OpenCLArgTypeQualifiersVec[10]", !""}
!555 = !{!"m_OpenCLArgTypeQualifiersVec[11]", !""}
!556 = !{!"m_OpenCLArgTypeQualifiersVec[12]", !""}
!557 = !{!"m_OpenCLArgNames", !384, !385, !423, !424, !425, !426, !558, !559, !560, !561, !562, !563, !564}
!558 = !{!"m_OpenCLArgNamesVec[6]", !""}
!559 = !{!"m_OpenCLArgNamesVec[7]", !""}
!560 = !{!"m_OpenCLArgNamesVec[8]", !""}
!561 = !{!"m_OpenCLArgNamesVec[9]", !""}
!562 = !{!"m_OpenCLArgNamesVec[10]", !""}
!563 = !{!"m_OpenCLArgNamesVec[11]", !""}
!564 = !{!"m_OpenCLArgNamesVec[12]", !""}
!565 = !{!"m_OpenCLArgScalarAsPointers", !566, !567, !568, !569}
!566 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 25}
!567 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 27}
!568 = !{!"m_OpenCLArgScalarAsPointersSet[2]", i32 29}
!569 = !{!"m_OpenCLArgScalarAsPointersSet[3]", i32 31}
!570 = !{!"FuncMDMap[10]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!571 = !{!"FuncMDValue[10]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !493, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !504, !513, !521, !535, !549, !557, !565, !389}
!572 = !{!"FuncMDMap[11]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE}
!573 = !{!"FuncMDValue[11]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !493, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !504, !513, !521, !535, !549, !557, !565, !389}
!574 = !{!"FuncMDMap[12]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!575 = !{!"FuncMDValue[12]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !493, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !504, !513, !521, !535, !549, !557, !565, !389}
!576 = !{!"FuncMDMap[13]", void ()* @Intel_Symbol_Table_Void_Program}
!577 = !{!"FuncMDValue[13]", !279, !578, !284, !285, !286, !287, !288, !289, !290, !581, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !583, !584, !585, !586, !587, !588, !427, !389}
!578 = !{!"workGroupWalkOrder", !281, !579, !580}
!579 = !{!"dim1", i32 0}
!580 = !{!"dim2", i32 0}
!581 = !{!"resAllocMD", !315, !316, !317, !582, !347}
!582 = !{!"argAllocMDList"}
!583 = !{!"m_OpenCLArgAddressSpaces"}
!584 = !{!"m_OpenCLArgAccessQualifiers"}
!585 = !{!"m_OpenCLArgTypes"}
!586 = !{!"m_OpenCLArgBaseTypes"}
!587 = !{!"m_OpenCLArgTypeQualifiers"}
!588 = !{!"m_OpenCLArgNames"}
!589 = !{!"pushInfo", !590, !591, !592, !596, !597, !598, !599, !600, !601, !602, !603, !616, !617, !618, !619}
!590 = !{!"pushableAddresses"}
!591 = !{!"bindlessPushInfo"}
!592 = !{!"dynamicBufferInfo", !593, !594, !595}
!593 = !{!"firstIndex", i32 0}
!594 = !{!"numOffsets", i32 0}
!595 = !{!"forceDisabled", i1 false}
!596 = !{!"MaxNumberOfPushedBuffers", i32 0}
!597 = !{!"inlineConstantBufferSlot", i32 -1}
!598 = !{!"inlineConstantBufferOffset", i32 -1}
!599 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!600 = !{!"constants"}
!601 = !{!"inputs"}
!602 = !{!"constantReg"}
!603 = !{!"simplePushInfoArr", !604, !613, !614, !615}
!604 = !{!"simplePushInfoArrVec[0]", !605, !606, !607, !608, !609, !610, !611, !612}
!605 = !{!"cbIdx", i32 0}
!606 = !{!"pushableAddressGrfOffset", i32 -1}
!607 = !{!"pushableOffsetGrfOffset", i32 -1}
!608 = !{!"offset", i32 0}
!609 = !{!"size", i32 0}
!610 = !{!"isStateless", i1 false}
!611 = !{!"isBindless", i1 false}
!612 = !{!"simplePushLoads"}
!613 = !{!"simplePushInfoArrVec[1]", !605, !606, !607, !608, !609, !610, !611, !612}
!614 = !{!"simplePushInfoArrVec[2]", !605, !606, !607, !608, !609, !610, !611, !612}
!615 = !{!"simplePushInfoArrVec[3]", !605, !606, !607, !608, !609, !610, !611, !612}
!616 = !{!"simplePushBufferUsed", i32 0}
!617 = !{!"pushAnalysisWIInfos"}
!618 = !{!"inlineRTGlobalPtrOffset", i32 0}
!619 = !{!"rtSyncSurfPtrOffset", i32 0}
!620 = !{!"pISAInfo", !621, !622, !626, !627, !635, !639, !641}
!621 = !{!"shaderType", !"UNKNOWN"}
!622 = !{!"geometryInfo", !623, !624, !625}
!623 = !{!"needsVertexHandles", i1 false}
!624 = !{!"needsPrimitiveIDEnable", i1 false}
!625 = !{!"VertexCount", i32 0}
!626 = !{!"hullInfo", !623, !624}
!627 = !{!"pixelInfo", !628, !629, !630, !631, !632, !633, !634}
!628 = !{!"perPolyStartGrf", i32 0}
!629 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!630 = !{!"hasNonPerspBaryPlanes", i1 false}
!631 = !{!"maxPerPrimConstDataId", i32 -1}
!632 = !{!"maxSetupId", i32 -1}
!633 = !{!"hasVMask", i1 false}
!634 = !{!"PixelGRFBitmask", i32 0}
!635 = !{!"domainInfo", !636, !637, !638}
!636 = !{!"DomainPointUArgIdx", i32 -1}
!637 = !{!"DomainPointVArgIdx", i32 -1}
!638 = !{!"DomainPointWArgIdx", i32 -1}
!639 = !{!"computeInfo", !640}
!640 = !{!"EnableHWGenerateLID", i1 true}
!641 = !{!"URBOutputLength", i32 0}
!642 = !{!"WaEnableICBPromotion", i1 false}
!643 = !{!"vsInfo", !644, !645, !646}
!644 = !{!"DrawIndirectBufferIndex", i32 -1}
!645 = !{!"vertexReordering", i32 -1}
!646 = !{!"MaxNumOfOutputs", i32 0}
!647 = !{!"hsInfo", !648, !649}
!648 = !{!"numPatchAttributesPatchBaseName", !""}
!649 = !{!"numVertexAttributesPatchBaseName", !""}
!650 = !{!"dsInfo", !646}
!651 = !{!"gsInfo", !646}
!652 = !{!"psInfo", !653, !654, !655, !656, !657, !658, !659, !660, !661, !662, !663, !664, !665, !666, !667, !668, !669, !670, !671, !672, !673, !674, !675, !676, !677, !678, !679, !680, !681, !682, !683, !684, !685, !686, !687, !688, !689, !690}
!653 = !{!"BlendStateDisabledMask", i8 0}
!654 = !{!"SkipSrc0Alpha", i1 false}
!655 = !{!"DualSourceBlendingDisabled", i1 false}
!656 = !{!"ForceEnableSimd32", i1 false}
!657 = !{!"DisableSimd32WithDiscard", i1 false}
!658 = !{!"outputDepth", i1 false}
!659 = !{!"outputStencil", i1 false}
!660 = !{!"outputMask", i1 false}
!661 = !{!"blendToFillEnabled", i1 false}
!662 = !{!"forceEarlyZ", i1 false}
!663 = !{!"hasVersionedLoop", i1 false}
!664 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!665 = !{!"requestCPSizeRelevant", i1 false}
!666 = !{!"requestCPSize", i1 false}
!667 = !{!"texelMaskFastClearMode", !"Disabled"}
!668 = !{!"NumSamples", i8 0}
!669 = !{!"blendOptimizationMode"}
!670 = !{!"colorOutputMask"}
!671 = !{!"ProvokingVertexModeNosIndex", i32 0}
!672 = !{!"ProvokingVertexModeNosPatch", !""}
!673 = !{!"ProvokingVertexModeLast", !"Negative"}
!674 = !{!"VertexAttributesBypass", i1 false}
!675 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!676 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!677 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!678 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!679 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!680 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!681 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!682 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!683 = !{!"generatePatchesForRTWriteSends", i1 false}
!684 = !{!"generatePatchesForRT_BTIndex", i1 false}
!685 = !{!"forceVMask", i1 false}
!686 = !{!"isNumPerPrimAttributesSet", i1 false}
!687 = !{!"numPerPrimAttributes", i32 0}
!688 = !{!"WaDisableVRS", i1 false}
!689 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!690 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!691 = !{!"csInfo", !692, !693, !694, !695, !178, !154, !155, !696, !156, !697, !698, !699, !700, !701, !702, !703, !704, !705, !706, !707, !189, !708, !709, !710, !711, !712, !713, !714, !715}
!692 = !{!"maxWorkGroupSize", i32 0}
!693 = !{!"waveSize", i32 0}
!694 = !{!"ComputeShaderSecondCompile"}
!695 = !{!"forcedSIMDSize", i8 0}
!696 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!697 = !{!"forceSpillCompression", i1 false}
!698 = !{!"allowLowerSimd", i1 false}
!699 = !{!"disableSimd32Slicing", i1 false}
!700 = !{!"disableSplitOnSpill", i1 false}
!701 = !{!"enableNewSpillCostFunction", i1 false}
!702 = !{!"forceVISAPreSched", i1 false}
!703 = !{!"disableLocalIdOrderOptimizations", i1 false}
!704 = !{!"disableDispatchAlongY", i1 false}
!705 = !{!"neededThreadIdLayout", i1* null}
!706 = !{!"forceTileYWalk", i1 false}
!707 = !{!"atomicBranch", i32 0}
!708 = !{!"disableEarlyOut", i1 false}
!709 = !{!"walkOrderEnabled", i1 false}
!710 = !{!"walkOrderOverride", i32 0}
!711 = !{!"ResForHfPacking"}
!712 = !{!"constantFoldSimdSize", i1 false}
!713 = !{!"isNodeShader", i1 false}
!714 = !{!"threadGroupMergeSize", i32 0}
!715 = !{!"threadGroupMergeOverY", i1 false}
!716 = !{!"msInfo", !717, !718, !719, !720, !721, !722, !723, !724, !725, !726, !727, !673, !671, !728, !729, !713}
!717 = !{!"PrimitiveTopology", i32 3}
!718 = !{!"MaxNumOfPrimitives", i32 0}
!719 = !{!"MaxNumOfVertices", i32 0}
!720 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!721 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!722 = !{!"WorkGroupSize", i32 0}
!723 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!724 = !{!"IndexFormat", i32 6}
!725 = !{!"SubgroupSize", i32 0}
!726 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!727 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!728 = !{!"Is16BMUEModeAllowed", i1 false}
!729 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!730 = !{!"taskInfo", !646, !722, !723, !725}
!731 = !{!"NBarrierCnt", i32 0}
!732 = !{!"rtInfo", !733, !734, !735, !736, !737, !738, !739, !740, !741, !742, !743, !744, !745, !746, !747, !748, !310}
!733 = !{!"RayQueryAllocSizeInBytes", i32 0}
!734 = !{!"NumContinuations", i32 0}
!735 = !{!"RTAsyncStackAddrspace", i32 -1}
!736 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!737 = !{!"SWHotZoneAddrspace", i32 -1}
!738 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!739 = !{!"SWStackAddrspace", i32 -1}
!740 = !{!"SWStackSurfaceStateOffset", i1* null}
!741 = !{!"RTSyncStackAddrspace", i32 -1}
!742 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!743 = !{!"doSyncDispatchRays", i1 false}
!744 = !{!"MemStyle", !"Xe"}
!745 = !{!"GlobalDataStyle", !"Xe"}
!746 = !{!"NeedsBTD", i1 true}
!747 = !{!"SERHitObjectFullType", i1* null}
!748 = !{!"uberTileDimensions", i1* null}
!749 = !{!"CurUniqueIndirectIdx", i32 0}
!750 = !{!"inlineDynTextures"}
!751 = !{!"inlineResInfoData"}
!752 = !{!"immConstant", !753, !754, !755}
!753 = !{!"data"}
!754 = !{!"sizes"}
!755 = !{!"zeroIdxs"}
!756 = !{!"stringConstants"}
!757 = !{!"inlineBuffers", !758, !762, !764}
!758 = !{!"inlineBuffersVec[0]", !759, !760, !761}
!759 = !{!"alignment", i32 0}
!760 = !{!"allocSize", i64 64}
!761 = !{!"Buffer"}
!762 = !{!"inlineBuffersVec[1]", !759, !763, !761}
!763 = !{!"allocSize", i64 0}
!764 = !{!"inlineBuffersVec[2]", !759, !763, !761}
!765 = !{!"GlobalPointerProgramBinaryInfos"}
!766 = !{!"ConstantPointerProgramBinaryInfos"}
!767 = !{!"GlobalBufferAddressRelocInfo"}
!768 = !{!"ConstantBufferAddressRelocInfo"}
!769 = !{!"forceLscCacheList"}
!770 = !{!"SrvMap"}
!771 = !{!"RootConstantBufferOffsetInBytes"}
!772 = !{!"RasterizerOrderedByteAddressBuffer"}
!773 = !{!"RasterizerOrderedViews"}
!774 = !{!"MinNOSPushConstantSize", i32 0}
!775 = !{!"inlineProgramScopeOffsets", !776, !777, !778, !779}
!776 = !{!"inlineProgramScopeOffsetsMap[0]", [36 x i8]* @gVar}
!777 = !{!"inlineProgramScopeOffsetsValue[0]", i64 0}
!778 = !{!"inlineProgramScopeOffsetsMap[1]", [24 x i8]* @gVar.61}
!779 = !{!"inlineProgramScopeOffsetsValue[1]", i64 40}
!780 = !{!"shaderData", !781}
!781 = !{!"numReplicas", i32 0}
!782 = !{!"URBInfo", !783, !784, !785}
!783 = !{!"has64BVertexHeaderInput", i1 false}
!784 = !{!"has64BVertexHeaderOutput", i1 false}
!785 = !{!"hasVertexHeader", i1 true}
!786 = !{!"m_ForcePullModel", i1 false}
!787 = !{!"UseBindlessImage", i1 true}
!788 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!789 = !{!"enableRangeReduce", i1 false}
!790 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!791 = !{!"enableFRemToSRemOpt", i1 false}
!792 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!793 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!794 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!795 = !{!"allowMatchMadOptimizationforVS", i1 false}
!796 = !{!"disableMatchMadOptimizationForCS", i1 false}
!797 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!798 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!799 = !{!"statefulResourcesNotAliased", i1 false}
!800 = !{!"disableMixMode", i1 false}
!801 = !{!"genericAccessesResolved", i1 false}
!802 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!803 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!804 = !{!"disableSeparateScratchWA", i1 false}
!805 = !{!"enableRemoveUnusedTGMFence", i1 false}
!806 = !{!"PrivateMemoryPerFG"}
!807 = !{!"m_OptsToDisable"}
!808 = !{!"capabilities", !809}
!809 = !{!"globalVariableDecorationsINTEL", i1 false}
!810 = !{!"extensions", !811}
!811 = !{!"spvINTELBindlessImages", i1 false}
!812 = !{!"m_ShaderResourceViewMcsMask", !813, !814}
!813 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!814 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!815 = !{!"computedDepthMode", i32 0}
!816 = !{!"isHDCFastClearShader", i1 false}
!817 = !{!"argRegisterReservations", !818}
!818 = !{!"argRegisterReservationsVec[0]", i32 0}
!819 = !{!"SIMD16_SpillThreshold", i8 0}
!820 = !{!"SIMD32_SpillThreshold", i8 0}
!821 = !{!"m_CacheControlOption", !822, !823, !824, !825}
!822 = !{!"LscLoadCacheControlOverride", i8 0}
!823 = !{!"LscStoreCacheControlOverride", i8 0}
!824 = !{!"TgmLoadCacheControlOverride", i8 0}
!825 = !{!"TgmStoreCacheControlOverride", i8 0}
!826 = !{!"ModuleUsesBindless", i1 false}
!827 = !{!"predicationMap"}
!828 = !{!"lifeTimeStartMap"}
!829 = !{!"HitGroups"}
!830 = !{i32 2, i32 0}
!831 = !{!"clang version 16.0.6"}
!832 = !{i32 1, !"wchar_size", i32 4}
!833 = !{!834, !835}
!834 = !{i32 4469}
!835 = !{i32 4470}
!836 = !{!835}
!837 = !{!834}
!838 = !{!839}
!839 = distinct !{!839, !840}
!840 = distinct !{!840}
!841 = !{!842, !839}
!842 = distinct !{!842, !843}
!843 = distinct !{!843}
!844 = !{!845}
!845 = !{i32 40, i32 196620}
!846 = !{!847}
!847 = !{i32 44, i32 4}
!848 = distinct !{!848, !849}
!849 = !{!"llvm.loop.unroll.enable"}
!850 = distinct !{!850, !849}
!851 = distinct !{!851, !849}
!852 = !{!853}
!853 = distinct !{!853, !854}
!854 = distinct !{!854}
!855 = !{!856}
!856 = distinct !{!856, !857}
!857 = distinct !{!857}
!858 = distinct !{!858, !849}
!859 = distinct !{!859, !849}
!860 = !{!861}
!861 = distinct !{!861, !862}
!862 = distinct !{!862}
!863 = distinct !{!863, !849}
!864 = distinct !{!864, !849}
!865 = distinct !{!865, !849}
!866 = distinct !{!866, !849}
!867 = !{!868}
!868 = distinct !{!868, !869}
!869 = distinct !{!869}
!870 = !{!871}
!871 = distinct !{!871, !872}
!872 = distinct !{!872}
!873 = distinct !{!873, !849}
!874 = distinct !{!874, !849}
!875 = !{!876}
!876 = distinct !{!876, !877}
!877 = distinct !{!877}
!878 = distinct !{!878, !849}
!879 = distinct !{!879, !849}
!880 = distinct !{!880, !849}
!881 = distinct !{!881, !849}
!882 = !{!883}
!883 = distinct !{!883, !884}
!884 = distinct !{!884}
!885 = !{!886}
!886 = distinct !{!886, !887}
!887 = distinct !{!887}
!888 = distinct !{!888, !849}
!889 = distinct !{!889, !849}
!890 = !{!891}
!891 = distinct !{!891, !892}
!892 = distinct !{!892}
!893 = distinct !{!893, !849}
!894 = distinct !{!894, !849}
!895 = distinct !{!895, !849}
!896 = distinct !{!896, !849}
!897 = !{!898}
!898 = distinct !{!898, !899}
!899 = distinct !{!899}
!900 = !{!901}
!901 = distinct !{!901, !902}
!902 = distinct !{!902}
!903 = distinct !{!903, !849}
!904 = distinct !{!904, !849}
!905 = !{!906}
!906 = distinct !{!906, !907}
!907 = distinct !{!907}
!908 = distinct !{!908, !849}

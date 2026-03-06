; ------------------------------------------------
; OCL_asm23954d4a795eca46_optimized.ll
; LLVM major version: 16
; ------------------------------------------------
; ModuleID = '<origin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [3 x i64] }
%class.__generated_ = type { i8 addrspace(1)*, i64, %"class.sycl::_V1::range", i8 addrspace(1)*, i64, %"class.sycl::_V1::range" }
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
%"struct.cutlass::bfloat16_t" = type { i16 }

@gVar = internal global [36 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@gVar.61 = internal global [24 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@llvm.used = appending global [2 x i8*] [i8* getelementptr inbounds ([36 x i8], [36 x i8]* @gVar, i32 0, i32 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @gVar.61, i32 0, i32 0)], section "llvm.metadata"

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE(%"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %0, %class.__generated_* byval(%class.__generated_) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i64 %const_reg_qword12, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i64 0
  %4 = extractelement <3 x i32> %globalSize, i64 1
  %5 = extractelement <3 x i32> %globalSize, i64 2
  %6 = extractelement <3 x i32> %globalOffset, i64 0
  %7 = extractelement <3 x i32> %globalOffset, i64 1
  %8 = extractelement <3 x i32> %globalOffset, i64 2
  %9 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i64 2
  %12 = extractelement <8 x i32> %r0, i64 1
  %13 = extractelement <8 x i32> %r0, i64 6
  %14 = extractelement <8 x i32> %r0, i64 7
  %15 = inttoptr i64 %const_reg_qword3 to i8 addrspace(4)*
  %16 = inttoptr i64 %const_reg_qword8 to i8 addrspace(4)*
  %17 = zext i32 %14 to i64
  %18 = zext i32 %11 to i64
  %19 = mul nuw i64 %18, %17
  %20 = zext i16 %localIdZ to i64
  %21 = add nuw i64 %19, %20
  %22 = zext i32 %8 to i64
  %23 = add nuw i64 %21, %22
  %24 = zext i32 %13 to i64
  %25 = zext i32 %10 to i64
  %26 = mul nuw i64 %25, %24
  %27 = zext i16 %localIdY to i64
  %28 = add nuw i64 %26, %27
  %29 = zext i32 %7 to i64
  %30 = add nuw i64 %28, %29
  %31 = zext i32 %12 to i64
  %32 = zext i32 %9 to i64
  %33 = mul nuw i64 %32, %31
  %34 = zext i16 %localIdX to i64
  %35 = add nuw i64 %33, %34
  %36 = zext i32 %6 to i64
  %37 = add nuw i64 %35, %36
  %38 = zext i32 %5 to i64
  %39 = zext i32 %4 to i64
  %40 = zext i32 %3 to i64
  %41 = icmp ult i64 %23, %const_reg_qword
  %42 = icmp ult i64 %30, %const_reg_qword1
  %43 = icmp ult i64 %37, %const_reg_qword2
  %44 = and i1 %43, %42
  %45 = and i1 %44, %41
  br i1 %45, label %.lr.ph, label %._crit_edge101

.lr.ph:                                           ; preds = %._crit_edge99, %70, %2
  %46 = phi i64 [ %46, %._crit_edge99 ], [ %37, %2 ], [ %spec.select, %70 ]
  %47 = phi i64 [ %73, %._crit_edge99 ], [ %30, %2 ], [ %30, %70 ]
  %48 = phi i64 [ %74, %._crit_edge99 ], [ %23, %2 ], [ %23, %70 ]
  %49 = icmp ult i64 %48, 2147483648
  call void @llvm.assume(i1 %49)
  %50 = icmp ult i64 %47, 2147483648
  call void @llvm.assume(i1 %50)
  %51 = icmp ult i64 %46, 2147483648
  call void @llvm.assume(i1 %51)
  %52 = mul i64 %46, %const_reg_qword9
  %53 = mul i64 %47, %const_reg_qword10
  %54 = getelementptr i8, i8 addrspace(4)* %16, i64 %52
  %55 = getelementptr i8, i8 addrspace(4)* %54, i64 %53
  %56 = getelementptr i8, i8 addrspace(4)* %55, i64 %48
  %57 = addrspacecast i8 addrspace(4)* %56 to i8 addrspace(1)*
  %58 = load i8, i8 addrspace(1)* %57, align 1
  %59 = mul i64 %46, %const_reg_qword4
  %60 = mul i64 %47, %const_reg_qword5
  %61 = getelementptr i8, i8 addrspace(4)* %15, i64 %59
  %62 = getelementptr i8, i8 addrspace(4)* %61, i64 %60
  %63 = getelementptr i8, i8 addrspace(4)* %62, i64 %48
  %64 = addrspacecast i8 addrspace(4)* %63 to i8 addrspace(1)*
  store i8 %58, i8 addrspace(1)* %64, align 1
  %65 = add nuw nsw i64 %48, %38
  %66 = icmp ult i64 %65, %const_reg_qword
  br i1 %66, label %._crit_edge99, label %67

67:                                               ; preds = %.lr.ph
  %68 = add nuw nsw i64 %47, %39
  %69 = icmp ult i64 %68, %const_reg_qword1
  br i1 %69, label %._crit_edge99, label %70

70:                                               ; preds = %67
  %71 = add nuw nsw i64 %46, %40
  %72 = icmp ult i64 %71, %const_reg_qword2
  %spec.select = select i1 %72, i64 %71, i64 %37
  br i1 %72, label %.lr.ph, label %._crit_edge101

._crit_edge99:                                    ; preds = %.lr.ph, %67
  %73 = phi i64 [ %47, %.lr.ph ], [ %68, %67 ]
  %74 = phi i64 [ %65, %.lr.ph ], [ %23, %67 ]
  br label %.lr.ph

._crit_edge101:                                   ; preds = %70, %2
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
  %7 = extractelement <3 x i32> %globalOffset, i64 0
  %8 = extractelement <3 x i32> %globalOffset, i64 1
  %9 = extractelement <3 x i32> %globalOffset, i64 2
  %10 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %11 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %12 = extractelement <3 x i32> %enqueuedLocalSize, i64 2
  %13 = extractelement <8 x i32> %r0, i64 1
  %14 = extractelement <8 x i32> %r0, i64 6
  %15 = extractelement <8 x i32> %r0, i64 7
  %16 = mul i32 %12, %15
  %17 = zext i16 %localIdZ to i32
  %18 = add i32 %16, %17
  %19 = add i32 %18, %9
  %20 = zext i32 %19 to i64
  %21 = mul i32 %11, %14
  %22 = zext i16 %localIdY to i32
  %23 = add i32 %21, %22
  %24 = add i32 %23, %8
  %25 = zext i32 %24 to i64
  %26 = mul i32 %10, %13
  %27 = zext i16 %localIdX to i32
  %28 = add i32 %26, %27
  %29 = add i32 %28, %7
  %30 = zext i32 %29 to i64
  %31 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %31)
  %32 = icmp sgt i32 %24, -1
  call void @llvm.assume(i1 %32)
  %33 = icmp sgt i32 %29, -1
  call void @llvm.assume(i1 %33)
  %34 = mul i64 %30, %4
  %35 = mul i64 %25, %const_reg_qword3
  %36 = getelementptr i8, i8 addrspace(1)* %3, i64 %34
  %37 = getelementptr i8, i8 addrspace(1)* %36, i64 %35
  %38 = getelementptr i8, i8 addrspace(1)* %37, i64 %20
  %39 = load i8, i8 addrspace(1)* %38, align 1
  %40 = mul i64 %30, %1
  %41 = mul i64 %25, %const_reg_qword
  %42 = getelementptr i8, i8 addrspace(1)* %0, i64 %40
  %43 = getelementptr i8, i8 addrspace(1)* %42, i64 %41
  %44 = getelementptr i8, i8 addrspace(1)* %43, i64 %20
  store i8 %39, i8 addrspace(1)* %44, align 1
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.2* byval(%class.__generated_.2) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i16 %const_reg_word, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i64 0
  %4 = extractelement <3 x i32> %globalOffset, i64 0
  %5 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %6 = extractelement <8 x i32> %r0, i64 1
  %7 = inttoptr i64 %const_reg_qword1 to i16 addrspace(4)*
  %8 = zext i32 %6 to i64
  %9 = zext i32 %5 to i64
  %10 = mul nuw i64 %9, %8
  %11 = zext i16 %localIdX to i64
  %12 = add nuw i64 %10, %11
  %13 = zext i32 %4 to i64
  %14 = add nuw i64 %12, %13
  %15 = zext i32 %3 to i64
  %16 = icmp ult i64 %14, %const_reg_qword
  br i1 %16, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %2, %.lr.ph
  %17 = phi i64 [ %22, %.lr.ph ], [ %14, %2 ]
  %18 = getelementptr inbounds i16, i16 addrspace(4)* %7, i64 %17
  %19 = addrspacecast i16 addrspace(4)* %18 to i16 addrspace(1)*
  store i16 %const_reg_word, i16 addrspace(1)* %19, align 2
  %20 = add nuw nsw i64 %17, %15
  %21 = icmp ult i64 %20, %const_reg_qword
  %22 = select i1 %21, i64 %20, i64 %14
  br i1 %21, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %2
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_(i16 addrspace(1)* align 2 %0, i16 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i64 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %5 = extractelement <8 x i32> %r0, i64 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = icmp sgt i32 %9, -1
  call void @llvm.assume(i1 %11)
  %12 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 %10
  store i16 %1, i16 addrspace(1)* %12, align 2
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.9* byval(%class.__generated_.9) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i32 %const_reg_dword, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i64 0
  %4 = extractelement <3 x i32> %globalOffset, i64 0
  %5 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %6 = extractelement <8 x i32> %r0, i64 1
  %7 = inttoptr i64 %const_reg_qword1 to i32 addrspace(4)*
  %8 = zext i32 %6 to i64
  %9 = zext i32 %5 to i64
  %10 = mul nuw i64 %9, %8
  %11 = zext i16 %localIdX to i64
  %12 = add nuw i64 %10, %11
  %13 = zext i32 %4 to i64
  %14 = add nuw i64 %12, %13
  %15 = zext i32 %3 to i64
  %16 = icmp ult i64 %14, %const_reg_qword
  br i1 %16, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %2, %.lr.ph
  %17 = phi i64 [ %22, %.lr.ph ], [ %14, %2 ]
  %18 = getelementptr inbounds i32, i32 addrspace(4)* %7, i64 %17
  %19 = addrspacecast i32 addrspace(4)* %18 to i32 addrspace(1)*
  store i32 %const_reg_dword, i32 addrspace(1)* %19, align 4
  %20 = add nuw nsw i64 %17, %15
  %21 = icmp ult i64 %20, %const_reg_qword
  %22 = select i1 %21, i64 %20, i64 %14
  br i1 %21, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %2
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_(i32 addrspace(1)* align 4 %0, i32 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i64 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %5 = extractelement <8 x i32> %r0, i64 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = icmp sgt i32 %9, -1
  call void @llvm.assume(i1 %11)
  %12 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %10
  store i32 %1, i32 addrspace(1)* %12, align 4
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.12* byval(%class.__generated_.12) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i8 %const_reg_byte7, i8 %const_reg_byte8, i32 %bindlessOffset) #0 {
  %3 = extractelement <3 x i32> %globalSize, i64 0
  %4 = extractelement <3 x i32> %globalOffset, i64 0
  %5 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %6 = extractelement <8 x i32> %r0, i64 1
  %7 = inttoptr i64 %const_reg_qword1 to i8 addrspace(4)*
  %8 = zext i32 %6 to i64
  %9 = zext i32 %5 to i64
  %10 = mul nuw i64 %9, %8
  %11 = zext i16 %localIdX to i64
  %12 = add nuw i64 %10, %11
  %13 = zext i32 %4 to i64
  %14 = add nuw i64 %12, %13
  %15 = zext i32 %3 to i64
  %16 = icmp ult i64 %14, %const_reg_qword
  br i1 %16, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %2, %.lr.ph
  %17 = phi i64 [ %22, %.lr.ph ], [ %14, %2 ]
  %18 = getelementptr inbounds i8, i8 addrspace(4)* %7, i64 %17
  %19 = addrspacecast i8 addrspace(4)* %18 to i8 addrspace(1)*
  store i8 %const_reg_byte, i8 addrspace(1)* %19, align 1
  %20 = add nuw nsw i64 %17, %15
  %21 = icmp ult i64 %20, %const_reg_qword
  %22 = select i1 %21, i64 %20, i64 %14
  br i1 %21, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %2
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_(i8 addrspace(1)* align 1 %0, i8 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i64 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %5 = extractelement <8 x i32> %r0, i64 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = icmp sgt i32 %9, -1
  call void @llvm.assume(i1 %11)
  %12 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %10
  store i8 %1, i8 addrspace(1)* %12, align 1
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE(i16 addrspace(1)* align 2 %0, i64 %1, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"* byval(%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params") align 8 %2, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, float %const_reg_fp32, float %const_reg_fp321, i32 %const_reg_dword, float %const_reg_fp322, float %const_reg_fp323, i8 %const_reg_byte, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset7) #0 {
._crit_edge:
  %3 = extractelement <3 x i32> %localSize, i64 0
  %4 = extractelement <8 x i32> %r0, i64 1
  %5 = alloca [3 x i64], align 8, !spirv.Decorations !0
  %6 = alloca %"struct.cutlass::reference::device::detail::RandomUniformFunc", align 8, !spirv.Decorations !0
  %7 = bitcast %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 88, i8* nonnull %7)
  %.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 0
  store i64 %const_reg_qword, i64* %.sroa.0.0..sroa_idx, align 8
  %.sroa.5.0..sroa_idx2 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 1
  store float %const_reg_fp32, float* %.sroa.5.0..sroa_idx2, align 8
  %.sroa.7.0..sroa_idx4 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 2
  store float %const_reg_fp321, float* %.sroa.7.0..sroa_idx4, align 4
  %.sroa.9.0..sroa_idx6 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 3
  store i32 %const_reg_dword, i32* %.sroa.9.0..sroa_idx6, align 8
  %.sroa.10.0..sroa_idx8 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 4
  store float %const_reg_fp322, float* %.sroa.10.0..sroa_idx8, align 4
  %.sroa.11.0..sroa_idx10 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 0, i32 5
  store float %const_reg_fp323, float* %.sroa.11.0..sroa_idx10, align 8
  %8 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 1, i32 0, i32 0
  store float %const_reg_fp321, float* %8, align 8
  %9 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 1, i32 0, i32 1
  store float %const_reg_fp32, float* %9, align 4
  %10 = icmp sgt i32 %4, -1
  call void @llvm.assume(i1 %10)
  %11 = icmp sgt i32 %3, -1
  call void @llvm.assume(i1 %11)
  %12 = mul i32 %4, %3
  %13 = zext i16 %localIdX to i32
  %14 = add i32 %12, %13
  %15 = zext i32 %14 to i64
  %16 = trunc i64 %const_reg_qword to i32
  %17 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 0, i64 0
  store i32 %16, i32* %17, align 8
  %18 = lshr i64 %const_reg_qword, 32
  %19 = trunc i64 %18 to i32
  %20 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  store i32 %19, i32* %20, align 4
  %21 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 1
  %22 = bitcast [4 x i32]* %21 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(36) %22, i8* noundef nonnull align 8 dereferenceable(36) getelementptr inbounds ([36 x i8], [36 x i8]* @gVar, i64 0, i64 0), i64 36, i1 false)
  %23 = bitcast [3 x i64]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %23)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %23, i8* noundef nonnull align 8 dereferenceable(24) getelementptr inbounds ([24 x i8], [24 x i8]* @gVar.61, i64 0, i64 0), i64 24, i1 false)
  %24 = getelementptr inbounds [3 x i64], [3 x i64]* %5, i64 0, i64 0
  %25 = getelementptr inbounds [3 x i64], [3 x i64]* %5, i64 0, i64 1
  store i64 %15, i64* %25, align 8
  %26 = icmp eq i32 %14, 0
  %27 = select i1 %26, i32 0, i32 2
  %28 = extractelement <3 x i32> %numWorkGroups, i64 0
  %29 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 1, i64 1
  %30 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 1, i64 2
  %31 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 1, i64 3
  %32 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 2
  %33 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 1
  %34 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 2
  %35 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 3
  br i1 %26, label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit, label %LeafBlock35._crit_edge

LeafBlock35._crit_edge:                           ; preds = %._crit_edge
  store i64 0, i64* %24, align 8
  br i1 false, label %41, label %.lr.ph51

.lr.ph51:                                         ; preds = %LeafBlock35._crit_edge
  %36 = add nsw i32 %27, -1, !spirv.Decorations !836
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds [3 x i64], [3 x i64]* %5, i64 0, i64 %37
  %39 = load i64, i64* %38, align 8
  %40 = lshr i64 %39, 2
  store i64 %40, i64* %38, align 8
  br i1 false, label %._crit_edge52, label %192

._crit_edge52:                                    ; preds = %192, %.lr.ph51
  %.pre = load i64, i64* %24, align 8
  %.pre64 = load i64, i64* %25, align 8
  br label %41

41:                                               ; preds = %._crit_edge52, %LeafBlock35._crit_edge
  %42 = phi i64 [ %.pre64, %._crit_edge52 ], [ %15, %LeafBlock35._crit_edge ]
  %43 = phi i64 [ %.pre, %._crit_edge52 ], [ 0, %LeafBlock35._crit_edge ]
  store i32 4, i32* %32, align 8
  %44 = and i64 %43, 4294967295
  %45 = mul nuw i64 %44, 3528531795, !spirv.Decorations !838
  %46 = and i64 %42, 4294967295
  %47 = mul nuw i64 %46, 3449720151, !spirv.Decorations !838
  %48 = xor i64 %43, %47
  %49 = lshr i64 %48, 32
  %50 = xor i64 %42, %45
  %51 = xor i64 %50, %const_reg_qword
  %52 = lshr i64 %51, 32
  %53 = add i32 %16, -1640531527
  %54 = add i32 %19, -1150833019
  %.masked = and i64 %const_reg_qword, 4294967295
  %55 = xor i64 %49, %.masked
  %56 = mul nuw i64 %55, 3528531795, !spirv.Decorations !838
  %57 = lshr i64 %56, 32
  %58 = mul nuw i64 %52, 3449720151, !spirv.Decorations !838
  %59 = lshr i64 %58, 32
  %60 = xor i64 %47, %59
  %61 = trunc i64 %60 to i32
  %62 = xor i32 %53, %61
  %63 = xor i64 %45, %57
  %64 = trunc i64 %63 to i32
  %65 = xor i32 %54, %64
  %66 = add i32 %16, 1013904242
  %67 = add i32 %19, 1993301258
  %68 = zext i32 %62 to i64
  %69 = mul nuw i64 %68, 3528531795, !spirv.Decorations !838
  %70 = lshr i64 %69, 32
  %71 = zext i32 %65 to i64
  %72 = mul nuw i64 %71, 3449720151, !spirv.Decorations !838
  %73 = lshr i64 %72, 32
  %74 = xor i64 %58, %73
  %75 = trunc i64 %74 to i32
  %76 = xor i32 %66, %75
  %77 = xor i64 %56, %70
  %78 = trunc i64 %77 to i32
  %79 = xor i32 %67, %78
  %80 = add i32 %16, -626627285
  %81 = add i32 %19, 842468239
  %82 = zext i32 %76 to i64
  %83 = mul nuw i64 %82, 3528531795, !spirv.Decorations !838
  %84 = lshr i64 %83, 32
  %85 = zext i32 %79 to i64
  %86 = mul nuw i64 %85, 3449720151, !spirv.Decorations !838
  %87 = lshr i64 %86, 32
  %88 = xor i64 %72, %87
  %89 = trunc i64 %88 to i32
  %90 = xor i32 %80, %89
  %91 = xor i64 %69, %84
  %92 = trunc i64 %91 to i32
  %93 = xor i32 %81, %92
  %94 = add i32 %16, 2027808484
  %95 = add i32 %19, -308364780
  %96 = zext i32 %90 to i64
  %97 = mul nuw i64 %96, 3528531795, !spirv.Decorations !838
  %98 = lshr i64 %97, 32
  %99 = zext i32 %93 to i64
  %100 = mul nuw i64 %99, 3449720151, !spirv.Decorations !838
  %101 = lshr i64 %100, 32
  %102 = xor i64 %86, %101
  %103 = trunc i64 %102 to i32
  %104 = xor i32 %94, %103
  %105 = xor i64 %83, %98
  %106 = trunc i64 %105 to i32
  %107 = xor i32 %95, %106
  %108 = add i32 %16, 387276957
  %109 = add i32 %19, -1459197799
  %110 = zext i32 %104 to i64
  %111 = mul nuw i64 %110, 3528531795, !spirv.Decorations !838
  %112 = lshr i64 %111, 32
  %113 = zext i32 %107 to i64
  %114 = mul nuw i64 %113, 3449720151, !spirv.Decorations !838
  %115 = lshr i64 %114, 32
  %116 = xor i64 %100, %115
  %117 = trunc i64 %116 to i32
  %118 = xor i32 %108, %117
  %119 = xor i64 %97, %112
  %120 = trunc i64 %119 to i32
  %121 = xor i32 %109, %120
  %122 = add i32 %16, -1253254570
  %123 = add i32 %19, 1684936478
  %124 = zext i32 %118 to i64
  %125 = mul nuw i64 %124, 3528531795, !spirv.Decorations !838
  %126 = lshr i64 %125, 32
  %127 = zext i32 %121 to i64
  %128 = mul nuw i64 %127, 3449720151, !spirv.Decorations !838
  %129 = lshr i64 %128, 32
  %130 = xor i64 %114, %129
  %131 = trunc i64 %130 to i32
  %132 = xor i32 %122, %131
  %133 = xor i64 %111, %126
  %134 = trunc i64 %133 to i32
  %135 = xor i32 %123, %134
  %136 = add i32 %16, 1401181199
  %137 = add i32 %19, 534103459
  %138 = zext i32 %132 to i64
  %139 = mul nuw i64 %138, 3528531795, !spirv.Decorations !838
  %140 = lshr i64 %139, 32
  %141 = zext i32 %135 to i64
  %142 = mul nuw i64 %141, 3449720151, !spirv.Decorations !838
  %143 = lshr i64 %142, 32
  %144 = xor i64 %128, %143
  %145 = trunc i64 %144 to i32
  %146 = xor i32 %136, %145
  %147 = xor i64 %125, %140
  %148 = trunc i64 %147 to i32
  %149 = xor i32 %137, %148
  %150 = add i32 %16, -239350328
  %151 = add i32 %19, -616729560
  %152 = zext i32 %146 to i64
  %153 = mul nuw i64 %152, 3528531795, !spirv.Decorations !838
  %154 = lshr i64 %153, 32
  %155 = zext i32 %149 to i64
  %156 = mul nuw i64 %155, 3449720151, !spirv.Decorations !838
  %157 = lshr i64 %156, 32
  %158 = xor i64 %142, %157
  %159 = trunc i64 %158 to i32
  %160 = xor i32 %150, %159
  %161 = xor i64 %139, %154
  %162 = trunc i64 %161 to i32
  %163 = xor i32 %151, %162
  %164 = add i32 %16, -1879881855
  %165 = add i32 %19, -1767562579
  %166 = zext i32 %160 to i64
  %167 = mul nuw i64 %166, 3528531795, !spirv.Decorations !838
  %168 = trunc i64 %167 to i32
  %169 = lshr i64 %167, 32
  %170 = zext i32 %163 to i64
  %171 = mul nuw i64 %170, 3449720151, !spirv.Decorations !838
  %172 = trunc i64 %171 to i32
  %173 = lshr i64 %171, 32
  %174 = xor i64 %156, %173
  %175 = trunc i64 %174 to i32
  %176 = xor i32 %164, %175
  %177 = xor i64 %153, %169
  %178 = trunc i64 %177 to i32
  %179 = xor i32 %165, %178
  %180 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 0
  store i32 %176, i32* %180, align 4
  store i32 %172, i32* %33, align 8
  store i32 %179, i32* %34, align 4
  store i32 %168, i32* %35, align 8
  %181 = add i64 %43, 1
  %182 = icmp eq i64 %181, 0
  %183 = zext i1 %182 to i64
  %184 = add i64 %42, %183
  %185 = trunc i64 %181 to i32
  %186 = getelementptr inbounds [4 x i32], [4 x i32]* %21, i64 0, i64 0
  store i32 %185, i32* %186, align 8
  %187 = lshr i64 %181, 32
  %188 = trunc i64 %187 to i32
  store i32 %188, i32* %29, align 4
  %189 = trunc i64 %184 to i32
  store i32 %189, i32* %30, align 8
  %190 = lshr i64 %184, 32
  %191 = trunc i64 %190 to i32
  store i32 %191, i32* %31, align 4
  br label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit

192:                                              ; preds = %.lr.ph51
  %193 = add nsw i32 %27, -2, !spirv.Decorations !836
  %194 = zext i32 %193 to i64
  %195 = getelementptr inbounds [3 x i64], [3 x i64]* %5, i64 0, i64 %194
  %196 = load i64, i64* %195, align 8
  %197 = call i64 @llvm.fshl.i64(i64 %39, i64 %196, i64 62)
  store i64 %197, i64* %195, align 8
  br label %._crit_edge52

_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit: ; preds = %._crit_edge, %41
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %23)
  %198 = icmp ult i64 %15, %1
  br i1 %198, label %.lr.ph48, label %._crit_edge49

.lr.ph48:                                         ; preds = %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit
  %199 = load float, float* %8, align 8, !noalias !840
  %200 = load float, float* %9, align 4, !noalias !840
  %201 = getelementptr inbounds [4 x i32], [4 x i32]* %21, i64 0, i64 0
  %202 = load i32, i32* %17, align 8
  %203 = load i32, i32* %20, align 4
  %204 = add i32 %202, -1640531527
  %205 = add i32 %203, -1150833019
  %206 = add i32 %202, 1013904242
  %207 = add i32 %203, 1993301258
  %208 = add i32 %202, -626627285
  %209 = add i32 %203, 842468239
  %210 = add i32 %202, 2027808484
  %211 = add i32 %203, -308364780
  %212 = add i32 %202, 387276957
  %213 = add i32 %203, -1459197799
  %214 = add i32 %202, -1253254570
  %215 = add i32 %203, 1684936478
  %216 = add i32 %202, 1401181199
  %217 = add i32 %203, 534103459
  %218 = add i32 %203, -616729560
  %219 = add i32 %202, -1879881855
  %220 = add i32 %202, -239350328
  %221 = add i32 %203, -1767562579
  %222 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 0
  %223 = fadd reassoc nsz arcp contract float %200, %199, !spirv.Decorations !843
  %224 = fmul reassoc nsz arcp contract float %223, 5.000000e-01
  %225 = fsub reassoc nsz arcp contract float %200, %199, !spirv.Decorations !843
  %226 = fmul reassoc nsz arcp contract float %225, 0x3DF0000000000000
  %227 = load i32, i32* %.sroa.9.0..sroa_idx6, align 8, !noalias !840
  %228 = icmp sgt i32 %227, -1
  %229 = load float, float* %.sroa.10.0..sroa_idx8, align 4
  %230 = load float, float* %.sroa.11.0..sroa_idx10, align 8
  %231 = icmp sgt i32 %28, -1
  call void @llvm.assume(i1 %231)
  %.narrow = mul i32 %3, %28
  %232 = zext i32 %.narrow to i64
  %.promoted = load i32, i32* %32, align 8, !noalias !840
  %.promoted57 = load i32, i32* %201, align 8, !noalias !840
  %.promoted58 = load i32, i32* %29, align 4, !noalias !840
  %.promoted59 = load i32, i32* %30, align 8, !noalias !840
  %.promoted60 = load i32, i32* %31, align 4, !noalias !840
  br label %233

233:                                              ; preds = %.lr.ph48, %407
  %234 = phi i32 [ %.promoted60, %.lr.ph48 ], [ %383, %407 ]
  %235 = phi i32 [ %.promoted59, %.lr.ph48 ], [ %384, %407 ]
  %236 = phi i32 [ %.promoted58, %.lr.ph48 ], [ %385, %407 ]
  %237 = phi i32 [ %.promoted57, %.lr.ph48 ], [ %386, %407 ]
  %238 = phi i32 [ %.promoted, %.lr.ph48 ], [ %387, %407 ]
  %239 = phi i64 [ %15, %.lr.ph48 ], [ %410, %407 ]
  %.not.not = icmp eq i32 %238, 0
  br i1 %.not.not, label %246, label %240

240:                                              ; preds = %233
  %241 = sub nsw i32 4, %238, !spirv.Decorations !836
  %242 = sext i32 %241 to i64
  %243 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc"* %6, i64 0, i32 2, i32 0, i32 0, i32 3, i64 %242
  %244 = load i32, i32* %243, align 4, !noalias !845
  %245 = add i32 %238, -1
  store i32 %245, i32* %32, align 8, !noalias !840
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

246:                                              ; preds = %233
  %247 = zext i32 %237 to i64
  %248 = mul nuw i64 %247, 3528531795, !spirv.Decorations !838
  %249 = lshr i64 %248, 32
  %250 = trunc i64 %249 to i32
  %251 = zext i32 %235 to i64
  %252 = mul nuw i64 %251, 3449720151, !spirv.Decorations !838
  %253 = lshr i64 %252, 32
  %254 = trunc i64 %253 to i32
  %255 = xor i32 %236, %254
  %256 = xor i32 %255, %202
  %257 = xor i32 %234, %250
  %258 = xor i32 %257, %203
  %259 = zext i32 %256 to i64
  %260 = mul nuw i64 %259, 3528531795, !spirv.Decorations !838
  %261 = lshr i64 %260, 32
  %262 = zext i32 %258 to i64
  %263 = mul nuw i64 %262, 3449720151, !spirv.Decorations !838
  %264 = lshr i64 %263, 32
  %265 = xor i64 %252, %264
  %266 = trunc i64 %265 to i32
  %267 = xor i32 %204, %266
  %268 = xor i64 %248, %261
  %269 = trunc i64 %268 to i32
  %270 = xor i32 %205, %269
  %271 = zext i32 %267 to i64
  %272 = mul nuw i64 %271, 3528531795, !spirv.Decorations !838
  %273 = lshr i64 %272, 32
  %274 = zext i32 %270 to i64
  %275 = mul nuw i64 %274, 3449720151, !spirv.Decorations !838
  %276 = lshr i64 %275, 32
  %277 = xor i64 %263, %276
  %278 = trunc i64 %277 to i32
  %279 = xor i32 %206, %278
  %280 = xor i64 %260, %273
  %281 = trunc i64 %280 to i32
  %282 = xor i32 %207, %281
  %283 = zext i32 %279 to i64
  %284 = mul nuw i64 %283, 3528531795, !spirv.Decorations !838
  %285 = lshr i64 %284, 32
  %286 = zext i32 %282 to i64
  %287 = mul nuw i64 %286, 3449720151, !spirv.Decorations !838
  %288 = lshr i64 %287, 32
  %289 = xor i64 %275, %288
  %290 = trunc i64 %289 to i32
  %291 = xor i32 %208, %290
  %292 = xor i64 %272, %285
  %293 = trunc i64 %292 to i32
  %294 = xor i32 %209, %293
  %295 = zext i32 %291 to i64
  %296 = mul nuw i64 %295, 3528531795, !spirv.Decorations !838
  %297 = lshr i64 %296, 32
  %298 = zext i32 %294 to i64
  %299 = mul nuw i64 %298, 3449720151, !spirv.Decorations !838
  %300 = lshr i64 %299, 32
  %301 = xor i64 %287, %300
  %302 = trunc i64 %301 to i32
  %303 = xor i32 %210, %302
  %304 = xor i64 %284, %297
  %305 = trunc i64 %304 to i32
  %306 = xor i32 %211, %305
  %307 = zext i32 %303 to i64
  %308 = mul nuw i64 %307, 3528531795, !spirv.Decorations !838
  %309 = lshr i64 %308, 32
  %310 = zext i32 %306 to i64
  %311 = mul nuw i64 %310, 3449720151, !spirv.Decorations !838
  %312 = lshr i64 %311, 32
  %313 = xor i64 %299, %312
  %314 = trunc i64 %313 to i32
  %315 = xor i32 %212, %314
  %316 = xor i64 %296, %309
  %317 = trunc i64 %316 to i32
  %318 = xor i32 %213, %317
  %319 = zext i32 %315 to i64
  %320 = mul nuw i64 %319, 3528531795, !spirv.Decorations !838
  %321 = lshr i64 %320, 32
  %322 = zext i32 %318 to i64
  %323 = mul nuw i64 %322, 3449720151, !spirv.Decorations !838
  %324 = lshr i64 %323, 32
  %325 = xor i64 %311, %324
  %326 = trunc i64 %325 to i32
  %327 = xor i32 %214, %326
  %328 = xor i64 %308, %321
  %329 = trunc i64 %328 to i32
  %330 = xor i32 %215, %329
  %331 = zext i32 %327 to i64
  %332 = mul nuw i64 %331, 3528531795, !spirv.Decorations !838
  %333 = lshr i64 %332, 32
  %334 = zext i32 %330 to i64
  %335 = mul nuw i64 %334, 3449720151, !spirv.Decorations !838
  %336 = lshr i64 %335, 32
  %337 = xor i64 %323, %336
  %338 = trunc i64 %337 to i32
  %339 = xor i32 %216, %338
  %340 = xor i64 %320, %333
  %341 = trunc i64 %340 to i32
  %342 = xor i32 %217, %341
  %343 = zext i32 %339 to i64
  %344 = mul nuw i64 %343, 3528531795, !spirv.Decorations !838
  %345 = lshr i64 %344, 32
  %346 = mul i32 %342, -845247145
  %347 = xor i64 %332, %345
  %348 = trunc i64 %347 to i32
  %349 = xor i32 %218, %348
  %350 = zext i32 %349 to i64
  %351 = mul nuw i64 %350, 3449720151, !spirv.Decorations !838
  %352 = lshr i64 %351, 32
  %353 = trunc i64 %352 to i32
  %354 = xor i32 %346, %353
  %355 = xor i32 %354, %219
  store i32 3, i32* %32, align 8, !noalias !840
  %356 = zext i32 %342 to i64
  %357 = mul nuw i64 %356, 3449720151, !spirv.Decorations !838
  %358 = lshr i64 %357, 32
  %359 = xor i64 %335, %358
  %360 = trunc i64 %359 to i32
  %361 = xor i32 %220, %360
  %362 = zext i32 %361 to i64
  %363 = mul nuw i64 %362, 3528531795, !spirv.Decorations !838
  %364 = trunc i64 %363 to i32
  %365 = lshr i64 %363, 32
  %366 = trunc i64 %351 to i32
  %367 = xor i64 %357, %352
  %368 = trunc i64 %367 to i32
  %369 = xor i32 %219, %368
  %370 = xor i64 %344, %365
  %371 = trunc i64 %370 to i32
  %372 = xor i32 %221, %371
  store i32 %369, i32* %222, align 4, !noalias !840
  store i32 %366, i32* %33, align 8, !noalias !840
  store i32 %372, i32* %34, align 4, !noalias !840
  store i32 %364, i32* %35, align 8, !noalias !840
  %373 = add i32 %237, 1
  store i32 %373, i32* %201, align 8, !noalias !840
  %374 = icmp eq i32 %373, 0
  br i1 %374, label %375, label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

375:                                              ; preds = %246
  %376 = add i32 %236, 1
  store i32 %376, i32* %29, align 4, !noalias !840
  %377 = icmp eq i32 %376, 0
  br i1 %377, label %378, label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

378:                                              ; preds = %375
  %379 = add i32 %235, 1
  store i32 %379, i32* %30, align 8, !noalias !840
  %380 = icmp eq i32 %379, 0
  br i1 %380, label %381, label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

381:                                              ; preds = %378
  %382 = add i32 %234, 1
  store i32 %382, i32* %31, align 4, !noalias !840
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit

_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit: ; preds = %378, %375, %246, %381, %240
  %383 = phi i32 [ %234, %240 ], [ %382, %381 ], [ %234, %246 ], [ %234, %375 ], [ %234, %378 ]
  %384 = phi i32 [ %235, %240 ], [ 0, %381 ], [ %235, %246 ], [ %235, %375 ], [ %379, %378 ]
  %385 = phi i32 [ %236, %240 ], [ 0, %381 ], [ %236, %246 ], [ %376, %375 ], [ 0, %378 ]
  %386 = phi i32 [ %237, %240 ], [ 0, %381 ], [ %373, %246 ], [ 0, %375 ], [ 0, %378 ]
  %387 = phi i32 [ %245, %240 ], [ 3, %381 ], [ 3, %246 ], [ 3, %375 ], [ 3, %378 ]
  %388 = phi i32 [ %244, %240 ], [ %355, %381 ], [ %355, %246 ], [ %355, %375 ], [ %355, %378 ]
  %389 = sitofp i32 %388 to float
  %390 = fmul reassoc nsz arcp contract float %226, %389, !spirv.Decorations !843
  %391 = fadd reassoc nsz arcp contract float %390, %224, !spirv.Decorations !843
  br i1 %228, label %392, label %406

392:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %393 = fmul reassoc nsz arcp contract float %391, %229, !spirv.Decorations !843
  %394 = fcmp olt float %393, 0.000000e+00
  %395 = select i1 %394, float 0xBFDFFFFFE0000000, float 0x3FDFFFFFE0000000
  %396 = fadd float %395, %393
  %397 = call float @llvm.trunc.f32(float %396)
  %398 = fptosi float %397 to i32
  %399 = sitofp i32 %398 to float
  %400 = fmul reassoc nsz arcp contract float %230, %399, !spirv.Decorations !843
  %401 = fptosi float %400 to i32
  %402 = sitofp i32 %401 to float
  %403 = bitcast float %402 to i32
  %404 = lshr i32 %403, 16
  %405 = trunc i32 %404 to i16
  br label %407

406:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %bf_cvt = call i16 @llvm.genx.GenISA.ftobf.i16.f32(float %391, i32 0)
  br label %407

407:                                              ; preds = %406, %392
  %408 = phi i16 [ %405, %392 ], [ %bf_cvt, %406 ]
  %409 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 %239
  store i16 %408, i16 addrspace(1)* %409, align 2
  %410 = add i64 %239, %232
  %411 = icmp ult i64 %410, %1
  br i1 %411, label %233, label %._crit_edge49

._crit_edge49:                                    ; preds = %407, %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit
  call void @llvm.lifetime.end.p0i8(i64 88, i8* nonnull %7)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %21 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %22 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %23 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %24 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %24)
  %25 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %25)
  %26 = mul i32 %17, %15
  %27 = zext i16 %localIdX to i32
  %28 = add i32 %26, %27
  %29 = shl i32 %28, 2
  %30 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %18, %16
  %33 = zext i16 %localIdY to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 2
  %36 = zext i32 %19 to i64
  %37 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %37)
  %38 = mul nsw i64 %36, %9, !spirv.Decorations !836
  %39 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %23, i64 %38
  %40 = mul nsw i64 %36, %10, !spirv.Decorations !836
  %41 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %22, i64 %40
  %42 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !843
  %43 = mul nsw i64 %36, %11, !spirv.Decorations !836
  %44 = select i1 %42, i64 %43, i64 0
  %45 = getelementptr inbounds float, float addrspace(4)* %21, i64 %44
  %46 = mul nsw i64 %36, %12, !spirv.Decorations !836
  %47 = getelementptr inbounds float, float addrspace(4)* %20, i64 %46
  %48 = icmp slt i32 %19, %8
  br i1 %48, label %.lr.ph, label %._crit_edge72

.lr.ph:                                           ; preds = %13
  %49 = icmp sgt i32 %const_reg_dword2, 0
  %50 = zext i32 %14 to i64
  %51 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %51)
  %52 = mul nsw i64 %50, %9, !spirv.Decorations !836
  %53 = mul nsw i64 %50, %10, !spirv.Decorations !836
  %54 = mul nsw i64 %50, %11
  %55 = mul nsw i64 %50, %12, !spirv.Decorations !836
  %56 = icmp slt i32 %35, %const_reg_dword1
  %57 = icmp slt i32 %29, %const_reg_dword
  %58 = and i1 %57, %56
  %59 = or i32 %29, 1
  %60 = icmp slt i32 %59, %const_reg_dword
  %61 = and i1 %60, %56
  %62 = or i32 %29, 2
  %63 = icmp slt i32 %62, %const_reg_dword
  %64 = and i1 %63, %56
  %65 = or i32 %29, 3
  %66 = icmp slt i32 %65, %const_reg_dword
  %67 = and i1 %66, %56
  %68 = or i32 %35, 1
  %69 = icmp slt i32 %68, %const_reg_dword1
  %70 = and i1 %57, %69
  %71 = and i1 %60, %69
  %72 = and i1 %63, %69
  %73 = and i1 %66, %69
  %74 = or i32 %35, 2
  %75 = icmp slt i32 %74, %const_reg_dword1
  %76 = and i1 %57, %75
  %77 = and i1 %60, %75
  %78 = and i1 %63, %75
  %79 = and i1 %66, %75
  %80 = or i32 %35, 3
  %81 = icmp slt i32 %80, %const_reg_dword1
  %82 = and i1 %57, %81
  %83 = and i1 %60, %81
  %84 = and i1 %63, %81
  %85 = and i1 %66, %81
  br label %.preheader2.preheader

.preheader2.preheader:                            ; preds = %.lr.ph, %.preheader1.3
  %86 = phi i32 [ %19, %.lr.ph ], [ %786, %.preheader1.3 ]
  %87 = phi float addrspace(4)* [ %47, %.lr.ph ], [ %785, %.preheader1.3 ]
  %88 = phi float addrspace(4)* [ %45, %.lr.ph ], [ %784, %.preheader1.3 ]
  %89 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %41, %.lr.ph ], [ %783, %.preheader1.3 ]
  %90 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %39, %.lr.ph ], [ %782, %.preheader1.3 ]
  br i1 %49, label %.preheader.preheader, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.preheader.3, %.preheader2.preheader
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader ], [ %475, %.preheader.3 ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader ], [ %383, %.preheader.3 ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader ], [ %291, %.preheader.3 ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader ], [ %199, %.preheader.3 ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader ], [ %452, %.preheader.3 ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader ], [ %360, %.preheader.3 ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader ], [ %268, %.preheader.3 ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader ], [ %176, %.preheader.3 ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader ], [ %429, %.preheader.3 ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader ], [ %337, %.preheader.3 ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader ], [ %245, %.preheader.3 ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader ], [ %153, %.preheader.3 ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader ], [ %406, %.preheader.3 ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader ], [ %314, %.preheader.3 ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader ], [ %222, %.preheader.3 ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader ], [ %130, %.preheader.3 ]
  br i1 %58, label %478, label %._crit_edge70

.preheader.preheader:                             ; preds = %.preheader2.preheader, %.preheader.3
  %91 = phi float [ %475, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %92 = phi float [ %452, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %93 = phi float [ %429, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %94 = phi float [ %406, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %95 = phi float [ %383, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %96 = phi float [ %360, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %97 = phi float [ %337, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %98 = phi float [ %314, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %99 = phi float [ %291, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %100 = phi float [ %268, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %101 = phi float [ %245, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %102 = phi float [ %222, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %103 = phi float [ %199, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %104 = phi float [ %176, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %105 = phi float [ %153, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %106 = phi float [ %130, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %107 = phi i32 [ %476, %.preheader.3 ], [ 0, %.preheader2.preheader ]
  br i1 %58, label %108, label %._crit_edge

108:                                              ; preds = %.preheader.preheader
  %.sroa.64400.0.insert.ext = zext i32 %107 to i64
  %109 = sext i32 %29 to i64
  %110 = mul nsw i64 %109, %const_reg_qword3, !spirv.Decorations !836
  %111 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %110, i32 0
  %112 = getelementptr i16, i16 addrspace(4)* %111, i64 %.sroa.64400.0.insert.ext
  %113 = addrspacecast i16 addrspace(4)* %112 to i16 addrspace(1)*
  %114 = load i16, i16 addrspace(1)* %113, align 2
  %115 = sext i32 %35 to i64
  %116 = mul nsw i64 %115, %const_reg_qword5, !spirv.Decorations !836
  %117 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %116
  %118 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %117, i64 %.sroa.64400.0.insert.ext
  %119 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %118, i64 0, i32 0
  %120 = addrspacecast i16 addrspace(4)* %119 to i16 addrspace(1)*
  %121 = load i16, i16 addrspace(1)* %120, align 2
  %122 = zext i16 %114 to i32
  %123 = shl nuw i32 %122, 16, !spirv.Decorations !838
  %124 = bitcast i32 %123 to float
  %125 = zext i16 %121 to i32
  %126 = shl nuw i32 %125, 16, !spirv.Decorations !838
  %127 = bitcast i32 %126 to float
  %128 = fmul reassoc nsz arcp contract float %124, %127, !spirv.Decorations !843
  %129 = fadd reassoc nsz arcp contract float %128, %106, !spirv.Decorations !843
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %108
  %130 = phi float [ %129, %108 ], [ %106, %.preheader.preheader ]
  br i1 %61, label %131, label %._crit_edge.1

131:                                              ; preds = %._crit_edge
  %.sroa.64400.0.insert.ext402 = zext i32 %107 to i64
  %132 = sext i32 %59 to i64
  %133 = mul nsw i64 %132, %const_reg_qword3, !spirv.Decorations !836
  %134 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %133, i32 0
  %135 = getelementptr i16, i16 addrspace(4)* %134, i64 %.sroa.64400.0.insert.ext402
  %136 = addrspacecast i16 addrspace(4)* %135 to i16 addrspace(1)*
  %137 = load i16, i16 addrspace(1)* %136, align 2
  %138 = sext i32 %35 to i64
  %139 = mul nsw i64 %138, %const_reg_qword5, !spirv.Decorations !836
  %140 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %139
  %141 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %140, i64 %.sroa.64400.0.insert.ext402
  %142 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %141, i64 0, i32 0
  %143 = addrspacecast i16 addrspace(4)* %142 to i16 addrspace(1)*
  %144 = load i16, i16 addrspace(1)* %143, align 2
  %145 = zext i16 %137 to i32
  %146 = shl nuw i32 %145, 16, !spirv.Decorations !838
  %147 = bitcast i32 %146 to float
  %148 = zext i16 %144 to i32
  %149 = shl nuw i32 %148, 16, !spirv.Decorations !838
  %150 = bitcast i32 %149 to float
  %151 = fmul reassoc nsz arcp contract float %147, %150, !spirv.Decorations !843
  %152 = fadd reassoc nsz arcp contract float %151, %105, !spirv.Decorations !843
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %131
  %153 = phi float [ %152, %131 ], [ %105, %._crit_edge ]
  br i1 %64, label %154, label %._crit_edge.2

154:                                              ; preds = %._crit_edge.1
  %.sroa.64400.0.insert.ext407 = zext i32 %107 to i64
  %155 = sext i32 %62 to i64
  %156 = mul nsw i64 %155, %const_reg_qword3, !spirv.Decorations !836
  %157 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %156, i32 0
  %158 = getelementptr i16, i16 addrspace(4)* %157, i64 %.sroa.64400.0.insert.ext407
  %159 = addrspacecast i16 addrspace(4)* %158 to i16 addrspace(1)*
  %160 = load i16, i16 addrspace(1)* %159, align 2
  %161 = sext i32 %35 to i64
  %162 = mul nsw i64 %161, %const_reg_qword5, !spirv.Decorations !836
  %163 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %162
  %164 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %163, i64 %.sroa.64400.0.insert.ext407
  %165 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %164, i64 0, i32 0
  %166 = addrspacecast i16 addrspace(4)* %165 to i16 addrspace(1)*
  %167 = load i16, i16 addrspace(1)* %166, align 2
  %168 = zext i16 %160 to i32
  %169 = shl nuw i32 %168, 16, !spirv.Decorations !838
  %170 = bitcast i32 %169 to float
  %171 = zext i16 %167 to i32
  %172 = shl nuw i32 %171, 16, !spirv.Decorations !838
  %173 = bitcast i32 %172 to float
  %174 = fmul reassoc nsz arcp contract float %170, %173, !spirv.Decorations !843
  %175 = fadd reassoc nsz arcp contract float %174, %104, !spirv.Decorations !843
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %154
  %176 = phi float [ %175, %154 ], [ %104, %._crit_edge.1 ]
  br i1 %67, label %177, label %.preheader

177:                                              ; preds = %._crit_edge.2
  %.sroa.64400.0.insert.ext412 = zext i32 %107 to i64
  %178 = sext i32 %65 to i64
  %179 = mul nsw i64 %178, %const_reg_qword3, !spirv.Decorations !836
  %180 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %179, i32 0
  %181 = getelementptr i16, i16 addrspace(4)* %180, i64 %.sroa.64400.0.insert.ext412
  %182 = addrspacecast i16 addrspace(4)* %181 to i16 addrspace(1)*
  %183 = load i16, i16 addrspace(1)* %182, align 2
  %184 = sext i32 %35 to i64
  %185 = mul nsw i64 %184, %const_reg_qword5, !spirv.Decorations !836
  %186 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %185
  %187 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %186, i64 %.sroa.64400.0.insert.ext412
  %188 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %187, i64 0, i32 0
  %189 = addrspacecast i16 addrspace(4)* %188 to i16 addrspace(1)*
  %190 = load i16, i16 addrspace(1)* %189, align 2
  %191 = zext i16 %183 to i32
  %192 = shl nuw i32 %191, 16, !spirv.Decorations !838
  %193 = bitcast i32 %192 to float
  %194 = zext i16 %190 to i32
  %195 = shl nuw i32 %194, 16, !spirv.Decorations !838
  %196 = bitcast i32 %195 to float
  %197 = fmul reassoc nsz arcp contract float %193, %196, !spirv.Decorations !843
  %198 = fadd reassoc nsz arcp contract float %197, %103, !spirv.Decorations !843
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %177
  %199 = phi float [ %198, %177 ], [ %103, %._crit_edge.2 ]
  br i1 %70, label %200, label %._crit_edge.173

200:                                              ; preds = %.preheader
  %.sroa.64400.0.insert.ext417 = zext i32 %107 to i64
  %201 = sext i32 %29 to i64
  %202 = mul nsw i64 %201, %const_reg_qword3, !spirv.Decorations !836
  %203 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %202, i32 0
  %204 = getelementptr i16, i16 addrspace(4)* %203, i64 %.sroa.64400.0.insert.ext417
  %205 = addrspacecast i16 addrspace(4)* %204 to i16 addrspace(1)*
  %206 = load i16, i16 addrspace(1)* %205, align 2
  %207 = sext i32 %68 to i64
  %208 = mul nsw i64 %207, %const_reg_qword5, !spirv.Decorations !836
  %209 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %208
  %210 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %209, i64 %.sroa.64400.0.insert.ext417
  %211 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %210, i64 0, i32 0
  %212 = addrspacecast i16 addrspace(4)* %211 to i16 addrspace(1)*
  %213 = load i16, i16 addrspace(1)* %212, align 2
  %214 = zext i16 %206 to i32
  %215 = shl nuw i32 %214, 16, !spirv.Decorations !838
  %216 = bitcast i32 %215 to float
  %217 = zext i16 %213 to i32
  %218 = shl nuw i32 %217, 16, !spirv.Decorations !838
  %219 = bitcast i32 %218 to float
  %220 = fmul reassoc nsz arcp contract float %216, %219, !spirv.Decorations !843
  %221 = fadd reassoc nsz arcp contract float %220, %102, !spirv.Decorations !843
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %200
  %222 = phi float [ %221, %200 ], [ %102, %.preheader ]
  br i1 %71, label %223, label %._crit_edge.1.1

223:                                              ; preds = %._crit_edge.173
  %.sroa.64400.0.insert.ext422 = zext i32 %107 to i64
  %224 = sext i32 %59 to i64
  %225 = mul nsw i64 %224, %const_reg_qword3, !spirv.Decorations !836
  %226 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %225, i32 0
  %227 = getelementptr i16, i16 addrspace(4)* %226, i64 %.sroa.64400.0.insert.ext422
  %228 = addrspacecast i16 addrspace(4)* %227 to i16 addrspace(1)*
  %229 = load i16, i16 addrspace(1)* %228, align 2
  %230 = sext i32 %68 to i64
  %231 = mul nsw i64 %230, %const_reg_qword5, !spirv.Decorations !836
  %232 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %231
  %233 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %232, i64 %.sroa.64400.0.insert.ext422
  %234 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %233, i64 0, i32 0
  %235 = addrspacecast i16 addrspace(4)* %234 to i16 addrspace(1)*
  %236 = load i16, i16 addrspace(1)* %235, align 2
  %237 = zext i16 %229 to i32
  %238 = shl nuw i32 %237, 16, !spirv.Decorations !838
  %239 = bitcast i32 %238 to float
  %240 = zext i16 %236 to i32
  %241 = shl nuw i32 %240, 16, !spirv.Decorations !838
  %242 = bitcast i32 %241 to float
  %243 = fmul reassoc nsz arcp contract float %239, %242, !spirv.Decorations !843
  %244 = fadd reassoc nsz arcp contract float %243, %101, !spirv.Decorations !843
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %223
  %245 = phi float [ %244, %223 ], [ %101, %._crit_edge.173 ]
  br i1 %72, label %246, label %._crit_edge.2.1

246:                                              ; preds = %._crit_edge.1.1
  %.sroa.64400.0.insert.ext427 = zext i32 %107 to i64
  %247 = sext i32 %62 to i64
  %248 = mul nsw i64 %247, %const_reg_qword3, !spirv.Decorations !836
  %249 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %248, i32 0
  %250 = getelementptr i16, i16 addrspace(4)* %249, i64 %.sroa.64400.0.insert.ext427
  %251 = addrspacecast i16 addrspace(4)* %250 to i16 addrspace(1)*
  %252 = load i16, i16 addrspace(1)* %251, align 2
  %253 = sext i32 %68 to i64
  %254 = mul nsw i64 %253, %const_reg_qword5, !spirv.Decorations !836
  %255 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %254
  %256 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %255, i64 %.sroa.64400.0.insert.ext427
  %257 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %256, i64 0, i32 0
  %258 = addrspacecast i16 addrspace(4)* %257 to i16 addrspace(1)*
  %259 = load i16, i16 addrspace(1)* %258, align 2
  %260 = zext i16 %252 to i32
  %261 = shl nuw i32 %260, 16, !spirv.Decorations !838
  %262 = bitcast i32 %261 to float
  %263 = zext i16 %259 to i32
  %264 = shl nuw i32 %263, 16, !spirv.Decorations !838
  %265 = bitcast i32 %264 to float
  %266 = fmul reassoc nsz arcp contract float %262, %265, !spirv.Decorations !843
  %267 = fadd reassoc nsz arcp contract float %266, %100, !spirv.Decorations !843
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %246
  %268 = phi float [ %267, %246 ], [ %100, %._crit_edge.1.1 ]
  br i1 %73, label %269, label %.preheader.1

269:                                              ; preds = %._crit_edge.2.1
  %.sroa.64400.0.insert.ext432 = zext i32 %107 to i64
  %270 = sext i32 %65 to i64
  %271 = mul nsw i64 %270, %const_reg_qword3, !spirv.Decorations !836
  %272 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %271, i32 0
  %273 = getelementptr i16, i16 addrspace(4)* %272, i64 %.sroa.64400.0.insert.ext432
  %274 = addrspacecast i16 addrspace(4)* %273 to i16 addrspace(1)*
  %275 = load i16, i16 addrspace(1)* %274, align 2
  %276 = sext i32 %68 to i64
  %277 = mul nsw i64 %276, %const_reg_qword5, !spirv.Decorations !836
  %278 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %277
  %279 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %278, i64 %.sroa.64400.0.insert.ext432
  %280 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %279, i64 0, i32 0
  %281 = addrspacecast i16 addrspace(4)* %280 to i16 addrspace(1)*
  %282 = load i16, i16 addrspace(1)* %281, align 2
  %283 = zext i16 %275 to i32
  %284 = shl nuw i32 %283, 16, !spirv.Decorations !838
  %285 = bitcast i32 %284 to float
  %286 = zext i16 %282 to i32
  %287 = shl nuw i32 %286, 16, !spirv.Decorations !838
  %288 = bitcast i32 %287 to float
  %289 = fmul reassoc nsz arcp contract float %285, %288, !spirv.Decorations !843
  %290 = fadd reassoc nsz arcp contract float %289, %99, !spirv.Decorations !843
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %269
  %291 = phi float [ %290, %269 ], [ %99, %._crit_edge.2.1 ]
  br i1 %76, label %292, label %._crit_edge.274

292:                                              ; preds = %.preheader.1
  %.sroa.64400.0.insert.ext437 = zext i32 %107 to i64
  %293 = sext i32 %29 to i64
  %294 = mul nsw i64 %293, %const_reg_qword3, !spirv.Decorations !836
  %295 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %294, i32 0
  %296 = getelementptr i16, i16 addrspace(4)* %295, i64 %.sroa.64400.0.insert.ext437
  %297 = addrspacecast i16 addrspace(4)* %296 to i16 addrspace(1)*
  %298 = load i16, i16 addrspace(1)* %297, align 2
  %299 = sext i32 %74 to i64
  %300 = mul nsw i64 %299, %const_reg_qword5, !spirv.Decorations !836
  %301 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %300
  %302 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %301, i64 %.sroa.64400.0.insert.ext437
  %303 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %302, i64 0, i32 0
  %304 = addrspacecast i16 addrspace(4)* %303 to i16 addrspace(1)*
  %305 = load i16, i16 addrspace(1)* %304, align 2
  %306 = zext i16 %298 to i32
  %307 = shl nuw i32 %306, 16, !spirv.Decorations !838
  %308 = bitcast i32 %307 to float
  %309 = zext i16 %305 to i32
  %310 = shl nuw i32 %309, 16, !spirv.Decorations !838
  %311 = bitcast i32 %310 to float
  %312 = fmul reassoc nsz arcp contract float %308, %311, !spirv.Decorations !843
  %313 = fadd reassoc nsz arcp contract float %312, %98, !spirv.Decorations !843
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %292
  %314 = phi float [ %313, %292 ], [ %98, %.preheader.1 ]
  br i1 %77, label %315, label %._crit_edge.1.2

315:                                              ; preds = %._crit_edge.274
  %.sroa.64400.0.insert.ext442 = zext i32 %107 to i64
  %316 = sext i32 %59 to i64
  %317 = mul nsw i64 %316, %const_reg_qword3, !spirv.Decorations !836
  %318 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %317, i32 0
  %319 = getelementptr i16, i16 addrspace(4)* %318, i64 %.sroa.64400.0.insert.ext442
  %320 = addrspacecast i16 addrspace(4)* %319 to i16 addrspace(1)*
  %321 = load i16, i16 addrspace(1)* %320, align 2
  %322 = sext i32 %74 to i64
  %323 = mul nsw i64 %322, %const_reg_qword5, !spirv.Decorations !836
  %324 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %323
  %325 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %324, i64 %.sroa.64400.0.insert.ext442
  %326 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %325, i64 0, i32 0
  %327 = addrspacecast i16 addrspace(4)* %326 to i16 addrspace(1)*
  %328 = load i16, i16 addrspace(1)* %327, align 2
  %329 = zext i16 %321 to i32
  %330 = shl nuw i32 %329, 16, !spirv.Decorations !838
  %331 = bitcast i32 %330 to float
  %332 = zext i16 %328 to i32
  %333 = shl nuw i32 %332, 16, !spirv.Decorations !838
  %334 = bitcast i32 %333 to float
  %335 = fmul reassoc nsz arcp contract float %331, %334, !spirv.Decorations !843
  %336 = fadd reassoc nsz arcp contract float %335, %97, !spirv.Decorations !843
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %315
  %337 = phi float [ %336, %315 ], [ %97, %._crit_edge.274 ]
  br i1 %78, label %338, label %._crit_edge.2.2

338:                                              ; preds = %._crit_edge.1.2
  %.sroa.64400.0.insert.ext447 = zext i32 %107 to i64
  %339 = sext i32 %62 to i64
  %340 = mul nsw i64 %339, %const_reg_qword3, !spirv.Decorations !836
  %341 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %340, i32 0
  %342 = getelementptr i16, i16 addrspace(4)* %341, i64 %.sroa.64400.0.insert.ext447
  %343 = addrspacecast i16 addrspace(4)* %342 to i16 addrspace(1)*
  %344 = load i16, i16 addrspace(1)* %343, align 2
  %345 = sext i32 %74 to i64
  %346 = mul nsw i64 %345, %const_reg_qword5, !spirv.Decorations !836
  %347 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %346
  %348 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %347, i64 %.sroa.64400.0.insert.ext447
  %349 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %348, i64 0, i32 0
  %350 = addrspacecast i16 addrspace(4)* %349 to i16 addrspace(1)*
  %351 = load i16, i16 addrspace(1)* %350, align 2
  %352 = zext i16 %344 to i32
  %353 = shl nuw i32 %352, 16, !spirv.Decorations !838
  %354 = bitcast i32 %353 to float
  %355 = zext i16 %351 to i32
  %356 = shl nuw i32 %355, 16, !spirv.Decorations !838
  %357 = bitcast i32 %356 to float
  %358 = fmul reassoc nsz arcp contract float %354, %357, !spirv.Decorations !843
  %359 = fadd reassoc nsz arcp contract float %358, %96, !spirv.Decorations !843
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %338
  %360 = phi float [ %359, %338 ], [ %96, %._crit_edge.1.2 ]
  br i1 %79, label %361, label %.preheader.2

361:                                              ; preds = %._crit_edge.2.2
  %.sroa.64400.0.insert.ext452 = zext i32 %107 to i64
  %362 = sext i32 %65 to i64
  %363 = mul nsw i64 %362, %const_reg_qword3, !spirv.Decorations !836
  %364 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %363, i32 0
  %365 = getelementptr i16, i16 addrspace(4)* %364, i64 %.sroa.64400.0.insert.ext452
  %366 = addrspacecast i16 addrspace(4)* %365 to i16 addrspace(1)*
  %367 = load i16, i16 addrspace(1)* %366, align 2
  %368 = sext i32 %74 to i64
  %369 = mul nsw i64 %368, %const_reg_qword5, !spirv.Decorations !836
  %370 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %369
  %371 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %370, i64 %.sroa.64400.0.insert.ext452
  %372 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %371, i64 0, i32 0
  %373 = addrspacecast i16 addrspace(4)* %372 to i16 addrspace(1)*
  %374 = load i16, i16 addrspace(1)* %373, align 2
  %375 = zext i16 %367 to i32
  %376 = shl nuw i32 %375, 16, !spirv.Decorations !838
  %377 = bitcast i32 %376 to float
  %378 = zext i16 %374 to i32
  %379 = shl nuw i32 %378, 16, !spirv.Decorations !838
  %380 = bitcast i32 %379 to float
  %381 = fmul reassoc nsz arcp contract float %377, %380, !spirv.Decorations !843
  %382 = fadd reassoc nsz arcp contract float %381, %95, !spirv.Decorations !843
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %361
  %383 = phi float [ %382, %361 ], [ %95, %._crit_edge.2.2 ]
  br i1 %82, label %384, label %._crit_edge.375

384:                                              ; preds = %.preheader.2
  %.sroa.64400.0.insert.ext457 = zext i32 %107 to i64
  %385 = sext i32 %29 to i64
  %386 = mul nsw i64 %385, %const_reg_qword3, !spirv.Decorations !836
  %387 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %386, i32 0
  %388 = getelementptr i16, i16 addrspace(4)* %387, i64 %.sroa.64400.0.insert.ext457
  %389 = addrspacecast i16 addrspace(4)* %388 to i16 addrspace(1)*
  %390 = load i16, i16 addrspace(1)* %389, align 2
  %391 = sext i32 %80 to i64
  %392 = mul nsw i64 %391, %const_reg_qword5, !spirv.Decorations !836
  %393 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %392
  %394 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %393, i64 %.sroa.64400.0.insert.ext457
  %395 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %394, i64 0, i32 0
  %396 = addrspacecast i16 addrspace(4)* %395 to i16 addrspace(1)*
  %397 = load i16, i16 addrspace(1)* %396, align 2
  %398 = zext i16 %390 to i32
  %399 = shl nuw i32 %398, 16, !spirv.Decorations !838
  %400 = bitcast i32 %399 to float
  %401 = zext i16 %397 to i32
  %402 = shl nuw i32 %401, 16, !spirv.Decorations !838
  %403 = bitcast i32 %402 to float
  %404 = fmul reassoc nsz arcp contract float %400, %403, !spirv.Decorations !843
  %405 = fadd reassoc nsz arcp contract float %404, %94, !spirv.Decorations !843
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %384
  %406 = phi float [ %405, %384 ], [ %94, %.preheader.2 ]
  br i1 %83, label %407, label %._crit_edge.1.3

407:                                              ; preds = %._crit_edge.375
  %.sroa.64400.0.insert.ext462 = zext i32 %107 to i64
  %408 = sext i32 %59 to i64
  %409 = mul nsw i64 %408, %const_reg_qword3, !spirv.Decorations !836
  %410 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %409, i32 0
  %411 = getelementptr i16, i16 addrspace(4)* %410, i64 %.sroa.64400.0.insert.ext462
  %412 = addrspacecast i16 addrspace(4)* %411 to i16 addrspace(1)*
  %413 = load i16, i16 addrspace(1)* %412, align 2
  %414 = sext i32 %80 to i64
  %415 = mul nsw i64 %414, %const_reg_qword5, !spirv.Decorations !836
  %416 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %415
  %417 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %416, i64 %.sroa.64400.0.insert.ext462
  %418 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %417, i64 0, i32 0
  %419 = addrspacecast i16 addrspace(4)* %418 to i16 addrspace(1)*
  %420 = load i16, i16 addrspace(1)* %419, align 2
  %421 = zext i16 %413 to i32
  %422 = shl nuw i32 %421, 16, !spirv.Decorations !838
  %423 = bitcast i32 %422 to float
  %424 = zext i16 %420 to i32
  %425 = shl nuw i32 %424, 16, !spirv.Decorations !838
  %426 = bitcast i32 %425 to float
  %427 = fmul reassoc nsz arcp contract float %423, %426, !spirv.Decorations !843
  %428 = fadd reassoc nsz arcp contract float %427, %93, !spirv.Decorations !843
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %407
  %429 = phi float [ %428, %407 ], [ %93, %._crit_edge.375 ]
  br i1 %84, label %430, label %._crit_edge.2.3

430:                                              ; preds = %._crit_edge.1.3
  %.sroa.64400.0.insert.ext467 = zext i32 %107 to i64
  %431 = sext i32 %62 to i64
  %432 = mul nsw i64 %431, %const_reg_qword3, !spirv.Decorations !836
  %433 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %432, i32 0
  %434 = getelementptr i16, i16 addrspace(4)* %433, i64 %.sroa.64400.0.insert.ext467
  %435 = addrspacecast i16 addrspace(4)* %434 to i16 addrspace(1)*
  %436 = load i16, i16 addrspace(1)* %435, align 2
  %437 = sext i32 %80 to i64
  %438 = mul nsw i64 %437, %const_reg_qword5, !spirv.Decorations !836
  %439 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %438
  %440 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %439, i64 %.sroa.64400.0.insert.ext467
  %441 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %440, i64 0, i32 0
  %442 = addrspacecast i16 addrspace(4)* %441 to i16 addrspace(1)*
  %443 = load i16, i16 addrspace(1)* %442, align 2
  %444 = zext i16 %436 to i32
  %445 = shl nuw i32 %444, 16, !spirv.Decorations !838
  %446 = bitcast i32 %445 to float
  %447 = zext i16 %443 to i32
  %448 = shl nuw i32 %447, 16, !spirv.Decorations !838
  %449 = bitcast i32 %448 to float
  %450 = fmul reassoc nsz arcp contract float %446, %449, !spirv.Decorations !843
  %451 = fadd reassoc nsz arcp contract float %450, %92, !spirv.Decorations !843
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %430
  %452 = phi float [ %451, %430 ], [ %92, %._crit_edge.1.3 ]
  br i1 %85, label %453, label %.preheader.3

453:                                              ; preds = %._crit_edge.2.3
  %.sroa.64400.0.insert.ext472 = zext i32 %107 to i64
  %454 = sext i32 %65 to i64
  %455 = mul nsw i64 %454, %const_reg_qword3, !spirv.Decorations !836
  %456 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %455, i32 0
  %457 = getelementptr i16, i16 addrspace(4)* %456, i64 %.sroa.64400.0.insert.ext472
  %458 = addrspacecast i16 addrspace(4)* %457 to i16 addrspace(1)*
  %459 = load i16, i16 addrspace(1)* %458, align 2
  %460 = sext i32 %80 to i64
  %461 = mul nsw i64 %460, %const_reg_qword5, !spirv.Decorations !836
  %462 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %461
  %463 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %462, i64 %.sroa.64400.0.insert.ext472
  %464 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %463, i64 0, i32 0
  %465 = addrspacecast i16 addrspace(4)* %464 to i16 addrspace(1)*
  %466 = load i16, i16 addrspace(1)* %465, align 2
  %467 = zext i16 %459 to i32
  %468 = shl nuw i32 %467, 16, !spirv.Decorations !838
  %469 = bitcast i32 %468 to float
  %470 = zext i16 %466 to i32
  %471 = shl nuw i32 %470, 16, !spirv.Decorations !838
  %472 = bitcast i32 %471 to float
  %473 = fmul reassoc nsz arcp contract float %469, %472, !spirv.Decorations !843
  %474 = fadd reassoc nsz arcp contract float %473, %91, !spirv.Decorations !843
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %453
  %475 = phi float [ %474, %453 ], [ %91, %._crit_edge.2.3 ]
  %476 = add nuw nsw i32 %107, 1, !spirv.Decorations !848
  %477 = icmp slt i32 %476, %const_reg_dword2
  br i1 %477, label %.preheader.preheader, label %.preheader1.preheader

478:                                              ; preds = %.preheader1.preheader
  %479 = sext i32 %29 to i64
  %480 = sext i32 %35 to i64
  %481 = mul nsw i64 %479, %const_reg_qword9, !spirv.Decorations !836
  %482 = add nsw i64 %481, %480, !spirv.Decorations !836
  %483 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !843
  br i1 %42, label %484, label %494

484:                                              ; preds = %478
  %485 = mul nsw i64 %479, %const_reg_qword7, !spirv.Decorations !836
  %486 = getelementptr float, float addrspace(4)* %88, i64 %485
  %487 = getelementptr float, float addrspace(4)* %486, i64 %480
  %488 = addrspacecast float addrspace(4)* %487 to float addrspace(1)*
  %489 = load float, float addrspace(1)* %488, align 4
  %490 = fmul reassoc nsz arcp contract float %489, %4, !spirv.Decorations !843
  %491 = fadd reassoc nsz arcp contract float %483, %490, !spirv.Decorations !843
  %492 = getelementptr inbounds float, float addrspace(4)* %87, i64 %482
  %493 = addrspacecast float addrspace(4)* %492 to float addrspace(1)*
  store float %491, float addrspace(1)* %493, align 4
  br label %._crit_edge70

494:                                              ; preds = %478
  %495 = getelementptr inbounds float, float addrspace(4)* %87, i64 %482
  %496 = addrspacecast float addrspace(4)* %495 to float addrspace(1)*
  store float %483, float addrspace(1)* %496, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %.preheader1.preheader, %494, %484
  br i1 %61, label %497, label %._crit_edge70.1

497:                                              ; preds = %._crit_edge70
  %498 = sext i32 %59 to i64
  %499 = sext i32 %35 to i64
  %500 = mul nsw i64 %498, %const_reg_qword9, !spirv.Decorations !836
  %501 = add nsw i64 %500, %499, !spirv.Decorations !836
  %502 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !843
  br i1 %42, label %506, label %503

503:                                              ; preds = %497
  %504 = getelementptr inbounds float, float addrspace(4)* %87, i64 %501
  %505 = addrspacecast float addrspace(4)* %504 to float addrspace(1)*
  store float %502, float addrspace(1)* %505, align 4
  br label %._crit_edge70.1

506:                                              ; preds = %497
  %507 = mul nsw i64 %498, %const_reg_qword7, !spirv.Decorations !836
  %508 = getelementptr float, float addrspace(4)* %88, i64 %507
  %509 = getelementptr float, float addrspace(4)* %508, i64 %499
  %510 = addrspacecast float addrspace(4)* %509 to float addrspace(1)*
  %511 = load float, float addrspace(1)* %510, align 4
  %512 = fmul reassoc nsz arcp contract float %511, %4, !spirv.Decorations !843
  %513 = fadd reassoc nsz arcp contract float %502, %512, !spirv.Decorations !843
  %514 = getelementptr inbounds float, float addrspace(4)* %87, i64 %501
  %515 = addrspacecast float addrspace(4)* %514 to float addrspace(1)*
  store float %513, float addrspace(1)* %515, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %._crit_edge70, %506, %503
  br i1 %64, label %516, label %._crit_edge70.2

516:                                              ; preds = %._crit_edge70.1
  %517 = sext i32 %62 to i64
  %518 = sext i32 %35 to i64
  %519 = mul nsw i64 %517, %const_reg_qword9, !spirv.Decorations !836
  %520 = add nsw i64 %519, %518, !spirv.Decorations !836
  %521 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !843
  br i1 %42, label %525, label %522

522:                                              ; preds = %516
  %523 = getelementptr inbounds float, float addrspace(4)* %87, i64 %520
  %524 = addrspacecast float addrspace(4)* %523 to float addrspace(1)*
  store float %521, float addrspace(1)* %524, align 4
  br label %._crit_edge70.2

525:                                              ; preds = %516
  %526 = mul nsw i64 %517, %const_reg_qword7, !spirv.Decorations !836
  %527 = getelementptr float, float addrspace(4)* %88, i64 %526
  %528 = getelementptr float, float addrspace(4)* %527, i64 %518
  %529 = addrspacecast float addrspace(4)* %528 to float addrspace(1)*
  %530 = load float, float addrspace(1)* %529, align 4
  %531 = fmul reassoc nsz arcp contract float %530, %4, !spirv.Decorations !843
  %532 = fadd reassoc nsz arcp contract float %521, %531, !spirv.Decorations !843
  %533 = getelementptr inbounds float, float addrspace(4)* %87, i64 %520
  %534 = addrspacecast float addrspace(4)* %533 to float addrspace(1)*
  store float %532, float addrspace(1)* %534, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %._crit_edge70.1, %525, %522
  br i1 %67, label %535, label %.preheader1

535:                                              ; preds = %._crit_edge70.2
  %536 = sext i32 %65 to i64
  %537 = sext i32 %35 to i64
  %538 = mul nsw i64 %536, %const_reg_qword9, !spirv.Decorations !836
  %539 = add nsw i64 %538, %537, !spirv.Decorations !836
  %540 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !843
  br i1 %42, label %544, label %541

541:                                              ; preds = %535
  %542 = getelementptr inbounds float, float addrspace(4)* %87, i64 %539
  %543 = addrspacecast float addrspace(4)* %542 to float addrspace(1)*
  store float %540, float addrspace(1)* %543, align 4
  br label %.preheader1

544:                                              ; preds = %535
  %545 = mul nsw i64 %536, %const_reg_qword7, !spirv.Decorations !836
  %546 = getelementptr float, float addrspace(4)* %88, i64 %545
  %547 = getelementptr float, float addrspace(4)* %546, i64 %537
  %548 = addrspacecast float addrspace(4)* %547 to float addrspace(1)*
  %549 = load float, float addrspace(1)* %548, align 4
  %550 = fmul reassoc nsz arcp contract float %549, %4, !spirv.Decorations !843
  %551 = fadd reassoc nsz arcp contract float %540, %550, !spirv.Decorations !843
  %552 = getelementptr inbounds float, float addrspace(4)* %87, i64 %539
  %553 = addrspacecast float addrspace(4)* %552 to float addrspace(1)*
  store float %551, float addrspace(1)* %553, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %._crit_edge70.2, %544, %541
  br i1 %70, label %554, label %._crit_edge70.176

554:                                              ; preds = %.preheader1
  %555 = sext i32 %29 to i64
  %556 = sext i32 %68 to i64
  %557 = mul nsw i64 %555, %const_reg_qword9, !spirv.Decorations !836
  %558 = add nsw i64 %557, %556, !spirv.Decorations !836
  %559 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !843
  br i1 %42, label %563, label %560

560:                                              ; preds = %554
  %561 = getelementptr inbounds float, float addrspace(4)* %87, i64 %558
  %562 = addrspacecast float addrspace(4)* %561 to float addrspace(1)*
  store float %559, float addrspace(1)* %562, align 4
  br label %._crit_edge70.176

563:                                              ; preds = %554
  %564 = mul nsw i64 %555, %const_reg_qword7, !spirv.Decorations !836
  %565 = getelementptr float, float addrspace(4)* %88, i64 %564
  %566 = getelementptr float, float addrspace(4)* %565, i64 %556
  %567 = addrspacecast float addrspace(4)* %566 to float addrspace(1)*
  %568 = load float, float addrspace(1)* %567, align 4
  %569 = fmul reassoc nsz arcp contract float %568, %4, !spirv.Decorations !843
  %570 = fadd reassoc nsz arcp contract float %559, %569, !spirv.Decorations !843
  %571 = getelementptr inbounds float, float addrspace(4)* %87, i64 %558
  %572 = addrspacecast float addrspace(4)* %571 to float addrspace(1)*
  store float %570, float addrspace(1)* %572, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %.preheader1, %563, %560
  br i1 %71, label %573, label %._crit_edge70.1.1

573:                                              ; preds = %._crit_edge70.176
  %574 = sext i32 %59 to i64
  %575 = sext i32 %68 to i64
  %576 = mul nsw i64 %574, %const_reg_qword9, !spirv.Decorations !836
  %577 = add nsw i64 %576, %575, !spirv.Decorations !836
  %578 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !843
  br i1 %42, label %582, label %579

579:                                              ; preds = %573
  %580 = getelementptr inbounds float, float addrspace(4)* %87, i64 %577
  %581 = addrspacecast float addrspace(4)* %580 to float addrspace(1)*
  store float %578, float addrspace(1)* %581, align 4
  br label %._crit_edge70.1.1

582:                                              ; preds = %573
  %583 = mul nsw i64 %574, %const_reg_qword7, !spirv.Decorations !836
  %584 = getelementptr float, float addrspace(4)* %88, i64 %583
  %585 = getelementptr float, float addrspace(4)* %584, i64 %575
  %586 = addrspacecast float addrspace(4)* %585 to float addrspace(1)*
  %587 = load float, float addrspace(1)* %586, align 4
  %588 = fmul reassoc nsz arcp contract float %587, %4, !spirv.Decorations !843
  %589 = fadd reassoc nsz arcp contract float %578, %588, !spirv.Decorations !843
  %590 = getelementptr inbounds float, float addrspace(4)* %87, i64 %577
  %591 = addrspacecast float addrspace(4)* %590 to float addrspace(1)*
  store float %589, float addrspace(1)* %591, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176, %582, %579
  br i1 %72, label %592, label %._crit_edge70.2.1

592:                                              ; preds = %._crit_edge70.1.1
  %593 = sext i32 %62 to i64
  %594 = sext i32 %68 to i64
  %595 = mul nsw i64 %593, %const_reg_qword9, !spirv.Decorations !836
  %596 = add nsw i64 %595, %594, !spirv.Decorations !836
  %597 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !843
  br i1 %42, label %601, label %598

598:                                              ; preds = %592
  %599 = getelementptr inbounds float, float addrspace(4)* %87, i64 %596
  %600 = addrspacecast float addrspace(4)* %599 to float addrspace(1)*
  store float %597, float addrspace(1)* %600, align 4
  br label %._crit_edge70.2.1

601:                                              ; preds = %592
  %602 = mul nsw i64 %593, %const_reg_qword7, !spirv.Decorations !836
  %603 = getelementptr float, float addrspace(4)* %88, i64 %602
  %604 = getelementptr float, float addrspace(4)* %603, i64 %594
  %605 = addrspacecast float addrspace(4)* %604 to float addrspace(1)*
  %606 = load float, float addrspace(1)* %605, align 4
  %607 = fmul reassoc nsz arcp contract float %606, %4, !spirv.Decorations !843
  %608 = fadd reassoc nsz arcp contract float %597, %607, !spirv.Decorations !843
  %609 = getelementptr inbounds float, float addrspace(4)* %87, i64 %596
  %610 = addrspacecast float addrspace(4)* %609 to float addrspace(1)*
  store float %608, float addrspace(1)* %610, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1, %601, %598
  br i1 %73, label %611, label %.preheader1.1

611:                                              ; preds = %._crit_edge70.2.1
  %612 = sext i32 %65 to i64
  %613 = sext i32 %68 to i64
  %614 = mul nsw i64 %612, %const_reg_qword9, !spirv.Decorations !836
  %615 = add nsw i64 %614, %613, !spirv.Decorations !836
  %616 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !843
  br i1 %42, label %620, label %617

617:                                              ; preds = %611
  %618 = getelementptr inbounds float, float addrspace(4)* %87, i64 %615
  %619 = addrspacecast float addrspace(4)* %618 to float addrspace(1)*
  store float %616, float addrspace(1)* %619, align 4
  br label %.preheader1.1

620:                                              ; preds = %611
  %621 = mul nsw i64 %612, %const_reg_qword7, !spirv.Decorations !836
  %622 = getelementptr float, float addrspace(4)* %88, i64 %621
  %623 = getelementptr float, float addrspace(4)* %622, i64 %613
  %624 = addrspacecast float addrspace(4)* %623 to float addrspace(1)*
  %625 = load float, float addrspace(1)* %624, align 4
  %626 = fmul reassoc nsz arcp contract float %625, %4, !spirv.Decorations !843
  %627 = fadd reassoc nsz arcp contract float %616, %626, !spirv.Decorations !843
  %628 = getelementptr inbounds float, float addrspace(4)* %87, i64 %615
  %629 = addrspacecast float addrspace(4)* %628 to float addrspace(1)*
  store float %627, float addrspace(1)* %629, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %._crit_edge70.2.1, %620, %617
  br i1 %76, label %630, label %._crit_edge70.277

630:                                              ; preds = %.preheader1.1
  %631 = sext i32 %29 to i64
  %632 = sext i32 %74 to i64
  %633 = mul nsw i64 %631, %const_reg_qword9, !spirv.Decorations !836
  %634 = add nsw i64 %633, %632, !spirv.Decorations !836
  %635 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !843
  br i1 %42, label %639, label %636

636:                                              ; preds = %630
  %637 = getelementptr inbounds float, float addrspace(4)* %87, i64 %634
  %638 = addrspacecast float addrspace(4)* %637 to float addrspace(1)*
  store float %635, float addrspace(1)* %638, align 4
  br label %._crit_edge70.277

639:                                              ; preds = %630
  %640 = mul nsw i64 %631, %const_reg_qword7, !spirv.Decorations !836
  %641 = getelementptr float, float addrspace(4)* %88, i64 %640
  %642 = getelementptr float, float addrspace(4)* %641, i64 %632
  %643 = addrspacecast float addrspace(4)* %642 to float addrspace(1)*
  %644 = load float, float addrspace(1)* %643, align 4
  %645 = fmul reassoc nsz arcp contract float %644, %4, !spirv.Decorations !843
  %646 = fadd reassoc nsz arcp contract float %635, %645, !spirv.Decorations !843
  %647 = getelementptr inbounds float, float addrspace(4)* %87, i64 %634
  %648 = addrspacecast float addrspace(4)* %647 to float addrspace(1)*
  store float %646, float addrspace(1)* %648, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %.preheader1.1, %639, %636
  br i1 %77, label %649, label %._crit_edge70.1.2

649:                                              ; preds = %._crit_edge70.277
  %650 = sext i32 %59 to i64
  %651 = sext i32 %74 to i64
  %652 = mul nsw i64 %650, %const_reg_qword9, !spirv.Decorations !836
  %653 = add nsw i64 %652, %651, !spirv.Decorations !836
  %654 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !843
  br i1 %42, label %658, label %655

655:                                              ; preds = %649
  %656 = getelementptr inbounds float, float addrspace(4)* %87, i64 %653
  %657 = addrspacecast float addrspace(4)* %656 to float addrspace(1)*
  store float %654, float addrspace(1)* %657, align 4
  br label %._crit_edge70.1.2

658:                                              ; preds = %649
  %659 = mul nsw i64 %650, %const_reg_qword7, !spirv.Decorations !836
  %660 = getelementptr float, float addrspace(4)* %88, i64 %659
  %661 = getelementptr float, float addrspace(4)* %660, i64 %651
  %662 = addrspacecast float addrspace(4)* %661 to float addrspace(1)*
  %663 = load float, float addrspace(1)* %662, align 4
  %664 = fmul reassoc nsz arcp contract float %663, %4, !spirv.Decorations !843
  %665 = fadd reassoc nsz arcp contract float %654, %664, !spirv.Decorations !843
  %666 = getelementptr inbounds float, float addrspace(4)* %87, i64 %653
  %667 = addrspacecast float addrspace(4)* %666 to float addrspace(1)*
  store float %665, float addrspace(1)* %667, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277, %658, %655
  br i1 %78, label %668, label %._crit_edge70.2.2

668:                                              ; preds = %._crit_edge70.1.2
  %669 = sext i32 %62 to i64
  %670 = sext i32 %74 to i64
  %671 = mul nsw i64 %669, %const_reg_qword9, !spirv.Decorations !836
  %672 = add nsw i64 %671, %670, !spirv.Decorations !836
  %673 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !843
  br i1 %42, label %677, label %674

674:                                              ; preds = %668
  %675 = getelementptr inbounds float, float addrspace(4)* %87, i64 %672
  %676 = addrspacecast float addrspace(4)* %675 to float addrspace(1)*
  store float %673, float addrspace(1)* %676, align 4
  br label %._crit_edge70.2.2

677:                                              ; preds = %668
  %678 = mul nsw i64 %669, %const_reg_qword7, !spirv.Decorations !836
  %679 = getelementptr float, float addrspace(4)* %88, i64 %678
  %680 = getelementptr float, float addrspace(4)* %679, i64 %670
  %681 = addrspacecast float addrspace(4)* %680 to float addrspace(1)*
  %682 = load float, float addrspace(1)* %681, align 4
  %683 = fmul reassoc nsz arcp contract float %682, %4, !spirv.Decorations !843
  %684 = fadd reassoc nsz arcp contract float %673, %683, !spirv.Decorations !843
  %685 = getelementptr inbounds float, float addrspace(4)* %87, i64 %672
  %686 = addrspacecast float addrspace(4)* %685 to float addrspace(1)*
  store float %684, float addrspace(1)* %686, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2, %677, %674
  br i1 %79, label %687, label %.preheader1.2

687:                                              ; preds = %._crit_edge70.2.2
  %688 = sext i32 %65 to i64
  %689 = sext i32 %74 to i64
  %690 = mul nsw i64 %688, %const_reg_qword9, !spirv.Decorations !836
  %691 = add nsw i64 %690, %689, !spirv.Decorations !836
  %692 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !843
  br i1 %42, label %696, label %693

693:                                              ; preds = %687
  %694 = getelementptr inbounds float, float addrspace(4)* %87, i64 %691
  %695 = addrspacecast float addrspace(4)* %694 to float addrspace(1)*
  store float %692, float addrspace(1)* %695, align 4
  br label %.preheader1.2

696:                                              ; preds = %687
  %697 = mul nsw i64 %688, %const_reg_qword7, !spirv.Decorations !836
  %698 = getelementptr float, float addrspace(4)* %88, i64 %697
  %699 = getelementptr float, float addrspace(4)* %698, i64 %689
  %700 = addrspacecast float addrspace(4)* %699 to float addrspace(1)*
  %701 = load float, float addrspace(1)* %700, align 4
  %702 = fmul reassoc nsz arcp contract float %701, %4, !spirv.Decorations !843
  %703 = fadd reassoc nsz arcp contract float %692, %702, !spirv.Decorations !843
  %704 = getelementptr inbounds float, float addrspace(4)* %87, i64 %691
  %705 = addrspacecast float addrspace(4)* %704 to float addrspace(1)*
  store float %703, float addrspace(1)* %705, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %._crit_edge70.2.2, %696, %693
  br i1 %82, label %706, label %._crit_edge70.378

706:                                              ; preds = %.preheader1.2
  %707 = sext i32 %29 to i64
  %708 = sext i32 %80 to i64
  %709 = mul nsw i64 %707, %const_reg_qword9, !spirv.Decorations !836
  %710 = add nsw i64 %709, %708, !spirv.Decorations !836
  %711 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !843
  br i1 %42, label %715, label %712

712:                                              ; preds = %706
  %713 = getelementptr inbounds float, float addrspace(4)* %87, i64 %710
  %714 = addrspacecast float addrspace(4)* %713 to float addrspace(1)*
  store float %711, float addrspace(1)* %714, align 4
  br label %._crit_edge70.378

715:                                              ; preds = %706
  %716 = mul nsw i64 %707, %const_reg_qword7, !spirv.Decorations !836
  %717 = getelementptr float, float addrspace(4)* %88, i64 %716
  %718 = getelementptr float, float addrspace(4)* %717, i64 %708
  %719 = addrspacecast float addrspace(4)* %718 to float addrspace(1)*
  %720 = load float, float addrspace(1)* %719, align 4
  %721 = fmul reassoc nsz arcp contract float %720, %4, !spirv.Decorations !843
  %722 = fadd reassoc nsz arcp contract float %711, %721, !spirv.Decorations !843
  %723 = getelementptr inbounds float, float addrspace(4)* %87, i64 %710
  %724 = addrspacecast float addrspace(4)* %723 to float addrspace(1)*
  store float %722, float addrspace(1)* %724, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %.preheader1.2, %715, %712
  br i1 %83, label %725, label %._crit_edge70.1.3

725:                                              ; preds = %._crit_edge70.378
  %726 = sext i32 %59 to i64
  %727 = sext i32 %80 to i64
  %728 = mul nsw i64 %726, %const_reg_qword9, !spirv.Decorations !836
  %729 = add nsw i64 %728, %727, !spirv.Decorations !836
  %730 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !843
  br i1 %42, label %734, label %731

731:                                              ; preds = %725
  %732 = getelementptr inbounds float, float addrspace(4)* %87, i64 %729
  %733 = addrspacecast float addrspace(4)* %732 to float addrspace(1)*
  store float %730, float addrspace(1)* %733, align 4
  br label %._crit_edge70.1.3

734:                                              ; preds = %725
  %735 = mul nsw i64 %726, %const_reg_qword7, !spirv.Decorations !836
  %736 = getelementptr float, float addrspace(4)* %88, i64 %735
  %737 = getelementptr float, float addrspace(4)* %736, i64 %727
  %738 = addrspacecast float addrspace(4)* %737 to float addrspace(1)*
  %739 = load float, float addrspace(1)* %738, align 4
  %740 = fmul reassoc nsz arcp contract float %739, %4, !spirv.Decorations !843
  %741 = fadd reassoc nsz arcp contract float %730, %740, !spirv.Decorations !843
  %742 = getelementptr inbounds float, float addrspace(4)* %87, i64 %729
  %743 = addrspacecast float addrspace(4)* %742 to float addrspace(1)*
  store float %741, float addrspace(1)* %743, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378, %734, %731
  br i1 %84, label %744, label %._crit_edge70.2.3

744:                                              ; preds = %._crit_edge70.1.3
  %745 = sext i32 %62 to i64
  %746 = sext i32 %80 to i64
  %747 = mul nsw i64 %745, %const_reg_qword9, !spirv.Decorations !836
  %748 = add nsw i64 %747, %746, !spirv.Decorations !836
  %749 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !843
  br i1 %42, label %753, label %750

750:                                              ; preds = %744
  %751 = getelementptr inbounds float, float addrspace(4)* %87, i64 %748
  %752 = addrspacecast float addrspace(4)* %751 to float addrspace(1)*
  store float %749, float addrspace(1)* %752, align 4
  br label %._crit_edge70.2.3

753:                                              ; preds = %744
  %754 = mul nsw i64 %745, %const_reg_qword7, !spirv.Decorations !836
  %755 = getelementptr float, float addrspace(4)* %88, i64 %754
  %756 = getelementptr float, float addrspace(4)* %755, i64 %746
  %757 = addrspacecast float addrspace(4)* %756 to float addrspace(1)*
  %758 = load float, float addrspace(1)* %757, align 4
  %759 = fmul reassoc nsz arcp contract float %758, %4, !spirv.Decorations !843
  %760 = fadd reassoc nsz arcp contract float %749, %759, !spirv.Decorations !843
  %761 = getelementptr inbounds float, float addrspace(4)* %87, i64 %748
  %762 = addrspacecast float addrspace(4)* %761 to float addrspace(1)*
  store float %760, float addrspace(1)* %762, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3, %753, %750
  br i1 %85, label %763, label %.preheader1.3

763:                                              ; preds = %._crit_edge70.2.3
  %764 = sext i32 %65 to i64
  %765 = sext i32 %80 to i64
  %766 = mul nsw i64 %764, %const_reg_qword9, !spirv.Decorations !836
  %767 = add nsw i64 %766, %765, !spirv.Decorations !836
  %768 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !843
  br i1 %42, label %772, label %769

769:                                              ; preds = %763
  %770 = getelementptr inbounds float, float addrspace(4)* %87, i64 %767
  %771 = addrspacecast float addrspace(4)* %770 to float addrspace(1)*
  store float %768, float addrspace(1)* %771, align 4
  br label %.preheader1.3

772:                                              ; preds = %763
  %773 = mul nsw i64 %764, %const_reg_qword7, !spirv.Decorations !836
  %774 = getelementptr float, float addrspace(4)* %88, i64 %773
  %775 = getelementptr float, float addrspace(4)* %774, i64 %765
  %776 = addrspacecast float addrspace(4)* %775 to float addrspace(1)*
  %777 = load float, float addrspace(1)* %776, align 4
  %778 = fmul reassoc nsz arcp contract float %777, %4, !spirv.Decorations !843
  %779 = fadd reassoc nsz arcp contract float %768, %778, !spirv.Decorations !843
  %780 = getelementptr inbounds float, float addrspace(4)* %87, i64 %767
  %781 = addrspacecast float addrspace(4)* %780 to float addrspace(1)*
  store float %779, float addrspace(1)* %781, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %._crit_edge70.2.3, %772, %769
  %782 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %52
  %783 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %53
  %.idx = select i1 %42, i64 %54, i64 0
  %784 = getelementptr float, float addrspace(4)* %88, i64 %.idx
  %785 = getelementptr inbounds float, float addrspace(4)* %87, i64 %55
  %786 = add i32 %86, %14
  %787 = icmp slt i32 %786, %8
  br i1 %787, label %.preheader2.preheader, label %._crit_edge72

._crit_edge72:                                    ; preds = %.preheader1.3, %13
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %21 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %22 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %23 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %24 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %24)
  %25 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %25)
  %26 = mul i32 %17, %15
  %27 = zext i16 %localIdX to i32
  %28 = add i32 %26, %27
  %29 = shl i32 %28, 2
  %30 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %18, %16
  %33 = zext i16 %localIdY to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 4
  %36 = zext i32 %19 to i64
  %37 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %37)
  %38 = mul nsw i64 %36, %9, !spirv.Decorations !836
  %39 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %23, i64 %38
  %40 = mul nsw i64 %36, %10, !spirv.Decorations !836
  %41 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %22, i64 %40
  %42 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !843
  %43 = mul nsw i64 %36, %11, !spirv.Decorations !836
  %44 = select i1 %42, i64 %43, i64 0
  %45 = getelementptr inbounds float, float addrspace(4)* %21, i64 %44
  %46 = mul nsw i64 %36, %12, !spirv.Decorations !836
  %47 = getelementptr inbounds float, float addrspace(4)* %20, i64 %46
  %48 = icmp slt i32 %19, %8
  br i1 %48, label %.lr.ph, label %._crit_edge72

.lr.ph:                                           ; preds = %13
  %49 = icmp sgt i32 %const_reg_dword2, 0
  %50 = zext i32 %14 to i64
  %51 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %51)
  %52 = mul nsw i64 %50, %9, !spirv.Decorations !836
  %53 = mul nsw i64 %50, %10, !spirv.Decorations !836
  %54 = mul nsw i64 %50, %11
  %55 = mul nsw i64 %50, %12, !spirv.Decorations !836
  %56 = icmp slt i32 %35, %const_reg_dword1
  %57 = icmp slt i32 %29, %const_reg_dword
  %58 = and i1 %57, %56
  %59 = or i32 %29, 1
  %60 = icmp slt i32 %59, %const_reg_dword
  %61 = and i1 %60, %56
  %62 = or i32 %29, 2
  %63 = icmp slt i32 %62, %const_reg_dword
  %64 = and i1 %63, %56
  %65 = or i32 %29, 3
  %66 = icmp slt i32 %65, %const_reg_dword
  %67 = and i1 %66, %56
  %68 = or i32 %35, 1
  %69 = icmp slt i32 %68, %const_reg_dword1
  %70 = and i1 %57, %69
  %71 = and i1 %60, %69
  %72 = and i1 %63, %69
  %73 = and i1 %66, %69
  %74 = or i32 %35, 2
  %75 = icmp slt i32 %74, %const_reg_dword1
  %76 = and i1 %57, %75
  %77 = and i1 %60, %75
  %78 = and i1 %63, %75
  %79 = and i1 %66, %75
  %80 = or i32 %35, 3
  %81 = icmp slt i32 %80, %const_reg_dword1
  %82 = and i1 %57, %81
  %83 = and i1 %60, %81
  %84 = and i1 %63, %81
  %85 = and i1 %66, %81
  %86 = or i32 %35, 4
  %87 = icmp slt i32 %86, %const_reg_dword1
  %88 = and i1 %57, %87
  %89 = and i1 %60, %87
  %90 = and i1 %63, %87
  %91 = and i1 %66, %87
  %92 = or i32 %35, 5
  %93 = icmp slt i32 %92, %const_reg_dword1
  %94 = and i1 %57, %93
  %95 = and i1 %60, %93
  %96 = and i1 %63, %93
  %97 = and i1 %66, %93
  %98 = or i32 %35, 6
  %99 = icmp slt i32 %98, %const_reg_dword1
  %100 = and i1 %57, %99
  %101 = and i1 %60, %99
  %102 = and i1 %63, %99
  %103 = and i1 %66, %99
  %104 = or i32 %35, 7
  %105 = icmp slt i32 %104, %const_reg_dword1
  %106 = and i1 %57, %105
  %107 = and i1 %60, %105
  %108 = and i1 %63, %105
  %109 = and i1 %66, %105
  %110 = or i32 %35, 8
  %111 = icmp slt i32 %110, %const_reg_dword1
  %112 = and i1 %57, %111
  %113 = and i1 %60, %111
  %114 = and i1 %63, %111
  %115 = and i1 %66, %111
  %116 = or i32 %35, 9
  %117 = icmp slt i32 %116, %const_reg_dword1
  %118 = and i1 %57, %117
  %119 = and i1 %60, %117
  %120 = and i1 %63, %117
  %121 = and i1 %66, %117
  %122 = or i32 %35, 10
  %123 = icmp slt i32 %122, %const_reg_dword1
  %124 = and i1 %57, %123
  %125 = and i1 %60, %123
  %126 = and i1 %63, %123
  %127 = and i1 %66, %123
  %128 = or i32 %35, 11
  %129 = icmp slt i32 %128, %const_reg_dword1
  %130 = and i1 %57, %129
  %131 = and i1 %60, %129
  %132 = and i1 %63, %129
  %133 = and i1 %66, %129
  %134 = or i32 %35, 12
  %135 = icmp slt i32 %134, %const_reg_dword1
  %136 = and i1 %57, %135
  %137 = and i1 %60, %135
  %138 = and i1 %63, %135
  %139 = and i1 %66, %135
  %140 = or i32 %35, 13
  %141 = icmp slt i32 %140, %const_reg_dword1
  %142 = and i1 %57, %141
  %143 = and i1 %60, %141
  %144 = and i1 %63, %141
  %145 = and i1 %66, %141
  %146 = or i32 %35, 14
  %147 = icmp slt i32 %146, %const_reg_dword1
  %148 = and i1 %57, %147
  %149 = and i1 %60, %147
  %150 = and i1 %63, %147
  %151 = and i1 %66, %147
  %152 = or i32 %35, 15
  %153 = icmp slt i32 %152, %const_reg_dword1
  %154 = and i1 %57, %153
  %155 = and i1 %60, %153
  %156 = and i1 %63, %153
  %157 = and i1 %66, %153
  br label %.preheader2.preheader

.preheader2.preheader:                            ; preds = %.lr.ph, %.preheader1.15
  %158 = phi i32 [ %19, %.lr.ph ], [ %2794, %.preheader1.15 ]
  %159 = phi float addrspace(4)* [ %47, %.lr.ph ], [ %2793, %.preheader1.15 ]
  %160 = phi float addrspace(4)* [ %45, %.lr.ph ], [ %2792, %.preheader1.15 ]
  %161 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %41, %.lr.ph ], [ %2791, %.preheader1.15 ]
  %162 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %39, %.lr.ph ], [ %2790, %.preheader1.15 ]
  br i1 %49, label %.preheader.preheader, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.preheader.15, %.preheader2.preheader
  %.sroa.254.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.254.2, %.preheader.15 ]
  %.sroa.250.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.250.2, %.preheader.15 ]
  %.sroa.246.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.246.2, %.preheader.15 ]
  %.sroa.242.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.242.2, %.preheader.15 ]
  %.sroa.238.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.238.2, %.preheader.15 ]
  %.sroa.234.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.234.2, %.preheader.15 ]
  %.sroa.230.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.230.2, %.preheader.15 ]
  %.sroa.226.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.226.2, %.preheader.15 ]
  %.sroa.222.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.222.2, %.preheader.15 ]
  %.sroa.218.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.218.2, %.preheader.15 ]
  %.sroa.214.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.214.2, %.preheader.15 ]
  %.sroa.210.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.210.2, %.preheader.15 ]
  %.sroa.206.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.206.2, %.preheader.15 ]
  %.sroa.202.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.202.2, %.preheader.15 ]
  %.sroa.198.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.198.2, %.preheader.15 ]
  %.sroa.194.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.194.2, %.preheader.15 ]
  %.sroa.190.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.190.2, %.preheader.15 ]
  %.sroa.186.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.186.2, %.preheader.15 ]
  %.sroa.182.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.182.2, %.preheader.15 ]
  %.sroa.178.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.178.2, %.preheader.15 ]
  %.sroa.174.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.174.2, %.preheader.15 ]
  %.sroa.170.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.170.2, %.preheader.15 ]
  %.sroa.166.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.166.2, %.preheader.15 ]
  %.sroa.162.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.162.2, %.preheader.15 ]
  %.sroa.158.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.158.2, %.preheader.15 ]
  %.sroa.154.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.154.2, %.preheader.15 ]
  %.sroa.150.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.150.2, %.preheader.15 ]
  %.sroa.146.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.146.2, %.preheader.15 ]
  %.sroa.142.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.142.2, %.preheader.15 ]
  %.sroa.138.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.138.2, %.preheader.15 ]
  %.sroa.134.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.134.2, %.preheader.15 ]
  %.sroa.130.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.130.2, %.preheader.15 ]
  %.sroa.126.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.126.2, %.preheader.15 ]
  %.sroa.122.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.122.2, %.preheader.15 ]
  %.sroa.118.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.118.2, %.preheader.15 ]
  %.sroa.114.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.114.2, %.preheader.15 ]
  %.sroa.110.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.110.2, %.preheader.15 ]
  %.sroa.106.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.106.2, %.preheader.15 ]
  %.sroa.102.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.102.2, %.preheader.15 ]
  %.sroa.98.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.98.2, %.preheader.15 ]
  %.sroa.94.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.94.2, %.preheader.15 ]
  %.sroa.90.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.90.2, %.preheader.15 ]
  %.sroa.86.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.86.2, %.preheader.15 ]
  %.sroa.82.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.82.2, %.preheader.15 ]
  %.sroa.78.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.78.2, %.preheader.15 ]
  %.sroa.74.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.74.2, %.preheader.15 ]
  %.sroa.70.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.70.2, %.preheader.15 ]
  %.sroa.66.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.66.2, %.preheader.15 ]
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.62.2, %.preheader.15 ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.58.2, %.preheader.15 ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.54.2, %.preheader.15 ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.50.2, %.preheader.15 ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.46.2, %.preheader.15 ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.42.2, %.preheader.15 ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.38.2, %.preheader.15 ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.34.2, %.preheader.15 ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.30.2, %.preheader.15 ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.26.2, %.preheader.15 ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.22.2, %.preheader.15 ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.18.2, %.preheader.15 ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.14.2, %.preheader.15 ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.10.2, %.preheader.15 ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.6.2, %.preheader.15 ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.0.2, %.preheader.15 ]
  br i1 %58, label %1574, label %._crit_edge70

.preheader.preheader:                             ; preds = %.preheader2.preheader, %.preheader.15
  %.sroa.254.1 = phi float [ %.sroa.254.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.250.1 = phi float [ %.sroa.250.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.246.1 = phi float [ %.sroa.246.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.242.1 = phi float [ %.sroa.242.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.238.1 = phi float [ %.sroa.238.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.234.1 = phi float [ %.sroa.234.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.230.1 = phi float [ %.sroa.230.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.226.1 = phi float [ %.sroa.226.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.222.1 = phi float [ %.sroa.222.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.218.1 = phi float [ %.sroa.218.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.214.1 = phi float [ %.sroa.214.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.210.1 = phi float [ %.sroa.210.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.206.1 = phi float [ %.sroa.206.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.202.1 = phi float [ %.sroa.202.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.198.1 = phi float [ %.sroa.198.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.194.1 = phi float [ %.sroa.194.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.190.1 = phi float [ %.sroa.190.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.186.1 = phi float [ %.sroa.186.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.182.1 = phi float [ %.sroa.182.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.178.1 = phi float [ %.sroa.178.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.174.1 = phi float [ %.sroa.174.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.170.1 = phi float [ %.sroa.170.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.166.1 = phi float [ %.sroa.166.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.162.1 = phi float [ %.sroa.162.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.158.1 = phi float [ %.sroa.158.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.154.1 = phi float [ %.sroa.154.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.150.1 = phi float [ %.sroa.150.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.146.1 = phi float [ %.sroa.146.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.142.1 = phi float [ %.sroa.142.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.138.1 = phi float [ %.sroa.138.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.134.1 = phi float [ %.sroa.134.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.130.1 = phi float [ %.sroa.130.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.126.1 = phi float [ %.sroa.126.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.122.1 = phi float [ %.sroa.122.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.118.1 = phi float [ %.sroa.118.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.114.1 = phi float [ %.sroa.114.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.110.1 = phi float [ %.sroa.110.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.106.1 = phi float [ %.sroa.106.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.102.1 = phi float [ %.sroa.102.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.98.1 = phi float [ %.sroa.98.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.94.1 = phi float [ %.sroa.94.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.90.1 = phi float [ %.sroa.90.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.86.1 = phi float [ %.sroa.86.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.82.1 = phi float [ %.sroa.82.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.78.1 = phi float [ %.sroa.78.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.74.1 = phi float [ %.sroa.74.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.70.1 = phi float [ %.sroa.70.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.66.1 = phi float [ %.sroa.66.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.62.1 = phi float [ %.sroa.62.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.58.1 = phi float [ %.sroa.58.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.54.1 = phi float [ %.sroa.54.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.50.1 = phi float [ %.sroa.50.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.46.1 = phi float [ %.sroa.46.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.42.1 = phi float [ %.sroa.42.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.38.1 = phi float [ %.sroa.38.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.34.1 = phi float [ %.sroa.34.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.30.1 = phi float [ %.sroa.30.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.26.1 = phi float [ %.sroa.26.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.22.1 = phi float [ %.sroa.22.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.18.1 = phi float [ %.sroa.18.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.14.1 = phi float [ %.sroa.14.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.10.1 = phi float [ %.sroa.10.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.6.1 = phi float [ %.sroa.6.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.0.1 = phi float [ %.sroa.0.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %163 = phi i32 [ %1572, %.preheader.15 ], [ 0, %.preheader2.preheader ]
  br i1 %58, label %164, label %._crit_edge

164:                                              ; preds = %.preheader.preheader
  %.sroa.256.0.insert.ext = zext i32 %163 to i64
  %165 = sext i32 %29 to i64
  %166 = mul nsw i64 %165, %const_reg_qword3, !spirv.Decorations !836
  %167 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %166, i32 0
  %168 = getelementptr i16, i16 addrspace(4)* %167, i64 %.sroa.256.0.insert.ext
  %169 = addrspacecast i16 addrspace(4)* %168 to i16 addrspace(1)*
  %170 = load i16, i16 addrspace(1)* %169, align 2
  %171 = sext i32 %35 to i64
  %172 = mul nsw i64 %171, %const_reg_qword5, !spirv.Decorations !836
  %173 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %172
  %174 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %173, i64 %.sroa.256.0.insert.ext
  %175 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %174, i64 0, i32 0
  %176 = addrspacecast i16 addrspace(4)* %175 to i16 addrspace(1)*
  %177 = load i16, i16 addrspace(1)* %176, align 2
  %178 = zext i16 %170 to i32
  %179 = shl nuw i32 %178, 16, !spirv.Decorations !838
  %180 = bitcast i32 %179 to float
  %181 = zext i16 %177 to i32
  %182 = shl nuw i32 %181, 16, !spirv.Decorations !838
  %183 = bitcast i32 %182 to float
  %184 = fmul reassoc nsz arcp contract float %180, %183, !spirv.Decorations !843
  %185 = fadd reassoc nsz arcp contract float %184, %.sroa.0.1, !spirv.Decorations !843
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %164
  %.sroa.0.2 = phi float [ %185, %164 ], [ %.sroa.0.1, %.preheader.preheader ]
  br i1 %61, label %186, label %._crit_edge.1

186:                                              ; preds = %._crit_edge
  %.sroa.256.0.insert.ext588 = zext i32 %163 to i64
  %187 = sext i32 %59 to i64
  %188 = mul nsw i64 %187, %const_reg_qword3, !spirv.Decorations !836
  %189 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %188, i32 0
  %190 = getelementptr i16, i16 addrspace(4)* %189, i64 %.sroa.256.0.insert.ext588
  %191 = addrspacecast i16 addrspace(4)* %190 to i16 addrspace(1)*
  %192 = load i16, i16 addrspace(1)* %191, align 2
  %193 = sext i32 %35 to i64
  %194 = mul nsw i64 %193, %const_reg_qword5, !spirv.Decorations !836
  %195 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %194
  %196 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %195, i64 %.sroa.256.0.insert.ext588
  %197 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %196, i64 0, i32 0
  %198 = addrspacecast i16 addrspace(4)* %197 to i16 addrspace(1)*
  %199 = load i16, i16 addrspace(1)* %198, align 2
  %200 = zext i16 %192 to i32
  %201 = shl nuw i32 %200, 16, !spirv.Decorations !838
  %202 = bitcast i32 %201 to float
  %203 = zext i16 %199 to i32
  %204 = shl nuw i32 %203, 16, !spirv.Decorations !838
  %205 = bitcast i32 %204 to float
  %206 = fmul reassoc nsz arcp contract float %202, %205, !spirv.Decorations !843
  %207 = fadd reassoc nsz arcp contract float %206, %.sroa.66.1, !spirv.Decorations !843
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %186
  %.sroa.66.2 = phi float [ %207, %186 ], [ %.sroa.66.1, %._crit_edge ]
  br i1 %64, label %208, label %._crit_edge.2

208:                                              ; preds = %._crit_edge.1
  %.sroa.256.0.insert.ext593 = zext i32 %163 to i64
  %209 = sext i32 %62 to i64
  %210 = mul nsw i64 %209, %const_reg_qword3, !spirv.Decorations !836
  %211 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %210, i32 0
  %212 = getelementptr i16, i16 addrspace(4)* %211, i64 %.sroa.256.0.insert.ext593
  %213 = addrspacecast i16 addrspace(4)* %212 to i16 addrspace(1)*
  %214 = load i16, i16 addrspace(1)* %213, align 2
  %215 = sext i32 %35 to i64
  %216 = mul nsw i64 %215, %const_reg_qword5, !spirv.Decorations !836
  %217 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %216
  %218 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %217, i64 %.sroa.256.0.insert.ext593
  %219 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %218, i64 0, i32 0
  %220 = addrspacecast i16 addrspace(4)* %219 to i16 addrspace(1)*
  %221 = load i16, i16 addrspace(1)* %220, align 2
  %222 = zext i16 %214 to i32
  %223 = shl nuw i32 %222, 16, !spirv.Decorations !838
  %224 = bitcast i32 %223 to float
  %225 = zext i16 %221 to i32
  %226 = shl nuw i32 %225, 16, !spirv.Decorations !838
  %227 = bitcast i32 %226 to float
  %228 = fmul reassoc nsz arcp contract float %224, %227, !spirv.Decorations !843
  %229 = fadd reassoc nsz arcp contract float %228, %.sroa.130.1, !spirv.Decorations !843
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %208
  %.sroa.130.2 = phi float [ %229, %208 ], [ %.sroa.130.1, %._crit_edge.1 ]
  br i1 %67, label %230, label %.preheader

230:                                              ; preds = %._crit_edge.2
  %.sroa.256.0.insert.ext598 = zext i32 %163 to i64
  %231 = sext i32 %65 to i64
  %232 = mul nsw i64 %231, %const_reg_qword3, !spirv.Decorations !836
  %233 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %232, i32 0
  %234 = getelementptr i16, i16 addrspace(4)* %233, i64 %.sroa.256.0.insert.ext598
  %235 = addrspacecast i16 addrspace(4)* %234 to i16 addrspace(1)*
  %236 = load i16, i16 addrspace(1)* %235, align 2
  %237 = sext i32 %35 to i64
  %238 = mul nsw i64 %237, %const_reg_qword5, !spirv.Decorations !836
  %239 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %238
  %240 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %239, i64 %.sroa.256.0.insert.ext598
  %241 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %240, i64 0, i32 0
  %242 = addrspacecast i16 addrspace(4)* %241 to i16 addrspace(1)*
  %243 = load i16, i16 addrspace(1)* %242, align 2
  %244 = zext i16 %236 to i32
  %245 = shl nuw i32 %244, 16, !spirv.Decorations !838
  %246 = bitcast i32 %245 to float
  %247 = zext i16 %243 to i32
  %248 = shl nuw i32 %247, 16, !spirv.Decorations !838
  %249 = bitcast i32 %248 to float
  %250 = fmul reassoc nsz arcp contract float %246, %249, !spirv.Decorations !843
  %251 = fadd reassoc nsz arcp contract float %250, %.sroa.194.1, !spirv.Decorations !843
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %230
  %.sroa.194.2 = phi float [ %251, %230 ], [ %.sroa.194.1, %._crit_edge.2 ]
  br i1 %70, label %252, label %._crit_edge.173

252:                                              ; preds = %.preheader
  %.sroa.256.0.insert.ext603 = zext i32 %163 to i64
  %253 = sext i32 %29 to i64
  %254 = mul nsw i64 %253, %const_reg_qword3, !spirv.Decorations !836
  %255 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %254, i32 0
  %256 = getelementptr i16, i16 addrspace(4)* %255, i64 %.sroa.256.0.insert.ext603
  %257 = addrspacecast i16 addrspace(4)* %256 to i16 addrspace(1)*
  %258 = load i16, i16 addrspace(1)* %257, align 2
  %259 = sext i32 %68 to i64
  %260 = mul nsw i64 %259, %const_reg_qword5, !spirv.Decorations !836
  %261 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %260
  %262 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %261, i64 %.sroa.256.0.insert.ext603
  %263 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %262, i64 0, i32 0
  %264 = addrspacecast i16 addrspace(4)* %263 to i16 addrspace(1)*
  %265 = load i16, i16 addrspace(1)* %264, align 2
  %266 = zext i16 %258 to i32
  %267 = shl nuw i32 %266, 16, !spirv.Decorations !838
  %268 = bitcast i32 %267 to float
  %269 = zext i16 %265 to i32
  %270 = shl nuw i32 %269, 16, !spirv.Decorations !838
  %271 = bitcast i32 %270 to float
  %272 = fmul reassoc nsz arcp contract float %268, %271, !spirv.Decorations !843
  %273 = fadd reassoc nsz arcp contract float %272, %.sroa.6.1, !spirv.Decorations !843
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %252
  %.sroa.6.2 = phi float [ %273, %252 ], [ %.sroa.6.1, %.preheader ]
  br i1 %71, label %274, label %._crit_edge.1.1

274:                                              ; preds = %._crit_edge.173
  %.sroa.256.0.insert.ext608 = zext i32 %163 to i64
  %275 = sext i32 %59 to i64
  %276 = mul nsw i64 %275, %const_reg_qword3, !spirv.Decorations !836
  %277 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %276, i32 0
  %278 = getelementptr i16, i16 addrspace(4)* %277, i64 %.sroa.256.0.insert.ext608
  %279 = addrspacecast i16 addrspace(4)* %278 to i16 addrspace(1)*
  %280 = load i16, i16 addrspace(1)* %279, align 2
  %281 = sext i32 %68 to i64
  %282 = mul nsw i64 %281, %const_reg_qword5, !spirv.Decorations !836
  %283 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %282
  %284 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %283, i64 %.sroa.256.0.insert.ext608
  %285 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %284, i64 0, i32 0
  %286 = addrspacecast i16 addrspace(4)* %285 to i16 addrspace(1)*
  %287 = load i16, i16 addrspace(1)* %286, align 2
  %288 = zext i16 %280 to i32
  %289 = shl nuw i32 %288, 16, !spirv.Decorations !838
  %290 = bitcast i32 %289 to float
  %291 = zext i16 %287 to i32
  %292 = shl nuw i32 %291, 16, !spirv.Decorations !838
  %293 = bitcast i32 %292 to float
  %294 = fmul reassoc nsz arcp contract float %290, %293, !spirv.Decorations !843
  %295 = fadd reassoc nsz arcp contract float %294, %.sroa.70.1, !spirv.Decorations !843
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %274
  %.sroa.70.2 = phi float [ %295, %274 ], [ %.sroa.70.1, %._crit_edge.173 ]
  br i1 %72, label %296, label %._crit_edge.2.1

296:                                              ; preds = %._crit_edge.1.1
  %.sroa.256.0.insert.ext613 = zext i32 %163 to i64
  %297 = sext i32 %62 to i64
  %298 = mul nsw i64 %297, %const_reg_qword3, !spirv.Decorations !836
  %299 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %298, i32 0
  %300 = getelementptr i16, i16 addrspace(4)* %299, i64 %.sroa.256.0.insert.ext613
  %301 = addrspacecast i16 addrspace(4)* %300 to i16 addrspace(1)*
  %302 = load i16, i16 addrspace(1)* %301, align 2
  %303 = sext i32 %68 to i64
  %304 = mul nsw i64 %303, %const_reg_qword5, !spirv.Decorations !836
  %305 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %304
  %306 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %305, i64 %.sroa.256.0.insert.ext613
  %307 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %306, i64 0, i32 0
  %308 = addrspacecast i16 addrspace(4)* %307 to i16 addrspace(1)*
  %309 = load i16, i16 addrspace(1)* %308, align 2
  %310 = zext i16 %302 to i32
  %311 = shl nuw i32 %310, 16, !spirv.Decorations !838
  %312 = bitcast i32 %311 to float
  %313 = zext i16 %309 to i32
  %314 = shl nuw i32 %313, 16, !spirv.Decorations !838
  %315 = bitcast i32 %314 to float
  %316 = fmul reassoc nsz arcp contract float %312, %315, !spirv.Decorations !843
  %317 = fadd reassoc nsz arcp contract float %316, %.sroa.134.1, !spirv.Decorations !843
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %296
  %.sroa.134.2 = phi float [ %317, %296 ], [ %.sroa.134.1, %._crit_edge.1.1 ]
  br i1 %73, label %318, label %.preheader.1

318:                                              ; preds = %._crit_edge.2.1
  %.sroa.256.0.insert.ext618 = zext i32 %163 to i64
  %319 = sext i32 %65 to i64
  %320 = mul nsw i64 %319, %const_reg_qword3, !spirv.Decorations !836
  %321 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %320, i32 0
  %322 = getelementptr i16, i16 addrspace(4)* %321, i64 %.sroa.256.0.insert.ext618
  %323 = addrspacecast i16 addrspace(4)* %322 to i16 addrspace(1)*
  %324 = load i16, i16 addrspace(1)* %323, align 2
  %325 = sext i32 %68 to i64
  %326 = mul nsw i64 %325, %const_reg_qword5, !spirv.Decorations !836
  %327 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %326
  %328 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %327, i64 %.sroa.256.0.insert.ext618
  %329 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %328, i64 0, i32 0
  %330 = addrspacecast i16 addrspace(4)* %329 to i16 addrspace(1)*
  %331 = load i16, i16 addrspace(1)* %330, align 2
  %332 = zext i16 %324 to i32
  %333 = shl nuw i32 %332, 16, !spirv.Decorations !838
  %334 = bitcast i32 %333 to float
  %335 = zext i16 %331 to i32
  %336 = shl nuw i32 %335, 16, !spirv.Decorations !838
  %337 = bitcast i32 %336 to float
  %338 = fmul reassoc nsz arcp contract float %334, %337, !spirv.Decorations !843
  %339 = fadd reassoc nsz arcp contract float %338, %.sroa.198.1, !spirv.Decorations !843
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %318
  %.sroa.198.2 = phi float [ %339, %318 ], [ %.sroa.198.1, %._crit_edge.2.1 ]
  br i1 %76, label %340, label %._crit_edge.274

340:                                              ; preds = %.preheader.1
  %.sroa.256.0.insert.ext623 = zext i32 %163 to i64
  %341 = sext i32 %29 to i64
  %342 = mul nsw i64 %341, %const_reg_qword3, !spirv.Decorations !836
  %343 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %342, i32 0
  %344 = getelementptr i16, i16 addrspace(4)* %343, i64 %.sroa.256.0.insert.ext623
  %345 = addrspacecast i16 addrspace(4)* %344 to i16 addrspace(1)*
  %346 = load i16, i16 addrspace(1)* %345, align 2
  %347 = sext i32 %74 to i64
  %348 = mul nsw i64 %347, %const_reg_qword5, !spirv.Decorations !836
  %349 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %348
  %350 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %349, i64 %.sroa.256.0.insert.ext623
  %351 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %350, i64 0, i32 0
  %352 = addrspacecast i16 addrspace(4)* %351 to i16 addrspace(1)*
  %353 = load i16, i16 addrspace(1)* %352, align 2
  %354 = zext i16 %346 to i32
  %355 = shl nuw i32 %354, 16, !spirv.Decorations !838
  %356 = bitcast i32 %355 to float
  %357 = zext i16 %353 to i32
  %358 = shl nuw i32 %357, 16, !spirv.Decorations !838
  %359 = bitcast i32 %358 to float
  %360 = fmul reassoc nsz arcp contract float %356, %359, !spirv.Decorations !843
  %361 = fadd reassoc nsz arcp contract float %360, %.sroa.10.1, !spirv.Decorations !843
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %340
  %.sroa.10.2 = phi float [ %361, %340 ], [ %.sroa.10.1, %.preheader.1 ]
  br i1 %77, label %362, label %._crit_edge.1.2

362:                                              ; preds = %._crit_edge.274
  %.sroa.256.0.insert.ext628 = zext i32 %163 to i64
  %363 = sext i32 %59 to i64
  %364 = mul nsw i64 %363, %const_reg_qword3, !spirv.Decorations !836
  %365 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %364, i32 0
  %366 = getelementptr i16, i16 addrspace(4)* %365, i64 %.sroa.256.0.insert.ext628
  %367 = addrspacecast i16 addrspace(4)* %366 to i16 addrspace(1)*
  %368 = load i16, i16 addrspace(1)* %367, align 2
  %369 = sext i32 %74 to i64
  %370 = mul nsw i64 %369, %const_reg_qword5, !spirv.Decorations !836
  %371 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %370
  %372 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %371, i64 %.sroa.256.0.insert.ext628
  %373 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %372, i64 0, i32 0
  %374 = addrspacecast i16 addrspace(4)* %373 to i16 addrspace(1)*
  %375 = load i16, i16 addrspace(1)* %374, align 2
  %376 = zext i16 %368 to i32
  %377 = shl nuw i32 %376, 16, !spirv.Decorations !838
  %378 = bitcast i32 %377 to float
  %379 = zext i16 %375 to i32
  %380 = shl nuw i32 %379, 16, !spirv.Decorations !838
  %381 = bitcast i32 %380 to float
  %382 = fmul reassoc nsz arcp contract float %378, %381, !spirv.Decorations !843
  %383 = fadd reassoc nsz arcp contract float %382, %.sroa.74.1, !spirv.Decorations !843
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %362
  %.sroa.74.2 = phi float [ %383, %362 ], [ %.sroa.74.1, %._crit_edge.274 ]
  br i1 %78, label %384, label %._crit_edge.2.2

384:                                              ; preds = %._crit_edge.1.2
  %.sroa.256.0.insert.ext633 = zext i32 %163 to i64
  %385 = sext i32 %62 to i64
  %386 = mul nsw i64 %385, %const_reg_qword3, !spirv.Decorations !836
  %387 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %386, i32 0
  %388 = getelementptr i16, i16 addrspace(4)* %387, i64 %.sroa.256.0.insert.ext633
  %389 = addrspacecast i16 addrspace(4)* %388 to i16 addrspace(1)*
  %390 = load i16, i16 addrspace(1)* %389, align 2
  %391 = sext i32 %74 to i64
  %392 = mul nsw i64 %391, %const_reg_qword5, !spirv.Decorations !836
  %393 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %392
  %394 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %393, i64 %.sroa.256.0.insert.ext633
  %395 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %394, i64 0, i32 0
  %396 = addrspacecast i16 addrspace(4)* %395 to i16 addrspace(1)*
  %397 = load i16, i16 addrspace(1)* %396, align 2
  %398 = zext i16 %390 to i32
  %399 = shl nuw i32 %398, 16, !spirv.Decorations !838
  %400 = bitcast i32 %399 to float
  %401 = zext i16 %397 to i32
  %402 = shl nuw i32 %401, 16, !spirv.Decorations !838
  %403 = bitcast i32 %402 to float
  %404 = fmul reassoc nsz arcp contract float %400, %403, !spirv.Decorations !843
  %405 = fadd reassoc nsz arcp contract float %404, %.sroa.138.1, !spirv.Decorations !843
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %384
  %.sroa.138.2 = phi float [ %405, %384 ], [ %.sroa.138.1, %._crit_edge.1.2 ]
  br i1 %79, label %406, label %.preheader.2

406:                                              ; preds = %._crit_edge.2.2
  %.sroa.256.0.insert.ext638 = zext i32 %163 to i64
  %407 = sext i32 %65 to i64
  %408 = mul nsw i64 %407, %const_reg_qword3, !spirv.Decorations !836
  %409 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %408, i32 0
  %410 = getelementptr i16, i16 addrspace(4)* %409, i64 %.sroa.256.0.insert.ext638
  %411 = addrspacecast i16 addrspace(4)* %410 to i16 addrspace(1)*
  %412 = load i16, i16 addrspace(1)* %411, align 2
  %413 = sext i32 %74 to i64
  %414 = mul nsw i64 %413, %const_reg_qword5, !spirv.Decorations !836
  %415 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %414
  %416 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %415, i64 %.sroa.256.0.insert.ext638
  %417 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %416, i64 0, i32 0
  %418 = addrspacecast i16 addrspace(4)* %417 to i16 addrspace(1)*
  %419 = load i16, i16 addrspace(1)* %418, align 2
  %420 = zext i16 %412 to i32
  %421 = shl nuw i32 %420, 16, !spirv.Decorations !838
  %422 = bitcast i32 %421 to float
  %423 = zext i16 %419 to i32
  %424 = shl nuw i32 %423, 16, !spirv.Decorations !838
  %425 = bitcast i32 %424 to float
  %426 = fmul reassoc nsz arcp contract float %422, %425, !spirv.Decorations !843
  %427 = fadd reassoc nsz arcp contract float %426, %.sroa.202.1, !spirv.Decorations !843
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %406
  %.sroa.202.2 = phi float [ %427, %406 ], [ %.sroa.202.1, %._crit_edge.2.2 ]
  br i1 %82, label %428, label %._crit_edge.375

428:                                              ; preds = %.preheader.2
  %.sroa.256.0.insert.ext643 = zext i32 %163 to i64
  %429 = sext i32 %29 to i64
  %430 = mul nsw i64 %429, %const_reg_qword3, !spirv.Decorations !836
  %431 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %430, i32 0
  %432 = getelementptr i16, i16 addrspace(4)* %431, i64 %.sroa.256.0.insert.ext643
  %433 = addrspacecast i16 addrspace(4)* %432 to i16 addrspace(1)*
  %434 = load i16, i16 addrspace(1)* %433, align 2
  %435 = sext i32 %80 to i64
  %436 = mul nsw i64 %435, %const_reg_qword5, !spirv.Decorations !836
  %437 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %436
  %438 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %437, i64 %.sroa.256.0.insert.ext643
  %439 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %438, i64 0, i32 0
  %440 = addrspacecast i16 addrspace(4)* %439 to i16 addrspace(1)*
  %441 = load i16, i16 addrspace(1)* %440, align 2
  %442 = zext i16 %434 to i32
  %443 = shl nuw i32 %442, 16, !spirv.Decorations !838
  %444 = bitcast i32 %443 to float
  %445 = zext i16 %441 to i32
  %446 = shl nuw i32 %445, 16, !spirv.Decorations !838
  %447 = bitcast i32 %446 to float
  %448 = fmul reassoc nsz arcp contract float %444, %447, !spirv.Decorations !843
  %449 = fadd reassoc nsz arcp contract float %448, %.sroa.14.1, !spirv.Decorations !843
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %428
  %.sroa.14.2 = phi float [ %449, %428 ], [ %.sroa.14.1, %.preheader.2 ]
  br i1 %83, label %450, label %._crit_edge.1.3

450:                                              ; preds = %._crit_edge.375
  %.sroa.256.0.insert.ext648 = zext i32 %163 to i64
  %451 = sext i32 %59 to i64
  %452 = mul nsw i64 %451, %const_reg_qword3, !spirv.Decorations !836
  %453 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %452, i32 0
  %454 = getelementptr i16, i16 addrspace(4)* %453, i64 %.sroa.256.0.insert.ext648
  %455 = addrspacecast i16 addrspace(4)* %454 to i16 addrspace(1)*
  %456 = load i16, i16 addrspace(1)* %455, align 2
  %457 = sext i32 %80 to i64
  %458 = mul nsw i64 %457, %const_reg_qword5, !spirv.Decorations !836
  %459 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %458
  %460 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %459, i64 %.sroa.256.0.insert.ext648
  %461 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %460, i64 0, i32 0
  %462 = addrspacecast i16 addrspace(4)* %461 to i16 addrspace(1)*
  %463 = load i16, i16 addrspace(1)* %462, align 2
  %464 = zext i16 %456 to i32
  %465 = shl nuw i32 %464, 16, !spirv.Decorations !838
  %466 = bitcast i32 %465 to float
  %467 = zext i16 %463 to i32
  %468 = shl nuw i32 %467, 16, !spirv.Decorations !838
  %469 = bitcast i32 %468 to float
  %470 = fmul reassoc nsz arcp contract float %466, %469, !spirv.Decorations !843
  %471 = fadd reassoc nsz arcp contract float %470, %.sroa.78.1, !spirv.Decorations !843
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %450
  %.sroa.78.2 = phi float [ %471, %450 ], [ %.sroa.78.1, %._crit_edge.375 ]
  br i1 %84, label %472, label %._crit_edge.2.3

472:                                              ; preds = %._crit_edge.1.3
  %.sroa.256.0.insert.ext653 = zext i32 %163 to i64
  %473 = sext i32 %62 to i64
  %474 = mul nsw i64 %473, %const_reg_qword3, !spirv.Decorations !836
  %475 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %474, i32 0
  %476 = getelementptr i16, i16 addrspace(4)* %475, i64 %.sroa.256.0.insert.ext653
  %477 = addrspacecast i16 addrspace(4)* %476 to i16 addrspace(1)*
  %478 = load i16, i16 addrspace(1)* %477, align 2
  %479 = sext i32 %80 to i64
  %480 = mul nsw i64 %479, %const_reg_qword5, !spirv.Decorations !836
  %481 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %480
  %482 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %481, i64 %.sroa.256.0.insert.ext653
  %483 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %482, i64 0, i32 0
  %484 = addrspacecast i16 addrspace(4)* %483 to i16 addrspace(1)*
  %485 = load i16, i16 addrspace(1)* %484, align 2
  %486 = zext i16 %478 to i32
  %487 = shl nuw i32 %486, 16, !spirv.Decorations !838
  %488 = bitcast i32 %487 to float
  %489 = zext i16 %485 to i32
  %490 = shl nuw i32 %489, 16, !spirv.Decorations !838
  %491 = bitcast i32 %490 to float
  %492 = fmul reassoc nsz arcp contract float %488, %491, !spirv.Decorations !843
  %493 = fadd reassoc nsz arcp contract float %492, %.sroa.142.1, !spirv.Decorations !843
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %472
  %.sroa.142.2 = phi float [ %493, %472 ], [ %.sroa.142.1, %._crit_edge.1.3 ]
  br i1 %85, label %494, label %.preheader.3

494:                                              ; preds = %._crit_edge.2.3
  %.sroa.256.0.insert.ext658 = zext i32 %163 to i64
  %495 = sext i32 %65 to i64
  %496 = mul nsw i64 %495, %const_reg_qword3, !spirv.Decorations !836
  %497 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %496, i32 0
  %498 = getelementptr i16, i16 addrspace(4)* %497, i64 %.sroa.256.0.insert.ext658
  %499 = addrspacecast i16 addrspace(4)* %498 to i16 addrspace(1)*
  %500 = load i16, i16 addrspace(1)* %499, align 2
  %501 = sext i32 %80 to i64
  %502 = mul nsw i64 %501, %const_reg_qword5, !spirv.Decorations !836
  %503 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %502
  %504 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %503, i64 %.sroa.256.0.insert.ext658
  %505 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %504, i64 0, i32 0
  %506 = addrspacecast i16 addrspace(4)* %505 to i16 addrspace(1)*
  %507 = load i16, i16 addrspace(1)* %506, align 2
  %508 = zext i16 %500 to i32
  %509 = shl nuw i32 %508, 16, !spirv.Decorations !838
  %510 = bitcast i32 %509 to float
  %511 = zext i16 %507 to i32
  %512 = shl nuw i32 %511, 16, !spirv.Decorations !838
  %513 = bitcast i32 %512 to float
  %514 = fmul reassoc nsz arcp contract float %510, %513, !spirv.Decorations !843
  %515 = fadd reassoc nsz arcp contract float %514, %.sroa.206.1, !spirv.Decorations !843
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %494
  %.sroa.206.2 = phi float [ %515, %494 ], [ %.sroa.206.1, %._crit_edge.2.3 ]
  br i1 %88, label %516, label %._crit_edge.4

516:                                              ; preds = %.preheader.3
  %.sroa.256.0.insert.ext663 = zext i32 %163 to i64
  %517 = sext i32 %29 to i64
  %518 = mul nsw i64 %517, %const_reg_qword3, !spirv.Decorations !836
  %519 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %518, i32 0
  %520 = getelementptr i16, i16 addrspace(4)* %519, i64 %.sroa.256.0.insert.ext663
  %521 = addrspacecast i16 addrspace(4)* %520 to i16 addrspace(1)*
  %522 = load i16, i16 addrspace(1)* %521, align 2
  %523 = sext i32 %86 to i64
  %524 = mul nsw i64 %523, %const_reg_qword5, !spirv.Decorations !836
  %525 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %524
  %526 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %525, i64 %.sroa.256.0.insert.ext663
  %527 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %526, i64 0, i32 0
  %528 = addrspacecast i16 addrspace(4)* %527 to i16 addrspace(1)*
  %529 = load i16, i16 addrspace(1)* %528, align 2
  %530 = zext i16 %522 to i32
  %531 = shl nuw i32 %530, 16, !spirv.Decorations !838
  %532 = bitcast i32 %531 to float
  %533 = zext i16 %529 to i32
  %534 = shl nuw i32 %533, 16, !spirv.Decorations !838
  %535 = bitcast i32 %534 to float
  %536 = fmul reassoc nsz arcp contract float %532, %535, !spirv.Decorations !843
  %537 = fadd reassoc nsz arcp contract float %536, %.sroa.18.1, !spirv.Decorations !843
  br label %._crit_edge.4

._crit_edge.4:                                    ; preds = %.preheader.3, %516
  %.sroa.18.2 = phi float [ %537, %516 ], [ %.sroa.18.1, %.preheader.3 ]
  br i1 %89, label %538, label %._crit_edge.1.4

538:                                              ; preds = %._crit_edge.4
  %.sroa.256.0.insert.ext668 = zext i32 %163 to i64
  %539 = sext i32 %59 to i64
  %540 = mul nsw i64 %539, %const_reg_qword3, !spirv.Decorations !836
  %541 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %540, i32 0
  %542 = getelementptr i16, i16 addrspace(4)* %541, i64 %.sroa.256.0.insert.ext668
  %543 = addrspacecast i16 addrspace(4)* %542 to i16 addrspace(1)*
  %544 = load i16, i16 addrspace(1)* %543, align 2
  %545 = sext i32 %86 to i64
  %546 = mul nsw i64 %545, %const_reg_qword5, !spirv.Decorations !836
  %547 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %546
  %548 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %547, i64 %.sroa.256.0.insert.ext668
  %549 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %548, i64 0, i32 0
  %550 = addrspacecast i16 addrspace(4)* %549 to i16 addrspace(1)*
  %551 = load i16, i16 addrspace(1)* %550, align 2
  %552 = zext i16 %544 to i32
  %553 = shl nuw i32 %552, 16, !spirv.Decorations !838
  %554 = bitcast i32 %553 to float
  %555 = zext i16 %551 to i32
  %556 = shl nuw i32 %555, 16, !spirv.Decorations !838
  %557 = bitcast i32 %556 to float
  %558 = fmul reassoc nsz arcp contract float %554, %557, !spirv.Decorations !843
  %559 = fadd reassoc nsz arcp contract float %558, %.sroa.82.1, !spirv.Decorations !843
  br label %._crit_edge.1.4

._crit_edge.1.4:                                  ; preds = %._crit_edge.4, %538
  %.sroa.82.2 = phi float [ %559, %538 ], [ %.sroa.82.1, %._crit_edge.4 ]
  br i1 %90, label %560, label %._crit_edge.2.4

560:                                              ; preds = %._crit_edge.1.4
  %.sroa.256.0.insert.ext673 = zext i32 %163 to i64
  %561 = sext i32 %62 to i64
  %562 = mul nsw i64 %561, %const_reg_qword3, !spirv.Decorations !836
  %563 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %562, i32 0
  %564 = getelementptr i16, i16 addrspace(4)* %563, i64 %.sroa.256.0.insert.ext673
  %565 = addrspacecast i16 addrspace(4)* %564 to i16 addrspace(1)*
  %566 = load i16, i16 addrspace(1)* %565, align 2
  %567 = sext i32 %86 to i64
  %568 = mul nsw i64 %567, %const_reg_qword5, !spirv.Decorations !836
  %569 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %568
  %570 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %569, i64 %.sroa.256.0.insert.ext673
  %571 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %570, i64 0, i32 0
  %572 = addrspacecast i16 addrspace(4)* %571 to i16 addrspace(1)*
  %573 = load i16, i16 addrspace(1)* %572, align 2
  %574 = zext i16 %566 to i32
  %575 = shl nuw i32 %574, 16, !spirv.Decorations !838
  %576 = bitcast i32 %575 to float
  %577 = zext i16 %573 to i32
  %578 = shl nuw i32 %577, 16, !spirv.Decorations !838
  %579 = bitcast i32 %578 to float
  %580 = fmul reassoc nsz arcp contract float %576, %579, !spirv.Decorations !843
  %581 = fadd reassoc nsz arcp contract float %580, %.sroa.146.1, !spirv.Decorations !843
  br label %._crit_edge.2.4

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4, %560
  %.sroa.146.2 = phi float [ %581, %560 ], [ %.sroa.146.1, %._crit_edge.1.4 ]
  br i1 %91, label %582, label %.preheader.4

582:                                              ; preds = %._crit_edge.2.4
  %.sroa.256.0.insert.ext678 = zext i32 %163 to i64
  %583 = sext i32 %65 to i64
  %584 = mul nsw i64 %583, %const_reg_qword3, !spirv.Decorations !836
  %585 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %584, i32 0
  %586 = getelementptr i16, i16 addrspace(4)* %585, i64 %.sroa.256.0.insert.ext678
  %587 = addrspacecast i16 addrspace(4)* %586 to i16 addrspace(1)*
  %588 = load i16, i16 addrspace(1)* %587, align 2
  %589 = sext i32 %86 to i64
  %590 = mul nsw i64 %589, %const_reg_qword5, !spirv.Decorations !836
  %591 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %590
  %592 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %591, i64 %.sroa.256.0.insert.ext678
  %593 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %592, i64 0, i32 0
  %594 = addrspacecast i16 addrspace(4)* %593 to i16 addrspace(1)*
  %595 = load i16, i16 addrspace(1)* %594, align 2
  %596 = zext i16 %588 to i32
  %597 = shl nuw i32 %596, 16, !spirv.Decorations !838
  %598 = bitcast i32 %597 to float
  %599 = zext i16 %595 to i32
  %600 = shl nuw i32 %599, 16, !spirv.Decorations !838
  %601 = bitcast i32 %600 to float
  %602 = fmul reassoc nsz arcp contract float %598, %601, !spirv.Decorations !843
  %603 = fadd reassoc nsz arcp contract float %602, %.sroa.210.1, !spirv.Decorations !843
  br label %.preheader.4

.preheader.4:                                     ; preds = %._crit_edge.2.4, %582
  %.sroa.210.2 = phi float [ %603, %582 ], [ %.sroa.210.1, %._crit_edge.2.4 ]
  br i1 %94, label %604, label %._crit_edge.5

604:                                              ; preds = %.preheader.4
  %.sroa.256.0.insert.ext683 = zext i32 %163 to i64
  %605 = sext i32 %29 to i64
  %606 = mul nsw i64 %605, %const_reg_qword3, !spirv.Decorations !836
  %607 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %606, i32 0
  %608 = getelementptr i16, i16 addrspace(4)* %607, i64 %.sroa.256.0.insert.ext683
  %609 = addrspacecast i16 addrspace(4)* %608 to i16 addrspace(1)*
  %610 = load i16, i16 addrspace(1)* %609, align 2
  %611 = sext i32 %92 to i64
  %612 = mul nsw i64 %611, %const_reg_qword5, !spirv.Decorations !836
  %613 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %612
  %614 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %613, i64 %.sroa.256.0.insert.ext683
  %615 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %614, i64 0, i32 0
  %616 = addrspacecast i16 addrspace(4)* %615 to i16 addrspace(1)*
  %617 = load i16, i16 addrspace(1)* %616, align 2
  %618 = zext i16 %610 to i32
  %619 = shl nuw i32 %618, 16, !spirv.Decorations !838
  %620 = bitcast i32 %619 to float
  %621 = zext i16 %617 to i32
  %622 = shl nuw i32 %621, 16, !spirv.Decorations !838
  %623 = bitcast i32 %622 to float
  %624 = fmul reassoc nsz arcp contract float %620, %623, !spirv.Decorations !843
  %625 = fadd reassoc nsz arcp contract float %624, %.sroa.22.1, !spirv.Decorations !843
  br label %._crit_edge.5

._crit_edge.5:                                    ; preds = %.preheader.4, %604
  %.sroa.22.2 = phi float [ %625, %604 ], [ %.sroa.22.1, %.preheader.4 ]
  br i1 %95, label %626, label %._crit_edge.1.5

626:                                              ; preds = %._crit_edge.5
  %.sroa.256.0.insert.ext688 = zext i32 %163 to i64
  %627 = sext i32 %59 to i64
  %628 = mul nsw i64 %627, %const_reg_qword3, !spirv.Decorations !836
  %629 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %628, i32 0
  %630 = getelementptr i16, i16 addrspace(4)* %629, i64 %.sroa.256.0.insert.ext688
  %631 = addrspacecast i16 addrspace(4)* %630 to i16 addrspace(1)*
  %632 = load i16, i16 addrspace(1)* %631, align 2
  %633 = sext i32 %92 to i64
  %634 = mul nsw i64 %633, %const_reg_qword5, !spirv.Decorations !836
  %635 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %634
  %636 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %635, i64 %.sroa.256.0.insert.ext688
  %637 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %636, i64 0, i32 0
  %638 = addrspacecast i16 addrspace(4)* %637 to i16 addrspace(1)*
  %639 = load i16, i16 addrspace(1)* %638, align 2
  %640 = zext i16 %632 to i32
  %641 = shl nuw i32 %640, 16, !spirv.Decorations !838
  %642 = bitcast i32 %641 to float
  %643 = zext i16 %639 to i32
  %644 = shl nuw i32 %643, 16, !spirv.Decorations !838
  %645 = bitcast i32 %644 to float
  %646 = fmul reassoc nsz arcp contract float %642, %645, !spirv.Decorations !843
  %647 = fadd reassoc nsz arcp contract float %646, %.sroa.86.1, !spirv.Decorations !843
  br label %._crit_edge.1.5

._crit_edge.1.5:                                  ; preds = %._crit_edge.5, %626
  %.sroa.86.2 = phi float [ %647, %626 ], [ %.sroa.86.1, %._crit_edge.5 ]
  br i1 %96, label %648, label %._crit_edge.2.5

648:                                              ; preds = %._crit_edge.1.5
  %.sroa.256.0.insert.ext693 = zext i32 %163 to i64
  %649 = sext i32 %62 to i64
  %650 = mul nsw i64 %649, %const_reg_qword3, !spirv.Decorations !836
  %651 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %650, i32 0
  %652 = getelementptr i16, i16 addrspace(4)* %651, i64 %.sroa.256.0.insert.ext693
  %653 = addrspacecast i16 addrspace(4)* %652 to i16 addrspace(1)*
  %654 = load i16, i16 addrspace(1)* %653, align 2
  %655 = sext i32 %92 to i64
  %656 = mul nsw i64 %655, %const_reg_qword5, !spirv.Decorations !836
  %657 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %656
  %658 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %657, i64 %.sroa.256.0.insert.ext693
  %659 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %658, i64 0, i32 0
  %660 = addrspacecast i16 addrspace(4)* %659 to i16 addrspace(1)*
  %661 = load i16, i16 addrspace(1)* %660, align 2
  %662 = zext i16 %654 to i32
  %663 = shl nuw i32 %662, 16, !spirv.Decorations !838
  %664 = bitcast i32 %663 to float
  %665 = zext i16 %661 to i32
  %666 = shl nuw i32 %665, 16, !spirv.Decorations !838
  %667 = bitcast i32 %666 to float
  %668 = fmul reassoc nsz arcp contract float %664, %667, !spirv.Decorations !843
  %669 = fadd reassoc nsz arcp contract float %668, %.sroa.150.1, !spirv.Decorations !843
  br label %._crit_edge.2.5

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5, %648
  %.sroa.150.2 = phi float [ %669, %648 ], [ %.sroa.150.1, %._crit_edge.1.5 ]
  br i1 %97, label %670, label %.preheader.5

670:                                              ; preds = %._crit_edge.2.5
  %.sroa.256.0.insert.ext698 = zext i32 %163 to i64
  %671 = sext i32 %65 to i64
  %672 = mul nsw i64 %671, %const_reg_qword3, !spirv.Decorations !836
  %673 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %672, i32 0
  %674 = getelementptr i16, i16 addrspace(4)* %673, i64 %.sroa.256.0.insert.ext698
  %675 = addrspacecast i16 addrspace(4)* %674 to i16 addrspace(1)*
  %676 = load i16, i16 addrspace(1)* %675, align 2
  %677 = sext i32 %92 to i64
  %678 = mul nsw i64 %677, %const_reg_qword5, !spirv.Decorations !836
  %679 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %678
  %680 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %679, i64 %.sroa.256.0.insert.ext698
  %681 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %680, i64 0, i32 0
  %682 = addrspacecast i16 addrspace(4)* %681 to i16 addrspace(1)*
  %683 = load i16, i16 addrspace(1)* %682, align 2
  %684 = zext i16 %676 to i32
  %685 = shl nuw i32 %684, 16, !spirv.Decorations !838
  %686 = bitcast i32 %685 to float
  %687 = zext i16 %683 to i32
  %688 = shl nuw i32 %687, 16, !spirv.Decorations !838
  %689 = bitcast i32 %688 to float
  %690 = fmul reassoc nsz arcp contract float %686, %689, !spirv.Decorations !843
  %691 = fadd reassoc nsz arcp contract float %690, %.sroa.214.1, !spirv.Decorations !843
  br label %.preheader.5

.preheader.5:                                     ; preds = %._crit_edge.2.5, %670
  %.sroa.214.2 = phi float [ %691, %670 ], [ %.sroa.214.1, %._crit_edge.2.5 ]
  br i1 %100, label %692, label %._crit_edge.6

692:                                              ; preds = %.preheader.5
  %.sroa.256.0.insert.ext703 = zext i32 %163 to i64
  %693 = sext i32 %29 to i64
  %694 = mul nsw i64 %693, %const_reg_qword3, !spirv.Decorations !836
  %695 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %694, i32 0
  %696 = getelementptr i16, i16 addrspace(4)* %695, i64 %.sroa.256.0.insert.ext703
  %697 = addrspacecast i16 addrspace(4)* %696 to i16 addrspace(1)*
  %698 = load i16, i16 addrspace(1)* %697, align 2
  %699 = sext i32 %98 to i64
  %700 = mul nsw i64 %699, %const_reg_qword5, !spirv.Decorations !836
  %701 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %700
  %702 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %701, i64 %.sroa.256.0.insert.ext703
  %703 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %702, i64 0, i32 0
  %704 = addrspacecast i16 addrspace(4)* %703 to i16 addrspace(1)*
  %705 = load i16, i16 addrspace(1)* %704, align 2
  %706 = zext i16 %698 to i32
  %707 = shl nuw i32 %706, 16, !spirv.Decorations !838
  %708 = bitcast i32 %707 to float
  %709 = zext i16 %705 to i32
  %710 = shl nuw i32 %709, 16, !spirv.Decorations !838
  %711 = bitcast i32 %710 to float
  %712 = fmul reassoc nsz arcp contract float %708, %711, !spirv.Decorations !843
  %713 = fadd reassoc nsz arcp contract float %712, %.sroa.26.1, !spirv.Decorations !843
  br label %._crit_edge.6

._crit_edge.6:                                    ; preds = %.preheader.5, %692
  %.sroa.26.2 = phi float [ %713, %692 ], [ %.sroa.26.1, %.preheader.5 ]
  br i1 %101, label %714, label %._crit_edge.1.6

714:                                              ; preds = %._crit_edge.6
  %.sroa.256.0.insert.ext708 = zext i32 %163 to i64
  %715 = sext i32 %59 to i64
  %716 = mul nsw i64 %715, %const_reg_qword3, !spirv.Decorations !836
  %717 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %716, i32 0
  %718 = getelementptr i16, i16 addrspace(4)* %717, i64 %.sroa.256.0.insert.ext708
  %719 = addrspacecast i16 addrspace(4)* %718 to i16 addrspace(1)*
  %720 = load i16, i16 addrspace(1)* %719, align 2
  %721 = sext i32 %98 to i64
  %722 = mul nsw i64 %721, %const_reg_qword5, !spirv.Decorations !836
  %723 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %722
  %724 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %723, i64 %.sroa.256.0.insert.ext708
  %725 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %724, i64 0, i32 0
  %726 = addrspacecast i16 addrspace(4)* %725 to i16 addrspace(1)*
  %727 = load i16, i16 addrspace(1)* %726, align 2
  %728 = zext i16 %720 to i32
  %729 = shl nuw i32 %728, 16, !spirv.Decorations !838
  %730 = bitcast i32 %729 to float
  %731 = zext i16 %727 to i32
  %732 = shl nuw i32 %731, 16, !spirv.Decorations !838
  %733 = bitcast i32 %732 to float
  %734 = fmul reassoc nsz arcp contract float %730, %733, !spirv.Decorations !843
  %735 = fadd reassoc nsz arcp contract float %734, %.sroa.90.1, !spirv.Decorations !843
  br label %._crit_edge.1.6

._crit_edge.1.6:                                  ; preds = %._crit_edge.6, %714
  %.sroa.90.2 = phi float [ %735, %714 ], [ %.sroa.90.1, %._crit_edge.6 ]
  br i1 %102, label %736, label %._crit_edge.2.6

736:                                              ; preds = %._crit_edge.1.6
  %.sroa.256.0.insert.ext713 = zext i32 %163 to i64
  %737 = sext i32 %62 to i64
  %738 = mul nsw i64 %737, %const_reg_qword3, !spirv.Decorations !836
  %739 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %738, i32 0
  %740 = getelementptr i16, i16 addrspace(4)* %739, i64 %.sroa.256.0.insert.ext713
  %741 = addrspacecast i16 addrspace(4)* %740 to i16 addrspace(1)*
  %742 = load i16, i16 addrspace(1)* %741, align 2
  %743 = sext i32 %98 to i64
  %744 = mul nsw i64 %743, %const_reg_qword5, !spirv.Decorations !836
  %745 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %744
  %746 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %745, i64 %.sroa.256.0.insert.ext713
  %747 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %746, i64 0, i32 0
  %748 = addrspacecast i16 addrspace(4)* %747 to i16 addrspace(1)*
  %749 = load i16, i16 addrspace(1)* %748, align 2
  %750 = zext i16 %742 to i32
  %751 = shl nuw i32 %750, 16, !spirv.Decorations !838
  %752 = bitcast i32 %751 to float
  %753 = zext i16 %749 to i32
  %754 = shl nuw i32 %753, 16, !spirv.Decorations !838
  %755 = bitcast i32 %754 to float
  %756 = fmul reassoc nsz arcp contract float %752, %755, !spirv.Decorations !843
  %757 = fadd reassoc nsz arcp contract float %756, %.sroa.154.1, !spirv.Decorations !843
  br label %._crit_edge.2.6

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6, %736
  %.sroa.154.2 = phi float [ %757, %736 ], [ %.sroa.154.1, %._crit_edge.1.6 ]
  br i1 %103, label %758, label %.preheader.6

758:                                              ; preds = %._crit_edge.2.6
  %.sroa.256.0.insert.ext718 = zext i32 %163 to i64
  %759 = sext i32 %65 to i64
  %760 = mul nsw i64 %759, %const_reg_qword3, !spirv.Decorations !836
  %761 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %760, i32 0
  %762 = getelementptr i16, i16 addrspace(4)* %761, i64 %.sroa.256.0.insert.ext718
  %763 = addrspacecast i16 addrspace(4)* %762 to i16 addrspace(1)*
  %764 = load i16, i16 addrspace(1)* %763, align 2
  %765 = sext i32 %98 to i64
  %766 = mul nsw i64 %765, %const_reg_qword5, !spirv.Decorations !836
  %767 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %766
  %768 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %767, i64 %.sroa.256.0.insert.ext718
  %769 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %768, i64 0, i32 0
  %770 = addrspacecast i16 addrspace(4)* %769 to i16 addrspace(1)*
  %771 = load i16, i16 addrspace(1)* %770, align 2
  %772 = zext i16 %764 to i32
  %773 = shl nuw i32 %772, 16, !spirv.Decorations !838
  %774 = bitcast i32 %773 to float
  %775 = zext i16 %771 to i32
  %776 = shl nuw i32 %775, 16, !spirv.Decorations !838
  %777 = bitcast i32 %776 to float
  %778 = fmul reassoc nsz arcp contract float %774, %777, !spirv.Decorations !843
  %779 = fadd reassoc nsz arcp contract float %778, %.sroa.218.1, !spirv.Decorations !843
  br label %.preheader.6

.preheader.6:                                     ; preds = %._crit_edge.2.6, %758
  %.sroa.218.2 = phi float [ %779, %758 ], [ %.sroa.218.1, %._crit_edge.2.6 ]
  br i1 %106, label %780, label %._crit_edge.7

780:                                              ; preds = %.preheader.6
  %.sroa.256.0.insert.ext723 = zext i32 %163 to i64
  %781 = sext i32 %29 to i64
  %782 = mul nsw i64 %781, %const_reg_qword3, !spirv.Decorations !836
  %783 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %782, i32 0
  %784 = getelementptr i16, i16 addrspace(4)* %783, i64 %.sroa.256.0.insert.ext723
  %785 = addrspacecast i16 addrspace(4)* %784 to i16 addrspace(1)*
  %786 = load i16, i16 addrspace(1)* %785, align 2
  %787 = sext i32 %104 to i64
  %788 = mul nsw i64 %787, %const_reg_qword5, !spirv.Decorations !836
  %789 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %788
  %790 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %789, i64 %.sroa.256.0.insert.ext723
  %791 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %790, i64 0, i32 0
  %792 = addrspacecast i16 addrspace(4)* %791 to i16 addrspace(1)*
  %793 = load i16, i16 addrspace(1)* %792, align 2
  %794 = zext i16 %786 to i32
  %795 = shl nuw i32 %794, 16, !spirv.Decorations !838
  %796 = bitcast i32 %795 to float
  %797 = zext i16 %793 to i32
  %798 = shl nuw i32 %797, 16, !spirv.Decorations !838
  %799 = bitcast i32 %798 to float
  %800 = fmul reassoc nsz arcp contract float %796, %799, !spirv.Decorations !843
  %801 = fadd reassoc nsz arcp contract float %800, %.sroa.30.1, !spirv.Decorations !843
  br label %._crit_edge.7

._crit_edge.7:                                    ; preds = %.preheader.6, %780
  %.sroa.30.2 = phi float [ %801, %780 ], [ %.sroa.30.1, %.preheader.6 ]
  br i1 %107, label %802, label %._crit_edge.1.7

802:                                              ; preds = %._crit_edge.7
  %.sroa.256.0.insert.ext728 = zext i32 %163 to i64
  %803 = sext i32 %59 to i64
  %804 = mul nsw i64 %803, %const_reg_qword3, !spirv.Decorations !836
  %805 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %804, i32 0
  %806 = getelementptr i16, i16 addrspace(4)* %805, i64 %.sroa.256.0.insert.ext728
  %807 = addrspacecast i16 addrspace(4)* %806 to i16 addrspace(1)*
  %808 = load i16, i16 addrspace(1)* %807, align 2
  %809 = sext i32 %104 to i64
  %810 = mul nsw i64 %809, %const_reg_qword5, !spirv.Decorations !836
  %811 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %810
  %812 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %811, i64 %.sroa.256.0.insert.ext728
  %813 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %812, i64 0, i32 0
  %814 = addrspacecast i16 addrspace(4)* %813 to i16 addrspace(1)*
  %815 = load i16, i16 addrspace(1)* %814, align 2
  %816 = zext i16 %808 to i32
  %817 = shl nuw i32 %816, 16, !spirv.Decorations !838
  %818 = bitcast i32 %817 to float
  %819 = zext i16 %815 to i32
  %820 = shl nuw i32 %819, 16, !spirv.Decorations !838
  %821 = bitcast i32 %820 to float
  %822 = fmul reassoc nsz arcp contract float %818, %821, !spirv.Decorations !843
  %823 = fadd reassoc nsz arcp contract float %822, %.sroa.94.1, !spirv.Decorations !843
  br label %._crit_edge.1.7

._crit_edge.1.7:                                  ; preds = %._crit_edge.7, %802
  %.sroa.94.2 = phi float [ %823, %802 ], [ %.sroa.94.1, %._crit_edge.7 ]
  br i1 %108, label %824, label %._crit_edge.2.7

824:                                              ; preds = %._crit_edge.1.7
  %.sroa.256.0.insert.ext733 = zext i32 %163 to i64
  %825 = sext i32 %62 to i64
  %826 = mul nsw i64 %825, %const_reg_qword3, !spirv.Decorations !836
  %827 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %826, i32 0
  %828 = getelementptr i16, i16 addrspace(4)* %827, i64 %.sroa.256.0.insert.ext733
  %829 = addrspacecast i16 addrspace(4)* %828 to i16 addrspace(1)*
  %830 = load i16, i16 addrspace(1)* %829, align 2
  %831 = sext i32 %104 to i64
  %832 = mul nsw i64 %831, %const_reg_qword5, !spirv.Decorations !836
  %833 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %832
  %834 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %833, i64 %.sroa.256.0.insert.ext733
  %835 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %834, i64 0, i32 0
  %836 = addrspacecast i16 addrspace(4)* %835 to i16 addrspace(1)*
  %837 = load i16, i16 addrspace(1)* %836, align 2
  %838 = zext i16 %830 to i32
  %839 = shl nuw i32 %838, 16, !spirv.Decorations !838
  %840 = bitcast i32 %839 to float
  %841 = zext i16 %837 to i32
  %842 = shl nuw i32 %841, 16, !spirv.Decorations !838
  %843 = bitcast i32 %842 to float
  %844 = fmul reassoc nsz arcp contract float %840, %843, !spirv.Decorations !843
  %845 = fadd reassoc nsz arcp contract float %844, %.sroa.158.1, !spirv.Decorations !843
  br label %._crit_edge.2.7

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7, %824
  %.sroa.158.2 = phi float [ %845, %824 ], [ %.sroa.158.1, %._crit_edge.1.7 ]
  br i1 %109, label %846, label %.preheader.7

846:                                              ; preds = %._crit_edge.2.7
  %.sroa.256.0.insert.ext738 = zext i32 %163 to i64
  %847 = sext i32 %65 to i64
  %848 = mul nsw i64 %847, %const_reg_qword3, !spirv.Decorations !836
  %849 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %848, i32 0
  %850 = getelementptr i16, i16 addrspace(4)* %849, i64 %.sroa.256.0.insert.ext738
  %851 = addrspacecast i16 addrspace(4)* %850 to i16 addrspace(1)*
  %852 = load i16, i16 addrspace(1)* %851, align 2
  %853 = sext i32 %104 to i64
  %854 = mul nsw i64 %853, %const_reg_qword5, !spirv.Decorations !836
  %855 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %854
  %856 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %855, i64 %.sroa.256.0.insert.ext738
  %857 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %856, i64 0, i32 0
  %858 = addrspacecast i16 addrspace(4)* %857 to i16 addrspace(1)*
  %859 = load i16, i16 addrspace(1)* %858, align 2
  %860 = zext i16 %852 to i32
  %861 = shl nuw i32 %860, 16, !spirv.Decorations !838
  %862 = bitcast i32 %861 to float
  %863 = zext i16 %859 to i32
  %864 = shl nuw i32 %863, 16, !spirv.Decorations !838
  %865 = bitcast i32 %864 to float
  %866 = fmul reassoc nsz arcp contract float %862, %865, !spirv.Decorations !843
  %867 = fadd reassoc nsz arcp contract float %866, %.sroa.222.1, !spirv.Decorations !843
  br label %.preheader.7

.preheader.7:                                     ; preds = %._crit_edge.2.7, %846
  %.sroa.222.2 = phi float [ %867, %846 ], [ %.sroa.222.1, %._crit_edge.2.7 ]
  br i1 %112, label %868, label %._crit_edge.8

868:                                              ; preds = %.preheader.7
  %.sroa.256.0.insert.ext743 = zext i32 %163 to i64
  %869 = sext i32 %29 to i64
  %870 = mul nsw i64 %869, %const_reg_qword3, !spirv.Decorations !836
  %871 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %870, i32 0
  %872 = getelementptr i16, i16 addrspace(4)* %871, i64 %.sroa.256.0.insert.ext743
  %873 = addrspacecast i16 addrspace(4)* %872 to i16 addrspace(1)*
  %874 = load i16, i16 addrspace(1)* %873, align 2
  %875 = sext i32 %110 to i64
  %876 = mul nsw i64 %875, %const_reg_qword5, !spirv.Decorations !836
  %877 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %876
  %878 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %877, i64 %.sroa.256.0.insert.ext743
  %879 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %878, i64 0, i32 0
  %880 = addrspacecast i16 addrspace(4)* %879 to i16 addrspace(1)*
  %881 = load i16, i16 addrspace(1)* %880, align 2
  %882 = zext i16 %874 to i32
  %883 = shl nuw i32 %882, 16, !spirv.Decorations !838
  %884 = bitcast i32 %883 to float
  %885 = zext i16 %881 to i32
  %886 = shl nuw i32 %885, 16, !spirv.Decorations !838
  %887 = bitcast i32 %886 to float
  %888 = fmul reassoc nsz arcp contract float %884, %887, !spirv.Decorations !843
  %889 = fadd reassoc nsz arcp contract float %888, %.sroa.34.1, !spirv.Decorations !843
  br label %._crit_edge.8

._crit_edge.8:                                    ; preds = %.preheader.7, %868
  %.sroa.34.2 = phi float [ %889, %868 ], [ %.sroa.34.1, %.preheader.7 ]
  br i1 %113, label %890, label %._crit_edge.1.8

890:                                              ; preds = %._crit_edge.8
  %.sroa.256.0.insert.ext748 = zext i32 %163 to i64
  %891 = sext i32 %59 to i64
  %892 = mul nsw i64 %891, %const_reg_qword3, !spirv.Decorations !836
  %893 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %892, i32 0
  %894 = getelementptr i16, i16 addrspace(4)* %893, i64 %.sroa.256.0.insert.ext748
  %895 = addrspacecast i16 addrspace(4)* %894 to i16 addrspace(1)*
  %896 = load i16, i16 addrspace(1)* %895, align 2
  %897 = sext i32 %110 to i64
  %898 = mul nsw i64 %897, %const_reg_qword5, !spirv.Decorations !836
  %899 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %898
  %900 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %899, i64 %.sroa.256.0.insert.ext748
  %901 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %900, i64 0, i32 0
  %902 = addrspacecast i16 addrspace(4)* %901 to i16 addrspace(1)*
  %903 = load i16, i16 addrspace(1)* %902, align 2
  %904 = zext i16 %896 to i32
  %905 = shl nuw i32 %904, 16, !spirv.Decorations !838
  %906 = bitcast i32 %905 to float
  %907 = zext i16 %903 to i32
  %908 = shl nuw i32 %907, 16, !spirv.Decorations !838
  %909 = bitcast i32 %908 to float
  %910 = fmul reassoc nsz arcp contract float %906, %909, !spirv.Decorations !843
  %911 = fadd reassoc nsz arcp contract float %910, %.sroa.98.1, !spirv.Decorations !843
  br label %._crit_edge.1.8

._crit_edge.1.8:                                  ; preds = %._crit_edge.8, %890
  %.sroa.98.2 = phi float [ %911, %890 ], [ %.sroa.98.1, %._crit_edge.8 ]
  br i1 %114, label %912, label %._crit_edge.2.8

912:                                              ; preds = %._crit_edge.1.8
  %.sroa.256.0.insert.ext753 = zext i32 %163 to i64
  %913 = sext i32 %62 to i64
  %914 = mul nsw i64 %913, %const_reg_qword3, !spirv.Decorations !836
  %915 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %914, i32 0
  %916 = getelementptr i16, i16 addrspace(4)* %915, i64 %.sroa.256.0.insert.ext753
  %917 = addrspacecast i16 addrspace(4)* %916 to i16 addrspace(1)*
  %918 = load i16, i16 addrspace(1)* %917, align 2
  %919 = sext i32 %110 to i64
  %920 = mul nsw i64 %919, %const_reg_qword5, !spirv.Decorations !836
  %921 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %920
  %922 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %921, i64 %.sroa.256.0.insert.ext753
  %923 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %922, i64 0, i32 0
  %924 = addrspacecast i16 addrspace(4)* %923 to i16 addrspace(1)*
  %925 = load i16, i16 addrspace(1)* %924, align 2
  %926 = zext i16 %918 to i32
  %927 = shl nuw i32 %926, 16, !spirv.Decorations !838
  %928 = bitcast i32 %927 to float
  %929 = zext i16 %925 to i32
  %930 = shl nuw i32 %929, 16, !spirv.Decorations !838
  %931 = bitcast i32 %930 to float
  %932 = fmul reassoc nsz arcp contract float %928, %931, !spirv.Decorations !843
  %933 = fadd reassoc nsz arcp contract float %932, %.sroa.162.1, !spirv.Decorations !843
  br label %._crit_edge.2.8

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8, %912
  %.sroa.162.2 = phi float [ %933, %912 ], [ %.sroa.162.1, %._crit_edge.1.8 ]
  br i1 %115, label %934, label %.preheader.8

934:                                              ; preds = %._crit_edge.2.8
  %.sroa.256.0.insert.ext758 = zext i32 %163 to i64
  %935 = sext i32 %65 to i64
  %936 = mul nsw i64 %935, %const_reg_qword3, !spirv.Decorations !836
  %937 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %936, i32 0
  %938 = getelementptr i16, i16 addrspace(4)* %937, i64 %.sroa.256.0.insert.ext758
  %939 = addrspacecast i16 addrspace(4)* %938 to i16 addrspace(1)*
  %940 = load i16, i16 addrspace(1)* %939, align 2
  %941 = sext i32 %110 to i64
  %942 = mul nsw i64 %941, %const_reg_qword5, !spirv.Decorations !836
  %943 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %942
  %944 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %943, i64 %.sroa.256.0.insert.ext758
  %945 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %944, i64 0, i32 0
  %946 = addrspacecast i16 addrspace(4)* %945 to i16 addrspace(1)*
  %947 = load i16, i16 addrspace(1)* %946, align 2
  %948 = zext i16 %940 to i32
  %949 = shl nuw i32 %948, 16, !spirv.Decorations !838
  %950 = bitcast i32 %949 to float
  %951 = zext i16 %947 to i32
  %952 = shl nuw i32 %951, 16, !spirv.Decorations !838
  %953 = bitcast i32 %952 to float
  %954 = fmul reassoc nsz arcp contract float %950, %953, !spirv.Decorations !843
  %955 = fadd reassoc nsz arcp contract float %954, %.sroa.226.1, !spirv.Decorations !843
  br label %.preheader.8

.preheader.8:                                     ; preds = %._crit_edge.2.8, %934
  %.sroa.226.2 = phi float [ %955, %934 ], [ %.sroa.226.1, %._crit_edge.2.8 ]
  br i1 %118, label %956, label %._crit_edge.9

956:                                              ; preds = %.preheader.8
  %.sroa.256.0.insert.ext763 = zext i32 %163 to i64
  %957 = sext i32 %29 to i64
  %958 = mul nsw i64 %957, %const_reg_qword3, !spirv.Decorations !836
  %959 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %958, i32 0
  %960 = getelementptr i16, i16 addrspace(4)* %959, i64 %.sroa.256.0.insert.ext763
  %961 = addrspacecast i16 addrspace(4)* %960 to i16 addrspace(1)*
  %962 = load i16, i16 addrspace(1)* %961, align 2
  %963 = sext i32 %116 to i64
  %964 = mul nsw i64 %963, %const_reg_qword5, !spirv.Decorations !836
  %965 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %964
  %966 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %965, i64 %.sroa.256.0.insert.ext763
  %967 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %966, i64 0, i32 0
  %968 = addrspacecast i16 addrspace(4)* %967 to i16 addrspace(1)*
  %969 = load i16, i16 addrspace(1)* %968, align 2
  %970 = zext i16 %962 to i32
  %971 = shl nuw i32 %970, 16, !spirv.Decorations !838
  %972 = bitcast i32 %971 to float
  %973 = zext i16 %969 to i32
  %974 = shl nuw i32 %973, 16, !spirv.Decorations !838
  %975 = bitcast i32 %974 to float
  %976 = fmul reassoc nsz arcp contract float %972, %975, !spirv.Decorations !843
  %977 = fadd reassoc nsz arcp contract float %976, %.sroa.38.1, !spirv.Decorations !843
  br label %._crit_edge.9

._crit_edge.9:                                    ; preds = %.preheader.8, %956
  %.sroa.38.2 = phi float [ %977, %956 ], [ %.sroa.38.1, %.preheader.8 ]
  br i1 %119, label %978, label %._crit_edge.1.9

978:                                              ; preds = %._crit_edge.9
  %.sroa.256.0.insert.ext768 = zext i32 %163 to i64
  %979 = sext i32 %59 to i64
  %980 = mul nsw i64 %979, %const_reg_qword3, !spirv.Decorations !836
  %981 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %980, i32 0
  %982 = getelementptr i16, i16 addrspace(4)* %981, i64 %.sroa.256.0.insert.ext768
  %983 = addrspacecast i16 addrspace(4)* %982 to i16 addrspace(1)*
  %984 = load i16, i16 addrspace(1)* %983, align 2
  %985 = sext i32 %116 to i64
  %986 = mul nsw i64 %985, %const_reg_qword5, !spirv.Decorations !836
  %987 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %986
  %988 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %987, i64 %.sroa.256.0.insert.ext768
  %989 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %988, i64 0, i32 0
  %990 = addrspacecast i16 addrspace(4)* %989 to i16 addrspace(1)*
  %991 = load i16, i16 addrspace(1)* %990, align 2
  %992 = zext i16 %984 to i32
  %993 = shl nuw i32 %992, 16, !spirv.Decorations !838
  %994 = bitcast i32 %993 to float
  %995 = zext i16 %991 to i32
  %996 = shl nuw i32 %995, 16, !spirv.Decorations !838
  %997 = bitcast i32 %996 to float
  %998 = fmul reassoc nsz arcp contract float %994, %997, !spirv.Decorations !843
  %999 = fadd reassoc nsz arcp contract float %998, %.sroa.102.1, !spirv.Decorations !843
  br label %._crit_edge.1.9

._crit_edge.1.9:                                  ; preds = %._crit_edge.9, %978
  %.sroa.102.2 = phi float [ %999, %978 ], [ %.sroa.102.1, %._crit_edge.9 ]
  br i1 %120, label %1000, label %._crit_edge.2.9

1000:                                             ; preds = %._crit_edge.1.9
  %.sroa.256.0.insert.ext773 = zext i32 %163 to i64
  %1001 = sext i32 %62 to i64
  %1002 = mul nsw i64 %1001, %const_reg_qword3, !spirv.Decorations !836
  %1003 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1002, i32 0
  %1004 = getelementptr i16, i16 addrspace(4)* %1003, i64 %.sroa.256.0.insert.ext773
  %1005 = addrspacecast i16 addrspace(4)* %1004 to i16 addrspace(1)*
  %1006 = load i16, i16 addrspace(1)* %1005, align 2
  %1007 = sext i32 %116 to i64
  %1008 = mul nsw i64 %1007, %const_reg_qword5, !spirv.Decorations !836
  %1009 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1008
  %1010 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1009, i64 %.sroa.256.0.insert.ext773
  %1011 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1010, i64 0, i32 0
  %1012 = addrspacecast i16 addrspace(4)* %1011 to i16 addrspace(1)*
  %1013 = load i16, i16 addrspace(1)* %1012, align 2
  %1014 = zext i16 %1006 to i32
  %1015 = shl nuw i32 %1014, 16, !spirv.Decorations !838
  %1016 = bitcast i32 %1015 to float
  %1017 = zext i16 %1013 to i32
  %1018 = shl nuw i32 %1017, 16, !spirv.Decorations !838
  %1019 = bitcast i32 %1018 to float
  %1020 = fmul reassoc nsz arcp contract float %1016, %1019, !spirv.Decorations !843
  %1021 = fadd reassoc nsz arcp contract float %1020, %.sroa.166.1, !spirv.Decorations !843
  br label %._crit_edge.2.9

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9, %1000
  %.sroa.166.2 = phi float [ %1021, %1000 ], [ %.sroa.166.1, %._crit_edge.1.9 ]
  br i1 %121, label %1022, label %.preheader.9

1022:                                             ; preds = %._crit_edge.2.9
  %.sroa.256.0.insert.ext778 = zext i32 %163 to i64
  %1023 = sext i32 %65 to i64
  %1024 = mul nsw i64 %1023, %const_reg_qword3, !spirv.Decorations !836
  %1025 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1024, i32 0
  %1026 = getelementptr i16, i16 addrspace(4)* %1025, i64 %.sroa.256.0.insert.ext778
  %1027 = addrspacecast i16 addrspace(4)* %1026 to i16 addrspace(1)*
  %1028 = load i16, i16 addrspace(1)* %1027, align 2
  %1029 = sext i32 %116 to i64
  %1030 = mul nsw i64 %1029, %const_reg_qword5, !spirv.Decorations !836
  %1031 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1030
  %1032 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1031, i64 %.sroa.256.0.insert.ext778
  %1033 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1032, i64 0, i32 0
  %1034 = addrspacecast i16 addrspace(4)* %1033 to i16 addrspace(1)*
  %1035 = load i16, i16 addrspace(1)* %1034, align 2
  %1036 = zext i16 %1028 to i32
  %1037 = shl nuw i32 %1036, 16, !spirv.Decorations !838
  %1038 = bitcast i32 %1037 to float
  %1039 = zext i16 %1035 to i32
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !838
  %1041 = bitcast i32 %1040 to float
  %1042 = fmul reassoc nsz arcp contract float %1038, %1041, !spirv.Decorations !843
  %1043 = fadd reassoc nsz arcp contract float %1042, %.sroa.230.1, !spirv.Decorations !843
  br label %.preheader.9

.preheader.9:                                     ; preds = %._crit_edge.2.9, %1022
  %.sroa.230.2 = phi float [ %1043, %1022 ], [ %.sroa.230.1, %._crit_edge.2.9 ]
  br i1 %124, label %1044, label %._crit_edge.10

1044:                                             ; preds = %.preheader.9
  %.sroa.256.0.insert.ext783 = zext i32 %163 to i64
  %1045 = sext i32 %29 to i64
  %1046 = mul nsw i64 %1045, %const_reg_qword3, !spirv.Decorations !836
  %1047 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1046, i32 0
  %1048 = getelementptr i16, i16 addrspace(4)* %1047, i64 %.sroa.256.0.insert.ext783
  %1049 = addrspacecast i16 addrspace(4)* %1048 to i16 addrspace(1)*
  %1050 = load i16, i16 addrspace(1)* %1049, align 2
  %1051 = sext i32 %122 to i64
  %1052 = mul nsw i64 %1051, %const_reg_qword5, !spirv.Decorations !836
  %1053 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1052
  %1054 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1053, i64 %.sroa.256.0.insert.ext783
  %1055 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1054, i64 0, i32 0
  %1056 = addrspacecast i16 addrspace(4)* %1055 to i16 addrspace(1)*
  %1057 = load i16, i16 addrspace(1)* %1056, align 2
  %1058 = zext i16 %1050 to i32
  %1059 = shl nuw i32 %1058, 16, !spirv.Decorations !838
  %1060 = bitcast i32 %1059 to float
  %1061 = zext i16 %1057 to i32
  %1062 = shl nuw i32 %1061, 16, !spirv.Decorations !838
  %1063 = bitcast i32 %1062 to float
  %1064 = fmul reassoc nsz arcp contract float %1060, %1063, !spirv.Decorations !843
  %1065 = fadd reassoc nsz arcp contract float %1064, %.sroa.42.1, !spirv.Decorations !843
  br label %._crit_edge.10

._crit_edge.10:                                   ; preds = %.preheader.9, %1044
  %.sroa.42.2 = phi float [ %1065, %1044 ], [ %.sroa.42.1, %.preheader.9 ]
  br i1 %125, label %1066, label %._crit_edge.1.10

1066:                                             ; preds = %._crit_edge.10
  %.sroa.256.0.insert.ext788 = zext i32 %163 to i64
  %1067 = sext i32 %59 to i64
  %1068 = mul nsw i64 %1067, %const_reg_qword3, !spirv.Decorations !836
  %1069 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1068, i32 0
  %1070 = getelementptr i16, i16 addrspace(4)* %1069, i64 %.sroa.256.0.insert.ext788
  %1071 = addrspacecast i16 addrspace(4)* %1070 to i16 addrspace(1)*
  %1072 = load i16, i16 addrspace(1)* %1071, align 2
  %1073 = sext i32 %122 to i64
  %1074 = mul nsw i64 %1073, %const_reg_qword5, !spirv.Decorations !836
  %1075 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1074
  %1076 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1075, i64 %.sroa.256.0.insert.ext788
  %1077 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1076, i64 0, i32 0
  %1078 = addrspacecast i16 addrspace(4)* %1077 to i16 addrspace(1)*
  %1079 = load i16, i16 addrspace(1)* %1078, align 2
  %1080 = zext i16 %1072 to i32
  %1081 = shl nuw i32 %1080, 16, !spirv.Decorations !838
  %1082 = bitcast i32 %1081 to float
  %1083 = zext i16 %1079 to i32
  %1084 = shl nuw i32 %1083, 16, !spirv.Decorations !838
  %1085 = bitcast i32 %1084 to float
  %1086 = fmul reassoc nsz arcp contract float %1082, %1085, !spirv.Decorations !843
  %1087 = fadd reassoc nsz arcp contract float %1086, %.sroa.106.1, !spirv.Decorations !843
  br label %._crit_edge.1.10

._crit_edge.1.10:                                 ; preds = %._crit_edge.10, %1066
  %.sroa.106.2 = phi float [ %1087, %1066 ], [ %.sroa.106.1, %._crit_edge.10 ]
  br i1 %126, label %1088, label %._crit_edge.2.10

1088:                                             ; preds = %._crit_edge.1.10
  %.sroa.256.0.insert.ext793 = zext i32 %163 to i64
  %1089 = sext i32 %62 to i64
  %1090 = mul nsw i64 %1089, %const_reg_qword3, !spirv.Decorations !836
  %1091 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1090, i32 0
  %1092 = getelementptr i16, i16 addrspace(4)* %1091, i64 %.sroa.256.0.insert.ext793
  %1093 = addrspacecast i16 addrspace(4)* %1092 to i16 addrspace(1)*
  %1094 = load i16, i16 addrspace(1)* %1093, align 2
  %1095 = sext i32 %122 to i64
  %1096 = mul nsw i64 %1095, %const_reg_qword5, !spirv.Decorations !836
  %1097 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1096
  %1098 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1097, i64 %.sroa.256.0.insert.ext793
  %1099 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1098, i64 0, i32 0
  %1100 = addrspacecast i16 addrspace(4)* %1099 to i16 addrspace(1)*
  %1101 = load i16, i16 addrspace(1)* %1100, align 2
  %1102 = zext i16 %1094 to i32
  %1103 = shl nuw i32 %1102, 16, !spirv.Decorations !838
  %1104 = bitcast i32 %1103 to float
  %1105 = zext i16 %1101 to i32
  %1106 = shl nuw i32 %1105, 16, !spirv.Decorations !838
  %1107 = bitcast i32 %1106 to float
  %1108 = fmul reassoc nsz arcp contract float %1104, %1107, !spirv.Decorations !843
  %1109 = fadd reassoc nsz arcp contract float %1108, %.sroa.170.1, !spirv.Decorations !843
  br label %._crit_edge.2.10

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10, %1088
  %.sroa.170.2 = phi float [ %1109, %1088 ], [ %.sroa.170.1, %._crit_edge.1.10 ]
  br i1 %127, label %1110, label %.preheader.10

1110:                                             ; preds = %._crit_edge.2.10
  %.sroa.256.0.insert.ext798 = zext i32 %163 to i64
  %1111 = sext i32 %65 to i64
  %1112 = mul nsw i64 %1111, %const_reg_qword3, !spirv.Decorations !836
  %1113 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1112, i32 0
  %1114 = getelementptr i16, i16 addrspace(4)* %1113, i64 %.sroa.256.0.insert.ext798
  %1115 = addrspacecast i16 addrspace(4)* %1114 to i16 addrspace(1)*
  %1116 = load i16, i16 addrspace(1)* %1115, align 2
  %1117 = sext i32 %122 to i64
  %1118 = mul nsw i64 %1117, %const_reg_qword5, !spirv.Decorations !836
  %1119 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1118
  %1120 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1119, i64 %.sroa.256.0.insert.ext798
  %1121 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1120, i64 0, i32 0
  %1122 = addrspacecast i16 addrspace(4)* %1121 to i16 addrspace(1)*
  %1123 = load i16, i16 addrspace(1)* %1122, align 2
  %1124 = zext i16 %1116 to i32
  %1125 = shl nuw i32 %1124, 16, !spirv.Decorations !838
  %1126 = bitcast i32 %1125 to float
  %1127 = zext i16 %1123 to i32
  %1128 = shl nuw i32 %1127, 16, !spirv.Decorations !838
  %1129 = bitcast i32 %1128 to float
  %1130 = fmul reassoc nsz arcp contract float %1126, %1129, !spirv.Decorations !843
  %1131 = fadd reassoc nsz arcp contract float %1130, %.sroa.234.1, !spirv.Decorations !843
  br label %.preheader.10

.preheader.10:                                    ; preds = %._crit_edge.2.10, %1110
  %.sroa.234.2 = phi float [ %1131, %1110 ], [ %.sroa.234.1, %._crit_edge.2.10 ]
  br i1 %130, label %1132, label %._crit_edge.11

1132:                                             ; preds = %.preheader.10
  %.sroa.256.0.insert.ext803 = zext i32 %163 to i64
  %1133 = sext i32 %29 to i64
  %1134 = mul nsw i64 %1133, %const_reg_qword3, !spirv.Decorations !836
  %1135 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1134, i32 0
  %1136 = getelementptr i16, i16 addrspace(4)* %1135, i64 %.sroa.256.0.insert.ext803
  %1137 = addrspacecast i16 addrspace(4)* %1136 to i16 addrspace(1)*
  %1138 = load i16, i16 addrspace(1)* %1137, align 2
  %1139 = sext i32 %128 to i64
  %1140 = mul nsw i64 %1139, %const_reg_qword5, !spirv.Decorations !836
  %1141 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1140
  %1142 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1141, i64 %.sroa.256.0.insert.ext803
  %1143 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1142, i64 0, i32 0
  %1144 = addrspacecast i16 addrspace(4)* %1143 to i16 addrspace(1)*
  %1145 = load i16, i16 addrspace(1)* %1144, align 2
  %1146 = zext i16 %1138 to i32
  %1147 = shl nuw i32 %1146, 16, !spirv.Decorations !838
  %1148 = bitcast i32 %1147 to float
  %1149 = zext i16 %1145 to i32
  %1150 = shl nuw i32 %1149, 16, !spirv.Decorations !838
  %1151 = bitcast i32 %1150 to float
  %1152 = fmul reassoc nsz arcp contract float %1148, %1151, !spirv.Decorations !843
  %1153 = fadd reassoc nsz arcp contract float %1152, %.sroa.46.1, !spirv.Decorations !843
  br label %._crit_edge.11

._crit_edge.11:                                   ; preds = %.preheader.10, %1132
  %.sroa.46.2 = phi float [ %1153, %1132 ], [ %.sroa.46.1, %.preheader.10 ]
  br i1 %131, label %1154, label %._crit_edge.1.11

1154:                                             ; preds = %._crit_edge.11
  %.sroa.256.0.insert.ext808 = zext i32 %163 to i64
  %1155 = sext i32 %59 to i64
  %1156 = mul nsw i64 %1155, %const_reg_qword3, !spirv.Decorations !836
  %1157 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1156, i32 0
  %1158 = getelementptr i16, i16 addrspace(4)* %1157, i64 %.sroa.256.0.insert.ext808
  %1159 = addrspacecast i16 addrspace(4)* %1158 to i16 addrspace(1)*
  %1160 = load i16, i16 addrspace(1)* %1159, align 2
  %1161 = sext i32 %128 to i64
  %1162 = mul nsw i64 %1161, %const_reg_qword5, !spirv.Decorations !836
  %1163 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1162
  %1164 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1163, i64 %.sroa.256.0.insert.ext808
  %1165 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1164, i64 0, i32 0
  %1166 = addrspacecast i16 addrspace(4)* %1165 to i16 addrspace(1)*
  %1167 = load i16, i16 addrspace(1)* %1166, align 2
  %1168 = zext i16 %1160 to i32
  %1169 = shl nuw i32 %1168, 16, !spirv.Decorations !838
  %1170 = bitcast i32 %1169 to float
  %1171 = zext i16 %1167 to i32
  %1172 = shl nuw i32 %1171, 16, !spirv.Decorations !838
  %1173 = bitcast i32 %1172 to float
  %1174 = fmul reassoc nsz arcp contract float %1170, %1173, !spirv.Decorations !843
  %1175 = fadd reassoc nsz arcp contract float %1174, %.sroa.110.1, !spirv.Decorations !843
  br label %._crit_edge.1.11

._crit_edge.1.11:                                 ; preds = %._crit_edge.11, %1154
  %.sroa.110.2 = phi float [ %1175, %1154 ], [ %.sroa.110.1, %._crit_edge.11 ]
  br i1 %132, label %1176, label %._crit_edge.2.11

1176:                                             ; preds = %._crit_edge.1.11
  %.sroa.256.0.insert.ext813 = zext i32 %163 to i64
  %1177 = sext i32 %62 to i64
  %1178 = mul nsw i64 %1177, %const_reg_qword3, !spirv.Decorations !836
  %1179 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1178, i32 0
  %1180 = getelementptr i16, i16 addrspace(4)* %1179, i64 %.sroa.256.0.insert.ext813
  %1181 = addrspacecast i16 addrspace(4)* %1180 to i16 addrspace(1)*
  %1182 = load i16, i16 addrspace(1)* %1181, align 2
  %1183 = sext i32 %128 to i64
  %1184 = mul nsw i64 %1183, %const_reg_qword5, !spirv.Decorations !836
  %1185 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1184
  %1186 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1185, i64 %.sroa.256.0.insert.ext813
  %1187 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1186, i64 0, i32 0
  %1188 = addrspacecast i16 addrspace(4)* %1187 to i16 addrspace(1)*
  %1189 = load i16, i16 addrspace(1)* %1188, align 2
  %1190 = zext i16 %1182 to i32
  %1191 = shl nuw i32 %1190, 16, !spirv.Decorations !838
  %1192 = bitcast i32 %1191 to float
  %1193 = zext i16 %1189 to i32
  %1194 = shl nuw i32 %1193, 16, !spirv.Decorations !838
  %1195 = bitcast i32 %1194 to float
  %1196 = fmul reassoc nsz arcp contract float %1192, %1195, !spirv.Decorations !843
  %1197 = fadd reassoc nsz arcp contract float %1196, %.sroa.174.1, !spirv.Decorations !843
  br label %._crit_edge.2.11

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11, %1176
  %.sroa.174.2 = phi float [ %1197, %1176 ], [ %.sroa.174.1, %._crit_edge.1.11 ]
  br i1 %133, label %1198, label %.preheader.11

1198:                                             ; preds = %._crit_edge.2.11
  %.sroa.256.0.insert.ext818 = zext i32 %163 to i64
  %1199 = sext i32 %65 to i64
  %1200 = mul nsw i64 %1199, %const_reg_qword3, !spirv.Decorations !836
  %1201 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1200, i32 0
  %1202 = getelementptr i16, i16 addrspace(4)* %1201, i64 %.sroa.256.0.insert.ext818
  %1203 = addrspacecast i16 addrspace(4)* %1202 to i16 addrspace(1)*
  %1204 = load i16, i16 addrspace(1)* %1203, align 2
  %1205 = sext i32 %128 to i64
  %1206 = mul nsw i64 %1205, %const_reg_qword5, !spirv.Decorations !836
  %1207 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1206
  %1208 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1207, i64 %.sroa.256.0.insert.ext818
  %1209 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1208, i64 0, i32 0
  %1210 = addrspacecast i16 addrspace(4)* %1209 to i16 addrspace(1)*
  %1211 = load i16, i16 addrspace(1)* %1210, align 2
  %1212 = zext i16 %1204 to i32
  %1213 = shl nuw i32 %1212, 16, !spirv.Decorations !838
  %1214 = bitcast i32 %1213 to float
  %1215 = zext i16 %1211 to i32
  %1216 = shl nuw i32 %1215, 16, !spirv.Decorations !838
  %1217 = bitcast i32 %1216 to float
  %1218 = fmul reassoc nsz arcp contract float %1214, %1217, !spirv.Decorations !843
  %1219 = fadd reassoc nsz arcp contract float %1218, %.sroa.238.1, !spirv.Decorations !843
  br label %.preheader.11

.preheader.11:                                    ; preds = %._crit_edge.2.11, %1198
  %.sroa.238.2 = phi float [ %1219, %1198 ], [ %.sroa.238.1, %._crit_edge.2.11 ]
  br i1 %136, label %1220, label %._crit_edge.12

1220:                                             ; preds = %.preheader.11
  %.sroa.256.0.insert.ext823 = zext i32 %163 to i64
  %1221 = sext i32 %29 to i64
  %1222 = mul nsw i64 %1221, %const_reg_qword3, !spirv.Decorations !836
  %1223 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1222, i32 0
  %1224 = getelementptr i16, i16 addrspace(4)* %1223, i64 %.sroa.256.0.insert.ext823
  %1225 = addrspacecast i16 addrspace(4)* %1224 to i16 addrspace(1)*
  %1226 = load i16, i16 addrspace(1)* %1225, align 2
  %1227 = sext i32 %134 to i64
  %1228 = mul nsw i64 %1227, %const_reg_qword5, !spirv.Decorations !836
  %1229 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1228
  %1230 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1229, i64 %.sroa.256.0.insert.ext823
  %1231 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1230, i64 0, i32 0
  %1232 = addrspacecast i16 addrspace(4)* %1231 to i16 addrspace(1)*
  %1233 = load i16, i16 addrspace(1)* %1232, align 2
  %1234 = zext i16 %1226 to i32
  %1235 = shl nuw i32 %1234, 16, !spirv.Decorations !838
  %1236 = bitcast i32 %1235 to float
  %1237 = zext i16 %1233 to i32
  %1238 = shl nuw i32 %1237, 16, !spirv.Decorations !838
  %1239 = bitcast i32 %1238 to float
  %1240 = fmul reassoc nsz arcp contract float %1236, %1239, !spirv.Decorations !843
  %1241 = fadd reassoc nsz arcp contract float %1240, %.sroa.50.1, !spirv.Decorations !843
  br label %._crit_edge.12

._crit_edge.12:                                   ; preds = %.preheader.11, %1220
  %.sroa.50.2 = phi float [ %1241, %1220 ], [ %.sroa.50.1, %.preheader.11 ]
  br i1 %137, label %1242, label %._crit_edge.1.12

1242:                                             ; preds = %._crit_edge.12
  %.sroa.256.0.insert.ext828 = zext i32 %163 to i64
  %1243 = sext i32 %59 to i64
  %1244 = mul nsw i64 %1243, %const_reg_qword3, !spirv.Decorations !836
  %1245 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1244, i32 0
  %1246 = getelementptr i16, i16 addrspace(4)* %1245, i64 %.sroa.256.0.insert.ext828
  %1247 = addrspacecast i16 addrspace(4)* %1246 to i16 addrspace(1)*
  %1248 = load i16, i16 addrspace(1)* %1247, align 2
  %1249 = sext i32 %134 to i64
  %1250 = mul nsw i64 %1249, %const_reg_qword5, !spirv.Decorations !836
  %1251 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1250
  %1252 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1251, i64 %.sroa.256.0.insert.ext828
  %1253 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1252, i64 0, i32 0
  %1254 = addrspacecast i16 addrspace(4)* %1253 to i16 addrspace(1)*
  %1255 = load i16, i16 addrspace(1)* %1254, align 2
  %1256 = zext i16 %1248 to i32
  %1257 = shl nuw i32 %1256, 16, !spirv.Decorations !838
  %1258 = bitcast i32 %1257 to float
  %1259 = zext i16 %1255 to i32
  %1260 = shl nuw i32 %1259, 16, !spirv.Decorations !838
  %1261 = bitcast i32 %1260 to float
  %1262 = fmul reassoc nsz arcp contract float %1258, %1261, !spirv.Decorations !843
  %1263 = fadd reassoc nsz arcp contract float %1262, %.sroa.114.1, !spirv.Decorations !843
  br label %._crit_edge.1.12

._crit_edge.1.12:                                 ; preds = %._crit_edge.12, %1242
  %.sroa.114.2 = phi float [ %1263, %1242 ], [ %.sroa.114.1, %._crit_edge.12 ]
  br i1 %138, label %1264, label %._crit_edge.2.12

1264:                                             ; preds = %._crit_edge.1.12
  %.sroa.256.0.insert.ext833 = zext i32 %163 to i64
  %1265 = sext i32 %62 to i64
  %1266 = mul nsw i64 %1265, %const_reg_qword3, !spirv.Decorations !836
  %1267 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1266, i32 0
  %1268 = getelementptr i16, i16 addrspace(4)* %1267, i64 %.sroa.256.0.insert.ext833
  %1269 = addrspacecast i16 addrspace(4)* %1268 to i16 addrspace(1)*
  %1270 = load i16, i16 addrspace(1)* %1269, align 2
  %1271 = sext i32 %134 to i64
  %1272 = mul nsw i64 %1271, %const_reg_qword5, !spirv.Decorations !836
  %1273 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1272
  %1274 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1273, i64 %.sroa.256.0.insert.ext833
  %1275 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1274, i64 0, i32 0
  %1276 = addrspacecast i16 addrspace(4)* %1275 to i16 addrspace(1)*
  %1277 = load i16, i16 addrspace(1)* %1276, align 2
  %1278 = zext i16 %1270 to i32
  %1279 = shl nuw i32 %1278, 16, !spirv.Decorations !838
  %1280 = bitcast i32 %1279 to float
  %1281 = zext i16 %1277 to i32
  %1282 = shl nuw i32 %1281, 16, !spirv.Decorations !838
  %1283 = bitcast i32 %1282 to float
  %1284 = fmul reassoc nsz arcp contract float %1280, %1283, !spirv.Decorations !843
  %1285 = fadd reassoc nsz arcp contract float %1284, %.sroa.178.1, !spirv.Decorations !843
  br label %._crit_edge.2.12

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12, %1264
  %.sroa.178.2 = phi float [ %1285, %1264 ], [ %.sroa.178.1, %._crit_edge.1.12 ]
  br i1 %139, label %1286, label %.preheader.12

1286:                                             ; preds = %._crit_edge.2.12
  %.sroa.256.0.insert.ext838 = zext i32 %163 to i64
  %1287 = sext i32 %65 to i64
  %1288 = mul nsw i64 %1287, %const_reg_qword3, !spirv.Decorations !836
  %1289 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1288, i32 0
  %1290 = getelementptr i16, i16 addrspace(4)* %1289, i64 %.sroa.256.0.insert.ext838
  %1291 = addrspacecast i16 addrspace(4)* %1290 to i16 addrspace(1)*
  %1292 = load i16, i16 addrspace(1)* %1291, align 2
  %1293 = sext i32 %134 to i64
  %1294 = mul nsw i64 %1293, %const_reg_qword5, !spirv.Decorations !836
  %1295 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1294
  %1296 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1295, i64 %.sroa.256.0.insert.ext838
  %1297 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1296, i64 0, i32 0
  %1298 = addrspacecast i16 addrspace(4)* %1297 to i16 addrspace(1)*
  %1299 = load i16, i16 addrspace(1)* %1298, align 2
  %1300 = zext i16 %1292 to i32
  %1301 = shl nuw i32 %1300, 16, !spirv.Decorations !838
  %1302 = bitcast i32 %1301 to float
  %1303 = zext i16 %1299 to i32
  %1304 = shl nuw i32 %1303, 16, !spirv.Decorations !838
  %1305 = bitcast i32 %1304 to float
  %1306 = fmul reassoc nsz arcp contract float %1302, %1305, !spirv.Decorations !843
  %1307 = fadd reassoc nsz arcp contract float %1306, %.sroa.242.1, !spirv.Decorations !843
  br label %.preheader.12

.preheader.12:                                    ; preds = %._crit_edge.2.12, %1286
  %.sroa.242.2 = phi float [ %1307, %1286 ], [ %.sroa.242.1, %._crit_edge.2.12 ]
  br i1 %142, label %1308, label %._crit_edge.13

1308:                                             ; preds = %.preheader.12
  %.sroa.256.0.insert.ext843 = zext i32 %163 to i64
  %1309 = sext i32 %29 to i64
  %1310 = mul nsw i64 %1309, %const_reg_qword3, !spirv.Decorations !836
  %1311 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1310, i32 0
  %1312 = getelementptr i16, i16 addrspace(4)* %1311, i64 %.sroa.256.0.insert.ext843
  %1313 = addrspacecast i16 addrspace(4)* %1312 to i16 addrspace(1)*
  %1314 = load i16, i16 addrspace(1)* %1313, align 2
  %1315 = sext i32 %140 to i64
  %1316 = mul nsw i64 %1315, %const_reg_qword5, !spirv.Decorations !836
  %1317 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1316
  %1318 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1317, i64 %.sroa.256.0.insert.ext843
  %1319 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1318, i64 0, i32 0
  %1320 = addrspacecast i16 addrspace(4)* %1319 to i16 addrspace(1)*
  %1321 = load i16, i16 addrspace(1)* %1320, align 2
  %1322 = zext i16 %1314 to i32
  %1323 = shl nuw i32 %1322, 16, !spirv.Decorations !838
  %1324 = bitcast i32 %1323 to float
  %1325 = zext i16 %1321 to i32
  %1326 = shl nuw i32 %1325, 16, !spirv.Decorations !838
  %1327 = bitcast i32 %1326 to float
  %1328 = fmul reassoc nsz arcp contract float %1324, %1327, !spirv.Decorations !843
  %1329 = fadd reassoc nsz arcp contract float %1328, %.sroa.54.1, !spirv.Decorations !843
  br label %._crit_edge.13

._crit_edge.13:                                   ; preds = %.preheader.12, %1308
  %.sroa.54.2 = phi float [ %1329, %1308 ], [ %.sroa.54.1, %.preheader.12 ]
  br i1 %143, label %1330, label %._crit_edge.1.13

1330:                                             ; preds = %._crit_edge.13
  %.sroa.256.0.insert.ext848 = zext i32 %163 to i64
  %1331 = sext i32 %59 to i64
  %1332 = mul nsw i64 %1331, %const_reg_qword3, !spirv.Decorations !836
  %1333 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1332, i32 0
  %1334 = getelementptr i16, i16 addrspace(4)* %1333, i64 %.sroa.256.0.insert.ext848
  %1335 = addrspacecast i16 addrspace(4)* %1334 to i16 addrspace(1)*
  %1336 = load i16, i16 addrspace(1)* %1335, align 2
  %1337 = sext i32 %140 to i64
  %1338 = mul nsw i64 %1337, %const_reg_qword5, !spirv.Decorations !836
  %1339 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1338
  %1340 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1339, i64 %.sroa.256.0.insert.ext848
  %1341 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1340, i64 0, i32 0
  %1342 = addrspacecast i16 addrspace(4)* %1341 to i16 addrspace(1)*
  %1343 = load i16, i16 addrspace(1)* %1342, align 2
  %1344 = zext i16 %1336 to i32
  %1345 = shl nuw i32 %1344, 16, !spirv.Decorations !838
  %1346 = bitcast i32 %1345 to float
  %1347 = zext i16 %1343 to i32
  %1348 = shl nuw i32 %1347, 16, !spirv.Decorations !838
  %1349 = bitcast i32 %1348 to float
  %1350 = fmul reassoc nsz arcp contract float %1346, %1349, !spirv.Decorations !843
  %1351 = fadd reassoc nsz arcp contract float %1350, %.sroa.118.1, !spirv.Decorations !843
  br label %._crit_edge.1.13

._crit_edge.1.13:                                 ; preds = %._crit_edge.13, %1330
  %.sroa.118.2 = phi float [ %1351, %1330 ], [ %.sroa.118.1, %._crit_edge.13 ]
  br i1 %144, label %1352, label %._crit_edge.2.13

1352:                                             ; preds = %._crit_edge.1.13
  %.sroa.256.0.insert.ext853 = zext i32 %163 to i64
  %1353 = sext i32 %62 to i64
  %1354 = mul nsw i64 %1353, %const_reg_qword3, !spirv.Decorations !836
  %1355 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1354, i32 0
  %1356 = getelementptr i16, i16 addrspace(4)* %1355, i64 %.sroa.256.0.insert.ext853
  %1357 = addrspacecast i16 addrspace(4)* %1356 to i16 addrspace(1)*
  %1358 = load i16, i16 addrspace(1)* %1357, align 2
  %1359 = sext i32 %140 to i64
  %1360 = mul nsw i64 %1359, %const_reg_qword5, !spirv.Decorations !836
  %1361 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1360
  %1362 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1361, i64 %.sroa.256.0.insert.ext853
  %1363 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1362, i64 0, i32 0
  %1364 = addrspacecast i16 addrspace(4)* %1363 to i16 addrspace(1)*
  %1365 = load i16, i16 addrspace(1)* %1364, align 2
  %1366 = zext i16 %1358 to i32
  %1367 = shl nuw i32 %1366, 16, !spirv.Decorations !838
  %1368 = bitcast i32 %1367 to float
  %1369 = zext i16 %1365 to i32
  %1370 = shl nuw i32 %1369, 16, !spirv.Decorations !838
  %1371 = bitcast i32 %1370 to float
  %1372 = fmul reassoc nsz arcp contract float %1368, %1371, !spirv.Decorations !843
  %1373 = fadd reassoc nsz arcp contract float %1372, %.sroa.182.1, !spirv.Decorations !843
  br label %._crit_edge.2.13

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13, %1352
  %.sroa.182.2 = phi float [ %1373, %1352 ], [ %.sroa.182.1, %._crit_edge.1.13 ]
  br i1 %145, label %1374, label %.preheader.13

1374:                                             ; preds = %._crit_edge.2.13
  %.sroa.256.0.insert.ext858 = zext i32 %163 to i64
  %1375 = sext i32 %65 to i64
  %1376 = mul nsw i64 %1375, %const_reg_qword3, !spirv.Decorations !836
  %1377 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1376, i32 0
  %1378 = getelementptr i16, i16 addrspace(4)* %1377, i64 %.sroa.256.0.insert.ext858
  %1379 = addrspacecast i16 addrspace(4)* %1378 to i16 addrspace(1)*
  %1380 = load i16, i16 addrspace(1)* %1379, align 2
  %1381 = sext i32 %140 to i64
  %1382 = mul nsw i64 %1381, %const_reg_qword5, !spirv.Decorations !836
  %1383 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1382
  %1384 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1383, i64 %.sroa.256.0.insert.ext858
  %1385 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1384, i64 0, i32 0
  %1386 = addrspacecast i16 addrspace(4)* %1385 to i16 addrspace(1)*
  %1387 = load i16, i16 addrspace(1)* %1386, align 2
  %1388 = zext i16 %1380 to i32
  %1389 = shl nuw i32 %1388, 16, !spirv.Decorations !838
  %1390 = bitcast i32 %1389 to float
  %1391 = zext i16 %1387 to i32
  %1392 = shl nuw i32 %1391, 16, !spirv.Decorations !838
  %1393 = bitcast i32 %1392 to float
  %1394 = fmul reassoc nsz arcp contract float %1390, %1393, !spirv.Decorations !843
  %1395 = fadd reassoc nsz arcp contract float %1394, %.sroa.246.1, !spirv.Decorations !843
  br label %.preheader.13

.preheader.13:                                    ; preds = %._crit_edge.2.13, %1374
  %.sroa.246.2 = phi float [ %1395, %1374 ], [ %.sroa.246.1, %._crit_edge.2.13 ]
  br i1 %148, label %1396, label %._crit_edge.14

1396:                                             ; preds = %.preheader.13
  %.sroa.256.0.insert.ext863 = zext i32 %163 to i64
  %1397 = sext i32 %29 to i64
  %1398 = mul nsw i64 %1397, %const_reg_qword3, !spirv.Decorations !836
  %1399 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1398, i32 0
  %1400 = getelementptr i16, i16 addrspace(4)* %1399, i64 %.sroa.256.0.insert.ext863
  %1401 = addrspacecast i16 addrspace(4)* %1400 to i16 addrspace(1)*
  %1402 = load i16, i16 addrspace(1)* %1401, align 2
  %1403 = sext i32 %146 to i64
  %1404 = mul nsw i64 %1403, %const_reg_qword5, !spirv.Decorations !836
  %1405 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1404
  %1406 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1405, i64 %.sroa.256.0.insert.ext863
  %1407 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1406, i64 0, i32 0
  %1408 = addrspacecast i16 addrspace(4)* %1407 to i16 addrspace(1)*
  %1409 = load i16, i16 addrspace(1)* %1408, align 2
  %1410 = zext i16 %1402 to i32
  %1411 = shl nuw i32 %1410, 16, !spirv.Decorations !838
  %1412 = bitcast i32 %1411 to float
  %1413 = zext i16 %1409 to i32
  %1414 = shl nuw i32 %1413, 16, !spirv.Decorations !838
  %1415 = bitcast i32 %1414 to float
  %1416 = fmul reassoc nsz arcp contract float %1412, %1415, !spirv.Decorations !843
  %1417 = fadd reassoc nsz arcp contract float %1416, %.sroa.58.1, !spirv.Decorations !843
  br label %._crit_edge.14

._crit_edge.14:                                   ; preds = %.preheader.13, %1396
  %.sroa.58.2 = phi float [ %1417, %1396 ], [ %.sroa.58.1, %.preheader.13 ]
  br i1 %149, label %1418, label %._crit_edge.1.14

1418:                                             ; preds = %._crit_edge.14
  %.sroa.256.0.insert.ext868 = zext i32 %163 to i64
  %1419 = sext i32 %59 to i64
  %1420 = mul nsw i64 %1419, %const_reg_qword3, !spirv.Decorations !836
  %1421 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1420, i32 0
  %1422 = getelementptr i16, i16 addrspace(4)* %1421, i64 %.sroa.256.0.insert.ext868
  %1423 = addrspacecast i16 addrspace(4)* %1422 to i16 addrspace(1)*
  %1424 = load i16, i16 addrspace(1)* %1423, align 2
  %1425 = sext i32 %146 to i64
  %1426 = mul nsw i64 %1425, %const_reg_qword5, !spirv.Decorations !836
  %1427 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1426
  %1428 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1427, i64 %.sroa.256.0.insert.ext868
  %1429 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1428, i64 0, i32 0
  %1430 = addrspacecast i16 addrspace(4)* %1429 to i16 addrspace(1)*
  %1431 = load i16, i16 addrspace(1)* %1430, align 2
  %1432 = zext i16 %1424 to i32
  %1433 = shl nuw i32 %1432, 16, !spirv.Decorations !838
  %1434 = bitcast i32 %1433 to float
  %1435 = zext i16 %1431 to i32
  %1436 = shl nuw i32 %1435, 16, !spirv.Decorations !838
  %1437 = bitcast i32 %1436 to float
  %1438 = fmul reassoc nsz arcp contract float %1434, %1437, !spirv.Decorations !843
  %1439 = fadd reassoc nsz arcp contract float %1438, %.sroa.122.1, !spirv.Decorations !843
  br label %._crit_edge.1.14

._crit_edge.1.14:                                 ; preds = %._crit_edge.14, %1418
  %.sroa.122.2 = phi float [ %1439, %1418 ], [ %.sroa.122.1, %._crit_edge.14 ]
  br i1 %150, label %1440, label %._crit_edge.2.14

1440:                                             ; preds = %._crit_edge.1.14
  %.sroa.256.0.insert.ext873 = zext i32 %163 to i64
  %1441 = sext i32 %62 to i64
  %1442 = mul nsw i64 %1441, %const_reg_qword3, !spirv.Decorations !836
  %1443 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1442, i32 0
  %1444 = getelementptr i16, i16 addrspace(4)* %1443, i64 %.sroa.256.0.insert.ext873
  %1445 = addrspacecast i16 addrspace(4)* %1444 to i16 addrspace(1)*
  %1446 = load i16, i16 addrspace(1)* %1445, align 2
  %1447 = sext i32 %146 to i64
  %1448 = mul nsw i64 %1447, %const_reg_qword5, !spirv.Decorations !836
  %1449 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1448
  %1450 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1449, i64 %.sroa.256.0.insert.ext873
  %1451 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1450, i64 0, i32 0
  %1452 = addrspacecast i16 addrspace(4)* %1451 to i16 addrspace(1)*
  %1453 = load i16, i16 addrspace(1)* %1452, align 2
  %1454 = zext i16 %1446 to i32
  %1455 = shl nuw i32 %1454, 16, !spirv.Decorations !838
  %1456 = bitcast i32 %1455 to float
  %1457 = zext i16 %1453 to i32
  %1458 = shl nuw i32 %1457, 16, !spirv.Decorations !838
  %1459 = bitcast i32 %1458 to float
  %1460 = fmul reassoc nsz arcp contract float %1456, %1459, !spirv.Decorations !843
  %1461 = fadd reassoc nsz arcp contract float %1460, %.sroa.186.1, !spirv.Decorations !843
  br label %._crit_edge.2.14

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14, %1440
  %.sroa.186.2 = phi float [ %1461, %1440 ], [ %.sroa.186.1, %._crit_edge.1.14 ]
  br i1 %151, label %1462, label %.preheader.14

1462:                                             ; preds = %._crit_edge.2.14
  %.sroa.256.0.insert.ext878 = zext i32 %163 to i64
  %1463 = sext i32 %65 to i64
  %1464 = mul nsw i64 %1463, %const_reg_qword3, !spirv.Decorations !836
  %1465 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1464, i32 0
  %1466 = getelementptr i16, i16 addrspace(4)* %1465, i64 %.sroa.256.0.insert.ext878
  %1467 = addrspacecast i16 addrspace(4)* %1466 to i16 addrspace(1)*
  %1468 = load i16, i16 addrspace(1)* %1467, align 2
  %1469 = sext i32 %146 to i64
  %1470 = mul nsw i64 %1469, %const_reg_qword5, !spirv.Decorations !836
  %1471 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1470
  %1472 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1471, i64 %.sroa.256.0.insert.ext878
  %1473 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1472, i64 0, i32 0
  %1474 = addrspacecast i16 addrspace(4)* %1473 to i16 addrspace(1)*
  %1475 = load i16, i16 addrspace(1)* %1474, align 2
  %1476 = zext i16 %1468 to i32
  %1477 = shl nuw i32 %1476, 16, !spirv.Decorations !838
  %1478 = bitcast i32 %1477 to float
  %1479 = zext i16 %1475 to i32
  %1480 = shl nuw i32 %1479, 16, !spirv.Decorations !838
  %1481 = bitcast i32 %1480 to float
  %1482 = fmul reassoc nsz arcp contract float %1478, %1481, !spirv.Decorations !843
  %1483 = fadd reassoc nsz arcp contract float %1482, %.sroa.250.1, !spirv.Decorations !843
  br label %.preheader.14

.preheader.14:                                    ; preds = %._crit_edge.2.14, %1462
  %.sroa.250.2 = phi float [ %1483, %1462 ], [ %.sroa.250.1, %._crit_edge.2.14 ]
  br i1 %154, label %1484, label %._crit_edge.15

1484:                                             ; preds = %.preheader.14
  %.sroa.256.0.insert.ext883 = zext i32 %163 to i64
  %1485 = sext i32 %29 to i64
  %1486 = mul nsw i64 %1485, %const_reg_qword3, !spirv.Decorations !836
  %1487 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1486, i32 0
  %1488 = getelementptr i16, i16 addrspace(4)* %1487, i64 %.sroa.256.0.insert.ext883
  %1489 = addrspacecast i16 addrspace(4)* %1488 to i16 addrspace(1)*
  %1490 = load i16, i16 addrspace(1)* %1489, align 2
  %1491 = sext i32 %152 to i64
  %1492 = mul nsw i64 %1491, %const_reg_qword5, !spirv.Decorations !836
  %1493 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1492
  %1494 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1493, i64 %.sroa.256.0.insert.ext883
  %1495 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1494, i64 0, i32 0
  %1496 = addrspacecast i16 addrspace(4)* %1495 to i16 addrspace(1)*
  %1497 = load i16, i16 addrspace(1)* %1496, align 2
  %1498 = zext i16 %1490 to i32
  %1499 = shl nuw i32 %1498, 16, !spirv.Decorations !838
  %1500 = bitcast i32 %1499 to float
  %1501 = zext i16 %1497 to i32
  %1502 = shl nuw i32 %1501, 16, !spirv.Decorations !838
  %1503 = bitcast i32 %1502 to float
  %1504 = fmul reassoc nsz arcp contract float %1500, %1503, !spirv.Decorations !843
  %1505 = fadd reassoc nsz arcp contract float %1504, %.sroa.62.1, !spirv.Decorations !843
  br label %._crit_edge.15

._crit_edge.15:                                   ; preds = %.preheader.14, %1484
  %.sroa.62.2 = phi float [ %1505, %1484 ], [ %.sroa.62.1, %.preheader.14 ]
  br i1 %155, label %1506, label %._crit_edge.1.15

1506:                                             ; preds = %._crit_edge.15
  %.sroa.256.0.insert.ext888 = zext i32 %163 to i64
  %1507 = sext i32 %59 to i64
  %1508 = mul nsw i64 %1507, %const_reg_qword3, !spirv.Decorations !836
  %1509 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1508, i32 0
  %1510 = getelementptr i16, i16 addrspace(4)* %1509, i64 %.sroa.256.0.insert.ext888
  %1511 = addrspacecast i16 addrspace(4)* %1510 to i16 addrspace(1)*
  %1512 = load i16, i16 addrspace(1)* %1511, align 2
  %1513 = sext i32 %152 to i64
  %1514 = mul nsw i64 %1513, %const_reg_qword5, !spirv.Decorations !836
  %1515 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1514
  %1516 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1515, i64 %.sroa.256.0.insert.ext888
  %1517 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1516, i64 0, i32 0
  %1518 = addrspacecast i16 addrspace(4)* %1517 to i16 addrspace(1)*
  %1519 = load i16, i16 addrspace(1)* %1518, align 2
  %1520 = zext i16 %1512 to i32
  %1521 = shl nuw i32 %1520, 16, !spirv.Decorations !838
  %1522 = bitcast i32 %1521 to float
  %1523 = zext i16 %1519 to i32
  %1524 = shl nuw i32 %1523, 16, !spirv.Decorations !838
  %1525 = bitcast i32 %1524 to float
  %1526 = fmul reassoc nsz arcp contract float %1522, %1525, !spirv.Decorations !843
  %1527 = fadd reassoc nsz arcp contract float %1526, %.sroa.126.1, !spirv.Decorations !843
  br label %._crit_edge.1.15

._crit_edge.1.15:                                 ; preds = %._crit_edge.15, %1506
  %.sroa.126.2 = phi float [ %1527, %1506 ], [ %.sroa.126.1, %._crit_edge.15 ]
  br i1 %156, label %1528, label %._crit_edge.2.15

1528:                                             ; preds = %._crit_edge.1.15
  %.sroa.256.0.insert.ext893 = zext i32 %163 to i64
  %1529 = sext i32 %62 to i64
  %1530 = mul nsw i64 %1529, %const_reg_qword3, !spirv.Decorations !836
  %1531 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1530, i32 0
  %1532 = getelementptr i16, i16 addrspace(4)* %1531, i64 %.sroa.256.0.insert.ext893
  %1533 = addrspacecast i16 addrspace(4)* %1532 to i16 addrspace(1)*
  %1534 = load i16, i16 addrspace(1)* %1533, align 2
  %1535 = sext i32 %152 to i64
  %1536 = mul nsw i64 %1535, %const_reg_qword5, !spirv.Decorations !836
  %1537 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1536
  %1538 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1537, i64 %.sroa.256.0.insert.ext893
  %1539 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1538, i64 0, i32 0
  %1540 = addrspacecast i16 addrspace(4)* %1539 to i16 addrspace(1)*
  %1541 = load i16, i16 addrspace(1)* %1540, align 2
  %1542 = zext i16 %1534 to i32
  %1543 = shl nuw i32 %1542, 16, !spirv.Decorations !838
  %1544 = bitcast i32 %1543 to float
  %1545 = zext i16 %1541 to i32
  %1546 = shl nuw i32 %1545, 16, !spirv.Decorations !838
  %1547 = bitcast i32 %1546 to float
  %1548 = fmul reassoc nsz arcp contract float %1544, %1547, !spirv.Decorations !843
  %1549 = fadd reassoc nsz arcp contract float %1548, %.sroa.190.1, !spirv.Decorations !843
  br label %._crit_edge.2.15

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15, %1528
  %.sroa.190.2 = phi float [ %1549, %1528 ], [ %.sroa.190.1, %._crit_edge.1.15 ]
  br i1 %157, label %1550, label %.preheader.15

1550:                                             ; preds = %._crit_edge.2.15
  %.sroa.256.0.insert.ext898 = zext i32 %163 to i64
  %1551 = sext i32 %65 to i64
  %1552 = mul nsw i64 %1551, %const_reg_qword3, !spirv.Decorations !836
  %1553 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1552, i32 0
  %1554 = getelementptr i16, i16 addrspace(4)* %1553, i64 %.sroa.256.0.insert.ext898
  %1555 = addrspacecast i16 addrspace(4)* %1554 to i16 addrspace(1)*
  %1556 = load i16, i16 addrspace(1)* %1555, align 2
  %1557 = sext i32 %152 to i64
  %1558 = mul nsw i64 %1557, %const_reg_qword5, !spirv.Decorations !836
  %1559 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1558
  %1560 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1559, i64 %.sroa.256.0.insert.ext898
  %1561 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %1560, i64 0, i32 0
  %1562 = addrspacecast i16 addrspace(4)* %1561 to i16 addrspace(1)*
  %1563 = load i16, i16 addrspace(1)* %1562, align 2
  %1564 = zext i16 %1556 to i32
  %1565 = shl nuw i32 %1564, 16, !spirv.Decorations !838
  %1566 = bitcast i32 %1565 to float
  %1567 = zext i16 %1563 to i32
  %1568 = shl nuw i32 %1567, 16, !spirv.Decorations !838
  %1569 = bitcast i32 %1568 to float
  %1570 = fmul reassoc nsz arcp contract float %1566, %1569, !spirv.Decorations !843
  %1571 = fadd reassoc nsz arcp contract float %1570, %.sroa.254.1, !spirv.Decorations !843
  br label %.preheader.15

.preheader.15:                                    ; preds = %._crit_edge.2.15, %1550
  %.sroa.254.2 = phi float [ %1571, %1550 ], [ %.sroa.254.1, %._crit_edge.2.15 ]
  %1572 = add nuw nsw i32 %163, 1, !spirv.Decorations !848
  %1573 = icmp slt i32 %1572, %const_reg_dword2
  br i1 %1573, label %.preheader.preheader, label %.preheader1.preheader

1574:                                             ; preds = %.preheader1.preheader
  %1575 = sext i32 %29 to i64
  %1576 = sext i32 %35 to i64
  %1577 = mul nsw i64 %1575, %const_reg_qword9, !spirv.Decorations !836
  %1578 = add nsw i64 %1577, %1576, !spirv.Decorations !836
  %1579 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !843
  br i1 %42, label %1580, label %1590

1580:                                             ; preds = %1574
  %1581 = mul nsw i64 %1575, %const_reg_qword7, !spirv.Decorations !836
  %1582 = getelementptr float, float addrspace(4)* %160, i64 %1581
  %1583 = getelementptr float, float addrspace(4)* %1582, i64 %1576
  %1584 = addrspacecast float addrspace(4)* %1583 to float addrspace(1)*
  %1585 = load float, float addrspace(1)* %1584, align 4
  %1586 = fmul reassoc nsz arcp contract float %1585, %4, !spirv.Decorations !843
  %1587 = fadd reassoc nsz arcp contract float %1579, %1586, !spirv.Decorations !843
  %1588 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1578
  %1589 = addrspacecast float addrspace(4)* %1588 to float addrspace(1)*
  store float %1587, float addrspace(1)* %1589, align 4
  br label %._crit_edge70

1590:                                             ; preds = %1574
  %1591 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1578
  %1592 = addrspacecast float addrspace(4)* %1591 to float addrspace(1)*
  store float %1579, float addrspace(1)* %1592, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %.preheader1.preheader, %1590, %1580
  br i1 %61, label %1593, label %._crit_edge70.1

1593:                                             ; preds = %._crit_edge70
  %1594 = sext i32 %59 to i64
  %1595 = sext i32 %35 to i64
  %1596 = mul nsw i64 %1594, %const_reg_qword9, !spirv.Decorations !836
  %1597 = add nsw i64 %1596, %1595, !spirv.Decorations !836
  %1598 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !843
  br i1 %42, label %1602, label %1599

1599:                                             ; preds = %1593
  %1600 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1597
  %1601 = addrspacecast float addrspace(4)* %1600 to float addrspace(1)*
  store float %1598, float addrspace(1)* %1601, align 4
  br label %._crit_edge70.1

1602:                                             ; preds = %1593
  %1603 = mul nsw i64 %1594, %const_reg_qword7, !spirv.Decorations !836
  %1604 = getelementptr float, float addrspace(4)* %160, i64 %1603
  %1605 = getelementptr float, float addrspace(4)* %1604, i64 %1595
  %1606 = addrspacecast float addrspace(4)* %1605 to float addrspace(1)*
  %1607 = load float, float addrspace(1)* %1606, align 4
  %1608 = fmul reassoc nsz arcp contract float %1607, %4, !spirv.Decorations !843
  %1609 = fadd reassoc nsz arcp contract float %1598, %1608, !spirv.Decorations !843
  %1610 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1597
  %1611 = addrspacecast float addrspace(4)* %1610 to float addrspace(1)*
  store float %1609, float addrspace(1)* %1611, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %._crit_edge70, %1602, %1599
  br i1 %64, label %1612, label %._crit_edge70.2

1612:                                             ; preds = %._crit_edge70.1
  %1613 = sext i32 %62 to i64
  %1614 = sext i32 %35 to i64
  %1615 = mul nsw i64 %1613, %const_reg_qword9, !spirv.Decorations !836
  %1616 = add nsw i64 %1615, %1614, !spirv.Decorations !836
  %1617 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !843
  br i1 %42, label %1621, label %1618

1618:                                             ; preds = %1612
  %1619 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1616
  %1620 = addrspacecast float addrspace(4)* %1619 to float addrspace(1)*
  store float %1617, float addrspace(1)* %1620, align 4
  br label %._crit_edge70.2

1621:                                             ; preds = %1612
  %1622 = mul nsw i64 %1613, %const_reg_qword7, !spirv.Decorations !836
  %1623 = getelementptr float, float addrspace(4)* %160, i64 %1622
  %1624 = getelementptr float, float addrspace(4)* %1623, i64 %1614
  %1625 = addrspacecast float addrspace(4)* %1624 to float addrspace(1)*
  %1626 = load float, float addrspace(1)* %1625, align 4
  %1627 = fmul reassoc nsz arcp contract float %1626, %4, !spirv.Decorations !843
  %1628 = fadd reassoc nsz arcp contract float %1617, %1627, !spirv.Decorations !843
  %1629 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1616
  %1630 = addrspacecast float addrspace(4)* %1629 to float addrspace(1)*
  store float %1628, float addrspace(1)* %1630, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %._crit_edge70.1, %1621, %1618
  br i1 %67, label %1631, label %.preheader1

1631:                                             ; preds = %._crit_edge70.2
  %1632 = sext i32 %65 to i64
  %1633 = sext i32 %35 to i64
  %1634 = mul nsw i64 %1632, %const_reg_qword9, !spirv.Decorations !836
  %1635 = add nsw i64 %1634, %1633, !spirv.Decorations !836
  %1636 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !843
  br i1 %42, label %1640, label %1637

1637:                                             ; preds = %1631
  %1638 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1635
  %1639 = addrspacecast float addrspace(4)* %1638 to float addrspace(1)*
  store float %1636, float addrspace(1)* %1639, align 4
  br label %.preheader1

1640:                                             ; preds = %1631
  %1641 = mul nsw i64 %1632, %const_reg_qword7, !spirv.Decorations !836
  %1642 = getelementptr float, float addrspace(4)* %160, i64 %1641
  %1643 = getelementptr float, float addrspace(4)* %1642, i64 %1633
  %1644 = addrspacecast float addrspace(4)* %1643 to float addrspace(1)*
  %1645 = load float, float addrspace(1)* %1644, align 4
  %1646 = fmul reassoc nsz arcp contract float %1645, %4, !spirv.Decorations !843
  %1647 = fadd reassoc nsz arcp contract float %1636, %1646, !spirv.Decorations !843
  %1648 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1635
  %1649 = addrspacecast float addrspace(4)* %1648 to float addrspace(1)*
  store float %1647, float addrspace(1)* %1649, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %._crit_edge70.2, %1640, %1637
  br i1 %70, label %1650, label %._crit_edge70.176

1650:                                             ; preds = %.preheader1
  %1651 = sext i32 %29 to i64
  %1652 = sext i32 %68 to i64
  %1653 = mul nsw i64 %1651, %const_reg_qword9, !spirv.Decorations !836
  %1654 = add nsw i64 %1653, %1652, !spirv.Decorations !836
  %1655 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !843
  br i1 %42, label %1659, label %1656

1656:                                             ; preds = %1650
  %1657 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1654
  %1658 = addrspacecast float addrspace(4)* %1657 to float addrspace(1)*
  store float %1655, float addrspace(1)* %1658, align 4
  br label %._crit_edge70.176

1659:                                             ; preds = %1650
  %1660 = mul nsw i64 %1651, %const_reg_qword7, !spirv.Decorations !836
  %1661 = getelementptr float, float addrspace(4)* %160, i64 %1660
  %1662 = getelementptr float, float addrspace(4)* %1661, i64 %1652
  %1663 = addrspacecast float addrspace(4)* %1662 to float addrspace(1)*
  %1664 = load float, float addrspace(1)* %1663, align 4
  %1665 = fmul reassoc nsz arcp contract float %1664, %4, !spirv.Decorations !843
  %1666 = fadd reassoc nsz arcp contract float %1655, %1665, !spirv.Decorations !843
  %1667 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1654
  %1668 = addrspacecast float addrspace(4)* %1667 to float addrspace(1)*
  store float %1666, float addrspace(1)* %1668, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %.preheader1, %1659, %1656
  br i1 %71, label %1669, label %._crit_edge70.1.1

1669:                                             ; preds = %._crit_edge70.176
  %1670 = sext i32 %59 to i64
  %1671 = sext i32 %68 to i64
  %1672 = mul nsw i64 %1670, %const_reg_qword9, !spirv.Decorations !836
  %1673 = add nsw i64 %1672, %1671, !spirv.Decorations !836
  %1674 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !843
  br i1 %42, label %1678, label %1675

1675:                                             ; preds = %1669
  %1676 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1673
  %1677 = addrspacecast float addrspace(4)* %1676 to float addrspace(1)*
  store float %1674, float addrspace(1)* %1677, align 4
  br label %._crit_edge70.1.1

1678:                                             ; preds = %1669
  %1679 = mul nsw i64 %1670, %const_reg_qword7, !spirv.Decorations !836
  %1680 = getelementptr float, float addrspace(4)* %160, i64 %1679
  %1681 = getelementptr float, float addrspace(4)* %1680, i64 %1671
  %1682 = addrspacecast float addrspace(4)* %1681 to float addrspace(1)*
  %1683 = load float, float addrspace(1)* %1682, align 4
  %1684 = fmul reassoc nsz arcp contract float %1683, %4, !spirv.Decorations !843
  %1685 = fadd reassoc nsz arcp contract float %1674, %1684, !spirv.Decorations !843
  %1686 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1673
  %1687 = addrspacecast float addrspace(4)* %1686 to float addrspace(1)*
  store float %1685, float addrspace(1)* %1687, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176, %1678, %1675
  br i1 %72, label %1688, label %._crit_edge70.2.1

1688:                                             ; preds = %._crit_edge70.1.1
  %1689 = sext i32 %62 to i64
  %1690 = sext i32 %68 to i64
  %1691 = mul nsw i64 %1689, %const_reg_qword9, !spirv.Decorations !836
  %1692 = add nsw i64 %1691, %1690, !spirv.Decorations !836
  %1693 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !843
  br i1 %42, label %1697, label %1694

1694:                                             ; preds = %1688
  %1695 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1692
  %1696 = addrspacecast float addrspace(4)* %1695 to float addrspace(1)*
  store float %1693, float addrspace(1)* %1696, align 4
  br label %._crit_edge70.2.1

1697:                                             ; preds = %1688
  %1698 = mul nsw i64 %1689, %const_reg_qword7, !spirv.Decorations !836
  %1699 = getelementptr float, float addrspace(4)* %160, i64 %1698
  %1700 = getelementptr float, float addrspace(4)* %1699, i64 %1690
  %1701 = addrspacecast float addrspace(4)* %1700 to float addrspace(1)*
  %1702 = load float, float addrspace(1)* %1701, align 4
  %1703 = fmul reassoc nsz arcp contract float %1702, %4, !spirv.Decorations !843
  %1704 = fadd reassoc nsz arcp contract float %1693, %1703, !spirv.Decorations !843
  %1705 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1692
  %1706 = addrspacecast float addrspace(4)* %1705 to float addrspace(1)*
  store float %1704, float addrspace(1)* %1706, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1, %1697, %1694
  br i1 %73, label %1707, label %.preheader1.1

1707:                                             ; preds = %._crit_edge70.2.1
  %1708 = sext i32 %65 to i64
  %1709 = sext i32 %68 to i64
  %1710 = mul nsw i64 %1708, %const_reg_qword9, !spirv.Decorations !836
  %1711 = add nsw i64 %1710, %1709, !spirv.Decorations !836
  %1712 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !843
  br i1 %42, label %1716, label %1713

1713:                                             ; preds = %1707
  %1714 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1711
  %1715 = addrspacecast float addrspace(4)* %1714 to float addrspace(1)*
  store float %1712, float addrspace(1)* %1715, align 4
  br label %.preheader1.1

1716:                                             ; preds = %1707
  %1717 = mul nsw i64 %1708, %const_reg_qword7, !spirv.Decorations !836
  %1718 = getelementptr float, float addrspace(4)* %160, i64 %1717
  %1719 = getelementptr float, float addrspace(4)* %1718, i64 %1709
  %1720 = addrspacecast float addrspace(4)* %1719 to float addrspace(1)*
  %1721 = load float, float addrspace(1)* %1720, align 4
  %1722 = fmul reassoc nsz arcp contract float %1721, %4, !spirv.Decorations !843
  %1723 = fadd reassoc nsz arcp contract float %1712, %1722, !spirv.Decorations !843
  %1724 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1711
  %1725 = addrspacecast float addrspace(4)* %1724 to float addrspace(1)*
  store float %1723, float addrspace(1)* %1725, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %._crit_edge70.2.1, %1716, %1713
  br i1 %76, label %1726, label %._crit_edge70.277

1726:                                             ; preds = %.preheader1.1
  %1727 = sext i32 %29 to i64
  %1728 = sext i32 %74 to i64
  %1729 = mul nsw i64 %1727, %const_reg_qword9, !spirv.Decorations !836
  %1730 = add nsw i64 %1729, %1728, !spirv.Decorations !836
  %1731 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !843
  br i1 %42, label %1735, label %1732

1732:                                             ; preds = %1726
  %1733 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1730
  %1734 = addrspacecast float addrspace(4)* %1733 to float addrspace(1)*
  store float %1731, float addrspace(1)* %1734, align 4
  br label %._crit_edge70.277

1735:                                             ; preds = %1726
  %1736 = mul nsw i64 %1727, %const_reg_qword7, !spirv.Decorations !836
  %1737 = getelementptr float, float addrspace(4)* %160, i64 %1736
  %1738 = getelementptr float, float addrspace(4)* %1737, i64 %1728
  %1739 = addrspacecast float addrspace(4)* %1738 to float addrspace(1)*
  %1740 = load float, float addrspace(1)* %1739, align 4
  %1741 = fmul reassoc nsz arcp contract float %1740, %4, !spirv.Decorations !843
  %1742 = fadd reassoc nsz arcp contract float %1731, %1741, !spirv.Decorations !843
  %1743 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1730
  %1744 = addrspacecast float addrspace(4)* %1743 to float addrspace(1)*
  store float %1742, float addrspace(1)* %1744, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %.preheader1.1, %1735, %1732
  br i1 %77, label %1745, label %._crit_edge70.1.2

1745:                                             ; preds = %._crit_edge70.277
  %1746 = sext i32 %59 to i64
  %1747 = sext i32 %74 to i64
  %1748 = mul nsw i64 %1746, %const_reg_qword9, !spirv.Decorations !836
  %1749 = add nsw i64 %1748, %1747, !spirv.Decorations !836
  %1750 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !843
  br i1 %42, label %1754, label %1751

1751:                                             ; preds = %1745
  %1752 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1749
  %1753 = addrspacecast float addrspace(4)* %1752 to float addrspace(1)*
  store float %1750, float addrspace(1)* %1753, align 4
  br label %._crit_edge70.1.2

1754:                                             ; preds = %1745
  %1755 = mul nsw i64 %1746, %const_reg_qword7, !spirv.Decorations !836
  %1756 = getelementptr float, float addrspace(4)* %160, i64 %1755
  %1757 = getelementptr float, float addrspace(4)* %1756, i64 %1747
  %1758 = addrspacecast float addrspace(4)* %1757 to float addrspace(1)*
  %1759 = load float, float addrspace(1)* %1758, align 4
  %1760 = fmul reassoc nsz arcp contract float %1759, %4, !spirv.Decorations !843
  %1761 = fadd reassoc nsz arcp contract float %1750, %1760, !spirv.Decorations !843
  %1762 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1749
  %1763 = addrspacecast float addrspace(4)* %1762 to float addrspace(1)*
  store float %1761, float addrspace(1)* %1763, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277, %1754, %1751
  br i1 %78, label %1764, label %._crit_edge70.2.2

1764:                                             ; preds = %._crit_edge70.1.2
  %1765 = sext i32 %62 to i64
  %1766 = sext i32 %74 to i64
  %1767 = mul nsw i64 %1765, %const_reg_qword9, !spirv.Decorations !836
  %1768 = add nsw i64 %1767, %1766, !spirv.Decorations !836
  %1769 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !843
  br i1 %42, label %1773, label %1770

1770:                                             ; preds = %1764
  %1771 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1768
  %1772 = addrspacecast float addrspace(4)* %1771 to float addrspace(1)*
  store float %1769, float addrspace(1)* %1772, align 4
  br label %._crit_edge70.2.2

1773:                                             ; preds = %1764
  %1774 = mul nsw i64 %1765, %const_reg_qword7, !spirv.Decorations !836
  %1775 = getelementptr float, float addrspace(4)* %160, i64 %1774
  %1776 = getelementptr float, float addrspace(4)* %1775, i64 %1766
  %1777 = addrspacecast float addrspace(4)* %1776 to float addrspace(1)*
  %1778 = load float, float addrspace(1)* %1777, align 4
  %1779 = fmul reassoc nsz arcp contract float %1778, %4, !spirv.Decorations !843
  %1780 = fadd reassoc nsz arcp contract float %1769, %1779, !spirv.Decorations !843
  %1781 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1768
  %1782 = addrspacecast float addrspace(4)* %1781 to float addrspace(1)*
  store float %1780, float addrspace(1)* %1782, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2, %1773, %1770
  br i1 %79, label %1783, label %.preheader1.2

1783:                                             ; preds = %._crit_edge70.2.2
  %1784 = sext i32 %65 to i64
  %1785 = sext i32 %74 to i64
  %1786 = mul nsw i64 %1784, %const_reg_qword9, !spirv.Decorations !836
  %1787 = add nsw i64 %1786, %1785, !spirv.Decorations !836
  %1788 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !843
  br i1 %42, label %1792, label %1789

1789:                                             ; preds = %1783
  %1790 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1787
  %1791 = addrspacecast float addrspace(4)* %1790 to float addrspace(1)*
  store float %1788, float addrspace(1)* %1791, align 4
  br label %.preheader1.2

1792:                                             ; preds = %1783
  %1793 = mul nsw i64 %1784, %const_reg_qword7, !spirv.Decorations !836
  %1794 = getelementptr float, float addrspace(4)* %160, i64 %1793
  %1795 = getelementptr float, float addrspace(4)* %1794, i64 %1785
  %1796 = addrspacecast float addrspace(4)* %1795 to float addrspace(1)*
  %1797 = load float, float addrspace(1)* %1796, align 4
  %1798 = fmul reassoc nsz arcp contract float %1797, %4, !spirv.Decorations !843
  %1799 = fadd reassoc nsz arcp contract float %1788, %1798, !spirv.Decorations !843
  %1800 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1787
  %1801 = addrspacecast float addrspace(4)* %1800 to float addrspace(1)*
  store float %1799, float addrspace(1)* %1801, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %._crit_edge70.2.2, %1792, %1789
  br i1 %82, label %1802, label %._crit_edge70.378

1802:                                             ; preds = %.preheader1.2
  %1803 = sext i32 %29 to i64
  %1804 = sext i32 %80 to i64
  %1805 = mul nsw i64 %1803, %const_reg_qword9, !spirv.Decorations !836
  %1806 = add nsw i64 %1805, %1804, !spirv.Decorations !836
  %1807 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !843
  br i1 %42, label %1811, label %1808

1808:                                             ; preds = %1802
  %1809 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1806
  %1810 = addrspacecast float addrspace(4)* %1809 to float addrspace(1)*
  store float %1807, float addrspace(1)* %1810, align 4
  br label %._crit_edge70.378

1811:                                             ; preds = %1802
  %1812 = mul nsw i64 %1803, %const_reg_qword7, !spirv.Decorations !836
  %1813 = getelementptr float, float addrspace(4)* %160, i64 %1812
  %1814 = getelementptr float, float addrspace(4)* %1813, i64 %1804
  %1815 = addrspacecast float addrspace(4)* %1814 to float addrspace(1)*
  %1816 = load float, float addrspace(1)* %1815, align 4
  %1817 = fmul reassoc nsz arcp contract float %1816, %4, !spirv.Decorations !843
  %1818 = fadd reassoc nsz arcp contract float %1807, %1817, !spirv.Decorations !843
  %1819 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1806
  %1820 = addrspacecast float addrspace(4)* %1819 to float addrspace(1)*
  store float %1818, float addrspace(1)* %1820, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %.preheader1.2, %1811, %1808
  br i1 %83, label %1821, label %._crit_edge70.1.3

1821:                                             ; preds = %._crit_edge70.378
  %1822 = sext i32 %59 to i64
  %1823 = sext i32 %80 to i64
  %1824 = mul nsw i64 %1822, %const_reg_qword9, !spirv.Decorations !836
  %1825 = add nsw i64 %1824, %1823, !spirv.Decorations !836
  %1826 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !843
  br i1 %42, label %1830, label %1827

1827:                                             ; preds = %1821
  %1828 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1825
  %1829 = addrspacecast float addrspace(4)* %1828 to float addrspace(1)*
  store float %1826, float addrspace(1)* %1829, align 4
  br label %._crit_edge70.1.3

1830:                                             ; preds = %1821
  %1831 = mul nsw i64 %1822, %const_reg_qword7, !spirv.Decorations !836
  %1832 = getelementptr float, float addrspace(4)* %160, i64 %1831
  %1833 = getelementptr float, float addrspace(4)* %1832, i64 %1823
  %1834 = addrspacecast float addrspace(4)* %1833 to float addrspace(1)*
  %1835 = load float, float addrspace(1)* %1834, align 4
  %1836 = fmul reassoc nsz arcp contract float %1835, %4, !spirv.Decorations !843
  %1837 = fadd reassoc nsz arcp contract float %1826, %1836, !spirv.Decorations !843
  %1838 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1825
  %1839 = addrspacecast float addrspace(4)* %1838 to float addrspace(1)*
  store float %1837, float addrspace(1)* %1839, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378, %1830, %1827
  br i1 %84, label %1840, label %._crit_edge70.2.3

1840:                                             ; preds = %._crit_edge70.1.3
  %1841 = sext i32 %62 to i64
  %1842 = sext i32 %80 to i64
  %1843 = mul nsw i64 %1841, %const_reg_qword9, !spirv.Decorations !836
  %1844 = add nsw i64 %1843, %1842, !spirv.Decorations !836
  %1845 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !843
  br i1 %42, label %1849, label %1846

1846:                                             ; preds = %1840
  %1847 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1844
  %1848 = addrspacecast float addrspace(4)* %1847 to float addrspace(1)*
  store float %1845, float addrspace(1)* %1848, align 4
  br label %._crit_edge70.2.3

1849:                                             ; preds = %1840
  %1850 = mul nsw i64 %1841, %const_reg_qword7, !spirv.Decorations !836
  %1851 = getelementptr float, float addrspace(4)* %160, i64 %1850
  %1852 = getelementptr float, float addrspace(4)* %1851, i64 %1842
  %1853 = addrspacecast float addrspace(4)* %1852 to float addrspace(1)*
  %1854 = load float, float addrspace(1)* %1853, align 4
  %1855 = fmul reassoc nsz arcp contract float %1854, %4, !spirv.Decorations !843
  %1856 = fadd reassoc nsz arcp contract float %1845, %1855, !spirv.Decorations !843
  %1857 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1844
  %1858 = addrspacecast float addrspace(4)* %1857 to float addrspace(1)*
  store float %1856, float addrspace(1)* %1858, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3, %1849, %1846
  br i1 %85, label %1859, label %.preheader1.3

1859:                                             ; preds = %._crit_edge70.2.3
  %1860 = sext i32 %65 to i64
  %1861 = sext i32 %80 to i64
  %1862 = mul nsw i64 %1860, %const_reg_qword9, !spirv.Decorations !836
  %1863 = add nsw i64 %1862, %1861, !spirv.Decorations !836
  %1864 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !843
  br i1 %42, label %1868, label %1865

1865:                                             ; preds = %1859
  %1866 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1863
  %1867 = addrspacecast float addrspace(4)* %1866 to float addrspace(1)*
  store float %1864, float addrspace(1)* %1867, align 4
  br label %.preheader1.3

1868:                                             ; preds = %1859
  %1869 = mul nsw i64 %1860, %const_reg_qword7, !spirv.Decorations !836
  %1870 = getelementptr float, float addrspace(4)* %160, i64 %1869
  %1871 = getelementptr float, float addrspace(4)* %1870, i64 %1861
  %1872 = addrspacecast float addrspace(4)* %1871 to float addrspace(1)*
  %1873 = load float, float addrspace(1)* %1872, align 4
  %1874 = fmul reassoc nsz arcp contract float %1873, %4, !spirv.Decorations !843
  %1875 = fadd reassoc nsz arcp contract float %1864, %1874, !spirv.Decorations !843
  %1876 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1863
  %1877 = addrspacecast float addrspace(4)* %1876 to float addrspace(1)*
  store float %1875, float addrspace(1)* %1877, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %._crit_edge70.2.3, %1868, %1865
  br i1 %88, label %1878, label %._crit_edge70.4

1878:                                             ; preds = %.preheader1.3
  %1879 = sext i32 %29 to i64
  %1880 = sext i32 %86 to i64
  %1881 = mul nsw i64 %1879, %const_reg_qword9, !spirv.Decorations !836
  %1882 = add nsw i64 %1881, %1880, !spirv.Decorations !836
  %1883 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !843
  br i1 %42, label %1887, label %1884

1884:                                             ; preds = %1878
  %1885 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1882
  %1886 = addrspacecast float addrspace(4)* %1885 to float addrspace(1)*
  store float %1883, float addrspace(1)* %1886, align 4
  br label %._crit_edge70.4

1887:                                             ; preds = %1878
  %1888 = mul nsw i64 %1879, %const_reg_qword7, !spirv.Decorations !836
  %1889 = getelementptr float, float addrspace(4)* %160, i64 %1888
  %1890 = getelementptr float, float addrspace(4)* %1889, i64 %1880
  %1891 = addrspacecast float addrspace(4)* %1890 to float addrspace(1)*
  %1892 = load float, float addrspace(1)* %1891, align 4
  %1893 = fmul reassoc nsz arcp contract float %1892, %4, !spirv.Decorations !843
  %1894 = fadd reassoc nsz arcp contract float %1883, %1893, !spirv.Decorations !843
  %1895 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1882
  %1896 = addrspacecast float addrspace(4)* %1895 to float addrspace(1)*
  store float %1894, float addrspace(1)* %1896, align 4
  br label %._crit_edge70.4

._crit_edge70.4:                                  ; preds = %.preheader1.3, %1887, %1884
  br i1 %89, label %1897, label %._crit_edge70.1.4

1897:                                             ; preds = %._crit_edge70.4
  %1898 = sext i32 %59 to i64
  %1899 = sext i32 %86 to i64
  %1900 = mul nsw i64 %1898, %const_reg_qword9, !spirv.Decorations !836
  %1901 = add nsw i64 %1900, %1899, !spirv.Decorations !836
  %1902 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !843
  br i1 %42, label %1906, label %1903

1903:                                             ; preds = %1897
  %1904 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1901
  %1905 = addrspacecast float addrspace(4)* %1904 to float addrspace(1)*
  store float %1902, float addrspace(1)* %1905, align 4
  br label %._crit_edge70.1.4

1906:                                             ; preds = %1897
  %1907 = mul nsw i64 %1898, %const_reg_qword7, !spirv.Decorations !836
  %1908 = getelementptr float, float addrspace(4)* %160, i64 %1907
  %1909 = getelementptr float, float addrspace(4)* %1908, i64 %1899
  %1910 = addrspacecast float addrspace(4)* %1909 to float addrspace(1)*
  %1911 = load float, float addrspace(1)* %1910, align 4
  %1912 = fmul reassoc nsz arcp contract float %1911, %4, !spirv.Decorations !843
  %1913 = fadd reassoc nsz arcp contract float %1902, %1912, !spirv.Decorations !843
  %1914 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1901
  %1915 = addrspacecast float addrspace(4)* %1914 to float addrspace(1)*
  store float %1913, float addrspace(1)* %1915, align 4
  br label %._crit_edge70.1.4

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4, %1906, %1903
  br i1 %90, label %1916, label %._crit_edge70.2.4

1916:                                             ; preds = %._crit_edge70.1.4
  %1917 = sext i32 %62 to i64
  %1918 = sext i32 %86 to i64
  %1919 = mul nsw i64 %1917, %const_reg_qword9, !spirv.Decorations !836
  %1920 = add nsw i64 %1919, %1918, !spirv.Decorations !836
  %1921 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !843
  br i1 %42, label %1925, label %1922

1922:                                             ; preds = %1916
  %1923 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1920
  %1924 = addrspacecast float addrspace(4)* %1923 to float addrspace(1)*
  store float %1921, float addrspace(1)* %1924, align 4
  br label %._crit_edge70.2.4

1925:                                             ; preds = %1916
  %1926 = mul nsw i64 %1917, %const_reg_qword7, !spirv.Decorations !836
  %1927 = getelementptr float, float addrspace(4)* %160, i64 %1926
  %1928 = getelementptr float, float addrspace(4)* %1927, i64 %1918
  %1929 = addrspacecast float addrspace(4)* %1928 to float addrspace(1)*
  %1930 = load float, float addrspace(1)* %1929, align 4
  %1931 = fmul reassoc nsz arcp contract float %1930, %4, !spirv.Decorations !843
  %1932 = fadd reassoc nsz arcp contract float %1921, %1931, !spirv.Decorations !843
  %1933 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1920
  %1934 = addrspacecast float addrspace(4)* %1933 to float addrspace(1)*
  store float %1932, float addrspace(1)* %1934, align 4
  br label %._crit_edge70.2.4

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4, %1925, %1922
  br i1 %91, label %1935, label %.preheader1.4

1935:                                             ; preds = %._crit_edge70.2.4
  %1936 = sext i32 %65 to i64
  %1937 = sext i32 %86 to i64
  %1938 = mul nsw i64 %1936, %const_reg_qword9, !spirv.Decorations !836
  %1939 = add nsw i64 %1938, %1937, !spirv.Decorations !836
  %1940 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !843
  br i1 %42, label %1944, label %1941

1941:                                             ; preds = %1935
  %1942 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1939
  %1943 = addrspacecast float addrspace(4)* %1942 to float addrspace(1)*
  store float %1940, float addrspace(1)* %1943, align 4
  br label %.preheader1.4

1944:                                             ; preds = %1935
  %1945 = mul nsw i64 %1936, %const_reg_qword7, !spirv.Decorations !836
  %1946 = getelementptr float, float addrspace(4)* %160, i64 %1945
  %1947 = getelementptr float, float addrspace(4)* %1946, i64 %1937
  %1948 = addrspacecast float addrspace(4)* %1947 to float addrspace(1)*
  %1949 = load float, float addrspace(1)* %1948, align 4
  %1950 = fmul reassoc nsz arcp contract float %1949, %4, !spirv.Decorations !843
  %1951 = fadd reassoc nsz arcp contract float %1940, %1950, !spirv.Decorations !843
  %1952 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1939
  %1953 = addrspacecast float addrspace(4)* %1952 to float addrspace(1)*
  store float %1951, float addrspace(1)* %1953, align 4
  br label %.preheader1.4

.preheader1.4:                                    ; preds = %._crit_edge70.2.4, %1944, %1941
  br i1 %94, label %1954, label %._crit_edge70.5

1954:                                             ; preds = %.preheader1.4
  %1955 = sext i32 %29 to i64
  %1956 = sext i32 %92 to i64
  %1957 = mul nsw i64 %1955, %const_reg_qword9, !spirv.Decorations !836
  %1958 = add nsw i64 %1957, %1956, !spirv.Decorations !836
  %1959 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !843
  br i1 %42, label %1963, label %1960

1960:                                             ; preds = %1954
  %1961 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1958
  %1962 = addrspacecast float addrspace(4)* %1961 to float addrspace(1)*
  store float %1959, float addrspace(1)* %1962, align 4
  br label %._crit_edge70.5

1963:                                             ; preds = %1954
  %1964 = mul nsw i64 %1955, %const_reg_qword7, !spirv.Decorations !836
  %1965 = getelementptr float, float addrspace(4)* %160, i64 %1964
  %1966 = getelementptr float, float addrspace(4)* %1965, i64 %1956
  %1967 = addrspacecast float addrspace(4)* %1966 to float addrspace(1)*
  %1968 = load float, float addrspace(1)* %1967, align 4
  %1969 = fmul reassoc nsz arcp contract float %1968, %4, !spirv.Decorations !843
  %1970 = fadd reassoc nsz arcp contract float %1959, %1969, !spirv.Decorations !843
  %1971 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1958
  %1972 = addrspacecast float addrspace(4)* %1971 to float addrspace(1)*
  store float %1970, float addrspace(1)* %1972, align 4
  br label %._crit_edge70.5

._crit_edge70.5:                                  ; preds = %.preheader1.4, %1963, %1960
  br i1 %95, label %1973, label %._crit_edge70.1.5

1973:                                             ; preds = %._crit_edge70.5
  %1974 = sext i32 %59 to i64
  %1975 = sext i32 %92 to i64
  %1976 = mul nsw i64 %1974, %const_reg_qword9, !spirv.Decorations !836
  %1977 = add nsw i64 %1976, %1975, !spirv.Decorations !836
  %1978 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !843
  br i1 %42, label %1982, label %1979

1979:                                             ; preds = %1973
  %1980 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1977
  %1981 = addrspacecast float addrspace(4)* %1980 to float addrspace(1)*
  store float %1978, float addrspace(1)* %1981, align 4
  br label %._crit_edge70.1.5

1982:                                             ; preds = %1973
  %1983 = mul nsw i64 %1974, %const_reg_qword7, !spirv.Decorations !836
  %1984 = getelementptr float, float addrspace(4)* %160, i64 %1983
  %1985 = getelementptr float, float addrspace(4)* %1984, i64 %1975
  %1986 = addrspacecast float addrspace(4)* %1985 to float addrspace(1)*
  %1987 = load float, float addrspace(1)* %1986, align 4
  %1988 = fmul reassoc nsz arcp contract float %1987, %4, !spirv.Decorations !843
  %1989 = fadd reassoc nsz arcp contract float %1978, %1988, !spirv.Decorations !843
  %1990 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1977
  %1991 = addrspacecast float addrspace(4)* %1990 to float addrspace(1)*
  store float %1989, float addrspace(1)* %1991, align 4
  br label %._crit_edge70.1.5

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5, %1982, %1979
  br i1 %96, label %1992, label %._crit_edge70.2.5

1992:                                             ; preds = %._crit_edge70.1.5
  %1993 = sext i32 %62 to i64
  %1994 = sext i32 %92 to i64
  %1995 = mul nsw i64 %1993, %const_reg_qword9, !spirv.Decorations !836
  %1996 = add nsw i64 %1995, %1994, !spirv.Decorations !836
  %1997 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !843
  br i1 %42, label %2001, label %1998

1998:                                             ; preds = %1992
  %1999 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1996
  %2000 = addrspacecast float addrspace(4)* %1999 to float addrspace(1)*
  store float %1997, float addrspace(1)* %2000, align 4
  br label %._crit_edge70.2.5

2001:                                             ; preds = %1992
  %2002 = mul nsw i64 %1993, %const_reg_qword7, !spirv.Decorations !836
  %2003 = getelementptr float, float addrspace(4)* %160, i64 %2002
  %2004 = getelementptr float, float addrspace(4)* %2003, i64 %1994
  %2005 = addrspacecast float addrspace(4)* %2004 to float addrspace(1)*
  %2006 = load float, float addrspace(1)* %2005, align 4
  %2007 = fmul reassoc nsz arcp contract float %2006, %4, !spirv.Decorations !843
  %2008 = fadd reassoc nsz arcp contract float %1997, %2007, !spirv.Decorations !843
  %2009 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1996
  %2010 = addrspacecast float addrspace(4)* %2009 to float addrspace(1)*
  store float %2008, float addrspace(1)* %2010, align 4
  br label %._crit_edge70.2.5

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5, %2001, %1998
  br i1 %97, label %2011, label %.preheader1.5

2011:                                             ; preds = %._crit_edge70.2.5
  %2012 = sext i32 %65 to i64
  %2013 = sext i32 %92 to i64
  %2014 = mul nsw i64 %2012, %const_reg_qword9, !spirv.Decorations !836
  %2015 = add nsw i64 %2014, %2013, !spirv.Decorations !836
  %2016 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !843
  br i1 %42, label %2020, label %2017

2017:                                             ; preds = %2011
  %2018 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2015
  %2019 = addrspacecast float addrspace(4)* %2018 to float addrspace(1)*
  store float %2016, float addrspace(1)* %2019, align 4
  br label %.preheader1.5

2020:                                             ; preds = %2011
  %2021 = mul nsw i64 %2012, %const_reg_qword7, !spirv.Decorations !836
  %2022 = getelementptr float, float addrspace(4)* %160, i64 %2021
  %2023 = getelementptr float, float addrspace(4)* %2022, i64 %2013
  %2024 = addrspacecast float addrspace(4)* %2023 to float addrspace(1)*
  %2025 = load float, float addrspace(1)* %2024, align 4
  %2026 = fmul reassoc nsz arcp contract float %2025, %4, !spirv.Decorations !843
  %2027 = fadd reassoc nsz arcp contract float %2016, %2026, !spirv.Decorations !843
  %2028 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2015
  %2029 = addrspacecast float addrspace(4)* %2028 to float addrspace(1)*
  store float %2027, float addrspace(1)* %2029, align 4
  br label %.preheader1.5

.preheader1.5:                                    ; preds = %._crit_edge70.2.5, %2020, %2017
  br i1 %100, label %2030, label %._crit_edge70.6

2030:                                             ; preds = %.preheader1.5
  %2031 = sext i32 %29 to i64
  %2032 = sext i32 %98 to i64
  %2033 = mul nsw i64 %2031, %const_reg_qword9, !spirv.Decorations !836
  %2034 = add nsw i64 %2033, %2032, !spirv.Decorations !836
  %2035 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !843
  br i1 %42, label %2039, label %2036

2036:                                             ; preds = %2030
  %2037 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2034
  %2038 = addrspacecast float addrspace(4)* %2037 to float addrspace(1)*
  store float %2035, float addrspace(1)* %2038, align 4
  br label %._crit_edge70.6

2039:                                             ; preds = %2030
  %2040 = mul nsw i64 %2031, %const_reg_qword7, !spirv.Decorations !836
  %2041 = getelementptr float, float addrspace(4)* %160, i64 %2040
  %2042 = getelementptr float, float addrspace(4)* %2041, i64 %2032
  %2043 = addrspacecast float addrspace(4)* %2042 to float addrspace(1)*
  %2044 = load float, float addrspace(1)* %2043, align 4
  %2045 = fmul reassoc nsz arcp contract float %2044, %4, !spirv.Decorations !843
  %2046 = fadd reassoc nsz arcp contract float %2035, %2045, !spirv.Decorations !843
  %2047 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2034
  %2048 = addrspacecast float addrspace(4)* %2047 to float addrspace(1)*
  store float %2046, float addrspace(1)* %2048, align 4
  br label %._crit_edge70.6

._crit_edge70.6:                                  ; preds = %.preheader1.5, %2039, %2036
  br i1 %101, label %2049, label %._crit_edge70.1.6

2049:                                             ; preds = %._crit_edge70.6
  %2050 = sext i32 %59 to i64
  %2051 = sext i32 %98 to i64
  %2052 = mul nsw i64 %2050, %const_reg_qword9, !spirv.Decorations !836
  %2053 = add nsw i64 %2052, %2051, !spirv.Decorations !836
  %2054 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !843
  br i1 %42, label %2058, label %2055

2055:                                             ; preds = %2049
  %2056 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2053
  %2057 = addrspacecast float addrspace(4)* %2056 to float addrspace(1)*
  store float %2054, float addrspace(1)* %2057, align 4
  br label %._crit_edge70.1.6

2058:                                             ; preds = %2049
  %2059 = mul nsw i64 %2050, %const_reg_qword7, !spirv.Decorations !836
  %2060 = getelementptr float, float addrspace(4)* %160, i64 %2059
  %2061 = getelementptr float, float addrspace(4)* %2060, i64 %2051
  %2062 = addrspacecast float addrspace(4)* %2061 to float addrspace(1)*
  %2063 = load float, float addrspace(1)* %2062, align 4
  %2064 = fmul reassoc nsz arcp contract float %2063, %4, !spirv.Decorations !843
  %2065 = fadd reassoc nsz arcp contract float %2054, %2064, !spirv.Decorations !843
  %2066 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2053
  %2067 = addrspacecast float addrspace(4)* %2066 to float addrspace(1)*
  store float %2065, float addrspace(1)* %2067, align 4
  br label %._crit_edge70.1.6

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6, %2058, %2055
  br i1 %102, label %2068, label %._crit_edge70.2.6

2068:                                             ; preds = %._crit_edge70.1.6
  %2069 = sext i32 %62 to i64
  %2070 = sext i32 %98 to i64
  %2071 = mul nsw i64 %2069, %const_reg_qword9, !spirv.Decorations !836
  %2072 = add nsw i64 %2071, %2070, !spirv.Decorations !836
  %2073 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !843
  br i1 %42, label %2077, label %2074

2074:                                             ; preds = %2068
  %2075 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2072
  %2076 = addrspacecast float addrspace(4)* %2075 to float addrspace(1)*
  store float %2073, float addrspace(1)* %2076, align 4
  br label %._crit_edge70.2.6

2077:                                             ; preds = %2068
  %2078 = mul nsw i64 %2069, %const_reg_qword7, !spirv.Decorations !836
  %2079 = getelementptr float, float addrspace(4)* %160, i64 %2078
  %2080 = getelementptr float, float addrspace(4)* %2079, i64 %2070
  %2081 = addrspacecast float addrspace(4)* %2080 to float addrspace(1)*
  %2082 = load float, float addrspace(1)* %2081, align 4
  %2083 = fmul reassoc nsz arcp contract float %2082, %4, !spirv.Decorations !843
  %2084 = fadd reassoc nsz arcp contract float %2073, %2083, !spirv.Decorations !843
  %2085 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2072
  %2086 = addrspacecast float addrspace(4)* %2085 to float addrspace(1)*
  store float %2084, float addrspace(1)* %2086, align 4
  br label %._crit_edge70.2.6

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6, %2077, %2074
  br i1 %103, label %2087, label %.preheader1.6

2087:                                             ; preds = %._crit_edge70.2.6
  %2088 = sext i32 %65 to i64
  %2089 = sext i32 %98 to i64
  %2090 = mul nsw i64 %2088, %const_reg_qword9, !spirv.Decorations !836
  %2091 = add nsw i64 %2090, %2089, !spirv.Decorations !836
  %2092 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !843
  br i1 %42, label %2096, label %2093

2093:                                             ; preds = %2087
  %2094 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2091
  %2095 = addrspacecast float addrspace(4)* %2094 to float addrspace(1)*
  store float %2092, float addrspace(1)* %2095, align 4
  br label %.preheader1.6

2096:                                             ; preds = %2087
  %2097 = mul nsw i64 %2088, %const_reg_qword7, !spirv.Decorations !836
  %2098 = getelementptr float, float addrspace(4)* %160, i64 %2097
  %2099 = getelementptr float, float addrspace(4)* %2098, i64 %2089
  %2100 = addrspacecast float addrspace(4)* %2099 to float addrspace(1)*
  %2101 = load float, float addrspace(1)* %2100, align 4
  %2102 = fmul reassoc nsz arcp contract float %2101, %4, !spirv.Decorations !843
  %2103 = fadd reassoc nsz arcp contract float %2092, %2102, !spirv.Decorations !843
  %2104 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2091
  %2105 = addrspacecast float addrspace(4)* %2104 to float addrspace(1)*
  store float %2103, float addrspace(1)* %2105, align 4
  br label %.preheader1.6

.preheader1.6:                                    ; preds = %._crit_edge70.2.6, %2096, %2093
  br i1 %106, label %2106, label %._crit_edge70.7

2106:                                             ; preds = %.preheader1.6
  %2107 = sext i32 %29 to i64
  %2108 = sext i32 %104 to i64
  %2109 = mul nsw i64 %2107, %const_reg_qword9, !spirv.Decorations !836
  %2110 = add nsw i64 %2109, %2108, !spirv.Decorations !836
  %2111 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !843
  br i1 %42, label %2115, label %2112

2112:                                             ; preds = %2106
  %2113 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2110
  %2114 = addrspacecast float addrspace(4)* %2113 to float addrspace(1)*
  store float %2111, float addrspace(1)* %2114, align 4
  br label %._crit_edge70.7

2115:                                             ; preds = %2106
  %2116 = mul nsw i64 %2107, %const_reg_qword7, !spirv.Decorations !836
  %2117 = getelementptr float, float addrspace(4)* %160, i64 %2116
  %2118 = getelementptr float, float addrspace(4)* %2117, i64 %2108
  %2119 = addrspacecast float addrspace(4)* %2118 to float addrspace(1)*
  %2120 = load float, float addrspace(1)* %2119, align 4
  %2121 = fmul reassoc nsz arcp contract float %2120, %4, !spirv.Decorations !843
  %2122 = fadd reassoc nsz arcp contract float %2111, %2121, !spirv.Decorations !843
  %2123 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2110
  %2124 = addrspacecast float addrspace(4)* %2123 to float addrspace(1)*
  store float %2122, float addrspace(1)* %2124, align 4
  br label %._crit_edge70.7

._crit_edge70.7:                                  ; preds = %.preheader1.6, %2115, %2112
  br i1 %107, label %2125, label %._crit_edge70.1.7

2125:                                             ; preds = %._crit_edge70.7
  %2126 = sext i32 %59 to i64
  %2127 = sext i32 %104 to i64
  %2128 = mul nsw i64 %2126, %const_reg_qword9, !spirv.Decorations !836
  %2129 = add nsw i64 %2128, %2127, !spirv.Decorations !836
  %2130 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !843
  br i1 %42, label %2134, label %2131

2131:                                             ; preds = %2125
  %2132 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2129
  %2133 = addrspacecast float addrspace(4)* %2132 to float addrspace(1)*
  store float %2130, float addrspace(1)* %2133, align 4
  br label %._crit_edge70.1.7

2134:                                             ; preds = %2125
  %2135 = mul nsw i64 %2126, %const_reg_qword7, !spirv.Decorations !836
  %2136 = getelementptr float, float addrspace(4)* %160, i64 %2135
  %2137 = getelementptr float, float addrspace(4)* %2136, i64 %2127
  %2138 = addrspacecast float addrspace(4)* %2137 to float addrspace(1)*
  %2139 = load float, float addrspace(1)* %2138, align 4
  %2140 = fmul reassoc nsz arcp contract float %2139, %4, !spirv.Decorations !843
  %2141 = fadd reassoc nsz arcp contract float %2130, %2140, !spirv.Decorations !843
  %2142 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2129
  %2143 = addrspacecast float addrspace(4)* %2142 to float addrspace(1)*
  store float %2141, float addrspace(1)* %2143, align 4
  br label %._crit_edge70.1.7

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7, %2134, %2131
  br i1 %108, label %2144, label %._crit_edge70.2.7

2144:                                             ; preds = %._crit_edge70.1.7
  %2145 = sext i32 %62 to i64
  %2146 = sext i32 %104 to i64
  %2147 = mul nsw i64 %2145, %const_reg_qword9, !spirv.Decorations !836
  %2148 = add nsw i64 %2147, %2146, !spirv.Decorations !836
  %2149 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !843
  br i1 %42, label %2153, label %2150

2150:                                             ; preds = %2144
  %2151 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2148
  %2152 = addrspacecast float addrspace(4)* %2151 to float addrspace(1)*
  store float %2149, float addrspace(1)* %2152, align 4
  br label %._crit_edge70.2.7

2153:                                             ; preds = %2144
  %2154 = mul nsw i64 %2145, %const_reg_qword7, !spirv.Decorations !836
  %2155 = getelementptr float, float addrspace(4)* %160, i64 %2154
  %2156 = getelementptr float, float addrspace(4)* %2155, i64 %2146
  %2157 = addrspacecast float addrspace(4)* %2156 to float addrspace(1)*
  %2158 = load float, float addrspace(1)* %2157, align 4
  %2159 = fmul reassoc nsz arcp contract float %2158, %4, !spirv.Decorations !843
  %2160 = fadd reassoc nsz arcp contract float %2149, %2159, !spirv.Decorations !843
  %2161 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2148
  %2162 = addrspacecast float addrspace(4)* %2161 to float addrspace(1)*
  store float %2160, float addrspace(1)* %2162, align 4
  br label %._crit_edge70.2.7

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7, %2153, %2150
  br i1 %109, label %2163, label %.preheader1.7

2163:                                             ; preds = %._crit_edge70.2.7
  %2164 = sext i32 %65 to i64
  %2165 = sext i32 %104 to i64
  %2166 = mul nsw i64 %2164, %const_reg_qword9, !spirv.Decorations !836
  %2167 = add nsw i64 %2166, %2165, !spirv.Decorations !836
  %2168 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !843
  br i1 %42, label %2172, label %2169

2169:                                             ; preds = %2163
  %2170 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2167
  %2171 = addrspacecast float addrspace(4)* %2170 to float addrspace(1)*
  store float %2168, float addrspace(1)* %2171, align 4
  br label %.preheader1.7

2172:                                             ; preds = %2163
  %2173 = mul nsw i64 %2164, %const_reg_qword7, !spirv.Decorations !836
  %2174 = getelementptr float, float addrspace(4)* %160, i64 %2173
  %2175 = getelementptr float, float addrspace(4)* %2174, i64 %2165
  %2176 = addrspacecast float addrspace(4)* %2175 to float addrspace(1)*
  %2177 = load float, float addrspace(1)* %2176, align 4
  %2178 = fmul reassoc nsz arcp contract float %2177, %4, !spirv.Decorations !843
  %2179 = fadd reassoc nsz arcp contract float %2168, %2178, !spirv.Decorations !843
  %2180 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2167
  %2181 = addrspacecast float addrspace(4)* %2180 to float addrspace(1)*
  store float %2179, float addrspace(1)* %2181, align 4
  br label %.preheader1.7

.preheader1.7:                                    ; preds = %._crit_edge70.2.7, %2172, %2169
  br i1 %112, label %2182, label %._crit_edge70.8

2182:                                             ; preds = %.preheader1.7
  %2183 = sext i32 %29 to i64
  %2184 = sext i32 %110 to i64
  %2185 = mul nsw i64 %2183, %const_reg_qword9, !spirv.Decorations !836
  %2186 = add nsw i64 %2185, %2184, !spirv.Decorations !836
  %2187 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !843
  br i1 %42, label %2191, label %2188

2188:                                             ; preds = %2182
  %2189 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2186
  %2190 = addrspacecast float addrspace(4)* %2189 to float addrspace(1)*
  store float %2187, float addrspace(1)* %2190, align 4
  br label %._crit_edge70.8

2191:                                             ; preds = %2182
  %2192 = mul nsw i64 %2183, %const_reg_qword7, !spirv.Decorations !836
  %2193 = getelementptr float, float addrspace(4)* %160, i64 %2192
  %2194 = getelementptr float, float addrspace(4)* %2193, i64 %2184
  %2195 = addrspacecast float addrspace(4)* %2194 to float addrspace(1)*
  %2196 = load float, float addrspace(1)* %2195, align 4
  %2197 = fmul reassoc nsz arcp contract float %2196, %4, !spirv.Decorations !843
  %2198 = fadd reassoc nsz arcp contract float %2187, %2197, !spirv.Decorations !843
  %2199 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2186
  %2200 = addrspacecast float addrspace(4)* %2199 to float addrspace(1)*
  store float %2198, float addrspace(1)* %2200, align 4
  br label %._crit_edge70.8

._crit_edge70.8:                                  ; preds = %.preheader1.7, %2191, %2188
  br i1 %113, label %2201, label %._crit_edge70.1.8

2201:                                             ; preds = %._crit_edge70.8
  %2202 = sext i32 %59 to i64
  %2203 = sext i32 %110 to i64
  %2204 = mul nsw i64 %2202, %const_reg_qword9, !spirv.Decorations !836
  %2205 = add nsw i64 %2204, %2203, !spirv.Decorations !836
  %2206 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !843
  br i1 %42, label %2210, label %2207

2207:                                             ; preds = %2201
  %2208 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2205
  %2209 = addrspacecast float addrspace(4)* %2208 to float addrspace(1)*
  store float %2206, float addrspace(1)* %2209, align 4
  br label %._crit_edge70.1.8

2210:                                             ; preds = %2201
  %2211 = mul nsw i64 %2202, %const_reg_qword7, !spirv.Decorations !836
  %2212 = getelementptr float, float addrspace(4)* %160, i64 %2211
  %2213 = getelementptr float, float addrspace(4)* %2212, i64 %2203
  %2214 = addrspacecast float addrspace(4)* %2213 to float addrspace(1)*
  %2215 = load float, float addrspace(1)* %2214, align 4
  %2216 = fmul reassoc nsz arcp contract float %2215, %4, !spirv.Decorations !843
  %2217 = fadd reassoc nsz arcp contract float %2206, %2216, !spirv.Decorations !843
  %2218 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2205
  %2219 = addrspacecast float addrspace(4)* %2218 to float addrspace(1)*
  store float %2217, float addrspace(1)* %2219, align 4
  br label %._crit_edge70.1.8

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8, %2210, %2207
  br i1 %114, label %2220, label %._crit_edge70.2.8

2220:                                             ; preds = %._crit_edge70.1.8
  %2221 = sext i32 %62 to i64
  %2222 = sext i32 %110 to i64
  %2223 = mul nsw i64 %2221, %const_reg_qword9, !spirv.Decorations !836
  %2224 = add nsw i64 %2223, %2222, !spirv.Decorations !836
  %2225 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !843
  br i1 %42, label %2229, label %2226

2226:                                             ; preds = %2220
  %2227 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2224
  %2228 = addrspacecast float addrspace(4)* %2227 to float addrspace(1)*
  store float %2225, float addrspace(1)* %2228, align 4
  br label %._crit_edge70.2.8

2229:                                             ; preds = %2220
  %2230 = mul nsw i64 %2221, %const_reg_qword7, !spirv.Decorations !836
  %2231 = getelementptr float, float addrspace(4)* %160, i64 %2230
  %2232 = getelementptr float, float addrspace(4)* %2231, i64 %2222
  %2233 = addrspacecast float addrspace(4)* %2232 to float addrspace(1)*
  %2234 = load float, float addrspace(1)* %2233, align 4
  %2235 = fmul reassoc nsz arcp contract float %2234, %4, !spirv.Decorations !843
  %2236 = fadd reassoc nsz arcp contract float %2225, %2235, !spirv.Decorations !843
  %2237 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2224
  %2238 = addrspacecast float addrspace(4)* %2237 to float addrspace(1)*
  store float %2236, float addrspace(1)* %2238, align 4
  br label %._crit_edge70.2.8

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8, %2229, %2226
  br i1 %115, label %2239, label %.preheader1.8

2239:                                             ; preds = %._crit_edge70.2.8
  %2240 = sext i32 %65 to i64
  %2241 = sext i32 %110 to i64
  %2242 = mul nsw i64 %2240, %const_reg_qword9, !spirv.Decorations !836
  %2243 = add nsw i64 %2242, %2241, !spirv.Decorations !836
  %2244 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !843
  br i1 %42, label %2248, label %2245

2245:                                             ; preds = %2239
  %2246 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2243
  %2247 = addrspacecast float addrspace(4)* %2246 to float addrspace(1)*
  store float %2244, float addrspace(1)* %2247, align 4
  br label %.preheader1.8

2248:                                             ; preds = %2239
  %2249 = mul nsw i64 %2240, %const_reg_qword7, !spirv.Decorations !836
  %2250 = getelementptr float, float addrspace(4)* %160, i64 %2249
  %2251 = getelementptr float, float addrspace(4)* %2250, i64 %2241
  %2252 = addrspacecast float addrspace(4)* %2251 to float addrspace(1)*
  %2253 = load float, float addrspace(1)* %2252, align 4
  %2254 = fmul reassoc nsz arcp contract float %2253, %4, !spirv.Decorations !843
  %2255 = fadd reassoc nsz arcp contract float %2244, %2254, !spirv.Decorations !843
  %2256 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2243
  %2257 = addrspacecast float addrspace(4)* %2256 to float addrspace(1)*
  store float %2255, float addrspace(1)* %2257, align 4
  br label %.preheader1.8

.preheader1.8:                                    ; preds = %._crit_edge70.2.8, %2248, %2245
  br i1 %118, label %2258, label %._crit_edge70.9

2258:                                             ; preds = %.preheader1.8
  %2259 = sext i32 %29 to i64
  %2260 = sext i32 %116 to i64
  %2261 = mul nsw i64 %2259, %const_reg_qword9, !spirv.Decorations !836
  %2262 = add nsw i64 %2261, %2260, !spirv.Decorations !836
  %2263 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !843
  br i1 %42, label %2267, label %2264

2264:                                             ; preds = %2258
  %2265 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2262
  %2266 = addrspacecast float addrspace(4)* %2265 to float addrspace(1)*
  store float %2263, float addrspace(1)* %2266, align 4
  br label %._crit_edge70.9

2267:                                             ; preds = %2258
  %2268 = mul nsw i64 %2259, %const_reg_qword7, !spirv.Decorations !836
  %2269 = getelementptr float, float addrspace(4)* %160, i64 %2268
  %2270 = getelementptr float, float addrspace(4)* %2269, i64 %2260
  %2271 = addrspacecast float addrspace(4)* %2270 to float addrspace(1)*
  %2272 = load float, float addrspace(1)* %2271, align 4
  %2273 = fmul reassoc nsz arcp contract float %2272, %4, !spirv.Decorations !843
  %2274 = fadd reassoc nsz arcp contract float %2263, %2273, !spirv.Decorations !843
  %2275 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2262
  %2276 = addrspacecast float addrspace(4)* %2275 to float addrspace(1)*
  store float %2274, float addrspace(1)* %2276, align 4
  br label %._crit_edge70.9

._crit_edge70.9:                                  ; preds = %.preheader1.8, %2267, %2264
  br i1 %119, label %2277, label %._crit_edge70.1.9

2277:                                             ; preds = %._crit_edge70.9
  %2278 = sext i32 %59 to i64
  %2279 = sext i32 %116 to i64
  %2280 = mul nsw i64 %2278, %const_reg_qword9, !spirv.Decorations !836
  %2281 = add nsw i64 %2280, %2279, !spirv.Decorations !836
  %2282 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !843
  br i1 %42, label %2286, label %2283

2283:                                             ; preds = %2277
  %2284 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2281
  %2285 = addrspacecast float addrspace(4)* %2284 to float addrspace(1)*
  store float %2282, float addrspace(1)* %2285, align 4
  br label %._crit_edge70.1.9

2286:                                             ; preds = %2277
  %2287 = mul nsw i64 %2278, %const_reg_qword7, !spirv.Decorations !836
  %2288 = getelementptr float, float addrspace(4)* %160, i64 %2287
  %2289 = getelementptr float, float addrspace(4)* %2288, i64 %2279
  %2290 = addrspacecast float addrspace(4)* %2289 to float addrspace(1)*
  %2291 = load float, float addrspace(1)* %2290, align 4
  %2292 = fmul reassoc nsz arcp contract float %2291, %4, !spirv.Decorations !843
  %2293 = fadd reassoc nsz arcp contract float %2282, %2292, !spirv.Decorations !843
  %2294 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2281
  %2295 = addrspacecast float addrspace(4)* %2294 to float addrspace(1)*
  store float %2293, float addrspace(1)* %2295, align 4
  br label %._crit_edge70.1.9

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9, %2286, %2283
  br i1 %120, label %2296, label %._crit_edge70.2.9

2296:                                             ; preds = %._crit_edge70.1.9
  %2297 = sext i32 %62 to i64
  %2298 = sext i32 %116 to i64
  %2299 = mul nsw i64 %2297, %const_reg_qword9, !spirv.Decorations !836
  %2300 = add nsw i64 %2299, %2298, !spirv.Decorations !836
  %2301 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !843
  br i1 %42, label %2305, label %2302

2302:                                             ; preds = %2296
  %2303 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2300
  %2304 = addrspacecast float addrspace(4)* %2303 to float addrspace(1)*
  store float %2301, float addrspace(1)* %2304, align 4
  br label %._crit_edge70.2.9

2305:                                             ; preds = %2296
  %2306 = mul nsw i64 %2297, %const_reg_qword7, !spirv.Decorations !836
  %2307 = getelementptr float, float addrspace(4)* %160, i64 %2306
  %2308 = getelementptr float, float addrspace(4)* %2307, i64 %2298
  %2309 = addrspacecast float addrspace(4)* %2308 to float addrspace(1)*
  %2310 = load float, float addrspace(1)* %2309, align 4
  %2311 = fmul reassoc nsz arcp contract float %2310, %4, !spirv.Decorations !843
  %2312 = fadd reassoc nsz arcp contract float %2301, %2311, !spirv.Decorations !843
  %2313 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2300
  %2314 = addrspacecast float addrspace(4)* %2313 to float addrspace(1)*
  store float %2312, float addrspace(1)* %2314, align 4
  br label %._crit_edge70.2.9

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9, %2305, %2302
  br i1 %121, label %2315, label %.preheader1.9

2315:                                             ; preds = %._crit_edge70.2.9
  %2316 = sext i32 %65 to i64
  %2317 = sext i32 %116 to i64
  %2318 = mul nsw i64 %2316, %const_reg_qword9, !spirv.Decorations !836
  %2319 = add nsw i64 %2318, %2317, !spirv.Decorations !836
  %2320 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !843
  br i1 %42, label %2324, label %2321

2321:                                             ; preds = %2315
  %2322 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2319
  %2323 = addrspacecast float addrspace(4)* %2322 to float addrspace(1)*
  store float %2320, float addrspace(1)* %2323, align 4
  br label %.preheader1.9

2324:                                             ; preds = %2315
  %2325 = mul nsw i64 %2316, %const_reg_qword7, !spirv.Decorations !836
  %2326 = getelementptr float, float addrspace(4)* %160, i64 %2325
  %2327 = getelementptr float, float addrspace(4)* %2326, i64 %2317
  %2328 = addrspacecast float addrspace(4)* %2327 to float addrspace(1)*
  %2329 = load float, float addrspace(1)* %2328, align 4
  %2330 = fmul reassoc nsz arcp contract float %2329, %4, !spirv.Decorations !843
  %2331 = fadd reassoc nsz arcp contract float %2320, %2330, !spirv.Decorations !843
  %2332 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2319
  %2333 = addrspacecast float addrspace(4)* %2332 to float addrspace(1)*
  store float %2331, float addrspace(1)* %2333, align 4
  br label %.preheader1.9

.preheader1.9:                                    ; preds = %._crit_edge70.2.9, %2324, %2321
  br i1 %124, label %2334, label %._crit_edge70.10

2334:                                             ; preds = %.preheader1.9
  %2335 = sext i32 %29 to i64
  %2336 = sext i32 %122 to i64
  %2337 = mul nsw i64 %2335, %const_reg_qword9, !spirv.Decorations !836
  %2338 = add nsw i64 %2337, %2336, !spirv.Decorations !836
  %2339 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !843
  br i1 %42, label %2343, label %2340

2340:                                             ; preds = %2334
  %2341 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2338
  %2342 = addrspacecast float addrspace(4)* %2341 to float addrspace(1)*
  store float %2339, float addrspace(1)* %2342, align 4
  br label %._crit_edge70.10

2343:                                             ; preds = %2334
  %2344 = mul nsw i64 %2335, %const_reg_qword7, !spirv.Decorations !836
  %2345 = getelementptr float, float addrspace(4)* %160, i64 %2344
  %2346 = getelementptr float, float addrspace(4)* %2345, i64 %2336
  %2347 = addrspacecast float addrspace(4)* %2346 to float addrspace(1)*
  %2348 = load float, float addrspace(1)* %2347, align 4
  %2349 = fmul reassoc nsz arcp contract float %2348, %4, !spirv.Decorations !843
  %2350 = fadd reassoc nsz arcp contract float %2339, %2349, !spirv.Decorations !843
  %2351 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2338
  %2352 = addrspacecast float addrspace(4)* %2351 to float addrspace(1)*
  store float %2350, float addrspace(1)* %2352, align 4
  br label %._crit_edge70.10

._crit_edge70.10:                                 ; preds = %.preheader1.9, %2343, %2340
  br i1 %125, label %2353, label %._crit_edge70.1.10

2353:                                             ; preds = %._crit_edge70.10
  %2354 = sext i32 %59 to i64
  %2355 = sext i32 %122 to i64
  %2356 = mul nsw i64 %2354, %const_reg_qword9, !spirv.Decorations !836
  %2357 = add nsw i64 %2356, %2355, !spirv.Decorations !836
  %2358 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !843
  br i1 %42, label %2362, label %2359

2359:                                             ; preds = %2353
  %2360 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2357
  %2361 = addrspacecast float addrspace(4)* %2360 to float addrspace(1)*
  store float %2358, float addrspace(1)* %2361, align 4
  br label %._crit_edge70.1.10

2362:                                             ; preds = %2353
  %2363 = mul nsw i64 %2354, %const_reg_qword7, !spirv.Decorations !836
  %2364 = getelementptr float, float addrspace(4)* %160, i64 %2363
  %2365 = getelementptr float, float addrspace(4)* %2364, i64 %2355
  %2366 = addrspacecast float addrspace(4)* %2365 to float addrspace(1)*
  %2367 = load float, float addrspace(1)* %2366, align 4
  %2368 = fmul reassoc nsz arcp contract float %2367, %4, !spirv.Decorations !843
  %2369 = fadd reassoc nsz arcp contract float %2358, %2368, !spirv.Decorations !843
  %2370 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2357
  %2371 = addrspacecast float addrspace(4)* %2370 to float addrspace(1)*
  store float %2369, float addrspace(1)* %2371, align 4
  br label %._crit_edge70.1.10

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10, %2362, %2359
  br i1 %126, label %2372, label %._crit_edge70.2.10

2372:                                             ; preds = %._crit_edge70.1.10
  %2373 = sext i32 %62 to i64
  %2374 = sext i32 %122 to i64
  %2375 = mul nsw i64 %2373, %const_reg_qword9, !spirv.Decorations !836
  %2376 = add nsw i64 %2375, %2374, !spirv.Decorations !836
  %2377 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !843
  br i1 %42, label %2381, label %2378

2378:                                             ; preds = %2372
  %2379 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2376
  %2380 = addrspacecast float addrspace(4)* %2379 to float addrspace(1)*
  store float %2377, float addrspace(1)* %2380, align 4
  br label %._crit_edge70.2.10

2381:                                             ; preds = %2372
  %2382 = mul nsw i64 %2373, %const_reg_qword7, !spirv.Decorations !836
  %2383 = getelementptr float, float addrspace(4)* %160, i64 %2382
  %2384 = getelementptr float, float addrspace(4)* %2383, i64 %2374
  %2385 = addrspacecast float addrspace(4)* %2384 to float addrspace(1)*
  %2386 = load float, float addrspace(1)* %2385, align 4
  %2387 = fmul reassoc nsz arcp contract float %2386, %4, !spirv.Decorations !843
  %2388 = fadd reassoc nsz arcp contract float %2377, %2387, !spirv.Decorations !843
  %2389 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2376
  %2390 = addrspacecast float addrspace(4)* %2389 to float addrspace(1)*
  store float %2388, float addrspace(1)* %2390, align 4
  br label %._crit_edge70.2.10

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10, %2381, %2378
  br i1 %127, label %2391, label %.preheader1.10

2391:                                             ; preds = %._crit_edge70.2.10
  %2392 = sext i32 %65 to i64
  %2393 = sext i32 %122 to i64
  %2394 = mul nsw i64 %2392, %const_reg_qword9, !spirv.Decorations !836
  %2395 = add nsw i64 %2394, %2393, !spirv.Decorations !836
  %2396 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !843
  br i1 %42, label %2400, label %2397

2397:                                             ; preds = %2391
  %2398 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2395
  %2399 = addrspacecast float addrspace(4)* %2398 to float addrspace(1)*
  store float %2396, float addrspace(1)* %2399, align 4
  br label %.preheader1.10

2400:                                             ; preds = %2391
  %2401 = mul nsw i64 %2392, %const_reg_qword7, !spirv.Decorations !836
  %2402 = getelementptr float, float addrspace(4)* %160, i64 %2401
  %2403 = getelementptr float, float addrspace(4)* %2402, i64 %2393
  %2404 = addrspacecast float addrspace(4)* %2403 to float addrspace(1)*
  %2405 = load float, float addrspace(1)* %2404, align 4
  %2406 = fmul reassoc nsz arcp contract float %2405, %4, !spirv.Decorations !843
  %2407 = fadd reassoc nsz arcp contract float %2396, %2406, !spirv.Decorations !843
  %2408 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2395
  %2409 = addrspacecast float addrspace(4)* %2408 to float addrspace(1)*
  store float %2407, float addrspace(1)* %2409, align 4
  br label %.preheader1.10

.preheader1.10:                                   ; preds = %._crit_edge70.2.10, %2400, %2397
  br i1 %130, label %2410, label %._crit_edge70.11

2410:                                             ; preds = %.preheader1.10
  %2411 = sext i32 %29 to i64
  %2412 = sext i32 %128 to i64
  %2413 = mul nsw i64 %2411, %const_reg_qword9, !spirv.Decorations !836
  %2414 = add nsw i64 %2413, %2412, !spirv.Decorations !836
  %2415 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !843
  br i1 %42, label %2419, label %2416

2416:                                             ; preds = %2410
  %2417 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2414
  %2418 = addrspacecast float addrspace(4)* %2417 to float addrspace(1)*
  store float %2415, float addrspace(1)* %2418, align 4
  br label %._crit_edge70.11

2419:                                             ; preds = %2410
  %2420 = mul nsw i64 %2411, %const_reg_qword7, !spirv.Decorations !836
  %2421 = getelementptr float, float addrspace(4)* %160, i64 %2420
  %2422 = getelementptr float, float addrspace(4)* %2421, i64 %2412
  %2423 = addrspacecast float addrspace(4)* %2422 to float addrspace(1)*
  %2424 = load float, float addrspace(1)* %2423, align 4
  %2425 = fmul reassoc nsz arcp contract float %2424, %4, !spirv.Decorations !843
  %2426 = fadd reassoc nsz arcp contract float %2415, %2425, !spirv.Decorations !843
  %2427 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2414
  %2428 = addrspacecast float addrspace(4)* %2427 to float addrspace(1)*
  store float %2426, float addrspace(1)* %2428, align 4
  br label %._crit_edge70.11

._crit_edge70.11:                                 ; preds = %.preheader1.10, %2419, %2416
  br i1 %131, label %2429, label %._crit_edge70.1.11

2429:                                             ; preds = %._crit_edge70.11
  %2430 = sext i32 %59 to i64
  %2431 = sext i32 %128 to i64
  %2432 = mul nsw i64 %2430, %const_reg_qword9, !spirv.Decorations !836
  %2433 = add nsw i64 %2432, %2431, !spirv.Decorations !836
  %2434 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !843
  br i1 %42, label %2438, label %2435

2435:                                             ; preds = %2429
  %2436 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2433
  %2437 = addrspacecast float addrspace(4)* %2436 to float addrspace(1)*
  store float %2434, float addrspace(1)* %2437, align 4
  br label %._crit_edge70.1.11

2438:                                             ; preds = %2429
  %2439 = mul nsw i64 %2430, %const_reg_qword7, !spirv.Decorations !836
  %2440 = getelementptr float, float addrspace(4)* %160, i64 %2439
  %2441 = getelementptr float, float addrspace(4)* %2440, i64 %2431
  %2442 = addrspacecast float addrspace(4)* %2441 to float addrspace(1)*
  %2443 = load float, float addrspace(1)* %2442, align 4
  %2444 = fmul reassoc nsz arcp contract float %2443, %4, !spirv.Decorations !843
  %2445 = fadd reassoc nsz arcp contract float %2434, %2444, !spirv.Decorations !843
  %2446 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2433
  %2447 = addrspacecast float addrspace(4)* %2446 to float addrspace(1)*
  store float %2445, float addrspace(1)* %2447, align 4
  br label %._crit_edge70.1.11

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11, %2438, %2435
  br i1 %132, label %2448, label %._crit_edge70.2.11

2448:                                             ; preds = %._crit_edge70.1.11
  %2449 = sext i32 %62 to i64
  %2450 = sext i32 %128 to i64
  %2451 = mul nsw i64 %2449, %const_reg_qword9, !spirv.Decorations !836
  %2452 = add nsw i64 %2451, %2450, !spirv.Decorations !836
  %2453 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !843
  br i1 %42, label %2457, label %2454

2454:                                             ; preds = %2448
  %2455 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2452
  %2456 = addrspacecast float addrspace(4)* %2455 to float addrspace(1)*
  store float %2453, float addrspace(1)* %2456, align 4
  br label %._crit_edge70.2.11

2457:                                             ; preds = %2448
  %2458 = mul nsw i64 %2449, %const_reg_qword7, !spirv.Decorations !836
  %2459 = getelementptr float, float addrspace(4)* %160, i64 %2458
  %2460 = getelementptr float, float addrspace(4)* %2459, i64 %2450
  %2461 = addrspacecast float addrspace(4)* %2460 to float addrspace(1)*
  %2462 = load float, float addrspace(1)* %2461, align 4
  %2463 = fmul reassoc nsz arcp contract float %2462, %4, !spirv.Decorations !843
  %2464 = fadd reassoc nsz arcp contract float %2453, %2463, !spirv.Decorations !843
  %2465 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2452
  %2466 = addrspacecast float addrspace(4)* %2465 to float addrspace(1)*
  store float %2464, float addrspace(1)* %2466, align 4
  br label %._crit_edge70.2.11

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11, %2457, %2454
  br i1 %133, label %2467, label %.preheader1.11

2467:                                             ; preds = %._crit_edge70.2.11
  %2468 = sext i32 %65 to i64
  %2469 = sext i32 %128 to i64
  %2470 = mul nsw i64 %2468, %const_reg_qword9, !spirv.Decorations !836
  %2471 = add nsw i64 %2470, %2469, !spirv.Decorations !836
  %2472 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !843
  br i1 %42, label %2476, label %2473

2473:                                             ; preds = %2467
  %2474 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2471
  %2475 = addrspacecast float addrspace(4)* %2474 to float addrspace(1)*
  store float %2472, float addrspace(1)* %2475, align 4
  br label %.preheader1.11

2476:                                             ; preds = %2467
  %2477 = mul nsw i64 %2468, %const_reg_qword7, !spirv.Decorations !836
  %2478 = getelementptr float, float addrspace(4)* %160, i64 %2477
  %2479 = getelementptr float, float addrspace(4)* %2478, i64 %2469
  %2480 = addrspacecast float addrspace(4)* %2479 to float addrspace(1)*
  %2481 = load float, float addrspace(1)* %2480, align 4
  %2482 = fmul reassoc nsz arcp contract float %2481, %4, !spirv.Decorations !843
  %2483 = fadd reassoc nsz arcp contract float %2472, %2482, !spirv.Decorations !843
  %2484 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2471
  %2485 = addrspacecast float addrspace(4)* %2484 to float addrspace(1)*
  store float %2483, float addrspace(1)* %2485, align 4
  br label %.preheader1.11

.preheader1.11:                                   ; preds = %._crit_edge70.2.11, %2476, %2473
  br i1 %136, label %2486, label %._crit_edge70.12

2486:                                             ; preds = %.preheader1.11
  %2487 = sext i32 %29 to i64
  %2488 = sext i32 %134 to i64
  %2489 = mul nsw i64 %2487, %const_reg_qword9, !spirv.Decorations !836
  %2490 = add nsw i64 %2489, %2488, !spirv.Decorations !836
  %2491 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !843
  br i1 %42, label %2495, label %2492

2492:                                             ; preds = %2486
  %2493 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2490
  %2494 = addrspacecast float addrspace(4)* %2493 to float addrspace(1)*
  store float %2491, float addrspace(1)* %2494, align 4
  br label %._crit_edge70.12

2495:                                             ; preds = %2486
  %2496 = mul nsw i64 %2487, %const_reg_qword7, !spirv.Decorations !836
  %2497 = getelementptr float, float addrspace(4)* %160, i64 %2496
  %2498 = getelementptr float, float addrspace(4)* %2497, i64 %2488
  %2499 = addrspacecast float addrspace(4)* %2498 to float addrspace(1)*
  %2500 = load float, float addrspace(1)* %2499, align 4
  %2501 = fmul reassoc nsz arcp contract float %2500, %4, !spirv.Decorations !843
  %2502 = fadd reassoc nsz arcp contract float %2491, %2501, !spirv.Decorations !843
  %2503 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2490
  %2504 = addrspacecast float addrspace(4)* %2503 to float addrspace(1)*
  store float %2502, float addrspace(1)* %2504, align 4
  br label %._crit_edge70.12

._crit_edge70.12:                                 ; preds = %.preheader1.11, %2495, %2492
  br i1 %137, label %2505, label %._crit_edge70.1.12

2505:                                             ; preds = %._crit_edge70.12
  %2506 = sext i32 %59 to i64
  %2507 = sext i32 %134 to i64
  %2508 = mul nsw i64 %2506, %const_reg_qword9, !spirv.Decorations !836
  %2509 = add nsw i64 %2508, %2507, !spirv.Decorations !836
  %2510 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !843
  br i1 %42, label %2514, label %2511

2511:                                             ; preds = %2505
  %2512 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2509
  %2513 = addrspacecast float addrspace(4)* %2512 to float addrspace(1)*
  store float %2510, float addrspace(1)* %2513, align 4
  br label %._crit_edge70.1.12

2514:                                             ; preds = %2505
  %2515 = mul nsw i64 %2506, %const_reg_qword7, !spirv.Decorations !836
  %2516 = getelementptr float, float addrspace(4)* %160, i64 %2515
  %2517 = getelementptr float, float addrspace(4)* %2516, i64 %2507
  %2518 = addrspacecast float addrspace(4)* %2517 to float addrspace(1)*
  %2519 = load float, float addrspace(1)* %2518, align 4
  %2520 = fmul reassoc nsz arcp contract float %2519, %4, !spirv.Decorations !843
  %2521 = fadd reassoc nsz arcp contract float %2510, %2520, !spirv.Decorations !843
  %2522 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2509
  %2523 = addrspacecast float addrspace(4)* %2522 to float addrspace(1)*
  store float %2521, float addrspace(1)* %2523, align 4
  br label %._crit_edge70.1.12

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12, %2514, %2511
  br i1 %138, label %2524, label %._crit_edge70.2.12

2524:                                             ; preds = %._crit_edge70.1.12
  %2525 = sext i32 %62 to i64
  %2526 = sext i32 %134 to i64
  %2527 = mul nsw i64 %2525, %const_reg_qword9, !spirv.Decorations !836
  %2528 = add nsw i64 %2527, %2526, !spirv.Decorations !836
  %2529 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !843
  br i1 %42, label %2533, label %2530

2530:                                             ; preds = %2524
  %2531 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2528
  %2532 = addrspacecast float addrspace(4)* %2531 to float addrspace(1)*
  store float %2529, float addrspace(1)* %2532, align 4
  br label %._crit_edge70.2.12

2533:                                             ; preds = %2524
  %2534 = mul nsw i64 %2525, %const_reg_qword7, !spirv.Decorations !836
  %2535 = getelementptr float, float addrspace(4)* %160, i64 %2534
  %2536 = getelementptr float, float addrspace(4)* %2535, i64 %2526
  %2537 = addrspacecast float addrspace(4)* %2536 to float addrspace(1)*
  %2538 = load float, float addrspace(1)* %2537, align 4
  %2539 = fmul reassoc nsz arcp contract float %2538, %4, !spirv.Decorations !843
  %2540 = fadd reassoc nsz arcp contract float %2529, %2539, !spirv.Decorations !843
  %2541 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2528
  %2542 = addrspacecast float addrspace(4)* %2541 to float addrspace(1)*
  store float %2540, float addrspace(1)* %2542, align 4
  br label %._crit_edge70.2.12

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12, %2533, %2530
  br i1 %139, label %2543, label %.preheader1.12

2543:                                             ; preds = %._crit_edge70.2.12
  %2544 = sext i32 %65 to i64
  %2545 = sext i32 %134 to i64
  %2546 = mul nsw i64 %2544, %const_reg_qword9, !spirv.Decorations !836
  %2547 = add nsw i64 %2546, %2545, !spirv.Decorations !836
  %2548 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !843
  br i1 %42, label %2552, label %2549

2549:                                             ; preds = %2543
  %2550 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2547
  %2551 = addrspacecast float addrspace(4)* %2550 to float addrspace(1)*
  store float %2548, float addrspace(1)* %2551, align 4
  br label %.preheader1.12

2552:                                             ; preds = %2543
  %2553 = mul nsw i64 %2544, %const_reg_qword7, !spirv.Decorations !836
  %2554 = getelementptr float, float addrspace(4)* %160, i64 %2553
  %2555 = getelementptr float, float addrspace(4)* %2554, i64 %2545
  %2556 = addrspacecast float addrspace(4)* %2555 to float addrspace(1)*
  %2557 = load float, float addrspace(1)* %2556, align 4
  %2558 = fmul reassoc nsz arcp contract float %2557, %4, !spirv.Decorations !843
  %2559 = fadd reassoc nsz arcp contract float %2548, %2558, !spirv.Decorations !843
  %2560 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2547
  %2561 = addrspacecast float addrspace(4)* %2560 to float addrspace(1)*
  store float %2559, float addrspace(1)* %2561, align 4
  br label %.preheader1.12

.preheader1.12:                                   ; preds = %._crit_edge70.2.12, %2552, %2549
  br i1 %142, label %2562, label %._crit_edge70.13

2562:                                             ; preds = %.preheader1.12
  %2563 = sext i32 %29 to i64
  %2564 = sext i32 %140 to i64
  %2565 = mul nsw i64 %2563, %const_reg_qword9, !spirv.Decorations !836
  %2566 = add nsw i64 %2565, %2564, !spirv.Decorations !836
  %2567 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !843
  br i1 %42, label %2571, label %2568

2568:                                             ; preds = %2562
  %2569 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2566
  %2570 = addrspacecast float addrspace(4)* %2569 to float addrspace(1)*
  store float %2567, float addrspace(1)* %2570, align 4
  br label %._crit_edge70.13

2571:                                             ; preds = %2562
  %2572 = mul nsw i64 %2563, %const_reg_qword7, !spirv.Decorations !836
  %2573 = getelementptr float, float addrspace(4)* %160, i64 %2572
  %2574 = getelementptr float, float addrspace(4)* %2573, i64 %2564
  %2575 = addrspacecast float addrspace(4)* %2574 to float addrspace(1)*
  %2576 = load float, float addrspace(1)* %2575, align 4
  %2577 = fmul reassoc nsz arcp contract float %2576, %4, !spirv.Decorations !843
  %2578 = fadd reassoc nsz arcp contract float %2567, %2577, !spirv.Decorations !843
  %2579 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2566
  %2580 = addrspacecast float addrspace(4)* %2579 to float addrspace(1)*
  store float %2578, float addrspace(1)* %2580, align 4
  br label %._crit_edge70.13

._crit_edge70.13:                                 ; preds = %.preheader1.12, %2571, %2568
  br i1 %143, label %2581, label %._crit_edge70.1.13

2581:                                             ; preds = %._crit_edge70.13
  %2582 = sext i32 %59 to i64
  %2583 = sext i32 %140 to i64
  %2584 = mul nsw i64 %2582, %const_reg_qword9, !spirv.Decorations !836
  %2585 = add nsw i64 %2584, %2583, !spirv.Decorations !836
  %2586 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !843
  br i1 %42, label %2590, label %2587

2587:                                             ; preds = %2581
  %2588 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2585
  %2589 = addrspacecast float addrspace(4)* %2588 to float addrspace(1)*
  store float %2586, float addrspace(1)* %2589, align 4
  br label %._crit_edge70.1.13

2590:                                             ; preds = %2581
  %2591 = mul nsw i64 %2582, %const_reg_qword7, !spirv.Decorations !836
  %2592 = getelementptr float, float addrspace(4)* %160, i64 %2591
  %2593 = getelementptr float, float addrspace(4)* %2592, i64 %2583
  %2594 = addrspacecast float addrspace(4)* %2593 to float addrspace(1)*
  %2595 = load float, float addrspace(1)* %2594, align 4
  %2596 = fmul reassoc nsz arcp contract float %2595, %4, !spirv.Decorations !843
  %2597 = fadd reassoc nsz arcp contract float %2586, %2596, !spirv.Decorations !843
  %2598 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2585
  %2599 = addrspacecast float addrspace(4)* %2598 to float addrspace(1)*
  store float %2597, float addrspace(1)* %2599, align 4
  br label %._crit_edge70.1.13

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13, %2590, %2587
  br i1 %144, label %2600, label %._crit_edge70.2.13

2600:                                             ; preds = %._crit_edge70.1.13
  %2601 = sext i32 %62 to i64
  %2602 = sext i32 %140 to i64
  %2603 = mul nsw i64 %2601, %const_reg_qword9, !spirv.Decorations !836
  %2604 = add nsw i64 %2603, %2602, !spirv.Decorations !836
  %2605 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !843
  br i1 %42, label %2609, label %2606

2606:                                             ; preds = %2600
  %2607 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2604
  %2608 = addrspacecast float addrspace(4)* %2607 to float addrspace(1)*
  store float %2605, float addrspace(1)* %2608, align 4
  br label %._crit_edge70.2.13

2609:                                             ; preds = %2600
  %2610 = mul nsw i64 %2601, %const_reg_qword7, !spirv.Decorations !836
  %2611 = getelementptr float, float addrspace(4)* %160, i64 %2610
  %2612 = getelementptr float, float addrspace(4)* %2611, i64 %2602
  %2613 = addrspacecast float addrspace(4)* %2612 to float addrspace(1)*
  %2614 = load float, float addrspace(1)* %2613, align 4
  %2615 = fmul reassoc nsz arcp contract float %2614, %4, !spirv.Decorations !843
  %2616 = fadd reassoc nsz arcp contract float %2605, %2615, !spirv.Decorations !843
  %2617 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2604
  %2618 = addrspacecast float addrspace(4)* %2617 to float addrspace(1)*
  store float %2616, float addrspace(1)* %2618, align 4
  br label %._crit_edge70.2.13

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13, %2609, %2606
  br i1 %145, label %2619, label %.preheader1.13

2619:                                             ; preds = %._crit_edge70.2.13
  %2620 = sext i32 %65 to i64
  %2621 = sext i32 %140 to i64
  %2622 = mul nsw i64 %2620, %const_reg_qword9, !spirv.Decorations !836
  %2623 = add nsw i64 %2622, %2621, !spirv.Decorations !836
  %2624 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !843
  br i1 %42, label %2628, label %2625

2625:                                             ; preds = %2619
  %2626 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2623
  %2627 = addrspacecast float addrspace(4)* %2626 to float addrspace(1)*
  store float %2624, float addrspace(1)* %2627, align 4
  br label %.preheader1.13

2628:                                             ; preds = %2619
  %2629 = mul nsw i64 %2620, %const_reg_qword7, !spirv.Decorations !836
  %2630 = getelementptr float, float addrspace(4)* %160, i64 %2629
  %2631 = getelementptr float, float addrspace(4)* %2630, i64 %2621
  %2632 = addrspacecast float addrspace(4)* %2631 to float addrspace(1)*
  %2633 = load float, float addrspace(1)* %2632, align 4
  %2634 = fmul reassoc nsz arcp contract float %2633, %4, !spirv.Decorations !843
  %2635 = fadd reassoc nsz arcp contract float %2624, %2634, !spirv.Decorations !843
  %2636 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2623
  %2637 = addrspacecast float addrspace(4)* %2636 to float addrspace(1)*
  store float %2635, float addrspace(1)* %2637, align 4
  br label %.preheader1.13

.preheader1.13:                                   ; preds = %._crit_edge70.2.13, %2628, %2625
  br i1 %148, label %2638, label %._crit_edge70.14

2638:                                             ; preds = %.preheader1.13
  %2639 = sext i32 %29 to i64
  %2640 = sext i32 %146 to i64
  %2641 = mul nsw i64 %2639, %const_reg_qword9, !spirv.Decorations !836
  %2642 = add nsw i64 %2641, %2640, !spirv.Decorations !836
  %2643 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !843
  br i1 %42, label %2647, label %2644

2644:                                             ; preds = %2638
  %2645 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2642
  %2646 = addrspacecast float addrspace(4)* %2645 to float addrspace(1)*
  store float %2643, float addrspace(1)* %2646, align 4
  br label %._crit_edge70.14

2647:                                             ; preds = %2638
  %2648 = mul nsw i64 %2639, %const_reg_qword7, !spirv.Decorations !836
  %2649 = getelementptr float, float addrspace(4)* %160, i64 %2648
  %2650 = getelementptr float, float addrspace(4)* %2649, i64 %2640
  %2651 = addrspacecast float addrspace(4)* %2650 to float addrspace(1)*
  %2652 = load float, float addrspace(1)* %2651, align 4
  %2653 = fmul reassoc nsz arcp contract float %2652, %4, !spirv.Decorations !843
  %2654 = fadd reassoc nsz arcp contract float %2643, %2653, !spirv.Decorations !843
  %2655 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2642
  %2656 = addrspacecast float addrspace(4)* %2655 to float addrspace(1)*
  store float %2654, float addrspace(1)* %2656, align 4
  br label %._crit_edge70.14

._crit_edge70.14:                                 ; preds = %.preheader1.13, %2647, %2644
  br i1 %149, label %2657, label %._crit_edge70.1.14

2657:                                             ; preds = %._crit_edge70.14
  %2658 = sext i32 %59 to i64
  %2659 = sext i32 %146 to i64
  %2660 = mul nsw i64 %2658, %const_reg_qword9, !spirv.Decorations !836
  %2661 = add nsw i64 %2660, %2659, !spirv.Decorations !836
  %2662 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !843
  br i1 %42, label %2666, label %2663

2663:                                             ; preds = %2657
  %2664 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2661
  %2665 = addrspacecast float addrspace(4)* %2664 to float addrspace(1)*
  store float %2662, float addrspace(1)* %2665, align 4
  br label %._crit_edge70.1.14

2666:                                             ; preds = %2657
  %2667 = mul nsw i64 %2658, %const_reg_qword7, !spirv.Decorations !836
  %2668 = getelementptr float, float addrspace(4)* %160, i64 %2667
  %2669 = getelementptr float, float addrspace(4)* %2668, i64 %2659
  %2670 = addrspacecast float addrspace(4)* %2669 to float addrspace(1)*
  %2671 = load float, float addrspace(1)* %2670, align 4
  %2672 = fmul reassoc nsz arcp contract float %2671, %4, !spirv.Decorations !843
  %2673 = fadd reassoc nsz arcp contract float %2662, %2672, !spirv.Decorations !843
  %2674 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2661
  %2675 = addrspacecast float addrspace(4)* %2674 to float addrspace(1)*
  store float %2673, float addrspace(1)* %2675, align 4
  br label %._crit_edge70.1.14

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14, %2666, %2663
  br i1 %150, label %2676, label %._crit_edge70.2.14

2676:                                             ; preds = %._crit_edge70.1.14
  %2677 = sext i32 %62 to i64
  %2678 = sext i32 %146 to i64
  %2679 = mul nsw i64 %2677, %const_reg_qword9, !spirv.Decorations !836
  %2680 = add nsw i64 %2679, %2678, !spirv.Decorations !836
  %2681 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !843
  br i1 %42, label %2685, label %2682

2682:                                             ; preds = %2676
  %2683 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2680
  %2684 = addrspacecast float addrspace(4)* %2683 to float addrspace(1)*
  store float %2681, float addrspace(1)* %2684, align 4
  br label %._crit_edge70.2.14

2685:                                             ; preds = %2676
  %2686 = mul nsw i64 %2677, %const_reg_qword7, !spirv.Decorations !836
  %2687 = getelementptr float, float addrspace(4)* %160, i64 %2686
  %2688 = getelementptr float, float addrspace(4)* %2687, i64 %2678
  %2689 = addrspacecast float addrspace(4)* %2688 to float addrspace(1)*
  %2690 = load float, float addrspace(1)* %2689, align 4
  %2691 = fmul reassoc nsz arcp contract float %2690, %4, !spirv.Decorations !843
  %2692 = fadd reassoc nsz arcp contract float %2681, %2691, !spirv.Decorations !843
  %2693 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2680
  %2694 = addrspacecast float addrspace(4)* %2693 to float addrspace(1)*
  store float %2692, float addrspace(1)* %2694, align 4
  br label %._crit_edge70.2.14

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14, %2685, %2682
  br i1 %151, label %2695, label %.preheader1.14

2695:                                             ; preds = %._crit_edge70.2.14
  %2696 = sext i32 %65 to i64
  %2697 = sext i32 %146 to i64
  %2698 = mul nsw i64 %2696, %const_reg_qword9, !spirv.Decorations !836
  %2699 = add nsw i64 %2698, %2697, !spirv.Decorations !836
  %2700 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !843
  br i1 %42, label %2704, label %2701

2701:                                             ; preds = %2695
  %2702 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2699
  %2703 = addrspacecast float addrspace(4)* %2702 to float addrspace(1)*
  store float %2700, float addrspace(1)* %2703, align 4
  br label %.preheader1.14

2704:                                             ; preds = %2695
  %2705 = mul nsw i64 %2696, %const_reg_qword7, !spirv.Decorations !836
  %2706 = getelementptr float, float addrspace(4)* %160, i64 %2705
  %2707 = getelementptr float, float addrspace(4)* %2706, i64 %2697
  %2708 = addrspacecast float addrspace(4)* %2707 to float addrspace(1)*
  %2709 = load float, float addrspace(1)* %2708, align 4
  %2710 = fmul reassoc nsz arcp contract float %2709, %4, !spirv.Decorations !843
  %2711 = fadd reassoc nsz arcp contract float %2700, %2710, !spirv.Decorations !843
  %2712 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2699
  %2713 = addrspacecast float addrspace(4)* %2712 to float addrspace(1)*
  store float %2711, float addrspace(1)* %2713, align 4
  br label %.preheader1.14

.preheader1.14:                                   ; preds = %._crit_edge70.2.14, %2704, %2701
  br i1 %154, label %2714, label %._crit_edge70.15

2714:                                             ; preds = %.preheader1.14
  %2715 = sext i32 %29 to i64
  %2716 = sext i32 %152 to i64
  %2717 = mul nsw i64 %2715, %const_reg_qword9, !spirv.Decorations !836
  %2718 = add nsw i64 %2717, %2716, !spirv.Decorations !836
  %2719 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !843
  br i1 %42, label %2723, label %2720

2720:                                             ; preds = %2714
  %2721 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2718
  %2722 = addrspacecast float addrspace(4)* %2721 to float addrspace(1)*
  store float %2719, float addrspace(1)* %2722, align 4
  br label %._crit_edge70.15

2723:                                             ; preds = %2714
  %2724 = mul nsw i64 %2715, %const_reg_qword7, !spirv.Decorations !836
  %2725 = getelementptr float, float addrspace(4)* %160, i64 %2724
  %2726 = getelementptr float, float addrspace(4)* %2725, i64 %2716
  %2727 = addrspacecast float addrspace(4)* %2726 to float addrspace(1)*
  %2728 = load float, float addrspace(1)* %2727, align 4
  %2729 = fmul reassoc nsz arcp contract float %2728, %4, !spirv.Decorations !843
  %2730 = fadd reassoc nsz arcp contract float %2719, %2729, !spirv.Decorations !843
  %2731 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2718
  %2732 = addrspacecast float addrspace(4)* %2731 to float addrspace(1)*
  store float %2730, float addrspace(1)* %2732, align 4
  br label %._crit_edge70.15

._crit_edge70.15:                                 ; preds = %.preheader1.14, %2723, %2720
  br i1 %155, label %2733, label %._crit_edge70.1.15

2733:                                             ; preds = %._crit_edge70.15
  %2734 = sext i32 %59 to i64
  %2735 = sext i32 %152 to i64
  %2736 = mul nsw i64 %2734, %const_reg_qword9, !spirv.Decorations !836
  %2737 = add nsw i64 %2736, %2735, !spirv.Decorations !836
  %2738 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !843
  br i1 %42, label %2742, label %2739

2739:                                             ; preds = %2733
  %2740 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2737
  %2741 = addrspacecast float addrspace(4)* %2740 to float addrspace(1)*
  store float %2738, float addrspace(1)* %2741, align 4
  br label %._crit_edge70.1.15

2742:                                             ; preds = %2733
  %2743 = mul nsw i64 %2734, %const_reg_qword7, !spirv.Decorations !836
  %2744 = getelementptr float, float addrspace(4)* %160, i64 %2743
  %2745 = getelementptr float, float addrspace(4)* %2744, i64 %2735
  %2746 = addrspacecast float addrspace(4)* %2745 to float addrspace(1)*
  %2747 = load float, float addrspace(1)* %2746, align 4
  %2748 = fmul reassoc nsz arcp contract float %2747, %4, !spirv.Decorations !843
  %2749 = fadd reassoc nsz arcp contract float %2738, %2748, !spirv.Decorations !843
  %2750 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2737
  %2751 = addrspacecast float addrspace(4)* %2750 to float addrspace(1)*
  store float %2749, float addrspace(1)* %2751, align 4
  br label %._crit_edge70.1.15

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15, %2742, %2739
  br i1 %156, label %2752, label %._crit_edge70.2.15

2752:                                             ; preds = %._crit_edge70.1.15
  %2753 = sext i32 %62 to i64
  %2754 = sext i32 %152 to i64
  %2755 = mul nsw i64 %2753, %const_reg_qword9, !spirv.Decorations !836
  %2756 = add nsw i64 %2755, %2754, !spirv.Decorations !836
  %2757 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !843
  br i1 %42, label %2761, label %2758

2758:                                             ; preds = %2752
  %2759 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2756
  %2760 = addrspacecast float addrspace(4)* %2759 to float addrspace(1)*
  store float %2757, float addrspace(1)* %2760, align 4
  br label %._crit_edge70.2.15

2761:                                             ; preds = %2752
  %2762 = mul nsw i64 %2753, %const_reg_qword7, !spirv.Decorations !836
  %2763 = getelementptr float, float addrspace(4)* %160, i64 %2762
  %2764 = getelementptr float, float addrspace(4)* %2763, i64 %2754
  %2765 = addrspacecast float addrspace(4)* %2764 to float addrspace(1)*
  %2766 = load float, float addrspace(1)* %2765, align 4
  %2767 = fmul reassoc nsz arcp contract float %2766, %4, !spirv.Decorations !843
  %2768 = fadd reassoc nsz arcp contract float %2757, %2767, !spirv.Decorations !843
  %2769 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2756
  %2770 = addrspacecast float addrspace(4)* %2769 to float addrspace(1)*
  store float %2768, float addrspace(1)* %2770, align 4
  br label %._crit_edge70.2.15

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15, %2761, %2758
  br i1 %157, label %2771, label %.preheader1.15

2771:                                             ; preds = %._crit_edge70.2.15
  %2772 = sext i32 %65 to i64
  %2773 = sext i32 %152 to i64
  %2774 = mul nsw i64 %2772, %const_reg_qword9, !spirv.Decorations !836
  %2775 = add nsw i64 %2774, %2773, !spirv.Decorations !836
  %2776 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !843
  br i1 %42, label %2780, label %2777

2777:                                             ; preds = %2771
  %2778 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2775
  %2779 = addrspacecast float addrspace(4)* %2778 to float addrspace(1)*
  store float %2776, float addrspace(1)* %2779, align 4
  br label %.preheader1.15

2780:                                             ; preds = %2771
  %2781 = mul nsw i64 %2772, %const_reg_qword7, !spirv.Decorations !836
  %2782 = getelementptr float, float addrspace(4)* %160, i64 %2781
  %2783 = getelementptr float, float addrspace(4)* %2782, i64 %2773
  %2784 = addrspacecast float addrspace(4)* %2783 to float addrspace(1)*
  %2785 = load float, float addrspace(1)* %2784, align 4
  %2786 = fmul reassoc nsz arcp contract float %2785, %4, !spirv.Decorations !843
  %2787 = fadd reassoc nsz arcp contract float %2776, %2786, !spirv.Decorations !843
  %2788 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2775
  %2789 = addrspacecast float addrspace(4)* %2788 to float addrspace(1)*
  store float %2787, float addrspace(1)* %2789, align 4
  br label %.preheader1.15

.preheader1.15:                                   ; preds = %._crit_edge70.2.15, %2780, %2777
  %2790 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %52
  %2791 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %53
  %.idx = select i1 %42, i64 %54, i64 0
  %2792 = getelementptr float, float addrspace(4)* %160, i64 %.idx
  %2793 = getelementptr inbounds float, float addrspace(4)* %159, i64 %55
  %2794 = add i32 %158, %14
  %2795 = icmp slt i32 %2794, %8
  br i1 %2795, label %.preheader2.preheader, label %._crit_edge72

._crit_edge72:                                    ; preds = %.preheader1.15, %13
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %21 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %22 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %23 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %24 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %24)
  %25 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %25)
  %26 = mul i32 %17, %15
  %27 = zext i16 %localIdX to i32
  %28 = add i32 %26, %27
  %29 = shl i32 %28, 2
  %30 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %18, %16
  %33 = zext i16 %localIdY to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 2
  %36 = zext i32 %19 to i64
  %37 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %37)
  %38 = mul nsw i64 %36, %9, !spirv.Decorations !836
  %39 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %23, i64 %38
  %40 = mul nsw i64 %36, %10, !spirv.Decorations !836
  %41 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %22, i64 %40
  %42 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !843
  %43 = mul nsw i64 %36, %11, !spirv.Decorations !836
  %44 = select i1 %42, i64 %43, i64 0
  %45 = getelementptr inbounds float, float addrspace(4)* %21, i64 %44
  %46 = mul nsw i64 %36, %12, !spirv.Decorations !836
  %47 = getelementptr inbounds float, float addrspace(4)* %20, i64 %46
  %48 = icmp slt i32 %19, %8
  br i1 %48, label %.lr.ph, label %._crit_edge72

.lr.ph:                                           ; preds = %13
  %49 = icmp sgt i32 %const_reg_dword2, 0
  %50 = zext i32 %14 to i64
  %51 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %51)
  %52 = mul nsw i64 %50, %9, !spirv.Decorations !836
  %53 = mul nsw i64 %50, %10, !spirv.Decorations !836
  %54 = mul nsw i64 %50, %11
  %55 = mul nsw i64 %50, %12, !spirv.Decorations !836
  %56 = icmp slt i32 %35, %const_reg_dword1
  %57 = icmp slt i32 %29, %const_reg_dword
  %58 = and i1 %57, %56
  %59 = or i32 %29, 1
  %60 = icmp slt i32 %59, %const_reg_dword
  %61 = and i1 %60, %56
  %62 = or i32 %29, 2
  %63 = icmp slt i32 %62, %const_reg_dword
  %64 = and i1 %63, %56
  %65 = or i32 %29, 3
  %66 = icmp slt i32 %65, %const_reg_dword
  %67 = and i1 %66, %56
  %68 = or i32 %35, 1
  %69 = icmp slt i32 %68, %const_reg_dword1
  %70 = and i1 %57, %69
  %71 = and i1 %60, %69
  %72 = and i1 %63, %69
  %73 = and i1 %66, %69
  %74 = or i32 %35, 2
  %75 = icmp slt i32 %74, %const_reg_dword1
  %76 = and i1 %57, %75
  %77 = and i1 %60, %75
  %78 = and i1 %63, %75
  %79 = and i1 %66, %75
  %80 = or i32 %35, 3
  %81 = icmp slt i32 %80, %const_reg_dword1
  %82 = and i1 %57, %81
  %83 = and i1 %60, %81
  %84 = and i1 %63, %81
  %85 = and i1 %66, %81
  br label %.preheader2.preheader

.preheader2.preheader:                            ; preds = %.lr.ph, %.preheader1.3
  %86 = phi i32 [ %19, %.lr.ph ], [ %770, %.preheader1.3 ]
  %87 = phi float addrspace(4)* [ %47, %.lr.ph ], [ %769, %.preheader1.3 ]
  %88 = phi float addrspace(4)* [ %45, %.lr.ph ], [ %768, %.preheader1.3 ]
  %89 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %41, %.lr.ph ], [ %767, %.preheader1.3 ]
  %90 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %39, %.lr.ph ], [ %766, %.preheader1.3 ]
  br i1 %49, label %.preheader.preheader, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.preheader.3, %.preheader2.preheader
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader ], [ %459, %.preheader.3 ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader ], [ %371, %.preheader.3 ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader ], [ %283, %.preheader.3 ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader ], [ %195, %.preheader.3 ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader ], [ %437, %.preheader.3 ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader ], [ %349, %.preheader.3 ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader ], [ %261, %.preheader.3 ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader ], [ %173, %.preheader.3 ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader ], [ %415, %.preheader.3 ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader ], [ %327, %.preheader.3 ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader ], [ %239, %.preheader.3 ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader ], [ %151, %.preheader.3 ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader ], [ %393, %.preheader.3 ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader ], [ %305, %.preheader.3 ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader ], [ %217, %.preheader.3 ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader ], [ %129, %.preheader.3 ]
  br i1 %58, label %462, label %._crit_edge70

.preheader.preheader:                             ; preds = %.preheader2.preheader, %.preheader.3
  %91 = phi float [ %459, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %92 = phi float [ %437, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %93 = phi float [ %415, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %94 = phi float [ %393, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %95 = phi float [ %371, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %96 = phi float [ %349, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %97 = phi float [ %327, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %98 = phi float [ %305, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %99 = phi float [ %283, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %100 = phi float [ %261, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %101 = phi float [ %239, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %102 = phi float [ %217, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %103 = phi float [ %195, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %104 = phi float [ %173, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %105 = phi float [ %151, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %106 = phi float [ %129, %.preheader.3 ], [ %7, %.preheader2.preheader ]
  %107 = phi i32 [ %460, %.preheader.3 ], [ 0, %.preheader2.preheader ]
  br i1 %58, label %108, label %._crit_edge

108:                                              ; preds = %.preheader.preheader
  %.sroa.64.0.insert.ext = zext i32 %107 to i64
  %109 = sext i32 %29 to i64
  %110 = mul nsw i64 %109, %const_reg_qword3, !spirv.Decorations !836
  %111 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %110, i32 0
  %112 = getelementptr i16, i16 addrspace(4)* %111, i64 %.sroa.64.0.insert.ext
  %113 = addrspacecast i16 addrspace(4)* %112 to i16 addrspace(1)*
  %114 = load i16, i16 addrspace(1)* %113, align 2
  %115 = mul nsw i64 %.sroa.64.0.insert.ext, %const_reg_qword5, !spirv.Decorations !836
  %116 = sext i32 %35 to i64
  %117 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %115, i32 0
  %118 = getelementptr i16, i16 addrspace(4)* %117, i64 %116
  %119 = addrspacecast i16 addrspace(4)* %118 to i16 addrspace(1)*
  %120 = load i16, i16 addrspace(1)* %119, align 2
  %121 = zext i16 %114 to i32
  %122 = shl nuw i32 %121, 16, !spirv.Decorations !838
  %123 = bitcast i32 %122 to float
  %124 = zext i16 %120 to i32
  %125 = shl nuw i32 %124, 16, !spirv.Decorations !838
  %126 = bitcast i32 %125 to float
  %127 = fmul reassoc nsz arcp contract float %123, %126, !spirv.Decorations !843
  %128 = fadd reassoc nsz arcp contract float %127, %106, !spirv.Decorations !843
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %108
  %129 = phi float [ %128, %108 ], [ %106, %.preheader.preheader ]
  br i1 %61, label %130, label %._crit_edge.1

130:                                              ; preds = %._crit_edge
  %.sroa.64.0.insert.ext203 = zext i32 %107 to i64
  %131 = sext i32 %59 to i64
  %132 = mul nsw i64 %131, %const_reg_qword3, !spirv.Decorations !836
  %133 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %132, i32 0
  %134 = getelementptr i16, i16 addrspace(4)* %133, i64 %.sroa.64.0.insert.ext203
  %135 = addrspacecast i16 addrspace(4)* %134 to i16 addrspace(1)*
  %136 = load i16, i16 addrspace(1)* %135, align 2
  %137 = mul nsw i64 %.sroa.64.0.insert.ext203, %const_reg_qword5, !spirv.Decorations !836
  %138 = sext i32 %35 to i64
  %139 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %137, i32 0
  %140 = getelementptr i16, i16 addrspace(4)* %139, i64 %138
  %141 = addrspacecast i16 addrspace(4)* %140 to i16 addrspace(1)*
  %142 = load i16, i16 addrspace(1)* %141, align 2
  %143 = zext i16 %136 to i32
  %144 = shl nuw i32 %143, 16, !spirv.Decorations !838
  %145 = bitcast i32 %144 to float
  %146 = zext i16 %142 to i32
  %147 = shl nuw i32 %146, 16, !spirv.Decorations !838
  %148 = bitcast i32 %147 to float
  %149 = fmul reassoc nsz arcp contract float %145, %148, !spirv.Decorations !843
  %150 = fadd reassoc nsz arcp contract float %149, %105, !spirv.Decorations !843
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %130
  %151 = phi float [ %150, %130 ], [ %105, %._crit_edge ]
  br i1 %64, label %152, label %._crit_edge.2

152:                                              ; preds = %._crit_edge.1
  %.sroa.64.0.insert.ext208 = zext i32 %107 to i64
  %153 = sext i32 %62 to i64
  %154 = mul nsw i64 %153, %const_reg_qword3, !spirv.Decorations !836
  %155 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %154, i32 0
  %156 = getelementptr i16, i16 addrspace(4)* %155, i64 %.sroa.64.0.insert.ext208
  %157 = addrspacecast i16 addrspace(4)* %156 to i16 addrspace(1)*
  %158 = load i16, i16 addrspace(1)* %157, align 2
  %159 = mul nsw i64 %.sroa.64.0.insert.ext208, %const_reg_qword5, !spirv.Decorations !836
  %160 = sext i32 %35 to i64
  %161 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %159, i32 0
  %162 = getelementptr i16, i16 addrspace(4)* %161, i64 %160
  %163 = addrspacecast i16 addrspace(4)* %162 to i16 addrspace(1)*
  %164 = load i16, i16 addrspace(1)* %163, align 2
  %165 = zext i16 %158 to i32
  %166 = shl nuw i32 %165, 16, !spirv.Decorations !838
  %167 = bitcast i32 %166 to float
  %168 = zext i16 %164 to i32
  %169 = shl nuw i32 %168, 16, !spirv.Decorations !838
  %170 = bitcast i32 %169 to float
  %171 = fmul reassoc nsz arcp contract float %167, %170, !spirv.Decorations !843
  %172 = fadd reassoc nsz arcp contract float %171, %104, !spirv.Decorations !843
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %152
  %173 = phi float [ %172, %152 ], [ %104, %._crit_edge.1 ]
  br i1 %67, label %174, label %.preheader

174:                                              ; preds = %._crit_edge.2
  %.sroa.64.0.insert.ext213 = zext i32 %107 to i64
  %175 = sext i32 %65 to i64
  %176 = mul nsw i64 %175, %const_reg_qword3, !spirv.Decorations !836
  %177 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %176, i32 0
  %178 = getelementptr i16, i16 addrspace(4)* %177, i64 %.sroa.64.0.insert.ext213
  %179 = addrspacecast i16 addrspace(4)* %178 to i16 addrspace(1)*
  %180 = load i16, i16 addrspace(1)* %179, align 2
  %181 = mul nsw i64 %.sroa.64.0.insert.ext213, %const_reg_qword5, !spirv.Decorations !836
  %182 = sext i32 %35 to i64
  %183 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %181, i32 0
  %184 = getelementptr i16, i16 addrspace(4)* %183, i64 %182
  %185 = addrspacecast i16 addrspace(4)* %184 to i16 addrspace(1)*
  %186 = load i16, i16 addrspace(1)* %185, align 2
  %187 = zext i16 %180 to i32
  %188 = shl nuw i32 %187, 16, !spirv.Decorations !838
  %189 = bitcast i32 %188 to float
  %190 = zext i16 %186 to i32
  %191 = shl nuw i32 %190, 16, !spirv.Decorations !838
  %192 = bitcast i32 %191 to float
  %193 = fmul reassoc nsz arcp contract float %189, %192, !spirv.Decorations !843
  %194 = fadd reassoc nsz arcp contract float %193, %103, !spirv.Decorations !843
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %174
  %195 = phi float [ %194, %174 ], [ %103, %._crit_edge.2 ]
  br i1 %70, label %196, label %._crit_edge.173

196:                                              ; preds = %.preheader
  %.sroa.64.0.insert.ext218 = zext i32 %107 to i64
  %197 = sext i32 %29 to i64
  %198 = mul nsw i64 %197, %const_reg_qword3, !spirv.Decorations !836
  %199 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %198, i32 0
  %200 = getelementptr i16, i16 addrspace(4)* %199, i64 %.sroa.64.0.insert.ext218
  %201 = addrspacecast i16 addrspace(4)* %200 to i16 addrspace(1)*
  %202 = load i16, i16 addrspace(1)* %201, align 2
  %203 = mul nsw i64 %.sroa.64.0.insert.ext218, %const_reg_qword5, !spirv.Decorations !836
  %204 = sext i32 %68 to i64
  %205 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %203, i32 0
  %206 = getelementptr i16, i16 addrspace(4)* %205, i64 %204
  %207 = addrspacecast i16 addrspace(4)* %206 to i16 addrspace(1)*
  %208 = load i16, i16 addrspace(1)* %207, align 2
  %209 = zext i16 %202 to i32
  %210 = shl nuw i32 %209, 16, !spirv.Decorations !838
  %211 = bitcast i32 %210 to float
  %212 = zext i16 %208 to i32
  %213 = shl nuw i32 %212, 16, !spirv.Decorations !838
  %214 = bitcast i32 %213 to float
  %215 = fmul reassoc nsz arcp contract float %211, %214, !spirv.Decorations !843
  %216 = fadd reassoc nsz arcp contract float %215, %102, !spirv.Decorations !843
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %196
  %217 = phi float [ %216, %196 ], [ %102, %.preheader ]
  br i1 %71, label %218, label %._crit_edge.1.1

218:                                              ; preds = %._crit_edge.173
  %.sroa.64.0.insert.ext223 = zext i32 %107 to i64
  %219 = sext i32 %59 to i64
  %220 = mul nsw i64 %219, %const_reg_qword3, !spirv.Decorations !836
  %221 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %220, i32 0
  %222 = getelementptr i16, i16 addrspace(4)* %221, i64 %.sroa.64.0.insert.ext223
  %223 = addrspacecast i16 addrspace(4)* %222 to i16 addrspace(1)*
  %224 = load i16, i16 addrspace(1)* %223, align 2
  %225 = mul nsw i64 %.sroa.64.0.insert.ext223, %const_reg_qword5, !spirv.Decorations !836
  %226 = sext i32 %68 to i64
  %227 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %225, i32 0
  %228 = getelementptr i16, i16 addrspace(4)* %227, i64 %226
  %229 = addrspacecast i16 addrspace(4)* %228 to i16 addrspace(1)*
  %230 = load i16, i16 addrspace(1)* %229, align 2
  %231 = zext i16 %224 to i32
  %232 = shl nuw i32 %231, 16, !spirv.Decorations !838
  %233 = bitcast i32 %232 to float
  %234 = zext i16 %230 to i32
  %235 = shl nuw i32 %234, 16, !spirv.Decorations !838
  %236 = bitcast i32 %235 to float
  %237 = fmul reassoc nsz arcp contract float %233, %236, !spirv.Decorations !843
  %238 = fadd reassoc nsz arcp contract float %237, %101, !spirv.Decorations !843
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %218
  %239 = phi float [ %238, %218 ], [ %101, %._crit_edge.173 ]
  br i1 %72, label %240, label %._crit_edge.2.1

240:                                              ; preds = %._crit_edge.1.1
  %.sroa.64.0.insert.ext228 = zext i32 %107 to i64
  %241 = sext i32 %62 to i64
  %242 = mul nsw i64 %241, %const_reg_qword3, !spirv.Decorations !836
  %243 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %242, i32 0
  %244 = getelementptr i16, i16 addrspace(4)* %243, i64 %.sroa.64.0.insert.ext228
  %245 = addrspacecast i16 addrspace(4)* %244 to i16 addrspace(1)*
  %246 = load i16, i16 addrspace(1)* %245, align 2
  %247 = mul nsw i64 %.sroa.64.0.insert.ext228, %const_reg_qword5, !spirv.Decorations !836
  %248 = sext i32 %68 to i64
  %249 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %247, i32 0
  %250 = getelementptr i16, i16 addrspace(4)* %249, i64 %248
  %251 = addrspacecast i16 addrspace(4)* %250 to i16 addrspace(1)*
  %252 = load i16, i16 addrspace(1)* %251, align 2
  %253 = zext i16 %246 to i32
  %254 = shl nuw i32 %253, 16, !spirv.Decorations !838
  %255 = bitcast i32 %254 to float
  %256 = zext i16 %252 to i32
  %257 = shl nuw i32 %256, 16, !spirv.Decorations !838
  %258 = bitcast i32 %257 to float
  %259 = fmul reassoc nsz arcp contract float %255, %258, !spirv.Decorations !843
  %260 = fadd reassoc nsz arcp contract float %259, %100, !spirv.Decorations !843
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %240
  %261 = phi float [ %260, %240 ], [ %100, %._crit_edge.1.1 ]
  br i1 %73, label %262, label %.preheader.1

262:                                              ; preds = %._crit_edge.2.1
  %.sroa.64.0.insert.ext233 = zext i32 %107 to i64
  %263 = sext i32 %65 to i64
  %264 = mul nsw i64 %263, %const_reg_qword3, !spirv.Decorations !836
  %265 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %264, i32 0
  %266 = getelementptr i16, i16 addrspace(4)* %265, i64 %.sroa.64.0.insert.ext233
  %267 = addrspacecast i16 addrspace(4)* %266 to i16 addrspace(1)*
  %268 = load i16, i16 addrspace(1)* %267, align 2
  %269 = mul nsw i64 %.sroa.64.0.insert.ext233, %const_reg_qword5, !spirv.Decorations !836
  %270 = sext i32 %68 to i64
  %271 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %269, i32 0
  %272 = getelementptr i16, i16 addrspace(4)* %271, i64 %270
  %273 = addrspacecast i16 addrspace(4)* %272 to i16 addrspace(1)*
  %274 = load i16, i16 addrspace(1)* %273, align 2
  %275 = zext i16 %268 to i32
  %276 = shl nuw i32 %275, 16, !spirv.Decorations !838
  %277 = bitcast i32 %276 to float
  %278 = zext i16 %274 to i32
  %279 = shl nuw i32 %278, 16, !spirv.Decorations !838
  %280 = bitcast i32 %279 to float
  %281 = fmul reassoc nsz arcp contract float %277, %280, !spirv.Decorations !843
  %282 = fadd reassoc nsz arcp contract float %281, %99, !spirv.Decorations !843
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %262
  %283 = phi float [ %282, %262 ], [ %99, %._crit_edge.2.1 ]
  br i1 %76, label %284, label %._crit_edge.274

284:                                              ; preds = %.preheader.1
  %.sroa.64.0.insert.ext238 = zext i32 %107 to i64
  %285 = sext i32 %29 to i64
  %286 = mul nsw i64 %285, %const_reg_qword3, !spirv.Decorations !836
  %287 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %286, i32 0
  %288 = getelementptr i16, i16 addrspace(4)* %287, i64 %.sroa.64.0.insert.ext238
  %289 = addrspacecast i16 addrspace(4)* %288 to i16 addrspace(1)*
  %290 = load i16, i16 addrspace(1)* %289, align 2
  %291 = mul nsw i64 %.sroa.64.0.insert.ext238, %const_reg_qword5, !spirv.Decorations !836
  %292 = sext i32 %74 to i64
  %293 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %291, i32 0
  %294 = getelementptr i16, i16 addrspace(4)* %293, i64 %292
  %295 = addrspacecast i16 addrspace(4)* %294 to i16 addrspace(1)*
  %296 = load i16, i16 addrspace(1)* %295, align 2
  %297 = zext i16 %290 to i32
  %298 = shl nuw i32 %297, 16, !spirv.Decorations !838
  %299 = bitcast i32 %298 to float
  %300 = zext i16 %296 to i32
  %301 = shl nuw i32 %300, 16, !spirv.Decorations !838
  %302 = bitcast i32 %301 to float
  %303 = fmul reassoc nsz arcp contract float %299, %302, !spirv.Decorations !843
  %304 = fadd reassoc nsz arcp contract float %303, %98, !spirv.Decorations !843
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %284
  %305 = phi float [ %304, %284 ], [ %98, %.preheader.1 ]
  br i1 %77, label %306, label %._crit_edge.1.2

306:                                              ; preds = %._crit_edge.274
  %.sroa.64.0.insert.ext243 = zext i32 %107 to i64
  %307 = sext i32 %59 to i64
  %308 = mul nsw i64 %307, %const_reg_qword3, !spirv.Decorations !836
  %309 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %308, i32 0
  %310 = getelementptr i16, i16 addrspace(4)* %309, i64 %.sroa.64.0.insert.ext243
  %311 = addrspacecast i16 addrspace(4)* %310 to i16 addrspace(1)*
  %312 = load i16, i16 addrspace(1)* %311, align 2
  %313 = mul nsw i64 %.sroa.64.0.insert.ext243, %const_reg_qword5, !spirv.Decorations !836
  %314 = sext i32 %74 to i64
  %315 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %313, i32 0
  %316 = getelementptr i16, i16 addrspace(4)* %315, i64 %314
  %317 = addrspacecast i16 addrspace(4)* %316 to i16 addrspace(1)*
  %318 = load i16, i16 addrspace(1)* %317, align 2
  %319 = zext i16 %312 to i32
  %320 = shl nuw i32 %319, 16, !spirv.Decorations !838
  %321 = bitcast i32 %320 to float
  %322 = zext i16 %318 to i32
  %323 = shl nuw i32 %322, 16, !spirv.Decorations !838
  %324 = bitcast i32 %323 to float
  %325 = fmul reassoc nsz arcp contract float %321, %324, !spirv.Decorations !843
  %326 = fadd reassoc nsz arcp contract float %325, %97, !spirv.Decorations !843
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %306
  %327 = phi float [ %326, %306 ], [ %97, %._crit_edge.274 ]
  br i1 %78, label %328, label %._crit_edge.2.2

328:                                              ; preds = %._crit_edge.1.2
  %.sroa.64.0.insert.ext248 = zext i32 %107 to i64
  %329 = sext i32 %62 to i64
  %330 = mul nsw i64 %329, %const_reg_qword3, !spirv.Decorations !836
  %331 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %330, i32 0
  %332 = getelementptr i16, i16 addrspace(4)* %331, i64 %.sroa.64.0.insert.ext248
  %333 = addrspacecast i16 addrspace(4)* %332 to i16 addrspace(1)*
  %334 = load i16, i16 addrspace(1)* %333, align 2
  %335 = mul nsw i64 %.sroa.64.0.insert.ext248, %const_reg_qword5, !spirv.Decorations !836
  %336 = sext i32 %74 to i64
  %337 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %335, i32 0
  %338 = getelementptr i16, i16 addrspace(4)* %337, i64 %336
  %339 = addrspacecast i16 addrspace(4)* %338 to i16 addrspace(1)*
  %340 = load i16, i16 addrspace(1)* %339, align 2
  %341 = zext i16 %334 to i32
  %342 = shl nuw i32 %341, 16, !spirv.Decorations !838
  %343 = bitcast i32 %342 to float
  %344 = zext i16 %340 to i32
  %345 = shl nuw i32 %344, 16, !spirv.Decorations !838
  %346 = bitcast i32 %345 to float
  %347 = fmul reassoc nsz arcp contract float %343, %346, !spirv.Decorations !843
  %348 = fadd reassoc nsz arcp contract float %347, %96, !spirv.Decorations !843
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %328
  %349 = phi float [ %348, %328 ], [ %96, %._crit_edge.1.2 ]
  br i1 %79, label %350, label %.preheader.2

350:                                              ; preds = %._crit_edge.2.2
  %.sroa.64.0.insert.ext253 = zext i32 %107 to i64
  %351 = sext i32 %65 to i64
  %352 = mul nsw i64 %351, %const_reg_qword3, !spirv.Decorations !836
  %353 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %352, i32 0
  %354 = getelementptr i16, i16 addrspace(4)* %353, i64 %.sroa.64.0.insert.ext253
  %355 = addrspacecast i16 addrspace(4)* %354 to i16 addrspace(1)*
  %356 = load i16, i16 addrspace(1)* %355, align 2
  %357 = mul nsw i64 %.sroa.64.0.insert.ext253, %const_reg_qword5, !spirv.Decorations !836
  %358 = sext i32 %74 to i64
  %359 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %357, i32 0
  %360 = getelementptr i16, i16 addrspace(4)* %359, i64 %358
  %361 = addrspacecast i16 addrspace(4)* %360 to i16 addrspace(1)*
  %362 = load i16, i16 addrspace(1)* %361, align 2
  %363 = zext i16 %356 to i32
  %364 = shl nuw i32 %363, 16, !spirv.Decorations !838
  %365 = bitcast i32 %364 to float
  %366 = zext i16 %362 to i32
  %367 = shl nuw i32 %366, 16, !spirv.Decorations !838
  %368 = bitcast i32 %367 to float
  %369 = fmul reassoc nsz arcp contract float %365, %368, !spirv.Decorations !843
  %370 = fadd reassoc nsz arcp contract float %369, %95, !spirv.Decorations !843
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %350
  %371 = phi float [ %370, %350 ], [ %95, %._crit_edge.2.2 ]
  br i1 %82, label %372, label %._crit_edge.375

372:                                              ; preds = %.preheader.2
  %.sroa.64.0.insert.ext258 = zext i32 %107 to i64
  %373 = sext i32 %29 to i64
  %374 = mul nsw i64 %373, %const_reg_qword3, !spirv.Decorations !836
  %375 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %374, i32 0
  %376 = getelementptr i16, i16 addrspace(4)* %375, i64 %.sroa.64.0.insert.ext258
  %377 = addrspacecast i16 addrspace(4)* %376 to i16 addrspace(1)*
  %378 = load i16, i16 addrspace(1)* %377, align 2
  %379 = mul nsw i64 %.sroa.64.0.insert.ext258, %const_reg_qword5, !spirv.Decorations !836
  %380 = sext i32 %80 to i64
  %381 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %379, i32 0
  %382 = getelementptr i16, i16 addrspace(4)* %381, i64 %380
  %383 = addrspacecast i16 addrspace(4)* %382 to i16 addrspace(1)*
  %384 = load i16, i16 addrspace(1)* %383, align 2
  %385 = zext i16 %378 to i32
  %386 = shl nuw i32 %385, 16, !spirv.Decorations !838
  %387 = bitcast i32 %386 to float
  %388 = zext i16 %384 to i32
  %389 = shl nuw i32 %388, 16, !spirv.Decorations !838
  %390 = bitcast i32 %389 to float
  %391 = fmul reassoc nsz arcp contract float %387, %390, !spirv.Decorations !843
  %392 = fadd reassoc nsz arcp contract float %391, %94, !spirv.Decorations !843
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %372
  %393 = phi float [ %392, %372 ], [ %94, %.preheader.2 ]
  br i1 %83, label %394, label %._crit_edge.1.3

394:                                              ; preds = %._crit_edge.375
  %.sroa.64.0.insert.ext263 = zext i32 %107 to i64
  %395 = sext i32 %59 to i64
  %396 = mul nsw i64 %395, %const_reg_qword3, !spirv.Decorations !836
  %397 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %396, i32 0
  %398 = getelementptr i16, i16 addrspace(4)* %397, i64 %.sroa.64.0.insert.ext263
  %399 = addrspacecast i16 addrspace(4)* %398 to i16 addrspace(1)*
  %400 = load i16, i16 addrspace(1)* %399, align 2
  %401 = mul nsw i64 %.sroa.64.0.insert.ext263, %const_reg_qword5, !spirv.Decorations !836
  %402 = sext i32 %80 to i64
  %403 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %401, i32 0
  %404 = getelementptr i16, i16 addrspace(4)* %403, i64 %402
  %405 = addrspacecast i16 addrspace(4)* %404 to i16 addrspace(1)*
  %406 = load i16, i16 addrspace(1)* %405, align 2
  %407 = zext i16 %400 to i32
  %408 = shl nuw i32 %407, 16, !spirv.Decorations !838
  %409 = bitcast i32 %408 to float
  %410 = zext i16 %406 to i32
  %411 = shl nuw i32 %410, 16, !spirv.Decorations !838
  %412 = bitcast i32 %411 to float
  %413 = fmul reassoc nsz arcp contract float %409, %412, !spirv.Decorations !843
  %414 = fadd reassoc nsz arcp contract float %413, %93, !spirv.Decorations !843
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %394
  %415 = phi float [ %414, %394 ], [ %93, %._crit_edge.375 ]
  br i1 %84, label %416, label %._crit_edge.2.3

416:                                              ; preds = %._crit_edge.1.3
  %.sroa.64.0.insert.ext268 = zext i32 %107 to i64
  %417 = sext i32 %62 to i64
  %418 = mul nsw i64 %417, %const_reg_qword3, !spirv.Decorations !836
  %419 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %418, i32 0
  %420 = getelementptr i16, i16 addrspace(4)* %419, i64 %.sroa.64.0.insert.ext268
  %421 = addrspacecast i16 addrspace(4)* %420 to i16 addrspace(1)*
  %422 = load i16, i16 addrspace(1)* %421, align 2
  %423 = mul nsw i64 %.sroa.64.0.insert.ext268, %const_reg_qword5, !spirv.Decorations !836
  %424 = sext i32 %80 to i64
  %425 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %423, i32 0
  %426 = getelementptr i16, i16 addrspace(4)* %425, i64 %424
  %427 = addrspacecast i16 addrspace(4)* %426 to i16 addrspace(1)*
  %428 = load i16, i16 addrspace(1)* %427, align 2
  %429 = zext i16 %422 to i32
  %430 = shl nuw i32 %429, 16, !spirv.Decorations !838
  %431 = bitcast i32 %430 to float
  %432 = zext i16 %428 to i32
  %433 = shl nuw i32 %432, 16, !spirv.Decorations !838
  %434 = bitcast i32 %433 to float
  %435 = fmul reassoc nsz arcp contract float %431, %434, !spirv.Decorations !843
  %436 = fadd reassoc nsz arcp contract float %435, %92, !spirv.Decorations !843
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %416
  %437 = phi float [ %436, %416 ], [ %92, %._crit_edge.1.3 ]
  br i1 %85, label %438, label %.preheader.3

438:                                              ; preds = %._crit_edge.2.3
  %.sroa.64.0.insert.ext273 = zext i32 %107 to i64
  %439 = sext i32 %65 to i64
  %440 = mul nsw i64 %439, %const_reg_qword3, !spirv.Decorations !836
  %441 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %440, i32 0
  %442 = getelementptr i16, i16 addrspace(4)* %441, i64 %.sroa.64.0.insert.ext273
  %443 = addrspacecast i16 addrspace(4)* %442 to i16 addrspace(1)*
  %444 = load i16, i16 addrspace(1)* %443, align 2
  %445 = mul nsw i64 %.sroa.64.0.insert.ext273, %const_reg_qword5, !spirv.Decorations !836
  %446 = sext i32 %80 to i64
  %447 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %445, i32 0
  %448 = getelementptr i16, i16 addrspace(4)* %447, i64 %446
  %449 = addrspacecast i16 addrspace(4)* %448 to i16 addrspace(1)*
  %450 = load i16, i16 addrspace(1)* %449, align 2
  %451 = zext i16 %444 to i32
  %452 = shl nuw i32 %451, 16, !spirv.Decorations !838
  %453 = bitcast i32 %452 to float
  %454 = zext i16 %450 to i32
  %455 = shl nuw i32 %454, 16, !spirv.Decorations !838
  %456 = bitcast i32 %455 to float
  %457 = fmul reassoc nsz arcp contract float %453, %456, !spirv.Decorations !843
  %458 = fadd reassoc nsz arcp contract float %457, %91, !spirv.Decorations !843
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %438
  %459 = phi float [ %458, %438 ], [ %91, %._crit_edge.2.3 ]
  %460 = add nuw nsw i32 %107, 1, !spirv.Decorations !848
  %461 = icmp slt i32 %460, %const_reg_dword2
  br i1 %461, label %.preheader.preheader, label %.preheader1.preheader

462:                                              ; preds = %.preheader1.preheader
  %463 = sext i32 %29 to i64
  %464 = sext i32 %35 to i64
  %465 = mul nsw i64 %463, %const_reg_qword9, !spirv.Decorations !836
  %466 = add nsw i64 %465, %464, !spirv.Decorations !836
  %467 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !843
  br i1 %42, label %468, label %478

468:                                              ; preds = %462
  %469 = mul nsw i64 %463, %const_reg_qword7, !spirv.Decorations !836
  %470 = getelementptr float, float addrspace(4)* %88, i64 %469
  %471 = getelementptr float, float addrspace(4)* %470, i64 %464
  %472 = addrspacecast float addrspace(4)* %471 to float addrspace(1)*
  %473 = load float, float addrspace(1)* %472, align 4
  %474 = fmul reassoc nsz arcp contract float %473, %4, !spirv.Decorations !843
  %475 = fadd reassoc nsz arcp contract float %467, %474, !spirv.Decorations !843
  %476 = getelementptr inbounds float, float addrspace(4)* %87, i64 %466
  %477 = addrspacecast float addrspace(4)* %476 to float addrspace(1)*
  store float %475, float addrspace(1)* %477, align 4
  br label %._crit_edge70

478:                                              ; preds = %462
  %479 = getelementptr inbounds float, float addrspace(4)* %87, i64 %466
  %480 = addrspacecast float addrspace(4)* %479 to float addrspace(1)*
  store float %467, float addrspace(1)* %480, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %.preheader1.preheader, %478, %468
  br i1 %61, label %481, label %._crit_edge70.1

481:                                              ; preds = %._crit_edge70
  %482 = sext i32 %59 to i64
  %483 = sext i32 %35 to i64
  %484 = mul nsw i64 %482, %const_reg_qword9, !spirv.Decorations !836
  %485 = add nsw i64 %484, %483, !spirv.Decorations !836
  %486 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !843
  br i1 %42, label %490, label %487

487:                                              ; preds = %481
  %488 = getelementptr inbounds float, float addrspace(4)* %87, i64 %485
  %489 = addrspacecast float addrspace(4)* %488 to float addrspace(1)*
  store float %486, float addrspace(1)* %489, align 4
  br label %._crit_edge70.1

490:                                              ; preds = %481
  %491 = mul nsw i64 %482, %const_reg_qword7, !spirv.Decorations !836
  %492 = getelementptr float, float addrspace(4)* %88, i64 %491
  %493 = getelementptr float, float addrspace(4)* %492, i64 %483
  %494 = addrspacecast float addrspace(4)* %493 to float addrspace(1)*
  %495 = load float, float addrspace(1)* %494, align 4
  %496 = fmul reassoc nsz arcp contract float %495, %4, !spirv.Decorations !843
  %497 = fadd reassoc nsz arcp contract float %486, %496, !spirv.Decorations !843
  %498 = getelementptr inbounds float, float addrspace(4)* %87, i64 %485
  %499 = addrspacecast float addrspace(4)* %498 to float addrspace(1)*
  store float %497, float addrspace(1)* %499, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %._crit_edge70, %490, %487
  br i1 %64, label %500, label %._crit_edge70.2

500:                                              ; preds = %._crit_edge70.1
  %501 = sext i32 %62 to i64
  %502 = sext i32 %35 to i64
  %503 = mul nsw i64 %501, %const_reg_qword9, !spirv.Decorations !836
  %504 = add nsw i64 %503, %502, !spirv.Decorations !836
  %505 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !843
  br i1 %42, label %509, label %506

506:                                              ; preds = %500
  %507 = getelementptr inbounds float, float addrspace(4)* %87, i64 %504
  %508 = addrspacecast float addrspace(4)* %507 to float addrspace(1)*
  store float %505, float addrspace(1)* %508, align 4
  br label %._crit_edge70.2

509:                                              ; preds = %500
  %510 = mul nsw i64 %501, %const_reg_qword7, !spirv.Decorations !836
  %511 = getelementptr float, float addrspace(4)* %88, i64 %510
  %512 = getelementptr float, float addrspace(4)* %511, i64 %502
  %513 = addrspacecast float addrspace(4)* %512 to float addrspace(1)*
  %514 = load float, float addrspace(1)* %513, align 4
  %515 = fmul reassoc nsz arcp contract float %514, %4, !spirv.Decorations !843
  %516 = fadd reassoc nsz arcp contract float %505, %515, !spirv.Decorations !843
  %517 = getelementptr inbounds float, float addrspace(4)* %87, i64 %504
  %518 = addrspacecast float addrspace(4)* %517 to float addrspace(1)*
  store float %516, float addrspace(1)* %518, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %._crit_edge70.1, %509, %506
  br i1 %67, label %519, label %.preheader1

519:                                              ; preds = %._crit_edge70.2
  %520 = sext i32 %65 to i64
  %521 = sext i32 %35 to i64
  %522 = mul nsw i64 %520, %const_reg_qword9, !spirv.Decorations !836
  %523 = add nsw i64 %522, %521, !spirv.Decorations !836
  %524 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !843
  br i1 %42, label %528, label %525

525:                                              ; preds = %519
  %526 = getelementptr inbounds float, float addrspace(4)* %87, i64 %523
  %527 = addrspacecast float addrspace(4)* %526 to float addrspace(1)*
  store float %524, float addrspace(1)* %527, align 4
  br label %.preheader1

528:                                              ; preds = %519
  %529 = mul nsw i64 %520, %const_reg_qword7, !spirv.Decorations !836
  %530 = getelementptr float, float addrspace(4)* %88, i64 %529
  %531 = getelementptr float, float addrspace(4)* %530, i64 %521
  %532 = addrspacecast float addrspace(4)* %531 to float addrspace(1)*
  %533 = load float, float addrspace(1)* %532, align 4
  %534 = fmul reassoc nsz arcp contract float %533, %4, !spirv.Decorations !843
  %535 = fadd reassoc nsz arcp contract float %524, %534, !spirv.Decorations !843
  %536 = getelementptr inbounds float, float addrspace(4)* %87, i64 %523
  %537 = addrspacecast float addrspace(4)* %536 to float addrspace(1)*
  store float %535, float addrspace(1)* %537, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %._crit_edge70.2, %528, %525
  br i1 %70, label %538, label %._crit_edge70.176

538:                                              ; preds = %.preheader1
  %539 = sext i32 %29 to i64
  %540 = sext i32 %68 to i64
  %541 = mul nsw i64 %539, %const_reg_qword9, !spirv.Decorations !836
  %542 = add nsw i64 %541, %540, !spirv.Decorations !836
  %543 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !843
  br i1 %42, label %547, label %544

544:                                              ; preds = %538
  %545 = getelementptr inbounds float, float addrspace(4)* %87, i64 %542
  %546 = addrspacecast float addrspace(4)* %545 to float addrspace(1)*
  store float %543, float addrspace(1)* %546, align 4
  br label %._crit_edge70.176

547:                                              ; preds = %538
  %548 = mul nsw i64 %539, %const_reg_qword7, !spirv.Decorations !836
  %549 = getelementptr float, float addrspace(4)* %88, i64 %548
  %550 = getelementptr float, float addrspace(4)* %549, i64 %540
  %551 = addrspacecast float addrspace(4)* %550 to float addrspace(1)*
  %552 = load float, float addrspace(1)* %551, align 4
  %553 = fmul reassoc nsz arcp contract float %552, %4, !spirv.Decorations !843
  %554 = fadd reassoc nsz arcp contract float %543, %553, !spirv.Decorations !843
  %555 = getelementptr inbounds float, float addrspace(4)* %87, i64 %542
  %556 = addrspacecast float addrspace(4)* %555 to float addrspace(1)*
  store float %554, float addrspace(1)* %556, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %.preheader1, %547, %544
  br i1 %71, label %557, label %._crit_edge70.1.1

557:                                              ; preds = %._crit_edge70.176
  %558 = sext i32 %59 to i64
  %559 = sext i32 %68 to i64
  %560 = mul nsw i64 %558, %const_reg_qword9, !spirv.Decorations !836
  %561 = add nsw i64 %560, %559, !spirv.Decorations !836
  %562 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !843
  br i1 %42, label %566, label %563

563:                                              ; preds = %557
  %564 = getelementptr inbounds float, float addrspace(4)* %87, i64 %561
  %565 = addrspacecast float addrspace(4)* %564 to float addrspace(1)*
  store float %562, float addrspace(1)* %565, align 4
  br label %._crit_edge70.1.1

566:                                              ; preds = %557
  %567 = mul nsw i64 %558, %const_reg_qword7, !spirv.Decorations !836
  %568 = getelementptr float, float addrspace(4)* %88, i64 %567
  %569 = getelementptr float, float addrspace(4)* %568, i64 %559
  %570 = addrspacecast float addrspace(4)* %569 to float addrspace(1)*
  %571 = load float, float addrspace(1)* %570, align 4
  %572 = fmul reassoc nsz arcp contract float %571, %4, !spirv.Decorations !843
  %573 = fadd reassoc nsz arcp contract float %562, %572, !spirv.Decorations !843
  %574 = getelementptr inbounds float, float addrspace(4)* %87, i64 %561
  %575 = addrspacecast float addrspace(4)* %574 to float addrspace(1)*
  store float %573, float addrspace(1)* %575, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176, %566, %563
  br i1 %72, label %576, label %._crit_edge70.2.1

576:                                              ; preds = %._crit_edge70.1.1
  %577 = sext i32 %62 to i64
  %578 = sext i32 %68 to i64
  %579 = mul nsw i64 %577, %const_reg_qword9, !spirv.Decorations !836
  %580 = add nsw i64 %579, %578, !spirv.Decorations !836
  %581 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !843
  br i1 %42, label %585, label %582

582:                                              ; preds = %576
  %583 = getelementptr inbounds float, float addrspace(4)* %87, i64 %580
  %584 = addrspacecast float addrspace(4)* %583 to float addrspace(1)*
  store float %581, float addrspace(1)* %584, align 4
  br label %._crit_edge70.2.1

585:                                              ; preds = %576
  %586 = mul nsw i64 %577, %const_reg_qword7, !spirv.Decorations !836
  %587 = getelementptr float, float addrspace(4)* %88, i64 %586
  %588 = getelementptr float, float addrspace(4)* %587, i64 %578
  %589 = addrspacecast float addrspace(4)* %588 to float addrspace(1)*
  %590 = load float, float addrspace(1)* %589, align 4
  %591 = fmul reassoc nsz arcp contract float %590, %4, !spirv.Decorations !843
  %592 = fadd reassoc nsz arcp contract float %581, %591, !spirv.Decorations !843
  %593 = getelementptr inbounds float, float addrspace(4)* %87, i64 %580
  %594 = addrspacecast float addrspace(4)* %593 to float addrspace(1)*
  store float %592, float addrspace(1)* %594, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1, %585, %582
  br i1 %73, label %595, label %.preheader1.1

595:                                              ; preds = %._crit_edge70.2.1
  %596 = sext i32 %65 to i64
  %597 = sext i32 %68 to i64
  %598 = mul nsw i64 %596, %const_reg_qword9, !spirv.Decorations !836
  %599 = add nsw i64 %598, %597, !spirv.Decorations !836
  %600 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !843
  br i1 %42, label %604, label %601

601:                                              ; preds = %595
  %602 = getelementptr inbounds float, float addrspace(4)* %87, i64 %599
  %603 = addrspacecast float addrspace(4)* %602 to float addrspace(1)*
  store float %600, float addrspace(1)* %603, align 4
  br label %.preheader1.1

604:                                              ; preds = %595
  %605 = mul nsw i64 %596, %const_reg_qword7, !spirv.Decorations !836
  %606 = getelementptr float, float addrspace(4)* %88, i64 %605
  %607 = getelementptr float, float addrspace(4)* %606, i64 %597
  %608 = addrspacecast float addrspace(4)* %607 to float addrspace(1)*
  %609 = load float, float addrspace(1)* %608, align 4
  %610 = fmul reassoc nsz arcp contract float %609, %4, !spirv.Decorations !843
  %611 = fadd reassoc nsz arcp contract float %600, %610, !spirv.Decorations !843
  %612 = getelementptr inbounds float, float addrspace(4)* %87, i64 %599
  %613 = addrspacecast float addrspace(4)* %612 to float addrspace(1)*
  store float %611, float addrspace(1)* %613, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %._crit_edge70.2.1, %604, %601
  br i1 %76, label %614, label %._crit_edge70.277

614:                                              ; preds = %.preheader1.1
  %615 = sext i32 %29 to i64
  %616 = sext i32 %74 to i64
  %617 = mul nsw i64 %615, %const_reg_qword9, !spirv.Decorations !836
  %618 = add nsw i64 %617, %616, !spirv.Decorations !836
  %619 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !843
  br i1 %42, label %623, label %620

620:                                              ; preds = %614
  %621 = getelementptr inbounds float, float addrspace(4)* %87, i64 %618
  %622 = addrspacecast float addrspace(4)* %621 to float addrspace(1)*
  store float %619, float addrspace(1)* %622, align 4
  br label %._crit_edge70.277

623:                                              ; preds = %614
  %624 = mul nsw i64 %615, %const_reg_qword7, !spirv.Decorations !836
  %625 = getelementptr float, float addrspace(4)* %88, i64 %624
  %626 = getelementptr float, float addrspace(4)* %625, i64 %616
  %627 = addrspacecast float addrspace(4)* %626 to float addrspace(1)*
  %628 = load float, float addrspace(1)* %627, align 4
  %629 = fmul reassoc nsz arcp contract float %628, %4, !spirv.Decorations !843
  %630 = fadd reassoc nsz arcp contract float %619, %629, !spirv.Decorations !843
  %631 = getelementptr inbounds float, float addrspace(4)* %87, i64 %618
  %632 = addrspacecast float addrspace(4)* %631 to float addrspace(1)*
  store float %630, float addrspace(1)* %632, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %.preheader1.1, %623, %620
  br i1 %77, label %633, label %._crit_edge70.1.2

633:                                              ; preds = %._crit_edge70.277
  %634 = sext i32 %59 to i64
  %635 = sext i32 %74 to i64
  %636 = mul nsw i64 %634, %const_reg_qword9, !spirv.Decorations !836
  %637 = add nsw i64 %636, %635, !spirv.Decorations !836
  %638 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !843
  br i1 %42, label %642, label %639

639:                                              ; preds = %633
  %640 = getelementptr inbounds float, float addrspace(4)* %87, i64 %637
  %641 = addrspacecast float addrspace(4)* %640 to float addrspace(1)*
  store float %638, float addrspace(1)* %641, align 4
  br label %._crit_edge70.1.2

642:                                              ; preds = %633
  %643 = mul nsw i64 %634, %const_reg_qword7, !spirv.Decorations !836
  %644 = getelementptr float, float addrspace(4)* %88, i64 %643
  %645 = getelementptr float, float addrspace(4)* %644, i64 %635
  %646 = addrspacecast float addrspace(4)* %645 to float addrspace(1)*
  %647 = load float, float addrspace(1)* %646, align 4
  %648 = fmul reassoc nsz arcp contract float %647, %4, !spirv.Decorations !843
  %649 = fadd reassoc nsz arcp contract float %638, %648, !spirv.Decorations !843
  %650 = getelementptr inbounds float, float addrspace(4)* %87, i64 %637
  %651 = addrspacecast float addrspace(4)* %650 to float addrspace(1)*
  store float %649, float addrspace(1)* %651, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277, %642, %639
  br i1 %78, label %652, label %._crit_edge70.2.2

652:                                              ; preds = %._crit_edge70.1.2
  %653 = sext i32 %62 to i64
  %654 = sext i32 %74 to i64
  %655 = mul nsw i64 %653, %const_reg_qword9, !spirv.Decorations !836
  %656 = add nsw i64 %655, %654, !spirv.Decorations !836
  %657 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !843
  br i1 %42, label %661, label %658

658:                                              ; preds = %652
  %659 = getelementptr inbounds float, float addrspace(4)* %87, i64 %656
  %660 = addrspacecast float addrspace(4)* %659 to float addrspace(1)*
  store float %657, float addrspace(1)* %660, align 4
  br label %._crit_edge70.2.2

661:                                              ; preds = %652
  %662 = mul nsw i64 %653, %const_reg_qword7, !spirv.Decorations !836
  %663 = getelementptr float, float addrspace(4)* %88, i64 %662
  %664 = getelementptr float, float addrspace(4)* %663, i64 %654
  %665 = addrspacecast float addrspace(4)* %664 to float addrspace(1)*
  %666 = load float, float addrspace(1)* %665, align 4
  %667 = fmul reassoc nsz arcp contract float %666, %4, !spirv.Decorations !843
  %668 = fadd reassoc nsz arcp contract float %657, %667, !spirv.Decorations !843
  %669 = getelementptr inbounds float, float addrspace(4)* %87, i64 %656
  %670 = addrspacecast float addrspace(4)* %669 to float addrspace(1)*
  store float %668, float addrspace(1)* %670, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2, %661, %658
  br i1 %79, label %671, label %.preheader1.2

671:                                              ; preds = %._crit_edge70.2.2
  %672 = sext i32 %65 to i64
  %673 = sext i32 %74 to i64
  %674 = mul nsw i64 %672, %const_reg_qword9, !spirv.Decorations !836
  %675 = add nsw i64 %674, %673, !spirv.Decorations !836
  %676 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !843
  br i1 %42, label %680, label %677

677:                                              ; preds = %671
  %678 = getelementptr inbounds float, float addrspace(4)* %87, i64 %675
  %679 = addrspacecast float addrspace(4)* %678 to float addrspace(1)*
  store float %676, float addrspace(1)* %679, align 4
  br label %.preheader1.2

680:                                              ; preds = %671
  %681 = mul nsw i64 %672, %const_reg_qword7, !spirv.Decorations !836
  %682 = getelementptr float, float addrspace(4)* %88, i64 %681
  %683 = getelementptr float, float addrspace(4)* %682, i64 %673
  %684 = addrspacecast float addrspace(4)* %683 to float addrspace(1)*
  %685 = load float, float addrspace(1)* %684, align 4
  %686 = fmul reassoc nsz arcp contract float %685, %4, !spirv.Decorations !843
  %687 = fadd reassoc nsz arcp contract float %676, %686, !spirv.Decorations !843
  %688 = getelementptr inbounds float, float addrspace(4)* %87, i64 %675
  %689 = addrspacecast float addrspace(4)* %688 to float addrspace(1)*
  store float %687, float addrspace(1)* %689, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %._crit_edge70.2.2, %680, %677
  br i1 %82, label %690, label %._crit_edge70.378

690:                                              ; preds = %.preheader1.2
  %691 = sext i32 %29 to i64
  %692 = sext i32 %80 to i64
  %693 = mul nsw i64 %691, %const_reg_qword9, !spirv.Decorations !836
  %694 = add nsw i64 %693, %692, !spirv.Decorations !836
  %695 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !843
  br i1 %42, label %699, label %696

696:                                              ; preds = %690
  %697 = getelementptr inbounds float, float addrspace(4)* %87, i64 %694
  %698 = addrspacecast float addrspace(4)* %697 to float addrspace(1)*
  store float %695, float addrspace(1)* %698, align 4
  br label %._crit_edge70.378

699:                                              ; preds = %690
  %700 = mul nsw i64 %691, %const_reg_qword7, !spirv.Decorations !836
  %701 = getelementptr float, float addrspace(4)* %88, i64 %700
  %702 = getelementptr float, float addrspace(4)* %701, i64 %692
  %703 = addrspacecast float addrspace(4)* %702 to float addrspace(1)*
  %704 = load float, float addrspace(1)* %703, align 4
  %705 = fmul reassoc nsz arcp contract float %704, %4, !spirv.Decorations !843
  %706 = fadd reassoc nsz arcp contract float %695, %705, !spirv.Decorations !843
  %707 = getelementptr inbounds float, float addrspace(4)* %87, i64 %694
  %708 = addrspacecast float addrspace(4)* %707 to float addrspace(1)*
  store float %706, float addrspace(1)* %708, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %.preheader1.2, %699, %696
  br i1 %83, label %709, label %._crit_edge70.1.3

709:                                              ; preds = %._crit_edge70.378
  %710 = sext i32 %59 to i64
  %711 = sext i32 %80 to i64
  %712 = mul nsw i64 %710, %const_reg_qword9, !spirv.Decorations !836
  %713 = add nsw i64 %712, %711, !spirv.Decorations !836
  %714 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !843
  br i1 %42, label %718, label %715

715:                                              ; preds = %709
  %716 = getelementptr inbounds float, float addrspace(4)* %87, i64 %713
  %717 = addrspacecast float addrspace(4)* %716 to float addrspace(1)*
  store float %714, float addrspace(1)* %717, align 4
  br label %._crit_edge70.1.3

718:                                              ; preds = %709
  %719 = mul nsw i64 %710, %const_reg_qword7, !spirv.Decorations !836
  %720 = getelementptr float, float addrspace(4)* %88, i64 %719
  %721 = getelementptr float, float addrspace(4)* %720, i64 %711
  %722 = addrspacecast float addrspace(4)* %721 to float addrspace(1)*
  %723 = load float, float addrspace(1)* %722, align 4
  %724 = fmul reassoc nsz arcp contract float %723, %4, !spirv.Decorations !843
  %725 = fadd reassoc nsz arcp contract float %714, %724, !spirv.Decorations !843
  %726 = getelementptr inbounds float, float addrspace(4)* %87, i64 %713
  %727 = addrspacecast float addrspace(4)* %726 to float addrspace(1)*
  store float %725, float addrspace(1)* %727, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378, %718, %715
  br i1 %84, label %728, label %._crit_edge70.2.3

728:                                              ; preds = %._crit_edge70.1.3
  %729 = sext i32 %62 to i64
  %730 = sext i32 %80 to i64
  %731 = mul nsw i64 %729, %const_reg_qword9, !spirv.Decorations !836
  %732 = add nsw i64 %731, %730, !spirv.Decorations !836
  %733 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !843
  br i1 %42, label %737, label %734

734:                                              ; preds = %728
  %735 = getelementptr inbounds float, float addrspace(4)* %87, i64 %732
  %736 = addrspacecast float addrspace(4)* %735 to float addrspace(1)*
  store float %733, float addrspace(1)* %736, align 4
  br label %._crit_edge70.2.3

737:                                              ; preds = %728
  %738 = mul nsw i64 %729, %const_reg_qword7, !spirv.Decorations !836
  %739 = getelementptr float, float addrspace(4)* %88, i64 %738
  %740 = getelementptr float, float addrspace(4)* %739, i64 %730
  %741 = addrspacecast float addrspace(4)* %740 to float addrspace(1)*
  %742 = load float, float addrspace(1)* %741, align 4
  %743 = fmul reassoc nsz arcp contract float %742, %4, !spirv.Decorations !843
  %744 = fadd reassoc nsz arcp contract float %733, %743, !spirv.Decorations !843
  %745 = getelementptr inbounds float, float addrspace(4)* %87, i64 %732
  %746 = addrspacecast float addrspace(4)* %745 to float addrspace(1)*
  store float %744, float addrspace(1)* %746, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3, %737, %734
  br i1 %85, label %747, label %.preheader1.3

747:                                              ; preds = %._crit_edge70.2.3
  %748 = sext i32 %65 to i64
  %749 = sext i32 %80 to i64
  %750 = mul nsw i64 %748, %const_reg_qword9, !spirv.Decorations !836
  %751 = add nsw i64 %750, %749, !spirv.Decorations !836
  %752 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !843
  br i1 %42, label %756, label %753

753:                                              ; preds = %747
  %754 = getelementptr inbounds float, float addrspace(4)* %87, i64 %751
  %755 = addrspacecast float addrspace(4)* %754 to float addrspace(1)*
  store float %752, float addrspace(1)* %755, align 4
  br label %.preheader1.3

756:                                              ; preds = %747
  %757 = mul nsw i64 %748, %const_reg_qword7, !spirv.Decorations !836
  %758 = getelementptr float, float addrspace(4)* %88, i64 %757
  %759 = getelementptr float, float addrspace(4)* %758, i64 %749
  %760 = addrspacecast float addrspace(4)* %759 to float addrspace(1)*
  %761 = load float, float addrspace(1)* %760, align 4
  %762 = fmul reassoc nsz arcp contract float %761, %4, !spirv.Decorations !843
  %763 = fadd reassoc nsz arcp contract float %752, %762, !spirv.Decorations !843
  %764 = getelementptr inbounds float, float addrspace(4)* %87, i64 %751
  %765 = addrspacecast float addrspace(4)* %764 to float addrspace(1)*
  store float %763, float addrspace(1)* %765, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %._crit_edge70.2.3, %756, %753
  %766 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %90, i64 %52
  %767 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %89, i64 %53
  %.idx = select i1 %42, i64 %54, i64 0
  %768 = getelementptr float, float addrspace(4)* %88, i64 %.idx
  %769 = getelementptr inbounds float, float addrspace(4)* %87, i64 %55
  %770 = add i32 %86, %14
  %771 = icmp slt i32 %770, %8
  br i1 %771, label %.preheader2.preheader, label %._crit_edge72

._crit_edge72:                                    ; preds = %.preheader1.3, %13
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %21 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %22 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %23 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %24 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %24)
  %25 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %25)
  %26 = mul i32 %17, %15
  %27 = zext i16 %localIdX to i32
  %28 = add i32 %26, %27
  %29 = shl i32 %28, 2
  %30 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %18, %16
  %33 = zext i16 %localIdY to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 4
  %36 = zext i32 %19 to i64
  %37 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %37)
  %38 = mul nsw i64 %36, %9, !spirv.Decorations !836
  %39 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %23, i64 %38
  %40 = mul nsw i64 %36, %10, !spirv.Decorations !836
  %41 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %22, i64 %40
  %42 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !843
  %43 = mul nsw i64 %36, %11, !spirv.Decorations !836
  %44 = select i1 %42, i64 %43, i64 0
  %45 = getelementptr inbounds float, float addrspace(4)* %21, i64 %44
  %46 = mul nsw i64 %36, %12, !spirv.Decorations !836
  %47 = getelementptr inbounds float, float addrspace(4)* %20, i64 %46
  %48 = icmp slt i32 %19, %8
  br i1 %48, label %.lr.ph, label %._crit_edge72

.lr.ph:                                           ; preds = %13
  %49 = icmp sgt i32 %const_reg_dword2, 0
  %50 = zext i32 %14 to i64
  %51 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %51)
  %52 = mul nsw i64 %50, %9, !spirv.Decorations !836
  %53 = mul nsw i64 %50, %10, !spirv.Decorations !836
  %54 = mul nsw i64 %50, %11
  %55 = mul nsw i64 %50, %12, !spirv.Decorations !836
  %56 = icmp slt i32 %35, %const_reg_dword1
  %57 = icmp slt i32 %29, %const_reg_dword
  %58 = and i1 %57, %56
  %59 = or i32 %29, 1
  %60 = icmp slt i32 %59, %const_reg_dword
  %61 = and i1 %60, %56
  %62 = or i32 %29, 2
  %63 = icmp slt i32 %62, %const_reg_dword
  %64 = and i1 %63, %56
  %65 = or i32 %29, 3
  %66 = icmp slt i32 %65, %const_reg_dword
  %67 = and i1 %66, %56
  %68 = or i32 %35, 1
  %69 = icmp slt i32 %68, %const_reg_dword1
  %70 = and i1 %57, %69
  %71 = and i1 %60, %69
  %72 = and i1 %63, %69
  %73 = and i1 %66, %69
  %74 = or i32 %35, 2
  %75 = icmp slt i32 %74, %const_reg_dword1
  %76 = and i1 %57, %75
  %77 = and i1 %60, %75
  %78 = and i1 %63, %75
  %79 = and i1 %66, %75
  %80 = or i32 %35, 3
  %81 = icmp slt i32 %80, %const_reg_dword1
  %82 = and i1 %57, %81
  %83 = and i1 %60, %81
  %84 = and i1 %63, %81
  %85 = and i1 %66, %81
  %86 = or i32 %35, 4
  %87 = icmp slt i32 %86, %const_reg_dword1
  %88 = and i1 %57, %87
  %89 = and i1 %60, %87
  %90 = and i1 %63, %87
  %91 = and i1 %66, %87
  %92 = or i32 %35, 5
  %93 = icmp slt i32 %92, %const_reg_dword1
  %94 = and i1 %57, %93
  %95 = and i1 %60, %93
  %96 = and i1 %63, %93
  %97 = and i1 %66, %93
  %98 = or i32 %35, 6
  %99 = icmp slt i32 %98, %const_reg_dword1
  %100 = and i1 %57, %99
  %101 = and i1 %60, %99
  %102 = and i1 %63, %99
  %103 = and i1 %66, %99
  %104 = or i32 %35, 7
  %105 = icmp slt i32 %104, %const_reg_dword1
  %106 = and i1 %57, %105
  %107 = and i1 %60, %105
  %108 = and i1 %63, %105
  %109 = and i1 %66, %105
  %110 = or i32 %35, 8
  %111 = icmp slt i32 %110, %const_reg_dword1
  %112 = and i1 %57, %111
  %113 = and i1 %60, %111
  %114 = and i1 %63, %111
  %115 = and i1 %66, %111
  %116 = or i32 %35, 9
  %117 = icmp slt i32 %116, %const_reg_dword1
  %118 = and i1 %57, %117
  %119 = and i1 %60, %117
  %120 = and i1 %63, %117
  %121 = and i1 %66, %117
  %122 = or i32 %35, 10
  %123 = icmp slt i32 %122, %const_reg_dword1
  %124 = and i1 %57, %123
  %125 = and i1 %60, %123
  %126 = and i1 %63, %123
  %127 = and i1 %66, %123
  %128 = or i32 %35, 11
  %129 = icmp slt i32 %128, %const_reg_dword1
  %130 = and i1 %57, %129
  %131 = and i1 %60, %129
  %132 = and i1 %63, %129
  %133 = and i1 %66, %129
  %134 = or i32 %35, 12
  %135 = icmp slt i32 %134, %const_reg_dword1
  %136 = and i1 %57, %135
  %137 = and i1 %60, %135
  %138 = and i1 %63, %135
  %139 = and i1 %66, %135
  %140 = or i32 %35, 13
  %141 = icmp slt i32 %140, %const_reg_dword1
  %142 = and i1 %57, %141
  %143 = and i1 %60, %141
  %144 = and i1 %63, %141
  %145 = and i1 %66, %141
  %146 = or i32 %35, 14
  %147 = icmp slt i32 %146, %const_reg_dword1
  %148 = and i1 %57, %147
  %149 = and i1 %60, %147
  %150 = and i1 %63, %147
  %151 = and i1 %66, %147
  %152 = or i32 %35, 15
  %153 = icmp slt i32 %152, %const_reg_dword1
  %154 = and i1 %57, %153
  %155 = and i1 %60, %153
  %156 = and i1 %63, %153
  %157 = and i1 %66, %153
  br label %.preheader2.preheader

.preheader2.preheader:                            ; preds = %.lr.ph, %.preheader1.15
  %158 = phi i32 [ %19, %.lr.ph ], [ %2730, %.preheader1.15 ]
  %159 = phi float addrspace(4)* [ %47, %.lr.ph ], [ %2729, %.preheader1.15 ]
  %160 = phi float addrspace(4)* [ %45, %.lr.ph ], [ %2728, %.preheader1.15 ]
  %161 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %41, %.lr.ph ], [ %2727, %.preheader1.15 ]
  %162 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %39, %.lr.ph ], [ %2726, %.preheader1.15 ]
  br i1 %49, label %.preheader.preheader, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.preheader.15, %.preheader2.preheader
  %.sroa.254.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.254.2, %.preheader.15 ]
  %.sroa.250.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.250.2, %.preheader.15 ]
  %.sroa.246.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.246.2, %.preheader.15 ]
  %.sroa.242.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.242.2, %.preheader.15 ]
  %.sroa.238.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.238.2, %.preheader.15 ]
  %.sroa.234.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.234.2, %.preheader.15 ]
  %.sroa.230.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.230.2, %.preheader.15 ]
  %.sroa.226.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.226.2, %.preheader.15 ]
  %.sroa.222.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.222.2, %.preheader.15 ]
  %.sroa.218.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.218.2, %.preheader.15 ]
  %.sroa.214.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.214.2, %.preheader.15 ]
  %.sroa.210.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.210.2, %.preheader.15 ]
  %.sroa.206.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.206.2, %.preheader.15 ]
  %.sroa.202.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.202.2, %.preheader.15 ]
  %.sroa.198.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.198.2, %.preheader.15 ]
  %.sroa.194.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.194.2, %.preheader.15 ]
  %.sroa.190.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.190.2, %.preheader.15 ]
  %.sroa.186.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.186.2, %.preheader.15 ]
  %.sroa.182.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.182.2, %.preheader.15 ]
  %.sroa.178.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.178.2, %.preheader.15 ]
  %.sroa.174.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.174.2, %.preheader.15 ]
  %.sroa.170.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.170.2, %.preheader.15 ]
  %.sroa.166.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.166.2, %.preheader.15 ]
  %.sroa.162.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.162.2, %.preheader.15 ]
  %.sroa.158.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.158.2, %.preheader.15 ]
  %.sroa.154.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.154.2, %.preheader.15 ]
  %.sroa.150.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.150.2, %.preheader.15 ]
  %.sroa.146.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.146.2, %.preheader.15 ]
  %.sroa.142.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.142.2, %.preheader.15 ]
  %.sroa.138.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.138.2, %.preheader.15 ]
  %.sroa.134.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.134.2, %.preheader.15 ]
  %.sroa.130.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.130.2, %.preheader.15 ]
  %.sroa.126.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.126.2, %.preheader.15 ]
  %.sroa.122.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.122.2, %.preheader.15 ]
  %.sroa.118.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.118.2, %.preheader.15 ]
  %.sroa.114.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.114.2, %.preheader.15 ]
  %.sroa.110.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.110.2, %.preheader.15 ]
  %.sroa.106.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.106.2, %.preheader.15 ]
  %.sroa.102.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.102.2, %.preheader.15 ]
  %.sroa.98.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.98.2, %.preheader.15 ]
  %.sroa.94.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.94.2, %.preheader.15 ]
  %.sroa.90.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.90.2, %.preheader.15 ]
  %.sroa.86.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.86.2, %.preheader.15 ]
  %.sroa.82.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.82.2, %.preheader.15 ]
  %.sroa.78.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.78.2, %.preheader.15 ]
  %.sroa.74.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.74.2, %.preheader.15 ]
  %.sroa.70.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.70.2, %.preheader.15 ]
  %.sroa.66.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.66.2, %.preheader.15 ]
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.62.2, %.preheader.15 ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.58.2, %.preheader.15 ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.54.2, %.preheader.15 ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.50.2, %.preheader.15 ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.46.2, %.preheader.15 ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.42.2, %.preheader.15 ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.38.2, %.preheader.15 ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.34.2, %.preheader.15 ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.30.2, %.preheader.15 ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.26.2, %.preheader.15 ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.22.2, %.preheader.15 ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.18.2, %.preheader.15 ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.14.2, %.preheader.15 ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.10.2, %.preheader.15 ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.6.2, %.preheader.15 ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader ], [ %.sroa.0.2, %.preheader.15 ]
  br i1 %58, label %1510, label %._crit_edge70

.preheader.preheader:                             ; preds = %.preheader2.preheader, %.preheader.15
  %.sroa.254.1 = phi float [ %.sroa.254.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.250.1 = phi float [ %.sroa.250.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.246.1 = phi float [ %.sroa.246.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.242.1 = phi float [ %.sroa.242.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.238.1 = phi float [ %.sroa.238.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.234.1 = phi float [ %.sroa.234.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.230.1 = phi float [ %.sroa.230.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.226.1 = phi float [ %.sroa.226.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.222.1 = phi float [ %.sroa.222.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.218.1 = phi float [ %.sroa.218.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.214.1 = phi float [ %.sroa.214.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.210.1 = phi float [ %.sroa.210.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.206.1 = phi float [ %.sroa.206.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.202.1 = phi float [ %.sroa.202.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.198.1 = phi float [ %.sroa.198.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.194.1 = phi float [ %.sroa.194.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.190.1 = phi float [ %.sroa.190.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.186.1 = phi float [ %.sroa.186.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.182.1 = phi float [ %.sroa.182.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.178.1 = phi float [ %.sroa.178.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.174.1 = phi float [ %.sroa.174.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.170.1 = phi float [ %.sroa.170.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.166.1 = phi float [ %.sroa.166.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.162.1 = phi float [ %.sroa.162.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.158.1 = phi float [ %.sroa.158.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.154.1 = phi float [ %.sroa.154.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.150.1 = phi float [ %.sroa.150.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.146.1 = phi float [ %.sroa.146.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.142.1 = phi float [ %.sroa.142.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.138.1 = phi float [ %.sroa.138.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.134.1 = phi float [ %.sroa.134.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.130.1 = phi float [ %.sroa.130.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.126.1 = phi float [ %.sroa.126.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.122.1 = phi float [ %.sroa.122.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.118.1 = phi float [ %.sroa.118.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.114.1 = phi float [ %.sroa.114.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.110.1 = phi float [ %.sroa.110.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.106.1 = phi float [ %.sroa.106.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.102.1 = phi float [ %.sroa.102.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.98.1 = phi float [ %.sroa.98.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.94.1 = phi float [ %.sroa.94.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.90.1 = phi float [ %.sroa.90.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.86.1 = phi float [ %.sroa.86.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.82.1 = phi float [ %.sroa.82.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.78.1 = phi float [ %.sroa.78.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.74.1 = phi float [ %.sroa.74.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.70.1 = phi float [ %.sroa.70.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.66.1 = phi float [ %.sroa.66.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.62.1 = phi float [ %.sroa.62.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.58.1 = phi float [ %.sroa.58.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.54.1 = phi float [ %.sroa.54.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.50.1 = phi float [ %.sroa.50.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.46.1 = phi float [ %.sroa.46.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.42.1 = phi float [ %.sroa.42.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.38.1 = phi float [ %.sroa.38.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.34.1 = phi float [ %.sroa.34.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.30.1 = phi float [ %.sroa.30.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.26.1 = phi float [ %.sroa.26.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.22.1 = phi float [ %.sroa.22.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.18.1 = phi float [ %.sroa.18.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.14.1 = phi float [ %.sroa.14.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.10.1 = phi float [ %.sroa.10.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.6.1 = phi float [ %.sroa.6.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %.sroa.0.1 = phi float [ %.sroa.0.2, %.preheader.15 ], [ %7, %.preheader2.preheader ]
  %163 = phi i32 [ %1508, %.preheader.15 ], [ 0, %.preheader2.preheader ]
  br i1 %58, label %164, label %._crit_edge

164:                                              ; preds = %.preheader.preheader
  %.sroa.256.0.insert.ext = zext i32 %163 to i64
  %165 = sext i32 %29 to i64
  %166 = mul nsw i64 %165, %const_reg_qword3, !spirv.Decorations !836
  %167 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %166, i32 0
  %168 = getelementptr i16, i16 addrspace(4)* %167, i64 %.sroa.256.0.insert.ext
  %169 = addrspacecast i16 addrspace(4)* %168 to i16 addrspace(1)*
  %170 = load i16, i16 addrspace(1)* %169, align 2
  %171 = mul nsw i64 %.sroa.256.0.insert.ext, %const_reg_qword5, !spirv.Decorations !836
  %172 = sext i32 %35 to i64
  %173 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %171, i32 0
  %174 = getelementptr i16, i16 addrspace(4)* %173, i64 %172
  %175 = addrspacecast i16 addrspace(4)* %174 to i16 addrspace(1)*
  %176 = load i16, i16 addrspace(1)* %175, align 2
  %177 = zext i16 %170 to i32
  %178 = shl nuw i32 %177, 16, !spirv.Decorations !838
  %179 = bitcast i32 %178 to float
  %180 = zext i16 %176 to i32
  %181 = shl nuw i32 %180, 16, !spirv.Decorations !838
  %182 = bitcast i32 %181 to float
  %183 = fmul reassoc nsz arcp contract float %179, %182, !spirv.Decorations !843
  %184 = fadd reassoc nsz arcp contract float %183, %.sroa.0.1, !spirv.Decorations !843
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %164
  %.sroa.0.2 = phi float [ %184, %164 ], [ %.sroa.0.1, %.preheader.preheader ]
  br i1 %61, label %185, label %._crit_edge.1

185:                                              ; preds = %._crit_edge
  %.sroa.256.0.insert.ext588 = zext i32 %163 to i64
  %186 = sext i32 %59 to i64
  %187 = mul nsw i64 %186, %const_reg_qword3, !spirv.Decorations !836
  %188 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %187, i32 0
  %189 = getelementptr i16, i16 addrspace(4)* %188, i64 %.sroa.256.0.insert.ext588
  %190 = addrspacecast i16 addrspace(4)* %189 to i16 addrspace(1)*
  %191 = load i16, i16 addrspace(1)* %190, align 2
  %192 = mul nsw i64 %.sroa.256.0.insert.ext588, %const_reg_qword5, !spirv.Decorations !836
  %193 = sext i32 %35 to i64
  %194 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %192, i32 0
  %195 = getelementptr i16, i16 addrspace(4)* %194, i64 %193
  %196 = addrspacecast i16 addrspace(4)* %195 to i16 addrspace(1)*
  %197 = load i16, i16 addrspace(1)* %196, align 2
  %198 = zext i16 %191 to i32
  %199 = shl nuw i32 %198, 16, !spirv.Decorations !838
  %200 = bitcast i32 %199 to float
  %201 = zext i16 %197 to i32
  %202 = shl nuw i32 %201, 16, !spirv.Decorations !838
  %203 = bitcast i32 %202 to float
  %204 = fmul reassoc nsz arcp contract float %200, %203, !spirv.Decorations !843
  %205 = fadd reassoc nsz arcp contract float %204, %.sroa.66.1, !spirv.Decorations !843
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %185
  %.sroa.66.2 = phi float [ %205, %185 ], [ %.sroa.66.1, %._crit_edge ]
  br i1 %64, label %206, label %._crit_edge.2

206:                                              ; preds = %._crit_edge.1
  %.sroa.256.0.insert.ext593 = zext i32 %163 to i64
  %207 = sext i32 %62 to i64
  %208 = mul nsw i64 %207, %const_reg_qword3, !spirv.Decorations !836
  %209 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %208, i32 0
  %210 = getelementptr i16, i16 addrspace(4)* %209, i64 %.sroa.256.0.insert.ext593
  %211 = addrspacecast i16 addrspace(4)* %210 to i16 addrspace(1)*
  %212 = load i16, i16 addrspace(1)* %211, align 2
  %213 = mul nsw i64 %.sroa.256.0.insert.ext593, %const_reg_qword5, !spirv.Decorations !836
  %214 = sext i32 %35 to i64
  %215 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %213, i32 0
  %216 = getelementptr i16, i16 addrspace(4)* %215, i64 %214
  %217 = addrspacecast i16 addrspace(4)* %216 to i16 addrspace(1)*
  %218 = load i16, i16 addrspace(1)* %217, align 2
  %219 = zext i16 %212 to i32
  %220 = shl nuw i32 %219, 16, !spirv.Decorations !838
  %221 = bitcast i32 %220 to float
  %222 = zext i16 %218 to i32
  %223 = shl nuw i32 %222, 16, !spirv.Decorations !838
  %224 = bitcast i32 %223 to float
  %225 = fmul reassoc nsz arcp contract float %221, %224, !spirv.Decorations !843
  %226 = fadd reassoc nsz arcp contract float %225, %.sroa.130.1, !spirv.Decorations !843
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %206
  %.sroa.130.2 = phi float [ %226, %206 ], [ %.sroa.130.1, %._crit_edge.1 ]
  br i1 %67, label %227, label %.preheader

227:                                              ; preds = %._crit_edge.2
  %.sroa.256.0.insert.ext598 = zext i32 %163 to i64
  %228 = sext i32 %65 to i64
  %229 = mul nsw i64 %228, %const_reg_qword3, !spirv.Decorations !836
  %230 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %229, i32 0
  %231 = getelementptr i16, i16 addrspace(4)* %230, i64 %.sroa.256.0.insert.ext598
  %232 = addrspacecast i16 addrspace(4)* %231 to i16 addrspace(1)*
  %233 = load i16, i16 addrspace(1)* %232, align 2
  %234 = mul nsw i64 %.sroa.256.0.insert.ext598, %const_reg_qword5, !spirv.Decorations !836
  %235 = sext i32 %35 to i64
  %236 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %234, i32 0
  %237 = getelementptr i16, i16 addrspace(4)* %236, i64 %235
  %238 = addrspacecast i16 addrspace(4)* %237 to i16 addrspace(1)*
  %239 = load i16, i16 addrspace(1)* %238, align 2
  %240 = zext i16 %233 to i32
  %241 = shl nuw i32 %240, 16, !spirv.Decorations !838
  %242 = bitcast i32 %241 to float
  %243 = zext i16 %239 to i32
  %244 = shl nuw i32 %243, 16, !spirv.Decorations !838
  %245 = bitcast i32 %244 to float
  %246 = fmul reassoc nsz arcp contract float %242, %245, !spirv.Decorations !843
  %247 = fadd reassoc nsz arcp contract float %246, %.sroa.194.1, !spirv.Decorations !843
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %227
  %.sroa.194.2 = phi float [ %247, %227 ], [ %.sroa.194.1, %._crit_edge.2 ]
  br i1 %70, label %248, label %._crit_edge.173

248:                                              ; preds = %.preheader
  %.sroa.256.0.insert.ext603 = zext i32 %163 to i64
  %249 = sext i32 %29 to i64
  %250 = mul nsw i64 %249, %const_reg_qword3, !spirv.Decorations !836
  %251 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %250, i32 0
  %252 = getelementptr i16, i16 addrspace(4)* %251, i64 %.sroa.256.0.insert.ext603
  %253 = addrspacecast i16 addrspace(4)* %252 to i16 addrspace(1)*
  %254 = load i16, i16 addrspace(1)* %253, align 2
  %255 = mul nsw i64 %.sroa.256.0.insert.ext603, %const_reg_qword5, !spirv.Decorations !836
  %256 = sext i32 %68 to i64
  %257 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %255, i32 0
  %258 = getelementptr i16, i16 addrspace(4)* %257, i64 %256
  %259 = addrspacecast i16 addrspace(4)* %258 to i16 addrspace(1)*
  %260 = load i16, i16 addrspace(1)* %259, align 2
  %261 = zext i16 %254 to i32
  %262 = shl nuw i32 %261, 16, !spirv.Decorations !838
  %263 = bitcast i32 %262 to float
  %264 = zext i16 %260 to i32
  %265 = shl nuw i32 %264, 16, !spirv.Decorations !838
  %266 = bitcast i32 %265 to float
  %267 = fmul reassoc nsz arcp contract float %263, %266, !spirv.Decorations !843
  %268 = fadd reassoc nsz arcp contract float %267, %.sroa.6.1, !spirv.Decorations !843
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %248
  %.sroa.6.2 = phi float [ %268, %248 ], [ %.sroa.6.1, %.preheader ]
  br i1 %71, label %269, label %._crit_edge.1.1

269:                                              ; preds = %._crit_edge.173
  %.sroa.256.0.insert.ext608 = zext i32 %163 to i64
  %270 = sext i32 %59 to i64
  %271 = mul nsw i64 %270, %const_reg_qword3, !spirv.Decorations !836
  %272 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %271, i32 0
  %273 = getelementptr i16, i16 addrspace(4)* %272, i64 %.sroa.256.0.insert.ext608
  %274 = addrspacecast i16 addrspace(4)* %273 to i16 addrspace(1)*
  %275 = load i16, i16 addrspace(1)* %274, align 2
  %276 = mul nsw i64 %.sroa.256.0.insert.ext608, %const_reg_qword5, !spirv.Decorations !836
  %277 = sext i32 %68 to i64
  %278 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %276, i32 0
  %279 = getelementptr i16, i16 addrspace(4)* %278, i64 %277
  %280 = addrspacecast i16 addrspace(4)* %279 to i16 addrspace(1)*
  %281 = load i16, i16 addrspace(1)* %280, align 2
  %282 = zext i16 %275 to i32
  %283 = shl nuw i32 %282, 16, !spirv.Decorations !838
  %284 = bitcast i32 %283 to float
  %285 = zext i16 %281 to i32
  %286 = shl nuw i32 %285, 16, !spirv.Decorations !838
  %287 = bitcast i32 %286 to float
  %288 = fmul reassoc nsz arcp contract float %284, %287, !spirv.Decorations !843
  %289 = fadd reassoc nsz arcp contract float %288, %.sroa.70.1, !spirv.Decorations !843
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %269
  %.sroa.70.2 = phi float [ %289, %269 ], [ %.sroa.70.1, %._crit_edge.173 ]
  br i1 %72, label %290, label %._crit_edge.2.1

290:                                              ; preds = %._crit_edge.1.1
  %.sroa.256.0.insert.ext613 = zext i32 %163 to i64
  %291 = sext i32 %62 to i64
  %292 = mul nsw i64 %291, %const_reg_qword3, !spirv.Decorations !836
  %293 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %292, i32 0
  %294 = getelementptr i16, i16 addrspace(4)* %293, i64 %.sroa.256.0.insert.ext613
  %295 = addrspacecast i16 addrspace(4)* %294 to i16 addrspace(1)*
  %296 = load i16, i16 addrspace(1)* %295, align 2
  %297 = mul nsw i64 %.sroa.256.0.insert.ext613, %const_reg_qword5, !spirv.Decorations !836
  %298 = sext i32 %68 to i64
  %299 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %297, i32 0
  %300 = getelementptr i16, i16 addrspace(4)* %299, i64 %298
  %301 = addrspacecast i16 addrspace(4)* %300 to i16 addrspace(1)*
  %302 = load i16, i16 addrspace(1)* %301, align 2
  %303 = zext i16 %296 to i32
  %304 = shl nuw i32 %303, 16, !spirv.Decorations !838
  %305 = bitcast i32 %304 to float
  %306 = zext i16 %302 to i32
  %307 = shl nuw i32 %306, 16, !spirv.Decorations !838
  %308 = bitcast i32 %307 to float
  %309 = fmul reassoc nsz arcp contract float %305, %308, !spirv.Decorations !843
  %310 = fadd reassoc nsz arcp contract float %309, %.sroa.134.1, !spirv.Decorations !843
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %290
  %.sroa.134.2 = phi float [ %310, %290 ], [ %.sroa.134.1, %._crit_edge.1.1 ]
  br i1 %73, label %311, label %.preheader.1

311:                                              ; preds = %._crit_edge.2.1
  %.sroa.256.0.insert.ext618 = zext i32 %163 to i64
  %312 = sext i32 %65 to i64
  %313 = mul nsw i64 %312, %const_reg_qword3, !spirv.Decorations !836
  %314 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %313, i32 0
  %315 = getelementptr i16, i16 addrspace(4)* %314, i64 %.sroa.256.0.insert.ext618
  %316 = addrspacecast i16 addrspace(4)* %315 to i16 addrspace(1)*
  %317 = load i16, i16 addrspace(1)* %316, align 2
  %318 = mul nsw i64 %.sroa.256.0.insert.ext618, %const_reg_qword5, !spirv.Decorations !836
  %319 = sext i32 %68 to i64
  %320 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %318, i32 0
  %321 = getelementptr i16, i16 addrspace(4)* %320, i64 %319
  %322 = addrspacecast i16 addrspace(4)* %321 to i16 addrspace(1)*
  %323 = load i16, i16 addrspace(1)* %322, align 2
  %324 = zext i16 %317 to i32
  %325 = shl nuw i32 %324, 16, !spirv.Decorations !838
  %326 = bitcast i32 %325 to float
  %327 = zext i16 %323 to i32
  %328 = shl nuw i32 %327, 16, !spirv.Decorations !838
  %329 = bitcast i32 %328 to float
  %330 = fmul reassoc nsz arcp contract float %326, %329, !spirv.Decorations !843
  %331 = fadd reassoc nsz arcp contract float %330, %.sroa.198.1, !spirv.Decorations !843
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %311
  %.sroa.198.2 = phi float [ %331, %311 ], [ %.sroa.198.1, %._crit_edge.2.1 ]
  br i1 %76, label %332, label %._crit_edge.274

332:                                              ; preds = %.preheader.1
  %.sroa.256.0.insert.ext623 = zext i32 %163 to i64
  %333 = sext i32 %29 to i64
  %334 = mul nsw i64 %333, %const_reg_qword3, !spirv.Decorations !836
  %335 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %334, i32 0
  %336 = getelementptr i16, i16 addrspace(4)* %335, i64 %.sroa.256.0.insert.ext623
  %337 = addrspacecast i16 addrspace(4)* %336 to i16 addrspace(1)*
  %338 = load i16, i16 addrspace(1)* %337, align 2
  %339 = mul nsw i64 %.sroa.256.0.insert.ext623, %const_reg_qword5, !spirv.Decorations !836
  %340 = sext i32 %74 to i64
  %341 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %339, i32 0
  %342 = getelementptr i16, i16 addrspace(4)* %341, i64 %340
  %343 = addrspacecast i16 addrspace(4)* %342 to i16 addrspace(1)*
  %344 = load i16, i16 addrspace(1)* %343, align 2
  %345 = zext i16 %338 to i32
  %346 = shl nuw i32 %345, 16, !spirv.Decorations !838
  %347 = bitcast i32 %346 to float
  %348 = zext i16 %344 to i32
  %349 = shl nuw i32 %348, 16, !spirv.Decorations !838
  %350 = bitcast i32 %349 to float
  %351 = fmul reassoc nsz arcp contract float %347, %350, !spirv.Decorations !843
  %352 = fadd reassoc nsz arcp contract float %351, %.sroa.10.1, !spirv.Decorations !843
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %332
  %.sroa.10.2 = phi float [ %352, %332 ], [ %.sroa.10.1, %.preheader.1 ]
  br i1 %77, label %353, label %._crit_edge.1.2

353:                                              ; preds = %._crit_edge.274
  %.sroa.256.0.insert.ext628 = zext i32 %163 to i64
  %354 = sext i32 %59 to i64
  %355 = mul nsw i64 %354, %const_reg_qword3, !spirv.Decorations !836
  %356 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %355, i32 0
  %357 = getelementptr i16, i16 addrspace(4)* %356, i64 %.sroa.256.0.insert.ext628
  %358 = addrspacecast i16 addrspace(4)* %357 to i16 addrspace(1)*
  %359 = load i16, i16 addrspace(1)* %358, align 2
  %360 = mul nsw i64 %.sroa.256.0.insert.ext628, %const_reg_qword5, !spirv.Decorations !836
  %361 = sext i32 %74 to i64
  %362 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %360, i32 0
  %363 = getelementptr i16, i16 addrspace(4)* %362, i64 %361
  %364 = addrspacecast i16 addrspace(4)* %363 to i16 addrspace(1)*
  %365 = load i16, i16 addrspace(1)* %364, align 2
  %366 = zext i16 %359 to i32
  %367 = shl nuw i32 %366, 16, !spirv.Decorations !838
  %368 = bitcast i32 %367 to float
  %369 = zext i16 %365 to i32
  %370 = shl nuw i32 %369, 16, !spirv.Decorations !838
  %371 = bitcast i32 %370 to float
  %372 = fmul reassoc nsz arcp contract float %368, %371, !spirv.Decorations !843
  %373 = fadd reassoc nsz arcp contract float %372, %.sroa.74.1, !spirv.Decorations !843
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %353
  %.sroa.74.2 = phi float [ %373, %353 ], [ %.sroa.74.1, %._crit_edge.274 ]
  br i1 %78, label %374, label %._crit_edge.2.2

374:                                              ; preds = %._crit_edge.1.2
  %.sroa.256.0.insert.ext633 = zext i32 %163 to i64
  %375 = sext i32 %62 to i64
  %376 = mul nsw i64 %375, %const_reg_qword3, !spirv.Decorations !836
  %377 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %376, i32 0
  %378 = getelementptr i16, i16 addrspace(4)* %377, i64 %.sroa.256.0.insert.ext633
  %379 = addrspacecast i16 addrspace(4)* %378 to i16 addrspace(1)*
  %380 = load i16, i16 addrspace(1)* %379, align 2
  %381 = mul nsw i64 %.sroa.256.0.insert.ext633, %const_reg_qword5, !spirv.Decorations !836
  %382 = sext i32 %74 to i64
  %383 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %381, i32 0
  %384 = getelementptr i16, i16 addrspace(4)* %383, i64 %382
  %385 = addrspacecast i16 addrspace(4)* %384 to i16 addrspace(1)*
  %386 = load i16, i16 addrspace(1)* %385, align 2
  %387 = zext i16 %380 to i32
  %388 = shl nuw i32 %387, 16, !spirv.Decorations !838
  %389 = bitcast i32 %388 to float
  %390 = zext i16 %386 to i32
  %391 = shl nuw i32 %390, 16, !spirv.Decorations !838
  %392 = bitcast i32 %391 to float
  %393 = fmul reassoc nsz arcp contract float %389, %392, !spirv.Decorations !843
  %394 = fadd reassoc nsz arcp contract float %393, %.sroa.138.1, !spirv.Decorations !843
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %374
  %.sroa.138.2 = phi float [ %394, %374 ], [ %.sroa.138.1, %._crit_edge.1.2 ]
  br i1 %79, label %395, label %.preheader.2

395:                                              ; preds = %._crit_edge.2.2
  %.sroa.256.0.insert.ext638 = zext i32 %163 to i64
  %396 = sext i32 %65 to i64
  %397 = mul nsw i64 %396, %const_reg_qword3, !spirv.Decorations !836
  %398 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %397, i32 0
  %399 = getelementptr i16, i16 addrspace(4)* %398, i64 %.sroa.256.0.insert.ext638
  %400 = addrspacecast i16 addrspace(4)* %399 to i16 addrspace(1)*
  %401 = load i16, i16 addrspace(1)* %400, align 2
  %402 = mul nsw i64 %.sroa.256.0.insert.ext638, %const_reg_qword5, !spirv.Decorations !836
  %403 = sext i32 %74 to i64
  %404 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %402, i32 0
  %405 = getelementptr i16, i16 addrspace(4)* %404, i64 %403
  %406 = addrspacecast i16 addrspace(4)* %405 to i16 addrspace(1)*
  %407 = load i16, i16 addrspace(1)* %406, align 2
  %408 = zext i16 %401 to i32
  %409 = shl nuw i32 %408, 16, !spirv.Decorations !838
  %410 = bitcast i32 %409 to float
  %411 = zext i16 %407 to i32
  %412 = shl nuw i32 %411, 16, !spirv.Decorations !838
  %413 = bitcast i32 %412 to float
  %414 = fmul reassoc nsz arcp contract float %410, %413, !spirv.Decorations !843
  %415 = fadd reassoc nsz arcp contract float %414, %.sroa.202.1, !spirv.Decorations !843
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %395
  %.sroa.202.2 = phi float [ %415, %395 ], [ %.sroa.202.1, %._crit_edge.2.2 ]
  br i1 %82, label %416, label %._crit_edge.375

416:                                              ; preds = %.preheader.2
  %.sroa.256.0.insert.ext643 = zext i32 %163 to i64
  %417 = sext i32 %29 to i64
  %418 = mul nsw i64 %417, %const_reg_qword3, !spirv.Decorations !836
  %419 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %418, i32 0
  %420 = getelementptr i16, i16 addrspace(4)* %419, i64 %.sroa.256.0.insert.ext643
  %421 = addrspacecast i16 addrspace(4)* %420 to i16 addrspace(1)*
  %422 = load i16, i16 addrspace(1)* %421, align 2
  %423 = mul nsw i64 %.sroa.256.0.insert.ext643, %const_reg_qword5, !spirv.Decorations !836
  %424 = sext i32 %80 to i64
  %425 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %423, i32 0
  %426 = getelementptr i16, i16 addrspace(4)* %425, i64 %424
  %427 = addrspacecast i16 addrspace(4)* %426 to i16 addrspace(1)*
  %428 = load i16, i16 addrspace(1)* %427, align 2
  %429 = zext i16 %422 to i32
  %430 = shl nuw i32 %429, 16, !spirv.Decorations !838
  %431 = bitcast i32 %430 to float
  %432 = zext i16 %428 to i32
  %433 = shl nuw i32 %432, 16, !spirv.Decorations !838
  %434 = bitcast i32 %433 to float
  %435 = fmul reassoc nsz arcp contract float %431, %434, !spirv.Decorations !843
  %436 = fadd reassoc nsz arcp contract float %435, %.sroa.14.1, !spirv.Decorations !843
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %416
  %.sroa.14.2 = phi float [ %436, %416 ], [ %.sroa.14.1, %.preheader.2 ]
  br i1 %83, label %437, label %._crit_edge.1.3

437:                                              ; preds = %._crit_edge.375
  %.sroa.256.0.insert.ext648 = zext i32 %163 to i64
  %438 = sext i32 %59 to i64
  %439 = mul nsw i64 %438, %const_reg_qword3, !spirv.Decorations !836
  %440 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %439, i32 0
  %441 = getelementptr i16, i16 addrspace(4)* %440, i64 %.sroa.256.0.insert.ext648
  %442 = addrspacecast i16 addrspace(4)* %441 to i16 addrspace(1)*
  %443 = load i16, i16 addrspace(1)* %442, align 2
  %444 = mul nsw i64 %.sroa.256.0.insert.ext648, %const_reg_qword5, !spirv.Decorations !836
  %445 = sext i32 %80 to i64
  %446 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %444, i32 0
  %447 = getelementptr i16, i16 addrspace(4)* %446, i64 %445
  %448 = addrspacecast i16 addrspace(4)* %447 to i16 addrspace(1)*
  %449 = load i16, i16 addrspace(1)* %448, align 2
  %450 = zext i16 %443 to i32
  %451 = shl nuw i32 %450, 16, !spirv.Decorations !838
  %452 = bitcast i32 %451 to float
  %453 = zext i16 %449 to i32
  %454 = shl nuw i32 %453, 16, !spirv.Decorations !838
  %455 = bitcast i32 %454 to float
  %456 = fmul reassoc nsz arcp contract float %452, %455, !spirv.Decorations !843
  %457 = fadd reassoc nsz arcp contract float %456, %.sroa.78.1, !spirv.Decorations !843
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %437
  %.sroa.78.2 = phi float [ %457, %437 ], [ %.sroa.78.1, %._crit_edge.375 ]
  br i1 %84, label %458, label %._crit_edge.2.3

458:                                              ; preds = %._crit_edge.1.3
  %.sroa.256.0.insert.ext653 = zext i32 %163 to i64
  %459 = sext i32 %62 to i64
  %460 = mul nsw i64 %459, %const_reg_qword3, !spirv.Decorations !836
  %461 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %460, i32 0
  %462 = getelementptr i16, i16 addrspace(4)* %461, i64 %.sroa.256.0.insert.ext653
  %463 = addrspacecast i16 addrspace(4)* %462 to i16 addrspace(1)*
  %464 = load i16, i16 addrspace(1)* %463, align 2
  %465 = mul nsw i64 %.sroa.256.0.insert.ext653, %const_reg_qword5, !spirv.Decorations !836
  %466 = sext i32 %80 to i64
  %467 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %465, i32 0
  %468 = getelementptr i16, i16 addrspace(4)* %467, i64 %466
  %469 = addrspacecast i16 addrspace(4)* %468 to i16 addrspace(1)*
  %470 = load i16, i16 addrspace(1)* %469, align 2
  %471 = zext i16 %464 to i32
  %472 = shl nuw i32 %471, 16, !spirv.Decorations !838
  %473 = bitcast i32 %472 to float
  %474 = zext i16 %470 to i32
  %475 = shl nuw i32 %474, 16, !spirv.Decorations !838
  %476 = bitcast i32 %475 to float
  %477 = fmul reassoc nsz arcp contract float %473, %476, !spirv.Decorations !843
  %478 = fadd reassoc nsz arcp contract float %477, %.sroa.142.1, !spirv.Decorations !843
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %458
  %.sroa.142.2 = phi float [ %478, %458 ], [ %.sroa.142.1, %._crit_edge.1.3 ]
  br i1 %85, label %479, label %.preheader.3

479:                                              ; preds = %._crit_edge.2.3
  %.sroa.256.0.insert.ext658 = zext i32 %163 to i64
  %480 = sext i32 %65 to i64
  %481 = mul nsw i64 %480, %const_reg_qword3, !spirv.Decorations !836
  %482 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %481, i32 0
  %483 = getelementptr i16, i16 addrspace(4)* %482, i64 %.sroa.256.0.insert.ext658
  %484 = addrspacecast i16 addrspace(4)* %483 to i16 addrspace(1)*
  %485 = load i16, i16 addrspace(1)* %484, align 2
  %486 = mul nsw i64 %.sroa.256.0.insert.ext658, %const_reg_qword5, !spirv.Decorations !836
  %487 = sext i32 %80 to i64
  %488 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %486, i32 0
  %489 = getelementptr i16, i16 addrspace(4)* %488, i64 %487
  %490 = addrspacecast i16 addrspace(4)* %489 to i16 addrspace(1)*
  %491 = load i16, i16 addrspace(1)* %490, align 2
  %492 = zext i16 %485 to i32
  %493 = shl nuw i32 %492, 16, !spirv.Decorations !838
  %494 = bitcast i32 %493 to float
  %495 = zext i16 %491 to i32
  %496 = shl nuw i32 %495, 16, !spirv.Decorations !838
  %497 = bitcast i32 %496 to float
  %498 = fmul reassoc nsz arcp contract float %494, %497, !spirv.Decorations !843
  %499 = fadd reassoc nsz arcp contract float %498, %.sroa.206.1, !spirv.Decorations !843
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %479
  %.sroa.206.2 = phi float [ %499, %479 ], [ %.sroa.206.1, %._crit_edge.2.3 ]
  br i1 %88, label %500, label %._crit_edge.4

500:                                              ; preds = %.preheader.3
  %.sroa.256.0.insert.ext663 = zext i32 %163 to i64
  %501 = sext i32 %29 to i64
  %502 = mul nsw i64 %501, %const_reg_qword3, !spirv.Decorations !836
  %503 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %502, i32 0
  %504 = getelementptr i16, i16 addrspace(4)* %503, i64 %.sroa.256.0.insert.ext663
  %505 = addrspacecast i16 addrspace(4)* %504 to i16 addrspace(1)*
  %506 = load i16, i16 addrspace(1)* %505, align 2
  %507 = mul nsw i64 %.sroa.256.0.insert.ext663, %const_reg_qword5, !spirv.Decorations !836
  %508 = sext i32 %86 to i64
  %509 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %507, i32 0
  %510 = getelementptr i16, i16 addrspace(4)* %509, i64 %508
  %511 = addrspacecast i16 addrspace(4)* %510 to i16 addrspace(1)*
  %512 = load i16, i16 addrspace(1)* %511, align 2
  %513 = zext i16 %506 to i32
  %514 = shl nuw i32 %513, 16, !spirv.Decorations !838
  %515 = bitcast i32 %514 to float
  %516 = zext i16 %512 to i32
  %517 = shl nuw i32 %516, 16, !spirv.Decorations !838
  %518 = bitcast i32 %517 to float
  %519 = fmul reassoc nsz arcp contract float %515, %518, !spirv.Decorations !843
  %520 = fadd reassoc nsz arcp contract float %519, %.sroa.18.1, !spirv.Decorations !843
  br label %._crit_edge.4

._crit_edge.4:                                    ; preds = %.preheader.3, %500
  %.sroa.18.2 = phi float [ %520, %500 ], [ %.sroa.18.1, %.preheader.3 ]
  br i1 %89, label %521, label %._crit_edge.1.4

521:                                              ; preds = %._crit_edge.4
  %.sroa.256.0.insert.ext668 = zext i32 %163 to i64
  %522 = sext i32 %59 to i64
  %523 = mul nsw i64 %522, %const_reg_qword3, !spirv.Decorations !836
  %524 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %523, i32 0
  %525 = getelementptr i16, i16 addrspace(4)* %524, i64 %.sroa.256.0.insert.ext668
  %526 = addrspacecast i16 addrspace(4)* %525 to i16 addrspace(1)*
  %527 = load i16, i16 addrspace(1)* %526, align 2
  %528 = mul nsw i64 %.sroa.256.0.insert.ext668, %const_reg_qword5, !spirv.Decorations !836
  %529 = sext i32 %86 to i64
  %530 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %528, i32 0
  %531 = getelementptr i16, i16 addrspace(4)* %530, i64 %529
  %532 = addrspacecast i16 addrspace(4)* %531 to i16 addrspace(1)*
  %533 = load i16, i16 addrspace(1)* %532, align 2
  %534 = zext i16 %527 to i32
  %535 = shl nuw i32 %534, 16, !spirv.Decorations !838
  %536 = bitcast i32 %535 to float
  %537 = zext i16 %533 to i32
  %538 = shl nuw i32 %537, 16, !spirv.Decorations !838
  %539 = bitcast i32 %538 to float
  %540 = fmul reassoc nsz arcp contract float %536, %539, !spirv.Decorations !843
  %541 = fadd reassoc nsz arcp contract float %540, %.sroa.82.1, !spirv.Decorations !843
  br label %._crit_edge.1.4

._crit_edge.1.4:                                  ; preds = %._crit_edge.4, %521
  %.sroa.82.2 = phi float [ %541, %521 ], [ %.sroa.82.1, %._crit_edge.4 ]
  br i1 %90, label %542, label %._crit_edge.2.4

542:                                              ; preds = %._crit_edge.1.4
  %.sroa.256.0.insert.ext673 = zext i32 %163 to i64
  %543 = sext i32 %62 to i64
  %544 = mul nsw i64 %543, %const_reg_qword3, !spirv.Decorations !836
  %545 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %544, i32 0
  %546 = getelementptr i16, i16 addrspace(4)* %545, i64 %.sroa.256.0.insert.ext673
  %547 = addrspacecast i16 addrspace(4)* %546 to i16 addrspace(1)*
  %548 = load i16, i16 addrspace(1)* %547, align 2
  %549 = mul nsw i64 %.sroa.256.0.insert.ext673, %const_reg_qword5, !spirv.Decorations !836
  %550 = sext i32 %86 to i64
  %551 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %549, i32 0
  %552 = getelementptr i16, i16 addrspace(4)* %551, i64 %550
  %553 = addrspacecast i16 addrspace(4)* %552 to i16 addrspace(1)*
  %554 = load i16, i16 addrspace(1)* %553, align 2
  %555 = zext i16 %548 to i32
  %556 = shl nuw i32 %555, 16, !spirv.Decorations !838
  %557 = bitcast i32 %556 to float
  %558 = zext i16 %554 to i32
  %559 = shl nuw i32 %558, 16, !spirv.Decorations !838
  %560 = bitcast i32 %559 to float
  %561 = fmul reassoc nsz arcp contract float %557, %560, !spirv.Decorations !843
  %562 = fadd reassoc nsz arcp contract float %561, %.sroa.146.1, !spirv.Decorations !843
  br label %._crit_edge.2.4

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4, %542
  %.sroa.146.2 = phi float [ %562, %542 ], [ %.sroa.146.1, %._crit_edge.1.4 ]
  br i1 %91, label %563, label %.preheader.4

563:                                              ; preds = %._crit_edge.2.4
  %.sroa.256.0.insert.ext678 = zext i32 %163 to i64
  %564 = sext i32 %65 to i64
  %565 = mul nsw i64 %564, %const_reg_qword3, !spirv.Decorations !836
  %566 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %565, i32 0
  %567 = getelementptr i16, i16 addrspace(4)* %566, i64 %.sroa.256.0.insert.ext678
  %568 = addrspacecast i16 addrspace(4)* %567 to i16 addrspace(1)*
  %569 = load i16, i16 addrspace(1)* %568, align 2
  %570 = mul nsw i64 %.sroa.256.0.insert.ext678, %const_reg_qword5, !spirv.Decorations !836
  %571 = sext i32 %86 to i64
  %572 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %570, i32 0
  %573 = getelementptr i16, i16 addrspace(4)* %572, i64 %571
  %574 = addrspacecast i16 addrspace(4)* %573 to i16 addrspace(1)*
  %575 = load i16, i16 addrspace(1)* %574, align 2
  %576 = zext i16 %569 to i32
  %577 = shl nuw i32 %576, 16, !spirv.Decorations !838
  %578 = bitcast i32 %577 to float
  %579 = zext i16 %575 to i32
  %580 = shl nuw i32 %579, 16, !spirv.Decorations !838
  %581 = bitcast i32 %580 to float
  %582 = fmul reassoc nsz arcp contract float %578, %581, !spirv.Decorations !843
  %583 = fadd reassoc nsz arcp contract float %582, %.sroa.210.1, !spirv.Decorations !843
  br label %.preheader.4

.preheader.4:                                     ; preds = %._crit_edge.2.4, %563
  %.sroa.210.2 = phi float [ %583, %563 ], [ %.sroa.210.1, %._crit_edge.2.4 ]
  br i1 %94, label %584, label %._crit_edge.5

584:                                              ; preds = %.preheader.4
  %.sroa.256.0.insert.ext683 = zext i32 %163 to i64
  %585 = sext i32 %29 to i64
  %586 = mul nsw i64 %585, %const_reg_qword3, !spirv.Decorations !836
  %587 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %586, i32 0
  %588 = getelementptr i16, i16 addrspace(4)* %587, i64 %.sroa.256.0.insert.ext683
  %589 = addrspacecast i16 addrspace(4)* %588 to i16 addrspace(1)*
  %590 = load i16, i16 addrspace(1)* %589, align 2
  %591 = mul nsw i64 %.sroa.256.0.insert.ext683, %const_reg_qword5, !spirv.Decorations !836
  %592 = sext i32 %92 to i64
  %593 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %591, i32 0
  %594 = getelementptr i16, i16 addrspace(4)* %593, i64 %592
  %595 = addrspacecast i16 addrspace(4)* %594 to i16 addrspace(1)*
  %596 = load i16, i16 addrspace(1)* %595, align 2
  %597 = zext i16 %590 to i32
  %598 = shl nuw i32 %597, 16, !spirv.Decorations !838
  %599 = bitcast i32 %598 to float
  %600 = zext i16 %596 to i32
  %601 = shl nuw i32 %600, 16, !spirv.Decorations !838
  %602 = bitcast i32 %601 to float
  %603 = fmul reassoc nsz arcp contract float %599, %602, !spirv.Decorations !843
  %604 = fadd reassoc nsz arcp contract float %603, %.sroa.22.1, !spirv.Decorations !843
  br label %._crit_edge.5

._crit_edge.5:                                    ; preds = %.preheader.4, %584
  %.sroa.22.2 = phi float [ %604, %584 ], [ %.sroa.22.1, %.preheader.4 ]
  br i1 %95, label %605, label %._crit_edge.1.5

605:                                              ; preds = %._crit_edge.5
  %.sroa.256.0.insert.ext688 = zext i32 %163 to i64
  %606 = sext i32 %59 to i64
  %607 = mul nsw i64 %606, %const_reg_qword3, !spirv.Decorations !836
  %608 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %607, i32 0
  %609 = getelementptr i16, i16 addrspace(4)* %608, i64 %.sroa.256.0.insert.ext688
  %610 = addrspacecast i16 addrspace(4)* %609 to i16 addrspace(1)*
  %611 = load i16, i16 addrspace(1)* %610, align 2
  %612 = mul nsw i64 %.sroa.256.0.insert.ext688, %const_reg_qword5, !spirv.Decorations !836
  %613 = sext i32 %92 to i64
  %614 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %612, i32 0
  %615 = getelementptr i16, i16 addrspace(4)* %614, i64 %613
  %616 = addrspacecast i16 addrspace(4)* %615 to i16 addrspace(1)*
  %617 = load i16, i16 addrspace(1)* %616, align 2
  %618 = zext i16 %611 to i32
  %619 = shl nuw i32 %618, 16, !spirv.Decorations !838
  %620 = bitcast i32 %619 to float
  %621 = zext i16 %617 to i32
  %622 = shl nuw i32 %621, 16, !spirv.Decorations !838
  %623 = bitcast i32 %622 to float
  %624 = fmul reassoc nsz arcp contract float %620, %623, !spirv.Decorations !843
  %625 = fadd reassoc nsz arcp contract float %624, %.sroa.86.1, !spirv.Decorations !843
  br label %._crit_edge.1.5

._crit_edge.1.5:                                  ; preds = %._crit_edge.5, %605
  %.sroa.86.2 = phi float [ %625, %605 ], [ %.sroa.86.1, %._crit_edge.5 ]
  br i1 %96, label %626, label %._crit_edge.2.5

626:                                              ; preds = %._crit_edge.1.5
  %.sroa.256.0.insert.ext693 = zext i32 %163 to i64
  %627 = sext i32 %62 to i64
  %628 = mul nsw i64 %627, %const_reg_qword3, !spirv.Decorations !836
  %629 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %628, i32 0
  %630 = getelementptr i16, i16 addrspace(4)* %629, i64 %.sroa.256.0.insert.ext693
  %631 = addrspacecast i16 addrspace(4)* %630 to i16 addrspace(1)*
  %632 = load i16, i16 addrspace(1)* %631, align 2
  %633 = mul nsw i64 %.sroa.256.0.insert.ext693, %const_reg_qword5, !spirv.Decorations !836
  %634 = sext i32 %92 to i64
  %635 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %633, i32 0
  %636 = getelementptr i16, i16 addrspace(4)* %635, i64 %634
  %637 = addrspacecast i16 addrspace(4)* %636 to i16 addrspace(1)*
  %638 = load i16, i16 addrspace(1)* %637, align 2
  %639 = zext i16 %632 to i32
  %640 = shl nuw i32 %639, 16, !spirv.Decorations !838
  %641 = bitcast i32 %640 to float
  %642 = zext i16 %638 to i32
  %643 = shl nuw i32 %642, 16, !spirv.Decorations !838
  %644 = bitcast i32 %643 to float
  %645 = fmul reassoc nsz arcp contract float %641, %644, !spirv.Decorations !843
  %646 = fadd reassoc nsz arcp contract float %645, %.sroa.150.1, !spirv.Decorations !843
  br label %._crit_edge.2.5

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5, %626
  %.sroa.150.2 = phi float [ %646, %626 ], [ %.sroa.150.1, %._crit_edge.1.5 ]
  br i1 %97, label %647, label %.preheader.5

647:                                              ; preds = %._crit_edge.2.5
  %.sroa.256.0.insert.ext698 = zext i32 %163 to i64
  %648 = sext i32 %65 to i64
  %649 = mul nsw i64 %648, %const_reg_qword3, !spirv.Decorations !836
  %650 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %649, i32 0
  %651 = getelementptr i16, i16 addrspace(4)* %650, i64 %.sroa.256.0.insert.ext698
  %652 = addrspacecast i16 addrspace(4)* %651 to i16 addrspace(1)*
  %653 = load i16, i16 addrspace(1)* %652, align 2
  %654 = mul nsw i64 %.sroa.256.0.insert.ext698, %const_reg_qword5, !spirv.Decorations !836
  %655 = sext i32 %92 to i64
  %656 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %654, i32 0
  %657 = getelementptr i16, i16 addrspace(4)* %656, i64 %655
  %658 = addrspacecast i16 addrspace(4)* %657 to i16 addrspace(1)*
  %659 = load i16, i16 addrspace(1)* %658, align 2
  %660 = zext i16 %653 to i32
  %661 = shl nuw i32 %660, 16, !spirv.Decorations !838
  %662 = bitcast i32 %661 to float
  %663 = zext i16 %659 to i32
  %664 = shl nuw i32 %663, 16, !spirv.Decorations !838
  %665 = bitcast i32 %664 to float
  %666 = fmul reassoc nsz arcp contract float %662, %665, !spirv.Decorations !843
  %667 = fadd reassoc nsz arcp contract float %666, %.sroa.214.1, !spirv.Decorations !843
  br label %.preheader.5

.preheader.5:                                     ; preds = %._crit_edge.2.5, %647
  %.sroa.214.2 = phi float [ %667, %647 ], [ %.sroa.214.1, %._crit_edge.2.5 ]
  br i1 %100, label %668, label %._crit_edge.6

668:                                              ; preds = %.preheader.5
  %.sroa.256.0.insert.ext703 = zext i32 %163 to i64
  %669 = sext i32 %29 to i64
  %670 = mul nsw i64 %669, %const_reg_qword3, !spirv.Decorations !836
  %671 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %670, i32 0
  %672 = getelementptr i16, i16 addrspace(4)* %671, i64 %.sroa.256.0.insert.ext703
  %673 = addrspacecast i16 addrspace(4)* %672 to i16 addrspace(1)*
  %674 = load i16, i16 addrspace(1)* %673, align 2
  %675 = mul nsw i64 %.sroa.256.0.insert.ext703, %const_reg_qword5, !spirv.Decorations !836
  %676 = sext i32 %98 to i64
  %677 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %675, i32 0
  %678 = getelementptr i16, i16 addrspace(4)* %677, i64 %676
  %679 = addrspacecast i16 addrspace(4)* %678 to i16 addrspace(1)*
  %680 = load i16, i16 addrspace(1)* %679, align 2
  %681 = zext i16 %674 to i32
  %682 = shl nuw i32 %681, 16, !spirv.Decorations !838
  %683 = bitcast i32 %682 to float
  %684 = zext i16 %680 to i32
  %685 = shl nuw i32 %684, 16, !spirv.Decorations !838
  %686 = bitcast i32 %685 to float
  %687 = fmul reassoc nsz arcp contract float %683, %686, !spirv.Decorations !843
  %688 = fadd reassoc nsz arcp contract float %687, %.sroa.26.1, !spirv.Decorations !843
  br label %._crit_edge.6

._crit_edge.6:                                    ; preds = %.preheader.5, %668
  %.sroa.26.2 = phi float [ %688, %668 ], [ %.sroa.26.1, %.preheader.5 ]
  br i1 %101, label %689, label %._crit_edge.1.6

689:                                              ; preds = %._crit_edge.6
  %.sroa.256.0.insert.ext708 = zext i32 %163 to i64
  %690 = sext i32 %59 to i64
  %691 = mul nsw i64 %690, %const_reg_qword3, !spirv.Decorations !836
  %692 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %691, i32 0
  %693 = getelementptr i16, i16 addrspace(4)* %692, i64 %.sroa.256.0.insert.ext708
  %694 = addrspacecast i16 addrspace(4)* %693 to i16 addrspace(1)*
  %695 = load i16, i16 addrspace(1)* %694, align 2
  %696 = mul nsw i64 %.sroa.256.0.insert.ext708, %const_reg_qword5, !spirv.Decorations !836
  %697 = sext i32 %98 to i64
  %698 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %696, i32 0
  %699 = getelementptr i16, i16 addrspace(4)* %698, i64 %697
  %700 = addrspacecast i16 addrspace(4)* %699 to i16 addrspace(1)*
  %701 = load i16, i16 addrspace(1)* %700, align 2
  %702 = zext i16 %695 to i32
  %703 = shl nuw i32 %702, 16, !spirv.Decorations !838
  %704 = bitcast i32 %703 to float
  %705 = zext i16 %701 to i32
  %706 = shl nuw i32 %705, 16, !spirv.Decorations !838
  %707 = bitcast i32 %706 to float
  %708 = fmul reassoc nsz arcp contract float %704, %707, !spirv.Decorations !843
  %709 = fadd reassoc nsz arcp contract float %708, %.sroa.90.1, !spirv.Decorations !843
  br label %._crit_edge.1.6

._crit_edge.1.6:                                  ; preds = %._crit_edge.6, %689
  %.sroa.90.2 = phi float [ %709, %689 ], [ %.sroa.90.1, %._crit_edge.6 ]
  br i1 %102, label %710, label %._crit_edge.2.6

710:                                              ; preds = %._crit_edge.1.6
  %.sroa.256.0.insert.ext713 = zext i32 %163 to i64
  %711 = sext i32 %62 to i64
  %712 = mul nsw i64 %711, %const_reg_qword3, !spirv.Decorations !836
  %713 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %712, i32 0
  %714 = getelementptr i16, i16 addrspace(4)* %713, i64 %.sroa.256.0.insert.ext713
  %715 = addrspacecast i16 addrspace(4)* %714 to i16 addrspace(1)*
  %716 = load i16, i16 addrspace(1)* %715, align 2
  %717 = mul nsw i64 %.sroa.256.0.insert.ext713, %const_reg_qword5, !spirv.Decorations !836
  %718 = sext i32 %98 to i64
  %719 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %717, i32 0
  %720 = getelementptr i16, i16 addrspace(4)* %719, i64 %718
  %721 = addrspacecast i16 addrspace(4)* %720 to i16 addrspace(1)*
  %722 = load i16, i16 addrspace(1)* %721, align 2
  %723 = zext i16 %716 to i32
  %724 = shl nuw i32 %723, 16, !spirv.Decorations !838
  %725 = bitcast i32 %724 to float
  %726 = zext i16 %722 to i32
  %727 = shl nuw i32 %726, 16, !spirv.Decorations !838
  %728 = bitcast i32 %727 to float
  %729 = fmul reassoc nsz arcp contract float %725, %728, !spirv.Decorations !843
  %730 = fadd reassoc nsz arcp contract float %729, %.sroa.154.1, !spirv.Decorations !843
  br label %._crit_edge.2.6

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6, %710
  %.sroa.154.2 = phi float [ %730, %710 ], [ %.sroa.154.1, %._crit_edge.1.6 ]
  br i1 %103, label %731, label %.preheader.6

731:                                              ; preds = %._crit_edge.2.6
  %.sroa.256.0.insert.ext718 = zext i32 %163 to i64
  %732 = sext i32 %65 to i64
  %733 = mul nsw i64 %732, %const_reg_qword3, !spirv.Decorations !836
  %734 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %733, i32 0
  %735 = getelementptr i16, i16 addrspace(4)* %734, i64 %.sroa.256.0.insert.ext718
  %736 = addrspacecast i16 addrspace(4)* %735 to i16 addrspace(1)*
  %737 = load i16, i16 addrspace(1)* %736, align 2
  %738 = mul nsw i64 %.sroa.256.0.insert.ext718, %const_reg_qword5, !spirv.Decorations !836
  %739 = sext i32 %98 to i64
  %740 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %738, i32 0
  %741 = getelementptr i16, i16 addrspace(4)* %740, i64 %739
  %742 = addrspacecast i16 addrspace(4)* %741 to i16 addrspace(1)*
  %743 = load i16, i16 addrspace(1)* %742, align 2
  %744 = zext i16 %737 to i32
  %745 = shl nuw i32 %744, 16, !spirv.Decorations !838
  %746 = bitcast i32 %745 to float
  %747 = zext i16 %743 to i32
  %748 = shl nuw i32 %747, 16, !spirv.Decorations !838
  %749 = bitcast i32 %748 to float
  %750 = fmul reassoc nsz arcp contract float %746, %749, !spirv.Decorations !843
  %751 = fadd reassoc nsz arcp contract float %750, %.sroa.218.1, !spirv.Decorations !843
  br label %.preheader.6

.preheader.6:                                     ; preds = %._crit_edge.2.6, %731
  %.sroa.218.2 = phi float [ %751, %731 ], [ %.sroa.218.1, %._crit_edge.2.6 ]
  br i1 %106, label %752, label %._crit_edge.7

752:                                              ; preds = %.preheader.6
  %.sroa.256.0.insert.ext723 = zext i32 %163 to i64
  %753 = sext i32 %29 to i64
  %754 = mul nsw i64 %753, %const_reg_qword3, !spirv.Decorations !836
  %755 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %754, i32 0
  %756 = getelementptr i16, i16 addrspace(4)* %755, i64 %.sroa.256.0.insert.ext723
  %757 = addrspacecast i16 addrspace(4)* %756 to i16 addrspace(1)*
  %758 = load i16, i16 addrspace(1)* %757, align 2
  %759 = mul nsw i64 %.sroa.256.0.insert.ext723, %const_reg_qword5, !spirv.Decorations !836
  %760 = sext i32 %104 to i64
  %761 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %759, i32 0
  %762 = getelementptr i16, i16 addrspace(4)* %761, i64 %760
  %763 = addrspacecast i16 addrspace(4)* %762 to i16 addrspace(1)*
  %764 = load i16, i16 addrspace(1)* %763, align 2
  %765 = zext i16 %758 to i32
  %766 = shl nuw i32 %765, 16, !spirv.Decorations !838
  %767 = bitcast i32 %766 to float
  %768 = zext i16 %764 to i32
  %769 = shl nuw i32 %768, 16, !spirv.Decorations !838
  %770 = bitcast i32 %769 to float
  %771 = fmul reassoc nsz arcp contract float %767, %770, !spirv.Decorations !843
  %772 = fadd reassoc nsz arcp contract float %771, %.sroa.30.1, !spirv.Decorations !843
  br label %._crit_edge.7

._crit_edge.7:                                    ; preds = %.preheader.6, %752
  %.sroa.30.2 = phi float [ %772, %752 ], [ %.sroa.30.1, %.preheader.6 ]
  br i1 %107, label %773, label %._crit_edge.1.7

773:                                              ; preds = %._crit_edge.7
  %.sroa.256.0.insert.ext728 = zext i32 %163 to i64
  %774 = sext i32 %59 to i64
  %775 = mul nsw i64 %774, %const_reg_qword3, !spirv.Decorations !836
  %776 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %775, i32 0
  %777 = getelementptr i16, i16 addrspace(4)* %776, i64 %.sroa.256.0.insert.ext728
  %778 = addrspacecast i16 addrspace(4)* %777 to i16 addrspace(1)*
  %779 = load i16, i16 addrspace(1)* %778, align 2
  %780 = mul nsw i64 %.sroa.256.0.insert.ext728, %const_reg_qword5, !spirv.Decorations !836
  %781 = sext i32 %104 to i64
  %782 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %780, i32 0
  %783 = getelementptr i16, i16 addrspace(4)* %782, i64 %781
  %784 = addrspacecast i16 addrspace(4)* %783 to i16 addrspace(1)*
  %785 = load i16, i16 addrspace(1)* %784, align 2
  %786 = zext i16 %779 to i32
  %787 = shl nuw i32 %786, 16, !spirv.Decorations !838
  %788 = bitcast i32 %787 to float
  %789 = zext i16 %785 to i32
  %790 = shl nuw i32 %789, 16, !spirv.Decorations !838
  %791 = bitcast i32 %790 to float
  %792 = fmul reassoc nsz arcp contract float %788, %791, !spirv.Decorations !843
  %793 = fadd reassoc nsz arcp contract float %792, %.sroa.94.1, !spirv.Decorations !843
  br label %._crit_edge.1.7

._crit_edge.1.7:                                  ; preds = %._crit_edge.7, %773
  %.sroa.94.2 = phi float [ %793, %773 ], [ %.sroa.94.1, %._crit_edge.7 ]
  br i1 %108, label %794, label %._crit_edge.2.7

794:                                              ; preds = %._crit_edge.1.7
  %.sroa.256.0.insert.ext733 = zext i32 %163 to i64
  %795 = sext i32 %62 to i64
  %796 = mul nsw i64 %795, %const_reg_qword3, !spirv.Decorations !836
  %797 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %796, i32 0
  %798 = getelementptr i16, i16 addrspace(4)* %797, i64 %.sroa.256.0.insert.ext733
  %799 = addrspacecast i16 addrspace(4)* %798 to i16 addrspace(1)*
  %800 = load i16, i16 addrspace(1)* %799, align 2
  %801 = mul nsw i64 %.sroa.256.0.insert.ext733, %const_reg_qword5, !spirv.Decorations !836
  %802 = sext i32 %104 to i64
  %803 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %801, i32 0
  %804 = getelementptr i16, i16 addrspace(4)* %803, i64 %802
  %805 = addrspacecast i16 addrspace(4)* %804 to i16 addrspace(1)*
  %806 = load i16, i16 addrspace(1)* %805, align 2
  %807 = zext i16 %800 to i32
  %808 = shl nuw i32 %807, 16, !spirv.Decorations !838
  %809 = bitcast i32 %808 to float
  %810 = zext i16 %806 to i32
  %811 = shl nuw i32 %810, 16, !spirv.Decorations !838
  %812 = bitcast i32 %811 to float
  %813 = fmul reassoc nsz arcp contract float %809, %812, !spirv.Decorations !843
  %814 = fadd reassoc nsz arcp contract float %813, %.sroa.158.1, !spirv.Decorations !843
  br label %._crit_edge.2.7

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7, %794
  %.sroa.158.2 = phi float [ %814, %794 ], [ %.sroa.158.1, %._crit_edge.1.7 ]
  br i1 %109, label %815, label %.preheader.7

815:                                              ; preds = %._crit_edge.2.7
  %.sroa.256.0.insert.ext738 = zext i32 %163 to i64
  %816 = sext i32 %65 to i64
  %817 = mul nsw i64 %816, %const_reg_qword3, !spirv.Decorations !836
  %818 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %817, i32 0
  %819 = getelementptr i16, i16 addrspace(4)* %818, i64 %.sroa.256.0.insert.ext738
  %820 = addrspacecast i16 addrspace(4)* %819 to i16 addrspace(1)*
  %821 = load i16, i16 addrspace(1)* %820, align 2
  %822 = mul nsw i64 %.sroa.256.0.insert.ext738, %const_reg_qword5, !spirv.Decorations !836
  %823 = sext i32 %104 to i64
  %824 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %822, i32 0
  %825 = getelementptr i16, i16 addrspace(4)* %824, i64 %823
  %826 = addrspacecast i16 addrspace(4)* %825 to i16 addrspace(1)*
  %827 = load i16, i16 addrspace(1)* %826, align 2
  %828 = zext i16 %821 to i32
  %829 = shl nuw i32 %828, 16, !spirv.Decorations !838
  %830 = bitcast i32 %829 to float
  %831 = zext i16 %827 to i32
  %832 = shl nuw i32 %831, 16, !spirv.Decorations !838
  %833 = bitcast i32 %832 to float
  %834 = fmul reassoc nsz arcp contract float %830, %833, !spirv.Decorations !843
  %835 = fadd reassoc nsz arcp contract float %834, %.sroa.222.1, !spirv.Decorations !843
  br label %.preheader.7

.preheader.7:                                     ; preds = %._crit_edge.2.7, %815
  %.sroa.222.2 = phi float [ %835, %815 ], [ %.sroa.222.1, %._crit_edge.2.7 ]
  br i1 %112, label %836, label %._crit_edge.8

836:                                              ; preds = %.preheader.7
  %.sroa.256.0.insert.ext743 = zext i32 %163 to i64
  %837 = sext i32 %29 to i64
  %838 = mul nsw i64 %837, %const_reg_qword3, !spirv.Decorations !836
  %839 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %838, i32 0
  %840 = getelementptr i16, i16 addrspace(4)* %839, i64 %.sroa.256.0.insert.ext743
  %841 = addrspacecast i16 addrspace(4)* %840 to i16 addrspace(1)*
  %842 = load i16, i16 addrspace(1)* %841, align 2
  %843 = mul nsw i64 %.sroa.256.0.insert.ext743, %const_reg_qword5, !spirv.Decorations !836
  %844 = sext i32 %110 to i64
  %845 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %843, i32 0
  %846 = getelementptr i16, i16 addrspace(4)* %845, i64 %844
  %847 = addrspacecast i16 addrspace(4)* %846 to i16 addrspace(1)*
  %848 = load i16, i16 addrspace(1)* %847, align 2
  %849 = zext i16 %842 to i32
  %850 = shl nuw i32 %849, 16, !spirv.Decorations !838
  %851 = bitcast i32 %850 to float
  %852 = zext i16 %848 to i32
  %853 = shl nuw i32 %852, 16, !spirv.Decorations !838
  %854 = bitcast i32 %853 to float
  %855 = fmul reassoc nsz arcp contract float %851, %854, !spirv.Decorations !843
  %856 = fadd reassoc nsz arcp contract float %855, %.sroa.34.1, !spirv.Decorations !843
  br label %._crit_edge.8

._crit_edge.8:                                    ; preds = %.preheader.7, %836
  %.sroa.34.2 = phi float [ %856, %836 ], [ %.sroa.34.1, %.preheader.7 ]
  br i1 %113, label %857, label %._crit_edge.1.8

857:                                              ; preds = %._crit_edge.8
  %.sroa.256.0.insert.ext748 = zext i32 %163 to i64
  %858 = sext i32 %59 to i64
  %859 = mul nsw i64 %858, %const_reg_qword3, !spirv.Decorations !836
  %860 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %859, i32 0
  %861 = getelementptr i16, i16 addrspace(4)* %860, i64 %.sroa.256.0.insert.ext748
  %862 = addrspacecast i16 addrspace(4)* %861 to i16 addrspace(1)*
  %863 = load i16, i16 addrspace(1)* %862, align 2
  %864 = mul nsw i64 %.sroa.256.0.insert.ext748, %const_reg_qword5, !spirv.Decorations !836
  %865 = sext i32 %110 to i64
  %866 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %864, i32 0
  %867 = getelementptr i16, i16 addrspace(4)* %866, i64 %865
  %868 = addrspacecast i16 addrspace(4)* %867 to i16 addrspace(1)*
  %869 = load i16, i16 addrspace(1)* %868, align 2
  %870 = zext i16 %863 to i32
  %871 = shl nuw i32 %870, 16, !spirv.Decorations !838
  %872 = bitcast i32 %871 to float
  %873 = zext i16 %869 to i32
  %874 = shl nuw i32 %873, 16, !spirv.Decorations !838
  %875 = bitcast i32 %874 to float
  %876 = fmul reassoc nsz arcp contract float %872, %875, !spirv.Decorations !843
  %877 = fadd reassoc nsz arcp contract float %876, %.sroa.98.1, !spirv.Decorations !843
  br label %._crit_edge.1.8

._crit_edge.1.8:                                  ; preds = %._crit_edge.8, %857
  %.sroa.98.2 = phi float [ %877, %857 ], [ %.sroa.98.1, %._crit_edge.8 ]
  br i1 %114, label %878, label %._crit_edge.2.8

878:                                              ; preds = %._crit_edge.1.8
  %.sroa.256.0.insert.ext753 = zext i32 %163 to i64
  %879 = sext i32 %62 to i64
  %880 = mul nsw i64 %879, %const_reg_qword3, !spirv.Decorations !836
  %881 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %880, i32 0
  %882 = getelementptr i16, i16 addrspace(4)* %881, i64 %.sroa.256.0.insert.ext753
  %883 = addrspacecast i16 addrspace(4)* %882 to i16 addrspace(1)*
  %884 = load i16, i16 addrspace(1)* %883, align 2
  %885 = mul nsw i64 %.sroa.256.0.insert.ext753, %const_reg_qword5, !spirv.Decorations !836
  %886 = sext i32 %110 to i64
  %887 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %885, i32 0
  %888 = getelementptr i16, i16 addrspace(4)* %887, i64 %886
  %889 = addrspacecast i16 addrspace(4)* %888 to i16 addrspace(1)*
  %890 = load i16, i16 addrspace(1)* %889, align 2
  %891 = zext i16 %884 to i32
  %892 = shl nuw i32 %891, 16, !spirv.Decorations !838
  %893 = bitcast i32 %892 to float
  %894 = zext i16 %890 to i32
  %895 = shl nuw i32 %894, 16, !spirv.Decorations !838
  %896 = bitcast i32 %895 to float
  %897 = fmul reassoc nsz arcp contract float %893, %896, !spirv.Decorations !843
  %898 = fadd reassoc nsz arcp contract float %897, %.sroa.162.1, !spirv.Decorations !843
  br label %._crit_edge.2.8

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8, %878
  %.sroa.162.2 = phi float [ %898, %878 ], [ %.sroa.162.1, %._crit_edge.1.8 ]
  br i1 %115, label %899, label %.preheader.8

899:                                              ; preds = %._crit_edge.2.8
  %.sroa.256.0.insert.ext758 = zext i32 %163 to i64
  %900 = sext i32 %65 to i64
  %901 = mul nsw i64 %900, %const_reg_qword3, !spirv.Decorations !836
  %902 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %901, i32 0
  %903 = getelementptr i16, i16 addrspace(4)* %902, i64 %.sroa.256.0.insert.ext758
  %904 = addrspacecast i16 addrspace(4)* %903 to i16 addrspace(1)*
  %905 = load i16, i16 addrspace(1)* %904, align 2
  %906 = mul nsw i64 %.sroa.256.0.insert.ext758, %const_reg_qword5, !spirv.Decorations !836
  %907 = sext i32 %110 to i64
  %908 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %906, i32 0
  %909 = getelementptr i16, i16 addrspace(4)* %908, i64 %907
  %910 = addrspacecast i16 addrspace(4)* %909 to i16 addrspace(1)*
  %911 = load i16, i16 addrspace(1)* %910, align 2
  %912 = zext i16 %905 to i32
  %913 = shl nuw i32 %912, 16, !spirv.Decorations !838
  %914 = bitcast i32 %913 to float
  %915 = zext i16 %911 to i32
  %916 = shl nuw i32 %915, 16, !spirv.Decorations !838
  %917 = bitcast i32 %916 to float
  %918 = fmul reassoc nsz arcp contract float %914, %917, !spirv.Decorations !843
  %919 = fadd reassoc nsz arcp contract float %918, %.sroa.226.1, !spirv.Decorations !843
  br label %.preheader.8

.preheader.8:                                     ; preds = %._crit_edge.2.8, %899
  %.sroa.226.2 = phi float [ %919, %899 ], [ %.sroa.226.1, %._crit_edge.2.8 ]
  br i1 %118, label %920, label %._crit_edge.9

920:                                              ; preds = %.preheader.8
  %.sroa.256.0.insert.ext763 = zext i32 %163 to i64
  %921 = sext i32 %29 to i64
  %922 = mul nsw i64 %921, %const_reg_qword3, !spirv.Decorations !836
  %923 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %922, i32 0
  %924 = getelementptr i16, i16 addrspace(4)* %923, i64 %.sroa.256.0.insert.ext763
  %925 = addrspacecast i16 addrspace(4)* %924 to i16 addrspace(1)*
  %926 = load i16, i16 addrspace(1)* %925, align 2
  %927 = mul nsw i64 %.sroa.256.0.insert.ext763, %const_reg_qword5, !spirv.Decorations !836
  %928 = sext i32 %116 to i64
  %929 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %927, i32 0
  %930 = getelementptr i16, i16 addrspace(4)* %929, i64 %928
  %931 = addrspacecast i16 addrspace(4)* %930 to i16 addrspace(1)*
  %932 = load i16, i16 addrspace(1)* %931, align 2
  %933 = zext i16 %926 to i32
  %934 = shl nuw i32 %933, 16, !spirv.Decorations !838
  %935 = bitcast i32 %934 to float
  %936 = zext i16 %932 to i32
  %937 = shl nuw i32 %936, 16, !spirv.Decorations !838
  %938 = bitcast i32 %937 to float
  %939 = fmul reassoc nsz arcp contract float %935, %938, !spirv.Decorations !843
  %940 = fadd reassoc nsz arcp contract float %939, %.sroa.38.1, !spirv.Decorations !843
  br label %._crit_edge.9

._crit_edge.9:                                    ; preds = %.preheader.8, %920
  %.sroa.38.2 = phi float [ %940, %920 ], [ %.sroa.38.1, %.preheader.8 ]
  br i1 %119, label %941, label %._crit_edge.1.9

941:                                              ; preds = %._crit_edge.9
  %.sroa.256.0.insert.ext768 = zext i32 %163 to i64
  %942 = sext i32 %59 to i64
  %943 = mul nsw i64 %942, %const_reg_qword3, !spirv.Decorations !836
  %944 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %943, i32 0
  %945 = getelementptr i16, i16 addrspace(4)* %944, i64 %.sroa.256.0.insert.ext768
  %946 = addrspacecast i16 addrspace(4)* %945 to i16 addrspace(1)*
  %947 = load i16, i16 addrspace(1)* %946, align 2
  %948 = mul nsw i64 %.sroa.256.0.insert.ext768, %const_reg_qword5, !spirv.Decorations !836
  %949 = sext i32 %116 to i64
  %950 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %948, i32 0
  %951 = getelementptr i16, i16 addrspace(4)* %950, i64 %949
  %952 = addrspacecast i16 addrspace(4)* %951 to i16 addrspace(1)*
  %953 = load i16, i16 addrspace(1)* %952, align 2
  %954 = zext i16 %947 to i32
  %955 = shl nuw i32 %954, 16, !spirv.Decorations !838
  %956 = bitcast i32 %955 to float
  %957 = zext i16 %953 to i32
  %958 = shl nuw i32 %957, 16, !spirv.Decorations !838
  %959 = bitcast i32 %958 to float
  %960 = fmul reassoc nsz arcp contract float %956, %959, !spirv.Decorations !843
  %961 = fadd reassoc nsz arcp contract float %960, %.sroa.102.1, !spirv.Decorations !843
  br label %._crit_edge.1.9

._crit_edge.1.9:                                  ; preds = %._crit_edge.9, %941
  %.sroa.102.2 = phi float [ %961, %941 ], [ %.sroa.102.1, %._crit_edge.9 ]
  br i1 %120, label %962, label %._crit_edge.2.9

962:                                              ; preds = %._crit_edge.1.9
  %.sroa.256.0.insert.ext773 = zext i32 %163 to i64
  %963 = sext i32 %62 to i64
  %964 = mul nsw i64 %963, %const_reg_qword3, !spirv.Decorations !836
  %965 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %964, i32 0
  %966 = getelementptr i16, i16 addrspace(4)* %965, i64 %.sroa.256.0.insert.ext773
  %967 = addrspacecast i16 addrspace(4)* %966 to i16 addrspace(1)*
  %968 = load i16, i16 addrspace(1)* %967, align 2
  %969 = mul nsw i64 %.sroa.256.0.insert.ext773, %const_reg_qword5, !spirv.Decorations !836
  %970 = sext i32 %116 to i64
  %971 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %969, i32 0
  %972 = getelementptr i16, i16 addrspace(4)* %971, i64 %970
  %973 = addrspacecast i16 addrspace(4)* %972 to i16 addrspace(1)*
  %974 = load i16, i16 addrspace(1)* %973, align 2
  %975 = zext i16 %968 to i32
  %976 = shl nuw i32 %975, 16, !spirv.Decorations !838
  %977 = bitcast i32 %976 to float
  %978 = zext i16 %974 to i32
  %979 = shl nuw i32 %978, 16, !spirv.Decorations !838
  %980 = bitcast i32 %979 to float
  %981 = fmul reassoc nsz arcp contract float %977, %980, !spirv.Decorations !843
  %982 = fadd reassoc nsz arcp contract float %981, %.sroa.166.1, !spirv.Decorations !843
  br label %._crit_edge.2.9

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9, %962
  %.sroa.166.2 = phi float [ %982, %962 ], [ %.sroa.166.1, %._crit_edge.1.9 ]
  br i1 %121, label %983, label %.preheader.9

983:                                              ; preds = %._crit_edge.2.9
  %.sroa.256.0.insert.ext778 = zext i32 %163 to i64
  %984 = sext i32 %65 to i64
  %985 = mul nsw i64 %984, %const_reg_qword3, !spirv.Decorations !836
  %986 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %985, i32 0
  %987 = getelementptr i16, i16 addrspace(4)* %986, i64 %.sroa.256.0.insert.ext778
  %988 = addrspacecast i16 addrspace(4)* %987 to i16 addrspace(1)*
  %989 = load i16, i16 addrspace(1)* %988, align 2
  %990 = mul nsw i64 %.sroa.256.0.insert.ext778, %const_reg_qword5, !spirv.Decorations !836
  %991 = sext i32 %116 to i64
  %992 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %990, i32 0
  %993 = getelementptr i16, i16 addrspace(4)* %992, i64 %991
  %994 = addrspacecast i16 addrspace(4)* %993 to i16 addrspace(1)*
  %995 = load i16, i16 addrspace(1)* %994, align 2
  %996 = zext i16 %989 to i32
  %997 = shl nuw i32 %996, 16, !spirv.Decorations !838
  %998 = bitcast i32 %997 to float
  %999 = zext i16 %995 to i32
  %1000 = shl nuw i32 %999, 16, !spirv.Decorations !838
  %1001 = bitcast i32 %1000 to float
  %1002 = fmul reassoc nsz arcp contract float %998, %1001, !spirv.Decorations !843
  %1003 = fadd reassoc nsz arcp contract float %1002, %.sroa.230.1, !spirv.Decorations !843
  br label %.preheader.9

.preheader.9:                                     ; preds = %._crit_edge.2.9, %983
  %.sroa.230.2 = phi float [ %1003, %983 ], [ %.sroa.230.1, %._crit_edge.2.9 ]
  br i1 %124, label %1004, label %._crit_edge.10

1004:                                             ; preds = %.preheader.9
  %.sroa.256.0.insert.ext783 = zext i32 %163 to i64
  %1005 = sext i32 %29 to i64
  %1006 = mul nsw i64 %1005, %const_reg_qword3, !spirv.Decorations !836
  %1007 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1006, i32 0
  %1008 = getelementptr i16, i16 addrspace(4)* %1007, i64 %.sroa.256.0.insert.ext783
  %1009 = addrspacecast i16 addrspace(4)* %1008 to i16 addrspace(1)*
  %1010 = load i16, i16 addrspace(1)* %1009, align 2
  %1011 = mul nsw i64 %.sroa.256.0.insert.ext783, %const_reg_qword5, !spirv.Decorations !836
  %1012 = sext i32 %122 to i64
  %1013 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1011, i32 0
  %1014 = getelementptr i16, i16 addrspace(4)* %1013, i64 %1012
  %1015 = addrspacecast i16 addrspace(4)* %1014 to i16 addrspace(1)*
  %1016 = load i16, i16 addrspace(1)* %1015, align 2
  %1017 = zext i16 %1010 to i32
  %1018 = shl nuw i32 %1017, 16, !spirv.Decorations !838
  %1019 = bitcast i32 %1018 to float
  %1020 = zext i16 %1016 to i32
  %1021 = shl nuw i32 %1020, 16, !spirv.Decorations !838
  %1022 = bitcast i32 %1021 to float
  %1023 = fmul reassoc nsz arcp contract float %1019, %1022, !spirv.Decorations !843
  %1024 = fadd reassoc nsz arcp contract float %1023, %.sroa.42.1, !spirv.Decorations !843
  br label %._crit_edge.10

._crit_edge.10:                                   ; preds = %.preheader.9, %1004
  %.sroa.42.2 = phi float [ %1024, %1004 ], [ %.sroa.42.1, %.preheader.9 ]
  br i1 %125, label %1025, label %._crit_edge.1.10

1025:                                             ; preds = %._crit_edge.10
  %.sroa.256.0.insert.ext788 = zext i32 %163 to i64
  %1026 = sext i32 %59 to i64
  %1027 = mul nsw i64 %1026, %const_reg_qword3, !spirv.Decorations !836
  %1028 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1027, i32 0
  %1029 = getelementptr i16, i16 addrspace(4)* %1028, i64 %.sroa.256.0.insert.ext788
  %1030 = addrspacecast i16 addrspace(4)* %1029 to i16 addrspace(1)*
  %1031 = load i16, i16 addrspace(1)* %1030, align 2
  %1032 = mul nsw i64 %.sroa.256.0.insert.ext788, %const_reg_qword5, !spirv.Decorations !836
  %1033 = sext i32 %122 to i64
  %1034 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1032, i32 0
  %1035 = getelementptr i16, i16 addrspace(4)* %1034, i64 %1033
  %1036 = addrspacecast i16 addrspace(4)* %1035 to i16 addrspace(1)*
  %1037 = load i16, i16 addrspace(1)* %1036, align 2
  %1038 = zext i16 %1031 to i32
  %1039 = shl nuw i32 %1038, 16, !spirv.Decorations !838
  %1040 = bitcast i32 %1039 to float
  %1041 = zext i16 %1037 to i32
  %1042 = shl nuw i32 %1041, 16, !spirv.Decorations !838
  %1043 = bitcast i32 %1042 to float
  %1044 = fmul reassoc nsz arcp contract float %1040, %1043, !spirv.Decorations !843
  %1045 = fadd reassoc nsz arcp contract float %1044, %.sroa.106.1, !spirv.Decorations !843
  br label %._crit_edge.1.10

._crit_edge.1.10:                                 ; preds = %._crit_edge.10, %1025
  %.sroa.106.2 = phi float [ %1045, %1025 ], [ %.sroa.106.1, %._crit_edge.10 ]
  br i1 %126, label %1046, label %._crit_edge.2.10

1046:                                             ; preds = %._crit_edge.1.10
  %.sroa.256.0.insert.ext793 = zext i32 %163 to i64
  %1047 = sext i32 %62 to i64
  %1048 = mul nsw i64 %1047, %const_reg_qword3, !spirv.Decorations !836
  %1049 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1048, i32 0
  %1050 = getelementptr i16, i16 addrspace(4)* %1049, i64 %.sroa.256.0.insert.ext793
  %1051 = addrspacecast i16 addrspace(4)* %1050 to i16 addrspace(1)*
  %1052 = load i16, i16 addrspace(1)* %1051, align 2
  %1053 = mul nsw i64 %.sroa.256.0.insert.ext793, %const_reg_qword5, !spirv.Decorations !836
  %1054 = sext i32 %122 to i64
  %1055 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1053, i32 0
  %1056 = getelementptr i16, i16 addrspace(4)* %1055, i64 %1054
  %1057 = addrspacecast i16 addrspace(4)* %1056 to i16 addrspace(1)*
  %1058 = load i16, i16 addrspace(1)* %1057, align 2
  %1059 = zext i16 %1052 to i32
  %1060 = shl nuw i32 %1059, 16, !spirv.Decorations !838
  %1061 = bitcast i32 %1060 to float
  %1062 = zext i16 %1058 to i32
  %1063 = shl nuw i32 %1062, 16, !spirv.Decorations !838
  %1064 = bitcast i32 %1063 to float
  %1065 = fmul reassoc nsz arcp contract float %1061, %1064, !spirv.Decorations !843
  %1066 = fadd reassoc nsz arcp contract float %1065, %.sroa.170.1, !spirv.Decorations !843
  br label %._crit_edge.2.10

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10, %1046
  %.sroa.170.2 = phi float [ %1066, %1046 ], [ %.sroa.170.1, %._crit_edge.1.10 ]
  br i1 %127, label %1067, label %.preheader.10

1067:                                             ; preds = %._crit_edge.2.10
  %.sroa.256.0.insert.ext798 = zext i32 %163 to i64
  %1068 = sext i32 %65 to i64
  %1069 = mul nsw i64 %1068, %const_reg_qword3, !spirv.Decorations !836
  %1070 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1069, i32 0
  %1071 = getelementptr i16, i16 addrspace(4)* %1070, i64 %.sroa.256.0.insert.ext798
  %1072 = addrspacecast i16 addrspace(4)* %1071 to i16 addrspace(1)*
  %1073 = load i16, i16 addrspace(1)* %1072, align 2
  %1074 = mul nsw i64 %.sroa.256.0.insert.ext798, %const_reg_qword5, !spirv.Decorations !836
  %1075 = sext i32 %122 to i64
  %1076 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1074, i32 0
  %1077 = getelementptr i16, i16 addrspace(4)* %1076, i64 %1075
  %1078 = addrspacecast i16 addrspace(4)* %1077 to i16 addrspace(1)*
  %1079 = load i16, i16 addrspace(1)* %1078, align 2
  %1080 = zext i16 %1073 to i32
  %1081 = shl nuw i32 %1080, 16, !spirv.Decorations !838
  %1082 = bitcast i32 %1081 to float
  %1083 = zext i16 %1079 to i32
  %1084 = shl nuw i32 %1083, 16, !spirv.Decorations !838
  %1085 = bitcast i32 %1084 to float
  %1086 = fmul reassoc nsz arcp contract float %1082, %1085, !spirv.Decorations !843
  %1087 = fadd reassoc nsz arcp contract float %1086, %.sroa.234.1, !spirv.Decorations !843
  br label %.preheader.10

.preheader.10:                                    ; preds = %._crit_edge.2.10, %1067
  %.sroa.234.2 = phi float [ %1087, %1067 ], [ %.sroa.234.1, %._crit_edge.2.10 ]
  br i1 %130, label %1088, label %._crit_edge.11

1088:                                             ; preds = %.preheader.10
  %.sroa.256.0.insert.ext803 = zext i32 %163 to i64
  %1089 = sext i32 %29 to i64
  %1090 = mul nsw i64 %1089, %const_reg_qword3, !spirv.Decorations !836
  %1091 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1090, i32 0
  %1092 = getelementptr i16, i16 addrspace(4)* %1091, i64 %.sroa.256.0.insert.ext803
  %1093 = addrspacecast i16 addrspace(4)* %1092 to i16 addrspace(1)*
  %1094 = load i16, i16 addrspace(1)* %1093, align 2
  %1095 = mul nsw i64 %.sroa.256.0.insert.ext803, %const_reg_qword5, !spirv.Decorations !836
  %1096 = sext i32 %128 to i64
  %1097 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1095, i32 0
  %1098 = getelementptr i16, i16 addrspace(4)* %1097, i64 %1096
  %1099 = addrspacecast i16 addrspace(4)* %1098 to i16 addrspace(1)*
  %1100 = load i16, i16 addrspace(1)* %1099, align 2
  %1101 = zext i16 %1094 to i32
  %1102 = shl nuw i32 %1101, 16, !spirv.Decorations !838
  %1103 = bitcast i32 %1102 to float
  %1104 = zext i16 %1100 to i32
  %1105 = shl nuw i32 %1104, 16, !spirv.Decorations !838
  %1106 = bitcast i32 %1105 to float
  %1107 = fmul reassoc nsz arcp contract float %1103, %1106, !spirv.Decorations !843
  %1108 = fadd reassoc nsz arcp contract float %1107, %.sroa.46.1, !spirv.Decorations !843
  br label %._crit_edge.11

._crit_edge.11:                                   ; preds = %.preheader.10, %1088
  %.sroa.46.2 = phi float [ %1108, %1088 ], [ %.sroa.46.1, %.preheader.10 ]
  br i1 %131, label %1109, label %._crit_edge.1.11

1109:                                             ; preds = %._crit_edge.11
  %.sroa.256.0.insert.ext808 = zext i32 %163 to i64
  %1110 = sext i32 %59 to i64
  %1111 = mul nsw i64 %1110, %const_reg_qword3, !spirv.Decorations !836
  %1112 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1111, i32 0
  %1113 = getelementptr i16, i16 addrspace(4)* %1112, i64 %.sroa.256.0.insert.ext808
  %1114 = addrspacecast i16 addrspace(4)* %1113 to i16 addrspace(1)*
  %1115 = load i16, i16 addrspace(1)* %1114, align 2
  %1116 = mul nsw i64 %.sroa.256.0.insert.ext808, %const_reg_qword5, !spirv.Decorations !836
  %1117 = sext i32 %128 to i64
  %1118 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1116, i32 0
  %1119 = getelementptr i16, i16 addrspace(4)* %1118, i64 %1117
  %1120 = addrspacecast i16 addrspace(4)* %1119 to i16 addrspace(1)*
  %1121 = load i16, i16 addrspace(1)* %1120, align 2
  %1122 = zext i16 %1115 to i32
  %1123 = shl nuw i32 %1122, 16, !spirv.Decorations !838
  %1124 = bitcast i32 %1123 to float
  %1125 = zext i16 %1121 to i32
  %1126 = shl nuw i32 %1125, 16, !spirv.Decorations !838
  %1127 = bitcast i32 %1126 to float
  %1128 = fmul reassoc nsz arcp contract float %1124, %1127, !spirv.Decorations !843
  %1129 = fadd reassoc nsz arcp contract float %1128, %.sroa.110.1, !spirv.Decorations !843
  br label %._crit_edge.1.11

._crit_edge.1.11:                                 ; preds = %._crit_edge.11, %1109
  %.sroa.110.2 = phi float [ %1129, %1109 ], [ %.sroa.110.1, %._crit_edge.11 ]
  br i1 %132, label %1130, label %._crit_edge.2.11

1130:                                             ; preds = %._crit_edge.1.11
  %.sroa.256.0.insert.ext813 = zext i32 %163 to i64
  %1131 = sext i32 %62 to i64
  %1132 = mul nsw i64 %1131, %const_reg_qword3, !spirv.Decorations !836
  %1133 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1132, i32 0
  %1134 = getelementptr i16, i16 addrspace(4)* %1133, i64 %.sroa.256.0.insert.ext813
  %1135 = addrspacecast i16 addrspace(4)* %1134 to i16 addrspace(1)*
  %1136 = load i16, i16 addrspace(1)* %1135, align 2
  %1137 = mul nsw i64 %.sroa.256.0.insert.ext813, %const_reg_qword5, !spirv.Decorations !836
  %1138 = sext i32 %128 to i64
  %1139 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1137, i32 0
  %1140 = getelementptr i16, i16 addrspace(4)* %1139, i64 %1138
  %1141 = addrspacecast i16 addrspace(4)* %1140 to i16 addrspace(1)*
  %1142 = load i16, i16 addrspace(1)* %1141, align 2
  %1143 = zext i16 %1136 to i32
  %1144 = shl nuw i32 %1143, 16, !spirv.Decorations !838
  %1145 = bitcast i32 %1144 to float
  %1146 = zext i16 %1142 to i32
  %1147 = shl nuw i32 %1146, 16, !spirv.Decorations !838
  %1148 = bitcast i32 %1147 to float
  %1149 = fmul reassoc nsz arcp contract float %1145, %1148, !spirv.Decorations !843
  %1150 = fadd reassoc nsz arcp contract float %1149, %.sroa.174.1, !spirv.Decorations !843
  br label %._crit_edge.2.11

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11, %1130
  %.sroa.174.2 = phi float [ %1150, %1130 ], [ %.sroa.174.1, %._crit_edge.1.11 ]
  br i1 %133, label %1151, label %.preheader.11

1151:                                             ; preds = %._crit_edge.2.11
  %.sroa.256.0.insert.ext818 = zext i32 %163 to i64
  %1152 = sext i32 %65 to i64
  %1153 = mul nsw i64 %1152, %const_reg_qword3, !spirv.Decorations !836
  %1154 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1153, i32 0
  %1155 = getelementptr i16, i16 addrspace(4)* %1154, i64 %.sroa.256.0.insert.ext818
  %1156 = addrspacecast i16 addrspace(4)* %1155 to i16 addrspace(1)*
  %1157 = load i16, i16 addrspace(1)* %1156, align 2
  %1158 = mul nsw i64 %.sroa.256.0.insert.ext818, %const_reg_qword5, !spirv.Decorations !836
  %1159 = sext i32 %128 to i64
  %1160 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1158, i32 0
  %1161 = getelementptr i16, i16 addrspace(4)* %1160, i64 %1159
  %1162 = addrspacecast i16 addrspace(4)* %1161 to i16 addrspace(1)*
  %1163 = load i16, i16 addrspace(1)* %1162, align 2
  %1164 = zext i16 %1157 to i32
  %1165 = shl nuw i32 %1164, 16, !spirv.Decorations !838
  %1166 = bitcast i32 %1165 to float
  %1167 = zext i16 %1163 to i32
  %1168 = shl nuw i32 %1167, 16, !spirv.Decorations !838
  %1169 = bitcast i32 %1168 to float
  %1170 = fmul reassoc nsz arcp contract float %1166, %1169, !spirv.Decorations !843
  %1171 = fadd reassoc nsz arcp contract float %1170, %.sroa.238.1, !spirv.Decorations !843
  br label %.preheader.11

.preheader.11:                                    ; preds = %._crit_edge.2.11, %1151
  %.sroa.238.2 = phi float [ %1171, %1151 ], [ %.sroa.238.1, %._crit_edge.2.11 ]
  br i1 %136, label %1172, label %._crit_edge.12

1172:                                             ; preds = %.preheader.11
  %.sroa.256.0.insert.ext823 = zext i32 %163 to i64
  %1173 = sext i32 %29 to i64
  %1174 = mul nsw i64 %1173, %const_reg_qword3, !spirv.Decorations !836
  %1175 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1174, i32 0
  %1176 = getelementptr i16, i16 addrspace(4)* %1175, i64 %.sroa.256.0.insert.ext823
  %1177 = addrspacecast i16 addrspace(4)* %1176 to i16 addrspace(1)*
  %1178 = load i16, i16 addrspace(1)* %1177, align 2
  %1179 = mul nsw i64 %.sroa.256.0.insert.ext823, %const_reg_qword5, !spirv.Decorations !836
  %1180 = sext i32 %134 to i64
  %1181 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1179, i32 0
  %1182 = getelementptr i16, i16 addrspace(4)* %1181, i64 %1180
  %1183 = addrspacecast i16 addrspace(4)* %1182 to i16 addrspace(1)*
  %1184 = load i16, i16 addrspace(1)* %1183, align 2
  %1185 = zext i16 %1178 to i32
  %1186 = shl nuw i32 %1185, 16, !spirv.Decorations !838
  %1187 = bitcast i32 %1186 to float
  %1188 = zext i16 %1184 to i32
  %1189 = shl nuw i32 %1188, 16, !spirv.Decorations !838
  %1190 = bitcast i32 %1189 to float
  %1191 = fmul reassoc nsz arcp contract float %1187, %1190, !spirv.Decorations !843
  %1192 = fadd reassoc nsz arcp contract float %1191, %.sroa.50.1, !spirv.Decorations !843
  br label %._crit_edge.12

._crit_edge.12:                                   ; preds = %.preheader.11, %1172
  %.sroa.50.2 = phi float [ %1192, %1172 ], [ %.sroa.50.1, %.preheader.11 ]
  br i1 %137, label %1193, label %._crit_edge.1.12

1193:                                             ; preds = %._crit_edge.12
  %.sroa.256.0.insert.ext828 = zext i32 %163 to i64
  %1194 = sext i32 %59 to i64
  %1195 = mul nsw i64 %1194, %const_reg_qword3, !spirv.Decorations !836
  %1196 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1195, i32 0
  %1197 = getelementptr i16, i16 addrspace(4)* %1196, i64 %.sroa.256.0.insert.ext828
  %1198 = addrspacecast i16 addrspace(4)* %1197 to i16 addrspace(1)*
  %1199 = load i16, i16 addrspace(1)* %1198, align 2
  %1200 = mul nsw i64 %.sroa.256.0.insert.ext828, %const_reg_qword5, !spirv.Decorations !836
  %1201 = sext i32 %134 to i64
  %1202 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1200, i32 0
  %1203 = getelementptr i16, i16 addrspace(4)* %1202, i64 %1201
  %1204 = addrspacecast i16 addrspace(4)* %1203 to i16 addrspace(1)*
  %1205 = load i16, i16 addrspace(1)* %1204, align 2
  %1206 = zext i16 %1199 to i32
  %1207 = shl nuw i32 %1206, 16, !spirv.Decorations !838
  %1208 = bitcast i32 %1207 to float
  %1209 = zext i16 %1205 to i32
  %1210 = shl nuw i32 %1209, 16, !spirv.Decorations !838
  %1211 = bitcast i32 %1210 to float
  %1212 = fmul reassoc nsz arcp contract float %1208, %1211, !spirv.Decorations !843
  %1213 = fadd reassoc nsz arcp contract float %1212, %.sroa.114.1, !spirv.Decorations !843
  br label %._crit_edge.1.12

._crit_edge.1.12:                                 ; preds = %._crit_edge.12, %1193
  %.sroa.114.2 = phi float [ %1213, %1193 ], [ %.sroa.114.1, %._crit_edge.12 ]
  br i1 %138, label %1214, label %._crit_edge.2.12

1214:                                             ; preds = %._crit_edge.1.12
  %.sroa.256.0.insert.ext833 = zext i32 %163 to i64
  %1215 = sext i32 %62 to i64
  %1216 = mul nsw i64 %1215, %const_reg_qword3, !spirv.Decorations !836
  %1217 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1216, i32 0
  %1218 = getelementptr i16, i16 addrspace(4)* %1217, i64 %.sroa.256.0.insert.ext833
  %1219 = addrspacecast i16 addrspace(4)* %1218 to i16 addrspace(1)*
  %1220 = load i16, i16 addrspace(1)* %1219, align 2
  %1221 = mul nsw i64 %.sroa.256.0.insert.ext833, %const_reg_qword5, !spirv.Decorations !836
  %1222 = sext i32 %134 to i64
  %1223 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1221, i32 0
  %1224 = getelementptr i16, i16 addrspace(4)* %1223, i64 %1222
  %1225 = addrspacecast i16 addrspace(4)* %1224 to i16 addrspace(1)*
  %1226 = load i16, i16 addrspace(1)* %1225, align 2
  %1227 = zext i16 %1220 to i32
  %1228 = shl nuw i32 %1227, 16, !spirv.Decorations !838
  %1229 = bitcast i32 %1228 to float
  %1230 = zext i16 %1226 to i32
  %1231 = shl nuw i32 %1230, 16, !spirv.Decorations !838
  %1232 = bitcast i32 %1231 to float
  %1233 = fmul reassoc nsz arcp contract float %1229, %1232, !spirv.Decorations !843
  %1234 = fadd reassoc nsz arcp contract float %1233, %.sroa.178.1, !spirv.Decorations !843
  br label %._crit_edge.2.12

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12, %1214
  %.sroa.178.2 = phi float [ %1234, %1214 ], [ %.sroa.178.1, %._crit_edge.1.12 ]
  br i1 %139, label %1235, label %.preheader.12

1235:                                             ; preds = %._crit_edge.2.12
  %.sroa.256.0.insert.ext838 = zext i32 %163 to i64
  %1236 = sext i32 %65 to i64
  %1237 = mul nsw i64 %1236, %const_reg_qword3, !spirv.Decorations !836
  %1238 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1237, i32 0
  %1239 = getelementptr i16, i16 addrspace(4)* %1238, i64 %.sroa.256.0.insert.ext838
  %1240 = addrspacecast i16 addrspace(4)* %1239 to i16 addrspace(1)*
  %1241 = load i16, i16 addrspace(1)* %1240, align 2
  %1242 = mul nsw i64 %.sroa.256.0.insert.ext838, %const_reg_qword5, !spirv.Decorations !836
  %1243 = sext i32 %134 to i64
  %1244 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1242, i32 0
  %1245 = getelementptr i16, i16 addrspace(4)* %1244, i64 %1243
  %1246 = addrspacecast i16 addrspace(4)* %1245 to i16 addrspace(1)*
  %1247 = load i16, i16 addrspace(1)* %1246, align 2
  %1248 = zext i16 %1241 to i32
  %1249 = shl nuw i32 %1248, 16, !spirv.Decorations !838
  %1250 = bitcast i32 %1249 to float
  %1251 = zext i16 %1247 to i32
  %1252 = shl nuw i32 %1251, 16, !spirv.Decorations !838
  %1253 = bitcast i32 %1252 to float
  %1254 = fmul reassoc nsz arcp contract float %1250, %1253, !spirv.Decorations !843
  %1255 = fadd reassoc nsz arcp contract float %1254, %.sroa.242.1, !spirv.Decorations !843
  br label %.preheader.12

.preheader.12:                                    ; preds = %._crit_edge.2.12, %1235
  %.sroa.242.2 = phi float [ %1255, %1235 ], [ %.sroa.242.1, %._crit_edge.2.12 ]
  br i1 %142, label %1256, label %._crit_edge.13

1256:                                             ; preds = %.preheader.12
  %.sroa.256.0.insert.ext843 = zext i32 %163 to i64
  %1257 = sext i32 %29 to i64
  %1258 = mul nsw i64 %1257, %const_reg_qword3, !spirv.Decorations !836
  %1259 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1258, i32 0
  %1260 = getelementptr i16, i16 addrspace(4)* %1259, i64 %.sroa.256.0.insert.ext843
  %1261 = addrspacecast i16 addrspace(4)* %1260 to i16 addrspace(1)*
  %1262 = load i16, i16 addrspace(1)* %1261, align 2
  %1263 = mul nsw i64 %.sroa.256.0.insert.ext843, %const_reg_qword5, !spirv.Decorations !836
  %1264 = sext i32 %140 to i64
  %1265 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1263, i32 0
  %1266 = getelementptr i16, i16 addrspace(4)* %1265, i64 %1264
  %1267 = addrspacecast i16 addrspace(4)* %1266 to i16 addrspace(1)*
  %1268 = load i16, i16 addrspace(1)* %1267, align 2
  %1269 = zext i16 %1262 to i32
  %1270 = shl nuw i32 %1269, 16, !spirv.Decorations !838
  %1271 = bitcast i32 %1270 to float
  %1272 = zext i16 %1268 to i32
  %1273 = shl nuw i32 %1272, 16, !spirv.Decorations !838
  %1274 = bitcast i32 %1273 to float
  %1275 = fmul reassoc nsz arcp contract float %1271, %1274, !spirv.Decorations !843
  %1276 = fadd reassoc nsz arcp contract float %1275, %.sroa.54.1, !spirv.Decorations !843
  br label %._crit_edge.13

._crit_edge.13:                                   ; preds = %.preheader.12, %1256
  %.sroa.54.2 = phi float [ %1276, %1256 ], [ %.sroa.54.1, %.preheader.12 ]
  br i1 %143, label %1277, label %._crit_edge.1.13

1277:                                             ; preds = %._crit_edge.13
  %.sroa.256.0.insert.ext848 = zext i32 %163 to i64
  %1278 = sext i32 %59 to i64
  %1279 = mul nsw i64 %1278, %const_reg_qword3, !spirv.Decorations !836
  %1280 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1279, i32 0
  %1281 = getelementptr i16, i16 addrspace(4)* %1280, i64 %.sroa.256.0.insert.ext848
  %1282 = addrspacecast i16 addrspace(4)* %1281 to i16 addrspace(1)*
  %1283 = load i16, i16 addrspace(1)* %1282, align 2
  %1284 = mul nsw i64 %.sroa.256.0.insert.ext848, %const_reg_qword5, !spirv.Decorations !836
  %1285 = sext i32 %140 to i64
  %1286 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1284, i32 0
  %1287 = getelementptr i16, i16 addrspace(4)* %1286, i64 %1285
  %1288 = addrspacecast i16 addrspace(4)* %1287 to i16 addrspace(1)*
  %1289 = load i16, i16 addrspace(1)* %1288, align 2
  %1290 = zext i16 %1283 to i32
  %1291 = shl nuw i32 %1290, 16, !spirv.Decorations !838
  %1292 = bitcast i32 %1291 to float
  %1293 = zext i16 %1289 to i32
  %1294 = shl nuw i32 %1293, 16, !spirv.Decorations !838
  %1295 = bitcast i32 %1294 to float
  %1296 = fmul reassoc nsz arcp contract float %1292, %1295, !spirv.Decorations !843
  %1297 = fadd reassoc nsz arcp contract float %1296, %.sroa.118.1, !spirv.Decorations !843
  br label %._crit_edge.1.13

._crit_edge.1.13:                                 ; preds = %._crit_edge.13, %1277
  %.sroa.118.2 = phi float [ %1297, %1277 ], [ %.sroa.118.1, %._crit_edge.13 ]
  br i1 %144, label %1298, label %._crit_edge.2.13

1298:                                             ; preds = %._crit_edge.1.13
  %.sroa.256.0.insert.ext853 = zext i32 %163 to i64
  %1299 = sext i32 %62 to i64
  %1300 = mul nsw i64 %1299, %const_reg_qword3, !spirv.Decorations !836
  %1301 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1300, i32 0
  %1302 = getelementptr i16, i16 addrspace(4)* %1301, i64 %.sroa.256.0.insert.ext853
  %1303 = addrspacecast i16 addrspace(4)* %1302 to i16 addrspace(1)*
  %1304 = load i16, i16 addrspace(1)* %1303, align 2
  %1305 = mul nsw i64 %.sroa.256.0.insert.ext853, %const_reg_qword5, !spirv.Decorations !836
  %1306 = sext i32 %140 to i64
  %1307 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1305, i32 0
  %1308 = getelementptr i16, i16 addrspace(4)* %1307, i64 %1306
  %1309 = addrspacecast i16 addrspace(4)* %1308 to i16 addrspace(1)*
  %1310 = load i16, i16 addrspace(1)* %1309, align 2
  %1311 = zext i16 %1304 to i32
  %1312 = shl nuw i32 %1311, 16, !spirv.Decorations !838
  %1313 = bitcast i32 %1312 to float
  %1314 = zext i16 %1310 to i32
  %1315 = shl nuw i32 %1314, 16, !spirv.Decorations !838
  %1316 = bitcast i32 %1315 to float
  %1317 = fmul reassoc nsz arcp contract float %1313, %1316, !spirv.Decorations !843
  %1318 = fadd reassoc nsz arcp contract float %1317, %.sroa.182.1, !spirv.Decorations !843
  br label %._crit_edge.2.13

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13, %1298
  %.sroa.182.2 = phi float [ %1318, %1298 ], [ %.sroa.182.1, %._crit_edge.1.13 ]
  br i1 %145, label %1319, label %.preheader.13

1319:                                             ; preds = %._crit_edge.2.13
  %.sroa.256.0.insert.ext858 = zext i32 %163 to i64
  %1320 = sext i32 %65 to i64
  %1321 = mul nsw i64 %1320, %const_reg_qword3, !spirv.Decorations !836
  %1322 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1321, i32 0
  %1323 = getelementptr i16, i16 addrspace(4)* %1322, i64 %.sroa.256.0.insert.ext858
  %1324 = addrspacecast i16 addrspace(4)* %1323 to i16 addrspace(1)*
  %1325 = load i16, i16 addrspace(1)* %1324, align 2
  %1326 = mul nsw i64 %.sroa.256.0.insert.ext858, %const_reg_qword5, !spirv.Decorations !836
  %1327 = sext i32 %140 to i64
  %1328 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1326, i32 0
  %1329 = getelementptr i16, i16 addrspace(4)* %1328, i64 %1327
  %1330 = addrspacecast i16 addrspace(4)* %1329 to i16 addrspace(1)*
  %1331 = load i16, i16 addrspace(1)* %1330, align 2
  %1332 = zext i16 %1325 to i32
  %1333 = shl nuw i32 %1332, 16, !spirv.Decorations !838
  %1334 = bitcast i32 %1333 to float
  %1335 = zext i16 %1331 to i32
  %1336 = shl nuw i32 %1335, 16, !spirv.Decorations !838
  %1337 = bitcast i32 %1336 to float
  %1338 = fmul reassoc nsz arcp contract float %1334, %1337, !spirv.Decorations !843
  %1339 = fadd reassoc nsz arcp contract float %1338, %.sroa.246.1, !spirv.Decorations !843
  br label %.preheader.13

.preheader.13:                                    ; preds = %._crit_edge.2.13, %1319
  %.sroa.246.2 = phi float [ %1339, %1319 ], [ %.sroa.246.1, %._crit_edge.2.13 ]
  br i1 %148, label %1340, label %._crit_edge.14

1340:                                             ; preds = %.preheader.13
  %.sroa.256.0.insert.ext863 = zext i32 %163 to i64
  %1341 = sext i32 %29 to i64
  %1342 = mul nsw i64 %1341, %const_reg_qword3, !spirv.Decorations !836
  %1343 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1342, i32 0
  %1344 = getelementptr i16, i16 addrspace(4)* %1343, i64 %.sroa.256.0.insert.ext863
  %1345 = addrspacecast i16 addrspace(4)* %1344 to i16 addrspace(1)*
  %1346 = load i16, i16 addrspace(1)* %1345, align 2
  %1347 = mul nsw i64 %.sroa.256.0.insert.ext863, %const_reg_qword5, !spirv.Decorations !836
  %1348 = sext i32 %146 to i64
  %1349 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1347, i32 0
  %1350 = getelementptr i16, i16 addrspace(4)* %1349, i64 %1348
  %1351 = addrspacecast i16 addrspace(4)* %1350 to i16 addrspace(1)*
  %1352 = load i16, i16 addrspace(1)* %1351, align 2
  %1353 = zext i16 %1346 to i32
  %1354 = shl nuw i32 %1353, 16, !spirv.Decorations !838
  %1355 = bitcast i32 %1354 to float
  %1356 = zext i16 %1352 to i32
  %1357 = shl nuw i32 %1356, 16, !spirv.Decorations !838
  %1358 = bitcast i32 %1357 to float
  %1359 = fmul reassoc nsz arcp contract float %1355, %1358, !spirv.Decorations !843
  %1360 = fadd reassoc nsz arcp contract float %1359, %.sroa.58.1, !spirv.Decorations !843
  br label %._crit_edge.14

._crit_edge.14:                                   ; preds = %.preheader.13, %1340
  %.sroa.58.2 = phi float [ %1360, %1340 ], [ %.sroa.58.1, %.preheader.13 ]
  br i1 %149, label %1361, label %._crit_edge.1.14

1361:                                             ; preds = %._crit_edge.14
  %.sroa.256.0.insert.ext868 = zext i32 %163 to i64
  %1362 = sext i32 %59 to i64
  %1363 = mul nsw i64 %1362, %const_reg_qword3, !spirv.Decorations !836
  %1364 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1363, i32 0
  %1365 = getelementptr i16, i16 addrspace(4)* %1364, i64 %.sroa.256.0.insert.ext868
  %1366 = addrspacecast i16 addrspace(4)* %1365 to i16 addrspace(1)*
  %1367 = load i16, i16 addrspace(1)* %1366, align 2
  %1368 = mul nsw i64 %.sroa.256.0.insert.ext868, %const_reg_qword5, !spirv.Decorations !836
  %1369 = sext i32 %146 to i64
  %1370 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1368, i32 0
  %1371 = getelementptr i16, i16 addrspace(4)* %1370, i64 %1369
  %1372 = addrspacecast i16 addrspace(4)* %1371 to i16 addrspace(1)*
  %1373 = load i16, i16 addrspace(1)* %1372, align 2
  %1374 = zext i16 %1367 to i32
  %1375 = shl nuw i32 %1374, 16, !spirv.Decorations !838
  %1376 = bitcast i32 %1375 to float
  %1377 = zext i16 %1373 to i32
  %1378 = shl nuw i32 %1377, 16, !spirv.Decorations !838
  %1379 = bitcast i32 %1378 to float
  %1380 = fmul reassoc nsz arcp contract float %1376, %1379, !spirv.Decorations !843
  %1381 = fadd reassoc nsz arcp contract float %1380, %.sroa.122.1, !spirv.Decorations !843
  br label %._crit_edge.1.14

._crit_edge.1.14:                                 ; preds = %._crit_edge.14, %1361
  %.sroa.122.2 = phi float [ %1381, %1361 ], [ %.sroa.122.1, %._crit_edge.14 ]
  br i1 %150, label %1382, label %._crit_edge.2.14

1382:                                             ; preds = %._crit_edge.1.14
  %.sroa.256.0.insert.ext873 = zext i32 %163 to i64
  %1383 = sext i32 %62 to i64
  %1384 = mul nsw i64 %1383, %const_reg_qword3, !spirv.Decorations !836
  %1385 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1384, i32 0
  %1386 = getelementptr i16, i16 addrspace(4)* %1385, i64 %.sroa.256.0.insert.ext873
  %1387 = addrspacecast i16 addrspace(4)* %1386 to i16 addrspace(1)*
  %1388 = load i16, i16 addrspace(1)* %1387, align 2
  %1389 = mul nsw i64 %.sroa.256.0.insert.ext873, %const_reg_qword5, !spirv.Decorations !836
  %1390 = sext i32 %146 to i64
  %1391 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1389, i32 0
  %1392 = getelementptr i16, i16 addrspace(4)* %1391, i64 %1390
  %1393 = addrspacecast i16 addrspace(4)* %1392 to i16 addrspace(1)*
  %1394 = load i16, i16 addrspace(1)* %1393, align 2
  %1395 = zext i16 %1388 to i32
  %1396 = shl nuw i32 %1395, 16, !spirv.Decorations !838
  %1397 = bitcast i32 %1396 to float
  %1398 = zext i16 %1394 to i32
  %1399 = shl nuw i32 %1398, 16, !spirv.Decorations !838
  %1400 = bitcast i32 %1399 to float
  %1401 = fmul reassoc nsz arcp contract float %1397, %1400, !spirv.Decorations !843
  %1402 = fadd reassoc nsz arcp contract float %1401, %.sroa.186.1, !spirv.Decorations !843
  br label %._crit_edge.2.14

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14, %1382
  %.sroa.186.2 = phi float [ %1402, %1382 ], [ %.sroa.186.1, %._crit_edge.1.14 ]
  br i1 %151, label %1403, label %.preheader.14

1403:                                             ; preds = %._crit_edge.2.14
  %.sroa.256.0.insert.ext878 = zext i32 %163 to i64
  %1404 = sext i32 %65 to i64
  %1405 = mul nsw i64 %1404, %const_reg_qword3, !spirv.Decorations !836
  %1406 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1405, i32 0
  %1407 = getelementptr i16, i16 addrspace(4)* %1406, i64 %.sroa.256.0.insert.ext878
  %1408 = addrspacecast i16 addrspace(4)* %1407 to i16 addrspace(1)*
  %1409 = load i16, i16 addrspace(1)* %1408, align 2
  %1410 = mul nsw i64 %.sroa.256.0.insert.ext878, %const_reg_qword5, !spirv.Decorations !836
  %1411 = sext i32 %146 to i64
  %1412 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1410, i32 0
  %1413 = getelementptr i16, i16 addrspace(4)* %1412, i64 %1411
  %1414 = addrspacecast i16 addrspace(4)* %1413 to i16 addrspace(1)*
  %1415 = load i16, i16 addrspace(1)* %1414, align 2
  %1416 = zext i16 %1409 to i32
  %1417 = shl nuw i32 %1416, 16, !spirv.Decorations !838
  %1418 = bitcast i32 %1417 to float
  %1419 = zext i16 %1415 to i32
  %1420 = shl nuw i32 %1419, 16, !spirv.Decorations !838
  %1421 = bitcast i32 %1420 to float
  %1422 = fmul reassoc nsz arcp contract float %1418, %1421, !spirv.Decorations !843
  %1423 = fadd reassoc nsz arcp contract float %1422, %.sroa.250.1, !spirv.Decorations !843
  br label %.preheader.14

.preheader.14:                                    ; preds = %._crit_edge.2.14, %1403
  %.sroa.250.2 = phi float [ %1423, %1403 ], [ %.sroa.250.1, %._crit_edge.2.14 ]
  br i1 %154, label %1424, label %._crit_edge.15

1424:                                             ; preds = %.preheader.14
  %.sroa.256.0.insert.ext883 = zext i32 %163 to i64
  %1425 = sext i32 %29 to i64
  %1426 = mul nsw i64 %1425, %const_reg_qword3, !spirv.Decorations !836
  %1427 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1426, i32 0
  %1428 = getelementptr i16, i16 addrspace(4)* %1427, i64 %.sroa.256.0.insert.ext883
  %1429 = addrspacecast i16 addrspace(4)* %1428 to i16 addrspace(1)*
  %1430 = load i16, i16 addrspace(1)* %1429, align 2
  %1431 = mul nsw i64 %.sroa.256.0.insert.ext883, %const_reg_qword5, !spirv.Decorations !836
  %1432 = sext i32 %152 to i64
  %1433 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1431, i32 0
  %1434 = getelementptr i16, i16 addrspace(4)* %1433, i64 %1432
  %1435 = addrspacecast i16 addrspace(4)* %1434 to i16 addrspace(1)*
  %1436 = load i16, i16 addrspace(1)* %1435, align 2
  %1437 = zext i16 %1430 to i32
  %1438 = shl nuw i32 %1437, 16, !spirv.Decorations !838
  %1439 = bitcast i32 %1438 to float
  %1440 = zext i16 %1436 to i32
  %1441 = shl nuw i32 %1440, 16, !spirv.Decorations !838
  %1442 = bitcast i32 %1441 to float
  %1443 = fmul reassoc nsz arcp contract float %1439, %1442, !spirv.Decorations !843
  %1444 = fadd reassoc nsz arcp contract float %1443, %.sroa.62.1, !spirv.Decorations !843
  br label %._crit_edge.15

._crit_edge.15:                                   ; preds = %.preheader.14, %1424
  %.sroa.62.2 = phi float [ %1444, %1424 ], [ %.sroa.62.1, %.preheader.14 ]
  br i1 %155, label %1445, label %._crit_edge.1.15

1445:                                             ; preds = %._crit_edge.15
  %.sroa.256.0.insert.ext888 = zext i32 %163 to i64
  %1446 = sext i32 %59 to i64
  %1447 = mul nsw i64 %1446, %const_reg_qword3, !spirv.Decorations !836
  %1448 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1447, i32 0
  %1449 = getelementptr i16, i16 addrspace(4)* %1448, i64 %.sroa.256.0.insert.ext888
  %1450 = addrspacecast i16 addrspace(4)* %1449 to i16 addrspace(1)*
  %1451 = load i16, i16 addrspace(1)* %1450, align 2
  %1452 = mul nsw i64 %.sroa.256.0.insert.ext888, %const_reg_qword5, !spirv.Decorations !836
  %1453 = sext i32 %152 to i64
  %1454 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1452, i32 0
  %1455 = getelementptr i16, i16 addrspace(4)* %1454, i64 %1453
  %1456 = addrspacecast i16 addrspace(4)* %1455 to i16 addrspace(1)*
  %1457 = load i16, i16 addrspace(1)* %1456, align 2
  %1458 = zext i16 %1451 to i32
  %1459 = shl nuw i32 %1458, 16, !spirv.Decorations !838
  %1460 = bitcast i32 %1459 to float
  %1461 = zext i16 %1457 to i32
  %1462 = shl nuw i32 %1461, 16, !spirv.Decorations !838
  %1463 = bitcast i32 %1462 to float
  %1464 = fmul reassoc nsz arcp contract float %1460, %1463, !spirv.Decorations !843
  %1465 = fadd reassoc nsz arcp contract float %1464, %.sroa.126.1, !spirv.Decorations !843
  br label %._crit_edge.1.15

._crit_edge.1.15:                                 ; preds = %._crit_edge.15, %1445
  %.sroa.126.2 = phi float [ %1465, %1445 ], [ %.sroa.126.1, %._crit_edge.15 ]
  br i1 %156, label %1466, label %._crit_edge.2.15

1466:                                             ; preds = %._crit_edge.1.15
  %.sroa.256.0.insert.ext893 = zext i32 %163 to i64
  %1467 = sext i32 %62 to i64
  %1468 = mul nsw i64 %1467, %const_reg_qword3, !spirv.Decorations !836
  %1469 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1468, i32 0
  %1470 = getelementptr i16, i16 addrspace(4)* %1469, i64 %.sroa.256.0.insert.ext893
  %1471 = addrspacecast i16 addrspace(4)* %1470 to i16 addrspace(1)*
  %1472 = load i16, i16 addrspace(1)* %1471, align 2
  %1473 = mul nsw i64 %.sroa.256.0.insert.ext893, %const_reg_qword5, !spirv.Decorations !836
  %1474 = sext i32 %152 to i64
  %1475 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1473, i32 0
  %1476 = getelementptr i16, i16 addrspace(4)* %1475, i64 %1474
  %1477 = addrspacecast i16 addrspace(4)* %1476 to i16 addrspace(1)*
  %1478 = load i16, i16 addrspace(1)* %1477, align 2
  %1479 = zext i16 %1472 to i32
  %1480 = shl nuw i32 %1479, 16, !spirv.Decorations !838
  %1481 = bitcast i32 %1480 to float
  %1482 = zext i16 %1478 to i32
  %1483 = shl nuw i32 %1482, 16, !spirv.Decorations !838
  %1484 = bitcast i32 %1483 to float
  %1485 = fmul reassoc nsz arcp contract float %1481, %1484, !spirv.Decorations !843
  %1486 = fadd reassoc nsz arcp contract float %1485, %.sroa.190.1, !spirv.Decorations !843
  br label %._crit_edge.2.15

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15, %1466
  %.sroa.190.2 = phi float [ %1486, %1466 ], [ %.sroa.190.1, %._crit_edge.1.15 ]
  br i1 %157, label %1487, label %.preheader.15

1487:                                             ; preds = %._crit_edge.2.15
  %.sroa.256.0.insert.ext898 = zext i32 %163 to i64
  %1488 = sext i32 %65 to i64
  %1489 = mul nsw i64 %1488, %const_reg_qword3, !spirv.Decorations !836
  %1490 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %1489, i32 0
  %1491 = getelementptr i16, i16 addrspace(4)* %1490, i64 %.sroa.256.0.insert.ext898
  %1492 = addrspacecast i16 addrspace(4)* %1491 to i16 addrspace(1)*
  %1493 = load i16, i16 addrspace(1)* %1492, align 2
  %1494 = mul nsw i64 %.sroa.256.0.insert.ext898, %const_reg_qword5, !spirv.Decorations !836
  %1495 = sext i32 %152 to i64
  %1496 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %1494, i32 0
  %1497 = getelementptr i16, i16 addrspace(4)* %1496, i64 %1495
  %1498 = addrspacecast i16 addrspace(4)* %1497 to i16 addrspace(1)*
  %1499 = load i16, i16 addrspace(1)* %1498, align 2
  %1500 = zext i16 %1493 to i32
  %1501 = shl nuw i32 %1500, 16, !spirv.Decorations !838
  %1502 = bitcast i32 %1501 to float
  %1503 = zext i16 %1499 to i32
  %1504 = shl nuw i32 %1503, 16, !spirv.Decorations !838
  %1505 = bitcast i32 %1504 to float
  %1506 = fmul reassoc nsz arcp contract float %1502, %1505, !spirv.Decorations !843
  %1507 = fadd reassoc nsz arcp contract float %1506, %.sroa.254.1, !spirv.Decorations !843
  br label %.preheader.15

.preheader.15:                                    ; preds = %._crit_edge.2.15, %1487
  %.sroa.254.2 = phi float [ %1507, %1487 ], [ %.sroa.254.1, %._crit_edge.2.15 ]
  %1508 = add nuw nsw i32 %163, 1, !spirv.Decorations !848
  %1509 = icmp slt i32 %1508, %const_reg_dword2
  br i1 %1509, label %.preheader.preheader, label %.preheader1.preheader

1510:                                             ; preds = %.preheader1.preheader
  %1511 = sext i32 %29 to i64
  %1512 = sext i32 %35 to i64
  %1513 = mul nsw i64 %1511, %const_reg_qword9, !spirv.Decorations !836
  %1514 = add nsw i64 %1513, %1512, !spirv.Decorations !836
  %1515 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !843
  br i1 %42, label %1516, label %1526

1516:                                             ; preds = %1510
  %1517 = mul nsw i64 %1511, %const_reg_qword7, !spirv.Decorations !836
  %1518 = getelementptr float, float addrspace(4)* %160, i64 %1517
  %1519 = getelementptr float, float addrspace(4)* %1518, i64 %1512
  %1520 = addrspacecast float addrspace(4)* %1519 to float addrspace(1)*
  %1521 = load float, float addrspace(1)* %1520, align 4
  %1522 = fmul reassoc nsz arcp contract float %1521, %4, !spirv.Decorations !843
  %1523 = fadd reassoc nsz arcp contract float %1515, %1522, !spirv.Decorations !843
  %1524 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1514
  %1525 = addrspacecast float addrspace(4)* %1524 to float addrspace(1)*
  store float %1523, float addrspace(1)* %1525, align 4
  br label %._crit_edge70

1526:                                             ; preds = %1510
  %1527 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1514
  %1528 = addrspacecast float addrspace(4)* %1527 to float addrspace(1)*
  store float %1515, float addrspace(1)* %1528, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %.preheader1.preheader, %1526, %1516
  br i1 %61, label %1529, label %._crit_edge70.1

1529:                                             ; preds = %._crit_edge70
  %1530 = sext i32 %59 to i64
  %1531 = sext i32 %35 to i64
  %1532 = mul nsw i64 %1530, %const_reg_qword9, !spirv.Decorations !836
  %1533 = add nsw i64 %1532, %1531, !spirv.Decorations !836
  %1534 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !843
  br i1 %42, label %1538, label %1535

1535:                                             ; preds = %1529
  %1536 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1533
  %1537 = addrspacecast float addrspace(4)* %1536 to float addrspace(1)*
  store float %1534, float addrspace(1)* %1537, align 4
  br label %._crit_edge70.1

1538:                                             ; preds = %1529
  %1539 = mul nsw i64 %1530, %const_reg_qword7, !spirv.Decorations !836
  %1540 = getelementptr float, float addrspace(4)* %160, i64 %1539
  %1541 = getelementptr float, float addrspace(4)* %1540, i64 %1531
  %1542 = addrspacecast float addrspace(4)* %1541 to float addrspace(1)*
  %1543 = load float, float addrspace(1)* %1542, align 4
  %1544 = fmul reassoc nsz arcp contract float %1543, %4, !spirv.Decorations !843
  %1545 = fadd reassoc nsz arcp contract float %1534, %1544, !spirv.Decorations !843
  %1546 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1533
  %1547 = addrspacecast float addrspace(4)* %1546 to float addrspace(1)*
  store float %1545, float addrspace(1)* %1547, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %._crit_edge70, %1538, %1535
  br i1 %64, label %1548, label %._crit_edge70.2

1548:                                             ; preds = %._crit_edge70.1
  %1549 = sext i32 %62 to i64
  %1550 = sext i32 %35 to i64
  %1551 = mul nsw i64 %1549, %const_reg_qword9, !spirv.Decorations !836
  %1552 = add nsw i64 %1551, %1550, !spirv.Decorations !836
  %1553 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !843
  br i1 %42, label %1557, label %1554

1554:                                             ; preds = %1548
  %1555 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1552
  %1556 = addrspacecast float addrspace(4)* %1555 to float addrspace(1)*
  store float %1553, float addrspace(1)* %1556, align 4
  br label %._crit_edge70.2

1557:                                             ; preds = %1548
  %1558 = mul nsw i64 %1549, %const_reg_qword7, !spirv.Decorations !836
  %1559 = getelementptr float, float addrspace(4)* %160, i64 %1558
  %1560 = getelementptr float, float addrspace(4)* %1559, i64 %1550
  %1561 = addrspacecast float addrspace(4)* %1560 to float addrspace(1)*
  %1562 = load float, float addrspace(1)* %1561, align 4
  %1563 = fmul reassoc nsz arcp contract float %1562, %4, !spirv.Decorations !843
  %1564 = fadd reassoc nsz arcp contract float %1553, %1563, !spirv.Decorations !843
  %1565 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1552
  %1566 = addrspacecast float addrspace(4)* %1565 to float addrspace(1)*
  store float %1564, float addrspace(1)* %1566, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %._crit_edge70.1, %1557, %1554
  br i1 %67, label %1567, label %.preheader1

1567:                                             ; preds = %._crit_edge70.2
  %1568 = sext i32 %65 to i64
  %1569 = sext i32 %35 to i64
  %1570 = mul nsw i64 %1568, %const_reg_qword9, !spirv.Decorations !836
  %1571 = add nsw i64 %1570, %1569, !spirv.Decorations !836
  %1572 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !843
  br i1 %42, label %1576, label %1573

1573:                                             ; preds = %1567
  %1574 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1571
  %1575 = addrspacecast float addrspace(4)* %1574 to float addrspace(1)*
  store float %1572, float addrspace(1)* %1575, align 4
  br label %.preheader1

1576:                                             ; preds = %1567
  %1577 = mul nsw i64 %1568, %const_reg_qword7, !spirv.Decorations !836
  %1578 = getelementptr float, float addrspace(4)* %160, i64 %1577
  %1579 = getelementptr float, float addrspace(4)* %1578, i64 %1569
  %1580 = addrspacecast float addrspace(4)* %1579 to float addrspace(1)*
  %1581 = load float, float addrspace(1)* %1580, align 4
  %1582 = fmul reassoc nsz arcp contract float %1581, %4, !spirv.Decorations !843
  %1583 = fadd reassoc nsz arcp contract float %1572, %1582, !spirv.Decorations !843
  %1584 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1571
  %1585 = addrspacecast float addrspace(4)* %1584 to float addrspace(1)*
  store float %1583, float addrspace(1)* %1585, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %._crit_edge70.2, %1576, %1573
  br i1 %70, label %1586, label %._crit_edge70.176

1586:                                             ; preds = %.preheader1
  %1587 = sext i32 %29 to i64
  %1588 = sext i32 %68 to i64
  %1589 = mul nsw i64 %1587, %const_reg_qword9, !spirv.Decorations !836
  %1590 = add nsw i64 %1589, %1588, !spirv.Decorations !836
  %1591 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !843
  br i1 %42, label %1595, label %1592

1592:                                             ; preds = %1586
  %1593 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1590
  %1594 = addrspacecast float addrspace(4)* %1593 to float addrspace(1)*
  store float %1591, float addrspace(1)* %1594, align 4
  br label %._crit_edge70.176

1595:                                             ; preds = %1586
  %1596 = mul nsw i64 %1587, %const_reg_qword7, !spirv.Decorations !836
  %1597 = getelementptr float, float addrspace(4)* %160, i64 %1596
  %1598 = getelementptr float, float addrspace(4)* %1597, i64 %1588
  %1599 = addrspacecast float addrspace(4)* %1598 to float addrspace(1)*
  %1600 = load float, float addrspace(1)* %1599, align 4
  %1601 = fmul reassoc nsz arcp contract float %1600, %4, !spirv.Decorations !843
  %1602 = fadd reassoc nsz arcp contract float %1591, %1601, !spirv.Decorations !843
  %1603 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1590
  %1604 = addrspacecast float addrspace(4)* %1603 to float addrspace(1)*
  store float %1602, float addrspace(1)* %1604, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %.preheader1, %1595, %1592
  br i1 %71, label %1605, label %._crit_edge70.1.1

1605:                                             ; preds = %._crit_edge70.176
  %1606 = sext i32 %59 to i64
  %1607 = sext i32 %68 to i64
  %1608 = mul nsw i64 %1606, %const_reg_qword9, !spirv.Decorations !836
  %1609 = add nsw i64 %1608, %1607, !spirv.Decorations !836
  %1610 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !843
  br i1 %42, label %1614, label %1611

1611:                                             ; preds = %1605
  %1612 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1609
  %1613 = addrspacecast float addrspace(4)* %1612 to float addrspace(1)*
  store float %1610, float addrspace(1)* %1613, align 4
  br label %._crit_edge70.1.1

1614:                                             ; preds = %1605
  %1615 = mul nsw i64 %1606, %const_reg_qword7, !spirv.Decorations !836
  %1616 = getelementptr float, float addrspace(4)* %160, i64 %1615
  %1617 = getelementptr float, float addrspace(4)* %1616, i64 %1607
  %1618 = addrspacecast float addrspace(4)* %1617 to float addrspace(1)*
  %1619 = load float, float addrspace(1)* %1618, align 4
  %1620 = fmul reassoc nsz arcp contract float %1619, %4, !spirv.Decorations !843
  %1621 = fadd reassoc nsz arcp contract float %1610, %1620, !spirv.Decorations !843
  %1622 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1609
  %1623 = addrspacecast float addrspace(4)* %1622 to float addrspace(1)*
  store float %1621, float addrspace(1)* %1623, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176, %1614, %1611
  br i1 %72, label %1624, label %._crit_edge70.2.1

1624:                                             ; preds = %._crit_edge70.1.1
  %1625 = sext i32 %62 to i64
  %1626 = sext i32 %68 to i64
  %1627 = mul nsw i64 %1625, %const_reg_qword9, !spirv.Decorations !836
  %1628 = add nsw i64 %1627, %1626, !spirv.Decorations !836
  %1629 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !843
  br i1 %42, label %1633, label %1630

1630:                                             ; preds = %1624
  %1631 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1628
  %1632 = addrspacecast float addrspace(4)* %1631 to float addrspace(1)*
  store float %1629, float addrspace(1)* %1632, align 4
  br label %._crit_edge70.2.1

1633:                                             ; preds = %1624
  %1634 = mul nsw i64 %1625, %const_reg_qword7, !spirv.Decorations !836
  %1635 = getelementptr float, float addrspace(4)* %160, i64 %1634
  %1636 = getelementptr float, float addrspace(4)* %1635, i64 %1626
  %1637 = addrspacecast float addrspace(4)* %1636 to float addrspace(1)*
  %1638 = load float, float addrspace(1)* %1637, align 4
  %1639 = fmul reassoc nsz arcp contract float %1638, %4, !spirv.Decorations !843
  %1640 = fadd reassoc nsz arcp contract float %1629, %1639, !spirv.Decorations !843
  %1641 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1628
  %1642 = addrspacecast float addrspace(4)* %1641 to float addrspace(1)*
  store float %1640, float addrspace(1)* %1642, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1, %1633, %1630
  br i1 %73, label %1643, label %.preheader1.1

1643:                                             ; preds = %._crit_edge70.2.1
  %1644 = sext i32 %65 to i64
  %1645 = sext i32 %68 to i64
  %1646 = mul nsw i64 %1644, %const_reg_qword9, !spirv.Decorations !836
  %1647 = add nsw i64 %1646, %1645, !spirv.Decorations !836
  %1648 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !843
  br i1 %42, label %1652, label %1649

1649:                                             ; preds = %1643
  %1650 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1647
  %1651 = addrspacecast float addrspace(4)* %1650 to float addrspace(1)*
  store float %1648, float addrspace(1)* %1651, align 4
  br label %.preheader1.1

1652:                                             ; preds = %1643
  %1653 = mul nsw i64 %1644, %const_reg_qword7, !spirv.Decorations !836
  %1654 = getelementptr float, float addrspace(4)* %160, i64 %1653
  %1655 = getelementptr float, float addrspace(4)* %1654, i64 %1645
  %1656 = addrspacecast float addrspace(4)* %1655 to float addrspace(1)*
  %1657 = load float, float addrspace(1)* %1656, align 4
  %1658 = fmul reassoc nsz arcp contract float %1657, %4, !spirv.Decorations !843
  %1659 = fadd reassoc nsz arcp contract float %1648, %1658, !spirv.Decorations !843
  %1660 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1647
  %1661 = addrspacecast float addrspace(4)* %1660 to float addrspace(1)*
  store float %1659, float addrspace(1)* %1661, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %._crit_edge70.2.1, %1652, %1649
  br i1 %76, label %1662, label %._crit_edge70.277

1662:                                             ; preds = %.preheader1.1
  %1663 = sext i32 %29 to i64
  %1664 = sext i32 %74 to i64
  %1665 = mul nsw i64 %1663, %const_reg_qword9, !spirv.Decorations !836
  %1666 = add nsw i64 %1665, %1664, !spirv.Decorations !836
  %1667 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !843
  br i1 %42, label %1671, label %1668

1668:                                             ; preds = %1662
  %1669 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1666
  %1670 = addrspacecast float addrspace(4)* %1669 to float addrspace(1)*
  store float %1667, float addrspace(1)* %1670, align 4
  br label %._crit_edge70.277

1671:                                             ; preds = %1662
  %1672 = mul nsw i64 %1663, %const_reg_qword7, !spirv.Decorations !836
  %1673 = getelementptr float, float addrspace(4)* %160, i64 %1672
  %1674 = getelementptr float, float addrspace(4)* %1673, i64 %1664
  %1675 = addrspacecast float addrspace(4)* %1674 to float addrspace(1)*
  %1676 = load float, float addrspace(1)* %1675, align 4
  %1677 = fmul reassoc nsz arcp contract float %1676, %4, !spirv.Decorations !843
  %1678 = fadd reassoc nsz arcp contract float %1667, %1677, !spirv.Decorations !843
  %1679 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1666
  %1680 = addrspacecast float addrspace(4)* %1679 to float addrspace(1)*
  store float %1678, float addrspace(1)* %1680, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %.preheader1.1, %1671, %1668
  br i1 %77, label %1681, label %._crit_edge70.1.2

1681:                                             ; preds = %._crit_edge70.277
  %1682 = sext i32 %59 to i64
  %1683 = sext i32 %74 to i64
  %1684 = mul nsw i64 %1682, %const_reg_qword9, !spirv.Decorations !836
  %1685 = add nsw i64 %1684, %1683, !spirv.Decorations !836
  %1686 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !843
  br i1 %42, label %1690, label %1687

1687:                                             ; preds = %1681
  %1688 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1685
  %1689 = addrspacecast float addrspace(4)* %1688 to float addrspace(1)*
  store float %1686, float addrspace(1)* %1689, align 4
  br label %._crit_edge70.1.2

1690:                                             ; preds = %1681
  %1691 = mul nsw i64 %1682, %const_reg_qword7, !spirv.Decorations !836
  %1692 = getelementptr float, float addrspace(4)* %160, i64 %1691
  %1693 = getelementptr float, float addrspace(4)* %1692, i64 %1683
  %1694 = addrspacecast float addrspace(4)* %1693 to float addrspace(1)*
  %1695 = load float, float addrspace(1)* %1694, align 4
  %1696 = fmul reassoc nsz arcp contract float %1695, %4, !spirv.Decorations !843
  %1697 = fadd reassoc nsz arcp contract float %1686, %1696, !spirv.Decorations !843
  %1698 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1685
  %1699 = addrspacecast float addrspace(4)* %1698 to float addrspace(1)*
  store float %1697, float addrspace(1)* %1699, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277, %1690, %1687
  br i1 %78, label %1700, label %._crit_edge70.2.2

1700:                                             ; preds = %._crit_edge70.1.2
  %1701 = sext i32 %62 to i64
  %1702 = sext i32 %74 to i64
  %1703 = mul nsw i64 %1701, %const_reg_qword9, !spirv.Decorations !836
  %1704 = add nsw i64 %1703, %1702, !spirv.Decorations !836
  %1705 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !843
  br i1 %42, label %1709, label %1706

1706:                                             ; preds = %1700
  %1707 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1704
  %1708 = addrspacecast float addrspace(4)* %1707 to float addrspace(1)*
  store float %1705, float addrspace(1)* %1708, align 4
  br label %._crit_edge70.2.2

1709:                                             ; preds = %1700
  %1710 = mul nsw i64 %1701, %const_reg_qword7, !spirv.Decorations !836
  %1711 = getelementptr float, float addrspace(4)* %160, i64 %1710
  %1712 = getelementptr float, float addrspace(4)* %1711, i64 %1702
  %1713 = addrspacecast float addrspace(4)* %1712 to float addrspace(1)*
  %1714 = load float, float addrspace(1)* %1713, align 4
  %1715 = fmul reassoc nsz arcp contract float %1714, %4, !spirv.Decorations !843
  %1716 = fadd reassoc nsz arcp contract float %1705, %1715, !spirv.Decorations !843
  %1717 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1704
  %1718 = addrspacecast float addrspace(4)* %1717 to float addrspace(1)*
  store float %1716, float addrspace(1)* %1718, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2, %1709, %1706
  br i1 %79, label %1719, label %.preheader1.2

1719:                                             ; preds = %._crit_edge70.2.2
  %1720 = sext i32 %65 to i64
  %1721 = sext i32 %74 to i64
  %1722 = mul nsw i64 %1720, %const_reg_qword9, !spirv.Decorations !836
  %1723 = add nsw i64 %1722, %1721, !spirv.Decorations !836
  %1724 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !843
  br i1 %42, label %1728, label %1725

1725:                                             ; preds = %1719
  %1726 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1723
  %1727 = addrspacecast float addrspace(4)* %1726 to float addrspace(1)*
  store float %1724, float addrspace(1)* %1727, align 4
  br label %.preheader1.2

1728:                                             ; preds = %1719
  %1729 = mul nsw i64 %1720, %const_reg_qword7, !spirv.Decorations !836
  %1730 = getelementptr float, float addrspace(4)* %160, i64 %1729
  %1731 = getelementptr float, float addrspace(4)* %1730, i64 %1721
  %1732 = addrspacecast float addrspace(4)* %1731 to float addrspace(1)*
  %1733 = load float, float addrspace(1)* %1732, align 4
  %1734 = fmul reassoc nsz arcp contract float %1733, %4, !spirv.Decorations !843
  %1735 = fadd reassoc nsz arcp contract float %1724, %1734, !spirv.Decorations !843
  %1736 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1723
  %1737 = addrspacecast float addrspace(4)* %1736 to float addrspace(1)*
  store float %1735, float addrspace(1)* %1737, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %._crit_edge70.2.2, %1728, %1725
  br i1 %82, label %1738, label %._crit_edge70.378

1738:                                             ; preds = %.preheader1.2
  %1739 = sext i32 %29 to i64
  %1740 = sext i32 %80 to i64
  %1741 = mul nsw i64 %1739, %const_reg_qword9, !spirv.Decorations !836
  %1742 = add nsw i64 %1741, %1740, !spirv.Decorations !836
  %1743 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !843
  br i1 %42, label %1747, label %1744

1744:                                             ; preds = %1738
  %1745 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1742
  %1746 = addrspacecast float addrspace(4)* %1745 to float addrspace(1)*
  store float %1743, float addrspace(1)* %1746, align 4
  br label %._crit_edge70.378

1747:                                             ; preds = %1738
  %1748 = mul nsw i64 %1739, %const_reg_qword7, !spirv.Decorations !836
  %1749 = getelementptr float, float addrspace(4)* %160, i64 %1748
  %1750 = getelementptr float, float addrspace(4)* %1749, i64 %1740
  %1751 = addrspacecast float addrspace(4)* %1750 to float addrspace(1)*
  %1752 = load float, float addrspace(1)* %1751, align 4
  %1753 = fmul reassoc nsz arcp contract float %1752, %4, !spirv.Decorations !843
  %1754 = fadd reassoc nsz arcp contract float %1743, %1753, !spirv.Decorations !843
  %1755 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1742
  %1756 = addrspacecast float addrspace(4)* %1755 to float addrspace(1)*
  store float %1754, float addrspace(1)* %1756, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %.preheader1.2, %1747, %1744
  br i1 %83, label %1757, label %._crit_edge70.1.3

1757:                                             ; preds = %._crit_edge70.378
  %1758 = sext i32 %59 to i64
  %1759 = sext i32 %80 to i64
  %1760 = mul nsw i64 %1758, %const_reg_qword9, !spirv.Decorations !836
  %1761 = add nsw i64 %1760, %1759, !spirv.Decorations !836
  %1762 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !843
  br i1 %42, label %1766, label %1763

1763:                                             ; preds = %1757
  %1764 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1761
  %1765 = addrspacecast float addrspace(4)* %1764 to float addrspace(1)*
  store float %1762, float addrspace(1)* %1765, align 4
  br label %._crit_edge70.1.3

1766:                                             ; preds = %1757
  %1767 = mul nsw i64 %1758, %const_reg_qword7, !spirv.Decorations !836
  %1768 = getelementptr float, float addrspace(4)* %160, i64 %1767
  %1769 = getelementptr float, float addrspace(4)* %1768, i64 %1759
  %1770 = addrspacecast float addrspace(4)* %1769 to float addrspace(1)*
  %1771 = load float, float addrspace(1)* %1770, align 4
  %1772 = fmul reassoc nsz arcp contract float %1771, %4, !spirv.Decorations !843
  %1773 = fadd reassoc nsz arcp contract float %1762, %1772, !spirv.Decorations !843
  %1774 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1761
  %1775 = addrspacecast float addrspace(4)* %1774 to float addrspace(1)*
  store float %1773, float addrspace(1)* %1775, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378, %1766, %1763
  br i1 %84, label %1776, label %._crit_edge70.2.3

1776:                                             ; preds = %._crit_edge70.1.3
  %1777 = sext i32 %62 to i64
  %1778 = sext i32 %80 to i64
  %1779 = mul nsw i64 %1777, %const_reg_qword9, !spirv.Decorations !836
  %1780 = add nsw i64 %1779, %1778, !spirv.Decorations !836
  %1781 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !843
  br i1 %42, label %1785, label %1782

1782:                                             ; preds = %1776
  %1783 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1780
  %1784 = addrspacecast float addrspace(4)* %1783 to float addrspace(1)*
  store float %1781, float addrspace(1)* %1784, align 4
  br label %._crit_edge70.2.3

1785:                                             ; preds = %1776
  %1786 = mul nsw i64 %1777, %const_reg_qword7, !spirv.Decorations !836
  %1787 = getelementptr float, float addrspace(4)* %160, i64 %1786
  %1788 = getelementptr float, float addrspace(4)* %1787, i64 %1778
  %1789 = addrspacecast float addrspace(4)* %1788 to float addrspace(1)*
  %1790 = load float, float addrspace(1)* %1789, align 4
  %1791 = fmul reassoc nsz arcp contract float %1790, %4, !spirv.Decorations !843
  %1792 = fadd reassoc nsz arcp contract float %1781, %1791, !spirv.Decorations !843
  %1793 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1780
  %1794 = addrspacecast float addrspace(4)* %1793 to float addrspace(1)*
  store float %1792, float addrspace(1)* %1794, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3, %1785, %1782
  br i1 %85, label %1795, label %.preheader1.3

1795:                                             ; preds = %._crit_edge70.2.3
  %1796 = sext i32 %65 to i64
  %1797 = sext i32 %80 to i64
  %1798 = mul nsw i64 %1796, %const_reg_qword9, !spirv.Decorations !836
  %1799 = add nsw i64 %1798, %1797, !spirv.Decorations !836
  %1800 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !843
  br i1 %42, label %1804, label %1801

1801:                                             ; preds = %1795
  %1802 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1799
  %1803 = addrspacecast float addrspace(4)* %1802 to float addrspace(1)*
  store float %1800, float addrspace(1)* %1803, align 4
  br label %.preheader1.3

1804:                                             ; preds = %1795
  %1805 = mul nsw i64 %1796, %const_reg_qword7, !spirv.Decorations !836
  %1806 = getelementptr float, float addrspace(4)* %160, i64 %1805
  %1807 = getelementptr float, float addrspace(4)* %1806, i64 %1797
  %1808 = addrspacecast float addrspace(4)* %1807 to float addrspace(1)*
  %1809 = load float, float addrspace(1)* %1808, align 4
  %1810 = fmul reassoc nsz arcp contract float %1809, %4, !spirv.Decorations !843
  %1811 = fadd reassoc nsz arcp contract float %1800, %1810, !spirv.Decorations !843
  %1812 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1799
  %1813 = addrspacecast float addrspace(4)* %1812 to float addrspace(1)*
  store float %1811, float addrspace(1)* %1813, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %._crit_edge70.2.3, %1804, %1801
  br i1 %88, label %1814, label %._crit_edge70.4

1814:                                             ; preds = %.preheader1.3
  %1815 = sext i32 %29 to i64
  %1816 = sext i32 %86 to i64
  %1817 = mul nsw i64 %1815, %const_reg_qword9, !spirv.Decorations !836
  %1818 = add nsw i64 %1817, %1816, !spirv.Decorations !836
  %1819 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !843
  br i1 %42, label %1823, label %1820

1820:                                             ; preds = %1814
  %1821 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1818
  %1822 = addrspacecast float addrspace(4)* %1821 to float addrspace(1)*
  store float %1819, float addrspace(1)* %1822, align 4
  br label %._crit_edge70.4

1823:                                             ; preds = %1814
  %1824 = mul nsw i64 %1815, %const_reg_qword7, !spirv.Decorations !836
  %1825 = getelementptr float, float addrspace(4)* %160, i64 %1824
  %1826 = getelementptr float, float addrspace(4)* %1825, i64 %1816
  %1827 = addrspacecast float addrspace(4)* %1826 to float addrspace(1)*
  %1828 = load float, float addrspace(1)* %1827, align 4
  %1829 = fmul reassoc nsz arcp contract float %1828, %4, !spirv.Decorations !843
  %1830 = fadd reassoc nsz arcp contract float %1819, %1829, !spirv.Decorations !843
  %1831 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1818
  %1832 = addrspacecast float addrspace(4)* %1831 to float addrspace(1)*
  store float %1830, float addrspace(1)* %1832, align 4
  br label %._crit_edge70.4

._crit_edge70.4:                                  ; preds = %.preheader1.3, %1823, %1820
  br i1 %89, label %1833, label %._crit_edge70.1.4

1833:                                             ; preds = %._crit_edge70.4
  %1834 = sext i32 %59 to i64
  %1835 = sext i32 %86 to i64
  %1836 = mul nsw i64 %1834, %const_reg_qword9, !spirv.Decorations !836
  %1837 = add nsw i64 %1836, %1835, !spirv.Decorations !836
  %1838 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !843
  br i1 %42, label %1842, label %1839

1839:                                             ; preds = %1833
  %1840 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1837
  %1841 = addrspacecast float addrspace(4)* %1840 to float addrspace(1)*
  store float %1838, float addrspace(1)* %1841, align 4
  br label %._crit_edge70.1.4

1842:                                             ; preds = %1833
  %1843 = mul nsw i64 %1834, %const_reg_qword7, !spirv.Decorations !836
  %1844 = getelementptr float, float addrspace(4)* %160, i64 %1843
  %1845 = getelementptr float, float addrspace(4)* %1844, i64 %1835
  %1846 = addrspacecast float addrspace(4)* %1845 to float addrspace(1)*
  %1847 = load float, float addrspace(1)* %1846, align 4
  %1848 = fmul reassoc nsz arcp contract float %1847, %4, !spirv.Decorations !843
  %1849 = fadd reassoc nsz arcp contract float %1838, %1848, !spirv.Decorations !843
  %1850 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1837
  %1851 = addrspacecast float addrspace(4)* %1850 to float addrspace(1)*
  store float %1849, float addrspace(1)* %1851, align 4
  br label %._crit_edge70.1.4

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4, %1842, %1839
  br i1 %90, label %1852, label %._crit_edge70.2.4

1852:                                             ; preds = %._crit_edge70.1.4
  %1853 = sext i32 %62 to i64
  %1854 = sext i32 %86 to i64
  %1855 = mul nsw i64 %1853, %const_reg_qword9, !spirv.Decorations !836
  %1856 = add nsw i64 %1855, %1854, !spirv.Decorations !836
  %1857 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !843
  br i1 %42, label %1861, label %1858

1858:                                             ; preds = %1852
  %1859 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1856
  %1860 = addrspacecast float addrspace(4)* %1859 to float addrspace(1)*
  store float %1857, float addrspace(1)* %1860, align 4
  br label %._crit_edge70.2.4

1861:                                             ; preds = %1852
  %1862 = mul nsw i64 %1853, %const_reg_qword7, !spirv.Decorations !836
  %1863 = getelementptr float, float addrspace(4)* %160, i64 %1862
  %1864 = getelementptr float, float addrspace(4)* %1863, i64 %1854
  %1865 = addrspacecast float addrspace(4)* %1864 to float addrspace(1)*
  %1866 = load float, float addrspace(1)* %1865, align 4
  %1867 = fmul reassoc nsz arcp contract float %1866, %4, !spirv.Decorations !843
  %1868 = fadd reassoc nsz arcp contract float %1857, %1867, !spirv.Decorations !843
  %1869 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1856
  %1870 = addrspacecast float addrspace(4)* %1869 to float addrspace(1)*
  store float %1868, float addrspace(1)* %1870, align 4
  br label %._crit_edge70.2.4

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4, %1861, %1858
  br i1 %91, label %1871, label %.preheader1.4

1871:                                             ; preds = %._crit_edge70.2.4
  %1872 = sext i32 %65 to i64
  %1873 = sext i32 %86 to i64
  %1874 = mul nsw i64 %1872, %const_reg_qword9, !spirv.Decorations !836
  %1875 = add nsw i64 %1874, %1873, !spirv.Decorations !836
  %1876 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !843
  br i1 %42, label %1880, label %1877

1877:                                             ; preds = %1871
  %1878 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1875
  %1879 = addrspacecast float addrspace(4)* %1878 to float addrspace(1)*
  store float %1876, float addrspace(1)* %1879, align 4
  br label %.preheader1.4

1880:                                             ; preds = %1871
  %1881 = mul nsw i64 %1872, %const_reg_qword7, !spirv.Decorations !836
  %1882 = getelementptr float, float addrspace(4)* %160, i64 %1881
  %1883 = getelementptr float, float addrspace(4)* %1882, i64 %1873
  %1884 = addrspacecast float addrspace(4)* %1883 to float addrspace(1)*
  %1885 = load float, float addrspace(1)* %1884, align 4
  %1886 = fmul reassoc nsz arcp contract float %1885, %4, !spirv.Decorations !843
  %1887 = fadd reassoc nsz arcp contract float %1876, %1886, !spirv.Decorations !843
  %1888 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1875
  %1889 = addrspacecast float addrspace(4)* %1888 to float addrspace(1)*
  store float %1887, float addrspace(1)* %1889, align 4
  br label %.preheader1.4

.preheader1.4:                                    ; preds = %._crit_edge70.2.4, %1880, %1877
  br i1 %94, label %1890, label %._crit_edge70.5

1890:                                             ; preds = %.preheader1.4
  %1891 = sext i32 %29 to i64
  %1892 = sext i32 %92 to i64
  %1893 = mul nsw i64 %1891, %const_reg_qword9, !spirv.Decorations !836
  %1894 = add nsw i64 %1893, %1892, !spirv.Decorations !836
  %1895 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !843
  br i1 %42, label %1899, label %1896

1896:                                             ; preds = %1890
  %1897 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1894
  %1898 = addrspacecast float addrspace(4)* %1897 to float addrspace(1)*
  store float %1895, float addrspace(1)* %1898, align 4
  br label %._crit_edge70.5

1899:                                             ; preds = %1890
  %1900 = mul nsw i64 %1891, %const_reg_qword7, !spirv.Decorations !836
  %1901 = getelementptr float, float addrspace(4)* %160, i64 %1900
  %1902 = getelementptr float, float addrspace(4)* %1901, i64 %1892
  %1903 = addrspacecast float addrspace(4)* %1902 to float addrspace(1)*
  %1904 = load float, float addrspace(1)* %1903, align 4
  %1905 = fmul reassoc nsz arcp contract float %1904, %4, !spirv.Decorations !843
  %1906 = fadd reassoc nsz arcp contract float %1895, %1905, !spirv.Decorations !843
  %1907 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1894
  %1908 = addrspacecast float addrspace(4)* %1907 to float addrspace(1)*
  store float %1906, float addrspace(1)* %1908, align 4
  br label %._crit_edge70.5

._crit_edge70.5:                                  ; preds = %.preheader1.4, %1899, %1896
  br i1 %95, label %1909, label %._crit_edge70.1.5

1909:                                             ; preds = %._crit_edge70.5
  %1910 = sext i32 %59 to i64
  %1911 = sext i32 %92 to i64
  %1912 = mul nsw i64 %1910, %const_reg_qword9, !spirv.Decorations !836
  %1913 = add nsw i64 %1912, %1911, !spirv.Decorations !836
  %1914 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !843
  br i1 %42, label %1918, label %1915

1915:                                             ; preds = %1909
  %1916 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1913
  %1917 = addrspacecast float addrspace(4)* %1916 to float addrspace(1)*
  store float %1914, float addrspace(1)* %1917, align 4
  br label %._crit_edge70.1.5

1918:                                             ; preds = %1909
  %1919 = mul nsw i64 %1910, %const_reg_qword7, !spirv.Decorations !836
  %1920 = getelementptr float, float addrspace(4)* %160, i64 %1919
  %1921 = getelementptr float, float addrspace(4)* %1920, i64 %1911
  %1922 = addrspacecast float addrspace(4)* %1921 to float addrspace(1)*
  %1923 = load float, float addrspace(1)* %1922, align 4
  %1924 = fmul reassoc nsz arcp contract float %1923, %4, !spirv.Decorations !843
  %1925 = fadd reassoc nsz arcp contract float %1914, %1924, !spirv.Decorations !843
  %1926 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1913
  %1927 = addrspacecast float addrspace(4)* %1926 to float addrspace(1)*
  store float %1925, float addrspace(1)* %1927, align 4
  br label %._crit_edge70.1.5

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5, %1918, %1915
  br i1 %96, label %1928, label %._crit_edge70.2.5

1928:                                             ; preds = %._crit_edge70.1.5
  %1929 = sext i32 %62 to i64
  %1930 = sext i32 %92 to i64
  %1931 = mul nsw i64 %1929, %const_reg_qword9, !spirv.Decorations !836
  %1932 = add nsw i64 %1931, %1930, !spirv.Decorations !836
  %1933 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !843
  br i1 %42, label %1937, label %1934

1934:                                             ; preds = %1928
  %1935 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1932
  %1936 = addrspacecast float addrspace(4)* %1935 to float addrspace(1)*
  store float %1933, float addrspace(1)* %1936, align 4
  br label %._crit_edge70.2.5

1937:                                             ; preds = %1928
  %1938 = mul nsw i64 %1929, %const_reg_qword7, !spirv.Decorations !836
  %1939 = getelementptr float, float addrspace(4)* %160, i64 %1938
  %1940 = getelementptr float, float addrspace(4)* %1939, i64 %1930
  %1941 = addrspacecast float addrspace(4)* %1940 to float addrspace(1)*
  %1942 = load float, float addrspace(1)* %1941, align 4
  %1943 = fmul reassoc nsz arcp contract float %1942, %4, !spirv.Decorations !843
  %1944 = fadd reassoc nsz arcp contract float %1933, %1943, !spirv.Decorations !843
  %1945 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1932
  %1946 = addrspacecast float addrspace(4)* %1945 to float addrspace(1)*
  store float %1944, float addrspace(1)* %1946, align 4
  br label %._crit_edge70.2.5

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5, %1937, %1934
  br i1 %97, label %1947, label %.preheader1.5

1947:                                             ; preds = %._crit_edge70.2.5
  %1948 = sext i32 %65 to i64
  %1949 = sext i32 %92 to i64
  %1950 = mul nsw i64 %1948, %const_reg_qword9, !spirv.Decorations !836
  %1951 = add nsw i64 %1950, %1949, !spirv.Decorations !836
  %1952 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !843
  br i1 %42, label %1956, label %1953

1953:                                             ; preds = %1947
  %1954 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1951
  %1955 = addrspacecast float addrspace(4)* %1954 to float addrspace(1)*
  store float %1952, float addrspace(1)* %1955, align 4
  br label %.preheader1.5

1956:                                             ; preds = %1947
  %1957 = mul nsw i64 %1948, %const_reg_qword7, !spirv.Decorations !836
  %1958 = getelementptr float, float addrspace(4)* %160, i64 %1957
  %1959 = getelementptr float, float addrspace(4)* %1958, i64 %1949
  %1960 = addrspacecast float addrspace(4)* %1959 to float addrspace(1)*
  %1961 = load float, float addrspace(1)* %1960, align 4
  %1962 = fmul reassoc nsz arcp contract float %1961, %4, !spirv.Decorations !843
  %1963 = fadd reassoc nsz arcp contract float %1952, %1962, !spirv.Decorations !843
  %1964 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1951
  %1965 = addrspacecast float addrspace(4)* %1964 to float addrspace(1)*
  store float %1963, float addrspace(1)* %1965, align 4
  br label %.preheader1.5

.preheader1.5:                                    ; preds = %._crit_edge70.2.5, %1956, %1953
  br i1 %100, label %1966, label %._crit_edge70.6

1966:                                             ; preds = %.preheader1.5
  %1967 = sext i32 %29 to i64
  %1968 = sext i32 %98 to i64
  %1969 = mul nsw i64 %1967, %const_reg_qword9, !spirv.Decorations !836
  %1970 = add nsw i64 %1969, %1968, !spirv.Decorations !836
  %1971 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !843
  br i1 %42, label %1975, label %1972

1972:                                             ; preds = %1966
  %1973 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1970
  %1974 = addrspacecast float addrspace(4)* %1973 to float addrspace(1)*
  store float %1971, float addrspace(1)* %1974, align 4
  br label %._crit_edge70.6

1975:                                             ; preds = %1966
  %1976 = mul nsw i64 %1967, %const_reg_qword7, !spirv.Decorations !836
  %1977 = getelementptr float, float addrspace(4)* %160, i64 %1976
  %1978 = getelementptr float, float addrspace(4)* %1977, i64 %1968
  %1979 = addrspacecast float addrspace(4)* %1978 to float addrspace(1)*
  %1980 = load float, float addrspace(1)* %1979, align 4
  %1981 = fmul reassoc nsz arcp contract float %1980, %4, !spirv.Decorations !843
  %1982 = fadd reassoc nsz arcp contract float %1971, %1981, !spirv.Decorations !843
  %1983 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1970
  %1984 = addrspacecast float addrspace(4)* %1983 to float addrspace(1)*
  store float %1982, float addrspace(1)* %1984, align 4
  br label %._crit_edge70.6

._crit_edge70.6:                                  ; preds = %.preheader1.5, %1975, %1972
  br i1 %101, label %1985, label %._crit_edge70.1.6

1985:                                             ; preds = %._crit_edge70.6
  %1986 = sext i32 %59 to i64
  %1987 = sext i32 %98 to i64
  %1988 = mul nsw i64 %1986, %const_reg_qword9, !spirv.Decorations !836
  %1989 = add nsw i64 %1988, %1987, !spirv.Decorations !836
  %1990 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !843
  br i1 %42, label %1994, label %1991

1991:                                             ; preds = %1985
  %1992 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1989
  %1993 = addrspacecast float addrspace(4)* %1992 to float addrspace(1)*
  store float %1990, float addrspace(1)* %1993, align 4
  br label %._crit_edge70.1.6

1994:                                             ; preds = %1985
  %1995 = mul nsw i64 %1986, %const_reg_qword7, !spirv.Decorations !836
  %1996 = getelementptr float, float addrspace(4)* %160, i64 %1995
  %1997 = getelementptr float, float addrspace(4)* %1996, i64 %1987
  %1998 = addrspacecast float addrspace(4)* %1997 to float addrspace(1)*
  %1999 = load float, float addrspace(1)* %1998, align 4
  %2000 = fmul reassoc nsz arcp contract float %1999, %4, !spirv.Decorations !843
  %2001 = fadd reassoc nsz arcp contract float %1990, %2000, !spirv.Decorations !843
  %2002 = getelementptr inbounds float, float addrspace(4)* %159, i64 %1989
  %2003 = addrspacecast float addrspace(4)* %2002 to float addrspace(1)*
  store float %2001, float addrspace(1)* %2003, align 4
  br label %._crit_edge70.1.6

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6, %1994, %1991
  br i1 %102, label %2004, label %._crit_edge70.2.6

2004:                                             ; preds = %._crit_edge70.1.6
  %2005 = sext i32 %62 to i64
  %2006 = sext i32 %98 to i64
  %2007 = mul nsw i64 %2005, %const_reg_qword9, !spirv.Decorations !836
  %2008 = add nsw i64 %2007, %2006, !spirv.Decorations !836
  %2009 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !843
  br i1 %42, label %2013, label %2010

2010:                                             ; preds = %2004
  %2011 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2008
  %2012 = addrspacecast float addrspace(4)* %2011 to float addrspace(1)*
  store float %2009, float addrspace(1)* %2012, align 4
  br label %._crit_edge70.2.6

2013:                                             ; preds = %2004
  %2014 = mul nsw i64 %2005, %const_reg_qword7, !spirv.Decorations !836
  %2015 = getelementptr float, float addrspace(4)* %160, i64 %2014
  %2016 = getelementptr float, float addrspace(4)* %2015, i64 %2006
  %2017 = addrspacecast float addrspace(4)* %2016 to float addrspace(1)*
  %2018 = load float, float addrspace(1)* %2017, align 4
  %2019 = fmul reassoc nsz arcp contract float %2018, %4, !spirv.Decorations !843
  %2020 = fadd reassoc nsz arcp contract float %2009, %2019, !spirv.Decorations !843
  %2021 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2008
  %2022 = addrspacecast float addrspace(4)* %2021 to float addrspace(1)*
  store float %2020, float addrspace(1)* %2022, align 4
  br label %._crit_edge70.2.6

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6, %2013, %2010
  br i1 %103, label %2023, label %.preheader1.6

2023:                                             ; preds = %._crit_edge70.2.6
  %2024 = sext i32 %65 to i64
  %2025 = sext i32 %98 to i64
  %2026 = mul nsw i64 %2024, %const_reg_qword9, !spirv.Decorations !836
  %2027 = add nsw i64 %2026, %2025, !spirv.Decorations !836
  %2028 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !843
  br i1 %42, label %2032, label %2029

2029:                                             ; preds = %2023
  %2030 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2027
  %2031 = addrspacecast float addrspace(4)* %2030 to float addrspace(1)*
  store float %2028, float addrspace(1)* %2031, align 4
  br label %.preheader1.6

2032:                                             ; preds = %2023
  %2033 = mul nsw i64 %2024, %const_reg_qword7, !spirv.Decorations !836
  %2034 = getelementptr float, float addrspace(4)* %160, i64 %2033
  %2035 = getelementptr float, float addrspace(4)* %2034, i64 %2025
  %2036 = addrspacecast float addrspace(4)* %2035 to float addrspace(1)*
  %2037 = load float, float addrspace(1)* %2036, align 4
  %2038 = fmul reassoc nsz arcp contract float %2037, %4, !spirv.Decorations !843
  %2039 = fadd reassoc nsz arcp contract float %2028, %2038, !spirv.Decorations !843
  %2040 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2027
  %2041 = addrspacecast float addrspace(4)* %2040 to float addrspace(1)*
  store float %2039, float addrspace(1)* %2041, align 4
  br label %.preheader1.6

.preheader1.6:                                    ; preds = %._crit_edge70.2.6, %2032, %2029
  br i1 %106, label %2042, label %._crit_edge70.7

2042:                                             ; preds = %.preheader1.6
  %2043 = sext i32 %29 to i64
  %2044 = sext i32 %104 to i64
  %2045 = mul nsw i64 %2043, %const_reg_qword9, !spirv.Decorations !836
  %2046 = add nsw i64 %2045, %2044, !spirv.Decorations !836
  %2047 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !843
  br i1 %42, label %2051, label %2048

2048:                                             ; preds = %2042
  %2049 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2046
  %2050 = addrspacecast float addrspace(4)* %2049 to float addrspace(1)*
  store float %2047, float addrspace(1)* %2050, align 4
  br label %._crit_edge70.7

2051:                                             ; preds = %2042
  %2052 = mul nsw i64 %2043, %const_reg_qword7, !spirv.Decorations !836
  %2053 = getelementptr float, float addrspace(4)* %160, i64 %2052
  %2054 = getelementptr float, float addrspace(4)* %2053, i64 %2044
  %2055 = addrspacecast float addrspace(4)* %2054 to float addrspace(1)*
  %2056 = load float, float addrspace(1)* %2055, align 4
  %2057 = fmul reassoc nsz arcp contract float %2056, %4, !spirv.Decorations !843
  %2058 = fadd reassoc nsz arcp contract float %2047, %2057, !spirv.Decorations !843
  %2059 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2046
  %2060 = addrspacecast float addrspace(4)* %2059 to float addrspace(1)*
  store float %2058, float addrspace(1)* %2060, align 4
  br label %._crit_edge70.7

._crit_edge70.7:                                  ; preds = %.preheader1.6, %2051, %2048
  br i1 %107, label %2061, label %._crit_edge70.1.7

2061:                                             ; preds = %._crit_edge70.7
  %2062 = sext i32 %59 to i64
  %2063 = sext i32 %104 to i64
  %2064 = mul nsw i64 %2062, %const_reg_qword9, !spirv.Decorations !836
  %2065 = add nsw i64 %2064, %2063, !spirv.Decorations !836
  %2066 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !843
  br i1 %42, label %2070, label %2067

2067:                                             ; preds = %2061
  %2068 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2065
  %2069 = addrspacecast float addrspace(4)* %2068 to float addrspace(1)*
  store float %2066, float addrspace(1)* %2069, align 4
  br label %._crit_edge70.1.7

2070:                                             ; preds = %2061
  %2071 = mul nsw i64 %2062, %const_reg_qword7, !spirv.Decorations !836
  %2072 = getelementptr float, float addrspace(4)* %160, i64 %2071
  %2073 = getelementptr float, float addrspace(4)* %2072, i64 %2063
  %2074 = addrspacecast float addrspace(4)* %2073 to float addrspace(1)*
  %2075 = load float, float addrspace(1)* %2074, align 4
  %2076 = fmul reassoc nsz arcp contract float %2075, %4, !spirv.Decorations !843
  %2077 = fadd reassoc nsz arcp contract float %2066, %2076, !spirv.Decorations !843
  %2078 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2065
  %2079 = addrspacecast float addrspace(4)* %2078 to float addrspace(1)*
  store float %2077, float addrspace(1)* %2079, align 4
  br label %._crit_edge70.1.7

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7, %2070, %2067
  br i1 %108, label %2080, label %._crit_edge70.2.7

2080:                                             ; preds = %._crit_edge70.1.7
  %2081 = sext i32 %62 to i64
  %2082 = sext i32 %104 to i64
  %2083 = mul nsw i64 %2081, %const_reg_qword9, !spirv.Decorations !836
  %2084 = add nsw i64 %2083, %2082, !spirv.Decorations !836
  %2085 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !843
  br i1 %42, label %2089, label %2086

2086:                                             ; preds = %2080
  %2087 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2084
  %2088 = addrspacecast float addrspace(4)* %2087 to float addrspace(1)*
  store float %2085, float addrspace(1)* %2088, align 4
  br label %._crit_edge70.2.7

2089:                                             ; preds = %2080
  %2090 = mul nsw i64 %2081, %const_reg_qword7, !spirv.Decorations !836
  %2091 = getelementptr float, float addrspace(4)* %160, i64 %2090
  %2092 = getelementptr float, float addrspace(4)* %2091, i64 %2082
  %2093 = addrspacecast float addrspace(4)* %2092 to float addrspace(1)*
  %2094 = load float, float addrspace(1)* %2093, align 4
  %2095 = fmul reassoc nsz arcp contract float %2094, %4, !spirv.Decorations !843
  %2096 = fadd reassoc nsz arcp contract float %2085, %2095, !spirv.Decorations !843
  %2097 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2084
  %2098 = addrspacecast float addrspace(4)* %2097 to float addrspace(1)*
  store float %2096, float addrspace(1)* %2098, align 4
  br label %._crit_edge70.2.7

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7, %2089, %2086
  br i1 %109, label %2099, label %.preheader1.7

2099:                                             ; preds = %._crit_edge70.2.7
  %2100 = sext i32 %65 to i64
  %2101 = sext i32 %104 to i64
  %2102 = mul nsw i64 %2100, %const_reg_qword9, !spirv.Decorations !836
  %2103 = add nsw i64 %2102, %2101, !spirv.Decorations !836
  %2104 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !843
  br i1 %42, label %2108, label %2105

2105:                                             ; preds = %2099
  %2106 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2103
  %2107 = addrspacecast float addrspace(4)* %2106 to float addrspace(1)*
  store float %2104, float addrspace(1)* %2107, align 4
  br label %.preheader1.7

2108:                                             ; preds = %2099
  %2109 = mul nsw i64 %2100, %const_reg_qword7, !spirv.Decorations !836
  %2110 = getelementptr float, float addrspace(4)* %160, i64 %2109
  %2111 = getelementptr float, float addrspace(4)* %2110, i64 %2101
  %2112 = addrspacecast float addrspace(4)* %2111 to float addrspace(1)*
  %2113 = load float, float addrspace(1)* %2112, align 4
  %2114 = fmul reassoc nsz arcp contract float %2113, %4, !spirv.Decorations !843
  %2115 = fadd reassoc nsz arcp contract float %2104, %2114, !spirv.Decorations !843
  %2116 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2103
  %2117 = addrspacecast float addrspace(4)* %2116 to float addrspace(1)*
  store float %2115, float addrspace(1)* %2117, align 4
  br label %.preheader1.7

.preheader1.7:                                    ; preds = %._crit_edge70.2.7, %2108, %2105
  br i1 %112, label %2118, label %._crit_edge70.8

2118:                                             ; preds = %.preheader1.7
  %2119 = sext i32 %29 to i64
  %2120 = sext i32 %110 to i64
  %2121 = mul nsw i64 %2119, %const_reg_qword9, !spirv.Decorations !836
  %2122 = add nsw i64 %2121, %2120, !spirv.Decorations !836
  %2123 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !843
  br i1 %42, label %2127, label %2124

2124:                                             ; preds = %2118
  %2125 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2122
  %2126 = addrspacecast float addrspace(4)* %2125 to float addrspace(1)*
  store float %2123, float addrspace(1)* %2126, align 4
  br label %._crit_edge70.8

2127:                                             ; preds = %2118
  %2128 = mul nsw i64 %2119, %const_reg_qword7, !spirv.Decorations !836
  %2129 = getelementptr float, float addrspace(4)* %160, i64 %2128
  %2130 = getelementptr float, float addrspace(4)* %2129, i64 %2120
  %2131 = addrspacecast float addrspace(4)* %2130 to float addrspace(1)*
  %2132 = load float, float addrspace(1)* %2131, align 4
  %2133 = fmul reassoc nsz arcp contract float %2132, %4, !spirv.Decorations !843
  %2134 = fadd reassoc nsz arcp contract float %2123, %2133, !spirv.Decorations !843
  %2135 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2122
  %2136 = addrspacecast float addrspace(4)* %2135 to float addrspace(1)*
  store float %2134, float addrspace(1)* %2136, align 4
  br label %._crit_edge70.8

._crit_edge70.8:                                  ; preds = %.preheader1.7, %2127, %2124
  br i1 %113, label %2137, label %._crit_edge70.1.8

2137:                                             ; preds = %._crit_edge70.8
  %2138 = sext i32 %59 to i64
  %2139 = sext i32 %110 to i64
  %2140 = mul nsw i64 %2138, %const_reg_qword9, !spirv.Decorations !836
  %2141 = add nsw i64 %2140, %2139, !spirv.Decorations !836
  %2142 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !843
  br i1 %42, label %2146, label %2143

2143:                                             ; preds = %2137
  %2144 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2141
  %2145 = addrspacecast float addrspace(4)* %2144 to float addrspace(1)*
  store float %2142, float addrspace(1)* %2145, align 4
  br label %._crit_edge70.1.8

2146:                                             ; preds = %2137
  %2147 = mul nsw i64 %2138, %const_reg_qword7, !spirv.Decorations !836
  %2148 = getelementptr float, float addrspace(4)* %160, i64 %2147
  %2149 = getelementptr float, float addrspace(4)* %2148, i64 %2139
  %2150 = addrspacecast float addrspace(4)* %2149 to float addrspace(1)*
  %2151 = load float, float addrspace(1)* %2150, align 4
  %2152 = fmul reassoc nsz arcp contract float %2151, %4, !spirv.Decorations !843
  %2153 = fadd reassoc nsz arcp contract float %2142, %2152, !spirv.Decorations !843
  %2154 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2141
  %2155 = addrspacecast float addrspace(4)* %2154 to float addrspace(1)*
  store float %2153, float addrspace(1)* %2155, align 4
  br label %._crit_edge70.1.8

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8, %2146, %2143
  br i1 %114, label %2156, label %._crit_edge70.2.8

2156:                                             ; preds = %._crit_edge70.1.8
  %2157 = sext i32 %62 to i64
  %2158 = sext i32 %110 to i64
  %2159 = mul nsw i64 %2157, %const_reg_qword9, !spirv.Decorations !836
  %2160 = add nsw i64 %2159, %2158, !spirv.Decorations !836
  %2161 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !843
  br i1 %42, label %2165, label %2162

2162:                                             ; preds = %2156
  %2163 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2160
  %2164 = addrspacecast float addrspace(4)* %2163 to float addrspace(1)*
  store float %2161, float addrspace(1)* %2164, align 4
  br label %._crit_edge70.2.8

2165:                                             ; preds = %2156
  %2166 = mul nsw i64 %2157, %const_reg_qword7, !spirv.Decorations !836
  %2167 = getelementptr float, float addrspace(4)* %160, i64 %2166
  %2168 = getelementptr float, float addrspace(4)* %2167, i64 %2158
  %2169 = addrspacecast float addrspace(4)* %2168 to float addrspace(1)*
  %2170 = load float, float addrspace(1)* %2169, align 4
  %2171 = fmul reassoc nsz arcp contract float %2170, %4, !spirv.Decorations !843
  %2172 = fadd reassoc nsz arcp contract float %2161, %2171, !spirv.Decorations !843
  %2173 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2160
  %2174 = addrspacecast float addrspace(4)* %2173 to float addrspace(1)*
  store float %2172, float addrspace(1)* %2174, align 4
  br label %._crit_edge70.2.8

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8, %2165, %2162
  br i1 %115, label %2175, label %.preheader1.8

2175:                                             ; preds = %._crit_edge70.2.8
  %2176 = sext i32 %65 to i64
  %2177 = sext i32 %110 to i64
  %2178 = mul nsw i64 %2176, %const_reg_qword9, !spirv.Decorations !836
  %2179 = add nsw i64 %2178, %2177, !spirv.Decorations !836
  %2180 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !843
  br i1 %42, label %2184, label %2181

2181:                                             ; preds = %2175
  %2182 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2179
  %2183 = addrspacecast float addrspace(4)* %2182 to float addrspace(1)*
  store float %2180, float addrspace(1)* %2183, align 4
  br label %.preheader1.8

2184:                                             ; preds = %2175
  %2185 = mul nsw i64 %2176, %const_reg_qword7, !spirv.Decorations !836
  %2186 = getelementptr float, float addrspace(4)* %160, i64 %2185
  %2187 = getelementptr float, float addrspace(4)* %2186, i64 %2177
  %2188 = addrspacecast float addrspace(4)* %2187 to float addrspace(1)*
  %2189 = load float, float addrspace(1)* %2188, align 4
  %2190 = fmul reassoc nsz arcp contract float %2189, %4, !spirv.Decorations !843
  %2191 = fadd reassoc nsz arcp contract float %2180, %2190, !spirv.Decorations !843
  %2192 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2179
  %2193 = addrspacecast float addrspace(4)* %2192 to float addrspace(1)*
  store float %2191, float addrspace(1)* %2193, align 4
  br label %.preheader1.8

.preheader1.8:                                    ; preds = %._crit_edge70.2.8, %2184, %2181
  br i1 %118, label %2194, label %._crit_edge70.9

2194:                                             ; preds = %.preheader1.8
  %2195 = sext i32 %29 to i64
  %2196 = sext i32 %116 to i64
  %2197 = mul nsw i64 %2195, %const_reg_qword9, !spirv.Decorations !836
  %2198 = add nsw i64 %2197, %2196, !spirv.Decorations !836
  %2199 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !843
  br i1 %42, label %2203, label %2200

2200:                                             ; preds = %2194
  %2201 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2198
  %2202 = addrspacecast float addrspace(4)* %2201 to float addrspace(1)*
  store float %2199, float addrspace(1)* %2202, align 4
  br label %._crit_edge70.9

2203:                                             ; preds = %2194
  %2204 = mul nsw i64 %2195, %const_reg_qword7, !spirv.Decorations !836
  %2205 = getelementptr float, float addrspace(4)* %160, i64 %2204
  %2206 = getelementptr float, float addrspace(4)* %2205, i64 %2196
  %2207 = addrspacecast float addrspace(4)* %2206 to float addrspace(1)*
  %2208 = load float, float addrspace(1)* %2207, align 4
  %2209 = fmul reassoc nsz arcp contract float %2208, %4, !spirv.Decorations !843
  %2210 = fadd reassoc nsz arcp contract float %2199, %2209, !spirv.Decorations !843
  %2211 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2198
  %2212 = addrspacecast float addrspace(4)* %2211 to float addrspace(1)*
  store float %2210, float addrspace(1)* %2212, align 4
  br label %._crit_edge70.9

._crit_edge70.9:                                  ; preds = %.preheader1.8, %2203, %2200
  br i1 %119, label %2213, label %._crit_edge70.1.9

2213:                                             ; preds = %._crit_edge70.9
  %2214 = sext i32 %59 to i64
  %2215 = sext i32 %116 to i64
  %2216 = mul nsw i64 %2214, %const_reg_qword9, !spirv.Decorations !836
  %2217 = add nsw i64 %2216, %2215, !spirv.Decorations !836
  %2218 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !843
  br i1 %42, label %2222, label %2219

2219:                                             ; preds = %2213
  %2220 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2217
  %2221 = addrspacecast float addrspace(4)* %2220 to float addrspace(1)*
  store float %2218, float addrspace(1)* %2221, align 4
  br label %._crit_edge70.1.9

2222:                                             ; preds = %2213
  %2223 = mul nsw i64 %2214, %const_reg_qword7, !spirv.Decorations !836
  %2224 = getelementptr float, float addrspace(4)* %160, i64 %2223
  %2225 = getelementptr float, float addrspace(4)* %2224, i64 %2215
  %2226 = addrspacecast float addrspace(4)* %2225 to float addrspace(1)*
  %2227 = load float, float addrspace(1)* %2226, align 4
  %2228 = fmul reassoc nsz arcp contract float %2227, %4, !spirv.Decorations !843
  %2229 = fadd reassoc nsz arcp contract float %2218, %2228, !spirv.Decorations !843
  %2230 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2217
  %2231 = addrspacecast float addrspace(4)* %2230 to float addrspace(1)*
  store float %2229, float addrspace(1)* %2231, align 4
  br label %._crit_edge70.1.9

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9, %2222, %2219
  br i1 %120, label %2232, label %._crit_edge70.2.9

2232:                                             ; preds = %._crit_edge70.1.9
  %2233 = sext i32 %62 to i64
  %2234 = sext i32 %116 to i64
  %2235 = mul nsw i64 %2233, %const_reg_qword9, !spirv.Decorations !836
  %2236 = add nsw i64 %2235, %2234, !spirv.Decorations !836
  %2237 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !843
  br i1 %42, label %2241, label %2238

2238:                                             ; preds = %2232
  %2239 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2236
  %2240 = addrspacecast float addrspace(4)* %2239 to float addrspace(1)*
  store float %2237, float addrspace(1)* %2240, align 4
  br label %._crit_edge70.2.9

2241:                                             ; preds = %2232
  %2242 = mul nsw i64 %2233, %const_reg_qword7, !spirv.Decorations !836
  %2243 = getelementptr float, float addrspace(4)* %160, i64 %2242
  %2244 = getelementptr float, float addrspace(4)* %2243, i64 %2234
  %2245 = addrspacecast float addrspace(4)* %2244 to float addrspace(1)*
  %2246 = load float, float addrspace(1)* %2245, align 4
  %2247 = fmul reassoc nsz arcp contract float %2246, %4, !spirv.Decorations !843
  %2248 = fadd reassoc nsz arcp contract float %2237, %2247, !spirv.Decorations !843
  %2249 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2236
  %2250 = addrspacecast float addrspace(4)* %2249 to float addrspace(1)*
  store float %2248, float addrspace(1)* %2250, align 4
  br label %._crit_edge70.2.9

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9, %2241, %2238
  br i1 %121, label %2251, label %.preheader1.9

2251:                                             ; preds = %._crit_edge70.2.9
  %2252 = sext i32 %65 to i64
  %2253 = sext i32 %116 to i64
  %2254 = mul nsw i64 %2252, %const_reg_qword9, !spirv.Decorations !836
  %2255 = add nsw i64 %2254, %2253, !spirv.Decorations !836
  %2256 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !843
  br i1 %42, label %2260, label %2257

2257:                                             ; preds = %2251
  %2258 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2255
  %2259 = addrspacecast float addrspace(4)* %2258 to float addrspace(1)*
  store float %2256, float addrspace(1)* %2259, align 4
  br label %.preheader1.9

2260:                                             ; preds = %2251
  %2261 = mul nsw i64 %2252, %const_reg_qword7, !spirv.Decorations !836
  %2262 = getelementptr float, float addrspace(4)* %160, i64 %2261
  %2263 = getelementptr float, float addrspace(4)* %2262, i64 %2253
  %2264 = addrspacecast float addrspace(4)* %2263 to float addrspace(1)*
  %2265 = load float, float addrspace(1)* %2264, align 4
  %2266 = fmul reassoc nsz arcp contract float %2265, %4, !spirv.Decorations !843
  %2267 = fadd reassoc nsz arcp contract float %2256, %2266, !spirv.Decorations !843
  %2268 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2255
  %2269 = addrspacecast float addrspace(4)* %2268 to float addrspace(1)*
  store float %2267, float addrspace(1)* %2269, align 4
  br label %.preheader1.9

.preheader1.9:                                    ; preds = %._crit_edge70.2.9, %2260, %2257
  br i1 %124, label %2270, label %._crit_edge70.10

2270:                                             ; preds = %.preheader1.9
  %2271 = sext i32 %29 to i64
  %2272 = sext i32 %122 to i64
  %2273 = mul nsw i64 %2271, %const_reg_qword9, !spirv.Decorations !836
  %2274 = add nsw i64 %2273, %2272, !spirv.Decorations !836
  %2275 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !843
  br i1 %42, label %2279, label %2276

2276:                                             ; preds = %2270
  %2277 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2274
  %2278 = addrspacecast float addrspace(4)* %2277 to float addrspace(1)*
  store float %2275, float addrspace(1)* %2278, align 4
  br label %._crit_edge70.10

2279:                                             ; preds = %2270
  %2280 = mul nsw i64 %2271, %const_reg_qword7, !spirv.Decorations !836
  %2281 = getelementptr float, float addrspace(4)* %160, i64 %2280
  %2282 = getelementptr float, float addrspace(4)* %2281, i64 %2272
  %2283 = addrspacecast float addrspace(4)* %2282 to float addrspace(1)*
  %2284 = load float, float addrspace(1)* %2283, align 4
  %2285 = fmul reassoc nsz arcp contract float %2284, %4, !spirv.Decorations !843
  %2286 = fadd reassoc nsz arcp contract float %2275, %2285, !spirv.Decorations !843
  %2287 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2274
  %2288 = addrspacecast float addrspace(4)* %2287 to float addrspace(1)*
  store float %2286, float addrspace(1)* %2288, align 4
  br label %._crit_edge70.10

._crit_edge70.10:                                 ; preds = %.preheader1.9, %2279, %2276
  br i1 %125, label %2289, label %._crit_edge70.1.10

2289:                                             ; preds = %._crit_edge70.10
  %2290 = sext i32 %59 to i64
  %2291 = sext i32 %122 to i64
  %2292 = mul nsw i64 %2290, %const_reg_qword9, !spirv.Decorations !836
  %2293 = add nsw i64 %2292, %2291, !spirv.Decorations !836
  %2294 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !843
  br i1 %42, label %2298, label %2295

2295:                                             ; preds = %2289
  %2296 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2293
  %2297 = addrspacecast float addrspace(4)* %2296 to float addrspace(1)*
  store float %2294, float addrspace(1)* %2297, align 4
  br label %._crit_edge70.1.10

2298:                                             ; preds = %2289
  %2299 = mul nsw i64 %2290, %const_reg_qword7, !spirv.Decorations !836
  %2300 = getelementptr float, float addrspace(4)* %160, i64 %2299
  %2301 = getelementptr float, float addrspace(4)* %2300, i64 %2291
  %2302 = addrspacecast float addrspace(4)* %2301 to float addrspace(1)*
  %2303 = load float, float addrspace(1)* %2302, align 4
  %2304 = fmul reassoc nsz arcp contract float %2303, %4, !spirv.Decorations !843
  %2305 = fadd reassoc nsz arcp contract float %2294, %2304, !spirv.Decorations !843
  %2306 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2293
  %2307 = addrspacecast float addrspace(4)* %2306 to float addrspace(1)*
  store float %2305, float addrspace(1)* %2307, align 4
  br label %._crit_edge70.1.10

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10, %2298, %2295
  br i1 %126, label %2308, label %._crit_edge70.2.10

2308:                                             ; preds = %._crit_edge70.1.10
  %2309 = sext i32 %62 to i64
  %2310 = sext i32 %122 to i64
  %2311 = mul nsw i64 %2309, %const_reg_qword9, !spirv.Decorations !836
  %2312 = add nsw i64 %2311, %2310, !spirv.Decorations !836
  %2313 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !843
  br i1 %42, label %2317, label %2314

2314:                                             ; preds = %2308
  %2315 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2312
  %2316 = addrspacecast float addrspace(4)* %2315 to float addrspace(1)*
  store float %2313, float addrspace(1)* %2316, align 4
  br label %._crit_edge70.2.10

2317:                                             ; preds = %2308
  %2318 = mul nsw i64 %2309, %const_reg_qword7, !spirv.Decorations !836
  %2319 = getelementptr float, float addrspace(4)* %160, i64 %2318
  %2320 = getelementptr float, float addrspace(4)* %2319, i64 %2310
  %2321 = addrspacecast float addrspace(4)* %2320 to float addrspace(1)*
  %2322 = load float, float addrspace(1)* %2321, align 4
  %2323 = fmul reassoc nsz arcp contract float %2322, %4, !spirv.Decorations !843
  %2324 = fadd reassoc nsz arcp contract float %2313, %2323, !spirv.Decorations !843
  %2325 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2312
  %2326 = addrspacecast float addrspace(4)* %2325 to float addrspace(1)*
  store float %2324, float addrspace(1)* %2326, align 4
  br label %._crit_edge70.2.10

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10, %2317, %2314
  br i1 %127, label %2327, label %.preheader1.10

2327:                                             ; preds = %._crit_edge70.2.10
  %2328 = sext i32 %65 to i64
  %2329 = sext i32 %122 to i64
  %2330 = mul nsw i64 %2328, %const_reg_qword9, !spirv.Decorations !836
  %2331 = add nsw i64 %2330, %2329, !spirv.Decorations !836
  %2332 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !843
  br i1 %42, label %2336, label %2333

2333:                                             ; preds = %2327
  %2334 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2331
  %2335 = addrspacecast float addrspace(4)* %2334 to float addrspace(1)*
  store float %2332, float addrspace(1)* %2335, align 4
  br label %.preheader1.10

2336:                                             ; preds = %2327
  %2337 = mul nsw i64 %2328, %const_reg_qword7, !spirv.Decorations !836
  %2338 = getelementptr float, float addrspace(4)* %160, i64 %2337
  %2339 = getelementptr float, float addrspace(4)* %2338, i64 %2329
  %2340 = addrspacecast float addrspace(4)* %2339 to float addrspace(1)*
  %2341 = load float, float addrspace(1)* %2340, align 4
  %2342 = fmul reassoc nsz arcp contract float %2341, %4, !spirv.Decorations !843
  %2343 = fadd reassoc nsz arcp contract float %2332, %2342, !spirv.Decorations !843
  %2344 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2331
  %2345 = addrspacecast float addrspace(4)* %2344 to float addrspace(1)*
  store float %2343, float addrspace(1)* %2345, align 4
  br label %.preheader1.10

.preheader1.10:                                   ; preds = %._crit_edge70.2.10, %2336, %2333
  br i1 %130, label %2346, label %._crit_edge70.11

2346:                                             ; preds = %.preheader1.10
  %2347 = sext i32 %29 to i64
  %2348 = sext i32 %128 to i64
  %2349 = mul nsw i64 %2347, %const_reg_qword9, !spirv.Decorations !836
  %2350 = add nsw i64 %2349, %2348, !spirv.Decorations !836
  %2351 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !843
  br i1 %42, label %2355, label %2352

2352:                                             ; preds = %2346
  %2353 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2350
  %2354 = addrspacecast float addrspace(4)* %2353 to float addrspace(1)*
  store float %2351, float addrspace(1)* %2354, align 4
  br label %._crit_edge70.11

2355:                                             ; preds = %2346
  %2356 = mul nsw i64 %2347, %const_reg_qword7, !spirv.Decorations !836
  %2357 = getelementptr float, float addrspace(4)* %160, i64 %2356
  %2358 = getelementptr float, float addrspace(4)* %2357, i64 %2348
  %2359 = addrspacecast float addrspace(4)* %2358 to float addrspace(1)*
  %2360 = load float, float addrspace(1)* %2359, align 4
  %2361 = fmul reassoc nsz arcp contract float %2360, %4, !spirv.Decorations !843
  %2362 = fadd reassoc nsz arcp contract float %2351, %2361, !spirv.Decorations !843
  %2363 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2350
  %2364 = addrspacecast float addrspace(4)* %2363 to float addrspace(1)*
  store float %2362, float addrspace(1)* %2364, align 4
  br label %._crit_edge70.11

._crit_edge70.11:                                 ; preds = %.preheader1.10, %2355, %2352
  br i1 %131, label %2365, label %._crit_edge70.1.11

2365:                                             ; preds = %._crit_edge70.11
  %2366 = sext i32 %59 to i64
  %2367 = sext i32 %128 to i64
  %2368 = mul nsw i64 %2366, %const_reg_qword9, !spirv.Decorations !836
  %2369 = add nsw i64 %2368, %2367, !spirv.Decorations !836
  %2370 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !843
  br i1 %42, label %2374, label %2371

2371:                                             ; preds = %2365
  %2372 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2369
  %2373 = addrspacecast float addrspace(4)* %2372 to float addrspace(1)*
  store float %2370, float addrspace(1)* %2373, align 4
  br label %._crit_edge70.1.11

2374:                                             ; preds = %2365
  %2375 = mul nsw i64 %2366, %const_reg_qword7, !spirv.Decorations !836
  %2376 = getelementptr float, float addrspace(4)* %160, i64 %2375
  %2377 = getelementptr float, float addrspace(4)* %2376, i64 %2367
  %2378 = addrspacecast float addrspace(4)* %2377 to float addrspace(1)*
  %2379 = load float, float addrspace(1)* %2378, align 4
  %2380 = fmul reassoc nsz arcp contract float %2379, %4, !spirv.Decorations !843
  %2381 = fadd reassoc nsz arcp contract float %2370, %2380, !spirv.Decorations !843
  %2382 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2369
  %2383 = addrspacecast float addrspace(4)* %2382 to float addrspace(1)*
  store float %2381, float addrspace(1)* %2383, align 4
  br label %._crit_edge70.1.11

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11, %2374, %2371
  br i1 %132, label %2384, label %._crit_edge70.2.11

2384:                                             ; preds = %._crit_edge70.1.11
  %2385 = sext i32 %62 to i64
  %2386 = sext i32 %128 to i64
  %2387 = mul nsw i64 %2385, %const_reg_qword9, !spirv.Decorations !836
  %2388 = add nsw i64 %2387, %2386, !spirv.Decorations !836
  %2389 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !843
  br i1 %42, label %2393, label %2390

2390:                                             ; preds = %2384
  %2391 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2388
  %2392 = addrspacecast float addrspace(4)* %2391 to float addrspace(1)*
  store float %2389, float addrspace(1)* %2392, align 4
  br label %._crit_edge70.2.11

2393:                                             ; preds = %2384
  %2394 = mul nsw i64 %2385, %const_reg_qword7, !spirv.Decorations !836
  %2395 = getelementptr float, float addrspace(4)* %160, i64 %2394
  %2396 = getelementptr float, float addrspace(4)* %2395, i64 %2386
  %2397 = addrspacecast float addrspace(4)* %2396 to float addrspace(1)*
  %2398 = load float, float addrspace(1)* %2397, align 4
  %2399 = fmul reassoc nsz arcp contract float %2398, %4, !spirv.Decorations !843
  %2400 = fadd reassoc nsz arcp contract float %2389, %2399, !spirv.Decorations !843
  %2401 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2388
  %2402 = addrspacecast float addrspace(4)* %2401 to float addrspace(1)*
  store float %2400, float addrspace(1)* %2402, align 4
  br label %._crit_edge70.2.11

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11, %2393, %2390
  br i1 %133, label %2403, label %.preheader1.11

2403:                                             ; preds = %._crit_edge70.2.11
  %2404 = sext i32 %65 to i64
  %2405 = sext i32 %128 to i64
  %2406 = mul nsw i64 %2404, %const_reg_qword9, !spirv.Decorations !836
  %2407 = add nsw i64 %2406, %2405, !spirv.Decorations !836
  %2408 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !843
  br i1 %42, label %2412, label %2409

2409:                                             ; preds = %2403
  %2410 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2407
  %2411 = addrspacecast float addrspace(4)* %2410 to float addrspace(1)*
  store float %2408, float addrspace(1)* %2411, align 4
  br label %.preheader1.11

2412:                                             ; preds = %2403
  %2413 = mul nsw i64 %2404, %const_reg_qword7, !spirv.Decorations !836
  %2414 = getelementptr float, float addrspace(4)* %160, i64 %2413
  %2415 = getelementptr float, float addrspace(4)* %2414, i64 %2405
  %2416 = addrspacecast float addrspace(4)* %2415 to float addrspace(1)*
  %2417 = load float, float addrspace(1)* %2416, align 4
  %2418 = fmul reassoc nsz arcp contract float %2417, %4, !spirv.Decorations !843
  %2419 = fadd reassoc nsz arcp contract float %2408, %2418, !spirv.Decorations !843
  %2420 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2407
  %2421 = addrspacecast float addrspace(4)* %2420 to float addrspace(1)*
  store float %2419, float addrspace(1)* %2421, align 4
  br label %.preheader1.11

.preheader1.11:                                   ; preds = %._crit_edge70.2.11, %2412, %2409
  br i1 %136, label %2422, label %._crit_edge70.12

2422:                                             ; preds = %.preheader1.11
  %2423 = sext i32 %29 to i64
  %2424 = sext i32 %134 to i64
  %2425 = mul nsw i64 %2423, %const_reg_qword9, !spirv.Decorations !836
  %2426 = add nsw i64 %2425, %2424, !spirv.Decorations !836
  %2427 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !843
  br i1 %42, label %2431, label %2428

2428:                                             ; preds = %2422
  %2429 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2426
  %2430 = addrspacecast float addrspace(4)* %2429 to float addrspace(1)*
  store float %2427, float addrspace(1)* %2430, align 4
  br label %._crit_edge70.12

2431:                                             ; preds = %2422
  %2432 = mul nsw i64 %2423, %const_reg_qword7, !spirv.Decorations !836
  %2433 = getelementptr float, float addrspace(4)* %160, i64 %2432
  %2434 = getelementptr float, float addrspace(4)* %2433, i64 %2424
  %2435 = addrspacecast float addrspace(4)* %2434 to float addrspace(1)*
  %2436 = load float, float addrspace(1)* %2435, align 4
  %2437 = fmul reassoc nsz arcp contract float %2436, %4, !spirv.Decorations !843
  %2438 = fadd reassoc nsz arcp contract float %2427, %2437, !spirv.Decorations !843
  %2439 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2426
  %2440 = addrspacecast float addrspace(4)* %2439 to float addrspace(1)*
  store float %2438, float addrspace(1)* %2440, align 4
  br label %._crit_edge70.12

._crit_edge70.12:                                 ; preds = %.preheader1.11, %2431, %2428
  br i1 %137, label %2441, label %._crit_edge70.1.12

2441:                                             ; preds = %._crit_edge70.12
  %2442 = sext i32 %59 to i64
  %2443 = sext i32 %134 to i64
  %2444 = mul nsw i64 %2442, %const_reg_qword9, !spirv.Decorations !836
  %2445 = add nsw i64 %2444, %2443, !spirv.Decorations !836
  %2446 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !843
  br i1 %42, label %2450, label %2447

2447:                                             ; preds = %2441
  %2448 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2445
  %2449 = addrspacecast float addrspace(4)* %2448 to float addrspace(1)*
  store float %2446, float addrspace(1)* %2449, align 4
  br label %._crit_edge70.1.12

2450:                                             ; preds = %2441
  %2451 = mul nsw i64 %2442, %const_reg_qword7, !spirv.Decorations !836
  %2452 = getelementptr float, float addrspace(4)* %160, i64 %2451
  %2453 = getelementptr float, float addrspace(4)* %2452, i64 %2443
  %2454 = addrspacecast float addrspace(4)* %2453 to float addrspace(1)*
  %2455 = load float, float addrspace(1)* %2454, align 4
  %2456 = fmul reassoc nsz arcp contract float %2455, %4, !spirv.Decorations !843
  %2457 = fadd reassoc nsz arcp contract float %2446, %2456, !spirv.Decorations !843
  %2458 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2445
  %2459 = addrspacecast float addrspace(4)* %2458 to float addrspace(1)*
  store float %2457, float addrspace(1)* %2459, align 4
  br label %._crit_edge70.1.12

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12, %2450, %2447
  br i1 %138, label %2460, label %._crit_edge70.2.12

2460:                                             ; preds = %._crit_edge70.1.12
  %2461 = sext i32 %62 to i64
  %2462 = sext i32 %134 to i64
  %2463 = mul nsw i64 %2461, %const_reg_qword9, !spirv.Decorations !836
  %2464 = add nsw i64 %2463, %2462, !spirv.Decorations !836
  %2465 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !843
  br i1 %42, label %2469, label %2466

2466:                                             ; preds = %2460
  %2467 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2464
  %2468 = addrspacecast float addrspace(4)* %2467 to float addrspace(1)*
  store float %2465, float addrspace(1)* %2468, align 4
  br label %._crit_edge70.2.12

2469:                                             ; preds = %2460
  %2470 = mul nsw i64 %2461, %const_reg_qword7, !spirv.Decorations !836
  %2471 = getelementptr float, float addrspace(4)* %160, i64 %2470
  %2472 = getelementptr float, float addrspace(4)* %2471, i64 %2462
  %2473 = addrspacecast float addrspace(4)* %2472 to float addrspace(1)*
  %2474 = load float, float addrspace(1)* %2473, align 4
  %2475 = fmul reassoc nsz arcp contract float %2474, %4, !spirv.Decorations !843
  %2476 = fadd reassoc nsz arcp contract float %2465, %2475, !spirv.Decorations !843
  %2477 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2464
  %2478 = addrspacecast float addrspace(4)* %2477 to float addrspace(1)*
  store float %2476, float addrspace(1)* %2478, align 4
  br label %._crit_edge70.2.12

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12, %2469, %2466
  br i1 %139, label %2479, label %.preheader1.12

2479:                                             ; preds = %._crit_edge70.2.12
  %2480 = sext i32 %65 to i64
  %2481 = sext i32 %134 to i64
  %2482 = mul nsw i64 %2480, %const_reg_qword9, !spirv.Decorations !836
  %2483 = add nsw i64 %2482, %2481, !spirv.Decorations !836
  %2484 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !843
  br i1 %42, label %2488, label %2485

2485:                                             ; preds = %2479
  %2486 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2483
  %2487 = addrspacecast float addrspace(4)* %2486 to float addrspace(1)*
  store float %2484, float addrspace(1)* %2487, align 4
  br label %.preheader1.12

2488:                                             ; preds = %2479
  %2489 = mul nsw i64 %2480, %const_reg_qword7, !spirv.Decorations !836
  %2490 = getelementptr float, float addrspace(4)* %160, i64 %2489
  %2491 = getelementptr float, float addrspace(4)* %2490, i64 %2481
  %2492 = addrspacecast float addrspace(4)* %2491 to float addrspace(1)*
  %2493 = load float, float addrspace(1)* %2492, align 4
  %2494 = fmul reassoc nsz arcp contract float %2493, %4, !spirv.Decorations !843
  %2495 = fadd reassoc nsz arcp contract float %2484, %2494, !spirv.Decorations !843
  %2496 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2483
  %2497 = addrspacecast float addrspace(4)* %2496 to float addrspace(1)*
  store float %2495, float addrspace(1)* %2497, align 4
  br label %.preheader1.12

.preheader1.12:                                   ; preds = %._crit_edge70.2.12, %2488, %2485
  br i1 %142, label %2498, label %._crit_edge70.13

2498:                                             ; preds = %.preheader1.12
  %2499 = sext i32 %29 to i64
  %2500 = sext i32 %140 to i64
  %2501 = mul nsw i64 %2499, %const_reg_qword9, !spirv.Decorations !836
  %2502 = add nsw i64 %2501, %2500, !spirv.Decorations !836
  %2503 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !843
  br i1 %42, label %2507, label %2504

2504:                                             ; preds = %2498
  %2505 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2502
  %2506 = addrspacecast float addrspace(4)* %2505 to float addrspace(1)*
  store float %2503, float addrspace(1)* %2506, align 4
  br label %._crit_edge70.13

2507:                                             ; preds = %2498
  %2508 = mul nsw i64 %2499, %const_reg_qword7, !spirv.Decorations !836
  %2509 = getelementptr float, float addrspace(4)* %160, i64 %2508
  %2510 = getelementptr float, float addrspace(4)* %2509, i64 %2500
  %2511 = addrspacecast float addrspace(4)* %2510 to float addrspace(1)*
  %2512 = load float, float addrspace(1)* %2511, align 4
  %2513 = fmul reassoc nsz arcp contract float %2512, %4, !spirv.Decorations !843
  %2514 = fadd reassoc nsz arcp contract float %2503, %2513, !spirv.Decorations !843
  %2515 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2502
  %2516 = addrspacecast float addrspace(4)* %2515 to float addrspace(1)*
  store float %2514, float addrspace(1)* %2516, align 4
  br label %._crit_edge70.13

._crit_edge70.13:                                 ; preds = %.preheader1.12, %2507, %2504
  br i1 %143, label %2517, label %._crit_edge70.1.13

2517:                                             ; preds = %._crit_edge70.13
  %2518 = sext i32 %59 to i64
  %2519 = sext i32 %140 to i64
  %2520 = mul nsw i64 %2518, %const_reg_qword9, !spirv.Decorations !836
  %2521 = add nsw i64 %2520, %2519, !spirv.Decorations !836
  %2522 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !843
  br i1 %42, label %2526, label %2523

2523:                                             ; preds = %2517
  %2524 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2521
  %2525 = addrspacecast float addrspace(4)* %2524 to float addrspace(1)*
  store float %2522, float addrspace(1)* %2525, align 4
  br label %._crit_edge70.1.13

2526:                                             ; preds = %2517
  %2527 = mul nsw i64 %2518, %const_reg_qword7, !spirv.Decorations !836
  %2528 = getelementptr float, float addrspace(4)* %160, i64 %2527
  %2529 = getelementptr float, float addrspace(4)* %2528, i64 %2519
  %2530 = addrspacecast float addrspace(4)* %2529 to float addrspace(1)*
  %2531 = load float, float addrspace(1)* %2530, align 4
  %2532 = fmul reassoc nsz arcp contract float %2531, %4, !spirv.Decorations !843
  %2533 = fadd reassoc nsz arcp contract float %2522, %2532, !spirv.Decorations !843
  %2534 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2521
  %2535 = addrspacecast float addrspace(4)* %2534 to float addrspace(1)*
  store float %2533, float addrspace(1)* %2535, align 4
  br label %._crit_edge70.1.13

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13, %2526, %2523
  br i1 %144, label %2536, label %._crit_edge70.2.13

2536:                                             ; preds = %._crit_edge70.1.13
  %2537 = sext i32 %62 to i64
  %2538 = sext i32 %140 to i64
  %2539 = mul nsw i64 %2537, %const_reg_qword9, !spirv.Decorations !836
  %2540 = add nsw i64 %2539, %2538, !spirv.Decorations !836
  %2541 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !843
  br i1 %42, label %2545, label %2542

2542:                                             ; preds = %2536
  %2543 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2540
  %2544 = addrspacecast float addrspace(4)* %2543 to float addrspace(1)*
  store float %2541, float addrspace(1)* %2544, align 4
  br label %._crit_edge70.2.13

2545:                                             ; preds = %2536
  %2546 = mul nsw i64 %2537, %const_reg_qword7, !spirv.Decorations !836
  %2547 = getelementptr float, float addrspace(4)* %160, i64 %2546
  %2548 = getelementptr float, float addrspace(4)* %2547, i64 %2538
  %2549 = addrspacecast float addrspace(4)* %2548 to float addrspace(1)*
  %2550 = load float, float addrspace(1)* %2549, align 4
  %2551 = fmul reassoc nsz arcp contract float %2550, %4, !spirv.Decorations !843
  %2552 = fadd reassoc nsz arcp contract float %2541, %2551, !spirv.Decorations !843
  %2553 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2540
  %2554 = addrspacecast float addrspace(4)* %2553 to float addrspace(1)*
  store float %2552, float addrspace(1)* %2554, align 4
  br label %._crit_edge70.2.13

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13, %2545, %2542
  br i1 %145, label %2555, label %.preheader1.13

2555:                                             ; preds = %._crit_edge70.2.13
  %2556 = sext i32 %65 to i64
  %2557 = sext i32 %140 to i64
  %2558 = mul nsw i64 %2556, %const_reg_qword9, !spirv.Decorations !836
  %2559 = add nsw i64 %2558, %2557, !spirv.Decorations !836
  %2560 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !843
  br i1 %42, label %2564, label %2561

2561:                                             ; preds = %2555
  %2562 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2559
  %2563 = addrspacecast float addrspace(4)* %2562 to float addrspace(1)*
  store float %2560, float addrspace(1)* %2563, align 4
  br label %.preheader1.13

2564:                                             ; preds = %2555
  %2565 = mul nsw i64 %2556, %const_reg_qword7, !spirv.Decorations !836
  %2566 = getelementptr float, float addrspace(4)* %160, i64 %2565
  %2567 = getelementptr float, float addrspace(4)* %2566, i64 %2557
  %2568 = addrspacecast float addrspace(4)* %2567 to float addrspace(1)*
  %2569 = load float, float addrspace(1)* %2568, align 4
  %2570 = fmul reassoc nsz arcp contract float %2569, %4, !spirv.Decorations !843
  %2571 = fadd reassoc nsz arcp contract float %2560, %2570, !spirv.Decorations !843
  %2572 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2559
  %2573 = addrspacecast float addrspace(4)* %2572 to float addrspace(1)*
  store float %2571, float addrspace(1)* %2573, align 4
  br label %.preheader1.13

.preheader1.13:                                   ; preds = %._crit_edge70.2.13, %2564, %2561
  br i1 %148, label %2574, label %._crit_edge70.14

2574:                                             ; preds = %.preheader1.13
  %2575 = sext i32 %29 to i64
  %2576 = sext i32 %146 to i64
  %2577 = mul nsw i64 %2575, %const_reg_qword9, !spirv.Decorations !836
  %2578 = add nsw i64 %2577, %2576, !spirv.Decorations !836
  %2579 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !843
  br i1 %42, label %2583, label %2580

2580:                                             ; preds = %2574
  %2581 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2578
  %2582 = addrspacecast float addrspace(4)* %2581 to float addrspace(1)*
  store float %2579, float addrspace(1)* %2582, align 4
  br label %._crit_edge70.14

2583:                                             ; preds = %2574
  %2584 = mul nsw i64 %2575, %const_reg_qword7, !spirv.Decorations !836
  %2585 = getelementptr float, float addrspace(4)* %160, i64 %2584
  %2586 = getelementptr float, float addrspace(4)* %2585, i64 %2576
  %2587 = addrspacecast float addrspace(4)* %2586 to float addrspace(1)*
  %2588 = load float, float addrspace(1)* %2587, align 4
  %2589 = fmul reassoc nsz arcp contract float %2588, %4, !spirv.Decorations !843
  %2590 = fadd reassoc nsz arcp contract float %2579, %2589, !spirv.Decorations !843
  %2591 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2578
  %2592 = addrspacecast float addrspace(4)* %2591 to float addrspace(1)*
  store float %2590, float addrspace(1)* %2592, align 4
  br label %._crit_edge70.14

._crit_edge70.14:                                 ; preds = %.preheader1.13, %2583, %2580
  br i1 %149, label %2593, label %._crit_edge70.1.14

2593:                                             ; preds = %._crit_edge70.14
  %2594 = sext i32 %59 to i64
  %2595 = sext i32 %146 to i64
  %2596 = mul nsw i64 %2594, %const_reg_qword9, !spirv.Decorations !836
  %2597 = add nsw i64 %2596, %2595, !spirv.Decorations !836
  %2598 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !843
  br i1 %42, label %2602, label %2599

2599:                                             ; preds = %2593
  %2600 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2597
  %2601 = addrspacecast float addrspace(4)* %2600 to float addrspace(1)*
  store float %2598, float addrspace(1)* %2601, align 4
  br label %._crit_edge70.1.14

2602:                                             ; preds = %2593
  %2603 = mul nsw i64 %2594, %const_reg_qword7, !spirv.Decorations !836
  %2604 = getelementptr float, float addrspace(4)* %160, i64 %2603
  %2605 = getelementptr float, float addrspace(4)* %2604, i64 %2595
  %2606 = addrspacecast float addrspace(4)* %2605 to float addrspace(1)*
  %2607 = load float, float addrspace(1)* %2606, align 4
  %2608 = fmul reassoc nsz arcp contract float %2607, %4, !spirv.Decorations !843
  %2609 = fadd reassoc nsz arcp contract float %2598, %2608, !spirv.Decorations !843
  %2610 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2597
  %2611 = addrspacecast float addrspace(4)* %2610 to float addrspace(1)*
  store float %2609, float addrspace(1)* %2611, align 4
  br label %._crit_edge70.1.14

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14, %2602, %2599
  br i1 %150, label %2612, label %._crit_edge70.2.14

2612:                                             ; preds = %._crit_edge70.1.14
  %2613 = sext i32 %62 to i64
  %2614 = sext i32 %146 to i64
  %2615 = mul nsw i64 %2613, %const_reg_qword9, !spirv.Decorations !836
  %2616 = add nsw i64 %2615, %2614, !spirv.Decorations !836
  %2617 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !843
  br i1 %42, label %2621, label %2618

2618:                                             ; preds = %2612
  %2619 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2616
  %2620 = addrspacecast float addrspace(4)* %2619 to float addrspace(1)*
  store float %2617, float addrspace(1)* %2620, align 4
  br label %._crit_edge70.2.14

2621:                                             ; preds = %2612
  %2622 = mul nsw i64 %2613, %const_reg_qword7, !spirv.Decorations !836
  %2623 = getelementptr float, float addrspace(4)* %160, i64 %2622
  %2624 = getelementptr float, float addrspace(4)* %2623, i64 %2614
  %2625 = addrspacecast float addrspace(4)* %2624 to float addrspace(1)*
  %2626 = load float, float addrspace(1)* %2625, align 4
  %2627 = fmul reassoc nsz arcp contract float %2626, %4, !spirv.Decorations !843
  %2628 = fadd reassoc nsz arcp contract float %2617, %2627, !spirv.Decorations !843
  %2629 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2616
  %2630 = addrspacecast float addrspace(4)* %2629 to float addrspace(1)*
  store float %2628, float addrspace(1)* %2630, align 4
  br label %._crit_edge70.2.14

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14, %2621, %2618
  br i1 %151, label %2631, label %.preheader1.14

2631:                                             ; preds = %._crit_edge70.2.14
  %2632 = sext i32 %65 to i64
  %2633 = sext i32 %146 to i64
  %2634 = mul nsw i64 %2632, %const_reg_qword9, !spirv.Decorations !836
  %2635 = add nsw i64 %2634, %2633, !spirv.Decorations !836
  %2636 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !843
  br i1 %42, label %2640, label %2637

2637:                                             ; preds = %2631
  %2638 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2635
  %2639 = addrspacecast float addrspace(4)* %2638 to float addrspace(1)*
  store float %2636, float addrspace(1)* %2639, align 4
  br label %.preheader1.14

2640:                                             ; preds = %2631
  %2641 = mul nsw i64 %2632, %const_reg_qword7, !spirv.Decorations !836
  %2642 = getelementptr float, float addrspace(4)* %160, i64 %2641
  %2643 = getelementptr float, float addrspace(4)* %2642, i64 %2633
  %2644 = addrspacecast float addrspace(4)* %2643 to float addrspace(1)*
  %2645 = load float, float addrspace(1)* %2644, align 4
  %2646 = fmul reassoc nsz arcp contract float %2645, %4, !spirv.Decorations !843
  %2647 = fadd reassoc nsz arcp contract float %2636, %2646, !spirv.Decorations !843
  %2648 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2635
  %2649 = addrspacecast float addrspace(4)* %2648 to float addrspace(1)*
  store float %2647, float addrspace(1)* %2649, align 4
  br label %.preheader1.14

.preheader1.14:                                   ; preds = %._crit_edge70.2.14, %2640, %2637
  br i1 %154, label %2650, label %._crit_edge70.15

2650:                                             ; preds = %.preheader1.14
  %2651 = sext i32 %29 to i64
  %2652 = sext i32 %152 to i64
  %2653 = mul nsw i64 %2651, %const_reg_qword9, !spirv.Decorations !836
  %2654 = add nsw i64 %2653, %2652, !spirv.Decorations !836
  %2655 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !843
  br i1 %42, label %2659, label %2656

2656:                                             ; preds = %2650
  %2657 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2654
  %2658 = addrspacecast float addrspace(4)* %2657 to float addrspace(1)*
  store float %2655, float addrspace(1)* %2658, align 4
  br label %._crit_edge70.15

2659:                                             ; preds = %2650
  %2660 = mul nsw i64 %2651, %const_reg_qword7, !spirv.Decorations !836
  %2661 = getelementptr float, float addrspace(4)* %160, i64 %2660
  %2662 = getelementptr float, float addrspace(4)* %2661, i64 %2652
  %2663 = addrspacecast float addrspace(4)* %2662 to float addrspace(1)*
  %2664 = load float, float addrspace(1)* %2663, align 4
  %2665 = fmul reassoc nsz arcp contract float %2664, %4, !spirv.Decorations !843
  %2666 = fadd reassoc nsz arcp contract float %2655, %2665, !spirv.Decorations !843
  %2667 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2654
  %2668 = addrspacecast float addrspace(4)* %2667 to float addrspace(1)*
  store float %2666, float addrspace(1)* %2668, align 4
  br label %._crit_edge70.15

._crit_edge70.15:                                 ; preds = %.preheader1.14, %2659, %2656
  br i1 %155, label %2669, label %._crit_edge70.1.15

2669:                                             ; preds = %._crit_edge70.15
  %2670 = sext i32 %59 to i64
  %2671 = sext i32 %152 to i64
  %2672 = mul nsw i64 %2670, %const_reg_qword9, !spirv.Decorations !836
  %2673 = add nsw i64 %2672, %2671, !spirv.Decorations !836
  %2674 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !843
  br i1 %42, label %2678, label %2675

2675:                                             ; preds = %2669
  %2676 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2673
  %2677 = addrspacecast float addrspace(4)* %2676 to float addrspace(1)*
  store float %2674, float addrspace(1)* %2677, align 4
  br label %._crit_edge70.1.15

2678:                                             ; preds = %2669
  %2679 = mul nsw i64 %2670, %const_reg_qword7, !spirv.Decorations !836
  %2680 = getelementptr float, float addrspace(4)* %160, i64 %2679
  %2681 = getelementptr float, float addrspace(4)* %2680, i64 %2671
  %2682 = addrspacecast float addrspace(4)* %2681 to float addrspace(1)*
  %2683 = load float, float addrspace(1)* %2682, align 4
  %2684 = fmul reassoc nsz arcp contract float %2683, %4, !spirv.Decorations !843
  %2685 = fadd reassoc nsz arcp contract float %2674, %2684, !spirv.Decorations !843
  %2686 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2673
  %2687 = addrspacecast float addrspace(4)* %2686 to float addrspace(1)*
  store float %2685, float addrspace(1)* %2687, align 4
  br label %._crit_edge70.1.15

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15, %2678, %2675
  br i1 %156, label %2688, label %._crit_edge70.2.15

2688:                                             ; preds = %._crit_edge70.1.15
  %2689 = sext i32 %62 to i64
  %2690 = sext i32 %152 to i64
  %2691 = mul nsw i64 %2689, %const_reg_qword9, !spirv.Decorations !836
  %2692 = add nsw i64 %2691, %2690, !spirv.Decorations !836
  %2693 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !843
  br i1 %42, label %2697, label %2694

2694:                                             ; preds = %2688
  %2695 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2692
  %2696 = addrspacecast float addrspace(4)* %2695 to float addrspace(1)*
  store float %2693, float addrspace(1)* %2696, align 4
  br label %._crit_edge70.2.15

2697:                                             ; preds = %2688
  %2698 = mul nsw i64 %2689, %const_reg_qword7, !spirv.Decorations !836
  %2699 = getelementptr float, float addrspace(4)* %160, i64 %2698
  %2700 = getelementptr float, float addrspace(4)* %2699, i64 %2690
  %2701 = addrspacecast float addrspace(4)* %2700 to float addrspace(1)*
  %2702 = load float, float addrspace(1)* %2701, align 4
  %2703 = fmul reassoc nsz arcp contract float %2702, %4, !spirv.Decorations !843
  %2704 = fadd reassoc nsz arcp contract float %2693, %2703, !spirv.Decorations !843
  %2705 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2692
  %2706 = addrspacecast float addrspace(4)* %2705 to float addrspace(1)*
  store float %2704, float addrspace(1)* %2706, align 4
  br label %._crit_edge70.2.15

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15, %2697, %2694
  br i1 %157, label %2707, label %.preheader1.15

2707:                                             ; preds = %._crit_edge70.2.15
  %2708 = sext i32 %65 to i64
  %2709 = sext i32 %152 to i64
  %2710 = mul nsw i64 %2708, %const_reg_qword9, !spirv.Decorations !836
  %2711 = add nsw i64 %2710, %2709, !spirv.Decorations !836
  %2712 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !843
  br i1 %42, label %2716, label %2713

2713:                                             ; preds = %2707
  %2714 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2711
  %2715 = addrspacecast float addrspace(4)* %2714 to float addrspace(1)*
  store float %2712, float addrspace(1)* %2715, align 4
  br label %.preheader1.15

2716:                                             ; preds = %2707
  %2717 = mul nsw i64 %2708, %const_reg_qword7, !spirv.Decorations !836
  %2718 = getelementptr float, float addrspace(4)* %160, i64 %2717
  %2719 = getelementptr float, float addrspace(4)* %2718, i64 %2709
  %2720 = addrspacecast float addrspace(4)* %2719 to float addrspace(1)*
  %2721 = load float, float addrspace(1)* %2720, align 4
  %2722 = fmul reassoc nsz arcp contract float %2721, %4, !spirv.Decorations !843
  %2723 = fadd reassoc nsz arcp contract float %2712, %2722, !spirv.Decorations !843
  %2724 = getelementptr inbounds float, float addrspace(4)* %159, i64 %2711
  %2725 = addrspacecast float addrspace(4)* %2724 to float addrspace(1)*
  store float %2723, float addrspace(1)* %2725, align 4
  br label %.preheader1.15

.preheader1.15:                                   ; preds = %._crit_edge70.2.15, %2716, %2713
  %2726 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %162, i64 %52
  %2727 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %161, i64 %53
  %.idx = select i1 %42, i64 %54, i64 0
  %2728 = getelementptr float, float addrspace(4)* %160, i64 %.idx
  %2729 = getelementptr inbounds float, float addrspace(4)* %159, i64 %55
  %2730 = add i32 %158, %14
  %2731 = icmp slt i32 %2730, %8
  br i1 %2731, label %.preheader2.preheader, label %._crit_edge72

._crit_edge72:                                    ; preds = %.preheader1.15, %13
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

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.fshl.i64(i64, i64, i64) #6

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
!opencl.ocl.version = !{!833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833}
!opencl.spir.version = !{!833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833, !833}
!llvm.ident = !{!834, !834, !834, !834, !834, !834, !834, !834, !834, !834, !834, !834, !834}
!llvm.module.flags = !{!835}

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
!136 = !{!"ModuleMD", !137, !138, !276, !592, !623, !645, !646, !650, !653, !654, !655, !694, !719, !733, !734, !735, !752, !753, !754, !755, !759, !760, !768, !769, !770, !771, !772, !773, !774, !775, !776, !777, !778, !783, !785, !789, !790, !791, !792, !793, !794, !795, !796, !797, !798, !799, !800, !801, !802, !803, !804, !805, !806, !807, !808, !356, !809, !810, !811, !813, !815, !818, !819, !820, !822, !823, !824, !829, !830, !831, !832}
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
!276 = !{!"FuncMD", !277, !278, !393, !394, !431, !432, !443, !444, !454, !455, !462, !463, !470, !471, !478, !479, !484, !485, !494, !495, !573, !574, !575, !576, !577, !578, !579, !580}
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
!389 = !{!"m_OptsToDisablePerFunc", !390, !391, !392}
!390 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!391 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!392 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!393 = !{!"FuncMDMap[1]", void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail}
!394 = !{!"FuncMDValue[1]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !314, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !395, !401, !406, !413, !420, !425, !430, !389}
!395 = !{!"m_OpenCLArgAddressSpaces", !396, !370, !397, !398, !399, !400}
!396 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!397 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!398 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!399 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 0}
!400 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!401 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !402, !403, !404, !405}
!402 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!403 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!404 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!405 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!406 = !{!"m_OpenCLArgTypes", !407, !408, !409, !410, !411, !412}
!407 = !{!"m_OpenCLArgTypesVec[0]", !"char*"}
!408 = !{!"m_OpenCLArgTypesVec[1]", !"long"}
!409 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!410 = !{!"m_OpenCLArgTypesVec[3]", !"char*"}
!411 = !{!"m_OpenCLArgTypesVec[4]", !"long"}
!412 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!413 = !{!"m_OpenCLArgBaseTypes", !414, !415, !416, !417, !418, !419}
!414 = !{!"m_OpenCLArgBaseTypesVec[0]", !"char*"}
!415 = !{!"m_OpenCLArgBaseTypesVec[1]", !"long"}
!416 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!417 = !{!"m_OpenCLArgBaseTypesVec[3]", !"char*"}
!418 = !{!"m_OpenCLArgBaseTypesVec[4]", !"long"}
!419 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!420 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !421, !422, !423, !424}
!421 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!422 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!423 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!424 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!425 = !{!"m_OpenCLArgNames", !384, !385, !426, !427, !428, !429}
!426 = !{!"m_OpenCLArgNamesVec[2]", !""}
!427 = !{!"m_OpenCLArgNamesVec[3]", !""}
!428 = !{!"m_OpenCLArgNamesVec[4]", !""}
!429 = !{!"m_OpenCLArgNamesVec[5]", !""}
!430 = !{!"m_OpenCLArgScalarAsPointers"}
!431 = !{!"FuncMDMap[2]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!432 = !{!"FuncMDValue[2]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !433, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !435, !438, !380, !383, !441, !389}
!433 = !{!"resAllocMD", !315, !316, !317, !434, !347}
!434 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342}
!435 = !{!"m_OpenCLArgTypes", !436, !437}
!436 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::range.0"}
!437 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.2"}
!438 = !{!"m_OpenCLArgBaseTypes", !439, !440}
!439 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::range.0"}
!440 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.2"}
!441 = !{!"m_OpenCLArgScalarAsPointers", !442}
!442 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 12}
!443 = !{!"FuncMDMap[3]", void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!444 = !{!"FuncMDValue[3]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !445, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !447, !371, !448, !451, !380, !383, !430, !389}
!445 = !{!"resAllocMD", !315, !316, !317, !446, !347}
!446 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333}
!447 = !{!"m_OpenCLArgAddressSpaces", !396, !370}
!448 = !{!"m_OpenCLArgTypes", !449, !450}
!449 = !{!"m_OpenCLArgTypesVec[0]", !"short*"}
!450 = !{!"m_OpenCLArgTypesVec[1]", !"ushort"}
!451 = !{!"m_OpenCLArgBaseTypes", !452, !453}
!452 = !{!"m_OpenCLArgBaseTypesVec[0]", !"short*"}
!453 = !{!"m_OpenCLArgBaseTypesVec[1]", !"ushort"}
!454 = !{!"FuncMDMap[4]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!455 = !{!"FuncMDValue[4]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !456, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !458, !460, !380, !383, !441, !389}
!456 = !{!"resAllocMD", !315, !316, !317, !457, !347}
!457 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340}
!458 = !{!"m_OpenCLArgTypes", !436, !459}
!459 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.9"}
!460 = !{!"m_OpenCLArgBaseTypes", !439, !461}
!461 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.9"}
!462 = !{!"FuncMDMap[5]", void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!463 = !{!"FuncMDValue[5]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !445, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !447, !371, !464, !467, !380, !383, !430, !389}
!464 = !{!"m_OpenCLArgTypes", !465, !466}
!465 = !{!"m_OpenCLArgTypesVec[0]", !"int*"}
!466 = !{!"m_OpenCLArgTypesVec[1]", !"int"}
!467 = !{!"m_OpenCLArgBaseTypes", !468, !469}
!468 = !{!"m_OpenCLArgBaseTypesVec[0]", !"int*"}
!469 = !{!"m_OpenCLArgBaseTypesVec[1]", !"int"}
!470 = !{!"FuncMDMap[6]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!471 = !{!"FuncMDValue[6]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !472, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !371, !474, !476, !380, !383, !441, !389}
!472 = !{!"resAllocMD", !315, !316, !317, !473, !347}
!473 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343}
!474 = !{!"m_OpenCLArgTypes", !436, !475}
!475 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.12"}
!476 = !{!"m_OpenCLArgBaseTypes", !439, !477}
!477 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.12"}
!478 = !{!"FuncMDMap[7]", void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!479 = !{!"FuncMDValue[7]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !445, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !447, !371, !480, !482, !380, !383, !430, !389}
!480 = !{!"m_OpenCLArgTypes", !407, !481}
!481 = !{!"m_OpenCLArgTypesVec[1]", !"uchar"}
!482 = !{!"m_OpenCLArgBaseTypes", !414, !483}
!483 = !{!"m_OpenCLArgBaseTypesVec[1]", !"uchar"}
!484 = !{!"FuncMDMap[8]", void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE}
!485 = !{!"FuncMDValue[8]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !314, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !486, !487, !488, !490, !492, !493, !430, !389}
!486 = !{!"m_OpenCLArgAddressSpaces", !396, !370, !397}
!487 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !402}
!488 = !{!"m_OpenCLArgTypes", !449, !408, !489}
!489 = !{!"m_OpenCLArgTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!490 = !{!"m_OpenCLArgBaseTypes", !452, !415, !491}
!491 = !{!"m_OpenCLArgBaseTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!492 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !421}
!493 = !{!"m_OpenCLArgNames", !384, !385, !426}
!494 = !{!"FuncMDMap[9]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE}
!495 = !{!"FuncMDValue[9]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !496, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !507, !516, !524, !538, !552, !560, !568, !389}
!496 = !{!"resAllocMD", !315, !316, !317, !497, !347}
!497 = !{!"argAllocMDList", !319, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !498, !499, !500, !501, !502, !503, !504, !505, !506}
!498 = !{!"argAllocMDListVec[25]", !320, !321, !322}
!499 = !{!"argAllocMDListVec[26]", !320, !321, !322}
!500 = !{!"argAllocMDListVec[27]", !320, !321, !322}
!501 = !{!"argAllocMDListVec[28]", !320, !321, !322}
!502 = !{!"argAllocMDListVec[29]", !320, !321, !322}
!503 = !{!"argAllocMDListVec[30]", !320, !321, !322}
!504 = !{!"argAllocMDListVec[31]", !320, !321, !322}
!505 = !{!"argAllocMDListVec[32]", !320, !321, !322}
!506 = !{!"argAllocMDListVec[33]", !320, !321, !322}
!507 = !{!"m_OpenCLArgAddressSpaces", !369, !370, !397, !508, !399, !400, !509, !510, !511, !512, !513, !514, !515}
!508 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!509 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!510 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 0}
!511 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!512 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!513 = !{!"m_OpenCLArgAddressSpacesVec[10]", i32 0}
!514 = !{!"m_OpenCLArgAddressSpacesVec[11]", i32 0}
!515 = !{!"m_OpenCLArgAddressSpacesVec[12]", i32 0}
!516 = !{!"m_OpenCLArgAccessQualifiers", !372, !373, !402, !403, !404, !405, !517, !518, !519, !520, !521, !522, !523}
!517 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!518 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!519 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!520 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!521 = !{!"m_OpenCLArgAccessQualifiersVec[10]", !"none"}
!522 = !{!"m_OpenCLArgAccessQualifiersVec[11]", !"none"}
!523 = !{!"m_OpenCLArgAccessQualifiersVec[12]", !"none"}
!524 = !{!"m_OpenCLArgTypes", !525, !526, !527, !528, !529, !530, !531, !532, !533, !534, !535, !536, !537}
!525 = !{!"m_OpenCLArgTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!526 = !{!"m_OpenCLArgTypesVec[1]", !"float"}
!527 = !{!"m_OpenCLArgTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!528 = !{!"m_OpenCLArgTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!529 = !{!"m_OpenCLArgTypesVec[4]", !"float"}
!530 = !{!"m_OpenCLArgTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!531 = !{!"m_OpenCLArgTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!532 = !{!"m_OpenCLArgTypesVec[7]", !"float"}
!533 = !{!"m_OpenCLArgTypesVec[8]", !"int"}
!534 = !{!"m_OpenCLArgTypesVec[9]", !"long"}
!535 = !{!"m_OpenCLArgTypesVec[10]", !"long"}
!536 = !{!"m_OpenCLArgTypesVec[11]", !"long"}
!537 = !{!"m_OpenCLArgTypesVec[12]", !"long"}
!538 = !{!"m_OpenCLArgBaseTypes", !539, !540, !541, !542, !543, !544, !545, !546, !547, !548, !549, !550, !551}
!539 = !{!"m_OpenCLArgBaseTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!540 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float"}
!541 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!542 = !{!"m_OpenCLArgBaseTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!543 = !{!"m_OpenCLArgBaseTypesVec[4]", !"float"}
!544 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!545 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!546 = !{!"m_OpenCLArgBaseTypesVec[7]", !"float"}
!547 = !{!"m_OpenCLArgBaseTypesVec[8]", !"int"}
!548 = !{!"m_OpenCLArgBaseTypesVec[9]", !"long"}
!549 = !{!"m_OpenCLArgBaseTypesVec[10]", !"long"}
!550 = !{!"m_OpenCLArgBaseTypesVec[11]", !"long"}
!551 = !{!"m_OpenCLArgBaseTypesVec[12]", !"long"}
!552 = !{!"m_OpenCLArgTypeQualifiers", !381, !382, !421, !422, !423, !424, !553, !554, !555, !556, !557, !558, !559}
!553 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!554 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!555 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!556 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!557 = !{!"m_OpenCLArgTypeQualifiersVec[10]", !""}
!558 = !{!"m_OpenCLArgTypeQualifiersVec[11]", !""}
!559 = !{!"m_OpenCLArgTypeQualifiersVec[12]", !""}
!560 = !{!"m_OpenCLArgNames", !384, !385, !426, !427, !428, !429, !561, !562, !563, !564, !565, !566, !567}
!561 = !{!"m_OpenCLArgNamesVec[6]", !""}
!562 = !{!"m_OpenCLArgNamesVec[7]", !""}
!563 = !{!"m_OpenCLArgNamesVec[8]", !""}
!564 = !{!"m_OpenCLArgNamesVec[9]", !""}
!565 = !{!"m_OpenCLArgNamesVec[10]", !""}
!566 = !{!"m_OpenCLArgNamesVec[11]", !""}
!567 = !{!"m_OpenCLArgNamesVec[12]", !""}
!568 = !{!"m_OpenCLArgScalarAsPointers", !569, !570, !571, !572}
!569 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 25}
!570 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 27}
!571 = !{!"m_OpenCLArgScalarAsPointersSet[2]", i32 29}
!572 = !{!"m_OpenCLArgScalarAsPointersSet[3]", i32 31}
!573 = !{!"FuncMDMap[10]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!574 = !{!"FuncMDValue[10]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !496, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !507, !516, !524, !538, !552, !560, !568, !389}
!575 = !{!"FuncMDMap[11]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE}
!576 = !{!"FuncMDValue[11]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !496, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !507, !516, !524, !538, !552, !560, !568, !389}
!577 = !{!"FuncMDMap[12]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!578 = !{!"FuncMDValue[12]", !279, !280, !284, !285, !286, !287, !288, !289, !290, !496, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !507, !516, !524, !538, !552, !560, !568, !389}
!579 = !{!"FuncMDMap[13]", void ()* @Intel_Symbol_Table_Void_Program}
!580 = !{!"FuncMDValue[13]", !279, !581, !284, !285, !286, !287, !288, !289, !290, !584, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !586, !587, !588, !589, !590, !591, !430, !389}
!581 = !{!"workGroupWalkOrder", !281, !582, !583}
!582 = !{!"dim1", i32 0}
!583 = !{!"dim2", i32 0}
!584 = !{!"resAllocMD", !315, !316, !317, !585, !347}
!585 = !{!"argAllocMDList"}
!586 = !{!"m_OpenCLArgAddressSpaces"}
!587 = !{!"m_OpenCLArgAccessQualifiers"}
!588 = !{!"m_OpenCLArgTypes"}
!589 = !{!"m_OpenCLArgBaseTypes"}
!590 = !{!"m_OpenCLArgTypeQualifiers"}
!591 = !{!"m_OpenCLArgNames"}
!592 = !{!"pushInfo", !593, !594, !595, !599, !600, !601, !602, !603, !604, !605, !606, !619, !620, !621, !622}
!593 = !{!"pushableAddresses"}
!594 = !{!"bindlessPushInfo"}
!595 = !{!"dynamicBufferInfo", !596, !597, !598}
!596 = !{!"firstIndex", i32 0}
!597 = !{!"numOffsets", i32 0}
!598 = !{!"forceDisabled", i1 false}
!599 = !{!"MaxNumberOfPushedBuffers", i32 0}
!600 = !{!"inlineConstantBufferSlot", i32 -1}
!601 = !{!"inlineConstantBufferOffset", i32 -1}
!602 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!603 = !{!"constants"}
!604 = !{!"inputs"}
!605 = !{!"constantReg"}
!606 = !{!"simplePushInfoArr", !607, !616, !617, !618}
!607 = !{!"simplePushInfoArrVec[0]", !608, !609, !610, !611, !612, !613, !614, !615}
!608 = !{!"cbIdx", i32 0}
!609 = !{!"pushableAddressGrfOffset", i32 -1}
!610 = !{!"pushableOffsetGrfOffset", i32 -1}
!611 = !{!"offset", i32 0}
!612 = !{!"size", i32 0}
!613 = !{!"isStateless", i1 false}
!614 = !{!"isBindless", i1 false}
!615 = !{!"simplePushLoads"}
!616 = !{!"simplePushInfoArrVec[1]", !608, !609, !610, !611, !612, !613, !614, !615}
!617 = !{!"simplePushInfoArrVec[2]", !608, !609, !610, !611, !612, !613, !614, !615}
!618 = !{!"simplePushInfoArrVec[3]", !608, !609, !610, !611, !612, !613, !614, !615}
!619 = !{!"simplePushBufferUsed", i32 0}
!620 = !{!"pushAnalysisWIInfos"}
!621 = !{!"inlineRTGlobalPtrOffset", i32 0}
!622 = !{!"rtSyncSurfPtrOffset", i32 0}
!623 = !{!"pISAInfo", !624, !625, !629, !630, !638, !642, !644}
!624 = !{!"shaderType", !"UNKNOWN"}
!625 = !{!"geometryInfo", !626, !627, !628}
!626 = !{!"needsVertexHandles", i1 false}
!627 = !{!"needsPrimitiveIDEnable", i1 false}
!628 = !{!"VertexCount", i32 0}
!629 = !{!"hullInfo", !626, !627}
!630 = !{!"pixelInfo", !631, !632, !633, !634, !635, !636, !637}
!631 = !{!"perPolyStartGrf", i32 0}
!632 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!633 = !{!"hasNonPerspBaryPlanes", i1 false}
!634 = !{!"maxPerPrimConstDataId", i32 -1}
!635 = !{!"maxSetupId", i32 -1}
!636 = !{!"hasVMask", i1 false}
!637 = !{!"PixelGRFBitmask", i32 0}
!638 = !{!"domainInfo", !639, !640, !641}
!639 = !{!"DomainPointUArgIdx", i32 -1}
!640 = !{!"DomainPointVArgIdx", i32 -1}
!641 = !{!"DomainPointWArgIdx", i32 -1}
!642 = !{!"computeInfo", !643}
!643 = !{!"EnableHWGenerateLID", i1 true}
!644 = !{!"URBOutputLength", i32 0}
!645 = !{!"WaEnableICBPromotion", i1 false}
!646 = !{!"vsInfo", !647, !648, !649}
!647 = !{!"DrawIndirectBufferIndex", i32 -1}
!648 = !{!"vertexReordering", i32 -1}
!649 = !{!"MaxNumOfOutputs", i32 0}
!650 = !{!"hsInfo", !651, !652}
!651 = !{!"numPatchAttributesPatchBaseName", !""}
!652 = !{!"numVertexAttributesPatchBaseName", !""}
!653 = !{!"dsInfo", !649}
!654 = !{!"gsInfo", !649}
!655 = !{!"psInfo", !656, !657, !658, !659, !660, !661, !662, !663, !664, !665, !666, !667, !668, !669, !670, !671, !672, !673, !674, !675, !676, !677, !678, !679, !680, !681, !682, !683, !684, !685, !686, !687, !688, !689, !690, !691, !692, !693}
!656 = !{!"BlendStateDisabledMask", i8 0}
!657 = !{!"SkipSrc0Alpha", i1 false}
!658 = !{!"DualSourceBlendingDisabled", i1 false}
!659 = !{!"ForceEnableSimd32", i1 false}
!660 = !{!"DisableSimd32WithDiscard", i1 false}
!661 = !{!"outputDepth", i1 false}
!662 = !{!"outputStencil", i1 false}
!663 = !{!"outputMask", i1 false}
!664 = !{!"blendToFillEnabled", i1 false}
!665 = !{!"forceEarlyZ", i1 false}
!666 = !{!"hasVersionedLoop", i1 false}
!667 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!668 = !{!"requestCPSizeRelevant", i1 false}
!669 = !{!"requestCPSize", i1 false}
!670 = !{!"texelMaskFastClearMode", !"Disabled"}
!671 = !{!"NumSamples", i8 0}
!672 = !{!"blendOptimizationMode"}
!673 = !{!"colorOutputMask"}
!674 = !{!"ProvokingVertexModeNosIndex", i32 0}
!675 = !{!"ProvokingVertexModeNosPatch", !""}
!676 = !{!"ProvokingVertexModeLast", !"Negative"}
!677 = !{!"VertexAttributesBypass", i1 false}
!678 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!679 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!680 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!681 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!682 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!683 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!684 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!685 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!686 = !{!"generatePatchesForRTWriteSends", i1 false}
!687 = !{!"generatePatchesForRT_BTIndex", i1 false}
!688 = !{!"forceVMask", i1 false}
!689 = !{!"isNumPerPrimAttributesSet", i1 false}
!690 = !{!"numPerPrimAttributes", i32 0}
!691 = !{!"WaDisableVRS", i1 false}
!692 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!693 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!694 = !{!"csInfo", !695, !696, !697, !698, !178, !154, !155, !699, !156, !700, !701, !702, !703, !704, !705, !706, !707, !708, !709, !710, !189, !711, !712, !713, !714, !715, !716, !717, !718}
!695 = !{!"maxWorkGroupSize", i32 0}
!696 = !{!"waveSize", i32 0}
!697 = !{!"ComputeShaderSecondCompile"}
!698 = !{!"forcedSIMDSize", i8 0}
!699 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!700 = !{!"forceSpillCompression", i1 false}
!701 = !{!"allowLowerSimd", i1 false}
!702 = !{!"disableSimd32Slicing", i1 false}
!703 = !{!"disableSplitOnSpill", i1 false}
!704 = !{!"enableNewSpillCostFunction", i1 false}
!705 = !{!"forceVISAPreSched", i1 false}
!706 = !{!"disableLocalIdOrderOptimizations", i1 false}
!707 = !{!"disableDispatchAlongY", i1 false}
!708 = !{!"neededThreadIdLayout", i1* null}
!709 = !{!"forceTileYWalk", i1 false}
!710 = !{!"atomicBranch", i32 0}
!711 = !{!"disableEarlyOut", i1 false}
!712 = !{!"walkOrderEnabled", i1 false}
!713 = !{!"walkOrderOverride", i32 0}
!714 = !{!"ResForHfPacking"}
!715 = !{!"constantFoldSimdSize", i1 false}
!716 = !{!"isNodeShader", i1 false}
!717 = !{!"threadGroupMergeSize", i32 0}
!718 = !{!"threadGroupMergeOverY", i1 false}
!719 = !{!"msInfo", !720, !721, !722, !723, !724, !725, !726, !727, !728, !729, !730, !676, !674, !731, !732, !716}
!720 = !{!"PrimitiveTopology", i32 3}
!721 = !{!"MaxNumOfPrimitives", i32 0}
!722 = !{!"MaxNumOfVertices", i32 0}
!723 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!724 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!725 = !{!"WorkGroupSize", i32 0}
!726 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!727 = !{!"IndexFormat", i32 6}
!728 = !{!"SubgroupSize", i32 0}
!729 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!730 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!731 = !{!"Is16BMUEModeAllowed", i1 false}
!732 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!733 = !{!"taskInfo", !649, !725, !726, !728}
!734 = !{!"NBarrierCnt", i32 0}
!735 = !{!"rtInfo", !736, !737, !738, !739, !740, !741, !742, !743, !744, !745, !746, !747, !748, !749, !750, !751, !310}
!736 = !{!"RayQueryAllocSizeInBytes", i32 0}
!737 = !{!"NumContinuations", i32 0}
!738 = !{!"RTAsyncStackAddrspace", i32 -1}
!739 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!740 = !{!"SWHotZoneAddrspace", i32 -1}
!741 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!742 = !{!"SWStackAddrspace", i32 -1}
!743 = !{!"SWStackSurfaceStateOffset", i1* null}
!744 = !{!"RTSyncStackAddrspace", i32 -1}
!745 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!746 = !{!"doSyncDispatchRays", i1 false}
!747 = !{!"MemStyle", !"Xe"}
!748 = !{!"GlobalDataStyle", !"Xe"}
!749 = !{!"NeedsBTD", i1 true}
!750 = !{!"SERHitObjectFullType", i1* null}
!751 = !{!"uberTileDimensions", i1* null}
!752 = !{!"CurUniqueIndirectIdx", i32 0}
!753 = !{!"inlineDynTextures"}
!754 = !{!"inlineResInfoData"}
!755 = !{!"immConstant", !756, !757, !758}
!756 = !{!"data"}
!757 = !{!"sizes"}
!758 = !{!"zeroIdxs"}
!759 = !{!"stringConstants"}
!760 = !{!"inlineBuffers", !761, !765, !767}
!761 = !{!"inlineBuffersVec[0]", !762, !763, !764}
!762 = !{!"alignment", i32 0}
!763 = !{!"allocSize", i64 64}
!764 = !{!"Buffer"}
!765 = !{!"inlineBuffersVec[1]", !762, !766, !764}
!766 = !{!"allocSize", i64 0}
!767 = !{!"inlineBuffersVec[2]", !762, !766, !764}
!768 = !{!"GlobalPointerProgramBinaryInfos"}
!769 = !{!"ConstantPointerProgramBinaryInfos"}
!770 = !{!"GlobalBufferAddressRelocInfo"}
!771 = !{!"ConstantBufferAddressRelocInfo"}
!772 = !{!"forceLscCacheList"}
!773 = !{!"SrvMap"}
!774 = !{!"RootConstantBufferOffsetInBytes"}
!775 = !{!"RasterizerOrderedByteAddressBuffer"}
!776 = !{!"RasterizerOrderedViews"}
!777 = !{!"MinNOSPushConstantSize", i32 0}
!778 = !{!"inlineProgramScopeOffsets", !779, !780, !781, !782}
!779 = !{!"inlineProgramScopeOffsetsMap[0]", [36 x i8]* @gVar}
!780 = !{!"inlineProgramScopeOffsetsValue[0]", i64 0}
!781 = !{!"inlineProgramScopeOffsetsMap[1]", [24 x i8]* @gVar.61}
!782 = !{!"inlineProgramScopeOffsetsValue[1]", i64 40}
!783 = !{!"shaderData", !784}
!784 = !{!"numReplicas", i32 0}
!785 = !{!"URBInfo", !786, !787, !788}
!786 = !{!"has64BVertexHeaderInput", i1 false}
!787 = !{!"has64BVertexHeaderOutput", i1 false}
!788 = !{!"hasVertexHeader", i1 true}
!789 = !{!"m_ForcePullModel", i1 false}
!790 = !{!"UseBindlessImage", i1 true}
!791 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!792 = !{!"enableRangeReduce", i1 false}
!793 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!794 = !{!"enableFRemToSRemOpt", i1 false}
!795 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!796 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!797 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!798 = !{!"allowMatchMadOptimizationforVS", i1 false}
!799 = !{!"disableMatchMadOptimizationForCS", i1 false}
!800 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!801 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!802 = !{!"statefulResourcesNotAliased", i1 false}
!803 = !{!"disableMixMode", i1 false}
!804 = !{!"genericAccessesResolved", i1 false}
!805 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!806 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!807 = !{!"disableSeparateScratchWA", i1 false}
!808 = !{!"enableRemoveUnusedTGMFence", i1 false}
!809 = !{!"PrivateMemoryPerFG"}
!810 = !{!"m_OptsToDisable"}
!811 = !{!"capabilities", !812}
!812 = !{!"globalVariableDecorationsINTEL", i1 false}
!813 = !{!"extensions", !814}
!814 = !{!"spvINTELBindlessImages", i1 false}
!815 = !{!"m_ShaderResourceViewMcsMask", !816, !817}
!816 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!817 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!818 = !{!"computedDepthMode", i32 0}
!819 = !{!"isHDCFastClearShader", i1 false}
!820 = !{!"argRegisterReservations", !821}
!821 = !{!"argRegisterReservationsVec[0]", i32 0}
!822 = !{!"SIMD16_SpillThreshold", i8 0}
!823 = !{!"SIMD32_SpillThreshold", i8 0}
!824 = !{!"m_CacheControlOption", !825, !826, !827, !828}
!825 = !{!"LscLoadCacheControlOverride", i8 0}
!826 = !{!"LscStoreCacheControlOverride", i8 0}
!827 = !{!"TgmLoadCacheControlOverride", i8 0}
!828 = !{!"TgmStoreCacheControlOverride", i8 0}
!829 = !{!"ModuleUsesBindless", i1 false}
!830 = !{!"predicationMap"}
!831 = !{!"lifeTimeStartMap"}
!832 = !{!"HitGroups"}
!833 = !{i32 2, i32 0}
!834 = !{!"clang version 16.0.6"}
!835 = !{i32 1, !"wchar_size", i32 4}
!836 = !{!837}
!837 = !{i32 4469}
!838 = !{!839}
!839 = !{i32 4470}
!840 = !{!841}
!841 = distinct !{!841, !842}
!842 = distinct !{!842}
!843 = !{!844}
!844 = !{i32 40, i32 196620}
!845 = !{!846, !841}
!846 = distinct !{!846, !847}
!847 = distinct !{!847}
!848 = !{!837, !839}

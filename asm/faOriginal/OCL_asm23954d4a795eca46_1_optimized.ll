; ------------------------------------------------
; OCL_asm23954d4a795eca46_1_optimized.ll
; LLVM major version: 16
; ------------------------------------------------
; ModuleID = '<origin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"struct.cutlass::gemm::GemmCoord" = type { %"struct.cutlass::Coord" }
%"struct.cutlass::Coord" = type { [3 x i32] }
%"class.cutlass::__generated_TensorRef" = type { i8 addrspace(1)*, %"class.sycl::_V1::range.0" }
%"class.sycl::_V1::range.0" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
%structtype.0 = type { i64 }
%"struct.cutlass::bfloat16_t" = type { i16 }

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %21 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %22 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %23 = alloca %structtype.0, align 8
  %24 = alloca %structtype.0, align 8
  %25 = alloca %structtype.0, align 8
  %26 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %27 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %28 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %29 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %30 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %17, %15
  %33 = zext i16 %localIdX to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 2
  %36 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %36)
  %37 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %37)
  %38 = mul i32 %18, %16
  %39 = zext i16 %localIdY to i32
  %40 = add i32 %38, %39
  %41 = shl i32 %40, 4
  %42 = zext i32 %19 to i64
  %43 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %43)
  %44 = mul nsw i64 %42, %9, !spirv.Decorations !610
  %45 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %29, i64 %44
  %46 = mul nsw i64 %42, %10, !spirv.Decorations !610
  %47 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %28, i64 %46
  %48 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !612
  %49 = mul nsw i64 %42, %11, !spirv.Decorations !610
  %50 = select i1 %48, i64 %49, i64 0
  %51 = getelementptr inbounds float, float addrspace(4)* %27, i64 %50
  %52 = mul nsw i64 %42, %12, !spirv.Decorations !610
  %53 = getelementptr inbounds float, float addrspace(4)* %26, i64 %52
  %54 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 1
  %55 = bitcast %structtype.0* %25 to [2 x i32]*
  %56 = getelementptr inbounds %structtype.0, %structtype.0* %25, i64 0, i32 0
  %57 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 1
  %58 = bitcast %structtype.0* %24 to [2 x i32]*
  %59 = getelementptr inbounds %structtype.0, %structtype.0* %24, i64 0, i32 0
  %60 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 1
  %61 = bitcast %structtype.0* %23 to [2 x i32]*
  %62 = getelementptr inbounds %structtype.0, %structtype.0* %23, i64 0, i32 0
  %63 = icmp slt i32 %19, %8
  br i1 %63, label %.preheader2.preheader, label %._crit_edge72

.preheader2.preheader:                            ; preds = %13, %.preheader1.15
  %64 = phi i32 [ %5602, %.preheader1.15 ], [ %19, %13 ]
  %65 = phi float addrspace(4)* [ %5601, %.preheader1.15 ], [ %53, %13 ]
  %66 = phi float addrspace(4)* [ %5599, %.preheader1.15 ], [ %51, %13 ]
  %67 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %5597, %.preheader1.15 ], [ %47, %13 ]
  %68 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %5595, %.preheader1.15 ], [ %45, %13 ]
  %69 = icmp sgt i32 %const_reg_dword2, 0
  br i1 %69, label %.preheader.preheader, label %.preheader1.preheader

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
  %70 = icmp slt i32 %41, %const_reg_dword1
  %71 = bitcast %structtype.0* %23 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  %72 = bitcast [2 x i32]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  %73 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 0
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %41, i32* %60, align 4, !noalias !614
  br label %3571

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
  %74 = phi i32 [ %3569, %.preheader.15 ], [ 0, %.preheader2.preheader ]
  %75 = icmp slt i32 %41, %const_reg_dword1
  %76 = icmp slt i32 %35, %const_reg_dword
  %77 = and i1 %76, %75
  br i1 %77, label %78, label %._crit_edge

78:                                               ; preds = %.preheader.preheader
  %79 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %79)
  %80 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %80)
  %81 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %81, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %82

82:                                               ; preds = %78, %82
  %83 = phi i32 [ 0, %78 ], [ %88, %82 ]
  %84 = zext i32 %83 to i64
  %85 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %84
  %86 = load i32, i32* %85, align 4, !noalias !617
  %87 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %84
  store i32 %86, i32* %87, align 4, !alias.scope !617
  %88 = add nuw nsw i32 %83, 1, !spirv.Decorations !620
  %89 = icmp eq i32 %83, 0
  br i1 %89, label %82, label %90, !llvm.loop !622

90:                                               ; preds = %82
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %80)
  %91 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %79)
  %92 = shl i64 %91, 32
  %93 = ashr exact i64 %92, 32
  %94 = mul nsw i64 %93, %const_reg_qword3, !spirv.Decorations !610
  %95 = ashr i64 %91, 32
  %96 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %94, i32 0
  %97 = getelementptr i16, i16 addrspace(4)* %96, i64 %95
  %98 = addrspacecast i16 addrspace(4)* %97 to i16 addrspace(1)*
  %99 = load i16, i16 addrspace(1)* %98, align 2
  %100 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %100)
  %101 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %101)
  %102 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %102, align 4, !noalias !624
  store i32 %41, i32* %57, align 4, !noalias !624
  br label %103

103:                                              ; preds = %90, %103
  %104 = phi i32 [ 0, %90 ], [ %109, %103 ]
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %105
  %107 = load i32, i32* %106, align 4, !noalias !624
  %108 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %105
  store i32 %107, i32* %108, align 4, !alias.scope !624
  %109 = add nuw nsw i32 %104, 1, !spirv.Decorations !620
  %110 = icmp eq i32 %104, 0
  br i1 %110, label %103, label %111, !llvm.loop !627

111:                                              ; preds = %103
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %101)
  %112 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %100)
  %113 = ashr i64 %112, 32
  %114 = mul nsw i64 %113, %const_reg_qword5, !spirv.Decorations !610
  %115 = shl i64 %112, 32
  %116 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %114
  %117 = ashr exact i64 %115, 31
  %118 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %116 to i8 addrspace(4)*
  %119 = getelementptr i8, i8 addrspace(4)* %118, i64 %117
  %120 = bitcast i8 addrspace(4)* %119 to i16 addrspace(4)*
  %121 = addrspacecast i16 addrspace(4)* %120 to i16 addrspace(1)*
  %122 = load i16, i16 addrspace(1)* %121, align 2
  %123 = zext i16 %99 to i32
  %124 = shl nuw i32 %123, 16, !spirv.Decorations !628
  %125 = bitcast i32 %124 to float
  %126 = zext i16 %122 to i32
  %127 = shl nuw i32 %126, 16, !spirv.Decorations !628
  %128 = bitcast i32 %127 to float
  %129 = fmul reassoc nsz arcp contract float %125, %128, !spirv.Decorations !612
  %130 = fadd reassoc nsz arcp contract float %129, %.sroa.0.1, !spirv.Decorations !612
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %111
  %.sroa.0.2 = phi float [ %130, %111 ], [ %.sroa.0.1, %.preheader.preheader ]
  %131 = or i32 %35, 1
  %132 = icmp slt i32 %131, %const_reg_dword
  %133 = and i1 %132, %75
  br i1 %133, label %134, label %._crit_edge.1

134:                                              ; preds = %._crit_edge
  %135 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %135)
  %136 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %136)
  %137 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %137, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %138

138:                                              ; preds = %138, %134
  %139 = phi i32 [ 0, %134 ], [ %144, %138 ]
  %140 = zext i32 %139 to i64
  %141 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %140
  %142 = load i32, i32* %141, align 4, !noalias !617
  %143 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %140
  store i32 %142, i32* %143, align 4, !alias.scope !617
  %144 = add nuw nsw i32 %139, 1, !spirv.Decorations !620
  %145 = icmp eq i32 %139, 0
  br i1 %145, label %138, label %146, !llvm.loop !622

146:                                              ; preds = %138
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %136)
  %147 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %135)
  %148 = shl i64 %147, 32
  %149 = ashr exact i64 %148, 32
  %150 = mul nsw i64 %149, %const_reg_qword3, !spirv.Decorations !610
  %151 = ashr i64 %147, 32
  %152 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %150, i32 0
  %153 = getelementptr i16, i16 addrspace(4)* %152, i64 %151
  %154 = addrspacecast i16 addrspace(4)* %153 to i16 addrspace(1)*
  %155 = load i16, i16 addrspace(1)* %154, align 2
  %156 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %156)
  %157 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %157)
  %158 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %158, align 4, !noalias !624
  store i32 %41, i32* %57, align 4, !noalias !624
  br label %159

159:                                              ; preds = %159, %146
  %160 = phi i32 [ 0, %146 ], [ %165, %159 ]
  %161 = zext i32 %160 to i64
  %162 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %161
  %163 = load i32, i32* %162, align 4, !noalias !624
  %164 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %161
  store i32 %163, i32* %164, align 4, !alias.scope !624
  %165 = add nuw nsw i32 %160, 1, !spirv.Decorations !620
  %166 = icmp eq i32 %160, 0
  br i1 %166, label %159, label %167, !llvm.loop !627

167:                                              ; preds = %159
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %157)
  %168 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %169 = ashr i64 %168, 32
  %170 = mul nsw i64 %169, %const_reg_qword5, !spirv.Decorations !610
  %171 = shl i64 %168, 32
  %172 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %170
  %173 = ashr exact i64 %171, 31
  %174 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %172 to i8 addrspace(4)*
  %175 = getelementptr i8, i8 addrspace(4)* %174, i64 %173
  %176 = bitcast i8 addrspace(4)* %175 to i16 addrspace(4)*
  %177 = addrspacecast i16 addrspace(4)* %176 to i16 addrspace(1)*
  %178 = load i16, i16 addrspace(1)* %177, align 2
  %179 = zext i16 %155 to i32
  %180 = shl nuw i32 %179, 16, !spirv.Decorations !628
  %181 = bitcast i32 %180 to float
  %182 = zext i16 %178 to i32
  %183 = shl nuw i32 %182, 16, !spirv.Decorations !628
  %184 = bitcast i32 %183 to float
  %185 = fmul reassoc nsz arcp contract float %181, %184, !spirv.Decorations !612
  %186 = fadd reassoc nsz arcp contract float %185, %.sroa.66.1, !spirv.Decorations !612
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %167
  %.sroa.66.2 = phi float [ %186, %167 ], [ %.sroa.66.1, %._crit_edge ]
  %187 = or i32 %35, 2
  %188 = icmp slt i32 %187, %const_reg_dword
  %189 = and i1 %188, %75
  br i1 %189, label %190, label %._crit_edge.2

190:                                              ; preds = %._crit_edge.1
  %191 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %191)
  %192 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %192)
  %193 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %193, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %194

194:                                              ; preds = %194, %190
  %195 = phi i32 [ 0, %190 ], [ %200, %194 ]
  %196 = zext i32 %195 to i64
  %197 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %196
  %198 = load i32, i32* %197, align 4, !noalias !617
  %199 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %196
  store i32 %198, i32* %199, align 4, !alias.scope !617
  %200 = add nuw nsw i32 %195, 1, !spirv.Decorations !620
  %201 = icmp eq i32 %195, 0
  br i1 %201, label %194, label %202, !llvm.loop !622

202:                                              ; preds = %194
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %192)
  %203 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %191)
  %204 = shl i64 %203, 32
  %205 = ashr exact i64 %204, 32
  %206 = mul nsw i64 %205, %const_reg_qword3, !spirv.Decorations !610
  %207 = ashr i64 %203, 32
  %208 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %206, i32 0
  %209 = getelementptr i16, i16 addrspace(4)* %208, i64 %207
  %210 = addrspacecast i16 addrspace(4)* %209 to i16 addrspace(1)*
  %211 = load i16, i16 addrspace(1)* %210, align 2
  %212 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %212)
  %213 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %213)
  %214 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %214, align 4, !noalias !624
  store i32 %41, i32* %57, align 4, !noalias !624
  br label %215

215:                                              ; preds = %215, %202
  %216 = phi i32 [ 0, %202 ], [ %221, %215 ]
  %217 = zext i32 %216 to i64
  %218 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %217
  %219 = load i32, i32* %218, align 4, !noalias !624
  %220 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %217
  store i32 %219, i32* %220, align 4, !alias.scope !624
  %221 = add nuw nsw i32 %216, 1, !spirv.Decorations !620
  %222 = icmp eq i32 %216, 0
  br i1 %222, label %215, label %223, !llvm.loop !627

223:                                              ; preds = %215
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %213)
  %224 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %212)
  %225 = ashr i64 %224, 32
  %226 = mul nsw i64 %225, %const_reg_qword5, !spirv.Decorations !610
  %227 = shl i64 %224, 32
  %228 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %226
  %229 = ashr exact i64 %227, 31
  %230 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %228 to i8 addrspace(4)*
  %231 = getelementptr i8, i8 addrspace(4)* %230, i64 %229
  %232 = bitcast i8 addrspace(4)* %231 to i16 addrspace(4)*
  %233 = addrspacecast i16 addrspace(4)* %232 to i16 addrspace(1)*
  %234 = load i16, i16 addrspace(1)* %233, align 2
  %235 = zext i16 %211 to i32
  %236 = shl nuw i32 %235, 16, !spirv.Decorations !628
  %237 = bitcast i32 %236 to float
  %238 = zext i16 %234 to i32
  %239 = shl nuw i32 %238, 16, !spirv.Decorations !628
  %240 = bitcast i32 %239 to float
  %241 = fmul reassoc nsz arcp contract float %237, %240, !spirv.Decorations !612
  %242 = fadd reassoc nsz arcp contract float %241, %.sroa.130.1, !spirv.Decorations !612
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %223
  %.sroa.130.2 = phi float [ %242, %223 ], [ %.sroa.130.1, %._crit_edge.1 ]
  %243 = or i32 %35, 3
  %244 = icmp slt i32 %243, %const_reg_dword
  %245 = and i1 %244, %75
  br i1 %245, label %246, label %.preheader

246:                                              ; preds = %._crit_edge.2
  %247 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %247)
  %248 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %248)
  %249 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %249, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %250

250:                                              ; preds = %250, %246
  %251 = phi i32 [ 0, %246 ], [ %256, %250 ]
  %252 = zext i32 %251 to i64
  %253 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %252
  %254 = load i32, i32* %253, align 4, !noalias !617
  %255 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %252
  store i32 %254, i32* %255, align 4, !alias.scope !617
  %256 = add nuw nsw i32 %251, 1, !spirv.Decorations !620
  %257 = icmp eq i32 %251, 0
  br i1 %257, label %250, label %258, !llvm.loop !622

258:                                              ; preds = %250
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %248)
  %259 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %247)
  %260 = shl i64 %259, 32
  %261 = ashr exact i64 %260, 32
  %262 = mul nsw i64 %261, %const_reg_qword3, !spirv.Decorations !610
  %263 = ashr i64 %259, 32
  %264 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %262, i32 0
  %265 = getelementptr i16, i16 addrspace(4)* %264, i64 %263
  %266 = addrspacecast i16 addrspace(4)* %265 to i16 addrspace(1)*
  %267 = load i16, i16 addrspace(1)* %266, align 2
  %268 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %268)
  %269 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %269)
  %270 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %270, align 4, !noalias !624
  store i32 %41, i32* %57, align 4, !noalias !624
  br label %271

271:                                              ; preds = %271, %258
  %272 = phi i32 [ 0, %258 ], [ %277, %271 ]
  %273 = zext i32 %272 to i64
  %274 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %273
  %275 = load i32, i32* %274, align 4, !noalias !624
  %276 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %273
  store i32 %275, i32* %276, align 4, !alias.scope !624
  %277 = add nuw nsw i32 %272, 1, !spirv.Decorations !620
  %278 = icmp eq i32 %272, 0
  br i1 %278, label %271, label %279, !llvm.loop !627

279:                                              ; preds = %271
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %269)
  %280 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %268)
  %281 = ashr i64 %280, 32
  %282 = mul nsw i64 %281, %const_reg_qword5, !spirv.Decorations !610
  %283 = shl i64 %280, 32
  %284 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %282
  %285 = ashr exact i64 %283, 31
  %286 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %284 to i8 addrspace(4)*
  %287 = getelementptr i8, i8 addrspace(4)* %286, i64 %285
  %288 = bitcast i8 addrspace(4)* %287 to i16 addrspace(4)*
  %289 = addrspacecast i16 addrspace(4)* %288 to i16 addrspace(1)*
  %290 = load i16, i16 addrspace(1)* %289, align 2
  %291 = zext i16 %267 to i32
  %292 = shl nuw i32 %291, 16, !spirv.Decorations !628
  %293 = bitcast i32 %292 to float
  %294 = zext i16 %290 to i32
  %295 = shl nuw i32 %294, 16, !spirv.Decorations !628
  %296 = bitcast i32 %295 to float
  %297 = fmul reassoc nsz arcp contract float %293, %296, !spirv.Decorations !612
  %298 = fadd reassoc nsz arcp contract float %297, %.sroa.194.1, !spirv.Decorations !612
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %279
  %.sroa.194.2 = phi float [ %298, %279 ], [ %.sroa.194.1, %._crit_edge.2 ]
  %299 = or i32 %41, 1
  %300 = icmp slt i32 %299, %const_reg_dword1
  %301 = and i1 %76, %300
  br i1 %301, label %302, label %._crit_edge.173

302:                                              ; preds = %.preheader
  %303 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %303)
  %304 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %304)
  %305 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %305, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %306

306:                                              ; preds = %306, %302
  %307 = phi i32 [ 0, %302 ], [ %312, %306 ]
  %308 = zext i32 %307 to i64
  %309 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %308
  %310 = load i32, i32* %309, align 4, !noalias !617
  %311 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %308
  store i32 %310, i32* %311, align 4, !alias.scope !617
  %312 = add nuw nsw i32 %307, 1, !spirv.Decorations !620
  %313 = icmp eq i32 %307, 0
  br i1 %313, label %306, label %314, !llvm.loop !622

314:                                              ; preds = %306
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %304)
  %315 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %303)
  %316 = shl i64 %315, 32
  %317 = ashr exact i64 %316, 32
  %318 = mul nsw i64 %317, %const_reg_qword3, !spirv.Decorations !610
  %319 = ashr i64 %315, 32
  %320 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %318, i32 0
  %321 = getelementptr i16, i16 addrspace(4)* %320, i64 %319
  %322 = addrspacecast i16 addrspace(4)* %321 to i16 addrspace(1)*
  %323 = load i16, i16 addrspace(1)* %322, align 2
  %324 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %324)
  %325 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %325)
  %326 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %326, align 4, !noalias !624
  store i32 %299, i32* %57, align 4, !noalias !624
  br label %327

327:                                              ; preds = %327, %314
  %328 = phi i32 [ 0, %314 ], [ %333, %327 ]
  %329 = zext i32 %328 to i64
  %330 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %329
  %331 = load i32, i32* %330, align 4, !noalias !624
  %332 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %329
  store i32 %331, i32* %332, align 4, !alias.scope !624
  %333 = add nuw nsw i32 %328, 1, !spirv.Decorations !620
  %334 = icmp eq i32 %328, 0
  br i1 %334, label %327, label %335, !llvm.loop !627

335:                                              ; preds = %327
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %325)
  %336 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %324)
  %337 = ashr i64 %336, 32
  %338 = mul nsw i64 %337, %const_reg_qword5, !spirv.Decorations !610
  %339 = shl i64 %336, 32
  %340 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %338
  %341 = ashr exact i64 %339, 31
  %342 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %340 to i8 addrspace(4)*
  %343 = getelementptr i8, i8 addrspace(4)* %342, i64 %341
  %344 = bitcast i8 addrspace(4)* %343 to i16 addrspace(4)*
  %345 = addrspacecast i16 addrspace(4)* %344 to i16 addrspace(1)*
  %346 = load i16, i16 addrspace(1)* %345, align 2
  %347 = zext i16 %323 to i32
  %348 = shl nuw i32 %347, 16, !spirv.Decorations !628
  %349 = bitcast i32 %348 to float
  %350 = zext i16 %346 to i32
  %351 = shl nuw i32 %350, 16, !spirv.Decorations !628
  %352 = bitcast i32 %351 to float
  %353 = fmul reassoc nsz arcp contract float %349, %352, !spirv.Decorations !612
  %354 = fadd reassoc nsz arcp contract float %353, %.sroa.6.1, !spirv.Decorations !612
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %335
  %.sroa.6.2 = phi float [ %354, %335 ], [ %.sroa.6.1, %.preheader ]
  %355 = and i1 %132, %300
  br i1 %355, label %356, label %._crit_edge.1.1

356:                                              ; preds = %._crit_edge.173
  %357 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %357)
  %358 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %358)
  %359 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %359, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %360

360:                                              ; preds = %360, %356
  %361 = phi i32 [ 0, %356 ], [ %366, %360 ]
  %362 = zext i32 %361 to i64
  %363 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %362
  %364 = load i32, i32* %363, align 4, !noalias !617
  %365 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %362
  store i32 %364, i32* %365, align 4, !alias.scope !617
  %366 = add nuw nsw i32 %361, 1, !spirv.Decorations !620
  %367 = icmp eq i32 %361, 0
  br i1 %367, label %360, label %368, !llvm.loop !622

368:                                              ; preds = %360
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %358)
  %369 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %357)
  %370 = shl i64 %369, 32
  %371 = ashr exact i64 %370, 32
  %372 = mul nsw i64 %371, %const_reg_qword3, !spirv.Decorations !610
  %373 = ashr i64 %369, 32
  %374 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %372, i32 0
  %375 = getelementptr i16, i16 addrspace(4)* %374, i64 %373
  %376 = addrspacecast i16 addrspace(4)* %375 to i16 addrspace(1)*
  %377 = load i16, i16 addrspace(1)* %376, align 2
  %378 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %378)
  %379 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %379)
  %380 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %380, align 4, !noalias !624
  store i32 %299, i32* %57, align 4, !noalias !624
  br label %381

381:                                              ; preds = %381, %368
  %382 = phi i32 [ 0, %368 ], [ %387, %381 ]
  %383 = zext i32 %382 to i64
  %384 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %383
  %385 = load i32, i32* %384, align 4, !noalias !624
  %386 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %383
  store i32 %385, i32* %386, align 4, !alias.scope !624
  %387 = add nuw nsw i32 %382, 1, !spirv.Decorations !620
  %388 = icmp eq i32 %382, 0
  br i1 %388, label %381, label %389, !llvm.loop !627

389:                                              ; preds = %381
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %379)
  %390 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %378)
  %391 = ashr i64 %390, 32
  %392 = mul nsw i64 %391, %const_reg_qword5, !spirv.Decorations !610
  %393 = shl i64 %390, 32
  %394 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %392
  %395 = ashr exact i64 %393, 31
  %396 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %394 to i8 addrspace(4)*
  %397 = getelementptr i8, i8 addrspace(4)* %396, i64 %395
  %398 = bitcast i8 addrspace(4)* %397 to i16 addrspace(4)*
  %399 = addrspacecast i16 addrspace(4)* %398 to i16 addrspace(1)*
  %400 = load i16, i16 addrspace(1)* %399, align 2
  %401 = zext i16 %377 to i32
  %402 = shl nuw i32 %401, 16, !spirv.Decorations !628
  %403 = bitcast i32 %402 to float
  %404 = zext i16 %400 to i32
  %405 = shl nuw i32 %404, 16, !spirv.Decorations !628
  %406 = bitcast i32 %405 to float
  %407 = fmul reassoc nsz arcp contract float %403, %406, !spirv.Decorations !612
  %408 = fadd reassoc nsz arcp contract float %407, %.sroa.70.1, !spirv.Decorations !612
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %389
  %.sroa.70.2 = phi float [ %408, %389 ], [ %.sroa.70.1, %._crit_edge.173 ]
  %409 = and i1 %188, %300
  br i1 %409, label %410, label %._crit_edge.2.1

410:                                              ; preds = %._crit_edge.1.1
  %411 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %411)
  %412 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %412)
  %413 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %413, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %414

414:                                              ; preds = %414, %410
  %415 = phi i32 [ 0, %410 ], [ %420, %414 ]
  %416 = zext i32 %415 to i64
  %417 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %416
  %418 = load i32, i32* %417, align 4, !noalias !617
  %419 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %416
  store i32 %418, i32* %419, align 4, !alias.scope !617
  %420 = add nuw nsw i32 %415, 1, !spirv.Decorations !620
  %421 = icmp eq i32 %415, 0
  br i1 %421, label %414, label %422, !llvm.loop !622

422:                                              ; preds = %414
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %412)
  %423 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %411)
  %424 = shl i64 %423, 32
  %425 = ashr exact i64 %424, 32
  %426 = mul nsw i64 %425, %const_reg_qword3, !spirv.Decorations !610
  %427 = ashr i64 %423, 32
  %428 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %426, i32 0
  %429 = getelementptr i16, i16 addrspace(4)* %428, i64 %427
  %430 = addrspacecast i16 addrspace(4)* %429 to i16 addrspace(1)*
  %431 = load i16, i16 addrspace(1)* %430, align 2
  %432 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %432)
  %433 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %433)
  %434 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %434, align 4, !noalias !624
  store i32 %299, i32* %57, align 4, !noalias !624
  br label %435

435:                                              ; preds = %435, %422
  %436 = phi i32 [ 0, %422 ], [ %441, %435 ]
  %437 = zext i32 %436 to i64
  %438 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %437
  %439 = load i32, i32* %438, align 4, !noalias !624
  %440 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %437
  store i32 %439, i32* %440, align 4, !alias.scope !624
  %441 = add nuw nsw i32 %436, 1, !spirv.Decorations !620
  %442 = icmp eq i32 %436, 0
  br i1 %442, label %435, label %443, !llvm.loop !627

443:                                              ; preds = %435
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %433)
  %444 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %432)
  %445 = ashr i64 %444, 32
  %446 = mul nsw i64 %445, %const_reg_qword5, !spirv.Decorations !610
  %447 = shl i64 %444, 32
  %448 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %446
  %449 = ashr exact i64 %447, 31
  %450 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %448 to i8 addrspace(4)*
  %451 = getelementptr i8, i8 addrspace(4)* %450, i64 %449
  %452 = bitcast i8 addrspace(4)* %451 to i16 addrspace(4)*
  %453 = addrspacecast i16 addrspace(4)* %452 to i16 addrspace(1)*
  %454 = load i16, i16 addrspace(1)* %453, align 2
  %455 = zext i16 %431 to i32
  %456 = shl nuw i32 %455, 16, !spirv.Decorations !628
  %457 = bitcast i32 %456 to float
  %458 = zext i16 %454 to i32
  %459 = shl nuw i32 %458, 16, !spirv.Decorations !628
  %460 = bitcast i32 %459 to float
  %461 = fmul reassoc nsz arcp contract float %457, %460, !spirv.Decorations !612
  %462 = fadd reassoc nsz arcp contract float %461, %.sroa.134.1, !spirv.Decorations !612
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %443
  %.sroa.134.2 = phi float [ %462, %443 ], [ %.sroa.134.1, %._crit_edge.1.1 ]
  %463 = and i1 %244, %300
  br i1 %463, label %464, label %.preheader.1

464:                                              ; preds = %._crit_edge.2.1
  %465 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %465)
  %466 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %466)
  %467 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %467, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %468

468:                                              ; preds = %468, %464
  %469 = phi i32 [ 0, %464 ], [ %474, %468 ]
  %470 = zext i32 %469 to i64
  %471 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %470
  %472 = load i32, i32* %471, align 4, !noalias !617
  %473 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %470
  store i32 %472, i32* %473, align 4, !alias.scope !617
  %474 = add nuw nsw i32 %469, 1, !spirv.Decorations !620
  %475 = icmp eq i32 %469, 0
  br i1 %475, label %468, label %476, !llvm.loop !622

476:                                              ; preds = %468
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %466)
  %477 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %465)
  %478 = shl i64 %477, 32
  %479 = ashr exact i64 %478, 32
  %480 = mul nsw i64 %479, %const_reg_qword3, !spirv.Decorations !610
  %481 = ashr i64 %477, 32
  %482 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %480, i32 0
  %483 = getelementptr i16, i16 addrspace(4)* %482, i64 %481
  %484 = addrspacecast i16 addrspace(4)* %483 to i16 addrspace(1)*
  %485 = load i16, i16 addrspace(1)* %484, align 2
  %486 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %486)
  %487 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %487)
  %488 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %488, align 4, !noalias !624
  store i32 %299, i32* %57, align 4, !noalias !624
  br label %489

489:                                              ; preds = %489, %476
  %490 = phi i32 [ 0, %476 ], [ %495, %489 ]
  %491 = zext i32 %490 to i64
  %492 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %491
  %493 = load i32, i32* %492, align 4, !noalias !624
  %494 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %491
  store i32 %493, i32* %494, align 4, !alias.scope !624
  %495 = add nuw nsw i32 %490, 1, !spirv.Decorations !620
  %496 = icmp eq i32 %490, 0
  br i1 %496, label %489, label %497, !llvm.loop !627

497:                                              ; preds = %489
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %487)
  %498 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %486)
  %499 = ashr i64 %498, 32
  %500 = mul nsw i64 %499, %const_reg_qword5, !spirv.Decorations !610
  %501 = shl i64 %498, 32
  %502 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %500
  %503 = ashr exact i64 %501, 31
  %504 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %502 to i8 addrspace(4)*
  %505 = getelementptr i8, i8 addrspace(4)* %504, i64 %503
  %506 = bitcast i8 addrspace(4)* %505 to i16 addrspace(4)*
  %507 = addrspacecast i16 addrspace(4)* %506 to i16 addrspace(1)*
  %508 = load i16, i16 addrspace(1)* %507, align 2
  %509 = zext i16 %485 to i32
  %510 = shl nuw i32 %509, 16, !spirv.Decorations !628
  %511 = bitcast i32 %510 to float
  %512 = zext i16 %508 to i32
  %513 = shl nuw i32 %512, 16, !spirv.Decorations !628
  %514 = bitcast i32 %513 to float
  %515 = fmul reassoc nsz arcp contract float %511, %514, !spirv.Decorations !612
  %516 = fadd reassoc nsz arcp contract float %515, %.sroa.198.1, !spirv.Decorations !612
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %497
  %.sroa.198.2 = phi float [ %516, %497 ], [ %.sroa.198.1, %._crit_edge.2.1 ]
  %517 = or i32 %41, 2
  %518 = icmp slt i32 %517, %const_reg_dword1
  %519 = and i1 %76, %518
  br i1 %519, label %520, label %._crit_edge.274

520:                                              ; preds = %.preheader.1
  %521 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %521)
  %522 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %522)
  %523 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %523, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %524

524:                                              ; preds = %524, %520
  %525 = phi i32 [ 0, %520 ], [ %530, %524 ]
  %526 = zext i32 %525 to i64
  %527 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %526
  %528 = load i32, i32* %527, align 4, !noalias !617
  %529 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %526
  store i32 %528, i32* %529, align 4, !alias.scope !617
  %530 = add nuw nsw i32 %525, 1, !spirv.Decorations !620
  %531 = icmp eq i32 %525, 0
  br i1 %531, label %524, label %532, !llvm.loop !622

532:                                              ; preds = %524
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %522)
  %533 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %521)
  %534 = shl i64 %533, 32
  %535 = ashr exact i64 %534, 32
  %536 = mul nsw i64 %535, %const_reg_qword3, !spirv.Decorations !610
  %537 = ashr i64 %533, 32
  %538 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %536, i32 0
  %539 = getelementptr i16, i16 addrspace(4)* %538, i64 %537
  %540 = addrspacecast i16 addrspace(4)* %539 to i16 addrspace(1)*
  %541 = load i16, i16 addrspace(1)* %540, align 2
  %542 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %542)
  %543 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %543)
  %544 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %544, align 4, !noalias !624
  store i32 %517, i32* %57, align 4, !noalias !624
  br label %545

545:                                              ; preds = %545, %532
  %546 = phi i32 [ 0, %532 ], [ %551, %545 ]
  %547 = zext i32 %546 to i64
  %548 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %547
  %549 = load i32, i32* %548, align 4, !noalias !624
  %550 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %547
  store i32 %549, i32* %550, align 4, !alias.scope !624
  %551 = add nuw nsw i32 %546, 1, !spirv.Decorations !620
  %552 = icmp eq i32 %546, 0
  br i1 %552, label %545, label %553, !llvm.loop !627

553:                                              ; preds = %545
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %543)
  %554 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %542)
  %555 = ashr i64 %554, 32
  %556 = mul nsw i64 %555, %const_reg_qword5, !spirv.Decorations !610
  %557 = shl i64 %554, 32
  %558 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %556
  %559 = ashr exact i64 %557, 31
  %560 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %558 to i8 addrspace(4)*
  %561 = getelementptr i8, i8 addrspace(4)* %560, i64 %559
  %562 = bitcast i8 addrspace(4)* %561 to i16 addrspace(4)*
  %563 = addrspacecast i16 addrspace(4)* %562 to i16 addrspace(1)*
  %564 = load i16, i16 addrspace(1)* %563, align 2
  %565 = zext i16 %541 to i32
  %566 = shl nuw i32 %565, 16, !spirv.Decorations !628
  %567 = bitcast i32 %566 to float
  %568 = zext i16 %564 to i32
  %569 = shl nuw i32 %568, 16, !spirv.Decorations !628
  %570 = bitcast i32 %569 to float
  %571 = fmul reassoc nsz arcp contract float %567, %570, !spirv.Decorations !612
  %572 = fadd reassoc nsz arcp contract float %571, %.sroa.10.1, !spirv.Decorations !612
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %553
  %.sroa.10.2 = phi float [ %572, %553 ], [ %.sroa.10.1, %.preheader.1 ]
  %573 = and i1 %132, %518
  br i1 %573, label %574, label %._crit_edge.1.2

574:                                              ; preds = %._crit_edge.274
  %575 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %575)
  %576 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %576)
  %577 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %577, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %578

578:                                              ; preds = %578, %574
  %579 = phi i32 [ 0, %574 ], [ %584, %578 ]
  %580 = zext i32 %579 to i64
  %581 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %580
  %582 = load i32, i32* %581, align 4, !noalias !617
  %583 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %580
  store i32 %582, i32* %583, align 4, !alias.scope !617
  %584 = add nuw nsw i32 %579, 1, !spirv.Decorations !620
  %585 = icmp eq i32 %579, 0
  br i1 %585, label %578, label %586, !llvm.loop !622

586:                                              ; preds = %578
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %576)
  %587 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %575)
  %588 = shl i64 %587, 32
  %589 = ashr exact i64 %588, 32
  %590 = mul nsw i64 %589, %const_reg_qword3, !spirv.Decorations !610
  %591 = ashr i64 %587, 32
  %592 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %590, i32 0
  %593 = getelementptr i16, i16 addrspace(4)* %592, i64 %591
  %594 = addrspacecast i16 addrspace(4)* %593 to i16 addrspace(1)*
  %595 = load i16, i16 addrspace(1)* %594, align 2
  %596 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %596)
  %597 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %597)
  %598 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %598, align 4, !noalias !624
  store i32 %517, i32* %57, align 4, !noalias !624
  br label %599

599:                                              ; preds = %599, %586
  %600 = phi i32 [ 0, %586 ], [ %605, %599 ]
  %601 = zext i32 %600 to i64
  %602 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %601
  %603 = load i32, i32* %602, align 4, !noalias !624
  %604 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %601
  store i32 %603, i32* %604, align 4, !alias.scope !624
  %605 = add nuw nsw i32 %600, 1, !spirv.Decorations !620
  %606 = icmp eq i32 %600, 0
  br i1 %606, label %599, label %607, !llvm.loop !627

607:                                              ; preds = %599
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %597)
  %608 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %596)
  %609 = ashr i64 %608, 32
  %610 = mul nsw i64 %609, %const_reg_qword5, !spirv.Decorations !610
  %611 = shl i64 %608, 32
  %612 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %610
  %613 = ashr exact i64 %611, 31
  %614 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %612 to i8 addrspace(4)*
  %615 = getelementptr i8, i8 addrspace(4)* %614, i64 %613
  %616 = bitcast i8 addrspace(4)* %615 to i16 addrspace(4)*
  %617 = addrspacecast i16 addrspace(4)* %616 to i16 addrspace(1)*
  %618 = load i16, i16 addrspace(1)* %617, align 2
  %619 = zext i16 %595 to i32
  %620 = shl nuw i32 %619, 16, !spirv.Decorations !628
  %621 = bitcast i32 %620 to float
  %622 = zext i16 %618 to i32
  %623 = shl nuw i32 %622, 16, !spirv.Decorations !628
  %624 = bitcast i32 %623 to float
  %625 = fmul reassoc nsz arcp contract float %621, %624, !spirv.Decorations !612
  %626 = fadd reassoc nsz arcp contract float %625, %.sroa.74.1, !spirv.Decorations !612
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %607
  %.sroa.74.2 = phi float [ %626, %607 ], [ %.sroa.74.1, %._crit_edge.274 ]
  %627 = and i1 %188, %518
  br i1 %627, label %628, label %._crit_edge.2.2

628:                                              ; preds = %._crit_edge.1.2
  %629 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %629)
  %630 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %630)
  %631 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %631, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %632

632:                                              ; preds = %632, %628
  %633 = phi i32 [ 0, %628 ], [ %638, %632 ]
  %634 = zext i32 %633 to i64
  %635 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %634
  %636 = load i32, i32* %635, align 4, !noalias !617
  %637 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %634
  store i32 %636, i32* %637, align 4, !alias.scope !617
  %638 = add nuw nsw i32 %633, 1, !spirv.Decorations !620
  %639 = icmp eq i32 %633, 0
  br i1 %639, label %632, label %640, !llvm.loop !622

640:                                              ; preds = %632
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %630)
  %641 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %629)
  %642 = shl i64 %641, 32
  %643 = ashr exact i64 %642, 32
  %644 = mul nsw i64 %643, %const_reg_qword3, !spirv.Decorations !610
  %645 = ashr i64 %641, 32
  %646 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %644, i32 0
  %647 = getelementptr i16, i16 addrspace(4)* %646, i64 %645
  %648 = addrspacecast i16 addrspace(4)* %647 to i16 addrspace(1)*
  %649 = load i16, i16 addrspace(1)* %648, align 2
  %650 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %650)
  %651 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %651)
  %652 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %652, align 4, !noalias !624
  store i32 %517, i32* %57, align 4, !noalias !624
  br label %653

653:                                              ; preds = %653, %640
  %654 = phi i32 [ 0, %640 ], [ %659, %653 ]
  %655 = zext i32 %654 to i64
  %656 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %655
  %657 = load i32, i32* %656, align 4, !noalias !624
  %658 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %655
  store i32 %657, i32* %658, align 4, !alias.scope !624
  %659 = add nuw nsw i32 %654, 1, !spirv.Decorations !620
  %660 = icmp eq i32 %654, 0
  br i1 %660, label %653, label %661, !llvm.loop !627

661:                                              ; preds = %653
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %651)
  %662 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %650)
  %663 = ashr i64 %662, 32
  %664 = mul nsw i64 %663, %const_reg_qword5, !spirv.Decorations !610
  %665 = shl i64 %662, 32
  %666 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %664
  %667 = ashr exact i64 %665, 31
  %668 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %666 to i8 addrspace(4)*
  %669 = getelementptr i8, i8 addrspace(4)* %668, i64 %667
  %670 = bitcast i8 addrspace(4)* %669 to i16 addrspace(4)*
  %671 = addrspacecast i16 addrspace(4)* %670 to i16 addrspace(1)*
  %672 = load i16, i16 addrspace(1)* %671, align 2
  %673 = zext i16 %649 to i32
  %674 = shl nuw i32 %673, 16, !spirv.Decorations !628
  %675 = bitcast i32 %674 to float
  %676 = zext i16 %672 to i32
  %677 = shl nuw i32 %676, 16, !spirv.Decorations !628
  %678 = bitcast i32 %677 to float
  %679 = fmul reassoc nsz arcp contract float %675, %678, !spirv.Decorations !612
  %680 = fadd reassoc nsz arcp contract float %679, %.sroa.138.1, !spirv.Decorations !612
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %661
  %.sroa.138.2 = phi float [ %680, %661 ], [ %.sroa.138.1, %._crit_edge.1.2 ]
  %681 = and i1 %244, %518
  br i1 %681, label %682, label %.preheader.2

682:                                              ; preds = %._crit_edge.2.2
  %683 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %683)
  %684 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %684)
  %685 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %685, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %686

686:                                              ; preds = %686, %682
  %687 = phi i32 [ 0, %682 ], [ %692, %686 ]
  %688 = zext i32 %687 to i64
  %689 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %688
  %690 = load i32, i32* %689, align 4, !noalias !617
  %691 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %688
  store i32 %690, i32* %691, align 4, !alias.scope !617
  %692 = add nuw nsw i32 %687, 1, !spirv.Decorations !620
  %693 = icmp eq i32 %687, 0
  br i1 %693, label %686, label %694, !llvm.loop !622

694:                                              ; preds = %686
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %684)
  %695 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %683)
  %696 = shl i64 %695, 32
  %697 = ashr exact i64 %696, 32
  %698 = mul nsw i64 %697, %const_reg_qword3, !spirv.Decorations !610
  %699 = ashr i64 %695, 32
  %700 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %698, i32 0
  %701 = getelementptr i16, i16 addrspace(4)* %700, i64 %699
  %702 = addrspacecast i16 addrspace(4)* %701 to i16 addrspace(1)*
  %703 = load i16, i16 addrspace(1)* %702, align 2
  %704 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %704)
  %705 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %705)
  %706 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %706, align 4, !noalias !624
  store i32 %517, i32* %57, align 4, !noalias !624
  br label %707

707:                                              ; preds = %707, %694
  %708 = phi i32 [ 0, %694 ], [ %713, %707 ]
  %709 = zext i32 %708 to i64
  %710 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %709
  %711 = load i32, i32* %710, align 4, !noalias !624
  %712 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %709
  store i32 %711, i32* %712, align 4, !alias.scope !624
  %713 = add nuw nsw i32 %708, 1, !spirv.Decorations !620
  %714 = icmp eq i32 %708, 0
  br i1 %714, label %707, label %715, !llvm.loop !627

715:                                              ; preds = %707
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %705)
  %716 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %704)
  %717 = ashr i64 %716, 32
  %718 = mul nsw i64 %717, %const_reg_qword5, !spirv.Decorations !610
  %719 = shl i64 %716, 32
  %720 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %718
  %721 = ashr exact i64 %719, 31
  %722 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %720 to i8 addrspace(4)*
  %723 = getelementptr i8, i8 addrspace(4)* %722, i64 %721
  %724 = bitcast i8 addrspace(4)* %723 to i16 addrspace(4)*
  %725 = addrspacecast i16 addrspace(4)* %724 to i16 addrspace(1)*
  %726 = load i16, i16 addrspace(1)* %725, align 2
  %727 = zext i16 %703 to i32
  %728 = shl nuw i32 %727, 16, !spirv.Decorations !628
  %729 = bitcast i32 %728 to float
  %730 = zext i16 %726 to i32
  %731 = shl nuw i32 %730, 16, !spirv.Decorations !628
  %732 = bitcast i32 %731 to float
  %733 = fmul reassoc nsz arcp contract float %729, %732, !spirv.Decorations !612
  %734 = fadd reassoc nsz arcp contract float %733, %.sroa.202.1, !spirv.Decorations !612
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %715
  %.sroa.202.2 = phi float [ %734, %715 ], [ %.sroa.202.1, %._crit_edge.2.2 ]
  %735 = or i32 %41, 3
  %736 = icmp slt i32 %735, %const_reg_dword1
  %737 = and i1 %76, %736
  br i1 %737, label %738, label %._crit_edge.375

738:                                              ; preds = %.preheader.2
  %739 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %739)
  %740 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %740)
  %741 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %741, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %742

742:                                              ; preds = %742, %738
  %743 = phi i32 [ 0, %738 ], [ %748, %742 ]
  %744 = zext i32 %743 to i64
  %745 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %744
  %746 = load i32, i32* %745, align 4, !noalias !617
  %747 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %744
  store i32 %746, i32* %747, align 4, !alias.scope !617
  %748 = add nuw nsw i32 %743, 1, !spirv.Decorations !620
  %749 = icmp eq i32 %743, 0
  br i1 %749, label %742, label %750, !llvm.loop !622

750:                                              ; preds = %742
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %740)
  %751 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %739)
  %752 = shl i64 %751, 32
  %753 = ashr exact i64 %752, 32
  %754 = mul nsw i64 %753, %const_reg_qword3, !spirv.Decorations !610
  %755 = ashr i64 %751, 32
  %756 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %754, i32 0
  %757 = getelementptr i16, i16 addrspace(4)* %756, i64 %755
  %758 = addrspacecast i16 addrspace(4)* %757 to i16 addrspace(1)*
  %759 = load i16, i16 addrspace(1)* %758, align 2
  %760 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %760)
  %761 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %761)
  %762 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %762, align 4, !noalias !624
  store i32 %735, i32* %57, align 4, !noalias !624
  br label %763

763:                                              ; preds = %763, %750
  %764 = phi i32 [ 0, %750 ], [ %769, %763 ]
  %765 = zext i32 %764 to i64
  %766 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %765
  %767 = load i32, i32* %766, align 4, !noalias !624
  %768 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %765
  store i32 %767, i32* %768, align 4, !alias.scope !624
  %769 = add nuw nsw i32 %764, 1, !spirv.Decorations !620
  %770 = icmp eq i32 %764, 0
  br i1 %770, label %763, label %771, !llvm.loop !627

771:                                              ; preds = %763
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %761)
  %772 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %760)
  %773 = ashr i64 %772, 32
  %774 = mul nsw i64 %773, %const_reg_qword5, !spirv.Decorations !610
  %775 = shl i64 %772, 32
  %776 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %774
  %777 = ashr exact i64 %775, 31
  %778 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %776 to i8 addrspace(4)*
  %779 = getelementptr i8, i8 addrspace(4)* %778, i64 %777
  %780 = bitcast i8 addrspace(4)* %779 to i16 addrspace(4)*
  %781 = addrspacecast i16 addrspace(4)* %780 to i16 addrspace(1)*
  %782 = load i16, i16 addrspace(1)* %781, align 2
  %783 = zext i16 %759 to i32
  %784 = shl nuw i32 %783, 16, !spirv.Decorations !628
  %785 = bitcast i32 %784 to float
  %786 = zext i16 %782 to i32
  %787 = shl nuw i32 %786, 16, !spirv.Decorations !628
  %788 = bitcast i32 %787 to float
  %789 = fmul reassoc nsz arcp contract float %785, %788, !spirv.Decorations !612
  %790 = fadd reassoc nsz arcp contract float %789, %.sroa.14.1, !spirv.Decorations !612
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %771
  %.sroa.14.2 = phi float [ %790, %771 ], [ %.sroa.14.1, %.preheader.2 ]
  %791 = and i1 %132, %736
  br i1 %791, label %792, label %._crit_edge.1.3

792:                                              ; preds = %._crit_edge.375
  %793 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %793)
  %794 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %794)
  %795 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %795, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %796

796:                                              ; preds = %796, %792
  %797 = phi i32 [ 0, %792 ], [ %802, %796 ]
  %798 = zext i32 %797 to i64
  %799 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %798
  %800 = load i32, i32* %799, align 4, !noalias !617
  %801 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %798
  store i32 %800, i32* %801, align 4, !alias.scope !617
  %802 = add nuw nsw i32 %797, 1, !spirv.Decorations !620
  %803 = icmp eq i32 %797, 0
  br i1 %803, label %796, label %804, !llvm.loop !622

804:                                              ; preds = %796
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %794)
  %805 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %793)
  %806 = shl i64 %805, 32
  %807 = ashr exact i64 %806, 32
  %808 = mul nsw i64 %807, %const_reg_qword3, !spirv.Decorations !610
  %809 = ashr i64 %805, 32
  %810 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %808, i32 0
  %811 = getelementptr i16, i16 addrspace(4)* %810, i64 %809
  %812 = addrspacecast i16 addrspace(4)* %811 to i16 addrspace(1)*
  %813 = load i16, i16 addrspace(1)* %812, align 2
  %814 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %814)
  %815 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %815)
  %816 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %816, align 4, !noalias !624
  store i32 %735, i32* %57, align 4, !noalias !624
  br label %817

817:                                              ; preds = %817, %804
  %818 = phi i32 [ 0, %804 ], [ %823, %817 ]
  %819 = zext i32 %818 to i64
  %820 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %819
  %821 = load i32, i32* %820, align 4, !noalias !624
  %822 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %819
  store i32 %821, i32* %822, align 4, !alias.scope !624
  %823 = add nuw nsw i32 %818, 1, !spirv.Decorations !620
  %824 = icmp eq i32 %818, 0
  br i1 %824, label %817, label %825, !llvm.loop !627

825:                                              ; preds = %817
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %815)
  %826 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %814)
  %827 = ashr i64 %826, 32
  %828 = mul nsw i64 %827, %const_reg_qword5, !spirv.Decorations !610
  %829 = shl i64 %826, 32
  %830 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %828
  %831 = ashr exact i64 %829, 31
  %832 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %830 to i8 addrspace(4)*
  %833 = getelementptr i8, i8 addrspace(4)* %832, i64 %831
  %834 = bitcast i8 addrspace(4)* %833 to i16 addrspace(4)*
  %835 = addrspacecast i16 addrspace(4)* %834 to i16 addrspace(1)*
  %836 = load i16, i16 addrspace(1)* %835, align 2
  %837 = zext i16 %813 to i32
  %838 = shl nuw i32 %837, 16, !spirv.Decorations !628
  %839 = bitcast i32 %838 to float
  %840 = zext i16 %836 to i32
  %841 = shl nuw i32 %840, 16, !spirv.Decorations !628
  %842 = bitcast i32 %841 to float
  %843 = fmul reassoc nsz arcp contract float %839, %842, !spirv.Decorations !612
  %844 = fadd reassoc nsz arcp contract float %843, %.sroa.78.1, !spirv.Decorations !612
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %825
  %.sroa.78.2 = phi float [ %844, %825 ], [ %.sroa.78.1, %._crit_edge.375 ]
  %845 = and i1 %188, %736
  br i1 %845, label %846, label %._crit_edge.2.3

846:                                              ; preds = %._crit_edge.1.3
  %847 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %847)
  %848 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %848)
  %849 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %849, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %850

850:                                              ; preds = %850, %846
  %851 = phi i32 [ 0, %846 ], [ %856, %850 ]
  %852 = zext i32 %851 to i64
  %853 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %852
  %854 = load i32, i32* %853, align 4, !noalias !617
  %855 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %852
  store i32 %854, i32* %855, align 4, !alias.scope !617
  %856 = add nuw nsw i32 %851, 1, !spirv.Decorations !620
  %857 = icmp eq i32 %851, 0
  br i1 %857, label %850, label %858, !llvm.loop !622

858:                                              ; preds = %850
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %848)
  %859 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %847)
  %860 = shl i64 %859, 32
  %861 = ashr exact i64 %860, 32
  %862 = mul nsw i64 %861, %const_reg_qword3, !spirv.Decorations !610
  %863 = ashr i64 %859, 32
  %864 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %862, i32 0
  %865 = getelementptr i16, i16 addrspace(4)* %864, i64 %863
  %866 = addrspacecast i16 addrspace(4)* %865 to i16 addrspace(1)*
  %867 = load i16, i16 addrspace(1)* %866, align 2
  %868 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %868)
  %869 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %869)
  %870 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %870, align 4, !noalias !624
  store i32 %735, i32* %57, align 4, !noalias !624
  br label %871

871:                                              ; preds = %871, %858
  %872 = phi i32 [ 0, %858 ], [ %877, %871 ]
  %873 = zext i32 %872 to i64
  %874 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %873
  %875 = load i32, i32* %874, align 4, !noalias !624
  %876 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %873
  store i32 %875, i32* %876, align 4, !alias.scope !624
  %877 = add nuw nsw i32 %872, 1, !spirv.Decorations !620
  %878 = icmp eq i32 %872, 0
  br i1 %878, label %871, label %879, !llvm.loop !627

879:                                              ; preds = %871
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %869)
  %880 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %868)
  %881 = ashr i64 %880, 32
  %882 = mul nsw i64 %881, %const_reg_qword5, !spirv.Decorations !610
  %883 = shl i64 %880, 32
  %884 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %882
  %885 = ashr exact i64 %883, 31
  %886 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %884 to i8 addrspace(4)*
  %887 = getelementptr i8, i8 addrspace(4)* %886, i64 %885
  %888 = bitcast i8 addrspace(4)* %887 to i16 addrspace(4)*
  %889 = addrspacecast i16 addrspace(4)* %888 to i16 addrspace(1)*
  %890 = load i16, i16 addrspace(1)* %889, align 2
  %891 = zext i16 %867 to i32
  %892 = shl nuw i32 %891, 16, !spirv.Decorations !628
  %893 = bitcast i32 %892 to float
  %894 = zext i16 %890 to i32
  %895 = shl nuw i32 %894, 16, !spirv.Decorations !628
  %896 = bitcast i32 %895 to float
  %897 = fmul reassoc nsz arcp contract float %893, %896, !spirv.Decorations !612
  %898 = fadd reassoc nsz arcp contract float %897, %.sroa.142.1, !spirv.Decorations !612
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %879
  %.sroa.142.2 = phi float [ %898, %879 ], [ %.sroa.142.1, %._crit_edge.1.3 ]
  %899 = and i1 %244, %736
  br i1 %899, label %900, label %.preheader.3

900:                                              ; preds = %._crit_edge.2.3
  %901 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %901)
  %902 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %902)
  %903 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %903, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %904

904:                                              ; preds = %904, %900
  %905 = phi i32 [ 0, %900 ], [ %910, %904 ]
  %906 = zext i32 %905 to i64
  %907 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %906
  %908 = load i32, i32* %907, align 4, !noalias !617
  %909 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %906
  store i32 %908, i32* %909, align 4, !alias.scope !617
  %910 = add nuw nsw i32 %905, 1, !spirv.Decorations !620
  %911 = icmp eq i32 %905, 0
  br i1 %911, label %904, label %912, !llvm.loop !622

912:                                              ; preds = %904
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %902)
  %913 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %901)
  %914 = shl i64 %913, 32
  %915 = ashr exact i64 %914, 32
  %916 = mul nsw i64 %915, %const_reg_qword3, !spirv.Decorations !610
  %917 = ashr i64 %913, 32
  %918 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %916, i32 0
  %919 = getelementptr i16, i16 addrspace(4)* %918, i64 %917
  %920 = addrspacecast i16 addrspace(4)* %919 to i16 addrspace(1)*
  %921 = load i16, i16 addrspace(1)* %920, align 2
  %922 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %922)
  %923 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %923)
  %924 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %924, align 4, !noalias !624
  store i32 %735, i32* %57, align 4, !noalias !624
  br label %925

925:                                              ; preds = %925, %912
  %926 = phi i32 [ 0, %912 ], [ %931, %925 ]
  %927 = zext i32 %926 to i64
  %928 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %927
  %929 = load i32, i32* %928, align 4, !noalias !624
  %930 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %927
  store i32 %929, i32* %930, align 4, !alias.scope !624
  %931 = add nuw nsw i32 %926, 1, !spirv.Decorations !620
  %932 = icmp eq i32 %926, 0
  br i1 %932, label %925, label %933, !llvm.loop !627

933:                                              ; preds = %925
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %923)
  %934 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %922)
  %935 = ashr i64 %934, 32
  %936 = mul nsw i64 %935, %const_reg_qword5, !spirv.Decorations !610
  %937 = shl i64 %934, 32
  %938 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %936
  %939 = ashr exact i64 %937, 31
  %940 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %938 to i8 addrspace(4)*
  %941 = getelementptr i8, i8 addrspace(4)* %940, i64 %939
  %942 = bitcast i8 addrspace(4)* %941 to i16 addrspace(4)*
  %943 = addrspacecast i16 addrspace(4)* %942 to i16 addrspace(1)*
  %944 = load i16, i16 addrspace(1)* %943, align 2
  %945 = zext i16 %921 to i32
  %946 = shl nuw i32 %945, 16, !spirv.Decorations !628
  %947 = bitcast i32 %946 to float
  %948 = zext i16 %944 to i32
  %949 = shl nuw i32 %948, 16, !spirv.Decorations !628
  %950 = bitcast i32 %949 to float
  %951 = fmul reassoc nsz arcp contract float %947, %950, !spirv.Decorations !612
  %952 = fadd reassoc nsz arcp contract float %951, %.sroa.206.1, !spirv.Decorations !612
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %933
  %.sroa.206.2 = phi float [ %952, %933 ], [ %.sroa.206.1, %._crit_edge.2.3 ]
  %953 = or i32 %41, 4
  %954 = icmp slt i32 %953, %const_reg_dword1
  %955 = and i1 %76, %954
  br i1 %955, label %956, label %._crit_edge.4

956:                                              ; preds = %.preheader.3
  %957 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %957)
  %958 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %958)
  %959 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %959, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %960

960:                                              ; preds = %960, %956
  %961 = phi i32 [ 0, %956 ], [ %966, %960 ]
  %962 = zext i32 %961 to i64
  %963 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %962
  %964 = load i32, i32* %963, align 4, !noalias !617
  %965 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %962
  store i32 %964, i32* %965, align 4, !alias.scope !617
  %966 = add nuw nsw i32 %961, 1, !spirv.Decorations !620
  %967 = icmp eq i32 %961, 0
  br i1 %967, label %960, label %968, !llvm.loop !622

968:                                              ; preds = %960
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %958)
  %969 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %957)
  %970 = shl i64 %969, 32
  %971 = ashr exact i64 %970, 32
  %972 = mul nsw i64 %971, %const_reg_qword3, !spirv.Decorations !610
  %973 = ashr i64 %969, 32
  %974 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %972, i32 0
  %975 = getelementptr i16, i16 addrspace(4)* %974, i64 %973
  %976 = addrspacecast i16 addrspace(4)* %975 to i16 addrspace(1)*
  %977 = load i16, i16 addrspace(1)* %976, align 2
  %978 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %978)
  %979 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %979)
  %980 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %980, align 4, !noalias !624
  store i32 %953, i32* %57, align 4, !noalias !624
  br label %981

981:                                              ; preds = %981, %968
  %982 = phi i32 [ 0, %968 ], [ %987, %981 ]
  %983 = zext i32 %982 to i64
  %984 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %983
  %985 = load i32, i32* %984, align 4, !noalias !624
  %986 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %983
  store i32 %985, i32* %986, align 4, !alias.scope !624
  %987 = add nuw nsw i32 %982, 1, !spirv.Decorations !620
  %988 = icmp eq i32 %982, 0
  br i1 %988, label %981, label %989, !llvm.loop !627

989:                                              ; preds = %981
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %979)
  %990 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %978)
  %991 = ashr i64 %990, 32
  %992 = mul nsw i64 %991, %const_reg_qword5, !spirv.Decorations !610
  %993 = shl i64 %990, 32
  %994 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %992
  %995 = ashr exact i64 %993, 31
  %996 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %994 to i8 addrspace(4)*
  %997 = getelementptr i8, i8 addrspace(4)* %996, i64 %995
  %998 = bitcast i8 addrspace(4)* %997 to i16 addrspace(4)*
  %999 = addrspacecast i16 addrspace(4)* %998 to i16 addrspace(1)*
  %1000 = load i16, i16 addrspace(1)* %999, align 2
  %1001 = zext i16 %977 to i32
  %1002 = shl nuw i32 %1001, 16, !spirv.Decorations !628
  %1003 = bitcast i32 %1002 to float
  %1004 = zext i16 %1000 to i32
  %1005 = shl nuw i32 %1004, 16, !spirv.Decorations !628
  %1006 = bitcast i32 %1005 to float
  %1007 = fmul reassoc nsz arcp contract float %1003, %1006, !spirv.Decorations !612
  %1008 = fadd reassoc nsz arcp contract float %1007, %.sroa.18.1, !spirv.Decorations !612
  br label %._crit_edge.4

._crit_edge.4:                                    ; preds = %.preheader.3, %989
  %.sroa.18.2 = phi float [ %1008, %989 ], [ %.sroa.18.1, %.preheader.3 ]
  %1009 = and i1 %132, %954
  br i1 %1009, label %1010, label %._crit_edge.1.4

1010:                                             ; preds = %._crit_edge.4
  %1011 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1011)
  %1012 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1012)
  %1013 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %1013, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1014

1014:                                             ; preds = %1014, %1010
  %1015 = phi i32 [ 0, %1010 ], [ %1020, %1014 ]
  %1016 = zext i32 %1015 to i64
  %1017 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1016
  %1018 = load i32, i32* %1017, align 4, !noalias !617
  %1019 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1016
  store i32 %1018, i32* %1019, align 4, !alias.scope !617
  %1020 = add nuw nsw i32 %1015, 1, !spirv.Decorations !620
  %1021 = icmp eq i32 %1015, 0
  br i1 %1021, label %1014, label %1022, !llvm.loop !622

1022:                                             ; preds = %1014
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1012)
  %1023 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1011)
  %1024 = shl i64 %1023, 32
  %1025 = ashr exact i64 %1024, 32
  %1026 = mul nsw i64 %1025, %const_reg_qword3, !spirv.Decorations !610
  %1027 = ashr i64 %1023, 32
  %1028 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1026, i32 0
  %1029 = getelementptr i16, i16 addrspace(4)* %1028, i64 %1027
  %1030 = addrspacecast i16 addrspace(4)* %1029 to i16 addrspace(1)*
  %1031 = load i16, i16 addrspace(1)* %1030, align 2
  %1032 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1032)
  %1033 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1033)
  %1034 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1034, align 4, !noalias !624
  store i32 %953, i32* %57, align 4, !noalias !624
  br label %1035

1035:                                             ; preds = %1035, %1022
  %1036 = phi i32 [ 0, %1022 ], [ %1041, %1035 ]
  %1037 = zext i32 %1036 to i64
  %1038 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1037
  %1039 = load i32, i32* %1038, align 4, !noalias !624
  %1040 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1037
  store i32 %1039, i32* %1040, align 4, !alias.scope !624
  %1041 = add nuw nsw i32 %1036, 1, !spirv.Decorations !620
  %1042 = icmp eq i32 %1036, 0
  br i1 %1042, label %1035, label %1043, !llvm.loop !627

1043:                                             ; preds = %1035
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1033)
  %1044 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1032)
  %1045 = ashr i64 %1044, 32
  %1046 = mul nsw i64 %1045, %const_reg_qword5, !spirv.Decorations !610
  %1047 = shl i64 %1044, 32
  %1048 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1046
  %1049 = ashr exact i64 %1047, 31
  %1050 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1048 to i8 addrspace(4)*
  %1051 = getelementptr i8, i8 addrspace(4)* %1050, i64 %1049
  %1052 = bitcast i8 addrspace(4)* %1051 to i16 addrspace(4)*
  %1053 = addrspacecast i16 addrspace(4)* %1052 to i16 addrspace(1)*
  %1054 = load i16, i16 addrspace(1)* %1053, align 2
  %1055 = zext i16 %1031 to i32
  %1056 = shl nuw i32 %1055, 16, !spirv.Decorations !628
  %1057 = bitcast i32 %1056 to float
  %1058 = zext i16 %1054 to i32
  %1059 = shl nuw i32 %1058, 16, !spirv.Decorations !628
  %1060 = bitcast i32 %1059 to float
  %1061 = fmul reassoc nsz arcp contract float %1057, %1060, !spirv.Decorations !612
  %1062 = fadd reassoc nsz arcp contract float %1061, %.sroa.82.1, !spirv.Decorations !612
  br label %._crit_edge.1.4

._crit_edge.1.4:                                  ; preds = %._crit_edge.4, %1043
  %.sroa.82.2 = phi float [ %1062, %1043 ], [ %.sroa.82.1, %._crit_edge.4 ]
  %1063 = and i1 %188, %954
  br i1 %1063, label %1064, label %._crit_edge.2.4

1064:                                             ; preds = %._crit_edge.1.4
  %1065 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1065)
  %1066 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1066)
  %1067 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %1067, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1068

1068:                                             ; preds = %1068, %1064
  %1069 = phi i32 [ 0, %1064 ], [ %1074, %1068 ]
  %1070 = zext i32 %1069 to i64
  %1071 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1070
  %1072 = load i32, i32* %1071, align 4, !noalias !617
  %1073 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1070
  store i32 %1072, i32* %1073, align 4, !alias.scope !617
  %1074 = add nuw nsw i32 %1069, 1, !spirv.Decorations !620
  %1075 = icmp eq i32 %1069, 0
  br i1 %1075, label %1068, label %1076, !llvm.loop !622

1076:                                             ; preds = %1068
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1066)
  %1077 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1065)
  %1078 = shl i64 %1077, 32
  %1079 = ashr exact i64 %1078, 32
  %1080 = mul nsw i64 %1079, %const_reg_qword3, !spirv.Decorations !610
  %1081 = ashr i64 %1077, 32
  %1082 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1080, i32 0
  %1083 = getelementptr i16, i16 addrspace(4)* %1082, i64 %1081
  %1084 = addrspacecast i16 addrspace(4)* %1083 to i16 addrspace(1)*
  %1085 = load i16, i16 addrspace(1)* %1084, align 2
  %1086 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1086)
  %1087 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1087)
  %1088 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1088, align 4, !noalias !624
  store i32 %953, i32* %57, align 4, !noalias !624
  br label %1089

1089:                                             ; preds = %1089, %1076
  %1090 = phi i32 [ 0, %1076 ], [ %1095, %1089 ]
  %1091 = zext i32 %1090 to i64
  %1092 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1091
  %1093 = load i32, i32* %1092, align 4, !noalias !624
  %1094 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1091
  store i32 %1093, i32* %1094, align 4, !alias.scope !624
  %1095 = add nuw nsw i32 %1090, 1, !spirv.Decorations !620
  %1096 = icmp eq i32 %1090, 0
  br i1 %1096, label %1089, label %1097, !llvm.loop !627

1097:                                             ; preds = %1089
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1087)
  %1098 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1086)
  %1099 = ashr i64 %1098, 32
  %1100 = mul nsw i64 %1099, %const_reg_qword5, !spirv.Decorations !610
  %1101 = shl i64 %1098, 32
  %1102 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1100
  %1103 = ashr exact i64 %1101, 31
  %1104 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1102 to i8 addrspace(4)*
  %1105 = getelementptr i8, i8 addrspace(4)* %1104, i64 %1103
  %1106 = bitcast i8 addrspace(4)* %1105 to i16 addrspace(4)*
  %1107 = addrspacecast i16 addrspace(4)* %1106 to i16 addrspace(1)*
  %1108 = load i16, i16 addrspace(1)* %1107, align 2
  %1109 = zext i16 %1085 to i32
  %1110 = shl nuw i32 %1109, 16, !spirv.Decorations !628
  %1111 = bitcast i32 %1110 to float
  %1112 = zext i16 %1108 to i32
  %1113 = shl nuw i32 %1112, 16, !spirv.Decorations !628
  %1114 = bitcast i32 %1113 to float
  %1115 = fmul reassoc nsz arcp contract float %1111, %1114, !spirv.Decorations !612
  %1116 = fadd reassoc nsz arcp contract float %1115, %.sroa.146.1, !spirv.Decorations !612
  br label %._crit_edge.2.4

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4, %1097
  %.sroa.146.2 = phi float [ %1116, %1097 ], [ %.sroa.146.1, %._crit_edge.1.4 ]
  %1117 = and i1 %244, %954
  br i1 %1117, label %1118, label %.preheader.4

1118:                                             ; preds = %._crit_edge.2.4
  %1119 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1119)
  %1120 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1120)
  %1121 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %1121, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1122

1122:                                             ; preds = %1122, %1118
  %1123 = phi i32 [ 0, %1118 ], [ %1128, %1122 ]
  %1124 = zext i32 %1123 to i64
  %1125 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1124
  %1126 = load i32, i32* %1125, align 4, !noalias !617
  %1127 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1124
  store i32 %1126, i32* %1127, align 4, !alias.scope !617
  %1128 = add nuw nsw i32 %1123, 1, !spirv.Decorations !620
  %1129 = icmp eq i32 %1123, 0
  br i1 %1129, label %1122, label %1130, !llvm.loop !622

1130:                                             ; preds = %1122
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1120)
  %1131 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1119)
  %1132 = shl i64 %1131, 32
  %1133 = ashr exact i64 %1132, 32
  %1134 = mul nsw i64 %1133, %const_reg_qword3, !spirv.Decorations !610
  %1135 = ashr i64 %1131, 32
  %1136 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1134, i32 0
  %1137 = getelementptr i16, i16 addrspace(4)* %1136, i64 %1135
  %1138 = addrspacecast i16 addrspace(4)* %1137 to i16 addrspace(1)*
  %1139 = load i16, i16 addrspace(1)* %1138, align 2
  %1140 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1140)
  %1141 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1141)
  %1142 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1142, align 4, !noalias !624
  store i32 %953, i32* %57, align 4, !noalias !624
  br label %1143

1143:                                             ; preds = %1143, %1130
  %1144 = phi i32 [ 0, %1130 ], [ %1149, %1143 ]
  %1145 = zext i32 %1144 to i64
  %1146 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1145
  %1147 = load i32, i32* %1146, align 4, !noalias !624
  %1148 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1145
  store i32 %1147, i32* %1148, align 4, !alias.scope !624
  %1149 = add nuw nsw i32 %1144, 1, !spirv.Decorations !620
  %1150 = icmp eq i32 %1144, 0
  br i1 %1150, label %1143, label %1151, !llvm.loop !627

1151:                                             ; preds = %1143
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1141)
  %1152 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1140)
  %1153 = ashr i64 %1152, 32
  %1154 = mul nsw i64 %1153, %const_reg_qword5, !spirv.Decorations !610
  %1155 = shl i64 %1152, 32
  %1156 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1154
  %1157 = ashr exact i64 %1155, 31
  %1158 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1156 to i8 addrspace(4)*
  %1159 = getelementptr i8, i8 addrspace(4)* %1158, i64 %1157
  %1160 = bitcast i8 addrspace(4)* %1159 to i16 addrspace(4)*
  %1161 = addrspacecast i16 addrspace(4)* %1160 to i16 addrspace(1)*
  %1162 = load i16, i16 addrspace(1)* %1161, align 2
  %1163 = zext i16 %1139 to i32
  %1164 = shl nuw i32 %1163, 16, !spirv.Decorations !628
  %1165 = bitcast i32 %1164 to float
  %1166 = zext i16 %1162 to i32
  %1167 = shl nuw i32 %1166, 16, !spirv.Decorations !628
  %1168 = bitcast i32 %1167 to float
  %1169 = fmul reassoc nsz arcp contract float %1165, %1168, !spirv.Decorations !612
  %1170 = fadd reassoc nsz arcp contract float %1169, %.sroa.210.1, !spirv.Decorations !612
  br label %.preheader.4

.preheader.4:                                     ; preds = %._crit_edge.2.4, %1151
  %.sroa.210.2 = phi float [ %1170, %1151 ], [ %.sroa.210.1, %._crit_edge.2.4 ]
  %1171 = or i32 %41, 5
  %1172 = icmp slt i32 %1171, %const_reg_dword1
  %1173 = and i1 %76, %1172
  br i1 %1173, label %1174, label %._crit_edge.5

1174:                                             ; preds = %.preheader.4
  %1175 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1175)
  %1176 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1176)
  %1177 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1177, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1178

1178:                                             ; preds = %1178, %1174
  %1179 = phi i32 [ 0, %1174 ], [ %1184, %1178 ]
  %1180 = zext i32 %1179 to i64
  %1181 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1180
  %1182 = load i32, i32* %1181, align 4, !noalias !617
  %1183 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1180
  store i32 %1182, i32* %1183, align 4, !alias.scope !617
  %1184 = add nuw nsw i32 %1179, 1, !spirv.Decorations !620
  %1185 = icmp eq i32 %1179, 0
  br i1 %1185, label %1178, label %1186, !llvm.loop !622

1186:                                             ; preds = %1178
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1176)
  %1187 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1175)
  %1188 = shl i64 %1187, 32
  %1189 = ashr exact i64 %1188, 32
  %1190 = mul nsw i64 %1189, %const_reg_qword3, !spirv.Decorations !610
  %1191 = ashr i64 %1187, 32
  %1192 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1190, i32 0
  %1193 = getelementptr i16, i16 addrspace(4)* %1192, i64 %1191
  %1194 = addrspacecast i16 addrspace(4)* %1193 to i16 addrspace(1)*
  %1195 = load i16, i16 addrspace(1)* %1194, align 2
  %1196 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1196)
  %1197 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1197)
  %1198 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1198, align 4, !noalias !624
  store i32 %1171, i32* %57, align 4, !noalias !624
  br label %1199

1199:                                             ; preds = %1199, %1186
  %1200 = phi i32 [ 0, %1186 ], [ %1205, %1199 ]
  %1201 = zext i32 %1200 to i64
  %1202 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1201
  %1203 = load i32, i32* %1202, align 4, !noalias !624
  %1204 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1201
  store i32 %1203, i32* %1204, align 4, !alias.scope !624
  %1205 = add nuw nsw i32 %1200, 1, !spirv.Decorations !620
  %1206 = icmp eq i32 %1200, 0
  br i1 %1206, label %1199, label %1207, !llvm.loop !627

1207:                                             ; preds = %1199
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1197)
  %1208 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1196)
  %1209 = ashr i64 %1208, 32
  %1210 = mul nsw i64 %1209, %const_reg_qword5, !spirv.Decorations !610
  %1211 = shl i64 %1208, 32
  %1212 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1210
  %1213 = ashr exact i64 %1211, 31
  %1214 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1212 to i8 addrspace(4)*
  %1215 = getelementptr i8, i8 addrspace(4)* %1214, i64 %1213
  %1216 = bitcast i8 addrspace(4)* %1215 to i16 addrspace(4)*
  %1217 = addrspacecast i16 addrspace(4)* %1216 to i16 addrspace(1)*
  %1218 = load i16, i16 addrspace(1)* %1217, align 2
  %1219 = zext i16 %1195 to i32
  %1220 = shl nuw i32 %1219, 16, !spirv.Decorations !628
  %1221 = bitcast i32 %1220 to float
  %1222 = zext i16 %1218 to i32
  %1223 = shl nuw i32 %1222, 16, !spirv.Decorations !628
  %1224 = bitcast i32 %1223 to float
  %1225 = fmul reassoc nsz arcp contract float %1221, %1224, !spirv.Decorations !612
  %1226 = fadd reassoc nsz arcp contract float %1225, %.sroa.22.1, !spirv.Decorations !612
  br label %._crit_edge.5

._crit_edge.5:                                    ; preds = %.preheader.4, %1207
  %.sroa.22.2 = phi float [ %1226, %1207 ], [ %.sroa.22.1, %.preheader.4 ]
  %1227 = and i1 %132, %1172
  br i1 %1227, label %1228, label %._crit_edge.1.5

1228:                                             ; preds = %._crit_edge.5
  %1229 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1229)
  %1230 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1230)
  %1231 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %1231, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1232

1232:                                             ; preds = %1232, %1228
  %1233 = phi i32 [ 0, %1228 ], [ %1238, %1232 ]
  %1234 = zext i32 %1233 to i64
  %1235 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1234
  %1236 = load i32, i32* %1235, align 4, !noalias !617
  %1237 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1234
  store i32 %1236, i32* %1237, align 4, !alias.scope !617
  %1238 = add nuw nsw i32 %1233, 1, !spirv.Decorations !620
  %1239 = icmp eq i32 %1233, 0
  br i1 %1239, label %1232, label %1240, !llvm.loop !622

1240:                                             ; preds = %1232
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1230)
  %1241 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1229)
  %1242 = shl i64 %1241, 32
  %1243 = ashr exact i64 %1242, 32
  %1244 = mul nsw i64 %1243, %const_reg_qword3, !spirv.Decorations !610
  %1245 = ashr i64 %1241, 32
  %1246 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1244, i32 0
  %1247 = getelementptr i16, i16 addrspace(4)* %1246, i64 %1245
  %1248 = addrspacecast i16 addrspace(4)* %1247 to i16 addrspace(1)*
  %1249 = load i16, i16 addrspace(1)* %1248, align 2
  %1250 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1250)
  %1251 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1251)
  %1252 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1252, align 4, !noalias !624
  store i32 %1171, i32* %57, align 4, !noalias !624
  br label %1253

1253:                                             ; preds = %1253, %1240
  %1254 = phi i32 [ 0, %1240 ], [ %1259, %1253 ]
  %1255 = zext i32 %1254 to i64
  %1256 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1255
  %1257 = load i32, i32* %1256, align 4, !noalias !624
  %1258 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1255
  store i32 %1257, i32* %1258, align 4, !alias.scope !624
  %1259 = add nuw nsw i32 %1254, 1, !spirv.Decorations !620
  %1260 = icmp eq i32 %1254, 0
  br i1 %1260, label %1253, label %1261, !llvm.loop !627

1261:                                             ; preds = %1253
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1251)
  %1262 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1250)
  %1263 = ashr i64 %1262, 32
  %1264 = mul nsw i64 %1263, %const_reg_qword5, !spirv.Decorations !610
  %1265 = shl i64 %1262, 32
  %1266 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1264
  %1267 = ashr exact i64 %1265, 31
  %1268 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1266 to i8 addrspace(4)*
  %1269 = getelementptr i8, i8 addrspace(4)* %1268, i64 %1267
  %1270 = bitcast i8 addrspace(4)* %1269 to i16 addrspace(4)*
  %1271 = addrspacecast i16 addrspace(4)* %1270 to i16 addrspace(1)*
  %1272 = load i16, i16 addrspace(1)* %1271, align 2
  %1273 = zext i16 %1249 to i32
  %1274 = shl nuw i32 %1273, 16, !spirv.Decorations !628
  %1275 = bitcast i32 %1274 to float
  %1276 = zext i16 %1272 to i32
  %1277 = shl nuw i32 %1276, 16, !spirv.Decorations !628
  %1278 = bitcast i32 %1277 to float
  %1279 = fmul reassoc nsz arcp contract float %1275, %1278, !spirv.Decorations !612
  %1280 = fadd reassoc nsz arcp contract float %1279, %.sroa.86.1, !spirv.Decorations !612
  br label %._crit_edge.1.5

._crit_edge.1.5:                                  ; preds = %._crit_edge.5, %1261
  %.sroa.86.2 = phi float [ %1280, %1261 ], [ %.sroa.86.1, %._crit_edge.5 ]
  %1281 = and i1 %188, %1172
  br i1 %1281, label %1282, label %._crit_edge.2.5

1282:                                             ; preds = %._crit_edge.1.5
  %1283 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1283)
  %1284 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1284)
  %1285 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %1285, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1286

1286:                                             ; preds = %1286, %1282
  %1287 = phi i32 [ 0, %1282 ], [ %1292, %1286 ]
  %1288 = zext i32 %1287 to i64
  %1289 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1288
  %1290 = load i32, i32* %1289, align 4, !noalias !617
  %1291 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1288
  store i32 %1290, i32* %1291, align 4, !alias.scope !617
  %1292 = add nuw nsw i32 %1287, 1, !spirv.Decorations !620
  %1293 = icmp eq i32 %1287, 0
  br i1 %1293, label %1286, label %1294, !llvm.loop !622

1294:                                             ; preds = %1286
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1284)
  %1295 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1283)
  %1296 = shl i64 %1295, 32
  %1297 = ashr exact i64 %1296, 32
  %1298 = mul nsw i64 %1297, %const_reg_qword3, !spirv.Decorations !610
  %1299 = ashr i64 %1295, 32
  %1300 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1298, i32 0
  %1301 = getelementptr i16, i16 addrspace(4)* %1300, i64 %1299
  %1302 = addrspacecast i16 addrspace(4)* %1301 to i16 addrspace(1)*
  %1303 = load i16, i16 addrspace(1)* %1302, align 2
  %1304 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1304)
  %1305 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1305)
  %1306 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1306, align 4, !noalias !624
  store i32 %1171, i32* %57, align 4, !noalias !624
  br label %1307

1307:                                             ; preds = %1307, %1294
  %1308 = phi i32 [ 0, %1294 ], [ %1313, %1307 ]
  %1309 = zext i32 %1308 to i64
  %1310 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1309
  %1311 = load i32, i32* %1310, align 4, !noalias !624
  %1312 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1309
  store i32 %1311, i32* %1312, align 4, !alias.scope !624
  %1313 = add nuw nsw i32 %1308, 1, !spirv.Decorations !620
  %1314 = icmp eq i32 %1308, 0
  br i1 %1314, label %1307, label %1315, !llvm.loop !627

1315:                                             ; preds = %1307
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1305)
  %1316 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1304)
  %1317 = ashr i64 %1316, 32
  %1318 = mul nsw i64 %1317, %const_reg_qword5, !spirv.Decorations !610
  %1319 = shl i64 %1316, 32
  %1320 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1318
  %1321 = ashr exact i64 %1319, 31
  %1322 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1320 to i8 addrspace(4)*
  %1323 = getelementptr i8, i8 addrspace(4)* %1322, i64 %1321
  %1324 = bitcast i8 addrspace(4)* %1323 to i16 addrspace(4)*
  %1325 = addrspacecast i16 addrspace(4)* %1324 to i16 addrspace(1)*
  %1326 = load i16, i16 addrspace(1)* %1325, align 2
  %1327 = zext i16 %1303 to i32
  %1328 = shl nuw i32 %1327, 16, !spirv.Decorations !628
  %1329 = bitcast i32 %1328 to float
  %1330 = zext i16 %1326 to i32
  %1331 = shl nuw i32 %1330, 16, !spirv.Decorations !628
  %1332 = bitcast i32 %1331 to float
  %1333 = fmul reassoc nsz arcp contract float %1329, %1332, !spirv.Decorations !612
  %1334 = fadd reassoc nsz arcp contract float %1333, %.sroa.150.1, !spirv.Decorations !612
  br label %._crit_edge.2.5

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5, %1315
  %.sroa.150.2 = phi float [ %1334, %1315 ], [ %.sroa.150.1, %._crit_edge.1.5 ]
  %1335 = and i1 %244, %1172
  br i1 %1335, label %1336, label %.preheader.5

1336:                                             ; preds = %._crit_edge.2.5
  %1337 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1337)
  %1338 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1338)
  %1339 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %1339, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1340

1340:                                             ; preds = %1340, %1336
  %1341 = phi i32 [ 0, %1336 ], [ %1346, %1340 ]
  %1342 = zext i32 %1341 to i64
  %1343 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1342
  %1344 = load i32, i32* %1343, align 4, !noalias !617
  %1345 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1342
  store i32 %1344, i32* %1345, align 4, !alias.scope !617
  %1346 = add nuw nsw i32 %1341, 1, !spirv.Decorations !620
  %1347 = icmp eq i32 %1341, 0
  br i1 %1347, label %1340, label %1348, !llvm.loop !622

1348:                                             ; preds = %1340
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1338)
  %1349 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1337)
  %1350 = shl i64 %1349, 32
  %1351 = ashr exact i64 %1350, 32
  %1352 = mul nsw i64 %1351, %const_reg_qword3, !spirv.Decorations !610
  %1353 = ashr i64 %1349, 32
  %1354 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1352, i32 0
  %1355 = getelementptr i16, i16 addrspace(4)* %1354, i64 %1353
  %1356 = addrspacecast i16 addrspace(4)* %1355 to i16 addrspace(1)*
  %1357 = load i16, i16 addrspace(1)* %1356, align 2
  %1358 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1358)
  %1359 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1359)
  %1360 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1360, align 4, !noalias !624
  store i32 %1171, i32* %57, align 4, !noalias !624
  br label %1361

1361:                                             ; preds = %1361, %1348
  %1362 = phi i32 [ 0, %1348 ], [ %1367, %1361 ]
  %1363 = zext i32 %1362 to i64
  %1364 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1363
  %1365 = load i32, i32* %1364, align 4, !noalias !624
  %1366 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1363
  store i32 %1365, i32* %1366, align 4, !alias.scope !624
  %1367 = add nuw nsw i32 %1362, 1, !spirv.Decorations !620
  %1368 = icmp eq i32 %1362, 0
  br i1 %1368, label %1361, label %1369, !llvm.loop !627

1369:                                             ; preds = %1361
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1359)
  %1370 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1358)
  %1371 = ashr i64 %1370, 32
  %1372 = mul nsw i64 %1371, %const_reg_qword5, !spirv.Decorations !610
  %1373 = shl i64 %1370, 32
  %1374 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1372
  %1375 = ashr exact i64 %1373, 31
  %1376 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1374 to i8 addrspace(4)*
  %1377 = getelementptr i8, i8 addrspace(4)* %1376, i64 %1375
  %1378 = bitcast i8 addrspace(4)* %1377 to i16 addrspace(4)*
  %1379 = addrspacecast i16 addrspace(4)* %1378 to i16 addrspace(1)*
  %1380 = load i16, i16 addrspace(1)* %1379, align 2
  %1381 = zext i16 %1357 to i32
  %1382 = shl nuw i32 %1381, 16, !spirv.Decorations !628
  %1383 = bitcast i32 %1382 to float
  %1384 = zext i16 %1380 to i32
  %1385 = shl nuw i32 %1384, 16, !spirv.Decorations !628
  %1386 = bitcast i32 %1385 to float
  %1387 = fmul reassoc nsz arcp contract float %1383, %1386, !spirv.Decorations !612
  %1388 = fadd reassoc nsz arcp contract float %1387, %.sroa.214.1, !spirv.Decorations !612
  br label %.preheader.5

.preheader.5:                                     ; preds = %._crit_edge.2.5, %1369
  %.sroa.214.2 = phi float [ %1388, %1369 ], [ %.sroa.214.1, %._crit_edge.2.5 ]
  %1389 = or i32 %41, 6
  %1390 = icmp slt i32 %1389, %const_reg_dword1
  %1391 = and i1 %76, %1390
  br i1 %1391, label %1392, label %._crit_edge.6

1392:                                             ; preds = %.preheader.5
  %1393 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1393)
  %1394 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1394)
  %1395 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1395, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1396

1396:                                             ; preds = %1396, %1392
  %1397 = phi i32 [ 0, %1392 ], [ %1402, %1396 ]
  %1398 = zext i32 %1397 to i64
  %1399 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1398
  %1400 = load i32, i32* %1399, align 4, !noalias !617
  %1401 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1398
  store i32 %1400, i32* %1401, align 4, !alias.scope !617
  %1402 = add nuw nsw i32 %1397, 1, !spirv.Decorations !620
  %1403 = icmp eq i32 %1397, 0
  br i1 %1403, label %1396, label %1404, !llvm.loop !622

1404:                                             ; preds = %1396
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1394)
  %1405 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1393)
  %1406 = shl i64 %1405, 32
  %1407 = ashr exact i64 %1406, 32
  %1408 = mul nsw i64 %1407, %const_reg_qword3, !spirv.Decorations !610
  %1409 = ashr i64 %1405, 32
  %1410 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1408, i32 0
  %1411 = getelementptr i16, i16 addrspace(4)* %1410, i64 %1409
  %1412 = addrspacecast i16 addrspace(4)* %1411 to i16 addrspace(1)*
  %1413 = load i16, i16 addrspace(1)* %1412, align 2
  %1414 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1414)
  %1415 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1415)
  %1416 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1416, align 4, !noalias !624
  store i32 %1389, i32* %57, align 4, !noalias !624
  br label %1417

1417:                                             ; preds = %1417, %1404
  %1418 = phi i32 [ 0, %1404 ], [ %1423, %1417 ]
  %1419 = zext i32 %1418 to i64
  %1420 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1419
  %1421 = load i32, i32* %1420, align 4, !noalias !624
  %1422 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1419
  store i32 %1421, i32* %1422, align 4, !alias.scope !624
  %1423 = add nuw nsw i32 %1418, 1, !spirv.Decorations !620
  %1424 = icmp eq i32 %1418, 0
  br i1 %1424, label %1417, label %1425, !llvm.loop !627

1425:                                             ; preds = %1417
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1415)
  %1426 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1414)
  %1427 = ashr i64 %1426, 32
  %1428 = mul nsw i64 %1427, %const_reg_qword5, !spirv.Decorations !610
  %1429 = shl i64 %1426, 32
  %1430 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1428
  %1431 = ashr exact i64 %1429, 31
  %1432 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1430 to i8 addrspace(4)*
  %1433 = getelementptr i8, i8 addrspace(4)* %1432, i64 %1431
  %1434 = bitcast i8 addrspace(4)* %1433 to i16 addrspace(4)*
  %1435 = addrspacecast i16 addrspace(4)* %1434 to i16 addrspace(1)*
  %1436 = load i16, i16 addrspace(1)* %1435, align 2
  %1437 = zext i16 %1413 to i32
  %1438 = shl nuw i32 %1437, 16, !spirv.Decorations !628
  %1439 = bitcast i32 %1438 to float
  %1440 = zext i16 %1436 to i32
  %1441 = shl nuw i32 %1440, 16, !spirv.Decorations !628
  %1442 = bitcast i32 %1441 to float
  %1443 = fmul reassoc nsz arcp contract float %1439, %1442, !spirv.Decorations !612
  %1444 = fadd reassoc nsz arcp contract float %1443, %.sroa.26.1, !spirv.Decorations !612
  br label %._crit_edge.6

._crit_edge.6:                                    ; preds = %.preheader.5, %1425
  %.sroa.26.2 = phi float [ %1444, %1425 ], [ %.sroa.26.1, %.preheader.5 ]
  %1445 = and i1 %132, %1390
  br i1 %1445, label %1446, label %._crit_edge.1.6

1446:                                             ; preds = %._crit_edge.6
  %1447 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1447)
  %1448 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1448)
  %1449 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %1449, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1450

1450:                                             ; preds = %1450, %1446
  %1451 = phi i32 [ 0, %1446 ], [ %1456, %1450 ]
  %1452 = zext i32 %1451 to i64
  %1453 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1452
  %1454 = load i32, i32* %1453, align 4, !noalias !617
  %1455 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1452
  store i32 %1454, i32* %1455, align 4, !alias.scope !617
  %1456 = add nuw nsw i32 %1451, 1, !spirv.Decorations !620
  %1457 = icmp eq i32 %1451, 0
  br i1 %1457, label %1450, label %1458, !llvm.loop !622

1458:                                             ; preds = %1450
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1448)
  %1459 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1447)
  %1460 = shl i64 %1459, 32
  %1461 = ashr exact i64 %1460, 32
  %1462 = mul nsw i64 %1461, %const_reg_qword3, !spirv.Decorations !610
  %1463 = ashr i64 %1459, 32
  %1464 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1462, i32 0
  %1465 = getelementptr i16, i16 addrspace(4)* %1464, i64 %1463
  %1466 = addrspacecast i16 addrspace(4)* %1465 to i16 addrspace(1)*
  %1467 = load i16, i16 addrspace(1)* %1466, align 2
  %1468 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1468)
  %1469 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1469)
  %1470 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1470, align 4, !noalias !624
  store i32 %1389, i32* %57, align 4, !noalias !624
  br label %1471

1471:                                             ; preds = %1471, %1458
  %1472 = phi i32 [ 0, %1458 ], [ %1477, %1471 ]
  %1473 = zext i32 %1472 to i64
  %1474 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1473
  %1475 = load i32, i32* %1474, align 4, !noalias !624
  %1476 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1473
  store i32 %1475, i32* %1476, align 4, !alias.scope !624
  %1477 = add nuw nsw i32 %1472, 1, !spirv.Decorations !620
  %1478 = icmp eq i32 %1472, 0
  br i1 %1478, label %1471, label %1479, !llvm.loop !627

1479:                                             ; preds = %1471
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1469)
  %1480 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1468)
  %1481 = ashr i64 %1480, 32
  %1482 = mul nsw i64 %1481, %const_reg_qword5, !spirv.Decorations !610
  %1483 = shl i64 %1480, 32
  %1484 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1482
  %1485 = ashr exact i64 %1483, 31
  %1486 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1484 to i8 addrspace(4)*
  %1487 = getelementptr i8, i8 addrspace(4)* %1486, i64 %1485
  %1488 = bitcast i8 addrspace(4)* %1487 to i16 addrspace(4)*
  %1489 = addrspacecast i16 addrspace(4)* %1488 to i16 addrspace(1)*
  %1490 = load i16, i16 addrspace(1)* %1489, align 2
  %1491 = zext i16 %1467 to i32
  %1492 = shl nuw i32 %1491, 16, !spirv.Decorations !628
  %1493 = bitcast i32 %1492 to float
  %1494 = zext i16 %1490 to i32
  %1495 = shl nuw i32 %1494, 16, !spirv.Decorations !628
  %1496 = bitcast i32 %1495 to float
  %1497 = fmul reassoc nsz arcp contract float %1493, %1496, !spirv.Decorations !612
  %1498 = fadd reassoc nsz arcp contract float %1497, %.sroa.90.1, !spirv.Decorations !612
  br label %._crit_edge.1.6

._crit_edge.1.6:                                  ; preds = %._crit_edge.6, %1479
  %.sroa.90.2 = phi float [ %1498, %1479 ], [ %.sroa.90.1, %._crit_edge.6 ]
  %1499 = and i1 %188, %1390
  br i1 %1499, label %1500, label %._crit_edge.2.6

1500:                                             ; preds = %._crit_edge.1.6
  %1501 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1501)
  %1502 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1502)
  %1503 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %1503, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1504

1504:                                             ; preds = %1504, %1500
  %1505 = phi i32 [ 0, %1500 ], [ %1510, %1504 ]
  %1506 = zext i32 %1505 to i64
  %1507 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1506
  %1508 = load i32, i32* %1507, align 4, !noalias !617
  %1509 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1506
  store i32 %1508, i32* %1509, align 4, !alias.scope !617
  %1510 = add nuw nsw i32 %1505, 1, !spirv.Decorations !620
  %1511 = icmp eq i32 %1505, 0
  br i1 %1511, label %1504, label %1512, !llvm.loop !622

1512:                                             ; preds = %1504
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1502)
  %1513 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1501)
  %1514 = shl i64 %1513, 32
  %1515 = ashr exact i64 %1514, 32
  %1516 = mul nsw i64 %1515, %const_reg_qword3, !spirv.Decorations !610
  %1517 = ashr i64 %1513, 32
  %1518 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1516, i32 0
  %1519 = getelementptr i16, i16 addrspace(4)* %1518, i64 %1517
  %1520 = addrspacecast i16 addrspace(4)* %1519 to i16 addrspace(1)*
  %1521 = load i16, i16 addrspace(1)* %1520, align 2
  %1522 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1522)
  %1523 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1523)
  %1524 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1524, align 4, !noalias !624
  store i32 %1389, i32* %57, align 4, !noalias !624
  br label %1525

1525:                                             ; preds = %1525, %1512
  %1526 = phi i32 [ 0, %1512 ], [ %1531, %1525 ]
  %1527 = zext i32 %1526 to i64
  %1528 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1527
  %1529 = load i32, i32* %1528, align 4, !noalias !624
  %1530 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1527
  store i32 %1529, i32* %1530, align 4, !alias.scope !624
  %1531 = add nuw nsw i32 %1526, 1, !spirv.Decorations !620
  %1532 = icmp eq i32 %1526, 0
  br i1 %1532, label %1525, label %1533, !llvm.loop !627

1533:                                             ; preds = %1525
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1523)
  %1534 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1522)
  %1535 = ashr i64 %1534, 32
  %1536 = mul nsw i64 %1535, %const_reg_qword5, !spirv.Decorations !610
  %1537 = shl i64 %1534, 32
  %1538 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1536
  %1539 = ashr exact i64 %1537, 31
  %1540 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1538 to i8 addrspace(4)*
  %1541 = getelementptr i8, i8 addrspace(4)* %1540, i64 %1539
  %1542 = bitcast i8 addrspace(4)* %1541 to i16 addrspace(4)*
  %1543 = addrspacecast i16 addrspace(4)* %1542 to i16 addrspace(1)*
  %1544 = load i16, i16 addrspace(1)* %1543, align 2
  %1545 = zext i16 %1521 to i32
  %1546 = shl nuw i32 %1545, 16, !spirv.Decorations !628
  %1547 = bitcast i32 %1546 to float
  %1548 = zext i16 %1544 to i32
  %1549 = shl nuw i32 %1548, 16, !spirv.Decorations !628
  %1550 = bitcast i32 %1549 to float
  %1551 = fmul reassoc nsz arcp contract float %1547, %1550, !spirv.Decorations !612
  %1552 = fadd reassoc nsz arcp contract float %1551, %.sroa.154.1, !spirv.Decorations !612
  br label %._crit_edge.2.6

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6, %1533
  %.sroa.154.2 = phi float [ %1552, %1533 ], [ %.sroa.154.1, %._crit_edge.1.6 ]
  %1553 = and i1 %244, %1390
  br i1 %1553, label %1554, label %.preheader.6

1554:                                             ; preds = %._crit_edge.2.6
  %1555 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1555)
  %1556 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1556)
  %1557 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %1557, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1558

1558:                                             ; preds = %1558, %1554
  %1559 = phi i32 [ 0, %1554 ], [ %1564, %1558 ]
  %1560 = zext i32 %1559 to i64
  %1561 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1560
  %1562 = load i32, i32* %1561, align 4, !noalias !617
  %1563 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1560
  store i32 %1562, i32* %1563, align 4, !alias.scope !617
  %1564 = add nuw nsw i32 %1559, 1, !spirv.Decorations !620
  %1565 = icmp eq i32 %1559, 0
  br i1 %1565, label %1558, label %1566, !llvm.loop !622

1566:                                             ; preds = %1558
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1556)
  %1567 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1555)
  %1568 = shl i64 %1567, 32
  %1569 = ashr exact i64 %1568, 32
  %1570 = mul nsw i64 %1569, %const_reg_qword3, !spirv.Decorations !610
  %1571 = ashr i64 %1567, 32
  %1572 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1570, i32 0
  %1573 = getelementptr i16, i16 addrspace(4)* %1572, i64 %1571
  %1574 = addrspacecast i16 addrspace(4)* %1573 to i16 addrspace(1)*
  %1575 = load i16, i16 addrspace(1)* %1574, align 2
  %1576 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1576)
  %1577 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1577)
  %1578 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1578, align 4, !noalias !624
  store i32 %1389, i32* %57, align 4, !noalias !624
  br label %1579

1579:                                             ; preds = %1579, %1566
  %1580 = phi i32 [ 0, %1566 ], [ %1585, %1579 ]
  %1581 = zext i32 %1580 to i64
  %1582 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1581
  %1583 = load i32, i32* %1582, align 4, !noalias !624
  %1584 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1581
  store i32 %1583, i32* %1584, align 4, !alias.scope !624
  %1585 = add nuw nsw i32 %1580, 1, !spirv.Decorations !620
  %1586 = icmp eq i32 %1580, 0
  br i1 %1586, label %1579, label %1587, !llvm.loop !627

1587:                                             ; preds = %1579
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1577)
  %1588 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1576)
  %1589 = ashr i64 %1588, 32
  %1590 = mul nsw i64 %1589, %const_reg_qword5, !spirv.Decorations !610
  %1591 = shl i64 %1588, 32
  %1592 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1590
  %1593 = ashr exact i64 %1591, 31
  %1594 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1592 to i8 addrspace(4)*
  %1595 = getelementptr i8, i8 addrspace(4)* %1594, i64 %1593
  %1596 = bitcast i8 addrspace(4)* %1595 to i16 addrspace(4)*
  %1597 = addrspacecast i16 addrspace(4)* %1596 to i16 addrspace(1)*
  %1598 = load i16, i16 addrspace(1)* %1597, align 2
  %1599 = zext i16 %1575 to i32
  %1600 = shl nuw i32 %1599, 16, !spirv.Decorations !628
  %1601 = bitcast i32 %1600 to float
  %1602 = zext i16 %1598 to i32
  %1603 = shl nuw i32 %1602, 16, !spirv.Decorations !628
  %1604 = bitcast i32 %1603 to float
  %1605 = fmul reassoc nsz arcp contract float %1601, %1604, !spirv.Decorations !612
  %1606 = fadd reassoc nsz arcp contract float %1605, %.sroa.218.1, !spirv.Decorations !612
  br label %.preheader.6

.preheader.6:                                     ; preds = %._crit_edge.2.6, %1587
  %.sroa.218.2 = phi float [ %1606, %1587 ], [ %.sroa.218.1, %._crit_edge.2.6 ]
  %1607 = or i32 %41, 7
  %1608 = icmp slt i32 %1607, %const_reg_dword1
  %1609 = and i1 %76, %1608
  br i1 %1609, label %1610, label %._crit_edge.7

1610:                                             ; preds = %.preheader.6
  %1611 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1611)
  %1612 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1612)
  %1613 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1613, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1614

1614:                                             ; preds = %1614, %1610
  %1615 = phi i32 [ 0, %1610 ], [ %1620, %1614 ]
  %1616 = zext i32 %1615 to i64
  %1617 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1616
  %1618 = load i32, i32* %1617, align 4, !noalias !617
  %1619 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1616
  store i32 %1618, i32* %1619, align 4, !alias.scope !617
  %1620 = add nuw nsw i32 %1615, 1, !spirv.Decorations !620
  %1621 = icmp eq i32 %1615, 0
  br i1 %1621, label %1614, label %1622, !llvm.loop !622

1622:                                             ; preds = %1614
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1612)
  %1623 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1611)
  %1624 = shl i64 %1623, 32
  %1625 = ashr exact i64 %1624, 32
  %1626 = mul nsw i64 %1625, %const_reg_qword3, !spirv.Decorations !610
  %1627 = ashr i64 %1623, 32
  %1628 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1626, i32 0
  %1629 = getelementptr i16, i16 addrspace(4)* %1628, i64 %1627
  %1630 = addrspacecast i16 addrspace(4)* %1629 to i16 addrspace(1)*
  %1631 = load i16, i16 addrspace(1)* %1630, align 2
  %1632 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1632)
  %1633 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1633)
  %1634 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1634, align 4, !noalias !624
  store i32 %1607, i32* %57, align 4, !noalias !624
  br label %1635

1635:                                             ; preds = %1635, %1622
  %1636 = phi i32 [ 0, %1622 ], [ %1641, %1635 ]
  %1637 = zext i32 %1636 to i64
  %1638 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1637
  %1639 = load i32, i32* %1638, align 4, !noalias !624
  %1640 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1637
  store i32 %1639, i32* %1640, align 4, !alias.scope !624
  %1641 = add nuw nsw i32 %1636, 1, !spirv.Decorations !620
  %1642 = icmp eq i32 %1636, 0
  br i1 %1642, label %1635, label %1643, !llvm.loop !627

1643:                                             ; preds = %1635
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1633)
  %1644 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1632)
  %1645 = ashr i64 %1644, 32
  %1646 = mul nsw i64 %1645, %const_reg_qword5, !spirv.Decorations !610
  %1647 = shl i64 %1644, 32
  %1648 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1646
  %1649 = ashr exact i64 %1647, 31
  %1650 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1648 to i8 addrspace(4)*
  %1651 = getelementptr i8, i8 addrspace(4)* %1650, i64 %1649
  %1652 = bitcast i8 addrspace(4)* %1651 to i16 addrspace(4)*
  %1653 = addrspacecast i16 addrspace(4)* %1652 to i16 addrspace(1)*
  %1654 = load i16, i16 addrspace(1)* %1653, align 2
  %1655 = zext i16 %1631 to i32
  %1656 = shl nuw i32 %1655, 16, !spirv.Decorations !628
  %1657 = bitcast i32 %1656 to float
  %1658 = zext i16 %1654 to i32
  %1659 = shl nuw i32 %1658, 16, !spirv.Decorations !628
  %1660 = bitcast i32 %1659 to float
  %1661 = fmul reassoc nsz arcp contract float %1657, %1660, !spirv.Decorations !612
  %1662 = fadd reassoc nsz arcp contract float %1661, %.sroa.30.1, !spirv.Decorations !612
  br label %._crit_edge.7

._crit_edge.7:                                    ; preds = %.preheader.6, %1643
  %.sroa.30.2 = phi float [ %1662, %1643 ], [ %.sroa.30.1, %.preheader.6 ]
  %1663 = and i1 %132, %1608
  br i1 %1663, label %1664, label %._crit_edge.1.7

1664:                                             ; preds = %._crit_edge.7
  %1665 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1665)
  %1666 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1666)
  %1667 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %1667, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1668

1668:                                             ; preds = %1668, %1664
  %1669 = phi i32 [ 0, %1664 ], [ %1674, %1668 ]
  %1670 = zext i32 %1669 to i64
  %1671 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1670
  %1672 = load i32, i32* %1671, align 4, !noalias !617
  %1673 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1670
  store i32 %1672, i32* %1673, align 4, !alias.scope !617
  %1674 = add nuw nsw i32 %1669, 1, !spirv.Decorations !620
  %1675 = icmp eq i32 %1669, 0
  br i1 %1675, label %1668, label %1676, !llvm.loop !622

1676:                                             ; preds = %1668
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1666)
  %1677 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1665)
  %1678 = shl i64 %1677, 32
  %1679 = ashr exact i64 %1678, 32
  %1680 = mul nsw i64 %1679, %const_reg_qword3, !spirv.Decorations !610
  %1681 = ashr i64 %1677, 32
  %1682 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1680, i32 0
  %1683 = getelementptr i16, i16 addrspace(4)* %1682, i64 %1681
  %1684 = addrspacecast i16 addrspace(4)* %1683 to i16 addrspace(1)*
  %1685 = load i16, i16 addrspace(1)* %1684, align 2
  %1686 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1686)
  %1687 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1687)
  %1688 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1688, align 4, !noalias !624
  store i32 %1607, i32* %57, align 4, !noalias !624
  br label %1689

1689:                                             ; preds = %1689, %1676
  %1690 = phi i32 [ 0, %1676 ], [ %1695, %1689 ]
  %1691 = zext i32 %1690 to i64
  %1692 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1691
  %1693 = load i32, i32* %1692, align 4, !noalias !624
  %1694 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1691
  store i32 %1693, i32* %1694, align 4, !alias.scope !624
  %1695 = add nuw nsw i32 %1690, 1, !spirv.Decorations !620
  %1696 = icmp eq i32 %1690, 0
  br i1 %1696, label %1689, label %1697, !llvm.loop !627

1697:                                             ; preds = %1689
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1687)
  %1698 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1686)
  %1699 = ashr i64 %1698, 32
  %1700 = mul nsw i64 %1699, %const_reg_qword5, !spirv.Decorations !610
  %1701 = shl i64 %1698, 32
  %1702 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1700
  %1703 = ashr exact i64 %1701, 31
  %1704 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1702 to i8 addrspace(4)*
  %1705 = getelementptr i8, i8 addrspace(4)* %1704, i64 %1703
  %1706 = bitcast i8 addrspace(4)* %1705 to i16 addrspace(4)*
  %1707 = addrspacecast i16 addrspace(4)* %1706 to i16 addrspace(1)*
  %1708 = load i16, i16 addrspace(1)* %1707, align 2
  %1709 = zext i16 %1685 to i32
  %1710 = shl nuw i32 %1709, 16, !spirv.Decorations !628
  %1711 = bitcast i32 %1710 to float
  %1712 = zext i16 %1708 to i32
  %1713 = shl nuw i32 %1712, 16, !spirv.Decorations !628
  %1714 = bitcast i32 %1713 to float
  %1715 = fmul reassoc nsz arcp contract float %1711, %1714, !spirv.Decorations !612
  %1716 = fadd reassoc nsz arcp contract float %1715, %.sroa.94.1, !spirv.Decorations !612
  br label %._crit_edge.1.7

._crit_edge.1.7:                                  ; preds = %._crit_edge.7, %1697
  %.sroa.94.2 = phi float [ %1716, %1697 ], [ %.sroa.94.1, %._crit_edge.7 ]
  %1717 = and i1 %188, %1608
  br i1 %1717, label %1718, label %._crit_edge.2.7

1718:                                             ; preds = %._crit_edge.1.7
  %1719 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1719)
  %1720 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1720)
  %1721 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %1721, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1722

1722:                                             ; preds = %1722, %1718
  %1723 = phi i32 [ 0, %1718 ], [ %1728, %1722 ]
  %1724 = zext i32 %1723 to i64
  %1725 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1724
  %1726 = load i32, i32* %1725, align 4, !noalias !617
  %1727 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1724
  store i32 %1726, i32* %1727, align 4, !alias.scope !617
  %1728 = add nuw nsw i32 %1723, 1, !spirv.Decorations !620
  %1729 = icmp eq i32 %1723, 0
  br i1 %1729, label %1722, label %1730, !llvm.loop !622

1730:                                             ; preds = %1722
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1720)
  %1731 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1719)
  %1732 = shl i64 %1731, 32
  %1733 = ashr exact i64 %1732, 32
  %1734 = mul nsw i64 %1733, %const_reg_qword3, !spirv.Decorations !610
  %1735 = ashr i64 %1731, 32
  %1736 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1734, i32 0
  %1737 = getelementptr i16, i16 addrspace(4)* %1736, i64 %1735
  %1738 = addrspacecast i16 addrspace(4)* %1737 to i16 addrspace(1)*
  %1739 = load i16, i16 addrspace(1)* %1738, align 2
  %1740 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1740)
  %1741 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1741)
  %1742 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1742, align 4, !noalias !624
  store i32 %1607, i32* %57, align 4, !noalias !624
  br label %1743

1743:                                             ; preds = %1743, %1730
  %1744 = phi i32 [ 0, %1730 ], [ %1749, %1743 ]
  %1745 = zext i32 %1744 to i64
  %1746 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1745
  %1747 = load i32, i32* %1746, align 4, !noalias !624
  %1748 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1745
  store i32 %1747, i32* %1748, align 4, !alias.scope !624
  %1749 = add nuw nsw i32 %1744, 1, !spirv.Decorations !620
  %1750 = icmp eq i32 %1744, 0
  br i1 %1750, label %1743, label %1751, !llvm.loop !627

1751:                                             ; preds = %1743
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1741)
  %1752 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1740)
  %1753 = ashr i64 %1752, 32
  %1754 = mul nsw i64 %1753, %const_reg_qword5, !spirv.Decorations !610
  %1755 = shl i64 %1752, 32
  %1756 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1754
  %1757 = ashr exact i64 %1755, 31
  %1758 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1756 to i8 addrspace(4)*
  %1759 = getelementptr i8, i8 addrspace(4)* %1758, i64 %1757
  %1760 = bitcast i8 addrspace(4)* %1759 to i16 addrspace(4)*
  %1761 = addrspacecast i16 addrspace(4)* %1760 to i16 addrspace(1)*
  %1762 = load i16, i16 addrspace(1)* %1761, align 2
  %1763 = zext i16 %1739 to i32
  %1764 = shl nuw i32 %1763, 16, !spirv.Decorations !628
  %1765 = bitcast i32 %1764 to float
  %1766 = zext i16 %1762 to i32
  %1767 = shl nuw i32 %1766, 16, !spirv.Decorations !628
  %1768 = bitcast i32 %1767 to float
  %1769 = fmul reassoc nsz arcp contract float %1765, %1768, !spirv.Decorations !612
  %1770 = fadd reassoc nsz arcp contract float %1769, %.sroa.158.1, !spirv.Decorations !612
  br label %._crit_edge.2.7

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7, %1751
  %.sroa.158.2 = phi float [ %1770, %1751 ], [ %.sroa.158.1, %._crit_edge.1.7 ]
  %1771 = and i1 %244, %1608
  br i1 %1771, label %1772, label %.preheader.7

1772:                                             ; preds = %._crit_edge.2.7
  %1773 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1773)
  %1774 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1774)
  %1775 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %1775, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1776

1776:                                             ; preds = %1776, %1772
  %1777 = phi i32 [ 0, %1772 ], [ %1782, %1776 ]
  %1778 = zext i32 %1777 to i64
  %1779 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1778
  %1780 = load i32, i32* %1779, align 4, !noalias !617
  %1781 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1778
  store i32 %1780, i32* %1781, align 4, !alias.scope !617
  %1782 = add nuw nsw i32 %1777, 1, !spirv.Decorations !620
  %1783 = icmp eq i32 %1777, 0
  br i1 %1783, label %1776, label %1784, !llvm.loop !622

1784:                                             ; preds = %1776
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1774)
  %1785 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1773)
  %1786 = shl i64 %1785, 32
  %1787 = ashr exact i64 %1786, 32
  %1788 = mul nsw i64 %1787, %const_reg_qword3, !spirv.Decorations !610
  %1789 = ashr i64 %1785, 32
  %1790 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1788, i32 0
  %1791 = getelementptr i16, i16 addrspace(4)* %1790, i64 %1789
  %1792 = addrspacecast i16 addrspace(4)* %1791 to i16 addrspace(1)*
  %1793 = load i16, i16 addrspace(1)* %1792, align 2
  %1794 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1794)
  %1795 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1795)
  %1796 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1796, align 4, !noalias !624
  store i32 %1607, i32* %57, align 4, !noalias !624
  br label %1797

1797:                                             ; preds = %1797, %1784
  %1798 = phi i32 [ 0, %1784 ], [ %1803, %1797 ]
  %1799 = zext i32 %1798 to i64
  %1800 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1799
  %1801 = load i32, i32* %1800, align 4, !noalias !624
  %1802 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1799
  store i32 %1801, i32* %1802, align 4, !alias.scope !624
  %1803 = add nuw nsw i32 %1798, 1, !spirv.Decorations !620
  %1804 = icmp eq i32 %1798, 0
  br i1 %1804, label %1797, label %1805, !llvm.loop !627

1805:                                             ; preds = %1797
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1795)
  %1806 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1794)
  %1807 = ashr i64 %1806, 32
  %1808 = mul nsw i64 %1807, %const_reg_qword5, !spirv.Decorations !610
  %1809 = shl i64 %1806, 32
  %1810 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1808
  %1811 = ashr exact i64 %1809, 31
  %1812 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1810 to i8 addrspace(4)*
  %1813 = getelementptr i8, i8 addrspace(4)* %1812, i64 %1811
  %1814 = bitcast i8 addrspace(4)* %1813 to i16 addrspace(4)*
  %1815 = addrspacecast i16 addrspace(4)* %1814 to i16 addrspace(1)*
  %1816 = load i16, i16 addrspace(1)* %1815, align 2
  %1817 = zext i16 %1793 to i32
  %1818 = shl nuw i32 %1817, 16, !spirv.Decorations !628
  %1819 = bitcast i32 %1818 to float
  %1820 = zext i16 %1816 to i32
  %1821 = shl nuw i32 %1820, 16, !spirv.Decorations !628
  %1822 = bitcast i32 %1821 to float
  %1823 = fmul reassoc nsz arcp contract float %1819, %1822, !spirv.Decorations !612
  %1824 = fadd reassoc nsz arcp contract float %1823, %.sroa.222.1, !spirv.Decorations !612
  br label %.preheader.7

.preheader.7:                                     ; preds = %._crit_edge.2.7, %1805
  %.sroa.222.2 = phi float [ %1824, %1805 ], [ %.sroa.222.1, %._crit_edge.2.7 ]
  %1825 = or i32 %41, 8
  %1826 = icmp slt i32 %1825, %const_reg_dword1
  %1827 = and i1 %76, %1826
  br i1 %1827, label %1828, label %._crit_edge.8

1828:                                             ; preds = %.preheader.7
  %1829 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1829)
  %1830 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1830)
  %1831 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1831, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1832

1832:                                             ; preds = %1832, %1828
  %1833 = phi i32 [ 0, %1828 ], [ %1838, %1832 ]
  %1834 = zext i32 %1833 to i64
  %1835 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1834
  %1836 = load i32, i32* %1835, align 4, !noalias !617
  %1837 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1834
  store i32 %1836, i32* %1837, align 4, !alias.scope !617
  %1838 = add nuw nsw i32 %1833, 1, !spirv.Decorations !620
  %1839 = icmp eq i32 %1833, 0
  br i1 %1839, label %1832, label %1840, !llvm.loop !622

1840:                                             ; preds = %1832
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1830)
  %1841 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1829)
  %1842 = shl i64 %1841, 32
  %1843 = ashr exact i64 %1842, 32
  %1844 = mul nsw i64 %1843, %const_reg_qword3, !spirv.Decorations !610
  %1845 = ashr i64 %1841, 32
  %1846 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1844, i32 0
  %1847 = getelementptr i16, i16 addrspace(4)* %1846, i64 %1845
  %1848 = addrspacecast i16 addrspace(4)* %1847 to i16 addrspace(1)*
  %1849 = load i16, i16 addrspace(1)* %1848, align 2
  %1850 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1850)
  %1851 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1851)
  %1852 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1852, align 4, !noalias !624
  store i32 %1825, i32* %57, align 4, !noalias !624
  br label %1853

1853:                                             ; preds = %1853, %1840
  %1854 = phi i32 [ 0, %1840 ], [ %1859, %1853 ]
  %1855 = zext i32 %1854 to i64
  %1856 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1855
  %1857 = load i32, i32* %1856, align 4, !noalias !624
  %1858 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1855
  store i32 %1857, i32* %1858, align 4, !alias.scope !624
  %1859 = add nuw nsw i32 %1854, 1, !spirv.Decorations !620
  %1860 = icmp eq i32 %1854, 0
  br i1 %1860, label %1853, label %1861, !llvm.loop !627

1861:                                             ; preds = %1853
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1851)
  %1862 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1850)
  %1863 = ashr i64 %1862, 32
  %1864 = mul nsw i64 %1863, %const_reg_qword5, !spirv.Decorations !610
  %1865 = shl i64 %1862, 32
  %1866 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1864
  %1867 = ashr exact i64 %1865, 31
  %1868 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1866 to i8 addrspace(4)*
  %1869 = getelementptr i8, i8 addrspace(4)* %1868, i64 %1867
  %1870 = bitcast i8 addrspace(4)* %1869 to i16 addrspace(4)*
  %1871 = addrspacecast i16 addrspace(4)* %1870 to i16 addrspace(1)*
  %1872 = load i16, i16 addrspace(1)* %1871, align 2
  %1873 = zext i16 %1849 to i32
  %1874 = shl nuw i32 %1873, 16, !spirv.Decorations !628
  %1875 = bitcast i32 %1874 to float
  %1876 = zext i16 %1872 to i32
  %1877 = shl nuw i32 %1876, 16, !spirv.Decorations !628
  %1878 = bitcast i32 %1877 to float
  %1879 = fmul reassoc nsz arcp contract float %1875, %1878, !spirv.Decorations !612
  %1880 = fadd reassoc nsz arcp contract float %1879, %.sroa.34.1, !spirv.Decorations !612
  br label %._crit_edge.8

._crit_edge.8:                                    ; preds = %.preheader.7, %1861
  %.sroa.34.2 = phi float [ %1880, %1861 ], [ %.sroa.34.1, %.preheader.7 ]
  %1881 = and i1 %132, %1826
  br i1 %1881, label %1882, label %._crit_edge.1.8

1882:                                             ; preds = %._crit_edge.8
  %1883 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1883)
  %1884 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1884)
  %1885 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %1885, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1886

1886:                                             ; preds = %1886, %1882
  %1887 = phi i32 [ 0, %1882 ], [ %1892, %1886 ]
  %1888 = zext i32 %1887 to i64
  %1889 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1888
  %1890 = load i32, i32* %1889, align 4, !noalias !617
  %1891 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1888
  store i32 %1890, i32* %1891, align 4, !alias.scope !617
  %1892 = add nuw nsw i32 %1887, 1, !spirv.Decorations !620
  %1893 = icmp eq i32 %1887, 0
  br i1 %1893, label %1886, label %1894, !llvm.loop !622

1894:                                             ; preds = %1886
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1884)
  %1895 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1883)
  %1896 = shl i64 %1895, 32
  %1897 = ashr exact i64 %1896, 32
  %1898 = mul nsw i64 %1897, %const_reg_qword3, !spirv.Decorations !610
  %1899 = ashr i64 %1895, 32
  %1900 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1898, i32 0
  %1901 = getelementptr i16, i16 addrspace(4)* %1900, i64 %1899
  %1902 = addrspacecast i16 addrspace(4)* %1901 to i16 addrspace(1)*
  %1903 = load i16, i16 addrspace(1)* %1902, align 2
  %1904 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1904)
  %1905 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1905)
  %1906 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1906, align 4, !noalias !624
  store i32 %1825, i32* %57, align 4, !noalias !624
  br label %1907

1907:                                             ; preds = %1907, %1894
  %1908 = phi i32 [ 0, %1894 ], [ %1913, %1907 ]
  %1909 = zext i32 %1908 to i64
  %1910 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1909
  %1911 = load i32, i32* %1910, align 4, !noalias !624
  %1912 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1909
  store i32 %1911, i32* %1912, align 4, !alias.scope !624
  %1913 = add nuw nsw i32 %1908, 1, !spirv.Decorations !620
  %1914 = icmp eq i32 %1908, 0
  br i1 %1914, label %1907, label %1915, !llvm.loop !627

1915:                                             ; preds = %1907
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1905)
  %1916 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1904)
  %1917 = ashr i64 %1916, 32
  %1918 = mul nsw i64 %1917, %const_reg_qword5, !spirv.Decorations !610
  %1919 = shl i64 %1916, 32
  %1920 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1918
  %1921 = ashr exact i64 %1919, 31
  %1922 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1920 to i8 addrspace(4)*
  %1923 = getelementptr i8, i8 addrspace(4)* %1922, i64 %1921
  %1924 = bitcast i8 addrspace(4)* %1923 to i16 addrspace(4)*
  %1925 = addrspacecast i16 addrspace(4)* %1924 to i16 addrspace(1)*
  %1926 = load i16, i16 addrspace(1)* %1925, align 2
  %1927 = zext i16 %1903 to i32
  %1928 = shl nuw i32 %1927, 16, !spirv.Decorations !628
  %1929 = bitcast i32 %1928 to float
  %1930 = zext i16 %1926 to i32
  %1931 = shl nuw i32 %1930, 16, !spirv.Decorations !628
  %1932 = bitcast i32 %1931 to float
  %1933 = fmul reassoc nsz arcp contract float %1929, %1932, !spirv.Decorations !612
  %1934 = fadd reassoc nsz arcp contract float %1933, %.sroa.98.1, !spirv.Decorations !612
  br label %._crit_edge.1.8

._crit_edge.1.8:                                  ; preds = %._crit_edge.8, %1915
  %.sroa.98.2 = phi float [ %1934, %1915 ], [ %.sroa.98.1, %._crit_edge.8 ]
  %1935 = and i1 %188, %1826
  br i1 %1935, label %1936, label %._crit_edge.2.8

1936:                                             ; preds = %._crit_edge.1.8
  %1937 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1937)
  %1938 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1938)
  %1939 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %1939, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1940

1940:                                             ; preds = %1940, %1936
  %1941 = phi i32 [ 0, %1936 ], [ %1946, %1940 ]
  %1942 = zext i32 %1941 to i64
  %1943 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1942
  %1944 = load i32, i32* %1943, align 4, !noalias !617
  %1945 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1942
  store i32 %1944, i32* %1945, align 4, !alias.scope !617
  %1946 = add nuw nsw i32 %1941, 1, !spirv.Decorations !620
  %1947 = icmp eq i32 %1941, 0
  br i1 %1947, label %1940, label %1948, !llvm.loop !622

1948:                                             ; preds = %1940
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1938)
  %1949 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1937)
  %1950 = shl i64 %1949, 32
  %1951 = ashr exact i64 %1950, 32
  %1952 = mul nsw i64 %1951, %const_reg_qword3, !spirv.Decorations !610
  %1953 = ashr i64 %1949, 32
  %1954 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1952, i32 0
  %1955 = getelementptr i16, i16 addrspace(4)* %1954, i64 %1953
  %1956 = addrspacecast i16 addrspace(4)* %1955 to i16 addrspace(1)*
  %1957 = load i16, i16 addrspace(1)* %1956, align 2
  %1958 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1958)
  %1959 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1959)
  %1960 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1960, align 4, !noalias !624
  store i32 %1825, i32* %57, align 4, !noalias !624
  br label %1961

1961:                                             ; preds = %1961, %1948
  %1962 = phi i32 [ 0, %1948 ], [ %1967, %1961 ]
  %1963 = zext i32 %1962 to i64
  %1964 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1963
  %1965 = load i32, i32* %1964, align 4, !noalias !624
  %1966 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1963
  store i32 %1965, i32* %1966, align 4, !alias.scope !624
  %1967 = add nuw nsw i32 %1962, 1, !spirv.Decorations !620
  %1968 = icmp eq i32 %1962, 0
  br i1 %1968, label %1961, label %1969, !llvm.loop !627

1969:                                             ; preds = %1961
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1959)
  %1970 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1958)
  %1971 = ashr i64 %1970, 32
  %1972 = mul nsw i64 %1971, %const_reg_qword5, !spirv.Decorations !610
  %1973 = shl i64 %1970, 32
  %1974 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1972
  %1975 = ashr exact i64 %1973, 31
  %1976 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %1974 to i8 addrspace(4)*
  %1977 = getelementptr i8, i8 addrspace(4)* %1976, i64 %1975
  %1978 = bitcast i8 addrspace(4)* %1977 to i16 addrspace(4)*
  %1979 = addrspacecast i16 addrspace(4)* %1978 to i16 addrspace(1)*
  %1980 = load i16, i16 addrspace(1)* %1979, align 2
  %1981 = zext i16 %1957 to i32
  %1982 = shl nuw i32 %1981, 16, !spirv.Decorations !628
  %1983 = bitcast i32 %1982 to float
  %1984 = zext i16 %1980 to i32
  %1985 = shl nuw i32 %1984, 16, !spirv.Decorations !628
  %1986 = bitcast i32 %1985 to float
  %1987 = fmul reassoc nsz arcp contract float %1983, %1986, !spirv.Decorations !612
  %1988 = fadd reassoc nsz arcp contract float %1987, %.sroa.162.1, !spirv.Decorations !612
  br label %._crit_edge.2.8

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8, %1969
  %.sroa.162.2 = phi float [ %1988, %1969 ], [ %.sroa.162.1, %._crit_edge.1.8 ]
  %1989 = and i1 %244, %1826
  br i1 %1989, label %1990, label %.preheader.8

1990:                                             ; preds = %._crit_edge.2.8
  %1991 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1991)
  %1992 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1992)
  %1993 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %1993, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %1994

1994:                                             ; preds = %1994, %1990
  %1995 = phi i32 [ 0, %1990 ], [ %2000, %1994 ]
  %1996 = zext i32 %1995 to i64
  %1997 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1996
  %1998 = load i32, i32* %1997, align 4, !noalias !617
  %1999 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1996
  store i32 %1998, i32* %1999, align 4, !alias.scope !617
  %2000 = add nuw nsw i32 %1995, 1, !spirv.Decorations !620
  %2001 = icmp eq i32 %1995, 0
  br i1 %2001, label %1994, label %2002, !llvm.loop !622

2002:                                             ; preds = %1994
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1992)
  %2003 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1991)
  %2004 = shl i64 %2003, 32
  %2005 = ashr exact i64 %2004, 32
  %2006 = mul nsw i64 %2005, %const_reg_qword3, !spirv.Decorations !610
  %2007 = ashr i64 %2003, 32
  %2008 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2006, i32 0
  %2009 = getelementptr i16, i16 addrspace(4)* %2008, i64 %2007
  %2010 = addrspacecast i16 addrspace(4)* %2009 to i16 addrspace(1)*
  %2011 = load i16, i16 addrspace(1)* %2010, align 2
  %2012 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2012)
  %2013 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2013)
  %2014 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2014, align 4, !noalias !624
  store i32 %1825, i32* %57, align 4, !noalias !624
  br label %2015

2015:                                             ; preds = %2015, %2002
  %2016 = phi i32 [ 0, %2002 ], [ %2021, %2015 ]
  %2017 = zext i32 %2016 to i64
  %2018 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2017
  %2019 = load i32, i32* %2018, align 4, !noalias !624
  %2020 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2017
  store i32 %2019, i32* %2020, align 4, !alias.scope !624
  %2021 = add nuw nsw i32 %2016, 1, !spirv.Decorations !620
  %2022 = icmp eq i32 %2016, 0
  br i1 %2022, label %2015, label %2023, !llvm.loop !627

2023:                                             ; preds = %2015
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2013)
  %2024 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2012)
  %2025 = ashr i64 %2024, 32
  %2026 = mul nsw i64 %2025, %const_reg_qword5, !spirv.Decorations !610
  %2027 = shl i64 %2024, 32
  %2028 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2026
  %2029 = ashr exact i64 %2027, 31
  %2030 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2028 to i8 addrspace(4)*
  %2031 = getelementptr i8, i8 addrspace(4)* %2030, i64 %2029
  %2032 = bitcast i8 addrspace(4)* %2031 to i16 addrspace(4)*
  %2033 = addrspacecast i16 addrspace(4)* %2032 to i16 addrspace(1)*
  %2034 = load i16, i16 addrspace(1)* %2033, align 2
  %2035 = zext i16 %2011 to i32
  %2036 = shl nuw i32 %2035, 16, !spirv.Decorations !628
  %2037 = bitcast i32 %2036 to float
  %2038 = zext i16 %2034 to i32
  %2039 = shl nuw i32 %2038, 16, !spirv.Decorations !628
  %2040 = bitcast i32 %2039 to float
  %2041 = fmul reassoc nsz arcp contract float %2037, %2040, !spirv.Decorations !612
  %2042 = fadd reassoc nsz arcp contract float %2041, %.sroa.226.1, !spirv.Decorations !612
  br label %.preheader.8

.preheader.8:                                     ; preds = %._crit_edge.2.8, %2023
  %.sroa.226.2 = phi float [ %2042, %2023 ], [ %.sroa.226.1, %._crit_edge.2.8 ]
  %2043 = or i32 %41, 9
  %2044 = icmp slt i32 %2043, %const_reg_dword1
  %2045 = and i1 %76, %2044
  br i1 %2045, label %2046, label %._crit_edge.9

2046:                                             ; preds = %.preheader.8
  %2047 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2047)
  %2048 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2048)
  %2049 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2049, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2050

2050:                                             ; preds = %2050, %2046
  %2051 = phi i32 [ 0, %2046 ], [ %2056, %2050 ]
  %2052 = zext i32 %2051 to i64
  %2053 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2052
  %2054 = load i32, i32* %2053, align 4, !noalias !617
  %2055 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2052
  store i32 %2054, i32* %2055, align 4, !alias.scope !617
  %2056 = add nuw nsw i32 %2051, 1, !spirv.Decorations !620
  %2057 = icmp eq i32 %2051, 0
  br i1 %2057, label %2050, label %2058, !llvm.loop !622

2058:                                             ; preds = %2050
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2048)
  %2059 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2047)
  %2060 = shl i64 %2059, 32
  %2061 = ashr exact i64 %2060, 32
  %2062 = mul nsw i64 %2061, %const_reg_qword3, !spirv.Decorations !610
  %2063 = ashr i64 %2059, 32
  %2064 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2062, i32 0
  %2065 = getelementptr i16, i16 addrspace(4)* %2064, i64 %2063
  %2066 = addrspacecast i16 addrspace(4)* %2065 to i16 addrspace(1)*
  %2067 = load i16, i16 addrspace(1)* %2066, align 2
  %2068 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2068)
  %2069 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2069)
  %2070 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2070, align 4, !noalias !624
  store i32 %2043, i32* %57, align 4, !noalias !624
  br label %2071

2071:                                             ; preds = %2071, %2058
  %2072 = phi i32 [ 0, %2058 ], [ %2077, %2071 ]
  %2073 = zext i32 %2072 to i64
  %2074 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2073
  %2075 = load i32, i32* %2074, align 4, !noalias !624
  %2076 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2073
  store i32 %2075, i32* %2076, align 4, !alias.scope !624
  %2077 = add nuw nsw i32 %2072, 1, !spirv.Decorations !620
  %2078 = icmp eq i32 %2072, 0
  br i1 %2078, label %2071, label %2079, !llvm.loop !627

2079:                                             ; preds = %2071
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2069)
  %2080 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2068)
  %2081 = ashr i64 %2080, 32
  %2082 = mul nsw i64 %2081, %const_reg_qword5, !spirv.Decorations !610
  %2083 = shl i64 %2080, 32
  %2084 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2082
  %2085 = ashr exact i64 %2083, 31
  %2086 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2084 to i8 addrspace(4)*
  %2087 = getelementptr i8, i8 addrspace(4)* %2086, i64 %2085
  %2088 = bitcast i8 addrspace(4)* %2087 to i16 addrspace(4)*
  %2089 = addrspacecast i16 addrspace(4)* %2088 to i16 addrspace(1)*
  %2090 = load i16, i16 addrspace(1)* %2089, align 2
  %2091 = zext i16 %2067 to i32
  %2092 = shl nuw i32 %2091, 16, !spirv.Decorations !628
  %2093 = bitcast i32 %2092 to float
  %2094 = zext i16 %2090 to i32
  %2095 = shl nuw i32 %2094, 16, !spirv.Decorations !628
  %2096 = bitcast i32 %2095 to float
  %2097 = fmul reassoc nsz arcp contract float %2093, %2096, !spirv.Decorations !612
  %2098 = fadd reassoc nsz arcp contract float %2097, %.sroa.38.1, !spirv.Decorations !612
  br label %._crit_edge.9

._crit_edge.9:                                    ; preds = %.preheader.8, %2079
  %.sroa.38.2 = phi float [ %2098, %2079 ], [ %.sroa.38.1, %.preheader.8 ]
  %2099 = and i1 %132, %2044
  br i1 %2099, label %2100, label %._crit_edge.1.9

2100:                                             ; preds = %._crit_edge.9
  %2101 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2101)
  %2102 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2102)
  %2103 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %2103, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2104

2104:                                             ; preds = %2104, %2100
  %2105 = phi i32 [ 0, %2100 ], [ %2110, %2104 ]
  %2106 = zext i32 %2105 to i64
  %2107 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2106
  %2108 = load i32, i32* %2107, align 4, !noalias !617
  %2109 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2106
  store i32 %2108, i32* %2109, align 4, !alias.scope !617
  %2110 = add nuw nsw i32 %2105, 1, !spirv.Decorations !620
  %2111 = icmp eq i32 %2105, 0
  br i1 %2111, label %2104, label %2112, !llvm.loop !622

2112:                                             ; preds = %2104
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2102)
  %2113 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2101)
  %2114 = shl i64 %2113, 32
  %2115 = ashr exact i64 %2114, 32
  %2116 = mul nsw i64 %2115, %const_reg_qword3, !spirv.Decorations !610
  %2117 = ashr i64 %2113, 32
  %2118 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2116, i32 0
  %2119 = getelementptr i16, i16 addrspace(4)* %2118, i64 %2117
  %2120 = addrspacecast i16 addrspace(4)* %2119 to i16 addrspace(1)*
  %2121 = load i16, i16 addrspace(1)* %2120, align 2
  %2122 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2122)
  %2123 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2123)
  %2124 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2124, align 4, !noalias !624
  store i32 %2043, i32* %57, align 4, !noalias !624
  br label %2125

2125:                                             ; preds = %2125, %2112
  %2126 = phi i32 [ 0, %2112 ], [ %2131, %2125 ]
  %2127 = zext i32 %2126 to i64
  %2128 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2127
  %2129 = load i32, i32* %2128, align 4, !noalias !624
  %2130 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2127
  store i32 %2129, i32* %2130, align 4, !alias.scope !624
  %2131 = add nuw nsw i32 %2126, 1, !spirv.Decorations !620
  %2132 = icmp eq i32 %2126, 0
  br i1 %2132, label %2125, label %2133, !llvm.loop !627

2133:                                             ; preds = %2125
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2123)
  %2134 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2122)
  %2135 = ashr i64 %2134, 32
  %2136 = mul nsw i64 %2135, %const_reg_qword5, !spirv.Decorations !610
  %2137 = shl i64 %2134, 32
  %2138 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2136
  %2139 = ashr exact i64 %2137, 31
  %2140 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2138 to i8 addrspace(4)*
  %2141 = getelementptr i8, i8 addrspace(4)* %2140, i64 %2139
  %2142 = bitcast i8 addrspace(4)* %2141 to i16 addrspace(4)*
  %2143 = addrspacecast i16 addrspace(4)* %2142 to i16 addrspace(1)*
  %2144 = load i16, i16 addrspace(1)* %2143, align 2
  %2145 = zext i16 %2121 to i32
  %2146 = shl nuw i32 %2145, 16, !spirv.Decorations !628
  %2147 = bitcast i32 %2146 to float
  %2148 = zext i16 %2144 to i32
  %2149 = shl nuw i32 %2148, 16, !spirv.Decorations !628
  %2150 = bitcast i32 %2149 to float
  %2151 = fmul reassoc nsz arcp contract float %2147, %2150, !spirv.Decorations !612
  %2152 = fadd reassoc nsz arcp contract float %2151, %.sroa.102.1, !spirv.Decorations !612
  br label %._crit_edge.1.9

._crit_edge.1.9:                                  ; preds = %._crit_edge.9, %2133
  %.sroa.102.2 = phi float [ %2152, %2133 ], [ %.sroa.102.1, %._crit_edge.9 ]
  %2153 = and i1 %188, %2044
  br i1 %2153, label %2154, label %._crit_edge.2.9

2154:                                             ; preds = %._crit_edge.1.9
  %2155 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2155)
  %2156 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2156)
  %2157 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %2157, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2158

2158:                                             ; preds = %2158, %2154
  %2159 = phi i32 [ 0, %2154 ], [ %2164, %2158 ]
  %2160 = zext i32 %2159 to i64
  %2161 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2160
  %2162 = load i32, i32* %2161, align 4, !noalias !617
  %2163 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2160
  store i32 %2162, i32* %2163, align 4, !alias.scope !617
  %2164 = add nuw nsw i32 %2159, 1, !spirv.Decorations !620
  %2165 = icmp eq i32 %2159, 0
  br i1 %2165, label %2158, label %2166, !llvm.loop !622

2166:                                             ; preds = %2158
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2156)
  %2167 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2155)
  %2168 = shl i64 %2167, 32
  %2169 = ashr exact i64 %2168, 32
  %2170 = mul nsw i64 %2169, %const_reg_qword3, !spirv.Decorations !610
  %2171 = ashr i64 %2167, 32
  %2172 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2170, i32 0
  %2173 = getelementptr i16, i16 addrspace(4)* %2172, i64 %2171
  %2174 = addrspacecast i16 addrspace(4)* %2173 to i16 addrspace(1)*
  %2175 = load i16, i16 addrspace(1)* %2174, align 2
  %2176 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2176)
  %2177 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2177)
  %2178 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2178, align 4, !noalias !624
  store i32 %2043, i32* %57, align 4, !noalias !624
  br label %2179

2179:                                             ; preds = %2179, %2166
  %2180 = phi i32 [ 0, %2166 ], [ %2185, %2179 ]
  %2181 = zext i32 %2180 to i64
  %2182 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2181
  %2183 = load i32, i32* %2182, align 4, !noalias !624
  %2184 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2181
  store i32 %2183, i32* %2184, align 4, !alias.scope !624
  %2185 = add nuw nsw i32 %2180, 1, !spirv.Decorations !620
  %2186 = icmp eq i32 %2180, 0
  br i1 %2186, label %2179, label %2187, !llvm.loop !627

2187:                                             ; preds = %2179
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2177)
  %2188 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2176)
  %2189 = ashr i64 %2188, 32
  %2190 = mul nsw i64 %2189, %const_reg_qword5, !spirv.Decorations !610
  %2191 = shl i64 %2188, 32
  %2192 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2190
  %2193 = ashr exact i64 %2191, 31
  %2194 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2192 to i8 addrspace(4)*
  %2195 = getelementptr i8, i8 addrspace(4)* %2194, i64 %2193
  %2196 = bitcast i8 addrspace(4)* %2195 to i16 addrspace(4)*
  %2197 = addrspacecast i16 addrspace(4)* %2196 to i16 addrspace(1)*
  %2198 = load i16, i16 addrspace(1)* %2197, align 2
  %2199 = zext i16 %2175 to i32
  %2200 = shl nuw i32 %2199, 16, !spirv.Decorations !628
  %2201 = bitcast i32 %2200 to float
  %2202 = zext i16 %2198 to i32
  %2203 = shl nuw i32 %2202, 16, !spirv.Decorations !628
  %2204 = bitcast i32 %2203 to float
  %2205 = fmul reassoc nsz arcp contract float %2201, %2204, !spirv.Decorations !612
  %2206 = fadd reassoc nsz arcp contract float %2205, %.sroa.166.1, !spirv.Decorations !612
  br label %._crit_edge.2.9

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9, %2187
  %.sroa.166.2 = phi float [ %2206, %2187 ], [ %.sroa.166.1, %._crit_edge.1.9 ]
  %2207 = and i1 %244, %2044
  br i1 %2207, label %2208, label %.preheader.9

2208:                                             ; preds = %._crit_edge.2.9
  %2209 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2209)
  %2210 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2210)
  %2211 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %2211, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2212

2212:                                             ; preds = %2212, %2208
  %2213 = phi i32 [ 0, %2208 ], [ %2218, %2212 ]
  %2214 = zext i32 %2213 to i64
  %2215 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2214
  %2216 = load i32, i32* %2215, align 4, !noalias !617
  %2217 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2214
  store i32 %2216, i32* %2217, align 4, !alias.scope !617
  %2218 = add nuw nsw i32 %2213, 1, !spirv.Decorations !620
  %2219 = icmp eq i32 %2213, 0
  br i1 %2219, label %2212, label %2220, !llvm.loop !622

2220:                                             ; preds = %2212
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2210)
  %2221 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2209)
  %2222 = shl i64 %2221, 32
  %2223 = ashr exact i64 %2222, 32
  %2224 = mul nsw i64 %2223, %const_reg_qword3, !spirv.Decorations !610
  %2225 = ashr i64 %2221, 32
  %2226 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2224, i32 0
  %2227 = getelementptr i16, i16 addrspace(4)* %2226, i64 %2225
  %2228 = addrspacecast i16 addrspace(4)* %2227 to i16 addrspace(1)*
  %2229 = load i16, i16 addrspace(1)* %2228, align 2
  %2230 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2230)
  %2231 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2231)
  %2232 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2232, align 4, !noalias !624
  store i32 %2043, i32* %57, align 4, !noalias !624
  br label %2233

2233:                                             ; preds = %2233, %2220
  %2234 = phi i32 [ 0, %2220 ], [ %2239, %2233 ]
  %2235 = zext i32 %2234 to i64
  %2236 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2235
  %2237 = load i32, i32* %2236, align 4, !noalias !624
  %2238 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2235
  store i32 %2237, i32* %2238, align 4, !alias.scope !624
  %2239 = add nuw nsw i32 %2234, 1, !spirv.Decorations !620
  %2240 = icmp eq i32 %2234, 0
  br i1 %2240, label %2233, label %2241, !llvm.loop !627

2241:                                             ; preds = %2233
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2231)
  %2242 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2230)
  %2243 = ashr i64 %2242, 32
  %2244 = mul nsw i64 %2243, %const_reg_qword5, !spirv.Decorations !610
  %2245 = shl i64 %2242, 32
  %2246 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2244
  %2247 = ashr exact i64 %2245, 31
  %2248 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2246 to i8 addrspace(4)*
  %2249 = getelementptr i8, i8 addrspace(4)* %2248, i64 %2247
  %2250 = bitcast i8 addrspace(4)* %2249 to i16 addrspace(4)*
  %2251 = addrspacecast i16 addrspace(4)* %2250 to i16 addrspace(1)*
  %2252 = load i16, i16 addrspace(1)* %2251, align 2
  %2253 = zext i16 %2229 to i32
  %2254 = shl nuw i32 %2253, 16, !spirv.Decorations !628
  %2255 = bitcast i32 %2254 to float
  %2256 = zext i16 %2252 to i32
  %2257 = shl nuw i32 %2256, 16, !spirv.Decorations !628
  %2258 = bitcast i32 %2257 to float
  %2259 = fmul reassoc nsz arcp contract float %2255, %2258, !spirv.Decorations !612
  %2260 = fadd reassoc nsz arcp contract float %2259, %.sroa.230.1, !spirv.Decorations !612
  br label %.preheader.9

.preheader.9:                                     ; preds = %._crit_edge.2.9, %2241
  %.sroa.230.2 = phi float [ %2260, %2241 ], [ %.sroa.230.1, %._crit_edge.2.9 ]
  %2261 = or i32 %41, 10
  %2262 = icmp slt i32 %2261, %const_reg_dword1
  %2263 = and i1 %76, %2262
  br i1 %2263, label %2264, label %._crit_edge.10

2264:                                             ; preds = %.preheader.9
  %2265 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2265)
  %2266 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2266)
  %2267 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2267, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2268

2268:                                             ; preds = %2268, %2264
  %2269 = phi i32 [ 0, %2264 ], [ %2274, %2268 ]
  %2270 = zext i32 %2269 to i64
  %2271 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2270
  %2272 = load i32, i32* %2271, align 4, !noalias !617
  %2273 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2270
  store i32 %2272, i32* %2273, align 4, !alias.scope !617
  %2274 = add nuw nsw i32 %2269, 1, !spirv.Decorations !620
  %2275 = icmp eq i32 %2269, 0
  br i1 %2275, label %2268, label %2276, !llvm.loop !622

2276:                                             ; preds = %2268
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2266)
  %2277 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2265)
  %2278 = shl i64 %2277, 32
  %2279 = ashr exact i64 %2278, 32
  %2280 = mul nsw i64 %2279, %const_reg_qword3, !spirv.Decorations !610
  %2281 = ashr i64 %2277, 32
  %2282 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2280, i32 0
  %2283 = getelementptr i16, i16 addrspace(4)* %2282, i64 %2281
  %2284 = addrspacecast i16 addrspace(4)* %2283 to i16 addrspace(1)*
  %2285 = load i16, i16 addrspace(1)* %2284, align 2
  %2286 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2286)
  %2287 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2287)
  %2288 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2288, align 4, !noalias !624
  store i32 %2261, i32* %57, align 4, !noalias !624
  br label %2289

2289:                                             ; preds = %2289, %2276
  %2290 = phi i32 [ 0, %2276 ], [ %2295, %2289 ]
  %2291 = zext i32 %2290 to i64
  %2292 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2291
  %2293 = load i32, i32* %2292, align 4, !noalias !624
  %2294 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2291
  store i32 %2293, i32* %2294, align 4, !alias.scope !624
  %2295 = add nuw nsw i32 %2290, 1, !spirv.Decorations !620
  %2296 = icmp eq i32 %2290, 0
  br i1 %2296, label %2289, label %2297, !llvm.loop !627

2297:                                             ; preds = %2289
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2287)
  %2298 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2286)
  %2299 = ashr i64 %2298, 32
  %2300 = mul nsw i64 %2299, %const_reg_qword5, !spirv.Decorations !610
  %2301 = shl i64 %2298, 32
  %2302 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2300
  %2303 = ashr exact i64 %2301, 31
  %2304 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2302 to i8 addrspace(4)*
  %2305 = getelementptr i8, i8 addrspace(4)* %2304, i64 %2303
  %2306 = bitcast i8 addrspace(4)* %2305 to i16 addrspace(4)*
  %2307 = addrspacecast i16 addrspace(4)* %2306 to i16 addrspace(1)*
  %2308 = load i16, i16 addrspace(1)* %2307, align 2
  %2309 = zext i16 %2285 to i32
  %2310 = shl nuw i32 %2309, 16, !spirv.Decorations !628
  %2311 = bitcast i32 %2310 to float
  %2312 = zext i16 %2308 to i32
  %2313 = shl nuw i32 %2312, 16, !spirv.Decorations !628
  %2314 = bitcast i32 %2313 to float
  %2315 = fmul reassoc nsz arcp contract float %2311, %2314, !spirv.Decorations !612
  %2316 = fadd reassoc nsz arcp contract float %2315, %.sroa.42.1, !spirv.Decorations !612
  br label %._crit_edge.10

._crit_edge.10:                                   ; preds = %.preheader.9, %2297
  %.sroa.42.2 = phi float [ %2316, %2297 ], [ %.sroa.42.1, %.preheader.9 ]
  %2317 = and i1 %132, %2262
  br i1 %2317, label %2318, label %._crit_edge.1.10

2318:                                             ; preds = %._crit_edge.10
  %2319 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2319)
  %2320 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2320)
  %2321 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %2321, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2322

2322:                                             ; preds = %2322, %2318
  %2323 = phi i32 [ 0, %2318 ], [ %2328, %2322 ]
  %2324 = zext i32 %2323 to i64
  %2325 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2324
  %2326 = load i32, i32* %2325, align 4, !noalias !617
  %2327 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2324
  store i32 %2326, i32* %2327, align 4, !alias.scope !617
  %2328 = add nuw nsw i32 %2323, 1, !spirv.Decorations !620
  %2329 = icmp eq i32 %2323, 0
  br i1 %2329, label %2322, label %2330, !llvm.loop !622

2330:                                             ; preds = %2322
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2320)
  %2331 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2319)
  %2332 = shl i64 %2331, 32
  %2333 = ashr exact i64 %2332, 32
  %2334 = mul nsw i64 %2333, %const_reg_qword3, !spirv.Decorations !610
  %2335 = ashr i64 %2331, 32
  %2336 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2334, i32 0
  %2337 = getelementptr i16, i16 addrspace(4)* %2336, i64 %2335
  %2338 = addrspacecast i16 addrspace(4)* %2337 to i16 addrspace(1)*
  %2339 = load i16, i16 addrspace(1)* %2338, align 2
  %2340 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2340)
  %2341 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2341)
  %2342 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2342, align 4, !noalias !624
  store i32 %2261, i32* %57, align 4, !noalias !624
  br label %2343

2343:                                             ; preds = %2343, %2330
  %2344 = phi i32 [ 0, %2330 ], [ %2349, %2343 ]
  %2345 = zext i32 %2344 to i64
  %2346 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2345
  %2347 = load i32, i32* %2346, align 4, !noalias !624
  %2348 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2345
  store i32 %2347, i32* %2348, align 4, !alias.scope !624
  %2349 = add nuw nsw i32 %2344, 1, !spirv.Decorations !620
  %2350 = icmp eq i32 %2344, 0
  br i1 %2350, label %2343, label %2351, !llvm.loop !627

2351:                                             ; preds = %2343
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2341)
  %2352 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2340)
  %2353 = ashr i64 %2352, 32
  %2354 = mul nsw i64 %2353, %const_reg_qword5, !spirv.Decorations !610
  %2355 = shl i64 %2352, 32
  %2356 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2354
  %2357 = ashr exact i64 %2355, 31
  %2358 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2356 to i8 addrspace(4)*
  %2359 = getelementptr i8, i8 addrspace(4)* %2358, i64 %2357
  %2360 = bitcast i8 addrspace(4)* %2359 to i16 addrspace(4)*
  %2361 = addrspacecast i16 addrspace(4)* %2360 to i16 addrspace(1)*
  %2362 = load i16, i16 addrspace(1)* %2361, align 2
  %2363 = zext i16 %2339 to i32
  %2364 = shl nuw i32 %2363, 16, !spirv.Decorations !628
  %2365 = bitcast i32 %2364 to float
  %2366 = zext i16 %2362 to i32
  %2367 = shl nuw i32 %2366, 16, !spirv.Decorations !628
  %2368 = bitcast i32 %2367 to float
  %2369 = fmul reassoc nsz arcp contract float %2365, %2368, !spirv.Decorations !612
  %2370 = fadd reassoc nsz arcp contract float %2369, %.sroa.106.1, !spirv.Decorations !612
  br label %._crit_edge.1.10

._crit_edge.1.10:                                 ; preds = %._crit_edge.10, %2351
  %.sroa.106.2 = phi float [ %2370, %2351 ], [ %.sroa.106.1, %._crit_edge.10 ]
  %2371 = and i1 %188, %2262
  br i1 %2371, label %2372, label %._crit_edge.2.10

2372:                                             ; preds = %._crit_edge.1.10
  %2373 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2373)
  %2374 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2374)
  %2375 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %2375, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2376

2376:                                             ; preds = %2376, %2372
  %2377 = phi i32 [ 0, %2372 ], [ %2382, %2376 ]
  %2378 = zext i32 %2377 to i64
  %2379 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2378
  %2380 = load i32, i32* %2379, align 4, !noalias !617
  %2381 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2378
  store i32 %2380, i32* %2381, align 4, !alias.scope !617
  %2382 = add nuw nsw i32 %2377, 1, !spirv.Decorations !620
  %2383 = icmp eq i32 %2377, 0
  br i1 %2383, label %2376, label %2384, !llvm.loop !622

2384:                                             ; preds = %2376
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2374)
  %2385 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2373)
  %2386 = shl i64 %2385, 32
  %2387 = ashr exact i64 %2386, 32
  %2388 = mul nsw i64 %2387, %const_reg_qword3, !spirv.Decorations !610
  %2389 = ashr i64 %2385, 32
  %2390 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2388, i32 0
  %2391 = getelementptr i16, i16 addrspace(4)* %2390, i64 %2389
  %2392 = addrspacecast i16 addrspace(4)* %2391 to i16 addrspace(1)*
  %2393 = load i16, i16 addrspace(1)* %2392, align 2
  %2394 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2394)
  %2395 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2395)
  %2396 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2396, align 4, !noalias !624
  store i32 %2261, i32* %57, align 4, !noalias !624
  br label %2397

2397:                                             ; preds = %2397, %2384
  %2398 = phi i32 [ 0, %2384 ], [ %2403, %2397 ]
  %2399 = zext i32 %2398 to i64
  %2400 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2399
  %2401 = load i32, i32* %2400, align 4, !noalias !624
  %2402 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2399
  store i32 %2401, i32* %2402, align 4, !alias.scope !624
  %2403 = add nuw nsw i32 %2398, 1, !spirv.Decorations !620
  %2404 = icmp eq i32 %2398, 0
  br i1 %2404, label %2397, label %2405, !llvm.loop !627

2405:                                             ; preds = %2397
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2395)
  %2406 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2394)
  %2407 = ashr i64 %2406, 32
  %2408 = mul nsw i64 %2407, %const_reg_qword5, !spirv.Decorations !610
  %2409 = shl i64 %2406, 32
  %2410 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2408
  %2411 = ashr exact i64 %2409, 31
  %2412 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2410 to i8 addrspace(4)*
  %2413 = getelementptr i8, i8 addrspace(4)* %2412, i64 %2411
  %2414 = bitcast i8 addrspace(4)* %2413 to i16 addrspace(4)*
  %2415 = addrspacecast i16 addrspace(4)* %2414 to i16 addrspace(1)*
  %2416 = load i16, i16 addrspace(1)* %2415, align 2
  %2417 = zext i16 %2393 to i32
  %2418 = shl nuw i32 %2417, 16, !spirv.Decorations !628
  %2419 = bitcast i32 %2418 to float
  %2420 = zext i16 %2416 to i32
  %2421 = shl nuw i32 %2420, 16, !spirv.Decorations !628
  %2422 = bitcast i32 %2421 to float
  %2423 = fmul reassoc nsz arcp contract float %2419, %2422, !spirv.Decorations !612
  %2424 = fadd reassoc nsz arcp contract float %2423, %.sroa.170.1, !spirv.Decorations !612
  br label %._crit_edge.2.10

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10, %2405
  %.sroa.170.2 = phi float [ %2424, %2405 ], [ %.sroa.170.1, %._crit_edge.1.10 ]
  %2425 = and i1 %244, %2262
  br i1 %2425, label %2426, label %.preheader.10

2426:                                             ; preds = %._crit_edge.2.10
  %2427 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2427)
  %2428 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2428)
  %2429 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %2429, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2430

2430:                                             ; preds = %2430, %2426
  %2431 = phi i32 [ 0, %2426 ], [ %2436, %2430 ]
  %2432 = zext i32 %2431 to i64
  %2433 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2432
  %2434 = load i32, i32* %2433, align 4, !noalias !617
  %2435 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2432
  store i32 %2434, i32* %2435, align 4, !alias.scope !617
  %2436 = add nuw nsw i32 %2431, 1, !spirv.Decorations !620
  %2437 = icmp eq i32 %2431, 0
  br i1 %2437, label %2430, label %2438, !llvm.loop !622

2438:                                             ; preds = %2430
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2428)
  %2439 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2427)
  %2440 = shl i64 %2439, 32
  %2441 = ashr exact i64 %2440, 32
  %2442 = mul nsw i64 %2441, %const_reg_qword3, !spirv.Decorations !610
  %2443 = ashr i64 %2439, 32
  %2444 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2442, i32 0
  %2445 = getelementptr i16, i16 addrspace(4)* %2444, i64 %2443
  %2446 = addrspacecast i16 addrspace(4)* %2445 to i16 addrspace(1)*
  %2447 = load i16, i16 addrspace(1)* %2446, align 2
  %2448 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2448)
  %2449 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2449)
  %2450 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2450, align 4, !noalias !624
  store i32 %2261, i32* %57, align 4, !noalias !624
  br label %2451

2451:                                             ; preds = %2451, %2438
  %2452 = phi i32 [ 0, %2438 ], [ %2457, %2451 ]
  %2453 = zext i32 %2452 to i64
  %2454 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2453
  %2455 = load i32, i32* %2454, align 4, !noalias !624
  %2456 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2453
  store i32 %2455, i32* %2456, align 4, !alias.scope !624
  %2457 = add nuw nsw i32 %2452, 1, !spirv.Decorations !620
  %2458 = icmp eq i32 %2452, 0
  br i1 %2458, label %2451, label %2459, !llvm.loop !627

2459:                                             ; preds = %2451
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2449)
  %2460 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2448)
  %2461 = ashr i64 %2460, 32
  %2462 = mul nsw i64 %2461, %const_reg_qword5, !spirv.Decorations !610
  %2463 = shl i64 %2460, 32
  %2464 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2462
  %2465 = ashr exact i64 %2463, 31
  %2466 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2464 to i8 addrspace(4)*
  %2467 = getelementptr i8, i8 addrspace(4)* %2466, i64 %2465
  %2468 = bitcast i8 addrspace(4)* %2467 to i16 addrspace(4)*
  %2469 = addrspacecast i16 addrspace(4)* %2468 to i16 addrspace(1)*
  %2470 = load i16, i16 addrspace(1)* %2469, align 2
  %2471 = zext i16 %2447 to i32
  %2472 = shl nuw i32 %2471, 16, !spirv.Decorations !628
  %2473 = bitcast i32 %2472 to float
  %2474 = zext i16 %2470 to i32
  %2475 = shl nuw i32 %2474, 16, !spirv.Decorations !628
  %2476 = bitcast i32 %2475 to float
  %2477 = fmul reassoc nsz arcp contract float %2473, %2476, !spirv.Decorations !612
  %2478 = fadd reassoc nsz arcp contract float %2477, %.sroa.234.1, !spirv.Decorations !612
  br label %.preheader.10

.preheader.10:                                    ; preds = %._crit_edge.2.10, %2459
  %.sroa.234.2 = phi float [ %2478, %2459 ], [ %.sroa.234.1, %._crit_edge.2.10 ]
  %2479 = or i32 %41, 11
  %2480 = icmp slt i32 %2479, %const_reg_dword1
  %2481 = and i1 %76, %2480
  br i1 %2481, label %2482, label %._crit_edge.11

2482:                                             ; preds = %.preheader.10
  %2483 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2483)
  %2484 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2484)
  %2485 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2485, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2486

2486:                                             ; preds = %2486, %2482
  %2487 = phi i32 [ 0, %2482 ], [ %2492, %2486 ]
  %2488 = zext i32 %2487 to i64
  %2489 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2488
  %2490 = load i32, i32* %2489, align 4, !noalias !617
  %2491 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2488
  store i32 %2490, i32* %2491, align 4, !alias.scope !617
  %2492 = add nuw nsw i32 %2487, 1, !spirv.Decorations !620
  %2493 = icmp eq i32 %2487, 0
  br i1 %2493, label %2486, label %2494, !llvm.loop !622

2494:                                             ; preds = %2486
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2484)
  %2495 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2483)
  %2496 = shl i64 %2495, 32
  %2497 = ashr exact i64 %2496, 32
  %2498 = mul nsw i64 %2497, %const_reg_qword3, !spirv.Decorations !610
  %2499 = ashr i64 %2495, 32
  %2500 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2498, i32 0
  %2501 = getelementptr i16, i16 addrspace(4)* %2500, i64 %2499
  %2502 = addrspacecast i16 addrspace(4)* %2501 to i16 addrspace(1)*
  %2503 = load i16, i16 addrspace(1)* %2502, align 2
  %2504 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2504)
  %2505 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2505)
  %2506 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2506, align 4, !noalias !624
  store i32 %2479, i32* %57, align 4, !noalias !624
  br label %2507

2507:                                             ; preds = %2507, %2494
  %2508 = phi i32 [ 0, %2494 ], [ %2513, %2507 ]
  %2509 = zext i32 %2508 to i64
  %2510 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2509
  %2511 = load i32, i32* %2510, align 4, !noalias !624
  %2512 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2509
  store i32 %2511, i32* %2512, align 4, !alias.scope !624
  %2513 = add nuw nsw i32 %2508, 1, !spirv.Decorations !620
  %2514 = icmp eq i32 %2508, 0
  br i1 %2514, label %2507, label %2515, !llvm.loop !627

2515:                                             ; preds = %2507
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2505)
  %2516 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2504)
  %2517 = ashr i64 %2516, 32
  %2518 = mul nsw i64 %2517, %const_reg_qword5, !spirv.Decorations !610
  %2519 = shl i64 %2516, 32
  %2520 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2518
  %2521 = ashr exact i64 %2519, 31
  %2522 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2520 to i8 addrspace(4)*
  %2523 = getelementptr i8, i8 addrspace(4)* %2522, i64 %2521
  %2524 = bitcast i8 addrspace(4)* %2523 to i16 addrspace(4)*
  %2525 = addrspacecast i16 addrspace(4)* %2524 to i16 addrspace(1)*
  %2526 = load i16, i16 addrspace(1)* %2525, align 2
  %2527 = zext i16 %2503 to i32
  %2528 = shl nuw i32 %2527, 16, !spirv.Decorations !628
  %2529 = bitcast i32 %2528 to float
  %2530 = zext i16 %2526 to i32
  %2531 = shl nuw i32 %2530, 16, !spirv.Decorations !628
  %2532 = bitcast i32 %2531 to float
  %2533 = fmul reassoc nsz arcp contract float %2529, %2532, !spirv.Decorations !612
  %2534 = fadd reassoc nsz arcp contract float %2533, %.sroa.46.1, !spirv.Decorations !612
  br label %._crit_edge.11

._crit_edge.11:                                   ; preds = %.preheader.10, %2515
  %.sroa.46.2 = phi float [ %2534, %2515 ], [ %.sroa.46.1, %.preheader.10 ]
  %2535 = and i1 %132, %2480
  br i1 %2535, label %2536, label %._crit_edge.1.11

2536:                                             ; preds = %._crit_edge.11
  %2537 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2537)
  %2538 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2538)
  %2539 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %2539, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2540

2540:                                             ; preds = %2540, %2536
  %2541 = phi i32 [ 0, %2536 ], [ %2546, %2540 ]
  %2542 = zext i32 %2541 to i64
  %2543 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2542
  %2544 = load i32, i32* %2543, align 4, !noalias !617
  %2545 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2542
  store i32 %2544, i32* %2545, align 4, !alias.scope !617
  %2546 = add nuw nsw i32 %2541, 1, !spirv.Decorations !620
  %2547 = icmp eq i32 %2541, 0
  br i1 %2547, label %2540, label %2548, !llvm.loop !622

2548:                                             ; preds = %2540
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2538)
  %2549 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2537)
  %2550 = shl i64 %2549, 32
  %2551 = ashr exact i64 %2550, 32
  %2552 = mul nsw i64 %2551, %const_reg_qword3, !spirv.Decorations !610
  %2553 = ashr i64 %2549, 32
  %2554 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2552, i32 0
  %2555 = getelementptr i16, i16 addrspace(4)* %2554, i64 %2553
  %2556 = addrspacecast i16 addrspace(4)* %2555 to i16 addrspace(1)*
  %2557 = load i16, i16 addrspace(1)* %2556, align 2
  %2558 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2558)
  %2559 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2559)
  %2560 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2560, align 4, !noalias !624
  store i32 %2479, i32* %57, align 4, !noalias !624
  br label %2561

2561:                                             ; preds = %2561, %2548
  %2562 = phi i32 [ 0, %2548 ], [ %2567, %2561 ]
  %2563 = zext i32 %2562 to i64
  %2564 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2563
  %2565 = load i32, i32* %2564, align 4, !noalias !624
  %2566 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2563
  store i32 %2565, i32* %2566, align 4, !alias.scope !624
  %2567 = add nuw nsw i32 %2562, 1, !spirv.Decorations !620
  %2568 = icmp eq i32 %2562, 0
  br i1 %2568, label %2561, label %2569, !llvm.loop !627

2569:                                             ; preds = %2561
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2559)
  %2570 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2558)
  %2571 = ashr i64 %2570, 32
  %2572 = mul nsw i64 %2571, %const_reg_qword5, !spirv.Decorations !610
  %2573 = shl i64 %2570, 32
  %2574 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2572
  %2575 = ashr exact i64 %2573, 31
  %2576 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2574 to i8 addrspace(4)*
  %2577 = getelementptr i8, i8 addrspace(4)* %2576, i64 %2575
  %2578 = bitcast i8 addrspace(4)* %2577 to i16 addrspace(4)*
  %2579 = addrspacecast i16 addrspace(4)* %2578 to i16 addrspace(1)*
  %2580 = load i16, i16 addrspace(1)* %2579, align 2
  %2581 = zext i16 %2557 to i32
  %2582 = shl nuw i32 %2581, 16, !spirv.Decorations !628
  %2583 = bitcast i32 %2582 to float
  %2584 = zext i16 %2580 to i32
  %2585 = shl nuw i32 %2584, 16, !spirv.Decorations !628
  %2586 = bitcast i32 %2585 to float
  %2587 = fmul reassoc nsz arcp contract float %2583, %2586, !spirv.Decorations !612
  %2588 = fadd reassoc nsz arcp contract float %2587, %.sroa.110.1, !spirv.Decorations !612
  br label %._crit_edge.1.11

._crit_edge.1.11:                                 ; preds = %._crit_edge.11, %2569
  %.sroa.110.2 = phi float [ %2588, %2569 ], [ %.sroa.110.1, %._crit_edge.11 ]
  %2589 = and i1 %188, %2480
  br i1 %2589, label %2590, label %._crit_edge.2.11

2590:                                             ; preds = %._crit_edge.1.11
  %2591 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2591)
  %2592 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2592)
  %2593 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %2593, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2594

2594:                                             ; preds = %2594, %2590
  %2595 = phi i32 [ 0, %2590 ], [ %2600, %2594 ]
  %2596 = zext i32 %2595 to i64
  %2597 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2596
  %2598 = load i32, i32* %2597, align 4, !noalias !617
  %2599 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2596
  store i32 %2598, i32* %2599, align 4, !alias.scope !617
  %2600 = add nuw nsw i32 %2595, 1, !spirv.Decorations !620
  %2601 = icmp eq i32 %2595, 0
  br i1 %2601, label %2594, label %2602, !llvm.loop !622

2602:                                             ; preds = %2594
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2592)
  %2603 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2591)
  %2604 = shl i64 %2603, 32
  %2605 = ashr exact i64 %2604, 32
  %2606 = mul nsw i64 %2605, %const_reg_qword3, !spirv.Decorations !610
  %2607 = ashr i64 %2603, 32
  %2608 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2606, i32 0
  %2609 = getelementptr i16, i16 addrspace(4)* %2608, i64 %2607
  %2610 = addrspacecast i16 addrspace(4)* %2609 to i16 addrspace(1)*
  %2611 = load i16, i16 addrspace(1)* %2610, align 2
  %2612 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2612)
  %2613 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2613)
  %2614 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2614, align 4, !noalias !624
  store i32 %2479, i32* %57, align 4, !noalias !624
  br label %2615

2615:                                             ; preds = %2615, %2602
  %2616 = phi i32 [ 0, %2602 ], [ %2621, %2615 ]
  %2617 = zext i32 %2616 to i64
  %2618 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2617
  %2619 = load i32, i32* %2618, align 4, !noalias !624
  %2620 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2617
  store i32 %2619, i32* %2620, align 4, !alias.scope !624
  %2621 = add nuw nsw i32 %2616, 1, !spirv.Decorations !620
  %2622 = icmp eq i32 %2616, 0
  br i1 %2622, label %2615, label %2623, !llvm.loop !627

2623:                                             ; preds = %2615
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2613)
  %2624 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2612)
  %2625 = ashr i64 %2624, 32
  %2626 = mul nsw i64 %2625, %const_reg_qword5, !spirv.Decorations !610
  %2627 = shl i64 %2624, 32
  %2628 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2626
  %2629 = ashr exact i64 %2627, 31
  %2630 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2628 to i8 addrspace(4)*
  %2631 = getelementptr i8, i8 addrspace(4)* %2630, i64 %2629
  %2632 = bitcast i8 addrspace(4)* %2631 to i16 addrspace(4)*
  %2633 = addrspacecast i16 addrspace(4)* %2632 to i16 addrspace(1)*
  %2634 = load i16, i16 addrspace(1)* %2633, align 2
  %2635 = zext i16 %2611 to i32
  %2636 = shl nuw i32 %2635, 16, !spirv.Decorations !628
  %2637 = bitcast i32 %2636 to float
  %2638 = zext i16 %2634 to i32
  %2639 = shl nuw i32 %2638, 16, !spirv.Decorations !628
  %2640 = bitcast i32 %2639 to float
  %2641 = fmul reassoc nsz arcp contract float %2637, %2640, !spirv.Decorations !612
  %2642 = fadd reassoc nsz arcp contract float %2641, %.sroa.174.1, !spirv.Decorations !612
  br label %._crit_edge.2.11

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11, %2623
  %.sroa.174.2 = phi float [ %2642, %2623 ], [ %.sroa.174.1, %._crit_edge.1.11 ]
  %2643 = and i1 %244, %2480
  br i1 %2643, label %2644, label %.preheader.11

2644:                                             ; preds = %._crit_edge.2.11
  %2645 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2645)
  %2646 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2646)
  %2647 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %2647, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2648

2648:                                             ; preds = %2648, %2644
  %2649 = phi i32 [ 0, %2644 ], [ %2654, %2648 ]
  %2650 = zext i32 %2649 to i64
  %2651 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2650
  %2652 = load i32, i32* %2651, align 4, !noalias !617
  %2653 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2650
  store i32 %2652, i32* %2653, align 4, !alias.scope !617
  %2654 = add nuw nsw i32 %2649, 1, !spirv.Decorations !620
  %2655 = icmp eq i32 %2649, 0
  br i1 %2655, label %2648, label %2656, !llvm.loop !622

2656:                                             ; preds = %2648
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2646)
  %2657 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2645)
  %2658 = shl i64 %2657, 32
  %2659 = ashr exact i64 %2658, 32
  %2660 = mul nsw i64 %2659, %const_reg_qword3, !spirv.Decorations !610
  %2661 = ashr i64 %2657, 32
  %2662 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2660, i32 0
  %2663 = getelementptr i16, i16 addrspace(4)* %2662, i64 %2661
  %2664 = addrspacecast i16 addrspace(4)* %2663 to i16 addrspace(1)*
  %2665 = load i16, i16 addrspace(1)* %2664, align 2
  %2666 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2666)
  %2667 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2667)
  %2668 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2668, align 4, !noalias !624
  store i32 %2479, i32* %57, align 4, !noalias !624
  br label %2669

2669:                                             ; preds = %2669, %2656
  %2670 = phi i32 [ 0, %2656 ], [ %2675, %2669 ]
  %2671 = zext i32 %2670 to i64
  %2672 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2671
  %2673 = load i32, i32* %2672, align 4, !noalias !624
  %2674 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2671
  store i32 %2673, i32* %2674, align 4, !alias.scope !624
  %2675 = add nuw nsw i32 %2670, 1, !spirv.Decorations !620
  %2676 = icmp eq i32 %2670, 0
  br i1 %2676, label %2669, label %2677, !llvm.loop !627

2677:                                             ; preds = %2669
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2667)
  %2678 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2666)
  %2679 = ashr i64 %2678, 32
  %2680 = mul nsw i64 %2679, %const_reg_qword5, !spirv.Decorations !610
  %2681 = shl i64 %2678, 32
  %2682 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2680
  %2683 = ashr exact i64 %2681, 31
  %2684 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2682 to i8 addrspace(4)*
  %2685 = getelementptr i8, i8 addrspace(4)* %2684, i64 %2683
  %2686 = bitcast i8 addrspace(4)* %2685 to i16 addrspace(4)*
  %2687 = addrspacecast i16 addrspace(4)* %2686 to i16 addrspace(1)*
  %2688 = load i16, i16 addrspace(1)* %2687, align 2
  %2689 = zext i16 %2665 to i32
  %2690 = shl nuw i32 %2689, 16, !spirv.Decorations !628
  %2691 = bitcast i32 %2690 to float
  %2692 = zext i16 %2688 to i32
  %2693 = shl nuw i32 %2692, 16, !spirv.Decorations !628
  %2694 = bitcast i32 %2693 to float
  %2695 = fmul reassoc nsz arcp contract float %2691, %2694, !spirv.Decorations !612
  %2696 = fadd reassoc nsz arcp contract float %2695, %.sroa.238.1, !spirv.Decorations !612
  br label %.preheader.11

.preheader.11:                                    ; preds = %._crit_edge.2.11, %2677
  %.sroa.238.2 = phi float [ %2696, %2677 ], [ %.sroa.238.1, %._crit_edge.2.11 ]
  %2697 = or i32 %41, 12
  %2698 = icmp slt i32 %2697, %const_reg_dword1
  %2699 = and i1 %76, %2698
  br i1 %2699, label %2700, label %._crit_edge.12

2700:                                             ; preds = %.preheader.11
  %2701 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2701)
  %2702 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2702)
  %2703 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2703, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2704

2704:                                             ; preds = %2704, %2700
  %2705 = phi i32 [ 0, %2700 ], [ %2710, %2704 ]
  %2706 = zext i32 %2705 to i64
  %2707 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2706
  %2708 = load i32, i32* %2707, align 4, !noalias !617
  %2709 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2706
  store i32 %2708, i32* %2709, align 4, !alias.scope !617
  %2710 = add nuw nsw i32 %2705, 1, !spirv.Decorations !620
  %2711 = icmp eq i32 %2705, 0
  br i1 %2711, label %2704, label %2712, !llvm.loop !622

2712:                                             ; preds = %2704
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2702)
  %2713 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2701)
  %2714 = shl i64 %2713, 32
  %2715 = ashr exact i64 %2714, 32
  %2716 = mul nsw i64 %2715, %const_reg_qword3, !spirv.Decorations !610
  %2717 = ashr i64 %2713, 32
  %2718 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2716, i32 0
  %2719 = getelementptr i16, i16 addrspace(4)* %2718, i64 %2717
  %2720 = addrspacecast i16 addrspace(4)* %2719 to i16 addrspace(1)*
  %2721 = load i16, i16 addrspace(1)* %2720, align 2
  %2722 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2722)
  %2723 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2723)
  %2724 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2724, align 4, !noalias !624
  store i32 %2697, i32* %57, align 4, !noalias !624
  br label %2725

2725:                                             ; preds = %2725, %2712
  %2726 = phi i32 [ 0, %2712 ], [ %2731, %2725 ]
  %2727 = zext i32 %2726 to i64
  %2728 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2727
  %2729 = load i32, i32* %2728, align 4, !noalias !624
  %2730 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2727
  store i32 %2729, i32* %2730, align 4, !alias.scope !624
  %2731 = add nuw nsw i32 %2726, 1, !spirv.Decorations !620
  %2732 = icmp eq i32 %2726, 0
  br i1 %2732, label %2725, label %2733, !llvm.loop !627

2733:                                             ; preds = %2725
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2723)
  %2734 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2722)
  %2735 = ashr i64 %2734, 32
  %2736 = mul nsw i64 %2735, %const_reg_qword5, !spirv.Decorations !610
  %2737 = shl i64 %2734, 32
  %2738 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2736
  %2739 = ashr exact i64 %2737, 31
  %2740 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2738 to i8 addrspace(4)*
  %2741 = getelementptr i8, i8 addrspace(4)* %2740, i64 %2739
  %2742 = bitcast i8 addrspace(4)* %2741 to i16 addrspace(4)*
  %2743 = addrspacecast i16 addrspace(4)* %2742 to i16 addrspace(1)*
  %2744 = load i16, i16 addrspace(1)* %2743, align 2
  %2745 = zext i16 %2721 to i32
  %2746 = shl nuw i32 %2745, 16, !spirv.Decorations !628
  %2747 = bitcast i32 %2746 to float
  %2748 = zext i16 %2744 to i32
  %2749 = shl nuw i32 %2748, 16, !spirv.Decorations !628
  %2750 = bitcast i32 %2749 to float
  %2751 = fmul reassoc nsz arcp contract float %2747, %2750, !spirv.Decorations !612
  %2752 = fadd reassoc nsz arcp contract float %2751, %.sroa.50.1, !spirv.Decorations !612
  br label %._crit_edge.12

._crit_edge.12:                                   ; preds = %.preheader.11, %2733
  %.sroa.50.2 = phi float [ %2752, %2733 ], [ %.sroa.50.1, %.preheader.11 ]
  %2753 = and i1 %132, %2698
  br i1 %2753, label %2754, label %._crit_edge.1.12

2754:                                             ; preds = %._crit_edge.12
  %2755 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2755)
  %2756 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2756)
  %2757 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %2757, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2758

2758:                                             ; preds = %2758, %2754
  %2759 = phi i32 [ 0, %2754 ], [ %2764, %2758 ]
  %2760 = zext i32 %2759 to i64
  %2761 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2760
  %2762 = load i32, i32* %2761, align 4, !noalias !617
  %2763 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2760
  store i32 %2762, i32* %2763, align 4, !alias.scope !617
  %2764 = add nuw nsw i32 %2759, 1, !spirv.Decorations !620
  %2765 = icmp eq i32 %2759, 0
  br i1 %2765, label %2758, label %2766, !llvm.loop !622

2766:                                             ; preds = %2758
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2756)
  %2767 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2755)
  %2768 = shl i64 %2767, 32
  %2769 = ashr exact i64 %2768, 32
  %2770 = mul nsw i64 %2769, %const_reg_qword3, !spirv.Decorations !610
  %2771 = ashr i64 %2767, 32
  %2772 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2770, i32 0
  %2773 = getelementptr i16, i16 addrspace(4)* %2772, i64 %2771
  %2774 = addrspacecast i16 addrspace(4)* %2773 to i16 addrspace(1)*
  %2775 = load i16, i16 addrspace(1)* %2774, align 2
  %2776 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2776)
  %2777 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2777)
  %2778 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2778, align 4, !noalias !624
  store i32 %2697, i32* %57, align 4, !noalias !624
  br label %2779

2779:                                             ; preds = %2779, %2766
  %2780 = phi i32 [ 0, %2766 ], [ %2785, %2779 ]
  %2781 = zext i32 %2780 to i64
  %2782 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2781
  %2783 = load i32, i32* %2782, align 4, !noalias !624
  %2784 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2781
  store i32 %2783, i32* %2784, align 4, !alias.scope !624
  %2785 = add nuw nsw i32 %2780, 1, !spirv.Decorations !620
  %2786 = icmp eq i32 %2780, 0
  br i1 %2786, label %2779, label %2787, !llvm.loop !627

2787:                                             ; preds = %2779
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2777)
  %2788 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2776)
  %2789 = ashr i64 %2788, 32
  %2790 = mul nsw i64 %2789, %const_reg_qword5, !spirv.Decorations !610
  %2791 = shl i64 %2788, 32
  %2792 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2790
  %2793 = ashr exact i64 %2791, 31
  %2794 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2792 to i8 addrspace(4)*
  %2795 = getelementptr i8, i8 addrspace(4)* %2794, i64 %2793
  %2796 = bitcast i8 addrspace(4)* %2795 to i16 addrspace(4)*
  %2797 = addrspacecast i16 addrspace(4)* %2796 to i16 addrspace(1)*
  %2798 = load i16, i16 addrspace(1)* %2797, align 2
  %2799 = zext i16 %2775 to i32
  %2800 = shl nuw i32 %2799, 16, !spirv.Decorations !628
  %2801 = bitcast i32 %2800 to float
  %2802 = zext i16 %2798 to i32
  %2803 = shl nuw i32 %2802, 16, !spirv.Decorations !628
  %2804 = bitcast i32 %2803 to float
  %2805 = fmul reassoc nsz arcp contract float %2801, %2804, !spirv.Decorations !612
  %2806 = fadd reassoc nsz arcp contract float %2805, %.sroa.114.1, !spirv.Decorations !612
  br label %._crit_edge.1.12

._crit_edge.1.12:                                 ; preds = %._crit_edge.12, %2787
  %.sroa.114.2 = phi float [ %2806, %2787 ], [ %.sroa.114.1, %._crit_edge.12 ]
  %2807 = and i1 %188, %2698
  br i1 %2807, label %2808, label %._crit_edge.2.12

2808:                                             ; preds = %._crit_edge.1.12
  %2809 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2809)
  %2810 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2810)
  %2811 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %2811, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2812

2812:                                             ; preds = %2812, %2808
  %2813 = phi i32 [ 0, %2808 ], [ %2818, %2812 ]
  %2814 = zext i32 %2813 to i64
  %2815 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2814
  %2816 = load i32, i32* %2815, align 4, !noalias !617
  %2817 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2814
  store i32 %2816, i32* %2817, align 4, !alias.scope !617
  %2818 = add nuw nsw i32 %2813, 1, !spirv.Decorations !620
  %2819 = icmp eq i32 %2813, 0
  br i1 %2819, label %2812, label %2820, !llvm.loop !622

2820:                                             ; preds = %2812
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2810)
  %2821 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2809)
  %2822 = shl i64 %2821, 32
  %2823 = ashr exact i64 %2822, 32
  %2824 = mul nsw i64 %2823, %const_reg_qword3, !spirv.Decorations !610
  %2825 = ashr i64 %2821, 32
  %2826 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2824, i32 0
  %2827 = getelementptr i16, i16 addrspace(4)* %2826, i64 %2825
  %2828 = addrspacecast i16 addrspace(4)* %2827 to i16 addrspace(1)*
  %2829 = load i16, i16 addrspace(1)* %2828, align 2
  %2830 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2830)
  %2831 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2831)
  %2832 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2832, align 4, !noalias !624
  store i32 %2697, i32* %57, align 4, !noalias !624
  br label %2833

2833:                                             ; preds = %2833, %2820
  %2834 = phi i32 [ 0, %2820 ], [ %2839, %2833 ]
  %2835 = zext i32 %2834 to i64
  %2836 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2835
  %2837 = load i32, i32* %2836, align 4, !noalias !624
  %2838 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2835
  store i32 %2837, i32* %2838, align 4, !alias.scope !624
  %2839 = add nuw nsw i32 %2834, 1, !spirv.Decorations !620
  %2840 = icmp eq i32 %2834, 0
  br i1 %2840, label %2833, label %2841, !llvm.loop !627

2841:                                             ; preds = %2833
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2831)
  %2842 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2830)
  %2843 = ashr i64 %2842, 32
  %2844 = mul nsw i64 %2843, %const_reg_qword5, !spirv.Decorations !610
  %2845 = shl i64 %2842, 32
  %2846 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2844
  %2847 = ashr exact i64 %2845, 31
  %2848 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2846 to i8 addrspace(4)*
  %2849 = getelementptr i8, i8 addrspace(4)* %2848, i64 %2847
  %2850 = bitcast i8 addrspace(4)* %2849 to i16 addrspace(4)*
  %2851 = addrspacecast i16 addrspace(4)* %2850 to i16 addrspace(1)*
  %2852 = load i16, i16 addrspace(1)* %2851, align 2
  %2853 = zext i16 %2829 to i32
  %2854 = shl nuw i32 %2853, 16, !spirv.Decorations !628
  %2855 = bitcast i32 %2854 to float
  %2856 = zext i16 %2852 to i32
  %2857 = shl nuw i32 %2856, 16, !spirv.Decorations !628
  %2858 = bitcast i32 %2857 to float
  %2859 = fmul reassoc nsz arcp contract float %2855, %2858, !spirv.Decorations !612
  %2860 = fadd reassoc nsz arcp contract float %2859, %.sroa.178.1, !spirv.Decorations !612
  br label %._crit_edge.2.12

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12, %2841
  %.sroa.178.2 = phi float [ %2860, %2841 ], [ %.sroa.178.1, %._crit_edge.1.12 ]
  %2861 = and i1 %244, %2698
  br i1 %2861, label %2862, label %.preheader.12

2862:                                             ; preds = %._crit_edge.2.12
  %2863 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2863)
  %2864 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2864)
  %2865 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %2865, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2866

2866:                                             ; preds = %2866, %2862
  %2867 = phi i32 [ 0, %2862 ], [ %2872, %2866 ]
  %2868 = zext i32 %2867 to i64
  %2869 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2868
  %2870 = load i32, i32* %2869, align 4, !noalias !617
  %2871 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2868
  store i32 %2870, i32* %2871, align 4, !alias.scope !617
  %2872 = add nuw nsw i32 %2867, 1, !spirv.Decorations !620
  %2873 = icmp eq i32 %2867, 0
  br i1 %2873, label %2866, label %2874, !llvm.loop !622

2874:                                             ; preds = %2866
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2864)
  %2875 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2863)
  %2876 = shl i64 %2875, 32
  %2877 = ashr exact i64 %2876, 32
  %2878 = mul nsw i64 %2877, %const_reg_qword3, !spirv.Decorations !610
  %2879 = ashr i64 %2875, 32
  %2880 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2878, i32 0
  %2881 = getelementptr i16, i16 addrspace(4)* %2880, i64 %2879
  %2882 = addrspacecast i16 addrspace(4)* %2881 to i16 addrspace(1)*
  %2883 = load i16, i16 addrspace(1)* %2882, align 2
  %2884 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2884)
  %2885 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2885)
  %2886 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2886, align 4, !noalias !624
  store i32 %2697, i32* %57, align 4, !noalias !624
  br label %2887

2887:                                             ; preds = %2887, %2874
  %2888 = phi i32 [ 0, %2874 ], [ %2893, %2887 ]
  %2889 = zext i32 %2888 to i64
  %2890 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2889
  %2891 = load i32, i32* %2890, align 4, !noalias !624
  %2892 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2889
  store i32 %2891, i32* %2892, align 4, !alias.scope !624
  %2893 = add nuw nsw i32 %2888, 1, !spirv.Decorations !620
  %2894 = icmp eq i32 %2888, 0
  br i1 %2894, label %2887, label %2895, !llvm.loop !627

2895:                                             ; preds = %2887
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2885)
  %2896 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2884)
  %2897 = ashr i64 %2896, 32
  %2898 = mul nsw i64 %2897, %const_reg_qword5, !spirv.Decorations !610
  %2899 = shl i64 %2896, 32
  %2900 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2898
  %2901 = ashr exact i64 %2899, 31
  %2902 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2900 to i8 addrspace(4)*
  %2903 = getelementptr i8, i8 addrspace(4)* %2902, i64 %2901
  %2904 = bitcast i8 addrspace(4)* %2903 to i16 addrspace(4)*
  %2905 = addrspacecast i16 addrspace(4)* %2904 to i16 addrspace(1)*
  %2906 = load i16, i16 addrspace(1)* %2905, align 2
  %2907 = zext i16 %2883 to i32
  %2908 = shl nuw i32 %2907, 16, !spirv.Decorations !628
  %2909 = bitcast i32 %2908 to float
  %2910 = zext i16 %2906 to i32
  %2911 = shl nuw i32 %2910, 16, !spirv.Decorations !628
  %2912 = bitcast i32 %2911 to float
  %2913 = fmul reassoc nsz arcp contract float %2909, %2912, !spirv.Decorations !612
  %2914 = fadd reassoc nsz arcp contract float %2913, %.sroa.242.1, !spirv.Decorations !612
  br label %.preheader.12

.preheader.12:                                    ; preds = %._crit_edge.2.12, %2895
  %.sroa.242.2 = phi float [ %2914, %2895 ], [ %.sroa.242.1, %._crit_edge.2.12 ]
  %2915 = or i32 %41, 13
  %2916 = icmp slt i32 %2915, %const_reg_dword1
  %2917 = and i1 %76, %2916
  br i1 %2917, label %2918, label %._crit_edge.13

2918:                                             ; preds = %.preheader.12
  %2919 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2919)
  %2920 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2920)
  %2921 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2921, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2922

2922:                                             ; preds = %2922, %2918
  %2923 = phi i32 [ 0, %2918 ], [ %2928, %2922 ]
  %2924 = zext i32 %2923 to i64
  %2925 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2924
  %2926 = load i32, i32* %2925, align 4, !noalias !617
  %2927 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2924
  store i32 %2926, i32* %2927, align 4, !alias.scope !617
  %2928 = add nuw nsw i32 %2923, 1, !spirv.Decorations !620
  %2929 = icmp eq i32 %2923, 0
  br i1 %2929, label %2922, label %2930, !llvm.loop !622

2930:                                             ; preds = %2922
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2920)
  %2931 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2919)
  %2932 = shl i64 %2931, 32
  %2933 = ashr exact i64 %2932, 32
  %2934 = mul nsw i64 %2933, %const_reg_qword3, !spirv.Decorations !610
  %2935 = ashr i64 %2931, 32
  %2936 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2934, i32 0
  %2937 = getelementptr i16, i16 addrspace(4)* %2936, i64 %2935
  %2938 = addrspacecast i16 addrspace(4)* %2937 to i16 addrspace(1)*
  %2939 = load i16, i16 addrspace(1)* %2938, align 2
  %2940 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2940)
  %2941 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2941)
  %2942 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2942, align 4, !noalias !624
  store i32 %2915, i32* %57, align 4, !noalias !624
  br label %2943

2943:                                             ; preds = %2943, %2930
  %2944 = phi i32 [ 0, %2930 ], [ %2949, %2943 ]
  %2945 = zext i32 %2944 to i64
  %2946 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2945
  %2947 = load i32, i32* %2946, align 4, !noalias !624
  %2948 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2945
  store i32 %2947, i32* %2948, align 4, !alias.scope !624
  %2949 = add nuw nsw i32 %2944, 1, !spirv.Decorations !620
  %2950 = icmp eq i32 %2944, 0
  br i1 %2950, label %2943, label %2951, !llvm.loop !627

2951:                                             ; preds = %2943
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2941)
  %2952 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2940)
  %2953 = ashr i64 %2952, 32
  %2954 = mul nsw i64 %2953, %const_reg_qword5, !spirv.Decorations !610
  %2955 = shl i64 %2952, 32
  %2956 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2954
  %2957 = ashr exact i64 %2955, 31
  %2958 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %2956 to i8 addrspace(4)*
  %2959 = getelementptr i8, i8 addrspace(4)* %2958, i64 %2957
  %2960 = bitcast i8 addrspace(4)* %2959 to i16 addrspace(4)*
  %2961 = addrspacecast i16 addrspace(4)* %2960 to i16 addrspace(1)*
  %2962 = load i16, i16 addrspace(1)* %2961, align 2
  %2963 = zext i16 %2939 to i32
  %2964 = shl nuw i32 %2963, 16, !spirv.Decorations !628
  %2965 = bitcast i32 %2964 to float
  %2966 = zext i16 %2962 to i32
  %2967 = shl nuw i32 %2966, 16, !spirv.Decorations !628
  %2968 = bitcast i32 %2967 to float
  %2969 = fmul reassoc nsz arcp contract float %2965, %2968, !spirv.Decorations !612
  %2970 = fadd reassoc nsz arcp contract float %2969, %.sroa.54.1, !spirv.Decorations !612
  br label %._crit_edge.13

._crit_edge.13:                                   ; preds = %.preheader.12, %2951
  %.sroa.54.2 = phi float [ %2970, %2951 ], [ %.sroa.54.1, %.preheader.12 ]
  %2971 = and i1 %132, %2916
  br i1 %2971, label %2972, label %._crit_edge.1.13

2972:                                             ; preds = %._crit_edge.13
  %2973 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2973)
  %2974 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2974)
  %2975 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %2975, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %2976

2976:                                             ; preds = %2976, %2972
  %2977 = phi i32 [ 0, %2972 ], [ %2982, %2976 ]
  %2978 = zext i32 %2977 to i64
  %2979 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2978
  %2980 = load i32, i32* %2979, align 4, !noalias !617
  %2981 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2978
  store i32 %2980, i32* %2981, align 4, !alias.scope !617
  %2982 = add nuw nsw i32 %2977, 1, !spirv.Decorations !620
  %2983 = icmp eq i32 %2977, 0
  br i1 %2983, label %2976, label %2984, !llvm.loop !622

2984:                                             ; preds = %2976
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2974)
  %2985 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2973)
  %2986 = shl i64 %2985, 32
  %2987 = ashr exact i64 %2986, 32
  %2988 = mul nsw i64 %2987, %const_reg_qword3, !spirv.Decorations !610
  %2989 = ashr i64 %2985, 32
  %2990 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2988, i32 0
  %2991 = getelementptr i16, i16 addrspace(4)* %2990, i64 %2989
  %2992 = addrspacecast i16 addrspace(4)* %2991 to i16 addrspace(1)*
  %2993 = load i16, i16 addrspace(1)* %2992, align 2
  %2994 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2994)
  %2995 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2995)
  %2996 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2996, align 4, !noalias !624
  store i32 %2915, i32* %57, align 4, !noalias !624
  br label %2997

2997:                                             ; preds = %2997, %2984
  %2998 = phi i32 [ 0, %2984 ], [ %3003, %2997 ]
  %2999 = zext i32 %2998 to i64
  %3000 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2999
  %3001 = load i32, i32* %3000, align 4, !noalias !624
  %3002 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2999
  store i32 %3001, i32* %3002, align 4, !alias.scope !624
  %3003 = add nuw nsw i32 %2998, 1, !spirv.Decorations !620
  %3004 = icmp eq i32 %2998, 0
  br i1 %3004, label %2997, label %3005, !llvm.loop !627

3005:                                             ; preds = %2997
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2995)
  %3006 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2994)
  %3007 = ashr i64 %3006, 32
  %3008 = mul nsw i64 %3007, %const_reg_qword5, !spirv.Decorations !610
  %3009 = shl i64 %3006, 32
  %3010 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3008
  %3011 = ashr exact i64 %3009, 31
  %3012 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3010 to i8 addrspace(4)*
  %3013 = getelementptr i8, i8 addrspace(4)* %3012, i64 %3011
  %3014 = bitcast i8 addrspace(4)* %3013 to i16 addrspace(4)*
  %3015 = addrspacecast i16 addrspace(4)* %3014 to i16 addrspace(1)*
  %3016 = load i16, i16 addrspace(1)* %3015, align 2
  %3017 = zext i16 %2993 to i32
  %3018 = shl nuw i32 %3017, 16, !spirv.Decorations !628
  %3019 = bitcast i32 %3018 to float
  %3020 = zext i16 %3016 to i32
  %3021 = shl nuw i32 %3020, 16, !spirv.Decorations !628
  %3022 = bitcast i32 %3021 to float
  %3023 = fmul reassoc nsz arcp contract float %3019, %3022, !spirv.Decorations !612
  %3024 = fadd reassoc nsz arcp contract float %3023, %.sroa.118.1, !spirv.Decorations !612
  br label %._crit_edge.1.13

._crit_edge.1.13:                                 ; preds = %._crit_edge.13, %3005
  %.sroa.118.2 = phi float [ %3024, %3005 ], [ %.sroa.118.1, %._crit_edge.13 ]
  %3025 = and i1 %188, %2916
  br i1 %3025, label %3026, label %._crit_edge.2.13

3026:                                             ; preds = %._crit_edge.1.13
  %3027 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3027)
  %3028 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3028)
  %3029 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %3029, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3030

3030:                                             ; preds = %3030, %3026
  %3031 = phi i32 [ 0, %3026 ], [ %3036, %3030 ]
  %3032 = zext i32 %3031 to i64
  %3033 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3032
  %3034 = load i32, i32* %3033, align 4, !noalias !617
  %3035 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3032
  store i32 %3034, i32* %3035, align 4, !alias.scope !617
  %3036 = add nuw nsw i32 %3031, 1, !spirv.Decorations !620
  %3037 = icmp eq i32 %3031, 0
  br i1 %3037, label %3030, label %3038, !llvm.loop !622

3038:                                             ; preds = %3030
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3028)
  %3039 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3027)
  %3040 = shl i64 %3039, 32
  %3041 = ashr exact i64 %3040, 32
  %3042 = mul nsw i64 %3041, %const_reg_qword3, !spirv.Decorations !610
  %3043 = ashr i64 %3039, 32
  %3044 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3042, i32 0
  %3045 = getelementptr i16, i16 addrspace(4)* %3044, i64 %3043
  %3046 = addrspacecast i16 addrspace(4)* %3045 to i16 addrspace(1)*
  %3047 = load i16, i16 addrspace(1)* %3046, align 2
  %3048 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3048)
  %3049 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3049)
  %3050 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3050, align 4, !noalias !624
  store i32 %2915, i32* %57, align 4, !noalias !624
  br label %3051

3051:                                             ; preds = %3051, %3038
  %3052 = phi i32 [ 0, %3038 ], [ %3057, %3051 ]
  %3053 = zext i32 %3052 to i64
  %3054 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3053
  %3055 = load i32, i32* %3054, align 4, !noalias !624
  %3056 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3053
  store i32 %3055, i32* %3056, align 4, !alias.scope !624
  %3057 = add nuw nsw i32 %3052, 1, !spirv.Decorations !620
  %3058 = icmp eq i32 %3052, 0
  br i1 %3058, label %3051, label %3059, !llvm.loop !627

3059:                                             ; preds = %3051
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3049)
  %3060 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3048)
  %3061 = ashr i64 %3060, 32
  %3062 = mul nsw i64 %3061, %const_reg_qword5, !spirv.Decorations !610
  %3063 = shl i64 %3060, 32
  %3064 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3062
  %3065 = ashr exact i64 %3063, 31
  %3066 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3064 to i8 addrspace(4)*
  %3067 = getelementptr i8, i8 addrspace(4)* %3066, i64 %3065
  %3068 = bitcast i8 addrspace(4)* %3067 to i16 addrspace(4)*
  %3069 = addrspacecast i16 addrspace(4)* %3068 to i16 addrspace(1)*
  %3070 = load i16, i16 addrspace(1)* %3069, align 2
  %3071 = zext i16 %3047 to i32
  %3072 = shl nuw i32 %3071, 16, !spirv.Decorations !628
  %3073 = bitcast i32 %3072 to float
  %3074 = zext i16 %3070 to i32
  %3075 = shl nuw i32 %3074, 16, !spirv.Decorations !628
  %3076 = bitcast i32 %3075 to float
  %3077 = fmul reassoc nsz arcp contract float %3073, %3076, !spirv.Decorations !612
  %3078 = fadd reassoc nsz arcp contract float %3077, %.sroa.182.1, !spirv.Decorations !612
  br label %._crit_edge.2.13

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13, %3059
  %.sroa.182.2 = phi float [ %3078, %3059 ], [ %.sroa.182.1, %._crit_edge.1.13 ]
  %3079 = and i1 %244, %2916
  br i1 %3079, label %3080, label %.preheader.13

3080:                                             ; preds = %._crit_edge.2.13
  %3081 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3081)
  %3082 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3082)
  %3083 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %3083, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3084

3084:                                             ; preds = %3084, %3080
  %3085 = phi i32 [ 0, %3080 ], [ %3090, %3084 ]
  %3086 = zext i32 %3085 to i64
  %3087 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3086
  %3088 = load i32, i32* %3087, align 4, !noalias !617
  %3089 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3086
  store i32 %3088, i32* %3089, align 4, !alias.scope !617
  %3090 = add nuw nsw i32 %3085, 1, !spirv.Decorations !620
  %3091 = icmp eq i32 %3085, 0
  br i1 %3091, label %3084, label %3092, !llvm.loop !622

3092:                                             ; preds = %3084
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3082)
  %3093 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3081)
  %3094 = shl i64 %3093, 32
  %3095 = ashr exact i64 %3094, 32
  %3096 = mul nsw i64 %3095, %const_reg_qword3, !spirv.Decorations !610
  %3097 = ashr i64 %3093, 32
  %3098 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3096, i32 0
  %3099 = getelementptr i16, i16 addrspace(4)* %3098, i64 %3097
  %3100 = addrspacecast i16 addrspace(4)* %3099 to i16 addrspace(1)*
  %3101 = load i16, i16 addrspace(1)* %3100, align 2
  %3102 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3102)
  %3103 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3103)
  %3104 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3104, align 4, !noalias !624
  store i32 %2915, i32* %57, align 4, !noalias !624
  br label %3105

3105:                                             ; preds = %3105, %3092
  %3106 = phi i32 [ 0, %3092 ], [ %3111, %3105 ]
  %3107 = zext i32 %3106 to i64
  %3108 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3107
  %3109 = load i32, i32* %3108, align 4, !noalias !624
  %3110 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3107
  store i32 %3109, i32* %3110, align 4, !alias.scope !624
  %3111 = add nuw nsw i32 %3106, 1, !spirv.Decorations !620
  %3112 = icmp eq i32 %3106, 0
  br i1 %3112, label %3105, label %3113, !llvm.loop !627

3113:                                             ; preds = %3105
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3103)
  %3114 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3102)
  %3115 = ashr i64 %3114, 32
  %3116 = mul nsw i64 %3115, %const_reg_qword5, !spirv.Decorations !610
  %3117 = shl i64 %3114, 32
  %3118 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3116
  %3119 = ashr exact i64 %3117, 31
  %3120 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3118 to i8 addrspace(4)*
  %3121 = getelementptr i8, i8 addrspace(4)* %3120, i64 %3119
  %3122 = bitcast i8 addrspace(4)* %3121 to i16 addrspace(4)*
  %3123 = addrspacecast i16 addrspace(4)* %3122 to i16 addrspace(1)*
  %3124 = load i16, i16 addrspace(1)* %3123, align 2
  %3125 = zext i16 %3101 to i32
  %3126 = shl nuw i32 %3125, 16, !spirv.Decorations !628
  %3127 = bitcast i32 %3126 to float
  %3128 = zext i16 %3124 to i32
  %3129 = shl nuw i32 %3128, 16, !spirv.Decorations !628
  %3130 = bitcast i32 %3129 to float
  %3131 = fmul reassoc nsz arcp contract float %3127, %3130, !spirv.Decorations !612
  %3132 = fadd reassoc nsz arcp contract float %3131, %.sroa.246.1, !spirv.Decorations !612
  br label %.preheader.13

.preheader.13:                                    ; preds = %._crit_edge.2.13, %3113
  %.sroa.246.2 = phi float [ %3132, %3113 ], [ %.sroa.246.1, %._crit_edge.2.13 ]
  %3133 = or i32 %41, 14
  %3134 = icmp slt i32 %3133, %const_reg_dword1
  %3135 = and i1 %76, %3134
  br i1 %3135, label %3136, label %._crit_edge.14

3136:                                             ; preds = %.preheader.13
  %3137 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3137)
  %3138 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3138)
  %3139 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %3139, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3140

3140:                                             ; preds = %3140, %3136
  %3141 = phi i32 [ 0, %3136 ], [ %3146, %3140 ]
  %3142 = zext i32 %3141 to i64
  %3143 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3142
  %3144 = load i32, i32* %3143, align 4, !noalias !617
  %3145 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3142
  store i32 %3144, i32* %3145, align 4, !alias.scope !617
  %3146 = add nuw nsw i32 %3141, 1, !spirv.Decorations !620
  %3147 = icmp eq i32 %3141, 0
  br i1 %3147, label %3140, label %3148, !llvm.loop !622

3148:                                             ; preds = %3140
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3138)
  %3149 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3137)
  %3150 = shl i64 %3149, 32
  %3151 = ashr exact i64 %3150, 32
  %3152 = mul nsw i64 %3151, %const_reg_qword3, !spirv.Decorations !610
  %3153 = ashr i64 %3149, 32
  %3154 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3152, i32 0
  %3155 = getelementptr i16, i16 addrspace(4)* %3154, i64 %3153
  %3156 = addrspacecast i16 addrspace(4)* %3155 to i16 addrspace(1)*
  %3157 = load i16, i16 addrspace(1)* %3156, align 2
  %3158 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3158)
  %3159 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3159)
  %3160 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3160, align 4, !noalias !624
  store i32 %3133, i32* %57, align 4, !noalias !624
  br label %3161

3161:                                             ; preds = %3161, %3148
  %3162 = phi i32 [ 0, %3148 ], [ %3167, %3161 ]
  %3163 = zext i32 %3162 to i64
  %3164 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3163
  %3165 = load i32, i32* %3164, align 4, !noalias !624
  %3166 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3163
  store i32 %3165, i32* %3166, align 4, !alias.scope !624
  %3167 = add nuw nsw i32 %3162, 1, !spirv.Decorations !620
  %3168 = icmp eq i32 %3162, 0
  br i1 %3168, label %3161, label %3169, !llvm.loop !627

3169:                                             ; preds = %3161
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3159)
  %3170 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3158)
  %3171 = ashr i64 %3170, 32
  %3172 = mul nsw i64 %3171, %const_reg_qword5, !spirv.Decorations !610
  %3173 = shl i64 %3170, 32
  %3174 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3172
  %3175 = ashr exact i64 %3173, 31
  %3176 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3174 to i8 addrspace(4)*
  %3177 = getelementptr i8, i8 addrspace(4)* %3176, i64 %3175
  %3178 = bitcast i8 addrspace(4)* %3177 to i16 addrspace(4)*
  %3179 = addrspacecast i16 addrspace(4)* %3178 to i16 addrspace(1)*
  %3180 = load i16, i16 addrspace(1)* %3179, align 2
  %3181 = zext i16 %3157 to i32
  %3182 = shl nuw i32 %3181, 16, !spirv.Decorations !628
  %3183 = bitcast i32 %3182 to float
  %3184 = zext i16 %3180 to i32
  %3185 = shl nuw i32 %3184, 16, !spirv.Decorations !628
  %3186 = bitcast i32 %3185 to float
  %3187 = fmul reassoc nsz arcp contract float %3183, %3186, !spirv.Decorations !612
  %3188 = fadd reassoc nsz arcp contract float %3187, %.sroa.58.1, !spirv.Decorations !612
  br label %._crit_edge.14

._crit_edge.14:                                   ; preds = %.preheader.13, %3169
  %.sroa.58.2 = phi float [ %3188, %3169 ], [ %.sroa.58.1, %.preheader.13 ]
  %3189 = and i1 %132, %3134
  br i1 %3189, label %3190, label %._crit_edge.1.14

3190:                                             ; preds = %._crit_edge.14
  %3191 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3191)
  %3192 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3192)
  %3193 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %3193, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3194

3194:                                             ; preds = %3194, %3190
  %3195 = phi i32 [ 0, %3190 ], [ %3200, %3194 ]
  %3196 = zext i32 %3195 to i64
  %3197 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3196
  %3198 = load i32, i32* %3197, align 4, !noalias !617
  %3199 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3196
  store i32 %3198, i32* %3199, align 4, !alias.scope !617
  %3200 = add nuw nsw i32 %3195, 1, !spirv.Decorations !620
  %3201 = icmp eq i32 %3195, 0
  br i1 %3201, label %3194, label %3202, !llvm.loop !622

3202:                                             ; preds = %3194
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3192)
  %3203 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3191)
  %3204 = shl i64 %3203, 32
  %3205 = ashr exact i64 %3204, 32
  %3206 = mul nsw i64 %3205, %const_reg_qword3, !spirv.Decorations !610
  %3207 = ashr i64 %3203, 32
  %3208 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3206, i32 0
  %3209 = getelementptr i16, i16 addrspace(4)* %3208, i64 %3207
  %3210 = addrspacecast i16 addrspace(4)* %3209 to i16 addrspace(1)*
  %3211 = load i16, i16 addrspace(1)* %3210, align 2
  %3212 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3212)
  %3213 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3213)
  %3214 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3214, align 4, !noalias !624
  store i32 %3133, i32* %57, align 4, !noalias !624
  br label %3215

3215:                                             ; preds = %3215, %3202
  %3216 = phi i32 [ 0, %3202 ], [ %3221, %3215 ]
  %3217 = zext i32 %3216 to i64
  %3218 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3217
  %3219 = load i32, i32* %3218, align 4, !noalias !624
  %3220 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3217
  store i32 %3219, i32* %3220, align 4, !alias.scope !624
  %3221 = add nuw nsw i32 %3216, 1, !spirv.Decorations !620
  %3222 = icmp eq i32 %3216, 0
  br i1 %3222, label %3215, label %3223, !llvm.loop !627

3223:                                             ; preds = %3215
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3213)
  %3224 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3212)
  %3225 = ashr i64 %3224, 32
  %3226 = mul nsw i64 %3225, %const_reg_qword5, !spirv.Decorations !610
  %3227 = shl i64 %3224, 32
  %3228 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3226
  %3229 = ashr exact i64 %3227, 31
  %3230 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3228 to i8 addrspace(4)*
  %3231 = getelementptr i8, i8 addrspace(4)* %3230, i64 %3229
  %3232 = bitcast i8 addrspace(4)* %3231 to i16 addrspace(4)*
  %3233 = addrspacecast i16 addrspace(4)* %3232 to i16 addrspace(1)*
  %3234 = load i16, i16 addrspace(1)* %3233, align 2
  %3235 = zext i16 %3211 to i32
  %3236 = shl nuw i32 %3235, 16, !spirv.Decorations !628
  %3237 = bitcast i32 %3236 to float
  %3238 = zext i16 %3234 to i32
  %3239 = shl nuw i32 %3238, 16, !spirv.Decorations !628
  %3240 = bitcast i32 %3239 to float
  %3241 = fmul reassoc nsz arcp contract float %3237, %3240, !spirv.Decorations !612
  %3242 = fadd reassoc nsz arcp contract float %3241, %.sroa.122.1, !spirv.Decorations !612
  br label %._crit_edge.1.14

._crit_edge.1.14:                                 ; preds = %._crit_edge.14, %3223
  %.sroa.122.2 = phi float [ %3242, %3223 ], [ %.sroa.122.1, %._crit_edge.14 ]
  %3243 = and i1 %188, %3134
  br i1 %3243, label %3244, label %._crit_edge.2.14

3244:                                             ; preds = %._crit_edge.1.14
  %3245 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3245)
  %3246 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3246)
  %3247 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %3247, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3248

3248:                                             ; preds = %3248, %3244
  %3249 = phi i32 [ 0, %3244 ], [ %3254, %3248 ]
  %3250 = zext i32 %3249 to i64
  %3251 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3250
  %3252 = load i32, i32* %3251, align 4, !noalias !617
  %3253 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3250
  store i32 %3252, i32* %3253, align 4, !alias.scope !617
  %3254 = add nuw nsw i32 %3249, 1, !spirv.Decorations !620
  %3255 = icmp eq i32 %3249, 0
  br i1 %3255, label %3248, label %3256, !llvm.loop !622

3256:                                             ; preds = %3248
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3246)
  %3257 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3245)
  %3258 = shl i64 %3257, 32
  %3259 = ashr exact i64 %3258, 32
  %3260 = mul nsw i64 %3259, %const_reg_qword3, !spirv.Decorations !610
  %3261 = ashr i64 %3257, 32
  %3262 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3260, i32 0
  %3263 = getelementptr i16, i16 addrspace(4)* %3262, i64 %3261
  %3264 = addrspacecast i16 addrspace(4)* %3263 to i16 addrspace(1)*
  %3265 = load i16, i16 addrspace(1)* %3264, align 2
  %3266 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3266)
  %3267 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3267)
  %3268 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3268, align 4, !noalias !624
  store i32 %3133, i32* %57, align 4, !noalias !624
  br label %3269

3269:                                             ; preds = %3269, %3256
  %3270 = phi i32 [ 0, %3256 ], [ %3275, %3269 ]
  %3271 = zext i32 %3270 to i64
  %3272 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3271
  %3273 = load i32, i32* %3272, align 4, !noalias !624
  %3274 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3271
  store i32 %3273, i32* %3274, align 4, !alias.scope !624
  %3275 = add nuw nsw i32 %3270, 1, !spirv.Decorations !620
  %3276 = icmp eq i32 %3270, 0
  br i1 %3276, label %3269, label %3277, !llvm.loop !627

3277:                                             ; preds = %3269
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3267)
  %3278 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3266)
  %3279 = ashr i64 %3278, 32
  %3280 = mul nsw i64 %3279, %const_reg_qword5, !spirv.Decorations !610
  %3281 = shl i64 %3278, 32
  %3282 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3280
  %3283 = ashr exact i64 %3281, 31
  %3284 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3282 to i8 addrspace(4)*
  %3285 = getelementptr i8, i8 addrspace(4)* %3284, i64 %3283
  %3286 = bitcast i8 addrspace(4)* %3285 to i16 addrspace(4)*
  %3287 = addrspacecast i16 addrspace(4)* %3286 to i16 addrspace(1)*
  %3288 = load i16, i16 addrspace(1)* %3287, align 2
  %3289 = zext i16 %3265 to i32
  %3290 = shl nuw i32 %3289, 16, !spirv.Decorations !628
  %3291 = bitcast i32 %3290 to float
  %3292 = zext i16 %3288 to i32
  %3293 = shl nuw i32 %3292, 16, !spirv.Decorations !628
  %3294 = bitcast i32 %3293 to float
  %3295 = fmul reassoc nsz arcp contract float %3291, %3294, !spirv.Decorations !612
  %3296 = fadd reassoc nsz arcp contract float %3295, %.sroa.186.1, !spirv.Decorations !612
  br label %._crit_edge.2.14

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14, %3277
  %.sroa.186.2 = phi float [ %3296, %3277 ], [ %.sroa.186.1, %._crit_edge.1.14 ]
  %3297 = and i1 %244, %3134
  br i1 %3297, label %3298, label %.preheader.14

3298:                                             ; preds = %._crit_edge.2.14
  %3299 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3299)
  %3300 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3300)
  %3301 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %3301, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3302

3302:                                             ; preds = %3302, %3298
  %3303 = phi i32 [ 0, %3298 ], [ %3308, %3302 ]
  %3304 = zext i32 %3303 to i64
  %3305 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3304
  %3306 = load i32, i32* %3305, align 4, !noalias !617
  %3307 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3304
  store i32 %3306, i32* %3307, align 4, !alias.scope !617
  %3308 = add nuw nsw i32 %3303, 1, !spirv.Decorations !620
  %3309 = icmp eq i32 %3303, 0
  br i1 %3309, label %3302, label %3310, !llvm.loop !622

3310:                                             ; preds = %3302
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3300)
  %3311 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3299)
  %3312 = shl i64 %3311, 32
  %3313 = ashr exact i64 %3312, 32
  %3314 = mul nsw i64 %3313, %const_reg_qword3, !spirv.Decorations !610
  %3315 = ashr i64 %3311, 32
  %3316 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3314, i32 0
  %3317 = getelementptr i16, i16 addrspace(4)* %3316, i64 %3315
  %3318 = addrspacecast i16 addrspace(4)* %3317 to i16 addrspace(1)*
  %3319 = load i16, i16 addrspace(1)* %3318, align 2
  %3320 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3320)
  %3321 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3321)
  %3322 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3322, align 4, !noalias !624
  store i32 %3133, i32* %57, align 4, !noalias !624
  br label %3323

3323:                                             ; preds = %3323, %3310
  %3324 = phi i32 [ 0, %3310 ], [ %3329, %3323 ]
  %3325 = zext i32 %3324 to i64
  %3326 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3325
  %3327 = load i32, i32* %3326, align 4, !noalias !624
  %3328 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3325
  store i32 %3327, i32* %3328, align 4, !alias.scope !624
  %3329 = add nuw nsw i32 %3324, 1, !spirv.Decorations !620
  %3330 = icmp eq i32 %3324, 0
  br i1 %3330, label %3323, label %3331, !llvm.loop !627

3331:                                             ; preds = %3323
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3321)
  %3332 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3320)
  %3333 = ashr i64 %3332, 32
  %3334 = mul nsw i64 %3333, %const_reg_qword5, !spirv.Decorations !610
  %3335 = shl i64 %3332, 32
  %3336 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3334
  %3337 = ashr exact i64 %3335, 31
  %3338 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3336 to i8 addrspace(4)*
  %3339 = getelementptr i8, i8 addrspace(4)* %3338, i64 %3337
  %3340 = bitcast i8 addrspace(4)* %3339 to i16 addrspace(4)*
  %3341 = addrspacecast i16 addrspace(4)* %3340 to i16 addrspace(1)*
  %3342 = load i16, i16 addrspace(1)* %3341, align 2
  %3343 = zext i16 %3319 to i32
  %3344 = shl nuw i32 %3343, 16, !spirv.Decorations !628
  %3345 = bitcast i32 %3344 to float
  %3346 = zext i16 %3342 to i32
  %3347 = shl nuw i32 %3346, 16, !spirv.Decorations !628
  %3348 = bitcast i32 %3347 to float
  %3349 = fmul reassoc nsz arcp contract float %3345, %3348, !spirv.Decorations !612
  %3350 = fadd reassoc nsz arcp contract float %3349, %.sroa.250.1, !spirv.Decorations !612
  br label %.preheader.14

.preheader.14:                                    ; preds = %._crit_edge.2.14, %3331
  %.sroa.250.2 = phi float [ %3350, %3331 ], [ %.sroa.250.1, %._crit_edge.2.14 ]
  %3351 = or i32 %41, 15
  %3352 = icmp slt i32 %3351, %const_reg_dword1
  %3353 = and i1 %76, %3352
  br i1 %3353, label %3354, label %._crit_edge.15

3354:                                             ; preds = %.preheader.14
  %3355 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3355)
  %3356 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3356)
  %3357 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %3357, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3358

3358:                                             ; preds = %3358, %3354
  %3359 = phi i32 [ 0, %3354 ], [ %3364, %3358 ]
  %3360 = zext i32 %3359 to i64
  %3361 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3360
  %3362 = load i32, i32* %3361, align 4, !noalias !617
  %3363 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3360
  store i32 %3362, i32* %3363, align 4, !alias.scope !617
  %3364 = add nuw nsw i32 %3359, 1, !spirv.Decorations !620
  %3365 = icmp eq i32 %3359, 0
  br i1 %3365, label %3358, label %3366, !llvm.loop !622

3366:                                             ; preds = %3358
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3356)
  %3367 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3355)
  %3368 = shl i64 %3367, 32
  %3369 = ashr exact i64 %3368, 32
  %3370 = mul nsw i64 %3369, %const_reg_qword3, !spirv.Decorations !610
  %3371 = ashr i64 %3367, 32
  %3372 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3370, i32 0
  %3373 = getelementptr i16, i16 addrspace(4)* %3372, i64 %3371
  %3374 = addrspacecast i16 addrspace(4)* %3373 to i16 addrspace(1)*
  %3375 = load i16, i16 addrspace(1)* %3374, align 2
  %3376 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3376)
  %3377 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3377)
  %3378 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3378, align 4, !noalias !624
  store i32 %3351, i32* %57, align 4, !noalias !624
  br label %3379

3379:                                             ; preds = %3379, %3366
  %3380 = phi i32 [ 0, %3366 ], [ %3385, %3379 ]
  %3381 = zext i32 %3380 to i64
  %3382 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3381
  %3383 = load i32, i32* %3382, align 4, !noalias !624
  %3384 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3381
  store i32 %3383, i32* %3384, align 4, !alias.scope !624
  %3385 = add nuw nsw i32 %3380, 1, !spirv.Decorations !620
  %3386 = icmp eq i32 %3380, 0
  br i1 %3386, label %3379, label %3387, !llvm.loop !627

3387:                                             ; preds = %3379
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3377)
  %3388 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3376)
  %3389 = ashr i64 %3388, 32
  %3390 = mul nsw i64 %3389, %const_reg_qword5, !spirv.Decorations !610
  %3391 = shl i64 %3388, 32
  %3392 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3390
  %3393 = ashr exact i64 %3391, 31
  %3394 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3392 to i8 addrspace(4)*
  %3395 = getelementptr i8, i8 addrspace(4)* %3394, i64 %3393
  %3396 = bitcast i8 addrspace(4)* %3395 to i16 addrspace(4)*
  %3397 = addrspacecast i16 addrspace(4)* %3396 to i16 addrspace(1)*
  %3398 = load i16, i16 addrspace(1)* %3397, align 2
  %3399 = zext i16 %3375 to i32
  %3400 = shl nuw i32 %3399, 16, !spirv.Decorations !628
  %3401 = bitcast i32 %3400 to float
  %3402 = zext i16 %3398 to i32
  %3403 = shl nuw i32 %3402, 16, !spirv.Decorations !628
  %3404 = bitcast i32 %3403 to float
  %3405 = fmul reassoc nsz arcp contract float %3401, %3404, !spirv.Decorations !612
  %3406 = fadd reassoc nsz arcp contract float %3405, %.sroa.62.1, !spirv.Decorations !612
  br label %._crit_edge.15

._crit_edge.15:                                   ; preds = %.preheader.14, %3387
  %.sroa.62.2 = phi float [ %3406, %3387 ], [ %.sroa.62.1, %.preheader.14 ]
  %3407 = and i1 %132, %3352
  br i1 %3407, label %3408, label %._crit_edge.1.15

3408:                                             ; preds = %._crit_edge.15
  %3409 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3409)
  %3410 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3410)
  %3411 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %131, i32* %3411, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3412

3412:                                             ; preds = %3412, %3408
  %3413 = phi i32 [ 0, %3408 ], [ %3418, %3412 ]
  %3414 = zext i32 %3413 to i64
  %3415 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3414
  %3416 = load i32, i32* %3415, align 4, !noalias !617
  %3417 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3414
  store i32 %3416, i32* %3417, align 4, !alias.scope !617
  %3418 = add nuw nsw i32 %3413, 1, !spirv.Decorations !620
  %3419 = icmp eq i32 %3413, 0
  br i1 %3419, label %3412, label %3420, !llvm.loop !622

3420:                                             ; preds = %3412
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3410)
  %3421 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3409)
  %3422 = shl i64 %3421, 32
  %3423 = ashr exact i64 %3422, 32
  %3424 = mul nsw i64 %3423, %const_reg_qword3, !spirv.Decorations !610
  %3425 = ashr i64 %3421, 32
  %3426 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3424, i32 0
  %3427 = getelementptr i16, i16 addrspace(4)* %3426, i64 %3425
  %3428 = addrspacecast i16 addrspace(4)* %3427 to i16 addrspace(1)*
  %3429 = load i16, i16 addrspace(1)* %3428, align 2
  %3430 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3430)
  %3431 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3431)
  %3432 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3432, align 4, !noalias !624
  store i32 %3351, i32* %57, align 4, !noalias !624
  br label %3433

3433:                                             ; preds = %3433, %3420
  %3434 = phi i32 [ 0, %3420 ], [ %3439, %3433 ]
  %3435 = zext i32 %3434 to i64
  %3436 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3435
  %3437 = load i32, i32* %3436, align 4, !noalias !624
  %3438 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3435
  store i32 %3437, i32* %3438, align 4, !alias.scope !624
  %3439 = add nuw nsw i32 %3434, 1, !spirv.Decorations !620
  %3440 = icmp eq i32 %3434, 0
  br i1 %3440, label %3433, label %3441, !llvm.loop !627

3441:                                             ; preds = %3433
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3431)
  %3442 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3430)
  %3443 = ashr i64 %3442, 32
  %3444 = mul nsw i64 %3443, %const_reg_qword5, !spirv.Decorations !610
  %3445 = shl i64 %3442, 32
  %3446 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3444
  %3447 = ashr exact i64 %3445, 31
  %3448 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3446 to i8 addrspace(4)*
  %3449 = getelementptr i8, i8 addrspace(4)* %3448, i64 %3447
  %3450 = bitcast i8 addrspace(4)* %3449 to i16 addrspace(4)*
  %3451 = addrspacecast i16 addrspace(4)* %3450 to i16 addrspace(1)*
  %3452 = load i16, i16 addrspace(1)* %3451, align 2
  %3453 = zext i16 %3429 to i32
  %3454 = shl nuw i32 %3453, 16, !spirv.Decorations !628
  %3455 = bitcast i32 %3454 to float
  %3456 = zext i16 %3452 to i32
  %3457 = shl nuw i32 %3456, 16, !spirv.Decorations !628
  %3458 = bitcast i32 %3457 to float
  %3459 = fmul reassoc nsz arcp contract float %3455, %3458, !spirv.Decorations !612
  %3460 = fadd reassoc nsz arcp contract float %3459, %.sroa.126.1, !spirv.Decorations !612
  br label %._crit_edge.1.15

._crit_edge.1.15:                                 ; preds = %._crit_edge.15, %3441
  %.sroa.126.2 = phi float [ %3460, %3441 ], [ %.sroa.126.1, %._crit_edge.15 ]
  %3461 = and i1 %188, %3352
  br i1 %3461, label %3462, label %._crit_edge.2.15

3462:                                             ; preds = %._crit_edge.1.15
  %3463 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3463)
  %3464 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3464)
  %3465 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %187, i32* %3465, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3466

3466:                                             ; preds = %3466, %3462
  %3467 = phi i32 [ 0, %3462 ], [ %3472, %3466 ]
  %3468 = zext i32 %3467 to i64
  %3469 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3468
  %3470 = load i32, i32* %3469, align 4, !noalias !617
  %3471 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3468
  store i32 %3470, i32* %3471, align 4, !alias.scope !617
  %3472 = add nuw nsw i32 %3467, 1, !spirv.Decorations !620
  %3473 = icmp eq i32 %3467, 0
  br i1 %3473, label %3466, label %3474, !llvm.loop !622

3474:                                             ; preds = %3466
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3464)
  %3475 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3463)
  %3476 = shl i64 %3475, 32
  %3477 = ashr exact i64 %3476, 32
  %3478 = mul nsw i64 %3477, %const_reg_qword3, !spirv.Decorations !610
  %3479 = ashr i64 %3475, 32
  %3480 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3478, i32 0
  %3481 = getelementptr i16, i16 addrspace(4)* %3480, i64 %3479
  %3482 = addrspacecast i16 addrspace(4)* %3481 to i16 addrspace(1)*
  %3483 = load i16, i16 addrspace(1)* %3482, align 2
  %3484 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3484)
  %3485 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3485)
  %3486 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3486, align 4, !noalias !624
  store i32 %3351, i32* %57, align 4, !noalias !624
  br label %3487

3487:                                             ; preds = %3487, %3474
  %3488 = phi i32 [ 0, %3474 ], [ %3493, %3487 ]
  %3489 = zext i32 %3488 to i64
  %3490 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3489
  %3491 = load i32, i32* %3490, align 4, !noalias !624
  %3492 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3489
  store i32 %3491, i32* %3492, align 4, !alias.scope !624
  %3493 = add nuw nsw i32 %3488, 1, !spirv.Decorations !620
  %3494 = icmp eq i32 %3488, 0
  br i1 %3494, label %3487, label %3495, !llvm.loop !627

3495:                                             ; preds = %3487
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3485)
  %3496 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3484)
  %3497 = ashr i64 %3496, 32
  %3498 = mul nsw i64 %3497, %const_reg_qword5, !spirv.Decorations !610
  %3499 = shl i64 %3496, 32
  %3500 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3498
  %3501 = ashr exact i64 %3499, 31
  %3502 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3500 to i8 addrspace(4)*
  %3503 = getelementptr i8, i8 addrspace(4)* %3502, i64 %3501
  %3504 = bitcast i8 addrspace(4)* %3503 to i16 addrspace(4)*
  %3505 = addrspacecast i16 addrspace(4)* %3504 to i16 addrspace(1)*
  %3506 = load i16, i16 addrspace(1)* %3505, align 2
  %3507 = zext i16 %3483 to i32
  %3508 = shl nuw i32 %3507, 16, !spirv.Decorations !628
  %3509 = bitcast i32 %3508 to float
  %3510 = zext i16 %3506 to i32
  %3511 = shl nuw i32 %3510, 16, !spirv.Decorations !628
  %3512 = bitcast i32 %3511 to float
  %3513 = fmul reassoc nsz arcp contract float %3509, %3512, !spirv.Decorations !612
  %3514 = fadd reassoc nsz arcp contract float %3513, %.sroa.190.1, !spirv.Decorations !612
  br label %._crit_edge.2.15

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15, %3495
  %.sroa.190.2 = phi float [ %3514, %3495 ], [ %.sroa.190.1, %._crit_edge.1.15 ]
  %3515 = and i1 %244, %3352
  br i1 %3515, label %3516, label %.preheader.15

3516:                                             ; preds = %._crit_edge.2.15
  %3517 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3517)
  %3518 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3518)
  %3519 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %243, i32* %3519, align 4, !noalias !617
  store i32 %74, i32* %54, align 4, !noalias !617
  br label %3520

3520:                                             ; preds = %3520, %3516
  %3521 = phi i32 [ 0, %3516 ], [ %3526, %3520 ]
  %3522 = zext i32 %3521 to i64
  %3523 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3522
  %3524 = load i32, i32* %3523, align 4, !noalias !617
  %3525 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3522
  store i32 %3524, i32* %3525, align 4, !alias.scope !617
  %3526 = add nuw nsw i32 %3521, 1, !spirv.Decorations !620
  %3527 = icmp eq i32 %3521, 0
  br i1 %3527, label %3520, label %3528, !llvm.loop !622

3528:                                             ; preds = %3520
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3518)
  %3529 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3517)
  %3530 = shl i64 %3529, 32
  %3531 = ashr exact i64 %3530, 32
  %3532 = mul nsw i64 %3531, %const_reg_qword3, !spirv.Decorations !610
  %3533 = ashr i64 %3529, 32
  %3534 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3532, i32 0
  %3535 = getelementptr i16, i16 addrspace(4)* %3534, i64 %3533
  %3536 = addrspacecast i16 addrspace(4)* %3535 to i16 addrspace(1)*
  %3537 = load i16, i16 addrspace(1)* %3536, align 2
  %3538 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3538)
  %3539 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3539)
  %3540 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3540, align 4, !noalias !624
  store i32 %3351, i32* %57, align 4, !noalias !624
  br label %3541

3541:                                             ; preds = %3541, %3528
  %3542 = phi i32 [ 0, %3528 ], [ %3547, %3541 ]
  %3543 = zext i32 %3542 to i64
  %3544 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3543
  %3545 = load i32, i32* %3544, align 4, !noalias !624
  %3546 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3543
  store i32 %3545, i32* %3546, align 4, !alias.scope !624
  %3547 = add nuw nsw i32 %3542, 1, !spirv.Decorations !620
  %3548 = icmp eq i32 %3542, 0
  br i1 %3548, label %3541, label %3549, !llvm.loop !627

3549:                                             ; preds = %3541
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3539)
  %3550 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3538)
  %3551 = ashr i64 %3550, 32
  %3552 = mul nsw i64 %3551, %const_reg_qword5, !spirv.Decorations !610
  %3553 = shl i64 %3550, 32
  %3554 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3552
  %3555 = ashr exact i64 %3553, 31
  %3556 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %3554 to i8 addrspace(4)*
  %3557 = getelementptr i8, i8 addrspace(4)* %3556, i64 %3555
  %3558 = bitcast i8 addrspace(4)* %3557 to i16 addrspace(4)*
  %3559 = addrspacecast i16 addrspace(4)* %3558 to i16 addrspace(1)*
  %3560 = load i16, i16 addrspace(1)* %3559, align 2
  %3561 = zext i16 %3537 to i32
  %3562 = shl nuw i32 %3561, 16, !spirv.Decorations !628
  %3563 = bitcast i32 %3562 to float
  %3564 = zext i16 %3560 to i32
  %3565 = shl nuw i32 %3564, 16, !spirv.Decorations !628
  %3566 = bitcast i32 %3565 to float
  %3567 = fmul reassoc nsz arcp contract float %3563, %3566, !spirv.Decorations !612
  %3568 = fadd reassoc nsz arcp contract float %3567, %.sroa.254.1, !spirv.Decorations !612
  br label %.preheader.15

.preheader.15:                                    ; preds = %._crit_edge.2.15, %3549
  %.sroa.254.2 = phi float [ %3568, %3549 ], [ %.sroa.254.1, %._crit_edge.2.15 ]
  %3569 = add nuw nsw i32 %74, 1, !spirv.Decorations !620
  %3570 = icmp slt i32 %3569, %const_reg_dword2
  br i1 %3570, label %.preheader.preheader, label %.preheader1.preheader, !llvm.loop !629

3571:                                             ; preds = %.preheader1.preheader, %3571
  %3572 = phi i32 [ 0, %.preheader1.preheader ], [ %3577, %3571 ]
  %3573 = zext i32 %3572 to i64
  %3574 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3573
  %3575 = load i32, i32* %3574, align 4, !noalias !614
  %3576 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3573
  store i32 %3575, i32* %3576, align 4, !alias.scope !614
  %3577 = add nuw nsw i32 %3572, 1, !spirv.Decorations !620
  %3578 = icmp eq i32 %3572, 0
  br i1 %3578, label %3571, label %3579, !llvm.loop !630

3579:                                             ; preds = %3571
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3580 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3581 = icmp slt i32 %35, %const_reg_dword
  %3582 = and i1 %3581, %70
  br i1 %3582, label %3583, label %._crit_edge70

3583:                                             ; preds = %3579
  %3584 = shl i64 %3580, 32
  %3585 = ashr exact i64 %3584, 32
  %3586 = ashr i64 %3580, 32
  %3587 = mul nsw i64 %3585, %const_reg_qword9, !spirv.Decorations !610
  %3588 = add nsw i64 %3587, %3586, !spirv.Decorations !610
  %3589 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !612
  br i1 %48, label %3590, label %3600

3590:                                             ; preds = %3583
  %3591 = mul nsw i64 %3585, %const_reg_qword7, !spirv.Decorations !610
  %3592 = getelementptr float, float addrspace(4)* %66, i64 %3591
  %3593 = getelementptr float, float addrspace(4)* %3592, i64 %3586
  %3594 = addrspacecast float addrspace(4)* %3593 to float addrspace(1)*
  %3595 = load float, float addrspace(1)* %3594, align 4
  %3596 = fmul reassoc nsz arcp contract float %3595, %4, !spirv.Decorations !612
  %3597 = fadd reassoc nsz arcp contract float %3589, %3596, !spirv.Decorations !612
  %3598 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3588
  %3599 = addrspacecast float addrspace(4)* %3598 to float addrspace(1)*
  store float %3597, float addrspace(1)* %3599, align 4
  br label %._crit_edge70

3600:                                             ; preds = %3583
  %3601 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3588
  %3602 = addrspacecast float addrspace(4)* %3601 to float addrspace(1)*
  store float %3589, float addrspace(1)* %3602, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %3579, %3600, %3590
  %3603 = or i32 %35, 1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %41, i32* %60, align 4, !noalias !614
  br label %3604

3604:                                             ; preds = %3604, %._crit_edge70
  %3605 = phi i32 [ 0, %._crit_edge70 ], [ %3610, %3604 ]
  %3606 = zext i32 %3605 to i64
  %3607 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3606
  %3608 = load i32, i32* %3607, align 4, !noalias !614
  %3609 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3606
  store i32 %3608, i32* %3609, align 4, !alias.scope !614
  %3610 = add nuw nsw i32 %3605, 1, !spirv.Decorations !620
  %3611 = icmp eq i32 %3605, 0
  br i1 %3611, label %3604, label %3612, !llvm.loop !630

3612:                                             ; preds = %3604
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3613 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3614 = icmp slt i32 %3603, %const_reg_dword
  %3615 = and i1 %3614, %70
  br i1 %3615, label %3616, label %._crit_edge70.1

3616:                                             ; preds = %3612
  %3617 = shl i64 %3613, 32
  %3618 = ashr exact i64 %3617, 32
  %3619 = ashr i64 %3613, 32
  %3620 = mul nsw i64 %3618, %const_reg_qword9, !spirv.Decorations !610
  %3621 = add nsw i64 %3620, %3619, !spirv.Decorations !610
  %3622 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !612
  br i1 %48, label %3626, label %3623

3623:                                             ; preds = %3616
  %3624 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3621
  %3625 = addrspacecast float addrspace(4)* %3624 to float addrspace(1)*
  store float %3622, float addrspace(1)* %3625, align 4
  br label %._crit_edge70.1

3626:                                             ; preds = %3616
  %3627 = mul nsw i64 %3618, %const_reg_qword7, !spirv.Decorations !610
  %3628 = getelementptr float, float addrspace(4)* %66, i64 %3627
  %3629 = getelementptr float, float addrspace(4)* %3628, i64 %3619
  %3630 = addrspacecast float addrspace(4)* %3629 to float addrspace(1)*
  %3631 = load float, float addrspace(1)* %3630, align 4
  %3632 = fmul reassoc nsz arcp contract float %3631, %4, !spirv.Decorations !612
  %3633 = fadd reassoc nsz arcp contract float %3622, %3632, !spirv.Decorations !612
  %3634 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3621
  %3635 = addrspacecast float addrspace(4)* %3634 to float addrspace(1)*
  store float %3633, float addrspace(1)* %3635, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %3612, %3626, %3623
  %3636 = or i32 %35, 2
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %41, i32* %60, align 4, !noalias !614
  br label %3637

3637:                                             ; preds = %3637, %._crit_edge70.1
  %3638 = phi i32 [ 0, %._crit_edge70.1 ], [ %3643, %3637 ]
  %3639 = zext i32 %3638 to i64
  %3640 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3639
  %3641 = load i32, i32* %3640, align 4, !noalias !614
  %3642 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3639
  store i32 %3641, i32* %3642, align 4, !alias.scope !614
  %3643 = add nuw nsw i32 %3638, 1, !spirv.Decorations !620
  %3644 = icmp eq i32 %3638, 0
  br i1 %3644, label %3637, label %3645, !llvm.loop !630

3645:                                             ; preds = %3637
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3646 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3647 = icmp slt i32 %3636, %const_reg_dword
  %3648 = and i1 %3647, %70
  br i1 %3648, label %3649, label %._crit_edge70.2

3649:                                             ; preds = %3645
  %3650 = shl i64 %3646, 32
  %3651 = ashr exact i64 %3650, 32
  %3652 = ashr i64 %3646, 32
  %3653 = mul nsw i64 %3651, %const_reg_qword9, !spirv.Decorations !610
  %3654 = add nsw i64 %3653, %3652, !spirv.Decorations !610
  %3655 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !612
  br i1 %48, label %3659, label %3656

3656:                                             ; preds = %3649
  %3657 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3654
  %3658 = addrspacecast float addrspace(4)* %3657 to float addrspace(1)*
  store float %3655, float addrspace(1)* %3658, align 4
  br label %._crit_edge70.2

3659:                                             ; preds = %3649
  %3660 = mul nsw i64 %3651, %const_reg_qword7, !spirv.Decorations !610
  %3661 = getelementptr float, float addrspace(4)* %66, i64 %3660
  %3662 = getelementptr float, float addrspace(4)* %3661, i64 %3652
  %3663 = addrspacecast float addrspace(4)* %3662 to float addrspace(1)*
  %3664 = load float, float addrspace(1)* %3663, align 4
  %3665 = fmul reassoc nsz arcp contract float %3664, %4, !spirv.Decorations !612
  %3666 = fadd reassoc nsz arcp contract float %3655, %3665, !spirv.Decorations !612
  %3667 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3654
  %3668 = addrspacecast float addrspace(4)* %3667 to float addrspace(1)*
  store float %3666, float addrspace(1)* %3668, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %3645, %3659, %3656
  %3669 = or i32 %35, 3
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %41, i32* %60, align 4, !noalias !614
  br label %3670

3670:                                             ; preds = %3670, %._crit_edge70.2
  %3671 = phi i32 [ 0, %._crit_edge70.2 ], [ %3676, %3670 ]
  %3672 = zext i32 %3671 to i64
  %3673 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3672
  %3674 = load i32, i32* %3673, align 4, !noalias !614
  %3675 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3672
  store i32 %3674, i32* %3675, align 4, !alias.scope !614
  %3676 = add nuw nsw i32 %3671, 1, !spirv.Decorations !620
  %3677 = icmp eq i32 %3671, 0
  br i1 %3677, label %3670, label %3678, !llvm.loop !630

3678:                                             ; preds = %3670
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3679 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3680 = icmp slt i32 %3669, %const_reg_dword
  %3681 = and i1 %3680, %70
  br i1 %3681, label %3682, label %.preheader1

3682:                                             ; preds = %3678
  %3683 = shl i64 %3679, 32
  %3684 = ashr exact i64 %3683, 32
  %3685 = ashr i64 %3679, 32
  %3686 = mul nsw i64 %3684, %const_reg_qword9, !spirv.Decorations !610
  %3687 = add nsw i64 %3686, %3685, !spirv.Decorations !610
  %3688 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !612
  br i1 %48, label %3692, label %3689

3689:                                             ; preds = %3682
  %3690 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3687
  %3691 = addrspacecast float addrspace(4)* %3690 to float addrspace(1)*
  store float %3688, float addrspace(1)* %3691, align 4
  br label %.preheader1

3692:                                             ; preds = %3682
  %3693 = mul nsw i64 %3684, %const_reg_qword7, !spirv.Decorations !610
  %3694 = getelementptr float, float addrspace(4)* %66, i64 %3693
  %3695 = getelementptr float, float addrspace(4)* %3694, i64 %3685
  %3696 = addrspacecast float addrspace(4)* %3695 to float addrspace(1)*
  %3697 = load float, float addrspace(1)* %3696, align 4
  %3698 = fmul reassoc nsz arcp contract float %3697, %4, !spirv.Decorations !612
  %3699 = fadd reassoc nsz arcp contract float %3688, %3698, !spirv.Decorations !612
  %3700 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3687
  %3701 = addrspacecast float addrspace(4)* %3700 to float addrspace(1)*
  store float %3699, float addrspace(1)* %3701, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %3678, %3692, %3689
  %3702 = or i32 %41, 1
  %3703 = icmp slt i32 %3702, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %3702, i32* %60, align 4, !noalias !614
  br label %3704

3704:                                             ; preds = %3704, %.preheader1
  %3705 = phi i32 [ 0, %.preheader1 ], [ %3710, %3704 ]
  %3706 = zext i32 %3705 to i64
  %3707 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3706
  %3708 = load i32, i32* %3707, align 4, !noalias !614
  %3709 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3706
  store i32 %3708, i32* %3709, align 4, !alias.scope !614
  %3710 = add nuw nsw i32 %3705, 1, !spirv.Decorations !620
  %3711 = icmp eq i32 %3705, 0
  br i1 %3711, label %3704, label %3712, !llvm.loop !630

3712:                                             ; preds = %3704
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3713 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3714 = and i1 %3581, %3703
  br i1 %3714, label %3715, label %._crit_edge70.176

3715:                                             ; preds = %3712
  %3716 = shl i64 %3713, 32
  %3717 = ashr exact i64 %3716, 32
  %3718 = ashr i64 %3713, 32
  %3719 = mul nsw i64 %3717, %const_reg_qword9, !spirv.Decorations !610
  %3720 = add nsw i64 %3719, %3718, !spirv.Decorations !610
  %3721 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !612
  br i1 %48, label %3725, label %3722

3722:                                             ; preds = %3715
  %3723 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3720
  %3724 = addrspacecast float addrspace(4)* %3723 to float addrspace(1)*
  store float %3721, float addrspace(1)* %3724, align 4
  br label %._crit_edge70.176

3725:                                             ; preds = %3715
  %3726 = mul nsw i64 %3717, %const_reg_qword7, !spirv.Decorations !610
  %3727 = getelementptr float, float addrspace(4)* %66, i64 %3726
  %3728 = getelementptr float, float addrspace(4)* %3727, i64 %3718
  %3729 = addrspacecast float addrspace(4)* %3728 to float addrspace(1)*
  %3730 = load float, float addrspace(1)* %3729, align 4
  %3731 = fmul reassoc nsz arcp contract float %3730, %4, !spirv.Decorations !612
  %3732 = fadd reassoc nsz arcp contract float %3721, %3731, !spirv.Decorations !612
  %3733 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3720
  %3734 = addrspacecast float addrspace(4)* %3733 to float addrspace(1)*
  store float %3732, float addrspace(1)* %3734, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %3712, %3725, %3722
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %3702, i32* %60, align 4, !noalias !614
  br label %3735

3735:                                             ; preds = %3735, %._crit_edge70.176
  %3736 = phi i32 [ 0, %._crit_edge70.176 ], [ %3741, %3735 ]
  %3737 = zext i32 %3736 to i64
  %3738 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3737
  %3739 = load i32, i32* %3738, align 4, !noalias !614
  %3740 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3737
  store i32 %3739, i32* %3740, align 4, !alias.scope !614
  %3741 = add nuw nsw i32 %3736, 1, !spirv.Decorations !620
  %3742 = icmp eq i32 %3736, 0
  br i1 %3742, label %3735, label %3743, !llvm.loop !630

3743:                                             ; preds = %3735
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3744 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3745 = and i1 %3614, %3703
  br i1 %3745, label %3746, label %._crit_edge70.1.1

3746:                                             ; preds = %3743
  %3747 = shl i64 %3744, 32
  %3748 = ashr exact i64 %3747, 32
  %3749 = ashr i64 %3744, 32
  %3750 = mul nsw i64 %3748, %const_reg_qword9, !spirv.Decorations !610
  %3751 = add nsw i64 %3750, %3749, !spirv.Decorations !610
  %3752 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !612
  br i1 %48, label %3756, label %3753

3753:                                             ; preds = %3746
  %3754 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3751
  %3755 = addrspacecast float addrspace(4)* %3754 to float addrspace(1)*
  store float %3752, float addrspace(1)* %3755, align 4
  br label %._crit_edge70.1.1

3756:                                             ; preds = %3746
  %3757 = mul nsw i64 %3748, %const_reg_qword7, !spirv.Decorations !610
  %3758 = getelementptr float, float addrspace(4)* %66, i64 %3757
  %3759 = getelementptr float, float addrspace(4)* %3758, i64 %3749
  %3760 = addrspacecast float addrspace(4)* %3759 to float addrspace(1)*
  %3761 = load float, float addrspace(1)* %3760, align 4
  %3762 = fmul reassoc nsz arcp contract float %3761, %4, !spirv.Decorations !612
  %3763 = fadd reassoc nsz arcp contract float %3752, %3762, !spirv.Decorations !612
  %3764 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3751
  %3765 = addrspacecast float addrspace(4)* %3764 to float addrspace(1)*
  store float %3763, float addrspace(1)* %3765, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %3743, %3756, %3753
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %3702, i32* %60, align 4, !noalias !614
  br label %3766

3766:                                             ; preds = %3766, %._crit_edge70.1.1
  %3767 = phi i32 [ 0, %._crit_edge70.1.1 ], [ %3772, %3766 ]
  %3768 = zext i32 %3767 to i64
  %3769 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3768
  %3770 = load i32, i32* %3769, align 4, !noalias !614
  %3771 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3768
  store i32 %3770, i32* %3771, align 4, !alias.scope !614
  %3772 = add nuw nsw i32 %3767, 1, !spirv.Decorations !620
  %3773 = icmp eq i32 %3767, 0
  br i1 %3773, label %3766, label %3774, !llvm.loop !630

3774:                                             ; preds = %3766
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3775 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3776 = and i1 %3647, %3703
  br i1 %3776, label %3777, label %._crit_edge70.2.1

3777:                                             ; preds = %3774
  %3778 = shl i64 %3775, 32
  %3779 = ashr exact i64 %3778, 32
  %3780 = ashr i64 %3775, 32
  %3781 = mul nsw i64 %3779, %const_reg_qword9, !spirv.Decorations !610
  %3782 = add nsw i64 %3781, %3780, !spirv.Decorations !610
  %3783 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !612
  br i1 %48, label %3787, label %3784

3784:                                             ; preds = %3777
  %3785 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3782
  %3786 = addrspacecast float addrspace(4)* %3785 to float addrspace(1)*
  store float %3783, float addrspace(1)* %3786, align 4
  br label %._crit_edge70.2.1

3787:                                             ; preds = %3777
  %3788 = mul nsw i64 %3779, %const_reg_qword7, !spirv.Decorations !610
  %3789 = getelementptr float, float addrspace(4)* %66, i64 %3788
  %3790 = getelementptr float, float addrspace(4)* %3789, i64 %3780
  %3791 = addrspacecast float addrspace(4)* %3790 to float addrspace(1)*
  %3792 = load float, float addrspace(1)* %3791, align 4
  %3793 = fmul reassoc nsz arcp contract float %3792, %4, !spirv.Decorations !612
  %3794 = fadd reassoc nsz arcp contract float %3783, %3793, !spirv.Decorations !612
  %3795 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3782
  %3796 = addrspacecast float addrspace(4)* %3795 to float addrspace(1)*
  store float %3794, float addrspace(1)* %3796, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %3774, %3787, %3784
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %3702, i32* %60, align 4, !noalias !614
  br label %3797

3797:                                             ; preds = %3797, %._crit_edge70.2.1
  %3798 = phi i32 [ 0, %._crit_edge70.2.1 ], [ %3803, %3797 ]
  %3799 = zext i32 %3798 to i64
  %3800 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3799
  %3801 = load i32, i32* %3800, align 4, !noalias !614
  %3802 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3799
  store i32 %3801, i32* %3802, align 4, !alias.scope !614
  %3803 = add nuw nsw i32 %3798, 1, !spirv.Decorations !620
  %3804 = icmp eq i32 %3798, 0
  br i1 %3804, label %3797, label %3805, !llvm.loop !630

3805:                                             ; preds = %3797
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3806 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3807 = and i1 %3680, %3703
  br i1 %3807, label %3808, label %.preheader1.1

3808:                                             ; preds = %3805
  %3809 = shl i64 %3806, 32
  %3810 = ashr exact i64 %3809, 32
  %3811 = ashr i64 %3806, 32
  %3812 = mul nsw i64 %3810, %const_reg_qword9, !spirv.Decorations !610
  %3813 = add nsw i64 %3812, %3811, !spirv.Decorations !610
  %3814 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !612
  br i1 %48, label %3818, label %3815

3815:                                             ; preds = %3808
  %3816 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3813
  %3817 = addrspacecast float addrspace(4)* %3816 to float addrspace(1)*
  store float %3814, float addrspace(1)* %3817, align 4
  br label %.preheader1.1

3818:                                             ; preds = %3808
  %3819 = mul nsw i64 %3810, %const_reg_qword7, !spirv.Decorations !610
  %3820 = getelementptr float, float addrspace(4)* %66, i64 %3819
  %3821 = getelementptr float, float addrspace(4)* %3820, i64 %3811
  %3822 = addrspacecast float addrspace(4)* %3821 to float addrspace(1)*
  %3823 = load float, float addrspace(1)* %3822, align 4
  %3824 = fmul reassoc nsz arcp contract float %3823, %4, !spirv.Decorations !612
  %3825 = fadd reassoc nsz arcp contract float %3814, %3824, !spirv.Decorations !612
  %3826 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3813
  %3827 = addrspacecast float addrspace(4)* %3826 to float addrspace(1)*
  store float %3825, float addrspace(1)* %3827, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %3805, %3818, %3815
  %3828 = or i32 %41, 2
  %3829 = icmp slt i32 %3828, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %3828, i32* %60, align 4, !noalias !614
  br label %3830

3830:                                             ; preds = %3830, %.preheader1.1
  %3831 = phi i32 [ 0, %.preheader1.1 ], [ %3836, %3830 ]
  %3832 = zext i32 %3831 to i64
  %3833 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3832
  %3834 = load i32, i32* %3833, align 4, !noalias !614
  %3835 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3832
  store i32 %3834, i32* %3835, align 4, !alias.scope !614
  %3836 = add nuw nsw i32 %3831, 1, !spirv.Decorations !620
  %3837 = icmp eq i32 %3831, 0
  br i1 %3837, label %3830, label %3838, !llvm.loop !630

3838:                                             ; preds = %3830
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3839 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3840 = and i1 %3581, %3829
  br i1 %3840, label %3841, label %._crit_edge70.277

3841:                                             ; preds = %3838
  %3842 = shl i64 %3839, 32
  %3843 = ashr exact i64 %3842, 32
  %3844 = ashr i64 %3839, 32
  %3845 = mul nsw i64 %3843, %const_reg_qword9, !spirv.Decorations !610
  %3846 = add nsw i64 %3845, %3844, !spirv.Decorations !610
  %3847 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !612
  br i1 %48, label %3851, label %3848

3848:                                             ; preds = %3841
  %3849 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3846
  %3850 = addrspacecast float addrspace(4)* %3849 to float addrspace(1)*
  store float %3847, float addrspace(1)* %3850, align 4
  br label %._crit_edge70.277

3851:                                             ; preds = %3841
  %3852 = mul nsw i64 %3843, %const_reg_qword7, !spirv.Decorations !610
  %3853 = getelementptr float, float addrspace(4)* %66, i64 %3852
  %3854 = getelementptr float, float addrspace(4)* %3853, i64 %3844
  %3855 = addrspacecast float addrspace(4)* %3854 to float addrspace(1)*
  %3856 = load float, float addrspace(1)* %3855, align 4
  %3857 = fmul reassoc nsz arcp contract float %3856, %4, !spirv.Decorations !612
  %3858 = fadd reassoc nsz arcp contract float %3847, %3857, !spirv.Decorations !612
  %3859 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3846
  %3860 = addrspacecast float addrspace(4)* %3859 to float addrspace(1)*
  store float %3858, float addrspace(1)* %3860, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %3838, %3851, %3848
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %3828, i32* %60, align 4, !noalias !614
  br label %3861

3861:                                             ; preds = %3861, %._crit_edge70.277
  %3862 = phi i32 [ 0, %._crit_edge70.277 ], [ %3867, %3861 ]
  %3863 = zext i32 %3862 to i64
  %3864 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3863
  %3865 = load i32, i32* %3864, align 4, !noalias !614
  %3866 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3863
  store i32 %3865, i32* %3866, align 4, !alias.scope !614
  %3867 = add nuw nsw i32 %3862, 1, !spirv.Decorations !620
  %3868 = icmp eq i32 %3862, 0
  br i1 %3868, label %3861, label %3869, !llvm.loop !630

3869:                                             ; preds = %3861
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3870 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3871 = and i1 %3614, %3829
  br i1 %3871, label %3872, label %._crit_edge70.1.2

3872:                                             ; preds = %3869
  %3873 = shl i64 %3870, 32
  %3874 = ashr exact i64 %3873, 32
  %3875 = ashr i64 %3870, 32
  %3876 = mul nsw i64 %3874, %const_reg_qword9, !spirv.Decorations !610
  %3877 = add nsw i64 %3876, %3875, !spirv.Decorations !610
  %3878 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !612
  br i1 %48, label %3882, label %3879

3879:                                             ; preds = %3872
  %3880 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3877
  %3881 = addrspacecast float addrspace(4)* %3880 to float addrspace(1)*
  store float %3878, float addrspace(1)* %3881, align 4
  br label %._crit_edge70.1.2

3882:                                             ; preds = %3872
  %3883 = mul nsw i64 %3874, %const_reg_qword7, !spirv.Decorations !610
  %3884 = getelementptr float, float addrspace(4)* %66, i64 %3883
  %3885 = getelementptr float, float addrspace(4)* %3884, i64 %3875
  %3886 = addrspacecast float addrspace(4)* %3885 to float addrspace(1)*
  %3887 = load float, float addrspace(1)* %3886, align 4
  %3888 = fmul reassoc nsz arcp contract float %3887, %4, !spirv.Decorations !612
  %3889 = fadd reassoc nsz arcp contract float %3878, %3888, !spirv.Decorations !612
  %3890 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3877
  %3891 = addrspacecast float addrspace(4)* %3890 to float addrspace(1)*
  store float %3889, float addrspace(1)* %3891, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %3869, %3882, %3879
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %3828, i32* %60, align 4, !noalias !614
  br label %3892

3892:                                             ; preds = %3892, %._crit_edge70.1.2
  %3893 = phi i32 [ 0, %._crit_edge70.1.2 ], [ %3898, %3892 ]
  %3894 = zext i32 %3893 to i64
  %3895 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3894
  %3896 = load i32, i32* %3895, align 4, !noalias !614
  %3897 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3894
  store i32 %3896, i32* %3897, align 4, !alias.scope !614
  %3898 = add nuw nsw i32 %3893, 1, !spirv.Decorations !620
  %3899 = icmp eq i32 %3893, 0
  br i1 %3899, label %3892, label %3900, !llvm.loop !630

3900:                                             ; preds = %3892
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3901 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3902 = and i1 %3647, %3829
  br i1 %3902, label %3903, label %._crit_edge70.2.2

3903:                                             ; preds = %3900
  %3904 = shl i64 %3901, 32
  %3905 = ashr exact i64 %3904, 32
  %3906 = ashr i64 %3901, 32
  %3907 = mul nsw i64 %3905, %const_reg_qword9, !spirv.Decorations !610
  %3908 = add nsw i64 %3907, %3906, !spirv.Decorations !610
  %3909 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !612
  br i1 %48, label %3913, label %3910

3910:                                             ; preds = %3903
  %3911 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3908
  %3912 = addrspacecast float addrspace(4)* %3911 to float addrspace(1)*
  store float %3909, float addrspace(1)* %3912, align 4
  br label %._crit_edge70.2.2

3913:                                             ; preds = %3903
  %3914 = mul nsw i64 %3905, %const_reg_qword7, !spirv.Decorations !610
  %3915 = getelementptr float, float addrspace(4)* %66, i64 %3914
  %3916 = getelementptr float, float addrspace(4)* %3915, i64 %3906
  %3917 = addrspacecast float addrspace(4)* %3916 to float addrspace(1)*
  %3918 = load float, float addrspace(1)* %3917, align 4
  %3919 = fmul reassoc nsz arcp contract float %3918, %4, !spirv.Decorations !612
  %3920 = fadd reassoc nsz arcp contract float %3909, %3919, !spirv.Decorations !612
  %3921 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3908
  %3922 = addrspacecast float addrspace(4)* %3921 to float addrspace(1)*
  store float %3920, float addrspace(1)* %3922, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %3900, %3913, %3910
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %3828, i32* %60, align 4, !noalias !614
  br label %3923

3923:                                             ; preds = %3923, %._crit_edge70.2.2
  %3924 = phi i32 [ 0, %._crit_edge70.2.2 ], [ %3929, %3923 ]
  %3925 = zext i32 %3924 to i64
  %3926 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3925
  %3927 = load i32, i32* %3926, align 4, !noalias !614
  %3928 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3925
  store i32 %3927, i32* %3928, align 4, !alias.scope !614
  %3929 = add nuw nsw i32 %3924, 1, !spirv.Decorations !620
  %3930 = icmp eq i32 %3924, 0
  br i1 %3930, label %3923, label %3931, !llvm.loop !630

3931:                                             ; preds = %3923
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3932 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3933 = and i1 %3680, %3829
  br i1 %3933, label %3934, label %.preheader1.2

3934:                                             ; preds = %3931
  %3935 = shl i64 %3932, 32
  %3936 = ashr exact i64 %3935, 32
  %3937 = ashr i64 %3932, 32
  %3938 = mul nsw i64 %3936, %const_reg_qword9, !spirv.Decorations !610
  %3939 = add nsw i64 %3938, %3937, !spirv.Decorations !610
  %3940 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !612
  br i1 %48, label %3944, label %3941

3941:                                             ; preds = %3934
  %3942 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3939
  %3943 = addrspacecast float addrspace(4)* %3942 to float addrspace(1)*
  store float %3940, float addrspace(1)* %3943, align 4
  br label %.preheader1.2

3944:                                             ; preds = %3934
  %3945 = mul nsw i64 %3936, %const_reg_qword7, !spirv.Decorations !610
  %3946 = getelementptr float, float addrspace(4)* %66, i64 %3945
  %3947 = getelementptr float, float addrspace(4)* %3946, i64 %3937
  %3948 = addrspacecast float addrspace(4)* %3947 to float addrspace(1)*
  %3949 = load float, float addrspace(1)* %3948, align 4
  %3950 = fmul reassoc nsz arcp contract float %3949, %4, !spirv.Decorations !612
  %3951 = fadd reassoc nsz arcp contract float %3940, %3950, !spirv.Decorations !612
  %3952 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3939
  %3953 = addrspacecast float addrspace(4)* %3952 to float addrspace(1)*
  store float %3951, float addrspace(1)* %3953, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %3931, %3944, %3941
  %3954 = or i32 %41, 3
  %3955 = icmp slt i32 %3954, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %3954, i32* %60, align 4, !noalias !614
  br label %3956

3956:                                             ; preds = %3956, %.preheader1.2
  %3957 = phi i32 [ 0, %.preheader1.2 ], [ %3962, %3956 ]
  %3958 = zext i32 %3957 to i64
  %3959 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3958
  %3960 = load i32, i32* %3959, align 4, !noalias !614
  %3961 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3958
  store i32 %3960, i32* %3961, align 4, !alias.scope !614
  %3962 = add nuw nsw i32 %3957, 1, !spirv.Decorations !620
  %3963 = icmp eq i32 %3957, 0
  br i1 %3963, label %3956, label %3964, !llvm.loop !630

3964:                                             ; preds = %3956
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3965 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3966 = and i1 %3581, %3955
  br i1 %3966, label %3967, label %._crit_edge70.378

3967:                                             ; preds = %3964
  %3968 = shl i64 %3965, 32
  %3969 = ashr exact i64 %3968, 32
  %3970 = ashr i64 %3965, 32
  %3971 = mul nsw i64 %3969, %const_reg_qword9, !spirv.Decorations !610
  %3972 = add nsw i64 %3971, %3970, !spirv.Decorations !610
  %3973 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !612
  br i1 %48, label %3977, label %3974

3974:                                             ; preds = %3967
  %3975 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3972
  %3976 = addrspacecast float addrspace(4)* %3975 to float addrspace(1)*
  store float %3973, float addrspace(1)* %3976, align 4
  br label %._crit_edge70.378

3977:                                             ; preds = %3967
  %3978 = mul nsw i64 %3969, %const_reg_qword7, !spirv.Decorations !610
  %3979 = getelementptr float, float addrspace(4)* %66, i64 %3978
  %3980 = getelementptr float, float addrspace(4)* %3979, i64 %3970
  %3981 = addrspacecast float addrspace(4)* %3980 to float addrspace(1)*
  %3982 = load float, float addrspace(1)* %3981, align 4
  %3983 = fmul reassoc nsz arcp contract float %3982, %4, !spirv.Decorations !612
  %3984 = fadd reassoc nsz arcp contract float %3973, %3983, !spirv.Decorations !612
  %3985 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3972
  %3986 = addrspacecast float addrspace(4)* %3985 to float addrspace(1)*
  store float %3984, float addrspace(1)* %3986, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %3964, %3977, %3974
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %3954, i32* %60, align 4, !noalias !614
  br label %3987

3987:                                             ; preds = %3987, %._crit_edge70.378
  %3988 = phi i32 [ 0, %._crit_edge70.378 ], [ %3993, %3987 ]
  %3989 = zext i32 %3988 to i64
  %3990 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3989
  %3991 = load i32, i32* %3990, align 4, !noalias !614
  %3992 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3989
  store i32 %3991, i32* %3992, align 4, !alias.scope !614
  %3993 = add nuw nsw i32 %3988, 1, !spirv.Decorations !620
  %3994 = icmp eq i32 %3988, 0
  br i1 %3994, label %3987, label %3995, !llvm.loop !630

3995:                                             ; preds = %3987
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3996 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3997 = and i1 %3614, %3955
  br i1 %3997, label %3998, label %._crit_edge70.1.3

3998:                                             ; preds = %3995
  %3999 = shl i64 %3996, 32
  %4000 = ashr exact i64 %3999, 32
  %4001 = ashr i64 %3996, 32
  %4002 = mul nsw i64 %4000, %const_reg_qword9, !spirv.Decorations !610
  %4003 = add nsw i64 %4002, %4001, !spirv.Decorations !610
  %4004 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !612
  br i1 %48, label %4008, label %4005

4005:                                             ; preds = %3998
  %4006 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4003
  %4007 = addrspacecast float addrspace(4)* %4006 to float addrspace(1)*
  store float %4004, float addrspace(1)* %4007, align 4
  br label %._crit_edge70.1.3

4008:                                             ; preds = %3998
  %4009 = mul nsw i64 %4000, %const_reg_qword7, !spirv.Decorations !610
  %4010 = getelementptr float, float addrspace(4)* %66, i64 %4009
  %4011 = getelementptr float, float addrspace(4)* %4010, i64 %4001
  %4012 = addrspacecast float addrspace(4)* %4011 to float addrspace(1)*
  %4013 = load float, float addrspace(1)* %4012, align 4
  %4014 = fmul reassoc nsz arcp contract float %4013, %4, !spirv.Decorations !612
  %4015 = fadd reassoc nsz arcp contract float %4004, %4014, !spirv.Decorations !612
  %4016 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4003
  %4017 = addrspacecast float addrspace(4)* %4016 to float addrspace(1)*
  store float %4015, float addrspace(1)* %4017, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %3995, %4008, %4005
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %3954, i32* %60, align 4, !noalias !614
  br label %4018

4018:                                             ; preds = %4018, %._crit_edge70.1.3
  %4019 = phi i32 [ 0, %._crit_edge70.1.3 ], [ %4024, %4018 ]
  %4020 = zext i32 %4019 to i64
  %4021 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4020
  %4022 = load i32, i32* %4021, align 4, !noalias !614
  %4023 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4020
  store i32 %4022, i32* %4023, align 4, !alias.scope !614
  %4024 = add nuw nsw i32 %4019, 1, !spirv.Decorations !620
  %4025 = icmp eq i32 %4019, 0
  br i1 %4025, label %4018, label %4026, !llvm.loop !630

4026:                                             ; preds = %4018
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4027 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4028 = and i1 %3647, %3955
  br i1 %4028, label %4029, label %._crit_edge70.2.3

4029:                                             ; preds = %4026
  %4030 = shl i64 %4027, 32
  %4031 = ashr exact i64 %4030, 32
  %4032 = ashr i64 %4027, 32
  %4033 = mul nsw i64 %4031, %const_reg_qword9, !spirv.Decorations !610
  %4034 = add nsw i64 %4033, %4032, !spirv.Decorations !610
  %4035 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !612
  br i1 %48, label %4039, label %4036

4036:                                             ; preds = %4029
  %4037 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4034
  %4038 = addrspacecast float addrspace(4)* %4037 to float addrspace(1)*
  store float %4035, float addrspace(1)* %4038, align 4
  br label %._crit_edge70.2.3

4039:                                             ; preds = %4029
  %4040 = mul nsw i64 %4031, %const_reg_qword7, !spirv.Decorations !610
  %4041 = getelementptr float, float addrspace(4)* %66, i64 %4040
  %4042 = getelementptr float, float addrspace(4)* %4041, i64 %4032
  %4043 = addrspacecast float addrspace(4)* %4042 to float addrspace(1)*
  %4044 = load float, float addrspace(1)* %4043, align 4
  %4045 = fmul reassoc nsz arcp contract float %4044, %4, !spirv.Decorations !612
  %4046 = fadd reassoc nsz arcp contract float %4035, %4045, !spirv.Decorations !612
  %4047 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4034
  %4048 = addrspacecast float addrspace(4)* %4047 to float addrspace(1)*
  store float %4046, float addrspace(1)* %4048, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %4026, %4039, %4036
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %3954, i32* %60, align 4, !noalias !614
  br label %4049

4049:                                             ; preds = %4049, %._crit_edge70.2.3
  %4050 = phi i32 [ 0, %._crit_edge70.2.3 ], [ %4055, %4049 ]
  %4051 = zext i32 %4050 to i64
  %4052 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4051
  %4053 = load i32, i32* %4052, align 4, !noalias !614
  %4054 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4051
  store i32 %4053, i32* %4054, align 4, !alias.scope !614
  %4055 = add nuw nsw i32 %4050, 1, !spirv.Decorations !620
  %4056 = icmp eq i32 %4050, 0
  br i1 %4056, label %4049, label %4057, !llvm.loop !630

4057:                                             ; preds = %4049
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4058 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4059 = and i1 %3680, %3955
  br i1 %4059, label %4060, label %.preheader1.3

4060:                                             ; preds = %4057
  %4061 = shl i64 %4058, 32
  %4062 = ashr exact i64 %4061, 32
  %4063 = ashr i64 %4058, 32
  %4064 = mul nsw i64 %4062, %const_reg_qword9, !spirv.Decorations !610
  %4065 = add nsw i64 %4064, %4063, !spirv.Decorations !610
  %4066 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !612
  br i1 %48, label %4070, label %4067

4067:                                             ; preds = %4060
  %4068 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4065
  %4069 = addrspacecast float addrspace(4)* %4068 to float addrspace(1)*
  store float %4066, float addrspace(1)* %4069, align 4
  br label %.preheader1.3

4070:                                             ; preds = %4060
  %4071 = mul nsw i64 %4062, %const_reg_qword7, !spirv.Decorations !610
  %4072 = getelementptr float, float addrspace(4)* %66, i64 %4071
  %4073 = getelementptr float, float addrspace(4)* %4072, i64 %4063
  %4074 = addrspacecast float addrspace(4)* %4073 to float addrspace(1)*
  %4075 = load float, float addrspace(1)* %4074, align 4
  %4076 = fmul reassoc nsz arcp contract float %4075, %4, !spirv.Decorations !612
  %4077 = fadd reassoc nsz arcp contract float %4066, %4076, !spirv.Decorations !612
  %4078 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4065
  %4079 = addrspacecast float addrspace(4)* %4078 to float addrspace(1)*
  store float %4077, float addrspace(1)* %4079, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %4057, %4070, %4067
  %4080 = or i32 %41, 4
  %4081 = icmp slt i32 %4080, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4080, i32* %60, align 4, !noalias !614
  br label %4082

4082:                                             ; preds = %4082, %.preheader1.3
  %4083 = phi i32 [ 0, %.preheader1.3 ], [ %4088, %4082 ]
  %4084 = zext i32 %4083 to i64
  %4085 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4084
  %4086 = load i32, i32* %4085, align 4, !noalias !614
  %4087 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4084
  store i32 %4086, i32* %4087, align 4, !alias.scope !614
  %4088 = add nuw nsw i32 %4083, 1, !spirv.Decorations !620
  %4089 = icmp eq i32 %4083, 0
  br i1 %4089, label %4082, label %4090, !llvm.loop !630

4090:                                             ; preds = %4082
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4091 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4092 = and i1 %3581, %4081
  br i1 %4092, label %4093, label %._crit_edge70.4

4093:                                             ; preds = %4090
  %4094 = shl i64 %4091, 32
  %4095 = ashr exact i64 %4094, 32
  %4096 = ashr i64 %4091, 32
  %4097 = mul nsw i64 %4095, %const_reg_qword9, !spirv.Decorations !610
  %4098 = add nsw i64 %4097, %4096, !spirv.Decorations !610
  %4099 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !612
  br i1 %48, label %4103, label %4100

4100:                                             ; preds = %4093
  %4101 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4098
  %4102 = addrspacecast float addrspace(4)* %4101 to float addrspace(1)*
  store float %4099, float addrspace(1)* %4102, align 4
  br label %._crit_edge70.4

4103:                                             ; preds = %4093
  %4104 = mul nsw i64 %4095, %const_reg_qword7, !spirv.Decorations !610
  %4105 = getelementptr float, float addrspace(4)* %66, i64 %4104
  %4106 = getelementptr float, float addrspace(4)* %4105, i64 %4096
  %4107 = addrspacecast float addrspace(4)* %4106 to float addrspace(1)*
  %4108 = load float, float addrspace(1)* %4107, align 4
  %4109 = fmul reassoc nsz arcp contract float %4108, %4, !spirv.Decorations !612
  %4110 = fadd reassoc nsz arcp contract float %4099, %4109, !spirv.Decorations !612
  %4111 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4098
  %4112 = addrspacecast float addrspace(4)* %4111 to float addrspace(1)*
  store float %4110, float addrspace(1)* %4112, align 4
  br label %._crit_edge70.4

._crit_edge70.4:                                  ; preds = %4090, %4103, %4100
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4080, i32* %60, align 4, !noalias !614
  br label %4113

4113:                                             ; preds = %4113, %._crit_edge70.4
  %4114 = phi i32 [ 0, %._crit_edge70.4 ], [ %4119, %4113 ]
  %4115 = zext i32 %4114 to i64
  %4116 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4115
  %4117 = load i32, i32* %4116, align 4, !noalias !614
  %4118 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4115
  store i32 %4117, i32* %4118, align 4, !alias.scope !614
  %4119 = add nuw nsw i32 %4114, 1, !spirv.Decorations !620
  %4120 = icmp eq i32 %4114, 0
  br i1 %4120, label %4113, label %4121, !llvm.loop !630

4121:                                             ; preds = %4113
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4122 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4123 = and i1 %3614, %4081
  br i1 %4123, label %4124, label %._crit_edge70.1.4

4124:                                             ; preds = %4121
  %4125 = shl i64 %4122, 32
  %4126 = ashr exact i64 %4125, 32
  %4127 = ashr i64 %4122, 32
  %4128 = mul nsw i64 %4126, %const_reg_qword9, !spirv.Decorations !610
  %4129 = add nsw i64 %4128, %4127, !spirv.Decorations !610
  %4130 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !612
  br i1 %48, label %4134, label %4131

4131:                                             ; preds = %4124
  %4132 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4129
  %4133 = addrspacecast float addrspace(4)* %4132 to float addrspace(1)*
  store float %4130, float addrspace(1)* %4133, align 4
  br label %._crit_edge70.1.4

4134:                                             ; preds = %4124
  %4135 = mul nsw i64 %4126, %const_reg_qword7, !spirv.Decorations !610
  %4136 = getelementptr float, float addrspace(4)* %66, i64 %4135
  %4137 = getelementptr float, float addrspace(4)* %4136, i64 %4127
  %4138 = addrspacecast float addrspace(4)* %4137 to float addrspace(1)*
  %4139 = load float, float addrspace(1)* %4138, align 4
  %4140 = fmul reassoc nsz arcp contract float %4139, %4, !spirv.Decorations !612
  %4141 = fadd reassoc nsz arcp contract float %4130, %4140, !spirv.Decorations !612
  %4142 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4129
  %4143 = addrspacecast float addrspace(4)* %4142 to float addrspace(1)*
  store float %4141, float addrspace(1)* %4143, align 4
  br label %._crit_edge70.1.4

._crit_edge70.1.4:                                ; preds = %4121, %4134, %4131
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4080, i32* %60, align 4, !noalias !614
  br label %4144

4144:                                             ; preds = %4144, %._crit_edge70.1.4
  %4145 = phi i32 [ 0, %._crit_edge70.1.4 ], [ %4150, %4144 ]
  %4146 = zext i32 %4145 to i64
  %4147 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4146
  %4148 = load i32, i32* %4147, align 4, !noalias !614
  %4149 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4146
  store i32 %4148, i32* %4149, align 4, !alias.scope !614
  %4150 = add nuw nsw i32 %4145, 1, !spirv.Decorations !620
  %4151 = icmp eq i32 %4145, 0
  br i1 %4151, label %4144, label %4152, !llvm.loop !630

4152:                                             ; preds = %4144
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4153 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4154 = and i1 %3647, %4081
  br i1 %4154, label %4155, label %._crit_edge70.2.4

4155:                                             ; preds = %4152
  %4156 = shl i64 %4153, 32
  %4157 = ashr exact i64 %4156, 32
  %4158 = ashr i64 %4153, 32
  %4159 = mul nsw i64 %4157, %const_reg_qword9, !spirv.Decorations !610
  %4160 = add nsw i64 %4159, %4158, !spirv.Decorations !610
  %4161 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !612
  br i1 %48, label %4165, label %4162

4162:                                             ; preds = %4155
  %4163 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4160
  %4164 = addrspacecast float addrspace(4)* %4163 to float addrspace(1)*
  store float %4161, float addrspace(1)* %4164, align 4
  br label %._crit_edge70.2.4

4165:                                             ; preds = %4155
  %4166 = mul nsw i64 %4157, %const_reg_qword7, !spirv.Decorations !610
  %4167 = getelementptr float, float addrspace(4)* %66, i64 %4166
  %4168 = getelementptr float, float addrspace(4)* %4167, i64 %4158
  %4169 = addrspacecast float addrspace(4)* %4168 to float addrspace(1)*
  %4170 = load float, float addrspace(1)* %4169, align 4
  %4171 = fmul reassoc nsz arcp contract float %4170, %4, !spirv.Decorations !612
  %4172 = fadd reassoc nsz arcp contract float %4161, %4171, !spirv.Decorations !612
  %4173 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4160
  %4174 = addrspacecast float addrspace(4)* %4173 to float addrspace(1)*
  store float %4172, float addrspace(1)* %4174, align 4
  br label %._crit_edge70.2.4

._crit_edge70.2.4:                                ; preds = %4152, %4165, %4162
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4080, i32* %60, align 4, !noalias !614
  br label %4175

4175:                                             ; preds = %4175, %._crit_edge70.2.4
  %4176 = phi i32 [ 0, %._crit_edge70.2.4 ], [ %4181, %4175 ]
  %4177 = zext i32 %4176 to i64
  %4178 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4177
  %4179 = load i32, i32* %4178, align 4, !noalias !614
  %4180 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4177
  store i32 %4179, i32* %4180, align 4, !alias.scope !614
  %4181 = add nuw nsw i32 %4176, 1, !spirv.Decorations !620
  %4182 = icmp eq i32 %4176, 0
  br i1 %4182, label %4175, label %4183, !llvm.loop !630

4183:                                             ; preds = %4175
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4184 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4185 = and i1 %3680, %4081
  br i1 %4185, label %4186, label %.preheader1.4

4186:                                             ; preds = %4183
  %4187 = shl i64 %4184, 32
  %4188 = ashr exact i64 %4187, 32
  %4189 = ashr i64 %4184, 32
  %4190 = mul nsw i64 %4188, %const_reg_qword9, !spirv.Decorations !610
  %4191 = add nsw i64 %4190, %4189, !spirv.Decorations !610
  %4192 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !612
  br i1 %48, label %4196, label %4193

4193:                                             ; preds = %4186
  %4194 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4191
  %4195 = addrspacecast float addrspace(4)* %4194 to float addrspace(1)*
  store float %4192, float addrspace(1)* %4195, align 4
  br label %.preheader1.4

4196:                                             ; preds = %4186
  %4197 = mul nsw i64 %4188, %const_reg_qword7, !spirv.Decorations !610
  %4198 = getelementptr float, float addrspace(4)* %66, i64 %4197
  %4199 = getelementptr float, float addrspace(4)* %4198, i64 %4189
  %4200 = addrspacecast float addrspace(4)* %4199 to float addrspace(1)*
  %4201 = load float, float addrspace(1)* %4200, align 4
  %4202 = fmul reassoc nsz arcp contract float %4201, %4, !spirv.Decorations !612
  %4203 = fadd reassoc nsz arcp contract float %4192, %4202, !spirv.Decorations !612
  %4204 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4191
  %4205 = addrspacecast float addrspace(4)* %4204 to float addrspace(1)*
  store float %4203, float addrspace(1)* %4205, align 4
  br label %.preheader1.4

.preheader1.4:                                    ; preds = %4183, %4196, %4193
  %4206 = or i32 %41, 5
  %4207 = icmp slt i32 %4206, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4206, i32* %60, align 4, !noalias !614
  br label %4208

4208:                                             ; preds = %4208, %.preheader1.4
  %4209 = phi i32 [ 0, %.preheader1.4 ], [ %4214, %4208 ]
  %4210 = zext i32 %4209 to i64
  %4211 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4210
  %4212 = load i32, i32* %4211, align 4, !noalias !614
  %4213 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4210
  store i32 %4212, i32* %4213, align 4, !alias.scope !614
  %4214 = add nuw nsw i32 %4209, 1, !spirv.Decorations !620
  %4215 = icmp eq i32 %4209, 0
  br i1 %4215, label %4208, label %4216, !llvm.loop !630

4216:                                             ; preds = %4208
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4217 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4218 = and i1 %3581, %4207
  br i1 %4218, label %4219, label %._crit_edge70.5

4219:                                             ; preds = %4216
  %4220 = shl i64 %4217, 32
  %4221 = ashr exact i64 %4220, 32
  %4222 = ashr i64 %4217, 32
  %4223 = mul nsw i64 %4221, %const_reg_qword9, !spirv.Decorations !610
  %4224 = add nsw i64 %4223, %4222, !spirv.Decorations !610
  %4225 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !612
  br i1 %48, label %4229, label %4226

4226:                                             ; preds = %4219
  %4227 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4224
  %4228 = addrspacecast float addrspace(4)* %4227 to float addrspace(1)*
  store float %4225, float addrspace(1)* %4228, align 4
  br label %._crit_edge70.5

4229:                                             ; preds = %4219
  %4230 = mul nsw i64 %4221, %const_reg_qword7, !spirv.Decorations !610
  %4231 = getelementptr float, float addrspace(4)* %66, i64 %4230
  %4232 = getelementptr float, float addrspace(4)* %4231, i64 %4222
  %4233 = addrspacecast float addrspace(4)* %4232 to float addrspace(1)*
  %4234 = load float, float addrspace(1)* %4233, align 4
  %4235 = fmul reassoc nsz arcp contract float %4234, %4, !spirv.Decorations !612
  %4236 = fadd reassoc nsz arcp contract float %4225, %4235, !spirv.Decorations !612
  %4237 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4224
  %4238 = addrspacecast float addrspace(4)* %4237 to float addrspace(1)*
  store float %4236, float addrspace(1)* %4238, align 4
  br label %._crit_edge70.5

._crit_edge70.5:                                  ; preds = %4216, %4229, %4226
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4206, i32* %60, align 4, !noalias !614
  br label %4239

4239:                                             ; preds = %4239, %._crit_edge70.5
  %4240 = phi i32 [ 0, %._crit_edge70.5 ], [ %4245, %4239 ]
  %4241 = zext i32 %4240 to i64
  %4242 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4241
  %4243 = load i32, i32* %4242, align 4, !noalias !614
  %4244 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4241
  store i32 %4243, i32* %4244, align 4, !alias.scope !614
  %4245 = add nuw nsw i32 %4240, 1, !spirv.Decorations !620
  %4246 = icmp eq i32 %4240, 0
  br i1 %4246, label %4239, label %4247, !llvm.loop !630

4247:                                             ; preds = %4239
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4248 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4249 = and i1 %3614, %4207
  br i1 %4249, label %4250, label %._crit_edge70.1.5

4250:                                             ; preds = %4247
  %4251 = shl i64 %4248, 32
  %4252 = ashr exact i64 %4251, 32
  %4253 = ashr i64 %4248, 32
  %4254 = mul nsw i64 %4252, %const_reg_qword9, !spirv.Decorations !610
  %4255 = add nsw i64 %4254, %4253, !spirv.Decorations !610
  %4256 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !612
  br i1 %48, label %4260, label %4257

4257:                                             ; preds = %4250
  %4258 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4255
  %4259 = addrspacecast float addrspace(4)* %4258 to float addrspace(1)*
  store float %4256, float addrspace(1)* %4259, align 4
  br label %._crit_edge70.1.5

4260:                                             ; preds = %4250
  %4261 = mul nsw i64 %4252, %const_reg_qword7, !spirv.Decorations !610
  %4262 = getelementptr float, float addrspace(4)* %66, i64 %4261
  %4263 = getelementptr float, float addrspace(4)* %4262, i64 %4253
  %4264 = addrspacecast float addrspace(4)* %4263 to float addrspace(1)*
  %4265 = load float, float addrspace(1)* %4264, align 4
  %4266 = fmul reassoc nsz arcp contract float %4265, %4, !spirv.Decorations !612
  %4267 = fadd reassoc nsz arcp contract float %4256, %4266, !spirv.Decorations !612
  %4268 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4255
  %4269 = addrspacecast float addrspace(4)* %4268 to float addrspace(1)*
  store float %4267, float addrspace(1)* %4269, align 4
  br label %._crit_edge70.1.5

._crit_edge70.1.5:                                ; preds = %4247, %4260, %4257
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4206, i32* %60, align 4, !noalias !614
  br label %4270

4270:                                             ; preds = %4270, %._crit_edge70.1.5
  %4271 = phi i32 [ 0, %._crit_edge70.1.5 ], [ %4276, %4270 ]
  %4272 = zext i32 %4271 to i64
  %4273 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4272
  %4274 = load i32, i32* %4273, align 4, !noalias !614
  %4275 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4272
  store i32 %4274, i32* %4275, align 4, !alias.scope !614
  %4276 = add nuw nsw i32 %4271, 1, !spirv.Decorations !620
  %4277 = icmp eq i32 %4271, 0
  br i1 %4277, label %4270, label %4278, !llvm.loop !630

4278:                                             ; preds = %4270
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4279 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4280 = and i1 %3647, %4207
  br i1 %4280, label %4281, label %._crit_edge70.2.5

4281:                                             ; preds = %4278
  %4282 = shl i64 %4279, 32
  %4283 = ashr exact i64 %4282, 32
  %4284 = ashr i64 %4279, 32
  %4285 = mul nsw i64 %4283, %const_reg_qword9, !spirv.Decorations !610
  %4286 = add nsw i64 %4285, %4284, !spirv.Decorations !610
  %4287 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !612
  br i1 %48, label %4291, label %4288

4288:                                             ; preds = %4281
  %4289 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4286
  %4290 = addrspacecast float addrspace(4)* %4289 to float addrspace(1)*
  store float %4287, float addrspace(1)* %4290, align 4
  br label %._crit_edge70.2.5

4291:                                             ; preds = %4281
  %4292 = mul nsw i64 %4283, %const_reg_qword7, !spirv.Decorations !610
  %4293 = getelementptr float, float addrspace(4)* %66, i64 %4292
  %4294 = getelementptr float, float addrspace(4)* %4293, i64 %4284
  %4295 = addrspacecast float addrspace(4)* %4294 to float addrspace(1)*
  %4296 = load float, float addrspace(1)* %4295, align 4
  %4297 = fmul reassoc nsz arcp contract float %4296, %4, !spirv.Decorations !612
  %4298 = fadd reassoc nsz arcp contract float %4287, %4297, !spirv.Decorations !612
  %4299 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4286
  %4300 = addrspacecast float addrspace(4)* %4299 to float addrspace(1)*
  store float %4298, float addrspace(1)* %4300, align 4
  br label %._crit_edge70.2.5

._crit_edge70.2.5:                                ; preds = %4278, %4291, %4288
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4206, i32* %60, align 4, !noalias !614
  br label %4301

4301:                                             ; preds = %4301, %._crit_edge70.2.5
  %4302 = phi i32 [ 0, %._crit_edge70.2.5 ], [ %4307, %4301 ]
  %4303 = zext i32 %4302 to i64
  %4304 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4303
  %4305 = load i32, i32* %4304, align 4, !noalias !614
  %4306 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4303
  store i32 %4305, i32* %4306, align 4, !alias.scope !614
  %4307 = add nuw nsw i32 %4302, 1, !spirv.Decorations !620
  %4308 = icmp eq i32 %4302, 0
  br i1 %4308, label %4301, label %4309, !llvm.loop !630

4309:                                             ; preds = %4301
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4310 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4311 = and i1 %3680, %4207
  br i1 %4311, label %4312, label %.preheader1.5

4312:                                             ; preds = %4309
  %4313 = shl i64 %4310, 32
  %4314 = ashr exact i64 %4313, 32
  %4315 = ashr i64 %4310, 32
  %4316 = mul nsw i64 %4314, %const_reg_qword9, !spirv.Decorations !610
  %4317 = add nsw i64 %4316, %4315, !spirv.Decorations !610
  %4318 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !612
  br i1 %48, label %4322, label %4319

4319:                                             ; preds = %4312
  %4320 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4317
  %4321 = addrspacecast float addrspace(4)* %4320 to float addrspace(1)*
  store float %4318, float addrspace(1)* %4321, align 4
  br label %.preheader1.5

4322:                                             ; preds = %4312
  %4323 = mul nsw i64 %4314, %const_reg_qword7, !spirv.Decorations !610
  %4324 = getelementptr float, float addrspace(4)* %66, i64 %4323
  %4325 = getelementptr float, float addrspace(4)* %4324, i64 %4315
  %4326 = addrspacecast float addrspace(4)* %4325 to float addrspace(1)*
  %4327 = load float, float addrspace(1)* %4326, align 4
  %4328 = fmul reassoc nsz arcp contract float %4327, %4, !spirv.Decorations !612
  %4329 = fadd reassoc nsz arcp contract float %4318, %4328, !spirv.Decorations !612
  %4330 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4317
  %4331 = addrspacecast float addrspace(4)* %4330 to float addrspace(1)*
  store float %4329, float addrspace(1)* %4331, align 4
  br label %.preheader1.5

.preheader1.5:                                    ; preds = %4309, %4322, %4319
  %4332 = or i32 %41, 6
  %4333 = icmp slt i32 %4332, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4332, i32* %60, align 4, !noalias !614
  br label %4334

4334:                                             ; preds = %4334, %.preheader1.5
  %4335 = phi i32 [ 0, %.preheader1.5 ], [ %4340, %4334 ]
  %4336 = zext i32 %4335 to i64
  %4337 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4336
  %4338 = load i32, i32* %4337, align 4, !noalias !614
  %4339 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4336
  store i32 %4338, i32* %4339, align 4, !alias.scope !614
  %4340 = add nuw nsw i32 %4335, 1, !spirv.Decorations !620
  %4341 = icmp eq i32 %4335, 0
  br i1 %4341, label %4334, label %4342, !llvm.loop !630

4342:                                             ; preds = %4334
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4343 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4344 = and i1 %3581, %4333
  br i1 %4344, label %4345, label %._crit_edge70.6

4345:                                             ; preds = %4342
  %4346 = shl i64 %4343, 32
  %4347 = ashr exact i64 %4346, 32
  %4348 = ashr i64 %4343, 32
  %4349 = mul nsw i64 %4347, %const_reg_qword9, !spirv.Decorations !610
  %4350 = add nsw i64 %4349, %4348, !spirv.Decorations !610
  %4351 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !612
  br i1 %48, label %4355, label %4352

4352:                                             ; preds = %4345
  %4353 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4350
  %4354 = addrspacecast float addrspace(4)* %4353 to float addrspace(1)*
  store float %4351, float addrspace(1)* %4354, align 4
  br label %._crit_edge70.6

4355:                                             ; preds = %4345
  %4356 = mul nsw i64 %4347, %const_reg_qword7, !spirv.Decorations !610
  %4357 = getelementptr float, float addrspace(4)* %66, i64 %4356
  %4358 = getelementptr float, float addrspace(4)* %4357, i64 %4348
  %4359 = addrspacecast float addrspace(4)* %4358 to float addrspace(1)*
  %4360 = load float, float addrspace(1)* %4359, align 4
  %4361 = fmul reassoc nsz arcp contract float %4360, %4, !spirv.Decorations !612
  %4362 = fadd reassoc nsz arcp contract float %4351, %4361, !spirv.Decorations !612
  %4363 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4350
  %4364 = addrspacecast float addrspace(4)* %4363 to float addrspace(1)*
  store float %4362, float addrspace(1)* %4364, align 4
  br label %._crit_edge70.6

._crit_edge70.6:                                  ; preds = %4342, %4355, %4352
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4332, i32* %60, align 4, !noalias !614
  br label %4365

4365:                                             ; preds = %4365, %._crit_edge70.6
  %4366 = phi i32 [ 0, %._crit_edge70.6 ], [ %4371, %4365 ]
  %4367 = zext i32 %4366 to i64
  %4368 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4367
  %4369 = load i32, i32* %4368, align 4, !noalias !614
  %4370 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4367
  store i32 %4369, i32* %4370, align 4, !alias.scope !614
  %4371 = add nuw nsw i32 %4366, 1, !spirv.Decorations !620
  %4372 = icmp eq i32 %4366, 0
  br i1 %4372, label %4365, label %4373, !llvm.loop !630

4373:                                             ; preds = %4365
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4374 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4375 = and i1 %3614, %4333
  br i1 %4375, label %4376, label %._crit_edge70.1.6

4376:                                             ; preds = %4373
  %4377 = shl i64 %4374, 32
  %4378 = ashr exact i64 %4377, 32
  %4379 = ashr i64 %4374, 32
  %4380 = mul nsw i64 %4378, %const_reg_qword9, !spirv.Decorations !610
  %4381 = add nsw i64 %4380, %4379, !spirv.Decorations !610
  %4382 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !612
  br i1 %48, label %4386, label %4383

4383:                                             ; preds = %4376
  %4384 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4381
  %4385 = addrspacecast float addrspace(4)* %4384 to float addrspace(1)*
  store float %4382, float addrspace(1)* %4385, align 4
  br label %._crit_edge70.1.6

4386:                                             ; preds = %4376
  %4387 = mul nsw i64 %4378, %const_reg_qword7, !spirv.Decorations !610
  %4388 = getelementptr float, float addrspace(4)* %66, i64 %4387
  %4389 = getelementptr float, float addrspace(4)* %4388, i64 %4379
  %4390 = addrspacecast float addrspace(4)* %4389 to float addrspace(1)*
  %4391 = load float, float addrspace(1)* %4390, align 4
  %4392 = fmul reassoc nsz arcp contract float %4391, %4, !spirv.Decorations !612
  %4393 = fadd reassoc nsz arcp contract float %4382, %4392, !spirv.Decorations !612
  %4394 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4381
  %4395 = addrspacecast float addrspace(4)* %4394 to float addrspace(1)*
  store float %4393, float addrspace(1)* %4395, align 4
  br label %._crit_edge70.1.6

._crit_edge70.1.6:                                ; preds = %4373, %4386, %4383
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4332, i32* %60, align 4, !noalias !614
  br label %4396

4396:                                             ; preds = %4396, %._crit_edge70.1.6
  %4397 = phi i32 [ 0, %._crit_edge70.1.6 ], [ %4402, %4396 ]
  %4398 = zext i32 %4397 to i64
  %4399 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4398
  %4400 = load i32, i32* %4399, align 4, !noalias !614
  %4401 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4398
  store i32 %4400, i32* %4401, align 4, !alias.scope !614
  %4402 = add nuw nsw i32 %4397, 1, !spirv.Decorations !620
  %4403 = icmp eq i32 %4397, 0
  br i1 %4403, label %4396, label %4404, !llvm.loop !630

4404:                                             ; preds = %4396
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4405 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4406 = and i1 %3647, %4333
  br i1 %4406, label %4407, label %._crit_edge70.2.6

4407:                                             ; preds = %4404
  %4408 = shl i64 %4405, 32
  %4409 = ashr exact i64 %4408, 32
  %4410 = ashr i64 %4405, 32
  %4411 = mul nsw i64 %4409, %const_reg_qword9, !spirv.Decorations !610
  %4412 = add nsw i64 %4411, %4410, !spirv.Decorations !610
  %4413 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !612
  br i1 %48, label %4417, label %4414

4414:                                             ; preds = %4407
  %4415 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4412
  %4416 = addrspacecast float addrspace(4)* %4415 to float addrspace(1)*
  store float %4413, float addrspace(1)* %4416, align 4
  br label %._crit_edge70.2.6

4417:                                             ; preds = %4407
  %4418 = mul nsw i64 %4409, %const_reg_qword7, !spirv.Decorations !610
  %4419 = getelementptr float, float addrspace(4)* %66, i64 %4418
  %4420 = getelementptr float, float addrspace(4)* %4419, i64 %4410
  %4421 = addrspacecast float addrspace(4)* %4420 to float addrspace(1)*
  %4422 = load float, float addrspace(1)* %4421, align 4
  %4423 = fmul reassoc nsz arcp contract float %4422, %4, !spirv.Decorations !612
  %4424 = fadd reassoc nsz arcp contract float %4413, %4423, !spirv.Decorations !612
  %4425 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4412
  %4426 = addrspacecast float addrspace(4)* %4425 to float addrspace(1)*
  store float %4424, float addrspace(1)* %4426, align 4
  br label %._crit_edge70.2.6

._crit_edge70.2.6:                                ; preds = %4404, %4417, %4414
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4332, i32* %60, align 4, !noalias !614
  br label %4427

4427:                                             ; preds = %4427, %._crit_edge70.2.6
  %4428 = phi i32 [ 0, %._crit_edge70.2.6 ], [ %4433, %4427 ]
  %4429 = zext i32 %4428 to i64
  %4430 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4429
  %4431 = load i32, i32* %4430, align 4, !noalias !614
  %4432 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4429
  store i32 %4431, i32* %4432, align 4, !alias.scope !614
  %4433 = add nuw nsw i32 %4428, 1, !spirv.Decorations !620
  %4434 = icmp eq i32 %4428, 0
  br i1 %4434, label %4427, label %4435, !llvm.loop !630

4435:                                             ; preds = %4427
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4436 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4437 = and i1 %3680, %4333
  br i1 %4437, label %4438, label %.preheader1.6

4438:                                             ; preds = %4435
  %4439 = shl i64 %4436, 32
  %4440 = ashr exact i64 %4439, 32
  %4441 = ashr i64 %4436, 32
  %4442 = mul nsw i64 %4440, %const_reg_qword9, !spirv.Decorations !610
  %4443 = add nsw i64 %4442, %4441, !spirv.Decorations !610
  %4444 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !612
  br i1 %48, label %4448, label %4445

4445:                                             ; preds = %4438
  %4446 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4443
  %4447 = addrspacecast float addrspace(4)* %4446 to float addrspace(1)*
  store float %4444, float addrspace(1)* %4447, align 4
  br label %.preheader1.6

4448:                                             ; preds = %4438
  %4449 = mul nsw i64 %4440, %const_reg_qword7, !spirv.Decorations !610
  %4450 = getelementptr float, float addrspace(4)* %66, i64 %4449
  %4451 = getelementptr float, float addrspace(4)* %4450, i64 %4441
  %4452 = addrspacecast float addrspace(4)* %4451 to float addrspace(1)*
  %4453 = load float, float addrspace(1)* %4452, align 4
  %4454 = fmul reassoc nsz arcp contract float %4453, %4, !spirv.Decorations !612
  %4455 = fadd reassoc nsz arcp contract float %4444, %4454, !spirv.Decorations !612
  %4456 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4443
  %4457 = addrspacecast float addrspace(4)* %4456 to float addrspace(1)*
  store float %4455, float addrspace(1)* %4457, align 4
  br label %.preheader1.6

.preheader1.6:                                    ; preds = %4435, %4448, %4445
  %4458 = or i32 %41, 7
  %4459 = icmp slt i32 %4458, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4458, i32* %60, align 4, !noalias !614
  br label %4460

4460:                                             ; preds = %4460, %.preheader1.6
  %4461 = phi i32 [ 0, %.preheader1.6 ], [ %4466, %4460 ]
  %4462 = zext i32 %4461 to i64
  %4463 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4462
  %4464 = load i32, i32* %4463, align 4, !noalias !614
  %4465 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4462
  store i32 %4464, i32* %4465, align 4, !alias.scope !614
  %4466 = add nuw nsw i32 %4461, 1, !spirv.Decorations !620
  %4467 = icmp eq i32 %4461, 0
  br i1 %4467, label %4460, label %4468, !llvm.loop !630

4468:                                             ; preds = %4460
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4469 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4470 = and i1 %3581, %4459
  br i1 %4470, label %4471, label %._crit_edge70.7

4471:                                             ; preds = %4468
  %4472 = shl i64 %4469, 32
  %4473 = ashr exact i64 %4472, 32
  %4474 = ashr i64 %4469, 32
  %4475 = mul nsw i64 %4473, %const_reg_qword9, !spirv.Decorations !610
  %4476 = add nsw i64 %4475, %4474, !spirv.Decorations !610
  %4477 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !612
  br i1 %48, label %4481, label %4478

4478:                                             ; preds = %4471
  %4479 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4476
  %4480 = addrspacecast float addrspace(4)* %4479 to float addrspace(1)*
  store float %4477, float addrspace(1)* %4480, align 4
  br label %._crit_edge70.7

4481:                                             ; preds = %4471
  %4482 = mul nsw i64 %4473, %const_reg_qword7, !spirv.Decorations !610
  %4483 = getelementptr float, float addrspace(4)* %66, i64 %4482
  %4484 = getelementptr float, float addrspace(4)* %4483, i64 %4474
  %4485 = addrspacecast float addrspace(4)* %4484 to float addrspace(1)*
  %4486 = load float, float addrspace(1)* %4485, align 4
  %4487 = fmul reassoc nsz arcp contract float %4486, %4, !spirv.Decorations !612
  %4488 = fadd reassoc nsz arcp contract float %4477, %4487, !spirv.Decorations !612
  %4489 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4476
  %4490 = addrspacecast float addrspace(4)* %4489 to float addrspace(1)*
  store float %4488, float addrspace(1)* %4490, align 4
  br label %._crit_edge70.7

._crit_edge70.7:                                  ; preds = %4468, %4481, %4478
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4458, i32* %60, align 4, !noalias !614
  br label %4491

4491:                                             ; preds = %4491, %._crit_edge70.7
  %4492 = phi i32 [ 0, %._crit_edge70.7 ], [ %4497, %4491 ]
  %4493 = zext i32 %4492 to i64
  %4494 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4493
  %4495 = load i32, i32* %4494, align 4, !noalias !614
  %4496 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4493
  store i32 %4495, i32* %4496, align 4, !alias.scope !614
  %4497 = add nuw nsw i32 %4492, 1, !spirv.Decorations !620
  %4498 = icmp eq i32 %4492, 0
  br i1 %4498, label %4491, label %4499, !llvm.loop !630

4499:                                             ; preds = %4491
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4500 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4501 = and i1 %3614, %4459
  br i1 %4501, label %4502, label %._crit_edge70.1.7

4502:                                             ; preds = %4499
  %4503 = shl i64 %4500, 32
  %4504 = ashr exact i64 %4503, 32
  %4505 = ashr i64 %4500, 32
  %4506 = mul nsw i64 %4504, %const_reg_qword9, !spirv.Decorations !610
  %4507 = add nsw i64 %4506, %4505, !spirv.Decorations !610
  %4508 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !612
  br i1 %48, label %4512, label %4509

4509:                                             ; preds = %4502
  %4510 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4507
  %4511 = addrspacecast float addrspace(4)* %4510 to float addrspace(1)*
  store float %4508, float addrspace(1)* %4511, align 4
  br label %._crit_edge70.1.7

4512:                                             ; preds = %4502
  %4513 = mul nsw i64 %4504, %const_reg_qword7, !spirv.Decorations !610
  %4514 = getelementptr float, float addrspace(4)* %66, i64 %4513
  %4515 = getelementptr float, float addrspace(4)* %4514, i64 %4505
  %4516 = addrspacecast float addrspace(4)* %4515 to float addrspace(1)*
  %4517 = load float, float addrspace(1)* %4516, align 4
  %4518 = fmul reassoc nsz arcp contract float %4517, %4, !spirv.Decorations !612
  %4519 = fadd reassoc nsz arcp contract float %4508, %4518, !spirv.Decorations !612
  %4520 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4507
  %4521 = addrspacecast float addrspace(4)* %4520 to float addrspace(1)*
  store float %4519, float addrspace(1)* %4521, align 4
  br label %._crit_edge70.1.7

._crit_edge70.1.7:                                ; preds = %4499, %4512, %4509
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4458, i32* %60, align 4, !noalias !614
  br label %4522

4522:                                             ; preds = %4522, %._crit_edge70.1.7
  %4523 = phi i32 [ 0, %._crit_edge70.1.7 ], [ %4528, %4522 ]
  %4524 = zext i32 %4523 to i64
  %4525 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4524
  %4526 = load i32, i32* %4525, align 4, !noalias !614
  %4527 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4524
  store i32 %4526, i32* %4527, align 4, !alias.scope !614
  %4528 = add nuw nsw i32 %4523, 1, !spirv.Decorations !620
  %4529 = icmp eq i32 %4523, 0
  br i1 %4529, label %4522, label %4530, !llvm.loop !630

4530:                                             ; preds = %4522
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4531 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4532 = and i1 %3647, %4459
  br i1 %4532, label %4533, label %._crit_edge70.2.7

4533:                                             ; preds = %4530
  %4534 = shl i64 %4531, 32
  %4535 = ashr exact i64 %4534, 32
  %4536 = ashr i64 %4531, 32
  %4537 = mul nsw i64 %4535, %const_reg_qword9, !spirv.Decorations !610
  %4538 = add nsw i64 %4537, %4536, !spirv.Decorations !610
  %4539 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !612
  br i1 %48, label %4543, label %4540

4540:                                             ; preds = %4533
  %4541 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4538
  %4542 = addrspacecast float addrspace(4)* %4541 to float addrspace(1)*
  store float %4539, float addrspace(1)* %4542, align 4
  br label %._crit_edge70.2.7

4543:                                             ; preds = %4533
  %4544 = mul nsw i64 %4535, %const_reg_qword7, !spirv.Decorations !610
  %4545 = getelementptr float, float addrspace(4)* %66, i64 %4544
  %4546 = getelementptr float, float addrspace(4)* %4545, i64 %4536
  %4547 = addrspacecast float addrspace(4)* %4546 to float addrspace(1)*
  %4548 = load float, float addrspace(1)* %4547, align 4
  %4549 = fmul reassoc nsz arcp contract float %4548, %4, !spirv.Decorations !612
  %4550 = fadd reassoc nsz arcp contract float %4539, %4549, !spirv.Decorations !612
  %4551 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4538
  %4552 = addrspacecast float addrspace(4)* %4551 to float addrspace(1)*
  store float %4550, float addrspace(1)* %4552, align 4
  br label %._crit_edge70.2.7

._crit_edge70.2.7:                                ; preds = %4530, %4543, %4540
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4458, i32* %60, align 4, !noalias !614
  br label %4553

4553:                                             ; preds = %4553, %._crit_edge70.2.7
  %4554 = phi i32 [ 0, %._crit_edge70.2.7 ], [ %4559, %4553 ]
  %4555 = zext i32 %4554 to i64
  %4556 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4555
  %4557 = load i32, i32* %4556, align 4, !noalias !614
  %4558 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4555
  store i32 %4557, i32* %4558, align 4, !alias.scope !614
  %4559 = add nuw nsw i32 %4554, 1, !spirv.Decorations !620
  %4560 = icmp eq i32 %4554, 0
  br i1 %4560, label %4553, label %4561, !llvm.loop !630

4561:                                             ; preds = %4553
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4562 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4563 = and i1 %3680, %4459
  br i1 %4563, label %4564, label %.preheader1.7

4564:                                             ; preds = %4561
  %4565 = shl i64 %4562, 32
  %4566 = ashr exact i64 %4565, 32
  %4567 = ashr i64 %4562, 32
  %4568 = mul nsw i64 %4566, %const_reg_qword9, !spirv.Decorations !610
  %4569 = add nsw i64 %4568, %4567, !spirv.Decorations !610
  %4570 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !612
  br i1 %48, label %4574, label %4571

4571:                                             ; preds = %4564
  %4572 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4569
  %4573 = addrspacecast float addrspace(4)* %4572 to float addrspace(1)*
  store float %4570, float addrspace(1)* %4573, align 4
  br label %.preheader1.7

4574:                                             ; preds = %4564
  %4575 = mul nsw i64 %4566, %const_reg_qword7, !spirv.Decorations !610
  %4576 = getelementptr float, float addrspace(4)* %66, i64 %4575
  %4577 = getelementptr float, float addrspace(4)* %4576, i64 %4567
  %4578 = addrspacecast float addrspace(4)* %4577 to float addrspace(1)*
  %4579 = load float, float addrspace(1)* %4578, align 4
  %4580 = fmul reassoc nsz arcp contract float %4579, %4, !spirv.Decorations !612
  %4581 = fadd reassoc nsz arcp contract float %4570, %4580, !spirv.Decorations !612
  %4582 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4569
  %4583 = addrspacecast float addrspace(4)* %4582 to float addrspace(1)*
  store float %4581, float addrspace(1)* %4583, align 4
  br label %.preheader1.7

.preheader1.7:                                    ; preds = %4561, %4574, %4571
  %4584 = or i32 %41, 8
  %4585 = icmp slt i32 %4584, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4584, i32* %60, align 4, !noalias !614
  br label %4586

4586:                                             ; preds = %4586, %.preheader1.7
  %4587 = phi i32 [ 0, %.preheader1.7 ], [ %4592, %4586 ]
  %4588 = zext i32 %4587 to i64
  %4589 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4588
  %4590 = load i32, i32* %4589, align 4, !noalias !614
  %4591 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4588
  store i32 %4590, i32* %4591, align 4, !alias.scope !614
  %4592 = add nuw nsw i32 %4587, 1, !spirv.Decorations !620
  %4593 = icmp eq i32 %4587, 0
  br i1 %4593, label %4586, label %4594, !llvm.loop !630

4594:                                             ; preds = %4586
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4595 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4596 = and i1 %3581, %4585
  br i1 %4596, label %4597, label %._crit_edge70.8

4597:                                             ; preds = %4594
  %4598 = shl i64 %4595, 32
  %4599 = ashr exact i64 %4598, 32
  %4600 = ashr i64 %4595, 32
  %4601 = mul nsw i64 %4599, %const_reg_qword9, !spirv.Decorations !610
  %4602 = add nsw i64 %4601, %4600, !spirv.Decorations !610
  %4603 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !612
  br i1 %48, label %4607, label %4604

4604:                                             ; preds = %4597
  %4605 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4602
  %4606 = addrspacecast float addrspace(4)* %4605 to float addrspace(1)*
  store float %4603, float addrspace(1)* %4606, align 4
  br label %._crit_edge70.8

4607:                                             ; preds = %4597
  %4608 = mul nsw i64 %4599, %const_reg_qword7, !spirv.Decorations !610
  %4609 = getelementptr float, float addrspace(4)* %66, i64 %4608
  %4610 = getelementptr float, float addrspace(4)* %4609, i64 %4600
  %4611 = addrspacecast float addrspace(4)* %4610 to float addrspace(1)*
  %4612 = load float, float addrspace(1)* %4611, align 4
  %4613 = fmul reassoc nsz arcp contract float %4612, %4, !spirv.Decorations !612
  %4614 = fadd reassoc nsz arcp contract float %4603, %4613, !spirv.Decorations !612
  %4615 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4602
  %4616 = addrspacecast float addrspace(4)* %4615 to float addrspace(1)*
  store float %4614, float addrspace(1)* %4616, align 4
  br label %._crit_edge70.8

._crit_edge70.8:                                  ; preds = %4594, %4607, %4604
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4584, i32* %60, align 4, !noalias !614
  br label %4617

4617:                                             ; preds = %4617, %._crit_edge70.8
  %4618 = phi i32 [ 0, %._crit_edge70.8 ], [ %4623, %4617 ]
  %4619 = zext i32 %4618 to i64
  %4620 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4619
  %4621 = load i32, i32* %4620, align 4, !noalias !614
  %4622 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4619
  store i32 %4621, i32* %4622, align 4, !alias.scope !614
  %4623 = add nuw nsw i32 %4618, 1, !spirv.Decorations !620
  %4624 = icmp eq i32 %4618, 0
  br i1 %4624, label %4617, label %4625, !llvm.loop !630

4625:                                             ; preds = %4617
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4626 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4627 = and i1 %3614, %4585
  br i1 %4627, label %4628, label %._crit_edge70.1.8

4628:                                             ; preds = %4625
  %4629 = shl i64 %4626, 32
  %4630 = ashr exact i64 %4629, 32
  %4631 = ashr i64 %4626, 32
  %4632 = mul nsw i64 %4630, %const_reg_qword9, !spirv.Decorations !610
  %4633 = add nsw i64 %4632, %4631, !spirv.Decorations !610
  %4634 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !612
  br i1 %48, label %4638, label %4635

4635:                                             ; preds = %4628
  %4636 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4633
  %4637 = addrspacecast float addrspace(4)* %4636 to float addrspace(1)*
  store float %4634, float addrspace(1)* %4637, align 4
  br label %._crit_edge70.1.8

4638:                                             ; preds = %4628
  %4639 = mul nsw i64 %4630, %const_reg_qword7, !spirv.Decorations !610
  %4640 = getelementptr float, float addrspace(4)* %66, i64 %4639
  %4641 = getelementptr float, float addrspace(4)* %4640, i64 %4631
  %4642 = addrspacecast float addrspace(4)* %4641 to float addrspace(1)*
  %4643 = load float, float addrspace(1)* %4642, align 4
  %4644 = fmul reassoc nsz arcp contract float %4643, %4, !spirv.Decorations !612
  %4645 = fadd reassoc nsz arcp contract float %4634, %4644, !spirv.Decorations !612
  %4646 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4633
  %4647 = addrspacecast float addrspace(4)* %4646 to float addrspace(1)*
  store float %4645, float addrspace(1)* %4647, align 4
  br label %._crit_edge70.1.8

._crit_edge70.1.8:                                ; preds = %4625, %4638, %4635
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4584, i32* %60, align 4, !noalias !614
  br label %4648

4648:                                             ; preds = %4648, %._crit_edge70.1.8
  %4649 = phi i32 [ 0, %._crit_edge70.1.8 ], [ %4654, %4648 ]
  %4650 = zext i32 %4649 to i64
  %4651 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4650
  %4652 = load i32, i32* %4651, align 4, !noalias !614
  %4653 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4650
  store i32 %4652, i32* %4653, align 4, !alias.scope !614
  %4654 = add nuw nsw i32 %4649, 1, !spirv.Decorations !620
  %4655 = icmp eq i32 %4649, 0
  br i1 %4655, label %4648, label %4656, !llvm.loop !630

4656:                                             ; preds = %4648
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4657 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4658 = and i1 %3647, %4585
  br i1 %4658, label %4659, label %._crit_edge70.2.8

4659:                                             ; preds = %4656
  %4660 = shl i64 %4657, 32
  %4661 = ashr exact i64 %4660, 32
  %4662 = ashr i64 %4657, 32
  %4663 = mul nsw i64 %4661, %const_reg_qword9, !spirv.Decorations !610
  %4664 = add nsw i64 %4663, %4662, !spirv.Decorations !610
  %4665 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !612
  br i1 %48, label %4669, label %4666

4666:                                             ; preds = %4659
  %4667 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4664
  %4668 = addrspacecast float addrspace(4)* %4667 to float addrspace(1)*
  store float %4665, float addrspace(1)* %4668, align 4
  br label %._crit_edge70.2.8

4669:                                             ; preds = %4659
  %4670 = mul nsw i64 %4661, %const_reg_qword7, !spirv.Decorations !610
  %4671 = getelementptr float, float addrspace(4)* %66, i64 %4670
  %4672 = getelementptr float, float addrspace(4)* %4671, i64 %4662
  %4673 = addrspacecast float addrspace(4)* %4672 to float addrspace(1)*
  %4674 = load float, float addrspace(1)* %4673, align 4
  %4675 = fmul reassoc nsz arcp contract float %4674, %4, !spirv.Decorations !612
  %4676 = fadd reassoc nsz arcp contract float %4665, %4675, !spirv.Decorations !612
  %4677 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4664
  %4678 = addrspacecast float addrspace(4)* %4677 to float addrspace(1)*
  store float %4676, float addrspace(1)* %4678, align 4
  br label %._crit_edge70.2.8

._crit_edge70.2.8:                                ; preds = %4656, %4669, %4666
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4584, i32* %60, align 4, !noalias !614
  br label %4679

4679:                                             ; preds = %4679, %._crit_edge70.2.8
  %4680 = phi i32 [ 0, %._crit_edge70.2.8 ], [ %4685, %4679 ]
  %4681 = zext i32 %4680 to i64
  %4682 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4681
  %4683 = load i32, i32* %4682, align 4, !noalias !614
  %4684 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4681
  store i32 %4683, i32* %4684, align 4, !alias.scope !614
  %4685 = add nuw nsw i32 %4680, 1, !spirv.Decorations !620
  %4686 = icmp eq i32 %4680, 0
  br i1 %4686, label %4679, label %4687, !llvm.loop !630

4687:                                             ; preds = %4679
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4688 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4689 = and i1 %3680, %4585
  br i1 %4689, label %4690, label %.preheader1.8

4690:                                             ; preds = %4687
  %4691 = shl i64 %4688, 32
  %4692 = ashr exact i64 %4691, 32
  %4693 = ashr i64 %4688, 32
  %4694 = mul nsw i64 %4692, %const_reg_qword9, !spirv.Decorations !610
  %4695 = add nsw i64 %4694, %4693, !spirv.Decorations !610
  %4696 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !612
  br i1 %48, label %4700, label %4697

4697:                                             ; preds = %4690
  %4698 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4695
  %4699 = addrspacecast float addrspace(4)* %4698 to float addrspace(1)*
  store float %4696, float addrspace(1)* %4699, align 4
  br label %.preheader1.8

4700:                                             ; preds = %4690
  %4701 = mul nsw i64 %4692, %const_reg_qword7, !spirv.Decorations !610
  %4702 = getelementptr float, float addrspace(4)* %66, i64 %4701
  %4703 = getelementptr float, float addrspace(4)* %4702, i64 %4693
  %4704 = addrspacecast float addrspace(4)* %4703 to float addrspace(1)*
  %4705 = load float, float addrspace(1)* %4704, align 4
  %4706 = fmul reassoc nsz arcp contract float %4705, %4, !spirv.Decorations !612
  %4707 = fadd reassoc nsz arcp contract float %4696, %4706, !spirv.Decorations !612
  %4708 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4695
  %4709 = addrspacecast float addrspace(4)* %4708 to float addrspace(1)*
  store float %4707, float addrspace(1)* %4709, align 4
  br label %.preheader1.8

.preheader1.8:                                    ; preds = %4687, %4700, %4697
  %4710 = or i32 %41, 9
  %4711 = icmp slt i32 %4710, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4710, i32* %60, align 4, !noalias !614
  br label %4712

4712:                                             ; preds = %4712, %.preheader1.8
  %4713 = phi i32 [ 0, %.preheader1.8 ], [ %4718, %4712 ]
  %4714 = zext i32 %4713 to i64
  %4715 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4714
  %4716 = load i32, i32* %4715, align 4, !noalias !614
  %4717 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4714
  store i32 %4716, i32* %4717, align 4, !alias.scope !614
  %4718 = add nuw nsw i32 %4713, 1, !spirv.Decorations !620
  %4719 = icmp eq i32 %4713, 0
  br i1 %4719, label %4712, label %4720, !llvm.loop !630

4720:                                             ; preds = %4712
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4721 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4722 = and i1 %3581, %4711
  br i1 %4722, label %4723, label %._crit_edge70.9

4723:                                             ; preds = %4720
  %4724 = shl i64 %4721, 32
  %4725 = ashr exact i64 %4724, 32
  %4726 = ashr i64 %4721, 32
  %4727 = mul nsw i64 %4725, %const_reg_qword9, !spirv.Decorations !610
  %4728 = add nsw i64 %4727, %4726, !spirv.Decorations !610
  %4729 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !612
  br i1 %48, label %4733, label %4730

4730:                                             ; preds = %4723
  %4731 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4728
  %4732 = addrspacecast float addrspace(4)* %4731 to float addrspace(1)*
  store float %4729, float addrspace(1)* %4732, align 4
  br label %._crit_edge70.9

4733:                                             ; preds = %4723
  %4734 = mul nsw i64 %4725, %const_reg_qword7, !spirv.Decorations !610
  %4735 = getelementptr float, float addrspace(4)* %66, i64 %4734
  %4736 = getelementptr float, float addrspace(4)* %4735, i64 %4726
  %4737 = addrspacecast float addrspace(4)* %4736 to float addrspace(1)*
  %4738 = load float, float addrspace(1)* %4737, align 4
  %4739 = fmul reassoc nsz arcp contract float %4738, %4, !spirv.Decorations !612
  %4740 = fadd reassoc nsz arcp contract float %4729, %4739, !spirv.Decorations !612
  %4741 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4728
  %4742 = addrspacecast float addrspace(4)* %4741 to float addrspace(1)*
  store float %4740, float addrspace(1)* %4742, align 4
  br label %._crit_edge70.9

._crit_edge70.9:                                  ; preds = %4720, %4733, %4730
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4710, i32* %60, align 4, !noalias !614
  br label %4743

4743:                                             ; preds = %4743, %._crit_edge70.9
  %4744 = phi i32 [ 0, %._crit_edge70.9 ], [ %4749, %4743 ]
  %4745 = zext i32 %4744 to i64
  %4746 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4745
  %4747 = load i32, i32* %4746, align 4, !noalias !614
  %4748 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4745
  store i32 %4747, i32* %4748, align 4, !alias.scope !614
  %4749 = add nuw nsw i32 %4744, 1, !spirv.Decorations !620
  %4750 = icmp eq i32 %4744, 0
  br i1 %4750, label %4743, label %4751, !llvm.loop !630

4751:                                             ; preds = %4743
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4752 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4753 = and i1 %3614, %4711
  br i1 %4753, label %4754, label %._crit_edge70.1.9

4754:                                             ; preds = %4751
  %4755 = shl i64 %4752, 32
  %4756 = ashr exact i64 %4755, 32
  %4757 = ashr i64 %4752, 32
  %4758 = mul nsw i64 %4756, %const_reg_qword9, !spirv.Decorations !610
  %4759 = add nsw i64 %4758, %4757, !spirv.Decorations !610
  %4760 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !612
  br i1 %48, label %4764, label %4761

4761:                                             ; preds = %4754
  %4762 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4759
  %4763 = addrspacecast float addrspace(4)* %4762 to float addrspace(1)*
  store float %4760, float addrspace(1)* %4763, align 4
  br label %._crit_edge70.1.9

4764:                                             ; preds = %4754
  %4765 = mul nsw i64 %4756, %const_reg_qword7, !spirv.Decorations !610
  %4766 = getelementptr float, float addrspace(4)* %66, i64 %4765
  %4767 = getelementptr float, float addrspace(4)* %4766, i64 %4757
  %4768 = addrspacecast float addrspace(4)* %4767 to float addrspace(1)*
  %4769 = load float, float addrspace(1)* %4768, align 4
  %4770 = fmul reassoc nsz arcp contract float %4769, %4, !spirv.Decorations !612
  %4771 = fadd reassoc nsz arcp contract float %4760, %4770, !spirv.Decorations !612
  %4772 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4759
  %4773 = addrspacecast float addrspace(4)* %4772 to float addrspace(1)*
  store float %4771, float addrspace(1)* %4773, align 4
  br label %._crit_edge70.1.9

._crit_edge70.1.9:                                ; preds = %4751, %4764, %4761
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4710, i32* %60, align 4, !noalias !614
  br label %4774

4774:                                             ; preds = %4774, %._crit_edge70.1.9
  %4775 = phi i32 [ 0, %._crit_edge70.1.9 ], [ %4780, %4774 ]
  %4776 = zext i32 %4775 to i64
  %4777 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4776
  %4778 = load i32, i32* %4777, align 4, !noalias !614
  %4779 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4776
  store i32 %4778, i32* %4779, align 4, !alias.scope !614
  %4780 = add nuw nsw i32 %4775, 1, !spirv.Decorations !620
  %4781 = icmp eq i32 %4775, 0
  br i1 %4781, label %4774, label %4782, !llvm.loop !630

4782:                                             ; preds = %4774
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4783 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4784 = and i1 %3647, %4711
  br i1 %4784, label %4785, label %._crit_edge70.2.9

4785:                                             ; preds = %4782
  %4786 = shl i64 %4783, 32
  %4787 = ashr exact i64 %4786, 32
  %4788 = ashr i64 %4783, 32
  %4789 = mul nsw i64 %4787, %const_reg_qword9, !spirv.Decorations !610
  %4790 = add nsw i64 %4789, %4788, !spirv.Decorations !610
  %4791 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !612
  br i1 %48, label %4795, label %4792

4792:                                             ; preds = %4785
  %4793 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4790
  %4794 = addrspacecast float addrspace(4)* %4793 to float addrspace(1)*
  store float %4791, float addrspace(1)* %4794, align 4
  br label %._crit_edge70.2.9

4795:                                             ; preds = %4785
  %4796 = mul nsw i64 %4787, %const_reg_qword7, !spirv.Decorations !610
  %4797 = getelementptr float, float addrspace(4)* %66, i64 %4796
  %4798 = getelementptr float, float addrspace(4)* %4797, i64 %4788
  %4799 = addrspacecast float addrspace(4)* %4798 to float addrspace(1)*
  %4800 = load float, float addrspace(1)* %4799, align 4
  %4801 = fmul reassoc nsz arcp contract float %4800, %4, !spirv.Decorations !612
  %4802 = fadd reassoc nsz arcp contract float %4791, %4801, !spirv.Decorations !612
  %4803 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4790
  %4804 = addrspacecast float addrspace(4)* %4803 to float addrspace(1)*
  store float %4802, float addrspace(1)* %4804, align 4
  br label %._crit_edge70.2.9

._crit_edge70.2.9:                                ; preds = %4782, %4795, %4792
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4710, i32* %60, align 4, !noalias !614
  br label %4805

4805:                                             ; preds = %4805, %._crit_edge70.2.9
  %4806 = phi i32 [ 0, %._crit_edge70.2.9 ], [ %4811, %4805 ]
  %4807 = zext i32 %4806 to i64
  %4808 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4807
  %4809 = load i32, i32* %4808, align 4, !noalias !614
  %4810 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4807
  store i32 %4809, i32* %4810, align 4, !alias.scope !614
  %4811 = add nuw nsw i32 %4806, 1, !spirv.Decorations !620
  %4812 = icmp eq i32 %4806, 0
  br i1 %4812, label %4805, label %4813, !llvm.loop !630

4813:                                             ; preds = %4805
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4814 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4815 = and i1 %3680, %4711
  br i1 %4815, label %4816, label %.preheader1.9

4816:                                             ; preds = %4813
  %4817 = shl i64 %4814, 32
  %4818 = ashr exact i64 %4817, 32
  %4819 = ashr i64 %4814, 32
  %4820 = mul nsw i64 %4818, %const_reg_qword9, !spirv.Decorations !610
  %4821 = add nsw i64 %4820, %4819, !spirv.Decorations !610
  %4822 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !612
  br i1 %48, label %4826, label %4823

4823:                                             ; preds = %4816
  %4824 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4821
  %4825 = addrspacecast float addrspace(4)* %4824 to float addrspace(1)*
  store float %4822, float addrspace(1)* %4825, align 4
  br label %.preheader1.9

4826:                                             ; preds = %4816
  %4827 = mul nsw i64 %4818, %const_reg_qword7, !spirv.Decorations !610
  %4828 = getelementptr float, float addrspace(4)* %66, i64 %4827
  %4829 = getelementptr float, float addrspace(4)* %4828, i64 %4819
  %4830 = addrspacecast float addrspace(4)* %4829 to float addrspace(1)*
  %4831 = load float, float addrspace(1)* %4830, align 4
  %4832 = fmul reassoc nsz arcp contract float %4831, %4, !spirv.Decorations !612
  %4833 = fadd reassoc nsz arcp contract float %4822, %4832, !spirv.Decorations !612
  %4834 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4821
  %4835 = addrspacecast float addrspace(4)* %4834 to float addrspace(1)*
  store float %4833, float addrspace(1)* %4835, align 4
  br label %.preheader1.9

.preheader1.9:                                    ; preds = %4813, %4826, %4823
  %4836 = or i32 %41, 10
  %4837 = icmp slt i32 %4836, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4836, i32* %60, align 4, !noalias !614
  br label %4838

4838:                                             ; preds = %4838, %.preheader1.9
  %4839 = phi i32 [ 0, %.preheader1.9 ], [ %4844, %4838 ]
  %4840 = zext i32 %4839 to i64
  %4841 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4840
  %4842 = load i32, i32* %4841, align 4, !noalias !614
  %4843 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4840
  store i32 %4842, i32* %4843, align 4, !alias.scope !614
  %4844 = add nuw nsw i32 %4839, 1, !spirv.Decorations !620
  %4845 = icmp eq i32 %4839, 0
  br i1 %4845, label %4838, label %4846, !llvm.loop !630

4846:                                             ; preds = %4838
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4847 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4848 = and i1 %3581, %4837
  br i1 %4848, label %4849, label %._crit_edge70.10

4849:                                             ; preds = %4846
  %4850 = shl i64 %4847, 32
  %4851 = ashr exact i64 %4850, 32
  %4852 = ashr i64 %4847, 32
  %4853 = mul nsw i64 %4851, %const_reg_qword9, !spirv.Decorations !610
  %4854 = add nsw i64 %4853, %4852, !spirv.Decorations !610
  %4855 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !612
  br i1 %48, label %4859, label %4856

4856:                                             ; preds = %4849
  %4857 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4854
  %4858 = addrspacecast float addrspace(4)* %4857 to float addrspace(1)*
  store float %4855, float addrspace(1)* %4858, align 4
  br label %._crit_edge70.10

4859:                                             ; preds = %4849
  %4860 = mul nsw i64 %4851, %const_reg_qword7, !spirv.Decorations !610
  %4861 = getelementptr float, float addrspace(4)* %66, i64 %4860
  %4862 = getelementptr float, float addrspace(4)* %4861, i64 %4852
  %4863 = addrspacecast float addrspace(4)* %4862 to float addrspace(1)*
  %4864 = load float, float addrspace(1)* %4863, align 4
  %4865 = fmul reassoc nsz arcp contract float %4864, %4, !spirv.Decorations !612
  %4866 = fadd reassoc nsz arcp contract float %4855, %4865, !spirv.Decorations !612
  %4867 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4854
  %4868 = addrspacecast float addrspace(4)* %4867 to float addrspace(1)*
  store float %4866, float addrspace(1)* %4868, align 4
  br label %._crit_edge70.10

._crit_edge70.10:                                 ; preds = %4846, %4859, %4856
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4836, i32* %60, align 4, !noalias !614
  br label %4869

4869:                                             ; preds = %4869, %._crit_edge70.10
  %4870 = phi i32 [ 0, %._crit_edge70.10 ], [ %4875, %4869 ]
  %4871 = zext i32 %4870 to i64
  %4872 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4871
  %4873 = load i32, i32* %4872, align 4, !noalias !614
  %4874 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4871
  store i32 %4873, i32* %4874, align 4, !alias.scope !614
  %4875 = add nuw nsw i32 %4870, 1, !spirv.Decorations !620
  %4876 = icmp eq i32 %4870, 0
  br i1 %4876, label %4869, label %4877, !llvm.loop !630

4877:                                             ; preds = %4869
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4878 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4879 = and i1 %3614, %4837
  br i1 %4879, label %4880, label %._crit_edge70.1.10

4880:                                             ; preds = %4877
  %4881 = shl i64 %4878, 32
  %4882 = ashr exact i64 %4881, 32
  %4883 = ashr i64 %4878, 32
  %4884 = mul nsw i64 %4882, %const_reg_qword9, !spirv.Decorations !610
  %4885 = add nsw i64 %4884, %4883, !spirv.Decorations !610
  %4886 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !612
  br i1 %48, label %4890, label %4887

4887:                                             ; preds = %4880
  %4888 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4885
  %4889 = addrspacecast float addrspace(4)* %4888 to float addrspace(1)*
  store float %4886, float addrspace(1)* %4889, align 4
  br label %._crit_edge70.1.10

4890:                                             ; preds = %4880
  %4891 = mul nsw i64 %4882, %const_reg_qword7, !spirv.Decorations !610
  %4892 = getelementptr float, float addrspace(4)* %66, i64 %4891
  %4893 = getelementptr float, float addrspace(4)* %4892, i64 %4883
  %4894 = addrspacecast float addrspace(4)* %4893 to float addrspace(1)*
  %4895 = load float, float addrspace(1)* %4894, align 4
  %4896 = fmul reassoc nsz arcp contract float %4895, %4, !spirv.Decorations !612
  %4897 = fadd reassoc nsz arcp contract float %4886, %4896, !spirv.Decorations !612
  %4898 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4885
  %4899 = addrspacecast float addrspace(4)* %4898 to float addrspace(1)*
  store float %4897, float addrspace(1)* %4899, align 4
  br label %._crit_edge70.1.10

._crit_edge70.1.10:                               ; preds = %4877, %4890, %4887
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4836, i32* %60, align 4, !noalias !614
  br label %4900

4900:                                             ; preds = %4900, %._crit_edge70.1.10
  %4901 = phi i32 [ 0, %._crit_edge70.1.10 ], [ %4906, %4900 ]
  %4902 = zext i32 %4901 to i64
  %4903 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4902
  %4904 = load i32, i32* %4903, align 4, !noalias !614
  %4905 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4902
  store i32 %4904, i32* %4905, align 4, !alias.scope !614
  %4906 = add nuw nsw i32 %4901, 1, !spirv.Decorations !620
  %4907 = icmp eq i32 %4901, 0
  br i1 %4907, label %4900, label %4908, !llvm.loop !630

4908:                                             ; preds = %4900
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4909 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4910 = and i1 %3647, %4837
  br i1 %4910, label %4911, label %._crit_edge70.2.10

4911:                                             ; preds = %4908
  %4912 = shl i64 %4909, 32
  %4913 = ashr exact i64 %4912, 32
  %4914 = ashr i64 %4909, 32
  %4915 = mul nsw i64 %4913, %const_reg_qword9, !spirv.Decorations !610
  %4916 = add nsw i64 %4915, %4914, !spirv.Decorations !610
  %4917 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !612
  br i1 %48, label %4921, label %4918

4918:                                             ; preds = %4911
  %4919 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4916
  %4920 = addrspacecast float addrspace(4)* %4919 to float addrspace(1)*
  store float %4917, float addrspace(1)* %4920, align 4
  br label %._crit_edge70.2.10

4921:                                             ; preds = %4911
  %4922 = mul nsw i64 %4913, %const_reg_qword7, !spirv.Decorations !610
  %4923 = getelementptr float, float addrspace(4)* %66, i64 %4922
  %4924 = getelementptr float, float addrspace(4)* %4923, i64 %4914
  %4925 = addrspacecast float addrspace(4)* %4924 to float addrspace(1)*
  %4926 = load float, float addrspace(1)* %4925, align 4
  %4927 = fmul reassoc nsz arcp contract float %4926, %4, !spirv.Decorations !612
  %4928 = fadd reassoc nsz arcp contract float %4917, %4927, !spirv.Decorations !612
  %4929 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4916
  %4930 = addrspacecast float addrspace(4)* %4929 to float addrspace(1)*
  store float %4928, float addrspace(1)* %4930, align 4
  br label %._crit_edge70.2.10

._crit_edge70.2.10:                               ; preds = %4908, %4921, %4918
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4836, i32* %60, align 4, !noalias !614
  br label %4931

4931:                                             ; preds = %4931, %._crit_edge70.2.10
  %4932 = phi i32 [ 0, %._crit_edge70.2.10 ], [ %4937, %4931 ]
  %4933 = zext i32 %4932 to i64
  %4934 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4933
  %4935 = load i32, i32* %4934, align 4, !noalias !614
  %4936 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4933
  store i32 %4935, i32* %4936, align 4, !alias.scope !614
  %4937 = add nuw nsw i32 %4932, 1, !spirv.Decorations !620
  %4938 = icmp eq i32 %4932, 0
  br i1 %4938, label %4931, label %4939, !llvm.loop !630

4939:                                             ; preds = %4931
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4940 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4941 = and i1 %3680, %4837
  br i1 %4941, label %4942, label %.preheader1.10

4942:                                             ; preds = %4939
  %4943 = shl i64 %4940, 32
  %4944 = ashr exact i64 %4943, 32
  %4945 = ashr i64 %4940, 32
  %4946 = mul nsw i64 %4944, %const_reg_qword9, !spirv.Decorations !610
  %4947 = add nsw i64 %4946, %4945, !spirv.Decorations !610
  %4948 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !612
  br i1 %48, label %4952, label %4949

4949:                                             ; preds = %4942
  %4950 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4947
  %4951 = addrspacecast float addrspace(4)* %4950 to float addrspace(1)*
  store float %4948, float addrspace(1)* %4951, align 4
  br label %.preheader1.10

4952:                                             ; preds = %4942
  %4953 = mul nsw i64 %4944, %const_reg_qword7, !spirv.Decorations !610
  %4954 = getelementptr float, float addrspace(4)* %66, i64 %4953
  %4955 = getelementptr float, float addrspace(4)* %4954, i64 %4945
  %4956 = addrspacecast float addrspace(4)* %4955 to float addrspace(1)*
  %4957 = load float, float addrspace(1)* %4956, align 4
  %4958 = fmul reassoc nsz arcp contract float %4957, %4, !spirv.Decorations !612
  %4959 = fadd reassoc nsz arcp contract float %4948, %4958, !spirv.Decorations !612
  %4960 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4947
  %4961 = addrspacecast float addrspace(4)* %4960 to float addrspace(1)*
  store float %4959, float addrspace(1)* %4961, align 4
  br label %.preheader1.10

.preheader1.10:                                   ; preds = %4939, %4952, %4949
  %4962 = or i32 %41, 11
  %4963 = icmp slt i32 %4962, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %4962, i32* %60, align 4, !noalias !614
  br label %4964

4964:                                             ; preds = %4964, %.preheader1.10
  %4965 = phi i32 [ 0, %.preheader1.10 ], [ %4970, %4964 ]
  %4966 = zext i32 %4965 to i64
  %4967 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4966
  %4968 = load i32, i32* %4967, align 4, !noalias !614
  %4969 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4966
  store i32 %4968, i32* %4969, align 4, !alias.scope !614
  %4970 = add nuw nsw i32 %4965, 1, !spirv.Decorations !620
  %4971 = icmp eq i32 %4965, 0
  br i1 %4971, label %4964, label %4972, !llvm.loop !630

4972:                                             ; preds = %4964
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4973 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4974 = and i1 %3581, %4963
  br i1 %4974, label %4975, label %._crit_edge70.11

4975:                                             ; preds = %4972
  %4976 = shl i64 %4973, 32
  %4977 = ashr exact i64 %4976, 32
  %4978 = ashr i64 %4973, 32
  %4979 = mul nsw i64 %4977, %const_reg_qword9, !spirv.Decorations !610
  %4980 = add nsw i64 %4979, %4978, !spirv.Decorations !610
  %4981 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !612
  br i1 %48, label %4985, label %4982

4982:                                             ; preds = %4975
  %4983 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4980
  %4984 = addrspacecast float addrspace(4)* %4983 to float addrspace(1)*
  store float %4981, float addrspace(1)* %4984, align 4
  br label %._crit_edge70.11

4985:                                             ; preds = %4975
  %4986 = mul nsw i64 %4977, %const_reg_qword7, !spirv.Decorations !610
  %4987 = getelementptr float, float addrspace(4)* %66, i64 %4986
  %4988 = getelementptr float, float addrspace(4)* %4987, i64 %4978
  %4989 = addrspacecast float addrspace(4)* %4988 to float addrspace(1)*
  %4990 = load float, float addrspace(1)* %4989, align 4
  %4991 = fmul reassoc nsz arcp contract float %4990, %4, !spirv.Decorations !612
  %4992 = fadd reassoc nsz arcp contract float %4981, %4991, !spirv.Decorations !612
  %4993 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4980
  %4994 = addrspacecast float addrspace(4)* %4993 to float addrspace(1)*
  store float %4992, float addrspace(1)* %4994, align 4
  br label %._crit_edge70.11

._crit_edge70.11:                                 ; preds = %4972, %4985, %4982
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %4962, i32* %60, align 4, !noalias !614
  br label %4995

4995:                                             ; preds = %4995, %._crit_edge70.11
  %4996 = phi i32 [ 0, %._crit_edge70.11 ], [ %5001, %4995 ]
  %4997 = zext i32 %4996 to i64
  %4998 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4997
  %4999 = load i32, i32* %4998, align 4, !noalias !614
  %5000 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4997
  store i32 %4999, i32* %5000, align 4, !alias.scope !614
  %5001 = add nuw nsw i32 %4996, 1, !spirv.Decorations !620
  %5002 = icmp eq i32 %4996, 0
  br i1 %5002, label %4995, label %5003, !llvm.loop !630

5003:                                             ; preds = %4995
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5004 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5005 = and i1 %3614, %4963
  br i1 %5005, label %5006, label %._crit_edge70.1.11

5006:                                             ; preds = %5003
  %5007 = shl i64 %5004, 32
  %5008 = ashr exact i64 %5007, 32
  %5009 = ashr i64 %5004, 32
  %5010 = mul nsw i64 %5008, %const_reg_qword9, !spirv.Decorations !610
  %5011 = add nsw i64 %5010, %5009, !spirv.Decorations !610
  %5012 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !612
  br i1 %48, label %5016, label %5013

5013:                                             ; preds = %5006
  %5014 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5011
  %5015 = addrspacecast float addrspace(4)* %5014 to float addrspace(1)*
  store float %5012, float addrspace(1)* %5015, align 4
  br label %._crit_edge70.1.11

5016:                                             ; preds = %5006
  %5017 = mul nsw i64 %5008, %const_reg_qword7, !spirv.Decorations !610
  %5018 = getelementptr float, float addrspace(4)* %66, i64 %5017
  %5019 = getelementptr float, float addrspace(4)* %5018, i64 %5009
  %5020 = addrspacecast float addrspace(4)* %5019 to float addrspace(1)*
  %5021 = load float, float addrspace(1)* %5020, align 4
  %5022 = fmul reassoc nsz arcp contract float %5021, %4, !spirv.Decorations !612
  %5023 = fadd reassoc nsz arcp contract float %5012, %5022, !spirv.Decorations !612
  %5024 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5011
  %5025 = addrspacecast float addrspace(4)* %5024 to float addrspace(1)*
  store float %5023, float addrspace(1)* %5025, align 4
  br label %._crit_edge70.1.11

._crit_edge70.1.11:                               ; preds = %5003, %5016, %5013
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %4962, i32* %60, align 4, !noalias !614
  br label %5026

5026:                                             ; preds = %5026, %._crit_edge70.1.11
  %5027 = phi i32 [ 0, %._crit_edge70.1.11 ], [ %5032, %5026 ]
  %5028 = zext i32 %5027 to i64
  %5029 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5028
  %5030 = load i32, i32* %5029, align 4, !noalias !614
  %5031 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5028
  store i32 %5030, i32* %5031, align 4, !alias.scope !614
  %5032 = add nuw nsw i32 %5027, 1, !spirv.Decorations !620
  %5033 = icmp eq i32 %5027, 0
  br i1 %5033, label %5026, label %5034, !llvm.loop !630

5034:                                             ; preds = %5026
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5035 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5036 = and i1 %3647, %4963
  br i1 %5036, label %5037, label %._crit_edge70.2.11

5037:                                             ; preds = %5034
  %5038 = shl i64 %5035, 32
  %5039 = ashr exact i64 %5038, 32
  %5040 = ashr i64 %5035, 32
  %5041 = mul nsw i64 %5039, %const_reg_qword9, !spirv.Decorations !610
  %5042 = add nsw i64 %5041, %5040, !spirv.Decorations !610
  %5043 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !612
  br i1 %48, label %5047, label %5044

5044:                                             ; preds = %5037
  %5045 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5042
  %5046 = addrspacecast float addrspace(4)* %5045 to float addrspace(1)*
  store float %5043, float addrspace(1)* %5046, align 4
  br label %._crit_edge70.2.11

5047:                                             ; preds = %5037
  %5048 = mul nsw i64 %5039, %const_reg_qword7, !spirv.Decorations !610
  %5049 = getelementptr float, float addrspace(4)* %66, i64 %5048
  %5050 = getelementptr float, float addrspace(4)* %5049, i64 %5040
  %5051 = addrspacecast float addrspace(4)* %5050 to float addrspace(1)*
  %5052 = load float, float addrspace(1)* %5051, align 4
  %5053 = fmul reassoc nsz arcp contract float %5052, %4, !spirv.Decorations !612
  %5054 = fadd reassoc nsz arcp contract float %5043, %5053, !spirv.Decorations !612
  %5055 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5042
  %5056 = addrspacecast float addrspace(4)* %5055 to float addrspace(1)*
  store float %5054, float addrspace(1)* %5056, align 4
  br label %._crit_edge70.2.11

._crit_edge70.2.11:                               ; preds = %5034, %5047, %5044
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %4962, i32* %60, align 4, !noalias !614
  br label %5057

5057:                                             ; preds = %5057, %._crit_edge70.2.11
  %5058 = phi i32 [ 0, %._crit_edge70.2.11 ], [ %5063, %5057 ]
  %5059 = zext i32 %5058 to i64
  %5060 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5059
  %5061 = load i32, i32* %5060, align 4, !noalias !614
  %5062 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5059
  store i32 %5061, i32* %5062, align 4, !alias.scope !614
  %5063 = add nuw nsw i32 %5058, 1, !spirv.Decorations !620
  %5064 = icmp eq i32 %5058, 0
  br i1 %5064, label %5057, label %5065, !llvm.loop !630

5065:                                             ; preds = %5057
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5066 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5067 = and i1 %3680, %4963
  br i1 %5067, label %5068, label %.preheader1.11

5068:                                             ; preds = %5065
  %5069 = shl i64 %5066, 32
  %5070 = ashr exact i64 %5069, 32
  %5071 = ashr i64 %5066, 32
  %5072 = mul nsw i64 %5070, %const_reg_qword9, !spirv.Decorations !610
  %5073 = add nsw i64 %5072, %5071, !spirv.Decorations !610
  %5074 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !612
  br i1 %48, label %5078, label %5075

5075:                                             ; preds = %5068
  %5076 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5073
  %5077 = addrspacecast float addrspace(4)* %5076 to float addrspace(1)*
  store float %5074, float addrspace(1)* %5077, align 4
  br label %.preheader1.11

5078:                                             ; preds = %5068
  %5079 = mul nsw i64 %5070, %const_reg_qword7, !spirv.Decorations !610
  %5080 = getelementptr float, float addrspace(4)* %66, i64 %5079
  %5081 = getelementptr float, float addrspace(4)* %5080, i64 %5071
  %5082 = addrspacecast float addrspace(4)* %5081 to float addrspace(1)*
  %5083 = load float, float addrspace(1)* %5082, align 4
  %5084 = fmul reassoc nsz arcp contract float %5083, %4, !spirv.Decorations !612
  %5085 = fadd reassoc nsz arcp contract float %5074, %5084, !spirv.Decorations !612
  %5086 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5073
  %5087 = addrspacecast float addrspace(4)* %5086 to float addrspace(1)*
  store float %5085, float addrspace(1)* %5087, align 4
  br label %.preheader1.11

.preheader1.11:                                   ; preds = %5065, %5078, %5075
  %5088 = or i32 %41, 12
  %5089 = icmp slt i32 %5088, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %5088, i32* %60, align 4, !noalias !614
  br label %5090

5090:                                             ; preds = %5090, %.preheader1.11
  %5091 = phi i32 [ 0, %.preheader1.11 ], [ %5096, %5090 ]
  %5092 = zext i32 %5091 to i64
  %5093 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5092
  %5094 = load i32, i32* %5093, align 4, !noalias !614
  %5095 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5092
  store i32 %5094, i32* %5095, align 4, !alias.scope !614
  %5096 = add nuw nsw i32 %5091, 1, !spirv.Decorations !620
  %5097 = icmp eq i32 %5091, 0
  br i1 %5097, label %5090, label %5098, !llvm.loop !630

5098:                                             ; preds = %5090
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5099 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5100 = and i1 %3581, %5089
  br i1 %5100, label %5101, label %._crit_edge70.12

5101:                                             ; preds = %5098
  %5102 = shl i64 %5099, 32
  %5103 = ashr exact i64 %5102, 32
  %5104 = ashr i64 %5099, 32
  %5105 = mul nsw i64 %5103, %const_reg_qword9, !spirv.Decorations !610
  %5106 = add nsw i64 %5105, %5104, !spirv.Decorations !610
  %5107 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !612
  br i1 %48, label %5111, label %5108

5108:                                             ; preds = %5101
  %5109 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5106
  %5110 = addrspacecast float addrspace(4)* %5109 to float addrspace(1)*
  store float %5107, float addrspace(1)* %5110, align 4
  br label %._crit_edge70.12

5111:                                             ; preds = %5101
  %5112 = mul nsw i64 %5103, %const_reg_qword7, !spirv.Decorations !610
  %5113 = getelementptr float, float addrspace(4)* %66, i64 %5112
  %5114 = getelementptr float, float addrspace(4)* %5113, i64 %5104
  %5115 = addrspacecast float addrspace(4)* %5114 to float addrspace(1)*
  %5116 = load float, float addrspace(1)* %5115, align 4
  %5117 = fmul reassoc nsz arcp contract float %5116, %4, !spirv.Decorations !612
  %5118 = fadd reassoc nsz arcp contract float %5107, %5117, !spirv.Decorations !612
  %5119 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5106
  %5120 = addrspacecast float addrspace(4)* %5119 to float addrspace(1)*
  store float %5118, float addrspace(1)* %5120, align 4
  br label %._crit_edge70.12

._crit_edge70.12:                                 ; preds = %5098, %5111, %5108
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %5088, i32* %60, align 4, !noalias !614
  br label %5121

5121:                                             ; preds = %5121, %._crit_edge70.12
  %5122 = phi i32 [ 0, %._crit_edge70.12 ], [ %5127, %5121 ]
  %5123 = zext i32 %5122 to i64
  %5124 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5123
  %5125 = load i32, i32* %5124, align 4, !noalias !614
  %5126 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5123
  store i32 %5125, i32* %5126, align 4, !alias.scope !614
  %5127 = add nuw nsw i32 %5122, 1, !spirv.Decorations !620
  %5128 = icmp eq i32 %5122, 0
  br i1 %5128, label %5121, label %5129, !llvm.loop !630

5129:                                             ; preds = %5121
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5130 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5131 = and i1 %3614, %5089
  br i1 %5131, label %5132, label %._crit_edge70.1.12

5132:                                             ; preds = %5129
  %5133 = shl i64 %5130, 32
  %5134 = ashr exact i64 %5133, 32
  %5135 = ashr i64 %5130, 32
  %5136 = mul nsw i64 %5134, %const_reg_qword9, !spirv.Decorations !610
  %5137 = add nsw i64 %5136, %5135, !spirv.Decorations !610
  %5138 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !612
  br i1 %48, label %5142, label %5139

5139:                                             ; preds = %5132
  %5140 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5137
  %5141 = addrspacecast float addrspace(4)* %5140 to float addrspace(1)*
  store float %5138, float addrspace(1)* %5141, align 4
  br label %._crit_edge70.1.12

5142:                                             ; preds = %5132
  %5143 = mul nsw i64 %5134, %const_reg_qword7, !spirv.Decorations !610
  %5144 = getelementptr float, float addrspace(4)* %66, i64 %5143
  %5145 = getelementptr float, float addrspace(4)* %5144, i64 %5135
  %5146 = addrspacecast float addrspace(4)* %5145 to float addrspace(1)*
  %5147 = load float, float addrspace(1)* %5146, align 4
  %5148 = fmul reassoc nsz arcp contract float %5147, %4, !spirv.Decorations !612
  %5149 = fadd reassoc nsz arcp contract float %5138, %5148, !spirv.Decorations !612
  %5150 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5137
  %5151 = addrspacecast float addrspace(4)* %5150 to float addrspace(1)*
  store float %5149, float addrspace(1)* %5151, align 4
  br label %._crit_edge70.1.12

._crit_edge70.1.12:                               ; preds = %5129, %5142, %5139
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %5088, i32* %60, align 4, !noalias !614
  br label %5152

5152:                                             ; preds = %5152, %._crit_edge70.1.12
  %5153 = phi i32 [ 0, %._crit_edge70.1.12 ], [ %5158, %5152 ]
  %5154 = zext i32 %5153 to i64
  %5155 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5154
  %5156 = load i32, i32* %5155, align 4, !noalias !614
  %5157 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5154
  store i32 %5156, i32* %5157, align 4, !alias.scope !614
  %5158 = add nuw nsw i32 %5153, 1, !spirv.Decorations !620
  %5159 = icmp eq i32 %5153, 0
  br i1 %5159, label %5152, label %5160, !llvm.loop !630

5160:                                             ; preds = %5152
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5161 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5162 = and i1 %3647, %5089
  br i1 %5162, label %5163, label %._crit_edge70.2.12

5163:                                             ; preds = %5160
  %5164 = shl i64 %5161, 32
  %5165 = ashr exact i64 %5164, 32
  %5166 = ashr i64 %5161, 32
  %5167 = mul nsw i64 %5165, %const_reg_qword9, !spirv.Decorations !610
  %5168 = add nsw i64 %5167, %5166, !spirv.Decorations !610
  %5169 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !612
  br i1 %48, label %5173, label %5170

5170:                                             ; preds = %5163
  %5171 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5168
  %5172 = addrspacecast float addrspace(4)* %5171 to float addrspace(1)*
  store float %5169, float addrspace(1)* %5172, align 4
  br label %._crit_edge70.2.12

5173:                                             ; preds = %5163
  %5174 = mul nsw i64 %5165, %const_reg_qword7, !spirv.Decorations !610
  %5175 = getelementptr float, float addrspace(4)* %66, i64 %5174
  %5176 = getelementptr float, float addrspace(4)* %5175, i64 %5166
  %5177 = addrspacecast float addrspace(4)* %5176 to float addrspace(1)*
  %5178 = load float, float addrspace(1)* %5177, align 4
  %5179 = fmul reassoc nsz arcp contract float %5178, %4, !spirv.Decorations !612
  %5180 = fadd reassoc nsz arcp contract float %5169, %5179, !spirv.Decorations !612
  %5181 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5168
  %5182 = addrspacecast float addrspace(4)* %5181 to float addrspace(1)*
  store float %5180, float addrspace(1)* %5182, align 4
  br label %._crit_edge70.2.12

._crit_edge70.2.12:                               ; preds = %5160, %5173, %5170
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %5088, i32* %60, align 4, !noalias !614
  br label %5183

5183:                                             ; preds = %5183, %._crit_edge70.2.12
  %5184 = phi i32 [ 0, %._crit_edge70.2.12 ], [ %5189, %5183 ]
  %5185 = zext i32 %5184 to i64
  %5186 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5185
  %5187 = load i32, i32* %5186, align 4, !noalias !614
  %5188 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5185
  store i32 %5187, i32* %5188, align 4, !alias.scope !614
  %5189 = add nuw nsw i32 %5184, 1, !spirv.Decorations !620
  %5190 = icmp eq i32 %5184, 0
  br i1 %5190, label %5183, label %5191, !llvm.loop !630

5191:                                             ; preds = %5183
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5192 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5193 = and i1 %3680, %5089
  br i1 %5193, label %5194, label %.preheader1.12

5194:                                             ; preds = %5191
  %5195 = shl i64 %5192, 32
  %5196 = ashr exact i64 %5195, 32
  %5197 = ashr i64 %5192, 32
  %5198 = mul nsw i64 %5196, %const_reg_qword9, !spirv.Decorations !610
  %5199 = add nsw i64 %5198, %5197, !spirv.Decorations !610
  %5200 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !612
  br i1 %48, label %5204, label %5201

5201:                                             ; preds = %5194
  %5202 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5199
  %5203 = addrspacecast float addrspace(4)* %5202 to float addrspace(1)*
  store float %5200, float addrspace(1)* %5203, align 4
  br label %.preheader1.12

5204:                                             ; preds = %5194
  %5205 = mul nsw i64 %5196, %const_reg_qword7, !spirv.Decorations !610
  %5206 = getelementptr float, float addrspace(4)* %66, i64 %5205
  %5207 = getelementptr float, float addrspace(4)* %5206, i64 %5197
  %5208 = addrspacecast float addrspace(4)* %5207 to float addrspace(1)*
  %5209 = load float, float addrspace(1)* %5208, align 4
  %5210 = fmul reassoc nsz arcp contract float %5209, %4, !spirv.Decorations !612
  %5211 = fadd reassoc nsz arcp contract float %5200, %5210, !spirv.Decorations !612
  %5212 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5199
  %5213 = addrspacecast float addrspace(4)* %5212 to float addrspace(1)*
  store float %5211, float addrspace(1)* %5213, align 4
  br label %.preheader1.12

.preheader1.12:                                   ; preds = %5191, %5204, %5201
  %5214 = or i32 %41, 13
  %5215 = icmp slt i32 %5214, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %5214, i32* %60, align 4, !noalias !614
  br label %5216

5216:                                             ; preds = %5216, %.preheader1.12
  %5217 = phi i32 [ 0, %.preheader1.12 ], [ %5222, %5216 ]
  %5218 = zext i32 %5217 to i64
  %5219 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5218
  %5220 = load i32, i32* %5219, align 4, !noalias !614
  %5221 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5218
  store i32 %5220, i32* %5221, align 4, !alias.scope !614
  %5222 = add nuw nsw i32 %5217, 1, !spirv.Decorations !620
  %5223 = icmp eq i32 %5217, 0
  br i1 %5223, label %5216, label %5224, !llvm.loop !630

5224:                                             ; preds = %5216
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5225 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5226 = and i1 %3581, %5215
  br i1 %5226, label %5227, label %._crit_edge70.13

5227:                                             ; preds = %5224
  %5228 = shl i64 %5225, 32
  %5229 = ashr exact i64 %5228, 32
  %5230 = ashr i64 %5225, 32
  %5231 = mul nsw i64 %5229, %const_reg_qword9, !spirv.Decorations !610
  %5232 = add nsw i64 %5231, %5230, !spirv.Decorations !610
  %5233 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !612
  br i1 %48, label %5237, label %5234

5234:                                             ; preds = %5227
  %5235 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5232
  %5236 = addrspacecast float addrspace(4)* %5235 to float addrspace(1)*
  store float %5233, float addrspace(1)* %5236, align 4
  br label %._crit_edge70.13

5237:                                             ; preds = %5227
  %5238 = mul nsw i64 %5229, %const_reg_qword7, !spirv.Decorations !610
  %5239 = getelementptr float, float addrspace(4)* %66, i64 %5238
  %5240 = getelementptr float, float addrspace(4)* %5239, i64 %5230
  %5241 = addrspacecast float addrspace(4)* %5240 to float addrspace(1)*
  %5242 = load float, float addrspace(1)* %5241, align 4
  %5243 = fmul reassoc nsz arcp contract float %5242, %4, !spirv.Decorations !612
  %5244 = fadd reassoc nsz arcp contract float %5233, %5243, !spirv.Decorations !612
  %5245 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5232
  %5246 = addrspacecast float addrspace(4)* %5245 to float addrspace(1)*
  store float %5244, float addrspace(1)* %5246, align 4
  br label %._crit_edge70.13

._crit_edge70.13:                                 ; preds = %5224, %5237, %5234
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %5214, i32* %60, align 4, !noalias !614
  br label %5247

5247:                                             ; preds = %5247, %._crit_edge70.13
  %5248 = phi i32 [ 0, %._crit_edge70.13 ], [ %5253, %5247 ]
  %5249 = zext i32 %5248 to i64
  %5250 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5249
  %5251 = load i32, i32* %5250, align 4, !noalias !614
  %5252 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5249
  store i32 %5251, i32* %5252, align 4, !alias.scope !614
  %5253 = add nuw nsw i32 %5248, 1, !spirv.Decorations !620
  %5254 = icmp eq i32 %5248, 0
  br i1 %5254, label %5247, label %5255, !llvm.loop !630

5255:                                             ; preds = %5247
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5256 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5257 = and i1 %3614, %5215
  br i1 %5257, label %5258, label %._crit_edge70.1.13

5258:                                             ; preds = %5255
  %5259 = shl i64 %5256, 32
  %5260 = ashr exact i64 %5259, 32
  %5261 = ashr i64 %5256, 32
  %5262 = mul nsw i64 %5260, %const_reg_qword9, !spirv.Decorations !610
  %5263 = add nsw i64 %5262, %5261, !spirv.Decorations !610
  %5264 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !612
  br i1 %48, label %5268, label %5265

5265:                                             ; preds = %5258
  %5266 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5263
  %5267 = addrspacecast float addrspace(4)* %5266 to float addrspace(1)*
  store float %5264, float addrspace(1)* %5267, align 4
  br label %._crit_edge70.1.13

5268:                                             ; preds = %5258
  %5269 = mul nsw i64 %5260, %const_reg_qword7, !spirv.Decorations !610
  %5270 = getelementptr float, float addrspace(4)* %66, i64 %5269
  %5271 = getelementptr float, float addrspace(4)* %5270, i64 %5261
  %5272 = addrspacecast float addrspace(4)* %5271 to float addrspace(1)*
  %5273 = load float, float addrspace(1)* %5272, align 4
  %5274 = fmul reassoc nsz arcp contract float %5273, %4, !spirv.Decorations !612
  %5275 = fadd reassoc nsz arcp contract float %5264, %5274, !spirv.Decorations !612
  %5276 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5263
  %5277 = addrspacecast float addrspace(4)* %5276 to float addrspace(1)*
  store float %5275, float addrspace(1)* %5277, align 4
  br label %._crit_edge70.1.13

._crit_edge70.1.13:                               ; preds = %5255, %5268, %5265
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %5214, i32* %60, align 4, !noalias !614
  br label %5278

5278:                                             ; preds = %5278, %._crit_edge70.1.13
  %5279 = phi i32 [ 0, %._crit_edge70.1.13 ], [ %5284, %5278 ]
  %5280 = zext i32 %5279 to i64
  %5281 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5280
  %5282 = load i32, i32* %5281, align 4, !noalias !614
  %5283 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5280
  store i32 %5282, i32* %5283, align 4, !alias.scope !614
  %5284 = add nuw nsw i32 %5279, 1, !spirv.Decorations !620
  %5285 = icmp eq i32 %5279, 0
  br i1 %5285, label %5278, label %5286, !llvm.loop !630

5286:                                             ; preds = %5278
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5287 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5288 = and i1 %3647, %5215
  br i1 %5288, label %5289, label %._crit_edge70.2.13

5289:                                             ; preds = %5286
  %5290 = shl i64 %5287, 32
  %5291 = ashr exact i64 %5290, 32
  %5292 = ashr i64 %5287, 32
  %5293 = mul nsw i64 %5291, %const_reg_qword9, !spirv.Decorations !610
  %5294 = add nsw i64 %5293, %5292, !spirv.Decorations !610
  %5295 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !612
  br i1 %48, label %5299, label %5296

5296:                                             ; preds = %5289
  %5297 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5294
  %5298 = addrspacecast float addrspace(4)* %5297 to float addrspace(1)*
  store float %5295, float addrspace(1)* %5298, align 4
  br label %._crit_edge70.2.13

5299:                                             ; preds = %5289
  %5300 = mul nsw i64 %5291, %const_reg_qword7, !spirv.Decorations !610
  %5301 = getelementptr float, float addrspace(4)* %66, i64 %5300
  %5302 = getelementptr float, float addrspace(4)* %5301, i64 %5292
  %5303 = addrspacecast float addrspace(4)* %5302 to float addrspace(1)*
  %5304 = load float, float addrspace(1)* %5303, align 4
  %5305 = fmul reassoc nsz arcp contract float %5304, %4, !spirv.Decorations !612
  %5306 = fadd reassoc nsz arcp contract float %5295, %5305, !spirv.Decorations !612
  %5307 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5294
  %5308 = addrspacecast float addrspace(4)* %5307 to float addrspace(1)*
  store float %5306, float addrspace(1)* %5308, align 4
  br label %._crit_edge70.2.13

._crit_edge70.2.13:                               ; preds = %5286, %5299, %5296
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %5214, i32* %60, align 4, !noalias !614
  br label %5309

5309:                                             ; preds = %5309, %._crit_edge70.2.13
  %5310 = phi i32 [ 0, %._crit_edge70.2.13 ], [ %5315, %5309 ]
  %5311 = zext i32 %5310 to i64
  %5312 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5311
  %5313 = load i32, i32* %5312, align 4, !noalias !614
  %5314 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5311
  store i32 %5313, i32* %5314, align 4, !alias.scope !614
  %5315 = add nuw nsw i32 %5310, 1, !spirv.Decorations !620
  %5316 = icmp eq i32 %5310, 0
  br i1 %5316, label %5309, label %5317, !llvm.loop !630

5317:                                             ; preds = %5309
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5318 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5319 = and i1 %3680, %5215
  br i1 %5319, label %5320, label %.preheader1.13

5320:                                             ; preds = %5317
  %5321 = shl i64 %5318, 32
  %5322 = ashr exact i64 %5321, 32
  %5323 = ashr i64 %5318, 32
  %5324 = mul nsw i64 %5322, %const_reg_qword9, !spirv.Decorations !610
  %5325 = add nsw i64 %5324, %5323, !spirv.Decorations !610
  %5326 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !612
  br i1 %48, label %5330, label %5327

5327:                                             ; preds = %5320
  %5328 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5325
  %5329 = addrspacecast float addrspace(4)* %5328 to float addrspace(1)*
  store float %5326, float addrspace(1)* %5329, align 4
  br label %.preheader1.13

5330:                                             ; preds = %5320
  %5331 = mul nsw i64 %5322, %const_reg_qword7, !spirv.Decorations !610
  %5332 = getelementptr float, float addrspace(4)* %66, i64 %5331
  %5333 = getelementptr float, float addrspace(4)* %5332, i64 %5323
  %5334 = addrspacecast float addrspace(4)* %5333 to float addrspace(1)*
  %5335 = load float, float addrspace(1)* %5334, align 4
  %5336 = fmul reassoc nsz arcp contract float %5335, %4, !spirv.Decorations !612
  %5337 = fadd reassoc nsz arcp contract float %5326, %5336, !spirv.Decorations !612
  %5338 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5325
  %5339 = addrspacecast float addrspace(4)* %5338 to float addrspace(1)*
  store float %5337, float addrspace(1)* %5339, align 4
  br label %.preheader1.13

.preheader1.13:                                   ; preds = %5317, %5330, %5327
  %5340 = or i32 %41, 14
  %5341 = icmp slt i32 %5340, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %5340, i32* %60, align 4, !noalias !614
  br label %5342

5342:                                             ; preds = %5342, %.preheader1.13
  %5343 = phi i32 [ 0, %.preheader1.13 ], [ %5348, %5342 ]
  %5344 = zext i32 %5343 to i64
  %5345 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5344
  %5346 = load i32, i32* %5345, align 4, !noalias !614
  %5347 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5344
  store i32 %5346, i32* %5347, align 4, !alias.scope !614
  %5348 = add nuw nsw i32 %5343, 1, !spirv.Decorations !620
  %5349 = icmp eq i32 %5343, 0
  br i1 %5349, label %5342, label %5350, !llvm.loop !630

5350:                                             ; preds = %5342
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5351 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5352 = and i1 %3581, %5341
  br i1 %5352, label %5353, label %._crit_edge70.14

5353:                                             ; preds = %5350
  %5354 = shl i64 %5351, 32
  %5355 = ashr exact i64 %5354, 32
  %5356 = ashr i64 %5351, 32
  %5357 = mul nsw i64 %5355, %const_reg_qword9, !spirv.Decorations !610
  %5358 = add nsw i64 %5357, %5356, !spirv.Decorations !610
  %5359 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !612
  br i1 %48, label %5363, label %5360

5360:                                             ; preds = %5353
  %5361 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5358
  %5362 = addrspacecast float addrspace(4)* %5361 to float addrspace(1)*
  store float %5359, float addrspace(1)* %5362, align 4
  br label %._crit_edge70.14

5363:                                             ; preds = %5353
  %5364 = mul nsw i64 %5355, %const_reg_qword7, !spirv.Decorations !610
  %5365 = getelementptr float, float addrspace(4)* %66, i64 %5364
  %5366 = getelementptr float, float addrspace(4)* %5365, i64 %5356
  %5367 = addrspacecast float addrspace(4)* %5366 to float addrspace(1)*
  %5368 = load float, float addrspace(1)* %5367, align 4
  %5369 = fmul reassoc nsz arcp contract float %5368, %4, !spirv.Decorations !612
  %5370 = fadd reassoc nsz arcp contract float %5359, %5369, !spirv.Decorations !612
  %5371 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5358
  %5372 = addrspacecast float addrspace(4)* %5371 to float addrspace(1)*
  store float %5370, float addrspace(1)* %5372, align 4
  br label %._crit_edge70.14

._crit_edge70.14:                                 ; preds = %5350, %5363, %5360
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %5340, i32* %60, align 4, !noalias !614
  br label %5373

5373:                                             ; preds = %5373, %._crit_edge70.14
  %5374 = phi i32 [ 0, %._crit_edge70.14 ], [ %5379, %5373 ]
  %5375 = zext i32 %5374 to i64
  %5376 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5375
  %5377 = load i32, i32* %5376, align 4, !noalias !614
  %5378 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5375
  store i32 %5377, i32* %5378, align 4, !alias.scope !614
  %5379 = add nuw nsw i32 %5374, 1, !spirv.Decorations !620
  %5380 = icmp eq i32 %5374, 0
  br i1 %5380, label %5373, label %5381, !llvm.loop !630

5381:                                             ; preds = %5373
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5382 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5383 = and i1 %3614, %5341
  br i1 %5383, label %5384, label %._crit_edge70.1.14

5384:                                             ; preds = %5381
  %5385 = shl i64 %5382, 32
  %5386 = ashr exact i64 %5385, 32
  %5387 = ashr i64 %5382, 32
  %5388 = mul nsw i64 %5386, %const_reg_qword9, !spirv.Decorations !610
  %5389 = add nsw i64 %5388, %5387, !spirv.Decorations !610
  %5390 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !612
  br i1 %48, label %5394, label %5391

5391:                                             ; preds = %5384
  %5392 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5389
  %5393 = addrspacecast float addrspace(4)* %5392 to float addrspace(1)*
  store float %5390, float addrspace(1)* %5393, align 4
  br label %._crit_edge70.1.14

5394:                                             ; preds = %5384
  %5395 = mul nsw i64 %5386, %const_reg_qword7, !spirv.Decorations !610
  %5396 = getelementptr float, float addrspace(4)* %66, i64 %5395
  %5397 = getelementptr float, float addrspace(4)* %5396, i64 %5387
  %5398 = addrspacecast float addrspace(4)* %5397 to float addrspace(1)*
  %5399 = load float, float addrspace(1)* %5398, align 4
  %5400 = fmul reassoc nsz arcp contract float %5399, %4, !spirv.Decorations !612
  %5401 = fadd reassoc nsz arcp contract float %5390, %5400, !spirv.Decorations !612
  %5402 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5389
  %5403 = addrspacecast float addrspace(4)* %5402 to float addrspace(1)*
  store float %5401, float addrspace(1)* %5403, align 4
  br label %._crit_edge70.1.14

._crit_edge70.1.14:                               ; preds = %5381, %5394, %5391
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %5340, i32* %60, align 4, !noalias !614
  br label %5404

5404:                                             ; preds = %5404, %._crit_edge70.1.14
  %5405 = phi i32 [ 0, %._crit_edge70.1.14 ], [ %5410, %5404 ]
  %5406 = zext i32 %5405 to i64
  %5407 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5406
  %5408 = load i32, i32* %5407, align 4, !noalias !614
  %5409 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5406
  store i32 %5408, i32* %5409, align 4, !alias.scope !614
  %5410 = add nuw nsw i32 %5405, 1, !spirv.Decorations !620
  %5411 = icmp eq i32 %5405, 0
  br i1 %5411, label %5404, label %5412, !llvm.loop !630

5412:                                             ; preds = %5404
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5413 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5414 = and i1 %3647, %5341
  br i1 %5414, label %5415, label %._crit_edge70.2.14

5415:                                             ; preds = %5412
  %5416 = shl i64 %5413, 32
  %5417 = ashr exact i64 %5416, 32
  %5418 = ashr i64 %5413, 32
  %5419 = mul nsw i64 %5417, %const_reg_qword9, !spirv.Decorations !610
  %5420 = add nsw i64 %5419, %5418, !spirv.Decorations !610
  %5421 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !612
  br i1 %48, label %5425, label %5422

5422:                                             ; preds = %5415
  %5423 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5420
  %5424 = addrspacecast float addrspace(4)* %5423 to float addrspace(1)*
  store float %5421, float addrspace(1)* %5424, align 4
  br label %._crit_edge70.2.14

5425:                                             ; preds = %5415
  %5426 = mul nsw i64 %5417, %const_reg_qword7, !spirv.Decorations !610
  %5427 = getelementptr float, float addrspace(4)* %66, i64 %5426
  %5428 = getelementptr float, float addrspace(4)* %5427, i64 %5418
  %5429 = addrspacecast float addrspace(4)* %5428 to float addrspace(1)*
  %5430 = load float, float addrspace(1)* %5429, align 4
  %5431 = fmul reassoc nsz arcp contract float %5430, %4, !spirv.Decorations !612
  %5432 = fadd reassoc nsz arcp contract float %5421, %5431, !spirv.Decorations !612
  %5433 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5420
  %5434 = addrspacecast float addrspace(4)* %5433 to float addrspace(1)*
  store float %5432, float addrspace(1)* %5434, align 4
  br label %._crit_edge70.2.14

._crit_edge70.2.14:                               ; preds = %5412, %5425, %5422
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %5340, i32* %60, align 4, !noalias !614
  br label %5435

5435:                                             ; preds = %5435, %._crit_edge70.2.14
  %5436 = phi i32 [ 0, %._crit_edge70.2.14 ], [ %5441, %5435 ]
  %5437 = zext i32 %5436 to i64
  %5438 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5437
  %5439 = load i32, i32* %5438, align 4, !noalias !614
  %5440 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5437
  store i32 %5439, i32* %5440, align 4, !alias.scope !614
  %5441 = add nuw nsw i32 %5436, 1, !spirv.Decorations !620
  %5442 = icmp eq i32 %5436, 0
  br i1 %5442, label %5435, label %5443, !llvm.loop !630

5443:                                             ; preds = %5435
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5444 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5445 = and i1 %3680, %5341
  br i1 %5445, label %5446, label %.preheader1.14

5446:                                             ; preds = %5443
  %5447 = shl i64 %5444, 32
  %5448 = ashr exact i64 %5447, 32
  %5449 = ashr i64 %5444, 32
  %5450 = mul nsw i64 %5448, %const_reg_qword9, !spirv.Decorations !610
  %5451 = add nsw i64 %5450, %5449, !spirv.Decorations !610
  %5452 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !612
  br i1 %48, label %5456, label %5453

5453:                                             ; preds = %5446
  %5454 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5451
  %5455 = addrspacecast float addrspace(4)* %5454 to float addrspace(1)*
  store float %5452, float addrspace(1)* %5455, align 4
  br label %.preheader1.14

5456:                                             ; preds = %5446
  %5457 = mul nsw i64 %5448, %const_reg_qword7, !spirv.Decorations !610
  %5458 = getelementptr float, float addrspace(4)* %66, i64 %5457
  %5459 = getelementptr float, float addrspace(4)* %5458, i64 %5449
  %5460 = addrspacecast float addrspace(4)* %5459 to float addrspace(1)*
  %5461 = load float, float addrspace(1)* %5460, align 4
  %5462 = fmul reassoc nsz arcp contract float %5461, %4, !spirv.Decorations !612
  %5463 = fadd reassoc nsz arcp contract float %5452, %5462, !spirv.Decorations !612
  %5464 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5451
  %5465 = addrspacecast float addrspace(4)* %5464 to float addrspace(1)*
  store float %5463, float addrspace(1)* %5465, align 4
  br label %.preheader1.14

.preheader1.14:                                   ; preds = %5443, %5456, %5453
  %5466 = or i32 %41, 15
  %5467 = icmp slt i32 %5466, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !614
  store i32 %5466, i32* %60, align 4, !noalias !614
  br label %5468

5468:                                             ; preds = %5468, %.preheader1.14
  %5469 = phi i32 [ 0, %.preheader1.14 ], [ %5474, %5468 ]
  %5470 = zext i32 %5469 to i64
  %5471 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5470
  %5472 = load i32, i32* %5471, align 4, !noalias !614
  %5473 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5470
  store i32 %5472, i32* %5473, align 4, !alias.scope !614
  %5474 = add nuw nsw i32 %5469, 1, !spirv.Decorations !620
  %5475 = icmp eq i32 %5469, 0
  br i1 %5475, label %5468, label %5476, !llvm.loop !630

5476:                                             ; preds = %5468
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5477 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5478 = and i1 %3581, %5467
  br i1 %5478, label %5479, label %._crit_edge70.15

5479:                                             ; preds = %5476
  %5480 = shl i64 %5477, 32
  %5481 = ashr exact i64 %5480, 32
  %5482 = ashr i64 %5477, 32
  %5483 = mul nsw i64 %5481, %const_reg_qword9, !spirv.Decorations !610
  %5484 = add nsw i64 %5483, %5482, !spirv.Decorations !610
  %5485 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !612
  br i1 %48, label %5489, label %5486

5486:                                             ; preds = %5479
  %5487 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5484
  %5488 = addrspacecast float addrspace(4)* %5487 to float addrspace(1)*
  store float %5485, float addrspace(1)* %5488, align 4
  br label %._crit_edge70.15

5489:                                             ; preds = %5479
  %5490 = mul nsw i64 %5481, %const_reg_qword7, !spirv.Decorations !610
  %5491 = getelementptr float, float addrspace(4)* %66, i64 %5490
  %5492 = getelementptr float, float addrspace(4)* %5491, i64 %5482
  %5493 = addrspacecast float addrspace(4)* %5492 to float addrspace(1)*
  %5494 = load float, float addrspace(1)* %5493, align 4
  %5495 = fmul reassoc nsz arcp contract float %5494, %4, !spirv.Decorations !612
  %5496 = fadd reassoc nsz arcp contract float %5485, %5495, !spirv.Decorations !612
  %5497 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5484
  %5498 = addrspacecast float addrspace(4)* %5497 to float addrspace(1)*
  store float %5496, float addrspace(1)* %5498, align 4
  br label %._crit_edge70.15

._crit_edge70.15:                                 ; preds = %5476, %5489, %5486
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3603, i32* %73, align 4, !noalias !614
  store i32 %5466, i32* %60, align 4, !noalias !614
  br label %5499

5499:                                             ; preds = %5499, %._crit_edge70.15
  %5500 = phi i32 [ 0, %._crit_edge70.15 ], [ %5505, %5499 ]
  %5501 = zext i32 %5500 to i64
  %5502 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5501
  %5503 = load i32, i32* %5502, align 4, !noalias !614
  %5504 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5501
  store i32 %5503, i32* %5504, align 4, !alias.scope !614
  %5505 = add nuw nsw i32 %5500, 1, !spirv.Decorations !620
  %5506 = icmp eq i32 %5500, 0
  br i1 %5506, label %5499, label %5507, !llvm.loop !630

5507:                                             ; preds = %5499
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5508 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5509 = and i1 %3614, %5467
  br i1 %5509, label %5510, label %._crit_edge70.1.15

5510:                                             ; preds = %5507
  %5511 = shl i64 %5508, 32
  %5512 = ashr exact i64 %5511, 32
  %5513 = ashr i64 %5508, 32
  %5514 = mul nsw i64 %5512, %const_reg_qword9, !spirv.Decorations !610
  %5515 = add nsw i64 %5514, %5513, !spirv.Decorations !610
  %5516 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !612
  br i1 %48, label %5520, label %5517

5517:                                             ; preds = %5510
  %5518 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5515
  %5519 = addrspacecast float addrspace(4)* %5518 to float addrspace(1)*
  store float %5516, float addrspace(1)* %5519, align 4
  br label %._crit_edge70.1.15

5520:                                             ; preds = %5510
  %5521 = mul nsw i64 %5512, %const_reg_qword7, !spirv.Decorations !610
  %5522 = getelementptr float, float addrspace(4)* %66, i64 %5521
  %5523 = getelementptr float, float addrspace(4)* %5522, i64 %5513
  %5524 = addrspacecast float addrspace(4)* %5523 to float addrspace(1)*
  %5525 = load float, float addrspace(1)* %5524, align 4
  %5526 = fmul reassoc nsz arcp contract float %5525, %4, !spirv.Decorations !612
  %5527 = fadd reassoc nsz arcp contract float %5516, %5526, !spirv.Decorations !612
  %5528 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5515
  %5529 = addrspacecast float addrspace(4)* %5528 to float addrspace(1)*
  store float %5527, float addrspace(1)* %5529, align 4
  br label %._crit_edge70.1.15

._crit_edge70.1.15:                               ; preds = %5507, %5520, %5517
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3636, i32* %73, align 4, !noalias !614
  store i32 %5466, i32* %60, align 4, !noalias !614
  br label %5530

5530:                                             ; preds = %5530, %._crit_edge70.1.15
  %5531 = phi i32 [ 0, %._crit_edge70.1.15 ], [ %5536, %5530 ]
  %5532 = zext i32 %5531 to i64
  %5533 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5532
  %5534 = load i32, i32* %5533, align 4, !noalias !614
  %5535 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5532
  store i32 %5534, i32* %5535, align 4, !alias.scope !614
  %5536 = add nuw nsw i32 %5531, 1, !spirv.Decorations !620
  %5537 = icmp eq i32 %5531, 0
  br i1 %5537, label %5530, label %5538, !llvm.loop !630

5538:                                             ; preds = %5530
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5539 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5540 = and i1 %3647, %5467
  br i1 %5540, label %5541, label %._crit_edge70.2.15

5541:                                             ; preds = %5538
  %5542 = shl i64 %5539, 32
  %5543 = ashr exact i64 %5542, 32
  %5544 = ashr i64 %5539, 32
  %5545 = mul nsw i64 %5543, %const_reg_qword9, !spirv.Decorations !610
  %5546 = add nsw i64 %5545, %5544, !spirv.Decorations !610
  %5547 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !612
  br i1 %48, label %5551, label %5548

5548:                                             ; preds = %5541
  %5549 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5546
  %5550 = addrspacecast float addrspace(4)* %5549 to float addrspace(1)*
  store float %5547, float addrspace(1)* %5550, align 4
  br label %._crit_edge70.2.15

5551:                                             ; preds = %5541
  %5552 = mul nsw i64 %5543, %const_reg_qword7, !spirv.Decorations !610
  %5553 = getelementptr float, float addrspace(4)* %66, i64 %5552
  %5554 = getelementptr float, float addrspace(4)* %5553, i64 %5544
  %5555 = addrspacecast float addrspace(4)* %5554 to float addrspace(1)*
  %5556 = load float, float addrspace(1)* %5555, align 4
  %5557 = fmul reassoc nsz arcp contract float %5556, %4, !spirv.Decorations !612
  %5558 = fadd reassoc nsz arcp contract float %5547, %5557, !spirv.Decorations !612
  %5559 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5546
  %5560 = addrspacecast float addrspace(4)* %5559 to float addrspace(1)*
  store float %5558, float addrspace(1)* %5560, align 4
  br label %._crit_edge70.2.15

._crit_edge70.2.15:                               ; preds = %5538, %5551, %5548
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3669, i32* %73, align 4, !noalias !614
  store i32 %5466, i32* %60, align 4, !noalias !614
  br label %5561

5561:                                             ; preds = %5561, %._crit_edge70.2.15
  %5562 = phi i32 [ 0, %._crit_edge70.2.15 ], [ %5567, %5561 ]
  %5563 = zext i32 %5562 to i64
  %5564 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5563
  %5565 = load i32, i32* %5564, align 4, !noalias !614
  %5566 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5563
  store i32 %5565, i32* %5566, align 4, !alias.scope !614
  %5567 = add nuw nsw i32 %5562, 1, !spirv.Decorations !620
  %5568 = icmp eq i32 %5562, 0
  br i1 %5568, label %5561, label %5569, !llvm.loop !630

5569:                                             ; preds = %5561
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5570 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5571 = and i1 %3680, %5467
  br i1 %5571, label %5572, label %.preheader1.15

5572:                                             ; preds = %5569
  %5573 = shl i64 %5570, 32
  %5574 = ashr exact i64 %5573, 32
  %5575 = ashr i64 %5570, 32
  %5576 = mul nsw i64 %5574, %const_reg_qword9, !spirv.Decorations !610
  %5577 = add nsw i64 %5576, %5575, !spirv.Decorations !610
  %5578 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !612
  br i1 %48, label %5582, label %5579

5579:                                             ; preds = %5572
  %5580 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5577
  %5581 = addrspacecast float addrspace(4)* %5580 to float addrspace(1)*
  store float %5578, float addrspace(1)* %5581, align 4
  br label %.preheader1.15

5582:                                             ; preds = %5572
  %5583 = mul nsw i64 %5574, %const_reg_qword7, !spirv.Decorations !610
  %5584 = getelementptr float, float addrspace(4)* %66, i64 %5583
  %5585 = getelementptr float, float addrspace(4)* %5584, i64 %5575
  %5586 = addrspacecast float addrspace(4)* %5585 to float addrspace(1)*
  %5587 = load float, float addrspace(1)* %5586, align 4
  %5588 = fmul reassoc nsz arcp contract float %5587, %4, !spirv.Decorations !612
  %5589 = fadd reassoc nsz arcp contract float %5578, %5588, !spirv.Decorations !612
  %5590 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5577
  %5591 = addrspacecast float addrspace(4)* %5590 to float addrspace(1)*
  store float %5589, float addrspace(1)* %5591, align 4
  br label %.preheader1.15

.preheader1.15:                                   ; preds = %5569, %5582, %5579
  %5592 = zext i32 %14 to i64
  %5593 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %5593)
  %5594 = mul nsw i64 %5592, %9, !spirv.Decorations !610
  %5595 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %5594
  %5596 = mul nsw i64 %5592, %10, !spirv.Decorations !610
  %5597 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %5596
  %5598 = mul nsw i64 %5592, %11
  %.idx = select i1 %48, i64 %5598, i64 0
  %5599 = getelementptr float, float addrspace(4)* %66, i64 %.idx
  %5600 = mul nsw i64 %5592, %12, !spirv.Decorations !610
  %5601 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5600
  %5602 = add i32 %64, %14
  %5603 = icmp slt i32 %5602, %8
  br i1 %5603, label %.preheader2.preheader, label %._crit_edge72, !llvm.loop !631

._crit_edge72:                                    ; preds = %.preheader1.15, %13
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
  %14 = extractelement <3 x i32> %numWorkGroups, i64 2
  %15 = extractelement <3 x i32> %localSize, i64 0
  %16 = extractelement <3 x i32> %localSize, i64 1
  %17 = extractelement <8 x i32> %r0, i64 1
  %18 = extractelement <8 x i32> %r0, i64 6
  %19 = extractelement <8 x i32> %r0, i64 7
  %20 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %21 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %22 = alloca [2 x i32], align 4, !spirv.Decorations !608
  %23 = alloca %structtype.0, align 8
  %24 = alloca %structtype.0, align 8
  %25 = alloca %structtype.0, align 8
  %26 = inttoptr i64 %const_reg_qword8 to float addrspace(4)*
  %27 = inttoptr i64 %const_reg_qword6 to float addrspace(4)*
  %28 = inttoptr i64 %const_reg_qword4 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %29 = inttoptr i64 %const_reg_qword to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %30 = icmp sgt i32 %17, -1
  call void @llvm.assume(i1 %30)
  %31 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %31)
  %32 = mul i32 %17, %15
  %33 = zext i16 %localIdX to i32
  %34 = add i32 %32, %33
  %35 = shl i32 %34, 2
  %36 = icmp sgt i32 %18, -1
  call void @llvm.assume(i1 %36)
  %37 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %37)
  %38 = mul i32 %18, %16
  %39 = zext i16 %localIdY to i32
  %40 = add i32 %38, %39
  %41 = shl i32 %40, 4
  %42 = zext i32 %19 to i64
  %43 = icmp sgt i32 %19, -1
  call void @llvm.assume(i1 %43)
  %44 = mul nsw i64 %42, %9, !spirv.Decorations !610
  %45 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %29, i64 %44
  %46 = mul nsw i64 %42, %10, !spirv.Decorations !610
  %47 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %28, i64 %46
  %48 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !612
  %49 = mul nsw i64 %42, %11, !spirv.Decorations !610
  %50 = select i1 %48, i64 %49, i64 0
  %51 = getelementptr inbounds float, float addrspace(4)* %27, i64 %50
  %52 = mul nsw i64 %42, %12, !spirv.Decorations !610
  %53 = getelementptr inbounds float, float addrspace(4)* %26, i64 %52
  %54 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 1
  %55 = bitcast %structtype.0* %25 to [2 x i32]*
  %56 = getelementptr inbounds %structtype.0, %structtype.0* %25, i64 0, i32 0
  %57 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 1
  %58 = bitcast %structtype.0* %24 to [2 x i32]*
  %59 = getelementptr inbounds %structtype.0, %structtype.0* %24, i64 0, i32 0
  %60 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 1
  %61 = bitcast %structtype.0* %23 to [2 x i32]*
  %62 = getelementptr inbounds %structtype.0, %structtype.0* %23, i64 0, i32 0
  %63 = icmp slt i32 %19, %8
  br i1 %63, label %.preheader2.preheader, label %._crit_edge72

.preheader2.preheader:                            ; preds = %13, %.preheader1.15
  %64 = phi i32 [ %5474, %.preheader1.15 ], [ %19, %13 ]
  %65 = phi float addrspace(4)* [ %5473, %.preheader1.15 ], [ %53, %13 ]
  %66 = phi float addrspace(4)* [ %5471, %.preheader1.15 ], [ %51, %13 ]
  %67 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %5469, %.preheader1.15 ], [ %47, %13 ]
  %68 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %5467, %.preheader1.15 ], [ %45, %13 ]
  %69 = icmp sgt i32 %const_reg_dword2, 0
  br i1 %69, label %.preheader.preheader, label %.preheader1.preheader

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
  %70 = icmp slt i32 %41, %const_reg_dword1
  %71 = bitcast %structtype.0* %23 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  %72 = bitcast [2 x i32]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  %73 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 0
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %41, i32* %60, align 4, !noalias !632
  br label %3443

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
  %74 = phi i32 [ %3441, %.preheader.15 ], [ 0, %.preheader2.preheader ]
  %75 = icmp slt i32 %41, %const_reg_dword1
  %76 = icmp slt i32 %35, %const_reg_dword
  %77 = and i1 %76, %75
  br i1 %77, label %78, label %._crit_edge

78:                                               ; preds = %.preheader.preheader
  %79 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %79)
  %80 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %80)
  %81 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %81, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %82

82:                                               ; preds = %78, %82
  %83 = phi i32 [ 0, %78 ], [ %88, %82 ]
  %84 = zext i32 %83 to i64
  %85 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %84
  %86 = load i32, i32* %85, align 4, !noalias !635
  %87 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %84
  store i32 %86, i32* %87, align 4, !alias.scope !635
  %88 = add nuw nsw i32 %83, 1, !spirv.Decorations !620
  %89 = icmp eq i32 %83, 0
  br i1 %89, label %82, label %90, !llvm.loop !638

90:                                               ; preds = %82
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %80)
  %91 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %79)
  %92 = shl i64 %91, 32
  %93 = ashr exact i64 %92, 32
  %94 = mul nsw i64 %93, %const_reg_qword3, !spirv.Decorations !610
  %95 = ashr i64 %91, 32
  %96 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %94, i32 0
  %97 = getelementptr i16, i16 addrspace(4)* %96, i64 %95
  %98 = addrspacecast i16 addrspace(4)* %97 to i16 addrspace(1)*
  %99 = load i16, i16 addrspace(1)* %98, align 2
  %100 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %100)
  %101 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %101)
  %102 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %102, align 4, !noalias !639
  store i32 %41, i32* %57, align 4, !noalias !639
  br label %103

103:                                              ; preds = %90, %103
  %104 = phi i32 [ 0, %90 ], [ %109, %103 ]
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %105
  %107 = load i32, i32* %106, align 4, !noalias !639
  %108 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %105
  store i32 %107, i32* %108, align 4, !alias.scope !639
  %109 = add nuw nsw i32 %104, 1, !spirv.Decorations !620
  %110 = icmp eq i32 %104, 0
  br i1 %110, label %103, label %111, !llvm.loop !642

111:                                              ; preds = %103
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %101)
  %112 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %100)
  %113 = shl i64 %112, 32
  %114 = ashr exact i64 %113, 32
  %115 = mul nsw i64 %114, %const_reg_qword5, !spirv.Decorations !610
  %116 = ashr i64 %112, 32
  %117 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %115, i32 0
  %118 = getelementptr i16, i16 addrspace(4)* %117, i64 %116
  %119 = addrspacecast i16 addrspace(4)* %118 to i16 addrspace(1)*
  %120 = load i16, i16 addrspace(1)* %119, align 2
  %121 = zext i16 %99 to i32
  %122 = shl nuw i32 %121, 16, !spirv.Decorations !628
  %123 = bitcast i32 %122 to float
  %124 = zext i16 %120 to i32
  %125 = shl nuw i32 %124, 16, !spirv.Decorations !628
  %126 = bitcast i32 %125 to float
  %127 = fmul reassoc nsz arcp contract float %123, %126, !spirv.Decorations !612
  %128 = fadd reassoc nsz arcp contract float %127, %.sroa.0.1, !spirv.Decorations !612
  br label %._crit_edge

._crit_edge:                                      ; preds = %.preheader.preheader, %111
  %.sroa.0.2 = phi float [ %128, %111 ], [ %.sroa.0.1, %.preheader.preheader ]
  %129 = or i32 %35, 1
  %130 = icmp slt i32 %129, %const_reg_dword
  %131 = and i1 %130, %75
  br i1 %131, label %132, label %._crit_edge.1

132:                                              ; preds = %._crit_edge
  %133 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %133)
  %134 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)
  %135 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %135, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %136

136:                                              ; preds = %136, %132
  %137 = phi i32 [ 0, %132 ], [ %142, %136 ]
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %138
  %140 = load i32, i32* %139, align 4, !noalias !635
  %141 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %138
  store i32 %140, i32* %141, align 4, !alias.scope !635
  %142 = add nuw nsw i32 %137, 1, !spirv.Decorations !620
  %143 = icmp eq i32 %137, 0
  br i1 %143, label %136, label %144, !llvm.loop !638

144:                                              ; preds = %136
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)
  %145 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %133)
  %146 = shl i64 %145, 32
  %147 = ashr exact i64 %146, 32
  %148 = mul nsw i64 %147, %const_reg_qword3, !spirv.Decorations !610
  %149 = ashr i64 %145, 32
  %150 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %148, i32 0
  %151 = getelementptr i16, i16 addrspace(4)* %150, i64 %149
  %152 = addrspacecast i16 addrspace(4)* %151 to i16 addrspace(1)*
  %153 = load i16, i16 addrspace(1)* %152, align 2
  %154 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %154)
  %155 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %155)
  %156 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %156, align 4, !noalias !639
  store i32 %41, i32* %57, align 4, !noalias !639
  br label %157

157:                                              ; preds = %157, %144
  %158 = phi i32 [ 0, %144 ], [ %163, %157 ]
  %159 = zext i32 %158 to i64
  %160 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %159
  %161 = load i32, i32* %160, align 4, !noalias !639
  %162 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %159
  store i32 %161, i32* %162, align 4, !alias.scope !639
  %163 = add nuw nsw i32 %158, 1, !spirv.Decorations !620
  %164 = icmp eq i32 %158, 0
  br i1 %164, label %157, label %165, !llvm.loop !642

165:                                              ; preds = %157
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %155)
  %166 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %167 = shl i64 %166, 32
  %168 = ashr exact i64 %167, 32
  %169 = mul nsw i64 %168, %const_reg_qword5, !spirv.Decorations !610
  %170 = ashr i64 %166, 32
  %171 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %169, i32 0
  %172 = getelementptr i16, i16 addrspace(4)* %171, i64 %170
  %173 = addrspacecast i16 addrspace(4)* %172 to i16 addrspace(1)*
  %174 = load i16, i16 addrspace(1)* %173, align 2
  %175 = zext i16 %153 to i32
  %176 = shl nuw i32 %175, 16, !spirv.Decorations !628
  %177 = bitcast i32 %176 to float
  %178 = zext i16 %174 to i32
  %179 = shl nuw i32 %178, 16, !spirv.Decorations !628
  %180 = bitcast i32 %179 to float
  %181 = fmul reassoc nsz arcp contract float %177, %180, !spirv.Decorations !612
  %182 = fadd reassoc nsz arcp contract float %181, %.sroa.66.1, !spirv.Decorations !612
  br label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge, %165
  %.sroa.66.2 = phi float [ %182, %165 ], [ %.sroa.66.1, %._crit_edge ]
  %183 = or i32 %35, 2
  %184 = icmp slt i32 %183, %const_reg_dword
  %185 = and i1 %184, %75
  br i1 %185, label %186, label %._crit_edge.2

186:                                              ; preds = %._crit_edge.1
  %187 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %187)
  %188 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %188)
  %189 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %189, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %190

190:                                              ; preds = %190, %186
  %191 = phi i32 [ 0, %186 ], [ %196, %190 ]
  %192 = zext i32 %191 to i64
  %193 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %192
  %194 = load i32, i32* %193, align 4, !noalias !635
  %195 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %192
  store i32 %194, i32* %195, align 4, !alias.scope !635
  %196 = add nuw nsw i32 %191, 1, !spirv.Decorations !620
  %197 = icmp eq i32 %191, 0
  br i1 %197, label %190, label %198, !llvm.loop !638

198:                                              ; preds = %190
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %188)
  %199 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %187)
  %200 = shl i64 %199, 32
  %201 = ashr exact i64 %200, 32
  %202 = mul nsw i64 %201, %const_reg_qword3, !spirv.Decorations !610
  %203 = ashr i64 %199, 32
  %204 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %202, i32 0
  %205 = getelementptr i16, i16 addrspace(4)* %204, i64 %203
  %206 = addrspacecast i16 addrspace(4)* %205 to i16 addrspace(1)*
  %207 = load i16, i16 addrspace(1)* %206, align 2
  %208 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %208)
  %209 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %209)
  %210 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %210, align 4, !noalias !639
  store i32 %41, i32* %57, align 4, !noalias !639
  br label %211

211:                                              ; preds = %211, %198
  %212 = phi i32 [ 0, %198 ], [ %217, %211 ]
  %213 = zext i32 %212 to i64
  %214 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %213
  %215 = load i32, i32* %214, align 4, !noalias !639
  %216 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %213
  store i32 %215, i32* %216, align 4, !alias.scope !639
  %217 = add nuw nsw i32 %212, 1, !spirv.Decorations !620
  %218 = icmp eq i32 %212, 0
  br i1 %218, label %211, label %219, !llvm.loop !642

219:                                              ; preds = %211
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %209)
  %220 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %208)
  %221 = shl i64 %220, 32
  %222 = ashr exact i64 %221, 32
  %223 = mul nsw i64 %222, %const_reg_qword5, !spirv.Decorations !610
  %224 = ashr i64 %220, 32
  %225 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %223, i32 0
  %226 = getelementptr i16, i16 addrspace(4)* %225, i64 %224
  %227 = addrspacecast i16 addrspace(4)* %226 to i16 addrspace(1)*
  %228 = load i16, i16 addrspace(1)* %227, align 2
  %229 = zext i16 %207 to i32
  %230 = shl nuw i32 %229, 16, !spirv.Decorations !628
  %231 = bitcast i32 %230 to float
  %232 = zext i16 %228 to i32
  %233 = shl nuw i32 %232, 16, !spirv.Decorations !628
  %234 = bitcast i32 %233 to float
  %235 = fmul reassoc nsz arcp contract float %231, %234, !spirv.Decorations !612
  %236 = fadd reassoc nsz arcp contract float %235, %.sroa.130.1, !spirv.Decorations !612
  br label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge.1, %219
  %.sroa.130.2 = phi float [ %236, %219 ], [ %.sroa.130.1, %._crit_edge.1 ]
  %237 = or i32 %35, 3
  %238 = icmp slt i32 %237, %const_reg_dword
  %239 = and i1 %238, %75
  br i1 %239, label %240, label %.preheader

240:                                              ; preds = %._crit_edge.2
  %241 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %241)
  %242 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %242)
  %243 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %243, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %244

244:                                              ; preds = %244, %240
  %245 = phi i32 [ 0, %240 ], [ %250, %244 ]
  %246 = zext i32 %245 to i64
  %247 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %246
  %248 = load i32, i32* %247, align 4, !noalias !635
  %249 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %246
  store i32 %248, i32* %249, align 4, !alias.scope !635
  %250 = add nuw nsw i32 %245, 1, !spirv.Decorations !620
  %251 = icmp eq i32 %245, 0
  br i1 %251, label %244, label %252, !llvm.loop !638

252:                                              ; preds = %244
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %242)
  %253 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %241)
  %254 = shl i64 %253, 32
  %255 = ashr exact i64 %254, 32
  %256 = mul nsw i64 %255, %const_reg_qword3, !spirv.Decorations !610
  %257 = ashr i64 %253, 32
  %258 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %256, i32 0
  %259 = getelementptr i16, i16 addrspace(4)* %258, i64 %257
  %260 = addrspacecast i16 addrspace(4)* %259 to i16 addrspace(1)*
  %261 = load i16, i16 addrspace(1)* %260, align 2
  %262 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %262)
  %263 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %263)
  %264 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %264, align 4, !noalias !639
  store i32 %41, i32* %57, align 4, !noalias !639
  br label %265

265:                                              ; preds = %265, %252
  %266 = phi i32 [ 0, %252 ], [ %271, %265 ]
  %267 = zext i32 %266 to i64
  %268 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %267
  %269 = load i32, i32* %268, align 4, !noalias !639
  %270 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %267
  store i32 %269, i32* %270, align 4, !alias.scope !639
  %271 = add nuw nsw i32 %266, 1, !spirv.Decorations !620
  %272 = icmp eq i32 %266, 0
  br i1 %272, label %265, label %273, !llvm.loop !642

273:                                              ; preds = %265
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %263)
  %274 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %262)
  %275 = shl i64 %274, 32
  %276 = ashr exact i64 %275, 32
  %277 = mul nsw i64 %276, %const_reg_qword5, !spirv.Decorations !610
  %278 = ashr i64 %274, 32
  %279 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %277, i32 0
  %280 = getelementptr i16, i16 addrspace(4)* %279, i64 %278
  %281 = addrspacecast i16 addrspace(4)* %280 to i16 addrspace(1)*
  %282 = load i16, i16 addrspace(1)* %281, align 2
  %283 = zext i16 %261 to i32
  %284 = shl nuw i32 %283, 16, !spirv.Decorations !628
  %285 = bitcast i32 %284 to float
  %286 = zext i16 %282 to i32
  %287 = shl nuw i32 %286, 16, !spirv.Decorations !628
  %288 = bitcast i32 %287 to float
  %289 = fmul reassoc nsz arcp contract float %285, %288, !spirv.Decorations !612
  %290 = fadd reassoc nsz arcp contract float %289, %.sroa.194.1, !spirv.Decorations !612
  br label %.preheader

.preheader:                                       ; preds = %._crit_edge.2, %273
  %.sroa.194.2 = phi float [ %290, %273 ], [ %.sroa.194.1, %._crit_edge.2 ]
  %291 = or i32 %41, 1
  %292 = icmp slt i32 %291, %const_reg_dword1
  %293 = and i1 %76, %292
  br i1 %293, label %294, label %._crit_edge.173

294:                                              ; preds = %.preheader
  %295 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %295)
  %296 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %296)
  %297 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %297, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %298

298:                                              ; preds = %298, %294
  %299 = phi i32 [ 0, %294 ], [ %304, %298 ]
  %300 = zext i32 %299 to i64
  %301 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %300
  %302 = load i32, i32* %301, align 4, !noalias !635
  %303 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %300
  store i32 %302, i32* %303, align 4, !alias.scope !635
  %304 = add nuw nsw i32 %299, 1, !spirv.Decorations !620
  %305 = icmp eq i32 %299, 0
  br i1 %305, label %298, label %306, !llvm.loop !638

306:                                              ; preds = %298
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %296)
  %307 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %295)
  %308 = shl i64 %307, 32
  %309 = ashr exact i64 %308, 32
  %310 = mul nsw i64 %309, %const_reg_qword3, !spirv.Decorations !610
  %311 = ashr i64 %307, 32
  %312 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %310, i32 0
  %313 = getelementptr i16, i16 addrspace(4)* %312, i64 %311
  %314 = addrspacecast i16 addrspace(4)* %313 to i16 addrspace(1)*
  %315 = load i16, i16 addrspace(1)* %314, align 2
  %316 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %316)
  %317 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %317)
  %318 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %318, align 4, !noalias !639
  store i32 %291, i32* %57, align 4, !noalias !639
  br label %319

319:                                              ; preds = %319, %306
  %320 = phi i32 [ 0, %306 ], [ %325, %319 ]
  %321 = zext i32 %320 to i64
  %322 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %321
  %323 = load i32, i32* %322, align 4, !noalias !639
  %324 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %321
  store i32 %323, i32* %324, align 4, !alias.scope !639
  %325 = add nuw nsw i32 %320, 1, !spirv.Decorations !620
  %326 = icmp eq i32 %320, 0
  br i1 %326, label %319, label %327, !llvm.loop !642

327:                                              ; preds = %319
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %317)
  %328 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %316)
  %329 = shl i64 %328, 32
  %330 = ashr exact i64 %329, 32
  %331 = mul nsw i64 %330, %const_reg_qword5, !spirv.Decorations !610
  %332 = ashr i64 %328, 32
  %333 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %331, i32 0
  %334 = getelementptr i16, i16 addrspace(4)* %333, i64 %332
  %335 = addrspacecast i16 addrspace(4)* %334 to i16 addrspace(1)*
  %336 = load i16, i16 addrspace(1)* %335, align 2
  %337 = zext i16 %315 to i32
  %338 = shl nuw i32 %337, 16, !spirv.Decorations !628
  %339 = bitcast i32 %338 to float
  %340 = zext i16 %336 to i32
  %341 = shl nuw i32 %340, 16, !spirv.Decorations !628
  %342 = bitcast i32 %341 to float
  %343 = fmul reassoc nsz arcp contract float %339, %342, !spirv.Decorations !612
  %344 = fadd reassoc nsz arcp contract float %343, %.sroa.6.1, !spirv.Decorations !612
  br label %._crit_edge.173

._crit_edge.173:                                  ; preds = %.preheader, %327
  %.sroa.6.2 = phi float [ %344, %327 ], [ %.sroa.6.1, %.preheader ]
  %345 = and i1 %130, %292
  br i1 %345, label %346, label %._crit_edge.1.1

346:                                              ; preds = %._crit_edge.173
  %347 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %347)
  %348 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %348)
  %349 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %349, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %350

350:                                              ; preds = %350, %346
  %351 = phi i32 [ 0, %346 ], [ %356, %350 ]
  %352 = zext i32 %351 to i64
  %353 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %352
  %354 = load i32, i32* %353, align 4, !noalias !635
  %355 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %352
  store i32 %354, i32* %355, align 4, !alias.scope !635
  %356 = add nuw nsw i32 %351, 1, !spirv.Decorations !620
  %357 = icmp eq i32 %351, 0
  br i1 %357, label %350, label %358, !llvm.loop !638

358:                                              ; preds = %350
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %348)
  %359 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %347)
  %360 = shl i64 %359, 32
  %361 = ashr exact i64 %360, 32
  %362 = mul nsw i64 %361, %const_reg_qword3, !spirv.Decorations !610
  %363 = ashr i64 %359, 32
  %364 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %362, i32 0
  %365 = getelementptr i16, i16 addrspace(4)* %364, i64 %363
  %366 = addrspacecast i16 addrspace(4)* %365 to i16 addrspace(1)*
  %367 = load i16, i16 addrspace(1)* %366, align 2
  %368 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %368)
  %369 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %369)
  %370 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %370, align 4, !noalias !639
  store i32 %291, i32* %57, align 4, !noalias !639
  br label %371

371:                                              ; preds = %371, %358
  %372 = phi i32 [ 0, %358 ], [ %377, %371 ]
  %373 = zext i32 %372 to i64
  %374 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %373
  %375 = load i32, i32* %374, align 4, !noalias !639
  %376 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %373
  store i32 %375, i32* %376, align 4, !alias.scope !639
  %377 = add nuw nsw i32 %372, 1, !spirv.Decorations !620
  %378 = icmp eq i32 %372, 0
  br i1 %378, label %371, label %379, !llvm.loop !642

379:                                              ; preds = %371
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %369)
  %380 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %368)
  %381 = shl i64 %380, 32
  %382 = ashr exact i64 %381, 32
  %383 = mul nsw i64 %382, %const_reg_qword5, !spirv.Decorations !610
  %384 = ashr i64 %380, 32
  %385 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %383, i32 0
  %386 = getelementptr i16, i16 addrspace(4)* %385, i64 %384
  %387 = addrspacecast i16 addrspace(4)* %386 to i16 addrspace(1)*
  %388 = load i16, i16 addrspace(1)* %387, align 2
  %389 = zext i16 %367 to i32
  %390 = shl nuw i32 %389, 16, !spirv.Decorations !628
  %391 = bitcast i32 %390 to float
  %392 = zext i16 %388 to i32
  %393 = shl nuw i32 %392, 16, !spirv.Decorations !628
  %394 = bitcast i32 %393 to float
  %395 = fmul reassoc nsz arcp contract float %391, %394, !spirv.Decorations !612
  %396 = fadd reassoc nsz arcp contract float %395, %.sroa.70.1, !spirv.Decorations !612
  br label %._crit_edge.1.1

._crit_edge.1.1:                                  ; preds = %._crit_edge.173, %379
  %.sroa.70.2 = phi float [ %396, %379 ], [ %.sroa.70.1, %._crit_edge.173 ]
  %397 = and i1 %184, %292
  br i1 %397, label %398, label %._crit_edge.2.1

398:                                              ; preds = %._crit_edge.1.1
  %399 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %399)
  %400 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %400)
  %401 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %401, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %402

402:                                              ; preds = %402, %398
  %403 = phi i32 [ 0, %398 ], [ %408, %402 ]
  %404 = zext i32 %403 to i64
  %405 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %404
  %406 = load i32, i32* %405, align 4, !noalias !635
  %407 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %404
  store i32 %406, i32* %407, align 4, !alias.scope !635
  %408 = add nuw nsw i32 %403, 1, !spirv.Decorations !620
  %409 = icmp eq i32 %403, 0
  br i1 %409, label %402, label %410, !llvm.loop !638

410:                                              ; preds = %402
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %400)
  %411 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %399)
  %412 = shl i64 %411, 32
  %413 = ashr exact i64 %412, 32
  %414 = mul nsw i64 %413, %const_reg_qword3, !spirv.Decorations !610
  %415 = ashr i64 %411, 32
  %416 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %414, i32 0
  %417 = getelementptr i16, i16 addrspace(4)* %416, i64 %415
  %418 = addrspacecast i16 addrspace(4)* %417 to i16 addrspace(1)*
  %419 = load i16, i16 addrspace(1)* %418, align 2
  %420 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %420)
  %421 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %421)
  %422 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %422, align 4, !noalias !639
  store i32 %291, i32* %57, align 4, !noalias !639
  br label %423

423:                                              ; preds = %423, %410
  %424 = phi i32 [ 0, %410 ], [ %429, %423 ]
  %425 = zext i32 %424 to i64
  %426 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %425
  %427 = load i32, i32* %426, align 4, !noalias !639
  %428 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %425
  store i32 %427, i32* %428, align 4, !alias.scope !639
  %429 = add nuw nsw i32 %424, 1, !spirv.Decorations !620
  %430 = icmp eq i32 %424, 0
  br i1 %430, label %423, label %431, !llvm.loop !642

431:                                              ; preds = %423
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %421)
  %432 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %420)
  %433 = shl i64 %432, 32
  %434 = ashr exact i64 %433, 32
  %435 = mul nsw i64 %434, %const_reg_qword5, !spirv.Decorations !610
  %436 = ashr i64 %432, 32
  %437 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %435, i32 0
  %438 = getelementptr i16, i16 addrspace(4)* %437, i64 %436
  %439 = addrspacecast i16 addrspace(4)* %438 to i16 addrspace(1)*
  %440 = load i16, i16 addrspace(1)* %439, align 2
  %441 = zext i16 %419 to i32
  %442 = shl nuw i32 %441, 16, !spirv.Decorations !628
  %443 = bitcast i32 %442 to float
  %444 = zext i16 %440 to i32
  %445 = shl nuw i32 %444, 16, !spirv.Decorations !628
  %446 = bitcast i32 %445 to float
  %447 = fmul reassoc nsz arcp contract float %443, %446, !spirv.Decorations !612
  %448 = fadd reassoc nsz arcp contract float %447, %.sroa.134.1, !spirv.Decorations !612
  br label %._crit_edge.2.1

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1, %431
  %.sroa.134.2 = phi float [ %448, %431 ], [ %.sroa.134.1, %._crit_edge.1.1 ]
  %449 = and i1 %238, %292
  br i1 %449, label %450, label %.preheader.1

450:                                              ; preds = %._crit_edge.2.1
  %451 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %451)
  %452 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %452)
  %453 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %453, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %454

454:                                              ; preds = %454, %450
  %455 = phi i32 [ 0, %450 ], [ %460, %454 ]
  %456 = zext i32 %455 to i64
  %457 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %456
  %458 = load i32, i32* %457, align 4, !noalias !635
  %459 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %456
  store i32 %458, i32* %459, align 4, !alias.scope !635
  %460 = add nuw nsw i32 %455, 1, !spirv.Decorations !620
  %461 = icmp eq i32 %455, 0
  br i1 %461, label %454, label %462, !llvm.loop !638

462:                                              ; preds = %454
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %452)
  %463 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %451)
  %464 = shl i64 %463, 32
  %465 = ashr exact i64 %464, 32
  %466 = mul nsw i64 %465, %const_reg_qword3, !spirv.Decorations !610
  %467 = ashr i64 %463, 32
  %468 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %466, i32 0
  %469 = getelementptr i16, i16 addrspace(4)* %468, i64 %467
  %470 = addrspacecast i16 addrspace(4)* %469 to i16 addrspace(1)*
  %471 = load i16, i16 addrspace(1)* %470, align 2
  %472 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %472)
  %473 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %473)
  %474 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %474, align 4, !noalias !639
  store i32 %291, i32* %57, align 4, !noalias !639
  br label %475

475:                                              ; preds = %475, %462
  %476 = phi i32 [ 0, %462 ], [ %481, %475 ]
  %477 = zext i32 %476 to i64
  %478 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %477
  %479 = load i32, i32* %478, align 4, !noalias !639
  %480 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %477
  store i32 %479, i32* %480, align 4, !alias.scope !639
  %481 = add nuw nsw i32 %476, 1, !spirv.Decorations !620
  %482 = icmp eq i32 %476, 0
  br i1 %482, label %475, label %483, !llvm.loop !642

483:                                              ; preds = %475
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %473)
  %484 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %472)
  %485 = shl i64 %484, 32
  %486 = ashr exact i64 %485, 32
  %487 = mul nsw i64 %486, %const_reg_qword5, !spirv.Decorations !610
  %488 = ashr i64 %484, 32
  %489 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %487, i32 0
  %490 = getelementptr i16, i16 addrspace(4)* %489, i64 %488
  %491 = addrspacecast i16 addrspace(4)* %490 to i16 addrspace(1)*
  %492 = load i16, i16 addrspace(1)* %491, align 2
  %493 = zext i16 %471 to i32
  %494 = shl nuw i32 %493, 16, !spirv.Decorations !628
  %495 = bitcast i32 %494 to float
  %496 = zext i16 %492 to i32
  %497 = shl nuw i32 %496, 16, !spirv.Decorations !628
  %498 = bitcast i32 %497 to float
  %499 = fmul reassoc nsz arcp contract float %495, %498, !spirv.Decorations !612
  %500 = fadd reassoc nsz arcp contract float %499, %.sroa.198.1, !spirv.Decorations !612
  br label %.preheader.1

.preheader.1:                                     ; preds = %._crit_edge.2.1, %483
  %.sroa.198.2 = phi float [ %500, %483 ], [ %.sroa.198.1, %._crit_edge.2.1 ]
  %501 = or i32 %41, 2
  %502 = icmp slt i32 %501, %const_reg_dword1
  %503 = and i1 %76, %502
  br i1 %503, label %504, label %._crit_edge.274

504:                                              ; preds = %.preheader.1
  %505 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %505)
  %506 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %506)
  %507 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %507, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %508

508:                                              ; preds = %508, %504
  %509 = phi i32 [ 0, %504 ], [ %514, %508 ]
  %510 = zext i32 %509 to i64
  %511 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %510
  %512 = load i32, i32* %511, align 4, !noalias !635
  %513 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %510
  store i32 %512, i32* %513, align 4, !alias.scope !635
  %514 = add nuw nsw i32 %509, 1, !spirv.Decorations !620
  %515 = icmp eq i32 %509, 0
  br i1 %515, label %508, label %516, !llvm.loop !638

516:                                              ; preds = %508
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %506)
  %517 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %505)
  %518 = shl i64 %517, 32
  %519 = ashr exact i64 %518, 32
  %520 = mul nsw i64 %519, %const_reg_qword3, !spirv.Decorations !610
  %521 = ashr i64 %517, 32
  %522 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %520, i32 0
  %523 = getelementptr i16, i16 addrspace(4)* %522, i64 %521
  %524 = addrspacecast i16 addrspace(4)* %523 to i16 addrspace(1)*
  %525 = load i16, i16 addrspace(1)* %524, align 2
  %526 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %526)
  %527 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %527)
  %528 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %528, align 4, !noalias !639
  store i32 %501, i32* %57, align 4, !noalias !639
  br label %529

529:                                              ; preds = %529, %516
  %530 = phi i32 [ 0, %516 ], [ %535, %529 ]
  %531 = zext i32 %530 to i64
  %532 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %531
  %533 = load i32, i32* %532, align 4, !noalias !639
  %534 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %531
  store i32 %533, i32* %534, align 4, !alias.scope !639
  %535 = add nuw nsw i32 %530, 1, !spirv.Decorations !620
  %536 = icmp eq i32 %530, 0
  br i1 %536, label %529, label %537, !llvm.loop !642

537:                                              ; preds = %529
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %527)
  %538 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %526)
  %539 = shl i64 %538, 32
  %540 = ashr exact i64 %539, 32
  %541 = mul nsw i64 %540, %const_reg_qword5, !spirv.Decorations !610
  %542 = ashr i64 %538, 32
  %543 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %541, i32 0
  %544 = getelementptr i16, i16 addrspace(4)* %543, i64 %542
  %545 = addrspacecast i16 addrspace(4)* %544 to i16 addrspace(1)*
  %546 = load i16, i16 addrspace(1)* %545, align 2
  %547 = zext i16 %525 to i32
  %548 = shl nuw i32 %547, 16, !spirv.Decorations !628
  %549 = bitcast i32 %548 to float
  %550 = zext i16 %546 to i32
  %551 = shl nuw i32 %550, 16, !spirv.Decorations !628
  %552 = bitcast i32 %551 to float
  %553 = fmul reassoc nsz arcp contract float %549, %552, !spirv.Decorations !612
  %554 = fadd reassoc nsz arcp contract float %553, %.sroa.10.1, !spirv.Decorations !612
  br label %._crit_edge.274

._crit_edge.274:                                  ; preds = %.preheader.1, %537
  %.sroa.10.2 = phi float [ %554, %537 ], [ %.sroa.10.1, %.preheader.1 ]
  %555 = and i1 %130, %502
  br i1 %555, label %556, label %._crit_edge.1.2

556:                                              ; preds = %._crit_edge.274
  %557 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %557)
  %558 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %558)
  %559 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %559, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %560

560:                                              ; preds = %560, %556
  %561 = phi i32 [ 0, %556 ], [ %566, %560 ]
  %562 = zext i32 %561 to i64
  %563 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %562
  %564 = load i32, i32* %563, align 4, !noalias !635
  %565 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %562
  store i32 %564, i32* %565, align 4, !alias.scope !635
  %566 = add nuw nsw i32 %561, 1, !spirv.Decorations !620
  %567 = icmp eq i32 %561, 0
  br i1 %567, label %560, label %568, !llvm.loop !638

568:                                              ; preds = %560
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %558)
  %569 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %557)
  %570 = shl i64 %569, 32
  %571 = ashr exact i64 %570, 32
  %572 = mul nsw i64 %571, %const_reg_qword3, !spirv.Decorations !610
  %573 = ashr i64 %569, 32
  %574 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %572, i32 0
  %575 = getelementptr i16, i16 addrspace(4)* %574, i64 %573
  %576 = addrspacecast i16 addrspace(4)* %575 to i16 addrspace(1)*
  %577 = load i16, i16 addrspace(1)* %576, align 2
  %578 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %578)
  %579 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %579)
  %580 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %580, align 4, !noalias !639
  store i32 %501, i32* %57, align 4, !noalias !639
  br label %581

581:                                              ; preds = %581, %568
  %582 = phi i32 [ 0, %568 ], [ %587, %581 ]
  %583 = zext i32 %582 to i64
  %584 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %583
  %585 = load i32, i32* %584, align 4, !noalias !639
  %586 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %583
  store i32 %585, i32* %586, align 4, !alias.scope !639
  %587 = add nuw nsw i32 %582, 1, !spirv.Decorations !620
  %588 = icmp eq i32 %582, 0
  br i1 %588, label %581, label %589, !llvm.loop !642

589:                                              ; preds = %581
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %579)
  %590 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %578)
  %591 = shl i64 %590, 32
  %592 = ashr exact i64 %591, 32
  %593 = mul nsw i64 %592, %const_reg_qword5, !spirv.Decorations !610
  %594 = ashr i64 %590, 32
  %595 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %593, i32 0
  %596 = getelementptr i16, i16 addrspace(4)* %595, i64 %594
  %597 = addrspacecast i16 addrspace(4)* %596 to i16 addrspace(1)*
  %598 = load i16, i16 addrspace(1)* %597, align 2
  %599 = zext i16 %577 to i32
  %600 = shl nuw i32 %599, 16, !spirv.Decorations !628
  %601 = bitcast i32 %600 to float
  %602 = zext i16 %598 to i32
  %603 = shl nuw i32 %602, 16, !spirv.Decorations !628
  %604 = bitcast i32 %603 to float
  %605 = fmul reassoc nsz arcp contract float %601, %604, !spirv.Decorations !612
  %606 = fadd reassoc nsz arcp contract float %605, %.sroa.74.1, !spirv.Decorations !612
  br label %._crit_edge.1.2

._crit_edge.1.2:                                  ; preds = %._crit_edge.274, %589
  %.sroa.74.2 = phi float [ %606, %589 ], [ %.sroa.74.1, %._crit_edge.274 ]
  %607 = and i1 %184, %502
  br i1 %607, label %608, label %._crit_edge.2.2

608:                                              ; preds = %._crit_edge.1.2
  %609 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %609)
  %610 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %610)
  %611 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %611, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %612

612:                                              ; preds = %612, %608
  %613 = phi i32 [ 0, %608 ], [ %618, %612 ]
  %614 = zext i32 %613 to i64
  %615 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %614
  %616 = load i32, i32* %615, align 4, !noalias !635
  %617 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %614
  store i32 %616, i32* %617, align 4, !alias.scope !635
  %618 = add nuw nsw i32 %613, 1, !spirv.Decorations !620
  %619 = icmp eq i32 %613, 0
  br i1 %619, label %612, label %620, !llvm.loop !638

620:                                              ; preds = %612
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %610)
  %621 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %609)
  %622 = shl i64 %621, 32
  %623 = ashr exact i64 %622, 32
  %624 = mul nsw i64 %623, %const_reg_qword3, !spirv.Decorations !610
  %625 = ashr i64 %621, 32
  %626 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %624, i32 0
  %627 = getelementptr i16, i16 addrspace(4)* %626, i64 %625
  %628 = addrspacecast i16 addrspace(4)* %627 to i16 addrspace(1)*
  %629 = load i16, i16 addrspace(1)* %628, align 2
  %630 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %630)
  %631 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %631)
  %632 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %632, align 4, !noalias !639
  store i32 %501, i32* %57, align 4, !noalias !639
  br label %633

633:                                              ; preds = %633, %620
  %634 = phi i32 [ 0, %620 ], [ %639, %633 ]
  %635 = zext i32 %634 to i64
  %636 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %635
  %637 = load i32, i32* %636, align 4, !noalias !639
  %638 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %635
  store i32 %637, i32* %638, align 4, !alias.scope !639
  %639 = add nuw nsw i32 %634, 1, !spirv.Decorations !620
  %640 = icmp eq i32 %634, 0
  br i1 %640, label %633, label %641, !llvm.loop !642

641:                                              ; preds = %633
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %631)
  %642 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %630)
  %643 = shl i64 %642, 32
  %644 = ashr exact i64 %643, 32
  %645 = mul nsw i64 %644, %const_reg_qword5, !spirv.Decorations !610
  %646 = ashr i64 %642, 32
  %647 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %645, i32 0
  %648 = getelementptr i16, i16 addrspace(4)* %647, i64 %646
  %649 = addrspacecast i16 addrspace(4)* %648 to i16 addrspace(1)*
  %650 = load i16, i16 addrspace(1)* %649, align 2
  %651 = zext i16 %629 to i32
  %652 = shl nuw i32 %651, 16, !spirv.Decorations !628
  %653 = bitcast i32 %652 to float
  %654 = zext i16 %650 to i32
  %655 = shl nuw i32 %654, 16, !spirv.Decorations !628
  %656 = bitcast i32 %655 to float
  %657 = fmul reassoc nsz arcp contract float %653, %656, !spirv.Decorations !612
  %658 = fadd reassoc nsz arcp contract float %657, %.sroa.138.1, !spirv.Decorations !612
  br label %._crit_edge.2.2

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2, %641
  %.sroa.138.2 = phi float [ %658, %641 ], [ %.sroa.138.1, %._crit_edge.1.2 ]
  %659 = and i1 %238, %502
  br i1 %659, label %660, label %.preheader.2

660:                                              ; preds = %._crit_edge.2.2
  %661 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %661)
  %662 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %662)
  %663 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %663, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %664

664:                                              ; preds = %664, %660
  %665 = phi i32 [ 0, %660 ], [ %670, %664 ]
  %666 = zext i32 %665 to i64
  %667 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %666
  %668 = load i32, i32* %667, align 4, !noalias !635
  %669 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %666
  store i32 %668, i32* %669, align 4, !alias.scope !635
  %670 = add nuw nsw i32 %665, 1, !spirv.Decorations !620
  %671 = icmp eq i32 %665, 0
  br i1 %671, label %664, label %672, !llvm.loop !638

672:                                              ; preds = %664
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %662)
  %673 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %661)
  %674 = shl i64 %673, 32
  %675 = ashr exact i64 %674, 32
  %676 = mul nsw i64 %675, %const_reg_qword3, !spirv.Decorations !610
  %677 = ashr i64 %673, 32
  %678 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %676, i32 0
  %679 = getelementptr i16, i16 addrspace(4)* %678, i64 %677
  %680 = addrspacecast i16 addrspace(4)* %679 to i16 addrspace(1)*
  %681 = load i16, i16 addrspace(1)* %680, align 2
  %682 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %682)
  %683 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %683)
  %684 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %684, align 4, !noalias !639
  store i32 %501, i32* %57, align 4, !noalias !639
  br label %685

685:                                              ; preds = %685, %672
  %686 = phi i32 [ 0, %672 ], [ %691, %685 ]
  %687 = zext i32 %686 to i64
  %688 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %687
  %689 = load i32, i32* %688, align 4, !noalias !639
  %690 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %687
  store i32 %689, i32* %690, align 4, !alias.scope !639
  %691 = add nuw nsw i32 %686, 1, !spirv.Decorations !620
  %692 = icmp eq i32 %686, 0
  br i1 %692, label %685, label %693, !llvm.loop !642

693:                                              ; preds = %685
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %683)
  %694 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %682)
  %695 = shl i64 %694, 32
  %696 = ashr exact i64 %695, 32
  %697 = mul nsw i64 %696, %const_reg_qword5, !spirv.Decorations !610
  %698 = ashr i64 %694, 32
  %699 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %697, i32 0
  %700 = getelementptr i16, i16 addrspace(4)* %699, i64 %698
  %701 = addrspacecast i16 addrspace(4)* %700 to i16 addrspace(1)*
  %702 = load i16, i16 addrspace(1)* %701, align 2
  %703 = zext i16 %681 to i32
  %704 = shl nuw i32 %703, 16, !spirv.Decorations !628
  %705 = bitcast i32 %704 to float
  %706 = zext i16 %702 to i32
  %707 = shl nuw i32 %706, 16, !spirv.Decorations !628
  %708 = bitcast i32 %707 to float
  %709 = fmul reassoc nsz arcp contract float %705, %708, !spirv.Decorations !612
  %710 = fadd reassoc nsz arcp contract float %709, %.sroa.202.1, !spirv.Decorations !612
  br label %.preheader.2

.preheader.2:                                     ; preds = %._crit_edge.2.2, %693
  %.sroa.202.2 = phi float [ %710, %693 ], [ %.sroa.202.1, %._crit_edge.2.2 ]
  %711 = or i32 %41, 3
  %712 = icmp slt i32 %711, %const_reg_dword1
  %713 = and i1 %76, %712
  br i1 %713, label %714, label %._crit_edge.375

714:                                              ; preds = %.preheader.2
  %715 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %715)
  %716 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %716)
  %717 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %717, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %718

718:                                              ; preds = %718, %714
  %719 = phi i32 [ 0, %714 ], [ %724, %718 ]
  %720 = zext i32 %719 to i64
  %721 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %720
  %722 = load i32, i32* %721, align 4, !noalias !635
  %723 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %720
  store i32 %722, i32* %723, align 4, !alias.scope !635
  %724 = add nuw nsw i32 %719, 1, !spirv.Decorations !620
  %725 = icmp eq i32 %719, 0
  br i1 %725, label %718, label %726, !llvm.loop !638

726:                                              ; preds = %718
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %716)
  %727 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %715)
  %728 = shl i64 %727, 32
  %729 = ashr exact i64 %728, 32
  %730 = mul nsw i64 %729, %const_reg_qword3, !spirv.Decorations !610
  %731 = ashr i64 %727, 32
  %732 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %730, i32 0
  %733 = getelementptr i16, i16 addrspace(4)* %732, i64 %731
  %734 = addrspacecast i16 addrspace(4)* %733 to i16 addrspace(1)*
  %735 = load i16, i16 addrspace(1)* %734, align 2
  %736 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %736)
  %737 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %737)
  %738 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %738, align 4, !noalias !639
  store i32 %711, i32* %57, align 4, !noalias !639
  br label %739

739:                                              ; preds = %739, %726
  %740 = phi i32 [ 0, %726 ], [ %745, %739 ]
  %741 = zext i32 %740 to i64
  %742 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %741
  %743 = load i32, i32* %742, align 4, !noalias !639
  %744 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %741
  store i32 %743, i32* %744, align 4, !alias.scope !639
  %745 = add nuw nsw i32 %740, 1, !spirv.Decorations !620
  %746 = icmp eq i32 %740, 0
  br i1 %746, label %739, label %747, !llvm.loop !642

747:                                              ; preds = %739
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %737)
  %748 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %736)
  %749 = shl i64 %748, 32
  %750 = ashr exact i64 %749, 32
  %751 = mul nsw i64 %750, %const_reg_qword5, !spirv.Decorations !610
  %752 = ashr i64 %748, 32
  %753 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %751, i32 0
  %754 = getelementptr i16, i16 addrspace(4)* %753, i64 %752
  %755 = addrspacecast i16 addrspace(4)* %754 to i16 addrspace(1)*
  %756 = load i16, i16 addrspace(1)* %755, align 2
  %757 = zext i16 %735 to i32
  %758 = shl nuw i32 %757, 16, !spirv.Decorations !628
  %759 = bitcast i32 %758 to float
  %760 = zext i16 %756 to i32
  %761 = shl nuw i32 %760, 16, !spirv.Decorations !628
  %762 = bitcast i32 %761 to float
  %763 = fmul reassoc nsz arcp contract float %759, %762, !spirv.Decorations !612
  %764 = fadd reassoc nsz arcp contract float %763, %.sroa.14.1, !spirv.Decorations !612
  br label %._crit_edge.375

._crit_edge.375:                                  ; preds = %.preheader.2, %747
  %.sroa.14.2 = phi float [ %764, %747 ], [ %.sroa.14.1, %.preheader.2 ]
  %765 = and i1 %130, %712
  br i1 %765, label %766, label %._crit_edge.1.3

766:                                              ; preds = %._crit_edge.375
  %767 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %767)
  %768 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %768)
  %769 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %769, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %770

770:                                              ; preds = %770, %766
  %771 = phi i32 [ 0, %766 ], [ %776, %770 ]
  %772 = zext i32 %771 to i64
  %773 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %772
  %774 = load i32, i32* %773, align 4, !noalias !635
  %775 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %772
  store i32 %774, i32* %775, align 4, !alias.scope !635
  %776 = add nuw nsw i32 %771, 1, !spirv.Decorations !620
  %777 = icmp eq i32 %771, 0
  br i1 %777, label %770, label %778, !llvm.loop !638

778:                                              ; preds = %770
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %768)
  %779 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %767)
  %780 = shl i64 %779, 32
  %781 = ashr exact i64 %780, 32
  %782 = mul nsw i64 %781, %const_reg_qword3, !spirv.Decorations !610
  %783 = ashr i64 %779, 32
  %784 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %782, i32 0
  %785 = getelementptr i16, i16 addrspace(4)* %784, i64 %783
  %786 = addrspacecast i16 addrspace(4)* %785 to i16 addrspace(1)*
  %787 = load i16, i16 addrspace(1)* %786, align 2
  %788 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %788)
  %789 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %789)
  %790 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %790, align 4, !noalias !639
  store i32 %711, i32* %57, align 4, !noalias !639
  br label %791

791:                                              ; preds = %791, %778
  %792 = phi i32 [ 0, %778 ], [ %797, %791 ]
  %793 = zext i32 %792 to i64
  %794 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %793
  %795 = load i32, i32* %794, align 4, !noalias !639
  %796 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %793
  store i32 %795, i32* %796, align 4, !alias.scope !639
  %797 = add nuw nsw i32 %792, 1, !spirv.Decorations !620
  %798 = icmp eq i32 %792, 0
  br i1 %798, label %791, label %799, !llvm.loop !642

799:                                              ; preds = %791
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %789)
  %800 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %788)
  %801 = shl i64 %800, 32
  %802 = ashr exact i64 %801, 32
  %803 = mul nsw i64 %802, %const_reg_qword5, !spirv.Decorations !610
  %804 = ashr i64 %800, 32
  %805 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %803, i32 0
  %806 = getelementptr i16, i16 addrspace(4)* %805, i64 %804
  %807 = addrspacecast i16 addrspace(4)* %806 to i16 addrspace(1)*
  %808 = load i16, i16 addrspace(1)* %807, align 2
  %809 = zext i16 %787 to i32
  %810 = shl nuw i32 %809, 16, !spirv.Decorations !628
  %811 = bitcast i32 %810 to float
  %812 = zext i16 %808 to i32
  %813 = shl nuw i32 %812, 16, !spirv.Decorations !628
  %814 = bitcast i32 %813 to float
  %815 = fmul reassoc nsz arcp contract float %811, %814, !spirv.Decorations !612
  %816 = fadd reassoc nsz arcp contract float %815, %.sroa.78.1, !spirv.Decorations !612
  br label %._crit_edge.1.3

._crit_edge.1.3:                                  ; preds = %._crit_edge.375, %799
  %.sroa.78.2 = phi float [ %816, %799 ], [ %.sroa.78.1, %._crit_edge.375 ]
  %817 = and i1 %184, %712
  br i1 %817, label %818, label %._crit_edge.2.3

818:                                              ; preds = %._crit_edge.1.3
  %819 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %819)
  %820 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %820)
  %821 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %821, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %822

822:                                              ; preds = %822, %818
  %823 = phi i32 [ 0, %818 ], [ %828, %822 ]
  %824 = zext i32 %823 to i64
  %825 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %824
  %826 = load i32, i32* %825, align 4, !noalias !635
  %827 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %824
  store i32 %826, i32* %827, align 4, !alias.scope !635
  %828 = add nuw nsw i32 %823, 1, !spirv.Decorations !620
  %829 = icmp eq i32 %823, 0
  br i1 %829, label %822, label %830, !llvm.loop !638

830:                                              ; preds = %822
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %820)
  %831 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %819)
  %832 = shl i64 %831, 32
  %833 = ashr exact i64 %832, 32
  %834 = mul nsw i64 %833, %const_reg_qword3, !spirv.Decorations !610
  %835 = ashr i64 %831, 32
  %836 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %834, i32 0
  %837 = getelementptr i16, i16 addrspace(4)* %836, i64 %835
  %838 = addrspacecast i16 addrspace(4)* %837 to i16 addrspace(1)*
  %839 = load i16, i16 addrspace(1)* %838, align 2
  %840 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %840)
  %841 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %841)
  %842 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %842, align 4, !noalias !639
  store i32 %711, i32* %57, align 4, !noalias !639
  br label %843

843:                                              ; preds = %843, %830
  %844 = phi i32 [ 0, %830 ], [ %849, %843 ]
  %845 = zext i32 %844 to i64
  %846 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %845
  %847 = load i32, i32* %846, align 4, !noalias !639
  %848 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %845
  store i32 %847, i32* %848, align 4, !alias.scope !639
  %849 = add nuw nsw i32 %844, 1, !spirv.Decorations !620
  %850 = icmp eq i32 %844, 0
  br i1 %850, label %843, label %851, !llvm.loop !642

851:                                              ; preds = %843
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %841)
  %852 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %840)
  %853 = shl i64 %852, 32
  %854 = ashr exact i64 %853, 32
  %855 = mul nsw i64 %854, %const_reg_qword5, !spirv.Decorations !610
  %856 = ashr i64 %852, 32
  %857 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %855, i32 0
  %858 = getelementptr i16, i16 addrspace(4)* %857, i64 %856
  %859 = addrspacecast i16 addrspace(4)* %858 to i16 addrspace(1)*
  %860 = load i16, i16 addrspace(1)* %859, align 2
  %861 = zext i16 %839 to i32
  %862 = shl nuw i32 %861, 16, !spirv.Decorations !628
  %863 = bitcast i32 %862 to float
  %864 = zext i16 %860 to i32
  %865 = shl nuw i32 %864, 16, !spirv.Decorations !628
  %866 = bitcast i32 %865 to float
  %867 = fmul reassoc nsz arcp contract float %863, %866, !spirv.Decorations !612
  %868 = fadd reassoc nsz arcp contract float %867, %.sroa.142.1, !spirv.Decorations !612
  br label %._crit_edge.2.3

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3, %851
  %.sroa.142.2 = phi float [ %868, %851 ], [ %.sroa.142.1, %._crit_edge.1.3 ]
  %869 = and i1 %238, %712
  br i1 %869, label %870, label %.preheader.3

870:                                              ; preds = %._crit_edge.2.3
  %871 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %871)
  %872 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %872)
  %873 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %873, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %874

874:                                              ; preds = %874, %870
  %875 = phi i32 [ 0, %870 ], [ %880, %874 ]
  %876 = zext i32 %875 to i64
  %877 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %876
  %878 = load i32, i32* %877, align 4, !noalias !635
  %879 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %876
  store i32 %878, i32* %879, align 4, !alias.scope !635
  %880 = add nuw nsw i32 %875, 1, !spirv.Decorations !620
  %881 = icmp eq i32 %875, 0
  br i1 %881, label %874, label %882, !llvm.loop !638

882:                                              ; preds = %874
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %872)
  %883 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %871)
  %884 = shl i64 %883, 32
  %885 = ashr exact i64 %884, 32
  %886 = mul nsw i64 %885, %const_reg_qword3, !spirv.Decorations !610
  %887 = ashr i64 %883, 32
  %888 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %886, i32 0
  %889 = getelementptr i16, i16 addrspace(4)* %888, i64 %887
  %890 = addrspacecast i16 addrspace(4)* %889 to i16 addrspace(1)*
  %891 = load i16, i16 addrspace(1)* %890, align 2
  %892 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %892)
  %893 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %893)
  %894 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %894, align 4, !noalias !639
  store i32 %711, i32* %57, align 4, !noalias !639
  br label %895

895:                                              ; preds = %895, %882
  %896 = phi i32 [ 0, %882 ], [ %901, %895 ]
  %897 = zext i32 %896 to i64
  %898 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %897
  %899 = load i32, i32* %898, align 4, !noalias !639
  %900 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %897
  store i32 %899, i32* %900, align 4, !alias.scope !639
  %901 = add nuw nsw i32 %896, 1, !spirv.Decorations !620
  %902 = icmp eq i32 %896, 0
  br i1 %902, label %895, label %903, !llvm.loop !642

903:                                              ; preds = %895
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %893)
  %904 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %892)
  %905 = shl i64 %904, 32
  %906 = ashr exact i64 %905, 32
  %907 = mul nsw i64 %906, %const_reg_qword5, !spirv.Decorations !610
  %908 = ashr i64 %904, 32
  %909 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %907, i32 0
  %910 = getelementptr i16, i16 addrspace(4)* %909, i64 %908
  %911 = addrspacecast i16 addrspace(4)* %910 to i16 addrspace(1)*
  %912 = load i16, i16 addrspace(1)* %911, align 2
  %913 = zext i16 %891 to i32
  %914 = shl nuw i32 %913, 16, !spirv.Decorations !628
  %915 = bitcast i32 %914 to float
  %916 = zext i16 %912 to i32
  %917 = shl nuw i32 %916, 16, !spirv.Decorations !628
  %918 = bitcast i32 %917 to float
  %919 = fmul reassoc nsz arcp contract float %915, %918, !spirv.Decorations !612
  %920 = fadd reassoc nsz arcp contract float %919, %.sroa.206.1, !spirv.Decorations !612
  br label %.preheader.3

.preheader.3:                                     ; preds = %._crit_edge.2.3, %903
  %.sroa.206.2 = phi float [ %920, %903 ], [ %.sroa.206.1, %._crit_edge.2.3 ]
  %921 = or i32 %41, 4
  %922 = icmp slt i32 %921, %const_reg_dword1
  %923 = and i1 %76, %922
  br i1 %923, label %924, label %._crit_edge.4

924:                                              ; preds = %.preheader.3
  %925 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %925)
  %926 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %926)
  %927 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %927, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %928

928:                                              ; preds = %928, %924
  %929 = phi i32 [ 0, %924 ], [ %934, %928 ]
  %930 = zext i32 %929 to i64
  %931 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %930
  %932 = load i32, i32* %931, align 4, !noalias !635
  %933 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %930
  store i32 %932, i32* %933, align 4, !alias.scope !635
  %934 = add nuw nsw i32 %929, 1, !spirv.Decorations !620
  %935 = icmp eq i32 %929, 0
  br i1 %935, label %928, label %936, !llvm.loop !638

936:                                              ; preds = %928
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %926)
  %937 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %925)
  %938 = shl i64 %937, 32
  %939 = ashr exact i64 %938, 32
  %940 = mul nsw i64 %939, %const_reg_qword3, !spirv.Decorations !610
  %941 = ashr i64 %937, 32
  %942 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %940, i32 0
  %943 = getelementptr i16, i16 addrspace(4)* %942, i64 %941
  %944 = addrspacecast i16 addrspace(4)* %943 to i16 addrspace(1)*
  %945 = load i16, i16 addrspace(1)* %944, align 2
  %946 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %946)
  %947 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %947)
  %948 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %948, align 4, !noalias !639
  store i32 %921, i32* %57, align 4, !noalias !639
  br label %949

949:                                              ; preds = %949, %936
  %950 = phi i32 [ 0, %936 ], [ %955, %949 ]
  %951 = zext i32 %950 to i64
  %952 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %951
  %953 = load i32, i32* %952, align 4, !noalias !639
  %954 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %951
  store i32 %953, i32* %954, align 4, !alias.scope !639
  %955 = add nuw nsw i32 %950, 1, !spirv.Decorations !620
  %956 = icmp eq i32 %950, 0
  br i1 %956, label %949, label %957, !llvm.loop !642

957:                                              ; preds = %949
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %947)
  %958 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %946)
  %959 = shl i64 %958, 32
  %960 = ashr exact i64 %959, 32
  %961 = mul nsw i64 %960, %const_reg_qword5, !spirv.Decorations !610
  %962 = ashr i64 %958, 32
  %963 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %961, i32 0
  %964 = getelementptr i16, i16 addrspace(4)* %963, i64 %962
  %965 = addrspacecast i16 addrspace(4)* %964 to i16 addrspace(1)*
  %966 = load i16, i16 addrspace(1)* %965, align 2
  %967 = zext i16 %945 to i32
  %968 = shl nuw i32 %967, 16, !spirv.Decorations !628
  %969 = bitcast i32 %968 to float
  %970 = zext i16 %966 to i32
  %971 = shl nuw i32 %970, 16, !spirv.Decorations !628
  %972 = bitcast i32 %971 to float
  %973 = fmul reassoc nsz arcp contract float %969, %972, !spirv.Decorations !612
  %974 = fadd reassoc nsz arcp contract float %973, %.sroa.18.1, !spirv.Decorations !612
  br label %._crit_edge.4

._crit_edge.4:                                    ; preds = %.preheader.3, %957
  %.sroa.18.2 = phi float [ %974, %957 ], [ %.sroa.18.1, %.preheader.3 ]
  %975 = and i1 %130, %922
  br i1 %975, label %976, label %._crit_edge.1.4

976:                                              ; preds = %._crit_edge.4
  %977 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %977)
  %978 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %978)
  %979 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %979, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %980

980:                                              ; preds = %980, %976
  %981 = phi i32 [ 0, %976 ], [ %986, %980 ]
  %982 = zext i32 %981 to i64
  %983 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %982
  %984 = load i32, i32* %983, align 4, !noalias !635
  %985 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %982
  store i32 %984, i32* %985, align 4, !alias.scope !635
  %986 = add nuw nsw i32 %981, 1, !spirv.Decorations !620
  %987 = icmp eq i32 %981, 0
  br i1 %987, label %980, label %988, !llvm.loop !638

988:                                              ; preds = %980
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %978)
  %989 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %977)
  %990 = shl i64 %989, 32
  %991 = ashr exact i64 %990, 32
  %992 = mul nsw i64 %991, %const_reg_qword3, !spirv.Decorations !610
  %993 = ashr i64 %989, 32
  %994 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %992, i32 0
  %995 = getelementptr i16, i16 addrspace(4)* %994, i64 %993
  %996 = addrspacecast i16 addrspace(4)* %995 to i16 addrspace(1)*
  %997 = load i16, i16 addrspace(1)* %996, align 2
  %998 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %998)
  %999 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %999)
  %1000 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1000, align 4, !noalias !639
  store i32 %921, i32* %57, align 4, !noalias !639
  br label %1001

1001:                                             ; preds = %1001, %988
  %1002 = phi i32 [ 0, %988 ], [ %1007, %1001 ]
  %1003 = zext i32 %1002 to i64
  %1004 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1003
  %1005 = load i32, i32* %1004, align 4, !noalias !639
  %1006 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1003
  store i32 %1005, i32* %1006, align 4, !alias.scope !639
  %1007 = add nuw nsw i32 %1002, 1, !spirv.Decorations !620
  %1008 = icmp eq i32 %1002, 0
  br i1 %1008, label %1001, label %1009, !llvm.loop !642

1009:                                             ; preds = %1001
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %999)
  %1010 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %998)
  %1011 = shl i64 %1010, 32
  %1012 = ashr exact i64 %1011, 32
  %1013 = mul nsw i64 %1012, %const_reg_qword5, !spirv.Decorations !610
  %1014 = ashr i64 %1010, 32
  %1015 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1013, i32 0
  %1016 = getelementptr i16, i16 addrspace(4)* %1015, i64 %1014
  %1017 = addrspacecast i16 addrspace(4)* %1016 to i16 addrspace(1)*
  %1018 = load i16, i16 addrspace(1)* %1017, align 2
  %1019 = zext i16 %997 to i32
  %1020 = shl nuw i32 %1019, 16, !spirv.Decorations !628
  %1021 = bitcast i32 %1020 to float
  %1022 = zext i16 %1018 to i32
  %1023 = shl nuw i32 %1022, 16, !spirv.Decorations !628
  %1024 = bitcast i32 %1023 to float
  %1025 = fmul reassoc nsz arcp contract float %1021, %1024, !spirv.Decorations !612
  %1026 = fadd reassoc nsz arcp contract float %1025, %.sroa.82.1, !spirv.Decorations !612
  br label %._crit_edge.1.4

._crit_edge.1.4:                                  ; preds = %._crit_edge.4, %1009
  %.sroa.82.2 = phi float [ %1026, %1009 ], [ %.sroa.82.1, %._crit_edge.4 ]
  %1027 = and i1 %184, %922
  br i1 %1027, label %1028, label %._crit_edge.2.4

1028:                                             ; preds = %._crit_edge.1.4
  %1029 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1029)
  %1030 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1030)
  %1031 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %1031, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1032

1032:                                             ; preds = %1032, %1028
  %1033 = phi i32 [ 0, %1028 ], [ %1038, %1032 ]
  %1034 = zext i32 %1033 to i64
  %1035 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1034
  %1036 = load i32, i32* %1035, align 4, !noalias !635
  %1037 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1034
  store i32 %1036, i32* %1037, align 4, !alias.scope !635
  %1038 = add nuw nsw i32 %1033, 1, !spirv.Decorations !620
  %1039 = icmp eq i32 %1033, 0
  br i1 %1039, label %1032, label %1040, !llvm.loop !638

1040:                                             ; preds = %1032
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1030)
  %1041 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1029)
  %1042 = shl i64 %1041, 32
  %1043 = ashr exact i64 %1042, 32
  %1044 = mul nsw i64 %1043, %const_reg_qword3, !spirv.Decorations !610
  %1045 = ashr i64 %1041, 32
  %1046 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1044, i32 0
  %1047 = getelementptr i16, i16 addrspace(4)* %1046, i64 %1045
  %1048 = addrspacecast i16 addrspace(4)* %1047 to i16 addrspace(1)*
  %1049 = load i16, i16 addrspace(1)* %1048, align 2
  %1050 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1050)
  %1051 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1051)
  %1052 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1052, align 4, !noalias !639
  store i32 %921, i32* %57, align 4, !noalias !639
  br label %1053

1053:                                             ; preds = %1053, %1040
  %1054 = phi i32 [ 0, %1040 ], [ %1059, %1053 ]
  %1055 = zext i32 %1054 to i64
  %1056 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1055
  %1057 = load i32, i32* %1056, align 4, !noalias !639
  %1058 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1055
  store i32 %1057, i32* %1058, align 4, !alias.scope !639
  %1059 = add nuw nsw i32 %1054, 1, !spirv.Decorations !620
  %1060 = icmp eq i32 %1054, 0
  br i1 %1060, label %1053, label %1061, !llvm.loop !642

1061:                                             ; preds = %1053
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1051)
  %1062 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1050)
  %1063 = shl i64 %1062, 32
  %1064 = ashr exact i64 %1063, 32
  %1065 = mul nsw i64 %1064, %const_reg_qword5, !spirv.Decorations !610
  %1066 = ashr i64 %1062, 32
  %1067 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1065, i32 0
  %1068 = getelementptr i16, i16 addrspace(4)* %1067, i64 %1066
  %1069 = addrspacecast i16 addrspace(4)* %1068 to i16 addrspace(1)*
  %1070 = load i16, i16 addrspace(1)* %1069, align 2
  %1071 = zext i16 %1049 to i32
  %1072 = shl nuw i32 %1071, 16, !spirv.Decorations !628
  %1073 = bitcast i32 %1072 to float
  %1074 = zext i16 %1070 to i32
  %1075 = shl nuw i32 %1074, 16, !spirv.Decorations !628
  %1076 = bitcast i32 %1075 to float
  %1077 = fmul reassoc nsz arcp contract float %1073, %1076, !spirv.Decorations !612
  %1078 = fadd reassoc nsz arcp contract float %1077, %.sroa.146.1, !spirv.Decorations !612
  br label %._crit_edge.2.4

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4, %1061
  %.sroa.146.2 = phi float [ %1078, %1061 ], [ %.sroa.146.1, %._crit_edge.1.4 ]
  %1079 = and i1 %238, %922
  br i1 %1079, label %1080, label %.preheader.4

1080:                                             ; preds = %._crit_edge.2.4
  %1081 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1081)
  %1082 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1082)
  %1083 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %1083, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1084

1084:                                             ; preds = %1084, %1080
  %1085 = phi i32 [ 0, %1080 ], [ %1090, %1084 ]
  %1086 = zext i32 %1085 to i64
  %1087 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1086
  %1088 = load i32, i32* %1087, align 4, !noalias !635
  %1089 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1086
  store i32 %1088, i32* %1089, align 4, !alias.scope !635
  %1090 = add nuw nsw i32 %1085, 1, !spirv.Decorations !620
  %1091 = icmp eq i32 %1085, 0
  br i1 %1091, label %1084, label %1092, !llvm.loop !638

1092:                                             ; preds = %1084
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1082)
  %1093 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1081)
  %1094 = shl i64 %1093, 32
  %1095 = ashr exact i64 %1094, 32
  %1096 = mul nsw i64 %1095, %const_reg_qword3, !spirv.Decorations !610
  %1097 = ashr i64 %1093, 32
  %1098 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1096, i32 0
  %1099 = getelementptr i16, i16 addrspace(4)* %1098, i64 %1097
  %1100 = addrspacecast i16 addrspace(4)* %1099 to i16 addrspace(1)*
  %1101 = load i16, i16 addrspace(1)* %1100, align 2
  %1102 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1102)
  %1103 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1103)
  %1104 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1104, align 4, !noalias !639
  store i32 %921, i32* %57, align 4, !noalias !639
  br label %1105

1105:                                             ; preds = %1105, %1092
  %1106 = phi i32 [ 0, %1092 ], [ %1111, %1105 ]
  %1107 = zext i32 %1106 to i64
  %1108 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1107
  %1109 = load i32, i32* %1108, align 4, !noalias !639
  %1110 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1107
  store i32 %1109, i32* %1110, align 4, !alias.scope !639
  %1111 = add nuw nsw i32 %1106, 1, !spirv.Decorations !620
  %1112 = icmp eq i32 %1106, 0
  br i1 %1112, label %1105, label %1113, !llvm.loop !642

1113:                                             ; preds = %1105
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1103)
  %1114 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1102)
  %1115 = shl i64 %1114, 32
  %1116 = ashr exact i64 %1115, 32
  %1117 = mul nsw i64 %1116, %const_reg_qword5, !spirv.Decorations !610
  %1118 = ashr i64 %1114, 32
  %1119 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1117, i32 0
  %1120 = getelementptr i16, i16 addrspace(4)* %1119, i64 %1118
  %1121 = addrspacecast i16 addrspace(4)* %1120 to i16 addrspace(1)*
  %1122 = load i16, i16 addrspace(1)* %1121, align 2
  %1123 = zext i16 %1101 to i32
  %1124 = shl nuw i32 %1123, 16, !spirv.Decorations !628
  %1125 = bitcast i32 %1124 to float
  %1126 = zext i16 %1122 to i32
  %1127 = shl nuw i32 %1126, 16, !spirv.Decorations !628
  %1128 = bitcast i32 %1127 to float
  %1129 = fmul reassoc nsz arcp contract float %1125, %1128, !spirv.Decorations !612
  %1130 = fadd reassoc nsz arcp contract float %1129, %.sroa.210.1, !spirv.Decorations !612
  br label %.preheader.4

.preheader.4:                                     ; preds = %._crit_edge.2.4, %1113
  %.sroa.210.2 = phi float [ %1130, %1113 ], [ %.sroa.210.1, %._crit_edge.2.4 ]
  %1131 = or i32 %41, 5
  %1132 = icmp slt i32 %1131, %const_reg_dword1
  %1133 = and i1 %76, %1132
  br i1 %1133, label %1134, label %._crit_edge.5

1134:                                             ; preds = %.preheader.4
  %1135 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1135)
  %1136 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1136)
  %1137 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1137, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1138

1138:                                             ; preds = %1138, %1134
  %1139 = phi i32 [ 0, %1134 ], [ %1144, %1138 ]
  %1140 = zext i32 %1139 to i64
  %1141 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1140
  %1142 = load i32, i32* %1141, align 4, !noalias !635
  %1143 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1140
  store i32 %1142, i32* %1143, align 4, !alias.scope !635
  %1144 = add nuw nsw i32 %1139, 1, !spirv.Decorations !620
  %1145 = icmp eq i32 %1139, 0
  br i1 %1145, label %1138, label %1146, !llvm.loop !638

1146:                                             ; preds = %1138
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1136)
  %1147 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1135)
  %1148 = shl i64 %1147, 32
  %1149 = ashr exact i64 %1148, 32
  %1150 = mul nsw i64 %1149, %const_reg_qword3, !spirv.Decorations !610
  %1151 = ashr i64 %1147, 32
  %1152 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1150, i32 0
  %1153 = getelementptr i16, i16 addrspace(4)* %1152, i64 %1151
  %1154 = addrspacecast i16 addrspace(4)* %1153 to i16 addrspace(1)*
  %1155 = load i16, i16 addrspace(1)* %1154, align 2
  %1156 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1156)
  %1157 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1157)
  %1158 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1158, align 4, !noalias !639
  store i32 %1131, i32* %57, align 4, !noalias !639
  br label %1159

1159:                                             ; preds = %1159, %1146
  %1160 = phi i32 [ 0, %1146 ], [ %1165, %1159 ]
  %1161 = zext i32 %1160 to i64
  %1162 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1161
  %1163 = load i32, i32* %1162, align 4, !noalias !639
  %1164 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1161
  store i32 %1163, i32* %1164, align 4, !alias.scope !639
  %1165 = add nuw nsw i32 %1160, 1, !spirv.Decorations !620
  %1166 = icmp eq i32 %1160, 0
  br i1 %1166, label %1159, label %1167, !llvm.loop !642

1167:                                             ; preds = %1159
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1157)
  %1168 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1156)
  %1169 = shl i64 %1168, 32
  %1170 = ashr exact i64 %1169, 32
  %1171 = mul nsw i64 %1170, %const_reg_qword5, !spirv.Decorations !610
  %1172 = ashr i64 %1168, 32
  %1173 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1171, i32 0
  %1174 = getelementptr i16, i16 addrspace(4)* %1173, i64 %1172
  %1175 = addrspacecast i16 addrspace(4)* %1174 to i16 addrspace(1)*
  %1176 = load i16, i16 addrspace(1)* %1175, align 2
  %1177 = zext i16 %1155 to i32
  %1178 = shl nuw i32 %1177, 16, !spirv.Decorations !628
  %1179 = bitcast i32 %1178 to float
  %1180 = zext i16 %1176 to i32
  %1181 = shl nuw i32 %1180, 16, !spirv.Decorations !628
  %1182 = bitcast i32 %1181 to float
  %1183 = fmul reassoc nsz arcp contract float %1179, %1182, !spirv.Decorations !612
  %1184 = fadd reassoc nsz arcp contract float %1183, %.sroa.22.1, !spirv.Decorations !612
  br label %._crit_edge.5

._crit_edge.5:                                    ; preds = %.preheader.4, %1167
  %.sroa.22.2 = phi float [ %1184, %1167 ], [ %.sroa.22.1, %.preheader.4 ]
  %1185 = and i1 %130, %1132
  br i1 %1185, label %1186, label %._crit_edge.1.5

1186:                                             ; preds = %._crit_edge.5
  %1187 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1187)
  %1188 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1188)
  %1189 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %1189, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1190

1190:                                             ; preds = %1190, %1186
  %1191 = phi i32 [ 0, %1186 ], [ %1196, %1190 ]
  %1192 = zext i32 %1191 to i64
  %1193 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1192
  %1194 = load i32, i32* %1193, align 4, !noalias !635
  %1195 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1192
  store i32 %1194, i32* %1195, align 4, !alias.scope !635
  %1196 = add nuw nsw i32 %1191, 1, !spirv.Decorations !620
  %1197 = icmp eq i32 %1191, 0
  br i1 %1197, label %1190, label %1198, !llvm.loop !638

1198:                                             ; preds = %1190
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1188)
  %1199 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1187)
  %1200 = shl i64 %1199, 32
  %1201 = ashr exact i64 %1200, 32
  %1202 = mul nsw i64 %1201, %const_reg_qword3, !spirv.Decorations !610
  %1203 = ashr i64 %1199, 32
  %1204 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1202, i32 0
  %1205 = getelementptr i16, i16 addrspace(4)* %1204, i64 %1203
  %1206 = addrspacecast i16 addrspace(4)* %1205 to i16 addrspace(1)*
  %1207 = load i16, i16 addrspace(1)* %1206, align 2
  %1208 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1208)
  %1209 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1209)
  %1210 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1210, align 4, !noalias !639
  store i32 %1131, i32* %57, align 4, !noalias !639
  br label %1211

1211:                                             ; preds = %1211, %1198
  %1212 = phi i32 [ 0, %1198 ], [ %1217, %1211 ]
  %1213 = zext i32 %1212 to i64
  %1214 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1213
  %1215 = load i32, i32* %1214, align 4, !noalias !639
  %1216 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1213
  store i32 %1215, i32* %1216, align 4, !alias.scope !639
  %1217 = add nuw nsw i32 %1212, 1, !spirv.Decorations !620
  %1218 = icmp eq i32 %1212, 0
  br i1 %1218, label %1211, label %1219, !llvm.loop !642

1219:                                             ; preds = %1211
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1209)
  %1220 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1208)
  %1221 = shl i64 %1220, 32
  %1222 = ashr exact i64 %1221, 32
  %1223 = mul nsw i64 %1222, %const_reg_qword5, !spirv.Decorations !610
  %1224 = ashr i64 %1220, 32
  %1225 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1223, i32 0
  %1226 = getelementptr i16, i16 addrspace(4)* %1225, i64 %1224
  %1227 = addrspacecast i16 addrspace(4)* %1226 to i16 addrspace(1)*
  %1228 = load i16, i16 addrspace(1)* %1227, align 2
  %1229 = zext i16 %1207 to i32
  %1230 = shl nuw i32 %1229, 16, !spirv.Decorations !628
  %1231 = bitcast i32 %1230 to float
  %1232 = zext i16 %1228 to i32
  %1233 = shl nuw i32 %1232, 16, !spirv.Decorations !628
  %1234 = bitcast i32 %1233 to float
  %1235 = fmul reassoc nsz arcp contract float %1231, %1234, !spirv.Decorations !612
  %1236 = fadd reassoc nsz arcp contract float %1235, %.sroa.86.1, !spirv.Decorations !612
  br label %._crit_edge.1.5

._crit_edge.1.5:                                  ; preds = %._crit_edge.5, %1219
  %.sroa.86.2 = phi float [ %1236, %1219 ], [ %.sroa.86.1, %._crit_edge.5 ]
  %1237 = and i1 %184, %1132
  br i1 %1237, label %1238, label %._crit_edge.2.5

1238:                                             ; preds = %._crit_edge.1.5
  %1239 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1239)
  %1240 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1240)
  %1241 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %1241, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1242

1242:                                             ; preds = %1242, %1238
  %1243 = phi i32 [ 0, %1238 ], [ %1248, %1242 ]
  %1244 = zext i32 %1243 to i64
  %1245 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1244
  %1246 = load i32, i32* %1245, align 4, !noalias !635
  %1247 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1244
  store i32 %1246, i32* %1247, align 4, !alias.scope !635
  %1248 = add nuw nsw i32 %1243, 1, !spirv.Decorations !620
  %1249 = icmp eq i32 %1243, 0
  br i1 %1249, label %1242, label %1250, !llvm.loop !638

1250:                                             ; preds = %1242
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1240)
  %1251 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1239)
  %1252 = shl i64 %1251, 32
  %1253 = ashr exact i64 %1252, 32
  %1254 = mul nsw i64 %1253, %const_reg_qword3, !spirv.Decorations !610
  %1255 = ashr i64 %1251, 32
  %1256 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1254, i32 0
  %1257 = getelementptr i16, i16 addrspace(4)* %1256, i64 %1255
  %1258 = addrspacecast i16 addrspace(4)* %1257 to i16 addrspace(1)*
  %1259 = load i16, i16 addrspace(1)* %1258, align 2
  %1260 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1260)
  %1261 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1261)
  %1262 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1262, align 4, !noalias !639
  store i32 %1131, i32* %57, align 4, !noalias !639
  br label %1263

1263:                                             ; preds = %1263, %1250
  %1264 = phi i32 [ 0, %1250 ], [ %1269, %1263 ]
  %1265 = zext i32 %1264 to i64
  %1266 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1265
  %1267 = load i32, i32* %1266, align 4, !noalias !639
  %1268 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1265
  store i32 %1267, i32* %1268, align 4, !alias.scope !639
  %1269 = add nuw nsw i32 %1264, 1, !spirv.Decorations !620
  %1270 = icmp eq i32 %1264, 0
  br i1 %1270, label %1263, label %1271, !llvm.loop !642

1271:                                             ; preds = %1263
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1261)
  %1272 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1260)
  %1273 = shl i64 %1272, 32
  %1274 = ashr exact i64 %1273, 32
  %1275 = mul nsw i64 %1274, %const_reg_qword5, !spirv.Decorations !610
  %1276 = ashr i64 %1272, 32
  %1277 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1275, i32 0
  %1278 = getelementptr i16, i16 addrspace(4)* %1277, i64 %1276
  %1279 = addrspacecast i16 addrspace(4)* %1278 to i16 addrspace(1)*
  %1280 = load i16, i16 addrspace(1)* %1279, align 2
  %1281 = zext i16 %1259 to i32
  %1282 = shl nuw i32 %1281, 16, !spirv.Decorations !628
  %1283 = bitcast i32 %1282 to float
  %1284 = zext i16 %1280 to i32
  %1285 = shl nuw i32 %1284, 16, !spirv.Decorations !628
  %1286 = bitcast i32 %1285 to float
  %1287 = fmul reassoc nsz arcp contract float %1283, %1286, !spirv.Decorations !612
  %1288 = fadd reassoc nsz arcp contract float %1287, %.sroa.150.1, !spirv.Decorations !612
  br label %._crit_edge.2.5

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5, %1271
  %.sroa.150.2 = phi float [ %1288, %1271 ], [ %.sroa.150.1, %._crit_edge.1.5 ]
  %1289 = and i1 %238, %1132
  br i1 %1289, label %1290, label %.preheader.5

1290:                                             ; preds = %._crit_edge.2.5
  %1291 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1291)
  %1292 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1292)
  %1293 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %1293, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1294

1294:                                             ; preds = %1294, %1290
  %1295 = phi i32 [ 0, %1290 ], [ %1300, %1294 ]
  %1296 = zext i32 %1295 to i64
  %1297 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1296
  %1298 = load i32, i32* %1297, align 4, !noalias !635
  %1299 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1296
  store i32 %1298, i32* %1299, align 4, !alias.scope !635
  %1300 = add nuw nsw i32 %1295, 1, !spirv.Decorations !620
  %1301 = icmp eq i32 %1295, 0
  br i1 %1301, label %1294, label %1302, !llvm.loop !638

1302:                                             ; preds = %1294
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1292)
  %1303 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1291)
  %1304 = shl i64 %1303, 32
  %1305 = ashr exact i64 %1304, 32
  %1306 = mul nsw i64 %1305, %const_reg_qword3, !spirv.Decorations !610
  %1307 = ashr i64 %1303, 32
  %1308 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1306, i32 0
  %1309 = getelementptr i16, i16 addrspace(4)* %1308, i64 %1307
  %1310 = addrspacecast i16 addrspace(4)* %1309 to i16 addrspace(1)*
  %1311 = load i16, i16 addrspace(1)* %1310, align 2
  %1312 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1312)
  %1313 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1313)
  %1314 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1314, align 4, !noalias !639
  store i32 %1131, i32* %57, align 4, !noalias !639
  br label %1315

1315:                                             ; preds = %1315, %1302
  %1316 = phi i32 [ 0, %1302 ], [ %1321, %1315 ]
  %1317 = zext i32 %1316 to i64
  %1318 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1317
  %1319 = load i32, i32* %1318, align 4, !noalias !639
  %1320 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1317
  store i32 %1319, i32* %1320, align 4, !alias.scope !639
  %1321 = add nuw nsw i32 %1316, 1, !spirv.Decorations !620
  %1322 = icmp eq i32 %1316, 0
  br i1 %1322, label %1315, label %1323, !llvm.loop !642

1323:                                             ; preds = %1315
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1313)
  %1324 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1312)
  %1325 = shl i64 %1324, 32
  %1326 = ashr exact i64 %1325, 32
  %1327 = mul nsw i64 %1326, %const_reg_qword5, !spirv.Decorations !610
  %1328 = ashr i64 %1324, 32
  %1329 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1327, i32 0
  %1330 = getelementptr i16, i16 addrspace(4)* %1329, i64 %1328
  %1331 = addrspacecast i16 addrspace(4)* %1330 to i16 addrspace(1)*
  %1332 = load i16, i16 addrspace(1)* %1331, align 2
  %1333 = zext i16 %1311 to i32
  %1334 = shl nuw i32 %1333, 16, !spirv.Decorations !628
  %1335 = bitcast i32 %1334 to float
  %1336 = zext i16 %1332 to i32
  %1337 = shl nuw i32 %1336, 16, !spirv.Decorations !628
  %1338 = bitcast i32 %1337 to float
  %1339 = fmul reassoc nsz arcp contract float %1335, %1338, !spirv.Decorations !612
  %1340 = fadd reassoc nsz arcp contract float %1339, %.sroa.214.1, !spirv.Decorations !612
  br label %.preheader.5

.preheader.5:                                     ; preds = %._crit_edge.2.5, %1323
  %.sroa.214.2 = phi float [ %1340, %1323 ], [ %.sroa.214.1, %._crit_edge.2.5 ]
  %1341 = or i32 %41, 6
  %1342 = icmp slt i32 %1341, %const_reg_dword1
  %1343 = and i1 %76, %1342
  br i1 %1343, label %1344, label %._crit_edge.6

1344:                                             ; preds = %.preheader.5
  %1345 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1345)
  %1346 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1346)
  %1347 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1347, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1348

1348:                                             ; preds = %1348, %1344
  %1349 = phi i32 [ 0, %1344 ], [ %1354, %1348 ]
  %1350 = zext i32 %1349 to i64
  %1351 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1350
  %1352 = load i32, i32* %1351, align 4, !noalias !635
  %1353 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1350
  store i32 %1352, i32* %1353, align 4, !alias.scope !635
  %1354 = add nuw nsw i32 %1349, 1, !spirv.Decorations !620
  %1355 = icmp eq i32 %1349, 0
  br i1 %1355, label %1348, label %1356, !llvm.loop !638

1356:                                             ; preds = %1348
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1346)
  %1357 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1345)
  %1358 = shl i64 %1357, 32
  %1359 = ashr exact i64 %1358, 32
  %1360 = mul nsw i64 %1359, %const_reg_qword3, !spirv.Decorations !610
  %1361 = ashr i64 %1357, 32
  %1362 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1360, i32 0
  %1363 = getelementptr i16, i16 addrspace(4)* %1362, i64 %1361
  %1364 = addrspacecast i16 addrspace(4)* %1363 to i16 addrspace(1)*
  %1365 = load i16, i16 addrspace(1)* %1364, align 2
  %1366 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1366)
  %1367 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1367)
  %1368 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1368, align 4, !noalias !639
  store i32 %1341, i32* %57, align 4, !noalias !639
  br label %1369

1369:                                             ; preds = %1369, %1356
  %1370 = phi i32 [ 0, %1356 ], [ %1375, %1369 ]
  %1371 = zext i32 %1370 to i64
  %1372 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1371
  %1373 = load i32, i32* %1372, align 4, !noalias !639
  %1374 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1371
  store i32 %1373, i32* %1374, align 4, !alias.scope !639
  %1375 = add nuw nsw i32 %1370, 1, !spirv.Decorations !620
  %1376 = icmp eq i32 %1370, 0
  br i1 %1376, label %1369, label %1377, !llvm.loop !642

1377:                                             ; preds = %1369
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1367)
  %1378 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1366)
  %1379 = shl i64 %1378, 32
  %1380 = ashr exact i64 %1379, 32
  %1381 = mul nsw i64 %1380, %const_reg_qword5, !spirv.Decorations !610
  %1382 = ashr i64 %1378, 32
  %1383 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1381, i32 0
  %1384 = getelementptr i16, i16 addrspace(4)* %1383, i64 %1382
  %1385 = addrspacecast i16 addrspace(4)* %1384 to i16 addrspace(1)*
  %1386 = load i16, i16 addrspace(1)* %1385, align 2
  %1387 = zext i16 %1365 to i32
  %1388 = shl nuw i32 %1387, 16, !spirv.Decorations !628
  %1389 = bitcast i32 %1388 to float
  %1390 = zext i16 %1386 to i32
  %1391 = shl nuw i32 %1390, 16, !spirv.Decorations !628
  %1392 = bitcast i32 %1391 to float
  %1393 = fmul reassoc nsz arcp contract float %1389, %1392, !spirv.Decorations !612
  %1394 = fadd reassoc nsz arcp contract float %1393, %.sroa.26.1, !spirv.Decorations !612
  br label %._crit_edge.6

._crit_edge.6:                                    ; preds = %.preheader.5, %1377
  %.sroa.26.2 = phi float [ %1394, %1377 ], [ %.sroa.26.1, %.preheader.5 ]
  %1395 = and i1 %130, %1342
  br i1 %1395, label %1396, label %._crit_edge.1.6

1396:                                             ; preds = %._crit_edge.6
  %1397 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1397)
  %1398 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1398)
  %1399 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %1399, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1400

1400:                                             ; preds = %1400, %1396
  %1401 = phi i32 [ 0, %1396 ], [ %1406, %1400 ]
  %1402 = zext i32 %1401 to i64
  %1403 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1402
  %1404 = load i32, i32* %1403, align 4, !noalias !635
  %1405 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1402
  store i32 %1404, i32* %1405, align 4, !alias.scope !635
  %1406 = add nuw nsw i32 %1401, 1, !spirv.Decorations !620
  %1407 = icmp eq i32 %1401, 0
  br i1 %1407, label %1400, label %1408, !llvm.loop !638

1408:                                             ; preds = %1400
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1398)
  %1409 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1397)
  %1410 = shl i64 %1409, 32
  %1411 = ashr exact i64 %1410, 32
  %1412 = mul nsw i64 %1411, %const_reg_qword3, !spirv.Decorations !610
  %1413 = ashr i64 %1409, 32
  %1414 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1412, i32 0
  %1415 = getelementptr i16, i16 addrspace(4)* %1414, i64 %1413
  %1416 = addrspacecast i16 addrspace(4)* %1415 to i16 addrspace(1)*
  %1417 = load i16, i16 addrspace(1)* %1416, align 2
  %1418 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1418)
  %1419 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1419)
  %1420 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1420, align 4, !noalias !639
  store i32 %1341, i32* %57, align 4, !noalias !639
  br label %1421

1421:                                             ; preds = %1421, %1408
  %1422 = phi i32 [ 0, %1408 ], [ %1427, %1421 ]
  %1423 = zext i32 %1422 to i64
  %1424 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1423
  %1425 = load i32, i32* %1424, align 4, !noalias !639
  %1426 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1423
  store i32 %1425, i32* %1426, align 4, !alias.scope !639
  %1427 = add nuw nsw i32 %1422, 1, !spirv.Decorations !620
  %1428 = icmp eq i32 %1422, 0
  br i1 %1428, label %1421, label %1429, !llvm.loop !642

1429:                                             ; preds = %1421
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1419)
  %1430 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1418)
  %1431 = shl i64 %1430, 32
  %1432 = ashr exact i64 %1431, 32
  %1433 = mul nsw i64 %1432, %const_reg_qword5, !spirv.Decorations !610
  %1434 = ashr i64 %1430, 32
  %1435 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1433, i32 0
  %1436 = getelementptr i16, i16 addrspace(4)* %1435, i64 %1434
  %1437 = addrspacecast i16 addrspace(4)* %1436 to i16 addrspace(1)*
  %1438 = load i16, i16 addrspace(1)* %1437, align 2
  %1439 = zext i16 %1417 to i32
  %1440 = shl nuw i32 %1439, 16, !spirv.Decorations !628
  %1441 = bitcast i32 %1440 to float
  %1442 = zext i16 %1438 to i32
  %1443 = shl nuw i32 %1442, 16, !spirv.Decorations !628
  %1444 = bitcast i32 %1443 to float
  %1445 = fmul reassoc nsz arcp contract float %1441, %1444, !spirv.Decorations !612
  %1446 = fadd reassoc nsz arcp contract float %1445, %.sroa.90.1, !spirv.Decorations !612
  br label %._crit_edge.1.6

._crit_edge.1.6:                                  ; preds = %._crit_edge.6, %1429
  %.sroa.90.2 = phi float [ %1446, %1429 ], [ %.sroa.90.1, %._crit_edge.6 ]
  %1447 = and i1 %184, %1342
  br i1 %1447, label %1448, label %._crit_edge.2.6

1448:                                             ; preds = %._crit_edge.1.6
  %1449 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1449)
  %1450 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1450)
  %1451 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %1451, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1452

1452:                                             ; preds = %1452, %1448
  %1453 = phi i32 [ 0, %1448 ], [ %1458, %1452 ]
  %1454 = zext i32 %1453 to i64
  %1455 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1454
  %1456 = load i32, i32* %1455, align 4, !noalias !635
  %1457 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1454
  store i32 %1456, i32* %1457, align 4, !alias.scope !635
  %1458 = add nuw nsw i32 %1453, 1, !spirv.Decorations !620
  %1459 = icmp eq i32 %1453, 0
  br i1 %1459, label %1452, label %1460, !llvm.loop !638

1460:                                             ; preds = %1452
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1450)
  %1461 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1449)
  %1462 = shl i64 %1461, 32
  %1463 = ashr exact i64 %1462, 32
  %1464 = mul nsw i64 %1463, %const_reg_qword3, !spirv.Decorations !610
  %1465 = ashr i64 %1461, 32
  %1466 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1464, i32 0
  %1467 = getelementptr i16, i16 addrspace(4)* %1466, i64 %1465
  %1468 = addrspacecast i16 addrspace(4)* %1467 to i16 addrspace(1)*
  %1469 = load i16, i16 addrspace(1)* %1468, align 2
  %1470 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1470)
  %1471 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1471)
  %1472 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1472, align 4, !noalias !639
  store i32 %1341, i32* %57, align 4, !noalias !639
  br label %1473

1473:                                             ; preds = %1473, %1460
  %1474 = phi i32 [ 0, %1460 ], [ %1479, %1473 ]
  %1475 = zext i32 %1474 to i64
  %1476 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1475
  %1477 = load i32, i32* %1476, align 4, !noalias !639
  %1478 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1475
  store i32 %1477, i32* %1478, align 4, !alias.scope !639
  %1479 = add nuw nsw i32 %1474, 1, !spirv.Decorations !620
  %1480 = icmp eq i32 %1474, 0
  br i1 %1480, label %1473, label %1481, !llvm.loop !642

1481:                                             ; preds = %1473
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1471)
  %1482 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1470)
  %1483 = shl i64 %1482, 32
  %1484 = ashr exact i64 %1483, 32
  %1485 = mul nsw i64 %1484, %const_reg_qword5, !spirv.Decorations !610
  %1486 = ashr i64 %1482, 32
  %1487 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1485, i32 0
  %1488 = getelementptr i16, i16 addrspace(4)* %1487, i64 %1486
  %1489 = addrspacecast i16 addrspace(4)* %1488 to i16 addrspace(1)*
  %1490 = load i16, i16 addrspace(1)* %1489, align 2
  %1491 = zext i16 %1469 to i32
  %1492 = shl nuw i32 %1491, 16, !spirv.Decorations !628
  %1493 = bitcast i32 %1492 to float
  %1494 = zext i16 %1490 to i32
  %1495 = shl nuw i32 %1494, 16, !spirv.Decorations !628
  %1496 = bitcast i32 %1495 to float
  %1497 = fmul reassoc nsz arcp contract float %1493, %1496, !spirv.Decorations !612
  %1498 = fadd reassoc nsz arcp contract float %1497, %.sroa.154.1, !spirv.Decorations !612
  br label %._crit_edge.2.6

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6, %1481
  %.sroa.154.2 = phi float [ %1498, %1481 ], [ %.sroa.154.1, %._crit_edge.1.6 ]
  %1499 = and i1 %238, %1342
  br i1 %1499, label %1500, label %.preheader.6

1500:                                             ; preds = %._crit_edge.2.6
  %1501 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1501)
  %1502 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1502)
  %1503 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %1503, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1504

1504:                                             ; preds = %1504, %1500
  %1505 = phi i32 [ 0, %1500 ], [ %1510, %1504 ]
  %1506 = zext i32 %1505 to i64
  %1507 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1506
  %1508 = load i32, i32* %1507, align 4, !noalias !635
  %1509 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1506
  store i32 %1508, i32* %1509, align 4, !alias.scope !635
  %1510 = add nuw nsw i32 %1505, 1, !spirv.Decorations !620
  %1511 = icmp eq i32 %1505, 0
  br i1 %1511, label %1504, label %1512, !llvm.loop !638

1512:                                             ; preds = %1504
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1502)
  %1513 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1501)
  %1514 = shl i64 %1513, 32
  %1515 = ashr exact i64 %1514, 32
  %1516 = mul nsw i64 %1515, %const_reg_qword3, !spirv.Decorations !610
  %1517 = ashr i64 %1513, 32
  %1518 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1516, i32 0
  %1519 = getelementptr i16, i16 addrspace(4)* %1518, i64 %1517
  %1520 = addrspacecast i16 addrspace(4)* %1519 to i16 addrspace(1)*
  %1521 = load i16, i16 addrspace(1)* %1520, align 2
  %1522 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1522)
  %1523 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1523)
  %1524 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1524, align 4, !noalias !639
  store i32 %1341, i32* %57, align 4, !noalias !639
  br label %1525

1525:                                             ; preds = %1525, %1512
  %1526 = phi i32 [ 0, %1512 ], [ %1531, %1525 ]
  %1527 = zext i32 %1526 to i64
  %1528 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1527
  %1529 = load i32, i32* %1528, align 4, !noalias !639
  %1530 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1527
  store i32 %1529, i32* %1530, align 4, !alias.scope !639
  %1531 = add nuw nsw i32 %1526, 1, !spirv.Decorations !620
  %1532 = icmp eq i32 %1526, 0
  br i1 %1532, label %1525, label %1533, !llvm.loop !642

1533:                                             ; preds = %1525
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1523)
  %1534 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1522)
  %1535 = shl i64 %1534, 32
  %1536 = ashr exact i64 %1535, 32
  %1537 = mul nsw i64 %1536, %const_reg_qword5, !spirv.Decorations !610
  %1538 = ashr i64 %1534, 32
  %1539 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1537, i32 0
  %1540 = getelementptr i16, i16 addrspace(4)* %1539, i64 %1538
  %1541 = addrspacecast i16 addrspace(4)* %1540 to i16 addrspace(1)*
  %1542 = load i16, i16 addrspace(1)* %1541, align 2
  %1543 = zext i16 %1521 to i32
  %1544 = shl nuw i32 %1543, 16, !spirv.Decorations !628
  %1545 = bitcast i32 %1544 to float
  %1546 = zext i16 %1542 to i32
  %1547 = shl nuw i32 %1546, 16, !spirv.Decorations !628
  %1548 = bitcast i32 %1547 to float
  %1549 = fmul reassoc nsz arcp contract float %1545, %1548, !spirv.Decorations !612
  %1550 = fadd reassoc nsz arcp contract float %1549, %.sroa.218.1, !spirv.Decorations !612
  br label %.preheader.6

.preheader.6:                                     ; preds = %._crit_edge.2.6, %1533
  %.sroa.218.2 = phi float [ %1550, %1533 ], [ %.sroa.218.1, %._crit_edge.2.6 ]
  %1551 = or i32 %41, 7
  %1552 = icmp slt i32 %1551, %const_reg_dword1
  %1553 = and i1 %76, %1552
  br i1 %1553, label %1554, label %._crit_edge.7

1554:                                             ; preds = %.preheader.6
  %1555 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1555)
  %1556 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1556)
  %1557 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1557, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1558

1558:                                             ; preds = %1558, %1554
  %1559 = phi i32 [ 0, %1554 ], [ %1564, %1558 ]
  %1560 = zext i32 %1559 to i64
  %1561 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1560
  %1562 = load i32, i32* %1561, align 4, !noalias !635
  %1563 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1560
  store i32 %1562, i32* %1563, align 4, !alias.scope !635
  %1564 = add nuw nsw i32 %1559, 1, !spirv.Decorations !620
  %1565 = icmp eq i32 %1559, 0
  br i1 %1565, label %1558, label %1566, !llvm.loop !638

1566:                                             ; preds = %1558
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1556)
  %1567 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1555)
  %1568 = shl i64 %1567, 32
  %1569 = ashr exact i64 %1568, 32
  %1570 = mul nsw i64 %1569, %const_reg_qword3, !spirv.Decorations !610
  %1571 = ashr i64 %1567, 32
  %1572 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1570, i32 0
  %1573 = getelementptr i16, i16 addrspace(4)* %1572, i64 %1571
  %1574 = addrspacecast i16 addrspace(4)* %1573 to i16 addrspace(1)*
  %1575 = load i16, i16 addrspace(1)* %1574, align 2
  %1576 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1576)
  %1577 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1577)
  %1578 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1578, align 4, !noalias !639
  store i32 %1551, i32* %57, align 4, !noalias !639
  br label %1579

1579:                                             ; preds = %1579, %1566
  %1580 = phi i32 [ 0, %1566 ], [ %1585, %1579 ]
  %1581 = zext i32 %1580 to i64
  %1582 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1581
  %1583 = load i32, i32* %1582, align 4, !noalias !639
  %1584 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1581
  store i32 %1583, i32* %1584, align 4, !alias.scope !639
  %1585 = add nuw nsw i32 %1580, 1, !spirv.Decorations !620
  %1586 = icmp eq i32 %1580, 0
  br i1 %1586, label %1579, label %1587, !llvm.loop !642

1587:                                             ; preds = %1579
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1577)
  %1588 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1576)
  %1589 = shl i64 %1588, 32
  %1590 = ashr exact i64 %1589, 32
  %1591 = mul nsw i64 %1590, %const_reg_qword5, !spirv.Decorations !610
  %1592 = ashr i64 %1588, 32
  %1593 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1591, i32 0
  %1594 = getelementptr i16, i16 addrspace(4)* %1593, i64 %1592
  %1595 = addrspacecast i16 addrspace(4)* %1594 to i16 addrspace(1)*
  %1596 = load i16, i16 addrspace(1)* %1595, align 2
  %1597 = zext i16 %1575 to i32
  %1598 = shl nuw i32 %1597, 16, !spirv.Decorations !628
  %1599 = bitcast i32 %1598 to float
  %1600 = zext i16 %1596 to i32
  %1601 = shl nuw i32 %1600, 16, !spirv.Decorations !628
  %1602 = bitcast i32 %1601 to float
  %1603 = fmul reassoc nsz arcp contract float %1599, %1602, !spirv.Decorations !612
  %1604 = fadd reassoc nsz arcp contract float %1603, %.sroa.30.1, !spirv.Decorations !612
  br label %._crit_edge.7

._crit_edge.7:                                    ; preds = %.preheader.6, %1587
  %.sroa.30.2 = phi float [ %1604, %1587 ], [ %.sroa.30.1, %.preheader.6 ]
  %1605 = and i1 %130, %1552
  br i1 %1605, label %1606, label %._crit_edge.1.7

1606:                                             ; preds = %._crit_edge.7
  %1607 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1607)
  %1608 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1608)
  %1609 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %1609, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1610

1610:                                             ; preds = %1610, %1606
  %1611 = phi i32 [ 0, %1606 ], [ %1616, %1610 ]
  %1612 = zext i32 %1611 to i64
  %1613 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1612
  %1614 = load i32, i32* %1613, align 4, !noalias !635
  %1615 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1612
  store i32 %1614, i32* %1615, align 4, !alias.scope !635
  %1616 = add nuw nsw i32 %1611, 1, !spirv.Decorations !620
  %1617 = icmp eq i32 %1611, 0
  br i1 %1617, label %1610, label %1618, !llvm.loop !638

1618:                                             ; preds = %1610
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1608)
  %1619 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1607)
  %1620 = shl i64 %1619, 32
  %1621 = ashr exact i64 %1620, 32
  %1622 = mul nsw i64 %1621, %const_reg_qword3, !spirv.Decorations !610
  %1623 = ashr i64 %1619, 32
  %1624 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1622, i32 0
  %1625 = getelementptr i16, i16 addrspace(4)* %1624, i64 %1623
  %1626 = addrspacecast i16 addrspace(4)* %1625 to i16 addrspace(1)*
  %1627 = load i16, i16 addrspace(1)* %1626, align 2
  %1628 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1628)
  %1629 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1629)
  %1630 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1630, align 4, !noalias !639
  store i32 %1551, i32* %57, align 4, !noalias !639
  br label %1631

1631:                                             ; preds = %1631, %1618
  %1632 = phi i32 [ 0, %1618 ], [ %1637, %1631 ]
  %1633 = zext i32 %1632 to i64
  %1634 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1633
  %1635 = load i32, i32* %1634, align 4, !noalias !639
  %1636 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1633
  store i32 %1635, i32* %1636, align 4, !alias.scope !639
  %1637 = add nuw nsw i32 %1632, 1, !spirv.Decorations !620
  %1638 = icmp eq i32 %1632, 0
  br i1 %1638, label %1631, label %1639, !llvm.loop !642

1639:                                             ; preds = %1631
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1629)
  %1640 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1628)
  %1641 = shl i64 %1640, 32
  %1642 = ashr exact i64 %1641, 32
  %1643 = mul nsw i64 %1642, %const_reg_qword5, !spirv.Decorations !610
  %1644 = ashr i64 %1640, 32
  %1645 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1643, i32 0
  %1646 = getelementptr i16, i16 addrspace(4)* %1645, i64 %1644
  %1647 = addrspacecast i16 addrspace(4)* %1646 to i16 addrspace(1)*
  %1648 = load i16, i16 addrspace(1)* %1647, align 2
  %1649 = zext i16 %1627 to i32
  %1650 = shl nuw i32 %1649, 16, !spirv.Decorations !628
  %1651 = bitcast i32 %1650 to float
  %1652 = zext i16 %1648 to i32
  %1653 = shl nuw i32 %1652, 16, !spirv.Decorations !628
  %1654 = bitcast i32 %1653 to float
  %1655 = fmul reassoc nsz arcp contract float %1651, %1654, !spirv.Decorations !612
  %1656 = fadd reassoc nsz arcp contract float %1655, %.sroa.94.1, !spirv.Decorations !612
  br label %._crit_edge.1.7

._crit_edge.1.7:                                  ; preds = %._crit_edge.7, %1639
  %.sroa.94.2 = phi float [ %1656, %1639 ], [ %.sroa.94.1, %._crit_edge.7 ]
  %1657 = and i1 %184, %1552
  br i1 %1657, label %1658, label %._crit_edge.2.7

1658:                                             ; preds = %._crit_edge.1.7
  %1659 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1659)
  %1660 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1660)
  %1661 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %1661, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1662

1662:                                             ; preds = %1662, %1658
  %1663 = phi i32 [ 0, %1658 ], [ %1668, %1662 ]
  %1664 = zext i32 %1663 to i64
  %1665 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1664
  %1666 = load i32, i32* %1665, align 4, !noalias !635
  %1667 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1664
  store i32 %1666, i32* %1667, align 4, !alias.scope !635
  %1668 = add nuw nsw i32 %1663, 1, !spirv.Decorations !620
  %1669 = icmp eq i32 %1663, 0
  br i1 %1669, label %1662, label %1670, !llvm.loop !638

1670:                                             ; preds = %1662
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1660)
  %1671 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1659)
  %1672 = shl i64 %1671, 32
  %1673 = ashr exact i64 %1672, 32
  %1674 = mul nsw i64 %1673, %const_reg_qword3, !spirv.Decorations !610
  %1675 = ashr i64 %1671, 32
  %1676 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1674, i32 0
  %1677 = getelementptr i16, i16 addrspace(4)* %1676, i64 %1675
  %1678 = addrspacecast i16 addrspace(4)* %1677 to i16 addrspace(1)*
  %1679 = load i16, i16 addrspace(1)* %1678, align 2
  %1680 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1680)
  %1681 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1681)
  %1682 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1682, align 4, !noalias !639
  store i32 %1551, i32* %57, align 4, !noalias !639
  br label %1683

1683:                                             ; preds = %1683, %1670
  %1684 = phi i32 [ 0, %1670 ], [ %1689, %1683 ]
  %1685 = zext i32 %1684 to i64
  %1686 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1685
  %1687 = load i32, i32* %1686, align 4, !noalias !639
  %1688 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1685
  store i32 %1687, i32* %1688, align 4, !alias.scope !639
  %1689 = add nuw nsw i32 %1684, 1, !spirv.Decorations !620
  %1690 = icmp eq i32 %1684, 0
  br i1 %1690, label %1683, label %1691, !llvm.loop !642

1691:                                             ; preds = %1683
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1681)
  %1692 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1680)
  %1693 = shl i64 %1692, 32
  %1694 = ashr exact i64 %1693, 32
  %1695 = mul nsw i64 %1694, %const_reg_qword5, !spirv.Decorations !610
  %1696 = ashr i64 %1692, 32
  %1697 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1695, i32 0
  %1698 = getelementptr i16, i16 addrspace(4)* %1697, i64 %1696
  %1699 = addrspacecast i16 addrspace(4)* %1698 to i16 addrspace(1)*
  %1700 = load i16, i16 addrspace(1)* %1699, align 2
  %1701 = zext i16 %1679 to i32
  %1702 = shl nuw i32 %1701, 16, !spirv.Decorations !628
  %1703 = bitcast i32 %1702 to float
  %1704 = zext i16 %1700 to i32
  %1705 = shl nuw i32 %1704, 16, !spirv.Decorations !628
  %1706 = bitcast i32 %1705 to float
  %1707 = fmul reassoc nsz arcp contract float %1703, %1706, !spirv.Decorations !612
  %1708 = fadd reassoc nsz arcp contract float %1707, %.sroa.158.1, !spirv.Decorations !612
  br label %._crit_edge.2.7

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7, %1691
  %.sroa.158.2 = phi float [ %1708, %1691 ], [ %.sroa.158.1, %._crit_edge.1.7 ]
  %1709 = and i1 %238, %1552
  br i1 %1709, label %1710, label %.preheader.7

1710:                                             ; preds = %._crit_edge.2.7
  %1711 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1711)
  %1712 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1712)
  %1713 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %1713, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1714

1714:                                             ; preds = %1714, %1710
  %1715 = phi i32 [ 0, %1710 ], [ %1720, %1714 ]
  %1716 = zext i32 %1715 to i64
  %1717 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1716
  %1718 = load i32, i32* %1717, align 4, !noalias !635
  %1719 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1716
  store i32 %1718, i32* %1719, align 4, !alias.scope !635
  %1720 = add nuw nsw i32 %1715, 1, !spirv.Decorations !620
  %1721 = icmp eq i32 %1715, 0
  br i1 %1721, label %1714, label %1722, !llvm.loop !638

1722:                                             ; preds = %1714
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1712)
  %1723 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1711)
  %1724 = shl i64 %1723, 32
  %1725 = ashr exact i64 %1724, 32
  %1726 = mul nsw i64 %1725, %const_reg_qword3, !spirv.Decorations !610
  %1727 = ashr i64 %1723, 32
  %1728 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1726, i32 0
  %1729 = getelementptr i16, i16 addrspace(4)* %1728, i64 %1727
  %1730 = addrspacecast i16 addrspace(4)* %1729 to i16 addrspace(1)*
  %1731 = load i16, i16 addrspace(1)* %1730, align 2
  %1732 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1732)
  %1733 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1733)
  %1734 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1734, align 4, !noalias !639
  store i32 %1551, i32* %57, align 4, !noalias !639
  br label %1735

1735:                                             ; preds = %1735, %1722
  %1736 = phi i32 [ 0, %1722 ], [ %1741, %1735 ]
  %1737 = zext i32 %1736 to i64
  %1738 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1737
  %1739 = load i32, i32* %1738, align 4, !noalias !639
  %1740 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1737
  store i32 %1739, i32* %1740, align 4, !alias.scope !639
  %1741 = add nuw nsw i32 %1736, 1, !spirv.Decorations !620
  %1742 = icmp eq i32 %1736, 0
  br i1 %1742, label %1735, label %1743, !llvm.loop !642

1743:                                             ; preds = %1735
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1733)
  %1744 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1732)
  %1745 = shl i64 %1744, 32
  %1746 = ashr exact i64 %1745, 32
  %1747 = mul nsw i64 %1746, %const_reg_qword5, !spirv.Decorations !610
  %1748 = ashr i64 %1744, 32
  %1749 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1747, i32 0
  %1750 = getelementptr i16, i16 addrspace(4)* %1749, i64 %1748
  %1751 = addrspacecast i16 addrspace(4)* %1750 to i16 addrspace(1)*
  %1752 = load i16, i16 addrspace(1)* %1751, align 2
  %1753 = zext i16 %1731 to i32
  %1754 = shl nuw i32 %1753, 16, !spirv.Decorations !628
  %1755 = bitcast i32 %1754 to float
  %1756 = zext i16 %1752 to i32
  %1757 = shl nuw i32 %1756, 16, !spirv.Decorations !628
  %1758 = bitcast i32 %1757 to float
  %1759 = fmul reassoc nsz arcp contract float %1755, %1758, !spirv.Decorations !612
  %1760 = fadd reassoc nsz arcp contract float %1759, %.sroa.222.1, !spirv.Decorations !612
  br label %.preheader.7

.preheader.7:                                     ; preds = %._crit_edge.2.7, %1743
  %.sroa.222.2 = phi float [ %1760, %1743 ], [ %.sroa.222.1, %._crit_edge.2.7 ]
  %1761 = or i32 %41, 8
  %1762 = icmp slt i32 %1761, %const_reg_dword1
  %1763 = and i1 %76, %1762
  br i1 %1763, label %1764, label %._crit_edge.8

1764:                                             ; preds = %.preheader.7
  %1765 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1765)
  %1766 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1766)
  %1767 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1767, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1768

1768:                                             ; preds = %1768, %1764
  %1769 = phi i32 [ 0, %1764 ], [ %1774, %1768 ]
  %1770 = zext i32 %1769 to i64
  %1771 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1770
  %1772 = load i32, i32* %1771, align 4, !noalias !635
  %1773 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1770
  store i32 %1772, i32* %1773, align 4, !alias.scope !635
  %1774 = add nuw nsw i32 %1769, 1, !spirv.Decorations !620
  %1775 = icmp eq i32 %1769, 0
  br i1 %1775, label %1768, label %1776, !llvm.loop !638

1776:                                             ; preds = %1768
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1766)
  %1777 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1765)
  %1778 = shl i64 %1777, 32
  %1779 = ashr exact i64 %1778, 32
  %1780 = mul nsw i64 %1779, %const_reg_qword3, !spirv.Decorations !610
  %1781 = ashr i64 %1777, 32
  %1782 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1780, i32 0
  %1783 = getelementptr i16, i16 addrspace(4)* %1782, i64 %1781
  %1784 = addrspacecast i16 addrspace(4)* %1783 to i16 addrspace(1)*
  %1785 = load i16, i16 addrspace(1)* %1784, align 2
  %1786 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1786)
  %1787 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1787)
  %1788 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1788, align 4, !noalias !639
  store i32 %1761, i32* %57, align 4, !noalias !639
  br label %1789

1789:                                             ; preds = %1789, %1776
  %1790 = phi i32 [ 0, %1776 ], [ %1795, %1789 ]
  %1791 = zext i32 %1790 to i64
  %1792 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1791
  %1793 = load i32, i32* %1792, align 4, !noalias !639
  %1794 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1791
  store i32 %1793, i32* %1794, align 4, !alias.scope !639
  %1795 = add nuw nsw i32 %1790, 1, !spirv.Decorations !620
  %1796 = icmp eq i32 %1790, 0
  br i1 %1796, label %1789, label %1797, !llvm.loop !642

1797:                                             ; preds = %1789
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1787)
  %1798 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1786)
  %1799 = shl i64 %1798, 32
  %1800 = ashr exact i64 %1799, 32
  %1801 = mul nsw i64 %1800, %const_reg_qword5, !spirv.Decorations !610
  %1802 = ashr i64 %1798, 32
  %1803 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1801, i32 0
  %1804 = getelementptr i16, i16 addrspace(4)* %1803, i64 %1802
  %1805 = addrspacecast i16 addrspace(4)* %1804 to i16 addrspace(1)*
  %1806 = load i16, i16 addrspace(1)* %1805, align 2
  %1807 = zext i16 %1785 to i32
  %1808 = shl nuw i32 %1807, 16, !spirv.Decorations !628
  %1809 = bitcast i32 %1808 to float
  %1810 = zext i16 %1806 to i32
  %1811 = shl nuw i32 %1810, 16, !spirv.Decorations !628
  %1812 = bitcast i32 %1811 to float
  %1813 = fmul reassoc nsz arcp contract float %1809, %1812, !spirv.Decorations !612
  %1814 = fadd reassoc nsz arcp contract float %1813, %.sroa.34.1, !spirv.Decorations !612
  br label %._crit_edge.8

._crit_edge.8:                                    ; preds = %.preheader.7, %1797
  %.sroa.34.2 = phi float [ %1814, %1797 ], [ %.sroa.34.1, %.preheader.7 ]
  %1815 = and i1 %130, %1762
  br i1 %1815, label %1816, label %._crit_edge.1.8

1816:                                             ; preds = %._crit_edge.8
  %1817 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1817)
  %1818 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1818)
  %1819 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %1819, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1820

1820:                                             ; preds = %1820, %1816
  %1821 = phi i32 [ 0, %1816 ], [ %1826, %1820 ]
  %1822 = zext i32 %1821 to i64
  %1823 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1822
  %1824 = load i32, i32* %1823, align 4, !noalias !635
  %1825 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1822
  store i32 %1824, i32* %1825, align 4, !alias.scope !635
  %1826 = add nuw nsw i32 %1821, 1, !spirv.Decorations !620
  %1827 = icmp eq i32 %1821, 0
  br i1 %1827, label %1820, label %1828, !llvm.loop !638

1828:                                             ; preds = %1820
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1818)
  %1829 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1817)
  %1830 = shl i64 %1829, 32
  %1831 = ashr exact i64 %1830, 32
  %1832 = mul nsw i64 %1831, %const_reg_qword3, !spirv.Decorations !610
  %1833 = ashr i64 %1829, 32
  %1834 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1832, i32 0
  %1835 = getelementptr i16, i16 addrspace(4)* %1834, i64 %1833
  %1836 = addrspacecast i16 addrspace(4)* %1835 to i16 addrspace(1)*
  %1837 = load i16, i16 addrspace(1)* %1836, align 2
  %1838 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1838)
  %1839 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1839)
  %1840 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1840, align 4, !noalias !639
  store i32 %1761, i32* %57, align 4, !noalias !639
  br label %1841

1841:                                             ; preds = %1841, %1828
  %1842 = phi i32 [ 0, %1828 ], [ %1847, %1841 ]
  %1843 = zext i32 %1842 to i64
  %1844 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1843
  %1845 = load i32, i32* %1844, align 4, !noalias !639
  %1846 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1843
  store i32 %1845, i32* %1846, align 4, !alias.scope !639
  %1847 = add nuw nsw i32 %1842, 1, !spirv.Decorations !620
  %1848 = icmp eq i32 %1842, 0
  br i1 %1848, label %1841, label %1849, !llvm.loop !642

1849:                                             ; preds = %1841
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1839)
  %1850 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1838)
  %1851 = shl i64 %1850, 32
  %1852 = ashr exact i64 %1851, 32
  %1853 = mul nsw i64 %1852, %const_reg_qword5, !spirv.Decorations !610
  %1854 = ashr i64 %1850, 32
  %1855 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1853, i32 0
  %1856 = getelementptr i16, i16 addrspace(4)* %1855, i64 %1854
  %1857 = addrspacecast i16 addrspace(4)* %1856 to i16 addrspace(1)*
  %1858 = load i16, i16 addrspace(1)* %1857, align 2
  %1859 = zext i16 %1837 to i32
  %1860 = shl nuw i32 %1859, 16, !spirv.Decorations !628
  %1861 = bitcast i32 %1860 to float
  %1862 = zext i16 %1858 to i32
  %1863 = shl nuw i32 %1862, 16, !spirv.Decorations !628
  %1864 = bitcast i32 %1863 to float
  %1865 = fmul reassoc nsz arcp contract float %1861, %1864, !spirv.Decorations !612
  %1866 = fadd reassoc nsz arcp contract float %1865, %.sroa.98.1, !spirv.Decorations !612
  br label %._crit_edge.1.8

._crit_edge.1.8:                                  ; preds = %._crit_edge.8, %1849
  %.sroa.98.2 = phi float [ %1866, %1849 ], [ %.sroa.98.1, %._crit_edge.8 ]
  %1867 = and i1 %184, %1762
  br i1 %1867, label %1868, label %._crit_edge.2.8

1868:                                             ; preds = %._crit_edge.1.8
  %1869 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1869)
  %1870 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1870)
  %1871 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %1871, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1872

1872:                                             ; preds = %1872, %1868
  %1873 = phi i32 [ 0, %1868 ], [ %1878, %1872 ]
  %1874 = zext i32 %1873 to i64
  %1875 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1874
  %1876 = load i32, i32* %1875, align 4, !noalias !635
  %1877 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1874
  store i32 %1876, i32* %1877, align 4, !alias.scope !635
  %1878 = add nuw nsw i32 %1873, 1, !spirv.Decorations !620
  %1879 = icmp eq i32 %1873, 0
  br i1 %1879, label %1872, label %1880, !llvm.loop !638

1880:                                             ; preds = %1872
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1870)
  %1881 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1869)
  %1882 = shl i64 %1881, 32
  %1883 = ashr exact i64 %1882, 32
  %1884 = mul nsw i64 %1883, %const_reg_qword3, !spirv.Decorations !610
  %1885 = ashr i64 %1881, 32
  %1886 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1884, i32 0
  %1887 = getelementptr i16, i16 addrspace(4)* %1886, i64 %1885
  %1888 = addrspacecast i16 addrspace(4)* %1887 to i16 addrspace(1)*
  %1889 = load i16, i16 addrspace(1)* %1888, align 2
  %1890 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1890)
  %1891 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1891)
  %1892 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1892, align 4, !noalias !639
  store i32 %1761, i32* %57, align 4, !noalias !639
  br label %1893

1893:                                             ; preds = %1893, %1880
  %1894 = phi i32 [ 0, %1880 ], [ %1899, %1893 ]
  %1895 = zext i32 %1894 to i64
  %1896 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1895
  %1897 = load i32, i32* %1896, align 4, !noalias !639
  %1898 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1895
  store i32 %1897, i32* %1898, align 4, !alias.scope !639
  %1899 = add nuw nsw i32 %1894, 1, !spirv.Decorations !620
  %1900 = icmp eq i32 %1894, 0
  br i1 %1900, label %1893, label %1901, !llvm.loop !642

1901:                                             ; preds = %1893
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1891)
  %1902 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1890)
  %1903 = shl i64 %1902, 32
  %1904 = ashr exact i64 %1903, 32
  %1905 = mul nsw i64 %1904, %const_reg_qword5, !spirv.Decorations !610
  %1906 = ashr i64 %1902, 32
  %1907 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1905, i32 0
  %1908 = getelementptr i16, i16 addrspace(4)* %1907, i64 %1906
  %1909 = addrspacecast i16 addrspace(4)* %1908 to i16 addrspace(1)*
  %1910 = load i16, i16 addrspace(1)* %1909, align 2
  %1911 = zext i16 %1889 to i32
  %1912 = shl nuw i32 %1911, 16, !spirv.Decorations !628
  %1913 = bitcast i32 %1912 to float
  %1914 = zext i16 %1910 to i32
  %1915 = shl nuw i32 %1914, 16, !spirv.Decorations !628
  %1916 = bitcast i32 %1915 to float
  %1917 = fmul reassoc nsz arcp contract float %1913, %1916, !spirv.Decorations !612
  %1918 = fadd reassoc nsz arcp contract float %1917, %.sroa.162.1, !spirv.Decorations !612
  br label %._crit_edge.2.8

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8, %1901
  %.sroa.162.2 = phi float [ %1918, %1901 ], [ %.sroa.162.1, %._crit_edge.1.8 ]
  %1919 = and i1 %238, %1762
  br i1 %1919, label %1920, label %.preheader.8

1920:                                             ; preds = %._crit_edge.2.8
  %1921 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1921)
  %1922 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1922)
  %1923 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %1923, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1924

1924:                                             ; preds = %1924, %1920
  %1925 = phi i32 [ 0, %1920 ], [ %1930, %1924 ]
  %1926 = zext i32 %1925 to i64
  %1927 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1926
  %1928 = load i32, i32* %1927, align 4, !noalias !635
  %1929 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1926
  store i32 %1928, i32* %1929, align 4, !alias.scope !635
  %1930 = add nuw nsw i32 %1925, 1, !spirv.Decorations !620
  %1931 = icmp eq i32 %1925, 0
  br i1 %1931, label %1924, label %1932, !llvm.loop !638

1932:                                             ; preds = %1924
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1922)
  %1933 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1921)
  %1934 = shl i64 %1933, 32
  %1935 = ashr exact i64 %1934, 32
  %1936 = mul nsw i64 %1935, %const_reg_qword3, !spirv.Decorations !610
  %1937 = ashr i64 %1933, 32
  %1938 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1936, i32 0
  %1939 = getelementptr i16, i16 addrspace(4)* %1938, i64 %1937
  %1940 = addrspacecast i16 addrspace(4)* %1939 to i16 addrspace(1)*
  %1941 = load i16, i16 addrspace(1)* %1940, align 2
  %1942 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1942)
  %1943 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1943)
  %1944 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1944, align 4, !noalias !639
  store i32 %1761, i32* %57, align 4, !noalias !639
  br label %1945

1945:                                             ; preds = %1945, %1932
  %1946 = phi i32 [ 0, %1932 ], [ %1951, %1945 ]
  %1947 = zext i32 %1946 to i64
  %1948 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %1947
  %1949 = load i32, i32* %1948, align 4, !noalias !639
  %1950 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %1947
  store i32 %1949, i32* %1950, align 4, !alias.scope !639
  %1951 = add nuw nsw i32 %1946, 1, !spirv.Decorations !620
  %1952 = icmp eq i32 %1946, 0
  br i1 %1952, label %1945, label %1953, !llvm.loop !642

1953:                                             ; preds = %1945
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1943)
  %1954 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1942)
  %1955 = shl i64 %1954, 32
  %1956 = ashr exact i64 %1955, 32
  %1957 = mul nsw i64 %1956, %const_reg_qword5, !spirv.Decorations !610
  %1958 = ashr i64 %1954, 32
  %1959 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %1957, i32 0
  %1960 = getelementptr i16, i16 addrspace(4)* %1959, i64 %1958
  %1961 = addrspacecast i16 addrspace(4)* %1960 to i16 addrspace(1)*
  %1962 = load i16, i16 addrspace(1)* %1961, align 2
  %1963 = zext i16 %1941 to i32
  %1964 = shl nuw i32 %1963, 16, !spirv.Decorations !628
  %1965 = bitcast i32 %1964 to float
  %1966 = zext i16 %1962 to i32
  %1967 = shl nuw i32 %1966, 16, !spirv.Decorations !628
  %1968 = bitcast i32 %1967 to float
  %1969 = fmul reassoc nsz arcp contract float %1965, %1968, !spirv.Decorations !612
  %1970 = fadd reassoc nsz arcp contract float %1969, %.sroa.226.1, !spirv.Decorations !612
  br label %.preheader.8

.preheader.8:                                     ; preds = %._crit_edge.2.8, %1953
  %.sroa.226.2 = phi float [ %1970, %1953 ], [ %.sroa.226.1, %._crit_edge.2.8 ]
  %1971 = or i32 %41, 9
  %1972 = icmp slt i32 %1971, %const_reg_dword1
  %1973 = and i1 %76, %1972
  br i1 %1973, label %1974, label %._crit_edge.9

1974:                                             ; preds = %.preheader.8
  %1975 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1975)
  %1976 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1976)
  %1977 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %1977, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %1978

1978:                                             ; preds = %1978, %1974
  %1979 = phi i32 [ 0, %1974 ], [ %1984, %1978 ]
  %1980 = zext i32 %1979 to i64
  %1981 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %1980
  %1982 = load i32, i32* %1981, align 4, !noalias !635
  %1983 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %1980
  store i32 %1982, i32* %1983, align 4, !alias.scope !635
  %1984 = add nuw nsw i32 %1979, 1, !spirv.Decorations !620
  %1985 = icmp eq i32 %1979, 0
  br i1 %1985, label %1978, label %1986, !llvm.loop !638

1986:                                             ; preds = %1978
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1976)
  %1987 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1975)
  %1988 = shl i64 %1987, 32
  %1989 = ashr exact i64 %1988, 32
  %1990 = mul nsw i64 %1989, %const_reg_qword3, !spirv.Decorations !610
  %1991 = ashr i64 %1987, 32
  %1992 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %1990, i32 0
  %1993 = getelementptr i16, i16 addrspace(4)* %1992, i64 %1991
  %1994 = addrspacecast i16 addrspace(4)* %1993 to i16 addrspace(1)*
  %1995 = load i16, i16 addrspace(1)* %1994, align 2
  %1996 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1996)
  %1997 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1997)
  %1998 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %1998, align 4, !noalias !639
  store i32 %1971, i32* %57, align 4, !noalias !639
  br label %1999

1999:                                             ; preds = %1999, %1986
  %2000 = phi i32 [ 0, %1986 ], [ %2005, %1999 ]
  %2001 = zext i32 %2000 to i64
  %2002 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2001
  %2003 = load i32, i32* %2002, align 4, !noalias !639
  %2004 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2001
  store i32 %2003, i32* %2004, align 4, !alias.scope !639
  %2005 = add nuw nsw i32 %2000, 1, !spirv.Decorations !620
  %2006 = icmp eq i32 %2000, 0
  br i1 %2006, label %1999, label %2007, !llvm.loop !642

2007:                                             ; preds = %1999
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1997)
  %2008 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1996)
  %2009 = shl i64 %2008, 32
  %2010 = ashr exact i64 %2009, 32
  %2011 = mul nsw i64 %2010, %const_reg_qword5, !spirv.Decorations !610
  %2012 = ashr i64 %2008, 32
  %2013 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2011, i32 0
  %2014 = getelementptr i16, i16 addrspace(4)* %2013, i64 %2012
  %2015 = addrspacecast i16 addrspace(4)* %2014 to i16 addrspace(1)*
  %2016 = load i16, i16 addrspace(1)* %2015, align 2
  %2017 = zext i16 %1995 to i32
  %2018 = shl nuw i32 %2017, 16, !spirv.Decorations !628
  %2019 = bitcast i32 %2018 to float
  %2020 = zext i16 %2016 to i32
  %2021 = shl nuw i32 %2020, 16, !spirv.Decorations !628
  %2022 = bitcast i32 %2021 to float
  %2023 = fmul reassoc nsz arcp contract float %2019, %2022, !spirv.Decorations !612
  %2024 = fadd reassoc nsz arcp contract float %2023, %.sroa.38.1, !spirv.Decorations !612
  br label %._crit_edge.9

._crit_edge.9:                                    ; preds = %.preheader.8, %2007
  %.sroa.38.2 = phi float [ %2024, %2007 ], [ %.sroa.38.1, %.preheader.8 ]
  %2025 = and i1 %130, %1972
  br i1 %2025, label %2026, label %._crit_edge.1.9

2026:                                             ; preds = %._crit_edge.9
  %2027 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2027)
  %2028 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2028)
  %2029 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %2029, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2030

2030:                                             ; preds = %2030, %2026
  %2031 = phi i32 [ 0, %2026 ], [ %2036, %2030 ]
  %2032 = zext i32 %2031 to i64
  %2033 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2032
  %2034 = load i32, i32* %2033, align 4, !noalias !635
  %2035 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2032
  store i32 %2034, i32* %2035, align 4, !alias.scope !635
  %2036 = add nuw nsw i32 %2031, 1, !spirv.Decorations !620
  %2037 = icmp eq i32 %2031, 0
  br i1 %2037, label %2030, label %2038, !llvm.loop !638

2038:                                             ; preds = %2030
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2028)
  %2039 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2027)
  %2040 = shl i64 %2039, 32
  %2041 = ashr exact i64 %2040, 32
  %2042 = mul nsw i64 %2041, %const_reg_qword3, !spirv.Decorations !610
  %2043 = ashr i64 %2039, 32
  %2044 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2042, i32 0
  %2045 = getelementptr i16, i16 addrspace(4)* %2044, i64 %2043
  %2046 = addrspacecast i16 addrspace(4)* %2045 to i16 addrspace(1)*
  %2047 = load i16, i16 addrspace(1)* %2046, align 2
  %2048 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2048)
  %2049 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2049)
  %2050 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2050, align 4, !noalias !639
  store i32 %1971, i32* %57, align 4, !noalias !639
  br label %2051

2051:                                             ; preds = %2051, %2038
  %2052 = phi i32 [ 0, %2038 ], [ %2057, %2051 ]
  %2053 = zext i32 %2052 to i64
  %2054 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2053
  %2055 = load i32, i32* %2054, align 4, !noalias !639
  %2056 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2053
  store i32 %2055, i32* %2056, align 4, !alias.scope !639
  %2057 = add nuw nsw i32 %2052, 1, !spirv.Decorations !620
  %2058 = icmp eq i32 %2052, 0
  br i1 %2058, label %2051, label %2059, !llvm.loop !642

2059:                                             ; preds = %2051
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2049)
  %2060 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2048)
  %2061 = shl i64 %2060, 32
  %2062 = ashr exact i64 %2061, 32
  %2063 = mul nsw i64 %2062, %const_reg_qword5, !spirv.Decorations !610
  %2064 = ashr i64 %2060, 32
  %2065 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2063, i32 0
  %2066 = getelementptr i16, i16 addrspace(4)* %2065, i64 %2064
  %2067 = addrspacecast i16 addrspace(4)* %2066 to i16 addrspace(1)*
  %2068 = load i16, i16 addrspace(1)* %2067, align 2
  %2069 = zext i16 %2047 to i32
  %2070 = shl nuw i32 %2069, 16, !spirv.Decorations !628
  %2071 = bitcast i32 %2070 to float
  %2072 = zext i16 %2068 to i32
  %2073 = shl nuw i32 %2072, 16, !spirv.Decorations !628
  %2074 = bitcast i32 %2073 to float
  %2075 = fmul reassoc nsz arcp contract float %2071, %2074, !spirv.Decorations !612
  %2076 = fadd reassoc nsz arcp contract float %2075, %.sroa.102.1, !spirv.Decorations !612
  br label %._crit_edge.1.9

._crit_edge.1.9:                                  ; preds = %._crit_edge.9, %2059
  %.sroa.102.2 = phi float [ %2076, %2059 ], [ %.sroa.102.1, %._crit_edge.9 ]
  %2077 = and i1 %184, %1972
  br i1 %2077, label %2078, label %._crit_edge.2.9

2078:                                             ; preds = %._crit_edge.1.9
  %2079 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2079)
  %2080 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2080)
  %2081 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %2081, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2082

2082:                                             ; preds = %2082, %2078
  %2083 = phi i32 [ 0, %2078 ], [ %2088, %2082 ]
  %2084 = zext i32 %2083 to i64
  %2085 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2084
  %2086 = load i32, i32* %2085, align 4, !noalias !635
  %2087 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2084
  store i32 %2086, i32* %2087, align 4, !alias.scope !635
  %2088 = add nuw nsw i32 %2083, 1, !spirv.Decorations !620
  %2089 = icmp eq i32 %2083, 0
  br i1 %2089, label %2082, label %2090, !llvm.loop !638

2090:                                             ; preds = %2082
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2080)
  %2091 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2079)
  %2092 = shl i64 %2091, 32
  %2093 = ashr exact i64 %2092, 32
  %2094 = mul nsw i64 %2093, %const_reg_qword3, !spirv.Decorations !610
  %2095 = ashr i64 %2091, 32
  %2096 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2094, i32 0
  %2097 = getelementptr i16, i16 addrspace(4)* %2096, i64 %2095
  %2098 = addrspacecast i16 addrspace(4)* %2097 to i16 addrspace(1)*
  %2099 = load i16, i16 addrspace(1)* %2098, align 2
  %2100 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2100)
  %2101 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2101)
  %2102 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2102, align 4, !noalias !639
  store i32 %1971, i32* %57, align 4, !noalias !639
  br label %2103

2103:                                             ; preds = %2103, %2090
  %2104 = phi i32 [ 0, %2090 ], [ %2109, %2103 ]
  %2105 = zext i32 %2104 to i64
  %2106 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2105
  %2107 = load i32, i32* %2106, align 4, !noalias !639
  %2108 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2105
  store i32 %2107, i32* %2108, align 4, !alias.scope !639
  %2109 = add nuw nsw i32 %2104, 1, !spirv.Decorations !620
  %2110 = icmp eq i32 %2104, 0
  br i1 %2110, label %2103, label %2111, !llvm.loop !642

2111:                                             ; preds = %2103
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2101)
  %2112 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2100)
  %2113 = shl i64 %2112, 32
  %2114 = ashr exact i64 %2113, 32
  %2115 = mul nsw i64 %2114, %const_reg_qword5, !spirv.Decorations !610
  %2116 = ashr i64 %2112, 32
  %2117 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2115, i32 0
  %2118 = getelementptr i16, i16 addrspace(4)* %2117, i64 %2116
  %2119 = addrspacecast i16 addrspace(4)* %2118 to i16 addrspace(1)*
  %2120 = load i16, i16 addrspace(1)* %2119, align 2
  %2121 = zext i16 %2099 to i32
  %2122 = shl nuw i32 %2121, 16, !spirv.Decorations !628
  %2123 = bitcast i32 %2122 to float
  %2124 = zext i16 %2120 to i32
  %2125 = shl nuw i32 %2124, 16, !spirv.Decorations !628
  %2126 = bitcast i32 %2125 to float
  %2127 = fmul reassoc nsz arcp contract float %2123, %2126, !spirv.Decorations !612
  %2128 = fadd reassoc nsz arcp contract float %2127, %.sroa.166.1, !spirv.Decorations !612
  br label %._crit_edge.2.9

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9, %2111
  %.sroa.166.2 = phi float [ %2128, %2111 ], [ %.sroa.166.1, %._crit_edge.1.9 ]
  %2129 = and i1 %238, %1972
  br i1 %2129, label %2130, label %.preheader.9

2130:                                             ; preds = %._crit_edge.2.9
  %2131 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2131)
  %2132 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2132)
  %2133 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %2133, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2134

2134:                                             ; preds = %2134, %2130
  %2135 = phi i32 [ 0, %2130 ], [ %2140, %2134 ]
  %2136 = zext i32 %2135 to i64
  %2137 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2136
  %2138 = load i32, i32* %2137, align 4, !noalias !635
  %2139 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2136
  store i32 %2138, i32* %2139, align 4, !alias.scope !635
  %2140 = add nuw nsw i32 %2135, 1, !spirv.Decorations !620
  %2141 = icmp eq i32 %2135, 0
  br i1 %2141, label %2134, label %2142, !llvm.loop !638

2142:                                             ; preds = %2134
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2132)
  %2143 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2131)
  %2144 = shl i64 %2143, 32
  %2145 = ashr exact i64 %2144, 32
  %2146 = mul nsw i64 %2145, %const_reg_qword3, !spirv.Decorations !610
  %2147 = ashr i64 %2143, 32
  %2148 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2146, i32 0
  %2149 = getelementptr i16, i16 addrspace(4)* %2148, i64 %2147
  %2150 = addrspacecast i16 addrspace(4)* %2149 to i16 addrspace(1)*
  %2151 = load i16, i16 addrspace(1)* %2150, align 2
  %2152 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2152)
  %2153 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2153)
  %2154 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2154, align 4, !noalias !639
  store i32 %1971, i32* %57, align 4, !noalias !639
  br label %2155

2155:                                             ; preds = %2155, %2142
  %2156 = phi i32 [ 0, %2142 ], [ %2161, %2155 ]
  %2157 = zext i32 %2156 to i64
  %2158 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2157
  %2159 = load i32, i32* %2158, align 4, !noalias !639
  %2160 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2157
  store i32 %2159, i32* %2160, align 4, !alias.scope !639
  %2161 = add nuw nsw i32 %2156, 1, !spirv.Decorations !620
  %2162 = icmp eq i32 %2156, 0
  br i1 %2162, label %2155, label %2163, !llvm.loop !642

2163:                                             ; preds = %2155
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2153)
  %2164 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2152)
  %2165 = shl i64 %2164, 32
  %2166 = ashr exact i64 %2165, 32
  %2167 = mul nsw i64 %2166, %const_reg_qword5, !spirv.Decorations !610
  %2168 = ashr i64 %2164, 32
  %2169 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2167, i32 0
  %2170 = getelementptr i16, i16 addrspace(4)* %2169, i64 %2168
  %2171 = addrspacecast i16 addrspace(4)* %2170 to i16 addrspace(1)*
  %2172 = load i16, i16 addrspace(1)* %2171, align 2
  %2173 = zext i16 %2151 to i32
  %2174 = shl nuw i32 %2173, 16, !spirv.Decorations !628
  %2175 = bitcast i32 %2174 to float
  %2176 = zext i16 %2172 to i32
  %2177 = shl nuw i32 %2176, 16, !spirv.Decorations !628
  %2178 = bitcast i32 %2177 to float
  %2179 = fmul reassoc nsz arcp contract float %2175, %2178, !spirv.Decorations !612
  %2180 = fadd reassoc nsz arcp contract float %2179, %.sroa.230.1, !spirv.Decorations !612
  br label %.preheader.9

.preheader.9:                                     ; preds = %._crit_edge.2.9, %2163
  %.sroa.230.2 = phi float [ %2180, %2163 ], [ %.sroa.230.1, %._crit_edge.2.9 ]
  %2181 = or i32 %41, 10
  %2182 = icmp slt i32 %2181, %const_reg_dword1
  %2183 = and i1 %76, %2182
  br i1 %2183, label %2184, label %._crit_edge.10

2184:                                             ; preds = %.preheader.9
  %2185 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2185)
  %2186 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2186)
  %2187 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2187, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2188

2188:                                             ; preds = %2188, %2184
  %2189 = phi i32 [ 0, %2184 ], [ %2194, %2188 ]
  %2190 = zext i32 %2189 to i64
  %2191 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2190
  %2192 = load i32, i32* %2191, align 4, !noalias !635
  %2193 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2190
  store i32 %2192, i32* %2193, align 4, !alias.scope !635
  %2194 = add nuw nsw i32 %2189, 1, !spirv.Decorations !620
  %2195 = icmp eq i32 %2189, 0
  br i1 %2195, label %2188, label %2196, !llvm.loop !638

2196:                                             ; preds = %2188
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2186)
  %2197 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2185)
  %2198 = shl i64 %2197, 32
  %2199 = ashr exact i64 %2198, 32
  %2200 = mul nsw i64 %2199, %const_reg_qword3, !spirv.Decorations !610
  %2201 = ashr i64 %2197, 32
  %2202 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2200, i32 0
  %2203 = getelementptr i16, i16 addrspace(4)* %2202, i64 %2201
  %2204 = addrspacecast i16 addrspace(4)* %2203 to i16 addrspace(1)*
  %2205 = load i16, i16 addrspace(1)* %2204, align 2
  %2206 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2206)
  %2207 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2207)
  %2208 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2208, align 4, !noalias !639
  store i32 %2181, i32* %57, align 4, !noalias !639
  br label %2209

2209:                                             ; preds = %2209, %2196
  %2210 = phi i32 [ 0, %2196 ], [ %2215, %2209 ]
  %2211 = zext i32 %2210 to i64
  %2212 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2211
  %2213 = load i32, i32* %2212, align 4, !noalias !639
  %2214 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2211
  store i32 %2213, i32* %2214, align 4, !alias.scope !639
  %2215 = add nuw nsw i32 %2210, 1, !spirv.Decorations !620
  %2216 = icmp eq i32 %2210, 0
  br i1 %2216, label %2209, label %2217, !llvm.loop !642

2217:                                             ; preds = %2209
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2207)
  %2218 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2206)
  %2219 = shl i64 %2218, 32
  %2220 = ashr exact i64 %2219, 32
  %2221 = mul nsw i64 %2220, %const_reg_qword5, !spirv.Decorations !610
  %2222 = ashr i64 %2218, 32
  %2223 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2221, i32 0
  %2224 = getelementptr i16, i16 addrspace(4)* %2223, i64 %2222
  %2225 = addrspacecast i16 addrspace(4)* %2224 to i16 addrspace(1)*
  %2226 = load i16, i16 addrspace(1)* %2225, align 2
  %2227 = zext i16 %2205 to i32
  %2228 = shl nuw i32 %2227, 16, !spirv.Decorations !628
  %2229 = bitcast i32 %2228 to float
  %2230 = zext i16 %2226 to i32
  %2231 = shl nuw i32 %2230, 16, !spirv.Decorations !628
  %2232 = bitcast i32 %2231 to float
  %2233 = fmul reassoc nsz arcp contract float %2229, %2232, !spirv.Decorations !612
  %2234 = fadd reassoc nsz arcp contract float %2233, %.sroa.42.1, !spirv.Decorations !612
  br label %._crit_edge.10

._crit_edge.10:                                   ; preds = %.preheader.9, %2217
  %.sroa.42.2 = phi float [ %2234, %2217 ], [ %.sroa.42.1, %.preheader.9 ]
  %2235 = and i1 %130, %2182
  br i1 %2235, label %2236, label %._crit_edge.1.10

2236:                                             ; preds = %._crit_edge.10
  %2237 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2237)
  %2238 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2238)
  %2239 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %2239, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2240

2240:                                             ; preds = %2240, %2236
  %2241 = phi i32 [ 0, %2236 ], [ %2246, %2240 ]
  %2242 = zext i32 %2241 to i64
  %2243 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2242
  %2244 = load i32, i32* %2243, align 4, !noalias !635
  %2245 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2242
  store i32 %2244, i32* %2245, align 4, !alias.scope !635
  %2246 = add nuw nsw i32 %2241, 1, !spirv.Decorations !620
  %2247 = icmp eq i32 %2241, 0
  br i1 %2247, label %2240, label %2248, !llvm.loop !638

2248:                                             ; preds = %2240
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2238)
  %2249 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2237)
  %2250 = shl i64 %2249, 32
  %2251 = ashr exact i64 %2250, 32
  %2252 = mul nsw i64 %2251, %const_reg_qword3, !spirv.Decorations !610
  %2253 = ashr i64 %2249, 32
  %2254 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2252, i32 0
  %2255 = getelementptr i16, i16 addrspace(4)* %2254, i64 %2253
  %2256 = addrspacecast i16 addrspace(4)* %2255 to i16 addrspace(1)*
  %2257 = load i16, i16 addrspace(1)* %2256, align 2
  %2258 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2258)
  %2259 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2259)
  %2260 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2260, align 4, !noalias !639
  store i32 %2181, i32* %57, align 4, !noalias !639
  br label %2261

2261:                                             ; preds = %2261, %2248
  %2262 = phi i32 [ 0, %2248 ], [ %2267, %2261 ]
  %2263 = zext i32 %2262 to i64
  %2264 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2263
  %2265 = load i32, i32* %2264, align 4, !noalias !639
  %2266 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2263
  store i32 %2265, i32* %2266, align 4, !alias.scope !639
  %2267 = add nuw nsw i32 %2262, 1, !spirv.Decorations !620
  %2268 = icmp eq i32 %2262, 0
  br i1 %2268, label %2261, label %2269, !llvm.loop !642

2269:                                             ; preds = %2261
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2259)
  %2270 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2258)
  %2271 = shl i64 %2270, 32
  %2272 = ashr exact i64 %2271, 32
  %2273 = mul nsw i64 %2272, %const_reg_qword5, !spirv.Decorations !610
  %2274 = ashr i64 %2270, 32
  %2275 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2273, i32 0
  %2276 = getelementptr i16, i16 addrspace(4)* %2275, i64 %2274
  %2277 = addrspacecast i16 addrspace(4)* %2276 to i16 addrspace(1)*
  %2278 = load i16, i16 addrspace(1)* %2277, align 2
  %2279 = zext i16 %2257 to i32
  %2280 = shl nuw i32 %2279, 16, !spirv.Decorations !628
  %2281 = bitcast i32 %2280 to float
  %2282 = zext i16 %2278 to i32
  %2283 = shl nuw i32 %2282, 16, !spirv.Decorations !628
  %2284 = bitcast i32 %2283 to float
  %2285 = fmul reassoc nsz arcp contract float %2281, %2284, !spirv.Decorations !612
  %2286 = fadd reassoc nsz arcp contract float %2285, %.sroa.106.1, !spirv.Decorations !612
  br label %._crit_edge.1.10

._crit_edge.1.10:                                 ; preds = %._crit_edge.10, %2269
  %.sroa.106.2 = phi float [ %2286, %2269 ], [ %.sroa.106.1, %._crit_edge.10 ]
  %2287 = and i1 %184, %2182
  br i1 %2287, label %2288, label %._crit_edge.2.10

2288:                                             ; preds = %._crit_edge.1.10
  %2289 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2289)
  %2290 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2290)
  %2291 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %2291, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2292

2292:                                             ; preds = %2292, %2288
  %2293 = phi i32 [ 0, %2288 ], [ %2298, %2292 ]
  %2294 = zext i32 %2293 to i64
  %2295 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2294
  %2296 = load i32, i32* %2295, align 4, !noalias !635
  %2297 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2294
  store i32 %2296, i32* %2297, align 4, !alias.scope !635
  %2298 = add nuw nsw i32 %2293, 1, !spirv.Decorations !620
  %2299 = icmp eq i32 %2293, 0
  br i1 %2299, label %2292, label %2300, !llvm.loop !638

2300:                                             ; preds = %2292
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2290)
  %2301 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2289)
  %2302 = shl i64 %2301, 32
  %2303 = ashr exact i64 %2302, 32
  %2304 = mul nsw i64 %2303, %const_reg_qword3, !spirv.Decorations !610
  %2305 = ashr i64 %2301, 32
  %2306 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2304, i32 0
  %2307 = getelementptr i16, i16 addrspace(4)* %2306, i64 %2305
  %2308 = addrspacecast i16 addrspace(4)* %2307 to i16 addrspace(1)*
  %2309 = load i16, i16 addrspace(1)* %2308, align 2
  %2310 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2310)
  %2311 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2311)
  %2312 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2312, align 4, !noalias !639
  store i32 %2181, i32* %57, align 4, !noalias !639
  br label %2313

2313:                                             ; preds = %2313, %2300
  %2314 = phi i32 [ 0, %2300 ], [ %2319, %2313 ]
  %2315 = zext i32 %2314 to i64
  %2316 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2315
  %2317 = load i32, i32* %2316, align 4, !noalias !639
  %2318 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2315
  store i32 %2317, i32* %2318, align 4, !alias.scope !639
  %2319 = add nuw nsw i32 %2314, 1, !spirv.Decorations !620
  %2320 = icmp eq i32 %2314, 0
  br i1 %2320, label %2313, label %2321, !llvm.loop !642

2321:                                             ; preds = %2313
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2311)
  %2322 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2310)
  %2323 = shl i64 %2322, 32
  %2324 = ashr exact i64 %2323, 32
  %2325 = mul nsw i64 %2324, %const_reg_qword5, !spirv.Decorations !610
  %2326 = ashr i64 %2322, 32
  %2327 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2325, i32 0
  %2328 = getelementptr i16, i16 addrspace(4)* %2327, i64 %2326
  %2329 = addrspacecast i16 addrspace(4)* %2328 to i16 addrspace(1)*
  %2330 = load i16, i16 addrspace(1)* %2329, align 2
  %2331 = zext i16 %2309 to i32
  %2332 = shl nuw i32 %2331, 16, !spirv.Decorations !628
  %2333 = bitcast i32 %2332 to float
  %2334 = zext i16 %2330 to i32
  %2335 = shl nuw i32 %2334, 16, !spirv.Decorations !628
  %2336 = bitcast i32 %2335 to float
  %2337 = fmul reassoc nsz arcp contract float %2333, %2336, !spirv.Decorations !612
  %2338 = fadd reassoc nsz arcp contract float %2337, %.sroa.170.1, !spirv.Decorations !612
  br label %._crit_edge.2.10

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10, %2321
  %.sroa.170.2 = phi float [ %2338, %2321 ], [ %.sroa.170.1, %._crit_edge.1.10 ]
  %2339 = and i1 %238, %2182
  br i1 %2339, label %2340, label %.preheader.10

2340:                                             ; preds = %._crit_edge.2.10
  %2341 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2341)
  %2342 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2342)
  %2343 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %2343, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2344

2344:                                             ; preds = %2344, %2340
  %2345 = phi i32 [ 0, %2340 ], [ %2350, %2344 ]
  %2346 = zext i32 %2345 to i64
  %2347 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2346
  %2348 = load i32, i32* %2347, align 4, !noalias !635
  %2349 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2346
  store i32 %2348, i32* %2349, align 4, !alias.scope !635
  %2350 = add nuw nsw i32 %2345, 1, !spirv.Decorations !620
  %2351 = icmp eq i32 %2345, 0
  br i1 %2351, label %2344, label %2352, !llvm.loop !638

2352:                                             ; preds = %2344
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2342)
  %2353 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2341)
  %2354 = shl i64 %2353, 32
  %2355 = ashr exact i64 %2354, 32
  %2356 = mul nsw i64 %2355, %const_reg_qword3, !spirv.Decorations !610
  %2357 = ashr i64 %2353, 32
  %2358 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2356, i32 0
  %2359 = getelementptr i16, i16 addrspace(4)* %2358, i64 %2357
  %2360 = addrspacecast i16 addrspace(4)* %2359 to i16 addrspace(1)*
  %2361 = load i16, i16 addrspace(1)* %2360, align 2
  %2362 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2362)
  %2363 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2363)
  %2364 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2364, align 4, !noalias !639
  store i32 %2181, i32* %57, align 4, !noalias !639
  br label %2365

2365:                                             ; preds = %2365, %2352
  %2366 = phi i32 [ 0, %2352 ], [ %2371, %2365 ]
  %2367 = zext i32 %2366 to i64
  %2368 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2367
  %2369 = load i32, i32* %2368, align 4, !noalias !639
  %2370 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2367
  store i32 %2369, i32* %2370, align 4, !alias.scope !639
  %2371 = add nuw nsw i32 %2366, 1, !spirv.Decorations !620
  %2372 = icmp eq i32 %2366, 0
  br i1 %2372, label %2365, label %2373, !llvm.loop !642

2373:                                             ; preds = %2365
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2363)
  %2374 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2362)
  %2375 = shl i64 %2374, 32
  %2376 = ashr exact i64 %2375, 32
  %2377 = mul nsw i64 %2376, %const_reg_qword5, !spirv.Decorations !610
  %2378 = ashr i64 %2374, 32
  %2379 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2377, i32 0
  %2380 = getelementptr i16, i16 addrspace(4)* %2379, i64 %2378
  %2381 = addrspacecast i16 addrspace(4)* %2380 to i16 addrspace(1)*
  %2382 = load i16, i16 addrspace(1)* %2381, align 2
  %2383 = zext i16 %2361 to i32
  %2384 = shl nuw i32 %2383, 16, !spirv.Decorations !628
  %2385 = bitcast i32 %2384 to float
  %2386 = zext i16 %2382 to i32
  %2387 = shl nuw i32 %2386, 16, !spirv.Decorations !628
  %2388 = bitcast i32 %2387 to float
  %2389 = fmul reassoc nsz arcp contract float %2385, %2388, !spirv.Decorations !612
  %2390 = fadd reassoc nsz arcp contract float %2389, %.sroa.234.1, !spirv.Decorations !612
  br label %.preheader.10

.preheader.10:                                    ; preds = %._crit_edge.2.10, %2373
  %.sroa.234.2 = phi float [ %2390, %2373 ], [ %.sroa.234.1, %._crit_edge.2.10 ]
  %2391 = or i32 %41, 11
  %2392 = icmp slt i32 %2391, %const_reg_dword1
  %2393 = and i1 %76, %2392
  br i1 %2393, label %2394, label %._crit_edge.11

2394:                                             ; preds = %.preheader.10
  %2395 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2395)
  %2396 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2396)
  %2397 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2397, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2398

2398:                                             ; preds = %2398, %2394
  %2399 = phi i32 [ 0, %2394 ], [ %2404, %2398 ]
  %2400 = zext i32 %2399 to i64
  %2401 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2400
  %2402 = load i32, i32* %2401, align 4, !noalias !635
  %2403 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2400
  store i32 %2402, i32* %2403, align 4, !alias.scope !635
  %2404 = add nuw nsw i32 %2399, 1, !spirv.Decorations !620
  %2405 = icmp eq i32 %2399, 0
  br i1 %2405, label %2398, label %2406, !llvm.loop !638

2406:                                             ; preds = %2398
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2396)
  %2407 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2395)
  %2408 = shl i64 %2407, 32
  %2409 = ashr exact i64 %2408, 32
  %2410 = mul nsw i64 %2409, %const_reg_qword3, !spirv.Decorations !610
  %2411 = ashr i64 %2407, 32
  %2412 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2410, i32 0
  %2413 = getelementptr i16, i16 addrspace(4)* %2412, i64 %2411
  %2414 = addrspacecast i16 addrspace(4)* %2413 to i16 addrspace(1)*
  %2415 = load i16, i16 addrspace(1)* %2414, align 2
  %2416 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2416)
  %2417 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2417)
  %2418 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2418, align 4, !noalias !639
  store i32 %2391, i32* %57, align 4, !noalias !639
  br label %2419

2419:                                             ; preds = %2419, %2406
  %2420 = phi i32 [ 0, %2406 ], [ %2425, %2419 ]
  %2421 = zext i32 %2420 to i64
  %2422 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2421
  %2423 = load i32, i32* %2422, align 4, !noalias !639
  %2424 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2421
  store i32 %2423, i32* %2424, align 4, !alias.scope !639
  %2425 = add nuw nsw i32 %2420, 1, !spirv.Decorations !620
  %2426 = icmp eq i32 %2420, 0
  br i1 %2426, label %2419, label %2427, !llvm.loop !642

2427:                                             ; preds = %2419
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2417)
  %2428 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2416)
  %2429 = shl i64 %2428, 32
  %2430 = ashr exact i64 %2429, 32
  %2431 = mul nsw i64 %2430, %const_reg_qword5, !spirv.Decorations !610
  %2432 = ashr i64 %2428, 32
  %2433 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2431, i32 0
  %2434 = getelementptr i16, i16 addrspace(4)* %2433, i64 %2432
  %2435 = addrspacecast i16 addrspace(4)* %2434 to i16 addrspace(1)*
  %2436 = load i16, i16 addrspace(1)* %2435, align 2
  %2437 = zext i16 %2415 to i32
  %2438 = shl nuw i32 %2437, 16, !spirv.Decorations !628
  %2439 = bitcast i32 %2438 to float
  %2440 = zext i16 %2436 to i32
  %2441 = shl nuw i32 %2440, 16, !spirv.Decorations !628
  %2442 = bitcast i32 %2441 to float
  %2443 = fmul reassoc nsz arcp contract float %2439, %2442, !spirv.Decorations !612
  %2444 = fadd reassoc nsz arcp contract float %2443, %.sroa.46.1, !spirv.Decorations !612
  br label %._crit_edge.11

._crit_edge.11:                                   ; preds = %.preheader.10, %2427
  %.sroa.46.2 = phi float [ %2444, %2427 ], [ %.sroa.46.1, %.preheader.10 ]
  %2445 = and i1 %130, %2392
  br i1 %2445, label %2446, label %._crit_edge.1.11

2446:                                             ; preds = %._crit_edge.11
  %2447 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2447)
  %2448 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2448)
  %2449 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %2449, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2450

2450:                                             ; preds = %2450, %2446
  %2451 = phi i32 [ 0, %2446 ], [ %2456, %2450 ]
  %2452 = zext i32 %2451 to i64
  %2453 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2452
  %2454 = load i32, i32* %2453, align 4, !noalias !635
  %2455 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2452
  store i32 %2454, i32* %2455, align 4, !alias.scope !635
  %2456 = add nuw nsw i32 %2451, 1, !spirv.Decorations !620
  %2457 = icmp eq i32 %2451, 0
  br i1 %2457, label %2450, label %2458, !llvm.loop !638

2458:                                             ; preds = %2450
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2448)
  %2459 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2447)
  %2460 = shl i64 %2459, 32
  %2461 = ashr exact i64 %2460, 32
  %2462 = mul nsw i64 %2461, %const_reg_qword3, !spirv.Decorations !610
  %2463 = ashr i64 %2459, 32
  %2464 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2462, i32 0
  %2465 = getelementptr i16, i16 addrspace(4)* %2464, i64 %2463
  %2466 = addrspacecast i16 addrspace(4)* %2465 to i16 addrspace(1)*
  %2467 = load i16, i16 addrspace(1)* %2466, align 2
  %2468 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2468)
  %2469 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2469)
  %2470 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2470, align 4, !noalias !639
  store i32 %2391, i32* %57, align 4, !noalias !639
  br label %2471

2471:                                             ; preds = %2471, %2458
  %2472 = phi i32 [ 0, %2458 ], [ %2477, %2471 ]
  %2473 = zext i32 %2472 to i64
  %2474 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2473
  %2475 = load i32, i32* %2474, align 4, !noalias !639
  %2476 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2473
  store i32 %2475, i32* %2476, align 4, !alias.scope !639
  %2477 = add nuw nsw i32 %2472, 1, !spirv.Decorations !620
  %2478 = icmp eq i32 %2472, 0
  br i1 %2478, label %2471, label %2479, !llvm.loop !642

2479:                                             ; preds = %2471
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2469)
  %2480 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2468)
  %2481 = shl i64 %2480, 32
  %2482 = ashr exact i64 %2481, 32
  %2483 = mul nsw i64 %2482, %const_reg_qword5, !spirv.Decorations !610
  %2484 = ashr i64 %2480, 32
  %2485 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2483, i32 0
  %2486 = getelementptr i16, i16 addrspace(4)* %2485, i64 %2484
  %2487 = addrspacecast i16 addrspace(4)* %2486 to i16 addrspace(1)*
  %2488 = load i16, i16 addrspace(1)* %2487, align 2
  %2489 = zext i16 %2467 to i32
  %2490 = shl nuw i32 %2489, 16, !spirv.Decorations !628
  %2491 = bitcast i32 %2490 to float
  %2492 = zext i16 %2488 to i32
  %2493 = shl nuw i32 %2492, 16, !spirv.Decorations !628
  %2494 = bitcast i32 %2493 to float
  %2495 = fmul reassoc nsz arcp contract float %2491, %2494, !spirv.Decorations !612
  %2496 = fadd reassoc nsz arcp contract float %2495, %.sroa.110.1, !spirv.Decorations !612
  br label %._crit_edge.1.11

._crit_edge.1.11:                                 ; preds = %._crit_edge.11, %2479
  %.sroa.110.2 = phi float [ %2496, %2479 ], [ %.sroa.110.1, %._crit_edge.11 ]
  %2497 = and i1 %184, %2392
  br i1 %2497, label %2498, label %._crit_edge.2.11

2498:                                             ; preds = %._crit_edge.1.11
  %2499 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2499)
  %2500 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2500)
  %2501 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %2501, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2502

2502:                                             ; preds = %2502, %2498
  %2503 = phi i32 [ 0, %2498 ], [ %2508, %2502 ]
  %2504 = zext i32 %2503 to i64
  %2505 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2504
  %2506 = load i32, i32* %2505, align 4, !noalias !635
  %2507 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2504
  store i32 %2506, i32* %2507, align 4, !alias.scope !635
  %2508 = add nuw nsw i32 %2503, 1, !spirv.Decorations !620
  %2509 = icmp eq i32 %2503, 0
  br i1 %2509, label %2502, label %2510, !llvm.loop !638

2510:                                             ; preds = %2502
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2500)
  %2511 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2499)
  %2512 = shl i64 %2511, 32
  %2513 = ashr exact i64 %2512, 32
  %2514 = mul nsw i64 %2513, %const_reg_qword3, !spirv.Decorations !610
  %2515 = ashr i64 %2511, 32
  %2516 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2514, i32 0
  %2517 = getelementptr i16, i16 addrspace(4)* %2516, i64 %2515
  %2518 = addrspacecast i16 addrspace(4)* %2517 to i16 addrspace(1)*
  %2519 = load i16, i16 addrspace(1)* %2518, align 2
  %2520 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2520)
  %2521 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2521)
  %2522 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2522, align 4, !noalias !639
  store i32 %2391, i32* %57, align 4, !noalias !639
  br label %2523

2523:                                             ; preds = %2523, %2510
  %2524 = phi i32 [ 0, %2510 ], [ %2529, %2523 ]
  %2525 = zext i32 %2524 to i64
  %2526 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2525
  %2527 = load i32, i32* %2526, align 4, !noalias !639
  %2528 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2525
  store i32 %2527, i32* %2528, align 4, !alias.scope !639
  %2529 = add nuw nsw i32 %2524, 1, !spirv.Decorations !620
  %2530 = icmp eq i32 %2524, 0
  br i1 %2530, label %2523, label %2531, !llvm.loop !642

2531:                                             ; preds = %2523
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2521)
  %2532 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2520)
  %2533 = shl i64 %2532, 32
  %2534 = ashr exact i64 %2533, 32
  %2535 = mul nsw i64 %2534, %const_reg_qword5, !spirv.Decorations !610
  %2536 = ashr i64 %2532, 32
  %2537 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2535, i32 0
  %2538 = getelementptr i16, i16 addrspace(4)* %2537, i64 %2536
  %2539 = addrspacecast i16 addrspace(4)* %2538 to i16 addrspace(1)*
  %2540 = load i16, i16 addrspace(1)* %2539, align 2
  %2541 = zext i16 %2519 to i32
  %2542 = shl nuw i32 %2541, 16, !spirv.Decorations !628
  %2543 = bitcast i32 %2542 to float
  %2544 = zext i16 %2540 to i32
  %2545 = shl nuw i32 %2544, 16, !spirv.Decorations !628
  %2546 = bitcast i32 %2545 to float
  %2547 = fmul reassoc nsz arcp contract float %2543, %2546, !spirv.Decorations !612
  %2548 = fadd reassoc nsz arcp contract float %2547, %.sroa.174.1, !spirv.Decorations !612
  br label %._crit_edge.2.11

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11, %2531
  %.sroa.174.2 = phi float [ %2548, %2531 ], [ %.sroa.174.1, %._crit_edge.1.11 ]
  %2549 = and i1 %238, %2392
  br i1 %2549, label %2550, label %.preheader.11

2550:                                             ; preds = %._crit_edge.2.11
  %2551 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2551)
  %2552 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2552)
  %2553 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %2553, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2554

2554:                                             ; preds = %2554, %2550
  %2555 = phi i32 [ 0, %2550 ], [ %2560, %2554 ]
  %2556 = zext i32 %2555 to i64
  %2557 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2556
  %2558 = load i32, i32* %2557, align 4, !noalias !635
  %2559 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2556
  store i32 %2558, i32* %2559, align 4, !alias.scope !635
  %2560 = add nuw nsw i32 %2555, 1, !spirv.Decorations !620
  %2561 = icmp eq i32 %2555, 0
  br i1 %2561, label %2554, label %2562, !llvm.loop !638

2562:                                             ; preds = %2554
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2552)
  %2563 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2551)
  %2564 = shl i64 %2563, 32
  %2565 = ashr exact i64 %2564, 32
  %2566 = mul nsw i64 %2565, %const_reg_qword3, !spirv.Decorations !610
  %2567 = ashr i64 %2563, 32
  %2568 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2566, i32 0
  %2569 = getelementptr i16, i16 addrspace(4)* %2568, i64 %2567
  %2570 = addrspacecast i16 addrspace(4)* %2569 to i16 addrspace(1)*
  %2571 = load i16, i16 addrspace(1)* %2570, align 2
  %2572 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2572)
  %2573 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2573)
  %2574 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2574, align 4, !noalias !639
  store i32 %2391, i32* %57, align 4, !noalias !639
  br label %2575

2575:                                             ; preds = %2575, %2562
  %2576 = phi i32 [ 0, %2562 ], [ %2581, %2575 ]
  %2577 = zext i32 %2576 to i64
  %2578 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2577
  %2579 = load i32, i32* %2578, align 4, !noalias !639
  %2580 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2577
  store i32 %2579, i32* %2580, align 4, !alias.scope !639
  %2581 = add nuw nsw i32 %2576, 1, !spirv.Decorations !620
  %2582 = icmp eq i32 %2576, 0
  br i1 %2582, label %2575, label %2583, !llvm.loop !642

2583:                                             ; preds = %2575
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2573)
  %2584 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2572)
  %2585 = shl i64 %2584, 32
  %2586 = ashr exact i64 %2585, 32
  %2587 = mul nsw i64 %2586, %const_reg_qword5, !spirv.Decorations !610
  %2588 = ashr i64 %2584, 32
  %2589 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2587, i32 0
  %2590 = getelementptr i16, i16 addrspace(4)* %2589, i64 %2588
  %2591 = addrspacecast i16 addrspace(4)* %2590 to i16 addrspace(1)*
  %2592 = load i16, i16 addrspace(1)* %2591, align 2
  %2593 = zext i16 %2571 to i32
  %2594 = shl nuw i32 %2593, 16, !spirv.Decorations !628
  %2595 = bitcast i32 %2594 to float
  %2596 = zext i16 %2592 to i32
  %2597 = shl nuw i32 %2596, 16, !spirv.Decorations !628
  %2598 = bitcast i32 %2597 to float
  %2599 = fmul reassoc nsz arcp contract float %2595, %2598, !spirv.Decorations !612
  %2600 = fadd reassoc nsz arcp contract float %2599, %.sroa.238.1, !spirv.Decorations !612
  br label %.preheader.11

.preheader.11:                                    ; preds = %._crit_edge.2.11, %2583
  %.sroa.238.2 = phi float [ %2600, %2583 ], [ %.sroa.238.1, %._crit_edge.2.11 ]
  %2601 = or i32 %41, 12
  %2602 = icmp slt i32 %2601, %const_reg_dword1
  %2603 = and i1 %76, %2602
  br i1 %2603, label %2604, label %._crit_edge.12

2604:                                             ; preds = %.preheader.11
  %2605 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2605)
  %2606 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2606)
  %2607 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2607, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2608

2608:                                             ; preds = %2608, %2604
  %2609 = phi i32 [ 0, %2604 ], [ %2614, %2608 ]
  %2610 = zext i32 %2609 to i64
  %2611 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2610
  %2612 = load i32, i32* %2611, align 4, !noalias !635
  %2613 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2610
  store i32 %2612, i32* %2613, align 4, !alias.scope !635
  %2614 = add nuw nsw i32 %2609, 1, !spirv.Decorations !620
  %2615 = icmp eq i32 %2609, 0
  br i1 %2615, label %2608, label %2616, !llvm.loop !638

2616:                                             ; preds = %2608
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2606)
  %2617 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2605)
  %2618 = shl i64 %2617, 32
  %2619 = ashr exact i64 %2618, 32
  %2620 = mul nsw i64 %2619, %const_reg_qword3, !spirv.Decorations !610
  %2621 = ashr i64 %2617, 32
  %2622 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2620, i32 0
  %2623 = getelementptr i16, i16 addrspace(4)* %2622, i64 %2621
  %2624 = addrspacecast i16 addrspace(4)* %2623 to i16 addrspace(1)*
  %2625 = load i16, i16 addrspace(1)* %2624, align 2
  %2626 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2626)
  %2627 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2627)
  %2628 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2628, align 4, !noalias !639
  store i32 %2601, i32* %57, align 4, !noalias !639
  br label %2629

2629:                                             ; preds = %2629, %2616
  %2630 = phi i32 [ 0, %2616 ], [ %2635, %2629 ]
  %2631 = zext i32 %2630 to i64
  %2632 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2631
  %2633 = load i32, i32* %2632, align 4, !noalias !639
  %2634 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2631
  store i32 %2633, i32* %2634, align 4, !alias.scope !639
  %2635 = add nuw nsw i32 %2630, 1, !spirv.Decorations !620
  %2636 = icmp eq i32 %2630, 0
  br i1 %2636, label %2629, label %2637, !llvm.loop !642

2637:                                             ; preds = %2629
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2627)
  %2638 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2626)
  %2639 = shl i64 %2638, 32
  %2640 = ashr exact i64 %2639, 32
  %2641 = mul nsw i64 %2640, %const_reg_qword5, !spirv.Decorations !610
  %2642 = ashr i64 %2638, 32
  %2643 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2641, i32 0
  %2644 = getelementptr i16, i16 addrspace(4)* %2643, i64 %2642
  %2645 = addrspacecast i16 addrspace(4)* %2644 to i16 addrspace(1)*
  %2646 = load i16, i16 addrspace(1)* %2645, align 2
  %2647 = zext i16 %2625 to i32
  %2648 = shl nuw i32 %2647, 16, !spirv.Decorations !628
  %2649 = bitcast i32 %2648 to float
  %2650 = zext i16 %2646 to i32
  %2651 = shl nuw i32 %2650, 16, !spirv.Decorations !628
  %2652 = bitcast i32 %2651 to float
  %2653 = fmul reassoc nsz arcp contract float %2649, %2652, !spirv.Decorations !612
  %2654 = fadd reassoc nsz arcp contract float %2653, %.sroa.50.1, !spirv.Decorations !612
  br label %._crit_edge.12

._crit_edge.12:                                   ; preds = %.preheader.11, %2637
  %.sroa.50.2 = phi float [ %2654, %2637 ], [ %.sroa.50.1, %.preheader.11 ]
  %2655 = and i1 %130, %2602
  br i1 %2655, label %2656, label %._crit_edge.1.12

2656:                                             ; preds = %._crit_edge.12
  %2657 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2657)
  %2658 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2658)
  %2659 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %2659, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2660

2660:                                             ; preds = %2660, %2656
  %2661 = phi i32 [ 0, %2656 ], [ %2666, %2660 ]
  %2662 = zext i32 %2661 to i64
  %2663 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2662
  %2664 = load i32, i32* %2663, align 4, !noalias !635
  %2665 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2662
  store i32 %2664, i32* %2665, align 4, !alias.scope !635
  %2666 = add nuw nsw i32 %2661, 1, !spirv.Decorations !620
  %2667 = icmp eq i32 %2661, 0
  br i1 %2667, label %2660, label %2668, !llvm.loop !638

2668:                                             ; preds = %2660
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2658)
  %2669 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2657)
  %2670 = shl i64 %2669, 32
  %2671 = ashr exact i64 %2670, 32
  %2672 = mul nsw i64 %2671, %const_reg_qword3, !spirv.Decorations !610
  %2673 = ashr i64 %2669, 32
  %2674 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2672, i32 0
  %2675 = getelementptr i16, i16 addrspace(4)* %2674, i64 %2673
  %2676 = addrspacecast i16 addrspace(4)* %2675 to i16 addrspace(1)*
  %2677 = load i16, i16 addrspace(1)* %2676, align 2
  %2678 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2678)
  %2679 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2679)
  %2680 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2680, align 4, !noalias !639
  store i32 %2601, i32* %57, align 4, !noalias !639
  br label %2681

2681:                                             ; preds = %2681, %2668
  %2682 = phi i32 [ 0, %2668 ], [ %2687, %2681 ]
  %2683 = zext i32 %2682 to i64
  %2684 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2683
  %2685 = load i32, i32* %2684, align 4, !noalias !639
  %2686 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2683
  store i32 %2685, i32* %2686, align 4, !alias.scope !639
  %2687 = add nuw nsw i32 %2682, 1, !spirv.Decorations !620
  %2688 = icmp eq i32 %2682, 0
  br i1 %2688, label %2681, label %2689, !llvm.loop !642

2689:                                             ; preds = %2681
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2679)
  %2690 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2678)
  %2691 = shl i64 %2690, 32
  %2692 = ashr exact i64 %2691, 32
  %2693 = mul nsw i64 %2692, %const_reg_qword5, !spirv.Decorations !610
  %2694 = ashr i64 %2690, 32
  %2695 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2693, i32 0
  %2696 = getelementptr i16, i16 addrspace(4)* %2695, i64 %2694
  %2697 = addrspacecast i16 addrspace(4)* %2696 to i16 addrspace(1)*
  %2698 = load i16, i16 addrspace(1)* %2697, align 2
  %2699 = zext i16 %2677 to i32
  %2700 = shl nuw i32 %2699, 16, !spirv.Decorations !628
  %2701 = bitcast i32 %2700 to float
  %2702 = zext i16 %2698 to i32
  %2703 = shl nuw i32 %2702, 16, !spirv.Decorations !628
  %2704 = bitcast i32 %2703 to float
  %2705 = fmul reassoc nsz arcp contract float %2701, %2704, !spirv.Decorations !612
  %2706 = fadd reassoc nsz arcp contract float %2705, %.sroa.114.1, !spirv.Decorations !612
  br label %._crit_edge.1.12

._crit_edge.1.12:                                 ; preds = %._crit_edge.12, %2689
  %.sroa.114.2 = phi float [ %2706, %2689 ], [ %.sroa.114.1, %._crit_edge.12 ]
  %2707 = and i1 %184, %2602
  br i1 %2707, label %2708, label %._crit_edge.2.12

2708:                                             ; preds = %._crit_edge.1.12
  %2709 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2709)
  %2710 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2710)
  %2711 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %2711, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2712

2712:                                             ; preds = %2712, %2708
  %2713 = phi i32 [ 0, %2708 ], [ %2718, %2712 ]
  %2714 = zext i32 %2713 to i64
  %2715 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2714
  %2716 = load i32, i32* %2715, align 4, !noalias !635
  %2717 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2714
  store i32 %2716, i32* %2717, align 4, !alias.scope !635
  %2718 = add nuw nsw i32 %2713, 1, !spirv.Decorations !620
  %2719 = icmp eq i32 %2713, 0
  br i1 %2719, label %2712, label %2720, !llvm.loop !638

2720:                                             ; preds = %2712
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2710)
  %2721 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2709)
  %2722 = shl i64 %2721, 32
  %2723 = ashr exact i64 %2722, 32
  %2724 = mul nsw i64 %2723, %const_reg_qword3, !spirv.Decorations !610
  %2725 = ashr i64 %2721, 32
  %2726 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2724, i32 0
  %2727 = getelementptr i16, i16 addrspace(4)* %2726, i64 %2725
  %2728 = addrspacecast i16 addrspace(4)* %2727 to i16 addrspace(1)*
  %2729 = load i16, i16 addrspace(1)* %2728, align 2
  %2730 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2730)
  %2731 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2731)
  %2732 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2732, align 4, !noalias !639
  store i32 %2601, i32* %57, align 4, !noalias !639
  br label %2733

2733:                                             ; preds = %2733, %2720
  %2734 = phi i32 [ 0, %2720 ], [ %2739, %2733 ]
  %2735 = zext i32 %2734 to i64
  %2736 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2735
  %2737 = load i32, i32* %2736, align 4, !noalias !639
  %2738 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2735
  store i32 %2737, i32* %2738, align 4, !alias.scope !639
  %2739 = add nuw nsw i32 %2734, 1, !spirv.Decorations !620
  %2740 = icmp eq i32 %2734, 0
  br i1 %2740, label %2733, label %2741, !llvm.loop !642

2741:                                             ; preds = %2733
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2731)
  %2742 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2730)
  %2743 = shl i64 %2742, 32
  %2744 = ashr exact i64 %2743, 32
  %2745 = mul nsw i64 %2744, %const_reg_qword5, !spirv.Decorations !610
  %2746 = ashr i64 %2742, 32
  %2747 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2745, i32 0
  %2748 = getelementptr i16, i16 addrspace(4)* %2747, i64 %2746
  %2749 = addrspacecast i16 addrspace(4)* %2748 to i16 addrspace(1)*
  %2750 = load i16, i16 addrspace(1)* %2749, align 2
  %2751 = zext i16 %2729 to i32
  %2752 = shl nuw i32 %2751, 16, !spirv.Decorations !628
  %2753 = bitcast i32 %2752 to float
  %2754 = zext i16 %2750 to i32
  %2755 = shl nuw i32 %2754, 16, !spirv.Decorations !628
  %2756 = bitcast i32 %2755 to float
  %2757 = fmul reassoc nsz arcp contract float %2753, %2756, !spirv.Decorations !612
  %2758 = fadd reassoc nsz arcp contract float %2757, %.sroa.178.1, !spirv.Decorations !612
  br label %._crit_edge.2.12

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12, %2741
  %.sroa.178.2 = phi float [ %2758, %2741 ], [ %.sroa.178.1, %._crit_edge.1.12 ]
  %2759 = and i1 %238, %2602
  br i1 %2759, label %2760, label %.preheader.12

2760:                                             ; preds = %._crit_edge.2.12
  %2761 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2761)
  %2762 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2762)
  %2763 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %2763, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2764

2764:                                             ; preds = %2764, %2760
  %2765 = phi i32 [ 0, %2760 ], [ %2770, %2764 ]
  %2766 = zext i32 %2765 to i64
  %2767 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2766
  %2768 = load i32, i32* %2767, align 4, !noalias !635
  %2769 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2766
  store i32 %2768, i32* %2769, align 4, !alias.scope !635
  %2770 = add nuw nsw i32 %2765, 1, !spirv.Decorations !620
  %2771 = icmp eq i32 %2765, 0
  br i1 %2771, label %2764, label %2772, !llvm.loop !638

2772:                                             ; preds = %2764
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2762)
  %2773 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2761)
  %2774 = shl i64 %2773, 32
  %2775 = ashr exact i64 %2774, 32
  %2776 = mul nsw i64 %2775, %const_reg_qword3, !spirv.Decorations !610
  %2777 = ashr i64 %2773, 32
  %2778 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2776, i32 0
  %2779 = getelementptr i16, i16 addrspace(4)* %2778, i64 %2777
  %2780 = addrspacecast i16 addrspace(4)* %2779 to i16 addrspace(1)*
  %2781 = load i16, i16 addrspace(1)* %2780, align 2
  %2782 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2782)
  %2783 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2783)
  %2784 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2784, align 4, !noalias !639
  store i32 %2601, i32* %57, align 4, !noalias !639
  br label %2785

2785:                                             ; preds = %2785, %2772
  %2786 = phi i32 [ 0, %2772 ], [ %2791, %2785 ]
  %2787 = zext i32 %2786 to i64
  %2788 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2787
  %2789 = load i32, i32* %2788, align 4, !noalias !639
  %2790 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2787
  store i32 %2789, i32* %2790, align 4, !alias.scope !639
  %2791 = add nuw nsw i32 %2786, 1, !spirv.Decorations !620
  %2792 = icmp eq i32 %2786, 0
  br i1 %2792, label %2785, label %2793, !llvm.loop !642

2793:                                             ; preds = %2785
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2783)
  %2794 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2782)
  %2795 = shl i64 %2794, 32
  %2796 = ashr exact i64 %2795, 32
  %2797 = mul nsw i64 %2796, %const_reg_qword5, !spirv.Decorations !610
  %2798 = ashr i64 %2794, 32
  %2799 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2797, i32 0
  %2800 = getelementptr i16, i16 addrspace(4)* %2799, i64 %2798
  %2801 = addrspacecast i16 addrspace(4)* %2800 to i16 addrspace(1)*
  %2802 = load i16, i16 addrspace(1)* %2801, align 2
  %2803 = zext i16 %2781 to i32
  %2804 = shl nuw i32 %2803, 16, !spirv.Decorations !628
  %2805 = bitcast i32 %2804 to float
  %2806 = zext i16 %2802 to i32
  %2807 = shl nuw i32 %2806, 16, !spirv.Decorations !628
  %2808 = bitcast i32 %2807 to float
  %2809 = fmul reassoc nsz arcp contract float %2805, %2808, !spirv.Decorations !612
  %2810 = fadd reassoc nsz arcp contract float %2809, %.sroa.242.1, !spirv.Decorations !612
  br label %.preheader.12

.preheader.12:                                    ; preds = %._crit_edge.2.12, %2793
  %.sroa.242.2 = phi float [ %2810, %2793 ], [ %.sroa.242.1, %._crit_edge.2.12 ]
  %2811 = or i32 %41, 13
  %2812 = icmp slt i32 %2811, %const_reg_dword1
  %2813 = and i1 %76, %2812
  br i1 %2813, label %2814, label %._crit_edge.13

2814:                                             ; preds = %.preheader.12
  %2815 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2815)
  %2816 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2816)
  %2817 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %2817, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2818

2818:                                             ; preds = %2818, %2814
  %2819 = phi i32 [ 0, %2814 ], [ %2824, %2818 ]
  %2820 = zext i32 %2819 to i64
  %2821 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2820
  %2822 = load i32, i32* %2821, align 4, !noalias !635
  %2823 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2820
  store i32 %2822, i32* %2823, align 4, !alias.scope !635
  %2824 = add nuw nsw i32 %2819, 1, !spirv.Decorations !620
  %2825 = icmp eq i32 %2819, 0
  br i1 %2825, label %2818, label %2826, !llvm.loop !638

2826:                                             ; preds = %2818
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2816)
  %2827 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2815)
  %2828 = shl i64 %2827, 32
  %2829 = ashr exact i64 %2828, 32
  %2830 = mul nsw i64 %2829, %const_reg_qword3, !spirv.Decorations !610
  %2831 = ashr i64 %2827, 32
  %2832 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2830, i32 0
  %2833 = getelementptr i16, i16 addrspace(4)* %2832, i64 %2831
  %2834 = addrspacecast i16 addrspace(4)* %2833 to i16 addrspace(1)*
  %2835 = load i16, i16 addrspace(1)* %2834, align 2
  %2836 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2836)
  %2837 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2837)
  %2838 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2838, align 4, !noalias !639
  store i32 %2811, i32* %57, align 4, !noalias !639
  br label %2839

2839:                                             ; preds = %2839, %2826
  %2840 = phi i32 [ 0, %2826 ], [ %2845, %2839 ]
  %2841 = zext i32 %2840 to i64
  %2842 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2841
  %2843 = load i32, i32* %2842, align 4, !noalias !639
  %2844 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2841
  store i32 %2843, i32* %2844, align 4, !alias.scope !639
  %2845 = add nuw nsw i32 %2840, 1, !spirv.Decorations !620
  %2846 = icmp eq i32 %2840, 0
  br i1 %2846, label %2839, label %2847, !llvm.loop !642

2847:                                             ; preds = %2839
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2837)
  %2848 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2836)
  %2849 = shl i64 %2848, 32
  %2850 = ashr exact i64 %2849, 32
  %2851 = mul nsw i64 %2850, %const_reg_qword5, !spirv.Decorations !610
  %2852 = ashr i64 %2848, 32
  %2853 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2851, i32 0
  %2854 = getelementptr i16, i16 addrspace(4)* %2853, i64 %2852
  %2855 = addrspacecast i16 addrspace(4)* %2854 to i16 addrspace(1)*
  %2856 = load i16, i16 addrspace(1)* %2855, align 2
  %2857 = zext i16 %2835 to i32
  %2858 = shl nuw i32 %2857, 16, !spirv.Decorations !628
  %2859 = bitcast i32 %2858 to float
  %2860 = zext i16 %2856 to i32
  %2861 = shl nuw i32 %2860, 16, !spirv.Decorations !628
  %2862 = bitcast i32 %2861 to float
  %2863 = fmul reassoc nsz arcp contract float %2859, %2862, !spirv.Decorations !612
  %2864 = fadd reassoc nsz arcp contract float %2863, %.sroa.54.1, !spirv.Decorations !612
  br label %._crit_edge.13

._crit_edge.13:                                   ; preds = %.preheader.12, %2847
  %.sroa.54.2 = phi float [ %2864, %2847 ], [ %.sroa.54.1, %.preheader.12 ]
  %2865 = and i1 %130, %2812
  br i1 %2865, label %2866, label %._crit_edge.1.13

2866:                                             ; preds = %._crit_edge.13
  %2867 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2867)
  %2868 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2868)
  %2869 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %2869, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2870

2870:                                             ; preds = %2870, %2866
  %2871 = phi i32 [ 0, %2866 ], [ %2876, %2870 ]
  %2872 = zext i32 %2871 to i64
  %2873 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2872
  %2874 = load i32, i32* %2873, align 4, !noalias !635
  %2875 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2872
  store i32 %2874, i32* %2875, align 4, !alias.scope !635
  %2876 = add nuw nsw i32 %2871, 1, !spirv.Decorations !620
  %2877 = icmp eq i32 %2871, 0
  br i1 %2877, label %2870, label %2878, !llvm.loop !638

2878:                                             ; preds = %2870
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2868)
  %2879 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2867)
  %2880 = shl i64 %2879, 32
  %2881 = ashr exact i64 %2880, 32
  %2882 = mul nsw i64 %2881, %const_reg_qword3, !spirv.Decorations !610
  %2883 = ashr i64 %2879, 32
  %2884 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2882, i32 0
  %2885 = getelementptr i16, i16 addrspace(4)* %2884, i64 %2883
  %2886 = addrspacecast i16 addrspace(4)* %2885 to i16 addrspace(1)*
  %2887 = load i16, i16 addrspace(1)* %2886, align 2
  %2888 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2888)
  %2889 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2889)
  %2890 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2890, align 4, !noalias !639
  store i32 %2811, i32* %57, align 4, !noalias !639
  br label %2891

2891:                                             ; preds = %2891, %2878
  %2892 = phi i32 [ 0, %2878 ], [ %2897, %2891 ]
  %2893 = zext i32 %2892 to i64
  %2894 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2893
  %2895 = load i32, i32* %2894, align 4, !noalias !639
  %2896 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2893
  store i32 %2895, i32* %2896, align 4, !alias.scope !639
  %2897 = add nuw nsw i32 %2892, 1, !spirv.Decorations !620
  %2898 = icmp eq i32 %2892, 0
  br i1 %2898, label %2891, label %2899, !llvm.loop !642

2899:                                             ; preds = %2891
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2889)
  %2900 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2888)
  %2901 = shl i64 %2900, 32
  %2902 = ashr exact i64 %2901, 32
  %2903 = mul nsw i64 %2902, %const_reg_qword5, !spirv.Decorations !610
  %2904 = ashr i64 %2900, 32
  %2905 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2903, i32 0
  %2906 = getelementptr i16, i16 addrspace(4)* %2905, i64 %2904
  %2907 = addrspacecast i16 addrspace(4)* %2906 to i16 addrspace(1)*
  %2908 = load i16, i16 addrspace(1)* %2907, align 2
  %2909 = zext i16 %2887 to i32
  %2910 = shl nuw i32 %2909, 16, !spirv.Decorations !628
  %2911 = bitcast i32 %2910 to float
  %2912 = zext i16 %2908 to i32
  %2913 = shl nuw i32 %2912, 16, !spirv.Decorations !628
  %2914 = bitcast i32 %2913 to float
  %2915 = fmul reassoc nsz arcp contract float %2911, %2914, !spirv.Decorations !612
  %2916 = fadd reassoc nsz arcp contract float %2915, %.sroa.118.1, !spirv.Decorations !612
  br label %._crit_edge.1.13

._crit_edge.1.13:                                 ; preds = %._crit_edge.13, %2899
  %.sroa.118.2 = phi float [ %2916, %2899 ], [ %.sroa.118.1, %._crit_edge.13 ]
  %2917 = and i1 %184, %2812
  br i1 %2917, label %2918, label %._crit_edge.2.13

2918:                                             ; preds = %._crit_edge.1.13
  %2919 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2919)
  %2920 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2920)
  %2921 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %2921, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2922

2922:                                             ; preds = %2922, %2918
  %2923 = phi i32 [ 0, %2918 ], [ %2928, %2922 ]
  %2924 = zext i32 %2923 to i64
  %2925 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2924
  %2926 = load i32, i32* %2925, align 4, !noalias !635
  %2927 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2924
  store i32 %2926, i32* %2927, align 4, !alias.scope !635
  %2928 = add nuw nsw i32 %2923, 1, !spirv.Decorations !620
  %2929 = icmp eq i32 %2923, 0
  br i1 %2929, label %2922, label %2930, !llvm.loop !638

2930:                                             ; preds = %2922
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2920)
  %2931 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2919)
  %2932 = shl i64 %2931, 32
  %2933 = ashr exact i64 %2932, 32
  %2934 = mul nsw i64 %2933, %const_reg_qword3, !spirv.Decorations !610
  %2935 = ashr i64 %2931, 32
  %2936 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2934, i32 0
  %2937 = getelementptr i16, i16 addrspace(4)* %2936, i64 %2935
  %2938 = addrspacecast i16 addrspace(4)* %2937 to i16 addrspace(1)*
  %2939 = load i16, i16 addrspace(1)* %2938, align 2
  %2940 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2940)
  %2941 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2941)
  %2942 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2942, align 4, !noalias !639
  store i32 %2811, i32* %57, align 4, !noalias !639
  br label %2943

2943:                                             ; preds = %2943, %2930
  %2944 = phi i32 [ 0, %2930 ], [ %2949, %2943 ]
  %2945 = zext i32 %2944 to i64
  %2946 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2945
  %2947 = load i32, i32* %2946, align 4, !noalias !639
  %2948 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2945
  store i32 %2947, i32* %2948, align 4, !alias.scope !639
  %2949 = add nuw nsw i32 %2944, 1, !spirv.Decorations !620
  %2950 = icmp eq i32 %2944, 0
  br i1 %2950, label %2943, label %2951, !llvm.loop !642

2951:                                             ; preds = %2943
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2941)
  %2952 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2940)
  %2953 = shl i64 %2952, 32
  %2954 = ashr exact i64 %2953, 32
  %2955 = mul nsw i64 %2954, %const_reg_qword5, !spirv.Decorations !610
  %2956 = ashr i64 %2952, 32
  %2957 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %2955, i32 0
  %2958 = getelementptr i16, i16 addrspace(4)* %2957, i64 %2956
  %2959 = addrspacecast i16 addrspace(4)* %2958 to i16 addrspace(1)*
  %2960 = load i16, i16 addrspace(1)* %2959, align 2
  %2961 = zext i16 %2939 to i32
  %2962 = shl nuw i32 %2961, 16, !spirv.Decorations !628
  %2963 = bitcast i32 %2962 to float
  %2964 = zext i16 %2960 to i32
  %2965 = shl nuw i32 %2964, 16, !spirv.Decorations !628
  %2966 = bitcast i32 %2965 to float
  %2967 = fmul reassoc nsz arcp contract float %2963, %2966, !spirv.Decorations !612
  %2968 = fadd reassoc nsz arcp contract float %2967, %.sroa.182.1, !spirv.Decorations !612
  br label %._crit_edge.2.13

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13, %2951
  %.sroa.182.2 = phi float [ %2968, %2951 ], [ %.sroa.182.1, %._crit_edge.1.13 ]
  %2969 = and i1 %238, %2812
  br i1 %2969, label %2970, label %.preheader.13

2970:                                             ; preds = %._crit_edge.2.13
  %2971 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2971)
  %2972 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2972)
  %2973 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %2973, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %2974

2974:                                             ; preds = %2974, %2970
  %2975 = phi i32 [ 0, %2970 ], [ %2980, %2974 ]
  %2976 = zext i32 %2975 to i64
  %2977 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %2976
  %2978 = load i32, i32* %2977, align 4, !noalias !635
  %2979 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %2976
  store i32 %2978, i32* %2979, align 4, !alias.scope !635
  %2980 = add nuw nsw i32 %2975, 1, !spirv.Decorations !620
  %2981 = icmp eq i32 %2975, 0
  br i1 %2981, label %2974, label %2982, !llvm.loop !638

2982:                                             ; preds = %2974
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2972)
  %2983 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2971)
  %2984 = shl i64 %2983, 32
  %2985 = ashr exact i64 %2984, 32
  %2986 = mul nsw i64 %2985, %const_reg_qword3, !spirv.Decorations !610
  %2987 = ashr i64 %2983, 32
  %2988 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %2986, i32 0
  %2989 = getelementptr i16, i16 addrspace(4)* %2988, i64 %2987
  %2990 = addrspacecast i16 addrspace(4)* %2989 to i16 addrspace(1)*
  %2991 = load i16, i16 addrspace(1)* %2990, align 2
  %2992 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2992)
  %2993 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %2993)
  %2994 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %2994, align 4, !noalias !639
  store i32 %2811, i32* %57, align 4, !noalias !639
  br label %2995

2995:                                             ; preds = %2995, %2982
  %2996 = phi i32 [ 0, %2982 ], [ %3001, %2995 ]
  %2997 = zext i32 %2996 to i64
  %2998 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %2997
  %2999 = load i32, i32* %2998, align 4, !noalias !639
  %3000 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %2997
  store i32 %2999, i32* %3000, align 4, !alias.scope !639
  %3001 = add nuw nsw i32 %2996, 1, !spirv.Decorations !620
  %3002 = icmp eq i32 %2996, 0
  br i1 %3002, label %2995, label %3003, !llvm.loop !642

3003:                                             ; preds = %2995
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2993)
  %3004 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %2992)
  %3005 = shl i64 %3004, 32
  %3006 = ashr exact i64 %3005, 32
  %3007 = mul nsw i64 %3006, %const_reg_qword5, !spirv.Decorations !610
  %3008 = ashr i64 %3004, 32
  %3009 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3007, i32 0
  %3010 = getelementptr i16, i16 addrspace(4)* %3009, i64 %3008
  %3011 = addrspacecast i16 addrspace(4)* %3010 to i16 addrspace(1)*
  %3012 = load i16, i16 addrspace(1)* %3011, align 2
  %3013 = zext i16 %2991 to i32
  %3014 = shl nuw i32 %3013, 16, !spirv.Decorations !628
  %3015 = bitcast i32 %3014 to float
  %3016 = zext i16 %3012 to i32
  %3017 = shl nuw i32 %3016, 16, !spirv.Decorations !628
  %3018 = bitcast i32 %3017 to float
  %3019 = fmul reassoc nsz arcp contract float %3015, %3018, !spirv.Decorations !612
  %3020 = fadd reassoc nsz arcp contract float %3019, %.sroa.246.1, !spirv.Decorations !612
  br label %.preheader.13

.preheader.13:                                    ; preds = %._crit_edge.2.13, %3003
  %.sroa.246.2 = phi float [ %3020, %3003 ], [ %.sroa.246.1, %._crit_edge.2.13 ]
  %3021 = or i32 %41, 14
  %3022 = icmp slt i32 %3021, %const_reg_dword1
  %3023 = and i1 %76, %3022
  br i1 %3023, label %3024, label %._crit_edge.14

3024:                                             ; preds = %.preheader.13
  %3025 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3025)
  %3026 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3026)
  %3027 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %3027, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3028

3028:                                             ; preds = %3028, %3024
  %3029 = phi i32 [ 0, %3024 ], [ %3034, %3028 ]
  %3030 = zext i32 %3029 to i64
  %3031 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3030
  %3032 = load i32, i32* %3031, align 4, !noalias !635
  %3033 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3030
  store i32 %3032, i32* %3033, align 4, !alias.scope !635
  %3034 = add nuw nsw i32 %3029, 1, !spirv.Decorations !620
  %3035 = icmp eq i32 %3029, 0
  br i1 %3035, label %3028, label %3036, !llvm.loop !638

3036:                                             ; preds = %3028
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3026)
  %3037 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3025)
  %3038 = shl i64 %3037, 32
  %3039 = ashr exact i64 %3038, 32
  %3040 = mul nsw i64 %3039, %const_reg_qword3, !spirv.Decorations !610
  %3041 = ashr i64 %3037, 32
  %3042 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3040, i32 0
  %3043 = getelementptr i16, i16 addrspace(4)* %3042, i64 %3041
  %3044 = addrspacecast i16 addrspace(4)* %3043 to i16 addrspace(1)*
  %3045 = load i16, i16 addrspace(1)* %3044, align 2
  %3046 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3046)
  %3047 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3047)
  %3048 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3048, align 4, !noalias !639
  store i32 %3021, i32* %57, align 4, !noalias !639
  br label %3049

3049:                                             ; preds = %3049, %3036
  %3050 = phi i32 [ 0, %3036 ], [ %3055, %3049 ]
  %3051 = zext i32 %3050 to i64
  %3052 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3051
  %3053 = load i32, i32* %3052, align 4, !noalias !639
  %3054 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3051
  store i32 %3053, i32* %3054, align 4, !alias.scope !639
  %3055 = add nuw nsw i32 %3050, 1, !spirv.Decorations !620
  %3056 = icmp eq i32 %3050, 0
  br i1 %3056, label %3049, label %3057, !llvm.loop !642

3057:                                             ; preds = %3049
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3047)
  %3058 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3046)
  %3059 = shl i64 %3058, 32
  %3060 = ashr exact i64 %3059, 32
  %3061 = mul nsw i64 %3060, %const_reg_qword5, !spirv.Decorations !610
  %3062 = ashr i64 %3058, 32
  %3063 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3061, i32 0
  %3064 = getelementptr i16, i16 addrspace(4)* %3063, i64 %3062
  %3065 = addrspacecast i16 addrspace(4)* %3064 to i16 addrspace(1)*
  %3066 = load i16, i16 addrspace(1)* %3065, align 2
  %3067 = zext i16 %3045 to i32
  %3068 = shl nuw i32 %3067, 16, !spirv.Decorations !628
  %3069 = bitcast i32 %3068 to float
  %3070 = zext i16 %3066 to i32
  %3071 = shl nuw i32 %3070, 16, !spirv.Decorations !628
  %3072 = bitcast i32 %3071 to float
  %3073 = fmul reassoc nsz arcp contract float %3069, %3072, !spirv.Decorations !612
  %3074 = fadd reassoc nsz arcp contract float %3073, %.sroa.58.1, !spirv.Decorations !612
  br label %._crit_edge.14

._crit_edge.14:                                   ; preds = %.preheader.13, %3057
  %.sroa.58.2 = phi float [ %3074, %3057 ], [ %.sroa.58.1, %.preheader.13 ]
  %3075 = and i1 %130, %3022
  br i1 %3075, label %3076, label %._crit_edge.1.14

3076:                                             ; preds = %._crit_edge.14
  %3077 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3077)
  %3078 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3078)
  %3079 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %3079, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3080

3080:                                             ; preds = %3080, %3076
  %3081 = phi i32 [ 0, %3076 ], [ %3086, %3080 ]
  %3082 = zext i32 %3081 to i64
  %3083 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3082
  %3084 = load i32, i32* %3083, align 4, !noalias !635
  %3085 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3082
  store i32 %3084, i32* %3085, align 4, !alias.scope !635
  %3086 = add nuw nsw i32 %3081, 1, !spirv.Decorations !620
  %3087 = icmp eq i32 %3081, 0
  br i1 %3087, label %3080, label %3088, !llvm.loop !638

3088:                                             ; preds = %3080
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3078)
  %3089 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3077)
  %3090 = shl i64 %3089, 32
  %3091 = ashr exact i64 %3090, 32
  %3092 = mul nsw i64 %3091, %const_reg_qword3, !spirv.Decorations !610
  %3093 = ashr i64 %3089, 32
  %3094 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3092, i32 0
  %3095 = getelementptr i16, i16 addrspace(4)* %3094, i64 %3093
  %3096 = addrspacecast i16 addrspace(4)* %3095 to i16 addrspace(1)*
  %3097 = load i16, i16 addrspace(1)* %3096, align 2
  %3098 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3098)
  %3099 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3099)
  %3100 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3100, align 4, !noalias !639
  store i32 %3021, i32* %57, align 4, !noalias !639
  br label %3101

3101:                                             ; preds = %3101, %3088
  %3102 = phi i32 [ 0, %3088 ], [ %3107, %3101 ]
  %3103 = zext i32 %3102 to i64
  %3104 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3103
  %3105 = load i32, i32* %3104, align 4, !noalias !639
  %3106 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3103
  store i32 %3105, i32* %3106, align 4, !alias.scope !639
  %3107 = add nuw nsw i32 %3102, 1, !spirv.Decorations !620
  %3108 = icmp eq i32 %3102, 0
  br i1 %3108, label %3101, label %3109, !llvm.loop !642

3109:                                             ; preds = %3101
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3099)
  %3110 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3098)
  %3111 = shl i64 %3110, 32
  %3112 = ashr exact i64 %3111, 32
  %3113 = mul nsw i64 %3112, %const_reg_qword5, !spirv.Decorations !610
  %3114 = ashr i64 %3110, 32
  %3115 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3113, i32 0
  %3116 = getelementptr i16, i16 addrspace(4)* %3115, i64 %3114
  %3117 = addrspacecast i16 addrspace(4)* %3116 to i16 addrspace(1)*
  %3118 = load i16, i16 addrspace(1)* %3117, align 2
  %3119 = zext i16 %3097 to i32
  %3120 = shl nuw i32 %3119, 16, !spirv.Decorations !628
  %3121 = bitcast i32 %3120 to float
  %3122 = zext i16 %3118 to i32
  %3123 = shl nuw i32 %3122, 16, !spirv.Decorations !628
  %3124 = bitcast i32 %3123 to float
  %3125 = fmul reassoc nsz arcp contract float %3121, %3124, !spirv.Decorations !612
  %3126 = fadd reassoc nsz arcp contract float %3125, %.sroa.122.1, !spirv.Decorations !612
  br label %._crit_edge.1.14

._crit_edge.1.14:                                 ; preds = %._crit_edge.14, %3109
  %.sroa.122.2 = phi float [ %3126, %3109 ], [ %.sroa.122.1, %._crit_edge.14 ]
  %3127 = and i1 %184, %3022
  br i1 %3127, label %3128, label %._crit_edge.2.14

3128:                                             ; preds = %._crit_edge.1.14
  %3129 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3129)
  %3130 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3130)
  %3131 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %3131, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3132

3132:                                             ; preds = %3132, %3128
  %3133 = phi i32 [ 0, %3128 ], [ %3138, %3132 ]
  %3134 = zext i32 %3133 to i64
  %3135 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3134
  %3136 = load i32, i32* %3135, align 4, !noalias !635
  %3137 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3134
  store i32 %3136, i32* %3137, align 4, !alias.scope !635
  %3138 = add nuw nsw i32 %3133, 1, !spirv.Decorations !620
  %3139 = icmp eq i32 %3133, 0
  br i1 %3139, label %3132, label %3140, !llvm.loop !638

3140:                                             ; preds = %3132
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3130)
  %3141 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3129)
  %3142 = shl i64 %3141, 32
  %3143 = ashr exact i64 %3142, 32
  %3144 = mul nsw i64 %3143, %const_reg_qword3, !spirv.Decorations !610
  %3145 = ashr i64 %3141, 32
  %3146 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3144, i32 0
  %3147 = getelementptr i16, i16 addrspace(4)* %3146, i64 %3145
  %3148 = addrspacecast i16 addrspace(4)* %3147 to i16 addrspace(1)*
  %3149 = load i16, i16 addrspace(1)* %3148, align 2
  %3150 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3150)
  %3151 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3151)
  %3152 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3152, align 4, !noalias !639
  store i32 %3021, i32* %57, align 4, !noalias !639
  br label %3153

3153:                                             ; preds = %3153, %3140
  %3154 = phi i32 [ 0, %3140 ], [ %3159, %3153 ]
  %3155 = zext i32 %3154 to i64
  %3156 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3155
  %3157 = load i32, i32* %3156, align 4, !noalias !639
  %3158 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3155
  store i32 %3157, i32* %3158, align 4, !alias.scope !639
  %3159 = add nuw nsw i32 %3154, 1, !spirv.Decorations !620
  %3160 = icmp eq i32 %3154, 0
  br i1 %3160, label %3153, label %3161, !llvm.loop !642

3161:                                             ; preds = %3153
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3151)
  %3162 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3150)
  %3163 = shl i64 %3162, 32
  %3164 = ashr exact i64 %3163, 32
  %3165 = mul nsw i64 %3164, %const_reg_qword5, !spirv.Decorations !610
  %3166 = ashr i64 %3162, 32
  %3167 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3165, i32 0
  %3168 = getelementptr i16, i16 addrspace(4)* %3167, i64 %3166
  %3169 = addrspacecast i16 addrspace(4)* %3168 to i16 addrspace(1)*
  %3170 = load i16, i16 addrspace(1)* %3169, align 2
  %3171 = zext i16 %3149 to i32
  %3172 = shl nuw i32 %3171, 16, !spirv.Decorations !628
  %3173 = bitcast i32 %3172 to float
  %3174 = zext i16 %3170 to i32
  %3175 = shl nuw i32 %3174, 16, !spirv.Decorations !628
  %3176 = bitcast i32 %3175 to float
  %3177 = fmul reassoc nsz arcp contract float %3173, %3176, !spirv.Decorations !612
  %3178 = fadd reassoc nsz arcp contract float %3177, %.sroa.186.1, !spirv.Decorations !612
  br label %._crit_edge.2.14

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14, %3161
  %.sroa.186.2 = phi float [ %3178, %3161 ], [ %.sroa.186.1, %._crit_edge.1.14 ]
  %3179 = and i1 %238, %3022
  br i1 %3179, label %3180, label %.preheader.14

3180:                                             ; preds = %._crit_edge.2.14
  %3181 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3181)
  %3182 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3182)
  %3183 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %3183, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3184

3184:                                             ; preds = %3184, %3180
  %3185 = phi i32 [ 0, %3180 ], [ %3190, %3184 ]
  %3186 = zext i32 %3185 to i64
  %3187 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3186
  %3188 = load i32, i32* %3187, align 4, !noalias !635
  %3189 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3186
  store i32 %3188, i32* %3189, align 4, !alias.scope !635
  %3190 = add nuw nsw i32 %3185, 1, !spirv.Decorations !620
  %3191 = icmp eq i32 %3185, 0
  br i1 %3191, label %3184, label %3192, !llvm.loop !638

3192:                                             ; preds = %3184
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3182)
  %3193 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3181)
  %3194 = shl i64 %3193, 32
  %3195 = ashr exact i64 %3194, 32
  %3196 = mul nsw i64 %3195, %const_reg_qword3, !spirv.Decorations !610
  %3197 = ashr i64 %3193, 32
  %3198 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3196, i32 0
  %3199 = getelementptr i16, i16 addrspace(4)* %3198, i64 %3197
  %3200 = addrspacecast i16 addrspace(4)* %3199 to i16 addrspace(1)*
  %3201 = load i16, i16 addrspace(1)* %3200, align 2
  %3202 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3202)
  %3203 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3203)
  %3204 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3204, align 4, !noalias !639
  store i32 %3021, i32* %57, align 4, !noalias !639
  br label %3205

3205:                                             ; preds = %3205, %3192
  %3206 = phi i32 [ 0, %3192 ], [ %3211, %3205 ]
  %3207 = zext i32 %3206 to i64
  %3208 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3207
  %3209 = load i32, i32* %3208, align 4, !noalias !639
  %3210 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3207
  store i32 %3209, i32* %3210, align 4, !alias.scope !639
  %3211 = add nuw nsw i32 %3206, 1, !spirv.Decorations !620
  %3212 = icmp eq i32 %3206, 0
  br i1 %3212, label %3205, label %3213, !llvm.loop !642

3213:                                             ; preds = %3205
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3203)
  %3214 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3202)
  %3215 = shl i64 %3214, 32
  %3216 = ashr exact i64 %3215, 32
  %3217 = mul nsw i64 %3216, %const_reg_qword5, !spirv.Decorations !610
  %3218 = ashr i64 %3214, 32
  %3219 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3217, i32 0
  %3220 = getelementptr i16, i16 addrspace(4)* %3219, i64 %3218
  %3221 = addrspacecast i16 addrspace(4)* %3220 to i16 addrspace(1)*
  %3222 = load i16, i16 addrspace(1)* %3221, align 2
  %3223 = zext i16 %3201 to i32
  %3224 = shl nuw i32 %3223, 16, !spirv.Decorations !628
  %3225 = bitcast i32 %3224 to float
  %3226 = zext i16 %3222 to i32
  %3227 = shl nuw i32 %3226, 16, !spirv.Decorations !628
  %3228 = bitcast i32 %3227 to float
  %3229 = fmul reassoc nsz arcp contract float %3225, %3228, !spirv.Decorations !612
  %3230 = fadd reassoc nsz arcp contract float %3229, %.sroa.250.1, !spirv.Decorations !612
  br label %.preheader.14

.preheader.14:                                    ; preds = %._crit_edge.2.14, %3213
  %.sroa.250.2 = phi float [ %3230, %3213 ], [ %.sroa.250.1, %._crit_edge.2.14 ]
  %3231 = or i32 %41, 15
  %3232 = icmp slt i32 %3231, %const_reg_dword1
  %3233 = and i1 %76, %3232
  br i1 %3233, label %3234, label %._crit_edge.15

3234:                                             ; preds = %.preheader.14
  %3235 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3235)
  %3236 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3236)
  %3237 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %35, i32* %3237, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3238

3238:                                             ; preds = %3238, %3234
  %3239 = phi i32 [ 0, %3234 ], [ %3244, %3238 ]
  %3240 = zext i32 %3239 to i64
  %3241 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3240
  %3242 = load i32, i32* %3241, align 4, !noalias !635
  %3243 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3240
  store i32 %3242, i32* %3243, align 4, !alias.scope !635
  %3244 = add nuw nsw i32 %3239, 1, !spirv.Decorations !620
  %3245 = icmp eq i32 %3239, 0
  br i1 %3245, label %3238, label %3246, !llvm.loop !638

3246:                                             ; preds = %3238
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3236)
  %3247 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3235)
  %3248 = shl i64 %3247, 32
  %3249 = ashr exact i64 %3248, 32
  %3250 = mul nsw i64 %3249, %const_reg_qword3, !spirv.Decorations !610
  %3251 = ashr i64 %3247, 32
  %3252 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3250, i32 0
  %3253 = getelementptr i16, i16 addrspace(4)* %3252, i64 %3251
  %3254 = addrspacecast i16 addrspace(4)* %3253 to i16 addrspace(1)*
  %3255 = load i16, i16 addrspace(1)* %3254, align 2
  %3256 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3256)
  %3257 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3257)
  %3258 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3258, align 4, !noalias !639
  store i32 %3231, i32* %57, align 4, !noalias !639
  br label %3259

3259:                                             ; preds = %3259, %3246
  %3260 = phi i32 [ 0, %3246 ], [ %3265, %3259 ]
  %3261 = zext i32 %3260 to i64
  %3262 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3261
  %3263 = load i32, i32* %3262, align 4, !noalias !639
  %3264 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3261
  store i32 %3263, i32* %3264, align 4, !alias.scope !639
  %3265 = add nuw nsw i32 %3260, 1, !spirv.Decorations !620
  %3266 = icmp eq i32 %3260, 0
  br i1 %3266, label %3259, label %3267, !llvm.loop !642

3267:                                             ; preds = %3259
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3257)
  %3268 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3256)
  %3269 = shl i64 %3268, 32
  %3270 = ashr exact i64 %3269, 32
  %3271 = mul nsw i64 %3270, %const_reg_qword5, !spirv.Decorations !610
  %3272 = ashr i64 %3268, 32
  %3273 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3271, i32 0
  %3274 = getelementptr i16, i16 addrspace(4)* %3273, i64 %3272
  %3275 = addrspacecast i16 addrspace(4)* %3274 to i16 addrspace(1)*
  %3276 = load i16, i16 addrspace(1)* %3275, align 2
  %3277 = zext i16 %3255 to i32
  %3278 = shl nuw i32 %3277, 16, !spirv.Decorations !628
  %3279 = bitcast i32 %3278 to float
  %3280 = zext i16 %3276 to i32
  %3281 = shl nuw i32 %3280, 16, !spirv.Decorations !628
  %3282 = bitcast i32 %3281 to float
  %3283 = fmul reassoc nsz arcp contract float %3279, %3282, !spirv.Decorations !612
  %3284 = fadd reassoc nsz arcp contract float %3283, %.sroa.62.1, !spirv.Decorations !612
  br label %._crit_edge.15

._crit_edge.15:                                   ; preds = %.preheader.14, %3267
  %.sroa.62.2 = phi float [ %3284, %3267 ], [ %.sroa.62.1, %.preheader.14 ]
  %3285 = and i1 %130, %3232
  br i1 %3285, label %3286, label %._crit_edge.1.15

3286:                                             ; preds = %._crit_edge.15
  %3287 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3287)
  %3288 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3288)
  %3289 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %129, i32* %3289, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3290

3290:                                             ; preds = %3290, %3286
  %3291 = phi i32 [ 0, %3286 ], [ %3296, %3290 ]
  %3292 = zext i32 %3291 to i64
  %3293 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3292
  %3294 = load i32, i32* %3293, align 4, !noalias !635
  %3295 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3292
  store i32 %3294, i32* %3295, align 4, !alias.scope !635
  %3296 = add nuw nsw i32 %3291, 1, !spirv.Decorations !620
  %3297 = icmp eq i32 %3291, 0
  br i1 %3297, label %3290, label %3298, !llvm.loop !638

3298:                                             ; preds = %3290
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3288)
  %3299 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3287)
  %3300 = shl i64 %3299, 32
  %3301 = ashr exact i64 %3300, 32
  %3302 = mul nsw i64 %3301, %const_reg_qword3, !spirv.Decorations !610
  %3303 = ashr i64 %3299, 32
  %3304 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3302, i32 0
  %3305 = getelementptr i16, i16 addrspace(4)* %3304, i64 %3303
  %3306 = addrspacecast i16 addrspace(4)* %3305 to i16 addrspace(1)*
  %3307 = load i16, i16 addrspace(1)* %3306, align 2
  %3308 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3308)
  %3309 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3309)
  %3310 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3310, align 4, !noalias !639
  store i32 %3231, i32* %57, align 4, !noalias !639
  br label %3311

3311:                                             ; preds = %3311, %3298
  %3312 = phi i32 [ 0, %3298 ], [ %3317, %3311 ]
  %3313 = zext i32 %3312 to i64
  %3314 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3313
  %3315 = load i32, i32* %3314, align 4, !noalias !639
  %3316 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3313
  store i32 %3315, i32* %3316, align 4, !alias.scope !639
  %3317 = add nuw nsw i32 %3312, 1, !spirv.Decorations !620
  %3318 = icmp eq i32 %3312, 0
  br i1 %3318, label %3311, label %3319, !llvm.loop !642

3319:                                             ; preds = %3311
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3309)
  %3320 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3308)
  %3321 = shl i64 %3320, 32
  %3322 = ashr exact i64 %3321, 32
  %3323 = mul nsw i64 %3322, %const_reg_qword5, !spirv.Decorations !610
  %3324 = ashr i64 %3320, 32
  %3325 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3323, i32 0
  %3326 = getelementptr i16, i16 addrspace(4)* %3325, i64 %3324
  %3327 = addrspacecast i16 addrspace(4)* %3326 to i16 addrspace(1)*
  %3328 = load i16, i16 addrspace(1)* %3327, align 2
  %3329 = zext i16 %3307 to i32
  %3330 = shl nuw i32 %3329, 16, !spirv.Decorations !628
  %3331 = bitcast i32 %3330 to float
  %3332 = zext i16 %3328 to i32
  %3333 = shl nuw i32 %3332, 16, !spirv.Decorations !628
  %3334 = bitcast i32 %3333 to float
  %3335 = fmul reassoc nsz arcp contract float %3331, %3334, !spirv.Decorations !612
  %3336 = fadd reassoc nsz arcp contract float %3335, %.sroa.126.1, !spirv.Decorations !612
  br label %._crit_edge.1.15

._crit_edge.1.15:                                 ; preds = %._crit_edge.15, %3319
  %.sroa.126.2 = phi float [ %3336, %3319 ], [ %.sroa.126.1, %._crit_edge.15 ]
  %3337 = and i1 %184, %3232
  br i1 %3337, label %3338, label %._crit_edge.2.15

3338:                                             ; preds = %._crit_edge.1.15
  %3339 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3339)
  %3340 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3340)
  %3341 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %183, i32* %3341, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3342

3342:                                             ; preds = %3342, %3338
  %3343 = phi i32 [ 0, %3338 ], [ %3348, %3342 ]
  %3344 = zext i32 %3343 to i64
  %3345 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3344
  %3346 = load i32, i32* %3345, align 4, !noalias !635
  %3347 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3344
  store i32 %3346, i32* %3347, align 4, !alias.scope !635
  %3348 = add nuw nsw i32 %3343, 1, !spirv.Decorations !620
  %3349 = icmp eq i32 %3343, 0
  br i1 %3349, label %3342, label %3350, !llvm.loop !638

3350:                                             ; preds = %3342
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3340)
  %3351 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3339)
  %3352 = shl i64 %3351, 32
  %3353 = ashr exact i64 %3352, 32
  %3354 = mul nsw i64 %3353, %const_reg_qword3, !spirv.Decorations !610
  %3355 = ashr i64 %3351, 32
  %3356 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3354, i32 0
  %3357 = getelementptr i16, i16 addrspace(4)* %3356, i64 %3355
  %3358 = addrspacecast i16 addrspace(4)* %3357 to i16 addrspace(1)*
  %3359 = load i16, i16 addrspace(1)* %3358, align 2
  %3360 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3360)
  %3361 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3361)
  %3362 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3362, align 4, !noalias !639
  store i32 %3231, i32* %57, align 4, !noalias !639
  br label %3363

3363:                                             ; preds = %3363, %3350
  %3364 = phi i32 [ 0, %3350 ], [ %3369, %3363 ]
  %3365 = zext i32 %3364 to i64
  %3366 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3365
  %3367 = load i32, i32* %3366, align 4, !noalias !639
  %3368 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3365
  store i32 %3367, i32* %3368, align 4, !alias.scope !639
  %3369 = add nuw nsw i32 %3364, 1, !spirv.Decorations !620
  %3370 = icmp eq i32 %3364, 0
  br i1 %3370, label %3363, label %3371, !llvm.loop !642

3371:                                             ; preds = %3363
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3361)
  %3372 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3360)
  %3373 = shl i64 %3372, 32
  %3374 = ashr exact i64 %3373, 32
  %3375 = mul nsw i64 %3374, %const_reg_qword5, !spirv.Decorations !610
  %3376 = ashr i64 %3372, 32
  %3377 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3375, i32 0
  %3378 = getelementptr i16, i16 addrspace(4)* %3377, i64 %3376
  %3379 = addrspacecast i16 addrspace(4)* %3378 to i16 addrspace(1)*
  %3380 = load i16, i16 addrspace(1)* %3379, align 2
  %3381 = zext i16 %3359 to i32
  %3382 = shl nuw i32 %3381, 16, !spirv.Decorations !628
  %3383 = bitcast i32 %3382 to float
  %3384 = zext i16 %3380 to i32
  %3385 = shl nuw i32 %3384, 16, !spirv.Decorations !628
  %3386 = bitcast i32 %3385 to float
  %3387 = fmul reassoc nsz arcp contract float %3383, %3386, !spirv.Decorations !612
  %3388 = fadd reassoc nsz arcp contract float %3387, %.sroa.190.1, !spirv.Decorations !612
  br label %._crit_edge.2.15

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15, %3371
  %.sroa.190.2 = phi float [ %3388, %3371 ], [ %.sroa.190.1, %._crit_edge.1.15 ]
  %3389 = and i1 %238, %3232
  br i1 %3389, label %3390, label %.preheader.15

3390:                                             ; preds = %._crit_edge.2.15
  %3391 = bitcast %structtype.0* %25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3391)
  %3392 = bitcast [2 x i32]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3392)
  %3393 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 0
  store i32 %237, i32* %3393, align 4, !noalias !635
  store i32 %74, i32* %54, align 4, !noalias !635
  br label %3394

3394:                                             ; preds = %3394, %3390
  %3395 = phi i32 [ 0, %3390 ], [ %3400, %3394 ]
  %3396 = zext i32 %3395 to i64
  %3397 = getelementptr inbounds [2 x i32], [2 x i32]* %21, i64 0, i64 %3396
  %3398 = load i32, i32* %3397, align 4, !noalias !635
  %3399 = getelementptr inbounds [2 x i32], [2 x i32]* %55, i64 0, i64 %3396
  store i32 %3398, i32* %3399, align 4, !alias.scope !635
  %3400 = add nuw nsw i32 %3395, 1, !spirv.Decorations !620
  %3401 = icmp eq i32 %3395, 0
  br i1 %3401, label %3394, label %3402, !llvm.loop !638

3402:                                             ; preds = %3394
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3392)
  %3403 = load i64, i64* %56, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3391)
  %3404 = shl i64 %3403, 32
  %3405 = ashr exact i64 %3404, 32
  %3406 = mul nsw i64 %3405, %const_reg_qword3, !spirv.Decorations !610
  %3407 = ashr i64 %3403, 32
  %3408 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %3406, i32 0
  %3409 = getelementptr i16, i16 addrspace(4)* %3408, i64 %3407
  %3410 = addrspacecast i16 addrspace(4)* %3409 to i16 addrspace(1)*
  %3411 = load i16, i16 addrspace(1)* %3410, align 2
  %3412 = bitcast %structtype.0* %24 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3412)
  %3413 = bitcast [2 x i32]* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3413)
  %3414 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 0
  store i32 %74, i32* %3414, align 4, !noalias !639
  store i32 %3231, i32* %57, align 4, !noalias !639
  br label %3415

3415:                                             ; preds = %3415, %3402
  %3416 = phi i32 [ 0, %3402 ], [ %3421, %3415 ]
  %3417 = zext i32 %3416 to i64
  %3418 = getelementptr inbounds [2 x i32], [2 x i32]* %22, i64 0, i64 %3417
  %3419 = load i32, i32* %3418, align 4, !noalias !639
  %3420 = getelementptr inbounds [2 x i32], [2 x i32]* %58, i64 0, i64 %3417
  store i32 %3419, i32* %3420, align 4, !alias.scope !639
  %3421 = add nuw nsw i32 %3416, 1, !spirv.Decorations !620
  %3422 = icmp eq i32 %3416, 0
  br i1 %3422, label %3415, label %3423, !llvm.loop !642

3423:                                             ; preds = %3415
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3413)
  %3424 = load i64, i64* %59, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3412)
  %3425 = shl i64 %3424, 32
  %3426 = ashr exact i64 %3425, 32
  %3427 = mul nsw i64 %3426, %const_reg_qword5, !spirv.Decorations !610
  %3428 = ashr i64 %3424, 32
  %3429 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %3427, i32 0
  %3430 = getelementptr i16, i16 addrspace(4)* %3429, i64 %3428
  %3431 = addrspacecast i16 addrspace(4)* %3430 to i16 addrspace(1)*
  %3432 = load i16, i16 addrspace(1)* %3431, align 2
  %3433 = zext i16 %3411 to i32
  %3434 = shl nuw i32 %3433, 16, !spirv.Decorations !628
  %3435 = bitcast i32 %3434 to float
  %3436 = zext i16 %3432 to i32
  %3437 = shl nuw i32 %3436, 16, !spirv.Decorations !628
  %3438 = bitcast i32 %3437 to float
  %3439 = fmul reassoc nsz arcp contract float %3435, %3438, !spirv.Decorations !612
  %3440 = fadd reassoc nsz arcp contract float %3439, %.sroa.254.1, !spirv.Decorations !612
  br label %.preheader.15

.preheader.15:                                    ; preds = %._crit_edge.2.15, %3423
  %.sroa.254.2 = phi float [ %3440, %3423 ], [ %.sroa.254.1, %._crit_edge.2.15 ]
  %3441 = add nuw nsw i32 %74, 1, !spirv.Decorations !620
  %3442 = icmp slt i32 %3441, %const_reg_dword2
  br i1 %3442, label %.preheader.preheader, label %.preheader1.preheader, !llvm.loop !643

3443:                                             ; preds = %.preheader1.preheader, %3443
  %3444 = phi i32 [ 0, %.preheader1.preheader ], [ %3449, %3443 ]
  %3445 = zext i32 %3444 to i64
  %3446 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3445
  %3447 = load i32, i32* %3446, align 4, !noalias !632
  %3448 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3445
  store i32 %3447, i32* %3448, align 4, !alias.scope !632
  %3449 = add nuw nsw i32 %3444, 1, !spirv.Decorations !620
  %3450 = icmp eq i32 %3444, 0
  br i1 %3450, label %3443, label %3451, !llvm.loop !644

3451:                                             ; preds = %3443
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3452 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3453 = icmp slt i32 %35, %const_reg_dword
  %3454 = and i1 %3453, %70
  br i1 %3454, label %3455, label %._crit_edge70

3455:                                             ; preds = %3451
  %3456 = shl i64 %3452, 32
  %3457 = ashr exact i64 %3456, 32
  %3458 = ashr i64 %3452, 32
  %3459 = mul nsw i64 %3457, %const_reg_qword9, !spirv.Decorations !610
  %3460 = add nsw i64 %3459, %3458, !spirv.Decorations !610
  %3461 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !612
  br i1 %48, label %3462, label %3472

3462:                                             ; preds = %3455
  %3463 = mul nsw i64 %3457, %const_reg_qword7, !spirv.Decorations !610
  %3464 = getelementptr float, float addrspace(4)* %66, i64 %3463
  %3465 = getelementptr float, float addrspace(4)* %3464, i64 %3458
  %3466 = addrspacecast float addrspace(4)* %3465 to float addrspace(1)*
  %3467 = load float, float addrspace(1)* %3466, align 4
  %3468 = fmul reassoc nsz arcp contract float %3467, %4, !spirv.Decorations !612
  %3469 = fadd reassoc nsz arcp contract float %3461, %3468, !spirv.Decorations !612
  %3470 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3460
  %3471 = addrspacecast float addrspace(4)* %3470 to float addrspace(1)*
  store float %3469, float addrspace(1)* %3471, align 4
  br label %._crit_edge70

3472:                                             ; preds = %3455
  %3473 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3460
  %3474 = addrspacecast float addrspace(4)* %3473 to float addrspace(1)*
  store float %3461, float addrspace(1)* %3474, align 4
  br label %._crit_edge70

._crit_edge70:                                    ; preds = %3451, %3472, %3462
  %3475 = or i32 %35, 1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %41, i32* %60, align 4, !noalias !632
  br label %3476

3476:                                             ; preds = %3476, %._crit_edge70
  %3477 = phi i32 [ 0, %._crit_edge70 ], [ %3482, %3476 ]
  %3478 = zext i32 %3477 to i64
  %3479 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3478
  %3480 = load i32, i32* %3479, align 4, !noalias !632
  %3481 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3478
  store i32 %3480, i32* %3481, align 4, !alias.scope !632
  %3482 = add nuw nsw i32 %3477, 1, !spirv.Decorations !620
  %3483 = icmp eq i32 %3477, 0
  br i1 %3483, label %3476, label %3484, !llvm.loop !644

3484:                                             ; preds = %3476
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3485 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3486 = icmp slt i32 %3475, %const_reg_dword
  %3487 = and i1 %3486, %70
  br i1 %3487, label %3488, label %._crit_edge70.1

3488:                                             ; preds = %3484
  %3489 = shl i64 %3485, 32
  %3490 = ashr exact i64 %3489, 32
  %3491 = ashr i64 %3485, 32
  %3492 = mul nsw i64 %3490, %const_reg_qword9, !spirv.Decorations !610
  %3493 = add nsw i64 %3492, %3491, !spirv.Decorations !610
  %3494 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !612
  br i1 %48, label %3498, label %3495

3495:                                             ; preds = %3488
  %3496 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3493
  %3497 = addrspacecast float addrspace(4)* %3496 to float addrspace(1)*
  store float %3494, float addrspace(1)* %3497, align 4
  br label %._crit_edge70.1

3498:                                             ; preds = %3488
  %3499 = mul nsw i64 %3490, %const_reg_qword7, !spirv.Decorations !610
  %3500 = getelementptr float, float addrspace(4)* %66, i64 %3499
  %3501 = getelementptr float, float addrspace(4)* %3500, i64 %3491
  %3502 = addrspacecast float addrspace(4)* %3501 to float addrspace(1)*
  %3503 = load float, float addrspace(1)* %3502, align 4
  %3504 = fmul reassoc nsz arcp contract float %3503, %4, !spirv.Decorations !612
  %3505 = fadd reassoc nsz arcp contract float %3494, %3504, !spirv.Decorations !612
  %3506 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3493
  %3507 = addrspacecast float addrspace(4)* %3506 to float addrspace(1)*
  store float %3505, float addrspace(1)* %3507, align 4
  br label %._crit_edge70.1

._crit_edge70.1:                                  ; preds = %3484, %3498, %3495
  %3508 = or i32 %35, 2
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %41, i32* %60, align 4, !noalias !632
  br label %3509

3509:                                             ; preds = %3509, %._crit_edge70.1
  %3510 = phi i32 [ 0, %._crit_edge70.1 ], [ %3515, %3509 ]
  %3511 = zext i32 %3510 to i64
  %3512 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3511
  %3513 = load i32, i32* %3512, align 4, !noalias !632
  %3514 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3511
  store i32 %3513, i32* %3514, align 4, !alias.scope !632
  %3515 = add nuw nsw i32 %3510, 1, !spirv.Decorations !620
  %3516 = icmp eq i32 %3510, 0
  br i1 %3516, label %3509, label %3517, !llvm.loop !644

3517:                                             ; preds = %3509
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3518 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3519 = icmp slt i32 %3508, %const_reg_dword
  %3520 = and i1 %3519, %70
  br i1 %3520, label %3521, label %._crit_edge70.2

3521:                                             ; preds = %3517
  %3522 = shl i64 %3518, 32
  %3523 = ashr exact i64 %3522, 32
  %3524 = ashr i64 %3518, 32
  %3525 = mul nsw i64 %3523, %const_reg_qword9, !spirv.Decorations !610
  %3526 = add nsw i64 %3525, %3524, !spirv.Decorations !610
  %3527 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !612
  br i1 %48, label %3531, label %3528

3528:                                             ; preds = %3521
  %3529 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3526
  %3530 = addrspacecast float addrspace(4)* %3529 to float addrspace(1)*
  store float %3527, float addrspace(1)* %3530, align 4
  br label %._crit_edge70.2

3531:                                             ; preds = %3521
  %3532 = mul nsw i64 %3523, %const_reg_qword7, !spirv.Decorations !610
  %3533 = getelementptr float, float addrspace(4)* %66, i64 %3532
  %3534 = getelementptr float, float addrspace(4)* %3533, i64 %3524
  %3535 = addrspacecast float addrspace(4)* %3534 to float addrspace(1)*
  %3536 = load float, float addrspace(1)* %3535, align 4
  %3537 = fmul reassoc nsz arcp contract float %3536, %4, !spirv.Decorations !612
  %3538 = fadd reassoc nsz arcp contract float %3527, %3537, !spirv.Decorations !612
  %3539 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3526
  %3540 = addrspacecast float addrspace(4)* %3539 to float addrspace(1)*
  store float %3538, float addrspace(1)* %3540, align 4
  br label %._crit_edge70.2

._crit_edge70.2:                                  ; preds = %3517, %3531, %3528
  %3541 = or i32 %35, 3
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %41, i32* %60, align 4, !noalias !632
  br label %3542

3542:                                             ; preds = %3542, %._crit_edge70.2
  %3543 = phi i32 [ 0, %._crit_edge70.2 ], [ %3548, %3542 ]
  %3544 = zext i32 %3543 to i64
  %3545 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3544
  %3546 = load i32, i32* %3545, align 4, !noalias !632
  %3547 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3544
  store i32 %3546, i32* %3547, align 4, !alias.scope !632
  %3548 = add nuw nsw i32 %3543, 1, !spirv.Decorations !620
  %3549 = icmp eq i32 %3543, 0
  br i1 %3549, label %3542, label %3550, !llvm.loop !644

3550:                                             ; preds = %3542
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3551 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3552 = icmp slt i32 %3541, %const_reg_dword
  %3553 = and i1 %3552, %70
  br i1 %3553, label %3554, label %.preheader1

3554:                                             ; preds = %3550
  %3555 = shl i64 %3551, 32
  %3556 = ashr exact i64 %3555, 32
  %3557 = ashr i64 %3551, 32
  %3558 = mul nsw i64 %3556, %const_reg_qword9, !spirv.Decorations !610
  %3559 = add nsw i64 %3558, %3557, !spirv.Decorations !610
  %3560 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !612
  br i1 %48, label %3564, label %3561

3561:                                             ; preds = %3554
  %3562 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3559
  %3563 = addrspacecast float addrspace(4)* %3562 to float addrspace(1)*
  store float %3560, float addrspace(1)* %3563, align 4
  br label %.preheader1

3564:                                             ; preds = %3554
  %3565 = mul nsw i64 %3556, %const_reg_qword7, !spirv.Decorations !610
  %3566 = getelementptr float, float addrspace(4)* %66, i64 %3565
  %3567 = getelementptr float, float addrspace(4)* %3566, i64 %3557
  %3568 = addrspacecast float addrspace(4)* %3567 to float addrspace(1)*
  %3569 = load float, float addrspace(1)* %3568, align 4
  %3570 = fmul reassoc nsz arcp contract float %3569, %4, !spirv.Decorations !612
  %3571 = fadd reassoc nsz arcp contract float %3560, %3570, !spirv.Decorations !612
  %3572 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3559
  %3573 = addrspacecast float addrspace(4)* %3572 to float addrspace(1)*
  store float %3571, float addrspace(1)* %3573, align 4
  br label %.preheader1

.preheader1:                                      ; preds = %3550, %3564, %3561
  %3574 = or i32 %41, 1
  %3575 = icmp slt i32 %3574, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %3574, i32* %60, align 4, !noalias !632
  br label %3576

3576:                                             ; preds = %3576, %.preheader1
  %3577 = phi i32 [ 0, %.preheader1 ], [ %3582, %3576 ]
  %3578 = zext i32 %3577 to i64
  %3579 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3578
  %3580 = load i32, i32* %3579, align 4, !noalias !632
  %3581 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3578
  store i32 %3580, i32* %3581, align 4, !alias.scope !632
  %3582 = add nuw nsw i32 %3577, 1, !spirv.Decorations !620
  %3583 = icmp eq i32 %3577, 0
  br i1 %3583, label %3576, label %3584, !llvm.loop !644

3584:                                             ; preds = %3576
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3585 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3586 = and i1 %3453, %3575
  br i1 %3586, label %3587, label %._crit_edge70.176

3587:                                             ; preds = %3584
  %3588 = shl i64 %3585, 32
  %3589 = ashr exact i64 %3588, 32
  %3590 = ashr i64 %3585, 32
  %3591 = mul nsw i64 %3589, %const_reg_qword9, !spirv.Decorations !610
  %3592 = add nsw i64 %3591, %3590, !spirv.Decorations !610
  %3593 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !612
  br i1 %48, label %3597, label %3594

3594:                                             ; preds = %3587
  %3595 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3592
  %3596 = addrspacecast float addrspace(4)* %3595 to float addrspace(1)*
  store float %3593, float addrspace(1)* %3596, align 4
  br label %._crit_edge70.176

3597:                                             ; preds = %3587
  %3598 = mul nsw i64 %3589, %const_reg_qword7, !spirv.Decorations !610
  %3599 = getelementptr float, float addrspace(4)* %66, i64 %3598
  %3600 = getelementptr float, float addrspace(4)* %3599, i64 %3590
  %3601 = addrspacecast float addrspace(4)* %3600 to float addrspace(1)*
  %3602 = load float, float addrspace(1)* %3601, align 4
  %3603 = fmul reassoc nsz arcp contract float %3602, %4, !spirv.Decorations !612
  %3604 = fadd reassoc nsz arcp contract float %3593, %3603, !spirv.Decorations !612
  %3605 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3592
  %3606 = addrspacecast float addrspace(4)* %3605 to float addrspace(1)*
  store float %3604, float addrspace(1)* %3606, align 4
  br label %._crit_edge70.176

._crit_edge70.176:                                ; preds = %3584, %3597, %3594
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %3574, i32* %60, align 4, !noalias !632
  br label %3607

3607:                                             ; preds = %3607, %._crit_edge70.176
  %3608 = phi i32 [ 0, %._crit_edge70.176 ], [ %3613, %3607 ]
  %3609 = zext i32 %3608 to i64
  %3610 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3609
  %3611 = load i32, i32* %3610, align 4, !noalias !632
  %3612 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3609
  store i32 %3611, i32* %3612, align 4, !alias.scope !632
  %3613 = add nuw nsw i32 %3608, 1, !spirv.Decorations !620
  %3614 = icmp eq i32 %3608, 0
  br i1 %3614, label %3607, label %3615, !llvm.loop !644

3615:                                             ; preds = %3607
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3616 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3617 = and i1 %3486, %3575
  br i1 %3617, label %3618, label %._crit_edge70.1.1

3618:                                             ; preds = %3615
  %3619 = shl i64 %3616, 32
  %3620 = ashr exact i64 %3619, 32
  %3621 = ashr i64 %3616, 32
  %3622 = mul nsw i64 %3620, %const_reg_qword9, !spirv.Decorations !610
  %3623 = add nsw i64 %3622, %3621, !spirv.Decorations !610
  %3624 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !612
  br i1 %48, label %3628, label %3625

3625:                                             ; preds = %3618
  %3626 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3623
  %3627 = addrspacecast float addrspace(4)* %3626 to float addrspace(1)*
  store float %3624, float addrspace(1)* %3627, align 4
  br label %._crit_edge70.1.1

3628:                                             ; preds = %3618
  %3629 = mul nsw i64 %3620, %const_reg_qword7, !spirv.Decorations !610
  %3630 = getelementptr float, float addrspace(4)* %66, i64 %3629
  %3631 = getelementptr float, float addrspace(4)* %3630, i64 %3621
  %3632 = addrspacecast float addrspace(4)* %3631 to float addrspace(1)*
  %3633 = load float, float addrspace(1)* %3632, align 4
  %3634 = fmul reassoc nsz arcp contract float %3633, %4, !spirv.Decorations !612
  %3635 = fadd reassoc nsz arcp contract float %3624, %3634, !spirv.Decorations !612
  %3636 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3623
  %3637 = addrspacecast float addrspace(4)* %3636 to float addrspace(1)*
  store float %3635, float addrspace(1)* %3637, align 4
  br label %._crit_edge70.1.1

._crit_edge70.1.1:                                ; preds = %3615, %3628, %3625
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %3574, i32* %60, align 4, !noalias !632
  br label %3638

3638:                                             ; preds = %3638, %._crit_edge70.1.1
  %3639 = phi i32 [ 0, %._crit_edge70.1.1 ], [ %3644, %3638 ]
  %3640 = zext i32 %3639 to i64
  %3641 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3640
  %3642 = load i32, i32* %3641, align 4, !noalias !632
  %3643 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3640
  store i32 %3642, i32* %3643, align 4, !alias.scope !632
  %3644 = add nuw nsw i32 %3639, 1, !spirv.Decorations !620
  %3645 = icmp eq i32 %3639, 0
  br i1 %3645, label %3638, label %3646, !llvm.loop !644

3646:                                             ; preds = %3638
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3647 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3648 = and i1 %3519, %3575
  br i1 %3648, label %3649, label %._crit_edge70.2.1

3649:                                             ; preds = %3646
  %3650 = shl i64 %3647, 32
  %3651 = ashr exact i64 %3650, 32
  %3652 = ashr i64 %3647, 32
  %3653 = mul nsw i64 %3651, %const_reg_qword9, !spirv.Decorations !610
  %3654 = add nsw i64 %3653, %3652, !spirv.Decorations !610
  %3655 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !612
  br i1 %48, label %3659, label %3656

3656:                                             ; preds = %3649
  %3657 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3654
  %3658 = addrspacecast float addrspace(4)* %3657 to float addrspace(1)*
  store float %3655, float addrspace(1)* %3658, align 4
  br label %._crit_edge70.2.1

3659:                                             ; preds = %3649
  %3660 = mul nsw i64 %3651, %const_reg_qword7, !spirv.Decorations !610
  %3661 = getelementptr float, float addrspace(4)* %66, i64 %3660
  %3662 = getelementptr float, float addrspace(4)* %3661, i64 %3652
  %3663 = addrspacecast float addrspace(4)* %3662 to float addrspace(1)*
  %3664 = load float, float addrspace(1)* %3663, align 4
  %3665 = fmul reassoc nsz arcp contract float %3664, %4, !spirv.Decorations !612
  %3666 = fadd reassoc nsz arcp contract float %3655, %3665, !spirv.Decorations !612
  %3667 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3654
  %3668 = addrspacecast float addrspace(4)* %3667 to float addrspace(1)*
  store float %3666, float addrspace(1)* %3668, align 4
  br label %._crit_edge70.2.1

._crit_edge70.2.1:                                ; preds = %3646, %3659, %3656
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %3574, i32* %60, align 4, !noalias !632
  br label %3669

3669:                                             ; preds = %3669, %._crit_edge70.2.1
  %3670 = phi i32 [ 0, %._crit_edge70.2.1 ], [ %3675, %3669 ]
  %3671 = zext i32 %3670 to i64
  %3672 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3671
  %3673 = load i32, i32* %3672, align 4, !noalias !632
  %3674 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3671
  store i32 %3673, i32* %3674, align 4, !alias.scope !632
  %3675 = add nuw nsw i32 %3670, 1, !spirv.Decorations !620
  %3676 = icmp eq i32 %3670, 0
  br i1 %3676, label %3669, label %3677, !llvm.loop !644

3677:                                             ; preds = %3669
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3678 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3679 = and i1 %3552, %3575
  br i1 %3679, label %3680, label %.preheader1.1

3680:                                             ; preds = %3677
  %3681 = shl i64 %3678, 32
  %3682 = ashr exact i64 %3681, 32
  %3683 = ashr i64 %3678, 32
  %3684 = mul nsw i64 %3682, %const_reg_qword9, !spirv.Decorations !610
  %3685 = add nsw i64 %3684, %3683, !spirv.Decorations !610
  %3686 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !612
  br i1 %48, label %3690, label %3687

3687:                                             ; preds = %3680
  %3688 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3685
  %3689 = addrspacecast float addrspace(4)* %3688 to float addrspace(1)*
  store float %3686, float addrspace(1)* %3689, align 4
  br label %.preheader1.1

3690:                                             ; preds = %3680
  %3691 = mul nsw i64 %3682, %const_reg_qword7, !spirv.Decorations !610
  %3692 = getelementptr float, float addrspace(4)* %66, i64 %3691
  %3693 = getelementptr float, float addrspace(4)* %3692, i64 %3683
  %3694 = addrspacecast float addrspace(4)* %3693 to float addrspace(1)*
  %3695 = load float, float addrspace(1)* %3694, align 4
  %3696 = fmul reassoc nsz arcp contract float %3695, %4, !spirv.Decorations !612
  %3697 = fadd reassoc nsz arcp contract float %3686, %3696, !spirv.Decorations !612
  %3698 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3685
  %3699 = addrspacecast float addrspace(4)* %3698 to float addrspace(1)*
  store float %3697, float addrspace(1)* %3699, align 4
  br label %.preheader1.1

.preheader1.1:                                    ; preds = %3677, %3690, %3687
  %3700 = or i32 %41, 2
  %3701 = icmp slt i32 %3700, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %3700, i32* %60, align 4, !noalias !632
  br label %3702

3702:                                             ; preds = %3702, %.preheader1.1
  %3703 = phi i32 [ 0, %.preheader1.1 ], [ %3708, %3702 ]
  %3704 = zext i32 %3703 to i64
  %3705 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3704
  %3706 = load i32, i32* %3705, align 4, !noalias !632
  %3707 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3704
  store i32 %3706, i32* %3707, align 4, !alias.scope !632
  %3708 = add nuw nsw i32 %3703, 1, !spirv.Decorations !620
  %3709 = icmp eq i32 %3703, 0
  br i1 %3709, label %3702, label %3710, !llvm.loop !644

3710:                                             ; preds = %3702
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3711 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3712 = and i1 %3453, %3701
  br i1 %3712, label %3713, label %._crit_edge70.277

3713:                                             ; preds = %3710
  %3714 = shl i64 %3711, 32
  %3715 = ashr exact i64 %3714, 32
  %3716 = ashr i64 %3711, 32
  %3717 = mul nsw i64 %3715, %const_reg_qword9, !spirv.Decorations !610
  %3718 = add nsw i64 %3717, %3716, !spirv.Decorations !610
  %3719 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !612
  br i1 %48, label %3723, label %3720

3720:                                             ; preds = %3713
  %3721 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3718
  %3722 = addrspacecast float addrspace(4)* %3721 to float addrspace(1)*
  store float %3719, float addrspace(1)* %3722, align 4
  br label %._crit_edge70.277

3723:                                             ; preds = %3713
  %3724 = mul nsw i64 %3715, %const_reg_qword7, !spirv.Decorations !610
  %3725 = getelementptr float, float addrspace(4)* %66, i64 %3724
  %3726 = getelementptr float, float addrspace(4)* %3725, i64 %3716
  %3727 = addrspacecast float addrspace(4)* %3726 to float addrspace(1)*
  %3728 = load float, float addrspace(1)* %3727, align 4
  %3729 = fmul reassoc nsz arcp contract float %3728, %4, !spirv.Decorations !612
  %3730 = fadd reassoc nsz arcp contract float %3719, %3729, !spirv.Decorations !612
  %3731 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3718
  %3732 = addrspacecast float addrspace(4)* %3731 to float addrspace(1)*
  store float %3730, float addrspace(1)* %3732, align 4
  br label %._crit_edge70.277

._crit_edge70.277:                                ; preds = %3710, %3723, %3720
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %3700, i32* %60, align 4, !noalias !632
  br label %3733

3733:                                             ; preds = %3733, %._crit_edge70.277
  %3734 = phi i32 [ 0, %._crit_edge70.277 ], [ %3739, %3733 ]
  %3735 = zext i32 %3734 to i64
  %3736 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3735
  %3737 = load i32, i32* %3736, align 4, !noalias !632
  %3738 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3735
  store i32 %3737, i32* %3738, align 4, !alias.scope !632
  %3739 = add nuw nsw i32 %3734, 1, !spirv.Decorations !620
  %3740 = icmp eq i32 %3734, 0
  br i1 %3740, label %3733, label %3741, !llvm.loop !644

3741:                                             ; preds = %3733
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3742 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3743 = and i1 %3486, %3701
  br i1 %3743, label %3744, label %._crit_edge70.1.2

3744:                                             ; preds = %3741
  %3745 = shl i64 %3742, 32
  %3746 = ashr exact i64 %3745, 32
  %3747 = ashr i64 %3742, 32
  %3748 = mul nsw i64 %3746, %const_reg_qword9, !spirv.Decorations !610
  %3749 = add nsw i64 %3748, %3747, !spirv.Decorations !610
  %3750 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !612
  br i1 %48, label %3754, label %3751

3751:                                             ; preds = %3744
  %3752 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3749
  %3753 = addrspacecast float addrspace(4)* %3752 to float addrspace(1)*
  store float %3750, float addrspace(1)* %3753, align 4
  br label %._crit_edge70.1.2

3754:                                             ; preds = %3744
  %3755 = mul nsw i64 %3746, %const_reg_qword7, !spirv.Decorations !610
  %3756 = getelementptr float, float addrspace(4)* %66, i64 %3755
  %3757 = getelementptr float, float addrspace(4)* %3756, i64 %3747
  %3758 = addrspacecast float addrspace(4)* %3757 to float addrspace(1)*
  %3759 = load float, float addrspace(1)* %3758, align 4
  %3760 = fmul reassoc nsz arcp contract float %3759, %4, !spirv.Decorations !612
  %3761 = fadd reassoc nsz arcp contract float %3750, %3760, !spirv.Decorations !612
  %3762 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3749
  %3763 = addrspacecast float addrspace(4)* %3762 to float addrspace(1)*
  store float %3761, float addrspace(1)* %3763, align 4
  br label %._crit_edge70.1.2

._crit_edge70.1.2:                                ; preds = %3741, %3754, %3751
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %3700, i32* %60, align 4, !noalias !632
  br label %3764

3764:                                             ; preds = %3764, %._crit_edge70.1.2
  %3765 = phi i32 [ 0, %._crit_edge70.1.2 ], [ %3770, %3764 ]
  %3766 = zext i32 %3765 to i64
  %3767 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3766
  %3768 = load i32, i32* %3767, align 4, !noalias !632
  %3769 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3766
  store i32 %3768, i32* %3769, align 4, !alias.scope !632
  %3770 = add nuw nsw i32 %3765, 1, !spirv.Decorations !620
  %3771 = icmp eq i32 %3765, 0
  br i1 %3771, label %3764, label %3772, !llvm.loop !644

3772:                                             ; preds = %3764
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3773 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3774 = and i1 %3519, %3701
  br i1 %3774, label %3775, label %._crit_edge70.2.2

3775:                                             ; preds = %3772
  %3776 = shl i64 %3773, 32
  %3777 = ashr exact i64 %3776, 32
  %3778 = ashr i64 %3773, 32
  %3779 = mul nsw i64 %3777, %const_reg_qword9, !spirv.Decorations !610
  %3780 = add nsw i64 %3779, %3778, !spirv.Decorations !610
  %3781 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !612
  br i1 %48, label %3785, label %3782

3782:                                             ; preds = %3775
  %3783 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3780
  %3784 = addrspacecast float addrspace(4)* %3783 to float addrspace(1)*
  store float %3781, float addrspace(1)* %3784, align 4
  br label %._crit_edge70.2.2

3785:                                             ; preds = %3775
  %3786 = mul nsw i64 %3777, %const_reg_qword7, !spirv.Decorations !610
  %3787 = getelementptr float, float addrspace(4)* %66, i64 %3786
  %3788 = getelementptr float, float addrspace(4)* %3787, i64 %3778
  %3789 = addrspacecast float addrspace(4)* %3788 to float addrspace(1)*
  %3790 = load float, float addrspace(1)* %3789, align 4
  %3791 = fmul reassoc nsz arcp contract float %3790, %4, !spirv.Decorations !612
  %3792 = fadd reassoc nsz arcp contract float %3781, %3791, !spirv.Decorations !612
  %3793 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3780
  %3794 = addrspacecast float addrspace(4)* %3793 to float addrspace(1)*
  store float %3792, float addrspace(1)* %3794, align 4
  br label %._crit_edge70.2.2

._crit_edge70.2.2:                                ; preds = %3772, %3785, %3782
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %3700, i32* %60, align 4, !noalias !632
  br label %3795

3795:                                             ; preds = %3795, %._crit_edge70.2.2
  %3796 = phi i32 [ 0, %._crit_edge70.2.2 ], [ %3801, %3795 ]
  %3797 = zext i32 %3796 to i64
  %3798 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3797
  %3799 = load i32, i32* %3798, align 4, !noalias !632
  %3800 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3797
  store i32 %3799, i32* %3800, align 4, !alias.scope !632
  %3801 = add nuw nsw i32 %3796, 1, !spirv.Decorations !620
  %3802 = icmp eq i32 %3796, 0
  br i1 %3802, label %3795, label %3803, !llvm.loop !644

3803:                                             ; preds = %3795
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3804 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3805 = and i1 %3552, %3701
  br i1 %3805, label %3806, label %.preheader1.2

3806:                                             ; preds = %3803
  %3807 = shl i64 %3804, 32
  %3808 = ashr exact i64 %3807, 32
  %3809 = ashr i64 %3804, 32
  %3810 = mul nsw i64 %3808, %const_reg_qword9, !spirv.Decorations !610
  %3811 = add nsw i64 %3810, %3809, !spirv.Decorations !610
  %3812 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !612
  br i1 %48, label %3816, label %3813

3813:                                             ; preds = %3806
  %3814 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3811
  %3815 = addrspacecast float addrspace(4)* %3814 to float addrspace(1)*
  store float %3812, float addrspace(1)* %3815, align 4
  br label %.preheader1.2

3816:                                             ; preds = %3806
  %3817 = mul nsw i64 %3808, %const_reg_qword7, !spirv.Decorations !610
  %3818 = getelementptr float, float addrspace(4)* %66, i64 %3817
  %3819 = getelementptr float, float addrspace(4)* %3818, i64 %3809
  %3820 = addrspacecast float addrspace(4)* %3819 to float addrspace(1)*
  %3821 = load float, float addrspace(1)* %3820, align 4
  %3822 = fmul reassoc nsz arcp contract float %3821, %4, !spirv.Decorations !612
  %3823 = fadd reassoc nsz arcp contract float %3812, %3822, !spirv.Decorations !612
  %3824 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3811
  %3825 = addrspacecast float addrspace(4)* %3824 to float addrspace(1)*
  store float %3823, float addrspace(1)* %3825, align 4
  br label %.preheader1.2

.preheader1.2:                                    ; preds = %3803, %3816, %3813
  %3826 = or i32 %41, 3
  %3827 = icmp slt i32 %3826, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %3826, i32* %60, align 4, !noalias !632
  br label %3828

3828:                                             ; preds = %3828, %.preheader1.2
  %3829 = phi i32 [ 0, %.preheader1.2 ], [ %3834, %3828 ]
  %3830 = zext i32 %3829 to i64
  %3831 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3830
  %3832 = load i32, i32* %3831, align 4, !noalias !632
  %3833 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3830
  store i32 %3832, i32* %3833, align 4, !alias.scope !632
  %3834 = add nuw nsw i32 %3829, 1, !spirv.Decorations !620
  %3835 = icmp eq i32 %3829, 0
  br i1 %3835, label %3828, label %3836, !llvm.loop !644

3836:                                             ; preds = %3828
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3837 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3838 = and i1 %3453, %3827
  br i1 %3838, label %3839, label %._crit_edge70.378

3839:                                             ; preds = %3836
  %3840 = shl i64 %3837, 32
  %3841 = ashr exact i64 %3840, 32
  %3842 = ashr i64 %3837, 32
  %3843 = mul nsw i64 %3841, %const_reg_qword9, !spirv.Decorations !610
  %3844 = add nsw i64 %3843, %3842, !spirv.Decorations !610
  %3845 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !612
  br i1 %48, label %3849, label %3846

3846:                                             ; preds = %3839
  %3847 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3844
  %3848 = addrspacecast float addrspace(4)* %3847 to float addrspace(1)*
  store float %3845, float addrspace(1)* %3848, align 4
  br label %._crit_edge70.378

3849:                                             ; preds = %3839
  %3850 = mul nsw i64 %3841, %const_reg_qword7, !spirv.Decorations !610
  %3851 = getelementptr float, float addrspace(4)* %66, i64 %3850
  %3852 = getelementptr float, float addrspace(4)* %3851, i64 %3842
  %3853 = addrspacecast float addrspace(4)* %3852 to float addrspace(1)*
  %3854 = load float, float addrspace(1)* %3853, align 4
  %3855 = fmul reassoc nsz arcp contract float %3854, %4, !spirv.Decorations !612
  %3856 = fadd reassoc nsz arcp contract float %3845, %3855, !spirv.Decorations !612
  %3857 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3844
  %3858 = addrspacecast float addrspace(4)* %3857 to float addrspace(1)*
  store float %3856, float addrspace(1)* %3858, align 4
  br label %._crit_edge70.378

._crit_edge70.378:                                ; preds = %3836, %3849, %3846
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %3826, i32* %60, align 4, !noalias !632
  br label %3859

3859:                                             ; preds = %3859, %._crit_edge70.378
  %3860 = phi i32 [ 0, %._crit_edge70.378 ], [ %3865, %3859 ]
  %3861 = zext i32 %3860 to i64
  %3862 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3861
  %3863 = load i32, i32* %3862, align 4, !noalias !632
  %3864 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3861
  store i32 %3863, i32* %3864, align 4, !alias.scope !632
  %3865 = add nuw nsw i32 %3860, 1, !spirv.Decorations !620
  %3866 = icmp eq i32 %3860, 0
  br i1 %3866, label %3859, label %3867, !llvm.loop !644

3867:                                             ; preds = %3859
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3868 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3869 = and i1 %3486, %3827
  br i1 %3869, label %3870, label %._crit_edge70.1.3

3870:                                             ; preds = %3867
  %3871 = shl i64 %3868, 32
  %3872 = ashr exact i64 %3871, 32
  %3873 = ashr i64 %3868, 32
  %3874 = mul nsw i64 %3872, %const_reg_qword9, !spirv.Decorations !610
  %3875 = add nsw i64 %3874, %3873, !spirv.Decorations !610
  %3876 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !612
  br i1 %48, label %3880, label %3877

3877:                                             ; preds = %3870
  %3878 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3875
  %3879 = addrspacecast float addrspace(4)* %3878 to float addrspace(1)*
  store float %3876, float addrspace(1)* %3879, align 4
  br label %._crit_edge70.1.3

3880:                                             ; preds = %3870
  %3881 = mul nsw i64 %3872, %const_reg_qword7, !spirv.Decorations !610
  %3882 = getelementptr float, float addrspace(4)* %66, i64 %3881
  %3883 = getelementptr float, float addrspace(4)* %3882, i64 %3873
  %3884 = addrspacecast float addrspace(4)* %3883 to float addrspace(1)*
  %3885 = load float, float addrspace(1)* %3884, align 4
  %3886 = fmul reassoc nsz arcp contract float %3885, %4, !spirv.Decorations !612
  %3887 = fadd reassoc nsz arcp contract float %3876, %3886, !spirv.Decorations !612
  %3888 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3875
  %3889 = addrspacecast float addrspace(4)* %3888 to float addrspace(1)*
  store float %3887, float addrspace(1)* %3889, align 4
  br label %._crit_edge70.1.3

._crit_edge70.1.3:                                ; preds = %3867, %3880, %3877
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %3826, i32* %60, align 4, !noalias !632
  br label %3890

3890:                                             ; preds = %3890, %._crit_edge70.1.3
  %3891 = phi i32 [ 0, %._crit_edge70.1.3 ], [ %3896, %3890 ]
  %3892 = zext i32 %3891 to i64
  %3893 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3892
  %3894 = load i32, i32* %3893, align 4, !noalias !632
  %3895 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3892
  store i32 %3894, i32* %3895, align 4, !alias.scope !632
  %3896 = add nuw nsw i32 %3891, 1, !spirv.Decorations !620
  %3897 = icmp eq i32 %3891, 0
  br i1 %3897, label %3890, label %3898, !llvm.loop !644

3898:                                             ; preds = %3890
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3899 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3900 = and i1 %3519, %3827
  br i1 %3900, label %3901, label %._crit_edge70.2.3

3901:                                             ; preds = %3898
  %3902 = shl i64 %3899, 32
  %3903 = ashr exact i64 %3902, 32
  %3904 = ashr i64 %3899, 32
  %3905 = mul nsw i64 %3903, %const_reg_qword9, !spirv.Decorations !610
  %3906 = add nsw i64 %3905, %3904, !spirv.Decorations !610
  %3907 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !612
  br i1 %48, label %3911, label %3908

3908:                                             ; preds = %3901
  %3909 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3906
  %3910 = addrspacecast float addrspace(4)* %3909 to float addrspace(1)*
  store float %3907, float addrspace(1)* %3910, align 4
  br label %._crit_edge70.2.3

3911:                                             ; preds = %3901
  %3912 = mul nsw i64 %3903, %const_reg_qword7, !spirv.Decorations !610
  %3913 = getelementptr float, float addrspace(4)* %66, i64 %3912
  %3914 = getelementptr float, float addrspace(4)* %3913, i64 %3904
  %3915 = addrspacecast float addrspace(4)* %3914 to float addrspace(1)*
  %3916 = load float, float addrspace(1)* %3915, align 4
  %3917 = fmul reassoc nsz arcp contract float %3916, %4, !spirv.Decorations !612
  %3918 = fadd reassoc nsz arcp contract float %3907, %3917, !spirv.Decorations !612
  %3919 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3906
  %3920 = addrspacecast float addrspace(4)* %3919 to float addrspace(1)*
  store float %3918, float addrspace(1)* %3920, align 4
  br label %._crit_edge70.2.3

._crit_edge70.2.3:                                ; preds = %3898, %3911, %3908
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %3826, i32* %60, align 4, !noalias !632
  br label %3921

3921:                                             ; preds = %3921, %._crit_edge70.2.3
  %3922 = phi i32 [ 0, %._crit_edge70.2.3 ], [ %3927, %3921 ]
  %3923 = zext i32 %3922 to i64
  %3924 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3923
  %3925 = load i32, i32* %3924, align 4, !noalias !632
  %3926 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3923
  store i32 %3925, i32* %3926, align 4, !alias.scope !632
  %3927 = add nuw nsw i32 %3922, 1, !spirv.Decorations !620
  %3928 = icmp eq i32 %3922, 0
  br i1 %3928, label %3921, label %3929, !llvm.loop !644

3929:                                             ; preds = %3921
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3930 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3931 = and i1 %3552, %3827
  br i1 %3931, label %3932, label %.preheader1.3

3932:                                             ; preds = %3929
  %3933 = shl i64 %3930, 32
  %3934 = ashr exact i64 %3933, 32
  %3935 = ashr i64 %3930, 32
  %3936 = mul nsw i64 %3934, %const_reg_qword9, !spirv.Decorations !610
  %3937 = add nsw i64 %3936, %3935, !spirv.Decorations !610
  %3938 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !612
  br i1 %48, label %3942, label %3939

3939:                                             ; preds = %3932
  %3940 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3937
  %3941 = addrspacecast float addrspace(4)* %3940 to float addrspace(1)*
  store float %3938, float addrspace(1)* %3941, align 4
  br label %.preheader1.3

3942:                                             ; preds = %3932
  %3943 = mul nsw i64 %3934, %const_reg_qword7, !spirv.Decorations !610
  %3944 = getelementptr float, float addrspace(4)* %66, i64 %3943
  %3945 = getelementptr float, float addrspace(4)* %3944, i64 %3935
  %3946 = addrspacecast float addrspace(4)* %3945 to float addrspace(1)*
  %3947 = load float, float addrspace(1)* %3946, align 4
  %3948 = fmul reassoc nsz arcp contract float %3947, %4, !spirv.Decorations !612
  %3949 = fadd reassoc nsz arcp contract float %3938, %3948, !spirv.Decorations !612
  %3950 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3937
  %3951 = addrspacecast float addrspace(4)* %3950 to float addrspace(1)*
  store float %3949, float addrspace(1)* %3951, align 4
  br label %.preheader1.3

.preheader1.3:                                    ; preds = %3929, %3942, %3939
  %3952 = or i32 %41, 4
  %3953 = icmp slt i32 %3952, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %3952, i32* %60, align 4, !noalias !632
  br label %3954

3954:                                             ; preds = %3954, %.preheader1.3
  %3955 = phi i32 [ 0, %.preheader1.3 ], [ %3960, %3954 ]
  %3956 = zext i32 %3955 to i64
  %3957 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3956
  %3958 = load i32, i32* %3957, align 4, !noalias !632
  %3959 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3956
  store i32 %3958, i32* %3959, align 4, !alias.scope !632
  %3960 = add nuw nsw i32 %3955, 1, !spirv.Decorations !620
  %3961 = icmp eq i32 %3955, 0
  br i1 %3961, label %3954, label %3962, !llvm.loop !644

3962:                                             ; preds = %3954
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3963 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3964 = and i1 %3453, %3953
  br i1 %3964, label %3965, label %._crit_edge70.4

3965:                                             ; preds = %3962
  %3966 = shl i64 %3963, 32
  %3967 = ashr exact i64 %3966, 32
  %3968 = ashr i64 %3963, 32
  %3969 = mul nsw i64 %3967, %const_reg_qword9, !spirv.Decorations !610
  %3970 = add nsw i64 %3969, %3968, !spirv.Decorations !610
  %3971 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !612
  br i1 %48, label %3975, label %3972

3972:                                             ; preds = %3965
  %3973 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3970
  %3974 = addrspacecast float addrspace(4)* %3973 to float addrspace(1)*
  store float %3971, float addrspace(1)* %3974, align 4
  br label %._crit_edge70.4

3975:                                             ; preds = %3965
  %3976 = mul nsw i64 %3967, %const_reg_qword7, !spirv.Decorations !610
  %3977 = getelementptr float, float addrspace(4)* %66, i64 %3976
  %3978 = getelementptr float, float addrspace(4)* %3977, i64 %3968
  %3979 = addrspacecast float addrspace(4)* %3978 to float addrspace(1)*
  %3980 = load float, float addrspace(1)* %3979, align 4
  %3981 = fmul reassoc nsz arcp contract float %3980, %4, !spirv.Decorations !612
  %3982 = fadd reassoc nsz arcp contract float %3971, %3981, !spirv.Decorations !612
  %3983 = getelementptr inbounds float, float addrspace(4)* %65, i64 %3970
  %3984 = addrspacecast float addrspace(4)* %3983 to float addrspace(1)*
  store float %3982, float addrspace(1)* %3984, align 4
  br label %._crit_edge70.4

._crit_edge70.4:                                  ; preds = %3962, %3975, %3972
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %3952, i32* %60, align 4, !noalias !632
  br label %3985

3985:                                             ; preds = %3985, %._crit_edge70.4
  %3986 = phi i32 [ 0, %._crit_edge70.4 ], [ %3991, %3985 ]
  %3987 = zext i32 %3986 to i64
  %3988 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %3987
  %3989 = load i32, i32* %3988, align 4, !noalias !632
  %3990 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %3987
  store i32 %3989, i32* %3990, align 4, !alias.scope !632
  %3991 = add nuw nsw i32 %3986, 1, !spirv.Decorations !620
  %3992 = icmp eq i32 %3986, 0
  br i1 %3992, label %3985, label %3993, !llvm.loop !644

3993:                                             ; preds = %3985
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %3994 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %3995 = and i1 %3486, %3953
  br i1 %3995, label %3996, label %._crit_edge70.1.4

3996:                                             ; preds = %3993
  %3997 = shl i64 %3994, 32
  %3998 = ashr exact i64 %3997, 32
  %3999 = ashr i64 %3994, 32
  %4000 = mul nsw i64 %3998, %const_reg_qword9, !spirv.Decorations !610
  %4001 = add nsw i64 %4000, %3999, !spirv.Decorations !610
  %4002 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !612
  br i1 %48, label %4006, label %4003

4003:                                             ; preds = %3996
  %4004 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4001
  %4005 = addrspacecast float addrspace(4)* %4004 to float addrspace(1)*
  store float %4002, float addrspace(1)* %4005, align 4
  br label %._crit_edge70.1.4

4006:                                             ; preds = %3996
  %4007 = mul nsw i64 %3998, %const_reg_qword7, !spirv.Decorations !610
  %4008 = getelementptr float, float addrspace(4)* %66, i64 %4007
  %4009 = getelementptr float, float addrspace(4)* %4008, i64 %3999
  %4010 = addrspacecast float addrspace(4)* %4009 to float addrspace(1)*
  %4011 = load float, float addrspace(1)* %4010, align 4
  %4012 = fmul reassoc nsz arcp contract float %4011, %4, !spirv.Decorations !612
  %4013 = fadd reassoc nsz arcp contract float %4002, %4012, !spirv.Decorations !612
  %4014 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4001
  %4015 = addrspacecast float addrspace(4)* %4014 to float addrspace(1)*
  store float %4013, float addrspace(1)* %4015, align 4
  br label %._crit_edge70.1.4

._crit_edge70.1.4:                                ; preds = %3993, %4006, %4003
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %3952, i32* %60, align 4, !noalias !632
  br label %4016

4016:                                             ; preds = %4016, %._crit_edge70.1.4
  %4017 = phi i32 [ 0, %._crit_edge70.1.4 ], [ %4022, %4016 ]
  %4018 = zext i32 %4017 to i64
  %4019 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4018
  %4020 = load i32, i32* %4019, align 4, !noalias !632
  %4021 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4018
  store i32 %4020, i32* %4021, align 4, !alias.scope !632
  %4022 = add nuw nsw i32 %4017, 1, !spirv.Decorations !620
  %4023 = icmp eq i32 %4017, 0
  br i1 %4023, label %4016, label %4024, !llvm.loop !644

4024:                                             ; preds = %4016
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4025 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4026 = and i1 %3519, %3953
  br i1 %4026, label %4027, label %._crit_edge70.2.4

4027:                                             ; preds = %4024
  %4028 = shl i64 %4025, 32
  %4029 = ashr exact i64 %4028, 32
  %4030 = ashr i64 %4025, 32
  %4031 = mul nsw i64 %4029, %const_reg_qword9, !spirv.Decorations !610
  %4032 = add nsw i64 %4031, %4030, !spirv.Decorations !610
  %4033 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !612
  br i1 %48, label %4037, label %4034

4034:                                             ; preds = %4027
  %4035 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4032
  %4036 = addrspacecast float addrspace(4)* %4035 to float addrspace(1)*
  store float %4033, float addrspace(1)* %4036, align 4
  br label %._crit_edge70.2.4

4037:                                             ; preds = %4027
  %4038 = mul nsw i64 %4029, %const_reg_qword7, !spirv.Decorations !610
  %4039 = getelementptr float, float addrspace(4)* %66, i64 %4038
  %4040 = getelementptr float, float addrspace(4)* %4039, i64 %4030
  %4041 = addrspacecast float addrspace(4)* %4040 to float addrspace(1)*
  %4042 = load float, float addrspace(1)* %4041, align 4
  %4043 = fmul reassoc nsz arcp contract float %4042, %4, !spirv.Decorations !612
  %4044 = fadd reassoc nsz arcp contract float %4033, %4043, !spirv.Decorations !612
  %4045 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4032
  %4046 = addrspacecast float addrspace(4)* %4045 to float addrspace(1)*
  store float %4044, float addrspace(1)* %4046, align 4
  br label %._crit_edge70.2.4

._crit_edge70.2.4:                                ; preds = %4024, %4037, %4034
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %3952, i32* %60, align 4, !noalias !632
  br label %4047

4047:                                             ; preds = %4047, %._crit_edge70.2.4
  %4048 = phi i32 [ 0, %._crit_edge70.2.4 ], [ %4053, %4047 ]
  %4049 = zext i32 %4048 to i64
  %4050 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4049
  %4051 = load i32, i32* %4050, align 4, !noalias !632
  %4052 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4049
  store i32 %4051, i32* %4052, align 4, !alias.scope !632
  %4053 = add nuw nsw i32 %4048, 1, !spirv.Decorations !620
  %4054 = icmp eq i32 %4048, 0
  br i1 %4054, label %4047, label %4055, !llvm.loop !644

4055:                                             ; preds = %4047
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4056 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4057 = and i1 %3552, %3953
  br i1 %4057, label %4058, label %.preheader1.4

4058:                                             ; preds = %4055
  %4059 = shl i64 %4056, 32
  %4060 = ashr exact i64 %4059, 32
  %4061 = ashr i64 %4056, 32
  %4062 = mul nsw i64 %4060, %const_reg_qword9, !spirv.Decorations !610
  %4063 = add nsw i64 %4062, %4061, !spirv.Decorations !610
  %4064 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !612
  br i1 %48, label %4068, label %4065

4065:                                             ; preds = %4058
  %4066 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4063
  %4067 = addrspacecast float addrspace(4)* %4066 to float addrspace(1)*
  store float %4064, float addrspace(1)* %4067, align 4
  br label %.preheader1.4

4068:                                             ; preds = %4058
  %4069 = mul nsw i64 %4060, %const_reg_qword7, !spirv.Decorations !610
  %4070 = getelementptr float, float addrspace(4)* %66, i64 %4069
  %4071 = getelementptr float, float addrspace(4)* %4070, i64 %4061
  %4072 = addrspacecast float addrspace(4)* %4071 to float addrspace(1)*
  %4073 = load float, float addrspace(1)* %4072, align 4
  %4074 = fmul reassoc nsz arcp contract float %4073, %4, !spirv.Decorations !612
  %4075 = fadd reassoc nsz arcp contract float %4064, %4074, !spirv.Decorations !612
  %4076 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4063
  %4077 = addrspacecast float addrspace(4)* %4076 to float addrspace(1)*
  store float %4075, float addrspace(1)* %4077, align 4
  br label %.preheader1.4

.preheader1.4:                                    ; preds = %4055, %4068, %4065
  %4078 = or i32 %41, 5
  %4079 = icmp slt i32 %4078, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4078, i32* %60, align 4, !noalias !632
  br label %4080

4080:                                             ; preds = %4080, %.preheader1.4
  %4081 = phi i32 [ 0, %.preheader1.4 ], [ %4086, %4080 ]
  %4082 = zext i32 %4081 to i64
  %4083 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4082
  %4084 = load i32, i32* %4083, align 4, !noalias !632
  %4085 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4082
  store i32 %4084, i32* %4085, align 4, !alias.scope !632
  %4086 = add nuw nsw i32 %4081, 1, !spirv.Decorations !620
  %4087 = icmp eq i32 %4081, 0
  br i1 %4087, label %4080, label %4088, !llvm.loop !644

4088:                                             ; preds = %4080
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4089 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4090 = and i1 %3453, %4079
  br i1 %4090, label %4091, label %._crit_edge70.5

4091:                                             ; preds = %4088
  %4092 = shl i64 %4089, 32
  %4093 = ashr exact i64 %4092, 32
  %4094 = ashr i64 %4089, 32
  %4095 = mul nsw i64 %4093, %const_reg_qword9, !spirv.Decorations !610
  %4096 = add nsw i64 %4095, %4094, !spirv.Decorations !610
  %4097 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !612
  br i1 %48, label %4101, label %4098

4098:                                             ; preds = %4091
  %4099 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4096
  %4100 = addrspacecast float addrspace(4)* %4099 to float addrspace(1)*
  store float %4097, float addrspace(1)* %4100, align 4
  br label %._crit_edge70.5

4101:                                             ; preds = %4091
  %4102 = mul nsw i64 %4093, %const_reg_qword7, !spirv.Decorations !610
  %4103 = getelementptr float, float addrspace(4)* %66, i64 %4102
  %4104 = getelementptr float, float addrspace(4)* %4103, i64 %4094
  %4105 = addrspacecast float addrspace(4)* %4104 to float addrspace(1)*
  %4106 = load float, float addrspace(1)* %4105, align 4
  %4107 = fmul reassoc nsz arcp contract float %4106, %4, !spirv.Decorations !612
  %4108 = fadd reassoc nsz arcp contract float %4097, %4107, !spirv.Decorations !612
  %4109 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4096
  %4110 = addrspacecast float addrspace(4)* %4109 to float addrspace(1)*
  store float %4108, float addrspace(1)* %4110, align 4
  br label %._crit_edge70.5

._crit_edge70.5:                                  ; preds = %4088, %4101, %4098
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4078, i32* %60, align 4, !noalias !632
  br label %4111

4111:                                             ; preds = %4111, %._crit_edge70.5
  %4112 = phi i32 [ 0, %._crit_edge70.5 ], [ %4117, %4111 ]
  %4113 = zext i32 %4112 to i64
  %4114 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4113
  %4115 = load i32, i32* %4114, align 4, !noalias !632
  %4116 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4113
  store i32 %4115, i32* %4116, align 4, !alias.scope !632
  %4117 = add nuw nsw i32 %4112, 1, !spirv.Decorations !620
  %4118 = icmp eq i32 %4112, 0
  br i1 %4118, label %4111, label %4119, !llvm.loop !644

4119:                                             ; preds = %4111
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4120 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4121 = and i1 %3486, %4079
  br i1 %4121, label %4122, label %._crit_edge70.1.5

4122:                                             ; preds = %4119
  %4123 = shl i64 %4120, 32
  %4124 = ashr exact i64 %4123, 32
  %4125 = ashr i64 %4120, 32
  %4126 = mul nsw i64 %4124, %const_reg_qword9, !spirv.Decorations !610
  %4127 = add nsw i64 %4126, %4125, !spirv.Decorations !610
  %4128 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !612
  br i1 %48, label %4132, label %4129

4129:                                             ; preds = %4122
  %4130 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4127
  %4131 = addrspacecast float addrspace(4)* %4130 to float addrspace(1)*
  store float %4128, float addrspace(1)* %4131, align 4
  br label %._crit_edge70.1.5

4132:                                             ; preds = %4122
  %4133 = mul nsw i64 %4124, %const_reg_qword7, !spirv.Decorations !610
  %4134 = getelementptr float, float addrspace(4)* %66, i64 %4133
  %4135 = getelementptr float, float addrspace(4)* %4134, i64 %4125
  %4136 = addrspacecast float addrspace(4)* %4135 to float addrspace(1)*
  %4137 = load float, float addrspace(1)* %4136, align 4
  %4138 = fmul reassoc nsz arcp contract float %4137, %4, !spirv.Decorations !612
  %4139 = fadd reassoc nsz arcp contract float %4128, %4138, !spirv.Decorations !612
  %4140 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4127
  %4141 = addrspacecast float addrspace(4)* %4140 to float addrspace(1)*
  store float %4139, float addrspace(1)* %4141, align 4
  br label %._crit_edge70.1.5

._crit_edge70.1.5:                                ; preds = %4119, %4132, %4129
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4078, i32* %60, align 4, !noalias !632
  br label %4142

4142:                                             ; preds = %4142, %._crit_edge70.1.5
  %4143 = phi i32 [ 0, %._crit_edge70.1.5 ], [ %4148, %4142 ]
  %4144 = zext i32 %4143 to i64
  %4145 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4144
  %4146 = load i32, i32* %4145, align 4, !noalias !632
  %4147 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4144
  store i32 %4146, i32* %4147, align 4, !alias.scope !632
  %4148 = add nuw nsw i32 %4143, 1, !spirv.Decorations !620
  %4149 = icmp eq i32 %4143, 0
  br i1 %4149, label %4142, label %4150, !llvm.loop !644

4150:                                             ; preds = %4142
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4151 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4152 = and i1 %3519, %4079
  br i1 %4152, label %4153, label %._crit_edge70.2.5

4153:                                             ; preds = %4150
  %4154 = shl i64 %4151, 32
  %4155 = ashr exact i64 %4154, 32
  %4156 = ashr i64 %4151, 32
  %4157 = mul nsw i64 %4155, %const_reg_qword9, !spirv.Decorations !610
  %4158 = add nsw i64 %4157, %4156, !spirv.Decorations !610
  %4159 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !612
  br i1 %48, label %4163, label %4160

4160:                                             ; preds = %4153
  %4161 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4158
  %4162 = addrspacecast float addrspace(4)* %4161 to float addrspace(1)*
  store float %4159, float addrspace(1)* %4162, align 4
  br label %._crit_edge70.2.5

4163:                                             ; preds = %4153
  %4164 = mul nsw i64 %4155, %const_reg_qword7, !spirv.Decorations !610
  %4165 = getelementptr float, float addrspace(4)* %66, i64 %4164
  %4166 = getelementptr float, float addrspace(4)* %4165, i64 %4156
  %4167 = addrspacecast float addrspace(4)* %4166 to float addrspace(1)*
  %4168 = load float, float addrspace(1)* %4167, align 4
  %4169 = fmul reassoc nsz arcp contract float %4168, %4, !spirv.Decorations !612
  %4170 = fadd reassoc nsz arcp contract float %4159, %4169, !spirv.Decorations !612
  %4171 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4158
  %4172 = addrspacecast float addrspace(4)* %4171 to float addrspace(1)*
  store float %4170, float addrspace(1)* %4172, align 4
  br label %._crit_edge70.2.5

._crit_edge70.2.5:                                ; preds = %4150, %4163, %4160
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4078, i32* %60, align 4, !noalias !632
  br label %4173

4173:                                             ; preds = %4173, %._crit_edge70.2.5
  %4174 = phi i32 [ 0, %._crit_edge70.2.5 ], [ %4179, %4173 ]
  %4175 = zext i32 %4174 to i64
  %4176 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4175
  %4177 = load i32, i32* %4176, align 4, !noalias !632
  %4178 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4175
  store i32 %4177, i32* %4178, align 4, !alias.scope !632
  %4179 = add nuw nsw i32 %4174, 1, !spirv.Decorations !620
  %4180 = icmp eq i32 %4174, 0
  br i1 %4180, label %4173, label %4181, !llvm.loop !644

4181:                                             ; preds = %4173
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4182 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4183 = and i1 %3552, %4079
  br i1 %4183, label %4184, label %.preheader1.5

4184:                                             ; preds = %4181
  %4185 = shl i64 %4182, 32
  %4186 = ashr exact i64 %4185, 32
  %4187 = ashr i64 %4182, 32
  %4188 = mul nsw i64 %4186, %const_reg_qword9, !spirv.Decorations !610
  %4189 = add nsw i64 %4188, %4187, !spirv.Decorations !610
  %4190 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !612
  br i1 %48, label %4194, label %4191

4191:                                             ; preds = %4184
  %4192 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4189
  %4193 = addrspacecast float addrspace(4)* %4192 to float addrspace(1)*
  store float %4190, float addrspace(1)* %4193, align 4
  br label %.preheader1.5

4194:                                             ; preds = %4184
  %4195 = mul nsw i64 %4186, %const_reg_qword7, !spirv.Decorations !610
  %4196 = getelementptr float, float addrspace(4)* %66, i64 %4195
  %4197 = getelementptr float, float addrspace(4)* %4196, i64 %4187
  %4198 = addrspacecast float addrspace(4)* %4197 to float addrspace(1)*
  %4199 = load float, float addrspace(1)* %4198, align 4
  %4200 = fmul reassoc nsz arcp contract float %4199, %4, !spirv.Decorations !612
  %4201 = fadd reassoc nsz arcp contract float %4190, %4200, !spirv.Decorations !612
  %4202 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4189
  %4203 = addrspacecast float addrspace(4)* %4202 to float addrspace(1)*
  store float %4201, float addrspace(1)* %4203, align 4
  br label %.preheader1.5

.preheader1.5:                                    ; preds = %4181, %4194, %4191
  %4204 = or i32 %41, 6
  %4205 = icmp slt i32 %4204, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4204, i32* %60, align 4, !noalias !632
  br label %4206

4206:                                             ; preds = %4206, %.preheader1.5
  %4207 = phi i32 [ 0, %.preheader1.5 ], [ %4212, %4206 ]
  %4208 = zext i32 %4207 to i64
  %4209 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4208
  %4210 = load i32, i32* %4209, align 4, !noalias !632
  %4211 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4208
  store i32 %4210, i32* %4211, align 4, !alias.scope !632
  %4212 = add nuw nsw i32 %4207, 1, !spirv.Decorations !620
  %4213 = icmp eq i32 %4207, 0
  br i1 %4213, label %4206, label %4214, !llvm.loop !644

4214:                                             ; preds = %4206
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4215 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4216 = and i1 %3453, %4205
  br i1 %4216, label %4217, label %._crit_edge70.6

4217:                                             ; preds = %4214
  %4218 = shl i64 %4215, 32
  %4219 = ashr exact i64 %4218, 32
  %4220 = ashr i64 %4215, 32
  %4221 = mul nsw i64 %4219, %const_reg_qword9, !spirv.Decorations !610
  %4222 = add nsw i64 %4221, %4220, !spirv.Decorations !610
  %4223 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !612
  br i1 %48, label %4227, label %4224

4224:                                             ; preds = %4217
  %4225 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4222
  %4226 = addrspacecast float addrspace(4)* %4225 to float addrspace(1)*
  store float %4223, float addrspace(1)* %4226, align 4
  br label %._crit_edge70.6

4227:                                             ; preds = %4217
  %4228 = mul nsw i64 %4219, %const_reg_qword7, !spirv.Decorations !610
  %4229 = getelementptr float, float addrspace(4)* %66, i64 %4228
  %4230 = getelementptr float, float addrspace(4)* %4229, i64 %4220
  %4231 = addrspacecast float addrspace(4)* %4230 to float addrspace(1)*
  %4232 = load float, float addrspace(1)* %4231, align 4
  %4233 = fmul reassoc nsz arcp contract float %4232, %4, !spirv.Decorations !612
  %4234 = fadd reassoc nsz arcp contract float %4223, %4233, !spirv.Decorations !612
  %4235 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4222
  %4236 = addrspacecast float addrspace(4)* %4235 to float addrspace(1)*
  store float %4234, float addrspace(1)* %4236, align 4
  br label %._crit_edge70.6

._crit_edge70.6:                                  ; preds = %4214, %4227, %4224
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4204, i32* %60, align 4, !noalias !632
  br label %4237

4237:                                             ; preds = %4237, %._crit_edge70.6
  %4238 = phi i32 [ 0, %._crit_edge70.6 ], [ %4243, %4237 ]
  %4239 = zext i32 %4238 to i64
  %4240 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4239
  %4241 = load i32, i32* %4240, align 4, !noalias !632
  %4242 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4239
  store i32 %4241, i32* %4242, align 4, !alias.scope !632
  %4243 = add nuw nsw i32 %4238, 1, !spirv.Decorations !620
  %4244 = icmp eq i32 %4238, 0
  br i1 %4244, label %4237, label %4245, !llvm.loop !644

4245:                                             ; preds = %4237
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4246 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4247 = and i1 %3486, %4205
  br i1 %4247, label %4248, label %._crit_edge70.1.6

4248:                                             ; preds = %4245
  %4249 = shl i64 %4246, 32
  %4250 = ashr exact i64 %4249, 32
  %4251 = ashr i64 %4246, 32
  %4252 = mul nsw i64 %4250, %const_reg_qword9, !spirv.Decorations !610
  %4253 = add nsw i64 %4252, %4251, !spirv.Decorations !610
  %4254 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !612
  br i1 %48, label %4258, label %4255

4255:                                             ; preds = %4248
  %4256 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4253
  %4257 = addrspacecast float addrspace(4)* %4256 to float addrspace(1)*
  store float %4254, float addrspace(1)* %4257, align 4
  br label %._crit_edge70.1.6

4258:                                             ; preds = %4248
  %4259 = mul nsw i64 %4250, %const_reg_qword7, !spirv.Decorations !610
  %4260 = getelementptr float, float addrspace(4)* %66, i64 %4259
  %4261 = getelementptr float, float addrspace(4)* %4260, i64 %4251
  %4262 = addrspacecast float addrspace(4)* %4261 to float addrspace(1)*
  %4263 = load float, float addrspace(1)* %4262, align 4
  %4264 = fmul reassoc nsz arcp contract float %4263, %4, !spirv.Decorations !612
  %4265 = fadd reassoc nsz arcp contract float %4254, %4264, !spirv.Decorations !612
  %4266 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4253
  %4267 = addrspacecast float addrspace(4)* %4266 to float addrspace(1)*
  store float %4265, float addrspace(1)* %4267, align 4
  br label %._crit_edge70.1.6

._crit_edge70.1.6:                                ; preds = %4245, %4258, %4255
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4204, i32* %60, align 4, !noalias !632
  br label %4268

4268:                                             ; preds = %4268, %._crit_edge70.1.6
  %4269 = phi i32 [ 0, %._crit_edge70.1.6 ], [ %4274, %4268 ]
  %4270 = zext i32 %4269 to i64
  %4271 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4270
  %4272 = load i32, i32* %4271, align 4, !noalias !632
  %4273 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4270
  store i32 %4272, i32* %4273, align 4, !alias.scope !632
  %4274 = add nuw nsw i32 %4269, 1, !spirv.Decorations !620
  %4275 = icmp eq i32 %4269, 0
  br i1 %4275, label %4268, label %4276, !llvm.loop !644

4276:                                             ; preds = %4268
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4277 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4278 = and i1 %3519, %4205
  br i1 %4278, label %4279, label %._crit_edge70.2.6

4279:                                             ; preds = %4276
  %4280 = shl i64 %4277, 32
  %4281 = ashr exact i64 %4280, 32
  %4282 = ashr i64 %4277, 32
  %4283 = mul nsw i64 %4281, %const_reg_qword9, !spirv.Decorations !610
  %4284 = add nsw i64 %4283, %4282, !spirv.Decorations !610
  %4285 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !612
  br i1 %48, label %4289, label %4286

4286:                                             ; preds = %4279
  %4287 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4284
  %4288 = addrspacecast float addrspace(4)* %4287 to float addrspace(1)*
  store float %4285, float addrspace(1)* %4288, align 4
  br label %._crit_edge70.2.6

4289:                                             ; preds = %4279
  %4290 = mul nsw i64 %4281, %const_reg_qword7, !spirv.Decorations !610
  %4291 = getelementptr float, float addrspace(4)* %66, i64 %4290
  %4292 = getelementptr float, float addrspace(4)* %4291, i64 %4282
  %4293 = addrspacecast float addrspace(4)* %4292 to float addrspace(1)*
  %4294 = load float, float addrspace(1)* %4293, align 4
  %4295 = fmul reassoc nsz arcp contract float %4294, %4, !spirv.Decorations !612
  %4296 = fadd reassoc nsz arcp contract float %4285, %4295, !spirv.Decorations !612
  %4297 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4284
  %4298 = addrspacecast float addrspace(4)* %4297 to float addrspace(1)*
  store float %4296, float addrspace(1)* %4298, align 4
  br label %._crit_edge70.2.6

._crit_edge70.2.6:                                ; preds = %4276, %4289, %4286
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4204, i32* %60, align 4, !noalias !632
  br label %4299

4299:                                             ; preds = %4299, %._crit_edge70.2.6
  %4300 = phi i32 [ 0, %._crit_edge70.2.6 ], [ %4305, %4299 ]
  %4301 = zext i32 %4300 to i64
  %4302 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4301
  %4303 = load i32, i32* %4302, align 4, !noalias !632
  %4304 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4301
  store i32 %4303, i32* %4304, align 4, !alias.scope !632
  %4305 = add nuw nsw i32 %4300, 1, !spirv.Decorations !620
  %4306 = icmp eq i32 %4300, 0
  br i1 %4306, label %4299, label %4307, !llvm.loop !644

4307:                                             ; preds = %4299
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4308 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4309 = and i1 %3552, %4205
  br i1 %4309, label %4310, label %.preheader1.6

4310:                                             ; preds = %4307
  %4311 = shl i64 %4308, 32
  %4312 = ashr exact i64 %4311, 32
  %4313 = ashr i64 %4308, 32
  %4314 = mul nsw i64 %4312, %const_reg_qword9, !spirv.Decorations !610
  %4315 = add nsw i64 %4314, %4313, !spirv.Decorations !610
  %4316 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !612
  br i1 %48, label %4320, label %4317

4317:                                             ; preds = %4310
  %4318 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4315
  %4319 = addrspacecast float addrspace(4)* %4318 to float addrspace(1)*
  store float %4316, float addrspace(1)* %4319, align 4
  br label %.preheader1.6

4320:                                             ; preds = %4310
  %4321 = mul nsw i64 %4312, %const_reg_qword7, !spirv.Decorations !610
  %4322 = getelementptr float, float addrspace(4)* %66, i64 %4321
  %4323 = getelementptr float, float addrspace(4)* %4322, i64 %4313
  %4324 = addrspacecast float addrspace(4)* %4323 to float addrspace(1)*
  %4325 = load float, float addrspace(1)* %4324, align 4
  %4326 = fmul reassoc nsz arcp contract float %4325, %4, !spirv.Decorations !612
  %4327 = fadd reassoc nsz arcp contract float %4316, %4326, !spirv.Decorations !612
  %4328 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4315
  %4329 = addrspacecast float addrspace(4)* %4328 to float addrspace(1)*
  store float %4327, float addrspace(1)* %4329, align 4
  br label %.preheader1.6

.preheader1.6:                                    ; preds = %4307, %4320, %4317
  %4330 = or i32 %41, 7
  %4331 = icmp slt i32 %4330, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4330, i32* %60, align 4, !noalias !632
  br label %4332

4332:                                             ; preds = %4332, %.preheader1.6
  %4333 = phi i32 [ 0, %.preheader1.6 ], [ %4338, %4332 ]
  %4334 = zext i32 %4333 to i64
  %4335 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4334
  %4336 = load i32, i32* %4335, align 4, !noalias !632
  %4337 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4334
  store i32 %4336, i32* %4337, align 4, !alias.scope !632
  %4338 = add nuw nsw i32 %4333, 1, !spirv.Decorations !620
  %4339 = icmp eq i32 %4333, 0
  br i1 %4339, label %4332, label %4340, !llvm.loop !644

4340:                                             ; preds = %4332
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4341 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4342 = and i1 %3453, %4331
  br i1 %4342, label %4343, label %._crit_edge70.7

4343:                                             ; preds = %4340
  %4344 = shl i64 %4341, 32
  %4345 = ashr exact i64 %4344, 32
  %4346 = ashr i64 %4341, 32
  %4347 = mul nsw i64 %4345, %const_reg_qword9, !spirv.Decorations !610
  %4348 = add nsw i64 %4347, %4346, !spirv.Decorations !610
  %4349 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !612
  br i1 %48, label %4353, label %4350

4350:                                             ; preds = %4343
  %4351 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4348
  %4352 = addrspacecast float addrspace(4)* %4351 to float addrspace(1)*
  store float %4349, float addrspace(1)* %4352, align 4
  br label %._crit_edge70.7

4353:                                             ; preds = %4343
  %4354 = mul nsw i64 %4345, %const_reg_qword7, !spirv.Decorations !610
  %4355 = getelementptr float, float addrspace(4)* %66, i64 %4354
  %4356 = getelementptr float, float addrspace(4)* %4355, i64 %4346
  %4357 = addrspacecast float addrspace(4)* %4356 to float addrspace(1)*
  %4358 = load float, float addrspace(1)* %4357, align 4
  %4359 = fmul reassoc nsz arcp contract float %4358, %4, !spirv.Decorations !612
  %4360 = fadd reassoc nsz arcp contract float %4349, %4359, !spirv.Decorations !612
  %4361 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4348
  %4362 = addrspacecast float addrspace(4)* %4361 to float addrspace(1)*
  store float %4360, float addrspace(1)* %4362, align 4
  br label %._crit_edge70.7

._crit_edge70.7:                                  ; preds = %4340, %4353, %4350
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4330, i32* %60, align 4, !noalias !632
  br label %4363

4363:                                             ; preds = %4363, %._crit_edge70.7
  %4364 = phi i32 [ 0, %._crit_edge70.7 ], [ %4369, %4363 ]
  %4365 = zext i32 %4364 to i64
  %4366 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4365
  %4367 = load i32, i32* %4366, align 4, !noalias !632
  %4368 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4365
  store i32 %4367, i32* %4368, align 4, !alias.scope !632
  %4369 = add nuw nsw i32 %4364, 1, !spirv.Decorations !620
  %4370 = icmp eq i32 %4364, 0
  br i1 %4370, label %4363, label %4371, !llvm.loop !644

4371:                                             ; preds = %4363
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4372 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4373 = and i1 %3486, %4331
  br i1 %4373, label %4374, label %._crit_edge70.1.7

4374:                                             ; preds = %4371
  %4375 = shl i64 %4372, 32
  %4376 = ashr exact i64 %4375, 32
  %4377 = ashr i64 %4372, 32
  %4378 = mul nsw i64 %4376, %const_reg_qword9, !spirv.Decorations !610
  %4379 = add nsw i64 %4378, %4377, !spirv.Decorations !610
  %4380 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !612
  br i1 %48, label %4384, label %4381

4381:                                             ; preds = %4374
  %4382 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4379
  %4383 = addrspacecast float addrspace(4)* %4382 to float addrspace(1)*
  store float %4380, float addrspace(1)* %4383, align 4
  br label %._crit_edge70.1.7

4384:                                             ; preds = %4374
  %4385 = mul nsw i64 %4376, %const_reg_qword7, !spirv.Decorations !610
  %4386 = getelementptr float, float addrspace(4)* %66, i64 %4385
  %4387 = getelementptr float, float addrspace(4)* %4386, i64 %4377
  %4388 = addrspacecast float addrspace(4)* %4387 to float addrspace(1)*
  %4389 = load float, float addrspace(1)* %4388, align 4
  %4390 = fmul reassoc nsz arcp contract float %4389, %4, !spirv.Decorations !612
  %4391 = fadd reassoc nsz arcp contract float %4380, %4390, !spirv.Decorations !612
  %4392 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4379
  %4393 = addrspacecast float addrspace(4)* %4392 to float addrspace(1)*
  store float %4391, float addrspace(1)* %4393, align 4
  br label %._crit_edge70.1.7

._crit_edge70.1.7:                                ; preds = %4371, %4384, %4381
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4330, i32* %60, align 4, !noalias !632
  br label %4394

4394:                                             ; preds = %4394, %._crit_edge70.1.7
  %4395 = phi i32 [ 0, %._crit_edge70.1.7 ], [ %4400, %4394 ]
  %4396 = zext i32 %4395 to i64
  %4397 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4396
  %4398 = load i32, i32* %4397, align 4, !noalias !632
  %4399 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4396
  store i32 %4398, i32* %4399, align 4, !alias.scope !632
  %4400 = add nuw nsw i32 %4395, 1, !spirv.Decorations !620
  %4401 = icmp eq i32 %4395, 0
  br i1 %4401, label %4394, label %4402, !llvm.loop !644

4402:                                             ; preds = %4394
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4403 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4404 = and i1 %3519, %4331
  br i1 %4404, label %4405, label %._crit_edge70.2.7

4405:                                             ; preds = %4402
  %4406 = shl i64 %4403, 32
  %4407 = ashr exact i64 %4406, 32
  %4408 = ashr i64 %4403, 32
  %4409 = mul nsw i64 %4407, %const_reg_qword9, !spirv.Decorations !610
  %4410 = add nsw i64 %4409, %4408, !spirv.Decorations !610
  %4411 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !612
  br i1 %48, label %4415, label %4412

4412:                                             ; preds = %4405
  %4413 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4410
  %4414 = addrspacecast float addrspace(4)* %4413 to float addrspace(1)*
  store float %4411, float addrspace(1)* %4414, align 4
  br label %._crit_edge70.2.7

4415:                                             ; preds = %4405
  %4416 = mul nsw i64 %4407, %const_reg_qword7, !spirv.Decorations !610
  %4417 = getelementptr float, float addrspace(4)* %66, i64 %4416
  %4418 = getelementptr float, float addrspace(4)* %4417, i64 %4408
  %4419 = addrspacecast float addrspace(4)* %4418 to float addrspace(1)*
  %4420 = load float, float addrspace(1)* %4419, align 4
  %4421 = fmul reassoc nsz arcp contract float %4420, %4, !spirv.Decorations !612
  %4422 = fadd reassoc nsz arcp contract float %4411, %4421, !spirv.Decorations !612
  %4423 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4410
  %4424 = addrspacecast float addrspace(4)* %4423 to float addrspace(1)*
  store float %4422, float addrspace(1)* %4424, align 4
  br label %._crit_edge70.2.7

._crit_edge70.2.7:                                ; preds = %4402, %4415, %4412
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4330, i32* %60, align 4, !noalias !632
  br label %4425

4425:                                             ; preds = %4425, %._crit_edge70.2.7
  %4426 = phi i32 [ 0, %._crit_edge70.2.7 ], [ %4431, %4425 ]
  %4427 = zext i32 %4426 to i64
  %4428 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4427
  %4429 = load i32, i32* %4428, align 4, !noalias !632
  %4430 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4427
  store i32 %4429, i32* %4430, align 4, !alias.scope !632
  %4431 = add nuw nsw i32 %4426, 1, !spirv.Decorations !620
  %4432 = icmp eq i32 %4426, 0
  br i1 %4432, label %4425, label %4433, !llvm.loop !644

4433:                                             ; preds = %4425
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4434 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4435 = and i1 %3552, %4331
  br i1 %4435, label %4436, label %.preheader1.7

4436:                                             ; preds = %4433
  %4437 = shl i64 %4434, 32
  %4438 = ashr exact i64 %4437, 32
  %4439 = ashr i64 %4434, 32
  %4440 = mul nsw i64 %4438, %const_reg_qword9, !spirv.Decorations !610
  %4441 = add nsw i64 %4440, %4439, !spirv.Decorations !610
  %4442 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !612
  br i1 %48, label %4446, label %4443

4443:                                             ; preds = %4436
  %4444 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4441
  %4445 = addrspacecast float addrspace(4)* %4444 to float addrspace(1)*
  store float %4442, float addrspace(1)* %4445, align 4
  br label %.preheader1.7

4446:                                             ; preds = %4436
  %4447 = mul nsw i64 %4438, %const_reg_qword7, !spirv.Decorations !610
  %4448 = getelementptr float, float addrspace(4)* %66, i64 %4447
  %4449 = getelementptr float, float addrspace(4)* %4448, i64 %4439
  %4450 = addrspacecast float addrspace(4)* %4449 to float addrspace(1)*
  %4451 = load float, float addrspace(1)* %4450, align 4
  %4452 = fmul reassoc nsz arcp contract float %4451, %4, !spirv.Decorations !612
  %4453 = fadd reassoc nsz arcp contract float %4442, %4452, !spirv.Decorations !612
  %4454 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4441
  %4455 = addrspacecast float addrspace(4)* %4454 to float addrspace(1)*
  store float %4453, float addrspace(1)* %4455, align 4
  br label %.preheader1.7

.preheader1.7:                                    ; preds = %4433, %4446, %4443
  %4456 = or i32 %41, 8
  %4457 = icmp slt i32 %4456, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4456, i32* %60, align 4, !noalias !632
  br label %4458

4458:                                             ; preds = %4458, %.preheader1.7
  %4459 = phi i32 [ 0, %.preheader1.7 ], [ %4464, %4458 ]
  %4460 = zext i32 %4459 to i64
  %4461 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4460
  %4462 = load i32, i32* %4461, align 4, !noalias !632
  %4463 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4460
  store i32 %4462, i32* %4463, align 4, !alias.scope !632
  %4464 = add nuw nsw i32 %4459, 1, !spirv.Decorations !620
  %4465 = icmp eq i32 %4459, 0
  br i1 %4465, label %4458, label %4466, !llvm.loop !644

4466:                                             ; preds = %4458
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4467 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4468 = and i1 %3453, %4457
  br i1 %4468, label %4469, label %._crit_edge70.8

4469:                                             ; preds = %4466
  %4470 = shl i64 %4467, 32
  %4471 = ashr exact i64 %4470, 32
  %4472 = ashr i64 %4467, 32
  %4473 = mul nsw i64 %4471, %const_reg_qword9, !spirv.Decorations !610
  %4474 = add nsw i64 %4473, %4472, !spirv.Decorations !610
  %4475 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !612
  br i1 %48, label %4479, label %4476

4476:                                             ; preds = %4469
  %4477 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4474
  %4478 = addrspacecast float addrspace(4)* %4477 to float addrspace(1)*
  store float %4475, float addrspace(1)* %4478, align 4
  br label %._crit_edge70.8

4479:                                             ; preds = %4469
  %4480 = mul nsw i64 %4471, %const_reg_qword7, !spirv.Decorations !610
  %4481 = getelementptr float, float addrspace(4)* %66, i64 %4480
  %4482 = getelementptr float, float addrspace(4)* %4481, i64 %4472
  %4483 = addrspacecast float addrspace(4)* %4482 to float addrspace(1)*
  %4484 = load float, float addrspace(1)* %4483, align 4
  %4485 = fmul reassoc nsz arcp contract float %4484, %4, !spirv.Decorations !612
  %4486 = fadd reassoc nsz arcp contract float %4475, %4485, !spirv.Decorations !612
  %4487 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4474
  %4488 = addrspacecast float addrspace(4)* %4487 to float addrspace(1)*
  store float %4486, float addrspace(1)* %4488, align 4
  br label %._crit_edge70.8

._crit_edge70.8:                                  ; preds = %4466, %4479, %4476
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4456, i32* %60, align 4, !noalias !632
  br label %4489

4489:                                             ; preds = %4489, %._crit_edge70.8
  %4490 = phi i32 [ 0, %._crit_edge70.8 ], [ %4495, %4489 ]
  %4491 = zext i32 %4490 to i64
  %4492 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4491
  %4493 = load i32, i32* %4492, align 4, !noalias !632
  %4494 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4491
  store i32 %4493, i32* %4494, align 4, !alias.scope !632
  %4495 = add nuw nsw i32 %4490, 1, !spirv.Decorations !620
  %4496 = icmp eq i32 %4490, 0
  br i1 %4496, label %4489, label %4497, !llvm.loop !644

4497:                                             ; preds = %4489
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4498 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4499 = and i1 %3486, %4457
  br i1 %4499, label %4500, label %._crit_edge70.1.8

4500:                                             ; preds = %4497
  %4501 = shl i64 %4498, 32
  %4502 = ashr exact i64 %4501, 32
  %4503 = ashr i64 %4498, 32
  %4504 = mul nsw i64 %4502, %const_reg_qword9, !spirv.Decorations !610
  %4505 = add nsw i64 %4504, %4503, !spirv.Decorations !610
  %4506 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !612
  br i1 %48, label %4510, label %4507

4507:                                             ; preds = %4500
  %4508 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4505
  %4509 = addrspacecast float addrspace(4)* %4508 to float addrspace(1)*
  store float %4506, float addrspace(1)* %4509, align 4
  br label %._crit_edge70.1.8

4510:                                             ; preds = %4500
  %4511 = mul nsw i64 %4502, %const_reg_qword7, !spirv.Decorations !610
  %4512 = getelementptr float, float addrspace(4)* %66, i64 %4511
  %4513 = getelementptr float, float addrspace(4)* %4512, i64 %4503
  %4514 = addrspacecast float addrspace(4)* %4513 to float addrspace(1)*
  %4515 = load float, float addrspace(1)* %4514, align 4
  %4516 = fmul reassoc nsz arcp contract float %4515, %4, !spirv.Decorations !612
  %4517 = fadd reassoc nsz arcp contract float %4506, %4516, !spirv.Decorations !612
  %4518 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4505
  %4519 = addrspacecast float addrspace(4)* %4518 to float addrspace(1)*
  store float %4517, float addrspace(1)* %4519, align 4
  br label %._crit_edge70.1.8

._crit_edge70.1.8:                                ; preds = %4497, %4510, %4507
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4456, i32* %60, align 4, !noalias !632
  br label %4520

4520:                                             ; preds = %4520, %._crit_edge70.1.8
  %4521 = phi i32 [ 0, %._crit_edge70.1.8 ], [ %4526, %4520 ]
  %4522 = zext i32 %4521 to i64
  %4523 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4522
  %4524 = load i32, i32* %4523, align 4, !noalias !632
  %4525 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4522
  store i32 %4524, i32* %4525, align 4, !alias.scope !632
  %4526 = add nuw nsw i32 %4521, 1, !spirv.Decorations !620
  %4527 = icmp eq i32 %4521, 0
  br i1 %4527, label %4520, label %4528, !llvm.loop !644

4528:                                             ; preds = %4520
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4529 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4530 = and i1 %3519, %4457
  br i1 %4530, label %4531, label %._crit_edge70.2.8

4531:                                             ; preds = %4528
  %4532 = shl i64 %4529, 32
  %4533 = ashr exact i64 %4532, 32
  %4534 = ashr i64 %4529, 32
  %4535 = mul nsw i64 %4533, %const_reg_qword9, !spirv.Decorations !610
  %4536 = add nsw i64 %4535, %4534, !spirv.Decorations !610
  %4537 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !612
  br i1 %48, label %4541, label %4538

4538:                                             ; preds = %4531
  %4539 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4536
  %4540 = addrspacecast float addrspace(4)* %4539 to float addrspace(1)*
  store float %4537, float addrspace(1)* %4540, align 4
  br label %._crit_edge70.2.8

4541:                                             ; preds = %4531
  %4542 = mul nsw i64 %4533, %const_reg_qword7, !spirv.Decorations !610
  %4543 = getelementptr float, float addrspace(4)* %66, i64 %4542
  %4544 = getelementptr float, float addrspace(4)* %4543, i64 %4534
  %4545 = addrspacecast float addrspace(4)* %4544 to float addrspace(1)*
  %4546 = load float, float addrspace(1)* %4545, align 4
  %4547 = fmul reassoc nsz arcp contract float %4546, %4, !spirv.Decorations !612
  %4548 = fadd reassoc nsz arcp contract float %4537, %4547, !spirv.Decorations !612
  %4549 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4536
  %4550 = addrspacecast float addrspace(4)* %4549 to float addrspace(1)*
  store float %4548, float addrspace(1)* %4550, align 4
  br label %._crit_edge70.2.8

._crit_edge70.2.8:                                ; preds = %4528, %4541, %4538
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4456, i32* %60, align 4, !noalias !632
  br label %4551

4551:                                             ; preds = %4551, %._crit_edge70.2.8
  %4552 = phi i32 [ 0, %._crit_edge70.2.8 ], [ %4557, %4551 ]
  %4553 = zext i32 %4552 to i64
  %4554 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4553
  %4555 = load i32, i32* %4554, align 4, !noalias !632
  %4556 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4553
  store i32 %4555, i32* %4556, align 4, !alias.scope !632
  %4557 = add nuw nsw i32 %4552, 1, !spirv.Decorations !620
  %4558 = icmp eq i32 %4552, 0
  br i1 %4558, label %4551, label %4559, !llvm.loop !644

4559:                                             ; preds = %4551
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4560 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4561 = and i1 %3552, %4457
  br i1 %4561, label %4562, label %.preheader1.8

4562:                                             ; preds = %4559
  %4563 = shl i64 %4560, 32
  %4564 = ashr exact i64 %4563, 32
  %4565 = ashr i64 %4560, 32
  %4566 = mul nsw i64 %4564, %const_reg_qword9, !spirv.Decorations !610
  %4567 = add nsw i64 %4566, %4565, !spirv.Decorations !610
  %4568 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !612
  br i1 %48, label %4572, label %4569

4569:                                             ; preds = %4562
  %4570 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4567
  %4571 = addrspacecast float addrspace(4)* %4570 to float addrspace(1)*
  store float %4568, float addrspace(1)* %4571, align 4
  br label %.preheader1.8

4572:                                             ; preds = %4562
  %4573 = mul nsw i64 %4564, %const_reg_qword7, !spirv.Decorations !610
  %4574 = getelementptr float, float addrspace(4)* %66, i64 %4573
  %4575 = getelementptr float, float addrspace(4)* %4574, i64 %4565
  %4576 = addrspacecast float addrspace(4)* %4575 to float addrspace(1)*
  %4577 = load float, float addrspace(1)* %4576, align 4
  %4578 = fmul reassoc nsz arcp contract float %4577, %4, !spirv.Decorations !612
  %4579 = fadd reassoc nsz arcp contract float %4568, %4578, !spirv.Decorations !612
  %4580 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4567
  %4581 = addrspacecast float addrspace(4)* %4580 to float addrspace(1)*
  store float %4579, float addrspace(1)* %4581, align 4
  br label %.preheader1.8

.preheader1.8:                                    ; preds = %4559, %4572, %4569
  %4582 = or i32 %41, 9
  %4583 = icmp slt i32 %4582, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4582, i32* %60, align 4, !noalias !632
  br label %4584

4584:                                             ; preds = %4584, %.preheader1.8
  %4585 = phi i32 [ 0, %.preheader1.8 ], [ %4590, %4584 ]
  %4586 = zext i32 %4585 to i64
  %4587 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4586
  %4588 = load i32, i32* %4587, align 4, !noalias !632
  %4589 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4586
  store i32 %4588, i32* %4589, align 4, !alias.scope !632
  %4590 = add nuw nsw i32 %4585, 1, !spirv.Decorations !620
  %4591 = icmp eq i32 %4585, 0
  br i1 %4591, label %4584, label %4592, !llvm.loop !644

4592:                                             ; preds = %4584
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4593 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4594 = and i1 %3453, %4583
  br i1 %4594, label %4595, label %._crit_edge70.9

4595:                                             ; preds = %4592
  %4596 = shl i64 %4593, 32
  %4597 = ashr exact i64 %4596, 32
  %4598 = ashr i64 %4593, 32
  %4599 = mul nsw i64 %4597, %const_reg_qword9, !spirv.Decorations !610
  %4600 = add nsw i64 %4599, %4598, !spirv.Decorations !610
  %4601 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !612
  br i1 %48, label %4605, label %4602

4602:                                             ; preds = %4595
  %4603 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4600
  %4604 = addrspacecast float addrspace(4)* %4603 to float addrspace(1)*
  store float %4601, float addrspace(1)* %4604, align 4
  br label %._crit_edge70.9

4605:                                             ; preds = %4595
  %4606 = mul nsw i64 %4597, %const_reg_qword7, !spirv.Decorations !610
  %4607 = getelementptr float, float addrspace(4)* %66, i64 %4606
  %4608 = getelementptr float, float addrspace(4)* %4607, i64 %4598
  %4609 = addrspacecast float addrspace(4)* %4608 to float addrspace(1)*
  %4610 = load float, float addrspace(1)* %4609, align 4
  %4611 = fmul reassoc nsz arcp contract float %4610, %4, !spirv.Decorations !612
  %4612 = fadd reassoc nsz arcp contract float %4601, %4611, !spirv.Decorations !612
  %4613 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4600
  %4614 = addrspacecast float addrspace(4)* %4613 to float addrspace(1)*
  store float %4612, float addrspace(1)* %4614, align 4
  br label %._crit_edge70.9

._crit_edge70.9:                                  ; preds = %4592, %4605, %4602
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4582, i32* %60, align 4, !noalias !632
  br label %4615

4615:                                             ; preds = %4615, %._crit_edge70.9
  %4616 = phi i32 [ 0, %._crit_edge70.9 ], [ %4621, %4615 ]
  %4617 = zext i32 %4616 to i64
  %4618 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4617
  %4619 = load i32, i32* %4618, align 4, !noalias !632
  %4620 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4617
  store i32 %4619, i32* %4620, align 4, !alias.scope !632
  %4621 = add nuw nsw i32 %4616, 1, !spirv.Decorations !620
  %4622 = icmp eq i32 %4616, 0
  br i1 %4622, label %4615, label %4623, !llvm.loop !644

4623:                                             ; preds = %4615
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4624 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4625 = and i1 %3486, %4583
  br i1 %4625, label %4626, label %._crit_edge70.1.9

4626:                                             ; preds = %4623
  %4627 = shl i64 %4624, 32
  %4628 = ashr exact i64 %4627, 32
  %4629 = ashr i64 %4624, 32
  %4630 = mul nsw i64 %4628, %const_reg_qword9, !spirv.Decorations !610
  %4631 = add nsw i64 %4630, %4629, !spirv.Decorations !610
  %4632 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !612
  br i1 %48, label %4636, label %4633

4633:                                             ; preds = %4626
  %4634 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4631
  %4635 = addrspacecast float addrspace(4)* %4634 to float addrspace(1)*
  store float %4632, float addrspace(1)* %4635, align 4
  br label %._crit_edge70.1.9

4636:                                             ; preds = %4626
  %4637 = mul nsw i64 %4628, %const_reg_qword7, !spirv.Decorations !610
  %4638 = getelementptr float, float addrspace(4)* %66, i64 %4637
  %4639 = getelementptr float, float addrspace(4)* %4638, i64 %4629
  %4640 = addrspacecast float addrspace(4)* %4639 to float addrspace(1)*
  %4641 = load float, float addrspace(1)* %4640, align 4
  %4642 = fmul reassoc nsz arcp contract float %4641, %4, !spirv.Decorations !612
  %4643 = fadd reassoc nsz arcp contract float %4632, %4642, !spirv.Decorations !612
  %4644 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4631
  %4645 = addrspacecast float addrspace(4)* %4644 to float addrspace(1)*
  store float %4643, float addrspace(1)* %4645, align 4
  br label %._crit_edge70.1.9

._crit_edge70.1.9:                                ; preds = %4623, %4636, %4633
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4582, i32* %60, align 4, !noalias !632
  br label %4646

4646:                                             ; preds = %4646, %._crit_edge70.1.9
  %4647 = phi i32 [ 0, %._crit_edge70.1.9 ], [ %4652, %4646 ]
  %4648 = zext i32 %4647 to i64
  %4649 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4648
  %4650 = load i32, i32* %4649, align 4, !noalias !632
  %4651 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4648
  store i32 %4650, i32* %4651, align 4, !alias.scope !632
  %4652 = add nuw nsw i32 %4647, 1, !spirv.Decorations !620
  %4653 = icmp eq i32 %4647, 0
  br i1 %4653, label %4646, label %4654, !llvm.loop !644

4654:                                             ; preds = %4646
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4655 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4656 = and i1 %3519, %4583
  br i1 %4656, label %4657, label %._crit_edge70.2.9

4657:                                             ; preds = %4654
  %4658 = shl i64 %4655, 32
  %4659 = ashr exact i64 %4658, 32
  %4660 = ashr i64 %4655, 32
  %4661 = mul nsw i64 %4659, %const_reg_qword9, !spirv.Decorations !610
  %4662 = add nsw i64 %4661, %4660, !spirv.Decorations !610
  %4663 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !612
  br i1 %48, label %4667, label %4664

4664:                                             ; preds = %4657
  %4665 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4662
  %4666 = addrspacecast float addrspace(4)* %4665 to float addrspace(1)*
  store float %4663, float addrspace(1)* %4666, align 4
  br label %._crit_edge70.2.9

4667:                                             ; preds = %4657
  %4668 = mul nsw i64 %4659, %const_reg_qword7, !spirv.Decorations !610
  %4669 = getelementptr float, float addrspace(4)* %66, i64 %4668
  %4670 = getelementptr float, float addrspace(4)* %4669, i64 %4660
  %4671 = addrspacecast float addrspace(4)* %4670 to float addrspace(1)*
  %4672 = load float, float addrspace(1)* %4671, align 4
  %4673 = fmul reassoc nsz arcp contract float %4672, %4, !spirv.Decorations !612
  %4674 = fadd reassoc nsz arcp contract float %4663, %4673, !spirv.Decorations !612
  %4675 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4662
  %4676 = addrspacecast float addrspace(4)* %4675 to float addrspace(1)*
  store float %4674, float addrspace(1)* %4676, align 4
  br label %._crit_edge70.2.9

._crit_edge70.2.9:                                ; preds = %4654, %4667, %4664
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4582, i32* %60, align 4, !noalias !632
  br label %4677

4677:                                             ; preds = %4677, %._crit_edge70.2.9
  %4678 = phi i32 [ 0, %._crit_edge70.2.9 ], [ %4683, %4677 ]
  %4679 = zext i32 %4678 to i64
  %4680 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4679
  %4681 = load i32, i32* %4680, align 4, !noalias !632
  %4682 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4679
  store i32 %4681, i32* %4682, align 4, !alias.scope !632
  %4683 = add nuw nsw i32 %4678, 1, !spirv.Decorations !620
  %4684 = icmp eq i32 %4678, 0
  br i1 %4684, label %4677, label %4685, !llvm.loop !644

4685:                                             ; preds = %4677
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4686 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4687 = and i1 %3552, %4583
  br i1 %4687, label %4688, label %.preheader1.9

4688:                                             ; preds = %4685
  %4689 = shl i64 %4686, 32
  %4690 = ashr exact i64 %4689, 32
  %4691 = ashr i64 %4686, 32
  %4692 = mul nsw i64 %4690, %const_reg_qword9, !spirv.Decorations !610
  %4693 = add nsw i64 %4692, %4691, !spirv.Decorations !610
  %4694 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !612
  br i1 %48, label %4698, label %4695

4695:                                             ; preds = %4688
  %4696 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4693
  %4697 = addrspacecast float addrspace(4)* %4696 to float addrspace(1)*
  store float %4694, float addrspace(1)* %4697, align 4
  br label %.preheader1.9

4698:                                             ; preds = %4688
  %4699 = mul nsw i64 %4690, %const_reg_qword7, !spirv.Decorations !610
  %4700 = getelementptr float, float addrspace(4)* %66, i64 %4699
  %4701 = getelementptr float, float addrspace(4)* %4700, i64 %4691
  %4702 = addrspacecast float addrspace(4)* %4701 to float addrspace(1)*
  %4703 = load float, float addrspace(1)* %4702, align 4
  %4704 = fmul reassoc nsz arcp contract float %4703, %4, !spirv.Decorations !612
  %4705 = fadd reassoc nsz arcp contract float %4694, %4704, !spirv.Decorations !612
  %4706 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4693
  %4707 = addrspacecast float addrspace(4)* %4706 to float addrspace(1)*
  store float %4705, float addrspace(1)* %4707, align 4
  br label %.preheader1.9

.preheader1.9:                                    ; preds = %4685, %4698, %4695
  %4708 = or i32 %41, 10
  %4709 = icmp slt i32 %4708, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4708, i32* %60, align 4, !noalias !632
  br label %4710

4710:                                             ; preds = %4710, %.preheader1.9
  %4711 = phi i32 [ 0, %.preheader1.9 ], [ %4716, %4710 ]
  %4712 = zext i32 %4711 to i64
  %4713 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4712
  %4714 = load i32, i32* %4713, align 4, !noalias !632
  %4715 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4712
  store i32 %4714, i32* %4715, align 4, !alias.scope !632
  %4716 = add nuw nsw i32 %4711, 1, !spirv.Decorations !620
  %4717 = icmp eq i32 %4711, 0
  br i1 %4717, label %4710, label %4718, !llvm.loop !644

4718:                                             ; preds = %4710
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4719 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4720 = and i1 %3453, %4709
  br i1 %4720, label %4721, label %._crit_edge70.10

4721:                                             ; preds = %4718
  %4722 = shl i64 %4719, 32
  %4723 = ashr exact i64 %4722, 32
  %4724 = ashr i64 %4719, 32
  %4725 = mul nsw i64 %4723, %const_reg_qword9, !spirv.Decorations !610
  %4726 = add nsw i64 %4725, %4724, !spirv.Decorations !610
  %4727 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !612
  br i1 %48, label %4731, label %4728

4728:                                             ; preds = %4721
  %4729 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4726
  %4730 = addrspacecast float addrspace(4)* %4729 to float addrspace(1)*
  store float %4727, float addrspace(1)* %4730, align 4
  br label %._crit_edge70.10

4731:                                             ; preds = %4721
  %4732 = mul nsw i64 %4723, %const_reg_qword7, !spirv.Decorations !610
  %4733 = getelementptr float, float addrspace(4)* %66, i64 %4732
  %4734 = getelementptr float, float addrspace(4)* %4733, i64 %4724
  %4735 = addrspacecast float addrspace(4)* %4734 to float addrspace(1)*
  %4736 = load float, float addrspace(1)* %4735, align 4
  %4737 = fmul reassoc nsz arcp contract float %4736, %4, !spirv.Decorations !612
  %4738 = fadd reassoc nsz arcp contract float %4727, %4737, !spirv.Decorations !612
  %4739 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4726
  %4740 = addrspacecast float addrspace(4)* %4739 to float addrspace(1)*
  store float %4738, float addrspace(1)* %4740, align 4
  br label %._crit_edge70.10

._crit_edge70.10:                                 ; preds = %4718, %4731, %4728
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4708, i32* %60, align 4, !noalias !632
  br label %4741

4741:                                             ; preds = %4741, %._crit_edge70.10
  %4742 = phi i32 [ 0, %._crit_edge70.10 ], [ %4747, %4741 ]
  %4743 = zext i32 %4742 to i64
  %4744 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4743
  %4745 = load i32, i32* %4744, align 4, !noalias !632
  %4746 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4743
  store i32 %4745, i32* %4746, align 4, !alias.scope !632
  %4747 = add nuw nsw i32 %4742, 1, !spirv.Decorations !620
  %4748 = icmp eq i32 %4742, 0
  br i1 %4748, label %4741, label %4749, !llvm.loop !644

4749:                                             ; preds = %4741
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4750 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4751 = and i1 %3486, %4709
  br i1 %4751, label %4752, label %._crit_edge70.1.10

4752:                                             ; preds = %4749
  %4753 = shl i64 %4750, 32
  %4754 = ashr exact i64 %4753, 32
  %4755 = ashr i64 %4750, 32
  %4756 = mul nsw i64 %4754, %const_reg_qword9, !spirv.Decorations !610
  %4757 = add nsw i64 %4756, %4755, !spirv.Decorations !610
  %4758 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !612
  br i1 %48, label %4762, label %4759

4759:                                             ; preds = %4752
  %4760 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4757
  %4761 = addrspacecast float addrspace(4)* %4760 to float addrspace(1)*
  store float %4758, float addrspace(1)* %4761, align 4
  br label %._crit_edge70.1.10

4762:                                             ; preds = %4752
  %4763 = mul nsw i64 %4754, %const_reg_qword7, !spirv.Decorations !610
  %4764 = getelementptr float, float addrspace(4)* %66, i64 %4763
  %4765 = getelementptr float, float addrspace(4)* %4764, i64 %4755
  %4766 = addrspacecast float addrspace(4)* %4765 to float addrspace(1)*
  %4767 = load float, float addrspace(1)* %4766, align 4
  %4768 = fmul reassoc nsz arcp contract float %4767, %4, !spirv.Decorations !612
  %4769 = fadd reassoc nsz arcp contract float %4758, %4768, !spirv.Decorations !612
  %4770 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4757
  %4771 = addrspacecast float addrspace(4)* %4770 to float addrspace(1)*
  store float %4769, float addrspace(1)* %4771, align 4
  br label %._crit_edge70.1.10

._crit_edge70.1.10:                               ; preds = %4749, %4762, %4759
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4708, i32* %60, align 4, !noalias !632
  br label %4772

4772:                                             ; preds = %4772, %._crit_edge70.1.10
  %4773 = phi i32 [ 0, %._crit_edge70.1.10 ], [ %4778, %4772 ]
  %4774 = zext i32 %4773 to i64
  %4775 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4774
  %4776 = load i32, i32* %4775, align 4, !noalias !632
  %4777 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4774
  store i32 %4776, i32* %4777, align 4, !alias.scope !632
  %4778 = add nuw nsw i32 %4773, 1, !spirv.Decorations !620
  %4779 = icmp eq i32 %4773, 0
  br i1 %4779, label %4772, label %4780, !llvm.loop !644

4780:                                             ; preds = %4772
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4781 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4782 = and i1 %3519, %4709
  br i1 %4782, label %4783, label %._crit_edge70.2.10

4783:                                             ; preds = %4780
  %4784 = shl i64 %4781, 32
  %4785 = ashr exact i64 %4784, 32
  %4786 = ashr i64 %4781, 32
  %4787 = mul nsw i64 %4785, %const_reg_qword9, !spirv.Decorations !610
  %4788 = add nsw i64 %4787, %4786, !spirv.Decorations !610
  %4789 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !612
  br i1 %48, label %4793, label %4790

4790:                                             ; preds = %4783
  %4791 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4788
  %4792 = addrspacecast float addrspace(4)* %4791 to float addrspace(1)*
  store float %4789, float addrspace(1)* %4792, align 4
  br label %._crit_edge70.2.10

4793:                                             ; preds = %4783
  %4794 = mul nsw i64 %4785, %const_reg_qword7, !spirv.Decorations !610
  %4795 = getelementptr float, float addrspace(4)* %66, i64 %4794
  %4796 = getelementptr float, float addrspace(4)* %4795, i64 %4786
  %4797 = addrspacecast float addrspace(4)* %4796 to float addrspace(1)*
  %4798 = load float, float addrspace(1)* %4797, align 4
  %4799 = fmul reassoc nsz arcp contract float %4798, %4, !spirv.Decorations !612
  %4800 = fadd reassoc nsz arcp contract float %4789, %4799, !spirv.Decorations !612
  %4801 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4788
  %4802 = addrspacecast float addrspace(4)* %4801 to float addrspace(1)*
  store float %4800, float addrspace(1)* %4802, align 4
  br label %._crit_edge70.2.10

._crit_edge70.2.10:                               ; preds = %4780, %4793, %4790
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4708, i32* %60, align 4, !noalias !632
  br label %4803

4803:                                             ; preds = %4803, %._crit_edge70.2.10
  %4804 = phi i32 [ 0, %._crit_edge70.2.10 ], [ %4809, %4803 ]
  %4805 = zext i32 %4804 to i64
  %4806 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4805
  %4807 = load i32, i32* %4806, align 4, !noalias !632
  %4808 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4805
  store i32 %4807, i32* %4808, align 4, !alias.scope !632
  %4809 = add nuw nsw i32 %4804, 1, !spirv.Decorations !620
  %4810 = icmp eq i32 %4804, 0
  br i1 %4810, label %4803, label %4811, !llvm.loop !644

4811:                                             ; preds = %4803
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4812 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4813 = and i1 %3552, %4709
  br i1 %4813, label %4814, label %.preheader1.10

4814:                                             ; preds = %4811
  %4815 = shl i64 %4812, 32
  %4816 = ashr exact i64 %4815, 32
  %4817 = ashr i64 %4812, 32
  %4818 = mul nsw i64 %4816, %const_reg_qword9, !spirv.Decorations !610
  %4819 = add nsw i64 %4818, %4817, !spirv.Decorations !610
  %4820 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !612
  br i1 %48, label %4824, label %4821

4821:                                             ; preds = %4814
  %4822 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4819
  %4823 = addrspacecast float addrspace(4)* %4822 to float addrspace(1)*
  store float %4820, float addrspace(1)* %4823, align 4
  br label %.preheader1.10

4824:                                             ; preds = %4814
  %4825 = mul nsw i64 %4816, %const_reg_qword7, !spirv.Decorations !610
  %4826 = getelementptr float, float addrspace(4)* %66, i64 %4825
  %4827 = getelementptr float, float addrspace(4)* %4826, i64 %4817
  %4828 = addrspacecast float addrspace(4)* %4827 to float addrspace(1)*
  %4829 = load float, float addrspace(1)* %4828, align 4
  %4830 = fmul reassoc nsz arcp contract float %4829, %4, !spirv.Decorations !612
  %4831 = fadd reassoc nsz arcp contract float %4820, %4830, !spirv.Decorations !612
  %4832 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4819
  %4833 = addrspacecast float addrspace(4)* %4832 to float addrspace(1)*
  store float %4831, float addrspace(1)* %4833, align 4
  br label %.preheader1.10

.preheader1.10:                                   ; preds = %4811, %4824, %4821
  %4834 = or i32 %41, 11
  %4835 = icmp slt i32 %4834, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4834, i32* %60, align 4, !noalias !632
  br label %4836

4836:                                             ; preds = %4836, %.preheader1.10
  %4837 = phi i32 [ 0, %.preheader1.10 ], [ %4842, %4836 ]
  %4838 = zext i32 %4837 to i64
  %4839 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4838
  %4840 = load i32, i32* %4839, align 4, !noalias !632
  %4841 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4838
  store i32 %4840, i32* %4841, align 4, !alias.scope !632
  %4842 = add nuw nsw i32 %4837, 1, !spirv.Decorations !620
  %4843 = icmp eq i32 %4837, 0
  br i1 %4843, label %4836, label %4844, !llvm.loop !644

4844:                                             ; preds = %4836
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4845 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4846 = and i1 %3453, %4835
  br i1 %4846, label %4847, label %._crit_edge70.11

4847:                                             ; preds = %4844
  %4848 = shl i64 %4845, 32
  %4849 = ashr exact i64 %4848, 32
  %4850 = ashr i64 %4845, 32
  %4851 = mul nsw i64 %4849, %const_reg_qword9, !spirv.Decorations !610
  %4852 = add nsw i64 %4851, %4850, !spirv.Decorations !610
  %4853 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !612
  br i1 %48, label %4857, label %4854

4854:                                             ; preds = %4847
  %4855 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4852
  %4856 = addrspacecast float addrspace(4)* %4855 to float addrspace(1)*
  store float %4853, float addrspace(1)* %4856, align 4
  br label %._crit_edge70.11

4857:                                             ; preds = %4847
  %4858 = mul nsw i64 %4849, %const_reg_qword7, !spirv.Decorations !610
  %4859 = getelementptr float, float addrspace(4)* %66, i64 %4858
  %4860 = getelementptr float, float addrspace(4)* %4859, i64 %4850
  %4861 = addrspacecast float addrspace(4)* %4860 to float addrspace(1)*
  %4862 = load float, float addrspace(1)* %4861, align 4
  %4863 = fmul reassoc nsz arcp contract float %4862, %4, !spirv.Decorations !612
  %4864 = fadd reassoc nsz arcp contract float %4853, %4863, !spirv.Decorations !612
  %4865 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4852
  %4866 = addrspacecast float addrspace(4)* %4865 to float addrspace(1)*
  store float %4864, float addrspace(1)* %4866, align 4
  br label %._crit_edge70.11

._crit_edge70.11:                                 ; preds = %4844, %4857, %4854
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4834, i32* %60, align 4, !noalias !632
  br label %4867

4867:                                             ; preds = %4867, %._crit_edge70.11
  %4868 = phi i32 [ 0, %._crit_edge70.11 ], [ %4873, %4867 ]
  %4869 = zext i32 %4868 to i64
  %4870 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4869
  %4871 = load i32, i32* %4870, align 4, !noalias !632
  %4872 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4869
  store i32 %4871, i32* %4872, align 4, !alias.scope !632
  %4873 = add nuw nsw i32 %4868, 1, !spirv.Decorations !620
  %4874 = icmp eq i32 %4868, 0
  br i1 %4874, label %4867, label %4875, !llvm.loop !644

4875:                                             ; preds = %4867
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4876 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4877 = and i1 %3486, %4835
  br i1 %4877, label %4878, label %._crit_edge70.1.11

4878:                                             ; preds = %4875
  %4879 = shl i64 %4876, 32
  %4880 = ashr exact i64 %4879, 32
  %4881 = ashr i64 %4876, 32
  %4882 = mul nsw i64 %4880, %const_reg_qword9, !spirv.Decorations !610
  %4883 = add nsw i64 %4882, %4881, !spirv.Decorations !610
  %4884 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !612
  br i1 %48, label %4888, label %4885

4885:                                             ; preds = %4878
  %4886 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4883
  %4887 = addrspacecast float addrspace(4)* %4886 to float addrspace(1)*
  store float %4884, float addrspace(1)* %4887, align 4
  br label %._crit_edge70.1.11

4888:                                             ; preds = %4878
  %4889 = mul nsw i64 %4880, %const_reg_qword7, !spirv.Decorations !610
  %4890 = getelementptr float, float addrspace(4)* %66, i64 %4889
  %4891 = getelementptr float, float addrspace(4)* %4890, i64 %4881
  %4892 = addrspacecast float addrspace(4)* %4891 to float addrspace(1)*
  %4893 = load float, float addrspace(1)* %4892, align 4
  %4894 = fmul reassoc nsz arcp contract float %4893, %4, !spirv.Decorations !612
  %4895 = fadd reassoc nsz arcp contract float %4884, %4894, !spirv.Decorations !612
  %4896 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4883
  %4897 = addrspacecast float addrspace(4)* %4896 to float addrspace(1)*
  store float %4895, float addrspace(1)* %4897, align 4
  br label %._crit_edge70.1.11

._crit_edge70.1.11:                               ; preds = %4875, %4888, %4885
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4834, i32* %60, align 4, !noalias !632
  br label %4898

4898:                                             ; preds = %4898, %._crit_edge70.1.11
  %4899 = phi i32 [ 0, %._crit_edge70.1.11 ], [ %4904, %4898 ]
  %4900 = zext i32 %4899 to i64
  %4901 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4900
  %4902 = load i32, i32* %4901, align 4, !noalias !632
  %4903 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4900
  store i32 %4902, i32* %4903, align 4, !alias.scope !632
  %4904 = add nuw nsw i32 %4899, 1, !spirv.Decorations !620
  %4905 = icmp eq i32 %4899, 0
  br i1 %4905, label %4898, label %4906, !llvm.loop !644

4906:                                             ; preds = %4898
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4907 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4908 = and i1 %3519, %4835
  br i1 %4908, label %4909, label %._crit_edge70.2.11

4909:                                             ; preds = %4906
  %4910 = shl i64 %4907, 32
  %4911 = ashr exact i64 %4910, 32
  %4912 = ashr i64 %4907, 32
  %4913 = mul nsw i64 %4911, %const_reg_qword9, !spirv.Decorations !610
  %4914 = add nsw i64 %4913, %4912, !spirv.Decorations !610
  %4915 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !612
  br i1 %48, label %4919, label %4916

4916:                                             ; preds = %4909
  %4917 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4914
  %4918 = addrspacecast float addrspace(4)* %4917 to float addrspace(1)*
  store float %4915, float addrspace(1)* %4918, align 4
  br label %._crit_edge70.2.11

4919:                                             ; preds = %4909
  %4920 = mul nsw i64 %4911, %const_reg_qword7, !spirv.Decorations !610
  %4921 = getelementptr float, float addrspace(4)* %66, i64 %4920
  %4922 = getelementptr float, float addrspace(4)* %4921, i64 %4912
  %4923 = addrspacecast float addrspace(4)* %4922 to float addrspace(1)*
  %4924 = load float, float addrspace(1)* %4923, align 4
  %4925 = fmul reassoc nsz arcp contract float %4924, %4, !spirv.Decorations !612
  %4926 = fadd reassoc nsz arcp contract float %4915, %4925, !spirv.Decorations !612
  %4927 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4914
  %4928 = addrspacecast float addrspace(4)* %4927 to float addrspace(1)*
  store float %4926, float addrspace(1)* %4928, align 4
  br label %._crit_edge70.2.11

._crit_edge70.2.11:                               ; preds = %4906, %4919, %4916
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4834, i32* %60, align 4, !noalias !632
  br label %4929

4929:                                             ; preds = %4929, %._crit_edge70.2.11
  %4930 = phi i32 [ 0, %._crit_edge70.2.11 ], [ %4935, %4929 ]
  %4931 = zext i32 %4930 to i64
  %4932 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4931
  %4933 = load i32, i32* %4932, align 4, !noalias !632
  %4934 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4931
  store i32 %4933, i32* %4934, align 4, !alias.scope !632
  %4935 = add nuw nsw i32 %4930, 1, !spirv.Decorations !620
  %4936 = icmp eq i32 %4930, 0
  br i1 %4936, label %4929, label %4937, !llvm.loop !644

4937:                                             ; preds = %4929
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4938 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4939 = and i1 %3552, %4835
  br i1 %4939, label %4940, label %.preheader1.11

4940:                                             ; preds = %4937
  %4941 = shl i64 %4938, 32
  %4942 = ashr exact i64 %4941, 32
  %4943 = ashr i64 %4938, 32
  %4944 = mul nsw i64 %4942, %const_reg_qword9, !spirv.Decorations !610
  %4945 = add nsw i64 %4944, %4943, !spirv.Decorations !610
  %4946 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !612
  br i1 %48, label %4950, label %4947

4947:                                             ; preds = %4940
  %4948 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4945
  %4949 = addrspacecast float addrspace(4)* %4948 to float addrspace(1)*
  store float %4946, float addrspace(1)* %4949, align 4
  br label %.preheader1.11

4950:                                             ; preds = %4940
  %4951 = mul nsw i64 %4942, %const_reg_qword7, !spirv.Decorations !610
  %4952 = getelementptr float, float addrspace(4)* %66, i64 %4951
  %4953 = getelementptr float, float addrspace(4)* %4952, i64 %4943
  %4954 = addrspacecast float addrspace(4)* %4953 to float addrspace(1)*
  %4955 = load float, float addrspace(1)* %4954, align 4
  %4956 = fmul reassoc nsz arcp contract float %4955, %4, !spirv.Decorations !612
  %4957 = fadd reassoc nsz arcp contract float %4946, %4956, !spirv.Decorations !612
  %4958 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4945
  %4959 = addrspacecast float addrspace(4)* %4958 to float addrspace(1)*
  store float %4957, float addrspace(1)* %4959, align 4
  br label %.preheader1.11

.preheader1.11:                                   ; preds = %4937, %4950, %4947
  %4960 = or i32 %41, 12
  %4961 = icmp slt i32 %4960, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %4960, i32* %60, align 4, !noalias !632
  br label %4962

4962:                                             ; preds = %4962, %.preheader1.11
  %4963 = phi i32 [ 0, %.preheader1.11 ], [ %4968, %4962 ]
  %4964 = zext i32 %4963 to i64
  %4965 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4964
  %4966 = load i32, i32* %4965, align 4, !noalias !632
  %4967 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4964
  store i32 %4966, i32* %4967, align 4, !alias.scope !632
  %4968 = add nuw nsw i32 %4963, 1, !spirv.Decorations !620
  %4969 = icmp eq i32 %4963, 0
  br i1 %4969, label %4962, label %4970, !llvm.loop !644

4970:                                             ; preds = %4962
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %4971 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %4972 = and i1 %3453, %4961
  br i1 %4972, label %4973, label %._crit_edge70.12

4973:                                             ; preds = %4970
  %4974 = shl i64 %4971, 32
  %4975 = ashr exact i64 %4974, 32
  %4976 = ashr i64 %4971, 32
  %4977 = mul nsw i64 %4975, %const_reg_qword9, !spirv.Decorations !610
  %4978 = add nsw i64 %4977, %4976, !spirv.Decorations !610
  %4979 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !612
  br i1 %48, label %4983, label %4980

4980:                                             ; preds = %4973
  %4981 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4978
  %4982 = addrspacecast float addrspace(4)* %4981 to float addrspace(1)*
  store float %4979, float addrspace(1)* %4982, align 4
  br label %._crit_edge70.12

4983:                                             ; preds = %4973
  %4984 = mul nsw i64 %4975, %const_reg_qword7, !spirv.Decorations !610
  %4985 = getelementptr float, float addrspace(4)* %66, i64 %4984
  %4986 = getelementptr float, float addrspace(4)* %4985, i64 %4976
  %4987 = addrspacecast float addrspace(4)* %4986 to float addrspace(1)*
  %4988 = load float, float addrspace(1)* %4987, align 4
  %4989 = fmul reassoc nsz arcp contract float %4988, %4, !spirv.Decorations !612
  %4990 = fadd reassoc nsz arcp contract float %4979, %4989, !spirv.Decorations !612
  %4991 = getelementptr inbounds float, float addrspace(4)* %65, i64 %4978
  %4992 = addrspacecast float addrspace(4)* %4991 to float addrspace(1)*
  store float %4990, float addrspace(1)* %4992, align 4
  br label %._crit_edge70.12

._crit_edge70.12:                                 ; preds = %4970, %4983, %4980
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %4960, i32* %60, align 4, !noalias !632
  br label %4993

4993:                                             ; preds = %4993, %._crit_edge70.12
  %4994 = phi i32 [ 0, %._crit_edge70.12 ], [ %4999, %4993 ]
  %4995 = zext i32 %4994 to i64
  %4996 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %4995
  %4997 = load i32, i32* %4996, align 4, !noalias !632
  %4998 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %4995
  store i32 %4997, i32* %4998, align 4, !alias.scope !632
  %4999 = add nuw nsw i32 %4994, 1, !spirv.Decorations !620
  %5000 = icmp eq i32 %4994, 0
  br i1 %5000, label %4993, label %5001, !llvm.loop !644

5001:                                             ; preds = %4993
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5002 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5003 = and i1 %3486, %4961
  br i1 %5003, label %5004, label %._crit_edge70.1.12

5004:                                             ; preds = %5001
  %5005 = shl i64 %5002, 32
  %5006 = ashr exact i64 %5005, 32
  %5007 = ashr i64 %5002, 32
  %5008 = mul nsw i64 %5006, %const_reg_qword9, !spirv.Decorations !610
  %5009 = add nsw i64 %5008, %5007, !spirv.Decorations !610
  %5010 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !612
  br i1 %48, label %5014, label %5011

5011:                                             ; preds = %5004
  %5012 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5009
  %5013 = addrspacecast float addrspace(4)* %5012 to float addrspace(1)*
  store float %5010, float addrspace(1)* %5013, align 4
  br label %._crit_edge70.1.12

5014:                                             ; preds = %5004
  %5015 = mul nsw i64 %5006, %const_reg_qword7, !spirv.Decorations !610
  %5016 = getelementptr float, float addrspace(4)* %66, i64 %5015
  %5017 = getelementptr float, float addrspace(4)* %5016, i64 %5007
  %5018 = addrspacecast float addrspace(4)* %5017 to float addrspace(1)*
  %5019 = load float, float addrspace(1)* %5018, align 4
  %5020 = fmul reassoc nsz arcp contract float %5019, %4, !spirv.Decorations !612
  %5021 = fadd reassoc nsz arcp contract float %5010, %5020, !spirv.Decorations !612
  %5022 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5009
  %5023 = addrspacecast float addrspace(4)* %5022 to float addrspace(1)*
  store float %5021, float addrspace(1)* %5023, align 4
  br label %._crit_edge70.1.12

._crit_edge70.1.12:                               ; preds = %5001, %5014, %5011
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %4960, i32* %60, align 4, !noalias !632
  br label %5024

5024:                                             ; preds = %5024, %._crit_edge70.1.12
  %5025 = phi i32 [ 0, %._crit_edge70.1.12 ], [ %5030, %5024 ]
  %5026 = zext i32 %5025 to i64
  %5027 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5026
  %5028 = load i32, i32* %5027, align 4, !noalias !632
  %5029 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5026
  store i32 %5028, i32* %5029, align 4, !alias.scope !632
  %5030 = add nuw nsw i32 %5025, 1, !spirv.Decorations !620
  %5031 = icmp eq i32 %5025, 0
  br i1 %5031, label %5024, label %5032, !llvm.loop !644

5032:                                             ; preds = %5024
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5033 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5034 = and i1 %3519, %4961
  br i1 %5034, label %5035, label %._crit_edge70.2.12

5035:                                             ; preds = %5032
  %5036 = shl i64 %5033, 32
  %5037 = ashr exact i64 %5036, 32
  %5038 = ashr i64 %5033, 32
  %5039 = mul nsw i64 %5037, %const_reg_qword9, !spirv.Decorations !610
  %5040 = add nsw i64 %5039, %5038, !spirv.Decorations !610
  %5041 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !612
  br i1 %48, label %5045, label %5042

5042:                                             ; preds = %5035
  %5043 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5040
  %5044 = addrspacecast float addrspace(4)* %5043 to float addrspace(1)*
  store float %5041, float addrspace(1)* %5044, align 4
  br label %._crit_edge70.2.12

5045:                                             ; preds = %5035
  %5046 = mul nsw i64 %5037, %const_reg_qword7, !spirv.Decorations !610
  %5047 = getelementptr float, float addrspace(4)* %66, i64 %5046
  %5048 = getelementptr float, float addrspace(4)* %5047, i64 %5038
  %5049 = addrspacecast float addrspace(4)* %5048 to float addrspace(1)*
  %5050 = load float, float addrspace(1)* %5049, align 4
  %5051 = fmul reassoc nsz arcp contract float %5050, %4, !spirv.Decorations !612
  %5052 = fadd reassoc nsz arcp contract float %5041, %5051, !spirv.Decorations !612
  %5053 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5040
  %5054 = addrspacecast float addrspace(4)* %5053 to float addrspace(1)*
  store float %5052, float addrspace(1)* %5054, align 4
  br label %._crit_edge70.2.12

._crit_edge70.2.12:                               ; preds = %5032, %5045, %5042
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %4960, i32* %60, align 4, !noalias !632
  br label %5055

5055:                                             ; preds = %5055, %._crit_edge70.2.12
  %5056 = phi i32 [ 0, %._crit_edge70.2.12 ], [ %5061, %5055 ]
  %5057 = zext i32 %5056 to i64
  %5058 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5057
  %5059 = load i32, i32* %5058, align 4, !noalias !632
  %5060 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5057
  store i32 %5059, i32* %5060, align 4, !alias.scope !632
  %5061 = add nuw nsw i32 %5056, 1, !spirv.Decorations !620
  %5062 = icmp eq i32 %5056, 0
  br i1 %5062, label %5055, label %5063, !llvm.loop !644

5063:                                             ; preds = %5055
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5064 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5065 = and i1 %3552, %4961
  br i1 %5065, label %5066, label %.preheader1.12

5066:                                             ; preds = %5063
  %5067 = shl i64 %5064, 32
  %5068 = ashr exact i64 %5067, 32
  %5069 = ashr i64 %5064, 32
  %5070 = mul nsw i64 %5068, %const_reg_qword9, !spirv.Decorations !610
  %5071 = add nsw i64 %5070, %5069, !spirv.Decorations !610
  %5072 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !612
  br i1 %48, label %5076, label %5073

5073:                                             ; preds = %5066
  %5074 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5071
  %5075 = addrspacecast float addrspace(4)* %5074 to float addrspace(1)*
  store float %5072, float addrspace(1)* %5075, align 4
  br label %.preheader1.12

5076:                                             ; preds = %5066
  %5077 = mul nsw i64 %5068, %const_reg_qword7, !spirv.Decorations !610
  %5078 = getelementptr float, float addrspace(4)* %66, i64 %5077
  %5079 = getelementptr float, float addrspace(4)* %5078, i64 %5069
  %5080 = addrspacecast float addrspace(4)* %5079 to float addrspace(1)*
  %5081 = load float, float addrspace(1)* %5080, align 4
  %5082 = fmul reassoc nsz arcp contract float %5081, %4, !spirv.Decorations !612
  %5083 = fadd reassoc nsz arcp contract float %5072, %5082, !spirv.Decorations !612
  %5084 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5071
  %5085 = addrspacecast float addrspace(4)* %5084 to float addrspace(1)*
  store float %5083, float addrspace(1)* %5085, align 4
  br label %.preheader1.12

.preheader1.12:                                   ; preds = %5063, %5076, %5073
  %5086 = or i32 %41, 13
  %5087 = icmp slt i32 %5086, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %5086, i32* %60, align 4, !noalias !632
  br label %5088

5088:                                             ; preds = %5088, %.preheader1.12
  %5089 = phi i32 [ 0, %.preheader1.12 ], [ %5094, %5088 ]
  %5090 = zext i32 %5089 to i64
  %5091 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5090
  %5092 = load i32, i32* %5091, align 4, !noalias !632
  %5093 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5090
  store i32 %5092, i32* %5093, align 4, !alias.scope !632
  %5094 = add nuw nsw i32 %5089, 1, !spirv.Decorations !620
  %5095 = icmp eq i32 %5089, 0
  br i1 %5095, label %5088, label %5096, !llvm.loop !644

5096:                                             ; preds = %5088
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5097 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5098 = and i1 %3453, %5087
  br i1 %5098, label %5099, label %._crit_edge70.13

5099:                                             ; preds = %5096
  %5100 = shl i64 %5097, 32
  %5101 = ashr exact i64 %5100, 32
  %5102 = ashr i64 %5097, 32
  %5103 = mul nsw i64 %5101, %const_reg_qword9, !spirv.Decorations !610
  %5104 = add nsw i64 %5103, %5102, !spirv.Decorations !610
  %5105 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !612
  br i1 %48, label %5109, label %5106

5106:                                             ; preds = %5099
  %5107 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5104
  %5108 = addrspacecast float addrspace(4)* %5107 to float addrspace(1)*
  store float %5105, float addrspace(1)* %5108, align 4
  br label %._crit_edge70.13

5109:                                             ; preds = %5099
  %5110 = mul nsw i64 %5101, %const_reg_qword7, !spirv.Decorations !610
  %5111 = getelementptr float, float addrspace(4)* %66, i64 %5110
  %5112 = getelementptr float, float addrspace(4)* %5111, i64 %5102
  %5113 = addrspacecast float addrspace(4)* %5112 to float addrspace(1)*
  %5114 = load float, float addrspace(1)* %5113, align 4
  %5115 = fmul reassoc nsz arcp contract float %5114, %4, !spirv.Decorations !612
  %5116 = fadd reassoc nsz arcp contract float %5105, %5115, !spirv.Decorations !612
  %5117 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5104
  %5118 = addrspacecast float addrspace(4)* %5117 to float addrspace(1)*
  store float %5116, float addrspace(1)* %5118, align 4
  br label %._crit_edge70.13

._crit_edge70.13:                                 ; preds = %5096, %5109, %5106
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %5086, i32* %60, align 4, !noalias !632
  br label %5119

5119:                                             ; preds = %5119, %._crit_edge70.13
  %5120 = phi i32 [ 0, %._crit_edge70.13 ], [ %5125, %5119 ]
  %5121 = zext i32 %5120 to i64
  %5122 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5121
  %5123 = load i32, i32* %5122, align 4, !noalias !632
  %5124 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5121
  store i32 %5123, i32* %5124, align 4, !alias.scope !632
  %5125 = add nuw nsw i32 %5120, 1, !spirv.Decorations !620
  %5126 = icmp eq i32 %5120, 0
  br i1 %5126, label %5119, label %5127, !llvm.loop !644

5127:                                             ; preds = %5119
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5128 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5129 = and i1 %3486, %5087
  br i1 %5129, label %5130, label %._crit_edge70.1.13

5130:                                             ; preds = %5127
  %5131 = shl i64 %5128, 32
  %5132 = ashr exact i64 %5131, 32
  %5133 = ashr i64 %5128, 32
  %5134 = mul nsw i64 %5132, %const_reg_qword9, !spirv.Decorations !610
  %5135 = add nsw i64 %5134, %5133, !spirv.Decorations !610
  %5136 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !612
  br i1 %48, label %5140, label %5137

5137:                                             ; preds = %5130
  %5138 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5135
  %5139 = addrspacecast float addrspace(4)* %5138 to float addrspace(1)*
  store float %5136, float addrspace(1)* %5139, align 4
  br label %._crit_edge70.1.13

5140:                                             ; preds = %5130
  %5141 = mul nsw i64 %5132, %const_reg_qword7, !spirv.Decorations !610
  %5142 = getelementptr float, float addrspace(4)* %66, i64 %5141
  %5143 = getelementptr float, float addrspace(4)* %5142, i64 %5133
  %5144 = addrspacecast float addrspace(4)* %5143 to float addrspace(1)*
  %5145 = load float, float addrspace(1)* %5144, align 4
  %5146 = fmul reassoc nsz arcp contract float %5145, %4, !spirv.Decorations !612
  %5147 = fadd reassoc nsz arcp contract float %5136, %5146, !spirv.Decorations !612
  %5148 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5135
  %5149 = addrspacecast float addrspace(4)* %5148 to float addrspace(1)*
  store float %5147, float addrspace(1)* %5149, align 4
  br label %._crit_edge70.1.13

._crit_edge70.1.13:                               ; preds = %5127, %5140, %5137
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %5086, i32* %60, align 4, !noalias !632
  br label %5150

5150:                                             ; preds = %5150, %._crit_edge70.1.13
  %5151 = phi i32 [ 0, %._crit_edge70.1.13 ], [ %5156, %5150 ]
  %5152 = zext i32 %5151 to i64
  %5153 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5152
  %5154 = load i32, i32* %5153, align 4, !noalias !632
  %5155 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5152
  store i32 %5154, i32* %5155, align 4, !alias.scope !632
  %5156 = add nuw nsw i32 %5151, 1, !spirv.Decorations !620
  %5157 = icmp eq i32 %5151, 0
  br i1 %5157, label %5150, label %5158, !llvm.loop !644

5158:                                             ; preds = %5150
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5159 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5160 = and i1 %3519, %5087
  br i1 %5160, label %5161, label %._crit_edge70.2.13

5161:                                             ; preds = %5158
  %5162 = shl i64 %5159, 32
  %5163 = ashr exact i64 %5162, 32
  %5164 = ashr i64 %5159, 32
  %5165 = mul nsw i64 %5163, %const_reg_qword9, !spirv.Decorations !610
  %5166 = add nsw i64 %5165, %5164, !spirv.Decorations !610
  %5167 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !612
  br i1 %48, label %5171, label %5168

5168:                                             ; preds = %5161
  %5169 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5166
  %5170 = addrspacecast float addrspace(4)* %5169 to float addrspace(1)*
  store float %5167, float addrspace(1)* %5170, align 4
  br label %._crit_edge70.2.13

5171:                                             ; preds = %5161
  %5172 = mul nsw i64 %5163, %const_reg_qword7, !spirv.Decorations !610
  %5173 = getelementptr float, float addrspace(4)* %66, i64 %5172
  %5174 = getelementptr float, float addrspace(4)* %5173, i64 %5164
  %5175 = addrspacecast float addrspace(4)* %5174 to float addrspace(1)*
  %5176 = load float, float addrspace(1)* %5175, align 4
  %5177 = fmul reassoc nsz arcp contract float %5176, %4, !spirv.Decorations !612
  %5178 = fadd reassoc nsz arcp contract float %5167, %5177, !spirv.Decorations !612
  %5179 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5166
  %5180 = addrspacecast float addrspace(4)* %5179 to float addrspace(1)*
  store float %5178, float addrspace(1)* %5180, align 4
  br label %._crit_edge70.2.13

._crit_edge70.2.13:                               ; preds = %5158, %5171, %5168
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %5086, i32* %60, align 4, !noalias !632
  br label %5181

5181:                                             ; preds = %5181, %._crit_edge70.2.13
  %5182 = phi i32 [ 0, %._crit_edge70.2.13 ], [ %5187, %5181 ]
  %5183 = zext i32 %5182 to i64
  %5184 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5183
  %5185 = load i32, i32* %5184, align 4, !noalias !632
  %5186 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5183
  store i32 %5185, i32* %5186, align 4, !alias.scope !632
  %5187 = add nuw nsw i32 %5182, 1, !spirv.Decorations !620
  %5188 = icmp eq i32 %5182, 0
  br i1 %5188, label %5181, label %5189, !llvm.loop !644

5189:                                             ; preds = %5181
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5190 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5191 = and i1 %3552, %5087
  br i1 %5191, label %5192, label %.preheader1.13

5192:                                             ; preds = %5189
  %5193 = shl i64 %5190, 32
  %5194 = ashr exact i64 %5193, 32
  %5195 = ashr i64 %5190, 32
  %5196 = mul nsw i64 %5194, %const_reg_qword9, !spirv.Decorations !610
  %5197 = add nsw i64 %5196, %5195, !spirv.Decorations !610
  %5198 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !612
  br i1 %48, label %5202, label %5199

5199:                                             ; preds = %5192
  %5200 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5197
  %5201 = addrspacecast float addrspace(4)* %5200 to float addrspace(1)*
  store float %5198, float addrspace(1)* %5201, align 4
  br label %.preheader1.13

5202:                                             ; preds = %5192
  %5203 = mul nsw i64 %5194, %const_reg_qword7, !spirv.Decorations !610
  %5204 = getelementptr float, float addrspace(4)* %66, i64 %5203
  %5205 = getelementptr float, float addrspace(4)* %5204, i64 %5195
  %5206 = addrspacecast float addrspace(4)* %5205 to float addrspace(1)*
  %5207 = load float, float addrspace(1)* %5206, align 4
  %5208 = fmul reassoc nsz arcp contract float %5207, %4, !spirv.Decorations !612
  %5209 = fadd reassoc nsz arcp contract float %5198, %5208, !spirv.Decorations !612
  %5210 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5197
  %5211 = addrspacecast float addrspace(4)* %5210 to float addrspace(1)*
  store float %5209, float addrspace(1)* %5211, align 4
  br label %.preheader1.13

.preheader1.13:                                   ; preds = %5189, %5202, %5199
  %5212 = or i32 %41, 14
  %5213 = icmp slt i32 %5212, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %5212, i32* %60, align 4, !noalias !632
  br label %5214

5214:                                             ; preds = %5214, %.preheader1.13
  %5215 = phi i32 [ 0, %.preheader1.13 ], [ %5220, %5214 ]
  %5216 = zext i32 %5215 to i64
  %5217 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5216
  %5218 = load i32, i32* %5217, align 4, !noalias !632
  %5219 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5216
  store i32 %5218, i32* %5219, align 4, !alias.scope !632
  %5220 = add nuw nsw i32 %5215, 1, !spirv.Decorations !620
  %5221 = icmp eq i32 %5215, 0
  br i1 %5221, label %5214, label %5222, !llvm.loop !644

5222:                                             ; preds = %5214
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5223 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5224 = and i1 %3453, %5213
  br i1 %5224, label %5225, label %._crit_edge70.14

5225:                                             ; preds = %5222
  %5226 = shl i64 %5223, 32
  %5227 = ashr exact i64 %5226, 32
  %5228 = ashr i64 %5223, 32
  %5229 = mul nsw i64 %5227, %const_reg_qword9, !spirv.Decorations !610
  %5230 = add nsw i64 %5229, %5228, !spirv.Decorations !610
  %5231 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !612
  br i1 %48, label %5235, label %5232

5232:                                             ; preds = %5225
  %5233 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5230
  %5234 = addrspacecast float addrspace(4)* %5233 to float addrspace(1)*
  store float %5231, float addrspace(1)* %5234, align 4
  br label %._crit_edge70.14

5235:                                             ; preds = %5225
  %5236 = mul nsw i64 %5227, %const_reg_qword7, !spirv.Decorations !610
  %5237 = getelementptr float, float addrspace(4)* %66, i64 %5236
  %5238 = getelementptr float, float addrspace(4)* %5237, i64 %5228
  %5239 = addrspacecast float addrspace(4)* %5238 to float addrspace(1)*
  %5240 = load float, float addrspace(1)* %5239, align 4
  %5241 = fmul reassoc nsz arcp contract float %5240, %4, !spirv.Decorations !612
  %5242 = fadd reassoc nsz arcp contract float %5231, %5241, !spirv.Decorations !612
  %5243 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5230
  %5244 = addrspacecast float addrspace(4)* %5243 to float addrspace(1)*
  store float %5242, float addrspace(1)* %5244, align 4
  br label %._crit_edge70.14

._crit_edge70.14:                                 ; preds = %5222, %5235, %5232
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %5212, i32* %60, align 4, !noalias !632
  br label %5245

5245:                                             ; preds = %5245, %._crit_edge70.14
  %5246 = phi i32 [ 0, %._crit_edge70.14 ], [ %5251, %5245 ]
  %5247 = zext i32 %5246 to i64
  %5248 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5247
  %5249 = load i32, i32* %5248, align 4, !noalias !632
  %5250 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5247
  store i32 %5249, i32* %5250, align 4, !alias.scope !632
  %5251 = add nuw nsw i32 %5246, 1, !spirv.Decorations !620
  %5252 = icmp eq i32 %5246, 0
  br i1 %5252, label %5245, label %5253, !llvm.loop !644

5253:                                             ; preds = %5245
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5254 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5255 = and i1 %3486, %5213
  br i1 %5255, label %5256, label %._crit_edge70.1.14

5256:                                             ; preds = %5253
  %5257 = shl i64 %5254, 32
  %5258 = ashr exact i64 %5257, 32
  %5259 = ashr i64 %5254, 32
  %5260 = mul nsw i64 %5258, %const_reg_qword9, !spirv.Decorations !610
  %5261 = add nsw i64 %5260, %5259, !spirv.Decorations !610
  %5262 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !612
  br i1 %48, label %5266, label %5263

5263:                                             ; preds = %5256
  %5264 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5261
  %5265 = addrspacecast float addrspace(4)* %5264 to float addrspace(1)*
  store float %5262, float addrspace(1)* %5265, align 4
  br label %._crit_edge70.1.14

5266:                                             ; preds = %5256
  %5267 = mul nsw i64 %5258, %const_reg_qword7, !spirv.Decorations !610
  %5268 = getelementptr float, float addrspace(4)* %66, i64 %5267
  %5269 = getelementptr float, float addrspace(4)* %5268, i64 %5259
  %5270 = addrspacecast float addrspace(4)* %5269 to float addrspace(1)*
  %5271 = load float, float addrspace(1)* %5270, align 4
  %5272 = fmul reassoc nsz arcp contract float %5271, %4, !spirv.Decorations !612
  %5273 = fadd reassoc nsz arcp contract float %5262, %5272, !spirv.Decorations !612
  %5274 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5261
  %5275 = addrspacecast float addrspace(4)* %5274 to float addrspace(1)*
  store float %5273, float addrspace(1)* %5275, align 4
  br label %._crit_edge70.1.14

._crit_edge70.1.14:                               ; preds = %5253, %5266, %5263
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %5212, i32* %60, align 4, !noalias !632
  br label %5276

5276:                                             ; preds = %5276, %._crit_edge70.1.14
  %5277 = phi i32 [ 0, %._crit_edge70.1.14 ], [ %5282, %5276 ]
  %5278 = zext i32 %5277 to i64
  %5279 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5278
  %5280 = load i32, i32* %5279, align 4, !noalias !632
  %5281 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5278
  store i32 %5280, i32* %5281, align 4, !alias.scope !632
  %5282 = add nuw nsw i32 %5277, 1, !spirv.Decorations !620
  %5283 = icmp eq i32 %5277, 0
  br i1 %5283, label %5276, label %5284, !llvm.loop !644

5284:                                             ; preds = %5276
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5285 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5286 = and i1 %3519, %5213
  br i1 %5286, label %5287, label %._crit_edge70.2.14

5287:                                             ; preds = %5284
  %5288 = shl i64 %5285, 32
  %5289 = ashr exact i64 %5288, 32
  %5290 = ashr i64 %5285, 32
  %5291 = mul nsw i64 %5289, %const_reg_qword9, !spirv.Decorations !610
  %5292 = add nsw i64 %5291, %5290, !spirv.Decorations !610
  %5293 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !612
  br i1 %48, label %5297, label %5294

5294:                                             ; preds = %5287
  %5295 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5292
  %5296 = addrspacecast float addrspace(4)* %5295 to float addrspace(1)*
  store float %5293, float addrspace(1)* %5296, align 4
  br label %._crit_edge70.2.14

5297:                                             ; preds = %5287
  %5298 = mul nsw i64 %5289, %const_reg_qword7, !spirv.Decorations !610
  %5299 = getelementptr float, float addrspace(4)* %66, i64 %5298
  %5300 = getelementptr float, float addrspace(4)* %5299, i64 %5290
  %5301 = addrspacecast float addrspace(4)* %5300 to float addrspace(1)*
  %5302 = load float, float addrspace(1)* %5301, align 4
  %5303 = fmul reassoc nsz arcp contract float %5302, %4, !spirv.Decorations !612
  %5304 = fadd reassoc nsz arcp contract float %5293, %5303, !spirv.Decorations !612
  %5305 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5292
  %5306 = addrspacecast float addrspace(4)* %5305 to float addrspace(1)*
  store float %5304, float addrspace(1)* %5306, align 4
  br label %._crit_edge70.2.14

._crit_edge70.2.14:                               ; preds = %5284, %5297, %5294
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %5212, i32* %60, align 4, !noalias !632
  br label %5307

5307:                                             ; preds = %5307, %._crit_edge70.2.14
  %5308 = phi i32 [ 0, %._crit_edge70.2.14 ], [ %5313, %5307 ]
  %5309 = zext i32 %5308 to i64
  %5310 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5309
  %5311 = load i32, i32* %5310, align 4, !noalias !632
  %5312 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5309
  store i32 %5311, i32* %5312, align 4, !alias.scope !632
  %5313 = add nuw nsw i32 %5308, 1, !spirv.Decorations !620
  %5314 = icmp eq i32 %5308, 0
  br i1 %5314, label %5307, label %5315, !llvm.loop !644

5315:                                             ; preds = %5307
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5316 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5317 = and i1 %3552, %5213
  br i1 %5317, label %5318, label %.preheader1.14

5318:                                             ; preds = %5315
  %5319 = shl i64 %5316, 32
  %5320 = ashr exact i64 %5319, 32
  %5321 = ashr i64 %5316, 32
  %5322 = mul nsw i64 %5320, %const_reg_qword9, !spirv.Decorations !610
  %5323 = add nsw i64 %5322, %5321, !spirv.Decorations !610
  %5324 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !612
  br i1 %48, label %5328, label %5325

5325:                                             ; preds = %5318
  %5326 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5323
  %5327 = addrspacecast float addrspace(4)* %5326 to float addrspace(1)*
  store float %5324, float addrspace(1)* %5327, align 4
  br label %.preheader1.14

5328:                                             ; preds = %5318
  %5329 = mul nsw i64 %5320, %const_reg_qword7, !spirv.Decorations !610
  %5330 = getelementptr float, float addrspace(4)* %66, i64 %5329
  %5331 = getelementptr float, float addrspace(4)* %5330, i64 %5321
  %5332 = addrspacecast float addrspace(4)* %5331 to float addrspace(1)*
  %5333 = load float, float addrspace(1)* %5332, align 4
  %5334 = fmul reassoc nsz arcp contract float %5333, %4, !spirv.Decorations !612
  %5335 = fadd reassoc nsz arcp contract float %5324, %5334, !spirv.Decorations !612
  %5336 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5323
  %5337 = addrspacecast float addrspace(4)* %5336 to float addrspace(1)*
  store float %5335, float addrspace(1)* %5337, align 4
  br label %.preheader1.14

.preheader1.14:                                   ; preds = %5315, %5328, %5325
  %5338 = or i32 %41, 15
  %5339 = icmp slt i32 %5338, %const_reg_dword1
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %35, i32* %73, align 4, !noalias !632
  store i32 %5338, i32* %60, align 4, !noalias !632
  br label %5340

5340:                                             ; preds = %5340, %.preheader1.14
  %5341 = phi i32 [ 0, %.preheader1.14 ], [ %5346, %5340 ]
  %5342 = zext i32 %5341 to i64
  %5343 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5342
  %5344 = load i32, i32* %5343, align 4, !noalias !632
  %5345 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5342
  store i32 %5344, i32* %5345, align 4, !alias.scope !632
  %5346 = add nuw nsw i32 %5341, 1, !spirv.Decorations !620
  %5347 = icmp eq i32 %5341, 0
  br i1 %5347, label %5340, label %5348, !llvm.loop !644

5348:                                             ; preds = %5340
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5349 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5350 = and i1 %3453, %5339
  br i1 %5350, label %5351, label %._crit_edge70.15

5351:                                             ; preds = %5348
  %5352 = shl i64 %5349, 32
  %5353 = ashr exact i64 %5352, 32
  %5354 = ashr i64 %5349, 32
  %5355 = mul nsw i64 %5353, %const_reg_qword9, !spirv.Decorations !610
  %5356 = add nsw i64 %5355, %5354, !spirv.Decorations !610
  %5357 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !612
  br i1 %48, label %5361, label %5358

5358:                                             ; preds = %5351
  %5359 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5356
  %5360 = addrspacecast float addrspace(4)* %5359 to float addrspace(1)*
  store float %5357, float addrspace(1)* %5360, align 4
  br label %._crit_edge70.15

5361:                                             ; preds = %5351
  %5362 = mul nsw i64 %5353, %const_reg_qword7, !spirv.Decorations !610
  %5363 = getelementptr float, float addrspace(4)* %66, i64 %5362
  %5364 = getelementptr float, float addrspace(4)* %5363, i64 %5354
  %5365 = addrspacecast float addrspace(4)* %5364 to float addrspace(1)*
  %5366 = load float, float addrspace(1)* %5365, align 4
  %5367 = fmul reassoc nsz arcp contract float %5366, %4, !spirv.Decorations !612
  %5368 = fadd reassoc nsz arcp contract float %5357, %5367, !spirv.Decorations !612
  %5369 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5356
  %5370 = addrspacecast float addrspace(4)* %5369 to float addrspace(1)*
  store float %5368, float addrspace(1)* %5370, align 4
  br label %._crit_edge70.15

._crit_edge70.15:                                 ; preds = %5348, %5361, %5358
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3475, i32* %73, align 4, !noalias !632
  store i32 %5338, i32* %60, align 4, !noalias !632
  br label %5371

5371:                                             ; preds = %5371, %._crit_edge70.15
  %5372 = phi i32 [ 0, %._crit_edge70.15 ], [ %5377, %5371 ]
  %5373 = zext i32 %5372 to i64
  %5374 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5373
  %5375 = load i32, i32* %5374, align 4, !noalias !632
  %5376 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5373
  store i32 %5375, i32* %5376, align 4, !alias.scope !632
  %5377 = add nuw nsw i32 %5372, 1, !spirv.Decorations !620
  %5378 = icmp eq i32 %5372, 0
  br i1 %5378, label %5371, label %5379, !llvm.loop !644

5379:                                             ; preds = %5371
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5380 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5381 = and i1 %3486, %5339
  br i1 %5381, label %5382, label %._crit_edge70.1.15

5382:                                             ; preds = %5379
  %5383 = shl i64 %5380, 32
  %5384 = ashr exact i64 %5383, 32
  %5385 = ashr i64 %5380, 32
  %5386 = mul nsw i64 %5384, %const_reg_qword9, !spirv.Decorations !610
  %5387 = add nsw i64 %5386, %5385, !spirv.Decorations !610
  %5388 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !612
  br i1 %48, label %5392, label %5389

5389:                                             ; preds = %5382
  %5390 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5387
  %5391 = addrspacecast float addrspace(4)* %5390 to float addrspace(1)*
  store float %5388, float addrspace(1)* %5391, align 4
  br label %._crit_edge70.1.15

5392:                                             ; preds = %5382
  %5393 = mul nsw i64 %5384, %const_reg_qword7, !spirv.Decorations !610
  %5394 = getelementptr float, float addrspace(4)* %66, i64 %5393
  %5395 = getelementptr float, float addrspace(4)* %5394, i64 %5385
  %5396 = addrspacecast float addrspace(4)* %5395 to float addrspace(1)*
  %5397 = load float, float addrspace(1)* %5396, align 4
  %5398 = fmul reassoc nsz arcp contract float %5397, %4, !spirv.Decorations !612
  %5399 = fadd reassoc nsz arcp contract float %5388, %5398, !spirv.Decorations !612
  %5400 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5387
  %5401 = addrspacecast float addrspace(4)* %5400 to float addrspace(1)*
  store float %5399, float addrspace(1)* %5401, align 4
  br label %._crit_edge70.1.15

._crit_edge70.1.15:                               ; preds = %5379, %5392, %5389
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3508, i32* %73, align 4, !noalias !632
  store i32 %5338, i32* %60, align 4, !noalias !632
  br label %5402

5402:                                             ; preds = %5402, %._crit_edge70.1.15
  %5403 = phi i32 [ 0, %._crit_edge70.1.15 ], [ %5408, %5402 ]
  %5404 = zext i32 %5403 to i64
  %5405 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5404
  %5406 = load i32, i32* %5405, align 4, !noalias !632
  %5407 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5404
  store i32 %5406, i32* %5407, align 4, !alias.scope !632
  %5408 = add nuw nsw i32 %5403, 1, !spirv.Decorations !620
  %5409 = icmp eq i32 %5403, 0
  br i1 %5409, label %5402, label %5410, !llvm.loop !644

5410:                                             ; preds = %5402
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5411 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5412 = and i1 %3519, %5339
  br i1 %5412, label %5413, label %._crit_edge70.2.15

5413:                                             ; preds = %5410
  %5414 = shl i64 %5411, 32
  %5415 = ashr exact i64 %5414, 32
  %5416 = ashr i64 %5411, 32
  %5417 = mul nsw i64 %5415, %const_reg_qword9, !spirv.Decorations !610
  %5418 = add nsw i64 %5417, %5416, !spirv.Decorations !610
  %5419 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !612
  br i1 %48, label %5423, label %5420

5420:                                             ; preds = %5413
  %5421 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5418
  %5422 = addrspacecast float addrspace(4)* %5421 to float addrspace(1)*
  store float %5419, float addrspace(1)* %5422, align 4
  br label %._crit_edge70.2.15

5423:                                             ; preds = %5413
  %5424 = mul nsw i64 %5415, %const_reg_qword7, !spirv.Decorations !610
  %5425 = getelementptr float, float addrspace(4)* %66, i64 %5424
  %5426 = getelementptr float, float addrspace(4)* %5425, i64 %5416
  %5427 = addrspacecast float addrspace(4)* %5426 to float addrspace(1)*
  %5428 = load float, float addrspace(1)* %5427, align 4
  %5429 = fmul reassoc nsz arcp contract float %5428, %4, !spirv.Decorations !612
  %5430 = fadd reassoc nsz arcp contract float %5419, %5429, !spirv.Decorations !612
  %5431 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5418
  %5432 = addrspacecast float addrspace(4)* %5431 to float addrspace(1)*
  store float %5430, float addrspace(1)* %5432, align 4
  br label %._crit_edge70.2.15

._crit_edge70.2.15:                               ; preds = %5410, %5423, %5420
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %71)
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %72)
  store i32 %3541, i32* %73, align 4, !noalias !632
  store i32 %5338, i32* %60, align 4, !noalias !632
  br label %5433

5433:                                             ; preds = %5433, %._crit_edge70.2.15
  %5434 = phi i32 [ 0, %._crit_edge70.2.15 ], [ %5439, %5433 ]
  %5435 = zext i32 %5434 to i64
  %5436 = getelementptr inbounds [2 x i32], [2 x i32]* %20, i64 0, i64 %5435
  %5437 = load i32, i32* %5436, align 4, !noalias !632
  %5438 = getelementptr inbounds [2 x i32], [2 x i32]* %61, i64 0, i64 %5435
  store i32 %5437, i32* %5438, align 4, !alias.scope !632
  %5439 = add nuw nsw i32 %5434, 1, !spirv.Decorations !620
  %5440 = icmp eq i32 %5434, 0
  br i1 %5440, label %5433, label %5441, !llvm.loop !644

5441:                                             ; preds = %5433
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %72)
  %5442 = load i64, i64* %62, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %71)
  %5443 = and i1 %3552, %5339
  br i1 %5443, label %5444, label %.preheader1.15

5444:                                             ; preds = %5441
  %5445 = shl i64 %5442, 32
  %5446 = ashr exact i64 %5445, 32
  %5447 = ashr i64 %5442, 32
  %5448 = mul nsw i64 %5446, %const_reg_qword9, !spirv.Decorations !610
  %5449 = add nsw i64 %5448, %5447, !spirv.Decorations !610
  %5450 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !612
  br i1 %48, label %5454, label %5451

5451:                                             ; preds = %5444
  %5452 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5449
  %5453 = addrspacecast float addrspace(4)* %5452 to float addrspace(1)*
  store float %5450, float addrspace(1)* %5453, align 4
  br label %.preheader1.15

5454:                                             ; preds = %5444
  %5455 = mul nsw i64 %5446, %const_reg_qword7, !spirv.Decorations !610
  %5456 = getelementptr float, float addrspace(4)* %66, i64 %5455
  %5457 = getelementptr float, float addrspace(4)* %5456, i64 %5447
  %5458 = addrspacecast float addrspace(4)* %5457 to float addrspace(1)*
  %5459 = load float, float addrspace(1)* %5458, align 4
  %5460 = fmul reassoc nsz arcp contract float %5459, %4, !spirv.Decorations !612
  %5461 = fadd reassoc nsz arcp contract float %5450, %5460, !spirv.Decorations !612
  %5462 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5449
  %5463 = addrspacecast float addrspace(4)* %5462 to float addrspace(1)*
  store float %5461, float addrspace(1)* %5463, align 4
  br label %.preheader1.15

.preheader1.15:                                   ; preds = %5441, %5454, %5451
  %5464 = zext i32 %14 to i64
  %5465 = icmp sgt i32 %14, -1
  call void @llvm.assume(i1 %5465)
  %5466 = mul nsw i64 %5464, %9, !spirv.Decorations !610
  %5467 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %68, i64 %5466
  %5468 = mul nsw i64 %5464, %10, !spirv.Decorations !610
  %5469 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %67, i64 %5468
  %5470 = mul nsw i64 %5464, %11
  %.idx = select i1 %48, i64 %5470, i64 0
  %5471 = getelementptr float, float addrspace(4)* %66, i64 %.idx
  %5472 = mul nsw i64 %5464, %12, !spirv.Decorations !610
  %5473 = getelementptr inbounds float, float addrspace(4)* %65, i64 %5472
  %5474 = add i32 %64, %14
  %5475 = icmp slt i32 %5474, %8
  br i1 %5475, label %.preheader2.preheader, label %._crit_edge72, !llvm.loop !645

._crit_edge72:                                    ; preds = %.preheader1.15, %13
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_num_groups(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #3

declare i32 @printf(i8 addrspace(2)*, ...)

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #3 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!spirv.MemoryModel = !{!0, !0}
!spirv.Source = !{!1, !1}
!spirv.Generator = !{!2, !2}
!igc.functions = !{!3, !34}
!IGCMetadata = !{!35}
!opencl.ocl.version = !{!605, !605, !605, !605, !605, !605, !605, !605, !605, !605}
!opencl.spir.version = !{!605, !605, !605, !605, !605, !605, !605, !605, !605, !605}
!llvm.ident = !{!606, !606, !606, !606, !606, !606, !606, !606, !606, !606}
!llvm.module.flags = !{!607}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !4}
!4 = !{!5, !6}
!5 = !{!"function_type", i32 0}
!6 = !{!"implicit_arg_desc", !7, !8, !9, !10, !11, !12, !13, !14, !15, !18, !20, !22, !24, !25, !27, !28, !30, !31, !33}
!7 = !{i32 0}
!8 = !{i32 2}
!9 = !{i32 4}
!10 = !{i32 6}
!11 = !{i32 8}
!12 = !{i32 9}
!13 = !{i32 10}
!14 = !{i32 13}
!15 = !{i32 18, !16, !17}
!16 = !{!"explicit_arg_num", i32 0}
!17 = !{!"struct_arg_offset", i32 0}
!18 = !{i32 18, !16, !19}
!19 = !{!"struct_arg_offset", i32 4}
!20 = !{i32 18, !16, !21}
!21 = !{!"struct_arg_offset", i32 8}
!22 = !{i32 17, !23, !17}
!23 = !{!"explicit_arg_num", i32 2}
!24 = !{i32 17, !23, !21}
!25 = !{i32 17, !26, !17}
!26 = !{!"explicit_arg_num", i32 3}
!27 = !{i32 17, !26, !21}
!28 = !{i32 17, !29, !17}
!29 = !{!"explicit_arg_num", i32 5}
!30 = !{i32 17, !29, !21}
!31 = !{i32 17, !32, !17}
!32 = !{!"explicit_arg_num", i32 6}
!33 = !{i32 17, !32, !21}
!34 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !4}
!35 = !{!"ModuleMD", !36, !37, !175, !369, !400, !422, !423, !427, !430, !431, !432, !471, !496, !510, !511, !512, !529, !530, !531, !532, !536, !537, !544, !545, !546, !547, !548, !549, !550, !551, !552, !553, !554, !555, !557, !561, !562, !563, !564, !565, !566, !567, !568, !569, !570, !571, !572, !573, !574, !575, !576, !577, !578, !579, !580, !262, !581, !582, !583, !585, !587, !590, !591, !592, !594, !595, !596, !601, !602, !603, !604}
!36 = !{!"isPrecise", i1 false}
!37 = !{!"compOpt", !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174}
!38 = !{!"DenormsAreZero", i1 false}
!39 = !{!"BFTFDenormsAreZero", i1 false}
!40 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!41 = !{!"OptDisable", i1 false}
!42 = !{!"MadEnable", i1 true}
!43 = !{!"NoSignedZeros", i1 false}
!44 = !{!"NoNaNs", i1 false}
!45 = !{!"FloatDenormMode16", !"FLOAT_DENORM_RETAIN"}
!46 = !{!"FloatDenormMode32", !"FLOAT_DENORM_RETAIN"}
!47 = !{!"FloatDenormMode64", !"FLOAT_DENORM_RETAIN"}
!48 = !{!"FloatDenormModeBFTF", !"FLOAT_DENORM_RETAIN"}
!49 = !{!"FloatRoundingMode", i32 0}
!50 = !{!"FloatCvtIntRoundingMode", i32 3}
!51 = !{!"LoadCacheDefault", i32 4}
!52 = !{!"StoreCacheDefault", i32 2}
!53 = !{!"VISAPreSchedRPThreshold", i32 0}
!54 = !{!"VISAPreSchedCtrl", i32 0}
!55 = !{!"SetLoopUnrollThreshold", i32 0}
!56 = !{!"UnsafeMathOptimizations", i1 false}
!57 = !{!"disableCustomUnsafeOpts", i1 false}
!58 = !{!"disableReducePow", i1 false}
!59 = !{!"disableSqrtOpt", i1 false}
!60 = !{!"FiniteMathOnly", i1 false}
!61 = !{!"FastRelaxedMath", i1 false}
!62 = !{!"DashGSpecified", i1 false}
!63 = !{!"FastCompilation", i1 false}
!64 = !{!"UseScratchSpacePrivateMemory", i1 true}
!65 = !{!"RelaxedBuiltins", i1 false}
!66 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!67 = !{!"GreaterThan2GBBufferRequired", i1 true}
!68 = !{!"GreaterThan4GBBufferRequired", i1 true}
!69 = !{!"DisableA64WA", i1 false}
!70 = !{!"ForceEnableA64WA", i1 false}
!71 = !{!"PushConstantsEnable", i1 true}
!72 = !{!"HasPositivePointerOffset", i1 false}
!73 = !{!"HasBufferOffsetArg", i1 true}
!74 = !{!"BufferOffsetArgOptional", i1 true}
!75 = !{!"replaceGlobalOffsetsByZero", i1 false}
!76 = !{!"forcePixelShaderSIMDMode", i32 0}
!77 = !{!"forceTotalGRFNum", i32 0}
!78 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!79 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!80 = !{!"UniformWGS", i1 false}
!81 = !{!"disableVertexComponentPacking", i1 false}
!82 = !{!"disablePartialVertexComponentPacking", i1 false}
!83 = !{!"PreferBindlessImages", i1 true}
!84 = !{!"UseBindlessMode", i1 true}
!85 = !{!"UseLegacyBindlessMode", i1 false}
!86 = !{!"disableMathRefactoring", i1 false}
!87 = !{!"atomicBranch", i1 false}
!88 = !{!"spillCompression", i1 false}
!89 = !{!"AllowLICM", i1 true}
!90 = !{!"DisableEarlyOut", i1 false}
!91 = !{!"ForceInt32DivRemEmu", i1 false}
!92 = !{!"ForceInt32DivRemEmuSP", i1 false}
!93 = !{!"DisableIntDivRemIncrementReduction", i1 false}
!94 = !{!"WaveIntrinsicUsed", i1 false}
!95 = !{!"DisableMultiPolyPS", i1 false}
!96 = !{!"NeedTexture3DLODWA", i1 false}
!97 = !{!"UseLivePrologueKernelForRaytracingDispatch", i1 false}
!98 = !{!"DisableFastestSingleCSSIMD", i1 false}
!99 = !{!"DisableFastestLinearScan", i1 false}
!100 = !{!"UseStatelessforPrivateMemory", i1 false}
!101 = !{!"EnableTakeGlobalAddress", i1 false}
!102 = !{!"IsLibraryCompilation", i1 false}
!103 = !{!"LibraryCompileSIMDSize", i32 0}
!104 = !{!"FastVISACompile", i1 false}
!105 = !{!"MatchSinCosPi", i1 false}
!106 = !{!"ExcludeIRFromZEBinary", i1 false}
!107 = !{!"EmitZeBinVISASections", i1 false}
!108 = !{!"FP64GenEmulationEnabled", i1 false}
!109 = !{!"FP64GenConvEmulationEnabled", i1 false}
!110 = !{!"allowDisableRematforCS", i1 false}
!111 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!112 = !{!"DisableCPSOmaskWA", i1 false}
!113 = !{!"DisableFastestGopt", i1 false}
!114 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!115 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!116 = !{!"DisableConstantCoalescing", i1 false}
!117 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!118 = !{!"WaEnableALTModeVisaWA", i1 false}
!119 = !{!"EnableLdStCombineforLoad", i1 false}
!120 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!121 = !{!"WaEnableAtomicWaveFusion", i1 false}
!122 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!123 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!124 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!125 = !{!"WaEnableAtomicWaveFusionPartial", i1 false}
!126 = !{!"WaEnableAtomicWaveFusionMoreDimensions", i1 false}
!127 = !{!"WaEnableAtomicWaveFusionLoop", i1 false}
!128 = !{!"WaEnableAtomicWaveFusionReturnValuePolicy", i32 0}
!129 = !{!"ForceCBThroughSampler3D", i1 false}
!130 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!131 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!132 = !{!"WaTypedAtomicBinToRawAtomicBin", i1 false}
!133 = !{!"WaRawAtomicBinToTypedAtomicBin", i1 false}
!134 = !{!"WaSampleLoadToTypedRead", i1 false}
!135 = !{!"EnableTypedBufferStoreToUntypedStore", i1 false}
!136 = !{!"WaZeroSLMBeforeUse", i1 false}
!137 = !{!"EnableEmitMoreMoviCases", i1 false}
!138 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!139 = !{!"EnableFastSampleD", i1 false}
!140 = !{!"ForceUniformBuffer", i1 false}
!141 = !{!"ForceUniformSurfaceSampler", i1 false}
!142 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!143 = !{!"NewSpillCostFunction", i1 false}
!144 = !{!"EnableVRT", i1 false}
!145 = !{!"ForceLargeGRFNum4RQ", i1 false}
!146 = !{!"Enable2xGRFRetry", i1 false}
!147 = !{!"Detect2xGRFCandidate", i1 false}
!148 = !{!"EnableURBWritesMerging", i1 true}
!149 = !{!"ForceCacheLineAlignedURBWriteMerging", i1 false}
!150 = !{!"DisableURBLayoutAlignmentToCacheLine", i1 false}
!151 = !{!"DisableEUFusion", i1 false}
!152 = !{!"DisableFDivToFMulInvOpt", i1 false}
!153 = !{!"initializePhiSampleSourceWA", i1 false}
!154 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!155 = !{!"DisableLoosenSimd32Occu", i1 false}
!156 = !{!"FastestS1Options", i32 0}
!157 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!158 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!159 = !{!"LscSamplerRouting", i32 0}
!160 = !{!"UseBarrierControlFlowOptimization", i1 false}
!161 = !{!"EnableDynamicRQManagement", i1 false}
!162 = !{!"WaDisablePayloadCoalescing", i1 false}
!163 = !{!"Quad8InputThreshold", i32 0}
!164 = !{!"UseResourceLoopUnrollNested", i1 false}
!165 = !{!"DisableLoopUnroll", i1 false}
!166 = !{!"ForcePushConstantMode", i32 0}
!167 = !{!"UseInstructionHoistingOptimization", i1 false}
!168 = !{!"DisableResourceLoopDestLifeTimeStart", i1 false}
!169 = !{!"ForceVRTGRFCeiling", i32 0}
!170 = !{!"DisableSamplerBackingByLSC", i32 0}
!171 = !{!"UseLinearScanRA", i1 false}
!172 = !{!"DisableConvertingAtomicIAddToIncDec", i1 false}
!173 = !{!"EnableInlinedCrossThreadData", i1 false}
!174 = !{!"ZeroInitRegistersBeforeExecution", i1 false}
!175 = !{!"FuncMD", !176, !177, !367, !368}
!176 = !{!"FuncMDMap[0]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!177 = !{!"FuncMDValue[0]", !178, !179, !183, !184, !185, !186, !187, !188, !189, !213, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !288, !302, !316, !330, !344, !358, !363}
!178 = !{!"localOffsets"}
!179 = !{!"workGroupWalkOrder", !180, !181, !182}
!180 = !{!"dim0", i32 0}
!181 = !{!"dim1", i32 1}
!182 = !{!"dim2", i32 2}
!183 = !{!"funcArgs"}
!184 = !{!"functionType", !"KernelFunction"}
!185 = !{!"inlineDynConstants"}
!186 = !{!"inlineDynRootConstant"}
!187 = !{!"inlineDynConstantDescTable"}
!188 = !{!"m_pInterestingConstants"}
!189 = !{!"rtInfo", !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !208, !209, !210, !211, !212}
!190 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!191 = !{!"isContinuation", i1 false}
!192 = !{!"isMonolithic", i1 false}
!193 = !{!"hasTraceRayPayload", i1 false}
!194 = !{!"hasHitAttributes", i1 false}
!195 = !{!"hasCallableData", i1 false}
!196 = !{!"ShaderStackSize", i32 0}
!197 = !{!"ShaderHash", i64 0}
!198 = !{!"ShaderName", !""}
!199 = !{!"ParentName", !""}
!200 = !{!"SlotNum", i1* null}
!201 = !{!"NOSSize", i32 0}
!202 = !{!"globalRootSignatureSize", i32 0}
!203 = !{!"Entries"}
!204 = !{!"SpillUnions"}
!205 = !{!"CustomHitAttrSizeInBytes", i32 0}
!206 = !{!"Types", !207}
!207 = !{!"FullFrameTys"}
!208 = !{!"Aliases"}
!209 = !{!"numSyncRTStacks", i32 0}
!210 = !{!"NumCoherenceHintBits", i32 0}
!211 = !{!"useSyncHWStack", i1 false}
!212 = !{!"OriginatingShaderName", !""}
!213 = !{!"resAllocMD", !214, !215, !216, !217, !253}
!214 = !{!"uavsNumType", i32 0}
!215 = !{!"srvsNumType", i32 0}
!216 = !{!"samplersNumType", i32 0}
!217 = !{!"argAllocMDList", !218, !222, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252}
!218 = !{!"argAllocMDListVec[0]", !219, !220, !221}
!219 = !{!"type", i32 0}
!220 = !{!"extensionType", i32 -1}
!221 = !{!"indexType", i32 -1}
!222 = !{!"argAllocMDListVec[1]", !219, !220, !221}
!223 = !{!"argAllocMDListVec[2]", !219, !220, !221}
!224 = !{!"argAllocMDListVec[3]", !219, !220, !221}
!225 = !{!"argAllocMDListVec[4]", !219, !220, !221}
!226 = !{!"argAllocMDListVec[5]", !219, !220, !221}
!227 = !{!"argAllocMDListVec[6]", !219, !220, !221}
!228 = !{!"argAllocMDListVec[7]", !219, !220, !221}
!229 = !{!"argAllocMDListVec[8]", !219, !220, !221}
!230 = !{!"argAllocMDListVec[9]", !219, !220, !221}
!231 = !{!"argAllocMDListVec[10]", !219, !220, !221}
!232 = !{!"argAllocMDListVec[11]", !219, !220, !221}
!233 = !{!"argAllocMDListVec[12]", !219, !220, !221}
!234 = !{!"argAllocMDListVec[13]", !219, !220, !221}
!235 = !{!"argAllocMDListVec[14]", !219, !220, !221}
!236 = !{!"argAllocMDListVec[15]", !219, !220, !221}
!237 = !{!"argAllocMDListVec[16]", !219, !220, !221}
!238 = !{!"argAllocMDListVec[17]", !219, !220, !221}
!239 = !{!"argAllocMDListVec[18]", !219, !220, !221}
!240 = !{!"argAllocMDListVec[19]", !219, !220, !221}
!241 = !{!"argAllocMDListVec[20]", !219, !220, !221}
!242 = !{!"argAllocMDListVec[21]", !219, !220, !221}
!243 = !{!"argAllocMDListVec[22]", !219, !220, !221}
!244 = !{!"argAllocMDListVec[23]", !219, !220, !221}
!245 = !{!"argAllocMDListVec[24]", !219, !220, !221}
!246 = !{!"argAllocMDListVec[25]", !219, !220, !221}
!247 = !{!"argAllocMDListVec[26]", !219, !220, !221}
!248 = !{!"argAllocMDListVec[27]", !219, !220, !221}
!249 = !{!"argAllocMDListVec[28]", !219, !220, !221}
!250 = !{!"argAllocMDListVec[29]", !219, !220, !221}
!251 = !{!"argAllocMDListVec[30]", !219, !220, !221}
!252 = !{!"argAllocMDListVec[31]", !219, !220, !221}
!253 = !{!"inlineSamplersMD"}
!254 = !{!"maxByteOffsets"}
!255 = !{!"IsInitializer", i1 false}
!256 = !{!"IsFinalizer", i1 false}
!257 = !{!"CompiledSubGroupsNumber", i32 0}
!258 = !{!"hasInlineVmeSamplers", i1 false}
!259 = !{!"localSize", i32 0}
!260 = !{!"localIDPresent", i1 false}
!261 = !{!"groupIDPresent", i1 false}
!262 = !{!"privateMemoryPerWI", i32 0}
!263 = !{!"prevFPOffset", i32 0}
!264 = !{!"globalIDPresent", i1 false}
!265 = !{!"hasSyncRTCalls", i1 false}
!266 = !{!"hasPrintfCalls", i1 false}
!267 = !{!"requireAssertBuffer", i1 false}
!268 = !{!"requireSyncBuffer", i1 false}
!269 = !{!"hasIndirectCalls", i1 false}
!270 = !{!"hasNonKernelArgLoad", i1 false}
!271 = !{!"hasNonKernelArgStore", i1 false}
!272 = !{!"hasNonKernelArgAtomic", i1 false}
!273 = !{!"UserAnnotations"}
!274 = !{!"m_OpenCLArgAddressSpaces", !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287}
!275 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 0}
!276 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!277 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!278 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!279 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 0}
!280 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!281 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!282 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 0}
!283 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!284 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!285 = !{!"m_OpenCLArgAddressSpacesVec[10]", i32 0}
!286 = !{!"m_OpenCLArgAddressSpacesVec[11]", i32 0}
!287 = !{!"m_OpenCLArgAddressSpacesVec[12]", i32 0}
!288 = !{!"m_OpenCLArgAccessQualifiers", !289, !290, !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301}
!289 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!290 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!291 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!292 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!293 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!294 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!295 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!296 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!297 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!298 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!299 = !{!"m_OpenCLArgAccessQualifiersVec[10]", !"none"}
!300 = !{!"m_OpenCLArgAccessQualifiersVec[11]", !"none"}
!301 = !{!"m_OpenCLArgAccessQualifiersVec[12]", !"none"}
!302 = !{!"m_OpenCLArgTypes", !303, !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314, !315}
!303 = !{!"m_OpenCLArgTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!304 = !{!"m_OpenCLArgTypesVec[1]", !"float"}
!305 = !{!"m_OpenCLArgTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!306 = !{!"m_OpenCLArgTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!307 = !{!"m_OpenCLArgTypesVec[4]", !"float"}
!308 = !{!"m_OpenCLArgTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!309 = !{!"m_OpenCLArgTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!310 = !{!"m_OpenCLArgTypesVec[7]", !"float"}
!311 = !{!"m_OpenCLArgTypesVec[8]", !"int"}
!312 = !{!"m_OpenCLArgTypesVec[9]", !"long"}
!313 = !{!"m_OpenCLArgTypesVec[10]", !"long"}
!314 = !{!"m_OpenCLArgTypesVec[11]", !"long"}
!315 = !{!"m_OpenCLArgTypesVec[12]", !"long"}
!316 = !{!"m_OpenCLArgBaseTypes", !317, !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329}
!317 = !{!"m_OpenCLArgBaseTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!318 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float"}
!319 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!320 = !{!"m_OpenCLArgBaseTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!321 = !{!"m_OpenCLArgBaseTypesVec[4]", !"float"}
!322 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!323 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!324 = !{!"m_OpenCLArgBaseTypesVec[7]", !"float"}
!325 = !{!"m_OpenCLArgBaseTypesVec[8]", !"int"}
!326 = !{!"m_OpenCLArgBaseTypesVec[9]", !"long"}
!327 = !{!"m_OpenCLArgBaseTypesVec[10]", !"long"}
!328 = !{!"m_OpenCLArgBaseTypesVec[11]", !"long"}
!329 = !{!"m_OpenCLArgBaseTypesVec[12]", !"long"}
!330 = !{!"m_OpenCLArgTypeQualifiers", !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343}
!331 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!332 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!333 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!334 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!335 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!336 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!337 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!338 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!339 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!340 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!341 = !{!"m_OpenCLArgTypeQualifiersVec[10]", !""}
!342 = !{!"m_OpenCLArgTypeQualifiersVec[11]", !""}
!343 = !{!"m_OpenCLArgTypeQualifiersVec[12]", !""}
!344 = !{!"m_OpenCLArgNames", !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357}
!345 = !{!"m_OpenCLArgNamesVec[0]", !""}
!346 = !{!"m_OpenCLArgNamesVec[1]", !""}
!347 = !{!"m_OpenCLArgNamesVec[2]", !""}
!348 = !{!"m_OpenCLArgNamesVec[3]", !""}
!349 = !{!"m_OpenCLArgNamesVec[4]", !""}
!350 = !{!"m_OpenCLArgNamesVec[5]", !""}
!351 = !{!"m_OpenCLArgNamesVec[6]", !""}
!352 = !{!"m_OpenCLArgNamesVec[7]", !""}
!353 = !{!"m_OpenCLArgNamesVec[8]", !""}
!354 = !{!"m_OpenCLArgNamesVec[9]", !""}
!355 = !{!"m_OpenCLArgNamesVec[10]", !""}
!356 = !{!"m_OpenCLArgNamesVec[11]", !""}
!357 = !{!"m_OpenCLArgNamesVec[12]", !""}
!358 = !{!"m_OpenCLArgScalarAsPointers", !359, !360, !361, !362}
!359 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 24}
!360 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 26}
!361 = !{!"m_OpenCLArgScalarAsPointersSet[2]", i32 28}
!362 = !{!"m_OpenCLArgScalarAsPointersSet[3]", i32 30}
!363 = !{!"m_OptsToDisablePerFunc", !364, !365, !366}
!364 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-ConstantCoalescing"}
!365 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-LowerGEPForPrivMem"}
!366 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-MergeURBWrites"}
!367 = !{!"FuncMDMap[1]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!368 = !{!"FuncMDValue[1]", !178, !179, !183, !184, !185, !186, !187, !188, !189, !213, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !288, !302, !316, !330, !344, !358, !363}
!369 = !{!"pushInfo", !370, !371, !372, !376, !377, !378, !379, !380, !381, !382, !383, !396, !397, !398, !399}
!370 = !{!"pushableAddresses"}
!371 = !{!"bindlessPushInfo"}
!372 = !{!"dynamicBufferInfo", !373, !374, !375}
!373 = !{!"firstIndex", i32 0}
!374 = !{!"numOffsets", i32 0}
!375 = !{!"forceDisabled", i1 false}
!376 = !{!"MaxNumberOfPushedBuffers", i32 0}
!377 = !{!"inlineConstantBufferSlot", i32 -1}
!378 = !{!"inlineConstantBufferOffset", i32 -1}
!379 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!380 = !{!"constants"}
!381 = !{!"inputs"}
!382 = !{!"constantReg"}
!383 = !{!"simplePushInfoArr", !384, !393, !394, !395}
!384 = !{!"simplePushInfoArrVec[0]", !385, !386, !387, !388, !389, !390, !391, !392}
!385 = !{!"cbIdx", i32 0}
!386 = !{!"pushableAddressGrfOffset", i32 -1}
!387 = !{!"pushableOffsetGrfOffset", i32 -1}
!388 = !{!"offset", i32 0}
!389 = !{!"size", i32 0}
!390 = !{!"isStateless", i1 false}
!391 = !{!"isBindless", i1 false}
!392 = !{!"simplePushLoads"}
!393 = !{!"simplePushInfoArrVec[1]", !385, !386, !387, !388, !389, !390, !391, !392}
!394 = !{!"simplePushInfoArrVec[2]", !385, !386, !387, !388, !389, !390, !391, !392}
!395 = !{!"simplePushInfoArrVec[3]", !385, !386, !387, !388, !389, !390, !391, !392}
!396 = !{!"simplePushBufferUsed", i32 0}
!397 = !{!"pushAnalysisWIInfos"}
!398 = !{!"inlineRTGlobalPtrOffset", i32 0}
!399 = !{!"rtSyncSurfPtrOffset", i32 0}
!400 = !{!"pISAInfo", !401, !402, !406, !407, !415, !419, !421}
!401 = !{!"shaderType", !"UNKNOWN"}
!402 = !{!"geometryInfo", !403, !404, !405}
!403 = !{!"needsVertexHandles", i1 false}
!404 = !{!"needsPrimitiveIDEnable", i1 false}
!405 = !{!"VertexCount", i32 0}
!406 = !{!"hullInfo", !403, !404}
!407 = !{!"pixelInfo", !408, !409, !410, !411, !412, !413, !414}
!408 = !{!"perPolyStartGrf", i32 0}
!409 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!410 = !{!"hasNonPerspBaryPlanes", i1 false}
!411 = !{!"maxPerPrimConstDataId", i32 -1}
!412 = !{!"maxSetupId", i32 -1}
!413 = !{!"hasVMask", i1 false}
!414 = !{!"PixelGRFBitmask", i32 0}
!415 = !{!"domainInfo", !416, !417, !418}
!416 = !{!"DomainPointUArgIdx", i32 -1}
!417 = !{!"DomainPointVArgIdx", i32 -1}
!418 = !{!"DomainPointWArgIdx", i32 -1}
!419 = !{!"computeInfo", !420}
!420 = !{!"EnableHWGenerateLID", i1 true}
!421 = !{!"URBOutputLength", i32 0}
!422 = !{!"WaEnableICBPromotion", i1 false}
!423 = !{!"vsInfo", !424, !425, !426}
!424 = !{!"DrawIndirectBufferIndex", i32 -1}
!425 = !{!"vertexReordering", i32 -1}
!426 = !{!"MaxNumOfOutputs", i32 0}
!427 = !{!"hsInfo", !428, !429}
!428 = !{!"numPatchAttributesPatchBaseName", !""}
!429 = !{!"numVertexAttributesPatchBaseName", !""}
!430 = !{!"dsInfo", !426}
!431 = !{!"gsInfo", !426}
!432 = !{!"psInfo", !433, !434, !435, !436, !437, !438, !439, !440, !441, !442, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !453, !454, !455, !456, !457, !458, !459, !460, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470}
!433 = !{!"BlendStateDisabledMask", i8 0}
!434 = !{!"SkipSrc0Alpha", i1 false}
!435 = !{!"DualSourceBlendingDisabled", i1 false}
!436 = !{!"ForceEnableSimd32", i1 false}
!437 = !{!"DisableSimd32WithDiscard", i1 false}
!438 = !{!"outputDepth", i1 false}
!439 = !{!"outputStencil", i1 false}
!440 = !{!"outputMask", i1 false}
!441 = !{!"blendToFillEnabled", i1 false}
!442 = !{!"forceEarlyZ", i1 false}
!443 = !{!"hasVersionedLoop", i1 false}
!444 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!445 = !{!"requestCPSizeRelevant", i1 false}
!446 = !{!"requestCPSize", i1 false}
!447 = !{!"texelMaskFastClearMode", !"Disabled"}
!448 = !{!"NumSamples", i8 0}
!449 = !{!"blendOptimizationMode"}
!450 = !{!"colorOutputMask"}
!451 = !{!"ProvokingVertexModeNosIndex", i32 0}
!452 = !{!"ProvokingVertexModeNosPatch", !""}
!453 = !{!"ProvokingVertexModeLast", !"Negative"}
!454 = !{!"VertexAttributesBypass", i1 false}
!455 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!456 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!457 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!458 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!459 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!460 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!461 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!462 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!463 = !{!"generatePatchesForRTWriteSends", i1 false}
!464 = !{!"generatePatchesForRT_BTIndex", i1 false}
!465 = !{!"forceVMask", i1 false}
!466 = !{!"isNumPerPrimAttributesSet", i1 false}
!467 = !{!"numPerPrimAttributes", i32 0}
!468 = !{!"WaDisableVRS", i1 false}
!469 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!470 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!471 = !{!"csInfo", !472, !473, !474, !475, !77, !53, !54, !476, !55, !477, !478, !479, !480, !481, !482, !483, !484, !485, !486, !487, !88, !488, !489, !490, !491, !492, !493, !494, !495}
!472 = !{!"maxWorkGroupSize", i32 0}
!473 = !{!"waveSize", i32 0}
!474 = !{!"ComputeShaderSecondCompile"}
!475 = !{!"forcedSIMDSize", i8 0}
!476 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!477 = !{!"forceSpillCompression", i1 false}
!478 = !{!"allowLowerSimd", i1 false}
!479 = !{!"disableSimd32Slicing", i1 false}
!480 = !{!"disableSplitOnSpill", i1 false}
!481 = !{!"enableNewSpillCostFunction", i1 false}
!482 = !{!"forceVISAPreSched", i1 false}
!483 = !{!"disableLocalIdOrderOptimizations", i1 false}
!484 = !{!"disableDispatchAlongY", i1 false}
!485 = !{!"neededThreadIdLayout", i1* null}
!486 = !{!"forceTileYWalk", i1 false}
!487 = !{!"atomicBranch", i32 0}
!488 = !{!"disableEarlyOut", i1 false}
!489 = !{!"walkOrderEnabled", i1 false}
!490 = !{!"walkOrderOverride", i32 0}
!491 = !{!"ResForHfPacking"}
!492 = !{!"constantFoldSimdSize", i1 false}
!493 = !{!"isNodeShader", i1 false}
!494 = !{!"threadGroupMergeSize", i32 0}
!495 = !{!"threadGroupMergeOverY", i1 false}
!496 = !{!"msInfo", !497, !498, !499, !500, !501, !502, !503, !504, !505, !506, !507, !453, !451, !508, !509, !493}
!497 = !{!"PrimitiveTopology", i32 3}
!498 = !{!"MaxNumOfPrimitives", i32 0}
!499 = !{!"MaxNumOfVertices", i32 0}
!500 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!501 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!502 = !{!"WorkGroupSize", i32 0}
!503 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!504 = !{!"IndexFormat", i32 6}
!505 = !{!"SubgroupSize", i32 0}
!506 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!507 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!508 = !{!"Is16BMUEModeAllowed", i1 false}
!509 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!510 = !{!"taskInfo", !426, !502, !503, !505}
!511 = !{!"NBarrierCnt", i32 0}
!512 = !{!"rtInfo", !513, !514, !515, !516, !517, !518, !519, !520, !521, !522, !523, !524, !525, !526, !527, !528, !209}
!513 = !{!"RayQueryAllocSizeInBytes", i32 0}
!514 = !{!"NumContinuations", i32 0}
!515 = !{!"RTAsyncStackAddrspace", i32 -1}
!516 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!517 = !{!"SWHotZoneAddrspace", i32 -1}
!518 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!519 = !{!"SWStackAddrspace", i32 -1}
!520 = !{!"SWStackSurfaceStateOffset", i1* null}
!521 = !{!"RTSyncStackAddrspace", i32 -1}
!522 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!523 = !{!"doSyncDispatchRays", i1 false}
!524 = !{!"MemStyle", !"Xe"}
!525 = !{!"GlobalDataStyle", !"Xe"}
!526 = !{!"NeedsBTD", i1 true}
!527 = !{!"SERHitObjectFullType", i1* null}
!528 = !{!"uberTileDimensions", i1* null}
!529 = !{!"CurUniqueIndirectIdx", i32 0}
!530 = !{!"inlineDynTextures"}
!531 = !{!"inlineResInfoData"}
!532 = !{!"immConstant", !533, !534, !535}
!533 = !{!"data"}
!534 = !{!"sizes"}
!535 = !{!"zeroIdxs"}
!536 = !{!"stringConstants"}
!537 = !{!"inlineBuffers", !538, !542, !543}
!538 = !{!"inlineBuffersVec[0]", !539, !540, !541}
!539 = !{!"alignment", i32 0}
!540 = !{!"allocSize", i64 0}
!541 = !{!"Buffer"}
!542 = !{!"inlineBuffersVec[1]", !539, !540, !541}
!543 = !{!"inlineBuffersVec[2]", !539, !540, !541}
!544 = !{!"GlobalPointerProgramBinaryInfos"}
!545 = !{!"ConstantPointerProgramBinaryInfos"}
!546 = !{!"GlobalBufferAddressRelocInfo"}
!547 = !{!"ConstantBufferAddressRelocInfo"}
!548 = !{!"forceLscCacheList"}
!549 = !{!"SrvMap"}
!550 = !{!"RootConstantBufferOffsetInBytes"}
!551 = !{!"RasterizerOrderedByteAddressBuffer"}
!552 = !{!"RasterizerOrderedViews"}
!553 = !{!"MinNOSPushConstantSize", i32 0}
!554 = !{!"inlineProgramScopeOffsets"}
!555 = !{!"shaderData", !556}
!556 = !{!"numReplicas", i32 0}
!557 = !{!"URBInfo", !558, !559, !560}
!558 = !{!"has64BVertexHeaderInput", i1 false}
!559 = !{!"has64BVertexHeaderOutput", i1 false}
!560 = !{!"hasVertexHeader", i1 true}
!561 = !{!"m_ForcePullModel", i1 false}
!562 = !{!"UseBindlessImage", i1 true}
!563 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!564 = !{!"enableRangeReduce", i1 false}
!565 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!566 = !{!"enableFRemToSRemOpt", i1 false}
!567 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!568 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!569 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!570 = !{!"allowMatchMadOptimizationforVS", i1 false}
!571 = !{!"disableMatchMadOptimizationForCS", i1 false}
!572 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!573 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!574 = !{!"statefulResourcesNotAliased", i1 false}
!575 = !{!"disableMixMode", i1 false}
!576 = !{!"genericAccessesResolved", i1 false}
!577 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!578 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!579 = !{!"disableSeparateScratchWA", i1 false}
!580 = !{!"enableRemoveUnusedTGMFence", i1 false}
!581 = !{!"PrivateMemoryPerFG"}
!582 = !{!"m_OptsToDisable"}
!583 = !{!"capabilities", !584}
!584 = !{!"globalVariableDecorationsINTEL", i1 false}
!585 = !{!"extensions", !586}
!586 = !{!"spvINTELBindlessImages", i1 false}
!587 = !{!"m_ShaderResourceViewMcsMask", !588, !589}
!588 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!589 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!590 = !{!"computedDepthMode", i32 0}
!591 = !{!"isHDCFastClearShader", i1 false}
!592 = !{!"argRegisterReservations", !593}
!593 = !{!"argRegisterReservationsVec[0]", i32 0}
!594 = !{!"SIMD16_SpillThreshold", i8 0}
!595 = !{!"SIMD32_SpillThreshold", i8 0}
!596 = !{!"m_CacheControlOption", !597, !598, !599, !600}
!597 = !{!"LscLoadCacheControlOverride", i8 0}
!598 = !{!"LscStoreCacheControlOverride", i8 0}
!599 = !{!"TgmLoadCacheControlOverride", i8 0}
!600 = !{!"TgmStoreCacheControlOverride", i8 0}
!601 = !{!"ModuleUsesBindless", i1 false}
!602 = !{!"predicationMap"}
!603 = !{!"lifeTimeStartMap"}
!604 = !{!"HitGroups"}
!605 = !{i32 2, i32 0}
!606 = !{!"clang version 16.0.6"}
!607 = !{i32 1, !"wchar_size", i32 4}
!608 = !{!609}
!609 = !{i32 44, i32 4}
!610 = !{!611}
!611 = !{i32 4469}
!612 = !{!613}
!613 = !{i32 40, i32 196620}
!614 = !{!615}
!615 = distinct !{!615, !616}
!616 = distinct !{!616}
!617 = !{!618}
!618 = distinct !{!618, !619}
!619 = distinct !{!619}
!620 = !{!611, !621}
!621 = !{i32 4470}
!622 = distinct !{!622, !623}
!623 = !{!"llvm.loop.unroll.disable"}
!624 = !{!625}
!625 = distinct !{!625, !626}
!626 = distinct !{!626}
!627 = distinct !{!627, !623}
!628 = !{!621}
!629 = distinct !{!629, !623}
!630 = distinct !{!630, !623}
!631 = distinct !{!631, !623}
!632 = !{!633}
!633 = distinct !{!633, !634}
!634 = distinct !{!634}
!635 = !{!636}
!636 = distinct !{!636, !637}
!637 = distinct !{!637}
!638 = distinct !{!638, !623}
!639 = !{!640}
!640 = distinct !{!640, !641}
!641 = distinct !{!641}
!642 = distinct !{!642, !623}
!643 = distinct !{!643, !623}
!644 = distinct !{!644, !623}
!645 = distinct !{!645, !623}

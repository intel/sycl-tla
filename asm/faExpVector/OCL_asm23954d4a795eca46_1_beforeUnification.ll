; ------------------------------------------------
; OCL_asm23954d4a795eca46_1_beforeUnification.ll
; LLVM major version: 16
; ------------------------------------------------
; ModuleID = '<origin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"struct.cutlass::reference::device::detail::RandomUniformFunc" = type { %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params", %"class.oneapi::mkl::rng::device::uniform", %"class.oneapi::mkl::rng::device::philox4x32x10", [4 x i8] }
%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" = type <{ i64, float, float, i32, float, float, [4 x i8] }>
%"class.oneapi::mkl::rng::device::uniform" = type { %"class.oneapi::mkl::rng::device::detail::distribution_base" }
%"class.oneapi::mkl::rng::device::detail::distribution_base" = type { float, float }
%"class.oneapi::mkl::rng::device::philox4x32x10" = type { %"class.oneapi::mkl::rng::device::detail::engine_base" }
%"class.oneapi::mkl::rng::device::detail::engine_base" = type { %"struct.oneapi::mkl::rng::device::detail::engine_state" }
%"struct.oneapi::mkl::rng::device::detail::engine_state" = type { [2 x i32], [4 x i32], i32, [4 x i32] }
%"struct.cutlass::gemm::GemmCoord" = type { %"struct.cutlass::Coord" }
%"struct.cutlass::Coord" = type { [3 x i32] }
%"class.cutlass::__generated_TensorRef" = type { i8 addrspace(1)*, %"class.sycl::_V1::range.0" }
%"class.sycl::_V1::range.0" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
%"struct.cutlass::Coord.8930" = type { [2 x i32] }
%structtype = type { i64, i64 }
%"struct.cutlass::bfloat16_t" = type { i16 }
%structtype.0 = type { i64 }

@gVar = internal global [36 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@gVar.61 = internal global [24 x i8] zeroinitializer, align 8, !spirv.Decorations !0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #2

; Function Attrs: nounwind
define internal spir_func void @_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE(%"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* align 8 %0, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* align 8 dereferenceable(28) %1) #3 !spirv.ParameterDecorations !390 {
  %3 = alloca [3 x i64], align 8, !spirv.Decorations !0
  %4 = alloca [4 x i32], align 4, !spirv.Decorations !393
  %5 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %6 = alloca [2 x i64], align 8, !spirv.Decorations !0
  %7 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 0
  %8 = bitcast %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* %7 to i8 addrspace(4)*
  %9 = bitcast %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* %1 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %8, i8 addrspace(4)* align 8 %9, i64 32, i1 false)
  %10 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params", %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* %1, i64 0, i32 2
  %11 = load float, float addrspace(4)* %10, align 4
  %12 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params", %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* %1, i64 0, i32 1
  %13 = load float, float addrspace(4)* %12, align 8
  %14 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 1
  %15 = bitcast %"class.oneapi::mkl::rng::device::uniform" addrspace(4)* %14 to float addrspace(4)*
  store float %11, float addrspace(4)* %15, align 4
  %16 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 1, i32 0, i32 1
  store float %13, float addrspace(4)* %16, align 4
  %17 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params", %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" addrspace(4)* %1, i64 0, i32 0
  %18 = load i64, i64 addrspace(4)* %17, align 8
  %19 = bitcast [2 x i64]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %19)
  %20 = bitcast [2 x i64]* %6 to i64*
  store i64 0, i64* %20, align 8
  %21 = bitcast [2 x i64]* %6 to i64*
  %22 = getelementptr inbounds i64, i64* %21, i64 1
  %23 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %24 = insertelement <3 x i64> undef, i64 %23, i32 0
  %25 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %26 = insertelement <3 x i64> %24, i64 %25, i32 1
  %27 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #5
  %28 = insertelement <3 x i64> %26, i64 %27, i32 2
  %29 = extractelement <3 x i64> %28, i32 0
  %30 = select i1 true, i64 %29, i64 0
  %31 = icmp ult i64 %30, 2147483648
  call void @llvm.assume(i1 %31)
  %32 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %33 = insertelement <3 x i64> undef, i64 %32, i32 0
  %34 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %35 = insertelement <3 x i64> %33, i64 %34, i32 1
  %36 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %37 = insertelement <3 x i64> %35, i64 %36, i32 2
  %38 = extractelement <3 x i64> %37, i32 0
  %39 = select i1 true, i64 %38, i64 0
  %40 = icmp ult i64 %39, 2147483648
  call void @llvm.assume(i1 %40)
  %41 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %42 = insertelement <3 x i64> undef, i64 %41, i32 0
  %43 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %44 = insertelement <3 x i64> %42, i64 %43, i32 1
  %45 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %46 = insertelement <3 x i64> %44, i64 %45, i32 2
  %47 = extractelement <3 x i64> %46, i32 0
  %48 = select i1 true, i64 %47, i64 1
  %49 = icmp ult i64 %48, 2147483648
  call void @llvm.assume(i1 %49)
  %50 = mul nuw nsw i64 %39, %48, !spirv.Decorations !395
  %51 = add nuw nsw i64 %50, %30, !spirv.Decorations !395
  %52 = and i64 %51, 4294967295
  store i64 %52, i64* %22, align 8
  %53 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2
  %54 = trunc i64 %18 to i32
  %55 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %53 to i32 addrspace(4)*
  store i32 %54, i32 addrspace(4)* %55, align 4
  %56 = lshr i64 %18, 32
  %57 = trunc i64 %56 to i32
  %58 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  store i32 %57, i32 addrspace(4)* %58, align 4
  %59 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 1
  %60 = bitcast [4 x i32] addrspace(4)* %59 to i8 addrspace(4)*
  %61 = bitcast [36 x i8]* @gVar to i8*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 %60, i8* align 8 %61, i64 36, i1 false)
  %62 = bitcast [3 x i64]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %62)
  %63 = bitcast [3 x i64]* %3 to i8*
  %64 = bitcast [24 x i8]* @gVar.61 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %63, i8* align 8 %64, i64 24, i1 false)
  br label %65

65:                                               ; preds = %65, %2
  %66 = phi i1 [ false, %65 ], [ true, %2 ]
  %67 = phi i64 [ 1, %65 ], [ 0, %2 ]
  %68 = phi i32 [ %76, %65 ], [ 0, %2 ]
  %69 = bitcast [2 x i64]* %6 to i64*
  %70 = getelementptr inbounds i64, i64* %69, i64 %67
  %71 = load i64, i64* %70, align 8
  %72 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 %67
  store i64 %71, i64* %72, align 8
  %73 = icmp eq i64 %71, 0
  %74 = trunc i64 %67 to i32
  %75 = add nuw nsw i32 %74, 1, !spirv.Decorations !395
  %76 = select i1 %73, i32 %68, i32 %75
  br i1 %66, label %65, label %77

77:                                               ; preds = %65
  %78 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 1, i64 1
  %79 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 1, i64 2
  %80 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 1, i64 3
  %81 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 2
  %82 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 3
  %83 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 3, i64 1
  %84 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 3, i64 2
  %85 = getelementptr inbounds %"struct.cutlass::reference::device::detail::RandomUniformFunc", %"struct.cutlass::reference::device::detail::RandomUniformFunc" addrspace(4)* %0, i64 0, i32 2, i32 0, i32 0, i32 3, i64 3
  switch i32 %76, label %86 [
    i32 0, label %150
    i32 1, label %89
  ]

86:                                               ; preds = %77
  %87 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 0
  %88 = load i64, i64* %87, align 8
  br label %94

89:                                               ; preds = %77
  %90 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 0
  %91 = load i64, i64* %90, align 8
  %92 = icmp eq i64 %91, 0
  br i1 %92, label %93, label %94

93:                                               ; preds = %89
  store i32 0, i32 addrspace(4)* %81, align 4
  br label %150

94:                                               ; preds = %89, %86
  %95 = phi i64 [ %88, %86 ], [ %91, %89 ]
  %96 = bitcast [4 x i32]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %96)
  %97 = bitcast [2 x i32]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %97)
  %98 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 0
  store i64 %95, i64* %98, align 8
  br label %99

99:                                               ; preds = %142, %94
  %100 = phi i64 [ 0, %94 ], [ %147, %142 ]
  %101 = phi i32 [ %76, %94 ], [ %143, %142 ]
  %102 = icmp sgt i32 %101, 0
  br i1 %102, label %142, label %103

103:                                              ; preds = %99
  %104 = trunc i64 %95 to i32
  %105 = and i32 %104, 3
  %106 = sub nuw nsw i32 4, %105, !spirv.Decorations !395
  store i32 %106, i32 addrspace(4)* %81, align 4
  %107 = load i64, i64* %98, align 8
  %108 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 1
  %109 = load i64, i64* %108, align 8
  %110 = trunc i64 %107 to i32
  %111 = getelementptr inbounds [4 x i32], [4 x i32]* %4, i64 0, i64 0
  %112 = addrspacecast i32* %111 to i32 addrspace(4)*
  store i32 %110, i32* %111, align 4
  %113 = lshr i64 %107, 32
  %114 = trunc i64 %113 to i32
  %115 = getelementptr inbounds [4 x i32], [4 x i32]* %4, i64 0, i64 1
  store i32 %114, i32* %115, align 4
  %116 = trunc i64 %109 to i32
  %117 = getelementptr inbounds [4 x i32], [4 x i32]* %4, i64 0, i64 2
  store i32 %116, i32* %117, align 4
  %118 = lshr i64 %109, 32
  %119 = trunc i64 %118 to i32
  %120 = getelementptr inbounds [4 x i32], [4 x i32]* %4, i64 0, i64 3
  store i32 %119, i32* %120, align 4
  %121 = getelementptr inbounds [2 x i32], [2 x i32]* %5, i64 0, i64 0
  %122 = addrspacecast i32* %121 to i32 addrspace(4)*
  store i32 %54, i32* %121, align 4
  %123 = getelementptr inbounds [2 x i32], [2 x i32]* %5, i64 0, i64 1
  store i32 %57, i32* %123, align 4
  call spir_func void @_ZN6oneapi3mkl3rng6device6detail18philox4x32x10_implL8round_10EPjS5_(i32 addrspace(4)* noalias nocapture %112, i32 addrspace(4)* noalias nocapture %122) #3
  %124 = load i32, i32* %111, align 4
  %125 = bitcast [4 x i32] addrspace(4)* %82 to i32 addrspace(4)*
  store i32 %124, i32 addrspace(4)* %125, align 4
  %126 = load i32, i32* %115, align 4
  store i32 %126, i32 addrspace(4)* %83, align 4
  %127 = load i32, i32* %117, align 4
  store i32 %127, i32 addrspace(4)* %84, align 4
  %128 = load i32, i32* %120, align 4
  store i32 %128, i32 addrspace(4)* %85, align 4
  %129 = add i64 %107, 1
  %130 = icmp eq i64 %129, 0
  %131 = select i1 %130, i64 1, i64 0
  %132 = add i64 %109, %131
  %133 = trunc i64 %129 to i32
  %134 = bitcast [4 x i32] addrspace(4)* %59 to i32 addrspace(4)*
  store i32 %133, i32 addrspace(4)* %134, align 4
  %135 = lshr i64 %129, 32
  %136 = trunc i64 %135 to i32
  store i32 %136, i32 addrspace(4)* %78, align 4
  %137 = trunc i64 %132 to i32
  store i32 %137, i32 addrspace(4)* %79, align 4
  %138 = lshr i64 %132, 32
  %139 = trunc i64 %138 to i32
  store i32 %139, i32 addrspace(4)* %80, align 4
  %140 = bitcast [2 x i32]* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %140)
  %141 = bitcast [4 x i32]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %141)
  br label %150

142:                                              ; preds = %99
  %143 = add nsw i32 %101, -1, !spirv.Decorations !398
  %144 = zext i32 %143 to i64
  %145 = getelementptr inbounds [3 x i64], [3 x i64]* %3, i64 0, i64 %144
  %146 = load i64, i64* %145, align 8
  %147 = shl i64 %146, 62
  %148 = lshr i64 %146, 2
  %149 = or i64 %148, %100
  store i64 %149, i64* %145, align 8
  br label %99

150:                                              ; preds = %103, %93, %77
  %151 = bitcast [3 x i64]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %151)
  %152 = bitcast [2 x i64]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %152)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nounwind
define internal spir_func void @_ZN6oneapi3mkl3rng6device6detail18philox4x32x10_implL8round_10EPjS5_(i32 addrspace(4)* noalias nocapture %0, i32 addrspace(4)* noalias nocapture %1) #3 !spirv.ParameterDecorations !399 {
  %3 = load i32, i32 addrspace(4)* %0, align 4
  %4 = zext i32 %3 to i64
  %5 = mul nuw i64 %4, 3528531795, !spirv.Decorations !403
  %6 = trunc i64 %5 to i32
  %7 = lshr i64 %5, 32
  %8 = trunc i64 %7 to i32
  %9 = getelementptr inbounds i32, i32 addrspace(4)* %0, i64 2
  %10 = load i32, i32 addrspace(4)* %9, align 4
  %11 = zext i32 %10 to i64
  %12 = mul nuw i64 %11, 3449720151, !spirv.Decorations !403
  %13 = trunc i64 %12 to i32
  %14 = lshr i64 %12, 32
  %15 = trunc i64 %14 to i32
  %16 = getelementptr inbounds i32, i32 addrspace(4)* %0, i64 1
  %17 = load i32, i32 addrspace(4)* %16, align 4
  %18 = xor i32 %17, %15
  %19 = load i32, i32 addrspace(4)* %1, align 4
  %20 = xor i32 %18, %19
  store i32 %20, i32 addrspace(4)* %0, align 4
  store i32 %13, i32 addrspace(4)* %16, align 4
  %21 = getelementptr inbounds i32, i32 addrspace(4)* %0, i64 3
  %22 = load i32, i32 addrspace(4)* %21, align 4
  %23 = xor i32 %22, %8
  %24 = getelementptr inbounds i32, i32 addrspace(4)* %1, i64 1
  %25 = load i32, i32 addrspace(4)* %24, align 4
  %26 = xor i32 %23, %25
  store i32 %26, i32 addrspace(4)* %9, align 4
  store i32 %6, i32 addrspace(4)* %21, align 4
  %27 = load i32, i32 addrspace(4)* %1, align 4
  %28 = add i32 %27, -1640531527
  store i32 %28, i32 addrspace(4)* %1, align 4
  %29 = load i32, i32 addrspace(4)* %24, align 4
  %30 = add i32 %29, -1150833019
  store i32 %30, i32 addrspace(4)* %24, align 4
  %31 = load i32, i32 addrspace(4)* %0, align 4
  %32 = zext i32 %31 to i64
  %33 = mul nuw i64 %32, 3528531795, !spirv.Decorations !403
  %34 = trunc i64 %33 to i32
  %35 = lshr i64 %33, 32
  %36 = trunc i64 %35 to i32
  %37 = load i32, i32 addrspace(4)* %9, align 4
  %38 = zext i32 %37 to i64
  %39 = mul nuw i64 %38, 3449720151, !spirv.Decorations !403
  %40 = trunc i64 %39 to i32
  %41 = lshr i64 %39, 32
  %42 = trunc i64 %41 to i32
  %43 = load i32, i32 addrspace(4)* %16, align 4
  %44 = xor i32 %43, %42
  %45 = xor i32 %44, %28
  store i32 %45, i32 addrspace(4)* %0, align 4
  store i32 %40, i32 addrspace(4)* %16, align 4
  %46 = load i32, i32 addrspace(4)* %21, align 4
  %47 = xor i32 %46, %36
  %48 = load i32, i32 addrspace(4)* %24, align 4
  %49 = xor i32 %47, %48
  store i32 %49, i32 addrspace(4)* %9, align 4
  store i32 %34, i32 addrspace(4)* %21, align 4
  %50 = load i32, i32 addrspace(4)* %1, align 4
  %51 = add i32 %50, -1640531527
  store i32 %51, i32 addrspace(4)* %1, align 4
  %52 = load i32, i32 addrspace(4)* %24, align 4
  %53 = add i32 %52, -1150833019
  store i32 %53, i32 addrspace(4)* %24, align 4
  %54 = load i32, i32 addrspace(4)* %0, align 4
  %55 = zext i32 %54 to i64
  %56 = mul nuw i64 %55, 3528531795, !spirv.Decorations !403
  %57 = trunc i64 %56 to i32
  %58 = lshr i64 %56, 32
  %59 = trunc i64 %58 to i32
  %60 = load i32, i32 addrspace(4)* %9, align 4
  %61 = zext i32 %60 to i64
  %62 = mul nuw i64 %61, 3449720151, !spirv.Decorations !403
  %63 = trunc i64 %62 to i32
  %64 = lshr i64 %62, 32
  %65 = trunc i64 %64 to i32
  %66 = load i32, i32 addrspace(4)* %16, align 4
  %67 = xor i32 %66, %65
  %68 = xor i32 %67, %51
  store i32 %68, i32 addrspace(4)* %0, align 4
  store i32 %63, i32 addrspace(4)* %16, align 4
  %69 = load i32, i32 addrspace(4)* %21, align 4
  %70 = xor i32 %69, %59
  %71 = load i32, i32 addrspace(4)* %24, align 4
  %72 = xor i32 %70, %71
  store i32 %72, i32 addrspace(4)* %9, align 4
  store i32 %57, i32 addrspace(4)* %21, align 4
  %73 = load i32, i32 addrspace(4)* %1, align 4
  %74 = add i32 %73, -1640531527
  store i32 %74, i32 addrspace(4)* %1, align 4
  %75 = load i32, i32 addrspace(4)* %24, align 4
  %76 = add i32 %75, -1150833019
  store i32 %76, i32 addrspace(4)* %24, align 4
  %77 = load i32, i32 addrspace(4)* %0, align 4
  %78 = zext i32 %77 to i64
  %79 = mul nuw i64 %78, 3528531795, !spirv.Decorations !403
  %80 = trunc i64 %79 to i32
  %81 = lshr i64 %79, 32
  %82 = trunc i64 %81 to i32
  %83 = load i32, i32 addrspace(4)* %9, align 4
  %84 = zext i32 %83 to i64
  %85 = mul nuw i64 %84, 3449720151, !spirv.Decorations !403
  %86 = trunc i64 %85 to i32
  %87 = lshr i64 %85, 32
  %88 = trunc i64 %87 to i32
  %89 = load i32, i32 addrspace(4)* %16, align 4
  %90 = xor i32 %89, %88
  %91 = xor i32 %90, %74
  store i32 %91, i32 addrspace(4)* %0, align 4
  store i32 %86, i32 addrspace(4)* %16, align 4
  %92 = load i32, i32 addrspace(4)* %21, align 4
  %93 = xor i32 %92, %82
  %94 = load i32, i32 addrspace(4)* %24, align 4
  %95 = xor i32 %93, %94
  store i32 %95, i32 addrspace(4)* %9, align 4
  store i32 %80, i32 addrspace(4)* %21, align 4
  %96 = load i32, i32 addrspace(4)* %1, align 4
  %97 = add i32 %96, -1640531527
  store i32 %97, i32 addrspace(4)* %1, align 4
  %98 = load i32, i32 addrspace(4)* %24, align 4
  %99 = add i32 %98, -1150833019
  store i32 %99, i32 addrspace(4)* %24, align 4
  %100 = load i32, i32 addrspace(4)* %0, align 4
  %101 = zext i32 %100 to i64
  %102 = mul nuw i64 %101, 3528531795, !spirv.Decorations !403
  %103 = trunc i64 %102 to i32
  %104 = lshr i64 %102, 32
  %105 = trunc i64 %104 to i32
  %106 = load i32, i32 addrspace(4)* %9, align 4
  %107 = zext i32 %106 to i64
  %108 = mul nuw i64 %107, 3449720151, !spirv.Decorations !403
  %109 = trunc i64 %108 to i32
  %110 = lshr i64 %108, 32
  %111 = trunc i64 %110 to i32
  %112 = load i32, i32 addrspace(4)* %16, align 4
  %113 = xor i32 %112, %111
  %114 = xor i32 %113, %97
  store i32 %114, i32 addrspace(4)* %0, align 4
  store i32 %109, i32 addrspace(4)* %16, align 4
  %115 = load i32, i32 addrspace(4)* %21, align 4
  %116 = xor i32 %115, %105
  %117 = load i32, i32 addrspace(4)* %24, align 4
  %118 = xor i32 %116, %117
  store i32 %118, i32 addrspace(4)* %9, align 4
  store i32 %103, i32 addrspace(4)* %21, align 4
  %119 = load i32, i32 addrspace(4)* %1, align 4
  %120 = add i32 %119, -1640531527
  store i32 %120, i32 addrspace(4)* %1, align 4
  %121 = load i32, i32 addrspace(4)* %24, align 4
  %122 = add i32 %121, -1150833019
  store i32 %122, i32 addrspace(4)* %24, align 4
  %123 = load i32, i32 addrspace(4)* %0, align 4
  %124 = zext i32 %123 to i64
  %125 = mul nuw i64 %124, 3528531795, !spirv.Decorations !403
  %126 = trunc i64 %125 to i32
  %127 = lshr i64 %125, 32
  %128 = trunc i64 %127 to i32
  %129 = load i32, i32 addrspace(4)* %9, align 4
  %130 = zext i32 %129 to i64
  %131 = mul nuw i64 %130, 3449720151, !spirv.Decorations !403
  %132 = trunc i64 %131 to i32
  %133 = lshr i64 %131, 32
  %134 = trunc i64 %133 to i32
  %135 = load i32, i32 addrspace(4)* %16, align 4
  %136 = xor i32 %135, %134
  %137 = xor i32 %136, %120
  store i32 %137, i32 addrspace(4)* %0, align 4
  store i32 %132, i32 addrspace(4)* %16, align 4
  %138 = load i32, i32 addrspace(4)* %21, align 4
  %139 = xor i32 %138, %128
  %140 = load i32, i32 addrspace(4)* %24, align 4
  %141 = xor i32 %139, %140
  store i32 %141, i32 addrspace(4)* %9, align 4
  store i32 %126, i32 addrspace(4)* %21, align 4
  %142 = load i32, i32 addrspace(4)* %1, align 4
  %143 = add i32 %142, -1640531527
  store i32 %143, i32 addrspace(4)* %1, align 4
  %144 = load i32, i32 addrspace(4)* %24, align 4
  %145 = add i32 %144, -1150833019
  store i32 %145, i32 addrspace(4)* %24, align 4
  %146 = load i32, i32 addrspace(4)* %0, align 4
  %147 = zext i32 %146 to i64
  %148 = mul nuw i64 %147, 3528531795, !spirv.Decorations !403
  %149 = trunc i64 %148 to i32
  %150 = lshr i64 %148, 32
  %151 = trunc i64 %150 to i32
  %152 = load i32, i32 addrspace(4)* %9, align 4
  %153 = zext i32 %152 to i64
  %154 = mul nuw i64 %153, 3449720151, !spirv.Decorations !403
  %155 = trunc i64 %154 to i32
  %156 = lshr i64 %154, 32
  %157 = trunc i64 %156 to i32
  %158 = load i32, i32 addrspace(4)* %16, align 4
  %159 = xor i32 %158, %157
  %160 = xor i32 %159, %143
  store i32 %160, i32 addrspace(4)* %0, align 4
  store i32 %155, i32 addrspace(4)* %16, align 4
  %161 = load i32, i32 addrspace(4)* %21, align 4
  %162 = xor i32 %161, %151
  %163 = load i32, i32 addrspace(4)* %24, align 4
  %164 = xor i32 %162, %163
  store i32 %164, i32 addrspace(4)* %9, align 4
  store i32 %149, i32 addrspace(4)* %21, align 4
  %165 = load i32, i32 addrspace(4)* %1, align 4
  %166 = add i32 %165, -1640531527
  store i32 %166, i32 addrspace(4)* %1, align 4
  %167 = load i32, i32 addrspace(4)* %24, align 4
  %168 = add i32 %167, -1150833019
  store i32 %168, i32 addrspace(4)* %24, align 4
  %169 = load i32, i32 addrspace(4)* %0, align 4
  %170 = zext i32 %169 to i64
  %171 = mul nuw i64 %170, 3528531795, !spirv.Decorations !403
  %172 = trunc i64 %171 to i32
  %173 = lshr i64 %171, 32
  %174 = trunc i64 %173 to i32
  %175 = load i32, i32 addrspace(4)* %9, align 4
  %176 = zext i32 %175 to i64
  %177 = mul nuw i64 %176, 3449720151, !spirv.Decorations !403
  %178 = trunc i64 %177 to i32
  %179 = lshr i64 %177, 32
  %180 = trunc i64 %179 to i32
  %181 = load i32, i32 addrspace(4)* %16, align 4
  %182 = xor i32 %181, %180
  %183 = xor i32 %182, %166
  store i32 %183, i32 addrspace(4)* %0, align 4
  store i32 %178, i32 addrspace(4)* %16, align 4
  %184 = load i32, i32 addrspace(4)* %21, align 4
  %185 = xor i32 %184, %174
  %186 = load i32, i32 addrspace(4)* %24, align 4
  %187 = xor i32 %185, %186
  store i32 %187, i32 addrspace(4)* %9, align 4
  store i32 %172, i32 addrspace(4)* %21, align 4
  %188 = load i32, i32 addrspace(4)* %1, align 4
  %189 = add i32 %188, -1640531527
  store i32 %189, i32 addrspace(4)* %1, align 4
  %190 = load i32, i32 addrspace(4)* %24, align 4
  %191 = add i32 %190, -1150833019
  store i32 %191, i32 addrspace(4)* %24, align 4
  %192 = load i32, i32 addrspace(4)* %0, align 4
  %193 = zext i32 %192 to i64
  %194 = mul nuw i64 %193, 3528531795, !spirv.Decorations !403
  %195 = trunc i64 %194 to i32
  %196 = lshr i64 %194, 32
  %197 = trunc i64 %196 to i32
  %198 = load i32, i32 addrspace(4)* %9, align 4
  %199 = zext i32 %198 to i64
  %200 = mul nuw i64 %199, 3449720151, !spirv.Decorations !403
  %201 = trunc i64 %200 to i32
  %202 = lshr i64 %200, 32
  %203 = trunc i64 %202 to i32
  %204 = load i32, i32 addrspace(4)* %16, align 4
  %205 = xor i32 %204, %203
  %206 = xor i32 %205, %189
  store i32 %206, i32 addrspace(4)* %0, align 4
  store i32 %201, i32 addrspace(4)* %16, align 4
  %207 = load i32, i32 addrspace(4)* %21, align 4
  %208 = xor i32 %207, %197
  %209 = load i32, i32 addrspace(4)* %24, align 4
  %210 = xor i32 %208, %209
  store i32 %210, i32 addrspace(4)* %9, align 4
  store i32 %195, i32 addrspace(4)* %21, align 4
  %211 = load i32, i32 addrspace(4)* %1, align 4
  %212 = add i32 %211, -1640531527
  store i32 %212, i32 addrspace(4)* %1, align 4
  %213 = load i32, i32 addrspace(4)* %24, align 4
  %214 = add i32 %213, -1150833019
  store i32 %214, i32 addrspace(4)* %24, align 4
  %215 = load i32, i32 addrspace(4)* %0, align 4
  %216 = zext i32 %215 to i64
  %217 = mul nuw i64 %216, 3528531795, !spirv.Decorations !403
  %218 = trunc i64 %217 to i32
  %219 = lshr i64 %217, 32
  %220 = trunc i64 %219 to i32
  %221 = load i32, i32 addrspace(4)* %9, align 4
  %222 = zext i32 %221 to i64
  %223 = mul nuw i64 %222, 3449720151, !spirv.Decorations !403
  %224 = trunc i64 %223 to i32
  %225 = lshr i64 %223, 32
  %226 = trunc i64 %225 to i32
  %227 = load i32, i32 addrspace(4)* %16, align 4
  %228 = xor i32 %227, %226
  %229 = xor i32 %228, %212
  store i32 %229, i32 addrspace(4)* %0, align 4
  store i32 %224, i32 addrspace(4)* %16, align 4
  %230 = load i32, i32 addrspace(4)* %21, align 4
  %231 = xor i32 %230, %220
  %232 = load i32, i32 addrspace(4)* %24, align 4
  %233 = xor i32 %231, %232
  store i32 %233, i32 addrspace(4)* %9, align 4
  store i32 %218, i32 addrspace(4)* %21, align 4
  ret void
}

; Function Attrs: nounwind
define internal spir_func float @_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_(%"class.oneapi::mkl::rng::device::uniform" addrspace(4)* align 4 dereferenceable(8) %0, %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* align 4 dereferenceable(44) %1) #3 !spirv.ParameterDecorations !404 {
  %3 = alloca [4 x i32], align 4, !spirv.Decorations !393
  %4 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %5 = alloca [4 x i32], align 4, !spirv.Decorations !393
  %6 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %7 = alloca [4 x i32], align 4, !spirv.Decorations !393
  %8 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %9 = bitcast %"class.oneapi::mkl::rng::device::uniform" addrspace(4)* %0 to %"class.oneapi::mkl::rng::device::detail::distribution_base" addrspace(4)*
  %10 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::distribution_base", %"class.oneapi::mkl::rng::device::detail::distribution_base" addrspace(4)* %9, i64 0, i32 0
  %11 = load float, float addrspace(4)* %10, align 4
  %12 = bitcast %"class.oneapi::mkl::rng::device::uniform" addrspace(4)* %0 to %"class.oneapi::mkl::rng::device::detail::distribution_base" addrspace(4)*
  %13 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::distribution_base", %"class.oneapi::mkl::rng::device::detail::distribution_base" addrspace(4)* %12, i64 0, i32 1
  %14 = load float, float addrspace(4)* %13, align 4
  %15 = bitcast [4 x i32]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %15)
  %16 = bitcast [2 x i32]* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %16)
  %17 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %18 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %17, i64 0, i32 0, i32 2
  %19 = load i32, i32 addrspace(4)* %18, align 4, !noalias !409
  %20 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %21 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %20, i64 0, i32 0, i32 3
  br label %22

22:                                               ; preds = %29, %2
  %23 = phi i32 [ undef, %2 ], [ %34, %29 ]
  %24 = phi i1 [ true, %2 ], [ false, %29 ]
  %25 = phi i1 [ false, %2 ], [ true, %29 ]
  %26 = phi i32 [ %19, %2 ], [ %30, %29 ]
  %27 = icmp ne i32 %26, 0
  %28 = and i1 %27, %24
  br i1 %28, label %29, label %35

29:                                               ; preds = %22
  %30 = add nsw i32 %26, -1, !spirv.Decorations !398
  %31 = sub nsw i32 4, %26, !spirv.Decorations !398
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(4)* %21, i64 0, i64 %32
  %34 = load i32, i32 addrspace(4)* %33, align 4, !noalias !409
  br label %22

35:                                               ; preds = %22
  %36 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %37 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %36, i64 0, i32 0
  br i1 %25, label %38, label %96

38:                                               ; preds = %35
  %39 = bitcast [4 x i32]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %39)
  %40 = bitcast [2 x i32]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %40)
  %41 = icmp eq i32 %19, 0
  br i1 %41, label %44, label %42

42:                                               ; preds = %38
  %43 = add i32 %19, -1
  store i32 %43, i32 addrspace(4)* %18, align 4
  br label %93

44:                                               ; preds = %38
  store i32 3, i32 addrspace(4)* %18, align 4
  %45 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %46 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %45, i64 0, i32 0, i32 1
  %47 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %48 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %47, i64 0, i32 0, i32 1, i64 1
  %49 = load i32, i32 addrspace(4)* %48, align 4
  %50 = bitcast [4 x i32] addrspace(4)* %46 to i32 addrspace(4)*
  %51 = load i32, i32 addrspace(4)* %50, align 4
  %52 = getelementptr inbounds [4 x i32], [4 x i32]* %5, i64 0, i64 0
  %53 = addrspacecast i32* %52 to i32 addrspace(4)*
  store i32 %51, i32* %52, align 4
  %54 = getelementptr inbounds [4 x i32], [4 x i32]* %5, i64 0, i64 1
  store i32 %49, i32* %54, align 4
  %55 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %56 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %55, i64 0, i32 0, i32 1, i64 2
  %57 = load i32, i32 addrspace(4)* %56, align 4
  %58 = getelementptr inbounds [4 x i32], [4 x i32]* %5, i64 0, i64 2
  store i32 %57, i32* %58, align 4
  %59 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %60 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %59, i64 0, i32 0, i32 1, i64 3
  %61 = load i32, i32 addrspace(4)* %60, align 4
  %62 = getelementptr inbounds [4 x i32], [4 x i32]* %5, i64 0, i64 3
  store i32 %61, i32* %62, align 4
  %63 = bitcast %"struct.oneapi::mkl::rng::device::detail::engine_state" addrspace(4)* %37 to i32 addrspace(4)*
  %64 = load i32, i32 addrspace(4)* %63, align 4
  %65 = getelementptr inbounds [2 x i32], [2 x i32]* %6, i64 0, i64 0
  %66 = addrspacecast i32* %65 to i32 addrspace(4)*
  store i32 %64, i32* %65, align 4
  %67 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %68 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %67, i64 0, i32 0, i32 0, i64 1
  %69 = load i32, i32 addrspace(4)* %68, align 4
  %70 = getelementptr inbounds [2 x i32], [2 x i32]* %6, i64 0, i64 1
  store i32 %69, i32* %70, align 4
  call spir_func void @_ZN6oneapi3mkl3rng6device6detail18philox4x32x10_implL8round_10EPjS5_(i32 addrspace(4)* noalias nocapture %53, i32 addrspace(4)* noalias nocapture %66) #3
  %71 = load i32, i32* %52, align 4
  %72 = bitcast [4 x i32] addrspace(4)* %21 to i32 addrspace(4)*
  store i32 %71, i32 addrspace(4)* %72, align 4
  %73 = load i32, i32* %54, align 4
  %74 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %75 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %74, i64 0, i32 0, i32 3, i64 1
  store i32 %73, i32 addrspace(4)* %75, align 4
  %76 = load i32, i32* %58, align 4
  %77 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %78 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %77, i64 0, i32 0, i32 3, i64 2
  store i32 %76, i32 addrspace(4)* %78, align 4
  %79 = load i32, i32* %62, align 4
  %80 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %81 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %80, i64 0, i32 0, i32 3, i64 3
  store i32 %79, i32 addrspace(4)* %81, align 4
  %82 = add i32 %51, 1
  %83 = bitcast [4 x i32] addrspace(4)* %46 to i32 addrspace(4)*
  store i32 %82, i32 addrspace(4)* %83, align 4
  %84 = icmp eq i32 %82, 0
  br i1 %84, label %85, label %93

85:                                               ; preds = %44
  %86 = add i32 %49, 1
  store i32 %86, i32 addrspace(4)* %48, align 4
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %88, label %93

88:                                               ; preds = %85
  %89 = add i32 %57, 1
  store i32 %89, i32 addrspace(4)* %56, align 4
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %91, label %93

91:                                               ; preds = %88
  %92 = add i32 %61, 1
  store i32 %92, i32 addrspace(4)* %60, align 4
  br label %93

93:                                               ; preds = %91, %88, %85, %44, %42
  %94 = bitcast [2 x i32]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %94)
  %95 = bitcast [4 x i32]* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %95)
  br label %163

96:                                               ; preds = %35
  %97 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %98 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %97, i64 0, i32 0, i32 1
  %99 = bitcast [4 x i32] addrspace(4)* %98 to i32 addrspace(4)*
  %100 = load i32, i32 addrspace(4)* %99, align 4, !noalias !409
  %101 = getelementptr inbounds [4 x i32], [4 x i32]* %7, i64 0, i64 0
  %102 = addrspacecast i32* %101 to i32 addrspace(4)*
  store i32 %100, i32* %101, align 4, !noalias !409
  %103 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %104 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %103, i64 0, i32 0, i32 1, i64 1
  %105 = load i32, i32 addrspace(4)* %104, align 4, !noalias !409
  %106 = getelementptr inbounds [4 x i32], [4 x i32]* %7, i64 0, i64 1
  store i32 %105, i32* %106, align 4, !noalias !409
  %107 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %108 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %107, i64 0, i32 0, i32 1, i64 2
  %109 = load i32, i32 addrspace(4)* %108, align 4, !noalias !409
  %110 = getelementptr inbounds [4 x i32], [4 x i32]* %7, i64 0, i64 2
  store i32 %109, i32* %110, align 4, !noalias !409
  %111 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %112 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %111, i64 0, i32 0, i32 1, i64 3
  %113 = load i32, i32 addrspace(4)* %112, align 4, !noalias !409
  %114 = getelementptr inbounds [4 x i32], [4 x i32]* %7, i64 0, i64 3
  store i32 %113, i32* %114, align 4, !noalias !409
  %115 = bitcast %"struct.oneapi::mkl::rng::device::detail::engine_state" addrspace(4)* %37 to i32 addrspace(4)*
  %116 = load i32, i32 addrspace(4)* %115, align 4, !noalias !409
  %117 = getelementptr inbounds [2 x i32], [2 x i32]* %8, i64 0, i64 0
  %118 = addrspacecast i32* %117 to i32 addrspace(4)*
  store i32 %116, i32* %117, align 4, !noalias !409
  %119 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %120 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %119, i64 0, i32 0, i32 0, i64 1
  %121 = load i32, i32 addrspace(4)* %120, align 4, !noalias !409
  %122 = getelementptr inbounds [2 x i32], [2 x i32]* %8, i64 0, i64 1
  store i32 %121, i32* %122, align 4, !noalias !409
  call spir_func void @_ZN6oneapi3mkl3rng6device6detail18philox4x32x10_implL8round_10EPjS5_(i32 addrspace(4)* noalias nocapture %102, i32 addrspace(4)* noalias nocapture %118) #3
  %123 = load i32, i32* %101, align 4, !noalias !409
  %124 = bitcast [4 x i32]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %124)
  %125 = bitcast [2 x i32]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %125)
  %126 = icmp eq i32 %19, 0
  br i1 %126, label %129, label %127

127:                                              ; preds = %96
  %128 = add i32 %19, -1
  store i32 %128, i32 addrspace(4)* %18, align 4
  br label %160

129:                                              ; preds = %96
  store i32 3, i32 addrspace(4)* %18, align 4
  %130 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i64 0, i64 0
  %131 = addrspacecast i32* %130 to i32 addrspace(4)*
  store i32 %100, i32* %130, align 4
  %132 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i64 0, i64 1
  store i32 %105, i32* %132, align 4
  %133 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i64 0, i64 2
  store i32 %109, i32* %133, align 4
  %134 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i64 0, i64 3
  store i32 %113, i32* %134, align 4
  %135 = getelementptr inbounds [2 x i32], [2 x i32]* %4, i64 0, i64 0
  %136 = addrspacecast i32* %135 to i32 addrspace(4)*
  store i32 %116, i32* %135, align 4
  %137 = getelementptr inbounds [2 x i32], [2 x i32]* %4, i64 0, i64 1
  store i32 %121, i32* %137, align 4
  call spir_func void @_ZN6oneapi3mkl3rng6device6detail18philox4x32x10_implL8round_10EPjS5_(i32 addrspace(4)* noalias nocapture %131, i32 addrspace(4)* noalias nocapture %136) #3
  %138 = load i32, i32* %130, align 4
  %139 = bitcast [4 x i32] addrspace(4)* %21 to i32 addrspace(4)*
  store i32 %138, i32 addrspace(4)* %139, align 4
  %140 = load i32, i32* %132, align 4
  %141 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %142 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %141, i64 0, i32 0, i32 3, i64 1
  store i32 %140, i32 addrspace(4)* %142, align 4
  %143 = load i32, i32* %133, align 4
  %144 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %145 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %144, i64 0, i32 0, i32 3, i64 2
  store i32 %143, i32 addrspace(4)* %145, align 4
  %146 = load i32, i32* %134, align 4
  %147 = bitcast %"class.oneapi::mkl::rng::device::philox4x32x10" addrspace(4)* %1 to %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)*
  %148 = getelementptr inbounds %"class.oneapi::mkl::rng::device::detail::engine_base", %"class.oneapi::mkl::rng::device::detail::engine_base" addrspace(4)* %147, i64 0, i32 0, i32 3, i64 3
  store i32 %146, i32 addrspace(4)* %148, align 4
  %149 = add i32 %100, 1
  %150 = bitcast [4 x i32] addrspace(4)* %98 to i32 addrspace(4)*
  store i32 %149, i32 addrspace(4)* %150, align 4
  %151 = icmp eq i32 %149, 0
  br i1 %151, label %152, label %160

152:                                              ; preds = %129
  %153 = add i32 %105, 1
  store i32 %153, i32 addrspace(4)* %104, align 4
  %154 = icmp eq i32 %153, 0
  br i1 %154, label %155, label %160

155:                                              ; preds = %152
  %156 = add i32 %109, 1
  store i32 %156, i32 addrspace(4)* %108, align 4
  %157 = icmp eq i32 %156, 0
  br i1 %157, label %158, label %160

158:                                              ; preds = %155
  %159 = add i32 %113, 1
  store i32 %159, i32 addrspace(4)* %112, align 4
  br label %160

160:                                              ; preds = %158, %155, %152, %129, %127
  %161 = bitcast [2 x i32]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %161)
  %162 = bitcast [4 x i32]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %162)
  br label %163

163:                                              ; preds = %160, %93
  %164 = phi i32 [ %23, %93 ], [ %123, %160 ]
  %165 = fadd reassoc nsz arcp contract float %14, %11, !spirv.Decorations !412
  %166 = fdiv reassoc nsz arcp contract float %165, 2.000000e+00, !spirv.Decorations !412
  %167 = fsub reassoc nsz arcp contract float %14, %11, !spirv.Decorations !412
  %168 = fdiv reassoc nsz arcp contract float %167, 0x41F0000000000000, !spirv.Decorations !412
  %169 = bitcast [2 x i32]* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %169)
  %170 = bitcast [4 x i32]* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %170)
  %171 = sitofp i32 %164 to float
  %172 = fmul reassoc nsz arcp contract float %168, %171, !spirv.Decorations !412
  %173 = fadd reassoc nsz arcp contract float %172, %166, !spirv.Decorations !412
  ret float %173
}

; Function Attrs: nounwind memory(none)
declare spir_func float @_Z17__spirv_ocl_roundf(float) #4

; Function Attrs: nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12) #3 !spirv.ParameterDecorations !414 !kernel_arg_addr_space !418 !kernel_arg_access_qual !419 !kernel_arg_type !420 !kernel_arg_type_qual !421 !kernel_arg_base_type !420 !kernel_arg_name !421 {
  %14 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %15 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %16 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %17 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %18 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %19 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %20 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !393
  %21 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %22 = load i32, i32* %21, align 4
  %23 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %24 = getelementptr inbounds i32, i32* %23, i64 1
  %25 = load i32, i32* %24, align 4
  %26 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %27 = getelementptr inbounds i32, i32* %26, i64 2
  %28 = load i32, i32* %27, align 4
  %29 = bitcast %"class.cutlass::__generated_TensorRef"* %2 to %structtype*
  %30 = getelementptr inbounds %structtype, %structtype* %29, i64 0, i32 0
  %31 = load i64, i64* %30, align 8
  %32 = bitcast %"class.cutlass::__generated_TensorRef"* %2 to %structtype*
  %33 = getelementptr inbounds %structtype, %structtype* %32, i64 0, i32 1
  %34 = load i64, i64* %33, align 8
  %35 = bitcast %"class.cutlass::__generated_TensorRef"* %3 to %structtype*
  %36 = getelementptr inbounds %structtype, %structtype* %35, i64 0, i32 0
  %37 = load i64, i64* %36, align 8
  %38 = bitcast %"class.cutlass::__generated_TensorRef"* %3 to %structtype*
  %39 = getelementptr inbounds %structtype, %structtype* %38, i64 0, i32 1
  %40 = load i64, i64* %39, align 8
  %41 = bitcast %"class.cutlass::__generated_TensorRef"* %5 to %structtype*
  %42 = getelementptr inbounds %structtype, %structtype* %41, i64 0, i32 0
  %43 = load i64, i64* %42, align 8
  %44 = bitcast %"class.cutlass::__generated_TensorRef"* %5 to %structtype*
  %45 = getelementptr inbounds %structtype, %structtype* %44, i64 0, i32 1
  %46 = load i64, i64* %45, align 8
  %47 = bitcast %"class.cutlass::__generated_TensorRef"* %6 to %structtype*
  %48 = getelementptr inbounds %structtype, %structtype* %47, i64 0, i32 0
  %49 = load i64, i64* %48, align 8
  %50 = bitcast %"class.cutlass::__generated_TensorRef"* %6 to %structtype*
  %51 = getelementptr inbounds %structtype, %structtype* %50, i64 0, i32 1
  %52 = load i64, i64* %51, align 8
  %53 = inttoptr i64 %49 to float addrspace(4)*
  %54 = inttoptr i64 %43 to float addrspace(4)*
  %55 = inttoptr i64 %37 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %56 = inttoptr i64 %31 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %57 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %58 = insertelement <3 x i64> undef, i64 %57, i32 0
  %59 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %60 = insertelement <3 x i64> %58, i64 %59, i32 1
  %61 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %62 = insertelement <3 x i64> %60, i64 %61, i32 2
  %63 = extractelement <3 x i64> %62, i32 0
  %64 = select i1 true, i64 %63, i64 0
  %65 = icmp ult i64 %64, 2147483648
  call void @llvm.assume(i1 %65)
  %66 = trunc i64 %64 to i32
  %67 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %68 = insertelement <3 x i64> undef, i64 %67, i32 0
  %69 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %70 = insertelement <3 x i64> %68, i64 %69, i32 1
  %71 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %72 = insertelement <3 x i64> %70, i64 %71, i32 2
  %73 = extractelement <3 x i64> %72, i32 0
  %74 = select i1 true, i64 %73, i64 1
  %75 = icmp ult i64 %74, 2147483648
  call void @llvm.assume(i1 %75)
  %76 = trunc i64 %74 to i32
  %77 = mul i32 %66, %76
  %78 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %79 = insertelement <3 x i64> undef, i64 %78, i32 0
  %80 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %81 = insertelement <3 x i64> %79, i64 %80, i32 1
  %82 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #5
  %83 = insertelement <3 x i64> %81, i64 %82, i32 2
  %84 = extractelement <3 x i64> %83, i32 0
  %85 = select i1 true, i64 %84, i64 0
  %86 = icmp ult i64 %85, 2147483648
  call void @llvm.assume(i1 %86)
  %87 = trunc i64 %85 to i32
  %88 = add i32 %77, %87
  %89 = shl i32 %88, 2
  %90 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %91 = insertelement <3 x i64> undef, i64 %90, i32 0
  %92 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %93 = insertelement <3 x i64> %91, i64 %92, i32 1
  %94 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %95 = insertelement <3 x i64> %93, i64 %94, i32 2
  %96 = extractelement <3 x i64> %95, i32 1
  %97 = select i1 true, i64 %96, i64 0
  %98 = icmp ult i64 %97, 2147483648
  call void @llvm.assume(i1 %98)
  %99 = trunc i64 %97 to i32
  %100 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %101 = insertelement <3 x i64> undef, i64 %100, i32 0
  %102 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %103 = insertelement <3 x i64> %101, i64 %102, i32 1
  %104 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %105 = insertelement <3 x i64> %103, i64 %104, i32 2
  %106 = extractelement <3 x i64> %105, i32 1
  %107 = select i1 true, i64 %106, i64 1
  %108 = icmp ult i64 %107, 2147483648
  call void @llvm.assume(i1 %108)
  %109 = trunc i64 %107 to i32
  %110 = mul i32 %99, %109
  %111 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %112 = insertelement <3 x i64> undef, i64 %111, i32 0
  %113 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %114 = insertelement <3 x i64> %112, i64 %113, i32 1
  %115 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #5
  %116 = insertelement <3 x i64> %114, i64 %115, i32 2
  %117 = extractelement <3 x i64> %116, i32 1
  %118 = select i1 true, i64 %117, i64 0
  %119 = icmp ult i64 %118, 2147483648
  call void @llvm.assume(i1 %119)
  %120 = trunc i64 %118 to i32
  %121 = add i32 %110, %120
  %122 = shl i32 %121, 4
  %123 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %124 = insertelement <3 x i64> undef, i64 %123, i32 0
  %125 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %126 = insertelement <3 x i64> %124, i64 %125, i32 1
  %127 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %128 = insertelement <3 x i64> %126, i64 %127, i32 2
  %129 = extractelement <3 x i64> %128, i32 2
  %130 = select i1 true, i64 %129, i64 0
  %131 = icmp ult i64 %130, 2147483648
  call void @llvm.assume(i1 %131)
  %132 = trunc i64 %130 to i32
  %133 = mul nsw i64 %9, %130, !spirv.Decorations !398
  %134 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %56, i64 %133
  %135 = mul nsw i64 %10, %130, !spirv.Decorations !398
  %136 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %55, i64 %135
  %137 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !412
  %138 = mul nsw i64 %11, %130, !spirv.Decorations !398
  %139 = select i1 %137, i64 %138, i64 0
  %140 = getelementptr inbounds float, float addrspace(4)* %54, i64 %139
  %141 = mul nsw i64 %12, %130, !spirv.Decorations !398
  %142 = getelementptr inbounds float, float addrspace(4)* %53, i64 %141
  %143 = bitcast [2 x i32]* %15 to i32*
  %144 = getelementptr inbounds i32, i32* %143, i64 1
  %145 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %19, i64 0, i32 0
  %146 = bitcast %"struct.cutlass::Coord.8930"* %19 to %structtype.0*
  %147 = getelementptr inbounds %structtype.0, %structtype.0* %146, i64 0, i32 0
  %148 = bitcast [2 x i32]* %16 to i32*
  %149 = getelementptr inbounds i32, i32* %148, i64 1
  %150 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %18, i64 0, i32 0
  %151 = bitcast %"struct.cutlass::Coord.8930"* %18 to %structtype.0*
  %152 = getelementptr inbounds %structtype.0, %structtype.0* %151, i64 0, i32 0
  %153 = bitcast [2 x i32]* %14 to i32*
  %154 = getelementptr inbounds i32, i32* %153, i64 1
  %155 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %17, i64 0, i32 0
  %156 = bitcast %"struct.cutlass::Coord.8930"* %17 to %structtype.0*
  %157 = getelementptr inbounds %structtype.0, %structtype.0* %156, i64 0, i32 0
  br label %158

158:                                              ; preds = %345, %13
  %159 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %134, %13 ], [ %283, %345 ]
  %160 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %136, %13 ], [ %285, %345 ]
  %161 = phi float addrspace(4)* [ %140, %13 ], [ %346, %345 ]
  %162 = phi float addrspace(4)* [ %142, %13 ], [ %358, %345 ]
  %163 = phi i32 [ %132, %13 ], [ %360, %345 ]
  %164 = icmp slt i32 %163, %8
  br i1 %164, label %165, label %361

165:                                              ; preds = %158
  %166 = bitcast [4 x [16 x float]]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %166)
  br label %167

167:                                              ; preds = %175, %165
  %168 = phi i32 [ 0, %165 ], [ %176, %175 ]
  %169 = icmp ult i32 %168, 16
  br i1 %169, label %170, label %.preheader2

.preheader2:                                      ; preds = %167
  br label %181

170:                                              ; preds = %167
  %171 = sext i32 %168 to i64
  br label %172

172:                                              ; preds = %177, %170
  %173 = phi i32 [ %180, %177 ], [ 0, %170 ]
  %174 = icmp ult i32 %173, 4
  br i1 %174, label %177, label %175

175:                                              ; preds = %172
  %176 = add nuw nsw i32 %168, 1, !spirv.Decorations !395
  br label %167, !llvm.loop !422

177:                                              ; preds = %172
  %178 = zext i32 %173 to i64
  %179 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %178, i64 %171
  store float %7, float* %179, align 4
  %180 = add nuw nsw i32 %173, 1, !spirv.Decorations !395
  br label %172, !llvm.loop !424

181:                                              ; preds = %191, %.preheader2
  %182 = phi i32 [ %192, %191 ], [ 0, %.preheader2 ]
  %183 = icmp slt i32 %182, %28
  br i1 %183, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %181
  br label %265

.preheader:                                       ; preds = %181
  br label %184

184:                                              ; preds = %196, %.preheader
  %185 = phi i32 [ %197, %196 ], [ 0, %.preheader ]
  %186 = icmp ult i32 %185, 16
  br i1 %186, label %187, label %191

187:                                              ; preds = %184
  %188 = or i32 %122, %185
  %189 = icmp slt i32 %188, %25
  %190 = sext i32 %185 to i64
  br label %193

191:                                              ; preds = %184
  %192 = add nuw nsw i32 %182, 1, !spirv.Decorations !395
  br label %181

193:                                              ; preds = %263, %187
  %194 = phi i32 [ %264, %263 ], [ 0, %187 ]
  %195 = icmp ult i32 %194, 4
  br i1 %195, label %198, label %196

196:                                              ; preds = %193
  %197 = add nuw nsw i32 %185, 1, !spirv.Decorations !395
  br label %184, !llvm.loop !425

198:                                              ; preds = %193
  %199 = or i32 %89, %194
  %200 = icmp slt i32 %199, %22
  %201 = select i1 %200, i1 %189, i1 false
  br i1 %201, label %202, label %263

202:                                              ; preds = %198
  %203 = bitcast %"struct.cutlass::Coord.8930"* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %203)
  %204 = bitcast [2 x i32]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %204)
  %205 = bitcast [2 x i32]* %15 to i32*
  store i32 %199, i32* %205, align 4, !noalias !426
  store i32 %182, i32* %144, align 4, !noalias !426
  br label %206

206:                                              ; preds = %209, %202
  %207 = phi i32 [ 0, %202 ], [ %214, %209 ]
  %208 = icmp ult i32 %207, 2
  br i1 %208, label %209, label %215

209:                                              ; preds = %206
  %210 = zext i32 %207 to i64
  %211 = getelementptr inbounds [2 x i32], [2 x i32]* %15, i64 0, i64 %210
  %212 = load i32, i32* %211, align 4, !noalias !426
  %213 = getelementptr inbounds [2 x i32], [2 x i32]* %145, i64 0, i64 %210
  store i32 %212, i32* %213, align 4, !alias.scope !426
  %214 = add nuw nsw i32 %207, 1, !spirv.Decorations !395
  br label %206

215:                                              ; preds = %206
  %216 = bitcast [2 x i32]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %216)
  %217 = load i64, i64* %147, align 4
  %218 = bitcast %"struct.cutlass::Coord.8930"* %19 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %218)
  %219 = shl i64 %217, 32
  %220 = ashr i64 %219, 32
  %221 = mul nsw i64 %34, %220, !spirv.Decorations !398
  %222 = ashr i64 %217, 32
  %223 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %159, i64 %221
  %224 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %223 to i16 addrspace(4)*
  %225 = getelementptr i16, i16 addrspace(4)* %224, i64 %222
  %226 = load i16, i16 addrspace(4)* %225, align 2
  %227 = bitcast %"struct.cutlass::Coord.8930"* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %227)
  %228 = bitcast [2 x i32]* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %228)
  %229 = bitcast [2 x i32]* %16 to i32*
  store i32 %182, i32* %229, align 4, !noalias !429
  store i32 %188, i32* %149, align 4, !noalias !429
  br label %230

230:                                              ; preds = %233, %215
  %231 = phi i32 [ 0, %215 ], [ %238, %233 ]
  %232 = icmp ult i32 %231, 2
  br i1 %232, label %233, label %239

233:                                              ; preds = %230
  %234 = zext i32 %231 to i64
  %235 = getelementptr inbounds [2 x i32], [2 x i32]* %16, i64 0, i64 %234
  %236 = load i32, i32* %235, align 4, !noalias !429
  %237 = getelementptr inbounds [2 x i32], [2 x i32]* %150, i64 0, i64 %234
  store i32 %236, i32* %237, align 4, !alias.scope !429
  %238 = add nuw nsw i32 %231, 1, !spirv.Decorations !395
  br label %230

239:                                              ; preds = %230
  %240 = bitcast [2 x i32]* %16 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %240)
  %241 = load i64, i64* %152, align 4
  %242 = bitcast %"struct.cutlass::Coord.8930"* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %242)
  %243 = ashr i64 %241, 32
  %244 = mul nsw i64 %40, %243, !spirv.Decorations !398
  %245 = shl i64 %241, 32
  %246 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %160, i64 %244
  %247 = ashr i64 %245, 31
  %248 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %246 to i8 addrspace(4)*
  %249 = getelementptr i8, i8 addrspace(4)* %248, i64 %247
  %250 = bitcast i8 addrspace(4)* %249 to i16 addrspace(4)*
  %251 = load i16, i16 addrspace(4)* %250, align 2
  %252 = zext i16 %226 to i32
  %253 = shl nuw i32 %252, 16, !spirv.Decorations !403
  %254 = bitcast i32 %253 to float
  %255 = zext i16 %251 to i32
  %256 = shl nuw i32 %255, 16, !spirv.Decorations !403
  %257 = bitcast i32 %256 to float
  %258 = zext i32 %194 to i64
  %259 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %258, i64 %190
  %260 = fmul reassoc nsz arcp contract float %254, %257, !spirv.Decorations !412
  %261 = load float, float* %259, align 4
  %262 = fadd reassoc nsz arcp contract float %260, %261, !spirv.Decorations !412
  store float %262, float* %259, align 4
  br label %263

263:                                              ; preds = %239, %198
  %264 = add nuw nsw i32 %194, 1, !spirv.Decorations !395
  br label %193, !llvm.loop !432

265:                                              ; preds = %289, %.preheader1
  %266 = phi i32 [ %290, %289 ], [ 0, %.preheader1 ]
  %267 = icmp ult i32 %266, 16
  br i1 %267, label %268, label %272

268:                                              ; preds = %265
  %269 = or i32 %122, %266
  %270 = icmp slt i32 %269, %25
  %271 = sext i32 %266 to i64
  br label %286

272:                                              ; preds = %265
  %273 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %274 = insertelement <3 x i64> undef, i64 %273, i32 0
  %275 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %276 = insertelement <3 x i64> %274, i64 %275, i32 1
  %277 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %278 = insertelement <3 x i64> %276, i64 %277, i32 2
  %279 = extractelement <3 x i64> %278, i32 2
  %280 = select i1 true, i64 %279, i64 1
  %281 = icmp ult i64 %280, 2147483648
  call void @llvm.assume(i1 %281)
  %282 = mul nsw i64 %9, %280, !spirv.Decorations !398
  %283 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %159, i64 %282
  %284 = mul nsw i64 %10, %280, !spirv.Decorations !398
  %285 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %160, i64 %284
  br i1 %137, label %333, label %345

286:                                              ; preds = %331, %268
  %287 = phi i32 [ %332, %331 ], [ 0, %268 ]
  %288 = icmp ult i32 %287, 4
  br i1 %288, label %291, label %289

289:                                              ; preds = %286
  %290 = add nuw nsw i32 %266, 1, !spirv.Decorations !395
  br label %265, !llvm.loop !433

291:                                              ; preds = %286
  %292 = or i32 %89, %287
  %293 = bitcast %"struct.cutlass::Coord.8930"* %17 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %293)
  %294 = bitcast [2 x i32]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %294)
  %295 = bitcast [2 x i32]* %14 to i32*
  store i32 %292, i32* %295, align 4, !noalias !434
  store i32 %269, i32* %154, align 4, !noalias !434
  br label %296

296:                                              ; preds = %299, %291
  %297 = phi i32 [ 0, %291 ], [ %304, %299 ]
  %298 = icmp ult i32 %297, 2
  br i1 %298, label %299, label %305

299:                                              ; preds = %296
  %300 = zext i32 %297 to i64
  %301 = getelementptr inbounds [2 x i32], [2 x i32]* %14, i64 0, i64 %300
  %302 = load i32, i32* %301, align 4, !noalias !434
  %303 = getelementptr inbounds [2 x i32], [2 x i32]* %155, i64 0, i64 %300
  store i32 %302, i32* %303, align 4, !alias.scope !434
  %304 = add nuw nsw i32 %297, 1, !spirv.Decorations !395
  br label %296

305:                                              ; preds = %296
  %306 = bitcast [2 x i32]* %14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %306)
  %307 = load i64, i64* %157, align 4
  %308 = bitcast %"struct.cutlass::Coord.8930"* %17 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %308)
  %309 = icmp slt i32 %292, %22
  %310 = select i1 %309, i1 %270, i1 false
  br i1 %310, label %311, label %331

311:                                              ; preds = %305
  %312 = zext i32 %287 to i64
  %313 = shl i64 %307, 32
  %314 = ashr i64 %313, 32
  %315 = ashr i64 %307, 32
  %316 = mul nsw i64 %52, %314, !spirv.Decorations !398
  %317 = add nsw i64 %316, %315, !spirv.Decorations !398
  %318 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %312, i64 %271
  %319 = load float, float* %318, align 4
  %320 = fmul reassoc nsz arcp contract float %1, %319, !spirv.Decorations !412
  br i1 %137, label %321, label %329

321:                                              ; preds = %311
  %322 = mul nsw i64 %46, %314, !spirv.Decorations !398
  %323 = getelementptr float, float addrspace(4)* %161, i64 %322
  %324 = getelementptr float, float addrspace(4)* %323, i64 %315
  %325 = load float, float addrspace(4)* %324, align 4
  %326 = fmul reassoc nsz arcp contract float %4, %325, !spirv.Decorations !412
  %327 = fadd reassoc nsz arcp contract float %320, %326, !spirv.Decorations !412
  %328 = getelementptr inbounds float, float addrspace(4)* %162, i64 %317
  store float %327, float addrspace(4)* %328, align 4
  br label %331

329:                                              ; preds = %311
  %330 = getelementptr inbounds float, float addrspace(4)* %162, i64 %317
  store float %320, float addrspace(4)* %330, align 4
  br label %331

331:                                              ; preds = %329, %321, %305
  %332 = add nuw nsw i32 %287, 1, !spirv.Decorations !395
  br label %286, !llvm.loop !437

333:                                              ; preds = %272
  %334 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %335 = insertelement <3 x i64> undef, i64 %334, i32 0
  %336 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %337 = insertelement <3 x i64> %335, i64 %336, i32 1
  %338 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %339 = insertelement <3 x i64> %337, i64 %338, i32 2
  %340 = extractelement <3 x i64> %339, i32 2
  %341 = select i1 true, i64 %340, i64 1
  %342 = icmp ult i64 %341, 2147483648
  call void @llvm.assume(i1 %342)
  %343 = mul nsw i64 %11, %341, !spirv.Decorations !398
  %344 = getelementptr inbounds float, float addrspace(4)* %161, i64 %343
  br label %345

345:                                              ; preds = %333, %272
  %346 = phi float addrspace(4)* [ %344, %333 ], [ %161, %272 ]
  %347 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %348 = insertelement <3 x i64> undef, i64 %347, i32 0
  %349 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %350 = insertelement <3 x i64> %348, i64 %349, i32 1
  %351 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %352 = insertelement <3 x i64> %350, i64 %351, i32 2
  %353 = extractelement <3 x i64> %352, i32 2
  %354 = select i1 true, i64 %353, i64 1
  %355 = icmp ult i64 %354, 2147483648
  call void @llvm.assume(i1 %355)
  %356 = trunc i64 %354 to i32
  %357 = mul nsw i64 %12, %354, !spirv.Decorations !398
  %358 = getelementptr inbounds float, float addrspace(4)* %162, i64 %357
  %359 = bitcast [4 x [16 x float]]* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %359)
  %360 = add i32 %163, %356
  br label %158

361:                                              ; preds = %158
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12) #3 !spirv.ParameterDecorations !414 !kernel_arg_addr_space !418 !kernel_arg_access_qual !419 !kernel_arg_type !420 !kernel_arg_type_qual !421 !kernel_arg_base_type !420 !kernel_arg_name !421 {
  %14 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %15 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %16 = alloca [2 x i32], align 4, !spirv.Decorations !393
  %17 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %18 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %19 = alloca %"struct.cutlass::Coord.8930", align 4, !spirv.Decorations !393
  %20 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !393
  %21 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %22 = load i32, i32* %21, align 4
  %23 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %24 = getelementptr inbounds i32, i32* %23, i64 1
  %25 = load i32, i32* %24, align 4
  %26 = bitcast %"struct.cutlass::gemm::GemmCoord"* %0 to i32*
  %27 = getelementptr inbounds i32, i32* %26, i64 2
  %28 = load i32, i32* %27, align 4
  %29 = bitcast %"class.cutlass::__generated_TensorRef"* %2 to %structtype*
  %30 = getelementptr inbounds %structtype, %structtype* %29, i64 0, i32 0
  %31 = load i64, i64* %30, align 8
  %32 = bitcast %"class.cutlass::__generated_TensorRef"* %2 to %structtype*
  %33 = getelementptr inbounds %structtype, %structtype* %32, i64 0, i32 1
  %34 = load i64, i64* %33, align 8
  %35 = bitcast %"class.cutlass::__generated_TensorRef"* %3 to %structtype*
  %36 = getelementptr inbounds %structtype, %structtype* %35, i64 0, i32 0
  %37 = load i64, i64* %36, align 8
  %38 = bitcast %"class.cutlass::__generated_TensorRef"* %3 to %structtype*
  %39 = getelementptr inbounds %structtype, %structtype* %38, i64 0, i32 1
  %40 = load i64, i64* %39, align 8
  %41 = bitcast %"class.cutlass::__generated_TensorRef"* %5 to %structtype*
  %42 = getelementptr inbounds %structtype, %structtype* %41, i64 0, i32 0
  %43 = load i64, i64* %42, align 8
  %44 = bitcast %"class.cutlass::__generated_TensorRef"* %5 to %structtype*
  %45 = getelementptr inbounds %structtype, %structtype* %44, i64 0, i32 1
  %46 = load i64, i64* %45, align 8
  %47 = bitcast %"class.cutlass::__generated_TensorRef"* %6 to %structtype*
  %48 = getelementptr inbounds %structtype, %structtype* %47, i64 0, i32 0
  %49 = load i64, i64* %48, align 8
  %50 = bitcast %"class.cutlass::__generated_TensorRef"* %6 to %structtype*
  %51 = getelementptr inbounds %structtype, %structtype* %50, i64 0, i32 1
  %52 = load i64, i64* %51, align 8
  %53 = inttoptr i64 %49 to float addrspace(4)*
  %54 = inttoptr i64 %43 to float addrspace(4)*
  %55 = inttoptr i64 %37 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %56 = inttoptr i64 %31 to %"struct.cutlass::bfloat16_t" addrspace(4)*
  %57 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %58 = insertelement <3 x i64> undef, i64 %57, i32 0
  %59 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %60 = insertelement <3 x i64> %58, i64 %59, i32 1
  %61 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %62 = insertelement <3 x i64> %60, i64 %61, i32 2
  %63 = extractelement <3 x i64> %62, i32 0
  %64 = select i1 true, i64 %63, i64 0
  %65 = icmp ult i64 %64, 2147483648
  call void @llvm.assume(i1 %65)
  %66 = trunc i64 %64 to i32
  %67 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %68 = insertelement <3 x i64> undef, i64 %67, i32 0
  %69 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %70 = insertelement <3 x i64> %68, i64 %69, i32 1
  %71 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %72 = insertelement <3 x i64> %70, i64 %71, i32 2
  %73 = extractelement <3 x i64> %72, i32 0
  %74 = select i1 true, i64 %73, i64 1
  %75 = icmp ult i64 %74, 2147483648
  call void @llvm.assume(i1 %75)
  %76 = trunc i64 %74 to i32
  %77 = mul i32 %66, %76
  %78 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %79 = insertelement <3 x i64> undef, i64 %78, i32 0
  %80 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %81 = insertelement <3 x i64> %79, i64 %80, i32 1
  %82 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #5
  %83 = insertelement <3 x i64> %81, i64 %82, i32 2
  %84 = extractelement <3 x i64> %83, i32 0
  %85 = select i1 true, i64 %84, i64 0
  %86 = icmp ult i64 %85, 2147483648
  call void @llvm.assume(i1 %86)
  %87 = trunc i64 %85 to i32
  %88 = add i32 %77, %87
  %89 = shl i32 %88, 2
  %90 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %91 = insertelement <3 x i64> undef, i64 %90, i32 0
  %92 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %93 = insertelement <3 x i64> %91, i64 %92, i32 1
  %94 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %95 = insertelement <3 x i64> %93, i64 %94, i32 2
  %96 = extractelement <3 x i64> %95, i32 1
  %97 = select i1 true, i64 %96, i64 0
  %98 = icmp ult i64 %97, 2147483648
  call void @llvm.assume(i1 %98)
  %99 = trunc i64 %97 to i32
  %100 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %101 = insertelement <3 x i64> undef, i64 %100, i32 0
  %102 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %103 = insertelement <3 x i64> %101, i64 %102, i32 1
  %104 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %105 = insertelement <3 x i64> %103, i64 %104, i32 2
  %106 = extractelement <3 x i64> %105, i32 1
  %107 = select i1 true, i64 %106, i64 1
  %108 = icmp ult i64 %107, 2147483648
  call void @llvm.assume(i1 %108)
  %109 = trunc i64 %107 to i32
  %110 = mul i32 %99, %109
  %111 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %112 = insertelement <3 x i64> undef, i64 %111, i32 0
  %113 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %114 = insertelement <3 x i64> %112, i64 %113, i32 1
  %115 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #5
  %116 = insertelement <3 x i64> %114, i64 %115, i32 2
  %117 = extractelement <3 x i64> %116, i32 1
  %118 = select i1 true, i64 %117, i64 0
  %119 = icmp ult i64 %118, 2147483648
  call void @llvm.assume(i1 %119)
  %120 = trunc i64 %118 to i32
  %121 = add i32 %110, %120
  %122 = shl i32 %121, 4
  %123 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %124 = insertelement <3 x i64> undef, i64 %123, i32 0
  %125 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %126 = insertelement <3 x i64> %124, i64 %125, i32 1
  %127 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %128 = insertelement <3 x i64> %126, i64 %127, i32 2
  %129 = extractelement <3 x i64> %128, i32 2
  %130 = select i1 true, i64 %129, i64 0
  %131 = icmp ult i64 %130, 2147483648
  call void @llvm.assume(i1 %131)
  %132 = trunc i64 %130 to i32
  %133 = mul nsw i64 %9, %130, !spirv.Decorations !398
  %134 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %56, i64 %133
  %135 = mul nsw i64 %10, %130, !spirv.Decorations !398
  %136 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %55, i64 %135
  %137 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !412
  %138 = mul nsw i64 %11, %130, !spirv.Decorations !398
  %139 = select i1 %137, i64 %138, i64 0
  %140 = getelementptr inbounds float, float addrspace(4)* %54, i64 %139
  %141 = mul nsw i64 %12, %130, !spirv.Decorations !398
  %142 = getelementptr inbounds float, float addrspace(4)* %53, i64 %141
  %143 = bitcast [2 x i32]* %15 to i32*
  %144 = getelementptr inbounds i32, i32* %143, i64 1
  %145 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %19, i64 0, i32 0
  %146 = bitcast %"struct.cutlass::Coord.8930"* %19 to %structtype.0*
  %147 = getelementptr inbounds %structtype.0, %structtype.0* %146, i64 0, i32 0
  %148 = bitcast [2 x i32]* %16 to i32*
  %149 = getelementptr inbounds i32, i32* %148, i64 1
  %150 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %18, i64 0, i32 0
  %151 = bitcast %"struct.cutlass::Coord.8930"* %18 to %structtype.0*
  %152 = getelementptr inbounds %structtype.0, %structtype.0* %151, i64 0, i32 0
  %153 = bitcast [2 x i32]* %14 to i32*
  %154 = getelementptr inbounds i32, i32* %153, i64 1
  %155 = getelementptr inbounds %"struct.cutlass::Coord.8930", %"struct.cutlass::Coord.8930"* %17, i64 0, i32 0
  %156 = bitcast %"struct.cutlass::Coord.8930"* %17 to %structtype.0*
  %157 = getelementptr inbounds %structtype.0, %structtype.0* %156, i64 0, i32 0
  br label %158

158:                                              ; preds = %344, %13
  %159 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %134, %13 ], [ %282, %344 ]
  %160 = phi %"struct.cutlass::bfloat16_t" addrspace(4)* [ %136, %13 ], [ %284, %344 ]
  %161 = phi float addrspace(4)* [ %140, %13 ], [ %345, %344 ]
  %162 = phi float addrspace(4)* [ %142, %13 ], [ %357, %344 ]
  %163 = phi i32 [ %132, %13 ], [ %359, %344 ]
  %164 = icmp slt i32 %163, %8
  br i1 %164, label %165, label %360

165:                                              ; preds = %158
  %166 = bitcast [4 x [16 x float]]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 256, i8* %166)
  br label %167

167:                                              ; preds = %175, %165
  %168 = phi i32 [ 0, %165 ], [ %176, %175 ]
  %169 = icmp ult i32 %168, 16
  br i1 %169, label %170, label %.preheader2

.preheader2:                                      ; preds = %167
  br label %181

170:                                              ; preds = %167
  %171 = sext i32 %168 to i64
  br label %172

172:                                              ; preds = %177, %170
  %173 = phi i32 [ %180, %177 ], [ 0, %170 ]
  %174 = icmp ult i32 %173, 4
  br i1 %174, label %177, label %175

175:                                              ; preds = %172
  %176 = add nuw nsw i32 %168, 1, !spirv.Decorations !395
  br label %167, !llvm.loop !438

177:                                              ; preds = %172
  %178 = zext i32 %173 to i64
  %179 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %178, i64 %171
  store float %7, float* %179, align 4
  %180 = add nuw nsw i32 %173, 1, !spirv.Decorations !395
  br label %172, !llvm.loop !439

181:                                              ; preds = %191, %.preheader2
  %182 = phi i32 [ %192, %191 ], [ 0, %.preheader2 ]
  %183 = icmp slt i32 %182, %28
  br i1 %183, label %.preheader, label %.preheader1

.preheader1:                                      ; preds = %181
  br label %264

.preheader:                                       ; preds = %181
  br label %184

184:                                              ; preds = %196, %.preheader
  %185 = phi i32 [ %197, %196 ], [ 0, %.preheader ]
  %186 = icmp ult i32 %185, 16
  br i1 %186, label %187, label %191

187:                                              ; preds = %184
  %188 = or i32 %122, %185
  %189 = icmp slt i32 %188, %25
  %190 = sext i32 %185 to i64
  br label %193

191:                                              ; preds = %184
  %192 = add nuw nsw i32 %182, 1, !spirv.Decorations !395
  br label %181

193:                                              ; preds = %262, %187
  %194 = phi i32 [ %263, %262 ], [ 0, %187 ]
  %195 = icmp ult i32 %194, 4
  br i1 %195, label %198, label %196

196:                                              ; preds = %193
  %197 = add nuw nsw i32 %185, 1, !spirv.Decorations !395
  br label %184, !llvm.loop !440

198:                                              ; preds = %193
  %199 = or i32 %89, %194
  %200 = icmp slt i32 %199, %22
  %201 = select i1 %200, i1 %189, i1 false
  br i1 %201, label %202, label %262

202:                                              ; preds = %198
  %203 = bitcast %"struct.cutlass::Coord.8930"* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %203)
  %204 = bitcast [2 x i32]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %204)
  %205 = bitcast [2 x i32]* %15 to i32*
  store i32 %199, i32* %205, align 4, !noalias !441
  store i32 %182, i32* %144, align 4, !noalias !441
  br label %206

206:                                              ; preds = %209, %202
  %207 = phi i32 [ 0, %202 ], [ %214, %209 ]
  %208 = icmp ult i32 %207, 2
  br i1 %208, label %209, label %215

209:                                              ; preds = %206
  %210 = zext i32 %207 to i64
  %211 = getelementptr inbounds [2 x i32], [2 x i32]* %15, i64 0, i64 %210
  %212 = load i32, i32* %211, align 4, !noalias !441
  %213 = getelementptr inbounds [2 x i32], [2 x i32]* %145, i64 0, i64 %210
  store i32 %212, i32* %213, align 4, !alias.scope !441
  %214 = add nuw nsw i32 %207, 1, !spirv.Decorations !395
  br label %206

215:                                              ; preds = %206
  %216 = bitcast [2 x i32]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %216)
  %217 = load i64, i64* %147, align 4
  %218 = bitcast %"struct.cutlass::Coord.8930"* %19 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %218)
  %219 = shl i64 %217, 32
  %220 = ashr i64 %219, 32
  %221 = mul nsw i64 %34, %220, !spirv.Decorations !398
  %222 = ashr i64 %217, 32
  %223 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %159, i64 %221
  %224 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %223 to i16 addrspace(4)*
  %225 = getelementptr i16, i16 addrspace(4)* %224, i64 %222
  %226 = load i16, i16 addrspace(4)* %225, align 2
  %227 = bitcast %"struct.cutlass::Coord.8930"* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %227)
  %228 = bitcast [2 x i32]* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %228)
  %229 = bitcast [2 x i32]* %16 to i32*
  store i32 %182, i32* %229, align 4, !noalias !444
  store i32 %188, i32* %149, align 4, !noalias !444
  br label %230

230:                                              ; preds = %233, %215
  %231 = phi i32 [ 0, %215 ], [ %238, %233 ]
  %232 = icmp ult i32 %231, 2
  br i1 %232, label %233, label %239

233:                                              ; preds = %230
  %234 = zext i32 %231 to i64
  %235 = getelementptr inbounds [2 x i32], [2 x i32]* %16, i64 0, i64 %234
  %236 = load i32, i32* %235, align 4, !noalias !444
  %237 = getelementptr inbounds [2 x i32], [2 x i32]* %150, i64 0, i64 %234
  store i32 %236, i32* %237, align 4, !alias.scope !444
  %238 = add nuw nsw i32 %231, 1, !spirv.Decorations !395
  br label %230

239:                                              ; preds = %230
  %240 = bitcast [2 x i32]* %16 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %240)
  %241 = load i64, i64* %152, align 4
  %242 = bitcast %"struct.cutlass::Coord.8930"* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %242)
  %243 = shl i64 %241, 32
  %244 = ashr i64 %243, 32
  %245 = mul nsw i64 %40, %244, !spirv.Decorations !398
  %246 = ashr i64 %241, 32
  %247 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %160, i64 %245
  %248 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %247 to i16 addrspace(4)*
  %249 = getelementptr i16, i16 addrspace(4)* %248, i64 %246
  %250 = load i16, i16 addrspace(4)* %249, align 2
  %251 = zext i16 %226 to i32
  %252 = shl nuw i32 %251, 16, !spirv.Decorations !403
  %253 = bitcast i32 %252 to float
  %254 = zext i16 %250 to i32
  %255 = shl nuw i32 %254, 16, !spirv.Decorations !403
  %256 = bitcast i32 %255 to float
  %257 = zext i32 %194 to i64
  %258 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %257, i64 %190
  %259 = fmul reassoc nsz arcp contract float %253, %256, !spirv.Decorations !412
  %260 = load float, float* %258, align 4
  %261 = fadd reassoc nsz arcp contract float %259, %260, !spirv.Decorations !412
  store float %261, float* %258, align 4
  br label %262

262:                                              ; preds = %239, %198
  %263 = add nuw nsw i32 %194, 1, !spirv.Decorations !395
  br label %193, !llvm.loop !447

264:                                              ; preds = %288, %.preheader1
  %265 = phi i32 [ %289, %288 ], [ 0, %.preheader1 ]
  %266 = icmp ult i32 %265, 16
  br i1 %266, label %267, label %271

267:                                              ; preds = %264
  %268 = or i32 %122, %265
  %269 = icmp slt i32 %268, %25
  %270 = sext i32 %265 to i64
  br label %285

271:                                              ; preds = %264
  %272 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %273 = insertelement <3 x i64> undef, i64 %272, i32 0
  %274 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %275 = insertelement <3 x i64> %273, i64 %274, i32 1
  %276 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %277 = insertelement <3 x i64> %275, i64 %276, i32 2
  %278 = extractelement <3 x i64> %277, i32 2
  %279 = select i1 true, i64 %278, i64 1
  %280 = icmp ult i64 %279, 2147483648
  call void @llvm.assume(i1 %280)
  %281 = mul nsw i64 %9, %279, !spirv.Decorations !398
  %282 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %159, i64 %281
  %283 = mul nsw i64 %10, %279, !spirv.Decorations !398
  %284 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %160, i64 %283
  br i1 %137, label %332, label %344

285:                                              ; preds = %330, %267
  %286 = phi i32 [ %331, %330 ], [ 0, %267 ]
  %287 = icmp ult i32 %286, 4
  br i1 %287, label %290, label %288

288:                                              ; preds = %285
  %289 = add nuw nsw i32 %265, 1, !spirv.Decorations !395
  br label %264, !llvm.loop !448

290:                                              ; preds = %285
  %291 = or i32 %89, %286
  %292 = bitcast %"struct.cutlass::Coord.8930"* %17 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %292)
  %293 = bitcast [2 x i32]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %293)
  %294 = bitcast [2 x i32]* %14 to i32*
  store i32 %291, i32* %294, align 4, !noalias !449
  store i32 %268, i32* %154, align 4, !noalias !449
  br label %295

295:                                              ; preds = %298, %290
  %296 = phi i32 [ 0, %290 ], [ %303, %298 ]
  %297 = icmp ult i32 %296, 2
  br i1 %297, label %298, label %304

298:                                              ; preds = %295
  %299 = zext i32 %296 to i64
  %300 = getelementptr inbounds [2 x i32], [2 x i32]* %14, i64 0, i64 %299
  %301 = load i32, i32* %300, align 4, !noalias !449
  %302 = getelementptr inbounds [2 x i32], [2 x i32]* %155, i64 0, i64 %299
  store i32 %301, i32* %302, align 4, !alias.scope !449
  %303 = add nuw nsw i32 %296, 1, !spirv.Decorations !395
  br label %295

304:                                              ; preds = %295
  %305 = bitcast [2 x i32]* %14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %305)
  %306 = load i64, i64* %157, align 4
  %307 = bitcast %"struct.cutlass::Coord.8930"* %17 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %307)
  %308 = icmp slt i32 %291, %22
  %309 = select i1 %308, i1 %269, i1 false
  br i1 %309, label %310, label %330

310:                                              ; preds = %304
  %311 = zext i32 %286 to i64
  %312 = shl i64 %306, 32
  %313 = ashr i64 %312, 32
  %314 = ashr i64 %306, 32
  %315 = mul nsw i64 %52, %313, !spirv.Decorations !398
  %316 = add nsw i64 %315, %314, !spirv.Decorations !398
  %317 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %20, i64 0, i64 %311, i64 %270
  %318 = load float, float* %317, align 4
  %319 = fmul reassoc nsz arcp contract float %1, %318, !spirv.Decorations !412
  br i1 %137, label %320, label %328

320:                                              ; preds = %310
  %321 = mul nsw i64 %46, %313, !spirv.Decorations !398
  %322 = getelementptr float, float addrspace(4)* %161, i64 %321
  %323 = getelementptr float, float addrspace(4)* %322, i64 %314
  %324 = load float, float addrspace(4)* %323, align 4
  %325 = fmul reassoc nsz arcp contract float %4, %324, !spirv.Decorations !412
  %326 = fadd reassoc nsz arcp contract float %319, %325, !spirv.Decorations !412
  %327 = getelementptr inbounds float, float addrspace(4)* %162, i64 %316
  store float %326, float addrspace(4)* %327, align 4
  br label %330

328:                                              ; preds = %310
  %329 = getelementptr inbounds float, float addrspace(4)* %162, i64 %316
  store float %319, float addrspace(4)* %329, align 4
  br label %330

330:                                              ; preds = %328, %320, %304
  %331 = add nuw nsw i32 %286, 1, !spirv.Decorations !395
  br label %285, !llvm.loop !452

332:                                              ; preds = %271
  %333 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %334 = insertelement <3 x i64> undef, i64 %333, i32 0
  %335 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %336 = insertelement <3 x i64> %334, i64 %335, i32 1
  %337 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %338 = insertelement <3 x i64> %336, i64 %337, i32 2
  %339 = extractelement <3 x i64> %338, i32 2
  %340 = select i1 true, i64 %339, i64 1
  %341 = icmp ult i64 %340, 2147483648
  call void @llvm.assume(i1 %341)
  %342 = mul nsw i64 %11, %340, !spirv.Decorations !398
  %343 = getelementptr inbounds float, float addrspace(4)* %161, i64 %342
  br label %344

344:                                              ; preds = %332, %271
  %345 = phi float addrspace(4)* [ %343, %332 ], [ %161, %271 ]
  %346 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 0) #5
  %347 = insertelement <3 x i64> undef, i64 %346, i32 0
  %348 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 1) #5
  %349 = insertelement <3 x i64> %347, i64 %348, i32 1
  %350 = call spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32 2) #5
  %351 = insertelement <3 x i64> %349, i64 %350, i32 2
  %352 = extractelement <3 x i64> %351, i32 2
  %353 = select i1 true, i64 %352, i64 1
  %354 = icmp ult i64 %353, 2147483648
  call void @llvm.assume(i1 %354)
  %355 = trunc i64 %353 to i32
  %356 = mul nsw i64 %12, %353, !spirv.Decorations !398
  %357 = getelementptr inbounds float, float addrspace(4)* %162, i64 %356
  %358 = bitcast [4 x [16 x float]]* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* %358)
  %359 = add i32 %163, %355
  br label %158

360:                                              ; preds = %158
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) #5

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z25__spirv_BuiltInGlobalSizei(i32) #5

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32) #5

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #5

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32) #5

; Function Attrs: nounwind willreturn memory(none)
declare spir_func i64 @_Z28__spirv_BuiltInNumWorkgroupsi(i32) #5

; Function Attrs: alwaysinline nounwind
define spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(float addrspace(4)* align 4 dereferenceable(4) %0) #6 !spirv.ParameterDecorations !453 {
  %2 = load float, float addrspace(4)* %0, align 4
  %3 = call spir_func i16 @_Z27__spirv_ConvertFToBF16INTELf(float %2) #3
  ret i16 %3
}

; Function Attrs: nounwind
declare spir_func i16 @_Z27__spirv_ConvertFToBF16INTELf(float) #3

; Function Attrs: alwaysinline nounwind
define spir_func float @__devicelib_ConvertBF16ToFINTEL(i16 addrspace(4)* align 2 dereferenceable(2) %0) #6 !spirv.ParameterDecorations !456 {
  %2 = load i16, i16 addrspace(4)* %0, align 2
  %3 = call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 %2) #3
  ret float %3
}

; Function Attrs: nounwind
declare spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec1(float addrspace(4)* %0, i16 addrspace(4)* %1) #6 {
  %3 = load float, float addrspace(4)* %0, align 4
  %4 = call spir_func i16 @_Z27__spirv_ConvertFToBF16INTELf(float %3) #3
  store i16 %4, i16 addrspace(4)* %1, align 2
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec1(i16 addrspace(4)* %0, float addrspace(4)* %1) #6 {
  %3 = load i16, i16 addrspace(4)* %0, align 2
  %4 = call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 %3) #3
  store float %4, float addrspace(4)* %1, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec2(<2 x float> addrspace(4)* %0, <2 x i16> addrspace(4)* %1) #6 {
  %3 = load <2 x float>, <2 x float> addrspace(4)* %0, align 8
  %4 = call spir_func <2 x i16> @_Z27__spirv_ConvertFToBF16INTELDv2_f(<2 x float> %3) #3
  store <2 x i16> %4, <2 x i16> addrspace(4)* %1, align 4
  ret void
}

; Function Attrs: nounwind
declare spir_func <2 x i16> @_Z27__spirv_ConvertFToBF16INTELDv2_f(<2 x float>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec2(<2 x i16> addrspace(4)* %0, <2 x float> addrspace(4)* %1) #6 {
  %3 = load <2 x i16>, <2 x i16> addrspace(4)* %0, align 4
  %4 = call spir_func <2 x float> @_Z27__spirv_ConvertBF16ToFINTELDv2_s(<2 x i16> %3) #3
  store <2 x float> %4, <2 x float> addrspace(4)* %1, align 8
  ret void
}

; Function Attrs: nounwind
declare spir_func <2 x float> @_Z27__spirv_ConvertBF16ToFINTELDv2_s(<2 x i16>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec3(<4 x float> addrspace(4)* %0, <4 x i16> addrspace(4)* %1) #6 {
  %3 = load <4 x float>, <4 x float> addrspace(4)* %0, align 16
  %4 = shufflevector <4 x float> %3, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %5 = call spir_func <3 x i16> @_Z27__spirv_ConvertFToBF16INTELDv3_f(<3 x float> %4) #3
  %6 = shufflevector <3 x i16> %5, <3 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %6, <4 x i16> addrspace(4)* %1, align 8
  ret void
}

; Function Attrs: nounwind
declare spir_func <3 x i16> @_Z27__spirv_ConvertFToBF16INTELDv3_f(<3 x float>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec3(<4 x i16> addrspace(4)* %0, <4 x float> addrspace(4)* %1) #6 {
  %3 = load <4 x i16>, <4 x i16> addrspace(4)* %0, align 8
  %4 = shufflevector <4 x i16> %3, <4 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %5 = call spir_func <3 x float> @_Z27__spirv_ConvertBF16ToFINTELDv3_s(<3 x i16> %4) #3
  %6 = shufflevector <3 x float> %5, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x float> %6, <4 x float> addrspace(4)* %1, align 16
  ret void
}

; Function Attrs: nounwind
declare spir_func <3 x float> @_Z27__spirv_ConvertBF16ToFINTELDv3_s(<3 x i16>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec4(<4 x float> addrspace(4)* %0, <4 x i16> addrspace(4)* %1) #6 {
  %3 = load <4 x float>, <4 x float> addrspace(4)* %0, align 16
  %4 = call spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv4_f(<4 x float> %3) #3
  store <4 x i16> %4, <4 x i16> addrspace(4)* %1, align 8
  ret void
}

; Function Attrs: nounwind
declare spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv4_f(<4 x float>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec4(<4 x i16> addrspace(4)* %0, <4 x float> addrspace(4)* %1) #6 {
  %3 = load <4 x i16>, <4 x i16> addrspace(4)* %0, align 8
  %4 = call spir_func <4 x float> @_Z27__spirv_ConvertBF16ToFINTELDv4_s(<4 x i16> %3) #3
  store <4 x float> %4, <4 x float> addrspace(4)* %1, align 16
  ret void
}

; Function Attrs: nounwind
declare spir_func <4 x float> @_Z27__spirv_ConvertBF16ToFINTELDv4_s(<4 x i16>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec8(<8 x float> addrspace(4)* %0, <8 x i16> addrspace(4)* %1) #6 {
  %3 = load <8 x float>, <8 x float> addrspace(4)* %0, align 32
  %4 = call spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float> %3) #3
  store <8 x i16> %4, <8 x i16> addrspace(4)* %1, align 16
  ret void
}

; Function Attrs: nounwind
declare spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec8(<8 x i16> addrspace(4)* %0, <8 x float> addrspace(4)* %1) #6 {
  %3 = load <8 x i16>, <8 x i16> addrspace(4)* %0, align 16
  %4 = call spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16> %3) #3
  store <8 x float> %4, <8 x float> addrspace(4)* %1, align 32
  ret void
}

; Function Attrs: nounwind
declare spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertFToBF16INTELVec16(<16 x float> addrspace(4)* %0, <16 x i16> addrspace(4)* %1) #6 {
  %3 = load <16 x float>, <16 x float> addrspace(4)* %0, align 64
  %4 = call spir_func <16 x i16> @_Z27__spirv_ConvertFToBF16INTELDv16_f(<16 x float> %3) #3
  store <16 x i16> %4, <16 x i16> addrspace(4)* %1, align 32
  ret void
}

; Function Attrs: nounwind
declare spir_func <16 x i16> @_Z27__spirv_ConvertFToBF16INTELDv16_f(<16 x float>) #3

; Function Attrs: alwaysinline nounwind
define spir_func void @__devicelib_ConvertBF16ToFINTELVec16(<16 x i16> addrspace(4)* %0, <16 x float> addrspace(4)* %1) #6 {
  %3 = load <16 x i16>, <16 x i16> addrspace(4)* %0, align 32
  %4 = call spir_func <16 x float> @_Z27__spirv_ConvertBF16ToFINTELDv16_s(<16 x i16> %3) #3
  store <16 x float> %4, <16 x float> addrspace(4)* %1, align 64
  ret void
}

; Function Attrs: nounwind
declare spir_func <16 x float> @_Z27__spirv_ConvertBF16ToFINTELDv16_s(<16 x i16>) #3

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #3 = { nounwind }
attributes #4 = { nounwind memory(none) }
attributes #5 = { nounwind willreturn memory(none) }
attributes #6 = { alwaysinline nounwind }

!spirv.MemoryModel = !{!2, !2}
!spirv.Source = !{!3, !3}
!opencl.spir.version = !{!4, !4}
!opencl.ocl.version = !{!5, !5}
!opencl.used.extensions = !{!6, !6}
!opencl.used.optional.core.features = !{!6, !6}
!spirv.Generator = !{!7, !7}
!opencl.compiler.options = !{!8, !8}
!igc.spirv.extensions = !{!9, !10}
!opencl.enable.FP_CONTRACT = !{}
!igc.functions = !{}
!IGCMetadata = !{!11}

!0 = !{!1}
!1 = !{i32 44, i32 8}
!2 = !{i32 2, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{i32 1, i32 2}
!5 = !{i32 1, i32 0}
!6 = !{}
!7 = !{i16 6, i16 14}
!8 = !{!"-device", !"bmg_g21", !"-ze-opt-level=O2", !"-device", !"bmg_g21", !"-ze-opt-level=O2", !"-cl-intel-256-GRF-per-thread"}
!9 = !{!"SPV_INTEL_fp_fast_math_mode", !"SPV_INTEL_memory_access_aliasing", !"SPV_KHR_expect_assume"}
!10 = !{!"SPV_INTEL_bfloat16_conversion"}
!11 = !{!"ModuleMD", !12, !13, !151, !152, !183, !205, !206, !210, !213, !214, !215, !254, !279, !293, !294, !295, !313, !314, !315, !316, !320, !321, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !341, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !370, !372, !375, !376, !377, !379, !380, !381, !386, !387, !388, !389}
!12 = !{!"isPrecise", i1 false}
!13 = !{!"compOpt", !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149, !150}
!14 = !{!"DenormsAreZero", i1 false}
!15 = !{!"BFTFDenormsAreZero", i1 false}
!16 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!17 = !{!"OptDisable", i1 false}
!18 = !{!"MadEnable", i1 false}
!19 = !{!"NoSignedZeros", i1 false}
!20 = !{!"NoNaNs", i1 false}
!21 = !{!"FloatDenormMode16", !"FLOAT_DENORM_RETAIN"}
!22 = !{!"FloatDenormMode32", !"FLOAT_DENORM_RETAIN"}
!23 = !{!"FloatDenormMode64", !"FLOAT_DENORM_RETAIN"}
!24 = !{!"FloatDenormModeBFTF", !"FLOAT_DENORM_RETAIN"}
!25 = !{!"FloatRoundingMode", i32 0}
!26 = !{!"FloatCvtIntRoundingMode", i32 3}
!27 = !{!"LoadCacheDefault", i32 -1}
!28 = !{!"StoreCacheDefault", i32 -1}
!29 = !{!"VISAPreSchedRPThreshold", i32 0}
!30 = !{!"VISAPreSchedCtrl", i32 0}
!31 = !{!"SetLoopUnrollThreshold", i32 0}
!32 = !{!"UnsafeMathOptimizations", i1 false}
!33 = !{!"disableCustomUnsafeOpts", i1 false}
!34 = !{!"disableReducePow", i1 false}
!35 = !{!"disableSqrtOpt", i1 false}
!36 = !{!"FiniteMathOnly", i1 false}
!37 = !{!"FastRelaxedMath", i1 false}
!38 = !{!"DashGSpecified", i1 false}
!39 = !{!"FastCompilation", i1 false}
!40 = !{!"UseScratchSpacePrivateMemory", i1 true}
!41 = !{!"RelaxedBuiltins", i1 false}
!42 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!43 = !{!"GreaterThan2GBBufferRequired", i1 true}
!44 = !{!"GreaterThan4GBBufferRequired", i1 true}
!45 = !{!"DisableA64WA", i1 false}
!46 = !{!"ForceEnableA64WA", i1 false}
!47 = !{!"PushConstantsEnable", i1 true}
!48 = !{!"HasPositivePointerOffset", i1 false}
!49 = !{!"HasBufferOffsetArg", i1 false}
!50 = !{!"BufferOffsetArgOptional", i1 true}
!51 = !{!"replaceGlobalOffsetsByZero", i1 false}
!52 = !{!"forcePixelShaderSIMDMode", i32 0}
!53 = !{!"forceTotalGRFNum", i32 0}
!54 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!55 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!56 = !{!"UniformWGS", i1 false}
!57 = !{!"disableVertexComponentPacking", i1 false}
!58 = !{!"disablePartialVertexComponentPacking", i1 false}
!59 = !{!"PreferBindlessImages", i1 false}
!60 = !{!"UseBindlessMode", i1 false}
!61 = !{!"UseLegacyBindlessMode", i1 true}
!62 = !{!"disableMathRefactoring", i1 false}
!63 = !{!"atomicBranch", i1 false}
!64 = !{!"spillCompression", i1 false}
!65 = !{!"AllowLICM", i1 true}
!66 = !{!"DisableEarlyOut", i1 false}
!67 = !{!"ForceInt32DivRemEmu", i1 false}
!68 = !{!"ForceInt32DivRemEmuSP", i1 false}
!69 = !{!"DisableIntDivRemIncrementReduction", i1 false}
!70 = !{!"WaveIntrinsicUsed", i1 false}
!71 = !{!"DisableMultiPolyPS", i1 false}
!72 = !{!"NeedTexture3DLODWA", i1 false}
!73 = !{!"UseLivePrologueKernelForRaytracingDispatch", i1 false}
!74 = !{!"DisableFastestSingleCSSIMD", i1 false}
!75 = !{!"DisableFastestLinearScan", i1 false}
!76 = !{!"UseStatelessforPrivateMemory", i1 false}
!77 = !{!"EnableTakeGlobalAddress", i1 false}
!78 = !{!"IsLibraryCompilation", i1 false}
!79 = !{!"LibraryCompileSIMDSize", i32 0}
!80 = !{!"FastVISACompile", i1 false}
!81 = !{!"MatchSinCosPi", i1 false}
!82 = !{!"ExcludeIRFromZEBinary", i1 false}
!83 = !{!"EmitZeBinVISASections", i1 false}
!84 = !{!"FP64GenEmulationEnabled", i1 false}
!85 = !{!"FP64GenConvEmulationEnabled", i1 false}
!86 = !{!"allowDisableRematforCS", i1 false}
!87 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!88 = !{!"DisableCPSOmaskWA", i1 false}
!89 = !{!"DisableFastestGopt", i1 false}
!90 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!91 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!92 = !{!"DisableConstantCoalescing", i1 false}
!93 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!94 = !{!"WaEnableALTModeVisaWA", i1 false}
!95 = !{!"EnableLdStCombineforLoad", i1 false}
!96 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!97 = !{!"WaEnableAtomicWaveFusion", i1 false}
!98 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!99 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!100 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!101 = !{!"WaEnableAtomicWaveFusionPartial", i1 false}
!102 = !{!"WaEnableAtomicWaveFusionMoreDimensions", i1 false}
!103 = !{!"WaEnableAtomicWaveFusionLoop", i1 false}
!104 = !{!"WaEnableAtomicWaveFusionReturnValuePolicy", i32 0}
!105 = !{!"ForceCBThroughSampler3D", i1 false}
!106 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!107 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!108 = !{!"WaTypedAtomicBinToRawAtomicBin", i1 false}
!109 = !{!"WaRawAtomicBinToTypedAtomicBin", i1 false}
!110 = !{!"WaSampleLoadToTypedRead", i1 false}
!111 = !{!"EnableTypedBufferStoreToUntypedStore", i1 false}
!112 = !{!"WaZeroSLMBeforeUse", i1 false}
!113 = !{!"EnableEmitMoreMoviCases", i1 false}
!114 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!115 = !{!"EnableFastSampleD", i1 false}
!116 = !{!"ForceUniformBuffer", i1 false}
!117 = !{!"ForceUniformSurfaceSampler", i1 false}
!118 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!119 = !{!"NewSpillCostFunction", i1 false}
!120 = !{!"EnableVRT", i1 false}
!121 = !{!"ForceLargeGRFNum4RQ", i1 false}
!122 = !{!"Enable2xGRFRetry", i1 false}
!123 = !{!"Detect2xGRFCandidate", i1 false}
!124 = !{!"EnableURBWritesMerging", i1 true}
!125 = !{!"ForceCacheLineAlignedURBWriteMerging", i1 false}
!126 = !{!"DisableURBLayoutAlignmentToCacheLine", i1 false}
!127 = !{!"DisableEUFusion", i1 false}
!128 = !{!"DisableFDivToFMulInvOpt", i1 false}
!129 = !{!"initializePhiSampleSourceWA", i1 false}
!130 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!131 = !{!"DisableLoosenSimd32Occu", i1 false}
!132 = !{!"FastestS1Options", i32 0}
!133 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!134 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!135 = !{!"LscSamplerRouting", i32 0}
!136 = !{!"UseBarrierControlFlowOptimization", i1 false}
!137 = !{!"EnableDynamicRQManagement", i1 false}
!138 = !{!"WaDisablePayloadCoalescing", i1 false}
!139 = !{!"Quad8InputThreshold", i32 0}
!140 = !{!"UseResourceLoopUnrollNested", i1 false}
!141 = !{!"DisableLoopUnroll", i1 false}
!142 = !{!"ForcePushConstantMode", i32 0}
!143 = !{!"UseInstructionHoistingOptimization", i1 false}
!144 = !{!"DisableResourceLoopDestLifeTimeStart", i1 false}
!145 = !{!"ForceVRTGRFCeiling", i32 0}
!146 = !{!"DisableSamplerBackingByLSC", i32 0}
!147 = !{!"UseLinearScanRA", i1 false}
!148 = !{!"DisableConvertingAtomicIAddToIncDec", i1 false}
!149 = !{!"EnableInlinedCrossThreadData", i1 false}
!150 = !{!"ZeroInitRegistersBeforeExecution", i1 false}
!151 = !{!"FuncMD"}
!152 = !{!"pushInfo", !153, !154, !155, !159, !160, !161, !162, !163, !164, !165, !166, !179, !180, !181, !182}
!153 = !{!"pushableAddresses"}
!154 = !{!"bindlessPushInfo"}
!155 = !{!"dynamicBufferInfo", !156, !157, !158}
!156 = !{!"firstIndex", i32 0}
!157 = !{!"numOffsets", i32 0}
!158 = !{!"forceDisabled", i1 false}
!159 = !{!"MaxNumberOfPushedBuffers", i32 0}
!160 = !{!"inlineConstantBufferSlot", i32 -1}
!161 = !{!"inlineConstantBufferOffset", i32 -1}
!162 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!163 = !{!"constants"}
!164 = !{!"inputs"}
!165 = !{!"constantReg"}
!166 = !{!"simplePushInfoArr", !167, !176, !177, !178}
!167 = !{!"simplePushInfoArrVec[0]", !168, !169, !170, !171, !172, !173, !174, !175}
!168 = !{!"cbIdx", i32 0}
!169 = !{!"pushableAddressGrfOffset", i32 -1}
!170 = !{!"pushableOffsetGrfOffset", i32 -1}
!171 = !{!"offset", i32 0}
!172 = !{!"size", i32 0}
!173 = !{!"isStateless", i1 false}
!174 = !{!"isBindless", i1 false}
!175 = !{!"simplePushLoads"}
!176 = !{!"simplePushInfoArrVec[1]", !168, !169, !170, !171, !172, !173, !174, !175}
!177 = !{!"simplePushInfoArrVec[2]", !168, !169, !170, !171, !172, !173, !174, !175}
!178 = !{!"simplePushInfoArrVec[3]", !168, !169, !170, !171, !172, !173, !174, !175}
!179 = !{!"simplePushBufferUsed", i32 0}
!180 = !{!"pushAnalysisWIInfos"}
!181 = !{!"inlineRTGlobalPtrOffset", i32 0}
!182 = !{!"rtSyncSurfPtrOffset", i32 0}
!183 = !{!"pISAInfo", !184, !185, !189, !190, !198, !202, !204}
!184 = !{!"shaderType", !"UNKNOWN"}
!185 = !{!"geometryInfo", !186, !187, !188}
!186 = !{!"needsVertexHandles", i1 false}
!187 = !{!"needsPrimitiveIDEnable", i1 false}
!188 = !{!"VertexCount", i32 0}
!189 = !{!"hullInfo", !186, !187}
!190 = !{!"pixelInfo", !191, !192, !193, !194, !195, !196, !197}
!191 = !{!"perPolyStartGrf", i32 0}
!192 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!193 = !{!"hasNonPerspBaryPlanes", i1 false}
!194 = !{!"maxPerPrimConstDataId", i32 -1}
!195 = !{!"maxSetupId", i32 -1}
!196 = !{!"hasVMask", i1 false}
!197 = !{!"PixelGRFBitmask", i32 0}
!198 = !{!"domainInfo", !199, !200, !201}
!199 = !{!"DomainPointUArgIdx", i32 -1}
!200 = !{!"DomainPointVArgIdx", i32 -1}
!201 = !{!"DomainPointWArgIdx", i32 -1}
!202 = !{!"computeInfo", !203}
!203 = !{!"EnableHWGenerateLID", i1 true}
!204 = !{!"URBOutputLength", i32 0}
!205 = !{!"WaEnableICBPromotion", i1 false}
!206 = !{!"vsInfo", !207, !208, !209}
!207 = !{!"DrawIndirectBufferIndex", i32 -1}
!208 = !{!"vertexReordering", i32 -1}
!209 = !{!"MaxNumOfOutputs", i32 0}
!210 = !{!"hsInfo", !211, !212}
!211 = !{!"numPatchAttributesPatchBaseName", !""}
!212 = !{!"numVertexAttributesPatchBaseName", !""}
!213 = !{!"dsInfo", !209}
!214 = !{!"gsInfo", !209}
!215 = !{!"psInfo", !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252, !253}
!216 = !{!"BlendStateDisabledMask", i8 0}
!217 = !{!"SkipSrc0Alpha", i1 false}
!218 = !{!"DualSourceBlendingDisabled", i1 false}
!219 = !{!"ForceEnableSimd32", i1 false}
!220 = !{!"DisableSimd32WithDiscard", i1 false}
!221 = !{!"outputDepth", i1 false}
!222 = !{!"outputStencil", i1 false}
!223 = !{!"outputMask", i1 false}
!224 = !{!"blendToFillEnabled", i1 false}
!225 = !{!"forceEarlyZ", i1 false}
!226 = !{!"hasVersionedLoop", i1 false}
!227 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!228 = !{!"requestCPSizeRelevant", i1 false}
!229 = !{!"requestCPSize", i1 false}
!230 = !{!"texelMaskFastClearMode", !"Disabled"}
!231 = !{!"NumSamples", i8 0}
!232 = !{!"blendOptimizationMode"}
!233 = !{!"colorOutputMask"}
!234 = !{!"ProvokingVertexModeNosIndex", i32 0}
!235 = !{!"ProvokingVertexModeNosPatch", !""}
!236 = !{!"ProvokingVertexModeLast", !"Negative"}
!237 = !{!"VertexAttributesBypass", i1 false}
!238 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!239 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!240 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!241 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!242 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!243 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!244 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!245 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!246 = !{!"generatePatchesForRTWriteSends", i1 false}
!247 = !{!"generatePatchesForRT_BTIndex", i1 false}
!248 = !{!"forceVMask", i1 false}
!249 = !{!"isNumPerPrimAttributesSet", i1 false}
!250 = !{!"numPerPrimAttributes", i32 0}
!251 = !{!"WaDisableVRS", i1 false}
!252 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!253 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!254 = !{!"csInfo", !255, !256, !257, !258, !53, !29, !30, !259, !31, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !64, !271, !272, !273, !274, !275, !276, !277, !278}
!255 = !{!"maxWorkGroupSize", i32 0}
!256 = !{!"waveSize", i32 0}
!257 = !{!"ComputeShaderSecondCompile"}
!258 = !{!"forcedSIMDSize", i8 0}
!259 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!260 = !{!"forceSpillCompression", i1 false}
!261 = !{!"allowLowerSimd", i1 false}
!262 = !{!"disableSimd32Slicing", i1 false}
!263 = !{!"disableSplitOnSpill", i1 false}
!264 = !{!"enableNewSpillCostFunction", i1 false}
!265 = !{!"forceVISAPreSched", i1 false}
!266 = !{!"disableLocalIdOrderOptimizations", i1 false}
!267 = !{!"disableDispatchAlongY", i1 false}
!268 = !{!"neededThreadIdLayout", i1* null}
!269 = !{!"forceTileYWalk", i1 false}
!270 = !{!"atomicBranch", i32 0}
!271 = !{!"disableEarlyOut", i1 false}
!272 = !{!"walkOrderEnabled", i1 false}
!273 = !{!"walkOrderOverride", i32 0}
!274 = !{!"ResForHfPacking"}
!275 = !{!"constantFoldSimdSize", i1 false}
!276 = !{!"isNodeShader", i1 false}
!277 = !{!"threadGroupMergeSize", i32 0}
!278 = !{!"threadGroupMergeOverY", i1 false}
!279 = !{!"msInfo", !280, !281, !282, !283, !284, !285, !286, !287, !288, !289, !290, !236, !234, !291, !292, !276}
!280 = !{!"PrimitiveTopology", i32 3}
!281 = !{!"MaxNumOfPrimitives", i32 0}
!282 = !{!"MaxNumOfVertices", i32 0}
!283 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!284 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!285 = !{!"WorkGroupSize", i32 0}
!286 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!287 = !{!"IndexFormat", i32 6}
!288 = !{!"SubgroupSize", i32 0}
!289 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!290 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!291 = !{!"Is16BMUEModeAllowed", i1 false}
!292 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!293 = !{!"taskInfo", !209, !285, !286, !288}
!294 = !{!"NBarrierCnt", i32 0}
!295 = !{!"rtInfo", !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307, !308, !309, !310, !311, !312}
!296 = !{!"RayQueryAllocSizeInBytes", i32 0}
!297 = !{!"NumContinuations", i32 0}
!298 = !{!"RTAsyncStackAddrspace", i32 -1}
!299 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!300 = !{!"SWHotZoneAddrspace", i32 -1}
!301 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!302 = !{!"SWStackAddrspace", i32 -1}
!303 = !{!"SWStackSurfaceStateOffset", i1* null}
!304 = !{!"RTSyncStackAddrspace", i32 -1}
!305 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!306 = !{!"doSyncDispatchRays", i1 false}
!307 = !{!"MemStyle", !"Xe"}
!308 = !{!"GlobalDataStyle", !"Xe"}
!309 = !{!"NeedsBTD", i1 true}
!310 = !{!"SERHitObjectFullType", i1* null}
!311 = !{!"uberTileDimensions", i1* null}
!312 = !{!"numSyncRTStacks", i32 0}
!313 = !{!"CurUniqueIndirectIdx", i32 0}
!314 = !{!"inlineDynTextures"}
!315 = !{!"inlineResInfoData"}
!316 = !{!"immConstant", !317, !318, !319}
!317 = !{!"data"}
!318 = !{!"sizes"}
!319 = !{!"zeroIdxs"}
!320 = !{!"stringConstants"}
!321 = !{!"inlineBuffers", !322, !326, !327}
!322 = !{!"inlineBuffersVec[0]", !323, !324, !325}
!323 = !{!"alignment", i32 0}
!324 = !{!"allocSize", i64 0}
!325 = !{!"Buffer"}
!326 = !{!"inlineBuffersVec[1]", !323, !324, !325}
!327 = !{!"inlineBuffersVec[2]", !323, !324, !325}
!328 = !{!"GlobalPointerProgramBinaryInfos"}
!329 = !{!"ConstantPointerProgramBinaryInfos"}
!330 = !{!"GlobalBufferAddressRelocInfo"}
!331 = !{!"ConstantBufferAddressRelocInfo"}
!332 = !{!"forceLscCacheList"}
!333 = !{!"SrvMap"}
!334 = !{!"RootConstantBufferOffsetInBytes"}
!335 = !{!"RasterizerOrderedByteAddressBuffer"}
!336 = !{!"RasterizerOrderedViews"}
!337 = !{!"MinNOSPushConstantSize", i32 0}
!338 = !{!"inlineProgramScopeOffsets"}
!339 = !{!"shaderData", !340}
!340 = !{!"numReplicas", i32 0}
!341 = !{!"URBInfo", !342, !343, !344}
!342 = !{!"has64BVertexHeaderInput", i1 false}
!343 = !{!"has64BVertexHeaderOutput", i1 false}
!344 = !{!"hasVertexHeader", i1 true}
!345 = !{!"m_ForcePullModel", i1 false}
!346 = !{!"UseBindlessImage", i1 false}
!347 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!348 = !{!"enableRangeReduce", i1 false}
!349 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!350 = !{!"enableFRemToSRemOpt", i1 false}
!351 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!352 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!353 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!354 = !{!"allowMatchMadOptimizationforVS", i1 false}
!355 = !{!"disableMatchMadOptimizationForCS", i1 false}
!356 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!357 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!358 = !{!"statefulResourcesNotAliased", i1 false}
!359 = !{!"disableMixMode", i1 false}
!360 = !{!"genericAccessesResolved", i1 false}
!361 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!362 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!363 = !{!"disableSeparateScratchWA", i1 false}
!364 = !{!"enableRemoveUnusedTGMFence", i1 false}
!365 = !{!"privateMemoryPerWI", i32 0}
!366 = !{!"PrivateMemoryPerFG"}
!367 = !{!"m_OptsToDisable"}
!368 = !{!"capabilities", !369}
!369 = !{!"globalVariableDecorationsINTEL", i1 false}
!370 = !{!"extensions", !371}
!371 = !{!"spvINTELBindlessImages", i1 false}
!372 = !{!"m_ShaderResourceViewMcsMask", !373, !374}
!373 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!374 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!375 = !{!"computedDepthMode", i32 0}
!376 = !{!"isHDCFastClearShader", i1 false}
!377 = !{!"argRegisterReservations", !378}
!378 = !{!"argRegisterReservationsVec[0]", i32 0}
!379 = !{!"SIMD16_SpillThreshold", i8 0}
!380 = !{!"SIMD32_SpillThreshold", i8 0}
!381 = !{!"m_CacheControlOption", !382, !383, !384, !385}
!382 = !{!"LscLoadCacheControlOverride", i8 0}
!383 = !{!"LscStoreCacheControlOverride", i8 0}
!384 = !{!"TgmLoadCacheControlOverride", i8 0}
!385 = !{!"TgmStoreCacheControlOverride", i8 0}
!386 = !{!"ModuleUsesBindless", i1 false}
!387 = !{!"predicationMap"}
!388 = !{!"lifeTimeStartMap"}
!389 = !{!"HitGroups"}
!390 = !{!0, !391}
!391 = !{!1, !392}
!392 = !{i32 45, i32 28}
!393 = !{!394}
!394 = !{i32 44, i32 4}
!395 = !{!396, !397}
!396 = !{i32 4469}
!397 = !{i32 4470}
!398 = !{!396}
!399 = !{!400, !400}
!400 = !{!401, !402}
!401 = !{i32 38, i32 4}
!402 = !{i32 38, i32 5}
!403 = !{!397}
!404 = !{!405, !407}
!405 = !{!394, !406}
!406 = !{i32 45, i32 8}
!407 = !{!394, !408}
!408 = !{i32 45, i32 44}
!409 = !{!410}
!410 = distinct !{!410, !411}
!411 = distinct !{!411}
!412 = !{!413}
!413 = !{i32 40, i32 196620}
!414 = !{!415, !6, !417, !417, !6, !417, !417, !6, !6, !6, !6, !6, !6}
!415 = !{!416, !394}
!416 = !{i32 38, i32 2}
!417 = !{!416, !1}
!418 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!419 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!420 = !{!"struct cutlass::gemm::GemmCoord", !"float", !"class.cutlass::__generated_TensorRef", !"class.cutlass::__generated_TensorRef", !"float", !"class.cutlass::__generated_TensorRef", !"class.cutlass::__generated_TensorRef", !"float", !"int", !"long", !"long", !"long", !"long"}
!421 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!422 = distinct !{!422, !423}
!423 = !{!"llvm.loop.unroll.enable"}
!424 = distinct !{!424, !423}
!425 = distinct !{!425, !423}
!426 = !{!427}
!427 = distinct !{!427, !428}
!428 = distinct !{!428}
!429 = !{!430}
!430 = distinct !{!430, !431}
!431 = distinct !{!431}
!432 = distinct !{!432, !423}
!433 = distinct !{!433, !423}
!434 = !{!435}
!435 = distinct !{!435, !436}
!436 = distinct !{!436}
!437 = distinct !{!437, !423}
!438 = distinct !{!438, !423}
!439 = distinct !{!439, !423}
!440 = distinct !{!440, !423}
!441 = !{!442}
!442 = distinct !{!442, !443}
!443 = distinct !{!443}
!444 = !{!445}
!445 = distinct !{!445, !446}
!446 = distinct !{!446}
!447 = distinct !{!447, !423}
!448 = distinct !{!448, !423}
!449 = !{!450}
!450 = distinct !{!450, !451}
!451 = distinct !{!451}
!452 = distinct !{!452, !423}
!453 = !{!454}
!454 = !{!394, !455}
!455 = !{i32 45, i32 4}
!456 = !{!457}
!457 = !{!458, !459}
!458 = !{i32 44, i32 2}
!459 = !{i32 45, i32 2}

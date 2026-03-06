; ------------------------------------------------
; OCL_asm23954d4a795eca46_1_afterUnification.ll
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

@0 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"
@1 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
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
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !605
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
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !607
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !607
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !609
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !607
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !607
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
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !611
  br label %81, !llvm.loop !613

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !611
  br label %86, !llvm.loop !615

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
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !611
  br label %95

107:                                              ; preds = %178, %101
  %108 = phi i32 [ %179, %178 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !611
  br label %98, !llvm.loop !616

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
  store i32 %113, i32* %119, align 4, !noalias !617
  store i32 %96, i32* %63, align 4, !noalias !617
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !617
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !617
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !611
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !607
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
  store i32 %96, i32* %143, align 4, !noalias !620
  store i32 %102, i32* %66, align 4, !noalias !620
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !620
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !620
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !611
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = ashr i64 %155, 32
  %158 = mul nsw i64 %157, %const_reg_qword5, !spirv.Decorations !607
  %159 = shl i64 %155, 32
  %160 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %158
  %161 = ashr exact i64 %159, 31
  %162 = bitcast %"struct.cutlass::bfloat16_t" addrspace(4)* %160 to i8 addrspace(4)*
  %163 = getelementptr i8, i8 addrspace(4)* %162, i64 %161
  %164 = bitcast i8 addrspace(4)* %163 to i16 addrspace(4)*
  %165 = addrspacecast i16 addrspace(4)* %164 to i16 addrspace(1)*
  %166 = load i16, i16 addrspace(1)* %165, align 2
  %167 = zext i16 %140 to i32
  %168 = shl nuw i32 %167, 16, !spirv.Decorations !623
  %169 = bitcast i32 %168 to float
  %170 = zext i16 %166 to i32
  %171 = shl nuw i32 %170, 16, !spirv.Decorations !623
  %172 = bitcast i32 %171 to float
  %173 = zext i32 %108 to i64
  %174 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %173, i64 %104
  %175 = fmul reassoc nsz arcp contract float %169, %172, !spirv.Decorations !609
  %176 = load float, float* %174, align 4
  %177 = fadd reassoc nsz arcp contract float %175, %176, !spirv.Decorations !609
  store float %177, float* %174, align 4
  br label %178

178:                                              ; preds = %153, %112
  %179 = add nuw nsw i32 %108, 1, !spirv.Decorations !611
  br label %107, !llvm.loop !624

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
  %190 = mul nsw i64 %188, %9, !spirv.Decorations !607
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %190
  %192 = mul nsw i64 %188, %10, !spirv.Decorations !607
  %193 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %192
  br i1 %57, label %244, label %249

194:                                              ; preds = %242, %183
  %195 = phi i32 [ %243, %242 ], [ 0, %183 ]
  %196 = icmp ult i32 %195, 4
  br i1 %196, label %199, label %197

197:                                              ; preds = %194
  %198 = add nuw nsw i32 %181, 1, !spirv.Decorations !611
  br label %180, !llvm.loop !625

199:                                              ; preds = %194
  %200 = or i32 %44, %195
  %201 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %201)
  %202 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %202)
  %203 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %200, i32* %203, align 4, !noalias !626
  store i32 %184, i32* %69, align 4, !noalias !626
  br label %204

204:                                              ; preds = %207, %199
  %205 = phi i32 [ 0, %199 ], [ %212, %207 ]
  %206 = icmp ult i32 %205, 2
  br i1 %206, label %207, label %213

207:                                              ; preds = %204
  %208 = zext i32 %205 to i64
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %208
  %210 = load i32, i32* %209, align 4, !noalias !626
  %211 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %208
  store i32 %210, i32* %211, align 4, !alias.scope !626
  %212 = add nuw nsw i32 %205, 1, !spirv.Decorations !611
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
  %224 = mul nsw i64 %222, %const_reg_qword9, !spirv.Decorations !607
  %225 = add nsw i64 %224, %223, !spirv.Decorations !607
  %226 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %220, i64 %186
  %227 = load float, float* %226, align 4
  %228 = fmul reassoc nsz arcp contract float %227, %1, !spirv.Decorations !609
  br i1 %57, label %229, label %239

229:                                              ; preds = %219
  %230 = mul nsw i64 %222, %const_reg_qword7, !spirv.Decorations !607
  %231 = getelementptr float, float addrspace(4)* %75, i64 %230
  %232 = getelementptr float, float addrspace(4)* %231, i64 %223
  %233 = addrspacecast float addrspace(4)* %232 to float addrspace(1)*
  %234 = load float, float addrspace(1)* %233, align 4
  %235 = fmul reassoc nsz arcp contract float %234, %4, !spirv.Decorations !609
  %236 = fadd reassoc nsz arcp contract float %228, %235, !spirv.Decorations !609
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
  %243 = add nuw nsw i32 %195, 1, !spirv.Decorations !611
  br label %194, !llvm.loop !629

244:                                              ; preds = %187
  %245 = zext i32 %16 to i64
  %246 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %246)
  %247 = mul nsw i64 %245, %11, !spirv.Decorations !607
  %248 = getelementptr inbounds float, float addrspace(4)* %75, i64 %247
  br label %249

249:                                              ; preds = %244, %187
  %250 = phi float addrspace(4)* [ %248, %244 ], [ %75, %187 ]
  %251 = zext i32 %16 to i64
  %252 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %252)
  %253 = mul nsw i64 %251, %12, !spirv.Decorations !607
  %254 = getelementptr inbounds float, float addrspace(4)* %76, i64 %253
  %255 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %255)
  %256 = add i32 %77, %16
  br label %72

257:                                              ; preds = %72
  ret void
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
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
  %28 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %29 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %30 = alloca [2 x i32], align 4, !spirv.Decorations !605
  %31 = alloca %structtype.0, align 8
  %32 = alloca %structtype.0, align 8
  %33 = alloca %structtype.0, align 8
  %34 = alloca [4 x [16 x float]], align 4, !spirv.Decorations !605
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
  %53 = mul nsw i64 %51, %9, !spirv.Decorations !607
  %54 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %38, i64 %53
  %55 = mul nsw i64 %51, %10, !spirv.Decorations !607
  %56 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %37, i64 %55
  %57 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !609
  %58 = mul nsw i64 %51, %11, !spirv.Decorations !607
  %59 = select i1 %57, i64 %58, i64 0
  %60 = getelementptr inbounds float, float addrspace(4)* %36, i64 %59
  %61 = mul nsw i64 %51, %12, !spirv.Decorations !607
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
  %90 = add nuw nsw i32 %82, 1, !spirv.Decorations !611
  br label %81, !llvm.loop !630

91:                                               ; preds = %86
  %92 = zext i32 %87 to i64
  %93 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %92, i64 %85
  store float %7, float* %93, align 4
  %94 = add nuw nsw i32 %87, 1, !spirv.Decorations !611
  br label %86, !llvm.loop !631

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
  %106 = add nuw nsw i32 %96, 1, !spirv.Decorations !611
  br label %95

107:                                              ; preds = %176, %101
  %108 = phi i32 [ %177, %176 ], [ 0, %101 ]
  %109 = icmp ult i32 %108, 4
  br i1 %109, label %112, label %110

110:                                              ; preds = %107
  %111 = add nuw nsw i32 %99, 1, !spirv.Decorations !611
  br label %98, !llvm.loop !632

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
  store i32 %113, i32* %119, align 4, !noalias !633
  store i32 %96, i32* %63, align 4, !noalias !633
  br label %120

120:                                              ; preds = %123, %116
  %121 = phi i32 [ 0, %116 ], [ %128, %123 ]
  %122 = icmp ult i32 %121, 2
  br i1 %122, label %123, label %129

123:                                              ; preds = %120
  %124 = zext i32 %121 to i64
  %125 = getelementptr inbounds [2 x i32], [2 x i32]* %29, i64 0, i64 %124
  %126 = load i32, i32* %125, align 4, !noalias !633
  %127 = getelementptr inbounds [2 x i32], [2 x i32]* %64, i64 0, i64 %124
  store i32 %126, i32* %127, align 4, !alias.scope !633
  %128 = add nuw nsw i32 %121, 1, !spirv.Decorations !611
  br label %120

129:                                              ; preds = %120
  %130 = bitcast [2 x i32]* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)
  %131 = load i64, i64* %65, align 8
  %132 = bitcast %structtype.0* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %132)
  %133 = shl i64 %131, 32
  %134 = ashr exact i64 %133, 32
  %135 = mul nsw i64 %134, %const_reg_qword3, !spirv.Decorations !607
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
  store i32 %96, i32* %143, align 4, !noalias !636
  store i32 %102, i32* %66, align 4, !noalias !636
  br label %144

144:                                              ; preds = %147, %129
  %145 = phi i32 [ 0, %129 ], [ %152, %147 ]
  %146 = icmp ult i32 %145, 2
  br i1 %146, label %147, label %153

147:                                              ; preds = %144
  %148 = zext i32 %145 to i64
  %149 = getelementptr inbounds [2 x i32], [2 x i32]* %30, i64 0, i64 %148
  %150 = load i32, i32* %149, align 4, !noalias !636
  %151 = getelementptr inbounds [2 x i32], [2 x i32]* %67, i64 0, i64 %148
  store i32 %150, i32* %151, align 4, !alias.scope !636
  %152 = add nuw nsw i32 %145, 1, !spirv.Decorations !611
  br label %144

153:                                              ; preds = %144
  %154 = bitcast [2 x i32]* %30 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %154)
  %155 = load i64, i64* %68, align 8
  %156 = bitcast %structtype.0* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %156)
  %157 = shl i64 %155, 32
  %158 = ashr exact i64 %157, 32
  %159 = mul nsw i64 %158, %const_reg_qword5, !spirv.Decorations !607
  %160 = ashr i64 %155, 32
  %161 = getelementptr %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %159, i32 0
  %162 = getelementptr i16, i16 addrspace(4)* %161, i64 %160
  %163 = addrspacecast i16 addrspace(4)* %162 to i16 addrspace(1)*
  %164 = load i16, i16 addrspace(1)* %163, align 2
  %165 = zext i16 %140 to i32
  %166 = shl nuw i32 %165, 16, !spirv.Decorations !623
  %167 = bitcast i32 %166 to float
  %168 = zext i16 %164 to i32
  %169 = shl nuw i32 %168, 16, !spirv.Decorations !623
  %170 = bitcast i32 %169 to float
  %171 = zext i32 %108 to i64
  %172 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %171, i64 %104
  %173 = fmul reassoc nsz arcp contract float %167, %170, !spirv.Decorations !609
  %174 = load float, float* %172, align 4
  %175 = fadd reassoc nsz arcp contract float %173, %174, !spirv.Decorations !609
  store float %175, float* %172, align 4
  br label %176

176:                                              ; preds = %153, %112
  %177 = add nuw nsw i32 %108, 1, !spirv.Decorations !611
  br label %107, !llvm.loop !639

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
  %188 = mul nsw i64 %186, %9, !spirv.Decorations !607
  %189 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %73, i64 %188
  %190 = mul nsw i64 %186, %10, !spirv.Decorations !607
  %191 = getelementptr inbounds %"struct.cutlass::bfloat16_t", %"struct.cutlass::bfloat16_t" addrspace(4)* %74, i64 %190
  br i1 %57, label %242, label %247

192:                                              ; preds = %240, %181
  %193 = phi i32 [ %241, %240 ], [ 0, %181 ]
  %194 = icmp ult i32 %193, 4
  br i1 %194, label %197, label %195

195:                                              ; preds = %192
  %196 = add nuw nsw i32 %179, 1, !spirv.Decorations !611
  br label %178, !llvm.loop !640

197:                                              ; preds = %192
  %198 = or i32 %44, %193
  %199 = bitcast %structtype.0* %31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %199)
  %200 = bitcast [2 x i32]* %28 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %200)
  %201 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 0
  store i32 %198, i32* %201, align 4, !noalias !641
  store i32 %182, i32* %69, align 4, !noalias !641
  br label %202

202:                                              ; preds = %205, %197
  %203 = phi i32 [ 0, %197 ], [ %210, %205 ]
  %204 = icmp ult i32 %203, 2
  br i1 %204, label %205, label %211

205:                                              ; preds = %202
  %206 = zext i32 %203 to i64
  %207 = getelementptr inbounds [2 x i32], [2 x i32]* %28, i64 0, i64 %206
  %208 = load i32, i32* %207, align 4, !noalias !641
  %209 = getelementptr inbounds [2 x i32], [2 x i32]* %70, i64 0, i64 %206
  store i32 %208, i32* %209, align 4, !alias.scope !641
  %210 = add nuw nsw i32 %203, 1, !spirv.Decorations !611
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
  %222 = mul nsw i64 %220, %const_reg_qword9, !spirv.Decorations !607
  %223 = add nsw i64 %222, %221, !spirv.Decorations !607
  %224 = getelementptr inbounds [4 x [16 x float]], [4 x [16 x float]]* %34, i64 0, i64 %218, i64 %184
  %225 = load float, float* %224, align 4
  %226 = fmul reassoc nsz arcp contract float %225, %1, !spirv.Decorations !609
  br i1 %57, label %227, label %237

227:                                              ; preds = %217
  %228 = mul nsw i64 %220, %const_reg_qword7, !spirv.Decorations !607
  %229 = getelementptr float, float addrspace(4)* %75, i64 %228
  %230 = getelementptr float, float addrspace(4)* %229, i64 %221
  %231 = addrspacecast float addrspace(4)* %230 to float addrspace(1)*
  %232 = load float, float addrspace(1)* %231, align 4
  %233 = fmul reassoc nsz arcp contract float %232, %4, !spirv.Decorations !609
  %234 = fadd reassoc nsz arcp contract float %226, %233, !spirv.Decorations !609
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
  %241 = add nuw nsw i32 %193, 1, !spirv.Decorations !611
  br label %192, !llvm.loop !644

242:                                              ; preds = %185
  %243 = zext i32 %16 to i64
  %244 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %244)
  %245 = mul nsw i64 %243, %11, !spirv.Decorations !607
  %246 = getelementptr inbounds float, float addrspace(4)* %75, i64 %245
  br label %247

247:                                              ; preds = %242, %185
  %248 = phi float addrspace(4)* [ %246, %242 ], [ %75, %185 ]
  %249 = zext i32 %16 to i64
  %250 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %250)
  %251 = mul nsw i64 %249, %12, !spirv.Decorations !607
  %252 = getelementptr inbounds float, float addrspace(4)* %76, i64 %251
  %253 = bitcast [4 x [16 x float]]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 256, i8* nonnull %253)
  %254 = add i32 %77, %16
  br label %72

255:                                              ; preds = %72
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
!opencl.ocl.version = !{!602, !602, !602, !602, !602, !602, !602, !602, !602, !602}
!opencl.spir.version = !{!602, !602, !602, !602, !602, !602, !602, !602, !602, !602}
!llvm.ident = !{!603, !603, !603, !603, !603, !603, !603, !603, !603, !603}
!llvm.module.flags = !{!604}

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
!35 = !{!"ModuleMD", !36, !37, !175, !366, !397, !419, !420, !424, !427, !428, !429, !468, !493, !507, !508, !509, !526, !527, !528, !529, !533, !534, !541, !542, !543, !544, !545, !546, !547, !548, !549, !550, !551, !552, !554, !558, !559, !560, !561, !562, !563, !564, !565, !566, !567, !568, !569, !570, !571, !572, !573, !574, !575, !576, !577, !262, !578, !579, !580, !582, !584, !587, !588, !589, !591, !592, !593, !598, !599, !600, !601}
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
!175 = !{!"FuncMD", !176, !177, !364, !365}
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
!363 = !{!"m_OptsToDisablePerFunc"}
!364 = !{!"FuncMDMap[1]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!365 = !{!"FuncMDValue[1]", !178, !179, !183, !184, !185, !186, !187, !188, !189, !213, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !288, !302, !316, !330, !344, !358, !363}
!366 = !{!"pushInfo", !367, !368, !369, !373, !374, !375, !376, !377, !378, !379, !380, !393, !394, !395, !396}
!367 = !{!"pushableAddresses"}
!368 = !{!"bindlessPushInfo"}
!369 = !{!"dynamicBufferInfo", !370, !371, !372}
!370 = !{!"firstIndex", i32 0}
!371 = !{!"numOffsets", i32 0}
!372 = !{!"forceDisabled", i1 false}
!373 = !{!"MaxNumberOfPushedBuffers", i32 0}
!374 = !{!"inlineConstantBufferSlot", i32 -1}
!375 = !{!"inlineConstantBufferOffset", i32 -1}
!376 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!377 = !{!"constants"}
!378 = !{!"inputs"}
!379 = !{!"constantReg"}
!380 = !{!"simplePushInfoArr", !381, !390, !391, !392}
!381 = !{!"simplePushInfoArrVec[0]", !382, !383, !384, !385, !386, !387, !388, !389}
!382 = !{!"cbIdx", i32 0}
!383 = !{!"pushableAddressGrfOffset", i32 -1}
!384 = !{!"pushableOffsetGrfOffset", i32 -1}
!385 = !{!"offset", i32 0}
!386 = !{!"size", i32 0}
!387 = !{!"isStateless", i1 false}
!388 = !{!"isBindless", i1 false}
!389 = !{!"simplePushLoads"}
!390 = !{!"simplePushInfoArrVec[1]", !382, !383, !384, !385, !386, !387, !388, !389}
!391 = !{!"simplePushInfoArrVec[2]", !382, !383, !384, !385, !386, !387, !388, !389}
!392 = !{!"simplePushInfoArrVec[3]", !382, !383, !384, !385, !386, !387, !388, !389}
!393 = !{!"simplePushBufferUsed", i32 0}
!394 = !{!"pushAnalysisWIInfos"}
!395 = !{!"inlineRTGlobalPtrOffset", i32 0}
!396 = !{!"rtSyncSurfPtrOffset", i32 0}
!397 = !{!"pISAInfo", !398, !399, !403, !404, !412, !416, !418}
!398 = !{!"shaderType", !"UNKNOWN"}
!399 = !{!"geometryInfo", !400, !401, !402}
!400 = !{!"needsVertexHandles", i1 false}
!401 = !{!"needsPrimitiveIDEnable", i1 false}
!402 = !{!"VertexCount", i32 0}
!403 = !{!"hullInfo", !400, !401}
!404 = !{!"pixelInfo", !405, !406, !407, !408, !409, !410, !411}
!405 = !{!"perPolyStartGrf", i32 0}
!406 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!407 = !{!"hasNonPerspBaryPlanes", i1 false}
!408 = !{!"maxPerPrimConstDataId", i32 -1}
!409 = !{!"maxSetupId", i32 -1}
!410 = !{!"hasVMask", i1 false}
!411 = !{!"PixelGRFBitmask", i32 0}
!412 = !{!"domainInfo", !413, !414, !415}
!413 = !{!"DomainPointUArgIdx", i32 -1}
!414 = !{!"DomainPointVArgIdx", i32 -1}
!415 = !{!"DomainPointWArgIdx", i32 -1}
!416 = !{!"computeInfo", !417}
!417 = !{!"EnableHWGenerateLID", i1 true}
!418 = !{!"URBOutputLength", i32 0}
!419 = !{!"WaEnableICBPromotion", i1 false}
!420 = !{!"vsInfo", !421, !422, !423}
!421 = !{!"DrawIndirectBufferIndex", i32 -1}
!422 = !{!"vertexReordering", i32 -1}
!423 = !{!"MaxNumOfOutputs", i32 0}
!424 = !{!"hsInfo", !425, !426}
!425 = !{!"numPatchAttributesPatchBaseName", !""}
!426 = !{!"numVertexAttributesPatchBaseName", !""}
!427 = !{!"dsInfo", !423}
!428 = !{!"gsInfo", !423}
!429 = !{!"psInfo", !430, !431, !432, !433, !434, !435, !436, !437, !438, !439, !440, !441, !442, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !453, !454, !455, !456, !457, !458, !459, !460, !461, !462, !463, !464, !465, !466, !467}
!430 = !{!"BlendStateDisabledMask", i8 0}
!431 = !{!"SkipSrc0Alpha", i1 false}
!432 = !{!"DualSourceBlendingDisabled", i1 false}
!433 = !{!"ForceEnableSimd32", i1 false}
!434 = !{!"DisableSimd32WithDiscard", i1 false}
!435 = !{!"outputDepth", i1 false}
!436 = !{!"outputStencil", i1 false}
!437 = !{!"outputMask", i1 false}
!438 = !{!"blendToFillEnabled", i1 false}
!439 = !{!"forceEarlyZ", i1 false}
!440 = !{!"hasVersionedLoop", i1 false}
!441 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!442 = !{!"requestCPSizeRelevant", i1 false}
!443 = !{!"requestCPSize", i1 false}
!444 = !{!"texelMaskFastClearMode", !"Disabled"}
!445 = !{!"NumSamples", i8 0}
!446 = !{!"blendOptimizationMode"}
!447 = !{!"colorOutputMask"}
!448 = !{!"ProvokingVertexModeNosIndex", i32 0}
!449 = !{!"ProvokingVertexModeNosPatch", !""}
!450 = !{!"ProvokingVertexModeLast", !"Negative"}
!451 = !{!"VertexAttributesBypass", i1 false}
!452 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!453 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!454 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!455 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!456 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!457 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!458 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!459 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!460 = !{!"generatePatchesForRTWriteSends", i1 false}
!461 = !{!"generatePatchesForRT_BTIndex", i1 false}
!462 = !{!"forceVMask", i1 false}
!463 = !{!"isNumPerPrimAttributesSet", i1 false}
!464 = !{!"numPerPrimAttributes", i32 0}
!465 = !{!"WaDisableVRS", i1 false}
!466 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!467 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!468 = !{!"csInfo", !469, !470, !471, !472, !77, !53, !54, !473, !55, !474, !475, !476, !477, !478, !479, !480, !481, !482, !483, !484, !88, !485, !486, !487, !488, !489, !490, !491, !492}
!469 = !{!"maxWorkGroupSize", i32 0}
!470 = !{!"waveSize", i32 0}
!471 = !{!"ComputeShaderSecondCompile"}
!472 = !{!"forcedSIMDSize", i8 0}
!473 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!474 = !{!"forceSpillCompression", i1 false}
!475 = !{!"allowLowerSimd", i1 false}
!476 = !{!"disableSimd32Slicing", i1 false}
!477 = !{!"disableSplitOnSpill", i1 false}
!478 = !{!"enableNewSpillCostFunction", i1 false}
!479 = !{!"forceVISAPreSched", i1 false}
!480 = !{!"disableLocalIdOrderOptimizations", i1 false}
!481 = !{!"disableDispatchAlongY", i1 false}
!482 = !{!"neededThreadIdLayout", i1* null}
!483 = !{!"forceTileYWalk", i1 false}
!484 = !{!"atomicBranch", i32 0}
!485 = !{!"disableEarlyOut", i1 false}
!486 = !{!"walkOrderEnabled", i1 false}
!487 = !{!"walkOrderOverride", i32 0}
!488 = !{!"ResForHfPacking"}
!489 = !{!"constantFoldSimdSize", i1 false}
!490 = !{!"isNodeShader", i1 false}
!491 = !{!"threadGroupMergeSize", i32 0}
!492 = !{!"threadGroupMergeOverY", i1 false}
!493 = !{!"msInfo", !494, !495, !496, !497, !498, !499, !500, !501, !502, !503, !504, !450, !448, !505, !506, !490}
!494 = !{!"PrimitiveTopology", i32 3}
!495 = !{!"MaxNumOfPrimitives", i32 0}
!496 = !{!"MaxNumOfVertices", i32 0}
!497 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!498 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!499 = !{!"WorkGroupSize", i32 0}
!500 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!501 = !{!"IndexFormat", i32 6}
!502 = !{!"SubgroupSize", i32 0}
!503 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!504 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!505 = !{!"Is16BMUEModeAllowed", i1 false}
!506 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!507 = !{!"taskInfo", !423, !499, !500, !502}
!508 = !{!"NBarrierCnt", i32 0}
!509 = !{!"rtInfo", !510, !511, !512, !513, !514, !515, !516, !517, !518, !519, !520, !521, !522, !523, !524, !525, !209}
!510 = !{!"RayQueryAllocSizeInBytes", i32 0}
!511 = !{!"NumContinuations", i32 0}
!512 = !{!"RTAsyncStackAddrspace", i32 -1}
!513 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!514 = !{!"SWHotZoneAddrspace", i32 -1}
!515 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!516 = !{!"SWStackAddrspace", i32 -1}
!517 = !{!"SWStackSurfaceStateOffset", i1* null}
!518 = !{!"RTSyncStackAddrspace", i32 -1}
!519 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!520 = !{!"doSyncDispatchRays", i1 false}
!521 = !{!"MemStyle", !"Xe"}
!522 = !{!"GlobalDataStyle", !"Xe"}
!523 = !{!"NeedsBTD", i1 true}
!524 = !{!"SERHitObjectFullType", i1* null}
!525 = !{!"uberTileDimensions", i1* null}
!526 = !{!"CurUniqueIndirectIdx", i32 0}
!527 = !{!"inlineDynTextures"}
!528 = !{!"inlineResInfoData"}
!529 = !{!"immConstant", !530, !531, !532}
!530 = !{!"data"}
!531 = !{!"sizes"}
!532 = !{!"zeroIdxs"}
!533 = !{!"stringConstants"}
!534 = !{!"inlineBuffers", !535, !539, !540}
!535 = !{!"inlineBuffersVec[0]", !536, !537, !538}
!536 = !{!"alignment", i32 0}
!537 = !{!"allocSize", i64 0}
!538 = !{!"Buffer"}
!539 = !{!"inlineBuffersVec[1]", !536, !537, !538}
!540 = !{!"inlineBuffersVec[2]", !536, !537, !538}
!541 = !{!"GlobalPointerProgramBinaryInfos"}
!542 = !{!"ConstantPointerProgramBinaryInfos"}
!543 = !{!"GlobalBufferAddressRelocInfo"}
!544 = !{!"ConstantBufferAddressRelocInfo"}
!545 = !{!"forceLscCacheList"}
!546 = !{!"SrvMap"}
!547 = !{!"RootConstantBufferOffsetInBytes"}
!548 = !{!"RasterizerOrderedByteAddressBuffer"}
!549 = !{!"RasterizerOrderedViews"}
!550 = !{!"MinNOSPushConstantSize", i32 0}
!551 = !{!"inlineProgramScopeOffsets"}
!552 = !{!"shaderData", !553}
!553 = !{!"numReplicas", i32 0}
!554 = !{!"URBInfo", !555, !556, !557}
!555 = !{!"has64BVertexHeaderInput", i1 false}
!556 = !{!"has64BVertexHeaderOutput", i1 false}
!557 = !{!"hasVertexHeader", i1 true}
!558 = !{!"m_ForcePullModel", i1 false}
!559 = !{!"UseBindlessImage", i1 true}
!560 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!561 = !{!"enableRangeReduce", i1 false}
!562 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!563 = !{!"enableFRemToSRemOpt", i1 false}
!564 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!565 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!566 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!567 = !{!"allowMatchMadOptimizationforVS", i1 false}
!568 = !{!"disableMatchMadOptimizationForCS", i1 false}
!569 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!570 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!571 = !{!"statefulResourcesNotAliased", i1 false}
!572 = !{!"disableMixMode", i1 false}
!573 = !{!"genericAccessesResolved", i1 false}
!574 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!575 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!576 = !{!"disableSeparateScratchWA", i1 false}
!577 = !{!"enableRemoveUnusedTGMFence", i1 false}
!578 = !{!"PrivateMemoryPerFG"}
!579 = !{!"m_OptsToDisable"}
!580 = !{!"capabilities", !581}
!581 = !{!"globalVariableDecorationsINTEL", i1 false}
!582 = !{!"extensions", !583}
!583 = !{!"spvINTELBindlessImages", i1 false}
!584 = !{!"m_ShaderResourceViewMcsMask", !585, !586}
!585 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!586 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!587 = !{!"computedDepthMode", i32 0}
!588 = !{!"isHDCFastClearShader", i1 false}
!589 = !{!"argRegisterReservations", !590}
!590 = !{!"argRegisterReservationsVec[0]", i32 0}
!591 = !{!"SIMD16_SpillThreshold", i8 0}
!592 = !{!"SIMD32_SpillThreshold", i8 0}
!593 = !{!"m_CacheControlOption", !594, !595, !596, !597}
!594 = !{!"LscLoadCacheControlOverride", i8 0}
!595 = !{!"LscStoreCacheControlOverride", i8 0}
!596 = !{!"TgmLoadCacheControlOverride", i8 0}
!597 = !{!"TgmStoreCacheControlOverride", i8 0}
!598 = !{!"ModuleUsesBindless", i1 false}
!599 = !{!"predicationMap"}
!600 = !{!"lifeTimeStartMap"}
!601 = !{!"HitGroups"}
!602 = !{i32 2, i32 0}
!603 = !{!"clang version 16.0.6"}
!604 = !{i32 1, !"wchar_size", i32 4}
!605 = !{!606}
!606 = !{i32 44, i32 4}
!607 = !{!608}
!608 = !{i32 4469}
!609 = !{!610}
!610 = !{i32 40, i32 196620}
!611 = !{!608, !612}
!612 = !{i32 4470}
!613 = distinct !{!613, !614}
!614 = !{!"llvm.loop.unroll.enable"}
!615 = distinct !{!615, !614}
!616 = distinct !{!616, !614}
!617 = !{!618}
!618 = distinct !{!618, !619}
!619 = distinct !{!619}
!620 = !{!621}
!621 = distinct !{!621, !622}
!622 = distinct !{!622}
!623 = !{!612}
!624 = distinct !{!624, !614}
!625 = distinct !{!625, !614}
!626 = !{!627}
!627 = distinct !{!627, !628}
!628 = distinct !{!628}
!629 = distinct !{!629, !614}
!630 = distinct !{!630, !614}
!631 = distinct !{!631, !614}
!632 = distinct !{!632, !614}
!633 = !{!634}
!634 = distinct !{!634, !635}
!635 = distinct !{!635}
!636 = !{!637}
!637 = distinct !{!637, !638}
!638 = distinct !{!638}
!639 = distinct !{!639, !614}
!640 = distinct !{!640, !614}
!641 = !{!642}
!642 = distinct !{!642, !643}
!643 = distinct !{!643}
!644 = distinct !{!644, !614}

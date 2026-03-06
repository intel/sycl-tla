; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0012.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
; BB0 :
  %14 = bitcast i64 %9 to <2 x i32>		; visa id: 2
  %15 = extractelement <2 x i32> %14, i32 0		; visa id: 3
  %16 = extractelement <2 x i32> %14, i32 1		; visa id: 3
  %17 = bitcast i64 %10 to <2 x i32>		; visa id: 3
  %18 = extractelement <2 x i32> %17, i32 0		; visa id: 4
  %19 = extractelement <2 x i32> %17, i32 1		; visa id: 4
  %20 = bitcast i64 %11 to <2 x i32>		; visa id: 4
  %21 = extractelement <2 x i32> %20, i32 0		; visa id: 5
  %22 = extractelement <2 x i32> %20, i32 1		; visa id: 5
  %23 = bitcast i64 %12 to <2 x i32>		; visa id: 5
  %24 = extractelement <2 x i32> %23, i32 0		; visa id: 6
  %25 = extractelement <2 x i32> %23, i32 1		; visa id: 6
  %26 = extractelement <8 x i32> %r0, i32 7		; visa id: 6
  %27 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %15, i32 %16)
  %28 = extractvalue { i32, i32 } %27, 0		; visa id: 7
  %29 = extractvalue { i32, i32 } %27, 1		; visa id: 7
  %30 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %18, i32 %19)
  %31 = extractvalue { i32, i32 } %30, 0		; visa id: 14
  %32 = extractvalue { i32, i32 } %30, 1		; visa id: 14
  %33 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %21, i32 %22)
  %34 = extractvalue { i32, i32 } %33, 0		; visa id: 21
  %35 = extractvalue { i32, i32 } %33, 1		; visa id: 21
  %36 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %24, i32 %25)
  %37 = extractvalue { i32, i32 } %36, 0		; visa id: 28
  %38 = extractvalue { i32, i32 } %36, 1		; visa id: 28
  %39 = icmp slt i32 %26, %8		; visa id: 35
  br i1 %39, label %.lr.ph, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 36

.._crit_edge72_crit_edge:                         ; preds = %13
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %13
; BB2 :
  %40 = bitcast i64 %const_reg_qword3 to <2 x i32>		; visa id: 38
  %41 = extractelement <2 x i32> %40, i32 0		; visa id: 39
  %42 = extractelement <2 x i32> %40, i32 1		; visa id: 39
  %43 = bitcast i64 %const_reg_qword5 to <2 x i32>		; visa id: 39
  %44 = extractelement <2 x i32> %43, i32 0		; visa id: 40
  %45 = extractelement <2 x i32> %43, i32 1		; visa id: 40
  %46 = bitcast i64 %const_reg_qword7 to <2 x i32>		; visa id: 40
  %47 = extractelement <2 x i32> %46, i32 0		; visa id: 41
  %48 = extractelement <2 x i32> %46, i32 1		; visa id: 41
  %49 = bitcast i64 %const_reg_qword9 to <2 x i32>		; visa id: 41
  %50 = extractelement <2 x i32> %49, i32 0		; visa id: 42
  %51 = extractelement <2 x i32> %49, i32 1		; visa id: 42
  %52 = extractelement <3 x i32> %numWorkGroups, i32 2		; visa id: 42
  %53 = extractelement <3 x i32> %localSize, i32 0		; visa id: 42
  %54 = extractelement <3 x i32> %localSize, i32 1		; visa id: 42
  %55 = extractelement <8 x i32> %r0, i32 1		; visa id: 42
  %56 = extractelement <8 x i32> %r0, i32 6		; visa id: 43
  %57 = mul i32 %55, %53		; visa id: 44
  %58 = zext i16 %localIdX to i32		; visa id: 45
  %59 = add i32 %57, %58		; visa id: 46
  %60 = shl i32 %59, 2		; visa id: 47
  %61 = mul i32 %56, %54		; visa id: 48
  %62 = zext i16 %localIdY to i32		; visa id: 49
  %63 = add i32 %61, %62		; visa id: 50
  %64 = shl i32 %63, 2		; visa id: 51
  %65 = insertelement <2 x i32> undef, i32 %28, i32 0		; visa id: 52
  %66 = insertelement <2 x i32> %65, i32 %29, i32 1		; visa id: 53
  %67 = bitcast <2 x i32> %66 to i64		; visa id: 54
  %68 = shl i64 %67, 1		; visa id: 56
  %69 = add i64 %68, %const_reg_qword		; visa id: 57
  %70 = insertelement <2 x i32> undef, i32 %31, i32 0		; visa id: 58
  %71 = insertelement <2 x i32> %70, i32 %32, i32 1		; visa id: 59
  %72 = bitcast <2 x i32> %71 to i64		; visa id: 60
  %73 = shl i64 %72, 1		; visa id: 62
  %74 = add i64 %73, %const_reg_qword4		; visa id: 63
  %75 = insertelement <2 x i32> undef, i32 %34, i32 0		; visa id: 64
  %76 = insertelement <2 x i32> %75, i32 %35, i32 1		; visa id: 65
  %77 = bitcast <2 x i32> %76 to i64		; visa id: 66
  %.op = shl i64 %77, 2		; visa id: 68
  %78 = bitcast i64 %.op to <2 x i32>		; visa id: 69
  %79 = extractelement <2 x i32> %78, i32 0		; visa id: 70
  %80 = extractelement <2 x i32> %78, i32 1		; visa id: 70
  %81 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !881		; visa id: 70
  %82 = select i1 %81, i32 %79, i32 0		; visa id: 71
  %83 = select i1 %81, i32 %80, i32 0		; visa id: 72
  %84 = insertelement <2 x i32> undef, i32 %82, i32 0		; visa id: 73
  %85 = insertelement <2 x i32> %84, i32 %83, i32 1		; visa id: 74
  %86 = bitcast <2 x i32> %85 to i64		; visa id: 75
  %87 = add i64 %86, %const_reg_qword6		; visa id: 77
  %88 = insertelement <2 x i32> undef, i32 %37, i32 0		; visa id: 78
  %89 = insertelement <2 x i32> %88, i32 %38, i32 1		; visa id: 79
  %90 = bitcast <2 x i32> %89 to i64		; visa id: 80
  %91 = shl i64 %90, 2		; visa id: 82
  %92 = add i64 %91, %const_reg_qword8		; visa id: 83
  %93 = icmp sgt i32 %const_reg_dword2, 0		; visa id: 84
  %94 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %15, i32 %16)
  %95 = extractvalue { i32, i32 } %94, 0		; visa id: 85
  %96 = extractvalue { i32, i32 } %94, 1		; visa id: 85
  %97 = insertelement <2 x i32> undef, i32 %95, i32 0		; visa id: 92
  %98 = insertelement <2 x i32> %97, i32 %96, i32 1		; visa id: 93
  %99 = bitcast <2 x i32> %98 to i64		; visa id: 94
  %100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %18, i32 %19)
  %101 = extractvalue { i32, i32 } %100, 0		; visa id: 96
  %102 = extractvalue { i32, i32 } %100, 1		; visa id: 96
  %103 = insertelement <2 x i32> undef, i32 %101, i32 0		; visa id: 103
  %104 = insertelement <2 x i32> %103, i32 %102, i32 1		; visa id: 104
  %105 = bitcast <2 x i32> %104 to i64		; visa id: 105
  %106 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %21, i32 %22)
  %107 = extractvalue { i32, i32 } %106, 0		; visa id: 107
  %108 = extractvalue { i32, i32 } %106, 1		; visa id: 107
  %109 = insertelement <2 x i32> undef, i32 %107, i32 0		; visa id: 114
  %110 = insertelement <2 x i32> %109, i32 %108, i32 1		; visa id: 115
  %111 = bitcast <2 x i32> %110 to i64		; visa id: 116
  %112 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %24, i32 %25)
  %113 = extractvalue { i32, i32 } %112, 0		; visa id: 118
  %114 = extractvalue { i32, i32 } %112, 1		; visa id: 118
  %115 = insertelement <2 x i32> undef, i32 %113, i32 0		; visa id: 125
  %116 = insertelement <2 x i32> %115, i32 %114, i32 1		; visa id: 126
  %117 = bitcast <2 x i32> %116 to i64		; visa id: 127
  %118 = icmp slt i32 %60, %const_reg_dword
  %119 = icmp slt i32 %64, %const_reg_dword1		; visa id: 129
  %120 = and i1 %118, %119		; visa id: 130
  %121 = add i32 %60, 1		; visa id: 132
  %122 = icmp slt i32 %121, %const_reg_dword
  %123 = icmp slt i32 %64, %const_reg_dword1		; visa id: 133
  %124 = and i1 %122, %123		; visa id: 134
  %125 = add i32 %60, 2		; visa id: 136
  %126 = icmp slt i32 %125, %const_reg_dword
  %127 = icmp slt i32 %64, %const_reg_dword1		; visa id: 137
  %128 = and i1 %126, %127		; visa id: 138
  %129 = add i32 %60, 3		; visa id: 140
  %130 = icmp slt i32 %129, %const_reg_dword
  %131 = icmp slt i32 %64, %const_reg_dword1		; visa id: 141
  %132 = and i1 %130, %131		; visa id: 142
  %133 = add i32 %64, 1		; visa id: 144
  %134 = icmp slt i32 %133, %const_reg_dword1		; visa id: 145
  %135 = icmp slt i32 %60, %const_reg_dword
  %136 = and i1 %135, %134		; visa id: 146
  %137 = icmp slt i32 %121, %const_reg_dword
  %138 = icmp slt i32 %133, %const_reg_dword1		; visa id: 148
  %139 = and i1 %137, %138		; visa id: 149
  %140 = icmp slt i32 %125, %const_reg_dword
  %141 = icmp slt i32 %133, %const_reg_dword1		; visa id: 151
  %142 = and i1 %140, %141		; visa id: 152
  %143 = icmp slt i32 %129, %const_reg_dword
  %144 = icmp slt i32 %133, %const_reg_dword1		; visa id: 154
  %145 = and i1 %143, %144		; visa id: 155
  %146 = add i32 %64, 2		; visa id: 157
  %147 = icmp slt i32 %146, %const_reg_dword1		; visa id: 158
  %148 = icmp slt i32 %60, %const_reg_dword
  %149 = and i1 %148, %147		; visa id: 159
  %150 = icmp slt i32 %121, %const_reg_dword
  %151 = icmp slt i32 %146, %const_reg_dword1		; visa id: 161
  %152 = and i1 %150, %151		; visa id: 162
  %153 = icmp slt i32 %125, %const_reg_dword
  %154 = icmp slt i32 %146, %const_reg_dword1		; visa id: 164
  %155 = and i1 %153, %154		; visa id: 165
  %156 = icmp slt i32 %129, %const_reg_dword
  %157 = icmp slt i32 %146, %const_reg_dword1		; visa id: 167
  %158 = and i1 %156, %157		; visa id: 168
  %159 = add i32 %64, 3		; visa id: 170
  %160 = icmp slt i32 %159, %const_reg_dword1		; visa id: 171
  %161 = icmp slt i32 %60, %const_reg_dword
  %162 = and i1 %161, %160		; visa id: 172
  %163 = icmp slt i32 %121, %const_reg_dword
  %164 = icmp slt i32 %159, %const_reg_dword1		; visa id: 174
  %165 = and i1 %163, %164		; visa id: 175
  %166 = icmp slt i32 %125, %const_reg_dword
  %167 = icmp slt i32 %159, %const_reg_dword1		; visa id: 177
  %168 = and i1 %166, %167		; visa id: 178
  %169 = icmp slt i32 %129, %const_reg_dword
  %170 = icmp slt i32 %159, %const_reg_dword1		; visa id: 180
  %171 = and i1 %169, %170		; visa id: 181
  %172 = ashr i32 %60, 31		; visa id: 183
  %173 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %41, i32 %42)
  %174 = extractvalue { i32, i32 } %173, 0		; visa id: 184
  %175 = extractvalue { i32, i32 } %173, 1		; visa id: 184
  %176 = insertelement <2 x i32> undef, i32 %174, i32 0		; visa id: 191
  %177 = insertelement <2 x i32> %176, i32 %175, i32 1		; visa id: 192
  %178 = bitcast <2 x i32> %177 to i64		; visa id: 193
  %179 = shl i64 %178, 1		; visa id: 197
  %180 = sext i32 %64 to i64		; visa id: 198
  %181 = shl nsw i64 %180, 1		; visa id: 199
  %182 = ashr i32 %121, 31		; visa id: 200
  %183 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %41, i32 %42)
  %184 = extractvalue { i32, i32 } %183, 0		; visa id: 201
  %185 = extractvalue { i32, i32 } %183, 1		; visa id: 201
  %186 = insertelement <2 x i32> undef, i32 %184, i32 0		; visa id: 208
  %187 = insertelement <2 x i32> %186, i32 %185, i32 1		; visa id: 209
  %188 = bitcast <2 x i32> %187 to i64		; visa id: 210
  %189 = shl i64 %188, 1		; visa id: 214
  %190 = ashr i32 %125, 31		; visa id: 215
  %191 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %41, i32 %42)
  %192 = extractvalue { i32, i32 } %191, 0		; visa id: 216
  %193 = extractvalue { i32, i32 } %191, 1		; visa id: 216
  %194 = insertelement <2 x i32> undef, i32 %192, i32 0		; visa id: 223
  %195 = insertelement <2 x i32> %194, i32 %193, i32 1		; visa id: 224
  %196 = bitcast <2 x i32> %195 to i64		; visa id: 225
  %197 = shl i64 %196, 1		; visa id: 229
  %198 = ashr i32 %129, 31		; visa id: 230
  %199 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %41, i32 %42)
  %200 = extractvalue { i32, i32 } %199, 0		; visa id: 231
  %201 = extractvalue { i32, i32 } %199, 1		; visa id: 231
  %202 = insertelement <2 x i32> undef, i32 %200, i32 0		; visa id: 238
  %203 = insertelement <2 x i32> %202, i32 %201, i32 1		; visa id: 239
  %204 = bitcast <2 x i32> %203 to i64		; visa id: 240
  %205 = shl i64 %204, 1		; visa id: 244
  %206 = sext i32 %133 to i64		; visa id: 245
  %207 = shl nsw i64 %206, 1		; visa id: 246
  %208 = sext i32 %146 to i64		; visa id: 247
  %209 = shl nsw i64 %208, 1		; visa id: 248
  %210 = sext i32 %159 to i64		; visa id: 249
  %211 = shl nsw i64 %210, 1		; visa id: 250
  %212 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %50, i32 %51)
  %213 = extractvalue { i32, i32 } %212, 0		; visa id: 251
  %214 = extractvalue { i32, i32 } %212, 1		; visa id: 251
  %215 = insertelement <2 x i32> undef, i32 %213, i32 0		; visa id: 258
  %216 = insertelement <2 x i32> %215, i32 %214, i32 1		; visa id: 259
  %217 = bitcast <2 x i32> %216 to i64		; visa id: 260
  %218 = add nsw i64 %217, %180		; visa id: 264
  %219 = shl i64 %218, 2		; visa id: 265
  %220 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %47, i32 %48)
  %221 = extractvalue { i32, i32 } %220, 0		; visa id: 266
  %222 = extractvalue { i32, i32 } %220, 1		; visa id: 266
  %223 = insertelement <2 x i32> undef, i32 %221, i32 0		; visa id: 273
  %224 = insertelement <2 x i32> %223, i32 %222, i32 1		; visa id: 274
  %225 = bitcast <2 x i32> %224 to i64		; visa id: 275
  %226 = shl i64 %225, 2		; visa id: 279
  %227 = shl nsw i64 %180, 2		; visa id: 280
  %228 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %50, i32 %51)
  %229 = extractvalue { i32, i32 } %228, 0		; visa id: 281
  %230 = extractvalue { i32, i32 } %228, 1		; visa id: 281
  %231 = insertelement <2 x i32> undef, i32 %229, i32 0		; visa id: 288
  %232 = insertelement <2 x i32> %231, i32 %230, i32 1		; visa id: 289
  %233 = bitcast <2 x i32> %232 to i64		; visa id: 290
  %234 = add nsw i64 %233, %180		; visa id: 294
  %235 = shl i64 %234, 2		; visa id: 295
  %236 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %47, i32 %48)
  %237 = extractvalue { i32, i32 } %236, 0		; visa id: 296
  %238 = extractvalue { i32, i32 } %236, 1		; visa id: 296
  %239 = insertelement <2 x i32> undef, i32 %237, i32 0		; visa id: 303
  %240 = insertelement <2 x i32> %239, i32 %238, i32 1		; visa id: 304
  %241 = bitcast <2 x i32> %240 to i64		; visa id: 305
  %242 = shl i64 %241, 2		; visa id: 309
  %243 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %50, i32 %51)
  %244 = extractvalue { i32, i32 } %243, 0		; visa id: 310
  %245 = extractvalue { i32, i32 } %243, 1		; visa id: 310
  %246 = insertelement <2 x i32> undef, i32 %244, i32 0		; visa id: 317
  %247 = insertelement <2 x i32> %246, i32 %245, i32 1		; visa id: 318
  %248 = bitcast <2 x i32> %247 to i64		; visa id: 319
  %249 = add nsw i64 %248, %180		; visa id: 323
  %250 = shl i64 %249, 2		; visa id: 324
  %251 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %47, i32 %48)
  %252 = extractvalue { i32, i32 } %251, 0		; visa id: 325
  %253 = extractvalue { i32, i32 } %251, 1		; visa id: 325
  %254 = insertelement <2 x i32> undef, i32 %252, i32 0		; visa id: 332
  %255 = insertelement <2 x i32> %254, i32 %253, i32 1		; visa id: 333
  %256 = bitcast <2 x i32> %255 to i64		; visa id: 334
  %257 = shl i64 %256, 2		; visa id: 338
  %258 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %50, i32 %51)
  %259 = extractvalue { i32, i32 } %258, 0		; visa id: 339
  %260 = extractvalue { i32, i32 } %258, 1		; visa id: 339
  %261 = insertelement <2 x i32> undef, i32 %259, i32 0		; visa id: 346
  %262 = insertelement <2 x i32> %261, i32 %260, i32 1		; visa id: 347
  %263 = bitcast <2 x i32> %262 to i64		; visa id: 348
  %264 = add nsw i64 %263, %180		; visa id: 352
  %265 = shl i64 %264, 2		; visa id: 353
  %266 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %47, i32 %48)
  %267 = extractvalue { i32, i32 } %266, 0		; visa id: 354
  %268 = extractvalue { i32, i32 } %266, 1		; visa id: 354
  %269 = insertelement <2 x i32> undef, i32 %267, i32 0		; visa id: 361
  %270 = insertelement <2 x i32> %269, i32 %268, i32 1		; visa id: 362
  %271 = bitcast <2 x i32> %270 to i64		; visa id: 363
  %272 = shl i64 %271, 2		; visa id: 367
  %273 = add nsw i64 %217, %206		; visa id: 368
  %274 = shl i64 %273, 2		; visa id: 369
  %275 = shl nsw i64 %206, 2		; visa id: 370
  %276 = add nsw i64 %233, %206		; visa id: 371
  %277 = shl i64 %276, 2		; visa id: 372
  %278 = add nsw i64 %248, %206		; visa id: 373
  %279 = shl i64 %278, 2		; visa id: 374
  %280 = add nsw i64 %263, %206		; visa id: 375
  %281 = shl i64 %280, 2		; visa id: 376
  %282 = add nsw i64 %217, %208		; visa id: 377
  %283 = shl i64 %282, 2		; visa id: 378
  %284 = shl nsw i64 %208, 2		; visa id: 379
  %285 = add nsw i64 %233, %208		; visa id: 380
  %286 = shl i64 %285, 2		; visa id: 381
  %287 = add nsw i64 %248, %208		; visa id: 382
  %288 = shl i64 %287, 2		; visa id: 383
  %289 = add nsw i64 %263, %208		; visa id: 384
  %290 = shl i64 %289, 2		; visa id: 385
  %291 = add nsw i64 %217, %210		; visa id: 386
  %292 = shl i64 %291, 2		; visa id: 387
  %293 = shl nsw i64 %210, 2		; visa id: 388
  %294 = add nsw i64 %233, %210		; visa id: 389
  %295 = shl i64 %294, 2		; visa id: 390
  %296 = add nsw i64 %248, %210		; visa id: 391
  %297 = shl i64 %296, 2		; visa id: 392
  %298 = add nsw i64 %263, %210		; visa id: 393
  %299 = shl i64 %298, 2		; visa id: 394
  %300 = shl i64 %99, 1		; visa id: 395
  %301 = shl i64 %105, 1		; visa id: 396
  %.op991 = shl i64 %111, 2		; visa id: 397
  %302 = bitcast i64 %.op991 to <2 x i32>		; visa id: 398
  %303 = extractelement <2 x i32> %302, i32 0		; visa id: 399
  %304 = extractelement <2 x i32> %302, i32 1		; visa id: 399
  %305 = select i1 %81, i32 %303, i32 0		; visa id: 399
  %306 = select i1 %81, i32 %304, i32 0		; visa id: 400
  %307 = insertelement <2 x i32> undef, i32 %305, i32 0		; visa id: 401
  %308 = insertelement <2 x i32> %307, i32 %306, i32 1		; visa id: 402
  %309 = bitcast <2 x i32> %308 to i64		; visa id: 403
  %310 = shl i64 %117, 2		; visa id: 405
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879		; visa id: 406

.preheader2.preheader:                            ; preds = %.preheader1.3..preheader2.preheader_crit_edge, %.lr.ph
; BB3 :
  %311 = phi i32 [ %26, %.lr.ph ], [ %1039, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %92, %.lr.ph ], [ %1044, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in988 = phi i64 [ %87, %.lr.ph ], [ %1043, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in989 = phi i64 [ %74, %.lr.ph ], [ %1042, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in990 = phi i64 [ %69, %.lr.ph ], [ %1041, %.preheader1.3..preheader2.preheader_crit_edge ]
  br i1 %93, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 407

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
; BB4 :
  br label %.preheader1.preheader, !stats.blockFrequency.digits !884, !stats.blockFrequency.scale !879		; visa id: 425

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
; BB5 :
  %312 = add i64 %.in990, %179		; visa id: 427
  %313 = add i64 %.in990, %189		; visa id: 428
  %314 = add i64 %.in990, %197		; visa id: 429
  %315 = add i64 %.in990, %205		; visa id: 430
  br label %.preheader.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879		; visa id: 448

.preheader.preheader:                             ; preds = %.preheader.3..preheader.preheader_crit_edge, %.preheader.preheader.preheader
; BB6 :
  %316 = phi float [ %764, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %317 = phi float [ %737, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %318 = phi float [ %710, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %319 = phi float [ %683, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %320 = phi float [ %656, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %321 = phi float [ %629, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %322 = phi float [ %602, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %323 = phi float [ %575, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %324 = phi float [ %548, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %325 = phi float [ %521, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %326 = phi float [ %494, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %327 = phi float [ %467, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %328 = phi float [ %440, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %329 = phi float [ %413, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %330 = phi float [ %386, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %331 = phi float [ %359, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %332 = phi i32 [ %765, %.preheader.3..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
  br i1 %120, label %333, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 449

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

333:                                              ; preds = %.preheader.preheader
; BB8 :
  %.sroa.64.0.insert.ext = zext i32 %332 to i64		; visa id: 451
  %334 = shl nuw nsw i64 %.sroa.64.0.insert.ext, 1		; visa id: 452
  %335 = add i64 %312, %334		; visa id: 453
  %336 = inttoptr i64 %335 to i16 addrspace(4)*		; visa id: 454
  %337 = addrspacecast i16 addrspace(4)* %336 to i16 addrspace(1)*		; visa id: 454
  %338 = load i16, i16 addrspace(1)* %337, align 2		; visa id: 455
  %339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %340 = extractvalue { i32, i32 } %339, 0		; visa id: 457
  %341 = extractvalue { i32, i32 } %339, 1		; visa id: 457
  %342 = insertelement <2 x i32> undef, i32 %340, i32 0		; visa id: 464
  %343 = insertelement <2 x i32> %342, i32 %341, i32 1		; visa id: 465
  %344 = bitcast <2 x i32> %343 to i64		; visa id: 466
  %345 = shl i64 %344, 1		; visa id: 468
  %346 = add i64 %.in989, %345		; visa id: 469
  %347 = add i64 %346, %181		; visa id: 470
  %348 = inttoptr i64 %347 to i16 addrspace(4)*		; visa id: 471
  %349 = addrspacecast i16 addrspace(4)* %348 to i16 addrspace(1)*		; visa id: 471
  %350 = load i16, i16 addrspace(1)* %349, align 2		; visa id: 472
  %351 = zext i16 %338 to i32		; visa id: 474
  %352 = shl nuw i32 %351, 16, !spirv.Decorations !888		; visa id: 475
  %353 = bitcast i32 %352 to float
  %354 = zext i16 %350 to i32		; visa id: 476
  %355 = shl nuw i32 %354, 16, !spirv.Decorations !888		; visa id: 477
  %356 = bitcast i32 %355 to float
  %357 = fmul reassoc nsz arcp contract float %353, %356, !spirv.Decorations !881
  %358 = fadd reassoc nsz arcp contract float %357, %331, !spirv.Decorations !881		; visa id: 478
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 479

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %333
; BB9 :
  %359 = phi float [ %358, %333 ], [ %331, %.preheader.preheader.._crit_edge_crit_edge ]
  br i1 %124, label %360, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 480

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
; BB:
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

360:                                              ; preds = %._crit_edge
; BB11 :
  %.sroa.64.0.insert.ext203 = zext i32 %332 to i64		; visa id: 482
  %361 = shl nuw nsw i64 %.sroa.64.0.insert.ext203, 1		; visa id: 483
  %362 = add i64 %313, %361		; visa id: 484
  %363 = inttoptr i64 %362 to i16 addrspace(4)*		; visa id: 485
  %364 = addrspacecast i16 addrspace(4)* %363 to i16 addrspace(1)*		; visa id: 485
  %365 = load i16, i16 addrspace(1)* %364, align 2		; visa id: 486
  %366 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %367 = extractvalue { i32, i32 } %366, 0		; visa id: 488
  %368 = extractvalue { i32, i32 } %366, 1		; visa id: 488
  %369 = insertelement <2 x i32> undef, i32 %367, i32 0		; visa id: 495
  %370 = insertelement <2 x i32> %369, i32 %368, i32 1		; visa id: 496
  %371 = bitcast <2 x i32> %370 to i64		; visa id: 497
  %372 = shl i64 %371, 1		; visa id: 499
  %373 = add i64 %.in989, %372		; visa id: 500
  %374 = add i64 %373, %181		; visa id: 501
  %375 = inttoptr i64 %374 to i16 addrspace(4)*		; visa id: 502
  %376 = addrspacecast i16 addrspace(4)* %375 to i16 addrspace(1)*		; visa id: 502
  %377 = load i16, i16 addrspace(1)* %376, align 2		; visa id: 503
  %378 = zext i16 %365 to i32		; visa id: 505
  %379 = shl nuw i32 %378, 16, !spirv.Decorations !888		; visa id: 506
  %380 = bitcast i32 %379 to float
  %381 = zext i16 %377 to i32		; visa id: 507
  %382 = shl nuw i32 %381, 16, !spirv.Decorations !888		; visa id: 508
  %383 = bitcast i32 %382 to float
  %384 = fmul reassoc nsz arcp contract float %380, %383, !spirv.Decorations !881
  %385 = fadd reassoc nsz arcp contract float %384, %330, !spirv.Decorations !881		; visa id: 509
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 510

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %360
; BB12 :
  %386 = phi float [ %385, %360 ], [ %330, %._crit_edge.._crit_edge.1_crit_edge ]
  br i1 %128, label %387, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 511

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
; BB:
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

387:                                              ; preds = %._crit_edge.1
; BB14 :
  %.sroa.64.0.insert.ext208 = zext i32 %332 to i64		; visa id: 513
  %388 = shl nuw nsw i64 %.sroa.64.0.insert.ext208, 1		; visa id: 514
  %389 = add i64 %314, %388		; visa id: 515
  %390 = inttoptr i64 %389 to i16 addrspace(4)*		; visa id: 516
  %391 = addrspacecast i16 addrspace(4)* %390 to i16 addrspace(1)*		; visa id: 516
  %392 = load i16, i16 addrspace(1)* %391, align 2		; visa id: 517
  %393 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %394 = extractvalue { i32, i32 } %393, 0		; visa id: 519
  %395 = extractvalue { i32, i32 } %393, 1		; visa id: 519
  %396 = insertelement <2 x i32> undef, i32 %394, i32 0		; visa id: 526
  %397 = insertelement <2 x i32> %396, i32 %395, i32 1		; visa id: 527
  %398 = bitcast <2 x i32> %397 to i64		; visa id: 528
  %399 = shl i64 %398, 1		; visa id: 530
  %400 = add i64 %.in989, %399		; visa id: 531
  %401 = add i64 %400, %181		; visa id: 532
  %402 = inttoptr i64 %401 to i16 addrspace(4)*		; visa id: 533
  %403 = addrspacecast i16 addrspace(4)* %402 to i16 addrspace(1)*		; visa id: 533
  %404 = load i16, i16 addrspace(1)* %403, align 2		; visa id: 534
  %405 = zext i16 %392 to i32		; visa id: 536
  %406 = shl nuw i32 %405, 16, !spirv.Decorations !888		; visa id: 537
  %407 = bitcast i32 %406 to float
  %408 = zext i16 %404 to i32		; visa id: 538
  %409 = shl nuw i32 %408, 16, !spirv.Decorations !888		; visa id: 539
  %410 = bitcast i32 %409 to float
  %411 = fmul reassoc nsz arcp contract float %407, %410, !spirv.Decorations !881
  %412 = fadd reassoc nsz arcp contract float %411, %329, !spirv.Decorations !881		; visa id: 540
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 541

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %387
; BB15 :
  %413 = phi float [ %412, %387 ], [ %329, %._crit_edge.1.._crit_edge.2_crit_edge ]
  br i1 %132, label %414, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 542

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
; BB:
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

414:                                              ; preds = %._crit_edge.2
; BB17 :
  %.sroa.64.0.insert.ext213 = zext i32 %332 to i64		; visa id: 544
  %415 = shl nuw nsw i64 %.sroa.64.0.insert.ext213, 1		; visa id: 545
  %416 = add i64 %315, %415		; visa id: 546
  %417 = inttoptr i64 %416 to i16 addrspace(4)*		; visa id: 547
  %418 = addrspacecast i16 addrspace(4)* %417 to i16 addrspace(1)*		; visa id: 547
  %419 = load i16, i16 addrspace(1)* %418, align 2		; visa id: 548
  %420 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %421 = extractvalue { i32, i32 } %420, 0		; visa id: 550
  %422 = extractvalue { i32, i32 } %420, 1		; visa id: 550
  %423 = insertelement <2 x i32> undef, i32 %421, i32 0		; visa id: 557
  %424 = insertelement <2 x i32> %423, i32 %422, i32 1		; visa id: 558
  %425 = bitcast <2 x i32> %424 to i64		; visa id: 559
  %426 = shl i64 %425, 1		; visa id: 561
  %427 = add i64 %.in989, %426		; visa id: 562
  %428 = add i64 %427, %181		; visa id: 563
  %429 = inttoptr i64 %428 to i16 addrspace(4)*		; visa id: 564
  %430 = addrspacecast i16 addrspace(4)* %429 to i16 addrspace(1)*		; visa id: 564
  %431 = load i16, i16 addrspace(1)* %430, align 2		; visa id: 565
  %432 = zext i16 %419 to i32		; visa id: 567
  %433 = shl nuw i32 %432, 16, !spirv.Decorations !888		; visa id: 568
  %434 = bitcast i32 %433 to float
  %435 = zext i16 %431 to i32		; visa id: 569
  %436 = shl nuw i32 %435, 16, !spirv.Decorations !888		; visa id: 570
  %437 = bitcast i32 %436 to float
  %438 = fmul reassoc nsz arcp contract float %434, %437, !spirv.Decorations !881
  %439 = fadd reassoc nsz arcp contract float %438, %328, !spirv.Decorations !881		; visa id: 571
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 572

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %414
; BB18 :
  %440 = phi float [ %439, %414 ], [ %328, %._crit_edge.2..preheader_crit_edge ]
  br i1 %136, label %441, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 573

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
; BB:
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

441:                                              ; preds = %.preheader
; BB20 :
  %.sroa.64.0.insert.ext218 = zext i32 %332 to i64		; visa id: 575
  %442 = shl nuw nsw i64 %.sroa.64.0.insert.ext218, 1		; visa id: 576
  %443 = add i64 %312, %442		; visa id: 577
  %444 = inttoptr i64 %443 to i16 addrspace(4)*		; visa id: 578
  %445 = addrspacecast i16 addrspace(4)* %444 to i16 addrspace(1)*		; visa id: 578
  %446 = load i16, i16 addrspace(1)* %445, align 2		; visa id: 579
  %447 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %448 = extractvalue { i32, i32 } %447, 0		; visa id: 581
  %449 = extractvalue { i32, i32 } %447, 1		; visa id: 581
  %450 = insertelement <2 x i32> undef, i32 %448, i32 0		; visa id: 588
  %451 = insertelement <2 x i32> %450, i32 %449, i32 1		; visa id: 589
  %452 = bitcast <2 x i32> %451 to i64		; visa id: 590
  %453 = shl i64 %452, 1		; visa id: 592
  %454 = add i64 %.in989, %453		; visa id: 593
  %455 = add i64 %454, %207		; visa id: 594
  %456 = inttoptr i64 %455 to i16 addrspace(4)*		; visa id: 595
  %457 = addrspacecast i16 addrspace(4)* %456 to i16 addrspace(1)*		; visa id: 595
  %458 = load i16, i16 addrspace(1)* %457, align 2		; visa id: 596
  %459 = zext i16 %446 to i32		; visa id: 598
  %460 = shl nuw i32 %459, 16, !spirv.Decorations !888		; visa id: 599
  %461 = bitcast i32 %460 to float
  %462 = zext i16 %458 to i32		; visa id: 600
  %463 = shl nuw i32 %462, 16, !spirv.Decorations !888		; visa id: 601
  %464 = bitcast i32 %463 to float
  %465 = fmul reassoc nsz arcp contract float %461, %464, !spirv.Decorations !881
  %466 = fadd reassoc nsz arcp contract float %465, %327, !spirv.Decorations !881		; visa id: 602
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 603

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %441
; BB21 :
  %467 = phi float [ %466, %441 ], [ %327, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %139, label %468, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 604

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
; BB:
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

468:                                              ; preds = %._crit_edge.173
; BB23 :
  %.sroa.64.0.insert.ext223 = zext i32 %332 to i64		; visa id: 606
  %469 = shl nuw nsw i64 %.sroa.64.0.insert.ext223, 1		; visa id: 607
  %470 = add i64 %313, %469		; visa id: 608
  %471 = inttoptr i64 %470 to i16 addrspace(4)*		; visa id: 609
  %472 = addrspacecast i16 addrspace(4)* %471 to i16 addrspace(1)*		; visa id: 609
  %473 = load i16, i16 addrspace(1)* %472, align 2		; visa id: 610
  %474 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %475 = extractvalue { i32, i32 } %474, 0		; visa id: 612
  %476 = extractvalue { i32, i32 } %474, 1		; visa id: 612
  %477 = insertelement <2 x i32> undef, i32 %475, i32 0		; visa id: 619
  %478 = insertelement <2 x i32> %477, i32 %476, i32 1		; visa id: 620
  %479 = bitcast <2 x i32> %478 to i64		; visa id: 621
  %480 = shl i64 %479, 1		; visa id: 623
  %481 = add i64 %.in989, %480		; visa id: 624
  %482 = add i64 %481, %207		; visa id: 625
  %483 = inttoptr i64 %482 to i16 addrspace(4)*		; visa id: 626
  %484 = addrspacecast i16 addrspace(4)* %483 to i16 addrspace(1)*		; visa id: 626
  %485 = load i16, i16 addrspace(1)* %484, align 2		; visa id: 627
  %486 = zext i16 %473 to i32		; visa id: 629
  %487 = shl nuw i32 %486, 16, !spirv.Decorations !888		; visa id: 630
  %488 = bitcast i32 %487 to float
  %489 = zext i16 %485 to i32		; visa id: 631
  %490 = shl nuw i32 %489, 16, !spirv.Decorations !888		; visa id: 632
  %491 = bitcast i32 %490 to float
  %492 = fmul reassoc nsz arcp contract float %488, %491, !spirv.Decorations !881
  %493 = fadd reassoc nsz arcp contract float %492, %326, !spirv.Decorations !881		; visa id: 633
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 634

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %468
; BB24 :
  %494 = phi float [ %493, %468 ], [ %326, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %142, label %495, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 635

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
; BB:
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

495:                                              ; preds = %._crit_edge.1.1
; BB26 :
  %.sroa.64.0.insert.ext228 = zext i32 %332 to i64		; visa id: 637
  %496 = shl nuw nsw i64 %.sroa.64.0.insert.ext228, 1		; visa id: 638
  %497 = add i64 %314, %496		; visa id: 639
  %498 = inttoptr i64 %497 to i16 addrspace(4)*		; visa id: 640
  %499 = addrspacecast i16 addrspace(4)* %498 to i16 addrspace(1)*		; visa id: 640
  %500 = load i16, i16 addrspace(1)* %499, align 2		; visa id: 641
  %501 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %502 = extractvalue { i32, i32 } %501, 0		; visa id: 643
  %503 = extractvalue { i32, i32 } %501, 1		; visa id: 643
  %504 = insertelement <2 x i32> undef, i32 %502, i32 0		; visa id: 650
  %505 = insertelement <2 x i32> %504, i32 %503, i32 1		; visa id: 651
  %506 = bitcast <2 x i32> %505 to i64		; visa id: 652
  %507 = shl i64 %506, 1		; visa id: 654
  %508 = add i64 %.in989, %507		; visa id: 655
  %509 = add i64 %508, %207		; visa id: 656
  %510 = inttoptr i64 %509 to i16 addrspace(4)*		; visa id: 657
  %511 = addrspacecast i16 addrspace(4)* %510 to i16 addrspace(1)*		; visa id: 657
  %512 = load i16, i16 addrspace(1)* %511, align 2		; visa id: 658
  %513 = zext i16 %500 to i32		; visa id: 660
  %514 = shl nuw i32 %513, 16, !spirv.Decorations !888		; visa id: 661
  %515 = bitcast i32 %514 to float
  %516 = zext i16 %512 to i32		; visa id: 662
  %517 = shl nuw i32 %516, 16, !spirv.Decorations !888		; visa id: 663
  %518 = bitcast i32 %517 to float
  %519 = fmul reassoc nsz arcp contract float %515, %518, !spirv.Decorations !881
  %520 = fadd reassoc nsz arcp contract float %519, %325, !spirv.Decorations !881		; visa id: 664
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 665

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %495
; BB27 :
  %521 = phi float [ %520, %495 ], [ %325, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %145, label %522, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 666

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
; BB:
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

522:                                              ; preds = %._crit_edge.2.1
; BB29 :
  %.sroa.64.0.insert.ext233 = zext i32 %332 to i64		; visa id: 668
  %523 = shl nuw nsw i64 %.sroa.64.0.insert.ext233, 1		; visa id: 669
  %524 = add i64 %315, %523		; visa id: 670
  %525 = inttoptr i64 %524 to i16 addrspace(4)*		; visa id: 671
  %526 = addrspacecast i16 addrspace(4)* %525 to i16 addrspace(1)*		; visa id: 671
  %527 = load i16, i16 addrspace(1)* %526, align 2		; visa id: 672
  %528 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %529 = extractvalue { i32, i32 } %528, 0		; visa id: 674
  %530 = extractvalue { i32, i32 } %528, 1		; visa id: 674
  %531 = insertelement <2 x i32> undef, i32 %529, i32 0		; visa id: 681
  %532 = insertelement <2 x i32> %531, i32 %530, i32 1		; visa id: 682
  %533 = bitcast <2 x i32> %532 to i64		; visa id: 683
  %534 = shl i64 %533, 1		; visa id: 685
  %535 = add i64 %.in989, %534		; visa id: 686
  %536 = add i64 %535, %207		; visa id: 687
  %537 = inttoptr i64 %536 to i16 addrspace(4)*		; visa id: 688
  %538 = addrspacecast i16 addrspace(4)* %537 to i16 addrspace(1)*		; visa id: 688
  %539 = load i16, i16 addrspace(1)* %538, align 2		; visa id: 689
  %540 = zext i16 %527 to i32		; visa id: 691
  %541 = shl nuw i32 %540, 16, !spirv.Decorations !888		; visa id: 692
  %542 = bitcast i32 %541 to float
  %543 = zext i16 %539 to i32		; visa id: 693
  %544 = shl nuw i32 %543, 16, !spirv.Decorations !888		; visa id: 694
  %545 = bitcast i32 %544 to float
  %546 = fmul reassoc nsz arcp contract float %542, %545, !spirv.Decorations !881
  %547 = fadd reassoc nsz arcp contract float %546, %324, !spirv.Decorations !881		; visa id: 695
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 696

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %522
; BB30 :
  %548 = phi float [ %547, %522 ], [ %324, %._crit_edge.2.1..preheader.1_crit_edge ]
  br i1 %149, label %549, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 697

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
; BB:
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

549:                                              ; preds = %.preheader.1
; BB32 :
  %.sroa.64.0.insert.ext238 = zext i32 %332 to i64		; visa id: 699
  %550 = shl nuw nsw i64 %.sroa.64.0.insert.ext238, 1		; visa id: 700
  %551 = add i64 %312, %550		; visa id: 701
  %552 = inttoptr i64 %551 to i16 addrspace(4)*		; visa id: 702
  %553 = addrspacecast i16 addrspace(4)* %552 to i16 addrspace(1)*		; visa id: 702
  %554 = load i16, i16 addrspace(1)* %553, align 2		; visa id: 703
  %555 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %556 = extractvalue { i32, i32 } %555, 0		; visa id: 705
  %557 = extractvalue { i32, i32 } %555, 1		; visa id: 705
  %558 = insertelement <2 x i32> undef, i32 %556, i32 0		; visa id: 712
  %559 = insertelement <2 x i32> %558, i32 %557, i32 1		; visa id: 713
  %560 = bitcast <2 x i32> %559 to i64		; visa id: 714
  %561 = shl i64 %560, 1		; visa id: 716
  %562 = add i64 %.in989, %561		; visa id: 717
  %563 = add i64 %562, %209		; visa id: 718
  %564 = inttoptr i64 %563 to i16 addrspace(4)*		; visa id: 719
  %565 = addrspacecast i16 addrspace(4)* %564 to i16 addrspace(1)*		; visa id: 719
  %566 = load i16, i16 addrspace(1)* %565, align 2		; visa id: 720
  %567 = zext i16 %554 to i32		; visa id: 722
  %568 = shl nuw i32 %567, 16, !spirv.Decorations !888		; visa id: 723
  %569 = bitcast i32 %568 to float
  %570 = zext i16 %566 to i32		; visa id: 724
  %571 = shl nuw i32 %570, 16, !spirv.Decorations !888		; visa id: 725
  %572 = bitcast i32 %571 to float
  %573 = fmul reassoc nsz arcp contract float %569, %572, !spirv.Decorations !881
  %574 = fadd reassoc nsz arcp contract float %573, %323, !spirv.Decorations !881		; visa id: 726
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 727

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %549
; BB33 :
  %575 = phi float [ %574, %549 ], [ %323, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %152, label %576, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 728

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
; BB:
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

576:                                              ; preds = %._crit_edge.274
; BB35 :
  %.sroa.64.0.insert.ext243 = zext i32 %332 to i64		; visa id: 730
  %577 = shl nuw nsw i64 %.sroa.64.0.insert.ext243, 1		; visa id: 731
  %578 = add i64 %313, %577		; visa id: 732
  %579 = inttoptr i64 %578 to i16 addrspace(4)*		; visa id: 733
  %580 = addrspacecast i16 addrspace(4)* %579 to i16 addrspace(1)*		; visa id: 733
  %581 = load i16, i16 addrspace(1)* %580, align 2		; visa id: 734
  %582 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %583 = extractvalue { i32, i32 } %582, 0		; visa id: 736
  %584 = extractvalue { i32, i32 } %582, 1		; visa id: 736
  %585 = insertelement <2 x i32> undef, i32 %583, i32 0		; visa id: 743
  %586 = insertelement <2 x i32> %585, i32 %584, i32 1		; visa id: 744
  %587 = bitcast <2 x i32> %586 to i64		; visa id: 745
  %588 = shl i64 %587, 1		; visa id: 747
  %589 = add i64 %.in989, %588		; visa id: 748
  %590 = add i64 %589, %209		; visa id: 749
  %591 = inttoptr i64 %590 to i16 addrspace(4)*		; visa id: 750
  %592 = addrspacecast i16 addrspace(4)* %591 to i16 addrspace(1)*		; visa id: 750
  %593 = load i16, i16 addrspace(1)* %592, align 2		; visa id: 751
  %594 = zext i16 %581 to i32		; visa id: 753
  %595 = shl nuw i32 %594, 16, !spirv.Decorations !888		; visa id: 754
  %596 = bitcast i32 %595 to float
  %597 = zext i16 %593 to i32		; visa id: 755
  %598 = shl nuw i32 %597, 16, !spirv.Decorations !888		; visa id: 756
  %599 = bitcast i32 %598 to float
  %600 = fmul reassoc nsz arcp contract float %596, %599, !spirv.Decorations !881
  %601 = fadd reassoc nsz arcp contract float %600, %322, !spirv.Decorations !881		; visa id: 757
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 758

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %576
; BB36 :
  %602 = phi float [ %601, %576 ], [ %322, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %155, label %603, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 759

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
; BB:
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

603:                                              ; preds = %._crit_edge.1.2
; BB38 :
  %.sroa.64.0.insert.ext248 = zext i32 %332 to i64		; visa id: 761
  %604 = shl nuw nsw i64 %.sroa.64.0.insert.ext248, 1		; visa id: 762
  %605 = add i64 %314, %604		; visa id: 763
  %606 = inttoptr i64 %605 to i16 addrspace(4)*		; visa id: 764
  %607 = addrspacecast i16 addrspace(4)* %606 to i16 addrspace(1)*		; visa id: 764
  %608 = load i16, i16 addrspace(1)* %607, align 2		; visa id: 765
  %609 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %610 = extractvalue { i32, i32 } %609, 0		; visa id: 767
  %611 = extractvalue { i32, i32 } %609, 1		; visa id: 767
  %612 = insertelement <2 x i32> undef, i32 %610, i32 0		; visa id: 774
  %613 = insertelement <2 x i32> %612, i32 %611, i32 1		; visa id: 775
  %614 = bitcast <2 x i32> %613 to i64		; visa id: 776
  %615 = shl i64 %614, 1		; visa id: 778
  %616 = add i64 %.in989, %615		; visa id: 779
  %617 = add i64 %616, %209		; visa id: 780
  %618 = inttoptr i64 %617 to i16 addrspace(4)*		; visa id: 781
  %619 = addrspacecast i16 addrspace(4)* %618 to i16 addrspace(1)*		; visa id: 781
  %620 = load i16, i16 addrspace(1)* %619, align 2		; visa id: 782
  %621 = zext i16 %608 to i32		; visa id: 784
  %622 = shl nuw i32 %621, 16, !spirv.Decorations !888		; visa id: 785
  %623 = bitcast i32 %622 to float
  %624 = zext i16 %620 to i32		; visa id: 786
  %625 = shl nuw i32 %624, 16, !spirv.Decorations !888		; visa id: 787
  %626 = bitcast i32 %625 to float
  %627 = fmul reassoc nsz arcp contract float %623, %626, !spirv.Decorations !881
  %628 = fadd reassoc nsz arcp contract float %627, %321, !spirv.Decorations !881		; visa id: 788
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 789

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %603
; BB39 :
  %629 = phi float [ %628, %603 ], [ %321, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %158, label %630, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 790

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
; BB:
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

630:                                              ; preds = %._crit_edge.2.2
; BB41 :
  %.sroa.64.0.insert.ext253 = zext i32 %332 to i64		; visa id: 792
  %631 = shl nuw nsw i64 %.sroa.64.0.insert.ext253, 1		; visa id: 793
  %632 = add i64 %315, %631		; visa id: 794
  %633 = inttoptr i64 %632 to i16 addrspace(4)*		; visa id: 795
  %634 = addrspacecast i16 addrspace(4)* %633 to i16 addrspace(1)*		; visa id: 795
  %635 = load i16, i16 addrspace(1)* %634, align 2		; visa id: 796
  %636 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %637 = extractvalue { i32, i32 } %636, 0		; visa id: 798
  %638 = extractvalue { i32, i32 } %636, 1		; visa id: 798
  %639 = insertelement <2 x i32> undef, i32 %637, i32 0		; visa id: 805
  %640 = insertelement <2 x i32> %639, i32 %638, i32 1		; visa id: 806
  %641 = bitcast <2 x i32> %640 to i64		; visa id: 807
  %642 = shl i64 %641, 1		; visa id: 809
  %643 = add i64 %.in989, %642		; visa id: 810
  %644 = add i64 %643, %209		; visa id: 811
  %645 = inttoptr i64 %644 to i16 addrspace(4)*		; visa id: 812
  %646 = addrspacecast i16 addrspace(4)* %645 to i16 addrspace(1)*		; visa id: 812
  %647 = load i16, i16 addrspace(1)* %646, align 2		; visa id: 813
  %648 = zext i16 %635 to i32		; visa id: 815
  %649 = shl nuw i32 %648, 16, !spirv.Decorations !888		; visa id: 816
  %650 = bitcast i32 %649 to float
  %651 = zext i16 %647 to i32		; visa id: 817
  %652 = shl nuw i32 %651, 16, !spirv.Decorations !888		; visa id: 818
  %653 = bitcast i32 %652 to float
  %654 = fmul reassoc nsz arcp contract float %650, %653, !spirv.Decorations !881
  %655 = fadd reassoc nsz arcp contract float %654, %320, !spirv.Decorations !881		; visa id: 819
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 820

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %630
; BB42 :
  %656 = phi float [ %655, %630 ], [ %320, %._crit_edge.2.2..preheader.2_crit_edge ]
  br i1 %162, label %657, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 821

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
; BB:
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

657:                                              ; preds = %.preheader.2
; BB44 :
  %.sroa.64.0.insert.ext258 = zext i32 %332 to i64		; visa id: 823
  %658 = shl nuw nsw i64 %.sroa.64.0.insert.ext258, 1		; visa id: 824
  %659 = add i64 %312, %658		; visa id: 825
  %660 = inttoptr i64 %659 to i16 addrspace(4)*		; visa id: 826
  %661 = addrspacecast i16 addrspace(4)* %660 to i16 addrspace(1)*		; visa id: 826
  %662 = load i16, i16 addrspace(1)* %661, align 2		; visa id: 827
  %663 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %664 = extractvalue { i32, i32 } %663, 0		; visa id: 829
  %665 = extractvalue { i32, i32 } %663, 1		; visa id: 829
  %666 = insertelement <2 x i32> undef, i32 %664, i32 0		; visa id: 836
  %667 = insertelement <2 x i32> %666, i32 %665, i32 1		; visa id: 837
  %668 = bitcast <2 x i32> %667 to i64		; visa id: 838
  %669 = shl i64 %668, 1		; visa id: 840
  %670 = add i64 %.in989, %669		; visa id: 841
  %671 = add i64 %670, %211		; visa id: 842
  %672 = inttoptr i64 %671 to i16 addrspace(4)*		; visa id: 843
  %673 = addrspacecast i16 addrspace(4)* %672 to i16 addrspace(1)*		; visa id: 843
  %674 = load i16, i16 addrspace(1)* %673, align 2		; visa id: 844
  %675 = zext i16 %662 to i32		; visa id: 846
  %676 = shl nuw i32 %675, 16, !spirv.Decorations !888		; visa id: 847
  %677 = bitcast i32 %676 to float
  %678 = zext i16 %674 to i32		; visa id: 848
  %679 = shl nuw i32 %678, 16, !spirv.Decorations !888		; visa id: 849
  %680 = bitcast i32 %679 to float
  %681 = fmul reassoc nsz arcp contract float %677, %680, !spirv.Decorations !881
  %682 = fadd reassoc nsz arcp contract float %681, %319, !spirv.Decorations !881		; visa id: 850
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 851

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %657
; BB45 :
  %683 = phi float [ %682, %657 ], [ %319, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %165, label %684, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 852

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
; BB:
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

684:                                              ; preds = %._crit_edge.375
; BB47 :
  %.sroa.64.0.insert.ext263 = zext i32 %332 to i64		; visa id: 854
  %685 = shl nuw nsw i64 %.sroa.64.0.insert.ext263, 1		; visa id: 855
  %686 = add i64 %313, %685		; visa id: 856
  %687 = inttoptr i64 %686 to i16 addrspace(4)*		; visa id: 857
  %688 = addrspacecast i16 addrspace(4)* %687 to i16 addrspace(1)*		; visa id: 857
  %689 = load i16, i16 addrspace(1)* %688, align 2		; visa id: 858
  %690 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %691 = extractvalue { i32, i32 } %690, 0		; visa id: 860
  %692 = extractvalue { i32, i32 } %690, 1		; visa id: 860
  %693 = insertelement <2 x i32> undef, i32 %691, i32 0		; visa id: 867
  %694 = insertelement <2 x i32> %693, i32 %692, i32 1		; visa id: 868
  %695 = bitcast <2 x i32> %694 to i64		; visa id: 869
  %696 = shl i64 %695, 1		; visa id: 871
  %697 = add i64 %.in989, %696		; visa id: 872
  %698 = add i64 %697, %211		; visa id: 873
  %699 = inttoptr i64 %698 to i16 addrspace(4)*		; visa id: 874
  %700 = addrspacecast i16 addrspace(4)* %699 to i16 addrspace(1)*		; visa id: 874
  %701 = load i16, i16 addrspace(1)* %700, align 2		; visa id: 875
  %702 = zext i16 %689 to i32		; visa id: 877
  %703 = shl nuw i32 %702, 16, !spirv.Decorations !888		; visa id: 878
  %704 = bitcast i32 %703 to float
  %705 = zext i16 %701 to i32		; visa id: 879
  %706 = shl nuw i32 %705, 16, !spirv.Decorations !888		; visa id: 880
  %707 = bitcast i32 %706 to float
  %708 = fmul reassoc nsz arcp contract float %704, %707, !spirv.Decorations !881
  %709 = fadd reassoc nsz arcp contract float %708, %318, !spirv.Decorations !881		; visa id: 881
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 882

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %684
; BB48 :
  %710 = phi float [ %709, %684 ], [ %318, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %168, label %711, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 883

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
; BB:
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

711:                                              ; preds = %._crit_edge.1.3
; BB50 :
  %.sroa.64.0.insert.ext268 = zext i32 %332 to i64		; visa id: 885
  %712 = shl nuw nsw i64 %.sroa.64.0.insert.ext268, 1		; visa id: 886
  %713 = add i64 %314, %712		; visa id: 887
  %714 = inttoptr i64 %713 to i16 addrspace(4)*		; visa id: 888
  %715 = addrspacecast i16 addrspace(4)* %714 to i16 addrspace(1)*		; visa id: 888
  %716 = load i16, i16 addrspace(1)* %715, align 2		; visa id: 889
  %717 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %718 = extractvalue { i32, i32 } %717, 0		; visa id: 891
  %719 = extractvalue { i32, i32 } %717, 1		; visa id: 891
  %720 = insertelement <2 x i32> undef, i32 %718, i32 0		; visa id: 898
  %721 = insertelement <2 x i32> %720, i32 %719, i32 1		; visa id: 899
  %722 = bitcast <2 x i32> %721 to i64		; visa id: 900
  %723 = shl i64 %722, 1		; visa id: 902
  %724 = add i64 %.in989, %723		; visa id: 903
  %725 = add i64 %724, %211		; visa id: 904
  %726 = inttoptr i64 %725 to i16 addrspace(4)*		; visa id: 905
  %727 = addrspacecast i16 addrspace(4)* %726 to i16 addrspace(1)*		; visa id: 905
  %728 = load i16, i16 addrspace(1)* %727, align 2		; visa id: 906
  %729 = zext i16 %716 to i32		; visa id: 908
  %730 = shl nuw i32 %729, 16, !spirv.Decorations !888		; visa id: 909
  %731 = bitcast i32 %730 to float
  %732 = zext i16 %728 to i32		; visa id: 910
  %733 = shl nuw i32 %732, 16, !spirv.Decorations !888		; visa id: 911
  %734 = bitcast i32 %733 to float
  %735 = fmul reassoc nsz arcp contract float %731, %734, !spirv.Decorations !881
  %736 = fadd reassoc nsz arcp contract float %735, %317, !spirv.Decorations !881		; visa id: 912
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 913

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %711
; BB51 :
  %737 = phi float [ %736, %711 ], [ %317, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %171, label %738, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 914

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
; BB:
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

738:                                              ; preds = %._crit_edge.2.3
; BB53 :
  %.sroa.64.0.insert.ext273 = zext i32 %332 to i64		; visa id: 916
  %739 = shl nuw nsw i64 %.sroa.64.0.insert.ext273, 1		; visa id: 917
  %740 = add i64 %315, %739		; visa id: 918
  %741 = inttoptr i64 %740 to i16 addrspace(4)*		; visa id: 919
  %742 = addrspacecast i16 addrspace(4)* %741 to i16 addrspace(1)*		; visa id: 919
  %743 = load i16, i16 addrspace(1)* %742, align 2		; visa id: 920
  %744 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %745 = extractvalue { i32, i32 } %744, 0		; visa id: 922
  %746 = extractvalue { i32, i32 } %744, 1		; visa id: 922
  %747 = insertelement <2 x i32> undef, i32 %745, i32 0		; visa id: 929
  %748 = insertelement <2 x i32> %747, i32 %746, i32 1		; visa id: 930
  %749 = bitcast <2 x i32> %748 to i64		; visa id: 931
  %750 = shl i64 %749, 1		; visa id: 933
  %751 = add i64 %.in989, %750		; visa id: 934
  %752 = add i64 %751, %211		; visa id: 935
  %753 = inttoptr i64 %752 to i16 addrspace(4)*		; visa id: 936
  %754 = addrspacecast i16 addrspace(4)* %753 to i16 addrspace(1)*		; visa id: 936
  %755 = load i16, i16 addrspace(1)* %754, align 2		; visa id: 937
  %756 = zext i16 %743 to i32		; visa id: 939
  %757 = shl nuw i32 %756, 16, !spirv.Decorations !888		; visa id: 940
  %758 = bitcast i32 %757 to float
  %759 = zext i16 %755 to i32		; visa id: 941
  %760 = shl nuw i32 %759, 16, !spirv.Decorations !888		; visa id: 942
  %761 = bitcast i32 %760 to float
  %762 = fmul reassoc nsz arcp contract float %758, %761, !spirv.Decorations !881
  %763 = fadd reassoc nsz arcp contract float %762, %316, !spirv.Decorations !881		; visa id: 943
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 944

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %738
; BB54 :
  %764 = phi float [ %763, %738 ], [ %316, %._crit_edge.2.3..preheader.3_crit_edge ]
  %765 = add nuw nsw i32 %332, 1, !spirv.Decorations !890		; visa id: 945
  %766 = icmp slt i32 %765, %const_reg_dword2		; visa id: 946
  br i1 %766, label %.preheader.3..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 947

.preheader.3..preheader.preheader_crit_edge:      ; preds = %.preheader.3
; BB:
  br label %.preheader.preheader, !stats.blockFrequency.digits !892, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.3
; BB:
  %.lcssa1021 = phi float [ %764, %.preheader.3 ]
  %.lcssa1020 = phi float [ %737, %.preheader.3 ]
  %.lcssa1019 = phi float [ %710, %.preheader.3 ]
  %.lcssa1018 = phi float [ %683, %.preheader.3 ]
  %.lcssa1017 = phi float [ %656, %.preheader.3 ]
  %.lcssa1016 = phi float [ %629, %.preheader.3 ]
  %.lcssa1015 = phi float [ %602, %.preheader.3 ]
  %.lcssa1014 = phi float [ %575, %.preheader.3 ]
  %.lcssa1013 = phi float [ %548, %.preheader.3 ]
  %.lcssa1012 = phi float [ %521, %.preheader.3 ]
  %.lcssa1011 = phi float [ %494, %.preheader.3 ]
  %.lcssa1010 = phi float [ %467, %.preheader.3 ]
  %.lcssa1009 = phi float [ %440, %.preheader.3 ]
  %.lcssa1008 = phi float [ %413, %.preheader.3 ]
  %.lcssa1007 = phi float [ %386, %.preheader.3 ]
  %.lcssa = phi float [ %359, %.preheader.3 ]
  br label %.preheader1.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
; BB57 :
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1021, %.preheader1.preheader.loopexit ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1017, %.preheader1.preheader.loopexit ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1013, %.preheader1.preheader.loopexit ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1009, %.preheader1.preheader.loopexit ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1020, %.preheader1.preheader.loopexit ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1016, %.preheader1.preheader.loopexit ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1012, %.preheader1.preheader.loopexit ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1008, %.preheader1.preheader.loopexit ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1019, %.preheader1.preheader.loopexit ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1015, %.preheader1.preheader.loopexit ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1011, %.preheader1.preheader.loopexit ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1007, %.preheader1.preheader.loopexit ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1018, %.preheader1.preheader.loopexit ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1014, %.preheader1.preheader.loopexit ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa1010, %.preheader1.preheader.loopexit ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.lcssa, %.preheader1.preheader.loopexit ]
  br i1 %120, label %767, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 949

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
; BB:
  br label %._crit_edge70, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

767:                                              ; preds = %.preheader1.preheader
; BB59 :
  %768 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !881		; visa id: 951
  br i1 %81, label %773, label %769, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 952

769:                                              ; preds = %767
; BB60 :
  %770 = add i64 %.in, %219		; visa id: 954
  %771 = inttoptr i64 %770 to float addrspace(4)*		; visa id: 955
  %772 = addrspacecast float addrspace(4)* %771 to float addrspace(1)*		; visa id: 955
  store float %768, float addrspace(1)* %772, align 4		; visa id: 956
  br label %._crit_edge70, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 957

773:                                              ; preds = %767
; BB61 :
  %774 = add i64 %.in988, %226		; visa id: 959
  %775 = add i64 %774, %227		; visa id: 960
  %776 = inttoptr i64 %775 to float addrspace(4)*		; visa id: 961
  %777 = addrspacecast float addrspace(4)* %776 to float addrspace(1)*		; visa id: 961
  %778 = load float, float addrspace(1)* %777, align 4		; visa id: 962
  %779 = fmul reassoc nsz arcp contract float %778, %4, !spirv.Decorations !881		; visa id: 963
  %780 = fadd reassoc nsz arcp contract float %768, %779, !spirv.Decorations !881		; visa id: 964
  %781 = add i64 %.in, %219		; visa id: 965
  %782 = inttoptr i64 %781 to float addrspace(4)*		; visa id: 966
  %783 = addrspacecast float addrspace(4)* %782 to float addrspace(1)*		; visa id: 966
  store float %780, float addrspace(1)* %783, align 4		; visa id: 967
  br label %._crit_edge70, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 968

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %769, %773
; BB62 :
  br i1 %124, label %784, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 969

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
; BB:
  br label %._crit_edge70.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

784:                                              ; preds = %._crit_edge70
; BB64 :
  %785 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !881		; visa id: 971
  br i1 %81, label %790, label %786, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 972

786:                                              ; preds = %784
; BB65 :
  %787 = add i64 %.in, %235		; visa id: 974
  %788 = inttoptr i64 %787 to float addrspace(4)*		; visa id: 975
  %789 = addrspacecast float addrspace(4)* %788 to float addrspace(1)*		; visa id: 975
  store float %785, float addrspace(1)* %789, align 4		; visa id: 976
  br label %._crit_edge70.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 977

790:                                              ; preds = %784
; BB66 :
  %791 = add i64 %.in988, %242		; visa id: 979
  %792 = add i64 %791, %227		; visa id: 980
  %793 = inttoptr i64 %792 to float addrspace(4)*		; visa id: 981
  %794 = addrspacecast float addrspace(4)* %793 to float addrspace(1)*		; visa id: 981
  %795 = load float, float addrspace(1)* %794, align 4		; visa id: 982
  %796 = fmul reassoc nsz arcp contract float %795, %4, !spirv.Decorations !881		; visa id: 983
  %797 = fadd reassoc nsz arcp contract float %785, %796, !spirv.Decorations !881		; visa id: 984
  %798 = add i64 %.in, %235		; visa id: 985
  %799 = inttoptr i64 %798 to float addrspace(4)*		; visa id: 986
  %800 = addrspacecast float addrspace(4)* %799 to float addrspace(1)*		; visa id: 986
  store float %797, float addrspace(1)* %800, align 4		; visa id: 987
  br label %._crit_edge70.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 988

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %790, %786
; BB67 :
  br i1 %128, label %801, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 989

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
; BB:
  br label %._crit_edge70.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

801:                                              ; preds = %._crit_edge70.1
; BB69 :
  %802 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !881		; visa id: 991
  br i1 %81, label %807, label %803, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 992

803:                                              ; preds = %801
; BB70 :
  %804 = add i64 %.in, %250		; visa id: 994
  %805 = inttoptr i64 %804 to float addrspace(4)*		; visa id: 995
  %806 = addrspacecast float addrspace(4)* %805 to float addrspace(1)*		; visa id: 995
  store float %802, float addrspace(1)* %806, align 4		; visa id: 996
  br label %._crit_edge70.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 997

807:                                              ; preds = %801
; BB71 :
  %808 = add i64 %.in988, %257		; visa id: 999
  %809 = add i64 %808, %227		; visa id: 1000
  %810 = inttoptr i64 %809 to float addrspace(4)*		; visa id: 1001
  %811 = addrspacecast float addrspace(4)* %810 to float addrspace(1)*		; visa id: 1001
  %812 = load float, float addrspace(1)* %811, align 4		; visa id: 1002
  %813 = fmul reassoc nsz arcp contract float %812, %4, !spirv.Decorations !881		; visa id: 1003
  %814 = fadd reassoc nsz arcp contract float %802, %813, !spirv.Decorations !881		; visa id: 1004
  %815 = add i64 %.in, %250		; visa id: 1005
  %816 = inttoptr i64 %815 to float addrspace(4)*		; visa id: 1006
  %817 = addrspacecast float addrspace(4)* %816 to float addrspace(1)*		; visa id: 1006
  store float %814, float addrspace(1)* %817, align 4		; visa id: 1007
  br label %._crit_edge70.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1008

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %807, %803
; BB72 :
  br i1 %132, label %818, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1009

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
; BB:
  br label %.preheader1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

818:                                              ; preds = %._crit_edge70.2
; BB74 :
  %819 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !881		; visa id: 1011
  br i1 %81, label %824, label %820, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1012

820:                                              ; preds = %818
; BB75 :
  %821 = add i64 %.in, %265		; visa id: 1014
  %822 = inttoptr i64 %821 to float addrspace(4)*		; visa id: 1015
  %823 = addrspacecast float addrspace(4)* %822 to float addrspace(1)*		; visa id: 1015
  store float %819, float addrspace(1)* %823, align 4		; visa id: 1016
  br label %.preheader1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1017

824:                                              ; preds = %818
; BB76 :
  %825 = add i64 %.in988, %272		; visa id: 1019
  %826 = add i64 %825, %227		; visa id: 1020
  %827 = inttoptr i64 %826 to float addrspace(4)*		; visa id: 1021
  %828 = addrspacecast float addrspace(4)* %827 to float addrspace(1)*		; visa id: 1021
  %829 = load float, float addrspace(1)* %828, align 4		; visa id: 1022
  %830 = fmul reassoc nsz arcp contract float %829, %4, !spirv.Decorations !881		; visa id: 1023
  %831 = fadd reassoc nsz arcp contract float %819, %830, !spirv.Decorations !881		; visa id: 1024
  %832 = add i64 %.in, %265		; visa id: 1025
  %833 = inttoptr i64 %832 to float addrspace(4)*		; visa id: 1026
  %834 = addrspacecast float addrspace(4)* %833 to float addrspace(1)*		; visa id: 1026
  store float %831, float addrspace(1)* %834, align 4		; visa id: 1027
  br label %.preheader1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1028

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %824, %820
; BB77 :
  br i1 %136, label %835, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1029

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
; BB:
  br label %._crit_edge70.176, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

835:                                              ; preds = %.preheader1
; BB79 :
  %836 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !881		; visa id: 1031
  br i1 %81, label %841, label %837, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1032

837:                                              ; preds = %835
; BB80 :
  %838 = add i64 %.in, %274		; visa id: 1034
  %839 = inttoptr i64 %838 to float addrspace(4)*		; visa id: 1035
  %840 = addrspacecast float addrspace(4)* %839 to float addrspace(1)*		; visa id: 1035
  store float %836, float addrspace(1)* %840, align 4		; visa id: 1036
  br label %._crit_edge70.176, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1037

841:                                              ; preds = %835
; BB81 :
  %842 = add i64 %.in988, %226		; visa id: 1039
  %843 = add i64 %842, %275		; visa id: 1040
  %844 = inttoptr i64 %843 to float addrspace(4)*		; visa id: 1041
  %845 = addrspacecast float addrspace(4)* %844 to float addrspace(1)*		; visa id: 1041
  %846 = load float, float addrspace(1)* %845, align 4		; visa id: 1042
  %847 = fmul reassoc nsz arcp contract float %846, %4, !spirv.Decorations !881		; visa id: 1043
  %848 = fadd reassoc nsz arcp contract float %836, %847, !spirv.Decorations !881		; visa id: 1044
  %849 = add i64 %.in, %274		; visa id: 1045
  %850 = inttoptr i64 %849 to float addrspace(4)*		; visa id: 1046
  %851 = addrspacecast float addrspace(4)* %850 to float addrspace(1)*		; visa id: 1046
  store float %848, float addrspace(1)* %851, align 4		; visa id: 1047
  br label %._crit_edge70.176, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1048

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %841, %837
; BB82 :
  br i1 %139, label %852, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1049

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
; BB:
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

852:                                              ; preds = %._crit_edge70.176
; BB84 :
  %853 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !881		; visa id: 1051
  br i1 %81, label %858, label %854, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1052

854:                                              ; preds = %852
; BB85 :
  %855 = add i64 %.in, %277		; visa id: 1054
  %856 = inttoptr i64 %855 to float addrspace(4)*		; visa id: 1055
  %857 = addrspacecast float addrspace(4)* %856 to float addrspace(1)*		; visa id: 1055
  store float %853, float addrspace(1)* %857, align 4		; visa id: 1056
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1057

858:                                              ; preds = %852
; BB86 :
  %859 = add i64 %.in988, %242		; visa id: 1059
  %860 = add i64 %859, %275		; visa id: 1060
  %861 = inttoptr i64 %860 to float addrspace(4)*		; visa id: 1061
  %862 = addrspacecast float addrspace(4)* %861 to float addrspace(1)*		; visa id: 1061
  %863 = load float, float addrspace(1)* %862, align 4		; visa id: 1062
  %864 = fmul reassoc nsz arcp contract float %863, %4, !spirv.Decorations !881		; visa id: 1063
  %865 = fadd reassoc nsz arcp contract float %853, %864, !spirv.Decorations !881		; visa id: 1064
  %866 = add i64 %.in, %277		; visa id: 1065
  %867 = inttoptr i64 %866 to float addrspace(4)*		; visa id: 1066
  %868 = addrspacecast float addrspace(4)* %867 to float addrspace(1)*		; visa id: 1066
  store float %865, float addrspace(1)* %868, align 4		; visa id: 1067
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1068

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %858, %854
; BB87 :
  br i1 %142, label %869, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1069

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
; BB:
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

869:                                              ; preds = %._crit_edge70.1.1
; BB89 :
  %870 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !881		; visa id: 1071
  br i1 %81, label %875, label %871, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1072

871:                                              ; preds = %869
; BB90 :
  %872 = add i64 %.in, %279		; visa id: 1074
  %873 = inttoptr i64 %872 to float addrspace(4)*		; visa id: 1075
  %874 = addrspacecast float addrspace(4)* %873 to float addrspace(1)*		; visa id: 1075
  store float %870, float addrspace(1)* %874, align 4		; visa id: 1076
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1077

875:                                              ; preds = %869
; BB91 :
  %876 = add i64 %.in988, %257		; visa id: 1079
  %877 = add i64 %876, %275		; visa id: 1080
  %878 = inttoptr i64 %877 to float addrspace(4)*		; visa id: 1081
  %879 = addrspacecast float addrspace(4)* %878 to float addrspace(1)*		; visa id: 1081
  %880 = load float, float addrspace(1)* %879, align 4		; visa id: 1082
  %881 = fmul reassoc nsz arcp contract float %880, %4, !spirv.Decorations !881		; visa id: 1083
  %882 = fadd reassoc nsz arcp contract float %870, %881, !spirv.Decorations !881		; visa id: 1084
  %883 = add i64 %.in, %279		; visa id: 1085
  %884 = inttoptr i64 %883 to float addrspace(4)*		; visa id: 1086
  %885 = addrspacecast float addrspace(4)* %884 to float addrspace(1)*		; visa id: 1086
  store float %882, float addrspace(1)* %885, align 4		; visa id: 1087
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1088

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %875, %871
; BB92 :
  br i1 %145, label %886, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1089

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
; BB:
  br label %.preheader1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

886:                                              ; preds = %._crit_edge70.2.1
; BB94 :
  %887 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !881		; visa id: 1091
  br i1 %81, label %892, label %888, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1092

888:                                              ; preds = %886
; BB95 :
  %889 = add i64 %.in, %281		; visa id: 1094
  %890 = inttoptr i64 %889 to float addrspace(4)*		; visa id: 1095
  %891 = addrspacecast float addrspace(4)* %890 to float addrspace(1)*		; visa id: 1095
  store float %887, float addrspace(1)* %891, align 4		; visa id: 1096
  br label %.preheader1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1097

892:                                              ; preds = %886
; BB96 :
  %893 = add i64 %.in988, %272		; visa id: 1099
  %894 = add i64 %893, %275		; visa id: 1100
  %895 = inttoptr i64 %894 to float addrspace(4)*		; visa id: 1101
  %896 = addrspacecast float addrspace(4)* %895 to float addrspace(1)*		; visa id: 1101
  %897 = load float, float addrspace(1)* %896, align 4		; visa id: 1102
  %898 = fmul reassoc nsz arcp contract float %897, %4, !spirv.Decorations !881		; visa id: 1103
  %899 = fadd reassoc nsz arcp contract float %887, %898, !spirv.Decorations !881		; visa id: 1104
  %900 = add i64 %.in, %281		; visa id: 1105
  %901 = inttoptr i64 %900 to float addrspace(4)*		; visa id: 1106
  %902 = addrspacecast float addrspace(4)* %901 to float addrspace(1)*		; visa id: 1106
  store float %899, float addrspace(1)* %902, align 4		; visa id: 1107
  br label %.preheader1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1108

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %892, %888
; BB97 :
  br i1 %149, label %903, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1109

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
; BB:
  br label %._crit_edge70.277, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

903:                                              ; preds = %.preheader1.1
; BB99 :
  %904 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !881		; visa id: 1111
  br i1 %81, label %909, label %905, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1112

905:                                              ; preds = %903
; BB100 :
  %906 = add i64 %.in, %283		; visa id: 1114
  %907 = inttoptr i64 %906 to float addrspace(4)*		; visa id: 1115
  %908 = addrspacecast float addrspace(4)* %907 to float addrspace(1)*		; visa id: 1115
  store float %904, float addrspace(1)* %908, align 4		; visa id: 1116
  br label %._crit_edge70.277, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1117

909:                                              ; preds = %903
; BB101 :
  %910 = add i64 %.in988, %226		; visa id: 1119
  %911 = add i64 %910, %284		; visa id: 1120
  %912 = inttoptr i64 %911 to float addrspace(4)*		; visa id: 1121
  %913 = addrspacecast float addrspace(4)* %912 to float addrspace(1)*		; visa id: 1121
  %914 = load float, float addrspace(1)* %913, align 4		; visa id: 1122
  %915 = fmul reassoc nsz arcp contract float %914, %4, !spirv.Decorations !881		; visa id: 1123
  %916 = fadd reassoc nsz arcp contract float %904, %915, !spirv.Decorations !881		; visa id: 1124
  %917 = add i64 %.in, %283		; visa id: 1125
  %918 = inttoptr i64 %917 to float addrspace(4)*		; visa id: 1126
  %919 = addrspacecast float addrspace(4)* %918 to float addrspace(1)*		; visa id: 1126
  store float %916, float addrspace(1)* %919, align 4		; visa id: 1127
  br label %._crit_edge70.277, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1128

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %909, %905
; BB102 :
  br i1 %152, label %920, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1129

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
; BB:
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

920:                                              ; preds = %._crit_edge70.277
; BB104 :
  %921 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !881		; visa id: 1131
  br i1 %81, label %926, label %922, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1132

922:                                              ; preds = %920
; BB105 :
  %923 = add i64 %.in, %286		; visa id: 1134
  %924 = inttoptr i64 %923 to float addrspace(4)*		; visa id: 1135
  %925 = addrspacecast float addrspace(4)* %924 to float addrspace(1)*		; visa id: 1135
  store float %921, float addrspace(1)* %925, align 4		; visa id: 1136
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1137

926:                                              ; preds = %920
; BB106 :
  %927 = add i64 %.in988, %242		; visa id: 1139
  %928 = add i64 %927, %284		; visa id: 1140
  %929 = inttoptr i64 %928 to float addrspace(4)*		; visa id: 1141
  %930 = addrspacecast float addrspace(4)* %929 to float addrspace(1)*		; visa id: 1141
  %931 = load float, float addrspace(1)* %930, align 4		; visa id: 1142
  %932 = fmul reassoc nsz arcp contract float %931, %4, !spirv.Decorations !881		; visa id: 1143
  %933 = fadd reassoc nsz arcp contract float %921, %932, !spirv.Decorations !881		; visa id: 1144
  %934 = add i64 %.in, %286		; visa id: 1145
  %935 = inttoptr i64 %934 to float addrspace(4)*		; visa id: 1146
  %936 = addrspacecast float addrspace(4)* %935 to float addrspace(1)*		; visa id: 1146
  store float %933, float addrspace(1)* %936, align 4		; visa id: 1147
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1148

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %926, %922
; BB107 :
  br i1 %155, label %937, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1149

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
; BB:
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

937:                                              ; preds = %._crit_edge70.1.2
; BB109 :
  %938 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !881		; visa id: 1151
  br i1 %81, label %943, label %939, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1152

939:                                              ; preds = %937
; BB110 :
  %940 = add i64 %.in, %288		; visa id: 1154
  %941 = inttoptr i64 %940 to float addrspace(4)*		; visa id: 1155
  %942 = addrspacecast float addrspace(4)* %941 to float addrspace(1)*		; visa id: 1155
  store float %938, float addrspace(1)* %942, align 4		; visa id: 1156
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1157

943:                                              ; preds = %937
; BB111 :
  %944 = add i64 %.in988, %257		; visa id: 1159
  %945 = add i64 %944, %284		; visa id: 1160
  %946 = inttoptr i64 %945 to float addrspace(4)*		; visa id: 1161
  %947 = addrspacecast float addrspace(4)* %946 to float addrspace(1)*		; visa id: 1161
  %948 = load float, float addrspace(1)* %947, align 4		; visa id: 1162
  %949 = fmul reassoc nsz arcp contract float %948, %4, !spirv.Decorations !881		; visa id: 1163
  %950 = fadd reassoc nsz arcp contract float %938, %949, !spirv.Decorations !881		; visa id: 1164
  %951 = add i64 %.in, %288		; visa id: 1165
  %952 = inttoptr i64 %951 to float addrspace(4)*		; visa id: 1166
  %953 = addrspacecast float addrspace(4)* %952 to float addrspace(1)*		; visa id: 1166
  store float %950, float addrspace(1)* %953, align 4		; visa id: 1167
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1168

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %943, %939
; BB112 :
  br i1 %158, label %954, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1169

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
; BB:
  br label %.preheader1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

954:                                              ; preds = %._crit_edge70.2.2
; BB114 :
  %955 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !881		; visa id: 1171
  br i1 %81, label %960, label %956, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1172

956:                                              ; preds = %954
; BB115 :
  %957 = add i64 %.in, %290		; visa id: 1174
  %958 = inttoptr i64 %957 to float addrspace(4)*		; visa id: 1175
  %959 = addrspacecast float addrspace(4)* %958 to float addrspace(1)*		; visa id: 1175
  store float %955, float addrspace(1)* %959, align 4		; visa id: 1176
  br label %.preheader1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1177

960:                                              ; preds = %954
; BB116 :
  %961 = add i64 %.in988, %272		; visa id: 1179
  %962 = add i64 %961, %284		; visa id: 1180
  %963 = inttoptr i64 %962 to float addrspace(4)*		; visa id: 1181
  %964 = addrspacecast float addrspace(4)* %963 to float addrspace(1)*		; visa id: 1181
  %965 = load float, float addrspace(1)* %964, align 4		; visa id: 1182
  %966 = fmul reassoc nsz arcp contract float %965, %4, !spirv.Decorations !881		; visa id: 1183
  %967 = fadd reassoc nsz arcp contract float %955, %966, !spirv.Decorations !881		; visa id: 1184
  %968 = add i64 %.in, %290		; visa id: 1185
  %969 = inttoptr i64 %968 to float addrspace(4)*		; visa id: 1186
  %970 = addrspacecast float addrspace(4)* %969 to float addrspace(1)*		; visa id: 1186
  store float %967, float addrspace(1)* %970, align 4		; visa id: 1187
  br label %.preheader1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1188

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %960, %956
; BB117 :
  br i1 %162, label %971, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1189

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
; BB:
  br label %._crit_edge70.378, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

971:                                              ; preds = %.preheader1.2
; BB119 :
  %972 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !881		; visa id: 1191
  br i1 %81, label %977, label %973, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1192

973:                                              ; preds = %971
; BB120 :
  %974 = add i64 %.in, %292		; visa id: 1194
  %975 = inttoptr i64 %974 to float addrspace(4)*		; visa id: 1195
  %976 = addrspacecast float addrspace(4)* %975 to float addrspace(1)*		; visa id: 1195
  store float %972, float addrspace(1)* %976, align 4		; visa id: 1196
  br label %._crit_edge70.378, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1197

977:                                              ; preds = %971
; BB121 :
  %978 = add i64 %.in988, %226		; visa id: 1199
  %979 = add i64 %978, %293		; visa id: 1200
  %980 = inttoptr i64 %979 to float addrspace(4)*		; visa id: 1201
  %981 = addrspacecast float addrspace(4)* %980 to float addrspace(1)*		; visa id: 1201
  %982 = load float, float addrspace(1)* %981, align 4		; visa id: 1202
  %983 = fmul reassoc nsz arcp contract float %982, %4, !spirv.Decorations !881		; visa id: 1203
  %984 = fadd reassoc nsz arcp contract float %972, %983, !spirv.Decorations !881		; visa id: 1204
  %985 = add i64 %.in, %292		; visa id: 1205
  %986 = inttoptr i64 %985 to float addrspace(4)*		; visa id: 1206
  %987 = addrspacecast float addrspace(4)* %986 to float addrspace(1)*		; visa id: 1206
  store float %984, float addrspace(1)* %987, align 4		; visa id: 1207
  br label %._crit_edge70.378, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1208

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %977, %973
; BB122 :
  br i1 %165, label %988, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1209

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
; BB:
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

988:                                              ; preds = %._crit_edge70.378
; BB124 :
  %989 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !881		; visa id: 1211
  br i1 %81, label %994, label %990, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1212

990:                                              ; preds = %988
; BB125 :
  %991 = add i64 %.in, %295		; visa id: 1214
  %992 = inttoptr i64 %991 to float addrspace(4)*		; visa id: 1215
  %993 = addrspacecast float addrspace(4)* %992 to float addrspace(1)*		; visa id: 1215
  store float %989, float addrspace(1)* %993, align 4		; visa id: 1216
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1217

994:                                              ; preds = %988
; BB126 :
  %995 = add i64 %.in988, %242		; visa id: 1219
  %996 = add i64 %995, %293		; visa id: 1220
  %997 = inttoptr i64 %996 to float addrspace(4)*		; visa id: 1221
  %998 = addrspacecast float addrspace(4)* %997 to float addrspace(1)*		; visa id: 1221
  %999 = load float, float addrspace(1)* %998, align 4		; visa id: 1222
  %1000 = fmul reassoc nsz arcp contract float %999, %4, !spirv.Decorations !881		; visa id: 1223
  %1001 = fadd reassoc nsz arcp contract float %989, %1000, !spirv.Decorations !881		; visa id: 1224
  %1002 = add i64 %.in, %295		; visa id: 1225
  %1003 = inttoptr i64 %1002 to float addrspace(4)*		; visa id: 1226
  %1004 = addrspacecast float addrspace(4)* %1003 to float addrspace(1)*		; visa id: 1226
  store float %1001, float addrspace(1)* %1004, align 4		; visa id: 1227
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1228

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %994, %990
; BB127 :
  br i1 %168, label %1005, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1229

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
; BB:
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1005:                                             ; preds = %._crit_edge70.1.3
; BB129 :
  %1006 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !881		; visa id: 1231
  br i1 %81, label %1011, label %1007, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1232

1007:                                             ; preds = %1005
; BB130 :
  %1008 = add i64 %.in, %297		; visa id: 1234
  %1009 = inttoptr i64 %1008 to float addrspace(4)*		; visa id: 1235
  %1010 = addrspacecast float addrspace(4)* %1009 to float addrspace(1)*		; visa id: 1235
  store float %1006, float addrspace(1)* %1010, align 4		; visa id: 1236
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1237

1011:                                             ; preds = %1005
; BB131 :
  %1012 = add i64 %.in988, %257		; visa id: 1239
  %1013 = add i64 %1012, %293		; visa id: 1240
  %1014 = inttoptr i64 %1013 to float addrspace(4)*		; visa id: 1241
  %1015 = addrspacecast float addrspace(4)* %1014 to float addrspace(1)*		; visa id: 1241
  %1016 = load float, float addrspace(1)* %1015, align 4		; visa id: 1242
  %1017 = fmul reassoc nsz arcp contract float %1016, %4, !spirv.Decorations !881		; visa id: 1243
  %1018 = fadd reassoc nsz arcp contract float %1006, %1017, !spirv.Decorations !881		; visa id: 1244
  %1019 = add i64 %.in, %297		; visa id: 1245
  %1020 = inttoptr i64 %1019 to float addrspace(4)*		; visa id: 1246
  %1021 = addrspacecast float addrspace(4)* %1020 to float addrspace(1)*		; visa id: 1246
  store float %1018, float addrspace(1)* %1021, align 4		; visa id: 1247
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1248

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %1011, %1007
; BB132 :
  br i1 %171, label %1022, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1249

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
; BB:
  br label %.preheader1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1022:                                             ; preds = %._crit_edge70.2.3
; BB134 :
  %1023 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !881		; visa id: 1251
  br i1 %81, label %1028, label %1024, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 1252

1024:                                             ; preds = %1022
; BB135 :
  %1025 = add i64 %.in, %299		; visa id: 1254
  %1026 = inttoptr i64 %1025 to float addrspace(4)*		; visa id: 1255
  %1027 = addrspacecast float addrspace(4)* %1026 to float addrspace(1)*		; visa id: 1255
  store float %1023, float addrspace(1)* %1027, align 4		; visa id: 1256
  br label %.preheader1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 1257

1028:                                             ; preds = %1022
; BB136 :
  %1029 = add i64 %.in988, %272		; visa id: 1259
  %1030 = add i64 %1029, %293		; visa id: 1260
  %1031 = inttoptr i64 %1030 to float addrspace(4)*		; visa id: 1261
  %1032 = addrspacecast float addrspace(4)* %1031 to float addrspace(1)*		; visa id: 1261
  %1033 = load float, float addrspace(1)* %1032, align 4		; visa id: 1262
  %1034 = fmul reassoc nsz arcp contract float %1033, %4, !spirv.Decorations !881		; visa id: 1263
  %1035 = fadd reassoc nsz arcp contract float %1023, %1034, !spirv.Decorations !881		; visa id: 1264
  %1036 = add i64 %.in, %299		; visa id: 1265
  %1037 = inttoptr i64 %1036 to float addrspace(4)*		; visa id: 1266
  %1038 = addrspacecast float addrspace(4)* %1037 to float addrspace(1)*		; visa id: 1266
  store float %1035, float addrspace(1)* %1038, align 4		; visa id: 1267
  br label %.preheader1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 1268

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %1028, %1024
; BB137 :
  %1039 = add i32 %311, %52		; visa id: 1269
  %1040 = icmp slt i32 %1039, %8		; visa id: 1270
  br i1 %1040, label %.preheader1.3..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 1271

._crit_edge72.loopexit:                           ; preds = %.preheader1.3
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.3..preheader2.preheader_crit_edge:    ; preds = %.preheader1.3
; BB139 :
  %1041 = add i64 %.in990, %300		; visa id: 1273
  %1042 = add i64 %.in989, %301		; visa id: 1274
  %1043 = add i64 %.in988, %309		; visa id: 1275
  %1044 = add i64 %.in, %310		; visa id: 1276
  br label %.preheader2.preheader, !stats.blockFrequency.digits !896, !stats.blockFrequency.scale !879		; visa id: 1277

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
; BB140 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 1279
}

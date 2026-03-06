; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd16_entry_0013.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
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
  %43 = bitcast i64 %const_reg_qword7 to <2 x i32>		; visa id: 39
  %44 = extractelement <2 x i32> %43, i32 0		; visa id: 40
  %45 = extractelement <2 x i32> %43, i32 1		; visa id: 40
  %46 = bitcast i64 %const_reg_qword9 to <2 x i32>		; visa id: 40
  %47 = extractelement <2 x i32> %46, i32 0		; visa id: 41
  %48 = extractelement <2 x i32> %46, i32 1		; visa id: 41
  %49 = extractelement <3 x i32> %numWorkGroups, i32 2		; visa id: 41
  %50 = extractelement <3 x i32> %localSize, i32 0		; visa id: 41
  %51 = extractelement <3 x i32> %localSize, i32 1		; visa id: 41
  %52 = extractelement <8 x i32> %r0, i32 1		; visa id: 41
  %53 = extractelement <8 x i32> %r0, i32 6		; visa id: 42
  %54 = mul i32 %52, %50		; visa id: 43
  %55 = zext i16 %localIdX to i32		; visa id: 44
  %56 = add i32 %54, %55		; visa id: 45
  %57 = shl i32 %56, 2		; visa id: 46
  %58 = mul i32 %53, %51		; visa id: 47
  %59 = zext i16 %localIdY to i32		; visa id: 48
  %60 = add i32 %58, %59		; visa id: 49
  %61 = shl i32 %60, 4		; visa id: 50
  %62 = insertelement <2 x i32> undef, i32 %28, i32 0		; visa id: 51
  %63 = insertelement <2 x i32> %62, i32 %29, i32 1		; visa id: 52
  %64 = bitcast <2 x i32> %63 to i64		; visa id: 53
  %65 = shl i64 %64, 1		; visa id: 55
  %66 = add i64 %65, %const_reg_qword		; visa id: 56
  %67 = insertelement <2 x i32> undef, i32 %31, i32 0		; visa id: 57
  %68 = insertelement <2 x i32> %67, i32 %32, i32 1		; visa id: 58
  %69 = bitcast <2 x i32> %68 to i64		; visa id: 59
  %70 = shl i64 %69, 1		; visa id: 61
  %71 = add i64 %70, %const_reg_qword4		; visa id: 62
  %72 = insertelement <2 x i32> undef, i32 %34, i32 0		; visa id: 63
  %73 = insertelement <2 x i32> %72, i32 %35, i32 1		; visa id: 64
  %74 = bitcast <2 x i32> %73 to i64		; visa id: 65
  %.op = shl i64 %74, 2		; visa id: 67
  %75 = bitcast i64 %.op to <2 x i32>		; visa id: 68
  %76 = extractelement <2 x i32> %75, i32 0		; visa id: 69
  %77 = extractelement <2 x i32> %75, i32 1		; visa id: 69
  %78 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !881		; visa id: 69
  %79 = select i1 %78, i32 %76, i32 0		; visa id: 70
  %80 = select i1 %78, i32 %77, i32 0		; visa id: 71
  %81 = insertelement <2 x i32> undef, i32 %79, i32 0		; visa id: 72
  %82 = insertelement <2 x i32> %81, i32 %80, i32 1		; visa id: 73
  %83 = bitcast <2 x i32> %82 to i64		; visa id: 74
  %84 = add i64 %83, %const_reg_qword6		; visa id: 76
  %85 = insertelement <2 x i32> undef, i32 %37, i32 0		; visa id: 77
  %86 = insertelement <2 x i32> %85, i32 %38, i32 1		; visa id: 78
  %87 = bitcast <2 x i32> %86 to i64		; visa id: 79
  %88 = shl i64 %87, 2		; visa id: 81
  %89 = add i64 %88, %const_reg_qword8		; visa id: 82
  %90 = icmp sgt i32 %const_reg_dword2, 0		; visa id: 83
  %91 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %15, i32 %16)
  %92 = extractvalue { i32, i32 } %91, 0		; visa id: 84
  %93 = extractvalue { i32, i32 } %91, 1		; visa id: 84
  %94 = insertelement <2 x i32> undef, i32 %92, i32 0		; visa id: 91
  %95 = insertelement <2 x i32> %94, i32 %93, i32 1		; visa id: 92
  %96 = bitcast <2 x i32> %95 to i64		; visa id: 93
  %97 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %18, i32 %19)
  %98 = extractvalue { i32, i32 } %97, 0		; visa id: 95
  %99 = extractvalue { i32, i32 } %97, 1		; visa id: 95
  %100 = insertelement <2 x i32> undef, i32 %98, i32 0		; visa id: 102
  %101 = insertelement <2 x i32> %100, i32 %99, i32 1		; visa id: 103
  %102 = bitcast <2 x i32> %101 to i64		; visa id: 104
  %103 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %21, i32 %22)
  %104 = extractvalue { i32, i32 } %103, 0		; visa id: 106
  %105 = extractvalue { i32, i32 } %103, 1		; visa id: 106
  %106 = insertelement <2 x i32> undef, i32 %104, i32 0		; visa id: 113
  %107 = insertelement <2 x i32> %106, i32 %105, i32 1		; visa id: 114
  %108 = bitcast <2 x i32> %107 to i64		; visa id: 115
  %109 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %24, i32 %25)
  %110 = extractvalue { i32, i32 } %109, 0		; visa id: 117
  %111 = extractvalue { i32, i32 } %109, 1		; visa id: 117
  %112 = insertelement <2 x i32> undef, i32 %110, i32 0		; visa id: 124
  %113 = insertelement <2 x i32> %112, i32 %111, i32 1		; visa id: 125
  %114 = bitcast <2 x i32> %113 to i64		; visa id: 126
  %115 = icmp slt i32 %57, %const_reg_dword
  %116 = icmp slt i32 %61, %const_reg_dword1		; visa id: 128
  %117 = and i1 %115, %116		; visa id: 129
  %118 = add i32 %57, 1		; visa id: 131
  %119 = icmp slt i32 %118, %const_reg_dword
  %120 = icmp slt i32 %61, %const_reg_dword1		; visa id: 132
  %121 = and i1 %119, %120		; visa id: 133
  %122 = add i32 %57, 2		; visa id: 135
  %123 = icmp slt i32 %122, %const_reg_dword
  %124 = icmp slt i32 %61, %const_reg_dword1		; visa id: 136
  %125 = and i1 %123, %124		; visa id: 137
  %126 = add i32 %57, 3		; visa id: 139
  %127 = icmp slt i32 %126, %const_reg_dword
  %128 = icmp slt i32 %61, %const_reg_dword1		; visa id: 140
  %129 = and i1 %127, %128		; visa id: 141
  %130 = add i32 %61, 1		; visa id: 143
  %131 = icmp slt i32 %130, %const_reg_dword1		; visa id: 144
  %132 = icmp slt i32 %57, %const_reg_dword
  %133 = and i1 %132, %131		; visa id: 145
  %134 = icmp slt i32 %118, %const_reg_dword
  %135 = icmp slt i32 %130, %const_reg_dword1		; visa id: 147
  %136 = and i1 %134, %135		; visa id: 148
  %137 = icmp slt i32 %122, %const_reg_dword
  %138 = icmp slt i32 %130, %const_reg_dword1		; visa id: 150
  %139 = and i1 %137, %138		; visa id: 151
  %140 = icmp slt i32 %126, %const_reg_dword
  %141 = icmp slt i32 %130, %const_reg_dword1		; visa id: 153
  %142 = and i1 %140, %141		; visa id: 154
  %143 = add i32 %61, 2		; visa id: 156
  %144 = icmp slt i32 %143, %const_reg_dword1		; visa id: 157
  %145 = icmp slt i32 %57, %const_reg_dword
  %146 = and i1 %145, %144		; visa id: 158
  %147 = icmp slt i32 %118, %const_reg_dword
  %148 = icmp slt i32 %143, %const_reg_dword1		; visa id: 160
  %149 = and i1 %147, %148		; visa id: 161
  %150 = icmp slt i32 %122, %const_reg_dword
  %151 = icmp slt i32 %143, %const_reg_dword1		; visa id: 163
  %152 = and i1 %150, %151		; visa id: 164
  %153 = icmp slt i32 %126, %const_reg_dword
  %154 = icmp slt i32 %143, %const_reg_dword1		; visa id: 166
  %155 = and i1 %153, %154		; visa id: 167
  %156 = add i32 %61, 3		; visa id: 169
  %157 = icmp slt i32 %156, %const_reg_dword1		; visa id: 170
  %158 = icmp slt i32 %57, %const_reg_dword
  %159 = and i1 %158, %157		; visa id: 171
  %160 = icmp slt i32 %118, %const_reg_dword
  %161 = icmp slt i32 %156, %const_reg_dword1		; visa id: 173
  %162 = and i1 %160, %161		; visa id: 174
  %163 = icmp slt i32 %122, %const_reg_dword
  %164 = icmp slt i32 %156, %const_reg_dword1		; visa id: 176
  %165 = and i1 %163, %164		; visa id: 177
  %166 = icmp slt i32 %126, %const_reg_dword
  %167 = icmp slt i32 %156, %const_reg_dword1		; visa id: 179
  %168 = and i1 %166, %167		; visa id: 180
  %169 = add i32 %61, 4		; visa id: 182
  %170 = icmp slt i32 %169, %const_reg_dword1		; visa id: 183
  %171 = icmp slt i32 %57, %const_reg_dword
  %172 = and i1 %171, %170		; visa id: 184
  %173 = icmp slt i32 %118, %const_reg_dword
  %174 = icmp slt i32 %169, %const_reg_dword1		; visa id: 186
  %175 = and i1 %173, %174		; visa id: 187
  %176 = icmp slt i32 %122, %const_reg_dword
  %177 = icmp slt i32 %169, %const_reg_dword1		; visa id: 189
  %178 = and i1 %176, %177		; visa id: 190
  %179 = icmp slt i32 %126, %const_reg_dword
  %180 = icmp slt i32 %169, %const_reg_dword1		; visa id: 192
  %181 = and i1 %179, %180		; visa id: 193
  %182 = add i32 %61, 5		; visa id: 195
  %183 = icmp slt i32 %182, %const_reg_dword1		; visa id: 196
  %184 = icmp slt i32 %57, %const_reg_dword
  %185 = and i1 %184, %183		; visa id: 197
  %186 = icmp slt i32 %118, %const_reg_dword
  %187 = icmp slt i32 %182, %const_reg_dword1		; visa id: 199
  %188 = and i1 %186, %187		; visa id: 200
  %189 = icmp slt i32 %122, %const_reg_dword
  %190 = icmp slt i32 %182, %const_reg_dword1		; visa id: 202
  %191 = and i1 %189, %190		; visa id: 203
  %192 = icmp slt i32 %126, %const_reg_dword
  %193 = icmp slt i32 %182, %const_reg_dword1		; visa id: 205
  %194 = and i1 %192, %193		; visa id: 206
  %195 = add i32 %61, 6		; visa id: 208
  %196 = icmp slt i32 %195, %const_reg_dword1		; visa id: 209
  %197 = icmp slt i32 %57, %const_reg_dword
  %198 = and i1 %197, %196		; visa id: 210
  %199 = icmp slt i32 %118, %const_reg_dword
  %200 = icmp slt i32 %195, %const_reg_dword1		; visa id: 212
  %201 = and i1 %199, %200		; visa id: 213
  %202 = icmp slt i32 %122, %const_reg_dword
  %203 = icmp slt i32 %195, %const_reg_dword1		; visa id: 215
  %204 = and i1 %202, %203		; visa id: 216
  %205 = icmp slt i32 %126, %const_reg_dword
  %206 = icmp slt i32 %195, %const_reg_dword1		; visa id: 218
  %207 = and i1 %205, %206		; visa id: 219
  %208 = add i32 %61, 7		; visa id: 221
  %209 = icmp slt i32 %208, %const_reg_dword1		; visa id: 222
  %210 = icmp slt i32 %57, %const_reg_dword
  %211 = and i1 %210, %209		; visa id: 223
  %212 = icmp slt i32 %118, %const_reg_dword
  %213 = icmp slt i32 %208, %const_reg_dword1		; visa id: 225
  %214 = and i1 %212, %213		; visa id: 226
  %215 = icmp slt i32 %122, %const_reg_dword
  %216 = icmp slt i32 %208, %const_reg_dword1		; visa id: 228
  %217 = and i1 %215, %216		; visa id: 229
  %218 = icmp slt i32 %126, %const_reg_dword
  %219 = icmp slt i32 %208, %const_reg_dword1		; visa id: 231
  %220 = and i1 %218, %219		; visa id: 232
  %221 = add i32 %61, 8		; visa id: 234
  %222 = icmp slt i32 %221, %const_reg_dword1		; visa id: 235
  %223 = icmp slt i32 %57, %const_reg_dword
  %224 = and i1 %223, %222		; visa id: 236
  %225 = icmp slt i32 %118, %const_reg_dword
  %226 = icmp slt i32 %221, %const_reg_dword1		; visa id: 238
  %227 = and i1 %225, %226		; visa id: 239
  %228 = icmp slt i32 %122, %const_reg_dword
  %229 = icmp slt i32 %221, %const_reg_dword1		; visa id: 241
  %230 = and i1 %228, %229		; visa id: 242
  %231 = icmp slt i32 %126, %const_reg_dword
  %232 = icmp slt i32 %221, %const_reg_dword1		; visa id: 244
  %233 = and i1 %231, %232		; visa id: 245
  %234 = add i32 %61, 9		; visa id: 247
  %235 = icmp slt i32 %234, %const_reg_dword1		; visa id: 248
  %236 = icmp slt i32 %57, %const_reg_dword
  %237 = and i1 %236, %235		; visa id: 249
  %238 = icmp slt i32 %118, %const_reg_dword
  %239 = icmp slt i32 %234, %const_reg_dword1		; visa id: 251
  %240 = and i1 %238, %239		; visa id: 252
  %241 = icmp slt i32 %122, %const_reg_dword
  %242 = icmp slt i32 %234, %const_reg_dword1		; visa id: 254
  %243 = and i1 %241, %242		; visa id: 255
  %244 = icmp slt i32 %126, %const_reg_dword
  %245 = icmp slt i32 %234, %const_reg_dword1		; visa id: 257
  %246 = and i1 %244, %245		; visa id: 258
  %247 = add i32 %61, 10		; visa id: 260
  %248 = icmp slt i32 %247, %const_reg_dword1		; visa id: 261
  %249 = icmp slt i32 %57, %const_reg_dword
  %250 = and i1 %249, %248		; visa id: 262
  %251 = icmp slt i32 %118, %const_reg_dword
  %252 = icmp slt i32 %247, %const_reg_dword1		; visa id: 264
  %253 = and i1 %251, %252		; visa id: 265
  %254 = icmp slt i32 %122, %const_reg_dword
  %255 = icmp slt i32 %247, %const_reg_dword1		; visa id: 267
  %256 = and i1 %254, %255		; visa id: 268
  %257 = icmp slt i32 %126, %const_reg_dword
  %258 = icmp slt i32 %247, %const_reg_dword1		; visa id: 270
  %259 = and i1 %257, %258		; visa id: 271
  %260 = add i32 %61, 11		; visa id: 273
  %261 = icmp slt i32 %260, %const_reg_dword1		; visa id: 274
  %262 = icmp slt i32 %57, %const_reg_dword
  %263 = and i1 %262, %261		; visa id: 275
  %264 = icmp slt i32 %118, %const_reg_dword
  %265 = icmp slt i32 %260, %const_reg_dword1		; visa id: 277
  %266 = and i1 %264, %265		; visa id: 278
  %267 = icmp slt i32 %122, %const_reg_dword
  %268 = icmp slt i32 %260, %const_reg_dword1		; visa id: 280
  %269 = and i1 %267, %268		; visa id: 281
  %270 = icmp slt i32 %126, %const_reg_dword
  %271 = icmp slt i32 %260, %const_reg_dword1		; visa id: 283
  %272 = and i1 %270, %271		; visa id: 284
  %273 = add i32 %61, 12		; visa id: 286
  %274 = icmp slt i32 %273, %const_reg_dword1		; visa id: 287
  %275 = icmp slt i32 %57, %const_reg_dword
  %276 = and i1 %275, %274		; visa id: 288
  %277 = icmp slt i32 %118, %const_reg_dword
  %278 = icmp slt i32 %273, %const_reg_dword1		; visa id: 290
  %279 = and i1 %277, %278		; visa id: 291
  %280 = icmp slt i32 %122, %const_reg_dword
  %281 = icmp slt i32 %273, %const_reg_dword1		; visa id: 293
  %282 = and i1 %280, %281		; visa id: 294
  %283 = icmp slt i32 %126, %const_reg_dword
  %284 = icmp slt i32 %273, %const_reg_dword1		; visa id: 296
  %285 = and i1 %283, %284		; visa id: 297
  %286 = add i32 %61, 13		; visa id: 299
  %287 = icmp slt i32 %286, %const_reg_dword1		; visa id: 300
  %288 = icmp slt i32 %57, %const_reg_dword
  %289 = and i1 %288, %287		; visa id: 301
  %290 = icmp slt i32 %118, %const_reg_dword
  %291 = icmp slt i32 %286, %const_reg_dword1		; visa id: 303
  %292 = and i1 %290, %291		; visa id: 304
  %293 = icmp slt i32 %122, %const_reg_dword
  %294 = icmp slt i32 %286, %const_reg_dword1		; visa id: 306
  %295 = and i1 %293, %294		; visa id: 307
  %296 = icmp slt i32 %126, %const_reg_dword
  %297 = icmp slt i32 %286, %const_reg_dword1		; visa id: 309
  %298 = and i1 %296, %297		; visa id: 310
  %299 = add i32 %61, 14		; visa id: 312
  %300 = icmp slt i32 %299, %const_reg_dword1		; visa id: 313
  %301 = icmp slt i32 %57, %const_reg_dword
  %302 = and i1 %301, %300		; visa id: 314
  %303 = icmp slt i32 %118, %const_reg_dword
  %304 = icmp slt i32 %299, %const_reg_dword1		; visa id: 316
  %305 = and i1 %303, %304		; visa id: 317
  %306 = icmp slt i32 %122, %const_reg_dword
  %307 = icmp slt i32 %299, %const_reg_dword1		; visa id: 319
  %308 = and i1 %306, %307		; visa id: 320
  %309 = icmp slt i32 %126, %const_reg_dword
  %310 = icmp slt i32 %299, %const_reg_dword1		; visa id: 322
  %311 = and i1 %309, %310		; visa id: 323
  %312 = add i32 %61, 15		; visa id: 325
  %313 = icmp slt i32 %312, %const_reg_dword1		; visa id: 326
  %314 = icmp slt i32 %57, %const_reg_dword
  %315 = and i1 %314, %313		; visa id: 327
  %316 = icmp slt i32 %118, %const_reg_dword
  %317 = icmp slt i32 %312, %const_reg_dword1		; visa id: 329
  %318 = and i1 %316, %317		; visa id: 330
  %319 = icmp slt i32 %122, %const_reg_dword
  %320 = icmp slt i32 %312, %const_reg_dword1		; visa id: 332
  %321 = and i1 %319, %320		; visa id: 333
  %322 = icmp slt i32 %126, %const_reg_dword
  %323 = icmp slt i32 %312, %const_reg_dword1		; visa id: 335
  %324 = and i1 %322, %323		; visa id: 336
  %325 = ashr i32 %57, 31		; visa id: 338
  %326 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %41, i32 %42)
  %327 = extractvalue { i32, i32 } %326, 0		; visa id: 339
  %328 = extractvalue { i32, i32 } %326, 1		; visa id: 339
  %329 = insertelement <2 x i32> undef, i32 %327, i32 0		; visa id: 346
  %330 = insertelement <2 x i32> %329, i32 %328, i32 1		; visa id: 347
  %331 = sext i32 %61 to i64		; visa id: 348
  %332 = ashr i32 %118, 31		; visa id: 349
  %333 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %41, i32 %42)
  %334 = extractvalue { i32, i32 } %333, 0		; visa id: 350
  %335 = extractvalue { i32, i32 } %333, 1		; visa id: 350
  %336 = insertelement <2 x i32> undef, i32 %334, i32 0		; visa id: 357
  %337 = insertelement <2 x i32> %336, i32 %335, i32 1		; visa id: 358
  %338 = ashr i32 %122, 31		; visa id: 359
  %339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %41, i32 %42)
  %340 = extractvalue { i32, i32 } %339, 0		; visa id: 360
  %341 = extractvalue { i32, i32 } %339, 1		; visa id: 360
  %342 = insertelement <2 x i32> undef, i32 %340, i32 0		; visa id: 367
  %343 = insertelement <2 x i32> %342, i32 %341, i32 1		; visa id: 368
  %344 = ashr i32 %126, 31		; visa id: 369
  %345 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %41, i32 %42)
  %346 = extractvalue { i32, i32 } %345, 0		; visa id: 370
  %347 = extractvalue { i32, i32 } %345, 1		; visa id: 370
  %348 = insertelement <2 x i32> undef, i32 %346, i32 0		; visa id: 377
  %349 = insertelement <2 x i32> %348, i32 %347, i32 1		; visa id: 378
  %350 = sext i32 %130 to i64		; visa id: 379
  %351 = sext i32 %143 to i64		; visa id: 380
  %352 = sext i32 %156 to i64		; visa id: 381
  %353 = sext i32 %169 to i64		; visa id: 382
  %354 = sext i32 %182 to i64		; visa id: 383
  %355 = sext i32 %195 to i64		; visa id: 384
  %356 = sext i32 %208 to i64		; visa id: 385
  %357 = sext i32 %221 to i64		; visa id: 386
  %358 = sext i32 %234 to i64		; visa id: 387
  %359 = sext i32 %247 to i64		; visa id: 388
  %360 = sext i32 %260 to i64		; visa id: 389
  %361 = sext i32 %273 to i64		; visa id: 390
  %362 = sext i32 %286 to i64		; visa id: 391
  %363 = sext i32 %299 to i64		; visa id: 392
  %364 = sext i32 %312 to i64		; visa id: 393
  %365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %47, i32 %48)
  %366 = extractvalue { i32, i32 } %365, 0		; visa id: 394
  %367 = extractvalue { i32, i32 } %365, 1		; visa id: 394
  %368 = insertelement <2 x i32> undef, i32 %366, i32 0		; visa id: 401
  %369 = insertelement <2 x i32> %368, i32 %367, i32 1		; visa id: 402
  %370 = bitcast <2 x i32> %369 to i64		; visa id: 403
  %371 = add nsw i64 %370, %331		; visa id: 405
  %372 = shl i64 %371, 2		; visa id: 406
  %373 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %44, i32 %45)
  %374 = extractvalue { i32, i32 } %373, 0		; visa id: 407
  %375 = extractvalue { i32, i32 } %373, 1		; visa id: 407
  %376 = insertelement <2 x i32> undef, i32 %374, i32 0		; visa id: 414
  %377 = insertelement <2 x i32> %376, i32 %375, i32 1		; visa id: 415
  %378 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %47, i32 %48)
  %379 = extractvalue { i32, i32 } %378, 0		; visa id: 416
  %380 = extractvalue { i32, i32 } %378, 1		; visa id: 416
  %381 = insertelement <2 x i32> undef, i32 %379, i32 0		; visa id: 423
  %382 = insertelement <2 x i32> %381, i32 %380, i32 1		; visa id: 424
  %383 = bitcast <2 x i32> %382 to i64		; visa id: 425
  %384 = add nsw i64 %383, %331		; visa id: 427
  %385 = shl i64 %384, 2		; visa id: 428
  %386 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %44, i32 %45)
  %387 = extractvalue { i32, i32 } %386, 0		; visa id: 429
  %388 = extractvalue { i32, i32 } %386, 1		; visa id: 429
  %389 = insertelement <2 x i32> undef, i32 %387, i32 0		; visa id: 436
  %390 = insertelement <2 x i32> %389, i32 %388, i32 1		; visa id: 437
  %391 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %47, i32 %48)
  %392 = extractvalue { i32, i32 } %391, 0		; visa id: 438
  %393 = extractvalue { i32, i32 } %391, 1		; visa id: 438
  %394 = insertelement <2 x i32> undef, i32 %392, i32 0		; visa id: 445
  %395 = insertelement <2 x i32> %394, i32 %393, i32 1		; visa id: 446
  %396 = bitcast <2 x i32> %395 to i64		; visa id: 447
  %397 = add nsw i64 %396, %331		; visa id: 449
  %398 = shl i64 %397, 2		; visa id: 450
  %399 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %44, i32 %45)
  %400 = extractvalue { i32, i32 } %399, 0		; visa id: 451
  %401 = extractvalue { i32, i32 } %399, 1		; visa id: 451
  %402 = insertelement <2 x i32> undef, i32 %400, i32 0		; visa id: 458
  %403 = insertelement <2 x i32> %402, i32 %401, i32 1		; visa id: 459
  %404 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %47, i32 %48)
  %405 = extractvalue { i32, i32 } %404, 0		; visa id: 460
  %406 = extractvalue { i32, i32 } %404, 1		; visa id: 460
  %407 = insertelement <2 x i32> undef, i32 %405, i32 0		; visa id: 467
  %408 = insertelement <2 x i32> %407, i32 %406, i32 1		; visa id: 468
  %409 = bitcast <2 x i32> %408 to i64		; visa id: 469
  %410 = add nsw i64 %409, %331		; visa id: 471
  %411 = shl i64 %410, 2		; visa id: 472
  %412 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %44, i32 %45)
  %413 = extractvalue { i32, i32 } %412, 0		; visa id: 473
  %414 = extractvalue { i32, i32 } %412, 1		; visa id: 473
  %415 = insertelement <2 x i32> undef, i32 %413, i32 0		; visa id: 480
  %416 = insertelement <2 x i32> %415, i32 %414, i32 1		; visa id: 481
  %417 = add nsw i64 %370, %350		; visa id: 482
  %418 = shl i64 %417, 2		; visa id: 483
  %419 = add nsw i64 %383, %350		; visa id: 484
  %420 = shl i64 %419, 2		; visa id: 485
  %421 = add nsw i64 %396, %350		; visa id: 486
  %422 = shl i64 %421, 2		; visa id: 487
  %423 = add nsw i64 %409, %350		; visa id: 488
  %424 = shl i64 %423, 2		; visa id: 489
  %425 = add nsw i64 %370, %351		; visa id: 490
  %426 = shl i64 %425, 2		; visa id: 491
  %427 = add nsw i64 %383, %351		; visa id: 492
  %428 = shl i64 %427, 2		; visa id: 493
  %429 = add nsw i64 %396, %351		; visa id: 494
  %430 = shl i64 %429, 2		; visa id: 495
  %431 = add nsw i64 %409, %351		; visa id: 496
  %432 = shl i64 %431, 2		; visa id: 497
  %433 = add nsw i64 %370, %352		; visa id: 498
  %434 = shl i64 %433, 2		; visa id: 499
  %435 = add nsw i64 %383, %352		; visa id: 500
  %436 = shl i64 %435, 2		; visa id: 501
  %437 = add nsw i64 %396, %352		; visa id: 502
  %438 = shl i64 %437, 2		; visa id: 503
  %439 = add nsw i64 %409, %352		; visa id: 504
  %440 = shl i64 %439, 2		; visa id: 505
  %441 = add nsw i64 %370, %353		; visa id: 506
  %442 = shl i64 %441, 2		; visa id: 507
  %443 = add nsw i64 %383, %353		; visa id: 508
  %444 = shl i64 %443, 2		; visa id: 509
  %445 = add nsw i64 %396, %353		; visa id: 510
  %446 = shl i64 %445, 2		; visa id: 511
  %447 = add nsw i64 %409, %353		; visa id: 512
  %448 = shl i64 %447, 2		; visa id: 513
  %449 = add nsw i64 %370, %354		; visa id: 514
  %450 = shl i64 %449, 2		; visa id: 515
  %451 = add nsw i64 %383, %354		; visa id: 516
  %452 = shl i64 %451, 2		; visa id: 517
  %453 = add nsw i64 %396, %354		; visa id: 518
  %454 = shl i64 %453, 2		; visa id: 519
  %455 = add nsw i64 %409, %354		; visa id: 520
  %456 = shl i64 %455, 2		; visa id: 521
  %457 = add nsw i64 %370, %355		; visa id: 522
  %458 = shl i64 %457, 2		; visa id: 523
  %459 = add nsw i64 %383, %355		; visa id: 524
  %460 = shl i64 %459, 2		; visa id: 525
  %461 = add nsw i64 %396, %355		; visa id: 526
  %462 = shl i64 %461, 2		; visa id: 527
  %463 = add nsw i64 %409, %355		; visa id: 528
  %464 = shl i64 %463, 2		; visa id: 529
  %465 = add nsw i64 %370, %356		; visa id: 530
  %466 = shl i64 %465, 2		; visa id: 531
  %467 = add nsw i64 %383, %356		; visa id: 532
  %468 = shl i64 %467, 2		; visa id: 533
  %469 = add nsw i64 %396, %356		; visa id: 534
  %470 = shl i64 %469, 2		; visa id: 535
  %471 = add nsw i64 %409, %356		; visa id: 536
  %472 = shl i64 %471, 2		; visa id: 537
  %473 = add nsw i64 %370, %357		; visa id: 538
  %474 = shl i64 %473, 2		; visa id: 539
  %475 = add nsw i64 %383, %357		; visa id: 540
  %476 = shl i64 %475, 2		; visa id: 541
  %477 = add nsw i64 %396, %357		; visa id: 542
  %478 = shl i64 %477, 2		; visa id: 543
  %479 = add nsw i64 %409, %357		; visa id: 544
  %480 = shl i64 %479, 2		; visa id: 545
  %481 = add nsw i64 %370, %358		; visa id: 546
  %482 = shl i64 %481, 2		; visa id: 547
  %483 = add nsw i64 %383, %358		; visa id: 548
  %484 = shl i64 %483, 2		; visa id: 549
  %485 = add nsw i64 %396, %358		; visa id: 550
  %486 = shl i64 %485, 2		; visa id: 551
  %487 = add nsw i64 %409, %358		; visa id: 552
  %488 = shl i64 %487, 2		; visa id: 553
  %489 = add nsw i64 %370, %359		; visa id: 554
  %490 = shl i64 %489, 2		; visa id: 555
  %491 = add nsw i64 %383, %359		; visa id: 556
  %492 = shl i64 %491, 2		; visa id: 557
  %493 = add nsw i64 %396, %359		; visa id: 558
  %494 = shl i64 %493, 2		; visa id: 559
  %495 = add nsw i64 %409, %359		; visa id: 560
  %496 = shl i64 %495, 2		; visa id: 561
  %497 = add nsw i64 %370, %360		; visa id: 562
  %498 = shl i64 %497, 2		; visa id: 563
  %499 = add nsw i64 %383, %360		; visa id: 564
  %500 = shl i64 %499, 2		; visa id: 565
  %501 = add nsw i64 %396, %360		; visa id: 566
  %502 = shl i64 %501, 2		; visa id: 567
  %503 = add nsw i64 %409, %360		; visa id: 568
  %504 = shl i64 %503, 2		; visa id: 569
  %505 = add nsw i64 %370, %361		; visa id: 570
  %506 = shl i64 %505, 2		; visa id: 571
  %507 = add nsw i64 %383, %361		; visa id: 572
  %508 = shl i64 %507, 2		; visa id: 573
  %509 = add nsw i64 %396, %361		; visa id: 574
  %510 = shl i64 %509, 2		; visa id: 575
  %511 = add nsw i64 %409, %361		; visa id: 576
  %512 = shl i64 %511, 2		; visa id: 577
  %513 = add nsw i64 %370, %362		; visa id: 578
  %514 = shl i64 %513, 2		; visa id: 579
  %515 = add nsw i64 %383, %362		; visa id: 580
  %516 = shl i64 %515, 2		; visa id: 581
  %517 = add nsw i64 %396, %362		; visa id: 582
  %518 = shl i64 %517, 2		; visa id: 583
  %519 = add nsw i64 %409, %362		; visa id: 584
  %520 = shl i64 %519, 2		; visa id: 585
  %521 = add nsw i64 %370, %363		; visa id: 586
  %522 = shl i64 %521, 2		; visa id: 587
  %523 = add nsw i64 %383, %363		; visa id: 588
  %524 = shl i64 %523, 2		; visa id: 589
  %525 = add nsw i64 %396, %363		; visa id: 590
  %526 = shl i64 %525, 2		; visa id: 591
  %527 = add nsw i64 %409, %363		; visa id: 592
  %528 = shl i64 %527, 2		; visa id: 593
  %529 = add nsw i64 %370, %364		; visa id: 594
  %530 = shl i64 %529, 2		; visa id: 595
  %531 = add nsw i64 %383, %364		; visa id: 596
  %532 = shl i64 %531, 2		; visa id: 597
  %533 = add nsw i64 %396, %364		; visa id: 598
  %534 = shl i64 %533, 2		; visa id: 599
  %535 = add nsw i64 %409, %364		; visa id: 600
  %536 = shl i64 %535, 2		; visa id: 601
  %537 = shl i64 %96, 1		; visa id: 602
  %538 = shl i64 %102, 1		; visa id: 603
  %.op3824 = shl i64 %108, 2		; visa id: 604
  %539 = bitcast i64 %.op3824 to <2 x i32>		; visa id: 605
  %540 = extractelement <2 x i32> %539, i32 0		; visa id: 606
  %541 = extractelement <2 x i32> %539, i32 1		; visa id: 606
  %542 = select i1 %78, i32 %540, i32 0		; visa id: 606
  %543 = select i1 %78, i32 %541, i32 0		; visa id: 607
  %544 = insertelement <2 x i32> undef, i32 %542, i32 0		; visa id: 608
  %545 = insertelement <2 x i32> %544, i32 %543, i32 1		; visa id: 609
  %546 = shl i64 %114, 2		; visa id: 610
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879		; visa id: 611

.preheader2.preheader:                            ; preds = %.preheader1.15..preheader2.preheader_crit_edge, %.lr.ph
; BB3 :
  %547 = phi i32 [ %26, %.lr.ph ], [ %3307, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %89, %.lr.ph ], [ %3312, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3821 = phi i64 [ %84, %.lr.ph ], [ %3311, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3822 = phi i64 [ %71, %.lr.ph ], [ %3310, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3823 = phi i64 [ %66, %.lr.ph ], [ %3309, %.preheader1.15..preheader2.preheader_crit_edge ]
  br i1 %90, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 612

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
; BB4 :
  br label %.preheader1.preheader, !stats.blockFrequency.digits !884, !stats.blockFrequency.scale !879		; visa id: 678

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
; BB5 :
  %sink_3874 = bitcast <2 x i32> %330 to i64		; visa id: 680
  %sink_3866 = shl i64 %sink_3874, 1		; visa id: 682
  %548 = add i64 %.in3823, %sink_3866		; visa id: 683
  %sink_3873 = bitcast <2 x i32> %337 to i64		; visa id: 684
  %sink_3865 = shl i64 %sink_3873, 1		; visa id: 686
  %549 = add i64 %.in3823, %sink_3865		; visa id: 687
  %sink_3872 = bitcast <2 x i32> %343 to i64		; visa id: 688
  %sink_3864 = shl i64 %sink_3872, 1		; visa id: 690
  %550 = add i64 %.in3823, %sink_3864		; visa id: 691
  %sink_3871 = bitcast <2 x i32> %349 to i64		; visa id: 692
  %sink_3863 = shl i64 %sink_3871, 1		; visa id: 694
  %551 = add i64 %.in3823, %sink_3863		; visa id: 695
  br label %.preheader.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879		; visa id: 761

.preheader.preheader:                             ; preds = %.preheader.15..preheader.preheader_crit_edge, %.preheader.preheader.preheader
; BB6 :
  %.sroa.254.1 = phi float [ %.sroa.254.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.250.1 = phi float [ %.sroa.250.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.246.1 = phi float [ %.sroa.246.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.242.1 = phi float [ %.sroa.242.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.238.1 = phi float [ %.sroa.238.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.234.1 = phi float [ %.sroa.234.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.230.1 = phi float [ %.sroa.230.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.226.1 = phi float [ %.sroa.226.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.222.1 = phi float [ %.sroa.222.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.218.1 = phi float [ %.sroa.218.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.214.1 = phi float [ %.sroa.214.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.210.1 = phi float [ %.sroa.210.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.206.1 = phi float [ %.sroa.206.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.202.1 = phi float [ %.sroa.202.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.198.1 = phi float [ %.sroa.198.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.194.1 = phi float [ %.sroa.194.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.190.1 = phi float [ %.sroa.190.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.186.1 = phi float [ %.sroa.186.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.182.1 = phi float [ %.sroa.182.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.178.1 = phi float [ %.sroa.178.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.174.1 = phi float [ %.sroa.174.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.170.1 = phi float [ %.sroa.170.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.166.1 = phi float [ %.sroa.166.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.162.1 = phi float [ %.sroa.162.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.158.1 = phi float [ %.sroa.158.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.154.1 = phi float [ %.sroa.154.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.150.1 = phi float [ %.sroa.150.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.146.1 = phi float [ %.sroa.146.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.142.1 = phi float [ %.sroa.142.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.138.1 = phi float [ %.sroa.138.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.134.1 = phi float [ %.sroa.134.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.130.1 = phi float [ %.sroa.130.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.126.1 = phi float [ %.sroa.126.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.122.1 = phi float [ %.sroa.122.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.118.1 = phi float [ %.sroa.118.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.114.1 = phi float [ %.sroa.114.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.110.1 = phi float [ %.sroa.110.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.106.1 = phi float [ %.sroa.106.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.102.1 = phi float [ %.sroa.102.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.98.1 = phi float [ %.sroa.98.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.94.1 = phi float [ %.sroa.94.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.90.1 = phi float [ %.sroa.90.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.86.1 = phi float [ %.sroa.86.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.82.1 = phi float [ %.sroa.82.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.78.1 = phi float [ %.sroa.78.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.74.1 = phi float [ %.sroa.74.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.70.1 = phi float [ %.sroa.70.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.66.1 = phi float [ %.sroa.66.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.62.1 = phi float [ %.sroa.62.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.58.1 = phi float [ %.sroa.58.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.54.1 = phi float [ %.sroa.54.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.50.1 = phi float [ %.sroa.50.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.46.1 = phi float [ %.sroa.46.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.42.1 = phi float [ %.sroa.42.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.38.1 = phi float [ %.sroa.38.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.34.1 = phi float [ %.sroa.34.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.30.1 = phi float [ %.sroa.30.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.26.1 = phi float [ %.sroa.26.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.22.1 = phi float [ %.sroa.22.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.18.1 = phi float [ %.sroa.18.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.14.1 = phi float [ %.sroa.14.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.10.1 = phi float [ %.sroa.10.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.6.1 = phi float [ %.sroa.6.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %.sroa.0.1 = phi float [ %.sroa.0.2, %.preheader.15..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %552 = phi i32 [ %2217, %.preheader.15..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
  %sink_3861 = shl nsw i64 %331, 1		; visa id: 762
  %sink_3875 = bitcast i64 %const_reg_qword5 to <2 x i32>		; visa id: 763
  %sink_3826 = extractelement <2 x i32> %sink_3875, i32 0		; visa id: 764
  %sink_3825 = extractelement <2 x i32> %sink_3875, i32 1		; visa id: 764
  br i1 %117, label %553, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 764

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

553:                                              ; preds = %.preheader.preheader
; BB8 :
  %.sroa.256.0.insert.ext = zext i32 %552 to i64		; visa id: 766
  %554 = shl nuw nsw i64 %.sroa.256.0.insert.ext, 1		; visa id: 767
  %555 = add i64 %548, %554		; visa id: 768
  %556 = inttoptr i64 %555 to i16 addrspace(4)*		; visa id: 769
  %557 = addrspacecast i16 addrspace(4)* %556 to i16 addrspace(1)*		; visa id: 769
  %558 = load i16, i16 addrspace(1)* %557, align 2		; visa id: 770
  %559 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %560 = extractvalue { i32, i32 } %559, 0		; visa id: 772
  %561 = extractvalue { i32, i32 } %559, 1		; visa id: 772
  %562 = insertelement <2 x i32> undef, i32 %560, i32 0		; visa id: 779
  %563 = insertelement <2 x i32> %562, i32 %561, i32 1		; visa id: 780
  %564 = bitcast <2 x i32> %563 to i64		; visa id: 781
  %565 = shl i64 %564, 1		; visa id: 783
  %566 = add i64 %.in3822, %565		; visa id: 784
  %567 = add i64 %566, %sink_3861		; visa id: 785
  %568 = inttoptr i64 %567 to i16 addrspace(4)*		; visa id: 786
  %569 = addrspacecast i16 addrspace(4)* %568 to i16 addrspace(1)*		; visa id: 786
  %570 = load i16, i16 addrspace(1)* %569, align 2		; visa id: 787
  %571 = zext i16 %558 to i32		; visa id: 789
  %572 = shl nuw i32 %571, 16, !spirv.Decorations !888		; visa id: 790
  %573 = bitcast i32 %572 to float
  %574 = zext i16 %570 to i32		; visa id: 791
  %575 = shl nuw i32 %574, 16, !spirv.Decorations !888		; visa id: 792
  %576 = bitcast i32 %575 to float
  %577 = fmul reassoc nsz arcp contract float %573, %576, !spirv.Decorations !881
  %578 = fadd reassoc nsz arcp contract float %577, %.sroa.0.1, !spirv.Decorations !881		; visa id: 793
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 794

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %553
; BB9 :
  %.sroa.0.2 = phi float [ %578, %553 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  br i1 %121, label %579, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 795

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
; BB:
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

579:                                              ; preds = %._crit_edge
; BB11 :
  %.sroa.256.0.insert.ext588 = zext i32 %552 to i64		; visa id: 797
  %580 = shl nuw nsw i64 %.sroa.256.0.insert.ext588, 1		; visa id: 798
  %581 = add i64 %549, %580		; visa id: 799
  %582 = inttoptr i64 %581 to i16 addrspace(4)*		; visa id: 800
  %583 = addrspacecast i16 addrspace(4)* %582 to i16 addrspace(1)*		; visa id: 800
  %584 = load i16, i16 addrspace(1)* %583, align 2		; visa id: 801
  %585 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %586 = extractvalue { i32, i32 } %585, 0		; visa id: 803
  %587 = extractvalue { i32, i32 } %585, 1		; visa id: 803
  %588 = insertelement <2 x i32> undef, i32 %586, i32 0		; visa id: 810
  %589 = insertelement <2 x i32> %588, i32 %587, i32 1		; visa id: 811
  %590 = bitcast <2 x i32> %589 to i64		; visa id: 812
  %591 = shl i64 %590, 1		; visa id: 814
  %592 = add i64 %.in3822, %591		; visa id: 815
  %593 = add i64 %592, %sink_3861		; visa id: 816
  %594 = inttoptr i64 %593 to i16 addrspace(4)*		; visa id: 817
  %595 = addrspacecast i16 addrspace(4)* %594 to i16 addrspace(1)*		; visa id: 817
  %596 = load i16, i16 addrspace(1)* %595, align 2		; visa id: 818
  %597 = zext i16 %584 to i32		; visa id: 820
  %598 = shl nuw i32 %597, 16, !spirv.Decorations !888		; visa id: 821
  %599 = bitcast i32 %598 to float
  %600 = zext i16 %596 to i32		; visa id: 822
  %601 = shl nuw i32 %600, 16, !spirv.Decorations !888		; visa id: 823
  %602 = bitcast i32 %601 to float
  %603 = fmul reassoc nsz arcp contract float %599, %602, !spirv.Decorations !881
  %604 = fadd reassoc nsz arcp contract float %603, %.sroa.66.1, !spirv.Decorations !881		; visa id: 824
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 825

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %579
; BB12 :
  %.sroa.66.2 = phi float [ %604, %579 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  br i1 %125, label %605, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 826

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
; BB:
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

605:                                              ; preds = %._crit_edge.1
; BB14 :
  %.sroa.256.0.insert.ext593 = zext i32 %552 to i64		; visa id: 828
  %606 = shl nuw nsw i64 %.sroa.256.0.insert.ext593, 1		; visa id: 829
  %607 = add i64 %550, %606		; visa id: 830
  %608 = inttoptr i64 %607 to i16 addrspace(4)*		; visa id: 831
  %609 = addrspacecast i16 addrspace(4)* %608 to i16 addrspace(1)*		; visa id: 831
  %610 = load i16, i16 addrspace(1)* %609, align 2		; visa id: 832
  %611 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %612 = extractvalue { i32, i32 } %611, 0		; visa id: 834
  %613 = extractvalue { i32, i32 } %611, 1		; visa id: 834
  %614 = insertelement <2 x i32> undef, i32 %612, i32 0		; visa id: 841
  %615 = insertelement <2 x i32> %614, i32 %613, i32 1		; visa id: 842
  %616 = bitcast <2 x i32> %615 to i64		; visa id: 843
  %617 = shl i64 %616, 1		; visa id: 845
  %618 = add i64 %.in3822, %617		; visa id: 846
  %619 = add i64 %618, %sink_3861		; visa id: 847
  %620 = inttoptr i64 %619 to i16 addrspace(4)*		; visa id: 848
  %621 = addrspacecast i16 addrspace(4)* %620 to i16 addrspace(1)*		; visa id: 848
  %622 = load i16, i16 addrspace(1)* %621, align 2		; visa id: 849
  %623 = zext i16 %610 to i32		; visa id: 851
  %624 = shl nuw i32 %623, 16, !spirv.Decorations !888		; visa id: 852
  %625 = bitcast i32 %624 to float
  %626 = zext i16 %622 to i32		; visa id: 853
  %627 = shl nuw i32 %626, 16, !spirv.Decorations !888		; visa id: 854
  %628 = bitcast i32 %627 to float
  %629 = fmul reassoc nsz arcp contract float %625, %628, !spirv.Decorations !881
  %630 = fadd reassoc nsz arcp contract float %629, %.sroa.130.1, !spirv.Decorations !881		; visa id: 855
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 856

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %605
; BB15 :
  %.sroa.130.2 = phi float [ %630, %605 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  br i1 %129, label %631, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 857

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
; BB:
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

631:                                              ; preds = %._crit_edge.2
; BB17 :
  %.sroa.256.0.insert.ext598 = zext i32 %552 to i64		; visa id: 859
  %632 = shl nuw nsw i64 %.sroa.256.0.insert.ext598, 1		; visa id: 860
  %633 = add i64 %551, %632		; visa id: 861
  %634 = inttoptr i64 %633 to i16 addrspace(4)*		; visa id: 862
  %635 = addrspacecast i16 addrspace(4)* %634 to i16 addrspace(1)*		; visa id: 862
  %636 = load i16, i16 addrspace(1)* %635, align 2		; visa id: 863
  %637 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %638 = extractvalue { i32, i32 } %637, 0		; visa id: 865
  %639 = extractvalue { i32, i32 } %637, 1		; visa id: 865
  %640 = insertelement <2 x i32> undef, i32 %638, i32 0		; visa id: 872
  %641 = insertelement <2 x i32> %640, i32 %639, i32 1		; visa id: 873
  %642 = bitcast <2 x i32> %641 to i64		; visa id: 874
  %643 = shl i64 %642, 1		; visa id: 876
  %644 = add i64 %.in3822, %643		; visa id: 877
  %645 = add i64 %644, %sink_3861		; visa id: 878
  %646 = inttoptr i64 %645 to i16 addrspace(4)*		; visa id: 879
  %647 = addrspacecast i16 addrspace(4)* %646 to i16 addrspace(1)*		; visa id: 879
  %648 = load i16, i16 addrspace(1)* %647, align 2		; visa id: 880
  %649 = zext i16 %636 to i32		; visa id: 882
  %650 = shl nuw i32 %649, 16, !spirv.Decorations !888		; visa id: 883
  %651 = bitcast i32 %650 to float
  %652 = zext i16 %648 to i32		; visa id: 884
  %653 = shl nuw i32 %652, 16, !spirv.Decorations !888		; visa id: 885
  %654 = bitcast i32 %653 to float
  %655 = fmul reassoc nsz arcp contract float %651, %654, !spirv.Decorations !881
  %656 = fadd reassoc nsz arcp contract float %655, %.sroa.194.1, !spirv.Decorations !881		; visa id: 886
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 887

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %631
; BB18 :
  %.sroa.194.2 = phi float [ %656, %631 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %sink_3856 = shl nsw i64 %350, 1		; visa id: 888
  br i1 %133, label %657, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 889

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
; BB:
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

657:                                              ; preds = %.preheader
; BB20 :
  %.sroa.256.0.insert.ext603 = zext i32 %552 to i64		; visa id: 891
  %658 = shl nuw nsw i64 %.sroa.256.0.insert.ext603, 1		; visa id: 892
  %659 = add i64 %548, %658		; visa id: 893
  %660 = inttoptr i64 %659 to i16 addrspace(4)*		; visa id: 894
  %661 = addrspacecast i16 addrspace(4)* %660 to i16 addrspace(1)*		; visa id: 894
  %662 = load i16, i16 addrspace(1)* %661, align 2		; visa id: 895
  %663 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %664 = extractvalue { i32, i32 } %663, 0		; visa id: 897
  %665 = extractvalue { i32, i32 } %663, 1		; visa id: 897
  %666 = insertelement <2 x i32> undef, i32 %664, i32 0		; visa id: 904
  %667 = insertelement <2 x i32> %666, i32 %665, i32 1		; visa id: 905
  %668 = bitcast <2 x i32> %667 to i64		; visa id: 906
  %669 = shl i64 %668, 1		; visa id: 908
  %670 = add i64 %.in3822, %669		; visa id: 909
  %671 = add i64 %670, %sink_3856		; visa id: 910
  %672 = inttoptr i64 %671 to i16 addrspace(4)*		; visa id: 911
  %673 = addrspacecast i16 addrspace(4)* %672 to i16 addrspace(1)*		; visa id: 911
  %674 = load i16, i16 addrspace(1)* %673, align 2		; visa id: 912
  %675 = zext i16 %662 to i32		; visa id: 914
  %676 = shl nuw i32 %675, 16, !spirv.Decorations !888		; visa id: 915
  %677 = bitcast i32 %676 to float
  %678 = zext i16 %674 to i32		; visa id: 916
  %679 = shl nuw i32 %678, 16, !spirv.Decorations !888		; visa id: 917
  %680 = bitcast i32 %679 to float
  %681 = fmul reassoc nsz arcp contract float %677, %680, !spirv.Decorations !881
  %682 = fadd reassoc nsz arcp contract float %681, %.sroa.6.1, !spirv.Decorations !881		; visa id: 918
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 919

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %657
; BB21 :
  %.sroa.6.2 = phi float [ %682, %657 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %136, label %683, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 920

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
; BB:
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

683:                                              ; preds = %._crit_edge.173
; BB23 :
  %.sroa.256.0.insert.ext608 = zext i32 %552 to i64		; visa id: 922
  %684 = shl nuw nsw i64 %.sroa.256.0.insert.ext608, 1		; visa id: 923
  %685 = add i64 %549, %684		; visa id: 924
  %686 = inttoptr i64 %685 to i16 addrspace(4)*		; visa id: 925
  %687 = addrspacecast i16 addrspace(4)* %686 to i16 addrspace(1)*		; visa id: 925
  %688 = load i16, i16 addrspace(1)* %687, align 2		; visa id: 926
  %689 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %690 = extractvalue { i32, i32 } %689, 0		; visa id: 928
  %691 = extractvalue { i32, i32 } %689, 1		; visa id: 928
  %692 = insertelement <2 x i32> undef, i32 %690, i32 0		; visa id: 935
  %693 = insertelement <2 x i32> %692, i32 %691, i32 1		; visa id: 936
  %694 = bitcast <2 x i32> %693 to i64		; visa id: 937
  %695 = shl i64 %694, 1		; visa id: 939
  %696 = add i64 %.in3822, %695		; visa id: 940
  %697 = add i64 %696, %sink_3856		; visa id: 941
  %698 = inttoptr i64 %697 to i16 addrspace(4)*		; visa id: 942
  %699 = addrspacecast i16 addrspace(4)* %698 to i16 addrspace(1)*		; visa id: 942
  %700 = load i16, i16 addrspace(1)* %699, align 2		; visa id: 943
  %701 = zext i16 %688 to i32		; visa id: 945
  %702 = shl nuw i32 %701, 16, !spirv.Decorations !888		; visa id: 946
  %703 = bitcast i32 %702 to float
  %704 = zext i16 %700 to i32		; visa id: 947
  %705 = shl nuw i32 %704, 16, !spirv.Decorations !888		; visa id: 948
  %706 = bitcast i32 %705 to float
  %707 = fmul reassoc nsz arcp contract float %703, %706, !spirv.Decorations !881
  %708 = fadd reassoc nsz arcp contract float %707, %.sroa.70.1, !spirv.Decorations !881		; visa id: 949
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 950

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %683
; BB24 :
  %.sroa.70.2 = phi float [ %708, %683 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %139, label %709, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 951

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
; BB:
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

709:                                              ; preds = %._crit_edge.1.1
; BB26 :
  %.sroa.256.0.insert.ext613 = zext i32 %552 to i64		; visa id: 953
  %710 = shl nuw nsw i64 %.sroa.256.0.insert.ext613, 1		; visa id: 954
  %711 = add i64 %550, %710		; visa id: 955
  %712 = inttoptr i64 %711 to i16 addrspace(4)*		; visa id: 956
  %713 = addrspacecast i16 addrspace(4)* %712 to i16 addrspace(1)*		; visa id: 956
  %714 = load i16, i16 addrspace(1)* %713, align 2		; visa id: 957
  %715 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %716 = extractvalue { i32, i32 } %715, 0		; visa id: 959
  %717 = extractvalue { i32, i32 } %715, 1		; visa id: 959
  %718 = insertelement <2 x i32> undef, i32 %716, i32 0		; visa id: 966
  %719 = insertelement <2 x i32> %718, i32 %717, i32 1		; visa id: 967
  %720 = bitcast <2 x i32> %719 to i64		; visa id: 968
  %721 = shl i64 %720, 1		; visa id: 970
  %722 = add i64 %.in3822, %721		; visa id: 971
  %723 = add i64 %722, %sink_3856		; visa id: 972
  %724 = inttoptr i64 %723 to i16 addrspace(4)*		; visa id: 973
  %725 = addrspacecast i16 addrspace(4)* %724 to i16 addrspace(1)*		; visa id: 973
  %726 = load i16, i16 addrspace(1)* %725, align 2		; visa id: 974
  %727 = zext i16 %714 to i32		; visa id: 976
  %728 = shl nuw i32 %727, 16, !spirv.Decorations !888		; visa id: 977
  %729 = bitcast i32 %728 to float
  %730 = zext i16 %726 to i32		; visa id: 978
  %731 = shl nuw i32 %730, 16, !spirv.Decorations !888		; visa id: 979
  %732 = bitcast i32 %731 to float
  %733 = fmul reassoc nsz arcp contract float %729, %732, !spirv.Decorations !881
  %734 = fadd reassoc nsz arcp contract float %733, %.sroa.134.1, !spirv.Decorations !881		; visa id: 980
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 981

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %709
; BB27 :
  %.sroa.134.2 = phi float [ %734, %709 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %142, label %735, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 982

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
; BB:
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

735:                                              ; preds = %._crit_edge.2.1
; BB29 :
  %.sroa.256.0.insert.ext618 = zext i32 %552 to i64		; visa id: 984
  %736 = shl nuw nsw i64 %.sroa.256.0.insert.ext618, 1		; visa id: 985
  %737 = add i64 %551, %736		; visa id: 986
  %738 = inttoptr i64 %737 to i16 addrspace(4)*		; visa id: 987
  %739 = addrspacecast i16 addrspace(4)* %738 to i16 addrspace(1)*		; visa id: 987
  %740 = load i16, i16 addrspace(1)* %739, align 2		; visa id: 988
  %741 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %742 = extractvalue { i32, i32 } %741, 0		; visa id: 990
  %743 = extractvalue { i32, i32 } %741, 1		; visa id: 990
  %744 = insertelement <2 x i32> undef, i32 %742, i32 0		; visa id: 997
  %745 = insertelement <2 x i32> %744, i32 %743, i32 1		; visa id: 998
  %746 = bitcast <2 x i32> %745 to i64		; visa id: 999
  %747 = shl i64 %746, 1		; visa id: 1001
  %748 = add i64 %.in3822, %747		; visa id: 1002
  %749 = add i64 %748, %sink_3856		; visa id: 1003
  %750 = inttoptr i64 %749 to i16 addrspace(4)*		; visa id: 1004
  %751 = addrspacecast i16 addrspace(4)* %750 to i16 addrspace(1)*		; visa id: 1004
  %752 = load i16, i16 addrspace(1)* %751, align 2		; visa id: 1005
  %753 = zext i16 %740 to i32		; visa id: 1007
  %754 = shl nuw i32 %753, 16, !spirv.Decorations !888		; visa id: 1008
  %755 = bitcast i32 %754 to float
  %756 = zext i16 %752 to i32		; visa id: 1009
  %757 = shl nuw i32 %756, 16, !spirv.Decorations !888		; visa id: 1010
  %758 = bitcast i32 %757 to float
  %759 = fmul reassoc nsz arcp contract float %755, %758, !spirv.Decorations !881
  %760 = fadd reassoc nsz arcp contract float %759, %.sroa.198.1, !spirv.Decorations !881		; visa id: 1011
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1012

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %735
; BB30 :
  %.sroa.198.2 = phi float [ %760, %735 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %sink_3854 = shl nsw i64 %351, 1		; visa id: 1013
  br i1 %146, label %761, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1014

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
; BB:
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

761:                                              ; preds = %.preheader.1
; BB32 :
  %.sroa.256.0.insert.ext623 = zext i32 %552 to i64		; visa id: 1016
  %762 = shl nuw nsw i64 %.sroa.256.0.insert.ext623, 1		; visa id: 1017
  %763 = add i64 %548, %762		; visa id: 1018
  %764 = inttoptr i64 %763 to i16 addrspace(4)*		; visa id: 1019
  %765 = addrspacecast i16 addrspace(4)* %764 to i16 addrspace(1)*		; visa id: 1019
  %766 = load i16, i16 addrspace(1)* %765, align 2		; visa id: 1020
  %767 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %768 = extractvalue { i32, i32 } %767, 0		; visa id: 1022
  %769 = extractvalue { i32, i32 } %767, 1		; visa id: 1022
  %770 = insertelement <2 x i32> undef, i32 %768, i32 0		; visa id: 1029
  %771 = insertelement <2 x i32> %770, i32 %769, i32 1		; visa id: 1030
  %772 = bitcast <2 x i32> %771 to i64		; visa id: 1031
  %773 = shl i64 %772, 1		; visa id: 1033
  %774 = add i64 %.in3822, %773		; visa id: 1034
  %775 = add i64 %774, %sink_3854		; visa id: 1035
  %776 = inttoptr i64 %775 to i16 addrspace(4)*		; visa id: 1036
  %777 = addrspacecast i16 addrspace(4)* %776 to i16 addrspace(1)*		; visa id: 1036
  %778 = load i16, i16 addrspace(1)* %777, align 2		; visa id: 1037
  %779 = zext i16 %766 to i32		; visa id: 1039
  %780 = shl nuw i32 %779, 16, !spirv.Decorations !888		; visa id: 1040
  %781 = bitcast i32 %780 to float
  %782 = zext i16 %778 to i32		; visa id: 1041
  %783 = shl nuw i32 %782, 16, !spirv.Decorations !888		; visa id: 1042
  %784 = bitcast i32 %783 to float
  %785 = fmul reassoc nsz arcp contract float %781, %784, !spirv.Decorations !881
  %786 = fadd reassoc nsz arcp contract float %785, %.sroa.10.1, !spirv.Decorations !881		; visa id: 1043
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1044

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %761
; BB33 :
  %.sroa.10.2 = phi float [ %786, %761 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %149, label %787, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1045

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
; BB:
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

787:                                              ; preds = %._crit_edge.274
; BB35 :
  %.sroa.256.0.insert.ext628 = zext i32 %552 to i64		; visa id: 1047
  %788 = shl nuw nsw i64 %.sroa.256.0.insert.ext628, 1		; visa id: 1048
  %789 = add i64 %549, %788		; visa id: 1049
  %790 = inttoptr i64 %789 to i16 addrspace(4)*		; visa id: 1050
  %791 = addrspacecast i16 addrspace(4)* %790 to i16 addrspace(1)*		; visa id: 1050
  %792 = load i16, i16 addrspace(1)* %791, align 2		; visa id: 1051
  %793 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %794 = extractvalue { i32, i32 } %793, 0		; visa id: 1053
  %795 = extractvalue { i32, i32 } %793, 1		; visa id: 1053
  %796 = insertelement <2 x i32> undef, i32 %794, i32 0		; visa id: 1060
  %797 = insertelement <2 x i32> %796, i32 %795, i32 1		; visa id: 1061
  %798 = bitcast <2 x i32> %797 to i64		; visa id: 1062
  %799 = shl i64 %798, 1		; visa id: 1064
  %800 = add i64 %.in3822, %799		; visa id: 1065
  %801 = add i64 %800, %sink_3854		; visa id: 1066
  %802 = inttoptr i64 %801 to i16 addrspace(4)*		; visa id: 1067
  %803 = addrspacecast i16 addrspace(4)* %802 to i16 addrspace(1)*		; visa id: 1067
  %804 = load i16, i16 addrspace(1)* %803, align 2		; visa id: 1068
  %805 = zext i16 %792 to i32		; visa id: 1070
  %806 = shl nuw i32 %805, 16, !spirv.Decorations !888		; visa id: 1071
  %807 = bitcast i32 %806 to float
  %808 = zext i16 %804 to i32		; visa id: 1072
  %809 = shl nuw i32 %808, 16, !spirv.Decorations !888		; visa id: 1073
  %810 = bitcast i32 %809 to float
  %811 = fmul reassoc nsz arcp contract float %807, %810, !spirv.Decorations !881
  %812 = fadd reassoc nsz arcp contract float %811, %.sroa.74.1, !spirv.Decorations !881		; visa id: 1074
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1075

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %787
; BB36 :
  %.sroa.74.2 = phi float [ %812, %787 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %152, label %813, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1076

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
; BB:
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

813:                                              ; preds = %._crit_edge.1.2
; BB38 :
  %.sroa.256.0.insert.ext633 = zext i32 %552 to i64		; visa id: 1078
  %814 = shl nuw nsw i64 %.sroa.256.0.insert.ext633, 1		; visa id: 1079
  %815 = add i64 %550, %814		; visa id: 1080
  %816 = inttoptr i64 %815 to i16 addrspace(4)*		; visa id: 1081
  %817 = addrspacecast i16 addrspace(4)* %816 to i16 addrspace(1)*		; visa id: 1081
  %818 = load i16, i16 addrspace(1)* %817, align 2		; visa id: 1082
  %819 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %820 = extractvalue { i32, i32 } %819, 0		; visa id: 1084
  %821 = extractvalue { i32, i32 } %819, 1		; visa id: 1084
  %822 = insertelement <2 x i32> undef, i32 %820, i32 0		; visa id: 1091
  %823 = insertelement <2 x i32> %822, i32 %821, i32 1		; visa id: 1092
  %824 = bitcast <2 x i32> %823 to i64		; visa id: 1093
  %825 = shl i64 %824, 1		; visa id: 1095
  %826 = add i64 %.in3822, %825		; visa id: 1096
  %827 = add i64 %826, %sink_3854		; visa id: 1097
  %828 = inttoptr i64 %827 to i16 addrspace(4)*		; visa id: 1098
  %829 = addrspacecast i16 addrspace(4)* %828 to i16 addrspace(1)*		; visa id: 1098
  %830 = load i16, i16 addrspace(1)* %829, align 2		; visa id: 1099
  %831 = zext i16 %818 to i32		; visa id: 1101
  %832 = shl nuw i32 %831, 16, !spirv.Decorations !888		; visa id: 1102
  %833 = bitcast i32 %832 to float
  %834 = zext i16 %830 to i32		; visa id: 1103
  %835 = shl nuw i32 %834, 16, !spirv.Decorations !888		; visa id: 1104
  %836 = bitcast i32 %835 to float
  %837 = fmul reassoc nsz arcp contract float %833, %836, !spirv.Decorations !881
  %838 = fadd reassoc nsz arcp contract float %837, %.sroa.138.1, !spirv.Decorations !881		; visa id: 1105
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1106

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %813
; BB39 :
  %.sroa.138.2 = phi float [ %838, %813 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %155, label %839, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1107

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
; BB:
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

839:                                              ; preds = %._crit_edge.2.2
; BB41 :
  %.sroa.256.0.insert.ext638 = zext i32 %552 to i64		; visa id: 1109
  %840 = shl nuw nsw i64 %.sroa.256.0.insert.ext638, 1		; visa id: 1110
  %841 = add i64 %551, %840		; visa id: 1111
  %842 = inttoptr i64 %841 to i16 addrspace(4)*		; visa id: 1112
  %843 = addrspacecast i16 addrspace(4)* %842 to i16 addrspace(1)*		; visa id: 1112
  %844 = load i16, i16 addrspace(1)* %843, align 2		; visa id: 1113
  %845 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %846 = extractvalue { i32, i32 } %845, 0		; visa id: 1115
  %847 = extractvalue { i32, i32 } %845, 1		; visa id: 1115
  %848 = insertelement <2 x i32> undef, i32 %846, i32 0		; visa id: 1122
  %849 = insertelement <2 x i32> %848, i32 %847, i32 1		; visa id: 1123
  %850 = bitcast <2 x i32> %849 to i64		; visa id: 1124
  %851 = shl i64 %850, 1		; visa id: 1126
  %852 = add i64 %.in3822, %851		; visa id: 1127
  %853 = add i64 %852, %sink_3854		; visa id: 1128
  %854 = inttoptr i64 %853 to i16 addrspace(4)*		; visa id: 1129
  %855 = addrspacecast i16 addrspace(4)* %854 to i16 addrspace(1)*		; visa id: 1129
  %856 = load i16, i16 addrspace(1)* %855, align 2		; visa id: 1130
  %857 = zext i16 %844 to i32		; visa id: 1132
  %858 = shl nuw i32 %857, 16, !spirv.Decorations !888		; visa id: 1133
  %859 = bitcast i32 %858 to float
  %860 = zext i16 %856 to i32		; visa id: 1134
  %861 = shl nuw i32 %860, 16, !spirv.Decorations !888		; visa id: 1135
  %862 = bitcast i32 %861 to float
  %863 = fmul reassoc nsz arcp contract float %859, %862, !spirv.Decorations !881
  %864 = fadd reassoc nsz arcp contract float %863, %.sroa.202.1, !spirv.Decorations !881		; visa id: 1136
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1137

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %839
; BB42 :
  %.sroa.202.2 = phi float [ %864, %839 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %sink_3852 = shl nsw i64 %352, 1		; visa id: 1138
  br i1 %159, label %865, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1139

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
; BB:
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

865:                                              ; preds = %.preheader.2
; BB44 :
  %.sroa.256.0.insert.ext643 = zext i32 %552 to i64		; visa id: 1141
  %866 = shl nuw nsw i64 %.sroa.256.0.insert.ext643, 1		; visa id: 1142
  %867 = add i64 %548, %866		; visa id: 1143
  %868 = inttoptr i64 %867 to i16 addrspace(4)*		; visa id: 1144
  %869 = addrspacecast i16 addrspace(4)* %868 to i16 addrspace(1)*		; visa id: 1144
  %870 = load i16, i16 addrspace(1)* %869, align 2		; visa id: 1145
  %871 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %872 = extractvalue { i32, i32 } %871, 0		; visa id: 1147
  %873 = extractvalue { i32, i32 } %871, 1		; visa id: 1147
  %874 = insertelement <2 x i32> undef, i32 %872, i32 0		; visa id: 1154
  %875 = insertelement <2 x i32> %874, i32 %873, i32 1		; visa id: 1155
  %876 = bitcast <2 x i32> %875 to i64		; visa id: 1156
  %877 = shl i64 %876, 1		; visa id: 1158
  %878 = add i64 %.in3822, %877		; visa id: 1159
  %879 = add i64 %878, %sink_3852		; visa id: 1160
  %880 = inttoptr i64 %879 to i16 addrspace(4)*		; visa id: 1161
  %881 = addrspacecast i16 addrspace(4)* %880 to i16 addrspace(1)*		; visa id: 1161
  %882 = load i16, i16 addrspace(1)* %881, align 2		; visa id: 1162
  %883 = zext i16 %870 to i32		; visa id: 1164
  %884 = shl nuw i32 %883, 16, !spirv.Decorations !888		; visa id: 1165
  %885 = bitcast i32 %884 to float
  %886 = zext i16 %882 to i32		; visa id: 1166
  %887 = shl nuw i32 %886, 16, !spirv.Decorations !888		; visa id: 1167
  %888 = bitcast i32 %887 to float
  %889 = fmul reassoc nsz arcp contract float %885, %888, !spirv.Decorations !881
  %890 = fadd reassoc nsz arcp contract float %889, %.sroa.14.1, !spirv.Decorations !881		; visa id: 1168
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1169

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %865
; BB45 :
  %.sroa.14.2 = phi float [ %890, %865 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %162, label %891, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1170

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
; BB:
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

891:                                              ; preds = %._crit_edge.375
; BB47 :
  %.sroa.256.0.insert.ext648 = zext i32 %552 to i64		; visa id: 1172
  %892 = shl nuw nsw i64 %.sroa.256.0.insert.ext648, 1		; visa id: 1173
  %893 = add i64 %549, %892		; visa id: 1174
  %894 = inttoptr i64 %893 to i16 addrspace(4)*		; visa id: 1175
  %895 = addrspacecast i16 addrspace(4)* %894 to i16 addrspace(1)*		; visa id: 1175
  %896 = load i16, i16 addrspace(1)* %895, align 2		; visa id: 1176
  %897 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %898 = extractvalue { i32, i32 } %897, 0		; visa id: 1178
  %899 = extractvalue { i32, i32 } %897, 1		; visa id: 1178
  %900 = insertelement <2 x i32> undef, i32 %898, i32 0		; visa id: 1185
  %901 = insertelement <2 x i32> %900, i32 %899, i32 1		; visa id: 1186
  %902 = bitcast <2 x i32> %901 to i64		; visa id: 1187
  %903 = shl i64 %902, 1		; visa id: 1189
  %904 = add i64 %.in3822, %903		; visa id: 1190
  %905 = add i64 %904, %sink_3852		; visa id: 1191
  %906 = inttoptr i64 %905 to i16 addrspace(4)*		; visa id: 1192
  %907 = addrspacecast i16 addrspace(4)* %906 to i16 addrspace(1)*		; visa id: 1192
  %908 = load i16, i16 addrspace(1)* %907, align 2		; visa id: 1193
  %909 = zext i16 %896 to i32		; visa id: 1195
  %910 = shl nuw i32 %909, 16, !spirv.Decorations !888		; visa id: 1196
  %911 = bitcast i32 %910 to float
  %912 = zext i16 %908 to i32		; visa id: 1197
  %913 = shl nuw i32 %912, 16, !spirv.Decorations !888		; visa id: 1198
  %914 = bitcast i32 %913 to float
  %915 = fmul reassoc nsz arcp contract float %911, %914, !spirv.Decorations !881
  %916 = fadd reassoc nsz arcp contract float %915, %.sroa.78.1, !spirv.Decorations !881		; visa id: 1199
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1200

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %891
; BB48 :
  %.sroa.78.2 = phi float [ %916, %891 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %165, label %917, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1201

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
; BB:
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

917:                                              ; preds = %._crit_edge.1.3
; BB50 :
  %.sroa.256.0.insert.ext653 = zext i32 %552 to i64		; visa id: 1203
  %918 = shl nuw nsw i64 %.sroa.256.0.insert.ext653, 1		; visa id: 1204
  %919 = add i64 %550, %918		; visa id: 1205
  %920 = inttoptr i64 %919 to i16 addrspace(4)*		; visa id: 1206
  %921 = addrspacecast i16 addrspace(4)* %920 to i16 addrspace(1)*		; visa id: 1206
  %922 = load i16, i16 addrspace(1)* %921, align 2		; visa id: 1207
  %923 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %924 = extractvalue { i32, i32 } %923, 0		; visa id: 1209
  %925 = extractvalue { i32, i32 } %923, 1		; visa id: 1209
  %926 = insertelement <2 x i32> undef, i32 %924, i32 0		; visa id: 1216
  %927 = insertelement <2 x i32> %926, i32 %925, i32 1		; visa id: 1217
  %928 = bitcast <2 x i32> %927 to i64		; visa id: 1218
  %929 = shl i64 %928, 1		; visa id: 1220
  %930 = add i64 %.in3822, %929		; visa id: 1221
  %931 = add i64 %930, %sink_3852		; visa id: 1222
  %932 = inttoptr i64 %931 to i16 addrspace(4)*		; visa id: 1223
  %933 = addrspacecast i16 addrspace(4)* %932 to i16 addrspace(1)*		; visa id: 1223
  %934 = load i16, i16 addrspace(1)* %933, align 2		; visa id: 1224
  %935 = zext i16 %922 to i32		; visa id: 1226
  %936 = shl nuw i32 %935, 16, !spirv.Decorations !888		; visa id: 1227
  %937 = bitcast i32 %936 to float
  %938 = zext i16 %934 to i32		; visa id: 1228
  %939 = shl nuw i32 %938, 16, !spirv.Decorations !888		; visa id: 1229
  %940 = bitcast i32 %939 to float
  %941 = fmul reassoc nsz arcp contract float %937, %940, !spirv.Decorations !881
  %942 = fadd reassoc nsz arcp contract float %941, %.sroa.142.1, !spirv.Decorations !881		; visa id: 1230
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1231

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %917
; BB51 :
  %.sroa.142.2 = phi float [ %942, %917 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %168, label %943, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1232

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
; BB:
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

943:                                              ; preds = %._crit_edge.2.3
; BB53 :
  %.sroa.256.0.insert.ext658 = zext i32 %552 to i64		; visa id: 1234
  %944 = shl nuw nsw i64 %.sroa.256.0.insert.ext658, 1		; visa id: 1235
  %945 = add i64 %551, %944		; visa id: 1236
  %946 = inttoptr i64 %945 to i16 addrspace(4)*		; visa id: 1237
  %947 = addrspacecast i16 addrspace(4)* %946 to i16 addrspace(1)*		; visa id: 1237
  %948 = load i16, i16 addrspace(1)* %947, align 2		; visa id: 1238
  %949 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %950 = extractvalue { i32, i32 } %949, 0		; visa id: 1240
  %951 = extractvalue { i32, i32 } %949, 1		; visa id: 1240
  %952 = insertelement <2 x i32> undef, i32 %950, i32 0		; visa id: 1247
  %953 = insertelement <2 x i32> %952, i32 %951, i32 1		; visa id: 1248
  %954 = bitcast <2 x i32> %953 to i64		; visa id: 1249
  %955 = shl i64 %954, 1		; visa id: 1251
  %956 = add i64 %.in3822, %955		; visa id: 1252
  %957 = add i64 %956, %sink_3852		; visa id: 1253
  %958 = inttoptr i64 %957 to i16 addrspace(4)*		; visa id: 1254
  %959 = addrspacecast i16 addrspace(4)* %958 to i16 addrspace(1)*		; visa id: 1254
  %960 = load i16, i16 addrspace(1)* %959, align 2		; visa id: 1255
  %961 = zext i16 %948 to i32		; visa id: 1257
  %962 = shl nuw i32 %961, 16, !spirv.Decorations !888		; visa id: 1258
  %963 = bitcast i32 %962 to float
  %964 = zext i16 %960 to i32		; visa id: 1259
  %965 = shl nuw i32 %964, 16, !spirv.Decorations !888		; visa id: 1260
  %966 = bitcast i32 %965 to float
  %967 = fmul reassoc nsz arcp contract float %963, %966, !spirv.Decorations !881
  %968 = fadd reassoc nsz arcp contract float %967, %.sroa.206.1, !spirv.Decorations !881		; visa id: 1261
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1262

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %943
; BB54 :
  %.sroa.206.2 = phi float [ %968, %943 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %sink_3850 = shl nsw i64 %353, 1		; visa id: 1263
  br i1 %172, label %969, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1264

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
; BB:
  br label %._crit_edge.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

969:                                              ; preds = %.preheader.3
; BB56 :
  %.sroa.256.0.insert.ext663 = zext i32 %552 to i64		; visa id: 1266
  %970 = shl nuw nsw i64 %.sroa.256.0.insert.ext663, 1		; visa id: 1267
  %971 = add i64 %548, %970		; visa id: 1268
  %972 = inttoptr i64 %971 to i16 addrspace(4)*		; visa id: 1269
  %973 = addrspacecast i16 addrspace(4)* %972 to i16 addrspace(1)*		; visa id: 1269
  %974 = load i16, i16 addrspace(1)* %973, align 2		; visa id: 1270
  %975 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %976 = extractvalue { i32, i32 } %975, 0		; visa id: 1272
  %977 = extractvalue { i32, i32 } %975, 1		; visa id: 1272
  %978 = insertelement <2 x i32> undef, i32 %976, i32 0		; visa id: 1279
  %979 = insertelement <2 x i32> %978, i32 %977, i32 1		; visa id: 1280
  %980 = bitcast <2 x i32> %979 to i64		; visa id: 1281
  %981 = shl i64 %980, 1		; visa id: 1283
  %982 = add i64 %.in3822, %981		; visa id: 1284
  %983 = add i64 %982, %sink_3850		; visa id: 1285
  %984 = inttoptr i64 %983 to i16 addrspace(4)*		; visa id: 1286
  %985 = addrspacecast i16 addrspace(4)* %984 to i16 addrspace(1)*		; visa id: 1286
  %986 = load i16, i16 addrspace(1)* %985, align 2		; visa id: 1287
  %987 = zext i16 %974 to i32		; visa id: 1289
  %988 = shl nuw i32 %987, 16, !spirv.Decorations !888		; visa id: 1290
  %989 = bitcast i32 %988 to float
  %990 = zext i16 %986 to i32		; visa id: 1291
  %991 = shl nuw i32 %990, 16, !spirv.Decorations !888		; visa id: 1292
  %992 = bitcast i32 %991 to float
  %993 = fmul reassoc nsz arcp contract float %989, %992, !spirv.Decorations !881
  %994 = fadd reassoc nsz arcp contract float %993, %.sroa.18.1, !spirv.Decorations !881		; visa id: 1293
  br label %._crit_edge.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1294

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %969
; BB57 :
  %.sroa.18.2 = phi float [ %994, %969 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  br i1 %175, label %995, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1295

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
; BB:
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

995:                                              ; preds = %._crit_edge.4
; BB59 :
  %.sroa.256.0.insert.ext668 = zext i32 %552 to i64		; visa id: 1297
  %996 = shl nuw nsw i64 %.sroa.256.0.insert.ext668, 1		; visa id: 1298
  %997 = add i64 %549, %996		; visa id: 1299
  %998 = inttoptr i64 %997 to i16 addrspace(4)*		; visa id: 1300
  %999 = addrspacecast i16 addrspace(4)* %998 to i16 addrspace(1)*		; visa id: 1300
  %1000 = load i16, i16 addrspace(1)* %999, align 2		; visa id: 1301
  %1001 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1002 = extractvalue { i32, i32 } %1001, 0		; visa id: 1303
  %1003 = extractvalue { i32, i32 } %1001, 1		; visa id: 1303
  %1004 = insertelement <2 x i32> undef, i32 %1002, i32 0		; visa id: 1310
  %1005 = insertelement <2 x i32> %1004, i32 %1003, i32 1		; visa id: 1311
  %1006 = bitcast <2 x i32> %1005 to i64		; visa id: 1312
  %1007 = shl i64 %1006, 1		; visa id: 1314
  %1008 = add i64 %.in3822, %1007		; visa id: 1315
  %1009 = add i64 %1008, %sink_3850		; visa id: 1316
  %1010 = inttoptr i64 %1009 to i16 addrspace(4)*		; visa id: 1317
  %1011 = addrspacecast i16 addrspace(4)* %1010 to i16 addrspace(1)*		; visa id: 1317
  %1012 = load i16, i16 addrspace(1)* %1011, align 2		; visa id: 1318
  %1013 = zext i16 %1000 to i32		; visa id: 1320
  %1014 = shl nuw i32 %1013, 16, !spirv.Decorations !888		; visa id: 1321
  %1015 = bitcast i32 %1014 to float
  %1016 = zext i16 %1012 to i32		; visa id: 1322
  %1017 = shl nuw i32 %1016, 16, !spirv.Decorations !888		; visa id: 1323
  %1018 = bitcast i32 %1017 to float
  %1019 = fmul reassoc nsz arcp contract float %1015, %1018, !spirv.Decorations !881
  %1020 = fadd reassoc nsz arcp contract float %1019, %.sroa.82.1, !spirv.Decorations !881		; visa id: 1324
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1325

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %995
; BB60 :
  %.sroa.82.2 = phi float [ %1020, %995 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  br i1 %178, label %1021, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1326

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
; BB:
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1021:                                             ; preds = %._crit_edge.1.4
; BB62 :
  %.sroa.256.0.insert.ext673 = zext i32 %552 to i64		; visa id: 1328
  %1022 = shl nuw nsw i64 %.sroa.256.0.insert.ext673, 1		; visa id: 1329
  %1023 = add i64 %550, %1022		; visa id: 1330
  %1024 = inttoptr i64 %1023 to i16 addrspace(4)*		; visa id: 1331
  %1025 = addrspacecast i16 addrspace(4)* %1024 to i16 addrspace(1)*		; visa id: 1331
  %1026 = load i16, i16 addrspace(1)* %1025, align 2		; visa id: 1332
  %1027 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1028 = extractvalue { i32, i32 } %1027, 0		; visa id: 1334
  %1029 = extractvalue { i32, i32 } %1027, 1		; visa id: 1334
  %1030 = insertelement <2 x i32> undef, i32 %1028, i32 0		; visa id: 1341
  %1031 = insertelement <2 x i32> %1030, i32 %1029, i32 1		; visa id: 1342
  %1032 = bitcast <2 x i32> %1031 to i64		; visa id: 1343
  %1033 = shl i64 %1032, 1		; visa id: 1345
  %1034 = add i64 %.in3822, %1033		; visa id: 1346
  %1035 = add i64 %1034, %sink_3850		; visa id: 1347
  %1036 = inttoptr i64 %1035 to i16 addrspace(4)*		; visa id: 1348
  %1037 = addrspacecast i16 addrspace(4)* %1036 to i16 addrspace(1)*		; visa id: 1348
  %1038 = load i16, i16 addrspace(1)* %1037, align 2		; visa id: 1349
  %1039 = zext i16 %1026 to i32		; visa id: 1351
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !888		; visa id: 1352
  %1041 = bitcast i32 %1040 to float
  %1042 = zext i16 %1038 to i32		; visa id: 1353
  %1043 = shl nuw i32 %1042, 16, !spirv.Decorations !888		; visa id: 1354
  %1044 = bitcast i32 %1043 to float
  %1045 = fmul reassoc nsz arcp contract float %1041, %1044, !spirv.Decorations !881
  %1046 = fadd reassoc nsz arcp contract float %1045, %.sroa.146.1, !spirv.Decorations !881		; visa id: 1355
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1356

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %1021
; BB63 :
  %.sroa.146.2 = phi float [ %1046, %1021 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  br i1 %181, label %1047, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1357

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
; BB:
  br label %.preheader.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1047:                                             ; preds = %._crit_edge.2.4
; BB65 :
  %.sroa.256.0.insert.ext678 = zext i32 %552 to i64		; visa id: 1359
  %1048 = shl nuw nsw i64 %.sroa.256.0.insert.ext678, 1		; visa id: 1360
  %1049 = add i64 %551, %1048		; visa id: 1361
  %1050 = inttoptr i64 %1049 to i16 addrspace(4)*		; visa id: 1362
  %1051 = addrspacecast i16 addrspace(4)* %1050 to i16 addrspace(1)*		; visa id: 1362
  %1052 = load i16, i16 addrspace(1)* %1051, align 2		; visa id: 1363
  %1053 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1054 = extractvalue { i32, i32 } %1053, 0		; visa id: 1365
  %1055 = extractvalue { i32, i32 } %1053, 1		; visa id: 1365
  %1056 = insertelement <2 x i32> undef, i32 %1054, i32 0		; visa id: 1372
  %1057 = insertelement <2 x i32> %1056, i32 %1055, i32 1		; visa id: 1373
  %1058 = bitcast <2 x i32> %1057 to i64		; visa id: 1374
  %1059 = shl i64 %1058, 1		; visa id: 1376
  %1060 = add i64 %.in3822, %1059		; visa id: 1377
  %1061 = add i64 %1060, %sink_3850		; visa id: 1378
  %1062 = inttoptr i64 %1061 to i16 addrspace(4)*		; visa id: 1379
  %1063 = addrspacecast i16 addrspace(4)* %1062 to i16 addrspace(1)*		; visa id: 1379
  %1064 = load i16, i16 addrspace(1)* %1063, align 2		; visa id: 1380
  %1065 = zext i16 %1052 to i32		; visa id: 1382
  %1066 = shl nuw i32 %1065, 16, !spirv.Decorations !888		; visa id: 1383
  %1067 = bitcast i32 %1066 to float
  %1068 = zext i16 %1064 to i32		; visa id: 1384
  %1069 = shl nuw i32 %1068, 16, !spirv.Decorations !888		; visa id: 1385
  %1070 = bitcast i32 %1069 to float
  %1071 = fmul reassoc nsz arcp contract float %1067, %1070, !spirv.Decorations !881
  %1072 = fadd reassoc nsz arcp contract float %1071, %.sroa.210.1, !spirv.Decorations !881		; visa id: 1386
  br label %.preheader.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1387

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %1047
; BB66 :
  %.sroa.210.2 = phi float [ %1072, %1047 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %sink_3848 = shl nsw i64 %354, 1		; visa id: 1388
  br i1 %185, label %1073, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1389

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
; BB:
  br label %._crit_edge.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1073:                                             ; preds = %.preheader.4
; BB68 :
  %.sroa.256.0.insert.ext683 = zext i32 %552 to i64		; visa id: 1391
  %1074 = shl nuw nsw i64 %.sroa.256.0.insert.ext683, 1		; visa id: 1392
  %1075 = add i64 %548, %1074		; visa id: 1393
  %1076 = inttoptr i64 %1075 to i16 addrspace(4)*		; visa id: 1394
  %1077 = addrspacecast i16 addrspace(4)* %1076 to i16 addrspace(1)*		; visa id: 1394
  %1078 = load i16, i16 addrspace(1)* %1077, align 2		; visa id: 1395
  %1079 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1080 = extractvalue { i32, i32 } %1079, 0		; visa id: 1397
  %1081 = extractvalue { i32, i32 } %1079, 1		; visa id: 1397
  %1082 = insertelement <2 x i32> undef, i32 %1080, i32 0		; visa id: 1404
  %1083 = insertelement <2 x i32> %1082, i32 %1081, i32 1		; visa id: 1405
  %1084 = bitcast <2 x i32> %1083 to i64		; visa id: 1406
  %1085 = shl i64 %1084, 1		; visa id: 1408
  %1086 = add i64 %.in3822, %1085		; visa id: 1409
  %1087 = add i64 %1086, %sink_3848		; visa id: 1410
  %1088 = inttoptr i64 %1087 to i16 addrspace(4)*		; visa id: 1411
  %1089 = addrspacecast i16 addrspace(4)* %1088 to i16 addrspace(1)*		; visa id: 1411
  %1090 = load i16, i16 addrspace(1)* %1089, align 2		; visa id: 1412
  %1091 = zext i16 %1078 to i32		; visa id: 1414
  %1092 = shl nuw i32 %1091, 16, !spirv.Decorations !888		; visa id: 1415
  %1093 = bitcast i32 %1092 to float
  %1094 = zext i16 %1090 to i32		; visa id: 1416
  %1095 = shl nuw i32 %1094, 16, !spirv.Decorations !888		; visa id: 1417
  %1096 = bitcast i32 %1095 to float
  %1097 = fmul reassoc nsz arcp contract float %1093, %1096, !spirv.Decorations !881
  %1098 = fadd reassoc nsz arcp contract float %1097, %.sroa.22.1, !spirv.Decorations !881		; visa id: 1418
  br label %._crit_edge.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1419

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %1073
; BB69 :
  %.sroa.22.2 = phi float [ %1098, %1073 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  br i1 %188, label %1099, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1420

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
; BB:
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1099:                                             ; preds = %._crit_edge.5
; BB71 :
  %.sroa.256.0.insert.ext688 = zext i32 %552 to i64		; visa id: 1422
  %1100 = shl nuw nsw i64 %.sroa.256.0.insert.ext688, 1		; visa id: 1423
  %1101 = add i64 %549, %1100		; visa id: 1424
  %1102 = inttoptr i64 %1101 to i16 addrspace(4)*		; visa id: 1425
  %1103 = addrspacecast i16 addrspace(4)* %1102 to i16 addrspace(1)*		; visa id: 1425
  %1104 = load i16, i16 addrspace(1)* %1103, align 2		; visa id: 1426
  %1105 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1106 = extractvalue { i32, i32 } %1105, 0		; visa id: 1428
  %1107 = extractvalue { i32, i32 } %1105, 1		; visa id: 1428
  %1108 = insertelement <2 x i32> undef, i32 %1106, i32 0		; visa id: 1435
  %1109 = insertelement <2 x i32> %1108, i32 %1107, i32 1		; visa id: 1436
  %1110 = bitcast <2 x i32> %1109 to i64		; visa id: 1437
  %1111 = shl i64 %1110, 1		; visa id: 1439
  %1112 = add i64 %.in3822, %1111		; visa id: 1440
  %1113 = add i64 %1112, %sink_3848		; visa id: 1441
  %1114 = inttoptr i64 %1113 to i16 addrspace(4)*		; visa id: 1442
  %1115 = addrspacecast i16 addrspace(4)* %1114 to i16 addrspace(1)*		; visa id: 1442
  %1116 = load i16, i16 addrspace(1)* %1115, align 2		; visa id: 1443
  %1117 = zext i16 %1104 to i32		; visa id: 1445
  %1118 = shl nuw i32 %1117, 16, !spirv.Decorations !888		; visa id: 1446
  %1119 = bitcast i32 %1118 to float
  %1120 = zext i16 %1116 to i32		; visa id: 1447
  %1121 = shl nuw i32 %1120, 16, !spirv.Decorations !888		; visa id: 1448
  %1122 = bitcast i32 %1121 to float
  %1123 = fmul reassoc nsz arcp contract float %1119, %1122, !spirv.Decorations !881
  %1124 = fadd reassoc nsz arcp contract float %1123, %.sroa.86.1, !spirv.Decorations !881		; visa id: 1449
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1450

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %1099
; BB72 :
  %.sroa.86.2 = phi float [ %1124, %1099 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  br i1 %191, label %1125, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1451

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
; BB:
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1125:                                             ; preds = %._crit_edge.1.5
; BB74 :
  %.sroa.256.0.insert.ext693 = zext i32 %552 to i64		; visa id: 1453
  %1126 = shl nuw nsw i64 %.sroa.256.0.insert.ext693, 1		; visa id: 1454
  %1127 = add i64 %550, %1126		; visa id: 1455
  %1128 = inttoptr i64 %1127 to i16 addrspace(4)*		; visa id: 1456
  %1129 = addrspacecast i16 addrspace(4)* %1128 to i16 addrspace(1)*		; visa id: 1456
  %1130 = load i16, i16 addrspace(1)* %1129, align 2		; visa id: 1457
  %1131 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1132 = extractvalue { i32, i32 } %1131, 0		; visa id: 1459
  %1133 = extractvalue { i32, i32 } %1131, 1		; visa id: 1459
  %1134 = insertelement <2 x i32> undef, i32 %1132, i32 0		; visa id: 1466
  %1135 = insertelement <2 x i32> %1134, i32 %1133, i32 1		; visa id: 1467
  %1136 = bitcast <2 x i32> %1135 to i64		; visa id: 1468
  %1137 = shl i64 %1136, 1		; visa id: 1470
  %1138 = add i64 %.in3822, %1137		; visa id: 1471
  %1139 = add i64 %1138, %sink_3848		; visa id: 1472
  %1140 = inttoptr i64 %1139 to i16 addrspace(4)*		; visa id: 1473
  %1141 = addrspacecast i16 addrspace(4)* %1140 to i16 addrspace(1)*		; visa id: 1473
  %1142 = load i16, i16 addrspace(1)* %1141, align 2		; visa id: 1474
  %1143 = zext i16 %1130 to i32		; visa id: 1476
  %1144 = shl nuw i32 %1143, 16, !spirv.Decorations !888		; visa id: 1477
  %1145 = bitcast i32 %1144 to float
  %1146 = zext i16 %1142 to i32		; visa id: 1478
  %1147 = shl nuw i32 %1146, 16, !spirv.Decorations !888		; visa id: 1479
  %1148 = bitcast i32 %1147 to float
  %1149 = fmul reassoc nsz arcp contract float %1145, %1148, !spirv.Decorations !881
  %1150 = fadd reassoc nsz arcp contract float %1149, %.sroa.150.1, !spirv.Decorations !881		; visa id: 1480
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1481

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %1125
; BB75 :
  %.sroa.150.2 = phi float [ %1150, %1125 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  br i1 %194, label %1151, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1482

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
; BB:
  br label %.preheader.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1151:                                             ; preds = %._crit_edge.2.5
; BB77 :
  %.sroa.256.0.insert.ext698 = zext i32 %552 to i64		; visa id: 1484
  %1152 = shl nuw nsw i64 %.sroa.256.0.insert.ext698, 1		; visa id: 1485
  %1153 = add i64 %551, %1152		; visa id: 1486
  %1154 = inttoptr i64 %1153 to i16 addrspace(4)*		; visa id: 1487
  %1155 = addrspacecast i16 addrspace(4)* %1154 to i16 addrspace(1)*		; visa id: 1487
  %1156 = load i16, i16 addrspace(1)* %1155, align 2		; visa id: 1488
  %1157 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1158 = extractvalue { i32, i32 } %1157, 0		; visa id: 1490
  %1159 = extractvalue { i32, i32 } %1157, 1		; visa id: 1490
  %1160 = insertelement <2 x i32> undef, i32 %1158, i32 0		; visa id: 1497
  %1161 = insertelement <2 x i32> %1160, i32 %1159, i32 1		; visa id: 1498
  %1162 = bitcast <2 x i32> %1161 to i64		; visa id: 1499
  %1163 = shl i64 %1162, 1		; visa id: 1501
  %1164 = add i64 %.in3822, %1163		; visa id: 1502
  %1165 = add i64 %1164, %sink_3848		; visa id: 1503
  %1166 = inttoptr i64 %1165 to i16 addrspace(4)*		; visa id: 1504
  %1167 = addrspacecast i16 addrspace(4)* %1166 to i16 addrspace(1)*		; visa id: 1504
  %1168 = load i16, i16 addrspace(1)* %1167, align 2		; visa id: 1505
  %1169 = zext i16 %1156 to i32		; visa id: 1507
  %1170 = shl nuw i32 %1169, 16, !spirv.Decorations !888		; visa id: 1508
  %1171 = bitcast i32 %1170 to float
  %1172 = zext i16 %1168 to i32		; visa id: 1509
  %1173 = shl nuw i32 %1172, 16, !spirv.Decorations !888		; visa id: 1510
  %1174 = bitcast i32 %1173 to float
  %1175 = fmul reassoc nsz arcp contract float %1171, %1174, !spirv.Decorations !881
  %1176 = fadd reassoc nsz arcp contract float %1175, %.sroa.214.1, !spirv.Decorations !881		; visa id: 1511
  br label %.preheader.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1512

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %1151
; BB78 :
  %.sroa.214.2 = phi float [ %1176, %1151 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %sink_3846 = shl nsw i64 %355, 1		; visa id: 1513
  br i1 %198, label %1177, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1514

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
; BB:
  br label %._crit_edge.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1177:                                             ; preds = %.preheader.5
; BB80 :
  %.sroa.256.0.insert.ext703 = zext i32 %552 to i64		; visa id: 1516
  %1178 = shl nuw nsw i64 %.sroa.256.0.insert.ext703, 1		; visa id: 1517
  %1179 = add i64 %548, %1178		; visa id: 1518
  %1180 = inttoptr i64 %1179 to i16 addrspace(4)*		; visa id: 1519
  %1181 = addrspacecast i16 addrspace(4)* %1180 to i16 addrspace(1)*		; visa id: 1519
  %1182 = load i16, i16 addrspace(1)* %1181, align 2		; visa id: 1520
  %1183 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1184 = extractvalue { i32, i32 } %1183, 0		; visa id: 1522
  %1185 = extractvalue { i32, i32 } %1183, 1		; visa id: 1522
  %1186 = insertelement <2 x i32> undef, i32 %1184, i32 0		; visa id: 1529
  %1187 = insertelement <2 x i32> %1186, i32 %1185, i32 1		; visa id: 1530
  %1188 = bitcast <2 x i32> %1187 to i64		; visa id: 1531
  %1189 = shl i64 %1188, 1		; visa id: 1533
  %1190 = add i64 %.in3822, %1189		; visa id: 1534
  %1191 = add i64 %1190, %sink_3846		; visa id: 1535
  %1192 = inttoptr i64 %1191 to i16 addrspace(4)*		; visa id: 1536
  %1193 = addrspacecast i16 addrspace(4)* %1192 to i16 addrspace(1)*		; visa id: 1536
  %1194 = load i16, i16 addrspace(1)* %1193, align 2		; visa id: 1537
  %1195 = zext i16 %1182 to i32		; visa id: 1539
  %1196 = shl nuw i32 %1195, 16, !spirv.Decorations !888		; visa id: 1540
  %1197 = bitcast i32 %1196 to float
  %1198 = zext i16 %1194 to i32		; visa id: 1541
  %1199 = shl nuw i32 %1198, 16, !spirv.Decorations !888		; visa id: 1542
  %1200 = bitcast i32 %1199 to float
  %1201 = fmul reassoc nsz arcp contract float %1197, %1200, !spirv.Decorations !881
  %1202 = fadd reassoc nsz arcp contract float %1201, %.sroa.26.1, !spirv.Decorations !881		; visa id: 1543
  br label %._crit_edge.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1544

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %1177
; BB81 :
  %.sroa.26.2 = phi float [ %1202, %1177 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  br i1 %201, label %1203, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1545

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
; BB:
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1203:                                             ; preds = %._crit_edge.6
; BB83 :
  %.sroa.256.0.insert.ext708 = zext i32 %552 to i64		; visa id: 1547
  %1204 = shl nuw nsw i64 %.sroa.256.0.insert.ext708, 1		; visa id: 1548
  %1205 = add i64 %549, %1204		; visa id: 1549
  %1206 = inttoptr i64 %1205 to i16 addrspace(4)*		; visa id: 1550
  %1207 = addrspacecast i16 addrspace(4)* %1206 to i16 addrspace(1)*		; visa id: 1550
  %1208 = load i16, i16 addrspace(1)* %1207, align 2		; visa id: 1551
  %1209 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1210 = extractvalue { i32, i32 } %1209, 0		; visa id: 1553
  %1211 = extractvalue { i32, i32 } %1209, 1		; visa id: 1553
  %1212 = insertelement <2 x i32> undef, i32 %1210, i32 0		; visa id: 1560
  %1213 = insertelement <2 x i32> %1212, i32 %1211, i32 1		; visa id: 1561
  %1214 = bitcast <2 x i32> %1213 to i64		; visa id: 1562
  %1215 = shl i64 %1214, 1		; visa id: 1564
  %1216 = add i64 %.in3822, %1215		; visa id: 1565
  %1217 = add i64 %1216, %sink_3846		; visa id: 1566
  %1218 = inttoptr i64 %1217 to i16 addrspace(4)*		; visa id: 1567
  %1219 = addrspacecast i16 addrspace(4)* %1218 to i16 addrspace(1)*		; visa id: 1567
  %1220 = load i16, i16 addrspace(1)* %1219, align 2		; visa id: 1568
  %1221 = zext i16 %1208 to i32		; visa id: 1570
  %1222 = shl nuw i32 %1221, 16, !spirv.Decorations !888		; visa id: 1571
  %1223 = bitcast i32 %1222 to float
  %1224 = zext i16 %1220 to i32		; visa id: 1572
  %1225 = shl nuw i32 %1224, 16, !spirv.Decorations !888		; visa id: 1573
  %1226 = bitcast i32 %1225 to float
  %1227 = fmul reassoc nsz arcp contract float %1223, %1226, !spirv.Decorations !881
  %1228 = fadd reassoc nsz arcp contract float %1227, %.sroa.90.1, !spirv.Decorations !881		; visa id: 1574
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1575

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %1203
; BB84 :
  %.sroa.90.2 = phi float [ %1228, %1203 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  br i1 %204, label %1229, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1576

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
; BB:
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1229:                                             ; preds = %._crit_edge.1.6
; BB86 :
  %.sroa.256.0.insert.ext713 = zext i32 %552 to i64		; visa id: 1578
  %1230 = shl nuw nsw i64 %.sroa.256.0.insert.ext713, 1		; visa id: 1579
  %1231 = add i64 %550, %1230		; visa id: 1580
  %1232 = inttoptr i64 %1231 to i16 addrspace(4)*		; visa id: 1581
  %1233 = addrspacecast i16 addrspace(4)* %1232 to i16 addrspace(1)*		; visa id: 1581
  %1234 = load i16, i16 addrspace(1)* %1233, align 2		; visa id: 1582
  %1235 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1236 = extractvalue { i32, i32 } %1235, 0		; visa id: 1584
  %1237 = extractvalue { i32, i32 } %1235, 1		; visa id: 1584
  %1238 = insertelement <2 x i32> undef, i32 %1236, i32 0		; visa id: 1591
  %1239 = insertelement <2 x i32> %1238, i32 %1237, i32 1		; visa id: 1592
  %1240 = bitcast <2 x i32> %1239 to i64		; visa id: 1593
  %1241 = shl i64 %1240, 1		; visa id: 1595
  %1242 = add i64 %.in3822, %1241		; visa id: 1596
  %1243 = add i64 %1242, %sink_3846		; visa id: 1597
  %1244 = inttoptr i64 %1243 to i16 addrspace(4)*		; visa id: 1598
  %1245 = addrspacecast i16 addrspace(4)* %1244 to i16 addrspace(1)*		; visa id: 1598
  %1246 = load i16, i16 addrspace(1)* %1245, align 2		; visa id: 1599
  %1247 = zext i16 %1234 to i32		; visa id: 1601
  %1248 = shl nuw i32 %1247, 16, !spirv.Decorations !888		; visa id: 1602
  %1249 = bitcast i32 %1248 to float
  %1250 = zext i16 %1246 to i32		; visa id: 1603
  %1251 = shl nuw i32 %1250, 16, !spirv.Decorations !888		; visa id: 1604
  %1252 = bitcast i32 %1251 to float
  %1253 = fmul reassoc nsz arcp contract float %1249, %1252, !spirv.Decorations !881
  %1254 = fadd reassoc nsz arcp contract float %1253, %.sroa.154.1, !spirv.Decorations !881		; visa id: 1605
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1606

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %1229
; BB87 :
  %.sroa.154.2 = phi float [ %1254, %1229 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  br i1 %207, label %1255, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1607

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
; BB:
  br label %.preheader.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1255:                                             ; preds = %._crit_edge.2.6
; BB89 :
  %.sroa.256.0.insert.ext718 = zext i32 %552 to i64		; visa id: 1609
  %1256 = shl nuw nsw i64 %.sroa.256.0.insert.ext718, 1		; visa id: 1610
  %1257 = add i64 %551, %1256		; visa id: 1611
  %1258 = inttoptr i64 %1257 to i16 addrspace(4)*		; visa id: 1612
  %1259 = addrspacecast i16 addrspace(4)* %1258 to i16 addrspace(1)*		; visa id: 1612
  %1260 = load i16, i16 addrspace(1)* %1259, align 2		; visa id: 1613
  %1261 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1262 = extractvalue { i32, i32 } %1261, 0		; visa id: 1615
  %1263 = extractvalue { i32, i32 } %1261, 1		; visa id: 1615
  %1264 = insertelement <2 x i32> undef, i32 %1262, i32 0		; visa id: 1622
  %1265 = insertelement <2 x i32> %1264, i32 %1263, i32 1		; visa id: 1623
  %1266 = bitcast <2 x i32> %1265 to i64		; visa id: 1624
  %1267 = shl i64 %1266, 1		; visa id: 1626
  %1268 = add i64 %.in3822, %1267		; visa id: 1627
  %1269 = add i64 %1268, %sink_3846		; visa id: 1628
  %1270 = inttoptr i64 %1269 to i16 addrspace(4)*		; visa id: 1629
  %1271 = addrspacecast i16 addrspace(4)* %1270 to i16 addrspace(1)*		; visa id: 1629
  %1272 = load i16, i16 addrspace(1)* %1271, align 2		; visa id: 1630
  %1273 = zext i16 %1260 to i32		; visa id: 1632
  %1274 = shl nuw i32 %1273, 16, !spirv.Decorations !888		; visa id: 1633
  %1275 = bitcast i32 %1274 to float
  %1276 = zext i16 %1272 to i32		; visa id: 1634
  %1277 = shl nuw i32 %1276, 16, !spirv.Decorations !888		; visa id: 1635
  %1278 = bitcast i32 %1277 to float
  %1279 = fmul reassoc nsz arcp contract float %1275, %1278, !spirv.Decorations !881
  %1280 = fadd reassoc nsz arcp contract float %1279, %.sroa.218.1, !spirv.Decorations !881		; visa id: 1636
  br label %.preheader.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1637

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %1255
; BB90 :
  %.sroa.218.2 = phi float [ %1280, %1255 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %sink_3844 = shl nsw i64 %356, 1		; visa id: 1638
  br i1 %211, label %1281, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1639

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
; BB:
  br label %._crit_edge.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1281:                                             ; preds = %.preheader.6
; BB92 :
  %.sroa.256.0.insert.ext723 = zext i32 %552 to i64		; visa id: 1641
  %1282 = shl nuw nsw i64 %.sroa.256.0.insert.ext723, 1		; visa id: 1642
  %1283 = add i64 %548, %1282		; visa id: 1643
  %1284 = inttoptr i64 %1283 to i16 addrspace(4)*		; visa id: 1644
  %1285 = addrspacecast i16 addrspace(4)* %1284 to i16 addrspace(1)*		; visa id: 1644
  %1286 = load i16, i16 addrspace(1)* %1285, align 2		; visa id: 1645
  %1287 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1288 = extractvalue { i32, i32 } %1287, 0		; visa id: 1647
  %1289 = extractvalue { i32, i32 } %1287, 1		; visa id: 1647
  %1290 = insertelement <2 x i32> undef, i32 %1288, i32 0		; visa id: 1654
  %1291 = insertelement <2 x i32> %1290, i32 %1289, i32 1		; visa id: 1655
  %1292 = bitcast <2 x i32> %1291 to i64		; visa id: 1656
  %1293 = shl i64 %1292, 1		; visa id: 1658
  %1294 = add i64 %.in3822, %1293		; visa id: 1659
  %1295 = add i64 %1294, %sink_3844		; visa id: 1660
  %1296 = inttoptr i64 %1295 to i16 addrspace(4)*		; visa id: 1661
  %1297 = addrspacecast i16 addrspace(4)* %1296 to i16 addrspace(1)*		; visa id: 1661
  %1298 = load i16, i16 addrspace(1)* %1297, align 2		; visa id: 1662
  %1299 = zext i16 %1286 to i32		; visa id: 1664
  %1300 = shl nuw i32 %1299, 16, !spirv.Decorations !888		; visa id: 1665
  %1301 = bitcast i32 %1300 to float
  %1302 = zext i16 %1298 to i32		; visa id: 1666
  %1303 = shl nuw i32 %1302, 16, !spirv.Decorations !888		; visa id: 1667
  %1304 = bitcast i32 %1303 to float
  %1305 = fmul reassoc nsz arcp contract float %1301, %1304, !spirv.Decorations !881
  %1306 = fadd reassoc nsz arcp contract float %1305, %.sroa.30.1, !spirv.Decorations !881		; visa id: 1668
  br label %._crit_edge.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1669

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %1281
; BB93 :
  %.sroa.30.2 = phi float [ %1306, %1281 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  br i1 %214, label %1307, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1670

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
; BB:
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1307:                                             ; preds = %._crit_edge.7
; BB95 :
  %.sroa.256.0.insert.ext728 = zext i32 %552 to i64		; visa id: 1672
  %1308 = shl nuw nsw i64 %.sroa.256.0.insert.ext728, 1		; visa id: 1673
  %1309 = add i64 %549, %1308		; visa id: 1674
  %1310 = inttoptr i64 %1309 to i16 addrspace(4)*		; visa id: 1675
  %1311 = addrspacecast i16 addrspace(4)* %1310 to i16 addrspace(1)*		; visa id: 1675
  %1312 = load i16, i16 addrspace(1)* %1311, align 2		; visa id: 1676
  %1313 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1314 = extractvalue { i32, i32 } %1313, 0		; visa id: 1678
  %1315 = extractvalue { i32, i32 } %1313, 1		; visa id: 1678
  %1316 = insertelement <2 x i32> undef, i32 %1314, i32 0		; visa id: 1685
  %1317 = insertelement <2 x i32> %1316, i32 %1315, i32 1		; visa id: 1686
  %1318 = bitcast <2 x i32> %1317 to i64		; visa id: 1687
  %1319 = shl i64 %1318, 1		; visa id: 1689
  %1320 = add i64 %.in3822, %1319		; visa id: 1690
  %1321 = add i64 %1320, %sink_3844		; visa id: 1691
  %1322 = inttoptr i64 %1321 to i16 addrspace(4)*		; visa id: 1692
  %1323 = addrspacecast i16 addrspace(4)* %1322 to i16 addrspace(1)*		; visa id: 1692
  %1324 = load i16, i16 addrspace(1)* %1323, align 2		; visa id: 1693
  %1325 = zext i16 %1312 to i32		; visa id: 1695
  %1326 = shl nuw i32 %1325, 16, !spirv.Decorations !888		; visa id: 1696
  %1327 = bitcast i32 %1326 to float
  %1328 = zext i16 %1324 to i32		; visa id: 1697
  %1329 = shl nuw i32 %1328, 16, !spirv.Decorations !888		; visa id: 1698
  %1330 = bitcast i32 %1329 to float
  %1331 = fmul reassoc nsz arcp contract float %1327, %1330, !spirv.Decorations !881
  %1332 = fadd reassoc nsz arcp contract float %1331, %.sroa.94.1, !spirv.Decorations !881		; visa id: 1699
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1700

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %1307
; BB96 :
  %.sroa.94.2 = phi float [ %1332, %1307 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  br i1 %217, label %1333, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1701

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
; BB:
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1333:                                             ; preds = %._crit_edge.1.7
; BB98 :
  %.sroa.256.0.insert.ext733 = zext i32 %552 to i64		; visa id: 1703
  %1334 = shl nuw nsw i64 %.sroa.256.0.insert.ext733, 1		; visa id: 1704
  %1335 = add i64 %550, %1334		; visa id: 1705
  %1336 = inttoptr i64 %1335 to i16 addrspace(4)*		; visa id: 1706
  %1337 = addrspacecast i16 addrspace(4)* %1336 to i16 addrspace(1)*		; visa id: 1706
  %1338 = load i16, i16 addrspace(1)* %1337, align 2		; visa id: 1707
  %1339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1340 = extractvalue { i32, i32 } %1339, 0		; visa id: 1709
  %1341 = extractvalue { i32, i32 } %1339, 1		; visa id: 1709
  %1342 = insertelement <2 x i32> undef, i32 %1340, i32 0		; visa id: 1716
  %1343 = insertelement <2 x i32> %1342, i32 %1341, i32 1		; visa id: 1717
  %1344 = bitcast <2 x i32> %1343 to i64		; visa id: 1718
  %1345 = shl i64 %1344, 1		; visa id: 1720
  %1346 = add i64 %.in3822, %1345		; visa id: 1721
  %1347 = add i64 %1346, %sink_3844		; visa id: 1722
  %1348 = inttoptr i64 %1347 to i16 addrspace(4)*		; visa id: 1723
  %1349 = addrspacecast i16 addrspace(4)* %1348 to i16 addrspace(1)*		; visa id: 1723
  %1350 = load i16, i16 addrspace(1)* %1349, align 2		; visa id: 1724
  %1351 = zext i16 %1338 to i32		; visa id: 1726
  %1352 = shl nuw i32 %1351, 16, !spirv.Decorations !888		; visa id: 1727
  %1353 = bitcast i32 %1352 to float
  %1354 = zext i16 %1350 to i32		; visa id: 1728
  %1355 = shl nuw i32 %1354, 16, !spirv.Decorations !888		; visa id: 1729
  %1356 = bitcast i32 %1355 to float
  %1357 = fmul reassoc nsz arcp contract float %1353, %1356, !spirv.Decorations !881
  %1358 = fadd reassoc nsz arcp contract float %1357, %.sroa.158.1, !spirv.Decorations !881		; visa id: 1730
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1731

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %1333
; BB99 :
  %.sroa.158.2 = phi float [ %1358, %1333 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  br i1 %220, label %1359, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1732

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
; BB:
  br label %.preheader.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1359:                                             ; preds = %._crit_edge.2.7
; BB101 :
  %.sroa.256.0.insert.ext738 = zext i32 %552 to i64		; visa id: 1734
  %1360 = shl nuw nsw i64 %.sroa.256.0.insert.ext738, 1		; visa id: 1735
  %1361 = add i64 %551, %1360		; visa id: 1736
  %1362 = inttoptr i64 %1361 to i16 addrspace(4)*		; visa id: 1737
  %1363 = addrspacecast i16 addrspace(4)* %1362 to i16 addrspace(1)*		; visa id: 1737
  %1364 = load i16, i16 addrspace(1)* %1363, align 2		; visa id: 1738
  %1365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1366 = extractvalue { i32, i32 } %1365, 0		; visa id: 1740
  %1367 = extractvalue { i32, i32 } %1365, 1		; visa id: 1740
  %1368 = insertelement <2 x i32> undef, i32 %1366, i32 0		; visa id: 1747
  %1369 = insertelement <2 x i32> %1368, i32 %1367, i32 1		; visa id: 1748
  %1370 = bitcast <2 x i32> %1369 to i64		; visa id: 1749
  %1371 = shl i64 %1370, 1		; visa id: 1751
  %1372 = add i64 %.in3822, %1371		; visa id: 1752
  %1373 = add i64 %1372, %sink_3844		; visa id: 1753
  %1374 = inttoptr i64 %1373 to i16 addrspace(4)*		; visa id: 1754
  %1375 = addrspacecast i16 addrspace(4)* %1374 to i16 addrspace(1)*		; visa id: 1754
  %1376 = load i16, i16 addrspace(1)* %1375, align 2		; visa id: 1755
  %1377 = zext i16 %1364 to i32		; visa id: 1757
  %1378 = shl nuw i32 %1377, 16, !spirv.Decorations !888		; visa id: 1758
  %1379 = bitcast i32 %1378 to float
  %1380 = zext i16 %1376 to i32		; visa id: 1759
  %1381 = shl nuw i32 %1380, 16, !spirv.Decorations !888		; visa id: 1760
  %1382 = bitcast i32 %1381 to float
  %1383 = fmul reassoc nsz arcp contract float %1379, %1382, !spirv.Decorations !881
  %1384 = fadd reassoc nsz arcp contract float %1383, %.sroa.222.1, !spirv.Decorations !881		; visa id: 1761
  br label %.preheader.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1762

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %1359
; BB102 :
  %.sroa.222.2 = phi float [ %1384, %1359 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %sink_3842 = shl nsw i64 %357, 1		; visa id: 1763
  br i1 %224, label %1385, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1764

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
; BB:
  br label %._crit_edge.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1385:                                             ; preds = %.preheader.7
; BB104 :
  %.sroa.256.0.insert.ext743 = zext i32 %552 to i64		; visa id: 1766
  %1386 = shl nuw nsw i64 %.sroa.256.0.insert.ext743, 1		; visa id: 1767
  %1387 = add i64 %548, %1386		; visa id: 1768
  %1388 = inttoptr i64 %1387 to i16 addrspace(4)*		; visa id: 1769
  %1389 = addrspacecast i16 addrspace(4)* %1388 to i16 addrspace(1)*		; visa id: 1769
  %1390 = load i16, i16 addrspace(1)* %1389, align 2		; visa id: 1770
  %1391 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1392 = extractvalue { i32, i32 } %1391, 0		; visa id: 1772
  %1393 = extractvalue { i32, i32 } %1391, 1		; visa id: 1772
  %1394 = insertelement <2 x i32> undef, i32 %1392, i32 0		; visa id: 1779
  %1395 = insertelement <2 x i32> %1394, i32 %1393, i32 1		; visa id: 1780
  %1396 = bitcast <2 x i32> %1395 to i64		; visa id: 1781
  %1397 = shl i64 %1396, 1		; visa id: 1783
  %1398 = add i64 %.in3822, %1397		; visa id: 1784
  %1399 = add i64 %1398, %sink_3842		; visa id: 1785
  %1400 = inttoptr i64 %1399 to i16 addrspace(4)*		; visa id: 1786
  %1401 = addrspacecast i16 addrspace(4)* %1400 to i16 addrspace(1)*		; visa id: 1786
  %1402 = load i16, i16 addrspace(1)* %1401, align 2		; visa id: 1787
  %1403 = zext i16 %1390 to i32		; visa id: 1789
  %1404 = shl nuw i32 %1403, 16, !spirv.Decorations !888		; visa id: 1790
  %1405 = bitcast i32 %1404 to float
  %1406 = zext i16 %1402 to i32		; visa id: 1791
  %1407 = shl nuw i32 %1406, 16, !spirv.Decorations !888		; visa id: 1792
  %1408 = bitcast i32 %1407 to float
  %1409 = fmul reassoc nsz arcp contract float %1405, %1408, !spirv.Decorations !881
  %1410 = fadd reassoc nsz arcp contract float %1409, %.sroa.34.1, !spirv.Decorations !881		; visa id: 1793
  br label %._crit_edge.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1794

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %1385
; BB105 :
  %.sroa.34.2 = phi float [ %1410, %1385 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  br i1 %227, label %1411, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1795

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
; BB:
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1411:                                             ; preds = %._crit_edge.8
; BB107 :
  %.sroa.256.0.insert.ext748 = zext i32 %552 to i64		; visa id: 1797
  %1412 = shl nuw nsw i64 %.sroa.256.0.insert.ext748, 1		; visa id: 1798
  %1413 = add i64 %549, %1412		; visa id: 1799
  %1414 = inttoptr i64 %1413 to i16 addrspace(4)*		; visa id: 1800
  %1415 = addrspacecast i16 addrspace(4)* %1414 to i16 addrspace(1)*		; visa id: 1800
  %1416 = load i16, i16 addrspace(1)* %1415, align 2		; visa id: 1801
  %1417 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1418 = extractvalue { i32, i32 } %1417, 0		; visa id: 1803
  %1419 = extractvalue { i32, i32 } %1417, 1		; visa id: 1803
  %1420 = insertelement <2 x i32> undef, i32 %1418, i32 0		; visa id: 1810
  %1421 = insertelement <2 x i32> %1420, i32 %1419, i32 1		; visa id: 1811
  %1422 = bitcast <2 x i32> %1421 to i64		; visa id: 1812
  %1423 = shl i64 %1422, 1		; visa id: 1814
  %1424 = add i64 %.in3822, %1423		; visa id: 1815
  %1425 = add i64 %1424, %sink_3842		; visa id: 1816
  %1426 = inttoptr i64 %1425 to i16 addrspace(4)*		; visa id: 1817
  %1427 = addrspacecast i16 addrspace(4)* %1426 to i16 addrspace(1)*		; visa id: 1817
  %1428 = load i16, i16 addrspace(1)* %1427, align 2		; visa id: 1818
  %1429 = zext i16 %1416 to i32		; visa id: 1820
  %1430 = shl nuw i32 %1429, 16, !spirv.Decorations !888		; visa id: 1821
  %1431 = bitcast i32 %1430 to float
  %1432 = zext i16 %1428 to i32		; visa id: 1822
  %1433 = shl nuw i32 %1432, 16, !spirv.Decorations !888		; visa id: 1823
  %1434 = bitcast i32 %1433 to float
  %1435 = fmul reassoc nsz arcp contract float %1431, %1434, !spirv.Decorations !881
  %1436 = fadd reassoc nsz arcp contract float %1435, %.sroa.98.1, !spirv.Decorations !881		; visa id: 1824
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1825

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %1411
; BB108 :
  %.sroa.98.2 = phi float [ %1436, %1411 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  br i1 %230, label %1437, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1826

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
; BB:
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1437:                                             ; preds = %._crit_edge.1.8
; BB110 :
  %.sroa.256.0.insert.ext753 = zext i32 %552 to i64		; visa id: 1828
  %1438 = shl nuw nsw i64 %.sroa.256.0.insert.ext753, 1		; visa id: 1829
  %1439 = add i64 %550, %1438		; visa id: 1830
  %1440 = inttoptr i64 %1439 to i16 addrspace(4)*		; visa id: 1831
  %1441 = addrspacecast i16 addrspace(4)* %1440 to i16 addrspace(1)*		; visa id: 1831
  %1442 = load i16, i16 addrspace(1)* %1441, align 2		; visa id: 1832
  %1443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1444 = extractvalue { i32, i32 } %1443, 0		; visa id: 1834
  %1445 = extractvalue { i32, i32 } %1443, 1		; visa id: 1834
  %1446 = insertelement <2 x i32> undef, i32 %1444, i32 0		; visa id: 1841
  %1447 = insertelement <2 x i32> %1446, i32 %1445, i32 1		; visa id: 1842
  %1448 = bitcast <2 x i32> %1447 to i64		; visa id: 1843
  %1449 = shl i64 %1448, 1		; visa id: 1845
  %1450 = add i64 %.in3822, %1449		; visa id: 1846
  %1451 = add i64 %1450, %sink_3842		; visa id: 1847
  %1452 = inttoptr i64 %1451 to i16 addrspace(4)*		; visa id: 1848
  %1453 = addrspacecast i16 addrspace(4)* %1452 to i16 addrspace(1)*		; visa id: 1848
  %1454 = load i16, i16 addrspace(1)* %1453, align 2		; visa id: 1849
  %1455 = zext i16 %1442 to i32		; visa id: 1851
  %1456 = shl nuw i32 %1455, 16, !spirv.Decorations !888		; visa id: 1852
  %1457 = bitcast i32 %1456 to float
  %1458 = zext i16 %1454 to i32		; visa id: 1853
  %1459 = shl nuw i32 %1458, 16, !spirv.Decorations !888		; visa id: 1854
  %1460 = bitcast i32 %1459 to float
  %1461 = fmul reassoc nsz arcp contract float %1457, %1460, !spirv.Decorations !881
  %1462 = fadd reassoc nsz arcp contract float %1461, %.sroa.162.1, !spirv.Decorations !881		; visa id: 1855
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1856

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %1437
; BB111 :
  %.sroa.162.2 = phi float [ %1462, %1437 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  br i1 %233, label %1463, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1857

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
; BB:
  br label %.preheader.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1463:                                             ; preds = %._crit_edge.2.8
; BB113 :
  %.sroa.256.0.insert.ext758 = zext i32 %552 to i64		; visa id: 1859
  %1464 = shl nuw nsw i64 %.sroa.256.0.insert.ext758, 1		; visa id: 1860
  %1465 = add i64 %551, %1464		; visa id: 1861
  %1466 = inttoptr i64 %1465 to i16 addrspace(4)*		; visa id: 1862
  %1467 = addrspacecast i16 addrspace(4)* %1466 to i16 addrspace(1)*		; visa id: 1862
  %1468 = load i16, i16 addrspace(1)* %1467, align 2		; visa id: 1863
  %1469 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1470 = extractvalue { i32, i32 } %1469, 0		; visa id: 1865
  %1471 = extractvalue { i32, i32 } %1469, 1		; visa id: 1865
  %1472 = insertelement <2 x i32> undef, i32 %1470, i32 0		; visa id: 1872
  %1473 = insertelement <2 x i32> %1472, i32 %1471, i32 1		; visa id: 1873
  %1474 = bitcast <2 x i32> %1473 to i64		; visa id: 1874
  %1475 = shl i64 %1474, 1		; visa id: 1876
  %1476 = add i64 %.in3822, %1475		; visa id: 1877
  %1477 = add i64 %1476, %sink_3842		; visa id: 1878
  %1478 = inttoptr i64 %1477 to i16 addrspace(4)*		; visa id: 1879
  %1479 = addrspacecast i16 addrspace(4)* %1478 to i16 addrspace(1)*		; visa id: 1879
  %1480 = load i16, i16 addrspace(1)* %1479, align 2		; visa id: 1880
  %1481 = zext i16 %1468 to i32		; visa id: 1882
  %1482 = shl nuw i32 %1481, 16, !spirv.Decorations !888		; visa id: 1883
  %1483 = bitcast i32 %1482 to float
  %1484 = zext i16 %1480 to i32		; visa id: 1884
  %1485 = shl nuw i32 %1484, 16, !spirv.Decorations !888		; visa id: 1885
  %1486 = bitcast i32 %1485 to float
  %1487 = fmul reassoc nsz arcp contract float %1483, %1486, !spirv.Decorations !881
  %1488 = fadd reassoc nsz arcp contract float %1487, %.sroa.226.1, !spirv.Decorations !881		; visa id: 1886
  br label %.preheader.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1887

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %1463
; BB114 :
  %.sroa.226.2 = phi float [ %1488, %1463 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %sink_3840 = shl nsw i64 %358, 1		; visa id: 1888
  br i1 %237, label %1489, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1889

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
; BB:
  br label %._crit_edge.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1489:                                             ; preds = %.preheader.8
; BB116 :
  %.sroa.256.0.insert.ext763 = zext i32 %552 to i64		; visa id: 1891
  %1490 = shl nuw nsw i64 %.sroa.256.0.insert.ext763, 1		; visa id: 1892
  %1491 = add i64 %548, %1490		; visa id: 1893
  %1492 = inttoptr i64 %1491 to i16 addrspace(4)*		; visa id: 1894
  %1493 = addrspacecast i16 addrspace(4)* %1492 to i16 addrspace(1)*		; visa id: 1894
  %1494 = load i16, i16 addrspace(1)* %1493, align 2		; visa id: 1895
  %1495 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1496 = extractvalue { i32, i32 } %1495, 0		; visa id: 1897
  %1497 = extractvalue { i32, i32 } %1495, 1		; visa id: 1897
  %1498 = insertelement <2 x i32> undef, i32 %1496, i32 0		; visa id: 1904
  %1499 = insertelement <2 x i32> %1498, i32 %1497, i32 1		; visa id: 1905
  %1500 = bitcast <2 x i32> %1499 to i64		; visa id: 1906
  %1501 = shl i64 %1500, 1		; visa id: 1908
  %1502 = add i64 %.in3822, %1501		; visa id: 1909
  %1503 = add i64 %1502, %sink_3840		; visa id: 1910
  %1504 = inttoptr i64 %1503 to i16 addrspace(4)*		; visa id: 1911
  %1505 = addrspacecast i16 addrspace(4)* %1504 to i16 addrspace(1)*		; visa id: 1911
  %1506 = load i16, i16 addrspace(1)* %1505, align 2		; visa id: 1912
  %1507 = zext i16 %1494 to i32		; visa id: 1914
  %1508 = shl nuw i32 %1507, 16, !spirv.Decorations !888		; visa id: 1915
  %1509 = bitcast i32 %1508 to float
  %1510 = zext i16 %1506 to i32		; visa id: 1916
  %1511 = shl nuw i32 %1510, 16, !spirv.Decorations !888		; visa id: 1917
  %1512 = bitcast i32 %1511 to float
  %1513 = fmul reassoc nsz arcp contract float %1509, %1512, !spirv.Decorations !881
  %1514 = fadd reassoc nsz arcp contract float %1513, %.sroa.38.1, !spirv.Decorations !881		; visa id: 1918
  br label %._crit_edge.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1919

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %1489
; BB117 :
  %.sroa.38.2 = phi float [ %1514, %1489 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  br i1 %240, label %1515, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1920

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
; BB:
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1515:                                             ; preds = %._crit_edge.9
; BB119 :
  %.sroa.256.0.insert.ext768 = zext i32 %552 to i64		; visa id: 1922
  %1516 = shl nuw nsw i64 %.sroa.256.0.insert.ext768, 1		; visa id: 1923
  %1517 = add i64 %549, %1516		; visa id: 1924
  %1518 = inttoptr i64 %1517 to i16 addrspace(4)*		; visa id: 1925
  %1519 = addrspacecast i16 addrspace(4)* %1518 to i16 addrspace(1)*		; visa id: 1925
  %1520 = load i16, i16 addrspace(1)* %1519, align 2		; visa id: 1926
  %1521 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1522 = extractvalue { i32, i32 } %1521, 0		; visa id: 1928
  %1523 = extractvalue { i32, i32 } %1521, 1		; visa id: 1928
  %1524 = insertelement <2 x i32> undef, i32 %1522, i32 0		; visa id: 1935
  %1525 = insertelement <2 x i32> %1524, i32 %1523, i32 1		; visa id: 1936
  %1526 = bitcast <2 x i32> %1525 to i64		; visa id: 1937
  %1527 = shl i64 %1526, 1		; visa id: 1939
  %1528 = add i64 %.in3822, %1527		; visa id: 1940
  %1529 = add i64 %1528, %sink_3840		; visa id: 1941
  %1530 = inttoptr i64 %1529 to i16 addrspace(4)*		; visa id: 1942
  %1531 = addrspacecast i16 addrspace(4)* %1530 to i16 addrspace(1)*		; visa id: 1942
  %1532 = load i16, i16 addrspace(1)* %1531, align 2		; visa id: 1943
  %1533 = zext i16 %1520 to i32		; visa id: 1945
  %1534 = shl nuw i32 %1533, 16, !spirv.Decorations !888		; visa id: 1946
  %1535 = bitcast i32 %1534 to float
  %1536 = zext i16 %1532 to i32		; visa id: 1947
  %1537 = shl nuw i32 %1536, 16, !spirv.Decorations !888		; visa id: 1948
  %1538 = bitcast i32 %1537 to float
  %1539 = fmul reassoc nsz arcp contract float %1535, %1538, !spirv.Decorations !881
  %1540 = fadd reassoc nsz arcp contract float %1539, %.sroa.102.1, !spirv.Decorations !881		; visa id: 1949
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1950

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %1515
; BB120 :
  %.sroa.102.2 = phi float [ %1540, %1515 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  br i1 %243, label %1541, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1951

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
; BB:
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1541:                                             ; preds = %._crit_edge.1.9
; BB122 :
  %.sroa.256.0.insert.ext773 = zext i32 %552 to i64		; visa id: 1953
  %1542 = shl nuw nsw i64 %.sroa.256.0.insert.ext773, 1		; visa id: 1954
  %1543 = add i64 %550, %1542		; visa id: 1955
  %1544 = inttoptr i64 %1543 to i16 addrspace(4)*		; visa id: 1956
  %1545 = addrspacecast i16 addrspace(4)* %1544 to i16 addrspace(1)*		; visa id: 1956
  %1546 = load i16, i16 addrspace(1)* %1545, align 2		; visa id: 1957
  %1547 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1548 = extractvalue { i32, i32 } %1547, 0		; visa id: 1959
  %1549 = extractvalue { i32, i32 } %1547, 1		; visa id: 1959
  %1550 = insertelement <2 x i32> undef, i32 %1548, i32 0		; visa id: 1966
  %1551 = insertelement <2 x i32> %1550, i32 %1549, i32 1		; visa id: 1967
  %1552 = bitcast <2 x i32> %1551 to i64		; visa id: 1968
  %1553 = shl i64 %1552, 1		; visa id: 1970
  %1554 = add i64 %.in3822, %1553		; visa id: 1971
  %1555 = add i64 %1554, %sink_3840		; visa id: 1972
  %1556 = inttoptr i64 %1555 to i16 addrspace(4)*		; visa id: 1973
  %1557 = addrspacecast i16 addrspace(4)* %1556 to i16 addrspace(1)*		; visa id: 1973
  %1558 = load i16, i16 addrspace(1)* %1557, align 2		; visa id: 1974
  %1559 = zext i16 %1546 to i32		; visa id: 1976
  %1560 = shl nuw i32 %1559, 16, !spirv.Decorations !888		; visa id: 1977
  %1561 = bitcast i32 %1560 to float
  %1562 = zext i16 %1558 to i32		; visa id: 1978
  %1563 = shl nuw i32 %1562, 16, !spirv.Decorations !888		; visa id: 1979
  %1564 = bitcast i32 %1563 to float
  %1565 = fmul reassoc nsz arcp contract float %1561, %1564, !spirv.Decorations !881
  %1566 = fadd reassoc nsz arcp contract float %1565, %.sroa.166.1, !spirv.Decorations !881		; visa id: 1980
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1981

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %1541
; BB123 :
  %.sroa.166.2 = phi float [ %1566, %1541 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  br i1 %246, label %1567, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1982

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
; BB:
  br label %.preheader.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1567:                                             ; preds = %._crit_edge.2.9
; BB125 :
  %.sroa.256.0.insert.ext778 = zext i32 %552 to i64		; visa id: 1984
  %1568 = shl nuw nsw i64 %.sroa.256.0.insert.ext778, 1		; visa id: 1985
  %1569 = add i64 %551, %1568		; visa id: 1986
  %1570 = inttoptr i64 %1569 to i16 addrspace(4)*		; visa id: 1987
  %1571 = addrspacecast i16 addrspace(4)* %1570 to i16 addrspace(1)*		; visa id: 1987
  %1572 = load i16, i16 addrspace(1)* %1571, align 2		; visa id: 1988
  %1573 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1574 = extractvalue { i32, i32 } %1573, 0		; visa id: 1990
  %1575 = extractvalue { i32, i32 } %1573, 1		; visa id: 1990
  %1576 = insertelement <2 x i32> undef, i32 %1574, i32 0		; visa id: 1997
  %1577 = insertelement <2 x i32> %1576, i32 %1575, i32 1		; visa id: 1998
  %1578 = bitcast <2 x i32> %1577 to i64		; visa id: 1999
  %1579 = shl i64 %1578, 1		; visa id: 2001
  %1580 = add i64 %.in3822, %1579		; visa id: 2002
  %1581 = add i64 %1580, %sink_3840		; visa id: 2003
  %1582 = inttoptr i64 %1581 to i16 addrspace(4)*		; visa id: 2004
  %1583 = addrspacecast i16 addrspace(4)* %1582 to i16 addrspace(1)*		; visa id: 2004
  %1584 = load i16, i16 addrspace(1)* %1583, align 2		; visa id: 2005
  %1585 = zext i16 %1572 to i32		; visa id: 2007
  %1586 = shl nuw i32 %1585, 16, !spirv.Decorations !888		; visa id: 2008
  %1587 = bitcast i32 %1586 to float
  %1588 = zext i16 %1584 to i32		; visa id: 2009
  %1589 = shl nuw i32 %1588, 16, !spirv.Decorations !888		; visa id: 2010
  %1590 = bitcast i32 %1589 to float
  %1591 = fmul reassoc nsz arcp contract float %1587, %1590, !spirv.Decorations !881
  %1592 = fadd reassoc nsz arcp contract float %1591, %.sroa.230.1, !spirv.Decorations !881		; visa id: 2011
  br label %.preheader.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2012

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %1567
; BB126 :
  %.sroa.230.2 = phi float [ %1592, %1567 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %sink_3838 = shl nsw i64 %359, 1		; visa id: 2013
  br i1 %250, label %1593, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2014

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
; BB:
  br label %._crit_edge.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1593:                                             ; preds = %.preheader.9
; BB128 :
  %.sroa.256.0.insert.ext783 = zext i32 %552 to i64		; visa id: 2016
  %1594 = shl nuw nsw i64 %.sroa.256.0.insert.ext783, 1		; visa id: 2017
  %1595 = add i64 %548, %1594		; visa id: 2018
  %1596 = inttoptr i64 %1595 to i16 addrspace(4)*		; visa id: 2019
  %1597 = addrspacecast i16 addrspace(4)* %1596 to i16 addrspace(1)*		; visa id: 2019
  %1598 = load i16, i16 addrspace(1)* %1597, align 2		; visa id: 2020
  %1599 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1600 = extractvalue { i32, i32 } %1599, 0		; visa id: 2022
  %1601 = extractvalue { i32, i32 } %1599, 1		; visa id: 2022
  %1602 = insertelement <2 x i32> undef, i32 %1600, i32 0		; visa id: 2029
  %1603 = insertelement <2 x i32> %1602, i32 %1601, i32 1		; visa id: 2030
  %1604 = bitcast <2 x i32> %1603 to i64		; visa id: 2031
  %1605 = shl i64 %1604, 1		; visa id: 2033
  %1606 = add i64 %.in3822, %1605		; visa id: 2034
  %1607 = add i64 %1606, %sink_3838		; visa id: 2035
  %1608 = inttoptr i64 %1607 to i16 addrspace(4)*		; visa id: 2036
  %1609 = addrspacecast i16 addrspace(4)* %1608 to i16 addrspace(1)*		; visa id: 2036
  %1610 = load i16, i16 addrspace(1)* %1609, align 2		; visa id: 2037
  %1611 = zext i16 %1598 to i32		; visa id: 2039
  %1612 = shl nuw i32 %1611, 16, !spirv.Decorations !888		; visa id: 2040
  %1613 = bitcast i32 %1612 to float
  %1614 = zext i16 %1610 to i32		; visa id: 2041
  %1615 = shl nuw i32 %1614, 16, !spirv.Decorations !888		; visa id: 2042
  %1616 = bitcast i32 %1615 to float
  %1617 = fmul reassoc nsz arcp contract float %1613, %1616, !spirv.Decorations !881
  %1618 = fadd reassoc nsz arcp contract float %1617, %.sroa.42.1, !spirv.Decorations !881		; visa id: 2043
  br label %._crit_edge.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2044

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %1593
; BB129 :
  %.sroa.42.2 = phi float [ %1618, %1593 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  br i1 %253, label %1619, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2045

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
; BB:
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1619:                                             ; preds = %._crit_edge.10
; BB131 :
  %.sroa.256.0.insert.ext788 = zext i32 %552 to i64		; visa id: 2047
  %1620 = shl nuw nsw i64 %.sroa.256.0.insert.ext788, 1		; visa id: 2048
  %1621 = add i64 %549, %1620		; visa id: 2049
  %1622 = inttoptr i64 %1621 to i16 addrspace(4)*		; visa id: 2050
  %1623 = addrspacecast i16 addrspace(4)* %1622 to i16 addrspace(1)*		; visa id: 2050
  %1624 = load i16, i16 addrspace(1)* %1623, align 2		; visa id: 2051
  %1625 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1626 = extractvalue { i32, i32 } %1625, 0		; visa id: 2053
  %1627 = extractvalue { i32, i32 } %1625, 1		; visa id: 2053
  %1628 = insertelement <2 x i32> undef, i32 %1626, i32 0		; visa id: 2060
  %1629 = insertelement <2 x i32> %1628, i32 %1627, i32 1		; visa id: 2061
  %1630 = bitcast <2 x i32> %1629 to i64		; visa id: 2062
  %1631 = shl i64 %1630, 1		; visa id: 2064
  %1632 = add i64 %.in3822, %1631		; visa id: 2065
  %1633 = add i64 %1632, %sink_3838		; visa id: 2066
  %1634 = inttoptr i64 %1633 to i16 addrspace(4)*		; visa id: 2067
  %1635 = addrspacecast i16 addrspace(4)* %1634 to i16 addrspace(1)*		; visa id: 2067
  %1636 = load i16, i16 addrspace(1)* %1635, align 2		; visa id: 2068
  %1637 = zext i16 %1624 to i32		; visa id: 2070
  %1638 = shl nuw i32 %1637, 16, !spirv.Decorations !888		; visa id: 2071
  %1639 = bitcast i32 %1638 to float
  %1640 = zext i16 %1636 to i32		; visa id: 2072
  %1641 = shl nuw i32 %1640, 16, !spirv.Decorations !888		; visa id: 2073
  %1642 = bitcast i32 %1641 to float
  %1643 = fmul reassoc nsz arcp contract float %1639, %1642, !spirv.Decorations !881
  %1644 = fadd reassoc nsz arcp contract float %1643, %.sroa.106.1, !spirv.Decorations !881		; visa id: 2074
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2075

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %1619
; BB132 :
  %.sroa.106.2 = phi float [ %1644, %1619 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  br i1 %256, label %1645, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2076

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
; BB:
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1645:                                             ; preds = %._crit_edge.1.10
; BB134 :
  %.sroa.256.0.insert.ext793 = zext i32 %552 to i64		; visa id: 2078
  %1646 = shl nuw nsw i64 %.sroa.256.0.insert.ext793, 1		; visa id: 2079
  %1647 = add i64 %550, %1646		; visa id: 2080
  %1648 = inttoptr i64 %1647 to i16 addrspace(4)*		; visa id: 2081
  %1649 = addrspacecast i16 addrspace(4)* %1648 to i16 addrspace(1)*		; visa id: 2081
  %1650 = load i16, i16 addrspace(1)* %1649, align 2		; visa id: 2082
  %1651 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1652 = extractvalue { i32, i32 } %1651, 0		; visa id: 2084
  %1653 = extractvalue { i32, i32 } %1651, 1		; visa id: 2084
  %1654 = insertelement <2 x i32> undef, i32 %1652, i32 0		; visa id: 2091
  %1655 = insertelement <2 x i32> %1654, i32 %1653, i32 1		; visa id: 2092
  %1656 = bitcast <2 x i32> %1655 to i64		; visa id: 2093
  %1657 = shl i64 %1656, 1		; visa id: 2095
  %1658 = add i64 %.in3822, %1657		; visa id: 2096
  %1659 = add i64 %1658, %sink_3838		; visa id: 2097
  %1660 = inttoptr i64 %1659 to i16 addrspace(4)*		; visa id: 2098
  %1661 = addrspacecast i16 addrspace(4)* %1660 to i16 addrspace(1)*		; visa id: 2098
  %1662 = load i16, i16 addrspace(1)* %1661, align 2		; visa id: 2099
  %1663 = zext i16 %1650 to i32		; visa id: 2101
  %1664 = shl nuw i32 %1663, 16, !spirv.Decorations !888		; visa id: 2102
  %1665 = bitcast i32 %1664 to float
  %1666 = zext i16 %1662 to i32		; visa id: 2103
  %1667 = shl nuw i32 %1666, 16, !spirv.Decorations !888		; visa id: 2104
  %1668 = bitcast i32 %1667 to float
  %1669 = fmul reassoc nsz arcp contract float %1665, %1668, !spirv.Decorations !881
  %1670 = fadd reassoc nsz arcp contract float %1669, %.sroa.170.1, !spirv.Decorations !881		; visa id: 2105
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2106

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %1645
; BB135 :
  %.sroa.170.2 = phi float [ %1670, %1645 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  br i1 %259, label %1671, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2107

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
; BB:
  br label %.preheader.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1671:                                             ; preds = %._crit_edge.2.10
; BB137 :
  %.sroa.256.0.insert.ext798 = zext i32 %552 to i64		; visa id: 2109
  %1672 = shl nuw nsw i64 %.sroa.256.0.insert.ext798, 1		; visa id: 2110
  %1673 = add i64 %551, %1672		; visa id: 2111
  %1674 = inttoptr i64 %1673 to i16 addrspace(4)*		; visa id: 2112
  %1675 = addrspacecast i16 addrspace(4)* %1674 to i16 addrspace(1)*		; visa id: 2112
  %1676 = load i16, i16 addrspace(1)* %1675, align 2		; visa id: 2113
  %1677 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1678 = extractvalue { i32, i32 } %1677, 0		; visa id: 2115
  %1679 = extractvalue { i32, i32 } %1677, 1		; visa id: 2115
  %1680 = insertelement <2 x i32> undef, i32 %1678, i32 0		; visa id: 2122
  %1681 = insertelement <2 x i32> %1680, i32 %1679, i32 1		; visa id: 2123
  %1682 = bitcast <2 x i32> %1681 to i64		; visa id: 2124
  %1683 = shl i64 %1682, 1		; visa id: 2126
  %1684 = add i64 %.in3822, %1683		; visa id: 2127
  %1685 = add i64 %1684, %sink_3838		; visa id: 2128
  %1686 = inttoptr i64 %1685 to i16 addrspace(4)*		; visa id: 2129
  %1687 = addrspacecast i16 addrspace(4)* %1686 to i16 addrspace(1)*		; visa id: 2129
  %1688 = load i16, i16 addrspace(1)* %1687, align 2		; visa id: 2130
  %1689 = zext i16 %1676 to i32		; visa id: 2132
  %1690 = shl nuw i32 %1689, 16, !spirv.Decorations !888		; visa id: 2133
  %1691 = bitcast i32 %1690 to float
  %1692 = zext i16 %1688 to i32		; visa id: 2134
  %1693 = shl nuw i32 %1692, 16, !spirv.Decorations !888		; visa id: 2135
  %1694 = bitcast i32 %1693 to float
  %1695 = fmul reassoc nsz arcp contract float %1691, %1694, !spirv.Decorations !881
  %1696 = fadd reassoc nsz arcp contract float %1695, %.sroa.234.1, !spirv.Decorations !881		; visa id: 2136
  br label %.preheader.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2137

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %1671
; BB138 :
  %.sroa.234.2 = phi float [ %1696, %1671 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %sink_3836 = shl nsw i64 %360, 1		; visa id: 2138
  br i1 %263, label %1697, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2139

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
; BB:
  br label %._crit_edge.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1697:                                             ; preds = %.preheader.10
; BB140 :
  %.sroa.256.0.insert.ext803 = zext i32 %552 to i64		; visa id: 2141
  %1698 = shl nuw nsw i64 %.sroa.256.0.insert.ext803, 1		; visa id: 2142
  %1699 = add i64 %548, %1698		; visa id: 2143
  %1700 = inttoptr i64 %1699 to i16 addrspace(4)*		; visa id: 2144
  %1701 = addrspacecast i16 addrspace(4)* %1700 to i16 addrspace(1)*		; visa id: 2144
  %1702 = load i16, i16 addrspace(1)* %1701, align 2		; visa id: 2145
  %1703 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1704 = extractvalue { i32, i32 } %1703, 0		; visa id: 2147
  %1705 = extractvalue { i32, i32 } %1703, 1		; visa id: 2147
  %1706 = insertelement <2 x i32> undef, i32 %1704, i32 0		; visa id: 2154
  %1707 = insertelement <2 x i32> %1706, i32 %1705, i32 1		; visa id: 2155
  %1708 = bitcast <2 x i32> %1707 to i64		; visa id: 2156
  %1709 = shl i64 %1708, 1		; visa id: 2158
  %1710 = add i64 %.in3822, %1709		; visa id: 2159
  %1711 = add i64 %1710, %sink_3836		; visa id: 2160
  %1712 = inttoptr i64 %1711 to i16 addrspace(4)*		; visa id: 2161
  %1713 = addrspacecast i16 addrspace(4)* %1712 to i16 addrspace(1)*		; visa id: 2161
  %1714 = load i16, i16 addrspace(1)* %1713, align 2		; visa id: 2162
  %1715 = zext i16 %1702 to i32		; visa id: 2164
  %1716 = shl nuw i32 %1715, 16, !spirv.Decorations !888		; visa id: 2165
  %1717 = bitcast i32 %1716 to float
  %1718 = zext i16 %1714 to i32		; visa id: 2166
  %1719 = shl nuw i32 %1718, 16, !spirv.Decorations !888		; visa id: 2167
  %1720 = bitcast i32 %1719 to float
  %1721 = fmul reassoc nsz arcp contract float %1717, %1720, !spirv.Decorations !881
  %1722 = fadd reassoc nsz arcp contract float %1721, %.sroa.46.1, !spirv.Decorations !881		; visa id: 2168
  br label %._crit_edge.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2169

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %1697
; BB141 :
  %.sroa.46.2 = phi float [ %1722, %1697 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  br i1 %266, label %1723, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2170

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
; BB:
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1723:                                             ; preds = %._crit_edge.11
; BB143 :
  %.sroa.256.0.insert.ext808 = zext i32 %552 to i64		; visa id: 2172
  %1724 = shl nuw nsw i64 %.sroa.256.0.insert.ext808, 1		; visa id: 2173
  %1725 = add i64 %549, %1724		; visa id: 2174
  %1726 = inttoptr i64 %1725 to i16 addrspace(4)*		; visa id: 2175
  %1727 = addrspacecast i16 addrspace(4)* %1726 to i16 addrspace(1)*		; visa id: 2175
  %1728 = load i16, i16 addrspace(1)* %1727, align 2		; visa id: 2176
  %1729 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1730 = extractvalue { i32, i32 } %1729, 0		; visa id: 2178
  %1731 = extractvalue { i32, i32 } %1729, 1		; visa id: 2178
  %1732 = insertelement <2 x i32> undef, i32 %1730, i32 0		; visa id: 2185
  %1733 = insertelement <2 x i32> %1732, i32 %1731, i32 1		; visa id: 2186
  %1734 = bitcast <2 x i32> %1733 to i64		; visa id: 2187
  %1735 = shl i64 %1734, 1		; visa id: 2189
  %1736 = add i64 %.in3822, %1735		; visa id: 2190
  %1737 = add i64 %1736, %sink_3836		; visa id: 2191
  %1738 = inttoptr i64 %1737 to i16 addrspace(4)*		; visa id: 2192
  %1739 = addrspacecast i16 addrspace(4)* %1738 to i16 addrspace(1)*		; visa id: 2192
  %1740 = load i16, i16 addrspace(1)* %1739, align 2		; visa id: 2193
  %1741 = zext i16 %1728 to i32		; visa id: 2195
  %1742 = shl nuw i32 %1741, 16, !spirv.Decorations !888		; visa id: 2196
  %1743 = bitcast i32 %1742 to float
  %1744 = zext i16 %1740 to i32		; visa id: 2197
  %1745 = shl nuw i32 %1744, 16, !spirv.Decorations !888		; visa id: 2198
  %1746 = bitcast i32 %1745 to float
  %1747 = fmul reassoc nsz arcp contract float %1743, %1746, !spirv.Decorations !881
  %1748 = fadd reassoc nsz arcp contract float %1747, %.sroa.110.1, !spirv.Decorations !881		; visa id: 2199
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2200

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %1723
; BB144 :
  %.sroa.110.2 = phi float [ %1748, %1723 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  br i1 %269, label %1749, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2201

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
; BB:
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1749:                                             ; preds = %._crit_edge.1.11
; BB146 :
  %.sroa.256.0.insert.ext813 = zext i32 %552 to i64		; visa id: 2203
  %1750 = shl nuw nsw i64 %.sroa.256.0.insert.ext813, 1		; visa id: 2204
  %1751 = add i64 %550, %1750		; visa id: 2205
  %1752 = inttoptr i64 %1751 to i16 addrspace(4)*		; visa id: 2206
  %1753 = addrspacecast i16 addrspace(4)* %1752 to i16 addrspace(1)*		; visa id: 2206
  %1754 = load i16, i16 addrspace(1)* %1753, align 2		; visa id: 2207
  %1755 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1756 = extractvalue { i32, i32 } %1755, 0		; visa id: 2209
  %1757 = extractvalue { i32, i32 } %1755, 1		; visa id: 2209
  %1758 = insertelement <2 x i32> undef, i32 %1756, i32 0		; visa id: 2216
  %1759 = insertelement <2 x i32> %1758, i32 %1757, i32 1		; visa id: 2217
  %1760 = bitcast <2 x i32> %1759 to i64		; visa id: 2218
  %1761 = shl i64 %1760, 1		; visa id: 2220
  %1762 = add i64 %.in3822, %1761		; visa id: 2221
  %1763 = add i64 %1762, %sink_3836		; visa id: 2222
  %1764 = inttoptr i64 %1763 to i16 addrspace(4)*		; visa id: 2223
  %1765 = addrspacecast i16 addrspace(4)* %1764 to i16 addrspace(1)*		; visa id: 2223
  %1766 = load i16, i16 addrspace(1)* %1765, align 2		; visa id: 2224
  %1767 = zext i16 %1754 to i32		; visa id: 2226
  %1768 = shl nuw i32 %1767, 16, !spirv.Decorations !888		; visa id: 2227
  %1769 = bitcast i32 %1768 to float
  %1770 = zext i16 %1766 to i32		; visa id: 2228
  %1771 = shl nuw i32 %1770, 16, !spirv.Decorations !888		; visa id: 2229
  %1772 = bitcast i32 %1771 to float
  %1773 = fmul reassoc nsz arcp contract float %1769, %1772, !spirv.Decorations !881
  %1774 = fadd reassoc nsz arcp contract float %1773, %.sroa.174.1, !spirv.Decorations !881		; visa id: 2230
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2231

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %1749
; BB147 :
  %.sroa.174.2 = phi float [ %1774, %1749 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  br i1 %272, label %1775, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2232

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
; BB:
  br label %.preheader.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1775:                                             ; preds = %._crit_edge.2.11
; BB149 :
  %.sroa.256.0.insert.ext818 = zext i32 %552 to i64		; visa id: 2234
  %1776 = shl nuw nsw i64 %.sroa.256.0.insert.ext818, 1		; visa id: 2235
  %1777 = add i64 %551, %1776		; visa id: 2236
  %1778 = inttoptr i64 %1777 to i16 addrspace(4)*		; visa id: 2237
  %1779 = addrspacecast i16 addrspace(4)* %1778 to i16 addrspace(1)*		; visa id: 2237
  %1780 = load i16, i16 addrspace(1)* %1779, align 2		; visa id: 2238
  %1781 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1782 = extractvalue { i32, i32 } %1781, 0		; visa id: 2240
  %1783 = extractvalue { i32, i32 } %1781, 1		; visa id: 2240
  %1784 = insertelement <2 x i32> undef, i32 %1782, i32 0		; visa id: 2247
  %1785 = insertelement <2 x i32> %1784, i32 %1783, i32 1		; visa id: 2248
  %1786 = bitcast <2 x i32> %1785 to i64		; visa id: 2249
  %1787 = shl i64 %1786, 1		; visa id: 2251
  %1788 = add i64 %.in3822, %1787		; visa id: 2252
  %1789 = add i64 %1788, %sink_3836		; visa id: 2253
  %1790 = inttoptr i64 %1789 to i16 addrspace(4)*		; visa id: 2254
  %1791 = addrspacecast i16 addrspace(4)* %1790 to i16 addrspace(1)*		; visa id: 2254
  %1792 = load i16, i16 addrspace(1)* %1791, align 2		; visa id: 2255
  %1793 = zext i16 %1780 to i32		; visa id: 2257
  %1794 = shl nuw i32 %1793, 16, !spirv.Decorations !888		; visa id: 2258
  %1795 = bitcast i32 %1794 to float
  %1796 = zext i16 %1792 to i32		; visa id: 2259
  %1797 = shl nuw i32 %1796, 16, !spirv.Decorations !888		; visa id: 2260
  %1798 = bitcast i32 %1797 to float
  %1799 = fmul reassoc nsz arcp contract float %1795, %1798, !spirv.Decorations !881
  %1800 = fadd reassoc nsz arcp contract float %1799, %.sroa.238.1, !spirv.Decorations !881		; visa id: 2261
  br label %.preheader.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2262

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %1775
; BB150 :
  %.sroa.238.2 = phi float [ %1800, %1775 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %sink_3834 = shl nsw i64 %361, 1		; visa id: 2263
  br i1 %276, label %1801, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2264

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
; BB:
  br label %._crit_edge.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1801:                                             ; preds = %.preheader.11
; BB152 :
  %.sroa.256.0.insert.ext823 = zext i32 %552 to i64		; visa id: 2266
  %1802 = shl nuw nsw i64 %.sroa.256.0.insert.ext823, 1		; visa id: 2267
  %1803 = add i64 %548, %1802		; visa id: 2268
  %1804 = inttoptr i64 %1803 to i16 addrspace(4)*		; visa id: 2269
  %1805 = addrspacecast i16 addrspace(4)* %1804 to i16 addrspace(1)*		; visa id: 2269
  %1806 = load i16, i16 addrspace(1)* %1805, align 2		; visa id: 2270
  %1807 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1808 = extractvalue { i32, i32 } %1807, 0		; visa id: 2272
  %1809 = extractvalue { i32, i32 } %1807, 1		; visa id: 2272
  %1810 = insertelement <2 x i32> undef, i32 %1808, i32 0		; visa id: 2279
  %1811 = insertelement <2 x i32> %1810, i32 %1809, i32 1		; visa id: 2280
  %1812 = bitcast <2 x i32> %1811 to i64		; visa id: 2281
  %1813 = shl i64 %1812, 1		; visa id: 2283
  %1814 = add i64 %.in3822, %1813		; visa id: 2284
  %1815 = add i64 %1814, %sink_3834		; visa id: 2285
  %1816 = inttoptr i64 %1815 to i16 addrspace(4)*		; visa id: 2286
  %1817 = addrspacecast i16 addrspace(4)* %1816 to i16 addrspace(1)*		; visa id: 2286
  %1818 = load i16, i16 addrspace(1)* %1817, align 2		; visa id: 2287
  %1819 = zext i16 %1806 to i32		; visa id: 2289
  %1820 = shl nuw i32 %1819, 16, !spirv.Decorations !888		; visa id: 2290
  %1821 = bitcast i32 %1820 to float
  %1822 = zext i16 %1818 to i32		; visa id: 2291
  %1823 = shl nuw i32 %1822, 16, !spirv.Decorations !888		; visa id: 2292
  %1824 = bitcast i32 %1823 to float
  %1825 = fmul reassoc nsz arcp contract float %1821, %1824, !spirv.Decorations !881
  %1826 = fadd reassoc nsz arcp contract float %1825, %.sroa.50.1, !spirv.Decorations !881		; visa id: 2293
  br label %._crit_edge.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2294

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %1801
; BB153 :
  %.sroa.50.2 = phi float [ %1826, %1801 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  br i1 %279, label %1827, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2295

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
; BB:
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1827:                                             ; preds = %._crit_edge.12
; BB155 :
  %.sroa.256.0.insert.ext828 = zext i32 %552 to i64		; visa id: 2297
  %1828 = shl nuw nsw i64 %.sroa.256.0.insert.ext828, 1		; visa id: 2298
  %1829 = add i64 %549, %1828		; visa id: 2299
  %1830 = inttoptr i64 %1829 to i16 addrspace(4)*		; visa id: 2300
  %1831 = addrspacecast i16 addrspace(4)* %1830 to i16 addrspace(1)*		; visa id: 2300
  %1832 = load i16, i16 addrspace(1)* %1831, align 2		; visa id: 2301
  %1833 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1834 = extractvalue { i32, i32 } %1833, 0		; visa id: 2303
  %1835 = extractvalue { i32, i32 } %1833, 1		; visa id: 2303
  %1836 = insertelement <2 x i32> undef, i32 %1834, i32 0		; visa id: 2310
  %1837 = insertelement <2 x i32> %1836, i32 %1835, i32 1		; visa id: 2311
  %1838 = bitcast <2 x i32> %1837 to i64		; visa id: 2312
  %1839 = shl i64 %1838, 1		; visa id: 2314
  %1840 = add i64 %.in3822, %1839		; visa id: 2315
  %1841 = add i64 %1840, %sink_3834		; visa id: 2316
  %1842 = inttoptr i64 %1841 to i16 addrspace(4)*		; visa id: 2317
  %1843 = addrspacecast i16 addrspace(4)* %1842 to i16 addrspace(1)*		; visa id: 2317
  %1844 = load i16, i16 addrspace(1)* %1843, align 2		; visa id: 2318
  %1845 = zext i16 %1832 to i32		; visa id: 2320
  %1846 = shl nuw i32 %1845, 16, !spirv.Decorations !888		; visa id: 2321
  %1847 = bitcast i32 %1846 to float
  %1848 = zext i16 %1844 to i32		; visa id: 2322
  %1849 = shl nuw i32 %1848, 16, !spirv.Decorations !888		; visa id: 2323
  %1850 = bitcast i32 %1849 to float
  %1851 = fmul reassoc nsz arcp contract float %1847, %1850, !spirv.Decorations !881
  %1852 = fadd reassoc nsz arcp contract float %1851, %.sroa.114.1, !spirv.Decorations !881		; visa id: 2324
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2325

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %1827
; BB156 :
  %.sroa.114.2 = phi float [ %1852, %1827 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  br i1 %282, label %1853, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2326

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
; BB:
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1853:                                             ; preds = %._crit_edge.1.12
; BB158 :
  %.sroa.256.0.insert.ext833 = zext i32 %552 to i64		; visa id: 2328
  %1854 = shl nuw nsw i64 %.sroa.256.0.insert.ext833, 1		; visa id: 2329
  %1855 = add i64 %550, %1854		; visa id: 2330
  %1856 = inttoptr i64 %1855 to i16 addrspace(4)*		; visa id: 2331
  %1857 = addrspacecast i16 addrspace(4)* %1856 to i16 addrspace(1)*		; visa id: 2331
  %1858 = load i16, i16 addrspace(1)* %1857, align 2		; visa id: 2332
  %1859 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1860 = extractvalue { i32, i32 } %1859, 0		; visa id: 2334
  %1861 = extractvalue { i32, i32 } %1859, 1		; visa id: 2334
  %1862 = insertelement <2 x i32> undef, i32 %1860, i32 0		; visa id: 2341
  %1863 = insertelement <2 x i32> %1862, i32 %1861, i32 1		; visa id: 2342
  %1864 = bitcast <2 x i32> %1863 to i64		; visa id: 2343
  %1865 = shl i64 %1864, 1		; visa id: 2345
  %1866 = add i64 %.in3822, %1865		; visa id: 2346
  %1867 = add i64 %1866, %sink_3834		; visa id: 2347
  %1868 = inttoptr i64 %1867 to i16 addrspace(4)*		; visa id: 2348
  %1869 = addrspacecast i16 addrspace(4)* %1868 to i16 addrspace(1)*		; visa id: 2348
  %1870 = load i16, i16 addrspace(1)* %1869, align 2		; visa id: 2349
  %1871 = zext i16 %1858 to i32		; visa id: 2351
  %1872 = shl nuw i32 %1871, 16, !spirv.Decorations !888		; visa id: 2352
  %1873 = bitcast i32 %1872 to float
  %1874 = zext i16 %1870 to i32		; visa id: 2353
  %1875 = shl nuw i32 %1874, 16, !spirv.Decorations !888		; visa id: 2354
  %1876 = bitcast i32 %1875 to float
  %1877 = fmul reassoc nsz arcp contract float %1873, %1876, !spirv.Decorations !881
  %1878 = fadd reassoc nsz arcp contract float %1877, %.sroa.178.1, !spirv.Decorations !881		; visa id: 2355
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2356

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %1853
; BB159 :
  %.sroa.178.2 = phi float [ %1878, %1853 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  br i1 %285, label %1879, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2357

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
; BB:
  br label %.preheader.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1879:                                             ; preds = %._crit_edge.2.12
; BB161 :
  %.sroa.256.0.insert.ext838 = zext i32 %552 to i64		; visa id: 2359
  %1880 = shl nuw nsw i64 %.sroa.256.0.insert.ext838, 1		; visa id: 2360
  %1881 = add i64 %551, %1880		; visa id: 2361
  %1882 = inttoptr i64 %1881 to i16 addrspace(4)*		; visa id: 2362
  %1883 = addrspacecast i16 addrspace(4)* %1882 to i16 addrspace(1)*		; visa id: 2362
  %1884 = load i16, i16 addrspace(1)* %1883, align 2		; visa id: 2363
  %1885 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1886 = extractvalue { i32, i32 } %1885, 0		; visa id: 2365
  %1887 = extractvalue { i32, i32 } %1885, 1		; visa id: 2365
  %1888 = insertelement <2 x i32> undef, i32 %1886, i32 0		; visa id: 2372
  %1889 = insertelement <2 x i32> %1888, i32 %1887, i32 1		; visa id: 2373
  %1890 = bitcast <2 x i32> %1889 to i64		; visa id: 2374
  %1891 = shl i64 %1890, 1		; visa id: 2376
  %1892 = add i64 %.in3822, %1891		; visa id: 2377
  %1893 = add i64 %1892, %sink_3834		; visa id: 2378
  %1894 = inttoptr i64 %1893 to i16 addrspace(4)*		; visa id: 2379
  %1895 = addrspacecast i16 addrspace(4)* %1894 to i16 addrspace(1)*		; visa id: 2379
  %1896 = load i16, i16 addrspace(1)* %1895, align 2		; visa id: 2380
  %1897 = zext i16 %1884 to i32		; visa id: 2382
  %1898 = shl nuw i32 %1897, 16, !spirv.Decorations !888		; visa id: 2383
  %1899 = bitcast i32 %1898 to float
  %1900 = zext i16 %1896 to i32		; visa id: 2384
  %1901 = shl nuw i32 %1900, 16, !spirv.Decorations !888		; visa id: 2385
  %1902 = bitcast i32 %1901 to float
  %1903 = fmul reassoc nsz arcp contract float %1899, %1902, !spirv.Decorations !881
  %1904 = fadd reassoc nsz arcp contract float %1903, %.sroa.242.1, !spirv.Decorations !881		; visa id: 2386
  br label %.preheader.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2387

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %1879
; BB162 :
  %.sroa.242.2 = phi float [ %1904, %1879 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %sink_3832 = shl nsw i64 %362, 1		; visa id: 2388
  br i1 %289, label %1905, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2389

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
; BB:
  br label %._crit_edge.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1905:                                             ; preds = %.preheader.12
; BB164 :
  %.sroa.256.0.insert.ext843 = zext i32 %552 to i64		; visa id: 2391
  %1906 = shl nuw nsw i64 %.sroa.256.0.insert.ext843, 1		; visa id: 2392
  %1907 = add i64 %548, %1906		; visa id: 2393
  %1908 = inttoptr i64 %1907 to i16 addrspace(4)*		; visa id: 2394
  %1909 = addrspacecast i16 addrspace(4)* %1908 to i16 addrspace(1)*		; visa id: 2394
  %1910 = load i16, i16 addrspace(1)* %1909, align 2		; visa id: 2395
  %1911 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1912 = extractvalue { i32, i32 } %1911, 0		; visa id: 2397
  %1913 = extractvalue { i32, i32 } %1911, 1		; visa id: 2397
  %1914 = insertelement <2 x i32> undef, i32 %1912, i32 0		; visa id: 2404
  %1915 = insertelement <2 x i32> %1914, i32 %1913, i32 1		; visa id: 2405
  %1916 = bitcast <2 x i32> %1915 to i64		; visa id: 2406
  %1917 = shl i64 %1916, 1		; visa id: 2408
  %1918 = add i64 %.in3822, %1917		; visa id: 2409
  %1919 = add i64 %1918, %sink_3832		; visa id: 2410
  %1920 = inttoptr i64 %1919 to i16 addrspace(4)*		; visa id: 2411
  %1921 = addrspacecast i16 addrspace(4)* %1920 to i16 addrspace(1)*		; visa id: 2411
  %1922 = load i16, i16 addrspace(1)* %1921, align 2		; visa id: 2412
  %1923 = zext i16 %1910 to i32		; visa id: 2414
  %1924 = shl nuw i32 %1923, 16, !spirv.Decorations !888		; visa id: 2415
  %1925 = bitcast i32 %1924 to float
  %1926 = zext i16 %1922 to i32		; visa id: 2416
  %1927 = shl nuw i32 %1926, 16, !spirv.Decorations !888		; visa id: 2417
  %1928 = bitcast i32 %1927 to float
  %1929 = fmul reassoc nsz arcp contract float %1925, %1928, !spirv.Decorations !881
  %1930 = fadd reassoc nsz arcp contract float %1929, %.sroa.54.1, !spirv.Decorations !881		; visa id: 2418
  br label %._crit_edge.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2419

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %1905
; BB165 :
  %.sroa.54.2 = phi float [ %1930, %1905 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  br i1 %292, label %1931, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2420

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
; BB:
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1931:                                             ; preds = %._crit_edge.13
; BB167 :
  %.sroa.256.0.insert.ext848 = zext i32 %552 to i64		; visa id: 2422
  %1932 = shl nuw nsw i64 %.sroa.256.0.insert.ext848, 1		; visa id: 2423
  %1933 = add i64 %549, %1932		; visa id: 2424
  %1934 = inttoptr i64 %1933 to i16 addrspace(4)*		; visa id: 2425
  %1935 = addrspacecast i16 addrspace(4)* %1934 to i16 addrspace(1)*		; visa id: 2425
  %1936 = load i16, i16 addrspace(1)* %1935, align 2		; visa id: 2426
  %1937 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1938 = extractvalue { i32, i32 } %1937, 0		; visa id: 2428
  %1939 = extractvalue { i32, i32 } %1937, 1		; visa id: 2428
  %1940 = insertelement <2 x i32> undef, i32 %1938, i32 0		; visa id: 2435
  %1941 = insertelement <2 x i32> %1940, i32 %1939, i32 1		; visa id: 2436
  %1942 = bitcast <2 x i32> %1941 to i64		; visa id: 2437
  %1943 = shl i64 %1942, 1		; visa id: 2439
  %1944 = add i64 %.in3822, %1943		; visa id: 2440
  %1945 = add i64 %1944, %sink_3832		; visa id: 2441
  %1946 = inttoptr i64 %1945 to i16 addrspace(4)*		; visa id: 2442
  %1947 = addrspacecast i16 addrspace(4)* %1946 to i16 addrspace(1)*		; visa id: 2442
  %1948 = load i16, i16 addrspace(1)* %1947, align 2		; visa id: 2443
  %1949 = zext i16 %1936 to i32		; visa id: 2445
  %1950 = shl nuw i32 %1949, 16, !spirv.Decorations !888		; visa id: 2446
  %1951 = bitcast i32 %1950 to float
  %1952 = zext i16 %1948 to i32		; visa id: 2447
  %1953 = shl nuw i32 %1952, 16, !spirv.Decorations !888		; visa id: 2448
  %1954 = bitcast i32 %1953 to float
  %1955 = fmul reassoc nsz arcp contract float %1951, %1954, !spirv.Decorations !881
  %1956 = fadd reassoc nsz arcp contract float %1955, %.sroa.118.1, !spirv.Decorations !881		; visa id: 2449
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2450

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %1931
; BB168 :
  %.sroa.118.2 = phi float [ %1956, %1931 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  br i1 %295, label %1957, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2451

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
; BB:
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1957:                                             ; preds = %._crit_edge.1.13
; BB170 :
  %.sroa.256.0.insert.ext853 = zext i32 %552 to i64		; visa id: 2453
  %1958 = shl nuw nsw i64 %.sroa.256.0.insert.ext853, 1		; visa id: 2454
  %1959 = add i64 %550, %1958		; visa id: 2455
  %1960 = inttoptr i64 %1959 to i16 addrspace(4)*		; visa id: 2456
  %1961 = addrspacecast i16 addrspace(4)* %1960 to i16 addrspace(1)*		; visa id: 2456
  %1962 = load i16, i16 addrspace(1)* %1961, align 2		; visa id: 2457
  %1963 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1964 = extractvalue { i32, i32 } %1963, 0		; visa id: 2459
  %1965 = extractvalue { i32, i32 } %1963, 1		; visa id: 2459
  %1966 = insertelement <2 x i32> undef, i32 %1964, i32 0		; visa id: 2466
  %1967 = insertelement <2 x i32> %1966, i32 %1965, i32 1		; visa id: 2467
  %1968 = bitcast <2 x i32> %1967 to i64		; visa id: 2468
  %1969 = shl i64 %1968, 1		; visa id: 2470
  %1970 = add i64 %.in3822, %1969		; visa id: 2471
  %1971 = add i64 %1970, %sink_3832		; visa id: 2472
  %1972 = inttoptr i64 %1971 to i16 addrspace(4)*		; visa id: 2473
  %1973 = addrspacecast i16 addrspace(4)* %1972 to i16 addrspace(1)*		; visa id: 2473
  %1974 = load i16, i16 addrspace(1)* %1973, align 2		; visa id: 2474
  %1975 = zext i16 %1962 to i32		; visa id: 2476
  %1976 = shl nuw i32 %1975, 16, !spirv.Decorations !888		; visa id: 2477
  %1977 = bitcast i32 %1976 to float
  %1978 = zext i16 %1974 to i32		; visa id: 2478
  %1979 = shl nuw i32 %1978, 16, !spirv.Decorations !888		; visa id: 2479
  %1980 = bitcast i32 %1979 to float
  %1981 = fmul reassoc nsz arcp contract float %1977, %1980, !spirv.Decorations !881
  %1982 = fadd reassoc nsz arcp contract float %1981, %.sroa.182.1, !spirv.Decorations !881		; visa id: 2480
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2481

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %1957
; BB171 :
  %.sroa.182.2 = phi float [ %1982, %1957 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  br i1 %298, label %1983, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2482

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
; BB:
  br label %.preheader.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1983:                                             ; preds = %._crit_edge.2.13
; BB173 :
  %.sroa.256.0.insert.ext858 = zext i32 %552 to i64		; visa id: 2484
  %1984 = shl nuw nsw i64 %.sroa.256.0.insert.ext858, 1		; visa id: 2485
  %1985 = add i64 %551, %1984		; visa id: 2486
  %1986 = inttoptr i64 %1985 to i16 addrspace(4)*		; visa id: 2487
  %1987 = addrspacecast i16 addrspace(4)* %1986 to i16 addrspace(1)*		; visa id: 2487
  %1988 = load i16, i16 addrspace(1)* %1987, align 2		; visa id: 2488
  %1989 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1990 = extractvalue { i32, i32 } %1989, 0		; visa id: 2490
  %1991 = extractvalue { i32, i32 } %1989, 1		; visa id: 2490
  %1992 = insertelement <2 x i32> undef, i32 %1990, i32 0		; visa id: 2497
  %1993 = insertelement <2 x i32> %1992, i32 %1991, i32 1		; visa id: 2498
  %1994 = bitcast <2 x i32> %1993 to i64		; visa id: 2499
  %1995 = shl i64 %1994, 1		; visa id: 2501
  %1996 = add i64 %.in3822, %1995		; visa id: 2502
  %1997 = add i64 %1996, %sink_3832		; visa id: 2503
  %1998 = inttoptr i64 %1997 to i16 addrspace(4)*		; visa id: 2504
  %1999 = addrspacecast i16 addrspace(4)* %1998 to i16 addrspace(1)*		; visa id: 2504
  %2000 = load i16, i16 addrspace(1)* %1999, align 2		; visa id: 2505
  %2001 = zext i16 %1988 to i32		; visa id: 2507
  %2002 = shl nuw i32 %2001, 16, !spirv.Decorations !888		; visa id: 2508
  %2003 = bitcast i32 %2002 to float
  %2004 = zext i16 %2000 to i32		; visa id: 2509
  %2005 = shl nuw i32 %2004, 16, !spirv.Decorations !888		; visa id: 2510
  %2006 = bitcast i32 %2005 to float
  %2007 = fmul reassoc nsz arcp contract float %2003, %2006, !spirv.Decorations !881
  %2008 = fadd reassoc nsz arcp contract float %2007, %.sroa.246.1, !spirv.Decorations !881		; visa id: 2511
  br label %.preheader.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2512

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %1983
; BB174 :
  %.sroa.246.2 = phi float [ %2008, %1983 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %sink_3830 = shl nsw i64 %363, 1		; visa id: 2513
  br i1 %302, label %2009, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2514

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
; BB:
  br label %._crit_edge.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2009:                                             ; preds = %.preheader.13
; BB176 :
  %.sroa.256.0.insert.ext863 = zext i32 %552 to i64		; visa id: 2516
  %2010 = shl nuw nsw i64 %.sroa.256.0.insert.ext863, 1		; visa id: 2517
  %2011 = add i64 %548, %2010		; visa id: 2518
  %2012 = inttoptr i64 %2011 to i16 addrspace(4)*		; visa id: 2519
  %2013 = addrspacecast i16 addrspace(4)* %2012 to i16 addrspace(1)*		; visa id: 2519
  %2014 = load i16, i16 addrspace(1)* %2013, align 2		; visa id: 2520
  %2015 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2016 = extractvalue { i32, i32 } %2015, 0		; visa id: 2522
  %2017 = extractvalue { i32, i32 } %2015, 1		; visa id: 2522
  %2018 = insertelement <2 x i32> undef, i32 %2016, i32 0		; visa id: 2529
  %2019 = insertelement <2 x i32> %2018, i32 %2017, i32 1		; visa id: 2530
  %2020 = bitcast <2 x i32> %2019 to i64		; visa id: 2531
  %2021 = shl i64 %2020, 1		; visa id: 2533
  %2022 = add i64 %.in3822, %2021		; visa id: 2534
  %2023 = add i64 %2022, %sink_3830		; visa id: 2535
  %2024 = inttoptr i64 %2023 to i16 addrspace(4)*		; visa id: 2536
  %2025 = addrspacecast i16 addrspace(4)* %2024 to i16 addrspace(1)*		; visa id: 2536
  %2026 = load i16, i16 addrspace(1)* %2025, align 2		; visa id: 2537
  %2027 = zext i16 %2014 to i32		; visa id: 2539
  %2028 = shl nuw i32 %2027, 16, !spirv.Decorations !888		; visa id: 2540
  %2029 = bitcast i32 %2028 to float
  %2030 = zext i16 %2026 to i32		; visa id: 2541
  %2031 = shl nuw i32 %2030, 16, !spirv.Decorations !888		; visa id: 2542
  %2032 = bitcast i32 %2031 to float
  %2033 = fmul reassoc nsz arcp contract float %2029, %2032, !spirv.Decorations !881
  %2034 = fadd reassoc nsz arcp contract float %2033, %.sroa.58.1, !spirv.Decorations !881		; visa id: 2543
  br label %._crit_edge.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2544

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %2009
; BB177 :
  %.sroa.58.2 = phi float [ %2034, %2009 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  br i1 %305, label %2035, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2545

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
; BB:
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2035:                                             ; preds = %._crit_edge.14
; BB179 :
  %.sroa.256.0.insert.ext868 = zext i32 %552 to i64		; visa id: 2547
  %2036 = shl nuw nsw i64 %.sroa.256.0.insert.ext868, 1		; visa id: 2548
  %2037 = add i64 %549, %2036		; visa id: 2549
  %2038 = inttoptr i64 %2037 to i16 addrspace(4)*		; visa id: 2550
  %2039 = addrspacecast i16 addrspace(4)* %2038 to i16 addrspace(1)*		; visa id: 2550
  %2040 = load i16, i16 addrspace(1)* %2039, align 2		; visa id: 2551
  %2041 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2042 = extractvalue { i32, i32 } %2041, 0		; visa id: 2553
  %2043 = extractvalue { i32, i32 } %2041, 1		; visa id: 2553
  %2044 = insertelement <2 x i32> undef, i32 %2042, i32 0		; visa id: 2560
  %2045 = insertelement <2 x i32> %2044, i32 %2043, i32 1		; visa id: 2561
  %2046 = bitcast <2 x i32> %2045 to i64		; visa id: 2562
  %2047 = shl i64 %2046, 1		; visa id: 2564
  %2048 = add i64 %.in3822, %2047		; visa id: 2565
  %2049 = add i64 %2048, %sink_3830		; visa id: 2566
  %2050 = inttoptr i64 %2049 to i16 addrspace(4)*		; visa id: 2567
  %2051 = addrspacecast i16 addrspace(4)* %2050 to i16 addrspace(1)*		; visa id: 2567
  %2052 = load i16, i16 addrspace(1)* %2051, align 2		; visa id: 2568
  %2053 = zext i16 %2040 to i32		; visa id: 2570
  %2054 = shl nuw i32 %2053, 16, !spirv.Decorations !888		; visa id: 2571
  %2055 = bitcast i32 %2054 to float
  %2056 = zext i16 %2052 to i32		; visa id: 2572
  %2057 = shl nuw i32 %2056, 16, !spirv.Decorations !888		; visa id: 2573
  %2058 = bitcast i32 %2057 to float
  %2059 = fmul reassoc nsz arcp contract float %2055, %2058, !spirv.Decorations !881
  %2060 = fadd reassoc nsz arcp contract float %2059, %.sroa.122.1, !spirv.Decorations !881		; visa id: 2574
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2575

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %2035
; BB180 :
  %.sroa.122.2 = phi float [ %2060, %2035 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  br i1 %308, label %2061, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2576

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
; BB:
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2061:                                             ; preds = %._crit_edge.1.14
; BB182 :
  %.sroa.256.0.insert.ext873 = zext i32 %552 to i64		; visa id: 2578
  %2062 = shl nuw nsw i64 %.sroa.256.0.insert.ext873, 1		; visa id: 2579
  %2063 = add i64 %550, %2062		; visa id: 2580
  %2064 = inttoptr i64 %2063 to i16 addrspace(4)*		; visa id: 2581
  %2065 = addrspacecast i16 addrspace(4)* %2064 to i16 addrspace(1)*		; visa id: 2581
  %2066 = load i16, i16 addrspace(1)* %2065, align 2		; visa id: 2582
  %2067 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2068 = extractvalue { i32, i32 } %2067, 0		; visa id: 2584
  %2069 = extractvalue { i32, i32 } %2067, 1		; visa id: 2584
  %2070 = insertelement <2 x i32> undef, i32 %2068, i32 0		; visa id: 2591
  %2071 = insertelement <2 x i32> %2070, i32 %2069, i32 1		; visa id: 2592
  %2072 = bitcast <2 x i32> %2071 to i64		; visa id: 2593
  %2073 = shl i64 %2072, 1		; visa id: 2595
  %2074 = add i64 %.in3822, %2073		; visa id: 2596
  %2075 = add i64 %2074, %sink_3830		; visa id: 2597
  %2076 = inttoptr i64 %2075 to i16 addrspace(4)*		; visa id: 2598
  %2077 = addrspacecast i16 addrspace(4)* %2076 to i16 addrspace(1)*		; visa id: 2598
  %2078 = load i16, i16 addrspace(1)* %2077, align 2		; visa id: 2599
  %2079 = zext i16 %2066 to i32		; visa id: 2601
  %2080 = shl nuw i32 %2079, 16, !spirv.Decorations !888		; visa id: 2602
  %2081 = bitcast i32 %2080 to float
  %2082 = zext i16 %2078 to i32		; visa id: 2603
  %2083 = shl nuw i32 %2082, 16, !spirv.Decorations !888		; visa id: 2604
  %2084 = bitcast i32 %2083 to float
  %2085 = fmul reassoc nsz arcp contract float %2081, %2084, !spirv.Decorations !881
  %2086 = fadd reassoc nsz arcp contract float %2085, %.sroa.186.1, !spirv.Decorations !881		; visa id: 2605
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2606

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %2061
; BB183 :
  %.sroa.186.2 = phi float [ %2086, %2061 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  br i1 %311, label %2087, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2607

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
; BB:
  br label %.preheader.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2087:                                             ; preds = %._crit_edge.2.14
; BB185 :
  %.sroa.256.0.insert.ext878 = zext i32 %552 to i64		; visa id: 2609
  %2088 = shl nuw nsw i64 %.sroa.256.0.insert.ext878, 1		; visa id: 2610
  %2089 = add i64 %551, %2088		; visa id: 2611
  %2090 = inttoptr i64 %2089 to i16 addrspace(4)*		; visa id: 2612
  %2091 = addrspacecast i16 addrspace(4)* %2090 to i16 addrspace(1)*		; visa id: 2612
  %2092 = load i16, i16 addrspace(1)* %2091, align 2		; visa id: 2613
  %2093 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2094 = extractvalue { i32, i32 } %2093, 0		; visa id: 2615
  %2095 = extractvalue { i32, i32 } %2093, 1		; visa id: 2615
  %2096 = insertelement <2 x i32> undef, i32 %2094, i32 0		; visa id: 2622
  %2097 = insertelement <2 x i32> %2096, i32 %2095, i32 1		; visa id: 2623
  %2098 = bitcast <2 x i32> %2097 to i64		; visa id: 2624
  %2099 = shl i64 %2098, 1		; visa id: 2626
  %2100 = add i64 %.in3822, %2099		; visa id: 2627
  %2101 = add i64 %2100, %sink_3830		; visa id: 2628
  %2102 = inttoptr i64 %2101 to i16 addrspace(4)*		; visa id: 2629
  %2103 = addrspacecast i16 addrspace(4)* %2102 to i16 addrspace(1)*		; visa id: 2629
  %2104 = load i16, i16 addrspace(1)* %2103, align 2		; visa id: 2630
  %2105 = zext i16 %2092 to i32		; visa id: 2632
  %2106 = shl nuw i32 %2105, 16, !spirv.Decorations !888		; visa id: 2633
  %2107 = bitcast i32 %2106 to float
  %2108 = zext i16 %2104 to i32		; visa id: 2634
  %2109 = shl nuw i32 %2108, 16, !spirv.Decorations !888		; visa id: 2635
  %2110 = bitcast i32 %2109 to float
  %2111 = fmul reassoc nsz arcp contract float %2107, %2110, !spirv.Decorations !881
  %2112 = fadd reassoc nsz arcp contract float %2111, %.sroa.250.1, !spirv.Decorations !881		; visa id: 2636
  br label %.preheader.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2637

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %2087
; BB186 :
  %.sroa.250.2 = phi float [ %2112, %2087 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %sink_3828 = shl nsw i64 %364, 1		; visa id: 2638
  br i1 %315, label %2113, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2639

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
; BB:
  br label %._crit_edge.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2113:                                             ; preds = %.preheader.14
; BB188 :
  %.sroa.256.0.insert.ext883 = zext i32 %552 to i64		; visa id: 2641
  %2114 = shl nuw nsw i64 %.sroa.256.0.insert.ext883, 1		; visa id: 2642
  %2115 = add i64 %548, %2114		; visa id: 2643
  %2116 = inttoptr i64 %2115 to i16 addrspace(4)*		; visa id: 2644
  %2117 = addrspacecast i16 addrspace(4)* %2116 to i16 addrspace(1)*		; visa id: 2644
  %2118 = load i16, i16 addrspace(1)* %2117, align 2		; visa id: 2645
  %2119 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2120 = extractvalue { i32, i32 } %2119, 0		; visa id: 2647
  %2121 = extractvalue { i32, i32 } %2119, 1		; visa id: 2647
  %2122 = insertelement <2 x i32> undef, i32 %2120, i32 0		; visa id: 2654
  %2123 = insertelement <2 x i32> %2122, i32 %2121, i32 1		; visa id: 2655
  %2124 = bitcast <2 x i32> %2123 to i64		; visa id: 2656
  %2125 = shl i64 %2124, 1		; visa id: 2658
  %2126 = add i64 %.in3822, %2125		; visa id: 2659
  %2127 = add i64 %2126, %sink_3828		; visa id: 2660
  %2128 = inttoptr i64 %2127 to i16 addrspace(4)*		; visa id: 2661
  %2129 = addrspacecast i16 addrspace(4)* %2128 to i16 addrspace(1)*		; visa id: 2661
  %2130 = load i16, i16 addrspace(1)* %2129, align 2		; visa id: 2662
  %2131 = zext i16 %2118 to i32		; visa id: 2664
  %2132 = shl nuw i32 %2131, 16, !spirv.Decorations !888		; visa id: 2665
  %2133 = bitcast i32 %2132 to float
  %2134 = zext i16 %2130 to i32		; visa id: 2666
  %2135 = shl nuw i32 %2134, 16, !spirv.Decorations !888		; visa id: 2667
  %2136 = bitcast i32 %2135 to float
  %2137 = fmul reassoc nsz arcp contract float %2133, %2136, !spirv.Decorations !881
  %2138 = fadd reassoc nsz arcp contract float %2137, %.sroa.62.1, !spirv.Decorations !881		; visa id: 2668
  br label %._crit_edge.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2669

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %2113
; BB189 :
  %.sroa.62.2 = phi float [ %2138, %2113 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  br i1 %318, label %2139, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2670

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
; BB:
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2139:                                             ; preds = %._crit_edge.15
; BB191 :
  %.sroa.256.0.insert.ext888 = zext i32 %552 to i64		; visa id: 2672
  %2140 = shl nuw nsw i64 %.sroa.256.0.insert.ext888, 1		; visa id: 2673
  %2141 = add i64 %549, %2140		; visa id: 2674
  %2142 = inttoptr i64 %2141 to i16 addrspace(4)*		; visa id: 2675
  %2143 = addrspacecast i16 addrspace(4)* %2142 to i16 addrspace(1)*		; visa id: 2675
  %2144 = load i16, i16 addrspace(1)* %2143, align 2		; visa id: 2676
  %2145 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2146 = extractvalue { i32, i32 } %2145, 0		; visa id: 2678
  %2147 = extractvalue { i32, i32 } %2145, 1		; visa id: 2678
  %2148 = insertelement <2 x i32> undef, i32 %2146, i32 0		; visa id: 2685
  %2149 = insertelement <2 x i32> %2148, i32 %2147, i32 1		; visa id: 2686
  %2150 = bitcast <2 x i32> %2149 to i64		; visa id: 2687
  %2151 = shl i64 %2150, 1		; visa id: 2689
  %2152 = add i64 %.in3822, %2151		; visa id: 2690
  %2153 = add i64 %2152, %sink_3828		; visa id: 2691
  %2154 = inttoptr i64 %2153 to i16 addrspace(4)*		; visa id: 2692
  %2155 = addrspacecast i16 addrspace(4)* %2154 to i16 addrspace(1)*		; visa id: 2692
  %2156 = load i16, i16 addrspace(1)* %2155, align 2		; visa id: 2693
  %2157 = zext i16 %2144 to i32		; visa id: 2695
  %2158 = shl nuw i32 %2157, 16, !spirv.Decorations !888		; visa id: 2696
  %2159 = bitcast i32 %2158 to float
  %2160 = zext i16 %2156 to i32		; visa id: 2697
  %2161 = shl nuw i32 %2160, 16, !spirv.Decorations !888		; visa id: 2698
  %2162 = bitcast i32 %2161 to float
  %2163 = fmul reassoc nsz arcp contract float %2159, %2162, !spirv.Decorations !881
  %2164 = fadd reassoc nsz arcp contract float %2163, %.sroa.126.1, !spirv.Decorations !881		; visa id: 2699
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2700

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %2139
; BB192 :
  %.sroa.126.2 = phi float [ %2164, %2139 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  br i1 %321, label %2165, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2701

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
; BB:
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2165:                                             ; preds = %._crit_edge.1.15
; BB194 :
  %.sroa.256.0.insert.ext893 = zext i32 %552 to i64		; visa id: 2703
  %2166 = shl nuw nsw i64 %.sroa.256.0.insert.ext893, 1		; visa id: 2704
  %2167 = add i64 %550, %2166		; visa id: 2705
  %2168 = inttoptr i64 %2167 to i16 addrspace(4)*		; visa id: 2706
  %2169 = addrspacecast i16 addrspace(4)* %2168 to i16 addrspace(1)*		; visa id: 2706
  %2170 = load i16, i16 addrspace(1)* %2169, align 2		; visa id: 2707
  %2171 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2172 = extractvalue { i32, i32 } %2171, 0		; visa id: 2709
  %2173 = extractvalue { i32, i32 } %2171, 1		; visa id: 2709
  %2174 = insertelement <2 x i32> undef, i32 %2172, i32 0		; visa id: 2716
  %2175 = insertelement <2 x i32> %2174, i32 %2173, i32 1		; visa id: 2717
  %2176 = bitcast <2 x i32> %2175 to i64		; visa id: 2718
  %2177 = shl i64 %2176, 1		; visa id: 2720
  %2178 = add i64 %.in3822, %2177		; visa id: 2721
  %2179 = add i64 %2178, %sink_3828		; visa id: 2722
  %2180 = inttoptr i64 %2179 to i16 addrspace(4)*		; visa id: 2723
  %2181 = addrspacecast i16 addrspace(4)* %2180 to i16 addrspace(1)*		; visa id: 2723
  %2182 = load i16, i16 addrspace(1)* %2181, align 2		; visa id: 2724
  %2183 = zext i16 %2170 to i32		; visa id: 2726
  %2184 = shl nuw i32 %2183, 16, !spirv.Decorations !888		; visa id: 2727
  %2185 = bitcast i32 %2184 to float
  %2186 = zext i16 %2182 to i32		; visa id: 2728
  %2187 = shl nuw i32 %2186, 16, !spirv.Decorations !888		; visa id: 2729
  %2188 = bitcast i32 %2187 to float
  %2189 = fmul reassoc nsz arcp contract float %2185, %2188, !spirv.Decorations !881
  %2190 = fadd reassoc nsz arcp contract float %2189, %.sroa.190.1, !spirv.Decorations !881		; visa id: 2730
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2731

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %2165
; BB195 :
  %.sroa.190.2 = phi float [ %2190, %2165 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  br i1 %324, label %2191, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2732

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
; BB:
  br label %.preheader.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

2191:                                             ; preds = %._crit_edge.2.15
; BB197 :
  %.sroa.256.0.insert.ext898 = zext i32 %552 to i64		; visa id: 2734
  %2192 = shl nuw nsw i64 %.sroa.256.0.insert.ext898, 1		; visa id: 2735
  %2193 = add i64 %551, %2192		; visa id: 2736
  %2194 = inttoptr i64 %2193 to i16 addrspace(4)*		; visa id: 2737
  %2195 = addrspacecast i16 addrspace(4)* %2194 to i16 addrspace(1)*		; visa id: 2737
  %2196 = load i16, i16 addrspace(1)* %2195, align 2		; visa id: 2738
  %2197 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2198 = extractvalue { i32, i32 } %2197, 0		; visa id: 2740
  %2199 = extractvalue { i32, i32 } %2197, 1		; visa id: 2740
  %2200 = insertelement <2 x i32> undef, i32 %2198, i32 0		; visa id: 2747
  %2201 = insertelement <2 x i32> %2200, i32 %2199, i32 1		; visa id: 2748
  %2202 = bitcast <2 x i32> %2201 to i64		; visa id: 2749
  %2203 = shl i64 %2202, 1		; visa id: 2751
  %2204 = add i64 %.in3822, %2203		; visa id: 2752
  %2205 = add i64 %2204, %sink_3828		; visa id: 2753
  %2206 = inttoptr i64 %2205 to i16 addrspace(4)*		; visa id: 2754
  %2207 = addrspacecast i16 addrspace(4)* %2206 to i16 addrspace(1)*		; visa id: 2754
  %2208 = load i16, i16 addrspace(1)* %2207, align 2		; visa id: 2755
  %2209 = zext i16 %2196 to i32		; visa id: 2757
  %2210 = shl nuw i32 %2209, 16, !spirv.Decorations !888		; visa id: 2758
  %2211 = bitcast i32 %2210 to float
  %2212 = zext i16 %2208 to i32		; visa id: 2759
  %2213 = shl nuw i32 %2212, 16, !spirv.Decorations !888		; visa id: 2760
  %2214 = bitcast i32 %2213 to float
  %2215 = fmul reassoc nsz arcp contract float %2211, %2214, !spirv.Decorations !881
  %2216 = fadd reassoc nsz arcp contract float %2215, %.sroa.254.1, !spirv.Decorations !881		; visa id: 2761
  br label %.preheader.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2762

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %2191
; BB198 :
  %.sroa.254.2 = phi float [ %2216, %2191 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %2217 = add nuw nsw i32 %552, 1, !spirv.Decorations !890		; visa id: 2763
  %2218 = icmp slt i32 %2217, %const_reg_dword2		; visa id: 2764
  br i1 %2218, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2765

.preheader.15..preheader.preheader_crit_edge:     ; preds = %.preheader.15
; BB:
  br label %.preheader.preheader, !stats.blockFrequency.digits !892, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.15
; BB:
  %.sroa.254.2.lcssa = phi float [ %.sroa.254.2, %.preheader.15 ]
  %.sroa.190.2.lcssa = phi float [ %.sroa.190.2, %.preheader.15 ]
  %.sroa.126.2.lcssa = phi float [ %.sroa.126.2, %.preheader.15 ]
  %.sroa.62.2.lcssa = phi float [ %.sroa.62.2, %.preheader.15 ]
  %.sroa.250.2.lcssa = phi float [ %.sroa.250.2, %.preheader.15 ]
  %.sroa.186.2.lcssa = phi float [ %.sroa.186.2, %.preheader.15 ]
  %.sroa.122.2.lcssa = phi float [ %.sroa.122.2, %.preheader.15 ]
  %.sroa.58.2.lcssa = phi float [ %.sroa.58.2, %.preheader.15 ]
  %.sroa.246.2.lcssa = phi float [ %.sroa.246.2, %.preheader.15 ]
  %.sroa.182.2.lcssa = phi float [ %.sroa.182.2, %.preheader.15 ]
  %.sroa.118.2.lcssa = phi float [ %.sroa.118.2, %.preheader.15 ]
  %.sroa.54.2.lcssa = phi float [ %.sroa.54.2, %.preheader.15 ]
  %.sroa.242.2.lcssa = phi float [ %.sroa.242.2, %.preheader.15 ]
  %.sroa.178.2.lcssa = phi float [ %.sroa.178.2, %.preheader.15 ]
  %.sroa.114.2.lcssa = phi float [ %.sroa.114.2, %.preheader.15 ]
  %.sroa.50.2.lcssa = phi float [ %.sroa.50.2, %.preheader.15 ]
  %.sroa.238.2.lcssa = phi float [ %.sroa.238.2, %.preheader.15 ]
  %.sroa.174.2.lcssa = phi float [ %.sroa.174.2, %.preheader.15 ]
  %.sroa.110.2.lcssa = phi float [ %.sroa.110.2, %.preheader.15 ]
  %.sroa.46.2.lcssa = phi float [ %.sroa.46.2, %.preheader.15 ]
  %.sroa.234.2.lcssa = phi float [ %.sroa.234.2, %.preheader.15 ]
  %.sroa.170.2.lcssa = phi float [ %.sroa.170.2, %.preheader.15 ]
  %.sroa.106.2.lcssa = phi float [ %.sroa.106.2, %.preheader.15 ]
  %.sroa.42.2.lcssa = phi float [ %.sroa.42.2, %.preheader.15 ]
  %.sroa.230.2.lcssa = phi float [ %.sroa.230.2, %.preheader.15 ]
  %.sroa.166.2.lcssa = phi float [ %.sroa.166.2, %.preheader.15 ]
  %.sroa.102.2.lcssa = phi float [ %.sroa.102.2, %.preheader.15 ]
  %.sroa.38.2.lcssa = phi float [ %.sroa.38.2, %.preheader.15 ]
  %.sroa.226.2.lcssa = phi float [ %.sroa.226.2, %.preheader.15 ]
  %.sroa.162.2.lcssa = phi float [ %.sroa.162.2, %.preheader.15 ]
  %.sroa.98.2.lcssa = phi float [ %.sroa.98.2, %.preheader.15 ]
  %.sroa.34.2.lcssa = phi float [ %.sroa.34.2, %.preheader.15 ]
  %.sroa.222.2.lcssa = phi float [ %.sroa.222.2, %.preheader.15 ]
  %.sroa.158.2.lcssa = phi float [ %.sroa.158.2, %.preheader.15 ]
  %.sroa.94.2.lcssa = phi float [ %.sroa.94.2, %.preheader.15 ]
  %.sroa.30.2.lcssa = phi float [ %.sroa.30.2, %.preheader.15 ]
  %.sroa.218.2.lcssa = phi float [ %.sroa.218.2, %.preheader.15 ]
  %.sroa.154.2.lcssa = phi float [ %.sroa.154.2, %.preheader.15 ]
  %.sroa.90.2.lcssa = phi float [ %.sroa.90.2, %.preheader.15 ]
  %.sroa.26.2.lcssa = phi float [ %.sroa.26.2, %.preheader.15 ]
  %.sroa.214.2.lcssa = phi float [ %.sroa.214.2, %.preheader.15 ]
  %.sroa.150.2.lcssa = phi float [ %.sroa.150.2, %.preheader.15 ]
  %.sroa.86.2.lcssa = phi float [ %.sroa.86.2, %.preheader.15 ]
  %.sroa.22.2.lcssa = phi float [ %.sroa.22.2, %.preheader.15 ]
  %.sroa.210.2.lcssa = phi float [ %.sroa.210.2, %.preheader.15 ]
  %.sroa.146.2.lcssa = phi float [ %.sroa.146.2, %.preheader.15 ]
  %.sroa.82.2.lcssa = phi float [ %.sroa.82.2, %.preheader.15 ]
  %.sroa.18.2.lcssa = phi float [ %.sroa.18.2, %.preheader.15 ]
  %.sroa.206.2.lcssa = phi float [ %.sroa.206.2, %.preheader.15 ]
  %.sroa.142.2.lcssa = phi float [ %.sroa.142.2, %.preheader.15 ]
  %.sroa.78.2.lcssa = phi float [ %.sroa.78.2, %.preheader.15 ]
  %.sroa.14.2.lcssa = phi float [ %.sroa.14.2, %.preheader.15 ]
  %.sroa.202.2.lcssa = phi float [ %.sroa.202.2, %.preheader.15 ]
  %.sroa.138.2.lcssa = phi float [ %.sroa.138.2, %.preheader.15 ]
  %.sroa.74.2.lcssa = phi float [ %.sroa.74.2, %.preheader.15 ]
  %.sroa.10.2.lcssa = phi float [ %.sroa.10.2, %.preheader.15 ]
  %.sroa.198.2.lcssa = phi float [ %.sroa.198.2, %.preheader.15 ]
  %.sroa.134.2.lcssa = phi float [ %.sroa.134.2, %.preheader.15 ]
  %.sroa.70.2.lcssa = phi float [ %.sroa.70.2, %.preheader.15 ]
  %.sroa.6.2.lcssa = phi float [ %.sroa.6.2, %.preheader.15 ]
  %.sroa.194.2.lcssa = phi float [ %.sroa.194.2, %.preheader.15 ]
  %.sroa.130.2.lcssa = phi float [ %.sroa.130.2, %.preheader.15 ]
  %.sroa.66.2.lcssa = phi float [ %.sroa.66.2, %.preheader.15 ]
  %.sroa.0.2.lcssa = phi float [ %.sroa.0.2, %.preheader.15 ]
  br label %.preheader1.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
; BB201 :
  %.sroa.254.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.254.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.250.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.250.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.246.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.246.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.242.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.242.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.238.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.238.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.234.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.234.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.230.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.230.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.226.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.226.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.222.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.222.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.218.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.218.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.214.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.214.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.210.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.210.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.206.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.206.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.202.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.202.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.198.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.198.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.194.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.194.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.190.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.190.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.186.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.186.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.182.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.182.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.178.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.178.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.174.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.174.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.170.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.170.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.166.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.166.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.162.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.162.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.158.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.158.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.154.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.154.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.150.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.150.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.146.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.146.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.142.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.142.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.138.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.138.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.134.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.134.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.130.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.130.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.126.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.126.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.122.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.122.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.118.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.118.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.114.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.114.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.110.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.110.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.106.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.106.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.102.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.102.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.98.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.98.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.94.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.94.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.90.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.90.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.86.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.86.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.82.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.82.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.78.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.78.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.74.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.74.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.70.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.70.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.66.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.66.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.62.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.62.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.58.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.58.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.54.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.54.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.50.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.50.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.46.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.46.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.42.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.42.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.38.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.38.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.34.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.34.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.30.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.30.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.26.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.26.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.22.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.22.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.18.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.18.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.14.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.14.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.10.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.10.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.6.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.6.2.lcssa, %.preheader1.preheader.loopexit ]
  %.sroa.0.0 = phi float [ %7, %.preheader2.preheader..preheader1.preheader_crit_edge ], [ %.sroa.0.2.lcssa, %.preheader1.preheader.loopexit ]
  %sink_3870 = bitcast <2 x i32> %377 to i64		; visa id: 2767
  %sink_3862 = shl i64 %sink_3870, 2		; visa id: 2769
  %sink_3860 = shl nsw i64 %331, 2		; visa id: 2770
  br i1 %117, label %2219, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2771

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
; BB:
  br label %._crit_edge70, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2219:                                             ; preds = %.preheader1.preheader
; BB203 :
  %2220 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !881		; visa id: 2773
  br i1 %78, label %2225, label %2221, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2774

2221:                                             ; preds = %2219
; BB204 :
  %2222 = add i64 %.in, %372		; visa id: 2776
  %2223 = inttoptr i64 %2222 to float addrspace(4)*		; visa id: 2777
  %2224 = addrspacecast float addrspace(4)* %2223 to float addrspace(1)*		; visa id: 2777
  store float %2220, float addrspace(1)* %2224, align 4		; visa id: 2778
  br label %._crit_edge70, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2779

2225:                                             ; preds = %2219
; BB205 :
  %2226 = add i64 %.in3821, %sink_3862		; visa id: 2781
  %2227 = add i64 %2226, %sink_3860		; visa id: 2782
  %2228 = inttoptr i64 %2227 to float addrspace(4)*		; visa id: 2783
  %2229 = addrspacecast float addrspace(4)* %2228 to float addrspace(1)*		; visa id: 2783
  %2230 = load float, float addrspace(1)* %2229, align 4		; visa id: 2784
  %2231 = fmul reassoc nsz arcp contract float %2230, %4, !spirv.Decorations !881		; visa id: 2785
  %2232 = fadd reassoc nsz arcp contract float %2220, %2231, !spirv.Decorations !881		; visa id: 2786
  %2233 = add i64 %.in, %372		; visa id: 2787
  %2234 = inttoptr i64 %2233 to float addrspace(4)*		; visa id: 2788
  %2235 = addrspacecast float addrspace(4)* %2234 to float addrspace(1)*		; visa id: 2788
  store float %2232, float addrspace(1)* %2235, align 4		; visa id: 2789
  br label %._crit_edge70, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2790

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %2221, %2225
; BB206 :
  %sink_3869 = bitcast <2 x i32> %390 to i64		; visa id: 2791
  %sink_3859 = shl i64 %sink_3869, 2		; visa id: 2793
  br i1 %121, label %2236, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2794

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
; BB:
  br label %._crit_edge70.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2236:                                             ; preds = %._crit_edge70
; BB208 :
  %2237 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !881		; visa id: 2796
  br i1 %78, label %2242, label %2238, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2797

2238:                                             ; preds = %2236
; BB209 :
  %2239 = add i64 %.in, %385		; visa id: 2799
  %2240 = inttoptr i64 %2239 to float addrspace(4)*		; visa id: 2800
  %2241 = addrspacecast float addrspace(4)* %2240 to float addrspace(1)*		; visa id: 2800
  store float %2237, float addrspace(1)* %2241, align 4		; visa id: 2801
  br label %._crit_edge70.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2802

2242:                                             ; preds = %2236
; BB210 :
  %2243 = add i64 %.in3821, %sink_3859		; visa id: 2804
  %2244 = add i64 %2243, %sink_3860		; visa id: 2805
  %2245 = inttoptr i64 %2244 to float addrspace(4)*		; visa id: 2806
  %2246 = addrspacecast float addrspace(4)* %2245 to float addrspace(1)*		; visa id: 2806
  %2247 = load float, float addrspace(1)* %2246, align 4		; visa id: 2807
  %2248 = fmul reassoc nsz arcp contract float %2247, %4, !spirv.Decorations !881		; visa id: 2808
  %2249 = fadd reassoc nsz arcp contract float %2237, %2248, !spirv.Decorations !881		; visa id: 2809
  %2250 = add i64 %.in, %385		; visa id: 2810
  %2251 = inttoptr i64 %2250 to float addrspace(4)*		; visa id: 2811
  %2252 = addrspacecast float addrspace(4)* %2251 to float addrspace(1)*		; visa id: 2811
  store float %2249, float addrspace(1)* %2252, align 4		; visa id: 2812
  br label %._crit_edge70.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2813

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %2242, %2238
; BB211 :
  %sink_3868 = bitcast <2 x i32> %403 to i64		; visa id: 2814
  %sink_3858 = shl i64 %sink_3868, 2		; visa id: 2816
  br i1 %125, label %2253, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2817

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
; BB:
  br label %._crit_edge70.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2253:                                             ; preds = %._crit_edge70.1
; BB213 :
  %2254 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !881		; visa id: 2819
  br i1 %78, label %2259, label %2255, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2820

2255:                                             ; preds = %2253
; BB214 :
  %2256 = add i64 %.in, %398		; visa id: 2822
  %2257 = inttoptr i64 %2256 to float addrspace(4)*		; visa id: 2823
  %2258 = addrspacecast float addrspace(4)* %2257 to float addrspace(1)*		; visa id: 2823
  store float %2254, float addrspace(1)* %2258, align 4		; visa id: 2824
  br label %._crit_edge70.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2825

2259:                                             ; preds = %2253
; BB215 :
  %2260 = add i64 %.in3821, %sink_3858		; visa id: 2827
  %2261 = add i64 %2260, %sink_3860		; visa id: 2828
  %2262 = inttoptr i64 %2261 to float addrspace(4)*		; visa id: 2829
  %2263 = addrspacecast float addrspace(4)* %2262 to float addrspace(1)*		; visa id: 2829
  %2264 = load float, float addrspace(1)* %2263, align 4		; visa id: 2830
  %2265 = fmul reassoc nsz arcp contract float %2264, %4, !spirv.Decorations !881		; visa id: 2831
  %2266 = fadd reassoc nsz arcp contract float %2254, %2265, !spirv.Decorations !881		; visa id: 2832
  %2267 = add i64 %.in, %398		; visa id: 2833
  %2268 = inttoptr i64 %2267 to float addrspace(4)*		; visa id: 2834
  %2269 = addrspacecast float addrspace(4)* %2268 to float addrspace(1)*		; visa id: 2834
  store float %2266, float addrspace(1)* %2269, align 4		; visa id: 2835
  br label %._crit_edge70.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2836

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %2259, %2255
; BB216 :
  %sink_3867 = bitcast <2 x i32> %416 to i64		; visa id: 2837
  %sink_3857 = shl i64 %sink_3867, 2		; visa id: 2839
  br i1 %129, label %2270, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2840

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
; BB:
  br label %.preheader1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2270:                                             ; preds = %._crit_edge70.2
; BB218 :
  %2271 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !881		; visa id: 2842
  br i1 %78, label %2276, label %2272, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2843

2272:                                             ; preds = %2270
; BB219 :
  %2273 = add i64 %.in, %411		; visa id: 2845
  %2274 = inttoptr i64 %2273 to float addrspace(4)*		; visa id: 2846
  %2275 = addrspacecast float addrspace(4)* %2274 to float addrspace(1)*		; visa id: 2846
  store float %2271, float addrspace(1)* %2275, align 4		; visa id: 2847
  br label %.preheader1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2848

2276:                                             ; preds = %2270
; BB220 :
  %2277 = add i64 %.in3821, %sink_3857		; visa id: 2850
  %2278 = add i64 %2277, %sink_3860		; visa id: 2851
  %2279 = inttoptr i64 %2278 to float addrspace(4)*		; visa id: 2852
  %2280 = addrspacecast float addrspace(4)* %2279 to float addrspace(1)*		; visa id: 2852
  %2281 = load float, float addrspace(1)* %2280, align 4		; visa id: 2853
  %2282 = fmul reassoc nsz arcp contract float %2281, %4, !spirv.Decorations !881		; visa id: 2854
  %2283 = fadd reassoc nsz arcp contract float %2271, %2282, !spirv.Decorations !881		; visa id: 2855
  %2284 = add i64 %.in, %411		; visa id: 2856
  %2285 = inttoptr i64 %2284 to float addrspace(4)*		; visa id: 2857
  %2286 = addrspacecast float addrspace(4)* %2285 to float addrspace(1)*		; visa id: 2857
  store float %2283, float addrspace(1)* %2286, align 4		; visa id: 2858
  br label %.preheader1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2859

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %2276, %2272
; BB221 :
  %sink_3855 = shl nsw i64 %350, 2		; visa id: 2860
  br i1 %133, label %2287, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2861

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
; BB:
  br label %._crit_edge70.176, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2287:                                             ; preds = %.preheader1
; BB223 :
  %2288 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !881		; visa id: 2863
  br i1 %78, label %2293, label %2289, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2864

2289:                                             ; preds = %2287
; BB224 :
  %2290 = add i64 %.in, %418		; visa id: 2866
  %2291 = inttoptr i64 %2290 to float addrspace(4)*		; visa id: 2867
  %2292 = addrspacecast float addrspace(4)* %2291 to float addrspace(1)*		; visa id: 2867
  store float %2288, float addrspace(1)* %2292, align 4		; visa id: 2868
  br label %._crit_edge70.176, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2869

2293:                                             ; preds = %2287
; BB225 :
  %2294 = add i64 %.in3821, %sink_3862		; visa id: 2871
  %2295 = add i64 %2294, %sink_3855		; visa id: 2872
  %2296 = inttoptr i64 %2295 to float addrspace(4)*		; visa id: 2873
  %2297 = addrspacecast float addrspace(4)* %2296 to float addrspace(1)*		; visa id: 2873
  %2298 = load float, float addrspace(1)* %2297, align 4		; visa id: 2874
  %2299 = fmul reassoc nsz arcp contract float %2298, %4, !spirv.Decorations !881		; visa id: 2875
  %2300 = fadd reassoc nsz arcp contract float %2288, %2299, !spirv.Decorations !881		; visa id: 2876
  %2301 = add i64 %.in, %418		; visa id: 2877
  %2302 = inttoptr i64 %2301 to float addrspace(4)*		; visa id: 2878
  %2303 = addrspacecast float addrspace(4)* %2302 to float addrspace(1)*		; visa id: 2878
  store float %2300, float addrspace(1)* %2303, align 4		; visa id: 2879
  br label %._crit_edge70.176, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2880

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %2293, %2289
; BB226 :
  br i1 %136, label %2304, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2881

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
; BB:
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2304:                                             ; preds = %._crit_edge70.176
; BB228 :
  %2305 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !881		; visa id: 2883
  br i1 %78, label %2310, label %2306, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2884

2306:                                             ; preds = %2304
; BB229 :
  %2307 = add i64 %.in, %420		; visa id: 2886
  %2308 = inttoptr i64 %2307 to float addrspace(4)*		; visa id: 2887
  %2309 = addrspacecast float addrspace(4)* %2308 to float addrspace(1)*		; visa id: 2887
  store float %2305, float addrspace(1)* %2309, align 4		; visa id: 2888
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2889

2310:                                             ; preds = %2304
; BB230 :
  %2311 = add i64 %.in3821, %sink_3859		; visa id: 2891
  %2312 = add i64 %2311, %sink_3855		; visa id: 2892
  %2313 = inttoptr i64 %2312 to float addrspace(4)*		; visa id: 2893
  %2314 = addrspacecast float addrspace(4)* %2313 to float addrspace(1)*		; visa id: 2893
  %2315 = load float, float addrspace(1)* %2314, align 4		; visa id: 2894
  %2316 = fmul reassoc nsz arcp contract float %2315, %4, !spirv.Decorations !881		; visa id: 2895
  %2317 = fadd reassoc nsz arcp contract float %2305, %2316, !spirv.Decorations !881		; visa id: 2896
  %2318 = add i64 %.in, %420		; visa id: 2897
  %2319 = inttoptr i64 %2318 to float addrspace(4)*		; visa id: 2898
  %2320 = addrspacecast float addrspace(4)* %2319 to float addrspace(1)*		; visa id: 2898
  store float %2317, float addrspace(1)* %2320, align 4		; visa id: 2899
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2900

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %2310, %2306
; BB231 :
  br i1 %139, label %2321, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2901

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
; BB:
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2321:                                             ; preds = %._crit_edge70.1.1
; BB233 :
  %2322 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !881		; visa id: 2903
  br i1 %78, label %2327, label %2323, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2904

2323:                                             ; preds = %2321
; BB234 :
  %2324 = add i64 %.in, %422		; visa id: 2906
  %2325 = inttoptr i64 %2324 to float addrspace(4)*		; visa id: 2907
  %2326 = addrspacecast float addrspace(4)* %2325 to float addrspace(1)*		; visa id: 2907
  store float %2322, float addrspace(1)* %2326, align 4		; visa id: 2908
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2909

2327:                                             ; preds = %2321
; BB235 :
  %2328 = add i64 %.in3821, %sink_3858		; visa id: 2911
  %2329 = add i64 %2328, %sink_3855		; visa id: 2912
  %2330 = inttoptr i64 %2329 to float addrspace(4)*		; visa id: 2913
  %2331 = addrspacecast float addrspace(4)* %2330 to float addrspace(1)*		; visa id: 2913
  %2332 = load float, float addrspace(1)* %2331, align 4		; visa id: 2914
  %2333 = fmul reassoc nsz arcp contract float %2332, %4, !spirv.Decorations !881		; visa id: 2915
  %2334 = fadd reassoc nsz arcp contract float %2322, %2333, !spirv.Decorations !881		; visa id: 2916
  %2335 = add i64 %.in, %422		; visa id: 2917
  %2336 = inttoptr i64 %2335 to float addrspace(4)*		; visa id: 2918
  %2337 = addrspacecast float addrspace(4)* %2336 to float addrspace(1)*		; visa id: 2918
  store float %2334, float addrspace(1)* %2337, align 4		; visa id: 2919
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2920

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %2327, %2323
; BB236 :
  br i1 %142, label %2338, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2921

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
; BB:
  br label %.preheader1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2338:                                             ; preds = %._crit_edge70.2.1
; BB238 :
  %2339 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !881		; visa id: 2923
  br i1 %78, label %2344, label %2340, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2924

2340:                                             ; preds = %2338
; BB239 :
  %2341 = add i64 %.in, %424		; visa id: 2926
  %2342 = inttoptr i64 %2341 to float addrspace(4)*		; visa id: 2927
  %2343 = addrspacecast float addrspace(4)* %2342 to float addrspace(1)*		; visa id: 2927
  store float %2339, float addrspace(1)* %2343, align 4		; visa id: 2928
  br label %.preheader1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2929

2344:                                             ; preds = %2338
; BB240 :
  %2345 = add i64 %.in3821, %sink_3857		; visa id: 2931
  %2346 = add i64 %2345, %sink_3855		; visa id: 2932
  %2347 = inttoptr i64 %2346 to float addrspace(4)*		; visa id: 2933
  %2348 = addrspacecast float addrspace(4)* %2347 to float addrspace(1)*		; visa id: 2933
  %2349 = load float, float addrspace(1)* %2348, align 4		; visa id: 2934
  %2350 = fmul reassoc nsz arcp contract float %2349, %4, !spirv.Decorations !881		; visa id: 2935
  %2351 = fadd reassoc nsz arcp contract float %2339, %2350, !spirv.Decorations !881		; visa id: 2936
  %2352 = add i64 %.in, %424		; visa id: 2937
  %2353 = inttoptr i64 %2352 to float addrspace(4)*		; visa id: 2938
  %2354 = addrspacecast float addrspace(4)* %2353 to float addrspace(1)*		; visa id: 2938
  store float %2351, float addrspace(1)* %2354, align 4		; visa id: 2939
  br label %.preheader1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2940

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %2344, %2340
; BB241 :
  %sink_3853 = shl nsw i64 %351, 2		; visa id: 2941
  br i1 %146, label %2355, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2942

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
; BB:
  br label %._crit_edge70.277, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2355:                                             ; preds = %.preheader1.1
; BB243 :
  %2356 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !881		; visa id: 2944
  br i1 %78, label %2361, label %2357, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2945

2357:                                             ; preds = %2355
; BB244 :
  %2358 = add i64 %.in, %426		; visa id: 2947
  %2359 = inttoptr i64 %2358 to float addrspace(4)*		; visa id: 2948
  %2360 = addrspacecast float addrspace(4)* %2359 to float addrspace(1)*		; visa id: 2948
  store float %2356, float addrspace(1)* %2360, align 4		; visa id: 2949
  br label %._crit_edge70.277, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2950

2361:                                             ; preds = %2355
; BB245 :
  %2362 = add i64 %.in3821, %sink_3862		; visa id: 2952
  %2363 = add i64 %2362, %sink_3853		; visa id: 2953
  %2364 = inttoptr i64 %2363 to float addrspace(4)*		; visa id: 2954
  %2365 = addrspacecast float addrspace(4)* %2364 to float addrspace(1)*		; visa id: 2954
  %2366 = load float, float addrspace(1)* %2365, align 4		; visa id: 2955
  %2367 = fmul reassoc nsz arcp contract float %2366, %4, !spirv.Decorations !881		; visa id: 2956
  %2368 = fadd reassoc nsz arcp contract float %2356, %2367, !spirv.Decorations !881		; visa id: 2957
  %2369 = add i64 %.in, %426		; visa id: 2958
  %2370 = inttoptr i64 %2369 to float addrspace(4)*		; visa id: 2959
  %2371 = addrspacecast float addrspace(4)* %2370 to float addrspace(1)*		; visa id: 2959
  store float %2368, float addrspace(1)* %2371, align 4		; visa id: 2960
  br label %._crit_edge70.277, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2961

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %2361, %2357
; BB246 :
  br i1 %149, label %2372, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2962

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
; BB:
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2372:                                             ; preds = %._crit_edge70.277
; BB248 :
  %2373 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !881		; visa id: 2964
  br i1 %78, label %2378, label %2374, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2965

2374:                                             ; preds = %2372
; BB249 :
  %2375 = add i64 %.in, %428		; visa id: 2967
  %2376 = inttoptr i64 %2375 to float addrspace(4)*		; visa id: 2968
  %2377 = addrspacecast float addrspace(4)* %2376 to float addrspace(1)*		; visa id: 2968
  store float %2373, float addrspace(1)* %2377, align 4		; visa id: 2969
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2970

2378:                                             ; preds = %2372
; BB250 :
  %2379 = add i64 %.in3821, %sink_3859		; visa id: 2972
  %2380 = add i64 %2379, %sink_3853		; visa id: 2973
  %2381 = inttoptr i64 %2380 to float addrspace(4)*		; visa id: 2974
  %2382 = addrspacecast float addrspace(4)* %2381 to float addrspace(1)*		; visa id: 2974
  %2383 = load float, float addrspace(1)* %2382, align 4		; visa id: 2975
  %2384 = fmul reassoc nsz arcp contract float %2383, %4, !spirv.Decorations !881		; visa id: 2976
  %2385 = fadd reassoc nsz arcp contract float %2373, %2384, !spirv.Decorations !881		; visa id: 2977
  %2386 = add i64 %.in, %428		; visa id: 2978
  %2387 = inttoptr i64 %2386 to float addrspace(4)*		; visa id: 2979
  %2388 = addrspacecast float addrspace(4)* %2387 to float addrspace(1)*		; visa id: 2979
  store float %2385, float addrspace(1)* %2388, align 4		; visa id: 2980
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2981

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %2378, %2374
; BB251 :
  br i1 %152, label %2389, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2982

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
; BB:
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2389:                                             ; preds = %._crit_edge70.1.2
; BB253 :
  %2390 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !881		; visa id: 2984
  br i1 %78, label %2395, label %2391, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2985

2391:                                             ; preds = %2389
; BB254 :
  %2392 = add i64 %.in, %430		; visa id: 2987
  %2393 = inttoptr i64 %2392 to float addrspace(4)*		; visa id: 2988
  %2394 = addrspacecast float addrspace(4)* %2393 to float addrspace(1)*		; visa id: 2988
  store float %2390, float addrspace(1)* %2394, align 4		; visa id: 2989
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2990

2395:                                             ; preds = %2389
; BB255 :
  %2396 = add i64 %.in3821, %sink_3858		; visa id: 2992
  %2397 = add i64 %2396, %sink_3853		; visa id: 2993
  %2398 = inttoptr i64 %2397 to float addrspace(4)*		; visa id: 2994
  %2399 = addrspacecast float addrspace(4)* %2398 to float addrspace(1)*		; visa id: 2994
  %2400 = load float, float addrspace(1)* %2399, align 4		; visa id: 2995
  %2401 = fmul reassoc nsz arcp contract float %2400, %4, !spirv.Decorations !881		; visa id: 2996
  %2402 = fadd reassoc nsz arcp contract float %2390, %2401, !spirv.Decorations !881		; visa id: 2997
  %2403 = add i64 %.in, %430		; visa id: 2998
  %2404 = inttoptr i64 %2403 to float addrspace(4)*		; visa id: 2999
  %2405 = addrspacecast float addrspace(4)* %2404 to float addrspace(1)*		; visa id: 2999
  store float %2402, float addrspace(1)* %2405, align 4		; visa id: 3000
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3001

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %2395, %2391
; BB256 :
  br i1 %155, label %2406, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3002

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
; BB:
  br label %.preheader1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2406:                                             ; preds = %._crit_edge70.2.2
; BB258 :
  %2407 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !881		; visa id: 3004
  br i1 %78, label %2412, label %2408, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3005

2408:                                             ; preds = %2406
; BB259 :
  %2409 = add i64 %.in, %432		; visa id: 3007
  %2410 = inttoptr i64 %2409 to float addrspace(4)*		; visa id: 3008
  %2411 = addrspacecast float addrspace(4)* %2410 to float addrspace(1)*		; visa id: 3008
  store float %2407, float addrspace(1)* %2411, align 4		; visa id: 3009
  br label %.preheader1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3010

2412:                                             ; preds = %2406
; BB260 :
  %2413 = add i64 %.in3821, %sink_3857		; visa id: 3012
  %2414 = add i64 %2413, %sink_3853		; visa id: 3013
  %2415 = inttoptr i64 %2414 to float addrspace(4)*		; visa id: 3014
  %2416 = addrspacecast float addrspace(4)* %2415 to float addrspace(1)*		; visa id: 3014
  %2417 = load float, float addrspace(1)* %2416, align 4		; visa id: 3015
  %2418 = fmul reassoc nsz arcp contract float %2417, %4, !spirv.Decorations !881		; visa id: 3016
  %2419 = fadd reassoc nsz arcp contract float %2407, %2418, !spirv.Decorations !881		; visa id: 3017
  %2420 = add i64 %.in, %432		; visa id: 3018
  %2421 = inttoptr i64 %2420 to float addrspace(4)*		; visa id: 3019
  %2422 = addrspacecast float addrspace(4)* %2421 to float addrspace(1)*		; visa id: 3019
  store float %2419, float addrspace(1)* %2422, align 4		; visa id: 3020
  br label %.preheader1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3021

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %2412, %2408
; BB261 :
  %sink_3851 = shl nsw i64 %352, 2		; visa id: 3022
  br i1 %159, label %2423, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3023

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
; BB:
  br label %._crit_edge70.378, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2423:                                             ; preds = %.preheader1.2
; BB263 :
  %2424 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !881		; visa id: 3025
  br i1 %78, label %2429, label %2425, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3026

2425:                                             ; preds = %2423
; BB264 :
  %2426 = add i64 %.in, %434		; visa id: 3028
  %2427 = inttoptr i64 %2426 to float addrspace(4)*		; visa id: 3029
  %2428 = addrspacecast float addrspace(4)* %2427 to float addrspace(1)*		; visa id: 3029
  store float %2424, float addrspace(1)* %2428, align 4		; visa id: 3030
  br label %._crit_edge70.378, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3031

2429:                                             ; preds = %2423
; BB265 :
  %2430 = add i64 %.in3821, %sink_3862		; visa id: 3033
  %2431 = add i64 %2430, %sink_3851		; visa id: 3034
  %2432 = inttoptr i64 %2431 to float addrspace(4)*		; visa id: 3035
  %2433 = addrspacecast float addrspace(4)* %2432 to float addrspace(1)*		; visa id: 3035
  %2434 = load float, float addrspace(1)* %2433, align 4		; visa id: 3036
  %2435 = fmul reassoc nsz arcp contract float %2434, %4, !spirv.Decorations !881		; visa id: 3037
  %2436 = fadd reassoc nsz arcp contract float %2424, %2435, !spirv.Decorations !881		; visa id: 3038
  %2437 = add i64 %.in, %434		; visa id: 3039
  %2438 = inttoptr i64 %2437 to float addrspace(4)*		; visa id: 3040
  %2439 = addrspacecast float addrspace(4)* %2438 to float addrspace(1)*		; visa id: 3040
  store float %2436, float addrspace(1)* %2439, align 4		; visa id: 3041
  br label %._crit_edge70.378, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3042

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %2429, %2425
; BB266 :
  br i1 %162, label %2440, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3043

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
; BB:
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2440:                                             ; preds = %._crit_edge70.378
; BB268 :
  %2441 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !881		; visa id: 3045
  br i1 %78, label %2446, label %2442, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3046

2442:                                             ; preds = %2440
; BB269 :
  %2443 = add i64 %.in, %436		; visa id: 3048
  %2444 = inttoptr i64 %2443 to float addrspace(4)*		; visa id: 3049
  %2445 = addrspacecast float addrspace(4)* %2444 to float addrspace(1)*		; visa id: 3049
  store float %2441, float addrspace(1)* %2445, align 4		; visa id: 3050
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3051

2446:                                             ; preds = %2440
; BB270 :
  %2447 = add i64 %.in3821, %sink_3859		; visa id: 3053
  %2448 = add i64 %2447, %sink_3851		; visa id: 3054
  %2449 = inttoptr i64 %2448 to float addrspace(4)*		; visa id: 3055
  %2450 = addrspacecast float addrspace(4)* %2449 to float addrspace(1)*		; visa id: 3055
  %2451 = load float, float addrspace(1)* %2450, align 4		; visa id: 3056
  %2452 = fmul reassoc nsz arcp contract float %2451, %4, !spirv.Decorations !881		; visa id: 3057
  %2453 = fadd reassoc nsz arcp contract float %2441, %2452, !spirv.Decorations !881		; visa id: 3058
  %2454 = add i64 %.in, %436		; visa id: 3059
  %2455 = inttoptr i64 %2454 to float addrspace(4)*		; visa id: 3060
  %2456 = addrspacecast float addrspace(4)* %2455 to float addrspace(1)*		; visa id: 3060
  store float %2453, float addrspace(1)* %2456, align 4		; visa id: 3061
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3062

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %2446, %2442
; BB271 :
  br i1 %165, label %2457, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3063

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
; BB:
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2457:                                             ; preds = %._crit_edge70.1.3
; BB273 :
  %2458 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !881		; visa id: 3065
  br i1 %78, label %2463, label %2459, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3066

2459:                                             ; preds = %2457
; BB274 :
  %2460 = add i64 %.in, %438		; visa id: 3068
  %2461 = inttoptr i64 %2460 to float addrspace(4)*		; visa id: 3069
  %2462 = addrspacecast float addrspace(4)* %2461 to float addrspace(1)*		; visa id: 3069
  store float %2458, float addrspace(1)* %2462, align 4		; visa id: 3070
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3071

2463:                                             ; preds = %2457
; BB275 :
  %2464 = add i64 %.in3821, %sink_3858		; visa id: 3073
  %2465 = add i64 %2464, %sink_3851		; visa id: 3074
  %2466 = inttoptr i64 %2465 to float addrspace(4)*		; visa id: 3075
  %2467 = addrspacecast float addrspace(4)* %2466 to float addrspace(1)*		; visa id: 3075
  %2468 = load float, float addrspace(1)* %2467, align 4		; visa id: 3076
  %2469 = fmul reassoc nsz arcp contract float %2468, %4, !spirv.Decorations !881		; visa id: 3077
  %2470 = fadd reassoc nsz arcp contract float %2458, %2469, !spirv.Decorations !881		; visa id: 3078
  %2471 = add i64 %.in, %438		; visa id: 3079
  %2472 = inttoptr i64 %2471 to float addrspace(4)*		; visa id: 3080
  %2473 = addrspacecast float addrspace(4)* %2472 to float addrspace(1)*		; visa id: 3080
  store float %2470, float addrspace(1)* %2473, align 4		; visa id: 3081
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3082

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %2463, %2459
; BB276 :
  br i1 %168, label %2474, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3083

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
; BB:
  br label %.preheader1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2474:                                             ; preds = %._crit_edge70.2.3
; BB278 :
  %2475 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !881		; visa id: 3085
  br i1 %78, label %2480, label %2476, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3086

2476:                                             ; preds = %2474
; BB279 :
  %2477 = add i64 %.in, %440		; visa id: 3088
  %2478 = inttoptr i64 %2477 to float addrspace(4)*		; visa id: 3089
  %2479 = addrspacecast float addrspace(4)* %2478 to float addrspace(1)*		; visa id: 3089
  store float %2475, float addrspace(1)* %2479, align 4		; visa id: 3090
  br label %.preheader1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3091

2480:                                             ; preds = %2474
; BB280 :
  %2481 = add i64 %.in3821, %sink_3857		; visa id: 3093
  %2482 = add i64 %2481, %sink_3851		; visa id: 3094
  %2483 = inttoptr i64 %2482 to float addrspace(4)*		; visa id: 3095
  %2484 = addrspacecast float addrspace(4)* %2483 to float addrspace(1)*		; visa id: 3095
  %2485 = load float, float addrspace(1)* %2484, align 4		; visa id: 3096
  %2486 = fmul reassoc nsz arcp contract float %2485, %4, !spirv.Decorations !881		; visa id: 3097
  %2487 = fadd reassoc nsz arcp contract float %2475, %2486, !spirv.Decorations !881		; visa id: 3098
  %2488 = add i64 %.in, %440		; visa id: 3099
  %2489 = inttoptr i64 %2488 to float addrspace(4)*		; visa id: 3100
  %2490 = addrspacecast float addrspace(4)* %2489 to float addrspace(1)*		; visa id: 3100
  store float %2487, float addrspace(1)* %2490, align 4		; visa id: 3101
  br label %.preheader1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3102

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %2480, %2476
; BB281 :
  %sink_3849 = shl nsw i64 %353, 2		; visa id: 3103
  br i1 %172, label %2491, label %.preheader1.3.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3104

.preheader1.3.._crit_edge70.4_crit_edge:          ; preds = %.preheader1.3
; BB:
  br label %._crit_edge70.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2491:                                             ; preds = %.preheader1.3
; BB283 :
  %2492 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !881		; visa id: 3106
  br i1 %78, label %2497, label %2493, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3107

2493:                                             ; preds = %2491
; BB284 :
  %2494 = add i64 %.in, %442		; visa id: 3109
  %2495 = inttoptr i64 %2494 to float addrspace(4)*		; visa id: 3110
  %2496 = addrspacecast float addrspace(4)* %2495 to float addrspace(1)*		; visa id: 3110
  store float %2492, float addrspace(1)* %2496, align 4		; visa id: 3111
  br label %._crit_edge70.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3112

2497:                                             ; preds = %2491
; BB285 :
  %2498 = add i64 %.in3821, %sink_3862		; visa id: 3114
  %2499 = add i64 %2498, %sink_3849		; visa id: 3115
  %2500 = inttoptr i64 %2499 to float addrspace(4)*		; visa id: 3116
  %2501 = addrspacecast float addrspace(4)* %2500 to float addrspace(1)*		; visa id: 3116
  %2502 = load float, float addrspace(1)* %2501, align 4		; visa id: 3117
  %2503 = fmul reassoc nsz arcp contract float %2502, %4, !spirv.Decorations !881		; visa id: 3118
  %2504 = fadd reassoc nsz arcp contract float %2492, %2503, !spirv.Decorations !881		; visa id: 3119
  %2505 = add i64 %.in, %442		; visa id: 3120
  %2506 = inttoptr i64 %2505 to float addrspace(4)*		; visa id: 3121
  %2507 = addrspacecast float addrspace(4)* %2506 to float addrspace(1)*		; visa id: 3121
  store float %2504, float addrspace(1)* %2507, align 4		; visa id: 3122
  br label %._crit_edge70.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3123

._crit_edge70.4:                                  ; preds = %.preheader1.3.._crit_edge70.4_crit_edge, %2497, %2493
; BB286 :
  br i1 %175, label %2508, label %._crit_edge70.4.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3124

._crit_edge70.4.._crit_edge70.1.4_crit_edge:      ; preds = %._crit_edge70.4
; BB:
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2508:                                             ; preds = %._crit_edge70.4
; BB288 :
  %2509 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !881		; visa id: 3126
  br i1 %78, label %2514, label %2510, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3127

2510:                                             ; preds = %2508
; BB289 :
  %2511 = add i64 %.in, %444		; visa id: 3129
  %2512 = inttoptr i64 %2511 to float addrspace(4)*		; visa id: 3130
  %2513 = addrspacecast float addrspace(4)* %2512 to float addrspace(1)*		; visa id: 3130
  store float %2509, float addrspace(1)* %2513, align 4		; visa id: 3131
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3132

2514:                                             ; preds = %2508
; BB290 :
  %2515 = add i64 %.in3821, %sink_3859		; visa id: 3134
  %2516 = add i64 %2515, %sink_3849		; visa id: 3135
  %2517 = inttoptr i64 %2516 to float addrspace(4)*		; visa id: 3136
  %2518 = addrspacecast float addrspace(4)* %2517 to float addrspace(1)*		; visa id: 3136
  %2519 = load float, float addrspace(1)* %2518, align 4		; visa id: 3137
  %2520 = fmul reassoc nsz arcp contract float %2519, %4, !spirv.Decorations !881		; visa id: 3138
  %2521 = fadd reassoc nsz arcp contract float %2509, %2520, !spirv.Decorations !881		; visa id: 3139
  %2522 = add i64 %.in, %444		; visa id: 3140
  %2523 = inttoptr i64 %2522 to float addrspace(4)*		; visa id: 3141
  %2524 = addrspacecast float addrspace(4)* %2523 to float addrspace(1)*		; visa id: 3141
  store float %2521, float addrspace(1)* %2524, align 4		; visa id: 3142
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3143

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4.._crit_edge70.1.4_crit_edge, %2514, %2510
; BB291 :
  br i1 %178, label %2525, label %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3144

._crit_edge70.1.4.._crit_edge70.2.4_crit_edge:    ; preds = %._crit_edge70.1.4
; BB:
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2525:                                             ; preds = %._crit_edge70.1.4
; BB293 :
  %2526 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !881		; visa id: 3146
  br i1 %78, label %2531, label %2527, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3147

2527:                                             ; preds = %2525
; BB294 :
  %2528 = add i64 %.in, %446		; visa id: 3149
  %2529 = inttoptr i64 %2528 to float addrspace(4)*		; visa id: 3150
  %2530 = addrspacecast float addrspace(4)* %2529 to float addrspace(1)*		; visa id: 3150
  store float %2526, float addrspace(1)* %2530, align 4		; visa id: 3151
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3152

2531:                                             ; preds = %2525
; BB295 :
  %2532 = add i64 %.in3821, %sink_3858		; visa id: 3154
  %2533 = add i64 %2532, %sink_3849		; visa id: 3155
  %2534 = inttoptr i64 %2533 to float addrspace(4)*		; visa id: 3156
  %2535 = addrspacecast float addrspace(4)* %2534 to float addrspace(1)*		; visa id: 3156
  %2536 = load float, float addrspace(1)* %2535, align 4		; visa id: 3157
  %2537 = fmul reassoc nsz arcp contract float %2536, %4, !spirv.Decorations !881		; visa id: 3158
  %2538 = fadd reassoc nsz arcp contract float %2526, %2537, !spirv.Decorations !881		; visa id: 3159
  %2539 = add i64 %.in, %446		; visa id: 3160
  %2540 = inttoptr i64 %2539 to float addrspace(4)*		; visa id: 3161
  %2541 = addrspacecast float addrspace(4)* %2540 to float addrspace(1)*		; visa id: 3161
  store float %2538, float addrspace(1)* %2541, align 4		; visa id: 3162
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3163

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, %2531, %2527
; BB296 :
  br i1 %181, label %2542, label %._crit_edge70.2.4..preheader1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3164

._crit_edge70.2.4..preheader1.4_crit_edge:        ; preds = %._crit_edge70.2.4
; BB:
  br label %.preheader1.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2542:                                             ; preds = %._crit_edge70.2.4
; BB298 :
  %2543 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !881		; visa id: 3166
  br i1 %78, label %2548, label %2544, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3167

2544:                                             ; preds = %2542
; BB299 :
  %2545 = add i64 %.in, %448		; visa id: 3169
  %2546 = inttoptr i64 %2545 to float addrspace(4)*		; visa id: 3170
  %2547 = addrspacecast float addrspace(4)* %2546 to float addrspace(1)*		; visa id: 3170
  store float %2543, float addrspace(1)* %2547, align 4		; visa id: 3171
  br label %.preheader1.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3172

2548:                                             ; preds = %2542
; BB300 :
  %2549 = add i64 %.in3821, %sink_3857		; visa id: 3174
  %2550 = add i64 %2549, %sink_3849		; visa id: 3175
  %2551 = inttoptr i64 %2550 to float addrspace(4)*		; visa id: 3176
  %2552 = addrspacecast float addrspace(4)* %2551 to float addrspace(1)*		; visa id: 3176
  %2553 = load float, float addrspace(1)* %2552, align 4		; visa id: 3177
  %2554 = fmul reassoc nsz arcp contract float %2553, %4, !spirv.Decorations !881		; visa id: 3178
  %2555 = fadd reassoc nsz arcp contract float %2543, %2554, !spirv.Decorations !881		; visa id: 3179
  %2556 = add i64 %.in, %448		; visa id: 3180
  %2557 = inttoptr i64 %2556 to float addrspace(4)*		; visa id: 3181
  %2558 = addrspacecast float addrspace(4)* %2557 to float addrspace(1)*		; visa id: 3181
  store float %2555, float addrspace(1)* %2558, align 4		; visa id: 3182
  br label %.preheader1.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3183

.preheader1.4:                                    ; preds = %._crit_edge70.2.4..preheader1.4_crit_edge, %2548, %2544
; BB301 :
  %sink_3847 = shl nsw i64 %354, 2		; visa id: 3184
  br i1 %185, label %2559, label %.preheader1.4.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3185

.preheader1.4.._crit_edge70.5_crit_edge:          ; preds = %.preheader1.4
; BB:
  br label %._crit_edge70.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2559:                                             ; preds = %.preheader1.4
; BB303 :
  %2560 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !881		; visa id: 3187
  br i1 %78, label %2565, label %2561, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3188

2561:                                             ; preds = %2559
; BB304 :
  %2562 = add i64 %.in, %450		; visa id: 3190
  %2563 = inttoptr i64 %2562 to float addrspace(4)*		; visa id: 3191
  %2564 = addrspacecast float addrspace(4)* %2563 to float addrspace(1)*		; visa id: 3191
  store float %2560, float addrspace(1)* %2564, align 4		; visa id: 3192
  br label %._crit_edge70.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3193

2565:                                             ; preds = %2559
; BB305 :
  %2566 = add i64 %.in3821, %sink_3862		; visa id: 3195
  %2567 = add i64 %2566, %sink_3847		; visa id: 3196
  %2568 = inttoptr i64 %2567 to float addrspace(4)*		; visa id: 3197
  %2569 = addrspacecast float addrspace(4)* %2568 to float addrspace(1)*		; visa id: 3197
  %2570 = load float, float addrspace(1)* %2569, align 4		; visa id: 3198
  %2571 = fmul reassoc nsz arcp contract float %2570, %4, !spirv.Decorations !881		; visa id: 3199
  %2572 = fadd reassoc nsz arcp contract float %2560, %2571, !spirv.Decorations !881		; visa id: 3200
  %2573 = add i64 %.in, %450		; visa id: 3201
  %2574 = inttoptr i64 %2573 to float addrspace(4)*		; visa id: 3202
  %2575 = addrspacecast float addrspace(4)* %2574 to float addrspace(1)*		; visa id: 3202
  store float %2572, float addrspace(1)* %2575, align 4		; visa id: 3203
  br label %._crit_edge70.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3204

._crit_edge70.5:                                  ; preds = %.preheader1.4.._crit_edge70.5_crit_edge, %2565, %2561
; BB306 :
  br i1 %188, label %2576, label %._crit_edge70.5.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3205

._crit_edge70.5.._crit_edge70.1.5_crit_edge:      ; preds = %._crit_edge70.5
; BB:
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2576:                                             ; preds = %._crit_edge70.5
; BB308 :
  %2577 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !881		; visa id: 3207
  br i1 %78, label %2582, label %2578, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3208

2578:                                             ; preds = %2576
; BB309 :
  %2579 = add i64 %.in, %452		; visa id: 3210
  %2580 = inttoptr i64 %2579 to float addrspace(4)*		; visa id: 3211
  %2581 = addrspacecast float addrspace(4)* %2580 to float addrspace(1)*		; visa id: 3211
  store float %2577, float addrspace(1)* %2581, align 4		; visa id: 3212
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3213

2582:                                             ; preds = %2576
; BB310 :
  %2583 = add i64 %.in3821, %sink_3859		; visa id: 3215
  %2584 = add i64 %2583, %sink_3847		; visa id: 3216
  %2585 = inttoptr i64 %2584 to float addrspace(4)*		; visa id: 3217
  %2586 = addrspacecast float addrspace(4)* %2585 to float addrspace(1)*		; visa id: 3217
  %2587 = load float, float addrspace(1)* %2586, align 4		; visa id: 3218
  %2588 = fmul reassoc nsz arcp contract float %2587, %4, !spirv.Decorations !881		; visa id: 3219
  %2589 = fadd reassoc nsz arcp contract float %2577, %2588, !spirv.Decorations !881		; visa id: 3220
  %2590 = add i64 %.in, %452		; visa id: 3221
  %2591 = inttoptr i64 %2590 to float addrspace(4)*		; visa id: 3222
  %2592 = addrspacecast float addrspace(4)* %2591 to float addrspace(1)*		; visa id: 3222
  store float %2589, float addrspace(1)* %2592, align 4		; visa id: 3223
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3224

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5.._crit_edge70.1.5_crit_edge, %2582, %2578
; BB311 :
  br i1 %191, label %2593, label %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3225

._crit_edge70.1.5.._crit_edge70.2.5_crit_edge:    ; preds = %._crit_edge70.1.5
; BB:
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2593:                                             ; preds = %._crit_edge70.1.5
; BB313 :
  %2594 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !881		; visa id: 3227
  br i1 %78, label %2599, label %2595, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3228

2595:                                             ; preds = %2593
; BB314 :
  %2596 = add i64 %.in, %454		; visa id: 3230
  %2597 = inttoptr i64 %2596 to float addrspace(4)*		; visa id: 3231
  %2598 = addrspacecast float addrspace(4)* %2597 to float addrspace(1)*		; visa id: 3231
  store float %2594, float addrspace(1)* %2598, align 4		; visa id: 3232
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3233

2599:                                             ; preds = %2593
; BB315 :
  %2600 = add i64 %.in3821, %sink_3858		; visa id: 3235
  %2601 = add i64 %2600, %sink_3847		; visa id: 3236
  %2602 = inttoptr i64 %2601 to float addrspace(4)*		; visa id: 3237
  %2603 = addrspacecast float addrspace(4)* %2602 to float addrspace(1)*		; visa id: 3237
  %2604 = load float, float addrspace(1)* %2603, align 4		; visa id: 3238
  %2605 = fmul reassoc nsz arcp contract float %2604, %4, !spirv.Decorations !881		; visa id: 3239
  %2606 = fadd reassoc nsz arcp contract float %2594, %2605, !spirv.Decorations !881		; visa id: 3240
  %2607 = add i64 %.in, %454		; visa id: 3241
  %2608 = inttoptr i64 %2607 to float addrspace(4)*		; visa id: 3242
  %2609 = addrspacecast float addrspace(4)* %2608 to float addrspace(1)*		; visa id: 3242
  store float %2606, float addrspace(1)* %2609, align 4		; visa id: 3243
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3244

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, %2599, %2595
; BB316 :
  br i1 %194, label %2610, label %._crit_edge70.2.5..preheader1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3245

._crit_edge70.2.5..preheader1.5_crit_edge:        ; preds = %._crit_edge70.2.5
; BB:
  br label %.preheader1.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2610:                                             ; preds = %._crit_edge70.2.5
; BB318 :
  %2611 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !881		; visa id: 3247
  br i1 %78, label %2616, label %2612, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3248

2612:                                             ; preds = %2610
; BB319 :
  %2613 = add i64 %.in, %456		; visa id: 3250
  %2614 = inttoptr i64 %2613 to float addrspace(4)*		; visa id: 3251
  %2615 = addrspacecast float addrspace(4)* %2614 to float addrspace(1)*		; visa id: 3251
  store float %2611, float addrspace(1)* %2615, align 4		; visa id: 3252
  br label %.preheader1.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3253

2616:                                             ; preds = %2610
; BB320 :
  %2617 = add i64 %.in3821, %sink_3857		; visa id: 3255
  %2618 = add i64 %2617, %sink_3847		; visa id: 3256
  %2619 = inttoptr i64 %2618 to float addrspace(4)*		; visa id: 3257
  %2620 = addrspacecast float addrspace(4)* %2619 to float addrspace(1)*		; visa id: 3257
  %2621 = load float, float addrspace(1)* %2620, align 4		; visa id: 3258
  %2622 = fmul reassoc nsz arcp contract float %2621, %4, !spirv.Decorations !881		; visa id: 3259
  %2623 = fadd reassoc nsz arcp contract float %2611, %2622, !spirv.Decorations !881		; visa id: 3260
  %2624 = add i64 %.in, %456		; visa id: 3261
  %2625 = inttoptr i64 %2624 to float addrspace(4)*		; visa id: 3262
  %2626 = addrspacecast float addrspace(4)* %2625 to float addrspace(1)*		; visa id: 3262
  store float %2623, float addrspace(1)* %2626, align 4		; visa id: 3263
  br label %.preheader1.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3264

.preheader1.5:                                    ; preds = %._crit_edge70.2.5..preheader1.5_crit_edge, %2616, %2612
; BB321 :
  %sink_3845 = shl nsw i64 %355, 2		; visa id: 3265
  br i1 %198, label %2627, label %.preheader1.5.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3266

.preheader1.5.._crit_edge70.6_crit_edge:          ; preds = %.preheader1.5
; BB:
  br label %._crit_edge70.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2627:                                             ; preds = %.preheader1.5
; BB323 :
  %2628 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !881		; visa id: 3268
  br i1 %78, label %2633, label %2629, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3269

2629:                                             ; preds = %2627
; BB324 :
  %2630 = add i64 %.in, %458		; visa id: 3271
  %2631 = inttoptr i64 %2630 to float addrspace(4)*		; visa id: 3272
  %2632 = addrspacecast float addrspace(4)* %2631 to float addrspace(1)*		; visa id: 3272
  store float %2628, float addrspace(1)* %2632, align 4		; visa id: 3273
  br label %._crit_edge70.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3274

2633:                                             ; preds = %2627
; BB325 :
  %2634 = add i64 %.in3821, %sink_3862		; visa id: 3276
  %2635 = add i64 %2634, %sink_3845		; visa id: 3277
  %2636 = inttoptr i64 %2635 to float addrspace(4)*		; visa id: 3278
  %2637 = addrspacecast float addrspace(4)* %2636 to float addrspace(1)*		; visa id: 3278
  %2638 = load float, float addrspace(1)* %2637, align 4		; visa id: 3279
  %2639 = fmul reassoc nsz arcp contract float %2638, %4, !spirv.Decorations !881		; visa id: 3280
  %2640 = fadd reassoc nsz arcp contract float %2628, %2639, !spirv.Decorations !881		; visa id: 3281
  %2641 = add i64 %.in, %458		; visa id: 3282
  %2642 = inttoptr i64 %2641 to float addrspace(4)*		; visa id: 3283
  %2643 = addrspacecast float addrspace(4)* %2642 to float addrspace(1)*		; visa id: 3283
  store float %2640, float addrspace(1)* %2643, align 4		; visa id: 3284
  br label %._crit_edge70.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3285

._crit_edge70.6:                                  ; preds = %.preheader1.5.._crit_edge70.6_crit_edge, %2633, %2629
; BB326 :
  br i1 %201, label %2644, label %._crit_edge70.6.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3286

._crit_edge70.6.._crit_edge70.1.6_crit_edge:      ; preds = %._crit_edge70.6
; BB:
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2644:                                             ; preds = %._crit_edge70.6
; BB328 :
  %2645 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !881		; visa id: 3288
  br i1 %78, label %2650, label %2646, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3289

2646:                                             ; preds = %2644
; BB329 :
  %2647 = add i64 %.in, %460		; visa id: 3291
  %2648 = inttoptr i64 %2647 to float addrspace(4)*		; visa id: 3292
  %2649 = addrspacecast float addrspace(4)* %2648 to float addrspace(1)*		; visa id: 3292
  store float %2645, float addrspace(1)* %2649, align 4		; visa id: 3293
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3294

2650:                                             ; preds = %2644
; BB330 :
  %2651 = add i64 %.in3821, %sink_3859		; visa id: 3296
  %2652 = add i64 %2651, %sink_3845		; visa id: 3297
  %2653 = inttoptr i64 %2652 to float addrspace(4)*		; visa id: 3298
  %2654 = addrspacecast float addrspace(4)* %2653 to float addrspace(1)*		; visa id: 3298
  %2655 = load float, float addrspace(1)* %2654, align 4		; visa id: 3299
  %2656 = fmul reassoc nsz arcp contract float %2655, %4, !spirv.Decorations !881		; visa id: 3300
  %2657 = fadd reassoc nsz arcp contract float %2645, %2656, !spirv.Decorations !881		; visa id: 3301
  %2658 = add i64 %.in, %460		; visa id: 3302
  %2659 = inttoptr i64 %2658 to float addrspace(4)*		; visa id: 3303
  %2660 = addrspacecast float addrspace(4)* %2659 to float addrspace(1)*		; visa id: 3303
  store float %2657, float addrspace(1)* %2660, align 4		; visa id: 3304
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3305

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6.._crit_edge70.1.6_crit_edge, %2650, %2646
; BB331 :
  br i1 %204, label %2661, label %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3306

._crit_edge70.1.6.._crit_edge70.2.6_crit_edge:    ; preds = %._crit_edge70.1.6
; BB:
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2661:                                             ; preds = %._crit_edge70.1.6
; BB333 :
  %2662 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !881		; visa id: 3308
  br i1 %78, label %2667, label %2663, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3309

2663:                                             ; preds = %2661
; BB334 :
  %2664 = add i64 %.in, %462		; visa id: 3311
  %2665 = inttoptr i64 %2664 to float addrspace(4)*		; visa id: 3312
  %2666 = addrspacecast float addrspace(4)* %2665 to float addrspace(1)*		; visa id: 3312
  store float %2662, float addrspace(1)* %2666, align 4		; visa id: 3313
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3314

2667:                                             ; preds = %2661
; BB335 :
  %2668 = add i64 %.in3821, %sink_3858		; visa id: 3316
  %2669 = add i64 %2668, %sink_3845		; visa id: 3317
  %2670 = inttoptr i64 %2669 to float addrspace(4)*		; visa id: 3318
  %2671 = addrspacecast float addrspace(4)* %2670 to float addrspace(1)*		; visa id: 3318
  %2672 = load float, float addrspace(1)* %2671, align 4		; visa id: 3319
  %2673 = fmul reassoc nsz arcp contract float %2672, %4, !spirv.Decorations !881		; visa id: 3320
  %2674 = fadd reassoc nsz arcp contract float %2662, %2673, !spirv.Decorations !881		; visa id: 3321
  %2675 = add i64 %.in, %462		; visa id: 3322
  %2676 = inttoptr i64 %2675 to float addrspace(4)*		; visa id: 3323
  %2677 = addrspacecast float addrspace(4)* %2676 to float addrspace(1)*		; visa id: 3323
  store float %2674, float addrspace(1)* %2677, align 4		; visa id: 3324
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3325

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, %2667, %2663
; BB336 :
  br i1 %207, label %2678, label %._crit_edge70.2.6..preheader1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3326

._crit_edge70.2.6..preheader1.6_crit_edge:        ; preds = %._crit_edge70.2.6
; BB:
  br label %.preheader1.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2678:                                             ; preds = %._crit_edge70.2.6
; BB338 :
  %2679 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !881		; visa id: 3328
  br i1 %78, label %2684, label %2680, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3329

2680:                                             ; preds = %2678
; BB339 :
  %2681 = add i64 %.in, %464		; visa id: 3331
  %2682 = inttoptr i64 %2681 to float addrspace(4)*		; visa id: 3332
  %2683 = addrspacecast float addrspace(4)* %2682 to float addrspace(1)*		; visa id: 3332
  store float %2679, float addrspace(1)* %2683, align 4		; visa id: 3333
  br label %.preheader1.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3334

2684:                                             ; preds = %2678
; BB340 :
  %2685 = add i64 %.in3821, %sink_3857		; visa id: 3336
  %2686 = add i64 %2685, %sink_3845		; visa id: 3337
  %2687 = inttoptr i64 %2686 to float addrspace(4)*		; visa id: 3338
  %2688 = addrspacecast float addrspace(4)* %2687 to float addrspace(1)*		; visa id: 3338
  %2689 = load float, float addrspace(1)* %2688, align 4		; visa id: 3339
  %2690 = fmul reassoc nsz arcp contract float %2689, %4, !spirv.Decorations !881		; visa id: 3340
  %2691 = fadd reassoc nsz arcp contract float %2679, %2690, !spirv.Decorations !881		; visa id: 3341
  %2692 = add i64 %.in, %464		; visa id: 3342
  %2693 = inttoptr i64 %2692 to float addrspace(4)*		; visa id: 3343
  %2694 = addrspacecast float addrspace(4)* %2693 to float addrspace(1)*		; visa id: 3343
  store float %2691, float addrspace(1)* %2694, align 4		; visa id: 3344
  br label %.preheader1.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3345

.preheader1.6:                                    ; preds = %._crit_edge70.2.6..preheader1.6_crit_edge, %2684, %2680
; BB341 :
  %sink_3843 = shl nsw i64 %356, 2		; visa id: 3346
  br i1 %211, label %2695, label %.preheader1.6.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3347

.preheader1.6.._crit_edge70.7_crit_edge:          ; preds = %.preheader1.6
; BB:
  br label %._crit_edge70.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2695:                                             ; preds = %.preheader1.6
; BB343 :
  %2696 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !881		; visa id: 3349
  br i1 %78, label %2701, label %2697, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3350

2697:                                             ; preds = %2695
; BB344 :
  %2698 = add i64 %.in, %466		; visa id: 3352
  %2699 = inttoptr i64 %2698 to float addrspace(4)*		; visa id: 3353
  %2700 = addrspacecast float addrspace(4)* %2699 to float addrspace(1)*		; visa id: 3353
  store float %2696, float addrspace(1)* %2700, align 4		; visa id: 3354
  br label %._crit_edge70.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3355

2701:                                             ; preds = %2695
; BB345 :
  %2702 = add i64 %.in3821, %sink_3862		; visa id: 3357
  %2703 = add i64 %2702, %sink_3843		; visa id: 3358
  %2704 = inttoptr i64 %2703 to float addrspace(4)*		; visa id: 3359
  %2705 = addrspacecast float addrspace(4)* %2704 to float addrspace(1)*		; visa id: 3359
  %2706 = load float, float addrspace(1)* %2705, align 4		; visa id: 3360
  %2707 = fmul reassoc nsz arcp contract float %2706, %4, !spirv.Decorations !881		; visa id: 3361
  %2708 = fadd reassoc nsz arcp contract float %2696, %2707, !spirv.Decorations !881		; visa id: 3362
  %2709 = add i64 %.in, %466		; visa id: 3363
  %2710 = inttoptr i64 %2709 to float addrspace(4)*		; visa id: 3364
  %2711 = addrspacecast float addrspace(4)* %2710 to float addrspace(1)*		; visa id: 3364
  store float %2708, float addrspace(1)* %2711, align 4		; visa id: 3365
  br label %._crit_edge70.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3366

._crit_edge70.7:                                  ; preds = %.preheader1.6.._crit_edge70.7_crit_edge, %2701, %2697
; BB346 :
  br i1 %214, label %2712, label %._crit_edge70.7.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3367

._crit_edge70.7.._crit_edge70.1.7_crit_edge:      ; preds = %._crit_edge70.7
; BB:
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2712:                                             ; preds = %._crit_edge70.7
; BB348 :
  %2713 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !881		; visa id: 3369
  br i1 %78, label %2718, label %2714, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3370

2714:                                             ; preds = %2712
; BB349 :
  %2715 = add i64 %.in, %468		; visa id: 3372
  %2716 = inttoptr i64 %2715 to float addrspace(4)*		; visa id: 3373
  %2717 = addrspacecast float addrspace(4)* %2716 to float addrspace(1)*		; visa id: 3373
  store float %2713, float addrspace(1)* %2717, align 4		; visa id: 3374
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3375

2718:                                             ; preds = %2712
; BB350 :
  %2719 = add i64 %.in3821, %sink_3859		; visa id: 3377
  %2720 = add i64 %2719, %sink_3843		; visa id: 3378
  %2721 = inttoptr i64 %2720 to float addrspace(4)*		; visa id: 3379
  %2722 = addrspacecast float addrspace(4)* %2721 to float addrspace(1)*		; visa id: 3379
  %2723 = load float, float addrspace(1)* %2722, align 4		; visa id: 3380
  %2724 = fmul reassoc nsz arcp contract float %2723, %4, !spirv.Decorations !881		; visa id: 3381
  %2725 = fadd reassoc nsz arcp contract float %2713, %2724, !spirv.Decorations !881		; visa id: 3382
  %2726 = add i64 %.in, %468		; visa id: 3383
  %2727 = inttoptr i64 %2726 to float addrspace(4)*		; visa id: 3384
  %2728 = addrspacecast float addrspace(4)* %2727 to float addrspace(1)*		; visa id: 3384
  store float %2725, float addrspace(1)* %2728, align 4		; visa id: 3385
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3386

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7.._crit_edge70.1.7_crit_edge, %2718, %2714
; BB351 :
  br i1 %217, label %2729, label %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3387

._crit_edge70.1.7.._crit_edge70.2.7_crit_edge:    ; preds = %._crit_edge70.1.7
; BB:
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2729:                                             ; preds = %._crit_edge70.1.7
; BB353 :
  %2730 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !881		; visa id: 3389
  br i1 %78, label %2735, label %2731, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3390

2731:                                             ; preds = %2729
; BB354 :
  %2732 = add i64 %.in, %470		; visa id: 3392
  %2733 = inttoptr i64 %2732 to float addrspace(4)*		; visa id: 3393
  %2734 = addrspacecast float addrspace(4)* %2733 to float addrspace(1)*		; visa id: 3393
  store float %2730, float addrspace(1)* %2734, align 4		; visa id: 3394
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3395

2735:                                             ; preds = %2729
; BB355 :
  %2736 = add i64 %.in3821, %sink_3858		; visa id: 3397
  %2737 = add i64 %2736, %sink_3843		; visa id: 3398
  %2738 = inttoptr i64 %2737 to float addrspace(4)*		; visa id: 3399
  %2739 = addrspacecast float addrspace(4)* %2738 to float addrspace(1)*		; visa id: 3399
  %2740 = load float, float addrspace(1)* %2739, align 4		; visa id: 3400
  %2741 = fmul reassoc nsz arcp contract float %2740, %4, !spirv.Decorations !881		; visa id: 3401
  %2742 = fadd reassoc nsz arcp contract float %2730, %2741, !spirv.Decorations !881		; visa id: 3402
  %2743 = add i64 %.in, %470		; visa id: 3403
  %2744 = inttoptr i64 %2743 to float addrspace(4)*		; visa id: 3404
  %2745 = addrspacecast float addrspace(4)* %2744 to float addrspace(1)*		; visa id: 3404
  store float %2742, float addrspace(1)* %2745, align 4		; visa id: 3405
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3406

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, %2735, %2731
; BB356 :
  br i1 %220, label %2746, label %._crit_edge70.2.7..preheader1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3407

._crit_edge70.2.7..preheader1.7_crit_edge:        ; preds = %._crit_edge70.2.7
; BB:
  br label %.preheader1.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2746:                                             ; preds = %._crit_edge70.2.7
; BB358 :
  %2747 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !881		; visa id: 3409
  br i1 %78, label %2752, label %2748, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3410

2748:                                             ; preds = %2746
; BB359 :
  %2749 = add i64 %.in, %472		; visa id: 3412
  %2750 = inttoptr i64 %2749 to float addrspace(4)*		; visa id: 3413
  %2751 = addrspacecast float addrspace(4)* %2750 to float addrspace(1)*		; visa id: 3413
  store float %2747, float addrspace(1)* %2751, align 4		; visa id: 3414
  br label %.preheader1.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3415

2752:                                             ; preds = %2746
; BB360 :
  %2753 = add i64 %.in3821, %sink_3857		; visa id: 3417
  %2754 = add i64 %2753, %sink_3843		; visa id: 3418
  %2755 = inttoptr i64 %2754 to float addrspace(4)*		; visa id: 3419
  %2756 = addrspacecast float addrspace(4)* %2755 to float addrspace(1)*		; visa id: 3419
  %2757 = load float, float addrspace(1)* %2756, align 4		; visa id: 3420
  %2758 = fmul reassoc nsz arcp contract float %2757, %4, !spirv.Decorations !881		; visa id: 3421
  %2759 = fadd reassoc nsz arcp contract float %2747, %2758, !spirv.Decorations !881		; visa id: 3422
  %2760 = add i64 %.in, %472		; visa id: 3423
  %2761 = inttoptr i64 %2760 to float addrspace(4)*		; visa id: 3424
  %2762 = addrspacecast float addrspace(4)* %2761 to float addrspace(1)*		; visa id: 3424
  store float %2759, float addrspace(1)* %2762, align 4		; visa id: 3425
  br label %.preheader1.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3426

.preheader1.7:                                    ; preds = %._crit_edge70.2.7..preheader1.7_crit_edge, %2752, %2748
; BB361 :
  %sink_3841 = shl nsw i64 %357, 2		; visa id: 3427
  br i1 %224, label %2763, label %.preheader1.7.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3428

.preheader1.7.._crit_edge70.8_crit_edge:          ; preds = %.preheader1.7
; BB:
  br label %._crit_edge70.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2763:                                             ; preds = %.preheader1.7
; BB363 :
  %2764 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !881		; visa id: 3430
  br i1 %78, label %2769, label %2765, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3431

2765:                                             ; preds = %2763
; BB364 :
  %2766 = add i64 %.in, %474		; visa id: 3433
  %2767 = inttoptr i64 %2766 to float addrspace(4)*		; visa id: 3434
  %2768 = addrspacecast float addrspace(4)* %2767 to float addrspace(1)*		; visa id: 3434
  store float %2764, float addrspace(1)* %2768, align 4		; visa id: 3435
  br label %._crit_edge70.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3436

2769:                                             ; preds = %2763
; BB365 :
  %2770 = add i64 %.in3821, %sink_3862		; visa id: 3438
  %2771 = add i64 %2770, %sink_3841		; visa id: 3439
  %2772 = inttoptr i64 %2771 to float addrspace(4)*		; visa id: 3440
  %2773 = addrspacecast float addrspace(4)* %2772 to float addrspace(1)*		; visa id: 3440
  %2774 = load float, float addrspace(1)* %2773, align 4		; visa id: 3441
  %2775 = fmul reassoc nsz arcp contract float %2774, %4, !spirv.Decorations !881		; visa id: 3442
  %2776 = fadd reassoc nsz arcp contract float %2764, %2775, !spirv.Decorations !881		; visa id: 3443
  %2777 = add i64 %.in, %474		; visa id: 3444
  %2778 = inttoptr i64 %2777 to float addrspace(4)*		; visa id: 3445
  %2779 = addrspacecast float addrspace(4)* %2778 to float addrspace(1)*		; visa id: 3445
  store float %2776, float addrspace(1)* %2779, align 4		; visa id: 3446
  br label %._crit_edge70.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3447

._crit_edge70.8:                                  ; preds = %.preheader1.7.._crit_edge70.8_crit_edge, %2769, %2765
; BB366 :
  br i1 %227, label %2780, label %._crit_edge70.8.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3448

._crit_edge70.8.._crit_edge70.1.8_crit_edge:      ; preds = %._crit_edge70.8
; BB:
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2780:                                             ; preds = %._crit_edge70.8
; BB368 :
  %2781 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !881		; visa id: 3450
  br i1 %78, label %2786, label %2782, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3451

2782:                                             ; preds = %2780
; BB369 :
  %2783 = add i64 %.in, %476		; visa id: 3453
  %2784 = inttoptr i64 %2783 to float addrspace(4)*		; visa id: 3454
  %2785 = addrspacecast float addrspace(4)* %2784 to float addrspace(1)*		; visa id: 3454
  store float %2781, float addrspace(1)* %2785, align 4		; visa id: 3455
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3456

2786:                                             ; preds = %2780
; BB370 :
  %2787 = add i64 %.in3821, %sink_3859		; visa id: 3458
  %2788 = add i64 %2787, %sink_3841		; visa id: 3459
  %2789 = inttoptr i64 %2788 to float addrspace(4)*		; visa id: 3460
  %2790 = addrspacecast float addrspace(4)* %2789 to float addrspace(1)*		; visa id: 3460
  %2791 = load float, float addrspace(1)* %2790, align 4		; visa id: 3461
  %2792 = fmul reassoc nsz arcp contract float %2791, %4, !spirv.Decorations !881		; visa id: 3462
  %2793 = fadd reassoc nsz arcp contract float %2781, %2792, !spirv.Decorations !881		; visa id: 3463
  %2794 = add i64 %.in, %476		; visa id: 3464
  %2795 = inttoptr i64 %2794 to float addrspace(4)*		; visa id: 3465
  %2796 = addrspacecast float addrspace(4)* %2795 to float addrspace(1)*		; visa id: 3465
  store float %2793, float addrspace(1)* %2796, align 4		; visa id: 3466
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3467

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8.._crit_edge70.1.8_crit_edge, %2786, %2782
; BB371 :
  br i1 %230, label %2797, label %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3468

._crit_edge70.1.8.._crit_edge70.2.8_crit_edge:    ; preds = %._crit_edge70.1.8
; BB:
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2797:                                             ; preds = %._crit_edge70.1.8
; BB373 :
  %2798 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !881		; visa id: 3470
  br i1 %78, label %2803, label %2799, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3471

2799:                                             ; preds = %2797
; BB374 :
  %2800 = add i64 %.in, %478		; visa id: 3473
  %2801 = inttoptr i64 %2800 to float addrspace(4)*		; visa id: 3474
  %2802 = addrspacecast float addrspace(4)* %2801 to float addrspace(1)*		; visa id: 3474
  store float %2798, float addrspace(1)* %2802, align 4		; visa id: 3475
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3476

2803:                                             ; preds = %2797
; BB375 :
  %2804 = add i64 %.in3821, %sink_3858		; visa id: 3478
  %2805 = add i64 %2804, %sink_3841		; visa id: 3479
  %2806 = inttoptr i64 %2805 to float addrspace(4)*		; visa id: 3480
  %2807 = addrspacecast float addrspace(4)* %2806 to float addrspace(1)*		; visa id: 3480
  %2808 = load float, float addrspace(1)* %2807, align 4		; visa id: 3481
  %2809 = fmul reassoc nsz arcp contract float %2808, %4, !spirv.Decorations !881		; visa id: 3482
  %2810 = fadd reassoc nsz arcp contract float %2798, %2809, !spirv.Decorations !881		; visa id: 3483
  %2811 = add i64 %.in, %478		; visa id: 3484
  %2812 = inttoptr i64 %2811 to float addrspace(4)*		; visa id: 3485
  %2813 = addrspacecast float addrspace(4)* %2812 to float addrspace(1)*		; visa id: 3485
  store float %2810, float addrspace(1)* %2813, align 4		; visa id: 3486
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3487

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, %2803, %2799
; BB376 :
  br i1 %233, label %2814, label %._crit_edge70.2.8..preheader1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3488

._crit_edge70.2.8..preheader1.8_crit_edge:        ; preds = %._crit_edge70.2.8
; BB:
  br label %.preheader1.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2814:                                             ; preds = %._crit_edge70.2.8
; BB378 :
  %2815 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !881		; visa id: 3490
  br i1 %78, label %2820, label %2816, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3491

2816:                                             ; preds = %2814
; BB379 :
  %2817 = add i64 %.in, %480		; visa id: 3493
  %2818 = inttoptr i64 %2817 to float addrspace(4)*		; visa id: 3494
  %2819 = addrspacecast float addrspace(4)* %2818 to float addrspace(1)*		; visa id: 3494
  store float %2815, float addrspace(1)* %2819, align 4		; visa id: 3495
  br label %.preheader1.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3496

2820:                                             ; preds = %2814
; BB380 :
  %2821 = add i64 %.in3821, %sink_3857		; visa id: 3498
  %2822 = add i64 %2821, %sink_3841		; visa id: 3499
  %2823 = inttoptr i64 %2822 to float addrspace(4)*		; visa id: 3500
  %2824 = addrspacecast float addrspace(4)* %2823 to float addrspace(1)*		; visa id: 3500
  %2825 = load float, float addrspace(1)* %2824, align 4		; visa id: 3501
  %2826 = fmul reassoc nsz arcp contract float %2825, %4, !spirv.Decorations !881		; visa id: 3502
  %2827 = fadd reassoc nsz arcp contract float %2815, %2826, !spirv.Decorations !881		; visa id: 3503
  %2828 = add i64 %.in, %480		; visa id: 3504
  %2829 = inttoptr i64 %2828 to float addrspace(4)*		; visa id: 3505
  %2830 = addrspacecast float addrspace(4)* %2829 to float addrspace(1)*		; visa id: 3505
  store float %2827, float addrspace(1)* %2830, align 4		; visa id: 3506
  br label %.preheader1.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3507

.preheader1.8:                                    ; preds = %._crit_edge70.2.8..preheader1.8_crit_edge, %2820, %2816
; BB381 :
  %sink_3839 = shl nsw i64 %358, 2		; visa id: 3508
  br i1 %237, label %2831, label %.preheader1.8.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3509

.preheader1.8.._crit_edge70.9_crit_edge:          ; preds = %.preheader1.8
; BB:
  br label %._crit_edge70.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2831:                                             ; preds = %.preheader1.8
; BB383 :
  %2832 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !881		; visa id: 3511
  br i1 %78, label %2837, label %2833, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3512

2833:                                             ; preds = %2831
; BB384 :
  %2834 = add i64 %.in, %482		; visa id: 3514
  %2835 = inttoptr i64 %2834 to float addrspace(4)*		; visa id: 3515
  %2836 = addrspacecast float addrspace(4)* %2835 to float addrspace(1)*		; visa id: 3515
  store float %2832, float addrspace(1)* %2836, align 4		; visa id: 3516
  br label %._crit_edge70.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3517

2837:                                             ; preds = %2831
; BB385 :
  %2838 = add i64 %.in3821, %sink_3862		; visa id: 3519
  %2839 = add i64 %2838, %sink_3839		; visa id: 3520
  %2840 = inttoptr i64 %2839 to float addrspace(4)*		; visa id: 3521
  %2841 = addrspacecast float addrspace(4)* %2840 to float addrspace(1)*		; visa id: 3521
  %2842 = load float, float addrspace(1)* %2841, align 4		; visa id: 3522
  %2843 = fmul reassoc nsz arcp contract float %2842, %4, !spirv.Decorations !881		; visa id: 3523
  %2844 = fadd reassoc nsz arcp contract float %2832, %2843, !spirv.Decorations !881		; visa id: 3524
  %2845 = add i64 %.in, %482		; visa id: 3525
  %2846 = inttoptr i64 %2845 to float addrspace(4)*		; visa id: 3526
  %2847 = addrspacecast float addrspace(4)* %2846 to float addrspace(1)*		; visa id: 3526
  store float %2844, float addrspace(1)* %2847, align 4		; visa id: 3527
  br label %._crit_edge70.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3528

._crit_edge70.9:                                  ; preds = %.preheader1.8.._crit_edge70.9_crit_edge, %2837, %2833
; BB386 :
  br i1 %240, label %2848, label %._crit_edge70.9.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3529

._crit_edge70.9.._crit_edge70.1.9_crit_edge:      ; preds = %._crit_edge70.9
; BB:
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2848:                                             ; preds = %._crit_edge70.9
; BB388 :
  %2849 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !881		; visa id: 3531
  br i1 %78, label %2854, label %2850, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3532

2850:                                             ; preds = %2848
; BB389 :
  %2851 = add i64 %.in, %484		; visa id: 3534
  %2852 = inttoptr i64 %2851 to float addrspace(4)*		; visa id: 3535
  %2853 = addrspacecast float addrspace(4)* %2852 to float addrspace(1)*		; visa id: 3535
  store float %2849, float addrspace(1)* %2853, align 4		; visa id: 3536
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3537

2854:                                             ; preds = %2848
; BB390 :
  %2855 = add i64 %.in3821, %sink_3859		; visa id: 3539
  %2856 = add i64 %2855, %sink_3839		; visa id: 3540
  %2857 = inttoptr i64 %2856 to float addrspace(4)*		; visa id: 3541
  %2858 = addrspacecast float addrspace(4)* %2857 to float addrspace(1)*		; visa id: 3541
  %2859 = load float, float addrspace(1)* %2858, align 4		; visa id: 3542
  %2860 = fmul reassoc nsz arcp contract float %2859, %4, !spirv.Decorations !881		; visa id: 3543
  %2861 = fadd reassoc nsz arcp contract float %2849, %2860, !spirv.Decorations !881		; visa id: 3544
  %2862 = add i64 %.in, %484		; visa id: 3545
  %2863 = inttoptr i64 %2862 to float addrspace(4)*		; visa id: 3546
  %2864 = addrspacecast float addrspace(4)* %2863 to float addrspace(1)*		; visa id: 3546
  store float %2861, float addrspace(1)* %2864, align 4		; visa id: 3547
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3548

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9.._crit_edge70.1.9_crit_edge, %2854, %2850
; BB391 :
  br i1 %243, label %2865, label %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3549

._crit_edge70.1.9.._crit_edge70.2.9_crit_edge:    ; preds = %._crit_edge70.1.9
; BB:
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2865:                                             ; preds = %._crit_edge70.1.9
; BB393 :
  %2866 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !881		; visa id: 3551
  br i1 %78, label %2871, label %2867, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3552

2867:                                             ; preds = %2865
; BB394 :
  %2868 = add i64 %.in, %486		; visa id: 3554
  %2869 = inttoptr i64 %2868 to float addrspace(4)*		; visa id: 3555
  %2870 = addrspacecast float addrspace(4)* %2869 to float addrspace(1)*		; visa id: 3555
  store float %2866, float addrspace(1)* %2870, align 4		; visa id: 3556
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3557

2871:                                             ; preds = %2865
; BB395 :
  %2872 = add i64 %.in3821, %sink_3858		; visa id: 3559
  %2873 = add i64 %2872, %sink_3839		; visa id: 3560
  %2874 = inttoptr i64 %2873 to float addrspace(4)*		; visa id: 3561
  %2875 = addrspacecast float addrspace(4)* %2874 to float addrspace(1)*		; visa id: 3561
  %2876 = load float, float addrspace(1)* %2875, align 4		; visa id: 3562
  %2877 = fmul reassoc nsz arcp contract float %2876, %4, !spirv.Decorations !881		; visa id: 3563
  %2878 = fadd reassoc nsz arcp contract float %2866, %2877, !spirv.Decorations !881		; visa id: 3564
  %2879 = add i64 %.in, %486		; visa id: 3565
  %2880 = inttoptr i64 %2879 to float addrspace(4)*		; visa id: 3566
  %2881 = addrspacecast float addrspace(4)* %2880 to float addrspace(1)*		; visa id: 3566
  store float %2878, float addrspace(1)* %2881, align 4		; visa id: 3567
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3568

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, %2871, %2867
; BB396 :
  br i1 %246, label %2882, label %._crit_edge70.2.9..preheader1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3569

._crit_edge70.2.9..preheader1.9_crit_edge:        ; preds = %._crit_edge70.2.9
; BB:
  br label %.preheader1.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2882:                                             ; preds = %._crit_edge70.2.9
; BB398 :
  %2883 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !881		; visa id: 3571
  br i1 %78, label %2888, label %2884, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3572

2884:                                             ; preds = %2882
; BB399 :
  %2885 = add i64 %.in, %488		; visa id: 3574
  %2886 = inttoptr i64 %2885 to float addrspace(4)*		; visa id: 3575
  %2887 = addrspacecast float addrspace(4)* %2886 to float addrspace(1)*		; visa id: 3575
  store float %2883, float addrspace(1)* %2887, align 4		; visa id: 3576
  br label %.preheader1.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3577

2888:                                             ; preds = %2882
; BB400 :
  %2889 = add i64 %.in3821, %sink_3857		; visa id: 3579
  %2890 = add i64 %2889, %sink_3839		; visa id: 3580
  %2891 = inttoptr i64 %2890 to float addrspace(4)*		; visa id: 3581
  %2892 = addrspacecast float addrspace(4)* %2891 to float addrspace(1)*		; visa id: 3581
  %2893 = load float, float addrspace(1)* %2892, align 4		; visa id: 3582
  %2894 = fmul reassoc nsz arcp contract float %2893, %4, !spirv.Decorations !881		; visa id: 3583
  %2895 = fadd reassoc nsz arcp contract float %2883, %2894, !spirv.Decorations !881		; visa id: 3584
  %2896 = add i64 %.in, %488		; visa id: 3585
  %2897 = inttoptr i64 %2896 to float addrspace(4)*		; visa id: 3586
  %2898 = addrspacecast float addrspace(4)* %2897 to float addrspace(1)*		; visa id: 3586
  store float %2895, float addrspace(1)* %2898, align 4		; visa id: 3587
  br label %.preheader1.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3588

.preheader1.9:                                    ; preds = %._crit_edge70.2.9..preheader1.9_crit_edge, %2888, %2884
; BB401 :
  %sink_3837 = shl nsw i64 %359, 2		; visa id: 3589
  br i1 %250, label %2899, label %.preheader1.9.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3590

.preheader1.9.._crit_edge70.10_crit_edge:         ; preds = %.preheader1.9
; BB:
  br label %._crit_edge70.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2899:                                             ; preds = %.preheader1.9
; BB403 :
  %2900 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !881		; visa id: 3592
  br i1 %78, label %2905, label %2901, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3593

2901:                                             ; preds = %2899
; BB404 :
  %2902 = add i64 %.in, %490		; visa id: 3595
  %2903 = inttoptr i64 %2902 to float addrspace(4)*		; visa id: 3596
  %2904 = addrspacecast float addrspace(4)* %2903 to float addrspace(1)*		; visa id: 3596
  store float %2900, float addrspace(1)* %2904, align 4		; visa id: 3597
  br label %._crit_edge70.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3598

2905:                                             ; preds = %2899
; BB405 :
  %2906 = add i64 %.in3821, %sink_3862		; visa id: 3600
  %2907 = add i64 %2906, %sink_3837		; visa id: 3601
  %2908 = inttoptr i64 %2907 to float addrspace(4)*		; visa id: 3602
  %2909 = addrspacecast float addrspace(4)* %2908 to float addrspace(1)*		; visa id: 3602
  %2910 = load float, float addrspace(1)* %2909, align 4		; visa id: 3603
  %2911 = fmul reassoc nsz arcp contract float %2910, %4, !spirv.Decorations !881		; visa id: 3604
  %2912 = fadd reassoc nsz arcp contract float %2900, %2911, !spirv.Decorations !881		; visa id: 3605
  %2913 = add i64 %.in, %490		; visa id: 3606
  %2914 = inttoptr i64 %2913 to float addrspace(4)*		; visa id: 3607
  %2915 = addrspacecast float addrspace(4)* %2914 to float addrspace(1)*		; visa id: 3607
  store float %2912, float addrspace(1)* %2915, align 4		; visa id: 3608
  br label %._crit_edge70.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3609

._crit_edge70.10:                                 ; preds = %.preheader1.9.._crit_edge70.10_crit_edge, %2905, %2901
; BB406 :
  br i1 %253, label %2916, label %._crit_edge70.10.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3610

._crit_edge70.10.._crit_edge70.1.10_crit_edge:    ; preds = %._crit_edge70.10
; BB:
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2916:                                             ; preds = %._crit_edge70.10
; BB408 :
  %2917 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !881		; visa id: 3612
  br i1 %78, label %2922, label %2918, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3613

2918:                                             ; preds = %2916
; BB409 :
  %2919 = add i64 %.in, %492		; visa id: 3615
  %2920 = inttoptr i64 %2919 to float addrspace(4)*		; visa id: 3616
  %2921 = addrspacecast float addrspace(4)* %2920 to float addrspace(1)*		; visa id: 3616
  store float %2917, float addrspace(1)* %2921, align 4		; visa id: 3617
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3618

2922:                                             ; preds = %2916
; BB410 :
  %2923 = add i64 %.in3821, %sink_3859		; visa id: 3620
  %2924 = add i64 %2923, %sink_3837		; visa id: 3621
  %2925 = inttoptr i64 %2924 to float addrspace(4)*		; visa id: 3622
  %2926 = addrspacecast float addrspace(4)* %2925 to float addrspace(1)*		; visa id: 3622
  %2927 = load float, float addrspace(1)* %2926, align 4		; visa id: 3623
  %2928 = fmul reassoc nsz arcp contract float %2927, %4, !spirv.Decorations !881		; visa id: 3624
  %2929 = fadd reassoc nsz arcp contract float %2917, %2928, !spirv.Decorations !881		; visa id: 3625
  %2930 = add i64 %.in, %492		; visa id: 3626
  %2931 = inttoptr i64 %2930 to float addrspace(4)*		; visa id: 3627
  %2932 = addrspacecast float addrspace(4)* %2931 to float addrspace(1)*		; visa id: 3627
  store float %2929, float addrspace(1)* %2932, align 4		; visa id: 3628
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3629

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10.._crit_edge70.1.10_crit_edge, %2922, %2918
; BB411 :
  br i1 %256, label %2933, label %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3630

._crit_edge70.1.10.._crit_edge70.2.10_crit_edge:  ; preds = %._crit_edge70.1.10
; BB:
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2933:                                             ; preds = %._crit_edge70.1.10
; BB413 :
  %2934 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !881		; visa id: 3632
  br i1 %78, label %2939, label %2935, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3633

2935:                                             ; preds = %2933
; BB414 :
  %2936 = add i64 %.in, %494		; visa id: 3635
  %2937 = inttoptr i64 %2936 to float addrspace(4)*		; visa id: 3636
  %2938 = addrspacecast float addrspace(4)* %2937 to float addrspace(1)*		; visa id: 3636
  store float %2934, float addrspace(1)* %2938, align 4		; visa id: 3637
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3638

2939:                                             ; preds = %2933
; BB415 :
  %2940 = add i64 %.in3821, %sink_3858		; visa id: 3640
  %2941 = add i64 %2940, %sink_3837		; visa id: 3641
  %2942 = inttoptr i64 %2941 to float addrspace(4)*		; visa id: 3642
  %2943 = addrspacecast float addrspace(4)* %2942 to float addrspace(1)*		; visa id: 3642
  %2944 = load float, float addrspace(1)* %2943, align 4		; visa id: 3643
  %2945 = fmul reassoc nsz arcp contract float %2944, %4, !spirv.Decorations !881		; visa id: 3644
  %2946 = fadd reassoc nsz arcp contract float %2934, %2945, !spirv.Decorations !881		; visa id: 3645
  %2947 = add i64 %.in, %494		; visa id: 3646
  %2948 = inttoptr i64 %2947 to float addrspace(4)*		; visa id: 3647
  %2949 = addrspacecast float addrspace(4)* %2948 to float addrspace(1)*		; visa id: 3647
  store float %2946, float addrspace(1)* %2949, align 4		; visa id: 3648
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3649

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, %2939, %2935
; BB416 :
  br i1 %259, label %2950, label %._crit_edge70.2.10..preheader1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3650

._crit_edge70.2.10..preheader1.10_crit_edge:      ; preds = %._crit_edge70.2.10
; BB:
  br label %.preheader1.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2950:                                             ; preds = %._crit_edge70.2.10
; BB418 :
  %2951 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !881		; visa id: 3652
  br i1 %78, label %2956, label %2952, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3653

2952:                                             ; preds = %2950
; BB419 :
  %2953 = add i64 %.in, %496		; visa id: 3655
  %2954 = inttoptr i64 %2953 to float addrspace(4)*		; visa id: 3656
  %2955 = addrspacecast float addrspace(4)* %2954 to float addrspace(1)*		; visa id: 3656
  store float %2951, float addrspace(1)* %2955, align 4		; visa id: 3657
  br label %.preheader1.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3658

2956:                                             ; preds = %2950
; BB420 :
  %2957 = add i64 %.in3821, %sink_3857		; visa id: 3660
  %2958 = add i64 %2957, %sink_3837		; visa id: 3661
  %2959 = inttoptr i64 %2958 to float addrspace(4)*		; visa id: 3662
  %2960 = addrspacecast float addrspace(4)* %2959 to float addrspace(1)*		; visa id: 3662
  %2961 = load float, float addrspace(1)* %2960, align 4		; visa id: 3663
  %2962 = fmul reassoc nsz arcp contract float %2961, %4, !spirv.Decorations !881		; visa id: 3664
  %2963 = fadd reassoc nsz arcp contract float %2951, %2962, !spirv.Decorations !881		; visa id: 3665
  %2964 = add i64 %.in, %496		; visa id: 3666
  %2965 = inttoptr i64 %2964 to float addrspace(4)*		; visa id: 3667
  %2966 = addrspacecast float addrspace(4)* %2965 to float addrspace(1)*		; visa id: 3667
  store float %2963, float addrspace(1)* %2966, align 4		; visa id: 3668
  br label %.preheader1.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3669

.preheader1.10:                                   ; preds = %._crit_edge70.2.10..preheader1.10_crit_edge, %2956, %2952
; BB421 :
  %sink_3835 = shl nsw i64 %360, 2		; visa id: 3670
  br i1 %263, label %2967, label %.preheader1.10.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3671

.preheader1.10.._crit_edge70.11_crit_edge:        ; preds = %.preheader1.10
; BB:
  br label %._crit_edge70.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2967:                                             ; preds = %.preheader1.10
; BB423 :
  %2968 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !881		; visa id: 3673
  br i1 %78, label %2973, label %2969, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3674

2969:                                             ; preds = %2967
; BB424 :
  %2970 = add i64 %.in, %498		; visa id: 3676
  %2971 = inttoptr i64 %2970 to float addrspace(4)*		; visa id: 3677
  %2972 = addrspacecast float addrspace(4)* %2971 to float addrspace(1)*		; visa id: 3677
  store float %2968, float addrspace(1)* %2972, align 4		; visa id: 3678
  br label %._crit_edge70.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3679

2973:                                             ; preds = %2967
; BB425 :
  %2974 = add i64 %.in3821, %sink_3862		; visa id: 3681
  %2975 = add i64 %2974, %sink_3835		; visa id: 3682
  %2976 = inttoptr i64 %2975 to float addrspace(4)*		; visa id: 3683
  %2977 = addrspacecast float addrspace(4)* %2976 to float addrspace(1)*		; visa id: 3683
  %2978 = load float, float addrspace(1)* %2977, align 4		; visa id: 3684
  %2979 = fmul reassoc nsz arcp contract float %2978, %4, !spirv.Decorations !881		; visa id: 3685
  %2980 = fadd reassoc nsz arcp contract float %2968, %2979, !spirv.Decorations !881		; visa id: 3686
  %2981 = add i64 %.in, %498		; visa id: 3687
  %2982 = inttoptr i64 %2981 to float addrspace(4)*		; visa id: 3688
  %2983 = addrspacecast float addrspace(4)* %2982 to float addrspace(1)*		; visa id: 3688
  store float %2980, float addrspace(1)* %2983, align 4		; visa id: 3689
  br label %._crit_edge70.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3690

._crit_edge70.11:                                 ; preds = %.preheader1.10.._crit_edge70.11_crit_edge, %2973, %2969
; BB426 :
  br i1 %266, label %2984, label %._crit_edge70.11.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3691

._crit_edge70.11.._crit_edge70.1.11_crit_edge:    ; preds = %._crit_edge70.11
; BB:
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2984:                                             ; preds = %._crit_edge70.11
; BB428 :
  %2985 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !881		; visa id: 3693
  br i1 %78, label %2990, label %2986, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3694

2986:                                             ; preds = %2984
; BB429 :
  %2987 = add i64 %.in, %500		; visa id: 3696
  %2988 = inttoptr i64 %2987 to float addrspace(4)*		; visa id: 3697
  %2989 = addrspacecast float addrspace(4)* %2988 to float addrspace(1)*		; visa id: 3697
  store float %2985, float addrspace(1)* %2989, align 4		; visa id: 3698
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3699

2990:                                             ; preds = %2984
; BB430 :
  %2991 = add i64 %.in3821, %sink_3859		; visa id: 3701
  %2992 = add i64 %2991, %sink_3835		; visa id: 3702
  %2993 = inttoptr i64 %2992 to float addrspace(4)*		; visa id: 3703
  %2994 = addrspacecast float addrspace(4)* %2993 to float addrspace(1)*		; visa id: 3703
  %2995 = load float, float addrspace(1)* %2994, align 4		; visa id: 3704
  %2996 = fmul reassoc nsz arcp contract float %2995, %4, !spirv.Decorations !881		; visa id: 3705
  %2997 = fadd reassoc nsz arcp contract float %2985, %2996, !spirv.Decorations !881		; visa id: 3706
  %2998 = add i64 %.in, %500		; visa id: 3707
  %2999 = inttoptr i64 %2998 to float addrspace(4)*		; visa id: 3708
  %3000 = addrspacecast float addrspace(4)* %2999 to float addrspace(1)*		; visa id: 3708
  store float %2997, float addrspace(1)* %3000, align 4		; visa id: 3709
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3710

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11.._crit_edge70.1.11_crit_edge, %2990, %2986
; BB431 :
  br i1 %269, label %3001, label %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3711

._crit_edge70.1.11.._crit_edge70.2.11_crit_edge:  ; preds = %._crit_edge70.1.11
; BB:
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3001:                                             ; preds = %._crit_edge70.1.11
; BB433 :
  %3002 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !881		; visa id: 3713
  br i1 %78, label %3007, label %3003, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3714

3003:                                             ; preds = %3001
; BB434 :
  %3004 = add i64 %.in, %502		; visa id: 3716
  %3005 = inttoptr i64 %3004 to float addrspace(4)*		; visa id: 3717
  %3006 = addrspacecast float addrspace(4)* %3005 to float addrspace(1)*		; visa id: 3717
  store float %3002, float addrspace(1)* %3006, align 4		; visa id: 3718
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3719

3007:                                             ; preds = %3001
; BB435 :
  %3008 = add i64 %.in3821, %sink_3858		; visa id: 3721
  %3009 = add i64 %3008, %sink_3835		; visa id: 3722
  %3010 = inttoptr i64 %3009 to float addrspace(4)*		; visa id: 3723
  %3011 = addrspacecast float addrspace(4)* %3010 to float addrspace(1)*		; visa id: 3723
  %3012 = load float, float addrspace(1)* %3011, align 4		; visa id: 3724
  %3013 = fmul reassoc nsz arcp contract float %3012, %4, !spirv.Decorations !881		; visa id: 3725
  %3014 = fadd reassoc nsz arcp contract float %3002, %3013, !spirv.Decorations !881		; visa id: 3726
  %3015 = add i64 %.in, %502		; visa id: 3727
  %3016 = inttoptr i64 %3015 to float addrspace(4)*		; visa id: 3728
  %3017 = addrspacecast float addrspace(4)* %3016 to float addrspace(1)*		; visa id: 3728
  store float %3014, float addrspace(1)* %3017, align 4		; visa id: 3729
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3730

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, %3007, %3003
; BB436 :
  br i1 %272, label %3018, label %._crit_edge70.2.11..preheader1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3731

._crit_edge70.2.11..preheader1.11_crit_edge:      ; preds = %._crit_edge70.2.11
; BB:
  br label %.preheader1.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3018:                                             ; preds = %._crit_edge70.2.11
; BB438 :
  %3019 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !881		; visa id: 3733
  br i1 %78, label %3024, label %3020, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3734

3020:                                             ; preds = %3018
; BB439 :
  %3021 = add i64 %.in, %504		; visa id: 3736
  %3022 = inttoptr i64 %3021 to float addrspace(4)*		; visa id: 3737
  %3023 = addrspacecast float addrspace(4)* %3022 to float addrspace(1)*		; visa id: 3737
  store float %3019, float addrspace(1)* %3023, align 4		; visa id: 3738
  br label %.preheader1.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3739

3024:                                             ; preds = %3018
; BB440 :
  %3025 = add i64 %.in3821, %sink_3857		; visa id: 3741
  %3026 = add i64 %3025, %sink_3835		; visa id: 3742
  %3027 = inttoptr i64 %3026 to float addrspace(4)*		; visa id: 3743
  %3028 = addrspacecast float addrspace(4)* %3027 to float addrspace(1)*		; visa id: 3743
  %3029 = load float, float addrspace(1)* %3028, align 4		; visa id: 3744
  %3030 = fmul reassoc nsz arcp contract float %3029, %4, !spirv.Decorations !881		; visa id: 3745
  %3031 = fadd reassoc nsz arcp contract float %3019, %3030, !spirv.Decorations !881		; visa id: 3746
  %3032 = add i64 %.in, %504		; visa id: 3747
  %3033 = inttoptr i64 %3032 to float addrspace(4)*		; visa id: 3748
  %3034 = addrspacecast float addrspace(4)* %3033 to float addrspace(1)*		; visa id: 3748
  store float %3031, float addrspace(1)* %3034, align 4		; visa id: 3749
  br label %.preheader1.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3750

.preheader1.11:                                   ; preds = %._crit_edge70.2.11..preheader1.11_crit_edge, %3024, %3020
; BB441 :
  %sink_3833 = shl nsw i64 %361, 2		; visa id: 3751
  br i1 %276, label %3035, label %.preheader1.11.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3752

.preheader1.11.._crit_edge70.12_crit_edge:        ; preds = %.preheader1.11
; BB:
  br label %._crit_edge70.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3035:                                             ; preds = %.preheader1.11
; BB443 :
  %3036 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !881		; visa id: 3754
  br i1 %78, label %3041, label %3037, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3755

3037:                                             ; preds = %3035
; BB444 :
  %3038 = add i64 %.in, %506		; visa id: 3757
  %3039 = inttoptr i64 %3038 to float addrspace(4)*		; visa id: 3758
  %3040 = addrspacecast float addrspace(4)* %3039 to float addrspace(1)*		; visa id: 3758
  store float %3036, float addrspace(1)* %3040, align 4		; visa id: 3759
  br label %._crit_edge70.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3760

3041:                                             ; preds = %3035
; BB445 :
  %3042 = add i64 %.in3821, %sink_3862		; visa id: 3762
  %3043 = add i64 %3042, %sink_3833		; visa id: 3763
  %3044 = inttoptr i64 %3043 to float addrspace(4)*		; visa id: 3764
  %3045 = addrspacecast float addrspace(4)* %3044 to float addrspace(1)*		; visa id: 3764
  %3046 = load float, float addrspace(1)* %3045, align 4		; visa id: 3765
  %3047 = fmul reassoc nsz arcp contract float %3046, %4, !spirv.Decorations !881		; visa id: 3766
  %3048 = fadd reassoc nsz arcp contract float %3036, %3047, !spirv.Decorations !881		; visa id: 3767
  %3049 = add i64 %.in, %506		; visa id: 3768
  %3050 = inttoptr i64 %3049 to float addrspace(4)*		; visa id: 3769
  %3051 = addrspacecast float addrspace(4)* %3050 to float addrspace(1)*		; visa id: 3769
  store float %3048, float addrspace(1)* %3051, align 4		; visa id: 3770
  br label %._crit_edge70.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3771

._crit_edge70.12:                                 ; preds = %.preheader1.11.._crit_edge70.12_crit_edge, %3041, %3037
; BB446 :
  br i1 %279, label %3052, label %._crit_edge70.12.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3772

._crit_edge70.12.._crit_edge70.1.12_crit_edge:    ; preds = %._crit_edge70.12
; BB:
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3052:                                             ; preds = %._crit_edge70.12
; BB448 :
  %3053 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !881		; visa id: 3774
  br i1 %78, label %3058, label %3054, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3775

3054:                                             ; preds = %3052
; BB449 :
  %3055 = add i64 %.in, %508		; visa id: 3777
  %3056 = inttoptr i64 %3055 to float addrspace(4)*		; visa id: 3778
  %3057 = addrspacecast float addrspace(4)* %3056 to float addrspace(1)*		; visa id: 3778
  store float %3053, float addrspace(1)* %3057, align 4		; visa id: 3779
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3780

3058:                                             ; preds = %3052
; BB450 :
  %3059 = add i64 %.in3821, %sink_3859		; visa id: 3782
  %3060 = add i64 %3059, %sink_3833		; visa id: 3783
  %3061 = inttoptr i64 %3060 to float addrspace(4)*		; visa id: 3784
  %3062 = addrspacecast float addrspace(4)* %3061 to float addrspace(1)*		; visa id: 3784
  %3063 = load float, float addrspace(1)* %3062, align 4		; visa id: 3785
  %3064 = fmul reassoc nsz arcp contract float %3063, %4, !spirv.Decorations !881		; visa id: 3786
  %3065 = fadd reassoc nsz arcp contract float %3053, %3064, !spirv.Decorations !881		; visa id: 3787
  %3066 = add i64 %.in, %508		; visa id: 3788
  %3067 = inttoptr i64 %3066 to float addrspace(4)*		; visa id: 3789
  %3068 = addrspacecast float addrspace(4)* %3067 to float addrspace(1)*		; visa id: 3789
  store float %3065, float addrspace(1)* %3068, align 4		; visa id: 3790
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3791

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12.._crit_edge70.1.12_crit_edge, %3058, %3054
; BB451 :
  br i1 %282, label %3069, label %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3792

._crit_edge70.1.12.._crit_edge70.2.12_crit_edge:  ; preds = %._crit_edge70.1.12
; BB:
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3069:                                             ; preds = %._crit_edge70.1.12
; BB453 :
  %3070 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !881		; visa id: 3794
  br i1 %78, label %3075, label %3071, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3795

3071:                                             ; preds = %3069
; BB454 :
  %3072 = add i64 %.in, %510		; visa id: 3797
  %3073 = inttoptr i64 %3072 to float addrspace(4)*		; visa id: 3798
  %3074 = addrspacecast float addrspace(4)* %3073 to float addrspace(1)*		; visa id: 3798
  store float %3070, float addrspace(1)* %3074, align 4		; visa id: 3799
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3800

3075:                                             ; preds = %3069
; BB455 :
  %3076 = add i64 %.in3821, %sink_3858		; visa id: 3802
  %3077 = add i64 %3076, %sink_3833		; visa id: 3803
  %3078 = inttoptr i64 %3077 to float addrspace(4)*		; visa id: 3804
  %3079 = addrspacecast float addrspace(4)* %3078 to float addrspace(1)*		; visa id: 3804
  %3080 = load float, float addrspace(1)* %3079, align 4		; visa id: 3805
  %3081 = fmul reassoc nsz arcp contract float %3080, %4, !spirv.Decorations !881		; visa id: 3806
  %3082 = fadd reassoc nsz arcp contract float %3070, %3081, !spirv.Decorations !881		; visa id: 3807
  %3083 = add i64 %.in, %510		; visa id: 3808
  %3084 = inttoptr i64 %3083 to float addrspace(4)*		; visa id: 3809
  %3085 = addrspacecast float addrspace(4)* %3084 to float addrspace(1)*		; visa id: 3809
  store float %3082, float addrspace(1)* %3085, align 4		; visa id: 3810
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3811

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, %3075, %3071
; BB456 :
  br i1 %285, label %3086, label %._crit_edge70.2.12..preheader1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3812

._crit_edge70.2.12..preheader1.12_crit_edge:      ; preds = %._crit_edge70.2.12
; BB:
  br label %.preheader1.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3086:                                             ; preds = %._crit_edge70.2.12
; BB458 :
  %3087 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !881		; visa id: 3814
  br i1 %78, label %3092, label %3088, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3815

3088:                                             ; preds = %3086
; BB459 :
  %3089 = add i64 %.in, %512		; visa id: 3817
  %3090 = inttoptr i64 %3089 to float addrspace(4)*		; visa id: 3818
  %3091 = addrspacecast float addrspace(4)* %3090 to float addrspace(1)*		; visa id: 3818
  store float %3087, float addrspace(1)* %3091, align 4		; visa id: 3819
  br label %.preheader1.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3820

3092:                                             ; preds = %3086
; BB460 :
  %3093 = add i64 %.in3821, %sink_3857		; visa id: 3822
  %3094 = add i64 %3093, %sink_3833		; visa id: 3823
  %3095 = inttoptr i64 %3094 to float addrspace(4)*		; visa id: 3824
  %3096 = addrspacecast float addrspace(4)* %3095 to float addrspace(1)*		; visa id: 3824
  %3097 = load float, float addrspace(1)* %3096, align 4		; visa id: 3825
  %3098 = fmul reassoc nsz arcp contract float %3097, %4, !spirv.Decorations !881		; visa id: 3826
  %3099 = fadd reassoc nsz arcp contract float %3087, %3098, !spirv.Decorations !881		; visa id: 3827
  %3100 = add i64 %.in, %512		; visa id: 3828
  %3101 = inttoptr i64 %3100 to float addrspace(4)*		; visa id: 3829
  %3102 = addrspacecast float addrspace(4)* %3101 to float addrspace(1)*		; visa id: 3829
  store float %3099, float addrspace(1)* %3102, align 4		; visa id: 3830
  br label %.preheader1.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3831

.preheader1.12:                                   ; preds = %._crit_edge70.2.12..preheader1.12_crit_edge, %3092, %3088
; BB461 :
  %sink_3831 = shl nsw i64 %362, 2		; visa id: 3832
  br i1 %289, label %3103, label %.preheader1.12.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3833

.preheader1.12.._crit_edge70.13_crit_edge:        ; preds = %.preheader1.12
; BB:
  br label %._crit_edge70.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3103:                                             ; preds = %.preheader1.12
; BB463 :
  %3104 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !881		; visa id: 3835
  br i1 %78, label %3109, label %3105, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3836

3105:                                             ; preds = %3103
; BB464 :
  %3106 = add i64 %.in, %514		; visa id: 3838
  %3107 = inttoptr i64 %3106 to float addrspace(4)*		; visa id: 3839
  %3108 = addrspacecast float addrspace(4)* %3107 to float addrspace(1)*		; visa id: 3839
  store float %3104, float addrspace(1)* %3108, align 4		; visa id: 3840
  br label %._crit_edge70.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3841

3109:                                             ; preds = %3103
; BB465 :
  %3110 = add i64 %.in3821, %sink_3862		; visa id: 3843
  %3111 = add i64 %3110, %sink_3831		; visa id: 3844
  %3112 = inttoptr i64 %3111 to float addrspace(4)*		; visa id: 3845
  %3113 = addrspacecast float addrspace(4)* %3112 to float addrspace(1)*		; visa id: 3845
  %3114 = load float, float addrspace(1)* %3113, align 4		; visa id: 3846
  %3115 = fmul reassoc nsz arcp contract float %3114, %4, !spirv.Decorations !881		; visa id: 3847
  %3116 = fadd reassoc nsz arcp contract float %3104, %3115, !spirv.Decorations !881		; visa id: 3848
  %3117 = add i64 %.in, %514		; visa id: 3849
  %3118 = inttoptr i64 %3117 to float addrspace(4)*		; visa id: 3850
  %3119 = addrspacecast float addrspace(4)* %3118 to float addrspace(1)*		; visa id: 3850
  store float %3116, float addrspace(1)* %3119, align 4		; visa id: 3851
  br label %._crit_edge70.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3852

._crit_edge70.13:                                 ; preds = %.preheader1.12.._crit_edge70.13_crit_edge, %3109, %3105
; BB466 :
  br i1 %292, label %3120, label %._crit_edge70.13.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3853

._crit_edge70.13.._crit_edge70.1.13_crit_edge:    ; preds = %._crit_edge70.13
; BB:
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3120:                                             ; preds = %._crit_edge70.13
; BB468 :
  %3121 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !881		; visa id: 3855
  br i1 %78, label %3126, label %3122, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3856

3122:                                             ; preds = %3120
; BB469 :
  %3123 = add i64 %.in, %516		; visa id: 3858
  %3124 = inttoptr i64 %3123 to float addrspace(4)*		; visa id: 3859
  %3125 = addrspacecast float addrspace(4)* %3124 to float addrspace(1)*		; visa id: 3859
  store float %3121, float addrspace(1)* %3125, align 4		; visa id: 3860
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3861

3126:                                             ; preds = %3120
; BB470 :
  %3127 = add i64 %.in3821, %sink_3859		; visa id: 3863
  %3128 = add i64 %3127, %sink_3831		; visa id: 3864
  %3129 = inttoptr i64 %3128 to float addrspace(4)*		; visa id: 3865
  %3130 = addrspacecast float addrspace(4)* %3129 to float addrspace(1)*		; visa id: 3865
  %3131 = load float, float addrspace(1)* %3130, align 4		; visa id: 3866
  %3132 = fmul reassoc nsz arcp contract float %3131, %4, !spirv.Decorations !881		; visa id: 3867
  %3133 = fadd reassoc nsz arcp contract float %3121, %3132, !spirv.Decorations !881		; visa id: 3868
  %3134 = add i64 %.in, %516		; visa id: 3869
  %3135 = inttoptr i64 %3134 to float addrspace(4)*		; visa id: 3870
  %3136 = addrspacecast float addrspace(4)* %3135 to float addrspace(1)*		; visa id: 3870
  store float %3133, float addrspace(1)* %3136, align 4		; visa id: 3871
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3872

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13.._crit_edge70.1.13_crit_edge, %3126, %3122
; BB471 :
  br i1 %295, label %3137, label %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3873

._crit_edge70.1.13.._crit_edge70.2.13_crit_edge:  ; preds = %._crit_edge70.1.13
; BB:
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3137:                                             ; preds = %._crit_edge70.1.13
; BB473 :
  %3138 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !881		; visa id: 3875
  br i1 %78, label %3143, label %3139, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3876

3139:                                             ; preds = %3137
; BB474 :
  %3140 = add i64 %.in, %518		; visa id: 3878
  %3141 = inttoptr i64 %3140 to float addrspace(4)*		; visa id: 3879
  %3142 = addrspacecast float addrspace(4)* %3141 to float addrspace(1)*		; visa id: 3879
  store float %3138, float addrspace(1)* %3142, align 4		; visa id: 3880
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3881

3143:                                             ; preds = %3137
; BB475 :
  %3144 = add i64 %.in3821, %sink_3858		; visa id: 3883
  %3145 = add i64 %3144, %sink_3831		; visa id: 3884
  %3146 = inttoptr i64 %3145 to float addrspace(4)*		; visa id: 3885
  %3147 = addrspacecast float addrspace(4)* %3146 to float addrspace(1)*		; visa id: 3885
  %3148 = load float, float addrspace(1)* %3147, align 4		; visa id: 3886
  %3149 = fmul reassoc nsz arcp contract float %3148, %4, !spirv.Decorations !881		; visa id: 3887
  %3150 = fadd reassoc nsz arcp contract float %3138, %3149, !spirv.Decorations !881		; visa id: 3888
  %3151 = add i64 %.in, %518		; visa id: 3889
  %3152 = inttoptr i64 %3151 to float addrspace(4)*		; visa id: 3890
  %3153 = addrspacecast float addrspace(4)* %3152 to float addrspace(1)*		; visa id: 3890
  store float %3150, float addrspace(1)* %3153, align 4		; visa id: 3891
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3892

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, %3143, %3139
; BB476 :
  br i1 %298, label %3154, label %._crit_edge70.2.13..preheader1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3893

._crit_edge70.2.13..preheader1.13_crit_edge:      ; preds = %._crit_edge70.2.13
; BB:
  br label %.preheader1.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3154:                                             ; preds = %._crit_edge70.2.13
; BB478 :
  %3155 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !881		; visa id: 3895
  br i1 %78, label %3160, label %3156, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3896

3156:                                             ; preds = %3154
; BB479 :
  %3157 = add i64 %.in, %520		; visa id: 3898
  %3158 = inttoptr i64 %3157 to float addrspace(4)*		; visa id: 3899
  %3159 = addrspacecast float addrspace(4)* %3158 to float addrspace(1)*		; visa id: 3899
  store float %3155, float addrspace(1)* %3159, align 4		; visa id: 3900
  br label %.preheader1.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3901

3160:                                             ; preds = %3154
; BB480 :
  %3161 = add i64 %.in3821, %sink_3857		; visa id: 3903
  %3162 = add i64 %3161, %sink_3831		; visa id: 3904
  %3163 = inttoptr i64 %3162 to float addrspace(4)*		; visa id: 3905
  %3164 = addrspacecast float addrspace(4)* %3163 to float addrspace(1)*		; visa id: 3905
  %3165 = load float, float addrspace(1)* %3164, align 4		; visa id: 3906
  %3166 = fmul reassoc nsz arcp contract float %3165, %4, !spirv.Decorations !881		; visa id: 3907
  %3167 = fadd reassoc nsz arcp contract float %3155, %3166, !spirv.Decorations !881		; visa id: 3908
  %3168 = add i64 %.in, %520		; visa id: 3909
  %3169 = inttoptr i64 %3168 to float addrspace(4)*		; visa id: 3910
  %3170 = addrspacecast float addrspace(4)* %3169 to float addrspace(1)*		; visa id: 3910
  store float %3167, float addrspace(1)* %3170, align 4		; visa id: 3911
  br label %.preheader1.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3912

.preheader1.13:                                   ; preds = %._crit_edge70.2.13..preheader1.13_crit_edge, %3160, %3156
; BB481 :
  %sink_3829 = shl nsw i64 %363, 2		; visa id: 3913
  br i1 %302, label %3171, label %.preheader1.13.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3914

.preheader1.13.._crit_edge70.14_crit_edge:        ; preds = %.preheader1.13
; BB:
  br label %._crit_edge70.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3171:                                             ; preds = %.preheader1.13
; BB483 :
  %3172 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !881		; visa id: 3916
  br i1 %78, label %3177, label %3173, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3917

3173:                                             ; preds = %3171
; BB484 :
  %3174 = add i64 %.in, %522		; visa id: 3919
  %3175 = inttoptr i64 %3174 to float addrspace(4)*		; visa id: 3920
  %3176 = addrspacecast float addrspace(4)* %3175 to float addrspace(1)*		; visa id: 3920
  store float %3172, float addrspace(1)* %3176, align 4		; visa id: 3921
  br label %._crit_edge70.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3922

3177:                                             ; preds = %3171
; BB485 :
  %3178 = add i64 %.in3821, %sink_3862		; visa id: 3924
  %3179 = add i64 %3178, %sink_3829		; visa id: 3925
  %3180 = inttoptr i64 %3179 to float addrspace(4)*		; visa id: 3926
  %3181 = addrspacecast float addrspace(4)* %3180 to float addrspace(1)*		; visa id: 3926
  %3182 = load float, float addrspace(1)* %3181, align 4		; visa id: 3927
  %3183 = fmul reassoc nsz arcp contract float %3182, %4, !spirv.Decorations !881		; visa id: 3928
  %3184 = fadd reassoc nsz arcp contract float %3172, %3183, !spirv.Decorations !881		; visa id: 3929
  %3185 = add i64 %.in, %522		; visa id: 3930
  %3186 = inttoptr i64 %3185 to float addrspace(4)*		; visa id: 3931
  %3187 = addrspacecast float addrspace(4)* %3186 to float addrspace(1)*		; visa id: 3931
  store float %3184, float addrspace(1)* %3187, align 4		; visa id: 3932
  br label %._crit_edge70.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3933

._crit_edge70.14:                                 ; preds = %.preheader1.13.._crit_edge70.14_crit_edge, %3177, %3173
; BB486 :
  br i1 %305, label %3188, label %._crit_edge70.14.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3934

._crit_edge70.14.._crit_edge70.1.14_crit_edge:    ; preds = %._crit_edge70.14
; BB:
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3188:                                             ; preds = %._crit_edge70.14
; BB488 :
  %3189 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !881		; visa id: 3936
  br i1 %78, label %3194, label %3190, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3937

3190:                                             ; preds = %3188
; BB489 :
  %3191 = add i64 %.in, %524		; visa id: 3939
  %3192 = inttoptr i64 %3191 to float addrspace(4)*		; visa id: 3940
  %3193 = addrspacecast float addrspace(4)* %3192 to float addrspace(1)*		; visa id: 3940
  store float %3189, float addrspace(1)* %3193, align 4		; visa id: 3941
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3942

3194:                                             ; preds = %3188
; BB490 :
  %3195 = add i64 %.in3821, %sink_3859		; visa id: 3944
  %3196 = add i64 %3195, %sink_3829		; visa id: 3945
  %3197 = inttoptr i64 %3196 to float addrspace(4)*		; visa id: 3946
  %3198 = addrspacecast float addrspace(4)* %3197 to float addrspace(1)*		; visa id: 3946
  %3199 = load float, float addrspace(1)* %3198, align 4		; visa id: 3947
  %3200 = fmul reassoc nsz arcp contract float %3199, %4, !spirv.Decorations !881		; visa id: 3948
  %3201 = fadd reassoc nsz arcp contract float %3189, %3200, !spirv.Decorations !881		; visa id: 3949
  %3202 = add i64 %.in, %524		; visa id: 3950
  %3203 = inttoptr i64 %3202 to float addrspace(4)*		; visa id: 3951
  %3204 = addrspacecast float addrspace(4)* %3203 to float addrspace(1)*		; visa id: 3951
  store float %3201, float addrspace(1)* %3204, align 4		; visa id: 3952
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3953

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14.._crit_edge70.1.14_crit_edge, %3194, %3190
; BB491 :
  br i1 %308, label %3205, label %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3954

._crit_edge70.1.14.._crit_edge70.2.14_crit_edge:  ; preds = %._crit_edge70.1.14
; BB:
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3205:                                             ; preds = %._crit_edge70.1.14
; BB493 :
  %3206 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !881		; visa id: 3956
  br i1 %78, label %3211, label %3207, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3957

3207:                                             ; preds = %3205
; BB494 :
  %3208 = add i64 %.in, %526		; visa id: 3959
  %3209 = inttoptr i64 %3208 to float addrspace(4)*		; visa id: 3960
  %3210 = addrspacecast float addrspace(4)* %3209 to float addrspace(1)*		; visa id: 3960
  store float %3206, float addrspace(1)* %3210, align 4		; visa id: 3961
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3962

3211:                                             ; preds = %3205
; BB495 :
  %3212 = add i64 %.in3821, %sink_3858		; visa id: 3964
  %3213 = add i64 %3212, %sink_3829		; visa id: 3965
  %3214 = inttoptr i64 %3213 to float addrspace(4)*		; visa id: 3966
  %3215 = addrspacecast float addrspace(4)* %3214 to float addrspace(1)*		; visa id: 3966
  %3216 = load float, float addrspace(1)* %3215, align 4		; visa id: 3967
  %3217 = fmul reassoc nsz arcp contract float %3216, %4, !spirv.Decorations !881		; visa id: 3968
  %3218 = fadd reassoc nsz arcp contract float %3206, %3217, !spirv.Decorations !881		; visa id: 3969
  %3219 = add i64 %.in, %526		; visa id: 3970
  %3220 = inttoptr i64 %3219 to float addrspace(4)*		; visa id: 3971
  %3221 = addrspacecast float addrspace(4)* %3220 to float addrspace(1)*		; visa id: 3971
  store float %3218, float addrspace(1)* %3221, align 4		; visa id: 3972
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3973

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, %3211, %3207
; BB496 :
  br i1 %311, label %3222, label %._crit_edge70.2.14..preheader1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3974

._crit_edge70.2.14..preheader1.14_crit_edge:      ; preds = %._crit_edge70.2.14
; BB:
  br label %.preheader1.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3222:                                             ; preds = %._crit_edge70.2.14
; BB498 :
  %3223 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !881		; visa id: 3976
  br i1 %78, label %3228, label %3224, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3977

3224:                                             ; preds = %3222
; BB499 :
  %3225 = add i64 %.in, %528		; visa id: 3979
  %3226 = inttoptr i64 %3225 to float addrspace(4)*		; visa id: 3980
  %3227 = addrspacecast float addrspace(4)* %3226 to float addrspace(1)*		; visa id: 3980
  store float %3223, float addrspace(1)* %3227, align 4		; visa id: 3981
  br label %.preheader1.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3982

3228:                                             ; preds = %3222
; BB500 :
  %3229 = add i64 %.in3821, %sink_3857		; visa id: 3984
  %3230 = add i64 %3229, %sink_3829		; visa id: 3985
  %3231 = inttoptr i64 %3230 to float addrspace(4)*		; visa id: 3986
  %3232 = addrspacecast float addrspace(4)* %3231 to float addrspace(1)*		; visa id: 3986
  %3233 = load float, float addrspace(1)* %3232, align 4		; visa id: 3987
  %3234 = fmul reassoc nsz arcp contract float %3233, %4, !spirv.Decorations !881		; visa id: 3988
  %3235 = fadd reassoc nsz arcp contract float %3223, %3234, !spirv.Decorations !881		; visa id: 3989
  %3236 = add i64 %.in, %528		; visa id: 3990
  %3237 = inttoptr i64 %3236 to float addrspace(4)*		; visa id: 3991
  %3238 = addrspacecast float addrspace(4)* %3237 to float addrspace(1)*		; visa id: 3991
  store float %3235, float addrspace(1)* %3238, align 4		; visa id: 3992
  br label %.preheader1.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3993

.preheader1.14:                                   ; preds = %._crit_edge70.2.14..preheader1.14_crit_edge, %3228, %3224
; BB501 :
  %sink_3827 = shl nsw i64 %364, 2		; visa id: 3994
  br i1 %315, label %3239, label %.preheader1.14.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3995

.preheader1.14.._crit_edge70.15_crit_edge:        ; preds = %.preheader1.14
; BB:
  br label %._crit_edge70.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3239:                                             ; preds = %.preheader1.14
; BB503 :
  %3240 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !881		; visa id: 3997
  br i1 %78, label %3245, label %3241, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3998

3241:                                             ; preds = %3239
; BB504 :
  %3242 = add i64 %.in, %530		; visa id: 4000
  %3243 = inttoptr i64 %3242 to float addrspace(4)*		; visa id: 4001
  %3244 = addrspacecast float addrspace(4)* %3243 to float addrspace(1)*		; visa id: 4001
  store float %3240, float addrspace(1)* %3244, align 4		; visa id: 4002
  br label %._crit_edge70.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 4003

3245:                                             ; preds = %3239
; BB505 :
  %3246 = add i64 %.in3821, %sink_3862		; visa id: 4005
  %3247 = add i64 %3246, %sink_3827		; visa id: 4006
  %3248 = inttoptr i64 %3247 to float addrspace(4)*		; visa id: 4007
  %3249 = addrspacecast float addrspace(4)* %3248 to float addrspace(1)*		; visa id: 4007
  %3250 = load float, float addrspace(1)* %3249, align 4		; visa id: 4008
  %3251 = fmul reassoc nsz arcp contract float %3250, %4, !spirv.Decorations !881		; visa id: 4009
  %3252 = fadd reassoc nsz arcp contract float %3240, %3251, !spirv.Decorations !881		; visa id: 4010
  %3253 = add i64 %.in, %530		; visa id: 4011
  %3254 = inttoptr i64 %3253 to float addrspace(4)*		; visa id: 4012
  %3255 = addrspacecast float addrspace(4)* %3254 to float addrspace(1)*		; visa id: 4012
  store float %3252, float addrspace(1)* %3255, align 4		; visa id: 4013
  br label %._crit_edge70.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 4014

._crit_edge70.15:                                 ; preds = %.preheader1.14.._crit_edge70.15_crit_edge, %3245, %3241
; BB506 :
  br i1 %318, label %3256, label %._crit_edge70.15.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 4015

._crit_edge70.15.._crit_edge70.1.15_crit_edge:    ; preds = %._crit_edge70.15
; BB:
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3256:                                             ; preds = %._crit_edge70.15
; BB508 :
  %3257 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !881		; visa id: 4017
  br i1 %78, label %3262, label %3258, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 4018

3258:                                             ; preds = %3256
; BB509 :
  %3259 = add i64 %.in, %532		; visa id: 4020
  %3260 = inttoptr i64 %3259 to float addrspace(4)*		; visa id: 4021
  %3261 = addrspacecast float addrspace(4)* %3260 to float addrspace(1)*		; visa id: 4021
  store float %3257, float addrspace(1)* %3261, align 4		; visa id: 4022
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 4023

3262:                                             ; preds = %3256
; BB510 :
  %3263 = add i64 %.in3821, %sink_3859		; visa id: 4025
  %3264 = add i64 %3263, %sink_3827		; visa id: 4026
  %3265 = inttoptr i64 %3264 to float addrspace(4)*		; visa id: 4027
  %3266 = addrspacecast float addrspace(4)* %3265 to float addrspace(1)*		; visa id: 4027
  %3267 = load float, float addrspace(1)* %3266, align 4		; visa id: 4028
  %3268 = fmul reassoc nsz arcp contract float %3267, %4, !spirv.Decorations !881		; visa id: 4029
  %3269 = fadd reassoc nsz arcp contract float %3257, %3268, !spirv.Decorations !881		; visa id: 4030
  %3270 = add i64 %.in, %532		; visa id: 4031
  %3271 = inttoptr i64 %3270 to float addrspace(4)*		; visa id: 4032
  %3272 = addrspacecast float addrspace(4)* %3271 to float addrspace(1)*		; visa id: 4032
  store float %3269, float addrspace(1)* %3272, align 4		; visa id: 4033
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 4034

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15.._crit_edge70.1.15_crit_edge, %3262, %3258
; BB511 :
  br i1 %321, label %3273, label %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 4035

._crit_edge70.1.15.._crit_edge70.2.15_crit_edge:  ; preds = %._crit_edge70.1.15
; BB:
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3273:                                             ; preds = %._crit_edge70.1.15
; BB513 :
  %3274 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !881		; visa id: 4037
  br i1 %78, label %3279, label %3275, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 4038

3275:                                             ; preds = %3273
; BB514 :
  %3276 = add i64 %.in, %534		; visa id: 4040
  %3277 = inttoptr i64 %3276 to float addrspace(4)*		; visa id: 4041
  %3278 = addrspacecast float addrspace(4)* %3277 to float addrspace(1)*		; visa id: 4041
  store float %3274, float addrspace(1)* %3278, align 4		; visa id: 4042
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 4043

3279:                                             ; preds = %3273
; BB515 :
  %3280 = add i64 %.in3821, %sink_3858		; visa id: 4045
  %3281 = add i64 %3280, %sink_3827		; visa id: 4046
  %3282 = inttoptr i64 %3281 to float addrspace(4)*		; visa id: 4047
  %3283 = addrspacecast float addrspace(4)* %3282 to float addrspace(1)*		; visa id: 4047
  %3284 = load float, float addrspace(1)* %3283, align 4		; visa id: 4048
  %3285 = fmul reassoc nsz arcp contract float %3284, %4, !spirv.Decorations !881		; visa id: 4049
  %3286 = fadd reassoc nsz arcp contract float %3274, %3285, !spirv.Decorations !881		; visa id: 4050
  %3287 = add i64 %.in, %534		; visa id: 4051
  %3288 = inttoptr i64 %3287 to float addrspace(4)*		; visa id: 4052
  %3289 = addrspacecast float addrspace(4)* %3288 to float addrspace(1)*		; visa id: 4052
  store float %3286, float addrspace(1)* %3289, align 4		; visa id: 4053
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 4054

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, %3279, %3275
; BB516 :
  br i1 %324, label %3290, label %._crit_edge70.2.15..preheader1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 4055

._crit_edge70.2.15..preheader1.15_crit_edge:      ; preds = %._crit_edge70.2.15
; BB:
  br label %.preheader1.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

3290:                                             ; preds = %._crit_edge70.2.15
; BB518 :
  %3291 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !881		; visa id: 4057
  br i1 %78, label %3296, label %3292, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 4058

3292:                                             ; preds = %3290
; BB519 :
  %3293 = add i64 %.in, %536		; visa id: 4060
  %3294 = inttoptr i64 %3293 to float addrspace(4)*		; visa id: 4061
  %3295 = addrspacecast float addrspace(4)* %3294 to float addrspace(1)*		; visa id: 4061
  store float %3291, float addrspace(1)* %3295, align 4		; visa id: 4062
  br label %.preheader1.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 4063

3296:                                             ; preds = %3290
; BB520 :
  %3297 = add i64 %.in3821, %sink_3857		; visa id: 4065
  %3298 = add i64 %3297, %sink_3827		; visa id: 4066
  %3299 = inttoptr i64 %3298 to float addrspace(4)*		; visa id: 4067
  %3300 = addrspacecast float addrspace(4)* %3299 to float addrspace(1)*		; visa id: 4067
  %3301 = load float, float addrspace(1)* %3300, align 4		; visa id: 4068
  %3302 = fmul reassoc nsz arcp contract float %3301, %4, !spirv.Decorations !881		; visa id: 4069
  %3303 = fadd reassoc nsz arcp contract float %3291, %3302, !spirv.Decorations !881		; visa id: 4070
  %3304 = add i64 %.in, %536		; visa id: 4071
  %3305 = inttoptr i64 %3304 to float addrspace(4)*		; visa id: 4072
  %3306 = addrspacecast float addrspace(4)* %3305 to float addrspace(1)*		; visa id: 4072
  store float %3303, float addrspace(1)* %3306, align 4		; visa id: 4073
  br label %.preheader1.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 4074

.preheader1.15:                                   ; preds = %._crit_edge70.2.15..preheader1.15_crit_edge, %3296, %3292
; BB521 :
  %3307 = add i32 %547, %49		; visa id: 4075
  %3308 = icmp slt i32 %3307, %8		; visa id: 4076
  br i1 %3308, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 4077

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
; BB523 :
  %3309 = add i64 %.in3823, %537		; visa id: 4079
  %3310 = add i64 %.in3822, %538		; visa id: 4080
  %sink_ = bitcast <2 x i32> %545 to i64		; visa id: 4081
  %3311 = add i64 %.in3821, %sink_		; visa id: 4083
  %3312 = add i64 %.in, %546		; visa id: 4084
  br label %.preheader2.preheader, !stats.blockFrequency.digits !896, !stats.blockFrequency.scale !879		; visa id: 4085

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
; BB524 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 4087
}

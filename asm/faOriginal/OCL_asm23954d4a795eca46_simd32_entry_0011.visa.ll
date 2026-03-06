; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0011.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
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
  %64 = shl i32 %63, 4		; visa id: 51
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
  %172 = add i32 %64, 4		; visa id: 183
  %173 = icmp slt i32 %172, %const_reg_dword1		; visa id: 184
  %174 = icmp slt i32 %60, %const_reg_dword
  %175 = and i1 %174, %173		; visa id: 185
  %176 = icmp slt i32 %121, %const_reg_dword
  %177 = icmp slt i32 %172, %const_reg_dword1		; visa id: 187
  %178 = and i1 %176, %177		; visa id: 188
  %179 = icmp slt i32 %125, %const_reg_dword
  %180 = icmp slt i32 %172, %const_reg_dword1		; visa id: 190
  %181 = and i1 %179, %180		; visa id: 191
  %182 = icmp slt i32 %129, %const_reg_dword
  %183 = icmp slt i32 %172, %const_reg_dword1		; visa id: 193
  %184 = and i1 %182, %183		; visa id: 194
  %185 = add i32 %64, 5		; visa id: 196
  %186 = icmp slt i32 %185, %const_reg_dword1		; visa id: 197
  %187 = icmp slt i32 %60, %const_reg_dword
  %188 = and i1 %187, %186		; visa id: 198
  %189 = icmp slt i32 %121, %const_reg_dword
  %190 = icmp slt i32 %185, %const_reg_dword1		; visa id: 200
  %191 = and i1 %189, %190		; visa id: 201
  %192 = icmp slt i32 %125, %const_reg_dword
  %193 = icmp slt i32 %185, %const_reg_dword1		; visa id: 203
  %194 = and i1 %192, %193		; visa id: 204
  %195 = icmp slt i32 %129, %const_reg_dword
  %196 = icmp slt i32 %185, %const_reg_dword1		; visa id: 206
  %197 = and i1 %195, %196		; visa id: 207
  %198 = add i32 %64, 6		; visa id: 209
  %199 = icmp slt i32 %198, %const_reg_dword1		; visa id: 210
  %200 = icmp slt i32 %60, %const_reg_dword
  %201 = and i1 %200, %199		; visa id: 211
  %202 = icmp slt i32 %121, %const_reg_dword
  %203 = icmp slt i32 %198, %const_reg_dword1		; visa id: 213
  %204 = and i1 %202, %203		; visa id: 214
  %205 = icmp slt i32 %125, %const_reg_dword
  %206 = icmp slt i32 %198, %const_reg_dword1		; visa id: 216
  %207 = and i1 %205, %206		; visa id: 217
  %208 = icmp slt i32 %129, %const_reg_dword
  %209 = icmp slt i32 %198, %const_reg_dword1		; visa id: 219
  %210 = and i1 %208, %209		; visa id: 220
  %211 = add i32 %64, 7		; visa id: 222
  %212 = icmp slt i32 %211, %const_reg_dword1		; visa id: 223
  %213 = icmp slt i32 %60, %const_reg_dword
  %214 = and i1 %213, %212		; visa id: 224
  %215 = icmp slt i32 %121, %const_reg_dword
  %216 = icmp slt i32 %211, %const_reg_dword1		; visa id: 226
  %217 = and i1 %215, %216		; visa id: 227
  %218 = icmp slt i32 %125, %const_reg_dword
  %219 = icmp slt i32 %211, %const_reg_dword1		; visa id: 229
  %220 = and i1 %218, %219		; visa id: 230
  %221 = icmp slt i32 %129, %const_reg_dword
  %222 = icmp slt i32 %211, %const_reg_dword1		; visa id: 232
  %223 = and i1 %221, %222		; visa id: 233
  %224 = add i32 %64, 8		; visa id: 235
  %225 = icmp slt i32 %224, %const_reg_dword1		; visa id: 236
  %226 = icmp slt i32 %60, %const_reg_dword
  %227 = and i1 %226, %225		; visa id: 237
  %228 = icmp slt i32 %121, %const_reg_dword
  %229 = icmp slt i32 %224, %const_reg_dword1		; visa id: 239
  %230 = and i1 %228, %229		; visa id: 240
  %231 = icmp slt i32 %125, %const_reg_dword
  %232 = icmp slt i32 %224, %const_reg_dword1		; visa id: 242
  %233 = and i1 %231, %232		; visa id: 243
  %234 = icmp slt i32 %129, %const_reg_dword
  %235 = icmp slt i32 %224, %const_reg_dword1		; visa id: 245
  %236 = and i1 %234, %235		; visa id: 246
  %237 = add i32 %64, 9		; visa id: 248
  %238 = icmp slt i32 %237, %const_reg_dword1		; visa id: 249
  %239 = icmp slt i32 %60, %const_reg_dword
  %240 = and i1 %239, %238		; visa id: 250
  %241 = icmp slt i32 %121, %const_reg_dword
  %242 = icmp slt i32 %237, %const_reg_dword1		; visa id: 252
  %243 = and i1 %241, %242		; visa id: 253
  %244 = icmp slt i32 %125, %const_reg_dword
  %245 = icmp slt i32 %237, %const_reg_dword1		; visa id: 255
  %246 = and i1 %244, %245		; visa id: 256
  %247 = icmp slt i32 %129, %const_reg_dword
  %248 = icmp slt i32 %237, %const_reg_dword1		; visa id: 258
  %249 = and i1 %247, %248		; visa id: 259
  %250 = add i32 %64, 10		; visa id: 261
  %251 = icmp slt i32 %250, %const_reg_dword1		; visa id: 262
  %252 = icmp slt i32 %60, %const_reg_dword
  %253 = and i1 %252, %251		; visa id: 263
  %254 = icmp slt i32 %121, %const_reg_dword
  %255 = icmp slt i32 %250, %const_reg_dword1		; visa id: 265
  %256 = and i1 %254, %255		; visa id: 266
  %257 = icmp slt i32 %125, %const_reg_dword
  %258 = icmp slt i32 %250, %const_reg_dword1		; visa id: 268
  %259 = and i1 %257, %258		; visa id: 269
  %260 = icmp slt i32 %129, %const_reg_dword
  %261 = icmp slt i32 %250, %const_reg_dword1		; visa id: 271
  %262 = and i1 %260, %261		; visa id: 272
  %263 = add i32 %64, 11		; visa id: 274
  %264 = icmp slt i32 %263, %const_reg_dword1		; visa id: 275
  %265 = icmp slt i32 %60, %const_reg_dword
  %266 = and i1 %265, %264		; visa id: 276
  %267 = icmp slt i32 %121, %const_reg_dword
  %268 = icmp slt i32 %263, %const_reg_dword1		; visa id: 278
  %269 = and i1 %267, %268		; visa id: 279
  %270 = icmp slt i32 %125, %const_reg_dword
  %271 = icmp slt i32 %263, %const_reg_dword1		; visa id: 281
  %272 = and i1 %270, %271		; visa id: 282
  %273 = icmp slt i32 %129, %const_reg_dword
  %274 = icmp slt i32 %263, %const_reg_dword1		; visa id: 284
  %275 = and i1 %273, %274		; visa id: 285
  %276 = add i32 %64, 12		; visa id: 287
  %277 = icmp slt i32 %276, %const_reg_dword1		; visa id: 288
  %278 = icmp slt i32 %60, %const_reg_dword
  %279 = and i1 %278, %277		; visa id: 289
  %280 = icmp slt i32 %121, %const_reg_dword
  %281 = icmp slt i32 %276, %const_reg_dword1		; visa id: 291
  %282 = and i1 %280, %281		; visa id: 292
  %283 = icmp slt i32 %125, %const_reg_dword
  %284 = icmp slt i32 %276, %const_reg_dword1		; visa id: 294
  %285 = and i1 %283, %284		; visa id: 295
  %286 = icmp slt i32 %129, %const_reg_dword
  %287 = icmp slt i32 %276, %const_reg_dword1		; visa id: 297
  %288 = and i1 %286, %287		; visa id: 298
  %289 = add i32 %64, 13		; visa id: 300
  %290 = icmp slt i32 %289, %const_reg_dword1		; visa id: 301
  %291 = icmp slt i32 %60, %const_reg_dword
  %292 = and i1 %291, %290		; visa id: 302
  %293 = icmp slt i32 %121, %const_reg_dword
  %294 = icmp slt i32 %289, %const_reg_dword1		; visa id: 304
  %295 = and i1 %293, %294		; visa id: 305
  %296 = icmp slt i32 %125, %const_reg_dword
  %297 = icmp slt i32 %289, %const_reg_dword1		; visa id: 307
  %298 = and i1 %296, %297		; visa id: 308
  %299 = icmp slt i32 %129, %const_reg_dword
  %300 = icmp slt i32 %289, %const_reg_dword1		; visa id: 310
  %301 = and i1 %299, %300		; visa id: 311
  %302 = add i32 %64, 14		; visa id: 313
  %303 = icmp slt i32 %302, %const_reg_dword1		; visa id: 314
  %304 = icmp slt i32 %60, %const_reg_dword
  %305 = and i1 %304, %303		; visa id: 315
  %306 = icmp slt i32 %121, %const_reg_dword
  %307 = icmp slt i32 %302, %const_reg_dword1		; visa id: 317
  %308 = and i1 %306, %307		; visa id: 318
  %309 = icmp slt i32 %125, %const_reg_dword
  %310 = icmp slt i32 %302, %const_reg_dword1		; visa id: 320
  %311 = and i1 %309, %310		; visa id: 321
  %312 = icmp slt i32 %129, %const_reg_dword
  %313 = icmp slt i32 %302, %const_reg_dword1		; visa id: 323
  %314 = and i1 %312, %313		; visa id: 324
  %315 = add i32 %64, 15		; visa id: 326
  %316 = icmp slt i32 %315, %const_reg_dword1		; visa id: 327
  %317 = icmp slt i32 %60, %const_reg_dword
  %318 = and i1 %317, %316		; visa id: 328
  %319 = icmp slt i32 %121, %const_reg_dword
  %320 = icmp slt i32 %315, %const_reg_dword1		; visa id: 330
  %321 = and i1 %319, %320		; visa id: 331
  %322 = icmp slt i32 %125, %const_reg_dword
  %323 = icmp slt i32 %315, %const_reg_dword1		; visa id: 333
  %324 = and i1 %322, %323		; visa id: 334
  %325 = icmp slt i32 %129, %const_reg_dword
  %326 = icmp slt i32 %315, %const_reg_dword1		; visa id: 336
  %327 = and i1 %325, %326		; visa id: 337
  %328 = ashr i32 %60, 31		; visa id: 339
  %329 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %41, i32 %42)
  %330 = extractvalue { i32, i32 } %329, 0		; visa id: 340
  %331 = extractvalue { i32, i32 } %329, 1		; visa id: 340
  %332 = insertelement <2 x i32> undef, i32 %330, i32 0		; visa id: 347
  %333 = insertelement <2 x i32> %332, i32 %331, i32 1		; visa id: 348
  %334 = ashr i32 %64, 31		; visa id: 349
  %335 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %64, i32 %334, i32 %44, i32 %45)
  %336 = extractvalue { i32, i32 } %335, 0		; visa id: 350
  %337 = extractvalue { i32, i32 } %335, 1		; visa id: 350
  %338 = insertelement <2 x i32> undef, i32 %336, i32 0		; visa id: 357
  %339 = insertelement <2 x i32> %338, i32 %337, i32 1		; visa id: 358
  %340 = ashr i32 %121, 31		; visa id: 359
  %341 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %41, i32 %42)
  %342 = extractvalue { i32, i32 } %341, 0		; visa id: 360
  %343 = extractvalue { i32, i32 } %341, 1		; visa id: 360
  %344 = insertelement <2 x i32> undef, i32 %342, i32 0		; visa id: 367
  %345 = insertelement <2 x i32> %344, i32 %343, i32 1		; visa id: 368
  %346 = ashr i32 %125, 31		; visa id: 369
  %347 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %41, i32 %42)
  %348 = extractvalue { i32, i32 } %347, 0		; visa id: 370
  %349 = extractvalue { i32, i32 } %347, 1		; visa id: 370
  %350 = insertelement <2 x i32> undef, i32 %348, i32 0		; visa id: 377
  %351 = insertelement <2 x i32> %350, i32 %349, i32 1		; visa id: 378
  %352 = ashr i32 %129, 31		; visa id: 379
  %353 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %41, i32 %42)
  %354 = extractvalue { i32, i32 } %353, 0		; visa id: 380
  %355 = extractvalue { i32, i32 } %353, 1		; visa id: 380
  %356 = insertelement <2 x i32> undef, i32 %354, i32 0		; visa id: 387
  %357 = insertelement <2 x i32> %356, i32 %355, i32 1		; visa id: 388
  %358 = ashr i32 %133, 31		; visa id: 389
  %359 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %133, i32 %358, i32 %44, i32 %45)
  %360 = extractvalue { i32, i32 } %359, 0		; visa id: 390
  %361 = extractvalue { i32, i32 } %359, 1		; visa id: 390
  %362 = insertelement <2 x i32> undef, i32 %360, i32 0		; visa id: 397
  %363 = insertelement <2 x i32> %362, i32 %361, i32 1		; visa id: 398
  %364 = ashr i32 %146, 31		; visa id: 399
  %365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %146, i32 %364, i32 %44, i32 %45)
  %366 = extractvalue { i32, i32 } %365, 0		; visa id: 400
  %367 = extractvalue { i32, i32 } %365, 1		; visa id: 400
  %368 = insertelement <2 x i32> undef, i32 %366, i32 0		; visa id: 407
  %369 = insertelement <2 x i32> %368, i32 %367, i32 1		; visa id: 408
  %370 = ashr i32 %159, 31		; visa id: 409
  %371 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %159, i32 %370, i32 %44, i32 %45)
  %372 = extractvalue { i32, i32 } %371, 0		; visa id: 410
  %373 = extractvalue { i32, i32 } %371, 1		; visa id: 410
  %374 = insertelement <2 x i32> undef, i32 %372, i32 0		; visa id: 417
  %375 = insertelement <2 x i32> %374, i32 %373, i32 1		; visa id: 418
  %376 = ashr i32 %172, 31		; visa id: 419
  %377 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %172, i32 %376, i32 %44, i32 %45)
  %378 = extractvalue { i32, i32 } %377, 0		; visa id: 420
  %379 = extractvalue { i32, i32 } %377, 1		; visa id: 420
  %380 = insertelement <2 x i32> undef, i32 %378, i32 0		; visa id: 427
  %381 = insertelement <2 x i32> %380, i32 %379, i32 1		; visa id: 428
  %382 = ashr i32 %185, 31		; visa id: 429
  %383 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %185, i32 %382, i32 %44, i32 %45)
  %384 = extractvalue { i32, i32 } %383, 0		; visa id: 430
  %385 = extractvalue { i32, i32 } %383, 1		; visa id: 430
  %386 = insertelement <2 x i32> undef, i32 %384, i32 0		; visa id: 437
  %387 = insertelement <2 x i32> %386, i32 %385, i32 1		; visa id: 438
  %388 = ashr i32 %198, 31		; visa id: 439
  %389 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %198, i32 %388, i32 %44, i32 %45)
  %390 = extractvalue { i32, i32 } %389, 0		; visa id: 440
  %391 = extractvalue { i32, i32 } %389, 1		; visa id: 440
  %392 = insertelement <2 x i32> undef, i32 %390, i32 0		; visa id: 447
  %393 = insertelement <2 x i32> %392, i32 %391, i32 1		; visa id: 448
  %394 = ashr i32 %211, 31		; visa id: 449
  %395 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %211, i32 %394, i32 %44, i32 %45)
  %396 = extractvalue { i32, i32 } %395, 0		; visa id: 450
  %397 = extractvalue { i32, i32 } %395, 1		; visa id: 450
  %398 = insertelement <2 x i32> undef, i32 %396, i32 0		; visa id: 457
  %399 = insertelement <2 x i32> %398, i32 %397, i32 1		; visa id: 458
  %400 = ashr i32 %224, 31		; visa id: 459
  %401 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %224, i32 %400, i32 %44, i32 %45)
  %402 = extractvalue { i32, i32 } %401, 0		; visa id: 460
  %403 = extractvalue { i32, i32 } %401, 1		; visa id: 460
  %404 = insertelement <2 x i32> undef, i32 %402, i32 0		; visa id: 467
  %405 = insertelement <2 x i32> %404, i32 %403, i32 1		; visa id: 468
  %406 = ashr i32 %237, 31		; visa id: 469
  %407 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %237, i32 %406, i32 %44, i32 %45)
  %408 = extractvalue { i32, i32 } %407, 0		; visa id: 470
  %409 = extractvalue { i32, i32 } %407, 1		; visa id: 470
  %410 = insertelement <2 x i32> undef, i32 %408, i32 0		; visa id: 477
  %411 = insertelement <2 x i32> %410, i32 %409, i32 1		; visa id: 478
  %412 = ashr i32 %250, 31		; visa id: 479
  %413 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %250, i32 %412, i32 %44, i32 %45)
  %414 = extractvalue { i32, i32 } %413, 0		; visa id: 480
  %415 = extractvalue { i32, i32 } %413, 1		; visa id: 480
  %416 = insertelement <2 x i32> undef, i32 %414, i32 0		; visa id: 487
  %417 = insertelement <2 x i32> %416, i32 %415, i32 1		; visa id: 488
  %418 = ashr i32 %263, 31		; visa id: 489
  %419 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %263, i32 %418, i32 %44, i32 %45)
  %420 = extractvalue { i32, i32 } %419, 0		; visa id: 490
  %421 = extractvalue { i32, i32 } %419, 1		; visa id: 490
  %422 = insertelement <2 x i32> undef, i32 %420, i32 0		; visa id: 497
  %423 = insertelement <2 x i32> %422, i32 %421, i32 1		; visa id: 498
  %424 = ashr i32 %276, 31		; visa id: 499
  %425 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %276, i32 %424, i32 %44, i32 %45)
  %426 = extractvalue { i32, i32 } %425, 0		; visa id: 500
  %427 = extractvalue { i32, i32 } %425, 1		; visa id: 500
  %428 = insertelement <2 x i32> undef, i32 %426, i32 0		; visa id: 507
  %429 = insertelement <2 x i32> %428, i32 %427, i32 1		; visa id: 508
  %430 = ashr i32 %289, 31		; visa id: 509
  %431 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %289, i32 %430, i32 %44, i32 %45)
  %432 = extractvalue { i32, i32 } %431, 0		; visa id: 510
  %433 = extractvalue { i32, i32 } %431, 1		; visa id: 510
  %434 = insertelement <2 x i32> undef, i32 %432, i32 0		; visa id: 517
  %435 = insertelement <2 x i32> %434, i32 %433, i32 1		; visa id: 518
  %436 = ashr i32 %302, 31		; visa id: 519
  %437 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %302, i32 %436, i32 %44, i32 %45)
  %438 = extractvalue { i32, i32 } %437, 0		; visa id: 520
  %439 = extractvalue { i32, i32 } %437, 1		; visa id: 520
  %440 = insertelement <2 x i32> undef, i32 %438, i32 0		; visa id: 527
  %441 = insertelement <2 x i32> %440, i32 %439, i32 1		; visa id: 528
  %442 = ashr i32 %315, 31		; visa id: 529
  %443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %315, i32 %442, i32 %44, i32 %45)
  %444 = extractvalue { i32, i32 } %443, 0		; visa id: 530
  %445 = extractvalue { i32, i32 } %443, 1		; visa id: 530
  %446 = insertelement <2 x i32> undef, i32 %444, i32 0		; visa id: 537
  %447 = insertelement <2 x i32> %446, i32 %445, i32 1		; visa id: 538
  %448 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %50, i32 %51)
  %449 = extractvalue { i32, i32 } %448, 0		; visa id: 539
  %450 = extractvalue { i32, i32 } %448, 1		; visa id: 539
  %451 = insertelement <2 x i32> undef, i32 %449, i32 0		; visa id: 546
  %452 = insertelement <2 x i32> %451, i32 %450, i32 1		; visa id: 547
  %453 = bitcast <2 x i32> %452 to i64		; visa id: 548
  %454 = sext i32 %64 to i64		; visa id: 552
  %455 = add nsw i64 %453, %454		; visa id: 553
  %456 = shl i64 %455, 2		; visa id: 554
  %457 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %47, i32 %48)
  %458 = extractvalue { i32, i32 } %457, 0		; visa id: 555
  %459 = extractvalue { i32, i32 } %457, 1		; visa id: 555
  %460 = insertelement <2 x i32> undef, i32 %458, i32 0		; visa id: 562
  %461 = insertelement <2 x i32> %460, i32 %459, i32 1		; visa id: 563
  %462 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %50, i32 %51)
  %463 = extractvalue { i32, i32 } %462, 0		; visa id: 564
  %464 = extractvalue { i32, i32 } %462, 1		; visa id: 564
  %465 = insertelement <2 x i32> undef, i32 %463, i32 0		; visa id: 571
  %466 = insertelement <2 x i32> %465, i32 %464, i32 1		; visa id: 572
  %467 = bitcast <2 x i32> %466 to i64		; visa id: 573
  %468 = add nsw i64 %467, %454		; visa id: 577
  %469 = shl i64 %468, 2		; visa id: 578
  %470 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %47, i32 %48)
  %471 = extractvalue { i32, i32 } %470, 0		; visa id: 579
  %472 = extractvalue { i32, i32 } %470, 1		; visa id: 579
  %473 = insertelement <2 x i32> undef, i32 %471, i32 0		; visa id: 586
  %474 = insertelement <2 x i32> %473, i32 %472, i32 1		; visa id: 587
  %475 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %50, i32 %51)
  %476 = extractvalue { i32, i32 } %475, 0		; visa id: 588
  %477 = extractvalue { i32, i32 } %475, 1		; visa id: 588
  %478 = insertelement <2 x i32> undef, i32 %476, i32 0		; visa id: 595
  %479 = insertelement <2 x i32> %478, i32 %477, i32 1		; visa id: 596
  %480 = bitcast <2 x i32> %479 to i64		; visa id: 597
  %481 = add nsw i64 %480, %454		; visa id: 601
  %482 = shl i64 %481, 2		; visa id: 602
  %483 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %47, i32 %48)
  %484 = extractvalue { i32, i32 } %483, 0		; visa id: 603
  %485 = extractvalue { i32, i32 } %483, 1		; visa id: 603
  %486 = insertelement <2 x i32> undef, i32 %484, i32 0		; visa id: 610
  %487 = insertelement <2 x i32> %486, i32 %485, i32 1		; visa id: 611
  %488 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %50, i32 %51)
  %489 = extractvalue { i32, i32 } %488, 0		; visa id: 612
  %490 = extractvalue { i32, i32 } %488, 1		; visa id: 612
  %491 = insertelement <2 x i32> undef, i32 %489, i32 0		; visa id: 619
  %492 = insertelement <2 x i32> %491, i32 %490, i32 1		; visa id: 620
  %493 = bitcast <2 x i32> %492 to i64		; visa id: 621
  %494 = add nsw i64 %493, %454		; visa id: 625
  %495 = shl i64 %494, 2		; visa id: 626
  %496 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %47, i32 %48)
  %497 = extractvalue { i32, i32 } %496, 0		; visa id: 627
  %498 = extractvalue { i32, i32 } %496, 1		; visa id: 627
  %499 = insertelement <2 x i32> undef, i32 %497, i32 0		; visa id: 634
  %500 = insertelement <2 x i32> %499, i32 %498, i32 1		; visa id: 635
  %501 = sext i32 %133 to i64		; visa id: 636
  %502 = add nsw i64 %453, %501		; visa id: 637
  %503 = shl i64 %502, 2		; visa id: 638
  %504 = add nsw i64 %467, %501		; visa id: 639
  %505 = shl i64 %504, 2		; visa id: 640
  %506 = add nsw i64 %480, %501		; visa id: 641
  %507 = shl i64 %506, 2		; visa id: 642
  %508 = add nsw i64 %493, %501		; visa id: 643
  %509 = shl i64 %508, 2		; visa id: 644
  %510 = sext i32 %146 to i64		; visa id: 645
  %511 = add nsw i64 %453, %510		; visa id: 646
  %512 = shl i64 %511, 2		; visa id: 647
  %513 = add nsw i64 %467, %510		; visa id: 648
  %514 = shl i64 %513, 2		; visa id: 649
  %515 = add nsw i64 %480, %510		; visa id: 650
  %516 = shl i64 %515, 2		; visa id: 651
  %517 = add nsw i64 %493, %510		; visa id: 652
  %518 = shl i64 %517, 2		; visa id: 653
  %519 = sext i32 %159 to i64		; visa id: 654
  %520 = add nsw i64 %453, %519		; visa id: 655
  %521 = shl i64 %520, 2		; visa id: 656
  %522 = add nsw i64 %467, %519		; visa id: 657
  %523 = shl i64 %522, 2		; visa id: 658
  %524 = add nsw i64 %480, %519		; visa id: 659
  %525 = shl i64 %524, 2		; visa id: 660
  %526 = add nsw i64 %493, %519		; visa id: 661
  %527 = shl i64 %526, 2		; visa id: 662
  %528 = sext i32 %172 to i64		; visa id: 663
  %529 = add nsw i64 %453, %528		; visa id: 664
  %530 = shl i64 %529, 2		; visa id: 665
  %531 = add nsw i64 %467, %528		; visa id: 666
  %532 = shl i64 %531, 2		; visa id: 667
  %533 = add nsw i64 %480, %528		; visa id: 668
  %534 = shl i64 %533, 2		; visa id: 669
  %535 = add nsw i64 %493, %528		; visa id: 670
  %536 = shl i64 %535, 2		; visa id: 671
  %537 = sext i32 %185 to i64		; visa id: 672
  %538 = add nsw i64 %453, %537		; visa id: 673
  %539 = shl i64 %538, 2		; visa id: 674
  %540 = add nsw i64 %467, %537		; visa id: 675
  %541 = shl i64 %540, 2		; visa id: 676
  %542 = add nsw i64 %480, %537		; visa id: 677
  %543 = shl i64 %542, 2		; visa id: 678
  %544 = add nsw i64 %493, %537		; visa id: 679
  %545 = shl i64 %544, 2		; visa id: 680
  %546 = sext i32 %198 to i64		; visa id: 681
  %547 = add nsw i64 %453, %546		; visa id: 682
  %548 = shl i64 %547, 2		; visa id: 683
  %549 = add nsw i64 %467, %546		; visa id: 684
  %550 = shl i64 %549, 2		; visa id: 685
  %551 = add nsw i64 %480, %546		; visa id: 686
  %552 = shl i64 %551, 2		; visa id: 687
  %553 = add nsw i64 %493, %546		; visa id: 688
  %554 = shl i64 %553, 2		; visa id: 689
  %555 = sext i32 %211 to i64		; visa id: 690
  %556 = add nsw i64 %453, %555		; visa id: 691
  %557 = shl i64 %556, 2		; visa id: 692
  %558 = add nsw i64 %467, %555		; visa id: 693
  %559 = shl i64 %558, 2		; visa id: 694
  %560 = add nsw i64 %480, %555		; visa id: 695
  %561 = shl i64 %560, 2		; visa id: 696
  %562 = add nsw i64 %493, %555		; visa id: 697
  %563 = shl i64 %562, 2		; visa id: 698
  %564 = sext i32 %224 to i64		; visa id: 699
  %565 = add nsw i64 %453, %564		; visa id: 700
  %566 = shl i64 %565, 2		; visa id: 701
  %567 = add nsw i64 %467, %564		; visa id: 702
  %568 = shl i64 %567, 2		; visa id: 703
  %569 = add nsw i64 %480, %564		; visa id: 704
  %570 = shl i64 %569, 2		; visa id: 705
  %571 = add nsw i64 %493, %564		; visa id: 706
  %572 = shl i64 %571, 2		; visa id: 707
  %573 = sext i32 %237 to i64		; visa id: 708
  %574 = add nsw i64 %453, %573		; visa id: 709
  %575 = shl i64 %574, 2		; visa id: 710
  %576 = add nsw i64 %467, %573		; visa id: 711
  %577 = shl i64 %576, 2		; visa id: 712
  %578 = add nsw i64 %480, %573		; visa id: 713
  %579 = shl i64 %578, 2		; visa id: 714
  %580 = add nsw i64 %493, %573		; visa id: 715
  %581 = shl i64 %580, 2		; visa id: 716
  %582 = sext i32 %250 to i64		; visa id: 717
  %583 = add nsw i64 %453, %582		; visa id: 718
  %584 = shl i64 %583, 2		; visa id: 719
  %585 = add nsw i64 %467, %582		; visa id: 720
  %586 = shl i64 %585, 2		; visa id: 721
  %587 = add nsw i64 %480, %582		; visa id: 722
  %588 = shl i64 %587, 2		; visa id: 723
  %589 = add nsw i64 %493, %582		; visa id: 724
  %590 = shl i64 %589, 2		; visa id: 725
  %591 = sext i32 %263 to i64		; visa id: 726
  %592 = add nsw i64 %453, %591		; visa id: 727
  %593 = shl i64 %592, 2		; visa id: 728
  %594 = add nsw i64 %467, %591		; visa id: 729
  %595 = shl i64 %594, 2		; visa id: 730
  %596 = add nsw i64 %480, %591		; visa id: 731
  %597 = shl i64 %596, 2		; visa id: 732
  %598 = add nsw i64 %493, %591		; visa id: 733
  %599 = shl i64 %598, 2		; visa id: 734
  %600 = sext i32 %276 to i64		; visa id: 735
  %601 = add nsw i64 %453, %600		; visa id: 736
  %602 = shl i64 %601, 2		; visa id: 737
  %603 = add nsw i64 %467, %600		; visa id: 738
  %604 = shl i64 %603, 2		; visa id: 739
  %605 = add nsw i64 %480, %600		; visa id: 740
  %606 = shl i64 %605, 2		; visa id: 741
  %607 = add nsw i64 %493, %600		; visa id: 742
  %608 = shl i64 %607, 2		; visa id: 743
  %609 = sext i32 %289 to i64		; visa id: 744
  %610 = add nsw i64 %453, %609		; visa id: 745
  %611 = shl i64 %610, 2		; visa id: 746
  %612 = add nsw i64 %467, %609		; visa id: 747
  %613 = shl i64 %612, 2		; visa id: 748
  %614 = add nsw i64 %480, %609		; visa id: 749
  %615 = shl i64 %614, 2		; visa id: 750
  %616 = add nsw i64 %493, %609		; visa id: 751
  %617 = shl i64 %616, 2		; visa id: 752
  %618 = sext i32 %302 to i64		; visa id: 753
  %619 = add nsw i64 %453, %618		; visa id: 754
  %620 = shl i64 %619, 2		; visa id: 755
  %621 = add nsw i64 %467, %618		; visa id: 756
  %622 = shl i64 %621, 2		; visa id: 757
  %623 = add nsw i64 %480, %618		; visa id: 758
  %624 = shl i64 %623, 2		; visa id: 759
  %625 = add nsw i64 %493, %618		; visa id: 760
  %626 = shl i64 %625, 2		; visa id: 761
  %627 = sext i32 %315 to i64		; visa id: 762
  %628 = add nsw i64 %453, %627		; visa id: 763
  %629 = shl i64 %628, 2		; visa id: 764
  %630 = add nsw i64 %467, %627		; visa id: 765
  %631 = shl i64 %630, 2		; visa id: 766
  %632 = add nsw i64 %480, %627		; visa id: 767
  %633 = shl i64 %632, 2		; visa id: 768
  %634 = add nsw i64 %493, %627		; visa id: 769
  %635 = shl i64 %634, 2		; visa id: 770
  %636 = shl i64 %99, 1		; visa id: 771
  %637 = shl i64 %105, 1		; visa id: 772
  %.op3824 = shl i64 %111, 2		; visa id: 773
  %638 = bitcast i64 %.op3824 to <2 x i32>		; visa id: 774
  %639 = extractelement <2 x i32> %638, i32 0		; visa id: 775
  %640 = extractelement <2 x i32> %638, i32 1		; visa id: 775
  %641 = select i1 %81, i32 %639, i32 0		; visa id: 775
  %642 = select i1 %81, i32 %640, i32 0		; visa id: 776
  %643 = insertelement <2 x i32> undef, i32 %641, i32 0		; visa id: 777
  %644 = insertelement <2 x i32> %643, i32 %642, i32 1		; visa id: 778
  %645 = shl i64 %117, 2		; visa id: 779
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879		; visa id: 780

.preheader2.preheader:                            ; preds = %.preheader1.15..preheader2.preheader_crit_edge, %.lr.ph
; BB3 :
  %646 = phi i32 [ %26, %.lr.ph ], [ %2890, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %92, %.lr.ph ], [ %2895, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3821 = phi i64 [ %87, %.lr.ph ], [ %2894, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3822 = phi i64 [ %74, %.lr.ph ], [ %2893, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3823 = phi i64 [ %69, %.lr.ph ], [ %2892, %.preheader1.15..preheader2.preheader_crit_edge ]
  br i1 %93, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 781

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
; BB4 :
  br label %.preheader1.preheader, !stats.blockFrequency.digits !884, !stats.blockFrequency.scale !879		; visa id: 847

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
; BB5 :
  br label %.preheader.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879		; visa id: 914

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
  %647 = phi i32 [ %1800, %.preheader.15..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
  %sink_sink_3888 = bitcast <2 x i32> %333 to i64		; visa id: 915
  %sink_sink_3864 = shl i64 %sink_sink_3888, 1		; visa id: 919
  %sink_3908 = add i64 %.in3823, %sink_sink_3864		; visa id: 920
  %sink_sink_3887 = bitcast <2 x i32> %339 to i64		; visa id: 921
  %sink_sink_3863 = shl i64 %sink_sink_3887, 1		; visa id: 925
  %sink_3907 = add i64 %.in3822, %sink_sink_3863		; visa id: 926
  br i1 %120, label %648, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 927

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

648:                                              ; preds = %.preheader.preheader
; BB8 :
  %.sroa.256.0.insert.ext = zext i32 %647 to i64		; visa id: 929
  %649 = shl nuw nsw i64 %.sroa.256.0.insert.ext, 1		; visa id: 930
  %650 = add i64 %sink_3908, %649		; visa id: 931
  %651 = inttoptr i64 %650 to i16 addrspace(4)*		; visa id: 932
  %652 = addrspacecast i16 addrspace(4)* %651 to i16 addrspace(1)*		; visa id: 932
  %653 = load i16, i16 addrspace(1)* %652, align 2		; visa id: 933
  %654 = add i64 %sink_3907, %649		; visa id: 935
  %655 = inttoptr i64 %654 to i16 addrspace(4)*		; visa id: 936
  %656 = addrspacecast i16 addrspace(4)* %655 to i16 addrspace(1)*		; visa id: 936
  %657 = load i16, i16 addrspace(1)* %656, align 2		; visa id: 937
  %658 = zext i16 %653 to i32		; visa id: 939
  %659 = shl nuw i32 %658, 16, !spirv.Decorations !888		; visa id: 940
  %660 = bitcast i32 %659 to float
  %661 = zext i16 %657 to i32		; visa id: 941
  %662 = shl nuw i32 %661, 16, !spirv.Decorations !888		; visa id: 942
  %663 = bitcast i32 %662 to float
  %664 = fmul reassoc nsz arcp contract float %660, %663, !spirv.Decorations !881
  %665 = fadd reassoc nsz arcp contract float %664, %.sroa.0.1, !spirv.Decorations !881		; visa id: 943
  br label %._crit_edge, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 944

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %648
; BB9 :
  %.sroa.0.2 = phi float [ %665, %648 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  %sink_sink_3886 = bitcast <2 x i32> %345 to i64		; visa id: 945
  %sink_sink_3862 = shl i64 %sink_sink_3886, 1		; visa id: 949
  %sink_3906 = add i64 %.in3823, %sink_sink_3862		; visa id: 950
  br i1 %124, label %666, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 951

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
; BB:
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

666:                                              ; preds = %._crit_edge
; BB11 :
  %.sroa.256.0.insert.ext588 = zext i32 %647 to i64		; visa id: 953
  %667 = shl nuw nsw i64 %.sroa.256.0.insert.ext588, 1		; visa id: 954
  %668 = add i64 %sink_3906, %667		; visa id: 955
  %669 = inttoptr i64 %668 to i16 addrspace(4)*		; visa id: 956
  %670 = addrspacecast i16 addrspace(4)* %669 to i16 addrspace(1)*		; visa id: 956
  %671 = load i16, i16 addrspace(1)* %670, align 2		; visa id: 957
  %672 = add i64 %sink_3907, %667		; visa id: 959
  %673 = inttoptr i64 %672 to i16 addrspace(4)*		; visa id: 960
  %674 = addrspacecast i16 addrspace(4)* %673 to i16 addrspace(1)*		; visa id: 960
  %675 = load i16, i16 addrspace(1)* %674, align 2		; visa id: 961
  %676 = zext i16 %671 to i32		; visa id: 963
  %677 = shl nuw i32 %676, 16, !spirv.Decorations !888		; visa id: 964
  %678 = bitcast i32 %677 to float
  %679 = zext i16 %675 to i32		; visa id: 965
  %680 = shl nuw i32 %679, 16, !spirv.Decorations !888		; visa id: 966
  %681 = bitcast i32 %680 to float
  %682 = fmul reassoc nsz arcp contract float %678, %681, !spirv.Decorations !881
  %683 = fadd reassoc nsz arcp contract float %682, %.sroa.66.1, !spirv.Decorations !881		; visa id: 967
  br label %._crit_edge.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 968

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %666
; BB12 :
  %.sroa.66.2 = phi float [ %683, %666 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  %sink_sink_3885 = bitcast <2 x i32> %351 to i64		; visa id: 969
  %sink_sink_3861 = shl i64 %sink_sink_3885, 1		; visa id: 973
  %sink_3905 = add i64 %.in3823, %sink_sink_3861		; visa id: 974
  br i1 %128, label %684, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 975

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
; BB:
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

684:                                              ; preds = %._crit_edge.1
; BB14 :
  %.sroa.256.0.insert.ext593 = zext i32 %647 to i64		; visa id: 977
  %685 = shl nuw nsw i64 %.sroa.256.0.insert.ext593, 1		; visa id: 978
  %686 = add i64 %sink_3905, %685		; visa id: 979
  %687 = inttoptr i64 %686 to i16 addrspace(4)*		; visa id: 980
  %688 = addrspacecast i16 addrspace(4)* %687 to i16 addrspace(1)*		; visa id: 980
  %689 = load i16, i16 addrspace(1)* %688, align 2		; visa id: 981
  %690 = add i64 %sink_3907, %685		; visa id: 983
  %691 = inttoptr i64 %690 to i16 addrspace(4)*		; visa id: 984
  %692 = addrspacecast i16 addrspace(4)* %691 to i16 addrspace(1)*		; visa id: 984
  %693 = load i16, i16 addrspace(1)* %692, align 2		; visa id: 985
  %694 = zext i16 %689 to i32		; visa id: 987
  %695 = shl nuw i32 %694, 16, !spirv.Decorations !888		; visa id: 988
  %696 = bitcast i32 %695 to float
  %697 = zext i16 %693 to i32		; visa id: 989
  %698 = shl nuw i32 %697, 16, !spirv.Decorations !888		; visa id: 990
  %699 = bitcast i32 %698 to float
  %700 = fmul reassoc nsz arcp contract float %696, %699, !spirv.Decorations !881
  %701 = fadd reassoc nsz arcp contract float %700, %.sroa.130.1, !spirv.Decorations !881		; visa id: 991
  br label %._crit_edge.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 992

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %684
; BB15 :
  %.sroa.130.2 = phi float [ %701, %684 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  %sink_sink_3884 = bitcast <2 x i32> %357 to i64		; visa id: 993
  %sink_sink_3860 = shl i64 %sink_sink_3884, 1		; visa id: 997
  %sink_3904 = add i64 %.in3823, %sink_sink_3860		; visa id: 998
  br i1 %132, label %702, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 999

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
; BB:
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

702:                                              ; preds = %._crit_edge.2
; BB17 :
  %.sroa.256.0.insert.ext598 = zext i32 %647 to i64		; visa id: 1001
  %703 = shl nuw nsw i64 %.sroa.256.0.insert.ext598, 1		; visa id: 1002
  %704 = add i64 %sink_3904, %703		; visa id: 1003
  %705 = inttoptr i64 %704 to i16 addrspace(4)*		; visa id: 1004
  %706 = addrspacecast i16 addrspace(4)* %705 to i16 addrspace(1)*		; visa id: 1004
  %707 = load i16, i16 addrspace(1)* %706, align 2		; visa id: 1005
  %708 = add i64 %sink_3907, %703		; visa id: 1007
  %709 = inttoptr i64 %708 to i16 addrspace(4)*		; visa id: 1008
  %710 = addrspacecast i16 addrspace(4)* %709 to i16 addrspace(1)*		; visa id: 1008
  %711 = load i16, i16 addrspace(1)* %710, align 2		; visa id: 1009
  %712 = zext i16 %707 to i32		; visa id: 1011
  %713 = shl nuw i32 %712, 16, !spirv.Decorations !888		; visa id: 1012
  %714 = bitcast i32 %713 to float
  %715 = zext i16 %711 to i32		; visa id: 1013
  %716 = shl nuw i32 %715, 16, !spirv.Decorations !888		; visa id: 1014
  %717 = bitcast i32 %716 to float
  %718 = fmul reassoc nsz arcp contract float %714, %717, !spirv.Decorations !881
  %719 = fadd reassoc nsz arcp contract float %718, %.sroa.194.1, !spirv.Decorations !881		; visa id: 1015
  br label %.preheader, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1016

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %702
; BB18 :
  %.sroa.194.2 = phi float [ %719, %702 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %sink_sink_3883 = bitcast <2 x i32> %363 to i64		; visa id: 1017
  %sink_sink_3859 = shl i64 %sink_sink_3883, 1		; visa id: 1021
  %sink_3903 = add i64 %.in3822, %sink_sink_3859		; visa id: 1022
  br i1 %136, label %720, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1023

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
; BB:
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

720:                                              ; preds = %.preheader
; BB20 :
  %.sroa.256.0.insert.ext603 = zext i32 %647 to i64		; visa id: 1025
  %721 = shl nuw nsw i64 %.sroa.256.0.insert.ext603, 1		; visa id: 1026
  %722 = add i64 %sink_3908, %721		; visa id: 1027
  %723 = inttoptr i64 %722 to i16 addrspace(4)*		; visa id: 1028
  %724 = addrspacecast i16 addrspace(4)* %723 to i16 addrspace(1)*		; visa id: 1028
  %725 = load i16, i16 addrspace(1)* %724, align 2		; visa id: 1029
  %726 = add i64 %sink_3903, %721		; visa id: 1031
  %727 = inttoptr i64 %726 to i16 addrspace(4)*		; visa id: 1032
  %728 = addrspacecast i16 addrspace(4)* %727 to i16 addrspace(1)*		; visa id: 1032
  %729 = load i16, i16 addrspace(1)* %728, align 2		; visa id: 1033
  %730 = zext i16 %725 to i32		; visa id: 1035
  %731 = shl nuw i32 %730, 16, !spirv.Decorations !888		; visa id: 1036
  %732 = bitcast i32 %731 to float
  %733 = zext i16 %729 to i32		; visa id: 1037
  %734 = shl nuw i32 %733, 16, !spirv.Decorations !888		; visa id: 1038
  %735 = bitcast i32 %734 to float
  %736 = fmul reassoc nsz arcp contract float %732, %735, !spirv.Decorations !881
  %737 = fadd reassoc nsz arcp contract float %736, %.sroa.6.1, !spirv.Decorations !881		; visa id: 1039
  br label %._crit_edge.173, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1040

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %720
; BB21 :
  %.sroa.6.2 = phi float [ %737, %720 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %139, label %738, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1041

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
; BB:
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

738:                                              ; preds = %._crit_edge.173
; BB23 :
  %.sroa.256.0.insert.ext608 = zext i32 %647 to i64		; visa id: 1043
  %739 = shl nuw nsw i64 %.sroa.256.0.insert.ext608, 1		; visa id: 1044
  %740 = add i64 %sink_3906, %739		; visa id: 1045
  %741 = inttoptr i64 %740 to i16 addrspace(4)*		; visa id: 1046
  %742 = addrspacecast i16 addrspace(4)* %741 to i16 addrspace(1)*		; visa id: 1046
  %743 = load i16, i16 addrspace(1)* %742, align 2		; visa id: 1047
  %744 = add i64 %sink_3903, %739		; visa id: 1049
  %745 = inttoptr i64 %744 to i16 addrspace(4)*		; visa id: 1050
  %746 = addrspacecast i16 addrspace(4)* %745 to i16 addrspace(1)*		; visa id: 1050
  %747 = load i16, i16 addrspace(1)* %746, align 2		; visa id: 1051
  %748 = zext i16 %743 to i32		; visa id: 1053
  %749 = shl nuw i32 %748, 16, !spirv.Decorations !888		; visa id: 1054
  %750 = bitcast i32 %749 to float
  %751 = zext i16 %747 to i32		; visa id: 1055
  %752 = shl nuw i32 %751, 16, !spirv.Decorations !888		; visa id: 1056
  %753 = bitcast i32 %752 to float
  %754 = fmul reassoc nsz arcp contract float %750, %753, !spirv.Decorations !881
  %755 = fadd reassoc nsz arcp contract float %754, %.sroa.70.1, !spirv.Decorations !881		; visa id: 1057
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1058

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %738
; BB24 :
  %.sroa.70.2 = phi float [ %755, %738 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %142, label %756, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1059

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
; BB:
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

756:                                              ; preds = %._crit_edge.1.1
; BB26 :
  %.sroa.256.0.insert.ext613 = zext i32 %647 to i64		; visa id: 1061
  %757 = shl nuw nsw i64 %.sroa.256.0.insert.ext613, 1		; visa id: 1062
  %758 = add i64 %sink_3905, %757		; visa id: 1063
  %759 = inttoptr i64 %758 to i16 addrspace(4)*		; visa id: 1064
  %760 = addrspacecast i16 addrspace(4)* %759 to i16 addrspace(1)*		; visa id: 1064
  %761 = load i16, i16 addrspace(1)* %760, align 2		; visa id: 1065
  %762 = add i64 %sink_3903, %757		; visa id: 1067
  %763 = inttoptr i64 %762 to i16 addrspace(4)*		; visa id: 1068
  %764 = addrspacecast i16 addrspace(4)* %763 to i16 addrspace(1)*		; visa id: 1068
  %765 = load i16, i16 addrspace(1)* %764, align 2		; visa id: 1069
  %766 = zext i16 %761 to i32		; visa id: 1071
  %767 = shl nuw i32 %766, 16, !spirv.Decorations !888		; visa id: 1072
  %768 = bitcast i32 %767 to float
  %769 = zext i16 %765 to i32		; visa id: 1073
  %770 = shl nuw i32 %769, 16, !spirv.Decorations !888		; visa id: 1074
  %771 = bitcast i32 %770 to float
  %772 = fmul reassoc nsz arcp contract float %768, %771, !spirv.Decorations !881
  %773 = fadd reassoc nsz arcp contract float %772, %.sroa.134.1, !spirv.Decorations !881		; visa id: 1075
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1076

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %756
; BB27 :
  %.sroa.134.2 = phi float [ %773, %756 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %145, label %774, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1077

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
; BB:
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

774:                                              ; preds = %._crit_edge.2.1
; BB29 :
  %.sroa.256.0.insert.ext618 = zext i32 %647 to i64		; visa id: 1079
  %775 = shl nuw nsw i64 %.sroa.256.0.insert.ext618, 1		; visa id: 1080
  %776 = add i64 %sink_3904, %775		; visa id: 1081
  %777 = inttoptr i64 %776 to i16 addrspace(4)*		; visa id: 1082
  %778 = addrspacecast i16 addrspace(4)* %777 to i16 addrspace(1)*		; visa id: 1082
  %779 = load i16, i16 addrspace(1)* %778, align 2		; visa id: 1083
  %780 = add i64 %sink_3903, %775		; visa id: 1085
  %781 = inttoptr i64 %780 to i16 addrspace(4)*		; visa id: 1086
  %782 = addrspacecast i16 addrspace(4)* %781 to i16 addrspace(1)*		; visa id: 1086
  %783 = load i16, i16 addrspace(1)* %782, align 2		; visa id: 1087
  %784 = zext i16 %779 to i32		; visa id: 1089
  %785 = shl nuw i32 %784, 16, !spirv.Decorations !888		; visa id: 1090
  %786 = bitcast i32 %785 to float
  %787 = zext i16 %783 to i32		; visa id: 1091
  %788 = shl nuw i32 %787, 16, !spirv.Decorations !888		; visa id: 1092
  %789 = bitcast i32 %788 to float
  %790 = fmul reassoc nsz arcp contract float %786, %789, !spirv.Decorations !881
  %791 = fadd reassoc nsz arcp contract float %790, %.sroa.198.1, !spirv.Decorations !881		; visa id: 1093
  br label %.preheader.1, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1094

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %774
; BB30 :
  %.sroa.198.2 = phi float [ %791, %774 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %sink_sink_3882 = bitcast <2 x i32> %369 to i64		; visa id: 1095
  %sink_sink_3858 = shl i64 %sink_sink_3882, 1		; visa id: 1099
  %sink_3902 = add i64 %.in3822, %sink_sink_3858		; visa id: 1100
  br i1 %149, label %792, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1101

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
; BB:
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

792:                                              ; preds = %.preheader.1
; BB32 :
  %.sroa.256.0.insert.ext623 = zext i32 %647 to i64		; visa id: 1103
  %793 = shl nuw nsw i64 %.sroa.256.0.insert.ext623, 1		; visa id: 1104
  %794 = add i64 %sink_3908, %793		; visa id: 1105
  %795 = inttoptr i64 %794 to i16 addrspace(4)*		; visa id: 1106
  %796 = addrspacecast i16 addrspace(4)* %795 to i16 addrspace(1)*		; visa id: 1106
  %797 = load i16, i16 addrspace(1)* %796, align 2		; visa id: 1107
  %798 = add i64 %sink_3902, %793		; visa id: 1109
  %799 = inttoptr i64 %798 to i16 addrspace(4)*		; visa id: 1110
  %800 = addrspacecast i16 addrspace(4)* %799 to i16 addrspace(1)*		; visa id: 1110
  %801 = load i16, i16 addrspace(1)* %800, align 2		; visa id: 1111
  %802 = zext i16 %797 to i32		; visa id: 1113
  %803 = shl nuw i32 %802, 16, !spirv.Decorations !888		; visa id: 1114
  %804 = bitcast i32 %803 to float
  %805 = zext i16 %801 to i32		; visa id: 1115
  %806 = shl nuw i32 %805, 16, !spirv.Decorations !888		; visa id: 1116
  %807 = bitcast i32 %806 to float
  %808 = fmul reassoc nsz arcp contract float %804, %807, !spirv.Decorations !881
  %809 = fadd reassoc nsz arcp contract float %808, %.sroa.10.1, !spirv.Decorations !881		; visa id: 1117
  br label %._crit_edge.274, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1118

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %792
; BB33 :
  %.sroa.10.2 = phi float [ %809, %792 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %152, label %810, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1119

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
; BB:
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

810:                                              ; preds = %._crit_edge.274
; BB35 :
  %.sroa.256.0.insert.ext628 = zext i32 %647 to i64		; visa id: 1121
  %811 = shl nuw nsw i64 %.sroa.256.0.insert.ext628, 1		; visa id: 1122
  %812 = add i64 %sink_3906, %811		; visa id: 1123
  %813 = inttoptr i64 %812 to i16 addrspace(4)*		; visa id: 1124
  %814 = addrspacecast i16 addrspace(4)* %813 to i16 addrspace(1)*		; visa id: 1124
  %815 = load i16, i16 addrspace(1)* %814, align 2		; visa id: 1125
  %816 = add i64 %sink_3902, %811		; visa id: 1127
  %817 = inttoptr i64 %816 to i16 addrspace(4)*		; visa id: 1128
  %818 = addrspacecast i16 addrspace(4)* %817 to i16 addrspace(1)*		; visa id: 1128
  %819 = load i16, i16 addrspace(1)* %818, align 2		; visa id: 1129
  %820 = zext i16 %815 to i32		; visa id: 1131
  %821 = shl nuw i32 %820, 16, !spirv.Decorations !888		; visa id: 1132
  %822 = bitcast i32 %821 to float
  %823 = zext i16 %819 to i32		; visa id: 1133
  %824 = shl nuw i32 %823, 16, !spirv.Decorations !888		; visa id: 1134
  %825 = bitcast i32 %824 to float
  %826 = fmul reassoc nsz arcp contract float %822, %825, !spirv.Decorations !881
  %827 = fadd reassoc nsz arcp contract float %826, %.sroa.74.1, !spirv.Decorations !881		; visa id: 1135
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1136

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %810
; BB36 :
  %.sroa.74.2 = phi float [ %827, %810 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %155, label %828, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1137

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
; BB:
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

828:                                              ; preds = %._crit_edge.1.2
; BB38 :
  %.sroa.256.0.insert.ext633 = zext i32 %647 to i64		; visa id: 1139
  %829 = shl nuw nsw i64 %.sroa.256.0.insert.ext633, 1		; visa id: 1140
  %830 = add i64 %sink_3905, %829		; visa id: 1141
  %831 = inttoptr i64 %830 to i16 addrspace(4)*		; visa id: 1142
  %832 = addrspacecast i16 addrspace(4)* %831 to i16 addrspace(1)*		; visa id: 1142
  %833 = load i16, i16 addrspace(1)* %832, align 2		; visa id: 1143
  %834 = add i64 %sink_3902, %829		; visa id: 1145
  %835 = inttoptr i64 %834 to i16 addrspace(4)*		; visa id: 1146
  %836 = addrspacecast i16 addrspace(4)* %835 to i16 addrspace(1)*		; visa id: 1146
  %837 = load i16, i16 addrspace(1)* %836, align 2		; visa id: 1147
  %838 = zext i16 %833 to i32		; visa id: 1149
  %839 = shl nuw i32 %838, 16, !spirv.Decorations !888		; visa id: 1150
  %840 = bitcast i32 %839 to float
  %841 = zext i16 %837 to i32		; visa id: 1151
  %842 = shl nuw i32 %841, 16, !spirv.Decorations !888		; visa id: 1152
  %843 = bitcast i32 %842 to float
  %844 = fmul reassoc nsz arcp contract float %840, %843, !spirv.Decorations !881
  %845 = fadd reassoc nsz arcp contract float %844, %.sroa.138.1, !spirv.Decorations !881		; visa id: 1153
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1154

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %828
; BB39 :
  %.sroa.138.2 = phi float [ %845, %828 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %158, label %846, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1155

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
; BB:
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

846:                                              ; preds = %._crit_edge.2.2
; BB41 :
  %.sroa.256.0.insert.ext638 = zext i32 %647 to i64		; visa id: 1157
  %847 = shl nuw nsw i64 %.sroa.256.0.insert.ext638, 1		; visa id: 1158
  %848 = add i64 %sink_3904, %847		; visa id: 1159
  %849 = inttoptr i64 %848 to i16 addrspace(4)*		; visa id: 1160
  %850 = addrspacecast i16 addrspace(4)* %849 to i16 addrspace(1)*		; visa id: 1160
  %851 = load i16, i16 addrspace(1)* %850, align 2		; visa id: 1161
  %852 = add i64 %sink_3902, %847		; visa id: 1163
  %853 = inttoptr i64 %852 to i16 addrspace(4)*		; visa id: 1164
  %854 = addrspacecast i16 addrspace(4)* %853 to i16 addrspace(1)*		; visa id: 1164
  %855 = load i16, i16 addrspace(1)* %854, align 2		; visa id: 1165
  %856 = zext i16 %851 to i32		; visa id: 1167
  %857 = shl nuw i32 %856, 16, !spirv.Decorations !888		; visa id: 1168
  %858 = bitcast i32 %857 to float
  %859 = zext i16 %855 to i32		; visa id: 1169
  %860 = shl nuw i32 %859, 16, !spirv.Decorations !888		; visa id: 1170
  %861 = bitcast i32 %860 to float
  %862 = fmul reassoc nsz arcp contract float %858, %861, !spirv.Decorations !881
  %863 = fadd reassoc nsz arcp contract float %862, %.sroa.202.1, !spirv.Decorations !881		; visa id: 1171
  br label %.preheader.2, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1172

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %846
; BB42 :
  %.sroa.202.2 = phi float [ %863, %846 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %sink_sink_3881 = bitcast <2 x i32> %375 to i64		; visa id: 1173
  %sink_sink_3857 = shl i64 %sink_sink_3881, 1		; visa id: 1177
  %sink_3901 = add i64 %.in3822, %sink_sink_3857		; visa id: 1178
  br i1 %162, label %864, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1179

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
; BB:
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

864:                                              ; preds = %.preheader.2
; BB44 :
  %.sroa.256.0.insert.ext643 = zext i32 %647 to i64		; visa id: 1181
  %865 = shl nuw nsw i64 %.sroa.256.0.insert.ext643, 1		; visa id: 1182
  %866 = add i64 %sink_3908, %865		; visa id: 1183
  %867 = inttoptr i64 %866 to i16 addrspace(4)*		; visa id: 1184
  %868 = addrspacecast i16 addrspace(4)* %867 to i16 addrspace(1)*		; visa id: 1184
  %869 = load i16, i16 addrspace(1)* %868, align 2		; visa id: 1185
  %870 = add i64 %sink_3901, %865		; visa id: 1187
  %871 = inttoptr i64 %870 to i16 addrspace(4)*		; visa id: 1188
  %872 = addrspacecast i16 addrspace(4)* %871 to i16 addrspace(1)*		; visa id: 1188
  %873 = load i16, i16 addrspace(1)* %872, align 2		; visa id: 1189
  %874 = zext i16 %869 to i32		; visa id: 1191
  %875 = shl nuw i32 %874, 16, !spirv.Decorations !888		; visa id: 1192
  %876 = bitcast i32 %875 to float
  %877 = zext i16 %873 to i32		; visa id: 1193
  %878 = shl nuw i32 %877, 16, !spirv.Decorations !888		; visa id: 1194
  %879 = bitcast i32 %878 to float
  %880 = fmul reassoc nsz arcp contract float %876, %879, !spirv.Decorations !881
  %881 = fadd reassoc nsz arcp contract float %880, %.sroa.14.1, !spirv.Decorations !881		; visa id: 1195
  br label %._crit_edge.375, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1196

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %864
; BB45 :
  %.sroa.14.2 = phi float [ %881, %864 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %165, label %882, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1197

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
; BB:
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

882:                                              ; preds = %._crit_edge.375
; BB47 :
  %.sroa.256.0.insert.ext648 = zext i32 %647 to i64		; visa id: 1199
  %883 = shl nuw nsw i64 %.sroa.256.0.insert.ext648, 1		; visa id: 1200
  %884 = add i64 %sink_3906, %883		; visa id: 1201
  %885 = inttoptr i64 %884 to i16 addrspace(4)*		; visa id: 1202
  %886 = addrspacecast i16 addrspace(4)* %885 to i16 addrspace(1)*		; visa id: 1202
  %887 = load i16, i16 addrspace(1)* %886, align 2		; visa id: 1203
  %888 = add i64 %sink_3901, %883		; visa id: 1205
  %889 = inttoptr i64 %888 to i16 addrspace(4)*		; visa id: 1206
  %890 = addrspacecast i16 addrspace(4)* %889 to i16 addrspace(1)*		; visa id: 1206
  %891 = load i16, i16 addrspace(1)* %890, align 2		; visa id: 1207
  %892 = zext i16 %887 to i32		; visa id: 1209
  %893 = shl nuw i32 %892, 16, !spirv.Decorations !888		; visa id: 1210
  %894 = bitcast i32 %893 to float
  %895 = zext i16 %891 to i32		; visa id: 1211
  %896 = shl nuw i32 %895, 16, !spirv.Decorations !888		; visa id: 1212
  %897 = bitcast i32 %896 to float
  %898 = fmul reassoc nsz arcp contract float %894, %897, !spirv.Decorations !881
  %899 = fadd reassoc nsz arcp contract float %898, %.sroa.78.1, !spirv.Decorations !881		; visa id: 1213
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1214

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %882
; BB48 :
  %.sroa.78.2 = phi float [ %899, %882 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %168, label %900, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1215

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
; BB:
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

900:                                              ; preds = %._crit_edge.1.3
; BB50 :
  %.sroa.256.0.insert.ext653 = zext i32 %647 to i64		; visa id: 1217
  %901 = shl nuw nsw i64 %.sroa.256.0.insert.ext653, 1		; visa id: 1218
  %902 = add i64 %sink_3905, %901		; visa id: 1219
  %903 = inttoptr i64 %902 to i16 addrspace(4)*		; visa id: 1220
  %904 = addrspacecast i16 addrspace(4)* %903 to i16 addrspace(1)*		; visa id: 1220
  %905 = load i16, i16 addrspace(1)* %904, align 2		; visa id: 1221
  %906 = add i64 %sink_3901, %901		; visa id: 1223
  %907 = inttoptr i64 %906 to i16 addrspace(4)*		; visa id: 1224
  %908 = addrspacecast i16 addrspace(4)* %907 to i16 addrspace(1)*		; visa id: 1224
  %909 = load i16, i16 addrspace(1)* %908, align 2		; visa id: 1225
  %910 = zext i16 %905 to i32		; visa id: 1227
  %911 = shl nuw i32 %910, 16, !spirv.Decorations !888		; visa id: 1228
  %912 = bitcast i32 %911 to float
  %913 = zext i16 %909 to i32		; visa id: 1229
  %914 = shl nuw i32 %913, 16, !spirv.Decorations !888		; visa id: 1230
  %915 = bitcast i32 %914 to float
  %916 = fmul reassoc nsz arcp contract float %912, %915, !spirv.Decorations !881
  %917 = fadd reassoc nsz arcp contract float %916, %.sroa.142.1, !spirv.Decorations !881		; visa id: 1231
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1232

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %900
; BB51 :
  %.sroa.142.2 = phi float [ %917, %900 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %171, label %918, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1233

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
; BB:
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

918:                                              ; preds = %._crit_edge.2.3
; BB53 :
  %.sroa.256.0.insert.ext658 = zext i32 %647 to i64		; visa id: 1235
  %919 = shl nuw nsw i64 %.sroa.256.0.insert.ext658, 1		; visa id: 1236
  %920 = add i64 %sink_3904, %919		; visa id: 1237
  %921 = inttoptr i64 %920 to i16 addrspace(4)*		; visa id: 1238
  %922 = addrspacecast i16 addrspace(4)* %921 to i16 addrspace(1)*		; visa id: 1238
  %923 = load i16, i16 addrspace(1)* %922, align 2		; visa id: 1239
  %924 = add i64 %sink_3901, %919		; visa id: 1241
  %925 = inttoptr i64 %924 to i16 addrspace(4)*		; visa id: 1242
  %926 = addrspacecast i16 addrspace(4)* %925 to i16 addrspace(1)*		; visa id: 1242
  %927 = load i16, i16 addrspace(1)* %926, align 2		; visa id: 1243
  %928 = zext i16 %923 to i32		; visa id: 1245
  %929 = shl nuw i32 %928, 16, !spirv.Decorations !888		; visa id: 1246
  %930 = bitcast i32 %929 to float
  %931 = zext i16 %927 to i32		; visa id: 1247
  %932 = shl nuw i32 %931, 16, !spirv.Decorations !888		; visa id: 1248
  %933 = bitcast i32 %932 to float
  %934 = fmul reassoc nsz arcp contract float %930, %933, !spirv.Decorations !881
  %935 = fadd reassoc nsz arcp contract float %934, %.sroa.206.1, !spirv.Decorations !881		; visa id: 1249
  br label %.preheader.3, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1250

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %918
; BB54 :
  %.sroa.206.2 = phi float [ %935, %918 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %sink_sink_3880 = bitcast <2 x i32> %381 to i64		; visa id: 1251
  %sink_sink_3856 = shl i64 %sink_sink_3880, 1		; visa id: 1255
  %sink_3900 = add i64 %.in3822, %sink_sink_3856		; visa id: 1256
  br i1 %175, label %936, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1257

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
; BB:
  br label %._crit_edge.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

936:                                              ; preds = %.preheader.3
; BB56 :
  %.sroa.256.0.insert.ext663 = zext i32 %647 to i64		; visa id: 1259
  %937 = shl nuw nsw i64 %.sroa.256.0.insert.ext663, 1		; visa id: 1260
  %938 = add i64 %sink_3908, %937		; visa id: 1261
  %939 = inttoptr i64 %938 to i16 addrspace(4)*		; visa id: 1262
  %940 = addrspacecast i16 addrspace(4)* %939 to i16 addrspace(1)*		; visa id: 1262
  %941 = load i16, i16 addrspace(1)* %940, align 2		; visa id: 1263
  %942 = add i64 %sink_3900, %937		; visa id: 1265
  %943 = inttoptr i64 %942 to i16 addrspace(4)*		; visa id: 1266
  %944 = addrspacecast i16 addrspace(4)* %943 to i16 addrspace(1)*		; visa id: 1266
  %945 = load i16, i16 addrspace(1)* %944, align 2		; visa id: 1267
  %946 = zext i16 %941 to i32		; visa id: 1269
  %947 = shl nuw i32 %946, 16, !spirv.Decorations !888		; visa id: 1270
  %948 = bitcast i32 %947 to float
  %949 = zext i16 %945 to i32		; visa id: 1271
  %950 = shl nuw i32 %949, 16, !spirv.Decorations !888		; visa id: 1272
  %951 = bitcast i32 %950 to float
  %952 = fmul reassoc nsz arcp contract float %948, %951, !spirv.Decorations !881
  %953 = fadd reassoc nsz arcp contract float %952, %.sroa.18.1, !spirv.Decorations !881		; visa id: 1273
  br label %._crit_edge.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1274

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %936
; BB57 :
  %.sroa.18.2 = phi float [ %953, %936 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  br i1 %178, label %954, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1275

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
; BB:
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

954:                                              ; preds = %._crit_edge.4
; BB59 :
  %.sroa.256.0.insert.ext668 = zext i32 %647 to i64		; visa id: 1277
  %955 = shl nuw nsw i64 %.sroa.256.0.insert.ext668, 1		; visa id: 1278
  %956 = add i64 %sink_3906, %955		; visa id: 1279
  %957 = inttoptr i64 %956 to i16 addrspace(4)*		; visa id: 1280
  %958 = addrspacecast i16 addrspace(4)* %957 to i16 addrspace(1)*		; visa id: 1280
  %959 = load i16, i16 addrspace(1)* %958, align 2		; visa id: 1281
  %960 = add i64 %sink_3900, %955		; visa id: 1283
  %961 = inttoptr i64 %960 to i16 addrspace(4)*		; visa id: 1284
  %962 = addrspacecast i16 addrspace(4)* %961 to i16 addrspace(1)*		; visa id: 1284
  %963 = load i16, i16 addrspace(1)* %962, align 2		; visa id: 1285
  %964 = zext i16 %959 to i32		; visa id: 1287
  %965 = shl nuw i32 %964, 16, !spirv.Decorations !888		; visa id: 1288
  %966 = bitcast i32 %965 to float
  %967 = zext i16 %963 to i32		; visa id: 1289
  %968 = shl nuw i32 %967, 16, !spirv.Decorations !888		; visa id: 1290
  %969 = bitcast i32 %968 to float
  %970 = fmul reassoc nsz arcp contract float %966, %969, !spirv.Decorations !881
  %971 = fadd reassoc nsz arcp contract float %970, %.sroa.82.1, !spirv.Decorations !881		; visa id: 1291
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1292

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %954
; BB60 :
  %.sroa.82.2 = phi float [ %971, %954 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  br i1 %181, label %972, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1293

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
; BB:
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

972:                                              ; preds = %._crit_edge.1.4
; BB62 :
  %.sroa.256.0.insert.ext673 = zext i32 %647 to i64		; visa id: 1295
  %973 = shl nuw nsw i64 %.sroa.256.0.insert.ext673, 1		; visa id: 1296
  %974 = add i64 %sink_3905, %973		; visa id: 1297
  %975 = inttoptr i64 %974 to i16 addrspace(4)*		; visa id: 1298
  %976 = addrspacecast i16 addrspace(4)* %975 to i16 addrspace(1)*		; visa id: 1298
  %977 = load i16, i16 addrspace(1)* %976, align 2		; visa id: 1299
  %978 = add i64 %sink_3900, %973		; visa id: 1301
  %979 = inttoptr i64 %978 to i16 addrspace(4)*		; visa id: 1302
  %980 = addrspacecast i16 addrspace(4)* %979 to i16 addrspace(1)*		; visa id: 1302
  %981 = load i16, i16 addrspace(1)* %980, align 2		; visa id: 1303
  %982 = zext i16 %977 to i32		; visa id: 1305
  %983 = shl nuw i32 %982, 16, !spirv.Decorations !888		; visa id: 1306
  %984 = bitcast i32 %983 to float
  %985 = zext i16 %981 to i32		; visa id: 1307
  %986 = shl nuw i32 %985, 16, !spirv.Decorations !888		; visa id: 1308
  %987 = bitcast i32 %986 to float
  %988 = fmul reassoc nsz arcp contract float %984, %987, !spirv.Decorations !881
  %989 = fadd reassoc nsz arcp contract float %988, %.sroa.146.1, !spirv.Decorations !881		; visa id: 1309
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1310

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %972
; BB63 :
  %.sroa.146.2 = phi float [ %989, %972 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  br i1 %184, label %990, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1311

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
; BB:
  br label %.preheader.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

990:                                              ; preds = %._crit_edge.2.4
; BB65 :
  %.sroa.256.0.insert.ext678 = zext i32 %647 to i64		; visa id: 1313
  %991 = shl nuw nsw i64 %.sroa.256.0.insert.ext678, 1		; visa id: 1314
  %992 = add i64 %sink_3904, %991		; visa id: 1315
  %993 = inttoptr i64 %992 to i16 addrspace(4)*		; visa id: 1316
  %994 = addrspacecast i16 addrspace(4)* %993 to i16 addrspace(1)*		; visa id: 1316
  %995 = load i16, i16 addrspace(1)* %994, align 2		; visa id: 1317
  %996 = add i64 %sink_3900, %991		; visa id: 1319
  %997 = inttoptr i64 %996 to i16 addrspace(4)*		; visa id: 1320
  %998 = addrspacecast i16 addrspace(4)* %997 to i16 addrspace(1)*		; visa id: 1320
  %999 = load i16, i16 addrspace(1)* %998, align 2		; visa id: 1321
  %1000 = zext i16 %995 to i32		; visa id: 1323
  %1001 = shl nuw i32 %1000, 16, !spirv.Decorations !888		; visa id: 1324
  %1002 = bitcast i32 %1001 to float
  %1003 = zext i16 %999 to i32		; visa id: 1325
  %1004 = shl nuw i32 %1003, 16, !spirv.Decorations !888		; visa id: 1326
  %1005 = bitcast i32 %1004 to float
  %1006 = fmul reassoc nsz arcp contract float %1002, %1005, !spirv.Decorations !881
  %1007 = fadd reassoc nsz arcp contract float %1006, %.sroa.210.1, !spirv.Decorations !881		; visa id: 1327
  br label %.preheader.4, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1328

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %990
; BB66 :
  %.sroa.210.2 = phi float [ %1007, %990 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %sink_sink_3879 = bitcast <2 x i32> %387 to i64		; visa id: 1329
  %sink_sink_3855 = shl i64 %sink_sink_3879, 1		; visa id: 1333
  %sink_3899 = add i64 %.in3822, %sink_sink_3855		; visa id: 1334
  br i1 %188, label %1008, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1335

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
; BB:
  br label %._crit_edge.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1008:                                             ; preds = %.preheader.4
; BB68 :
  %.sroa.256.0.insert.ext683 = zext i32 %647 to i64		; visa id: 1337
  %1009 = shl nuw nsw i64 %.sroa.256.0.insert.ext683, 1		; visa id: 1338
  %1010 = add i64 %sink_3908, %1009		; visa id: 1339
  %1011 = inttoptr i64 %1010 to i16 addrspace(4)*		; visa id: 1340
  %1012 = addrspacecast i16 addrspace(4)* %1011 to i16 addrspace(1)*		; visa id: 1340
  %1013 = load i16, i16 addrspace(1)* %1012, align 2		; visa id: 1341
  %1014 = add i64 %sink_3899, %1009		; visa id: 1343
  %1015 = inttoptr i64 %1014 to i16 addrspace(4)*		; visa id: 1344
  %1016 = addrspacecast i16 addrspace(4)* %1015 to i16 addrspace(1)*		; visa id: 1344
  %1017 = load i16, i16 addrspace(1)* %1016, align 2		; visa id: 1345
  %1018 = zext i16 %1013 to i32		; visa id: 1347
  %1019 = shl nuw i32 %1018, 16, !spirv.Decorations !888		; visa id: 1348
  %1020 = bitcast i32 %1019 to float
  %1021 = zext i16 %1017 to i32		; visa id: 1349
  %1022 = shl nuw i32 %1021, 16, !spirv.Decorations !888		; visa id: 1350
  %1023 = bitcast i32 %1022 to float
  %1024 = fmul reassoc nsz arcp contract float %1020, %1023, !spirv.Decorations !881
  %1025 = fadd reassoc nsz arcp contract float %1024, %.sroa.22.1, !spirv.Decorations !881		; visa id: 1351
  br label %._crit_edge.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1352

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %1008
; BB69 :
  %.sroa.22.2 = phi float [ %1025, %1008 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  br i1 %191, label %1026, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1353

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
; BB:
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1026:                                             ; preds = %._crit_edge.5
; BB71 :
  %.sroa.256.0.insert.ext688 = zext i32 %647 to i64		; visa id: 1355
  %1027 = shl nuw nsw i64 %.sroa.256.0.insert.ext688, 1		; visa id: 1356
  %1028 = add i64 %sink_3906, %1027		; visa id: 1357
  %1029 = inttoptr i64 %1028 to i16 addrspace(4)*		; visa id: 1358
  %1030 = addrspacecast i16 addrspace(4)* %1029 to i16 addrspace(1)*		; visa id: 1358
  %1031 = load i16, i16 addrspace(1)* %1030, align 2		; visa id: 1359
  %1032 = add i64 %sink_3899, %1027		; visa id: 1361
  %1033 = inttoptr i64 %1032 to i16 addrspace(4)*		; visa id: 1362
  %1034 = addrspacecast i16 addrspace(4)* %1033 to i16 addrspace(1)*		; visa id: 1362
  %1035 = load i16, i16 addrspace(1)* %1034, align 2		; visa id: 1363
  %1036 = zext i16 %1031 to i32		; visa id: 1365
  %1037 = shl nuw i32 %1036, 16, !spirv.Decorations !888		; visa id: 1366
  %1038 = bitcast i32 %1037 to float
  %1039 = zext i16 %1035 to i32		; visa id: 1367
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !888		; visa id: 1368
  %1041 = bitcast i32 %1040 to float
  %1042 = fmul reassoc nsz arcp contract float %1038, %1041, !spirv.Decorations !881
  %1043 = fadd reassoc nsz arcp contract float %1042, %.sroa.86.1, !spirv.Decorations !881		; visa id: 1369
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1370

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %1026
; BB72 :
  %.sroa.86.2 = phi float [ %1043, %1026 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  br i1 %194, label %1044, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1371

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
; BB:
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1044:                                             ; preds = %._crit_edge.1.5
; BB74 :
  %.sroa.256.0.insert.ext693 = zext i32 %647 to i64		; visa id: 1373
  %1045 = shl nuw nsw i64 %.sroa.256.0.insert.ext693, 1		; visa id: 1374
  %1046 = add i64 %sink_3905, %1045		; visa id: 1375
  %1047 = inttoptr i64 %1046 to i16 addrspace(4)*		; visa id: 1376
  %1048 = addrspacecast i16 addrspace(4)* %1047 to i16 addrspace(1)*		; visa id: 1376
  %1049 = load i16, i16 addrspace(1)* %1048, align 2		; visa id: 1377
  %1050 = add i64 %sink_3899, %1045		; visa id: 1379
  %1051 = inttoptr i64 %1050 to i16 addrspace(4)*		; visa id: 1380
  %1052 = addrspacecast i16 addrspace(4)* %1051 to i16 addrspace(1)*		; visa id: 1380
  %1053 = load i16, i16 addrspace(1)* %1052, align 2		; visa id: 1381
  %1054 = zext i16 %1049 to i32		; visa id: 1383
  %1055 = shl nuw i32 %1054, 16, !spirv.Decorations !888		; visa id: 1384
  %1056 = bitcast i32 %1055 to float
  %1057 = zext i16 %1053 to i32		; visa id: 1385
  %1058 = shl nuw i32 %1057, 16, !spirv.Decorations !888		; visa id: 1386
  %1059 = bitcast i32 %1058 to float
  %1060 = fmul reassoc nsz arcp contract float %1056, %1059, !spirv.Decorations !881
  %1061 = fadd reassoc nsz arcp contract float %1060, %.sroa.150.1, !spirv.Decorations !881		; visa id: 1387
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1388

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %1044
; BB75 :
  %.sroa.150.2 = phi float [ %1061, %1044 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  br i1 %197, label %1062, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1389

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
; BB:
  br label %.preheader.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1062:                                             ; preds = %._crit_edge.2.5
; BB77 :
  %.sroa.256.0.insert.ext698 = zext i32 %647 to i64		; visa id: 1391
  %1063 = shl nuw nsw i64 %.sroa.256.0.insert.ext698, 1		; visa id: 1392
  %1064 = add i64 %sink_3904, %1063		; visa id: 1393
  %1065 = inttoptr i64 %1064 to i16 addrspace(4)*		; visa id: 1394
  %1066 = addrspacecast i16 addrspace(4)* %1065 to i16 addrspace(1)*		; visa id: 1394
  %1067 = load i16, i16 addrspace(1)* %1066, align 2		; visa id: 1395
  %1068 = add i64 %sink_3899, %1063		; visa id: 1397
  %1069 = inttoptr i64 %1068 to i16 addrspace(4)*		; visa id: 1398
  %1070 = addrspacecast i16 addrspace(4)* %1069 to i16 addrspace(1)*		; visa id: 1398
  %1071 = load i16, i16 addrspace(1)* %1070, align 2		; visa id: 1399
  %1072 = zext i16 %1067 to i32		; visa id: 1401
  %1073 = shl nuw i32 %1072, 16, !spirv.Decorations !888		; visa id: 1402
  %1074 = bitcast i32 %1073 to float
  %1075 = zext i16 %1071 to i32		; visa id: 1403
  %1076 = shl nuw i32 %1075, 16, !spirv.Decorations !888		; visa id: 1404
  %1077 = bitcast i32 %1076 to float
  %1078 = fmul reassoc nsz arcp contract float %1074, %1077, !spirv.Decorations !881
  %1079 = fadd reassoc nsz arcp contract float %1078, %.sroa.214.1, !spirv.Decorations !881		; visa id: 1405
  br label %.preheader.5, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1406

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %1062
; BB78 :
  %.sroa.214.2 = phi float [ %1079, %1062 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %sink_sink_3878 = bitcast <2 x i32> %393 to i64		; visa id: 1407
  %sink_sink_3854 = shl i64 %sink_sink_3878, 1		; visa id: 1411
  %sink_3898 = add i64 %.in3822, %sink_sink_3854		; visa id: 1412
  br i1 %201, label %1080, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1413

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
; BB:
  br label %._crit_edge.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1080:                                             ; preds = %.preheader.5
; BB80 :
  %.sroa.256.0.insert.ext703 = zext i32 %647 to i64		; visa id: 1415
  %1081 = shl nuw nsw i64 %.sroa.256.0.insert.ext703, 1		; visa id: 1416
  %1082 = add i64 %sink_3908, %1081		; visa id: 1417
  %1083 = inttoptr i64 %1082 to i16 addrspace(4)*		; visa id: 1418
  %1084 = addrspacecast i16 addrspace(4)* %1083 to i16 addrspace(1)*		; visa id: 1418
  %1085 = load i16, i16 addrspace(1)* %1084, align 2		; visa id: 1419
  %1086 = add i64 %sink_3898, %1081		; visa id: 1421
  %1087 = inttoptr i64 %1086 to i16 addrspace(4)*		; visa id: 1422
  %1088 = addrspacecast i16 addrspace(4)* %1087 to i16 addrspace(1)*		; visa id: 1422
  %1089 = load i16, i16 addrspace(1)* %1088, align 2		; visa id: 1423
  %1090 = zext i16 %1085 to i32		; visa id: 1425
  %1091 = shl nuw i32 %1090, 16, !spirv.Decorations !888		; visa id: 1426
  %1092 = bitcast i32 %1091 to float
  %1093 = zext i16 %1089 to i32		; visa id: 1427
  %1094 = shl nuw i32 %1093, 16, !spirv.Decorations !888		; visa id: 1428
  %1095 = bitcast i32 %1094 to float
  %1096 = fmul reassoc nsz arcp contract float %1092, %1095, !spirv.Decorations !881
  %1097 = fadd reassoc nsz arcp contract float %1096, %.sroa.26.1, !spirv.Decorations !881		; visa id: 1429
  br label %._crit_edge.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1430

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %1080
; BB81 :
  %.sroa.26.2 = phi float [ %1097, %1080 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  br i1 %204, label %1098, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1431

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
; BB:
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1098:                                             ; preds = %._crit_edge.6
; BB83 :
  %.sroa.256.0.insert.ext708 = zext i32 %647 to i64		; visa id: 1433
  %1099 = shl nuw nsw i64 %.sroa.256.0.insert.ext708, 1		; visa id: 1434
  %1100 = add i64 %sink_3906, %1099		; visa id: 1435
  %1101 = inttoptr i64 %1100 to i16 addrspace(4)*		; visa id: 1436
  %1102 = addrspacecast i16 addrspace(4)* %1101 to i16 addrspace(1)*		; visa id: 1436
  %1103 = load i16, i16 addrspace(1)* %1102, align 2		; visa id: 1437
  %1104 = add i64 %sink_3898, %1099		; visa id: 1439
  %1105 = inttoptr i64 %1104 to i16 addrspace(4)*		; visa id: 1440
  %1106 = addrspacecast i16 addrspace(4)* %1105 to i16 addrspace(1)*		; visa id: 1440
  %1107 = load i16, i16 addrspace(1)* %1106, align 2		; visa id: 1441
  %1108 = zext i16 %1103 to i32		; visa id: 1443
  %1109 = shl nuw i32 %1108, 16, !spirv.Decorations !888		; visa id: 1444
  %1110 = bitcast i32 %1109 to float
  %1111 = zext i16 %1107 to i32		; visa id: 1445
  %1112 = shl nuw i32 %1111, 16, !spirv.Decorations !888		; visa id: 1446
  %1113 = bitcast i32 %1112 to float
  %1114 = fmul reassoc nsz arcp contract float %1110, %1113, !spirv.Decorations !881
  %1115 = fadd reassoc nsz arcp contract float %1114, %.sroa.90.1, !spirv.Decorations !881		; visa id: 1447
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1448

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %1098
; BB84 :
  %.sroa.90.2 = phi float [ %1115, %1098 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  br i1 %207, label %1116, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1449

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
; BB:
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1116:                                             ; preds = %._crit_edge.1.6
; BB86 :
  %.sroa.256.0.insert.ext713 = zext i32 %647 to i64		; visa id: 1451
  %1117 = shl nuw nsw i64 %.sroa.256.0.insert.ext713, 1		; visa id: 1452
  %1118 = add i64 %sink_3905, %1117		; visa id: 1453
  %1119 = inttoptr i64 %1118 to i16 addrspace(4)*		; visa id: 1454
  %1120 = addrspacecast i16 addrspace(4)* %1119 to i16 addrspace(1)*		; visa id: 1454
  %1121 = load i16, i16 addrspace(1)* %1120, align 2		; visa id: 1455
  %1122 = add i64 %sink_3898, %1117		; visa id: 1457
  %1123 = inttoptr i64 %1122 to i16 addrspace(4)*		; visa id: 1458
  %1124 = addrspacecast i16 addrspace(4)* %1123 to i16 addrspace(1)*		; visa id: 1458
  %1125 = load i16, i16 addrspace(1)* %1124, align 2		; visa id: 1459
  %1126 = zext i16 %1121 to i32		; visa id: 1461
  %1127 = shl nuw i32 %1126, 16, !spirv.Decorations !888		; visa id: 1462
  %1128 = bitcast i32 %1127 to float
  %1129 = zext i16 %1125 to i32		; visa id: 1463
  %1130 = shl nuw i32 %1129, 16, !spirv.Decorations !888		; visa id: 1464
  %1131 = bitcast i32 %1130 to float
  %1132 = fmul reassoc nsz arcp contract float %1128, %1131, !spirv.Decorations !881
  %1133 = fadd reassoc nsz arcp contract float %1132, %.sroa.154.1, !spirv.Decorations !881		; visa id: 1465
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1466

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %1116
; BB87 :
  %.sroa.154.2 = phi float [ %1133, %1116 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  br i1 %210, label %1134, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1467

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
; BB:
  br label %.preheader.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1134:                                             ; preds = %._crit_edge.2.6
; BB89 :
  %.sroa.256.0.insert.ext718 = zext i32 %647 to i64		; visa id: 1469
  %1135 = shl nuw nsw i64 %.sroa.256.0.insert.ext718, 1		; visa id: 1470
  %1136 = add i64 %sink_3904, %1135		; visa id: 1471
  %1137 = inttoptr i64 %1136 to i16 addrspace(4)*		; visa id: 1472
  %1138 = addrspacecast i16 addrspace(4)* %1137 to i16 addrspace(1)*		; visa id: 1472
  %1139 = load i16, i16 addrspace(1)* %1138, align 2		; visa id: 1473
  %1140 = add i64 %sink_3898, %1135		; visa id: 1475
  %1141 = inttoptr i64 %1140 to i16 addrspace(4)*		; visa id: 1476
  %1142 = addrspacecast i16 addrspace(4)* %1141 to i16 addrspace(1)*		; visa id: 1476
  %1143 = load i16, i16 addrspace(1)* %1142, align 2		; visa id: 1477
  %1144 = zext i16 %1139 to i32		; visa id: 1479
  %1145 = shl nuw i32 %1144, 16, !spirv.Decorations !888		; visa id: 1480
  %1146 = bitcast i32 %1145 to float
  %1147 = zext i16 %1143 to i32		; visa id: 1481
  %1148 = shl nuw i32 %1147, 16, !spirv.Decorations !888		; visa id: 1482
  %1149 = bitcast i32 %1148 to float
  %1150 = fmul reassoc nsz arcp contract float %1146, %1149, !spirv.Decorations !881
  %1151 = fadd reassoc nsz arcp contract float %1150, %.sroa.218.1, !spirv.Decorations !881		; visa id: 1483
  br label %.preheader.6, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1484

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %1134
; BB90 :
  %.sroa.218.2 = phi float [ %1151, %1134 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %sink_sink_3877 = bitcast <2 x i32> %399 to i64		; visa id: 1485
  %sink_sink_3853 = shl i64 %sink_sink_3877, 1		; visa id: 1489
  %sink_3897 = add i64 %.in3822, %sink_sink_3853		; visa id: 1490
  br i1 %214, label %1152, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1491

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
; BB:
  br label %._crit_edge.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1152:                                             ; preds = %.preheader.6
; BB92 :
  %.sroa.256.0.insert.ext723 = zext i32 %647 to i64		; visa id: 1493
  %1153 = shl nuw nsw i64 %.sroa.256.0.insert.ext723, 1		; visa id: 1494
  %1154 = add i64 %sink_3908, %1153		; visa id: 1495
  %1155 = inttoptr i64 %1154 to i16 addrspace(4)*		; visa id: 1496
  %1156 = addrspacecast i16 addrspace(4)* %1155 to i16 addrspace(1)*		; visa id: 1496
  %1157 = load i16, i16 addrspace(1)* %1156, align 2		; visa id: 1497
  %1158 = add i64 %sink_3897, %1153		; visa id: 1499
  %1159 = inttoptr i64 %1158 to i16 addrspace(4)*		; visa id: 1500
  %1160 = addrspacecast i16 addrspace(4)* %1159 to i16 addrspace(1)*		; visa id: 1500
  %1161 = load i16, i16 addrspace(1)* %1160, align 2		; visa id: 1501
  %1162 = zext i16 %1157 to i32		; visa id: 1503
  %1163 = shl nuw i32 %1162, 16, !spirv.Decorations !888		; visa id: 1504
  %1164 = bitcast i32 %1163 to float
  %1165 = zext i16 %1161 to i32		; visa id: 1505
  %1166 = shl nuw i32 %1165, 16, !spirv.Decorations !888		; visa id: 1506
  %1167 = bitcast i32 %1166 to float
  %1168 = fmul reassoc nsz arcp contract float %1164, %1167, !spirv.Decorations !881
  %1169 = fadd reassoc nsz arcp contract float %1168, %.sroa.30.1, !spirv.Decorations !881		; visa id: 1507
  br label %._crit_edge.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1508

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %1152
; BB93 :
  %.sroa.30.2 = phi float [ %1169, %1152 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  br i1 %217, label %1170, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1509

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
; BB:
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1170:                                             ; preds = %._crit_edge.7
; BB95 :
  %.sroa.256.0.insert.ext728 = zext i32 %647 to i64		; visa id: 1511
  %1171 = shl nuw nsw i64 %.sroa.256.0.insert.ext728, 1		; visa id: 1512
  %1172 = add i64 %sink_3906, %1171		; visa id: 1513
  %1173 = inttoptr i64 %1172 to i16 addrspace(4)*		; visa id: 1514
  %1174 = addrspacecast i16 addrspace(4)* %1173 to i16 addrspace(1)*		; visa id: 1514
  %1175 = load i16, i16 addrspace(1)* %1174, align 2		; visa id: 1515
  %1176 = add i64 %sink_3897, %1171		; visa id: 1517
  %1177 = inttoptr i64 %1176 to i16 addrspace(4)*		; visa id: 1518
  %1178 = addrspacecast i16 addrspace(4)* %1177 to i16 addrspace(1)*		; visa id: 1518
  %1179 = load i16, i16 addrspace(1)* %1178, align 2		; visa id: 1519
  %1180 = zext i16 %1175 to i32		; visa id: 1521
  %1181 = shl nuw i32 %1180, 16, !spirv.Decorations !888		; visa id: 1522
  %1182 = bitcast i32 %1181 to float
  %1183 = zext i16 %1179 to i32		; visa id: 1523
  %1184 = shl nuw i32 %1183, 16, !spirv.Decorations !888		; visa id: 1524
  %1185 = bitcast i32 %1184 to float
  %1186 = fmul reassoc nsz arcp contract float %1182, %1185, !spirv.Decorations !881
  %1187 = fadd reassoc nsz arcp contract float %1186, %.sroa.94.1, !spirv.Decorations !881		; visa id: 1525
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1526

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %1170
; BB96 :
  %.sroa.94.2 = phi float [ %1187, %1170 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  br i1 %220, label %1188, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1527

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
; BB:
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1188:                                             ; preds = %._crit_edge.1.7
; BB98 :
  %.sroa.256.0.insert.ext733 = zext i32 %647 to i64		; visa id: 1529
  %1189 = shl nuw nsw i64 %.sroa.256.0.insert.ext733, 1		; visa id: 1530
  %1190 = add i64 %sink_3905, %1189		; visa id: 1531
  %1191 = inttoptr i64 %1190 to i16 addrspace(4)*		; visa id: 1532
  %1192 = addrspacecast i16 addrspace(4)* %1191 to i16 addrspace(1)*		; visa id: 1532
  %1193 = load i16, i16 addrspace(1)* %1192, align 2		; visa id: 1533
  %1194 = add i64 %sink_3897, %1189		; visa id: 1535
  %1195 = inttoptr i64 %1194 to i16 addrspace(4)*		; visa id: 1536
  %1196 = addrspacecast i16 addrspace(4)* %1195 to i16 addrspace(1)*		; visa id: 1536
  %1197 = load i16, i16 addrspace(1)* %1196, align 2		; visa id: 1537
  %1198 = zext i16 %1193 to i32		; visa id: 1539
  %1199 = shl nuw i32 %1198, 16, !spirv.Decorations !888		; visa id: 1540
  %1200 = bitcast i32 %1199 to float
  %1201 = zext i16 %1197 to i32		; visa id: 1541
  %1202 = shl nuw i32 %1201, 16, !spirv.Decorations !888		; visa id: 1542
  %1203 = bitcast i32 %1202 to float
  %1204 = fmul reassoc nsz arcp contract float %1200, %1203, !spirv.Decorations !881
  %1205 = fadd reassoc nsz arcp contract float %1204, %.sroa.158.1, !spirv.Decorations !881		; visa id: 1543
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1544

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %1188
; BB99 :
  %.sroa.158.2 = phi float [ %1205, %1188 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  br i1 %223, label %1206, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1545

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
; BB:
  br label %.preheader.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1206:                                             ; preds = %._crit_edge.2.7
; BB101 :
  %.sroa.256.0.insert.ext738 = zext i32 %647 to i64		; visa id: 1547
  %1207 = shl nuw nsw i64 %.sroa.256.0.insert.ext738, 1		; visa id: 1548
  %1208 = add i64 %sink_3904, %1207		; visa id: 1549
  %1209 = inttoptr i64 %1208 to i16 addrspace(4)*		; visa id: 1550
  %1210 = addrspacecast i16 addrspace(4)* %1209 to i16 addrspace(1)*		; visa id: 1550
  %1211 = load i16, i16 addrspace(1)* %1210, align 2		; visa id: 1551
  %1212 = add i64 %sink_3897, %1207		; visa id: 1553
  %1213 = inttoptr i64 %1212 to i16 addrspace(4)*		; visa id: 1554
  %1214 = addrspacecast i16 addrspace(4)* %1213 to i16 addrspace(1)*		; visa id: 1554
  %1215 = load i16, i16 addrspace(1)* %1214, align 2		; visa id: 1555
  %1216 = zext i16 %1211 to i32		; visa id: 1557
  %1217 = shl nuw i32 %1216, 16, !spirv.Decorations !888		; visa id: 1558
  %1218 = bitcast i32 %1217 to float
  %1219 = zext i16 %1215 to i32		; visa id: 1559
  %1220 = shl nuw i32 %1219, 16, !spirv.Decorations !888		; visa id: 1560
  %1221 = bitcast i32 %1220 to float
  %1222 = fmul reassoc nsz arcp contract float %1218, %1221, !spirv.Decorations !881
  %1223 = fadd reassoc nsz arcp contract float %1222, %.sroa.222.1, !spirv.Decorations !881		; visa id: 1561
  br label %.preheader.7, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1562

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %1206
; BB102 :
  %.sroa.222.2 = phi float [ %1223, %1206 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %sink_sink_3876 = bitcast <2 x i32> %405 to i64		; visa id: 1563
  %sink_sink_3852 = shl i64 %sink_sink_3876, 1		; visa id: 1567
  %sink_3896 = add i64 %.in3822, %sink_sink_3852		; visa id: 1568
  br i1 %227, label %1224, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1569

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
; BB:
  br label %._crit_edge.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1224:                                             ; preds = %.preheader.7
; BB104 :
  %.sroa.256.0.insert.ext743 = zext i32 %647 to i64		; visa id: 1571
  %1225 = shl nuw nsw i64 %.sroa.256.0.insert.ext743, 1		; visa id: 1572
  %1226 = add i64 %sink_3908, %1225		; visa id: 1573
  %1227 = inttoptr i64 %1226 to i16 addrspace(4)*		; visa id: 1574
  %1228 = addrspacecast i16 addrspace(4)* %1227 to i16 addrspace(1)*		; visa id: 1574
  %1229 = load i16, i16 addrspace(1)* %1228, align 2		; visa id: 1575
  %1230 = add i64 %sink_3896, %1225		; visa id: 1577
  %1231 = inttoptr i64 %1230 to i16 addrspace(4)*		; visa id: 1578
  %1232 = addrspacecast i16 addrspace(4)* %1231 to i16 addrspace(1)*		; visa id: 1578
  %1233 = load i16, i16 addrspace(1)* %1232, align 2		; visa id: 1579
  %1234 = zext i16 %1229 to i32		; visa id: 1581
  %1235 = shl nuw i32 %1234, 16, !spirv.Decorations !888		; visa id: 1582
  %1236 = bitcast i32 %1235 to float
  %1237 = zext i16 %1233 to i32		; visa id: 1583
  %1238 = shl nuw i32 %1237, 16, !spirv.Decorations !888		; visa id: 1584
  %1239 = bitcast i32 %1238 to float
  %1240 = fmul reassoc nsz arcp contract float %1236, %1239, !spirv.Decorations !881
  %1241 = fadd reassoc nsz arcp contract float %1240, %.sroa.34.1, !spirv.Decorations !881		; visa id: 1585
  br label %._crit_edge.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1586

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %1224
; BB105 :
  %.sroa.34.2 = phi float [ %1241, %1224 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  br i1 %230, label %1242, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1587

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
; BB:
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1242:                                             ; preds = %._crit_edge.8
; BB107 :
  %.sroa.256.0.insert.ext748 = zext i32 %647 to i64		; visa id: 1589
  %1243 = shl nuw nsw i64 %.sroa.256.0.insert.ext748, 1		; visa id: 1590
  %1244 = add i64 %sink_3906, %1243		; visa id: 1591
  %1245 = inttoptr i64 %1244 to i16 addrspace(4)*		; visa id: 1592
  %1246 = addrspacecast i16 addrspace(4)* %1245 to i16 addrspace(1)*		; visa id: 1592
  %1247 = load i16, i16 addrspace(1)* %1246, align 2		; visa id: 1593
  %1248 = add i64 %sink_3896, %1243		; visa id: 1595
  %1249 = inttoptr i64 %1248 to i16 addrspace(4)*		; visa id: 1596
  %1250 = addrspacecast i16 addrspace(4)* %1249 to i16 addrspace(1)*		; visa id: 1596
  %1251 = load i16, i16 addrspace(1)* %1250, align 2		; visa id: 1597
  %1252 = zext i16 %1247 to i32		; visa id: 1599
  %1253 = shl nuw i32 %1252, 16, !spirv.Decorations !888		; visa id: 1600
  %1254 = bitcast i32 %1253 to float
  %1255 = zext i16 %1251 to i32		; visa id: 1601
  %1256 = shl nuw i32 %1255, 16, !spirv.Decorations !888		; visa id: 1602
  %1257 = bitcast i32 %1256 to float
  %1258 = fmul reassoc nsz arcp contract float %1254, %1257, !spirv.Decorations !881
  %1259 = fadd reassoc nsz arcp contract float %1258, %.sroa.98.1, !spirv.Decorations !881		; visa id: 1603
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1604

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %1242
; BB108 :
  %.sroa.98.2 = phi float [ %1259, %1242 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  br i1 %233, label %1260, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1605

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
; BB:
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1260:                                             ; preds = %._crit_edge.1.8
; BB110 :
  %.sroa.256.0.insert.ext753 = zext i32 %647 to i64		; visa id: 1607
  %1261 = shl nuw nsw i64 %.sroa.256.0.insert.ext753, 1		; visa id: 1608
  %1262 = add i64 %sink_3905, %1261		; visa id: 1609
  %1263 = inttoptr i64 %1262 to i16 addrspace(4)*		; visa id: 1610
  %1264 = addrspacecast i16 addrspace(4)* %1263 to i16 addrspace(1)*		; visa id: 1610
  %1265 = load i16, i16 addrspace(1)* %1264, align 2		; visa id: 1611
  %1266 = add i64 %sink_3896, %1261		; visa id: 1613
  %1267 = inttoptr i64 %1266 to i16 addrspace(4)*		; visa id: 1614
  %1268 = addrspacecast i16 addrspace(4)* %1267 to i16 addrspace(1)*		; visa id: 1614
  %1269 = load i16, i16 addrspace(1)* %1268, align 2		; visa id: 1615
  %1270 = zext i16 %1265 to i32		; visa id: 1617
  %1271 = shl nuw i32 %1270, 16, !spirv.Decorations !888		; visa id: 1618
  %1272 = bitcast i32 %1271 to float
  %1273 = zext i16 %1269 to i32		; visa id: 1619
  %1274 = shl nuw i32 %1273, 16, !spirv.Decorations !888		; visa id: 1620
  %1275 = bitcast i32 %1274 to float
  %1276 = fmul reassoc nsz arcp contract float %1272, %1275, !spirv.Decorations !881
  %1277 = fadd reassoc nsz arcp contract float %1276, %.sroa.162.1, !spirv.Decorations !881		; visa id: 1621
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1622

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %1260
; BB111 :
  %.sroa.162.2 = phi float [ %1277, %1260 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  br i1 %236, label %1278, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1623

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
; BB:
  br label %.preheader.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1278:                                             ; preds = %._crit_edge.2.8
; BB113 :
  %.sroa.256.0.insert.ext758 = zext i32 %647 to i64		; visa id: 1625
  %1279 = shl nuw nsw i64 %.sroa.256.0.insert.ext758, 1		; visa id: 1626
  %1280 = add i64 %sink_3904, %1279		; visa id: 1627
  %1281 = inttoptr i64 %1280 to i16 addrspace(4)*		; visa id: 1628
  %1282 = addrspacecast i16 addrspace(4)* %1281 to i16 addrspace(1)*		; visa id: 1628
  %1283 = load i16, i16 addrspace(1)* %1282, align 2		; visa id: 1629
  %1284 = add i64 %sink_3896, %1279		; visa id: 1631
  %1285 = inttoptr i64 %1284 to i16 addrspace(4)*		; visa id: 1632
  %1286 = addrspacecast i16 addrspace(4)* %1285 to i16 addrspace(1)*		; visa id: 1632
  %1287 = load i16, i16 addrspace(1)* %1286, align 2		; visa id: 1633
  %1288 = zext i16 %1283 to i32		; visa id: 1635
  %1289 = shl nuw i32 %1288, 16, !spirv.Decorations !888		; visa id: 1636
  %1290 = bitcast i32 %1289 to float
  %1291 = zext i16 %1287 to i32		; visa id: 1637
  %1292 = shl nuw i32 %1291, 16, !spirv.Decorations !888		; visa id: 1638
  %1293 = bitcast i32 %1292 to float
  %1294 = fmul reassoc nsz arcp contract float %1290, %1293, !spirv.Decorations !881
  %1295 = fadd reassoc nsz arcp contract float %1294, %.sroa.226.1, !spirv.Decorations !881		; visa id: 1639
  br label %.preheader.8, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1640

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %1278
; BB114 :
  %.sroa.226.2 = phi float [ %1295, %1278 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %sink_sink_3875 = bitcast <2 x i32> %411 to i64		; visa id: 1641
  %sink_sink_3851 = shl i64 %sink_sink_3875, 1		; visa id: 1645
  %sink_3895 = add i64 %.in3822, %sink_sink_3851		; visa id: 1646
  br i1 %240, label %1296, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1647

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
; BB:
  br label %._crit_edge.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1296:                                             ; preds = %.preheader.8
; BB116 :
  %.sroa.256.0.insert.ext763 = zext i32 %647 to i64		; visa id: 1649
  %1297 = shl nuw nsw i64 %.sroa.256.0.insert.ext763, 1		; visa id: 1650
  %1298 = add i64 %sink_3908, %1297		; visa id: 1651
  %1299 = inttoptr i64 %1298 to i16 addrspace(4)*		; visa id: 1652
  %1300 = addrspacecast i16 addrspace(4)* %1299 to i16 addrspace(1)*		; visa id: 1652
  %1301 = load i16, i16 addrspace(1)* %1300, align 2		; visa id: 1653
  %1302 = add i64 %sink_3895, %1297		; visa id: 1655
  %1303 = inttoptr i64 %1302 to i16 addrspace(4)*		; visa id: 1656
  %1304 = addrspacecast i16 addrspace(4)* %1303 to i16 addrspace(1)*		; visa id: 1656
  %1305 = load i16, i16 addrspace(1)* %1304, align 2		; visa id: 1657
  %1306 = zext i16 %1301 to i32		; visa id: 1659
  %1307 = shl nuw i32 %1306, 16, !spirv.Decorations !888		; visa id: 1660
  %1308 = bitcast i32 %1307 to float
  %1309 = zext i16 %1305 to i32		; visa id: 1661
  %1310 = shl nuw i32 %1309, 16, !spirv.Decorations !888		; visa id: 1662
  %1311 = bitcast i32 %1310 to float
  %1312 = fmul reassoc nsz arcp contract float %1308, %1311, !spirv.Decorations !881
  %1313 = fadd reassoc nsz arcp contract float %1312, %.sroa.38.1, !spirv.Decorations !881		; visa id: 1663
  br label %._crit_edge.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1664

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %1296
; BB117 :
  %.sroa.38.2 = phi float [ %1313, %1296 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  br i1 %243, label %1314, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1665

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
; BB:
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1314:                                             ; preds = %._crit_edge.9
; BB119 :
  %.sroa.256.0.insert.ext768 = zext i32 %647 to i64		; visa id: 1667
  %1315 = shl nuw nsw i64 %.sroa.256.0.insert.ext768, 1		; visa id: 1668
  %1316 = add i64 %sink_3906, %1315		; visa id: 1669
  %1317 = inttoptr i64 %1316 to i16 addrspace(4)*		; visa id: 1670
  %1318 = addrspacecast i16 addrspace(4)* %1317 to i16 addrspace(1)*		; visa id: 1670
  %1319 = load i16, i16 addrspace(1)* %1318, align 2		; visa id: 1671
  %1320 = add i64 %sink_3895, %1315		; visa id: 1673
  %1321 = inttoptr i64 %1320 to i16 addrspace(4)*		; visa id: 1674
  %1322 = addrspacecast i16 addrspace(4)* %1321 to i16 addrspace(1)*		; visa id: 1674
  %1323 = load i16, i16 addrspace(1)* %1322, align 2		; visa id: 1675
  %1324 = zext i16 %1319 to i32		; visa id: 1677
  %1325 = shl nuw i32 %1324, 16, !spirv.Decorations !888		; visa id: 1678
  %1326 = bitcast i32 %1325 to float
  %1327 = zext i16 %1323 to i32		; visa id: 1679
  %1328 = shl nuw i32 %1327, 16, !spirv.Decorations !888		; visa id: 1680
  %1329 = bitcast i32 %1328 to float
  %1330 = fmul reassoc nsz arcp contract float %1326, %1329, !spirv.Decorations !881
  %1331 = fadd reassoc nsz arcp contract float %1330, %.sroa.102.1, !spirv.Decorations !881		; visa id: 1681
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1682

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %1314
; BB120 :
  %.sroa.102.2 = phi float [ %1331, %1314 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  br i1 %246, label %1332, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1683

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
; BB:
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1332:                                             ; preds = %._crit_edge.1.9
; BB122 :
  %.sroa.256.0.insert.ext773 = zext i32 %647 to i64		; visa id: 1685
  %1333 = shl nuw nsw i64 %.sroa.256.0.insert.ext773, 1		; visa id: 1686
  %1334 = add i64 %sink_3905, %1333		; visa id: 1687
  %1335 = inttoptr i64 %1334 to i16 addrspace(4)*		; visa id: 1688
  %1336 = addrspacecast i16 addrspace(4)* %1335 to i16 addrspace(1)*		; visa id: 1688
  %1337 = load i16, i16 addrspace(1)* %1336, align 2		; visa id: 1689
  %1338 = add i64 %sink_3895, %1333		; visa id: 1691
  %1339 = inttoptr i64 %1338 to i16 addrspace(4)*		; visa id: 1692
  %1340 = addrspacecast i16 addrspace(4)* %1339 to i16 addrspace(1)*		; visa id: 1692
  %1341 = load i16, i16 addrspace(1)* %1340, align 2		; visa id: 1693
  %1342 = zext i16 %1337 to i32		; visa id: 1695
  %1343 = shl nuw i32 %1342, 16, !spirv.Decorations !888		; visa id: 1696
  %1344 = bitcast i32 %1343 to float
  %1345 = zext i16 %1341 to i32		; visa id: 1697
  %1346 = shl nuw i32 %1345, 16, !spirv.Decorations !888		; visa id: 1698
  %1347 = bitcast i32 %1346 to float
  %1348 = fmul reassoc nsz arcp contract float %1344, %1347, !spirv.Decorations !881
  %1349 = fadd reassoc nsz arcp contract float %1348, %.sroa.166.1, !spirv.Decorations !881		; visa id: 1699
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1700

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %1332
; BB123 :
  %.sroa.166.2 = phi float [ %1349, %1332 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  br i1 %249, label %1350, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1701

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
; BB:
  br label %.preheader.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1350:                                             ; preds = %._crit_edge.2.9
; BB125 :
  %.sroa.256.0.insert.ext778 = zext i32 %647 to i64		; visa id: 1703
  %1351 = shl nuw nsw i64 %.sroa.256.0.insert.ext778, 1		; visa id: 1704
  %1352 = add i64 %sink_3904, %1351		; visa id: 1705
  %1353 = inttoptr i64 %1352 to i16 addrspace(4)*		; visa id: 1706
  %1354 = addrspacecast i16 addrspace(4)* %1353 to i16 addrspace(1)*		; visa id: 1706
  %1355 = load i16, i16 addrspace(1)* %1354, align 2		; visa id: 1707
  %1356 = add i64 %sink_3895, %1351		; visa id: 1709
  %1357 = inttoptr i64 %1356 to i16 addrspace(4)*		; visa id: 1710
  %1358 = addrspacecast i16 addrspace(4)* %1357 to i16 addrspace(1)*		; visa id: 1710
  %1359 = load i16, i16 addrspace(1)* %1358, align 2		; visa id: 1711
  %1360 = zext i16 %1355 to i32		; visa id: 1713
  %1361 = shl nuw i32 %1360, 16, !spirv.Decorations !888		; visa id: 1714
  %1362 = bitcast i32 %1361 to float
  %1363 = zext i16 %1359 to i32		; visa id: 1715
  %1364 = shl nuw i32 %1363, 16, !spirv.Decorations !888		; visa id: 1716
  %1365 = bitcast i32 %1364 to float
  %1366 = fmul reassoc nsz arcp contract float %1362, %1365, !spirv.Decorations !881
  %1367 = fadd reassoc nsz arcp contract float %1366, %.sroa.230.1, !spirv.Decorations !881		; visa id: 1717
  br label %.preheader.9, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1718

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %1350
; BB126 :
  %.sroa.230.2 = phi float [ %1367, %1350 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %sink_sink_3874 = bitcast <2 x i32> %417 to i64		; visa id: 1719
  %sink_sink_3850 = shl i64 %sink_sink_3874, 1		; visa id: 1723
  %sink_3894 = add i64 %.in3822, %sink_sink_3850		; visa id: 1724
  br i1 %253, label %1368, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1725

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
; BB:
  br label %._crit_edge.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1368:                                             ; preds = %.preheader.9
; BB128 :
  %.sroa.256.0.insert.ext783 = zext i32 %647 to i64		; visa id: 1727
  %1369 = shl nuw nsw i64 %.sroa.256.0.insert.ext783, 1		; visa id: 1728
  %1370 = add i64 %sink_3908, %1369		; visa id: 1729
  %1371 = inttoptr i64 %1370 to i16 addrspace(4)*		; visa id: 1730
  %1372 = addrspacecast i16 addrspace(4)* %1371 to i16 addrspace(1)*		; visa id: 1730
  %1373 = load i16, i16 addrspace(1)* %1372, align 2		; visa id: 1731
  %1374 = add i64 %sink_3894, %1369		; visa id: 1733
  %1375 = inttoptr i64 %1374 to i16 addrspace(4)*		; visa id: 1734
  %1376 = addrspacecast i16 addrspace(4)* %1375 to i16 addrspace(1)*		; visa id: 1734
  %1377 = load i16, i16 addrspace(1)* %1376, align 2		; visa id: 1735
  %1378 = zext i16 %1373 to i32		; visa id: 1737
  %1379 = shl nuw i32 %1378, 16, !spirv.Decorations !888		; visa id: 1738
  %1380 = bitcast i32 %1379 to float
  %1381 = zext i16 %1377 to i32		; visa id: 1739
  %1382 = shl nuw i32 %1381, 16, !spirv.Decorations !888		; visa id: 1740
  %1383 = bitcast i32 %1382 to float
  %1384 = fmul reassoc nsz arcp contract float %1380, %1383, !spirv.Decorations !881
  %1385 = fadd reassoc nsz arcp contract float %1384, %.sroa.42.1, !spirv.Decorations !881		; visa id: 1741
  br label %._crit_edge.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1742

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %1368
; BB129 :
  %.sroa.42.2 = phi float [ %1385, %1368 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  br i1 %256, label %1386, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1743

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
; BB:
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1386:                                             ; preds = %._crit_edge.10
; BB131 :
  %.sroa.256.0.insert.ext788 = zext i32 %647 to i64		; visa id: 1745
  %1387 = shl nuw nsw i64 %.sroa.256.0.insert.ext788, 1		; visa id: 1746
  %1388 = add i64 %sink_3906, %1387		; visa id: 1747
  %1389 = inttoptr i64 %1388 to i16 addrspace(4)*		; visa id: 1748
  %1390 = addrspacecast i16 addrspace(4)* %1389 to i16 addrspace(1)*		; visa id: 1748
  %1391 = load i16, i16 addrspace(1)* %1390, align 2		; visa id: 1749
  %1392 = add i64 %sink_3894, %1387		; visa id: 1751
  %1393 = inttoptr i64 %1392 to i16 addrspace(4)*		; visa id: 1752
  %1394 = addrspacecast i16 addrspace(4)* %1393 to i16 addrspace(1)*		; visa id: 1752
  %1395 = load i16, i16 addrspace(1)* %1394, align 2		; visa id: 1753
  %1396 = zext i16 %1391 to i32		; visa id: 1755
  %1397 = shl nuw i32 %1396, 16, !spirv.Decorations !888		; visa id: 1756
  %1398 = bitcast i32 %1397 to float
  %1399 = zext i16 %1395 to i32		; visa id: 1757
  %1400 = shl nuw i32 %1399, 16, !spirv.Decorations !888		; visa id: 1758
  %1401 = bitcast i32 %1400 to float
  %1402 = fmul reassoc nsz arcp contract float %1398, %1401, !spirv.Decorations !881
  %1403 = fadd reassoc nsz arcp contract float %1402, %.sroa.106.1, !spirv.Decorations !881		; visa id: 1759
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1760

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %1386
; BB132 :
  %.sroa.106.2 = phi float [ %1403, %1386 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  br i1 %259, label %1404, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1761

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
; BB:
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1404:                                             ; preds = %._crit_edge.1.10
; BB134 :
  %.sroa.256.0.insert.ext793 = zext i32 %647 to i64		; visa id: 1763
  %1405 = shl nuw nsw i64 %.sroa.256.0.insert.ext793, 1		; visa id: 1764
  %1406 = add i64 %sink_3905, %1405		; visa id: 1765
  %1407 = inttoptr i64 %1406 to i16 addrspace(4)*		; visa id: 1766
  %1408 = addrspacecast i16 addrspace(4)* %1407 to i16 addrspace(1)*		; visa id: 1766
  %1409 = load i16, i16 addrspace(1)* %1408, align 2		; visa id: 1767
  %1410 = add i64 %sink_3894, %1405		; visa id: 1769
  %1411 = inttoptr i64 %1410 to i16 addrspace(4)*		; visa id: 1770
  %1412 = addrspacecast i16 addrspace(4)* %1411 to i16 addrspace(1)*		; visa id: 1770
  %1413 = load i16, i16 addrspace(1)* %1412, align 2		; visa id: 1771
  %1414 = zext i16 %1409 to i32		; visa id: 1773
  %1415 = shl nuw i32 %1414, 16, !spirv.Decorations !888		; visa id: 1774
  %1416 = bitcast i32 %1415 to float
  %1417 = zext i16 %1413 to i32		; visa id: 1775
  %1418 = shl nuw i32 %1417, 16, !spirv.Decorations !888		; visa id: 1776
  %1419 = bitcast i32 %1418 to float
  %1420 = fmul reassoc nsz arcp contract float %1416, %1419, !spirv.Decorations !881
  %1421 = fadd reassoc nsz arcp contract float %1420, %.sroa.170.1, !spirv.Decorations !881		; visa id: 1777
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1778

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %1404
; BB135 :
  %.sroa.170.2 = phi float [ %1421, %1404 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  br i1 %262, label %1422, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1779

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
; BB:
  br label %.preheader.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1422:                                             ; preds = %._crit_edge.2.10
; BB137 :
  %.sroa.256.0.insert.ext798 = zext i32 %647 to i64		; visa id: 1781
  %1423 = shl nuw nsw i64 %.sroa.256.0.insert.ext798, 1		; visa id: 1782
  %1424 = add i64 %sink_3904, %1423		; visa id: 1783
  %1425 = inttoptr i64 %1424 to i16 addrspace(4)*		; visa id: 1784
  %1426 = addrspacecast i16 addrspace(4)* %1425 to i16 addrspace(1)*		; visa id: 1784
  %1427 = load i16, i16 addrspace(1)* %1426, align 2		; visa id: 1785
  %1428 = add i64 %sink_3894, %1423		; visa id: 1787
  %1429 = inttoptr i64 %1428 to i16 addrspace(4)*		; visa id: 1788
  %1430 = addrspacecast i16 addrspace(4)* %1429 to i16 addrspace(1)*		; visa id: 1788
  %1431 = load i16, i16 addrspace(1)* %1430, align 2		; visa id: 1789
  %1432 = zext i16 %1427 to i32		; visa id: 1791
  %1433 = shl nuw i32 %1432, 16, !spirv.Decorations !888		; visa id: 1792
  %1434 = bitcast i32 %1433 to float
  %1435 = zext i16 %1431 to i32		; visa id: 1793
  %1436 = shl nuw i32 %1435, 16, !spirv.Decorations !888		; visa id: 1794
  %1437 = bitcast i32 %1436 to float
  %1438 = fmul reassoc nsz arcp contract float %1434, %1437, !spirv.Decorations !881
  %1439 = fadd reassoc nsz arcp contract float %1438, %.sroa.234.1, !spirv.Decorations !881		; visa id: 1795
  br label %.preheader.10, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1796

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %1422
; BB138 :
  %.sroa.234.2 = phi float [ %1439, %1422 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %sink_sink_3873 = bitcast <2 x i32> %423 to i64		; visa id: 1797
  %sink_sink_3849 = shl i64 %sink_sink_3873, 1		; visa id: 1801
  %sink_3893 = add i64 %.in3822, %sink_sink_3849		; visa id: 1802
  br i1 %266, label %1440, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1803

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
; BB:
  br label %._crit_edge.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1440:                                             ; preds = %.preheader.10
; BB140 :
  %.sroa.256.0.insert.ext803 = zext i32 %647 to i64		; visa id: 1805
  %1441 = shl nuw nsw i64 %.sroa.256.0.insert.ext803, 1		; visa id: 1806
  %1442 = add i64 %sink_3908, %1441		; visa id: 1807
  %1443 = inttoptr i64 %1442 to i16 addrspace(4)*		; visa id: 1808
  %1444 = addrspacecast i16 addrspace(4)* %1443 to i16 addrspace(1)*		; visa id: 1808
  %1445 = load i16, i16 addrspace(1)* %1444, align 2		; visa id: 1809
  %1446 = add i64 %sink_3893, %1441		; visa id: 1811
  %1447 = inttoptr i64 %1446 to i16 addrspace(4)*		; visa id: 1812
  %1448 = addrspacecast i16 addrspace(4)* %1447 to i16 addrspace(1)*		; visa id: 1812
  %1449 = load i16, i16 addrspace(1)* %1448, align 2		; visa id: 1813
  %1450 = zext i16 %1445 to i32		; visa id: 1815
  %1451 = shl nuw i32 %1450, 16, !spirv.Decorations !888		; visa id: 1816
  %1452 = bitcast i32 %1451 to float
  %1453 = zext i16 %1449 to i32		; visa id: 1817
  %1454 = shl nuw i32 %1453, 16, !spirv.Decorations !888		; visa id: 1818
  %1455 = bitcast i32 %1454 to float
  %1456 = fmul reassoc nsz arcp contract float %1452, %1455, !spirv.Decorations !881
  %1457 = fadd reassoc nsz arcp contract float %1456, %.sroa.46.1, !spirv.Decorations !881		; visa id: 1819
  br label %._crit_edge.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1820

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %1440
; BB141 :
  %.sroa.46.2 = phi float [ %1457, %1440 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  br i1 %269, label %1458, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1821

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
; BB:
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1458:                                             ; preds = %._crit_edge.11
; BB143 :
  %.sroa.256.0.insert.ext808 = zext i32 %647 to i64		; visa id: 1823
  %1459 = shl nuw nsw i64 %.sroa.256.0.insert.ext808, 1		; visa id: 1824
  %1460 = add i64 %sink_3906, %1459		; visa id: 1825
  %1461 = inttoptr i64 %1460 to i16 addrspace(4)*		; visa id: 1826
  %1462 = addrspacecast i16 addrspace(4)* %1461 to i16 addrspace(1)*		; visa id: 1826
  %1463 = load i16, i16 addrspace(1)* %1462, align 2		; visa id: 1827
  %1464 = add i64 %sink_3893, %1459		; visa id: 1829
  %1465 = inttoptr i64 %1464 to i16 addrspace(4)*		; visa id: 1830
  %1466 = addrspacecast i16 addrspace(4)* %1465 to i16 addrspace(1)*		; visa id: 1830
  %1467 = load i16, i16 addrspace(1)* %1466, align 2		; visa id: 1831
  %1468 = zext i16 %1463 to i32		; visa id: 1833
  %1469 = shl nuw i32 %1468, 16, !spirv.Decorations !888		; visa id: 1834
  %1470 = bitcast i32 %1469 to float
  %1471 = zext i16 %1467 to i32		; visa id: 1835
  %1472 = shl nuw i32 %1471, 16, !spirv.Decorations !888		; visa id: 1836
  %1473 = bitcast i32 %1472 to float
  %1474 = fmul reassoc nsz arcp contract float %1470, %1473, !spirv.Decorations !881
  %1475 = fadd reassoc nsz arcp contract float %1474, %.sroa.110.1, !spirv.Decorations !881		; visa id: 1837
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1838

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %1458
; BB144 :
  %.sroa.110.2 = phi float [ %1475, %1458 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  br i1 %272, label %1476, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1839

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
; BB:
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1476:                                             ; preds = %._crit_edge.1.11
; BB146 :
  %.sroa.256.0.insert.ext813 = zext i32 %647 to i64		; visa id: 1841
  %1477 = shl nuw nsw i64 %.sroa.256.0.insert.ext813, 1		; visa id: 1842
  %1478 = add i64 %sink_3905, %1477		; visa id: 1843
  %1479 = inttoptr i64 %1478 to i16 addrspace(4)*		; visa id: 1844
  %1480 = addrspacecast i16 addrspace(4)* %1479 to i16 addrspace(1)*		; visa id: 1844
  %1481 = load i16, i16 addrspace(1)* %1480, align 2		; visa id: 1845
  %1482 = add i64 %sink_3893, %1477		; visa id: 1847
  %1483 = inttoptr i64 %1482 to i16 addrspace(4)*		; visa id: 1848
  %1484 = addrspacecast i16 addrspace(4)* %1483 to i16 addrspace(1)*		; visa id: 1848
  %1485 = load i16, i16 addrspace(1)* %1484, align 2		; visa id: 1849
  %1486 = zext i16 %1481 to i32		; visa id: 1851
  %1487 = shl nuw i32 %1486, 16, !spirv.Decorations !888		; visa id: 1852
  %1488 = bitcast i32 %1487 to float
  %1489 = zext i16 %1485 to i32		; visa id: 1853
  %1490 = shl nuw i32 %1489, 16, !spirv.Decorations !888		; visa id: 1854
  %1491 = bitcast i32 %1490 to float
  %1492 = fmul reassoc nsz arcp contract float %1488, %1491, !spirv.Decorations !881
  %1493 = fadd reassoc nsz arcp contract float %1492, %.sroa.174.1, !spirv.Decorations !881		; visa id: 1855
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1856

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %1476
; BB147 :
  %.sroa.174.2 = phi float [ %1493, %1476 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  br i1 %275, label %1494, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1857

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
; BB:
  br label %.preheader.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1494:                                             ; preds = %._crit_edge.2.11
; BB149 :
  %.sroa.256.0.insert.ext818 = zext i32 %647 to i64		; visa id: 1859
  %1495 = shl nuw nsw i64 %.sroa.256.0.insert.ext818, 1		; visa id: 1860
  %1496 = add i64 %sink_3904, %1495		; visa id: 1861
  %1497 = inttoptr i64 %1496 to i16 addrspace(4)*		; visa id: 1862
  %1498 = addrspacecast i16 addrspace(4)* %1497 to i16 addrspace(1)*		; visa id: 1862
  %1499 = load i16, i16 addrspace(1)* %1498, align 2		; visa id: 1863
  %1500 = add i64 %sink_3893, %1495		; visa id: 1865
  %1501 = inttoptr i64 %1500 to i16 addrspace(4)*		; visa id: 1866
  %1502 = addrspacecast i16 addrspace(4)* %1501 to i16 addrspace(1)*		; visa id: 1866
  %1503 = load i16, i16 addrspace(1)* %1502, align 2		; visa id: 1867
  %1504 = zext i16 %1499 to i32		; visa id: 1869
  %1505 = shl nuw i32 %1504, 16, !spirv.Decorations !888		; visa id: 1870
  %1506 = bitcast i32 %1505 to float
  %1507 = zext i16 %1503 to i32		; visa id: 1871
  %1508 = shl nuw i32 %1507, 16, !spirv.Decorations !888		; visa id: 1872
  %1509 = bitcast i32 %1508 to float
  %1510 = fmul reassoc nsz arcp contract float %1506, %1509, !spirv.Decorations !881
  %1511 = fadd reassoc nsz arcp contract float %1510, %.sroa.238.1, !spirv.Decorations !881		; visa id: 1873
  br label %.preheader.11, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1874

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %1494
; BB150 :
  %.sroa.238.2 = phi float [ %1511, %1494 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %sink_sink_3872 = bitcast <2 x i32> %429 to i64		; visa id: 1875
  %sink_sink_3848 = shl i64 %sink_sink_3872, 1		; visa id: 1879
  %sink_3892 = add i64 %.in3822, %sink_sink_3848		; visa id: 1880
  br i1 %279, label %1512, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1881

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
; BB:
  br label %._crit_edge.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1512:                                             ; preds = %.preheader.11
; BB152 :
  %.sroa.256.0.insert.ext823 = zext i32 %647 to i64		; visa id: 1883
  %1513 = shl nuw nsw i64 %.sroa.256.0.insert.ext823, 1		; visa id: 1884
  %1514 = add i64 %sink_3908, %1513		; visa id: 1885
  %1515 = inttoptr i64 %1514 to i16 addrspace(4)*		; visa id: 1886
  %1516 = addrspacecast i16 addrspace(4)* %1515 to i16 addrspace(1)*		; visa id: 1886
  %1517 = load i16, i16 addrspace(1)* %1516, align 2		; visa id: 1887
  %1518 = add i64 %sink_3892, %1513		; visa id: 1889
  %1519 = inttoptr i64 %1518 to i16 addrspace(4)*		; visa id: 1890
  %1520 = addrspacecast i16 addrspace(4)* %1519 to i16 addrspace(1)*		; visa id: 1890
  %1521 = load i16, i16 addrspace(1)* %1520, align 2		; visa id: 1891
  %1522 = zext i16 %1517 to i32		; visa id: 1893
  %1523 = shl nuw i32 %1522, 16, !spirv.Decorations !888		; visa id: 1894
  %1524 = bitcast i32 %1523 to float
  %1525 = zext i16 %1521 to i32		; visa id: 1895
  %1526 = shl nuw i32 %1525, 16, !spirv.Decorations !888		; visa id: 1896
  %1527 = bitcast i32 %1526 to float
  %1528 = fmul reassoc nsz arcp contract float %1524, %1527, !spirv.Decorations !881
  %1529 = fadd reassoc nsz arcp contract float %1528, %.sroa.50.1, !spirv.Decorations !881		; visa id: 1897
  br label %._crit_edge.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1898

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %1512
; BB153 :
  %.sroa.50.2 = phi float [ %1529, %1512 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  br i1 %282, label %1530, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1899

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
; BB:
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1530:                                             ; preds = %._crit_edge.12
; BB155 :
  %.sroa.256.0.insert.ext828 = zext i32 %647 to i64		; visa id: 1901
  %1531 = shl nuw nsw i64 %.sroa.256.0.insert.ext828, 1		; visa id: 1902
  %1532 = add i64 %sink_3906, %1531		; visa id: 1903
  %1533 = inttoptr i64 %1532 to i16 addrspace(4)*		; visa id: 1904
  %1534 = addrspacecast i16 addrspace(4)* %1533 to i16 addrspace(1)*		; visa id: 1904
  %1535 = load i16, i16 addrspace(1)* %1534, align 2		; visa id: 1905
  %1536 = add i64 %sink_3892, %1531		; visa id: 1907
  %1537 = inttoptr i64 %1536 to i16 addrspace(4)*		; visa id: 1908
  %1538 = addrspacecast i16 addrspace(4)* %1537 to i16 addrspace(1)*		; visa id: 1908
  %1539 = load i16, i16 addrspace(1)* %1538, align 2		; visa id: 1909
  %1540 = zext i16 %1535 to i32		; visa id: 1911
  %1541 = shl nuw i32 %1540, 16, !spirv.Decorations !888		; visa id: 1912
  %1542 = bitcast i32 %1541 to float
  %1543 = zext i16 %1539 to i32		; visa id: 1913
  %1544 = shl nuw i32 %1543, 16, !spirv.Decorations !888		; visa id: 1914
  %1545 = bitcast i32 %1544 to float
  %1546 = fmul reassoc nsz arcp contract float %1542, %1545, !spirv.Decorations !881
  %1547 = fadd reassoc nsz arcp contract float %1546, %.sroa.114.1, !spirv.Decorations !881		; visa id: 1915
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1916

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %1530
; BB156 :
  %.sroa.114.2 = phi float [ %1547, %1530 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  br i1 %285, label %1548, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1917

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
; BB:
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1548:                                             ; preds = %._crit_edge.1.12
; BB158 :
  %.sroa.256.0.insert.ext833 = zext i32 %647 to i64		; visa id: 1919
  %1549 = shl nuw nsw i64 %.sroa.256.0.insert.ext833, 1		; visa id: 1920
  %1550 = add i64 %sink_3905, %1549		; visa id: 1921
  %1551 = inttoptr i64 %1550 to i16 addrspace(4)*		; visa id: 1922
  %1552 = addrspacecast i16 addrspace(4)* %1551 to i16 addrspace(1)*		; visa id: 1922
  %1553 = load i16, i16 addrspace(1)* %1552, align 2		; visa id: 1923
  %1554 = add i64 %sink_3892, %1549		; visa id: 1925
  %1555 = inttoptr i64 %1554 to i16 addrspace(4)*		; visa id: 1926
  %1556 = addrspacecast i16 addrspace(4)* %1555 to i16 addrspace(1)*		; visa id: 1926
  %1557 = load i16, i16 addrspace(1)* %1556, align 2		; visa id: 1927
  %1558 = zext i16 %1553 to i32		; visa id: 1929
  %1559 = shl nuw i32 %1558, 16, !spirv.Decorations !888		; visa id: 1930
  %1560 = bitcast i32 %1559 to float
  %1561 = zext i16 %1557 to i32		; visa id: 1931
  %1562 = shl nuw i32 %1561, 16, !spirv.Decorations !888		; visa id: 1932
  %1563 = bitcast i32 %1562 to float
  %1564 = fmul reassoc nsz arcp contract float %1560, %1563, !spirv.Decorations !881
  %1565 = fadd reassoc nsz arcp contract float %1564, %.sroa.178.1, !spirv.Decorations !881		; visa id: 1933
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1934

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %1548
; BB159 :
  %.sroa.178.2 = phi float [ %1565, %1548 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  br i1 %288, label %1566, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1935

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
; BB:
  br label %.preheader.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1566:                                             ; preds = %._crit_edge.2.12
; BB161 :
  %.sroa.256.0.insert.ext838 = zext i32 %647 to i64		; visa id: 1937
  %1567 = shl nuw nsw i64 %.sroa.256.0.insert.ext838, 1		; visa id: 1938
  %1568 = add i64 %sink_3904, %1567		; visa id: 1939
  %1569 = inttoptr i64 %1568 to i16 addrspace(4)*		; visa id: 1940
  %1570 = addrspacecast i16 addrspace(4)* %1569 to i16 addrspace(1)*		; visa id: 1940
  %1571 = load i16, i16 addrspace(1)* %1570, align 2		; visa id: 1941
  %1572 = add i64 %sink_3892, %1567		; visa id: 1943
  %1573 = inttoptr i64 %1572 to i16 addrspace(4)*		; visa id: 1944
  %1574 = addrspacecast i16 addrspace(4)* %1573 to i16 addrspace(1)*		; visa id: 1944
  %1575 = load i16, i16 addrspace(1)* %1574, align 2		; visa id: 1945
  %1576 = zext i16 %1571 to i32		; visa id: 1947
  %1577 = shl nuw i32 %1576, 16, !spirv.Decorations !888		; visa id: 1948
  %1578 = bitcast i32 %1577 to float
  %1579 = zext i16 %1575 to i32		; visa id: 1949
  %1580 = shl nuw i32 %1579, 16, !spirv.Decorations !888		; visa id: 1950
  %1581 = bitcast i32 %1580 to float
  %1582 = fmul reassoc nsz arcp contract float %1578, %1581, !spirv.Decorations !881
  %1583 = fadd reassoc nsz arcp contract float %1582, %.sroa.242.1, !spirv.Decorations !881		; visa id: 1951
  br label %.preheader.12, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1952

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %1566
; BB162 :
  %.sroa.242.2 = phi float [ %1583, %1566 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %sink_sink_3871 = bitcast <2 x i32> %435 to i64		; visa id: 1953
  %sink_sink_3847 = shl i64 %sink_sink_3871, 1		; visa id: 1957
  %sink_3891 = add i64 %.in3822, %sink_sink_3847		; visa id: 1958
  br i1 %292, label %1584, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1959

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
; BB:
  br label %._crit_edge.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1584:                                             ; preds = %.preheader.12
; BB164 :
  %.sroa.256.0.insert.ext843 = zext i32 %647 to i64		; visa id: 1961
  %1585 = shl nuw nsw i64 %.sroa.256.0.insert.ext843, 1		; visa id: 1962
  %1586 = add i64 %sink_3908, %1585		; visa id: 1963
  %1587 = inttoptr i64 %1586 to i16 addrspace(4)*		; visa id: 1964
  %1588 = addrspacecast i16 addrspace(4)* %1587 to i16 addrspace(1)*		; visa id: 1964
  %1589 = load i16, i16 addrspace(1)* %1588, align 2		; visa id: 1965
  %1590 = add i64 %sink_3891, %1585		; visa id: 1967
  %1591 = inttoptr i64 %1590 to i16 addrspace(4)*		; visa id: 1968
  %1592 = addrspacecast i16 addrspace(4)* %1591 to i16 addrspace(1)*		; visa id: 1968
  %1593 = load i16, i16 addrspace(1)* %1592, align 2		; visa id: 1969
  %1594 = zext i16 %1589 to i32		; visa id: 1971
  %1595 = shl nuw i32 %1594, 16, !spirv.Decorations !888		; visa id: 1972
  %1596 = bitcast i32 %1595 to float
  %1597 = zext i16 %1593 to i32		; visa id: 1973
  %1598 = shl nuw i32 %1597, 16, !spirv.Decorations !888		; visa id: 1974
  %1599 = bitcast i32 %1598 to float
  %1600 = fmul reassoc nsz arcp contract float %1596, %1599, !spirv.Decorations !881
  %1601 = fadd reassoc nsz arcp contract float %1600, %.sroa.54.1, !spirv.Decorations !881		; visa id: 1975
  br label %._crit_edge.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1976

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %1584
; BB165 :
  %.sroa.54.2 = phi float [ %1601, %1584 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  br i1 %295, label %1602, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1977

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
; BB:
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1602:                                             ; preds = %._crit_edge.13
; BB167 :
  %.sroa.256.0.insert.ext848 = zext i32 %647 to i64		; visa id: 1979
  %1603 = shl nuw nsw i64 %.sroa.256.0.insert.ext848, 1		; visa id: 1980
  %1604 = add i64 %sink_3906, %1603		; visa id: 1981
  %1605 = inttoptr i64 %1604 to i16 addrspace(4)*		; visa id: 1982
  %1606 = addrspacecast i16 addrspace(4)* %1605 to i16 addrspace(1)*		; visa id: 1982
  %1607 = load i16, i16 addrspace(1)* %1606, align 2		; visa id: 1983
  %1608 = add i64 %sink_3891, %1603		; visa id: 1985
  %1609 = inttoptr i64 %1608 to i16 addrspace(4)*		; visa id: 1986
  %1610 = addrspacecast i16 addrspace(4)* %1609 to i16 addrspace(1)*		; visa id: 1986
  %1611 = load i16, i16 addrspace(1)* %1610, align 2		; visa id: 1987
  %1612 = zext i16 %1607 to i32		; visa id: 1989
  %1613 = shl nuw i32 %1612, 16, !spirv.Decorations !888		; visa id: 1990
  %1614 = bitcast i32 %1613 to float
  %1615 = zext i16 %1611 to i32		; visa id: 1991
  %1616 = shl nuw i32 %1615, 16, !spirv.Decorations !888		; visa id: 1992
  %1617 = bitcast i32 %1616 to float
  %1618 = fmul reassoc nsz arcp contract float %1614, %1617, !spirv.Decorations !881
  %1619 = fadd reassoc nsz arcp contract float %1618, %.sroa.118.1, !spirv.Decorations !881		; visa id: 1993
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 1994

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %1602
; BB168 :
  %.sroa.118.2 = phi float [ %1619, %1602 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  br i1 %298, label %1620, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 1995

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
; BB:
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1620:                                             ; preds = %._crit_edge.1.13
; BB170 :
  %.sroa.256.0.insert.ext853 = zext i32 %647 to i64		; visa id: 1997
  %1621 = shl nuw nsw i64 %.sroa.256.0.insert.ext853, 1		; visa id: 1998
  %1622 = add i64 %sink_3905, %1621		; visa id: 1999
  %1623 = inttoptr i64 %1622 to i16 addrspace(4)*		; visa id: 2000
  %1624 = addrspacecast i16 addrspace(4)* %1623 to i16 addrspace(1)*		; visa id: 2000
  %1625 = load i16, i16 addrspace(1)* %1624, align 2		; visa id: 2001
  %1626 = add i64 %sink_3891, %1621		; visa id: 2003
  %1627 = inttoptr i64 %1626 to i16 addrspace(4)*		; visa id: 2004
  %1628 = addrspacecast i16 addrspace(4)* %1627 to i16 addrspace(1)*		; visa id: 2004
  %1629 = load i16, i16 addrspace(1)* %1628, align 2		; visa id: 2005
  %1630 = zext i16 %1625 to i32		; visa id: 2007
  %1631 = shl nuw i32 %1630, 16, !spirv.Decorations !888		; visa id: 2008
  %1632 = bitcast i32 %1631 to float
  %1633 = zext i16 %1629 to i32		; visa id: 2009
  %1634 = shl nuw i32 %1633, 16, !spirv.Decorations !888		; visa id: 2010
  %1635 = bitcast i32 %1634 to float
  %1636 = fmul reassoc nsz arcp contract float %1632, %1635, !spirv.Decorations !881
  %1637 = fadd reassoc nsz arcp contract float %1636, %.sroa.182.1, !spirv.Decorations !881		; visa id: 2011
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2012

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %1620
; BB171 :
  %.sroa.182.2 = phi float [ %1637, %1620 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  br i1 %301, label %1638, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2013

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
; BB:
  br label %.preheader.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1638:                                             ; preds = %._crit_edge.2.13
; BB173 :
  %.sroa.256.0.insert.ext858 = zext i32 %647 to i64		; visa id: 2015
  %1639 = shl nuw nsw i64 %.sroa.256.0.insert.ext858, 1		; visa id: 2016
  %1640 = add i64 %sink_3904, %1639		; visa id: 2017
  %1641 = inttoptr i64 %1640 to i16 addrspace(4)*		; visa id: 2018
  %1642 = addrspacecast i16 addrspace(4)* %1641 to i16 addrspace(1)*		; visa id: 2018
  %1643 = load i16, i16 addrspace(1)* %1642, align 2		; visa id: 2019
  %1644 = add i64 %sink_3891, %1639		; visa id: 2021
  %1645 = inttoptr i64 %1644 to i16 addrspace(4)*		; visa id: 2022
  %1646 = addrspacecast i16 addrspace(4)* %1645 to i16 addrspace(1)*		; visa id: 2022
  %1647 = load i16, i16 addrspace(1)* %1646, align 2		; visa id: 2023
  %1648 = zext i16 %1643 to i32		; visa id: 2025
  %1649 = shl nuw i32 %1648, 16, !spirv.Decorations !888		; visa id: 2026
  %1650 = bitcast i32 %1649 to float
  %1651 = zext i16 %1647 to i32		; visa id: 2027
  %1652 = shl nuw i32 %1651, 16, !spirv.Decorations !888		; visa id: 2028
  %1653 = bitcast i32 %1652 to float
  %1654 = fmul reassoc nsz arcp contract float %1650, %1653, !spirv.Decorations !881
  %1655 = fadd reassoc nsz arcp contract float %1654, %.sroa.246.1, !spirv.Decorations !881		; visa id: 2029
  br label %.preheader.13, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2030

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %1638
; BB174 :
  %.sroa.246.2 = phi float [ %1655, %1638 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %sink_sink_3870 = bitcast <2 x i32> %441 to i64		; visa id: 2031
  %sink_sink_3846 = shl i64 %sink_sink_3870, 1		; visa id: 2035
  %sink_3890 = add i64 %.in3822, %sink_sink_3846		; visa id: 2036
  br i1 %305, label %1656, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2037

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
; BB:
  br label %._crit_edge.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1656:                                             ; preds = %.preheader.13
; BB176 :
  %.sroa.256.0.insert.ext863 = zext i32 %647 to i64		; visa id: 2039
  %1657 = shl nuw nsw i64 %.sroa.256.0.insert.ext863, 1		; visa id: 2040
  %1658 = add i64 %sink_3908, %1657		; visa id: 2041
  %1659 = inttoptr i64 %1658 to i16 addrspace(4)*		; visa id: 2042
  %1660 = addrspacecast i16 addrspace(4)* %1659 to i16 addrspace(1)*		; visa id: 2042
  %1661 = load i16, i16 addrspace(1)* %1660, align 2		; visa id: 2043
  %1662 = add i64 %sink_3890, %1657		; visa id: 2045
  %1663 = inttoptr i64 %1662 to i16 addrspace(4)*		; visa id: 2046
  %1664 = addrspacecast i16 addrspace(4)* %1663 to i16 addrspace(1)*		; visa id: 2046
  %1665 = load i16, i16 addrspace(1)* %1664, align 2		; visa id: 2047
  %1666 = zext i16 %1661 to i32		; visa id: 2049
  %1667 = shl nuw i32 %1666, 16, !spirv.Decorations !888		; visa id: 2050
  %1668 = bitcast i32 %1667 to float
  %1669 = zext i16 %1665 to i32		; visa id: 2051
  %1670 = shl nuw i32 %1669, 16, !spirv.Decorations !888		; visa id: 2052
  %1671 = bitcast i32 %1670 to float
  %1672 = fmul reassoc nsz arcp contract float %1668, %1671, !spirv.Decorations !881
  %1673 = fadd reassoc nsz arcp contract float %1672, %.sroa.58.1, !spirv.Decorations !881		; visa id: 2053
  br label %._crit_edge.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2054

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %1656
; BB177 :
  %.sroa.58.2 = phi float [ %1673, %1656 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  br i1 %308, label %1674, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2055

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
; BB:
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1674:                                             ; preds = %._crit_edge.14
; BB179 :
  %.sroa.256.0.insert.ext868 = zext i32 %647 to i64		; visa id: 2057
  %1675 = shl nuw nsw i64 %.sroa.256.0.insert.ext868, 1		; visa id: 2058
  %1676 = add i64 %sink_3906, %1675		; visa id: 2059
  %1677 = inttoptr i64 %1676 to i16 addrspace(4)*		; visa id: 2060
  %1678 = addrspacecast i16 addrspace(4)* %1677 to i16 addrspace(1)*		; visa id: 2060
  %1679 = load i16, i16 addrspace(1)* %1678, align 2		; visa id: 2061
  %1680 = add i64 %sink_3890, %1675		; visa id: 2063
  %1681 = inttoptr i64 %1680 to i16 addrspace(4)*		; visa id: 2064
  %1682 = addrspacecast i16 addrspace(4)* %1681 to i16 addrspace(1)*		; visa id: 2064
  %1683 = load i16, i16 addrspace(1)* %1682, align 2		; visa id: 2065
  %1684 = zext i16 %1679 to i32		; visa id: 2067
  %1685 = shl nuw i32 %1684, 16, !spirv.Decorations !888		; visa id: 2068
  %1686 = bitcast i32 %1685 to float
  %1687 = zext i16 %1683 to i32		; visa id: 2069
  %1688 = shl nuw i32 %1687, 16, !spirv.Decorations !888		; visa id: 2070
  %1689 = bitcast i32 %1688 to float
  %1690 = fmul reassoc nsz arcp contract float %1686, %1689, !spirv.Decorations !881
  %1691 = fadd reassoc nsz arcp contract float %1690, %.sroa.122.1, !spirv.Decorations !881		; visa id: 2071
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2072

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %1674
; BB180 :
  %.sroa.122.2 = phi float [ %1691, %1674 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  br i1 %311, label %1692, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2073

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
; BB:
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1692:                                             ; preds = %._crit_edge.1.14
; BB182 :
  %.sroa.256.0.insert.ext873 = zext i32 %647 to i64		; visa id: 2075
  %1693 = shl nuw nsw i64 %.sroa.256.0.insert.ext873, 1		; visa id: 2076
  %1694 = add i64 %sink_3905, %1693		; visa id: 2077
  %1695 = inttoptr i64 %1694 to i16 addrspace(4)*		; visa id: 2078
  %1696 = addrspacecast i16 addrspace(4)* %1695 to i16 addrspace(1)*		; visa id: 2078
  %1697 = load i16, i16 addrspace(1)* %1696, align 2		; visa id: 2079
  %1698 = add i64 %sink_3890, %1693		; visa id: 2081
  %1699 = inttoptr i64 %1698 to i16 addrspace(4)*		; visa id: 2082
  %1700 = addrspacecast i16 addrspace(4)* %1699 to i16 addrspace(1)*		; visa id: 2082
  %1701 = load i16, i16 addrspace(1)* %1700, align 2		; visa id: 2083
  %1702 = zext i16 %1697 to i32		; visa id: 2085
  %1703 = shl nuw i32 %1702, 16, !spirv.Decorations !888		; visa id: 2086
  %1704 = bitcast i32 %1703 to float
  %1705 = zext i16 %1701 to i32		; visa id: 2087
  %1706 = shl nuw i32 %1705, 16, !spirv.Decorations !888		; visa id: 2088
  %1707 = bitcast i32 %1706 to float
  %1708 = fmul reassoc nsz arcp contract float %1704, %1707, !spirv.Decorations !881
  %1709 = fadd reassoc nsz arcp contract float %1708, %.sroa.186.1, !spirv.Decorations !881		; visa id: 2089
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2090

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %1692
; BB183 :
  %.sroa.186.2 = phi float [ %1709, %1692 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  br i1 %314, label %1710, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2091

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
; BB:
  br label %.preheader.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1710:                                             ; preds = %._crit_edge.2.14
; BB185 :
  %.sroa.256.0.insert.ext878 = zext i32 %647 to i64		; visa id: 2093
  %1711 = shl nuw nsw i64 %.sroa.256.0.insert.ext878, 1		; visa id: 2094
  %1712 = add i64 %sink_3904, %1711		; visa id: 2095
  %1713 = inttoptr i64 %1712 to i16 addrspace(4)*		; visa id: 2096
  %1714 = addrspacecast i16 addrspace(4)* %1713 to i16 addrspace(1)*		; visa id: 2096
  %1715 = load i16, i16 addrspace(1)* %1714, align 2		; visa id: 2097
  %1716 = add i64 %sink_3890, %1711		; visa id: 2099
  %1717 = inttoptr i64 %1716 to i16 addrspace(4)*		; visa id: 2100
  %1718 = addrspacecast i16 addrspace(4)* %1717 to i16 addrspace(1)*		; visa id: 2100
  %1719 = load i16, i16 addrspace(1)* %1718, align 2		; visa id: 2101
  %1720 = zext i16 %1715 to i32		; visa id: 2103
  %1721 = shl nuw i32 %1720, 16, !spirv.Decorations !888		; visa id: 2104
  %1722 = bitcast i32 %1721 to float
  %1723 = zext i16 %1719 to i32		; visa id: 2105
  %1724 = shl nuw i32 %1723, 16, !spirv.Decorations !888		; visa id: 2106
  %1725 = bitcast i32 %1724 to float
  %1726 = fmul reassoc nsz arcp contract float %1722, %1725, !spirv.Decorations !881
  %1727 = fadd reassoc nsz arcp contract float %1726, %.sroa.250.1, !spirv.Decorations !881		; visa id: 2107
  br label %.preheader.14, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2108

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %1710
; BB186 :
  %.sroa.250.2 = phi float [ %1727, %1710 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %sink_sink_3869 = bitcast <2 x i32> %447 to i64		; visa id: 2109
  %sink_sink_3845 = shl i64 %sink_sink_3869, 1		; visa id: 2113
  %sink_3889 = add i64 %.in3822, %sink_sink_3845		; visa id: 2114
  br i1 %318, label %1728, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2115

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
; BB:
  br label %._crit_edge.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1728:                                             ; preds = %.preheader.14
; BB188 :
  %.sroa.256.0.insert.ext883 = zext i32 %647 to i64		; visa id: 2117
  %1729 = shl nuw nsw i64 %.sroa.256.0.insert.ext883, 1		; visa id: 2118
  %1730 = add i64 %sink_3908, %1729		; visa id: 2119
  %1731 = inttoptr i64 %1730 to i16 addrspace(4)*		; visa id: 2120
  %1732 = addrspacecast i16 addrspace(4)* %1731 to i16 addrspace(1)*		; visa id: 2120
  %1733 = load i16, i16 addrspace(1)* %1732, align 2		; visa id: 2121
  %1734 = add i64 %sink_3889, %1729		; visa id: 2123
  %1735 = inttoptr i64 %1734 to i16 addrspace(4)*		; visa id: 2124
  %1736 = addrspacecast i16 addrspace(4)* %1735 to i16 addrspace(1)*		; visa id: 2124
  %1737 = load i16, i16 addrspace(1)* %1736, align 2		; visa id: 2125
  %1738 = zext i16 %1733 to i32		; visa id: 2127
  %1739 = shl nuw i32 %1738, 16, !spirv.Decorations !888		; visa id: 2128
  %1740 = bitcast i32 %1739 to float
  %1741 = zext i16 %1737 to i32		; visa id: 2129
  %1742 = shl nuw i32 %1741, 16, !spirv.Decorations !888		; visa id: 2130
  %1743 = bitcast i32 %1742 to float
  %1744 = fmul reassoc nsz arcp contract float %1740, %1743, !spirv.Decorations !881
  %1745 = fadd reassoc nsz arcp contract float %1744, %.sroa.62.1, !spirv.Decorations !881		; visa id: 2131
  br label %._crit_edge.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2132

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %1728
; BB189 :
  %.sroa.62.2 = phi float [ %1745, %1728 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  br i1 %321, label %1746, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2133

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
; BB:
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1746:                                             ; preds = %._crit_edge.15
; BB191 :
  %.sroa.256.0.insert.ext888 = zext i32 %647 to i64		; visa id: 2135
  %1747 = shl nuw nsw i64 %.sroa.256.0.insert.ext888, 1		; visa id: 2136
  %1748 = add i64 %sink_3906, %1747		; visa id: 2137
  %1749 = inttoptr i64 %1748 to i16 addrspace(4)*		; visa id: 2138
  %1750 = addrspacecast i16 addrspace(4)* %1749 to i16 addrspace(1)*		; visa id: 2138
  %1751 = load i16, i16 addrspace(1)* %1750, align 2		; visa id: 2139
  %1752 = add i64 %sink_3889, %1747		; visa id: 2141
  %1753 = inttoptr i64 %1752 to i16 addrspace(4)*		; visa id: 2142
  %1754 = addrspacecast i16 addrspace(4)* %1753 to i16 addrspace(1)*		; visa id: 2142
  %1755 = load i16, i16 addrspace(1)* %1754, align 2		; visa id: 2143
  %1756 = zext i16 %1751 to i32		; visa id: 2145
  %1757 = shl nuw i32 %1756, 16, !spirv.Decorations !888		; visa id: 2146
  %1758 = bitcast i32 %1757 to float
  %1759 = zext i16 %1755 to i32		; visa id: 2147
  %1760 = shl nuw i32 %1759, 16, !spirv.Decorations !888		; visa id: 2148
  %1761 = bitcast i32 %1760 to float
  %1762 = fmul reassoc nsz arcp contract float %1758, %1761, !spirv.Decorations !881
  %1763 = fadd reassoc nsz arcp contract float %1762, %.sroa.126.1, !spirv.Decorations !881		; visa id: 2149
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2150

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %1746
; BB192 :
  %.sroa.126.2 = phi float [ %1763, %1746 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  br i1 %324, label %1764, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2151

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
; BB:
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1764:                                             ; preds = %._crit_edge.1.15
; BB194 :
  %.sroa.256.0.insert.ext893 = zext i32 %647 to i64		; visa id: 2153
  %1765 = shl nuw nsw i64 %.sroa.256.0.insert.ext893, 1		; visa id: 2154
  %1766 = add i64 %sink_3905, %1765		; visa id: 2155
  %1767 = inttoptr i64 %1766 to i16 addrspace(4)*		; visa id: 2156
  %1768 = addrspacecast i16 addrspace(4)* %1767 to i16 addrspace(1)*		; visa id: 2156
  %1769 = load i16, i16 addrspace(1)* %1768, align 2		; visa id: 2157
  %1770 = add i64 %sink_3889, %1765		; visa id: 2159
  %1771 = inttoptr i64 %1770 to i16 addrspace(4)*		; visa id: 2160
  %1772 = addrspacecast i16 addrspace(4)* %1771 to i16 addrspace(1)*		; visa id: 2160
  %1773 = load i16, i16 addrspace(1)* %1772, align 2		; visa id: 2161
  %1774 = zext i16 %1769 to i32		; visa id: 2163
  %1775 = shl nuw i32 %1774, 16, !spirv.Decorations !888		; visa id: 2164
  %1776 = bitcast i32 %1775 to float
  %1777 = zext i16 %1773 to i32		; visa id: 2165
  %1778 = shl nuw i32 %1777, 16, !spirv.Decorations !888		; visa id: 2166
  %1779 = bitcast i32 %1778 to float
  %1780 = fmul reassoc nsz arcp contract float %1776, %1779, !spirv.Decorations !881
  %1781 = fadd reassoc nsz arcp contract float %1780, %.sroa.190.1, !spirv.Decorations !881		; visa id: 2167
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2168

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %1764
; BB195 :
  %.sroa.190.2 = phi float [ %1781, %1764 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  br i1 %327, label %1782, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2169

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
; BB:
  br label %.preheader.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879

1782:                                             ; preds = %._crit_edge.2.15
; BB197 :
  %.sroa.256.0.insert.ext898 = zext i32 %647 to i64		; visa id: 2171
  %1783 = shl nuw nsw i64 %.sroa.256.0.insert.ext898, 1		; visa id: 2172
  %1784 = add i64 %sink_3904, %1783		; visa id: 2173
  %1785 = inttoptr i64 %1784 to i16 addrspace(4)*		; visa id: 2174
  %1786 = addrspacecast i16 addrspace(4)* %1785 to i16 addrspace(1)*		; visa id: 2174
  %1787 = load i16, i16 addrspace(1)* %1786, align 2		; visa id: 2175
  %1788 = add i64 %sink_3889, %1783		; visa id: 2177
  %1789 = inttoptr i64 %1788 to i16 addrspace(4)*		; visa id: 2178
  %1790 = addrspacecast i16 addrspace(4)* %1789 to i16 addrspace(1)*		; visa id: 2178
  %1791 = load i16, i16 addrspace(1)* %1790, align 2		; visa id: 2179
  %1792 = zext i16 %1787 to i32		; visa id: 2181
  %1793 = shl nuw i32 %1792, 16, !spirv.Decorations !888		; visa id: 2182
  %1794 = bitcast i32 %1793 to float
  %1795 = zext i16 %1791 to i32		; visa id: 2183
  %1796 = shl nuw i32 %1795, 16, !spirv.Decorations !888		; visa id: 2184
  %1797 = bitcast i32 %1796 to float
  %1798 = fmul reassoc nsz arcp contract float %1794, %1797, !spirv.Decorations !881
  %1799 = fadd reassoc nsz arcp contract float %1798, %.sroa.254.1, !spirv.Decorations !881		; visa id: 2185
  br label %.preheader.15, !stats.blockFrequency.digits !887, !stats.blockFrequency.scale !879		; visa id: 2186

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %1782
; BB198 :
  %.sroa.254.2 = phi float [ %1799, %1782 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %1800 = add nuw nsw i32 %647, 1, !spirv.Decorations !890		; visa id: 2187
  %1801 = icmp slt i32 %1800, %const_reg_dword2		; visa id: 2188
  br i1 %1801, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 2189

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
  %sink_3868 = bitcast <2 x i32> %461 to i64		; visa id: 2191
  %sink_3844 = shl i64 %sink_3868, 2		; visa id: 2195
  %sink_3843 = shl nsw i64 %454, 2		; visa id: 2196
  br i1 %120, label %1802, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2197

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
; BB:
  br label %._crit_edge70, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1802:                                             ; preds = %.preheader1.preheader
; BB203 :
  %1803 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !881		; visa id: 2199
  br i1 %81, label %1808, label %1804, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2200

1804:                                             ; preds = %1802
; BB204 :
  %1805 = add i64 %.in, %456		; visa id: 2202
  %1806 = inttoptr i64 %1805 to float addrspace(4)*		; visa id: 2203
  %1807 = addrspacecast float addrspace(4)* %1806 to float addrspace(1)*		; visa id: 2203
  store float %1803, float addrspace(1)* %1807, align 4		; visa id: 2204
  br label %._crit_edge70, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2205

1808:                                             ; preds = %1802
; BB205 :
  %1809 = add i64 %.in3821, %sink_3844		; visa id: 2207
  %1810 = add i64 %1809, %sink_3843		; visa id: 2208
  %1811 = inttoptr i64 %1810 to float addrspace(4)*		; visa id: 2209
  %1812 = addrspacecast float addrspace(4)* %1811 to float addrspace(1)*		; visa id: 2209
  %1813 = load float, float addrspace(1)* %1812, align 4		; visa id: 2210
  %1814 = fmul reassoc nsz arcp contract float %1813, %4, !spirv.Decorations !881		; visa id: 2211
  %1815 = fadd reassoc nsz arcp contract float %1803, %1814, !spirv.Decorations !881		; visa id: 2212
  %1816 = add i64 %.in, %456		; visa id: 2213
  %1817 = inttoptr i64 %1816 to float addrspace(4)*		; visa id: 2214
  %1818 = addrspacecast float addrspace(4)* %1817 to float addrspace(1)*		; visa id: 2214
  store float %1815, float addrspace(1)* %1818, align 4		; visa id: 2215
  br label %._crit_edge70, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2216

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %1804, %1808
; BB206 :
  %sink_3867 = bitcast <2 x i32> %474 to i64		; visa id: 2217
  %sink_3842 = shl i64 %sink_3867, 2		; visa id: 2221
  br i1 %124, label %1819, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2222

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
; BB:
  br label %._crit_edge70.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1819:                                             ; preds = %._crit_edge70
; BB208 :
  %1820 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !881		; visa id: 2224
  br i1 %81, label %1825, label %1821, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2225

1821:                                             ; preds = %1819
; BB209 :
  %1822 = add i64 %.in, %469		; visa id: 2227
  %1823 = inttoptr i64 %1822 to float addrspace(4)*		; visa id: 2228
  %1824 = addrspacecast float addrspace(4)* %1823 to float addrspace(1)*		; visa id: 2228
  store float %1820, float addrspace(1)* %1824, align 4		; visa id: 2229
  br label %._crit_edge70.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2230

1825:                                             ; preds = %1819
; BB210 :
  %1826 = add i64 %.in3821, %sink_3842		; visa id: 2232
  %1827 = add i64 %1826, %sink_3843		; visa id: 2233
  %1828 = inttoptr i64 %1827 to float addrspace(4)*		; visa id: 2234
  %1829 = addrspacecast float addrspace(4)* %1828 to float addrspace(1)*		; visa id: 2234
  %1830 = load float, float addrspace(1)* %1829, align 4		; visa id: 2235
  %1831 = fmul reassoc nsz arcp contract float %1830, %4, !spirv.Decorations !881		; visa id: 2236
  %1832 = fadd reassoc nsz arcp contract float %1820, %1831, !spirv.Decorations !881		; visa id: 2237
  %1833 = add i64 %.in, %469		; visa id: 2238
  %1834 = inttoptr i64 %1833 to float addrspace(4)*		; visa id: 2239
  %1835 = addrspacecast float addrspace(4)* %1834 to float addrspace(1)*		; visa id: 2239
  store float %1832, float addrspace(1)* %1835, align 4		; visa id: 2240
  br label %._crit_edge70.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2241

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %1825, %1821
; BB211 :
  %sink_3866 = bitcast <2 x i32> %487 to i64		; visa id: 2242
  %sink_3841 = shl i64 %sink_3866, 2		; visa id: 2246
  br i1 %128, label %1836, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2247

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
; BB:
  br label %._crit_edge70.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1836:                                             ; preds = %._crit_edge70.1
; BB213 :
  %1837 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !881		; visa id: 2249
  br i1 %81, label %1842, label %1838, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2250

1838:                                             ; preds = %1836
; BB214 :
  %1839 = add i64 %.in, %482		; visa id: 2252
  %1840 = inttoptr i64 %1839 to float addrspace(4)*		; visa id: 2253
  %1841 = addrspacecast float addrspace(4)* %1840 to float addrspace(1)*		; visa id: 2253
  store float %1837, float addrspace(1)* %1841, align 4		; visa id: 2254
  br label %._crit_edge70.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2255

1842:                                             ; preds = %1836
; BB215 :
  %1843 = add i64 %.in3821, %sink_3841		; visa id: 2257
  %1844 = add i64 %1843, %sink_3843		; visa id: 2258
  %1845 = inttoptr i64 %1844 to float addrspace(4)*		; visa id: 2259
  %1846 = addrspacecast float addrspace(4)* %1845 to float addrspace(1)*		; visa id: 2259
  %1847 = load float, float addrspace(1)* %1846, align 4		; visa id: 2260
  %1848 = fmul reassoc nsz arcp contract float %1847, %4, !spirv.Decorations !881		; visa id: 2261
  %1849 = fadd reassoc nsz arcp contract float %1837, %1848, !spirv.Decorations !881		; visa id: 2262
  %1850 = add i64 %.in, %482		; visa id: 2263
  %1851 = inttoptr i64 %1850 to float addrspace(4)*		; visa id: 2264
  %1852 = addrspacecast float addrspace(4)* %1851 to float addrspace(1)*		; visa id: 2264
  store float %1849, float addrspace(1)* %1852, align 4		; visa id: 2265
  br label %._crit_edge70.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2266

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %1842, %1838
; BB216 :
  %sink_3865 = bitcast <2 x i32> %500 to i64		; visa id: 2267
  %sink_3840 = shl i64 %sink_3865, 2		; visa id: 2271
  br i1 %132, label %1853, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2272

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
; BB:
  br label %.preheader1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1853:                                             ; preds = %._crit_edge70.2
; BB218 :
  %1854 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !881		; visa id: 2274
  br i1 %81, label %1859, label %1855, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2275

1855:                                             ; preds = %1853
; BB219 :
  %1856 = add i64 %.in, %495		; visa id: 2277
  %1857 = inttoptr i64 %1856 to float addrspace(4)*		; visa id: 2278
  %1858 = addrspacecast float addrspace(4)* %1857 to float addrspace(1)*		; visa id: 2278
  store float %1854, float addrspace(1)* %1858, align 4		; visa id: 2279
  br label %.preheader1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2280

1859:                                             ; preds = %1853
; BB220 :
  %1860 = add i64 %.in3821, %sink_3840		; visa id: 2282
  %1861 = add i64 %1860, %sink_3843		; visa id: 2283
  %1862 = inttoptr i64 %1861 to float addrspace(4)*		; visa id: 2284
  %1863 = addrspacecast float addrspace(4)* %1862 to float addrspace(1)*		; visa id: 2284
  %1864 = load float, float addrspace(1)* %1863, align 4		; visa id: 2285
  %1865 = fmul reassoc nsz arcp contract float %1864, %4, !spirv.Decorations !881		; visa id: 2286
  %1866 = fadd reassoc nsz arcp contract float %1854, %1865, !spirv.Decorations !881		; visa id: 2287
  %1867 = add i64 %.in, %495		; visa id: 2288
  %1868 = inttoptr i64 %1867 to float addrspace(4)*		; visa id: 2289
  %1869 = addrspacecast float addrspace(4)* %1868 to float addrspace(1)*		; visa id: 2289
  store float %1866, float addrspace(1)* %1869, align 4		; visa id: 2290
  br label %.preheader1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2291

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %1859, %1855
; BB221 :
  %sink_3839 = shl nsw i64 %501, 2		; visa id: 2292
  br i1 %136, label %1870, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2293

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
; BB:
  br label %._crit_edge70.176, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1870:                                             ; preds = %.preheader1
; BB223 :
  %1871 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !881		; visa id: 2295
  br i1 %81, label %1876, label %1872, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2296

1872:                                             ; preds = %1870
; BB224 :
  %1873 = add i64 %.in, %503		; visa id: 2298
  %1874 = inttoptr i64 %1873 to float addrspace(4)*		; visa id: 2299
  %1875 = addrspacecast float addrspace(4)* %1874 to float addrspace(1)*		; visa id: 2299
  store float %1871, float addrspace(1)* %1875, align 4		; visa id: 2300
  br label %._crit_edge70.176, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2301

1876:                                             ; preds = %1870
; BB225 :
  %1877 = add i64 %.in3821, %sink_3844		; visa id: 2303
  %1878 = add i64 %1877, %sink_3839		; visa id: 2304
  %1879 = inttoptr i64 %1878 to float addrspace(4)*		; visa id: 2305
  %1880 = addrspacecast float addrspace(4)* %1879 to float addrspace(1)*		; visa id: 2305
  %1881 = load float, float addrspace(1)* %1880, align 4		; visa id: 2306
  %1882 = fmul reassoc nsz arcp contract float %1881, %4, !spirv.Decorations !881		; visa id: 2307
  %1883 = fadd reassoc nsz arcp contract float %1871, %1882, !spirv.Decorations !881		; visa id: 2308
  %1884 = add i64 %.in, %503		; visa id: 2309
  %1885 = inttoptr i64 %1884 to float addrspace(4)*		; visa id: 2310
  %1886 = addrspacecast float addrspace(4)* %1885 to float addrspace(1)*		; visa id: 2310
  store float %1883, float addrspace(1)* %1886, align 4		; visa id: 2311
  br label %._crit_edge70.176, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2312

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %1876, %1872
; BB226 :
  br i1 %139, label %1887, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2313

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
; BB:
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1887:                                             ; preds = %._crit_edge70.176
; BB228 :
  %1888 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !881		; visa id: 2315
  br i1 %81, label %1893, label %1889, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2316

1889:                                             ; preds = %1887
; BB229 :
  %1890 = add i64 %.in, %505		; visa id: 2318
  %1891 = inttoptr i64 %1890 to float addrspace(4)*		; visa id: 2319
  %1892 = addrspacecast float addrspace(4)* %1891 to float addrspace(1)*		; visa id: 2319
  store float %1888, float addrspace(1)* %1892, align 4		; visa id: 2320
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2321

1893:                                             ; preds = %1887
; BB230 :
  %1894 = add i64 %.in3821, %sink_3842		; visa id: 2323
  %1895 = add i64 %1894, %sink_3839		; visa id: 2324
  %1896 = inttoptr i64 %1895 to float addrspace(4)*		; visa id: 2325
  %1897 = addrspacecast float addrspace(4)* %1896 to float addrspace(1)*		; visa id: 2325
  %1898 = load float, float addrspace(1)* %1897, align 4		; visa id: 2326
  %1899 = fmul reassoc nsz arcp contract float %1898, %4, !spirv.Decorations !881		; visa id: 2327
  %1900 = fadd reassoc nsz arcp contract float %1888, %1899, !spirv.Decorations !881		; visa id: 2328
  %1901 = add i64 %.in, %505		; visa id: 2329
  %1902 = inttoptr i64 %1901 to float addrspace(4)*		; visa id: 2330
  %1903 = addrspacecast float addrspace(4)* %1902 to float addrspace(1)*		; visa id: 2330
  store float %1900, float addrspace(1)* %1903, align 4		; visa id: 2331
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2332

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %1893, %1889
; BB231 :
  br i1 %142, label %1904, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2333

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
; BB:
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1904:                                             ; preds = %._crit_edge70.1.1
; BB233 :
  %1905 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !881		; visa id: 2335
  br i1 %81, label %1910, label %1906, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2336

1906:                                             ; preds = %1904
; BB234 :
  %1907 = add i64 %.in, %507		; visa id: 2338
  %1908 = inttoptr i64 %1907 to float addrspace(4)*		; visa id: 2339
  %1909 = addrspacecast float addrspace(4)* %1908 to float addrspace(1)*		; visa id: 2339
  store float %1905, float addrspace(1)* %1909, align 4		; visa id: 2340
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2341

1910:                                             ; preds = %1904
; BB235 :
  %1911 = add i64 %.in3821, %sink_3841		; visa id: 2343
  %1912 = add i64 %1911, %sink_3839		; visa id: 2344
  %1913 = inttoptr i64 %1912 to float addrspace(4)*		; visa id: 2345
  %1914 = addrspacecast float addrspace(4)* %1913 to float addrspace(1)*		; visa id: 2345
  %1915 = load float, float addrspace(1)* %1914, align 4		; visa id: 2346
  %1916 = fmul reassoc nsz arcp contract float %1915, %4, !spirv.Decorations !881		; visa id: 2347
  %1917 = fadd reassoc nsz arcp contract float %1905, %1916, !spirv.Decorations !881		; visa id: 2348
  %1918 = add i64 %.in, %507		; visa id: 2349
  %1919 = inttoptr i64 %1918 to float addrspace(4)*		; visa id: 2350
  %1920 = addrspacecast float addrspace(4)* %1919 to float addrspace(1)*		; visa id: 2350
  store float %1917, float addrspace(1)* %1920, align 4		; visa id: 2351
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2352

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %1910, %1906
; BB236 :
  br i1 %145, label %1921, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2353

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
; BB:
  br label %.preheader1.1, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1921:                                             ; preds = %._crit_edge70.2.1
; BB238 :
  %1922 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !881		; visa id: 2355
  br i1 %81, label %1927, label %1923, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2356

1923:                                             ; preds = %1921
; BB239 :
  %1924 = add i64 %.in, %509		; visa id: 2358
  %1925 = inttoptr i64 %1924 to float addrspace(4)*		; visa id: 2359
  %1926 = addrspacecast float addrspace(4)* %1925 to float addrspace(1)*		; visa id: 2359
  store float %1922, float addrspace(1)* %1926, align 4		; visa id: 2360
  br label %.preheader1.1, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2361

1927:                                             ; preds = %1921
; BB240 :
  %1928 = add i64 %.in3821, %sink_3840		; visa id: 2363
  %1929 = add i64 %1928, %sink_3839		; visa id: 2364
  %1930 = inttoptr i64 %1929 to float addrspace(4)*		; visa id: 2365
  %1931 = addrspacecast float addrspace(4)* %1930 to float addrspace(1)*		; visa id: 2365
  %1932 = load float, float addrspace(1)* %1931, align 4		; visa id: 2366
  %1933 = fmul reassoc nsz arcp contract float %1932, %4, !spirv.Decorations !881		; visa id: 2367
  %1934 = fadd reassoc nsz arcp contract float %1922, %1933, !spirv.Decorations !881		; visa id: 2368
  %1935 = add i64 %.in, %509		; visa id: 2369
  %1936 = inttoptr i64 %1935 to float addrspace(4)*		; visa id: 2370
  %1937 = addrspacecast float addrspace(4)* %1936 to float addrspace(1)*		; visa id: 2370
  store float %1934, float addrspace(1)* %1937, align 4		; visa id: 2371
  br label %.preheader1.1, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2372

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %1927, %1923
; BB241 :
  %sink_3838 = shl nsw i64 %510, 2		; visa id: 2373
  br i1 %149, label %1938, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2374

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
; BB:
  br label %._crit_edge70.277, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1938:                                             ; preds = %.preheader1.1
; BB243 :
  %1939 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !881		; visa id: 2376
  br i1 %81, label %1944, label %1940, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2377

1940:                                             ; preds = %1938
; BB244 :
  %1941 = add i64 %.in, %512		; visa id: 2379
  %1942 = inttoptr i64 %1941 to float addrspace(4)*		; visa id: 2380
  %1943 = addrspacecast float addrspace(4)* %1942 to float addrspace(1)*		; visa id: 2380
  store float %1939, float addrspace(1)* %1943, align 4		; visa id: 2381
  br label %._crit_edge70.277, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2382

1944:                                             ; preds = %1938
; BB245 :
  %1945 = add i64 %.in3821, %sink_3844		; visa id: 2384
  %1946 = add i64 %1945, %sink_3838		; visa id: 2385
  %1947 = inttoptr i64 %1946 to float addrspace(4)*		; visa id: 2386
  %1948 = addrspacecast float addrspace(4)* %1947 to float addrspace(1)*		; visa id: 2386
  %1949 = load float, float addrspace(1)* %1948, align 4		; visa id: 2387
  %1950 = fmul reassoc nsz arcp contract float %1949, %4, !spirv.Decorations !881		; visa id: 2388
  %1951 = fadd reassoc nsz arcp contract float %1939, %1950, !spirv.Decorations !881		; visa id: 2389
  %1952 = add i64 %.in, %512		; visa id: 2390
  %1953 = inttoptr i64 %1952 to float addrspace(4)*		; visa id: 2391
  %1954 = addrspacecast float addrspace(4)* %1953 to float addrspace(1)*		; visa id: 2391
  store float %1951, float addrspace(1)* %1954, align 4		; visa id: 2392
  br label %._crit_edge70.277, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2393

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %1944, %1940
; BB246 :
  br i1 %152, label %1955, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2394

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
; BB:
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1955:                                             ; preds = %._crit_edge70.277
; BB248 :
  %1956 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !881		; visa id: 2396
  br i1 %81, label %1961, label %1957, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2397

1957:                                             ; preds = %1955
; BB249 :
  %1958 = add i64 %.in, %514		; visa id: 2399
  %1959 = inttoptr i64 %1958 to float addrspace(4)*		; visa id: 2400
  %1960 = addrspacecast float addrspace(4)* %1959 to float addrspace(1)*		; visa id: 2400
  store float %1956, float addrspace(1)* %1960, align 4		; visa id: 2401
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2402

1961:                                             ; preds = %1955
; BB250 :
  %1962 = add i64 %.in3821, %sink_3842		; visa id: 2404
  %1963 = add i64 %1962, %sink_3838		; visa id: 2405
  %1964 = inttoptr i64 %1963 to float addrspace(4)*		; visa id: 2406
  %1965 = addrspacecast float addrspace(4)* %1964 to float addrspace(1)*		; visa id: 2406
  %1966 = load float, float addrspace(1)* %1965, align 4		; visa id: 2407
  %1967 = fmul reassoc nsz arcp contract float %1966, %4, !spirv.Decorations !881		; visa id: 2408
  %1968 = fadd reassoc nsz arcp contract float %1956, %1967, !spirv.Decorations !881		; visa id: 2409
  %1969 = add i64 %.in, %514		; visa id: 2410
  %1970 = inttoptr i64 %1969 to float addrspace(4)*		; visa id: 2411
  %1971 = addrspacecast float addrspace(4)* %1970 to float addrspace(1)*		; visa id: 2411
  store float %1968, float addrspace(1)* %1971, align 4		; visa id: 2412
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2413

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %1961, %1957
; BB251 :
  br i1 %155, label %1972, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2414

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
; BB:
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1972:                                             ; preds = %._crit_edge70.1.2
; BB253 :
  %1973 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !881		; visa id: 2416
  br i1 %81, label %1978, label %1974, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2417

1974:                                             ; preds = %1972
; BB254 :
  %1975 = add i64 %.in, %516		; visa id: 2419
  %1976 = inttoptr i64 %1975 to float addrspace(4)*		; visa id: 2420
  %1977 = addrspacecast float addrspace(4)* %1976 to float addrspace(1)*		; visa id: 2420
  store float %1973, float addrspace(1)* %1977, align 4		; visa id: 2421
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2422

1978:                                             ; preds = %1972
; BB255 :
  %1979 = add i64 %.in3821, %sink_3841		; visa id: 2424
  %1980 = add i64 %1979, %sink_3838		; visa id: 2425
  %1981 = inttoptr i64 %1980 to float addrspace(4)*		; visa id: 2426
  %1982 = addrspacecast float addrspace(4)* %1981 to float addrspace(1)*		; visa id: 2426
  %1983 = load float, float addrspace(1)* %1982, align 4		; visa id: 2427
  %1984 = fmul reassoc nsz arcp contract float %1983, %4, !spirv.Decorations !881		; visa id: 2428
  %1985 = fadd reassoc nsz arcp contract float %1973, %1984, !spirv.Decorations !881		; visa id: 2429
  %1986 = add i64 %.in, %516		; visa id: 2430
  %1987 = inttoptr i64 %1986 to float addrspace(4)*		; visa id: 2431
  %1988 = addrspacecast float addrspace(4)* %1987 to float addrspace(1)*		; visa id: 2431
  store float %1985, float addrspace(1)* %1988, align 4		; visa id: 2432
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2433

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %1978, %1974
; BB256 :
  br i1 %158, label %1989, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2434

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
; BB:
  br label %.preheader1.2, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

1989:                                             ; preds = %._crit_edge70.2.2
; BB258 :
  %1990 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !881		; visa id: 2436
  br i1 %81, label %1995, label %1991, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2437

1991:                                             ; preds = %1989
; BB259 :
  %1992 = add i64 %.in, %518		; visa id: 2439
  %1993 = inttoptr i64 %1992 to float addrspace(4)*		; visa id: 2440
  %1994 = addrspacecast float addrspace(4)* %1993 to float addrspace(1)*		; visa id: 2440
  store float %1990, float addrspace(1)* %1994, align 4		; visa id: 2441
  br label %.preheader1.2, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2442

1995:                                             ; preds = %1989
; BB260 :
  %1996 = add i64 %.in3821, %sink_3840		; visa id: 2444
  %1997 = add i64 %1996, %sink_3838		; visa id: 2445
  %1998 = inttoptr i64 %1997 to float addrspace(4)*		; visa id: 2446
  %1999 = addrspacecast float addrspace(4)* %1998 to float addrspace(1)*		; visa id: 2446
  %2000 = load float, float addrspace(1)* %1999, align 4		; visa id: 2447
  %2001 = fmul reassoc nsz arcp contract float %2000, %4, !spirv.Decorations !881		; visa id: 2448
  %2002 = fadd reassoc nsz arcp contract float %1990, %2001, !spirv.Decorations !881		; visa id: 2449
  %2003 = add i64 %.in, %518		; visa id: 2450
  %2004 = inttoptr i64 %2003 to float addrspace(4)*		; visa id: 2451
  %2005 = addrspacecast float addrspace(4)* %2004 to float addrspace(1)*		; visa id: 2451
  store float %2002, float addrspace(1)* %2005, align 4		; visa id: 2452
  br label %.preheader1.2, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2453

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %1995, %1991
; BB261 :
  %sink_3837 = shl nsw i64 %519, 2		; visa id: 2454
  br i1 %162, label %2006, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2455

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
; BB:
  br label %._crit_edge70.378, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2006:                                             ; preds = %.preheader1.2
; BB263 :
  %2007 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !881		; visa id: 2457
  br i1 %81, label %2012, label %2008, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2458

2008:                                             ; preds = %2006
; BB264 :
  %2009 = add i64 %.in, %521		; visa id: 2460
  %2010 = inttoptr i64 %2009 to float addrspace(4)*		; visa id: 2461
  %2011 = addrspacecast float addrspace(4)* %2010 to float addrspace(1)*		; visa id: 2461
  store float %2007, float addrspace(1)* %2011, align 4		; visa id: 2462
  br label %._crit_edge70.378, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2463

2012:                                             ; preds = %2006
; BB265 :
  %2013 = add i64 %.in3821, %sink_3844		; visa id: 2465
  %2014 = add i64 %2013, %sink_3837		; visa id: 2466
  %2015 = inttoptr i64 %2014 to float addrspace(4)*		; visa id: 2467
  %2016 = addrspacecast float addrspace(4)* %2015 to float addrspace(1)*		; visa id: 2467
  %2017 = load float, float addrspace(1)* %2016, align 4		; visa id: 2468
  %2018 = fmul reassoc nsz arcp contract float %2017, %4, !spirv.Decorations !881		; visa id: 2469
  %2019 = fadd reassoc nsz arcp contract float %2007, %2018, !spirv.Decorations !881		; visa id: 2470
  %2020 = add i64 %.in, %521		; visa id: 2471
  %2021 = inttoptr i64 %2020 to float addrspace(4)*		; visa id: 2472
  %2022 = addrspacecast float addrspace(4)* %2021 to float addrspace(1)*		; visa id: 2472
  store float %2019, float addrspace(1)* %2022, align 4		; visa id: 2473
  br label %._crit_edge70.378, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2474

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %2012, %2008
; BB266 :
  br i1 %165, label %2023, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2475

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
; BB:
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2023:                                             ; preds = %._crit_edge70.378
; BB268 :
  %2024 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !881		; visa id: 2477
  br i1 %81, label %2029, label %2025, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2478

2025:                                             ; preds = %2023
; BB269 :
  %2026 = add i64 %.in, %523		; visa id: 2480
  %2027 = inttoptr i64 %2026 to float addrspace(4)*		; visa id: 2481
  %2028 = addrspacecast float addrspace(4)* %2027 to float addrspace(1)*		; visa id: 2481
  store float %2024, float addrspace(1)* %2028, align 4		; visa id: 2482
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2483

2029:                                             ; preds = %2023
; BB270 :
  %2030 = add i64 %.in3821, %sink_3842		; visa id: 2485
  %2031 = add i64 %2030, %sink_3837		; visa id: 2486
  %2032 = inttoptr i64 %2031 to float addrspace(4)*		; visa id: 2487
  %2033 = addrspacecast float addrspace(4)* %2032 to float addrspace(1)*		; visa id: 2487
  %2034 = load float, float addrspace(1)* %2033, align 4		; visa id: 2488
  %2035 = fmul reassoc nsz arcp contract float %2034, %4, !spirv.Decorations !881		; visa id: 2489
  %2036 = fadd reassoc nsz arcp contract float %2024, %2035, !spirv.Decorations !881		; visa id: 2490
  %2037 = add i64 %.in, %523		; visa id: 2491
  %2038 = inttoptr i64 %2037 to float addrspace(4)*		; visa id: 2492
  %2039 = addrspacecast float addrspace(4)* %2038 to float addrspace(1)*		; visa id: 2492
  store float %2036, float addrspace(1)* %2039, align 4		; visa id: 2493
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2494

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %2029, %2025
; BB271 :
  br i1 %168, label %2040, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2495

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
; BB:
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2040:                                             ; preds = %._crit_edge70.1.3
; BB273 :
  %2041 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !881		; visa id: 2497
  br i1 %81, label %2046, label %2042, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2498

2042:                                             ; preds = %2040
; BB274 :
  %2043 = add i64 %.in, %525		; visa id: 2500
  %2044 = inttoptr i64 %2043 to float addrspace(4)*		; visa id: 2501
  %2045 = addrspacecast float addrspace(4)* %2044 to float addrspace(1)*		; visa id: 2501
  store float %2041, float addrspace(1)* %2045, align 4		; visa id: 2502
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2503

2046:                                             ; preds = %2040
; BB275 :
  %2047 = add i64 %.in3821, %sink_3841		; visa id: 2505
  %2048 = add i64 %2047, %sink_3837		; visa id: 2506
  %2049 = inttoptr i64 %2048 to float addrspace(4)*		; visa id: 2507
  %2050 = addrspacecast float addrspace(4)* %2049 to float addrspace(1)*		; visa id: 2507
  %2051 = load float, float addrspace(1)* %2050, align 4		; visa id: 2508
  %2052 = fmul reassoc nsz arcp contract float %2051, %4, !spirv.Decorations !881		; visa id: 2509
  %2053 = fadd reassoc nsz arcp contract float %2041, %2052, !spirv.Decorations !881		; visa id: 2510
  %2054 = add i64 %.in, %525		; visa id: 2511
  %2055 = inttoptr i64 %2054 to float addrspace(4)*		; visa id: 2512
  %2056 = addrspacecast float addrspace(4)* %2055 to float addrspace(1)*		; visa id: 2512
  store float %2053, float addrspace(1)* %2056, align 4		; visa id: 2513
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2514

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %2046, %2042
; BB276 :
  br i1 %171, label %2057, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2515

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
; BB:
  br label %.preheader1.3, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2057:                                             ; preds = %._crit_edge70.2.3
; BB278 :
  %2058 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !881		; visa id: 2517
  br i1 %81, label %2063, label %2059, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2518

2059:                                             ; preds = %2057
; BB279 :
  %2060 = add i64 %.in, %527		; visa id: 2520
  %2061 = inttoptr i64 %2060 to float addrspace(4)*		; visa id: 2521
  %2062 = addrspacecast float addrspace(4)* %2061 to float addrspace(1)*		; visa id: 2521
  store float %2058, float addrspace(1)* %2062, align 4		; visa id: 2522
  br label %.preheader1.3, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2523

2063:                                             ; preds = %2057
; BB280 :
  %2064 = add i64 %.in3821, %sink_3840		; visa id: 2525
  %2065 = add i64 %2064, %sink_3837		; visa id: 2526
  %2066 = inttoptr i64 %2065 to float addrspace(4)*		; visa id: 2527
  %2067 = addrspacecast float addrspace(4)* %2066 to float addrspace(1)*		; visa id: 2527
  %2068 = load float, float addrspace(1)* %2067, align 4		; visa id: 2528
  %2069 = fmul reassoc nsz arcp contract float %2068, %4, !spirv.Decorations !881		; visa id: 2529
  %2070 = fadd reassoc nsz arcp contract float %2058, %2069, !spirv.Decorations !881		; visa id: 2530
  %2071 = add i64 %.in, %527		; visa id: 2531
  %2072 = inttoptr i64 %2071 to float addrspace(4)*		; visa id: 2532
  %2073 = addrspacecast float addrspace(4)* %2072 to float addrspace(1)*		; visa id: 2532
  store float %2070, float addrspace(1)* %2073, align 4		; visa id: 2533
  br label %.preheader1.3, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2534

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %2063, %2059
; BB281 :
  %sink_3836 = shl nsw i64 %528, 2		; visa id: 2535
  br i1 %175, label %2074, label %.preheader1.3.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2536

.preheader1.3.._crit_edge70.4_crit_edge:          ; preds = %.preheader1.3
; BB:
  br label %._crit_edge70.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2074:                                             ; preds = %.preheader1.3
; BB283 :
  %2075 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !881		; visa id: 2538
  br i1 %81, label %2080, label %2076, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2539

2076:                                             ; preds = %2074
; BB284 :
  %2077 = add i64 %.in, %530		; visa id: 2541
  %2078 = inttoptr i64 %2077 to float addrspace(4)*		; visa id: 2542
  %2079 = addrspacecast float addrspace(4)* %2078 to float addrspace(1)*		; visa id: 2542
  store float %2075, float addrspace(1)* %2079, align 4		; visa id: 2543
  br label %._crit_edge70.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2544

2080:                                             ; preds = %2074
; BB285 :
  %2081 = add i64 %.in3821, %sink_3844		; visa id: 2546
  %2082 = add i64 %2081, %sink_3836		; visa id: 2547
  %2083 = inttoptr i64 %2082 to float addrspace(4)*		; visa id: 2548
  %2084 = addrspacecast float addrspace(4)* %2083 to float addrspace(1)*		; visa id: 2548
  %2085 = load float, float addrspace(1)* %2084, align 4		; visa id: 2549
  %2086 = fmul reassoc nsz arcp contract float %2085, %4, !spirv.Decorations !881		; visa id: 2550
  %2087 = fadd reassoc nsz arcp contract float %2075, %2086, !spirv.Decorations !881		; visa id: 2551
  %2088 = add i64 %.in, %530		; visa id: 2552
  %2089 = inttoptr i64 %2088 to float addrspace(4)*		; visa id: 2553
  %2090 = addrspacecast float addrspace(4)* %2089 to float addrspace(1)*		; visa id: 2553
  store float %2087, float addrspace(1)* %2090, align 4		; visa id: 2554
  br label %._crit_edge70.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2555

._crit_edge70.4:                                  ; preds = %.preheader1.3.._crit_edge70.4_crit_edge, %2080, %2076
; BB286 :
  br i1 %178, label %2091, label %._crit_edge70.4.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2556

._crit_edge70.4.._crit_edge70.1.4_crit_edge:      ; preds = %._crit_edge70.4
; BB:
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2091:                                             ; preds = %._crit_edge70.4
; BB288 :
  %2092 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !881		; visa id: 2558
  br i1 %81, label %2097, label %2093, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2559

2093:                                             ; preds = %2091
; BB289 :
  %2094 = add i64 %.in, %532		; visa id: 2561
  %2095 = inttoptr i64 %2094 to float addrspace(4)*		; visa id: 2562
  %2096 = addrspacecast float addrspace(4)* %2095 to float addrspace(1)*		; visa id: 2562
  store float %2092, float addrspace(1)* %2096, align 4		; visa id: 2563
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2564

2097:                                             ; preds = %2091
; BB290 :
  %2098 = add i64 %.in3821, %sink_3842		; visa id: 2566
  %2099 = add i64 %2098, %sink_3836		; visa id: 2567
  %2100 = inttoptr i64 %2099 to float addrspace(4)*		; visa id: 2568
  %2101 = addrspacecast float addrspace(4)* %2100 to float addrspace(1)*		; visa id: 2568
  %2102 = load float, float addrspace(1)* %2101, align 4		; visa id: 2569
  %2103 = fmul reassoc nsz arcp contract float %2102, %4, !spirv.Decorations !881		; visa id: 2570
  %2104 = fadd reassoc nsz arcp contract float %2092, %2103, !spirv.Decorations !881		; visa id: 2571
  %2105 = add i64 %.in, %532		; visa id: 2572
  %2106 = inttoptr i64 %2105 to float addrspace(4)*		; visa id: 2573
  %2107 = addrspacecast float addrspace(4)* %2106 to float addrspace(1)*		; visa id: 2573
  store float %2104, float addrspace(1)* %2107, align 4		; visa id: 2574
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2575

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4.._crit_edge70.1.4_crit_edge, %2097, %2093
; BB291 :
  br i1 %181, label %2108, label %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2576

._crit_edge70.1.4.._crit_edge70.2.4_crit_edge:    ; preds = %._crit_edge70.1.4
; BB:
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2108:                                             ; preds = %._crit_edge70.1.4
; BB293 :
  %2109 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !881		; visa id: 2578
  br i1 %81, label %2114, label %2110, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2579

2110:                                             ; preds = %2108
; BB294 :
  %2111 = add i64 %.in, %534		; visa id: 2581
  %2112 = inttoptr i64 %2111 to float addrspace(4)*		; visa id: 2582
  %2113 = addrspacecast float addrspace(4)* %2112 to float addrspace(1)*		; visa id: 2582
  store float %2109, float addrspace(1)* %2113, align 4		; visa id: 2583
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2584

2114:                                             ; preds = %2108
; BB295 :
  %2115 = add i64 %.in3821, %sink_3841		; visa id: 2586
  %2116 = add i64 %2115, %sink_3836		; visa id: 2587
  %2117 = inttoptr i64 %2116 to float addrspace(4)*		; visa id: 2588
  %2118 = addrspacecast float addrspace(4)* %2117 to float addrspace(1)*		; visa id: 2588
  %2119 = load float, float addrspace(1)* %2118, align 4		; visa id: 2589
  %2120 = fmul reassoc nsz arcp contract float %2119, %4, !spirv.Decorations !881		; visa id: 2590
  %2121 = fadd reassoc nsz arcp contract float %2109, %2120, !spirv.Decorations !881		; visa id: 2591
  %2122 = add i64 %.in, %534		; visa id: 2592
  %2123 = inttoptr i64 %2122 to float addrspace(4)*		; visa id: 2593
  %2124 = addrspacecast float addrspace(4)* %2123 to float addrspace(1)*		; visa id: 2593
  store float %2121, float addrspace(1)* %2124, align 4		; visa id: 2594
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2595

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, %2114, %2110
; BB296 :
  br i1 %184, label %2125, label %._crit_edge70.2.4..preheader1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2596

._crit_edge70.2.4..preheader1.4_crit_edge:        ; preds = %._crit_edge70.2.4
; BB:
  br label %.preheader1.4, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2125:                                             ; preds = %._crit_edge70.2.4
; BB298 :
  %2126 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !881		; visa id: 2598
  br i1 %81, label %2131, label %2127, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2599

2127:                                             ; preds = %2125
; BB299 :
  %2128 = add i64 %.in, %536		; visa id: 2601
  %2129 = inttoptr i64 %2128 to float addrspace(4)*		; visa id: 2602
  %2130 = addrspacecast float addrspace(4)* %2129 to float addrspace(1)*		; visa id: 2602
  store float %2126, float addrspace(1)* %2130, align 4		; visa id: 2603
  br label %.preheader1.4, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2604

2131:                                             ; preds = %2125
; BB300 :
  %2132 = add i64 %.in3821, %sink_3840		; visa id: 2606
  %2133 = add i64 %2132, %sink_3836		; visa id: 2607
  %2134 = inttoptr i64 %2133 to float addrspace(4)*		; visa id: 2608
  %2135 = addrspacecast float addrspace(4)* %2134 to float addrspace(1)*		; visa id: 2608
  %2136 = load float, float addrspace(1)* %2135, align 4		; visa id: 2609
  %2137 = fmul reassoc nsz arcp contract float %2136, %4, !spirv.Decorations !881		; visa id: 2610
  %2138 = fadd reassoc nsz arcp contract float %2126, %2137, !spirv.Decorations !881		; visa id: 2611
  %2139 = add i64 %.in, %536		; visa id: 2612
  %2140 = inttoptr i64 %2139 to float addrspace(4)*		; visa id: 2613
  %2141 = addrspacecast float addrspace(4)* %2140 to float addrspace(1)*		; visa id: 2613
  store float %2138, float addrspace(1)* %2141, align 4		; visa id: 2614
  br label %.preheader1.4, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2615

.preheader1.4:                                    ; preds = %._crit_edge70.2.4..preheader1.4_crit_edge, %2131, %2127
; BB301 :
  %sink_3835 = shl nsw i64 %537, 2		; visa id: 2616
  br i1 %188, label %2142, label %.preheader1.4.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2617

.preheader1.4.._crit_edge70.5_crit_edge:          ; preds = %.preheader1.4
; BB:
  br label %._crit_edge70.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2142:                                             ; preds = %.preheader1.4
; BB303 :
  %2143 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !881		; visa id: 2619
  br i1 %81, label %2148, label %2144, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2620

2144:                                             ; preds = %2142
; BB304 :
  %2145 = add i64 %.in, %539		; visa id: 2622
  %2146 = inttoptr i64 %2145 to float addrspace(4)*		; visa id: 2623
  %2147 = addrspacecast float addrspace(4)* %2146 to float addrspace(1)*		; visa id: 2623
  store float %2143, float addrspace(1)* %2147, align 4		; visa id: 2624
  br label %._crit_edge70.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2625

2148:                                             ; preds = %2142
; BB305 :
  %2149 = add i64 %.in3821, %sink_3844		; visa id: 2627
  %2150 = add i64 %2149, %sink_3835		; visa id: 2628
  %2151 = inttoptr i64 %2150 to float addrspace(4)*		; visa id: 2629
  %2152 = addrspacecast float addrspace(4)* %2151 to float addrspace(1)*		; visa id: 2629
  %2153 = load float, float addrspace(1)* %2152, align 4		; visa id: 2630
  %2154 = fmul reassoc nsz arcp contract float %2153, %4, !spirv.Decorations !881		; visa id: 2631
  %2155 = fadd reassoc nsz arcp contract float %2143, %2154, !spirv.Decorations !881		; visa id: 2632
  %2156 = add i64 %.in, %539		; visa id: 2633
  %2157 = inttoptr i64 %2156 to float addrspace(4)*		; visa id: 2634
  %2158 = addrspacecast float addrspace(4)* %2157 to float addrspace(1)*		; visa id: 2634
  store float %2155, float addrspace(1)* %2158, align 4		; visa id: 2635
  br label %._crit_edge70.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2636

._crit_edge70.5:                                  ; preds = %.preheader1.4.._crit_edge70.5_crit_edge, %2148, %2144
; BB306 :
  br i1 %191, label %2159, label %._crit_edge70.5.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2637

._crit_edge70.5.._crit_edge70.1.5_crit_edge:      ; preds = %._crit_edge70.5
; BB:
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2159:                                             ; preds = %._crit_edge70.5
; BB308 :
  %2160 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !881		; visa id: 2639
  br i1 %81, label %2165, label %2161, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2640

2161:                                             ; preds = %2159
; BB309 :
  %2162 = add i64 %.in, %541		; visa id: 2642
  %2163 = inttoptr i64 %2162 to float addrspace(4)*		; visa id: 2643
  %2164 = addrspacecast float addrspace(4)* %2163 to float addrspace(1)*		; visa id: 2643
  store float %2160, float addrspace(1)* %2164, align 4		; visa id: 2644
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2645

2165:                                             ; preds = %2159
; BB310 :
  %2166 = add i64 %.in3821, %sink_3842		; visa id: 2647
  %2167 = add i64 %2166, %sink_3835		; visa id: 2648
  %2168 = inttoptr i64 %2167 to float addrspace(4)*		; visa id: 2649
  %2169 = addrspacecast float addrspace(4)* %2168 to float addrspace(1)*		; visa id: 2649
  %2170 = load float, float addrspace(1)* %2169, align 4		; visa id: 2650
  %2171 = fmul reassoc nsz arcp contract float %2170, %4, !spirv.Decorations !881		; visa id: 2651
  %2172 = fadd reassoc nsz arcp contract float %2160, %2171, !spirv.Decorations !881		; visa id: 2652
  %2173 = add i64 %.in, %541		; visa id: 2653
  %2174 = inttoptr i64 %2173 to float addrspace(4)*		; visa id: 2654
  %2175 = addrspacecast float addrspace(4)* %2174 to float addrspace(1)*		; visa id: 2654
  store float %2172, float addrspace(1)* %2175, align 4		; visa id: 2655
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2656

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5.._crit_edge70.1.5_crit_edge, %2165, %2161
; BB311 :
  br i1 %194, label %2176, label %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2657

._crit_edge70.1.5.._crit_edge70.2.5_crit_edge:    ; preds = %._crit_edge70.1.5
; BB:
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2176:                                             ; preds = %._crit_edge70.1.5
; BB313 :
  %2177 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !881		; visa id: 2659
  br i1 %81, label %2182, label %2178, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2660

2178:                                             ; preds = %2176
; BB314 :
  %2179 = add i64 %.in, %543		; visa id: 2662
  %2180 = inttoptr i64 %2179 to float addrspace(4)*		; visa id: 2663
  %2181 = addrspacecast float addrspace(4)* %2180 to float addrspace(1)*		; visa id: 2663
  store float %2177, float addrspace(1)* %2181, align 4		; visa id: 2664
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2665

2182:                                             ; preds = %2176
; BB315 :
  %2183 = add i64 %.in3821, %sink_3841		; visa id: 2667
  %2184 = add i64 %2183, %sink_3835		; visa id: 2668
  %2185 = inttoptr i64 %2184 to float addrspace(4)*		; visa id: 2669
  %2186 = addrspacecast float addrspace(4)* %2185 to float addrspace(1)*		; visa id: 2669
  %2187 = load float, float addrspace(1)* %2186, align 4		; visa id: 2670
  %2188 = fmul reassoc nsz arcp contract float %2187, %4, !spirv.Decorations !881		; visa id: 2671
  %2189 = fadd reassoc nsz arcp contract float %2177, %2188, !spirv.Decorations !881		; visa id: 2672
  %2190 = add i64 %.in, %543		; visa id: 2673
  %2191 = inttoptr i64 %2190 to float addrspace(4)*		; visa id: 2674
  %2192 = addrspacecast float addrspace(4)* %2191 to float addrspace(1)*		; visa id: 2674
  store float %2189, float addrspace(1)* %2192, align 4		; visa id: 2675
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2676

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, %2182, %2178
; BB316 :
  br i1 %197, label %2193, label %._crit_edge70.2.5..preheader1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2677

._crit_edge70.2.5..preheader1.5_crit_edge:        ; preds = %._crit_edge70.2.5
; BB:
  br label %.preheader1.5, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2193:                                             ; preds = %._crit_edge70.2.5
; BB318 :
  %2194 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !881		; visa id: 2679
  br i1 %81, label %2199, label %2195, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2680

2195:                                             ; preds = %2193
; BB319 :
  %2196 = add i64 %.in, %545		; visa id: 2682
  %2197 = inttoptr i64 %2196 to float addrspace(4)*		; visa id: 2683
  %2198 = addrspacecast float addrspace(4)* %2197 to float addrspace(1)*		; visa id: 2683
  store float %2194, float addrspace(1)* %2198, align 4		; visa id: 2684
  br label %.preheader1.5, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2685

2199:                                             ; preds = %2193
; BB320 :
  %2200 = add i64 %.in3821, %sink_3840		; visa id: 2687
  %2201 = add i64 %2200, %sink_3835		; visa id: 2688
  %2202 = inttoptr i64 %2201 to float addrspace(4)*		; visa id: 2689
  %2203 = addrspacecast float addrspace(4)* %2202 to float addrspace(1)*		; visa id: 2689
  %2204 = load float, float addrspace(1)* %2203, align 4		; visa id: 2690
  %2205 = fmul reassoc nsz arcp contract float %2204, %4, !spirv.Decorations !881		; visa id: 2691
  %2206 = fadd reassoc nsz arcp contract float %2194, %2205, !spirv.Decorations !881		; visa id: 2692
  %2207 = add i64 %.in, %545		; visa id: 2693
  %2208 = inttoptr i64 %2207 to float addrspace(4)*		; visa id: 2694
  %2209 = addrspacecast float addrspace(4)* %2208 to float addrspace(1)*		; visa id: 2694
  store float %2206, float addrspace(1)* %2209, align 4		; visa id: 2695
  br label %.preheader1.5, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2696

.preheader1.5:                                    ; preds = %._crit_edge70.2.5..preheader1.5_crit_edge, %2199, %2195
; BB321 :
  %sink_3834 = shl nsw i64 %546, 2		; visa id: 2697
  br i1 %201, label %2210, label %.preheader1.5.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2698

.preheader1.5.._crit_edge70.6_crit_edge:          ; preds = %.preheader1.5
; BB:
  br label %._crit_edge70.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2210:                                             ; preds = %.preheader1.5
; BB323 :
  %2211 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !881		; visa id: 2700
  br i1 %81, label %2216, label %2212, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2701

2212:                                             ; preds = %2210
; BB324 :
  %2213 = add i64 %.in, %548		; visa id: 2703
  %2214 = inttoptr i64 %2213 to float addrspace(4)*		; visa id: 2704
  %2215 = addrspacecast float addrspace(4)* %2214 to float addrspace(1)*		; visa id: 2704
  store float %2211, float addrspace(1)* %2215, align 4		; visa id: 2705
  br label %._crit_edge70.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2706

2216:                                             ; preds = %2210
; BB325 :
  %2217 = add i64 %.in3821, %sink_3844		; visa id: 2708
  %2218 = add i64 %2217, %sink_3834		; visa id: 2709
  %2219 = inttoptr i64 %2218 to float addrspace(4)*		; visa id: 2710
  %2220 = addrspacecast float addrspace(4)* %2219 to float addrspace(1)*		; visa id: 2710
  %2221 = load float, float addrspace(1)* %2220, align 4		; visa id: 2711
  %2222 = fmul reassoc nsz arcp contract float %2221, %4, !spirv.Decorations !881		; visa id: 2712
  %2223 = fadd reassoc nsz arcp contract float %2211, %2222, !spirv.Decorations !881		; visa id: 2713
  %2224 = add i64 %.in, %548		; visa id: 2714
  %2225 = inttoptr i64 %2224 to float addrspace(4)*		; visa id: 2715
  %2226 = addrspacecast float addrspace(4)* %2225 to float addrspace(1)*		; visa id: 2715
  store float %2223, float addrspace(1)* %2226, align 4		; visa id: 2716
  br label %._crit_edge70.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2717

._crit_edge70.6:                                  ; preds = %.preheader1.5.._crit_edge70.6_crit_edge, %2216, %2212
; BB326 :
  br i1 %204, label %2227, label %._crit_edge70.6.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2718

._crit_edge70.6.._crit_edge70.1.6_crit_edge:      ; preds = %._crit_edge70.6
; BB:
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2227:                                             ; preds = %._crit_edge70.6
; BB328 :
  %2228 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !881		; visa id: 2720
  br i1 %81, label %2233, label %2229, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2721

2229:                                             ; preds = %2227
; BB329 :
  %2230 = add i64 %.in, %550		; visa id: 2723
  %2231 = inttoptr i64 %2230 to float addrspace(4)*		; visa id: 2724
  %2232 = addrspacecast float addrspace(4)* %2231 to float addrspace(1)*		; visa id: 2724
  store float %2228, float addrspace(1)* %2232, align 4		; visa id: 2725
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2726

2233:                                             ; preds = %2227
; BB330 :
  %2234 = add i64 %.in3821, %sink_3842		; visa id: 2728
  %2235 = add i64 %2234, %sink_3834		; visa id: 2729
  %2236 = inttoptr i64 %2235 to float addrspace(4)*		; visa id: 2730
  %2237 = addrspacecast float addrspace(4)* %2236 to float addrspace(1)*		; visa id: 2730
  %2238 = load float, float addrspace(1)* %2237, align 4		; visa id: 2731
  %2239 = fmul reassoc nsz arcp contract float %2238, %4, !spirv.Decorations !881		; visa id: 2732
  %2240 = fadd reassoc nsz arcp contract float %2228, %2239, !spirv.Decorations !881		; visa id: 2733
  %2241 = add i64 %.in, %550		; visa id: 2734
  %2242 = inttoptr i64 %2241 to float addrspace(4)*		; visa id: 2735
  %2243 = addrspacecast float addrspace(4)* %2242 to float addrspace(1)*		; visa id: 2735
  store float %2240, float addrspace(1)* %2243, align 4		; visa id: 2736
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2737

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6.._crit_edge70.1.6_crit_edge, %2233, %2229
; BB331 :
  br i1 %207, label %2244, label %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2738

._crit_edge70.1.6.._crit_edge70.2.6_crit_edge:    ; preds = %._crit_edge70.1.6
; BB:
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2244:                                             ; preds = %._crit_edge70.1.6
; BB333 :
  %2245 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !881		; visa id: 2740
  br i1 %81, label %2250, label %2246, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2741

2246:                                             ; preds = %2244
; BB334 :
  %2247 = add i64 %.in, %552		; visa id: 2743
  %2248 = inttoptr i64 %2247 to float addrspace(4)*		; visa id: 2744
  %2249 = addrspacecast float addrspace(4)* %2248 to float addrspace(1)*		; visa id: 2744
  store float %2245, float addrspace(1)* %2249, align 4		; visa id: 2745
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2746

2250:                                             ; preds = %2244
; BB335 :
  %2251 = add i64 %.in3821, %sink_3841		; visa id: 2748
  %2252 = add i64 %2251, %sink_3834		; visa id: 2749
  %2253 = inttoptr i64 %2252 to float addrspace(4)*		; visa id: 2750
  %2254 = addrspacecast float addrspace(4)* %2253 to float addrspace(1)*		; visa id: 2750
  %2255 = load float, float addrspace(1)* %2254, align 4		; visa id: 2751
  %2256 = fmul reassoc nsz arcp contract float %2255, %4, !spirv.Decorations !881		; visa id: 2752
  %2257 = fadd reassoc nsz arcp contract float %2245, %2256, !spirv.Decorations !881		; visa id: 2753
  %2258 = add i64 %.in, %552		; visa id: 2754
  %2259 = inttoptr i64 %2258 to float addrspace(4)*		; visa id: 2755
  %2260 = addrspacecast float addrspace(4)* %2259 to float addrspace(1)*		; visa id: 2755
  store float %2257, float addrspace(1)* %2260, align 4		; visa id: 2756
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2757

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, %2250, %2246
; BB336 :
  br i1 %210, label %2261, label %._crit_edge70.2.6..preheader1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2758

._crit_edge70.2.6..preheader1.6_crit_edge:        ; preds = %._crit_edge70.2.6
; BB:
  br label %.preheader1.6, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2261:                                             ; preds = %._crit_edge70.2.6
; BB338 :
  %2262 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !881		; visa id: 2760
  br i1 %81, label %2267, label %2263, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2761

2263:                                             ; preds = %2261
; BB339 :
  %2264 = add i64 %.in, %554		; visa id: 2763
  %2265 = inttoptr i64 %2264 to float addrspace(4)*		; visa id: 2764
  %2266 = addrspacecast float addrspace(4)* %2265 to float addrspace(1)*		; visa id: 2764
  store float %2262, float addrspace(1)* %2266, align 4		; visa id: 2765
  br label %.preheader1.6, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2766

2267:                                             ; preds = %2261
; BB340 :
  %2268 = add i64 %.in3821, %sink_3840		; visa id: 2768
  %2269 = add i64 %2268, %sink_3834		; visa id: 2769
  %2270 = inttoptr i64 %2269 to float addrspace(4)*		; visa id: 2770
  %2271 = addrspacecast float addrspace(4)* %2270 to float addrspace(1)*		; visa id: 2770
  %2272 = load float, float addrspace(1)* %2271, align 4		; visa id: 2771
  %2273 = fmul reassoc nsz arcp contract float %2272, %4, !spirv.Decorations !881		; visa id: 2772
  %2274 = fadd reassoc nsz arcp contract float %2262, %2273, !spirv.Decorations !881		; visa id: 2773
  %2275 = add i64 %.in, %554		; visa id: 2774
  %2276 = inttoptr i64 %2275 to float addrspace(4)*		; visa id: 2775
  %2277 = addrspacecast float addrspace(4)* %2276 to float addrspace(1)*		; visa id: 2775
  store float %2274, float addrspace(1)* %2277, align 4		; visa id: 2776
  br label %.preheader1.6, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2777

.preheader1.6:                                    ; preds = %._crit_edge70.2.6..preheader1.6_crit_edge, %2267, %2263
; BB341 :
  %sink_3833 = shl nsw i64 %555, 2		; visa id: 2778
  br i1 %214, label %2278, label %.preheader1.6.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2779

.preheader1.6.._crit_edge70.7_crit_edge:          ; preds = %.preheader1.6
; BB:
  br label %._crit_edge70.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2278:                                             ; preds = %.preheader1.6
; BB343 :
  %2279 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !881		; visa id: 2781
  br i1 %81, label %2284, label %2280, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2782

2280:                                             ; preds = %2278
; BB344 :
  %2281 = add i64 %.in, %557		; visa id: 2784
  %2282 = inttoptr i64 %2281 to float addrspace(4)*		; visa id: 2785
  %2283 = addrspacecast float addrspace(4)* %2282 to float addrspace(1)*		; visa id: 2785
  store float %2279, float addrspace(1)* %2283, align 4		; visa id: 2786
  br label %._crit_edge70.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2787

2284:                                             ; preds = %2278
; BB345 :
  %2285 = add i64 %.in3821, %sink_3844		; visa id: 2789
  %2286 = add i64 %2285, %sink_3833		; visa id: 2790
  %2287 = inttoptr i64 %2286 to float addrspace(4)*		; visa id: 2791
  %2288 = addrspacecast float addrspace(4)* %2287 to float addrspace(1)*		; visa id: 2791
  %2289 = load float, float addrspace(1)* %2288, align 4		; visa id: 2792
  %2290 = fmul reassoc nsz arcp contract float %2289, %4, !spirv.Decorations !881		; visa id: 2793
  %2291 = fadd reassoc nsz arcp contract float %2279, %2290, !spirv.Decorations !881		; visa id: 2794
  %2292 = add i64 %.in, %557		; visa id: 2795
  %2293 = inttoptr i64 %2292 to float addrspace(4)*		; visa id: 2796
  %2294 = addrspacecast float addrspace(4)* %2293 to float addrspace(1)*		; visa id: 2796
  store float %2291, float addrspace(1)* %2294, align 4		; visa id: 2797
  br label %._crit_edge70.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2798

._crit_edge70.7:                                  ; preds = %.preheader1.6.._crit_edge70.7_crit_edge, %2284, %2280
; BB346 :
  br i1 %217, label %2295, label %._crit_edge70.7.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2799

._crit_edge70.7.._crit_edge70.1.7_crit_edge:      ; preds = %._crit_edge70.7
; BB:
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2295:                                             ; preds = %._crit_edge70.7
; BB348 :
  %2296 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !881		; visa id: 2801
  br i1 %81, label %2301, label %2297, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2802

2297:                                             ; preds = %2295
; BB349 :
  %2298 = add i64 %.in, %559		; visa id: 2804
  %2299 = inttoptr i64 %2298 to float addrspace(4)*		; visa id: 2805
  %2300 = addrspacecast float addrspace(4)* %2299 to float addrspace(1)*		; visa id: 2805
  store float %2296, float addrspace(1)* %2300, align 4		; visa id: 2806
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2807

2301:                                             ; preds = %2295
; BB350 :
  %2302 = add i64 %.in3821, %sink_3842		; visa id: 2809
  %2303 = add i64 %2302, %sink_3833		; visa id: 2810
  %2304 = inttoptr i64 %2303 to float addrspace(4)*		; visa id: 2811
  %2305 = addrspacecast float addrspace(4)* %2304 to float addrspace(1)*		; visa id: 2811
  %2306 = load float, float addrspace(1)* %2305, align 4		; visa id: 2812
  %2307 = fmul reassoc nsz arcp contract float %2306, %4, !spirv.Decorations !881		; visa id: 2813
  %2308 = fadd reassoc nsz arcp contract float %2296, %2307, !spirv.Decorations !881		; visa id: 2814
  %2309 = add i64 %.in, %559		; visa id: 2815
  %2310 = inttoptr i64 %2309 to float addrspace(4)*		; visa id: 2816
  %2311 = addrspacecast float addrspace(4)* %2310 to float addrspace(1)*		; visa id: 2816
  store float %2308, float addrspace(1)* %2311, align 4		; visa id: 2817
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2818

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7.._crit_edge70.1.7_crit_edge, %2301, %2297
; BB351 :
  br i1 %220, label %2312, label %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2819

._crit_edge70.1.7.._crit_edge70.2.7_crit_edge:    ; preds = %._crit_edge70.1.7
; BB:
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2312:                                             ; preds = %._crit_edge70.1.7
; BB353 :
  %2313 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !881		; visa id: 2821
  br i1 %81, label %2318, label %2314, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2822

2314:                                             ; preds = %2312
; BB354 :
  %2315 = add i64 %.in, %561		; visa id: 2824
  %2316 = inttoptr i64 %2315 to float addrspace(4)*		; visa id: 2825
  %2317 = addrspacecast float addrspace(4)* %2316 to float addrspace(1)*		; visa id: 2825
  store float %2313, float addrspace(1)* %2317, align 4		; visa id: 2826
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2827

2318:                                             ; preds = %2312
; BB355 :
  %2319 = add i64 %.in3821, %sink_3841		; visa id: 2829
  %2320 = add i64 %2319, %sink_3833		; visa id: 2830
  %2321 = inttoptr i64 %2320 to float addrspace(4)*		; visa id: 2831
  %2322 = addrspacecast float addrspace(4)* %2321 to float addrspace(1)*		; visa id: 2831
  %2323 = load float, float addrspace(1)* %2322, align 4		; visa id: 2832
  %2324 = fmul reassoc nsz arcp contract float %2323, %4, !spirv.Decorations !881		; visa id: 2833
  %2325 = fadd reassoc nsz arcp contract float %2313, %2324, !spirv.Decorations !881		; visa id: 2834
  %2326 = add i64 %.in, %561		; visa id: 2835
  %2327 = inttoptr i64 %2326 to float addrspace(4)*		; visa id: 2836
  %2328 = addrspacecast float addrspace(4)* %2327 to float addrspace(1)*		; visa id: 2836
  store float %2325, float addrspace(1)* %2328, align 4		; visa id: 2837
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2838

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, %2318, %2314
; BB356 :
  br i1 %223, label %2329, label %._crit_edge70.2.7..preheader1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2839

._crit_edge70.2.7..preheader1.7_crit_edge:        ; preds = %._crit_edge70.2.7
; BB:
  br label %.preheader1.7, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2329:                                             ; preds = %._crit_edge70.2.7
; BB358 :
  %2330 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !881		; visa id: 2841
  br i1 %81, label %2335, label %2331, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2842

2331:                                             ; preds = %2329
; BB359 :
  %2332 = add i64 %.in, %563		; visa id: 2844
  %2333 = inttoptr i64 %2332 to float addrspace(4)*		; visa id: 2845
  %2334 = addrspacecast float addrspace(4)* %2333 to float addrspace(1)*		; visa id: 2845
  store float %2330, float addrspace(1)* %2334, align 4		; visa id: 2846
  br label %.preheader1.7, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2847

2335:                                             ; preds = %2329
; BB360 :
  %2336 = add i64 %.in3821, %sink_3840		; visa id: 2849
  %2337 = add i64 %2336, %sink_3833		; visa id: 2850
  %2338 = inttoptr i64 %2337 to float addrspace(4)*		; visa id: 2851
  %2339 = addrspacecast float addrspace(4)* %2338 to float addrspace(1)*		; visa id: 2851
  %2340 = load float, float addrspace(1)* %2339, align 4		; visa id: 2852
  %2341 = fmul reassoc nsz arcp contract float %2340, %4, !spirv.Decorations !881		; visa id: 2853
  %2342 = fadd reassoc nsz arcp contract float %2330, %2341, !spirv.Decorations !881		; visa id: 2854
  %2343 = add i64 %.in, %563		; visa id: 2855
  %2344 = inttoptr i64 %2343 to float addrspace(4)*		; visa id: 2856
  %2345 = addrspacecast float addrspace(4)* %2344 to float addrspace(1)*		; visa id: 2856
  store float %2342, float addrspace(1)* %2345, align 4		; visa id: 2857
  br label %.preheader1.7, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2858

.preheader1.7:                                    ; preds = %._crit_edge70.2.7..preheader1.7_crit_edge, %2335, %2331
; BB361 :
  %sink_3832 = shl nsw i64 %564, 2		; visa id: 2859
  br i1 %227, label %2346, label %.preheader1.7.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2860

.preheader1.7.._crit_edge70.8_crit_edge:          ; preds = %.preheader1.7
; BB:
  br label %._crit_edge70.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2346:                                             ; preds = %.preheader1.7
; BB363 :
  %2347 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !881		; visa id: 2862
  br i1 %81, label %2352, label %2348, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2863

2348:                                             ; preds = %2346
; BB364 :
  %2349 = add i64 %.in, %566		; visa id: 2865
  %2350 = inttoptr i64 %2349 to float addrspace(4)*		; visa id: 2866
  %2351 = addrspacecast float addrspace(4)* %2350 to float addrspace(1)*		; visa id: 2866
  store float %2347, float addrspace(1)* %2351, align 4		; visa id: 2867
  br label %._crit_edge70.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2868

2352:                                             ; preds = %2346
; BB365 :
  %2353 = add i64 %.in3821, %sink_3844		; visa id: 2870
  %2354 = add i64 %2353, %sink_3832		; visa id: 2871
  %2355 = inttoptr i64 %2354 to float addrspace(4)*		; visa id: 2872
  %2356 = addrspacecast float addrspace(4)* %2355 to float addrspace(1)*		; visa id: 2872
  %2357 = load float, float addrspace(1)* %2356, align 4		; visa id: 2873
  %2358 = fmul reassoc nsz arcp contract float %2357, %4, !spirv.Decorations !881		; visa id: 2874
  %2359 = fadd reassoc nsz arcp contract float %2347, %2358, !spirv.Decorations !881		; visa id: 2875
  %2360 = add i64 %.in, %566		; visa id: 2876
  %2361 = inttoptr i64 %2360 to float addrspace(4)*		; visa id: 2877
  %2362 = addrspacecast float addrspace(4)* %2361 to float addrspace(1)*		; visa id: 2877
  store float %2359, float addrspace(1)* %2362, align 4		; visa id: 2878
  br label %._crit_edge70.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2879

._crit_edge70.8:                                  ; preds = %.preheader1.7.._crit_edge70.8_crit_edge, %2352, %2348
; BB366 :
  br i1 %230, label %2363, label %._crit_edge70.8.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2880

._crit_edge70.8.._crit_edge70.1.8_crit_edge:      ; preds = %._crit_edge70.8
; BB:
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2363:                                             ; preds = %._crit_edge70.8
; BB368 :
  %2364 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !881		; visa id: 2882
  br i1 %81, label %2369, label %2365, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2883

2365:                                             ; preds = %2363
; BB369 :
  %2366 = add i64 %.in, %568		; visa id: 2885
  %2367 = inttoptr i64 %2366 to float addrspace(4)*		; visa id: 2886
  %2368 = addrspacecast float addrspace(4)* %2367 to float addrspace(1)*		; visa id: 2886
  store float %2364, float addrspace(1)* %2368, align 4		; visa id: 2887
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2888

2369:                                             ; preds = %2363
; BB370 :
  %2370 = add i64 %.in3821, %sink_3842		; visa id: 2890
  %2371 = add i64 %2370, %sink_3832		; visa id: 2891
  %2372 = inttoptr i64 %2371 to float addrspace(4)*		; visa id: 2892
  %2373 = addrspacecast float addrspace(4)* %2372 to float addrspace(1)*		; visa id: 2892
  %2374 = load float, float addrspace(1)* %2373, align 4		; visa id: 2893
  %2375 = fmul reassoc nsz arcp contract float %2374, %4, !spirv.Decorations !881		; visa id: 2894
  %2376 = fadd reassoc nsz arcp contract float %2364, %2375, !spirv.Decorations !881		; visa id: 2895
  %2377 = add i64 %.in, %568		; visa id: 2896
  %2378 = inttoptr i64 %2377 to float addrspace(4)*		; visa id: 2897
  %2379 = addrspacecast float addrspace(4)* %2378 to float addrspace(1)*		; visa id: 2897
  store float %2376, float addrspace(1)* %2379, align 4		; visa id: 2898
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2899

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8.._crit_edge70.1.8_crit_edge, %2369, %2365
; BB371 :
  br i1 %233, label %2380, label %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2900

._crit_edge70.1.8.._crit_edge70.2.8_crit_edge:    ; preds = %._crit_edge70.1.8
; BB:
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2380:                                             ; preds = %._crit_edge70.1.8
; BB373 :
  %2381 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !881		; visa id: 2902
  br i1 %81, label %2386, label %2382, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2903

2382:                                             ; preds = %2380
; BB374 :
  %2383 = add i64 %.in, %570		; visa id: 2905
  %2384 = inttoptr i64 %2383 to float addrspace(4)*		; visa id: 2906
  %2385 = addrspacecast float addrspace(4)* %2384 to float addrspace(1)*		; visa id: 2906
  store float %2381, float addrspace(1)* %2385, align 4		; visa id: 2907
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2908

2386:                                             ; preds = %2380
; BB375 :
  %2387 = add i64 %.in3821, %sink_3841		; visa id: 2910
  %2388 = add i64 %2387, %sink_3832		; visa id: 2911
  %2389 = inttoptr i64 %2388 to float addrspace(4)*		; visa id: 2912
  %2390 = addrspacecast float addrspace(4)* %2389 to float addrspace(1)*		; visa id: 2912
  %2391 = load float, float addrspace(1)* %2390, align 4		; visa id: 2913
  %2392 = fmul reassoc nsz arcp contract float %2391, %4, !spirv.Decorations !881		; visa id: 2914
  %2393 = fadd reassoc nsz arcp contract float %2381, %2392, !spirv.Decorations !881		; visa id: 2915
  %2394 = add i64 %.in, %570		; visa id: 2916
  %2395 = inttoptr i64 %2394 to float addrspace(4)*		; visa id: 2917
  %2396 = addrspacecast float addrspace(4)* %2395 to float addrspace(1)*		; visa id: 2917
  store float %2393, float addrspace(1)* %2396, align 4		; visa id: 2918
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2919

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, %2386, %2382
; BB376 :
  br i1 %236, label %2397, label %._crit_edge70.2.8..preheader1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2920

._crit_edge70.2.8..preheader1.8_crit_edge:        ; preds = %._crit_edge70.2.8
; BB:
  br label %.preheader1.8, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2397:                                             ; preds = %._crit_edge70.2.8
; BB378 :
  %2398 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !881		; visa id: 2922
  br i1 %81, label %2403, label %2399, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2923

2399:                                             ; preds = %2397
; BB379 :
  %2400 = add i64 %.in, %572		; visa id: 2925
  %2401 = inttoptr i64 %2400 to float addrspace(4)*		; visa id: 2926
  %2402 = addrspacecast float addrspace(4)* %2401 to float addrspace(1)*		; visa id: 2926
  store float %2398, float addrspace(1)* %2402, align 4		; visa id: 2927
  br label %.preheader1.8, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2928

2403:                                             ; preds = %2397
; BB380 :
  %2404 = add i64 %.in3821, %sink_3840		; visa id: 2930
  %2405 = add i64 %2404, %sink_3832		; visa id: 2931
  %2406 = inttoptr i64 %2405 to float addrspace(4)*		; visa id: 2932
  %2407 = addrspacecast float addrspace(4)* %2406 to float addrspace(1)*		; visa id: 2932
  %2408 = load float, float addrspace(1)* %2407, align 4		; visa id: 2933
  %2409 = fmul reassoc nsz arcp contract float %2408, %4, !spirv.Decorations !881		; visa id: 2934
  %2410 = fadd reassoc nsz arcp contract float %2398, %2409, !spirv.Decorations !881		; visa id: 2935
  %2411 = add i64 %.in, %572		; visa id: 2936
  %2412 = inttoptr i64 %2411 to float addrspace(4)*		; visa id: 2937
  %2413 = addrspacecast float addrspace(4)* %2412 to float addrspace(1)*		; visa id: 2937
  store float %2410, float addrspace(1)* %2413, align 4		; visa id: 2938
  br label %.preheader1.8, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2939

.preheader1.8:                                    ; preds = %._crit_edge70.2.8..preheader1.8_crit_edge, %2403, %2399
; BB381 :
  %sink_3831 = shl nsw i64 %573, 2		; visa id: 2940
  br i1 %240, label %2414, label %.preheader1.8.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2941

.preheader1.8.._crit_edge70.9_crit_edge:          ; preds = %.preheader1.8
; BB:
  br label %._crit_edge70.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2414:                                             ; preds = %.preheader1.8
; BB383 :
  %2415 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !881		; visa id: 2943
  br i1 %81, label %2420, label %2416, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2944

2416:                                             ; preds = %2414
; BB384 :
  %2417 = add i64 %.in, %575		; visa id: 2946
  %2418 = inttoptr i64 %2417 to float addrspace(4)*		; visa id: 2947
  %2419 = addrspacecast float addrspace(4)* %2418 to float addrspace(1)*		; visa id: 2947
  store float %2415, float addrspace(1)* %2419, align 4		; visa id: 2948
  br label %._crit_edge70.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2949

2420:                                             ; preds = %2414
; BB385 :
  %2421 = add i64 %.in3821, %sink_3844		; visa id: 2951
  %2422 = add i64 %2421, %sink_3831		; visa id: 2952
  %2423 = inttoptr i64 %2422 to float addrspace(4)*		; visa id: 2953
  %2424 = addrspacecast float addrspace(4)* %2423 to float addrspace(1)*		; visa id: 2953
  %2425 = load float, float addrspace(1)* %2424, align 4		; visa id: 2954
  %2426 = fmul reassoc nsz arcp contract float %2425, %4, !spirv.Decorations !881		; visa id: 2955
  %2427 = fadd reassoc nsz arcp contract float %2415, %2426, !spirv.Decorations !881		; visa id: 2956
  %2428 = add i64 %.in, %575		; visa id: 2957
  %2429 = inttoptr i64 %2428 to float addrspace(4)*		; visa id: 2958
  %2430 = addrspacecast float addrspace(4)* %2429 to float addrspace(1)*		; visa id: 2958
  store float %2427, float addrspace(1)* %2430, align 4		; visa id: 2959
  br label %._crit_edge70.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2960

._crit_edge70.9:                                  ; preds = %.preheader1.8.._crit_edge70.9_crit_edge, %2420, %2416
; BB386 :
  br i1 %243, label %2431, label %._crit_edge70.9.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2961

._crit_edge70.9.._crit_edge70.1.9_crit_edge:      ; preds = %._crit_edge70.9
; BB:
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2431:                                             ; preds = %._crit_edge70.9
; BB388 :
  %2432 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !881		; visa id: 2963
  br i1 %81, label %2437, label %2433, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2964

2433:                                             ; preds = %2431
; BB389 :
  %2434 = add i64 %.in, %577		; visa id: 2966
  %2435 = inttoptr i64 %2434 to float addrspace(4)*		; visa id: 2967
  %2436 = addrspacecast float addrspace(4)* %2435 to float addrspace(1)*		; visa id: 2967
  store float %2432, float addrspace(1)* %2436, align 4		; visa id: 2968
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2969

2437:                                             ; preds = %2431
; BB390 :
  %2438 = add i64 %.in3821, %sink_3842		; visa id: 2971
  %2439 = add i64 %2438, %sink_3831		; visa id: 2972
  %2440 = inttoptr i64 %2439 to float addrspace(4)*		; visa id: 2973
  %2441 = addrspacecast float addrspace(4)* %2440 to float addrspace(1)*		; visa id: 2973
  %2442 = load float, float addrspace(1)* %2441, align 4		; visa id: 2974
  %2443 = fmul reassoc nsz arcp contract float %2442, %4, !spirv.Decorations !881		; visa id: 2975
  %2444 = fadd reassoc nsz arcp contract float %2432, %2443, !spirv.Decorations !881		; visa id: 2976
  %2445 = add i64 %.in, %577		; visa id: 2977
  %2446 = inttoptr i64 %2445 to float addrspace(4)*		; visa id: 2978
  %2447 = addrspacecast float addrspace(4)* %2446 to float addrspace(1)*		; visa id: 2978
  store float %2444, float addrspace(1)* %2447, align 4		; visa id: 2979
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 2980

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9.._crit_edge70.1.9_crit_edge, %2437, %2433
; BB391 :
  br i1 %246, label %2448, label %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 2981

._crit_edge70.1.9.._crit_edge70.2.9_crit_edge:    ; preds = %._crit_edge70.1.9
; BB:
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2448:                                             ; preds = %._crit_edge70.1.9
; BB393 :
  %2449 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !881		; visa id: 2983
  br i1 %81, label %2454, label %2450, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 2984

2450:                                             ; preds = %2448
; BB394 :
  %2451 = add i64 %.in, %579		; visa id: 2986
  %2452 = inttoptr i64 %2451 to float addrspace(4)*		; visa id: 2987
  %2453 = addrspacecast float addrspace(4)* %2452 to float addrspace(1)*		; visa id: 2987
  store float %2449, float addrspace(1)* %2453, align 4		; visa id: 2988
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 2989

2454:                                             ; preds = %2448
; BB395 :
  %2455 = add i64 %.in3821, %sink_3841		; visa id: 2991
  %2456 = add i64 %2455, %sink_3831		; visa id: 2992
  %2457 = inttoptr i64 %2456 to float addrspace(4)*		; visa id: 2993
  %2458 = addrspacecast float addrspace(4)* %2457 to float addrspace(1)*		; visa id: 2993
  %2459 = load float, float addrspace(1)* %2458, align 4		; visa id: 2994
  %2460 = fmul reassoc nsz arcp contract float %2459, %4, !spirv.Decorations !881		; visa id: 2995
  %2461 = fadd reassoc nsz arcp contract float %2449, %2460, !spirv.Decorations !881		; visa id: 2996
  %2462 = add i64 %.in, %579		; visa id: 2997
  %2463 = inttoptr i64 %2462 to float addrspace(4)*		; visa id: 2998
  %2464 = addrspacecast float addrspace(4)* %2463 to float addrspace(1)*		; visa id: 2998
  store float %2461, float addrspace(1)* %2464, align 4		; visa id: 2999
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3000

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, %2454, %2450
; BB396 :
  br i1 %249, label %2465, label %._crit_edge70.2.9..preheader1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3001

._crit_edge70.2.9..preheader1.9_crit_edge:        ; preds = %._crit_edge70.2.9
; BB:
  br label %.preheader1.9, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2465:                                             ; preds = %._crit_edge70.2.9
; BB398 :
  %2466 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !881		; visa id: 3003
  br i1 %81, label %2471, label %2467, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3004

2467:                                             ; preds = %2465
; BB399 :
  %2468 = add i64 %.in, %581		; visa id: 3006
  %2469 = inttoptr i64 %2468 to float addrspace(4)*		; visa id: 3007
  %2470 = addrspacecast float addrspace(4)* %2469 to float addrspace(1)*		; visa id: 3007
  store float %2466, float addrspace(1)* %2470, align 4		; visa id: 3008
  br label %.preheader1.9, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3009

2471:                                             ; preds = %2465
; BB400 :
  %2472 = add i64 %.in3821, %sink_3840		; visa id: 3011
  %2473 = add i64 %2472, %sink_3831		; visa id: 3012
  %2474 = inttoptr i64 %2473 to float addrspace(4)*		; visa id: 3013
  %2475 = addrspacecast float addrspace(4)* %2474 to float addrspace(1)*		; visa id: 3013
  %2476 = load float, float addrspace(1)* %2475, align 4		; visa id: 3014
  %2477 = fmul reassoc nsz arcp contract float %2476, %4, !spirv.Decorations !881		; visa id: 3015
  %2478 = fadd reassoc nsz arcp contract float %2466, %2477, !spirv.Decorations !881		; visa id: 3016
  %2479 = add i64 %.in, %581		; visa id: 3017
  %2480 = inttoptr i64 %2479 to float addrspace(4)*		; visa id: 3018
  %2481 = addrspacecast float addrspace(4)* %2480 to float addrspace(1)*		; visa id: 3018
  store float %2478, float addrspace(1)* %2481, align 4		; visa id: 3019
  br label %.preheader1.9, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3020

.preheader1.9:                                    ; preds = %._crit_edge70.2.9..preheader1.9_crit_edge, %2471, %2467
; BB401 :
  %sink_3830 = shl nsw i64 %582, 2		; visa id: 3021
  br i1 %253, label %2482, label %.preheader1.9.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3022

.preheader1.9.._crit_edge70.10_crit_edge:         ; preds = %.preheader1.9
; BB:
  br label %._crit_edge70.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2482:                                             ; preds = %.preheader1.9
; BB403 :
  %2483 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !881		; visa id: 3024
  br i1 %81, label %2488, label %2484, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3025

2484:                                             ; preds = %2482
; BB404 :
  %2485 = add i64 %.in, %584		; visa id: 3027
  %2486 = inttoptr i64 %2485 to float addrspace(4)*		; visa id: 3028
  %2487 = addrspacecast float addrspace(4)* %2486 to float addrspace(1)*		; visa id: 3028
  store float %2483, float addrspace(1)* %2487, align 4		; visa id: 3029
  br label %._crit_edge70.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3030

2488:                                             ; preds = %2482
; BB405 :
  %2489 = add i64 %.in3821, %sink_3844		; visa id: 3032
  %2490 = add i64 %2489, %sink_3830		; visa id: 3033
  %2491 = inttoptr i64 %2490 to float addrspace(4)*		; visa id: 3034
  %2492 = addrspacecast float addrspace(4)* %2491 to float addrspace(1)*		; visa id: 3034
  %2493 = load float, float addrspace(1)* %2492, align 4		; visa id: 3035
  %2494 = fmul reassoc nsz arcp contract float %2493, %4, !spirv.Decorations !881		; visa id: 3036
  %2495 = fadd reassoc nsz arcp contract float %2483, %2494, !spirv.Decorations !881		; visa id: 3037
  %2496 = add i64 %.in, %584		; visa id: 3038
  %2497 = inttoptr i64 %2496 to float addrspace(4)*		; visa id: 3039
  %2498 = addrspacecast float addrspace(4)* %2497 to float addrspace(1)*		; visa id: 3039
  store float %2495, float addrspace(1)* %2498, align 4		; visa id: 3040
  br label %._crit_edge70.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3041

._crit_edge70.10:                                 ; preds = %.preheader1.9.._crit_edge70.10_crit_edge, %2488, %2484
; BB406 :
  br i1 %256, label %2499, label %._crit_edge70.10.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3042

._crit_edge70.10.._crit_edge70.1.10_crit_edge:    ; preds = %._crit_edge70.10
; BB:
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2499:                                             ; preds = %._crit_edge70.10
; BB408 :
  %2500 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !881		; visa id: 3044
  br i1 %81, label %2505, label %2501, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3045

2501:                                             ; preds = %2499
; BB409 :
  %2502 = add i64 %.in, %586		; visa id: 3047
  %2503 = inttoptr i64 %2502 to float addrspace(4)*		; visa id: 3048
  %2504 = addrspacecast float addrspace(4)* %2503 to float addrspace(1)*		; visa id: 3048
  store float %2500, float addrspace(1)* %2504, align 4		; visa id: 3049
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3050

2505:                                             ; preds = %2499
; BB410 :
  %2506 = add i64 %.in3821, %sink_3842		; visa id: 3052
  %2507 = add i64 %2506, %sink_3830		; visa id: 3053
  %2508 = inttoptr i64 %2507 to float addrspace(4)*		; visa id: 3054
  %2509 = addrspacecast float addrspace(4)* %2508 to float addrspace(1)*		; visa id: 3054
  %2510 = load float, float addrspace(1)* %2509, align 4		; visa id: 3055
  %2511 = fmul reassoc nsz arcp contract float %2510, %4, !spirv.Decorations !881		; visa id: 3056
  %2512 = fadd reassoc nsz arcp contract float %2500, %2511, !spirv.Decorations !881		; visa id: 3057
  %2513 = add i64 %.in, %586		; visa id: 3058
  %2514 = inttoptr i64 %2513 to float addrspace(4)*		; visa id: 3059
  %2515 = addrspacecast float addrspace(4)* %2514 to float addrspace(1)*		; visa id: 3059
  store float %2512, float addrspace(1)* %2515, align 4		; visa id: 3060
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3061

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10.._crit_edge70.1.10_crit_edge, %2505, %2501
; BB411 :
  br i1 %259, label %2516, label %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3062

._crit_edge70.1.10.._crit_edge70.2.10_crit_edge:  ; preds = %._crit_edge70.1.10
; BB:
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2516:                                             ; preds = %._crit_edge70.1.10
; BB413 :
  %2517 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !881		; visa id: 3064
  br i1 %81, label %2522, label %2518, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3065

2518:                                             ; preds = %2516
; BB414 :
  %2519 = add i64 %.in, %588		; visa id: 3067
  %2520 = inttoptr i64 %2519 to float addrspace(4)*		; visa id: 3068
  %2521 = addrspacecast float addrspace(4)* %2520 to float addrspace(1)*		; visa id: 3068
  store float %2517, float addrspace(1)* %2521, align 4		; visa id: 3069
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3070

2522:                                             ; preds = %2516
; BB415 :
  %2523 = add i64 %.in3821, %sink_3841		; visa id: 3072
  %2524 = add i64 %2523, %sink_3830		; visa id: 3073
  %2525 = inttoptr i64 %2524 to float addrspace(4)*		; visa id: 3074
  %2526 = addrspacecast float addrspace(4)* %2525 to float addrspace(1)*		; visa id: 3074
  %2527 = load float, float addrspace(1)* %2526, align 4		; visa id: 3075
  %2528 = fmul reassoc nsz arcp contract float %2527, %4, !spirv.Decorations !881		; visa id: 3076
  %2529 = fadd reassoc nsz arcp contract float %2517, %2528, !spirv.Decorations !881		; visa id: 3077
  %2530 = add i64 %.in, %588		; visa id: 3078
  %2531 = inttoptr i64 %2530 to float addrspace(4)*		; visa id: 3079
  %2532 = addrspacecast float addrspace(4)* %2531 to float addrspace(1)*		; visa id: 3079
  store float %2529, float addrspace(1)* %2532, align 4		; visa id: 3080
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3081

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, %2522, %2518
; BB416 :
  br i1 %262, label %2533, label %._crit_edge70.2.10..preheader1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3082

._crit_edge70.2.10..preheader1.10_crit_edge:      ; preds = %._crit_edge70.2.10
; BB:
  br label %.preheader1.10, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2533:                                             ; preds = %._crit_edge70.2.10
; BB418 :
  %2534 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !881		; visa id: 3084
  br i1 %81, label %2539, label %2535, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3085

2535:                                             ; preds = %2533
; BB419 :
  %2536 = add i64 %.in, %590		; visa id: 3087
  %2537 = inttoptr i64 %2536 to float addrspace(4)*		; visa id: 3088
  %2538 = addrspacecast float addrspace(4)* %2537 to float addrspace(1)*		; visa id: 3088
  store float %2534, float addrspace(1)* %2538, align 4		; visa id: 3089
  br label %.preheader1.10, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3090

2539:                                             ; preds = %2533
; BB420 :
  %2540 = add i64 %.in3821, %sink_3840		; visa id: 3092
  %2541 = add i64 %2540, %sink_3830		; visa id: 3093
  %2542 = inttoptr i64 %2541 to float addrspace(4)*		; visa id: 3094
  %2543 = addrspacecast float addrspace(4)* %2542 to float addrspace(1)*		; visa id: 3094
  %2544 = load float, float addrspace(1)* %2543, align 4		; visa id: 3095
  %2545 = fmul reassoc nsz arcp contract float %2544, %4, !spirv.Decorations !881		; visa id: 3096
  %2546 = fadd reassoc nsz arcp contract float %2534, %2545, !spirv.Decorations !881		; visa id: 3097
  %2547 = add i64 %.in, %590		; visa id: 3098
  %2548 = inttoptr i64 %2547 to float addrspace(4)*		; visa id: 3099
  %2549 = addrspacecast float addrspace(4)* %2548 to float addrspace(1)*		; visa id: 3099
  store float %2546, float addrspace(1)* %2549, align 4		; visa id: 3100
  br label %.preheader1.10, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3101

.preheader1.10:                                   ; preds = %._crit_edge70.2.10..preheader1.10_crit_edge, %2539, %2535
; BB421 :
  %sink_3829 = shl nsw i64 %591, 2		; visa id: 3102
  br i1 %266, label %2550, label %.preheader1.10.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3103

.preheader1.10.._crit_edge70.11_crit_edge:        ; preds = %.preheader1.10
; BB:
  br label %._crit_edge70.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2550:                                             ; preds = %.preheader1.10
; BB423 :
  %2551 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !881		; visa id: 3105
  br i1 %81, label %2556, label %2552, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3106

2552:                                             ; preds = %2550
; BB424 :
  %2553 = add i64 %.in, %593		; visa id: 3108
  %2554 = inttoptr i64 %2553 to float addrspace(4)*		; visa id: 3109
  %2555 = addrspacecast float addrspace(4)* %2554 to float addrspace(1)*		; visa id: 3109
  store float %2551, float addrspace(1)* %2555, align 4		; visa id: 3110
  br label %._crit_edge70.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3111

2556:                                             ; preds = %2550
; BB425 :
  %2557 = add i64 %.in3821, %sink_3844		; visa id: 3113
  %2558 = add i64 %2557, %sink_3829		; visa id: 3114
  %2559 = inttoptr i64 %2558 to float addrspace(4)*		; visa id: 3115
  %2560 = addrspacecast float addrspace(4)* %2559 to float addrspace(1)*		; visa id: 3115
  %2561 = load float, float addrspace(1)* %2560, align 4		; visa id: 3116
  %2562 = fmul reassoc nsz arcp contract float %2561, %4, !spirv.Decorations !881		; visa id: 3117
  %2563 = fadd reassoc nsz arcp contract float %2551, %2562, !spirv.Decorations !881		; visa id: 3118
  %2564 = add i64 %.in, %593		; visa id: 3119
  %2565 = inttoptr i64 %2564 to float addrspace(4)*		; visa id: 3120
  %2566 = addrspacecast float addrspace(4)* %2565 to float addrspace(1)*		; visa id: 3120
  store float %2563, float addrspace(1)* %2566, align 4		; visa id: 3121
  br label %._crit_edge70.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3122

._crit_edge70.11:                                 ; preds = %.preheader1.10.._crit_edge70.11_crit_edge, %2556, %2552
; BB426 :
  br i1 %269, label %2567, label %._crit_edge70.11.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3123

._crit_edge70.11.._crit_edge70.1.11_crit_edge:    ; preds = %._crit_edge70.11
; BB:
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2567:                                             ; preds = %._crit_edge70.11
; BB428 :
  %2568 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !881		; visa id: 3125
  br i1 %81, label %2573, label %2569, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3126

2569:                                             ; preds = %2567
; BB429 :
  %2570 = add i64 %.in, %595		; visa id: 3128
  %2571 = inttoptr i64 %2570 to float addrspace(4)*		; visa id: 3129
  %2572 = addrspacecast float addrspace(4)* %2571 to float addrspace(1)*		; visa id: 3129
  store float %2568, float addrspace(1)* %2572, align 4		; visa id: 3130
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3131

2573:                                             ; preds = %2567
; BB430 :
  %2574 = add i64 %.in3821, %sink_3842		; visa id: 3133
  %2575 = add i64 %2574, %sink_3829		; visa id: 3134
  %2576 = inttoptr i64 %2575 to float addrspace(4)*		; visa id: 3135
  %2577 = addrspacecast float addrspace(4)* %2576 to float addrspace(1)*		; visa id: 3135
  %2578 = load float, float addrspace(1)* %2577, align 4		; visa id: 3136
  %2579 = fmul reassoc nsz arcp contract float %2578, %4, !spirv.Decorations !881		; visa id: 3137
  %2580 = fadd reassoc nsz arcp contract float %2568, %2579, !spirv.Decorations !881		; visa id: 3138
  %2581 = add i64 %.in, %595		; visa id: 3139
  %2582 = inttoptr i64 %2581 to float addrspace(4)*		; visa id: 3140
  %2583 = addrspacecast float addrspace(4)* %2582 to float addrspace(1)*		; visa id: 3140
  store float %2580, float addrspace(1)* %2583, align 4		; visa id: 3141
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3142

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11.._crit_edge70.1.11_crit_edge, %2573, %2569
; BB431 :
  br i1 %272, label %2584, label %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3143

._crit_edge70.1.11.._crit_edge70.2.11_crit_edge:  ; preds = %._crit_edge70.1.11
; BB:
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2584:                                             ; preds = %._crit_edge70.1.11
; BB433 :
  %2585 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !881		; visa id: 3145
  br i1 %81, label %2590, label %2586, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3146

2586:                                             ; preds = %2584
; BB434 :
  %2587 = add i64 %.in, %597		; visa id: 3148
  %2588 = inttoptr i64 %2587 to float addrspace(4)*		; visa id: 3149
  %2589 = addrspacecast float addrspace(4)* %2588 to float addrspace(1)*		; visa id: 3149
  store float %2585, float addrspace(1)* %2589, align 4		; visa id: 3150
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3151

2590:                                             ; preds = %2584
; BB435 :
  %2591 = add i64 %.in3821, %sink_3841		; visa id: 3153
  %2592 = add i64 %2591, %sink_3829		; visa id: 3154
  %2593 = inttoptr i64 %2592 to float addrspace(4)*		; visa id: 3155
  %2594 = addrspacecast float addrspace(4)* %2593 to float addrspace(1)*		; visa id: 3155
  %2595 = load float, float addrspace(1)* %2594, align 4		; visa id: 3156
  %2596 = fmul reassoc nsz arcp contract float %2595, %4, !spirv.Decorations !881		; visa id: 3157
  %2597 = fadd reassoc nsz arcp contract float %2585, %2596, !spirv.Decorations !881		; visa id: 3158
  %2598 = add i64 %.in, %597		; visa id: 3159
  %2599 = inttoptr i64 %2598 to float addrspace(4)*		; visa id: 3160
  %2600 = addrspacecast float addrspace(4)* %2599 to float addrspace(1)*		; visa id: 3160
  store float %2597, float addrspace(1)* %2600, align 4		; visa id: 3161
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3162

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, %2590, %2586
; BB436 :
  br i1 %275, label %2601, label %._crit_edge70.2.11..preheader1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3163

._crit_edge70.2.11..preheader1.11_crit_edge:      ; preds = %._crit_edge70.2.11
; BB:
  br label %.preheader1.11, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2601:                                             ; preds = %._crit_edge70.2.11
; BB438 :
  %2602 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !881		; visa id: 3165
  br i1 %81, label %2607, label %2603, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3166

2603:                                             ; preds = %2601
; BB439 :
  %2604 = add i64 %.in, %599		; visa id: 3168
  %2605 = inttoptr i64 %2604 to float addrspace(4)*		; visa id: 3169
  %2606 = addrspacecast float addrspace(4)* %2605 to float addrspace(1)*		; visa id: 3169
  store float %2602, float addrspace(1)* %2606, align 4		; visa id: 3170
  br label %.preheader1.11, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3171

2607:                                             ; preds = %2601
; BB440 :
  %2608 = add i64 %.in3821, %sink_3840		; visa id: 3173
  %2609 = add i64 %2608, %sink_3829		; visa id: 3174
  %2610 = inttoptr i64 %2609 to float addrspace(4)*		; visa id: 3175
  %2611 = addrspacecast float addrspace(4)* %2610 to float addrspace(1)*		; visa id: 3175
  %2612 = load float, float addrspace(1)* %2611, align 4		; visa id: 3176
  %2613 = fmul reassoc nsz arcp contract float %2612, %4, !spirv.Decorations !881		; visa id: 3177
  %2614 = fadd reassoc nsz arcp contract float %2602, %2613, !spirv.Decorations !881		; visa id: 3178
  %2615 = add i64 %.in, %599		; visa id: 3179
  %2616 = inttoptr i64 %2615 to float addrspace(4)*		; visa id: 3180
  %2617 = addrspacecast float addrspace(4)* %2616 to float addrspace(1)*		; visa id: 3180
  store float %2614, float addrspace(1)* %2617, align 4		; visa id: 3181
  br label %.preheader1.11, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3182

.preheader1.11:                                   ; preds = %._crit_edge70.2.11..preheader1.11_crit_edge, %2607, %2603
; BB441 :
  %sink_3828 = shl nsw i64 %600, 2		; visa id: 3183
  br i1 %279, label %2618, label %.preheader1.11.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3184

.preheader1.11.._crit_edge70.12_crit_edge:        ; preds = %.preheader1.11
; BB:
  br label %._crit_edge70.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2618:                                             ; preds = %.preheader1.11
; BB443 :
  %2619 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !881		; visa id: 3186
  br i1 %81, label %2624, label %2620, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3187

2620:                                             ; preds = %2618
; BB444 :
  %2621 = add i64 %.in, %602		; visa id: 3189
  %2622 = inttoptr i64 %2621 to float addrspace(4)*		; visa id: 3190
  %2623 = addrspacecast float addrspace(4)* %2622 to float addrspace(1)*		; visa id: 3190
  store float %2619, float addrspace(1)* %2623, align 4		; visa id: 3191
  br label %._crit_edge70.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3192

2624:                                             ; preds = %2618
; BB445 :
  %2625 = add i64 %.in3821, %sink_3844		; visa id: 3194
  %2626 = add i64 %2625, %sink_3828		; visa id: 3195
  %2627 = inttoptr i64 %2626 to float addrspace(4)*		; visa id: 3196
  %2628 = addrspacecast float addrspace(4)* %2627 to float addrspace(1)*		; visa id: 3196
  %2629 = load float, float addrspace(1)* %2628, align 4		; visa id: 3197
  %2630 = fmul reassoc nsz arcp contract float %2629, %4, !spirv.Decorations !881		; visa id: 3198
  %2631 = fadd reassoc nsz arcp contract float %2619, %2630, !spirv.Decorations !881		; visa id: 3199
  %2632 = add i64 %.in, %602		; visa id: 3200
  %2633 = inttoptr i64 %2632 to float addrspace(4)*		; visa id: 3201
  %2634 = addrspacecast float addrspace(4)* %2633 to float addrspace(1)*		; visa id: 3201
  store float %2631, float addrspace(1)* %2634, align 4		; visa id: 3202
  br label %._crit_edge70.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3203

._crit_edge70.12:                                 ; preds = %.preheader1.11.._crit_edge70.12_crit_edge, %2624, %2620
; BB446 :
  br i1 %282, label %2635, label %._crit_edge70.12.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3204

._crit_edge70.12.._crit_edge70.1.12_crit_edge:    ; preds = %._crit_edge70.12
; BB:
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2635:                                             ; preds = %._crit_edge70.12
; BB448 :
  %2636 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !881		; visa id: 3206
  br i1 %81, label %2641, label %2637, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3207

2637:                                             ; preds = %2635
; BB449 :
  %2638 = add i64 %.in, %604		; visa id: 3209
  %2639 = inttoptr i64 %2638 to float addrspace(4)*		; visa id: 3210
  %2640 = addrspacecast float addrspace(4)* %2639 to float addrspace(1)*		; visa id: 3210
  store float %2636, float addrspace(1)* %2640, align 4		; visa id: 3211
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3212

2641:                                             ; preds = %2635
; BB450 :
  %2642 = add i64 %.in3821, %sink_3842		; visa id: 3214
  %2643 = add i64 %2642, %sink_3828		; visa id: 3215
  %2644 = inttoptr i64 %2643 to float addrspace(4)*		; visa id: 3216
  %2645 = addrspacecast float addrspace(4)* %2644 to float addrspace(1)*		; visa id: 3216
  %2646 = load float, float addrspace(1)* %2645, align 4		; visa id: 3217
  %2647 = fmul reassoc nsz arcp contract float %2646, %4, !spirv.Decorations !881		; visa id: 3218
  %2648 = fadd reassoc nsz arcp contract float %2636, %2647, !spirv.Decorations !881		; visa id: 3219
  %2649 = add i64 %.in, %604		; visa id: 3220
  %2650 = inttoptr i64 %2649 to float addrspace(4)*		; visa id: 3221
  %2651 = addrspacecast float addrspace(4)* %2650 to float addrspace(1)*		; visa id: 3221
  store float %2648, float addrspace(1)* %2651, align 4		; visa id: 3222
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3223

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12.._crit_edge70.1.12_crit_edge, %2641, %2637
; BB451 :
  br i1 %285, label %2652, label %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3224

._crit_edge70.1.12.._crit_edge70.2.12_crit_edge:  ; preds = %._crit_edge70.1.12
; BB:
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2652:                                             ; preds = %._crit_edge70.1.12
; BB453 :
  %2653 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !881		; visa id: 3226
  br i1 %81, label %2658, label %2654, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3227

2654:                                             ; preds = %2652
; BB454 :
  %2655 = add i64 %.in, %606		; visa id: 3229
  %2656 = inttoptr i64 %2655 to float addrspace(4)*		; visa id: 3230
  %2657 = addrspacecast float addrspace(4)* %2656 to float addrspace(1)*		; visa id: 3230
  store float %2653, float addrspace(1)* %2657, align 4		; visa id: 3231
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3232

2658:                                             ; preds = %2652
; BB455 :
  %2659 = add i64 %.in3821, %sink_3841		; visa id: 3234
  %2660 = add i64 %2659, %sink_3828		; visa id: 3235
  %2661 = inttoptr i64 %2660 to float addrspace(4)*		; visa id: 3236
  %2662 = addrspacecast float addrspace(4)* %2661 to float addrspace(1)*		; visa id: 3236
  %2663 = load float, float addrspace(1)* %2662, align 4		; visa id: 3237
  %2664 = fmul reassoc nsz arcp contract float %2663, %4, !spirv.Decorations !881		; visa id: 3238
  %2665 = fadd reassoc nsz arcp contract float %2653, %2664, !spirv.Decorations !881		; visa id: 3239
  %2666 = add i64 %.in, %606		; visa id: 3240
  %2667 = inttoptr i64 %2666 to float addrspace(4)*		; visa id: 3241
  %2668 = addrspacecast float addrspace(4)* %2667 to float addrspace(1)*		; visa id: 3241
  store float %2665, float addrspace(1)* %2668, align 4		; visa id: 3242
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3243

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, %2658, %2654
; BB456 :
  br i1 %288, label %2669, label %._crit_edge70.2.12..preheader1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3244

._crit_edge70.2.12..preheader1.12_crit_edge:      ; preds = %._crit_edge70.2.12
; BB:
  br label %.preheader1.12, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2669:                                             ; preds = %._crit_edge70.2.12
; BB458 :
  %2670 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !881		; visa id: 3246
  br i1 %81, label %2675, label %2671, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3247

2671:                                             ; preds = %2669
; BB459 :
  %2672 = add i64 %.in, %608		; visa id: 3249
  %2673 = inttoptr i64 %2672 to float addrspace(4)*		; visa id: 3250
  %2674 = addrspacecast float addrspace(4)* %2673 to float addrspace(1)*		; visa id: 3250
  store float %2670, float addrspace(1)* %2674, align 4		; visa id: 3251
  br label %.preheader1.12, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3252

2675:                                             ; preds = %2669
; BB460 :
  %2676 = add i64 %.in3821, %sink_3840		; visa id: 3254
  %2677 = add i64 %2676, %sink_3828		; visa id: 3255
  %2678 = inttoptr i64 %2677 to float addrspace(4)*		; visa id: 3256
  %2679 = addrspacecast float addrspace(4)* %2678 to float addrspace(1)*		; visa id: 3256
  %2680 = load float, float addrspace(1)* %2679, align 4		; visa id: 3257
  %2681 = fmul reassoc nsz arcp contract float %2680, %4, !spirv.Decorations !881		; visa id: 3258
  %2682 = fadd reassoc nsz arcp contract float %2670, %2681, !spirv.Decorations !881		; visa id: 3259
  %2683 = add i64 %.in, %608		; visa id: 3260
  %2684 = inttoptr i64 %2683 to float addrspace(4)*		; visa id: 3261
  %2685 = addrspacecast float addrspace(4)* %2684 to float addrspace(1)*		; visa id: 3261
  store float %2682, float addrspace(1)* %2685, align 4		; visa id: 3262
  br label %.preheader1.12, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3263

.preheader1.12:                                   ; preds = %._crit_edge70.2.12..preheader1.12_crit_edge, %2675, %2671
; BB461 :
  %sink_3827 = shl nsw i64 %609, 2		; visa id: 3264
  br i1 %292, label %2686, label %.preheader1.12.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3265

.preheader1.12.._crit_edge70.13_crit_edge:        ; preds = %.preheader1.12
; BB:
  br label %._crit_edge70.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2686:                                             ; preds = %.preheader1.12
; BB463 :
  %2687 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !881		; visa id: 3267
  br i1 %81, label %2692, label %2688, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3268

2688:                                             ; preds = %2686
; BB464 :
  %2689 = add i64 %.in, %611		; visa id: 3270
  %2690 = inttoptr i64 %2689 to float addrspace(4)*		; visa id: 3271
  %2691 = addrspacecast float addrspace(4)* %2690 to float addrspace(1)*		; visa id: 3271
  store float %2687, float addrspace(1)* %2691, align 4		; visa id: 3272
  br label %._crit_edge70.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3273

2692:                                             ; preds = %2686
; BB465 :
  %2693 = add i64 %.in3821, %sink_3844		; visa id: 3275
  %2694 = add i64 %2693, %sink_3827		; visa id: 3276
  %2695 = inttoptr i64 %2694 to float addrspace(4)*		; visa id: 3277
  %2696 = addrspacecast float addrspace(4)* %2695 to float addrspace(1)*		; visa id: 3277
  %2697 = load float, float addrspace(1)* %2696, align 4		; visa id: 3278
  %2698 = fmul reassoc nsz arcp contract float %2697, %4, !spirv.Decorations !881		; visa id: 3279
  %2699 = fadd reassoc nsz arcp contract float %2687, %2698, !spirv.Decorations !881		; visa id: 3280
  %2700 = add i64 %.in, %611		; visa id: 3281
  %2701 = inttoptr i64 %2700 to float addrspace(4)*		; visa id: 3282
  %2702 = addrspacecast float addrspace(4)* %2701 to float addrspace(1)*		; visa id: 3282
  store float %2699, float addrspace(1)* %2702, align 4		; visa id: 3283
  br label %._crit_edge70.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3284

._crit_edge70.13:                                 ; preds = %.preheader1.12.._crit_edge70.13_crit_edge, %2692, %2688
; BB466 :
  br i1 %295, label %2703, label %._crit_edge70.13.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3285

._crit_edge70.13.._crit_edge70.1.13_crit_edge:    ; preds = %._crit_edge70.13
; BB:
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2703:                                             ; preds = %._crit_edge70.13
; BB468 :
  %2704 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !881		; visa id: 3287
  br i1 %81, label %2709, label %2705, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3288

2705:                                             ; preds = %2703
; BB469 :
  %2706 = add i64 %.in, %613		; visa id: 3290
  %2707 = inttoptr i64 %2706 to float addrspace(4)*		; visa id: 3291
  %2708 = addrspacecast float addrspace(4)* %2707 to float addrspace(1)*		; visa id: 3291
  store float %2704, float addrspace(1)* %2708, align 4		; visa id: 3292
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3293

2709:                                             ; preds = %2703
; BB470 :
  %2710 = add i64 %.in3821, %sink_3842		; visa id: 3295
  %2711 = add i64 %2710, %sink_3827		; visa id: 3296
  %2712 = inttoptr i64 %2711 to float addrspace(4)*		; visa id: 3297
  %2713 = addrspacecast float addrspace(4)* %2712 to float addrspace(1)*		; visa id: 3297
  %2714 = load float, float addrspace(1)* %2713, align 4		; visa id: 3298
  %2715 = fmul reassoc nsz arcp contract float %2714, %4, !spirv.Decorations !881		; visa id: 3299
  %2716 = fadd reassoc nsz arcp contract float %2704, %2715, !spirv.Decorations !881		; visa id: 3300
  %2717 = add i64 %.in, %613		; visa id: 3301
  %2718 = inttoptr i64 %2717 to float addrspace(4)*		; visa id: 3302
  %2719 = addrspacecast float addrspace(4)* %2718 to float addrspace(1)*		; visa id: 3302
  store float %2716, float addrspace(1)* %2719, align 4		; visa id: 3303
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3304

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13.._crit_edge70.1.13_crit_edge, %2709, %2705
; BB471 :
  br i1 %298, label %2720, label %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3305

._crit_edge70.1.13.._crit_edge70.2.13_crit_edge:  ; preds = %._crit_edge70.1.13
; BB:
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2720:                                             ; preds = %._crit_edge70.1.13
; BB473 :
  %2721 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !881		; visa id: 3307
  br i1 %81, label %2726, label %2722, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3308

2722:                                             ; preds = %2720
; BB474 :
  %2723 = add i64 %.in, %615		; visa id: 3310
  %2724 = inttoptr i64 %2723 to float addrspace(4)*		; visa id: 3311
  %2725 = addrspacecast float addrspace(4)* %2724 to float addrspace(1)*		; visa id: 3311
  store float %2721, float addrspace(1)* %2725, align 4		; visa id: 3312
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3313

2726:                                             ; preds = %2720
; BB475 :
  %2727 = add i64 %.in3821, %sink_3841		; visa id: 3315
  %2728 = add i64 %2727, %sink_3827		; visa id: 3316
  %2729 = inttoptr i64 %2728 to float addrspace(4)*		; visa id: 3317
  %2730 = addrspacecast float addrspace(4)* %2729 to float addrspace(1)*		; visa id: 3317
  %2731 = load float, float addrspace(1)* %2730, align 4		; visa id: 3318
  %2732 = fmul reassoc nsz arcp contract float %2731, %4, !spirv.Decorations !881		; visa id: 3319
  %2733 = fadd reassoc nsz arcp contract float %2721, %2732, !spirv.Decorations !881		; visa id: 3320
  %2734 = add i64 %.in, %615		; visa id: 3321
  %2735 = inttoptr i64 %2734 to float addrspace(4)*		; visa id: 3322
  %2736 = addrspacecast float addrspace(4)* %2735 to float addrspace(1)*		; visa id: 3322
  store float %2733, float addrspace(1)* %2736, align 4		; visa id: 3323
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3324

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, %2726, %2722
; BB476 :
  br i1 %301, label %2737, label %._crit_edge70.2.13..preheader1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3325

._crit_edge70.2.13..preheader1.13_crit_edge:      ; preds = %._crit_edge70.2.13
; BB:
  br label %.preheader1.13, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2737:                                             ; preds = %._crit_edge70.2.13
; BB478 :
  %2738 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !881		; visa id: 3327
  br i1 %81, label %2743, label %2739, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3328

2739:                                             ; preds = %2737
; BB479 :
  %2740 = add i64 %.in, %617		; visa id: 3330
  %2741 = inttoptr i64 %2740 to float addrspace(4)*		; visa id: 3331
  %2742 = addrspacecast float addrspace(4)* %2741 to float addrspace(1)*		; visa id: 3331
  store float %2738, float addrspace(1)* %2742, align 4		; visa id: 3332
  br label %.preheader1.13, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3333

2743:                                             ; preds = %2737
; BB480 :
  %2744 = add i64 %.in3821, %sink_3840		; visa id: 3335
  %2745 = add i64 %2744, %sink_3827		; visa id: 3336
  %2746 = inttoptr i64 %2745 to float addrspace(4)*		; visa id: 3337
  %2747 = addrspacecast float addrspace(4)* %2746 to float addrspace(1)*		; visa id: 3337
  %2748 = load float, float addrspace(1)* %2747, align 4		; visa id: 3338
  %2749 = fmul reassoc nsz arcp contract float %2748, %4, !spirv.Decorations !881		; visa id: 3339
  %2750 = fadd reassoc nsz arcp contract float %2738, %2749, !spirv.Decorations !881		; visa id: 3340
  %2751 = add i64 %.in, %617		; visa id: 3341
  %2752 = inttoptr i64 %2751 to float addrspace(4)*		; visa id: 3342
  %2753 = addrspacecast float addrspace(4)* %2752 to float addrspace(1)*		; visa id: 3342
  store float %2750, float addrspace(1)* %2753, align 4		; visa id: 3343
  br label %.preheader1.13, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3344

.preheader1.13:                                   ; preds = %._crit_edge70.2.13..preheader1.13_crit_edge, %2743, %2739
; BB481 :
  %sink_3826 = shl nsw i64 %618, 2		; visa id: 3345
  br i1 %305, label %2754, label %.preheader1.13.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3346

.preheader1.13.._crit_edge70.14_crit_edge:        ; preds = %.preheader1.13
; BB:
  br label %._crit_edge70.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2754:                                             ; preds = %.preheader1.13
; BB483 :
  %2755 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !881		; visa id: 3348
  br i1 %81, label %2760, label %2756, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3349

2756:                                             ; preds = %2754
; BB484 :
  %2757 = add i64 %.in, %620		; visa id: 3351
  %2758 = inttoptr i64 %2757 to float addrspace(4)*		; visa id: 3352
  %2759 = addrspacecast float addrspace(4)* %2758 to float addrspace(1)*		; visa id: 3352
  store float %2755, float addrspace(1)* %2759, align 4		; visa id: 3353
  br label %._crit_edge70.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3354

2760:                                             ; preds = %2754
; BB485 :
  %2761 = add i64 %.in3821, %sink_3844		; visa id: 3356
  %2762 = add i64 %2761, %sink_3826		; visa id: 3357
  %2763 = inttoptr i64 %2762 to float addrspace(4)*		; visa id: 3358
  %2764 = addrspacecast float addrspace(4)* %2763 to float addrspace(1)*		; visa id: 3358
  %2765 = load float, float addrspace(1)* %2764, align 4		; visa id: 3359
  %2766 = fmul reassoc nsz arcp contract float %2765, %4, !spirv.Decorations !881		; visa id: 3360
  %2767 = fadd reassoc nsz arcp contract float %2755, %2766, !spirv.Decorations !881		; visa id: 3361
  %2768 = add i64 %.in, %620		; visa id: 3362
  %2769 = inttoptr i64 %2768 to float addrspace(4)*		; visa id: 3363
  %2770 = addrspacecast float addrspace(4)* %2769 to float addrspace(1)*		; visa id: 3363
  store float %2767, float addrspace(1)* %2770, align 4		; visa id: 3364
  br label %._crit_edge70.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3365

._crit_edge70.14:                                 ; preds = %.preheader1.13.._crit_edge70.14_crit_edge, %2760, %2756
; BB486 :
  br i1 %308, label %2771, label %._crit_edge70.14.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3366

._crit_edge70.14.._crit_edge70.1.14_crit_edge:    ; preds = %._crit_edge70.14
; BB:
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2771:                                             ; preds = %._crit_edge70.14
; BB488 :
  %2772 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !881		; visa id: 3368
  br i1 %81, label %2777, label %2773, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3369

2773:                                             ; preds = %2771
; BB489 :
  %2774 = add i64 %.in, %622		; visa id: 3371
  %2775 = inttoptr i64 %2774 to float addrspace(4)*		; visa id: 3372
  %2776 = addrspacecast float addrspace(4)* %2775 to float addrspace(1)*		; visa id: 3372
  store float %2772, float addrspace(1)* %2776, align 4		; visa id: 3373
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3374

2777:                                             ; preds = %2771
; BB490 :
  %2778 = add i64 %.in3821, %sink_3842		; visa id: 3376
  %2779 = add i64 %2778, %sink_3826		; visa id: 3377
  %2780 = inttoptr i64 %2779 to float addrspace(4)*		; visa id: 3378
  %2781 = addrspacecast float addrspace(4)* %2780 to float addrspace(1)*		; visa id: 3378
  %2782 = load float, float addrspace(1)* %2781, align 4		; visa id: 3379
  %2783 = fmul reassoc nsz arcp contract float %2782, %4, !spirv.Decorations !881		; visa id: 3380
  %2784 = fadd reassoc nsz arcp contract float %2772, %2783, !spirv.Decorations !881		; visa id: 3381
  %2785 = add i64 %.in, %622		; visa id: 3382
  %2786 = inttoptr i64 %2785 to float addrspace(4)*		; visa id: 3383
  %2787 = addrspacecast float addrspace(4)* %2786 to float addrspace(1)*		; visa id: 3383
  store float %2784, float addrspace(1)* %2787, align 4		; visa id: 3384
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3385

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14.._crit_edge70.1.14_crit_edge, %2777, %2773
; BB491 :
  br i1 %311, label %2788, label %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3386

._crit_edge70.1.14.._crit_edge70.2.14_crit_edge:  ; preds = %._crit_edge70.1.14
; BB:
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2788:                                             ; preds = %._crit_edge70.1.14
; BB493 :
  %2789 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !881		; visa id: 3388
  br i1 %81, label %2794, label %2790, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3389

2790:                                             ; preds = %2788
; BB494 :
  %2791 = add i64 %.in, %624		; visa id: 3391
  %2792 = inttoptr i64 %2791 to float addrspace(4)*		; visa id: 3392
  %2793 = addrspacecast float addrspace(4)* %2792 to float addrspace(1)*		; visa id: 3392
  store float %2789, float addrspace(1)* %2793, align 4		; visa id: 3393
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3394

2794:                                             ; preds = %2788
; BB495 :
  %2795 = add i64 %.in3821, %sink_3841		; visa id: 3396
  %2796 = add i64 %2795, %sink_3826		; visa id: 3397
  %2797 = inttoptr i64 %2796 to float addrspace(4)*		; visa id: 3398
  %2798 = addrspacecast float addrspace(4)* %2797 to float addrspace(1)*		; visa id: 3398
  %2799 = load float, float addrspace(1)* %2798, align 4		; visa id: 3399
  %2800 = fmul reassoc nsz arcp contract float %2799, %4, !spirv.Decorations !881		; visa id: 3400
  %2801 = fadd reassoc nsz arcp contract float %2789, %2800, !spirv.Decorations !881		; visa id: 3401
  %2802 = add i64 %.in, %624		; visa id: 3402
  %2803 = inttoptr i64 %2802 to float addrspace(4)*		; visa id: 3403
  %2804 = addrspacecast float addrspace(4)* %2803 to float addrspace(1)*		; visa id: 3403
  store float %2801, float addrspace(1)* %2804, align 4		; visa id: 3404
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3405

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, %2794, %2790
; BB496 :
  br i1 %314, label %2805, label %._crit_edge70.2.14..preheader1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3406

._crit_edge70.2.14..preheader1.14_crit_edge:      ; preds = %._crit_edge70.2.14
; BB:
  br label %.preheader1.14, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2805:                                             ; preds = %._crit_edge70.2.14
; BB498 :
  %2806 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !881		; visa id: 3408
  br i1 %81, label %2811, label %2807, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3409

2807:                                             ; preds = %2805
; BB499 :
  %2808 = add i64 %.in, %626		; visa id: 3411
  %2809 = inttoptr i64 %2808 to float addrspace(4)*		; visa id: 3412
  %2810 = addrspacecast float addrspace(4)* %2809 to float addrspace(1)*		; visa id: 3412
  store float %2806, float addrspace(1)* %2810, align 4		; visa id: 3413
  br label %.preheader1.14, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3414

2811:                                             ; preds = %2805
; BB500 :
  %2812 = add i64 %.in3821, %sink_3840		; visa id: 3416
  %2813 = add i64 %2812, %sink_3826		; visa id: 3417
  %2814 = inttoptr i64 %2813 to float addrspace(4)*		; visa id: 3418
  %2815 = addrspacecast float addrspace(4)* %2814 to float addrspace(1)*		; visa id: 3418
  %2816 = load float, float addrspace(1)* %2815, align 4		; visa id: 3419
  %2817 = fmul reassoc nsz arcp contract float %2816, %4, !spirv.Decorations !881		; visa id: 3420
  %2818 = fadd reassoc nsz arcp contract float %2806, %2817, !spirv.Decorations !881		; visa id: 3421
  %2819 = add i64 %.in, %626		; visa id: 3422
  %2820 = inttoptr i64 %2819 to float addrspace(4)*		; visa id: 3423
  %2821 = addrspacecast float addrspace(4)* %2820 to float addrspace(1)*		; visa id: 3423
  store float %2818, float addrspace(1)* %2821, align 4		; visa id: 3424
  br label %.preheader1.14, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3425

.preheader1.14:                                   ; preds = %._crit_edge70.2.14..preheader1.14_crit_edge, %2811, %2807
; BB501 :
  %sink_3825 = shl nsw i64 %627, 2		; visa id: 3426
  br i1 %318, label %2822, label %.preheader1.14.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3427

.preheader1.14.._crit_edge70.15_crit_edge:        ; preds = %.preheader1.14
; BB:
  br label %._crit_edge70.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2822:                                             ; preds = %.preheader1.14
; BB503 :
  %2823 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !881		; visa id: 3429
  br i1 %81, label %2828, label %2824, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3430

2824:                                             ; preds = %2822
; BB504 :
  %2825 = add i64 %.in, %629		; visa id: 3432
  %2826 = inttoptr i64 %2825 to float addrspace(4)*		; visa id: 3433
  %2827 = addrspacecast float addrspace(4)* %2826 to float addrspace(1)*		; visa id: 3433
  store float %2823, float addrspace(1)* %2827, align 4		; visa id: 3434
  br label %._crit_edge70.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3435

2828:                                             ; preds = %2822
; BB505 :
  %2829 = add i64 %.in3821, %sink_3844		; visa id: 3437
  %2830 = add i64 %2829, %sink_3825		; visa id: 3438
  %2831 = inttoptr i64 %2830 to float addrspace(4)*		; visa id: 3439
  %2832 = addrspacecast float addrspace(4)* %2831 to float addrspace(1)*		; visa id: 3439
  %2833 = load float, float addrspace(1)* %2832, align 4		; visa id: 3440
  %2834 = fmul reassoc nsz arcp contract float %2833, %4, !spirv.Decorations !881		; visa id: 3441
  %2835 = fadd reassoc nsz arcp contract float %2823, %2834, !spirv.Decorations !881		; visa id: 3442
  %2836 = add i64 %.in, %629		; visa id: 3443
  %2837 = inttoptr i64 %2836 to float addrspace(4)*		; visa id: 3444
  %2838 = addrspacecast float addrspace(4)* %2837 to float addrspace(1)*		; visa id: 3444
  store float %2835, float addrspace(1)* %2838, align 4		; visa id: 3445
  br label %._crit_edge70.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3446

._crit_edge70.15:                                 ; preds = %.preheader1.14.._crit_edge70.15_crit_edge, %2828, %2824
; BB506 :
  br i1 %321, label %2839, label %._crit_edge70.15.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3447

._crit_edge70.15.._crit_edge70.1.15_crit_edge:    ; preds = %._crit_edge70.15
; BB:
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2839:                                             ; preds = %._crit_edge70.15
; BB508 :
  %2840 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !881		; visa id: 3449
  br i1 %81, label %2845, label %2841, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3450

2841:                                             ; preds = %2839
; BB509 :
  %2842 = add i64 %.in, %631		; visa id: 3452
  %2843 = inttoptr i64 %2842 to float addrspace(4)*		; visa id: 3453
  %2844 = addrspacecast float addrspace(4)* %2843 to float addrspace(1)*		; visa id: 3453
  store float %2840, float addrspace(1)* %2844, align 4		; visa id: 3454
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3455

2845:                                             ; preds = %2839
; BB510 :
  %2846 = add i64 %.in3821, %sink_3842		; visa id: 3457
  %2847 = add i64 %2846, %sink_3825		; visa id: 3458
  %2848 = inttoptr i64 %2847 to float addrspace(4)*		; visa id: 3459
  %2849 = addrspacecast float addrspace(4)* %2848 to float addrspace(1)*		; visa id: 3459
  %2850 = load float, float addrspace(1)* %2849, align 4		; visa id: 3460
  %2851 = fmul reassoc nsz arcp contract float %2850, %4, !spirv.Decorations !881		; visa id: 3461
  %2852 = fadd reassoc nsz arcp contract float %2840, %2851, !spirv.Decorations !881		; visa id: 3462
  %2853 = add i64 %.in, %631		; visa id: 3463
  %2854 = inttoptr i64 %2853 to float addrspace(4)*		; visa id: 3464
  %2855 = addrspacecast float addrspace(4)* %2854 to float addrspace(1)*		; visa id: 3464
  store float %2852, float addrspace(1)* %2855, align 4		; visa id: 3465
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3466

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15.._crit_edge70.1.15_crit_edge, %2845, %2841
; BB511 :
  br i1 %324, label %2856, label %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3467

._crit_edge70.1.15.._crit_edge70.2.15_crit_edge:  ; preds = %._crit_edge70.1.15
; BB:
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2856:                                             ; preds = %._crit_edge70.1.15
; BB513 :
  %2857 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !881		; visa id: 3469
  br i1 %81, label %2862, label %2858, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3470

2858:                                             ; preds = %2856
; BB514 :
  %2859 = add i64 %.in, %633		; visa id: 3472
  %2860 = inttoptr i64 %2859 to float addrspace(4)*		; visa id: 3473
  %2861 = addrspacecast float addrspace(4)* %2860 to float addrspace(1)*		; visa id: 3473
  store float %2857, float addrspace(1)* %2861, align 4		; visa id: 3474
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3475

2862:                                             ; preds = %2856
; BB515 :
  %2863 = add i64 %.in3821, %sink_3841		; visa id: 3477
  %2864 = add i64 %2863, %sink_3825		; visa id: 3478
  %2865 = inttoptr i64 %2864 to float addrspace(4)*		; visa id: 3479
  %2866 = addrspacecast float addrspace(4)* %2865 to float addrspace(1)*		; visa id: 3479
  %2867 = load float, float addrspace(1)* %2866, align 4		; visa id: 3480
  %2868 = fmul reassoc nsz arcp contract float %2867, %4, !spirv.Decorations !881		; visa id: 3481
  %2869 = fadd reassoc nsz arcp contract float %2857, %2868, !spirv.Decorations !881		; visa id: 3482
  %2870 = add i64 %.in, %633		; visa id: 3483
  %2871 = inttoptr i64 %2870 to float addrspace(4)*		; visa id: 3484
  %2872 = addrspacecast float addrspace(4)* %2871 to float addrspace(1)*		; visa id: 3484
  store float %2869, float addrspace(1)* %2872, align 4		; visa id: 3485
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3486

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, %2862, %2858
; BB516 :
  br i1 %327, label %2873, label %._crit_edge70.2.15..preheader1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3487

._crit_edge70.2.15..preheader1.15_crit_edge:      ; preds = %._crit_edge70.2.15
; BB:
  br label %.preheader1.15, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879

2873:                                             ; preds = %._crit_edge70.2.15
; BB518 :
  %2874 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !881		; visa id: 3489
  br i1 %81, label %2879, label %2875, !stats.blockFrequency.digits !893, !stats.blockFrequency.scale !879		; visa id: 3490

2875:                                             ; preds = %2873
; BB519 :
  %2876 = add i64 %.in, %635		; visa id: 3492
  %2877 = inttoptr i64 %2876 to float addrspace(4)*		; visa id: 3493
  %2878 = addrspacecast float addrspace(4)* %2877 to float addrspace(1)*		; visa id: 3493
  store float %2874, float addrspace(1)* %2878, align 4		; visa id: 3494
  br label %.preheader1.15, !stats.blockFrequency.digits !894, !stats.blockFrequency.scale !879		; visa id: 3495

2879:                                             ; preds = %2873
; BB520 :
  %2880 = add i64 %.in3821, %sink_3840		; visa id: 3497
  %2881 = add i64 %2880, %sink_3825		; visa id: 3498
  %2882 = inttoptr i64 %2881 to float addrspace(4)*		; visa id: 3499
  %2883 = addrspacecast float addrspace(4)* %2882 to float addrspace(1)*		; visa id: 3499
  %2884 = load float, float addrspace(1)* %2883, align 4		; visa id: 3500
  %2885 = fmul reassoc nsz arcp contract float %2884, %4, !spirv.Decorations !881		; visa id: 3501
  %2886 = fadd reassoc nsz arcp contract float %2874, %2885, !spirv.Decorations !881		; visa id: 3502
  %2887 = add i64 %.in, %635		; visa id: 3503
  %2888 = inttoptr i64 %2887 to float addrspace(4)*		; visa id: 3504
  %2889 = addrspacecast float addrspace(4)* %2888 to float addrspace(1)*		; visa id: 3504
  store float %2886, float addrspace(1)* %2889, align 4		; visa id: 3505
  br label %.preheader1.15, !stats.blockFrequency.digits !895, !stats.blockFrequency.scale !879		; visa id: 3506

.preheader1.15:                                   ; preds = %._crit_edge70.2.15..preheader1.15_crit_edge, %2879, %2875
; BB521 :
  %2890 = add i32 %646, %52		; visa id: 3507
  %2891 = icmp slt i32 %2890, %8		; visa id: 3508
  br i1 %2891, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 3509

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
; BB523 :
  %2892 = add i64 %.in3823, %636		; visa id: 3511
  %2893 = add i64 %.in3822, %637		; visa id: 3512
  %sink_ = bitcast <2 x i32> %644 to i64		; visa id: 3513
  %2894 = add i64 %.in3821, %sink_		; visa id: 3515
  %2895 = add i64 %.in, %645		; visa id: 3516
  br label %.preheader2.preheader, !stats.blockFrequency.digits !896, !stats.blockFrequency.scale !879		; visa id: 3517

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
; BB524 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 3519
}

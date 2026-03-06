; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0001.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE(%"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %0, %class.__generated_* byval(%class.__generated_) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i64 %const_reg_qword12, i32 %bindlessOffset) #0 {
; BB0 :
  %3 = bitcast i64 %const_reg_qword to <2 x i32>		; visa id: 2
  %4 = extractelement <2 x i32> %3, i32 0		; visa id: 3
  %5 = extractelement <2 x i32> %3, i32 1		; visa id: 3
  %6 = bitcast i64 %const_reg_qword1 to <2 x i32>		; visa id: 3
  %7 = extractelement <2 x i32> %6, i32 0		; visa id: 4
  %8 = extractelement <2 x i32> %6, i32 1		; visa id: 4
  %9 = bitcast i64 %const_reg_qword2 to <2 x i32>		; visa id: 4
  %10 = extractelement <2 x i32> %9, i32 0		; visa id: 5
  %11 = extractelement <2 x i32> %9, i32 1		; visa id: 5
  %12 = extractelement <3 x i32> %globalOffset, i32 0		; visa id: 5
  %13 = extractelement <3 x i32> %globalOffset, i32 1		; visa id: 5
  %14 = extractelement <3 x i32> %globalOffset, i32 2		; visa id: 5
  %15 = extractelement <3 x i32> %enqueuedLocalSize, i32 0		; visa id: 5
  %16 = extractelement <3 x i32> %enqueuedLocalSize, i32 1		; visa id: 5
  %17 = extractelement <3 x i32> %enqueuedLocalSize, i32 2		; visa id: 5
  %18 = extractelement <8 x i32> %r0, i32 1		; visa id: 5
  %19 = extractelement <8 x i32> %r0, i32 6		; visa id: 5
  %20 = extractelement <8 x i32> %r0, i32 7		; visa id: 5
  %21 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %17, i32 0, i32 %20, i32 0)
  %22 = extractvalue { i32, i32 } %21, 0		; visa id: 5
  %23 = extractvalue { i32, i32 } %21, 1		; visa id: 5
  %24 = insertelement <2 x i32> undef, i32 %22, i32 0		; visa id: 12
  %25 = insertelement <2 x i32> %24, i32 %23, i32 1		; visa id: 13
  %26 = bitcast <2 x i32> %25 to i64		; visa id: 14
  %27 = zext i16 %localIdZ to i64		; visa id: 16
  %28 = add nuw i64 %26, %27		; visa id: 17
  %29 = zext i32 %14 to i64		; visa id: 18
  %30 = add nuw i64 %28, %29		; visa id: 19
  %31 = bitcast i64 %30 to <2 x i32>		; visa id: 20
  %32 = extractelement <2 x i32> %31, i32 0		; visa id: 24
  %33 = extractelement <2 x i32> %31, i32 1		; visa id: 24
  %34 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %16, i32 0, i32 %19, i32 0)
  %35 = extractvalue { i32, i32 } %34, 0		; visa id: 24
  %36 = extractvalue { i32, i32 } %34, 1		; visa id: 24
  %37 = insertelement <2 x i32> undef, i32 %35, i32 0		; visa id: 31
  %38 = insertelement <2 x i32> %37, i32 %36, i32 1		; visa id: 32
  %39 = bitcast <2 x i32> %38 to i64		; visa id: 33
  %40 = zext i16 %localIdY to i64		; visa id: 35
  %41 = add nuw i64 %39, %40		; visa id: 36
  %42 = zext i32 %13 to i64		; visa id: 37
  %43 = add nuw i64 %41, %42		; visa id: 38
  %44 = bitcast i64 %43 to <2 x i32>		; visa id: 39
  %45 = extractelement <2 x i32> %44, i32 0		; visa id: 43
  %46 = extractelement <2 x i32> %44, i32 1		; visa id: 43
  %47 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %15, i32 0, i32 %18, i32 0)
  %48 = extractvalue { i32, i32 } %47, 0		; visa id: 43
  %49 = extractvalue { i32, i32 } %47, 1		; visa id: 43
  %50 = insertelement <2 x i32> undef, i32 %48, i32 0		; visa id: 50
  %51 = insertelement <2 x i32> %50, i32 %49, i32 1		; visa id: 51
  %52 = bitcast <2 x i32> %51 to i64		; visa id: 52
  %53 = zext i16 %localIdX to i64		; visa id: 54
  %54 = add nuw i64 %52, %53		; visa id: 55
  %55 = zext i32 %12 to i64		; visa id: 56
  %56 = add nuw i64 %54, %55		; visa id: 57
  %57 = bitcast i64 %56 to <2 x i32>		; visa id: 58
  %58 = extractelement <2 x i32> %57, i32 0		; visa id: 62
  %59 = extractelement <2 x i32> %57, i32 1		; visa id: 62
  %60 = icmp eq i32 %33, %5
  %61 = icmp ult i32 %32, %4		; visa id: 62
  %62 = and i1 %60, %61		; visa id: 63
  %63 = icmp ult i32 %33, %5
  %64 = or i1 %62, %63		; visa id: 65
  %65 = icmp eq i32 %46, %8
  %66 = icmp ult i32 %45, %7		; visa id: 67
  %67 = and i1 %65, %66		; visa id: 68
  %68 = icmp ult i32 %46, %8
  %69 = or i1 %67, %68		; visa id: 70
  %70 = icmp eq i32 %59, %11
  %71 = icmp ult i32 %58, %10		; visa id: 72
  %72 = and i1 %70, %71		; visa id: 73
  %73 = icmp ult i32 %59, %11
  %74 = or i1 %72, %73		; visa id: 75
  %75 = and i1 %74, %69		; visa id: 77
  %76 = and i1 %75, %64		; visa id: 78
  br i1 %76, label %.lr.ph.preheader, label %.._crit_edge101_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 79

.._crit_edge101_crit_edge:                        ; preds = %2
; BB:
  br label %._crit_edge101, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
; BB2 :
  %77 = bitcast i64 %const_reg_qword4 to <2 x i32>		; visa id: 81
  %78 = extractelement <2 x i32> %77, i32 0		; visa id: 82
  %79 = extractelement <2 x i32> %77, i32 1		; visa id: 82
  %80 = bitcast i64 %const_reg_qword5 to <2 x i32>		; visa id: 82
  %81 = extractelement <2 x i32> %80, i32 0		; visa id: 83
  %82 = extractelement <2 x i32> %80, i32 1		; visa id: 83
  %83 = bitcast i64 %const_reg_qword9 to <2 x i32>		; visa id: 83
  %84 = extractelement <2 x i32> %83, i32 0		; visa id: 84
  %85 = extractelement <2 x i32> %83, i32 1		; visa id: 84
  %86 = bitcast i64 %const_reg_qword10 to <2 x i32>		; visa id: 84
  %87 = extractelement <2 x i32> %86, i32 0		; visa id: 85
  %88 = extractelement <2 x i32> %86, i32 1		; visa id: 85
  %89 = extractelement <3 x i32> %globalSize, i32 0		; visa id: 85
  %90 = extractelement <3 x i32> %globalSize, i32 1		; visa id: 85
  %91 = extractelement <3 x i32> %globalSize, i32 2		; visa id: 85
  %92 = zext i32 %89 to i64		; visa id: 85
  %93 = zext i32 %90 to i64		; visa id: 86
  %94 = zext i32 %91 to i64		; visa id: 87
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879		; visa id: 90

.lr.ph:                                           ; preds = %.lr.ph.backedge, %.lr.ph.preheader
; BB3 :
  %95 = phi i64 [ %56, %.lr.ph.preheader ], [ %.be, %.lr.ph.backedge ]
  %96 = phi i64 [ %43, %.lr.ph.preheader ], [ %.be151, %.lr.ph.backedge ]
  %97 = phi i64 [ %30, %.lr.ph.preheader ], [ %.be152, %.lr.ph.backedge ]
  %98 = bitcast i64 %96 to <2 x i32>		; visa id: 91
  %99 = extractelement <2 x i32> %98, i32 0		; visa id: 95
  %100 = extractelement <2 x i32> %98, i32 1		; visa id: 95
  %101 = bitcast i64 %95 to <2 x i32>		; visa id: 95
  %102 = extractelement <2 x i32> %101, i32 0		; visa id: 99
  %103 = extractelement <2 x i32> %101, i32 1		; visa id: 99
  %104 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %102, i32 %103, i32 %84, i32 %85)
  %105 = extractvalue { i32, i32 } %104, 0		; visa id: 99
  %106 = extractvalue { i32, i32 } %104, 1		; visa id: 99
  %107 = insertelement <2 x i32> undef, i32 %105, i32 0		; visa id: 106
  %108 = insertelement <2 x i32> %107, i32 %106, i32 1		; visa id: 107
  %109 = bitcast <2 x i32> %108 to i64		; visa id: 108
  %110 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %99, i32 %100, i32 %87, i32 %88)
  %111 = extractvalue { i32, i32 } %110, 0		; visa id: 112
  %112 = extractvalue { i32, i32 } %110, 1		; visa id: 112
  %113 = insertelement <2 x i32> undef, i32 %111, i32 0		; visa id: 119
  %114 = insertelement <2 x i32> %113, i32 %112, i32 1		; visa id: 120
  %115 = bitcast <2 x i32> %114 to i64		; visa id: 121
  %116 = add i64 %109, %const_reg_qword8		; visa id: 125
  %117 = add i64 %116, %115		; visa id: 126
  %118 = add i64 %117, %97		; visa id: 127
  %119 = inttoptr i64 %118 to i8 addrspace(4)*		; visa id: 128
  %120 = addrspacecast i8 addrspace(4)* %119 to i8 addrspace(1)*		; visa id: 128
  %121 = load i8, i8 addrspace(1)* %120, align 1		; visa id: 129
  %122 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %102, i32 %103, i32 %78, i32 %79)
  %123 = extractvalue { i32, i32 } %122, 0		; visa id: 131
  %124 = extractvalue { i32, i32 } %122, 1		; visa id: 131
  %125 = insertelement <2 x i32> undef, i32 %123, i32 0		; visa id: 138
  %126 = insertelement <2 x i32> %125, i32 %124, i32 1		; visa id: 139
  %127 = bitcast <2 x i32> %126 to i64		; visa id: 140
  %128 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %99, i32 %100, i32 %81, i32 %82)
  %129 = extractvalue { i32, i32 } %128, 0		; visa id: 144
  %130 = extractvalue { i32, i32 } %128, 1		; visa id: 144
  %131 = insertelement <2 x i32> undef, i32 %129, i32 0		; visa id: 151
  %132 = insertelement <2 x i32> %131, i32 %130, i32 1		; visa id: 152
  %133 = bitcast <2 x i32> %132 to i64		; visa id: 153
  %134 = add i64 %127, %const_reg_qword3		; visa id: 157
  %135 = add i64 %134, %133		; visa id: 158
  %136 = add i64 %135, %97		; visa id: 159
  %137 = inttoptr i64 %136 to i8 addrspace(4)*		; visa id: 160
  %138 = addrspacecast i8 addrspace(4)* %137 to i8 addrspace(1)*		; visa id: 160
  store i8 %121, i8 addrspace(1)* %138, align 1		; visa id: 161
  %139 = add nuw nsw i64 %97, %94		; visa id: 163
  %140 = bitcast i64 %139 to <2 x i32>		; visa id: 164
  %141 = extractelement <2 x i32> %140, i32 0		; visa id: 168
  %142 = extractelement <2 x i32> %140, i32 1		; visa id: 168
  %143 = icmp eq i32 %142, %5
  %144 = icmp ult i32 %141, %4		; visa id: 168
  %145 = and i1 %143, %144		; visa id: 169
  %146 = icmp ult i32 %142, %5
  %147 = or i1 %145, %146		; visa id: 171
  br i1 %147, label %.lr.ph.._crit_edge99_crit_edge, label %148, !stats.blockFrequency.digits !881, !stats.blockFrequency.scale !879		; visa id: 173

.lr.ph.._crit_edge99_crit_edge:                   ; preds = %.lr.ph
; BB4 :
  br label %._crit_edge99, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879		; visa id: 177

148:                                              ; preds = %.lr.ph
; BB5 :
  %149 = add nuw nsw i64 %96, %93		; visa id: 179
  %150 = bitcast i64 %149 to <2 x i32>		; visa id: 180
  %151 = extractelement <2 x i32> %150, i32 0		; visa id: 184
  %152 = extractelement <2 x i32> %150, i32 1		; visa id: 184
  %153 = icmp eq i32 %152, %8
  %154 = icmp ult i32 %151, %7		; visa id: 184
  %155 = and i1 %153, %154		; visa id: 185
  %156 = icmp ult i32 %152, %8
  %157 = or i1 %155, %156		; visa id: 187
  br i1 %157, label %.._crit_edge99_crit_edge, label %160, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879		; visa id: 189

.._crit_edge99_crit_edge:                         ; preds = %148
; BB6 :
  br label %._crit_edge99, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 192

._crit_edge99:                                    ; preds = %.._crit_edge99_crit_edge, %.lr.ph.._crit_edge99_crit_edge
; BB:
  %158 = phi i64 [ %96, %.lr.ph.._crit_edge99_crit_edge ], [ %149, %.._crit_edge99_crit_edge ]
  %159 = phi i64 [ %139, %.lr.ph.._crit_edge99_crit_edge ], [ %30, %.._crit_edge99_crit_edge ]
  br label %.lr.ph.backedge, !stats.blockFrequency.digits !884, !stats.blockFrequency.scale !879

160:                                              ; preds = %148
; BB8 :
  %161 = add nuw nsw i64 %95, %92		; visa id: 194
  %162 = bitcast i64 %161 to <2 x i32>		; visa id: 195
  %163 = extractelement <2 x i32> %162, i32 0		; visa id: 199
  %164 = extractelement <2 x i32> %162, i32 1		; visa id: 199
  %165 = icmp eq i32 %164, %11
  %166 = icmp ult i32 %163, %10		; visa id: 199
  %167 = and i1 %165, %166		; visa id: 200
  %168 = icmp ult i32 %164, %11
  %169 = or i1 %167, %168		; visa id: 202
  br i1 %169, label %..lr.ph.backedge_crit_edge, label %._crit_edge101.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879		; visa id: 204

._crit_edge101.loopexit:                          ; preds = %160
; BB:
  br label %._crit_edge101, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

..lr.ph.backedge_crit_edge:                       ; preds = %160
; BB10 :
  %170 = select i1 %169, i32 %163, i32 %58		; visa id: 206
  %171 = select i1 %169, i32 %164, i32 %59		; visa id: 207
  %172 = insertelement <2 x i32> undef, i32 %170, i32 0		; visa id: 208
  %173 = insertelement <2 x i32> %172, i32 %171, i32 1		; visa id: 209
  %174 = bitcast <2 x i32> %173 to i64		; visa id: 210
  br label %.lr.ph.backedge, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879		; visa id: 216

.lr.ph.backedge:                                  ; preds = %..lr.ph.backedge_crit_edge, %._crit_edge99
; BB11 :
  %.be = phi i64 [ %95, %._crit_edge99 ], [ %174, %..lr.ph.backedge_crit_edge ]
  %.be151 = phi i64 [ %158, %._crit_edge99 ], [ %43, %..lr.ph.backedge_crit_edge ]
  %.be152 = phi i64 [ %159, %._crit_edge99 ], [ %30, %..lr.ph.backedge_crit_edge ]
  br label %.lr.ph, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879		; visa id: 219

._crit_edge101:                                   ; preds = %.._crit_edge101_crit_edge, %._crit_edge101.loopexit
; BB12 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 221
}

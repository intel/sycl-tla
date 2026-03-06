; ------------------------------------------------
; OCL_asm23954d4a795eca46_1_simd32_entry_0011.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
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
  br i1 %39, label %.preheader2.preheader.preheader, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !614, !stats.blockFrequency.scale !615		; visa id: 36

.._crit_edge72_crit_edge:                         ; preds = %13
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !616, !stats.blockFrequency.scale !615

.preheader2.preheader.preheader:                  ; preds = %13
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
  %52 = call i16 @llvm.genx.GenISA.simdLaneId()		; visa id: 42
  %53 = call i32 @llvm.genx.GenISA.simdSize()
  %54 = call i32 @llvm.genx.GenISA.hw.thread.id.alloca.i32()		; visa id: 45
  %55 = mul i32 %53, 48
  %56 = mul i32 %54, %55, !perThreadOffset !617		; visa id: 53
  %57 = extractelement <3 x i32> %numWorkGroups, i32 2		; visa id: 54
  %58 = extractelement <3 x i32> %localSize, i32 0		; visa id: 54
  %59 = extractelement <3 x i32> %localSize, i32 1		; visa id: 54
  %60 = extractelement <8 x i32> %r0, i32 1		; visa id: 54
  %61 = extractelement <8 x i32> %r0, i32 6		; visa id: 55
  %62 = mul i32 %60, %58		; visa id: 56
  %63 = zext i16 %localIdX to i32		; visa id: 57
  %64 = add i32 %62, %63		; visa id: 58
  %65 = shl i32 %64, 2		; visa id: 59
  %66 = mul i32 %61, %59		; visa id: 60
  %67 = zext i16 %localIdY to i32		; visa id: 61
  %68 = add i32 %66, %67		; visa id: 62
  %69 = shl i32 %68, 4		; visa id: 63
  %70 = insertelement <2 x i32> undef, i32 %28, i32 0		; visa id: 64
  %71 = insertelement <2 x i32> %70, i32 %29, i32 1		; visa id: 65
  %72 = bitcast <2 x i32> %71 to i64		; visa id: 66
  %73 = shl i64 %72, 1		; visa id: 68
  %74 = add i64 %73, %const_reg_qword		; visa id: 69
  %75 = insertelement <2 x i32> undef, i32 %31, i32 0		; visa id: 70
  %76 = insertelement <2 x i32> %75, i32 %32, i32 1		; visa id: 71
  %77 = bitcast <2 x i32> %76 to i64		; visa id: 72
  %78 = shl i64 %77, 1		; visa id: 74
  %79 = add i64 %78, %const_reg_qword4		; visa id: 75
  %80 = insertelement <2 x i32> undef, i32 %34, i32 0		; visa id: 76
  %81 = insertelement <2 x i32> %80, i32 %35, i32 1		; visa id: 77
  %82 = bitcast <2 x i32> %81 to i64		; visa id: 78
  %.op = shl i64 %82, 2		; visa id: 80
  %83 = bitcast i64 %.op to <2 x i32>		; visa id: 81
  %84 = extractelement <2 x i32> %83, i32 0		; visa id: 82
  %85 = extractelement <2 x i32> %83, i32 1		; visa id: 82
  %86 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !618		; visa id: 82
  %87 = select i1 %86, i32 %84, i32 0		; visa id: 83
  %88 = select i1 %86, i32 %85, i32 0		; visa id: 84
  %89 = insertelement <2 x i32> undef, i32 %87, i32 0		; visa id: 85
  %90 = insertelement <2 x i32> %89, i32 %88, i32 1		; visa id: 86
  %91 = bitcast <2 x i32> %90 to i64		; visa id: 87
  %92 = add i64 %91, %const_reg_qword6		; visa id: 89
  %93 = insertelement <2 x i32> undef, i32 %37, i32 0		; visa id: 90
  %94 = insertelement <2 x i32> %93, i32 %38, i32 1		; visa id: 91
  %95 = bitcast <2 x i32> %94 to i64		; visa id: 92
  %96 = shl i64 %95, 2		; visa id: 94
  %97 = add i64 %96, %const_reg_qword8		; visa id: 95
  %98 = zext i16 %52 to i32		; visa id: 96
  %99 = add i32 %53, %98		; visa id: 97
  %100 = shl i32 %99, 3		; visa id: 98
  %101 = mul i32 %53, 40
  %102 = shl nuw nsw i32 %98, 3		; visa id: 99
  %103 = add i32 %101, %102
  %104 = shl i32 %53, 4
  %105 = add i32 %104, %102
  %106 = shl i32 %53, 5
  %107 = add i32 %106, %102
  %108 = mul i32 %53, 24
  %109 = add i32 %108, %102
  br label %.preheader2.preheader, !stats.blockFrequency.digits !616, !stats.blockFrequency.scale !615		; visa id: 100

.preheader2.preheader:                            ; preds = %.preheader1.15..preheader2.preheader_crit_edge, %.preheader2.preheader.preheader
; BB3 :
  %110 = phi i32 [ %9796, %.preheader1.15..preheader2.preheader_crit_edge ], [ %26, %.preheader2.preheader.preheader ]
  %.in = phi i64 [ %9824, %.preheader1.15..preheader2.preheader_crit_edge ], [ %97, %.preheader2.preheader.preheader ]
  %.in399 = phi i64 [ %9819, %.preheader1.15..preheader2.preheader_crit_edge ], [ %92, %.preheader2.preheader.preheader ]
  %.in400 = phi i64 [ %9807, %.preheader1.15..preheader2.preheader_crit_edge ], [ %79, %.preheader2.preheader.preheader ]
  %.in401 = phi i64 [ %9802, %.preheader1.15..preheader2.preheader_crit_edge ], [ %74, %.preheader2.preheader.preheader ]
  %111 = ptrtoint i8* %privateBase to i64		; visa id: 101
  %112 = icmp sgt i32 %const_reg_dword2, 0		; visa id: 101
  br i1 %112, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 102

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
; BB4 :
  br label %.preheader1.preheader, !stats.blockFrequency.digits !621, !stats.blockFrequency.scale !615		; visa id: 168

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
; BB5 :
  br label %.preheader.preheader, !stats.blockFrequency.digits !622, !stats.blockFrequency.scale !615		; visa id: 235

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
  %113 = phi i32 [ %5974, %.preheader.15..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
  %114 = icmp slt i32 %65, %const_reg_dword
  %115 = icmp slt i32 %69, %const_reg_dword1		; visa id: 236
  %116 = and i1 %114, %115		; visa id: 237
  %117 = add nuw nsw i32 %56, %107		; visa id: 239
  %118 = zext i32 %117 to i64		; visa id: 240
  %119 = add i64 %111, %118		; visa id: 241
  %120 = inttoptr i64 %119 to i64*		; visa id: 242
  %121 = inttoptr i64 %119 to i8*		; visa id: 242
  %122 = add nuw nsw i32 %56, %105		; visa id: 242
  %123 = zext i32 %122 to i64		; visa id: 243
  %124 = add i64 %111, %123		; visa id: 244
  %125 = inttoptr i64 %124 to i8*		; visa id: 245
  %126 = add nuw nsw i32 %56, %103		; visa id: 245
  %127 = zext i32 %126 to i64		; visa id: 246
  %128 = add i64 %111, %127		; visa id: 247
  %129 = inttoptr i64 %128 to i64*		; visa id: 248
  %130 = inttoptr i64 %128 to i8*		; visa id: 248
  %131 = add nuw nsw i32 %56, %100		; visa id: 248
  %132 = zext i32 %131 to i64		; visa id: 249
  %133 = add i64 %111, %132		; visa id: 250
  %134 = inttoptr i64 %133 to i8*		; visa id: 251
  br i1 %116, label %135, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 251

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

135:                                              ; preds = %.preheader.preheader
; BB8 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 253
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 253
  %136 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 253
  %137 = insertelement <2 x i32> %136, i32 %113, i64 1		; visa id: 254
  %138 = inttoptr i64 %133 to <2 x i32>*		; visa id: 255
  store <2 x i32> %137, <2 x i32>* %138, align 4, !noalias !625		; visa id: 255
  br label %._crit_edge207, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 257

._crit_edge207:                                   ; preds = %._crit_edge207.._crit_edge207_crit_edge, %135
; BB9 :
  %139 = phi i32 [ 0, %135 ], [ %148, %._crit_edge207.._crit_edge207_crit_edge ]
  %140 = zext i32 %139 to i64		; visa id: 258
  %141 = shl nuw nsw i64 %140, 2		; visa id: 259
  %142 = add i64 %133, %141		; visa id: 260
  %143 = inttoptr i64 %142 to i32*		; visa id: 261
  %144 = load i32, i32* %143, align 4, !noalias !625		; visa id: 261
  %145 = add i64 %128, %141		; visa id: 262
  %146 = inttoptr i64 %145 to i32*		; visa id: 263
  store i32 %144, i32* %146, align 4, !alias.scope !625		; visa id: 263
  %147 = icmp eq i32 %139, 0		; visa id: 264
  br i1 %147, label %._crit_edge207.._crit_edge207_crit_edge, label %149, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 265

._crit_edge207.._crit_edge207_crit_edge:          ; preds = %._crit_edge207
; BB10 :
  %148 = add nuw nsw i32 %139, 1, !spirv.Decorations !631		; visa id: 267
  br label %._crit_edge207, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 268

149:                                              ; preds = %._crit_edge207
; BB11 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 270
  %150 = load i64, i64* %129, align 8		; visa id: 270
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 271
  %151 = bitcast i64 %150 to <2 x i32>		; visa id: 271
  %152 = extractelement <2 x i32> %151, i32 0		; visa id: 273
  %153 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %152, i32 1
  %154 = bitcast <2 x i32> %153 to i64		; visa id: 273
  %155 = ashr exact i64 %154, 32		; visa id: 274
  %156 = bitcast i64 %155 to <2 x i32>		; visa id: 275
  %157 = extractelement <2 x i32> %156, i32 0		; visa id: 279
  %158 = extractelement <2 x i32> %156, i32 1		; visa id: 279
  %159 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %157, i32 %158, i32 %41, i32 %42)
  %160 = extractvalue { i32, i32 } %159, 0		; visa id: 279
  %161 = extractvalue { i32, i32 } %159, 1		; visa id: 279
  %162 = insertelement <2 x i32> undef, i32 %160, i32 0		; visa id: 286
  %163 = insertelement <2 x i32> %162, i32 %161, i32 1		; visa id: 287
  %164 = bitcast <2 x i32> %163 to i64		; visa id: 288
  %165 = shl i64 %164, 1		; visa id: 292
  %166 = add i64 %.in401, %165		; visa id: 293
  %167 = ashr i64 %150, 31		; visa id: 294
  %168 = bitcast i64 %167 to <2 x i32>		; visa id: 295
  %169 = extractelement <2 x i32> %168, i32 0		; visa id: 299
  %170 = extractelement <2 x i32> %168, i32 1		; visa id: 299
  %171 = and i32 %169, -2		; visa id: 299
  %172 = insertelement <2 x i32> undef, i32 %171, i32 0		; visa id: 300
  %173 = insertelement <2 x i32> %172, i32 %170, i32 1		; visa id: 301
  %174 = bitcast <2 x i32> %173 to i64		; visa id: 302
  %175 = add i64 %166, %174		; visa id: 306
  %176 = inttoptr i64 %175 to i16 addrspace(4)*		; visa id: 307
  %177 = addrspacecast i16 addrspace(4)* %176 to i16 addrspace(1)*		; visa id: 307
  %178 = load i16, i16 addrspace(1)* %177, align 2		; visa id: 308
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 310
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 310
  %179 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 310
  %180 = insertelement <2 x i32> %179, i32 %69, i64 1		; visa id: 311
  %181 = inttoptr i64 %124 to <2 x i32>*		; visa id: 312
  store <2 x i32> %180, <2 x i32>* %181, align 4, !noalias !635		; visa id: 312
  br label %._crit_edge208, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 314

._crit_edge208:                                   ; preds = %._crit_edge208.._crit_edge208_crit_edge, %149
; BB12 :
  %182 = phi i32 [ 0, %149 ], [ %191, %._crit_edge208.._crit_edge208_crit_edge ]
  %183 = zext i32 %182 to i64		; visa id: 315
  %184 = shl nuw nsw i64 %183, 2		; visa id: 316
  %185 = add i64 %124, %184		; visa id: 317
  %186 = inttoptr i64 %185 to i32*		; visa id: 318
  %187 = load i32, i32* %186, align 4, !noalias !635		; visa id: 318
  %188 = add i64 %119, %184		; visa id: 319
  %189 = inttoptr i64 %188 to i32*		; visa id: 320
  store i32 %187, i32* %189, align 4, !alias.scope !635		; visa id: 320
  %190 = icmp eq i32 %182, 0		; visa id: 321
  br i1 %190, label %._crit_edge208.._crit_edge208_crit_edge, label %192, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 322

._crit_edge208.._crit_edge208_crit_edge:          ; preds = %._crit_edge208
; BB13 :
  %191 = add nuw nsw i32 %182, 1, !spirv.Decorations !631		; visa id: 324
  br label %._crit_edge208, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 325

192:                                              ; preds = %._crit_edge208
; BB14 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 327
  %193 = load i64, i64* %120, align 8		; visa id: 327
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 328
  %194 = ashr i64 %193, 32		; visa id: 328
  %195 = bitcast i64 %194 to <2 x i32>		; visa id: 329
  %196 = extractelement <2 x i32> %195, i32 0		; visa id: 333
  %197 = extractelement <2 x i32> %195, i32 1		; visa id: 333
  %198 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %196, i32 %197, i32 %44, i32 %45)
  %199 = extractvalue { i32, i32 } %198, 0		; visa id: 333
  %200 = extractvalue { i32, i32 } %198, 1		; visa id: 333
  %201 = insertelement <2 x i32> undef, i32 %199, i32 0		; visa id: 340
  %202 = insertelement <2 x i32> %201, i32 %200, i32 1		; visa id: 341
  %203 = bitcast <2 x i32> %202 to i64		; visa id: 342
  %204 = bitcast i64 %193 to <2 x i32>		; visa id: 346
  %205 = extractelement <2 x i32> %204, i32 0		; visa id: 348
  %206 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %205, i32 1
  %207 = bitcast <2 x i32> %206 to i64		; visa id: 348
  %208 = shl i64 %203, 1		; visa id: 349
  %209 = add i64 %.in400, %208		; visa id: 350
  %210 = ashr exact i64 %207, 31		; visa id: 351
  %211 = add i64 %209, %210		; visa id: 352
  %212 = inttoptr i64 %211 to i16 addrspace(4)*		; visa id: 353
  %213 = addrspacecast i16 addrspace(4)* %212 to i16 addrspace(1)*		; visa id: 353
  %214 = load i16, i16 addrspace(1)* %213, align 2		; visa id: 354
  %215 = zext i16 %178 to i32		; visa id: 356
  %216 = shl nuw i32 %215, 16, !spirv.Decorations !639		; visa id: 357
  %217 = bitcast i32 %216 to float
  %218 = zext i16 %214 to i32		; visa id: 358
  %219 = shl nuw i32 %218, 16, !spirv.Decorations !639		; visa id: 359
  %220 = bitcast i32 %219 to float
  %221 = fmul reassoc nsz arcp contract float %217, %220, !spirv.Decorations !618
  %222 = fadd reassoc nsz arcp contract float %221, %.sroa.0.1, !spirv.Decorations !618		; visa id: 360
  br label %._crit_edge, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 361

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %192
; BB15 :
  %.sroa.0.2 = phi float [ %222, %192 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  %223 = add i32 %65, 1		; visa id: 362
  %224 = icmp slt i32 %223, %const_reg_dword
  %225 = icmp slt i32 %69, %const_reg_dword1		; visa id: 363
  %226 = and i1 %224, %225		; visa id: 364
  br i1 %226, label %227, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 366

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
; BB:
  br label %._crit_edge.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

227:                                              ; preds = %._crit_edge
; BB17 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 368
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 368
  %228 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 368
  %229 = insertelement <2 x i32> %228, i32 %113, i64 1		; visa id: 369
  %230 = inttoptr i64 %133 to <2 x i32>*		; visa id: 370
  store <2 x i32> %229, <2 x i32>* %230, align 4, !noalias !625		; visa id: 370
  br label %._crit_edge209, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 372

._crit_edge209:                                   ; preds = %._crit_edge209.._crit_edge209_crit_edge, %227
; BB18 :
  %231 = phi i32 [ 0, %227 ], [ %240, %._crit_edge209.._crit_edge209_crit_edge ]
  %232 = zext i32 %231 to i64		; visa id: 373
  %233 = shl nuw nsw i64 %232, 2		; visa id: 374
  %234 = add i64 %133, %233		; visa id: 375
  %235 = inttoptr i64 %234 to i32*		; visa id: 376
  %236 = load i32, i32* %235, align 4, !noalias !625		; visa id: 376
  %237 = add i64 %128, %233		; visa id: 377
  %238 = inttoptr i64 %237 to i32*		; visa id: 378
  store i32 %236, i32* %238, align 4, !alias.scope !625		; visa id: 378
  %239 = icmp eq i32 %231, 0		; visa id: 379
  br i1 %239, label %._crit_edge209.._crit_edge209_crit_edge, label %241, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 380

._crit_edge209.._crit_edge209_crit_edge:          ; preds = %._crit_edge209
; BB19 :
  %240 = add nuw nsw i32 %231, 1, !spirv.Decorations !631		; visa id: 382
  br label %._crit_edge209, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 383

241:                                              ; preds = %._crit_edge209
; BB20 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 385
  %242 = load i64, i64* %129, align 8		; visa id: 385
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 386
  %243 = bitcast i64 %242 to <2 x i32>		; visa id: 386
  %244 = extractelement <2 x i32> %243, i32 0		; visa id: 388
  %245 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %244, i32 1
  %246 = bitcast <2 x i32> %245 to i64		; visa id: 388
  %247 = ashr exact i64 %246, 32		; visa id: 389
  %248 = bitcast i64 %247 to <2 x i32>		; visa id: 390
  %249 = extractelement <2 x i32> %248, i32 0		; visa id: 394
  %250 = extractelement <2 x i32> %248, i32 1		; visa id: 394
  %251 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %249, i32 %250, i32 %41, i32 %42)
  %252 = extractvalue { i32, i32 } %251, 0		; visa id: 394
  %253 = extractvalue { i32, i32 } %251, 1		; visa id: 394
  %254 = insertelement <2 x i32> undef, i32 %252, i32 0		; visa id: 401
  %255 = insertelement <2 x i32> %254, i32 %253, i32 1		; visa id: 402
  %256 = bitcast <2 x i32> %255 to i64		; visa id: 403
  %257 = shl i64 %256, 1		; visa id: 407
  %258 = add i64 %.in401, %257		; visa id: 408
  %259 = ashr i64 %242, 31		; visa id: 409
  %260 = bitcast i64 %259 to <2 x i32>		; visa id: 410
  %261 = extractelement <2 x i32> %260, i32 0		; visa id: 414
  %262 = extractelement <2 x i32> %260, i32 1		; visa id: 414
  %263 = and i32 %261, -2		; visa id: 414
  %264 = insertelement <2 x i32> undef, i32 %263, i32 0		; visa id: 415
  %265 = insertelement <2 x i32> %264, i32 %262, i32 1		; visa id: 416
  %266 = bitcast <2 x i32> %265 to i64		; visa id: 417
  %267 = add i64 %258, %266		; visa id: 421
  %268 = inttoptr i64 %267 to i16 addrspace(4)*		; visa id: 422
  %269 = addrspacecast i16 addrspace(4)* %268 to i16 addrspace(1)*		; visa id: 422
  %270 = load i16, i16 addrspace(1)* %269, align 2		; visa id: 423
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 425
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 425
  %271 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 425
  %272 = insertelement <2 x i32> %271, i32 %69, i64 1		; visa id: 426
  %273 = inttoptr i64 %124 to <2 x i32>*		; visa id: 427
  store <2 x i32> %272, <2 x i32>* %273, align 4, !noalias !635		; visa id: 427
  br label %._crit_edge210, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 429

._crit_edge210:                                   ; preds = %._crit_edge210.._crit_edge210_crit_edge, %241
; BB21 :
  %274 = phi i32 [ 0, %241 ], [ %283, %._crit_edge210.._crit_edge210_crit_edge ]
  %275 = zext i32 %274 to i64		; visa id: 430
  %276 = shl nuw nsw i64 %275, 2		; visa id: 431
  %277 = add i64 %124, %276		; visa id: 432
  %278 = inttoptr i64 %277 to i32*		; visa id: 433
  %279 = load i32, i32* %278, align 4, !noalias !635		; visa id: 433
  %280 = add i64 %119, %276		; visa id: 434
  %281 = inttoptr i64 %280 to i32*		; visa id: 435
  store i32 %279, i32* %281, align 4, !alias.scope !635		; visa id: 435
  %282 = icmp eq i32 %274, 0		; visa id: 436
  br i1 %282, label %._crit_edge210.._crit_edge210_crit_edge, label %284, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 437

._crit_edge210.._crit_edge210_crit_edge:          ; preds = %._crit_edge210
; BB22 :
  %283 = add nuw nsw i32 %274, 1, !spirv.Decorations !631		; visa id: 439
  br label %._crit_edge210, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 440

284:                                              ; preds = %._crit_edge210
; BB23 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 442
  %285 = load i64, i64* %120, align 8		; visa id: 442
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 443
  %286 = ashr i64 %285, 32		; visa id: 443
  %287 = bitcast i64 %286 to <2 x i32>		; visa id: 444
  %288 = extractelement <2 x i32> %287, i32 0		; visa id: 448
  %289 = extractelement <2 x i32> %287, i32 1		; visa id: 448
  %290 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %288, i32 %289, i32 %44, i32 %45)
  %291 = extractvalue { i32, i32 } %290, 0		; visa id: 448
  %292 = extractvalue { i32, i32 } %290, 1		; visa id: 448
  %293 = insertelement <2 x i32> undef, i32 %291, i32 0		; visa id: 455
  %294 = insertelement <2 x i32> %293, i32 %292, i32 1		; visa id: 456
  %295 = bitcast <2 x i32> %294 to i64		; visa id: 457
  %296 = bitcast i64 %285 to <2 x i32>		; visa id: 461
  %297 = extractelement <2 x i32> %296, i32 0		; visa id: 463
  %298 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %297, i32 1
  %299 = bitcast <2 x i32> %298 to i64		; visa id: 463
  %300 = shl i64 %295, 1		; visa id: 464
  %301 = add i64 %.in400, %300		; visa id: 465
  %302 = ashr exact i64 %299, 31		; visa id: 466
  %303 = add i64 %301, %302		; visa id: 467
  %304 = inttoptr i64 %303 to i16 addrspace(4)*		; visa id: 468
  %305 = addrspacecast i16 addrspace(4)* %304 to i16 addrspace(1)*		; visa id: 468
  %306 = load i16, i16 addrspace(1)* %305, align 2		; visa id: 469
  %307 = zext i16 %270 to i32		; visa id: 471
  %308 = shl nuw i32 %307, 16, !spirv.Decorations !639		; visa id: 472
  %309 = bitcast i32 %308 to float
  %310 = zext i16 %306 to i32		; visa id: 473
  %311 = shl nuw i32 %310, 16, !spirv.Decorations !639		; visa id: 474
  %312 = bitcast i32 %311 to float
  %313 = fmul reassoc nsz arcp contract float %309, %312, !spirv.Decorations !618
  %314 = fadd reassoc nsz arcp contract float %313, %.sroa.66.1, !spirv.Decorations !618		; visa id: 475
  br label %._crit_edge.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 476

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %284
; BB24 :
  %.sroa.66.2 = phi float [ %314, %284 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  %315 = add i32 %65, 2		; visa id: 477
  %316 = icmp slt i32 %315, %const_reg_dword
  %317 = icmp slt i32 %69, %const_reg_dword1		; visa id: 478
  %318 = and i1 %316, %317		; visa id: 479
  br i1 %318, label %319, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 481

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
; BB:
  br label %._crit_edge.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

319:                                              ; preds = %._crit_edge.1
; BB26 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 483
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 483
  %320 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 483
  %321 = insertelement <2 x i32> %320, i32 %113, i64 1		; visa id: 484
  %322 = inttoptr i64 %133 to <2 x i32>*		; visa id: 485
  store <2 x i32> %321, <2 x i32>* %322, align 4, !noalias !625		; visa id: 485
  br label %._crit_edge211, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 487

._crit_edge211:                                   ; preds = %._crit_edge211.._crit_edge211_crit_edge, %319
; BB27 :
  %323 = phi i32 [ 0, %319 ], [ %332, %._crit_edge211.._crit_edge211_crit_edge ]
  %324 = zext i32 %323 to i64		; visa id: 488
  %325 = shl nuw nsw i64 %324, 2		; visa id: 489
  %326 = add i64 %133, %325		; visa id: 490
  %327 = inttoptr i64 %326 to i32*		; visa id: 491
  %328 = load i32, i32* %327, align 4, !noalias !625		; visa id: 491
  %329 = add i64 %128, %325		; visa id: 492
  %330 = inttoptr i64 %329 to i32*		; visa id: 493
  store i32 %328, i32* %330, align 4, !alias.scope !625		; visa id: 493
  %331 = icmp eq i32 %323, 0		; visa id: 494
  br i1 %331, label %._crit_edge211.._crit_edge211_crit_edge, label %333, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 495

._crit_edge211.._crit_edge211_crit_edge:          ; preds = %._crit_edge211
; BB28 :
  %332 = add nuw nsw i32 %323, 1, !spirv.Decorations !631		; visa id: 497
  br label %._crit_edge211, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 498

333:                                              ; preds = %._crit_edge211
; BB29 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 500
  %334 = load i64, i64* %129, align 8		; visa id: 500
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 501
  %335 = bitcast i64 %334 to <2 x i32>		; visa id: 501
  %336 = extractelement <2 x i32> %335, i32 0		; visa id: 503
  %337 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %336, i32 1
  %338 = bitcast <2 x i32> %337 to i64		; visa id: 503
  %339 = ashr exact i64 %338, 32		; visa id: 504
  %340 = bitcast i64 %339 to <2 x i32>		; visa id: 505
  %341 = extractelement <2 x i32> %340, i32 0		; visa id: 509
  %342 = extractelement <2 x i32> %340, i32 1		; visa id: 509
  %343 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %341, i32 %342, i32 %41, i32 %42)
  %344 = extractvalue { i32, i32 } %343, 0		; visa id: 509
  %345 = extractvalue { i32, i32 } %343, 1		; visa id: 509
  %346 = insertelement <2 x i32> undef, i32 %344, i32 0		; visa id: 516
  %347 = insertelement <2 x i32> %346, i32 %345, i32 1		; visa id: 517
  %348 = bitcast <2 x i32> %347 to i64		; visa id: 518
  %349 = shl i64 %348, 1		; visa id: 522
  %350 = add i64 %.in401, %349		; visa id: 523
  %351 = ashr i64 %334, 31		; visa id: 524
  %352 = bitcast i64 %351 to <2 x i32>		; visa id: 525
  %353 = extractelement <2 x i32> %352, i32 0		; visa id: 529
  %354 = extractelement <2 x i32> %352, i32 1		; visa id: 529
  %355 = and i32 %353, -2		; visa id: 529
  %356 = insertelement <2 x i32> undef, i32 %355, i32 0		; visa id: 530
  %357 = insertelement <2 x i32> %356, i32 %354, i32 1		; visa id: 531
  %358 = bitcast <2 x i32> %357 to i64		; visa id: 532
  %359 = add i64 %350, %358		; visa id: 536
  %360 = inttoptr i64 %359 to i16 addrspace(4)*		; visa id: 537
  %361 = addrspacecast i16 addrspace(4)* %360 to i16 addrspace(1)*		; visa id: 537
  %362 = load i16, i16 addrspace(1)* %361, align 2		; visa id: 538
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 540
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 540
  %363 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 540
  %364 = insertelement <2 x i32> %363, i32 %69, i64 1		; visa id: 541
  %365 = inttoptr i64 %124 to <2 x i32>*		; visa id: 542
  store <2 x i32> %364, <2 x i32>* %365, align 4, !noalias !635		; visa id: 542
  br label %._crit_edge212, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 544

._crit_edge212:                                   ; preds = %._crit_edge212.._crit_edge212_crit_edge, %333
; BB30 :
  %366 = phi i32 [ 0, %333 ], [ %375, %._crit_edge212.._crit_edge212_crit_edge ]
  %367 = zext i32 %366 to i64		; visa id: 545
  %368 = shl nuw nsw i64 %367, 2		; visa id: 546
  %369 = add i64 %124, %368		; visa id: 547
  %370 = inttoptr i64 %369 to i32*		; visa id: 548
  %371 = load i32, i32* %370, align 4, !noalias !635		; visa id: 548
  %372 = add i64 %119, %368		; visa id: 549
  %373 = inttoptr i64 %372 to i32*		; visa id: 550
  store i32 %371, i32* %373, align 4, !alias.scope !635		; visa id: 550
  %374 = icmp eq i32 %366, 0		; visa id: 551
  br i1 %374, label %._crit_edge212.._crit_edge212_crit_edge, label %376, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 552

._crit_edge212.._crit_edge212_crit_edge:          ; preds = %._crit_edge212
; BB31 :
  %375 = add nuw nsw i32 %366, 1, !spirv.Decorations !631		; visa id: 554
  br label %._crit_edge212, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 555

376:                                              ; preds = %._crit_edge212
; BB32 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 557
  %377 = load i64, i64* %120, align 8		; visa id: 557
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 558
  %378 = ashr i64 %377, 32		; visa id: 558
  %379 = bitcast i64 %378 to <2 x i32>		; visa id: 559
  %380 = extractelement <2 x i32> %379, i32 0		; visa id: 563
  %381 = extractelement <2 x i32> %379, i32 1		; visa id: 563
  %382 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %380, i32 %381, i32 %44, i32 %45)
  %383 = extractvalue { i32, i32 } %382, 0		; visa id: 563
  %384 = extractvalue { i32, i32 } %382, 1		; visa id: 563
  %385 = insertelement <2 x i32> undef, i32 %383, i32 0		; visa id: 570
  %386 = insertelement <2 x i32> %385, i32 %384, i32 1		; visa id: 571
  %387 = bitcast <2 x i32> %386 to i64		; visa id: 572
  %388 = bitcast i64 %377 to <2 x i32>		; visa id: 576
  %389 = extractelement <2 x i32> %388, i32 0		; visa id: 578
  %390 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %389, i32 1
  %391 = bitcast <2 x i32> %390 to i64		; visa id: 578
  %392 = shl i64 %387, 1		; visa id: 579
  %393 = add i64 %.in400, %392		; visa id: 580
  %394 = ashr exact i64 %391, 31		; visa id: 581
  %395 = add i64 %393, %394		; visa id: 582
  %396 = inttoptr i64 %395 to i16 addrspace(4)*		; visa id: 583
  %397 = addrspacecast i16 addrspace(4)* %396 to i16 addrspace(1)*		; visa id: 583
  %398 = load i16, i16 addrspace(1)* %397, align 2		; visa id: 584
  %399 = zext i16 %362 to i32		; visa id: 586
  %400 = shl nuw i32 %399, 16, !spirv.Decorations !639		; visa id: 587
  %401 = bitcast i32 %400 to float
  %402 = zext i16 %398 to i32		; visa id: 588
  %403 = shl nuw i32 %402, 16, !spirv.Decorations !639		; visa id: 589
  %404 = bitcast i32 %403 to float
  %405 = fmul reassoc nsz arcp contract float %401, %404, !spirv.Decorations !618
  %406 = fadd reassoc nsz arcp contract float %405, %.sroa.130.1, !spirv.Decorations !618		; visa id: 590
  br label %._crit_edge.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 591

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %376
; BB33 :
  %.sroa.130.2 = phi float [ %406, %376 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  %407 = add i32 %65, 3		; visa id: 592
  %408 = icmp slt i32 %407, %const_reg_dword
  %409 = icmp slt i32 %69, %const_reg_dword1		; visa id: 593
  %410 = and i1 %408, %409		; visa id: 594
  br i1 %410, label %411, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 596

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
; BB:
  br label %.preheader, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

411:                                              ; preds = %._crit_edge.2
; BB35 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 598
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 598
  %412 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 598
  %413 = insertelement <2 x i32> %412, i32 %113, i64 1		; visa id: 599
  %414 = inttoptr i64 %133 to <2 x i32>*		; visa id: 600
  store <2 x i32> %413, <2 x i32>* %414, align 4, !noalias !625		; visa id: 600
  br label %._crit_edge213, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 602

._crit_edge213:                                   ; preds = %._crit_edge213.._crit_edge213_crit_edge, %411
; BB36 :
  %415 = phi i32 [ 0, %411 ], [ %424, %._crit_edge213.._crit_edge213_crit_edge ]
  %416 = zext i32 %415 to i64		; visa id: 603
  %417 = shl nuw nsw i64 %416, 2		; visa id: 604
  %418 = add i64 %133, %417		; visa id: 605
  %419 = inttoptr i64 %418 to i32*		; visa id: 606
  %420 = load i32, i32* %419, align 4, !noalias !625		; visa id: 606
  %421 = add i64 %128, %417		; visa id: 607
  %422 = inttoptr i64 %421 to i32*		; visa id: 608
  store i32 %420, i32* %422, align 4, !alias.scope !625		; visa id: 608
  %423 = icmp eq i32 %415, 0		; visa id: 609
  br i1 %423, label %._crit_edge213.._crit_edge213_crit_edge, label %425, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 610

._crit_edge213.._crit_edge213_crit_edge:          ; preds = %._crit_edge213
; BB37 :
  %424 = add nuw nsw i32 %415, 1, !spirv.Decorations !631		; visa id: 612
  br label %._crit_edge213, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 613

425:                                              ; preds = %._crit_edge213
; BB38 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 615
  %426 = load i64, i64* %129, align 8		; visa id: 615
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 616
  %427 = bitcast i64 %426 to <2 x i32>		; visa id: 616
  %428 = extractelement <2 x i32> %427, i32 0		; visa id: 618
  %429 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %428, i32 1
  %430 = bitcast <2 x i32> %429 to i64		; visa id: 618
  %431 = ashr exact i64 %430, 32		; visa id: 619
  %432 = bitcast i64 %431 to <2 x i32>		; visa id: 620
  %433 = extractelement <2 x i32> %432, i32 0		; visa id: 624
  %434 = extractelement <2 x i32> %432, i32 1		; visa id: 624
  %435 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %433, i32 %434, i32 %41, i32 %42)
  %436 = extractvalue { i32, i32 } %435, 0		; visa id: 624
  %437 = extractvalue { i32, i32 } %435, 1		; visa id: 624
  %438 = insertelement <2 x i32> undef, i32 %436, i32 0		; visa id: 631
  %439 = insertelement <2 x i32> %438, i32 %437, i32 1		; visa id: 632
  %440 = bitcast <2 x i32> %439 to i64		; visa id: 633
  %441 = shl i64 %440, 1		; visa id: 637
  %442 = add i64 %.in401, %441		; visa id: 638
  %443 = ashr i64 %426, 31		; visa id: 639
  %444 = bitcast i64 %443 to <2 x i32>		; visa id: 640
  %445 = extractelement <2 x i32> %444, i32 0		; visa id: 644
  %446 = extractelement <2 x i32> %444, i32 1		; visa id: 644
  %447 = and i32 %445, -2		; visa id: 644
  %448 = insertelement <2 x i32> undef, i32 %447, i32 0		; visa id: 645
  %449 = insertelement <2 x i32> %448, i32 %446, i32 1		; visa id: 646
  %450 = bitcast <2 x i32> %449 to i64		; visa id: 647
  %451 = add i64 %442, %450		; visa id: 651
  %452 = inttoptr i64 %451 to i16 addrspace(4)*		; visa id: 652
  %453 = addrspacecast i16 addrspace(4)* %452 to i16 addrspace(1)*		; visa id: 652
  %454 = load i16, i16 addrspace(1)* %453, align 2		; visa id: 653
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 655
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 655
  %455 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 655
  %456 = insertelement <2 x i32> %455, i32 %69, i64 1		; visa id: 656
  %457 = inttoptr i64 %124 to <2 x i32>*		; visa id: 657
  store <2 x i32> %456, <2 x i32>* %457, align 4, !noalias !635		; visa id: 657
  br label %._crit_edge214, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 659

._crit_edge214:                                   ; preds = %._crit_edge214.._crit_edge214_crit_edge, %425
; BB39 :
  %458 = phi i32 [ 0, %425 ], [ %467, %._crit_edge214.._crit_edge214_crit_edge ]
  %459 = zext i32 %458 to i64		; visa id: 660
  %460 = shl nuw nsw i64 %459, 2		; visa id: 661
  %461 = add i64 %124, %460		; visa id: 662
  %462 = inttoptr i64 %461 to i32*		; visa id: 663
  %463 = load i32, i32* %462, align 4, !noalias !635		; visa id: 663
  %464 = add i64 %119, %460		; visa id: 664
  %465 = inttoptr i64 %464 to i32*		; visa id: 665
  store i32 %463, i32* %465, align 4, !alias.scope !635		; visa id: 665
  %466 = icmp eq i32 %458, 0		; visa id: 666
  br i1 %466, label %._crit_edge214.._crit_edge214_crit_edge, label %468, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 667

._crit_edge214.._crit_edge214_crit_edge:          ; preds = %._crit_edge214
; BB40 :
  %467 = add nuw nsw i32 %458, 1, !spirv.Decorations !631		; visa id: 669
  br label %._crit_edge214, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 670

468:                                              ; preds = %._crit_edge214
; BB41 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 672
  %469 = load i64, i64* %120, align 8		; visa id: 672
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 673
  %470 = ashr i64 %469, 32		; visa id: 673
  %471 = bitcast i64 %470 to <2 x i32>		; visa id: 674
  %472 = extractelement <2 x i32> %471, i32 0		; visa id: 678
  %473 = extractelement <2 x i32> %471, i32 1		; visa id: 678
  %474 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %472, i32 %473, i32 %44, i32 %45)
  %475 = extractvalue { i32, i32 } %474, 0		; visa id: 678
  %476 = extractvalue { i32, i32 } %474, 1		; visa id: 678
  %477 = insertelement <2 x i32> undef, i32 %475, i32 0		; visa id: 685
  %478 = insertelement <2 x i32> %477, i32 %476, i32 1		; visa id: 686
  %479 = bitcast <2 x i32> %478 to i64		; visa id: 687
  %480 = bitcast i64 %469 to <2 x i32>		; visa id: 691
  %481 = extractelement <2 x i32> %480, i32 0		; visa id: 693
  %482 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %481, i32 1
  %483 = bitcast <2 x i32> %482 to i64		; visa id: 693
  %484 = shl i64 %479, 1		; visa id: 694
  %485 = add i64 %.in400, %484		; visa id: 695
  %486 = ashr exact i64 %483, 31		; visa id: 696
  %487 = add i64 %485, %486		; visa id: 697
  %488 = inttoptr i64 %487 to i16 addrspace(4)*		; visa id: 698
  %489 = addrspacecast i16 addrspace(4)* %488 to i16 addrspace(1)*		; visa id: 698
  %490 = load i16, i16 addrspace(1)* %489, align 2		; visa id: 699
  %491 = zext i16 %454 to i32		; visa id: 701
  %492 = shl nuw i32 %491, 16, !spirv.Decorations !639		; visa id: 702
  %493 = bitcast i32 %492 to float
  %494 = zext i16 %490 to i32		; visa id: 703
  %495 = shl nuw i32 %494, 16, !spirv.Decorations !639		; visa id: 704
  %496 = bitcast i32 %495 to float
  %497 = fmul reassoc nsz arcp contract float %493, %496, !spirv.Decorations !618
  %498 = fadd reassoc nsz arcp contract float %497, %.sroa.194.1, !spirv.Decorations !618		; visa id: 705
  br label %.preheader, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 706

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %468
; BB42 :
  %.sroa.194.2 = phi float [ %498, %468 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %499 = add i32 %69, 1		; visa id: 707
  %500 = icmp slt i32 %499, %const_reg_dword1		; visa id: 708
  %501 = icmp slt i32 %65, %const_reg_dword
  %502 = and i1 %501, %500		; visa id: 709
  br i1 %502, label %503, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 711

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
; BB:
  br label %._crit_edge.173, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

503:                                              ; preds = %.preheader
; BB44 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 713
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 713
  %504 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 713
  %505 = insertelement <2 x i32> %504, i32 %113, i64 1		; visa id: 714
  %506 = inttoptr i64 %133 to <2 x i32>*		; visa id: 715
  store <2 x i32> %505, <2 x i32>* %506, align 4, !noalias !625		; visa id: 715
  br label %._crit_edge215, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 717

._crit_edge215:                                   ; preds = %._crit_edge215.._crit_edge215_crit_edge, %503
; BB45 :
  %507 = phi i32 [ 0, %503 ], [ %516, %._crit_edge215.._crit_edge215_crit_edge ]
  %508 = zext i32 %507 to i64		; visa id: 718
  %509 = shl nuw nsw i64 %508, 2		; visa id: 719
  %510 = add i64 %133, %509		; visa id: 720
  %511 = inttoptr i64 %510 to i32*		; visa id: 721
  %512 = load i32, i32* %511, align 4, !noalias !625		; visa id: 721
  %513 = add i64 %128, %509		; visa id: 722
  %514 = inttoptr i64 %513 to i32*		; visa id: 723
  store i32 %512, i32* %514, align 4, !alias.scope !625		; visa id: 723
  %515 = icmp eq i32 %507, 0		; visa id: 724
  br i1 %515, label %._crit_edge215.._crit_edge215_crit_edge, label %517, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 725

._crit_edge215.._crit_edge215_crit_edge:          ; preds = %._crit_edge215
; BB46 :
  %516 = add nuw nsw i32 %507, 1, !spirv.Decorations !631		; visa id: 727
  br label %._crit_edge215, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 728

517:                                              ; preds = %._crit_edge215
; BB47 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 730
  %518 = load i64, i64* %129, align 8		; visa id: 730
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 731
  %519 = bitcast i64 %518 to <2 x i32>		; visa id: 731
  %520 = extractelement <2 x i32> %519, i32 0		; visa id: 733
  %521 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %520, i32 1
  %522 = bitcast <2 x i32> %521 to i64		; visa id: 733
  %523 = ashr exact i64 %522, 32		; visa id: 734
  %524 = bitcast i64 %523 to <2 x i32>		; visa id: 735
  %525 = extractelement <2 x i32> %524, i32 0		; visa id: 739
  %526 = extractelement <2 x i32> %524, i32 1		; visa id: 739
  %527 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %525, i32 %526, i32 %41, i32 %42)
  %528 = extractvalue { i32, i32 } %527, 0		; visa id: 739
  %529 = extractvalue { i32, i32 } %527, 1		; visa id: 739
  %530 = insertelement <2 x i32> undef, i32 %528, i32 0		; visa id: 746
  %531 = insertelement <2 x i32> %530, i32 %529, i32 1		; visa id: 747
  %532 = bitcast <2 x i32> %531 to i64		; visa id: 748
  %533 = shl i64 %532, 1		; visa id: 752
  %534 = add i64 %.in401, %533		; visa id: 753
  %535 = ashr i64 %518, 31		; visa id: 754
  %536 = bitcast i64 %535 to <2 x i32>		; visa id: 755
  %537 = extractelement <2 x i32> %536, i32 0		; visa id: 759
  %538 = extractelement <2 x i32> %536, i32 1		; visa id: 759
  %539 = and i32 %537, -2		; visa id: 759
  %540 = insertelement <2 x i32> undef, i32 %539, i32 0		; visa id: 760
  %541 = insertelement <2 x i32> %540, i32 %538, i32 1		; visa id: 761
  %542 = bitcast <2 x i32> %541 to i64		; visa id: 762
  %543 = add i64 %534, %542		; visa id: 766
  %544 = inttoptr i64 %543 to i16 addrspace(4)*		; visa id: 767
  %545 = addrspacecast i16 addrspace(4)* %544 to i16 addrspace(1)*		; visa id: 767
  %546 = load i16, i16 addrspace(1)* %545, align 2		; visa id: 768
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 770
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 770
  %547 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 770
  %548 = insertelement <2 x i32> %547, i32 %499, i64 1		; visa id: 771
  %549 = inttoptr i64 %124 to <2 x i32>*		; visa id: 772
  store <2 x i32> %548, <2 x i32>* %549, align 4, !noalias !635		; visa id: 772
  br label %._crit_edge216, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 774

._crit_edge216:                                   ; preds = %._crit_edge216.._crit_edge216_crit_edge, %517
; BB48 :
  %550 = phi i32 [ 0, %517 ], [ %559, %._crit_edge216.._crit_edge216_crit_edge ]
  %551 = zext i32 %550 to i64		; visa id: 775
  %552 = shl nuw nsw i64 %551, 2		; visa id: 776
  %553 = add i64 %124, %552		; visa id: 777
  %554 = inttoptr i64 %553 to i32*		; visa id: 778
  %555 = load i32, i32* %554, align 4, !noalias !635		; visa id: 778
  %556 = add i64 %119, %552		; visa id: 779
  %557 = inttoptr i64 %556 to i32*		; visa id: 780
  store i32 %555, i32* %557, align 4, !alias.scope !635		; visa id: 780
  %558 = icmp eq i32 %550, 0		; visa id: 781
  br i1 %558, label %._crit_edge216.._crit_edge216_crit_edge, label %560, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 782

._crit_edge216.._crit_edge216_crit_edge:          ; preds = %._crit_edge216
; BB49 :
  %559 = add nuw nsw i32 %550, 1, !spirv.Decorations !631		; visa id: 784
  br label %._crit_edge216, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 785

560:                                              ; preds = %._crit_edge216
; BB50 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 787
  %561 = load i64, i64* %120, align 8		; visa id: 787
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 788
  %562 = ashr i64 %561, 32		; visa id: 788
  %563 = bitcast i64 %562 to <2 x i32>		; visa id: 789
  %564 = extractelement <2 x i32> %563, i32 0		; visa id: 793
  %565 = extractelement <2 x i32> %563, i32 1		; visa id: 793
  %566 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %564, i32 %565, i32 %44, i32 %45)
  %567 = extractvalue { i32, i32 } %566, 0		; visa id: 793
  %568 = extractvalue { i32, i32 } %566, 1		; visa id: 793
  %569 = insertelement <2 x i32> undef, i32 %567, i32 0		; visa id: 800
  %570 = insertelement <2 x i32> %569, i32 %568, i32 1		; visa id: 801
  %571 = bitcast <2 x i32> %570 to i64		; visa id: 802
  %572 = bitcast i64 %561 to <2 x i32>		; visa id: 806
  %573 = extractelement <2 x i32> %572, i32 0		; visa id: 808
  %574 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %573, i32 1
  %575 = bitcast <2 x i32> %574 to i64		; visa id: 808
  %576 = shl i64 %571, 1		; visa id: 809
  %577 = add i64 %.in400, %576		; visa id: 810
  %578 = ashr exact i64 %575, 31		; visa id: 811
  %579 = add i64 %577, %578		; visa id: 812
  %580 = inttoptr i64 %579 to i16 addrspace(4)*		; visa id: 813
  %581 = addrspacecast i16 addrspace(4)* %580 to i16 addrspace(1)*		; visa id: 813
  %582 = load i16, i16 addrspace(1)* %581, align 2		; visa id: 814
  %583 = zext i16 %546 to i32		; visa id: 816
  %584 = shl nuw i32 %583, 16, !spirv.Decorations !639		; visa id: 817
  %585 = bitcast i32 %584 to float
  %586 = zext i16 %582 to i32		; visa id: 818
  %587 = shl nuw i32 %586, 16, !spirv.Decorations !639		; visa id: 819
  %588 = bitcast i32 %587 to float
  %589 = fmul reassoc nsz arcp contract float %585, %588, !spirv.Decorations !618
  %590 = fadd reassoc nsz arcp contract float %589, %.sroa.6.1, !spirv.Decorations !618		; visa id: 820
  br label %._crit_edge.173, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 821

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %560
; BB51 :
  %.sroa.6.2 = phi float [ %590, %560 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  %591 = icmp slt i32 %223, %const_reg_dword
  %592 = icmp slt i32 %499, %const_reg_dword1		; visa id: 822
  %593 = and i1 %591, %592		; visa id: 823
  br i1 %593, label %594, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 825

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
; BB:
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

594:                                              ; preds = %._crit_edge.173
; BB53 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 827
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 827
  %595 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 827
  %596 = insertelement <2 x i32> %595, i32 %113, i64 1		; visa id: 828
  %597 = inttoptr i64 %133 to <2 x i32>*		; visa id: 829
  store <2 x i32> %596, <2 x i32>* %597, align 4, !noalias !625		; visa id: 829
  br label %._crit_edge217, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 831

._crit_edge217:                                   ; preds = %._crit_edge217.._crit_edge217_crit_edge, %594
; BB54 :
  %598 = phi i32 [ 0, %594 ], [ %607, %._crit_edge217.._crit_edge217_crit_edge ]
  %599 = zext i32 %598 to i64		; visa id: 832
  %600 = shl nuw nsw i64 %599, 2		; visa id: 833
  %601 = add i64 %133, %600		; visa id: 834
  %602 = inttoptr i64 %601 to i32*		; visa id: 835
  %603 = load i32, i32* %602, align 4, !noalias !625		; visa id: 835
  %604 = add i64 %128, %600		; visa id: 836
  %605 = inttoptr i64 %604 to i32*		; visa id: 837
  store i32 %603, i32* %605, align 4, !alias.scope !625		; visa id: 837
  %606 = icmp eq i32 %598, 0		; visa id: 838
  br i1 %606, label %._crit_edge217.._crit_edge217_crit_edge, label %608, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 839

._crit_edge217.._crit_edge217_crit_edge:          ; preds = %._crit_edge217
; BB55 :
  %607 = add nuw nsw i32 %598, 1, !spirv.Decorations !631		; visa id: 841
  br label %._crit_edge217, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 842

608:                                              ; preds = %._crit_edge217
; BB56 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 844
  %609 = load i64, i64* %129, align 8		; visa id: 844
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 845
  %610 = bitcast i64 %609 to <2 x i32>		; visa id: 845
  %611 = extractelement <2 x i32> %610, i32 0		; visa id: 847
  %612 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %611, i32 1
  %613 = bitcast <2 x i32> %612 to i64		; visa id: 847
  %614 = ashr exact i64 %613, 32		; visa id: 848
  %615 = bitcast i64 %614 to <2 x i32>		; visa id: 849
  %616 = extractelement <2 x i32> %615, i32 0		; visa id: 853
  %617 = extractelement <2 x i32> %615, i32 1		; visa id: 853
  %618 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %616, i32 %617, i32 %41, i32 %42)
  %619 = extractvalue { i32, i32 } %618, 0		; visa id: 853
  %620 = extractvalue { i32, i32 } %618, 1		; visa id: 853
  %621 = insertelement <2 x i32> undef, i32 %619, i32 0		; visa id: 860
  %622 = insertelement <2 x i32> %621, i32 %620, i32 1		; visa id: 861
  %623 = bitcast <2 x i32> %622 to i64		; visa id: 862
  %624 = shl i64 %623, 1		; visa id: 866
  %625 = add i64 %.in401, %624		; visa id: 867
  %626 = ashr i64 %609, 31		; visa id: 868
  %627 = bitcast i64 %626 to <2 x i32>		; visa id: 869
  %628 = extractelement <2 x i32> %627, i32 0		; visa id: 873
  %629 = extractelement <2 x i32> %627, i32 1		; visa id: 873
  %630 = and i32 %628, -2		; visa id: 873
  %631 = insertelement <2 x i32> undef, i32 %630, i32 0		; visa id: 874
  %632 = insertelement <2 x i32> %631, i32 %629, i32 1		; visa id: 875
  %633 = bitcast <2 x i32> %632 to i64		; visa id: 876
  %634 = add i64 %625, %633		; visa id: 880
  %635 = inttoptr i64 %634 to i16 addrspace(4)*		; visa id: 881
  %636 = addrspacecast i16 addrspace(4)* %635 to i16 addrspace(1)*		; visa id: 881
  %637 = load i16, i16 addrspace(1)* %636, align 2		; visa id: 882
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 884
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 884
  %638 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 884
  %639 = insertelement <2 x i32> %638, i32 %499, i64 1		; visa id: 885
  %640 = inttoptr i64 %124 to <2 x i32>*		; visa id: 886
  store <2 x i32> %639, <2 x i32>* %640, align 4, !noalias !635		; visa id: 886
  br label %._crit_edge218, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 888

._crit_edge218:                                   ; preds = %._crit_edge218.._crit_edge218_crit_edge, %608
; BB57 :
  %641 = phi i32 [ 0, %608 ], [ %650, %._crit_edge218.._crit_edge218_crit_edge ]
  %642 = zext i32 %641 to i64		; visa id: 889
  %643 = shl nuw nsw i64 %642, 2		; visa id: 890
  %644 = add i64 %124, %643		; visa id: 891
  %645 = inttoptr i64 %644 to i32*		; visa id: 892
  %646 = load i32, i32* %645, align 4, !noalias !635		; visa id: 892
  %647 = add i64 %119, %643		; visa id: 893
  %648 = inttoptr i64 %647 to i32*		; visa id: 894
  store i32 %646, i32* %648, align 4, !alias.scope !635		; visa id: 894
  %649 = icmp eq i32 %641, 0		; visa id: 895
  br i1 %649, label %._crit_edge218.._crit_edge218_crit_edge, label %651, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 896

._crit_edge218.._crit_edge218_crit_edge:          ; preds = %._crit_edge218
; BB58 :
  %650 = add nuw nsw i32 %641, 1, !spirv.Decorations !631		; visa id: 898
  br label %._crit_edge218, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 899

651:                                              ; preds = %._crit_edge218
; BB59 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 901
  %652 = load i64, i64* %120, align 8		; visa id: 901
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 902
  %653 = ashr i64 %652, 32		; visa id: 902
  %654 = bitcast i64 %653 to <2 x i32>		; visa id: 903
  %655 = extractelement <2 x i32> %654, i32 0		; visa id: 907
  %656 = extractelement <2 x i32> %654, i32 1		; visa id: 907
  %657 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %655, i32 %656, i32 %44, i32 %45)
  %658 = extractvalue { i32, i32 } %657, 0		; visa id: 907
  %659 = extractvalue { i32, i32 } %657, 1		; visa id: 907
  %660 = insertelement <2 x i32> undef, i32 %658, i32 0		; visa id: 914
  %661 = insertelement <2 x i32> %660, i32 %659, i32 1		; visa id: 915
  %662 = bitcast <2 x i32> %661 to i64		; visa id: 916
  %663 = bitcast i64 %652 to <2 x i32>		; visa id: 920
  %664 = extractelement <2 x i32> %663, i32 0		; visa id: 922
  %665 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %664, i32 1
  %666 = bitcast <2 x i32> %665 to i64		; visa id: 922
  %667 = shl i64 %662, 1		; visa id: 923
  %668 = add i64 %.in400, %667		; visa id: 924
  %669 = ashr exact i64 %666, 31		; visa id: 925
  %670 = add i64 %668, %669		; visa id: 926
  %671 = inttoptr i64 %670 to i16 addrspace(4)*		; visa id: 927
  %672 = addrspacecast i16 addrspace(4)* %671 to i16 addrspace(1)*		; visa id: 927
  %673 = load i16, i16 addrspace(1)* %672, align 2		; visa id: 928
  %674 = zext i16 %637 to i32		; visa id: 930
  %675 = shl nuw i32 %674, 16, !spirv.Decorations !639		; visa id: 931
  %676 = bitcast i32 %675 to float
  %677 = zext i16 %673 to i32		; visa id: 932
  %678 = shl nuw i32 %677, 16, !spirv.Decorations !639		; visa id: 933
  %679 = bitcast i32 %678 to float
  %680 = fmul reassoc nsz arcp contract float %676, %679, !spirv.Decorations !618
  %681 = fadd reassoc nsz arcp contract float %680, %.sroa.70.1, !spirv.Decorations !618		; visa id: 934
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 935

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %651
; BB60 :
  %.sroa.70.2 = phi float [ %681, %651 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  %682 = icmp slt i32 %315, %const_reg_dword
  %683 = icmp slt i32 %499, %const_reg_dword1		; visa id: 936
  %684 = and i1 %682, %683		; visa id: 937
  br i1 %684, label %685, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 939

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
; BB:
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

685:                                              ; preds = %._crit_edge.1.1
; BB62 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 941
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 941
  %686 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 941
  %687 = insertelement <2 x i32> %686, i32 %113, i64 1		; visa id: 942
  %688 = inttoptr i64 %133 to <2 x i32>*		; visa id: 943
  store <2 x i32> %687, <2 x i32>* %688, align 4, !noalias !625		; visa id: 943
  br label %._crit_edge219, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 945

._crit_edge219:                                   ; preds = %._crit_edge219.._crit_edge219_crit_edge, %685
; BB63 :
  %689 = phi i32 [ 0, %685 ], [ %698, %._crit_edge219.._crit_edge219_crit_edge ]
  %690 = zext i32 %689 to i64		; visa id: 946
  %691 = shl nuw nsw i64 %690, 2		; visa id: 947
  %692 = add i64 %133, %691		; visa id: 948
  %693 = inttoptr i64 %692 to i32*		; visa id: 949
  %694 = load i32, i32* %693, align 4, !noalias !625		; visa id: 949
  %695 = add i64 %128, %691		; visa id: 950
  %696 = inttoptr i64 %695 to i32*		; visa id: 951
  store i32 %694, i32* %696, align 4, !alias.scope !625		; visa id: 951
  %697 = icmp eq i32 %689, 0		; visa id: 952
  br i1 %697, label %._crit_edge219.._crit_edge219_crit_edge, label %699, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 953

._crit_edge219.._crit_edge219_crit_edge:          ; preds = %._crit_edge219
; BB64 :
  %698 = add nuw nsw i32 %689, 1, !spirv.Decorations !631		; visa id: 955
  br label %._crit_edge219, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 956

699:                                              ; preds = %._crit_edge219
; BB65 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 958
  %700 = load i64, i64* %129, align 8		; visa id: 958
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 959
  %701 = bitcast i64 %700 to <2 x i32>		; visa id: 959
  %702 = extractelement <2 x i32> %701, i32 0		; visa id: 961
  %703 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %702, i32 1
  %704 = bitcast <2 x i32> %703 to i64		; visa id: 961
  %705 = ashr exact i64 %704, 32		; visa id: 962
  %706 = bitcast i64 %705 to <2 x i32>		; visa id: 963
  %707 = extractelement <2 x i32> %706, i32 0		; visa id: 967
  %708 = extractelement <2 x i32> %706, i32 1		; visa id: 967
  %709 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %707, i32 %708, i32 %41, i32 %42)
  %710 = extractvalue { i32, i32 } %709, 0		; visa id: 967
  %711 = extractvalue { i32, i32 } %709, 1		; visa id: 967
  %712 = insertelement <2 x i32> undef, i32 %710, i32 0		; visa id: 974
  %713 = insertelement <2 x i32> %712, i32 %711, i32 1		; visa id: 975
  %714 = bitcast <2 x i32> %713 to i64		; visa id: 976
  %715 = shl i64 %714, 1		; visa id: 980
  %716 = add i64 %.in401, %715		; visa id: 981
  %717 = ashr i64 %700, 31		; visa id: 982
  %718 = bitcast i64 %717 to <2 x i32>		; visa id: 983
  %719 = extractelement <2 x i32> %718, i32 0		; visa id: 987
  %720 = extractelement <2 x i32> %718, i32 1		; visa id: 987
  %721 = and i32 %719, -2		; visa id: 987
  %722 = insertelement <2 x i32> undef, i32 %721, i32 0		; visa id: 988
  %723 = insertelement <2 x i32> %722, i32 %720, i32 1		; visa id: 989
  %724 = bitcast <2 x i32> %723 to i64		; visa id: 990
  %725 = add i64 %716, %724		; visa id: 994
  %726 = inttoptr i64 %725 to i16 addrspace(4)*		; visa id: 995
  %727 = addrspacecast i16 addrspace(4)* %726 to i16 addrspace(1)*		; visa id: 995
  %728 = load i16, i16 addrspace(1)* %727, align 2		; visa id: 996
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 998
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 998
  %729 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 998
  %730 = insertelement <2 x i32> %729, i32 %499, i64 1		; visa id: 999
  %731 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1000
  store <2 x i32> %730, <2 x i32>* %731, align 4, !noalias !635		; visa id: 1000
  br label %._crit_edge220, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1002

._crit_edge220:                                   ; preds = %._crit_edge220.._crit_edge220_crit_edge, %699
; BB66 :
  %732 = phi i32 [ 0, %699 ], [ %741, %._crit_edge220.._crit_edge220_crit_edge ]
  %733 = zext i32 %732 to i64		; visa id: 1003
  %734 = shl nuw nsw i64 %733, 2		; visa id: 1004
  %735 = add i64 %124, %734		; visa id: 1005
  %736 = inttoptr i64 %735 to i32*		; visa id: 1006
  %737 = load i32, i32* %736, align 4, !noalias !635		; visa id: 1006
  %738 = add i64 %119, %734		; visa id: 1007
  %739 = inttoptr i64 %738 to i32*		; visa id: 1008
  store i32 %737, i32* %739, align 4, !alias.scope !635		; visa id: 1008
  %740 = icmp eq i32 %732, 0		; visa id: 1009
  br i1 %740, label %._crit_edge220.._crit_edge220_crit_edge, label %742, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1010

._crit_edge220.._crit_edge220_crit_edge:          ; preds = %._crit_edge220
; BB67 :
  %741 = add nuw nsw i32 %732, 1, !spirv.Decorations !631		; visa id: 1012
  br label %._crit_edge220, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1013

742:                                              ; preds = %._crit_edge220
; BB68 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1015
  %743 = load i64, i64* %120, align 8		; visa id: 1015
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1016
  %744 = ashr i64 %743, 32		; visa id: 1016
  %745 = bitcast i64 %744 to <2 x i32>		; visa id: 1017
  %746 = extractelement <2 x i32> %745, i32 0		; visa id: 1021
  %747 = extractelement <2 x i32> %745, i32 1		; visa id: 1021
  %748 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %746, i32 %747, i32 %44, i32 %45)
  %749 = extractvalue { i32, i32 } %748, 0		; visa id: 1021
  %750 = extractvalue { i32, i32 } %748, 1		; visa id: 1021
  %751 = insertelement <2 x i32> undef, i32 %749, i32 0		; visa id: 1028
  %752 = insertelement <2 x i32> %751, i32 %750, i32 1		; visa id: 1029
  %753 = bitcast <2 x i32> %752 to i64		; visa id: 1030
  %754 = bitcast i64 %743 to <2 x i32>		; visa id: 1034
  %755 = extractelement <2 x i32> %754, i32 0		; visa id: 1036
  %756 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %755, i32 1
  %757 = bitcast <2 x i32> %756 to i64		; visa id: 1036
  %758 = shl i64 %753, 1		; visa id: 1037
  %759 = add i64 %.in400, %758		; visa id: 1038
  %760 = ashr exact i64 %757, 31		; visa id: 1039
  %761 = add i64 %759, %760		; visa id: 1040
  %762 = inttoptr i64 %761 to i16 addrspace(4)*		; visa id: 1041
  %763 = addrspacecast i16 addrspace(4)* %762 to i16 addrspace(1)*		; visa id: 1041
  %764 = load i16, i16 addrspace(1)* %763, align 2		; visa id: 1042
  %765 = zext i16 %728 to i32		; visa id: 1044
  %766 = shl nuw i32 %765, 16, !spirv.Decorations !639		; visa id: 1045
  %767 = bitcast i32 %766 to float
  %768 = zext i16 %764 to i32		; visa id: 1046
  %769 = shl nuw i32 %768, 16, !spirv.Decorations !639		; visa id: 1047
  %770 = bitcast i32 %769 to float
  %771 = fmul reassoc nsz arcp contract float %767, %770, !spirv.Decorations !618
  %772 = fadd reassoc nsz arcp contract float %771, %.sroa.134.1, !spirv.Decorations !618		; visa id: 1048
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1049

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %742
; BB69 :
  %.sroa.134.2 = phi float [ %772, %742 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  %773 = icmp slt i32 %407, %const_reg_dword
  %774 = icmp slt i32 %499, %const_reg_dword1		; visa id: 1050
  %775 = and i1 %773, %774		; visa id: 1051
  br i1 %775, label %776, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1053

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
; BB:
  br label %.preheader.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

776:                                              ; preds = %._crit_edge.2.1
; BB71 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1055
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1055
  %777 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 1055
  %778 = insertelement <2 x i32> %777, i32 %113, i64 1		; visa id: 1056
  %779 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1057
  store <2 x i32> %778, <2 x i32>* %779, align 4, !noalias !625		; visa id: 1057
  br label %._crit_edge221, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1059

._crit_edge221:                                   ; preds = %._crit_edge221.._crit_edge221_crit_edge, %776
; BB72 :
  %780 = phi i32 [ 0, %776 ], [ %789, %._crit_edge221.._crit_edge221_crit_edge ]
  %781 = zext i32 %780 to i64		; visa id: 1060
  %782 = shl nuw nsw i64 %781, 2		; visa id: 1061
  %783 = add i64 %133, %782		; visa id: 1062
  %784 = inttoptr i64 %783 to i32*		; visa id: 1063
  %785 = load i32, i32* %784, align 4, !noalias !625		; visa id: 1063
  %786 = add i64 %128, %782		; visa id: 1064
  %787 = inttoptr i64 %786 to i32*		; visa id: 1065
  store i32 %785, i32* %787, align 4, !alias.scope !625		; visa id: 1065
  %788 = icmp eq i32 %780, 0		; visa id: 1066
  br i1 %788, label %._crit_edge221.._crit_edge221_crit_edge, label %790, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1067

._crit_edge221.._crit_edge221_crit_edge:          ; preds = %._crit_edge221
; BB73 :
  %789 = add nuw nsw i32 %780, 1, !spirv.Decorations !631		; visa id: 1069
  br label %._crit_edge221, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1070

790:                                              ; preds = %._crit_edge221
; BB74 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1072
  %791 = load i64, i64* %129, align 8		; visa id: 1072
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1073
  %792 = bitcast i64 %791 to <2 x i32>		; visa id: 1073
  %793 = extractelement <2 x i32> %792, i32 0		; visa id: 1075
  %794 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %793, i32 1
  %795 = bitcast <2 x i32> %794 to i64		; visa id: 1075
  %796 = ashr exact i64 %795, 32		; visa id: 1076
  %797 = bitcast i64 %796 to <2 x i32>		; visa id: 1077
  %798 = extractelement <2 x i32> %797, i32 0		; visa id: 1081
  %799 = extractelement <2 x i32> %797, i32 1		; visa id: 1081
  %800 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %798, i32 %799, i32 %41, i32 %42)
  %801 = extractvalue { i32, i32 } %800, 0		; visa id: 1081
  %802 = extractvalue { i32, i32 } %800, 1		; visa id: 1081
  %803 = insertelement <2 x i32> undef, i32 %801, i32 0		; visa id: 1088
  %804 = insertelement <2 x i32> %803, i32 %802, i32 1		; visa id: 1089
  %805 = bitcast <2 x i32> %804 to i64		; visa id: 1090
  %806 = shl i64 %805, 1		; visa id: 1094
  %807 = add i64 %.in401, %806		; visa id: 1095
  %808 = ashr i64 %791, 31		; visa id: 1096
  %809 = bitcast i64 %808 to <2 x i32>		; visa id: 1097
  %810 = extractelement <2 x i32> %809, i32 0		; visa id: 1101
  %811 = extractelement <2 x i32> %809, i32 1		; visa id: 1101
  %812 = and i32 %810, -2		; visa id: 1101
  %813 = insertelement <2 x i32> undef, i32 %812, i32 0		; visa id: 1102
  %814 = insertelement <2 x i32> %813, i32 %811, i32 1		; visa id: 1103
  %815 = bitcast <2 x i32> %814 to i64		; visa id: 1104
  %816 = add i64 %807, %815		; visa id: 1108
  %817 = inttoptr i64 %816 to i16 addrspace(4)*		; visa id: 1109
  %818 = addrspacecast i16 addrspace(4)* %817 to i16 addrspace(1)*		; visa id: 1109
  %819 = load i16, i16 addrspace(1)* %818, align 2		; visa id: 1110
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1112
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1112
  %820 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1112
  %821 = insertelement <2 x i32> %820, i32 %499, i64 1		; visa id: 1113
  %822 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1114
  store <2 x i32> %821, <2 x i32>* %822, align 4, !noalias !635		; visa id: 1114
  br label %._crit_edge222, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1116

._crit_edge222:                                   ; preds = %._crit_edge222.._crit_edge222_crit_edge, %790
; BB75 :
  %823 = phi i32 [ 0, %790 ], [ %832, %._crit_edge222.._crit_edge222_crit_edge ]
  %824 = zext i32 %823 to i64		; visa id: 1117
  %825 = shl nuw nsw i64 %824, 2		; visa id: 1118
  %826 = add i64 %124, %825		; visa id: 1119
  %827 = inttoptr i64 %826 to i32*		; visa id: 1120
  %828 = load i32, i32* %827, align 4, !noalias !635		; visa id: 1120
  %829 = add i64 %119, %825		; visa id: 1121
  %830 = inttoptr i64 %829 to i32*		; visa id: 1122
  store i32 %828, i32* %830, align 4, !alias.scope !635		; visa id: 1122
  %831 = icmp eq i32 %823, 0		; visa id: 1123
  br i1 %831, label %._crit_edge222.._crit_edge222_crit_edge, label %833, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1124

._crit_edge222.._crit_edge222_crit_edge:          ; preds = %._crit_edge222
; BB76 :
  %832 = add nuw nsw i32 %823, 1, !spirv.Decorations !631		; visa id: 1126
  br label %._crit_edge222, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1127

833:                                              ; preds = %._crit_edge222
; BB77 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1129
  %834 = load i64, i64* %120, align 8		; visa id: 1129
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1130
  %835 = ashr i64 %834, 32		; visa id: 1130
  %836 = bitcast i64 %835 to <2 x i32>		; visa id: 1131
  %837 = extractelement <2 x i32> %836, i32 0		; visa id: 1135
  %838 = extractelement <2 x i32> %836, i32 1		; visa id: 1135
  %839 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %837, i32 %838, i32 %44, i32 %45)
  %840 = extractvalue { i32, i32 } %839, 0		; visa id: 1135
  %841 = extractvalue { i32, i32 } %839, 1		; visa id: 1135
  %842 = insertelement <2 x i32> undef, i32 %840, i32 0		; visa id: 1142
  %843 = insertelement <2 x i32> %842, i32 %841, i32 1		; visa id: 1143
  %844 = bitcast <2 x i32> %843 to i64		; visa id: 1144
  %845 = bitcast i64 %834 to <2 x i32>		; visa id: 1148
  %846 = extractelement <2 x i32> %845, i32 0		; visa id: 1150
  %847 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %846, i32 1
  %848 = bitcast <2 x i32> %847 to i64		; visa id: 1150
  %849 = shl i64 %844, 1		; visa id: 1151
  %850 = add i64 %.in400, %849		; visa id: 1152
  %851 = ashr exact i64 %848, 31		; visa id: 1153
  %852 = add i64 %850, %851		; visa id: 1154
  %853 = inttoptr i64 %852 to i16 addrspace(4)*		; visa id: 1155
  %854 = addrspacecast i16 addrspace(4)* %853 to i16 addrspace(1)*		; visa id: 1155
  %855 = load i16, i16 addrspace(1)* %854, align 2		; visa id: 1156
  %856 = zext i16 %819 to i32		; visa id: 1158
  %857 = shl nuw i32 %856, 16, !spirv.Decorations !639		; visa id: 1159
  %858 = bitcast i32 %857 to float
  %859 = zext i16 %855 to i32		; visa id: 1160
  %860 = shl nuw i32 %859, 16, !spirv.Decorations !639		; visa id: 1161
  %861 = bitcast i32 %860 to float
  %862 = fmul reassoc nsz arcp contract float %858, %861, !spirv.Decorations !618
  %863 = fadd reassoc nsz arcp contract float %862, %.sroa.198.1, !spirv.Decorations !618		; visa id: 1162
  br label %.preheader.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1163

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %833
; BB78 :
  %.sroa.198.2 = phi float [ %863, %833 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %864 = add i32 %69, 2		; visa id: 1164
  %865 = icmp slt i32 %864, %const_reg_dword1		; visa id: 1165
  %866 = icmp slt i32 %65, %const_reg_dword
  %867 = and i1 %866, %865		; visa id: 1166
  br i1 %867, label %868, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1168

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
; BB:
  br label %._crit_edge.274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

868:                                              ; preds = %.preheader.1
; BB80 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1170
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1170
  %869 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 1170
  %870 = insertelement <2 x i32> %869, i32 %113, i64 1		; visa id: 1171
  %871 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1172
  store <2 x i32> %870, <2 x i32>* %871, align 4, !noalias !625		; visa id: 1172
  br label %._crit_edge223, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1174

._crit_edge223:                                   ; preds = %._crit_edge223.._crit_edge223_crit_edge, %868
; BB81 :
  %872 = phi i32 [ 0, %868 ], [ %881, %._crit_edge223.._crit_edge223_crit_edge ]
  %873 = zext i32 %872 to i64		; visa id: 1175
  %874 = shl nuw nsw i64 %873, 2		; visa id: 1176
  %875 = add i64 %133, %874		; visa id: 1177
  %876 = inttoptr i64 %875 to i32*		; visa id: 1178
  %877 = load i32, i32* %876, align 4, !noalias !625		; visa id: 1178
  %878 = add i64 %128, %874		; visa id: 1179
  %879 = inttoptr i64 %878 to i32*		; visa id: 1180
  store i32 %877, i32* %879, align 4, !alias.scope !625		; visa id: 1180
  %880 = icmp eq i32 %872, 0		; visa id: 1181
  br i1 %880, label %._crit_edge223.._crit_edge223_crit_edge, label %882, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1182

._crit_edge223.._crit_edge223_crit_edge:          ; preds = %._crit_edge223
; BB82 :
  %881 = add nuw nsw i32 %872, 1, !spirv.Decorations !631		; visa id: 1184
  br label %._crit_edge223, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1185

882:                                              ; preds = %._crit_edge223
; BB83 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1187
  %883 = load i64, i64* %129, align 8		; visa id: 1187
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1188
  %884 = bitcast i64 %883 to <2 x i32>		; visa id: 1188
  %885 = extractelement <2 x i32> %884, i32 0		; visa id: 1190
  %886 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %885, i32 1
  %887 = bitcast <2 x i32> %886 to i64		; visa id: 1190
  %888 = ashr exact i64 %887, 32		; visa id: 1191
  %889 = bitcast i64 %888 to <2 x i32>		; visa id: 1192
  %890 = extractelement <2 x i32> %889, i32 0		; visa id: 1196
  %891 = extractelement <2 x i32> %889, i32 1		; visa id: 1196
  %892 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %890, i32 %891, i32 %41, i32 %42)
  %893 = extractvalue { i32, i32 } %892, 0		; visa id: 1196
  %894 = extractvalue { i32, i32 } %892, 1		; visa id: 1196
  %895 = insertelement <2 x i32> undef, i32 %893, i32 0		; visa id: 1203
  %896 = insertelement <2 x i32> %895, i32 %894, i32 1		; visa id: 1204
  %897 = bitcast <2 x i32> %896 to i64		; visa id: 1205
  %898 = shl i64 %897, 1		; visa id: 1209
  %899 = add i64 %.in401, %898		; visa id: 1210
  %900 = ashr i64 %883, 31		; visa id: 1211
  %901 = bitcast i64 %900 to <2 x i32>		; visa id: 1212
  %902 = extractelement <2 x i32> %901, i32 0		; visa id: 1216
  %903 = extractelement <2 x i32> %901, i32 1		; visa id: 1216
  %904 = and i32 %902, -2		; visa id: 1216
  %905 = insertelement <2 x i32> undef, i32 %904, i32 0		; visa id: 1217
  %906 = insertelement <2 x i32> %905, i32 %903, i32 1		; visa id: 1218
  %907 = bitcast <2 x i32> %906 to i64		; visa id: 1219
  %908 = add i64 %899, %907		; visa id: 1223
  %909 = inttoptr i64 %908 to i16 addrspace(4)*		; visa id: 1224
  %910 = addrspacecast i16 addrspace(4)* %909 to i16 addrspace(1)*		; visa id: 1224
  %911 = load i16, i16 addrspace(1)* %910, align 2		; visa id: 1225
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1227
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1227
  %912 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1227
  %913 = insertelement <2 x i32> %912, i32 %864, i64 1		; visa id: 1228
  %914 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1229
  store <2 x i32> %913, <2 x i32>* %914, align 4, !noalias !635		; visa id: 1229
  br label %._crit_edge224, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1231

._crit_edge224:                                   ; preds = %._crit_edge224.._crit_edge224_crit_edge, %882
; BB84 :
  %915 = phi i32 [ 0, %882 ], [ %924, %._crit_edge224.._crit_edge224_crit_edge ]
  %916 = zext i32 %915 to i64		; visa id: 1232
  %917 = shl nuw nsw i64 %916, 2		; visa id: 1233
  %918 = add i64 %124, %917		; visa id: 1234
  %919 = inttoptr i64 %918 to i32*		; visa id: 1235
  %920 = load i32, i32* %919, align 4, !noalias !635		; visa id: 1235
  %921 = add i64 %119, %917		; visa id: 1236
  %922 = inttoptr i64 %921 to i32*		; visa id: 1237
  store i32 %920, i32* %922, align 4, !alias.scope !635		; visa id: 1237
  %923 = icmp eq i32 %915, 0		; visa id: 1238
  br i1 %923, label %._crit_edge224.._crit_edge224_crit_edge, label %925, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1239

._crit_edge224.._crit_edge224_crit_edge:          ; preds = %._crit_edge224
; BB85 :
  %924 = add nuw nsw i32 %915, 1, !spirv.Decorations !631		; visa id: 1241
  br label %._crit_edge224, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1242

925:                                              ; preds = %._crit_edge224
; BB86 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1244
  %926 = load i64, i64* %120, align 8		; visa id: 1244
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1245
  %927 = ashr i64 %926, 32		; visa id: 1245
  %928 = bitcast i64 %927 to <2 x i32>		; visa id: 1246
  %929 = extractelement <2 x i32> %928, i32 0		; visa id: 1250
  %930 = extractelement <2 x i32> %928, i32 1		; visa id: 1250
  %931 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %929, i32 %930, i32 %44, i32 %45)
  %932 = extractvalue { i32, i32 } %931, 0		; visa id: 1250
  %933 = extractvalue { i32, i32 } %931, 1		; visa id: 1250
  %934 = insertelement <2 x i32> undef, i32 %932, i32 0		; visa id: 1257
  %935 = insertelement <2 x i32> %934, i32 %933, i32 1		; visa id: 1258
  %936 = bitcast <2 x i32> %935 to i64		; visa id: 1259
  %937 = bitcast i64 %926 to <2 x i32>		; visa id: 1263
  %938 = extractelement <2 x i32> %937, i32 0		; visa id: 1265
  %939 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %938, i32 1
  %940 = bitcast <2 x i32> %939 to i64		; visa id: 1265
  %941 = shl i64 %936, 1		; visa id: 1266
  %942 = add i64 %.in400, %941		; visa id: 1267
  %943 = ashr exact i64 %940, 31		; visa id: 1268
  %944 = add i64 %942, %943		; visa id: 1269
  %945 = inttoptr i64 %944 to i16 addrspace(4)*		; visa id: 1270
  %946 = addrspacecast i16 addrspace(4)* %945 to i16 addrspace(1)*		; visa id: 1270
  %947 = load i16, i16 addrspace(1)* %946, align 2		; visa id: 1271
  %948 = zext i16 %911 to i32		; visa id: 1273
  %949 = shl nuw i32 %948, 16, !spirv.Decorations !639		; visa id: 1274
  %950 = bitcast i32 %949 to float
  %951 = zext i16 %947 to i32		; visa id: 1275
  %952 = shl nuw i32 %951, 16, !spirv.Decorations !639		; visa id: 1276
  %953 = bitcast i32 %952 to float
  %954 = fmul reassoc nsz arcp contract float %950, %953, !spirv.Decorations !618
  %955 = fadd reassoc nsz arcp contract float %954, %.sroa.10.1, !spirv.Decorations !618		; visa id: 1277
  br label %._crit_edge.274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1278

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %925
; BB87 :
  %.sroa.10.2 = phi float [ %955, %925 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  %956 = icmp slt i32 %223, %const_reg_dword
  %957 = icmp slt i32 %864, %const_reg_dword1		; visa id: 1279
  %958 = and i1 %956, %957		; visa id: 1280
  br i1 %958, label %959, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1282

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
; BB:
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

959:                                              ; preds = %._crit_edge.274
; BB89 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1284
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1284
  %960 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 1284
  %961 = insertelement <2 x i32> %960, i32 %113, i64 1		; visa id: 1285
  %962 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1286
  store <2 x i32> %961, <2 x i32>* %962, align 4, !noalias !625		; visa id: 1286
  br label %._crit_edge225, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1288

._crit_edge225:                                   ; preds = %._crit_edge225.._crit_edge225_crit_edge, %959
; BB90 :
  %963 = phi i32 [ 0, %959 ], [ %972, %._crit_edge225.._crit_edge225_crit_edge ]
  %964 = zext i32 %963 to i64		; visa id: 1289
  %965 = shl nuw nsw i64 %964, 2		; visa id: 1290
  %966 = add i64 %133, %965		; visa id: 1291
  %967 = inttoptr i64 %966 to i32*		; visa id: 1292
  %968 = load i32, i32* %967, align 4, !noalias !625		; visa id: 1292
  %969 = add i64 %128, %965		; visa id: 1293
  %970 = inttoptr i64 %969 to i32*		; visa id: 1294
  store i32 %968, i32* %970, align 4, !alias.scope !625		; visa id: 1294
  %971 = icmp eq i32 %963, 0		; visa id: 1295
  br i1 %971, label %._crit_edge225.._crit_edge225_crit_edge, label %973, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1296

._crit_edge225.._crit_edge225_crit_edge:          ; preds = %._crit_edge225
; BB91 :
  %972 = add nuw nsw i32 %963, 1, !spirv.Decorations !631		; visa id: 1298
  br label %._crit_edge225, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1299

973:                                              ; preds = %._crit_edge225
; BB92 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1301
  %974 = load i64, i64* %129, align 8		; visa id: 1301
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1302
  %975 = bitcast i64 %974 to <2 x i32>		; visa id: 1302
  %976 = extractelement <2 x i32> %975, i32 0		; visa id: 1304
  %977 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %976, i32 1
  %978 = bitcast <2 x i32> %977 to i64		; visa id: 1304
  %979 = ashr exact i64 %978, 32		; visa id: 1305
  %980 = bitcast i64 %979 to <2 x i32>		; visa id: 1306
  %981 = extractelement <2 x i32> %980, i32 0		; visa id: 1310
  %982 = extractelement <2 x i32> %980, i32 1		; visa id: 1310
  %983 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %981, i32 %982, i32 %41, i32 %42)
  %984 = extractvalue { i32, i32 } %983, 0		; visa id: 1310
  %985 = extractvalue { i32, i32 } %983, 1		; visa id: 1310
  %986 = insertelement <2 x i32> undef, i32 %984, i32 0		; visa id: 1317
  %987 = insertelement <2 x i32> %986, i32 %985, i32 1		; visa id: 1318
  %988 = bitcast <2 x i32> %987 to i64		; visa id: 1319
  %989 = shl i64 %988, 1		; visa id: 1323
  %990 = add i64 %.in401, %989		; visa id: 1324
  %991 = ashr i64 %974, 31		; visa id: 1325
  %992 = bitcast i64 %991 to <2 x i32>		; visa id: 1326
  %993 = extractelement <2 x i32> %992, i32 0		; visa id: 1330
  %994 = extractelement <2 x i32> %992, i32 1		; visa id: 1330
  %995 = and i32 %993, -2		; visa id: 1330
  %996 = insertelement <2 x i32> undef, i32 %995, i32 0		; visa id: 1331
  %997 = insertelement <2 x i32> %996, i32 %994, i32 1		; visa id: 1332
  %998 = bitcast <2 x i32> %997 to i64		; visa id: 1333
  %999 = add i64 %990, %998		; visa id: 1337
  %1000 = inttoptr i64 %999 to i16 addrspace(4)*		; visa id: 1338
  %1001 = addrspacecast i16 addrspace(4)* %1000 to i16 addrspace(1)*		; visa id: 1338
  %1002 = load i16, i16 addrspace(1)* %1001, align 2		; visa id: 1339
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1341
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1341
  %1003 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1341
  %1004 = insertelement <2 x i32> %1003, i32 %864, i64 1		; visa id: 1342
  %1005 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1343
  store <2 x i32> %1004, <2 x i32>* %1005, align 4, !noalias !635		; visa id: 1343
  br label %._crit_edge226, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1345

._crit_edge226:                                   ; preds = %._crit_edge226.._crit_edge226_crit_edge, %973
; BB93 :
  %1006 = phi i32 [ 0, %973 ], [ %1015, %._crit_edge226.._crit_edge226_crit_edge ]
  %1007 = zext i32 %1006 to i64		; visa id: 1346
  %1008 = shl nuw nsw i64 %1007, 2		; visa id: 1347
  %1009 = add i64 %124, %1008		; visa id: 1348
  %1010 = inttoptr i64 %1009 to i32*		; visa id: 1349
  %1011 = load i32, i32* %1010, align 4, !noalias !635		; visa id: 1349
  %1012 = add i64 %119, %1008		; visa id: 1350
  %1013 = inttoptr i64 %1012 to i32*		; visa id: 1351
  store i32 %1011, i32* %1013, align 4, !alias.scope !635		; visa id: 1351
  %1014 = icmp eq i32 %1006, 0		; visa id: 1352
  br i1 %1014, label %._crit_edge226.._crit_edge226_crit_edge, label %1016, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1353

._crit_edge226.._crit_edge226_crit_edge:          ; preds = %._crit_edge226
; BB94 :
  %1015 = add nuw nsw i32 %1006, 1, !spirv.Decorations !631		; visa id: 1355
  br label %._crit_edge226, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1356

1016:                                             ; preds = %._crit_edge226
; BB95 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1358
  %1017 = load i64, i64* %120, align 8		; visa id: 1358
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1359
  %1018 = ashr i64 %1017, 32		; visa id: 1359
  %1019 = bitcast i64 %1018 to <2 x i32>		; visa id: 1360
  %1020 = extractelement <2 x i32> %1019, i32 0		; visa id: 1364
  %1021 = extractelement <2 x i32> %1019, i32 1		; visa id: 1364
  %1022 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1020, i32 %1021, i32 %44, i32 %45)
  %1023 = extractvalue { i32, i32 } %1022, 0		; visa id: 1364
  %1024 = extractvalue { i32, i32 } %1022, 1		; visa id: 1364
  %1025 = insertelement <2 x i32> undef, i32 %1023, i32 0		; visa id: 1371
  %1026 = insertelement <2 x i32> %1025, i32 %1024, i32 1		; visa id: 1372
  %1027 = bitcast <2 x i32> %1026 to i64		; visa id: 1373
  %1028 = bitcast i64 %1017 to <2 x i32>		; visa id: 1377
  %1029 = extractelement <2 x i32> %1028, i32 0		; visa id: 1379
  %1030 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1029, i32 1
  %1031 = bitcast <2 x i32> %1030 to i64		; visa id: 1379
  %1032 = shl i64 %1027, 1		; visa id: 1380
  %1033 = add i64 %.in400, %1032		; visa id: 1381
  %1034 = ashr exact i64 %1031, 31		; visa id: 1382
  %1035 = add i64 %1033, %1034		; visa id: 1383
  %1036 = inttoptr i64 %1035 to i16 addrspace(4)*		; visa id: 1384
  %1037 = addrspacecast i16 addrspace(4)* %1036 to i16 addrspace(1)*		; visa id: 1384
  %1038 = load i16, i16 addrspace(1)* %1037, align 2		; visa id: 1385
  %1039 = zext i16 %1002 to i32		; visa id: 1387
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !639		; visa id: 1388
  %1041 = bitcast i32 %1040 to float
  %1042 = zext i16 %1038 to i32		; visa id: 1389
  %1043 = shl nuw i32 %1042, 16, !spirv.Decorations !639		; visa id: 1390
  %1044 = bitcast i32 %1043 to float
  %1045 = fmul reassoc nsz arcp contract float %1041, %1044, !spirv.Decorations !618
  %1046 = fadd reassoc nsz arcp contract float %1045, %.sroa.74.1, !spirv.Decorations !618		; visa id: 1391
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1392

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %1016
; BB96 :
  %.sroa.74.2 = phi float [ %1046, %1016 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  %1047 = icmp slt i32 %315, %const_reg_dword
  %1048 = icmp slt i32 %864, %const_reg_dword1		; visa id: 1393
  %1049 = and i1 %1047, %1048		; visa id: 1394
  br i1 %1049, label %1050, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1396

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
; BB:
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1050:                                             ; preds = %._crit_edge.1.2
; BB98 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1398
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1398
  %1051 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 1398
  %1052 = insertelement <2 x i32> %1051, i32 %113, i64 1		; visa id: 1399
  %1053 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1400
  store <2 x i32> %1052, <2 x i32>* %1053, align 4, !noalias !625		; visa id: 1400
  br label %._crit_edge227, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1402

._crit_edge227:                                   ; preds = %._crit_edge227.._crit_edge227_crit_edge, %1050
; BB99 :
  %1054 = phi i32 [ 0, %1050 ], [ %1063, %._crit_edge227.._crit_edge227_crit_edge ]
  %1055 = zext i32 %1054 to i64		; visa id: 1403
  %1056 = shl nuw nsw i64 %1055, 2		; visa id: 1404
  %1057 = add i64 %133, %1056		; visa id: 1405
  %1058 = inttoptr i64 %1057 to i32*		; visa id: 1406
  %1059 = load i32, i32* %1058, align 4, !noalias !625		; visa id: 1406
  %1060 = add i64 %128, %1056		; visa id: 1407
  %1061 = inttoptr i64 %1060 to i32*		; visa id: 1408
  store i32 %1059, i32* %1061, align 4, !alias.scope !625		; visa id: 1408
  %1062 = icmp eq i32 %1054, 0		; visa id: 1409
  br i1 %1062, label %._crit_edge227.._crit_edge227_crit_edge, label %1064, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1410

._crit_edge227.._crit_edge227_crit_edge:          ; preds = %._crit_edge227
; BB100 :
  %1063 = add nuw nsw i32 %1054, 1, !spirv.Decorations !631		; visa id: 1412
  br label %._crit_edge227, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1413

1064:                                             ; preds = %._crit_edge227
; BB101 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1415
  %1065 = load i64, i64* %129, align 8		; visa id: 1415
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1416
  %1066 = bitcast i64 %1065 to <2 x i32>		; visa id: 1416
  %1067 = extractelement <2 x i32> %1066, i32 0		; visa id: 1418
  %1068 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1067, i32 1
  %1069 = bitcast <2 x i32> %1068 to i64		; visa id: 1418
  %1070 = ashr exact i64 %1069, 32		; visa id: 1419
  %1071 = bitcast i64 %1070 to <2 x i32>		; visa id: 1420
  %1072 = extractelement <2 x i32> %1071, i32 0		; visa id: 1424
  %1073 = extractelement <2 x i32> %1071, i32 1		; visa id: 1424
  %1074 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1072, i32 %1073, i32 %41, i32 %42)
  %1075 = extractvalue { i32, i32 } %1074, 0		; visa id: 1424
  %1076 = extractvalue { i32, i32 } %1074, 1		; visa id: 1424
  %1077 = insertelement <2 x i32> undef, i32 %1075, i32 0		; visa id: 1431
  %1078 = insertelement <2 x i32> %1077, i32 %1076, i32 1		; visa id: 1432
  %1079 = bitcast <2 x i32> %1078 to i64		; visa id: 1433
  %1080 = shl i64 %1079, 1		; visa id: 1437
  %1081 = add i64 %.in401, %1080		; visa id: 1438
  %1082 = ashr i64 %1065, 31		; visa id: 1439
  %1083 = bitcast i64 %1082 to <2 x i32>		; visa id: 1440
  %1084 = extractelement <2 x i32> %1083, i32 0		; visa id: 1444
  %1085 = extractelement <2 x i32> %1083, i32 1		; visa id: 1444
  %1086 = and i32 %1084, -2		; visa id: 1444
  %1087 = insertelement <2 x i32> undef, i32 %1086, i32 0		; visa id: 1445
  %1088 = insertelement <2 x i32> %1087, i32 %1085, i32 1		; visa id: 1446
  %1089 = bitcast <2 x i32> %1088 to i64		; visa id: 1447
  %1090 = add i64 %1081, %1089		; visa id: 1451
  %1091 = inttoptr i64 %1090 to i16 addrspace(4)*		; visa id: 1452
  %1092 = addrspacecast i16 addrspace(4)* %1091 to i16 addrspace(1)*		; visa id: 1452
  %1093 = load i16, i16 addrspace(1)* %1092, align 2		; visa id: 1453
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1455
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1455
  %1094 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1455
  %1095 = insertelement <2 x i32> %1094, i32 %864, i64 1		; visa id: 1456
  %1096 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1457
  store <2 x i32> %1095, <2 x i32>* %1096, align 4, !noalias !635		; visa id: 1457
  br label %._crit_edge228, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1459

._crit_edge228:                                   ; preds = %._crit_edge228.._crit_edge228_crit_edge, %1064
; BB102 :
  %1097 = phi i32 [ 0, %1064 ], [ %1106, %._crit_edge228.._crit_edge228_crit_edge ]
  %1098 = zext i32 %1097 to i64		; visa id: 1460
  %1099 = shl nuw nsw i64 %1098, 2		; visa id: 1461
  %1100 = add i64 %124, %1099		; visa id: 1462
  %1101 = inttoptr i64 %1100 to i32*		; visa id: 1463
  %1102 = load i32, i32* %1101, align 4, !noalias !635		; visa id: 1463
  %1103 = add i64 %119, %1099		; visa id: 1464
  %1104 = inttoptr i64 %1103 to i32*		; visa id: 1465
  store i32 %1102, i32* %1104, align 4, !alias.scope !635		; visa id: 1465
  %1105 = icmp eq i32 %1097, 0		; visa id: 1466
  br i1 %1105, label %._crit_edge228.._crit_edge228_crit_edge, label %1107, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1467

._crit_edge228.._crit_edge228_crit_edge:          ; preds = %._crit_edge228
; BB103 :
  %1106 = add nuw nsw i32 %1097, 1, !spirv.Decorations !631		; visa id: 1469
  br label %._crit_edge228, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1470

1107:                                             ; preds = %._crit_edge228
; BB104 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1472
  %1108 = load i64, i64* %120, align 8		; visa id: 1472
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1473
  %1109 = ashr i64 %1108, 32		; visa id: 1473
  %1110 = bitcast i64 %1109 to <2 x i32>		; visa id: 1474
  %1111 = extractelement <2 x i32> %1110, i32 0		; visa id: 1478
  %1112 = extractelement <2 x i32> %1110, i32 1		; visa id: 1478
  %1113 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1111, i32 %1112, i32 %44, i32 %45)
  %1114 = extractvalue { i32, i32 } %1113, 0		; visa id: 1478
  %1115 = extractvalue { i32, i32 } %1113, 1		; visa id: 1478
  %1116 = insertelement <2 x i32> undef, i32 %1114, i32 0		; visa id: 1485
  %1117 = insertelement <2 x i32> %1116, i32 %1115, i32 1		; visa id: 1486
  %1118 = bitcast <2 x i32> %1117 to i64		; visa id: 1487
  %1119 = bitcast i64 %1108 to <2 x i32>		; visa id: 1491
  %1120 = extractelement <2 x i32> %1119, i32 0		; visa id: 1493
  %1121 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1120, i32 1
  %1122 = bitcast <2 x i32> %1121 to i64		; visa id: 1493
  %1123 = shl i64 %1118, 1		; visa id: 1494
  %1124 = add i64 %.in400, %1123		; visa id: 1495
  %1125 = ashr exact i64 %1122, 31		; visa id: 1496
  %1126 = add i64 %1124, %1125		; visa id: 1497
  %1127 = inttoptr i64 %1126 to i16 addrspace(4)*		; visa id: 1498
  %1128 = addrspacecast i16 addrspace(4)* %1127 to i16 addrspace(1)*		; visa id: 1498
  %1129 = load i16, i16 addrspace(1)* %1128, align 2		; visa id: 1499
  %1130 = zext i16 %1093 to i32		; visa id: 1501
  %1131 = shl nuw i32 %1130, 16, !spirv.Decorations !639		; visa id: 1502
  %1132 = bitcast i32 %1131 to float
  %1133 = zext i16 %1129 to i32		; visa id: 1503
  %1134 = shl nuw i32 %1133, 16, !spirv.Decorations !639		; visa id: 1504
  %1135 = bitcast i32 %1134 to float
  %1136 = fmul reassoc nsz arcp contract float %1132, %1135, !spirv.Decorations !618
  %1137 = fadd reassoc nsz arcp contract float %1136, %.sroa.138.1, !spirv.Decorations !618		; visa id: 1505
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1506

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %1107
; BB105 :
  %.sroa.138.2 = phi float [ %1137, %1107 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  %1138 = icmp slt i32 %407, %const_reg_dword
  %1139 = icmp slt i32 %864, %const_reg_dword1		; visa id: 1507
  %1140 = and i1 %1138, %1139		; visa id: 1508
  br i1 %1140, label %1141, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1510

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
; BB:
  br label %.preheader.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1141:                                             ; preds = %._crit_edge.2.2
; BB107 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1512
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1512
  %1142 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 1512
  %1143 = insertelement <2 x i32> %1142, i32 %113, i64 1		; visa id: 1513
  %1144 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1514
  store <2 x i32> %1143, <2 x i32>* %1144, align 4, !noalias !625		; visa id: 1514
  br label %._crit_edge229, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1516

._crit_edge229:                                   ; preds = %._crit_edge229.._crit_edge229_crit_edge, %1141
; BB108 :
  %1145 = phi i32 [ 0, %1141 ], [ %1154, %._crit_edge229.._crit_edge229_crit_edge ]
  %1146 = zext i32 %1145 to i64		; visa id: 1517
  %1147 = shl nuw nsw i64 %1146, 2		; visa id: 1518
  %1148 = add i64 %133, %1147		; visa id: 1519
  %1149 = inttoptr i64 %1148 to i32*		; visa id: 1520
  %1150 = load i32, i32* %1149, align 4, !noalias !625		; visa id: 1520
  %1151 = add i64 %128, %1147		; visa id: 1521
  %1152 = inttoptr i64 %1151 to i32*		; visa id: 1522
  store i32 %1150, i32* %1152, align 4, !alias.scope !625		; visa id: 1522
  %1153 = icmp eq i32 %1145, 0		; visa id: 1523
  br i1 %1153, label %._crit_edge229.._crit_edge229_crit_edge, label %1155, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1524

._crit_edge229.._crit_edge229_crit_edge:          ; preds = %._crit_edge229
; BB109 :
  %1154 = add nuw nsw i32 %1145, 1, !spirv.Decorations !631		; visa id: 1526
  br label %._crit_edge229, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1527

1155:                                             ; preds = %._crit_edge229
; BB110 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1529
  %1156 = load i64, i64* %129, align 8		; visa id: 1529
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1530
  %1157 = bitcast i64 %1156 to <2 x i32>		; visa id: 1530
  %1158 = extractelement <2 x i32> %1157, i32 0		; visa id: 1532
  %1159 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1158, i32 1
  %1160 = bitcast <2 x i32> %1159 to i64		; visa id: 1532
  %1161 = ashr exact i64 %1160, 32		; visa id: 1533
  %1162 = bitcast i64 %1161 to <2 x i32>		; visa id: 1534
  %1163 = extractelement <2 x i32> %1162, i32 0		; visa id: 1538
  %1164 = extractelement <2 x i32> %1162, i32 1		; visa id: 1538
  %1165 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1163, i32 %1164, i32 %41, i32 %42)
  %1166 = extractvalue { i32, i32 } %1165, 0		; visa id: 1538
  %1167 = extractvalue { i32, i32 } %1165, 1		; visa id: 1538
  %1168 = insertelement <2 x i32> undef, i32 %1166, i32 0		; visa id: 1545
  %1169 = insertelement <2 x i32> %1168, i32 %1167, i32 1		; visa id: 1546
  %1170 = bitcast <2 x i32> %1169 to i64		; visa id: 1547
  %1171 = shl i64 %1170, 1		; visa id: 1551
  %1172 = add i64 %.in401, %1171		; visa id: 1552
  %1173 = ashr i64 %1156, 31		; visa id: 1553
  %1174 = bitcast i64 %1173 to <2 x i32>		; visa id: 1554
  %1175 = extractelement <2 x i32> %1174, i32 0		; visa id: 1558
  %1176 = extractelement <2 x i32> %1174, i32 1		; visa id: 1558
  %1177 = and i32 %1175, -2		; visa id: 1558
  %1178 = insertelement <2 x i32> undef, i32 %1177, i32 0		; visa id: 1559
  %1179 = insertelement <2 x i32> %1178, i32 %1176, i32 1		; visa id: 1560
  %1180 = bitcast <2 x i32> %1179 to i64		; visa id: 1561
  %1181 = add i64 %1172, %1180		; visa id: 1565
  %1182 = inttoptr i64 %1181 to i16 addrspace(4)*		; visa id: 1566
  %1183 = addrspacecast i16 addrspace(4)* %1182 to i16 addrspace(1)*		; visa id: 1566
  %1184 = load i16, i16 addrspace(1)* %1183, align 2		; visa id: 1567
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1569
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1569
  %1185 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1569
  %1186 = insertelement <2 x i32> %1185, i32 %864, i64 1		; visa id: 1570
  %1187 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1571
  store <2 x i32> %1186, <2 x i32>* %1187, align 4, !noalias !635		; visa id: 1571
  br label %._crit_edge230, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1573

._crit_edge230:                                   ; preds = %._crit_edge230.._crit_edge230_crit_edge, %1155
; BB111 :
  %1188 = phi i32 [ 0, %1155 ], [ %1197, %._crit_edge230.._crit_edge230_crit_edge ]
  %1189 = zext i32 %1188 to i64		; visa id: 1574
  %1190 = shl nuw nsw i64 %1189, 2		; visa id: 1575
  %1191 = add i64 %124, %1190		; visa id: 1576
  %1192 = inttoptr i64 %1191 to i32*		; visa id: 1577
  %1193 = load i32, i32* %1192, align 4, !noalias !635		; visa id: 1577
  %1194 = add i64 %119, %1190		; visa id: 1578
  %1195 = inttoptr i64 %1194 to i32*		; visa id: 1579
  store i32 %1193, i32* %1195, align 4, !alias.scope !635		; visa id: 1579
  %1196 = icmp eq i32 %1188, 0		; visa id: 1580
  br i1 %1196, label %._crit_edge230.._crit_edge230_crit_edge, label %1198, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1581

._crit_edge230.._crit_edge230_crit_edge:          ; preds = %._crit_edge230
; BB112 :
  %1197 = add nuw nsw i32 %1188, 1, !spirv.Decorations !631		; visa id: 1583
  br label %._crit_edge230, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1584

1198:                                             ; preds = %._crit_edge230
; BB113 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1586
  %1199 = load i64, i64* %120, align 8		; visa id: 1586
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1587
  %1200 = ashr i64 %1199, 32		; visa id: 1587
  %1201 = bitcast i64 %1200 to <2 x i32>		; visa id: 1588
  %1202 = extractelement <2 x i32> %1201, i32 0		; visa id: 1592
  %1203 = extractelement <2 x i32> %1201, i32 1		; visa id: 1592
  %1204 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1202, i32 %1203, i32 %44, i32 %45)
  %1205 = extractvalue { i32, i32 } %1204, 0		; visa id: 1592
  %1206 = extractvalue { i32, i32 } %1204, 1		; visa id: 1592
  %1207 = insertelement <2 x i32> undef, i32 %1205, i32 0		; visa id: 1599
  %1208 = insertelement <2 x i32> %1207, i32 %1206, i32 1		; visa id: 1600
  %1209 = bitcast <2 x i32> %1208 to i64		; visa id: 1601
  %1210 = bitcast i64 %1199 to <2 x i32>		; visa id: 1605
  %1211 = extractelement <2 x i32> %1210, i32 0		; visa id: 1607
  %1212 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1211, i32 1
  %1213 = bitcast <2 x i32> %1212 to i64		; visa id: 1607
  %1214 = shl i64 %1209, 1		; visa id: 1608
  %1215 = add i64 %.in400, %1214		; visa id: 1609
  %1216 = ashr exact i64 %1213, 31		; visa id: 1610
  %1217 = add i64 %1215, %1216		; visa id: 1611
  %1218 = inttoptr i64 %1217 to i16 addrspace(4)*		; visa id: 1612
  %1219 = addrspacecast i16 addrspace(4)* %1218 to i16 addrspace(1)*		; visa id: 1612
  %1220 = load i16, i16 addrspace(1)* %1219, align 2		; visa id: 1613
  %1221 = zext i16 %1184 to i32		; visa id: 1615
  %1222 = shl nuw i32 %1221, 16, !spirv.Decorations !639		; visa id: 1616
  %1223 = bitcast i32 %1222 to float
  %1224 = zext i16 %1220 to i32		; visa id: 1617
  %1225 = shl nuw i32 %1224, 16, !spirv.Decorations !639		; visa id: 1618
  %1226 = bitcast i32 %1225 to float
  %1227 = fmul reassoc nsz arcp contract float %1223, %1226, !spirv.Decorations !618
  %1228 = fadd reassoc nsz arcp contract float %1227, %.sroa.202.1, !spirv.Decorations !618		; visa id: 1619
  br label %.preheader.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1620

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %1198
; BB114 :
  %.sroa.202.2 = phi float [ %1228, %1198 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %1229 = add i32 %69, 3		; visa id: 1621
  %1230 = icmp slt i32 %1229, %const_reg_dword1		; visa id: 1622
  %1231 = icmp slt i32 %65, %const_reg_dword
  %1232 = and i1 %1231, %1230		; visa id: 1623
  br i1 %1232, label %1233, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1625

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
; BB:
  br label %._crit_edge.375, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1233:                                             ; preds = %.preheader.2
; BB116 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1627
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1627
  %1234 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 1627
  %1235 = insertelement <2 x i32> %1234, i32 %113, i64 1		; visa id: 1628
  %1236 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1629
  store <2 x i32> %1235, <2 x i32>* %1236, align 4, !noalias !625		; visa id: 1629
  br label %._crit_edge231, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1631

._crit_edge231:                                   ; preds = %._crit_edge231.._crit_edge231_crit_edge, %1233
; BB117 :
  %1237 = phi i32 [ 0, %1233 ], [ %1246, %._crit_edge231.._crit_edge231_crit_edge ]
  %1238 = zext i32 %1237 to i64		; visa id: 1632
  %1239 = shl nuw nsw i64 %1238, 2		; visa id: 1633
  %1240 = add i64 %133, %1239		; visa id: 1634
  %1241 = inttoptr i64 %1240 to i32*		; visa id: 1635
  %1242 = load i32, i32* %1241, align 4, !noalias !625		; visa id: 1635
  %1243 = add i64 %128, %1239		; visa id: 1636
  %1244 = inttoptr i64 %1243 to i32*		; visa id: 1637
  store i32 %1242, i32* %1244, align 4, !alias.scope !625		; visa id: 1637
  %1245 = icmp eq i32 %1237, 0		; visa id: 1638
  br i1 %1245, label %._crit_edge231.._crit_edge231_crit_edge, label %1247, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1639

._crit_edge231.._crit_edge231_crit_edge:          ; preds = %._crit_edge231
; BB118 :
  %1246 = add nuw nsw i32 %1237, 1, !spirv.Decorations !631		; visa id: 1641
  br label %._crit_edge231, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1642

1247:                                             ; preds = %._crit_edge231
; BB119 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1644
  %1248 = load i64, i64* %129, align 8		; visa id: 1644
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1645
  %1249 = bitcast i64 %1248 to <2 x i32>		; visa id: 1645
  %1250 = extractelement <2 x i32> %1249, i32 0		; visa id: 1647
  %1251 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1250, i32 1
  %1252 = bitcast <2 x i32> %1251 to i64		; visa id: 1647
  %1253 = ashr exact i64 %1252, 32		; visa id: 1648
  %1254 = bitcast i64 %1253 to <2 x i32>		; visa id: 1649
  %1255 = extractelement <2 x i32> %1254, i32 0		; visa id: 1653
  %1256 = extractelement <2 x i32> %1254, i32 1		; visa id: 1653
  %1257 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1255, i32 %1256, i32 %41, i32 %42)
  %1258 = extractvalue { i32, i32 } %1257, 0		; visa id: 1653
  %1259 = extractvalue { i32, i32 } %1257, 1		; visa id: 1653
  %1260 = insertelement <2 x i32> undef, i32 %1258, i32 0		; visa id: 1660
  %1261 = insertelement <2 x i32> %1260, i32 %1259, i32 1		; visa id: 1661
  %1262 = bitcast <2 x i32> %1261 to i64		; visa id: 1662
  %1263 = shl i64 %1262, 1		; visa id: 1666
  %1264 = add i64 %.in401, %1263		; visa id: 1667
  %1265 = ashr i64 %1248, 31		; visa id: 1668
  %1266 = bitcast i64 %1265 to <2 x i32>		; visa id: 1669
  %1267 = extractelement <2 x i32> %1266, i32 0		; visa id: 1673
  %1268 = extractelement <2 x i32> %1266, i32 1		; visa id: 1673
  %1269 = and i32 %1267, -2		; visa id: 1673
  %1270 = insertelement <2 x i32> undef, i32 %1269, i32 0		; visa id: 1674
  %1271 = insertelement <2 x i32> %1270, i32 %1268, i32 1		; visa id: 1675
  %1272 = bitcast <2 x i32> %1271 to i64		; visa id: 1676
  %1273 = add i64 %1264, %1272		; visa id: 1680
  %1274 = inttoptr i64 %1273 to i16 addrspace(4)*		; visa id: 1681
  %1275 = addrspacecast i16 addrspace(4)* %1274 to i16 addrspace(1)*		; visa id: 1681
  %1276 = load i16, i16 addrspace(1)* %1275, align 2		; visa id: 1682
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1684
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1684
  %1277 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1684
  %1278 = insertelement <2 x i32> %1277, i32 %1229, i64 1		; visa id: 1685
  %1279 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1686
  store <2 x i32> %1278, <2 x i32>* %1279, align 4, !noalias !635		; visa id: 1686
  br label %._crit_edge232, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1688

._crit_edge232:                                   ; preds = %._crit_edge232.._crit_edge232_crit_edge, %1247
; BB120 :
  %1280 = phi i32 [ 0, %1247 ], [ %1289, %._crit_edge232.._crit_edge232_crit_edge ]
  %1281 = zext i32 %1280 to i64		; visa id: 1689
  %1282 = shl nuw nsw i64 %1281, 2		; visa id: 1690
  %1283 = add i64 %124, %1282		; visa id: 1691
  %1284 = inttoptr i64 %1283 to i32*		; visa id: 1692
  %1285 = load i32, i32* %1284, align 4, !noalias !635		; visa id: 1692
  %1286 = add i64 %119, %1282		; visa id: 1693
  %1287 = inttoptr i64 %1286 to i32*		; visa id: 1694
  store i32 %1285, i32* %1287, align 4, !alias.scope !635		; visa id: 1694
  %1288 = icmp eq i32 %1280, 0		; visa id: 1695
  br i1 %1288, label %._crit_edge232.._crit_edge232_crit_edge, label %1290, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1696

._crit_edge232.._crit_edge232_crit_edge:          ; preds = %._crit_edge232
; BB121 :
  %1289 = add nuw nsw i32 %1280, 1, !spirv.Decorations !631		; visa id: 1698
  br label %._crit_edge232, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1699

1290:                                             ; preds = %._crit_edge232
; BB122 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1701
  %1291 = load i64, i64* %120, align 8		; visa id: 1701
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1702
  %1292 = ashr i64 %1291, 32		; visa id: 1702
  %1293 = bitcast i64 %1292 to <2 x i32>		; visa id: 1703
  %1294 = extractelement <2 x i32> %1293, i32 0		; visa id: 1707
  %1295 = extractelement <2 x i32> %1293, i32 1		; visa id: 1707
  %1296 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1294, i32 %1295, i32 %44, i32 %45)
  %1297 = extractvalue { i32, i32 } %1296, 0		; visa id: 1707
  %1298 = extractvalue { i32, i32 } %1296, 1		; visa id: 1707
  %1299 = insertelement <2 x i32> undef, i32 %1297, i32 0		; visa id: 1714
  %1300 = insertelement <2 x i32> %1299, i32 %1298, i32 1		; visa id: 1715
  %1301 = bitcast <2 x i32> %1300 to i64		; visa id: 1716
  %1302 = bitcast i64 %1291 to <2 x i32>		; visa id: 1720
  %1303 = extractelement <2 x i32> %1302, i32 0		; visa id: 1722
  %1304 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1303, i32 1
  %1305 = bitcast <2 x i32> %1304 to i64		; visa id: 1722
  %1306 = shl i64 %1301, 1		; visa id: 1723
  %1307 = add i64 %.in400, %1306		; visa id: 1724
  %1308 = ashr exact i64 %1305, 31		; visa id: 1725
  %1309 = add i64 %1307, %1308		; visa id: 1726
  %1310 = inttoptr i64 %1309 to i16 addrspace(4)*		; visa id: 1727
  %1311 = addrspacecast i16 addrspace(4)* %1310 to i16 addrspace(1)*		; visa id: 1727
  %1312 = load i16, i16 addrspace(1)* %1311, align 2		; visa id: 1728
  %1313 = zext i16 %1276 to i32		; visa id: 1730
  %1314 = shl nuw i32 %1313, 16, !spirv.Decorations !639		; visa id: 1731
  %1315 = bitcast i32 %1314 to float
  %1316 = zext i16 %1312 to i32		; visa id: 1732
  %1317 = shl nuw i32 %1316, 16, !spirv.Decorations !639		; visa id: 1733
  %1318 = bitcast i32 %1317 to float
  %1319 = fmul reassoc nsz arcp contract float %1315, %1318, !spirv.Decorations !618
  %1320 = fadd reassoc nsz arcp contract float %1319, %.sroa.14.1, !spirv.Decorations !618		; visa id: 1734
  br label %._crit_edge.375, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1735

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %1290
; BB123 :
  %.sroa.14.2 = phi float [ %1320, %1290 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  %1321 = icmp slt i32 %223, %const_reg_dword
  %1322 = icmp slt i32 %1229, %const_reg_dword1		; visa id: 1736
  %1323 = and i1 %1321, %1322		; visa id: 1737
  br i1 %1323, label %1324, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1739

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
; BB:
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1324:                                             ; preds = %._crit_edge.375
; BB125 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1741
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1741
  %1325 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 1741
  %1326 = insertelement <2 x i32> %1325, i32 %113, i64 1		; visa id: 1742
  %1327 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1743
  store <2 x i32> %1326, <2 x i32>* %1327, align 4, !noalias !625		; visa id: 1743
  br label %._crit_edge233, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1745

._crit_edge233:                                   ; preds = %._crit_edge233.._crit_edge233_crit_edge, %1324
; BB126 :
  %1328 = phi i32 [ 0, %1324 ], [ %1337, %._crit_edge233.._crit_edge233_crit_edge ]
  %1329 = zext i32 %1328 to i64		; visa id: 1746
  %1330 = shl nuw nsw i64 %1329, 2		; visa id: 1747
  %1331 = add i64 %133, %1330		; visa id: 1748
  %1332 = inttoptr i64 %1331 to i32*		; visa id: 1749
  %1333 = load i32, i32* %1332, align 4, !noalias !625		; visa id: 1749
  %1334 = add i64 %128, %1330		; visa id: 1750
  %1335 = inttoptr i64 %1334 to i32*		; visa id: 1751
  store i32 %1333, i32* %1335, align 4, !alias.scope !625		; visa id: 1751
  %1336 = icmp eq i32 %1328, 0		; visa id: 1752
  br i1 %1336, label %._crit_edge233.._crit_edge233_crit_edge, label %1338, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1753

._crit_edge233.._crit_edge233_crit_edge:          ; preds = %._crit_edge233
; BB127 :
  %1337 = add nuw nsw i32 %1328, 1, !spirv.Decorations !631		; visa id: 1755
  br label %._crit_edge233, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1756

1338:                                             ; preds = %._crit_edge233
; BB128 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1758
  %1339 = load i64, i64* %129, align 8		; visa id: 1758
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1759
  %1340 = bitcast i64 %1339 to <2 x i32>		; visa id: 1759
  %1341 = extractelement <2 x i32> %1340, i32 0		; visa id: 1761
  %1342 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1341, i32 1
  %1343 = bitcast <2 x i32> %1342 to i64		; visa id: 1761
  %1344 = ashr exact i64 %1343, 32		; visa id: 1762
  %1345 = bitcast i64 %1344 to <2 x i32>		; visa id: 1763
  %1346 = extractelement <2 x i32> %1345, i32 0		; visa id: 1767
  %1347 = extractelement <2 x i32> %1345, i32 1		; visa id: 1767
  %1348 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1346, i32 %1347, i32 %41, i32 %42)
  %1349 = extractvalue { i32, i32 } %1348, 0		; visa id: 1767
  %1350 = extractvalue { i32, i32 } %1348, 1		; visa id: 1767
  %1351 = insertelement <2 x i32> undef, i32 %1349, i32 0		; visa id: 1774
  %1352 = insertelement <2 x i32> %1351, i32 %1350, i32 1		; visa id: 1775
  %1353 = bitcast <2 x i32> %1352 to i64		; visa id: 1776
  %1354 = shl i64 %1353, 1		; visa id: 1780
  %1355 = add i64 %.in401, %1354		; visa id: 1781
  %1356 = ashr i64 %1339, 31		; visa id: 1782
  %1357 = bitcast i64 %1356 to <2 x i32>		; visa id: 1783
  %1358 = extractelement <2 x i32> %1357, i32 0		; visa id: 1787
  %1359 = extractelement <2 x i32> %1357, i32 1		; visa id: 1787
  %1360 = and i32 %1358, -2		; visa id: 1787
  %1361 = insertelement <2 x i32> undef, i32 %1360, i32 0		; visa id: 1788
  %1362 = insertelement <2 x i32> %1361, i32 %1359, i32 1		; visa id: 1789
  %1363 = bitcast <2 x i32> %1362 to i64		; visa id: 1790
  %1364 = add i64 %1355, %1363		; visa id: 1794
  %1365 = inttoptr i64 %1364 to i16 addrspace(4)*		; visa id: 1795
  %1366 = addrspacecast i16 addrspace(4)* %1365 to i16 addrspace(1)*		; visa id: 1795
  %1367 = load i16, i16 addrspace(1)* %1366, align 2		; visa id: 1796
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1798
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1798
  %1368 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1798
  %1369 = insertelement <2 x i32> %1368, i32 %1229, i64 1		; visa id: 1799
  %1370 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1800
  store <2 x i32> %1369, <2 x i32>* %1370, align 4, !noalias !635		; visa id: 1800
  br label %._crit_edge234, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1802

._crit_edge234:                                   ; preds = %._crit_edge234.._crit_edge234_crit_edge, %1338
; BB129 :
  %1371 = phi i32 [ 0, %1338 ], [ %1380, %._crit_edge234.._crit_edge234_crit_edge ]
  %1372 = zext i32 %1371 to i64		; visa id: 1803
  %1373 = shl nuw nsw i64 %1372, 2		; visa id: 1804
  %1374 = add i64 %124, %1373		; visa id: 1805
  %1375 = inttoptr i64 %1374 to i32*		; visa id: 1806
  %1376 = load i32, i32* %1375, align 4, !noalias !635		; visa id: 1806
  %1377 = add i64 %119, %1373		; visa id: 1807
  %1378 = inttoptr i64 %1377 to i32*		; visa id: 1808
  store i32 %1376, i32* %1378, align 4, !alias.scope !635		; visa id: 1808
  %1379 = icmp eq i32 %1371, 0		; visa id: 1809
  br i1 %1379, label %._crit_edge234.._crit_edge234_crit_edge, label %1381, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1810

._crit_edge234.._crit_edge234_crit_edge:          ; preds = %._crit_edge234
; BB130 :
  %1380 = add nuw nsw i32 %1371, 1, !spirv.Decorations !631		; visa id: 1812
  br label %._crit_edge234, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1813

1381:                                             ; preds = %._crit_edge234
; BB131 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1815
  %1382 = load i64, i64* %120, align 8		; visa id: 1815
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1816
  %1383 = ashr i64 %1382, 32		; visa id: 1816
  %1384 = bitcast i64 %1383 to <2 x i32>		; visa id: 1817
  %1385 = extractelement <2 x i32> %1384, i32 0		; visa id: 1821
  %1386 = extractelement <2 x i32> %1384, i32 1		; visa id: 1821
  %1387 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1385, i32 %1386, i32 %44, i32 %45)
  %1388 = extractvalue { i32, i32 } %1387, 0		; visa id: 1821
  %1389 = extractvalue { i32, i32 } %1387, 1		; visa id: 1821
  %1390 = insertelement <2 x i32> undef, i32 %1388, i32 0		; visa id: 1828
  %1391 = insertelement <2 x i32> %1390, i32 %1389, i32 1		; visa id: 1829
  %1392 = bitcast <2 x i32> %1391 to i64		; visa id: 1830
  %1393 = bitcast i64 %1382 to <2 x i32>		; visa id: 1834
  %1394 = extractelement <2 x i32> %1393, i32 0		; visa id: 1836
  %1395 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1394, i32 1
  %1396 = bitcast <2 x i32> %1395 to i64		; visa id: 1836
  %1397 = shl i64 %1392, 1		; visa id: 1837
  %1398 = add i64 %.in400, %1397		; visa id: 1838
  %1399 = ashr exact i64 %1396, 31		; visa id: 1839
  %1400 = add i64 %1398, %1399		; visa id: 1840
  %1401 = inttoptr i64 %1400 to i16 addrspace(4)*		; visa id: 1841
  %1402 = addrspacecast i16 addrspace(4)* %1401 to i16 addrspace(1)*		; visa id: 1841
  %1403 = load i16, i16 addrspace(1)* %1402, align 2		; visa id: 1842
  %1404 = zext i16 %1367 to i32		; visa id: 1844
  %1405 = shl nuw i32 %1404, 16, !spirv.Decorations !639		; visa id: 1845
  %1406 = bitcast i32 %1405 to float
  %1407 = zext i16 %1403 to i32		; visa id: 1846
  %1408 = shl nuw i32 %1407, 16, !spirv.Decorations !639		; visa id: 1847
  %1409 = bitcast i32 %1408 to float
  %1410 = fmul reassoc nsz arcp contract float %1406, %1409, !spirv.Decorations !618
  %1411 = fadd reassoc nsz arcp contract float %1410, %.sroa.78.1, !spirv.Decorations !618		; visa id: 1848
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1849

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %1381
; BB132 :
  %.sroa.78.2 = phi float [ %1411, %1381 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  %1412 = icmp slt i32 %315, %const_reg_dword
  %1413 = icmp slt i32 %1229, %const_reg_dword1		; visa id: 1850
  %1414 = and i1 %1412, %1413		; visa id: 1851
  br i1 %1414, label %1415, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1853

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
; BB:
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1415:                                             ; preds = %._crit_edge.1.3
; BB134 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1855
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1855
  %1416 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 1855
  %1417 = insertelement <2 x i32> %1416, i32 %113, i64 1		; visa id: 1856
  %1418 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1857
  store <2 x i32> %1417, <2 x i32>* %1418, align 4, !noalias !625		; visa id: 1857
  br label %._crit_edge235, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1859

._crit_edge235:                                   ; preds = %._crit_edge235.._crit_edge235_crit_edge, %1415
; BB135 :
  %1419 = phi i32 [ 0, %1415 ], [ %1428, %._crit_edge235.._crit_edge235_crit_edge ]
  %1420 = zext i32 %1419 to i64		; visa id: 1860
  %1421 = shl nuw nsw i64 %1420, 2		; visa id: 1861
  %1422 = add i64 %133, %1421		; visa id: 1862
  %1423 = inttoptr i64 %1422 to i32*		; visa id: 1863
  %1424 = load i32, i32* %1423, align 4, !noalias !625		; visa id: 1863
  %1425 = add i64 %128, %1421		; visa id: 1864
  %1426 = inttoptr i64 %1425 to i32*		; visa id: 1865
  store i32 %1424, i32* %1426, align 4, !alias.scope !625		; visa id: 1865
  %1427 = icmp eq i32 %1419, 0		; visa id: 1866
  br i1 %1427, label %._crit_edge235.._crit_edge235_crit_edge, label %1429, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1867

._crit_edge235.._crit_edge235_crit_edge:          ; preds = %._crit_edge235
; BB136 :
  %1428 = add nuw nsw i32 %1419, 1, !spirv.Decorations !631		; visa id: 1869
  br label %._crit_edge235, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1870

1429:                                             ; preds = %._crit_edge235
; BB137 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1872
  %1430 = load i64, i64* %129, align 8		; visa id: 1872
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1873
  %1431 = bitcast i64 %1430 to <2 x i32>		; visa id: 1873
  %1432 = extractelement <2 x i32> %1431, i32 0		; visa id: 1875
  %1433 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1432, i32 1
  %1434 = bitcast <2 x i32> %1433 to i64		; visa id: 1875
  %1435 = ashr exact i64 %1434, 32		; visa id: 1876
  %1436 = bitcast i64 %1435 to <2 x i32>		; visa id: 1877
  %1437 = extractelement <2 x i32> %1436, i32 0		; visa id: 1881
  %1438 = extractelement <2 x i32> %1436, i32 1		; visa id: 1881
  %1439 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1437, i32 %1438, i32 %41, i32 %42)
  %1440 = extractvalue { i32, i32 } %1439, 0		; visa id: 1881
  %1441 = extractvalue { i32, i32 } %1439, 1		; visa id: 1881
  %1442 = insertelement <2 x i32> undef, i32 %1440, i32 0		; visa id: 1888
  %1443 = insertelement <2 x i32> %1442, i32 %1441, i32 1		; visa id: 1889
  %1444 = bitcast <2 x i32> %1443 to i64		; visa id: 1890
  %1445 = shl i64 %1444, 1		; visa id: 1894
  %1446 = add i64 %.in401, %1445		; visa id: 1895
  %1447 = ashr i64 %1430, 31		; visa id: 1896
  %1448 = bitcast i64 %1447 to <2 x i32>		; visa id: 1897
  %1449 = extractelement <2 x i32> %1448, i32 0		; visa id: 1901
  %1450 = extractelement <2 x i32> %1448, i32 1		; visa id: 1901
  %1451 = and i32 %1449, -2		; visa id: 1901
  %1452 = insertelement <2 x i32> undef, i32 %1451, i32 0		; visa id: 1902
  %1453 = insertelement <2 x i32> %1452, i32 %1450, i32 1		; visa id: 1903
  %1454 = bitcast <2 x i32> %1453 to i64		; visa id: 1904
  %1455 = add i64 %1446, %1454		; visa id: 1908
  %1456 = inttoptr i64 %1455 to i16 addrspace(4)*		; visa id: 1909
  %1457 = addrspacecast i16 addrspace(4)* %1456 to i16 addrspace(1)*		; visa id: 1909
  %1458 = load i16, i16 addrspace(1)* %1457, align 2		; visa id: 1910
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1912
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1912
  %1459 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1912
  %1460 = insertelement <2 x i32> %1459, i32 %1229, i64 1		; visa id: 1913
  %1461 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1914
  store <2 x i32> %1460, <2 x i32>* %1461, align 4, !noalias !635		; visa id: 1914
  br label %._crit_edge236, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1916

._crit_edge236:                                   ; preds = %._crit_edge236.._crit_edge236_crit_edge, %1429
; BB138 :
  %1462 = phi i32 [ 0, %1429 ], [ %1471, %._crit_edge236.._crit_edge236_crit_edge ]
  %1463 = zext i32 %1462 to i64		; visa id: 1917
  %1464 = shl nuw nsw i64 %1463, 2		; visa id: 1918
  %1465 = add i64 %124, %1464		; visa id: 1919
  %1466 = inttoptr i64 %1465 to i32*		; visa id: 1920
  %1467 = load i32, i32* %1466, align 4, !noalias !635		; visa id: 1920
  %1468 = add i64 %119, %1464		; visa id: 1921
  %1469 = inttoptr i64 %1468 to i32*		; visa id: 1922
  store i32 %1467, i32* %1469, align 4, !alias.scope !635		; visa id: 1922
  %1470 = icmp eq i32 %1462, 0		; visa id: 1923
  br i1 %1470, label %._crit_edge236.._crit_edge236_crit_edge, label %1472, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1924

._crit_edge236.._crit_edge236_crit_edge:          ; preds = %._crit_edge236
; BB139 :
  %1471 = add nuw nsw i32 %1462, 1, !spirv.Decorations !631		; visa id: 1926
  br label %._crit_edge236, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1927

1472:                                             ; preds = %._crit_edge236
; BB140 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1929
  %1473 = load i64, i64* %120, align 8		; visa id: 1929
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1930
  %1474 = ashr i64 %1473, 32		; visa id: 1930
  %1475 = bitcast i64 %1474 to <2 x i32>		; visa id: 1931
  %1476 = extractelement <2 x i32> %1475, i32 0		; visa id: 1935
  %1477 = extractelement <2 x i32> %1475, i32 1		; visa id: 1935
  %1478 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1476, i32 %1477, i32 %44, i32 %45)
  %1479 = extractvalue { i32, i32 } %1478, 0		; visa id: 1935
  %1480 = extractvalue { i32, i32 } %1478, 1		; visa id: 1935
  %1481 = insertelement <2 x i32> undef, i32 %1479, i32 0		; visa id: 1942
  %1482 = insertelement <2 x i32> %1481, i32 %1480, i32 1		; visa id: 1943
  %1483 = bitcast <2 x i32> %1482 to i64		; visa id: 1944
  %1484 = bitcast i64 %1473 to <2 x i32>		; visa id: 1948
  %1485 = extractelement <2 x i32> %1484, i32 0		; visa id: 1950
  %1486 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1485, i32 1
  %1487 = bitcast <2 x i32> %1486 to i64		; visa id: 1950
  %1488 = shl i64 %1483, 1		; visa id: 1951
  %1489 = add i64 %.in400, %1488		; visa id: 1952
  %1490 = ashr exact i64 %1487, 31		; visa id: 1953
  %1491 = add i64 %1489, %1490		; visa id: 1954
  %1492 = inttoptr i64 %1491 to i16 addrspace(4)*		; visa id: 1955
  %1493 = addrspacecast i16 addrspace(4)* %1492 to i16 addrspace(1)*		; visa id: 1955
  %1494 = load i16, i16 addrspace(1)* %1493, align 2		; visa id: 1956
  %1495 = zext i16 %1458 to i32		; visa id: 1958
  %1496 = shl nuw i32 %1495, 16, !spirv.Decorations !639		; visa id: 1959
  %1497 = bitcast i32 %1496 to float
  %1498 = zext i16 %1494 to i32		; visa id: 1960
  %1499 = shl nuw i32 %1498, 16, !spirv.Decorations !639		; visa id: 1961
  %1500 = bitcast i32 %1499 to float
  %1501 = fmul reassoc nsz arcp contract float %1497, %1500, !spirv.Decorations !618
  %1502 = fadd reassoc nsz arcp contract float %1501, %.sroa.142.1, !spirv.Decorations !618		; visa id: 1962
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1963

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %1472
; BB141 :
  %.sroa.142.2 = phi float [ %1502, %1472 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  %1503 = icmp slt i32 %407, %const_reg_dword
  %1504 = icmp slt i32 %1229, %const_reg_dword1		; visa id: 1964
  %1505 = and i1 %1503, %1504		; visa id: 1965
  br i1 %1505, label %1506, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1967

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
; BB:
  br label %.preheader.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1506:                                             ; preds = %._crit_edge.2.3
; BB143 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1969
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1969
  %1507 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 1969
  %1508 = insertelement <2 x i32> %1507, i32 %113, i64 1		; visa id: 1970
  %1509 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1971
  store <2 x i32> %1508, <2 x i32>* %1509, align 4, !noalias !625		; visa id: 1971
  br label %._crit_edge237, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1973

._crit_edge237:                                   ; preds = %._crit_edge237.._crit_edge237_crit_edge, %1506
; BB144 :
  %1510 = phi i32 [ 0, %1506 ], [ %1519, %._crit_edge237.._crit_edge237_crit_edge ]
  %1511 = zext i32 %1510 to i64		; visa id: 1974
  %1512 = shl nuw nsw i64 %1511, 2		; visa id: 1975
  %1513 = add i64 %133, %1512		; visa id: 1976
  %1514 = inttoptr i64 %1513 to i32*		; visa id: 1977
  %1515 = load i32, i32* %1514, align 4, !noalias !625		; visa id: 1977
  %1516 = add i64 %128, %1512		; visa id: 1978
  %1517 = inttoptr i64 %1516 to i32*		; visa id: 1979
  store i32 %1515, i32* %1517, align 4, !alias.scope !625		; visa id: 1979
  %1518 = icmp eq i32 %1510, 0		; visa id: 1980
  br i1 %1518, label %._crit_edge237.._crit_edge237_crit_edge, label %1520, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1981

._crit_edge237.._crit_edge237_crit_edge:          ; preds = %._crit_edge237
; BB145 :
  %1519 = add nuw nsw i32 %1510, 1, !spirv.Decorations !631		; visa id: 1983
  br label %._crit_edge237, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1984

1520:                                             ; preds = %._crit_edge237
; BB146 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1986
  %1521 = load i64, i64* %129, align 8		; visa id: 1986
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1987
  %1522 = bitcast i64 %1521 to <2 x i32>		; visa id: 1987
  %1523 = extractelement <2 x i32> %1522, i32 0		; visa id: 1989
  %1524 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1523, i32 1
  %1525 = bitcast <2 x i32> %1524 to i64		; visa id: 1989
  %1526 = ashr exact i64 %1525, 32		; visa id: 1990
  %1527 = bitcast i64 %1526 to <2 x i32>		; visa id: 1991
  %1528 = extractelement <2 x i32> %1527, i32 0		; visa id: 1995
  %1529 = extractelement <2 x i32> %1527, i32 1		; visa id: 1995
  %1530 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1528, i32 %1529, i32 %41, i32 %42)
  %1531 = extractvalue { i32, i32 } %1530, 0		; visa id: 1995
  %1532 = extractvalue { i32, i32 } %1530, 1		; visa id: 1995
  %1533 = insertelement <2 x i32> undef, i32 %1531, i32 0		; visa id: 2002
  %1534 = insertelement <2 x i32> %1533, i32 %1532, i32 1		; visa id: 2003
  %1535 = bitcast <2 x i32> %1534 to i64		; visa id: 2004
  %1536 = shl i64 %1535, 1		; visa id: 2008
  %1537 = add i64 %.in401, %1536		; visa id: 2009
  %1538 = ashr i64 %1521, 31		; visa id: 2010
  %1539 = bitcast i64 %1538 to <2 x i32>		; visa id: 2011
  %1540 = extractelement <2 x i32> %1539, i32 0		; visa id: 2015
  %1541 = extractelement <2 x i32> %1539, i32 1		; visa id: 2015
  %1542 = and i32 %1540, -2		; visa id: 2015
  %1543 = insertelement <2 x i32> undef, i32 %1542, i32 0		; visa id: 2016
  %1544 = insertelement <2 x i32> %1543, i32 %1541, i32 1		; visa id: 2017
  %1545 = bitcast <2 x i32> %1544 to i64		; visa id: 2018
  %1546 = add i64 %1537, %1545		; visa id: 2022
  %1547 = inttoptr i64 %1546 to i16 addrspace(4)*		; visa id: 2023
  %1548 = addrspacecast i16 addrspace(4)* %1547 to i16 addrspace(1)*		; visa id: 2023
  %1549 = load i16, i16 addrspace(1)* %1548, align 2		; visa id: 2024
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2026
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2026
  %1550 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2026
  %1551 = insertelement <2 x i32> %1550, i32 %1229, i64 1		; visa id: 2027
  %1552 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2028
  store <2 x i32> %1551, <2 x i32>* %1552, align 4, !noalias !635		; visa id: 2028
  br label %._crit_edge238, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2030

._crit_edge238:                                   ; preds = %._crit_edge238.._crit_edge238_crit_edge, %1520
; BB147 :
  %1553 = phi i32 [ 0, %1520 ], [ %1562, %._crit_edge238.._crit_edge238_crit_edge ]
  %1554 = zext i32 %1553 to i64		; visa id: 2031
  %1555 = shl nuw nsw i64 %1554, 2		; visa id: 2032
  %1556 = add i64 %124, %1555		; visa id: 2033
  %1557 = inttoptr i64 %1556 to i32*		; visa id: 2034
  %1558 = load i32, i32* %1557, align 4, !noalias !635		; visa id: 2034
  %1559 = add i64 %119, %1555		; visa id: 2035
  %1560 = inttoptr i64 %1559 to i32*		; visa id: 2036
  store i32 %1558, i32* %1560, align 4, !alias.scope !635		; visa id: 2036
  %1561 = icmp eq i32 %1553, 0		; visa id: 2037
  br i1 %1561, label %._crit_edge238.._crit_edge238_crit_edge, label %1563, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2038

._crit_edge238.._crit_edge238_crit_edge:          ; preds = %._crit_edge238
; BB148 :
  %1562 = add nuw nsw i32 %1553, 1, !spirv.Decorations !631		; visa id: 2040
  br label %._crit_edge238, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2041

1563:                                             ; preds = %._crit_edge238
; BB149 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2043
  %1564 = load i64, i64* %120, align 8		; visa id: 2043
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2044
  %1565 = ashr i64 %1564, 32		; visa id: 2044
  %1566 = bitcast i64 %1565 to <2 x i32>		; visa id: 2045
  %1567 = extractelement <2 x i32> %1566, i32 0		; visa id: 2049
  %1568 = extractelement <2 x i32> %1566, i32 1		; visa id: 2049
  %1569 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1567, i32 %1568, i32 %44, i32 %45)
  %1570 = extractvalue { i32, i32 } %1569, 0		; visa id: 2049
  %1571 = extractvalue { i32, i32 } %1569, 1		; visa id: 2049
  %1572 = insertelement <2 x i32> undef, i32 %1570, i32 0		; visa id: 2056
  %1573 = insertelement <2 x i32> %1572, i32 %1571, i32 1		; visa id: 2057
  %1574 = bitcast <2 x i32> %1573 to i64		; visa id: 2058
  %1575 = bitcast i64 %1564 to <2 x i32>		; visa id: 2062
  %1576 = extractelement <2 x i32> %1575, i32 0		; visa id: 2064
  %1577 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1576, i32 1
  %1578 = bitcast <2 x i32> %1577 to i64		; visa id: 2064
  %1579 = shl i64 %1574, 1		; visa id: 2065
  %1580 = add i64 %.in400, %1579		; visa id: 2066
  %1581 = ashr exact i64 %1578, 31		; visa id: 2067
  %1582 = add i64 %1580, %1581		; visa id: 2068
  %1583 = inttoptr i64 %1582 to i16 addrspace(4)*		; visa id: 2069
  %1584 = addrspacecast i16 addrspace(4)* %1583 to i16 addrspace(1)*		; visa id: 2069
  %1585 = load i16, i16 addrspace(1)* %1584, align 2		; visa id: 2070
  %1586 = zext i16 %1549 to i32		; visa id: 2072
  %1587 = shl nuw i32 %1586, 16, !spirv.Decorations !639		; visa id: 2073
  %1588 = bitcast i32 %1587 to float
  %1589 = zext i16 %1585 to i32		; visa id: 2074
  %1590 = shl nuw i32 %1589, 16, !spirv.Decorations !639		; visa id: 2075
  %1591 = bitcast i32 %1590 to float
  %1592 = fmul reassoc nsz arcp contract float %1588, %1591, !spirv.Decorations !618
  %1593 = fadd reassoc nsz arcp contract float %1592, %.sroa.206.1, !spirv.Decorations !618		; visa id: 2076
  br label %.preheader.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2077

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %1563
; BB150 :
  %.sroa.206.2 = phi float [ %1593, %1563 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %1594 = add i32 %69, 4		; visa id: 2078
  %1595 = icmp slt i32 %1594, %const_reg_dword1		; visa id: 2079
  %1596 = icmp slt i32 %65, %const_reg_dword
  %1597 = and i1 %1596, %1595		; visa id: 2080
  br i1 %1597, label %1598, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2082

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
; BB:
  br label %._crit_edge.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1598:                                             ; preds = %.preheader.3
; BB152 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2084
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2084
  %1599 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 2084
  %1600 = insertelement <2 x i32> %1599, i32 %113, i64 1		; visa id: 2085
  %1601 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2086
  store <2 x i32> %1600, <2 x i32>* %1601, align 4, !noalias !625		; visa id: 2086
  br label %._crit_edge239, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2088

._crit_edge239:                                   ; preds = %._crit_edge239.._crit_edge239_crit_edge, %1598
; BB153 :
  %1602 = phi i32 [ 0, %1598 ], [ %1611, %._crit_edge239.._crit_edge239_crit_edge ]
  %1603 = zext i32 %1602 to i64		; visa id: 2089
  %1604 = shl nuw nsw i64 %1603, 2		; visa id: 2090
  %1605 = add i64 %133, %1604		; visa id: 2091
  %1606 = inttoptr i64 %1605 to i32*		; visa id: 2092
  %1607 = load i32, i32* %1606, align 4, !noalias !625		; visa id: 2092
  %1608 = add i64 %128, %1604		; visa id: 2093
  %1609 = inttoptr i64 %1608 to i32*		; visa id: 2094
  store i32 %1607, i32* %1609, align 4, !alias.scope !625		; visa id: 2094
  %1610 = icmp eq i32 %1602, 0		; visa id: 2095
  br i1 %1610, label %._crit_edge239.._crit_edge239_crit_edge, label %1612, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2096

._crit_edge239.._crit_edge239_crit_edge:          ; preds = %._crit_edge239
; BB154 :
  %1611 = add nuw nsw i32 %1602, 1, !spirv.Decorations !631		; visa id: 2098
  br label %._crit_edge239, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2099

1612:                                             ; preds = %._crit_edge239
; BB155 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2101
  %1613 = load i64, i64* %129, align 8		; visa id: 2101
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2102
  %1614 = bitcast i64 %1613 to <2 x i32>		; visa id: 2102
  %1615 = extractelement <2 x i32> %1614, i32 0		; visa id: 2104
  %1616 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1615, i32 1
  %1617 = bitcast <2 x i32> %1616 to i64		; visa id: 2104
  %1618 = ashr exact i64 %1617, 32		; visa id: 2105
  %1619 = bitcast i64 %1618 to <2 x i32>		; visa id: 2106
  %1620 = extractelement <2 x i32> %1619, i32 0		; visa id: 2110
  %1621 = extractelement <2 x i32> %1619, i32 1		; visa id: 2110
  %1622 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1620, i32 %1621, i32 %41, i32 %42)
  %1623 = extractvalue { i32, i32 } %1622, 0		; visa id: 2110
  %1624 = extractvalue { i32, i32 } %1622, 1		; visa id: 2110
  %1625 = insertelement <2 x i32> undef, i32 %1623, i32 0		; visa id: 2117
  %1626 = insertelement <2 x i32> %1625, i32 %1624, i32 1		; visa id: 2118
  %1627 = bitcast <2 x i32> %1626 to i64		; visa id: 2119
  %1628 = shl i64 %1627, 1		; visa id: 2123
  %1629 = add i64 %.in401, %1628		; visa id: 2124
  %1630 = ashr i64 %1613, 31		; visa id: 2125
  %1631 = bitcast i64 %1630 to <2 x i32>		; visa id: 2126
  %1632 = extractelement <2 x i32> %1631, i32 0		; visa id: 2130
  %1633 = extractelement <2 x i32> %1631, i32 1		; visa id: 2130
  %1634 = and i32 %1632, -2		; visa id: 2130
  %1635 = insertelement <2 x i32> undef, i32 %1634, i32 0		; visa id: 2131
  %1636 = insertelement <2 x i32> %1635, i32 %1633, i32 1		; visa id: 2132
  %1637 = bitcast <2 x i32> %1636 to i64		; visa id: 2133
  %1638 = add i64 %1629, %1637		; visa id: 2137
  %1639 = inttoptr i64 %1638 to i16 addrspace(4)*		; visa id: 2138
  %1640 = addrspacecast i16 addrspace(4)* %1639 to i16 addrspace(1)*		; visa id: 2138
  %1641 = load i16, i16 addrspace(1)* %1640, align 2		; visa id: 2139
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2141
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2141
  %1642 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2141
  %1643 = insertelement <2 x i32> %1642, i32 %1594, i64 1		; visa id: 2142
  %1644 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2143
  store <2 x i32> %1643, <2 x i32>* %1644, align 4, !noalias !635		; visa id: 2143
  br label %._crit_edge240, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2145

._crit_edge240:                                   ; preds = %._crit_edge240.._crit_edge240_crit_edge, %1612
; BB156 :
  %1645 = phi i32 [ 0, %1612 ], [ %1654, %._crit_edge240.._crit_edge240_crit_edge ]
  %1646 = zext i32 %1645 to i64		; visa id: 2146
  %1647 = shl nuw nsw i64 %1646, 2		; visa id: 2147
  %1648 = add i64 %124, %1647		; visa id: 2148
  %1649 = inttoptr i64 %1648 to i32*		; visa id: 2149
  %1650 = load i32, i32* %1649, align 4, !noalias !635		; visa id: 2149
  %1651 = add i64 %119, %1647		; visa id: 2150
  %1652 = inttoptr i64 %1651 to i32*		; visa id: 2151
  store i32 %1650, i32* %1652, align 4, !alias.scope !635		; visa id: 2151
  %1653 = icmp eq i32 %1645, 0		; visa id: 2152
  br i1 %1653, label %._crit_edge240.._crit_edge240_crit_edge, label %1655, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2153

._crit_edge240.._crit_edge240_crit_edge:          ; preds = %._crit_edge240
; BB157 :
  %1654 = add nuw nsw i32 %1645, 1, !spirv.Decorations !631		; visa id: 2155
  br label %._crit_edge240, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2156

1655:                                             ; preds = %._crit_edge240
; BB158 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2158
  %1656 = load i64, i64* %120, align 8		; visa id: 2158
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2159
  %1657 = ashr i64 %1656, 32		; visa id: 2159
  %1658 = bitcast i64 %1657 to <2 x i32>		; visa id: 2160
  %1659 = extractelement <2 x i32> %1658, i32 0		; visa id: 2164
  %1660 = extractelement <2 x i32> %1658, i32 1		; visa id: 2164
  %1661 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1659, i32 %1660, i32 %44, i32 %45)
  %1662 = extractvalue { i32, i32 } %1661, 0		; visa id: 2164
  %1663 = extractvalue { i32, i32 } %1661, 1		; visa id: 2164
  %1664 = insertelement <2 x i32> undef, i32 %1662, i32 0		; visa id: 2171
  %1665 = insertelement <2 x i32> %1664, i32 %1663, i32 1		; visa id: 2172
  %1666 = bitcast <2 x i32> %1665 to i64		; visa id: 2173
  %1667 = bitcast i64 %1656 to <2 x i32>		; visa id: 2177
  %1668 = extractelement <2 x i32> %1667, i32 0		; visa id: 2179
  %1669 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1668, i32 1
  %1670 = bitcast <2 x i32> %1669 to i64		; visa id: 2179
  %1671 = shl i64 %1666, 1		; visa id: 2180
  %1672 = add i64 %.in400, %1671		; visa id: 2181
  %1673 = ashr exact i64 %1670, 31		; visa id: 2182
  %1674 = add i64 %1672, %1673		; visa id: 2183
  %1675 = inttoptr i64 %1674 to i16 addrspace(4)*		; visa id: 2184
  %1676 = addrspacecast i16 addrspace(4)* %1675 to i16 addrspace(1)*		; visa id: 2184
  %1677 = load i16, i16 addrspace(1)* %1676, align 2		; visa id: 2185
  %1678 = zext i16 %1641 to i32		; visa id: 2187
  %1679 = shl nuw i32 %1678, 16, !spirv.Decorations !639		; visa id: 2188
  %1680 = bitcast i32 %1679 to float
  %1681 = zext i16 %1677 to i32		; visa id: 2189
  %1682 = shl nuw i32 %1681, 16, !spirv.Decorations !639		; visa id: 2190
  %1683 = bitcast i32 %1682 to float
  %1684 = fmul reassoc nsz arcp contract float %1680, %1683, !spirv.Decorations !618
  %1685 = fadd reassoc nsz arcp contract float %1684, %.sroa.18.1, !spirv.Decorations !618		; visa id: 2191
  br label %._crit_edge.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2192

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %1655
; BB159 :
  %.sroa.18.2 = phi float [ %1685, %1655 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  %1686 = icmp slt i32 %223, %const_reg_dword
  %1687 = icmp slt i32 %1594, %const_reg_dword1		; visa id: 2193
  %1688 = and i1 %1686, %1687		; visa id: 2194
  br i1 %1688, label %1689, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2196

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
; BB:
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1689:                                             ; preds = %._crit_edge.4
; BB161 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2198
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2198
  %1690 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 2198
  %1691 = insertelement <2 x i32> %1690, i32 %113, i64 1		; visa id: 2199
  %1692 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2200
  store <2 x i32> %1691, <2 x i32>* %1692, align 4, !noalias !625		; visa id: 2200
  br label %._crit_edge241, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2202

._crit_edge241:                                   ; preds = %._crit_edge241.._crit_edge241_crit_edge, %1689
; BB162 :
  %1693 = phi i32 [ 0, %1689 ], [ %1702, %._crit_edge241.._crit_edge241_crit_edge ]
  %1694 = zext i32 %1693 to i64		; visa id: 2203
  %1695 = shl nuw nsw i64 %1694, 2		; visa id: 2204
  %1696 = add i64 %133, %1695		; visa id: 2205
  %1697 = inttoptr i64 %1696 to i32*		; visa id: 2206
  %1698 = load i32, i32* %1697, align 4, !noalias !625		; visa id: 2206
  %1699 = add i64 %128, %1695		; visa id: 2207
  %1700 = inttoptr i64 %1699 to i32*		; visa id: 2208
  store i32 %1698, i32* %1700, align 4, !alias.scope !625		; visa id: 2208
  %1701 = icmp eq i32 %1693, 0		; visa id: 2209
  br i1 %1701, label %._crit_edge241.._crit_edge241_crit_edge, label %1703, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2210

._crit_edge241.._crit_edge241_crit_edge:          ; preds = %._crit_edge241
; BB163 :
  %1702 = add nuw nsw i32 %1693, 1, !spirv.Decorations !631		; visa id: 2212
  br label %._crit_edge241, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2213

1703:                                             ; preds = %._crit_edge241
; BB164 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2215
  %1704 = load i64, i64* %129, align 8		; visa id: 2215
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2216
  %1705 = bitcast i64 %1704 to <2 x i32>		; visa id: 2216
  %1706 = extractelement <2 x i32> %1705, i32 0		; visa id: 2218
  %1707 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1706, i32 1
  %1708 = bitcast <2 x i32> %1707 to i64		; visa id: 2218
  %1709 = ashr exact i64 %1708, 32		; visa id: 2219
  %1710 = bitcast i64 %1709 to <2 x i32>		; visa id: 2220
  %1711 = extractelement <2 x i32> %1710, i32 0		; visa id: 2224
  %1712 = extractelement <2 x i32> %1710, i32 1		; visa id: 2224
  %1713 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1711, i32 %1712, i32 %41, i32 %42)
  %1714 = extractvalue { i32, i32 } %1713, 0		; visa id: 2224
  %1715 = extractvalue { i32, i32 } %1713, 1		; visa id: 2224
  %1716 = insertelement <2 x i32> undef, i32 %1714, i32 0		; visa id: 2231
  %1717 = insertelement <2 x i32> %1716, i32 %1715, i32 1		; visa id: 2232
  %1718 = bitcast <2 x i32> %1717 to i64		; visa id: 2233
  %1719 = shl i64 %1718, 1		; visa id: 2237
  %1720 = add i64 %.in401, %1719		; visa id: 2238
  %1721 = ashr i64 %1704, 31		; visa id: 2239
  %1722 = bitcast i64 %1721 to <2 x i32>		; visa id: 2240
  %1723 = extractelement <2 x i32> %1722, i32 0		; visa id: 2244
  %1724 = extractelement <2 x i32> %1722, i32 1		; visa id: 2244
  %1725 = and i32 %1723, -2		; visa id: 2244
  %1726 = insertelement <2 x i32> undef, i32 %1725, i32 0		; visa id: 2245
  %1727 = insertelement <2 x i32> %1726, i32 %1724, i32 1		; visa id: 2246
  %1728 = bitcast <2 x i32> %1727 to i64		; visa id: 2247
  %1729 = add i64 %1720, %1728		; visa id: 2251
  %1730 = inttoptr i64 %1729 to i16 addrspace(4)*		; visa id: 2252
  %1731 = addrspacecast i16 addrspace(4)* %1730 to i16 addrspace(1)*		; visa id: 2252
  %1732 = load i16, i16 addrspace(1)* %1731, align 2		; visa id: 2253
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2255
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2255
  %1733 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2255
  %1734 = insertelement <2 x i32> %1733, i32 %1594, i64 1		; visa id: 2256
  %1735 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2257
  store <2 x i32> %1734, <2 x i32>* %1735, align 4, !noalias !635		; visa id: 2257
  br label %._crit_edge242, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2259

._crit_edge242:                                   ; preds = %._crit_edge242.._crit_edge242_crit_edge, %1703
; BB165 :
  %1736 = phi i32 [ 0, %1703 ], [ %1745, %._crit_edge242.._crit_edge242_crit_edge ]
  %1737 = zext i32 %1736 to i64		; visa id: 2260
  %1738 = shl nuw nsw i64 %1737, 2		; visa id: 2261
  %1739 = add i64 %124, %1738		; visa id: 2262
  %1740 = inttoptr i64 %1739 to i32*		; visa id: 2263
  %1741 = load i32, i32* %1740, align 4, !noalias !635		; visa id: 2263
  %1742 = add i64 %119, %1738		; visa id: 2264
  %1743 = inttoptr i64 %1742 to i32*		; visa id: 2265
  store i32 %1741, i32* %1743, align 4, !alias.scope !635		; visa id: 2265
  %1744 = icmp eq i32 %1736, 0		; visa id: 2266
  br i1 %1744, label %._crit_edge242.._crit_edge242_crit_edge, label %1746, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2267

._crit_edge242.._crit_edge242_crit_edge:          ; preds = %._crit_edge242
; BB166 :
  %1745 = add nuw nsw i32 %1736, 1, !spirv.Decorations !631		; visa id: 2269
  br label %._crit_edge242, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2270

1746:                                             ; preds = %._crit_edge242
; BB167 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2272
  %1747 = load i64, i64* %120, align 8		; visa id: 2272
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2273
  %1748 = ashr i64 %1747, 32		; visa id: 2273
  %1749 = bitcast i64 %1748 to <2 x i32>		; visa id: 2274
  %1750 = extractelement <2 x i32> %1749, i32 0		; visa id: 2278
  %1751 = extractelement <2 x i32> %1749, i32 1		; visa id: 2278
  %1752 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1750, i32 %1751, i32 %44, i32 %45)
  %1753 = extractvalue { i32, i32 } %1752, 0		; visa id: 2278
  %1754 = extractvalue { i32, i32 } %1752, 1		; visa id: 2278
  %1755 = insertelement <2 x i32> undef, i32 %1753, i32 0		; visa id: 2285
  %1756 = insertelement <2 x i32> %1755, i32 %1754, i32 1		; visa id: 2286
  %1757 = bitcast <2 x i32> %1756 to i64		; visa id: 2287
  %1758 = bitcast i64 %1747 to <2 x i32>		; visa id: 2291
  %1759 = extractelement <2 x i32> %1758, i32 0		; visa id: 2293
  %1760 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1759, i32 1
  %1761 = bitcast <2 x i32> %1760 to i64		; visa id: 2293
  %1762 = shl i64 %1757, 1		; visa id: 2294
  %1763 = add i64 %.in400, %1762		; visa id: 2295
  %1764 = ashr exact i64 %1761, 31		; visa id: 2296
  %1765 = add i64 %1763, %1764		; visa id: 2297
  %1766 = inttoptr i64 %1765 to i16 addrspace(4)*		; visa id: 2298
  %1767 = addrspacecast i16 addrspace(4)* %1766 to i16 addrspace(1)*		; visa id: 2298
  %1768 = load i16, i16 addrspace(1)* %1767, align 2		; visa id: 2299
  %1769 = zext i16 %1732 to i32		; visa id: 2301
  %1770 = shl nuw i32 %1769, 16, !spirv.Decorations !639		; visa id: 2302
  %1771 = bitcast i32 %1770 to float
  %1772 = zext i16 %1768 to i32		; visa id: 2303
  %1773 = shl nuw i32 %1772, 16, !spirv.Decorations !639		; visa id: 2304
  %1774 = bitcast i32 %1773 to float
  %1775 = fmul reassoc nsz arcp contract float %1771, %1774, !spirv.Decorations !618
  %1776 = fadd reassoc nsz arcp contract float %1775, %.sroa.82.1, !spirv.Decorations !618		; visa id: 2305
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2306

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %1746
; BB168 :
  %.sroa.82.2 = phi float [ %1776, %1746 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  %1777 = icmp slt i32 %315, %const_reg_dword
  %1778 = icmp slt i32 %1594, %const_reg_dword1		; visa id: 2307
  %1779 = and i1 %1777, %1778		; visa id: 2308
  br i1 %1779, label %1780, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2310

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
; BB:
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1780:                                             ; preds = %._crit_edge.1.4
; BB170 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2312
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2312
  %1781 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 2312
  %1782 = insertelement <2 x i32> %1781, i32 %113, i64 1		; visa id: 2313
  %1783 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2314
  store <2 x i32> %1782, <2 x i32>* %1783, align 4, !noalias !625		; visa id: 2314
  br label %._crit_edge243, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2316

._crit_edge243:                                   ; preds = %._crit_edge243.._crit_edge243_crit_edge, %1780
; BB171 :
  %1784 = phi i32 [ 0, %1780 ], [ %1793, %._crit_edge243.._crit_edge243_crit_edge ]
  %1785 = zext i32 %1784 to i64		; visa id: 2317
  %1786 = shl nuw nsw i64 %1785, 2		; visa id: 2318
  %1787 = add i64 %133, %1786		; visa id: 2319
  %1788 = inttoptr i64 %1787 to i32*		; visa id: 2320
  %1789 = load i32, i32* %1788, align 4, !noalias !625		; visa id: 2320
  %1790 = add i64 %128, %1786		; visa id: 2321
  %1791 = inttoptr i64 %1790 to i32*		; visa id: 2322
  store i32 %1789, i32* %1791, align 4, !alias.scope !625		; visa id: 2322
  %1792 = icmp eq i32 %1784, 0		; visa id: 2323
  br i1 %1792, label %._crit_edge243.._crit_edge243_crit_edge, label %1794, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2324

._crit_edge243.._crit_edge243_crit_edge:          ; preds = %._crit_edge243
; BB172 :
  %1793 = add nuw nsw i32 %1784, 1, !spirv.Decorations !631		; visa id: 2326
  br label %._crit_edge243, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2327

1794:                                             ; preds = %._crit_edge243
; BB173 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2329
  %1795 = load i64, i64* %129, align 8		; visa id: 2329
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2330
  %1796 = bitcast i64 %1795 to <2 x i32>		; visa id: 2330
  %1797 = extractelement <2 x i32> %1796, i32 0		; visa id: 2332
  %1798 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1797, i32 1
  %1799 = bitcast <2 x i32> %1798 to i64		; visa id: 2332
  %1800 = ashr exact i64 %1799, 32		; visa id: 2333
  %1801 = bitcast i64 %1800 to <2 x i32>		; visa id: 2334
  %1802 = extractelement <2 x i32> %1801, i32 0		; visa id: 2338
  %1803 = extractelement <2 x i32> %1801, i32 1		; visa id: 2338
  %1804 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1802, i32 %1803, i32 %41, i32 %42)
  %1805 = extractvalue { i32, i32 } %1804, 0		; visa id: 2338
  %1806 = extractvalue { i32, i32 } %1804, 1		; visa id: 2338
  %1807 = insertelement <2 x i32> undef, i32 %1805, i32 0		; visa id: 2345
  %1808 = insertelement <2 x i32> %1807, i32 %1806, i32 1		; visa id: 2346
  %1809 = bitcast <2 x i32> %1808 to i64		; visa id: 2347
  %1810 = shl i64 %1809, 1		; visa id: 2351
  %1811 = add i64 %.in401, %1810		; visa id: 2352
  %1812 = ashr i64 %1795, 31		; visa id: 2353
  %1813 = bitcast i64 %1812 to <2 x i32>		; visa id: 2354
  %1814 = extractelement <2 x i32> %1813, i32 0		; visa id: 2358
  %1815 = extractelement <2 x i32> %1813, i32 1		; visa id: 2358
  %1816 = and i32 %1814, -2		; visa id: 2358
  %1817 = insertelement <2 x i32> undef, i32 %1816, i32 0		; visa id: 2359
  %1818 = insertelement <2 x i32> %1817, i32 %1815, i32 1		; visa id: 2360
  %1819 = bitcast <2 x i32> %1818 to i64		; visa id: 2361
  %1820 = add i64 %1811, %1819		; visa id: 2365
  %1821 = inttoptr i64 %1820 to i16 addrspace(4)*		; visa id: 2366
  %1822 = addrspacecast i16 addrspace(4)* %1821 to i16 addrspace(1)*		; visa id: 2366
  %1823 = load i16, i16 addrspace(1)* %1822, align 2		; visa id: 2367
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2369
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2369
  %1824 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2369
  %1825 = insertelement <2 x i32> %1824, i32 %1594, i64 1		; visa id: 2370
  %1826 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2371
  store <2 x i32> %1825, <2 x i32>* %1826, align 4, !noalias !635		; visa id: 2371
  br label %._crit_edge244, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2373

._crit_edge244:                                   ; preds = %._crit_edge244.._crit_edge244_crit_edge, %1794
; BB174 :
  %1827 = phi i32 [ 0, %1794 ], [ %1836, %._crit_edge244.._crit_edge244_crit_edge ]
  %1828 = zext i32 %1827 to i64		; visa id: 2374
  %1829 = shl nuw nsw i64 %1828, 2		; visa id: 2375
  %1830 = add i64 %124, %1829		; visa id: 2376
  %1831 = inttoptr i64 %1830 to i32*		; visa id: 2377
  %1832 = load i32, i32* %1831, align 4, !noalias !635		; visa id: 2377
  %1833 = add i64 %119, %1829		; visa id: 2378
  %1834 = inttoptr i64 %1833 to i32*		; visa id: 2379
  store i32 %1832, i32* %1834, align 4, !alias.scope !635		; visa id: 2379
  %1835 = icmp eq i32 %1827, 0		; visa id: 2380
  br i1 %1835, label %._crit_edge244.._crit_edge244_crit_edge, label %1837, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2381

._crit_edge244.._crit_edge244_crit_edge:          ; preds = %._crit_edge244
; BB175 :
  %1836 = add nuw nsw i32 %1827, 1, !spirv.Decorations !631		; visa id: 2383
  br label %._crit_edge244, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2384

1837:                                             ; preds = %._crit_edge244
; BB176 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2386
  %1838 = load i64, i64* %120, align 8		; visa id: 2386
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2387
  %1839 = ashr i64 %1838, 32		; visa id: 2387
  %1840 = bitcast i64 %1839 to <2 x i32>		; visa id: 2388
  %1841 = extractelement <2 x i32> %1840, i32 0		; visa id: 2392
  %1842 = extractelement <2 x i32> %1840, i32 1		; visa id: 2392
  %1843 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1841, i32 %1842, i32 %44, i32 %45)
  %1844 = extractvalue { i32, i32 } %1843, 0		; visa id: 2392
  %1845 = extractvalue { i32, i32 } %1843, 1		; visa id: 2392
  %1846 = insertelement <2 x i32> undef, i32 %1844, i32 0		; visa id: 2399
  %1847 = insertelement <2 x i32> %1846, i32 %1845, i32 1		; visa id: 2400
  %1848 = bitcast <2 x i32> %1847 to i64		; visa id: 2401
  %1849 = bitcast i64 %1838 to <2 x i32>		; visa id: 2405
  %1850 = extractelement <2 x i32> %1849, i32 0		; visa id: 2407
  %1851 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1850, i32 1
  %1852 = bitcast <2 x i32> %1851 to i64		; visa id: 2407
  %1853 = shl i64 %1848, 1		; visa id: 2408
  %1854 = add i64 %.in400, %1853		; visa id: 2409
  %1855 = ashr exact i64 %1852, 31		; visa id: 2410
  %1856 = add i64 %1854, %1855		; visa id: 2411
  %1857 = inttoptr i64 %1856 to i16 addrspace(4)*		; visa id: 2412
  %1858 = addrspacecast i16 addrspace(4)* %1857 to i16 addrspace(1)*		; visa id: 2412
  %1859 = load i16, i16 addrspace(1)* %1858, align 2		; visa id: 2413
  %1860 = zext i16 %1823 to i32		; visa id: 2415
  %1861 = shl nuw i32 %1860, 16, !spirv.Decorations !639		; visa id: 2416
  %1862 = bitcast i32 %1861 to float
  %1863 = zext i16 %1859 to i32		; visa id: 2417
  %1864 = shl nuw i32 %1863, 16, !spirv.Decorations !639		; visa id: 2418
  %1865 = bitcast i32 %1864 to float
  %1866 = fmul reassoc nsz arcp contract float %1862, %1865, !spirv.Decorations !618
  %1867 = fadd reassoc nsz arcp contract float %1866, %.sroa.146.1, !spirv.Decorations !618		; visa id: 2419
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2420

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %1837
; BB177 :
  %.sroa.146.2 = phi float [ %1867, %1837 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  %1868 = icmp slt i32 %407, %const_reg_dword
  %1869 = icmp slt i32 %1594, %const_reg_dword1		; visa id: 2421
  %1870 = and i1 %1868, %1869		; visa id: 2422
  br i1 %1870, label %1871, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2424

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
; BB:
  br label %.preheader.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1871:                                             ; preds = %._crit_edge.2.4
; BB179 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2426
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2426
  %1872 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 2426
  %1873 = insertelement <2 x i32> %1872, i32 %113, i64 1		; visa id: 2427
  %1874 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2428
  store <2 x i32> %1873, <2 x i32>* %1874, align 4, !noalias !625		; visa id: 2428
  br label %._crit_edge245, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2430

._crit_edge245:                                   ; preds = %._crit_edge245.._crit_edge245_crit_edge, %1871
; BB180 :
  %1875 = phi i32 [ 0, %1871 ], [ %1884, %._crit_edge245.._crit_edge245_crit_edge ]
  %1876 = zext i32 %1875 to i64		; visa id: 2431
  %1877 = shl nuw nsw i64 %1876, 2		; visa id: 2432
  %1878 = add i64 %133, %1877		; visa id: 2433
  %1879 = inttoptr i64 %1878 to i32*		; visa id: 2434
  %1880 = load i32, i32* %1879, align 4, !noalias !625		; visa id: 2434
  %1881 = add i64 %128, %1877		; visa id: 2435
  %1882 = inttoptr i64 %1881 to i32*		; visa id: 2436
  store i32 %1880, i32* %1882, align 4, !alias.scope !625		; visa id: 2436
  %1883 = icmp eq i32 %1875, 0		; visa id: 2437
  br i1 %1883, label %._crit_edge245.._crit_edge245_crit_edge, label %1885, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2438

._crit_edge245.._crit_edge245_crit_edge:          ; preds = %._crit_edge245
; BB181 :
  %1884 = add nuw nsw i32 %1875, 1, !spirv.Decorations !631		; visa id: 2440
  br label %._crit_edge245, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2441

1885:                                             ; preds = %._crit_edge245
; BB182 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2443
  %1886 = load i64, i64* %129, align 8		; visa id: 2443
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2444
  %1887 = bitcast i64 %1886 to <2 x i32>		; visa id: 2444
  %1888 = extractelement <2 x i32> %1887, i32 0		; visa id: 2446
  %1889 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1888, i32 1
  %1890 = bitcast <2 x i32> %1889 to i64		; visa id: 2446
  %1891 = ashr exact i64 %1890, 32		; visa id: 2447
  %1892 = bitcast i64 %1891 to <2 x i32>		; visa id: 2448
  %1893 = extractelement <2 x i32> %1892, i32 0		; visa id: 2452
  %1894 = extractelement <2 x i32> %1892, i32 1		; visa id: 2452
  %1895 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1893, i32 %1894, i32 %41, i32 %42)
  %1896 = extractvalue { i32, i32 } %1895, 0		; visa id: 2452
  %1897 = extractvalue { i32, i32 } %1895, 1		; visa id: 2452
  %1898 = insertelement <2 x i32> undef, i32 %1896, i32 0		; visa id: 2459
  %1899 = insertelement <2 x i32> %1898, i32 %1897, i32 1		; visa id: 2460
  %1900 = bitcast <2 x i32> %1899 to i64		; visa id: 2461
  %1901 = shl i64 %1900, 1		; visa id: 2465
  %1902 = add i64 %.in401, %1901		; visa id: 2466
  %1903 = ashr i64 %1886, 31		; visa id: 2467
  %1904 = bitcast i64 %1903 to <2 x i32>		; visa id: 2468
  %1905 = extractelement <2 x i32> %1904, i32 0		; visa id: 2472
  %1906 = extractelement <2 x i32> %1904, i32 1		; visa id: 2472
  %1907 = and i32 %1905, -2		; visa id: 2472
  %1908 = insertelement <2 x i32> undef, i32 %1907, i32 0		; visa id: 2473
  %1909 = insertelement <2 x i32> %1908, i32 %1906, i32 1		; visa id: 2474
  %1910 = bitcast <2 x i32> %1909 to i64		; visa id: 2475
  %1911 = add i64 %1902, %1910		; visa id: 2479
  %1912 = inttoptr i64 %1911 to i16 addrspace(4)*		; visa id: 2480
  %1913 = addrspacecast i16 addrspace(4)* %1912 to i16 addrspace(1)*		; visa id: 2480
  %1914 = load i16, i16 addrspace(1)* %1913, align 2		; visa id: 2481
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2483
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2483
  %1915 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2483
  %1916 = insertelement <2 x i32> %1915, i32 %1594, i64 1		; visa id: 2484
  %1917 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2485
  store <2 x i32> %1916, <2 x i32>* %1917, align 4, !noalias !635		; visa id: 2485
  br label %._crit_edge246, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2487

._crit_edge246:                                   ; preds = %._crit_edge246.._crit_edge246_crit_edge, %1885
; BB183 :
  %1918 = phi i32 [ 0, %1885 ], [ %1927, %._crit_edge246.._crit_edge246_crit_edge ]
  %1919 = zext i32 %1918 to i64		; visa id: 2488
  %1920 = shl nuw nsw i64 %1919, 2		; visa id: 2489
  %1921 = add i64 %124, %1920		; visa id: 2490
  %1922 = inttoptr i64 %1921 to i32*		; visa id: 2491
  %1923 = load i32, i32* %1922, align 4, !noalias !635		; visa id: 2491
  %1924 = add i64 %119, %1920		; visa id: 2492
  %1925 = inttoptr i64 %1924 to i32*		; visa id: 2493
  store i32 %1923, i32* %1925, align 4, !alias.scope !635		; visa id: 2493
  %1926 = icmp eq i32 %1918, 0		; visa id: 2494
  br i1 %1926, label %._crit_edge246.._crit_edge246_crit_edge, label %1928, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2495

._crit_edge246.._crit_edge246_crit_edge:          ; preds = %._crit_edge246
; BB184 :
  %1927 = add nuw nsw i32 %1918, 1, !spirv.Decorations !631		; visa id: 2497
  br label %._crit_edge246, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2498

1928:                                             ; preds = %._crit_edge246
; BB185 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2500
  %1929 = load i64, i64* %120, align 8		; visa id: 2500
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2501
  %1930 = ashr i64 %1929, 32		; visa id: 2501
  %1931 = bitcast i64 %1930 to <2 x i32>		; visa id: 2502
  %1932 = extractelement <2 x i32> %1931, i32 0		; visa id: 2506
  %1933 = extractelement <2 x i32> %1931, i32 1		; visa id: 2506
  %1934 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1932, i32 %1933, i32 %44, i32 %45)
  %1935 = extractvalue { i32, i32 } %1934, 0		; visa id: 2506
  %1936 = extractvalue { i32, i32 } %1934, 1		; visa id: 2506
  %1937 = insertelement <2 x i32> undef, i32 %1935, i32 0		; visa id: 2513
  %1938 = insertelement <2 x i32> %1937, i32 %1936, i32 1		; visa id: 2514
  %1939 = bitcast <2 x i32> %1938 to i64		; visa id: 2515
  %1940 = bitcast i64 %1929 to <2 x i32>		; visa id: 2519
  %1941 = extractelement <2 x i32> %1940, i32 0		; visa id: 2521
  %1942 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1941, i32 1
  %1943 = bitcast <2 x i32> %1942 to i64		; visa id: 2521
  %1944 = shl i64 %1939, 1		; visa id: 2522
  %1945 = add i64 %.in400, %1944		; visa id: 2523
  %1946 = ashr exact i64 %1943, 31		; visa id: 2524
  %1947 = add i64 %1945, %1946		; visa id: 2525
  %1948 = inttoptr i64 %1947 to i16 addrspace(4)*		; visa id: 2526
  %1949 = addrspacecast i16 addrspace(4)* %1948 to i16 addrspace(1)*		; visa id: 2526
  %1950 = load i16, i16 addrspace(1)* %1949, align 2		; visa id: 2527
  %1951 = zext i16 %1914 to i32		; visa id: 2529
  %1952 = shl nuw i32 %1951, 16, !spirv.Decorations !639		; visa id: 2530
  %1953 = bitcast i32 %1952 to float
  %1954 = zext i16 %1950 to i32		; visa id: 2531
  %1955 = shl nuw i32 %1954, 16, !spirv.Decorations !639		; visa id: 2532
  %1956 = bitcast i32 %1955 to float
  %1957 = fmul reassoc nsz arcp contract float %1953, %1956, !spirv.Decorations !618
  %1958 = fadd reassoc nsz arcp contract float %1957, %.sroa.210.1, !spirv.Decorations !618		; visa id: 2533
  br label %.preheader.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2534

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %1928
; BB186 :
  %.sroa.210.2 = phi float [ %1958, %1928 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %1959 = add i32 %69, 5		; visa id: 2535
  %1960 = icmp slt i32 %1959, %const_reg_dword1		; visa id: 2536
  %1961 = icmp slt i32 %65, %const_reg_dword
  %1962 = and i1 %1961, %1960		; visa id: 2537
  br i1 %1962, label %1963, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2539

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
; BB:
  br label %._crit_edge.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1963:                                             ; preds = %.preheader.4
; BB188 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2541
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2541
  %1964 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 2541
  %1965 = insertelement <2 x i32> %1964, i32 %113, i64 1		; visa id: 2542
  %1966 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2543
  store <2 x i32> %1965, <2 x i32>* %1966, align 4, !noalias !625		; visa id: 2543
  br label %._crit_edge247, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2545

._crit_edge247:                                   ; preds = %._crit_edge247.._crit_edge247_crit_edge, %1963
; BB189 :
  %1967 = phi i32 [ 0, %1963 ], [ %1976, %._crit_edge247.._crit_edge247_crit_edge ]
  %1968 = zext i32 %1967 to i64		; visa id: 2546
  %1969 = shl nuw nsw i64 %1968, 2		; visa id: 2547
  %1970 = add i64 %133, %1969		; visa id: 2548
  %1971 = inttoptr i64 %1970 to i32*		; visa id: 2549
  %1972 = load i32, i32* %1971, align 4, !noalias !625		; visa id: 2549
  %1973 = add i64 %128, %1969		; visa id: 2550
  %1974 = inttoptr i64 %1973 to i32*		; visa id: 2551
  store i32 %1972, i32* %1974, align 4, !alias.scope !625		; visa id: 2551
  %1975 = icmp eq i32 %1967, 0		; visa id: 2552
  br i1 %1975, label %._crit_edge247.._crit_edge247_crit_edge, label %1977, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2553

._crit_edge247.._crit_edge247_crit_edge:          ; preds = %._crit_edge247
; BB190 :
  %1976 = add nuw nsw i32 %1967, 1, !spirv.Decorations !631		; visa id: 2555
  br label %._crit_edge247, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2556

1977:                                             ; preds = %._crit_edge247
; BB191 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2558
  %1978 = load i64, i64* %129, align 8		; visa id: 2558
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2559
  %1979 = bitcast i64 %1978 to <2 x i32>		; visa id: 2559
  %1980 = extractelement <2 x i32> %1979, i32 0		; visa id: 2561
  %1981 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1980, i32 1
  %1982 = bitcast <2 x i32> %1981 to i64		; visa id: 2561
  %1983 = ashr exact i64 %1982, 32		; visa id: 2562
  %1984 = bitcast i64 %1983 to <2 x i32>		; visa id: 2563
  %1985 = extractelement <2 x i32> %1984, i32 0		; visa id: 2567
  %1986 = extractelement <2 x i32> %1984, i32 1		; visa id: 2567
  %1987 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1985, i32 %1986, i32 %41, i32 %42)
  %1988 = extractvalue { i32, i32 } %1987, 0		; visa id: 2567
  %1989 = extractvalue { i32, i32 } %1987, 1		; visa id: 2567
  %1990 = insertelement <2 x i32> undef, i32 %1988, i32 0		; visa id: 2574
  %1991 = insertelement <2 x i32> %1990, i32 %1989, i32 1		; visa id: 2575
  %1992 = bitcast <2 x i32> %1991 to i64		; visa id: 2576
  %1993 = shl i64 %1992, 1		; visa id: 2580
  %1994 = add i64 %.in401, %1993		; visa id: 2581
  %1995 = ashr i64 %1978, 31		; visa id: 2582
  %1996 = bitcast i64 %1995 to <2 x i32>		; visa id: 2583
  %1997 = extractelement <2 x i32> %1996, i32 0		; visa id: 2587
  %1998 = extractelement <2 x i32> %1996, i32 1		; visa id: 2587
  %1999 = and i32 %1997, -2		; visa id: 2587
  %2000 = insertelement <2 x i32> undef, i32 %1999, i32 0		; visa id: 2588
  %2001 = insertelement <2 x i32> %2000, i32 %1998, i32 1		; visa id: 2589
  %2002 = bitcast <2 x i32> %2001 to i64		; visa id: 2590
  %2003 = add i64 %1994, %2002		; visa id: 2594
  %2004 = inttoptr i64 %2003 to i16 addrspace(4)*		; visa id: 2595
  %2005 = addrspacecast i16 addrspace(4)* %2004 to i16 addrspace(1)*		; visa id: 2595
  %2006 = load i16, i16 addrspace(1)* %2005, align 2		; visa id: 2596
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2598
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2598
  %2007 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2598
  %2008 = insertelement <2 x i32> %2007, i32 %1959, i64 1		; visa id: 2599
  %2009 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2600
  store <2 x i32> %2008, <2 x i32>* %2009, align 4, !noalias !635		; visa id: 2600
  br label %._crit_edge248, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2602

._crit_edge248:                                   ; preds = %._crit_edge248.._crit_edge248_crit_edge, %1977
; BB192 :
  %2010 = phi i32 [ 0, %1977 ], [ %2019, %._crit_edge248.._crit_edge248_crit_edge ]
  %2011 = zext i32 %2010 to i64		; visa id: 2603
  %2012 = shl nuw nsw i64 %2011, 2		; visa id: 2604
  %2013 = add i64 %124, %2012		; visa id: 2605
  %2014 = inttoptr i64 %2013 to i32*		; visa id: 2606
  %2015 = load i32, i32* %2014, align 4, !noalias !635		; visa id: 2606
  %2016 = add i64 %119, %2012		; visa id: 2607
  %2017 = inttoptr i64 %2016 to i32*		; visa id: 2608
  store i32 %2015, i32* %2017, align 4, !alias.scope !635		; visa id: 2608
  %2018 = icmp eq i32 %2010, 0		; visa id: 2609
  br i1 %2018, label %._crit_edge248.._crit_edge248_crit_edge, label %2020, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2610

._crit_edge248.._crit_edge248_crit_edge:          ; preds = %._crit_edge248
; BB193 :
  %2019 = add nuw nsw i32 %2010, 1, !spirv.Decorations !631		; visa id: 2612
  br label %._crit_edge248, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2613

2020:                                             ; preds = %._crit_edge248
; BB194 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2615
  %2021 = load i64, i64* %120, align 8		; visa id: 2615
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2616
  %2022 = ashr i64 %2021, 32		; visa id: 2616
  %2023 = bitcast i64 %2022 to <2 x i32>		; visa id: 2617
  %2024 = extractelement <2 x i32> %2023, i32 0		; visa id: 2621
  %2025 = extractelement <2 x i32> %2023, i32 1		; visa id: 2621
  %2026 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2024, i32 %2025, i32 %44, i32 %45)
  %2027 = extractvalue { i32, i32 } %2026, 0		; visa id: 2621
  %2028 = extractvalue { i32, i32 } %2026, 1		; visa id: 2621
  %2029 = insertelement <2 x i32> undef, i32 %2027, i32 0		; visa id: 2628
  %2030 = insertelement <2 x i32> %2029, i32 %2028, i32 1		; visa id: 2629
  %2031 = bitcast <2 x i32> %2030 to i64		; visa id: 2630
  %2032 = bitcast i64 %2021 to <2 x i32>		; visa id: 2634
  %2033 = extractelement <2 x i32> %2032, i32 0		; visa id: 2636
  %2034 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2033, i32 1
  %2035 = bitcast <2 x i32> %2034 to i64		; visa id: 2636
  %2036 = shl i64 %2031, 1		; visa id: 2637
  %2037 = add i64 %.in400, %2036		; visa id: 2638
  %2038 = ashr exact i64 %2035, 31		; visa id: 2639
  %2039 = add i64 %2037, %2038		; visa id: 2640
  %2040 = inttoptr i64 %2039 to i16 addrspace(4)*		; visa id: 2641
  %2041 = addrspacecast i16 addrspace(4)* %2040 to i16 addrspace(1)*		; visa id: 2641
  %2042 = load i16, i16 addrspace(1)* %2041, align 2		; visa id: 2642
  %2043 = zext i16 %2006 to i32		; visa id: 2644
  %2044 = shl nuw i32 %2043, 16, !spirv.Decorations !639		; visa id: 2645
  %2045 = bitcast i32 %2044 to float
  %2046 = zext i16 %2042 to i32		; visa id: 2646
  %2047 = shl nuw i32 %2046, 16, !spirv.Decorations !639		; visa id: 2647
  %2048 = bitcast i32 %2047 to float
  %2049 = fmul reassoc nsz arcp contract float %2045, %2048, !spirv.Decorations !618
  %2050 = fadd reassoc nsz arcp contract float %2049, %.sroa.22.1, !spirv.Decorations !618		; visa id: 2648
  br label %._crit_edge.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2649

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %2020
; BB195 :
  %.sroa.22.2 = phi float [ %2050, %2020 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  %2051 = icmp slt i32 %223, %const_reg_dword
  %2052 = icmp slt i32 %1959, %const_reg_dword1		; visa id: 2650
  %2053 = and i1 %2051, %2052		; visa id: 2651
  br i1 %2053, label %2054, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2653

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
; BB:
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2054:                                             ; preds = %._crit_edge.5
; BB197 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2655
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2655
  %2055 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 2655
  %2056 = insertelement <2 x i32> %2055, i32 %113, i64 1		; visa id: 2656
  %2057 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2657
  store <2 x i32> %2056, <2 x i32>* %2057, align 4, !noalias !625		; visa id: 2657
  br label %._crit_edge249, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2659

._crit_edge249:                                   ; preds = %._crit_edge249.._crit_edge249_crit_edge, %2054
; BB198 :
  %2058 = phi i32 [ 0, %2054 ], [ %2067, %._crit_edge249.._crit_edge249_crit_edge ]
  %2059 = zext i32 %2058 to i64		; visa id: 2660
  %2060 = shl nuw nsw i64 %2059, 2		; visa id: 2661
  %2061 = add i64 %133, %2060		; visa id: 2662
  %2062 = inttoptr i64 %2061 to i32*		; visa id: 2663
  %2063 = load i32, i32* %2062, align 4, !noalias !625		; visa id: 2663
  %2064 = add i64 %128, %2060		; visa id: 2664
  %2065 = inttoptr i64 %2064 to i32*		; visa id: 2665
  store i32 %2063, i32* %2065, align 4, !alias.scope !625		; visa id: 2665
  %2066 = icmp eq i32 %2058, 0		; visa id: 2666
  br i1 %2066, label %._crit_edge249.._crit_edge249_crit_edge, label %2068, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2667

._crit_edge249.._crit_edge249_crit_edge:          ; preds = %._crit_edge249
; BB199 :
  %2067 = add nuw nsw i32 %2058, 1, !spirv.Decorations !631		; visa id: 2669
  br label %._crit_edge249, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2670

2068:                                             ; preds = %._crit_edge249
; BB200 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2672
  %2069 = load i64, i64* %129, align 8		; visa id: 2672
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2673
  %2070 = bitcast i64 %2069 to <2 x i32>		; visa id: 2673
  %2071 = extractelement <2 x i32> %2070, i32 0		; visa id: 2675
  %2072 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2071, i32 1
  %2073 = bitcast <2 x i32> %2072 to i64		; visa id: 2675
  %2074 = ashr exact i64 %2073, 32		; visa id: 2676
  %2075 = bitcast i64 %2074 to <2 x i32>		; visa id: 2677
  %2076 = extractelement <2 x i32> %2075, i32 0		; visa id: 2681
  %2077 = extractelement <2 x i32> %2075, i32 1		; visa id: 2681
  %2078 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2076, i32 %2077, i32 %41, i32 %42)
  %2079 = extractvalue { i32, i32 } %2078, 0		; visa id: 2681
  %2080 = extractvalue { i32, i32 } %2078, 1		; visa id: 2681
  %2081 = insertelement <2 x i32> undef, i32 %2079, i32 0		; visa id: 2688
  %2082 = insertelement <2 x i32> %2081, i32 %2080, i32 1		; visa id: 2689
  %2083 = bitcast <2 x i32> %2082 to i64		; visa id: 2690
  %2084 = shl i64 %2083, 1		; visa id: 2694
  %2085 = add i64 %.in401, %2084		; visa id: 2695
  %2086 = ashr i64 %2069, 31		; visa id: 2696
  %2087 = bitcast i64 %2086 to <2 x i32>		; visa id: 2697
  %2088 = extractelement <2 x i32> %2087, i32 0		; visa id: 2701
  %2089 = extractelement <2 x i32> %2087, i32 1		; visa id: 2701
  %2090 = and i32 %2088, -2		; visa id: 2701
  %2091 = insertelement <2 x i32> undef, i32 %2090, i32 0		; visa id: 2702
  %2092 = insertelement <2 x i32> %2091, i32 %2089, i32 1		; visa id: 2703
  %2093 = bitcast <2 x i32> %2092 to i64		; visa id: 2704
  %2094 = add i64 %2085, %2093		; visa id: 2708
  %2095 = inttoptr i64 %2094 to i16 addrspace(4)*		; visa id: 2709
  %2096 = addrspacecast i16 addrspace(4)* %2095 to i16 addrspace(1)*		; visa id: 2709
  %2097 = load i16, i16 addrspace(1)* %2096, align 2		; visa id: 2710
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2712
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2712
  %2098 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2712
  %2099 = insertelement <2 x i32> %2098, i32 %1959, i64 1		; visa id: 2713
  %2100 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2714
  store <2 x i32> %2099, <2 x i32>* %2100, align 4, !noalias !635		; visa id: 2714
  br label %._crit_edge250, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2716

._crit_edge250:                                   ; preds = %._crit_edge250.._crit_edge250_crit_edge, %2068
; BB201 :
  %2101 = phi i32 [ 0, %2068 ], [ %2110, %._crit_edge250.._crit_edge250_crit_edge ]
  %2102 = zext i32 %2101 to i64		; visa id: 2717
  %2103 = shl nuw nsw i64 %2102, 2		; visa id: 2718
  %2104 = add i64 %124, %2103		; visa id: 2719
  %2105 = inttoptr i64 %2104 to i32*		; visa id: 2720
  %2106 = load i32, i32* %2105, align 4, !noalias !635		; visa id: 2720
  %2107 = add i64 %119, %2103		; visa id: 2721
  %2108 = inttoptr i64 %2107 to i32*		; visa id: 2722
  store i32 %2106, i32* %2108, align 4, !alias.scope !635		; visa id: 2722
  %2109 = icmp eq i32 %2101, 0		; visa id: 2723
  br i1 %2109, label %._crit_edge250.._crit_edge250_crit_edge, label %2111, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2724

._crit_edge250.._crit_edge250_crit_edge:          ; preds = %._crit_edge250
; BB202 :
  %2110 = add nuw nsw i32 %2101, 1, !spirv.Decorations !631		; visa id: 2726
  br label %._crit_edge250, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2727

2111:                                             ; preds = %._crit_edge250
; BB203 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2729
  %2112 = load i64, i64* %120, align 8		; visa id: 2729
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2730
  %2113 = ashr i64 %2112, 32		; visa id: 2730
  %2114 = bitcast i64 %2113 to <2 x i32>		; visa id: 2731
  %2115 = extractelement <2 x i32> %2114, i32 0		; visa id: 2735
  %2116 = extractelement <2 x i32> %2114, i32 1		; visa id: 2735
  %2117 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2115, i32 %2116, i32 %44, i32 %45)
  %2118 = extractvalue { i32, i32 } %2117, 0		; visa id: 2735
  %2119 = extractvalue { i32, i32 } %2117, 1		; visa id: 2735
  %2120 = insertelement <2 x i32> undef, i32 %2118, i32 0		; visa id: 2742
  %2121 = insertelement <2 x i32> %2120, i32 %2119, i32 1		; visa id: 2743
  %2122 = bitcast <2 x i32> %2121 to i64		; visa id: 2744
  %2123 = bitcast i64 %2112 to <2 x i32>		; visa id: 2748
  %2124 = extractelement <2 x i32> %2123, i32 0		; visa id: 2750
  %2125 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2124, i32 1
  %2126 = bitcast <2 x i32> %2125 to i64		; visa id: 2750
  %2127 = shl i64 %2122, 1		; visa id: 2751
  %2128 = add i64 %.in400, %2127		; visa id: 2752
  %2129 = ashr exact i64 %2126, 31		; visa id: 2753
  %2130 = add i64 %2128, %2129		; visa id: 2754
  %2131 = inttoptr i64 %2130 to i16 addrspace(4)*		; visa id: 2755
  %2132 = addrspacecast i16 addrspace(4)* %2131 to i16 addrspace(1)*		; visa id: 2755
  %2133 = load i16, i16 addrspace(1)* %2132, align 2		; visa id: 2756
  %2134 = zext i16 %2097 to i32		; visa id: 2758
  %2135 = shl nuw i32 %2134, 16, !spirv.Decorations !639		; visa id: 2759
  %2136 = bitcast i32 %2135 to float
  %2137 = zext i16 %2133 to i32		; visa id: 2760
  %2138 = shl nuw i32 %2137, 16, !spirv.Decorations !639		; visa id: 2761
  %2139 = bitcast i32 %2138 to float
  %2140 = fmul reassoc nsz arcp contract float %2136, %2139, !spirv.Decorations !618
  %2141 = fadd reassoc nsz arcp contract float %2140, %.sroa.86.1, !spirv.Decorations !618		; visa id: 2762
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2763

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %2111
; BB204 :
  %.sroa.86.2 = phi float [ %2141, %2111 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  %2142 = icmp slt i32 %315, %const_reg_dword
  %2143 = icmp slt i32 %1959, %const_reg_dword1		; visa id: 2764
  %2144 = and i1 %2142, %2143		; visa id: 2765
  br i1 %2144, label %2145, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2767

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
; BB:
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2145:                                             ; preds = %._crit_edge.1.5
; BB206 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2769
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2769
  %2146 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 2769
  %2147 = insertelement <2 x i32> %2146, i32 %113, i64 1		; visa id: 2770
  %2148 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2771
  store <2 x i32> %2147, <2 x i32>* %2148, align 4, !noalias !625		; visa id: 2771
  br label %._crit_edge251, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2773

._crit_edge251:                                   ; preds = %._crit_edge251.._crit_edge251_crit_edge, %2145
; BB207 :
  %2149 = phi i32 [ 0, %2145 ], [ %2158, %._crit_edge251.._crit_edge251_crit_edge ]
  %2150 = zext i32 %2149 to i64		; visa id: 2774
  %2151 = shl nuw nsw i64 %2150, 2		; visa id: 2775
  %2152 = add i64 %133, %2151		; visa id: 2776
  %2153 = inttoptr i64 %2152 to i32*		; visa id: 2777
  %2154 = load i32, i32* %2153, align 4, !noalias !625		; visa id: 2777
  %2155 = add i64 %128, %2151		; visa id: 2778
  %2156 = inttoptr i64 %2155 to i32*		; visa id: 2779
  store i32 %2154, i32* %2156, align 4, !alias.scope !625		; visa id: 2779
  %2157 = icmp eq i32 %2149, 0		; visa id: 2780
  br i1 %2157, label %._crit_edge251.._crit_edge251_crit_edge, label %2159, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2781

._crit_edge251.._crit_edge251_crit_edge:          ; preds = %._crit_edge251
; BB208 :
  %2158 = add nuw nsw i32 %2149, 1, !spirv.Decorations !631		; visa id: 2783
  br label %._crit_edge251, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2784

2159:                                             ; preds = %._crit_edge251
; BB209 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2786
  %2160 = load i64, i64* %129, align 8		; visa id: 2786
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2787
  %2161 = bitcast i64 %2160 to <2 x i32>		; visa id: 2787
  %2162 = extractelement <2 x i32> %2161, i32 0		; visa id: 2789
  %2163 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2162, i32 1
  %2164 = bitcast <2 x i32> %2163 to i64		; visa id: 2789
  %2165 = ashr exact i64 %2164, 32		; visa id: 2790
  %2166 = bitcast i64 %2165 to <2 x i32>		; visa id: 2791
  %2167 = extractelement <2 x i32> %2166, i32 0		; visa id: 2795
  %2168 = extractelement <2 x i32> %2166, i32 1		; visa id: 2795
  %2169 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2167, i32 %2168, i32 %41, i32 %42)
  %2170 = extractvalue { i32, i32 } %2169, 0		; visa id: 2795
  %2171 = extractvalue { i32, i32 } %2169, 1		; visa id: 2795
  %2172 = insertelement <2 x i32> undef, i32 %2170, i32 0		; visa id: 2802
  %2173 = insertelement <2 x i32> %2172, i32 %2171, i32 1		; visa id: 2803
  %2174 = bitcast <2 x i32> %2173 to i64		; visa id: 2804
  %2175 = shl i64 %2174, 1		; visa id: 2808
  %2176 = add i64 %.in401, %2175		; visa id: 2809
  %2177 = ashr i64 %2160, 31		; visa id: 2810
  %2178 = bitcast i64 %2177 to <2 x i32>		; visa id: 2811
  %2179 = extractelement <2 x i32> %2178, i32 0		; visa id: 2815
  %2180 = extractelement <2 x i32> %2178, i32 1		; visa id: 2815
  %2181 = and i32 %2179, -2		; visa id: 2815
  %2182 = insertelement <2 x i32> undef, i32 %2181, i32 0		; visa id: 2816
  %2183 = insertelement <2 x i32> %2182, i32 %2180, i32 1		; visa id: 2817
  %2184 = bitcast <2 x i32> %2183 to i64		; visa id: 2818
  %2185 = add i64 %2176, %2184		; visa id: 2822
  %2186 = inttoptr i64 %2185 to i16 addrspace(4)*		; visa id: 2823
  %2187 = addrspacecast i16 addrspace(4)* %2186 to i16 addrspace(1)*		; visa id: 2823
  %2188 = load i16, i16 addrspace(1)* %2187, align 2		; visa id: 2824
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2826
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2826
  %2189 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2826
  %2190 = insertelement <2 x i32> %2189, i32 %1959, i64 1		; visa id: 2827
  %2191 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2828
  store <2 x i32> %2190, <2 x i32>* %2191, align 4, !noalias !635		; visa id: 2828
  br label %._crit_edge252, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2830

._crit_edge252:                                   ; preds = %._crit_edge252.._crit_edge252_crit_edge, %2159
; BB210 :
  %2192 = phi i32 [ 0, %2159 ], [ %2201, %._crit_edge252.._crit_edge252_crit_edge ]
  %2193 = zext i32 %2192 to i64		; visa id: 2831
  %2194 = shl nuw nsw i64 %2193, 2		; visa id: 2832
  %2195 = add i64 %124, %2194		; visa id: 2833
  %2196 = inttoptr i64 %2195 to i32*		; visa id: 2834
  %2197 = load i32, i32* %2196, align 4, !noalias !635		; visa id: 2834
  %2198 = add i64 %119, %2194		; visa id: 2835
  %2199 = inttoptr i64 %2198 to i32*		; visa id: 2836
  store i32 %2197, i32* %2199, align 4, !alias.scope !635		; visa id: 2836
  %2200 = icmp eq i32 %2192, 0		; visa id: 2837
  br i1 %2200, label %._crit_edge252.._crit_edge252_crit_edge, label %2202, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2838

._crit_edge252.._crit_edge252_crit_edge:          ; preds = %._crit_edge252
; BB211 :
  %2201 = add nuw nsw i32 %2192, 1, !spirv.Decorations !631		; visa id: 2840
  br label %._crit_edge252, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2841

2202:                                             ; preds = %._crit_edge252
; BB212 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2843
  %2203 = load i64, i64* %120, align 8		; visa id: 2843
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2844
  %2204 = ashr i64 %2203, 32		; visa id: 2844
  %2205 = bitcast i64 %2204 to <2 x i32>		; visa id: 2845
  %2206 = extractelement <2 x i32> %2205, i32 0		; visa id: 2849
  %2207 = extractelement <2 x i32> %2205, i32 1		; visa id: 2849
  %2208 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2206, i32 %2207, i32 %44, i32 %45)
  %2209 = extractvalue { i32, i32 } %2208, 0		; visa id: 2849
  %2210 = extractvalue { i32, i32 } %2208, 1		; visa id: 2849
  %2211 = insertelement <2 x i32> undef, i32 %2209, i32 0		; visa id: 2856
  %2212 = insertelement <2 x i32> %2211, i32 %2210, i32 1		; visa id: 2857
  %2213 = bitcast <2 x i32> %2212 to i64		; visa id: 2858
  %2214 = bitcast i64 %2203 to <2 x i32>		; visa id: 2862
  %2215 = extractelement <2 x i32> %2214, i32 0		; visa id: 2864
  %2216 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2215, i32 1
  %2217 = bitcast <2 x i32> %2216 to i64		; visa id: 2864
  %2218 = shl i64 %2213, 1		; visa id: 2865
  %2219 = add i64 %.in400, %2218		; visa id: 2866
  %2220 = ashr exact i64 %2217, 31		; visa id: 2867
  %2221 = add i64 %2219, %2220		; visa id: 2868
  %2222 = inttoptr i64 %2221 to i16 addrspace(4)*		; visa id: 2869
  %2223 = addrspacecast i16 addrspace(4)* %2222 to i16 addrspace(1)*		; visa id: 2869
  %2224 = load i16, i16 addrspace(1)* %2223, align 2		; visa id: 2870
  %2225 = zext i16 %2188 to i32		; visa id: 2872
  %2226 = shl nuw i32 %2225, 16, !spirv.Decorations !639		; visa id: 2873
  %2227 = bitcast i32 %2226 to float
  %2228 = zext i16 %2224 to i32		; visa id: 2874
  %2229 = shl nuw i32 %2228, 16, !spirv.Decorations !639		; visa id: 2875
  %2230 = bitcast i32 %2229 to float
  %2231 = fmul reassoc nsz arcp contract float %2227, %2230, !spirv.Decorations !618
  %2232 = fadd reassoc nsz arcp contract float %2231, %.sroa.150.1, !spirv.Decorations !618		; visa id: 2876
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2877

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %2202
; BB213 :
  %.sroa.150.2 = phi float [ %2232, %2202 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  %2233 = icmp slt i32 %407, %const_reg_dword
  %2234 = icmp slt i32 %1959, %const_reg_dword1		; visa id: 2878
  %2235 = and i1 %2233, %2234		; visa id: 2879
  br i1 %2235, label %2236, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2881

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
; BB:
  br label %.preheader.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2236:                                             ; preds = %._crit_edge.2.5
; BB215 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2883
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2883
  %2237 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 2883
  %2238 = insertelement <2 x i32> %2237, i32 %113, i64 1		; visa id: 2884
  %2239 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2885
  store <2 x i32> %2238, <2 x i32>* %2239, align 4, !noalias !625		; visa id: 2885
  br label %._crit_edge253, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2887

._crit_edge253:                                   ; preds = %._crit_edge253.._crit_edge253_crit_edge, %2236
; BB216 :
  %2240 = phi i32 [ 0, %2236 ], [ %2249, %._crit_edge253.._crit_edge253_crit_edge ]
  %2241 = zext i32 %2240 to i64		; visa id: 2888
  %2242 = shl nuw nsw i64 %2241, 2		; visa id: 2889
  %2243 = add i64 %133, %2242		; visa id: 2890
  %2244 = inttoptr i64 %2243 to i32*		; visa id: 2891
  %2245 = load i32, i32* %2244, align 4, !noalias !625		; visa id: 2891
  %2246 = add i64 %128, %2242		; visa id: 2892
  %2247 = inttoptr i64 %2246 to i32*		; visa id: 2893
  store i32 %2245, i32* %2247, align 4, !alias.scope !625		; visa id: 2893
  %2248 = icmp eq i32 %2240, 0		; visa id: 2894
  br i1 %2248, label %._crit_edge253.._crit_edge253_crit_edge, label %2250, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2895

._crit_edge253.._crit_edge253_crit_edge:          ; preds = %._crit_edge253
; BB217 :
  %2249 = add nuw nsw i32 %2240, 1, !spirv.Decorations !631		; visa id: 2897
  br label %._crit_edge253, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2898

2250:                                             ; preds = %._crit_edge253
; BB218 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2900
  %2251 = load i64, i64* %129, align 8		; visa id: 2900
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2901
  %2252 = bitcast i64 %2251 to <2 x i32>		; visa id: 2901
  %2253 = extractelement <2 x i32> %2252, i32 0		; visa id: 2903
  %2254 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2253, i32 1
  %2255 = bitcast <2 x i32> %2254 to i64		; visa id: 2903
  %2256 = ashr exact i64 %2255, 32		; visa id: 2904
  %2257 = bitcast i64 %2256 to <2 x i32>		; visa id: 2905
  %2258 = extractelement <2 x i32> %2257, i32 0		; visa id: 2909
  %2259 = extractelement <2 x i32> %2257, i32 1		; visa id: 2909
  %2260 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2258, i32 %2259, i32 %41, i32 %42)
  %2261 = extractvalue { i32, i32 } %2260, 0		; visa id: 2909
  %2262 = extractvalue { i32, i32 } %2260, 1		; visa id: 2909
  %2263 = insertelement <2 x i32> undef, i32 %2261, i32 0		; visa id: 2916
  %2264 = insertelement <2 x i32> %2263, i32 %2262, i32 1		; visa id: 2917
  %2265 = bitcast <2 x i32> %2264 to i64		; visa id: 2918
  %2266 = shl i64 %2265, 1		; visa id: 2922
  %2267 = add i64 %.in401, %2266		; visa id: 2923
  %2268 = ashr i64 %2251, 31		; visa id: 2924
  %2269 = bitcast i64 %2268 to <2 x i32>		; visa id: 2925
  %2270 = extractelement <2 x i32> %2269, i32 0		; visa id: 2929
  %2271 = extractelement <2 x i32> %2269, i32 1		; visa id: 2929
  %2272 = and i32 %2270, -2		; visa id: 2929
  %2273 = insertelement <2 x i32> undef, i32 %2272, i32 0		; visa id: 2930
  %2274 = insertelement <2 x i32> %2273, i32 %2271, i32 1		; visa id: 2931
  %2275 = bitcast <2 x i32> %2274 to i64		; visa id: 2932
  %2276 = add i64 %2267, %2275		; visa id: 2936
  %2277 = inttoptr i64 %2276 to i16 addrspace(4)*		; visa id: 2937
  %2278 = addrspacecast i16 addrspace(4)* %2277 to i16 addrspace(1)*		; visa id: 2937
  %2279 = load i16, i16 addrspace(1)* %2278, align 2		; visa id: 2938
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2940
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2940
  %2280 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2940
  %2281 = insertelement <2 x i32> %2280, i32 %1959, i64 1		; visa id: 2941
  %2282 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2942
  store <2 x i32> %2281, <2 x i32>* %2282, align 4, !noalias !635		; visa id: 2942
  br label %._crit_edge254, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2944

._crit_edge254:                                   ; preds = %._crit_edge254.._crit_edge254_crit_edge, %2250
; BB219 :
  %2283 = phi i32 [ 0, %2250 ], [ %2292, %._crit_edge254.._crit_edge254_crit_edge ]
  %2284 = zext i32 %2283 to i64		; visa id: 2945
  %2285 = shl nuw nsw i64 %2284, 2		; visa id: 2946
  %2286 = add i64 %124, %2285		; visa id: 2947
  %2287 = inttoptr i64 %2286 to i32*		; visa id: 2948
  %2288 = load i32, i32* %2287, align 4, !noalias !635		; visa id: 2948
  %2289 = add i64 %119, %2285		; visa id: 2949
  %2290 = inttoptr i64 %2289 to i32*		; visa id: 2950
  store i32 %2288, i32* %2290, align 4, !alias.scope !635		; visa id: 2950
  %2291 = icmp eq i32 %2283, 0		; visa id: 2951
  br i1 %2291, label %._crit_edge254.._crit_edge254_crit_edge, label %2293, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2952

._crit_edge254.._crit_edge254_crit_edge:          ; preds = %._crit_edge254
; BB220 :
  %2292 = add nuw nsw i32 %2283, 1, !spirv.Decorations !631		; visa id: 2954
  br label %._crit_edge254, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2955

2293:                                             ; preds = %._crit_edge254
; BB221 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2957
  %2294 = load i64, i64* %120, align 8		; visa id: 2957
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2958
  %2295 = ashr i64 %2294, 32		; visa id: 2958
  %2296 = bitcast i64 %2295 to <2 x i32>		; visa id: 2959
  %2297 = extractelement <2 x i32> %2296, i32 0		; visa id: 2963
  %2298 = extractelement <2 x i32> %2296, i32 1		; visa id: 2963
  %2299 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2297, i32 %2298, i32 %44, i32 %45)
  %2300 = extractvalue { i32, i32 } %2299, 0		; visa id: 2963
  %2301 = extractvalue { i32, i32 } %2299, 1		; visa id: 2963
  %2302 = insertelement <2 x i32> undef, i32 %2300, i32 0		; visa id: 2970
  %2303 = insertelement <2 x i32> %2302, i32 %2301, i32 1		; visa id: 2971
  %2304 = bitcast <2 x i32> %2303 to i64		; visa id: 2972
  %2305 = bitcast i64 %2294 to <2 x i32>		; visa id: 2976
  %2306 = extractelement <2 x i32> %2305, i32 0		; visa id: 2978
  %2307 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2306, i32 1
  %2308 = bitcast <2 x i32> %2307 to i64		; visa id: 2978
  %2309 = shl i64 %2304, 1		; visa id: 2979
  %2310 = add i64 %.in400, %2309		; visa id: 2980
  %2311 = ashr exact i64 %2308, 31		; visa id: 2981
  %2312 = add i64 %2310, %2311		; visa id: 2982
  %2313 = inttoptr i64 %2312 to i16 addrspace(4)*		; visa id: 2983
  %2314 = addrspacecast i16 addrspace(4)* %2313 to i16 addrspace(1)*		; visa id: 2983
  %2315 = load i16, i16 addrspace(1)* %2314, align 2		; visa id: 2984
  %2316 = zext i16 %2279 to i32		; visa id: 2986
  %2317 = shl nuw i32 %2316, 16, !spirv.Decorations !639		; visa id: 2987
  %2318 = bitcast i32 %2317 to float
  %2319 = zext i16 %2315 to i32		; visa id: 2988
  %2320 = shl nuw i32 %2319, 16, !spirv.Decorations !639		; visa id: 2989
  %2321 = bitcast i32 %2320 to float
  %2322 = fmul reassoc nsz arcp contract float %2318, %2321, !spirv.Decorations !618
  %2323 = fadd reassoc nsz arcp contract float %2322, %.sroa.214.1, !spirv.Decorations !618		; visa id: 2990
  br label %.preheader.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2991

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %2293
; BB222 :
  %.sroa.214.2 = phi float [ %2323, %2293 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %2324 = add i32 %69, 6		; visa id: 2992
  %2325 = icmp slt i32 %2324, %const_reg_dword1		; visa id: 2993
  %2326 = icmp slt i32 %65, %const_reg_dword
  %2327 = and i1 %2326, %2325		; visa id: 2994
  br i1 %2327, label %2328, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2996

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
; BB:
  br label %._crit_edge.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2328:                                             ; preds = %.preheader.5
; BB224 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2998
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2998
  %2329 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 2998
  %2330 = insertelement <2 x i32> %2329, i32 %113, i64 1		; visa id: 2999
  %2331 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3000
  store <2 x i32> %2330, <2 x i32>* %2331, align 4, !noalias !625		; visa id: 3000
  br label %._crit_edge255, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3002

._crit_edge255:                                   ; preds = %._crit_edge255.._crit_edge255_crit_edge, %2328
; BB225 :
  %2332 = phi i32 [ 0, %2328 ], [ %2341, %._crit_edge255.._crit_edge255_crit_edge ]
  %2333 = zext i32 %2332 to i64		; visa id: 3003
  %2334 = shl nuw nsw i64 %2333, 2		; visa id: 3004
  %2335 = add i64 %133, %2334		; visa id: 3005
  %2336 = inttoptr i64 %2335 to i32*		; visa id: 3006
  %2337 = load i32, i32* %2336, align 4, !noalias !625		; visa id: 3006
  %2338 = add i64 %128, %2334		; visa id: 3007
  %2339 = inttoptr i64 %2338 to i32*		; visa id: 3008
  store i32 %2337, i32* %2339, align 4, !alias.scope !625		; visa id: 3008
  %2340 = icmp eq i32 %2332, 0		; visa id: 3009
  br i1 %2340, label %._crit_edge255.._crit_edge255_crit_edge, label %2342, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3010

._crit_edge255.._crit_edge255_crit_edge:          ; preds = %._crit_edge255
; BB226 :
  %2341 = add nuw nsw i32 %2332, 1, !spirv.Decorations !631		; visa id: 3012
  br label %._crit_edge255, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3013

2342:                                             ; preds = %._crit_edge255
; BB227 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3015
  %2343 = load i64, i64* %129, align 8		; visa id: 3015
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3016
  %2344 = bitcast i64 %2343 to <2 x i32>		; visa id: 3016
  %2345 = extractelement <2 x i32> %2344, i32 0		; visa id: 3018
  %2346 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2345, i32 1
  %2347 = bitcast <2 x i32> %2346 to i64		; visa id: 3018
  %2348 = ashr exact i64 %2347, 32		; visa id: 3019
  %2349 = bitcast i64 %2348 to <2 x i32>		; visa id: 3020
  %2350 = extractelement <2 x i32> %2349, i32 0		; visa id: 3024
  %2351 = extractelement <2 x i32> %2349, i32 1		; visa id: 3024
  %2352 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2350, i32 %2351, i32 %41, i32 %42)
  %2353 = extractvalue { i32, i32 } %2352, 0		; visa id: 3024
  %2354 = extractvalue { i32, i32 } %2352, 1		; visa id: 3024
  %2355 = insertelement <2 x i32> undef, i32 %2353, i32 0		; visa id: 3031
  %2356 = insertelement <2 x i32> %2355, i32 %2354, i32 1		; visa id: 3032
  %2357 = bitcast <2 x i32> %2356 to i64		; visa id: 3033
  %2358 = shl i64 %2357, 1		; visa id: 3037
  %2359 = add i64 %.in401, %2358		; visa id: 3038
  %2360 = ashr i64 %2343, 31		; visa id: 3039
  %2361 = bitcast i64 %2360 to <2 x i32>		; visa id: 3040
  %2362 = extractelement <2 x i32> %2361, i32 0		; visa id: 3044
  %2363 = extractelement <2 x i32> %2361, i32 1		; visa id: 3044
  %2364 = and i32 %2362, -2		; visa id: 3044
  %2365 = insertelement <2 x i32> undef, i32 %2364, i32 0		; visa id: 3045
  %2366 = insertelement <2 x i32> %2365, i32 %2363, i32 1		; visa id: 3046
  %2367 = bitcast <2 x i32> %2366 to i64		; visa id: 3047
  %2368 = add i64 %2359, %2367		; visa id: 3051
  %2369 = inttoptr i64 %2368 to i16 addrspace(4)*		; visa id: 3052
  %2370 = addrspacecast i16 addrspace(4)* %2369 to i16 addrspace(1)*		; visa id: 3052
  %2371 = load i16, i16 addrspace(1)* %2370, align 2		; visa id: 3053
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3055
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3055
  %2372 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3055
  %2373 = insertelement <2 x i32> %2372, i32 %2324, i64 1		; visa id: 3056
  %2374 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3057
  store <2 x i32> %2373, <2 x i32>* %2374, align 4, !noalias !635		; visa id: 3057
  br label %._crit_edge256, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3059

._crit_edge256:                                   ; preds = %._crit_edge256.._crit_edge256_crit_edge, %2342
; BB228 :
  %2375 = phi i32 [ 0, %2342 ], [ %2384, %._crit_edge256.._crit_edge256_crit_edge ]
  %2376 = zext i32 %2375 to i64		; visa id: 3060
  %2377 = shl nuw nsw i64 %2376, 2		; visa id: 3061
  %2378 = add i64 %124, %2377		; visa id: 3062
  %2379 = inttoptr i64 %2378 to i32*		; visa id: 3063
  %2380 = load i32, i32* %2379, align 4, !noalias !635		; visa id: 3063
  %2381 = add i64 %119, %2377		; visa id: 3064
  %2382 = inttoptr i64 %2381 to i32*		; visa id: 3065
  store i32 %2380, i32* %2382, align 4, !alias.scope !635		; visa id: 3065
  %2383 = icmp eq i32 %2375, 0		; visa id: 3066
  br i1 %2383, label %._crit_edge256.._crit_edge256_crit_edge, label %2385, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3067

._crit_edge256.._crit_edge256_crit_edge:          ; preds = %._crit_edge256
; BB229 :
  %2384 = add nuw nsw i32 %2375, 1, !spirv.Decorations !631		; visa id: 3069
  br label %._crit_edge256, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3070

2385:                                             ; preds = %._crit_edge256
; BB230 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3072
  %2386 = load i64, i64* %120, align 8		; visa id: 3072
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3073
  %2387 = ashr i64 %2386, 32		; visa id: 3073
  %2388 = bitcast i64 %2387 to <2 x i32>		; visa id: 3074
  %2389 = extractelement <2 x i32> %2388, i32 0		; visa id: 3078
  %2390 = extractelement <2 x i32> %2388, i32 1		; visa id: 3078
  %2391 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2389, i32 %2390, i32 %44, i32 %45)
  %2392 = extractvalue { i32, i32 } %2391, 0		; visa id: 3078
  %2393 = extractvalue { i32, i32 } %2391, 1		; visa id: 3078
  %2394 = insertelement <2 x i32> undef, i32 %2392, i32 0		; visa id: 3085
  %2395 = insertelement <2 x i32> %2394, i32 %2393, i32 1		; visa id: 3086
  %2396 = bitcast <2 x i32> %2395 to i64		; visa id: 3087
  %2397 = bitcast i64 %2386 to <2 x i32>		; visa id: 3091
  %2398 = extractelement <2 x i32> %2397, i32 0		; visa id: 3093
  %2399 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2398, i32 1
  %2400 = bitcast <2 x i32> %2399 to i64		; visa id: 3093
  %2401 = shl i64 %2396, 1		; visa id: 3094
  %2402 = add i64 %.in400, %2401		; visa id: 3095
  %2403 = ashr exact i64 %2400, 31		; visa id: 3096
  %2404 = add i64 %2402, %2403		; visa id: 3097
  %2405 = inttoptr i64 %2404 to i16 addrspace(4)*		; visa id: 3098
  %2406 = addrspacecast i16 addrspace(4)* %2405 to i16 addrspace(1)*		; visa id: 3098
  %2407 = load i16, i16 addrspace(1)* %2406, align 2		; visa id: 3099
  %2408 = zext i16 %2371 to i32		; visa id: 3101
  %2409 = shl nuw i32 %2408, 16, !spirv.Decorations !639		; visa id: 3102
  %2410 = bitcast i32 %2409 to float
  %2411 = zext i16 %2407 to i32		; visa id: 3103
  %2412 = shl nuw i32 %2411, 16, !spirv.Decorations !639		; visa id: 3104
  %2413 = bitcast i32 %2412 to float
  %2414 = fmul reassoc nsz arcp contract float %2410, %2413, !spirv.Decorations !618
  %2415 = fadd reassoc nsz arcp contract float %2414, %.sroa.26.1, !spirv.Decorations !618		; visa id: 3105
  br label %._crit_edge.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3106

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %2385
; BB231 :
  %.sroa.26.2 = phi float [ %2415, %2385 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  %2416 = icmp slt i32 %223, %const_reg_dword
  %2417 = icmp slt i32 %2324, %const_reg_dword1		; visa id: 3107
  %2418 = and i1 %2416, %2417		; visa id: 3108
  br i1 %2418, label %2419, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3110

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
; BB:
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2419:                                             ; preds = %._crit_edge.6
; BB233 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3112
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3112
  %2420 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 3112
  %2421 = insertelement <2 x i32> %2420, i32 %113, i64 1		; visa id: 3113
  %2422 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3114
  store <2 x i32> %2421, <2 x i32>* %2422, align 4, !noalias !625		; visa id: 3114
  br label %._crit_edge257, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3116

._crit_edge257:                                   ; preds = %._crit_edge257.._crit_edge257_crit_edge, %2419
; BB234 :
  %2423 = phi i32 [ 0, %2419 ], [ %2432, %._crit_edge257.._crit_edge257_crit_edge ]
  %2424 = zext i32 %2423 to i64		; visa id: 3117
  %2425 = shl nuw nsw i64 %2424, 2		; visa id: 3118
  %2426 = add i64 %133, %2425		; visa id: 3119
  %2427 = inttoptr i64 %2426 to i32*		; visa id: 3120
  %2428 = load i32, i32* %2427, align 4, !noalias !625		; visa id: 3120
  %2429 = add i64 %128, %2425		; visa id: 3121
  %2430 = inttoptr i64 %2429 to i32*		; visa id: 3122
  store i32 %2428, i32* %2430, align 4, !alias.scope !625		; visa id: 3122
  %2431 = icmp eq i32 %2423, 0		; visa id: 3123
  br i1 %2431, label %._crit_edge257.._crit_edge257_crit_edge, label %2433, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3124

._crit_edge257.._crit_edge257_crit_edge:          ; preds = %._crit_edge257
; BB235 :
  %2432 = add nuw nsw i32 %2423, 1, !spirv.Decorations !631		; visa id: 3126
  br label %._crit_edge257, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3127

2433:                                             ; preds = %._crit_edge257
; BB236 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3129
  %2434 = load i64, i64* %129, align 8		; visa id: 3129
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3130
  %2435 = bitcast i64 %2434 to <2 x i32>		; visa id: 3130
  %2436 = extractelement <2 x i32> %2435, i32 0		; visa id: 3132
  %2437 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2436, i32 1
  %2438 = bitcast <2 x i32> %2437 to i64		; visa id: 3132
  %2439 = ashr exact i64 %2438, 32		; visa id: 3133
  %2440 = bitcast i64 %2439 to <2 x i32>		; visa id: 3134
  %2441 = extractelement <2 x i32> %2440, i32 0		; visa id: 3138
  %2442 = extractelement <2 x i32> %2440, i32 1		; visa id: 3138
  %2443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2441, i32 %2442, i32 %41, i32 %42)
  %2444 = extractvalue { i32, i32 } %2443, 0		; visa id: 3138
  %2445 = extractvalue { i32, i32 } %2443, 1		; visa id: 3138
  %2446 = insertelement <2 x i32> undef, i32 %2444, i32 0		; visa id: 3145
  %2447 = insertelement <2 x i32> %2446, i32 %2445, i32 1		; visa id: 3146
  %2448 = bitcast <2 x i32> %2447 to i64		; visa id: 3147
  %2449 = shl i64 %2448, 1		; visa id: 3151
  %2450 = add i64 %.in401, %2449		; visa id: 3152
  %2451 = ashr i64 %2434, 31		; visa id: 3153
  %2452 = bitcast i64 %2451 to <2 x i32>		; visa id: 3154
  %2453 = extractelement <2 x i32> %2452, i32 0		; visa id: 3158
  %2454 = extractelement <2 x i32> %2452, i32 1		; visa id: 3158
  %2455 = and i32 %2453, -2		; visa id: 3158
  %2456 = insertelement <2 x i32> undef, i32 %2455, i32 0		; visa id: 3159
  %2457 = insertelement <2 x i32> %2456, i32 %2454, i32 1		; visa id: 3160
  %2458 = bitcast <2 x i32> %2457 to i64		; visa id: 3161
  %2459 = add i64 %2450, %2458		; visa id: 3165
  %2460 = inttoptr i64 %2459 to i16 addrspace(4)*		; visa id: 3166
  %2461 = addrspacecast i16 addrspace(4)* %2460 to i16 addrspace(1)*		; visa id: 3166
  %2462 = load i16, i16 addrspace(1)* %2461, align 2		; visa id: 3167
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3169
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3169
  %2463 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3169
  %2464 = insertelement <2 x i32> %2463, i32 %2324, i64 1		; visa id: 3170
  %2465 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3171
  store <2 x i32> %2464, <2 x i32>* %2465, align 4, !noalias !635		; visa id: 3171
  br label %._crit_edge258, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3173

._crit_edge258:                                   ; preds = %._crit_edge258.._crit_edge258_crit_edge, %2433
; BB237 :
  %2466 = phi i32 [ 0, %2433 ], [ %2475, %._crit_edge258.._crit_edge258_crit_edge ]
  %2467 = zext i32 %2466 to i64		; visa id: 3174
  %2468 = shl nuw nsw i64 %2467, 2		; visa id: 3175
  %2469 = add i64 %124, %2468		; visa id: 3176
  %2470 = inttoptr i64 %2469 to i32*		; visa id: 3177
  %2471 = load i32, i32* %2470, align 4, !noalias !635		; visa id: 3177
  %2472 = add i64 %119, %2468		; visa id: 3178
  %2473 = inttoptr i64 %2472 to i32*		; visa id: 3179
  store i32 %2471, i32* %2473, align 4, !alias.scope !635		; visa id: 3179
  %2474 = icmp eq i32 %2466, 0		; visa id: 3180
  br i1 %2474, label %._crit_edge258.._crit_edge258_crit_edge, label %2476, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3181

._crit_edge258.._crit_edge258_crit_edge:          ; preds = %._crit_edge258
; BB238 :
  %2475 = add nuw nsw i32 %2466, 1, !spirv.Decorations !631		; visa id: 3183
  br label %._crit_edge258, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3184

2476:                                             ; preds = %._crit_edge258
; BB239 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3186
  %2477 = load i64, i64* %120, align 8		; visa id: 3186
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3187
  %2478 = ashr i64 %2477, 32		; visa id: 3187
  %2479 = bitcast i64 %2478 to <2 x i32>		; visa id: 3188
  %2480 = extractelement <2 x i32> %2479, i32 0		; visa id: 3192
  %2481 = extractelement <2 x i32> %2479, i32 1		; visa id: 3192
  %2482 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2480, i32 %2481, i32 %44, i32 %45)
  %2483 = extractvalue { i32, i32 } %2482, 0		; visa id: 3192
  %2484 = extractvalue { i32, i32 } %2482, 1		; visa id: 3192
  %2485 = insertelement <2 x i32> undef, i32 %2483, i32 0		; visa id: 3199
  %2486 = insertelement <2 x i32> %2485, i32 %2484, i32 1		; visa id: 3200
  %2487 = bitcast <2 x i32> %2486 to i64		; visa id: 3201
  %2488 = bitcast i64 %2477 to <2 x i32>		; visa id: 3205
  %2489 = extractelement <2 x i32> %2488, i32 0		; visa id: 3207
  %2490 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2489, i32 1
  %2491 = bitcast <2 x i32> %2490 to i64		; visa id: 3207
  %2492 = shl i64 %2487, 1		; visa id: 3208
  %2493 = add i64 %.in400, %2492		; visa id: 3209
  %2494 = ashr exact i64 %2491, 31		; visa id: 3210
  %2495 = add i64 %2493, %2494		; visa id: 3211
  %2496 = inttoptr i64 %2495 to i16 addrspace(4)*		; visa id: 3212
  %2497 = addrspacecast i16 addrspace(4)* %2496 to i16 addrspace(1)*		; visa id: 3212
  %2498 = load i16, i16 addrspace(1)* %2497, align 2		; visa id: 3213
  %2499 = zext i16 %2462 to i32		; visa id: 3215
  %2500 = shl nuw i32 %2499, 16, !spirv.Decorations !639		; visa id: 3216
  %2501 = bitcast i32 %2500 to float
  %2502 = zext i16 %2498 to i32		; visa id: 3217
  %2503 = shl nuw i32 %2502, 16, !spirv.Decorations !639		; visa id: 3218
  %2504 = bitcast i32 %2503 to float
  %2505 = fmul reassoc nsz arcp contract float %2501, %2504, !spirv.Decorations !618
  %2506 = fadd reassoc nsz arcp contract float %2505, %.sroa.90.1, !spirv.Decorations !618		; visa id: 3219
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3220

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %2476
; BB240 :
  %.sroa.90.2 = phi float [ %2506, %2476 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  %2507 = icmp slt i32 %315, %const_reg_dword
  %2508 = icmp slt i32 %2324, %const_reg_dword1		; visa id: 3221
  %2509 = and i1 %2507, %2508		; visa id: 3222
  br i1 %2509, label %2510, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3224

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
; BB:
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2510:                                             ; preds = %._crit_edge.1.6
; BB242 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3226
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3226
  %2511 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 3226
  %2512 = insertelement <2 x i32> %2511, i32 %113, i64 1		; visa id: 3227
  %2513 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3228
  store <2 x i32> %2512, <2 x i32>* %2513, align 4, !noalias !625		; visa id: 3228
  br label %._crit_edge259, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3230

._crit_edge259:                                   ; preds = %._crit_edge259.._crit_edge259_crit_edge, %2510
; BB243 :
  %2514 = phi i32 [ 0, %2510 ], [ %2523, %._crit_edge259.._crit_edge259_crit_edge ]
  %2515 = zext i32 %2514 to i64		; visa id: 3231
  %2516 = shl nuw nsw i64 %2515, 2		; visa id: 3232
  %2517 = add i64 %133, %2516		; visa id: 3233
  %2518 = inttoptr i64 %2517 to i32*		; visa id: 3234
  %2519 = load i32, i32* %2518, align 4, !noalias !625		; visa id: 3234
  %2520 = add i64 %128, %2516		; visa id: 3235
  %2521 = inttoptr i64 %2520 to i32*		; visa id: 3236
  store i32 %2519, i32* %2521, align 4, !alias.scope !625		; visa id: 3236
  %2522 = icmp eq i32 %2514, 0		; visa id: 3237
  br i1 %2522, label %._crit_edge259.._crit_edge259_crit_edge, label %2524, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3238

._crit_edge259.._crit_edge259_crit_edge:          ; preds = %._crit_edge259
; BB244 :
  %2523 = add nuw nsw i32 %2514, 1, !spirv.Decorations !631		; visa id: 3240
  br label %._crit_edge259, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3241

2524:                                             ; preds = %._crit_edge259
; BB245 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3243
  %2525 = load i64, i64* %129, align 8		; visa id: 3243
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3244
  %2526 = bitcast i64 %2525 to <2 x i32>		; visa id: 3244
  %2527 = extractelement <2 x i32> %2526, i32 0		; visa id: 3246
  %2528 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2527, i32 1
  %2529 = bitcast <2 x i32> %2528 to i64		; visa id: 3246
  %2530 = ashr exact i64 %2529, 32		; visa id: 3247
  %2531 = bitcast i64 %2530 to <2 x i32>		; visa id: 3248
  %2532 = extractelement <2 x i32> %2531, i32 0		; visa id: 3252
  %2533 = extractelement <2 x i32> %2531, i32 1		; visa id: 3252
  %2534 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2532, i32 %2533, i32 %41, i32 %42)
  %2535 = extractvalue { i32, i32 } %2534, 0		; visa id: 3252
  %2536 = extractvalue { i32, i32 } %2534, 1		; visa id: 3252
  %2537 = insertelement <2 x i32> undef, i32 %2535, i32 0		; visa id: 3259
  %2538 = insertelement <2 x i32> %2537, i32 %2536, i32 1		; visa id: 3260
  %2539 = bitcast <2 x i32> %2538 to i64		; visa id: 3261
  %2540 = shl i64 %2539, 1		; visa id: 3265
  %2541 = add i64 %.in401, %2540		; visa id: 3266
  %2542 = ashr i64 %2525, 31		; visa id: 3267
  %2543 = bitcast i64 %2542 to <2 x i32>		; visa id: 3268
  %2544 = extractelement <2 x i32> %2543, i32 0		; visa id: 3272
  %2545 = extractelement <2 x i32> %2543, i32 1		; visa id: 3272
  %2546 = and i32 %2544, -2		; visa id: 3272
  %2547 = insertelement <2 x i32> undef, i32 %2546, i32 0		; visa id: 3273
  %2548 = insertelement <2 x i32> %2547, i32 %2545, i32 1		; visa id: 3274
  %2549 = bitcast <2 x i32> %2548 to i64		; visa id: 3275
  %2550 = add i64 %2541, %2549		; visa id: 3279
  %2551 = inttoptr i64 %2550 to i16 addrspace(4)*		; visa id: 3280
  %2552 = addrspacecast i16 addrspace(4)* %2551 to i16 addrspace(1)*		; visa id: 3280
  %2553 = load i16, i16 addrspace(1)* %2552, align 2		; visa id: 3281
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3283
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3283
  %2554 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3283
  %2555 = insertelement <2 x i32> %2554, i32 %2324, i64 1		; visa id: 3284
  %2556 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3285
  store <2 x i32> %2555, <2 x i32>* %2556, align 4, !noalias !635		; visa id: 3285
  br label %._crit_edge260, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3287

._crit_edge260:                                   ; preds = %._crit_edge260.._crit_edge260_crit_edge, %2524
; BB246 :
  %2557 = phi i32 [ 0, %2524 ], [ %2566, %._crit_edge260.._crit_edge260_crit_edge ]
  %2558 = zext i32 %2557 to i64		; visa id: 3288
  %2559 = shl nuw nsw i64 %2558, 2		; visa id: 3289
  %2560 = add i64 %124, %2559		; visa id: 3290
  %2561 = inttoptr i64 %2560 to i32*		; visa id: 3291
  %2562 = load i32, i32* %2561, align 4, !noalias !635		; visa id: 3291
  %2563 = add i64 %119, %2559		; visa id: 3292
  %2564 = inttoptr i64 %2563 to i32*		; visa id: 3293
  store i32 %2562, i32* %2564, align 4, !alias.scope !635		; visa id: 3293
  %2565 = icmp eq i32 %2557, 0		; visa id: 3294
  br i1 %2565, label %._crit_edge260.._crit_edge260_crit_edge, label %2567, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3295

._crit_edge260.._crit_edge260_crit_edge:          ; preds = %._crit_edge260
; BB247 :
  %2566 = add nuw nsw i32 %2557, 1, !spirv.Decorations !631		; visa id: 3297
  br label %._crit_edge260, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3298

2567:                                             ; preds = %._crit_edge260
; BB248 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3300
  %2568 = load i64, i64* %120, align 8		; visa id: 3300
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3301
  %2569 = ashr i64 %2568, 32		; visa id: 3301
  %2570 = bitcast i64 %2569 to <2 x i32>		; visa id: 3302
  %2571 = extractelement <2 x i32> %2570, i32 0		; visa id: 3306
  %2572 = extractelement <2 x i32> %2570, i32 1		; visa id: 3306
  %2573 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2571, i32 %2572, i32 %44, i32 %45)
  %2574 = extractvalue { i32, i32 } %2573, 0		; visa id: 3306
  %2575 = extractvalue { i32, i32 } %2573, 1		; visa id: 3306
  %2576 = insertelement <2 x i32> undef, i32 %2574, i32 0		; visa id: 3313
  %2577 = insertelement <2 x i32> %2576, i32 %2575, i32 1		; visa id: 3314
  %2578 = bitcast <2 x i32> %2577 to i64		; visa id: 3315
  %2579 = bitcast i64 %2568 to <2 x i32>		; visa id: 3319
  %2580 = extractelement <2 x i32> %2579, i32 0		; visa id: 3321
  %2581 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2580, i32 1
  %2582 = bitcast <2 x i32> %2581 to i64		; visa id: 3321
  %2583 = shl i64 %2578, 1		; visa id: 3322
  %2584 = add i64 %.in400, %2583		; visa id: 3323
  %2585 = ashr exact i64 %2582, 31		; visa id: 3324
  %2586 = add i64 %2584, %2585		; visa id: 3325
  %2587 = inttoptr i64 %2586 to i16 addrspace(4)*		; visa id: 3326
  %2588 = addrspacecast i16 addrspace(4)* %2587 to i16 addrspace(1)*		; visa id: 3326
  %2589 = load i16, i16 addrspace(1)* %2588, align 2		; visa id: 3327
  %2590 = zext i16 %2553 to i32		; visa id: 3329
  %2591 = shl nuw i32 %2590, 16, !spirv.Decorations !639		; visa id: 3330
  %2592 = bitcast i32 %2591 to float
  %2593 = zext i16 %2589 to i32		; visa id: 3331
  %2594 = shl nuw i32 %2593, 16, !spirv.Decorations !639		; visa id: 3332
  %2595 = bitcast i32 %2594 to float
  %2596 = fmul reassoc nsz arcp contract float %2592, %2595, !spirv.Decorations !618
  %2597 = fadd reassoc nsz arcp contract float %2596, %.sroa.154.1, !spirv.Decorations !618		; visa id: 3333
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3334

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %2567
; BB249 :
  %.sroa.154.2 = phi float [ %2597, %2567 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  %2598 = icmp slt i32 %407, %const_reg_dword
  %2599 = icmp slt i32 %2324, %const_reg_dword1		; visa id: 3335
  %2600 = and i1 %2598, %2599		; visa id: 3336
  br i1 %2600, label %2601, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3338

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
; BB:
  br label %.preheader.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2601:                                             ; preds = %._crit_edge.2.6
; BB251 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3340
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3340
  %2602 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 3340
  %2603 = insertelement <2 x i32> %2602, i32 %113, i64 1		; visa id: 3341
  %2604 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3342
  store <2 x i32> %2603, <2 x i32>* %2604, align 4, !noalias !625		; visa id: 3342
  br label %._crit_edge261, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3344

._crit_edge261:                                   ; preds = %._crit_edge261.._crit_edge261_crit_edge, %2601
; BB252 :
  %2605 = phi i32 [ 0, %2601 ], [ %2614, %._crit_edge261.._crit_edge261_crit_edge ]
  %2606 = zext i32 %2605 to i64		; visa id: 3345
  %2607 = shl nuw nsw i64 %2606, 2		; visa id: 3346
  %2608 = add i64 %133, %2607		; visa id: 3347
  %2609 = inttoptr i64 %2608 to i32*		; visa id: 3348
  %2610 = load i32, i32* %2609, align 4, !noalias !625		; visa id: 3348
  %2611 = add i64 %128, %2607		; visa id: 3349
  %2612 = inttoptr i64 %2611 to i32*		; visa id: 3350
  store i32 %2610, i32* %2612, align 4, !alias.scope !625		; visa id: 3350
  %2613 = icmp eq i32 %2605, 0		; visa id: 3351
  br i1 %2613, label %._crit_edge261.._crit_edge261_crit_edge, label %2615, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3352

._crit_edge261.._crit_edge261_crit_edge:          ; preds = %._crit_edge261
; BB253 :
  %2614 = add nuw nsw i32 %2605, 1, !spirv.Decorations !631		; visa id: 3354
  br label %._crit_edge261, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3355

2615:                                             ; preds = %._crit_edge261
; BB254 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3357
  %2616 = load i64, i64* %129, align 8		; visa id: 3357
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3358
  %2617 = bitcast i64 %2616 to <2 x i32>		; visa id: 3358
  %2618 = extractelement <2 x i32> %2617, i32 0		; visa id: 3360
  %2619 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2618, i32 1
  %2620 = bitcast <2 x i32> %2619 to i64		; visa id: 3360
  %2621 = ashr exact i64 %2620, 32		; visa id: 3361
  %2622 = bitcast i64 %2621 to <2 x i32>		; visa id: 3362
  %2623 = extractelement <2 x i32> %2622, i32 0		; visa id: 3366
  %2624 = extractelement <2 x i32> %2622, i32 1		; visa id: 3366
  %2625 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2623, i32 %2624, i32 %41, i32 %42)
  %2626 = extractvalue { i32, i32 } %2625, 0		; visa id: 3366
  %2627 = extractvalue { i32, i32 } %2625, 1		; visa id: 3366
  %2628 = insertelement <2 x i32> undef, i32 %2626, i32 0		; visa id: 3373
  %2629 = insertelement <2 x i32> %2628, i32 %2627, i32 1		; visa id: 3374
  %2630 = bitcast <2 x i32> %2629 to i64		; visa id: 3375
  %2631 = shl i64 %2630, 1		; visa id: 3379
  %2632 = add i64 %.in401, %2631		; visa id: 3380
  %2633 = ashr i64 %2616, 31		; visa id: 3381
  %2634 = bitcast i64 %2633 to <2 x i32>		; visa id: 3382
  %2635 = extractelement <2 x i32> %2634, i32 0		; visa id: 3386
  %2636 = extractelement <2 x i32> %2634, i32 1		; visa id: 3386
  %2637 = and i32 %2635, -2		; visa id: 3386
  %2638 = insertelement <2 x i32> undef, i32 %2637, i32 0		; visa id: 3387
  %2639 = insertelement <2 x i32> %2638, i32 %2636, i32 1		; visa id: 3388
  %2640 = bitcast <2 x i32> %2639 to i64		; visa id: 3389
  %2641 = add i64 %2632, %2640		; visa id: 3393
  %2642 = inttoptr i64 %2641 to i16 addrspace(4)*		; visa id: 3394
  %2643 = addrspacecast i16 addrspace(4)* %2642 to i16 addrspace(1)*		; visa id: 3394
  %2644 = load i16, i16 addrspace(1)* %2643, align 2		; visa id: 3395
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3397
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3397
  %2645 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3397
  %2646 = insertelement <2 x i32> %2645, i32 %2324, i64 1		; visa id: 3398
  %2647 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3399
  store <2 x i32> %2646, <2 x i32>* %2647, align 4, !noalias !635		; visa id: 3399
  br label %._crit_edge262, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3401

._crit_edge262:                                   ; preds = %._crit_edge262.._crit_edge262_crit_edge, %2615
; BB255 :
  %2648 = phi i32 [ 0, %2615 ], [ %2657, %._crit_edge262.._crit_edge262_crit_edge ]
  %2649 = zext i32 %2648 to i64		; visa id: 3402
  %2650 = shl nuw nsw i64 %2649, 2		; visa id: 3403
  %2651 = add i64 %124, %2650		; visa id: 3404
  %2652 = inttoptr i64 %2651 to i32*		; visa id: 3405
  %2653 = load i32, i32* %2652, align 4, !noalias !635		; visa id: 3405
  %2654 = add i64 %119, %2650		; visa id: 3406
  %2655 = inttoptr i64 %2654 to i32*		; visa id: 3407
  store i32 %2653, i32* %2655, align 4, !alias.scope !635		; visa id: 3407
  %2656 = icmp eq i32 %2648, 0		; visa id: 3408
  br i1 %2656, label %._crit_edge262.._crit_edge262_crit_edge, label %2658, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3409

._crit_edge262.._crit_edge262_crit_edge:          ; preds = %._crit_edge262
; BB256 :
  %2657 = add nuw nsw i32 %2648, 1, !spirv.Decorations !631		; visa id: 3411
  br label %._crit_edge262, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3412

2658:                                             ; preds = %._crit_edge262
; BB257 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3414
  %2659 = load i64, i64* %120, align 8		; visa id: 3414
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3415
  %2660 = ashr i64 %2659, 32		; visa id: 3415
  %2661 = bitcast i64 %2660 to <2 x i32>		; visa id: 3416
  %2662 = extractelement <2 x i32> %2661, i32 0		; visa id: 3420
  %2663 = extractelement <2 x i32> %2661, i32 1		; visa id: 3420
  %2664 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2662, i32 %2663, i32 %44, i32 %45)
  %2665 = extractvalue { i32, i32 } %2664, 0		; visa id: 3420
  %2666 = extractvalue { i32, i32 } %2664, 1		; visa id: 3420
  %2667 = insertelement <2 x i32> undef, i32 %2665, i32 0		; visa id: 3427
  %2668 = insertelement <2 x i32> %2667, i32 %2666, i32 1		; visa id: 3428
  %2669 = bitcast <2 x i32> %2668 to i64		; visa id: 3429
  %2670 = bitcast i64 %2659 to <2 x i32>		; visa id: 3433
  %2671 = extractelement <2 x i32> %2670, i32 0		; visa id: 3435
  %2672 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2671, i32 1
  %2673 = bitcast <2 x i32> %2672 to i64		; visa id: 3435
  %2674 = shl i64 %2669, 1		; visa id: 3436
  %2675 = add i64 %.in400, %2674		; visa id: 3437
  %2676 = ashr exact i64 %2673, 31		; visa id: 3438
  %2677 = add i64 %2675, %2676		; visa id: 3439
  %2678 = inttoptr i64 %2677 to i16 addrspace(4)*		; visa id: 3440
  %2679 = addrspacecast i16 addrspace(4)* %2678 to i16 addrspace(1)*		; visa id: 3440
  %2680 = load i16, i16 addrspace(1)* %2679, align 2		; visa id: 3441
  %2681 = zext i16 %2644 to i32		; visa id: 3443
  %2682 = shl nuw i32 %2681, 16, !spirv.Decorations !639		; visa id: 3444
  %2683 = bitcast i32 %2682 to float
  %2684 = zext i16 %2680 to i32		; visa id: 3445
  %2685 = shl nuw i32 %2684, 16, !spirv.Decorations !639		; visa id: 3446
  %2686 = bitcast i32 %2685 to float
  %2687 = fmul reassoc nsz arcp contract float %2683, %2686, !spirv.Decorations !618
  %2688 = fadd reassoc nsz arcp contract float %2687, %.sroa.218.1, !spirv.Decorations !618		; visa id: 3447
  br label %.preheader.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3448

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %2658
; BB258 :
  %.sroa.218.2 = phi float [ %2688, %2658 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %2689 = add i32 %69, 7		; visa id: 3449
  %2690 = icmp slt i32 %2689, %const_reg_dword1		; visa id: 3450
  %2691 = icmp slt i32 %65, %const_reg_dword
  %2692 = and i1 %2691, %2690		; visa id: 3451
  br i1 %2692, label %2693, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3453

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
; BB:
  br label %._crit_edge.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2693:                                             ; preds = %.preheader.6
; BB260 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3455
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3455
  %2694 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 3455
  %2695 = insertelement <2 x i32> %2694, i32 %113, i64 1		; visa id: 3456
  %2696 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3457
  store <2 x i32> %2695, <2 x i32>* %2696, align 4, !noalias !625		; visa id: 3457
  br label %._crit_edge263, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3459

._crit_edge263:                                   ; preds = %._crit_edge263.._crit_edge263_crit_edge, %2693
; BB261 :
  %2697 = phi i32 [ 0, %2693 ], [ %2706, %._crit_edge263.._crit_edge263_crit_edge ]
  %2698 = zext i32 %2697 to i64		; visa id: 3460
  %2699 = shl nuw nsw i64 %2698, 2		; visa id: 3461
  %2700 = add i64 %133, %2699		; visa id: 3462
  %2701 = inttoptr i64 %2700 to i32*		; visa id: 3463
  %2702 = load i32, i32* %2701, align 4, !noalias !625		; visa id: 3463
  %2703 = add i64 %128, %2699		; visa id: 3464
  %2704 = inttoptr i64 %2703 to i32*		; visa id: 3465
  store i32 %2702, i32* %2704, align 4, !alias.scope !625		; visa id: 3465
  %2705 = icmp eq i32 %2697, 0		; visa id: 3466
  br i1 %2705, label %._crit_edge263.._crit_edge263_crit_edge, label %2707, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3467

._crit_edge263.._crit_edge263_crit_edge:          ; preds = %._crit_edge263
; BB262 :
  %2706 = add nuw nsw i32 %2697, 1, !spirv.Decorations !631		; visa id: 3469
  br label %._crit_edge263, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3470

2707:                                             ; preds = %._crit_edge263
; BB263 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3472
  %2708 = load i64, i64* %129, align 8		; visa id: 3472
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3473
  %2709 = bitcast i64 %2708 to <2 x i32>		; visa id: 3473
  %2710 = extractelement <2 x i32> %2709, i32 0		; visa id: 3475
  %2711 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2710, i32 1
  %2712 = bitcast <2 x i32> %2711 to i64		; visa id: 3475
  %2713 = ashr exact i64 %2712, 32		; visa id: 3476
  %2714 = bitcast i64 %2713 to <2 x i32>		; visa id: 3477
  %2715 = extractelement <2 x i32> %2714, i32 0		; visa id: 3481
  %2716 = extractelement <2 x i32> %2714, i32 1		; visa id: 3481
  %2717 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2715, i32 %2716, i32 %41, i32 %42)
  %2718 = extractvalue { i32, i32 } %2717, 0		; visa id: 3481
  %2719 = extractvalue { i32, i32 } %2717, 1		; visa id: 3481
  %2720 = insertelement <2 x i32> undef, i32 %2718, i32 0		; visa id: 3488
  %2721 = insertelement <2 x i32> %2720, i32 %2719, i32 1		; visa id: 3489
  %2722 = bitcast <2 x i32> %2721 to i64		; visa id: 3490
  %2723 = shl i64 %2722, 1		; visa id: 3494
  %2724 = add i64 %.in401, %2723		; visa id: 3495
  %2725 = ashr i64 %2708, 31		; visa id: 3496
  %2726 = bitcast i64 %2725 to <2 x i32>		; visa id: 3497
  %2727 = extractelement <2 x i32> %2726, i32 0		; visa id: 3501
  %2728 = extractelement <2 x i32> %2726, i32 1		; visa id: 3501
  %2729 = and i32 %2727, -2		; visa id: 3501
  %2730 = insertelement <2 x i32> undef, i32 %2729, i32 0		; visa id: 3502
  %2731 = insertelement <2 x i32> %2730, i32 %2728, i32 1		; visa id: 3503
  %2732 = bitcast <2 x i32> %2731 to i64		; visa id: 3504
  %2733 = add i64 %2724, %2732		; visa id: 3508
  %2734 = inttoptr i64 %2733 to i16 addrspace(4)*		; visa id: 3509
  %2735 = addrspacecast i16 addrspace(4)* %2734 to i16 addrspace(1)*		; visa id: 3509
  %2736 = load i16, i16 addrspace(1)* %2735, align 2		; visa id: 3510
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3512
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3512
  %2737 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3512
  %2738 = insertelement <2 x i32> %2737, i32 %2689, i64 1		; visa id: 3513
  %2739 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3514
  store <2 x i32> %2738, <2 x i32>* %2739, align 4, !noalias !635		; visa id: 3514
  br label %._crit_edge264, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3516

._crit_edge264:                                   ; preds = %._crit_edge264.._crit_edge264_crit_edge, %2707
; BB264 :
  %2740 = phi i32 [ 0, %2707 ], [ %2749, %._crit_edge264.._crit_edge264_crit_edge ]
  %2741 = zext i32 %2740 to i64		; visa id: 3517
  %2742 = shl nuw nsw i64 %2741, 2		; visa id: 3518
  %2743 = add i64 %124, %2742		; visa id: 3519
  %2744 = inttoptr i64 %2743 to i32*		; visa id: 3520
  %2745 = load i32, i32* %2744, align 4, !noalias !635		; visa id: 3520
  %2746 = add i64 %119, %2742		; visa id: 3521
  %2747 = inttoptr i64 %2746 to i32*		; visa id: 3522
  store i32 %2745, i32* %2747, align 4, !alias.scope !635		; visa id: 3522
  %2748 = icmp eq i32 %2740, 0		; visa id: 3523
  br i1 %2748, label %._crit_edge264.._crit_edge264_crit_edge, label %2750, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3524

._crit_edge264.._crit_edge264_crit_edge:          ; preds = %._crit_edge264
; BB265 :
  %2749 = add nuw nsw i32 %2740, 1, !spirv.Decorations !631		; visa id: 3526
  br label %._crit_edge264, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3527

2750:                                             ; preds = %._crit_edge264
; BB266 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3529
  %2751 = load i64, i64* %120, align 8		; visa id: 3529
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3530
  %2752 = ashr i64 %2751, 32		; visa id: 3530
  %2753 = bitcast i64 %2752 to <2 x i32>		; visa id: 3531
  %2754 = extractelement <2 x i32> %2753, i32 0		; visa id: 3535
  %2755 = extractelement <2 x i32> %2753, i32 1		; visa id: 3535
  %2756 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2754, i32 %2755, i32 %44, i32 %45)
  %2757 = extractvalue { i32, i32 } %2756, 0		; visa id: 3535
  %2758 = extractvalue { i32, i32 } %2756, 1		; visa id: 3535
  %2759 = insertelement <2 x i32> undef, i32 %2757, i32 0		; visa id: 3542
  %2760 = insertelement <2 x i32> %2759, i32 %2758, i32 1		; visa id: 3543
  %2761 = bitcast <2 x i32> %2760 to i64		; visa id: 3544
  %2762 = bitcast i64 %2751 to <2 x i32>		; visa id: 3548
  %2763 = extractelement <2 x i32> %2762, i32 0		; visa id: 3550
  %2764 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2763, i32 1
  %2765 = bitcast <2 x i32> %2764 to i64		; visa id: 3550
  %2766 = shl i64 %2761, 1		; visa id: 3551
  %2767 = add i64 %.in400, %2766		; visa id: 3552
  %2768 = ashr exact i64 %2765, 31		; visa id: 3553
  %2769 = add i64 %2767, %2768		; visa id: 3554
  %2770 = inttoptr i64 %2769 to i16 addrspace(4)*		; visa id: 3555
  %2771 = addrspacecast i16 addrspace(4)* %2770 to i16 addrspace(1)*		; visa id: 3555
  %2772 = load i16, i16 addrspace(1)* %2771, align 2		; visa id: 3556
  %2773 = zext i16 %2736 to i32		; visa id: 3558
  %2774 = shl nuw i32 %2773, 16, !spirv.Decorations !639		; visa id: 3559
  %2775 = bitcast i32 %2774 to float
  %2776 = zext i16 %2772 to i32		; visa id: 3560
  %2777 = shl nuw i32 %2776, 16, !spirv.Decorations !639		; visa id: 3561
  %2778 = bitcast i32 %2777 to float
  %2779 = fmul reassoc nsz arcp contract float %2775, %2778, !spirv.Decorations !618
  %2780 = fadd reassoc nsz arcp contract float %2779, %.sroa.30.1, !spirv.Decorations !618		; visa id: 3562
  br label %._crit_edge.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3563

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %2750
; BB267 :
  %.sroa.30.2 = phi float [ %2780, %2750 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  %2781 = icmp slt i32 %223, %const_reg_dword
  %2782 = icmp slt i32 %2689, %const_reg_dword1		; visa id: 3564
  %2783 = and i1 %2781, %2782		; visa id: 3565
  br i1 %2783, label %2784, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3567

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
; BB:
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2784:                                             ; preds = %._crit_edge.7
; BB269 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3569
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3569
  %2785 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 3569
  %2786 = insertelement <2 x i32> %2785, i32 %113, i64 1		; visa id: 3570
  %2787 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3571
  store <2 x i32> %2786, <2 x i32>* %2787, align 4, !noalias !625		; visa id: 3571
  br label %._crit_edge265, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3573

._crit_edge265:                                   ; preds = %._crit_edge265.._crit_edge265_crit_edge, %2784
; BB270 :
  %2788 = phi i32 [ 0, %2784 ], [ %2797, %._crit_edge265.._crit_edge265_crit_edge ]
  %2789 = zext i32 %2788 to i64		; visa id: 3574
  %2790 = shl nuw nsw i64 %2789, 2		; visa id: 3575
  %2791 = add i64 %133, %2790		; visa id: 3576
  %2792 = inttoptr i64 %2791 to i32*		; visa id: 3577
  %2793 = load i32, i32* %2792, align 4, !noalias !625		; visa id: 3577
  %2794 = add i64 %128, %2790		; visa id: 3578
  %2795 = inttoptr i64 %2794 to i32*		; visa id: 3579
  store i32 %2793, i32* %2795, align 4, !alias.scope !625		; visa id: 3579
  %2796 = icmp eq i32 %2788, 0		; visa id: 3580
  br i1 %2796, label %._crit_edge265.._crit_edge265_crit_edge, label %2798, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3581

._crit_edge265.._crit_edge265_crit_edge:          ; preds = %._crit_edge265
; BB271 :
  %2797 = add nuw nsw i32 %2788, 1, !spirv.Decorations !631		; visa id: 3583
  br label %._crit_edge265, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3584

2798:                                             ; preds = %._crit_edge265
; BB272 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3586
  %2799 = load i64, i64* %129, align 8		; visa id: 3586
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3587
  %2800 = bitcast i64 %2799 to <2 x i32>		; visa id: 3587
  %2801 = extractelement <2 x i32> %2800, i32 0		; visa id: 3589
  %2802 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2801, i32 1
  %2803 = bitcast <2 x i32> %2802 to i64		; visa id: 3589
  %2804 = ashr exact i64 %2803, 32		; visa id: 3590
  %2805 = bitcast i64 %2804 to <2 x i32>		; visa id: 3591
  %2806 = extractelement <2 x i32> %2805, i32 0		; visa id: 3595
  %2807 = extractelement <2 x i32> %2805, i32 1		; visa id: 3595
  %2808 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2806, i32 %2807, i32 %41, i32 %42)
  %2809 = extractvalue { i32, i32 } %2808, 0		; visa id: 3595
  %2810 = extractvalue { i32, i32 } %2808, 1		; visa id: 3595
  %2811 = insertelement <2 x i32> undef, i32 %2809, i32 0		; visa id: 3602
  %2812 = insertelement <2 x i32> %2811, i32 %2810, i32 1		; visa id: 3603
  %2813 = bitcast <2 x i32> %2812 to i64		; visa id: 3604
  %2814 = shl i64 %2813, 1		; visa id: 3608
  %2815 = add i64 %.in401, %2814		; visa id: 3609
  %2816 = ashr i64 %2799, 31		; visa id: 3610
  %2817 = bitcast i64 %2816 to <2 x i32>		; visa id: 3611
  %2818 = extractelement <2 x i32> %2817, i32 0		; visa id: 3615
  %2819 = extractelement <2 x i32> %2817, i32 1		; visa id: 3615
  %2820 = and i32 %2818, -2		; visa id: 3615
  %2821 = insertelement <2 x i32> undef, i32 %2820, i32 0		; visa id: 3616
  %2822 = insertelement <2 x i32> %2821, i32 %2819, i32 1		; visa id: 3617
  %2823 = bitcast <2 x i32> %2822 to i64		; visa id: 3618
  %2824 = add i64 %2815, %2823		; visa id: 3622
  %2825 = inttoptr i64 %2824 to i16 addrspace(4)*		; visa id: 3623
  %2826 = addrspacecast i16 addrspace(4)* %2825 to i16 addrspace(1)*		; visa id: 3623
  %2827 = load i16, i16 addrspace(1)* %2826, align 2		; visa id: 3624
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3626
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3626
  %2828 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3626
  %2829 = insertelement <2 x i32> %2828, i32 %2689, i64 1		; visa id: 3627
  %2830 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3628
  store <2 x i32> %2829, <2 x i32>* %2830, align 4, !noalias !635		; visa id: 3628
  br label %._crit_edge266, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3630

._crit_edge266:                                   ; preds = %._crit_edge266.._crit_edge266_crit_edge, %2798
; BB273 :
  %2831 = phi i32 [ 0, %2798 ], [ %2840, %._crit_edge266.._crit_edge266_crit_edge ]
  %2832 = zext i32 %2831 to i64		; visa id: 3631
  %2833 = shl nuw nsw i64 %2832, 2		; visa id: 3632
  %2834 = add i64 %124, %2833		; visa id: 3633
  %2835 = inttoptr i64 %2834 to i32*		; visa id: 3634
  %2836 = load i32, i32* %2835, align 4, !noalias !635		; visa id: 3634
  %2837 = add i64 %119, %2833		; visa id: 3635
  %2838 = inttoptr i64 %2837 to i32*		; visa id: 3636
  store i32 %2836, i32* %2838, align 4, !alias.scope !635		; visa id: 3636
  %2839 = icmp eq i32 %2831, 0		; visa id: 3637
  br i1 %2839, label %._crit_edge266.._crit_edge266_crit_edge, label %2841, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3638

._crit_edge266.._crit_edge266_crit_edge:          ; preds = %._crit_edge266
; BB274 :
  %2840 = add nuw nsw i32 %2831, 1, !spirv.Decorations !631		; visa id: 3640
  br label %._crit_edge266, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3641

2841:                                             ; preds = %._crit_edge266
; BB275 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3643
  %2842 = load i64, i64* %120, align 8		; visa id: 3643
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3644
  %2843 = ashr i64 %2842, 32		; visa id: 3644
  %2844 = bitcast i64 %2843 to <2 x i32>		; visa id: 3645
  %2845 = extractelement <2 x i32> %2844, i32 0		; visa id: 3649
  %2846 = extractelement <2 x i32> %2844, i32 1		; visa id: 3649
  %2847 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2845, i32 %2846, i32 %44, i32 %45)
  %2848 = extractvalue { i32, i32 } %2847, 0		; visa id: 3649
  %2849 = extractvalue { i32, i32 } %2847, 1		; visa id: 3649
  %2850 = insertelement <2 x i32> undef, i32 %2848, i32 0		; visa id: 3656
  %2851 = insertelement <2 x i32> %2850, i32 %2849, i32 1		; visa id: 3657
  %2852 = bitcast <2 x i32> %2851 to i64		; visa id: 3658
  %2853 = bitcast i64 %2842 to <2 x i32>		; visa id: 3662
  %2854 = extractelement <2 x i32> %2853, i32 0		; visa id: 3664
  %2855 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2854, i32 1
  %2856 = bitcast <2 x i32> %2855 to i64		; visa id: 3664
  %2857 = shl i64 %2852, 1		; visa id: 3665
  %2858 = add i64 %.in400, %2857		; visa id: 3666
  %2859 = ashr exact i64 %2856, 31		; visa id: 3667
  %2860 = add i64 %2858, %2859		; visa id: 3668
  %2861 = inttoptr i64 %2860 to i16 addrspace(4)*		; visa id: 3669
  %2862 = addrspacecast i16 addrspace(4)* %2861 to i16 addrspace(1)*		; visa id: 3669
  %2863 = load i16, i16 addrspace(1)* %2862, align 2		; visa id: 3670
  %2864 = zext i16 %2827 to i32		; visa id: 3672
  %2865 = shl nuw i32 %2864, 16, !spirv.Decorations !639		; visa id: 3673
  %2866 = bitcast i32 %2865 to float
  %2867 = zext i16 %2863 to i32		; visa id: 3674
  %2868 = shl nuw i32 %2867, 16, !spirv.Decorations !639		; visa id: 3675
  %2869 = bitcast i32 %2868 to float
  %2870 = fmul reassoc nsz arcp contract float %2866, %2869, !spirv.Decorations !618
  %2871 = fadd reassoc nsz arcp contract float %2870, %.sroa.94.1, !spirv.Decorations !618		; visa id: 3676
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3677

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %2841
; BB276 :
  %.sroa.94.2 = phi float [ %2871, %2841 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  %2872 = icmp slt i32 %315, %const_reg_dword
  %2873 = icmp slt i32 %2689, %const_reg_dword1		; visa id: 3678
  %2874 = and i1 %2872, %2873		; visa id: 3679
  br i1 %2874, label %2875, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3681

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
; BB:
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2875:                                             ; preds = %._crit_edge.1.7
; BB278 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3683
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3683
  %2876 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 3683
  %2877 = insertelement <2 x i32> %2876, i32 %113, i64 1		; visa id: 3684
  %2878 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3685
  store <2 x i32> %2877, <2 x i32>* %2878, align 4, !noalias !625		; visa id: 3685
  br label %._crit_edge267, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3687

._crit_edge267:                                   ; preds = %._crit_edge267.._crit_edge267_crit_edge, %2875
; BB279 :
  %2879 = phi i32 [ 0, %2875 ], [ %2888, %._crit_edge267.._crit_edge267_crit_edge ]
  %2880 = zext i32 %2879 to i64		; visa id: 3688
  %2881 = shl nuw nsw i64 %2880, 2		; visa id: 3689
  %2882 = add i64 %133, %2881		; visa id: 3690
  %2883 = inttoptr i64 %2882 to i32*		; visa id: 3691
  %2884 = load i32, i32* %2883, align 4, !noalias !625		; visa id: 3691
  %2885 = add i64 %128, %2881		; visa id: 3692
  %2886 = inttoptr i64 %2885 to i32*		; visa id: 3693
  store i32 %2884, i32* %2886, align 4, !alias.scope !625		; visa id: 3693
  %2887 = icmp eq i32 %2879, 0		; visa id: 3694
  br i1 %2887, label %._crit_edge267.._crit_edge267_crit_edge, label %2889, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3695

._crit_edge267.._crit_edge267_crit_edge:          ; preds = %._crit_edge267
; BB280 :
  %2888 = add nuw nsw i32 %2879, 1, !spirv.Decorations !631		; visa id: 3697
  br label %._crit_edge267, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3698

2889:                                             ; preds = %._crit_edge267
; BB281 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3700
  %2890 = load i64, i64* %129, align 8		; visa id: 3700
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3701
  %2891 = bitcast i64 %2890 to <2 x i32>		; visa id: 3701
  %2892 = extractelement <2 x i32> %2891, i32 0		; visa id: 3703
  %2893 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2892, i32 1
  %2894 = bitcast <2 x i32> %2893 to i64		; visa id: 3703
  %2895 = ashr exact i64 %2894, 32		; visa id: 3704
  %2896 = bitcast i64 %2895 to <2 x i32>		; visa id: 3705
  %2897 = extractelement <2 x i32> %2896, i32 0		; visa id: 3709
  %2898 = extractelement <2 x i32> %2896, i32 1		; visa id: 3709
  %2899 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2897, i32 %2898, i32 %41, i32 %42)
  %2900 = extractvalue { i32, i32 } %2899, 0		; visa id: 3709
  %2901 = extractvalue { i32, i32 } %2899, 1		; visa id: 3709
  %2902 = insertelement <2 x i32> undef, i32 %2900, i32 0		; visa id: 3716
  %2903 = insertelement <2 x i32> %2902, i32 %2901, i32 1		; visa id: 3717
  %2904 = bitcast <2 x i32> %2903 to i64		; visa id: 3718
  %2905 = shl i64 %2904, 1		; visa id: 3722
  %2906 = add i64 %.in401, %2905		; visa id: 3723
  %2907 = ashr i64 %2890, 31		; visa id: 3724
  %2908 = bitcast i64 %2907 to <2 x i32>		; visa id: 3725
  %2909 = extractelement <2 x i32> %2908, i32 0		; visa id: 3729
  %2910 = extractelement <2 x i32> %2908, i32 1		; visa id: 3729
  %2911 = and i32 %2909, -2		; visa id: 3729
  %2912 = insertelement <2 x i32> undef, i32 %2911, i32 0		; visa id: 3730
  %2913 = insertelement <2 x i32> %2912, i32 %2910, i32 1		; visa id: 3731
  %2914 = bitcast <2 x i32> %2913 to i64		; visa id: 3732
  %2915 = add i64 %2906, %2914		; visa id: 3736
  %2916 = inttoptr i64 %2915 to i16 addrspace(4)*		; visa id: 3737
  %2917 = addrspacecast i16 addrspace(4)* %2916 to i16 addrspace(1)*		; visa id: 3737
  %2918 = load i16, i16 addrspace(1)* %2917, align 2		; visa id: 3738
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3740
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3740
  %2919 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3740
  %2920 = insertelement <2 x i32> %2919, i32 %2689, i64 1		; visa id: 3741
  %2921 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3742
  store <2 x i32> %2920, <2 x i32>* %2921, align 4, !noalias !635		; visa id: 3742
  br label %._crit_edge268, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3744

._crit_edge268:                                   ; preds = %._crit_edge268.._crit_edge268_crit_edge, %2889
; BB282 :
  %2922 = phi i32 [ 0, %2889 ], [ %2931, %._crit_edge268.._crit_edge268_crit_edge ]
  %2923 = zext i32 %2922 to i64		; visa id: 3745
  %2924 = shl nuw nsw i64 %2923, 2		; visa id: 3746
  %2925 = add i64 %124, %2924		; visa id: 3747
  %2926 = inttoptr i64 %2925 to i32*		; visa id: 3748
  %2927 = load i32, i32* %2926, align 4, !noalias !635		; visa id: 3748
  %2928 = add i64 %119, %2924		; visa id: 3749
  %2929 = inttoptr i64 %2928 to i32*		; visa id: 3750
  store i32 %2927, i32* %2929, align 4, !alias.scope !635		; visa id: 3750
  %2930 = icmp eq i32 %2922, 0		; visa id: 3751
  br i1 %2930, label %._crit_edge268.._crit_edge268_crit_edge, label %2932, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3752

._crit_edge268.._crit_edge268_crit_edge:          ; preds = %._crit_edge268
; BB283 :
  %2931 = add nuw nsw i32 %2922, 1, !spirv.Decorations !631		; visa id: 3754
  br label %._crit_edge268, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3755

2932:                                             ; preds = %._crit_edge268
; BB284 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3757
  %2933 = load i64, i64* %120, align 8		; visa id: 3757
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3758
  %2934 = ashr i64 %2933, 32		; visa id: 3758
  %2935 = bitcast i64 %2934 to <2 x i32>		; visa id: 3759
  %2936 = extractelement <2 x i32> %2935, i32 0		; visa id: 3763
  %2937 = extractelement <2 x i32> %2935, i32 1		; visa id: 3763
  %2938 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2936, i32 %2937, i32 %44, i32 %45)
  %2939 = extractvalue { i32, i32 } %2938, 0		; visa id: 3763
  %2940 = extractvalue { i32, i32 } %2938, 1		; visa id: 3763
  %2941 = insertelement <2 x i32> undef, i32 %2939, i32 0		; visa id: 3770
  %2942 = insertelement <2 x i32> %2941, i32 %2940, i32 1		; visa id: 3771
  %2943 = bitcast <2 x i32> %2942 to i64		; visa id: 3772
  %2944 = bitcast i64 %2933 to <2 x i32>		; visa id: 3776
  %2945 = extractelement <2 x i32> %2944, i32 0		; visa id: 3778
  %2946 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2945, i32 1
  %2947 = bitcast <2 x i32> %2946 to i64		; visa id: 3778
  %2948 = shl i64 %2943, 1		; visa id: 3779
  %2949 = add i64 %.in400, %2948		; visa id: 3780
  %2950 = ashr exact i64 %2947, 31		; visa id: 3781
  %2951 = add i64 %2949, %2950		; visa id: 3782
  %2952 = inttoptr i64 %2951 to i16 addrspace(4)*		; visa id: 3783
  %2953 = addrspacecast i16 addrspace(4)* %2952 to i16 addrspace(1)*		; visa id: 3783
  %2954 = load i16, i16 addrspace(1)* %2953, align 2		; visa id: 3784
  %2955 = zext i16 %2918 to i32		; visa id: 3786
  %2956 = shl nuw i32 %2955, 16, !spirv.Decorations !639		; visa id: 3787
  %2957 = bitcast i32 %2956 to float
  %2958 = zext i16 %2954 to i32		; visa id: 3788
  %2959 = shl nuw i32 %2958, 16, !spirv.Decorations !639		; visa id: 3789
  %2960 = bitcast i32 %2959 to float
  %2961 = fmul reassoc nsz arcp contract float %2957, %2960, !spirv.Decorations !618
  %2962 = fadd reassoc nsz arcp contract float %2961, %.sroa.158.1, !spirv.Decorations !618		; visa id: 3790
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3791

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %2932
; BB285 :
  %.sroa.158.2 = phi float [ %2962, %2932 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  %2963 = icmp slt i32 %407, %const_reg_dword
  %2964 = icmp slt i32 %2689, %const_reg_dword1		; visa id: 3792
  %2965 = and i1 %2963, %2964		; visa id: 3793
  br i1 %2965, label %2966, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3795

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
; BB:
  br label %.preheader.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2966:                                             ; preds = %._crit_edge.2.7
; BB287 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3797
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3797
  %2967 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 3797
  %2968 = insertelement <2 x i32> %2967, i32 %113, i64 1		; visa id: 3798
  %2969 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3799
  store <2 x i32> %2968, <2 x i32>* %2969, align 4, !noalias !625		; visa id: 3799
  br label %._crit_edge269, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3801

._crit_edge269:                                   ; preds = %._crit_edge269.._crit_edge269_crit_edge, %2966
; BB288 :
  %2970 = phi i32 [ 0, %2966 ], [ %2979, %._crit_edge269.._crit_edge269_crit_edge ]
  %2971 = zext i32 %2970 to i64		; visa id: 3802
  %2972 = shl nuw nsw i64 %2971, 2		; visa id: 3803
  %2973 = add i64 %133, %2972		; visa id: 3804
  %2974 = inttoptr i64 %2973 to i32*		; visa id: 3805
  %2975 = load i32, i32* %2974, align 4, !noalias !625		; visa id: 3805
  %2976 = add i64 %128, %2972		; visa id: 3806
  %2977 = inttoptr i64 %2976 to i32*		; visa id: 3807
  store i32 %2975, i32* %2977, align 4, !alias.scope !625		; visa id: 3807
  %2978 = icmp eq i32 %2970, 0		; visa id: 3808
  br i1 %2978, label %._crit_edge269.._crit_edge269_crit_edge, label %2980, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3809

._crit_edge269.._crit_edge269_crit_edge:          ; preds = %._crit_edge269
; BB289 :
  %2979 = add nuw nsw i32 %2970, 1, !spirv.Decorations !631		; visa id: 3811
  br label %._crit_edge269, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3812

2980:                                             ; preds = %._crit_edge269
; BB290 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3814
  %2981 = load i64, i64* %129, align 8		; visa id: 3814
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3815
  %2982 = bitcast i64 %2981 to <2 x i32>		; visa id: 3815
  %2983 = extractelement <2 x i32> %2982, i32 0		; visa id: 3817
  %2984 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2983, i32 1
  %2985 = bitcast <2 x i32> %2984 to i64		; visa id: 3817
  %2986 = ashr exact i64 %2985, 32		; visa id: 3818
  %2987 = bitcast i64 %2986 to <2 x i32>		; visa id: 3819
  %2988 = extractelement <2 x i32> %2987, i32 0		; visa id: 3823
  %2989 = extractelement <2 x i32> %2987, i32 1		; visa id: 3823
  %2990 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2988, i32 %2989, i32 %41, i32 %42)
  %2991 = extractvalue { i32, i32 } %2990, 0		; visa id: 3823
  %2992 = extractvalue { i32, i32 } %2990, 1		; visa id: 3823
  %2993 = insertelement <2 x i32> undef, i32 %2991, i32 0		; visa id: 3830
  %2994 = insertelement <2 x i32> %2993, i32 %2992, i32 1		; visa id: 3831
  %2995 = bitcast <2 x i32> %2994 to i64		; visa id: 3832
  %2996 = shl i64 %2995, 1		; visa id: 3836
  %2997 = add i64 %.in401, %2996		; visa id: 3837
  %2998 = ashr i64 %2981, 31		; visa id: 3838
  %2999 = bitcast i64 %2998 to <2 x i32>		; visa id: 3839
  %3000 = extractelement <2 x i32> %2999, i32 0		; visa id: 3843
  %3001 = extractelement <2 x i32> %2999, i32 1		; visa id: 3843
  %3002 = and i32 %3000, -2		; visa id: 3843
  %3003 = insertelement <2 x i32> undef, i32 %3002, i32 0		; visa id: 3844
  %3004 = insertelement <2 x i32> %3003, i32 %3001, i32 1		; visa id: 3845
  %3005 = bitcast <2 x i32> %3004 to i64		; visa id: 3846
  %3006 = add i64 %2997, %3005		; visa id: 3850
  %3007 = inttoptr i64 %3006 to i16 addrspace(4)*		; visa id: 3851
  %3008 = addrspacecast i16 addrspace(4)* %3007 to i16 addrspace(1)*		; visa id: 3851
  %3009 = load i16, i16 addrspace(1)* %3008, align 2		; visa id: 3852
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3854
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3854
  %3010 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3854
  %3011 = insertelement <2 x i32> %3010, i32 %2689, i64 1		; visa id: 3855
  %3012 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3856
  store <2 x i32> %3011, <2 x i32>* %3012, align 4, !noalias !635		; visa id: 3856
  br label %._crit_edge270, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3858

._crit_edge270:                                   ; preds = %._crit_edge270.._crit_edge270_crit_edge, %2980
; BB291 :
  %3013 = phi i32 [ 0, %2980 ], [ %3022, %._crit_edge270.._crit_edge270_crit_edge ]
  %3014 = zext i32 %3013 to i64		; visa id: 3859
  %3015 = shl nuw nsw i64 %3014, 2		; visa id: 3860
  %3016 = add i64 %124, %3015		; visa id: 3861
  %3017 = inttoptr i64 %3016 to i32*		; visa id: 3862
  %3018 = load i32, i32* %3017, align 4, !noalias !635		; visa id: 3862
  %3019 = add i64 %119, %3015		; visa id: 3863
  %3020 = inttoptr i64 %3019 to i32*		; visa id: 3864
  store i32 %3018, i32* %3020, align 4, !alias.scope !635		; visa id: 3864
  %3021 = icmp eq i32 %3013, 0		; visa id: 3865
  br i1 %3021, label %._crit_edge270.._crit_edge270_crit_edge, label %3023, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3866

._crit_edge270.._crit_edge270_crit_edge:          ; preds = %._crit_edge270
; BB292 :
  %3022 = add nuw nsw i32 %3013, 1, !spirv.Decorations !631		; visa id: 3868
  br label %._crit_edge270, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3869

3023:                                             ; preds = %._crit_edge270
; BB293 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3871
  %3024 = load i64, i64* %120, align 8		; visa id: 3871
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3872
  %3025 = ashr i64 %3024, 32		; visa id: 3872
  %3026 = bitcast i64 %3025 to <2 x i32>		; visa id: 3873
  %3027 = extractelement <2 x i32> %3026, i32 0		; visa id: 3877
  %3028 = extractelement <2 x i32> %3026, i32 1		; visa id: 3877
  %3029 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3027, i32 %3028, i32 %44, i32 %45)
  %3030 = extractvalue { i32, i32 } %3029, 0		; visa id: 3877
  %3031 = extractvalue { i32, i32 } %3029, 1		; visa id: 3877
  %3032 = insertelement <2 x i32> undef, i32 %3030, i32 0		; visa id: 3884
  %3033 = insertelement <2 x i32> %3032, i32 %3031, i32 1		; visa id: 3885
  %3034 = bitcast <2 x i32> %3033 to i64		; visa id: 3886
  %3035 = bitcast i64 %3024 to <2 x i32>		; visa id: 3890
  %3036 = extractelement <2 x i32> %3035, i32 0		; visa id: 3892
  %3037 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3036, i32 1
  %3038 = bitcast <2 x i32> %3037 to i64		; visa id: 3892
  %3039 = shl i64 %3034, 1		; visa id: 3893
  %3040 = add i64 %.in400, %3039		; visa id: 3894
  %3041 = ashr exact i64 %3038, 31		; visa id: 3895
  %3042 = add i64 %3040, %3041		; visa id: 3896
  %3043 = inttoptr i64 %3042 to i16 addrspace(4)*		; visa id: 3897
  %3044 = addrspacecast i16 addrspace(4)* %3043 to i16 addrspace(1)*		; visa id: 3897
  %3045 = load i16, i16 addrspace(1)* %3044, align 2		; visa id: 3898
  %3046 = zext i16 %3009 to i32		; visa id: 3900
  %3047 = shl nuw i32 %3046, 16, !spirv.Decorations !639		; visa id: 3901
  %3048 = bitcast i32 %3047 to float
  %3049 = zext i16 %3045 to i32		; visa id: 3902
  %3050 = shl nuw i32 %3049, 16, !spirv.Decorations !639		; visa id: 3903
  %3051 = bitcast i32 %3050 to float
  %3052 = fmul reassoc nsz arcp contract float %3048, %3051, !spirv.Decorations !618
  %3053 = fadd reassoc nsz arcp contract float %3052, %.sroa.222.1, !spirv.Decorations !618		; visa id: 3904
  br label %.preheader.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3905

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %3023
; BB294 :
  %.sroa.222.2 = phi float [ %3053, %3023 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %3054 = add i32 %69, 8		; visa id: 3906
  %3055 = icmp slt i32 %3054, %const_reg_dword1		; visa id: 3907
  %3056 = icmp slt i32 %65, %const_reg_dword
  %3057 = and i1 %3056, %3055		; visa id: 3908
  br i1 %3057, label %3058, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3910

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
; BB:
  br label %._crit_edge.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3058:                                             ; preds = %.preheader.7
; BB296 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3912
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3912
  %3059 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 3912
  %3060 = insertelement <2 x i32> %3059, i32 %113, i64 1		; visa id: 3913
  %3061 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3914
  store <2 x i32> %3060, <2 x i32>* %3061, align 4, !noalias !625		; visa id: 3914
  br label %._crit_edge271, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3916

._crit_edge271:                                   ; preds = %._crit_edge271.._crit_edge271_crit_edge, %3058
; BB297 :
  %3062 = phi i32 [ 0, %3058 ], [ %3071, %._crit_edge271.._crit_edge271_crit_edge ]
  %3063 = zext i32 %3062 to i64		; visa id: 3917
  %3064 = shl nuw nsw i64 %3063, 2		; visa id: 3918
  %3065 = add i64 %133, %3064		; visa id: 3919
  %3066 = inttoptr i64 %3065 to i32*		; visa id: 3920
  %3067 = load i32, i32* %3066, align 4, !noalias !625		; visa id: 3920
  %3068 = add i64 %128, %3064		; visa id: 3921
  %3069 = inttoptr i64 %3068 to i32*		; visa id: 3922
  store i32 %3067, i32* %3069, align 4, !alias.scope !625		; visa id: 3922
  %3070 = icmp eq i32 %3062, 0		; visa id: 3923
  br i1 %3070, label %._crit_edge271.._crit_edge271_crit_edge, label %3072, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3924

._crit_edge271.._crit_edge271_crit_edge:          ; preds = %._crit_edge271
; BB298 :
  %3071 = add nuw nsw i32 %3062, 1, !spirv.Decorations !631		; visa id: 3926
  br label %._crit_edge271, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3927

3072:                                             ; preds = %._crit_edge271
; BB299 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3929
  %3073 = load i64, i64* %129, align 8		; visa id: 3929
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3930
  %3074 = bitcast i64 %3073 to <2 x i32>		; visa id: 3930
  %3075 = extractelement <2 x i32> %3074, i32 0		; visa id: 3932
  %3076 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3075, i32 1
  %3077 = bitcast <2 x i32> %3076 to i64		; visa id: 3932
  %3078 = ashr exact i64 %3077, 32		; visa id: 3933
  %3079 = bitcast i64 %3078 to <2 x i32>		; visa id: 3934
  %3080 = extractelement <2 x i32> %3079, i32 0		; visa id: 3938
  %3081 = extractelement <2 x i32> %3079, i32 1		; visa id: 3938
  %3082 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3080, i32 %3081, i32 %41, i32 %42)
  %3083 = extractvalue { i32, i32 } %3082, 0		; visa id: 3938
  %3084 = extractvalue { i32, i32 } %3082, 1		; visa id: 3938
  %3085 = insertelement <2 x i32> undef, i32 %3083, i32 0		; visa id: 3945
  %3086 = insertelement <2 x i32> %3085, i32 %3084, i32 1		; visa id: 3946
  %3087 = bitcast <2 x i32> %3086 to i64		; visa id: 3947
  %3088 = shl i64 %3087, 1		; visa id: 3951
  %3089 = add i64 %.in401, %3088		; visa id: 3952
  %3090 = ashr i64 %3073, 31		; visa id: 3953
  %3091 = bitcast i64 %3090 to <2 x i32>		; visa id: 3954
  %3092 = extractelement <2 x i32> %3091, i32 0		; visa id: 3958
  %3093 = extractelement <2 x i32> %3091, i32 1		; visa id: 3958
  %3094 = and i32 %3092, -2		; visa id: 3958
  %3095 = insertelement <2 x i32> undef, i32 %3094, i32 0		; visa id: 3959
  %3096 = insertelement <2 x i32> %3095, i32 %3093, i32 1		; visa id: 3960
  %3097 = bitcast <2 x i32> %3096 to i64		; visa id: 3961
  %3098 = add i64 %3089, %3097		; visa id: 3965
  %3099 = inttoptr i64 %3098 to i16 addrspace(4)*		; visa id: 3966
  %3100 = addrspacecast i16 addrspace(4)* %3099 to i16 addrspace(1)*		; visa id: 3966
  %3101 = load i16, i16 addrspace(1)* %3100, align 2		; visa id: 3967
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3969
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3969
  %3102 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3969
  %3103 = insertelement <2 x i32> %3102, i32 %3054, i64 1		; visa id: 3970
  %3104 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3971
  store <2 x i32> %3103, <2 x i32>* %3104, align 4, !noalias !635		; visa id: 3971
  br label %._crit_edge272, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3973

._crit_edge272:                                   ; preds = %._crit_edge272.._crit_edge272_crit_edge, %3072
; BB300 :
  %3105 = phi i32 [ 0, %3072 ], [ %3114, %._crit_edge272.._crit_edge272_crit_edge ]
  %3106 = zext i32 %3105 to i64		; visa id: 3974
  %3107 = shl nuw nsw i64 %3106, 2		; visa id: 3975
  %3108 = add i64 %124, %3107		; visa id: 3976
  %3109 = inttoptr i64 %3108 to i32*		; visa id: 3977
  %3110 = load i32, i32* %3109, align 4, !noalias !635		; visa id: 3977
  %3111 = add i64 %119, %3107		; visa id: 3978
  %3112 = inttoptr i64 %3111 to i32*		; visa id: 3979
  store i32 %3110, i32* %3112, align 4, !alias.scope !635		; visa id: 3979
  %3113 = icmp eq i32 %3105, 0		; visa id: 3980
  br i1 %3113, label %._crit_edge272.._crit_edge272_crit_edge, label %3115, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3981

._crit_edge272.._crit_edge272_crit_edge:          ; preds = %._crit_edge272
; BB301 :
  %3114 = add nuw nsw i32 %3105, 1, !spirv.Decorations !631		; visa id: 3983
  br label %._crit_edge272, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3984

3115:                                             ; preds = %._crit_edge272
; BB302 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3986
  %3116 = load i64, i64* %120, align 8		; visa id: 3986
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3987
  %3117 = ashr i64 %3116, 32		; visa id: 3987
  %3118 = bitcast i64 %3117 to <2 x i32>		; visa id: 3988
  %3119 = extractelement <2 x i32> %3118, i32 0		; visa id: 3992
  %3120 = extractelement <2 x i32> %3118, i32 1		; visa id: 3992
  %3121 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3119, i32 %3120, i32 %44, i32 %45)
  %3122 = extractvalue { i32, i32 } %3121, 0		; visa id: 3992
  %3123 = extractvalue { i32, i32 } %3121, 1		; visa id: 3992
  %3124 = insertelement <2 x i32> undef, i32 %3122, i32 0		; visa id: 3999
  %3125 = insertelement <2 x i32> %3124, i32 %3123, i32 1		; visa id: 4000
  %3126 = bitcast <2 x i32> %3125 to i64		; visa id: 4001
  %3127 = bitcast i64 %3116 to <2 x i32>		; visa id: 4005
  %3128 = extractelement <2 x i32> %3127, i32 0		; visa id: 4007
  %3129 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3128, i32 1
  %3130 = bitcast <2 x i32> %3129 to i64		; visa id: 4007
  %3131 = shl i64 %3126, 1		; visa id: 4008
  %3132 = add i64 %.in400, %3131		; visa id: 4009
  %3133 = ashr exact i64 %3130, 31		; visa id: 4010
  %3134 = add i64 %3132, %3133		; visa id: 4011
  %3135 = inttoptr i64 %3134 to i16 addrspace(4)*		; visa id: 4012
  %3136 = addrspacecast i16 addrspace(4)* %3135 to i16 addrspace(1)*		; visa id: 4012
  %3137 = load i16, i16 addrspace(1)* %3136, align 2		; visa id: 4013
  %3138 = zext i16 %3101 to i32		; visa id: 4015
  %3139 = shl nuw i32 %3138, 16, !spirv.Decorations !639		; visa id: 4016
  %3140 = bitcast i32 %3139 to float
  %3141 = zext i16 %3137 to i32		; visa id: 4017
  %3142 = shl nuw i32 %3141, 16, !spirv.Decorations !639		; visa id: 4018
  %3143 = bitcast i32 %3142 to float
  %3144 = fmul reassoc nsz arcp contract float %3140, %3143, !spirv.Decorations !618
  %3145 = fadd reassoc nsz arcp contract float %3144, %.sroa.34.1, !spirv.Decorations !618		; visa id: 4019
  br label %._crit_edge.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4020

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %3115
; BB303 :
  %.sroa.34.2 = phi float [ %3145, %3115 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  %3146 = icmp slt i32 %223, %const_reg_dword
  %3147 = icmp slt i32 %3054, %const_reg_dword1		; visa id: 4021
  %3148 = and i1 %3146, %3147		; visa id: 4022
  br i1 %3148, label %3149, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4024

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
; BB:
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3149:                                             ; preds = %._crit_edge.8
; BB305 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4026
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4026
  %3150 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 4026
  %3151 = insertelement <2 x i32> %3150, i32 %113, i64 1		; visa id: 4027
  %3152 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4028
  store <2 x i32> %3151, <2 x i32>* %3152, align 4, !noalias !625		; visa id: 4028
  br label %._crit_edge273, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4030

._crit_edge273:                                   ; preds = %._crit_edge273.._crit_edge273_crit_edge, %3149
; BB306 :
  %3153 = phi i32 [ 0, %3149 ], [ %3162, %._crit_edge273.._crit_edge273_crit_edge ]
  %3154 = zext i32 %3153 to i64		; visa id: 4031
  %3155 = shl nuw nsw i64 %3154, 2		; visa id: 4032
  %3156 = add i64 %133, %3155		; visa id: 4033
  %3157 = inttoptr i64 %3156 to i32*		; visa id: 4034
  %3158 = load i32, i32* %3157, align 4, !noalias !625		; visa id: 4034
  %3159 = add i64 %128, %3155		; visa id: 4035
  %3160 = inttoptr i64 %3159 to i32*		; visa id: 4036
  store i32 %3158, i32* %3160, align 4, !alias.scope !625		; visa id: 4036
  %3161 = icmp eq i32 %3153, 0		; visa id: 4037
  br i1 %3161, label %._crit_edge273.._crit_edge273_crit_edge, label %3163, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4038

._crit_edge273.._crit_edge273_crit_edge:          ; preds = %._crit_edge273
; BB307 :
  %3162 = add nuw nsw i32 %3153, 1, !spirv.Decorations !631		; visa id: 4040
  br label %._crit_edge273, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4041

3163:                                             ; preds = %._crit_edge273
; BB308 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4043
  %3164 = load i64, i64* %129, align 8		; visa id: 4043
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4044
  %3165 = bitcast i64 %3164 to <2 x i32>		; visa id: 4044
  %3166 = extractelement <2 x i32> %3165, i32 0		; visa id: 4046
  %3167 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3166, i32 1
  %3168 = bitcast <2 x i32> %3167 to i64		; visa id: 4046
  %3169 = ashr exact i64 %3168, 32		; visa id: 4047
  %3170 = bitcast i64 %3169 to <2 x i32>		; visa id: 4048
  %3171 = extractelement <2 x i32> %3170, i32 0		; visa id: 4052
  %3172 = extractelement <2 x i32> %3170, i32 1		; visa id: 4052
  %3173 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3171, i32 %3172, i32 %41, i32 %42)
  %3174 = extractvalue { i32, i32 } %3173, 0		; visa id: 4052
  %3175 = extractvalue { i32, i32 } %3173, 1		; visa id: 4052
  %3176 = insertelement <2 x i32> undef, i32 %3174, i32 0		; visa id: 4059
  %3177 = insertelement <2 x i32> %3176, i32 %3175, i32 1		; visa id: 4060
  %3178 = bitcast <2 x i32> %3177 to i64		; visa id: 4061
  %3179 = shl i64 %3178, 1		; visa id: 4065
  %3180 = add i64 %.in401, %3179		; visa id: 4066
  %3181 = ashr i64 %3164, 31		; visa id: 4067
  %3182 = bitcast i64 %3181 to <2 x i32>		; visa id: 4068
  %3183 = extractelement <2 x i32> %3182, i32 0		; visa id: 4072
  %3184 = extractelement <2 x i32> %3182, i32 1		; visa id: 4072
  %3185 = and i32 %3183, -2		; visa id: 4072
  %3186 = insertelement <2 x i32> undef, i32 %3185, i32 0		; visa id: 4073
  %3187 = insertelement <2 x i32> %3186, i32 %3184, i32 1		; visa id: 4074
  %3188 = bitcast <2 x i32> %3187 to i64		; visa id: 4075
  %3189 = add i64 %3180, %3188		; visa id: 4079
  %3190 = inttoptr i64 %3189 to i16 addrspace(4)*		; visa id: 4080
  %3191 = addrspacecast i16 addrspace(4)* %3190 to i16 addrspace(1)*		; visa id: 4080
  %3192 = load i16, i16 addrspace(1)* %3191, align 2		; visa id: 4081
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4083
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4083
  %3193 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4083
  %3194 = insertelement <2 x i32> %3193, i32 %3054, i64 1		; visa id: 4084
  %3195 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4085
  store <2 x i32> %3194, <2 x i32>* %3195, align 4, !noalias !635		; visa id: 4085
  br label %._crit_edge274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4087

._crit_edge274:                                   ; preds = %._crit_edge274.._crit_edge274_crit_edge, %3163
; BB309 :
  %3196 = phi i32 [ 0, %3163 ], [ %3205, %._crit_edge274.._crit_edge274_crit_edge ]
  %3197 = zext i32 %3196 to i64		; visa id: 4088
  %3198 = shl nuw nsw i64 %3197, 2		; visa id: 4089
  %3199 = add i64 %124, %3198		; visa id: 4090
  %3200 = inttoptr i64 %3199 to i32*		; visa id: 4091
  %3201 = load i32, i32* %3200, align 4, !noalias !635		; visa id: 4091
  %3202 = add i64 %119, %3198		; visa id: 4092
  %3203 = inttoptr i64 %3202 to i32*		; visa id: 4093
  store i32 %3201, i32* %3203, align 4, !alias.scope !635		; visa id: 4093
  %3204 = icmp eq i32 %3196, 0		; visa id: 4094
  br i1 %3204, label %._crit_edge274.._crit_edge274_crit_edge, label %3206, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4095

._crit_edge274.._crit_edge274_crit_edge:          ; preds = %._crit_edge274
; BB310 :
  %3205 = add nuw nsw i32 %3196, 1, !spirv.Decorations !631		; visa id: 4097
  br label %._crit_edge274, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4098

3206:                                             ; preds = %._crit_edge274
; BB311 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4100
  %3207 = load i64, i64* %120, align 8		; visa id: 4100
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4101
  %3208 = ashr i64 %3207, 32		; visa id: 4101
  %3209 = bitcast i64 %3208 to <2 x i32>		; visa id: 4102
  %3210 = extractelement <2 x i32> %3209, i32 0		; visa id: 4106
  %3211 = extractelement <2 x i32> %3209, i32 1		; visa id: 4106
  %3212 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3210, i32 %3211, i32 %44, i32 %45)
  %3213 = extractvalue { i32, i32 } %3212, 0		; visa id: 4106
  %3214 = extractvalue { i32, i32 } %3212, 1		; visa id: 4106
  %3215 = insertelement <2 x i32> undef, i32 %3213, i32 0		; visa id: 4113
  %3216 = insertelement <2 x i32> %3215, i32 %3214, i32 1		; visa id: 4114
  %3217 = bitcast <2 x i32> %3216 to i64		; visa id: 4115
  %3218 = bitcast i64 %3207 to <2 x i32>		; visa id: 4119
  %3219 = extractelement <2 x i32> %3218, i32 0		; visa id: 4121
  %3220 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3219, i32 1
  %3221 = bitcast <2 x i32> %3220 to i64		; visa id: 4121
  %3222 = shl i64 %3217, 1		; visa id: 4122
  %3223 = add i64 %.in400, %3222		; visa id: 4123
  %3224 = ashr exact i64 %3221, 31		; visa id: 4124
  %3225 = add i64 %3223, %3224		; visa id: 4125
  %3226 = inttoptr i64 %3225 to i16 addrspace(4)*		; visa id: 4126
  %3227 = addrspacecast i16 addrspace(4)* %3226 to i16 addrspace(1)*		; visa id: 4126
  %3228 = load i16, i16 addrspace(1)* %3227, align 2		; visa id: 4127
  %3229 = zext i16 %3192 to i32		; visa id: 4129
  %3230 = shl nuw i32 %3229, 16, !spirv.Decorations !639		; visa id: 4130
  %3231 = bitcast i32 %3230 to float
  %3232 = zext i16 %3228 to i32		; visa id: 4131
  %3233 = shl nuw i32 %3232, 16, !spirv.Decorations !639		; visa id: 4132
  %3234 = bitcast i32 %3233 to float
  %3235 = fmul reassoc nsz arcp contract float %3231, %3234, !spirv.Decorations !618
  %3236 = fadd reassoc nsz arcp contract float %3235, %.sroa.98.1, !spirv.Decorations !618		; visa id: 4133
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4134

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %3206
; BB312 :
  %.sroa.98.2 = phi float [ %3236, %3206 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  %3237 = icmp slt i32 %315, %const_reg_dword
  %3238 = icmp slt i32 %3054, %const_reg_dword1		; visa id: 4135
  %3239 = and i1 %3237, %3238		; visa id: 4136
  br i1 %3239, label %3240, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4138

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
; BB:
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3240:                                             ; preds = %._crit_edge.1.8
; BB314 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4140
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4140
  %3241 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 4140
  %3242 = insertelement <2 x i32> %3241, i32 %113, i64 1		; visa id: 4141
  %3243 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4142
  store <2 x i32> %3242, <2 x i32>* %3243, align 4, !noalias !625		; visa id: 4142
  br label %._crit_edge275, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4144

._crit_edge275:                                   ; preds = %._crit_edge275.._crit_edge275_crit_edge, %3240
; BB315 :
  %3244 = phi i32 [ 0, %3240 ], [ %3253, %._crit_edge275.._crit_edge275_crit_edge ]
  %3245 = zext i32 %3244 to i64		; visa id: 4145
  %3246 = shl nuw nsw i64 %3245, 2		; visa id: 4146
  %3247 = add i64 %133, %3246		; visa id: 4147
  %3248 = inttoptr i64 %3247 to i32*		; visa id: 4148
  %3249 = load i32, i32* %3248, align 4, !noalias !625		; visa id: 4148
  %3250 = add i64 %128, %3246		; visa id: 4149
  %3251 = inttoptr i64 %3250 to i32*		; visa id: 4150
  store i32 %3249, i32* %3251, align 4, !alias.scope !625		; visa id: 4150
  %3252 = icmp eq i32 %3244, 0		; visa id: 4151
  br i1 %3252, label %._crit_edge275.._crit_edge275_crit_edge, label %3254, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4152

._crit_edge275.._crit_edge275_crit_edge:          ; preds = %._crit_edge275
; BB316 :
  %3253 = add nuw nsw i32 %3244, 1, !spirv.Decorations !631		; visa id: 4154
  br label %._crit_edge275, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4155

3254:                                             ; preds = %._crit_edge275
; BB317 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4157
  %3255 = load i64, i64* %129, align 8		; visa id: 4157
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4158
  %3256 = bitcast i64 %3255 to <2 x i32>		; visa id: 4158
  %3257 = extractelement <2 x i32> %3256, i32 0		; visa id: 4160
  %3258 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3257, i32 1
  %3259 = bitcast <2 x i32> %3258 to i64		; visa id: 4160
  %3260 = ashr exact i64 %3259, 32		; visa id: 4161
  %3261 = bitcast i64 %3260 to <2 x i32>		; visa id: 4162
  %3262 = extractelement <2 x i32> %3261, i32 0		; visa id: 4166
  %3263 = extractelement <2 x i32> %3261, i32 1		; visa id: 4166
  %3264 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3262, i32 %3263, i32 %41, i32 %42)
  %3265 = extractvalue { i32, i32 } %3264, 0		; visa id: 4166
  %3266 = extractvalue { i32, i32 } %3264, 1		; visa id: 4166
  %3267 = insertelement <2 x i32> undef, i32 %3265, i32 0		; visa id: 4173
  %3268 = insertelement <2 x i32> %3267, i32 %3266, i32 1		; visa id: 4174
  %3269 = bitcast <2 x i32> %3268 to i64		; visa id: 4175
  %3270 = shl i64 %3269, 1		; visa id: 4179
  %3271 = add i64 %.in401, %3270		; visa id: 4180
  %3272 = ashr i64 %3255, 31		; visa id: 4181
  %3273 = bitcast i64 %3272 to <2 x i32>		; visa id: 4182
  %3274 = extractelement <2 x i32> %3273, i32 0		; visa id: 4186
  %3275 = extractelement <2 x i32> %3273, i32 1		; visa id: 4186
  %3276 = and i32 %3274, -2		; visa id: 4186
  %3277 = insertelement <2 x i32> undef, i32 %3276, i32 0		; visa id: 4187
  %3278 = insertelement <2 x i32> %3277, i32 %3275, i32 1		; visa id: 4188
  %3279 = bitcast <2 x i32> %3278 to i64		; visa id: 4189
  %3280 = add i64 %3271, %3279		; visa id: 4193
  %3281 = inttoptr i64 %3280 to i16 addrspace(4)*		; visa id: 4194
  %3282 = addrspacecast i16 addrspace(4)* %3281 to i16 addrspace(1)*		; visa id: 4194
  %3283 = load i16, i16 addrspace(1)* %3282, align 2		; visa id: 4195
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4197
  %3284 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4197
  %3285 = insertelement <2 x i32> %3284, i32 %3054, i64 1		; visa id: 4198
  %3286 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4199
  store <2 x i32> %3285, <2 x i32>* %3286, align 4, !noalias !635		; visa id: 4199
  br label %._crit_edge276, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4201

._crit_edge276:                                   ; preds = %._crit_edge276.._crit_edge276_crit_edge, %3254
; BB318 :
  %3287 = phi i32 [ 0, %3254 ], [ %3296, %._crit_edge276.._crit_edge276_crit_edge ]
  %3288 = zext i32 %3287 to i64		; visa id: 4202
  %3289 = shl nuw nsw i64 %3288, 2		; visa id: 4203
  %3290 = add i64 %124, %3289		; visa id: 4204
  %3291 = inttoptr i64 %3290 to i32*		; visa id: 4205
  %3292 = load i32, i32* %3291, align 4, !noalias !635		; visa id: 4205
  %3293 = add i64 %119, %3289		; visa id: 4206
  %3294 = inttoptr i64 %3293 to i32*		; visa id: 4207
  store i32 %3292, i32* %3294, align 4, !alias.scope !635		; visa id: 4207
  %3295 = icmp eq i32 %3287, 0		; visa id: 4208
  br i1 %3295, label %._crit_edge276.._crit_edge276_crit_edge, label %3297, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4209

._crit_edge276.._crit_edge276_crit_edge:          ; preds = %._crit_edge276
; BB319 :
  %3296 = add nuw nsw i32 %3287, 1, !spirv.Decorations !631		; visa id: 4211
  br label %._crit_edge276, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4212

3297:                                             ; preds = %._crit_edge276
; BB320 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4214
  %3298 = load i64, i64* %120, align 8		; visa id: 4214
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4215
  %3299 = ashr i64 %3298, 32		; visa id: 4215
  %3300 = bitcast i64 %3299 to <2 x i32>		; visa id: 4216
  %3301 = extractelement <2 x i32> %3300, i32 0		; visa id: 4220
  %3302 = extractelement <2 x i32> %3300, i32 1		; visa id: 4220
  %3303 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3301, i32 %3302, i32 %44, i32 %45)
  %3304 = extractvalue { i32, i32 } %3303, 0		; visa id: 4220
  %3305 = extractvalue { i32, i32 } %3303, 1		; visa id: 4220
  %3306 = insertelement <2 x i32> undef, i32 %3304, i32 0		; visa id: 4227
  %3307 = insertelement <2 x i32> %3306, i32 %3305, i32 1		; visa id: 4228
  %3308 = bitcast <2 x i32> %3307 to i64		; visa id: 4229
  %3309 = bitcast i64 %3298 to <2 x i32>		; visa id: 4233
  %3310 = extractelement <2 x i32> %3309, i32 0		; visa id: 4235
  %3311 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3310, i32 1
  %3312 = bitcast <2 x i32> %3311 to i64		; visa id: 4235
  %3313 = shl i64 %3308, 1		; visa id: 4236
  %3314 = add i64 %.in400, %3313		; visa id: 4237
  %3315 = ashr exact i64 %3312, 31		; visa id: 4238
  %3316 = add i64 %3314, %3315		; visa id: 4239
  %3317 = inttoptr i64 %3316 to i16 addrspace(4)*		; visa id: 4240
  %3318 = addrspacecast i16 addrspace(4)* %3317 to i16 addrspace(1)*		; visa id: 4240
  %3319 = load i16, i16 addrspace(1)* %3318, align 2		; visa id: 4241
  %3320 = zext i16 %3283 to i32		; visa id: 4243
  %3321 = shl nuw i32 %3320, 16, !spirv.Decorations !639		; visa id: 4244
  %3322 = bitcast i32 %3321 to float
  %3323 = zext i16 %3319 to i32		; visa id: 4245
  %3324 = shl nuw i32 %3323, 16, !spirv.Decorations !639		; visa id: 4246
  %3325 = bitcast i32 %3324 to float
  %3326 = fmul reassoc nsz arcp contract float %3322, %3325, !spirv.Decorations !618
  %3327 = fadd reassoc nsz arcp contract float %3326, %.sroa.162.1, !spirv.Decorations !618		; visa id: 4247
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4248

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %3297
; BB321 :
  %.sroa.162.2 = phi float [ %3327, %3297 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  %3328 = icmp slt i32 %407, %const_reg_dword
  %3329 = icmp slt i32 %3054, %const_reg_dword1		; visa id: 4249
  %3330 = and i1 %3328, %3329		; visa id: 4250
  br i1 %3330, label %3331, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4252

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
; BB:
  br label %.preheader.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3331:                                             ; preds = %._crit_edge.2.8
; BB323 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4254
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4254
  %3332 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 4254
  %3333 = insertelement <2 x i32> %3332, i32 %113, i64 1		; visa id: 4255
  %3334 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4256
  store <2 x i32> %3333, <2 x i32>* %3334, align 4, !noalias !625		; visa id: 4256
  br label %._crit_edge277, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4258

._crit_edge277:                                   ; preds = %._crit_edge277.._crit_edge277_crit_edge, %3331
; BB324 :
  %3335 = phi i32 [ 0, %3331 ], [ %3344, %._crit_edge277.._crit_edge277_crit_edge ]
  %3336 = zext i32 %3335 to i64		; visa id: 4259
  %3337 = shl nuw nsw i64 %3336, 2		; visa id: 4260
  %3338 = add i64 %133, %3337		; visa id: 4261
  %3339 = inttoptr i64 %3338 to i32*		; visa id: 4262
  %3340 = load i32, i32* %3339, align 4, !noalias !625		; visa id: 4262
  %3341 = add i64 %128, %3337		; visa id: 4263
  %3342 = inttoptr i64 %3341 to i32*		; visa id: 4264
  store i32 %3340, i32* %3342, align 4, !alias.scope !625		; visa id: 4264
  %3343 = icmp eq i32 %3335, 0		; visa id: 4265
  br i1 %3343, label %._crit_edge277.._crit_edge277_crit_edge, label %3345, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4266

._crit_edge277.._crit_edge277_crit_edge:          ; preds = %._crit_edge277
; BB325 :
  %3344 = add nuw nsw i32 %3335, 1, !spirv.Decorations !631		; visa id: 4268
  br label %._crit_edge277, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4269

3345:                                             ; preds = %._crit_edge277
; BB326 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4271
  %3346 = load i64, i64* %129, align 8		; visa id: 4271
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4272
  %3347 = bitcast i64 %3346 to <2 x i32>		; visa id: 4272
  %3348 = extractelement <2 x i32> %3347, i32 0		; visa id: 4274
  %3349 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3348, i32 1
  %3350 = bitcast <2 x i32> %3349 to i64		; visa id: 4274
  %3351 = ashr exact i64 %3350, 32		; visa id: 4275
  %3352 = bitcast i64 %3351 to <2 x i32>		; visa id: 4276
  %3353 = extractelement <2 x i32> %3352, i32 0		; visa id: 4280
  %3354 = extractelement <2 x i32> %3352, i32 1		; visa id: 4280
  %3355 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3353, i32 %3354, i32 %41, i32 %42)
  %3356 = extractvalue { i32, i32 } %3355, 0		; visa id: 4280
  %3357 = extractvalue { i32, i32 } %3355, 1		; visa id: 4280
  %3358 = insertelement <2 x i32> undef, i32 %3356, i32 0		; visa id: 4287
  %3359 = insertelement <2 x i32> %3358, i32 %3357, i32 1		; visa id: 4288
  %3360 = bitcast <2 x i32> %3359 to i64		; visa id: 4289
  %3361 = shl i64 %3360, 1		; visa id: 4293
  %3362 = add i64 %.in401, %3361		; visa id: 4294
  %3363 = ashr i64 %3346, 31		; visa id: 4295
  %3364 = bitcast i64 %3363 to <2 x i32>		; visa id: 4296
  %3365 = extractelement <2 x i32> %3364, i32 0		; visa id: 4300
  %3366 = extractelement <2 x i32> %3364, i32 1		; visa id: 4300
  %3367 = and i32 %3365, -2		; visa id: 4300
  %3368 = insertelement <2 x i32> undef, i32 %3367, i32 0		; visa id: 4301
  %3369 = insertelement <2 x i32> %3368, i32 %3366, i32 1		; visa id: 4302
  %3370 = bitcast <2 x i32> %3369 to i64		; visa id: 4303
  %3371 = add i64 %3362, %3370		; visa id: 4307
  %3372 = inttoptr i64 %3371 to i16 addrspace(4)*		; visa id: 4308
  %3373 = addrspacecast i16 addrspace(4)* %3372 to i16 addrspace(1)*		; visa id: 4308
  %3374 = load i16, i16 addrspace(1)* %3373, align 2		; visa id: 4309
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4311
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4311
  %3375 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4311
  %3376 = insertelement <2 x i32> %3375, i32 %3054, i64 1		; visa id: 4312
  %3377 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4313
  store <2 x i32> %3376, <2 x i32>* %3377, align 4, !noalias !635		; visa id: 4313
  br label %._crit_edge278, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4315

._crit_edge278:                                   ; preds = %._crit_edge278.._crit_edge278_crit_edge, %3345
; BB327 :
  %3378 = phi i32 [ 0, %3345 ], [ %3387, %._crit_edge278.._crit_edge278_crit_edge ]
  %3379 = zext i32 %3378 to i64		; visa id: 4316
  %3380 = shl nuw nsw i64 %3379, 2		; visa id: 4317
  %3381 = add i64 %124, %3380		; visa id: 4318
  %3382 = inttoptr i64 %3381 to i32*		; visa id: 4319
  %3383 = load i32, i32* %3382, align 4, !noalias !635		; visa id: 4319
  %3384 = add i64 %119, %3380		; visa id: 4320
  %3385 = inttoptr i64 %3384 to i32*		; visa id: 4321
  store i32 %3383, i32* %3385, align 4, !alias.scope !635		; visa id: 4321
  %3386 = icmp eq i32 %3378, 0		; visa id: 4322
  br i1 %3386, label %._crit_edge278.._crit_edge278_crit_edge, label %3388, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4323

._crit_edge278.._crit_edge278_crit_edge:          ; preds = %._crit_edge278
; BB328 :
  %3387 = add nuw nsw i32 %3378, 1, !spirv.Decorations !631		; visa id: 4325
  br label %._crit_edge278, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4326

3388:                                             ; preds = %._crit_edge278
; BB329 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4328
  %3389 = load i64, i64* %120, align 8		; visa id: 4328
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4329
  %3390 = ashr i64 %3389, 32		; visa id: 4329
  %3391 = bitcast i64 %3390 to <2 x i32>		; visa id: 4330
  %3392 = extractelement <2 x i32> %3391, i32 0		; visa id: 4334
  %3393 = extractelement <2 x i32> %3391, i32 1		; visa id: 4334
  %3394 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3392, i32 %3393, i32 %44, i32 %45)
  %3395 = extractvalue { i32, i32 } %3394, 0		; visa id: 4334
  %3396 = extractvalue { i32, i32 } %3394, 1		; visa id: 4334
  %3397 = insertelement <2 x i32> undef, i32 %3395, i32 0		; visa id: 4341
  %3398 = insertelement <2 x i32> %3397, i32 %3396, i32 1		; visa id: 4342
  %3399 = bitcast <2 x i32> %3398 to i64		; visa id: 4343
  %3400 = bitcast i64 %3389 to <2 x i32>		; visa id: 4347
  %3401 = extractelement <2 x i32> %3400, i32 0		; visa id: 4349
  %3402 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3401, i32 1
  %3403 = bitcast <2 x i32> %3402 to i64		; visa id: 4349
  %3404 = shl i64 %3399, 1		; visa id: 4350
  %3405 = add i64 %.in400, %3404		; visa id: 4351
  %3406 = ashr exact i64 %3403, 31		; visa id: 4352
  %3407 = add i64 %3405, %3406		; visa id: 4353
  %3408 = inttoptr i64 %3407 to i16 addrspace(4)*		; visa id: 4354
  %3409 = addrspacecast i16 addrspace(4)* %3408 to i16 addrspace(1)*		; visa id: 4354
  %3410 = load i16, i16 addrspace(1)* %3409, align 2		; visa id: 4355
  %3411 = zext i16 %3374 to i32		; visa id: 4357
  %3412 = shl nuw i32 %3411, 16, !spirv.Decorations !639		; visa id: 4358
  %3413 = bitcast i32 %3412 to float
  %3414 = zext i16 %3410 to i32		; visa id: 4359
  %3415 = shl nuw i32 %3414, 16, !spirv.Decorations !639		; visa id: 4360
  %3416 = bitcast i32 %3415 to float
  %3417 = fmul reassoc nsz arcp contract float %3413, %3416, !spirv.Decorations !618
  %3418 = fadd reassoc nsz arcp contract float %3417, %.sroa.226.1, !spirv.Decorations !618		; visa id: 4361
  br label %.preheader.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4362

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %3388
; BB330 :
  %.sroa.226.2 = phi float [ %3418, %3388 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %3419 = add i32 %69, 9		; visa id: 4363
  %3420 = icmp slt i32 %3419, %const_reg_dword1		; visa id: 4364
  %3421 = icmp slt i32 %65, %const_reg_dword
  %3422 = and i1 %3421, %3420		; visa id: 4365
  br i1 %3422, label %3423, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4367

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
; BB:
  br label %._crit_edge.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3423:                                             ; preds = %.preheader.8
; BB332 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4369
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4369
  %3424 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 4369
  %3425 = insertelement <2 x i32> %3424, i32 %113, i64 1		; visa id: 4370
  %3426 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4371
  store <2 x i32> %3425, <2 x i32>* %3426, align 4, !noalias !625		; visa id: 4371
  br label %._crit_edge279, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4373

._crit_edge279:                                   ; preds = %._crit_edge279.._crit_edge279_crit_edge, %3423
; BB333 :
  %3427 = phi i32 [ 0, %3423 ], [ %3436, %._crit_edge279.._crit_edge279_crit_edge ]
  %3428 = zext i32 %3427 to i64		; visa id: 4374
  %3429 = shl nuw nsw i64 %3428, 2		; visa id: 4375
  %3430 = add i64 %133, %3429		; visa id: 4376
  %3431 = inttoptr i64 %3430 to i32*		; visa id: 4377
  %3432 = load i32, i32* %3431, align 4, !noalias !625		; visa id: 4377
  %3433 = add i64 %128, %3429		; visa id: 4378
  %3434 = inttoptr i64 %3433 to i32*		; visa id: 4379
  store i32 %3432, i32* %3434, align 4, !alias.scope !625		; visa id: 4379
  %3435 = icmp eq i32 %3427, 0		; visa id: 4380
  br i1 %3435, label %._crit_edge279.._crit_edge279_crit_edge, label %3437, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4381

._crit_edge279.._crit_edge279_crit_edge:          ; preds = %._crit_edge279
; BB334 :
  %3436 = add nuw nsw i32 %3427, 1, !spirv.Decorations !631		; visa id: 4383
  br label %._crit_edge279, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4384

3437:                                             ; preds = %._crit_edge279
; BB335 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4386
  %3438 = load i64, i64* %129, align 8		; visa id: 4386
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4387
  %3439 = bitcast i64 %3438 to <2 x i32>		; visa id: 4387
  %3440 = extractelement <2 x i32> %3439, i32 0		; visa id: 4389
  %3441 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3440, i32 1
  %3442 = bitcast <2 x i32> %3441 to i64		; visa id: 4389
  %3443 = ashr exact i64 %3442, 32		; visa id: 4390
  %3444 = bitcast i64 %3443 to <2 x i32>		; visa id: 4391
  %3445 = extractelement <2 x i32> %3444, i32 0		; visa id: 4395
  %3446 = extractelement <2 x i32> %3444, i32 1		; visa id: 4395
  %3447 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3445, i32 %3446, i32 %41, i32 %42)
  %3448 = extractvalue { i32, i32 } %3447, 0		; visa id: 4395
  %3449 = extractvalue { i32, i32 } %3447, 1		; visa id: 4395
  %3450 = insertelement <2 x i32> undef, i32 %3448, i32 0		; visa id: 4402
  %3451 = insertelement <2 x i32> %3450, i32 %3449, i32 1		; visa id: 4403
  %3452 = bitcast <2 x i32> %3451 to i64		; visa id: 4404
  %3453 = shl i64 %3452, 1		; visa id: 4408
  %3454 = add i64 %.in401, %3453		; visa id: 4409
  %3455 = ashr i64 %3438, 31		; visa id: 4410
  %3456 = bitcast i64 %3455 to <2 x i32>		; visa id: 4411
  %3457 = extractelement <2 x i32> %3456, i32 0		; visa id: 4415
  %3458 = extractelement <2 x i32> %3456, i32 1		; visa id: 4415
  %3459 = and i32 %3457, -2		; visa id: 4415
  %3460 = insertelement <2 x i32> undef, i32 %3459, i32 0		; visa id: 4416
  %3461 = insertelement <2 x i32> %3460, i32 %3458, i32 1		; visa id: 4417
  %3462 = bitcast <2 x i32> %3461 to i64		; visa id: 4418
  %3463 = add i64 %3454, %3462		; visa id: 4422
  %3464 = inttoptr i64 %3463 to i16 addrspace(4)*		; visa id: 4423
  %3465 = addrspacecast i16 addrspace(4)* %3464 to i16 addrspace(1)*		; visa id: 4423
  %3466 = load i16, i16 addrspace(1)* %3465, align 2		; visa id: 4424
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4426
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4426
  %3467 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4426
  %3468 = insertelement <2 x i32> %3467, i32 %3419, i64 1		; visa id: 4427
  %3469 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4428
  store <2 x i32> %3468, <2 x i32>* %3469, align 4, !noalias !635		; visa id: 4428
  br label %._crit_edge280, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4430

._crit_edge280:                                   ; preds = %._crit_edge280.._crit_edge280_crit_edge, %3437
; BB336 :
  %3470 = phi i32 [ 0, %3437 ], [ %3479, %._crit_edge280.._crit_edge280_crit_edge ]
  %3471 = zext i32 %3470 to i64		; visa id: 4431
  %3472 = shl nuw nsw i64 %3471, 2		; visa id: 4432
  %3473 = add i64 %124, %3472		; visa id: 4433
  %3474 = inttoptr i64 %3473 to i32*		; visa id: 4434
  %3475 = load i32, i32* %3474, align 4, !noalias !635		; visa id: 4434
  %3476 = add i64 %119, %3472		; visa id: 4435
  %3477 = inttoptr i64 %3476 to i32*		; visa id: 4436
  store i32 %3475, i32* %3477, align 4, !alias.scope !635		; visa id: 4436
  %3478 = icmp eq i32 %3470, 0		; visa id: 4437
  br i1 %3478, label %._crit_edge280.._crit_edge280_crit_edge, label %3480, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4438

._crit_edge280.._crit_edge280_crit_edge:          ; preds = %._crit_edge280
; BB337 :
  %3479 = add nuw nsw i32 %3470, 1, !spirv.Decorations !631		; visa id: 4440
  br label %._crit_edge280, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4441

3480:                                             ; preds = %._crit_edge280
; BB338 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4443
  %3481 = load i64, i64* %120, align 8		; visa id: 4443
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4444
  %3482 = ashr i64 %3481, 32		; visa id: 4444
  %3483 = bitcast i64 %3482 to <2 x i32>		; visa id: 4445
  %3484 = extractelement <2 x i32> %3483, i32 0		; visa id: 4449
  %3485 = extractelement <2 x i32> %3483, i32 1		; visa id: 4449
  %3486 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3484, i32 %3485, i32 %44, i32 %45)
  %3487 = extractvalue { i32, i32 } %3486, 0		; visa id: 4449
  %3488 = extractvalue { i32, i32 } %3486, 1		; visa id: 4449
  %3489 = insertelement <2 x i32> undef, i32 %3487, i32 0		; visa id: 4456
  %3490 = insertelement <2 x i32> %3489, i32 %3488, i32 1		; visa id: 4457
  %3491 = bitcast <2 x i32> %3490 to i64		; visa id: 4458
  %3492 = bitcast i64 %3481 to <2 x i32>		; visa id: 4462
  %3493 = extractelement <2 x i32> %3492, i32 0		; visa id: 4464
  %3494 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3493, i32 1
  %3495 = bitcast <2 x i32> %3494 to i64		; visa id: 4464
  %3496 = shl i64 %3491, 1		; visa id: 4465
  %3497 = add i64 %.in400, %3496		; visa id: 4466
  %3498 = ashr exact i64 %3495, 31		; visa id: 4467
  %3499 = add i64 %3497, %3498		; visa id: 4468
  %3500 = inttoptr i64 %3499 to i16 addrspace(4)*		; visa id: 4469
  %3501 = addrspacecast i16 addrspace(4)* %3500 to i16 addrspace(1)*		; visa id: 4469
  %3502 = load i16, i16 addrspace(1)* %3501, align 2		; visa id: 4470
  %3503 = zext i16 %3466 to i32		; visa id: 4472
  %3504 = shl nuw i32 %3503, 16, !spirv.Decorations !639		; visa id: 4473
  %3505 = bitcast i32 %3504 to float
  %3506 = zext i16 %3502 to i32		; visa id: 4474
  %3507 = shl nuw i32 %3506, 16, !spirv.Decorations !639		; visa id: 4475
  %3508 = bitcast i32 %3507 to float
  %3509 = fmul reassoc nsz arcp contract float %3505, %3508, !spirv.Decorations !618
  %3510 = fadd reassoc nsz arcp contract float %3509, %.sroa.38.1, !spirv.Decorations !618		; visa id: 4476
  br label %._crit_edge.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4477

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %3480
; BB339 :
  %.sroa.38.2 = phi float [ %3510, %3480 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  %3511 = icmp slt i32 %223, %const_reg_dword
  %3512 = icmp slt i32 %3419, %const_reg_dword1		; visa id: 4478
  %3513 = and i1 %3511, %3512		; visa id: 4479
  br i1 %3513, label %3514, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4481

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
; BB:
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3514:                                             ; preds = %._crit_edge.9
; BB341 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4483
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4483
  %3515 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 4483
  %3516 = insertelement <2 x i32> %3515, i32 %113, i64 1		; visa id: 4484
  %3517 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4485
  store <2 x i32> %3516, <2 x i32>* %3517, align 4, !noalias !625		; visa id: 4485
  br label %._crit_edge281, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4487

._crit_edge281:                                   ; preds = %._crit_edge281.._crit_edge281_crit_edge, %3514
; BB342 :
  %3518 = phi i32 [ 0, %3514 ], [ %3527, %._crit_edge281.._crit_edge281_crit_edge ]
  %3519 = zext i32 %3518 to i64		; visa id: 4488
  %3520 = shl nuw nsw i64 %3519, 2		; visa id: 4489
  %3521 = add i64 %133, %3520		; visa id: 4490
  %3522 = inttoptr i64 %3521 to i32*		; visa id: 4491
  %3523 = load i32, i32* %3522, align 4, !noalias !625		; visa id: 4491
  %3524 = add i64 %128, %3520		; visa id: 4492
  %3525 = inttoptr i64 %3524 to i32*		; visa id: 4493
  store i32 %3523, i32* %3525, align 4, !alias.scope !625		; visa id: 4493
  %3526 = icmp eq i32 %3518, 0		; visa id: 4494
  br i1 %3526, label %._crit_edge281.._crit_edge281_crit_edge, label %3528, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4495

._crit_edge281.._crit_edge281_crit_edge:          ; preds = %._crit_edge281
; BB343 :
  %3527 = add nuw nsw i32 %3518, 1, !spirv.Decorations !631		; visa id: 4497
  br label %._crit_edge281, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4498

3528:                                             ; preds = %._crit_edge281
; BB344 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4500
  %3529 = load i64, i64* %129, align 8		; visa id: 4500
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4501
  %3530 = bitcast i64 %3529 to <2 x i32>		; visa id: 4501
  %3531 = extractelement <2 x i32> %3530, i32 0		; visa id: 4503
  %3532 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3531, i32 1
  %3533 = bitcast <2 x i32> %3532 to i64		; visa id: 4503
  %3534 = ashr exact i64 %3533, 32		; visa id: 4504
  %3535 = bitcast i64 %3534 to <2 x i32>		; visa id: 4505
  %3536 = extractelement <2 x i32> %3535, i32 0		; visa id: 4509
  %3537 = extractelement <2 x i32> %3535, i32 1		; visa id: 4509
  %3538 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3536, i32 %3537, i32 %41, i32 %42)
  %3539 = extractvalue { i32, i32 } %3538, 0		; visa id: 4509
  %3540 = extractvalue { i32, i32 } %3538, 1		; visa id: 4509
  %3541 = insertelement <2 x i32> undef, i32 %3539, i32 0		; visa id: 4516
  %3542 = insertelement <2 x i32> %3541, i32 %3540, i32 1		; visa id: 4517
  %3543 = bitcast <2 x i32> %3542 to i64		; visa id: 4518
  %3544 = shl i64 %3543, 1		; visa id: 4522
  %3545 = add i64 %.in401, %3544		; visa id: 4523
  %3546 = ashr i64 %3529, 31		; visa id: 4524
  %3547 = bitcast i64 %3546 to <2 x i32>		; visa id: 4525
  %3548 = extractelement <2 x i32> %3547, i32 0		; visa id: 4529
  %3549 = extractelement <2 x i32> %3547, i32 1		; visa id: 4529
  %3550 = and i32 %3548, -2		; visa id: 4529
  %3551 = insertelement <2 x i32> undef, i32 %3550, i32 0		; visa id: 4530
  %3552 = insertelement <2 x i32> %3551, i32 %3549, i32 1		; visa id: 4531
  %3553 = bitcast <2 x i32> %3552 to i64		; visa id: 4532
  %3554 = add i64 %3545, %3553		; visa id: 4536
  %3555 = inttoptr i64 %3554 to i16 addrspace(4)*		; visa id: 4537
  %3556 = addrspacecast i16 addrspace(4)* %3555 to i16 addrspace(1)*		; visa id: 4537
  %3557 = load i16, i16 addrspace(1)* %3556, align 2		; visa id: 4538
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4540
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4540
  %3558 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4540
  %3559 = insertelement <2 x i32> %3558, i32 %3419, i64 1		; visa id: 4541
  %3560 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4542
  store <2 x i32> %3559, <2 x i32>* %3560, align 4, !noalias !635		; visa id: 4542
  br label %._crit_edge282, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4544

._crit_edge282:                                   ; preds = %._crit_edge282.._crit_edge282_crit_edge, %3528
; BB345 :
  %3561 = phi i32 [ 0, %3528 ], [ %3570, %._crit_edge282.._crit_edge282_crit_edge ]
  %3562 = zext i32 %3561 to i64		; visa id: 4545
  %3563 = shl nuw nsw i64 %3562, 2		; visa id: 4546
  %3564 = add i64 %124, %3563		; visa id: 4547
  %3565 = inttoptr i64 %3564 to i32*		; visa id: 4548
  %3566 = load i32, i32* %3565, align 4, !noalias !635		; visa id: 4548
  %3567 = add i64 %119, %3563		; visa id: 4549
  %3568 = inttoptr i64 %3567 to i32*		; visa id: 4550
  store i32 %3566, i32* %3568, align 4, !alias.scope !635		; visa id: 4550
  %3569 = icmp eq i32 %3561, 0		; visa id: 4551
  br i1 %3569, label %._crit_edge282.._crit_edge282_crit_edge, label %3571, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4552

._crit_edge282.._crit_edge282_crit_edge:          ; preds = %._crit_edge282
; BB346 :
  %3570 = add nuw nsw i32 %3561, 1, !spirv.Decorations !631		; visa id: 4554
  br label %._crit_edge282, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4555

3571:                                             ; preds = %._crit_edge282
; BB347 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4557
  %3572 = load i64, i64* %120, align 8		; visa id: 4557
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4558
  %3573 = ashr i64 %3572, 32		; visa id: 4558
  %3574 = bitcast i64 %3573 to <2 x i32>		; visa id: 4559
  %3575 = extractelement <2 x i32> %3574, i32 0		; visa id: 4563
  %3576 = extractelement <2 x i32> %3574, i32 1		; visa id: 4563
  %3577 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3575, i32 %3576, i32 %44, i32 %45)
  %3578 = extractvalue { i32, i32 } %3577, 0		; visa id: 4563
  %3579 = extractvalue { i32, i32 } %3577, 1		; visa id: 4563
  %3580 = insertelement <2 x i32> undef, i32 %3578, i32 0		; visa id: 4570
  %3581 = insertelement <2 x i32> %3580, i32 %3579, i32 1		; visa id: 4571
  %3582 = bitcast <2 x i32> %3581 to i64		; visa id: 4572
  %3583 = bitcast i64 %3572 to <2 x i32>		; visa id: 4576
  %3584 = extractelement <2 x i32> %3583, i32 0		; visa id: 4578
  %3585 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3584, i32 1
  %3586 = bitcast <2 x i32> %3585 to i64		; visa id: 4578
  %3587 = shl i64 %3582, 1		; visa id: 4579
  %3588 = add i64 %.in400, %3587		; visa id: 4580
  %3589 = ashr exact i64 %3586, 31		; visa id: 4581
  %3590 = add i64 %3588, %3589		; visa id: 4582
  %3591 = inttoptr i64 %3590 to i16 addrspace(4)*		; visa id: 4583
  %3592 = addrspacecast i16 addrspace(4)* %3591 to i16 addrspace(1)*		; visa id: 4583
  %3593 = load i16, i16 addrspace(1)* %3592, align 2		; visa id: 4584
  %3594 = zext i16 %3557 to i32		; visa id: 4586
  %3595 = shl nuw i32 %3594, 16, !spirv.Decorations !639		; visa id: 4587
  %3596 = bitcast i32 %3595 to float
  %3597 = zext i16 %3593 to i32		; visa id: 4588
  %3598 = shl nuw i32 %3597, 16, !spirv.Decorations !639		; visa id: 4589
  %3599 = bitcast i32 %3598 to float
  %3600 = fmul reassoc nsz arcp contract float %3596, %3599, !spirv.Decorations !618
  %3601 = fadd reassoc nsz arcp contract float %3600, %.sroa.102.1, !spirv.Decorations !618		; visa id: 4590
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4591

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %3571
; BB348 :
  %.sroa.102.2 = phi float [ %3601, %3571 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  %3602 = icmp slt i32 %315, %const_reg_dword
  %3603 = icmp slt i32 %3419, %const_reg_dword1		; visa id: 4592
  %3604 = and i1 %3602, %3603		; visa id: 4593
  br i1 %3604, label %3605, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4595

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
; BB:
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3605:                                             ; preds = %._crit_edge.1.9
; BB350 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4597
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4597
  %3606 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 4597
  %3607 = insertelement <2 x i32> %3606, i32 %113, i64 1		; visa id: 4598
  %3608 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4599
  store <2 x i32> %3607, <2 x i32>* %3608, align 4, !noalias !625		; visa id: 4599
  br label %._crit_edge283, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4601

._crit_edge283:                                   ; preds = %._crit_edge283.._crit_edge283_crit_edge, %3605
; BB351 :
  %3609 = phi i32 [ 0, %3605 ], [ %3618, %._crit_edge283.._crit_edge283_crit_edge ]
  %3610 = zext i32 %3609 to i64		; visa id: 4602
  %3611 = shl nuw nsw i64 %3610, 2		; visa id: 4603
  %3612 = add i64 %133, %3611		; visa id: 4604
  %3613 = inttoptr i64 %3612 to i32*		; visa id: 4605
  %3614 = load i32, i32* %3613, align 4, !noalias !625		; visa id: 4605
  %3615 = add i64 %128, %3611		; visa id: 4606
  %3616 = inttoptr i64 %3615 to i32*		; visa id: 4607
  store i32 %3614, i32* %3616, align 4, !alias.scope !625		; visa id: 4607
  %3617 = icmp eq i32 %3609, 0		; visa id: 4608
  br i1 %3617, label %._crit_edge283.._crit_edge283_crit_edge, label %3619, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4609

._crit_edge283.._crit_edge283_crit_edge:          ; preds = %._crit_edge283
; BB352 :
  %3618 = add nuw nsw i32 %3609, 1, !spirv.Decorations !631		; visa id: 4611
  br label %._crit_edge283, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4612

3619:                                             ; preds = %._crit_edge283
; BB353 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4614
  %3620 = load i64, i64* %129, align 8		; visa id: 4614
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4615
  %3621 = bitcast i64 %3620 to <2 x i32>		; visa id: 4615
  %3622 = extractelement <2 x i32> %3621, i32 0		; visa id: 4617
  %3623 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3622, i32 1
  %3624 = bitcast <2 x i32> %3623 to i64		; visa id: 4617
  %3625 = ashr exact i64 %3624, 32		; visa id: 4618
  %3626 = bitcast i64 %3625 to <2 x i32>		; visa id: 4619
  %3627 = extractelement <2 x i32> %3626, i32 0		; visa id: 4623
  %3628 = extractelement <2 x i32> %3626, i32 1		; visa id: 4623
  %3629 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3627, i32 %3628, i32 %41, i32 %42)
  %3630 = extractvalue { i32, i32 } %3629, 0		; visa id: 4623
  %3631 = extractvalue { i32, i32 } %3629, 1		; visa id: 4623
  %3632 = insertelement <2 x i32> undef, i32 %3630, i32 0		; visa id: 4630
  %3633 = insertelement <2 x i32> %3632, i32 %3631, i32 1		; visa id: 4631
  %3634 = bitcast <2 x i32> %3633 to i64		; visa id: 4632
  %3635 = shl i64 %3634, 1		; visa id: 4636
  %3636 = add i64 %.in401, %3635		; visa id: 4637
  %3637 = ashr i64 %3620, 31		; visa id: 4638
  %3638 = bitcast i64 %3637 to <2 x i32>		; visa id: 4639
  %3639 = extractelement <2 x i32> %3638, i32 0		; visa id: 4643
  %3640 = extractelement <2 x i32> %3638, i32 1		; visa id: 4643
  %3641 = and i32 %3639, -2		; visa id: 4643
  %3642 = insertelement <2 x i32> undef, i32 %3641, i32 0		; visa id: 4644
  %3643 = insertelement <2 x i32> %3642, i32 %3640, i32 1		; visa id: 4645
  %3644 = bitcast <2 x i32> %3643 to i64		; visa id: 4646
  %3645 = add i64 %3636, %3644		; visa id: 4650
  %3646 = inttoptr i64 %3645 to i16 addrspace(4)*		; visa id: 4651
  %3647 = addrspacecast i16 addrspace(4)* %3646 to i16 addrspace(1)*		; visa id: 4651
  %3648 = load i16, i16 addrspace(1)* %3647, align 2		; visa id: 4652
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4654
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4654
  %3649 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4654
  %3650 = insertelement <2 x i32> %3649, i32 %3419, i64 1		; visa id: 4655
  %3651 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4656
  store <2 x i32> %3650, <2 x i32>* %3651, align 4, !noalias !635		; visa id: 4656
  br label %._crit_edge284, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4658

._crit_edge284:                                   ; preds = %._crit_edge284.._crit_edge284_crit_edge, %3619
; BB354 :
  %3652 = phi i32 [ 0, %3619 ], [ %3661, %._crit_edge284.._crit_edge284_crit_edge ]
  %3653 = zext i32 %3652 to i64		; visa id: 4659
  %3654 = shl nuw nsw i64 %3653, 2		; visa id: 4660
  %3655 = add i64 %124, %3654		; visa id: 4661
  %3656 = inttoptr i64 %3655 to i32*		; visa id: 4662
  %3657 = load i32, i32* %3656, align 4, !noalias !635		; visa id: 4662
  %3658 = add i64 %119, %3654		; visa id: 4663
  %3659 = inttoptr i64 %3658 to i32*		; visa id: 4664
  store i32 %3657, i32* %3659, align 4, !alias.scope !635		; visa id: 4664
  %3660 = icmp eq i32 %3652, 0		; visa id: 4665
  br i1 %3660, label %._crit_edge284.._crit_edge284_crit_edge, label %3662, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4666

._crit_edge284.._crit_edge284_crit_edge:          ; preds = %._crit_edge284
; BB355 :
  %3661 = add nuw nsw i32 %3652, 1, !spirv.Decorations !631		; visa id: 4668
  br label %._crit_edge284, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4669

3662:                                             ; preds = %._crit_edge284
; BB356 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4671
  %3663 = load i64, i64* %120, align 8		; visa id: 4671
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4672
  %3664 = ashr i64 %3663, 32		; visa id: 4672
  %3665 = bitcast i64 %3664 to <2 x i32>		; visa id: 4673
  %3666 = extractelement <2 x i32> %3665, i32 0		; visa id: 4677
  %3667 = extractelement <2 x i32> %3665, i32 1		; visa id: 4677
  %3668 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3666, i32 %3667, i32 %44, i32 %45)
  %3669 = extractvalue { i32, i32 } %3668, 0		; visa id: 4677
  %3670 = extractvalue { i32, i32 } %3668, 1		; visa id: 4677
  %3671 = insertelement <2 x i32> undef, i32 %3669, i32 0		; visa id: 4684
  %3672 = insertelement <2 x i32> %3671, i32 %3670, i32 1		; visa id: 4685
  %3673 = bitcast <2 x i32> %3672 to i64		; visa id: 4686
  %3674 = bitcast i64 %3663 to <2 x i32>		; visa id: 4690
  %3675 = extractelement <2 x i32> %3674, i32 0		; visa id: 4692
  %3676 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3675, i32 1
  %3677 = bitcast <2 x i32> %3676 to i64		; visa id: 4692
  %3678 = shl i64 %3673, 1		; visa id: 4693
  %3679 = add i64 %.in400, %3678		; visa id: 4694
  %3680 = ashr exact i64 %3677, 31		; visa id: 4695
  %3681 = add i64 %3679, %3680		; visa id: 4696
  %3682 = inttoptr i64 %3681 to i16 addrspace(4)*		; visa id: 4697
  %3683 = addrspacecast i16 addrspace(4)* %3682 to i16 addrspace(1)*		; visa id: 4697
  %3684 = load i16, i16 addrspace(1)* %3683, align 2		; visa id: 4698
  %3685 = zext i16 %3648 to i32		; visa id: 4700
  %3686 = shl nuw i32 %3685, 16, !spirv.Decorations !639		; visa id: 4701
  %3687 = bitcast i32 %3686 to float
  %3688 = zext i16 %3684 to i32		; visa id: 4702
  %3689 = shl nuw i32 %3688, 16, !spirv.Decorations !639		; visa id: 4703
  %3690 = bitcast i32 %3689 to float
  %3691 = fmul reassoc nsz arcp contract float %3687, %3690, !spirv.Decorations !618
  %3692 = fadd reassoc nsz arcp contract float %3691, %.sroa.166.1, !spirv.Decorations !618		; visa id: 4704
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4705

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %3662
; BB357 :
  %.sroa.166.2 = phi float [ %3692, %3662 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  %3693 = icmp slt i32 %407, %const_reg_dword
  %3694 = icmp slt i32 %3419, %const_reg_dword1		; visa id: 4706
  %3695 = and i1 %3693, %3694		; visa id: 4707
  br i1 %3695, label %3696, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4709

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
; BB:
  br label %.preheader.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3696:                                             ; preds = %._crit_edge.2.9
; BB359 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4711
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4711
  %3697 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 4711
  %3698 = insertelement <2 x i32> %3697, i32 %113, i64 1		; visa id: 4712
  %3699 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4713
  store <2 x i32> %3698, <2 x i32>* %3699, align 4, !noalias !625		; visa id: 4713
  br label %._crit_edge285, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4715

._crit_edge285:                                   ; preds = %._crit_edge285.._crit_edge285_crit_edge, %3696
; BB360 :
  %3700 = phi i32 [ 0, %3696 ], [ %3709, %._crit_edge285.._crit_edge285_crit_edge ]
  %3701 = zext i32 %3700 to i64		; visa id: 4716
  %3702 = shl nuw nsw i64 %3701, 2		; visa id: 4717
  %3703 = add i64 %133, %3702		; visa id: 4718
  %3704 = inttoptr i64 %3703 to i32*		; visa id: 4719
  %3705 = load i32, i32* %3704, align 4, !noalias !625		; visa id: 4719
  %3706 = add i64 %128, %3702		; visa id: 4720
  %3707 = inttoptr i64 %3706 to i32*		; visa id: 4721
  store i32 %3705, i32* %3707, align 4, !alias.scope !625		; visa id: 4721
  %3708 = icmp eq i32 %3700, 0		; visa id: 4722
  br i1 %3708, label %._crit_edge285.._crit_edge285_crit_edge, label %3710, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4723

._crit_edge285.._crit_edge285_crit_edge:          ; preds = %._crit_edge285
; BB361 :
  %3709 = add nuw nsw i32 %3700, 1, !spirv.Decorations !631		; visa id: 4725
  br label %._crit_edge285, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4726

3710:                                             ; preds = %._crit_edge285
; BB362 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4728
  %3711 = load i64, i64* %129, align 8		; visa id: 4728
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4729
  %3712 = bitcast i64 %3711 to <2 x i32>		; visa id: 4729
  %3713 = extractelement <2 x i32> %3712, i32 0		; visa id: 4731
  %3714 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3713, i32 1
  %3715 = bitcast <2 x i32> %3714 to i64		; visa id: 4731
  %3716 = ashr exact i64 %3715, 32		; visa id: 4732
  %3717 = bitcast i64 %3716 to <2 x i32>		; visa id: 4733
  %3718 = extractelement <2 x i32> %3717, i32 0		; visa id: 4737
  %3719 = extractelement <2 x i32> %3717, i32 1		; visa id: 4737
  %3720 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3718, i32 %3719, i32 %41, i32 %42)
  %3721 = extractvalue { i32, i32 } %3720, 0		; visa id: 4737
  %3722 = extractvalue { i32, i32 } %3720, 1		; visa id: 4737
  %3723 = insertelement <2 x i32> undef, i32 %3721, i32 0		; visa id: 4744
  %3724 = insertelement <2 x i32> %3723, i32 %3722, i32 1		; visa id: 4745
  %3725 = bitcast <2 x i32> %3724 to i64		; visa id: 4746
  %3726 = shl i64 %3725, 1		; visa id: 4750
  %3727 = add i64 %.in401, %3726		; visa id: 4751
  %3728 = ashr i64 %3711, 31		; visa id: 4752
  %3729 = bitcast i64 %3728 to <2 x i32>		; visa id: 4753
  %3730 = extractelement <2 x i32> %3729, i32 0		; visa id: 4757
  %3731 = extractelement <2 x i32> %3729, i32 1		; visa id: 4757
  %3732 = and i32 %3730, -2		; visa id: 4757
  %3733 = insertelement <2 x i32> undef, i32 %3732, i32 0		; visa id: 4758
  %3734 = insertelement <2 x i32> %3733, i32 %3731, i32 1		; visa id: 4759
  %3735 = bitcast <2 x i32> %3734 to i64		; visa id: 4760
  %3736 = add i64 %3727, %3735		; visa id: 4764
  %3737 = inttoptr i64 %3736 to i16 addrspace(4)*		; visa id: 4765
  %3738 = addrspacecast i16 addrspace(4)* %3737 to i16 addrspace(1)*		; visa id: 4765
  %3739 = load i16, i16 addrspace(1)* %3738, align 2		; visa id: 4766
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4768
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4768
  %3740 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4768
  %3741 = insertelement <2 x i32> %3740, i32 %3419, i64 1		; visa id: 4769
  %3742 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4770
  store <2 x i32> %3741, <2 x i32>* %3742, align 4, !noalias !635		; visa id: 4770
  br label %._crit_edge286, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4772

._crit_edge286:                                   ; preds = %._crit_edge286.._crit_edge286_crit_edge, %3710
; BB363 :
  %3743 = phi i32 [ 0, %3710 ], [ %3752, %._crit_edge286.._crit_edge286_crit_edge ]
  %3744 = zext i32 %3743 to i64		; visa id: 4773
  %3745 = shl nuw nsw i64 %3744, 2		; visa id: 4774
  %3746 = add i64 %124, %3745		; visa id: 4775
  %3747 = inttoptr i64 %3746 to i32*		; visa id: 4776
  %3748 = load i32, i32* %3747, align 4, !noalias !635		; visa id: 4776
  %3749 = add i64 %119, %3745		; visa id: 4777
  %3750 = inttoptr i64 %3749 to i32*		; visa id: 4778
  store i32 %3748, i32* %3750, align 4, !alias.scope !635		; visa id: 4778
  %3751 = icmp eq i32 %3743, 0		; visa id: 4779
  br i1 %3751, label %._crit_edge286.._crit_edge286_crit_edge, label %3753, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4780

._crit_edge286.._crit_edge286_crit_edge:          ; preds = %._crit_edge286
; BB364 :
  %3752 = add nuw nsw i32 %3743, 1, !spirv.Decorations !631		; visa id: 4782
  br label %._crit_edge286, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4783

3753:                                             ; preds = %._crit_edge286
; BB365 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4785
  %3754 = load i64, i64* %120, align 8		; visa id: 4785
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4786
  %3755 = ashr i64 %3754, 32		; visa id: 4786
  %3756 = bitcast i64 %3755 to <2 x i32>		; visa id: 4787
  %3757 = extractelement <2 x i32> %3756, i32 0		; visa id: 4791
  %3758 = extractelement <2 x i32> %3756, i32 1		; visa id: 4791
  %3759 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3757, i32 %3758, i32 %44, i32 %45)
  %3760 = extractvalue { i32, i32 } %3759, 0		; visa id: 4791
  %3761 = extractvalue { i32, i32 } %3759, 1		; visa id: 4791
  %3762 = insertelement <2 x i32> undef, i32 %3760, i32 0		; visa id: 4798
  %3763 = insertelement <2 x i32> %3762, i32 %3761, i32 1		; visa id: 4799
  %3764 = bitcast <2 x i32> %3763 to i64		; visa id: 4800
  %3765 = bitcast i64 %3754 to <2 x i32>		; visa id: 4804
  %3766 = extractelement <2 x i32> %3765, i32 0		; visa id: 4806
  %3767 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3766, i32 1
  %3768 = bitcast <2 x i32> %3767 to i64		; visa id: 4806
  %3769 = shl i64 %3764, 1		; visa id: 4807
  %3770 = add i64 %.in400, %3769		; visa id: 4808
  %3771 = ashr exact i64 %3768, 31		; visa id: 4809
  %3772 = add i64 %3770, %3771		; visa id: 4810
  %3773 = inttoptr i64 %3772 to i16 addrspace(4)*		; visa id: 4811
  %3774 = addrspacecast i16 addrspace(4)* %3773 to i16 addrspace(1)*		; visa id: 4811
  %3775 = load i16, i16 addrspace(1)* %3774, align 2		; visa id: 4812
  %3776 = zext i16 %3739 to i32		; visa id: 4814
  %3777 = shl nuw i32 %3776, 16, !spirv.Decorations !639		; visa id: 4815
  %3778 = bitcast i32 %3777 to float
  %3779 = zext i16 %3775 to i32		; visa id: 4816
  %3780 = shl nuw i32 %3779, 16, !spirv.Decorations !639		; visa id: 4817
  %3781 = bitcast i32 %3780 to float
  %3782 = fmul reassoc nsz arcp contract float %3778, %3781, !spirv.Decorations !618
  %3783 = fadd reassoc nsz arcp contract float %3782, %.sroa.230.1, !spirv.Decorations !618		; visa id: 4818
  br label %.preheader.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4819

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %3753
; BB366 :
  %.sroa.230.2 = phi float [ %3783, %3753 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %3784 = add i32 %69, 10		; visa id: 4820
  %3785 = icmp slt i32 %3784, %const_reg_dword1		; visa id: 4821
  %3786 = icmp slt i32 %65, %const_reg_dword
  %3787 = and i1 %3786, %3785		; visa id: 4822
  br i1 %3787, label %3788, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4824

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
; BB:
  br label %._crit_edge.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3788:                                             ; preds = %.preheader.9
; BB368 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4826
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4826
  %3789 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 4826
  %3790 = insertelement <2 x i32> %3789, i32 %113, i64 1		; visa id: 4827
  %3791 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4828
  store <2 x i32> %3790, <2 x i32>* %3791, align 4, !noalias !625		; visa id: 4828
  br label %._crit_edge287, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4830

._crit_edge287:                                   ; preds = %._crit_edge287.._crit_edge287_crit_edge, %3788
; BB369 :
  %3792 = phi i32 [ 0, %3788 ], [ %3801, %._crit_edge287.._crit_edge287_crit_edge ]
  %3793 = zext i32 %3792 to i64		; visa id: 4831
  %3794 = shl nuw nsw i64 %3793, 2		; visa id: 4832
  %3795 = add i64 %133, %3794		; visa id: 4833
  %3796 = inttoptr i64 %3795 to i32*		; visa id: 4834
  %3797 = load i32, i32* %3796, align 4, !noalias !625		; visa id: 4834
  %3798 = add i64 %128, %3794		; visa id: 4835
  %3799 = inttoptr i64 %3798 to i32*		; visa id: 4836
  store i32 %3797, i32* %3799, align 4, !alias.scope !625		; visa id: 4836
  %3800 = icmp eq i32 %3792, 0		; visa id: 4837
  br i1 %3800, label %._crit_edge287.._crit_edge287_crit_edge, label %3802, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4838

._crit_edge287.._crit_edge287_crit_edge:          ; preds = %._crit_edge287
; BB370 :
  %3801 = add nuw nsw i32 %3792, 1, !spirv.Decorations !631		; visa id: 4840
  br label %._crit_edge287, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4841

3802:                                             ; preds = %._crit_edge287
; BB371 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4843
  %3803 = load i64, i64* %129, align 8		; visa id: 4843
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4844
  %3804 = bitcast i64 %3803 to <2 x i32>		; visa id: 4844
  %3805 = extractelement <2 x i32> %3804, i32 0		; visa id: 4846
  %3806 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3805, i32 1
  %3807 = bitcast <2 x i32> %3806 to i64		; visa id: 4846
  %3808 = ashr exact i64 %3807, 32		; visa id: 4847
  %3809 = bitcast i64 %3808 to <2 x i32>		; visa id: 4848
  %3810 = extractelement <2 x i32> %3809, i32 0		; visa id: 4852
  %3811 = extractelement <2 x i32> %3809, i32 1		; visa id: 4852
  %3812 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3810, i32 %3811, i32 %41, i32 %42)
  %3813 = extractvalue { i32, i32 } %3812, 0		; visa id: 4852
  %3814 = extractvalue { i32, i32 } %3812, 1		; visa id: 4852
  %3815 = insertelement <2 x i32> undef, i32 %3813, i32 0		; visa id: 4859
  %3816 = insertelement <2 x i32> %3815, i32 %3814, i32 1		; visa id: 4860
  %3817 = bitcast <2 x i32> %3816 to i64		; visa id: 4861
  %3818 = shl i64 %3817, 1		; visa id: 4865
  %3819 = add i64 %.in401, %3818		; visa id: 4866
  %3820 = ashr i64 %3803, 31		; visa id: 4867
  %3821 = bitcast i64 %3820 to <2 x i32>		; visa id: 4868
  %3822 = extractelement <2 x i32> %3821, i32 0		; visa id: 4872
  %3823 = extractelement <2 x i32> %3821, i32 1		; visa id: 4872
  %3824 = and i32 %3822, -2		; visa id: 4872
  %3825 = insertelement <2 x i32> undef, i32 %3824, i32 0		; visa id: 4873
  %3826 = insertelement <2 x i32> %3825, i32 %3823, i32 1		; visa id: 4874
  %3827 = bitcast <2 x i32> %3826 to i64		; visa id: 4875
  %3828 = add i64 %3819, %3827		; visa id: 4879
  %3829 = inttoptr i64 %3828 to i16 addrspace(4)*		; visa id: 4880
  %3830 = addrspacecast i16 addrspace(4)* %3829 to i16 addrspace(1)*		; visa id: 4880
  %3831 = load i16, i16 addrspace(1)* %3830, align 2		; visa id: 4881
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4883
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4883
  %3832 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4883
  %3833 = insertelement <2 x i32> %3832, i32 %3784, i64 1		; visa id: 4884
  %3834 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4885
  store <2 x i32> %3833, <2 x i32>* %3834, align 4, !noalias !635		; visa id: 4885
  br label %._crit_edge288, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4887

._crit_edge288:                                   ; preds = %._crit_edge288.._crit_edge288_crit_edge, %3802
; BB372 :
  %3835 = phi i32 [ 0, %3802 ], [ %3844, %._crit_edge288.._crit_edge288_crit_edge ]
  %3836 = zext i32 %3835 to i64		; visa id: 4888
  %3837 = shl nuw nsw i64 %3836, 2		; visa id: 4889
  %3838 = add i64 %124, %3837		; visa id: 4890
  %3839 = inttoptr i64 %3838 to i32*		; visa id: 4891
  %3840 = load i32, i32* %3839, align 4, !noalias !635		; visa id: 4891
  %3841 = add i64 %119, %3837		; visa id: 4892
  %3842 = inttoptr i64 %3841 to i32*		; visa id: 4893
  store i32 %3840, i32* %3842, align 4, !alias.scope !635		; visa id: 4893
  %3843 = icmp eq i32 %3835, 0		; visa id: 4894
  br i1 %3843, label %._crit_edge288.._crit_edge288_crit_edge, label %3845, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4895

._crit_edge288.._crit_edge288_crit_edge:          ; preds = %._crit_edge288
; BB373 :
  %3844 = add nuw nsw i32 %3835, 1, !spirv.Decorations !631		; visa id: 4897
  br label %._crit_edge288, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4898

3845:                                             ; preds = %._crit_edge288
; BB374 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4900
  %3846 = load i64, i64* %120, align 8		; visa id: 4900
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4901
  %3847 = ashr i64 %3846, 32		; visa id: 4901
  %3848 = bitcast i64 %3847 to <2 x i32>		; visa id: 4902
  %3849 = extractelement <2 x i32> %3848, i32 0		; visa id: 4906
  %3850 = extractelement <2 x i32> %3848, i32 1		; visa id: 4906
  %3851 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3849, i32 %3850, i32 %44, i32 %45)
  %3852 = extractvalue { i32, i32 } %3851, 0		; visa id: 4906
  %3853 = extractvalue { i32, i32 } %3851, 1		; visa id: 4906
  %3854 = insertelement <2 x i32> undef, i32 %3852, i32 0		; visa id: 4913
  %3855 = insertelement <2 x i32> %3854, i32 %3853, i32 1		; visa id: 4914
  %3856 = bitcast <2 x i32> %3855 to i64		; visa id: 4915
  %3857 = bitcast i64 %3846 to <2 x i32>		; visa id: 4919
  %3858 = extractelement <2 x i32> %3857, i32 0		; visa id: 4921
  %3859 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3858, i32 1
  %3860 = bitcast <2 x i32> %3859 to i64		; visa id: 4921
  %3861 = shl i64 %3856, 1		; visa id: 4922
  %3862 = add i64 %.in400, %3861		; visa id: 4923
  %3863 = ashr exact i64 %3860, 31		; visa id: 4924
  %3864 = add i64 %3862, %3863		; visa id: 4925
  %3865 = inttoptr i64 %3864 to i16 addrspace(4)*		; visa id: 4926
  %3866 = addrspacecast i16 addrspace(4)* %3865 to i16 addrspace(1)*		; visa id: 4926
  %3867 = load i16, i16 addrspace(1)* %3866, align 2		; visa id: 4927
  %3868 = zext i16 %3831 to i32		; visa id: 4929
  %3869 = shl nuw i32 %3868, 16, !spirv.Decorations !639		; visa id: 4930
  %3870 = bitcast i32 %3869 to float
  %3871 = zext i16 %3867 to i32		; visa id: 4931
  %3872 = shl nuw i32 %3871, 16, !spirv.Decorations !639		; visa id: 4932
  %3873 = bitcast i32 %3872 to float
  %3874 = fmul reassoc nsz arcp contract float %3870, %3873, !spirv.Decorations !618
  %3875 = fadd reassoc nsz arcp contract float %3874, %.sroa.42.1, !spirv.Decorations !618		; visa id: 4933
  br label %._crit_edge.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4934

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %3845
; BB375 :
  %.sroa.42.2 = phi float [ %3875, %3845 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  %3876 = icmp slt i32 %223, %const_reg_dword
  %3877 = icmp slt i32 %3784, %const_reg_dword1		; visa id: 4935
  %3878 = and i1 %3876, %3877		; visa id: 4936
  br i1 %3878, label %3879, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4938

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
; BB:
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3879:                                             ; preds = %._crit_edge.10
; BB377 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4940
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4940
  %3880 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 4940
  %3881 = insertelement <2 x i32> %3880, i32 %113, i64 1		; visa id: 4941
  %3882 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4942
  store <2 x i32> %3881, <2 x i32>* %3882, align 4, !noalias !625		; visa id: 4942
  br label %._crit_edge289, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4944

._crit_edge289:                                   ; preds = %._crit_edge289.._crit_edge289_crit_edge, %3879
; BB378 :
  %3883 = phi i32 [ 0, %3879 ], [ %3892, %._crit_edge289.._crit_edge289_crit_edge ]
  %3884 = zext i32 %3883 to i64		; visa id: 4945
  %3885 = shl nuw nsw i64 %3884, 2		; visa id: 4946
  %3886 = add i64 %133, %3885		; visa id: 4947
  %3887 = inttoptr i64 %3886 to i32*		; visa id: 4948
  %3888 = load i32, i32* %3887, align 4, !noalias !625		; visa id: 4948
  %3889 = add i64 %128, %3885		; visa id: 4949
  %3890 = inttoptr i64 %3889 to i32*		; visa id: 4950
  store i32 %3888, i32* %3890, align 4, !alias.scope !625		; visa id: 4950
  %3891 = icmp eq i32 %3883, 0		; visa id: 4951
  br i1 %3891, label %._crit_edge289.._crit_edge289_crit_edge, label %3893, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4952

._crit_edge289.._crit_edge289_crit_edge:          ; preds = %._crit_edge289
; BB379 :
  %3892 = add nuw nsw i32 %3883, 1, !spirv.Decorations !631		; visa id: 4954
  br label %._crit_edge289, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4955

3893:                                             ; preds = %._crit_edge289
; BB380 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4957
  %3894 = load i64, i64* %129, align 8		; visa id: 4957
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4958
  %3895 = bitcast i64 %3894 to <2 x i32>		; visa id: 4958
  %3896 = extractelement <2 x i32> %3895, i32 0		; visa id: 4960
  %3897 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3896, i32 1
  %3898 = bitcast <2 x i32> %3897 to i64		; visa id: 4960
  %3899 = ashr exact i64 %3898, 32		; visa id: 4961
  %3900 = bitcast i64 %3899 to <2 x i32>		; visa id: 4962
  %3901 = extractelement <2 x i32> %3900, i32 0		; visa id: 4966
  %3902 = extractelement <2 x i32> %3900, i32 1		; visa id: 4966
  %3903 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3901, i32 %3902, i32 %41, i32 %42)
  %3904 = extractvalue { i32, i32 } %3903, 0		; visa id: 4966
  %3905 = extractvalue { i32, i32 } %3903, 1		; visa id: 4966
  %3906 = insertelement <2 x i32> undef, i32 %3904, i32 0		; visa id: 4973
  %3907 = insertelement <2 x i32> %3906, i32 %3905, i32 1		; visa id: 4974
  %3908 = bitcast <2 x i32> %3907 to i64		; visa id: 4975
  %3909 = shl i64 %3908, 1		; visa id: 4979
  %3910 = add i64 %.in401, %3909		; visa id: 4980
  %3911 = ashr i64 %3894, 31		; visa id: 4981
  %3912 = bitcast i64 %3911 to <2 x i32>		; visa id: 4982
  %3913 = extractelement <2 x i32> %3912, i32 0		; visa id: 4986
  %3914 = extractelement <2 x i32> %3912, i32 1		; visa id: 4986
  %3915 = and i32 %3913, -2		; visa id: 4986
  %3916 = insertelement <2 x i32> undef, i32 %3915, i32 0		; visa id: 4987
  %3917 = insertelement <2 x i32> %3916, i32 %3914, i32 1		; visa id: 4988
  %3918 = bitcast <2 x i32> %3917 to i64		; visa id: 4989
  %3919 = add i64 %3910, %3918		; visa id: 4993
  %3920 = inttoptr i64 %3919 to i16 addrspace(4)*		; visa id: 4994
  %3921 = addrspacecast i16 addrspace(4)* %3920 to i16 addrspace(1)*		; visa id: 4994
  %3922 = load i16, i16 addrspace(1)* %3921, align 2		; visa id: 4995
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4997
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4997
  %3923 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4997
  %3924 = insertelement <2 x i32> %3923, i32 %3784, i64 1		; visa id: 4998
  %3925 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4999
  store <2 x i32> %3924, <2 x i32>* %3925, align 4, !noalias !635		; visa id: 4999
  br label %._crit_edge290, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5001

._crit_edge290:                                   ; preds = %._crit_edge290.._crit_edge290_crit_edge, %3893
; BB381 :
  %3926 = phi i32 [ 0, %3893 ], [ %3935, %._crit_edge290.._crit_edge290_crit_edge ]
  %3927 = zext i32 %3926 to i64		; visa id: 5002
  %3928 = shl nuw nsw i64 %3927, 2		; visa id: 5003
  %3929 = add i64 %124, %3928		; visa id: 5004
  %3930 = inttoptr i64 %3929 to i32*		; visa id: 5005
  %3931 = load i32, i32* %3930, align 4, !noalias !635		; visa id: 5005
  %3932 = add i64 %119, %3928		; visa id: 5006
  %3933 = inttoptr i64 %3932 to i32*		; visa id: 5007
  store i32 %3931, i32* %3933, align 4, !alias.scope !635		; visa id: 5007
  %3934 = icmp eq i32 %3926, 0		; visa id: 5008
  br i1 %3934, label %._crit_edge290.._crit_edge290_crit_edge, label %3936, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5009

._crit_edge290.._crit_edge290_crit_edge:          ; preds = %._crit_edge290
; BB382 :
  %3935 = add nuw nsw i32 %3926, 1, !spirv.Decorations !631		; visa id: 5011
  br label %._crit_edge290, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5012

3936:                                             ; preds = %._crit_edge290
; BB383 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5014
  %3937 = load i64, i64* %120, align 8		; visa id: 5014
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5015
  %3938 = ashr i64 %3937, 32		; visa id: 5015
  %3939 = bitcast i64 %3938 to <2 x i32>		; visa id: 5016
  %3940 = extractelement <2 x i32> %3939, i32 0		; visa id: 5020
  %3941 = extractelement <2 x i32> %3939, i32 1		; visa id: 5020
  %3942 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3940, i32 %3941, i32 %44, i32 %45)
  %3943 = extractvalue { i32, i32 } %3942, 0		; visa id: 5020
  %3944 = extractvalue { i32, i32 } %3942, 1		; visa id: 5020
  %3945 = insertelement <2 x i32> undef, i32 %3943, i32 0		; visa id: 5027
  %3946 = insertelement <2 x i32> %3945, i32 %3944, i32 1		; visa id: 5028
  %3947 = bitcast <2 x i32> %3946 to i64		; visa id: 5029
  %3948 = bitcast i64 %3937 to <2 x i32>		; visa id: 5033
  %3949 = extractelement <2 x i32> %3948, i32 0		; visa id: 5035
  %3950 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3949, i32 1
  %3951 = bitcast <2 x i32> %3950 to i64		; visa id: 5035
  %3952 = shl i64 %3947, 1		; visa id: 5036
  %3953 = add i64 %.in400, %3952		; visa id: 5037
  %3954 = ashr exact i64 %3951, 31		; visa id: 5038
  %3955 = add i64 %3953, %3954		; visa id: 5039
  %3956 = inttoptr i64 %3955 to i16 addrspace(4)*		; visa id: 5040
  %3957 = addrspacecast i16 addrspace(4)* %3956 to i16 addrspace(1)*		; visa id: 5040
  %3958 = load i16, i16 addrspace(1)* %3957, align 2		; visa id: 5041
  %3959 = zext i16 %3922 to i32		; visa id: 5043
  %3960 = shl nuw i32 %3959, 16, !spirv.Decorations !639		; visa id: 5044
  %3961 = bitcast i32 %3960 to float
  %3962 = zext i16 %3958 to i32		; visa id: 5045
  %3963 = shl nuw i32 %3962, 16, !spirv.Decorations !639		; visa id: 5046
  %3964 = bitcast i32 %3963 to float
  %3965 = fmul reassoc nsz arcp contract float %3961, %3964, !spirv.Decorations !618
  %3966 = fadd reassoc nsz arcp contract float %3965, %.sroa.106.1, !spirv.Decorations !618		; visa id: 5047
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5048

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %3936
; BB384 :
  %.sroa.106.2 = phi float [ %3966, %3936 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  %3967 = icmp slt i32 %315, %const_reg_dword
  %3968 = icmp slt i32 %3784, %const_reg_dword1		; visa id: 5049
  %3969 = and i1 %3967, %3968		; visa id: 5050
  br i1 %3969, label %3970, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5052

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
; BB:
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3970:                                             ; preds = %._crit_edge.1.10
; BB386 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5054
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5054
  %3971 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 5054
  %3972 = insertelement <2 x i32> %3971, i32 %113, i64 1		; visa id: 5055
  %3973 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5056
  store <2 x i32> %3972, <2 x i32>* %3973, align 4, !noalias !625		; visa id: 5056
  br label %._crit_edge291, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5058

._crit_edge291:                                   ; preds = %._crit_edge291.._crit_edge291_crit_edge, %3970
; BB387 :
  %3974 = phi i32 [ 0, %3970 ], [ %3983, %._crit_edge291.._crit_edge291_crit_edge ]
  %3975 = zext i32 %3974 to i64		; visa id: 5059
  %3976 = shl nuw nsw i64 %3975, 2		; visa id: 5060
  %3977 = add i64 %133, %3976		; visa id: 5061
  %3978 = inttoptr i64 %3977 to i32*		; visa id: 5062
  %3979 = load i32, i32* %3978, align 4, !noalias !625		; visa id: 5062
  %3980 = add i64 %128, %3976		; visa id: 5063
  %3981 = inttoptr i64 %3980 to i32*		; visa id: 5064
  store i32 %3979, i32* %3981, align 4, !alias.scope !625		; visa id: 5064
  %3982 = icmp eq i32 %3974, 0		; visa id: 5065
  br i1 %3982, label %._crit_edge291.._crit_edge291_crit_edge, label %3984, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5066

._crit_edge291.._crit_edge291_crit_edge:          ; preds = %._crit_edge291
; BB388 :
  %3983 = add nuw nsw i32 %3974, 1, !spirv.Decorations !631		; visa id: 5068
  br label %._crit_edge291, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5069

3984:                                             ; preds = %._crit_edge291
; BB389 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5071
  %3985 = load i64, i64* %129, align 8		; visa id: 5071
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5072
  %3986 = bitcast i64 %3985 to <2 x i32>		; visa id: 5072
  %3987 = extractelement <2 x i32> %3986, i32 0		; visa id: 5074
  %3988 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3987, i32 1
  %3989 = bitcast <2 x i32> %3988 to i64		; visa id: 5074
  %3990 = ashr exact i64 %3989, 32		; visa id: 5075
  %3991 = bitcast i64 %3990 to <2 x i32>		; visa id: 5076
  %3992 = extractelement <2 x i32> %3991, i32 0		; visa id: 5080
  %3993 = extractelement <2 x i32> %3991, i32 1		; visa id: 5080
  %3994 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3992, i32 %3993, i32 %41, i32 %42)
  %3995 = extractvalue { i32, i32 } %3994, 0		; visa id: 5080
  %3996 = extractvalue { i32, i32 } %3994, 1		; visa id: 5080
  %3997 = insertelement <2 x i32> undef, i32 %3995, i32 0		; visa id: 5087
  %3998 = insertelement <2 x i32> %3997, i32 %3996, i32 1		; visa id: 5088
  %3999 = bitcast <2 x i32> %3998 to i64		; visa id: 5089
  %4000 = shl i64 %3999, 1		; visa id: 5093
  %4001 = add i64 %.in401, %4000		; visa id: 5094
  %4002 = ashr i64 %3985, 31		; visa id: 5095
  %4003 = bitcast i64 %4002 to <2 x i32>		; visa id: 5096
  %4004 = extractelement <2 x i32> %4003, i32 0		; visa id: 5100
  %4005 = extractelement <2 x i32> %4003, i32 1		; visa id: 5100
  %4006 = and i32 %4004, -2		; visa id: 5100
  %4007 = insertelement <2 x i32> undef, i32 %4006, i32 0		; visa id: 5101
  %4008 = insertelement <2 x i32> %4007, i32 %4005, i32 1		; visa id: 5102
  %4009 = bitcast <2 x i32> %4008 to i64		; visa id: 5103
  %4010 = add i64 %4001, %4009		; visa id: 5107
  %4011 = inttoptr i64 %4010 to i16 addrspace(4)*		; visa id: 5108
  %4012 = addrspacecast i16 addrspace(4)* %4011 to i16 addrspace(1)*		; visa id: 5108
  %4013 = load i16, i16 addrspace(1)* %4012, align 2		; visa id: 5109
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5111
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5111
  %4014 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5111
  %4015 = insertelement <2 x i32> %4014, i32 %3784, i64 1		; visa id: 5112
  %4016 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5113
  store <2 x i32> %4015, <2 x i32>* %4016, align 4, !noalias !635		; visa id: 5113
  br label %._crit_edge292, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5115

._crit_edge292:                                   ; preds = %._crit_edge292.._crit_edge292_crit_edge, %3984
; BB390 :
  %4017 = phi i32 [ 0, %3984 ], [ %4026, %._crit_edge292.._crit_edge292_crit_edge ]
  %4018 = zext i32 %4017 to i64		; visa id: 5116
  %4019 = shl nuw nsw i64 %4018, 2		; visa id: 5117
  %4020 = add i64 %124, %4019		; visa id: 5118
  %4021 = inttoptr i64 %4020 to i32*		; visa id: 5119
  %4022 = load i32, i32* %4021, align 4, !noalias !635		; visa id: 5119
  %4023 = add i64 %119, %4019		; visa id: 5120
  %4024 = inttoptr i64 %4023 to i32*		; visa id: 5121
  store i32 %4022, i32* %4024, align 4, !alias.scope !635		; visa id: 5121
  %4025 = icmp eq i32 %4017, 0		; visa id: 5122
  br i1 %4025, label %._crit_edge292.._crit_edge292_crit_edge, label %4027, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5123

._crit_edge292.._crit_edge292_crit_edge:          ; preds = %._crit_edge292
; BB391 :
  %4026 = add nuw nsw i32 %4017, 1, !spirv.Decorations !631		; visa id: 5125
  br label %._crit_edge292, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5126

4027:                                             ; preds = %._crit_edge292
; BB392 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5128
  %4028 = load i64, i64* %120, align 8		; visa id: 5128
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5129
  %4029 = ashr i64 %4028, 32		; visa id: 5129
  %4030 = bitcast i64 %4029 to <2 x i32>		; visa id: 5130
  %4031 = extractelement <2 x i32> %4030, i32 0		; visa id: 5134
  %4032 = extractelement <2 x i32> %4030, i32 1		; visa id: 5134
  %4033 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4031, i32 %4032, i32 %44, i32 %45)
  %4034 = extractvalue { i32, i32 } %4033, 0		; visa id: 5134
  %4035 = extractvalue { i32, i32 } %4033, 1		; visa id: 5134
  %4036 = insertelement <2 x i32> undef, i32 %4034, i32 0		; visa id: 5141
  %4037 = insertelement <2 x i32> %4036, i32 %4035, i32 1		; visa id: 5142
  %4038 = bitcast <2 x i32> %4037 to i64		; visa id: 5143
  %4039 = bitcast i64 %4028 to <2 x i32>		; visa id: 5147
  %4040 = extractelement <2 x i32> %4039, i32 0		; visa id: 5149
  %4041 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4040, i32 1
  %4042 = bitcast <2 x i32> %4041 to i64		; visa id: 5149
  %4043 = shl i64 %4038, 1		; visa id: 5150
  %4044 = add i64 %.in400, %4043		; visa id: 5151
  %4045 = ashr exact i64 %4042, 31		; visa id: 5152
  %4046 = add i64 %4044, %4045		; visa id: 5153
  %4047 = inttoptr i64 %4046 to i16 addrspace(4)*		; visa id: 5154
  %4048 = addrspacecast i16 addrspace(4)* %4047 to i16 addrspace(1)*		; visa id: 5154
  %4049 = load i16, i16 addrspace(1)* %4048, align 2		; visa id: 5155
  %4050 = zext i16 %4013 to i32		; visa id: 5157
  %4051 = shl nuw i32 %4050, 16, !spirv.Decorations !639		; visa id: 5158
  %4052 = bitcast i32 %4051 to float
  %4053 = zext i16 %4049 to i32		; visa id: 5159
  %4054 = shl nuw i32 %4053, 16, !spirv.Decorations !639		; visa id: 5160
  %4055 = bitcast i32 %4054 to float
  %4056 = fmul reassoc nsz arcp contract float %4052, %4055, !spirv.Decorations !618
  %4057 = fadd reassoc nsz arcp contract float %4056, %.sroa.170.1, !spirv.Decorations !618		; visa id: 5161
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5162

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %4027
; BB393 :
  %.sroa.170.2 = phi float [ %4057, %4027 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  %4058 = icmp slt i32 %407, %const_reg_dword
  %4059 = icmp slt i32 %3784, %const_reg_dword1		; visa id: 5163
  %4060 = and i1 %4058, %4059		; visa id: 5164
  br i1 %4060, label %4061, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5166

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
; BB:
  br label %.preheader.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4061:                                             ; preds = %._crit_edge.2.10
; BB395 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5168
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5168
  %4062 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 5168
  %4063 = insertelement <2 x i32> %4062, i32 %113, i64 1		; visa id: 5169
  %4064 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5170
  store <2 x i32> %4063, <2 x i32>* %4064, align 4, !noalias !625		; visa id: 5170
  br label %._crit_edge293, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5172

._crit_edge293:                                   ; preds = %._crit_edge293.._crit_edge293_crit_edge, %4061
; BB396 :
  %4065 = phi i32 [ 0, %4061 ], [ %4074, %._crit_edge293.._crit_edge293_crit_edge ]
  %4066 = zext i32 %4065 to i64		; visa id: 5173
  %4067 = shl nuw nsw i64 %4066, 2		; visa id: 5174
  %4068 = add i64 %133, %4067		; visa id: 5175
  %4069 = inttoptr i64 %4068 to i32*		; visa id: 5176
  %4070 = load i32, i32* %4069, align 4, !noalias !625		; visa id: 5176
  %4071 = add i64 %128, %4067		; visa id: 5177
  %4072 = inttoptr i64 %4071 to i32*		; visa id: 5178
  store i32 %4070, i32* %4072, align 4, !alias.scope !625		; visa id: 5178
  %4073 = icmp eq i32 %4065, 0		; visa id: 5179
  br i1 %4073, label %._crit_edge293.._crit_edge293_crit_edge, label %4075, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5180

._crit_edge293.._crit_edge293_crit_edge:          ; preds = %._crit_edge293
; BB397 :
  %4074 = add nuw nsw i32 %4065, 1, !spirv.Decorations !631		; visa id: 5182
  br label %._crit_edge293, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5183

4075:                                             ; preds = %._crit_edge293
; BB398 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5185
  %4076 = load i64, i64* %129, align 8		; visa id: 5185
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5186
  %4077 = bitcast i64 %4076 to <2 x i32>		; visa id: 5186
  %4078 = extractelement <2 x i32> %4077, i32 0		; visa id: 5188
  %4079 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4078, i32 1
  %4080 = bitcast <2 x i32> %4079 to i64		; visa id: 5188
  %4081 = ashr exact i64 %4080, 32		; visa id: 5189
  %4082 = bitcast i64 %4081 to <2 x i32>		; visa id: 5190
  %4083 = extractelement <2 x i32> %4082, i32 0		; visa id: 5194
  %4084 = extractelement <2 x i32> %4082, i32 1		; visa id: 5194
  %4085 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4083, i32 %4084, i32 %41, i32 %42)
  %4086 = extractvalue { i32, i32 } %4085, 0		; visa id: 5194
  %4087 = extractvalue { i32, i32 } %4085, 1		; visa id: 5194
  %4088 = insertelement <2 x i32> undef, i32 %4086, i32 0		; visa id: 5201
  %4089 = insertelement <2 x i32> %4088, i32 %4087, i32 1		; visa id: 5202
  %4090 = bitcast <2 x i32> %4089 to i64		; visa id: 5203
  %4091 = shl i64 %4090, 1		; visa id: 5207
  %4092 = add i64 %.in401, %4091		; visa id: 5208
  %4093 = ashr i64 %4076, 31		; visa id: 5209
  %4094 = bitcast i64 %4093 to <2 x i32>		; visa id: 5210
  %4095 = extractelement <2 x i32> %4094, i32 0		; visa id: 5214
  %4096 = extractelement <2 x i32> %4094, i32 1		; visa id: 5214
  %4097 = and i32 %4095, -2		; visa id: 5214
  %4098 = insertelement <2 x i32> undef, i32 %4097, i32 0		; visa id: 5215
  %4099 = insertelement <2 x i32> %4098, i32 %4096, i32 1		; visa id: 5216
  %4100 = bitcast <2 x i32> %4099 to i64		; visa id: 5217
  %4101 = add i64 %4092, %4100		; visa id: 5221
  %4102 = inttoptr i64 %4101 to i16 addrspace(4)*		; visa id: 5222
  %4103 = addrspacecast i16 addrspace(4)* %4102 to i16 addrspace(1)*		; visa id: 5222
  %4104 = load i16, i16 addrspace(1)* %4103, align 2		; visa id: 5223
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5225
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5225
  %4105 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5225
  %4106 = insertelement <2 x i32> %4105, i32 %3784, i64 1		; visa id: 5226
  %4107 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5227
  store <2 x i32> %4106, <2 x i32>* %4107, align 4, !noalias !635		; visa id: 5227
  br label %._crit_edge294, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5229

._crit_edge294:                                   ; preds = %._crit_edge294.._crit_edge294_crit_edge, %4075
; BB399 :
  %4108 = phi i32 [ 0, %4075 ], [ %4117, %._crit_edge294.._crit_edge294_crit_edge ]
  %4109 = zext i32 %4108 to i64		; visa id: 5230
  %4110 = shl nuw nsw i64 %4109, 2		; visa id: 5231
  %4111 = add i64 %124, %4110		; visa id: 5232
  %4112 = inttoptr i64 %4111 to i32*		; visa id: 5233
  %4113 = load i32, i32* %4112, align 4, !noalias !635		; visa id: 5233
  %4114 = add i64 %119, %4110		; visa id: 5234
  %4115 = inttoptr i64 %4114 to i32*		; visa id: 5235
  store i32 %4113, i32* %4115, align 4, !alias.scope !635		; visa id: 5235
  %4116 = icmp eq i32 %4108, 0		; visa id: 5236
  br i1 %4116, label %._crit_edge294.._crit_edge294_crit_edge, label %4118, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5237

._crit_edge294.._crit_edge294_crit_edge:          ; preds = %._crit_edge294
; BB400 :
  %4117 = add nuw nsw i32 %4108, 1, !spirv.Decorations !631		; visa id: 5239
  br label %._crit_edge294, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5240

4118:                                             ; preds = %._crit_edge294
; BB401 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5242
  %4119 = load i64, i64* %120, align 8		; visa id: 5242
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5243
  %4120 = ashr i64 %4119, 32		; visa id: 5243
  %4121 = bitcast i64 %4120 to <2 x i32>		; visa id: 5244
  %4122 = extractelement <2 x i32> %4121, i32 0		; visa id: 5248
  %4123 = extractelement <2 x i32> %4121, i32 1		; visa id: 5248
  %4124 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4122, i32 %4123, i32 %44, i32 %45)
  %4125 = extractvalue { i32, i32 } %4124, 0		; visa id: 5248
  %4126 = extractvalue { i32, i32 } %4124, 1		; visa id: 5248
  %4127 = insertelement <2 x i32> undef, i32 %4125, i32 0		; visa id: 5255
  %4128 = insertelement <2 x i32> %4127, i32 %4126, i32 1		; visa id: 5256
  %4129 = bitcast <2 x i32> %4128 to i64		; visa id: 5257
  %4130 = bitcast i64 %4119 to <2 x i32>		; visa id: 5261
  %4131 = extractelement <2 x i32> %4130, i32 0		; visa id: 5263
  %4132 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4131, i32 1
  %4133 = bitcast <2 x i32> %4132 to i64		; visa id: 5263
  %4134 = shl i64 %4129, 1		; visa id: 5264
  %4135 = add i64 %.in400, %4134		; visa id: 5265
  %4136 = ashr exact i64 %4133, 31		; visa id: 5266
  %4137 = add i64 %4135, %4136		; visa id: 5267
  %4138 = inttoptr i64 %4137 to i16 addrspace(4)*		; visa id: 5268
  %4139 = addrspacecast i16 addrspace(4)* %4138 to i16 addrspace(1)*		; visa id: 5268
  %4140 = load i16, i16 addrspace(1)* %4139, align 2		; visa id: 5269
  %4141 = zext i16 %4104 to i32		; visa id: 5271
  %4142 = shl nuw i32 %4141, 16, !spirv.Decorations !639		; visa id: 5272
  %4143 = bitcast i32 %4142 to float
  %4144 = zext i16 %4140 to i32		; visa id: 5273
  %4145 = shl nuw i32 %4144, 16, !spirv.Decorations !639		; visa id: 5274
  %4146 = bitcast i32 %4145 to float
  %4147 = fmul reassoc nsz arcp contract float %4143, %4146, !spirv.Decorations !618
  %4148 = fadd reassoc nsz arcp contract float %4147, %.sroa.234.1, !spirv.Decorations !618		; visa id: 5275
  br label %.preheader.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5276

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %4118
; BB402 :
  %.sroa.234.2 = phi float [ %4148, %4118 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %4149 = add i32 %69, 11		; visa id: 5277
  %4150 = icmp slt i32 %4149, %const_reg_dword1		; visa id: 5278
  %4151 = icmp slt i32 %65, %const_reg_dword
  %4152 = and i1 %4151, %4150		; visa id: 5279
  br i1 %4152, label %4153, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5281

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
; BB:
  br label %._crit_edge.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4153:                                             ; preds = %.preheader.10
; BB404 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5283
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5283
  %4154 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 5283
  %4155 = insertelement <2 x i32> %4154, i32 %113, i64 1		; visa id: 5284
  %4156 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5285
  store <2 x i32> %4155, <2 x i32>* %4156, align 4, !noalias !625		; visa id: 5285
  br label %._crit_edge295, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5287

._crit_edge295:                                   ; preds = %._crit_edge295.._crit_edge295_crit_edge, %4153
; BB405 :
  %4157 = phi i32 [ 0, %4153 ], [ %4166, %._crit_edge295.._crit_edge295_crit_edge ]
  %4158 = zext i32 %4157 to i64		; visa id: 5288
  %4159 = shl nuw nsw i64 %4158, 2		; visa id: 5289
  %4160 = add i64 %133, %4159		; visa id: 5290
  %4161 = inttoptr i64 %4160 to i32*		; visa id: 5291
  %4162 = load i32, i32* %4161, align 4, !noalias !625		; visa id: 5291
  %4163 = add i64 %128, %4159		; visa id: 5292
  %4164 = inttoptr i64 %4163 to i32*		; visa id: 5293
  store i32 %4162, i32* %4164, align 4, !alias.scope !625		; visa id: 5293
  %4165 = icmp eq i32 %4157, 0		; visa id: 5294
  br i1 %4165, label %._crit_edge295.._crit_edge295_crit_edge, label %4167, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5295

._crit_edge295.._crit_edge295_crit_edge:          ; preds = %._crit_edge295
; BB406 :
  %4166 = add nuw nsw i32 %4157, 1, !spirv.Decorations !631		; visa id: 5297
  br label %._crit_edge295, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5298

4167:                                             ; preds = %._crit_edge295
; BB407 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5300
  %4168 = load i64, i64* %129, align 8		; visa id: 5300
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5301
  %4169 = bitcast i64 %4168 to <2 x i32>		; visa id: 5301
  %4170 = extractelement <2 x i32> %4169, i32 0		; visa id: 5303
  %4171 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4170, i32 1
  %4172 = bitcast <2 x i32> %4171 to i64		; visa id: 5303
  %4173 = ashr exact i64 %4172, 32		; visa id: 5304
  %4174 = bitcast i64 %4173 to <2 x i32>		; visa id: 5305
  %4175 = extractelement <2 x i32> %4174, i32 0		; visa id: 5309
  %4176 = extractelement <2 x i32> %4174, i32 1		; visa id: 5309
  %4177 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4175, i32 %4176, i32 %41, i32 %42)
  %4178 = extractvalue { i32, i32 } %4177, 0		; visa id: 5309
  %4179 = extractvalue { i32, i32 } %4177, 1		; visa id: 5309
  %4180 = insertelement <2 x i32> undef, i32 %4178, i32 0		; visa id: 5316
  %4181 = insertelement <2 x i32> %4180, i32 %4179, i32 1		; visa id: 5317
  %4182 = bitcast <2 x i32> %4181 to i64		; visa id: 5318
  %4183 = shl i64 %4182, 1		; visa id: 5322
  %4184 = add i64 %.in401, %4183		; visa id: 5323
  %4185 = ashr i64 %4168, 31		; visa id: 5324
  %4186 = bitcast i64 %4185 to <2 x i32>		; visa id: 5325
  %4187 = extractelement <2 x i32> %4186, i32 0		; visa id: 5329
  %4188 = extractelement <2 x i32> %4186, i32 1		; visa id: 5329
  %4189 = and i32 %4187, -2		; visa id: 5329
  %4190 = insertelement <2 x i32> undef, i32 %4189, i32 0		; visa id: 5330
  %4191 = insertelement <2 x i32> %4190, i32 %4188, i32 1		; visa id: 5331
  %4192 = bitcast <2 x i32> %4191 to i64		; visa id: 5332
  %4193 = add i64 %4184, %4192		; visa id: 5336
  %4194 = inttoptr i64 %4193 to i16 addrspace(4)*		; visa id: 5337
  %4195 = addrspacecast i16 addrspace(4)* %4194 to i16 addrspace(1)*		; visa id: 5337
  %4196 = load i16, i16 addrspace(1)* %4195, align 2		; visa id: 5338
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5340
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5340
  %4197 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5340
  %4198 = insertelement <2 x i32> %4197, i32 %4149, i64 1		; visa id: 5341
  %4199 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5342
  store <2 x i32> %4198, <2 x i32>* %4199, align 4, !noalias !635		; visa id: 5342
  br label %._crit_edge296, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5344

._crit_edge296:                                   ; preds = %._crit_edge296.._crit_edge296_crit_edge, %4167
; BB408 :
  %4200 = phi i32 [ 0, %4167 ], [ %4209, %._crit_edge296.._crit_edge296_crit_edge ]
  %4201 = zext i32 %4200 to i64		; visa id: 5345
  %4202 = shl nuw nsw i64 %4201, 2		; visa id: 5346
  %4203 = add i64 %124, %4202		; visa id: 5347
  %4204 = inttoptr i64 %4203 to i32*		; visa id: 5348
  %4205 = load i32, i32* %4204, align 4, !noalias !635		; visa id: 5348
  %4206 = add i64 %119, %4202		; visa id: 5349
  %4207 = inttoptr i64 %4206 to i32*		; visa id: 5350
  store i32 %4205, i32* %4207, align 4, !alias.scope !635		; visa id: 5350
  %4208 = icmp eq i32 %4200, 0		; visa id: 5351
  br i1 %4208, label %._crit_edge296.._crit_edge296_crit_edge, label %4210, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5352

._crit_edge296.._crit_edge296_crit_edge:          ; preds = %._crit_edge296
; BB409 :
  %4209 = add nuw nsw i32 %4200, 1, !spirv.Decorations !631		; visa id: 5354
  br label %._crit_edge296, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5355

4210:                                             ; preds = %._crit_edge296
; BB410 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5357
  %4211 = load i64, i64* %120, align 8		; visa id: 5357
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5358
  %4212 = ashr i64 %4211, 32		; visa id: 5358
  %4213 = bitcast i64 %4212 to <2 x i32>		; visa id: 5359
  %4214 = extractelement <2 x i32> %4213, i32 0		; visa id: 5363
  %4215 = extractelement <2 x i32> %4213, i32 1		; visa id: 5363
  %4216 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4214, i32 %4215, i32 %44, i32 %45)
  %4217 = extractvalue { i32, i32 } %4216, 0		; visa id: 5363
  %4218 = extractvalue { i32, i32 } %4216, 1		; visa id: 5363
  %4219 = insertelement <2 x i32> undef, i32 %4217, i32 0		; visa id: 5370
  %4220 = insertelement <2 x i32> %4219, i32 %4218, i32 1		; visa id: 5371
  %4221 = bitcast <2 x i32> %4220 to i64		; visa id: 5372
  %4222 = bitcast i64 %4211 to <2 x i32>		; visa id: 5376
  %4223 = extractelement <2 x i32> %4222, i32 0		; visa id: 5378
  %4224 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4223, i32 1
  %4225 = bitcast <2 x i32> %4224 to i64		; visa id: 5378
  %4226 = shl i64 %4221, 1		; visa id: 5379
  %4227 = add i64 %.in400, %4226		; visa id: 5380
  %4228 = ashr exact i64 %4225, 31		; visa id: 5381
  %4229 = add i64 %4227, %4228		; visa id: 5382
  %4230 = inttoptr i64 %4229 to i16 addrspace(4)*		; visa id: 5383
  %4231 = addrspacecast i16 addrspace(4)* %4230 to i16 addrspace(1)*		; visa id: 5383
  %4232 = load i16, i16 addrspace(1)* %4231, align 2		; visa id: 5384
  %4233 = zext i16 %4196 to i32		; visa id: 5386
  %4234 = shl nuw i32 %4233, 16, !spirv.Decorations !639		; visa id: 5387
  %4235 = bitcast i32 %4234 to float
  %4236 = zext i16 %4232 to i32		; visa id: 5388
  %4237 = shl nuw i32 %4236, 16, !spirv.Decorations !639		; visa id: 5389
  %4238 = bitcast i32 %4237 to float
  %4239 = fmul reassoc nsz arcp contract float %4235, %4238, !spirv.Decorations !618
  %4240 = fadd reassoc nsz arcp contract float %4239, %.sroa.46.1, !spirv.Decorations !618		; visa id: 5390
  br label %._crit_edge.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5391

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %4210
; BB411 :
  %.sroa.46.2 = phi float [ %4240, %4210 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  %4241 = icmp slt i32 %223, %const_reg_dword
  %4242 = icmp slt i32 %4149, %const_reg_dword1		; visa id: 5392
  %4243 = and i1 %4241, %4242		; visa id: 5393
  br i1 %4243, label %4244, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5395

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
; BB:
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4244:                                             ; preds = %._crit_edge.11
; BB413 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5397
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5397
  %4245 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 5397
  %4246 = insertelement <2 x i32> %4245, i32 %113, i64 1		; visa id: 5398
  %4247 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5399
  store <2 x i32> %4246, <2 x i32>* %4247, align 4, !noalias !625		; visa id: 5399
  br label %._crit_edge297, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5401

._crit_edge297:                                   ; preds = %._crit_edge297.._crit_edge297_crit_edge, %4244
; BB414 :
  %4248 = phi i32 [ 0, %4244 ], [ %4257, %._crit_edge297.._crit_edge297_crit_edge ]
  %4249 = zext i32 %4248 to i64		; visa id: 5402
  %4250 = shl nuw nsw i64 %4249, 2		; visa id: 5403
  %4251 = add i64 %133, %4250		; visa id: 5404
  %4252 = inttoptr i64 %4251 to i32*		; visa id: 5405
  %4253 = load i32, i32* %4252, align 4, !noalias !625		; visa id: 5405
  %4254 = add i64 %128, %4250		; visa id: 5406
  %4255 = inttoptr i64 %4254 to i32*		; visa id: 5407
  store i32 %4253, i32* %4255, align 4, !alias.scope !625		; visa id: 5407
  %4256 = icmp eq i32 %4248, 0		; visa id: 5408
  br i1 %4256, label %._crit_edge297.._crit_edge297_crit_edge, label %4258, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5409

._crit_edge297.._crit_edge297_crit_edge:          ; preds = %._crit_edge297
; BB415 :
  %4257 = add nuw nsw i32 %4248, 1, !spirv.Decorations !631		; visa id: 5411
  br label %._crit_edge297, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5412

4258:                                             ; preds = %._crit_edge297
; BB416 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5414
  %4259 = load i64, i64* %129, align 8		; visa id: 5414
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5415
  %4260 = bitcast i64 %4259 to <2 x i32>		; visa id: 5415
  %4261 = extractelement <2 x i32> %4260, i32 0		; visa id: 5417
  %4262 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4261, i32 1
  %4263 = bitcast <2 x i32> %4262 to i64		; visa id: 5417
  %4264 = ashr exact i64 %4263, 32		; visa id: 5418
  %4265 = bitcast i64 %4264 to <2 x i32>		; visa id: 5419
  %4266 = extractelement <2 x i32> %4265, i32 0		; visa id: 5423
  %4267 = extractelement <2 x i32> %4265, i32 1		; visa id: 5423
  %4268 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4266, i32 %4267, i32 %41, i32 %42)
  %4269 = extractvalue { i32, i32 } %4268, 0		; visa id: 5423
  %4270 = extractvalue { i32, i32 } %4268, 1		; visa id: 5423
  %4271 = insertelement <2 x i32> undef, i32 %4269, i32 0		; visa id: 5430
  %4272 = insertelement <2 x i32> %4271, i32 %4270, i32 1		; visa id: 5431
  %4273 = bitcast <2 x i32> %4272 to i64		; visa id: 5432
  %4274 = shl i64 %4273, 1		; visa id: 5436
  %4275 = add i64 %.in401, %4274		; visa id: 5437
  %4276 = ashr i64 %4259, 31		; visa id: 5438
  %4277 = bitcast i64 %4276 to <2 x i32>		; visa id: 5439
  %4278 = extractelement <2 x i32> %4277, i32 0		; visa id: 5443
  %4279 = extractelement <2 x i32> %4277, i32 1		; visa id: 5443
  %4280 = and i32 %4278, -2		; visa id: 5443
  %4281 = insertelement <2 x i32> undef, i32 %4280, i32 0		; visa id: 5444
  %4282 = insertelement <2 x i32> %4281, i32 %4279, i32 1		; visa id: 5445
  %4283 = bitcast <2 x i32> %4282 to i64		; visa id: 5446
  %4284 = add i64 %4275, %4283		; visa id: 5450
  %4285 = inttoptr i64 %4284 to i16 addrspace(4)*		; visa id: 5451
  %4286 = addrspacecast i16 addrspace(4)* %4285 to i16 addrspace(1)*		; visa id: 5451
  %4287 = load i16, i16 addrspace(1)* %4286, align 2		; visa id: 5452
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5454
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5454
  %4288 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5454
  %4289 = insertelement <2 x i32> %4288, i32 %4149, i64 1		; visa id: 5455
  %4290 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5456
  store <2 x i32> %4289, <2 x i32>* %4290, align 4, !noalias !635		; visa id: 5456
  br label %._crit_edge298, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5458

._crit_edge298:                                   ; preds = %._crit_edge298.._crit_edge298_crit_edge, %4258
; BB417 :
  %4291 = phi i32 [ 0, %4258 ], [ %4300, %._crit_edge298.._crit_edge298_crit_edge ]
  %4292 = zext i32 %4291 to i64		; visa id: 5459
  %4293 = shl nuw nsw i64 %4292, 2		; visa id: 5460
  %4294 = add i64 %124, %4293		; visa id: 5461
  %4295 = inttoptr i64 %4294 to i32*		; visa id: 5462
  %4296 = load i32, i32* %4295, align 4, !noalias !635		; visa id: 5462
  %4297 = add i64 %119, %4293		; visa id: 5463
  %4298 = inttoptr i64 %4297 to i32*		; visa id: 5464
  store i32 %4296, i32* %4298, align 4, !alias.scope !635		; visa id: 5464
  %4299 = icmp eq i32 %4291, 0		; visa id: 5465
  br i1 %4299, label %._crit_edge298.._crit_edge298_crit_edge, label %4301, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5466

._crit_edge298.._crit_edge298_crit_edge:          ; preds = %._crit_edge298
; BB418 :
  %4300 = add nuw nsw i32 %4291, 1, !spirv.Decorations !631		; visa id: 5468
  br label %._crit_edge298, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5469

4301:                                             ; preds = %._crit_edge298
; BB419 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5471
  %4302 = load i64, i64* %120, align 8		; visa id: 5471
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5472
  %4303 = ashr i64 %4302, 32		; visa id: 5472
  %4304 = bitcast i64 %4303 to <2 x i32>		; visa id: 5473
  %4305 = extractelement <2 x i32> %4304, i32 0		; visa id: 5477
  %4306 = extractelement <2 x i32> %4304, i32 1		; visa id: 5477
  %4307 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4305, i32 %4306, i32 %44, i32 %45)
  %4308 = extractvalue { i32, i32 } %4307, 0		; visa id: 5477
  %4309 = extractvalue { i32, i32 } %4307, 1		; visa id: 5477
  %4310 = insertelement <2 x i32> undef, i32 %4308, i32 0		; visa id: 5484
  %4311 = insertelement <2 x i32> %4310, i32 %4309, i32 1		; visa id: 5485
  %4312 = bitcast <2 x i32> %4311 to i64		; visa id: 5486
  %4313 = bitcast i64 %4302 to <2 x i32>		; visa id: 5490
  %4314 = extractelement <2 x i32> %4313, i32 0		; visa id: 5492
  %4315 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4314, i32 1
  %4316 = bitcast <2 x i32> %4315 to i64		; visa id: 5492
  %4317 = shl i64 %4312, 1		; visa id: 5493
  %4318 = add i64 %.in400, %4317		; visa id: 5494
  %4319 = ashr exact i64 %4316, 31		; visa id: 5495
  %4320 = add i64 %4318, %4319		; visa id: 5496
  %4321 = inttoptr i64 %4320 to i16 addrspace(4)*		; visa id: 5497
  %4322 = addrspacecast i16 addrspace(4)* %4321 to i16 addrspace(1)*		; visa id: 5497
  %4323 = load i16, i16 addrspace(1)* %4322, align 2		; visa id: 5498
  %4324 = zext i16 %4287 to i32		; visa id: 5500
  %4325 = shl nuw i32 %4324, 16, !spirv.Decorations !639		; visa id: 5501
  %4326 = bitcast i32 %4325 to float
  %4327 = zext i16 %4323 to i32		; visa id: 5502
  %4328 = shl nuw i32 %4327, 16, !spirv.Decorations !639		; visa id: 5503
  %4329 = bitcast i32 %4328 to float
  %4330 = fmul reassoc nsz arcp contract float %4326, %4329, !spirv.Decorations !618
  %4331 = fadd reassoc nsz arcp contract float %4330, %.sroa.110.1, !spirv.Decorations !618		; visa id: 5504
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5505

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %4301
; BB420 :
  %.sroa.110.2 = phi float [ %4331, %4301 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  %4332 = icmp slt i32 %315, %const_reg_dword
  %4333 = icmp slt i32 %4149, %const_reg_dword1		; visa id: 5506
  %4334 = and i1 %4332, %4333		; visa id: 5507
  br i1 %4334, label %4335, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5509

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
; BB:
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4335:                                             ; preds = %._crit_edge.1.11
; BB422 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5511
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5511
  %4336 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 5511
  %4337 = insertelement <2 x i32> %4336, i32 %113, i64 1		; visa id: 5512
  %4338 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5513
  store <2 x i32> %4337, <2 x i32>* %4338, align 4, !noalias !625		; visa id: 5513
  br label %._crit_edge299, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5515

._crit_edge299:                                   ; preds = %._crit_edge299.._crit_edge299_crit_edge, %4335
; BB423 :
  %4339 = phi i32 [ 0, %4335 ], [ %4348, %._crit_edge299.._crit_edge299_crit_edge ]
  %4340 = zext i32 %4339 to i64		; visa id: 5516
  %4341 = shl nuw nsw i64 %4340, 2		; visa id: 5517
  %4342 = add i64 %133, %4341		; visa id: 5518
  %4343 = inttoptr i64 %4342 to i32*		; visa id: 5519
  %4344 = load i32, i32* %4343, align 4, !noalias !625		; visa id: 5519
  %4345 = add i64 %128, %4341		; visa id: 5520
  %4346 = inttoptr i64 %4345 to i32*		; visa id: 5521
  store i32 %4344, i32* %4346, align 4, !alias.scope !625		; visa id: 5521
  %4347 = icmp eq i32 %4339, 0		; visa id: 5522
  br i1 %4347, label %._crit_edge299.._crit_edge299_crit_edge, label %4349, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5523

._crit_edge299.._crit_edge299_crit_edge:          ; preds = %._crit_edge299
; BB424 :
  %4348 = add nuw nsw i32 %4339, 1, !spirv.Decorations !631		; visa id: 5525
  br label %._crit_edge299, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5526

4349:                                             ; preds = %._crit_edge299
; BB425 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5528
  %4350 = load i64, i64* %129, align 8		; visa id: 5528
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5529
  %4351 = bitcast i64 %4350 to <2 x i32>		; visa id: 5529
  %4352 = extractelement <2 x i32> %4351, i32 0		; visa id: 5531
  %4353 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4352, i32 1
  %4354 = bitcast <2 x i32> %4353 to i64		; visa id: 5531
  %4355 = ashr exact i64 %4354, 32		; visa id: 5532
  %4356 = bitcast i64 %4355 to <2 x i32>		; visa id: 5533
  %4357 = extractelement <2 x i32> %4356, i32 0		; visa id: 5537
  %4358 = extractelement <2 x i32> %4356, i32 1		; visa id: 5537
  %4359 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4357, i32 %4358, i32 %41, i32 %42)
  %4360 = extractvalue { i32, i32 } %4359, 0		; visa id: 5537
  %4361 = extractvalue { i32, i32 } %4359, 1		; visa id: 5537
  %4362 = insertelement <2 x i32> undef, i32 %4360, i32 0		; visa id: 5544
  %4363 = insertelement <2 x i32> %4362, i32 %4361, i32 1		; visa id: 5545
  %4364 = bitcast <2 x i32> %4363 to i64		; visa id: 5546
  %4365 = shl i64 %4364, 1		; visa id: 5550
  %4366 = add i64 %.in401, %4365		; visa id: 5551
  %4367 = ashr i64 %4350, 31		; visa id: 5552
  %4368 = bitcast i64 %4367 to <2 x i32>		; visa id: 5553
  %4369 = extractelement <2 x i32> %4368, i32 0		; visa id: 5557
  %4370 = extractelement <2 x i32> %4368, i32 1		; visa id: 5557
  %4371 = and i32 %4369, -2		; visa id: 5557
  %4372 = insertelement <2 x i32> undef, i32 %4371, i32 0		; visa id: 5558
  %4373 = insertelement <2 x i32> %4372, i32 %4370, i32 1		; visa id: 5559
  %4374 = bitcast <2 x i32> %4373 to i64		; visa id: 5560
  %4375 = add i64 %4366, %4374		; visa id: 5564
  %4376 = inttoptr i64 %4375 to i16 addrspace(4)*		; visa id: 5565
  %4377 = addrspacecast i16 addrspace(4)* %4376 to i16 addrspace(1)*		; visa id: 5565
  %4378 = load i16, i16 addrspace(1)* %4377, align 2		; visa id: 5566
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5568
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5568
  %4379 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5568
  %4380 = insertelement <2 x i32> %4379, i32 %4149, i64 1		; visa id: 5569
  %4381 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5570
  store <2 x i32> %4380, <2 x i32>* %4381, align 4, !noalias !635		; visa id: 5570
  br label %._crit_edge300, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5572

._crit_edge300:                                   ; preds = %._crit_edge300.._crit_edge300_crit_edge, %4349
; BB426 :
  %4382 = phi i32 [ 0, %4349 ], [ %4391, %._crit_edge300.._crit_edge300_crit_edge ]
  %4383 = zext i32 %4382 to i64		; visa id: 5573
  %4384 = shl nuw nsw i64 %4383, 2		; visa id: 5574
  %4385 = add i64 %124, %4384		; visa id: 5575
  %4386 = inttoptr i64 %4385 to i32*		; visa id: 5576
  %4387 = load i32, i32* %4386, align 4, !noalias !635		; visa id: 5576
  %4388 = add i64 %119, %4384		; visa id: 5577
  %4389 = inttoptr i64 %4388 to i32*		; visa id: 5578
  store i32 %4387, i32* %4389, align 4, !alias.scope !635		; visa id: 5578
  %4390 = icmp eq i32 %4382, 0		; visa id: 5579
  br i1 %4390, label %._crit_edge300.._crit_edge300_crit_edge, label %4392, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5580

._crit_edge300.._crit_edge300_crit_edge:          ; preds = %._crit_edge300
; BB427 :
  %4391 = add nuw nsw i32 %4382, 1, !spirv.Decorations !631		; visa id: 5582
  br label %._crit_edge300, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5583

4392:                                             ; preds = %._crit_edge300
; BB428 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5585
  %4393 = load i64, i64* %120, align 8		; visa id: 5585
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5586
  %4394 = ashr i64 %4393, 32		; visa id: 5586
  %4395 = bitcast i64 %4394 to <2 x i32>		; visa id: 5587
  %4396 = extractelement <2 x i32> %4395, i32 0		; visa id: 5591
  %4397 = extractelement <2 x i32> %4395, i32 1		; visa id: 5591
  %4398 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4396, i32 %4397, i32 %44, i32 %45)
  %4399 = extractvalue { i32, i32 } %4398, 0		; visa id: 5591
  %4400 = extractvalue { i32, i32 } %4398, 1		; visa id: 5591
  %4401 = insertelement <2 x i32> undef, i32 %4399, i32 0		; visa id: 5598
  %4402 = insertelement <2 x i32> %4401, i32 %4400, i32 1		; visa id: 5599
  %4403 = bitcast <2 x i32> %4402 to i64		; visa id: 5600
  %4404 = bitcast i64 %4393 to <2 x i32>		; visa id: 5604
  %4405 = extractelement <2 x i32> %4404, i32 0		; visa id: 5606
  %4406 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4405, i32 1
  %4407 = bitcast <2 x i32> %4406 to i64		; visa id: 5606
  %4408 = shl i64 %4403, 1		; visa id: 5607
  %4409 = add i64 %.in400, %4408		; visa id: 5608
  %4410 = ashr exact i64 %4407, 31		; visa id: 5609
  %4411 = add i64 %4409, %4410		; visa id: 5610
  %4412 = inttoptr i64 %4411 to i16 addrspace(4)*		; visa id: 5611
  %4413 = addrspacecast i16 addrspace(4)* %4412 to i16 addrspace(1)*		; visa id: 5611
  %4414 = load i16, i16 addrspace(1)* %4413, align 2		; visa id: 5612
  %4415 = zext i16 %4378 to i32		; visa id: 5614
  %4416 = shl nuw i32 %4415, 16, !spirv.Decorations !639		; visa id: 5615
  %4417 = bitcast i32 %4416 to float
  %4418 = zext i16 %4414 to i32		; visa id: 5616
  %4419 = shl nuw i32 %4418, 16, !spirv.Decorations !639		; visa id: 5617
  %4420 = bitcast i32 %4419 to float
  %4421 = fmul reassoc nsz arcp contract float %4417, %4420, !spirv.Decorations !618
  %4422 = fadd reassoc nsz arcp contract float %4421, %.sroa.174.1, !spirv.Decorations !618		; visa id: 5618
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5619

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %4392
; BB429 :
  %.sroa.174.2 = phi float [ %4422, %4392 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  %4423 = icmp slt i32 %407, %const_reg_dword
  %4424 = icmp slt i32 %4149, %const_reg_dword1		; visa id: 5620
  %4425 = and i1 %4423, %4424		; visa id: 5621
  br i1 %4425, label %4426, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5623

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
; BB:
  br label %.preheader.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4426:                                             ; preds = %._crit_edge.2.11
; BB431 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5625
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5625
  %4427 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 5625
  %4428 = insertelement <2 x i32> %4427, i32 %113, i64 1		; visa id: 5626
  %4429 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5627
  store <2 x i32> %4428, <2 x i32>* %4429, align 4, !noalias !625		; visa id: 5627
  br label %._crit_edge301, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5629

._crit_edge301:                                   ; preds = %._crit_edge301.._crit_edge301_crit_edge, %4426
; BB432 :
  %4430 = phi i32 [ 0, %4426 ], [ %4439, %._crit_edge301.._crit_edge301_crit_edge ]
  %4431 = zext i32 %4430 to i64		; visa id: 5630
  %4432 = shl nuw nsw i64 %4431, 2		; visa id: 5631
  %4433 = add i64 %133, %4432		; visa id: 5632
  %4434 = inttoptr i64 %4433 to i32*		; visa id: 5633
  %4435 = load i32, i32* %4434, align 4, !noalias !625		; visa id: 5633
  %4436 = add i64 %128, %4432		; visa id: 5634
  %4437 = inttoptr i64 %4436 to i32*		; visa id: 5635
  store i32 %4435, i32* %4437, align 4, !alias.scope !625		; visa id: 5635
  %4438 = icmp eq i32 %4430, 0		; visa id: 5636
  br i1 %4438, label %._crit_edge301.._crit_edge301_crit_edge, label %4440, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5637

._crit_edge301.._crit_edge301_crit_edge:          ; preds = %._crit_edge301
; BB433 :
  %4439 = add nuw nsw i32 %4430, 1, !spirv.Decorations !631		; visa id: 5639
  br label %._crit_edge301, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5640

4440:                                             ; preds = %._crit_edge301
; BB434 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5642
  %4441 = load i64, i64* %129, align 8		; visa id: 5642
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5643
  %4442 = bitcast i64 %4441 to <2 x i32>		; visa id: 5643
  %4443 = extractelement <2 x i32> %4442, i32 0		; visa id: 5645
  %4444 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4443, i32 1
  %4445 = bitcast <2 x i32> %4444 to i64		; visa id: 5645
  %4446 = ashr exact i64 %4445, 32		; visa id: 5646
  %4447 = bitcast i64 %4446 to <2 x i32>		; visa id: 5647
  %4448 = extractelement <2 x i32> %4447, i32 0		; visa id: 5651
  %4449 = extractelement <2 x i32> %4447, i32 1		; visa id: 5651
  %4450 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4448, i32 %4449, i32 %41, i32 %42)
  %4451 = extractvalue { i32, i32 } %4450, 0		; visa id: 5651
  %4452 = extractvalue { i32, i32 } %4450, 1		; visa id: 5651
  %4453 = insertelement <2 x i32> undef, i32 %4451, i32 0		; visa id: 5658
  %4454 = insertelement <2 x i32> %4453, i32 %4452, i32 1		; visa id: 5659
  %4455 = bitcast <2 x i32> %4454 to i64		; visa id: 5660
  %4456 = shl i64 %4455, 1		; visa id: 5664
  %4457 = add i64 %.in401, %4456		; visa id: 5665
  %4458 = ashr i64 %4441, 31		; visa id: 5666
  %4459 = bitcast i64 %4458 to <2 x i32>		; visa id: 5667
  %4460 = extractelement <2 x i32> %4459, i32 0		; visa id: 5671
  %4461 = extractelement <2 x i32> %4459, i32 1		; visa id: 5671
  %4462 = and i32 %4460, -2		; visa id: 5671
  %4463 = insertelement <2 x i32> undef, i32 %4462, i32 0		; visa id: 5672
  %4464 = insertelement <2 x i32> %4463, i32 %4461, i32 1		; visa id: 5673
  %4465 = bitcast <2 x i32> %4464 to i64		; visa id: 5674
  %4466 = add i64 %4457, %4465		; visa id: 5678
  %4467 = inttoptr i64 %4466 to i16 addrspace(4)*		; visa id: 5679
  %4468 = addrspacecast i16 addrspace(4)* %4467 to i16 addrspace(1)*		; visa id: 5679
  %4469 = load i16, i16 addrspace(1)* %4468, align 2		; visa id: 5680
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5682
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5682
  %4470 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5682
  %4471 = insertelement <2 x i32> %4470, i32 %4149, i64 1		; visa id: 5683
  %4472 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5684
  store <2 x i32> %4471, <2 x i32>* %4472, align 4, !noalias !635		; visa id: 5684
  br label %._crit_edge302, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5686

._crit_edge302:                                   ; preds = %._crit_edge302.._crit_edge302_crit_edge, %4440
; BB435 :
  %4473 = phi i32 [ 0, %4440 ], [ %4482, %._crit_edge302.._crit_edge302_crit_edge ]
  %4474 = zext i32 %4473 to i64		; visa id: 5687
  %4475 = shl nuw nsw i64 %4474, 2		; visa id: 5688
  %4476 = add i64 %124, %4475		; visa id: 5689
  %4477 = inttoptr i64 %4476 to i32*		; visa id: 5690
  %4478 = load i32, i32* %4477, align 4, !noalias !635		; visa id: 5690
  %4479 = add i64 %119, %4475		; visa id: 5691
  %4480 = inttoptr i64 %4479 to i32*		; visa id: 5692
  store i32 %4478, i32* %4480, align 4, !alias.scope !635		; visa id: 5692
  %4481 = icmp eq i32 %4473, 0		; visa id: 5693
  br i1 %4481, label %._crit_edge302.._crit_edge302_crit_edge, label %4483, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5694

._crit_edge302.._crit_edge302_crit_edge:          ; preds = %._crit_edge302
; BB436 :
  %4482 = add nuw nsw i32 %4473, 1, !spirv.Decorations !631		; visa id: 5696
  br label %._crit_edge302, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5697

4483:                                             ; preds = %._crit_edge302
; BB437 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5699
  %4484 = load i64, i64* %120, align 8		; visa id: 5699
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5700
  %4485 = ashr i64 %4484, 32		; visa id: 5700
  %4486 = bitcast i64 %4485 to <2 x i32>		; visa id: 5701
  %4487 = extractelement <2 x i32> %4486, i32 0		; visa id: 5705
  %4488 = extractelement <2 x i32> %4486, i32 1		; visa id: 5705
  %4489 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4487, i32 %4488, i32 %44, i32 %45)
  %4490 = extractvalue { i32, i32 } %4489, 0		; visa id: 5705
  %4491 = extractvalue { i32, i32 } %4489, 1		; visa id: 5705
  %4492 = insertelement <2 x i32> undef, i32 %4490, i32 0		; visa id: 5712
  %4493 = insertelement <2 x i32> %4492, i32 %4491, i32 1		; visa id: 5713
  %4494 = bitcast <2 x i32> %4493 to i64		; visa id: 5714
  %4495 = bitcast i64 %4484 to <2 x i32>		; visa id: 5718
  %4496 = extractelement <2 x i32> %4495, i32 0		; visa id: 5720
  %4497 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4496, i32 1
  %4498 = bitcast <2 x i32> %4497 to i64		; visa id: 5720
  %4499 = shl i64 %4494, 1		; visa id: 5721
  %4500 = add i64 %.in400, %4499		; visa id: 5722
  %4501 = ashr exact i64 %4498, 31		; visa id: 5723
  %4502 = add i64 %4500, %4501		; visa id: 5724
  %4503 = inttoptr i64 %4502 to i16 addrspace(4)*		; visa id: 5725
  %4504 = addrspacecast i16 addrspace(4)* %4503 to i16 addrspace(1)*		; visa id: 5725
  %4505 = load i16, i16 addrspace(1)* %4504, align 2		; visa id: 5726
  %4506 = zext i16 %4469 to i32		; visa id: 5728
  %4507 = shl nuw i32 %4506, 16, !spirv.Decorations !639		; visa id: 5729
  %4508 = bitcast i32 %4507 to float
  %4509 = zext i16 %4505 to i32		; visa id: 5730
  %4510 = shl nuw i32 %4509, 16, !spirv.Decorations !639		; visa id: 5731
  %4511 = bitcast i32 %4510 to float
  %4512 = fmul reassoc nsz arcp contract float %4508, %4511, !spirv.Decorations !618
  %4513 = fadd reassoc nsz arcp contract float %4512, %.sroa.238.1, !spirv.Decorations !618		; visa id: 5732
  br label %.preheader.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5733

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %4483
; BB438 :
  %.sroa.238.2 = phi float [ %4513, %4483 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %4514 = add i32 %69, 12		; visa id: 5734
  %4515 = icmp slt i32 %4514, %const_reg_dword1		; visa id: 5735
  %4516 = icmp slt i32 %65, %const_reg_dword
  %4517 = and i1 %4516, %4515		; visa id: 5736
  br i1 %4517, label %4518, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5738

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
; BB:
  br label %._crit_edge.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4518:                                             ; preds = %.preheader.11
; BB440 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5740
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5740
  %4519 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 5740
  %4520 = insertelement <2 x i32> %4519, i32 %113, i64 1		; visa id: 5741
  %4521 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5742
  store <2 x i32> %4520, <2 x i32>* %4521, align 4, !noalias !625		; visa id: 5742
  br label %._crit_edge303, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5744

._crit_edge303:                                   ; preds = %._crit_edge303.._crit_edge303_crit_edge, %4518
; BB441 :
  %4522 = phi i32 [ 0, %4518 ], [ %4531, %._crit_edge303.._crit_edge303_crit_edge ]
  %4523 = zext i32 %4522 to i64		; visa id: 5745
  %4524 = shl nuw nsw i64 %4523, 2		; visa id: 5746
  %4525 = add i64 %133, %4524		; visa id: 5747
  %4526 = inttoptr i64 %4525 to i32*		; visa id: 5748
  %4527 = load i32, i32* %4526, align 4, !noalias !625		; visa id: 5748
  %4528 = add i64 %128, %4524		; visa id: 5749
  %4529 = inttoptr i64 %4528 to i32*		; visa id: 5750
  store i32 %4527, i32* %4529, align 4, !alias.scope !625		; visa id: 5750
  %4530 = icmp eq i32 %4522, 0		; visa id: 5751
  br i1 %4530, label %._crit_edge303.._crit_edge303_crit_edge, label %4532, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5752

._crit_edge303.._crit_edge303_crit_edge:          ; preds = %._crit_edge303
; BB442 :
  %4531 = add nuw nsw i32 %4522, 1, !spirv.Decorations !631		; visa id: 5754
  br label %._crit_edge303, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5755

4532:                                             ; preds = %._crit_edge303
; BB443 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5757
  %4533 = load i64, i64* %129, align 8		; visa id: 5757
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5758
  %4534 = bitcast i64 %4533 to <2 x i32>		; visa id: 5758
  %4535 = extractelement <2 x i32> %4534, i32 0		; visa id: 5760
  %4536 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4535, i32 1
  %4537 = bitcast <2 x i32> %4536 to i64		; visa id: 5760
  %4538 = ashr exact i64 %4537, 32		; visa id: 5761
  %4539 = bitcast i64 %4538 to <2 x i32>		; visa id: 5762
  %4540 = extractelement <2 x i32> %4539, i32 0		; visa id: 5766
  %4541 = extractelement <2 x i32> %4539, i32 1		; visa id: 5766
  %4542 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4540, i32 %4541, i32 %41, i32 %42)
  %4543 = extractvalue { i32, i32 } %4542, 0		; visa id: 5766
  %4544 = extractvalue { i32, i32 } %4542, 1		; visa id: 5766
  %4545 = insertelement <2 x i32> undef, i32 %4543, i32 0		; visa id: 5773
  %4546 = insertelement <2 x i32> %4545, i32 %4544, i32 1		; visa id: 5774
  %4547 = bitcast <2 x i32> %4546 to i64		; visa id: 5775
  %4548 = shl i64 %4547, 1		; visa id: 5779
  %4549 = add i64 %.in401, %4548		; visa id: 5780
  %4550 = ashr i64 %4533, 31		; visa id: 5781
  %4551 = bitcast i64 %4550 to <2 x i32>		; visa id: 5782
  %4552 = extractelement <2 x i32> %4551, i32 0		; visa id: 5786
  %4553 = extractelement <2 x i32> %4551, i32 1		; visa id: 5786
  %4554 = and i32 %4552, -2		; visa id: 5786
  %4555 = insertelement <2 x i32> undef, i32 %4554, i32 0		; visa id: 5787
  %4556 = insertelement <2 x i32> %4555, i32 %4553, i32 1		; visa id: 5788
  %4557 = bitcast <2 x i32> %4556 to i64		; visa id: 5789
  %4558 = add i64 %4549, %4557		; visa id: 5793
  %4559 = inttoptr i64 %4558 to i16 addrspace(4)*		; visa id: 5794
  %4560 = addrspacecast i16 addrspace(4)* %4559 to i16 addrspace(1)*		; visa id: 5794
  %4561 = load i16, i16 addrspace(1)* %4560, align 2		; visa id: 5795
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5797
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5797
  %4562 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5797
  %4563 = insertelement <2 x i32> %4562, i32 %4514, i64 1		; visa id: 5798
  %4564 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5799
  store <2 x i32> %4563, <2 x i32>* %4564, align 4, !noalias !635		; visa id: 5799
  br label %._crit_edge304, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5801

._crit_edge304:                                   ; preds = %._crit_edge304.._crit_edge304_crit_edge, %4532
; BB444 :
  %4565 = phi i32 [ 0, %4532 ], [ %4574, %._crit_edge304.._crit_edge304_crit_edge ]
  %4566 = zext i32 %4565 to i64		; visa id: 5802
  %4567 = shl nuw nsw i64 %4566, 2		; visa id: 5803
  %4568 = add i64 %124, %4567		; visa id: 5804
  %4569 = inttoptr i64 %4568 to i32*		; visa id: 5805
  %4570 = load i32, i32* %4569, align 4, !noalias !635		; visa id: 5805
  %4571 = add i64 %119, %4567		; visa id: 5806
  %4572 = inttoptr i64 %4571 to i32*		; visa id: 5807
  store i32 %4570, i32* %4572, align 4, !alias.scope !635		; visa id: 5807
  %4573 = icmp eq i32 %4565, 0		; visa id: 5808
  br i1 %4573, label %._crit_edge304.._crit_edge304_crit_edge, label %4575, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5809

._crit_edge304.._crit_edge304_crit_edge:          ; preds = %._crit_edge304
; BB445 :
  %4574 = add nuw nsw i32 %4565, 1, !spirv.Decorations !631		; visa id: 5811
  br label %._crit_edge304, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5812

4575:                                             ; preds = %._crit_edge304
; BB446 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5814
  %4576 = load i64, i64* %120, align 8		; visa id: 5814
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5815
  %4577 = ashr i64 %4576, 32		; visa id: 5815
  %4578 = bitcast i64 %4577 to <2 x i32>		; visa id: 5816
  %4579 = extractelement <2 x i32> %4578, i32 0		; visa id: 5820
  %4580 = extractelement <2 x i32> %4578, i32 1		; visa id: 5820
  %4581 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4579, i32 %4580, i32 %44, i32 %45)
  %4582 = extractvalue { i32, i32 } %4581, 0		; visa id: 5820
  %4583 = extractvalue { i32, i32 } %4581, 1		; visa id: 5820
  %4584 = insertelement <2 x i32> undef, i32 %4582, i32 0		; visa id: 5827
  %4585 = insertelement <2 x i32> %4584, i32 %4583, i32 1		; visa id: 5828
  %4586 = bitcast <2 x i32> %4585 to i64		; visa id: 5829
  %4587 = bitcast i64 %4576 to <2 x i32>		; visa id: 5833
  %4588 = extractelement <2 x i32> %4587, i32 0		; visa id: 5835
  %4589 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4588, i32 1
  %4590 = bitcast <2 x i32> %4589 to i64		; visa id: 5835
  %4591 = shl i64 %4586, 1		; visa id: 5836
  %4592 = add i64 %.in400, %4591		; visa id: 5837
  %4593 = ashr exact i64 %4590, 31		; visa id: 5838
  %4594 = add i64 %4592, %4593		; visa id: 5839
  %4595 = inttoptr i64 %4594 to i16 addrspace(4)*		; visa id: 5840
  %4596 = addrspacecast i16 addrspace(4)* %4595 to i16 addrspace(1)*		; visa id: 5840
  %4597 = load i16, i16 addrspace(1)* %4596, align 2		; visa id: 5841
  %4598 = zext i16 %4561 to i32		; visa id: 5843
  %4599 = shl nuw i32 %4598, 16, !spirv.Decorations !639		; visa id: 5844
  %4600 = bitcast i32 %4599 to float
  %4601 = zext i16 %4597 to i32		; visa id: 5845
  %4602 = shl nuw i32 %4601, 16, !spirv.Decorations !639		; visa id: 5846
  %4603 = bitcast i32 %4602 to float
  %4604 = fmul reassoc nsz arcp contract float %4600, %4603, !spirv.Decorations !618
  %4605 = fadd reassoc nsz arcp contract float %4604, %.sroa.50.1, !spirv.Decorations !618		; visa id: 5847
  br label %._crit_edge.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5848

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %4575
; BB447 :
  %.sroa.50.2 = phi float [ %4605, %4575 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  %4606 = icmp slt i32 %223, %const_reg_dword
  %4607 = icmp slt i32 %4514, %const_reg_dword1		; visa id: 5849
  %4608 = and i1 %4606, %4607		; visa id: 5850
  br i1 %4608, label %4609, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5852

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
; BB:
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4609:                                             ; preds = %._crit_edge.12
; BB449 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5854
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5854
  %4610 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 5854
  %4611 = insertelement <2 x i32> %4610, i32 %113, i64 1		; visa id: 5855
  %4612 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5856
  store <2 x i32> %4611, <2 x i32>* %4612, align 4, !noalias !625		; visa id: 5856
  br label %._crit_edge305, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5858

._crit_edge305:                                   ; preds = %._crit_edge305.._crit_edge305_crit_edge, %4609
; BB450 :
  %4613 = phi i32 [ 0, %4609 ], [ %4622, %._crit_edge305.._crit_edge305_crit_edge ]
  %4614 = zext i32 %4613 to i64		; visa id: 5859
  %4615 = shl nuw nsw i64 %4614, 2		; visa id: 5860
  %4616 = add i64 %133, %4615		; visa id: 5861
  %4617 = inttoptr i64 %4616 to i32*		; visa id: 5862
  %4618 = load i32, i32* %4617, align 4, !noalias !625		; visa id: 5862
  %4619 = add i64 %128, %4615		; visa id: 5863
  %4620 = inttoptr i64 %4619 to i32*		; visa id: 5864
  store i32 %4618, i32* %4620, align 4, !alias.scope !625		; visa id: 5864
  %4621 = icmp eq i32 %4613, 0		; visa id: 5865
  br i1 %4621, label %._crit_edge305.._crit_edge305_crit_edge, label %4623, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5866

._crit_edge305.._crit_edge305_crit_edge:          ; preds = %._crit_edge305
; BB451 :
  %4622 = add nuw nsw i32 %4613, 1, !spirv.Decorations !631		; visa id: 5868
  br label %._crit_edge305, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5869

4623:                                             ; preds = %._crit_edge305
; BB452 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5871
  %4624 = load i64, i64* %129, align 8		; visa id: 5871
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5872
  %4625 = bitcast i64 %4624 to <2 x i32>		; visa id: 5872
  %4626 = extractelement <2 x i32> %4625, i32 0		; visa id: 5874
  %4627 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4626, i32 1
  %4628 = bitcast <2 x i32> %4627 to i64		; visa id: 5874
  %4629 = ashr exact i64 %4628, 32		; visa id: 5875
  %4630 = bitcast i64 %4629 to <2 x i32>		; visa id: 5876
  %4631 = extractelement <2 x i32> %4630, i32 0		; visa id: 5880
  %4632 = extractelement <2 x i32> %4630, i32 1		; visa id: 5880
  %4633 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4631, i32 %4632, i32 %41, i32 %42)
  %4634 = extractvalue { i32, i32 } %4633, 0		; visa id: 5880
  %4635 = extractvalue { i32, i32 } %4633, 1		; visa id: 5880
  %4636 = insertelement <2 x i32> undef, i32 %4634, i32 0		; visa id: 5887
  %4637 = insertelement <2 x i32> %4636, i32 %4635, i32 1		; visa id: 5888
  %4638 = bitcast <2 x i32> %4637 to i64		; visa id: 5889
  %4639 = shl i64 %4638, 1		; visa id: 5893
  %4640 = add i64 %.in401, %4639		; visa id: 5894
  %4641 = ashr i64 %4624, 31		; visa id: 5895
  %4642 = bitcast i64 %4641 to <2 x i32>		; visa id: 5896
  %4643 = extractelement <2 x i32> %4642, i32 0		; visa id: 5900
  %4644 = extractelement <2 x i32> %4642, i32 1		; visa id: 5900
  %4645 = and i32 %4643, -2		; visa id: 5900
  %4646 = insertelement <2 x i32> undef, i32 %4645, i32 0		; visa id: 5901
  %4647 = insertelement <2 x i32> %4646, i32 %4644, i32 1		; visa id: 5902
  %4648 = bitcast <2 x i32> %4647 to i64		; visa id: 5903
  %4649 = add i64 %4640, %4648		; visa id: 5907
  %4650 = inttoptr i64 %4649 to i16 addrspace(4)*		; visa id: 5908
  %4651 = addrspacecast i16 addrspace(4)* %4650 to i16 addrspace(1)*		; visa id: 5908
  %4652 = load i16, i16 addrspace(1)* %4651, align 2		; visa id: 5909
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5911
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5911
  %4653 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5911
  %4654 = insertelement <2 x i32> %4653, i32 %4514, i64 1		; visa id: 5912
  %4655 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5913
  store <2 x i32> %4654, <2 x i32>* %4655, align 4, !noalias !635		; visa id: 5913
  br label %._crit_edge306, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5915

._crit_edge306:                                   ; preds = %._crit_edge306.._crit_edge306_crit_edge, %4623
; BB453 :
  %4656 = phi i32 [ 0, %4623 ], [ %4665, %._crit_edge306.._crit_edge306_crit_edge ]
  %4657 = zext i32 %4656 to i64		; visa id: 5916
  %4658 = shl nuw nsw i64 %4657, 2		; visa id: 5917
  %4659 = add i64 %124, %4658		; visa id: 5918
  %4660 = inttoptr i64 %4659 to i32*		; visa id: 5919
  %4661 = load i32, i32* %4660, align 4, !noalias !635		; visa id: 5919
  %4662 = add i64 %119, %4658		; visa id: 5920
  %4663 = inttoptr i64 %4662 to i32*		; visa id: 5921
  store i32 %4661, i32* %4663, align 4, !alias.scope !635		; visa id: 5921
  %4664 = icmp eq i32 %4656, 0		; visa id: 5922
  br i1 %4664, label %._crit_edge306.._crit_edge306_crit_edge, label %4666, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5923

._crit_edge306.._crit_edge306_crit_edge:          ; preds = %._crit_edge306
; BB454 :
  %4665 = add nuw nsw i32 %4656, 1, !spirv.Decorations !631		; visa id: 5925
  br label %._crit_edge306, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5926

4666:                                             ; preds = %._crit_edge306
; BB455 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5928
  %4667 = load i64, i64* %120, align 8		; visa id: 5928
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5929
  %4668 = ashr i64 %4667, 32		; visa id: 5929
  %4669 = bitcast i64 %4668 to <2 x i32>		; visa id: 5930
  %4670 = extractelement <2 x i32> %4669, i32 0		; visa id: 5934
  %4671 = extractelement <2 x i32> %4669, i32 1		; visa id: 5934
  %4672 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4670, i32 %4671, i32 %44, i32 %45)
  %4673 = extractvalue { i32, i32 } %4672, 0		; visa id: 5934
  %4674 = extractvalue { i32, i32 } %4672, 1		; visa id: 5934
  %4675 = insertelement <2 x i32> undef, i32 %4673, i32 0		; visa id: 5941
  %4676 = insertelement <2 x i32> %4675, i32 %4674, i32 1		; visa id: 5942
  %4677 = bitcast <2 x i32> %4676 to i64		; visa id: 5943
  %4678 = bitcast i64 %4667 to <2 x i32>		; visa id: 5947
  %4679 = extractelement <2 x i32> %4678, i32 0		; visa id: 5949
  %4680 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4679, i32 1
  %4681 = bitcast <2 x i32> %4680 to i64		; visa id: 5949
  %4682 = shl i64 %4677, 1		; visa id: 5950
  %4683 = add i64 %.in400, %4682		; visa id: 5951
  %4684 = ashr exact i64 %4681, 31		; visa id: 5952
  %4685 = add i64 %4683, %4684		; visa id: 5953
  %4686 = inttoptr i64 %4685 to i16 addrspace(4)*		; visa id: 5954
  %4687 = addrspacecast i16 addrspace(4)* %4686 to i16 addrspace(1)*		; visa id: 5954
  %4688 = load i16, i16 addrspace(1)* %4687, align 2		; visa id: 5955
  %4689 = zext i16 %4652 to i32		; visa id: 5957
  %4690 = shl nuw i32 %4689, 16, !spirv.Decorations !639		; visa id: 5958
  %4691 = bitcast i32 %4690 to float
  %4692 = zext i16 %4688 to i32		; visa id: 5959
  %4693 = shl nuw i32 %4692, 16, !spirv.Decorations !639		; visa id: 5960
  %4694 = bitcast i32 %4693 to float
  %4695 = fmul reassoc nsz arcp contract float %4691, %4694, !spirv.Decorations !618
  %4696 = fadd reassoc nsz arcp contract float %4695, %.sroa.114.1, !spirv.Decorations !618		; visa id: 5961
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5962

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %4666
; BB456 :
  %.sroa.114.2 = phi float [ %4696, %4666 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  %4697 = icmp slt i32 %315, %const_reg_dword
  %4698 = icmp slt i32 %4514, %const_reg_dword1		; visa id: 5963
  %4699 = and i1 %4697, %4698		; visa id: 5964
  br i1 %4699, label %4700, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5966

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
; BB:
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4700:                                             ; preds = %._crit_edge.1.12
; BB458 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5968
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5968
  %4701 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 5968
  %4702 = insertelement <2 x i32> %4701, i32 %113, i64 1		; visa id: 5969
  %4703 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5970
  store <2 x i32> %4702, <2 x i32>* %4703, align 4, !noalias !625		; visa id: 5970
  br label %._crit_edge307, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5972

._crit_edge307:                                   ; preds = %._crit_edge307.._crit_edge307_crit_edge, %4700
; BB459 :
  %4704 = phi i32 [ 0, %4700 ], [ %4713, %._crit_edge307.._crit_edge307_crit_edge ]
  %4705 = zext i32 %4704 to i64		; visa id: 5973
  %4706 = shl nuw nsw i64 %4705, 2		; visa id: 5974
  %4707 = add i64 %133, %4706		; visa id: 5975
  %4708 = inttoptr i64 %4707 to i32*		; visa id: 5976
  %4709 = load i32, i32* %4708, align 4, !noalias !625		; visa id: 5976
  %4710 = add i64 %128, %4706		; visa id: 5977
  %4711 = inttoptr i64 %4710 to i32*		; visa id: 5978
  store i32 %4709, i32* %4711, align 4, !alias.scope !625		; visa id: 5978
  %4712 = icmp eq i32 %4704, 0		; visa id: 5979
  br i1 %4712, label %._crit_edge307.._crit_edge307_crit_edge, label %4714, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5980

._crit_edge307.._crit_edge307_crit_edge:          ; preds = %._crit_edge307
; BB460 :
  %4713 = add nuw nsw i32 %4704, 1, !spirv.Decorations !631		; visa id: 5982
  br label %._crit_edge307, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5983

4714:                                             ; preds = %._crit_edge307
; BB461 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5985
  %4715 = load i64, i64* %129, align 8		; visa id: 5985
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5986
  %4716 = bitcast i64 %4715 to <2 x i32>		; visa id: 5986
  %4717 = extractelement <2 x i32> %4716, i32 0		; visa id: 5988
  %4718 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4717, i32 1
  %4719 = bitcast <2 x i32> %4718 to i64		; visa id: 5988
  %4720 = ashr exact i64 %4719, 32		; visa id: 5989
  %4721 = bitcast i64 %4720 to <2 x i32>		; visa id: 5990
  %4722 = extractelement <2 x i32> %4721, i32 0		; visa id: 5994
  %4723 = extractelement <2 x i32> %4721, i32 1		; visa id: 5994
  %4724 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4722, i32 %4723, i32 %41, i32 %42)
  %4725 = extractvalue { i32, i32 } %4724, 0		; visa id: 5994
  %4726 = extractvalue { i32, i32 } %4724, 1		; visa id: 5994
  %4727 = insertelement <2 x i32> undef, i32 %4725, i32 0		; visa id: 6001
  %4728 = insertelement <2 x i32> %4727, i32 %4726, i32 1		; visa id: 6002
  %4729 = bitcast <2 x i32> %4728 to i64		; visa id: 6003
  %4730 = shl i64 %4729, 1		; visa id: 6007
  %4731 = add i64 %.in401, %4730		; visa id: 6008
  %4732 = ashr i64 %4715, 31		; visa id: 6009
  %4733 = bitcast i64 %4732 to <2 x i32>		; visa id: 6010
  %4734 = extractelement <2 x i32> %4733, i32 0		; visa id: 6014
  %4735 = extractelement <2 x i32> %4733, i32 1		; visa id: 6014
  %4736 = and i32 %4734, -2		; visa id: 6014
  %4737 = insertelement <2 x i32> undef, i32 %4736, i32 0		; visa id: 6015
  %4738 = insertelement <2 x i32> %4737, i32 %4735, i32 1		; visa id: 6016
  %4739 = bitcast <2 x i32> %4738 to i64		; visa id: 6017
  %4740 = add i64 %4731, %4739		; visa id: 6021
  %4741 = inttoptr i64 %4740 to i16 addrspace(4)*		; visa id: 6022
  %4742 = addrspacecast i16 addrspace(4)* %4741 to i16 addrspace(1)*		; visa id: 6022
  %4743 = load i16, i16 addrspace(1)* %4742, align 2		; visa id: 6023
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6025
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6025
  %4744 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6025
  %4745 = insertelement <2 x i32> %4744, i32 %4514, i64 1		; visa id: 6026
  %4746 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6027
  store <2 x i32> %4745, <2 x i32>* %4746, align 4, !noalias !635		; visa id: 6027
  br label %._crit_edge308, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6029

._crit_edge308:                                   ; preds = %._crit_edge308.._crit_edge308_crit_edge, %4714
; BB462 :
  %4747 = phi i32 [ 0, %4714 ], [ %4756, %._crit_edge308.._crit_edge308_crit_edge ]
  %4748 = zext i32 %4747 to i64		; visa id: 6030
  %4749 = shl nuw nsw i64 %4748, 2		; visa id: 6031
  %4750 = add i64 %124, %4749		; visa id: 6032
  %4751 = inttoptr i64 %4750 to i32*		; visa id: 6033
  %4752 = load i32, i32* %4751, align 4, !noalias !635		; visa id: 6033
  %4753 = add i64 %119, %4749		; visa id: 6034
  %4754 = inttoptr i64 %4753 to i32*		; visa id: 6035
  store i32 %4752, i32* %4754, align 4, !alias.scope !635		; visa id: 6035
  %4755 = icmp eq i32 %4747, 0		; visa id: 6036
  br i1 %4755, label %._crit_edge308.._crit_edge308_crit_edge, label %4757, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6037

._crit_edge308.._crit_edge308_crit_edge:          ; preds = %._crit_edge308
; BB463 :
  %4756 = add nuw nsw i32 %4747, 1, !spirv.Decorations !631		; visa id: 6039
  br label %._crit_edge308, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6040

4757:                                             ; preds = %._crit_edge308
; BB464 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6042
  %4758 = load i64, i64* %120, align 8		; visa id: 6042
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6043
  %4759 = ashr i64 %4758, 32		; visa id: 6043
  %4760 = bitcast i64 %4759 to <2 x i32>		; visa id: 6044
  %4761 = extractelement <2 x i32> %4760, i32 0		; visa id: 6048
  %4762 = extractelement <2 x i32> %4760, i32 1		; visa id: 6048
  %4763 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4761, i32 %4762, i32 %44, i32 %45)
  %4764 = extractvalue { i32, i32 } %4763, 0		; visa id: 6048
  %4765 = extractvalue { i32, i32 } %4763, 1		; visa id: 6048
  %4766 = insertelement <2 x i32> undef, i32 %4764, i32 0		; visa id: 6055
  %4767 = insertelement <2 x i32> %4766, i32 %4765, i32 1		; visa id: 6056
  %4768 = bitcast <2 x i32> %4767 to i64		; visa id: 6057
  %4769 = bitcast i64 %4758 to <2 x i32>		; visa id: 6061
  %4770 = extractelement <2 x i32> %4769, i32 0		; visa id: 6063
  %4771 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4770, i32 1
  %4772 = bitcast <2 x i32> %4771 to i64		; visa id: 6063
  %4773 = shl i64 %4768, 1		; visa id: 6064
  %4774 = add i64 %.in400, %4773		; visa id: 6065
  %4775 = ashr exact i64 %4772, 31		; visa id: 6066
  %4776 = add i64 %4774, %4775		; visa id: 6067
  %4777 = inttoptr i64 %4776 to i16 addrspace(4)*		; visa id: 6068
  %4778 = addrspacecast i16 addrspace(4)* %4777 to i16 addrspace(1)*		; visa id: 6068
  %4779 = load i16, i16 addrspace(1)* %4778, align 2		; visa id: 6069
  %4780 = zext i16 %4743 to i32		; visa id: 6071
  %4781 = shl nuw i32 %4780, 16, !spirv.Decorations !639		; visa id: 6072
  %4782 = bitcast i32 %4781 to float
  %4783 = zext i16 %4779 to i32		; visa id: 6073
  %4784 = shl nuw i32 %4783, 16, !spirv.Decorations !639		; visa id: 6074
  %4785 = bitcast i32 %4784 to float
  %4786 = fmul reassoc nsz arcp contract float %4782, %4785, !spirv.Decorations !618
  %4787 = fadd reassoc nsz arcp contract float %4786, %.sroa.178.1, !spirv.Decorations !618		; visa id: 6075
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6076

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %4757
; BB465 :
  %.sroa.178.2 = phi float [ %4787, %4757 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  %4788 = icmp slt i32 %407, %const_reg_dword
  %4789 = icmp slt i32 %4514, %const_reg_dword1		; visa id: 6077
  %4790 = and i1 %4788, %4789		; visa id: 6078
  br i1 %4790, label %4791, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6080

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
; BB:
  br label %.preheader.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4791:                                             ; preds = %._crit_edge.2.12
; BB467 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6082
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6082
  %4792 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 6082
  %4793 = insertelement <2 x i32> %4792, i32 %113, i64 1		; visa id: 6083
  %4794 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6084
  store <2 x i32> %4793, <2 x i32>* %4794, align 4, !noalias !625		; visa id: 6084
  br label %._crit_edge309, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6086

._crit_edge309:                                   ; preds = %._crit_edge309.._crit_edge309_crit_edge, %4791
; BB468 :
  %4795 = phi i32 [ 0, %4791 ], [ %4804, %._crit_edge309.._crit_edge309_crit_edge ]
  %4796 = zext i32 %4795 to i64		; visa id: 6087
  %4797 = shl nuw nsw i64 %4796, 2		; visa id: 6088
  %4798 = add i64 %133, %4797		; visa id: 6089
  %4799 = inttoptr i64 %4798 to i32*		; visa id: 6090
  %4800 = load i32, i32* %4799, align 4, !noalias !625		; visa id: 6090
  %4801 = add i64 %128, %4797		; visa id: 6091
  %4802 = inttoptr i64 %4801 to i32*		; visa id: 6092
  store i32 %4800, i32* %4802, align 4, !alias.scope !625		; visa id: 6092
  %4803 = icmp eq i32 %4795, 0		; visa id: 6093
  br i1 %4803, label %._crit_edge309.._crit_edge309_crit_edge, label %4805, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6094

._crit_edge309.._crit_edge309_crit_edge:          ; preds = %._crit_edge309
; BB469 :
  %4804 = add nuw nsw i32 %4795, 1, !spirv.Decorations !631		; visa id: 6096
  br label %._crit_edge309, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6097

4805:                                             ; preds = %._crit_edge309
; BB470 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6099
  %4806 = load i64, i64* %129, align 8		; visa id: 6099
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6100
  %4807 = bitcast i64 %4806 to <2 x i32>		; visa id: 6100
  %4808 = extractelement <2 x i32> %4807, i32 0		; visa id: 6102
  %4809 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4808, i32 1
  %4810 = bitcast <2 x i32> %4809 to i64		; visa id: 6102
  %4811 = ashr exact i64 %4810, 32		; visa id: 6103
  %4812 = bitcast i64 %4811 to <2 x i32>		; visa id: 6104
  %4813 = extractelement <2 x i32> %4812, i32 0		; visa id: 6108
  %4814 = extractelement <2 x i32> %4812, i32 1		; visa id: 6108
  %4815 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4813, i32 %4814, i32 %41, i32 %42)
  %4816 = extractvalue { i32, i32 } %4815, 0		; visa id: 6108
  %4817 = extractvalue { i32, i32 } %4815, 1		; visa id: 6108
  %4818 = insertelement <2 x i32> undef, i32 %4816, i32 0		; visa id: 6115
  %4819 = insertelement <2 x i32> %4818, i32 %4817, i32 1		; visa id: 6116
  %4820 = bitcast <2 x i32> %4819 to i64		; visa id: 6117
  %4821 = shl i64 %4820, 1		; visa id: 6121
  %4822 = add i64 %.in401, %4821		; visa id: 6122
  %4823 = ashr i64 %4806, 31		; visa id: 6123
  %4824 = bitcast i64 %4823 to <2 x i32>		; visa id: 6124
  %4825 = extractelement <2 x i32> %4824, i32 0		; visa id: 6128
  %4826 = extractelement <2 x i32> %4824, i32 1		; visa id: 6128
  %4827 = and i32 %4825, -2		; visa id: 6128
  %4828 = insertelement <2 x i32> undef, i32 %4827, i32 0		; visa id: 6129
  %4829 = insertelement <2 x i32> %4828, i32 %4826, i32 1		; visa id: 6130
  %4830 = bitcast <2 x i32> %4829 to i64		; visa id: 6131
  %4831 = add i64 %4822, %4830		; visa id: 6135
  %4832 = inttoptr i64 %4831 to i16 addrspace(4)*		; visa id: 6136
  %4833 = addrspacecast i16 addrspace(4)* %4832 to i16 addrspace(1)*		; visa id: 6136
  %4834 = load i16, i16 addrspace(1)* %4833, align 2		; visa id: 6137
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6139
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6139
  %4835 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6139
  %4836 = insertelement <2 x i32> %4835, i32 %4514, i64 1		; visa id: 6140
  %4837 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6141
  store <2 x i32> %4836, <2 x i32>* %4837, align 4, !noalias !635		; visa id: 6141
  br label %._crit_edge310, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6143

._crit_edge310:                                   ; preds = %._crit_edge310.._crit_edge310_crit_edge, %4805
; BB471 :
  %4838 = phi i32 [ 0, %4805 ], [ %4847, %._crit_edge310.._crit_edge310_crit_edge ]
  %4839 = zext i32 %4838 to i64		; visa id: 6144
  %4840 = shl nuw nsw i64 %4839, 2		; visa id: 6145
  %4841 = add i64 %124, %4840		; visa id: 6146
  %4842 = inttoptr i64 %4841 to i32*		; visa id: 6147
  %4843 = load i32, i32* %4842, align 4, !noalias !635		; visa id: 6147
  %4844 = add i64 %119, %4840		; visa id: 6148
  %4845 = inttoptr i64 %4844 to i32*		; visa id: 6149
  store i32 %4843, i32* %4845, align 4, !alias.scope !635		; visa id: 6149
  %4846 = icmp eq i32 %4838, 0		; visa id: 6150
  br i1 %4846, label %._crit_edge310.._crit_edge310_crit_edge, label %4848, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6151

._crit_edge310.._crit_edge310_crit_edge:          ; preds = %._crit_edge310
; BB472 :
  %4847 = add nuw nsw i32 %4838, 1, !spirv.Decorations !631		; visa id: 6153
  br label %._crit_edge310, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6154

4848:                                             ; preds = %._crit_edge310
; BB473 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6156
  %4849 = load i64, i64* %120, align 8		; visa id: 6156
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6157
  %4850 = ashr i64 %4849, 32		; visa id: 6157
  %4851 = bitcast i64 %4850 to <2 x i32>		; visa id: 6158
  %4852 = extractelement <2 x i32> %4851, i32 0		; visa id: 6162
  %4853 = extractelement <2 x i32> %4851, i32 1		; visa id: 6162
  %4854 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4852, i32 %4853, i32 %44, i32 %45)
  %4855 = extractvalue { i32, i32 } %4854, 0		; visa id: 6162
  %4856 = extractvalue { i32, i32 } %4854, 1		; visa id: 6162
  %4857 = insertelement <2 x i32> undef, i32 %4855, i32 0		; visa id: 6169
  %4858 = insertelement <2 x i32> %4857, i32 %4856, i32 1		; visa id: 6170
  %4859 = bitcast <2 x i32> %4858 to i64		; visa id: 6171
  %4860 = bitcast i64 %4849 to <2 x i32>		; visa id: 6175
  %4861 = extractelement <2 x i32> %4860, i32 0		; visa id: 6177
  %4862 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4861, i32 1
  %4863 = bitcast <2 x i32> %4862 to i64		; visa id: 6177
  %4864 = shl i64 %4859, 1		; visa id: 6178
  %4865 = add i64 %.in400, %4864		; visa id: 6179
  %4866 = ashr exact i64 %4863, 31		; visa id: 6180
  %4867 = add i64 %4865, %4866		; visa id: 6181
  %4868 = inttoptr i64 %4867 to i16 addrspace(4)*		; visa id: 6182
  %4869 = addrspacecast i16 addrspace(4)* %4868 to i16 addrspace(1)*		; visa id: 6182
  %4870 = load i16, i16 addrspace(1)* %4869, align 2		; visa id: 6183
  %4871 = zext i16 %4834 to i32		; visa id: 6185
  %4872 = shl nuw i32 %4871, 16, !spirv.Decorations !639		; visa id: 6186
  %4873 = bitcast i32 %4872 to float
  %4874 = zext i16 %4870 to i32		; visa id: 6187
  %4875 = shl nuw i32 %4874, 16, !spirv.Decorations !639		; visa id: 6188
  %4876 = bitcast i32 %4875 to float
  %4877 = fmul reassoc nsz arcp contract float %4873, %4876, !spirv.Decorations !618
  %4878 = fadd reassoc nsz arcp contract float %4877, %.sroa.242.1, !spirv.Decorations !618		; visa id: 6189
  br label %.preheader.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6190

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %4848
; BB474 :
  %.sroa.242.2 = phi float [ %4878, %4848 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %4879 = add i32 %69, 13		; visa id: 6191
  %4880 = icmp slt i32 %4879, %const_reg_dword1		; visa id: 6192
  %4881 = icmp slt i32 %65, %const_reg_dword
  %4882 = and i1 %4881, %4880		; visa id: 6193
  br i1 %4882, label %4883, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6195

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
; BB:
  br label %._crit_edge.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4883:                                             ; preds = %.preheader.12
; BB476 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6197
  %4884 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 6197
  %4885 = insertelement <2 x i32> %4884, i32 %113, i64 1		; visa id: 6198
  %4886 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6199
  store <2 x i32> %4885, <2 x i32>* %4886, align 4, !noalias !625		; visa id: 6199
  br label %._crit_edge311, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6201

._crit_edge311:                                   ; preds = %._crit_edge311.._crit_edge311_crit_edge, %4883
; BB477 :
  %4887 = phi i32 [ 0, %4883 ], [ %4896, %._crit_edge311.._crit_edge311_crit_edge ]
  %4888 = zext i32 %4887 to i64		; visa id: 6202
  %4889 = shl nuw nsw i64 %4888, 2		; visa id: 6203
  %4890 = add i64 %133, %4889		; visa id: 6204
  %4891 = inttoptr i64 %4890 to i32*		; visa id: 6205
  %4892 = load i32, i32* %4891, align 4, !noalias !625		; visa id: 6205
  %4893 = add i64 %128, %4889		; visa id: 6206
  %4894 = inttoptr i64 %4893 to i32*		; visa id: 6207
  store i32 %4892, i32* %4894, align 4, !alias.scope !625		; visa id: 6207
  %4895 = icmp eq i32 %4887, 0		; visa id: 6208
  br i1 %4895, label %._crit_edge311.._crit_edge311_crit_edge, label %4897, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6209

._crit_edge311.._crit_edge311_crit_edge:          ; preds = %._crit_edge311
; BB478 :
  %4896 = add nuw nsw i32 %4887, 1, !spirv.Decorations !631		; visa id: 6211
  br label %._crit_edge311, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6212

4897:                                             ; preds = %._crit_edge311
; BB479 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6214
  %4898 = load i64, i64* %129, align 8		; visa id: 6214
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6215
  %4899 = bitcast i64 %4898 to <2 x i32>		; visa id: 6215
  %4900 = extractelement <2 x i32> %4899, i32 0		; visa id: 6217
  %4901 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4900, i32 1
  %4902 = bitcast <2 x i32> %4901 to i64		; visa id: 6217
  %4903 = ashr exact i64 %4902, 32		; visa id: 6218
  %4904 = bitcast i64 %4903 to <2 x i32>		; visa id: 6219
  %4905 = extractelement <2 x i32> %4904, i32 0		; visa id: 6223
  %4906 = extractelement <2 x i32> %4904, i32 1		; visa id: 6223
  %4907 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4905, i32 %4906, i32 %41, i32 %42)
  %4908 = extractvalue { i32, i32 } %4907, 0		; visa id: 6223
  %4909 = extractvalue { i32, i32 } %4907, 1		; visa id: 6223
  %4910 = insertelement <2 x i32> undef, i32 %4908, i32 0		; visa id: 6230
  %4911 = insertelement <2 x i32> %4910, i32 %4909, i32 1		; visa id: 6231
  %4912 = bitcast <2 x i32> %4911 to i64		; visa id: 6232
  %4913 = shl i64 %4912, 1		; visa id: 6236
  %4914 = add i64 %.in401, %4913		; visa id: 6237
  %4915 = ashr i64 %4898, 31		; visa id: 6238
  %4916 = bitcast i64 %4915 to <2 x i32>		; visa id: 6239
  %4917 = extractelement <2 x i32> %4916, i32 0		; visa id: 6243
  %4918 = extractelement <2 x i32> %4916, i32 1		; visa id: 6243
  %4919 = and i32 %4917, -2		; visa id: 6243
  %4920 = insertelement <2 x i32> undef, i32 %4919, i32 0		; visa id: 6244
  %4921 = insertelement <2 x i32> %4920, i32 %4918, i32 1		; visa id: 6245
  %4922 = bitcast <2 x i32> %4921 to i64		; visa id: 6246
  %4923 = add i64 %4914, %4922		; visa id: 6250
  %4924 = inttoptr i64 %4923 to i16 addrspace(4)*		; visa id: 6251
  %4925 = addrspacecast i16 addrspace(4)* %4924 to i16 addrspace(1)*		; visa id: 6251
  %4926 = load i16, i16 addrspace(1)* %4925, align 2		; visa id: 6252
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6254
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6254
  %4927 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6254
  %4928 = insertelement <2 x i32> %4927, i32 %4879, i64 1		; visa id: 6255
  %4929 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6256
  store <2 x i32> %4928, <2 x i32>* %4929, align 4, !noalias !635		; visa id: 6256
  br label %._crit_edge312, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6258

._crit_edge312:                                   ; preds = %._crit_edge312.._crit_edge312_crit_edge, %4897
; BB480 :
  %4930 = phi i32 [ 0, %4897 ], [ %4939, %._crit_edge312.._crit_edge312_crit_edge ]
  %4931 = zext i32 %4930 to i64		; visa id: 6259
  %4932 = shl nuw nsw i64 %4931, 2		; visa id: 6260
  %4933 = add i64 %124, %4932		; visa id: 6261
  %4934 = inttoptr i64 %4933 to i32*		; visa id: 6262
  %4935 = load i32, i32* %4934, align 4, !noalias !635		; visa id: 6262
  %4936 = add i64 %119, %4932		; visa id: 6263
  %4937 = inttoptr i64 %4936 to i32*		; visa id: 6264
  store i32 %4935, i32* %4937, align 4, !alias.scope !635		; visa id: 6264
  %4938 = icmp eq i32 %4930, 0		; visa id: 6265
  br i1 %4938, label %._crit_edge312.._crit_edge312_crit_edge, label %4940, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6266

._crit_edge312.._crit_edge312_crit_edge:          ; preds = %._crit_edge312
; BB481 :
  %4939 = add nuw nsw i32 %4930, 1, !spirv.Decorations !631		; visa id: 6268
  br label %._crit_edge312, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6269

4940:                                             ; preds = %._crit_edge312
; BB482 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6271
  %4941 = load i64, i64* %120, align 8		; visa id: 6271
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6272
  %4942 = ashr i64 %4941, 32		; visa id: 6272
  %4943 = bitcast i64 %4942 to <2 x i32>		; visa id: 6273
  %4944 = extractelement <2 x i32> %4943, i32 0		; visa id: 6277
  %4945 = extractelement <2 x i32> %4943, i32 1		; visa id: 6277
  %4946 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4944, i32 %4945, i32 %44, i32 %45)
  %4947 = extractvalue { i32, i32 } %4946, 0		; visa id: 6277
  %4948 = extractvalue { i32, i32 } %4946, 1		; visa id: 6277
  %4949 = insertelement <2 x i32> undef, i32 %4947, i32 0		; visa id: 6284
  %4950 = insertelement <2 x i32> %4949, i32 %4948, i32 1		; visa id: 6285
  %4951 = bitcast <2 x i32> %4950 to i64		; visa id: 6286
  %4952 = bitcast i64 %4941 to <2 x i32>		; visa id: 6290
  %4953 = extractelement <2 x i32> %4952, i32 0		; visa id: 6292
  %4954 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4953, i32 1
  %4955 = bitcast <2 x i32> %4954 to i64		; visa id: 6292
  %4956 = shl i64 %4951, 1		; visa id: 6293
  %4957 = add i64 %.in400, %4956		; visa id: 6294
  %4958 = ashr exact i64 %4955, 31		; visa id: 6295
  %4959 = add i64 %4957, %4958		; visa id: 6296
  %4960 = inttoptr i64 %4959 to i16 addrspace(4)*		; visa id: 6297
  %4961 = addrspacecast i16 addrspace(4)* %4960 to i16 addrspace(1)*		; visa id: 6297
  %4962 = load i16, i16 addrspace(1)* %4961, align 2		; visa id: 6298
  %4963 = zext i16 %4926 to i32		; visa id: 6300
  %4964 = shl nuw i32 %4963, 16, !spirv.Decorations !639		; visa id: 6301
  %4965 = bitcast i32 %4964 to float
  %4966 = zext i16 %4962 to i32		; visa id: 6302
  %4967 = shl nuw i32 %4966, 16, !spirv.Decorations !639		; visa id: 6303
  %4968 = bitcast i32 %4967 to float
  %4969 = fmul reassoc nsz arcp contract float %4965, %4968, !spirv.Decorations !618
  %4970 = fadd reassoc nsz arcp contract float %4969, %.sroa.54.1, !spirv.Decorations !618		; visa id: 6304
  br label %._crit_edge.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6305

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %4940
; BB483 :
  %.sroa.54.2 = phi float [ %4970, %4940 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  %4971 = icmp slt i32 %223, %const_reg_dword
  %4972 = icmp slt i32 %4879, %const_reg_dword1		; visa id: 6306
  %4973 = and i1 %4971, %4972		; visa id: 6307
  br i1 %4973, label %4974, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6309

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
; BB:
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4974:                                             ; preds = %._crit_edge.13
; BB485 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6311
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6311
  %4975 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 6311
  %4976 = insertelement <2 x i32> %4975, i32 %113, i64 1		; visa id: 6312
  %4977 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6313
  store <2 x i32> %4976, <2 x i32>* %4977, align 4, !noalias !625		; visa id: 6313
  br label %._crit_edge313, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6315

._crit_edge313:                                   ; preds = %._crit_edge313.._crit_edge313_crit_edge, %4974
; BB486 :
  %4978 = phi i32 [ 0, %4974 ], [ %4987, %._crit_edge313.._crit_edge313_crit_edge ]
  %4979 = zext i32 %4978 to i64		; visa id: 6316
  %4980 = shl nuw nsw i64 %4979, 2		; visa id: 6317
  %4981 = add i64 %133, %4980		; visa id: 6318
  %4982 = inttoptr i64 %4981 to i32*		; visa id: 6319
  %4983 = load i32, i32* %4982, align 4, !noalias !625		; visa id: 6319
  %4984 = add i64 %128, %4980		; visa id: 6320
  %4985 = inttoptr i64 %4984 to i32*		; visa id: 6321
  store i32 %4983, i32* %4985, align 4, !alias.scope !625		; visa id: 6321
  %4986 = icmp eq i32 %4978, 0		; visa id: 6322
  br i1 %4986, label %._crit_edge313.._crit_edge313_crit_edge, label %4988, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6323

._crit_edge313.._crit_edge313_crit_edge:          ; preds = %._crit_edge313
; BB487 :
  %4987 = add nuw nsw i32 %4978, 1, !spirv.Decorations !631		; visa id: 6325
  br label %._crit_edge313, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6326

4988:                                             ; preds = %._crit_edge313
; BB488 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6328
  %4989 = load i64, i64* %129, align 8		; visa id: 6328
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6329
  %4990 = bitcast i64 %4989 to <2 x i32>		; visa id: 6329
  %4991 = extractelement <2 x i32> %4990, i32 0		; visa id: 6331
  %4992 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4991, i32 1
  %4993 = bitcast <2 x i32> %4992 to i64		; visa id: 6331
  %4994 = ashr exact i64 %4993, 32		; visa id: 6332
  %4995 = bitcast i64 %4994 to <2 x i32>		; visa id: 6333
  %4996 = extractelement <2 x i32> %4995, i32 0		; visa id: 6337
  %4997 = extractelement <2 x i32> %4995, i32 1		; visa id: 6337
  %4998 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4996, i32 %4997, i32 %41, i32 %42)
  %4999 = extractvalue { i32, i32 } %4998, 0		; visa id: 6337
  %5000 = extractvalue { i32, i32 } %4998, 1		; visa id: 6337
  %5001 = insertelement <2 x i32> undef, i32 %4999, i32 0		; visa id: 6344
  %5002 = insertelement <2 x i32> %5001, i32 %5000, i32 1		; visa id: 6345
  %5003 = bitcast <2 x i32> %5002 to i64		; visa id: 6346
  %5004 = shl i64 %5003, 1		; visa id: 6350
  %5005 = add i64 %.in401, %5004		; visa id: 6351
  %5006 = ashr i64 %4989, 31		; visa id: 6352
  %5007 = bitcast i64 %5006 to <2 x i32>		; visa id: 6353
  %5008 = extractelement <2 x i32> %5007, i32 0		; visa id: 6357
  %5009 = extractelement <2 x i32> %5007, i32 1		; visa id: 6357
  %5010 = and i32 %5008, -2		; visa id: 6357
  %5011 = insertelement <2 x i32> undef, i32 %5010, i32 0		; visa id: 6358
  %5012 = insertelement <2 x i32> %5011, i32 %5009, i32 1		; visa id: 6359
  %5013 = bitcast <2 x i32> %5012 to i64		; visa id: 6360
  %5014 = add i64 %5005, %5013		; visa id: 6364
  %5015 = inttoptr i64 %5014 to i16 addrspace(4)*		; visa id: 6365
  %5016 = addrspacecast i16 addrspace(4)* %5015 to i16 addrspace(1)*		; visa id: 6365
  %5017 = load i16, i16 addrspace(1)* %5016, align 2		; visa id: 6366
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6368
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6368
  %5018 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6368
  %5019 = insertelement <2 x i32> %5018, i32 %4879, i64 1		; visa id: 6369
  %5020 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6370
  store <2 x i32> %5019, <2 x i32>* %5020, align 4, !noalias !635		; visa id: 6370
  br label %._crit_edge314, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6372

._crit_edge314:                                   ; preds = %._crit_edge314.._crit_edge314_crit_edge, %4988
; BB489 :
  %5021 = phi i32 [ 0, %4988 ], [ %5030, %._crit_edge314.._crit_edge314_crit_edge ]
  %5022 = zext i32 %5021 to i64		; visa id: 6373
  %5023 = shl nuw nsw i64 %5022, 2		; visa id: 6374
  %5024 = add i64 %124, %5023		; visa id: 6375
  %5025 = inttoptr i64 %5024 to i32*		; visa id: 6376
  %5026 = load i32, i32* %5025, align 4, !noalias !635		; visa id: 6376
  %5027 = add i64 %119, %5023		; visa id: 6377
  %5028 = inttoptr i64 %5027 to i32*		; visa id: 6378
  store i32 %5026, i32* %5028, align 4, !alias.scope !635		; visa id: 6378
  %5029 = icmp eq i32 %5021, 0		; visa id: 6379
  br i1 %5029, label %._crit_edge314.._crit_edge314_crit_edge, label %5031, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6380

._crit_edge314.._crit_edge314_crit_edge:          ; preds = %._crit_edge314
; BB490 :
  %5030 = add nuw nsw i32 %5021, 1, !spirv.Decorations !631		; visa id: 6382
  br label %._crit_edge314, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6383

5031:                                             ; preds = %._crit_edge314
; BB491 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6385
  %5032 = load i64, i64* %120, align 8		; visa id: 6385
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6386
  %5033 = ashr i64 %5032, 32		; visa id: 6386
  %5034 = bitcast i64 %5033 to <2 x i32>		; visa id: 6387
  %5035 = extractelement <2 x i32> %5034, i32 0		; visa id: 6391
  %5036 = extractelement <2 x i32> %5034, i32 1		; visa id: 6391
  %5037 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5035, i32 %5036, i32 %44, i32 %45)
  %5038 = extractvalue { i32, i32 } %5037, 0		; visa id: 6391
  %5039 = extractvalue { i32, i32 } %5037, 1		; visa id: 6391
  %5040 = insertelement <2 x i32> undef, i32 %5038, i32 0		; visa id: 6398
  %5041 = insertelement <2 x i32> %5040, i32 %5039, i32 1		; visa id: 6399
  %5042 = bitcast <2 x i32> %5041 to i64		; visa id: 6400
  %5043 = bitcast i64 %5032 to <2 x i32>		; visa id: 6404
  %5044 = extractelement <2 x i32> %5043, i32 0		; visa id: 6406
  %5045 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5044, i32 1
  %5046 = bitcast <2 x i32> %5045 to i64		; visa id: 6406
  %5047 = shl i64 %5042, 1		; visa id: 6407
  %5048 = add i64 %.in400, %5047		; visa id: 6408
  %5049 = ashr exact i64 %5046, 31		; visa id: 6409
  %5050 = add i64 %5048, %5049		; visa id: 6410
  %5051 = inttoptr i64 %5050 to i16 addrspace(4)*		; visa id: 6411
  %5052 = addrspacecast i16 addrspace(4)* %5051 to i16 addrspace(1)*		; visa id: 6411
  %5053 = load i16, i16 addrspace(1)* %5052, align 2		; visa id: 6412
  %5054 = zext i16 %5017 to i32		; visa id: 6414
  %5055 = shl nuw i32 %5054, 16, !spirv.Decorations !639		; visa id: 6415
  %5056 = bitcast i32 %5055 to float
  %5057 = zext i16 %5053 to i32		; visa id: 6416
  %5058 = shl nuw i32 %5057, 16, !spirv.Decorations !639		; visa id: 6417
  %5059 = bitcast i32 %5058 to float
  %5060 = fmul reassoc nsz arcp contract float %5056, %5059, !spirv.Decorations !618
  %5061 = fadd reassoc nsz arcp contract float %5060, %.sroa.118.1, !spirv.Decorations !618		; visa id: 6418
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6419

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %5031
; BB492 :
  %.sroa.118.2 = phi float [ %5061, %5031 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  %5062 = icmp slt i32 %315, %const_reg_dword
  %5063 = icmp slt i32 %4879, %const_reg_dword1		; visa id: 6420
  %5064 = and i1 %5062, %5063		; visa id: 6421
  br i1 %5064, label %5065, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6423

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
; BB:
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5065:                                             ; preds = %._crit_edge.1.13
; BB494 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6425
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6425
  %5066 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 6425
  %5067 = insertelement <2 x i32> %5066, i32 %113, i64 1		; visa id: 6426
  %5068 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6427
  store <2 x i32> %5067, <2 x i32>* %5068, align 4, !noalias !625		; visa id: 6427
  br label %._crit_edge315, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6429

._crit_edge315:                                   ; preds = %._crit_edge315.._crit_edge315_crit_edge, %5065
; BB495 :
  %5069 = phi i32 [ 0, %5065 ], [ %5078, %._crit_edge315.._crit_edge315_crit_edge ]
  %5070 = zext i32 %5069 to i64		; visa id: 6430
  %5071 = shl nuw nsw i64 %5070, 2		; visa id: 6431
  %5072 = add i64 %133, %5071		; visa id: 6432
  %5073 = inttoptr i64 %5072 to i32*		; visa id: 6433
  %5074 = load i32, i32* %5073, align 4, !noalias !625		; visa id: 6433
  %5075 = add i64 %128, %5071		; visa id: 6434
  %5076 = inttoptr i64 %5075 to i32*		; visa id: 6435
  store i32 %5074, i32* %5076, align 4, !alias.scope !625		; visa id: 6435
  %5077 = icmp eq i32 %5069, 0		; visa id: 6436
  br i1 %5077, label %._crit_edge315.._crit_edge315_crit_edge, label %5079, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6437

._crit_edge315.._crit_edge315_crit_edge:          ; preds = %._crit_edge315
; BB496 :
  %5078 = add nuw nsw i32 %5069, 1, !spirv.Decorations !631		; visa id: 6439
  br label %._crit_edge315, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6440

5079:                                             ; preds = %._crit_edge315
; BB497 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6442
  %5080 = load i64, i64* %129, align 8		; visa id: 6442
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6443
  %5081 = bitcast i64 %5080 to <2 x i32>		; visa id: 6443
  %5082 = extractelement <2 x i32> %5081, i32 0		; visa id: 6445
  %5083 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5082, i32 1
  %5084 = bitcast <2 x i32> %5083 to i64		; visa id: 6445
  %5085 = ashr exact i64 %5084, 32		; visa id: 6446
  %5086 = bitcast i64 %5085 to <2 x i32>		; visa id: 6447
  %5087 = extractelement <2 x i32> %5086, i32 0		; visa id: 6451
  %5088 = extractelement <2 x i32> %5086, i32 1		; visa id: 6451
  %5089 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5087, i32 %5088, i32 %41, i32 %42)
  %5090 = extractvalue { i32, i32 } %5089, 0		; visa id: 6451
  %5091 = extractvalue { i32, i32 } %5089, 1		; visa id: 6451
  %5092 = insertelement <2 x i32> undef, i32 %5090, i32 0		; visa id: 6458
  %5093 = insertelement <2 x i32> %5092, i32 %5091, i32 1		; visa id: 6459
  %5094 = bitcast <2 x i32> %5093 to i64		; visa id: 6460
  %5095 = shl i64 %5094, 1		; visa id: 6464
  %5096 = add i64 %.in401, %5095		; visa id: 6465
  %5097 = ashr i64 %5080, 31		; visa id: 6466
  %5098 = bitcast i64 %5097 to <2 x i32>		; visa id: 6467
  %5099 = extractelement <2 x i32> %5098, i32 0		; visa id: 6471
  %5100 = extractelement <2 x i32> %5098, i32 1		; visa id: 6471
  %5101 = and i32 %5099, -2		; visa id: 6471
  %5102 = insertelement <2 x i32> undef, i32 %5101, i32 0		; visa id: 6472
  %5103 = insertelement <2 x i32> %5102, i32 %5100, i32 1		; visa id: 6473
  %5104 = bitcast <2 x i32> %5103 to i64		; visa id: 6474
  %5105 = add i64 %5096, %5104		; visa id: 6478
  %5106 = inttoptr i64 %5105 to i16 addrspace(4)*		; visa id: 6479
  %5107 = addrspacecast i16 addrspace(4)* %5106 to i16 addrspace(1)*		; visa id: 6479
  %5108 = load i16, i16 addrspace(1)* %5107, align 2		; visa id: 6480
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6482
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6482
  %5109 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6482
  %5110 = insertelement <2 x i32> %5109, i32 %4879, i64 1		; visa id: 6483
  %5111 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6484
  store <2 x i32> %5110, <2 x i32>* %5111, align 4, !noalias !635		; visa id: 6484
  br label %._crit_edge316, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6486

._crit_edge316:                                   ; preds = %._crit_edge316.._crit_edge316_crit_edge, %5079
; BB498 :
  %5112 = phi i32 [ 0, %5079 ], [ %5121, %._crit_edge316.._crit_edge316_crit_edge ]
  %5113 = zext i32 %5112 to i64		; visa id: 6487
  %5114 = shl nuw nsw i64 %5113, 2		; visa id: 6488
  %5115 = add i64 %124, %5114		; visa id: 6489
  %5116 = inttoptr i64 %5115 to i32*		; visa id: 6490
  %5117 = load i32, i32* %5116, align 4, !noalias !635		; visa id: 6490
  %5118 = add i64 %119, %5114		; visa id: 6491
  %5119 = inttoptr i64 %5118 to i32*		; visa id: 6492
  store i32 %5117, i32* %5119, align 4, !alias.scope !635		; visa id: 6492
  %5120 = icmp eq i32 %5112, 0		; visa id: 6493
  br i1 %5120, label %._crit_edge316.._crit_edge316_crit_edge, label %5122, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6494

._crit_edge316.._crit_edge316_crit_edge:          ; preds = %._crit_edge316
; BB499 :
  %5121 = add nuw nsw i32 %5112, 1, !spirv.Decorations !631		; visa id: 6496
  br label %._crit_edge316, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6497

5122:                                             ; preds = %._crit_edge316
; BB500 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6499
  %5123 = load i64, i64* %120, align 8		; visa id: 6499
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6500
  %5124 = ashr i64 %5123, 32		; visa id: 6500
  %5125 = bitcast i64 %5124 to <2 x i32>		; visa id: 6501
  %5126 = extractelement <2 x i32> %5125, i32 0		; visa id: 6505
  %5127 = extractelement <2 x i32> %5125, i32 1		; visa id: 6505
  %5128 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5126, i32 %5127, i32 %44, i32 %45)
  %5129 = extractvalue { i32, i32 } %5128, 0		; visa id: 6505
  %5130 = extractvalue { i32, i32 } %5128, 1		; visa id: 6505
  %5131 = insertelement <2 x i32> undef, i32 %5129, i32 0		; visa id: 6512
  %5132 = insertelement <2 x i32> %5131, i32 %5130, i32 1		; visa id: 6513
  %5133 = bitcast <2 x i32> %5132 to i64		; visa id: 6514
  %5134 = bitcast i64 %5123 to <2 x i32>		; visa id: 6518
  %5135 = extractelement <2 x i32> %5134, i32 0		; visa id: 6520
  %5136 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5135, i32 1
  %5137 = bitcast <2 x i32> %5136 to i64		; visa id: 6520
  %5138 = shl i64 %5133, 1		; visa id: 6521
  %5139 = add i64 %.in400, %5138		; visa id: 6522
  %5140 = ashr exact i64 %5137, 31		; visa id: 6523
  %5141 = add i64 %5139, %5140		; visa id: 6524
  %5142 = inttoptr i64 %5141 to i16 addrspace(4)*		; visa id: 6525
  %5143 = addrspacecast i16 addrspace(4)* %5142 to i16 addrspace(1)*		; visa id: 6525
  %5144 = load i16, i16 addrspace(1)* %5143, align 2		; visa id: 6526
  %5145 = zext i16 %5108 to i32		; visa id: 6528
  %5146 = shl nuw i32 %5145, 16, !spirv.Decorations !639		; visa id: 6529
  %5147 = bitcast i32 %5146 to float
  %5148 = zext i16 %5144 to i32		; visa id: 6530
  %5149 = shl nuw i32 %5148, 16, !spirv.Decorations !639		; visa id: 6531
  %5150 = bitcast i32 %5149 to float
  %5151 = fmul reassoc nsz arcp contract float %5147, %5150, !spirv.Decorations !618
  %5152 = fadd reassoc nsz arcp contract float %5151, %.sroa.182.1, !spirv.Decorations !618		; visa id: 6532
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6533

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %5122
; BB501 :
  %.sroa.182.2 = phi float [ %5152, %5122 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  %5153 = icmp slt i32 %407, %const_reg_dword
  %5154 = icmp slt i32 %4879, %const_reg_dword1		; visa id: 6534
  %5155 = and i1 %5153, %5154		; visa id: 6535
  br i1 %5155, label %5156, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6537

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
; BB:
  br label %.preheader.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5156:                                             ; preds = %._crit_edge.2.13
; BB503 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6539
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6539
  %5157 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 6539
  %5158 = insertelement <2 x i32> %5157, i32 %113, i64 1		; visa id: 6540
  %5159 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6541
  store <2 x i32> %5158, <2 x i32>* %5159, align 4, !noalias !625		; visa id: 6541
  br label %._crit_edge317, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6543

._crit_edge317:                                   ; preds = %._crit_edge317.._crit_edge317_crit_edge, %5156
; BB504 :
  %5160 = phi i32 [ 0, %5156 ], [ %5169, %._crit_edge317.._crit_edge317_crit_edge ]
  %5161 = zext i32 %5160 to i64		; visa id: 6544
  %5162 = shl nuw nsw i64 %5161, 2		; visa id: 6545
  %5163 = add i64 %133, %5162		; visa id: 6546
  %5164 = inttoptr i64 %5163 to i32*		; visa id: 6547
  %5165 = load i32, i32* %5164, align 4, !noalias !625		; visa id: 6547
  %5166 = add i64 %128, %5162		; visa id: 6548
  %5167 = inttoptr i64 %5166 to i32*		; visa id: 6549
  store i32 %5165, i32* %5167, align 4, !alias.scope !625		; visa id: 6549
  %5168 = icmp eq i32 %5160, 0		; visa id: 6550
  br i1 %5168, label %._crit_edge317.._crit_edge317_crit_edge, label %5170, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6551

._crit_edge317.._crit_edge317_crit_edge:          ; preds = %._crit_edge317
; BB505 :
  %5169 = add nuw nsw i32 %5160, 1, !spirv.Decorations !631		; visa id: 6553
  br label %._crit_edge317, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6554

5170:                                             ; preds = %._crit_edge317
; BB506 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6556
  %5171 = load i64, i64* %129, align 8		; visa id: 6556
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6557
  %5172 = bitcast i64 %5171 to <2 x i32>		; visa id: 6557
  %5173 = extractelement <2 x i32> %5172, i32 0		; visa id: 6559
  %5174 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5173, i32 1
  %5175 = bitcast <2 x i32> %5174 to i64		; visa id: 6559
  %5176 = ashr exact i64 %5175, 32		; visa id: 6560
  %5177 = bitcast i64 %5176 to <2 x i32>		; visa id: 6561
  %5178 = extractelement <2 x i32> %5177, i32 0		; visa id: 6565
  %5179 = extractelement <2 x i32> %5177, i32 1		; visa id: 6565
  %5180 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5178, i32 %5179, i32 %41, i32 %42)
  %5181 = extractvalue { i32, i32 } %5180, 0		; visa id: 6565
  %5182 = extractvalue { i32, i32 } %5180, 1		; visa id: 6565
  %5183 = insertelement <2 x i32> undef, i32 %5181, i32 0		; visa id: 6572
  %5184 = insertelement <2 x i32> %5183, i32 %5182, i32 1		; visa id: 6573
  %5185 = bitcast <2 x i32> %5184 to i64		; visa id: 6574
  %5186 = shl i64 %5185, 1		; visa id: 6578
  %5187 = add i64 %.in401, %5186		; visa id: 6579
  %5188 = ashr i64 %5171, 31		; visa id: 6580
  %5189 = bitcast i64 %5188 to <2 x i32>		; visa id: 6581
  %5190 = extractelement <2 x i32> %5189, i32 0		; visa id: 6585
  %5191 = extractelement <2 x i32> %5189, i32 1		; visa id: 6585
  %5192 = and i32 %5190, -2		; visa id: 6585
  %5193 = insertelement <2 x i32> undef, i32 %5192, i32 0		; visa id: 6586
  %5194 = insertelement <2 x i32> %5193, i32 %5191, i32 1		; visa id: 6587
  %5195 = bitcast <2 x i32> %5194 to i64		; visa id: 6588
  %5196 = add i64 %5187, %5195		; visa id: 6592
  %5197 = inttoptr i64 %5196 to i16 addrspace(4)*		; visa id: 6593
  %5198 = addrspacecast i16 addrspace(4)* %5197 to i16 addrspace(1)*		; visa id: 6593
  %5199 = load i16, i16 addrspace(1)* %5198, align 2		; visa id: 6594
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6596
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6596
  %5200 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6596
  %5201 = insertelement <2 x i32> %5200, i32 %4879, i64 1		; visa id: 6597
  %5202 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6598
  store <2 x i32> %5201, <2 x i32>* %5202, align 4, !noalias !635		; visa id: 6598
  br label %._crit_edge318, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6600

._crit_edge318:                                   ; preds = %._crit_edge318.._crit_edge318_crit_edge, %5170
; BB507 :
  %5203 = phi i32 [ 0, %5170 ], [ %5212, %._crit_edge318.._crit_edge318_crit_edge ]
  %5204 = zext i32 %5203 to i64		; visa id: 6601
  %5205 = shl nuw nsw i64 %5204, 2		; visa id: 6602
  %5206 = add i64 %124, %5205		; visa id: 6603
  %5207 = inttoptr i64 %5206 to i32*		; visa id: 6604
  %5208 = load i32, i32* %5207, align 4, !noalias !635		; visa id: 6604
  %5209 = add i64 %119, %5205		; visa id: 6605
  %5210 = inttoptr i64 %5209 to i32*		; visa id: 6606
  store i32 %5208, i32* %5210, align 4, !alias.scope !635		; visa id: 6606
  %5211 = icmp eq i32 %5203, 0		; visa id: 6607
  br i1 %5211, label %._crit_edge318.._crit_edge318_crit_edge, label %5213, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6608

._crit_edge318.._crit_edge318_crit_edge:          ; preds = %._crit_edge318
; BB508 :
  %5212 = add nuw nsw i32 %5203, 1, !spirv.Decorations !631		; visa id: 6610
  br label %._crit_edge318, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6611

5213:                                             ; preds = %._crit_edge318
; BB509 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6613
  %5214 = load i64, i64* %120, align 8		; visa id: 6613
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6614
  %5215 = ashr i64 %5214, 32		; visa id: 6614
  %5216 = bitcast i64 %5215 to <2 x i32>		; visa id: 6615
  %5217 = extractelement <2 x i32> %5216, i32 0		; visa id: 6619
  %5218 = extractelement <2 x i32> %5216, i32 1		; visa id: 6619
  %5219 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5217, i32 %5218, i32 %44, i32 %45)
  %5220 = extractvalue { i32, i32 } %5219, 0		; visa id: 6619
  %5221 = extractvalue { i32, i32 } %5219, 1		; visa id: 6619
  %5222 = insertelement <2 x i32> undef, i32 %5220, i32 0		; visa id: 6626
  %5223 = insertelement <2 x i32> %5222, i32 %5221, i32 1		; visa id: 6627
  %5224 = bitcast <2 x i32> %5223 to i64		; visa id: 6628
  %5225 = bitcast i64 %5214 to <2 x i32>		; visa id: 6632
  %5226 = extractelement <2 x i32> %5225, i32 0		; visa id: 6634
  %5227 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5226, i32 1
  %5228 = bitcast <2 x i32> %5227 to i64		; visa id: 6634
  %5229 = shl i64 %5224, 1		; visa id: 6635
  %5230 = add i64 %.in400, %5229		; visa id: 6636
  %5231 = ashr exact i64 %5228, 31		; visa id: 6637
  %5232 = add i64 %5230, %5231		; visa id: 6638
  %5233 = inttoptr i64 %5232 to i16 addrspace(4)*		; visa id: 6639
  %5234 = addrspacecast i16 addrspace(4)* %5233 to i16 addrspace(1)*		; visa id: 6639
  %5235 = load i16, i16 addrspace(1)* %5234, align 2		; visa id: 6640
  %5236 = zext i16 %5199 to i32		; visa id: 6642
  %5237 = shl nuw i32 %5236, 16, !spirv.Decorations !639		; visa id: 6643
  %5238 = bitcast i32 %5237 to float
  %5239 = zext i16 %5235 to i32		; visa id: 6644
  %5240 = shl nuw i32 %5239, 16, !spirv.Decorations !639		; visa id: 6645
  %5241 = bitcast i32 %5240 to float
  %5242 = fmul reassoc nsz arcp contract float %5238, %5241, !spirv.Decorations !618
  %5243 = fadd reassoc nsz arcp contract float %5242, %.sroa.246.1, !spirv.Decorations !618		; visa id: 6646
  br label %.preheader.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6647

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %5213
; BB510 :
  %.sroa.246.2 = phi float [ %5243, %5213 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %5244 = add i32 %69, 14		; visa id: 6648
  %5245 = icmp slt i32 %5244, %const_reg_dword1		; visa id: 6649
  %5246 = icmp slt i32 %65, %const_reg_dword
  %5247 = and i1 %5246, %5245		; visa id: 6650
  br i1 %5247, label %5248, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6652

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
; BB:
  br label %._crit_edge.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5248:                                             ; preds = %.preheader.13
; BB512 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6654
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6654
  %5249 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 6654
  %5250 = insertelement <2 x i32> %5249, i32 %113, i64 1		; visa id: 6655
  %5251 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6656
  store <2 x i32> %5250, <2 x i32>* %5251, align 4, !noalias !625		; visa id: 6656
  br label %._crit_edge319, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6658

._crit_edge319:                                   ; preds = %._crit_edge319.._crit_edge319_crit_edge, %5248
; BB513 :
  %5252 = phi i32 [ 0, %5248 ], [ %5261, %._crit_edge319.._crit_edge319_crit_edge ]
  %5253 = zext i32 %5252 to i64		; visa id: 6659
  %5254 = shl nuw nsw i64 %5253, 2		; visa id: 6660
  %5255 = add i64 %133, %5254		; visa id: 6661
  %5256 = inttoptr i64 %5255 to i32*		; visa id: 6662
  %5257 = load i32, i32* %5256, align 4, !noalias !625		; visa id: 6662
  %5258 = add i64 %128, %5254		; visa id: 6663
  %5259 = inttoptr i64 %5258 to i32*		; visa id: 6664
  store i32 %5257, i32* %5259, align 4, !alias.scope !625		; visa id: 6664
  %5260 = icmp eq i32 %5252, 0		; visa id: 6665
  br i1 %5260, label %._crit_edge319.._crit_edge319_crit_edge, label %5262, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6666

._crit_edge319.._crit_edge319_crit_edge:          ; preds = %._crit_edge319
; BB514 :
  %5261 = add nuw nsw i32 %5252, 1, !spirv.Decorations !631		; visa id: 6668
  br label %._crit_edge319, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6669

5262:                                             ; preds = %._crit_edge319
; BB515 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6671
  %5263 = load i64, i64* %129, align 8		; visa id: 6671
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6672
  %5264 = bitcast i64 %5263 to <2 x i32>		; visa id: 6672
  %5265 = extractelement <2 x i32> %5264, i32 0		; visa id: 6674
  %5266 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5265, i32 1
  %5267 = bitcast <2 x i32> %5266 to i64		; visa id: 6674
  %5268 = ashr exact i64 %5267, 32		; visa id: 6675
  %5269 = bitcast i64 %5268 to <2 x i32>		; visa id: 6676
  %5270 = extractelement <2 x i32> %5269, i32 0		; visa id: 6680
  %5271 = extractelement <2 x i32> %5269, i32 1		; visa id: 6680
  %5272 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5270, i32 %5271, i32 %41, i32 %42)
  %5273 = extractvalue { i32, i32 } %5272, 0		; visa id: 6680
  %5274 = extractvalue { i32, i32 } %5272, 1		; visa id: 6680
  %5275 = insertelement <2 x i32> undef, i32 %5273, i32 0		; visa id: 6687
  %5276 = insertelement <2 x i32> %5275, i32 %5274, i32 1		; visa id: 6688
  %5277 = bitcast <2 x i32> %5276 to i64		; visa id: 6689
  %5278 = shl i64 %5277, 1		; visa id: 6693
  %5279 = add i64 %.in401, %5278		; visa id: 6694
  %5280 = ashr i64 %5263, 31		; visa id: 6695
  %5281 = bitcast i64 %5280 to <2 x i32>		; visa id: 6696
  %5282 = extractelement <2 x i32> %5281, i32 0		; visa id: 6700
  %5283 = extractelement <2 x i32> %5281, i32 1		; visa id: 6700
  %5284 = and i32 %5282, -2		; visa id: 6700
  %5285 = insertelement <2 x i32> undef, i32 %5284, i32 0		; visa id: 6701
  %5286 = insertelement <2 x i32> %5285, i32 %5283, i32 1		; visa id: 6702
  %5287 = bitcast <2 x i32> %5286 to i64		; visa id: 6703
  %5288 = add i64 %5279, %5287		; visa id: 6707
  %5289 = inttoptr i64 %5288 to i16 addrspace(4)*		; visa id: 6708
  %5290 = addrspacecast i16 addrspace(4)* %5289 to i16 addrspace(1)*		; visa id: 6708
  %5291 = load i16, i16 addrspace(1)* %5290, align 2		; visa id: 6709
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6711
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6711
  %5292 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6711
  %5293 = insertelement <2 x i32> %5292, i32 %5244, i64 1		; visa id: 6712
  %5294 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6713
  store <2 x i32> %5293, <2 x i32>* %5294, align 4, !noalias !635		; visa id: 6713
  br label %._crit_edge320, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6715

._crit_edge320:                                   ; preds = %._crit_edge320.._crit_edge320_crit_edge, %5262
; BB516 :
  %5295 = phi i32 [ 0, %5262 ], [ %5304, %._crit_edge320.._crit_edge320_crit_edge ]
  %5296 = zext i32 %5295 to i64		; visa id: 6716
  %5297 = shl nuw nsw i64 %5296, 2		; visa id: 6717
  %5298 = add i64 %124, %5297		; visa id: 6718
  %5299 = inttoptr i64 %5298 to i32*		; visa id: 6719
  %5300 = load i32, i32* %5299, align 4, !noalias !635		; visa id: 6719
  %5301 = add i64 %119, %5297		; visa id: 6720
  %5302 = inttoptr i64 %5301 to i32*		; visa id: 6721
  store i32 %5300, i32* %5302, align 4, !alias.scope !635		; visa id: 6721
  %5303 = icmp eq i32 %5295, 0		; visa id: 6722
  br i1 %5303, label %._crit_edge320.._crit_edge320_crit_edge, label %5305, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6723

._crit_edge320.._crit_edge320_crit_edge:          ; preds = %._crit_edge320
; BB517 :
  %5304 = add nuw nsw i32 %5295, 1, !spirv.Decorations !631		; visa id: 6725
  br label %._crit_edge320, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6726

5305:                                             ; preds = %._crit_edge320
; BB518 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6728
  %5306 = load i64, i64* %120, align 8		; visa id: 6728
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6729
  %5307 = ashr i64 %5306, 32		; visa id: 6729
  %5308 = bitcast i64 %5307 to <2 x i32>		; visa id: 6730
  %5309 = extractelement <2 x i32> %5308, i32 0		; visa id: 6734
  %5310 = extractelement <2 x i32> %5308, i32 1		; visa id: 6734
  %5311 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5309, i32 %5310, i32 %44, i32 %45)
  %5312 = extractvalue { i32, i32 } %5311, 0		; visa id: 6734
  %5313 = extractvalue { i32, i32 } %5311, 1		; visa id: 6734
  %5314 = insertelement <2 x i32> undef, i32 %5312, i32 0		; visa id: 6741
  %5315 = insertelement <2 x i32> %5314, i32 %5313, i32 1		; visa id: 6742
  %5316 = bitcast <2 x i32> %5315 to i64		; visa id: 6743
  %5317 = bitcast i64 %5306 to <2 x i32>		; visa id: 6747
  %5318 = extractelement <2 x i32> %5317, i32 0		; visa id: 6749
  %5319 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5318, i32 1
  %5320 = bitcast <2 x i32> %5319 to i64		; visa id: 6749
  %5321 = shl i64 %5316, 1		; visa id: 6750
  %5322 = add i64 %.in400, %5321		; visa id: 6751
  %5323 = ashr exact i64 %5320, 31		; visa id: 6752
  %5324 = add i64 %5322, %5323		; visa id: 6753
  %5325 = inttoptr i64 %5324 to i16 addrspace(4)*		; visa id: 6754
  %5326 = addrspacecast i16 addrspace(4)* %5325 to i16 addrspace(1)*		; visa id: 6754
  %5327 = load i16, i16 addrspace(1)* %5326, align 2		; visa id: 6755
  %5328 = zext i16 %5291 to i32		; visa id: 6757
  %5329 = shl nuw i32 %5328, 16, !spirv.Decorations !639		; visa id: 6758
  %5330 = bitcast i32 %5329 to float
  %5331 = zext i16 %5327 to i32		; visa id: 6759
  %5332 = shl nuw i32 %5331, 16, !spirv.Decorations !639		; visa id: 6760
  %5333 = bitcast i32 %5332 to float
  %5334 = fmul reassoc nsz arcp contract float %5330, %5333, !spirv.Decorations !618
  %5335 = fadd reassoc nsz arcp contract float %5334, %.sroa.58.1, !spirv.Decorations !618		; visa id: 6761
  br label %._crit_edge.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6762

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %5305
; BB519 :
  %.sroa.58.2 = phi float [ %5335, %5305 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  %5336 = icmp slt i32 %223, %const_reg_dword
  %5337 = icmp slt i32 %5244, %const_reg_dword1		; visa id: 6763
  %5338 = and i1 %5336, %5337		; visa id: 6764
  br i1 %5338, label %5339, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6766

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
; BB:
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5339:                                             ; preds = %._crit_edge.14
; BB521 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6768
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6768
  %5340 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 6768
  %5341 = insertelement <2 x i32> %5340, i32 %113, i64 1		; visa id: 6769
  %5342 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6770
  store <2 x i32> %5341, <2 x i32>* %5342, align 4, !noalias !625		; visa id: 6770
  br label %._crit_edge321, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6772

._crit_edge321:                                   ; preds = %._crit_edge321.._crit_edge321_crit_edge, %5339
; BB522 :
  %5343 = phi i32 [ 0, %5339 ], [ %5352, %._crit_edge321.._crit_edge321_crit_edge ]
  %5344 = zext i32 %5343 to i64		; visa id: 6773
  %5345 = shl nuw nsw i64 %5344, 2		; visa id: 6774
  %5346 = add i64 %133, %5345		; visa id: 6775
  %5347 = inttoptr i64 %5346 to i32*		; visa id: 6776
  %5348 = load i32, i32* %5347, align 4, !noalias !625		; visa id: 6776
  %5349 = add i64 %128, %5345		; visa id: 6777
  %5350 = inttoptr i64 %5349 to i32*		; visa id: 6778
  store i32 %5348, i32* %5350, align 4, !alias.scope !625		; visa id: 6778
  %5351 = icmp eq i32 %5343, 0		; visa id: 6779
  br i1 %5351, label %._crit_edge321.._crit_edge321_crit_edge, label %5353, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6780

._crit_edge321.._crit_edge321_crit_edge:          ; preds = %._crit_edge321
; BB523 :
  %5352 = add nuw nsw i32 %5343, 1, !spirv.Decorations !631		; visa id: 6782
  br label %._crit_edge321, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6783

5353:                                             ; preds = %._crit_edge321
; BB524 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6785
  %5354 = load i64, i64* %129, align 8		; visa id: 6785
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6786
  %5355 = bitcast i64 %5354 to <2 x i32>		; visa id: 6786
  %5356 = extractelement <2 x i32> %5355, i32 0		; visa id: 6788
  %5357 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5356, i32 1
  %5358 = bitcast <2 x i32> %5357 to i64		; visa id: 6788
  %5359 = ashr exact i64 %5358, 32		; visa id: 6789
  %5360 = bitcast i64 %5359 to <2 x i32>		; visa id: 6790
  %5361 = extractelement <2 x i32> %5360, i32 0		; visa id: 6794
  %5362 = extractelement <2 x i32> %5360, i32 1		; visa id: 6794
  %5363 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5361, i32 %5362, i32 %41, i32 %42)
  %5364 = extractvalue { i32, i32 } %5363, 0		; visa id: 6794
  %5365 = extractvalue { i32, i32 } %5363, 1		; visa id: 6794
  %5366 = insertelement <2 x i32> undef, i32 %5364, i32 0		; visa id: 6801
  %5367 = insertelement <2 x i32> %5366, i32 %5365, i32 1		; visa id: 6802
  %5368 = bitcast <2 x i32> %5367 to i64		; visa id: 6803
  %5369 = shl i64 %5368, 1		; visa id: 6807
  %5370 = add i64 %.in401, %5369		; visa id: 6808
  %5371 = ashr i64 %5354, 31		; visa id: 6809
  %5372 = bitcast i64 %5371 to <2 x i32>		; visa id: 6810
  %5373 = extractelement <2 x i32> %5372, i32 0		; visa id: 6814
  %5374 = extractelement <2 x i32> %5372, i32 1		; visa id: 6814
  %5375 = and i32 %5373, -2		; visa id: 6814
  %5376 = insertelement <2 x i32> undef, i32 %5375, i32 0		; visa id: 6815
  %5377 = insertelement <2 x i32> %5376, i32 %5374, i32 1		; visa id: 6816
  %5378 = bitcast <2 x i32> %5377 to i64		; visa id: 6817
  %5379 = add i64 %5370, %5378		; visa id: 6821
  %5380 = inttoptr i64 %5379 to i16 addrspace(4)*		; visa id: 6822
  %5381 = addrspacecast i16 addrspace(4)* %5380 to i16 addrspace(1)*		; visa id: 6822
  %5382 = load i16, i16 addrspace(1)* %5381, align 2		; visa id: 6823
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6825
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6825
  %5383 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6825
  %5384 = insertelement <2 x i32> %5383, i32 %5244, i64 1		; visa id: 6826
  %5385 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6827
  store <2 x i32> %5384, <2 x i32>* %5385, align 4, !noalias !635		; visa id: 6827
  br label %._crit_edge322, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6829

._crit_edge322:                                   ; preds = %._crit_edge322.._crit_edge322_crit_edge, %5353
; BB525 :
  %5386 = phi i32 [ 0, %5353 ], [ %5395, %._crit_edge322.._crit_edge322_crit_edge ]
  %5387 = zext i32 %5386 to i64		; visa id: 6830
  %5388 = shl nuw nsw i64 %5387, 2		; visa id: 6831
  %5389 = add i64 %124, %5388		; visa id: 6832
  %5390 = inttoptr i64 %5389 to i32*		; visa id: 6833
  %5391 = load i32, i32* %5390, align 4, !noalias !635		; visa id: 6833
  %5392 = add i64 %119, %5388		; visa id: 6834
  %5393 = inttoptr i64 %5392 to i32*		; visa id: 6835
  store i32 %5391, i32* %5393, align 4, !alias.scope !635		; visa id: 6835
  %5394 = icmp eq i32 %5386, 0		; visa id: 6836
  br i1 %5394, label %._crit_edge322.._crit_edge322_crit_edge, label %5396, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6837

._crit_edge322.._crit_edge322_crit_edge:          ; preds = %._crit_edge322
; BB526 :
  %5395 = add nuw nsw i32 %5386, 1, !spirv.Decorations !631		; visa id: 6839
  br label %._crit_edge322, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6840

5396:                                             ; preds = %._crit_edge322
; BB527 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6842
  %5397 = load i64, i64* %120, align 8		; visa id: 6842
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6843
  %5398 = ashr i64 %5397, 32		; visa id: 6843
  %5399 = bitcast i64 %5398 to <2 x i32>		; visa id: 6844
  %5400 = extractelement <2 x i32> %5399, i32 0		; visa id: 6848
  %5401 = extractelement <2 x i32> %5399, i32 1		; visa id: 6848
  %5402 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5400, i32 %5401, i32 %44, i32 %45)
  %5403 = extractvalue { i32, i32 } %5402, 0		; visa id: 6848
  %5404 = extractvalue { i32, i32 } %5402, 1		; visa id: 6848
  %5405 = insertelement <2 x i32> undef, i32 %5403, i32 0		; visa id: 6855
  %5406 = insertelement <2 x i32> %5405, i32 %5404, i32 1		; visa id: 6856
  %5407 = bitcast <2 x i32> %5406 to i64		; visa id: 6857
  %5408 = bitcast i64 %5397 to <2 x i32>		; visa id: 6861
  %5409 = extractelement <2 x i32> %5408, i32 0		; visa id: 6863
  %5410 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5409, i32 1
  %5411 = bitcast <2 x i32> %5410 to i64		; visa id: 6863
  %5412 = shl i64 %5407, 1		; visa id: 6864
  %5413 = add i64 %.in400, %5412		; visa id: 6865
  %5414 = ashr exact i64 %5411, 31		; visa id: 6866
  %5415 = add i64 %5413, %5414		; visa id: 6867
  %5416 = inttoptr i64 %5415 to i16 addrspace(4)*		; visa id: 6868
  %5417 = addrspacecast i16 addrspace(4)* %5416 to i16 addrspace(1)*		; visa id: 6868
  %5418 = load i16, i16 addrspace(1)* %5417, align 2		; visa id: 6869
  %5419 = zext i16 %5382 to i32		; visa id: 6871
  %5420 = shl nuw i32 %5419, 16, !spirv.Decorations !639		; visa id: 6872
  %5421 = bitcast i32 %5420 to float
  %5422 = zext i16 %5418 to i32		; visa id: 6873
  %5423 = shl nuw i32 %5422, 16, !spirv.Decorations !639		; visa id: 6874
  %5424 = bitcast i32 %5423 to float
  %5425 = fmul reassoc nsz arcp contract float %5421, %5424, !spirv.Decorations !618
  %5426 = fadd reassoc nsz arcp contract float %5425, %.sroa.122.1, !spirv.Decorations !618		; visa id: 6875
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6876

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %5396
; BB528 :
  %.sroa.122.2 = phi float [ %5426, %5396 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  %5427 = icmp slt i32 %315, %const_reg_dword
  %5428 = icmp slt i32 %5244, %const_reg_dword1		; visa id: 6877
  %5429 = and i1 %5427, %5428		; visa id: 6878
  br i1 %5429, label %5430, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6880

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
; BB:
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5430:                                             ; preds = %._crit_edge.1.14
; BB530 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6882
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6882
  %5431 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 6882
  %5432 = insertelement <2 x i32> %5431, i32 %113, i64 1		; visa id: 6883
  %5433 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6884
  store <2 x i32> %5432, <2 x i32>* %5433, align 4, !noalias !625		; visa id: 6884
  br label %._crit_edge323, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6886

._crit_edge323:                                   ; preds = %._crit_edge323.._crit_edge323_crit_edge, %5430
; BB531 :
  %5434 = phi i32 [ 0, %5430 ], [ %5443, %._crit_edge323.._crit_edge323_crit_edge ]
  %5435 = zext i32 %5434 to i64		; visa id: 6887
  %5436 = shl nuw nsw i64 %5435, 2		; visa id: 6888
  %5437 = add i64 %133, %5436		; visa id: 6889
  %5438 = inttoptr i64 %5437 to i32*		; visa id: 6890
  %5439 = load i32, i32* %5438, align 4, !noalias !625		; visa id: 6890
  %5440 = add i64 %128, %5436		; visa id: 6891
  %5441 = inttoptr i64 %5440 to i32*		; visa id: 6892
  store i32 %5439, i32* %5441, align 4, !alias.scope !625		; visa id: 6892
  %5442 = icmp eq i32 %5434, 0		; visa id: 6893
  br i1 %5442, label %._crit_edge323.._crit_edge323_crit_edge, label %5444, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6894

._crit_edge323.._crit_edge323_crit_edge:          ; preds = %._crit_edge323
; BB532 :
  %5443 = add nuw nsw i32 %5434, 1, !spirv.Decorations !631		; visa id: 6896
  br label %._crit_edge323, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6897

5444:                                             ; preds = %._crit_edge323
; BB533 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6899
  %5445 = load i64, i64* %129, align 8		; visa id: 6899
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6900
  %5446 = bitcast i64 %5445 to <2 x i32>		; visa id: 6900
  %5447 = extractelement <2 x i32> %5446, i32 0		; visa id: 6902
  %5448 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5447, i32 1
  %5449 = bitcast <2 x i32> %5448 to i64		; visa id: 6902
  %5450 = ashr exact i64 %5449, 32		; visa id: 6903
  %5451 = bitcast i64 %5450 to <2 x i32>		; visa id: 6904
  %5452 = extractelement <2 x i32> %5451, i32 0		; visa id: 6908
  %5453 = extractelement <2 x i32> %5451, i32 1		; visa id: 6908
  %5454 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5452, i32 %5453, i32 %41, i32 %42)
  %5455 = extractvalue { i32, i32 } %5454, 0		; visa id: 6908
  %5456 = extractvalue { i32, i32 } %5454, 1		; visa id: 6908
  %5457 = insertelement <2 x i32> undef, i32 %5455, i32 0		; visa id: 6915
  %5458 = insertelement <2 x i32> %5457, i32 %5456, i32 1		; visa id: 6916
  %5459 = bitcast <2 x i32> %5458 to i64		; visa id: 6917
  %5460 = shl i64 %5459, 1		; visa id: 6921
  %5461 = add i64 %.in401, %5460		; visa id: 6922
  %5462 = ashr i64 %5445, 31		; visa id: 6923
  %5463 = bitcast i64 %5462 to <2 x i32>		; visa id: 6924
  %5464 = extractelement <2 x i32> %5463, i32 0		; visa id: 6928
  %5465 = extractelement <2 x i32> %5463, i32 1		; visa id: 6928
  %5466 = and i32 %5464, -2		; visa id: 6928
  %5467 = insertelement <2 x i32> undef, i32 %5466, i32 0		; visa id: 6929
  %5468 = insertelement <2 x i32> %5467, i32 %5465, i32 1		; visa id: 6930
  %5469 = bitcast <2 x i32> %5468 to i64		; visa id: 6931
  %5470 = add i64 %5461, %5469		; visa id: 6935
  %5471 = inttoptr i64 %5470 to i16 addrspace(4)*		; visa id: 6936
  %5472 = addrspacecast i16 addrspace(4)* %5471 to i16 addrspace(1)*		; visa id: 6936
  %5473 = load i16, i16 addrspace(1)* %5472, align 2		; visa id: 6937
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6939
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6939
  %5474 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6939
  %5475 = insertelement <2 x i32> %5474, i32 %5244, i64 1		; visa id: 6940
  %5476 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6941
  store <2 x i32> %5475, <2 x i32>* %5476, align 4, !noalias !635		; visa id: 6941
  br label %._crit_edge324, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6943

._crit_edge324:                                   ; preds = %._crit_edge324.._crit_edge324_crit_edge, %5444
; BB534 :
  %5477 = phi i32 [ 0, %5444 ], [ %5486, %._crit_edge324.._crit_edge324_crit_edge ]
  %5478 = zext i32 %5477 to i64		; visa id: 6944
  %5479 = shl nuw nsw i64 %5478, 2		; visa id: 6945
  %5480 = add i64 %124, %5479		; visa id: 6946
  %5481 = inttoptr i64 %5480 to i32*		; visa id: 6947
  %5482 = load i32, i32* %5481, align 4, !noalias !635		; visa id: 6947
  %5483 = add i64 %119, %5479		; visa id: 6948
  %5484 = inttoptr i64 %5483 to i32*		; visa id: 6949
  store i32 %5482, i32* %5484, align 4, !alias.scope !635		; visa id: 6949
  %5485 = icmp eq i32 %5477, 0		; visa id: 6950
  br i1 %5485, label %._crit_edge324.._crit_edge324_crit_edge, label %5487, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6951

._crit_edge324.._crit_edge324_crit_edge:          ; preds = %._crit_edge324
; BB535 :
  %5486 = add nuw nsw i32 %5477, 1, !spirv.Decorations !631		; visa id: 6953
  br label %._crit_edge324, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6954

5487:                                             ; preds = %._crit_edge324
; BB536 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6956
  %5488 = load i64, i64* %120, align 8		; visa id: 6956
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6957
  %5489 = ashr i64 %5488, 32		; visa id: 6957
  %5490 = bitcast i64 %5489 to <2 x i32>		; visa id: 6958
  %5491 = extractelement <2 x i32> %5490, i32 0		; visa id: 6962
  %5492 = extractelement <2 x i32> %5490, i32 1		; visa id: 6962
  %5493 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5491, i32 %5492, i32 %44, i32 %45)
  %5494 = extractvalue { i32, i32 } %5493, 0		; visa id: 6962
  %5495 = extractvalue { i32, i32 } %5493, 1		; visa id: 6962
  %5496 = insertelement <2 x i32> undef, i32 %5494, i32 0		; visa id: 6969
  %5497 = insertelement <2 x i32> %5496, i32 %5495, i32 1		; visa id: 6970
  %5498 = bitcast <2 x i32> %5497 to i64		; visa id: 6971
  %5499 = bitcast i64 %5488 to <2 x i32>		; visa id: 6975
  %5500 = extractelement <2 x i32> %5499, i32 0		; visa id: 6977
  %5501 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5500, i32 1
  %5502 = bitcast <2 x i32> %5501 to i64		; visa id: 6977
  %5503 = shl i64 %5498, 1		; visa id: 6978
  %5504 = add i64 %.in400, %5503		; visa id: 6979
  %5505 = ashr exact i64 %5502, 31		; visa id: 6980
  %5506 = add i64 %5504, %5505		; visa id: 6981
  %5507 = inttoptr i64 %5506 to i16 addrspace(4)*		; visa id: 6982
  %5508 = addrspacecast i16 addrspace(4)* %5507 to i16 addrspace(1)*		; visa id: 6982
  %5509 = load i16, i16 addrspace(1)* %5508, align 2		; visa id: 6983
  %5510 = zext i16 %5473 to i32		; visa id: 6985
  %5511 = shl nuw i32 %5510, 16, !spirv.Decorations !639		; visa id: 6986
  %5512 = bitcast i32 %5511 to float
  %5513 = zext i16 %5509 to i32		; visa id: 6987
  %5514 = shl nuw i32 %5513, 16, !spirv.Decorations !639		; visa id: 6988
  %5515 = bitcast i32 %5514 to float
  %5516 = fmul reassoc nsz arcp contract float %5512, %5515, !spirv.Decorations !618
  %5517 = fadd reassoc nsz arcp contract float %5516, %.sroa.186.1, !spirv.Decorations !618		; visa id: 6989
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6990

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %5487
; BB537 :
  %.sroa.186.2 = phi float [ %5517, %5487 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  %5518 = icmp slt i32 %407, %const_reg_dword
  %5519 = icmp slt i32 %5244, %const_reg_dword1		; visa id: 6991
  %5520 = and i1 %5518, %5519		; visa id: 6992
  br i1 %5520, label %5521, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6994

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
; BB:
  br label %.preheader.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5521:                                             ; preds = %._crit_edge.2.14
; BB539 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6996
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6996
  %5522 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 6996
  %5523 = insertelement <2 x i32> %5522, i32 %113, i64 1		; visa id: 6997
  %5524 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6998
  store <2 x i32> %5523, <2 x i32>* %5524, align 4, !noalias !625		; visa id: 6998
  br label %._crit_edge325, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7000

._crit_edge325:                                   ; preds = %._crit_edge325.._crit_edge325_crit_edge, %5521
; BB540 :
  %5525 = phi i32 [ 0, %5521 ], [ %5534, %._crit_edge325.._crit_edge325_crit_edge ]
  %5526 = zext i32 %5525 to i64		; visa id: 7001
  %5527 = shl nuw nsw i64 %5526, 2		; visa id: 7002
  %5528 = add i64 %133, %5527		; visa id: 7003
  %5529 = inttoptr i64 %5528 to i32*		; visa id: 7004
  %5530 = load i32, i32* %5529, align 4, !noalias !625		; visa id: 7004
  %5531 = add i64 %128, %5527		; visa id: 7005
  %5532 = inttoptr i64 %5531 to i32*		; visa id: 7006
  store i32 %5530, i32* %5532, align 4, !alias.scope !625		; visa id: 7006
  %5533 = icmp eq i32 %5525, 0		; visa id: 7007
  br i1 %5533, label %._crit_edge325.._crit_edge325_crit_edge, label %5535, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7008

._crit_edge325.._crit_edge325_crit_edge:          ; preds = %._crit_edge325
; BB541 :
  %5534 = add nuw nsw i32 %5525, 1, !spirv.Decorations !631		; visa id: 7010
  br label %._crit_edge325, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7011

5535:                                             ; preds = %._crit_edge325
; BB542 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7013
  %5536 = load i64, i64* %129, align 8		; visa id: 7013
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7014
  %5537 = bitcast i64 %5536 to <2 x i32>		; visa id: 7014
  %5538 = extractelement <2 x i32> %5537, i32 0		; visa id: 7016
  %5539 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5538, i32 1
  %5540 = bitcast <2 x i32> %5539 to i64		; visa id: 7016
  %5541 = ashr exact i64 %5540, 32		; visa id: 7017
  %5542 = bitcast i64 %5541 to <2 x i32>		; visa id: 7018
  %5543 = extractelement <2 x i32> %5542, i32 0		; visa id: 7022
  %5544 = extractelement <2 x i32> %5542, i32 1		; visa id: 7022
  %5545 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5543, i32 %5544, i32 %41, i32 %42)
  %5546 = extractvalue { i32, i32 } %5545, 0		; visa id: 7022
  %5547 = extractvalue { i32, i32 } %5545, 1		; visa id: 7022
  %5548 = insertelement <2 x i32> undef, i32 %5546, i32 0		; visa id: 7029
  %5549 = insertelement <2 x i32> %5548, i32 %5547, i32 1		; visa id: 7030
  %5550 = bitcast <2 x i32> %5549 to i64		; visa id: 7031
  %5551 = shl i64 %5550, 1		; visa id: 7035
  %5552 = add i64 %.in401, %5551		; visa id: 7036
  %5553 = ashr i64 %5536, 31		; visa id: 7037
  %5554 = bitcast i64 %5553 to <2 x i32>		; visa id: 7038
  %5555 = extractelement <2 x i32> %5554, i32 0		; visa id: 7042
  %5556 = extractelement <2 x i32> %5554, i32 1		; visa id: 7042
  %5557 = and i32 %5555, -2		; visa id: 7042
  %5558 = insertelement <2 x i32> undef, i32 %5557, i32 0		; visa id: 7043
  %5559 = insertelement <2 x i32> %5558, i32 %5556, i32 1		; visa id: 7044
  %5560 = bitcast <2 x i32> %5559 to i64		; visa id: 7045
  %5561 = add i64 %5552, %5560		; visa id: 7049
  %5562 = inttoptr i64 %5561 to i16 addrspace(4)*		; visa id: 7050
  %5563 = addrspacecast i16 addrspace(4)* %5562 to i16 addrspace(1)*		; visa id: 7050
  %5564 = load i16, i16 addrspace(1)* %5563, align 2		; visa id: 7051
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7053
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7053
  %5565 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7053
  %5566 = insertelement <2 x i32> %5565, i32 %5244, i64 1		; visa id: 7054
  %5567 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7055
  store <2 x i32> %5566, <2 x i32>* %5567, align 4, !noalias !635		; visa id: 7055
  br label %._crit_edge326, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7057

._crit_edge326:                                   ; preds = %._crit_edge326.._crit_edge326_crit_edge, %5535
; BB543 :
  %5568 = phi i32 [ 0, %5535 ], [ %5577, %._crit_edge326.._crit_edge326_crit_edge ]
  %5569 = zext i32 %5568 to i64		; visa id: 7058
  %5570 = shl nuw nsw i64 %5569, 2		; visa id: 7059
  %5571 = add i64 %124, %5570		; visa id: 7060
  %5572 = inttoptr i64 %5571 to i32*		; visa id: 7061
  %5573 = load i32, i32* %5572, align 4, !noalias !635		; visa id: 7061
  %5574 = add i64 %119, %5570		; visa id: 7062
  %5575 = inttoptr i64 %5574 to i32*		; visa id: 7063
  store i32 %5573, i32* %5575, align 4, !alias.scope !635		; visa id: 7063
  %5576 = icmp eq i32 %5568, 0		; visa id: 7064
  br i1 %5576, label %._crit_edge326.._crit_edge326_crit_edge, label %5578, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7065

._crit_edge326.._crit_edge326_crit_edge:          ; preds = %._crit_edge326
; BB544 :
  %5577 = add nuw nsw i32 %5568, 1, !spirv.Decorations !631		; visa id: 7067
  br label %._crit_edge326, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7068

5578:                                             ; preds = %._crit_edge326
; BB545 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7070
  %5579 = load i64, i64* %120, align 8		; visa id: 7070
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7071
  %5580 = ashr i64 %5579, 32		; visa id: 7071
  %5581 = bitcast i64 %5580 to <2 x i32>		; visa id: 7072
  %5582 = extractelement <2 x i32> %5581, i32 0		; visa id: 7076
  %5583 = extractelement <2 x i32> %5581, i32 1		; visa id: 7076
  %5584 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5582, i32 %5583, i32 %44, i32 %45)
  %5585 = extractvalue { i32, i32 } %5584, 0		; visa id: 7076
  %5586 = extractvalue { i32, i32 } %5584, 1		; visa id: 7076
  %5587 = insertelement <2 x i32> undef, i32 %5585, i32 0		; visa id: 7083
  %5588 = insertelement <2 x i32> %5587, i32 %5586, i32 1		; visa id: 7084
  %5589 = bitcast <2 x i32> %5588 to i64		; visa id: 7085
  %5590 = bitcast i64 %5579 to <2 x i32>		; visa id: 7089
  %5591 = extractelement <2 x i32> %5590, i32 0		; visa id: 7091
  %5592 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5591, i32 1
  %5593 = bitcast <2 x i32> %5592 to i64		; visa id: 7091
  %5594 = shl i64 %5589, 1		; visa id: 7092
  %5595 = add i64 %.in400, %5594		; visa id: 7093
  %5596 = ashr exact i64 %5593, 31		; visa id: 7094
  %5597 = add i64 %5595, %5596		; visa id: 7095
  %5598 = inttoptr i64 %5597 to i16 addrspace(4)*		; visa id: 7096
  %5599 = addrspacecast i16 addrspace(4)* %5598 to i16 addrspace(1)*		; visa id: 7096
  %5600 = load i16, i16 addrspace(1)* %5599, align 2		; visa id: 7097
  %5601 = zext i16 %5564 to i32		; visa id: 7099
  %5602 = shl nuw i32 %5601, 16, !spirv.Decorations !639		; visa id: 7100
  %5603 = bitcast i32 %5602 to float
  %5604 = zext i16 %5600 to i32		; visa id: 7101
  %5605 = shl nuw i32 %5604, 16, !spirv.Decorations !639		; visa id: 7102
  %5606 = bitcast i32 %5605 to float
  %5607 = fmul reassoc nsz arcp contract float %5603, %5606, !spirv.Decorations !618
  %5608 = fadd reassoc nsz arcp contract float %5607, %.sroa.250.1, !spirv.Decorations !618		; visa id: 7103
  br label %.preheader.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7104

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %5578
; BB546 :
  %.sroa.250.2 = phi float [ %5608, %5578 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %5609 = add i32 %69, 15		; visa id: 7105
  %5610 = icmp slt i32 %5609, %const_reg_dword1		; visa id: 7106
  %5611 = icmp slt i32 %65, %const_reg_dword
  %5612 = and i1 %5611, %5610		; visa id: 7107
  br i1 %5612, label %5613, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7109

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
; BB:
  br label %._crit_edge.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5613:                                             ; preds = %.preheader.14
; BB548 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7111
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7111
  %5614 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 7111
  %5615 = insertelement <2 x i32> %5614, i32 %113, i64 1		; visa id: 7112
  %5616 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7113
  store <2 x i32> %5615, <2 x i32>* %5616, align 4, !noalias !625		; visa id: 7113
  br label %._crit_edge327, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7115

._crit_edge327:                                   ; preds = %._crit_edge327.._crit_edge327_crit_edge, %5613
; BB549 :
  %5617 = phi i32 [ 0, %5613 ], [ %5626, %._crit_edge327.._crit_edge327_crit_edge ]
  %5618 = zext i32 %5617 to i64		; visa id: 7116
  %5619 = shl nuw nsw i64 %5618, 2		; visa id: 7117
  %5620 = add i64 %133, %5619		; visa id: 7118
  %5621 = inttoptr i64 %5620 to i32*		; visa id: 7119
  %5622 = load i32, i32* %5621, align 4, !noalias !625		; visa id: 7119
  %5623 = add i64 %128, %5619		; visa id: 7120
  %5624 = inttoptr i64 %5623 to i32*		; visa id: 7121
  store i32 %5622, i32* %5624, align 4, !alias.scope !625		; visa id: 7121
  %5625 = icmp eq i32 %5617, 0		; visa id: 7122
  br i1 %5625, label %._crit_edge327.._crit_edge327_crit_edge, label %5627, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7123

._crit_edge327.._crit_edge327_crit_edge:          ; preds = %._crit_edge327
; BB550 :
  %5626 = add nuw nsw i32 %5617, 1, !spirv.Decorations !631		; visa id: 7125
  br label %._crit_edge327, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7126

5627:                                             ; preds = %._crit_edge327
; BB551 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7128
  %5628 = load i64, i64* %129, align 8		; visa id: 7128
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7129
  %5629 = bitcast i64 %5628 to <2 x i32>		; visa id: 7129
  %5630 = extractelement <2 x i32> %5629, i32 0		; visa id: 7131
  %5631 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5630, i32 1
  %5632 = bitcast <2 x i32> %5631 to i64		; visa id: 7131
  %5633 = ashr exact i64 %5632, 32		; visa id: 7132
  %5634 = bitcast i64 %5633 to <2 x i32>		; visa id: 7133
  %5635 = extractelement <2 x i32> %5634, i32 0		; visa id: 7137
  %5636 = extractelement <2 x i32> %5634, i32 1		; visa id: 7137
  %5637 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5635, i32 %5636, i32 %41, i32 %42)
  %5638 = extractvalue { i32, i32 } %5637, 0		; visa id: 7137
  %5639 = extractvalue { i32, i32 } %5637, 1		; visa id: 7137
  %5640 = insertelement <2 x i32> undef, i32 %5638, i32 0		; visa id: 7144
  %5641 = insertelement <2 x i32> %5640, i32 %5639, i32 1		; visa id: 7145
  %5642 = bitcast <2 x i32> %5641 to i64		; visa id: 7146
  %5643 = shl i64 %5642, 1		; visa id: 7150
  %5644 = add i64 %.in401, %5643		; visa id: 7151
  %5645 = ashr i64 %5628, 31		; visa id: 7152
  %5646 = bitcast i64 %5645 to <2 x i32>		; visa id: 7153
  %5647 = extractelement <2 x i32> %5646, i32 0		; visa id: 7157
  %5648 = extractelement <2 x i32> %5646, i32 1		; visa id: 7157
  %5649 = and i32 %5647, -2		; visa id: 7157
  %5650 = insertelement <2 x i32> undef, i32 %5649, i32 0		; visa id: 7158
  %5651 = insertelement <2 x i32> %5650, i32 %5648, i32 1		; visa id: 7159
  %5652 = bitcast <2 x i32> %5651 to i64		; visa id: 7160
  %5653 = add i64 %5644, %5652		; visa id: 7164
  %5654 = inttoptr i64 %5653 to i16 addrspace(4)*		; visa id: 7165
  %5655 = addrspacecast i16 addrspace(4)* %5654 to i16 addrspace(1)*		; visa id: 7165
  %5656 = load i16, i16 addrspace(1)* %5655, align 2		; visa id: 7166
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7168
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7168
  %5657 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7168
  %5658 = insertelement <2 x i32> %5657, i32 %5609, i64 1		; visa id: 7169
  %5659 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7170
  store <2 x i32> %5658, <2 x i32>* %5659, align 4, !noalias !635		; visa id: 7170
  br label %._crit_edge328, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7172

._crit_edge328:                                   ; preds = %._crit_edge328.._crit_edge328_crit_edge, %5627
; BB552 :
  %5660 = phi i32 [ 0, %5627 ], [ %5669, %._crit_edge328.._crit_edge328_crit_edge ]
  %5661 = zext i32 %5660 to i64		; visa id: 7173
  %5662 = shl nuw nsw i64 %5661, 2		; visa id: 7174
  %5663 = add i64 %124, %5662		; visa id: 7175
  %5664 = inttoptr i64 %5663 to i32*		; visa id: 7176
  %5665 = load i32, i32* %5664, align 4, !noalias !635		; visa id: 7176
  %5666 = add i64 %119, %5662		; visa id: 7177
  %5667 = inttoptr i64 %5666 to i32*		; visa id: 7178
  store i32 %5665, i32* %5667, align 4, !alias.scope !635		; visa id: 7178
  %5668 = icmp eq i32 %5660, 0		; visa id: 7179
  br i1 %5668, label %._crit_edge328.._crit_edge328_crit_edge, label %5670, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7180

._crit_edge328.._crit_edge328_crit_edge:          ; preds = %._crit_edge328
; BB553 :
  %5669 = add nuw nsw i32 %5660, 1, !spirv.Decorations !631		; visa id: 7182
  br label %._crit_edge328, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7183

5670:                                             ; preds = %._crit_edge328
; BB554 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7185
  %5671 = load i64, i64* %120, align 8		; visa id: 7185
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7186
  %5672 = ashr i64 %5671, 32		; visa id: 7186
  %5673 = bitcast i64 %5672 to <2 x i32>		; visa id: 7187
  %5674 = extractelement <2 x i32> %5673, i32 0		; visa id: 7191
  %5675 = extractelement <2 x i32> %5673, i32 1		; visa id: 7191
  %5676 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5674, i32 %5675, i32 %44, i32 %45)
  %5677 = extractvalue { i32, i32 } %5676, 0		; visa id: 7191
  %5678 = extractvalue { i32, i32 } %5676, 1		; visa id: 7191
  %5679 = insertelement <2 x i32> undef, i32 %5677, i32 0		; visa id: 7198
  %5680 = insertelement <2 x i32> %5679, i32 %5678, i32 1		; visa id: 7199
  %5681 = bitcast <2 x i32> %5680 to i64		; visa id: 7200
  %5682 = bitcast i64 %5671 to <2 x i32>		; visa id: 7204
  %5683 = extractelement <2 x i32> %5682, i32 0		; visa id: 7206
  %5684 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5683, i32 1
  %5685 = bitcast <2 x i32> %5684 to i64		; visa id: 7206
  %5686 = shl i64 %5681, 1		; visa id: 7207
  %5687 = add i64 %.in400, %5686		; visa id: 7208
  %5688 = ashr exact i64 %5685, 31		; visa id: 7209
  %5689 = add i64 %5687, %5688		; visa id: 7210
  %5690 = inttoptr i64 %5689 to i16 addrspace(4)*		; visa id: 7211
  %5691 = addrspacecast i16 addrspace(4)* %5690 to i16 addrspace(1)*		; visa id: 7211
  %5692 = load i16, i16 addrspace(1)* %5691, align 2		; visa id: 7212
  %5693 = zext i16 %5656 to i32		; visa id: 7214
  %5694 = shl nuw i32 %5693, 16, !spirv.Decorations !639		; visa id: 7215
  %5695 = bitcast i32 %5694 to float
  %5696 = zext i16 %5692 to i32		; visa id: 7216
  %5697 = shl nuw i32 %5696, 16, !spirv.Decorations !639		; visa id: 7217
  %5698 = bitcast i32 %5697 to float
  %5699 = fmul reassoc nsz arcp contract float %5695, %5698, !spirv.Decorations !618
  %5700 = fadd reassoc nsz arcp contract float %5699, %.sroa.62.1, !spirv.Decorations !618		; visa id: 7218
  br label %._crit_edge.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7219

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %5670
; BB555 :
  %.sroa.62.2 = phi float [ %5700, %5670 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  %5701 = icmp slt i32 %223, %const_reg_dword
  %5702 = icmp slt i32 %5609, %const_reg_dword1		; visa id: 7220
  %5703 = and i1 %5701, %5702		; visa id: 7221
  br i1 %5703, label %5704, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7223

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
; BB:
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5704:                                             ; preds = %._crit_edge.15
; BB557 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7225
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7225
  %5705 = insertelement <2 x i32> undef, i32 %223, i64 0		; visa id: 7225
  %5706 = insertelement <2 x i32> %5705, i32 %113, i64 1		; visa id: 7226
  %5707 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7227
  store <2 x i32> %5706, <2 x i32>* %5707, align 4, !noalias !625		; visa id: 7227
  br label %._crit_edge329, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7229

._crit_edge329:                                   ; preds = %._crit_edge329.._crit_edge329_crit_edge, %5704
; BB558 :
  %5708 = phi i32 [ 0, %5704 ], [ %5717, %._crit_edge329.._crit_edge329_crit_edge ]
  %5709 = zext i32 %5708 to i64		; visa id: 7230
  %5710 = shl nuw nsw i64 %5709, 2		; visa id: 7231
  %5711 = add i64 %133, %5710		; visa id: 7232
  %5712 = inttoptr i64 %5711 to i32*		; visa id: 7233
  %5713 = load i32, i32* %5712, align 4, !noalias !625		; visa id: 7233
  %5714 = add i64 %128, %5710		; visa id: 7234
  %5715 = inttoptr i64 %5714 to i32*		; visa id: 7235
  store i32 %5713, i32* %5715, align 4, !alias.scope !625		; visa id: 7235
  %5716 = icmp eq i32 %5708, 0		; visa id: 7236
  br i1 %5716, label %._crit_edge329.._crit_edge329_crit_edge, label %5718, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7237

._crit_edge329.._crit_edge329_crit_edge:          ; preds = %._crit_edge329
; BB559 :
  %5717 = add nuw nsw i32 %5708, 1, !spirv.Decorations !631		; visa id: 7239
  br label %._crit_edge329, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7240

5718:                                             ; preds = %._crit_edge329
; BB560 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7242
  %5719 = load i64, i64* %129, align 8		; visa id: 7242
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7243
  %5720 = bitcast i64 %5719 to <2 x i32>		; visa id: 7243
  %5721 = extractelement <2 x i32> %5720, i32 0		; visa id: 7245
  %5722 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5721, i32 1
  %5723 = bitcast <2 x i32> %5722 to i64		; visa id: 7245
  %5724 = ashr exact i64 %5723, 32		; visa id: 7246
  %5725 = bitcast i64 %5724 to <2 x i32>		; visa id: 7247
  %5726 = extractelement <2 x i32> %5725, i32 0		; visa id: 7251
  %5727 = extractelement <2 x i32> %5725, i32 1		; visa id: 7251
  %5728 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5726, i32 %5727, i32 %41, i32 %42)
  %5729 = extractvalue { i32, i32 } %5728, 0		; visa id: 7251
  %5730 = extractvalue { i32, i32 } %5728, 1		; visa id: 7251
  %5731 = insertelement <2 x i32> undef, i32 %5729, i32 0		; visa id: 7258
  %5732 = insertelement <2 x i32> %5731, i32 %5730, i32 1		; visa id: 7259
  %5733 = bitcast <2 x i32> %5732 to i64		; visa id: 7260
  %5734 = shl i64 %5733, 1		; visa id: 7264
  %5735 = add i64 %.in401, %5734		; visa id: 7265
  %5736 = ashr i64 %5719, 31		; visa id: 7266
  %5737 = bitcast i64 %5736 to <2 x i32>		; visa id: 7267
  %5738 = extractelement <2 x i32> %5737, i32 0		; visa id: 7271
  %5739 = extractelement <2 x i32> %5737, i32 1		; visa id: 7271
  %5740 = and i32 %5738, -2		; visa id: 7271
  %5741 = insertelement <2 x i32> undef, i32 %5740, i32 0		; visa id: 7272
  %5742 = insertelement <2 x i32> %5741, i32 %5739, i32 1		; visa id: 7273
  %5743 = bitcast <2 x i32> %5742 to i64		; visa id: 7274
  %5744 = add i64 %5735, %5743		; visa id: 7278
  %5745 = inttoptr i64 %5744 to i16 addrspace(4)*		; visa id: 7279
  %5746 = addrspacecast i16 addrspace(4)* %5745 to i16 addrspace(1)*		; visa id: 7279
  %5747 = load i16, i16 addrspace(1)* %5746, align 2		; visa id: 7280
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7282
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7282
  %5748 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7282
  %5749 = insertelement <2 x i32> %5748, i32 %5609, i64 1		; visa id: 7283
  %5750 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7284
  store <2 x i32> %5749, <2 x i32>* %5750, align 4, !noalias !635		; visa id: 7284
  br label %._crit_edge330, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7286

._crit_edge330:                                   ; preds = %._crit_edge330.._crit_edge330_crit_edge, %5718
; BB561 :
  %5751 = phi i32 [ 0, %5718 ], [ %5760, %._crit_edge330.._crit_edge330_crit_edge ]
  %5752 = zext i32 %5751 to i64		; visa id: 7287
  %5753 = shl nuw nsw i64 %5752, 2		; visa id: 7288
  %5754 = add i64 %124, %5753		; visa id: 7289
  %5755 = inttoptr i64 %5754 to i32*		; visa id: 7290
  %5756 = load i32, i32* %5755, align 4, !noalias !635		; visa id: 7290
  %5757 = add i64 %119, %5753		; visa id: 7291
  %5758 = inttoptr i64 %5757 to i32*		; visa id: 7292
  store i32 %5756, i32* %5758, align 4, !alias.scope !635		; visa id: 7292
  %5759 = icmp eq i32 %5751, 0		; visa id: 7293
  br i1 %5759, label %._crit_edge330.._crit_edge330_crit_edge, label %5761, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7294

._crit_edge330.._crit_edge330_crit_edge:          ; preds = %._crit_edge330
; BB562 :
  %5760 = add nuw nsw i32 %5751, 1, !spirv.Decorations !631		; visa id: 7296
  br label %._crit_edge330, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7297

5761:                                             ; preds = %._crit_edge330
; BB563 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7299
  %5762 = load i64, i64* %120, align 8		; visa id: 7299
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7300
  %5763 = ashr i64 %5762, 32		; visa id: 7300
  %5764 = bitcast i64 %5763 to <2 x i32>		; visa id: 7301
  %5765 = extractelement <2 x i32> %5764, i32 0		; visa id: 7305
  %5766 = extractelement <2 x i32> %5764, i32 1		; visa id: 7305
  %5767 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5765, i32 %5766, i32 %44, i32 %45)
  %5768 = extractvalue { i32, i32 } %5767, 0		; visa id: 7305
  %5769 = extractvalue { i32, i32 } %5767, 1		; visa id: 7305
  %5770 = insertelement <2 x i32> undef, i32 %5768, i32 0		; visa id: 7312
  %5771 = insertelement <2 x i32> %5770, i32 %5769, i32 1		; visa id: 7313
  %5772 = bitcast <2 x i32> %5771 to i64		; visa id: 7314
  %5773 = bitcast i64 %5762 to <2 x i32>		; visa id: 7318
  %5774 = extractelement <2 x i32> %5773, i32 0		; visa id: 7320
  %5775 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5774, i32 1
  %5776 = bitcast <2 x i32> %5775 to i64		; visa id: 7320
  %5777 = shl i64 %5772, 1		; visa id: 7321
  %5778 = add i64 %.in400, %5777		; visa id: 7322
  %5779 = ashr exact i64 %5776, 31		; visa id: 7323
  %5780 = add i64 %5778, %5779		; visa id: 7324
  %5781 = inttoptr i64 %5780 to i16 addrspace(4)*		; visa id: 7325
  %5782 = addrspacecast i16 addrspace(4)* %5781 to i16 addrspace(1)*		; visa id: 7325
  %5783 = load i16, i16 addrspace(1)* %5782, align 2		; visa id: 7326
  %5784 = zext i16 %5747 to i32		; visa id: 7328
  %5785 = shl nuw i32 %5784, 16, !spirv.Decorations !639		; visa id: 7329
  %5786 = bitcast i32 %5785 to float
  %5787 = zext i16 %5783 to i32		; visa id: 7330
  %5788 = shl nuw i32 %5787, 16, !spirv.Decorations !639		; visa id: 7331
  %5789 = bitcast i32 %5788 to float
  %5790 = fmul reassoc nsz arcp contract float %5786, %5789, !spirv.Decorations !618
  %5791 = fadd reassoc nsz arcp contract float %5790, %.sroa.126.1, !spirv.Decorations !618		; visa id: 7332
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7333

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %5761
; BB564 :
  %.sroa.126.2 = phi float [ %5791, %5761 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  %5792 = icmp slt i32 %315, %const_reg_dword
  %5793 = icmp slt i32 %5609, %const_reg_dword1		; visa id: 7334
  %5794 = and i1 %5792, %5793		; visa id: 7335
  br i1 %5794, label %5795, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7337

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
; BB:
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5795:                                             ; preds = %._crit_edge.1.15
; BB566 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7339
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7339
  %5796 = insertelement <2 x i32> undef, i32 %315, i64 0		; visa id: 7339
  %5797 = insertelement <2 x i32> %5796, i32 %113, i64 1		; visa id: 7340
  %5798 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7341
  store <2 x i32> %5797, <2 x i32>* %5798, align 4, !noalias !625		; visa id: 7341
  br label %._crit_edge331, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7343

._crit_edge331:                                   ; preds = %._crit_edge331.._crit_edge331_crit_edge, %5795
; BB567 :
  %5799 = phi i32 [ 0, %5795 ], [ %5808, %._crit_edge331.._crit_edge331_crit_edge ]
  %5800 = zext i32 %5799 to i64		; visa id: 7344
  %5801 = shl nuw nsw i64 %5800, 2		; visa id: 7345
  %5802 = add i64 %133, %5801		; visa id: 7346
  %5803 = inttoptr i64 %5802 to i32*		; visa id: 7347
  %5804 = load i32, i32* %5803, align 4, !noalias !625		; visa id: 7347
  %5805 = add i64 %128, %5801		; visa id: 7348
  %5806 = inttoptr i64 %5805 to i32*		; visa id: 7349
  store i32 %5804, i32* %5806, align 4, !alias.scope !625		; visa id: 7349
  %5807 = icmp eq i32 %5799, 0		; visa id: 7350
  br i1 %5807, label %._crit_edge331.._crit_edge331_crit_edge, label %5809, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7351

._crit_edge331.._crit_edge331_crit_edge:          ; preds = %._crit_edge331
; BB568 :
  %5808 = add nuw nsw i32 %5799, 1, !spirv.Decorations !631		; visa id: 7353
  br label %._crit_edge331, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7354

5809:                                             ; preds = %._crit_edge331
; BB569 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7356
  %5810 = load i64, i64* %129, align 8		; visa id: 7356
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7357
  %5811 = bitcast i64 %5810 to <2 x i32>		; visa id: 7357
  %5812 = extractelement <2 x i32> %5811, i32 0		; visa id: 7359
  %5813 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5812, i32 1
  %5814 = bitcast <2 x i32> %5813 to i64		; visa id: 7359
  %5815 = ashr exact i64 %5814, 32		; visa id: 7360
  %5816 = bitcast i64 %5815 to <2 x i32>		; visa id: 7361
  %5817 = extractelement <2 x i32> %5816, i32 0		; visa id: 7365
  %5818 = extractelement <2 x i32> %5816, i32 1		; visa id: 7365
  %5819 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5817, i32 %5818, i32 %41, i32 %42)
  %5820 = extractvalue { i32, i32 } %5819, 0		; visa id: 7365
  %5821 = extractvalue { i32, i32 } %5819, 1		; visa id: 7365
  %5822 = insertelement <2 x i32> undef, i32 %5820, i32 0		; visa id: 7372
  %5823 = insertelement <2 x i32> %5822, i32 %5821, i32 1		; visa id: 7373
  %5824 = bitcast <2 x i32> %5823 to i64		; visa id: 7374
  %5825 = shl i64 %5824, 1		; visa id: 7378
  %5826 = add i64 %.in401, %5825		; visa id: 7379
  %5827 = ashr i64 %5810, 31		; visa id: 7380
  %5828 = bitcast i64 %5827 to <2 x i32>		; visa id: 7381
  %5829 = extractelement <2 x i32> %5828, i32 0		; visa id: 7385
  %5830 = extractelement <2 x i32> %5828, i32 1		; visa id: 7385
  %5831 = and i32 %5829, -2		; visa id: 7385
  %5832 = insertelement <2 x i32> undef, i32 %5831, i32 0		; visa id: 7386
  %5833 = insertelement <2 x i32> %5832, i32 %5830, i32 1		; visa id: 7387
  %5834 = bitcast <2 x i32> %5833 to i64		; visa id: 7388
  %5835 = add i64 %5826, %5834		; visa id: 7392
  %5836 = inttoptr i64 %5835 to i16 addrspace(4)*		; visa id: 7393
  %5837 = addrspacecast i16 addrspace(4)* %5836 to i16 addrspace(1)*		; visa id: 7393
  %5838 = load i16, i16 addrspace(1)* %5837, align 2		; visa id: 7394
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7396
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7396
  %5839 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7396
  %5840 = insertelement <2 x i32> %5839, i32 %5609, i64 1		; visa id: 7397
  %5841 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7398
  store <2 x i32> %5840, <2 x i32>* %5841, align 4, !noalias !635		; visa id: 7398
  br label %._crit_edge332, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7400

._crit_edge332:                                   ; preds = %._crit_edge332.._crit_edge332_crit_edge, %5809
; BB570 :
  %5842 = phi i32 [ 0, %5809 ], [ %5851, %._crit_edge332.._crit_edge332_crit_edge ]
  %5843 = zext i32 %5842 to i64		; visa id: 7401
  %5844 = shl nuw nsw i64 %5843, 2		; visa id: 7402
  %5845 = add i64 %124, %5844		; visa id: 7403
  %5846 = inttoptr i64 %5845 to i32*		; visa id: 7404
  %5847 = load i32, i32* %5846, align 4, !noalias !635		; visa id: 7404
  %5848 = add i64 %119, %5844		; visa id: 7405
  %5849 = inttoptr i64 %5848 to i32*		; visa id: 7406
  store i32 %5847, i32* %5849, align 4, !alias.scope !635		; visa id: 7406
  %5850 = icmp eq i32 %5842, 0		; visa id: 7407
  br i1 %5850, label %._crit_edge332.._crit_edge332_crit_edge, label %5852, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7408

._crit_edge332.._crit_edge332_crit_edge:          ; preds = %._crit_edge332
; BB571 :
  %5851 = add nuw nsw i32 %5842, 1, !spirv.Decorations !631		; visa id: 7410
  br label %._crit_edge332, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7411

5852:                                             ; preds = %._crit_edge332
; BB572 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7413
  %5853 = load i64, i64* %120, align 8		; visa id: 7413
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7414
  %5854 = ashr i64 %5853, 32		; visa id: 7414
  %5855 = bitcast i64 %5854 to <2 x i32>		; visa id: 7415
  %5856 = extractelement <2 x i32> %5855, i32 0		; visa id: 7419
  %5857 = extractelement <2 x i32> %5855, i32 1		; visa id: 7419
  %5858 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5856, i32 %5857, i32 %44, i32 %45)
  %5859 = extractvalue { i32, i32 } %5858, 0		; visa id: 7419
  %5860 = extractvalue { i32, i32 } %5858, 1		; visa id: 7419
  %5861 = insertelement <2 x i32> undef, i32 %5859, i32 0		; visa id: 7426
  %5862 = insertelement <2 x i32> %5861, i32 %5860, i32 1		; visa id: 7427
  %5863 = bitcast <2 x i32> %5862 to i64		; visa id: 7428
  %5864 = bitcast i64 %5853 to <2 x i32>		; visa id: 7432
  %5865 = extractelement <2 x i32> %5864, i32 0		; visa id: 7434
  %5866 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5865, i32 1
  %5867 = bitcast <2 x i32> %5866 to i64		; visa id: 7434
  %5868 = shl i64 %5863, 1		; visa id: 7435
  %5869 = add i64 %.in400, %5868		; visa id: 7436
  %5870 = ashr exact i64 %5867, 31		; visa id: 7437
  %5871 = add i64 %5869, %5870		; visa id: 7438
  %5872 = inttoptr i64 %5871 to i16 addrspace(4)*		; visa id: 7439
  %5873 = addrspacecast i16 addrspace(4)* %5872 to i16 addrspace(1)*		; visa id: 7439
  %5874 = load i16, i16 addrspace(1)* %5873, align 2		; visa id: 7440
  %5875 = zext i16 %5838 to i32		; visa id: 7442
  %5876 = shl nuw i32 %5875, 16, !spirv.Decorations !639		; visa id: 7443
  %5877 = bitcast i32 %5876 to float
  %5878 = zext i16 %5874 to i32		; visa id: 7444
  %5879 = shl nuw i32 %5878, 16, !spirv.Decorations !639		; visa id: 7445
  %5880 = bitcast i32 %5879 to float
  %5881 = fmul reassoc nsz arcp contract float %5877, %5880, !spirv.Decorations !618
  %5882 = fadd reassoc nsz arcp contract float %5881, %.sroa.190.1, !spirv.Decorations !618		; visa id: 7446
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7447

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %5852
; BB573 :
  %.sroa.190.2 = phi float [ %5882, %5852 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  %5883 = icmp slt i32 %407, %const_reg_dword
  %5884 = icmp slt i32 %5609, %const_reg_dword1		; visa id: 7448
  %5885 = and i1 %5883, %5884		; visa id: 7449
  br i1 %5885, label %5886, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7451

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
; BB:
  br label %.preheader.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5886:                                             ; preds = %._crit_edge.2.15
; BB575 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7453
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7453
  %5887 = insertelement <2 x i32> undef, i32 %407, i64 0		; visa id: 7453
  %5888 = insertelement <2 x i32> %5887, i32 %113, i64 1		; visa id: 7454
  %5889 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7455
  store <2 x i32> %5888, <2 x i32>* %5889, align 4, !noalias !625		; visa id: 7455
  br label %._crit_edge333, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7457

._crit_edge333:                                   ; preds = %._crit_edge333.._crit_edge333_crit_edge, %5886
; BB576 :
  %5890 = phi i32 [ 0, %5886 ], [ %5899, %._crit_edge333.._crit_edge333_crit_edge ]
  %5891 = zext i32 %5890 to i64		; visa id: 7458
  %5892 = shl nuw nsw i64 %5891, 2		; visa id: 7459
  %5893 = add i64 %133, %5892		; visa id: 7460
  %5894 = inttoptr i64 %5893 to i32*		; visa id: 7461
  %5895 = load i32, i32* %5894, align 4, !noalias !625		; visa id: 7461
  %5896 = add i64 %128, %5892		; visa id: 7462
  %5897 = inttoptr i64 %5896 to i32*		; visa id: 7463
  store i32 %5895, i32* %5897, align 4, !alias.scope !625		; visa id: 7463
  %5898 = icmp eq i32 %5890, 0		; visa id: 7464
  br i1 %5898, label %._crit_edge333.._crit_edge333_crit_edge, label %5900, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7465

._crit_edge333.._crit_edge333_crit_edge:          ; preds = %._crit_edge333
; BB577 :
  %5899 = add nuw nsw i32 %5890, 1, !spirv.Decorations !631		; visa id: 7467
  br label %._crit_edge333, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7468

5900:                                             ; preds = %._crit_edge333
; BB578 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7470
  %5901 = load i64, i64* %129, align 8		; visa id: 7470
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7471
  %5902 = bitcast i64 %5901 to <2 x i32>		; visa id: 7471
  %5903 = extractelement <2 x i32> %5902, i32 0		; visa id: 7473
  %5904 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5903, i32 1
  %5905 = bitcast <2 x i32> %5904 to i64		; visa id: 7473
  %5906 = ashr exact i64 %5905, 32		; visa id: 7474
  %5907 = bitcast i64 %5906 to <2 x i32>		; visa id: 7475
  %5908 = extractelement <2 x i32> %5907, i32 0		; visa id: 7479
  %5909 = extractelement <2 x i32> %5907, i32 1		; visa id: 7479
  %5910 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5908, i32 %5909, i32 %41, i32 %42)
  %5911 = extractvalue { i32, i32 } %5910, 0		; visa id: 7479
  %5912 = extractvalue { i32, i32 } %5910, 1		; visa id: 7479
  %5913 = insertelement <2 x i32> undef, i32 %5911, i32 0		; visa id: 7486
  %5914 = insertelement <2 x i32> %5913, i32 %5912, i32 1		; visa id: 7487
  %5915 = bitcast <2 x i32> %5914 to i64		; visa id: 7488
  %5916 = shl i64 %5915, 1		; visa id: 7492
  %5917 = add i64 %.in401, %5916		; visa id: 7493
  %5918 = ashr i64 %5901, 31		; visa id: 7494
  %5919 = bitcast i64 %5918 to <2 x i32>		; visa id: 7495
  %5920 = extractelement <2 x i32> %5919, i32 0		; visa id: 7499
  %5921 = extractelement <2 x i32> %5919, i32 1		; visa id: 7499
  %5922 = and i32 %5920, -2		; visa id: 7499
  %5923 = insertelement <2 x i32> undef, i32 %5922, i32 0		; visa id: 7500
  %5924 = insertelement <2 x i32> %5923, i32 %5921, i32 1		; visa id: 7501
  %5925 = bitcast <2 x i32> %5924 to i64		; visa id: 7502
  %5926 = add i64 %5917, %5925		; visa id: 7506
  %5927 = inttoptr i64 %5926 to i16 addrspace(4)*		; visa id: 7507
  %5928 = addrspacecast i16 addrspace(4)* %5927 to i16 addrspace(1)*		; visa id: 7507
  %5929 = load i16, i16 addrspace(1)* %5928, align 2		; visa id: 7508
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7510
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7510
  %5930 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7510
  %5931 = insertelement <2 x i32> %5930, i32 %5609, i64 1		; visa id: 7511
  %5932 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7512
  store <2 x i32> %5931, <2 x i32>* %5932, align 4, !noalias !635		; visa id: 7512
  br label %._crit_edge334, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7514

._crit_edge334:                                   ; preds = %._crit_edge334.._crit_edge334_crit_edge, %5900
; BB579 :
  %5933 = phi i32 [ 0, %5900 ], [ %5942, %._crit_edge334.._crit_edge334_crit_edge ]
  %5934 = zext i32 %5933 to i64		; visa id: 7515
  %5935 = shl nuw nsw i64 %5934, 2		; visa id: 7516
  %5936 = add i64 %124, %5935		; visa id: 7517
  %5937 = inttoptr i64 %5936 to i32*		; visa id: 7518
  %5938 = load i32, i32* %5937, align 4, !noalias !635		; visa id: 7518
  %5939 = add i64 %119, %5935		; visa id: 7519
  %5940 = inttoptr i64 %5939 to i32*		; visa id: 7520
  store i32 %5938, i32* %5940, align 4, !alias.scope !635		; visa id: 7520
  %5941 = icmp eq i32 %5933, 0		; visa id: 7521
  br i1 %5941, label %._crit_edge334.._crit_edge334_crit_edge, label %5943, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7522

._crit_edge334.._crit_edge334_crit_edge:          ; preds = %._crit_edge334
; BB580 :
  %5942 = add nuw nsw i32 %5933, 1, !spirv.Decorations !631		; visa id: 7524
  br label %._crit_edge334, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7525

5943:                                             ; preds = %._crit_edge334
; BB581 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7527
  %5944 = load i64, i64* %120, align 8		; visa id: 7527
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7528
  %5945 = ashr i64 %5944, 32		; visa id: 7528
  %5946 = bitcast i64 %5945 to <2 x i32>		; visa id: 7529
  %5947 = extractelement <2 x i32> %5946, i32 0		; visa id: 7533
  %5948 = extractelement <2 x i32> %5946, i32 1		; visa id: 7533
  %5949 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5947, i32 %5948, i32 %44, i32 %45)
  %5950 = extractvalue { i32, i32 } %5949, 0		; visa id: 7533
  %5951 = extractvalue { i32, i32 } %5949, 1		; visa id: 7533
  %5952 = insertelement <2 x i32> undef, i32 %5950, i32 0		; visa id: 7540
  %5953 = insertelement <2 x i32> %5952, i32 %5951, i32 1		; visa id: 7541
  %5954 = bitcast <2 x i32> %5953 to i64		; visa id: 7542
  %5955 = bitcast i64 %5944 to <2 x i32>		; visa id: 7546
  %5956 = extractelement <2 x i32> %5955, i32 0		; visa id: 7548
  %5957 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5956, i32 1
  %5958 = bitcast <2 x i32> %5957 to i64		; visa id: 7548
  %5959 = shl i64 %5954, 1		; visa id: 7549
  %5960 = add i64 %.in400, %5959		; visa id: 7550
  %5961 = ashr exact i64 %5958, 31		; visa id: 7551
  %5962 = add i64 %5960, %5961		; visa id: 7552
  %5963 = inttoptr i64 %5962 to i16 addrspace(4)*		; visa id: 7553
  %5964 = addrspacecast i16 addrspace(4)* %5963 to i16 addrspace(1)*		; visa id: 7553
  %5965 = load i16, i16 addrspace(1)* %5964, align 2		; visa id: 7554
  %5966 = zext i16 %5929 to i32		; visa id: 7556
  %5967 = shl nuw i32 %5966, 16, !spirv.Decorations !639		; visa id: 7557
  %5968 = bitcast i32 %5967 to float
  %5969 = zext i16 %5965 to i32		; visa id: 7558
  %5970 = shl nuw i32 %5969, 16, !spirv.Decorations !639		; visa id: 7559
  %5971 = bitcast i32 %5970 to float
  %5972 = fmul reassoc nsz arcp contract float %5968, %5971, !spirv.Decorations !618
  %5973 = fadd reassoc nsz arcp contract float %5972, %.sroa.254.1, !spirv.Decorations !618		; visa id: 7560
  br label %.preheader.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7561

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %5943
; BB582 :
  %.sroa.254.2 = phi float [ %5973, %5943 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %5974 = add nuw nsw i32 %113, 1, !spirv.Decorations !631		; visa id: 7562
  %5975 = icmp slt i32 %5974, %const_reg_dword2		; visa id: 7563
  br i1 %5975, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !llvm.loop !640, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7564

.preheader.15..preheader.preheader_crit_edge:     ; preds = %.preheader.15
; BB:
  br label %.preheader.preheader, !stats.blockFrequency.digits !641, !stats.blockFrequency.scale !615

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
  br label %.preheader1.preheader, !stats.blockFrequency.digits !622, !stats.blockFrequency.scale !615

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
; BB585 :
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
  %5976 = add nuw nsw i32 %56, %109		; visa id: 7566
  %5977 = zext i32 %5976 to i64		; visa id: 7567
  %5978 = add i64 %111, %5977		; visa id: 7568
  %5979 = inttoptr i64 %5978 to i8*		; visa id: 7569
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7569
  %5980 = add nuw nsw i32 %56, %102		; visa id: 7569
  %5981 = zext i32 %5980 to i64		; visa id: 7570
  %5982 = add i64 %111, %5981		; visa id: 7571
  %5983 = inttoptr i64 %5982 to i8*		; visa id: 7572
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7572
  %5984 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 7572
  %5985 = insertelement <2 x i32> %5984, i32 %69, i64 1		; visa id: 7573
  %5986 = inttoptr i64 %5982 to <2 x i32>*		; visa id: 7576
  store <2 x i32> %5985, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7576
  br label %._crit_edge335, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7578

._crit_edge335:                                   ; preds = %._crit_edge335.._crit_edge335_crit_edge, %.preheader1.preheader
; BB586 :
  %5987 = phi i32 [ 0, %.preheader1.preheader ], [ %5996, %._crit_edge335.._crit_edge335_crit_edge ]
  %5988 = zext i32 %5987 to i64		; visa id: 7579
  %5989 = shl nuw nsw i64 %5988, 2		; visa id: 7580
  %5990 = add i64 %5982, %5989		; visa id: 7581
  %5991 = inttoptr i64 %5990 to i32*		; visa id: 7582
  %5992 = load i32, i32* %5991, align 4, !noalias !642		; visa id: 7582
  %5993 = add i64 %5978, %5989		; visa id: 7583
  %5994 = inttoptr i64 %5993 to i32*		; visa id: 7584
  store i32 %5992, i32* %5994, align 4, !alias.scope !642		; visa id: 7584
  %5995 = icmp eq i32 %5987, 0		; visa id: 7585
  br i1 %5995, label %._crit_edge335.._crit_edge335_crit_edge, label %5997, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 7586

._crit_edge335.._crit_edge335_crit_edge:          ; preds = %._crit_edge335
; BB587 :
  %5996 = add nuw nsw i32 %5987, 1, !spirv.Decorations !631		; visa id: 7588
  br label %._crit_edge335, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 7589

5997:                                             ; preds = %._crit_edge335
; BB588 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7591
  %5998 = inttoptr i64 %5978 to i64*		; visa id: 7591
  %5999 = load i64, i64* %5998, align 8		; visa id: 7591
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7592
  %6000 = icmp slt i32 %65, %const_reg_dword
  %6001 = icmp slt i32 %69, %const_reg_dword1		; visa id: 7592
  %6002 = and i1 %6000, %6001		; visa id: 7593
  br i1 %6002, label %6003, label %.._crit_edge70_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7595

.._crit_edge70_crit_edge:                         ; preds = %5997
; BB:
  br label %._crit_edge70, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6003:                                             ; preds = %5997
; BB590 :
  %6004 = bitcast i64 %5999 to <2 x i32>		; visa id: 7597
  %6005 = extractelement <2 x i32> %6004, i32 0		; visa id: 7599
  %6006 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6005, i32 1
  %6007 = bitcast <2 x i32> %6006 to i64		; visa id: 7599
  %6008 = ashr exact i64 %6007, 32		; visa id: 7600
  %6009 = bitcast i64 %6008 to <2 x i32>		; visa id: 7601
  %6010 = extractelement <2 x i32> %6009, i32 0		; visa id: 7605
  %6011 = extractelement <2 x i32> %6009, i32 1		; visa id: 7605
  %6012 = ashr i64 %5999, 32		; visa id: 7605
  %6013 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6010, i32 %6011, i32 %50, i32 %51)
  %6014 = extractvalue { i32, i32 } %6013, 0		; visa id: 7606
  %6015 = extractvalue { i32, i32 } %6013, 1		; visa id: 7606
  %6016 = insertelement <2 x i32> undef, i32 %6014, i32 0		; visa id: 7613
  %6017 = insertelement <2 x i32> %6016, i32 %6015, i32 1		; visa id: 7614
  %6018 = bitcast <2 x i32> %6017 to i64		; visa id: 7615
  %6019 = add nsw i64 %6018, %6012, !spirv.Decorations !649		; visa id: 7619
  %6020 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !618		; visa id: 7620
  br i1 %86, label %6026, label %6021, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 7621

6021:                                             ; preds = %6003
; BB591 :
  %6022 = shl i64 %6019, 2		; visa id: 7623
  %6023 = add i64 %.in, %6022		; visa id: 7624
  %6024 = inttoptr i64 %6023 to float addrspace(4)*		; visa id: 7625
  %6025 = addrspacecast float addrspace(4)* %6024 to float addrspace(1)*		; visa id: 7625
  store float %6020, float addrspace(1)* %6025, align 4		; visa id: 7626
  br label %._crit_edge70, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 7627

6026:                                             ; preds = %6003
; BB592 :
  %6027 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6010, i32 %6011, i32 %47, i32 %48)
  %6028 = extractvalue { i32, i32 } %6027, 0		; visa id: 7629
  %6029 = extractvalue { i32, i32 } %6027, 1		; visa id: 7629
  %6030 = insertelement <2 x i32> undef, i32 %6028, i32 0		; visa id: 7636
  %6031 = insertelement <2 x i32> %6030, i32 %6029, i32 1		; visa id: 7637
  %6032 = bitcast <2 x i32> %6031 to i64		; visa id: 7638
  %6033 = shl i64 %6032, 2		; visa id: 7642
  %6034 = add i64 %.in399, %6033		; visa id: 7643
  %6035 = shl nsw i64 %6012, 2		; visa id: 7644
  %6036 = add i64 %6034, %6035		; visa id: 7645
  %6037 = inttoptr i64 %6036 to float addrspace(4)*		; visa id: 7646
  %6038 = addrspacecast float addrspace(4)* %6037 to float addrspace(1)*		; visa id: 7646
  %6039 = load float, float addrspace(1)* %6038, align 4		; visa id: 7647
  %6040 = fmul reassoc nsz arcp contract float %6039, %4, !spirv.Decorations !618		; visa id: 7648
  %6041 = fadd reassoc nsz arcp contract float %6020, %6040, !spirv.Decorations !618		; visa id: 7649
  %6042 = shl i64 %6019, 2		; visa id: 7650
  %6043 = add i64 %.in, %6042		; visa id: 7651
  %6044 = inttoptr i64 %6043 to float addrspace(4)*		; visa id: 7652
  %6045 = addrspacecast float addrspace(4)* %6044 to float addrspace(1)*		; visa id: 7652
  store float %6041, float addrspace(1)* %6045, align 4		; visa id: 7653
  br label %._crit_edge70, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 7654

._crit_edge70:                                    ; preds = %.._crit_edge70_crit_edge, %6021, %6026
; BB593 :
  %6046 = add i32 %65, 1		; visa id: 7655
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7656
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7656
  %6047 = insertelement <2 x i32> undef, i32 %6046, i64 0		; visa id: 7656
  %6048 = insertelement <2 x i32> %6047, i32 %69, i64 1		; visa id: 7657
  store <2 x i32> %6048, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7660
  br label %._crit_edge336, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7662

._crit_edge336:                                   ; preds = %._crit_edge336.._crit_edge336_crit_edge, %._crit_edge70
; BB594 :
  %6049 = phi i32 [ 0, %._crit_edge70 ], [ %6058, %._crit_edge336.._crit_edge336_crit_edge ]
  %6050 = zext i32 %6049 to i64		; visa id: 7663
  %6051 = shl nuw nsw i64 %6050, 2		; visa id: 7664
  %6052 = add i64 %5982, %6051		; visa id: 7665
  %6053 = inttoptr i64 %6052 to i32*		; visa id: 7666
  %6054 = load i32, i32* %6053, align 4, !noalias !642		; visa id: 7666
  %6055 = add i64 %5978, %6051		; visa id: 7667
  %6056 = inttoptr i64 %6055 to i32*		; visa id: 7668
  store i32 %6054, i32* %6056, align 4, !alias.scope !642		; visa id: 7668
  %6057 = icmp eq i32 %6049, 0		; visa id: 7669
  br i1 %6057, label %._crit_edge336.._crit_edge336_crit_edge, label %6059, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 7670

._crit_edge336.._crit_edge336_crit_edge:          ; preds = %._crit_edge336
; BB595 :
  %6058 = add nuw nsw i32 %6049, 1, !spirv.Decorations !631		; visa id: 7672
  br label %._crit_edge336, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 7673

6059:                                             ; preds = %._crit_edge336
; BB596 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7675
  %6060 = load i64, i64* %5998, align 8		; visa id: 7675
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7676
  %6061 = icmp slt i32 %6046, %const_reg_dword
  %6062 = icmp slt i32 %69, %const_reg_dword1		; visa id: 7676
  %6063 = and i1 %6061, %6062		; visa id: 7677
  br i1 %6063, label %6064, label %.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7679

.._crit_edge70.1_crit_edge:                       ; preds = %6059
; BB:
  br label %._crit_edge70.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6064:                                             ; preds = %6059
; BB598 :
  %6065 = bitcast i64 %6060 to <2 x i32>		; visa id: 7681
  %6066 = extractelement <2 x i32> %6065, i32 0		; visa id: 7683
  %6067 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6066, i32 1
  %6068 = bitcast <2 x i32> %6067 to i64		; visa id: 7683
  %6069 = ashr exact i64 %6068, 32		; visa id: 7684
  %6070 = bitcast i64 %6069 to <2 x i32>		; visa id: 7685
  %6071 = extractelement <2 x i32> %6070, i32 0		; visa id: 7689
  %6072 = extractelement <2 x i32> %6070, i32 1		; visa id: 7689
  %6073 = ashr i64 %6060, 32		; visa id: 7689
  %6074 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6071, i32 %6072, i32 %50, i32 %51)
  %6075 = extractvalue { i32, i32 } %6074, 0		; visa id: 7690
  %6076 = extractvalue { i32, i32 } %6074, 1		; visa id: 7690
  %6077 = insertelement <2 x i32> undef, i32 %6075, i32 0		; visa id: 7697
  %6078 = insertelement <2 x i32> %6077, i32 %6076, i32 1		; visa id: 7698
  %6079 = bitcast <2 x i32> %6078 to i64		; visa id: 7699
  %6080 = add nsw i64 %6079, %6073, !spirv.Decorations !649		; visa id: 7703
  %6081 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !618		; visa id: 7704
  br i1 %86, label %6087, label %6082, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 7705

6082:                                             ; preds = %6064
; BB599 :
  %6083 = shl i64 %6080, 2		; visa id: 7707
  %6084 = add i64 %.in, %6083		; visa id: 7708
  %6085 = inttoptr i64 %6084 to float addrspace(4)*		; visa id: 7709
  %6086 = addrspacecast float addrspace(4)* %6085 to float addrspace(1)*		; visa id: 7709
  store float %6081, float addrspace(1)* %6086, align 4		; visa id: 7710
  br label %._crit_edge70.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 7711

6087:                                             ; preds = %6064
; BB600 :
  %6088 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6071, i32 %6072, i32 %47, i32 %48)
  %6089 = extractvalue { i32, i32 } %6088, 0		; visa id: 7713
  %6090 = extractvalue { i32, i32 } %6088, 1		; visa id: 7713
  %6091 = insertelement <2 x i32> undef, i32 %6089, i32 0		; visa id: 7720
  %6092 = insertelement <2 x i32> %6091, i32 %6090, i32 1		; visa id: 7721
  %6093 = bitcast <2 x i32> %6092 to i64		; visa id: 7722
  %6094 = shl i64 %6093, 2		; visa id: 7726
  %6095 = add i64 %.in399, %6094		; visa id: 7727
  %6096 = shl nsw i64 %6073, 2		; visa id: 7728
  %6097 = add i64 %6095, %6096		; visa id: 7729
  %6098 = inttoptr i64 %6097 to float addrspace(4)*		; visa id: 7730
  %6099 = addrspacecast float addrspace(4)* %6098 to float addrspace(1)*		; visa id: 7730
  %6100 = load float, float addrspace(1)* %6099, align 4		; visa id: 7731
  %6101 = fmul reassoc nsz arcp contract float %6100, %4, !spirv.Decorations !618		; visa id: 7732
  %6102 = fadd reassoc nsz arcp contract float %6081, %6101, !spirv.Decorations !618		; visa id: 7733
  %6103 = shl i64 %6080, 2		; visa id: 7734
  %6104 = add i64 %.in, %6103		; visa id: 7735
  %6105 = inttoptr i64 %6104 to float addrspace(4)*		; visa id: 7736
  %6106 = addrspacecast float addrspace(4)* %6105 to float addrspace(1)*		; visa id: 7736
  store float %6102, float addrspace(1)* %6106, align 4		; visa id: 7737
  br label %._crit_edge70.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 7738

._crit_edge70.1:                                  ; preds = %.._crit_edge70.1_crit_edge, %6087, %6082
; BB601 :
  %6107 = add i32 %65, 2		; visa id: 7739
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7740
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7740
  %6108 = insertelement <2 x i32> undef, i32 %6107, i64 0		; visa id: 7740
  %6109 = insertelement <2 x i32> %6108, i32 %69, i64 1		; visa id: 7741
  store <2 x i32> %6109, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7744
  br label %._crit_edge337, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7746

._crit_edge337:                                   ; preds = %._crit_edge337.._crit_edge337_crit_edge, %._crit_edge70.1
; BB602 :
  %6110 = phi i32 [ 0, %._crit_edge70.1 ], [ %6119, %._crit_edge337.._crit_edge337_crit_edge ]
  %6111 = zext i32 %6110 to i64		; visa id: 7747
  %6112 = shl nuw nsw i64 %6111, 2		; visa id: 7748
  %6113 = add i64 %5982, %6112		; visa id: 7749
  %6114 = inttoptr i64 %6113 to i32*		; visa id: 7750
  %6115 = load i32, i32* %6114, align 4, !noalias !642		; visa id: 7750
  %6116 = add i64 %5978, %6112		; visa id: 7751
  %6117 = inttoptr i64 %6116 to i32*		; visa id: 7752
  store i32 %6115, i32* %6117, align 4, !alias.scope !642		; visa id: 7752
  %6118 = icmp eq i32 %6110, 0		; visa id: 7753
  br i1 %6118, label %._crit_edge337.._crit_edge337_crit_edge, label %6120, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 7754

._crit_edge337.._crit_edge337_crit_edge:          ; preds = %._crit_edge337
; BB603 :
  %6119 = add nuw nsw i32 %6110, 1, !spirv.Decorations !631		; visa id: 7756
  br label %._crit_edge337, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 7757

6120:                                             ; preds = %._crit_edge337
; BB604 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7759
  %6121 = load i64, i64* %5998, align 8		; visa id: 7759
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7760
  %6122 = icmp slt i32 %6107, %const_reg_dword
  %6123 = icmp slt i32 %69, %const_reg_dword1		; visa id: 7760
  %6124 = and i1 %6122, %6123		; visa id: 7761
  br i1 %6124, label %6125, label %.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7763

.._crit_edge70.2_crit_edge:                       ; preds = %6120
; BB:
  br label %._crit_edge70.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6125:                                             ; preds = %6120
; BB606 :
  %6126 = bitcast i64 %6121 to <2 x i32>		; visa id: 7765
  %6127 = extractelement <2 x i32> %6126, i32 0		; visa id: 7767
  %6128 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6127, i32 1
  %6129 = bitcast <2 x i32> %6128 to i64		; visa id: 7767
  %6130 = ashr exact i64 %6129, 32		; visa id: 7768
  %6131 = bitcast i64 %6130 to <2 x i32>		; visa id: 7769
  %6132 = extractelement <2 x i32> %6131, i32 0		; visa id: 7773
  %6133 = extractelement <2 x i32> %6131, i32 1		; visa id: 7773
  %6134 = ashr i64 %6121, 32		; visa id: 7773
  %6135 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6132, i32 %6133, i32 %50, i32 %51)
  %6136 = extractvalue { i32, i32 } %6135, 0		; visa id: 7774
  %6137 = extractvalue { i32, i32 } %6135, 1		; visa id: 7774
  %6138 = insertelement <2 x i32> undef, i32 %6136, i32 0		; visa id: 7781
  %6139 = insertelement <2 x i32> %6138, i32 %6137, i32 1		; visa id: 7782
  %6140 = bitcast <2 x i32> %6139 to i64		; visa id: 7783
  %6141 = add nsw i64 %6140, %6134, !spirv.Decorations !649		; visa id: 7787
  %6142 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !618		; visa id: 7788
  br i1 %86, label %6148, label %6143, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 7789

6143:                                             ; preds = %6125
; BB607 :
  %6144 = shl i64 %6141, 2		; visa id: 7791
  %6145 = add i64 %.in, %6144		; visa id: 7792
  %6146 = inttoptr i64 %6145 to float addrspace(4)*		; visa id: 7793
  %6147 = addrspacecast float addrspace(4)* %6146 to float addrspace(1)*		; visa id: 7793
  store float %6142, float addrspace(1)* %6147, align 4		; visa id: 7794
  br label %._crit_edge70.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 7795

6148:                                             ; preds = %6125
; BB608 :
  %6149 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6132, i32 %6133, i32 %47, i32 %48)
  %6150 = extractvalue { i32, i32 } %6149, 0		; visa id: 7797
  %6151 = extractvalue { i32, i32 } %6149, 1		; visa id: 7797
  %6152 = insertelement <2 x i32> undef, i32 %6150, i32 0		; visa id: 7804
  %6153 = insertelement <2 x i32> %6152, i32 %6151, i32 1		; visa id: 7805
  %6154 = bitcast <2 x i32> %6153 to i64		; visa id: 7806
  %6155 = shl i64 %6154, 2		; visa id: 7810
  %6156 = add i64 %.in399, %6155		; visa id: 7811
  %6157 = shl nsw i64 %6134, 2		; visa id: 7812
  %6158 = add i64 %6156, %6157		; visa id: 7813
  %6159 = inttoptr i64 %6158 to float addrspace(4)*		; visa id: 7814
  %6160 = addrspacecast float addrspace(4)* %6159 to float addrspace(1)*		; visa id: 7814
  %6161 = load float, float addrspace(1)* %6160, align 4		; visa id: 7815
  %6162 = fmul reassoc nsz arcp contract float %6161, %4, !spirv.Decorations !618		; visa id: 7816
  %6163 = fadd reassoc nsz arcp contract float %6142, %6162, !spirv.Decorations !618		; visa id: 7817
  %6164 = shl i64 %6141, 2		; visa id: 7818
  %6165 = add i64 %.in, %6164		; visa id: 7819
  %6166 = inttoptr i64 %6165 to float addrspace(4)*		; visa id: 7820
  %6167 = addrspacecast float addrspace(4)* %6166 to float addrspace(1)*		; visa id: 7820
  store float %6163, float addrspace(1)* %6167, align 4		; visa id: 7821
  br label %._crit_edge70.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 7822

._crit_edge70.2:                                  ; preds = %.._crit_edge70.2_crit_edge, %6148, %6143
; BB609 :
  %6168 = add i32 %65, 3		; visa id: 7823
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7824
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7824
  %6169 = insertelement <2 x i32> undef, i32 %6168, i64 0		; visa id: 7824
  %6170 = insertelement <2 x i32> %6169, i32 %69, i64 1		; visa id: 7825
  store <2 x i32> %6170, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7828
  br label %._crit_edge338, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7830

._crit_edge338:                                   ; preds = %._crit_edge338.._crit_edge338_crit_edge, %._crit_edge70.2
; BB610 :
  %6171 = phi i32 [ 0, %._crit_edge70.2 ], [ %6180, %._crit_edge338.._crit_edge338_crit_edge ]
  %6172 = zext i32 %6171 to i64		; visa id: 7831
  %6173 = shl nuw nsw i64 %6172, 2		; visa id: 7832
  %6174 = add i64 %5982, %6173		; visa id: 7833
  %6175 = inttoptr i64 %6174 to i32*		; visa id: 7834
  %6176 = load i32, i32* %6175, align 4, !noalias !642		; visa id: 7834
  %6177 = add i64 %5978, %6173		; visa id: 7835
  %6178 = inttoptr i64 %6177 to i32*		; visa id: 7836
  store i32 %6176, i32* %6178, align 4, !alias.scope !642		; visa id: 7836
  %6179 = icmp eq i32 %6171, 0		; visa id: 7837
  br i1 %6179, label %._crit_edge338.._crit_edge338_crit_edge, label %6181, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 7838

._crit_edge338.._crit_edge338_crit_edge:          ; preds = %._crit_edge338
; BB611 :
  %6180 = add nuw nsw i32 %6171, 1, !spirv.Decorations !631		; visa id: 7840
  br label %._crit_edge338, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 7841

6181:                                             ; preds = %._crit_edge338
; BB612 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7843
  %6182 = load i64, i64* %5998, align 8		; visa id: 7843
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7844
  %6183 = icmp slt i32 %6168, %const_reg_dword
  %6184 = icmp slt i32 %69, %const_reg_dword1		; visa id: 7844
  %6185 = and i1 %6183, %6184		; visa id: 7845
  br i1 %6185, label %6186, label %..preheader1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7847

..preheader1_crit_edge:                           ; preds = %6181
; BB:
  br label %.preheader1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6186:                                             ; preds = %6181
; BB614 :
  %6187 = bitcast i64 %6182 to <2 x i32>		; visa id: 7849
  %6188 = extractelement <2 x i32> %6187, i32 0		; visa id: 7851
  %6189 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6188, i32 1
  %6190 = bitcast <2 x i32> %6189 to i64		; visa id: 7851
  %6191 = ashr exact i64 %6190, 32		; visa id: 7852
  %6192 = bitcast i64 %6191 to <2 x i32>		; visa id: 7853
  %6193 = extractelement <2 x i32> %6192, i32 0		; visa id: 7857
  %6194 = extractelement <2 x i32> %6192, i32 1		; visa id: 7857
  %6195 = ashr i64 %6182, 32		; visa id: 7857
  %6196 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6193, i32 %6194, i32 %50, i32 %51)
  %6197 = extractvalue { i32, i32 } %6196, 0		; visa id: 7858
  %6198 = extractvalue { i32, i32 } %6196, 1		; visa id: 7858
  %6199 = insertelement <2 x i32> undef, i32 %6197, i32 0		; visa id: 7865
  %6200 = insertelement <2 x i32> %6199, i32 %6198, i32 1		; visa id: 7866
  %6201 = bitcast <2 x i32> %6200 to i64		; visa id: 7867
  %6202 = add nsw i64 %6201, %6195, !spirv.Decorations !649		; visa id: 7871
  %6203 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !618		; visa id: 7872
  br i1 %86, label %6209, label %6204, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 7873

6204:                                             ; preds = %6186
; BB615 :
  %6205 = shl i64 %6202, 2		; visa id: 7875
  %6206 = add i64 %.in, %6205		; visa id: 7876
  %6207 = inttoptr i64 %6206 to float addrspace(4)*		; visa id: 7877
  %6208 = addrspacecast float addrspace(4)* %6207 to float addrspace(1)*		; visa id: 7877
  store float %6203, float addrspace(1)* %6208, align 4		; visa id: 7878
  br label %.preheader1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 7879

6209:                                             ; preds = %6186
; BB616 :
  %6210 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6193, i32 %6194, i32 %47, i32 %48)
  %6211 = extractvalue { i32, i32 } %6210, 0		; visa id: 7881
  %6212 = extractvalue { i32, i32 } %6210, 1		; visa id: 7881
  %6213 = insertelement <2 x i32> undef, i32 %6211, i32 0		; visa id: 7888
  %6214 = insertelement <2 x i32> %6213, i32 %6212, i32 1		; visa id: 7889
  %6215 = bitcast <2 x i32> %6214 to i64		; visa id: 7890
  %6216 = shl i64 %6215, 2		; visa id: 7894
  %6217 = add i64 %.in399, %6216		; visa id: 7895
  %6218 = shl nsw i64 %6195, 2		; visa id: 7896
  %6219 = add i64 %6217, %6218		; visa id: 7897
  %6220 = inttoptr i64 %6219 to float addrspace(4)*		; visa id: 7898
  %6221 = addrspacecast float addrspace(4)* %6220 to float addrspace(1)*		; visa id: 7898
  %6222 = load float, float addrspace(1)* %6221, align 4		; visa id: 7899
  %6223 = fmul reassoc nsz arcp contract float %6222, %4, !spirv.Decorations !618		; visa id: 7900
  %6224 = fadd reassoc nsz arcp contract float %6203, %6223, !spirv.Decorations !618		; visa id: 7901
  %6225 = shl i64 %6202, 2		; visa id: 7902
  %6226 = add i64 %.in, %6225		; visa id: 7903
  %6227 = inttoptr i64 %6226 to float addrspace(4)*		; visa id: 7904
  %6228 = addrspacecast float addrspace(4)* %6227 to float addrspace(1)*		; visa id: 7904
  store float %6224, float addrspace(1)* %6228, align 4		; visa id: 7905
  br label %.preheader1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 7906

.preheader1:                                      ; preds = %..preheader1_crit_edge, %6209, %6204
; BB617 :
  %6229 = add i32 %69, 1		; visa id: 7907
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7908
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7908
  %6230 = insertelement <2 x i32> %5984, i32 %6229, i64 1		; visa id: 7908
  store <2 x i32> %6230, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7911
  br label %._crit_edge339, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7913

._crit_edge339:                                   ; preds = %._crit_edge339.._crit_edge339_crit_edge, %.preheader1
; BB618 :
  %6231 = phi i32 [ 0, %.preheader1 ], [ %6240, %._crit_edge339.._crit_edge339_crit_edge ]
  %6232 = zext i32 %6231 to i64		; visa id: 7914
  %6233 = shl nuw nsw i64 %6232, 2		; visa id: 7915
  %6234 = add i64 %5982, %6233		; visa id: 7916
  %6235 = inttoptr i64 %6234 to i32*		; visa id: 7917
  %6236 = load i32, i32* %6235, align 4, !noalias !642		; visa id: 7917
  %6237 = add i64 %5978, %6233		; visa id: 7918
  %6238 = inttoptr i64 %6237 to i32*		; visa id: 7919
  store i32 %6236, i32* %6238, align 4, !alias.scope !642		; visa id: 7919
  %6239 = icmp eq i32 %6231, 0		; visa id: 7920
  br i1 %6239, label %._crit_edge339.._crit_edge339_crit_edge, label %6241, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 7921

._crit_edge339.._crit_edge339_crit_edge:          ; preds = %._crit_edge339
; BB619 :
  %6240 = add nuw nsw i32 %6231, 1, !spirv.Decorations !631		; visa id: 7923
  br label %._crit_edge339, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 7924

6241:                                             ; preds = %._crit_edge339
; BB620 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7926
  %6242 = load i64, i64* %5998, align 8		; visa id: 7926
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7927
  %6243 = icmp slt i32 %6229, %const_reg_dword1		; visa id: 7927
  %6244 = icmp slt i32 %65, %const_reg_dword
  %6245 = and i1 %6244, %6243		; visa id: 7928
  br i1 %6245, label %6246, label %.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7930

.._crit_edge70.176_crit_edge:                     ; preds = %6241
; BB:
  br label %._crit_edge70.176, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6246:                                             ; preds = %6241
; BB622 :
  %6247 = bitcast i64 %6242 to <2 x i32>		; visa id: 7932
  %6248 = extractelement <2 x i32> %6247, i32 0		; visa id: 7934
  %6249 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6248, i32 1
  %6250 = bitcast <2 x i32> %6249 to i64		; visa id: 7934
  %6251 = ashr exact i64 %6250, 32		; visa id: 7935
  %6252 = bitcast i64 %6251 to <2 x i32>		; visa id: 7936
  %6253 = extractelement <2 x i32> %6252, i32 0		; visa id: 7940
  %6254 = extractelement <2 x i32> %6252, i32 1		; visa id: 7940
  %6255 = ashr i64 %6242, 32		; visa id: 7940
  %6256 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6253, i32 %6254, i32 %50, i32 %51)
  %6257 = extractvalue { i32, i32 } %6256, 0		; visa id: 7941
  %6258 = extractvalue { i32, i32 } %6256, 1		; visa id: 7941
  %6259 = insertelement <2 x i32> undef, i32 %6257, i32 0		; visa id: 7948
  %6260 = insertelement <2 x i32> %6259, i32 %6258, i32 1		; visa id: 7949
  %6261 = bitcast <2 x i32> %6260 to i64		; visa id: 7950
  %6262 = add nsw i64 %6261, %6255, !spirv.Decorations !649		; visa id: 7954
  %6263 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !618		; visa id: 7955
  br i1 %86, label %6269, label %6264, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 7956

6264:                                             ; preds = %6246
; BB623 :
  %6265 = shl i64 %6262, 2		; visa id: 7958
  %6266 = add i64 %.in, %6265		; visa id: 7959
  %6267 = inttoptr i64 %6266 to float addrspace(4)*		; visa id: 7960
  %6268 = addrspacecast float addrspace(4)* %6267 to float addrspace(1)*		; visa id: 7960
  store float %6263, float addrspace(1)* %6268, align 4		; visa id: 7961
  br label %._crit_edge70.176, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 7962

6269:                                             ; preds = %6246
; BB624 :
  %6270 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6253, i32 %6254, i32 %47, i32 %48)
  %6271 = extractvalue { i32, i32 } %6270, 0		; visa id: 7964
  %6272 = extractvalue { i32, i32 } %6270, 1		; visa id: 7964
  %6273 = insertelement <2 x i32> undef, i32 %6271, i32 0		; visa id: 7971
  %6274 = insertelement <2 x i32> %6273, i32 %6272, i32 1		; visa id: 7972
  %6275 = bitcast <2 x i32> %6274 to i64		; visa id: 7973
  %6276 = shl i64 %6275, 2		; visa id: 7977
  %6277 = add i64 %.in399, %6276		; visa id: 7978
  %6278 = shl nsw i64 %6255, 2		; visa id: 7979
  %6279 = add i64 %6277, %6278		; visa id: 7980
  %6280 = inttoptr i64 %6279 to float addrspace(4)*		; visa id: 7981
  %6281 = addrspacecast float addrspace(4)* %6280 to float addrspace(1)*		; visa id: 7981
  %6282 = load float, float addrspace(1)* %6281, align 4		; visa id: 7982
  %6283 = fmul reassoc nsz arcp contract float %6282, %4, !spirv.Decorations !618		; visa id: 7983
  %6284 = fadd reassoc nsz arcp contract float %6263, %6283, !spirv.Decorations !618		; visa id: 7984
  %6285 = shl i64 %6262, 2		; visa id: 7985
  %6286 = add i64 %.in, %6285		; visa id: 7986
  %6287 = inttoptr i64 %6286 to float addrspace(4)*		; visa id: 7987
  %6288 = addrspacecast float addrspace(4)* %6287 to float addrspace(1)*		; visa id: 7987
  store float %6284, float addrspace(1)* %6288, align 4		; visa id: 7988
  br label %._crit_edge70.176, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 7989

._crit_edge70.176:                                ; preds = %.._crit_edge70.176_crit_edge, %6269, %6264
; BB625 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 7990
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 7990
  %6289 = insertelement <2 x i32> %6047, i32 %6229, i64 1		; visa id: 7990
  store <2 x i32> %6289, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 7993
  br label %._crit_edge340, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 7995

._crit_edge340:                                   ; preds = %._crit_edge340.._crit_edge340_crit_edge, %._crit_edge70.176
; BB626 :
  %6290 = phi i32 [ 0, %._crit_edge70.176 ], [ %6299, %._crit_edge340.._crit_edge340_crit_edge ]
  %6291 = zext i32 %6290 to i64		; visa id: 7996
  %6292 = shl nuw nsw i64 %6291, 2		; visa id: 7997
  %6293 = add i64 %5982, %6292		; visa id: 7998
  %6294 = inttoptr i64 %6293 to i32*		; visa id: 7999
  %6295 = load i32, i32* %6294, align 4, !noalias !642		; visa id: 7999
  %6296 = add i64 %5978, %6292		; visa id: 8000
  %6297 = inttoptr i64 %6296 to i32*		; visa id: 8001
  store i32 %6295, i32* %6297, align 4, !alias.scope !642		; visa id: 8001
  %6298 = icmp eq i32 %6290, 0		; visa id: 8002
  br i1 %6298, label %._crit_edge340.._crit_edge340_crit_edge, label %6300, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8003

._crit_edge340.._crit_edge340_crit_edge:          ; preds = %._crit_edge340
; BB627 :
  %6299 = add nuw nsw i32 %6290, 1, !spirv.Decorations !631		; visa id: 8005
  br label %._crit_edge340, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8006

6300:                                             ; preds = %._crit_edge340
; BB628 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8008
  %6301 = load i64, i64* %5998, align 8		; visa id: 8008
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8009
  %6302 = icmp slt i32 %6046, %const_reg_dword
  %6303 = icmp slt i32 %6229, %const_reg_dword1		; visa id: 8009
  %6304 = and i1 %6302, %6303		; visa id: 8010
  br i1 %6304, label %6305, label %.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8012

.._crit_edge70.1.1_crit_edge:                     ; preds = %6300
; BB:
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6305:                                             ; preds = %6300
; BB630 :
  %6306 = bitcast i64 %6301 to <2 x i32>		; visa id: 8014
  %6307 = extractelement <2 x i32> %6306, i32 0		; visa id: 8016
  %6308 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6307, i32 1
  %6309 = bitcast <2 x i32> %6308 to i64		; visa id: 8016
  %6310 = ashr exact i64 %6309, 32		; visa id: 8017
  %6311 = bitcast i64 %6310 to <2 x i32>		; visa id: 8018
  %6312 = extractelement <2 x i32> %6311, i32 0		; visa id: 8022
  %6313 = extractelement <2 x i32> %6311, i32 1		; visa id: 8022
  %6314 = ashr i64 %6301, 32		; visa id: 8022
  %6315 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6312, i32 %6313, i32 %50, i32 %51)
  %6316 = extractvalue { i32, i32 } %6315, 0		; visa id: 8023
  %6317 = extractvalue { i32, i32 } %6315, 1		; visa id: 8023
  %6318 = insertelement <2 x i32> undef, i32 %6316, i32 0		; visa id: 8030
  %6319 = insertelement <2 x i32> %6318, i32 %6317, i32 1		; visa id: 8031
  %6320 = bitcast <2 x i32> %6319 to i64		; visa id: 8032
  %6321 = add nsw i64 %6320, %6314, !spirv.Decorations !649		; visa id: 8036
  %6322 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !618		; visa id: 8037
  br i1 %86, label %6328, label %6323, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8038

6323:                                             ; preds = %6305
; BB631 :
  %6324 = shl i64 %6321, 2		; visa id: 8040
  %6325 = add i64 %.in, %6324		; visa id: 8041
  %6326 = inttoptr i64 %6325 to float addrspace(4)*		; visa id: 8042
  %6327 = addrspacecast float addrspace(4)* %6326 to float addrspace(1)*		; visa id: 8042
  store float %6322, float addrspace(1)* %6327, align 4		; visa id: 8043
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8044

6328:                                             ; preds = %6305
; BB632 :
  %6329 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6312, i32 %6313, i32 %47, i32 %48)
  %6330 = extractvalue { i32, i32 } %6329, 0		; visa id: 8046
  %6331 = extractvalue { i32, i32 } %6329, 1		; visa id: 8046
  %6332 = insertelement <2 x i32> undef, i32 %6330, i32 0		; visa id: 8053
  %6333 = insertelement <2 x i32> %6332, i32 %6331, i32 1		; visa id: 8054
  %6334 = bitcast <2 x i32> %6333 to i64		; visa id: 8055
  %6335 = shl i64 %6334, 2		; visa id: 8059
  %6336 = add i64 %.in399, %6335		; visa id: 8060
  %6337 = shl nsw i64 %6314, 2		; visa id: 8061
  %6338 = add i64 %6336, %6337		; visa id: 8062
  %6339 = inttoptr i64 %6338 to float addrspace(4)*		; visa id: 8063
  %6340 = addrspacecast float addrspace(4)* %6339 to float addrspace(1)*		; visa id: 8063
  %6341 = load float, float addrspace(1)* %6340, align 4		; visa id: 8064
  %6342 = fmul reassoc nsz arcp contract float %6341, %4, !spirv.Decorations !618		; visa id: 8065
  %6343 = fadd reassoc nsz arcp contract float %6322, %6342, !spirv.Decorations !618		; visa id: 8066
  %6344 = shl i64 %6321, 2		; visa id: 8067
  %6345 = add i64 %.in, %6344		; visa id: 8068
  %6346 = inttoptr i64 %6345 to float addrspace(4)*		; visa id: 8069
  %6347 = addrspacecast float addrspace(4)* %6346 to float addrspace(1)*		; visa id: 8069
  store float %6343, float addrspace(1)* %6347, align 4		; visa id: 8070
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8071

._crit_edge70.1.1:                                ; preds = %.._crit_edge70.1.1_crit_edge, %6328, %6323
; BB633 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8072
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8072
  %6348 = insertelement <2 x i32> %6108, i32 %6229, i64 1		; visa id: 8072
  store <2 x i32> %6348, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8075
  br label %._crit_edge341, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8077

._crit_edge341:                                   ; preds = %._crit_edge341.._crit_edge341_crit_edge, %._crit_edge70.1.1
; BB634 :
  %6349 = phi i32 [ 0, %._crit_edge70.1.1 ], [ %6358, %._crit_edge341.._crit_edge341_crit_edge ]
  %6350 = zext i32 %6349 to i64		; visa id: 8078
  %6351 = shl nuw nsw i64 %6350, 2		; visa id: 8079
  %6352 = add i64 %5982, %6351		; visa id: 8080
  %6353 = inttoptr i64 %6352 to i32*		; visa id: 8081
  %6354 = load i32, i32* %6353, align 4, !noalias !642		; visa id: 8081
  %6355 = add i64 %5978, %6351		; visa id: 8082
  %6356 = inttoptr i64 %6355 to i32*		; visa id: 8083
  store i32 %6354, i32* %6356, align 4, !alias.scope !642		; visa id: 8083
  %6357 = icmp eq i32 %6349, 0		; visa id: 8084
  br i1 %6357, label %._crit_edge341.._crit_edge341_crit_edge, label %6359, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8085

._crit_edge341.._crit_edge341_crit_edge:          ; preds = %._crit_edge341
; BB635 :
  %6358 = add nuw nsw i32 %6349, 1, !spirv.Decorations !631		; visa id: 8087
  br label %._crit_edge341, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8088

6359:                                             ; preds = %._crit_edge341
; BB636 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8090
  %6360 = load i64, i64* %5998, align 8		; visa id: 8090
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8091
  %6361 = icmp slt i32 %6107, %const_reg_dword
  %6362 = icmp slt i32 %6229, %const_reg_dword1		; visa id: 8091
  %6363 = and i1 %6361, %6362		; visa id: 8092
  br i1 %6363, label %6364, label %.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8094

.._crit_edge70.2.1_crit_edge:                     ; preds = %6359
; BB:
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6364:                                             ; preds = %6359
; BB638 :
  %6365 = bitcast i64 %6360 to <2 x i32>		; visa id: 8096
  %6366 = extractelement <2 x i32> %6365, i32 0		; visa id: 8098
  %6367 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6366, i32 1
  %6368 = bitcast <2 x i32> %6367 to i64		; visa id: 8098
  %6369 = ashr exact i64 %6368, 32		; visa id: 8099
  %6370 = bitcast i64 %6369 to <2 x i32>		; visa id: 8100
  %6371 = extractelement <2 x i32> %6370, i32 0		; visa id: 8104
  %6372 = extractelement <2 x i32> %6370, i32 1		; visa id: 8104
  %6373 = ashr i64 %6360, 32		; visa id: 8104
  %6374 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6371, i32 %6372, i32 %50, i32 %51)
  %6375 = extractvalue { i32, i32 } %6374, 0		; visa id: 8105
  %6376 = extractvalue { i32, i32 } %6374, 1		; visa id: 8105
  %6377 = insertelement <2 x i32> undef, i32 %6375, i32 0		; visa id: 8112
  %6378 = insertelement <2 x i32> %6377, i32 %6376, i32 1		; visa id: 8113
  %6379 = bitcast <2 x i32> %6378 to i64		; visa id: 8114
  %6380 = add nsw i64 %6379, %6373, !spirv.Decorations !649		; visa id: 8118
  %6381 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !618		; visa id: 8119
  br i1 %86, label %6387, label %6382, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8120

6382:                                             ; preds = %6364
; BB639 :
  %6383 = shl i64 %6380, 2		; visa id: 8122
  %6384 = add i64 %.in, %6383		; visa id: 8123
  %6385 = inttoptr i64 %6384 to float addrspace(4)*		; visa id: 8124
  %6386 = addrspacecast float addrspace(4)* %6385 to float addrspace(1)*		; visa id: 8124
  store float %6381, float addrspace(1)* %6386, align 4		; visa id: 8125
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8126

6387:                                             ; preds = %6364
; BB640 :
  %6388 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6371, i32 %6372, i32 %47, i32 %48)
  %6389 = extractvalue { i32, i32 } %6388, 0		; visa id: 8128
  %6390 = extractvalue { i32, i32 } %6388, 1		; visa id: 8128
  %6391 = insertelement <2 x i32> undef, i32 %6389, i32 0		; visa id: 8135
  %6392 = insertelement <2 x i32> %6391, i32 %6390, i32 1		; visa id: 8136
  %6393 = bitcast <2 x i32> %6392 to i64		; visa id: 8137
  %6394 = shl i64 %6393, 2		; visa id: 8141
  %6395 = add i64 %.in399, %6394		; visa id: 8142
  %6396 = shl nsw i64 %6373, 2		; visa id: 8143
  %6397 = add i64 %6395, %6396		; visa id: 8144
  %6398 = inttoptr i64 %6397 to float addrspace(4)*		; visa id: 8145
  %6399 = addrspacecast float addrspace(4)* %6398 to float addrspace(1)*		; visa id: 8145
  %6400 = load float, float addrspace(1)* %6399, align 4		; visa id: 8146
  %6401 = fmul reassoc nsz arcp contract float %6400, %4, !spirv.Decorations !618		; visa id: 8147
  %6402 = fadd reassoc nsz arcp contract float %6381, %6401, !spirv.Decorations !618		; visa id: 8148
  %6403 = shl i64 %6380, 2		; visa id: 8149
  %6404 = add i64 %.in, %6403		; visa id: 8150
  %6405 = inttoptr i64 %6404 to float addrspace(4)*		; visa id: 8151
  %6406 = addrspacecast float addrspace(4)* %6405 to float addrspace(1)*		; visa id: 8151
  store float %6402, float addrspace(1)* %6406, align 4		; visa id: 8152
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8153

._crit_edge70.2.1:                                ; preds = %.._crit_edge70.2.1_crit_edge, %6387, %6382
; BB641 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8154
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8154
  %6407 = insertelement <2 x i32> %6169, i32 %6229, i64 1		; visa id: 8154
  store <2 x i32> %6407, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8157
  br label %._crit_edge342, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8159

._crit_edge342:                                   ; preds = %._crit_edge342.._crit_edge342_crit_edge, %._crit_edge70.2.1
; BB642 :
  %6408 = phi i32 [ 0, %._crit_edge70.2.1 ], [ %6417, %._crit_edge342.._crit_edge342_crit_edge ]
  %6409 = zext i32 %6408 to i64		; visa id: 8160
  %6410 = shl nuw nsw i64 %6409, 2		; visa id: 8161
  %6411 = add i64 %5982, %6410		; visa id: 8162
  %6412 = inttoptr i64 %6411 to i32*		; visa id: 8163
  %6413 = load i32, i32* %6412, align 4, !noalias !642		; visa id: 8163
  %6414 = add i64 %5978, %6410		; visa id: 8164
  %6415 = inttoptr i64 %6414 to i32*		; visa id: 8165
  store i32 %6413, i32* %6415, align 4, !alias.scope !642		; visa id: 8165
  %6416 = icmp eq i32 %6408, 0		; visa id: 8166
  br i1 %6416, label %._crit_edge342.._crit_edge342_crit_edge, label %6418, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8167

._crit_edge342.._crit_edge342_crit_edge:          ; preds = %._crit_edge342
; BB643 :
  %6417 = add nuw nsw i32 %6408, 1, !spirv.Decorations !631		; visa id: 8169
  br label %._crit_edge342, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8170

6418:                                             ; preds = %._crit_edge342
; BB644 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8172
  %6419 = load i64, i64* %5998, align 8		; visa id: 8172
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8173
  %6420 = icmp slt i32 %6168, %const_reg_dword
  %6421 = icmp slt i32 %6229, %const_reg_dword1		; visa id: 8173
  %6422 = and i1 %6420, %6421		; visa id: 8174
  br i1 %6422, label %6423, label %..preheader1.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8176

..preheader1.1_crit_edge:                         ; preds = %6418
; BB:
  br label %.preheader1.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6423:                                             ; preds = %6418
; BB646 :
  %6424 = bitcast i64 %6419 to <2 x i32>		; visa id: 8178
  %6425 = extractelement <2 x i32> %6424, i32 0		; visa id: 8180
  %6426 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6425, i32 1
  %6427 = bitcast <2 x i32> %6426 to i64		; visa id: 8180
  %6428 = ashr exact i64 %6427, 32		; visa id: 8181
  %6429 = bitcast i64 %6428 to <2 x i32>		; visa id: 8182
  %6430 = extractelement <2 x i32> %6429, i32 0		; visa id: 8186
  %6431 = extractelement <2 x i32> %6429, i32 1		; visa id: 8186
  %6432 = ashr i64 %6419, 32		; visa id: 8186
  %6433 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6430, i32 %6431, i32 %50, i32 %51)
  %6434 = extractvalue { i32, i32 } %6433, 0		; visa id: 8187
  %6435 = extractvalue { i32, i32 } %6433, 1		; visa id: 8187
  %6436 = insertelement <2 x i32> undef, i32 %6434, i32 0		; visa id: 8194
  %6437 = insertelement <2 x i32> %6436, i32 %6435, i32 1		; visa id: 8195
  %6438 = bitcast <2 x i32> %6437 to i64		; visa id: 8196
  %6439 = add nsw i64 %6438, %6432, !spirv.Decorations !649		; visa id: 8200
  %6440 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !618		; visa id: 8201
  br i1 %86, label %6446, label %6441, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8202

6441:                                             ; preds = %6423
; BB647 :
  %6442 = shl i64 %6439, 2		; visa id: 8204
  %6443 = add i64 %.in, %6442		; visa id: 8205
  %6444 = inttoptr i64 %6443 to float addrspace(4)*		; visa id: 8206
  %6445 = addrspacecast float addrspace(4)* %6444 to float addrspace(1)*		; visa id: 8206
  store float %6440, float addrspace(1)* %6445, align 4		; visa id: 8207
  br label %.preheader1.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8208

6446:                                             ; preds = %6423
; BB648 :
  %6447 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6430, i32 %6431, i32 %47, i32 %48)
  %6448 = extractvalue { i32, i32 } %6447, 0		; visa id: 8210
  %6449 = extractvalue { i32, i32 } %6447, 1		; visa id: 8210
  %6450 = insertelement <2 x i32> undef, i32 %6448, i32 0		; visa id: 8217
  %6451 = insertelement <2 x i32> %6450, i32 %6449, i32 1		; visa id: 8218
  %6452 = bitcast <2 x i32> %6451 to i64		; visa id: 8219
  %6453 = shl i64 %6452, 2		; visa id: 8223
  %6454 = add i64 %.in399, %6453		; visa id: 8224
  %6455 = shl nsw i64 %6432, 2		; visa id: 8225
  %6456 = add i64 %6454, %6455		; visa id: 8226
  %6457 = inttoptr i64 %6456 to float addrspace(4)*		; visa id: 8227
  %6458 = addrspacecast float addrspace(4)* %6457 to float addrspace(1)*		; visa id: 8227
  %6459 = load float, float addrspace(1)* %6458, align 4		; visa id: 8228
  %6460 = fmul reassoc nsz arcp contract float %6459, %4, !spirv.Decorations !618		; visa id: 8229
  %6461 = fadd reassoc nsz arcp contract float %6440, %6460, !spirv.Decorations !618		; visa id: 8230
  %6462 = shl i64 %6439, 2		; visa id: 8231
  %6463 = add i64 %.in, %6462		; visa id: 8232
  %6464 = inttoptr i64 %6463 to float addrspace(4)*		; visa id: 8233
  %6465 = addrspacecast float addrspace(4)* %6464 to float addrspace(1)*		; visa id: 8233
  store float %6461, float addrspace(1)* %6465, align 4		; visa id: 8234
  br label %.preheader1.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8235

.preheader1.1:                                    ; preds = %..preheader1.1_crit_edge, %6446, %6441
; BB649 :
  %6466 = add i32 %69, 2		; visa id: 8236
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8237
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8237
  %6467 = insertelement <2 x i32> %5984, i32 %6466, i64 1		; visa id: 8237
  store <2 x i32> %6467, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8240
  br label %._crit_edge343, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8242

._crit_edge343:                                   ; preds = %._crit_edge343.._crit_edge343_crit_edge, %.preheader1.1
; BB650 :
  %6468 = phi i32 [ 0, %.preheader1.1 ], [ %6477, %._crit_edge343.._crit_edge343_crit_edge ]
  %6469 = zext i32 %6468 to i64		; visa id: 8243
  %6470 = shl nuw nsw i64 %6469, 2		; visa id: 8244
  %6471 = add i64 %5982, %6470		; visa id: 8245
  %6472 = inttoptr i64 %6471 to i32*		; visa id: 8246
  %6473 = load i32, i32* %6472, align 4, !noalias !642		; visa id: 8246
  %6474 = add i64 %5978, %6470		; visa id: 8247
  %6475 = inttoptr i64 %6474 to i32*		; visa id: 8248
  store i32 %6473, i32* %6475, align 4, !alias.scope !642		; visa id: 8248
  %6476 = icmp eq i32 %6468, 0		; visa id: 8249
  br i1 %6476, label %._crit_edge343.._crit_edge343_crit_edge, label %6478, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8250

._crit_edge343.._crit_edge343_crit_edge:          ; preds = %._crit_edge343
; BB651 :
  %6477 = add nuw nsw i32 %6468, 1, !spirv.Decorations !631		; visa id: 8252
  br label %._crit_edge343, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8253

6478:                                             ; preds = %._crit_edge343
; BB652 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8255
  %6479 = load i64, i64* %5998, align 8		; visa id: 8255
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8256
  %6480 = icmp slt i32 %6466, %const_reg_dword1		; visa id: 8256
  %6481 = icmp slt i32 %65, %const_reg_dword
  %6482 = and i1 %6481, %6480		; visa id: 8257
  br i1 %6482, label %6483, label %.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8259

.._crit_edge70.277_crit_edge:                     ; preds = %6478
; BB:
  br label %._crit_edge70.277, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6483:                                             ; preds = %6478
; BB654 :
  %6484 = bitcast i64 %6479 to <2 x i32>		; visa id: 8261
  %6485 = extractelement <2 x i32> %6484, i32 0		; visa id: 8263
  %6486 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6485, i32 1
  %6487 = bitcast <2 x i32> %6486 to i64		; visa id: 8263
  %6488 = ashr exact i64 %6487, 32		; visa id: 8264
  %6489 = bitcast i64 %6488 to <2 x i32>		; visa id: 8265
  %6490 = extractelement <2 x i32> %6489, i32 0		; visa id: 8269
  %6491 = extractelement <2 x i32> %6489, i32 1		; visa id: 8269
  %6492 = ashr i64 %6479, 32		; visa id: 8269
  %6493 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6490, i32 %6491, i32 %50, i32 %51)
  %6494 = extractvalue { i32, i32 } %6493, 0		; visa id: 8270
  %6495 = extractvalue { i32, i32 } %6493, 1		; visa id: 8270
  %6496 = insertelement <2 x i32> undef, i32 %6494, i32 0		; visa id: 8277
  %6497 = insertelement <2 x i32> %6496, i32 %6495, i32 1		; visa id: 8278
  %6498 = bitcast <2 x i32> %6497 to i64		; visa id: 8279
  %6499 = add nsw i64 %6498, %6492, !spirv.Decorations !649		; visa id: 8283
  %6500 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !618		; visa id: 8284
  br i1 %86, label %6506, label %6501, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8285

6501:                                             ; preds = %6483
; BB655 :
  %6502 = shl i64 %6499, 2		; visa id: 8287
  %6503 = add i64 %.in, %6502		; visa id: 8288
  %6504 = inttoptr i64 %6503 to float addrspace(4)*		; visa id: 8289
  %6505 = addrspacecast float addrspace(4)* %6504 to float addrspace(1)*		; visa id: 8289
  store float %6500, float addrspace(1)* %6505, align 4		; visa id: 8290
  br label %._crit_edge70.277, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8291

6506:                                             ; preds = %6483
; BB656 :
  %6507 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6490, i32 %6491, i32 %47, i32 %48)
  %6508 = extractvalue { i32, i32 } %6507, 0		; visa id: 8293
  %6509 = extractvalue { i32, i32 } %6507, 1		; visa id: 8293
  %6510 = insertelement <2 x i32> undef, i32 %6508, i32 0		; visa id: 8300
  %6511 = insertelement <2 x i32> %6510, i32 %6509, i32 1		; visa id: 8301
  %6512 = bitcast <2 x i32> %6511 to i64		; visa id: 8302
  %6513 = shl i64 %6512, 2		; visa id: 8306
  %6514 = add i64 %.in399, %6513		; visa id: 8307
  %6515 = shl nsw i64 %6492, 2		; visa id: 8308
  %6516 = add i64 %6514, %6515		; visa id: 8309
  %6517 = inttoptr i64 %6516 to float addrspace(4)*		; visa id: 8310
  %6518 = addrspacecast float addrspace(4)* %6517 to float addrspace(1)*		; visa id: 8310
  %6519 = load float, float addrspace(1)* %6518, align 4		; visa id: 8311
  %6520 = fmul reassoc nsz arcp contract float %6519, %4, !spirv.Decorations !618		; visa id: 8312
  %6521 = fadd reassoc nsz arcp contract float %6500, %6520, !spirv.Decorations !618		; visa id: 8313
  %6522 = shl i64 %6499, 2		; visa id: 8314
  %6523 = add i64 %.in, %6522		; visa id: 8315
  %6524 = inttoptr i64 %6523 to float addrspace(4)*		; visa id: 8316
  %6525 = addrspacecast float addrspace(4)* %6524 to float addrspace(1)*		; visa id: 8316
  store float %6521, float addrspace(1)* %6525, align 4		; visa id: 8317
  br label %._crit_edge70.277, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8318

._crit_edge70.277:                                ; preds = %.._crit_edge70.277_crit_edge, %6506, %6501
; BB657 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8319
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8319
  %6526 = insertelement <2 x i32> %6047, i32 %6466, i64 1		; visa id: 8319
  store <2 x i32> %6526, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8322
  br label %._crit_edge344, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8324

._crit_edge344:                                   ; preds = %._crit_edge344.._crit_edge344_crit_edge, %._crit_edge70.277
; BB658 :
  %6527 = phi i32 [ 0, %._crit_edge70.277 ], [ %6536, %._crit_edge344.._crit_edge344_crit_edge ]
  %6528 = zext i32 %6527 to i64		; visa id: 8325
  %6529 = shl nuw nsw i64 %6528, 2		; visa id: 8326
  %6530 = add i64 %5982, %6529		; visa id: 8327
  %6531 = inttoptr i64 %6530 to i32*		; visa id: 8328
  %6532 = load i32, i32* %6531, align 4, !noalias !642		; visa id: 8328
  %6533 = add i64 %5978, %6529		; visa id: 8329
  %6534 = inttoptr i64 %6533 to i32*		; visa id: 8330
  store i32 %6532, i32* %6534, align 4, !alias.scope !642		; visa id: 8330
  %6535 = icmp eq i32 %6527, 0		; visa id: 8331
  br i1 %6535, label %._crit_edge344.._crit_edge344_crit_edge, label %6537, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8332

._crit_edge344.._crit_edge344_crit_edge:          ; preds = %._crit_edge344
; BB659 :
  %6536 = add nuw nsw i32 %6527, 1, !spirv.Decorations !631		; visa id: 8334
  br label %._crit_edge344, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8335

6537:                                             ; preds = %._crit_edge344
; BB660 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8337
  %6538 = load i64, i64* %5998, align 8		; visa id: 8337
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8338
  %6539 = icmp slt i32 %6046, %const_reg_dword
  %6540 = icmp slt i32 %6466, %const_reg_dword1		; visa id: 8338
  %6541 = and i1 %6539, %6540		; visa id: 8339
  br i1 %6541, label %6542, label %.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8341

.._crit_edge70.1.2_crit_edge:                     ; preds = %6537
; BB:
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6542:                                             ; preds = %6537
; BB662 :
  %6543 = bitcast i64 %6538 to <2 x i32>		; visa id: 8343
  %6544 = extractelement <2 x i32> %6543, i32 0		; visa id: 8345
  %6545 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6544, i32 1
  %6546 = bitcast <2 x i32> %6545 to i64		; visa id: 8345
  %6547 = ashr exact i64 %6546, 32		; visa id: 8346
  %6548 = bitcast i64 %6547 to <2 x i32>		; visa id: 8347
  %6549 = extractelement <2 x i32> %6548, i32 0		; visa id: 8351
  %6550 = extractelement <2 x i32> %6548, i32 1		; visa id: 8351
  %6551 = ashr i64 %6538, 32		; visa id: 8351
  %6552 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6549, i32 %6550, i32 %50, i32 %51)
  %6553 = extractvalue { i32, i32 } %6552, 0		; visa id: 8352
  %6554 = extractvalue { i32, i32 } %6552, 1		; visa id: 8352
  %6555 = insertelement <2 x i32> undef, i32 %6553, i32 0		; visa id: 8359
  %6556 = insertelement <2 x i32> %6555, i32 %6554, i32 1		; visa id: 8360
  %6557 = bitcast <2 x i32> %6556 to i64		; visa id: 8361
  %6558 = add nsw i64 %6557, %6551, !spirv.Decorations !649		; visa id: 8365
  %6559 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !618		; visa id: 8366
  br i1 %86, label %6565, label %6560, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8367

6560:                                             ; preds = %6542
; BB663 :
  %6561 = shl i64 %6558, 2		; visa id: 8369
  %6562 = add i64 %.in, %6561		; visa id: 8370
  %6563 = inttoptr i64 %6562 to float addrspace(4)*		; visa id: 8371
  %6564 = addrspacecast float addrspace(4)* %6563 to float addrspace(1)*		; visa id: 8371
  store float %6559, float addrspace(1)* %6564, align 4		; visa id: 8372
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8373

6565:                                             ; preds = %6542
; BB664 :
  %6566 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6549, i32 %6550, i32 %47, i32 %48)
  %6567 = extractvalue { i32, i32 } %6566, 0		; visa id: 8375
  %6568 = extractvalue { i32, i32 } %6566, 1		; visa id: 8375
  %6569 = insertelement <2 x i32> undef, i32 %6567, i32 0		; visa id: 8382
  %6570 = insertelement <2 x i32> %6569, i32 %6568, i32 1		; visa id: 8383
  %6571 = bitcast <2 x i32> %6570 to i64		; visa id: 8384
  %6572 = shl i64 %6571, 2		; visa id: 8388
  %6573 = add i64 %.in399, %6572		; visa id: 8389
  %6574 = shl nsw i64 %6551, 2		; visa id: 8390
  %6575 = add i64 %6573, %6574		; visa id: 8391
  %6576 = inttoptr i64 %6575 to float addrspace(4)*		; visa id: 8392
  %6577 = addrspacecast float addrspace(4)* %6576 to float addrspace(1)*		; visa id: 8392
  %6578 = load float, float addrspace(1)* %6577, align 4		; visa id: 8393
  %6579 = fmul reassoc nsz arcp contract float %6578, %4, !spirv.Decorations !618		; visa id: 8394
  %6580 = fadd reassoc nsz arcp contract float %6559, %6579, !spirv.Decorations !618		; visa id: 8395
  %6581 = shl i64 %6558, 2		; visa id: 8396
  %6582 = add i64 %.in, %6581		; visa id: 8397
  %6583 = inttoptr i64 %6582 to float addrspace(4)*		; visa id: 8398
  %6584 = addrspacecast float addrspace(4)* %6583 to float addrspace(1)*		; visa id: 8398
  store float %6580, float addrspace(1)* %6584, align 4		; visa id: 8399
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8400

._crit_edge70.1.2:                                ; preds = %.._crit_edge70.1.2_crit_edge, %6565, %6560
; BB665 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8401
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8401
  %6585 = insertelement <2 x i32> %6108, i32 %6466, i64 1		; visa id: 8401
  store <2 x i32> %6585, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8404
  br label %._crit_edge345, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8406

._crit_edge345:                                   ; preds = %._crit_edge345.._crit_edge345_crit_edge, %._crit_edge70.1.2
; BB666 :
  %6586 = phi i32 [ 0, %._crit_edge70.1.2 ], [ %6595, %._crit_edge345.._crit_edge345_crit_edge ]
  %6587 = zext i32 %6586 to i64		; visa id: 8407
  %6588 = shl nuw nsw i64 %6587, 2		; visa id: 8408
  %6589 = add i64 %5982, %6588		; visa id: 8409
  %6590 = inttoptr i64 %6589 to i32*		; visa id: 8410
  %6591 = load i32, i32* %6590, align 4, !noalias !642		; visa id: 8410
  %6592 = add i64 %5978, %6588		; visa id: 8411
  %6593 = inttoptr i64 %6592 to i32*		; visa id: 8412
  store i32 %6591, i32* %6593, align 4, !alias.scope !642		; visa id: 8412
  %6594 = icmp eq i32 %6586, 0		; visa id: 8413
  br i1 %6594, label %._crit_edge345.._crit_edge345_crit_edge, label %6596, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8414

._crit_edge345.._crit_edge345_crit_edge:          ; preds = %._crit_edge345
; BB667 :
  %6595 = add nuw nsw i32 %6586, 1, !spirv.Decorations !631		; visa id: 8416
  br label %._crit_edge345, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8417

6596:                                             ; preds = %._crit_edge345
; BB668 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8419
  %6597 = load i64, i64* %5998, align 8		; visa id: 8419
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8420
  %6598 = icmp slt i32 %6107, %const_reg_dword
  %6599 = icmp slt i32 %6466, %const_reg_dword1		; visa id: 8420
  %6600 = and i1 %6598, %6599		; visa id: 8421
  br i1 %6600, label %6601, label %.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8423

.._crit_edge70.2.2_crit_edge:                     ; preds = %6596
; BB:
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6601:                                             ; preds = %6596
; BB670 :
  %6602 = bitcast i64 %6597 to <2 x i32>		; visa id: 8425
  %6603 = extractelement <2 x i32> %6602, i32 0		; visa id: 8427
  %6604 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6603, i32 1
  %6605 = bitcast <2 x i32> %6604 to i64		; visa id: 8427
  %6606 = ashr exact i64 %6605, 32		; visa id: 8428
  %6607 = bitcast i64 %6606 to <2 x i32>		; visa id: 8429
  %6608 = extractelement <2 x i32> %6607, i32 0		; visa id: 8433
  %6609 = extractelement <2 x i32> %6607, i32 1		; visa id: 8433
  %6610 = ashr i64 %6597, 32		; visa id: 8433
  %6611 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6608, i32 %6609, i32 %50, i32 %51)
  %6612 = extractvalue { i32, i32 } %6611, 0		; visa id: 8434
  %6613 = extractvalue { i32, i32 } %6611, 1		; visa id: 8434
  %6614 = insertelement <2 x i32> undef, i32 %6612, i32 0		; visa id: 8441
  %6615 = insertelement <2 x i32> %6614, i32 %6613, i32 1		; visa id: 8442
  %6616 = bitcast <2 x i32> %6615 to i64		; visa id: 8443
  %6617 = add nsw i64 %6616, %6610, !spirv.Decorations !649		; visa id: 8447
  %6618 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !618		; visa id: 8448
  br i1 %86, label %6624, label %6619, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8449

6619:                                             ; preds = %6601
; BB671 :
  %6620 = shl i64 %6617, 2		; visa id: 8451
  %6621 = add i64 %.in, %6620		; visa id: 8452
  %6622 = inttoptr i64 %6621 to float addrspace(4)*		; visa id: 8453
  %6623 = addrspacecast float addrspace(4)* %6622 to float addrspace(1)*		; visa id: 8453
  store float %6618, float addrspace(1)* %6623, align 4		; visa id: 8454
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8455

6624:                                             ; preds = %6601
; BB672 :
  %6625 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6608, i32 %6609, i32 %47, i32 %48)
  %6626 = extractvalue { i32, i32 } %6625, 0		; visa id: 8457
  %6627 = extractvalue { i32, i32 } %6625, 1		; visa id: 8457
  %6628 = insertelement <2 x i32> undef, i32 %6626, i32 0		; visa id: 8464
  %6629 = insertelement <2 x i32> %6628, i32 %6627, i32 1		; visa id: 8465
  %6630 = bitcast <2 x i32> %6629 to i64		; visa id: 8466
  %6631 = shl i64 %6630, 2		; visa id: 8470
  %6632 = add i64 %.in399, %6631		; visa id: 8471
  %6633 = shl nsw i64 %6610, 2		; visa id: 8472
  %6634 = add i64 %6632, %6633		; visa id: 8473
  %6635 = inttoptr i64 %6634 to float addrspace(4)*		; visa id: 8474
  %6636 = addrspacecast float addrspace(4)* %6635 to float addrspace(1)*		; visa id: 8474
  %6637 = load float, float addrspace(1)* %6636, align 4		; visa id: 8475
  %6638 = fmul reassoc nsz arcp contract float %6637, %4, !spirv.Decorations !618		; visa id: 8476
  %6639 = fadd reassoc nsz arcp contract float %6618, %6638, !spirv.Decorations !618		; visa id: 8477
  %6640 = shl i64 %6617, 2		; visa id: 8478
  %6641 = add i64 %.in, %6640		; visa id: 8479
  %6642 = inttoptr i64 %6641 to float addrspace(4)*		; visa id: 8480
  %6643 = addrspacecast float addrspace(4)* %6642 to float addrspace(1)*		; visa id: 8480
  store float %6639, float addrspace(1)* %6643, align 4		; visa id: 8481
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8482

._crit_edge70.2.2:                                ; preds = %.._crit_edge70.2.2_crit_edge, %6624, %6619
; BB673 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8483
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8483
  %6644 = insertelement <2 x i32> %6169, i32 %6466, i64 1		; visa id: 8483
  store <2 x i32> %6644, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8486
  br label %._crit_edge346, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8488

._crit_edge346:                                   ; preds = %._crit_edge346.._crit_edge346_crit_edge, %._crit_edge70.2.2
; BB674 :
  %6645 = phi i32 [ 0, %._crit_edge70.2.2 ], [ %6654, %._crit_edge346.._crit_edge346_crit_edge ]
  %6646 = zext i32 %6645 to i64		; visa id: 8489
  %6647 = shl nuw nsw i64 %6646, 2		; visa id: 8490
  %6648 = add i64 %5982, %6647		; visa id: 8491
  %6649 = inttoptr i64 %6648 to i32*		; visa id: 8492
  %6650 = load i32, i32* %6649, align 4, !noalias !642		; visa id: 8492
  %6651 = add i64 %5978, %6647		; visa id: 8493
  %6652 = inttoptr i64 %6651 to i32*		; visa id: 8494
  store i32 %6650, i32* %6652, align 4, !alias.scope !642		; visa id: 8494
  %6653 = icmp eq i32 %6645, 0		; visa id: 8495
  br i1 %6653, label %._crit_edge346.._crit_edge346_crit_edge, label %6655, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8496

._crit_edge346.._crit_edge346_crit_edge:          ; preds = %._crit_edge346
; BB675 :
  %6654 = add nuw nsw i32 %6645, 1, !spirv.Decorations !631		; visa id: 8498
  br label %._crit_edge346, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8499

6655:                                             ; preds = %._crit_edge346
; BB676 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8501
  %6656 = load i64, i64* %5998, align 8		; visa id: 8501
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8502
  %6657 = icmp slt i32 %6168, %const_reg_dword
  %6658 = icmp slt i32 %6466, %const_reg_dword1		; visa id: 8502
  %6659 = and i1 %6657, %6658		; visa id: 8503
  br i1 %6659, label %6660, label %..preheader1.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8505

..preheader1.2_crit_edge:                         ; preds = %6655
; BB:
  br label %.preheader1.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6660:                                             ; preds = %6655
; BB678 :
  %6661 = bitcast i64 %6656 to <2 x i32>		; visa id: 8507
  %6662 = extractelement <2 x i32> %6661, i32 0		; visa id: 8509
  %6663 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6662, i32 1
  %6664 = bitcast <2 x i32> %6663 to i64		; visa id: 8509
  %6665 = ashr exact i64 %6664, 32		; visa id: 8510
  %6666 = bitcast i64 %6665 to <2 x i32>		; visa id: 8511
  %6667 = extractelement <2 x i32> %6666, i32 0		; visa id: 8515
  %6668 = extractelement <2 x i32> %6666, i32 1		; visa id: 8515
  %6669 = ashr i64 %6656, 32		; visa id: 8515
  %6670 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6667, i32 %6668, i32 %50, i32 %51)
  %6671 = extractvalue { i32, i32 } %6670, 0		; visa id: 8516
  %6672 = extractvalue { i32, i32 } %6670, 1		; visa id: 8516
  %6673 = insertelement <2 x i32> undef, i32 %6671, i32 0		; visa id: 8523
  %6674 = insertelement <2 x i32> %6673, i32 %6672, i32 1		; visa id: 8524
  %6675 = bitcast <2 x i32> %6674 to i64		; visa id: 8525
  %6676 = add nsw i64 %6675, %6669, !spirv.Decorations !649		; visa id: 8529
  %6677 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !618		; visa id: 8530
  br i1 %86, label %6683, label %6678, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8531

6678:                                             ; preds = %6660
; BB679 :
  %6679 = shl i64 %6676, 2		; visa id: 8533
  %6680 = add i64 %.in, %6679		; visa id: 8534
  %6681 = inttoptr i64 %6680 to float addrspace(4)*		; visa id: 8535
  %6682 = addrspacecast float addrspace(4)* %6681 to float addrspace(1)*		; visa id: 8535
  store float %6677, float addrspace(1)* %6682, align 4		; visa id: 8536
  br label %.preheader1.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8537

6683:                                             ; preds = %6660
; BB680 :
  %6684 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6667, i32 %6668, i32 %47, i32 %48)
  %6685 = extractvalue { i32, i32 } %6684, 0		; visa id: 8539
  %6686 = extractvalue { i32, i32 } %6684, 1		; visa id: 8539
  %6687 = insertelement <2 x i32> undef, i32 %6685, i32 0		; visa id: 8546
  %6688 = insertelement <2 x i32> %6687, i32 %6686, i32 1		; visa id: 8547
  %6689 = bitcast <2 x i32> %6688 to i64		; visa id: 8548
  %6690 = shl i64 %6689, 2		; visa id: 8552
  %6691 = add i64 %.in399, %6690		; visa id: 8553
  %6692 = shl nsw i64 %6669, 2		; visa id: 8554
  %6693 = add i64 %6691, %6692		; visa id: 8555
  %6694 = inttoptr i64 %6693 to float addrspace(4)*		; visa id: 8556
  %6695 = addrspacecast float addrspace(4)* %6694 to float addrspace(1)*		; visa id: 8556
  %6696 = load float, float addrspace(1)* %6695, align 4		; visa id: 8557
  %6697 = fmul reassoc nsz arcp contract float %6696, %4, !spirv.Decorations !618		; visa id: 8558
  %6698 = fadd reassoc nsz arcp contract float %6677, %6697, !spirv.Decorations !618		; visa id: 8559
  %6699 = shl i64 %6676, 2		; visa id: 8560
  %6700 = add i64 %.in, %6699		; visa id: 8561
  %6701 = inttoptr i64 %6700 to float addrspace(4)*		; visa id: 8562
  %6702 = addrspacecast float addrspace(4)* %6701 to float addrspace(1)*		; visa id: 8562
  store float %6698, float addrspace(1)* %6702, align 4		; visa id: 8563
  br label %.preheader1.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8564

.preheader1.2:                                    ; preds = %..preheader1.2_crit_edge, %6683, %6678
; BB681 :
  %6703 = add i32 %69, 3		; visa id: 8565
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8566
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8566
  %6704 = insertelement <2 x i32> %5984, i32 %6703, i64 1		; visa id: 8566
  store <2 x i32> %6704, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8569
  br label %._crit_edge347, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8571

._crit_edge347:                                   ; preds = %._crit_edge347.._crit_edge347_crit_edge, %.preheader1.2
; BB682 :
  %6705 = phi i32 [ 0, %.preheader1.2 ], [ %6714, %._crit_edge347.._crit_edge347_crit_edge ]
  %6706 = zext i32 %6705 to i64		; visa id: 8572
  %6707 = shl nuw nsw i64 %6706, 2		; visa id: 8573
  %6708 = add i64 %5982, %6707		; visa id: 8574
  %6709 = inttoptr i64 %6708 to i32*		; visa id: 8575
  %6710 = load i32, i32* %6709, align 4, !noalias !642		; visa id: 8575
  %6711 = add i64 %5978, %6707		; visa id: 8576
  %6712 = inttoptr i64 %6711 to i32*		; visa id: 8577
  store i32 %6710, i32* %6712, align 4, !alias.scope !642		; visa id: 8577
  %6713 = icmp eq i32 %6705, 0		; visa id: 8578
  br i1 %6713, label %._crit_edge347.._crit_edge347_crit_edge, label %6715, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8579

._crit_edge347.._crit_edge347_crit_edge:          ; preds = %._crit_edge347
; BB683 :
  %6714 = add nuw nsw i32 %6705, 1, !spirv.Decorations !631		; visa id: 8581
  br label %._crit_edge347, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8582

6715:                                             ; preds = %._crit_edge347
; BB684 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8584
  %6716 = load i64, i64* %5998, align 8		; visa id: 8584
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8585
  %6717 = icmp slt i32 %6703, %const_reg_dword1		; visa id: 8585
  %6718 = icmp slt i32 %65, %const_reg_dword
  %6719 = and i1 %6718, %6717		; visa id: 8586
  br i1 %6719, label %6720, label %.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8588

.._crit_edge70.378_crit_edge:                     ; preds = %6715
; BB:
  br label %._crit_edge70.378, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6720:                                             ; preds = %6715
; BB686 :
  %6721 = bitcast i64 %6716 to <2 x i32>		; visa id: 8590
  %6722 = extractelement <2 x i32> %6721, i32 0		; visa id: 8592
  %6723 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6722, i32 1
  %6724 = bitcast <2 x i32> %6723 to i64		; visa id: 8592
  %6725 = ashr exact i64 %6724, 32		; visa id: 8593
  %6726 = bitcast i64 %6725 to <2 x i32>		; visa id: 8594
  %6727 = extractelement <2 x i32> %6726, i32 0		; visa id: 8598
  %6728 = extractelement <2 x i32> %6726, i32 1		; visa id: 8598
  %6729 = ashr i64 %6716, 32		; visa id: 8598
  %6730 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6727, i32 %6728, i32 %50, i32 %51)
  %6731 = extractvalue { i32, i32 } %6730, 0		; visa id: 8599
  %6732 = extractvalue { i32, i32 } %6730, 1		; visa id: 8599
  %6733 = insertelement <2 x i32> undef, i32 %6731, i32 0		; visa id: 8606
  %6734 = insertelement <2 x i32> %6733, i32 %6732, i32 1		; visa id: 8607
  %6735 = bitcast <2 x i32> %6734 to i64		; visa id: 8608
  %6736 = add nsw i64 %6735, %6729, !spirv.Decorations !649		; visa id: 8612
  %6737 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !618		; visa id: 8613
  br i1 %86, label %6743, label %6738, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8614

6738:                                             ; preds = %6720
; BB687 :
  %6739 = shl i64 %6736, 2		; visa id: 8616
  %6740 = add i64 %.in, %6739		; visa id: 8617
  %6741 = inttoptr i64 %6740 to float addrspace(4)*		; visa id: 8618
  %6742 = addrspacecast float addrspace(4)* %6741 to float addrspace(1)*		; visa id: 8618
  store float %6737, float addrspace(1)* %6742, align 4		; visa id: 8619
  br label %._crit_edge70.378, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8620

6743:                                             ; preds = %6720
; BB688 :
  %6744 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6727, i32 %6728, i32 %47, i32 %48)
  %6745 = extractvalue { i32, i32 } %6744, 0		; visa id: 8622
  %6746 = extractvalue { i32, i32 } %6744, 1		; visa id: 8622
  %6747 = insertelement <2 x i32> undef, i32 %6745, i32 0		; visa id: 8629
  %6748 = insertelement <2 x i32> %6747, i32 %6746, i32 1		; visa id: 8630
  %6749 = bitcast <2 x i32> %6748 to i64		; visa id: 8631
  %6750 = shl i64 %6749, 2		; visa id: 8635
  %6751 = add i64 %.in399, %6750		; visa id: 8636
  %6752 = shl nsw i64 %6729, 2		; visa id: 8637
  %6753 = add i64 %6751, %6752		; visa id: 8638
  %6754 = inttoptr i64 %6753 to float addrspace(4)*		; visa id: 8639
  %6755 = addrspacecast float addrspace(4)* %6754 to float addrspace(1)*		; visa id: 8639
  %6756 = load float, float addrspace(1)* %6755, align 4		; visa id: 8640
  %6757 = fmul reassoc nsz arcp contract float %6756, %4, !spirv.Decorations !618		; visa id: 8641
  %6758 = fadd reassoc nsz arcp contract float %6737, %6757, !spirv.Decorations !618		; visa id: 8642
  %6759 = shl i64 %6736, 2		; visa id: 8643
  %6760 = add i64 %.in, %6759		; visa id: 8644
  %6761 = inttoptr i64 %6760 to float addrspace(4)*		; visa id: 8645
  %6762 = addrspacecast float addrspace(4)* %6761 to float addrspace(1)*		; visa id: 8645
  store float %6758, float addrspace(1)* %6762, align 4		; visa id: 8646
  br label %._crit_edge70.378, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8647

._crit_edge70.378:                                ; preds = %.._crit_edge70.378_crit_edge, %6743, %6738
; BB689 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8648
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8648
  %6763 = insertelement <2 x i32> %6047, i32 %6703, i64 1		; visa id: 8648
  store <2 x i32> %6763, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8651
  br label %._crit_edge348, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8653

._crit_edge348:                                   ; preds = %._crit_edge348.._crit_edge348_crit_edge, %._crit_edge70.378
; BB690 :
  %6764 = phi i32 [ 0, %._crit_edge70.378 ], [ %6773, %._crit_edge348.._crit_edge348_crit_edge ]
  %6765 = zext i32 %6764 to i64		; visa id: 8654
  %6766 = shl nuw nsw i64 %6765, 2		; visa id: 8655
  %6767 = add i64 %5982, %6766		; visa id: 8656
  %6768 = inttoptr i64 %6767 to i32*		; visa id: 8657
  %6769 = load i32, i32* %6768, align 4, !noalias !642		; visa id: 8657
  %6770 = add i64 %5978, %6766		; visa id: 8658
  %6771 = inttoptr i64 %6770 to i32*		; visa id: 8659
  store i32 %6769, i32* %6771, align 4, !alias.scope !642		; visa id: 8659
  %6772 = icmp eq i32 %6764, 0		; visa id: 8660
  br i1 %6772, label %._crit_edge348.._crit_edge348_crit_edge, label %6774, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8661

._crit_edge348.._crit_edge348_crit_edge:          ; preds = %._crit_edge348
; BB691 :
  %6773 = add nuw nsw i32 %6764, 1, !spirv.Decorations !631		; visa id: 8663
  br label %._crit_edge348, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8664

6774:                                             ; preds = %._crit_edge348
; BB692 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8666
  %6775 = load i64, i64* %5998, align 8		; visa id: 8666
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8667
  %6776 = icmp slt i32 %6046, %const_reg_dword
  %6777 = icmp slt i32 %6703, %const_reg_dword1		; visa id: 8667
  %6778 = and i1 %6776, %6777		; visa id: 8668
  br i1 %6778, label %6779, label %.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8670

.._crit_edge70.1.3_crit_edge:                     ; preds = %6774
; BB:
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6779:                                             ; preds = %6774
; BB694 :
  %6780 = bitcast i64 %6775 to <2 x i32>		; visa id: 8672
  %6781 = extractelement <2 x i32> %6780, i32 0		; visa id: 8674
  %6782 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6781, i32 1
  %6783 = bitcast <2 x i32> %6782 to i64		; visa id: 8674
  %6784 = ashr exact i64 %6783, 32		; visa id: 8675
  %6785 = bitcast i64 %6784 to <2 x i32>		; visa id: 8676
  %6786 = extractelement <2 x i32> %6785, i32 0		; visa id: 8680
  %6787 = extractelement <2 x i32> %6785, i32 1		; visa id: 8680
  %6788 = ashr i64 %6775, 32		; visa id: 8680
  %6789 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6786, i32 %6787, i32 %50, i32 %51)
  %6790 = extractvalue { i32, i32 } %6789, 0		; visa id: 8681
  %6791 = extractvalue { i32, i32 } %6789, 1		; visa id: 8681
  %6792 = insertelement <2 x i32> undef, i32 %6790, i32 0		; visa id: 8688
  %6793 = insertelement <2 x i32> %6792, i32 %6791, i32 1		; visa id: 8689
  %6794 = bitcast <2 x i32> %6793 to i64		; visa id: 8690
  %6795 = add nsw i64 %6794, %6788, !spirv.Decorations !649		; visa id: 8694
  %6796 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !618		; visa id: 8695
  br i1 %86, label %6802, label %6797, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8696

6797:                                             ; preds = %6779
; BB695 :
  %6798 = shl i64 %6795, 2		; visa id: 8698
  %6799 = add i64 %.in, %6798		; visa id: 8699
  %6800 = inttoptr i64 %6799 to float addrspace(4)*		; visa id: 8700
  %6801 = addrspacecast float addrspace(4)* %6800 to float addrspace(1)*		; visa id: 8700
  store float %6796, float addrspace(1)* %6801, align 4		; visa id: 8701
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8702

6802:                                             ; preds = %6779
; BB696 :
  %6803 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6786, i32 %6787, i32 %47, i32 %48)
  %6804 = extractvalue { i32, i32 } %6803, 0		; visa id: 8704
  %6805 = extractvalue { i32, i32 } %6803, 1		; visa id: 8704
  %6806 = insertelement <2 x i32> undef, i32 %6804, i32 0		; visa id: 8711
  %6807 = insertelement <2 x i32> %6806, i32 %6805, i32 1		; visa id: 8712
  %6808 = bitcast <2 x i32> %6807 to i64		; visa id: 8713
  %6809 = shl i64 %6808, 2		; visa id: 8717
  %6810 = add i64 %.in399, %6809		; visa id: 8718
  %6811 = shl nsw i64 %6788, 2		; visa id: 8719
  %6812 = add i64 %6810, %6811		; visa id: 8720
  %6813 = inttoptr i64 %6812 to float addrspace(4)*		; visa id: 8721
  %6814 = addrspacecast float addrspace(4)* %6813 to float addrspace(1)*		; visa id: 8721
  %6815 = load float, float addrspace(1)* %6814, align 4		; visa id: 8722
  %6816 = fmul reassoc nsz arcp contract float %6815, %4, !spirv.Decorations !618		; visa id: 8723
  %6817 = fadd reassoc nsz arcp contract float %6796, %6816, !spirv.Decorations !618		; visa id: 8724
  %6818 = shl i64 %6795, 2		; visa id: 8725
  %6819 = add i64 %.in, %6818		; visa id: 8726
  %6820 = inttoptr i64 %6819 to float addrspace(4)*		; visa id: 8727
  %6821 = addrspacecast float addrspace(4)* %6820 to float addrspace(1)*		; visa id: 8727
  store float %6817, float addrspace(1)* %6821, align 4		; visa id: 8728
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8729

._crit_edge70.1.3:                                ; preds = %.._crit_edge70.1.3_crit_edge, %6802, %6797
; BB697 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8730
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8730
  %6822 = insertelement <2 x i32> %6108, i32 %6703, i64 1		; visa id: 8730
  store <2 x i32> %6822, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8733
  br label %._crit_edge349, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8735

._crit_edge349:                                   ; preds = %._crit_edge349.._crit_edge349_crit_edge, %._crit_edge70.1.3
; BB698 :
  %6823 = phi i32 [ 0, %._crit_edge70.1.3 ], [ %6832, %._crit_edge349.._crit_edge349_crit_edge ]
  %6824 = zext i32 %6823 to i64		; visa id: 8736
  %6825 = shl nuw nsw i64 %6824, 2		; visa id: 8737
  %6826 = add i64 %5982, %6825		; visa id: 8738
  %6827 = inttoptr i64 %6826 to i32*		; visa id: 8739
  %6828 = load i32, i32* %6827, align 4, !noalias !642		; visa id: 8739
  %6829 = add i64 %5978, %6825		; visa id: 8740
  %6830 = inttoptr i64 %6829 to i32*		; visa id: 8741
  store i32 %6828, i32* %6830, align 4, !alias.scope !642		; visa id: 8741
  %6831 = icmp eq i32 %6823, 0		; visa id: 8742
  br i1 %6831, label %._crit_edge349.._crit_edge349_crit_edge, label %6833, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8743

._crit_edge349.._crit_edge349_crit_edge:          ; preds = %._crit_edge349
; BB699 :
  %6832 = add nuw nsw i32 %6823, 1, !spirv.Decorations !631		; visa id: 8745
  br label %._crit_edge349, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8746

6833:                                             ; preds = %._crit_edge349
; BB700 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8748
  %6834 = load i64, i64* %5998, align 8		; visa id: 8748
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8749
  %6835 = icmp slt i32 %6107, %const_reg_dword
  %6836 = icmp slt i32 %6703, %const_reg_dword1		; visa id: 8749
  %6837 = and i1 %6835, %6836		; visa id: 8750
  br i1 %6837, label %6838, label %.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8752

.._crit_edge70.2.3_crit_edge:                     ; preds = %6833
; BB:
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6838:                                             ; preds = %6833
; BB702 :
  %6839 = bitcast i64 %6834 to <2 x i32>		; visa id: 8754
  %6840 = extractelement <2 x i32> %6839, i32 0		; visa id: 8756
  %6841 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6840, i32 1
  %6842 = bitcast <2 x i32> %6841 to i64		; visa id: 8756
  %6843 = ashr exact i64 %6842, 32		; visa id: 8757
  %6844 = bitcast i64 %6843 to <2 x i32>		; visa id: 8758
  %6845 = extractelement <2 x i32> %6844, i32 0		; visa id: 8762
  %6846 = extractelement <2 x i32> %6844, i32 1		; visa id: 8762
  %6847 = ashr i64 %6834, 32		; visa id: 8762
  %6848 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6845, i32 %6846, i32 %50, i32 %51)
  %6849 = extractvalue { i32, i32 } %6848, 0		; visa id: 8763
  %6850 = extractvalue { i32, i32 } %6848, 1		; visa id: 8763
  %6851 = insertelement <2 x i32> undef, i32 %6849, i32 0		; visa id: 8770
  %6852 = insertelement <2 x i32> %6851, i32 %6850, i32 1		; visa id: 8771
  %6853 = bitcast <2 x i32> %6852 to i64		; visa id: 8772
  %6854 = add nsw i64 %6853, %6847, !spirv.Decorations !649		; visa id: 8776
  %6855 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !618		; visa id: 8777
  br i1 %86, label %6861, label %6856, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8778

6856:                                             ; preds = %6838
; BB703 :
  %6857 = shl i64 %6854, 2		; visa id: 8780
  %6858 = add i64 %.in, %6857		; visa id: 8781
  %6859 = inttoptr i64 %6858 to float addrspace(4)*		; visa id: 8782
  %6860 = addrspacecast float addrspace(4)* %6859 to float addrspace(1)*		; visa id: 8782
  store float %6855, float addrspace(1)* %6860, align 4		; visa id: 8783
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8784

6861:                                             ; preds = %6838
; BB704 :
  %6862 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6845, i32 %6846, i32 %47, i32 %48)
  %6863 = extractvalue { i32, i32 } %6862, 0		; visa id: 8786
  %6864 = extractvalue { i32, i32 } %6862, 1		; visa id: 8786
  %6865 = insertelement <2 x i32> undef, i32 %6863, i32 0		; visa id: 8793
  %6866 = insertelement <2 x i32> %6865, i32 %6864, i32 1		; visa id: 8794
  %6867 = bitcast <2 x i32> %6866 to i64		; visa id: 8795
  %6868 = shl i64 %6867, 2		; visa id: 8799
  %6869 = add i64 %.in399, %6868		; visa id: 8800
  %6870 = shl nsw i64 %6847, 2		; visa id: 8801
  %6871 = add i64 %6869, %6870		; visa id: 8802
  %6872 = inttoptr i64 %6871 to float addrspace(4)*		; visa id: 8803
  %6873 = addrspacecast float addrspace(4)* %6872 to float addrspace(1)*		; visa id: 8803
  %6874 = load float, float addrspace(1)* %6873, align 4		; visa id: 8804
  %6875 = fmul reassoc nsz arcp contract float %6874, %4, !spirv.Decorations !618		; visa id: 8805
  %6876 = fadd reassoc nsz arcp contract float %6855, %6875, !spirv.Decorations !618		; visa id: 8806
  %6877 = shl i64 %6854, 2		; visa id: 8807
  %6878 = add i64 %.in, %6877		; visa id: 8808
  %6879 = inttoptr i64 %6878 to float addrspace(4)*		; visa id: 8809
  %6880 = addrspacecast float addrspace(4)* %6879 to float addrspace(1)*		; visa id: 8809
  store float %6876, float addrspace(1)* %6880, align 4		; visa id: 8810
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8811

._crit_edge70.2.3:                                ; preds = %.._crit_edge70.2.3_crit_edge, %6861, %6856
; BB705 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8812
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8812
  %6881 = insertelement <2 x i32> %6169, i32 %6703, i64 1		; visa id: 8812
  store <2 x i32> %6881, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8815
  br label %._crit_edge350, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8817

._crit_edge350:                                   ; preds = %._crit_edge350.._crit_edge350_crit_edge, %._crit_edge70.2.3
; BB706 :
  %6882 = phi i32 [ 0, %._crit_edge70.2.3 ], [ %6891, %._crit_edge350.._crit_edge350_crit_edge ]
  %6883 = zext i32 %6882 to i64		; visa id: 8818
  %6884 = shl nuw nsw i64 %6883, 2		; visa id: 8819
  %6885 = add i64 %5982, %6884		; visa id: 8820
  %6886 = inttoptr i64 %6885 to i32*		; visa id: 8821
  %6887 = load i32, i32* %6886, align 4, !noalias !642		; visa id: 8821
  %6888 = add i64 %5978, %6884		; visa id: 8822
  %6889 = inttoptr i64 %6888 to i32*		; visa id: 8823
  store i32 %6887, i32* %6889, align 4, !alias.scope !642		; visa id: 8823
  %6890 = icmp eq i32 %6882, 0		; visa id: 8824
  br i1 %6890, label %._crit_edge350.._crit_edge350_crit_edge, label %6892, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8825

._crit_edge350.._crit_edge350_crit_edge:          ; preds = %._crit_edge350
; BB707 :
  %6891 = add nuw nsw i32 %6882, 1, !spirv.Decorations !631		; visa id: 8827
  br label %._crit_edge350, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8828

6892:                                             ; preds = %._crit_edge350
; BB708 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8830
  %6893 = load i64, i64* %5998, align 8		; visa id: 8830
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8831
  %6894 = icmp slt i32 %6168, %const_reg_dword
  %6895 = icmp slt i32 %6703, %const_reg_dword1		; visa id: 8831
  %6896 = and i1 %6894, %6895		; visa id: 8832
  br i1 %6896, label %6897, label %..preheader1.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8834

..preheader1.3_crit_edge:                         ; preds = %6892
; BB:
  br label %.preheader1.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6897:                                             ; preds = %6892
; BB710 :
  %6898 = bitcast i64 %6893 to <2 x i32>		; visa id: 8836
  %6899 = extractelement <2 x i32> %6898, i32 0		; visa id: 8838
  %6900 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6899, i32 1
  %6901 = bitcast <2 x i32> %6900 to i64		; visa id: 8838
  %6902 = ashr exact i64 %6901, 32		; visa id: 8839
  %6903 = bitcast i64 %6902 to <2 x i32>		; visa id: 8840
  %6904 = extractelement <2 x i32> %6903, i32 0		; visa id: 8844
  %6905 = extractelement <2 x i32> %6903, i32 1		; visa id: 8844
  %6906 = ashr i64 %6893, 32		; visa id: 8844
  %6907 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6904, i32 %6905, i32 %50, i32 %51)
  %6908 = extractvalue { i32, i32 } %6907, 0		; visa id: 8845
  %6909 = extractvalue { i32, i32 } %6907, 1		; visa id: 8845
  %6910 = insertelement <2 x i32> undef, i32 %6908, i32 0		; visa id: 8852
  %6911 = insertelement <2 x i32> %6910, i32 %6909, i32 1		; visa id: 8853
  %6912 = bitcast <2 x i32> %6911 to i64		; visa id: 8854
  %6913 = add nsw i64 %6912, %6906, !spirv.Decorations !649		; visa id: 8858
  %6914 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !618		; visa id: 8859
  br i1 %86, label %6920, label %6915, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8860

6915:                                             ; preds = %6897
; BB711 :
  %6916 = shl i64 %6913, 2		; visa id: 8862
  %6917 = add i64 %.in, %6916		; visa id: 8863
  %6918 = inttoptr i64 %6917 to float addrspace(4)*		; visa id: 8864
  %6919 = addrspacecast float addrspace(4)* %6918 to float addrspace(1)*		; visa id: 8864
  store float %6914, float addrspace(1)* %6919, align 4		; visa id: 8865
  br label %.preheader1.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8866

6920:                                             ; preds = %6897
; BB712 :
  %6921 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6904, i32 %6905, i32 %47, i32 %48)
  %6922 = extractvalue { i32, i32 } %6921, 0		; visa id: 8868
  %6923 = extractvalue { i32, i32 } %6921, 1		; visa id: 8868
  %6924 = insertelement <2 x i32> undef, i32 %6922, i32 0		; visa id: 8875
  %6925 = insertelement <2 x i32> %6924, i32 %6923, i32 1		; visa id: 8876
  %6926 = bitcast <2 x i32> %6925 to i64		; visa id: 8877
  %6927 = shl i64 %6926, 2		; visa id: 8881
  %6928 = add i64 %.in399, %6927		; visa id: 8882
  %6929 = shl nsw i64 %6906, 2		; visa id: 8883
  %6930 = add i64 %6928, %6929		; visa id: 8884
  %6931 = inttoptr i64 %6930 to float addrspace(4)*		; visa id: 8885
  %6932 = addrspacecast float addrspace(4)* %6931 to float addrspace(1)*		; visa id: 8885
  %6933 = load float, float addrspace(1)* %6932, align 4		; visa id: 8886
  %6934 = fmul reassoc nsz arcp contract float %6933, %4, !spirv.Decorations !618		; visa id: 8887
  %6935 = fadd reassoc nsz arcp contract float %6914, %6934, !spirv.Decorations !618		; visa id: 8888
  %6936 = shl i64 %6913, 2		; visa id: 8889
  %6937 = add i64 %.in, %6936		; visa id: 8890
  %6938 = inttoptr i64 %6937 to float addrspace(4)*		; visa id: 8891
  %6939 = addrspacecast float addrspace(4)* %6938 to float addrspace(1)*		; visa id: 8891
  store float %6935, float addrspace(1)* %6939, align 4		; visa id: 8892
  br label %.preheader1.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8893

.preheader1.3:                                    ; preds = %..preheader1.3_crit_edge, %6920, %6915
; BB713 :
  %6940 = add i32 %69, 4		; visa id: 8894
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8895
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8895
  %6941 = insertelement <2 x i32> %5984, i32 %6940, i64 1		; visa id: 8895
  store <2 x i32> %6941, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8898
  br label %._crit_edge351, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8900

._crit_edge351:                                   ; preds = %._crit_edge351.._crit_edge351_crit_edge, %.preheader1.3
; BB714 :
  %6942 = phi i32 [ 0, %.preheader1.3 ], [ %6951, %._crit_edge351.._crit_edge351_crit_edge ]
  %6943 = zext i32 %6942 to i64		; visa id: 8901
  %6944 = shl nuw nsw i64 %6943, 2		; visa id: 8902
  %6945 = add i64 %5982, %6944		; visa id: 8903
  %6946 = inttoptr i64 %6945 to i32*		; visa id: 8904
  %6947 = load i32, i32* %6946, align 4, !noalias !642		; visa id: 8904
  %6948 = add i64 %5978, %6944		; visa id: 8905
  %6949 = inttoptr i64 %6948 to i32*		; visa id: 8906
  store i32 %6947, i32* %6949, align 4, !alias.scope !642		; visa id: 8906
  %6950 = icmp eq i32 %6942, 0		; visa id: 8907
  br i1 %6950, label %._crit_edge351.._crit_edge351_crit_edge, label %6952, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8908

._crit_edge351.._crit_edge351_crit_edge:          ; preds = %._crit_edge351
; BB715 :
  %6951 = add nuw nsw i32 %6942, 1, !spirv.Decorations !631		; visa id: 8910
  br label %._crit_edge351, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8911

6952:                                             ; preds = %._crit_edge351
; BB716 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8913
  %6953 = load i64, i64* %5998, align 8		; visa id: 8913
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8914
  %6954 = icmp slt i32 %6940, %const_reg_dword1		; visa id: 8914
  %6955 = icmp slt i32 %65, %const_reg_dword
  %6956 = and i1 %6955, %6954		; visa id: 8915
  br i1 %6956, label %6957, label %.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8917

.._crit_edge70.4_crit_edge:                       ; preds = %6952
; BB:
  br label %._crit_edge70.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6957:                                             ; preds = %6952
; BB718 :
  %6958 = bitcast i64 %6953 to <2 x i32>		; visa id: 8919
  %6959 = extractelement <2 x i32> %6958, i32 0		; visa id: 8921
  %6960 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6959, i32 1
  %6961 = bitcast <2 x i32> %6960 to i64		; visa id: 8921
  %6962 = ashr exact i64 %6961, 32		; visa id: 8922
  %6963 = bitcast i64 %6962 to <2 x i32>		; visa id: 8923
  %6964 = extractelement <2 x i32> %6963, i32 0		; visa id: 8927
  %6965 = extractelement <2 x i32> %6963, i32 1		; visa id: 8927
  %6966 = ashr i64 %6953, 32		; visa id: 8927
  %6967 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6964, i32 %6965, i32 %50, i32 %51)
  %6968 = extractvalue { i32, i32 } %6967, 0		; visa id: 8928
  %6969 = extractvalue { i32, i32 } %6967, 1		; visa id: 8928
  %6970 = insertelement <2 x i32> undef, i32 %6968, i32 0		; visa id: 8935
  %6971 = insertelement <2 x i32> %6970, i32 %6969, i32 1		; visa id: 8936
  %6972 = bitcast <2 x i32> %6971 to i64		; visa id: 8937
  %6973 = add nsw i64 %6972, %6966, !spirv.Decorations !649		; visa id: 8941
  %6974 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !618		; visa id: 8942
  br i1 %86, label %6980, label %6975, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8943

6975:                                             ; preds = %6957
; BB719 :
  %6976 = shl i64 %6973, 2		; visa id: 8945
  %6977 = add i64 %.in, %6976		; visa id: 8946
  %6978 = inttoptr i64 %6977 to float addrspace(4)*		; visa id: 8947
  %6979 = addrspacecast float addrspace(4)* %6978 to float addrspace(1)*		; visa id: 8947
  store float %6974, float addrspace(1)* %6979, align 4		; visa id: 8948
  br label %._crit_edge70.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8949

6980:                                             ; preds = %6957
; BB720 :
  %6981 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6964, i32 %6965, i32 %47, i32 %48)
  %6982 = extractvalue { i32, i32 } %6981, 0		; visa id: 8951
  %6983 = extractvalue { i32, i32 } %6981, 1		; visa id: 8951
  %6984 = insertelement <2 x i32> undef, i32 %6982, i32 0		; visa id: 8958
  %6985 = insertelement <2 x i32> %6984, i32 %6983, i32 1		; visa id: 8959
  %6986 = bitcast <2 x i32> %6985 to i64		; visa id: 8960
  %6987 = shl i64 %6986, 2		; visa id: 8964
  %6988 = add i64 %.in399, %6987		; visa id: 8965
  %6989 = shl nsw i64 %6966, 2		; visa id: 8966
  %6990 = add i64 %6988, %6989		; visa id: 8967
  %6991 = inttoptr i64 %6990 to float addrspace(4)*		; visa id: 8968
  %6992 = addrspacecast float addrspace(4)* %6991 to float addrspace(1)*		; visa id: 8968
  %6993 = load float, float addrspace(1)* %6992, align 4		; visa id: 8969
  %6994 = fmul reassoc nsz arcp contract float %6993, %4, !spirv.Decorations !618		; visa id: 8970
  %6995 = fadd reassoc nsz arcp contract float %6974, %6994, !spirv.Decorations !618		; visa id: 8971
  %6996 = shl i64 %6973, 2		; visa id: 8972
  %6997 = add i64 %.in, %6996		; visa id: 8973
  %6998 = inttoptr i64 %6997 to float addrspace(4)*		; visa id: 8974
  %6999 = addrspacecast float addrspace(4)* %6998 to float addrspace(1)*		; visa id: 8974
  store float %6995, float addrspace(1)* %6999, align 4		; visa id: 8975
  br label %._crit_edge70.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8976

._crit_edge70.4:                                  ; preds = %.._crit_edge70.4_crit_edge, %6980, %6975
; BB721 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8977
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8977
  %7000 = insertelement <2 x i32> %6047, i32 %6940, i64 1		; visa id: 8977
  store <2 x i32> %7000, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 8980
  br label %._crit_edge352, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8982

._crit_edge352:                                   ; preds = %._crit_edge352.._crit_edge352_crit_edge, %._crit_edge70.4
; BB722 :
  %7001 = phi i32 [ 0, %._crit_edge70.4 ], [ %7010, %._crit_edge352.._crit_edge352_crit_edge ]
  %7002 = zext i32 %7001 to i64		; visa id: 8983
  %7003 = shl nuw nsw i64 %7002, 2		; visa id: 8984
  %7004 = add i64 %5982, %7003		; visa id: 8985
  %7005 = inttoptr i64 %7004 to i32*		; visa id: 8986
  %7006 = load i32, i32* %7005, align 4, !noalias !642		; visa id: 8986
  %7007 = add i64 %5978, %7003		; visa id: 8987
  %7008 = inttoptr i64 %7007 to i32*		; visa id: 8988
  store i32 %7006, i32* %7008, align 4, !alias.scope !642		; visa id: 8988
  %7009 = icmp eq i32 %7001, 0		; visa id: 8989
  br i1 %7009, label %._crit_edge352.._crit_edge352_crit_edge, label %7011, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8990

._crit_edge352.._crit_edge352_crit_edge:          ; preds = %._crit_edge352
; BB723 :
  %7010 = add nuw nsw i32 %7001, 1, !spirv.Decorations !631		; visa id: 8992
  br label %._crit_edge352, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8993

7011:                                             ; preds = %._crit_edge352
; BB724 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 8995
  %7012 = load i64, i64* %5998, align 8		; visa id: 8995
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 8996
  %7013 = icmp slt i32 %6046, %const_reg_dword
  %7014 = icmp slt i32 %6940, %const_reg_dword1		; visa id: 8996
  %7015 = and i1 %7013, %7014		; visa id: 8997
  br i1 %7015, label %7016, label %.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8999

.._crit_edge70.1.4_crit_edge:                     ; preds = %7011
; BB:
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7016:                                             ; preds = %7011
; BB726 :
  %7017 = bitcast i64 %7012 to <2 x i32>		; visa id: 9001
  %7018 = extractelement <2 x i32> %7017, i32 0		; visa id: 9003
  %7019 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7018, i32 1
  %7020 = bitcast <2 x i32> %7019 to i64		; visa id: 9003
  %7021 = ashr exact i64 %7020, 32		; visa id: 9004
  %7022 = bitcast i64 %7021 to <2 x i32>		; visa id: 9005
  %7023 = extractelement <2 x i32> %7022, i32 0		; visa id: 9009
  %7024 = extractelement <2 x i32> %7022, i32 1		; visa id: 9009
  %7025 = ashr i64 %7012, 32		; visa id: 9009
  %7026 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7023, i32 %7024, i32 %50, i32 %51)
  %7027 = extractvalue { i32, i32 } %7026, 0		; visa id: 9010
  %7028 = extractvalue { i32, i32 } %7026, 1		; visa id: 9010
  %7029 = insertelement <2 x i32> undef, i32 %7027, i32 0		; visa id: 9017
  %7030 = insertelement <2 x i32> %7029, i32 %7028, i32 1		; visa id: 9018
  %7031 = bitcast <2 x i32> %7030 to i64		; visa id: 9019
  %7032 = add nsw i64 %7031, %7025, !spirv.Decorations !649		; visa id: 9023
  %7033 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !618		; visa id: 9024
  br i1 %86, label %7039, label %7034, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9025

7034:                                             ; preds = %7016
; BB727 :
  %7035 = shl i64 %7032, 2		; visa id: 9027
  %7036 = add i64 %.in, %7035		; visa id: 9028
  %7037 = inttoptr i64 %7036 to float addrspace(4)*		; visa id: 9029
  %7038 = addrspacecast float addrspace(4)* %7037 to float addrspace(1)*		; visa id: 9029
  store float %7033, float addrspace(1)* %7038, align 4		; visa id: 9030
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9031

7039:                                             ; preds = %7016
; BB728 :
  %7040 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7023, i32 %7024, i32 %47, i32 %48)
  %7041 = extractvalue { i32, i32 } %7040, 0		; visa id: 9033
  %7042 = extractvalue { i32, i32 } %7040, 1		; visa id: 9033
  %7043 = insertelement <2 x i32> undef, i32 %7041, i32 0		; visa id: 9040
  %7044 = insertelement <2 x i32> %7043, i32 %7042, i32 1		; visa id: 9041
  %7045 = bitcast <2 x i32> %7044 to i64		; visa id: 9042
  %7046 = shl i64 %7045, 2		; visa id: 9046
  %7047 = add i64 %.in399, %7046		; visa id: 9047
  %7048 = shl nsw i64 %7025, 2		; visa id: 9048
  %7049 = add i64 %7047, %7048		; visa id: 9049
  %7050 = inttoptr i64 %7049 to float addrspace(4)*		; visa id: 9050
  %7051 = addrspacecast float addrspace(4)* %7050 to float addrspace(1)*		; visa id: 9050
  %7052 = load float, float addrspace(1)* %7051, align 4		; visa id: 9051
  %7053 = fmul reassoc nsz arcp contract float %7052, %4, !spirv.Decorations !618		; visa id: 9052
  %7054 = fadd reassoc nsz arcp contract float %7033, %7053, !spirv.Decorations !618		; visa id: 9053
  %7055 = shl i64 %7032, 2		; visa id: 9054
  %7056 = add i64 %.in, %7055		; visa id: 9055
  %7057 = inttoptr i64 %7056 to float addrspace(4)*		; visa id: 9056
  %7058 = addrspacecast float addrspace(4)* %7057 to float addrspace(1)*		; visa id: 9056
  store float %7054, float addrspace(1)* %7058, align 4		; visa id: 9057
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9058

._crit_edge70.1.4:                                ; preds = %.._crit_edge70.1.4_crit_edge, %7039, %7034
; BB729 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9059
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9059
  %7059 = insertelement <2 x i32> %6108, i32 %6940, i64 1		; visa id: 9059
  store <2 x i32> %7059, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9062
  br label %._crit_edge353, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9064

._crit_edge353:                                   ; preds = %._crit_edge353.._crit_edge353_crit_edge, %._crit_edge70.1.4
; BB730 :
  %7060 = phi i32 [ 0, %._crit_edge70.1.4 ], [ %7069, %._crit_edge353.._crit_edge353_crit_edge ]
  %7061 = zext i32 %7060 to i64		; visa id: 9065
  %7062 = shl nuw nsw i64 %7061, 2		; visa id: 9066
  %7063 = add i64 %5982, %7062		; visa id: 9067
  %7064 = inttoptr i64 %7063 to i32*		; visa id: 9068
  %7065 = load i32, i32* %7064, align 4, !noalias !642		; visa id: 9068
  %7066 = add i64 %5978, %7062		; visa id: 9069
  %7067 = inttoptr i64 %7066 to i32*		; visa id: 9070
  store i32 %7065, i32* %7067, align 4, !alias.scope !642		; visa id: 9070
  %7068 = icmp eq i32 %7060, 0		; visa id: 9071
  br i1 %7068, label %._crit_edge353.._crit_edge353_crit_edge, label %7070, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9072

._crit_edge353.._crit_edge353_crit_edge:          ; preds = %._crit_edge353
; BB731 :
  %7069 = add nuw nsw i32 %7060, 1, !spirv.Decorations !631		; visa id: 9074
  br label %._crit_edge353, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9075

7070:                                             ; preds = %._crit_edge353
; BB732 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9077
  %7071 = load i64, i64* %5998, align 8		; visa id: 9077
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9078
  %7072 = icmp slt i32 %6107, %const_reg_dword
  %7073 = icmp slt i32 %6940, %const_reg_dword1		; visa id: 9078
  %7074 = and i1 %7072, %7073		; visa id: 9079
  br i1 %7074, label %7075, label %.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9081

.._crit_edge70.2.4_crit_edge:                     ; preds = %7070
; BB:
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7075:                                             ; preds = %7070
; BB734 :
  %7076 = bitcast i64 %7071 to <2 x i32>		; visa id: 9083
  %7077 = extractelement <2 x i32> %7076, i32 0		; visa id: 9085
  %7078 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7077, i32 1
  %7079 = bitcast <2 x i32> %7078 to i64		; visa id: 9085
  %7080 = ashr exact i64 %7079, 32		; visa id: 9086
  %7081 = bitcast i64 %7080 to <2 x i32>		; visa id: 9087
  %7082 = extractelement <2 x i32> %7081, i32 0		; visa id: 9091
  %7083 = extractelement <2 x i32> %7081, i32 1		; visa id: 9091
  %7084 = ashr i64 %7071, 32		; visa id: 9091
  %7085 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7082, i32 %7083, i32 %50, i32 %51)
  %7086 = extractvalue { i32, i32 } %7085, 0		; visa id: 9092
  %7087 = extractvalue { i32, i32 } %7085, 1		; visa id: 9092
  %7088 = insertelement <2 x i32> undef, i32 %7086, i32 0		; visa id: 9099
  %7089 = insertelement <2 x i32> %7088, i32 %7087, i32 1		; visa id: 9100
  %7090 = bitcast <2 x i32> %7089 to i64		; visa id: 9101
  %7091 = add nsw i64 %7090, %7084, !spirv.Decorations !649		; visa id: 9105
  %7092 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !618		; visa id: 9106
  br i1 %86, label %7098, label %7093, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9107

7093:                                             ; preds = %7075
; BB735 :
  %7094 = shl i64 %7091, 2		; visa id: 9109
  %7095 = add i64 %.in, %7094		; visa id: 9110
  %7096 = inttoptr i64 %7095 to float addrspace(4)*		; visa id: 9111
  %7097 = addrspacecast float addrspace(4)* %7096 to float addrspace(1)*		; visa id: 9111
  store float %7092, float addrspace(1)* %7097, align 4		; visa id: 9112
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9113

7098:                                             ; preds = %7075
; BB736 :
  %7099 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7082, i32 %7083, i32 %47, i32 %48)
  %7100 = extractvalue { i32, i32 } %7099, 0		; visa id: 9115
  %7101 = extractvalue { i32, i32 } %7099, 1		; visa id: 9115
  %7102 = insertelement <2 x i32> undef, i32 %7100, i32 0		; visa id: 9122
  %7103 = insertelement <2 x i32> %7102, i32 %7101, i32 1		; visa id: 9123
  %7104 = bitcast <2 x i32> %7103 to i64		; visa id: 9124
  %7105 = shl i64 %7104, 2		; visa id: 9128
  %7106 = add i64 %.in399, %7105		; visa id: 9129
  %7107 = shl nsw i64 %7084, 2		; visa id: 9130
  %7108 = add i64 %7106, %7107		; visa id: 9131
  %7109 = inttoptr i64 %7108 to float addrspace(4)*		; visa id: 9132
  %7110 = addrspacecast float addrspace(4)* %7109 to float addrspace(1)*		; visa id: 9132
  %7111 = load float, float addrspace(1)* %7110, align 4		; visa id: 9133
  %7112 = fmul reassoc nsz arcp contract float %7111, %4, !spirv.Decorations !618		; visa id: 9134
  %7113 = fadd reassoc nsz arcp contract float %7092, %7112, !spirv.Decorations !618		; visa id: 9135
  %7114 = shl i64 %7091, 2		; visa id: 9136
  %7115 = add i64 %.in, %7114		; visa id: 9137
  %7116 = inttoptr i64 %7115 to float addrspace(4)*		; visa id: 9138
  %7117 = addrspacecast float addrspace(4)* %7116 to float addrspace(1)*		; visa id: 9138
  store float %7113, float addrspace(1)* %7117, align 4		; visa id: 9139
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9140

._crit_edge70.2.4:                                ; preds = %.._crit_edge70.2.4_crit_edge, %7098, %7093
; BB737 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9141
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9141
  %7118 = insertelement <2 x i32> %6169, i32 %6940, i64 1		; visa id: 9141
  store <2 x i32> %7118, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9144
  br label %._crit_edge354, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9146

._crit_edge354:                                   ; preds = %._crit_edge354.._crit_edge354_crit_edge, %._crit_edge70.2.4
; BB738 :
  %7119 = phi i32 [ 0, %._crit_edge70.2.4 ], [ %7128, %._crit_edge354.._crit_edge354_crit_edge ]
  %7120 = zext i32 %7119 to i64		; visa id: 9147
  %7121 = shl nuw nsw i64 %7120, 2		; visa id: 9148
  %7122 = add i64 %5982, %7121		; visa id: 9149
  %7123 = inttoptr i64 %7122 to i32*		; visa id: 9150
  %7124 = load i32, i32* %7123, align 4, !noalias !642		; visa id: 9150
  %7125 = add i64 %5978, %7121		; visa id: 9151
  %7126 = inttoptr i64 %7125 to i32*		; visa id: 9152
  store i32 %7124, i32* %7126, align 4, !alias.scope !642		; visa id: 9152
  %7127 = icmp eq i32 %7119, 0		; visa id: 9153
  br i1 %7127, label %._crit_edge354.._crit_edge354_crit_edge, label %7129, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9154

._crit_edge354.._crit_edge354_crit_edge:          ; preds = %._crit_edge354
; BB739 :
  %7128 = add nuw nsw i32 %7119, 1, !spirv.Decorations !631		; visa id: 9156
  br label %._crit_edge354, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9157

7129:                                             ; preds = %._crit_edge354
; BB740 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9159
  %7130 = load i64, i64* %5998, align 8		; visa id: 9159
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9160
  %7131 = icmp slt i32 %6168, %const_reg_dword
  %7132 = icmp slt i32 %6940, %const_reg_dword1		; visa id: 9160
  %7133 = and i1 %7131, %7132		; visa id: 9161
  br i1 %7133, label %7134, label %..preheader1.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9163

..preheader1.4_crit_edge:                         ; preds = %7129
; BB:
  br label %.preheader1.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7134:                                             ; preds = %7129
; BB742 :
  %7135 = bitcast i64 %7130 to <2 x i32>		; visa id: 9165
  %7136 = extractelement <2 x i32> %7135, i32 0		; visa id: 9167
  %7137 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7136, i32 1
  %7138 = bitcast <2 x i32> %7137 to i64		; visa id: 9167
  %7139 = ashr exact i64 %7138, 32		; visa id: 9168
  %7140 = bitcast i64 %7139 to <2 x i32>		; visa id: 9169
  %7141 = extractelement <2 x i32> %7140, i32 0		; visa id: 9173
  %7142 = extractelement <2 x i32> %7140, i32 1		; visa id: 9173
  %7143 = ashr i64 %7130, 32		; visa id: 9173
  %7144 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7141, i32 %7142, i32 %50, i32 %51)
  %7145 = extractvalue { i32, i32 } %7144, 0		; visa id: 9174
  %7146 = extractvalue { i32, i32 } %7144, 1		; visa id: 9174
  %7147 = insertelement <2 x i32> undef, i32 %7145, i32 0		; visa id: 9181
  %7148 = insertelement <2 x i32> %7147, i32 %7146, i32 1		; visa id: 9182
  %7149 = bitcast <2 x i32> %7148 to i64		; visa id: 9183
  %7150 = add nsw i64 %7149, %7143, !spirv.Decorations !649		; visa id: 9187
  %7151 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !618		; visa id: 9188
  br i1 %86, label %7157, label %7152, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9189

7152:                                             ; preds = %7134
; BB743 :
  %7153 = shl i64 %7150, 2		; visa id: 9191
  %7154 = add i64 %.in, %7153		; visa id: 9192
  %7155 = inttoptr i64 %7154 to float addrspace(4)*		; visa id: 9193
  %7156 = addrspacecast float addrspace(4)* %7155 to float addrspace(1)*		; visa id: 9193
  store float %7151, float addrspace(1)* %7156, align 4		; visa id: 9194
  br label %.preheader1.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9195

7157:                                             ; preds = %7134
; BB744 :
  %7158 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7141, i32 %7142, i32 %47, i32 %48)
  %7159 = extractvalue { i32, i32 } %7158, 0		; visa id: 9197
  %7160 = extractvalue { i32, i32 } %7158, 1		; visa id: 9197
  %7161 = insertelement <2 x i32> undef, i32 %7159, i32 0		; visa id: 9204
  %7162 = insertelement <2 x i32> %7161, i32 %7160, i32 1		; visa id: 9205
  %7163 = bitcast <2 x i32> %7162 to i64		; visa id: 9206
  %7164 = shl i64 %7163, 2		; visa id: 9210
  %7165 = add i64 %.in399, %7164		; visa id: 9211
  %7166 = shl nsw i64 %7143, 2		; visa id: 9212
  %7167 = add i64 %7165, %7166		; visa id: 9213
  %7168 = inttoptr i64 %7167 to float addrspace(4)*		; visa id: 9214
  %7169 = addrspacecast float addrspace(4)* %7168 to float addrspace(1)*		; visa id: 9214
  %7170 = load float, float addrspace(1)* %7169, align 4		; visa id: 9215
  %7171 = fmul reassoc nsz arcp contract float %7170, %4, !spirv.Decorations !618		; visa id: 9216
  %7172 = fadd reassoc nsz arcp contract float %7151, %7171, !spirv.Decorations !618		; visa id: 9217
  %7173 = shl i64 %7150, 2		; visa id: 9218
  %7174 = add i64 %.in, %7173		; visa id: 9219
  %7175 = inttoptr i64 %7174 to float addrspace(4)*		; visa id: 9220
  %7176 = addrspacecast float addrspace(4)* %7175 to float addrspace(1)*		; visa id: 9220
  store float %7172, float addrspace(1)* %7176, align 4		; visa id: 9221
  br label %.preheader1.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9222

.preheader1.4:                                    ; preds = %..preheader1.4_crit_edge, %7157, %7152
; BB745 :
  %7177 = add i32 %69, 5		; visa id: 9223
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9224
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9224
  %7178 = insertelement <2 x i32> %5984, i32 %7177, i64 1		; visa id: 9224
  store <2 x i32> %7178, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9227
  br label %._crit_edge355, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9229

._crit_edge355:                                   ; preds = %._crit_edge355.._crit_edge355_crit_edge, %.preheader1.4
; BB746 :
  %7179 = phi i32 [ 0, %.preheader1.4 ], [ %7188, %._crit_edge355.._crit_edge355_crit_edge ]
  %7180 = zext i32 %7179 to i64		; visa id: 9230
  %7181 = shl nuw nsw i64 %7180, 2		; visa id: 9231
  %7182 = add i64 %5982, %7181		; visa id: 9232
  %7183 = inttoptr i64 %7182 to i32*		; visa id: 9233
  %7184 = load i32, i32* %7183, align 4, !noalias !642		; visa id: 9233
  %7185 = add i64 %5978, %7181		; visa id: 9234
  %7186 = inttoptr i64 %7185 to i32*		; visa id: 9235
  store i32 %7184, i32* %7186, align 4, !alias.scope !642		; visa id: 9235
  %7187 = icmp eq i32 %7179, 0		; visa id: 9236
  br i1 %7187, label %._crit_edge355.._crit_edge355_crit_edge, label %7189, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9237

._crit_edge355.._crit_edge355_crit_edge:          ; preds = %._crit_edge355
; BB747 :
  %7188 = add nuw nsw i32 %7179, 1, !spirv.Decorations !631		; visa id: 9239
  br label %._crit_edge355, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9240

7189:                                             ; preds = %._crit_edge355
; BB748 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9242
  %7190 = load i64, i64* %5998, align 8		; visa id: 9242
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9243
  %7191 = icmp slt i32 %7177, %const_reg_dword1		; visa id: 9243
  %7192 = icmp slt i32 %65, %const_reg_dword
  %7193 = and i1 %7192, %7191		; visa id: 9244
  br i1 %7193, label %7194, label %.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9246

.._crit_edge70.5_crit_edge:                       ; preds = %7189
; BB:
  br label %._crit_edge70.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7194:                                             ; preds = %7189
; BB750 :
  %7195 = bitcast i64 %7190 to <2 x i32>		; visa id: 9248
  %7196 = extractelement <2 x i32> %7195, i32 0		; visa id: 9250
  %7197 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7196, i32 1
  %7198 = bitcast <2 x i32> %7197 to i64		; visa id: 9250
  %7199 = ashr exact i64 %7198, 32		; visa id: 9251
  %7200 = bitcast i64 %7199 to <2 x i32>		; visa id: 9252
  %7201 = extractelement <2 x i32> %7200, i32 0		; visa id: 9256
  %7202 = extractelement <2 x i32> %7200, i32 1		; visa id: 9256
  %7203 = ashr i64 %7190, 32		; visa id: 9256
  %7204 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7201, i32 %7202, i32 %50, i32 %51)
  %7205 = extractvalue { i32, i32 } %7204, 0		; visa id: 9257
  %7206 = extractvalue { i32, i32 } %7204, 1		; visa id: 9257
  %7207 = insertelement <2 x i32> undef, i32 %7205, i32 0		; visa id: 9264
  %7208 = insertelement <2 x i32> %7207, i32 %7206, i32 1		; visa id: 9265
  %7209 = bitcast <2 x i32> %7208 to i64		; visa id: 9266
  %7210 = add nsw i64 %7209, %7203, !spirv.Decorations !649		; visa id: 9270
  %7211 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !618		; visa id: 9271
  br i1 %86, label %7217, label %7212, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9272

7212:                                             ; preds = %7194
; BB751 :
  %7213 = shl i64 %7210, 2		; visa id: 9274
  %7214 = add i64 %.in, %7213		; visa id: 9275
  %7215 = inttoptr i64 %7214 to float addrspace(4)*		; visa id: 9276
  %7216 = addrspacecast float addrspace(4)* %7215 to float addrspace(1)*		; visa id: 9276
  store float %7211, float addrspace(1)* %7216, align 4		; visa id: 9277
  br label %._crit_edge70.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9278

7217:                                             ; preds = %7194
; BB752 :
  %7218 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7201, i32 %7202, i32 %47, i32 %48)
  %7219 = extractvalue { i32, i32 } %7218, 0		; visa id: 9280
  %7220 = extractvalue { i32, i32 } %7218, 1		; visa id: 9280
  %7221 = insertelement <2 x i32> undef, i32 %7219, i32 0		; visa id: 9287
  %7222 = insertelement <2 x i32> %7221, i32 %7220, i32 1		; visa id: 9288
  %7223 = bitcast <2 x i32> %7222 to i64		; visa id: 9289
  %7224 = shl i64 %7223, 2		; visa id: 9293
  %7225 = add i64 %.in399, %7224		; visa id: 9294
  %7226 = shl nsw i64 %7203, 2		; visa id: 9295
  %7227 = add i64 %7225, %7226		; visa id: 9296
  %7228 = inttoptr i64 %7227 to float addrspace(4)*		; visa id: 9297
  %7229 = addrspacecast float addrspace(4)* %7228 to float addrspace(1)*		; visa id: 9297
  %7230 = load float, float addrspace(1)* %7229, align 4		; visa id: 9298
  %7231 = fmul reassoc nsz arcp contract float %7230, %4, !spirv.Decorations !618		; visa id: 9299
  %7232 = fadd reassoc nsz arcp contract float %7211, %7231, !spirv.Decorations !618		; visa id: 9300
  %7233 = shl i64 %7210, 2		; visa id: 9301
  %7234 = add i64 %.in, %7233		; visa id: 9302
  %7235 = inttoptr i64 %7234 to float addrspace(4)*		; visa id: 9303
  %7236 = addrspacecast float addrspace(4)* %7235 to float addrspace(1)*		; visa id: 9303
  store float %7232, float addrspace(1)* %7236, align 4		; visa id: 9304
  br label %._crit_edge70.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9305

._crit_edge70.5:                                  ; preds = %.._crit_edge70.5_crit_edge, %7217, %7212
; BB753 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9306
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9306
  %7237 = insertelement <2 x i32> %6047, i32 %7177, i64 1		; visa id: 9306
  store <2 x i32> %7237, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9309
  br label %._crit_edge356, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9311

._crit_edge356:                                   ; preds = %._crit_edge356.._crit_edge356_crit_edge, %._crit_edge70.5
; BB754 :
  %7238 = phi i32 [ 0, %._crit_edge70.5 ], [ %7247, %._crit_edge356.._crit_edge356_crit_edge ]
  %7239 = zext i32 %7238 to i64		; visa id: 9312
  %7240 = shl nuw nsw i64 %7239, 2		; visa id: 9313
  %7241 = add i64 %5982, %7240		; visa id: 9314
  %7242 = inttoptr i64 %7241 to i32*		; visa id: 9315
  %7243 = load i32, i32* %7242, align 4, !noalias !642		; visa id: 9315
  %7244 = add i64 %5978, %7240		; visa id: 9316
  %7245 = inttoptr i64 %7244 to i32*		; visa id: 9317
  store i32 %7243, i32* %7245, align 4, !alias.scope !642		; visa id: 9317
  %7246 = icmp eq i32 %7238, 0		; visa id: 9318
  br i1 %7246, label %._crit_edge356.._crit_edge356_crit_edge, label %7248, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9319

._crit_edge356.._crit_edge356_crit_edge:          ; preds = %._crit_edge356
; BB755 :
  %7247 = add nuw nsw i32 %7238, 1, !spirv.Decorations !631		; visa id: 9321
  br label %._crit_edge356, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9322

7248:                                             ; preds = %._crit_edge356
; BB756 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9324
  %7249 = load i64, i64* %5998, align 8		; visa id: 9324
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9325
  %7250 = icmp slt i32 %6046, %const_reg_dword
  %7251 = icmp slt i32 %7177, %const_reg_dword1		; visa id: 9325
  %7252 = and i1 %7250, %7251		; visa id: 9326
  br i1 %7252, label %7253, label %.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9328

.._crit_edge70.1.5_crit_edge:                     ; preds = %7248
; BB:
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7253:                                             ; preds = %7248
; BB758 :
  %7254 = bitcast i64 %7249 to <2 x i32>		; visa id: 9330
  %7255 = extractelement <2 x i32> %7254, i32 0		; visa id: 9332
  %7256 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7255, i32 1
  %7257 = bitcast <2 x i32> %7256 to i64		; visa id: 9332
  %7258 = ashr exact i64 %7257, 32		; visa id: 9333
  %7259 = bitcast i64 %7258 to <2 x i32>		; visa id: 9334
  %7260 = extractelement <2 x i32> %7259, i32 0		; visa id: 9338
  %7261 = extractelement <2 x i32> %7259, i32 1		; visa id: 9338
  %7262 = ashr i64 %7249, 32		; visa id: 9338
  %7263 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7260, i32 %7261, i32 %50, i32 %51)
  %7264 = extractvalue { i32, i32 } %7263, 0		; visa id: 9339
  %7265 = extractvalue { i32, i32 } %7263, 1		; visa id: 9339
  %7266 = insertelement <2 x i32> undef, i32 %7264, i32 0		; visa id: 9346
  %7267 = insertelement <2 x i32> %7266, i32 %7265, i32 1		; visa id: 9347
  %7268 = bitcast <2 x i32> %7267 to i64		; visa id: 9348
  %7269 = add nsw i64 %7268, %7262, !spirv.Decorations !649		; visa id: 9352
  %7270 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !618		; visa id: 9353
  br i1 %86, label %7276, label %7271, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9354

7271:                                             ; preds = %7253
; BB759 :
  %7272 = shl i64 %7269, 2		; visa id: 9356
  %7273 = add i64 %.in, %7272		; visa id: 9357
  %7274 = inttoptr i64 %7273 to float addrspace(4)*		; visa id: 9358
  %7275 = addrspacecast float addrspace(4)* %7274 to float addrspace(1)*		; visa id: 9358
  store float %7270, float addrspace(1)* %7275, align 4		; visa id: 9359
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9360

7276:                                             ; preds = %7253
; BB760 :
  %7277 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7260, i32 %7261, i32 %47, i32 %48)
  %7278 = extractvalue { i32, i32 } %7277, 0		; visa id: 9362
  %7279 = extractvalue { i32, i32 } %7277, 1		; visa id: 9362
  %7280 = insertelement <2 x i32> undef, i32 %7278, i32 0		; visa id: 9369
  %7281 = insertelement <2 x i32> %7280, i32 %7279, i32 1		; visa id: 9370
  %7282 = bitcast <2 x i32> %7281 to i64		; visa id: 9371
  %7283 = shl i64 %7282, 2		; visa id: 9375
  %7284 = add i64 %.in399, %7283		; visa id: 9376
  %7285 = shl nsw i64 %7262, 2		; visa id: 9377
  %7286 = add i64 %7284, %7285		; visa id: 9378
  %7287 = inttoptr i64 %7286 to float addrspace(4)*		; visa id: 9379
  %7288 = addrspacecast float addrspace(4)* %7287 to float addrspace(1)*		; visa id: 9379
  %7289 = load float, float addrspace(1)* %7288, align 4		; visa id: 9380
  %7290 = fmul reassoc nsz arcp contract float %7289, %4, !spirv.Decorations !618		; visa id: 9381
  %7291 = fadd reassoc nsz arcp contract float %7270, %7290, !spirv.Decorations !618		; visa id: 9382
  %7292 = shl i64 %7269, 2		; visa id: 9383
  %7293 = add i64 %.in, %7292		; visa id: 9384
  %7294 = inttoptr i64 %7293 to float addrspace(4)*		; visa id: 9385
  %7295 = addrspacecast float addrspace(4)* %7294 to float addrspace(1)*		; visa id: 9385
  store float %7291, float addrspace(1)* %7295, align 4		; visa id: 9386
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9387

._crit_edge70.1.5:                                ; preds = %.._crit_edge70.1.5_crit_edge, %7276, %7271
; BB761 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9388
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9388
  %7296 = insertelement <2 x i32> %6108, i32 %7177, i64 1		; visa id: 9388
  store <2 x i32> %7296, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9391
  br label %._crit_edge357, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9393

._crit_edge357:                                   ; preds = %._crit_edge357.._crit_edge357_crit_edge, %._crit_edge70.1.5
; BB762 :
  %7297 = phi i32 [ 0, %._crit_edge70.1.5 ], [ %7306, %._crit_edge357.._crit_edge357_crit_edge ]
  %7298 = zext i32 %7297 to i64		; visa id: 9394
  %7299 = shl nuw nsw i64 %7298, 2		; visa id: 9395
  %7300 = add i64 %5982, %7299		; visa id: 9396
  %7301 = inttoptr i64 %7300 to i32*		; visa id: 9397
  %7302 = load i32, i32* %7301, align 4, !noalias !642		; visa id: 9397
  %7303 = add i64 %5978, %7299		; visa id: 9398
  %7304 = inttoptr i64 %7303 to i32*		; visa id: 9399
  store i32 %7302, i32* %7304, align 4, !alias.scope !642		; visa id: 9399
  %7305 = icmp eq i32 %7297, 0		; visa id: 9400
  br i1 %7305, label %._crit_edge357.._crit_edge357_crit_edge, label %7307, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9401

._crit_edge357.._crit_edge357_crit_edge:          ; preds = %._crit_edge357
; BB763 :
  %7306 = add nuw nsw i32 %7297, 1, !spirv.Decorations !631		; visa id: 9403
  br label %._crit_edge357, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9404

7307:                                             ; preds = %._crit_edge357
; BB764 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9406
  %7308 = load i64, i64* %5998, align 8		; visa id: 9406
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9407
  %7309 = icmp slt i32 %6107, %const_reg_dword
  %7310 = icmp slt i32 %7177, %const_reg_dword1		; visa id: 9407
  %7311 = and i1 %7309, %7310		; visa id: 9408
  br i1 %7311, label %7312, label %.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9410

.._crit_edge70.2.5_crit_edge:                     ; preds = %7307
; BB:
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7312:                                             ; preds = %7307
; BB766 :
  %7313 = bitcast i64 %7308 to <2 x i32>		; visa id: 9412
  %7314 = extractelement <2 x i32> %7313, i32 0		; visa id: 9414
  %7315 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7314, i32 1
  %7316 = bitcast <2 x i32> %7315 to i64		; visa id: 9414
  %7317 = ashr exact i64 %7316, 32		; visa id: 9415
  %7318 = bitcast i64 %7317 to <2 x i32>		; visa id: 9416
  %7319 = extractelement <2 x i32> %7318, i32 0		; visa id: 9420
  %7320 = extractelement <2 x i32> %7318, i32 1		; visa id: 9420
  %7321 = ashr i64 %7308, 32		; visa id: 9420
  %7322 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7319, i32 %7320, i32 %50, i32 %51)
  %7323 = extractvalue { i32, i32 } %7322, 0		; visa id: 9421
  %7324 = extractvalue { i32, i32 } %7322, 1		; visa id: 9421
  %7325 = insertelement <2 x i32> undef, i32 %7323, i32 0		; visa id: 9428
  %7326 = insertelement <2 x i32> %7325, i32 %7324, i32 1		; visa id: 9429
  %7327 = bitcast <2 x i32> %7326 to i64		; visa id: 9430
  %7328 = add nsw i64 %7327, %7321, !spirv.Decorations !649		; visa id: 9434
  %7329 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !618		; visa id: 9435
  br i1 %86, label %7335, label %7330, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9436

7330:                                             ; preds = %7312
; BB767 :
  %7331 = shl i64 %7328, 2		; visa id: 9438
  %7332 = add i64 %.in, %7331		; visa id: 9439
  %7333 = inttoptr i64 %7332 to float addrspace(4)*		; visa id: 9440
  %7334 = addrspacecast float addrspace(4)* %7333 to float addrspace(1)*		; visa id: 9440
  store float %7329, float addrspace(1)* %7334, align 4		; visa id: 9441
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9442

7335:                                             ; preds = %7312
; BB768 :
  %7336 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7319, i32 %7320, i32 %47, i32 %48)
  %7337 = extractvalue { i32, i32 } %7336, 0		; visa id: 9444
  %7338 = extractvalue { i32, i32 } %7336, 1		; visa id: 9444
  %7339 = insertelement <2 x i32> undef, i32 %7337, i32 0		; visa id: 9451
  %7340 = insertelement <2 x i32> %7339, i32 %7338, i32 1		; visa id: 9452
  %7341 = bitcast <2 x i32> %7340 to i64		; visa id: 9453
  %7342 = shl i64 %7341, 2		; visa id: 9457
  %7343 = add i64 %.in399, %7342		; visa id: 9458
  %7344 = shl nsw i64 %7321, 2		; visa id: 9459
  %7345 = add i64 %7343, %7344		; visa id: 9460
  %7346 = inttoptr i64 %7345 to float addrspace(4)*		; visa id: 9461
  %7347 = addrspacecast float addrspace(4)* %7346 to float addrspace(1)*		; visa id: 9461
  %7348 = load float, float addrspace(1)* %7347, align 4		; visa id: 9462
  %7349 = fmul reassoc nsz arcp contract float %7348, %4, !spirv.Decorations !618		; visa id: 9463
  %7350 = fadd reassoc nsz arcp contract float %7329, %7349, !spirv.Decorations !618		; visa id: 9464
  %7351 = shl i64 %7328, 2		; visa id: 9465
  %7352 = add i64 %.in, %7351		; visa id: 9466
  %7353 = inttoptr i64 %7352 to float addrspace(4)*		; visa id: 9467
  %7354 = addrspacecast float addrspace(4)* %7353 to float addrspace(1)*		; visa id: 9467
  store float %7350, float addrspace(1)* %7354, align 4		; visa id: 9468
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9469

._crit_edge70.2.5:                                ; preds = %.._crit_edge70.2.5_crit_edge, %7335, %7330
; BB769 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9470
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9470
  %7355 = insertelement <2 x i32> %6169, i32 %7177, i64 1		; visa id: 9470
  store <2 x i32> %7355, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9473
  br label %._crit_edge358, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9475

._crit_edge358:                                   ; preds = %._crit_edge358.._crit_edge358_crit_edge, %._crit_edge70.2.5
; BB770 :
  %7356 = phi i32 [ 0, %._crit_edge70.2.5 ], [ %7365, %._crit_edge358.._crit_edge358_crit_edge ]
  %7357 = zext i32 %7356 to i64		; visa id: 9476
  %7358 = shl nuw nsw i64 %7357, 2		; visa id: 9477
  %7359 = add i64 %5982, %7358		; visa id: 9478
  %7360 = inttoptr i64 %7359 to i32*		; visa id: 9479
  %7361 = load i32, i32* %7360, align 4, !noalias !642		; visa id: 9479
  %7362 = add i64 %5978, %7358		; visa id: 9480
  %7363 = inttoptr i64 %7362 to i32*		; visa id: 9481
  store i32 %7361, i32* %7363, align 4, !alias.scope !642		; visa id: 9481
  %7364 = icmp eq i32 %7356, 0		; visa id: 9482
  br i1 %7364, label %._crit_edge358.._crit_edge358_crit_edge, label %7366, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9483

._crit_edge358.._crit_edge358_crit_edge:          ; preds = %._crit_edge358
; BB771 :
  %7365 = add nuw nsw i32 %7356, 1, !spirv.Decorations !631		; visa id: 9485
  br label %._crit_edge358, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9486

7366:                                             ; preds = %._crit_edge358
; BB772 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9488
  %7367 = load i64, i64* %5998, align 8		; visa id: 9488
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9489
  %7368 = icmp slt i32 %6168, %const_reg_dword
  %7369 = icmp slt i32 %7177, %const_reg_dword1		; visa id: 9489
  %7370 = and i1 %7368, %7369		; visa id: 9490
  br i1 %7370, label %7371, label %..preheader1.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9492

..preheader1.5_crit_edge:                         ; preds = %7366
; BB:
  br label %.preheader1.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7371:                                             ; preds = %7366
; BB774 :
  %7372 = bitcast i64 %7367 to <2 x i32>		; visa id: 9494
  %7373 = extractelement <2 x i32> %7372, i32 0		; visa id: 9496
  %7374 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7373, i32 1
  %7375 = bitcast <2 x i32> %7374 to i64		; visa id: 9496
  %7376 = ashr exact i64 %7375, 32		; visa id: 9497
  %7377 = bitcast i64 %7376 to <2 x i32>		; visa id: 9498
  %7378 = extractelement <2 x i32> %7377, i32 0		; visa id: 9502
  %7379 = extractelement <2 x i32> %7377, i32 1		; visa id: 9502
  %7380 = ashr i64 %7367, 32		; visa id: 9502
  %7381 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7378, i32 %7379, i32 %50, i32 %51)
  %7382 = extractvalue { i32, i32 } %7381, 0		; visa id: 9503
  %7383 = extractvalue { i32, i32 } %7381, 1		; visa id: 9503
  %7384 = insertelement <2 x i32> undef, i32 %7382, i32 0		; visa id: 9510
  %7385 = insertelement <2 x i32> %7384, i32 %7383, i32 1		; visa id: 9511
  %7386 = bitcast <2 x i32> %7385 to i64		; visa id: 9512
  %7387 = add nsw i64 %7386, %7380, !spirv.Decorations !649		; visa id: 9516
  %7388 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !618		; visa id: 9517
  br i1 %86, label %7394, label %7389, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9518

7389:                                             ; preds = %7371
; BB775 :
  %7390 = shl i64 %7387, 2		; visa id: 9520
  %7391 = add i64 %.in, %7390		; visa id: 9521
  %7392 = inttoptr i64 %7391 to float addrspace(4)*		; visa id: 9522
  %7393 = addrspacecast float addrspace(4)* %7392 to float addrspace(1)*		; visa id: 9522
  store float %7388, float addrspace(1)* %7393, align 4		; visa id: 9523
  br label %.preheader1.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9524

7394:                                             ; preds = %7371
; BB776 :
  %7395 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7378, i32 %7379, i32 %47, i32 %48)
  %7396 = extractvalue { i32, i32 } %7395, 0		; visa id: 9526
  %7397 = extractvalue { i32, i32 } %7395, 1		; visa id: 9526
  %7398 = insertelement <2 x i32> undef, i32 %7396, i32 0		; visa id: 9533
  %7399 = insertelement <2 x i32> %7398, i32 %7397, i32 1		; visa id: 9534
  %7400 = bitcast <2 x i32> %7399 to i64		; visa id: 9535
  %7401 = shl i64 %7400, 2		; visa id: 9539
  %7402 = add i64 %.in399, %7401		; visa id: 9540
  %7403 = shl nsw i64 %7380, 2		; visa id: 9541
  %7404 = add i64 %7402, %7403		; visa id: 9542
  %7405 = inttoptr i64 %7404 to float addrspace(4)*		; visa id: 9543
  %7406 = addrspacecast float addrspace(4)* %7405 to float addrspace(1)*		; visa id: 9543
  %7407 = load float, float addrspace(1)* %7406, align 4		; visa id: 9544
  %7408 = fmul reassoc nsz arcp contract float %7407, %4, !spirv.Decorations !618		; visa id: 9545
  %7409 = fadd reassoc nsz arcp contract float %7388, %7408, !spirv.Decorations !618		; visa id: 9546
  %7410 = shl i64 %7387, 2		; visa id: 9547
  %7411 = add i64 %.in, %7410		; visa id: 9548
  %7412 = inttoptr i64 %7411 to float addrspace(4)*		; visa id: 9549
  %7413 = addrspacecast float addrspace(4)* %7412 to float addrspace(1)*		; visa id: 9549
  store float %7409, float addrspace(1)* %7413, align 4		; visa id: 9550
  br label %.preheader1.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9551

.preheader1.5:                                    ; preds = %..preheader1.5_crit_edge, %7394, %7389
; BB777 :
  %7414 = add i32 %69, 6		; visa id: 9552
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9553
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9553
  %7415 = insertelement <2 x i32> %5984, i32 %7414, i64 1		; visa id: 9553
  store <2 x i32> %7415, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9556
  br label %._crit_edge359, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9558

._crit_edge359:                                   ; preds = %._crit_edge359.._crit_edge359_crit_edge, %.preheader1.5
; BB778 :
  %7416 = phi i32 [ 0, %.preheader1.5 ], [ %7425, %._crit_edge359.._crit_edge359_crit_edge ]
  %7417 = zext i32 %7416 to i64		; visa id: 9559
  %7418 = shl nuw nsw i64 %7417, 2		; visa id: 9560
  %7419 = add i64 %5982, %7418		; visa id: 9561
  %7420 = inttoptr i64 %7419 to i32*		; visa id: 9562
  %7421 = load i32, i32* %7420, align 4, !noalias !642		; visa id: 9562
  %7422 = add i64 %5978, %7418		; visa id: 9563
  %7423 = inttoptr i64 %7422 to i32*		; visa id: 9564
  store i32 %7421, i32* %7423, align 4, !alias.scope !642		; visa id: 9564
  %7424 = icmp eq i32 %7416, 0		; visa id: 9565
  br i1 %7424, label %._crit_edge359.._crit_edge359_crit_edge, label %7426, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9566

._crit_edge359.._crit_edge359_crit_edge:          ; preds = %._crit_edge359
; BB779 :
  %7425 = add nuw nsw i32 %7416, 1, !spirv.Decorations !631		; visa id: 9568
  br label %._crit_edge359, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9569

7426:                                             ; preds = %._crit_edge359
; BB780 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9571
  %7427 = load i64, i64* %5998, align 8		; visa id: 9571
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9572
  %7428 = icmp slt i32 %7414, %const_reg_dword1		; visa id: 9572
  %7429 = icmp slt i32 %65, %const_reg_dword
  %7430 = and i1 %7429, %7428		; visa id: 9573
  br i1 %7430, label %7431, label %.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9575

.._crit_edge70.6_crit_edge:                       ; preds = %7426
; BB:
  br label %._crit_edge70.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7431:                                             ; preds = %7426
; BB782 :
  %7432 = bitcast i64 %7427 to <2 x i32>		; visa id: 9577
  %7433 = extractelement <2 x i32> %7432, i32 0		; visa id: 9579
  %7434 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7433, i32 1
  %7435 = bitcast <2 x i32> %7434 to i64		; visa id: 9579
  %7436 = ashr exact i64 %7435, 32		; visa id: 9580
  %7437 = bitcast i64 %7436 to <2 x i32>		; visa id: 9581
  %7438 = extractelement <2 x i32> %7437, i32 0		; visa id: 9585
  %7439 = extractelement <2 x i32> %7437, i32 1		; visa id: 9585
  %7440 = ashr i64 %7427, 32		; visa id: 9585
  %7441 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7438, i32 %7439, i32 %50, i32 %51)
  %7442 = extractvalue { i32, i32 } %7441, 0		; visa id: 9586
  %7443 = extractvalue { i32, i32 } %7441, 1		; visa id: 9586
  %7444 = insertelement <2 x i32> undef, i32 %7442, i32 0		; visa id: 9593
  %7445 = insertelement <2 x i32> %7444, i32 %7443, i32 1		; visa id: 9594
  %7446 = bitcast <2 x i32> %7445 to i64		; visa id: 9595
  %7447 = add nsw i64 %7446, %7440, !spirv.Decorations !649		; visa id: 9599
  %7448 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !618		; visa id: 9600
  br i1 %86, label %7454, label %7449, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9601

7449:                                             ; preds = %7431
; BB783 :
  %7450 = shl i64 %7447, 2		; visa id: 9603
  %7451 = add i64 %.in, %7450		; visa id: 9604
  %7452 = inttoptr i64 %7451 to float addrspace(4)*		; visa id: 9605
  %7453 = addrspacecast float addrspace(4)* %7452 to float addrspace(1)*		; visa id: 9605
  store float %7448, float addrspace(1)* %7453, align 4		; visa id: 9606
  br label %._crit_edge70.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9607

7454:                                             ; preds = %7431
; BB784 :
  %7455 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7438, i32 %7439, i32 %47, i32 %48)
  %7456 = extractvalue { i32, i32 } %7455, 0		; visa id: 9609
  %7457 = extractvalue { i32, i32 } %7455, 1		; visa id: 9609
  %7458 = insertelement <2 x i32> undef, i32 %7456, i32 0		; visa id: 9616
  %7459 = insertelement <2 x i32> %7458, i32 %7457, i32 1		; visa id: 9617
  %7460 = bitcast <2 x i32> %7459 to i64		; visa id: 9618
  %7461 = shl i64 %7460, 2		; visa id: 9622
  %7462 = add i64 %.in399, %7461		; visa id: 9623
  %7463 = shl nsw i64 %7440, 2		; visa id: 9624
  %7464 = add i64 %7462, %7463		; visa id: 9625
  %7465 = inttoptr i64 %7464 to float addrspace(4)*		; visa id: 9626
  %7466 = addrspacecast float addrspace(4)* %7465 to float addrspace(1)*		; visa id: 9626
  %7467 = load float, float addrspace(1)* %7466, align 4		; visa id: 9627
  %7468 = fmul reassoc nsz arcp contract float %7467, %4, !spirv.Decorations !618		; visa id: 9628
  %7469 = fadd reassoc nsz arcp contract float %7448, %7468, !spirv.Decorations !618		; visa id: 9629
  %7470 = shl i64 %7447, 2		; visa id: 9630
  %7471 = add i64 %.in, %7470		; visa id: 9631
  %7472 = inttoptr i64 %7471 to float addrspace(4)*		; visa id: 9632
  %7473 = addrspacecast float addrspace(4)* %7472 to float addrspace(1)*		; visa id: 9632
  store float %7469, float addrspace(1)* %7473, align 4		; visa id: 9633
  br label %._crit_edge70.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9634

._crit_edge70.6:                                  ; preds = %.._crit_edge70.6_crit_edge, %7454, %7449
; BB785 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9635
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9635
  %7474 = insertelement <2 x i32> %6047, i32 %7414, i64 1		; visa id: 9635
  store <2 x i32> %7474, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9638
  br label %._crit_edge360, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9640

._crit_edge360:                                   ; preds = %._crit_edge360.._crit_edge360_crit_edge, %._crit_edge70.6
; BB786 :
  %7475 = phi i32 [ 0, %._crit_edge70.6 ], [ %7484, %._crit_edge360.._crit_edge360_crit_edge ]
  %7476 = zext i32 %7475 to i64		; visa id: 9641
  %7477 = shl nuw nsw i64 %7476, 2		; visa id: 9642
  %7478 = add i64 %5982, %7477		; visa id: 9643
  %7479 = inttoptr i64 %7478 to i32*		; visa id: 9644
  %7480 = load i32, i32* %7479, align 4, !noalias !642		; visa id: 9644
  %7481 = add i64 %5978, %7477		; visa id: 9645
  %7482 = inttoptr i64 %7481 to i32*		; visa id: 9646
  store i32 %7480, i32* %7482, align 4, !alias.scope !642		; visa id: 9646
  %7483 = icmp eq i32 %7475, 0		; visa id: 9647
  br i1 %7483, label %._crit_edge360.._crit_edge360_crit_edge, label %7485, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9648

._crit_edge360.._crit_edge360_crit_edge:          ; preds = %._crit_edge360
; BB787 :
  %7484 = add nuw nsw i32 %7475, 1, !spirv.Decorations !631		; visa id: 9650
  br label %._crit_edge360, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9651

7485:                                             ; preds = %._crit_edge360
; BB788 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9653
  %7486 = load i64, i64* %5998, align 8		; visa id: 9653
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9654
  %7487 = icmp slt i32 %6046, %const_reg_dword
  %7488 = icmp slt i32 %7414, %const_reg_dword1		; visa id: 9654
  %7489 = and i1 %7487, %7488		; visa id: 9655
  br i1 %7489, label %7490, label %.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9657

.._crit_edge70.1.6_crit_edge:                     ; preds = %7485
; BB:
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7490:                                             ; preds = %7485
; BB790 :
  %7491 = bitcast i64 %7486 to <2 x i32>		; visa id: 9659
  %7492 = extractelement <2 x i32> %7491, i32 0		; visa id: 9661
  %7493 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7492, i32 1
  %7494 = bitcast <2 x i32> %7493 to i64		; visa id: 9661
  %7495 = ashr exact i64 %7494, 32		; visa id: 9662
  %7496 = bitcast i64 %7495 to <2 x i32>		; visa id: 9663
  %7497 = extractelement <2 x i32> %7496, i32 0		; visa id: 9667
  %7498 = extractelement <2 x i32> %7496, i32 1		; visa id: 9667
  %7499 = ashr i64 %7486, 32		; visa id: 9667
  %7500 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7497, i32 %7498, i32 %50, i32 %51)
  %7501 = extractvalue { i32, i32 } %7500, 0		; visa id: 9668
  %7502 = extractvalue { i32, i32 } %7500, 1		; visa id: 9668
  %7503 = insertelement <2 x i32> undef, i32 %7501, i32 0		; visa id: 9675
  %7504 = insertelement <2 x i32> %7503, i32 %7502, i32 1		; visa id: 9676
  %7505 = bitcast <2 x i32> %7504 to i64		; visa id: 9677
  %7506 = add nsw i64 %7505, %7499, !spirv.Decorations !649		; visa id: 9681
  %7507 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !618		; visa id: 9682
  br i1 %86, label %7513, label %7508, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9683

7508:                                             ; preds = %7490
; BB791 :
  %7509 = shl i64 %7506, 2		; visa id: 9685
  %7510 = add i64 %.in, %7509		; visa id: 9686
  %7511 = inttoptr i64 %7510 to float addrspace(4)*		; visa id: 9687
  %7512 = addrspacecast float addrspace(4)* %7511 to float addrspace(1)*		; visa id: 9687
  store float %7507, float addrspace(1)* %7512, align 4		; visa id: 9688
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9689

7513:                                             ; preds = %7490
; BB792 :
  %7514 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7497, i32 %7498, i32 %47, i32 %48)
  %7515 = extractvalue { i32, i32 } %7514, 0		; visa id: 9691
  %7516 = extractvalue { i32, i32 } %7514, 1		; visa id: 9691
  %7517 = insertelement <2 x i32> undef, i32 %7515, i32 0		; visa id: 9698
  %7518 = insertelement <2 x i32> %7517, i32 %7516, i32 1		; visa id: 9699
  %7519 = bitcast <2 x i32> %7518 to i64		; visa id: 9700
  %7520 = shl i64 %7519, 2		; visa id: 9704
  %7521 = add i64 %.in399, %7520		; visa id: 9705
  %7522 = shl nsw i64 %7499, 2		; visa id: 9706
  %7523 = add i64 %7521, %7522		; visa id: 9707
  %7524 = inttoptr i64 %7523 to float addrspace(4)*		; visa id: 9708
  %7525 = addrspacecast float addrspace(4)* %7524 to float addrspace(1)*		; visa id: 9708
  %7526 = load float, float addrspace(1)* %7525, align 4		; visa id: 9709
  %7527 = fmul reassoc nsz arcp contract float %7526, %4, !spirv.Decorations !618		; visa id: 9710
  %7528 = fadd reassoc nsz arcp contract float %7507, %7527, !spirv.Decorations !618		; visa id: 9711
  %7529 = shl i64 %7506, 2		; visa id: 9712
  %7530 = add i64 %.in, %7529		; visa id: 9713
  %7531 = inttoptr i64 %7530 to float addrspace(4)*		; visa id: 9714
  %7532 = addrspacecast float addrspace(4)* %7531 to float addrspace(1)*		; visa id: 9714
  store float %7528, float addrspace(1)* %7532, align 4		; visa id: 9715
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9716

._crit_edge70.1.6:                                ; preds = %.._crit_edge70.1.6_crit_edge, %7513, %7508
; BB793 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9717
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9717
  %7533 = insertelement <2 x i32> %6108, i32 %7414, i64 1		; visa id: 9717
  store <2 x i32> %7533, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9720
  br label %._crit_edge361, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9722

._crit_edge361:                                   ; preds = %._crit_edge361.._crit_edge361_crit_edge, %._crit_edge70.1.6
; BB794 :
  %7534 = phi i32 [ 0, %._crit_edge70.1.6 ], [ %7543, %._crit_edge361.._crit_edge361_crit_edge ]
  %7535 = zext i32 %7534 to i64		; visa id: 9723
  %7536 = shl nuw nsw i64 %7535, 2		; visa id: 9724
  %7537 = add i64 %5982, %7536		; visa id: 9725
  %7538 = inttoptr i64 %7537 to i32*		; visa id: 9726
  %7539 = load i32, i32* %7538, align 4, !noalias !642		; visa id: 9726
  %7540 = add i64 %5978, %7536		; visa id: 9727
  %7541 = inttoptr i64 %7540 to i32*		; visa id: 9728
  store i32 %7539, i32* %7541, align 4, !alias.scope !642		; visa id: 9728
  %7542 = icmp eq i32 %7534, 0		; visa id: 9729
  br i1 %7542, label %._crit_edge361.._crit_edge361_crit_edge, label %7544, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9730

._crit_edge361.._crit_edge361_crit_edge:          ; preds = %._crit_edge361
; BB795 :
  %7543 = add nuw nsw i32 %7534, 1, !spirv.Decorations !631		; visa id: 9732
  br label %._crit_edge361, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9733

7544:                                             ; preds = %._crit_edge361
; BB796 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9735
  %7545 = load i64, i64* %5998, align 8		; visa id: 9735
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9736
  %7546 = icmp slt i32 %6107, %const_reg_dword
  %7547 = icmp slt i32 %7414, %const_reg_dword1		; visa id: 9736
  %7548 = and i1 %7546, %7547		; visa id: 9737
  br i1 %7548, label %7549, label %.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9739

.._crit_edge70.2.6_crit_edge:                     ; preds = %7544
; BB:
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7549:                                             ; preds = %7544
; BB798 :
  %7550 = bitcast i64 %7545 to <2 x i32>		; visa id: 9741
  %7551 = extractelement <2 x i32> %7550, i32 0		; visa id: 9743
  %7552 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7551, i32 1
  %7553 = bitcast <2 x i32> %7552 to i64		; visa id: 9743
  %7554 = ashr exact i64 %7553, 32		; visa id: 9744
  %7555 = bitcast i64 %7554 to <2 x i32>		; visa id: 9745
  %7556 = extractelement <2 x i32> %7555, i32 0		; visa id: 9749
  %7557 = extractelement <2 x i32> %7555, i32 1		; visa id: 9749
  %7558 = ashr i64 %7545, 32		; visa id: 9749
  %7559 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7556, i32 %7557, i32 %50, i32 %51)
  %7560 = extractvalue { i32, i32 } %7559, 0		; visa id: 9750
  %7561 = extractvalue { i32, i32 } %7559, 1		; visa id: 9750
  %7562 = insertelement <2 x i32> undef, i32 %7560, i32 0		; visa id: 9757
  %7563 = insertelement <2 x i32> %7562, i32 %7561, i32 1		; visa id: 9758
  %7564 = bitcast <2 x i32> %7563 to i64		; visa id: 9759
  %7565 = add nsw i64 %7564, %7558, !spirv.Decorations !649		; visa id: 9763
  %7566 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !618		; visa id: 9764
  br i1 %86, label %7572, label %7567, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9765

7567:                                             ; preds = %7549
; BB799 :
  %7568 = shl i64 %7565, 2		; visa id: 9767
  %7569 = add i64 %.in, %7568		; visa id: 9768
  %7570 = inttoptr i64 %7569 to float addrspace(4)*		; visa id: 9769
  %7571 = addrspacecast float addrspace(4)* %7570 to float addrspace(1)*		; visa id: 9769
  store float %7566, float addrspace(1)* %7571, align 4		; visa id: 9770
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9771

7572:                                             ; preds = %7549
; BB800 :
  %7573 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7556, i32 %7557, i32 %47, i32 %48)
  %7574 = extractvalue { i32, i32 } %7573, 0		; visa id: 9773
  %7575 = extractvalue { i32, i32 } %7573, 1		; visa id: 9773
  %7576 = insertelement <2 x i32> undef, i32 %7574, i32 0		; visa id: 9780
  %7577 = insertelement <2 x i32> %7576, i32 %7575, i32 1		; visa id: 9781
  %7578 = bitcast <2 x i32> %7577 to i64		; visa id: 9782
  %7579 = shl i64 %7578, 2		; visa id: 9786
  %7580 = add i64 %.in399, %7579		; visa id: 9787
  %7581 = shl nsw i64 %7558, 2		; visa id: 9788
  %7582 = add i64 %7580, %7581		; visa id: 9789
  %7583 = inttoptr i64 %7582 to float addrspace(4)*		; visa id: 9790
  %7584 = addrspacecast float addrspace(4)* %7583 to float addrspace(1)*		; visa id: 9790
  %7585 = load float, float addrspace(1)* %7584, align 4		; visa id: 9791
  %7586 = fmul reassoc nsz arcp contract float %7585, %4, !spirv.Decorations !618		; visa id: 9792
  %7587 = fadd reassoc nsz arcp contract float %7566, %7586, !spirv.Decorations !618		; visa id: 9793
  %7588 = shl i64 %7565, 2		; visa id: 9794
  %7589 = add i64 %.in, %7588		; visa id: 9795
  %7590 = inttoptr i64 %7589 to float addrspace(4)*		; visa id: 9796
  %7591 = addrspacecast float addrspace(4)* %7590 to float addrspace(1)*		; visa id: 9796
  store float %7587, float addrspace(1)* %7591, align 4		; visa id: 9797
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9798

._crit_edge70.2.6:                                ; preds = %.._crit_edge70.2.6_crit_edge, %7572, %7567
; BB801 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9799
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9799
  %7592 = insertelement <2 x i32> %6169, i32 %7414, i64 1		; visa id: 9799
  store <2 x i32> %7592, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9802
  br label %._crit_edge362, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9804

._crit_edge362:                                   ; preds = %._crit_edge362.._crit_edge362_crit_edge, %._crit_edge70.2.6
; BB802 :
  %7593 = phi i32 [ 0, %._crit_edge70.2.6 ], [ %7602, %._crit_edge362.._crit_edge362_crit_edge ]
  %7594 = zext i32 %7593 to i64		; visa id: 9805
  %7595 = shl nuw nsw i64 %7594, 2		; visa id: 9806
  %7596 = add i64 %5982, %7595		; visa id: 9807
  %7597 = inttoptr i64 %7596 to i32*		; visa id: 9808
  %7598 = load i32, i32* %7597, align 4, !noalias !642		; visa id: 9808
  %7599 = add i64 %5978, %7595		; visa id: 9809
  %7600 = inttoptr i64 %7599 to i32*		; visa id: 9810
  store i32 %7598, i32* %7600, align 4, !alias.scope !642		; visa id: 9810
  %7601 = icmp eq i32 %7593, 0		; visa id: 9811
  br i1 %7601, label %._crit_edge362.._crit_edge362_crit_edge, label %7603, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9812

._crit_edge362.._crit_edge362_crit_edge:          ; preds = %._crit_edge362
; BB803 :
  %7602 = add nuw nsw i32 %7593, 1, !spirv.Decorations !631		; visa id: 9814
  br label %._crit_edge362, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9815

7603:                                             ; preds = %._crit_edge362
; BB804 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9817
  %7604 = load i64, i64* %5998, align 8		; visa id: 9817
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9818
  %7605 = icmp slt i32 %6168, %const_reg_dword
  %7606 = icmp slt i32 %7414, %const_reg_dword1		; visa id: 9818
  %7607 = and i1 %7605, %7606		; visa id: 9819
  br i1 %7607, label %7608, label %..preheader1.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9821

..preheader1.6_crit_edge:                         ; preds = %7603
; BB:
  br label %.preheader1.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7608:                                             ; preds = %7603
; BB806 :
  %7609 = bitcast i64 %7604 to <2 x i32>		; visa id: 9823
  %7610 = extractelement <2 x i32> %7609, i32 0		; visa id: 9825
  %7611 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7610, i32 1
  %7612 = bitcast <2 x i32> %7611 to i64		; visa id: 9825
  %7613 = ashr exact i64 %7612, 32		; visa id: 9826
  %7614 = bitcast i64 %7613 to <2 x i32>		; visa id: 9827
  %7615 = extractelement <2 x i32> %7614, i32 0		; visa id: 9831
  %7616 = extractelement <2 x i32> %7614, i32 1		; visa id: 9831
  %7617 = ashr i64 %7604, 32		; visa id: 9831
  %7618 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7615, i32 %7616, i32 %50, i32 %51)
  %7619 = extractvalue { i32, i32 } %7618, 0		; visa id: 9832
  %7620 = extractvalue { i32, i32 } %7618, 1		; visa id: 9832
  %7621 = insertelement <2 x i32> undef, i32 %7619, i32 0		; visa id: 9839
  %7622 = insertelement <2 x i32> %7621, i32 %7620, i32 1		; visa id: 9840
  %7623 = bitcast <2 x i32> %7622 to i64		; visa id: 9841
  %7624 = add nsw i64 %7623, %7617, !spirv.Decorations !649		; visa id: 9845
  %7625 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !618		; visa id: 9846
  br i1 %86, label %7631, label %7626, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9847

7626:                                             ; preds = %7608
; BB807 :
  %7627 = shl i64 %7624, 2		; visa id: 9849
  %7628 = add i64 %.in, %7627		; visa id: 9850
  %7629 = inttoptr i64 %7628 to float addrspace(4)*		; visa id: 9851
  %7630 = addrspacecast float addrspace(4)* %7629 to float addrspace(1)*		; visa id: 9851
  store float %7625, float addrspace(1)* %7630, align 4		; visa id: 9852
  br label %.preheader1.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9853

7631:                                             ; preds = %7608
; BB808 :
  %7632 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7615, i32 %7616, i32 %47, i32 %48)
  %7633 = extractvalue { i32, i32 } %7632, 0		; visa id: 9855
  %7634 = extractvalue { i32, i32 } %7632, 1		; visa id: 9855
  %7635 = insertelement <2 x i32> undef, i32 %7633, i32 0		; visa id: 9862
  %7636 = insertelement <2 x i32> %7635, i32 %7634, i32 1		; visa id: 9863
  %7637 = bitcast <2 x i32> %7636 to i64		; visa id: 9864
  %7638 = shl i64 %7637, 2		; visa id: 9868
  %7639 = add i64 %.in399, %7638		; visa id: 9869
  %7640 = shl nsw i64 %7617, 2		; visa id: 9870
  %7641 = add i64 %7639, %7640		; visa id: 9871
  %7642 = inttoptr i64 %7641 to float addrspace(4)*		; visa id: 9872
  %7643 = addrspacecast float addrspace(4)* %7642 to float addrspace(1)*		; visa id: 9872
  %7644 = load float, float addrspace(1)* %7643, align 4		; visa id: 9873
  %7645 = fmul reassoc nsz arcp contract float %7644, %4, !spirv.Decorations !618		; visa id: 9874
  %7646 = fadd reassoc nsz arcp contract float %7625, %7645, !spirv.Decorations !618		; visa id: 9875
  %7647 = shl i64 %7624, 2		; visa id: 9876
  %7648 = add i64 %.in, %7647		; visa id: 9877
  %7649 = inttoptr i64 %7648 to float addrspace(4)*		; visa id: 9878
  %7650 = addrspacecast float addrspace(4)* %7649 to float addrspace(1)*		; visa id: 9878
  store float %7646, float addrspace(1)* %7650, align 4		; visa id: 9879
  br label %.preheader1.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9880

.preheader1.6:                                    ; preds = %..preheader1.6_crit_edge, %7631, %7626
; BB809 :
  %7651 = add i32 %69, 7		; visa id: 9881
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9882
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9882
  %7652 = insertelement <2 x i32> %5984, i32 %7651, i64 1		; visa id: 9882
  store <2 x i32> %7652, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9885
  br label %._crit_edge363, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9887

._crit_edge363:                                   ; preds = %._crit_edge363.._crit_edge363_crit_edge, %.preheader1.6
; BB810 :
  %7653 = phi i32 [ 0, %.preheader1.6 ], [ %7662, %._crit_edge363.._crit_edge363_crit_edge ]
  %7654 = zext i32 %7653 to i64		; visa id: 9888
  %7655 = shl nuw nsw i64 %7654, 2		; visa id: 9889
  %7656 = add i64 %5982, %7655		; visa id: 9890
  %7657 = inttoptr i64 %7656 to i32*		; visa id: 9891
  %7658 = load i32, i32* %7657, align 4, !noalias !642		; visa id: 9891
  %7659 = add i64 %5978, %7655		; visa id: 9892
  %7660 = inttoptr i64 %7659 to i32*		; visa id: 9893
  store i32 %7658, i32* %7660, align 4, !alias.scope !642		; visa id: 9893
  %7661 = icmp eq i32 %7653, 0		; visa id: 9894
  br i1 %7661, label %._crit_edge363.._crit_edge363_crit_edge, label %7663, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9895

._crit_edge363.._crit_edge363_crit_edge:          ; preds = %._crit_edge363
; BB811 :
  %7662 = add nuw nsw i32 %7653, 1, !spirv.Decorations !631		; visa id: 9897
  br label %._crit_edge363, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9898

7663:                                             ; preds = %._crit_edge363
; BB812 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9900
  %7664 = load i64, i64* %5998, align 8		; visa id: 9900
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9901
  %7665 = icmp slt i32 %7651, %const_reg_dword1		; visa id: 9901
  %7666 = icmp slt i32 %65, %const_reg_dword
  %7667 = and i1 %7666, %7665		; visa id: 9902
  br i1 %7667, label %7668, label %.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9904

.._crit_edge70.7_crit_edge:                       ; preds = %7663
; BB:
  br label %._crit_edge70.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7668:                                             ; preds = %7663
; BB814 :
  %7669 = bitcast i64 %7664 to <2 x i32>		; visa id: 9906
  %7670 = extractelement <2 x i32> %7669, i32 0		; visa id: 9908
  %7671 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7670, i32 1
  %7672 = bitcast <2 x i32> %7671 to i64		; visa id: 9908
  %7673 = ashr exact i64 %7672, 32		; visa id: 9909
  %7674 = bitcast i64 %7673 to <2 x i32>		; visa id: 9910
  %7675 = extractelement <2 x i32> %7674, i32 0		; visa id: 9914
  %7676 = extractelement <2 x i32> %7674, i32 1		; visa id: 9914
  %7677 = ashr i64 %7664, 32		; visa id: 9914
  %7678 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7675, i32 %7676, i32 %50, i32 %51)
  %7679 = extractvalue { i32, i32 } %7678, 0		; visa id: 9915
  %7680 = extractvalue { i32, i32 } %7678, 1		; visa id: 9915
  %7681 = insertelement <2 x i32> undef, i32 %7679, i32 0		; visa id: 9922
  %7682 = insertelement <2 x i32> %7681, i32 %7680, i32 1		; visa id: 9923
  %7683 = bitcast <2 x i32> %7682 to i64		; visa id: 9924
  %7684 = add nsw i64 %7683, %7677, !spirv.Decorations !649		; visa id: 9928
  %7685 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !618		; visa id: 9929
  br i1 %86, label %7691, label %7686, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9930

7686:                                             ; preds = %7668
; BB815 :
  %7687 = shl i64 %7684, 2		; visa id: 9932
  %7688 = add i64 %.in, %7687		; visa id: 9933
  %7689 = inttoptr i64 %7688 to float addrspace(4)*		; visa id: 9934
  %7690 = addrspacecast float addrspace(4)* %7689 to float addrspace(1)*		; visa id: 9934
  store float %7685, float addrspace(1)* %7690, align 4		; visa id: 9935
  br label %._crit_edge70.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9936

7691:                                             ; preds = %7668
; BB816 :
  %7692 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7675, i32 %7676, i32 %47, i32 %48)
  %7693 = extractvalue { i32, i32 } %7692, 0		; visa id: 9938
  %7694 = extractvalue { i32, i32 } %7692, 1		; visa id: 9938
  %7695 = insertelement <2 x i32> undef, i32 %7693, i32 0		; visa id: 9945
  %7696 = insertelement <2 x i32> %7695, i32 %7694, i32 1		; visa id: 9946
  %7697 = bitcast <2 x i32> %7696 to i64		; visa id: 9947
  %7698 = shl i64 %7697, 2		; visa id: 9951
  %7699 = add i64 %.in399, %7698		; visa id: 9952
  %7700 = shl nsw i64 %7677, 2		; visa id: 9953
  %7701 = add i64 %7699, %7700		; visa id: 9954
  %7702 = inttoptr i64 %7701 to float addrspace(4)*		; visa id: 9955
  %7703 = addrspacecast float addrspace(4)* %7702 to float addrspace(1)*		; visa id: 9955
  %7704 = load float, float addrspace(1)* %7703, align 4		; visa id: 9956
  %7705 = fmul reassoc nsz arcp contract float %7704, %4, !spirv.Decorations !618		; visa id: 9957
  %7706 = fadd reassoc nsz arcp contract float %7685, %7705, !spirv.Decorations !618		; visa id: 9958
  %7707 = shl i64 %7684, 2		; visa id: 9959
  %7708 = add i64 %.in, %7707		; visa id: 9960
  %7709 = inttoptr i64 %7708 to float addrspace(4)*		; visa id: 9961
  %7710 = addrspacecast float addrspace(4)* %7709 to float addrspace(1)*		; visa id: 9961
  store float %7706, float addrspace(1)* %7710, align 4		; visa id: 9962
  br label %._crit_edge70.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9963

._crit_edge70.7:                                  ; preds = %.._crit_edge70.7_crit_edge, %7691, %7686
; BB817 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9964
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9964
  %7711 = insertelement <2 x i32> %6047, i32 %7651, i64 1		; visa id: 9964
  store <2 x i32> %7711, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 9967
  br label %._crit_edge364, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9969

._crit_edge364:                                   ; preds = %._crit_edge364.._crit_edge364_crit_edge, %._crit_edge70.7
; BB818 :
  %7712 = phi i32 [ 0, %._crit_edge70.7 ], [ %7721, %._crit_edge364.._crit_edge364_crit_edge ]
  %7713 = zext i32 %7712 to i64		; visa id: 9970
  %7714 = shl nuw nsw i64 %7713, 2		; visa id: 9971
  %7715 = add i64 %5982, %7714		; visa id: 9972
  %7716 = inttoptr i64 %7715 to i32*		; visa id: 9973
  %7717 = load i32, i32* %7716, align 4, !noalias !642		; visa id: 9973
  %7718 = add i64 %5978, %7714		; visa id: 9974
  %7719 = inttoptr i64 %7718 to i32*		; visa id: 9975
  store i32 %7717, i32* %7719, align 4, !alias.scope !642		; visa id: 9975
  %7720 = icmp eq i32 %7712, 0		; visa id: 9976
  br i1 %7720, label %._crit_edge364.._crit_edge364_crit_edge, label %7722, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9977

._crit_edge364.._crit_edge364_crit_edge:          ; preds = %._crit_edge364
; BB819 :
  %7721 = add nuw nsw i32 %7712, 1, !spirv.Decorations !631		; visa id: 9979
  br label %._crit_edge364, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9980

7722:                                             ; preds = %._crit_edge364
; BB820 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 9982
  %7723 = load i64, i64* %5998, align 8		; visa id: 9982
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 9983
  %7724 = icmp slt i32 %6046, %const_reg_dword
  %7725 = icmp slt i32 %7651, %const_reg_dword1		; visa id: 9983
  %7726 = and i1 %7724, %7725		; visa id: 9984
  br i1 %7726, label %7727, label %.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9986

.._crit_edge70.1.7_crit_edge:                     ; preds = %7722
; BB:
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7727:                                             ; preds = %7722
; BB822 :
  %7728 = bitcast i64 %7723 to <2 x i32>		; visa id: 9988
  %7729 = extractelement <2 x i32> %7728, i32 0		; visa id: 9990
  %7730 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7729, i32 1
  %7731 = bitcast <2 x i32> %7730 to i64		; visa id: 9990
  %7732 = ashr exact i64 %7731, 32		; visa id: 9991
  %7733 = bitcast i64 %7732 to <2 x i32>		; visa id: 9992
  %7734 = extractelement <2 x i32> %7733, i32 0		; visa id: 9996
  %7735 = extractelement <2 x i32> %7733, i32 1		; visa id: 9996
  %7736 = ashr i64 %7723, 32		; visa id: 9996
  %7737 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7734, i32 %7735, i32 %50, i32 %51)
  %7738 = extractvalue { i32, i32 } %7737, 0		; visa id: 9997
  %7739 = extractvalue { i32, i32 } %7737, 1		; visa id: 9997
  %7740 = insertelement <2 x i32> undef, i32 %7738, i32 0		; visa id: 10004
  %7741 = insertelement <2 x i32> %7740, i32 %7739, i32 1		; visa id: 10005
  %7742 = bitcast <2 x i32> %7741 to i64		; visa id: 10006
  %7743 = add nsw i64 %7742, %7736, !spirv.Decorations !649		; visa id: 10010
  %7744 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !618		; visa id: 10011
  br i1 %86, label %7750, label %7745, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10012

7745:                                             ; preds = %7727
; BB823 :
  %7746 = shl i64 %7743, 2		; visa id: 10014
  %7747 = add i64 %.in, %7746		; visa id: 10015
  %7748 = inttoptr i64 %7747 to float addrspace(4)*		; visa id: 10016
  %7749 = addrspacecast float addrspace(4)* %7748 to float addrspace(1)*		; visa id: 10016
  store float %7744, float addrspace(1)* %7749, align 4		; visa id: 10017
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10018

7750:                                             ; preds = %7727
; BB824 :
  %7751 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7734, i32 %7735, i32 %47, i32 %48)
  %7752 = extractvalue { i32, i32 } %7751, 0		; visa id: 10020
  %7753 = extractvalue { i32, i32 } %7751, 1		; visa id: 10020
  %7754 = insertelement <2 x i32> undef, i32 %7752, i32 0		; visa id: 10027
  %7755 = insertelement <2 x i32> %7754, i32 %7753, i32 1		; visa id: 10028
  %7756 = bitcast <2 x i32> %7755 to i64		; visa id: 10029
  %7757 = shl i64 %7756, 2		; visa id: 10033
  %7758 = add i64 %.in399, %7757		; visa id: 10034
  %7759 = shl nsw i64 %7736, 2		; visa id: 10035
  %7760 = add i64 %7758, %7759		; visa id: 10036
  %7761 = inttoptr i64 %7760 to float addrspace(4)*		; visa id: 10037
  %7762 = addrspacecast float addrspace(4)* %7761 to float addrspace(1)*		; visa id: 10037
  %7763 = load float, float addrspace(1)* %7762, align 4		; visa id: 10038
  %7764 = fmul reassoc nsz arcp contract float %7763, %4, !spirv.Decorations !618		; visa id: 10039
  %7765 = fadd reassoc nsz arcp contract float %7744, %7764, !spirv.Decorations !618		; visa id: 10040
  %7766 = shl i64 %7743, 2		; visa id: 10041
  %7767 = add i64 %.in, %7766		; visa id: 10042
  %7768 = inttoptr i64 %7767 to float addrspace(4)*		; visa id: 10043
  %7769 = addrspacecast float addrspace(4)* %7768 to float addrspace(1)*		; visa id: 10043
  store float %7765, float addrspace(1)* %7769, align 4		; visa id: 10044
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10045

._crit_edge70.1.7:                                ; preds = %.._crit_edge70.1.7_crit_edge, %7750, %7745
; BB825 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10046
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10046
  %7770 = insertelement <2 x i32> %6108, i32 %7651, i64 1		; visa id: 10046
  store <2 x i32> %7770, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10049
  br label %._crit_edge365, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10051

._crit_edge365:                                   ; preds = %._crit_edge365.._crit_edge365_crit_edge, %._crit_edge70.1.7
; BB826 :
  %7771 = phi i32 [ 0, %._crit_edge70.1.7 ], [ %7780, %._crit_edge365.._crit_edge365_crit_edge ]
  %7772 = zext i32 %7771 to i64		; visa id: 10052
  %7773 = shl nuw nsw i64 %7772, 2		; visa id: 10053
  %7774 = add i64 %5982, %7773		; visa id: 10054
  %7775 = inttoptr i64 %7774 to i32*		; visa id: 10055
  %7776 = load i32, i32* %7775, align 4, !noalias !642		; visa id: 10055
  %7777 = add i64 %5978, %7773		; visa id: 10056
  %7778 = inttoptr i64 %7777 to i32*		; visa id: 10057
  store i32 %7776, i32* %7778, align 4, !alias.scope !642		; visa id: 10057
  %7779 = icmp eq i32 %7771, 0		; visa id: 10058
  br i1 %7779, label %._crit_edge365.._crit_edge365_crit_edge, label %7781, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10059

._crit_edge365.._crit_edge365_crit_edge:          ; preds = %._crit_edge365
; BB827 :
  %7780 = add nuw nsw i32 %7771, 1, !spirv.Decorations !631		; visa id: 10061
  br label %._crit_edge365, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10062

7781:                                             ; preds = %._crit_edge365
; BB828 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10064
  %7782 = load i64, i64* %5998, align 8		; visa id: 10064
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10065
  %7783 = icmp slt i32 %6107, %const_reg_dword
  %7784 = icmp slt i32 %7651, %const_reg_dword1		; visa id: 10065
  %7785 = and i1 %7783, %7784		; visa id: 10066
  br i1 %7785, label %7786, label %.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10068

.._crit_edge70.2.7_crit_edge:                     ; preds = %7781
; BB:
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7786:                                             ; preds = %7781
; BB830 :
  %7787 = bitcast i64 %7782 to <2 x i32>		; visa id: 10070
  %7788 = extractelement <2 x i32> %7787, i32 0		; visa id: 10072
  %7789 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7788, i32 1
  %7790 = bitcast <2 x i32> %7789 to i64		; visa id: 10072
  %7791 = ashr exact i64 %7790, 32		; visa id: 10073
  %7792 = bitcast i64 %7791 to <2 x i32>		; visa id: 10074
  %7793 = extractelement <2 x i32> %7792, i32 0		; visa id: 10078
  %7794 = extractelement <2 x i32> %7792, i32 1		; visa id: 10078
  %7795 = ashr i64 %7782, 32		; visa id: 10078
  %7796 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7793, i32 %7794, i32 %50, i32 %51)
  %7797 = extractvalue { i32, i32 } %7796, 0		; visa id: 10079
  %7798 = extractvalue { i32, i32 } %7796, 1		; visa id: 10079
  %7799 = insertelement <2 x i32> undef, i32 %7797, i32 0		; visa id: 10086
  %7800 = insertelement <2 x i32> %7799, i32 %7798, i32 1		; visa id: 10087
  %7801 = bitcast <2 x i32> %7800 to i64		; visa id: 10088
  %7802 = add nsw i64 %7801, %7795, !spirv.Decorations !649		; visa id: 10092
  %7803 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !618		; visa id: 10093
  br i1 %86, label %7809, label %7804, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10094

7804:                                             ; preds = %7786
; BB831 :
  %7805 = shl i64 %7802, 2		; visa id: 10096
  %7806 = add i64 %.in, %7805		; visa id: 10097
  %7807 = inttoptr i64 %7806 to float addrspace(4)*		; visa id: 10098
  %7808 = addrspacecast float addrspace(4)* %7807 to float addrspace(1)*		; visa id: 10098
  store float %7803, float addrspace(1)* %7808, align 4		; visa id: 10099
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10100

7809:                                             ; preds = %7786
; BB832 :
  %7810 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7793, i32 %7794, i32 %47, i32 %48)
  %7811 = extractvalue { i32, i32 } %7810, 0		; visa id: 10102
  %7812 = extractvalue { i32, i32 } %7810, 1		; visa id: 10102
  %7813 = insertelement <2 x i32> undef, i32 %7811, i32 0		; visa id: 10109
  %7814 = insertelement <2 x i32> %7813, i32 %7812, i32 1		; visa id: 10110
  %7815 = bitcast <2 x i32> %7814 to i64		; visa id: 10111
  %7816 = shl i64 %7815, 2		; visa id: 10115
  %7817 = add i64 %.in399, %7816		; visa id: 10116
  %7818 = shl nsw i64 %7795, 2		; visa id: 10117
  %7819 = add i64 %7817, %7818		; visa id: 10118
  %7820 = inttoptr i64 %7819 to float addrspace(4)*		; visa id: 10119
  %7821 = addrspacecast float addrspace(4)* %7820 to float addrspace(1)*		; visa id: 10119
  %7822 = load float, float addrspace(1)* %7821, align 4		; visa id: 10120
  %7823 = fmul reassoc nsz arcp contract float %7822, %4, !spirv.Decorations !618		; visa id: 10121
  %7824 = fadd reassoc nsz arcp contract float %7803, %7823, !spirv.Decorations !618		; visa id: 10122
  %7825 = shl i64 %7802, 2		; visa id: 10123
  %7826 = add i64 %.in, %7825		; visa id: 10124
  %7827 = inttoptr i64 %7826 to float addrspace(4)*		; visa id: 10125
  %7828 = addrspacecast float addrspace(4)* %7827 to float addrspace(1)*		; visa id: 10125
  store float %7824, float addrspace(1)* %7828, align 4		; visa id: 10126
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10127

._crit_edge70.2.7:                                ; preds = %.._crit_edge70.2.7_crit_edge, %7809, %7804
; BB833 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10128
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10128
  %7829 = insertelement <2 x i32> %6169, i32 %7651, i64 1		; visa id: 10128
  store <2 x i32> %7829, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10131
  br label %._crit_edge366, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10133

._crit_edge366:                                   ; preds = %._crit_edge366.._crit_edge366_crit_edge, %._crit_edge70.2.7
; BB834 :
  %7830 = phi i32 [ 0, %._crit_edge70.2.7 ], [ %7839, %._crit_edge366.._crit_edge366_crit_edge ]
  %7831 = zext i32 %7830 to i64		; visa id: 10134
  %7832 = shl nuw nsw i64 %7831, 2		; visa id: 10135
  %7833 = add i64 %5982, %7832		; visa id: 10136
  %7834 = inttoptr i64 %7833 to i32*		; visa id: 10137
  %7835 = load i32, i32* %7834, align 4, !noalias !642		; visa id: 10137
  %7836 = add i64 %5978, %7832		; visa id: 10138
  %7837 = inttoptr i64 %7836 to i32*		; visa id: 10139
  store i32 %7835, i32* %7837, align 4, !alias.scope !642		; visa id: 10139
  %7838 = icmp eq i32 %7830, 0		; visa id: 10140
  br i1 %7838, label %._crit_edge366.._crit_edge366_crit_edge, label %7840, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10141

._crit_edge366.._crit_edge366_crit_edge:          ; preds = %._crit_edge366
; BB835 :
  %7839 = add nuw nsw i32 %7830, 1, !spirv.Decorations !631		; visa id: 10143
  br label %._crit_edge366, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10144

7840:                                             ; preds = %._crit_edge366
; BB836 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10146
  %7841 = load i64, i64* %5998, align 8		; visa id: 10146
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10147
  %7842 = icmp slt i32 %6168, %const_reg_dword
  %7843 = icmp slt i32 %7651, %const_reg_dword1		; visa id: 10147
  %7844 = and i1 %7842, %7843		; visa id: 10148
  br i1 %7844, label %7845, label %..preheader1.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10150

..preheader1.7_crit_edge:                         ; preds = %7840
; BB:
  br label %.preheader1.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7845:                                             ; preds = %7840
; BB838 :
  %7846 = bitcast i64 %7841 to <2 x i32>		; visa id: 10152
  %7847 = extractelement <2 x i32> %7846, i32 0		; visa id: 10154
  %7848 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7847, i32 1
  %7849 = bitcast <2 x i32> %7848 to i64		; visa id: 10154
  %7850 = ashr exact i64 %7849, 32		; visa id: 10155
  %7851 = bitcast i64 %7850 to <2 x i32>		; visa id: 10156
  %7852 = extractelement <2 x i32> %7851, i32 0		; visa id: 10160
  %7853 = extractelement <2 x i32> %7851, i32 1		; visa id: 10160
  %7854 = ashr i64 %7841, 32		; visa id: 10160
  %7855 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7852, i32 %7853, i32 %50, i32 %51)
  %7856 = extractvalue { i32, i32 } %7855, 0		; visa id: 10161
  %7857 = extractvalue { i32, i32 } %7855, 1		; visa id: 10161
  %7858 = insertelement <2 x i32> undef, i32 %7856, i32 0		; visa id: 10168
  %7859 = insertelement <2 x i32> %7858, i32 %7857, i32 1		; visa id: 10169
  %7860 = bitcast <2 x i32> %7859 to i64		; visa id: 10170
  %7861 = add nsw i64 %7860, %7854, !spirv.Decorations !649		; visa id: 10174
  %7862 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !618		; visa id: 10175
  br i1 %86, label %7868, label %7863, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10176

7863:                                             ; preds = %7845
; BB839 :
  %7864 = shl i64 %7861, 2		; visa id: 10178
  %7865 = add i64 %.in, %7864		; visa id: 10179
  %7866 = inttoptr i64 %7865 to float addrspace(4)*		; visa id: 10180
  %7867 = addrspacecast float addrspace(4)* %7866 to float addrspace(1)*		; visa id: 10180
  store float %7862, float addrspace(1)* %7867, align 4		; visa id: 10181
  br label %.preheader1.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10182

7868:                                             ; preds = %7845
; BB840 :
  %7869 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7852, i32 %7853, i32 %47, i32 %48)
  %7870 = extractvalue { i32, i32 } %7869, 0		; visa id: 10184
  %7871 = extractvalue { i32, i32 } %7869, 1		; visa id: 10184
  %7872 = insertelement <2 x i32> undef, i32 %7870, i32 0		; visa id: 10191
  %7873 = insertelement <2 x i32> %7872, i32 %7871, i32 1		; visa id: 10192
  %7874 = bitcast <2 x i32> %7873 to i64		; visa id: 10193
  %7875 = shl i64 %7874, 2		; visa id: 10197
  %7876 = add i64 %.in399, %7875		; visa id: 10198
  %7877 = shl nsw i64 %7854, 2		; visa id: 10199
  %7878 = add i64 %7876, %7877		; visa id: 10200
  %7879 = inttoptr i64 %7878 to float addrspace(4)*		; visa id: 10201
  %7880 = addrspacecast float addrspace(4)* %7879 to float addrspace(1)*		; visa id: 10201
  %7881 = load float, float addrspace(1)* %7880, align 4		; visa id: 10202
  %7882 = fmul reassoc nsz arcp contract float %7881, %4, !spirv.Decorations !618		; visa id: 10203
  %7883 = fadd reassoc nsz arcp contract float %7862, %7882, !spirv.Decorations !618		; visa id: 10204
  %7884 = shl i64 %7861, 2		; visa id: 10205
  %7885 = add i64 %.in, %7884		; visa id: 10206
  %7886 = inttoptr i64 %7885 to float addrspace(4)*		; visa id: 10207
  %7887 = addrspacecast float addrspace(4)* %7886 to float addrspace(1)*		; visa id: 10207
  store float %7883, float addrspace(1)* %7887, align 4		; visa id: 10208
  br label %.preheader1.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10209

.preheader1.7:                                    ; preds = %..preheader1.7_crit_edge, %7868, %7863
; BB841 :
  %7888 = add i32 %69, 8		; visa id: 10210
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10211
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10211
  %7889 = insertelement <2 x i32> %5984, i32 %7888, i64 1		; visa id: 10211
  store <2 x i32> %7889, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10214
  br label %._crit_edge367, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10216

._crit_edge367:                                   ; preds = %._crit_edge367.._crit_edge367_crit_edge, %.preheader1.7
; BB842 :
  %7890 = phi i32 [ 0, %.preheader1.7 ], [ %7899, %._crit_edge367.._crit_edge367_crit_edge ]
  %7891 = zext i32 %7890 to i64		; visa id: 10217
  %7892 = shl nuw nsw i64 %7891, 2		; visa id: 10218
  %7893 = add i64 %5982, %7892		; visa id: 10219
  %7894 = inttoptr i64 %7893 to i32*		; visa id: 10220
  %7895 = load i32, i32* %7894, align 4, !noalias !642		; visa id: 10220
  %7896 = add i64 %5978, %7892		; visa id: 10221
  %7897 = inttoptr i64 %7896 to i32*		; visa id: 10222
  store i32 %7895, i32* %7897, align 4, !alias.scope !642		; visa id: 10222
  %7898 = icmp eq i32 %7890, 0		; visa id: 10223
  br i1 %7898, label %._crit_edge367.._crit_edge367_crit_edge, label %7900, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10224

._crit_edge367.._crit_edge367_crit_edge:          ; preds = %._crit_edge367
; BB843 :
  %7899 = add nuw nsw i32 %7890, 1, !spirv.Decorations !631		; visa id: 10226
  br label %._crit_edge367, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10227

7900:                                             ; preds = %._crit_edge367
; BB844 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10229
  %7901 = load i64, i64* %5998, align 8		; visa id: 10229
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10230
  %7902 = icmp slt i32 %7888, %const_reg_dword1		; visa id: 10230
  %7903 = icmp slt i32 %65, %const_reg_dword
  %7904 = and i1 %7903, %7902		; visa id: 10231
  br i1 %7904, label %7905, label %.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10233

.._crit_edge70.8_crit_edge:                       ; preds = %7900
; BB:
  br label %._crit_edge70.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7905:                                             ; preds = %7900
; BB846 :
  %7906 = bitcast i64 %7901 to <2 x i32>		; visa id: 10235
  %7907 = extractelement <2 x i32> %7906, i32 0		; visa id: 10237
  %7908 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7907, i32 1
  %7909 = bitcast <2 x i32> %7908 to i64		; visa id: 10237
  %7910 = ashr exact i64 %7909, 32		; visa id: 10238
  %7911 = bitcast i64 %7910 to <2 x i32>		; visa id: 10239
  %7912 = extractelement <2 x i32> %7911, i32 0		; visa id: 10243
  %7913 = extractelement <2 x i32> %7911, i32 1		; visa id: 10243
  %7914 = ashr i64 %7901, 32		; visa id: 10243
  %7915 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7912, i32 %7913, i32 %50, i32 %51)
  %7916 = extractvalue { i32, i32 } %7915, 0		; visa id: 10244
  %7917 = extractvalue { i32, i32 } %7915, 1		; visa id: 10244
  %7918 = insertelement <2 x i32> undef, i32 %7916, i32 0		; visa id: 10251
  %7919 = insertelement <2 x i32> %7918, i32 %7917, i32 1		; visa id: 10252
  %7920 = bitcast <2 x i32> %7919 to i64		; visa id: 10253
  %7921 = add nsw i64 %7920, %7914, !spirv.Decorations !649		; visa id: 10257
  %7922 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !618		; visa id: 10258
  br i1 %86, label %7928, label %7923, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10259

7923:                                             ; preds = %7905
; BB847 :
  %7924 = shl i64 %7921, 2		; visa id: 10261
  %7925 = add i64 %.in, %7924		; visa id: 10262
  %7926 = inttoptr i64 %7925 to float addrspace(4)*		; visa id: 10263
  %7927 = addrspacecast float addrspace(4)* %7926 to float addrspace(1)*		; visa id: 10263
  store float %7922, float addrspace(1)* %7927, align 4		; visa id: 10264
  br label %._crit_edge70.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10265

7928:                                             ; preds = %7905
; BB848 :
  %7929 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7912, i32 %7913, i32 %47, i32 %48)
  %7930 = extractvalue { i32, i32 } %7929, 0		; visa id: 10267
  %7931 = extractvalue { i32, i32 } %7929, 1		; visa id: 10267
  %7932 = insertelement <2 x i32> undef, i32 %7930, i32 0		; visa id: 10274
  %7933 = insertelement <2 x i32> %7932, i32 %7931, i32 1		; visa id: 10275
  %7934 = bitcast <2 x i32> %7933 to i64		; visa id: 10276
  %7935 = shl i64 %7934, 2		; visa id: 10280
  %7936 = add i64 %.in399, %7935		; visa id: 10281
  %7937 = shl nsw i64 %7914, 2		; visa id: 10282
  %7938 = add i64 %7936, %7937		; visa id: 10283
  %7939 = inttoptr i64 %7938 to float addrspace(4)*		; visa id: 10284
  %7940 = addrspacecast float addrspace(4)* %7939 to float addrspace(1)*		; visa id: 10284
  %7941 = load float, float addrspace(1)* %7940, align 4		; visa id: 10285
  %7942 = fmul reassoc nsz arcp contract float %7941, %4, !spirv.Decorations !618		; visa id: 10286
  %7943 = fadd reassoc nsz arcp contract float %7922, %7942, !spirv.Decorations !618		; visa id: 10287
  %7944 = shl i64 %7921, 2		; visa id: 10288
  %7945 = add i64 %.in, %7944		; visa id: 10289
  %7946 = inttoptr i64 %7945 to float addrspace(4)*		; visa id: 10290
  %7947 = addrspacecast float addrspace(4)* %7946 to float addrspace(1)*		; visa id: 10290
  store float %7943, float addrspace(1)* %7947, align 4		; visa id: 10291
  br label %._crit_edge70.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10292

._crit_edge70.8:                                  ; preds = %.._crit_edge70.8_crit_edge, %7928, %7923
; BB849 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10293
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10293
  %7948 = insertelement <2 x i32> %6047, i32 %7888, i64 1		; visa id: 10293
  store <2 x i32> %7948, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10296
  br label %._crit_edge368, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10298

._crit_edge368:                                   ; preds = %._crit_edge368.._crit_edge368_crit_edge, %._crit_edge70.8
; BB850 :
  %7949 = phi i32 [ 0, %._crit_edge70.8 ], [ %7958, %._crit_edge368.._crit_edge368_crit_edge ]
  %7950 = zext i32 %7949 to i64		; visa id: 10299
  %7951 = shl nuw nsw i64 %7950, 2		; visa id: 10300
  %7952 = add i64 %5982, %7951		; visa id: 10301
  %7953 = inttoptr i64 %7952 to i32*		; visa id: 10302
  %7954 = load i32, i32* %7953, align 4, !noalias !642		; visa id: 10302
  %7955 = add i64 %5978, %7951		; visa id: 10303
  %7956 = inttoptr i64 %7955 to i32*		; visa id: 10304
  store i32 %7954, i32* %7956, align 4, !alias.scope !642		; visa id: 10304
  %7957 = icmp eq i32 %7949, 0		; visa id: 10305
  br i1 %7957, label %._crit_edge368.._crit_edge368_crit_edge, label %7959, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10306

._crit_edge368.._crit_edge368_crit_edge:          ; preds = %._crit_edge368
; BB851 :
  %7958 = add nuw nsw i32 %7949, 1, !spirv.Decorations !631		; visa id: 10308
  br label %._crit_edge368, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10309

7959:                                             ; preds = %._crit_edge368
; BB852 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10311
  %7960 = load i64, i64* %5998, align 8		; visa id: 10311
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10312
  %7961 = icmp slt i32 %6046, %const_reg_dword
  %7962 = icmp slt i32 %7888, %const_reg_dword1		; visa id: 10312
  %7963 = and i1 %7961, %7962		; visa id: 10313
  br i1 %7963, label %7964, label %.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10315

.._crit_edge70.1.8_crit_edge:                     ; preds = %7959
; BB:
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7964:                                             ; preds = %7959
; BB854 :
  %7965 = bitcast i64 %7960 to <2 x i32>		; visa id: 10317
  %7966 = extractelement <2 x i32> %7965, i32 0		; visa id: 10319
  %7967 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7966, i32 1
  %7968 = bitcast <2 x i32> %7967 to i64		; visa id: 10319
  %7969 = ashr exact i64 %7968, 32		; visa id: 10320
  %7970 = bitcast i64 %7969 to <2 x i32>		; visa id: 10321
  %7971 = extractelement <2 x i32> %7970, i32 0		; visa id: 10325
  %7972 = extractelement <2 x i32> %7970, i32 1		; visa id: 10325
  %7973 = ashr i64 %7960, 32		; visa id: 10325
  %7974 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7971, i32 %7972, i32 %50, i32 %51)
  %7975 = extractvalue { i32, i32 } %7974, 0		; visa id: 10326
  %7976 = extractvalue { i32, i32 } %7974, 1		; visa id: 10326
  %7977 = insertelement <2 x i32> undef, i32 %7975, i32 0		; visa id: 10333
  %7978 = insertelement <2 x i32> %7977, i32 %7976, i32 1		; visa id: 10334
  %7979 = bitcast <2 x i32> %7978 to i64		; visa id: 10335
  %7980 = add nsw i64 %7979, %7973, !spirv.Decorations !649		; visa id: 10339
  %7981 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !618		; visa id: 10340
  br i1 %86, label %7987, label %7982, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10341

7982:                                             ; preds = %7964
; BB855 :
  %7983 = shl i64 %7980, 2		; visa id: 10343
  %7984 = add i64 %.in, %7983		; visa id: 10344
  %7985 = inttoptr i64 %7984 to float addrspace(4)*		; visa id: 10345
  %7986 = addrspacecast float addrspace(4)* %7985 to float addrspace(1)*		; visa id: 10345
  store float %7981, float addrspace(1)* %7986, align 4		; visa id: 10346
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10347

7987:                                             ; preds = %7964
; BB856 :
  %7988 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7971, i32 %7972, i32 %47, i32 %48)
  %7989 = extractvalue { i32, i32 } %7988, 0		; visa id: 10349
  %7990 = extractvalue { i32, i32 } %7988, 1		; visa id: 10349
  %7991 = insertelement <2 x i32> undef, i32 %7989, i32 0		; visa id: 10356
  %7992 = insertelement <2 x i32> %7991, i32 %7990, i32 1		; visa id: 10357
  %7993 = bitcast <2 x i32> %7992 to i64		; visa id: 10358
  %7994 = shl i64 %7993, 2		; visa id: 10362
  %7995 = add i64 %.in399, %7994		; visa id: 10363
  %7996 = shl nsw i64 %7973, 2		; visa id: 10364
  %7997 = add i64 %7995, %7996		; visa id: 10365
  %7998 = inttoptr i64 %7997 to float addrspace(4)*		; visa id: 10366
  %7999 = addrspacecast float addrspace(4)* %7998 to float addrspace(1)*		; visa id: 10366
  %8000 = load float, float addrspace(1)* %7999, align 4		; visa id: 10367
  %8001 = fmul reassoc nsz arcp contract float %8000, %4, !spirv.Decorations !618		; visa id: 10368
  %8002 = fadd reassoc nsz arcp contract float %7981, %8001, !spirv.Decorations !618		; visa id: 10369
  %8003 = shl i64 %7980, 2		; visa id: 10370
  %8004 = add i64 %.in, %8003		; visa id: 10371
  %8005 = inttoptr i64 %8004 to float addrspace(4)*		; visa id: 10372
  %8006 = addrspacecast float addrspace(4)* %8005 to float addrspace(1)*		; visa id: 10372
  store float %8002, float addrspace(1)* %8006, align 4		; visa id: 10373
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10374

._crit_edge70.1.8:                                ; preds = %.._crit_edge70.1.8_crit_edge, %7987, %7982
; BB857 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10375
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10375
  %8007 = insertelement <2 x i32> %6108, i32 %7888, i64 1		; visa id: 10375
  store <2 x i32> %8007, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10378
  br label %._crit_edge369, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10380

._crit_edge369:                                   ; preds = %._crit_edge369.._crit_edge369_crit_edge, %._crit_edge70.1.8
; BB858 :
  %8008 = phi i32 [ 0, %._crit_edge70.1.8 ], [ %8017, %._crit_edge369.._crit_edge369_crit_edge ]
  %8009 = zext i32 %8008 to i64		; visa id: 10381
  %8010 = shl nuw nsw i64 %8009, 2		; visa id: 10382
  %8011 = add i64 %5982, %8010		; visa id: 10383
  %8012 = inttoptr i64 %8011 to i32*		; visa id: 10384
  %8013 = load i32, i32* %8012, align 4, !noalias !642		; visa id: 10384
  %8014 = add i64 %5978, %8010		; visa id: 10385
  %8015 = inttoptr i64 %8014 to i32*		; visa id: 10386
  store i32 %8013, i32* %8015, align 4, !alias.scope !642		; visa id: 10386
  %8016 = icmp eq i32 %8008, 0		; visa id: 10387
  br i1 %8016, label %._crit_edge369.._crit_edge369_crit_edge, label %8018, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10388

._crit_edge369.._crit_edge369_crit_edge:          ; preds = %._crit_edge369
; BB859 :
  %8017 = add nuw nsw i32 %8008, 1, !spirv.Decorations !631		; visa id: 10390
  br label %._crit_edge369, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10391

8018:                                             ; preds = %._crit_edge369
; BB860 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10393
  %8019 = load i64, i64* %5998, align 8		; visa id: 10393
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10394
  %8020 = icmp slt i32 %6107, %const_reg_dword
  %8021 = icmp slt i32 %7888, %const_reg_dword1		; visa id: 10394
  %8022 = and i1 %8020, %8021		; visa id: 10395
  br i1 %8022, label %8023, label %.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10397

.._crit_edge70.2.8_crit_edge:                     ; preds = %8018
; BB:
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8023:                                             ; preds = %8018
; BB862 :
  %8024 = bitcast i64 %8019 to <2 x i32>		; visa id: 10399
  %8025 = extractelement <2 x i32> %8024, i32 0		; visa id: 10401
  %8026 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8025, i32 1
  %8027 = bitcast <2 x i32> %8026 to i64		; visa id: 10401
  %8028 = ashr exact i64 %8027, 32		; visa id: 10402
  %8029 = bitcast i64 %8028 to <2 x i32>		; visa id: 10403
  %8030 = extractelement <2 x i32> %8029, i32 0		; visa id: 10407
  %8031 = extractelement <2 x i32> %8029, i32 1		; visa id: 10407
  %8032 = ashr i64 %8019, 32		; visa id: 10407
  %8033 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8030, i32 %8031, i32 %50, i32 %51)
  %8034 = extractvalue { i32, i32 } %8033, 0		; visa id: 10408
  %8035 = extractvalue { i32, i32 } %8033, 1		; visa id: 10408
  %8036 = insertelement <2 x i32> undef, i32 %8034, i32 0		; visa id: 10415
  %8037 = insertelement <2 x i32> %8036, i32 %8035, i32 1		; visa id: 10416
  %8038 = bitcast <2 x i32> %8037 to i64		; visa id: 10417
  %8039 = add nsw i64 %8038, %8032, !spirv.Decorations !649		; visa id: 10421
  %8040 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !618		; visa id: 10422
  br i1 %86, label %8046, label %8041, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10423

8041:                                             ; preds = %8023
; BB863 :
  %8042 = shl i64 %8039, 2		; visa id: 10425
  %8043 = add i64 %.in, %8042		; visa id: 10426
  %8044 = inttoptr i64 %8043 to float addrspace(4)*		; visa id: 10427
  %8045 = addrspacecast float addrspace(4)* %8044 to float addrspace(1)*		; visa id: 10427
  store float %8040, float addrspace(1)* %8045, align 4		; visa id: 10428
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10429

8046:                                             ; preds = %8023
; BB864 :
  %8047 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8030, i32 %8031, i32 %47, i32 %48)
  %8048 = extractvalue { i32, i32 } %8047, 0		; visa id: 10431
  %8049 = extractvalue { i32, i32 } %8047, 1		; visa id: 10431
  %8050 = insertelement <2 x i32> undef, i32 %8048, i32 0		; visa id: 10438
  %8051 = insertelement <2 x i32> %8050, i32 %8049, i32 1		; visa id: 10439
  %8052 = bitcast <2 x i32> %8051 to i64		; visa id: 10440
  %8053 = shl i64 %8052, 2		; visa id: 10444
  %8054 = add i64 %.in399, %8053		; visa id: 10445
  %8055 = shl nsw i64 %8032, 2		; visa id: 10446
  %8056 = add i64 %8054, %8055		; visa id: 10447
  %8057 = inttoptr i64 %8056 to float addrspace(4)*		; visa id: 10448
  %8058 = addrspacecast float addrspace(4)* %8057 to float addrspace(1)*		; visa id: 10448
  %8059 = load float, float addrspace(1)* %8058, align 4		; visa id: 10449
  %8060 = fmul reassoc nsz arcp contract float %8059, %4, !spirv.Decorations !618		; visa id: 10450
  %8061 = fadd reassoc nsz arcp contract float %8040, %8060, !spirv.Decorations !618		; visa id: 10451
  %8062 = shl i64 %8039, 2		; visa id: 10452
  %8063 = add i64 %.in, %8062		; visa id: 10453
  %8064 = inttoptr i64 %8063 to float addrspace(4)*		; visa id: 10454
  %8065 = addrspacecast float addrspace(4)* %8064 to float addrspace(1)*		; visa id: 10454
  store float %8061, float addrspace(1)* %8065, align 4		; visa id: 10455
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10456

._crit_edge70.2.8:                                ; preds = %.._crit_edge70.2.8_crit_edge, %8046, %8041
; BB865 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10457
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10457
  %8066 = insertelement <2 x i32> %6169, i32 %7888, i64 1		; visa id: 10457
  store <2 x i32> %8066, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10460
  br label %._crit_edge370, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10462

._crit_edge370:                                   ; preds = %._crit_edge370.._crit_edge370_crit_edge, %._crit_edge70.2.8
; BB866 :
  %8067 = phi i32 [ 0, %._crit_edge70.2.8 ], [ %8076, %._crit_edge370.._crit_edge370_crit_edge ]
  %8068 = zext i32 %8067 to i64		; visa id: 10463
  %8069 = shl nuw nsw i64 %8068, 2		; visa id: 10464
  %8070 = add i64 %5982, %8069		; visa id: 10465
  %8071 = inttoptr i64 %8070 to i32*		; visa id: 10466
  %8072 = load i32, i32* %8071, align 4, !noalias !642		; visa id: 10466
  %8073 = add i64 %5978, %8069		; visa id: 10467
  %8074 = inttoptr i64 %8073 to i32*		; visa id: 10468
  store i32 %8072, i32* %8074, align 4, !alias.scope !642		; visa id: 10468
  %8075 = icmp eq i32 %8067, 0		; visa id: 10469
  br i1 %8075, label %._crit_edge370.._crit_edge370_crit_edge, label %8077, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10470

._crit_edge370.._crit_edge370_crit_edge:          ; preds = %._crit_edge370
; BB867 :
  %8076 = add nuw nsw i32 %8067, 1, !spirv.Decorations !631		; visa id: 10472
  br label %._crit_edge370, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10473

8077:                                             ; preds = %._crit_edge370
; BB868 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10475
  %8078 = load i64, i64* %5998, align 8		; visa id: 10475
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10476
  %8079 = icmp slt i32 %6168, %const_reg_dword
  %8080 = icmp slt i32 %7888, %const_reg_dword1		; visa id: 10476
  %8081 = and i1 %8079, %8080		; visa id: 10477
  br i1 %8081, label %8082, label %..preheader1.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10479

..preheader1.8_crit_edge:                         ; preds = %8077
; BB:
  br label %.preheader1.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8082:                                             ; preds = %8077
; BB870 :
  %8083 = bitcast i64 %8078 to <2 x i32>		; visa id: 10481
  %8084 = extractelement <2 x i32> %8083, i32 0		; visa id: 10483
  %8085 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8084, i32 1
  %8086 = bitcast <2 x i32> %8085 to i64		; visa id: 10483
  %8087 = ashr exact i64 %8086, 32		; visa id: 10484
  %8088 = bitcast i64 %8087 to <2 x i32>		; visa id: 10485
  %8089 = extractelement <2 x i32> %8088, i32 0		; visa id: 10489
  %8090 = extractelement <2 x i32> %8088, i32 1		; visa id: 10489
  %8091 = ashr i64 %8078, 32		; visa id: 10489
  %8092 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8089, i32 %8090, i32 %50, i32 %51)
  %8093 = extractvalue { i32, i32 } %8092, 0		; visa id: 10490
  %8094 = extractvalue { i32, i32 } %8092, 1		; visa id: 10490
  %8095 = insertelement <2 x i32> undef, i32 %8093, i32 0		; visa id: 10497
  %8096 = insertelement <2 x i32> %8095, i32 %8094, i32 1		; visa id: 10498
  %8097 = bitcast <2 x i32> %8096 to i64		; visa id: 10499
  %8098 = add nsw i64 %8097, %8091, !spirv.Decorations !649		; visa id: 10503
  %8099 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !618		; visa id: 10504
  br i1 %86, label %8105, label %8100, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10505

8100:                                             ; preds = %8082
; BB871 :
  %8101 = shl i64 %8098, 2		; visa id: 10507
  %8102 = add i64 %.in, %8101		; visa id: 10508
  %8103 = inttoptr i64 %8102 to float addrspace(4)*		; visa id: 10509
  %8104 = addrspacecast float addrspace(4)* %8103 to float addrspace(1)*		; visa id: 10509
  store float %8099, float addrspace(1)* %8104, align 4		; visa id: 10510
  br label %.preheader1.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10511

8105:                                             ; preds = %8082
; BB872 :
  %8106 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8089, i32 %8090, i32 %47, i32 %48)
  %8107 = extractvalue { i32, i32 } %8106, 0		; visa id: 10513
  %8108 = extractvalue { i32, i32 } %8106, 1		; visa id: 10513
  %8109 = insertelement <2 x i32> undef, i32 %8107, i32 0		; visa id: 10520
  %8110 = insertelement <2 x i32> %8109, i32 %8108, i32 1		; visa id: 10521
  %8111 = bitcast <2 x i32> %8110 to i64		; visa id: 10522
  %8112 = shl i64 %8111, 2		; visa id: 10526
  %8113 = add i64 %.in399, %8112		; visa id: 10527
  %8114 = shl nsw i64 %8091, 2		; visa id: 10528
  %8115 = add i64 %8113, %8114		; visa id: 10529
  %8116 = inttoptr i64 %8115 to float addrspace(4)*		; visa id: 10530
  %8117 = addrspacecast float addrspace(4)* %8116 to float addrspace(1)*		; visa id: 10530
  %8118 = load float, float addrspace(1)* %8117, align 4		; visa id: 10531
  %8119 = fmul reassoc nsz arcp contract float %8118, %4, !spirv.Decorations !618		; visa id: 10532
  %8120 = fadd reassoc nsz arcp contract float %8099, %8119, !spirv.Decorations !618		; visa id: 10533
  %8121 = shl i64 %8098, 2		; visa id: 10534
  %8122 = add i64 %.in, %8121		; visa id: 10535
  %8123 = inttoptr i64 %8122 to float addrspace(4)*		; visa id: 10536
  %8124 = addrspacecast float addrspace(4)* %8123 to float addrspace(1)*		; visa id: 10536
  store float %8120, float addrspace(1)* %8124, align 4		; visa id: 10537
  br label %.preheader1.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10538

.preheader1.8:                                    ; preds = %..preheader1.8_crit_edge, %8105, %8100
; BB873 :
  %8125 = add i32 %69, 9		; visa id: 10539
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10540
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10540
  %8126 = insertelement <2 x i32> %5984, i32 %8125, i64 1		; visa id: 10540
  store <2 x i32> %8126, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10543
  br label %._crit_edge371, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10545

._crit_edge371:                                   ; preds = %._crit_edge371.._crit_edge371_crit_edge, %.preheader1.8
; BB874 :
  %8127 = phi i32 [ 0, %.preheader1.8 ], [ %8136, %._crit_edge371.._crit_edge371_crit_edge ]
  %8128 = zext i32 %8127 to i64		; visa id: 10546
  %8129 = shl nuw nsw i64 %8128, 2		; visa id: 10547
  %8130 = add i64 %5982, %8129		; visa id: 10548
  %8131 = inttoptr i64 %8130 to i32*		; visa id: 10549
  %8132 = load i32, i32* %8131, align 4, !noalias !642		; visa id: 10549
  %8133 = add i64 %5978, %8129		; visa id: 10550
  %8134 = inttoptr i64 %8133 to i32*		; visa id: 10551
  store i32 %8132, i32* %8134, align 4, !alias.scope !642		; visa id: 10551
  %8135 = icmp eq i32 %8127, 0		; visa id: 10552
  br i1 %8135, label %._crit_edge371.._crit_edge371_crit_edge, label %8137, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10553

._crit_edge371.._crit_edge371_crit_edge:          ; preds = %._crit_edge371
; BB875 :
  %8136 = add nuw nsw i32 %8127, 1, !spirv.Decorations !631		; visa id: 10555
  br label %._crit_edge371, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10556

8137:                                             ; preds = %._crit_edge371
; BB876 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10558
  %8138 = load i64, i64* %5998, align 8		; visa id: 10558
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10559
  %8139 = icmp slt i32 %8125, %const_reg_dword1		; visa id: 10559
  %8140 = icmp slt i32 %65, %const_reg_dword
  %8141 = and i1 %8140, %8139		; visa id: 10560
  br i1 %8141, label %8142, label %.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10562

.._crit_edge70.9_crit_edge:                       ; preds = %8137
; BB:
  br label %._crit_edge70.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8142:                                             ; preds = %8137
; BB878 :
  %8143 = bitcast i64 %8138 to <2 x i32>		; visa id: 10564
  %8144 = extractelement <2 x i32> %8143, i32 0		; visa id: 10566
  %8145 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8144, i32 1
  %8146 = bitcast <2 x i32> %8145 to i64		; visa id: 10566
  %8147 = ashr exact i64 %8146, 32		; visa id: 10567
  %8148 = bitcast i64 %8147 to <2 x i32>		; visa id: 10568
  %8149 = extractelement <2 x i32> %8148, i32 0		; visa id: 10572
  %8150 = extractelement <2 x i32> %8148, i32 1		; visa id: 10572
  %8151 = ashr i64 %8138, 32		; visa id: 10572
  %8152 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8149, i32 %8150, i32 %50, i32 %51)
  %8153 = extractvalue { i32, i32 } %8152, 0		; visa id: 10573
  %8154 = extractvalue { i32, i32 } %8152, 1		; visa id: 10573
  %8155 = insertelement <2 x i32> undef, i32 %8153, i32 0		; visa id: 10580
  %8156 = insertelement <2 x i32> %8155, i32 %8154, i32 1		; visa id: 10581
  %8157 = bitcast <2 x i32> %8156 to i64		; visa id: 10582
  %8158 = add nsw i64 %8157, %8151, !spirv.Decorations !649		; visa id: 10586
  %8159 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !618		; visa id: 10587
  br i1 %86, label %8165, label %8160, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10588

8160:                                             ; preds = %8142
; BB879 :
  %8161 = shl i64 %8158, 2		; visa id: 10590
  %8162 = add i64 %.in, %8161		; visa id: 10591
  %8163 = inttoptr i64 %8162 to float addrspace(4)*		; visa id: 10592
  %8164 = addrspacecast float addrspace(4)* %8163 to float addrspace(1)*		; visa id: 10592
  store float %8159, float addrspace(1)* %8164, align 4		; visa id: 10593
  br label %._crit_edge70.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10594

8165:                                             ; preds = %8142
; BB880 :
  %8166 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8149, i32 %8150, i32 %47, i32 %48)
  %8167 = extractvalue { i32, i32 } %8166, 0		; visa id: 10596
  %8168 = extractvalue { i32, i32 } %8166, 1		; visa id: 10596
  %8169 = insertelement <2 x i32> undef, i32 %8167, i32 0		; visa id: 10603
  %8170 = insertelement <2 x i32> %8169, i32 %8168, i32 1		; visa id: 10604
  %8171 = bitcast <2 x i32> %8170 to i64		; visa id: 10605
  %8172 = shl i64 %8171, 2		; visa id: 10609
  %8173 = add i64 %.in399, %8172		; visa id: 10610
  %8174 = shl nsw i64 %8151, 2		; visa id: 10611
  %8175 = add i64 %8173, %8174		; visa id: 10612
  %8176 = inttoptr i64 %8175 to float addrspace(4)*		; visa id: 10613
  %8177 = addrspacecast float addrspace(4)* %8176 to float addrspace(1)*		; visa id: 10613
  %8178 = load float, float addrspace(1)* %8177, align 4		; visa id: 10614
  %8179 = fmul reassoc nsz arcp contract float %8178, %4, !spirv.Decorations !618		; visa id: 10615
  %8180 = fadd reassoc nsz arcp contract float %8159, %8179, !spirv.Decorations !618		; visa id: 10616
  %8181 = shl i64 %8158, 2		; visa id: 10617
  %8182 = add i64 %.in, %8181		; visa id: 10618
  %8183 = inttoptr i64 %8182 to float addrspace(4)*		; visa id: 10619
  %8184 = addrspacecast float addrspace(4)* %8183 to float addrspace(1)*		; visa id: 10619
  store float %8180, float addrspace(1)* %8184, align 4		; visa id: 10620
  br label %._crit_edge70.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10621

._crit_edge70.9:                                  ; preds = %.._crit_edge70.9_crit_edge, %8165, %8160
; BB881 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10622
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10622
  %8185 = insertelement <2 x i32> %6047, i32 %8125, i64 1		; visa id: 10622
  store <2 x i32> %8185, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10625
  br label %._crit_edge372, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10627

._crit_edge372:                                   ; preds = %._crit_edge372.._crit_edge372_crit_edge, %._crit_edge70.9
; BB882 :
  %8186 = phi i32 [ 0, %._crit_edge70.9 ], [ %8195, %._crit_edge372.._crit_edge372_crit_edge ]
  %8187 = zext i32 %8186 to i64		; visa id: 10628
  %8188 = shl nuw nsw i64 %8187, 2		; visa id: 10629
  %8189 = add i64 %5982, %8188		; visa id: 10630
  %8190 = inttoptr i64 %8189 to i32*		; visa id: 10631
  %8191 = load i32, i32* %8190, align 4, !noalias !642		; visa id: 10631
  %8192 = add i64 %5978, %8188		; visa id: 10632
  %8193 = inttoptr i64 %8192 to i32*		; visa id: 10633
  store i32 %8191, i32* %8193, align 4, !alias.scope !642		; visa id: 10633
  %8194 = icmp eq i32 %8186, 0		; visa id: 10634
  br i1 %8194, label %._crit_edge372.._crit_edge372_crit_edge, label %8196, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10635

._crit_edge372.._crit_edge372_crit_edge:          ; preds = %._crit_edge372
; BB883 :
  %8195 = add nuw nsw i32 %8186, 1, !spirv.Decorations !631		; visa id: 10637
  br label %._crit_edge372, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10638

8196:                                             ; preds = %._crit_edge372
; BB884 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10640
  %8197 = load i64, i64* %5998, align 8		; visa id: 10640
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10641
  %8198 = icmp slt i32 %6046, %const_reg_dword
  %8199 = icmp slt i32 %8125, %const_reg_dword1		; visa id: 10641
  %8200 = and i1 %8198, %8199		; visa id: 10642
  br i1 %8200, label %8201, label %.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10644

.._crit_edge70.1.9_crit_edge:                     ; preds = %8196
; BB:
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8201:                                             ; preds = %8196
; BB886 :
  %8202 = bitcast i64 %8197 to <2 x i32>		; visa id: 10646
  %8203 = extractelement <2 x i32> %8202, i32 0		; visa id: 10648
  %8204 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8203, i32 1
  %8205 = bitcast <2 x i32> %8204 to i64		; visa id: 10648
  %8206 = ashr exact i64 %8205, 32		; visa id: 10649
  %8207 = bitcast i64 %8206 to <2 x i32>		; visa id: 10650
  %8208 = extractelement <2 x i32> %8207, i32 0		; visa id: 10654
  %8209 = extractelement <2 x i32> %8207, i32 1		; visa id: 10654
  %8210 = ashr i64 %8197, 32		; visa id: 10654
  %8211 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8208, i32 %8209, i32 %50, i32 %51)
  %8212 = extractvalue { i32, i32 } %8211, 0		; visa id: 10655
  %8213 = extractvalue { i32, i32 } %8211, 1		; visa id: 10655
  %8214 = insertelement <2 x i32> undef, i32 %8212, i32 0		; visa id: 10662
  %8215 = insertelement <2 x i32> %8214, i32 %8213, i32 1		; visa id: 10663
  %8216 = bitcast <2 x i32> %8215 to i64		; visa id: 10664
  %8217 = add nsw i64 %8216, %8210, !spirv.Decorations !649		; visa id: 10668
  %8218 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !618		; visa id: 10669
  br i1 %86, label %8224, label %8219, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10670

8219:                                             ; preds = %8201
; BB887 :
  %8220 = shl i64 %8217, 2		; visa id: 10672
  %8221 = add i64 %.in, %8220		; visa id: 10673
  %8222 = inttoptr i64 %8221 to float addrspace(4)*		; visa id: 10674
  %8223 = addrspacecast float addrspace(4)* %8222 to float addrspace(1)*		; visa id: 10674
  store float %8218, float addrspace(1)* %8223, align 4		; visa id: 10675
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10676

8224:                                             ; preds = %8201
; BB888 :
  %8225 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8208, i32 %8209, i32 %47, i32 %48)
  %8226 = extractvalue { i32, i32 } %8225, 0		; visa id: 10678
  %8227 = extractvalue { i32, i32 } %8225, 1		; visa id: 10678
  %8228 = insertelement <2 x i32> undef, i32 %8226, i32 0		; visa id: 10685
  %8229 = insertelement <2 x i32> %8228, i32 %8227, i32 1		; visa id: 10686
  %8230 = bitcast <2 x i32> %8229 to i64		; visa id: 10687
  %8231 = shl i64 %8230, 2		; visa id: 10691
  %8232 = add i64 %.in399, %8231		; visa id: 10692
  %8233 = shl nsw i64 %8210, 2		; visa id: 10693
  %8234 = add i64 %8232, %8233		; visa id: 10694
  %8235 = inttoptr i64 %8234 to float addrspace(4)*		; visa id: 10695
  %8236 = addrspacecast float addrspace(4)* %8235 to float addrspace(1)*		; visa id: 10695
  %8237 = load float, float addrspace(1)* %8236, align 4		; visa id: 10696
  %8238 = fmul reassoc nsz arcp contract float %8237, %4, !spirv.Decorations !618		; visa id: 10697
  %8239 = fadd reassoc nsz arcp contract float %8218, %8238, !spirv.Decorations !618		; visa id: 10698
  %8240 = shl i64 %8217, 2		; visa id: 10699
  %8241 = add i64 %.in, %8240		; visa id: 10700
  %8242 = inttoptr i64 %8241 to float addrspace(4)*		; visa id: 10701
  %8243 = addrspacecast float addrspace(4)* %8242 to float addrspace(1)*		; visa id: 10701
  store float %8239, float addrspace(1)* %8243, align 4		; visa id: 10702
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10703

._crit_edge70.1.9:                                ; preds = %.._crit_edge70.1.9_crit_edge, %8224, %8219
; BB889 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10704
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10704
  %8244 = insertelement <2 x i32> %6108, i32 %8125, i64 1		; visa id: 10704
  store <2 x i32> %8244, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10707
  br label %._crit_edge373, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10709

._crit_edge373:                                   ; preds = %._crit_edge373.._crit_edge373_crit_edge, %._crit_edge70.1.9
; BB890 :
  %8245 = phi i32 [ 0, %._crit_edge70.1.9 ], [ %8254, %._crit_edge373.._crit_edge373_crit_edge ]
  %8246 = zext i32 %8245 to i64		; visa id: 10710
  %8247 = shl nuw nsw i64 %8246, 2		; visa id: 10711
  %8248 = add i64 %5982, %8247		; visa id: 10712
  %8249 = inttoptr i64 %8248 to i32*		; visa id: 10713
  %8250 = load i32, i32* %8249, align 4, !noalias !642		; visa id: 10713
  %8251 = add i64 %5978, %8247		; visa id: 10714
  %8252 = inttoptr i64 %8251 to i32*		; visa id: 10715
  store i32 %8250, i32* %8252, align 4, !alias.scope !642		; visa id: 10715
  %8253 = icmp eq i32 %8245, 0		; visa id: 10716
  br i1 %8253, label %._crit_edge373.._crit_edge373_crit_edge, label %8255, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10717

._crit_edge373.._crit_edge373_crit_edge:          ; preds = %._crit_edge373
; BB891 :
  %8254 = add nuw nsw i32 %8245, 1, !spirv.Decorations !631		; visa id: 10719
  br label %._crit_edge373, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10720

8255:                                             ; preds = %._crit_edge373
; BB892 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10722
  %8256 = load i64, i64* %5998, align 8		; visa id: 10722
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10723
  %8257 = icmp slt i32 %6107, %const_reg_dword
  %8258 = icmp slt i32 %8125, %const_reg_dword1		; visa id: 10723
  %8259 = and i1 %8257, %8258		; visa id: 10724
  br i1 %8259, label %8260, label %.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10726

.._crit_edge70.2.9_crit_edge:                     ; preds = %8255
; BB:
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8260:                                             ; preds = %8255
; BB894 :
  %8261 = bitcast i64 %8256 to <2 x i32>		; visa id: 10728
  %8262 = extractelement <2 x i32> %8261, i32 0		; visa id: 10730
  %8263 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8262, i32 1
  %8264 = bitcast <2 x i32> %8263 to i64		; visa id: 10730
  %8265 = ashr exact i64 %8264, 32		; visa id: 10731
  %8266 = bitcast i64 %8265 to <2 x i32>		; visa id: 10732
  %8267 = extractelement <2 x i32> %8266, i32 0		; visa id: 10736
  %8268 = extractelement <2 x i32> %8266, i32 1		; visa id: 10736
  %8269 = ashr i64 %8256, 32		; visa id: 10736
  %8270 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8267, i32 %8268, i32 %50, i32 %51)
  %8271 = extractvalue { i32, i32 } %8270, 0		; visa id: 10737
  %8272 = extractvalue { i32, i32 } %8270, 1		; visa id: 10737
  %8273 = insertelement <2 x i32> undef, i32 %8271, i32 0		; visa id: 10744
  %8274 = insertelement <2 x i32> %8273, i32 %8272, i32 1		; visa id: 10745
  %8275 = bitcast <2 x i32> %8274 to i64		; visa id: 10746
  %8276 = add nsw i64 %8275, %8269, !spirv.Decorations !649		; visa id: 10750
  %8277 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !618		; visa id: 10751
  br i1 %86, label %8283, label %8278, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10752

8278:                                             ; preds = %8260
; BB895 :
  %8279 = shl i64 %8276, 2		; visa id: 10754
  %8280 = add i64 %.in, %8279		; visa id: 10755
  %8281 = inttoptr i64 %8280 to float addrspace(4)*		; visa id: 10756
  %8282 = addrspacecast float addrspace(4)* %8281 to float addrspace(1)*		; visa id: 10756
  store float %8277, float addrspace(1)* %8282, align 4		; visa id: 10757
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10758

8283:                                             ; preds = %8260
; BB896 :
  %8284 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8267, i32 %8268, i32 %47, i32 %48)
  %8285 = extractvalue { i32, i32 } %8284, 0		; visa id: 10760
  %8286 = extractvalue { i32, i32 } %8284, 1		; visa id: 10760
  %8287 = insertelement <2 x i32> undef, i32 %8285, i32 0		; visa id: 10767
  %8288 = insertelement <2 x i32> %8287, i32 %8286, i32 1		; visa id: 10768
  %8289 = bitcast <2 x i32> %8288 to i64		; visa id: 10769
  %8290 = shl i64 %8289, 2		; visa id: 10773
  %8291 = add i64 %.in399, %8290		; visa id: 10774
  %8292 = shl nsw i64 %8269, 2		; visa id: 10775
  %8293 = add i64 %8291, %8292		; visa id: 10776
  %8294 = inttoptr i64 %8293 to float addrspace(4)*		; visa id: 10777
  %8295 = addrspacecast float addrspace(4)* %8294 to float addrspace(1)*		; visa id: 10777
  %8296 = load float, float addrspace(1)* %8295, align 4		; visa id: 10778
  %8297 = fmul reassoc nsz arcp contract float %8296, %4, !spirv.Decorations !618		; visa id: 10779
  %8298 = fadd reassoc nsz arcp contract float %8277, %8297, !spirv.Decorations !618		; visa id: 10780
  %8299 = shl i64 %8276, 2		; visa id: 10781
  %8300 = add i64 %.in, %8299		; visa id: 10782
  %8301 = inttoptr i64 %8300 to float addrspace(4)*		; visa id: 10783
  %8302 = addrspacecast float addrspace(4)* %8301 to float addrspace(1)*		; visa id: 10783
  store float %8298, float addrspace(1)* %8302, align 4		; visa id: 10784
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10785

._crit_edge70.2.9:                                ; preds = %.._crit_edge70.2.9_crit_edge, %8283, %8278
; BB897 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10786
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10786
  %8303 = insertelement <2 x i32> %6169, i32 %8125, i64 1		; visa id: 10786
  store <2 x i32> %8303, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10789
  br label %._crit_edge374, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10791

._crit_edge374:                                   ; preds = %._crit_edge374.._crit_edge374_crit_edge, %._crit_edge70.2.9
; BB898 :
  %8304 = phi i32 [ 0, %._crit_edge70.2.9 ], [ %8313, %._crit_edge374.._crit_edge374_crit_edge ]
  %8305 = zext i32 %8304 to i64		; visa id: 10792
  %8306 = shl nuw nsw i64 %8305, 2		; visa id: 10793
  %8307 = add i64 %5982, %8306		; visa id: 10794
  %8308 = inttoptr i64 %8307 to i32*		; visa id: 10795
  %8309 = load i32, i32* %8308, align 4, !noalias !642		; visa id: 10795
  %8310 = add i64 %5978, %8306		; visa id: 10796
  %8311 = inttoptr i64 %8310 to i32*		; visa id: 10797
  store i32 %8309, i32* %8311, align 4, !alias.scope !642		; visa id: 10797
  %8312 = icmp eq i32 %8304, 0		; visa id: 10798
  br i1 %8312, label %._crit_edge374.._crit_edge374_crit_edge, label %8314, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10799

._crit_edge374.._crit_edge374_crit_edge:          ; preds = %._crit_edge374
; BB899 :
  %8313 = add nuw nsw i32 %8304, 1, !spirv.Decorations !631		; visa id: 10801
  br label %._crit_edge374, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10802

8314:                                             ; preds = %._crit_edge374
; BB900 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10804
  %8315 = load i64, i64* %5998, align 8		; visa id: 10804
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10805
  %8316 = icmp slt i32 %6168, %const_reg_dword
  %8317 = icmp slt i32 %8125, %const_reg_dword1		; visa id: 10805
  %8318 = and i1 %8316, %8317		; visa id: 10806
  br i1 %8318, label %8319, label %..preheader1.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10808

..preheader1.9_crit_edge:                         ; preds = %8314
; BB:
  br label %.preheader1.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8319:                                             ; preds = %8314
; BB902 :
  %8320 = bitcast i64 %8315 to <2 x i32>		; visa id: 10810
  %8321 = extractelement <2 x i32> %8320, i32 0		; visa id: 10812
  %8322 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8321, i32 1
  %8323 = bitcast <2 x i32> %8322 to i64		; visa id: 10812
  %8324 = ashr exact i64 %8323, 32		; visa id: 10813
  %8325 = bitcast i64 %8324 to <2 x i32>		; visa id: 10814
  %8326 = extractelement <2 x i32> %8325, i32 0		; visa id: 10818
  %8327 = extractelement <2 x i32> %8325, i32 1		; visa id: 10818
  %8328 = ashr i64 %8315, 32		; visa id: 10818
  %8329 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8326, i32 %8327, i32 %50, i32 %51)
  %8330 = extractvalue { i32, i32 } %8329, 0		; visa id: 10819
  %8331 = extractvalue { i32, i32 } %8329, 1		; visa id: 10819
  %8332 = insertelement <2 x i32> undef, i32 %8330, i32 0		; visa id: 10826
  %8333 = insertelement <2 x i32> %8332, i32 %8331, i32 1		; visa id: 10827
  %8334 = bitcast <2 x i32> %8333 to i64		; visa id: 10828
  %8335 = add nsw i64 %8334, %8328, !spirv.Decorations !649		; visa id: 10832
  %8336 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !618		; visa id: 10833
  br i1 %86, label %8342, label %8337, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10834

8337:                                             ; preds = %8319
; BB903 :
  %8338 = shl i64 %8335, 2		; visa id: 10836
  %8339 = add i64 %.in, %8338		; visa id: 10837
  %8340 = inttoptr i64 %8339 to float addrspace(4)*		; visa id: 10838
  %8341 = addrspacecast float addrspace(4)* %8340 to float addrspace(1)*		; visa id: 10838
  store float %8336, float addrspace(1)* %8341, align 4		; visa id: 10839
  br label %.preheader1.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10840

8342:                                             ; preds = %8319
; BB904 :
  %8343 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8326, i32 %8327, i32 %47, i32 %48)
  %8344 = extractvalue { i32, i32 } %8343, 0		; visa id: 10842
  %8345 = extractvalue { i32, i32 } %8343, 1		; visa id: 10842
  %8346 = insertelement <2 x i32> undef, i32 %8344, i32 0		; visa id: 10849
  %8347 = insertelement <2 x i32> %8346, i32 %8345, i32 1		; visa id: 10850
  %8348 = bitcast <2 x i32> %8347 to i64		; visa id: 10851
  %8349 = shl i64 %8348, 2		; visa id: 10855
  %8350 = add i64 %.in399, %8349		; visa id: 10856
  %8351 = shl nsw i64 %8328, 2		; visa id: 10857
  %8352 = add i64 %8350, %8351		; visa id: 10858
  %8353 = inttoptr i64 %8352 to float addrspace(4)*		; visa id: 10859
  %8354 = addrspacecast float addrspace(4)* %8353 to float addrspace(1)*		; visa id: 10859
  %8355 = load float, float addrspace(1)* %8354, align 4		; visa id: 10860
  %8356 = fmul reassoc nsz arcp contract float %8355, %4, !spirv.Decorations !618		; visa id: 10861
  %8357 = fadd reassoc nsz arcp contract float %8336, %8356, !spirv.Decorations !618		; visa id: 10862
  %8358 = shl i64 %8335, 2		; visa id: 10863
  %8359 = add i64 %.in, %8358		; visa id: 10864
  %8360 = inttoptr i64 %8359 to float addrspace(4)*		; visa id: 10865
  %8361 = addrspacecast float addrspace(4)* %8360 to float addrspace(1)*		; visa id: 10865
  store float %8357, float addrspace(1)* %8361, align 4		; visa id: 10866
  br label %.preheader1.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10867

.preheader1.9:                                    ; preds = %..preheader1.9_crit_edge, %8342, %8337
; BB905 :
  %8362 = add i32 %69, 10		; visa id: 10868
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10869
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10869
  %8363 = insertelement <2 x i32> %5984, i32 %8362, i64 1		; visa id: 10869
  store <2 x i32> %8363, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10872
  br label %._crit_edge375, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10874

._crit_edge375:                                   ; preds = %._crit_edge375.._crit_edge375_crit_edge, %.preheader1.9
; BB906 :
  %8364 = phi i32 [ 0, %.preheader1.9 ], [ %8373, %._crit_edge375.._crit_edge375_crit_edge ]
  %8365 = zext i32 %8364 to i64		; visa id: 10875
  %8366 = shl nuw nsw i64 %8365, 2		; visa id: 10876
  %8367 = add i64 %5982, %8366		; visa id: 10877
  %8368 = inttoptr i64 %8367 to i32*		; visa id: 10878
  %8369 = load i32, i32* %8368, align 4, !noalias !642		; visa id: 10878
  %8370 = add i64 %5978, %8366		; visa id: 10879
  %8371 = inttoptr i64 %8370 to i32*		; visa id: 10880
  store i32 %8369, i32* %8371, align 4, !alias.scope !642		; visa id: 10880
  %8372 = icmp eq i32 %8364, 0		; visa id: 10881
  br i1 %8372, label %._crit_edge375.._crit_edge375_crit_edge, label %8374, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10882

._crit_edge375.._crit_edge375_crit_edge:          ; preds = %._crit_edge375
; BB907 :
  %8373 = add nuw nsw i32 %8364, 1, !spirv.Decorations !631		; visa id: 10884
  br label %._crit_edge375, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10885

8374:                                             ; preds = %._crit_edge375
; BB908 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10887
  %8375 = load i64, i64* %5998, align 8		; visa id: 10887
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10888
  %8376 = icmp slt i32 %8362, %const_reg_dword1		; visa id: 10888
  %8377 = icmp slt i32 %65, %const_reg_dword
  %8378 = and i1 %8377, %8376		; visa id: 10889
  br i1 %8378, label %8379, label %.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10891

.._crit_edge70.10_crit_edge:                      ; preds = %8374
; BB:
  br label %._crit_edge70.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8379:                                             ; preds = %8374
; BB910 :
  %8380 = bitcast i64 %8375 to <2 x i32>		; visa id: 10893
  %8381 = extractelement <2 x i32> %8380, i32 0		; visa id: 10895
  %8382 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8381, i32 1
  %8383 = bitcast <2 x i32> %8382 to i64		; visa id: 10895
  %8384 = ashr exact i64 %8383, 32		; visa id: 10896
  %8385 = bitcast i64 %8384 to <2 x i32>		; visa id: 10897
  %8386 = extractelement <2 x i32> %8385, i32 0		; visa id: 10901
  %8387 = extractelement <2 x i32> %8385, i32 1		; visa id: 10901
  %8388 = ashr i64 %8375, 32		; visa id: 10901
  %8389 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8386, i32 %8387, i32 %50, i32 %51)
  %8390 = extractvalue { i32, i32 } %8389, 0		; visa id: 10902
  %8391 = extractvalue { i32, i32 } %8389, 1		; visa id: 10902
  %8392 = insertelement <2 x i32> undef, i32 %8390, i32 0		; visa id: 10909
  %8393 = insertelement <2 x i32> %8392, i32 %8391, i32 1		; visa id: 10910
  %8394 = bitcast <2 x i32> %8393 to i64		; visa id: 10911
  %8395 = add nsw i64 %8394, %8388, !spirv.Decorations !649		; visa id: 10915
  %8396 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !618		; visa id: 10916
  br i1 %86, label %8402, label %8397, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10917

8397:                                             ; preds = %8379
; BB911 :
  %8398 = shl i64 %8395, 2		; visa id: 10919
  %8399 = add i64 %.in, %8398		; visa id: 10920
  %8400 = inttoptr i64 %8399 to float addrspace(4)*		; visa id: 10921
  %8401 = addrspacecast float addrspace(4)* %8400 to float addrspace(1)*		; visa id: 10921
  store float %8396, float addrspace(1)* %8401, align 4		; visa id: 10922
  br label %._crit_edge70.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10923

8402:                                             ; preds = %8379
; BB912 :
  %8403 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8386, i32 %8387, i32 %47, i32 %48)
  %8404 = extractvalue { i32, i32 } %8403, 0		; visa id: 10925
  %8405 = extractvalue { i32, i32 } %8403, 1		; visa id: 10925
  %8406 = insertelement <2 x i32> undef, i32 %8404, i32 0		; visa id: 10932
  %8407 = insertelement <2 x i32> %8406, i32 %8405, i32 1		; visa id: 10933
  %8408 = bitcast <2 x i32> %8407 to i64		; visa id: 10934
  %8409 = shl i64 %8408, 2		; visa id: 10938
  %8410 = add i64 %.in399, %8409		; visa id: 10939
  %8411 = shl nsw i64 %8388, 2		; visa id: 10940
  %8412 = add i64 %8410, %8411		; visa id: 10941
  %8413 = inttoptr i64 %8412 to float addrspace(4)*		; visa id: 10942
  %8414 = addrspacecast float addrspace(4)* %8413 to float addrspace(1)*		; visa id: 10942
  %8415 = load float, float addrspace(1)* %8414, align 4		; visa id: 10943
  %8416 = fmul reassoc nsz arcp contract float %8415, %4, !spirv.Decorations !618		; visa id: 10944
  %8417 = fadd reassoc nsz arcp contract float %8396, %8416, !spirv.Decorations !618		; visa id: 10945
  %8418 = shl i64 %8395, 2		; visa id: 10946
  %8419 = add i64 %.in, %8418		; visa id: 10947
  %8420 = inttoptr i64 %8419 to float addrspace(4)*		; visa id: 10948
  %8421 = addrspacecast float addrspace(4)* %8420 to float addrspace(1)*		; visa id: 10948
  store float %8417, float addrspace(1)* %8421, align 4		; visa id: 10949
  br label %._crit_edge70.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10950

._crit_edge70.10:                                 ; preds = %.._crit_edge70.10_crit_edge, %8402, %8397
; BB913 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10951
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10951
  %8422 = insertelement <2 x i32> %6047, i32 %8362, i64 1		; visa id: 10951
  store <2 x i32> %8422, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 10954
  br label %._crit_edge376, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10956

._crit_edge376:                                   ; preds = %._crit_edge376.._crit_edge376_crit_edge, %._crit_edge70.10
; BB914 :
  %8423 = phi i32 [ 0, %._crit_edge70.10 ], [ %8432, %._crit_edge376.._crit_edge376_crit_edge ]
  %8424 = zext i32 %8423 to i64		; visa id: 10957
  %8425 = shl nuw nsw i64 %8424, 2		; visa id: 10958
  %8426 = add i64 %5982, %8425		; visa id: 10959
  %8427 = inttoptr i64 %8426 to i32*		; visa id: 10960
  %8428 = load i32, i32* %8427, align 4, !noalias !642		; visa id: 10960
  %8429 = add i64 %5978, %8425		; visa id: 10961
  %8430 = inttoptr i64 %8429 to i32*		; visa id: 10962
  store i32 %8428, i32* %8430, align 4, !alias.scope !642		; visa id: 10962
  %8431 = icmp eq i32 %8423, 0		; visa id: 10963
  br i1 %8431, label %._crit_edge376.._crit_edge376_crit_edge, label %8433, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10964

._crit_edge376.._crit_edge376_crit_edge:          ; preds = %._crit_edge376
; BB915 :
  %8432 = add nuw nsw i32 %8423, 1, !spirv.Decorations !631		; visa id: 10966
  br label %._crit_edge376, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10967

8433:                                             ; preds = %._crit_edge376
; BB916 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 10969
  %8434 = load i64, i64* %5998, align 8		; visa id: 10969
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 10970
  %8435 = icmp slt i32 %6046, %const_reg_dword
  %8436 = icmp slt i32 %8362, %const_reg_dword1		; visa id: 10970
  %8437 = and i1 %8435, %8436		; visa id: 10971
  br i1 %8437, label %8438, label %.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10973

.._crit_edge70.1.10_crit_edge:                    ; preds = %8433
; BB:
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8438:                                             ; preds = %8433
; BB918 :
  %8439 = bitcast i64 %8434 to <2 x i32>		; visa id: 10975
  %8440 = extractelement <2 x i32> %8439, i32 0		; visa id: 10977
  %8441 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8440, i32 1
  %8442 = bitcast <2 x i32> %8441 to i64		; visa id: 10977
  %8443 = ashr exact i64 %8442, 32		; visa id: 10978
  %8444 = bitcast i64 %8443 to <2 x i32>		; visa id: 10979
  %8445 = extractelement <2 x i32> %8444, i32 0		; visa id: 10983
  %8446 = extractelement <2 x i32> %8444, i32 1		; visa id: 10983
  %8447 = ashr i64 %8434, 32		; visa id: 10983
  %8448 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8445, i32 %8446, i32 %50, i32 %51)
  %8449 = extractvalue { i32, i32 } %8448, 0		; visa id: 10984
  %8450 = extractvalue { i32, i32 } %8448, 1		; visa id: 10984
  %8451 = insertelement <2 x i32> undef, i32 %8449, i32 0		; visa id: 10991
  %8452 = insertelement <2 x i32> %8451, i32 %8450, i32 1		; visa id: 10992
  %8453 = bitcast <2 x i32> %8452 to i64		; visa id: 10993
  %8454 = add nsw i64 %8453, %8447, !spirv.Decorations !649		; visa id: 10997
  %8455 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !618		; visa id: 10998
  br i1 %86, label %8461, label %8456, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10999

8456:                                             ; preds = %8438
; BB919 :
  %8457 = shl i64 %8454, 2		; visa id: 11001
  %8458 = add i64 %.in, %8457		; visa id: 11002
  %8459 = inttoptr i64 %8458 to float addrspace(4)*		; visa id: 11003
  %8460 = addrspacecast float addrspace(4)* %8459 to float addrspace(1)*		; visa id: 11003
  store float %8455, float addrspace(1)* %8460, align 4		; visa id: 11004
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11005

8461:                                             ; preds = %8438
; BB920 :
  %8462 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8445, i32 %8446, i32 %47, i32 %48)
  %8463 = extractvalue { i32, i32 } %8462, 0		; visa id: 11007
  %8464 = extractvalue { i32, i32 } %8462, 1		; visa id: 11007
  %8465 = insertelement <2 x i32> undef, i32 %8463, i32 0		; visa id: 11014
  %8466 = insertelement <2 x i32> %8465, i32 %8464, i32 1		; visa id: 11015
  %8467 = bitcast <2 x i32> %8466 to i64		; visa id: 11016
  %8468 = shl i64 %8467, 2		; visa id: 11020
  %8469 = add i64 %.in399, %8468		; visa id: 11021
  %8470 = shl nsw i64 %8447, 2		; visa id: 11022
  %8471 = add i64 %8469, %8470		; visa id: 11023
  %8472 = inttoptr i64 %8471 to float addrspace(4)*		; visa id: 11024
  %8473 = addrspacecast float addrspace(4)* %8472 to float addrspace(1)*		; visa id: 11024
  %8474 = load float, float addrspace(1)* %8473, align 4		; visa id: 11025
  %8475 = fmul reassoc nsz arcp contract float %8474, %4, !spirv.Decorations !618		; visa id: 11026
  %8476 = fadd reassoc nsz arcp contract float %8455, %8475, !spirv.Decorations !618		; visa id: 11027
  %8477 = shl i64 %8454, 2		; visa id: 11028
  %8478 = add i64 %.in, %8477		; visa id: 11029
  %8479 = inttoptr i64 %8478 to float addrspace(4)*		; visa id: 11030
  %8480 = addrspacecast float addrspace(4)* %8479 to float addrspace(1)*		; visa id: 11030
  store float %8476, float addrspace(1)* %8480, align 4		; visa id: 11031
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11032

._crit_edge70.1.10:                               ; preds = %.._crit_edge70.1.10_crit_edge, %8461, %8456
; BB921 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11033
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11033
  %8481 = insertelement <2 x i32> %6108, i32 %8362, i64 1		; visa id: 11033
  store <2 x i32> %8481, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11036
  br label %._crit_edge377, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11038

._crit_edge377:                                   ; preds = %._crit_edge377.._crit_edge377_crit_edge, %._crit_edge70.1.10
; BB922 :
  %8482 = phi i32 [ 0, %._crit_edge70.1.10 ], [ %8491, %._crit_edge377.._crit_edge377_crit_edge ]
  %8483 = zext i32 %8482 to i64		; visa id: 11039
  %8484 = shl nuw nsw i64 %8483, 2		; visa id: 11040
  %8485 = add i64 %5982, %8484		; visa id: 11041
  %8486 = inttoptr i64 %8485 to i32*		; visa id: 11042
  %8487 = load i32, i32* %8486, align 4, !noalias !642		; visa id: 11042
  %8488 = add i64 %5978, %8484		; visa id: 11043
  %8489 = inttoptr i64 %8488 to i32*		; visa id: 11044
  store i32 %8487, i32* %8489, align 4, !alias.scope !642		; visa id: 11044
  %8490 = icmp eq i32 %8482, 0		; visa id: 11045
  br i1 %8490, label %._crit_edge377.._crit_edge377_crit_edge, label %8492, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11046

._crit_edge377.._crit_edge377_crit_edge:          ; preds = %._crit_edge377
; BB923 :
  %8491 = add nuw nsw i32 %8482, 1, !spirv.Decorations !631		; visa id: 11048
  br label %._crit_edge377, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11049

8492:                                             ; preds = %._crit_edge377
; BB924 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11051
  %8493 = load i64, i64* %5998, align 8		; visa id: 11051
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11052
  %8494 = icmp slt i32 %6107, %const_reg_dword
  %8495 = icmp slt i32 %8362, %const_reg_dword1		; visa id: 11052
  %8496 = and i1 %8494, %8495		; visa id: 11053
  br i1 %8496, label %8497, label %.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11055

.._crit_edge70.2.10_crit_edge:                    ; preds = %8492
; BB:
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8497:                                             ; preds = %8492
; BB926 :
  %8498 = bitcast i64 %8493 to <2 x i32>		; visa id: 11057
  %8499 = extractelement <2 x i32> %8498, i32 0		; visa id: 11059
  %8500 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8499, i32 1
  %8501 = bitcast <2 x i32> %8500 to i64		; visa id: 11059
  %8502 = ashr exact i64 %8501, 32		; visa id: 11060
  %8503 = bitcast i64 %8502 to <2 x i32>		; visa id: 11061
  %8504 = extractelement <2 x i32> %8503, i32 0		; visa id: 11065
  %8505 = extractelement <2 x i32> %8503, i32 1		; visa id: 11065
  %8506 = ashr i64 %8493, 32		; visa id: 11065
  %8507 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8504, i32 %8505, i32 %50, i32 %51)
  %8508 = extractvalue { i32, i32 } %8507, 0		; visa id: 11066
  %8509 = extractvalue { i32, i32 } %8507, 1		; visa id: 11066
  %8510 = insertelement <2 x i32> undef, i32 %8508, i32 0		; visa id: 11073
  %8511 = insertelement <2 x i32> %8510, i32 %8509, i32 1		; visa id: 11074
  %8512 = bitcast <2 x i32> %8511 to i64		; visa id: 11075
  %8513 = add nsw i64 %8512, %8506, !spirv.Decorations !649		; visa id: 11079
  %8514 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !618		; visa id: 11080
  br i1 %86, label %8520, label %8515, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11081

8515:                                             ; preds = %8497
; BB927 :
  %8516 = shl i64 %8513, 2		; visa id: 11083
  %8517 = add i64 %.in, %8516		; visa id: 11084
  %8518 = inttoptr i64 %8517 to float addrspace(4)*		; visa id: 11085
  %8519 = addrspacecast float addrspace(4)* %8518 to float addrspace(1)*		; visa id: 11085
  store float %8514, float addrspace(1)* %8519, align 4		; visa id: 11086
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11087

8520:                                             ; preds = %8497
; BB928 :
  %8521 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8504, i32 %8505, i32 %47, i32 %48)
  %8522 = extractvalue { i32, i32 } %8521, 0		; visa id: 11089
  %8523 = extractvalue { i32, i32 } %8521, 1		; visa id: 11089
  %8524 = insertelement <2 x i32> undef, i32 %8522, i32 0		; visa id: 11096
  %8525 = insertelement <2 x i32> %8524, i32 %8523, i32 1		; visa id: 11097
  %8526 = bitcast <2 x i32> %8525 to i64		; visa id: 11098
  %8527 = shl i64 %8526, 2		; visa id: 11102
  %8528 = add i64 %.in399, %8527		; visa id: 11103
  %8529 = shl nsw i64 %8506, 2		; visa id: 11104
  %8530 = add i64 %8528, %8529		; visa id: 11105
  %8531 = inttoptr i64 %8530 to float addrspace(4)*		; visa id: 11106
  %8532 = addrspacecast float addrspace(4)* %8531 to float addrspace(1)*		; visa id: 11106
  %8533 = load float, float addrspace(1)* %8532, align 4		; visa id: 11107
  %8534 = fmul reassoc nsz arcp contract float %8533, %4, !spirv.Decorations !618		; visa id: 11108
  %8535 = fadd reassoc nsz arcp contract float %8514, %8534, !spirv.Decorations !618		; visa id: 11109
  %8536 = shl i64 %8513, 2		; visa id: 11110
  %8537 = add i64 %.in, %8536		; visa id: 11111
  %8538 = inttoptr i64 %8537 to float addrspace(4)*		; visa id: 11112
  %8539 = addrspacecast float addrspace(4)* %8538 to float addrspace(1)*		; visa id: 11112
  store float %8535, float addrspace(1)* %8539, align 4		; visa id: 11113
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11114

._crit_edge70.2.10:                               ; preds = %.._crit_edge70.2.10_crit_edge, %8520, %8515
; BB929 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11115
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11115
  %8540 = insertelement <2 x i32> %6169, i32 %8362, i64 1		; visa id: 11115
  store <2 x i32> %8540, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11118
  br label %._crit_edge378, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11120

._crit_edge378:                                   ; preds = %._crit_edge378.._crit_edge378_crit_edge, %._crit_edge70.2.10
; BB930 :
  %8541 = phi i32 [ 0, %._crit_edge70.2.10 ], [ %8550, %._crit_edge378.._crit_edge378_crit_edge ]
  %8542 = zext i32 %8541 to i64		; visa id: 11121
  %8543 = shl nuw nsw i64 %8542, 2		; visa id: 11122
  %8544 = add i64 %5982, %8543		; visa id: 11123
  %8545 = inttoptr i64 %8544 to i32*		; visa id: 11124
  %8546 = load i32, i32* %8545, align 4, !noalias !642		; visa id: 11124
  %8547 = add i64 %5978, %8543		; visa id: 11125
  %8548 = inttoptr i64 %8547 to i32*		; visa id: 11126
  store i32 %8546, i32* %8548, align 4, !alias.scope !642		; visa id: 11126
  %8549 = icmp eq i32 %8541, 0		; visa id: 11127
  br i1 %8549, label %._crit_edge378.._crit_edge378_crit_edge, label %8551, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11128

._crit_edge378.._crit_edge378_crit_edge:          ; preds = %._crit_edge378
; BB931 :
  %8550 = add nuw nsw i32 %8541, 1, !spirv.Decorations !631		; visa id: 11130
  br label %._crit_edge378, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11131

8551:                                             ; preds = %._crit_edge378
; BB932 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11133
  %8552 = load i64, i64* %5998, align 8		; visa id: 11133
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11134
  %8553 = icmp slt i32 %6168, %const_reg_dword
  %8554 = icmp slt i32 %8362, %const_reg_dword1		; visa id: 11134
  %8555 = and i1 %8553, %8554		; visa id: 11135
  br i1 %8555, label %8556, label %..preheader1.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11137

..preheader1.10_crit_edge:                        ; preds = %8551
; BB:
  br label %.preheader1.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8556:                                             ; preds = %8551
; BB934 :
  %8557 = bitcast i64 %8552 to <2 x i32>		; visa id: 11139
  %8558 = extractelement <2 x i32> %8557, i32 0		; visa id: 11141
  %8559 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8558, i32 1
  %8560 = bitcast <2 x i32> %8559 to i64		; visa id: 11141
  %8561 = ashr exact i64 %8560, 32		; visa id: 11142
  %8562 = bitcast i64 %8561 to <2 x i32>		; visa id: 11143
  %8563 = extractelement <2 x i32> %8562, i32 0		; visa id: 11147
  %8564 = extractelement <2 x i32> %8562, i32 1		; visa id: 11147
  %8565 = ashr i64 %8552, 32		; visa id: 11147
  %8566 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8563, i32 %8564, i32 %50, i32 %51)
  %8567 = extractvalue { i32, i32 } %8566, 0		; visa id: 11148
  %8568 = extractvalue { i32, i32 } %8566, 1		; visa id: 11148
  %8569 = insertelement <2 x i32> undef, i32 %8567, i32 0		; visa id: 11155
  %8570 = insertelement <2 x i32> %8569, i32 %8568, i32 1		; visa id: 11156
  %8571 = bitcast <2 x i32> %8570 to i64		; visa id: 11157
  %8572 = add nsw i64 %8571, %8565, !spirv.Decorations !649		; visa id: 11161
  %8573 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !618		; visa id: 11162
  br i1 %86, label %8579, label %8574, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11163

8574:                                             ; preds = %8556
; BB935 :
  %8575 = shl i64 %8572, 2		; visa id: 11165
  %8576 = add i64 %.in, %8575		; visa id: 11166
  %8577 = inttoptr i64 %8576 to float addrspace(4)*		; visa id: 11167
  %8578 = addrspacecast float addrspace(4)* %8577 to float addrspace(1)*		; visa id: 11167
  store float %8573, float addrspace(1)* %8578, align 4		; visa id: 11168
  br label %.preheader1.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11169

8579:                                             ; preds = %8556
; BB936 :
  %8580 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8563, i32 %8564, i32 %47, i32 %48)
  %8581 = extractvalue { i32, i32 } %8580, 0		; visa id: 11171
  %8582 = extractvalue { i32, i32 } %8580, 1		; visa id: 11171
  %8583 = insertelement <2 x i32> undef, i32 %8581, i32 0		; visa id: 11178
  %8584 = insertelement <2 x i32> %8583, i32 %8582, i32 1		; visa id: 11179
  %8585 = bitcast <2 x i32> %8584 to i64		; visa id: 11180
  %8586 = shl i64 %8585, 2		; visa id: 11184
  %8587 = add i64 %.in399, %8586		; visa id: 11185
  %8588 = shl nsw i64 %8565, 2		; visa id: 11186
  %8589 = add i64 %8587, %8588		; visa id: 11187
  %8590 = inttoptr i64 %8589 to float addrspace(4)*		; visa id: 11188
  %8591 = addrspacecast float addrspace(4)* %8590 to float addrspace(1)*		; visa id: 11188
  %8592 = load float, float addrspace(1)* %8591, align 4		; visa id: 11189
  %8593 = fmul reassoc nsz arcp contract float %8592, %4, !spirv.Decorations !618		; visa id: 11190
  %8594 = fadd reassoc nsz arcp contract float %8573, %8593, !spirv.Decorations !618		; visa id: 11191
  %8595 = shl i64 %8572, 2		; visa id: 11192
  %8596 = add i64 %.in, %8595		; visa id: 11193
  %8597 = inttoptr i64 %8596 to float addrspace(4)*		; visa id: 11194
  %8598 = addrspacecast float addrspace(4)* %8597 to float addrspace(1)*		; visa id: 11194
  store float %8594, float addrspace(1)* %8598, align 4		; visa id: 11195
  br label %.preheader1.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11196

.preheader1.10:                                   ; preds = %..preheader1.10_crit_edge, %8579, %8574
; BB937 :
  %8599 = add i32 %69, 11		; visa id: 11197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11198
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11198
  %8600 = insertelement <2 x i32> %5984, i32 %8599, i64 1		; visa id: 11198
  store <2 x i32> %8600, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11201
  br label %._crit_edge379, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11203

._crit_edge379:                                   ; preds = %._crit_edge379.._crit_edge379_crit_edge, %.preheader1.10
; BB938 :
  %8601 = phi i32 [ 0, %.preheader1.10 ], [ %8610, %._crit_edge379.._crit_edge379_crit_edge ]
  %8602 = zext i32 %8601 to i64		; visa id: 11204
  %8603 = shl nuw nsw i64 %8602, 2		; visa id: 11205
  %8604 = add i64 %5982, %8603		; visa id: 11206
  %8605 = inttoptr i64 %8604 to i32*		; visa id: 11207
  %8606 = load i32, i32* %8605, align 4, !noalias !642		; visa id: 11207
  %8607 = add i64 %5978, %8603		; visa id: 11208
  %8608 = inttoptr i64 %8607 to i32*		; visa id: 11209
  store i32 %8606, i32* %8608, align 4, !alias.scope !642		; visa id: 11209
  %8609 = icmp eq i32 %8601, 0		; visa id: 11210
  br i1 %8609, label %._crit_edge379.._crit_edge379_crit_edge, label %8611, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11211

._crit_edge379.._crit_edge379_crit_edge:          ; preds = %._crit_edge379
; BB939 :
  %8610 = add nuw nsw i32 %8601, 1, !spirv.Decorations !631		; visa id: 11213
  br label %._crit_edge379, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11214

8611:                                             ; preds = %._crit_edge379
; BB940 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11216
  %8612 = load i64, i64* %5998, align 8		; visa id: 11216
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11217
  %8613 = icmp slt i32 %8599, %const_reg_dword1		; visa id: 11217
  %8614 = icmp slt i32 %65, %const_reg_dword
  %8615 = and i1 %8614, %8613		; visa id: 11218
  br i1 %8615, label %8616, label %.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11220

.._crit_edge70.11_crit_edge:                      ; preds = %8611
; BB:
  br label %._crit_edge70.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8616:                                             ; preds = %8611
; BB942 :
  %8617 = bitcast i64 %8612 to <2 x i32>		; visa id: 11222
  %8618 = extractelement <2 x i32> %8617, i32 0		; visa id: 11224
  %8619 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8618, i32 1
  %8620 = bitcast <2 x i32> %8619 to i64		; visa id: 11224
  %8621 = ashr exact i64 %8620, 32		; visa id: 11225
  %8622 = bitcast i64 %8621 to <2 x i32>		; visa id: 11226
  %8623 = extractelement <2 x i32> %8622, i32 0		; visa id: 11230
  %8624 = extractelement <2 x i32> %8622, i32 1		; visa id: 11230
  %8625 = ashr i64 %8612, 32		; visa id: 11230
  %8626 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8623, i32 %8624, i32 %50, i32 %51)
  %8627 = extractvalue { i32, i32 } %8626, 0		; visa id: 11231
  %8628 = extractvalue { i32, i32 } %8626, 1		; visa id: 11231
  %8629 = insertelement <2 x i32> undef, i32 %8627, i32 0		; visa id: 11238
  %8630 = insertelement <2 x i32> %8629, i32 %8628, i32 1		; visa id: 11239
  %8631 = bitcast <2 x i32> %8630 to i64		; visa id: 11240
  %8632 = add nsw i64 %8631, %8625, !spirv.Decorations !649		; visa id: 11244
  %8633 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !618		; visa id: 11245
  br i1 %86, label %8639, label %8634, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11246

8634:                                             ; preds = %8616
; BB943 :
  %8635 = shl i64 %8632, 2		; visa id: 11248
  %8636 = add i64 %.in, %8635		; visa id: 11249
  %8637 = inttoptr i64 %8636 to float addrspace(4)*		; visa id: 11250
  %8638 = addrspacecast float addrspace(4)* %8637 to float addrspace(1)*		; visa id: 11250
  store float %8633, float addrspace(1)* %8638, align 4		; visa id: 11251
  br label %._crit_edge70.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11252

8639:                                             ; preds = %8616
; BB944 :
  %8640 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8623, i32 %8624, i32 %47, i32 %48)
  %8641 = extractvalue { i32, i32 } %8640, 0		; visa id: 11254
  %8642 = extractvalue { i32, i32 } %8640, 1		; visa id: 11254
  %8643 = insertelement <2 x i32> undef, i32 %8641, i32 0		; visa id: 11261
  %8644 = insertelement <2 x i32> %8643, i32 %8642, i32 1		; visa id: 11262
  %8645 = bitcast <2 x i32> %8644 to i64		; visa id: 11263
  %8646 = shl i64 %8645, 2		; visa id: 11267
  %8647 = add i64 %.in399, %8646		; visa id: 11268
  %8648 = shl nsw i64 %8625, 2		; visa id: 11269
  %8649 = add i64 %8647, %8648		; visa id: 11270
  %8650 = inttoptr i64 %8649 to float addrspace(4)*		; visa id: 11271
  %8651 = addrspacecast float addrspace(4)* %8650 to float addrspace(1)*		; visa id: 11271
  %8652 = load float, float addrspace(1)* %8651, align 4		; visa id: 11272
  %8653 = fmul reassoc nsz arcp contract float %8652, %4, !spirv.Decorations !618		; visa id: 11273
  %8654 = fadd reassoc nsz arcp contract float %8633, %8653, !spirv.Decorations !618		; visa id: 11274
  %8655 = shl i64 %8632, 2		; visa id: 11275
  %8656 = add i64 %.in, %8655		; visa id: 11276
  %8657 = inttoptr i64 %8656 to float addrspace(4)*		; visa id: 11277
  %8658 = addrspacecast float addrspace(4)* %8657 to float addrspace(1)*		; visa id: 11277
  store float %8654, float addrspace(1)* %8658, align 4		; visa id: 11278
  br label %._crit_edge70.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11279

._crit_edge70.11:                                 ; preds = %.._crit_edge70.11_crit_edge, %8639, %8634
; BB945 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11280
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11280
  %8659 = insertelement <2 x i32> %6047, i32 %8599, i64 1		; visa id: 11280
  store <2 x i32> %8659, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11283
  br label %._crit_edge380, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11285

._crit_edge380:                                   ; preds = %._crit_edge380.._crit_edge380_crit_edge, %._crit_edge70.11
; BB946 :
  %8660 = phi i32 [ 0, %._crit_edge70.11 ], [ %8669, %._crit_edge380.._crit_edge380_crit_edge ]
  %8661 = zext i32 %8660 to i64		; visa id: 11286
  %8662 = shl nuw nsw i64 %8661, 2		; visa id: 11287
  %8663 = add i64 %5982, %8662		; visa id: 11288
  %8664 = inttoptr i64 %8663 to i32*		; visa id: 11289
  %8665 = load i32, i32* %8664, align 4, !noalias !642		; visa id: 11289
  %8666 = add i64 %5978, %8662		; visa id: 11290
  %8667 = inttoptr i64 %8666 to i32*		; visa id: 11291
  store i32 %8665, i32* %8667, align 4, !alias.scope !642		; visa id: 11291
  %8668 = icmp eq i32 %8660, 0		; visa id: 11292
  br i1 %8668, label %._crit_edge380.._crit_edge380_crit_edge, label %8670, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11293

._crit_edge380.._crit_edge380_crit_edge:          ; preds = %._crit_edge380
; BB947 :
  %8669 = add nuw nsw i32 %8660, 1, !spirv.Decorations !631		; visa id: 11295
  br label %._crit_edge380, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11296

8670:                                             ; preds = %._crit_edge380
; BB948 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11298
  %8671 = load i64, i64* %5998, align 8		; visa id: 11298
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11299
  %8672 = icmp slt i32 %6046, %const_reg_dword
  %8673 = icmp slt i32 %8599, %const_reg_dword1		; visa id: 11299
  %8674 = and i1 %8672, %8673		; visa id: 11300
  br i1 %8674, label %8675, label %.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11302

.._crit_edge70.1.11_crit_edge:                    ; preds = %8670
; BB:
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8675:                                             ; preds = %8670
; BB950 :
  %8676 = bitcast i64 %8671 to <2 x i32>		; visa id: 11304
  %8677 = extractelement <2 x i32> %8676, i32 0		; visa id: 11306
  %8678 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8677, i32 1
  %8679 = bitcast <2 x i32> %8678 to i64		; visa id: 11306
  %8680 = ashr exact i64 %8679, 32		; visa id: 11307
  %8681 = bitcast i64 %8680 to <2 x i32>		; visa id: 11308
  %8682 = extractelement <2 x i32> %8681, i32 0		; visa id: 11312
  %8683 = extractelement <2 x i32> %8681, i32 1		; visa id: 11312
  %8684 = ashr i64 %8671, 32		; visa id: 11312
  %8685 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8682, i32 %8683, i32 %50, i32 %51)
  %8686 = extractvalue { i32, i32 } %8685, 0		; visa id: 11313
  %8687 = extractvalue { i32, i32 } %8685, 1		; visa id: 11313
  %8688 = insertelement <2 x i32> undef, i32 %8686, i32 0		; visa id: 11320
  %8689 = insertelement <2 x i32> %8688, i32 %8687, i32 1		; visa id: 11321
  %8690 = bitcast <2 x i32> %8689 to i64		; visa id: 11322
  %8691 = add nsw i64 %8690, %8684, !spirv.Decorations !649		; visa id: 11326
  %8692 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !618		; visa id: 11327
  br i1 %86, label %8698, label %8693, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11328

8693:                                             ; preds = %8675
; BB951 :
  %8694 = shl i64 %8691, 2		; visa id: 11330
  %8695 = add i64 %.in, %8694		; visa id: 11331
  %8696 = inttoptr i64 %8695 to float addrspace(4)*		; visa id: 11332
  %8697 = addrspacecast float addrspace(4)* %8696 to float addrspace(1)*		; visa id: 11332
  store float %8692, float addrspace(1)* %8697, align 4		; visa id: 11333
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11334

8698:                                             ; preds = %8675
; BB952 :
  %8699 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8682, i32 %8683, i32 %47, i32 %48)
  %8700 = extractvalue { i32, i32 } %8699, 0		; visa id: 11336
  %8701 = extractvalue { i32, i32 } %8699, 1		; visa id: 11336
  %8702 = insertelement <2 x i32> undef, i32 %8700, i32 0		; visa id: 11343
  %8703 = insertelement <2 x i32> %8702, i32 %8701, i32 1		; visa id: 11344
  %8704 = bitcast <2 x i32> %8703 to i64		; visa id: 11345
  %8705 = shl i64 %8704, 2		; visa id: 11349
  %8706 = add i64 %.in399, %8705		; visa id: 11350
  %8707 = shl nsw i64 %8684, 2		; visa id: 11351
  %8708 = add i64 %8706, %8707		; visa id: 11352
  %8709 = inttoptr i64 %8708 to float addrspace(4)*		; visa id: 11353
  %8710 = addrspacecast float addrspace(4)* %8709 to float addrspace(1)*		; visa id: 11353
  %8711 = load float, float addrspace(1)* %8710, align 4		; visa id: 11354
  %8712 = fmul reassoc nsz arcp contract float %8711, %4, !spirv.Decorations !618		; visa id: 11355
  %8713 = fadd reassoc nsz arcp contract float %8692, %8712, !spirv.Decorations !618		; visa id: 11356
  %8714 = shl i64 %8691, 2		; visa id: 11357
  %8715 = add i64 %.in, %8714		; visa id: 11358
  %8716 = inttoptr i64 %8715 to float addrspace(4)*		; visa id: 11359
  %8717 = addrspacecast float addrspace(4)* %8716 to float addrspace(1)*		; visa id: 11359
  store float %8713, float addrspace(1)* %8717, align 4		; visa id: 11360
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11361

._crit_edge70.1.11:                               ; preds = %.._crit_edge70.1.11_crit_edge, %8698, %8693
; BB953 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11362
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11362
  %8718 = insertelement <2 x i32> %6108, i32 %8599, i64 1		; visa id: 11362
  store <2 x i32> %8718, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11365
  br label %._crit_edge381, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11367

._crit_edge381:                                   ; preds = %._crit_edge381.._crit_edge381_crit_edge, %._crit_edge70.1.11
; BB954 :
  %8719 = phi i32 [ 0, %._crit_edge70.1.11 ], [ %8728, %._crit_edge381.._crit_edge381_crit_edge ]
  %8720 = zext i32 %8719 to i64		; visa id: 11368
  %8721 = shl nuw nsw i64 %8720, 2		; visa id: 11369
  %8722 = add i64 %5982, %8721		; visa id: 11370
  %8723 = inttoptr i64 %8722 to i32*		; visa id: 11371
  %8724 = load i32, i32* %8723, align 4, !noalias !642		; visa id: 11371
  %8725 = add i64 %5978, %8721		; visa id: 11372
  %8726 = inttoptr i64 %8725 to i32*		; visa id: 11373
  store i32 %8724, i32* %8726, align 4, !alias.scope !642		; visa id: 11373
  %8727 = icmp eq i32 %8719, 0		; visa id: 11374
  br i1 %8727, label %._crit_edge381.._crit_edge381_crit_edge, label %8729, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11375

._crit_edge381.._crit_edge381_crit_edge:          ; preds = %._crit_edge381
; BB955 :
  %8728 = add nuw nsw i32 %8719, 1, !spirv.Decorations !631		; visa id: 11377
  br label %._crit_edge381, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11378

8729:                                             ; preds = %._crit_edge381
; BB956 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11380
  %8730 = load i64, i64* %5998, align 8		; visa id: 11380
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11381
  %8731 = icmp slt i32 %6107, %const_reg_dword
  %8732 = icmp slt i32 %8599, %const_reg_dword1		; visa id: 11381
  %8733 = and i1 %8731, %8732		; visa id: 11382
  br i1 %8733, label %8734, label %.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11384

.._crit_edge70.2.11_crit_edge:                    ; preds = %8729
; BB:
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8734:                                             ; preds = %8729
; BB958 :
  %8735 = bitcast i64 %8730 to <2 x i32>		; visa id: 11386
  %8736 = extractelement <2 x i32> %8735, i32 0		; visa id: 11388
  %8737 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8736, i32 1
  %8738 = bitcast <2 x i32> %8737 to i64		; visa id: 11388
  %8739 = ashr exact i64 %8738, 32		; visa id: 11389
  %8740 = bitcast i64 %8739 to <2 x i32>		; visa id: 11390
  %8741 = extractelement <2 x i32> %8740, i32 0		; visa id: 11394
  %8742 = extractelement <2 x i32> %8740, i32 1		; visa id: 11394
  %8743 = ashr i64 %8730, 32		; visa id: 11394
  %8744 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8741, i32 %8742, i32 %50, i32 %51)
  %8745 = extractvalue { i32, i32 } %8744, 0		; visa id: 11395
  %8746 = extractvalue { i32, i32 } %8744, 1		; visa id: 11395
  %8747 = insertelement <2 x i32> undef, i32 %8745, i32 0		; visa id: 11402
  %8748 = insertelement <2 x i32> %8747, i32 %8746, i32 1		; visa id: 11403
  %8749 = bitcast <2 x i32> %8748 to i64		; visa id: 11404
  %8750 = add nsw i64 %8749, %8743, !spirv.Decorations !649		; visa id: 11408
  %8751 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !618		; visa id: 11409
  br i1 %86, label %8757, label %8752, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11410

8752:                                             ; preds = %8734
; BB959 :
  %8753 = shl i64 %8750, 2		; visa id: 11412
  %8754 = add i64 %.in, %8753		; visa id: 11413
  %8755 = inttoptr i64 %8754 to float addrspace(4)*		; visa id: 11414
  %8756 = addrspacecast float addrspace(4)* %8755 to float addrspace(1)*		; visa id: 11414
  store float %8751, float addrspace(1)* %8756, align 4		; visa id: 11415
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11416

8757:                                             ; preds = %8734
; BB960 :
  %8758 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8741, i32 %8742, i32 %47, i32 %48)
  %8759 = extractvalue { i32, i32 } %8758, 0		; visa id: 11418
  %8760 = extractvalue { i32, i32 } %8758, 1		; visa id: 11418
  %8761 = insertelement <2 x i32> undef, i32 %8759, i32 0		; visa id: 11425
  %8762 = insertelement <2 x i32> %8761, i32 %8760, i32 1		; visa id: 11426
  %8763 = bitcast <2 x i32> %8762 to i64		; visa id: 11427
  %8764 = shl i64 %8763, 2		; visa id: 11431
  %8765 = add i64 %.in399, %8764		; visa id: 11432
  %8766 = shl nsw i64 %8743, 2		; visa id: 11433
  %8767 = add i64 %8765, %8766		; visa id: 11434
  %8768 = inttoptr i64 %8767 to float addrspace(4)*		; visa id: 11435
  %8769 = addrspacecast float addrspace(4)* %8768 to float addrspace(1)*		; visa id: 11435
  %8770 = load float, float addrspace(1)* %8769, align 4		; visa id: 11436
  %8771 = fmul reassoc nsz arcp contract float %8770, %4, !spirv.Decorations !618		; visa id: 11437
  %8772 = fadd reassoc nsz arcp contract float %8751, %8771, !spirv.Decorations !618		; visa id: 11438
  %8773 = shl i64 %8750, 2		; visa id: 11439
  %8774 = add i64 %.in, %8773		; visa id: 11440
  %8775 = inttoptr i64 %8774 to float addrspace(4)*		; visa id: 11441
  %8776 = addrspacecast float addrspace(4)* %8775 to float addrspace(1)*		; visa id: 11441
  store float %8772, float addrspace(1)* %8776, align 4		; visa id: 11442
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11443

._crit_edge70.2.11:                               ; preds = %.._crit_edge70.2.11_crit_edge, %8757, %8752
; BB961 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11444
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11444
  %8777 = insertelement <2 x i32> %6169, i32 %8599, i64 1		; visa id: 11444
  store <2 x i32> %8777, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11447
  br label %._crit_edge382, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11449

._crit_edge382:                                   ; preds = %._crit_edge382.._crit_edge382_crit_edge, %._crit_edge70.2.11
; BB962 :
  %8778 = phi i32 [ 0, %._crit_edge70.2.11 ], [ %8787, %._crit_edge382.._crit_edge382_crit_edge ]
  %8779 = zext i32 %8778 to i64		; visa id: 11450
  %8780 = shl nuw nsw i64 %8779, 2		; visa id: 11451
  %8781 = add i64 %5982, %8780		; visa id: 11452
  %8782 = inttoptr i64 %8781 to i32*		; visa id: 11453
  %8783 = load i32, i32* %8782, align 4, !noalias !642		; visa id: 11453
  %8784 = add i64 %5978, %8780		; visa id: 11454
  %8785 = inttoptr i64 %8784 to i32*		; visa id: 11455
  store i32 %8783, i32* %8785, align 4, !alias.scope !642		; visa id: 11455
  %8786 = icmp eq i32 %8778, 0		; visa id: 11456
  br i1 %8786, label %._crit_edge382.._crit_edge382_crit_edge, label %8788, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11457

._crit_edge382.._crit_edge382_crit_edge:          ; preds = %._crit_edge382
; BB963 :
  %8787 = add nuw nsw i32 %8778, 1, !spirv.Decorations !631		; visa id: 11459
  br label %._crit_edge382, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11460

8788:                                             ; preds = %._crit_edge382
; BB964 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11462
  %8789 = load i64, i64* %5998, align 8		; visa id: 11462
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11463
  %8790 = icmp slt i32 %6168, %const_reg_dword
  %8791 = icmp slt i32 %8599, %const_reg_dword1		; visa id: 11463
  %8792 = and i1 %8790, %8791		; visa id: 11464
  br i1 %8792, label %8793, label %..preheader1.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11466

..preheader1.11_crit_edge:                        ; preds = %8788
; BB:
  br label %.preheader1.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8793:                                             ; preds = %8788
; BB966 :
  %8794 = bitcast i64 %8789 to <2 x i32>		; visa id: 11468
  %8795 = extractelement <2 x i32> %8794, i32 0		; visa id: 11470
  %8796 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8795, i32 1
  %8797 = bitcast <2 x i32> %8796 to i64		; visa id: 11470
  %8798 = ashr exact i64 %8797, 32		; visa id: 11471
  %8799 = bitcast i64 %8798 to <2 x i32>		; visa id: 11472
  %8800 = extractelement <2 x i32> %8799, i32 0		; visa id: 11476
  %8801 = extractelement <2 x i32> %8799, i32 1		; visa id: 11476
  %8802 = ashr i64 %8789, 32		; visa id: 11476
  %8803 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8800, i32 %8801, i32 %50, i32 %51)
  %8804 = extractvalue { i32, i32 } %8803, 0		; visa id: 11477
  %8805 = extractvalue { i32, i32 } %8803, 1		; visa id: 11477
  %8806 = insertelement <2 x i32> undef, i32 %8804, i32 0		; visa id: 11484
  %8807 = insertelement <2 x i32> %8806, i32 %8805, i32 1		; visa id: 11485
  %8808 = bitcast <2 x i32> %8807 to i64		; visa id: 11486
  %8809 = add nsw i64 %8808, %8802, !spirv.Decorations !649		; visa id: 11490
  %8810 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !618		; visa id: 11491
  br i1 %86, label %8816, label %8811, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11492

8811:                                             ; preds = %8793
; BB967 :
  %8812 = shl i64 %8809, 2		; visa id: 11494
  %8813 = add i64 %.in, %8812		; visa id: 11495
  %8814 = inttoptr i64 %8813 to float addrspace(4)*		; visa id: 11496
  %8815 = addrspacecast float addrspace(4)* %8814 to float addrspace(1)*		; visa id: 11496
  store float %8810, float addrspace(1)* %8815, align 4		; visa id: 11497
  br label %.preheader1.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11498

8816:                                             ; preds = %8793
; BB968 :
  %8817 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8800, i32 %8801, i32 %47, i32 %48)
  %8818 = extractvalue { i32, i32 } %8817, 0		; visa id: 11500
  %8819 = extractvalue { i32, i32 } %8817, 1		; visa id: 11500
  %8820 = insertelement <2 x i32> undef, i32 %8818, i32 0		; visa id: 11507
  %8821 = insertelement <2 x i32> %8820, i32 %8819, i32 1		; visa id: 11508
  %8822 = bitcast <2 x i32> %8821 to i64		; visa id: 11509
  %8823 = shl i64 %8822, 2		; visa id: 11513
  %8824 = add i64 %.in399, %8823		; visa id: 11514
  %8825 = shl nsw i64 %8802, 2		; visa id: 11515
  %8826 = add i64 %8824, %8825		; visa id: 11516
  %8827 = inttoptr i64 %8826 to float addrspace(4)*		; visa id: 11517
  %8828 = addrspacecast float addrspace(4)* %8827 to float addrspace(1)*		; visa id: 11517
  %8829 = load float, float addrspace(1)* %8828, align 4		; visa id: 11518
  %8830 = fmul reassoc nsz arcp contract float %8829, %4, !spirv.Decorations !618		; visa id: 11519
  %8831 = fadd reassoc nsz arcp contract float %8810, %8830, !spirv.Decorations !618		; visa id: 11520
  %8832 = shl i64 %8809, 2		; visa id: 11521
  %8833 = add i64 %.in, %8832		; visa id: 11522
  %8834 = inttoptr i64 %8833 to float addrspace(4)*		; visa id: 11523
  %8835 = addrspacecast float addrspace(4)* %8834 to float addrspace(1)*		; visa id: 11523
  store float %8831, float addrspace(1)* %8835, align 4		; visa id: 11524
  br label %.preheader1.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11525

.preheader1.11:                                   ; preds = %..preheader1.11_crit_edge, %8816, %8811
; BB969 :
  %8836 = add i32 %69, 12		; visa id: 11526
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11527
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11527
  %8837 = insertelement <2 x i32> %5984, i32 %8836, i64 1		; visa id: 11527
  store <2 x i32> %8837, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11530
  br label %._crit_edge383, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11532

._crit_edge383:                                   ; preds = %._crit_edge383.._crit_edge383_crit_edge, %.preheader1.11
; BB970 :
  %8838 = phi i32 [ 0, %.preheader1.11 ], [ %8847, %._crit_edge383.._crit_edge383_crit_edge ]
  %8839 = zext i32 %8838 to i64		; visa id: 11533
  %8840 = shl nuw nsw i64 %8839, 2		; visa id: 11534
  %8841 = add i64 %5982, %8840		; visa id: 11535
  %8842 = inttoptr i64 %8841 to i32*		; visa id: 11536
  %8843 = load i32, i32* %8842, align 4, !noalias !642		; visa id: 11536
  %8844 = add i64 %5978, %8840		; visa id: 11537
  %8845 = inttoptr i64 %8844 to i32*		; visa id: 11538
  store i32 %8843, i32* %8845, align 4, !alias.scope !642		; visa id: 11538
  %8846 = icmp eq i32 %8838, 0		; visa id: 11539
  br i1 %8846, label %._crit_edge383.._crit_edge383_crit_edge, label %8848, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11540

._crit_edge383.._crit_edge383_crit_edge:          ; preds = %._crit_edge383
; BB971 :
  %8847 = add nuw nsw i32 %8838, 1, !spirv.Decorations !631		; visa id: 11542
  br label %._crit_edge383, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11543

8848:                                             ; preds = %._crit_edge383
; BB972 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11545
  %8849 = load i64, i64* %5998, align 8		; visa id: 11545
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11546
  %8850 = icmp slt i32 %8836, %const_reg_dword1		; visa id: 11546
  %8851 = icmp slt i32 %65, %const_reg_dword
  %8852 = and i1 %8851, %8850		; visa id: 11547
  br i1 %8852, label %8853, label %.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11549

.._crit_edge70.12_crit_edge:                      ; preds = %8848
; BB:
  br label %._crit_edge70.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8853:                                             ; preds = %8848
; BB974 :
  %8854 = bitcast i64 %8849 to <2 x i32>		; visa id: 11551
  %8855 = extractelement <2 x i32> %8854, i32 0		; visa id: 11553
  %8856 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8855, i32 1
  %8857 = bitcast <2 x i32> %8856 to i64		; visa id: 11553
  %8858 = ashr exact i64 %8857, 32		; visa id: 11554
  %8859 = bitcast i64 %8858 to <2 x i32>		; visa id: 11555
  %8860 = extractelement <2 x i32> %8859, i32 0		; visa id: 11559
  %8861 = extractelement <2 x i32> %8859, i32 1		; visa id: 11559
  %8862 = ashr i64 %8849, 32		; visa id: 11559
  %8863 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8860, i32 %8861, i32 %50, i32 %51)
  %8864 = extractvalue { i32, i32 } %8863, 0		; visa id: 11560
  %8865 = extractvalue { i32, i32 } %8863, 1		; visa id: 11560
  %8866 = insertelement <2 x i32> undef, i32 %8864, i32 0		; visa id: 11567
  %8867 = insertelement <2 x i32> %8866, i32 %8865, i32 1		; visa id: 11568
  %8868 = bitcast <2 x i32> %8867 to i64		; visa id: 11569
  %8869 = add nsw i64 %8868, %8862, !spirv.Decorations !649		; visa id: 11573
  %8870 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !618		; visa id: 11574
  br i1 %86, label %8876, label %8871, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11575

8871:                                             ; preds = %8853
; BB975 :
  %8872 = shl i64 %8869, 2		; visa id: 11577
  %8873 = add i64 %.in, %8872		; visa id: 11578
  %8874 = inttoptr i64 %8873 to float addrspace(4)*		; visa id: 11579
  %8875 = addrspacecast float addrspace(4)* %8874 to float addrspace(1)*		; visa id: 11579
  store float %8870, float addrspace(1)* %8875, align 4		; visa id: 11580
  br label %._crit_edge70.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11581

8876:                                             ; preds = %8853
; BB976 :
  %8877 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8860, i32 %8861, i32 %47, i32 %48)
  %8878 = extractvalue { i32, i32 } %8877, 0		; visa id: 11583
  %8879 = extractvalue { i32, i32 } %8877, 1		; visa id: 11583
  %8880 = insertelement <2 x i32> undef, i32 %8878, i32 0		; visa id: 11590
  %8881 = insertelement <2 x i32> %8880, i32 %8879, i32 1		; visa id: 11591
  %8882 = bitcast <2 x i32> %8881 to i64		; visa id: 11592
  %8883 = shl i64 %8882, 2		; visa id: 11596
  %8884 = add i64 %.in399, %8883		; visa id: 11597
  %8885 = shl nsw i64 %8862, 2		; visa id: 11598
  %8886 = add i64 %8884, %8885		; visa id: 11599
  %8887 = inttoptr i64 %8886 to float addrspace(4)*		; visa id: 11600
  %8888 = addrspacecast float addrspace(4)* %8887 to float addrspace(1)*		; visa id: 11600
  %8889 = load float, float addrspace(1)* %8888, align 4		; visa id: 11601
  %8890 = fmul reassoc nsz arcp contract float %8889, %4, !spirv.Decorations !618		; visa id: 11602
  %8891 = fadd reassoc nsz arcp contract float %8870, %8890, !spirv.Decorations !618		; visa id: 11603
  %8892 = shl i64 %8869, 2		; visa id: 11604
  %8893 = add i64 %.in, %8892		; visa id: 11605
  %8894 = inttoptr i64 %8893 to float addrspace(4)*		; visa id: 11606
  %8895 = addrspacecast float addrspace(4)* %8894 to float addrspace(1)*		; visa id: 11606
  store float %8891, float addrspace(1)* %8895, align 4		; visa id: 11607
  br label %._crit_edge70.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11608

._crit_edge70.12:                                 ; preds = %.._crit_edge70.12_crit_edge, %8876, %8871
; BB977 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11609
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11609
  %8896 = insertelement <2 x i32> %6047, i32 %8836, i64 1		; visa id: 11609
  store <2 x i32> %8896, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11612
  br label %._crit_edge384, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11614

._crit_edge384:                                   ; preds = %._crit_edge384.._crit_edge384_crit_edge, %._crit_edge70.12
; BB978 :
  %8897 = phi i32 [ 0, %._crit_edge70.12 ], [ %8906, %._crit_edge384.._crit_edge384_crit_edge ]
  %8898 = zext i32 %8897 to i64		; visa id: 11615
  %8899 = shl nuw nsw i64 %8898, 2		; visa id: 11616
  %8900 = add i64 %5982, %8899		; visa id: 11617
  %8901 = inttoptr i64 %8900 to i32*		; visa id: 11618
  %8902 = load i32, i32* %8901, align 4, !noalias !642		; visa id: 11618
  %8903 = add i64 %5978, %8899		; visa id: 11619
  %8904 = inttoptr i64 %8903 to i32*		; visa id: 11620
  store i32 %8902, i32* %8904, align 4, !alias.scope !642		; visa id: 11620
  %8905 = icmp eq i32 %8897, 0		; visa id: 11621
  br i1 %8905, label %._crit_edge384.._crit_edge384_crit_edge, label %8907, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11622

._crit_edge384.._crit_edge384_crit_edge:          ; preds = %._crit_edge384
; BB979 :
  %8906 = add nuw nsw i32 %8897, 1, !spirv.Decorations !631		; visa id: 11624
  br label %._crit_edge384, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11625

8907:                                             ; preds = %._crit_edge384
; BB980 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11627
  %8908 = load i64, i64* %5998, align 8		; visa id: 11627
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11628
  %8909 = icmp slt i32 %6046, %const_reg_dword
  %8910 = icmp slt i32 %8836, %const_reg_dword1		; visa id: 11628
  %8911 = and i1 %8909, %8910		; visa id: 11629
  br i1 %8911, label %8912, label %.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11631

.._crit_edge70.1.12_crit_edge:                    ; preds = %8907
; BB:
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8912:                                             ; preds = %8907
; BB982 :
  %8913 = bitcast i64 %8908 to <2 x i32>		; visa id: 11633
  %8914 = extractelement <2 x i32> %8913, i32 0		; visa id: 11635
  %8915 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8914, i32 1
  %8916 = bitcast <2 x i32> %8915 to i64		; visa id: 11635
  %8917 = ashr exact i64 %8916, 32		; visa id: 11636
  %8918 = bitcast i64 %8917 to <2 x i32>		; visa id: 11637
  %8919 = extractelement <2 x i32> %8918, i32 0		; visa id: 11641
  %8920 = extractelement <2 x i32> %8918, i32 1		; visa id: 11641
  %8921 = ashr i64 %8908, 32		; visa id: 11641
  %8922 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8919, i32 %8920, i32 %50, i32 %51)
  %8923 = extractvalue { i32, i32 } %8922, 0		; visa id: 11642
  %8924 = extractvalue { i32, i32 } %8922, 1		; visa id: 11642
  %8925 = insertelement <2 x i32> undef, i32 %8923, i32 0		; visa id: 11649
  %8926 = insertelement <2 x i32> %8925, i32 %8924, i32 1		; visa id: 11650
  %8927 = bitcast <2 x i32> %8926 to i64		; visa id: 11651
  %8928 = add nsw i64 %8927, %8921, !spirv.Decorations !649		; visa id: 11655
  %8929 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !618		; visa id: 11656
  br i1 %86, label %8935, label %8930, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11657

8930:                                             ; preds = %8912
; BB983 :
  %8931 = shl i64 %8928, 2		; visa id: 11659
  %8932 = add i64 %.in, %8931		; visa id: 11660
  %8933 = inttoptr i64 %8932 to float addrspace(4)*		; visa id: 11661
  %8934 = addrspacecast float addrspace(4)* %8933 to float addrspace(1)*		; visa id: 11661
  store float %8929, float addrspace(1)* %8934, align 4		; visa id: 11662
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11663

8935:                                             ; preds = %8912
; BB984 :
  %8936 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8919, i32 %8920, i32 %47, i32 %48)
  %8937 = extractvalue { i32, i32 } %8936, 0		; visa id: 11665
  %8938 = extractvalue { i32, i32 } %8936, 1		; visa id: 11665
  %8939 = insertelement <2 x i32> undef, i32 %8937, i32 0		; visa id: 11672
  %8940 = insertelement <2 x i32> %8939, i32 %8938, i32 1		; visa id: 11673
  %8941 = bitcast <2 x i32> %8940 to i64		; visa id: 11674
  %8942 = shl i64 %8941, 2		; visa id: 11678
  %8943 = add i64 %.in399, %8942		; visa id: 11679
  %8944 = shl nsw i64 %8921, 2		; visa id: 11680
  %8945 = add i64 %8943, %8944		; visa id: 11681
  %8946 = inttoptr i64 %8945 to float addrspace(4)*		; visa id: 11682
  %8947 = addrspacecast float addrspace(4)* %8946 to float addrspace(1)*		; visa id: 11682
  %8948 = load float, float addrspace(1)* %8947, align 4		; visa id: 11683
  %8949 = fmul reassoc nsz arcp contract float %8948, %4, !spirv.Decorations !618		; visa id: 11684
  %8950 = fadd reassoc nsz arcp contract float %8929, %8949, !spirv.Decorations !618		; visa id: 11685
  %8951 = shl i64 %8928, 2		; visa id: 11686
  %8952 = add i64 %.in, %8951		; visa id: 11687
  %8953 = inttoptr i64 %8952 to float addrspace(4)*		; visa id: 11688
  %8954 = addrspacecast float addrspace(4)* %8953 to float addrspace(1)*		; visa id: 11688
  store float %8950, float addrspace(1)* %8954, align 4		; visa id: 11689
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11690

._crit_edge70.1.12:                               ; preds = %.._crit_edge70.1.12_crit_edge, %8935, %8930
; BB985 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11691
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11691
  %8955 = insertelement <2 x i32> %6108, i32 %8836, i64 1		; visa id: 11691
  store <2 x i32> %8955, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11694
  br label %._crit_edge385, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11696

._crit_edge385:                                   ; preds = %._crit_edge385.._crit_edge385_crit_edge, %._crit_edge70.1.12
; BB986 :
  %8956 = phi i32 [ 0, %._crit_edge70.1.12 ], [ %8965, %._crit_edge385.._crit_edge385_crit_edge ]
  %8957 = zext i32 %8956 to i64		; visa id: 11697
  %8958 = shl nuw nsw i64 %8957, 2		; visa id: 11698
  %8959 = add i64 %5982, %8958		; visa id: 11699
  %8960 = inttoptr i64 %8959 to i32*		; visa id: 11700
  %8961 = load i32, i32* %8960, align 4, !noalias !642		; visa id: 11700
  %8962 = add i64 %5978, %8958		; visa id: 11701
  %8963 = inttoptr i64 %8962 to i32*		; visa id: 11702
  store i32 %8961, i32* %8963, align 4, !alias.scope !642		; visa id: 11702
  %8964 = icmp eq i32 %8956, 0		; visa id: 11703
  br i1 %8964, label %._crit_edge385.._crit_edge385_crit_edge, label %8966, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11704

._crit_edge385.._crit_edge385_crit_edge:          ; preds = %._crit_edge385
; BB987 :
  %8965 = add nuw nsw i32 %8956, 1, !spirv.Decorations !631		; visa id: 11706
  br label %._crit_edge385, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11707

8966:                                             ; preds = %._crit_edge385
; BB988 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11709
  %8967 = load i64, i64* %5998, align 8		; visa id: 11709
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11710
  %8968 = icmp slt i32 %6107, %const_reg_dword
  %8969 = icmp slt i32 %8836, %const_reg_dword1		; visa id: 11710
  %8970 = and i1 %8968, %8969		; visa id: 11711
  br i1 %8970, label %8971, label %.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11713

.._crit_edge70.2.12_crit_edge:                    ; preds = %8966
; BB:
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8971:                                             ; preds = %8966
; BB990 :
  %8972 = bitcast i64 %8967 to <2 x i32>		; visa id: 11715
  %8973 = extractelement <2 x i32> %8972, i32 0		; visa id: 11717
  %8974 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8973, i32 1
  %8975 = bitcast <2 x i32> %8974 to i64		; visa id: 11717
  %8976 = ashr exact i64 %8975, 32		; visa id: 11718
  %8977 = bitcast i64 %8976 to <2 x i32>		; visa id: 11719
  %8978 = extractelement <2 x i32> %8977, i32 0		; visa id: 11723
  %8979 = extractelement <2 x i32> %8977, i32 1		; visa id: 11723
  %8980 = ashr i64 %8967, 32		; visa id: 11723
  %8981 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8978, i32 %8979, i32 %50, i32 %51)
  %8982 = extractvalue { i32, i32 } %8981, 0		; visa id: 11724
  %8983 = extractvalue { i32, i32 } %8981, 1		; visa id: 11724
  %8984 = insertelement <2 x i32> undef, i32 %8982, i32 0		; visa id: 11731
  %8985 = insertelement <2 x i32> %8984, i32 %8983, i32 1		; visa id: 11732
  %8986 = bitcast <2 x i32> %8985 to i64		; visa id: 11733
  %8987 = add nsw i64 %8986, %8980, !spirv.Decorations !649		; visa id: 11737
  %8988 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !618		; visa id: 11738
  br i1 %86, label %8994, label %8989, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11739

8989:                                             ; preds = %8971
; BB991 :
  %8990 = shl i64 %8987, 2		; visa id: 11741
  %8991 = add i64 %.in, %8990		; visa id: 11742
  %8992 = inttoptr i64 %8991 to float addrspace(4)*		; visa id: 11743
  %8993 = addrspacecast float addrspace(4)* %8992 to float addrspace(1)*		; visa id: 11743
  store float %8988, float addrspace(1)* %8993, align 4		; visa id: 11744
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11745

8994:                                             ; preds = %8971
; BB992 :
  %8995 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8978, i32 %8979, i32 %47, i32 %48)
  %8996 = extractvalue { i32, i32 } %8995, 0		; visa id: 11747
  %8997 = extractvalue { i32, i32 } %8995, 1		; visa id: 11747
  %8998 = insertelement <2 x i32> undef, i32 %8996, i32 0		; visa id: 11754
  %8999 = insertelement <2 x i32> %8998, i32 %8997, i32 1		; visa id: 11755
  %9000 = bitcast <2 x i32> %8999 to i64		; visa id: 11756
  %9001 = shl i64 %9000, 2		; visa id: 11760
  %9002 = add i64 %.in399, %9001		; visa id: 11761
  %9003 = shl nsw i64 %8980, 2		; visa id: 11762
  %9004 = add i64 %9002, %9003		; visa id: 11763
  %9005 = inttoptr i64 %9004 to float addrspace(4)*		; visa id: 11764
  %9006 = addrspacecast float addrspace(4)* %9005 to float addrspace(1)*		; visa id: 11764
  %9007 = load float, float addrspace(1)* %9006, align 4		; visa id: 11765
  %9008 = fmul reassoc nsz arcp contract float %9007, %4, !spirv.Decorations !618		; visa id: 11766
  %9009 = fadd reassoc nsz arcp contract float %8988, %9008, !spirv.Decorations !618		; visa id: 11767
  %9010 = shl i64 %8987, 2		; visa id: 11768
  %9011 = add i64 %.in, %9010		; visa id: 11769
  %9012 = inttoptr i64 %9011 to float addrspace(4)*		; visa id: 11770
  %9013 = addrspacecast float addrspace(4)* %9012 to float addrspace(1)*		; visa id: 11770
  store float %9009, float addrspace(1)* %9013, align 4		; visa id: 11771
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11772

._crit_edge70.2.12:                               ; preds = %.._crit_edge70.2.12_crit_edge, %8994, %8989
; BB993 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11773
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11773
  %9014 = insertelement <2 x i32> %6169, i32 %8836, i64 1		; visa id: 11773
  store <2 x i32> %9014, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11776
  br label %._crit_edge386, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11778

._crit_edge386:                                   ; preds = %._crit_edge386.._crit_edge386_crit_edge, %._crit_edge70.2.12
; BB994 :
  %9015 = phi i32 [ 0, %._crit_edge70.2.12 ], [ %9024, %._crit_edge386.._crit_edge386_crit_edge ]
  %9016 = zext i32 %9015 to i64		; visa id: 11779
  %9017 = shl nuw nsw i64 %9016, 2		; visa id: 11780
  %9018 = add i64 %5982, %9017		; visa id: 11781
  %9019 = inttoptr i64 %9018 to i32*		; visa id: 11782
  %9020 = load i32, i32* %9019, align 4, !noalias !642		; visa id: 11782
  %9021 = add i64 %5978, %9017		; visa id: 11783
  %9022 = inttoptr i64 %9021 to i32*		; visa id: 11784
  store i32 %9020, i32* %9022, align 4, !alias.scope !642		; visa id: 11784
  %9023 = icmp eq i32 %9015, 0		; visa id: 11785
  br i1 %9023, label %._crit_edge386.._crit_edge386_crit_edge, label %9025, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11786

._crit_edge386.._crit_edge386_crit_edge:          ; preds = %._crit_edge386
; BB995 :
  %9024 = add nuw nsw i32 %9015, 1, !spirv.Decorations !631		; visa id: 11788
  br label %._crit_edge386, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11789

9025:                                             ; preds = %._crit_edge386
; BB996 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11791
  %9026 = load i64, i64* %5998, align 8		; visa id: 11791
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11792
  %9027 = icmp slt i32 %6168, %const_reg_dword
  %9028 = icmp slt i32 %8836, %const_reg_dword1		; visa id: 11792
  %9029 = and i1 %9027, %9028		; visa id: 11793
  br i1 %9029, label %9030, label %..preheader1.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11795

..preheader1.12_crit_edge:                        ; preds = %9025
; BB:
  br label %.preheader1.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9030:                                             ; preds = %9025
; BB998 :
  %9031 = bitcast i64 %9026 to <2 x i32>		; visa id: 11797
  %9032 = extractelement <2 x i32> %9031, i32 0		; visa id: 11799
  %9033 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9032, i32 1
  %9034 = bitcast <2 x i32> %9033 to i64		; visa id: 11799
  %9035 = ashr exact i64 %9034, 32		; visa id: 11800
  %9036 = bitcast i64 %9035 to <2 x i32>		; visa id: 11801
  %9037 = extractelement <2 x i32> %9036, i32 0		; visa id: 11805
  %9038 = extractelement <2 x i32> %9036, i32 1		; visa id: 11805
  %9039 = ashr i64 %9026, 32		; visa id: 11805
  %9040 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9037, i32 %9038, i32 %50, i32 %51)
  %9041 = extractvalue { i32, i32 } %9040, 0		; visa id: 11806
  %9042 = extractvalue { i32, i32 } %9040, 1		; visa id: 11806
  %9043 = insertelement <2 x i32> undef, i32 %9041, i32 0		; visa id: 11813
  %9044 = insertelement <2 x i32> %9043, i32 %9042, i32 1		; visa id: 11814
  %9045 = bitcast <2 x i32> %9044 to i64		; visa id: 11815
  %9046 = add nsw i64 %9045, %9039, !spirv.Decorations !649		; visa id: 11819
  %9047 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !618		; visa id: 11820
  br i1 %86, label %9053, label %9048, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11821

9048:                                             ; preds = %9030
; BB999 :
  %9049 = shl i64 %9046, 2		; visa id: 11823
  %9050 = add i64 %.in, %9049		; visa id: 11824
  %9051 = inttoptr i64 %9050 to float addrspace(4)*		; visa id: 11825
  %9052 = addrspacecast float addrspace(4)* %9051 to float addrspace(1)*		; visa id: 11825
  store float %9047, float addrspace(1)* %9052, align 4		; visa id: 11826
  br label %.preheader1.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11827

9053:                                             ; preds = %9030
; BB1000 :
  %9054 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9037, i32 %9038, i32 %47, i32 %48)
  %9055 = extractvalue { i32, i32 } %9054, 0		; visa id: 11829
  %9056 = extractvalue { i32, i32 } %9054, 1		; visa id: 11829
  %9057 = insertelement <2 x i32> undef, i32 %9055, i32 0		; visa id: 11836
  %9058 = insertelement <2 x i32> %9057, i32 %9056, i32 1		; visa id: 11837
  %9059 = bitcast <2 x i32> %9058 to i64		; visa id: 11838
  %9060 = shl i64 %9059, 2		; visa id: 11842
  %9061 = add i64 %.in399, %9060		; visa id: 11843
  %9062 = shl nsw i64 %9039, 2		; visa id: 11844
  %9063 = add i64 %9061, %9062		; visa id: 11845
  %9064 = inttoptr i64 %9063 to float addrspace(4)*		; visa id: 11846
  %9065 = addrspacecast float addrspace(4)* %9064 to float addrspace(1)*		; visa id: 11846
  %9066 = load float, float addrspace(1)* %9065, align 4		; visa id: 11847
  %9067 = fmul reassoc nsz arcp contract float %9066, %4, !spirv.Decorations !618		; visa id: 11848
  %9068 = fadd reassoc nsz arcp contract float %9047, %9067, !spirv.Decorations !618		; visa id: 11849
  %9069 = shl i64 %9046, 2		; visa id: 11850
  %9070 = add i64 %.in, %9069		; visa id: 11851
  %9071 = inttoptr i64 %9070 to float addrspace(4)*		; visa id: 11852
  %9072 = addrspacecast float addrspace(4)* %9071 to float addrspace(1)*		; visa id: 11852
  store float %9068, float addrspace(1)* %9072, align 4		; visa id: 11853
  br label %.preheader1.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11854

.preheader1.12:                                   ; preds = %..preheader1.12_crit_edge, %9053, %9048
; BB1001 :
  %9073 = add i32 %69, 13		; visa id: 11855
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11856
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11856
  %9074 = insertelement <2 x i32> %5984, i32 %9073, i64 1		; visa id: 11856
  store <2 x i32> %9074, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11859
  br label %._crit_edge387, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11861

._crit_edge387:                                   ; preds = %._crit_edge387.._crit_edge387_crit_edge, %.preheader1.12
; BB1002 :
  %9075 = phi i32 [ 0, %.preheader1.12 ], [ %9084, %._crit_edge387.._crit_edge387_crit_edge ]
  %9076 = zext i32 %9075 to i64		; visa id: 11862
  %9077 = shl nuw nsw i64 %9076, 2		; visa id: 11863
  %9078 = add i64 %5982, %9077		; visa id: 11864
  %9079 = inttoptr i64 %9078 to i32*		; visa id: 11865
  %9080 = load i32, i32* %9079, align 4, !noalias !642		; visa id: 11865
  %9081 = add i64 %5978, %9077		; visa id: 11866
  %9082 = inttoptr i64 %9081 to i32*		; visa id: 11867
  store i32 %9080, i32* %9082, align 4, !alias.scope !642		; visa id: 11867
  %9083 = icmp eq i32 %9075, 0		; visa id: 11868
  br i1 %9083, label %._crit_edge387.._crit_edge387_crit_edge, label %9085, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11869

._crit_edge387.._crit_edge387_crit_edge:          ; preds = %._crit_edge387
; BB1003 :
  %9084 = add nuw nsw i32 %9075, 1, !spirv.Decorations !631		; visa id: 11871
  br label %._crit_edge387, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11872

9085:                                             ; preds = %._crit_edge387
; BB1004 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11874
  %9086 = load i64, i64* %5998, align 8		; visa id: 11874
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11875
  %9087 = icmp slt i32 %9073, %const_reg_dword1		; visa id: 11875
  %9088 = icmp slt i32 %65, %const_reg_dword
  %9089 = and i1 %9088, %9087		; visa id: 11876
  br i1 %9089, label %9090, label %.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11878

.._crit_edge70.13_crit_edge:                      ; preds = %9085
; BB:
  br label %._crit_edge70.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9090:                                             ; preds = %9085
; BB1006 :
  %9091 = bitcast i64 %9086 to <2 x i32>		; visa id: 11880
  %9092 = extractelement <2 x i32> %9091, i32 0		; visa id: 11882
  %9093 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9092, i32 1
  %9094 = bitcast <2 x i32> %9093 to i64		; visa id: 11882
  %9095 = ashr exact i64 %9094, 32		; visa id: 11883
  %9096 = bitcast i64 %9095 to <2 x i32>		; visa id: 11884
  %9097 = extractelement <2 x i32> %9096, i32 0		; visa id: 11888
  %9098 = extractelement <2 x i32> %9096, i32 1		; visa id: 11888
  %9099 = ashr i64 %9086, 32		; visa id: 11888
  %9100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9097, i32 %9098, i32 %50, i32 %51)
  %9101 = extractvalue { i32, i32 } %9100, 0		; visa id: 11889
  %9102 = extractvalue { i32, i32 } %9100, 1		; visa id: 11889
  %9103 = insertelement <2 x i32> undef, i32 %9101, i32 0		; visa id: 11896
  %9104 = insertelement <2 x i32> %9103, i32 %9102, i32 1		; visa id: 11897
  %9105 = bitcast <2 x i32> %9104 to i64		; visa id: 11898
  %9106 = add nsw i64 %9105, %9099, !spirv.Decorations !649		; visa id: 11902
  %9107 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !618		; visa id: 11903
  br i1 %86, label %9113, label %9108, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11904

9108:                                             ; preds = %9090
; BB1007 :
  %9109 = shl i64 %9106, 2		; visa id: 11906
  %9110 = add i64 %.in, %9109		; visa id: 11907
  %9111 = inttoptr i64 %9110 to float addrspace(4)*		; visa id: 11908
  %9112 = addrspacecast float addrspace(4)* %9111 to float addrspace(1)*		; visa id: 11908
  store float %9107, float addrspace(1)* %9112, align 4		; visa id: 11909
  br label %._crit_edge70.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11910

9113:                                             ; preds = %9090
; BB1008 :
  %9114 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9097, i32 %9098, i32 %47, i32 %48)
  %9115 = extractvalue { i32, i32 } %9114, 0		; visa id: 11912
  %9116 = extractvalue { i32, i32 } %9114, 1		; visa id: 11912
  %9117 = insertelement <2 x i32> undef, i32 %9115, i32 0		; visa id: 11919
  %9118 = insertelement <2 x i32> %9117, i32 %9116, i32 1		; visa id: 11920
  %9119 = bitcast <2 x i32> %9118 to i64		; visa id: 11921
  %9120 = shl i64 %9119, 2		; visa id: 11925
  %9121 = add i64 %.in399, %9120		; visa id: 11926
  %9122 = shl nsw i64 %9099, 2		; visa id: 11927
  %9123 = add i64 %9121, %9122		; visa id: 11928
  %9124 = inttoptr i64 %9123 to float addrspace(4)*		; visa id: 11929
  %9125 = addrspacecast float addrspace(4)* %9124 to float addrspace(1)*		; visa id: 11929
  %9126 = load float, float addrspace(1)* %9125, align 4		; visa id: 11930
  %9127 = fmul reassoc nsz arcp contract float %9126, %4, !spirv.Decorations !618		; visa id: 11931
  %9128 = fadd reassoc nsz arcp contract float %9107, %9127, !spirv.Decorations !618		; visa id: 11932
  %9129 = shl i64 %9106, 2		; visa id: 11933
  %9130 = add i64 %.in, %9129		; visa id: 11934
  %9131 = inttoptr i64 %9130 to float addrspace(4)*		; visa id: 11935
  %9132 = addrspacecast float addrspace(4)* %9131 to float addrspace(1)*		; visa id: 11935
  store float %9128, float addrspace(1)* %9132, align 4		; visa id: 11936
  br label %._crit_edge70.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11937

._crit_edge70.13:                                 ; preds = %.._crit_edge70.13_crit_edge, %9113, %9108
; BB1009 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11938
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11938
  %9133 = insertelement <2 x i32> %6047, i32 %9073, i64 1		; visa id: 11938
  store <2 x i32> %9133, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 11941
  br label %._crit_edge388, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11943

._crit_edge388:                                   ; preds = %._crit_edge388.._crit_edge388_crit_edge, %._crit_edge70.13
; BB1010 :
  %9134 = phi i32 [ 0, %._crit_edge70.13 ], [ %9143, %._crit_edge388.._crit_edge388_crit_edge ]
  %9135 = zext i32 %9134 to i64		; visa id: 11944
  %9136 = shl nuw nsw i64 %9135, 2		; visa id: 11945
  %9137 = add i64 %5982, %9136		; visa id: 11946
  %9138 = inttoptr i64 %9137 to i32*		; visa id: 11947
  %9139 = load i32, i32* %9138, align 4, !noalias !642		; visa id: 11947
  %9140 = add i64 %5978, %9136		; visa id: 11948
  %9141 = inttoptr i64 %9140 to i32*		; visa id: 11949
  store i32 %9139, i32* %9141, align 4, !alias.scope !642		; visa id: 11949
  %9142 = icmp eq i32 %9134, 0		; visa id: 11950
  br i1 %9142, label %._crit_edge388.._crit_edge388_crit_edge, label %9144, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11951

._crit_edge388.._crit_edge388_crit_edge:          ; preds = %._crit_edge388
; BB1011 :
  %9143 = add nuw nsw i32 %9134, 1, !spirv.Decorations !631		; visa id: 11953
  br label %._crit_edge388, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11954

9144:                                             ; preds = %._crit_edge388
; BB1012 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 11956
  %9145 = load i64, i64* %5998, align 8		; visa id: 11956
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 11957
  %9146 = icmp slt i32 %6046, %const_reg_dword
  %9147 = icmp slt i32 %9073, %const_reg_dword1		; visa id: 11957
  %9148 = and i1 %9146, %9147		; visa id: 11958
  br i1 %9148, label %9149, label %.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11960

.._crit_edge70.1.13_crit_edge:                    ; preds = %9144
; BB:
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9149:                                             ; preds = %9144
; BB1014 :
  %9150 = bitcast i64 %9145 to <2 x i32>		; visa id: 11962
  %9151 = extractelement <2 x i32> %9150, i32 0		; visa id: 11964
  %9152 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9151, i32 1
  %9153 = bitcast <2 x i32> %9152 to i64		; visa id: 11964
  %9154 = ashr exact i64 %9153, 32		; visa id: 11965
  %9155 = bitcast i64 %9154 to <2 x i32>		; visa id: 11966
  %9156 = extractelement <2 x i32> %9155, i32 0		; visa id: 11970
  %9157 = extractelement <2 x i32> %9155, i32 1		; visa id: 11970
  %9158 = ashr i64 %9145, 32		; visa id: 11970
  %9159 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9156, i32 %9157, i32 %50, i32 %51)
  %9160 = extractvalue { i32, i32 } %9159, 0		; visa id: 11971
  %9161 = extractvalue { i32, i32 } %9159, 1		; visa id: 11971
  %9162 = insertelement <2 x i32> undef, i32 %9160, i32 0		; visa id: 11978
  %9163 = insertelement <2 x i32> %9162, i32 %9161, i32 1		; visa id: 11979
  %9164 = bitcast <2 x i32> %9163 to i64		; visa id: 11980
  %9165 = add nsw i64 %9164, %9158, !spirv.Decorations !649		; visa id: 11984
  %9166 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !618		; visa id: 11985
  br i1 %86, label %9172, label %9167, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11986

9167:                                             ; preds = %9149
; BB1015 :
  %9168 = shl i64 %9165, 2		; visa id: 11988
  %9169 = add i64 %.in, %9168		; visa id: 11989
  %9170 = inttoptr i64 %9169 to float addrspace(4)*		; visa id: 11990
  %9171 = addrspacecast float addrspace(4)* %9170 to float addrspace(1)*		; visa id: 11990
  store float %9166, float addrspace(1)* %9171, align 4		; visa id: 11991
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11992

9172:                                             ; preds = %9149
; BB1016 :
  %9173 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9156, i32 %9157, i32 %47, i32 %48)
  %9174 = extractvalue { i32, i32 } %9173, 0		; visa id: 11994
  %9175 = extractvalue { i32, i32 } %9173, 1		; visa id: 11994
  %9176 = insertelement <2 x i32> undef, i32 %9174, i32 0		; visa id: 12001
  %9177 = insertelement <2 x i32> %9176, i32 %9175, i32 1		; visa id: 12002
  %9178 = bitcast <2 x i32> %9177 to i64		; visa id: 12003
  %9179 = shl i64 %9178, 2		; visa id: 12007
  %9180 = add i64 %.in399, %9179		; visa id: 12008
  %9181 = shl nsw i64 %9158, 2		; visa id: 12009
  %9182 = add i64 %9180, %9181		; visa id: 12010
  %9183 = inttoptr i64 %9182 to float addrspace(4)*		; visa id: 12011
  %9184 = addrspacecast float addrspace(4)* %9183 to float addrspace(1)*		; visa id: 12011
  %9185 = load float, float addrspace(1)* %9184, align 4		; visa id: 12012
  %9186 = fmul reassoc nsz arcp contract float %9185, %4, !spirv.Decorations !618		; visa id: 12013
  %9187 = fadd reassoc nsz arcp contract float %9166, %9186, !spirv.Decorations !618		; visa id: 12014
  %9188 = shl i64 %9165, 2		; visa id: 12015
  %9189 = add i64 %.in, %9188		; visa id: 12016
  %9190 = inttoptr i64 %9189 to float addrspace(4)*		; visa id: 12017
  %9191 = addrspacecast float addrspace(4)* %9190 to float addrspace(1)*		; visa id: 12017
  store float %9187, float addrspace(1)* %9191, align 4		; visa id: 12018
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12019

._crit_edge70.1.13:                               ; preds = %.._crit_edge70.1.13_crit_edge, %9172, %9167
; BB1017 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12020
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12020
  %9192 = insertelement <2 x i32> %6108, i32 %9073, i64 1		; visa id: 12020
  store <2 x i32> %9192, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12023
  br label %._crit_edge389, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12025

._crit_edge389:                                   ; preds = %._crit_edge389.._crit_edge389_crit_edge, %._crit_edge70.1.13
; BB1018 :
  %9193 = phi i32 [ 0, %._crit_edge70.1.13 ], [ %9202, %._crit_edge389.._crit_edge389_crit_edge ]
  %9194 = zext i32 %9193 to i64		; visa id: 12026
  %9195 = shl nuw nsw i64 %9194, 2		; visa id: 12027
  %9196 = add i64 %5982, %9195		; visa id: 12028
  %9197 = inttoptr i64 %9196 to i32*		; visa id: 12029
  %9198 = load i32, i32* %9197, align 4, !noalias !642		; visa id: 12029
  %9199 = add i64 %5978, %9195		; visa id: 12030
  %9200 = inttoptr i64 %9199 to i32*		; visa id: 12031
  store i32 %9198, i32* %9200, align 4, !alias.scope !642		; visa id: 12031
  %9201 = icmp eq i32 %9193, 0		; visa id: 12032
  br i1 %9201, label %._crit_edge389.._crit_edge389_crit_edge, label %9203, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12033

._crit_edge389.._crit_edge389_crit_edge:          ; preds = %._crit_edge389
; BB1019 :
  %9202 = add nuw nsw i32 %9193, 1, !spirv.Decorations !631		; visa id: 12035
  br label %._crit_edge389, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12036

9203:                                             ; preds = %._crit_edge389
; BB1020 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12038
  %9204 = load i64, i64* %5998, align 8		; visa id: 12038
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12039
  %9205 = icmp slt i32 %6107, %const_reg_dword
  %9206 = icmp slt i32 %9073, %const_reg_dword1		; visa id: 12039
  %9207 = and i1 %9205, %9206		; visa id: 12040
  br i1 %9207, label %9208, label %.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12042

.._crit_edge70.2.13_crit_edge:                    ; preds = %9203
; BB:
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9208:                                             ; preds = %9203
; BB1022 :
  %9209 = bitcast i64 %9204 to <2 x i32>		; visa id: 12044
  %9210 = extractelement <2 x i32> %9209, i32 0		; visa id: 12046
  %9211 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9210, i32 1
  %9212 = bitcast <2 x i32> %9211 to i64		; visa id: 12046
  %9213 = ashr exact i64 %9212, 32		; visa id: 12047
  %9214 = bitcast i64 %9213 to <2 x i32>		; visa id: 12048
  %9215 = extractelement <2 x i32> %9214, i32 0		; visa id: 12052
  %9216 = extractelement <2 x i32> %9214, i32 1		; visa id: 12052
  %9217 = ashr i64 %9204, 32		; visa id: 12052
  %9218 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9215, i32 %9216, i32 %50, i32 %51)
  %9219 = extractvalue { i32, i32 } %9218, 0		; visa id: 12053
  %9220 = extractvalue { i32, i32 } %9218, 1		; visa id: 12053
  %9221 = insertelement <2 x i32> undef, i32 %9219, i32 0		; visa id: 12060
  %9222 = insertelement <2 x i32> %9221, i32 %9220, i32 1		; visa id: 12061
  %9223 = bitcast <2 x i32> %9222 to i64		; visa id: 12062
  %9224 = add nsw i64 %9223, %9217, !spirv.Decorations !649		; visa id: 12066
  %9225 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !618		; visa id: 12067
  br i1 %86, label %9231, label %9226, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12068

9226:                                             ; preds = %9208
; BB1023 :
  %9227 = shl i64 %9224, 2		; visa id: 12070
  %9228 = add i64 %.in, %9227		; visa id: 12071
  %9229 = inttoptr i64 %9228 to float addrspace(4)*		; visa id: 12072
  %9230 = addrspacecast float addrspace(4)* %9229 to float addrspace(1)*		; visa id: 12072
  store float %9225, float addrspace(1)* %9230, align 4		; visa id: 12073
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12074

9231:                                             ; preds = %9208
; BB1024 :
  %9232 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9215, i32 %9216, i32 %47, i32 %48)
  %9233 = extractvalue { i32, i32 } %9232, 0		; visa id: 12076
  %9234 = extractvalue { i32, i32 } %9232, 1		; visa id: 12076
  %9235 = insertelement <2 x i32> undef, i32 %9233, i32 0		; visa id: 12083
  %9236 = insertelement <2 x i32> %9235, i32 %9234, i32 1		; visa id: 12084
  %9237 = bitcast <2 x i32> %9236 to i64		; visa id: 12085
  %9238 = shl i64 %9237, 2		; visa id: 12089
  %9239 = add i64 %.in399, %9238		; visa id: 12090
  %9240 = shl nsw i64 %9217, 2		; visa id: 12091
  %9241 = add i64 %9239, %9240		; visa id: 12092
  %9242 = inttoptr i64 %9241 to float addrspace(4)*		; visa id: 12093
  %9243 = addrspacecast float addrspace(4)* %9242 to float addrspace(1)*		; visa id: 12093
  %9244 = load float, float addrspace(1)* %9243, align 4		; visa id: 12094
  %9245 = fmul reassoc nsz arcp contract float %9244, %4, !spirv.Decorations !618		; visa id: 12095
  %9246 = fadd reassoc nsz arcp contract float %9225, %9245, !spirv.Decorations !618		; visa id: 12096
  %9247 = shl i64 %9224, 2		; visa id: 12097
  %9248 = add i64 %.in, %9247		; visa id: 12098
  %9249 = inttoptr i64 %9248 to float addrspace(4)*		; visa id: 12099
  %9250 = addrspacecast float addrspace(4)* %9249 to float addrspace(1)*		; visa id: 12099
  store float %9246, float addrspace(1)* %9250, align 4		; visa id: 12100
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12101

._crit_edge70.2.13:                               ; preds = %.._crit_edge70.2.13_crit_edge, %9231, %9226
; BB1025 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12102
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12102
  %9251 = insertelement <2 x i32> %6169, i32 %9073, i64 1		; visa id: 12102
  store <2 x i32> %9251, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12105
  br label %._crit_edge390, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12107

._crit_edge390:                                   ; preds = %._crit_edge390.._crit_edge390_crit_edge, %._crit_edge70.2.13
; BB1026 :
  %9252 = phi i32 [ 0, %._crit_edge70.2.13 ], [ %9261, %._crit_edge390.._crit_edge390_crit_edge ]
  %9253 = zext i32 %9252 to i64		; visa id: 12108
  %9254 = shl nuw nsw i64 %9253, 2		; visa id: 12109
  %9255 = add i64 %5982, %9254		; visa id: 12110
  %9256 = inttoptr i64 %9255 to i32*		; visa id: 12111
  %9257 = load i32, i32* %9256, align 4, !noalias !642		; visa id: 12111
  %9258 = add i64 %5978, %9254		; visa id: 12112
  %9259 = inttoptr i64 %9258 to i32*		; visa id: 12113
  store i32 %9257, i32* %9259, align 4, !alias.scope !642		; visa id: 12113
  %9260 = icmp eq i32 %9252, 0		; visa id: 12114
  br i1 %9260, label %._crit_edge390.._crit_edge390_crit_edge, label %9262, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12115

._crit_edge390.._crit_edge390_crit_edge:          ; preds = %._crit_edge390
; BB1027 :
  %9261 = add nuw nsw i32 %9252, 1, !spirv.Decorations !631		; visa id: 12117
  br label %._crit_edge390, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12118

9262:                                             ; preds = %._crit_edge390
; BB1028 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12120
  %9263 = load i64, i64* %5998, align 8		; visa id: 12120
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12121
  %9264 = icmp slt i32 %6168, %const_reg_dword
  %9265 = icmp slt i32 %9073, %const_reg_dword1		; visa id: 12121
  %9266 = and i1 %9264, %9265		; visa id: 12122
  br i1 %9266, label %9267, label %..preheader1.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12124

..preheader1.13_crit_edge:                        ; preds = %9262
; BB:
  br label %.preheader1.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9267:                                             ; preds = %9262
; BB1030 :
  %9268 = bitcast i64 %9263 to <2 x i32>		; visa id: 12126
  %9269 = extractelement <2 x i32> %9268, i32 0		; visa id: 12128
  %9270 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9269, i32 1
  %9271 = bitcast <2 x i32> %9270 to i64		; visa id: 12128
  %9272 = ashr exact i64 %9271, 32		; visa id: 12129
  %9273 = bitcast i64 %9272 to <2 x i32>		; visa id: 12130
  %9274 = extractelement <2 x i32> %9273, i32 0		; visa id: 12134
  %9275 = extractelement <2 x i32> %9273, i32 1		; visa id: 12134
  %9276 = ashr i64 %9263, 32		; visa id: 12134
  %9277 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9274, i32 %9275, i32 %50, i32 %51)
  %9278 = extractvalue { i32, i32 } %9277, 0		; visa id: 12135
  %9279 = extractvalue { i32, i32 } %9277, 1		; visa id: 12135
  %9280 = insertelement <2 x i32> undef, i32 %9278, i32 0		; visa id: 12142
  %9281 = insertelement <2 x i32> %9280, i32 %9279, i32 1		; visa id: 12143
  %9282 = bitcast <2 x i32> %9281 to i64		; visa id: 12144
  %9283 = add nsw i64 %9282, %9276, !spirv.Decorations !649		; visa id: 12148
  %9284 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !618		; visa id: 12149
  br i1 %86, label %9290, label %9285, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12150

9285:                                             ; preds = %9267
; BB1031 :
  %9286 = shl i64 %9283, 2		; visa id: 12152
  %9287 = add i64 %.in, %9286		; visa id: 12153
  %9288 = inttoptr i64 %9287 to float addrspace(4)*		; visa id: 12154
  %9289 = addrspacecast float addrspace(4)* %9288 to float addrspace(1)*		; visa id: 12154
  store float %9284, float addrspace(1)* %9289, align 4		; visa id: 12155
  br label %.preheader1.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12156

9290:                                             ; preds = %9267
; BB1032 :
  %9291 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9274, i32 %9275, i32 %47, i32 %48)
  %9292 = extractvalue { i32, i32 } %9291, 0		; visa id: 12158
  %9293 = extractvalue { i32, i32 } %9291, 1		; visa id: 12158
  %9294 = insertelement <2 x i32> undef, i32 %9292, i32 0		; visa id: 12165
  %9295 = insertelement <2 x i32> %9294, i32 %9293, i32 1		; visa id: 12166
  %9296 = bitcast <2 x i32> %9295 to i64		; visa id: 12167
  %9297 = shl i64 %9296, 2		; visa id: 12171
  %9298 = add i64 %.in399, %9297		; visa id: 12172
  %9299 = shl nsw i64 %9276, 2		; visa id: 12173
  %9300 = add i64 %9298, %9299		; visa id: 12174
  %9301 = inttoptr i64 %9300 to float addrspace(4)*		; visa id: 12175
  %9302 = addrspacecast float addrspace(4)* %9301 to float addrspace(1)*		; visa id: 12175
  %9303 = load float, float addrspace(1)* %9302, align 4		; visa id: 12176
  %9304 = fmul reassoc nsz arcp contract float %9303, %4, !spirv.Decorations !618		; visa id: 12177
  %9305 = fadd reassoc nsz arcp contract float %9284, %9304, !spirv.Decorations !618		; visa id: 12178
  %9306 = shl i64 %9283, 2		; visa id: 12179
  %9307 = add i64 %.in, %9306		; visa id: 12180
  %9308 = inttoptr i64 %9307 to float addrspace(4)*		; visa id: 12181
  %9309 = addrspacecast float addrspace(4)* %9308 to float addrspace(1)*		; visa id: 12181
  store float %9305, float addrspace(1)* %9309, align 4		; visa id: 12182
  br label %.preheader1.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12183

.preheader1.13:                                   ; preds = %..preheader1.13_crit_edge, %9290, %9285
; BB1033 :
  %9310 = add i32 %69, 14		; visa id: 12184
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12185
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12185
  %9311 = insertelement <2 x i32> %5984, i32 %9310, i64 1		; visa id: 12185
  store <2 x i32> %9311, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12188
  br label %._crit_edge391, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12190

._crit_edge391:                                   ; preds = %._crit_edge391.._crit_edge391_crit_edge, %.preheader1.13
; BB1034 :
  %9312 = phi i32 [ 0, %.preheader1.13 ], [ %9321, %._crit_edge391.._crit_edge391_crit_edge ]
  %9313 = zext i32 %9312 to i64		; visa id: 12191
  %9314 = shl nuw nsw i64 %9313, 2		; visa id: 12192
  %9315 = add i64 %5982, %9314		; visa id: 12193
  %9316 = inttoptr i64 %9315 to i32*		; visa id: 12194
  %9317 = load i32, i32* %9316, align 4, !noalias !642		; visa id: 12194
  %9318 = add i64 %5978, %9314		; visa id: 12195
  %9319 = inttoptr i64 %9318 to i32*		; visa id: 12196
  store i32 %9317, i32* %9319, align 4, !alias.scope !642		; visa id: 12196
  %9320 = icmp eq i32 %9312, 0		; visa id: 12197
  br i1 %9320, label %._crit_edge391.._crit_edge391_crit_edge, label %9322, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12198

._crit_edge391.._crit_edge391_crit_edge:          ; preds = %._crit_edge391
; BB1035 :
  %9321 = add nuw nsw i32 %9312, 1, !spirv.Decorations !631		; visa id: 12200
  br label %._crit_edge391, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12201

9322:                                             ; preds = %._crit_edge391
; BB1036 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12203
  %9323 = load i64, i64* %5998, align 8		; visa id: 12203
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12204
  %9324 = icmp slt i32 %9310, %const_reg_dword1		; visa id: 12204
  %9325 = icmp slt i32 %65, %const_reg_dword
  %9326 = and i1 %9325, %9324		; visa id: 12205
  br i1 %9326, label %9327, label %.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12207

.._crit_edge70.14_crit_edge:                      ; preds = %9322
; BB:
  br label %._crit_edge70.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9327:                                             ; preds = %9322
; BB1038 :
  %9328 = bitcast i64 %9323 to <2 x i32>		; visa id: 12209
  %9329 = extractelement <2 x i32> %9328, i32 0		; visa id: 12211
  %9330 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9329, i32 1
  %9331 = bitcast <2 x i32> %9330 to i64		; visa id: 12211
  %9332 = ashr exact i64 %9331, 32		; visa id: 12212
  %9333 = bitcast i64 %9332 to <2 x i32>		; visa id: 12213
  %9334 = extractelement <2 x i32> %9333, i32 0		; visa id: 12217
  %9335 = extractelement <2 x i32> %9333, i32 1		; visa id: 12217
  %9336 = ashr i64 %9323, 32		; visa id: 12217
  %9337 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9334, i32 %9335, i32 %50, i32 %51)
  %9338 = extractvalue { i32, i32 } %9337, 0		; visa id: 12218
  %9339 = extractvalue { i32, i32 } %9337, 1		; visa id: 12218
  %9340 = insertelement <2 x i32> undef, i32 %9338, i32 0		; visa id: 12225
  %9341 = insertelement <2 x i32> %9340, i32 %9339, i32 1		; visa id: 12226
  %9342 = bitcast <2 x i32> %9341 to i64		; visa id: 12227
  %9343 = add nsw i64 %9342, %9336, !spirv.Decorations !649		; visa id: 12231
  %9344 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !618		; visa id: 12232
  br i1 %86, label %9350, label %9345, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12233

9345:                                             ; preds = %9327
; BB1039 :
  %9346 = shl i64 %9343, 2		; visa id: 12235
  %9347 = add i64 %.in, %9346		; visa id: 12236
  %9348 = inttoptr i64 %9347 to float addrspace(4)*		; visa id: 12237
  %9349 = addrspacecast float addrspace(4)* %9348 to float addrspace(1)*		; visa id: 12237
  store float %9344, float addrspace(1)* %9349, align 4		; visa id: 12238
  br label %._crit_edge70.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12239

9350:                                             ; preds = %9327
; BB1040 :
  %9351 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9334, i32 %9335, i32 %47, i32 %48)
  %9352 = extractvalue { i32, i32 } %9351, 0		; visa id: 12241
  %9353 = extractvalue { i32, i32 } %9351, 1		; visa id: 12241
  %9354 = insertelement <2 x i32> undef, i32 %9352, i32 0		; visa id: 12248
  %9355 = insertelement <2 x i32> %9354, i32 %9353, i32 1		; visa id: 12249
  %9356 = bitcast <2 x i32> %9355 to i64		; visa id: 12250
  %9357 = shl i64 %9356, 2		; visa id: 12254
  %9358 = add i64 %.in399, %9357		; visa id: 12255
  %9359 = shl nsw i64 %9336, 2		; visa id: 12256
  %9360 = add i64 %9358, %9359		; visa id: 12257
  %9361 = inttoptr i64 %9360 to float addrspace(4)*		; visa id: 12258
  %9362 = addrspacecast float addrspace(4)* %9361 to float addrspace(1)*		; visa id: 12258
  %9363 = load float, float addrspace(1)* %9362, align 4		; visa id: 12259
  %9364 = fmul reassoc nsz arcp contract float %9363, %4, !spirv.Decorations !618		; visa id: 12260
  %9365 = fadd reassoc nsz arcp contract float %9344, %9364, !spirv.Decorations !618		; visa id: 12261
  %9366 = shl i64 %9343, 2		; visa id: 12262
  %9367 = add i64 %.in, %9366		; visa id: 12263
  %9368 = inttoptr i64 %9367 to float addrspace(4)*		; visa id: 12264
  %9369 = addrspacecast float addrspace(4)* %9368 to float addrspace(1)*		; visa id: 12264
  store float %9365, float addrspace(1)* %9369, align 4		; visa id: 12265
  br label %._crit_edge70.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12266

._crit_edge70.14:                                 ; preds = %.._crit_edge70.14_crit_edge, %9350, %9345
; BB1041 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12267
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12267
  %9370 = insertelement <2 x i32> %6047, i32 %9310, i64 1		; visa id: 12267
  store <2 x i32> %9370, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12270
  br label %._crit_edge392, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12272

._crit_edge392:                                   ; preds = %._crit_edge392.._crit_edge392_crit_edge, %._crit_edge70.14
; BB1042 :
  %9371 = phi i32 [ 0, %._crit_edge70.14 ], [ %9380, %._crit_edge392.._crit_edge392_crit_edge ]
  %9372 = zext i32 %9371 to i64		; visa id: 12273
  %9373 = shl nuw nsw i64 %9372, 2		; visa id: 12274
  %9374 = add i64 %5982, %9373		; visa id: 12275
  %9375 = inttoptr i64 %9374 to i32*		; visa id: 12276
  %9376 = load i32, i32* %9375, align 4, !noalias !642		; visa id: 12276
  %9377 = add i64 %5978, %9373		; visa id: 12277
  %9378 = inttoptr i64 %9377 to i32*		; visa id: 12278
  store i32 %9376, i32* %9378, align 4, !alias.scope !642		; visa id: 12278
  %9379 = icmp eq i32 %9371, 0		; visa id: 12279
  br i1 %9379, label %._crit_edge392.._crit_edge392_crit_edge, label %9381, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12280

._crit_edge392.._crit_edge392_crit_edge:          ; preds = %._crit_edge392
; BB1043 :
  %9380 = add nuw nsw i32 %9371, 1, !spirv.Decorations !631		; visa id: 12282
  br label %._crit_edge392, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12283

9381:                                             ; preds = %._crit_edge392
; BB1044 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12285
  %9382 = load i64, i64* %5998, align 8		; visa id: 12285
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12286
  %9383 = icmp slt i32 %6046, %const_reg_dword
  %9384 = icmp slt i32 %9310, %const_reg_dword1		; visa id: 12286
  %9385 = and i1 %9383, %9384		; visa id: 12287
  br i1 %9385, label %9386, label %.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12289

.._crit_edge70.1.14_crit_edge:                    ; preds = %9381
; BB:
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9386:                                             ; preds = %9381
; BB1046 :
  %9387 = bitcast i64 %9382 to <2 x i32>		; visa id: 12291
  %9388 = extractelement <2 x i32> %9387, i32 0		; visa id: 12293
  %9389 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9388, i32 1
  %9390 = bitcast <2 x i32> %9389 to i64		; visa id: 12293
  %9391 = ashr exact i64 %9390, 32		; visa id: 12294
  %9392 = bitcast i64 %9391 to <2 x i32>		; visa id: 12295
  %9393 = extractelement <2 x i32> %9392, i32 0		; visa id: 12299
  %9394 = extractelement <2 x i32> %9392, i32 1		; visa id: 12299
  %9395 = ashr i64 %9382, 32		; visa id: 12299
  %9396 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9393, i32 %9394, i32 %50, i32 %51)
  %9397 = extractvalue { i32, i32 } %9396, 0		; visa id: 12300
  %9398 = extractvalue { i32, i32 } %9396, 1		; visa id: 12300
  %9399 = insertelement <2 x i32> undef, i32 %9397, i32 0		; visa id: 12307
  %9400 = insertelement <2 x i32> %9399, i32 %9398, i32 1		; visa id: 12308
  %9401 = bitcast <2 x i32> %9400 to i64		; visa id: 12309
  %9402 = add nsw i64 %9401, %9395, !spirv.Decorations !649		; visa id: 12313
  %9403 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !618		; visa id: 12314
  br i1 %86, label %9409, label %9404, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12315

9404:                                             ; preds = %9386
; BB1047 :
  %9405 = shl i64 %9402, 2		; visa id: 12317
  %9406 = add i64 %.in, %9405		; visa id: 12318
  %9407 = inttoptr i64 %9406 to float addrspace(4)*		; visa id: 12319
  %9408 = addrspacecast float addrspace(4)* %9407 to float addrspace(1)*		; visa id: 12319
  store float %9403, float addrspace(1)* %9408, align 4		; visa id: 12320
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12321

9409:                                             ; preds = %9386
; BB1048 :
  %9410 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9393, i32 %9394, i32 %47, i32 %48)
  %9411 = extractvalue { i32, i32 } %9410, 0		; visa id: 12323
  %9412 = extractvalue { i32, i32 } %9410, 1		; visa id: 12323
  %9413 = insertelement <2 x i32> undef, i32 %9411, i32 0		; visa id: 12330
  %9414 = insertelement <2 x i32> %9413, i32 %9412, i32 1		; visa id: 12331
  %9415 = bitcast <2 x i32> %9414 to i64		; visa id: 12332
  %9416 = shl i64 %9415, 2		; visa id: 12336
  %9417 = add i64 %.in399, %9416		; visa id: 12337
  %9418 = shl nsw i64 %9395, 2		; visa id: 12338
  %9419 = add i64 %9417, %9418		; visa id: 12339
  %9420 = inttoptr i64 %9419 to float addrspace(4)*		; visa id: 12340
  %9421 = addrspacecast float addrspace(4)* %9420 to float addrspace(1)*		; visa id: 12340
  %9422 = load float, float addrspace(1)* %9421, align 4		; visa id: 12341
  %9423 = fmul reassoc nsz arcp contract float %9422, %4, !spirv.Decorations !618		; visa id: 12342
  %9424 = fadd reassoc nsz arcp contract float %9403, %9423, !spirv.Decorations !618		; visa id: 12343
  %9425 = shl i64 %9402, 2		; visa id: 12344
  %9426 = add i64 %.in, %9425		; visa id: 12345
  %9427 = inttoptr i64 %9426 to float addrspace(4)*		; visa id: 12346
  %9428 = addrspacecast float addrspace(4)* %9427 to float addrspace(1)*		; visa id: 12346
  store float %9424, float addrspace(1)* %9428, align 4		; visa id: 12347
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12348

._crit_edge70.1.14:                               ; preds = %.._crit_edge70.1.14_crit_edge, %9409, %9404
; BB1049 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12349
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12349
  %9429 = insertelement <2 x i32> %6108, i32 %9310, i64 1		; visa id: 12349
  store <2 x i32> %9429, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12352
  br label %._crit_edge393, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12354

._crit_edge393:                                   ; preds = %._crit_edge393.._crit_edge393_crit_edge, %._crit_edge70.1.14
; BB1050 :
  %9430 = phi i32 [ 0, %._crit_edge70.1.14 ], [ %9439, %._crit_edge393.._crit_edge393_crit_edge ]
  %9431 = zext i32 %9430 to i64		; visa id: 12355
  %9432 = shl nuw nsw i64 %9431, 2		; visa id: 12356
  %9433 = add i64 %5982, %9432		; visa id: 12357
  %9434 = inttoptr i64 %9433 to i32*		; visa id: 12358
  %9435 = load i32, i32* %9434, align 4, !noalias !642		; visa id: 12358
  %9436 = add i64 %5978, %9432		; visa id: 12359
  %9437 = inttoptr i64 %9436 to i32*		; visa id: 12360
  store i32 %9435, i32* %9437, align 4, !alias.scope !642		; visa id: 12360
  %9438 = icmp eq i32 %9430, 0		; visa id: 12361
  br i1 %9438, label %._crit_edge393.._crit_edge393_crit_edge, label %9440, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12362

._crit_edge393.._crit_edge393_crit_edge:          ; preds = %._crit_edge393
; BB1051 :
  %9439 = add nuw nsw i32 %9430, 1, !spirv.Decorations !631		; visa id: 12364
  br label %._crit_edge393, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12365

9440:                                             ; preds = %._crit_edge393
; BB1052 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12367
  %9441 = load i64, i64* %5998, align 8		; visa id: 12367
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12368
  %9442 = icmp slt i32 %6107, %const_reg_dword
  %9443 = icmp slt i32 %9310, %const_reg_dword1		; visa id: 12368
  %9444 = and i1 %9442, %9443		; visa id: 12369
  br i1 %9444, label %9445, label %.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12371

.._crit_edge70.2.14_crit_edge:                    ; preds = %9440
; BB:
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9445:                                             ; preds = %9440
; BB1054 :
  %9446 = bitcast i64 %9441 to <2 x i32>		; visa id: 12373
  %9447 = extractelement <2 x i32> %9446, i32 0		; visa id: 12375
  %9448 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9447, i32 1
  %9449 = bitcast <2 x i32> %9448 to i64		; visa id: 12375
  %9450 = ashr exact i64 %9449, 32		; visa id: 12376
  %9451 = bitcast i64 %9450 to <2 x i32>		; visa id: 12377
  %9452 = extractelement <2 x i32> %9451, i32 0		; visa id: 12381
  %9453 = extractelement <2 x i32> %9451, i32 1		; visa id: 12381
  %9454 = ashr i64 %9441, 32		; visa id: 12381
  %9455 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9452, i32 %9453, i32 %50, i32 %51)
  %9456 = extractvalue { i32, i32 } %9455, 0		; visa id: 12382
  %9457 = extractvalue { i32, i32 } %9455, 1		; visa id: 12382
  %9458 = insertelement <2 x i32> undef, i32 %9456, i32 0		; visa id: 12389
  %9459 = insertelement <2 x i32> %9458, i32 %9457, i32 1		; visa id: 12390
  %9460 = bitcast <2 x i32> %9459 to i64		; visa id: 12391
  %9461 = add nsw i64 %9460, %9454, !spirv.Decorations !649		; visa id: 12395
  %9462 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !618		; visa id: 12396
  br i1 %86, label %9468, label %9463, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12397

9463:                                             ; preds = %9445
; BB1055 :
  %9464 = shl i64 %9461, 2		; visa id: 12399
  %9465 = add i64 %.in, %9464		; visa id: 12400
  %9466 = inttoptr i64 %9465 to float addrspace(4)*		; visa id: 12401
  %9467 = addrspacecast float addrspace(4)* %9466 to float addrspace(1)*		; visa id: 12401
  store float %9462, float addrspace(1)* %9467, align 4		; visa id: 12402
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12403

9468:                                             ; preds = %9445
; BB1056 :
  %9469 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9452, i32 %9453, i32 %47, i32 %48)
  %9470 = extractvalue { i32, i32 } %9469, 0		; visa id: 12405
  %9471 = extractvalue { i32, i32 } %9469, 1		; visa id: 12405
  %9472 = insertelement <2 x i32> undef, i32 %9470, i32 0		; visa id: 12412
  %9473 = insertelement <2 x i32> %9472, i32 %9471, i32 1		; visa id: 12413
  %9474 = bitcast <2 x i32> %9473 to i64		; visa id: 12414
  %9475 = shl i64 %9474, 2		; visa id: 12418
  %9476 = add i64 %.in399, %9475		; visa id: 12419
  %9477 = shl nsw i64 %9454, 2		; visa id: 12420
  %9478 = add i64 %9476, %9477		; visa id: 12421
  %9479 = inttoptr i64 %9478 to float addrspace(4)*		; visa id: 12422
  %9480 = addrspacecast float addrspace(4)* %9479 to float addrspace(1)*		; visa id: 12422
  %9481 = load float, float addrspace(1)* %9480, align 4		; visa id: 12423
  %9482 = fmul reassoc nsz arcp contract float %9481, %4, !spirv.Decorations !618		; visa id: 12424
  %9483 = fadd reassoc nsz arcp contract float %9462, %9482, !spirv.Decorations !618		; visa id: 12425
  %9484 = shl i64 %9461, 2		; visa id: 12426
  %9485 = add i64 %.in, %9484		; visa id: 12427
  %9486 = inttoptr i64 %9485 to float addrspace(4)*		; visa id: 12428
  %9487 = addrspacecast float addrspace(4)* %9486 to float addrspace(1)*		; visa id: 12428
  store float %9483, float addrspace(1)* %9487, align 4		; visa id: 12429
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12430

._crit_edge70.2.14:                               ; preds = %.._crit_edge70.2.14_crit_edge, %9468, %9463
; BB1057 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12431
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12431
  %9488 = insertelement <2 x i32> %6169, i32 %9310, i64 1		; visa id: 12431
  store <2 x i32> %9488, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12434
  br label %._crit_edge394, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12436

._crit_edge394:                                   ; preds = %._crit_edge394.._crit_edge394_crit_edge, %._crit_edge70.2.14
; BB1058 :
  %9489 = phi i32 [ 0, %._crit_edge70.2.14 ], [ %9498, %._crit_edge394.._crit_edge394_crit_edge ]
  %9490 = zext i32 %9489 to i64		; visa id: 12437
  %9491 = shl nuw nsw i64 %9490, 2		; visa id: 12438
  %9492 = add i64 %5982, %9491		; visa id: 12439
  %9493 = inttoptr i64 %9492 to i32*		; visa id: 12440
  %9494 = load i32, i32* %9493, align 4, !noalias !642		; visa id: 12440
  %9495 = add i64 %5978, %9491		; visa id: 12441
  %9496 = inttoptr i64 %9495 to i32*		; visa id: 12442
  store i32 %9494, i32* %9496, align 4, !alias.scope !642		; visa id: 12442
  %9497 = icmp eq i32 %9489, 0		; visa id: 12443
  br i1 %9497, label %._crit_edge394.._crit_edge394_crit_edge, label %9499, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12444

._crit_edge394.._crit_edge394_crit_edge:          ; preds = %._crit_edge394
; BB1059 :
  %9498 = add nuw nsw i32 %9489, 1, !spirv.Decorations !631		; visa id: 12446
  br label %._crit_edge394, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12447

9499:                                             ; preds = %._crit_edge394
; BB1060 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12449
  %9500 = load i64, i64* %5998, align 8		; visa id: 12449
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12450
  %9501 = icmp slt i32 %6168, %const_reg_dword
  %9502 = icmp slt i32 %9310, %const_reg_dword1		; visa id: 12450
  %9503 = and i1 %9501, %9502		; visa id: 12451
  br i1 %9503, label %9504, label %..preheader1.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12453

..preheader1.14_crit_edge:                        ; preds = %9499
; BB:
  br label %.preheader1.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9504:                                             ; preds = %9499
; BB1062 :
  %9505 = bitcast i64 %9500 to <2 x i32>		; visa id: 12455
  %9506 = extractelement <2 x i32> %9505, i32 0		; visa id: 12457
  %9507 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9506, i32 1
  %9508 = bitcast <2 x i32> %9507 to i64		; visa id: 12457
  %9509 = ashr exact i64 %9508, 32		; visa id: 12458
  %9510 = bitcast i64 %9509 to <2 x i32>		; visa id: 12459
  %9511 = extractelement <2 x i32> %9510, i32 0		; visa id: 12463
  %9512 = extractelement <2 x i32> %9510, i32 1		; visa id: 12463
  %9513 = ashr i64 %9500, 32		; visa id: 12463
  %9514 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9511, i32 %9512, i32 %50, i32 %51)
  %9515 = extractvalue { i32, i32 } %9514, 0		; visa id: 12464
  %9516 = extractvalue { i32, i32 } %9514, 1		; visa id: 12464
  %9517 = insertelement <2 x i32> undef, i32 %9515, i32 0		; visa id: 12471
  %9518 = insertelement <2 x i32> %9517, i32 %9516, i32 1		; visa id: 12472
  %9519 = bitcast <2 x i32> %9518 to i64		; visa id: 12473
  %9520 = add nsw i64 %9519, %9513, !spirv.Decorations !649		; visa id: 12477
  %9521 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !618		; visa id: 12478
  br i1 %86, label %9527, label %9522, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12479

9522:                                             ; preds = %9504
; BB1063 :
  %9523 = shl i64 %9520, 2		; visa id: 12481
  %9524 = add i64 %.in, %9523		; visa id: 12482
  %9525 = inttoptr i64 %9524 to float addrspace(4)*		; visa id: 12483
  %9526 = addrspacecast float addrspace(4)* %9525 to float addrspace(1)*		; visa id: 12483
  store float %9521, float addrspace(1)* %9526, align 4		; visa id: 12484
  br label %.preheader1.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12485

9527:                                             ; preds = %9504
; BB1064 :
  %9528 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9511, i32 %9512, i32 %47, i32 %48)
  %9529 = extractvalue { i32, i32 } %9528, 0		; visa id: 12487
  %9530 = extractvalue { i32, i32 } %9528, 1		; visa id: 12487
  %9531 = insertelement <2 x i32> undef, i32 %9529, i32 0		; visa id: 12494
  %9532 = insertelement <2 x i32> %9531, i32 %9530, i32 1		; visa id: 12495
  %9533 = bitcast <2 x i32> %9532 to i64		; visa id: 12496
  %9534 = shl i64 %9533, 2		; visa id: 12500
  %9535 = add i64 %.in399, %9534		; visa id: 12501
  %9536 = shl nsw i64 %9513, 2		; visa id: 12502
  %9537 = add i64 %9535, %9536		; visa id: 12503
  %9538 = inttoptr i64 %9537 to float addrspace(4)*		; visa id: 12504
  %9539 = addrspacecast float addrspace(4)* %9538 to float addrspace(1)*		; visa id: 12504
  %9540 = load float, float addrspace(1)* %9539, align 4		; visa id: 12505
  %9541 = fmul reassoc nsz arcp contract float %9540, %4, !spirv.Decorations !618		; visa id: 12506
  %9542 = fadd reassoc nsz arcp contract float %9521, %9541, !spirv.Decorations !618		; visa id: 12507
  %9543 = shl i64 %9520, 2		; visa id: 12508
  %9544 = add i64 %.in, %9543		; visa id: 12509
  %9545 = inttoptr i64 %9544 to float addrspace(4)*		; visa id: 12510
  %9546 = addrspacecast float addrspace(4)* %9545 to float addrspace(1)*		; visa id: 12510
  store float %9542, float addrspace(1)* %9546, align 4		; visa id: 12511
  br label %.preheader1.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12512

.preheader1.14:                                   ; preds = %..preheader1.14_crit_edge, %9527, %9522
; BB1065 :
  %9547 = add i32 %69, 15		; visa id: 12513
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12514
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12514
  %9548 = insertelement <2 x i32> %5984, i32 %9547, i64 1		; visa id: 12514
  store <2 x i32> %9548, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12515
  br label %._crit_edge395, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12517

._crit_edge395:                                   ; preds = %._crit_edge395.._crit_edge395_crit_edge, %.preheader1.14
; BB1066 :
  %9549 = phi i32 [ 0, %.preheader1.14 ], [ %9558, %._crit_edge395.._crit_edge395_crit_edge ]
  %9550 = zext i32 %9549 to i64		; visa id: 12518
  %9551 = shl nuw nsw i64 %9550, 2		; visa id: 12519
  %9552 = add i64 %5982, %9551		; visa id: 12520
  %9553 = inttoptr i64 %9552 to i32*		; visa id: 12521
  %9554 = load i32, i32* %9553, align 4, !noalias !642		; visa id: 12521
  %9555 = add i64 %5978, %9551		; visa id: 12522
  %9556 = inttoptr i64 %9555 to i32*		; visa id: 12523
  store i32 %9554, i32* %9556, align 4, !alias.scope !642		; visa id: 12523
  %9557 = icmp eq i32 %9549, 0		; visa id: 12524
  br i1 %9557, label %._crit_edge395.._crit_edge395_crit_edge, label %9559, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12525

._crit_edge395.._crit_edge395_crit_edge:          ; preds = %._crit_edge395
; BB1067 :
  %9558 = add nuw nsw i32 %9549, 1, !spirv.Decorations !631		; visa id: 12527
  br label %._crit_edge395, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12528

9559:                                             ; preds = %._crit_edge395
; BB1068 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12530
  %9560 = load i64, i64* %5998, align 8		; visa id: 12530
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12531
  %9561 = icmp slt i32 %9547, %const_reg_dword1		; visa id: 12531
  %9562 = icmp slt i32 %65, %const_reg_dword
  %9563 = and i1 %9562, %9561		; visa id: 12532
  br i1 %9563, label %9564, label %.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12534

.._crit_edge70.15_crit_edge:                      ; preds = %9559
; BB:
  br label %._crit_edge70.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9564:                                             ; preds = %9559
; BB1070 :
  %9565 = bitcast i64 %9560 to <2 x i32>		; visa id: 12536
  %9566 = extractelement <2 x i32> %9565, i32 0		; visa id: 12538
  %9567 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9566, i32 1
  %9568 = bitcast <2 x i32> %9567 to i64		; visa id: 12538
  %9569 = ashr exact i64 %9568, 32		; visa id: 12539
  %9570 = bitcast i64 %9569 to <2 x i32>		; visa id: 12540
  %9571 = extractelement <2 x i32> %9570, i32 0		; visa id: 12544
  %9572 = extractelement <2 x i32> %9570, i32 1		; visa id: 12544
  %9573 = ashr i64 %9560, 32		; visa id: 12544
  %9574 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9571, i32 %9572, i32 %50, i32 %51)
  %9575 = extractvalue { i32, i32 } %9574, 0		; visa id: 12545
  %9576 = extractvalue { i32, i32 } %9574, 1		; visa id: 12545
  %9577 = insertelement <2 x i32> undef, i32 %9575, i32 0		; visa id: 12552
  %9578 = insertelement <2 x i32> %9577, i32 %9576, i32 1		; visa id: 12553
  %9579 = bitcast <2 x i32> %9578 to i64		; visa id: 12554
  %9580 = add nsw i64 %9579, %9573, !spirv.Decorations !649		; visa id: 12558
  %9581 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !618		; visa id: 12559
  br i1 %86, label %9587, label %9582, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12560

9582:                                             ; preds = %9564
; BB1071 :
  %9583 = shl i64 %9580, 2		; visa id: 12562
  %9584 = add i64 %.in, %9583		; visa id: 12563
  %9585 = inttoptr i64 %9584 to float addrspace(4)*		; visa id: 12564
  %9586 = addrspacecast float addrspace(4)* %9585 to float addrspace(1)*		; visa id: 12564
  store float %9581, float addrspace(1)* %9586, align 4		; visa id: 12565
  br label %._crit_edge70.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12566

9587:                                             ; preds = %9564
; BB1072 :
  %9588 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9571, i32 %9572, i32 %47, i32 %48)
  %9589 = extractvalue { i32, i32 } %9588, 0		; visa id: 12568
  %9590 = extractvalue { i32, i32 } %9588, 1		; visa id: 12568
  %9591 = insertelement <2 x i32> undef, i32 %9589, i32 0		; visa id: 12575
  %9592 = insertelement <2 x i32> %9591, i32 %9590, i32 1		; visa id: 12576
  %9593 = bitcast <2 x i32> %9592 to i64		; visa id: 12577
  %9594 = shl i64 %9593, 2		; visa id: 12581
  %9595 = add i64 %.in399, %9594		; visa id: 12582
  %9596 = shl nsw i64 %9573, 2		; visa id: 12583
  %9597 = add i64 %9595, %9596		; visa id: 12584
  %9598 = inttoptr i64 %9597 to float addrspace(4)*		; visa id: 12585
  %9599 = addrspacecast float addrspace(4)* %9598 to float addrspace(1)*		; visa id: 12585
  %9600 = load float, float addrspace(1)* %9599, align 4		; visa id: 12586
  %9601 = fmul reassoc nsz arcp contract float %9600, %4, !spirv.Decorations !618		; visa id: 12587
  %9602 = fadd reassoc nsz arcp contract float %9581, %9601, !spirv.Decorations !618		; visa id: 12588
  %9603 = shl i64 %9580, 2		; visa id: 12589
  %9604 = add i64 %.in, %9603		; visa id: 12590
  %9605 = inttoptr i64 %9604 to float addrspace(4)*		; visa id: 12591
  %9606 = addrspacecast float addrspace(4)* %9605 to float addrspace(1)*		; visa id: 12591
  store float %9602, float addrspace(1)* %9606, align 4		; visa id: 12592
  br label %._crit_edge70.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12593

._crit_edge70.15:                                 ; preds = %.._crit_edge70.15_crit_edge, %9587, %9582
; BB1073 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12594
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12594
  %9607 = insertelement <2 x i32> %6047, i32 %9547, i64 1		; visa id: 12594
  store <2 x i32> %9607, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12595
  br label %._crit_edge396, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12597

._crit_edge396:                                   ; preds = %._crit_edge396.._crit_edge396_crit_edge, %._crit_edge70.15
; BB1074 :
  %9608 = phi i32 [ 0, %._crit_edge70.15 ], [ %9617, %._crit_edge396.._crit_edge396_crit_edge ]
  %9609 = zext i32 %9608 to i64		; visa id: 12598
  %9610 = shl nuw nsw i64 %9609, 2		; visa id: 12599
  %9611 = add i64 %5982, %9610		; visa id: 12600
  %9612 = inttoptr i64 %9611 to i32*		; visa id: 12601
  %9613 = load i32, i32* %9612, align 4, !noalias !642		; visa id: 12601
  %9614 = add i64 %5978, %9610		; visa id: 12602
  %9615 = inttoptr i64 %9614 to i32*		; visa id: 12603
  store i32 %9613, i32* %9615, align 4, !alias.scope !642		; visa id: 12603
  %9616 = icmp eq i32 %9608, 0		; visa id: 12604
  br i1 %9616, label %._crit_edge396.._crit_edge396_crit_edge, label %9618, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12605

._crit_edge396.._crit_edge396_crit_edge:          ; preds = %._crit_edge396
; BB1075 :
  %9617 = add nuw nsw i32 %9608, 1, !spirv.Decorations !631		; visa id: 12607
  br label %._crit_edge396, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12608

9618:                                             ; preds = %._crit_edge396
; BB1076 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12610
  %9619 = load i64, i64* %5998, align 8		; visa id: 12610
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12611
  %9620 = icmp slt i32 %6046, %const_reg_dword
  %9621 = icmp slt i32 %9547, %const_reg_dword1		; visa id: 12611
  %9622 = and i1 %9620, %9621		; visa id: 12612
  br i1 %9622, label %9623, label %.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12614

.._crit_edge70.1.15_crit_edge:                    ; preds = %9618
; BB:
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9623:                                             ; preds = %9618
; BB1078 :
  %9624 = bitcast i64 %9619 to <2 x i32>		; visa id: 12616
  %9625 = extractelement <2 x i32> %9624, i32 0		; visa id: 12618
  %9626 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9625, i32 1
  %9627 = bitcast <2 x i32> %9626 to i64		; visa id: 12618
  %9628 = ashr exact i64 %9627, 32		; visa id: 12619
  %9629 = bitcast i64 %9628 to <2 x i32>		; visa id: 12620
  %9630 = extractelement <2 x i32> %9629, i32 0		; visa id: 12624
  %9631 = extractelement <2 x i32> %9629, i32 1		; visa id: 12624
  %9632 = ashr i64 %9619, 32		; visa id: 12624
  %9633 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9630, i32 %9631, i32 %50, i32 %51)
  %9634 = extractvalue { i32, i32 } %9633, 0		; visa id: 12625
  %9635 = extractvalue { i32, i32 } %9633, 1		; visa id: 12625
  %9636 = insertelement <2 x i32> undef, i32 %9634, i32 0		; visa id: 12632
  %9637 = insertelement <2 x i32> %9636, i32 %9635, i32 1		; visa id: 12633
  %9638 = bitcast <2 x i32> %9637 to i64		; visa id: 12634
  %9639 = add nsw i64 %9638, %9632, !spirv.Decorations !649		; visa id: 12638
  %9640 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !618		; visa id: 12639
  br i1 %86, label %9646, label %9641, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12640

9641:                                             ; preds = %9623
; BB1079 :
  %9642 = shl i64 %9639, 2		; visa id: 12642
  %9643 = add i64 %.in, %9642		; visa id: 12643
  %9644 = inttoptr i64 %9643 to float addrspace(4)*		; visa id: 12644
  %9645 = addrspacecast float addrspace(4)* %9644 to float addrspace(1)*		; visa id: 12644
  store float %9640, float addrspace(1)* %9645, align 4		; visa id: 12645
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12646

9646:                                             ; preds = %9623
; BB1080 :
  %9647 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9630, i32 %9631, i32 %47, i32 %48)
  %9648 = extractvalue { i32, i32 } %9647, 0		; visa id: 12648
  %9649 = extractvalue { i32, i32 } %9647, 1		; visa id: 12648
  %9650 = insertelement <2 x i32> undef, i32 %9648, i32 0		; visa id: 12655
  %9651 = insertelement <2 x i32> %9650, i32 %9649, i32 1		; visa id: 12656
  %9652 = bitcast <2 x i32> %9651 to i64		; visa id: 12657
  %9653 = shl i64 %9652, 2		; visa id: 12661
  %9654 = add i64 %.in399, %9653		; visa id: 12662
  %9655 = shl nsw i64 %9632, 2		; visa id: 12663
  %9656 = add i64 %9654, %9655		; visa id: 12664
  %9657 = inttoptr i64 %9656 to float addrspace(4)*		; visa id: 12665
  %9658 = addrspacecast float addrspace(4)* %9657 to float addrspace(1)*		; visa id: 12665
  %9659 = load float, float addrspace(1)* %9658, align 4		; visa id: 12666
  %9660 = fmul reassoc nsz arcp contract float %9659, %4, !spirv.Decorations !618		; visa id: 12667
  %9661 = fadd reassoc nsz arcp contract float %9640, %9660, !spirv.Decorations !618		; visa id: 12668
  %9662 = shl i64 %9639, 2		; visa id: 12669
  %9663 = add i64 %.in, %9662		; visa id: 12670
  %9664 = inttoptr i64 %9663 to float addrspace(4)*		; visa id: 12671
  %9665 = addrspacecast float addrspace(4)* %9664 to float addrspace(1)*		; visa id: 12671
  store float %9661, float addrspace(1)* %9665, align 4		; visa id: 12672
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12673

._crit_edge70.1.15:                               ; preds = %.._crit_edge70.1.15_crit_edge, %9646, %9641
; BB1081 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12674
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12674
  %9666 = insertelement <2 x i32> %6108, i32 %9547, i64 1		; visa id: 12674
  store <2 x i32> %9666, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12675
  br label %._crit_edge397, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12677

._crit_edge397:                                   ; preds = %._crit_edge397.._crit_edge397_crit_edge, %._crit_edge70.1.15
; BB1082 :
  %9667 = phi i32 [ 0, %._crit_edge70.1.15 ], [ %9676, %._crit_edge397.._crit_edge397_crit_edge ]
  %9668 = zext i32 %9667 to i64		; visa id: 12678
  %9669 = shl nuw nsw i64 %9668, 2		; visa id: 12679
  %9670 = add i64 %5982, %9669		; visa id: 12680
  %9671 = inttoptr i64 %9670 to i32*		; visa id: 12681
  %9672 = load i32, i32* %9671, align 4, !noalias !642		; visa id: 12681
  %9673 = add i64 %5978, %9669		; visa id: 12682
  %9674 = inttoptr i64 %9673 to i32*		; visa id: 12683
  store i32 %9672, i32* %9674, align 4, !alias.scope !642		; visa id: 12683
  %9675 = icmp eq i32 %9667, 0		; visa id: 12684
  br i1 %9675, label %._crit_edge397.._crit_edge397_crit_edge, label %9677, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12685

._crit_edge397.._crit_edge397_crit_edge:          ; preds = %._crit_edge397
; BB1083 :
  %9676 = add nuw nsw i32 %9667, 1, !spirv.Decorations !631		; visa id: 12687
  br label %._crit_edge397, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12688

9677:                                             ; preds = %._crit_edge397
; BB1084 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12690
  %9678 = load i64, i64* %5998, align 8		; visa id: 12690
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12691
  %9679 = icmp slt i32 %6107, %const_reg_dword
  %9680 = icmp slt i32 %9547, %const_reg_dword1		; visa id: 12691
  %9681 = and i1 %9679, %9680		; visa id: 12692
  br i1 %9681, label %9682, label %.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12694

.._crit_edge70.2.15_crit_edge:                    ; preds = %9677
; BB:
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9682:                                             ; preds = %9677
; BB1086 :
  %9683 = bitcast i64 %9678 to <2 x i32>		; visa id: 12696
  %9684 = extractelement <2 x i32> %9683, i32 0		; visa id: 12698
  %9685 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9684, i32 1
  %9686 = bitcast <2 x i32> %9685 to i64		; visa id: 12698
  %9687 = ashr exact i64 %9686, 32		; visa id: 12699
  %9688 = bitcast i64 %9687 to <2 x i32>		; visa id: 12700
  %9689 = extractelement <2 x i32> %9688, i32 0		; visa id: 12704
  %9690 = extractelement <2 x i32> %9688, i32 1		; visa id: 12704
  %9691 = ashr i64 %9678, 32		; visa id: 12704
  %9692 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9689, i32 %9690, i32 %50, i32 %51)
  %9693 = extractvalue { i32, i32 } %9692, 0		; visa id: 12705
  %9694 = extractvalue { i32, i32 } %9692, 1		; visa id: 12705
  %9695 = insertelement <2 x i32> undef, i32 %9693, i32 0		; visa id: 12712
  %9696 = insertelement <2 x i32> %9695, i32 %9694, i32 1		; visa id: 12713
  %9697 = bitcast <2 x i32> %9696 to i64		; visa id: 12714
  %9698 = add nsw i64 %9697, %9691, !spirv.Decorations !649		; visa id: 12718
  %9699 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !618		; visa id: 12719
  br i1 %86, label %9705, label %9700, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12720

9700:                                             ; preds = %9682
; BB1087 :
  %9701 = shl i64 %9698, 2		; visa id: 12722
  %9702 = add i64 %.in, %9701		; visa id: 12723
  %9703 = inttoptr i64 %9702 to float addrspace(4)*		; visa id: 12724
  %9704 = addrspacecast float addrspace(4)* %9703 to float addrspace(1)*		; visa id: 12724
  store float %9699, float addrspace(1)* %9704, align 4		; visa id: 12725
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12726

9705:                                             ; preds = %9682
; BB1088 :
  %9706 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9689, i32 %9690, i32 %47, i32 %48)
  %9707 = extractvalue { i32, i32 } %9706, 0		; visa id: 12728
  %9708 = extractvalue { i32, i32 } %9706, 1		; visa id: 12728
  %9709 = insertelement <2 x i32> undef, i32 %9707, i32 0		; visa id: 12735
  %9710 = insertelement <2 x i32> %9709, i32 %9708, i32 1		; visa id: 12736
  %9711 = bitcast <2 x i32> %9710 to i64		; visa id: 12737
  %9712 = shl i64 %9711, 2		; visa id: 12741
  %9713 = add i64 %.in399, %9712		; visa id: 12742
  %9714 = shl nsw i64 %9691, 2		; visa id: 12743
  %9715 = add i64 %9713, %9714		; visa id: 12744
  %9716 = inttoptr i64 %9715 to float addrspace(4)*		; visa id: 12745
  %9717 = addrspacecast float addrspace(4)* %9716 to float addrspace(1)*		; visa id: 12745
  %9718 = load float, float addrspace(1)* %9717, align 4		; visa id: 12746
  %9719 = fmul reassoc nsz arcp contract float %9718, %4, !spirv.Decorations !618		; visa id: 12747
  %9720 = fadd reassoc nsz arcp contract float %9699, %9719, !spirv.Decorations !618		; visa id: 12748
  %9721 = shl i64 %9698, 2		; visa id: 12749
  %9722 = add i64 %.in, %9721		; visa id: 12750
  %9723 = inttoptr i64 %9722 to float addrspace(4)*		; visa id: 12751
  %9724 = addrspacecast float addrspace(4)* %9723 to float addrspace(1)*		; visa id: 12751
  store float %9720, float addrspace(1)* %9724, align 4		; visa id: 12752
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12753

._crit_edge70.2.15:                               ; preds = %.._crit_edge70.2.15_crit_edge, %9705, %9700
; BB1089 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12754
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12754
  %9725 = insertelement <2 x i32> %6169, i32 %9547, i64 1		; visa id: 12754
  store <2 x i32> %9725, <2 x i32>* %5986, align 4, !noalias !642		; visa id: 12755
  br label %._crit_edge398, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12757

._crit_edge398:                                   ; preds = %._crit_edge398.._crit_edge398_crit_edge, %._crit_edge70.2.15
; BB1090 :
  %9726 = phi i32 [ 0, %._crit_edge70.2.15 ], [ %9735, %._crit_edge398.._crit_edge398_crit_edge ]
  %9727 = zext i32 %9726 to i64		; visa id: 12758
  %9728 = shl nuw nsw i64 %9727, 2		; visa id: 12759
  %9729 = add i64 %5982, %9728		; visa id: 12760
  %9730 = inttoptr i64 %9729 to i32*		; visa id: 12761
  %9731 = load i32, i32* %9730, align 4, !noalias !642		; visa id: 12761
  %9732 = add i64 %5978, %9728		; visa id: 12762
  %9733 = inttoptr i64 %9732 to i32*		; visa id: 12763
  store i32 %9731, i32* %9733, align 4, !alias.scope !642		; visa id: 12763
  %9734 = icmp eq i32 %9726, 0		; visa id: 12764
  br i1 %9734, label %._crit_edge398.._crit_edge398_crit_edge, label %9736, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12765

._crit_edge398.._crit_edge398_crit_edge:          ; preds = %._crit_edge398
; BB1091 :
  %9735 = add nuw nsw i32 %9726, 1, !spirv.Decorations !631		; visa id: 12767
  br label %._crit_edge398, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12768

9736:                                             ; preds = %._crit_edge398
; BB1092 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5983)		; visa id: 12770
  %9737 = load i64, i64* %5998, align 8		; visa id: 12770
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5979)		; visa id: 12771
  %9738 = icmp slt i32 %6168, %const_reg_dword
  %9739 = icmp slt i32 %9547, %const_reg_dword1		; visa id: 12771
  %9740 = and i1 %9738, %9739		; visa id: 12772
  br i1 %9740, label %9741, label %..preheader1.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12774

..preheader1.15_crit_edge:                        ; preds = %9736
; BB:
  br label %.preheader1.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9741:                                             ; preds = %9736
; BB1094 :
  %9742 = bitcast i64 %9737 to <2 x i32>		; visa id: 12776
  %9743 = extractelement <2 x i32> %9742, i32 0		; visa id: 12778
  %9744 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9743, i32 1
  %9745 = bitcast <2 x i32> %9744 to i64		; visa id: 12778
  %9746 = ashr exact i64 %9745, 32		; visa id: 12779
  %9747 = bitcast i64 %9746 to <2 x i32>		; visa id: 12780
  %9748 = extractelement <2 x i32> %9747, i32 0		; visa id: 12784
  %9749 = extractelement <2 x i32> %9747, i32 1		; visa id: 12784
  %9750 = ashr i64 %9737, 32		; visa id: 12784
  %9751 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9748, i32 %9749, i32 %50, i32 %51)
  %9752 = extractvalue { i32, i32 } %9751, 0		; visa id: 12785
  %9753 = extractvalue { i32, i32 } %9751, 1		; visa id: 12785
  %9754 = insertelement <2 x i32> undef, i32 %9752, i32 0		; visa id: 12792
  %9755 = insertelement <2 x i32> %9754, i32 %9753, i32 1		; visa id: 12793
  %9756 = bitcast <2 x i32> %9755 to i64		; visa id: 12794
  %9757 = add nsw i64 %9756, %9750, !spirv.Decorations !649		; visa id: 12798
  %9758 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !618		; visa id: 12799
  br i1 %86, label %9764, label %9759, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12800

9759:                                             ; preds = %9741
; BB1095 :
  %9760 = shl i64 %9757, 2		; visa id: 12802
  %9761 = add i64 %.in, %9760		; visa id: 12803
  %9762 = inttoptr i64 %9761 to float addrspace(4)*		; visa id: 12804
  %9763 = addrspacecast float addrspace(4)* %9762 to float addrspace(1)*		; visa id: 12804
  store float %9758, float addrspace(1)* %9763, align 4		; visa id: 12805
  br label %.preheader1.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12806

9764:                                             ; preds = %9741
; BB1096 :
  %9765 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9748, i32 %9749, i32 %47, i32 %48)
  %9766 = extractvalue { i32, i32 } %9765, 0		; visa id: 12808
  %9767 = extractvalue { i32, i32 } %9765, 1		; visa id: 12808
  %9768 = insertelement <2 x i32> undef, i32 %9766, i32 0		; visa id: 12815
  %9769 = insertelement <2 x i32> %9768, i32 %9767, i32 1		; visa id: 12816
  %9770 = bitcast <2 x i32> %9769 to i64		; visa id: 12817
  %9771 = shl i64 %9770, 2		; visa id: 12821
  %9772 = add i64 %.in399, %9771		; visa id: 12822
  %9773 = shl nsw i64 %9750, 2		; visa id: 12823
  %9774 = add i64 %9772, %9773		; visa id: 12824
  %9775 = inttoptr i64 %9774 to float addrspace(4)*		; visa id: 12825
  %9776 = addrspacecast float addrspace(4)* %9775 to float addrspace(1)*		; visa id: 12825
  %9777 = load float, float addrspace(1)* %9776, align 4		; visa id: 12826
  %9778 = fmul reassoc nsz arcp contract float %9777, %4, !spirv.Decorations !618		; visa id: 12827
  %9779 = fadd reassoc nsz arcp contract float %9758, %9778, !spirv.Decorations !618		; visa id: 12828
  %9780 = shl i64 %9757, 2		; visa id: 12829
  %9781 = add i64 %.in, %9780		; visa id: 12830
  %9782 = inttoptr i64 %9781 to float addrspace(4)*		; visa id: 12831
  %9783 = addrspacecast float addrspace(4)* %9782 to float addrspace(1)*		; visa id: 12831
  store float %9779, float addrspace(1)* %9783, align 4		; visa id: 12832
  br label %.preheader1.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12833

.preheader1.15:                                   ; preds = %..preheader1.15_crit_edge, %9764, %9759
; BB1097 :
  %9784 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %15, i32 %16)
  %9785 = extractvalue { i32, i32 } %9784, 0		; visa id: 12834
  %9786 = extractvalue { i32, i32 } %9784, 1		; visa id: 12834
  %9787 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %18, i32 %19)
  %9788 = extractvalue { i32, i32 } %9787, 0		; visa id: 12841
  %9789 = extractvalue { i32, i32 } %9787, 1		; visa id: 12841
  %9790 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %21, i32 %22)
  %9791 = extractvalue { i32, i32 } %9790, 0		; visa id: 12848
  %9792 = extractvalue { i32, i32 } %9790, 1		; visa id: 12848
  %9793 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %24, i32 %25)
  %9794 = extractvalue { i32, i32 } %9793, 0		; visa id: 12855
  %9795 = extractvalue { i32, i32 } %9793, 1		; visa id: 12855
  %9796 = add i32 %110, %57		; visa id: 12862
  %9797 = icmp slt i32 %9796, %8		; visa id: 12863
  br i1 %9797, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !llvm.loop !652, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12864

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !616, !stats.blockFrequency.scale !615

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
; BB1099 :
  %9798 = insertelement <2 x i32> undef, i32 %9785, i32 0		; visa id: 12866
  %9799 = insertelement <2 x i32> %9798, i32 %9786, i32 1		; visa id: 12867
  %9800 = bitcast <2 x i32> %9799 to i64		; visa id: 12868
  %9801 = shl i64 %9800, 1		; visa id: 12870
  %9802 = add i64 %.in401, %9801		; visa id: 12871
  %9803 = insertelement <2 x i32> undef, i32 %9788, i32 0		; visa id: 12872
  %9804 = insertelement <2 x i32> %9803, i32 %9789, i32 1		; visa id: 12873
  %9805 = bitcast <2 x i32> %9804 to i64		; visa id: 12874
  %9806 = shl i64 %9805, 1		; visa id: 12876
  %9807 = add i64 %.in400, %9806		; visa id: 12877
  %9808 = insertelement <2 x i32> undef, i32 %9791, i32 0		; visa id: 12878
  %9809 = insertelement <2 x i32> %9808, i32 %9792, i32 1		; visa id: 12879
  %9810 = bitcast <2 x i32> %9809 to i64		; visa id: 12880
  %.op402 = shl i64 %9810, 2		; visa id: 12882
  %9811 = bitcast i64 %.op402 to <2 x i32>		; visa id: 12883
  %9812 = extractelement <2 x i32> %9811, i32 0		; visa id: 12884
  %9813 = extractelement <2 x i32> %9811, i32 1		; visa id: 12884
  %9814 = select i1 %86, i32 %9812, i32 0		; visa id: 12884
  %9815 = select i1 %86, i32 %9813, i32 0		; visa id: 12885
  %9816 = insertelement <2 x i32> undef, i32 %9814, i32 0		; visa id: 12886
  %9817 = insertelement <2 x i32> %9816, i32 %9815, i32 1		; visa id: 12887
  %9818 = bitcast <2 x i32> %9817 to i64		; visa id: 12888
  %9819 = add i64 %.in399, %9818		; visa id: 12890
  %9820 = insertelement <2 x i32> undef, i32 %9794, i32 0		; visa id: 12891
  %9821 = insertelement <2 x i32> %9820, i32 %9795, i32 1		; visa id: 12892
  %9822 = bitcast <2 x i32> %9821 to i64		; visa id: 12893
  %9823 = shl i64 %9822, 2		; visa id: 12895
  %9824 = add i64 %.in, %9823		; visa id: 12896
  br label %.preheader2.preheader, !stats.blockFrequency.digits !653, !stats.blockFrequency.scale !615		; visa id: 12897

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
; BB1100 :
  ret void, !stats.blockFrequency.digits !614, !stats.blockFrequency.scale !615		; visa id: 12899
}

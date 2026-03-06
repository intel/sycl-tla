; ------------------------------------------------
; OCL_asm23954d4a795eca46_1_simd32_entry_0013.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9) #2 {
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
  %110 = phi i32 [ %10244, %.preheader1.15..preheader2.preheader_crit_edge ], [ %26, %.preheader2.preheader.preheader ]
  %.in = phi i64 [ %10272, %.preheader1.15..preheader2.preheader_crit_edge ], [ %97, %.preheader2.preheader.preheader ]
  %.in399 = phi i64 [ %10267, %.preheader1.15..preheader2.preheader_crit_edge ], [ %92, %.preheader2.preheader.preheader ]
  %.in400 = phi i64 [ %10255, %.preheader1.15..preheader2.preheader_crit_edge ], [ %79, %.preheader2.preheader.preheader ]
  %.in401 = phi i64 [ %10250, %.preheader1.15..preheader2.preheader_crit_edge ], [ %74, %.preheader2.preheader.preheader ]
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
  %113 = phi i32 [ %6422, %.preheader.15..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
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
  %194 = bitcast i64 %193 to <2 x i32>		; visa id: 328
  %195 = extractelement <2 x i32> %194, i32 0		; visa id: 330
  %196 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %195, i32 1
  %197 = bitcast <2 x i32> %196 to i64		; visa id: 330
  %198 = ashr exact i64 %197, 32		; visa id: 331
  %199 = bitcast i64 %198 to <2 x i32>		; visa id: 332
  %200 = extractelement <2 x i32> %199, i32 0		; visa id: 336
  %201 = extractelement <2 x i32> %199, i32 1		; visa id: 336
  %202 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %200, i32 %201, i32 %44, i32 %45)
  %203 = extractvalue { i32, i32 } %202, 0		; visa id: 336
  %204 = extractvalue { i32, i32 } %202, 1		; visa id: 336
  %205 = insertelement <2 x i32> undef, i32 %203, i32 0		; visa id: 343
  %206 = insertelement <2 x i32> %205, i32 %204, i32 1		; visa id: 344
  %207 = bitcast <2 x i32> %206 to i64		; visa id: 345
  %208 = shl i64 %207, 1		; visa id: 349
  %209 = add i64 %.in400, %208		; visa id: 350
  %210 = ashr i64 %193, 31		; visa id: 351
  %211 = bitcast i64 %210 to <2 x i32>		; visa id: 352
  %212 = extractelement <2 x i32> %211, i32 0		; visa id: 356
  %213 = extractelement <2 x i32> %211, i32 1		; visa id: 356
  %214 = and i32 %212, -2		; visa id: 356
  %215 = insertelement <2 x i32> undef, i32 %214, i32 0		; visa id: 357
  %216 = insertelement <2 x i32> %215, i32 %213, i32 1		; visa id: 358
  %217 = bitcast <2 x i32> %216 to i64		; visa id: 359
  %218 = add i64 %209, %217		; visa id: 363
  %219 = inttoptr i64 %218 to i16 addrspace(4)*		; visa id: 364
  %220 = addrspacecast i16 addrspace(4)* %219 to i16 addrspace(1)*		; visa id: 364
  %221 = load i16, i16 addrspace(1)* %220, align 2		; visa id: 365
  %222 = zext i16 %178 to i32		; visa id: 367
  %223 = shl nuw i32 %222, 16, !spirv.Decorations !639		; visa id: 368
  %224 = bitcast i32 %223 to float
  %225 = zext i16 %221 to i32		; visa id: 369
  %226 = shl nuw i32 %225, 16, !spirv.Decorations !639		; visa id: 370
  %227 = bitcast i32 %226 to float
  %228 = fmul reassoc nsz arcp contract float %224, %227, !spirv.Decorations !618
  %229 = fadd reassoc nsz arcp contract float %228, %.sroa.0.1, !spirv.Decorations !618		; visa id: 371
  br label %._crit_edge, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 372

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %192
; BB15 :
  %.sroa.0.2 = phi float [ %229, %192 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  %230 = add i32 %65, 1		; visa id: 373
  %231 = icmp slt i32 %230, %const_reg_dword
  %232 = icmp slt i32 %69, %const_reg_dword1		; visa id: 374
  %233 = and i1 %231, %232		; visa id: 375
  br i1 %233, label %234, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 377

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
; BB:
  br label %._crit_edge.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

234:                                              ; preds = %._crit_edge
; BB17 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 379
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 379
  %235 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 379
  %236 = insertelement <2 x i32> %235, i32 %113, i64 1		; visa id: 380
  %237 = inttoptr i64 %133 to <2 x i32>*		; visa id: 381
  store <2 x i32> %236, <2 x i32>* %237, align 4, !noalias !625		; visa id: 381
  br label %._crit_edge209, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 383

._crit_edge209:                                   ; preds = %._crit_edge209.._crit_edge209_crit_edge, %234
; BB18 :
  %238 = phi i32 [ 0, %234 ], [ %247, %._crit_edge209.._crit_edge209_crit_edge ]
  %239 = zext i32 %238 to i64		; visa id: 384
  %240 = shl nuw nsw i64 %239, 2		; visa id: 385
  %241 = add i64 %133, %240		; visa id: 386
  %242 = inttoptr i64 %241 to i32*		; visa id: 387
  %243 = load i32, i32* %242, align 4, !noalias !625		; visa id: 387
  %244 = add i64 %128, %240		; visa id: 388
  %245 = inttoptr i64 %244 to i32*		; visa id: 389
  store i32 %243, i32* %245, align 4, !alias.scope !625		; visa id: 389
  %246 = icmp eq i32 %238, 0		; visa id: 390
  br i1 %246, label %._crit_edge209.._crit_edge209_crit_edge, label %248, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 391

._crit_edge209.._crit_edge209_crit_edge:          ; preds = %._crit_edge209
; BB19 :
  %247 = add nuw nsw i32 %238, 1, !spirv.Decorations !631		; visa id: 393
  br label %._crit_edge209, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 394

248:                                              ; preds = %._crit_edge209
; BB20 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 396
  %249 = load i64, i64* %129, align 8		; visa id: 396
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 397
  %250 = bitcast i64 %249 to <2 x i32>		; visa id: 397
  %251 = extractelement <2 x i32> %250, i32 0		; visa id: 399
  %252 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %251, i32 1
  %253 = bitcast <2 x i32> %252 to i64		; visa id: 399
  %254 = ashr exact i64 %253, 32		; visa id: 400
  %255 = bitcast i64 %254 to <2 x i32>		; visa id: 401
  %256 = extractelement <2 x i32> %255, i32 0		; visa id: 405
  %257 = extractelement <2 x i32> %255, i32 1		; visa id: 405
  %258 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %256, i32 %257, i32 %41, i32 %42)
  %259 = extractvalue { i32, i32 } %258, 0		; visa id: 405
  %260 = extractvalue { i32, i32 } %258, 1		; visa id: 405
  %261 = insertelement <2 x i32> undef, i32 %259, i32 0		; visa id: 412
  %262 = insertelement <2 x i32> %261, i32 %260, i32 1		; visa id: 413
  %263 = bitcast <2 x i32> %262 to i64		; visa id: 414
  %264 = shl i64 %263, 1		; visa id: 418
  %265 = add i64 %.in401, %264		; visa id: 419
  %266 = ashr i64 %249, 31		; visa id: 420
  %267 = bitcast i64 %266 to <2 x i32>		; visa id: 421
  %268 = extractelement <2 x i32> %267, i32 0		; visa id: 425
  %269 = extractelement <2 x i32> %267, i32 1		; visa id: 425
  %270 = and i32 %268, -2		; visa id: 425
  %271 = insertelement <2 x i32> undef, i32 %270, i32 0		; visa id: 426
  %272 = insertelement <2 x i32> %271, i32 %269, i32 1		; visa id: 427
  %273 = bitcast <2 x i32> %272 to i64		; visa id: 428
  %274 = add i64 %265, %273		; visa id: 432
  %275 = inttoptr i64 %274 to i16 addrspace(4)*		; visa id: 433
  %276 = addrspacecast i16 addrspace(4)* %275 to i16 addrspace(1)*		; visa id: 433
  %277 = load i16, i16 addrspace(1)* %276, align 2		; visa id: 434
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 436
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 436
  %278 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 436
  %279 = insertelement <2 x i32> %278, i32 %69, i64 1		; visa id: 437
  %280 = inttoptr i64 %124 to <2 x i32>*		; visa id: 438
  store <2 x i32> %279, <2 x i32>* %280, align 4, !noalias !635		; visa id: 438
  br label %._crit_edge210, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 440

._crit_edge210:                                   ; preds = %._crit_edge210.._crit_edge210_crit_edge, %248
; BB21 :
  %281 = phi i32 [ 0, %248 ], [ %290, %._crit_edge210.._crit_edge210_crit_edge ]
  %282 = zext i32 %281 to i64		; visa id: 441
  %283 = shl nuw nsw i64 %282, 2		; visa id: 442
  %284 = add i64 %124, %283		; visa id: 443
  %285 = inttoptr i64 %284 to i32*		; visa id: 444
  %286 = load i32, i32* %285, align 4, !noalias !635		; visa id: 444
  %287 = add i64 %119, %283		; visa id: 445
  %288 = inttoptr i64 %287 to i32*		; visa id: 446
  store i32 %286, i32* %288, align 4, !alias.scope !635		; visa id: 446
  %289 = icmp eq i32 %281, 0		; visa id: 447
  br i1 %289, label %._crit_edge210.._crit_edge210_crit_edge, label %291, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 448

._crit_edge210.._crit_edge210_crit_edge:          ; preds = %._crit_edge210
; BB22 :
  %290 = add nuw nsw i32 %281, 1, !spirv.Decorations !631		; visa id: 450
  br label %._crit_edge210, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 451

291:                                              ; preds = %._crit_edge210
; BB23 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 453
  %292 = load i64, i64* %120, align 8		; visa id: 453
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 454
  %293 = bitcast i64 %292 to <2 x i32>		; visa id: 454
  %294 = extractelement <2 x i32> %293, i32 0		; visa id: 456
  %295 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %294, i32 1
  %296 = bitcast <2 x i32> %295 to i64		; visa id: 456
  %297 = ashr exact i64 %296, 32		; visa id: 457
  %298 = bitcast i64 %297 to <2 x i32>		; visa id: 458
  %299 = extractelement <2 x i32> %298, i32 0		; visa id: 462
  %300 = extractelement <2 x i32> %298, i32 1		; visa id: 462
  %301 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %299, i32 %300, i32 %44, i32 %45)
  %302 = extractvalue { i32, i32 } %301, 0		; visa id: 462
  %303 = extractvalue { i32, i32 } %301, 1		; visa id: 462
  %304 = insertelement <2 x i32> undef, i32 %302, i32 0		; visa id: 469
  %305 = insertelement <2 x i32> %304, i32 %303, i32 1		; visa id: 470
  %306 = bitcast <2 x i32> %305 to i64		; visa id: 471
  %307 = shl i64 %306, 1		; visa id: 475
  %308 = add i64 %.in400, %307		; visa id: 476
  %309 = ashr i64 %292, 31		; visa id: 477
  %310 = bitcast i64 %309 to <2 x i32>		; visa id: 478
  %311 = extractelement <2 x i32> %310, i32 0		; visa id: 482
  %312 = extractelement <2 x i32> %310, i32 1		; visa id: 482
  %313 = and i32 %311, -2		; visa id: 482
  %314 = insertelement <2 x i32> undef, i32 %313, i32 0		; visa id: 483
  %315 = insertelement <2 x i32> %314, i32 %312, i32 1		; visa id: 484
  %316 = bitcast <2 x i32> %315 to i64		; visa id: 485
  %317 = add i64 %308, %316		; visa id: 489
  %318 = inttoptr i64 %317 to i16 addrspace(4)*		; visa id: 490
  %319 = addrspacecast i16 addrspace(4)* %318 to i16 addrspace(1)*		; visa id: 490
  %320 = load i16, i16 addrspace(1)* %319, align 2		; visa id: 491
  %321 = zext i16 %277 to i32		; visa id: 493
  %322 = shl nuw i32 %321, 16, !spirv.Decorations !639		; visa id: 494
  %323 = bitcast i32 %322 to float
  %324 = zext i16 %320 to i32		; visa id: 495
  %325 = shl nuw i32 %324, 16, !spirv.Decorations !639		; visa id: 496
  %326 = bitcast i32 %325 to float
  %327 = fmul reassoc nsz arcp contract float %323, %326, !spirv.Decorations !618
  %328 = fadd reassoc nsz arcp contract float %327, %.sroa.66.1, !spirv.Decorations !618		; visa id: 497
  br label %._crit_edge.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 498

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %291
; BB24 :
  %.sroa.66.2 = phi float [ %328, %291 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  %329 = add i32 %65, 2		; visa id: 499
  %330 = icmp slt i32 %329, %const_reg_dword
  %331 = icmp slt i32 %69, %const_reg_dword1		; visa id: 500
  %332 = and i1 %330, %331		; visa id: 501
  br i1 %332, label %333, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 503

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
; BB:
  br label %._crit_edge.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

333:                                              ; preds = %._crit_edge.1
; BB26 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 505
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 505
  %334 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 505
  %335 = insertelement <2 x i32> %334, i32 %113, i64 1		; visa id: 506
  %336 = inttoptr i64 %133 to <2 x i32>*		; visa id: 507
  store <2 x i32> %335, <2 x i32>* %336, align 4, !noalias !625		; visa id: 507
  br label %._crit_edge211, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 509

._crit_edge211:                                   ; preds = %._crit_edge211.._crit_edge211_crit_edge, %333
; BB27 :
  %337 = phi i32 [ 0, %333 ], [ %346, %._crit_edge211.._crit_edge211_crit_edge ]
  %338 = zext i32 %337 to i64		; visa id: 510
  %339 = shl nuw nsw i64 %338, 2		; visa id: 511
  %340 = add i64 %133, %339		; visa id: 512
  %341 = inttoptr i64 %340 to i32*		; visa id: 513
  %342 = load i32, i32* %341, align 4, !noalias !625		; visa id: 513
  %343 = add i64 %128, %339		; visa id: 514
  %344 = inttoptr i64 %343 to i32*		; visa id: 515
  store i32 %342, i32* %344, align 4, !alias.scope !625		; visa id: 515
  %345 = icmp eq i32 %337, 0		; visa id: 516
  br i1 %345, label %._crit_edge211.._crit_edge211_crit_edge, label %347, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 517

._crit_edge211.._crit_edge211_crit_edge:          ; preds = %._crit_edge211
; BB28 :
  %346 = add nuw nsw i32 %337, 1, !spirv.Decorations !631		; visa id: 519
  br label %._crit_edge211, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 520

347:                                              ; preds = %._crit_edge211
; BB29 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 522
  %348 = load i64, i64* %129, align 8		; visa id: 522
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 523
  %349 = bitcast i64 %348 to <2 x i32>		; visa id: 523
  %350 = extractelement <2 x i32> %349, i32 0		; visa id: 525
  %351 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %350, i32 1
  %352 = bitcast <2 x i32> %351 to i64		; visa id: 525
  %353 = ashr exact i64 %352, 32		; visa id: 526
  %354 = bitcast i64 %353 to <2 x i32>		; visa id: 527
  %355 = extractelement <2 x i32> %354, i32 0		; visa id: 531
  %356 = extractelement <2 x i32> %354, i32 1		; visa id: 531
  %357 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %355, i32 %356, i32 %41, i32 %42)
  %358 = extractvalue { i32, i32 } %357, 0		; visa id: 531
  %359 = extractvalue { i32, i32 } %357, 1		; visa id: 531
  %360 = insertelement <2 x i32> undef, i32 %358, i32 0		; visa id: 538
  %361 = insertelement <2 x i32> %360, i32 %359, i32 1		; visa id: 539
  %362 = bitcast <2 x i32> %361 to i64		; visa id: 540
  %363 = shl i64 %362, 1		; visa id: 544
  %364 = add i64 %.in401, %363		; visa id: 545
  %365 = ashr i64 %348, 31		; visa id: 546
  %366 = bitcast i64 %365 to <2 x i32>		; visa id: 547
  %367 = extractelement <2 x i32> %366, i32 0		; visa id: 551
  %368 = extractelement <2 x i32> %366, i32 1		; visa id: 551
  %369 = and i32 %367, -2		; visa id: 551
  %370 = insertelement <2 x i32> undef, i32 %369, i32 0		; visa id: 552
  %371 = insertelement <2 x i32> %370, i32 %368, i32 1		; visa id: 553
  %372 = bitcast <2 x i32> %371 to i64		; visa id: 554
  %373 = add i64 %364, %372		; visa id: 558
  %374 = inttoptr i64 %373 to i16 addrspace(4)*		; visa id: 559
  %375 = addrspacecast i16 addrspace(4)* %374 to i16 addrspace(1)*		; visa id: 559
  %376 = load i16, i16 addrspace(1)* %375, align 2		; visa id: 560
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 562
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 562
  %377 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 562
  %378 = insertelement <2 x i32> %377, i32 %69, i64 1		; visa id: 563
  %379 = inttoptr i64 %124 to <2 x i32>*		; visa id: 564
  store <2 x i32> %378, <2 x i32>* %379, align 4, !noalias !635		; visa id: 564
  br label %._crit_edge212, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 566

._crit_edge212:                                   ; preds = %._crit_edge212.._crit_edge212_crit_edge, %347
; BB30 :
  %380 = phi i32 [ 0, %347 ], [ %389, %._crit_edge212.._crit_edge212_crit_edge ]
  %381 = zext i32 %380 to i64		; visa id: 567
  %382 = shl nuw nsw i64 %381, 2		; visa id: 568
  %383 = add i64 %124, %382		; visa id: 569
  %384 = inttoptr i64 %383 to i32*		; visa id: 570
  %385 = load i32, i32* %384, align 4, !noalias !635		; visa id: 570
  %386 = add i64 %119, %382		; visa id: 571
  %387 = inttoptr i64 %386 to i32*		; visa id: 572
  store i32 %385, i32* %387, align 4, !alias.scope !635		; visa id: 572
  %388 = icmp eq i32 %380, 0		; visa id: 573
  br i1 %388, label %._crit_edge212.._crit_edge212_crit_edge, label %390, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 574

._crit_edge212.._crit_edge212_crit_edge:          ; preds = %._crit_edge212
; BB31 :
  %389 = add nuw nsw i32 %380, 1, !spirv.Decorations !631		; visa id: 576
  br label %._crit_edge212, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 577

390:                                              ; preds = %._crit_edge212
; BB32 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 579
  %391 = load i64, i64* %120, align 8		; visa id: 579
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 580
  %392 = bitcast i64 %391 to <2 x i32>		; visa id: 580
  %393 = extractelement <2 x i32> %392, i32 0		; visa id: 582
  %394 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %393, i32 1
  %395 = bitcast <2 x i32> %394 to i64		; visa id: 582
  %396 = ashr exact i64 %395, 32		; visa id: 583
  %397 = bitcast i64 %396 to <2 x i32>		; visa id: 584
  %398 = extractelement <2 x i32> %397, i32 0		; visa id: 588
  %399 = extractelement <2 x i32> %397, i32 1		; visa id: 588
  %400 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %398, i32 %399, i32 %44, i32 %45)
  %401 = extractvalue { i32, i32 } %400, 0		; visa id: 588
  %402 = extractvalue { i32, i32 } %400, 1		; visa id: 588
  %403 = insertelement <2 x i32> undef, i32 %401, i32 0		; visa id: 595
  %404 = insertelement <2 x i32> %403, i32 %402, i32 1		; visa id: 596
  %405 = bitcast <2 x i32> %404 to i64		; visa id: 597
  %406 = shl i64 %405, 1		; visa id: 601
  %407 = add i64 %.in400, %406		; visa id: 602
  %408 = ashr i64 %391, 31		; visa id: 603
  %409 = bitcast i64 %408 to <2 x i32>		; visa id: 604
  %410 = extractelement <2 x i32> %409, i32 0		; visa id: 608
  %411 = extractelement <2 x i32> %409, i32 1		; visa id: 608
  %412 = and i32 %410, -2		; visa id: 608
  %413 = insertelement <2 x i32> undef, i32 %412, i32 0		; visa id: 609
  %414 = insertelement <2 x i32> %413, i32 %411, i32 1		; visa id: 610
  %415 = bitcast <2 x i32> %414 to i64		; visa id: 611
  %416 = add i64 %407, %415		; visa id: 615
  %417 = inttoptr i64 %416 to i16 addrspace(4)*		; visa id: 616
  %418 = addrspacecast i16 addrspace(4)* %417 to i16 addrspace(1)*		; visa id: 616
  %419 = load i16, i16 addrspace(1)* %418, align 2		; visa id: 617
  %420 = zext i16 %376 to i32		; visa id: 619
  %421 = shl nuw i32 %420, 16, !spirv.Decorations !639		; visa id: 620
  %422 = bitcast i32 %421 to float
  %423 = zext i16 %419 to i32		; visa id: 621
  %424 = shl nuw i32 %423, 16, !spirv.Decorations !639		; visa id: 622
  %425 = bitcast i32 %424 to float
  %426 = fmul reassoc nsz arcp contract float %422, %425, !spirv.Decorations !618
  %427 = fadd reassoc nsz arcp contract float %426, %.sroa.130.1, !spirv.Decorations !618		; visa id: 623
  br label %._crit_edge.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 624

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %390
; BB33 :
  %.sroa.130.2 = phi float [ %427, %390 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  %428 = add i32 %65, 3		; visa id: 625
  %429 = icmp slt i32 %428, %const_reg_dword
  %430 = icmp slt i32 %69, %const_reg_dword1		; visa id: 626
  %431 = and i1 %429, %430		; visa id: 627
  br i1 %431, label %432, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 629

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
; BB:
  br label %.preheader, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

432:                                              ; preds = %._crit_edge.2
; BB35 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 631
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 631
  %433 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 631
  %434 = insertelement <2 x i32> %433, i32 %113, i64 1		; visa id: 632
  %435 = inttoptr i64 %133 to <2 x i32>*		; visa id: 633
  store <2 x i32> %434, <2 x i32>* %435, align 4, !noalias !625		; visa id: 633
  br label %._crit_edge213, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 635

._crit_edge213:                                   ; preds = %._crit_edge213.._crit_edge213_crit_edge, %432
; BB36 :
  %436 = phi i32 [ 0, %432 ], [ %445, %._crit_edge213.._crit_edge213_crit_edge ]
  %437 = zext i32 %436 to i64		; visa id: 636
  %438 = shl nuw nsw i64 %437, 2		; visa id: 637
  %439 = add i64 %133, %438		; visa id: 638
  %440 = inttoptr i64 %439 to i32*		; visa id: 639
  %441 = load i32, i32* %440, align 4, !noalias !625		; visa id: 639
  %442 = add i64 %128, %438		; visa id: 640
  %443 = inttoptr i64 %442 to i32*		; visa id: 641
  store i32 %441, i32* %443, align 4, !alias.scope !625		; visa id: 641
  %444 = icmp eq i32 %436, 0		; visa id: 642
  br i1 %444, label %._crit_edge213.._crit_edge213_crit_edge, label %446, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 643

._crit_edge213.._crit_edge213_crit_edge:          ; preds = %._crit_edge213
; BB37 :
  %445 = add nuw nsw i32 %436, 1, !spirv.Decorations !631		; visa id: 645
  br label %._crit_edge213, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 646

446:                                              ; preds = %._crit_edge213
; BB38 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 648
  %447 = load i64, i64* %129, align 8		; visa id: 648
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 649
  %448 = bitcast i64 %447 to <2 x i32>		; visa id: 649
  %449 = extractelement <2 x i32> %448, i32 0		; visa id: 651
  %450 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %449, i32 1
  %451 = bitcast <2 x i32> %450 to i64		; visa id: 651
  %452 = ashr exact i64 %451, 32		; visa id: 652
  %453 = bitcast i64 %452 to <2 x i32>		; visa id: 653
  %454 = extractelement <2 x i32> %453, i32 0		; visa id: 657
  %455 = extractelement <2 x i32> %453, i32 1		; visa id: 657
  %456 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %454, i32 %455, i32 %41, i32 %42)
  %457 = extractvalue { i32, i32 } %456, 0		; visa id: 657
  %458 = extractvalue { i32, i32 } %456, 1		; visa id: 657
  %459 = insertelement <2 x i32> undef, i32 %457, i32 0		; visa id: 664
  %460 = insertelement <2 x i32> %459, i32 %458, i32 1		; visa id: 665
  %461 = bitcast <2 x i32> %460 to i64		; visa id: 666
  %462 = shl i64 %461, 1		; visa id: 670
  %463 = add i64 %.in401, %462		; visa id: 671
  %464 = ashr i64 %447, 31		; visa id: 672
  %465 = bitcast i64 %464 to <2 x i32>		; visa id: 673
  %466 = extractelement <2 x i32> %465, i32 0		; visa id: 677
  %467 = extractelement <2 x i32> %465, i32 1		; visa id: 677
  %468 = and i32 %466, -2		; visa id: 677
  %469 = insertelement <2 x i32> undef, i32 %468, i32 0		; visa id: 678
  %470 = insertelement <2 x i32> %469, i32 %467, i32 1		; visa id: 679
  %471 = bitcast <2 x i32> %470 to i64		; visa id: 680
  %472 = add i64 %463, %471		; visa id: 684
  %473 = inttoptr i64 %472 to i16 addrspace(4)*		; visa id: 685
  %474 = addrspacecast i16 addrspace(4)* %473 to i16 addrspace(1)*		; visa id: 685
  %475 = load i16, i16 addrspace(1)* %474, align 2		; visa id: 686
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 688
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 688
  %476 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 688
  %477 = insertelement <2 x i32> %476, i32 %69, i64 1		; visa id: 689
  %478 = inttoptr i64 %124 to <2 x i32>*		; visa id: 690
  store <2 x i32> %477, <2 x i32>* %478, align 4, !noalias !635		; visa id: 690
  br label %._crit_edge214, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 692

._crit_edge214:                                   ; preds = %._crit_edge214.._crit_edge214_crit_edge, %446
; BB39 :
  %479 = phi i32 [ 0, %446 ], [ %488, %._crit_edge214.._crit_edge214_crit_edge ]
  %480 = zext i32 %479 to i64		; visa id: 693
  %481 = shl nuw nsw i64 %480, 2		; visa id: 694
  %482 = add i64 %124, %481		; visa id: 695
  %483 = inttoptr i64 %482 to i32*		; visa id: 696
  %484 = load i32, i32* %483, align 4, !noalias !635		; visa id: 696
  %485 = add i64 %119, %481		; visa id: 697
  %486 = inttoptr i64 %485 to i32*		; visa id: 698
  store i32 %484, i32* %486, align 4, !alias.scope !635		; visa id: 698
  %487 = icmp eq i32 %479, 0		; visa id: 699
  br i1 %487, label %._crit_edge214.._crit_edge214_crit_edge, label %489, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 700

._crit_edge214.._crit_edge214_crit_edge:          ; preds = %._crit_edge214
; BB40 :
  %488 = add nuw nsw i32 %479, 1, !spirv.Decorations !631		; visa id: 702
  br label %._crit_edge214, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 703

489:                                              ; preds = %._crit_edge214
; BB41 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 705
  %490 = load i64, i64* %120, align 8		; visa id: 705
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 706
  %491 = bitcast i64 %490 to <2 x i32>		; visa id: 706
  %492 = extractelement <2 x i32> %491, i32 0		; visa id: 708
  %493 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %492, i32 1
  %494 = bitcast <2 x i32> %493 to i64		; visa id: 708
  %495 = ashr exact i64 %494, 32		; visa id: 709
  %496 = bitcast i64 %495 to <2 x i32>		; visa id: 710
  %497 = extractelement <2 x i32> %496, i32 0		; visa id: 714
  %498 = extractelement <2 x i32> %496, i32 1		; visa id: 714
  %499 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %497, i32 %498, i32 %44, i32 %45)
  %500 = extractvalue { i32, i32 } %499, 0		; visa id: 714
  %501 = extractvalue { i32, i32 } %499, 1		; visa id: 714
  %502 = insertelement <2 x i32> undef, i32 %500, i32 0		; visa id: 721
  %503 = insertelement <2 x i32> %502, i32 %501, i32 1		; visa id: 722
  %504 = bitcast <2 x i32> %503 to i64		; visa id: 723
  %505 = shl i64 %504, 1		; visa id: 727
  %506 = add i64 %.in400, %505		; visa id: 728
  %507 = ashr i64 %490, 31		; visa id: 729
  %508 = bitcast i64 %507 to <2 x i32>		; visa id: 730
  %509 = extractelement <2 x i32> %508, i32 0		; visa id: 734
  %510 = extractelement <2 x i32> %508, i32 1		; visa id: 734
  %511 = and i32 %509, -2		; visa id: 734
  %512 = insertelement <2 x i32> undef, i32 %511, i32 0		; visa id: 735
  %513 = insertelement <2 x i32> %512, i32 %510, i32 1		; visa id: 736
  %514 = bitcast <2 x i32> %513 to i64		; visa id: 737
  %515 = add i64 %506, %514		; visa id: 741
  %516 = inttoptr i64 %515 to i16 addrspace(4)*		; visa id: 742
  %517 = addrspacecast i16 addrspace(4)* %516 to i16 addrspace(1)*		; visa id: 742
  %518 = load i16, i16 addrspace(1)* %517, align 2		; visa id: 743
  %519 = zext i16 %475 to i32		; visa id: 745
  %520 = shl nuw i32 %519, 16, !spirv.Decorations !639		; visa id: 746
  %521 = bitcast i32 %520 to float
  %522 = zext i16 %518 to i32		; visa id: 747
  %523 = shl nuw i32 %522, 16, !spirv.Decorations !639		; visa id: 748
  %524 = bitcast i32 %523 to float
  %525 = fmul reassoc nsz arcp contract float %521, %524, !spirv.Decorations !618
  %526 = fadd reassoc nsz arcp contract float %525, %.sroa.194.1, !spirv.Decorations !618		; visa id: 749
  br label %.preheader, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 750

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %489
; BB42 :
  %.sroa.194.2 = phi float [ %526, %489 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %527 = add i32 %69, 1		; visa id: 751
  %528 = icmp slt i32 %527, %const_reg_dword1		; visa id: 752
  %529 = icmp slt i32 %65, %const_reg_dword
  %530 = and i1 %529, %528		; visa id: 753
  br i1 %530, label %531, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 755

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
; BB:
  br label %._crit_edge.173, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

531:                                              ; preds = %.preheader
; BB44 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 757
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 757
  %532 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 757
  %533 = insertelement <2 x i32> %532, i32 %113, i64 1		; visa id: 758
  %534 = inttoptr i64 %133 to <2 x i32>*		; visa id: 759
  store <2 x i32> %533, <2 x i32>* %534, align 4, !noalias !625		; visa id: 759
  br label %._crit_edge215, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 761

._crit_edge215:                                   ; preds = %._crit_edge215.._crit_edge215_crit_edge, %531
; BB45 :
  %535 = phi i32 [ 0, %531 ], [ %544, %._crit_edge215.._crit_edge215_crit_edge ]
  %536 = zext i32 %535 to i64		; visa id: 762
  %537 = shl nuw nsw i64 %536, 2		; visa id: 763
  %538 = add i64 %133, %537		; visa id: 764
  %539 = inttoptr i64 %538 to i32*		; visa id: 765
  %540 = load i32, i32* %539, align 4, !noalias !625		; visa id: 765
  %541 = add i64 %128, %537		; visa id: 766
  %542 = inttoptr i64 %541 to i32*		; visa id: 767
  store i32 %540, i32* %542, align 4, !alias.scope !625		; visa id: 767
  %543 = icmp eq i32 %535, 0		; visa id: 768
  br i1 %543, label %._crit_edge215.._crit_edge215_crit_edge, label %545, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 769

._crit_edge215.._crit_edge215_crit_edge:          ; preds = %._crit_edge215
; BB46 :
  %544 = add nuw nsw i32 %535, 1, !spirv.Decorations !631		; visa id: 771
  br label %._crit_edge215, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 772

545:                                              ; preds = %._crit_edge215
; BB47 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 774
  %546 = load i64, i64* %129, align 8		; visa id: 774
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 775
  %547 = bitcast i64 %546 to <2 x i32>		; visa id: 775
  %548 = extractelement <2 x i32> %547, i32 0		; visa id: 777
  %549 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %548, i32 1
  %550 = bitcast <2 x i32> %549 to i64		; visa id: 777
  %551 = ashr exact i64 %550, 32		; visa id: 778
  %552 = bitcast i64 %551 to <2 x i32>		; visa id: 779
  %553 = extractelement <2 x i32> %552, i32 0		; visa id: 783
  %554 = extractelement <2 x i32> %552, i32 1		; visa id: 783
  %555 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %553, i32 %554, i32 %41, i32 %42)
  %556 = extractvalue { i32, i32 } %555, 0		; visa id: 783
  %557 = extractvalue { i32, i32 } %555, 1		; visa id: 783
  %558 = insertelement <2 x i32> undef, i32 %556, i32 0		; visa id: 790
  %559 = insertelement <2 x i32> %558, i32 %557, i32 1		; visa id: 791
  %560 = bitcast <2 x i32> %559 to i64		; visa id: 792
  %561 = shl i64 %560, 1		; visa id: 796
  %562 = add i64 %.in401, %561		; visa id: 797
  %563 = ashr i64 %546, 31		; visa id: 798
  %564 = bitcast i64 %563 to <2 x i32>		; visa id: 799
  %565 = extractelement <2 x i32> %564, i32 0		; visa id: 803
  %566 = extractelement <2 x i32> %564, i32 1		; visa id: 803
  %567 = and i32 %565, -2		; visa id: 803
  %568 = insertelement <2 x i32> undef, i32 %567, i32 0		; visa id: 804
  %569 = insertelement <2 x i32> %568, i32 %566, i32 1		; visa id: 805
  %570 = bitcast <2 x i32> %569 to i64		; visa id: 806
  %571 = add i64 %562, %570		; visa id: 810
  %572 = inttoptr i64 %571 to i16 addrspace(4)*		; visa id: 811
  %573 = addrspacecast i16 addrspace(4)* %572 to i16 addrspace(1)*		; visa id: 811
  %574 = load i16, i16 addrspace(1)* %573, align 2		; visa id: 812
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 814
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 814
  %575 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 814
  %576 = insertelement <2 x i32> %575, i32 %527, i64 1		; visa id: 815
  %577 = inttoptr i64 %124 to <2 x i32>*		; visa id: 816
  store <2 x i32> %576, <2 x i32>* %577, align 4, !noalias !635		; visa id: 816
  br label %._crit_edge216, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 818

._crit_edge216:                                   ; preds = %._crit_edge216.._crit_edge216_crit_edge, %545
; BB48 :
  %578 = phi i32 [ 0, %545 ], [ %587, %._crit_edge216.._crit_edge216_crit_edge ]
  %579 = zext i32 %578 to i64		; visa id: 819
  %580 = shl nuw nsw i64 %579, 2		; visa id: 820
  %581 = add i64 %124, %580		; visa id: 821
  %582 = inttoptr i64 %581 to i32*		; visa id: 822
  %583 = load i32, i32* %582, align 4, !noalias !635		; visa id: 822
  %584 = add i64 %119, %580		; visa id: 823
  %585 = inttoptr i64 %584 to i32*		; visa id: 824
  store i32 %583, i32* %585, align 4, !alias.scope !635		; visa id: 824
  %586 = icmp eq i32 %578, 0		; visa id: 825
  br i1 %586, label %._crit_edge216.._crit_edge216_crit_edge, label %588, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 826

._crit_edge216.._crit_edge216_crit_edge:          ; preds = %._crit_edge216
; BB49 :
  %587 = add nuw nsw i32 %578, 1, !spirv.Decorations !631		; visa id: 828
  br label %._crit_edge216, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 829

588:                                              ; preds = %._crit_edge216
; BB50 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 831
  %589 = load i64, i64* %120, align 8		; visa id: 831
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 832
  %590 = bitcast i64 %589 to <2 x i32>		; visa id: 832
  %591 = extractelement <2 x i32> %590, i32 0		; visa id: 834
  %592 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %591, i32 1
  %593 = bitcast <2 x i32> %592 to i64		; visa id: 834
  %594 = ashr exact i64 %593, 32		; visa id: 835
  %595 = bitcast i64 %594 to <2 x i32>		; visa id: 836
  %596 = extractelement <2 x i32> %595, i32 0		; visa id: 840
  %597 = extractelement <2 x i32> %595, i32 1		; visa id: 840
  %598 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %596, i32 %597, i32 %44, i32 %45)
  %599 = extractvalue { i32, i32 } %598, 0		; visa id: 840
  %600 = extractvalue { i32, i32 } %598, 1		; visa id: 840
  %601 = insertelement <2 x i32> undef, i32 %599, i32 0		; visa id: 847
  %602 = insertelement <2 x i32> %601, i32 %600, i32 1		; visa id: 848
  %603 = bitcast <2 x i32> %602 to i64		; visa id: 849
  %604 = shl i64 %603, 1		; visa id: 853
  %605 = add i64 %.in400, %604		; visa id: 854
  %606 = ashr i64 %589, 31		; visa id: 855
  %607 = bitcast i64 %606 to <2 x i32>		; visa id: 856
  %608 = extractelement <2 x i32> %607, i32 0		; visa id: 860
  %609 = extractelement <2 x i32> %607, i32 1		; visa id: 860
  %610 = and i32 %608, -2		; visa id: 860
  %611 = insertelement <2 x i32> undef, i32 %610, i32 0		; visa id: 861
  %612 = insertelement <2 x i32> %611, i32 %609, i32 1		; visa id: 862
  %613 = bitcast <2 x i32> %612 to i64		; visa id: 863
  %614 = add i64 %605, %613		; visa id: 867
  %615 = inttoptr i64 %614 to i16 addrspace(4)*		; visa id: 868
  %616 = addrspacecast i16 addrspace(4)* %615 to i16 addrspace(1)*		; visa id: 868
  %617 = load i16, i16 addrspace(1)* %616, align 2		; visa id: 869
  %618 = zext i16 %574 to i32		; visa id: 871
  %619 = shl nuw i32 %618, 16, !spirv.Decorations !639		; visa id: 872
  %620 = bitcast i32 %619 to float
  %621 = zext i16 %617 to i32		; visa id: 873
  %622 = shl nuw i32 %621, 16, !spirv.Decorations !639		; visa id: 874
  %623 = bitcast i32 %622 to float
  %624 = fmul reassoc nsz arcp contract float %620, %623, !spirv.Decorations !618
  %625 = fadd reassoc nsz arcp contract float %624, %.sroa.6.1, !spirv.Decorations !618		; visa id: 875
  br label %._crit_edge.173, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 876

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %588
; BB51 :
  %.sroa.6.2 = phi float [ %625, %588 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  %626 = icmp slt i32 %230, %const_reg_dword
  %627 = icmp slt i32 %527, %const_reg_dword1		; visa id: 877
  %628 = and i1 %626, %627		; visa id: 878
  br i1 %628, label %629, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 880

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
; BB:
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

629:                                              ; preds = %._crit_edge.173
; BB53 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 882
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 882
  %630 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 882
  %631 = insertelement <2 x i32> %630, i32 %113, i64 1		; visa id: 883
  %632 = inttoptr i64 %133 to <2 x i32>*		; visa id: 884
  store <2 x i32> %631, <2 x i32>* %632, align 4, !noalias !625		; visa id: 884
  br label %._crit_edge217, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 886

._crit_edge217:                                   ; preds = %._crit_edge217.._crit_edge217_crit_edge, %629
; BB54 :
  %633 = phi i32 [ 0, %629 ], [ %642, %._crit_edge217.._crit_edge217_crit_edge ]
  %634 = zext i32 %633 to i64		; visa id: 887
  %635 = shl nuw nsw i64 %634, 2		; visa id: 888
  %636 = add i64 %133, %635		; visa id: 889
  %637 = inttoptr i64 %636 to i32*		; visa id: 890
  %638 = load i32, i32* %637, align 4, !noalias !625		; visa id: 890
  %639 = add i64 %128, %635		; visa id: 891
  %640 = inttoptr i64 %639 to i32*		; visa id: 892
  store i32 %638, i32* %640, align 4, !alias.scope !625		; visa id: 892
  %641 = icmp eq i32 %633, 0		; visa id: 893
  br i1 %641, label %._crit_edge217.._crit_edge217_crit_edge, label %643, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 894

._crit_edge217.._crit_edge217_crit_edge:          ; preds = %._crit_edge217
; BB55 :
  %642 = add nuw nsw i32 %633, 1, !spirv.Decorations !631		; visa id: 896
  br label %._crit_edge217, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 897

643:                                              ; preds = %._crit_edge217
; BB56 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 899
  %644 = load i64, i64* %129, align 8		; visa id: 899
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 900
  %645 = bitcast i64 %644 to <2 x i32>		; visa id: 900
  %646 = extractelement <2 x i32> %645, i32 0		; visa id: 902
  %647 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %646, i32 1
  %648 = bitcast <2 x i32> %647 to i64		; visa id: 902
  %649 = ashr exact i64 %648, 32		; visa id: 903
  %650 = bitcast i64 %649 to <2 x i32>		; visa id: 904
  %651 = extractelement <2 x i32> %650, i32 0		; visa id: 908
  %652 = extractelement <2 x i32> %650, i32 1		; visa id: 908
  %653 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %651, i32 %652, i32 %41, i32 %42)
  %654 = extractvalue { i32, i32 } %653, 0		; visa id: 908
  %655 = extractvalue { i32, i32 } %653, 1		; visa id: 908
  %656 = insertelement <2 x i32> undef, i32 %654, i32 0		; visa id: 915
  %657 = insertelement <2 x i32> %656, i32 %655, i32 1		; visa id: 916
  %658 = bitcast <2 x i32> %657 to i64		; visa id: 917
  %659 = shl i64 %658, 1		; visa id: 921
  %660 = add i64 %.in401, %659		; visa id: 922
  %661 = ashr i64 %644, 31		; visa id: 923
  %662 = bitcast i64 %661 to <2 x i32>		; visa id: 924
  %663 = extractelement <2 x i32> %662, i32 0		; visa id: 928
  %664 = extractelement <2 x i32> %662, i32 1		; visa id: 928
  %665 = and i32 %663, -2		; visa id: 928
  %666 = insertelement <2 x i32> undef, i32 %665, i32 0		; visa id: 929
  %667 = insertelement <2 x i32> %666, i32 %664, i32 1		; visa id: 930
  %668 = bitcast <2 x i32> %667 to i64		; visa id: 931
  %669 = add i64 %660, %668		; visa id: 935
  %670 = inttoptr i64 %669 to i16 addrspace(4)*		; visa id: 936
  %671 = addrspacecast i16 addrspace(4)* %670 to i16 addrspace(1)*		; visa id: 936
  %672 = load i16, i16 addrspace(1)* %671, align 2		; visa id: 937
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 939
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 939
  %673 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 939
  %674 = insertelement <2 x i32> %673, i32 %527, i64 1		; visa id: 940
  %675 = inttoptr i64 %124 to <2 x i32>*		; visa id: 941
  store <2 x i32> %674, <2 x i32>* %675, align 4, !noalias !635		; visa id: 941
  br label %._crit_edge218, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 943

._crit_edge218:                                   ; preds = %._crit_edge218.._crit_edge218_crit_edge, %643
; BB57 :
  %676 = phi i32 [ 0, %643 ], [ %685, %._crit_edge218.._crit_edge218_crit_edge ]
  %677 = zext i32 %676 to i64		; visa id: 944
  %678 = shl nuw nsw i64 %677, 2		; visa id: 945
  %679 = add i64 %124, %678		; visa id: 946
  %680 = inttoptr i64 %679 to i32*		; visa id: 947
  %681 = load i32, i32* %680, align 4, !noalias !635		; visa id: 947
  %682 = add i64 %119, %678		; visa id: 948
  %683 = inttoptr i64 %682 to i32*		; visa id: 949
  store i32 %681, i32* %683, align 4, !alias.scope !635		; visa id: 949
  %684 = icmp eq i32 %676, 0		; visa id: 950
  br i1 %684, label %._crit_edge218.._crit_edge218_crit_edge, label %686, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 951

._crit_edge218.._crit_edge218_crit_edge:          ; preds = %._crit_edge218
; BB58 :
  %685 = add nuw nsw i32 %676, 1, !spirv.Decorations !631		; visa id: 953
  br label %._crit_edge218, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 954

686:                                              ; preds = %._crit_edge218
; BB59 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 956
  %687 = load i64, i64* %120, align 8		; visa id: 956
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 957
  %688 = bitcast i64 %687 to <2 x i32>		; visa id: 957
  %689 = extractelement <2 x i32> %688, i32 0		; visa id: 959
  %690 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %689, i32 1
  %691 = bitcast <2 x i32> %690 to i64		; visa id: 959
  %692 = ashr exact i64 %691, 32		; visa id: 960
  %693 = bitcast i64 %692 to <2 x i32>		; visa id: 961
  %694 = extractelement <2 x i32> %693, i32 0		; visa id: 965
  %695 = extractelement <2 x i32> %693, i32 1		; visa id: 965
  %696 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %694, i32 %695, i32 %44, i32 %45)
  %697 = extractvalue { i32, i32 } %696, 0		; visa id: 965
  %698 = extractvalue { i32, i32 } %696, 1		; visa id: 965
  %699 = insertelement <2 x i32> undef, i32 %697, i32 0		; visa id: 972
  %700 = insertelement <2 x i32> %699, i32 %698, i32 1		; visa id: 973
  %701 = bitcast <2 x i32> %700 to i64		; visa id: 974
  %702 = shl i64 %701, 1		; visa id: 978
  %703 = add i64 %.in400, %702		; visa id: 979
  %704 = ashr i64 %687, 31		; visa id: 980
  %705 = bitcast i64 %704 to <2 x i32>		; visa id: 981
  %706 = extractelement <2 x i32> %705, i32 0		; visa id: 985
  %707 = extractelement <2 x i32> %705, i32 1		; visa id: 985
  %708 = and i32 %706, -2		; visa id: 985
  %709 = insertelement <2 x i32> undef, i32 %708, i32 0		; visa id: 986
  %710 = insertelement <2 x i32> %709, i32 %707, i32 1		; visa id: 987
  %711 = bitcast <2 x i32> %710 to i64		; visa id: 988
  %712 = add i64 %703, %711		; visa id: 992
  %713 = inttoptr i64 %712 to i16 addrspace(4)*		; visa id: 993
  %714 = addrspacecast i16 addrspace(4)* %713 to i16 addrspace(1)*		; visa id: 993
  %715 = load i16, i16 addrspace(1)* %714, align 2		; visa id: 994
  %716 = zext i16 %672 to i32		; visa id: 996
  %717 = shl nuw i32 %716, 16, !spirv.Decorations !639		; visa id: 997
  %718 = bitcast i32 %717 to float
  %719 = zext i16 %715 to i32		; visa id: 998
  %720 = shl nuw i32 %719, 16, !spirv.Decorations !639		; visa id: 999
  %721 = bitcast i32 %720 to float
  %722 = fmul reassoc nsz arcp contract float %718, %721, !spirv.Decorations !618
  %723 = fadd reassoc nsz arcp contract float %722, %.sroa.70.1, !spirv.Decorations !618		; visa id: 1000
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1001

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %686
; BB60 :
  %.sroa.70.2 = phi float [ %723, %686 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  %724 = icmp slt i32 %329, %const_reg_dword
  %725 = icmp slt i32 %527, %const_reg_dword1		; visa id: 1002
  %726 = and i1 %724, %725		; visa id: 1003
  br i1 %726, label %727, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1005

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
; BB:
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

727:                                              ; preds = %._crit_edge.1.1
; BB62 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1007
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1007
  %728 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 1007
  %729 = insertelement <2 x i32> %728, i32 %113, i64 1		; visa id: 1008
  %730 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1009
  store <2 x i32> %729, <2 x i32>* %730, align 4, !noalias !625		; visa id: 1009
  br label %._crit_edge219, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1011

._crit_edge219:                                   ; preds = %._crit_edge219.._crit_edge219_crit_edge, %727
; BB63 :
  %731 = phi i32 [ 0, %727 ], [ %740, %._crit_edge219.._crit_edge219_crit_edge ]
  %732 = zext i32 %731 to i64		; visa id: 1012
  %733 = shl nuw nsw i64 %732, 2		; visa id: 1013
  %734 = add i64 %133, %733		; visa id: 1014
  %735 = inttoptr i64 %734 to i32*		; visa id: 1015
  %736 = load i32, i32* %735, align 4, !noalias !625		; visa id: 1015
  %737 = add i64 %128, %733		; visa id: 1016
  %738 = inttoptr i64 %737 to i32*		; visa id: 1017
  store i32 %736, i32* %738, align 4, !alias.scope !625		; visa id: 1017
  %739 = icmp eq i32 %731, 0		; visa id: 1018
  br i1 %739, label %._crit_edge219.._crit_edge219_crit_edge, label %741, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1019

._crit_edge219.._crit_edge219_crit_edge:          ; preds = %._crit_edge219
; BB64 :
  %740 = add nuw nsw i32 %731, 1, !spirv.Decorations !631		; visa id: 1021
  br label %._crit_edge219, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1022

741:                                              ; preds = %._crit_edge219
; BB65 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1024
  %742 = load i64, i64* %129, align 8		; visa id: 1024
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1025
  %743 = bitcast i64 %742 to <2 x i32>		; visa id: 1025
  %744 = extractelement <2 x i32> %743, i32 0		; visa id: 1027
  %745 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %744, i32 1
  %746 = bitcast <2 x i32> %745 to i64		; visa id: 1027
  %747 = ashr exact i64 %746, 32		; visa id: 1028
  %748 = bitcast i64 %747 to <2 x i32>		; visa id: 1029
  %749 = extractelement <2 x i32> %748, i32 0		; visa id: 1033
  %750 = extractelement <2 x i32> %748, i32 1		; visa id: 1033
  %751 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %749, i32 %750, i32 %41, i32 %42)
  %752 = extractvalue { i32, i32 } %751, 0		; visa id: 1033
  %753 = extractvalue { i32, i32 } %751, 1		; visa id: 1033
  %754 = insertelement <2 x i32> undef, i32 %752, i32 0		; visa id: 1040
  %755 = insertelement <2 x i32> %754, i32 %753, i32 1		; visa id: 1041
  %756 = bitcast <2 x i32> %755 to i64		; visa id: 1042
  %757 = shl i64 %756, 1		; visa id: 1046
  %758 = add i64 %.in401, %757		; visa id: 1047
  %759 = ashr i64 %742, 31		; visa id: 1048
  %760 = bitcast i64 %759 to <2 x i32>		; visa id: 1049
  %761 = extractelement <2 x i32> %760, i32 0		; visa id: 1053
  %762 = extractelement <2 x i32> %760, i32 1		; visa id: 1053
  %763 = and i32 %761, -2		; visa id: 1053
  %764 = insertelement <2 x i32> undef, i32 %763, i32 0		; visa id: 1054
  %765 = insertelement <2 x i32> %764, i32 %762, i32 1		; visa id: 1055
  %766 = bitcast <2 x i32> %765 to i64		; visa id: 1056
  %767 = add i64 %758, %766		; visa id: 1060
  %768 = inttoptr i64 %767 to i16 addrspace(4)*		; visa id: 1061
  %769 = addrspacecast i16 addrspace(4)* %768 to i16 addrspace(1)*		; visa id: 1061
  %770 = load i16, i16 addrspace(1)* %769, align 2		; visa id: 1062
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1064
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1064
  %771 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1064
  %772 = insertelement <2 x i32> %771, i32 %527, i64 1		; visa id: 1065
  %773 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1066
  store <2 x i32> %772, <2 x i32>* %773, align 4, !noalias !635		; visa id: 1066
  br label %._crit_edge220, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1068

._crit_edge220:                                   ; preds = %._crit_edge220.._crit_edge220_crit_edge, %741
; BB66 :
  %774 = phi i32 [ 0, %741 ], [ %783, %._crit_edge220.._crit_edge220_crit_edge ]
  %775 = zext i32 %774 to i64		; visa id: 1069
  %776 = shl nuw nsw i64 %775, 2		; visa id: 1070
  %777 = add i64 %124, %776		; visa id: 1071
  %778 = inttoptr i64 %777 to i32*		; visa id: 1072
  %779 = load i32, i32* %778, align 4, !noalias !635		; visa id: 1072
  %780 = add i64 %119, %776		; visa id: 1073
  %781 = inttoptr i64 %780 to i32*		; visa id: 1074
  store i32 %779, i32* %781, align 4, !alias.scope !635		; visa id: 1074
  %782 = icmp eq i32 %774, 0		; visa id: 1075
  br i1 %782, label %._crit_edge220.._crit_edge220_crit_edge, label %784, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1076

._crit_edge220.._crit_edge220_crit_edge:          ; preds = %._crit_edge220
; BB67 :
  %783 = add nuw nsw i32 %774, 1, !spirv.Decorations !631		; visa id: 1078
  br label %._crit_edge220, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1079

784:                                              ; preds = %._crit_edge220
; BB68 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1081
  %785 = load i64, i64* %120, align 8		; visa id: 1081
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1082
  %786 = bitcast i64 %785 to <2 x i32>		; visa id: 1082
  %787 = extractelement <2 x i32> %786, i32 0		; visa id: 1084
  %788 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %787, i32 1
  %789 = bitcast <2 x i32> %788 to i64		; visa id: 1084
  %790 = ashr exact i64 %789, 32		; visa id: 1085
  %791 = bitcast i64 %790 to <2 x i32>		; visa id: 1086
  %792 = extractelement <2 x i32> %791, i32 0		; visa id: 1090
  %793 = extractelement <2 x i32> %791, i32 1		; visa id: 1090
  %794 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %792, i32 %793, i32 %44, i32 %45)
  %795 = extractvalue { i32, i32 } %794, 0		; visa id: 1090
  %796 = extractvalue { i32, i32 } %794, 1		; visa id: 1090
  %797 = insertelement <2 x i32> undef, i32 %795, i32 0		; visa id: 1097
  %798 = insertelement <2 x i32> %797, i32 %796, i32 1		; visa id: 1098
  %799 = bitcast <2 x i32> %798 to i64		; visa id: 1099
  %800 = shl i64 %799, 1		; visa id: 1103
  %801 = add i64 %.in400, %800		; visa id: 1104
  %802 = ashr i64 %785, 31		; visa id: 1105
  %803 = bitcast i64 %802 to <2 x i32>		; visa id: 1106
  %804 = extractelement <2 x i32> %803, i32 0		; visa id: 1110
  %805 = extractelement <2 x i32> %803, i32 1		; visa id: 1110
  %806 = and i32 %804, -2		; visa id: 1110
  %807 = insertelement <2 x i32> undef, i32 %806, i32 0		; visa id: 1111
  %808 = insertelement <2 x i32> %807, i32 %805, i32 1		; visa id: 1112
  %809 = bitcast <2 x i32> %808 to i64		; visa id: 1113
  %810 = add i64 %801, %809		; visa id: 1117
  %811 = inttoptr i64 %810 to i16 addrspace(4)*		; visa id: 1118
  %812 = addrspacecast i16 addrspace(4)* %811 to i16 addrspace(1)*		; visa id: 1118
  %813 = load i16, i16 addrspace(1)* %812, align 2		; visa id: 1119
  %814 = zext i16 %770 to i32		; visa id: 1121
  %815 = shl nuw i32 %814, 16, !spirv.Decorations !639		; visa id: 1122
  %816 = bitcast i32 %815 to float
  %817 = zext i16 %813 to i32		; visa id: 1123
  %818 = shl nuw i32 %817, 16, !spirv.Decorations !639		; visa id: 1124
  %819 = bitcast i32 %818 to float
  %820 = fmul reassoc nsz arcp contract float %816, %819, !spirv.Decorations !618
  %821 = fadd reassoc nsz arcp contract float %820, %.sroa.134.1, !spirv.Decorations !618		; visa id: 1125
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1126

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %784
; BB69 :
  %.sroa.134.2 = phi float [ %821, %784 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  %822 = icmp slt i32 %428, %const_reg_dword
  %823 = icmp slt i32 %527, %const_reg_dword1		; visa id: 1127
  %824 = and i1 %822, %823		; visa id: 1128
  br i1 %824, label %825, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1130

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
; BB:
  br label %.preheader.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

825:                                              ; preds = %._crit_edge.2.1
; BB71 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1132
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1132
  %826 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 1132
  %827 = insertelement <2 x i32> %826, i32 %113, i64 1		; visa id: 1133
  %828 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1134
  store <2 x i32> %827, <2 x i32>* %828, align 4, !noalias !625		; visa id: 1134
  br label %._crit_edge221, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1136

._crit_edge221:                                   ; preds = %._crit_edge221.._crit_edge221_crit_edge, %825
; BB72 :
  %829 = phi i32 [ 0, %825 ], [ %838, %._crit_edge221.._crit_edge221_crit_edge ]
  %830 = zext i32 %829 to i64		; visa id: 1137
  %831 = shl nuw nsw i64 %830, 2		; visa id: 1138
  %832 = add i64 %133, %831		; visa id: 1139
  %833 = inttoptr i64 %832 to i32*		; visa id: 1140
  %834 = load i32, i32* %833, align 4, !noalias !625		; visa id: 1140
  %835 = add i64 %128, %831		; visa id: 1141
  %836 = inttoptr i64 %835 to i32*		; visa id: 1142
  store i32 %834, i32* %836, align 4, !alias.scope !625		; visa id: 1142
  %837 = icmp eq i32 %829, 0		; visa id: 1143
  br i1 %837, label %._crit_edge221.._crit_edge221_crit_edge, label %839, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1144

._crit_edge221.._crit_edge221_crit_edge:          ; preds = %._crit_edge221
; BB73 :
  %838 = add nuw nsw i32 %829, 1, !spirv.Decorations !631		; visa id: 1146
  br label %._crit_edge221, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1147

839:                                              ; preds = %._crit_edge221
; BB74 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1149
  %840 = load i64, i64* %129, align 8		; visa id: 1149
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1150
  %841 = bitcast i64 %840 to <2 x i32>		; visa id: 1150
  %842 = extractelement <2 x i32> %841, i32 0		; visa id: 1152
  %843 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %842, i32 1
  %844 = bitcast <2 x i32> %843 to i64		; visa id: 1152
  %845 = ashr exact i64 %844, 32		; visa id: 1153
  %846 = bitcast i64 %845 to <2 x i32>		; visa id: 1154
  %847 = extractelement <2 x i32> %846, i32 0		; visa id: 1158
  %848 = extractelement <2 x i32> %846, i32 1		; visa id: 1158
  %849 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %847, i32 %848, i32 %41, i32 %42)
  %850 = extractvalue { i32, i32 } %849, 0		; visa id: 1158
  %851 = extractvalue { i32, i32 } %849, 1		; visa id: 1158
  %852 = insertelement <2 x i32> undef, i32 %850, i32 0		; visa id: 1165
  %853 = insertelement <2 x i32> %852, i32 %851, i32 1		; visa id: 1166
  %854 = bitcast <2 x i32> %853 to i64		; visa id: 1167
  %855 = shl i64 %854, 1		; visa id: 1171
  %856 = add i64 %.in401, %855		; visa id: 1172
  %857 = ashr i64 %840, 31		; visa id: 1173
  %858 = bitcast i64 %857 to <2 x i32>		; visa id: 1174
  %859 = extractelement <2 x i32> %858, i32 0		; visa id: 1178
  %860 = extractelement <2 x i32> %858, i32 1		; visa id: 1178
  %861 = and i32 %859, -2		; visa id: 1178
  %862 = insertelement <2 x i32> undef, i32 %861, i32 0		; visa id: 1179
  %863 = insertelement <2 x i32> %862, i32 %860, i32 1		; visa id: 1180
  %864 = bitcast <2 x i32> %863 to i64		; visa id: 1181
  %865 = add i64 %856, %864		; visa id: 1185
  %866 = inttoptr i64 %865 to i16 addrspace(4)*		; visa id: 1186
  %867 = addrspacecast i16 addrspace(4)* %866 to i16 addrspace(1)*		; visa id: 1186
  %868 = load i16, i16 addrspace(1)* %867, align 2		; visa id: 1187
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1189
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1189
  %869 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1189
  %870 = insertelement <2 x i32> %869, i32 %527, i64 1		; visa id: 1190
  %871 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1191
  store <2 x i32> %870, <2 x i32>* %871, align 4, !noalias !635		; visa id: 1191
  br label %._crit_edge222, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1193

._crit_edge222:                                   ; preds = %._crit_edge222.._crit_edge222_crit_edge, %839
; BB75 :
  %872 = phi i32 [ 0, %839 ], [ %881, %._crit_edge222.._crit_edge222_crit_edge ]
  %873 = zext i32 %872 to i64		; visa id: 1194
  %874 = shl nuw nsw i64 %873, 2		; visa id: 1195
  %875 = add i64 %124, %874		; visa id: 1196
  %876 = inttoptr i64 %875 to i32*		; visa id: 1197
  %877 = load i32, i32* %876, align 4, !noalias !635		; visa id: 1197
  %878 = add i64 %119, %874		; visa id: 1198
  %879 = inttoptr i64 %878 to i32*		; visa id: 1199
  store i32 %877, i32* %879, align 4, !alias.scope !635		; visa id: 1199
  %880 = icmp eq i32 %872, 0		; visa id: 1200
  br i1 %880, label %._crit_edge222.._crit_edge222_crit_edge, label %882, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1201

._crit_edge222.._crit_edge222_crit_edge:          ; preds = %._crit_edge222
; BB76 :
  %881 = add nuw nsw i32 %872, 1, !spirv.Decorations !631		; visa id: 1203
  br label %._crit_edge222, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1204

882:                                              ; preds = %._crit_edge222
; BB77 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1206
  %883 = load i64, i64* %120, align 8		; visa id: 1206
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1207
  %884 = bitcast i64 %883 to <2 x i32>		; visa id: 1207
  %885 = extractelement <2 x i32> %884, i32 0		; visa id: 1209
  %886 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %885, i32 1
  %887 = bitcast <2 x i32> %886 to i64		; visa id: 1209
  %888 = ashr exact i64 %887, 32		; visa id: 1210
  %889 = bitcast i64 %888 to <2 x i32>		; visa id: 1211
  %890 = extractelement <2 x i32> %889, i32 0		; visa id: 1215
  %891 = extractelement <2 x i32> %889, i32 1		; visa id: 1215
  %892 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %890, i32 %891, i32 %44, i32 %45)
  %893 = extractvalue { i32, i32 } %892, 0		; visa id: 1215
  %894 = extractvalue { i32, i32 } %892, 1		; visa id: 1215
  %895 = insertelement <2 x i32> undef, i32 %893, i32 0		; visa id: 1222
  %896 = insertelement <2 x i32> %895, i32 %894, i32 1		; visa id: 1223
  %897 = bitcast <2 x i32> %896 to i64		; visa id: 1224
  %898 = shl i64 %897, 1		; visa id: 1228
  %899 = add i64 %.in400, %898		; visa id: 1229
  %900 = ashr i64 %883, 31		; visa id: 1230
  %901 = bitcast i64 %900 to <2 x i32>		; visa id: 1231
  %902 = extractelement <2 x i32> %901, i32 0		; visa id: 1235
  %903 = extractelement <2 x i32> %901, i32 1		; visa id: 1235
  %904 = and i32 %902, -2		; visa id: 1235
  %905 = insertelement <2 x i32> undef, i32 %904, i32 0		; visa id: 1236
  %906 = insertelement <2 x i32> %905, i32 %903, i32 1		; visa id: 1237
  %907 = bitcast <2 x i32> %906 to i64		; visa id: 1238
  %908 = add i64 %899, %907		; visa id: 1242
  %909 = inttoptr i64 %908 to i16 addrspace(4)*		; visa id: 1243
  %910 = addrspacecast i16 addrspace(4)* %909 to i16 addrspace(1)*		; visa id: 1243
  %911 = load i16, i16 addrspace(1)* %910, align 2		; visa id: 1244
  %912 = zext i16 %868 to i32		; visa id: 1246
  %913 = shl nuw i32 %912, 16, !spirv.Decorations !639		; visa id: 1247
  %914 = bitcast i32 %913 to float
  %915 = zext i16 %911 to i32		; visa id: 1248
  %916 = shl nuw i32 %915, 16, !spirv.Decorations !639		; visa id: 1249
  %917 = bitcast i32 %916 to float
  %918 = fmul reassoc nsz arcp contract float %914, %917, !spirv.Decorations !618
  %919 = fadd reassoc nsz arcp contract float %918, %.sroa.198.1, !spirv.Decorations !618		; visa id: 1250
  br label %.preheader.1, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1251

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %882
; BB78 :
  %.sroa.198.2 = phi float [ %919, %882 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %920 = add i32 %69, 2		; visa id: 1252
  %921 = icmp slt i32 %920, %const_reg_dword1		; visa id: 1253
  %922 = icmp slt i32 %65, %const_reg_dword
  %923 = and i1 %922, %921		; visa id: 1254
  br i1 %923, label %924, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1256

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
; BB:
  br label %._crit_edge.274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

924:                                              ; preds = %.preheader.1
; BB80 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1258
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1258
  %925 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 1258
  %926 = insertelement <2 x i32> %925, i32 %113, i64 1		; visa id: 1259
  %927 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1260
  store <2 x i32> %926, <2 x i32>* %927, align 4, !noalias !625		; visa id: 1260
  br label %._crit_edge223, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1262

._crit_edge223:                                   ; preds = %._crit_edge223.._crit_edge223_crit_edge, %924
; BB81 :
  %928 = phi i32 [ 0, %924 ], [ %937, %._crit_edge223.._crit_edge223_crit_edge ]
  %929 = zext i32 %928 to i64		; visa id: 1263
  %930 = shl nuw nsw i64 %929, 2		; visa id: 1264
  %931 = add i64 %133, %930		; visa id: 1265
  %932 = inttoptr i64 %931 to i32*		; visa id: 1266
  %933 = load i32, i32* %932, align 4, !noalias !625		; visa id: 1266
  %934 = add i64 %128, %930		; visa id: 1267
  %935 = inttoptr i64 %934 to i32*		; visa id: 1268
  store i32 %933, i32* %935, align 4, !alias.scope !625		; visa id: 1268
  %936 = icmp eq i32 %928, 0		; visa id: 1269
  br i1 %936, label %._crit_edge223.._crit_edge223_crit_edge, label %938, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1270

._crit_edge223.._crit_edge223_crit_edge:          ; preds = %._crit_edge223
; BB82 :
  %937 = add nuw nsw i32 %928, 1, !spirv.Decorations !631		; visa id: 1272
  br label %._crit_edge223, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1273

938:                                              ; preds = %._crit_edge223
; BB83 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1275
  %939 = load i64, i64* %129, align 8		; visa id: 1275
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1276
  %940 = bitcast i64 %939 to <2 x i32>		; visa id: 1276
  %941 = extractelement <2 x i32> %940, i32 0		; visa id: 1278
  %942 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %941, i32 1
  %943 = bitcast <2 x i32> %942 to i64		; visa id: 1278
  %944 = ashr exact i64 %943, 32		; visa id: 1279
  %945 = bitcast i64 %944 to <2 x i32>		; visa id: 1280
  %946 = extractelement <2 x i32> %945, i32 0		; visa id: 1284
  %947 = extractelement <2 x i32> %945, i32 1		; visa id: 1284
  %948 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %946, i32 %947, i32 %41, i32 %42)
  %949 = extractvalue { i32, i32 } %948, 0		; visa id: 1284
  %950 = extractvalue { i32, i32 } %948, 1		; visa id: 1284
  %951 = insertelement <2 x i32> undef, i32 %949, i32 0		; visa id: 1291
  %952 = insertelement <2 x i32> %951, i32 %950, i32 1		; visa id: 1292
  %953 = bitcast <2 x i32> %952 to i64		; visa id: 1293
  %954 = shl i64 %953, 1		; visa id: 1297
  %955 = add i64 %.in401, %954		; visa id: 1298
  %956 = ashr i64 %939, 31		; visa id: 1299
  %957 = bitcast i64 %956 to <2 x i32>		; visa id: 1300
  %958 = extractelement <2 x i32> %957, i32 0		; visa id: 1304
  %959 = extractelement <2 x i32> %957, i32 1		; visa id: 1304
  %960 = and i32 %958, -2		; visa id: 1304
  %961 = insertelement <2 x i32> undef, i32 %960, i32 0		; visa id: 1305
  %962 = insertelement <2 x i32> %961, i32 %959, i32 1		; visa id: 1306
  %963 = bitcast <2 x i32> %962 to i64		; visa id: 1307
  %964 = add i64 %955, %963		; visa id: 1311
  %965 = inttoptr i64 %964 to i16 addrspace(4)*		; visa id: 1312
  %966 = addrspacecast i16 addrspace(4)* %965 to i16 addrspace(1)*		; visa id: 1312
  %967 = load i16, i16 addrspace(1)* %966, align 2		; visa id: 1313
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1315
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1315
  %968 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1315
  %969 = insertelement <2 x i32> %968, i32 %920, i64 1		; visa id: 1316
  %970 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1317
  store <2 x i32> %969, <2 x i32>* %970, align 4, !noalias !635		; visa id: 1317
  br label %._crit_edge224, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1319

._crit_edge224:                                   ; preds = %._crit_edge224.._crit_edge224_crit_edge, %938
; BB84 :
  %971 = phi i32 [ 0, %938 ], [ %980, %._crit_edge224.._crit_edge224_crit_edge ]
  %972 = zext i32 %971 to i64		; visa id: 1320
  %973 = shl nuw nsw i64 %972, 2		; visa id: 1321
  %974 = add i64 %124, %973		; visa id: 1322
  %975 = inttoptr i64 %974 to i32*		; visa id: 1323
  %976 = load i32, i32* %975, align 4, !noalias !635		; visa id: 1323
  %977 = add i64 %119, %973		; visa id: 1324
  %978 = inttoptr i64 %977 to i32*		; visa id: 1325
  store i32 %976, i32* %978, align 4, !alias.scope !635		; visa id: 1325
  %979 = icmp eq i32 %971, 0		; visa id: 1326
  br i1 %979, label %._crit_edge224.._crit_edge224_crit_edge, label %981, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1327

._crit_edge224.._crit_edge224_crit_edge:          ; preds = %._crit_edge224
; BB85 :
  %980 = add nuw nsw i32 %971, 1, !spirv.Decorations !631		; visa id: 1329
  br label %._crit_edge224, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1330

981:                                              ; preds = %._crit_edge224
; BB86 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1332
  %982 = load i64, i64* %120, align 8		; visa id: 1332
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1333
  %983 = bitcast i64 %982 to <2 x i32>		; visa id: 1333
  %984 = extractelement <2 x i32> %983, i32 0		; visa id: 1335
  %985 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %984, i32 1
  %986 = bitcast <2 x i32> %985 to i64		; visa id: 1335
  %987 = ashr exact i64 %986, 32		; visa id: 1336
  %988 = bitcast i64 %987 to <2 x i32>		; visa id: 1337
  %989 = extractelement <2 x i32> %988, i32 0		; visa id: 1341
  %990 = extractelement <2 x i32> %988, i32 1		; visa id: 1341
  %991 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %989, i32 %990, i32 %44, i32 %45)
  %992 = extractvalue { i32, i32 } %991, 0		; visa id: 1341
  %993 = extractvalue { i32, i32 } %991, 1		; visa id: 1341
  %994 = insertelement <2 x i32> undef, i32 %992, i32 0		; visa id: 1348
  %995 = insertelement <2 x i32> %994, i32 %993, i32 1		; visa id: 1349
  %996 = bitcast <2 x i32> %995 to i64		; visa id: 1350
  %997 = shl i64 %996, 1		; visa id: 1354
  %998 = add i64 %.in400, %997		; visa id: 1355
  %999 = ashr i64 %982, 31		; visa id: 1356
  %1000 = bitcast i64 %999 to <2 x i32>		; visa id: 1357
  %1001 = extractelement <2 x i32> %1000, i32 0		; visa id: 1361
  %1002 = extractelement <2 x i32> %1000, i32 1		; visa id: 1361
  %1003 = and i32 %1001, -2		; visa id: 1361
  %1004 = insertelement <2 x i32> undef, i32 %1003, i32 0		; visa id: 1362
  %1005 = insertelement <2 x i32> %1004, i32 %1002, i32 1		; visa id: 1363
  %1006 = bitcast <2 x i32> %1005 to i64		; visa id: 1364
  %1007 = add i64 %998, %1006		; visa id: 1368
  %1008 = inttoptr i64 %1007 to i16 addrspace(4)*		; visa id: 1369
  %1009 = addrspacecast i16 addrspace(4)* %1008 to i16 addrspace(1)*		; visa id: 1369
  %1010 = load i16, i16 addrspace(1)* %1009, align 2		; visa id: 1370
  %1011 = zext i16 %967 to i32		; visa id: 1372
  %1012 = shl nuw i32 %1011, 16, !spirv.Decorations !639		; visa id: 1373
  %1013 = bitcast i32 %1012 to float
  %1014 = zext i16 %1010 to i32		; visa id: 1374
  %1015 = shl nuw i32 %1014, 16, !spirv.Decorations !639		; visa id: 1375
  %1016 = bitcast i32 %1015 to float
  %1017 = fmul reassoc nsz arcp contract float %1013, %1016, !spirv.Decorations !618
  %1018 = fadd reassoc nsz arcp contract float %1017, %.sroa.10.1, !spirv.Decorations !618		; visa id: 1376
  br label %._crit_edge.274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1377

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %981
; BB87 :
  %.sroa.10.2 = phi float [ %1018, %981 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  %1019 = icmp slt i32 %230, %const_reg_dword
  %1020 = icmp slt i32 %920, %const_reg_dword1		; visa id: 1378
  %1021 = and i1 %1019, %1020		; visa id: 1379
  br i1 %1021, label %1022, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1381

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
; BB:
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1022:                                             ; preds = %._crit_edge.274
; BB89 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1383
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1383
  %1023 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 1383
  %1024 = insertelement <2 x i32> %1023, i32 %113, i64 1		; visa id: 1384
  %1025 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1385
  store <2 x i32> %1024, <2 x i32>* %1025, align 4, !noalias !625		; visa id: 1385
  br label %._crit_edge225, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1387

._crit_edge225:                                   ; preds = %._crit_edge225.._crit_edge225_crit_edge, %1022
; BB90 :
  %1026 = phi i32 [ 0, %1022 ], [ %1035, %._crit_edge225.._crit_edge225_crit_edge ]
  %1027 = zext i32 %1026 to i64		; visa id: 1388
  %1028 = shl nuw nsw i64 %1027, 2		; visa id: 1389
  %1029 = add i64 %133, %1028		; visa id: 1390
  %1030 = inttoptr i64 %1029 to i32*		; visa id: 1391
  %1031 = load i32, i32* %1030, align 4, !noalias !625		; visa id: 1391
  %1032 = add i64 %128, %1028		; visa id: 1392
  %1033 = inttoptr i64 %1032 to i32*		; visa id: 1393
  store i32 %1031, i32* %1033, align 4, !alias.scope !625		; visa id: 1393
  %1034 = icmp eq i32 %1026, 0		; visa id: 1394
  br i1 %1034, label %._crit_edge225.._crit_edge225_crit_edge, label %1036, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1395

._crit_edge225.._crit_edge225_crit_edge:          ; preds = %._crit_edge225
; BB91 :
  %1035 = add nuw nsw i32 %1026, 1, !spirv.Decorations !631		; visa id: 1397
  br label %._crit_edge225, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1398

1036:                                             ; preds = %._crit_edge225
; BB92 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1400
  %1037 = load i64, i64* %129, align 8		; visa id: 1400
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1401
  %1038 = bitcast i64 %1037 to <2 x i32>		; visa id: 1401
  %1039 = extractelement <2 x i32> %1038, i32 0		; visa id: 1403
  %1040 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1039, i32 1
  %1041 = bitcast <2 x i32> %1040 to i64		; visa id: 1403
  %1042 = ashr exact i64 %1041, 32		; visa id: 1404
  %1043 = bitcast i64 %1042 to <2 x i32>		; visa id: 1405
  %1044 = extractelement <2 x i32> %1043, i32 0		; visa id: 1409
  %1045 = extractelement <2 x i32> %1043, i32 1		; visa id: 1409
  %1046 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1044, i32 %1045, i32 %41, i32 %42)
  %1047 = extractvalue { i32, i32 } %1046, 0		; visa id: 1409
  %1048 = extractvalue { i32, i32 } %1046, 1		; visa id: 1409
  %1049 = insertelement <2 x i32> undef, i32 %1047, i32 0		; visa id: 1416
  %1050 = insertelement <2 x i32> %1049, i32 %1048, i32 1		; visa id: 1417
  %1051 = bitcast <2 x i32> %1050 to i64		; visa id: 1418
  %1052 = shl i64 %1051, 1		; visa id: 1422
  %1053 = add i64 %.in401, %1052		; visa id: 1423
  %1054 = ashr i64 %1037, 31		; visa id: 1424
  %1055 = bitcast i64 %1054 to <2 x i32>		; visa id: 1425
  %1056 = extractelement <2 x i32> %1055, i32 0		; visa id: 1429
  %1057 = extractelement <2 x i32> %1055, i32 1		; visa id: 1429
  %1058 = and i32 %1056, -2		; visa id: 1429
  %1059 = insertelement <2 x i32> undef, i32 %1058, i32 0		; visa id: 1430
  %1060 = insertelement <2 x i32> %1059, i32 %1057, i32 1		; visa id: 1431
  %1061 = bitcast <2 x i32> %1060 to i64		; visa id: 1432
  %1062 = add i64 %1053, %1061		; visa id: 1436
  %1063 = inttoptr i64 %1062 to i16 addrspace(4)*		; visa id: 1437
  %1064 = addrspacecast i16 addrspace(4)* %1063 to i16 addrspace(1)*		; visa id: 1437
  %1065 = load i16, i16 addrspace(1)* %1064, align 2		; visa id: 1438
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1440
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1440
  %1066 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1440
  %1067 = insertelement <2 x i32> %1066, i32 %920, i64 1		; visa id: 1441
  %1068 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1442
  store <2 x i32> %1067, <2 x i32>* %1068, align 4, !noalias !635		; visa id: 1442
  br label %._crit_edge226, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1444

._crit_edge226:                                   ; preds = %._crit_edge226.._crit_edge226_crit_edge, %1036
; BB93 :
  %1069 = phi i32 [ 0, %1036 ], [ %1078, %._crit_edge226.._crit_edge226_crit_edge ]
  %1070 = zext i32 %1069 to i64		; visa id: 1445
  %1071 = shl nuw nsw i64 %1070, 2		; visa id: 1446
  %1072 = add i64 %124, %1071		; visa id: 1447
  %1073 = inttoptr i64 %1072 to i32*		; visa id: 1448
  %1074 = load i32, i32* %1073, align 4, !noalias !635		; visa id: 1448
  %1075 = add i64 %119, %1071		; visa id: 1449
  %1076 = inttoptr i64 %1075 to i32*		; visa id: 1450
  store i32 %1074, i32* %1076, align 4, !alias.scope !635		; visa id: 1450
  %1077 = icmp eq i32 %1069, 0		; visa id: 1451
  br i1 %1077, label %._crit_edge226.._crit_edge226_crit_edge, label %1079, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1452

._crit_edge226.._crit_edge226_crit_edge:          ; preds = %._crit_edge226
; BB94 :
  %1078 = add nuw nsw i32 %1069, 1, !spirv.Decorations !631		; visa id: 1454
  br label %._crit_edge226, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1455

1079:                                             ; preds = %._crit_edge226
; BB95 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1457
  %1080 = load i64, i64* %120, align 8		; visa id: 1457
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1458
  %1081 = bitcast i64 %1080 to <2 x i32>		; visa id: 1458
  %1082 = extractelement <2 x i32> %1081, i32 0		; visa id: 1460
  %1083 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1082, i32 1
  %1084 = bitcast <2 x i32> %1083 to i64		; visa id: 1460
  %1085 = ashr exact i64 %1084, 32		; visa id: 1461
  %1086 = bitcast i64 %1085 to <2 x i32>		; visa id: 1462
  %1087 = extractelement <2 x i32> %1086, i32 0		; visa id: 1466
  %1088 = extractelement <2 x i32> %1086, i32 1		; visa id: 1466
  %1089 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1087, i32 %1088, i32 %44, i32 %45)
  %1090 = extractvalue { i32, i32 } %1089, 0		; visa id: 1466
  %1091 = extractvalue { i32, i32 } %1089, 1		; visa id: 1466
  %1092 = insertelement <2 x i32> undef, i32 %1090, i32 0		; visa id: 1473
  %1093 = insertelement <2 x i32> %1092, i32 %1091, i32 1		; visa id: 1474
  %1094 = bitcast <2 x i32> %1093 to i64		; visa id: 1475
  %1095 = shl i64 %1094, 1		; visa id: 1479
  %1096 = add i64 %.in400, %1095		; visa id: 1480
  %1097 = ashr i64 %1080, 31		; visa id: 1481
  %1098 = bitcast i64 %1097 to <2 x i32>		; visa id: 1482
  %1099 = extractelement <2 x i32> %1098, i32 0		; visa id: 1486
  %1100 = extractelement <2 x i32> %1098, i32 1		; visa id: 1486
  %1101 = and i32 %1099, -2		; visa id: 1486
  %1102 = insertelement <2 x i32> undef, i32 %1101, i32 0		; visa id: 1487
  %1103 = insertelement <2 x i32> %1102, i32 %1100, i32 1		; visa id: 1488
  %1104 = bitcast <2 x i32> %1103 to i64		; visa id: 1489
  %1105 = add i64 %1096, %1104		; visa id: 1493
  %1106 = inttoptr i64 %1105 to i16 addrspace(4)*		; visa id: 1494
  %1107 = addrspacecast i16 addrspace(4)* %1106 to i16 addrspace(1)*		; visa id: 1494
  %1108 = load i16, i16 addrspace(1)* %1107, align 2		; visa id: 1495
  %1109 = zext i16 %1065 to i32		; visa id: 1497
  %1110 = shl nuw i32 %1109, 16, !spirv.Decorations !639		; visa id: 1498
  %1111 = bitcast i32 %1110 to float
  %1112 = zext i16 %1108 to i32		; visa id: 1499
  %1113 = shl nuw i32 %1112, 16, !spirv.Decorations !639		; visa id: 1500
  %1114 = bitcast i32 %1113 to float
  %1115 = fmul reassoc nsz arcp contract float %1111, %1114, !spirv.Decorations !618
  %1116 = fadd reassoc nsz arcp contract float %1115, %.sroa.74.1, !spirv.Decorations !618		; visa id: 1501
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1502

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %1079
; BB96 :
  %.sroa.74.2 = phi float [ %1116, %1079 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  %1117 = icmp slt i32 %329, %const_reg_dword
  %1118 = icmp slt i32 %920, %const_reg_dword1		; visa id: 1503
  %1119 = and i1 %1117, %1118		; visa id: 1504
  br i1 %1119, label %1120, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1506

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
; BB:
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1120:                                             ; preds = %._crit_edge.1.2
; BB98 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1508
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1508
  %1121 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 1508
  %1122 = insertelement <2 x i32> %1121, i32 %113, i64 1		; visa id: 1509
  %1123 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1510
  store <2 x i32> %1122, <2 x i32>* %1123, align 4, !noalias !625		; visa id: 1510
  br label %._crit_edge227, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1512

._crit_edge227:                                   ; preds = %._crit_edge227.._crit_edge227_crit_edge, %1120
; BB99 :
  %1124 = phi i32 [ 0, %1120 ], [ %1133, %._crit_edge227.._crit_edge227_crit_edge ]
  %1125 = zext i32 %1124 to i64		; visa id: 1513
  %1126 = shl nuw nsw i64 %1125, 2		; visa id: 1514
  %1127 = add i64 %133, %1126		; visa id: 1515
  %1128 = inttoptr i64 %1127 to i32*		; visa id: 1516
  %1129 = load i32, i32* %1128, align 4, !noalias !625		; visa id: 1516
  %1130 = add i64 %128, %1126		; visa id: 1517
  %1131 = inttoptr i64 %1130 to i32*		; visa id: 1518
  store i32 %1129, i32* %1131, align 4, !alias.scope !625		; visa id: 1518
  %1132 = icmp eq i32 %1124, 0		; visa id: 1519
  br i1 %1132, label %._crit_edge227.._crit_edge227_crit_edge, label %1134, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1520

._crit_edge227.._crit_edge227_crit_edge:          ; preds = %._crit_edge227
; BB100 :
  %1133 = add nuw nsw i32 %1124, 1, !spirv.Decorations !631		; visa id: 1522
  br label %._crit_edge227, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1523

1134:                                             ; preds = %._crit_edge227
; BB101 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1525
  %1135 = load i64, i64* %129, align 8		; visa id: 1525
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1526
  %1136 = bitcast i64 %1135 to <2 x i32>		; visa id: 1526
  %1137 = extractelement <2 x i32> %1136, i32 0		; visa id: 1528
  %1138 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1137, i32 1
  %1139 = bitcast <2 x i32> %1138 to i64		; visa id: 1528
  %1140 = ashr exact i64 %1139, 32		; visa id: 1529
  %1141 = bitcast i64 %1140 to <2 x i32>		; visa id: 1530
  %1142 = extractelement <2 x i32> %1141, i32 0		; visa id: 1534
  %1143 = extractelement <2 x i32> %1141, i32 1		; visa id: 1534
  %1144 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1142, i32 %1143, i32 %41, i32 %42)
  %1145 = extractvalue { i32, i32 } %1144, 0		; visa id: 1534
  %1146 = extractvalue { i32, i32 } %1144, 1		; visa id: 1534
  %1147 = insertelement <2 x i32> undef, i32 %1145, i32 0		; visa id: 1541
  %1148 = insertelement <2 x i32> %1147, i32 %1146, i32 1		; visa id: 1542
  %1149 = bitcast <2 x i32> %1148 to i64		; visa id: 1543
  %1150 = shl i64 %1149, 1		; visa id: 1547
  %1151 = add i64 %.in401, %1150		; visa id: 1548
  %1152 = ashr i64 %1135, 31		; visa id: 1549
  %1153 = bitcast i64 %1152 to <2 x i32>		; visa id: 1550
  %1154 = extractelement <2 x i32> %1153, i32 0		; visa id: 1554
  %1155 = extractelement <2 x i32> %1153, i32 1		; visa id: 1554
  %1156 = and i32 %1154, -2		; visa id: 1554
  %1157 = insertelement <2 x i32> undef, i32 %1156, i32 0		; visa id: 1555
  %1158 = insertelement <2 x i32> %1157, i32 %1155, i32 1		; visa id: 1556
  %1159 = bitcast <2 x i32> %1158 to i64		; visa id: 1557
  %1160 = add i64 %1151, %1159		; visa id: 1561
  %1161 = inttoptr i64 %1160 to i16 addrspace(4)*		; visa id: 1562
  %1162 = addrspacecast i16 addrspace(4)* %1161 to i16 addrspace(1)*		; visa id: 1562
  %1163 = load i16, i16 addrspace(1)* %1162, align 2		; visa id: 1563
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1565
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1565
  %1164 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1565
  %1165 = insertelement <2 x i32> %1164, i32 %920, i64 1		; visa id: 1566
  %1166 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1567
  store <2 x i32> %1165, <2 x i32>* %1166, align 4, !noalias !635		; visa id: 1567
  br label %._crit_edge228, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1569

._crit_edge228:                                   ; preds = %._crit_edge228.._crit_edge228_crit_edge, %1134
; BB102 :
  %1167 = phi i32 [ 0, %1134 ], [ %1176, %._crit_edge228.._crit_edge228_crit_edge ]
  %1168 = zext i32 %1167 to i64		; visa id: 1570
  %1169 = shl nuw nsw i64 %1168, 2		; visa id: 1571
  %1170 = add i64 %124, %1169		; visa id: 1572
  %1171 = inttoptr i64 %1170 to i32*		; visa id: 1573
  %1172 = load i32, i32* %1171, align 4, !noalias !635		; visa id: 1573
  %1173 = add i64 %119, %1169		; visa id: 1574
  %1174 = inttoptr i64 %1173 to i32*		; visa id: 1575
  store i32 %1172, i32* %1174, align 4, !alias.scope !635		; visa id: 1575
  %1175 = icmp eq i32 %1167, 0		; visa id: 1576
  br i1 %1175, label %._crit_edge228.._crit_edge228_crit_edge, label %1177, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1577

._crit_edge228.._crit_edge228_crit_edge:          ; preds = %._crit_edge228
; BB103 :
  %1176 = add nuw nsw i32 %1167, 1, !spirv.Decorations !631		; visa id: 1579
  br label %._crit_edge228, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1580

1177:                                             ; preds = %._crit_edge228
; BB104 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1582
  %1178 = load i64, i64* %120, align 8		; visa id: 1582
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1583
  %1179 = bitcast i64 %1178 to <2 x i32>		; visa id: 1583
  %1180 = extractelement <2 x i32> %1179, i32 0		; visa id: 1585
  %1181 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1180, i32 1
  %1182 = bitcast <2 x i32> %1181 to i64		; visa id: 1585
  %1183 = ashr exact i64 %1182, 32		; visa id: 1586
  %1184 = bitcast i64 %1183 to <2 x i32>		; visa id: 1587
  %1185 = extractelement <2 x i32> %1184, i32 0		; visa id: 1591
  %1186 = extractelement <2 x i32> %1184, i32 1		; visa id: 1591
  %1187 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1185, i32 %1186, i32 %44, i32 %45)
  %1188 = extractvalue { i32, i32 } %1187, 0		; visa id: 1591
  %1189 = extractvalue { i32, i32 } %1187, 1		; visa id: 1591
  %1190 = insertelement <2 x i32> undef, i32 %1188, i32 0		; visa id: 1598
  %1191 = insertelement <2 x i32> %1190, i32 %1189, i32 1		; visa id: 1599
  %1192 = bitcast <2 x i32> %1191 to i64		; visa id: 1600
  %1193 = shl i64 %1192, 1		; visa id: 1604
  %1194 = add i64 %.in400, %1193		; visa id: 1605
  %1195 = ashr i64 %1178, 31		; visa id: 1606
  %1196 = bitcast i64 %1195 to <2 x i32>		; visa id: 1607
  %1197 = extractelement <2 x i32> %1196, i32 0		; visa id: 1611
  %1198 = extractelement <2 x i32> %1196, i32 1		; visa id: 1611
  %1199 = and i32 %1197, -2		; visa id: 1611
  %1200 = insertelement <2 x i32> undef, i32 %1199, i32 0		; visa id: 1612
  %1201 = insertelement <2 x i32> %1200, i32 %1198, i32 1		; visa id: 1613
  %1202 = bitcast <2 x i32> %1201 to i64		; visa id: 1614
  %1203 = add i64 %1194, %1202		; visa id: 1618
  %1204 = inttoptr i64 %1203 to i16 addrspace(4)*		; visa id: 1619
  %1205 = addrspacecast i16 addrspace(4)* %1204 to i16 addrspace(1)*		; visa id: 1619
  %1206 = load i16, i16 addrspace(1)* %1205, align 2		; visa id: 1620
  %1207 = zext i16 %1163 to i32		; visa id: 1622
  %1208 = shl nuw i32 %1207, 16, !spirv.Decorations !639		; visa id: 1623
  %1209 = bitcast i32 %1208 to float
  %1210 = zext i16 %1206 to i32		; visa id: 1624
  %1211 = shl nuw i32 %1210, 16, !spirv.Decorations !639		; visa id: 1625
  %1212 = bitcast i32 %1211 to float
  %1213 = fmul reassoc nsz arcp contract float %1209, %1212, !spirv.Decorations !618
  %1214 = fadd reassoc nsz arcp contract float %1213, %.sroa.138.1, !spirv.Decorations !618		; visa id: 1626
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1627

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %1177
; BB105 :
  %.sroa.138.2 = phi float [ %1214, %1177 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  %1215 = icmp slt i32 %428, %const_reg_dword
  %1216 = icmp slt i32 %920, %const_reg_dword1		; visa id: 1628
  %1217 = and i1 %1215, %1216		; visa id: 1629
  br i1 %1217, label %1218, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1631

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
; BB:
  br label %.preheader.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1218:                                             ; preds = %._crit_edge.2.2
; BB107 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1633
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1633
  %1219 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 1633
  %1220 = insertelement <2 x i32> %1219, i32 %113, i64 1		; visa id: 1634
  %1221 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1635
  store <2 x i32> %1220, <2 x i32>* %1221, align 4, !noalias !625		; visa id: 1635
  br label %._crit_edge229, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1637

._crit_edge229:                                   ; preds = %._crit_edge229.._crit_edge229_crit_edge, %1218
; BB108 :
  %1222 = phi i32 [ 0, %1218 ], [ %1231, %._crit_edge229.._crit_edge229_crit_edge ]
  %1223 = zext i32 %1222 to i64		; visa id: 1638
  %1224 = shl nuw nsw i64 %1223, 2		; visa id: 1639
  %1225 = add i64 %133, %1224		; visa id: 1640
  %1226 = inttoptr i64 %1225 to i32*		; visa id: 1641
  %1227 = load i32, i32* %1226, align 4, !noalias !625		; visa id: 1641
  %1228 = add i64 %128, %1224		; visa id: 1642
  %1229 = inttoptr i64 %1228 to i32*		; visa id: 1643
  store i32 %1227, i32* %1229, align 4, !alias.scope !625		; visa id: 1643
  %1230 = icmp eq i32 %1222, 0		; visa id: 1644
  br i1 %1230, label %._crit_edge229.._crit_edge229_crit_edge, label %1232, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1645

._crit_edge229.._crit_edge229_crit_edge:          ; preds = %._crit_edge229
; BB109 :
  %1231 = add nuw nsw i32 %1222, 1, !spirv.Decorations !631		; visa id: 1647
  br label %._crit_edge229, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1648

1232:                                             ; preds = %._crit_edge229
; BB110 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1650
  %1233 = load i64, i64* %129, align 8		; visa id: 1650
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1651
  %1234 = bitcast i64 %1233 to <2 x i32>		; visa id: 1651
  %1235 = extractelement <2 x i32> %1234, i32 0		; visa id: 1653
  %1236 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1235, i32 1
  %1237 = bitcast <2 x i32> %1236 to i64		; visa id: 1653
  %1238 = ashr exact i64 %1237, 32		; visa id: 1654
  %1239 = bitcast i64 %1238 to <2 x i32>		; visa id: 1655
  %1240 = extractelement <2 x i32> %1239, i32 0		; visa id: 1659
  %1241 = extractelement <2 x i32> %1239, i32 1		; visa id: 1659
  %1242 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1240, i32 %1241, i32 %41, i32 %42)
  %1243 = extractvalue { i32, i32 } %1242, 0		; visa id: 1659
  %1244 = extractvalue { i32, i32 } %1242, 1		; visa id: 1659
  %1245 = insertelement <2 x i32> undef, i32 %1243, i32 0		; visa id: 1666
  %1246 = insertelement <2 x i32> %1245, i32 %1244, i32 1		; visa id: 1667
  %1247 = bitcast <2 x i32> %1246 to i64		; visa id: 1668
  %1248 = shl i64 %1247, 1		; visa id: 1672
  %1249 = add i64 %.in401, %1248		; visa id: 1673
  %1250 = ashr i64 %1233, 31		; visa id: 1674
  %1251 = bitcast i64 %1250 to <2 x i32>		; visa id: 1675
  %1252 = extractelement <2 x i32> %1251, i32 0		; visa id: 1679
  %1253 = extractelement <2 x i32> %1251, i32 1		; visa id: 1679
  %1254 = and i32 %1252, -2		; visa id: 1679
  %1255 = insertelement <2 x i32> undef, i32 %1254, i32 0		; visa id: 1680
  %1256 = insertelement <2 x i32> %1255, i32 %1253, i32 1		; visa id: 1681
  %1257 = bitcast <2 x i32> %1256 to i64		; visa id: 1682
  %1258 = add i64 %1249, %1257		; visa id: 1686
  %1259 = inttoptr i64 %1258 to i16 addrspace(4)*		; visa id: 1687
  %1260 = addrspacecast i16 addrspace(4)* %1259 to i16 addrspace(1)*		; visa id: 1687
  %1261 = load i16, i16 addrspace(1)* %1260, align 2		; visa id: 1688
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1690
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1690
  %1262 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1690
  %1263 = insertelement <2 x i32> %1262, i32 %920, i64 1		; visa id: 1691
  %1264 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1692
  store <2 x i32> %1263, <2 x i32>* %1264, align 4, !noalias !635		; visa id: 1692
  br label %._crit_edge230, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1694

._crit_edge230:                                   ; preds = %._crit_edge230.._crit_edge230_crit_edge, %1232
; BB111 :
  %1265 = phi i32 [ 0, %1232 ], [ %1274, %._crit_edge230.._crit_edge230_crit_edge ]
  %1266 = zext i32 %1265 to i64		; visa id: 1695
  %1267 = shl nuw nsw i64 %1266, 2		; visa id: 1696
  %1268 = add i64 %124, %1267		; visa id: 1697
  %1269 = inttoptr i64 %1268 to i32*		; visa id: 1698
  %1270 = load i32, i32* %1269, align 4, !noalias !635		; visa id: 1698
  %1271 = add i64 %119, %1267		; visa id: 1699
  %1272 = inttoptr i64 %1271 to i32*		; visa id: 1700
  store i32 %1270, i32* %1272, align 4, !alias.scope !635		; visa id: 1700
  %1273 = icmp eq i32 %1265, 0		; visa id: 1701
  br i1 %1273, label %._crit_edge230.._crit_edge230_crit_edge, label %1275, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1702

._crit_edge230.._crit_edge230_crit_edge:          ; preds = %._crit_edge230
; BB112 :
  %1274 = add nuw nsw i32 %1265, 1, !spirv.Decorations !631		; visa id: 1704
  br label %._crit_edge230, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1705

1275:                                             ; preds = %._crit_edge230
; BB113 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1707
  %1276 = load i64, i64* %120, align 8		; visa id: 1707
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1708
  %1277 = bitcast i64 %1276 to <2 x i32>		; visa id: 1708
  %1278 = extractelement <2 x i32> %1277, i32 0		; visa id: 1710
  %1279 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1278, i32 1
  %1280 = bitcast <2 x i32> %1279 to i64		; visa id: 1710
  %1281 = ashr exact i64 %1280, 32		; visa id: 1711
  %1282 = bitcast i64 %1281 to <2 x i32>		; visa id: 1712
  %1283 = extractelement <2 x i32> %1282, i32 0		; visa id: 1716
  %1284 = extractelement <2 x i32> %1282, i32 1		; visa id: 1716
  %1285 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1283, i32 %1284, i32 %44, i32 %45)
  %1286 = extractvalue { i32, i32 } %1285, 0		; visa id: 1716
  %1287 = extractvalue { i32, i32 } %1285, 1		; visa id: 1716
  %1288 = insertelement <2 x i32> undef, i32 %1286, i32 0		; visa id: 1723
  %1289 = insertelement <2 x i32> %1288, i32 %1287, i32 1		; visa id: 1724
  %1290 = bitcast <2 x i32> %1289 to i64		; visa id: 1725
  %1291 = shl i64 %1290, 1		; visa id: 1729
  %1292 = add i64 %.in400, %1291		; visa id: 1730
  %1293 = ashr i64 %1276, 31		; visa id: 1731
  %1294 = bitcast i64 %1293 to <2 x i32>		; visa id: 1732
  %1295 = extractelement <2 x i32> %1294, i32 0		; visa id: 1736
  %1296 = extractelement <2 x i32> %1294, i32 1		; visa id: 1736
  %1297 = and i32 %1295, -2		; visa id: 1736
  %1298 = insertelement <2 x i32> undef, i32 %1297, i32 0		; visa id: 1737
  %1299 = insertelement <2 x i32> %1298, i32 %1296, i32 1		; visa id: 1738
  %1300 = bitcast <2 x i32> %1299 to i64		; visa id: 1739
  %1301 = add i64 %1292, %1300		; visa id: 1743
  %1302 = inttoptr i64 %1301 to i16 addrspace(4)*		; visa id: 1744
  %1303 = addrspacecast i16 addrspace(4)* %1302 to i16 addrspace(1)*		; visa id: 1744
  %1304 = load i16, i16 addrspace(1)* %1303, align 2		; visa id: 1745
  %1305 = zext i16 %1261 to i32		; visa id: 1747
  %1306 = shl nuw i32 %1305, 16, !spirv.Decorations !639		; visa id: 1748
  %1307 = bitcast i32 %1306 to float
  %1308 = zext i16 %1304 to i32		; visa id: 1749
  %1309 = shl nuw i32 %1308, 16, !spirv.Decorations !639		; visa id: 1750
  %1310 = bitcast i32 %1309 to float
  %1311 = fmul reassoc nsz arcp contract float %1307, %1310, !spirv.Decorations !618
  %1312 = fadd reassoc nsz arcp contract float %1311, %.sroa.202.1, !spirv.Decorations !618		; visa id: 1751
  br label %.preheader.2, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1752

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %1275
; BB114 :
  %.sroa.202.2 = phi float [ %1312, %1275 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %1313 = add i32 %69, 3		; visa id: 1753
  %1314 = icmp slt i32 %1313, %const_reg_dword1		; visa id: 1754
  %1315 = icmp slt i32 %65, %const_reg_dword
  %1316 = and i1 %1315, %1314		; visa id: 1755
  br i1 %1316, label %1317, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1757

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
; BB:
  br label %._crit_edge.375, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1317:                                             ; preds = %.preheader.2
; BB116 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1759
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1759
  %1318 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 1759
  %1319 = insertelement <2 x i32> %1318, i32 %113, i64 1		; visa id: 1760
  %1320 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1761
  store <2 x i32> %1319, <2 x i32>* %1320, align 4, !noalias !625		; visa id: 1761
  br label %._crit_edge231, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1763

._crit_edge231:                                   ; preds = %._crit_edge231.._crit_edge231_crit_edge, %1317
; BB117 :
  %1321 = phi i32 [ 0, %1317 ], [ %1330, %._crit_edge231.._crit_edge231_crit_edge ]
  %1322 = zext i32 %1321 to i64		; visa id: 1764
  %1323 = shl nuw nsw i64 %1322, 2		; visa id: 1765
  %1324 = add i64 %133, %1323		; visa id: 1766
  %1325 = inttoptr i64 %1324 to i32*		; visa id: 1767
  %1326 = load i32, i32* %1325, align 4, !noalias !625		; visa id: 1767
  %1327 = add i64 %128, %1323		; visa id: 1768
  %1328 = inttoptr i64 %1327 to i32*		; visa id: 1769
  store i32 %1326, i32* %1328, align 4, !alias.scope !625		; visa id: 1769
  %1329 = icmp eq i32 %1321, 0		; visa id: 1770
  br i1 %1329, label %._crit_edge231.._crit_edge231_crit_edge, label %1331, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1771

._crit_edge231.._crit_edge231_crit_edge:          ; preds = %._crit_edge231
; BB118 :
  %1330 = add nuw nsw i32 %1321, 1, !spirv.Decorations !631		; visa id: 1773
  br label %._crit_edge231, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1774

1331:                                             ; preds = %._crit_edge231
; BB119 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1776
  %1332 = load i64, i64* %129, align 8		; visa id: 1776
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1777
  %1333 = bitcast i64 %1332 to <2 x i32>		; visa id: 1777
  %1334 = extractelement <2 x i32> %1333, i32 0		; visa id: 1779
  %1335 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1334, i32 1
  %1336 = bitcast <2 x i32> %1335 to i64		; visa id: 1779
  %1337 = ashr exact i64 %1336, 32		; visa id: 1780
  %1338 = bitcast i64 %1337 to <2 x i32>		; visa id: 1781
  %1339 = extractelement <2 x i32> %1338, i32 0		; visa id: 1785
  %1340 = extractelement <2 x i32> %1338, i32 1		; visa id: 1785
  %1341 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1339, i32 %1340, i32 %41, i32 %42)
  %1342 = extractvalue { i32, i32 } %1341, 0		; visa id: 1785
  %1343 = extractvalue { i32, i32 } %1341, 1		; visa id: 1785
  %1344 = insertelement <2 x i32> undef, i32 %1342, i32 0		; visa id: 1792
  %1345 = insertelement <2 x i32> %1344, i32 %1343, i32 1		; visa id: 1793
  %1346 = bitcast <2 x i32> %1345 to i64		; visa id: 1794
  %1347 = shl i64 %1346, 1		; visa id: 1798
  %1348 = add i64 %.in401, %1347		; visa id: 1799
  %1349 = ashr i64 %1332, 31		; visa id: 1800
  %1350 = bitcast i64 %1349 to <2 x i32>		; visa id: 1801
  %1351 = extractelement <2 x i32> %1350, i32 0		; visa id: 1805
  %1352 = extractelement <2 x i32> %1350, i32 1		; visa id: 1805
  %1353 = and i32 %1351, -2		; visa id: 1805
  %1354 = insertelement <2 x i32> undef, i32 %1353, i32 0		; visa id: 1806
  %1355 = insertelement <2 x i32> %1354, i32 %1352, i32 1		; visa id: 1807
  %1356 = bitcast <2 x i32> %1355 to i64		; visa id: 1808
  %1357 = add i64 %1348, %1356		; visa id: 1812
  %1358 = inttoptr i64 %1357 to i16 addrspace(4)*		; visa id: 1813
  %1359 = addrspacecast i16 addrspace(4)* %1358 to i16 addrspace(1)*		; visa id: 1813
  %1360 = load i16, i16 addrspace(1)* %1359, align 2		; visa id: 1814
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1816
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1816
  %1361 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1816
  %1362 = insertelement <2 x i32> %1361, i32 %1313, i64 1		; visa id: 1817
  %1363 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1818
  store <2 x i32> %1362, <2 x i32>* %1363, align 4, !noalias !635		; visa id: 1818
  br label %._crit_edge232, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1820

._crit_edge232:                                   ; preds = %._crit_edge232.._crit_edge232_crit_edge, %1331
; BB120 :
  %1364 = phi i32 [ 0, %1331 ], [ %1373, %._crit_edge232.._crit_edge232_crit_edge ]
  %1365 = zext i32 %1364 to i64		; visa id: 1821
  %1366 = shl nuw nsw i64 %1365, 2		; visa id: 1822
  %1367 = add i64 %124, %1366		; visa id: 1823
  %1368 = inttoptr i64 %1367 to i32*		; visa id: 1824
  %1369 = load i32, i32* %1368, align 4, !noalias !635		; visa id: 1824
  %1370 = add i64 %119, %1366		; visa id: 1825
  %1371 = inttoptr i64 %1370 to i32*		; visa id: 1826
  store i32 %1369, i32* %1371, align 4, !alias.scope !635		; visa id: 1826
  %1372 = icmp eq i32 %1364, 0		; visa id: 1827
  br i1 %1372, label %._crit_edge232.._crit_edge232_crit_edge, label %1374, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1828

._crit_edge232.._crit_edge232_crit_edge:          ; preds = %._crit_edge232
; BB121 :
  %1373 = add nuw nsw i32 %1364, 1, !spirv.Decorations !631		; visa id: 1830
  br label %._crit_edge232, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1831

1374:                                             ; preds = %._crit_edge232
; BB122 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1833
  %1375 = load i64, i64* %120, align 8		; visa id: 1833
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1834
  %1376 = bitcast i64 %1375 to <2 x i32>		; visa id: 1834
  %1377 = extractelement <2 x i32> %1376, i32 0		; visa id: 1836
  %1378 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1377, i32 1
  %1379 = bitcast <2 x i32> %1378 to i64		; visa id: 1836
  %1380 = ashr exact i64 %1379, 32		; visa id: 1837
  %1381 = bitcast i64 %1380 to <2 x i32>		; visa id: 1838
  %1382 = extractelement <2 x i32> %1381, i32 0		; visa id: 1842
  %1383 = extractelement <2 x i32> %1381, i32 1		; visa id: 1842
  %1384 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1382, i32 %1383, i32 %44, i32 %45)
  %1385 = extractvalue { i32, i32 } %1384, 0		; visa id: 1842
  %1386 = extractvalue { i32, i32 } %1384, 1		; visa id: 1842
  %1387 = insertelement <2 x i32> undef, i32 %1385, i32 0		; visa id: 1849
  %1388 = insertelement <2 x i32> %1387, i32 %1386, i32 1		; visa id: 1850
  %1389 = bitcast <2 x i32> %1388 to i64		; visa id: 1851
  %1390 = shl i64 %1389, 1		; visa id: 1855
  %1391 = add i64 %.in400, %1390		; visa id: 1856
  %1392 = ashr i64 %1375, 31		; visa id: 1857
  %1393 = bitcast i64 %1392 to <2 x i32>		; visa id: 1858
  %1394 = extractelement <2 x i32> %1393, i32 0		; visa id: 1862
  %1395 = extractelement <2 x i32> %1393, i32 1		; visa id: 1862
  %1396 = and i32 %1394, -2		; visa id: 1862
  %1397 = insertelement <2 x i32> undef, i32 %1396, i32 0		; visa id: 1863
  %1398 = insertelement <2 x i32> %1397, i32 %1395, i32 1		; visa id: 1864
  %1399 = bitcast <2 x i32> %1398 to i64		; visa id: 1865
  %1400 = add i64 %1391, %1399		; visa id: 1869
  %1401 = inttoptr i64 %1400 to i16 addrspace(4)*		; visa id: 1870
  %1402 = addrspacecast i16 addrspace(4)* %1401 to i16 addrspace(1)*		; visa id: 1870
  %1403 = load i16, i16 addrspace(1)* %1402, align 2		; visa id: 1871
  %1404 = zext i16 %1360 to i32		; visa id: 1873
  %1405 = shl nuw i32 %1404, 16, !spirv.Decorations !639		; visa id: 1874
  %1406 = bitcast i32 %1405 to float
  %1407 = zext i16 %1403 to i32		; visa id: 1875
  %1408 = shl nuw i32 %1407, 16, !spirv.Decorations !639		; visa id: 1876
  %1409 = bitcast i32 %1408 to float
  %1410 = fmul reassoc nsz arcp contract float %1406, %1409, !spirv.Decorations !618
  %1411 = fadd reassoc nsz arcp contract float %1410, %.sroa.14.1, !spirv.Decorations !618		; visa id: 1877
  br label %._crit_edge.375, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1878

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %1374
; BB123 :
  %.sroa.14.2 = phi float [ %1411, %1374 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  %1412 = icmp slt i32 %230, %const_reg_dword
  %1413 = icmp slt i32 %1313, %const_reg_dword1		; visa id: 1879
  %1414 = and i1 %1412, %1413		; visa id: 1880
  br i1 %1414, label %1415, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 1882

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
; BB:
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1415:                                             ; preds = %._crit_edge.375
; BB125 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 1884
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 1884
  %1416 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 1884
  %1417 = insertelement <2 x i32> %1416, i32 %113, i64 1		; visa id: 1885
  %1418 = inttoptr i64 %133 to <2 x i32>*		; visa id: 1886
  store <2 x i32> %1417, <2 x i32>* %1418, align 4, !noalias !625		; visa id: 1886
  br label %._crit_edge233, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1888

._crit_edge233:                                   ; preds = %._crit_edge233.._crit_edge233_crit_edge, %1415
; BB126 :
  %1419 = phi i32 [ 0, %1415 ], [ %1428, %._crit_edge233.._crit_edge233_crit_edge ]
  %1420 = zext i32 %1419 to i64		; visa id: 1889
  %1421 = shl nuw nsw i64 %1420, 2		; visa id: 1890
  %1422 = add i64 %133, %1421		; visa id: 1891
  %1423 = inttoptr i64 %1422 to i32*		; visa id: 1892
  %1424 = load i32, i32* %1423, align 4, !noalias !625		; visa id: 1892
  %1425 = add i64 %128, %1421		; visa id: 1893
  %1426 = inttoptr i64 %1425 to i32*		; visa id: 1894
  store i32 %1424, i32* %1426, align 4, !alias.scope !625		; visa id: 1894
  %1427 = icmp eq i32 %1419, 0		; visa id: 1895
  br i1 %1427, label %._crit_edge233.._crit_edge233_crit_edge, label %1429, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1896

._crit_edge233.._crit_edge233_crit_edge:          ; preds = %._crit_edge233
; BB127 :
  %1428 = add nuw nsw i32 %1419, 1, !spirv.Decorations !631		; visa id: 1898
  br label %._crit_edge233, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1899

1429:                                             ; preds = %._crit_edge233
; BB128 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 1901
  %1430 = load i64, i64* %129, align 8		; visa id: 1901
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 1902
  %1431 = bitcast i64 %1430 to <2 x i32>		; visa id: 1902
  %1432 = extractelement <2 x i32> %1431, i32 0		; visa id: 1904
  %1433 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1432, i32 1
  %1434 = bitcast <2 x i32> %1433 to i64		; visa id: 1904
  %1435 = ashr exact i64 %1434, 32		; visa id: 1905
  %1436 = bitcast i64 %1435 to <2 x i32>		; visa id: 1906
  %1437 = extractelement <2 x i32> %1436, i32 0		; visa id: 1910
  %1438 = extractelement <2 x i32> %1436, i32 1		; visa id: 1910
  %1439 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1437, i32 %1438, i32 %41, i32 %42)
  %1440 = extractvalue { i32, i32 } %1439, 0		; visa id: 1910
  %1441 = extractvalue { i32, i32 } %1439, 1		; visa id: 1910
  %1442 = insertelement <2 x i32> undef, i32 %1440, i32 0		; visa id: 1917
  %1443 = insertelement <2 x i32> %1442, i32 %1441, i32 1		; visa id: 1918
  %1444 = bitcast <2 x i32> %1443 to i64		; visa id: 1919
  %1445 = shl i64 %1444, 1		; visa id: 1923
  %1446 = add i64 %.in401, %1445		; visa id: 1924
  %1447 = ashr i64 %1430, 31		; visa id: 1925
  %1448 = bitcast i64 %1447 to <2 x i32>		; visa id: 1926
  %1449 = extractelement <2 x i32> %1448, i32 0		; visa id: 1930
  %1450 = extractelement <2 x i32> %1448, i32 1		; visa id: 1930
  %1451 = and i32 %1449, -2		; visa id: 1930
  %1452 = insertelement <2 x i32> undef, i32 %1451, i32 0		; visa id: 1931
  %1453 = insertelement <2 x i32> %1452, i32 %1450, i32 1		; visa id: 1932
  %1454 = bitcast <2 x i32> %1453 to i64		; visa id: 1933
  %1455 = add i64 %1446, %1454		; visa id: 1937
  %1456 = inttoptr i64 %1455 to i16 addrspace(4)*		; visa id: 1938
  %1457 = addrspacecast i16 addrspace(4)* %1456 to i16 addrspace(1)*		; visa id: 1938
  %1458 = load i16, i16 addrspace(1)* %1457, align 2		; visa id: 1939
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 1941
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 1941
  %1459 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 1941
  %1460 = insertelement <2 x i32> %1459, i32 %1313, i64 1		; visa id: 1942
  %1461 = inttoptr i64 %124 to <2 x i32>*		; visa id: 1943
  store <2 x i32> %1460, <2 x i32>* %1461, align 4, !noalias !635		; visa id: 1943
  br label %._crit_edge234, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 1945

._crit_edge234:                                   ; preds = %._crit_edge234.._crit_edge234_crit_edge, %1429
; BB129 :
  %1462 = phi i32 [ 0, %1429 ], [ %1471, %._crit_edge234.._crit_edge234_crit_edge ]
  %1463 = zext i32 %1462 to i64		; visa id: 1946
  %1464 = shl nuw nsw i64 %1463, 2		; visa id: 1947
  %1465 = add i64 %124, %1464		; visa id: 1948
  %1466 = inttoptr i64 %1465 to i32*		; visa id: 1949
  %1467 = load i32, i32* %1466, align 4, !noalias !635		; visa id: 1949
  %1468 = add i64 %119, %1464		; visa id: 1950
  %1469 = inttoptr i64 %1468 to i32*		; visa id: 1951
  store i32 %1467, i32* %1469, align 4, !alias.scope !635		; visa id: 1951
  %1470 = icmp eq i32 %1462, 0		; visa id: 1952
  br i1 %1470, label %._crit_edge234.._crit_edge234_crit_edge, label %1472, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 1953

._crit_edge234.._crit_edge234_crit_edge:          ; preds = %._crit_edge234
; BB130 :
  %1471 = add nuw nsw i32 %1462, 1, !spirv.Decorations !631		; visa id: 1955
  br label %._crit_edge234, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 1956

1472:                                             ; preds = %._crit_edge234
; BB131 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 1958
  %1473 = load i64, i64* %120, align 8		; visa id: 1958
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 1959
  %1474 = bitcast i64 %1473 to <2 x i32>		; visa id: 1959
  %1475 = extractelement <2 x i32> %1474, i32 0		; visa id: 1961
  %1476 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1475, i32 1
  %1477 = bitcast <2 x i32> %1476 to i64		; visa id: 1961
  %1478 = ashr exact i64 %1477, 32		; visa id: 1962
  %1479 = bitcast i64 %1478 to <2 x i32>		; visa id: 1963
  %1480 = extractelement <2 x i32> %1479, i32 0		; visa id: 1967
  %1481 = extractelement <2 x i32> %1479, i32 1		; visa id: 1967
  %1482 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1480, i32 %1481, i32 %44, i32 %45)
  %1483 = extractvalue { i32, i32 } %1482, 0		; visa id: 1967
  %1484 = extractvalue { i32, i32 } %1482, 1		; visa id: 1967
  %1485 = insertelement <2 x i32> undef, i32 %1483, i32 0		; visa id: 1974
  %1486 = insertelement <2 x i32> %1485, i32 %1484, i32 1		; visa id: 1975
  %1487 = bitcast <2 x i32> %1486 to i64		; visa id: 1976
  %1488 = shl i64 %1487, 1		; visa id: 1980
  %1489 = add i64 %.in400, %1488		; visa id: 1981
  %1490 = ashr i64 %1473, 31		; visa id: 1982
  %1491 = bitcast i64 %1490 to <2 x i32>		; visa id: 1983
  %1492 = extractelement <2 x i32> %1491, i32 0		; visa id: 1987
  %1493 = extractelement <2 x i32> %1491, i32 1		; visa id: 1987
  %1494 = and i32 %1492, -2		; visa id: 1987
  %1495 = insertelement <2 x i32> undef, i32 %1494, i32 0		; visa id: 1988
  %1496 = insertelement <2 x i32> %1495, i32 %1493, i32 1		; visa id: 1989
  %1497 = bitcast <2 x i32> %1496 to i64		; visa id: 1990
  %1498 = add i64 %1489, %1497		; visa id: 1994
  %1499 = inttoptr i64 %1498 to i16 addrspace(4)*		; visa id: 1995
  %1500 = addrspacecast i16 addrspace(4)* %1499 to i16 addrspace(1)*		; visa id: 1995
  %1501 = load i16, i16 addrspace(1)* %1500, align 2		; visa id: 1996
  %1502 = zext i16 %1458 to i32		; visa id: 1998
  %1503 = shl nuw i32 %1502, 16, !spirv.Decorations !639		; visa id: 1999
  %1504 = bitcast i32 %1503 to float
  %1505 = zext i16 %1501 to i32		; visa id: 2000
  %1506 = shl nuw i32 %1505, 16, !spirv.Decorations !639		; visa id: 2001
  %1507 = bitcast i32 %1506 to float
  %1508 = fmul reassoc nsz arcp contract float %1504, %1507, !spirv.Decorations !618
  %1509 = fadd reassoc nsz arcp contract float %1508, %.sroa.78.1, !spirv.Decorations !618		; visa id: 2002
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2003

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %1472
; BB132 :
  %.sroa.78.2 = phi float [ %1509, %1472 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  %1510 = icmp slt i32 %329, %const_reg_dword
  %1511 = icmp slt i32 %1313, %const_reg_dword1		; visa id: 2004
  %1512 = and i1 %1510, %1511		; visa id: 2005
  br i1 %1512, label %1513, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2007

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
; BB:
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1513:                                             ; preds = %._crit_edge.1.3
; BB134 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2009
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2009
  %1514 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 2009
  %1515 = insertelement <2 x i32> %1514, i32 %113, i64 1		; visa id: 2010
  %1516 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2011
  store <2 x i32> %1515, <2 x i32>* %1516, align 4, !noalias !625		; visa id: 2011
  br label %._crit_edge235, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2013

._crit_edge235:                                   ; preds = %._crit_edge235.._crit_edge235_crit_edge, %1513
; BB135 :
  %1517 = phi i32 [ 0, %1513 ], [ %1526, %._crit_edge235.._crit_edge235_crit_edge ]
  %1518 = zext i32 %1517 to i64		; visa id: 2014
  %1519 = shl nuw nsw i64 %1518, 2		; visa id: 2015
  %1520 = add i64 %133, %1519		; visa id: 2016
  %1521 = inttoptr i64 %1520 to i32*		; visa id: 2017
  %1522 = load i32, i32* %1521, align 4, !noalias !625		; visa id: 2017
  %1523 = add i64 %128, %1519		; visa id: 2018
  %1524 = inttoptr i64 %1523 to i32*		; visa id: 2019
  store i32 %1522, i32* %1524, align 4, !alias.scope !625		; visa id: 2019
  %1525 = icmp eq i32 %1517, 0		; visa id: 2020
  br i1 %1525, label %._crit_edge235.._crit_edge235_crit_edge, label %1527, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2021

._crit_edge235.._crit_edge235_crit_edge:          ; preds = %._crit_edge235
; BB136 :
  %1526 = add nuw nsw i32 %1517, 1, !spirv.Decorations !631		; visa id: 2023
  br label %._crit_edge235, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2024

1527:                                             ; preds = %._crit_edge235
; BB137 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2026
  %1528 = load i64, i64* %129, align 8		; visa id: 2026
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2027
  %1529 = bitcast i64 %1528 to <2 x i32>		; visa id: 2027
  %1530 = extractelement <2 x i32> %1529, i32 0		; visa id: 2029
  %1531 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1530, i32 1
  %1532 = bitcast <2 x i32> %1531 to i64		; visa id: 2029
  %1533 = ashr exact i64 %1532, 32		; visa id: 2030
  %1534 = bitcast i64 %1533 to <2 x i32>		; visa id: 2031
  %1535 = extractelement <2 x i32> %1534, i32 0		; visa id: 2035
  %1536 = extractelement <2 x i32> %1534, i32 1		; visa id: 2035
  %1537 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1535, i32 %1536, i32 %41, i32 %42)
  %1538 = extractvalue { i32, i32 } %1537, 0		; visa id: 2035
  %1539 = extractvalue { i32, i32 } %1537, 1		; visa id: 2035
  %1540 = insertelement <2 x i32> undef, i32 %1538, i32 0		; visa id: 2042
  %1541 = insertelement <2 x i32> %1540, i32 %1539, i32 1		; visa id: 2043
  %1542 = bitcast <2 x i32> %1541 to i64		; visa id: 2044
  %1543 = shl i64 %1542, 1		; visa id: 2048
  %1544 = add i64 %.in401, %1543		; visa id: 2049
  %1545 = ashr i64 %1528, 31		; visa id: 2050
  %1546 = bitcast i64 %1545 to <2 x i32>		; visa id: 2051
  %1547 = extractelement <2 x i32> %1546, i32 0		; visa id: 2055
  %1548 = extractelement <2 x i32> %1546, i32 1		; visa id: 2055
  %1549 = and i32 %1547, -2		; visa id: 2055
  %1550 = insertelement <2 x i32> undef, i32 %1549, i32 0		; visa id: 2056
  %1551 = insertelement <2 x i32> %1550, i32 %1548, i32 1		; visa id: 2057
  %1552 = bitcast <2 x i32> %1551 to i64		; visa id: 2058
  %1553 = add i64 %1544, %1552		; visa id: 2062
  %1554 = inttoptr i64 %1553 to i16 addrspace(4)*		; visa id: 2063
  %1555 = addrspacecast i16 addrspace(4)* %1554 to i16 addrspace(1)*		; visa id: 2063
  %1556 = load i16, i16 addrspace(1)* %1555, align 2		; visa id: 2064
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2066
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2066
  %1557 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2066
  %1558 = insertelement <2 x i32> %1557, i32 %1313, i64 1		; visa id: 2067
  %1559 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2068
  store <2 x i32> %1558, <2 x i32>* %1559, align 4, !noalias !635		; visa id: 2068
  br label %._crit_edge236, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2070

._crit_edge236:                                   ; preds = %._crit_edge236.._crit_edge236_crit_edge, %1527
; BB138 :
  %1560 = phi i32 [ 0, %1527 ], [ %1569, %._crit_edge236.._crit_edge236_crit_edge ]
  %1561 = zext i32 %1560 to i64		; visa id: 2071
  %1562 = shl nuw nsw i64 %1561, 2		; visa id: 2072
  %1563 = add i64 %124, %1562		; visa id: 2073
  %1564 = inttoptr i64 %1563 to i32*		; visa id: 2074
  %1565 = load i32, i32* %1564, align 4, !noalias !635		; visa id: 2074
  %1566 = add i64 %119, %1562		; visa id: 2075
  %1567 = inttoptr i64 %1566 to i32*		; visa id: 2076
  store i32 %1565, i32* %1567, align 4, !alias.scope !635		; visa id: 2076
  %1568 = icmp eq i32 %1560, 0		; visa id: 2077
  br i1 %1568, label %._crit_edge236.._crit_edge236_crit_edge, label %1570, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2078

._crit_edge236.._crit_edge236_crit_edge:          ; preds = %._crit_edge236
; BB139 :
  %1569 = add nuw nsw i32 %1560, 1, !spirv.Decorations !631		; visa id: 2080
  br label %._crit_edge236, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2081

1570:                                             ; preds = %._crit_edge236
; BB140 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2083
  %1571 = load i64, i64* %120, align 8		; visa id: 2083
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2084
  %1572 = bitcast i64 %1571 to <2 x i32>		; visa id: 2084
  %1573 = extractelement <2 x i32> %1572, i32 0		; visa id: 2086
  %1574 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1573, i32 1
  %1575 = bitcast <2 x i32> %1574 to i64		; visa id: 2086
  %1576 = ashr exact i64 %1575, 32		; visa id: 2087
  %1577 = bitcast i64 %1576 to <2 x i32>		; visa id: 2088
  %1578 = extractelement <2 x i32> %1577, i32 0		; visa id: 2092
  %1579 = extractelement <2 x i32> %1577, i32 1		; visa id: 2092
  %1580 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1578, i32 %1579, i32 %44, i32 %45)
  %1581 = extractvalue { i32, i32 } %1580, 0		; visa id: 2092
  %1582 = extractvalue { i32, i32 } %1580, 1		; visa id: 2092
  %1583 = insertelement <2 x i32> undef, i32 %1581, i32 0		; visa id: 2099
  %1584 = insertelement <2 x i32> %1583, i32 %1582, i32 1		; visa id: 2100
  %1585 = bitcast <2 x i32> %1584 to i64		; visa id: 2101
  %1586 = shl i64 %1585, 1		; visa id: 2105
  %1587 = add i64 %.in400, %1586		; visa id: 2106
  %1588 = ashr i64 %1571, 31		; visa id: 2107
  %1589 = bitcast i64 %1588 to <2 x i32>		; visa id: 2108
  %1590 = extractelement <2 x i32> %1589, i32 0		; visa id: 2112
  %1591 = extractelement <2 x i32> %1589, i32 1		; visa id: 2112
  %1592 = and i32 %1590, -2		; visa id: 2112
  %1593 = insertelement <2 x i32> undef, i32 %1592, i32 0		; visa id: 2113
  %1594 = insertelement <2 x i32> %1593, i32 %1591, i32 1		; visa id: 2114
  %1595 = bitcast <2 x i32> %1594 to i64		; visa id: 2115
  %1596 = add i64 %1587, %1595		; visa id: 2119
  %1597 = inttoptr i64 %1596 to i16 addrspace(4)*		; visa id: 2120
  %1598 = addrspacecast i16 addrspace(4)* %1597 to i16 addrspace(1)*		; visa id: 2120
  %1599 = load i16, i16 addrspace(1)* %1598, align 2		; visa id: 2121
  %1600 = zext i16 %1556 to i32		; visa id: 2123
  %1601 = shl nuw i32 %1600, 16, !spirv.Decorations !639		; visa id: 2124
  %1602 = bitcast i32 %1601 to float
  %1603 = zext i16 %1599 to i32		; visa id: 2125
  %1604 = shl nuw i32 %1603, 16, !spirv.Decorations !639		; visa id: 2126
  %1605 = bitcast i32 %1604 to float
  %1606 = fmul reassoc nsz arcp contract float %1602, %1605, !spirv.Decorations !618
  %1607 = fadd reassoc nsz arcp contract float %1606, %.sroa.142.1, !spirv.Decorations !618		; visa id: 2127
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2128

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %1570
; BB141 :
  %.sroa.142.2 = phi float [ %1607, %1570 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  %1608 = icmp slt i32 %428, %const_reg_dword
  %1609 = icmp slt i32 %1313, %const_reg_dword1		; visa id: 2129
  %1610 = and i1 %1608, %1609		; visa id: 2130
  br i1 %1610, label %1611, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2132

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
; BB:
  br label %.preheader.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1611:                                             ; preds = %._crit_edge.2.3
; BB143 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2134
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2134
  %1612 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 2134
  %1613 = insertelement <2 x i32> %1612, i32 %113, i64 1		; visa id: 2135
  %1614 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2136
  store <2 x i32> %1613, <2 x i32>* %1614, align 4, !noalias !625		; visa id: 2136
  br label %._crit_edge237, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2138

._crit_edge237:                                   ; preds = %._crit_edge237.._crit_edge237_crit_edge, %1611
; BB144 :
  %1615 = phi i32 [ 0, %1611 ], [ %1624, %._crit_edge237.._crit_edge237_crit_edge ]
  %1616 = zext i32 %1615 to i64		; visa id: 2139
  %1617 = shl nuw nsw i64 %1616, 2		; visa id: 2140
  %1618 = add i64 %133, %1617		; visa id: 2141
  %1619 = inttoptr i64 %1618 to i32*		; visa id: 2142
  %1620 = load i32, i32* %1619, align 4, !noalias !625		; visa id: 2142
  %1621 = add i64 %128, %1617		; visa id: 2143
  %1622 = inttoptr i64 %1621 to i32*		; visa id: 2144
  store i32 %1620, i32* %1622, align 4, !alias.scope !625		; visa id: 2144
  %1623 = icmp eq i32 %1615, 0		; visa id: 2145
  br i1 %1623, label %._crit_edge237.._crit_edge237_crit_edge, label %1625, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2146

._crit_edge237.._crit_edge237_crit_edge:          ; preds = %._crit_edge237
; BB145 :
  %1624 = add nuw nsw i32 %1615, 1, !spirv.Decorations !631		; visa id: 2148
  br label %._crit_edge237, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2149

1625:                                             ; preds = %._crit_edge237
; BB146 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2151
  %1626 = load i64, i64* %129, align 8		; visa id: 2151
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2152
  %1627 = bitcast i64 %1626 to <2 x i32>		; visa id: 2152
  %1628 = extractelement <2 x i32> %1627, i32 0		; visa id: 2154
  %1629 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1628, i32 1
  %1630 = bitcast <2 x i32> %1629 to i64		; visa id: 2154
  %1631 = ashr exact i64 %1630, 32		; visa id: 2155
  %1632 = bitcast i64 %1631 to <2 x i32>		; visa id: 2156
  %1633 = extractelement <2 x i32> %1632, i32 0		; visa id: 2160
  %1634 = extractelement <2 x i32> %1632, i32 1		; visa id: 2160
  %1635 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1633, i32 %1634, i32 %41, i32 %42)
  %1636 = extractvalue { i32, i32 } %1635, 0		; visa id: 2160
  %1637 = extractvalue { i32, i32 } %1635, 1		; visa id: 2160
  %1638 = insertelement <2 x i32> undef, i32 %1636, i32 0		; visa id: 2167
  %1639 = insertelement <2 x i32> %1638, i32 %1637, i32 1		; visa id: 2168
  %1640 = bitcast <2 x i32> %1639 to i64		; visa id: 2169
  %1641 = shl i64 %1640, 1		; visa id: 2173
  %1642 = add i64 %.in401, %1641		; visa id: 2174
  %1643 = ashr i64 %1626, 31		; visa id: 2175
  %1644 = bitcast i64 %1643 to <2 x i32>		; visa id: 2176
  %1645 = extractelement <2 x i32> %1644, i32 0		; visa id: 2180
  %1646 = extractelement <2 x i32> %1644, i32 1		; visa id: 2180
  %1647 = and i32 %1645, -2		; visa id: 2180
  %1648 = insertelement <2 x i32> undef, i32 %1647, i32 0		; visa id: 2181
  %1649 = insertelement <2 x i32> %1648, i32 %1646, i32 1		; visa id: 2182
  %1650 = bitcast <2 x i32> %1649 to i64		; visa id: 2183
  %1651 = add i64 %1642, %1650		; visa id: 2187
  %1652 = inttoptr i64 %1651 to i16 addrspace(4)*		; visa id: 2188
  %1653 = addrspacecast i16 addrspace(4)* %1652 to i16 addrspace(1)*		; visa id: 2188
  %1654 = load i16, i16 addrspace(1)* %1653, align 2		; visa id: 2189
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2191
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2191
  %1655 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2191
  %1656 = insertelement <2 x i32> %1655, i32 %1313, i64 1		; visa id: 2192
  %1657 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2193
  store <2 x i32> %1656, <2 x i32>* %1657, align 4, !noalias !635		; visa id: 2193
  br label %._crit_edge238, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2195

._crit_edge238:                                   ; preds = %._crit_edge238.._crit_edge238_crit_edge, %1625
; BB147 :
  %1658 = phi i32 [ 0, %1625 ], [ %1667, %._crit_edge238.._crit_edge238_crit_edge ]
  %1659 = zext i32 %1658 to i64		; visa id: 2196
  %1660 = shl nuw nsw i64 %1659, 2		; visa id: 2197
  %1661 = add i64 %124, %1660		; visa id: 2198
  %1662 = inttoptr i64 %1661 to i32*		; visa id: 2199
  %1663 = load i32, i32* %1662, align 4, !noalias !635		; visa id: 2199
  %1664 = add i64 %119, %1660		; visa id: 2200
  %1665 = inttoptr i64 %1664 to i32*		; visa id: 2201
  store i32 %1663, i32* %1665, align 4, !alias.scope !635		; visa id: 2201
  %1666 = icmp eq i32 %1658, 0		; visa id: 2202
  br i1 %1666, label %._crit_edge238.._crit_edge238_crit_edge, label %1668, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2203

._crit_edge238.._crit_edge238_crit_edge:          ; preds = %._crit_edge238
; BB148 :
  %1667 = add nuw nsw i32 %1658, 1, !spirv.Decorations !631		; visa id: 2205
  br label %._crit_edge238, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2206

1668:                                             ; preds = %._crit_edge238
; BB149 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2208
  %1669 = load i64, i64* %120, align 8		; visa id: 2208
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2209
  %1670 = bitcast i64 %1669 to <2 x i32>		; visa id: 2209
  %1671 = extractelement <2 x i32> %1670, i32 0		; visa id: 2211
  %1672 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1671, i32 1
  %1673 = bitcast <2 x i32> %1672 to i64		; visa id: 2211
  %1674 = ashr exact i64 %1673, 32		; visa id: 2212
  %1675 = bitcast i64 %1674 to <2 x i32>		; visa id: 2213
  %1676 = extractelement <2 x i32> %1675, i32 0		; visa id: 2217
  %1677 = extractelement <2 x i32> %1675, i32 1		; visa id: 2217
  %1678 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1676, i32 %1677, i32 %44, i32 %45)
  %1679 = extractvalue { i32, i32 } %1678, 0		; visa id: 2217
  %1680 = extractvalue { i32, i32 } %1678, 1		; visa id: 2217
  %1681 = insertelement <2 x i32> undef, i32 %1679, i32 0		; visa id: 2224
  %1682 = insertelement <2 x i32> %1681, i32 %1680, i32 1		; visa id: 2225
  %1683 = bitcast <2 x i32> %1682 to i64		; visa id: 2226
  %1684 = shl i64 %1683, 1		; visa id: 2230
  %1685 = add i64 %.in400, %1684		; visa id: 2231
  %1686 = ashr i64 %1669, 31		; visa id: 2232
  %1687 = bitcast i64 %1686 to <2 x i32>		; visa id: 2233
  %1688 = extractelement <2 x i32> %1687, i32 0		; visa id: 2237
  %1689 = extractelement <2 x i32> %1687, i32 1		; visa id: 2237
  %1690 = and i32 %1688, -2		; visa id: 2237
  %1691 = insertelement <2 x i32> undef, i32 %1690, i32 0		; visa id: 2238
  %1692 = insertelement <2 x i32> %1691, i32 %1689, i32 1		; visa id: 2239
  %1693 = bitcast <2 x i32> %1692 to i64		; visa id: 2240
  %1694 = add i64 %1685, %1693		; visa id: 2244
  %1695 = inttoptr i64 %1694 to i16 addrspace(4)*		; visa id: 2245
  %1696 = addrspacecast i16 addrspace(4)* %1695 to i16 addrspace(1)*		; visa id: 2245
  %1697 = load i16, i16 addrspace(1)* %1696, align 2		; visa id: 2246
  %1698 = zext i16 %1654 to i32		; visa id: 2248
  %1699 = shl nuw i32 %1698, 16, !spirv.Decorations !639		; visa id: 2249
  %1700 = bitcast i32 %1699 to float
  %1701 = zext i16 %1697 to i32		; visa id: 2250
  %1702 = shl nuw i32 %1701, 16, !spirv.Decorations !639		; visa id: 2251
  %1703 = bitcast i32 %1702 to float
  %1704 = fmul reassoc nsz arcp contract float %1700, %1703, !spirv.Decorations !618
  %1705 = fadd reassoc nsz arcp contract float %1704, %.sroa.206.1, !spirv.Decorations !618		; visa id: 2252
  br label %.preheader.3, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2253

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %1668
; BB150 :
  %.sroa.206.2 = phi float [ %1705, %1668 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %1706 = add i32 %69, 4		; visa id: 2254
  %1707 = icmp slt i32 %1706, %const_reg_dword1		; visa id: 2255
  %1708 = icmp slt i32 %65, %const_reg_dword
  %1709 = and i1 %1708, %1707		; visa id: 2256
  br i1 %1709, label %1710, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2258

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
; BB:
  br label %._crit_edge.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1710:                                             ; preds = %.preheader.3
; BB152 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2260
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2260
  %1711 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 2260
  %1712 = insertelement <2 x i32> %1711, i32 %113, i64 1		; visa id: 2261
  %1713 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2262
  store <2 x i32> %1712, <2 x i32>* %1713, align 4, !noalias !625		; visa id: 2262
  br label %._crit_edge239, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2264

._crit_edge239:                                   ; preds = %._crit_edge239.._crit_edge239_crit_edge, %1710
; BB153 :
  %1714 = phi i32 [ 0, %1710 ], [ %1723, %._crit_edge239.._crit_edge239_crit_edge ]
  %1715 = zext i32 %1714 to i64		; visa id: 2265
  %1716 = shl nuw nsw i64 %1715, 2		; visa id: 2266
  %1717 = add i64 %133, %1716		; visa id: 2267
  %1718 = inttoptr i64 %1717 to i32*		; visa id: 2268
  %1719 = load i32, i32* %1718, align 4, !noalias !625		; visa id: 2268
  %1720 = add i64 %128, %1716		; visa id: 2269
  %1721 = inttoptr i64 %1720 to i32*		; visa id: 2270
  store i32 %1719, i32* %1721, align 4, !alias.scope !625		; visa id: 2270
  %1722 = icmp eq i32 %1714, 0		; visa id: 2271
  br i1 %1722, label %._crit_edge239.._crit_edge239_crit_edge, label %1724, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2272

._crit_edge239.._crit_edge239_crit_edge:          ; preds = %._crit_edge239
; BB154 :
  %1723 = add nuw nsw i32 %1714, 1, !spirv.Decorations !631		; visa id: 2274
  br label %._crit_edge239, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2275

1724:                                             ; preds = %._crit_edge239
; BB155 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2277
  %1725 = load i64, i64* %129, align 8		; visa id: 2277
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2278
  %1726 = bitcast i64 %1725 to <2 x i32>		; visa id: 2278
  %1727 = extractelement <2 x i32> %1726, i32 0		; visa id: 2280
  %1728 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1727, i32 1
  %1729 = bitcast <2 x i32> %1728 to i64		; visa id: 2280
  %1730 = ashr exact i64 %1729, 32		; visa id: 2281
  %1731 = bitcast i64 %1730 to <2 x i32>		; visa id: 2282
  %1732 = extractelement <2 x i32> %1731, i32 0		; visa id: 2286
  %1733 = extractelement <2 x i32> %1731, i32 1		; visa id: 2286
  %1734 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1732, i32 %1733, i32 %41, i32 %42)
  %1735 = extractvalue { i32, i32 } %1734, 0		; visa id: 2286
  %1736 = extractvalue { i32, i32 } %1734, 1		; visa id: 2286
  %1737 = insertelement <2 x i32> undef, i32 %1735, i32 0		; visa id: 2293
  %1738 = insertelement <2 x i32> %1737, i32 %1736, i32 1		; visa id: 2294
  %1739 = bitcast <2 x i32> %1738 to i64		; visa id: 2295
  %1740 = shl i64 %1739, 1		; visa id: 2299
  %1741 = add i64 %.in401, %1740		; visa id: 2300
  %1742 = ashr i64 %1725, 31		; visa id: 2301
  %1743 = bitcast i64 %1742 to <2 x i32>		; visa id: 2302
  %1744 = extractelement <2 x i32> %1743, i32 0		; visa id: 2306
  %1745 = extractelement <2 x i32> %1743, i32 1		; visa id: 2306
  %1746 = and i32 %1744, -2		; visa id: 2306
  %1747 = insertelement <2 x i32> undef, i32 %1746, i32 0		; visa id: 2307
  %1748 = insertelement <2 x i32> %1747, i32 %1745, i32 1		; visa id: 2308
  %1749 = bitcast <2 x i32> %1748 to i64		; visa id: 2309
  %1750 = add i64 %1741, %1749		; visa id: 2313
  %1751 = inttoptr i64 %1750 to i16 addrspace(4)*		; visa id: 2314
  %1752 = addrspacecast i16 addrspace(4)* %1751 to i16 addrspace(1)*		; visa id: 2314
  %1753 = load i16, i16 addrspace(1)* %1752, align 2		; visa id: 2315
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2317
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2317
  %1754 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2317
  %1755 = insertelement <2 x i32> %1754, i32 %1706, i64 1		; visa id: 2318
  %1756 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2319
  store <2 x i32> %1755, <2 x i32>* %1756, align 4, !noalias !635		; visa id: 2319
  br label %._crit_edge240, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2321

._crit_edge240:                                   ; preds = %._crit_edge240.._crit_edge240_crit_edge, %1724
; BB156 :
  %1757 = phi i32 [ 0, %1724 ], [ %1766, %._crit_edge240.._crit_edge240_crit_edge ]
  %1758 = zext i32 %1757 to i64		; visa id: 2322
  %1759 = shl nuw nsw i64 %1758, 2		; visa id: 2323
  %1760 = add i64 %124, %1759		; visa id: 2324
  %1761 = inttoptr i64 %1760 to i32*		; visa id: 2325
  %1762 = load i32, i32* %1761, align 4, !noalias !635		; visa id: 2325
  %1763 = add i64 %119, %1759		; visa id: 2326
  %1764 = inttoptr i64 %1763 to i32*		; visa id: 2327
  store i32 %1762, i32* %1764, align 4, !alias.scope !635		; visa id: 2327
  %1765 = icmp eq i32 %1757, 0		; visa id: 2328
  br i1 %1765, label %._crit_edge240.._crit_edge240_crit_edge, label %1767, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2329

._crit_edge240.._crit_edge240_crit_edge:          ; preds = %._crit_edge240
; BB157 :
  %1766 = add nuw nsw i32 %1757, 1, !spirv.Decorations !631		; visa id: 2331
  br label %._crit_edge240, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2332

1767:                                             ; preds = %._crit_edge240
; BB158 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2334
  %1768 = load i64, i64* %120, align 8		; visa id: 2334
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2335
  %1769 = bitcast i64 %1768 to <2 x i32>		; visa id: 2335
  %1770 = extractelement <2 x i32> %1769, i32 0		; visa id: 2337
  %1771 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1770, i32 1
  %1772 = bitcast <2 x i32> %1771 to i64		; visa id: 2337
  %1773 = ashr exact i64 %1772, 32		; visa id: 2338
  %1774 = bitcast i64 %1773 to <2 x i32>		; visa id: 2339
  %1775 = extractelement <2 x i32> %1774, i32 0		; visa id: 2343
  %1776 = extractelement <2 x i32> %1774, i32 1		; visa id: 2343
  %1777 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1775, i32 %1776, i32 %44, i32 %45)
  %1778 = extractvalue { i32, i32 } %1777, 0		; visa id: 2343
  %1779 = extractvalue { i32, i32 } %1777, 1		; visa id: 2343
  %1780 = insertelement <2 x i32> undef, i32 %1778, i32 0		; visa id: 2350
  %1781 = insertelement <2 x i32> %1780, i32 %1779, i32 1		; visa id: 2351
  %1782 = bitcast <2 x i32> %1781 to i64		; visa id: 2352
  %1783 = shl i64 %1782, 1		; visa id: 2356
  %1784 = add i64 %.in400, %1783		; visa id: 2357
  %1785 = ashr i64 %1768, 31		; visa id: 2358
  %1786 = bitcast i64 %1785 to <2 x i32>		; visa id: 2359
  %1787 = extractelement <2 x i32> %1786, i32 0		; visa id: 2363
  %1788 = extractelement <2 x i32> %1786, i32 1		; visa id: 2363
  %1789 = and i32 %1787, -2		; visa id: 2363
  %1790 = insertelement <2 x i32> undef, i32 %1789, i32 0		; visa id: 2364
  %1791 = insertelement <2 x i32> %1790, i32 %1788, i32 1		; visa id: 2365
  %1792 = bitcast <2 x i32> %1791 to i64		; visa id: 2366
  %1793 = add i64 %1784, %1792		; visa id: 2370
  %1794 = inttoptr i64 %1793 to i16 addrspace(4)*		; visa id: 2371
  %1795 = addrspacecast i16 addrspace(4)* %1794 to i16 addrspace(1)*		; visa id: 2371
  %1796 = load i16, i16 addrspace(1)* %1795, align 2		; visa id: 2372
  %1797 = zext i16 %1753 to i32		; visa id: 2374
  %1798 = shl nuw i32 %1797, 16, !spirv.Decorations !639		; visa id: 2375
  %1799 = bitcast i32 %1798 to float
  %1800 = zext i16 %1796 to i32		; visa id: 2376
  %1801 = shl nuw i32 %1800, 16, !spirv.Decorations !639		; visa id: 2377
  %1802 = bitcast i32 %1801 to float
  %1803 = fmul reassoc nsz arcp contract float %1799, %1802, !spirv.Decorations !618
  %1804 = fadd reassoc nsz arcp contract float %1803, %.sroa.18.1, !spirv.Decorations !618		; visa id: 2378
  br label %._crit_edge.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2379

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %1767
; BB159 :
  %.sroa.18.2 = phi float [ %1804, %1767 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  %1805 = icmp slt i32 %230, %const_reg_dword
  %1806 = icmp slt i32 %1706, %const_reg_dword1		; visa id: 2380
  %1807 = and i1 %1805, %1806		; visa id: 2381
  br i1 %1807, label %1808, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2383

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
; BB:
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1808:                                             ; preds = %._crit_edge.4
; BB161 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2385
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2385
  %1809 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 2385
  %1810 = insertelement <2 x i32> %1809, i32 %113, i64 1		; visa id: 2386
  %1811 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2387
  store <2 x i32> %1810, <2 x i32>* %1811, align 4, !noalias !625		; visa id: 2387
  br label %._crit_edge241, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2389

._crit_edge241:                                   ; preds = %._crit_edge241.._crit_edge241_crit_edge, %1808
; BB162 :
  %1812 = phi i32 [ 0, %1808 ], [ %1821, %._crit_edge241.._crit_edge241_crit_edge ]
  %1813 = zext i32 %1812 to i64		; visa id: 2390
  %1814 = shl nuw nsw i64 %1813, 2		; visa id: 2391
  %1815 = add i64 %133, %1814		; visa id: 2392
  %1816 = inttoptr i64 %1815 to i32*		; visa id: 2393
  %1817 = load i32, i32* %1816, align 4, !noalias !625		; visa id: 2393
  %1818 = add i64 %128, %1814		; visa id: 2394
  %1819 = inttoptr i64 %1818 to i32*		; visa id: 2395
  store i32 %1817, i32* %1819, align 4, !alias.scope !625		; visa id: 2395
  %1820 = icmp eq i32 %1812, 0		; visa id: 2396
  br i1 %1820, label %._crit_edge241.._crit_edge241_crit_edge, label %1822, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2397

._crit_edge241.._crit_edge241_crit_edge:          ; preds = %._crit_edge241
; BB163 :
  %1821 = add nuw nsw i32 %1812, 1, !spirv.Decorations !631		; visa id: 2399
  br label %._crit_edge241, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2400

1822:                                             ; preds = %._crit_edge241
; BB164 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2402
  %1823 = load i64, i64* %129, align 8		; visa id: 2402
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2403
  %1824 = bitcast i64 %1823 to <2 x i32>		; visa id: 2403
  %1825 = extractelement <2 x i32> %1824, i32 0		; visa id: 2405
  %1826 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1825, i32 1
  %1827 = bitcast <2 x i32> %1826 to i64		; visa id: 2405
  %1828 = ashr exact i64 %1827, 32		; visa id: 2406
  %1829 = bitcast i64 %1828 to <2 x i32>		; visa id: 2407
  %1830 = extractelement <2 x i32> %1829, i32 0		; visa id: 2411
  %1831 = extractelement <2 x i32> %1829, i32 1		; visa id: 2411
  %1832 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1830, i32 %1831, i32 %41, i32 %42)
  %1833 = extractvalue { i32, i32 } %1832, 0		; visa id: 2411
  %1834 = extractvalue { i32, i32 } %1832, 1		; visa id: 2411
  %1835 = insertelement <2 x i32> undef, i32 %1833, i32 0		; visa id: 2418
  %1836 = insertelement <2 x i32> %1835, i32 %1834, i32 1		; visa id: 2419
  %1837 = bitcast <2 x i32> %1836 to i64		; visa id: 2420
  %1838 = shl i64 %1837, 1		; visa id: 2424
  %1839 = add i64 %.in401, %1838		; visa id: 2425
  %1840 = ashr i64 %1823, 31		; visa id: 2426
  %1841 = bitcast i64 %1840 to <2 x i32>		; visa id: 2427
  %1842 = extractelement <2 x i32> %1841, i32 0		; visa id: 2431
  %1843 = extractelement <2 x i32> %1841, i32 1		; visa id: 2431
  %1844 = and i32 %1842, -2		; visa id: 2431
  %1845 = insertelement <2 x i32> undef, i32 %1844, i32 0		; visa id: 2432
  %1846 = insertelement <2 x i32> %1845, i32 %1843, i32 1		; visa id: 2433
  %1847 = bitcast <2 x i32> %1846 to i64		; visa id: 2434
  %1848 = add i64 %1839, %1847		; visa id: 2438
  %1849 = inttoptr i64 %1848 to i16 addrspace(4)*		; visa id: 2439
  %1850 = addrspacecast i16 addrspace(4)* %1849 to i16 addrspace(1)*		; visa id: 2439
  %1851 = load i16, i16 addrspace(1)* %1850, align 2		; visa id: 2440
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2442
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2442
  %1852 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2442
  %1853 = insertelement <2 x i32> %1852, i32 %1706, i64 1		; visa id: 2443
  %1854 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2444
  store <2 x i32> %1853, <2 x i32>* %1854, align 4, !noalias !635		; visa id: 2444
  br label %._crit_edge242, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2446

._crit_edge242:                                   ; preds = %._crit_edge242.._crit_edge242_crit_edge, %1822
; BB165 :
  %1855 = phi i32 [ 0, %1822 ], [ %1864, %._crit_edge242.._crit_edge242_crit_edge ]
  %1856 = zext i32 %1855 to i64		; visa id: 2447
  %1857 = shl nuw nsw i64 %1856, 2		; visa id: 2448
  %1858 = add i64 %124, %1857		; visa id: 2449
  %1859 = inttoptr i64 %1858 to i32*		; visa id: 2450
  %1860 = load i32, i32* %1859, align 4, !noalias !635		; visa id: 2450
  %1861 = add i64 %119, %1857		; visa id: 2451
  %1862 = inttoptr i64 %1861 to i32*		; visa id: 2452
  store i32 %1860, i32* %1862, align 4, !alias.scope !635		; visa id: 2452
  %1863 = icmp eq i32 %1855, 0		; visa id: 2453
  br i1 %1863, label %._crit_edge242.._crit_edge242_crit_edge, label %1865, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2454

._crit_edge242.._crit_edge242_crit_edge:          ; preds = %._crit_edge242
; BB166 :
  %1864 = add nuw nsw i32 %1855, 1, !spirv.Decorations !631		; visa id: 2456
  br label %._crit_edge242, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2457

1865:                                             ; preds = %._crit_edge242
; BB167 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2459
  %1866 = load i64, i64* %120, align 8		; visa id: 2459
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2460
  %1867 = bitcast i64 %1866 to <2 x i32>		; visa id: 2460
  %1868 = extractelement <2 x i32> %1867, i32 0		; visa id: 2462
  %1869 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1868, i32 1
  %1870 = bitcast <2 x i32> %1869 to i64		; visa id: 2462
  %1871 = ashr exact i64 %1870, 32		; visa id: 2463
  %1872 = bitcast i64 %1871 to <2 x i32>		; visa id: 2464
  %1873 = extractelement <2 x i32> %1872, i32 0		; visa id: 2468
  %1874 = extractelement <2 x i32> %1872, i32 1		; visa id: 2468
  %1875 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1873, i32 %1874, i32 %44, i32 %45)
  %1876 = extractvalue { i32, i32 } %1875, 0		; visa id: 2468
  %1877 = extractvalue { i32, i32 } %1875, 1		; visa id: 2468
  %1878 = insertelement <2 x i32> undef, i32 %1876, i32 0		; visa id: 2475
  %1879 = insertelement <2 x i32> %1878, i32 %1877, i32 1		; visa id: 2476
  %1880 = bitcast <2 x i32> %1879 to i64		; visa id: 2477
  %1881 = shl i64 %1880, 1		; visa id: 2481
  %1882 = add i64 %.in400, %1881		; visa id: 2482
  %1883 = ashr i64 %1866, 31		; visa id: 2483
  %1884 = bitcast i64 %1883 to <2 x i32>		; visa id: 2484
  %1885 = extractelement <2 x i32> %1884, i32 0		; visa id: 2488
  %1886 = extractelement <2 x i32> %1884, i32 1		; visa id: 2488
  %1887 = and i32 %1885, -2		; visa id: 2488
  %1888 = insertelement <2 x i32> undef, i32 %1887, i32 0		; visa id: 2489
  %1889 = insertelement <2 x i32> %1888, i32 %1886, i32 1		; visa id: 2490
  %1890 = bitcast <2 x i32> %1889 to i64		; visa id: 2491
  %1891 = add i64 %1882, %1890		; visa id: 2495
  %1892 = inttoptr i64 %1891 to i16 addrspace(4)*		; visa id: 2496
  %1893 = addrspacecast i16 addrspace(4)* %1892 to i16 addrspace(1)*		; visa id: 2496
  %1894 = load i16, i16 addrspace(1)* %1893, align 2		; visa id: 2497
  %1895 = zext i16 %1851 to i32		; visa id: 2499
  %1896 = shl nuw i32 %1895, 16, !spirv.Decorations !639		; visa id: 2500
  %1897 = bitcast i32 %1896 to float
  %1898 = zext i16 %1894 to i32		; visa id: 2501
  %1899 = shl nuw i32 %1898, 16, !spirv.Decorations !639		; visa id: 2502
  %1900 = bitcast i32 %1899 to float
  %1901 = fmul reassoc nsz arcp contract float %1897, %1900, !spirv.Decorations !618
  %1902 = fadd reassoc nsz arcp contract float %1901, %.sroa.82.1, !spirv.Decorations !618		; visa id: 2503
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2504

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %1865
; BB168 :
  %.sroa.82.2 = phi float [ %1902, %1865 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  %1903 = icmp slt i32 %329, %const_reg_dword
  %1904 = icmp slt i32 %1706, %const_reg_dword1		; visa id: 2505
  %1905 = and i1 %1903, %1904		; visa id: 2506
  br i1 %1905, label %1906, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2508

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
; BB:
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

1906:                                             ; preds = %._crit_edge.1.4
; BB170 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2510
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2510
  %1907 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 2510
  %1908 = insertelement <2 x i32> %1907, i32 %113, i64 1		; visa id: 2511
  %1909 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2512
  store <2 x i32> %1908, <2 x i32>* %1909, align 4, !noalias !625		; visa id: 2512
  br label %._crit_edge243, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2514

._crit_edge243:                                   ; preds = %._crit_edge243.._crit_edge243_crit_edge, %1906
; BB171 :
  %1910 = phi i32 [ 0, %1906 ], [ %1919, %._crit_edge243.._crit_edge243_crit_edge ]
  %1911 = zext i32 %1910 to i64		; visa id: 2515
  %1912 = shl nuw nsw i64 %1911, 2		; visa id: 2516
  %1913 = add i64 %133, %1912		; visa id: 2517
  %1914 = inttoptr i64 %1913 to i32*		; visa id: 2518
  %1915 = load i32, i32* %1914, align 4, !noalias !625		; visa id: 2518
  %1916 = add i64 %128, %1912		; visa id: 2519
  %1917 = inttoptr i64 %1916 to i32*		; visa id: 2520
  store i32 %1915, i32* %1917, align 4, !alias.scope !625		; visa id: 2520
  %1918 = icmp eq i32 %1910, 0		; visa id: 2521
  br i1 %1918, label %._crit_edge243.._crit_edge243_crit_edge, label %1920, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2522

._crit_edge243.._crit_edge243_crit_edge:          ; preds = %._crit_edge243
; BB172 :
  %1919 = add nuw nsw i32 %1910, 1, !spirv.Decorations !631		; visa id: 2524
  br label %._crit_edge243, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2525

1920:                                             ; preds = %._crit_edge243
; BB173 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2527
  %1921 = load i64, i64* %129, align 8		; visa id: 2527
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2528
  %1922 = bitcast i64 %1921 to <2 x i32>		; visa id: 2528
  %1923 = extractelement <2 x i32> %1922, i32 0		; visa id: 2530
  %1924 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1923, i32 1
  %1925 = bitcast <2 x i32> %1924 to i64		; visa id: 2530
  %1926 = ashr exact i64 %1925, 32		; visa id: 2531
  %1927 = bitcast i64 %1926 to <2 x i32>		; visa id: 2532
  %1928 = extractelement <2 x i32> %1927, i32 0		; visa id: 2536
  %1929 = extractelement <2 x i32> %1927, i32 1		; visa id: 2536
  %1930 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1928, i32 %1929, i32 %41, i32 %42)
  %1931 = extractvalue { i32, i32 } %1930, 0		; visa id: 2536
  %1932 = extractvalue { i32, i32 } %1930, 1		; visa id: 2536
  %1933 = insertelement <2 x i32> undef, i32 %1931, i32 0		; visa id: 2543
  %1934 = insertelement <2 x i32> %1933, i32 %1932, i32 1		; visa id: 2544
  %1935 = bitcast <2 x i32> %1934 to i64		; visa id: 2545
  %1936 = shl i64 %1935, 1		; visa id: 2549
  %1937 = add i64 %.in401, %1936		; visa id: 2550
  %1938 = ashr i64 %1921, 31		; visa id: 2551
  %1939 = bitcast i64 %1938 to <2 x i32>		; visa id: 2552
  %1940 = extractelement <2 x i32> %1939, i32 0		; visa id: 2556
  %1941 = extractelement <2 x i32> %1939, i32 1		; visa id: 2556
  %1942 = and i32 %1940, -2		; visa id: 2556
  %1943 = insertelement <2 x i32> undef, i32 %1942, i32 0		; visa id: 2557
  %1944 = insertelement <2 x i32> %1943, i32 %1941, i32 1		; visa id: 2558
  %1945 = bitcast <2 x i32> %1944 to i64		; visa id: 2559
  %1946 = add i64 %1937, %1945		; visa id: 2563
  %1947 = inttoptr i64 %1946 to i16 addrspace(4)*		; visa id: 2564
  %1948 = addrspacecast i16 addrspace(4)* %1947 to i16 addrspace(1)*		; visa id: 2564
  %1949 = load i16, i16 addrspace(1)* %1948, align 2		; visa id: 2565
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2567
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2567
  %1950 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2567
  %1951 = insertelement <2 x i32> %1950, i32 %1706, i64 1		; visa id: 2568
  %1952 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2569
  store <2 x i32> %1951, <2 x i32>* %1952, align 4, !noalias !635		; visa id: 2569
  br label %._crit_edge244, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2571

._crit_edge244:                                   ; preds = %._crit_edge244.._crit_edge244_crit_edge, %1920
; BB174 :
  %1953 = phi i32 [ 0, %1920 ], [ %1962, %._crit_edge244.._crit_edge244_crit_edge ]
  %1954 = zext i32 %1953 to i64		; visa id: 2572
  %1955 = shl nuw nsw i64 %1954, 2		; visa id: 2573
  %1956 = add i64 %124, %1955		; visa id: 2574
  %1957 = inttoptr i64 %1956 to i32*		; visa id: 2575
  %1958 = load i32, i32* %1957, align 4, !noalias !635		; visa id: 2575
  %1959 = add i64 %119, %1955		; visa id: 2576
  %1960 = inttoptr i64 %1959 to i32*		; visa id: 2577
  store i32 %1958, i32* %1960, align 4, !alias.scope !635		; visa id: 2577
  %1961 = icmp eq i32 %1953, 0		; visa id: 2578
  br i1 %1961, label %._crit_edge244.._crit_edge244_crit_edge, label %1963, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2579

._crit_edge244.._crit_edge244_crit_edge:          ; preds = %._crit_edge244
; BB175 :
  %1962 = add nuw nsw i32 %1953, 1, !spirv.Decorations !631		; visa id: 2581
  br label %._crit_edge244, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2582

1963:                                             ; preds = %._crit_edge244
; BB176 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2584
  %1964 = load i64, i64* %120, align 8		; visa id: 2584
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2585
  %1965 = bitcast i64 %1964 to <2 x i32>		; visa id: 2585
  %1966 = extractelement <2 x i32> %1965, i32 0		; visa id: 2587
  %1967 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %1966, i32 1
  %1968 = bitcast <2 x i32> %1967 to i64		; visa id: 2587
  %1969 = ashr exact i64 %1968, 32		; visa id: 2588
  %1970 = bitcast i64 %1969 to <2 x i32>		; visa id: 2589
  %1971 = extractelement <2 x i32> %1970, i32 0		; visa id: 2593
  %1972 = extractelement <2 x i32> %1970, i32 1		; visa id: 2593
  %1973 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %1971, i32 %1972, i32 %44, i32 %45)
  %1974 = extractvalue { i32, i32 } %1973, 0		; visa id: 2593
  %1975 = extractvalue { i32, i32 } %1973, 1		; visa id: 2593
  %1976 = insertelement <2 x i32> undef, i32 %1974, i32 0		; visa id: 2600
  %1977 = insertelement <2 x i32> %1976, i32 %1975, i32 1		; visa id: 2601
  %1978 = bitcast <2 x i32> %1977 to i64		; visa id: 2602
  %1979 = shl i64 %1978, 1		; visa id: 2606
  %1980 = add i64 %.in400, %1979		; visa id: 2607
  %1981 = ashr i64 %1964, 31		; visa id: 2608
  %1982 = bitcast i64 %1981 to <2 x i32>		; visa id: 2609
  %1983 = extractelement <2 x i32> %1982, i32 0		; visa id: 2613
  %1984 = extractelement <2 x i32> %1982, i32 1		; visa id: 2613
  %1985 = and i32 %1983, -2		; visa id: 2613
  %1986 = insertelement <2 x i32> undef, i32 %1985, i32 0		; visa id: 2614
  %1987 = insertelement <2 x i32> %1986, i32 %1984, i32 1		; visa id: 2615
  %1988 = bitcast <2 x i32> %1987 to i64		; visa id: 2616
  %1989 = add i64 %1980, %1988		; visa id: 2620
  %1990 = inttoptr i64 %1989 to i16 addrspace(4)*		; visa id: 2621
  %1991 = addrspacecast i16 addrspace(4)* %1990 to i16 addrspace(1)*		; visa id: 2621
  %1992 = load i16, i16 addrspace(1)* %1991, align 2		; visa id: 2622
  %1993 = zext i16 %1949 to i32		; visa id: 2624
  %1994 = shl nuw i32 %1993, 16, !spirv.Decorations !639		; visa id: 2625
  %1995 = bitcast i32 %1994 to float
  %1996 = zext i16 %1992 to i32		; visa id: 2626
  %1997 = shl nuw i32 %1996, 16, !spirv.Decorations !639		; visa id: 2627
  %1998 = bitcast i32 %1997 to float
  %1999 = fmul reassoc nsz arcp contract float %1995, %1998, !spirv.Decorations !618
  %2000 = fadd reassoc nsz arcp contract float %1999, %.sroa.146.1, !spirv.Decorations !618		; visa id: 2628
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2629

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %1963
; BB177 :
  %.sroa.146.2 = phi float [ %2000, %1963 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  %2001 = icmp slt i32 %428, %const_reg_dword
  %2002 = icmp slt i32 %1706, %const_reg_dword1		; visa id: 2630
  %2003 = and i1 %2001, %2002		; visa id: 2631
  br i1 %2003, label %2004, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2633

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
; BB:
  br label %.preheader.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2004:                                             ; preds = %._crit_edge.2.4
; BB179 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2635
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2635
  %2005 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 2635
  %2006 = insertelement <2 x i32> %2005, i32 %113, i64 1		; visa id: 2636
  %2007 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2637
  store <2 x i32> %2006, <2 x i32>* %2007, align 4, !noalias !625		; visa id: 2637
  br label %._crit_edge245, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2639

._crit_edge245:                                   ; preds = %._crit_edge245.._crit_edge245_crit_edge, %2004
; BB180 :
  %2008 = phi i32 [ 0, %2004 ], [ %2017, %._crit_edge245.._crit_edge245_crit_edge ]
  %2009 = zext i32 %2008 to i64		; visa id: 2640
  %2010 = shl nuw nsw i64 %2009, 2		; visa id: 2641
  %2011 = add i64 %133, %2010		; visa id: 2642
  %2012 = inttoptr i64 %2011 to i32*		; visa id: 2643
  %2013 = load i32, i32* %2012, align 4, !noalias !625		; visa id: 2643
  %2014 = add i64 %128, %2010		; visa id: 2644
  %2015 = inttoptr i64 %2014 to i32*		; visa id: 2645
  store i32 %2013, i32* %2015, align 4, !alias.scope !625		; visa id: 2645
  %2016 = icmp eq i32 %2008, 0		; visa id: 2646
  br i1 %2016, label %._crit_edge245.._crit_edge245_crit_edge, label %2018, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2647

._crit_edge245.._crit_edge245_crit_edge:          ; preds = %._crit_edge245
; BB181 :
  %2017 = add nuw nsw i32 %2008, 1, !spirv.Decorations !631		; visa id: 2649
  br label %._crit_edge245, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2650

2018:                                             ; preds = %._crit_edge245
; BB182 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2652
  %2019 = load i64, i64* %129, align 8		; visa id: 2652
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2653
  %2020 = bitcast i64 %2019 to <2 x i32>		; visa id: 2653
  %2021 = extractelement <2 x i32> %2020, i32 0		; visa id: 2655
  %2022 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2021, i32 1
  %2023 = bitcast <2 x i32> %2022 to i64		; visa id: 2655
  %2024 = ashr exact i64 %2023, 32		; visa id: 2656
  %2025 = bitcast i64 %2024 to <2 x i32>		; visa id: 2657
  %2026 = extractelement <2 x i32> %2025, i32 0		; visa id: 2661
  %2027 = extractelement <2 x i32> %2025, i32 1		; visa id: 2661
  %2028 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2026, i32 %2027, i32 %41, i32 %42)
  %2029 = extractvalue { i32, i32 } %2028, 0		; visa id: 2661
  %2030 = extractvalue { i32, i32 } %2028, 1		; visa id: 2661
  %2031 = insertelement <2 x i32> undef, i32 %2029, i32 0		; visa id: 2668
  %2032 = insertelement <2 x i32> %2031, i32 %2030, i32 1		; visa id: 2669
  %2033 = bitcast <2 x i32> %2032 to i64		; visa id: 2670
  %2034 = shl i64 %2033, 1		; visa id: 2674
  %2035 = add i64 %.in401, %2034		; visa id: 2675
  %2036 = ashr i64 %2019, 31		; visa id: 2676
  %2037 = bitcast i64 %2036 to <2 x i32>		; visa id: 2677
  %2038 = extractelement <2 x i32> %2037, i32 0		; visa id: 2681
  %2039 = extractelement <2 x i32> %2037, i32 1		; visa id: 2681
  %2040 = and i32 %2038, -2		; visa id: 2681
  %2041 = insertelement <2 x i32> undef, i32 %2040, i32 0		; visa id: 2682
  %2042 = insertelement <2 x i32> %2041, i32 %2039, i32 1		; visa id: 2683
  %2043 = bitcast <2 x i32> %2042 to i64		; visa id: 2684
  %2044 = add i64 %2035, %2043		; visa id: 2688
  %2045 = inttoptr i64 %2044 to i16 addrspace(4)*		; visa id: 2689
  %2046 = addrspacecast i16 addrspace(4)* %2045 to i16 addrspace(1)*		; visa id: 2689
  %2047 = load i16, i16 addrspace(1)* %2046, align 2		; visa id: 2690
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2692
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2692
  %2048 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2692
  %2049 = insertelement <2 x i32> %2048, i32 %1706, i64 1		; visa id: 2693
  %2050 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2694
  store <2 x i32> %2049, <2 x i32>* %2050, align 4, !noalias !635		; visa id: 2694
  br label %._crit_edge246, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2696

._crit_edge246:                                   ; preds = %._crit_edge246.._crit_edge246_crit_edge, %2018
; BB183 :
  %2051 = phi i32 [ 0, %2018 ], [ %2060, %._crit_edge246.._crit_edge246_crit_edge ]
  %2052 = zext i32 %2051 to i64		; visa id: 2697
  %2053 = shl nuw nsw i64 %2052, 2		; visa id: 2698
  %2054 = add i64 %124, %2053		; visa id: 2699
  %2055 = inttoptr i64 %2054 to i32*		; visa id: 2700
  %2056 = load i32, i32* %2055, align 4, !noalias !635		; visa id: 2700
  %2057 = add i64 %119, %2053		; visa id: 2701
  %2058 = inttoptr i64 %2057 to i32*		; visa id: 2702
  store i32 %2056, i32* %2058, align 4, !alias.scope !635		; visa id: 2702
  %2059 = icmp eq i32 %2051, 0		; visa id: 2703
  br i1 %2059, label %._crit_edge246.._crit_edge246_crit_edge, label %2061, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2704

._crit_edge246.._crit_edge246_crit_edge:          ; preds = %._crit_edge246
; BB184 :
  %2060 = add nuw nsw i32 %2051, 1, !spirv.Decorations !631		; visa id: 2706
  br label %._crit_edge246, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2707

2061:                                             ; preds = %._crit_edge246
; BB185 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2709
  %2062 = load i64, i64* %120, align 8		; visa id: 2709
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2710
  %2063 = bitcast i64 %2062 to <2 x i32>		; visa id: 2710
  %2064 = extractelement <2 x i32> %2063, i32 0		; visa id: 2712
  %2065 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2064, i32 1
  %2066 = bitcast <2 x i32> %2065 to i64		; visa id: 2712
  %2067 = ashr exact i64 %2066, 32		; visa id: 2713
  %2068 = bitcast i64 %2067 to <2 x i32>		; visa id: 2714
  %2069 = extractelement <2 x i32> %2068, i32 0		; visa id: 2718
  %2070 = extractelement <2 x i32> %2068, i32 1		; visa id: 2718
  %2071 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2069, i32 %2070, i32 %44, i32 %45)
  %2072 = extractvalue { i32, i32 } %2071, 0		; visa id: 2718
  %2073 = extractvalue { i32, i32 } %2071, 1		; visa id: 2718
  %2074 = insertelement <2 x i32> undef, i32 %2072, i32 0		; visa id: 2725
  %2075 = insertelement <2 x i32> %2074, i32 %2073, i32 1		; visa id: 2726
  %2076 = bitcast <2 x i32> %2075 to i64		; visa id: 2727
  %2077 = shl i64 %2076, 1		; visa id: 2731
  %2078 = add i64 %.in400, %2077		; visa id: 2732
  %2079 = ashr i64 %2062, 31		; visa id: 2733
  %2080 = bitcast i64 %2079 to <2 x i32>		; visa id: 2734
  %2081 = extractelement <2 x i32> %2080, i32 0		; visa id: 2738
  %2082 = extractelement <2 x i32> %2080, i32 1		; visa id: 2738
  %2083 = and i32 %2081, -2		; visa id: 2738
  %2084 = insertelement <2 x i32> undef, i32 %2083, i32 0		; visa id: 2739
  %2085 = insertelement <2 x i32> %2084, i32 %2082, i32 1		; visa id: 2740
  %2086 = bitcast <2 x i32> %2085 to i64		; visa id: 2741
  %2087 = add i64 %2078, %2086		; visa id: 2745
  %2088 = inttoptr i64 %2087 to i16 addrspace(4)*		; visa id: 2746
  %2089 = addrspacecast i16 addrspace(4)* %2088 to i16 addrspace(1)*		; visa id: 2746
  %2090 = load i16, i16 addrspace(1)* %2089, align 2		; visa id: 2747
  %2091 = zext i16 %2047 to i32		; visa id: 2749
  %2092 = shl nuw i32 %2091, 16, !spirv.Decorations !639		; visa id: 2750
  %2093 = bitcast i32 %2092 to float
  %2094 = zext i16 %2090 to i32		; visa id: 2751
  %2095 = shl nuw i32 %2094, 16, !spirv.Decorations !639		; visa id: 2752
  %2096 = bitcast i32 %2095 to float
  %2097 = fmul reassoc nsz arcp contract float %2093, %2096, !spirv.Decorations !618
  %2098 = fadd reassoc nsz arcp contract float %2097, %.sroa.210.1, !spirv.Decorations !618		; visa id: 2753
  br label %.preheader.4, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2754

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %2061
; BB186 :
  %.sroa.210.2 = phi float [ %2098, %2061 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %2099 = add i32 %69, 5		; visa id: 2755
  %2100 = icmp slt i32 %2099, %const_reg_dword1		; visa id: 2756
  %2101 = icmp slt i32 %65, %const_reg_dword
  %2102 = and i1 %2101, %2100		; visa id: 2757
  br i1 %2102, label %2103, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2759

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
; BB:
  br label %._crit_edge.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2103:                                             ; preds = %.preheader.4
; BB188 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2761
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2761
  %2104 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 2761
  %2105 = insertelement <2 x i32> %2104, i32 %113, i64 1		; visa id: 2762
  %2106 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2763
  store <2 x i32> %2105, <2 x i32>* %2106, align 4, !noalias !625		; visa id: 2763
  br label %._crit_edge247, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2765

._crit_edge247:                                   ; preds = %._crit_edge247.._crit_edge247_crit_edge, %2103
; BB189 :
  %2107 = phi i32 [ 0, %2103 ], [ %2116, %._crit_edge247.._crit_edge247_crit_edge ]
  %2108 = zext i32 %2107 to i64		; visa id: 2766
  %2109 = shl nuw nsw i64 %2108, 2		; visa id: 2767
  %2110 = add i64 %133, %2109		; visa id: 2768
  %2111 = inttoptr i64 %2110 to i32*		; visa id: 2769
  %2112 = load i32, i32* %2111, align 4, !noalias !625		; visa id: 2769
  %2113 = add i64 %128, %2109		; visa id: 2770
  %2114 = inttoptr i64 %2113 to i32*		; visa id: 2771
  store i32 %2112, i32* %2114, align 4, !alias.scope !625		; visa id: 2771
  %2115 = icmp eq i32 %2107, 0		; visa id: 2772
  br i1 %2115, label %._crit_edge247.._crit_edge247_crit_edge, label %2117, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2773

._crit_edge247.._crit_edge247_crit_edge:          ; preds = %._crit_edge247
; BB190 :
  %2116 = add nuw nsw i32 %2107, 1, !spirv.Decorations !631		; visa id: 2775
  br label %._crit_edge247, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2776

2117:                                             ; preds = %._crit_edge247
; BB191 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2778
  %2118 = load i64, i64* %129, align 8		; visa id: 2778
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2779
  %2119 = bitcast i64 %2118 to <2 x i32>		; visa id: 2779
  %2120 = extractelement <2 x i32> %2119, i32 0		; visa id: 2781
  %2121 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2120, i32 1
  %2122 = bitcast <2 x i32> %2121 to i64		; visa id: 2781
  %2123 = ashr exact i64 %2122, 32		; visa id: 2782
  %2124 = bitcast i64 %2123 to <2 x i32>		; visa id: 2783
  %2125 = extractelement <2 x i32> %2124, i32 0		; visa id: 2787
  %2126 = extractelement <2 x i32> %2124, i32 1		; visa id: 2787
  %2127 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2125, i32 %2126, i32 %41, i32 %42)
  %2128 = extractvalue { i32, i32 } %2127, 0		; visa id: 2787
  %2129 = extractvalue { i32, i32 } %2127, 1		; visa id: 2787
  %2130 = insertelement <2 x i32> undef, i32 %2128, i32 0		; visa id: 2794
  %2131 = insertelement <2 x i32> %2130, i32 %2129, i32 1		; visa id: 2795
  %2132 = bitcast <2 x i32> %2131 to i64		; visa id: 2796
  %2133 = shl i64 %2132, 1		; visa id: 2800
  %2134 = add i64 %.in401, %2133		; visa id: 2801
  %2135 = ashr i64 %2118, 31		; visa id: 2802
  %2136 = bitcast i64 %2135 to <2 x i32>		; visa id: 2803
  %2137 = extractelement <2 x i32> %2136, i32 0		; visa id: 2807
  %2138 = extractelement <2 x i32> %2136, i32 1		; visa id: 2807
  %2139 = and i32 %2137, -2		; visa id: 2807
  %2140 = insertelement <2 x i32> undef, i32 %2139, i32 0		; visa id: 2808
  %2141 = insertelement <2 x i32> %2140, i32 %2138, i32 1		; visa id: 2809
  %2142 = bitcast <2 x i32> %2141 to i64		; visa id: 2810
  %2143 = add i64 %2134, %2142		; visa id: 2814
  %2144 = inttoptr i64 %2143 to i16 addrspace(4)*		; visa id: 2815
  %2145 = addrspacecast i16 addrspace(4)* %2144 to i16 addrspace(1)*		; visa id: 2815
  %2146 = load i16, i16 addrspace(1)* %2145, align 2		; visa id: 2816
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2818
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2818
  %2147 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2818
  %2148 = insertelement <2 x i32> %2147, i32 %2099, i64 1		; visa id: 2819
  %2149 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2820
  store <2 x i32> %2148, <2 x i32>* %2149, align 4, !noalias !635		; visa id: 2820
  br label %._crit_edge248, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2822

._crit_edge248:                                   ; preds = %._crit_edge248.._crit_edge248_crit_edge, %2117
; BB192 :
  %2150 = phi i32 [ 0, %2117 ], [ %2159, %._crit_edge248.._crit_edge248_crit_edge ]
  %2151 = zext i32 %2150 to i64		; visa id: 2823
  %2152 = shl nuw nsw i64 %2151, 2		; visa id: 2824
  %2153 = add i64 %124, %2152		; visa id: 2825
  %2154 = inttoptr i64 %2153 to i32*		; visa id: 2826
  %2155 = load i32, i32* %2154, align 4, !noalias !635		; visa id: 2826
  %2156 = add i64 %119, %2152		; visa id: 2827
  %2157 = inttoptr i64 %2156 to i32*		; visa id: 2828
  store i32 %2155, i32* %2157, align 4, !alias.scope !635		; visa id: 2828
  %2158 = icmp eq i32 %2150, 0		; visa id: 2829
  br i1 %2158, label %._crit_edge248.._crit_edge248_crit_edge, label %2160, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2830

._crit_edge248.._crit_edge248_crit_edge:          ; preds = %._crit_edge248
; BB193 :
  %2159 = add nuw nsw i32 %2150, 1, !spirv.Decorations !631		; visa id: 2832
  br label %._crit_edge248, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2833

2160:                                             ; preds = %._crit_edge248
; BB194 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2835
  %2161 = load i64, i64* %120, align 8		; visa id: 2835
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2836
  %2162 = bitcast i64 %2161 to <2 x i32>		; visa id: 2836
  %2163 = extractelement <2 x i32> %2162, i32 0		; visa id: 2838
  %2164 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2163, i32 1
  %2165 = bitcast <2 x i32> %2164 to i64		; visa id: 2838
  %2166 = ashr exact i64 %2165, 32		; visa id: 2839
  %2167 = bitcast i64 %2166 to <2 x i32>		; visa id: 2840
  %2168 = extractelement <2 x i32> %2167, i32 0		; visa id: 2844
  %2169 = extractelement <2 x i32> %2167, i32 1		; visa id: 2844
  %2170 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2168, i32 %2169, i32 %44, i32 %45)
  %2171 = extractvalue { i32, i32 } %2170, 0		; visa id: 2844
  %2172 = extractvalue { i32, i32 } %2170, 1		; visa id: 2844
  %2173 = insertelement <2 x i32> undef, i32 %2171, i32 0		; visa id: 2851
  %2174 = insertelement <2 x i32> %2173, i32 %2172, i32 1		; visa id: 2852
  %2175 = bitcast <2 x i32> %2174 to i64		; visa id: 2853
  %2176 = shl i64 %2175, 1		; visa id: 2857
  %2177 = add i64 %.in400, %2176		; visa id: 2858
  %2178 = ashr i64 %2161, 31		; visa id: 2859
  %2179 = bitcast i64 %2178 to <2 x i32>		; visa id: 2860
  %2180 = extractelement <2 x i32> %2179, i32 0		; visa id: 2864
  %2181 = extractelement <2 x i32> %2179, i32 1		; visa id: 2864
  %2182 = and i32 %2180, -2		; visa id: 2864
  %2183 = insertelement <2 x i32> undef, i32 %2182, i32 0		; visa id: 2865
  %2184 = insertelement <2 x i32> %2183, i32 %2181, i32 1		; visa id: 2866
  %2185 = bitcast <2 x i32> %2184 to i64		; visa id: 2867
  %2186 = add i64 %2177, %2185		; visa id: 2871
  %2187 = inttoptr i64 %2186 to i16 addrspace(4)*		; visa id: 2872
  %2188 = addrspacecast i16 addrspace(4)* %2187 to i16 addrspace(1)*		; visa id: 2872
  %2189 = load i16, i16 addrspace(1)* %2188, align 2		; visa id: 2873
  %2190 = zext i16 %2146 to i32		; visa id: 2875
  %2191 = shl nuw i32 %2190, 16, !spirv.Decorations !639		; visa id: 2876
  %2192 = bitcast i32 %2191 to float
  %2193 = zext i16 %2189 to i32		; visa id: 2877
  %2194 = shl nuw i32 %2193, 16, !spirv.Decorations !639		; visa id: 2878
  %2195 = bitcast i32 %2194 to float
  %2196 = fmul reassoc nsz arcp contract float %2192, %2195, !spirv.Decorations !618
  %2197 = fadd reassoc nsz arcp contract float %2196, %.sroa.22.1, !spirv.Decorations !618		; visa id: 2879
  br label %._crit_edge.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2880

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %2160
; BB195 :
  %.sroa.22.2 = phi float [ %2197, %2160 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  %2198 = icmp slt i32 %230, %const_reg_dword
  %2199 = icmp slt i32 %2099, %const_reg_dword1		; visa id: 2881
  %2200 = and i1 %2198, %2199		; visa id: 2882
  br i1 %2200, label %2201, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 2884

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
; BB:
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2201:                                             ; preds = %._crit_edge.5
; BB197 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 2886
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 2886
  %2202 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 2886
  %2203 = insertelement <2 x i32> %2202, i32 %113, i64 1		; visa id: 2887
  %2204 = inttoptr i64 %133 to <2 x i32>*		; visa id: 2888
  store <2 x i32> %2203, <2 x i32>* %2204, align 4, !noalias !625		; visa id: 2888
  br label %._crit_edge249, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2890

._crit_edge249:                                   ; preds = %._crit_edge249.._crit_edge249_crit_edge, %2201
; BB198 :
  %2205 = phi i32 [ 0, %2201 ], [ %2214, %._crit_edge249.._crit_edge249_crit_edge ]
  %2206 = zext i32 %2205 to i64		; visa id: 2891
  %2207 = shl nuw nsw i64 %2206, 2		; visa id: 2892
  %2208 = add i64 %133, %2207		; visa id: 2893
  %2209 = inttoptr i64 %2208 to i32*		; visa id: 2894
  %2210 = load i32, i32* %2209, align 4, !noalias !625		; visa id: 2894
  %2211 = add i64 %128, %2207		; visa id: 2895
  %2212 = inttoptr i64 %2211 to i32*		; visa id: 2896
  store i32 %2210, i32* %2212, align 4, !alias.scope !625		; visa id: 2896
  %2213 = icmp eq i32 %2205, 0		; visa id: 2897
  br i1 %2213, label %._crit_edge249.._crit_edge249_crit_edge, label %2215, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2898

._crit_edge249.._crit_edge249_crit_edge:          ; preds = %._crit_edge249
; BB199 :
  %2214 = add nuw nsw i32 %2205, 1, !spirv.Decorations !631		; visa id: 2900
  br label %._crit_edge249, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2901

2215:                                             ; preds = %._crit_edge249
; BB200 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 2903
  %2216 = load i64, i64* %129, align 8		; visa id: 2903
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 2904
  %2217 = bitcast i64 %2216 to <2 x i32>		; visa id: 2904
  %2218 = extractelement <2 x i32> %2217, i32 0		; visa id: 2906
  %2219 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2218, i32 1
  %2220 = bitcast <2 x i32> %2219 to i64		; visa id: 2906
  %2221 = ashr exact i64 %2220, 32		; visa id: 2907
  %2222 = bitcast i64 %2221 to <2 x i32>		; visa id: 2908
  %2223 = extractelement <2 x i32> %2222, i32 0		; visa id: 2912
  %2224 = extractelement <2 x i32> %2222, i32 1		; visa id: 2912
  %2225 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2223, i32 %2224, i32 %41, i32 %42)
  %2226 = extractvalue { i32, i32 } %2225, 0		; visa id: 2912
  %2227 = extractvalue { i32, i32 } %2225, 1		; visa id: 2912
  %2228 = insertelement <2 x i32> undef, i32 %2226, i32 0		; visa id: 2919
  %2229 = insertelement <2 x i32> %2228, i32 %2227, i32 1		; visa id: 2920
  %2230 = bitcast <2 x i32> %2229 to i64		; visa id: 2921
  %2231 = shl i64 %2230, 1		; visa id: 2925
  %2232 = add i64 %.in401, %2231		; visa id: 2926
  %2233 = ashr i64 %2216, 31		; visa id: 2927
  %2234 = bitcast i64 %2233 to <2 x i32>		; visa id: 2928
  %2235 = extractelement <2 x i32> %2234, i32 0		; visa id: 2932
  %2236 = extractelement <2 x i32> %2234, i32 1		; visa id: 2932
  %2237 = and i32 %2235, -2		; visa id: 2932
  %2238 = insertelement <2 x i32> undef, i32 %2237, i32 0		; visa id: 2933
  %2239 = insertelement <2 x i32> %2238, i32 %2236, i32 1		; visa id: 2934
  %2240 = bitcast <2 x i32> %2239 to i64		; visa id: 2935
  %2241 = add i64 %2232, %2240		; visa id: 2939
  %2242 = inttoptr i64 %2241 to i16 addrspace(4)*		; visa id: 2940
  %2243 = addrspacecast i16 addrspace(4)* %2242 to i16 addrspace(1)*		; visa id: 2940
  %2244 = load i16, i16 addrspace(1)* %2243, align 2		; visa id: 2941
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 2943
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 2943
  %2245 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 2943
  %2246 = insertelement <2 x i32> %2245, i32 %2099, i64 1		; visa id: 2944
  %2247 = inttoptr i64 %124 to <2 x i32>*		; visa id: 2945
  store <2 x i32> %2246, <2 x i32>* %2247, align 4, !noalias !635		; visa id: 2945
  br label %._crit_edge250, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 2947

._crit_edge250:                                   ; preds = %._crit_edge250.._crit_edge250_crit_edge, %2215
; BB201 :
  %2248 = phi i32 [ 0, %2215 ], [ %2257, %._crit_edge250.._crit_edge250_crit_edge ]
  %2249 = zext i32 %2248 to i64		; visa id: 2948
  %2250 = shl nuw nsw i64 %2249, 2		; visa id: 2949
  %2251 = add i64 %124, %2250		; visa id: 2950
  %2252 = inttoptr i64 %2251 to i32*		; visa id: 2951
  %2253 = load i32, i32* %2252, align 4, !noalias !635		; visa id: 2951
  %2254 = add i64 %119, %2250		; visa id: 2952
  %2255 = inttoptr i64 %2254 to i32*		; visa id: 2953
  store i32 %2253, i32* %2255, align 4, !alias.scope !635		; visa id: 2953
  %2256 = icmp eq i32 %2248, 0		; visa id: 2954
  br i1 %2256, label %._crit_edge250.._crit_edge250_crit_edge, label %2258, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 2955

._crit_edge250.._crit_edge250_crit_edge:          ; preds = %._crit_edge250
; BB202 :
  %2257 = add nuw nsw i32 %2248, 1, !spirv.Decorations !631		; visa id: 2957
  br label %._crit_edge250, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 2958

2258:                                             ; preds = %._crit_edge250
; BB203 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 2960
  %2259 = load i64, i64* %120, align 8		; visa id: 2960
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 2961
  %2260 = bitcast i64 %2259 to <2 x i32>		; visa id: 2961
  %2261 = extractelement <2 x i32> %2260, i32 0		; visa id: 2963
  %2262 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2261, i32 1
  %2263 = bitcast <2 x i32> %2262 to i64		; visa id: 2963
  %2264 = ashr exact i64 %2263, 32		; visa id: 2964
  %2265 = bitcast i64 %2264 to <2 x i32>		; visa id: 2965
  %2266 = extractelement <2 x i32> %2265, i32 0		; visa id: 2969
  %2267 = extractelement <2 x i32> %2265, i32 1		; visa id: 2969
  %2268 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2266, i32 %2267, i32 %44, i32 %45)
  %2269 = extractvalue { i32, i32 } %2268, 0		; visa id: 2969
  %2270 = extractvalue { i32, i32 } %2268, 1		; visa id: 2969
  %2271 = insertelement <2 x i32> undef, i32 %2269, i32 0		; visa id: 2976
  %2272 = insertelement <2 x i32> %2271, i32 %2270, i32 1		; visa id: 2977
  %2273 = bitcast <2 x i32> %2272 to i64		; visa id: 2978
  %2274 = shl i64 %2273, 1		; visa id: 2982
  %2275 = add i64 %.in400, %2274		; visa id: 2983
  %2276 = ashr i64 %2259, 31		; visa id: 2984
  %2277 = bitcast i64 %2276 to <2 x i32>		; visa id: 2985
  %2278 = extractelement <2 x i32> %2277, i32 0		; visa id: 2989
  %2279 = extractelement <2 x i32> %2277, i32 1		; visa id: 2989
  %2280 = and i32 %2278, -2		; visa id: 2989
  %2281 = insertelement <2 x i32> undef, i32 %2280, i32 0		; visa id: 2990
  %2282 = insertelement <2 x i32> %2281, i32 %2279, i32 1		; visa id: 2991
  %2283 = bitcast <2 x i32> %2282 to i64		; visa id: 2992
  %2284 = add i64 %2275, %2283		; visa id: 2996
  %2285 = inttoptr i64 %2284 to i16 addrspace(4)*		; visa id: 2997
  %2286 = addrspacecast i16 addrspace(4)* %2285 to i16 addrspace(1)*		; visa id: 2997
  %2287 = load i16, i16 addrspace(1)* %2286, align 2		; visa id: 2998
  %2288 = zext i16 %2244 to i32		; visa id: 3000
  %2289 = shl nuw i32 %2288, 16, !spirv.Decorations !639		; visa id: 3001
  %2290 = bitcast i32 %2289 to float
  %2291 = zext i16 %2287 to i32		; visa id: 3002
  %2292 = shl nuw i32 %2291, 16, !spirv.Decorations !639		; visa id: 3003
  %2293 = bitcast i32 %2292 to float
  %2294 = fmul reassoc nsz arcp contract float %2290, %2293, !spirv.Decorations !618
  %2295 = fadd reassoc nsz arcp contract float %2294, %.sroa.86.1, !spirv.Decorations !618		; visa id: 3004
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3005

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %2258
; BB204 :
  %.sroa.86.2 = phi float [ %2295, %2258 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  %2296 = icmp slt i32 %329, %const_reg_dword
  %2297 = icmp slt i32 %2099, %const_reg_dword1		; visa id: 3006
  %2298 = and i1 %2296, %2297		; visa id: 3007
  br i1 %2298, label %2299, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3009

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
; BB:
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2299:                                             ; preds = %._crit_edge.1.5
; BB206 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3011
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3011
  %2300 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 3011
  %2301 = insertelement <2 x i32> %2300, i32 %113, i64 1		; visa id: 3012
  %2302 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3013
  store <2 x i32> %2301, <2 x i32>* %2302, align 4, !noalias !625		; visa id: 3013
  br label %._crit_edge251, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3015

._crit_edge251:                                   ; preds = %._crit_edge251.._crit_edge251_crit_edge, %2299
; BB207 :
  %2303 = phi i32 [ 0, %2299 ], [ %2312, %._crit_edge251.._crit_edge251_crit_edge ]
  %2304 = zext i32 %2303 to i64		; visa id: 3016
  %2305 = shl nuw nsw i64 %2304, 2		; visa id: 3017
  %2306 = add i64 %133, %2305		; visa id: 3018
  %2307 = inttoptr i64 %2306 to i32*		; visa id: 3019
  %2308 = load i32, i32* %2307, align 4, !noalias !625		; visa id: 3019
  %2309 = add i64 %128, %2305		; visa id: 3020
  %2310 = inttoptr i64 %2309 to i32*		; visa id: 3021
  store i32 %2308, i32* %2310, align 4, !alias.scope !625		; visa id: 3021
  %2311 = icmp eq i32 %2303, 0		; visa id: 3022
  br i1 %2311, label %._crit_edge251.._crit_edge251_crit_edge, label %2313, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3023

._crit_edge251.._crit_edge251_crit_edge:          ; preds = %._crit_edge251
; BB208 :
  %2312 = add nuw nsw i32 %2303, 1, !spirv.Decorations !631		; visa id: 3025
  br label %._crit_edge251, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3026

2313:                                             ; preds = %._crit_edge251
; BB209 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3028
  %2314 = load i64, i64* %129, align 8		; visa id: 3028
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3029
  %2315 = bitcast i64 %2314 to <2 x i32>		; visa id: 3029
  %2316 = extractelement <2 x i32> %2315, i32 0		; visa id: 3031
  %2317 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2316, i32 1
  %2318 = bitcast <2 x i32> %2317 to i64		; visa id: 3031
  %2319 = ashr exact i64 %2318, 32		; visa id: 3032
  %2320 = bitcast i64 %2319 to <2 x i32>		; visa id: 3033
  %2321 = extractelement <2 x i32> %2320, i32 0		; visa id: 3037
  %2322 = extractelement <2 x i32> %2320, i32 1		; visa id: 3037
  %2323 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2321, i32 %2322, i32 %41, i32 %42)
  %2324 = extractvalue { i32, i32 } %2323, 0		; visa id: 3037
  %2325 = extractvalue { i32, i32 } %2323, 1		; visa id: 3037
  %2326 = insertelement <2 x i32> undef, i32 %2324, i32 0		; visa id: 3044
  %2327 = insertelement <2 x i32> %2326, i32 %2325, i32 1		; visa id: 3045
  %2328 = bitcast <2 x i32> %2327 to i64		; visa id: 3046
  %2329 = shl i64 %2328, 1		; visa id: 3050
  %2330 = add i64 %.in401, %2329		; visa id: 3051
  %2331 = ashr i64 %2314, 31		; visa id: 3052
  %2332 = bitcast i64 %2331 to <2 x i32>		; visa id: 3053
  %2333 = extractelement <2 x i32> %2332, i32 0		; visa id: 3057
  %2334 = extractelement <2 x i32> %2332, i32 1		; visa id: 3057
  %2335 = and i32 %2333, -2		; visa id: 3057
  %2336 = insertelement <2 x i32> undef, i32 %2335, i32 0		; visa id: 3058
  %2337 = insertelement <2 x i32> %2336, i32 %2334, i32 1		; visa id: 3059
  %2338 = bitcast <2 x i32> %2337 to i64		; visa id: 3060
  %2339 = add i64 %2330, %2338		; visa id: 3064
  %2340 = inttoptr i64 %2339 to i16 addrspace(4)*		; visa id: 3065
  %2341 = addrspacecast i16 addrspace(4)* %2340 to i16 addrspace(1)*		; visa id: 3065
  %2342 = load i16, i16 addrspace(1)* %2341, align 2		; visa id: 3066
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3068
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3068
  %2343 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3068
  %2344 = insertelement <2 x i32> %2343, i32 %2099, i64 1		; visa id: 3069
  %2345 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3070
  store <2 x i32> %2344, <2 x i32>* %2345, align 4, !noalias !635		; visa id: 3070
  br label %._crit_edge252, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3072

._crit_edge252:                                   ; preds = %._crit_edge252.._crit_edge252_crit_edge, %2313
; BB210 :
  %2346 = phi i32 [ 0, %2313 ], [ %2355, %._crit_edge252.._crit_edge252_crit_edge ]
  %2347 = zext i32 %2346 to i64		; visa id: 3073
  %2348 = shl nuw nsw i64 %2347, 2		; visa id: 3074
  %2349 = add i64 %124, %2348		; visa id: 3075
  %2350 = inttoptr i64 %2349 to i32*		; visa id: 3076
  %2351 = load i32, i32* %2350, align 4, !noalias !635		; visa id: 3076
  %2352 = add i64 %119, %2348		; visa id: 3077
  %2353 = inttoptr i64 %2352 to i32*		; visa id: 3078
  store i32 %2351, i32* %2353, align 4, !alias.scope !635		; visa id: 3078
  %2354 = icmp eq i32 %2346, 0		; visa id: 3079
  br i1 %2354, label %._crit_edge252.._crit_edge252_crit_edge, label %2356, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3080

._crit_edge252.._crit_edge252_crit_edge:          ; preds = %._crit_edge252
; BB211 :
  %2355 = add nuw nsw i32 %2346, 1, !spirv.Decorations !631		; visa id: 3082
  br label %._crit_edge252, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3083

2356:                                             ; preds = %._crit_edge252
; BB212 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3085
  %2357 = load i64, i64* %120, align 8		; visa id: 3085
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3086
  %2358 = bitcast i64 %2357 to <2 x i32>		; visa id: 3086
  %2359 = extractelement <2 x i32> %2358, i32 0		; visa id: 3088
  %2360 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2359, i32 1
  %2361 = bitcast <2 x i32> %2360 to i64		; visa id: 3088
  %2362 = ashr exact i64 %2361, 32		; visa id: 3089
  %2363 = bitcast i64 %2362 to <2 x i32>		; visa id: 3090
  %2364 = extractelement <2 x i32> %2363, i32 0		; visa id: 3094
  %2365 = extractelement <2 x i32> %2363, i32 1		; visa id: 3094
  %2366 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2364, i32 %2365, i32 %44, i32 %45)
  %2367 = extractvalue { i32, i32 } %2366, 0		; visa id: 3094
  %2368 = extractvalue { i32, i32 } %2366, 1		; visa id: 3094
  %2369 = insertelement <2 x i32> undef, i32 %2367, i32 0		; visa id: 3101
  %2370 = insertelement <2 x i32> %2369, i32 %2368, i32 1		; visa id: 3102
  %2371 = bitcast <2 x i32> %2370 to i64		; visa id: 3103
  %2372 = shl i64 %2371, 1		; visa id: 3107
  %2373 = add i64 %.in400, %2372		; visa id: 3108
  %2374 = ashr i64 %2357, 31		; visa id: 3109
  %2375 = bitcast i64 %2374 to <2 x i32>		; visa id: 3110
  %2376 = extractelement <2 x i32> %2375, i32 0		; visa id: 3114
  %2377 = extractelement <2 x i32> %2375, i32 1		; visa id: 3114
  %2378 = and i32 %2376, -2		; visa id: 3114
  %2379 = insertelement <2 x i32> undef, i32 %2378, i32 0		; visa id: 3115
  %2380 = insertelement <2 x i32> %2379, i32 %2377, i32 1		; visa id: 3116
  %2381 = bitcast <2 x i32> %2380 to i64		; visa id: 3117
  %2382 = add i64 %2373, %2381		; visa id: 3121
  %2383 = inttoptr i64 %2382 to i16 addrspace(4)*		; visa id: 3122
  %2384 = addrspacecast i16 addrspace(4)* %2383 to i16 addrspace(1)*		; visa id: 3122
  %2385 = load i16, i16 addrspace(1)* %2384, align 2		; visa id: 3123
  %2386 = zext i16 %2342 to i32		; visa id: 3125
  %2387 = shl nuw i32 %2386, 16, !spirv.Decorations !639		; visa id: 3126
  %2388 = bitcast i32 %2387 to float
  %2389 = zext i16 %2385 to i32		; visa id: 3127
  %2390 = shl nuw i32 %2389, 16, !spirv.Decorations !639		; visa id: 3128
  %2391 = bitcast i32 %2390 to float
  %2392 = fmul reassoc nsz arcp contract float %2388, %2391, !spirv.Decorations !618
  %2393 = fadd reassoc nsz arcp contract float %2392, %.sroa.150.1, !spirv.Decorations !618		; visa id: 3129
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3130

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %2356
; BB213 :
  %.sroa.150.2 = phi float [ %2393, %2356 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  %2394 = icmp slt i32 %428, %const_reg_dword
  %2395 = icmp slt i32 %2099, %const_reg_dword1		; visa id: 3131
  %2396 = and i1 %2394, %2395		; visa id: 3132
  br i1 %2396, label %2397, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3134

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
; BB:
  br label %.preheader.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2397:                                             ; preds = %._crit_edge.2.5
; BB215 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3136
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3136
  %2398 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 3136
  %2399 = insertelement <2 x i32> %2398, i32 %113, i64 1		; visa id: 3137
  %2400 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3138
  store <2 x i32> %2399, <2 x i32>* %2400, align 4, !noalias !625		; visa id: 3138
  br label %._crit_edge253, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3140

._crit_edge253:                                   ; preds = %._crit_edge253.._crit_edge253_crit_edge, %2397
; BB216 :
  %2401 = phi i32 [ 0, %2397 ], [ %2410, %._crit_edge253.._crit_edge253_crit_edge ]
  %2402 = zext i32 %2401 to i64		; visa id: 3141
  %2403 = shl nuw nsw i64 %2402, 2		; visa id: 3142
  %2404 = add i64 %133, %2403		; visa id: 3143
  %2405 = inttoptr i64 %2404 to i32*		; visa id: 3144
  %2406 = load i32, i32* %2405, align 4, !noalias !625		; visa id: 3144
  %2407 = add i64 %128, %2403		; visa id: 3145
  %2408 = inttoptr i64 %2407 to i32*		; visa id: 3146
  store i32 %2406, i32* %2408, align 4, !alias.scope !625		; visa id: 3146
  %2409 = icmp eq i32 %2401, 0		; visa id: 3147
  br i1 %2409, label %._crit_edge253.._crit_edge253_crit_edge, label %2411, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3148

._crit_edge253.._crit_edge253_crit_edge:          ; preds = %._crit_edge253
; BB217 :
  %2410 = add nuw nsw i32 %2401, 1, !spirv.Decorations !631		; visa id: 3150
  br label %._crit_edge253, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3151

2411:                                             ; preds = %._crit_edge253
; BB218 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3153
  %2412 = load i64, i64* %129, align 8		; visa id: 3153
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3154
  %2413 = bitcast i64 %2412 to <2 x i32>		; visa id: 3154
  %2414 = extractelement <2 x i32> %2413, i32 0		; visa id: 3156
  %2415 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2414, i32 1
  %2416 = bitcast <2 x i32> %2415 to i64		; visa id: 3156
  %2417 = ashr exact i64 %2416, 32		; visa id: 3157
  %2418 = bitcast i64 %2417 to <2 x i32>		; visa id: 3158
  %2419 = extractelement <2 x i32> %2418, i32 0		; visa id: 3162
  %2420 = extractelement <2 x i32> %2418, i32 1		; visa id: 3162
  %2421 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2419, i32 %2420, i32 %41, i32 %42)
  %2422 = extractvalue { i32, i32 } %2421, 0		; visa id: 3162
  %2423 = extractvalue { i32, i32 } %2421, 1		; visa id: 3162
  %2424 = insertelement <2 x i32> undef, i32 %2422, i32 0		; visa id: 3169
  %2425 = insertelement <2 x i32> %2424, i32 %2423, i32 1		; visa id: 3170
  %2426 = bitcast <2 x i32> %2425 to i64		; visa id: 3171
  %2427 = shl i64 %2426, 1		; visa id: 3175
  %2428 = add i64 %.in401, %2427		; visa id: 3176
  %2429 = ashr i64 %2412, 31		; visa id: 3177
  %2430 = bitcast i64 %2429 to <2 x i32>		; visa id: 3178
  %2431 = extractelement <2 x i32> %2430, i32 0		; visa id: 3182
  %2432 = extractelement <2 x i32> %2430, i32 1		; visa id: 3182
  %2433 = and i32 %2431, -2		; visa id: 3182
  %2434 = insertelement <2 x i32> undef, i32 %2433, i32 0		; visa id: 3183
  %2435 = insertelement <2 x i32> %2434, i32 %2432, i32 1		; visa id: 3184
  %2436 = bitcast <2 x i32> %2435 to i64		; visa id: 3185
  %2437 = add i64 %2428, %2436		; visa id: 3189
  %2438 = inttoptr i64 %2437 to i16 addrspace(4)*		; visa id: 3190
  %2439 = addrspacecast i16 addrspace(4)* %2438 to i16 addrspace(1)*		; visa id: 3190
  %2440 = load i16, i16 addrspace(1)* %2439, align 2		; visa id: 3191
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3193
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3193
  %2441 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3193
  %2442 = insertelement <2 x i32> %2441, i32 %2099, i64 1		; visa id: 3194
  %2443 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3195
  store <2 x i32> %2442, <2 x i32>* %2443, align 4, !noalias !635		; visa id: 3195
  br label %._crit_edge254, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3197

._crit_edge254:                                   ; preds = %._crit_edge254.._crit_edge254_crit_edge, %2411
; BB219 :
  %2444 = phi i32 [ 0, %2411 ], [ %2453, %._crit_edge254.._crit_edge254_crit_edge ]
  %2445 = zext i32 %2444 to i64		; visa id: 3198
  %2446 = shl nuw nsw i64 %2445, 2		; visa id: 3199
  %2447 = add i64 %124, %2446		; visa id: 3200
  %2448 = inttoptr i64 %2447 to i32*		; visa id: 3201
  %2449 = load i32, i32* %2448, align 4, !noalias !635		; visa id: 3201
  %2450 = add i64 %119, %2446		; visa id: 3202
  %2451 = inttoptr i64 %2450 to i32*		; visa id: 3203
  store i32 %2449, i32* %2451, align 4, !alias.scope !635		; visa id: 3203
  %2452 = icmp eq i32 %2444, 0		; visa id: 3204
  br i1 %2452, label %._crit_edge254.._crit_edge254_crit_edge, label %2454, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3205

._crit_edge254.._crit_edge254_crit_edge:          ; preds = %._crit_edge254
; BB220 :
  %2453 = add nuw nsw i32 %2444, 1, !spirv.Decorations !631		; visa id: 3207
  br label %._crit_edge254, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3208

2454:                                             ; preds = %._crit_edge254
; BB221 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3210
  %2455 = load i64, i64* %120, align 8		; visa id: 3210
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3211
  %2456 = bitcast i64 %2455 to <2 x i32>		; visa id: 3211
  %2457 = extractelement <2 x i32> %2456, i32 0		; visa id: 3213
  %2458 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2457, i32 1
  %2459 = bitcast <2 x i32> %2458 to i64		; visa id: 3213
  %2460 = ashr exact i64 %2459, 32		; visa id: 3214
  %2461 = bitcast i64 %2460 to <2 x i32>		; visa id: 3215
  %2462 = extractelement <2 x i32> %2461, i32 0		; visa id: 3219
  %2463 = extractelement <2 x i32> %2461, i32 1		; visa id: 3219
  %2464 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2462, i32 %2463, i32 %44, i32 %45)
  %2465 = extractvalue { i32, i32 } %2464, 0		; visa id: 3219
  %2466 = extractvalue { i32, i32 } %2464, 1		; visa id: 3219
  %2467 = insertelement <2 x i32> undef, i32 %2465, i32 0		; visa id: 3226
  %2468 = insertelement <2 x i32> %2467, i32 %2466, i32 1		; visa id: 3227
  %2469 = bitcast <2 x i32> %2468 to i64		; visa id: 3228
  %2470 = shl i64 %2469, 1		; visa id: 3232
  %2471 = add i64 %.in400, %2470		; visa id: 3233
  %2472 = ashr i64 %2455, 31		; visa id: 3234
  %2473 = bitcast i64 %2472 to <2 x i32>		; visa id: 3235
  %2474 = extractelement <2 x i32> %2473, i32 0		; visa id: 3239
  %2475 = extractelement <2 x i32> %2473, i32 1		; visa id: 3239
  %2476 = and i32 %2474, -2		; visa id: 3239
  %2477 = insertelement <2 x i32> undef, i32 %2476, i32 0		; visa id: 3240
  %2478 = insertelement <2 x i32> %2477, i32 %2475, i32 1		; visa id: 3241
  %2479 = bitcast <2 x i32> %2478 to i64		; visa id: 3242
  %2480 = add i64 %2471, %2479		; visa id: 3246
  %2481 = inttoptr i64 %2480 to i16 addrspace(4)*		; visa id: 3247
  %2482 = addrspacecast i16 addrspace(4)* %2481 to i16 addrspace(1)*		; visa id: 3247
  %2483 = load i16, i16 addrspace(1)* %2482, align 2		; visa id: 3248
  %2484 = zext i16 %2440 to i32		; visa id: 3250
  %2485 = shl nuw i32 %2484, 16, !spirv.Decorations !639		; visa id: 3251
  %2486 = bitcast i32 %2485 to float
  %2487 = zext i16 %2483 to i32		; visa id: 3252
  %2488 = shl nuw i32 %2487, 16, !spirv.Decorations !639		; visa id: 3253
  %2489 = bitcast i32 %2488 to float
  %2490 = fmul reassoc nsz arcp contract float %2486, %2489, !spirv.Decorations !618
  %2491 = fadd reassoc nsz arcp contract float %2490, %.sroa.214.1, !spirv.Decorations !618		; visa id: 3254
  br label %.preheader.5, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3255

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %2454
; BB222 :
  %.sroa.214.2 = phi float [ %2491, %2454 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %2492 = add i32 %69, 6		; visa id: 3256
  %2493 = icmp slt i32 %2492, %const_reg_dword1		; visa id: 3257
  %2494 = icmp slt i32 %65, %const_reg_dword
  %2495 = and i1 %2494, %2493		; visa id: 3258
  br i1 %2495, label %2496, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3260

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
; BB:
  br label %._crit_edge.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2496:                                             ; preds = %.preheader.5
; BB224 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3262
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3262
  %2497 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 3262
  %2498 = insertelement <2 x i32> %2497, i32 %113, i64 1		; visa id: 3263
  %2499 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3264
  store <2 x i32> %2498, <2 x i32>* %2499, align 4, !noalias !625		; visa id: 3264
  br label %._crit_edge255, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3266

._crit_edge255:                                   ; preds = %._crit_edge255.._crit_edge255_crit_edge, %2496
; BB225 :
  %2500 = phi i32 [ 0, %2496 ], [ %2509, %._crit_edge255.._crit_edge255_crit_edge ]
  %2501 = zext i32 %2500 to i64		; visa id: 3267
  %2502 = shl nuw nsw i64 %2501, 2		; visa id: 3268
  %2503 = add i64 %133, %2502		; visa id: 3269
  %2504 = inttoptr i64 %2503 to i32*		; visa id: 3270
  %2505 = load i32, i32* %2504, align 4, !noalias !625		; visa id: 3270
  %2506 = add i64 %128, %2502		; visa id: 3271
  %2507 = inttoptr i64 %2506 to i32*		; visa id: 3272
  store i32 %2505, i32* %2507, align 4, !alias.scope !625		; visa id: 3272
  %2508 = icmp eq i32 %2500, 0		; visa id: 3273
  br i1 %2508, label %._crit_edge255.._crit_edge255_crit_edge, label %2510, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3274

._crit_edge255.._crit_edge255_crit_edge:          ; preds = %._crit_edge255
; BB226 :
  %2509 = add nuw nsw i32 %2500, 1, !spirv.Decorations !631		; visa id: 3276
  br label %._crit_edge255, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3277

2510:                                             ; preds = %._crit_edge255
; BB227 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3279
  %2511 = load i64, i64* %129, align 8		; visa id: 3279
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3280
  %2512 = bitcast i64 %2511 to <2 x i32>		; visa id: 3280
  %2513 = extractelement <2 x i32> %2512, i32 0		; visa id: 3282
  %2514 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2513, i32 1
  %2515 = bitcast <2 x i32> %2514 to i64		; visa id: 3282
  %2516 = ashr exact i64 %2515, 32		; visa id: 3283
  %2517 = bitcast i64 %2516 to <2 x i32>		; visa id: 3284
  %2518 = extractelement <2 x i32> %2517, i32 0		; visa id: 3288
  %2519 = extractelement <2 x i32> %2517, i32 1		; visa id: 3288
  %2520 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2518, i32 %2519, i32 %41, i32 %42)
  %2521 = extractvalue { i32, i32 } %2520, 0		; visa id: 3288
  %2522 = extractvalue { i32, i32 } %2520, 1		; visa id: 3288
  %2523 = insertelement <2 x i32> undef, i32 %2521, i32 0		; visa id: 3295
  %2524 = insertelement <2 x i32> %2523, i32 %2522, i32 1		; visa id: 3296
  %2525 = bitcast <2 x i32> %2524 to i64		; visa id: 3297
  %2526 = shl i64 %2525, 1		; visa id: 3301
  %2527 = add i64 %.in401, %2526		; visa id: 3302
  %2528 = ashr i64 %2511, 31		; visa id: 3303
  %2529 = bitcast i64 %2528 to <2 x i32>		; visa id: 3304
  %2530 = extractelement <2 x i32> %2529, i32 0		; visa id: 3308
  %2531 = extractelement <2 x i32> %2529, i32 1		; visa id: 3308
  %2532 = and i32 %2530, -2		; visa id: 3308
  %2533 = insertelement <2 x i32> undef, i32 %2532, i32 0		; visa id: 3309
  %2534 = insertelement <2 x i32> %2533, i32 %2531, i32 1		; visa id: 3310
  %2535 = bitcast <2 x i32> %2534 to i64		; visa id: 3311
  %2536 = add i64 %2527, %2535		; visa id: 3315
  %2537 = inttoptr i64 %2536 to i16 addrspace(4)*		; visa id: 3316
  %2538 = addrspacecast i16 addrspace(4)* %2537 to i16 addrspace(1)*		; visa id: 3316
  %2539 = load i16, i16 addrspace(1)* %2538, align 2		; visa id: 3317
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3319
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3319
  %2540 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3319
  %2541 = insertelement <2 x i32> %2540, i32 %2492, i64 1		; visa id: 3320
  %2542 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3321
  store <2 x i32> %2541, <2 x i32>* %2542, align 4, !noalias !635		; visa id: 3321
  br label %._crit_edge256, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3323

._crit_edge256:                                   ; preds = %._crit_edge256.._crit_edge256_crit_edge, %2510
; BB228 :
  %2543 = phi i32 [ 0, %2510 ], [ %2552, %._crit_edge256.._crit_edge256_crit_edge ]
  %2544 = zext i32 %2543 to i64		; visa id: 3324
  %2545 = shl nuw nsw i64 %2544, 2		; visa id: 3325
  %2546 = add i64 %124, %2545		; visa id: 3326
  %2547 = inttoptr i64 %2546 to i32*		; visa id: 3327
  %2548 = load i32, i32* %2547, align 4, !noalias !635		; visa id: 3327
  %2549 = add i64 %119, %2545		; visa id: 3328
  %2550 = inttoptr i64 %2549 to i32*		; visa id: 3329
  store i32 %2548, i32* %2550, align 4, !alias.scope !635		; visa id: 3329
  %2551 = icmp eq i32 %2543, 0		; visa id: 3330
  br i1 %2551, label %._crit_edge256.._crit_edge256_crit_edge, label %2553, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3331

._crit_edge256.._crit_edge256_crit_edge:          ; preds = %._crit_edge256
; BB229 :
  %2552 = add nuw nsw i32 %2543, 1, !spirv.Decorations !631		; visa id: 3333
  br label %._crit_edge256, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3334

2553:                                             ; preds = %._crit_edge256
; BB230 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3336
  %2554 = load i64, i64* %120, align 8		; visa id: 3336
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3337
  %2555 = bitcast i64 %2554 to <2 x i32>		; visa id: 3337
  %2556 = extractelement <2 x i32> %2555, i32 0		; visa id: 3339
  %2557 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2556, i32 1
  %2558 = bitcast <2 x i32> %2557 to i64		; visa id: 3339
  %2559 = ashr exact i64 %2558, 32		; visa id: 3340
  %2560 = bitcast i64 %2559 to <2 x i32>		; visa id: 3341
  %2561 = extractelement <2 x i32> %2560, i32 0		; visa id: 3345
  %2562 = extractelement <2 x i32> %2560, i32 1		; visa id: 3345
  %2563 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2561, i32 %2562, i32 %44, i32 %45)
  %2564 = extractvalue { i32, i32 } %2563, 0		; visa id: 3345
  %2565 = extractvalue { i32, i32 } %2563, 1		; visa id: 3345
  %2566 = insertelement <2 x i32> undef, i32 %2564, i32 0		; visa id: 3352
  %2567 = insertelement <2 x i32> %2566, i32 %2565, i32 1		; visa id: 3353
  %2568 = bitcast <2 x i32> %2567 to i64		; visa id: 3354
  %2569 = shl i64 %2568, 1		; visa id: 3358
  %2570 = add i64 %.in400, %2569		; visa id: 3359
  %2571 = ashr i64 %2554, 31		; visa id: 3360
  %2572 = bitcast i64 %2571 to <2 x i32>		; visa id: 3361
  %2573 = extractelement <2 x i32> %2572, i32 0		; visa id: 3365
  %2574 = extractelement <2 x i32> %2572, i32 1		; visa id: 3365
  %2575 = and i32 %2573, -2		; visa id: 3365
  %2576 = insertelement <2 x i32> undef, i32 %2575, i32 0		; visa id: 3366
  %2577 = insertelement <2 x i32> %2576, i32 %2574, i32 1		; visa id: 3367
  %2578 = bitcast <2 x i32> %2577 to i64		; visa id: 3368
  %2579 = add i64 %2570, %2578		; visa id: 3372
  %2580 = inttoptr i64 %2579 to i16 addrspace(4)*		; visa id: 3373
  %2581 = addrspacecast i16 addrspace(4)* %2580 to i16 addrspace(1)*		; visa id: 3373
  %2582 = load i16, i16 addrspace(1)* %2581, align 2		; visa id: 3374
  %2583 = zext i16 %2539 to i32		; visa id: 3376
  %2584 = shl nuw i32 %2583, 16, !spirv.Decorations !639		; visa id: 3377
  %2585 = bitcast i32 %2584 to float
  %2586 = zext i16 %2582 to i32		; visa id: 3378
  %2587 = shl nuw i32 %2586, 16, !spirv.Decorations !639		; visa id: 3379
  %2588 = bitcast i32 %2587 to float
  %2589 = fmul reassoc nsz arcp contract float %2585, %2588, !spirv.Decorations !618
  %2590 = fadd reassoc nsz arcp contract float %2589, %.sroa.26.1, !spirv.Decorations !618		; visa id: 3380
  br label %._crit_edge.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3381

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %2553
; BB231 :
  %.sroa.26.2 = phi float [ %2590, %2553 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  %2591 = icmp slt i32 %230, %const_reg_dword
  %2592 = icmp slt i32 %2492, %const_reg_dword1		; visa id: 3382
  %2593 = and i1 %2591, %2592		; visa id: 3383
  br i1 %2593, label %2594, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3385

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
; BB:
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2594:                                             ; preds = %._crit_edge.6
; BB233 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3387
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3387
  %2595 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 3387
  %2596 = insertelement <2 x i32> %2595, i32 %113, i64 1		; visa id: 3388
  %2597 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3389
  store <2 x i32> %2596, <2 x i32>* %2597, align 4, !noalias !625		; visa id: 3389
  br label %._crit_edge257, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3391

._crit_edge257:                                   ; preds = %._crit_edge257.._crit_edge257_crit_edge, %2594
; BB234 :
  %2598 = phi i32 [ 0, %2594 ], [ %2607, %._crit_edge257.._crit_edge257_crit_edge ]
  %2599 = zext i32 %2598 to i64		; visa id: 3392
  %2600 = shl nuw nsw i64 %2599, 2		; visa id: 3393
  %2601 = add i64 %133, %2600		; visa id: 3394
  %2602 = inttoptr i64 %2601 to i32*		; visa id: 3395
  %2603 = load i32, i32* %2602, align 4, !noalias !625		; visa id: 3395
  %2604 = add i64 %128, %2600		; visa id: 3396
  %2605 = inttoptr i64 %2604 to i32*		; visa id: 3397
  store i32 %2603, i32* %2605, align 4, !alias.scope !625		; visa id: 3397
  %2606 = icmp eq i32 %2598, 0		; visa id: 3398
  br i1 %2606, label %._crit_edge257.._crit_edge257_crit_edge, label %2608, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3399

._crit_edge257.._crit_edge257_crit_edge:          ; preds = %._crit_edge257
; BB235 :
  %2607 = add nuw nsw i32 %2598, 1, !spirv.Decorations !631		; visa id: 3401
  br label %._crit_edge257, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3402

2608:                                             ; preds = %._crit_edge257
; BB236 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3404
  %2609 = load i64, i64* %129, align 8		; visa id: 3404
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3405
  %2610 = bitcast i64 %2609 to <2 x i32>		; visa id: 3405
  %2611 = extractelement <2 x i32> %2610, i32 0		; visa id: 3407
  %2612 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2611, i32 1
  %2613 = bitcast <2 x i32> %2612 to i64		; visa id: 3407
  %2614 = ashr exact i64 %2613, 32		; visa id: 3408
  %2615 = bitcast i64 %2614 to <2 x i32>		; visa id: 3409
  %2616 = extractelement <2 x i32> %2615, i32 0		; visa id: 3413
  %2617 = extractelement <2 x i32> %2615, i32 1		; visa id: 3413
  %2618 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2616, i32 %2617, i32 %41, i32 %42)
  %2619 = extractvalue { i32, i32 } %2618, 0		; visa id: 3413
  %2620 = extractvalue { i32, i32 } %2618, 1		; visa id: 3413
  %2621 = insertelement <2 x i32> undef, i32 %2619, i32 0		; visa id: 3420
  %2622 = insertelement <2 x i32> %2621, i32 %2620, i32 1		; visa id: 3421
  %2623 = bitcast <2 x i32> %2622 to i64		; visa id: 3422
  %2624 = shl i64 %2623, 1		; visa id: 3426
  %2625 = add i64 %.in401, %2624		; visa id: 3427
  %2626 = ashr i64 %2609, 31		; visa id: 3428
  %2627 = bitcast i64 %2626 to <2 x i32>		; visa id: 3429
  %2628 = extractelement <2 x i32> %2627, i32 0		; visa id: 3433
  %2629 = extractelement <2 x i32> %2627, i32 1		; visa id: 3433
  %2630 = and i32 %2628, -2		; visa id: 3433
  %2631 = insertelement <2 x i32> undef, i32 %2630, i32 0		; visa id: 3434
  %2632 = insertelement <2 x i32> %2631, i32 %2629, i32 1		; visa id: 3435
  %2633 = bitcast <2 x i32> %2632 to i64		; visa id: 3436
  %2634 = add i64 %2625, %2633		; visa id: 3440
  %2635 = inttoptr i64 %2634 to i16 addrspace(4)*		; visa id: 3441
  %2636 = addrspacecast i16 addrspace(4)* %2635 to i16 addrspace(1)*		; visa id: 3441
  %2637 = load i16, i16 addrspace(1)* %2636, align 2		; visa id: 3442
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3444
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3444
  %2638 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3444
  %2639 = insertelement <2 x i32> %2638, i32 %2492, i64 1		; visa id: 3445
  %2640 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3446
  store <2 x i32> %2639, <2 x i32>* %2640, align 4, !noalias !635		; visa id: 3446
  br label %._crit_edge258, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3448

._crit_edge258:                                   ; preds = %._crit_edge258.._crit_edge258_crit_edge, %2608
; BB237 :
  %2641 = phi i32 [ 0, %2608 ], [ %2650, %._crit_edge258.._crit_edge258_crit_edge ]
  %2642 = zext i32 %2641 to i64		; visa id: 3449
  %2643 = shl nuw nsw i64 %2642, 2		; visa id: 3450
  %2644 = add i64 %124, %2643		; visa id: 3451
  %2645 = inttoptr i64 %2644 to i32*		; visa id: 3452
  %2646 = load i32, i32* %2645, align 4, !noalias !635		; visa id: 3452
  %2647 = add i64 %119, %2643		; visa id: 3453
  %2648 = inttoptr i64 %2647 to i32*		; visa id: 3454
  store i32 %2646, i32* %2648, align 4, !alias.scope !635		; visa id: 3454
  %2649 = icmp eq i32 %2641, 0		; visa id: 3455
  br i1 %2649, label %._crit_edge258.._crit_edge258_crit_edge, label %2651, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3456

._crit_edge258.._crit_edge258_crit_edge:          ; preds = %._crit_edge258
; BB238 :
  %2650 = add nuw nsw i32 %2641, 1, !spirv.Decorations !631		; visa id: 3458
  br label %._crit_edge258, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3459

2651:                                             ; preds = %._crit_edge258
; BB239 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3461
  %2652 = load i64, i64* %120, align 8		; visa id: 3461
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3462
  %2653 = bitcast i64 %2652 to <2 x i32>		; visa id: 3462
  %2654 = extractelement <2 x i32> %2653, i32 0		; visa id: 3464
  %2655 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2654, i32 1
  %2656 = bitcast <2 x i32> %2655 to i64		; visa id: 3464
  %2657 = ashr exact i64 %2656, 32		; visa id: 3465
  %2658 = bitcast i64 %2657 to <2 x i32>		; visa id: 3466
  %2659 = extractelement <2 x i32> %2658, i32 0		; visa id: 3470
  %2660 = extractelement <2 x i32> %2658, i32 1		; visa id: 3470
  %2661 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2659, i32 %2660, i32 %44, i32 %45)
  %2662 = extractvalue { i32, i32 } %2661, 0		; visa id: 3470
  %2663 = extractvalue { i32, i32 } %2661, 1		; visa id: 3470
  %2664 = insertelement <2 x i32> undef, i32 %2662, i32 0		; visa id: 3477
  %2665 = insertelement <2 x i32> %2664, i32 %2663, i32 1		; visa id: 3478
  %2666 = bitcast <2 x i32> %2665 to i64		; visa id: 3479
  %2667 = shl i64 %2666, 1		; visa id: 3483
  %2668 = add i64 %.in400, %2667		; visa id: 3484
  %2669 = ashr i64 %2652, 31		; visa id: 3485
  %2670 = bitcast i64 %2669 to <2 x i32>		; visa id: 3486
  %2671 = extractelement <2 x i32> %2670, i32 0		; visa id: 3490
  %2672 = extractelement <2 x i32> %2670, i32 1		; visa id: 3490
  %2673 = and i32 %2671, -2		; visa id: 3490
  %2674 = insertelement <2 x i32> undef, i32 %2673, i32 0		; visa id: 3491
  %2675 = insertelement <2 x i32> %2674, i32 %2672, i32 1		; visa id: 3492
  %2676 = bitcast <2 x i32> %2675 to i64		; visa id: 3493
  %2677 = add i64 %2668, %2676		; visa id: 3497
  %2678 = inttoptr i64 %2677 to i16 addrspace(4)*		; visa id: 3498
  %2679 = addrspacecast i16 addrspace(4)* %2678 to i16 addrspace(1)*		; visa id: 3498
  %2680 = load i16, i16 addrspace(1)* %2679, align 2		; visa id: 3499
  %2681 = zext i16 %2637 to i32		; visa id: 3501
  %2682 = shl nuw i32 %2681, 16, !spirv.Decorations !639		; visa id: 3502
  %2683 = bitcast i32 %2682 to float
  %2684 = zext i16 %2680 to i32		; visa id: 3503
  %2685 = shl nuw i32 %2684, 16, !spirv.Decorations !639		; visa id: 3504
  %2686 = bitcast i32 %2685 to float
  %2687 = fmul reassoc nsz arcp contract float %2683, %2686, !spirv.Decorations !618
  %2688 = fadd reassoc nsz arcp contract float %2687, %.sroa.90.1, !spirv.Decorations !618		; visa id: 3505
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3506

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %2651
; BB240 :
  %.sroa.90.2 = phi float [ %2688, %2651 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  %2689 = icmp slt i32 %329, %const_reg_dword
  %2690 = icmp slt i32 %2492, %const_reg_dword1		; visa id: 3507
  %2691 = and i1 %2689, %2690		; visa id: 3508
  br i1 %2691, label %2692, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3510

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
; BB:
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2692:                                             ; preds = %._crit_edge.1.6
; BB242 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3512
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3512
  %2693 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 3512
  %2694 = insertelement <2 x i32> %2693, i32 %113, i64 1		; visa id: 3513
  %2695 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3514
  store <2 x i32> %2694, <2 x i32>* %2695, align 4, !noalias !625		; visa id: 3514
  br label %._crit_edge259, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3516

._crit_edge259:                                   ; preds = %._crit_edge259.._crit_edge259_crit_edge, %2692
; BB243 :
  %2696 = phi i32 [ 0, %2692 ], [ %2705, %._crit_edge259.._crit_edge259_crit_edge ]
  %2697 = zext i32 %2696 to i64		; visa id: 3517
  %2698 = shl nuw nsw i64 %2697, 2		; visa id: 3518
  %2699 = add i64 %133, %2698		; visa id: 3519
  %2700 = inttoptr i64 %2699 to i32*		; visa id: 3520
  %2701 = load i32, i32* %2700, align 4, !noalias !625		; visa id: 3520
  %2702 = add i64 %128, %2698		; visa id: 3521
  %2703 = inttoptr i64 %2702 to i32*		; visa id: 3522
  store i32 %2701, i32* %2703, align 4, !alias.scope !625		; visa id: 3522
  %2704 = icmp eq i32 %2696, 0		; visa id: 3523
  br i1 %2704, label %._crit_edge259.._crit_edge259_crit_edge, label %2706, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3524

._crit_edge259.._crit_edge259_crit_edge:          ; preds = %._crit_edge259
; BB244 :
  %2705 = add nuw nsw i32 %2696, 1, !spirv.Decorations !631		; visa id: 3526
  br label %._crit_edge259, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3527

2706:                                             ; preds = %._crit_edge259
; BB245 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3529
  %2707 = load i64, i64* %129, align 8		; visa id: 3529
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3530
  %2708 = bitcast i64 %2707 to <2 x i32>		; visa id: 3530
  %2709 = extractelement <2 x i32> %2708, i32 0		; visa id: 3532
  %2710 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2709, i32 1
  %2711 = bitcast <2 x i32> %2710 to i64		; visa id: 3532
  %2712 = ashr exact i64 %2711, 32		; visa id: 3533
  %2713 = bitcast i64 %2712 to <2 x i32>		; visa id: 3534
  %2714 = extractelement <2 x i32> %2713, i32 0		; visa id: 3538
  %2715 = extractelement <2 x i32> %2713, i32 1		; visa id: 3538
  %2716 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2714, i32 %2715, i32 %41, i32 %42)
  %2717 = extractvalue { i32, i32 } %2716, 0		; visa id: 3538
  %2718 = extractvalue { i32, i32 } %2716, 1		; visa id: 3538
  %2719 = insertelement <2 x i32> undef, i32 %2717, i32 0		; visa id: 3545
  %2720 = insertelement <2 x i32> %2719, i32 %2718, i32 1		; visa id: 3546
  %2721 = bitcast <2 x i32> %2720 to i64		; visa id: 3547
  %2722 = shl i64 %2721, 1		; visa id: 3551
  %2723 = add i64 %.in401, %2722		; visa id: 3552
  %2724 = ashr i64 %2707, 31		; visa id: 3553
  %2725 = bitcast i64 %2724 to <2 x i32>		; visa id: 3554
  %2726 = extractelement <2 x i32> %2725, i32 0		; visa id: 3558
  %2727 = extractelement <2 x i32> %2725, i32 1		; visa id: 3558
  %2728 = and i32 %2726, -2		; visa id: 3558
  %2729 = insertelement <2 x i32> undef, i32 %2728, i32 0		; visa id: 3559
  %2730 = insertelement <2 x i32> %2729, i32 %2727, i32 1		; visa id: 3560
  %2731 = bitcast <2 x i32> %2730 to i64		; visa id: 3561
  %2732 = add i64 %2723, %2731		; visa id: 3565
  %2733 = inttoptr i64 %2732 to i16 addrspace(4)*		; visa id: 3566
  %2734 = addrspacecast i16 addrspace(4)* %2733 to i16 addrspace(1)*		; visa id: 3566
  %2735 = load i16, i16 addrspace(1)* %2734, align 2		; visa id: 3567
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3569
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3569
  %2736 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3569
  %2737 = insertelement <2 x i32> %2736, i32 %2492, i64 1		; visa id: 3570
  %2738 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3571
  store <2 x i32> %2737, <2 x i32>* %2738, align 4, !noalias !635		; visa id: 3571
  br label %._crit_edge260, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3573

._crit_edge260:                                   ; preds = %._crit_edge260.._crit_edge260_crit_edge, %2706
; BB246 :
  %2739 = phi i32 [ 0, %2706 ], [ %2748, %._crit_edge260.._crit_edge260_crit_edge ]
  %2740 = zext i32 %2739 to i64		; visa id: 3574
  %2741 = shl nuw nsw i64 %2740, 2		; visa id: 3575
  %2742 = add i64 %124, %2741		; visa id: 3576
  %2743 = inttoptr i64 %2742 to i32*		; visa id: 3577
  %2744 = load i32, i32* %2743, align 4, !noalias !635		; visa id: 3577
  %2745 = add i64 %119, %2741		; visa id: 3578
  %2746 = inttoptr i64 %2745 to i32*		; visa id: 3579
  store i32 %2744, i32* %2746, align 4, !alias.scope !635		; visa id: 3579
  %2747 = icmp eq i32 %2739, 0		; visa id: 3580
  br i1 %2747, label %._crit_edge260.._crit_edge260_crit_edge, label %2749, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3581

._crit_edge260.._crit_edge260_crit_edge:          ; preds = %._crit_edge260
; BB247 :
  %2748 = add nuw nsw i32 %2739, 1, !spirv.Decorations !631		; visa id: 3583
  br label %._crit_edge260, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3584

2749:                                             ; preds = %._crit_edge260
; BB248 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3586
  %2750 = load i64, i64* %120, align 8		; visa id: 3586
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3587
  %2751 = bitcast i64 %2750 to <2 x i32>		; visa id: 3587
  %2752 = extractelement <2 x i32> %2751, i32 0		; visa id: 3589
  %2753 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2752, i32 1
  %2754 = bitcast <2 x i32> %2753 to i64		; visa id: 3589
  %2755 = ashr exact i64 %2754, 32		; visa id: 3590
  %2756 = bitcast i64 %2755 to <2 x i32>		; visa id: 3591
  %2757 = extractelement <2 x i32> %2756, i32 0		; visa id: 3595
  %2758 = extractelement <2 x i32> %2756, i32 1		; visa id: 3595
  %2759 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2757, i32 %2758, i32 %44, i32 %45)
  %2760 = extractvalue { i32, i32 } %2759, 0		; visa id: 3595
  %2761 = extractvalue { i32, i32 } %2759, 1		; visa id: 3595
  %2762 = insertelement <2 x i32> undef, i32 %2760, i32 0		; visa id: 3602
  %2763 = insertelement <2 x i32> %2762, i32 %2761, i32 1		; visa id: 3603
  %2764 = bitcast <2 x i32> %2763 to i64		; visa id: 3604
  %2765 = shl i64 %2764, 1		; visa id: 3608
  %2766 = add i64 %.in400, %2765		; visa id: 3609
  %2767 = ashr i64 %2750, 31		; visa id: 3610
  %2768 = bitcast i64 %2767 to <2 x i32>		; visa id: 3611
  %2769 = extractelement <2 x i32> %2768, i32 0		; visa id: 3615
  %2770 = extractelement <2 x i32> %2768, i32 1		; visa id: 3615
  %2771 = and i32 %2769, -2		; visa id: 3615
  %2772 = insertelement <2 x i32> undef, i32 %2771, i32 0		; visa id: 3616
  %2773 = insertelement <2 x i32> %2772, i32 %2770, i32 1		; visa id: 3617
  %2774 = bitcast <2 x i32> %2773 to i64		; visa id: 3618
  %2775 = add i64 %2766, %2774		; visa id: 3622
  %2776 = inttoptr i64 %2775 to i16 addrspace(4)*		; visa id: 3623
  %2777 = addrspacecast i16 addrspace(4)* %2776 to i16 addrspace(1)*		; visa id: 3623
  %2778 = load i16, i16 addrspace(1)* %2777, align 2		; visa id: 3624
  %2779 = zext i16 %2735 to i32		; visa id: 3626
  %2780 = shl nuw i32 %2779, 16, !spirv.Decorations !639		; visa id: 3627
  %2781 = bitcast i32 %2780 to float
  %2782 = zext i16 %2778 to i32		; visa id: 3628
  %2783 = shl nuw i32 %2782, 16, !spirv.Decorations !639		; visa id: 3629
  %2784 = bitcast i32 %2783 to float
  %2785 = fmul reassoc nsz arcp contract float %2781, %2784, !spirv.Decorations !618
  %2786 = fadd reassoc nsz arcp contract float %2785, %.sroa.154.1, !spirv.Decorations !618		; visa id: 3630
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3631

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %2749
; BB249 :
  %.sroa.154.2 = phi float [ %2786, %2749 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  %2787 = icmp slt i32 %428, %const_reg_dword
  %2788 = icmp slt i32 %2492, %const_reg_dword1		; visa id: 3632
  %2789 = and i1 %2787, %2788		; visa id: 3633
  br i1 %2789, label %2790, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3635

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
; BB:
  br label %.preheader.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2790:                                             ; preds = %._crit_edge.2.6
; BB251 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3637
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3637
  %2791 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 3637
  %2792 = insertelement <2 x i32> %2791, i32 %113, i64 1		; visa id: 3638
  %2793 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3639
  store <2 x i32> %2792, <2 x i32>* %2793, align 4, !noalias !625		; visa id: 3639
  br label %._crit_edge261, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3641

._crit_edge261:                                   ; preds = %._crit_edge261.._crit_edge261_crit_edge, %2790
; BB252 :
  %2794 = phi i32 [ 0, %2790 ], [ %2803, %._crit_edge261.._crit_edge261_crit_edge ]
  %2795 = zext i32 %2794 to i64		; visa id: 3642
  %2796 = shl nuw nsw i64 %2795, 2		; visa id: 3643
  %2797 = add i64 %133, %2796		; visa id: 3644
  %2798 = inttoptr i64 %2797 to i32*		; visa id: 3645
  %2799 = load i32, i32* %2798, align 4, !noalias !625		; visa id: 3645
  %2800 = add i64 %128, %2796		; visa id: 3646
  %2801 = inttoptr i64 %2800 to i32*		; visa id: 3647
  store i32 %2799, i32* %2801, align 4, !alias.scope !625		; visa id: 3647
  %2802 = icmp eq i32 %2794, 0		; visa id: 3648
  br i1 %2802, label %._crit_edge261.._crit_edge261_crit_edge, label %2804, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3649

._crit_edge261.._crit_edge261_crit_edge:          ; preds = %._crit_edge261
; BB253 :
  %2803 = add nuw nsw i32 %2794, 1, !spirv.Decorations !631		; visa id: 3651
  br label %._crit_edge261, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3652

2804:                                             ; preds = %._crit_edge261
; BB254 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3654
  %2805 = load i64, i64* %129, align 8		; visa id: 3654
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3655
  %2806 = bitcast i64 %2805 to <2 x i32>		; visa id: 3655
  %2807 = extractelement <2 x i32> %2806, i32 0		; visa id: 3657
  %2808 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2807, i32 1
  %2809 = bitcast <2 x i32> %2808 to i64		; visa id: 3657
  %2810 = ashr exact i64 %2809, 32		; visa id: 3658
  %2811 = bitcast i64 %2810 to <2 x i32>		; visa id: 3659
  %2812 = extractelement <2 x i32> %2811, i32 0		; visa id: 3663
  %2813 = extractelement <2 x i32> %2811, i32 1		; visa id: 3663
  %2814 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2812, i32 %2813, i32 %41, i32 %42)
  %2815 = extractvalue { i32, i32 } %2814, 0		; visa id: 3663
  %2816 = extractvalue { i32, i32 } %2814, 1		; visa id: 3663
  %2817 = insertelement <2 x i32> undef, i32 %2815, i32 0		; visa id: 3670
  %2818 = insertelement <2 x i32> %2817, i32 %2816, i32 1		; visa id: 3671
  %2819 = bitcast <2 x i32> %2818 to i64		; visa id: 3672
  %2820 = shl i64 %2819, 1		; visa id: 3676
  %2821 = add i64 %.in401, %2820		; visa id: 3677
  %2822 = ashr i64 %2805, 31		; visa id: 3678
  %2823 = bitcast i64 %2822 to <2 x i32>		; visa id: 3679
  %2824 = extractelement <2 x i32> %2823, i32 0		; visa id: 3683
  %2825 = extractelement <2 x i32> %2823, i32 1		; visa id: 3683
  %2826 = and i32 %2824, -2		; visa id: 3683
  %2827 = insertelement <2 x i32> undef, i32 %2826, i32 0		; visa id: 3684
  %2828 = insertelement <2 x i32> %2827, i32 %2825, i32 1		; visa id: 3685
  %2829 = bitcast <2 x i32> %2828 to i64		; visa id: 3686
  %2830 = add i64 %2821, %2829		; visa id: 3690
  %2831 = inttoptr i64 %2830 to i16 addrspace(4)*		; visa id: 3691
  %2832 = addrspacecast i16 addrspace(4)* %2831 to i16 addrspace(1)*		; visa id: 3691
  %2833 = load i16, i16 addrspace(1)* %2832, align 2		; visa id: 3692
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3694
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3694
  %2834 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3694
  %2835 = insertelement <2 x i32> %2834, i32 %2492, i64 1		; visa id: 3695
  %2836 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3696
  store <2 x i32> %2835, <2 x i32>* %2836, align 4, !noalias !635		; visa id: 3696
  br label %._crit_edge262, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3698

._crit_edge262:                                   ; preds = %._crit_edge262.._crit_edge262_crit_edge, %2804
; BB255 :
  %2837 = phi i32 [ 0, %2804 ], [ %2846, %._crit_edge262.._crit_edge262_crit_edge ]
  %2838 = zext i32 %2837 to i64		; visa id: 3699
  %2839 = shl nuw nsw i64 %2838, 2		; visa id: 3700
  %2840 = add i64 %124, %2839		; visa id: 3701
  %2841 = inttoptr i64 %2840 to i32*		; visa id: 3702
  %2842 = load i32, i32* %2841, align 4, !noalias !635		; visa id: 3702
  %2843 = add i64 %119, %2839		; visa id: 3703
  %2844 = inttoptr i64 %2843 to i32*		; visa id: 3704
  store i32 %2842, i32* %2844, align 4, !alias.scope !635		; visa id: 3704
  %2845 = icmp eq i32 %2837, 0		; visa id: 3705
  br i1 %2845, label %._crit_edge262.._crit_edge262_crit_edge, label %2847, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3706

._crit_edge262.._crit_edge262_crit_edge:          ; preds = %._crit_edge262
; BB256 :
  %2846 = add nuw nsw i32 %2837, 1, !spirv.Decorations !631		; visa id: 3708
  br label %._crit_edge262, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3709

2847:                                             ; preds = %._crit_edge262
; BB257 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3711
  %2848 = load i64, i64* %120, align 8		; visa id: 3711
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3712
  %2849 = bitcast i64 %2848 to <2 x i32>		; visa id: 3712
  %2850 = extractelement <2 x i32> %2849, i32 0		; visa id: 3714
  %2851 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2850, i32 1
  %2852 = bitcast <2 x i32> %2851 to i64		; visa id: 3714
  %2853 = ashr exact i64 %2852, 32		; visa id: 3715
  %2854 = bitcast i64 %2853 to <2 x i32>		; visa id: 3716
  %2855 = extractelement <2 x i32> %2854, i32 0		; visa id: 3720
  %2856 = extractelement <2 x i32> %2854, i32 1		; visa id: 3720
  %2857 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2855, i32 %2856, i32 %44, i32 %45)
  %2858 = extractvalue { i32, i32 } %2857, 0		; visa id: 3720
  %2859 = extractvalue { i32, i32 } %2857, 1		; visa id: 3720
  %2860 = insertelement <2 x i32> undef, i32 %2858, i32 0		; visa id: 3727
  %2861 = insertelement <2 x i32> %2860, i32 %2859, i32 1		; visa id: 3728
  %2862 = bitcast <2 x i32> %2861 to i64		; visa id: 3729
  %2863 = shl i64 %2862, 1		; visa id: 3733
  %2864 = add i64 %.in400, %2863		; visa id: 3734
  %2865 = ashr i64 %2848, 31		; visa id: 3735
  %2866 = bitcast i64 %2865 to <2 x i32>		; visa id: 3736
  %2867 = extractelement <2 x i32> %2866, i32 0		; visa id: 3740
  %2868 = extractelement <2 x i32> %2866, i32 1		; visa id: 3740
  %2869 = and i32 %2867, -2		; visa id: 3740
  %2870 = insertelement <2 x i32> undef, i32 %2869, i32 0		; visa id: 3741
  %2871 = insertelement <2 x i32> %2870, i32 %2868, i32 1		; visa id: 3742
  %2872 = bitcast <2 x i32> %2871 to i64		; visa id: 3743
  %2873 = add i64 %2864, %2872		; visa id: 3747
  %2874 = inttoptr i64 %2873 to i16 addrspace(4)*		; visa id: 3748
  %2875 = addrspacecast i16 addrspace(4)* %2874 to i16 addrspace(1)*		; visa id: 3748
  %2876 = load i16, i16 addrspace(1)* %2875, align 2		; visa id: 3749
  %2877 = zext i16 %2833 to i32		; visa id: 3751
  %2878 = shl nuw i32 %2877, 16, !spirv.Decorations !639		; visa id: 3752
  %2879 = bitcast i32 %2878 to float
  %2880 = zext i16 %2876 to i32		; visa id: 3753
  %2881 = shl nuw i32 %2880, 16, !spirv.Decorations !639		; visa id: 3754
  %2882 = bitcast i32 %2881 to float
  %2883 = fmul reassoc nsz arcp contract float %2879, %2882, !spirv.Decorations !618
  %2884 = fadd reassoc nsz arcp contract float %2883, %.sroa.218.1, !spirv.Decorations !618		; visa id: 3755
  br label %.preheader.6, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3756

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %2847
; BB258 :
  %.sroa.218.2 = phi float [ %2884, %2847 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %2885 = add i32 %69, 7		; visa id: 3757
  %2886 = icmp slt i32 %2885, %const_reg_dword1		; visa id: 3758
  %2887 = icmp slt i32 %65, %const_reg_dword
  %2888 = and i1 %2887, %2886		; visa id: 3759
  br i1 %2888, label %2889, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3761

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
; BB:
  br label %._crit_edge.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2889:                                             ; preds = %.preheader.6
; BB260 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3763
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3763
  %2890 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 3763
  %2891 = insertelement <2 x i32> %2890, i32 %113, i64 1		; visa id: 3764
  %2892 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3765
  store <2 x i32> %2891, <2 x i32>* %2892, align 4, !noalias !625		; visa id: 3765
  br label %._crit_edge263, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3767

._crit_edge263:                                   ; preds = %._crit_edge263.._crit_edge263_crit_edge, %2889
; BB261 :
  %2893 = phi i32 [ 0, %2889 ], [ %2902, %._crit_edge263.._crit_edge263_crit_edge ]
  %2894 = zext i32 %2893 to i64		; visa id: 3768
  %2895 = shl nuw nsw i64 %2894, 2		; visa id: 3769
  %2896 = add i64 %133, %2895		; visa id: 3770
  %2897 = inttoptr i64 %2896 to i32*		; visa id: 3771
  %2898 = load i32, i32* %2897, align 4, !noalias !625		; visa id: 3771
  %2899 = add i64 %128, %2895		; visa id: 3772
  %2900 = inttoptr i64 %2899 to i32*		; visa id: 3773
  store i32 %2898, i32* %2900, align 4, !alias.scope !625		; visa id: 3773
  %2901 = icmp eq i32 %2893, 0		; visa id: 3774
  br i1 %2901, label %._crit_edge263.._crit_edge263_crit_edge, label %2903, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3775

._crit_edge263.._crit_edge263_crit_edge:          ; preds = %._crit_edge263
; BB262 :
  %2902 = add nuw nsw i32 %2893, 1, !spirv.Decorations !631		; visa id: 3777
  br label %._crit_edge263, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3778

2903:                                             ; preds = %._crit_edge263
; BB263 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3780
  %2904 = load i64, i64* %129, align 8		; visa id: 3780
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3781
  %2905 = bitcast i64 %2904 to <2 x i32>		; visa id: 3781
  %2906 = extractelement <2 x i32> %2905, i32 0		; visa id: 3783
  %2907 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2906, i32 1
  %2908 = bitcast <2 x i32> %2907 to i64		; visa id: 3783
  %2909 = ashr exact i64 %2908, 32		; visa id: 3784
  %2910 = bitcast i64 %2909 to <2 x i32>		; visa id: 3785
  %2911 = extractelement <2 x i32> %2910, i32 0		; visa id: 3789
  %2912 = extractelement <2 x i32> %2910, i32 1		; visa id: 3789
  %2913 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2911, i32 %2912, i32 %41, i32 %42)
  %2914 = extractvalue { i32, i32 } %2913, 0		; visa id: 3789
  %2915 = extractvalue { i32, i32 } %2913, 1		; visa id: 3789
  %2916 = insertelement <2 x i32> undef, i32 %2914, i32 0		; visa id: 3796
  %2917 = insertelement <2 x i32> %2916, i32 %2915, i32 1		; visa id: 3797
  %2918 = bitcast <2 x i32> %2917 to i64		; visa id: 3798
  %2919 = shl i64 %2918, 1		; visa id: 3802
  %2920 = add i64 %.in401, %2919		; visa id: 3803
  %2921 = ashr i64 %2904, 31		; visa id: 3804
  %2922 = bitcast i64 %2921 to <2 x i32>		; visa id: 3805
  %2923 = extractelement <2 x i32> %2922, i32 0		; visa id: 3809
  %2924 = extractelement <2 x i32> %2922, i32 1		; visa id: 3809
  %2925 = and i32 %2923, -2		; visa id: 3809
  %2926 = insertelement <2 x i32> undef, i32 %2925, i32 0		; visa id: 3810
  %2927 = insertelement <2 x i32> %2926, i32 %2924, i32 1		; visa id: 3811
  %2928 = bitcast <2 x i32> %2927 to i64		; visa id: 3812
  %2929 = add i64 %2920, %2928		; visa id: 3816
  %2930 = inttoptr i64 %2929 to i16 addrspace(4)*		; visa id: 3817
  %2931 = addrspacecast i16 addrspace(4)* %2930 to i16 addrspace(1)*		; visa id: 3817
  %2932 = load i16, i16 addrspace(1)* %2931, align 2		; visa id: 3818
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3820
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3820
  %2933 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3820
  %2934 = insertelement <2 x i32> %2933, i32 %2885, i64 1		; visa id: 3821
  %2935 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3822
  store <2 x i32> %2934, <2 x i32>* %2935, align 4, !noalias !635		; visa id: 3822
  br label %._crit_edge264, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3824

._crit_edge264:                                   ; preds = %._crit_edge264.._crit_edge264_crit_edge, %2903
; BB264 :
  %2936 = phi i32 [ 0, %2903 ], [ %2945, %._crit_edge264.._crit_edge264_crit_edge ]
  %2937 = zext i32 %2936 to i64		; visa id: 3825
  %2938 = shl nuw nsw i64 %2937, 2		; visa id: 3826
  %2939 = add i64 %124, %2938		; visa id: 3827
  %2940 = inttoptr i64 %2939 to i32*		; visa id: 3828
  %2941 = load i32, i32* %2940, align 4, !noalias !635		; visa id: 3828
  %2942 = add i64 %119, %2938		; visa id: 3829
  %2943 = inttoptr i64 %2942 to i32*		; visa id: 3830
  store i32 %2941, i32* %2943, align 4, !alias.scope !635		; visa id: 3830
  %2944 = icmp eq i32 %2936, 0		; visa id: 3831
  br i1 %2944, label %._crit_edge264.._crit_edge264_crit_edge, label %2946, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3832

._crit_edge264.._crit_edge264_crit_edge:          ; preds = %._crit_edge264
; BB265 :
  %2945 = add nuw nsw i32 %2936, 1, !spirv.Decorations !631		; visa id: 3834
  br label %._crit_edge264, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3835

2946:                                             ; preds = %._crit_edge264
; BB266 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3837
  %2947 = load i64, i64* %120, align 8		; visa id: 3837
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3838
  %2948 = bitcast i64 %2947 to <2 x i32>		; visa id: 3838
  %2949 = extractelement <2 x i32> %2948, i32 0		; visa id: 3840
  %2950 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %2949, i32 1
  %2951 = bitcast <2 x i32> %2950 to i64		; visa id: 3840
  %2952 = ashr exact i64 %2951, 32		; visa id: 3841
  %2953 = bitcast i64 %2952 to <2 x i32>		; visa id: 3842
  %2954 = extractelement <2 x i32> %2953, i32 0		; visa id: 3846
  %2955 = extractelement <2 x i32> %2953, i32 1		; visa id: 3846
  %2956 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %2954, i32 %2955, i32 %44, i32 %45)
  %2957 = extractvalue { i32, i32 } %2956, 0		; visa id: 3846
  %2958 = extractvalue { i32, i32 } %2956, 1		; visa id: 3846
  %2959 = insertelement <2 x i32> undef, i32 %2957, i32 0		; visa id: 3853
  %2960 = insertelement <2 x i32> %2959, i32 %2958, i32 1		; visa id: 3854
  %2961 = bitcast <2 x i32> %2960 to i64		; visa id: 3855
  %2962 = shl i64 %2961, 1		; visa id: 3859
  %2963 = add i64 %.in400, %2962		; visa id: 3860
  %2964 = ashr i64 %2947, 31		; visa id: 3861
  %2965 = bitcast i64 %2964 to <2 x i32>		; visa id: 3862
  %2966 = extractelement <2 x i32> %2965, i32 0		; visa id: 3866
  %2967 = extractelement <2 x i32> %2965, i32 1		; visa id: 3866
  %2968 = and i32 %2966, -2		; visa id: 3866
  %2969 = insertelement <2 x i32> undef, i32 %2968, i32 0		; visa id: 3867
  %2970 = insertelement <2 x i32> %2969, i32 %2967, i32 1		; visa id: 3868
  %2971 = bitcast <2 x i32> %2970 to i64		; visa id: 3869
  %2972 = add i64 %2963, %2971		; visa id: 3873
  %2973 = inttoptr i64 %2972 to i16 addrspace(4)*		; visa id: 3874
  %2974 = addrspacecast i16 addrspace(4)* %2973 to i16 addrspace(1)*		; visa id: 3874
  %2975 = load i16, i16 addrspace(1)* %2974, align 2		; visa id: 3875
  %2976 = zext i16 %2932 to i32		; visa id: 3877
  %2977 = shl nuw i32 %2976, 16, !spirv.Decorations !639		; visa id: 3878
  %2978 = bitcast i32 %2977 to float
  %2979 = zext i16 %2975 to i32		; visa id: 3879
  %2980 = shl nuw i32 %2979, 16, !spirv.Decorations !639		; visa id: 3880
  %2981 = bitcast i32 %2980 to float
  %2982 = fmul reassoc nsz arcp contract float %2978, %2981, !spirv.Decorations !618
  %2983 = fadd reassoc nsz arcp contract float %2982, %.sroa.30.1, !spirv.Decorations !618		; visa id: 3881
  br label %._crit_edge.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3882

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %2946
; BB267 :
  %.sroa.30.2 = phi float [ %2983, %2946 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  %2984 = icmp slt i32 %230, %const_reg_dword
  %2985 = icmp slt i32 %2885, %const_reg_dword1		; visa id: 3883
  %2986 = and i1 %2984, %2985		; visa id: 3884
  br i1 %2986, label %2987, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 3886

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
; BB:
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

2987:                                             ; preds = %._crit_edge.7
; BB269 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 3888
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 3888
  %2988 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 3888
  %2989 = insertelement <2 x i32> %2988, i32 %113, i64 1		; visa id: 3889
  %2990 = inttoptr i64 %133 to <2 x i32>*		; visa id: 3890
  store <2 x i32> %2989, <2 x i32>* %2990, align 4, !noalias !625		; visa id: 3890
  br label %._crit_edge265, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3892

._crit_edge265:                                   ; preds = %._crit_edge265.._crit_edge265_crit_edge, %2987
; BB270 :
  %2991 = phi i32 [ 0, %2987 ], [ %3000, %._crit_edge265.._crit_edge265_crit_edge ]
  %2992 = zext i32 %2991 to i64		; visa id: 3893
  %2993 = shl nuw nsw i64 %2992, 2		; visa id: 3894
  %2994 = add i64 %133, %2993		; visa id: 3895
  %2995 = inttoptr i64 %2994 to i32*		; visa id: 3896
  %2996 = load i32, i32* %2995, align 4, !noalias !625		; visa id: 3896
  %2997 = add i64 %128, %2993		; visa id: 3897
  %2998 = inttoptr i64 %2997 to i32*		; visa id: 3898
  store i32 %2996, i32* %2998, align 4, !alias.scope !625		; visa id: 3898
  %2999 = icmp eq i32 %2991, 0		; visa id: 3899
  br i1 %2999, label %._crit_edge265.._crit_edge265_crit_edge, label %3001, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3900

._crit_edge265.._crit_edge265_crit_edge:          ; preds = %._crit_edge265
; BB271 :
  %3000 = add nuw nsw i32 %2991, 1, !spirv.Decorations !631		; visa id: 3902
  br label %._crit_edge265, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3903

3001:                                             ; preds = %._crit_edge265
; BB272 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 3905
  %3002 = load i64, i64* %129, align 8		; visa id: 3905
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 3906
  %3003 = bitcast i64 %3002 to <2 x i32>		; visa id: 3906
  %3004 = extractelement <2 x i32> %3003, i32 0		; visa id: 3908
  %3005 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3004, i32 1
  %3006 = bitcast <2 x i32> %3005 to i64		; visa id: 3908
  %3007 = ashr exact i64 %3006, 32		; visa id: 3909
  %3008 = bitcast i64 %3007 to <2 x i32>		; visa id: 3910
  %3009 = extractelement <2 x i32> %3008, i32 0		; visa id: 3914
  %3010 = extractelement <2 x i32> %3008, i32 1		; visa id: 3914
  %3011 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3009, i32 %3010, i32 %41, i32 %42)
  %3012 = extractvalue { i32, i32 } %3011, 0		; visa id: 3914
  %3013 = extractvalue { i32, i32 } %3011, 1		; visa id: 3914
  %3014 = insertelement <2 x i32> undef, i32 %3012, i32 0		; visa id: 3921
  %3015 = insertelement <2 x i32> %3014, i32 %3013, i32 1		; visa id: 3922
  %3016 = bitcast <2 x i32> %3015 to i64		; visa id: 3923
  %3017 = shl i64 %3016, 1		; visa id: 3927
  %3018 = add i64 %.in401, %3017		; visa id: 3928
  %3019 = ashr i64 %3002, 31		; visa id: 3929
  %3020 = bitcast i64 %3019 to <2 x i32>		; visa id: 3930
  %3021 = extractelement <2 x i32> %3020, i32 0		; visa id: 3934
  %3022 = extractelement <2 x i32> %3020, i32 1		; visa id: 3934
  %3023 = and i32 %3021, -2		; visa id: 3934
  %3024 = insertelement <2 x i32> undef, i32 %3023, i32 0		; visa id: 3935
  %3025 = insertelement <2 x i32> %3024, i32 %3022, i32 1		; visa id: 3936
  %3026 = bitcast <2 x i32> %3025 to i64		; visa id: 3937
  %3027 = add i64 %3018, %3026		; visa id: 3941
  %3028 = inttoptr i64 %3027 to i16 addrspace(4)*		; visa id: 3942
  %3029 = addrspacecast i16 addrspace(4)* %3028 to i16 addrspace(1)*		; visa id: 3942
  %3030 = load i16, i16 addrspace(1)* %3029, align 2		; visa id: 3943
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 3945
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 3945
  %3031 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 3945
  %3032 = insertelement <2 x i32> %3031, i32 %2885, i64 1		; visa id: 3946
  %3033 = inttoptr i64 %124 to <2 x i32>*		; visa id: 3947
  store <2 x i32> %3032, <2 x i32>* %3033, align 4, !noalias !635		; visa id: 3947
  br label %._crit_edge266, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 3949

._crit_edge266:                                   ; preds = %._crit_edge266.._crit_edge266_crit_edge, %3001
; BB273 :
  %3034 = phi i32 [ 0, %3001 ], [ %3043, %._crit_edge266.._crit_edge266_crit_edge ]
  %3035 = zext i32 %3034 to i64		; visa id: 3950
  %3036 = shl nuw nsw i64 %3035, 2		; visa id: 3951
  %3037 = add i64 %124, %3036		; visa id: 3952
  %3038 = inttoptr i64 %3037 to i32*		; visa id: 3953
  %3039 = load i32, i32* %3038, align 4, !noalias !635		; visa id: 3953
  %3040 = add i64 %119, %3036		; visa id: 3954
  %3041 = inttoptr i64 %3040 to i32*		; visa id: 3955
  store i32 %3039, i32* %3041, align 4, !alias.scope !635		; visa id: 3955
  %3042 = icmp eq i32 %3034, 0		; visa id: 3956
  br i1 %3042, label %._crit_edge266.._crit_edge266_crit_edge, label %3044, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 3957

._crit_edge266.._crit_edge266_crit_edge:          ; preds = %._crit_edge266
; BB274 :
  %3043 = add nuw nsw i32 %3034, 1, !spirv.Decorations !631		; visa id: 3959
  br label %._crit_edge266, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 3960

3044:                                             ; preds = %._crit_edge266
; BB275 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 3962
  %3045 = load i64, i64* %120, align 8		; visa id: 3962
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 3963
  %3046 = bitcast i64 %3045 to <2 x i32>		; visa id: 3963
  %3047 = extractelement <2 x i32> %3046, i32 0		; visa id: 3965
  %3048 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3047, i32 1
  %3049 = bitcast <2 x i32> %3048 to i64		; visa id: 3965
  %3050 = ashr exact i64 %3049, 32		; visa id: 3966
  %3051 = bitcast i64 %3050 to <2 x i32>		; visa id: 3967
  %3052 = extractelement <2 x i32> %3051, i32 0		; visa id: 3971
  %3053 = extractelement <2 x i32> %3051, i32 1		; visa id: 3971
  %3054 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3052, i32 %3053, i32 %44, i32 %45)
  %3055 = extractvalue { i32, i32 } %3054, 0		; visa id: 3971
  %3056 = extractvalue { i32, i32 } %3054, 1		; visa id: 3971
  %3057 = insertelement <2 x i32> undef, i32 %3055, i32 0		; visa id: 3978
  %3058 = insertelement <2 x i32> %3057, i32 %3056, i32 1		; visa id: 3979
  %3059 = bitcast <2 x i32> %3058 to i64		; visa id: 3980
  %3060 = shl i64 %3059, 1		; visa id: 3984
  %3061 = add i64 %.in400, %3060		; visa id: 3985
  %3062 = ashr i64 %3045, 31		; visa id: 3986
  %3063 = bitcast i64 %3062 to <2 x i32>		; visa id: 3987
  %3064 = extractelement <2 x i32> %3063, i32 0		; visa id: 3991
  %3065 = extractelement <2 x i32> %3063, i32 1		; visa id: 3991
  %3066 = and i32 %3064, -2		; visa id: 3991
  %3067 = insertelement <2 x i32> undef, i32 %3066, i32 0		; visa id: 3992
  %3068 = insertelement <2 x i32> %3067, i32 %3065, i32 1		; visa id: 3993
  %3069 = bitcast <2 x i32> %3068 to i64		; visa id: 3994
  %3070 = add i64 %3061, %3069		; visa id: 3998
  %3071 = inttoptr i64 %3070 to i16 addrspace(4)*		; visa id: 3999
  %3072 = addrspacecast i16 addrspace(4)* %3071 to i16 addrspace(1)*		; visa id: 3999
  %3073 = load i16, i16 addrspace(1)* %3072, align 2		; visa id: 4000
  %3074 = zext i16 %3030 to i32		; visa id: 4002
  %3075 = shl nuw i32 %3074, 16, !spirv.Decorations !639		; visa id: 4003
  %3076 = bitcast i32 %3075 to float
  %3077 = zext i16 %3073 to i32		; visa id: 4004
  %3078 = shl nuw i32 %3077, 16, !spirv.Decorations !639		; visa id: 4005
  %3079 = bitcast i32 %3078 to float
  %3080 = fmul reassoc nsz arcp contract float %3076, %3079, !spirv.Decorations !618
  %3081 = fadd reassoc nsz arcp contract float %3080, %.sroa.94.1, !spirv.Decorations !618		; visa id: 4006
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4007

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %3044
; BB276 :
  %.sroa.94.2 = phi float [ %3081, %3044 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  %3082 = icmp slt i32 %329, %const_reg_dword
  %3083 = icmp slt i32 %2885, %const_reg_dword1		; visa id: 4008
  %3084 = and i1 %3082, %3083		; visa id: 4009
  br i1 %3084, label %3085, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4011

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
; BB:
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3085:                                             ; preds = %._crit_edge.1.7
; BB278 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4013
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4013
  %3086 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 4013
  %3087 = insertelement <2 x i32> %3086, i32 %113, i64 1		; visa id: 4014
  %3088 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4015
  store <2 x i32> %3087, <2 x i32>* %3088, align 4, !noalias !625		; visa id: 4015
  br label %._crit_edge267, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4017

._crit_edge267:                                   ; preds = %._crit_edge267.._crit_edge267_crit_edge, %3085
; BB279 :
  %3089 = phi i32 [ 0, %3085 ], [ %3098, %._crit_edge267.._crit_edge267_crit_edge ]
  %3090 = zext i32 %3089 to i64		; visa id: 4018
  %3091 = shl nuw nsw i64 %3090, 2		; visa id: 4019
  %3092 = add i64 %133, %3091		; visa id: 4020
  %3093 = inttoptr i64 %3092 to i32*		; visa id: 4021
  %3094 = load i32, i32* %3093, align 4, !noalias !625		; visa id: 4021
  %3095 = add i64 %128, %3091		; visa id: 4022
  %3096 = inttoptr i64 %3095 to i32*		; visa id: 4023
  store i32 %3094, i32* %3096, align 4, !alias.scope !625		; visa id: 4023
  %3097 = icmp eq i32 %3089, 0		; visa id: 4024
  br i1 %3097, label %._crit_edge267.._crit_edge267_crit_edge, label %3099, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4025

._crit_edge267.._crit_edge267_crit_edge:          ; preds = %._crit_edge267
; BB280 :
  %3098 = add nuw nsw i32 %3089, 1, !spirv.Decorations !631		; visa id: 4027
  br label %._crit_edge267, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4028

3099:                                             ; preds = %._crit_edge267
; BB281 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4030
  %3100 = load i64, i64* %129, align 8		; visa id: 4030
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4031
  %3101 = bitcast i64 %3100 to <2 x i32>		; visa id: 4031
  %3102 = extractelement <2 x i32> %3101, i32 0		; visa id: 4033
  %3103 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3102, i32 1
  %3104 = bitcast <2 x i32> %3103 to i64		; visa id: 4033
  %3105 = ashr exact i64 %3104, 32		; visa id: 4034
  %3106 = bitcast i64 %3105 to <2 x i32>		; visa id: 4035
  %3107 = extractelement <2 x i32> %3106, i32 0		; visa id: 4039
  %3108 = extractelement <2 x i32> %3106, i32 1		; visa id: 4039
  %3109 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3107, i32 %3108, i32 %41, i32 %42)
  %3110 = extractvalue { i32, i32 } %3109, 0		; visa id: 4039
  %3111 = extractvalue { i32, i32 } %3109, 1		; visa id: 4039
  %3112 = insertelement <2 x i32> undef, i32 %3110, i32 0		; visa id: 4046
  %3113 = insertelement <2 x i32> %3112, i32 %3111, i32 1		; visa id: 4047
  %3114 = bitcast <2 x i32> %3113 to i64		; visa id: 4048
  %3115 = shl i64 %3114, 1		; visa id: 4052
  %3116 = add i64 %.in401, %3115		; visa id: 4053
  %3117 = ashr i64 %3100, 31		; visa id: 4054
  %3118 = bitcast i64 %3117 to <2 x i32>		; visa id: 4055
  %3119 = extractelement <2 x i32> %3118, i32 0		; visa id: 4059
  %3120 = extractelement <2 x i32> %3118, i32 1		; visa id: 4059
  %3121 = and i32 %3119, -2		; visa id: 4059
  %3122 = insertelement <2 x i32> undef, i32 %3121, i32 0		; visa id: 4060
  %3123 = insertelement <2 x i32> %3122, i32 %3120, i32 1		; visa id: 4061
  %3124 = bitcast <2 x i32> %3123 to i64		; visa id: 4062
  %3125 = add i64 %3116, %3124		; visa id: 4066
  %3126 = inttoptr i64 %3125 to i16 addrspace(4)*		; visa id: 4067
  %3127 = addrspacecast i16 addrspace(4)* %3126 to i16 addrspace(1)*		; visa id: 4067
  %3128 = load i16, i16 addrspace(1)* %3127, align 2		; visa id: 4068
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4070
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4070
  %3129 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4070
  %3130 = insertelement <2 x i32> %3129, i32 %2885, i64 1		; visa id: 4071
  %3131 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4072
  store <2 x i32> %3130, <2 x i32>* %3131, align 4, !noalias !635		; visa id: 4072
  br label %._crit_edge268, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4074

._crit_edge268:                                   ; preds = %._crit_edge268.._crit_edge268_crit_edge, %3099
; BB282 :
  %3132 = phi i32 [ 0, %3099 ], [ %3141, %._crit_edge268.._crit_edge268_crit_edge ]
  %3133 = zext i32 %3132 to i64		; visa id: 4075
  %3134 = shl nuw nsw i64 %3133, 2		; visa id: 4076
  %3135 = add i64 %124, %3134		; visa id: 4077
  %3136 = inttoptr i64 %3135 to i32*		; visa id: 4078
  %3137 = load i32, i32* %3136, align 4, !noalias !635		; visa id: 4078
  %3138 = add i64 %119, %3134		; visa id: 4079
  %3139 = inttoptr i64 %3138 to i32*		; visa id: 4080
  store i32 %3137, i32* %3139, align 4, !alias.scope !635		; visa id: 4080
  %3140 = icmp eq i32 %3132, 0		; visa id: 4081
  br i1 %3140, label %._crit_edge268.._crit_edge268_crit_edge, label %3142, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4082

._crit_edge268.._crit_edge268_crit_edge:          ; preds = %._crit_edge268
; BB283 :
  %3141 = add nuw nsw i32 %3132, 1, !spirv.Decorations !631		; visa id: 4084
  br label %._crit_edge268, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4085

3142:                                             ; preds = %._crit_edge268
; BB284 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4087
  %3143 = load i64, i64* %120, align 8		; visa id: 4087
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4088
  %3144 = bitcast i64 %3143 to <2 x i32>		; visa id: 4088
  %3145 = extractelement <2 x i32> %3144, i32 0		; visa id: 4090
  %3146 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3145, i32 1
  %3147 = bitcast <2 x i32> %3146 to i64		; visa id: 4090
  %3148 = ashr exact i64 %3147, 32		; visa id: 4091
  %3149 = bitcast i64 %3148 to <2 x i32>		; visa id: 4092
  %3150 = extractelement <2 x i32> %3149, i32 0		; visa id: 4096
  %3151 = extractelement <2 x i32> %3149, i32 1		; visa id: 4096
  %3152 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3150, i32 %3151, i32 %44, i32 %45)
  %3153 = extractvalue { i32, i32 } %3152, 0		; visa id: 4096
  %3154 = extractvalue { i32, i32 } %3152, 1		; visa id: 4096
  %3155 = insertelement <2 x i32> undef, i32 %3153, i32 0		; visa id: 4103
  %3156 = insertelement <2 x i32> %3155, i32 %3154, i32 1		; visa id: 4104
  %3157 = bitcast <2 x i32> %3156 to i64		; visa id: 4105
  %3158 = shl i64 %3157, 1		; visa id: 4109
  %3159 = add i64 %.in400, %3158		; visa id: 4110
  %3160 = ashr i64 %3143, 31		; visa id: 4111
  %3161 = bitcast i64 %3160 to <2 x i32>		; visa id: 4112
  %3162 = extractelement <2 x i32> %3161, i32 0		; visa id: 4116
  %3163 = extractelement <2 x i32> %3161, i32 1		; visa id: 4116
  %3164 = and i32 %3162, -2		; visa id: 4116
  %3165 = insertelement <2 x i32> undef, i32 %3164, i32 0		; visa id: 4117
  %3166 = insertelement <2 x i32> %3165, i32 %3163, i32 1		; visa id: 4118
  %3167 = bitcast <2 x i32> %3166 to i64		; visa id: 4119
  %3168 = add i64 %3159, %3167		; visa id: 4123
  %3169 = inttoptr i64 %3168 to i16 addrspace(4)*		; visa id: 4124
  %3170 = addrspacecast i16 addrspace(4)* %3169 to i16 addrspace(1)*		; visa id: 4124
  %3171 = load i16, i16 addrspace(1)* %3170, align 2		; visa id: 4125
  %3172 = zext i16 %3128 to i32		; visa id: 4127
  %3173 = shl nuw i32 %3172, 16, !spirv.Decorations !639		; visa id: 4128
  %3174 = bitcast i32 %3173 to float
  %3175 = zext i16 %3171 to i32		; visa id: 4129
  %3176 = shl nuw i32 %3175, 16, !spirv.Decorations !639		; visa id: 4130
  %3177 = bitcast i32 %3176 to float
  %3178 = fmul reassoc nsz arcp contract float %3174, %3177, !spirv.Decorations !618
  %3179 = fadd reassoc nsz arcp contract float %3178, %.sroa.158.1, !spirv.Decorations !618		; visa id: 4131
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4132

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %3142
; BB285 :
  %.sroa.158.2 = phi float [ %3179, %3142 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  %3180 = icmp slt i32 %428, %const_reg_dword
  %3181 = icmp slt i32 %2885, %const_reg_dword1		; visa id: 4133
  %3182 = and i1 %3180, %3181		; visa id: 4134
  br i1 %3182, label %3183, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4136

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
; BB:
  br label %.preheader.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3183:                                             ; preds = %._crit_edge.2.7
; BB287 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4138
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4138
  %3184 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 4138
  %3185 = insertelement <2 x i32> %3184, i32 %113, i64 1		; visa id: 4139
  %3186 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4140
  store <2 x i32> %3185, <2 x i32>* %3186, align 4, !noalias !625		; visa id: 4140
  br label %._crit_edge269, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4142

._crit_edge269:                                   ; preds = %._crit_edge269.._crit_edge269_crit_edge, %3183
; BB288 :
  %3187 = phi i32 [ 0, %3183 ], [ %3196, %._crit_edge269.._crit_edge269_crit_edge ]
  %3188 = zext i32 %3187 to i64		; visa id: 4143
  %3189 = shl nuw nsw i64 %3188, 2		; visa id: 4144
  %3190 = add i64 %133, %3189		; visa id: 4145
  %3191 = inttoptr i64 %3190 to i32*		; visa id: 4146
  %3192 = load i32, i32* %3191, align 4, !noalias !625		; visa id: 4146
  %3193 = add i64 %128, %3189		; visa id: 4147
  %3194 = inttoptr i64 %3193 to i32*		; visa id: 4148
  store i32 %3192, i32* %3194, align 4, !alias.scope !625		; visa id: 4148
  %3195 = icmp eq i32 %3187, 0		; visa id: 4149
  br i1 %3195, label %._crit_edge269.._crit_edge269_crit_edge, label %3197, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4150

._crit_edge269.._crit_edge269_crit_edge:          ; preds = %._crit_edge269
; BB289 :
  %3196 = add nuw nsw i32 %3187, 1, !spirv.Decorations !631		; visa id: 4152
  br label %._crit_edge269, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4153

3197:                                             ; preds = %._crit_edge269
; BB290 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4155
  %3198 = load i64, i64* %129, align 8		; visa id: 4155
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4156
  %3199 = bitcast i64 %3198 to <2 x i32>		; visa id: 4156
  %3200 = extractelement <2 x i32> %3199, i32 0		; visa id: 4158
  %3201 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3200, i32 1
  %3202 = bitcast <2 x i32> %3201 to i64		; visa id: 4158
  %3203 = ashr exact i64 %3202, 32		; visa id: 4159
  %3204 = bitcast i64 %3203 to <2 x i32>		; visa id: 4160
  %3205 = extractelement <2 x i32> %3204, i32 0		; visa id: 4164
  %3206 = extractelement <2 x i32> %3204, i32 1		; visa id: 4164
  %3207 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3205, i32 %3206, i32 %41, i32 %42)
  %3208 = extractvalue { i32, i32 } %3207, 0		; visa id: 4164
  %3209 = extractvalue { i32, i32 } %3207, 1		; visa id: 4164
  %3210 = insertelement <2 x i32> undef, i32 %3208, i32 0		; visa id: 4171
  %3211 = insertelement <2 x i32> %3210, i32 %3209, i32 1		; visa id: 4172
  %3212 = bitcast <2 x i32> %3211 to i64		; visa id: 4173
  %3213 = shl i64 %3212, 1		; visa id: 4177
  %3214 = add i64 %.in401, %3213		; visa id: 4178
  %3215 = ashr i64 %3198, 31		; visa id: 4179
  %3216 = bitcast i64 %3215 to <2 x i32>		; visa id: 4180
  %3217 = extractelement <2 x i32> %3216, i32 0		; visa id: 4184
  %3218 = extractelement <2 x i32> %3216, i32 1		; visa id: 4184
  %3219 = and i32 %3217, -2		; visa id: 4184
  %3220 = insertelement <2 x i32> undef, i32 %3219, i32 0		; visa id: 4185
  %3221 = insertelement <2 x i32> %3220, i32 %3218, i32 1		; visa id: 4186
  %3222 = bitcast <2 x i32> %3221 to i64		; visa id: 4187
  %3223 = add i64 %3214, %3222		; visa id: 4191
  %3224 = inttoptr i64 %3223 to i16 addrspace(4)*		; visa id: 4192
  %3225 = addrspacecast i16 addrspace(4)* %3224 to i16 addrspace(1)*		; visa id: 4192
  %3226 = load i16, i16 addrspace(1)* %3225, align 2		; visa id: 4193
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4195
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4195
  %3227 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4195
  %3228 = insertelement <2 x i32> %3227, i32 %2885, i64 1		; visa id: 4196
  %3229 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4197
  store <2 x i32> %3228, <2 x i32>* %3229, align 4, !noalias !635		; visa id: 4197
  br label %._crit_edge270, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4199

._crit_edge270:                                   ; preds = %._crit_edge270.._crit_edge270_crit_edge, %3197
; BB291 :
  %3230 = phi i32 [ 0, %3197 ], [ %3239, %._crit_edge270.._crit_edge270_crit_edge ]
  %3231 = zext i32 %3230 to i64		; visa id: 4200
  %3232 = shl nuw nsw i64 %3231, 2		; visa id: 4201
  %3233 = add i64 %124, %3232		; visa id: 4202
  %3234 = inttoptr i64 %3233 to i32*		; visa id: 4203
  %3235 = load i32, i32* %3234, align 4, !noalias !635		; visa id: 4203
  %3236 = add i64 %119, %3232		; visa id: 4204
  %3237 = inttoptr i64 %3236 to i32*		; visa id: 4205
  store i32 %3235, i32* %3237, align 4, !alias.scope !635		; visa id: 4205
  %3238 = icmp eq i32 %3230, 0		; visa id: 4206
  br i1 %3238, label %._crit_edge270.._crit_edge270_crit_edge, label %3240, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4207

._crit_edge270.._crit_edge270_crit_edge:          ; preds = %._crit_edge270
; BB292 :
  %3239 = add nuw nsw i32 %3230, 1, !spirv.Decorations !631		; visa id: 4209
  br label %._crit_edge270, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4210

3240:                                             ; preds = %._crit_edge270
; BB293 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4212
  %3241 = load i64, i64* %120, align 8		; visa id: 4212
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4213
  %3242 = bitcast i64 %3241 to <2 x i32>		; visa id: 4213
  %3243 = extractelement <2 x i32> %3242, i32 0		; visa id: 4215
  %3244 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3243, i32 1
  %3245 = bitcast <2 x i32> %3244 to i64		; visa id: 4215
  %3246 = ashr exact i64 %3245, 32		; visa id: 4216
  %3247 = bitcast i64 %3246 to <2 x i32>		; visa id: 4217
  %3248 = extractelement <2 x i32> %3247, i32 0		; visa id: 4221
  %3249 = extractelement <2 x i32> %3247, i32 1		; visa id: 4221
  %3250 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3248, i32 %3249, i32 %44, i32 %45)
  %3251 = extractvalue { i32, i32 } %3250, 0		; visa id: 4221
  %3252 = extractvalue { i32, i32 } %3250, 1		; visa id: 4221
  %3253 = insertelement <2 x i32> undef, i32 %3251, i32 0		; visa id: 4228
  %3254 = insertelement <2 x i32> %3253, i32 %3252, i32 1		; visa id: 4229
  %3255 = bitcast <2 x i32> %3254 to i64		; visa id: 4230
  %3256 = shl i64 %3255, 1		; visa id: 4234
  %3257 = add i64 %.in400, %3256		; visa id: 4235
  %3258 = ashr i64 %3241, 31		; visa id: 4236
  %3259 = bitcast i64 %3258 to <2 x i32>		; visa id: 4237
  %3260 = extractelement <2 x i32> %3259, i32 0		; visa id: 4241
  %3261 = extractelement <2 x i32> %3259, i32 1		; visa id: 4241
  %3262 = and i32 %3260, -2		; visa id: 4241
  %3263 = insertelement <2 x i32> undef, i32 %3262, i32 0		; visa id: 4242
  %3264 = insertelement <2 x i32> %3263, i32 %3261, i32 1		; visa id: 4243
  %3265 = bitcast <2 x i32> %3264 to i64		; visa id: 4244
  %3266 = add i64 %3257, %3265		; visa id: 4248
  %3267 = inttoptr i64 %3266 to i16 addrspace(4)*		; visa id: 4249
  %3268 = addrspacecast i16 addrspace(4)* %3267 to i16 addrspace(1)*		; visa id: 4249
  %3269 = load i16, i16 addrspace(1)* %3268, align 2		; visa id: 4250
  %3270 = zext i16 %3226 to i32		; visa id: 4252
  %3271 = shl nuw i32 %3270, 16, !spirv.Decorations !639		; visa id: 4253
  %3272 = bitcast i32 %3271 to float
  %3273 = zext i16 %3269 to i32		; visa id: 4254
  %3274 = shl nuw i32 %3273, 16, !spirv.Decorations !639		; visa id: 4255
  %3275 = bitcast i32 %3274 to float
  %3276 = fmul reassoc nsz arcp contract float %3272, %3275, !spirv.Decorations !618
  %3277 = fadd reassoc nsz arcp contract float %3276, %.sroa.222.1, !spirv.Decorations !618		; visa id: 4256
  br label %.preheader.7, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4257

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %3240
; BB294 :
  %.sroa.222.2 = phi float [ %3277, %3240 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %3278 = add i32 %69, 8		; visa id: 4258
  %3279 = icmp slt i32 %3278, %const_reg_dword1		; visa id: 4259
  %3280 = icmp slt i32 %65, %const_reg_dword
  %3281 = and i1 %3280, %3279		; visa id: 4260
  br i1 %3281, label %3282, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4262

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
; BB:
  br label %._crit_edge.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3282:                                             ; preds = %.preheader.7
; BB296 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4264
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4264
  %3283 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 4264
  %3284 = insertelement <2 x i32> %3283, i32 %113, i64 1		; visa id: 4265
  %3285 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4266
  store <2 x i32> %3284, <2 x i32>* %3285, align 4, !noalias !625		; visa id: 4266
  br label %._crit_edge271, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4268

._crit_edge271:                                   ; preds = %._crit_edge271.._crit_edge271_crit_edge, %3282
; BB297 :
  %3286 = phi i32 [ 0, %3282 ], [ %3295, %._crit_edge271.._crit_edge271_crit_edge ]
  %3287 = zext i32 %3286 to i64		; visa id: 4269
  %3288 = shl nuw nsw i64 %3287, 2		; visa id: 4270
  %3289 = add i64 %133, %3288		; visa id: 4271
  %3290 = inttoptr i64 %3289 to i32*		; visa id: 4272
  %3291 = load i32, i32* %3290, align 4, !noalias !625		; visa id: 4272
  %3292 = add i64 %128, %3288		; visa id: 4273
  %3293 = inttoptr i64 %3292 to i32*		; visa id: 4274
  store i32 %3291, i32* %3293, align 4, !alias.scope !625		; visa id: 4274
  %3294 = icmp eq i32 %3286, 0		; visa id: 4275
  br i1 %3294, label %._crit_edge271.._crit_edge271_crit_edge, label %3296, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4276

._crit_edge271.._crit_edge271_crit_edge:          ; preds = %._crit_edge271
; BB298 :
  %3295 = add nuw nsw i32 %3286, 1, !spirv.Decorations !631		; visa id: 4278
  br label %._crit_edge271, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4279

3296:                                             ; preds = %._crit_edge271
; BB299 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4281
  %3297 = load i64, i64* %129, align 8		; visa id: 4281
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4282
  %3298 = bitcast i64 %3297 to <2 x i32>		; visa id: 4282
  %3299 = extractelement <2 x i32> %3298, i32 0		; visa id: 4284
  %3300 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3299, i32 1
  %3301 = bitcast <2 x i32> %3300 to i64		; visa id: 4284
  %3302 = ashr exact i64 %3301, 32		; visa id: 4285
  %3303 = bitcast i64 %3302 to <2 x i32>		; visa id: 4286
  %3304 = extractelement <2 x i32> %3303, i32 0		; visa id: 4290
  %3305 = extractelement <2 x i32> %3303, i32 1		; visa id: 4290
  %3306 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3304, i32 %3305, i32 %41, i32 %42)
  %3307 = extractvalue { i32, i32 } %3306, 0		; visa id: 4290
  %3308 = extractvalue { i32, i32 } %3306, 1		; visa id: 4290
  %3309 = insertelement <2 x i32> undef, i32 %3307, i32 0		; visa id: 4297
  %3310 = insertelement <2 x i32> %3309, i32 %3308, i32 1		; visa id: 4298
  %3311 = bitcast <2 x i32> %3310 to i64		; visa id: 4299
  %3312 = shl i64 %3311, 1		; visa id: 4303
  %3313 = add i64 %.in401, %3312		; visa id: 4304
  %3314 = ashr i64 %3297, 31		; visa id: 4305
  %3315 = bitcast i64 %3314 to <2 x i32>		; visa id: 4306
  %3316 = extractelement <2 x i32> %3315, i32 0		; visa id: 4310
  %3317 = extractelement <2 x i32> %3315, i32 1		; visa id: 4310
  %3318 = and i32 %3316, -2		; visa id: 4310
  %3319 = insertelement <2 x i32> undef, i32 %3318, i32 0		; visa id: 4311
  %3320 = insertelement <2 x i32> %3319, i32 %3317, i32 1		; visa id: 4312
  %3321 = bitcast <2 x i32> %3320 to i64		; visa id: 4313
  %3322 = add i64 %3313, %3321		; visa id: 4317
  %3323 = inttoptr i64 %3322 to i16 addrspace(4)*		; visa id: 4318
  %3324 = addrspacecast i16 addrspace(4)* %3323 to i16 addrspace(1)*		; visa id: 4318
  %3325 = load i16, i16 addrspace(1)* %3324, align 2		; visa id: 4319
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4321
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4321
  %3326 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4321
  %3327 = insertelement <2 x i32> %3326, i32 %3278, i64 1		; visa id: 4322
  %3328 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4323
  store <2 x i32> %3327, <2 x i32>* %3328, align 4, !noalias !635		; visa id: 4323
  br label %._crit_edge272, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4325

._crit_edge272:                                   ; preds = %._crit_edge272.._crit_edge272_crit_edge, %3296
; BB300 :
  %3329 = phi i32 [ 0, %3296 ], [ %3338, %._crit_edge272.._crit_edge272_crit_edge ]
  %3330 = zext i32 %3329 to i64		; visa id: 4326
  %3331 = shl nuw nsw i64 %3330, 2		; visa id: 4327
  %3332 = add i64 %124, %3331		; visa id: 4328
  %3333 = inttoptr i64 %3332 to i32*		; visa id: 4329
  %3334 = load i32, i32* %3333, align 4, !noalias !635		; visa id: 4329
  %3335 = add i64 %119, %3331		; visa id: 4330
  %3336 = inttoptr i64 %3335 to i32*		; visa id: 4331
  store i32 %3334, i32* %3336, align 4, !alias.scope !635		; visa id: 4331
  %3337 = icmp eq i32 %3329, 0		; visa id: 4332
  br i1 %3337, label %._crit_edge272.._crit_edge272_crit_edge, label %3339, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4333

._crit_edge272.._crit_edge272_crit_edge:          ; preds = %._crit_edge272
; BB301 :
  %3338 = add nuw nsw i32 %3329, 1, !spirv.Decorations !631		; visa id: 4335
  br label %._crit_edge272, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4336

3339:                                             ; preds = %._crit_edge272
; BB302 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4338
  %3340 = load i64, i64* %120, align 8		; visa id: 4338
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4339
  %3341 = bitcast i64 %3340 to <2 x i32>		; visa id: 4339
  %3342 = extractelement <2 x i32> %3341, i32 0		; visa id: 4341
  %3343 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3342, i32 1
  %3344 = bitcast <2 x i32> %3343 to i64		; visa id: 4341
  %3345 = ashr exact i64 %3344, 32		; visa id: 4342
  %3346 = bitcast i64 %3345 to <2 x i32>		; visa id: 4343
  %3347 = extractelement <2 x i32> %3346, i32 0		; visa id: 4347
  %3348 = extractelement <2 x i32> %3346, i32 1		; visa id: 4347
  %3349 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3347, i32 %3348, i32 %44, i32 %45)
  %3350 = extractvalue { i32, i32 } %3349, 0		; visa id: 4347
  %3351 = extractvalue { i32, i32 } %3349, 1		; visa id: 4347
  %3352 = insertelement <2 x i32> undef, i32 %3350, i32 0		; visa id: 4354
  %3353 = insertelement <2 x i32> %3352, i32 %3351, i32 1		; visa id: 4355
  %3354 = bitcast <2 x i32> %3353 to i64		; visa id: 4356
  %3355 = shl i64 %3354, 1		; visa id: 4360
  %3356 = add i64 %.in400, %3355		; visa id: 4361
  %3357 = ashr i64 %3340, 31		; visa id: 4362
  %3358 = bitcast i64 %3357 to <2 x i32>		; visa id: 4363
  %3359 = extractelement <2 x i32> %3358, i32 0		; visa id: 4367
  %3360 = extractelement <2 x i32> %3358, i32 1		; visa id: 4367
  %3361 = and i32 %3359, -2		; visa id: 4367
  %3362 = insertelement <2 x i32> undef, i32 %3361, i32 0		; visa id: 4368
  %3363 = insertelement <2 x i32> %3362, i32 %3360, i32 1		; visa id: 4369
  %3364 = bitcast <2 x i32> %3363 to i64		; visa id: 4370
  %3365 = add i64 %3356, %3364		; visa id: 4374
  %3366 = inttoptr i64 %3365 to i16 addrspace(4)*		; visa id: 4375
  %3367 = addrspacecast i16 addrspace(4)* %3366 to i16 addrspace(1)*		; visa id: 4375
  %3368 = load i16, i16 addrspace(1)* %3367, align 2		; visa id: 4376
  %3369 = zext i16 %3325 to i32		; visa id: 4378
  %3370 = shl nuw i32 %3369, 16, !spirv.Decorations !639		; visa id: 4379
  %3371 = bitcast i32 %3370 to float
  %3372 = zext i16 %3368 to i32		; visa id: 4380
  %3373 = shl nuw i32 %3372, 16, !spirv.Decorations !639		; visa id: 4381
  %3374 = bitcast i32 %3373 to float
  %3375 = fmul reassoc nsz arcp contract float %3371, %3374, !spirv.Decorations !618
  %3376 = fadd reassoc nsz arcp contract float %3375, %.sroa.34.1, !spirv.Decorations !618		; visa id: 4382
  br label %._crit_edge.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4383

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %3339
; BB303 :
  %.sroa.34.2 = phi float [ %3376, %3339 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  %3377 = icmp slt i32 %230, %const_reg_dword
  %3378 = icmp slt i32 %3278, %const_reg_dword1		; visa id: 4384
  %3379 = and i1 %3377, %3378		; visa id: 4385
  br i1 %3379, label %3380, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4387

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
; BB:
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3380:                                             ; preds = %._crit_edge.8
; BB305 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4389
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4389
  %3381 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 4389
  %3382 = insertelement <2 x i32> %3381, i32 %113, i64 1		; visa id: 4390
  %3383 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4391
  store <2 x i32> %3382, <2 x i32>* %3383, align 4, !noalias !625		; visa id: 4391
  br label %._crit_edge273, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4393

._crit_edge273:                                   ; preds = %._crit_edge273.._crit_edge273_crit_edge, %3380
; BB306 :
  %3384 = phi i32 [ 0, %3380 ], [ %3393, %._crit_edge273.._crit_edge273_crit_edge ]
  %3385 = zext i32 %3384 to i64		; visa id: 4394
  %3386 = shl nuw nsw i64 %3385, 2		; visa id: 4395
  %3387 = add i64 %133, %3386		; visa id: 4396
  %3388 = inttoptr i64 %3387 to i32*		; visa id: 4397
  %3389 = load i32, i32* %3388, align 4, !noalias !625		; visa id: 4397
  %3390 = add i64 %128, %3386		; visa id: 4398
  %3391 = inttoptr i64 %3390 to i32*		; visa id: 4399
  store i32 %3389, i32* %3391, align 4, !alias.scope !625		; visa id: 4399
  %3392 = icmp eq i32 %3384, 0		; visa id: 4400
  br i1 %3392, label %._crit_edge273.._crit_edge273_crit_edge, label %3394, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4401

._crit_edge273.._crit_edge273_crit_edge:          ; preds = %._crit_edge273
; BB307 :
  %3393 = add nuw nsw i32 %3384, 1, !spirv.Decorations !631		; visa id: 4403
  br label %._crit_edge273, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4404

3394:                                             ; preds = %._crit_edge273
; BB308 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4406
  %3395 = load i64, i64* %129, align 8		; visa id: 4406
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4407
  %3396 = bitcast i64 %3395 to <2 x i32>		; visa id: 4407
  %3397 = extractelement <2 x i32> %3396, i32 0		; visa id: 4409
  %3398 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3397, i32 1
  %3399 = bitcast <2 x i32> %3398 to i64		; visa id: 4409
  %3400 = ashr exact i64 %3399, 32		; visa id: 4410
  %3401 = bitcast i64 %3400 to <2 x i32>		; visa id: 4411
  %3402 = extractelement <2 x i32> %3401, i32 0		; visa id: 4415
  %3403 = extractelement <2 x i32> %3401, i32 1		; visa id: 4415
  %3404 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3402, i32 %3403, i32 %41, i32 %42)
  %3405 = extractvalue { i32, i32 } %3404, 0		; visa id: 4415
  %3406 = extractvalue { i32, i32 } %3404, 1		; visa id: 4415
  %3407 = insertelement <2 x i32> undef, i32 %3405, i32 0		; visa id: 4422
  %3408 = insertelement <2 x i32> %3407, i32 %3406, i32 1		; visa id: 4423
  %3409 = bitcast <2 x i32> %3408 to i64		; visa id: 4424
  %3410 = shl i64 %3409, 1		; visa id: 4428
  %3411 = add i64 %.in401, %3410		; visa id: 4429
  %3412 = ashr i64 %3395, 31		; visa id: 4430
  %3413 = bitcast i64 %3412 to <2 x i32>		; visa id: 4431
  %3414 = extractelement <2 x i32> %3413, i32 0		; visa id: 4435
  %3415 = extractelement <2 x i32> %3413, i32 1		; visa id: 4435
  %3416 = and i32 %3414, -2		; visa id: 4435
  %3417 = insertelement <2 x i32> undef, i32 %3416, i32 0		; visa id: 4436
  %3418 = insertelement <2 x i32> %3417, i32 %3415, i32 1		; visa id: 4437
  %3419 = bitcast <2 x i32> %3418 to i64		; visa id: 4438
  %3420 = add i64 %3411, %3419		; visa id: 4442
  %3421 = inttoptr i64 %3420 to i16 addrspace(4)*		; visa id: 4443
  %3422 = addrspacecast i16 addrspace(4)* %3421 to i16 addrspace(1)*		; visa id: 4443
  %3423 = load i16, i16 addrspace(1)* %3422, align 2		; visa id: 4444
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4446
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4446
  %3424 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4446
  %3425 = insertelement <2 x i32> %3424, i32 %3278, i64 1		; visa id: 4447
  %3426 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4448
  store <2 x i32> %3425, <2 x i32>* %3426, align 4, !noalias !635		; visa id: 4448
  br label %._crit_edge274, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4450

._crit_edge274:                                   ; preds = %._crit_edge274.._crit_edge274_crit_edge, %3394
; BB309 :
  %3427 = phi i32 [ 0, %3394 ], [ %3436, %._crit_edge274.._crit_edge274_crit_edge ]
  %3428 = zext i32 %3427 to i64		; visa id: 4451
  %3429 = shl nuw nsw i64 %3428, 2		; visa id: 4452
  %3430 = add i64 %124, %3429		; visa id: 4453
  %3431 = inttoptr i64 %3430 to i32*		; visa id: 4454
  %3432 = load i32, i32* %3431, align 4, !noalias !635		; visa id: 4454
  %3433 = add i64 %119, %3429		; visa id: 4455
  %3434 = inttoptr i64 %3433 to i32*		; visa id: 4456
  store i32 %3432, i32* %3434, align 4, !alias.scope !635		; visa id: 4456
  %3435 = icmp eq i32 %3427, 0		; visa id: 4457
  br i1 %3435, label %._crit_edge274.._crit_edge274_crit_edge, label %3437, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4458

._crit_edge274.._crit_edge274_crit_edge:          ; preds = %._crit_edge274
; BB310 :
  %3436 = add nuw nsw i32 %3427, 1, !spirv.Decorations !631		; visa id: 4460
  br label %._crit_edge274, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4461

3437:                                             ; preds = %._crit_edge274
; BB311 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4463
  %3438 = load i64, i64* %120, align 8		; visa id: 4463
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4464
  %3439 = bitcast i64 %3438 to <2 x i32>		; visa id: 4464
  %3440 = extractelement <2 x i32> %3439, i32 0		; visa id: 4466
  %3441 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3440, i32 1
  %3442 = bitcast <2 x i32> %3441 to i64		; visa id: 4466
  %3443 = ashr exact i64 %3442, 32		; visa id: 4467
  %3444 = bitcast i64 %3443 to <2 x i32>		; visa id: 4468
  %3445 = extractelement <2 x i32> %3444, i32 0		; visa id: 4472
  %3446 = extractelement <2 x i32> %3444, i32 1		; visa id: 4472
  %3447 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3445, i32 %3446, i32 %44, i32 %45)
  %3448 = extractvalue { i32, i32 } %3447, 0		; visa id: 4472
  %3449 = extractvalue { i32, i32 } %3447, 1		; visa id: 4472
  %3450 = insertelement <2 x i32> undef, i32 %3448, i32 0		; visa id: 4479
  %3451 = insertelement <2 x i32> %3450, i32 %3449, i32 1		; visa id: 4480
  %3452 = bitcast <2 x i32> %3451 to i64		; visa id: 4481
  %3453 = shl i64 %3452, 1		; visa id: 4485
  %3454 = add i64 %.in400, %3453		; visa id: 4486
  %3455 = ashr i64 %3438, 31		; visa id: 4487
  %3456 = bitcast i64 %3455 to <2 x i32>		; visa id: 4488
  %3457 = extractelement <2 x i32> %3456, i32 0		; visa id: 4492
  %3458 = extractelement <2 x i32> %3456, i32 1		; visa id: 4492
  %3459 = and i32 %3457, -2		; visa id: 4492
  %3460 = insertelement <2 x i32> undef, i32 %3459, i32 0		; visa id: 4493
  %3461 = insertelement <2 x i32> %3460, i32 %3458, i32 1		; visa id: 4494
  %3462 = bitcast <2 x i32> %3461 to i64		; visa id: 4495
  %3463 = add i64 %3454, %3462		; visa id: 4499
  %3464 = inttoptr i64 %3463 to i16 addrspace(4)*		; visa id: 4500
  %3465 = addrspacecast i16 addrspace(4)* %3464 to i16 addrspace(1)*		; visa id: 4500
  %3466 = load i16, i16 addrspace(1)* %3465, align 2		; visa id: 4501
  %3467 = zext i16 %3423 to i32		; visa id: 4503
  %3468 = shl nuw i32 %3467, 16, !spirv.Decorations !639		; visa id: 4504
  %3469 = bitcast i32 %3468 to float
  %3470 = zext i16 %3466 to i32		; visa id: 4505
  %3471 = shl nuw i32 %3470, 16, !spirv.Decorations !639		; visa id: 4506
  %3472 = bitcast i32 %3471 to float
  %3473 = fmul reassoc nsz arcp contract float %3469, %3472, !spirv.Decorations !618
  %3474 = fadd reassoc nsz arcp contract float %3473, %.sroa.98.1, !spirv.Decorations !618		; visa id: 4507
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4508

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %3437
; BB312 :
  %.sroa.98.2 = phi float [ %3474, %3437 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  %3475 = icmp slt i32 %329, %const_reg_dword
  %3476 = icmp slt i32 %3278, %const_reg_dword1		; visa id: 4509
  %3477 = and i1 %3475, %3476		; visa id: 4510
  br i1 %3477, label %3478, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4512

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
; BB:
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3478:                                             ; preds = %._crit_edge.1.8
; BB314 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4514
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4514
  %3479 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 4514
  %3480 = insertelement <2 x i32> %3479, i32 %113, i64 1		; visa id: 4515
  %3481 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4516
  store <2 x i32> %3480, <2 x i32>* %3481, align 4, !noalias !625		; visa id: 4516
  br label %._crit_edge275, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4518

._crit_edge275:                                   ; preds = %._crit_edge275.._crit_edge275_crit_edge, %3478
; BB315 :
  %3482 = phi i32 [ 0, %3478 ], [ %3491, %._crit_edge275.._crit_edge275_crit_edge ]
  %3483 = zext i32 %3482 to i64		; visa id: 4519
  %3484 = shl nuw nsw i64 %3483, 2		; visa id: 4520
  %3485 = add i64 %133, %3484		; visa id: 4521
  %3486 = inttoptr i64 %3485 to i32*		; visa id: 4522
  %3487 = load i32, i32* %3486, align 4, !noalias !625		; visa id: 4522
  %3488 = add i64 %128, %3484		; visa id: 4523
  %3489 = inttoptr i64 %3488 to i32*		; visa id: 4524
  store i32 %3487, i32* %3489, align 4, !alias.scope !625		; visa id: 4524
  %3490 = icmp eq i32 %3482, 0		; visa id: 4525
  br i1 %3490, label %._crit_edge275.._crit_edge275_crit_edge, label %3492, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4526

._crit_edge275.._crit_edge275_crit_edge:          ; preds = %._crit_edge275
; BB316 :
  %3491 = add nuw nsw i32 %3482, 1, !spirv.Decorations !631		; visa id: 4528
  br label %._crit_edge275, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4529

3492:                                             ; preds = %._crit_edge275
; BB317 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4531
  %3493 = load i64, i64* %129, align 8		; visa id: 4531
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4532
  %3494 = bitcast i64 %3493 to <2 x i32>		; visa id: 4532
  %3495 = extractelement <2 x i32> %3494, i32 0		; visa id: 4534
  %3496 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3495, i32 1
  %3497 = bitcast <2 x i32> %3496 to i64		; visa id: 4534
  %3498 = ashr exact i64 %3497, 32		; visa id: 4535
  %3499 = bitcast i64 %3498 to <2 x i32>		; visa id: 4536
  %3500 = extractelement <2 x i32> %3499, i32 0		; visa id: 4540
  %3501 = extractelement <2 x i32> %3499, i32 1		; visa id: 4540
  %3502 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3500, i32 %3501, i32 %41, i32 %42)
  %3503 = extractvalue { i32, i32 } %3502, 0		; visa id: 4540
  %3504 = extractvalue { i32, i32 } %3502, 1		; visa id: 4540
  %3505 = insertelement <2 x i32> undef, i32 %3503, i32 0		; visa id: 4547
  %3506 = insertelement <2 x i32> %3505, i32 %3504, i32 1		; visa id: 4548
  %3507 = bitcast <2 x i32> %3506 to i64		; visa id: 4549
  %3508 = shl i64 %3507, 1		; visa id: 4553
  %3509 = add i64 %.in401, %3508		; visa id: 4554
  %3510 = ashr i64 %3493, 31		; visa id: 4555
  %3511 = bitcast i64 %3510 to <2 x i32>		; visa id: 4556
  %3512 = extractelement <2 x i32> %3511, i32 0		; visa id: 4560
  %3513 = extractelement <2 x i32> %3511, i32 1		; visa id: 4560
  %3514 = and i32 %3512, -2		; visa id: 4560
  %3515 = insertelement <2 x i32> undef, i32 %3514, i32 0		; visa id: 4561
  %3516 = insertelement <2 x i32> %3515, i32 %3513, i32 1		; visa id: 4562
  %3517 = bitcast <2 x i32> %3516 to i64		; visa id: 4563
  %3518 = add i64 %3509, %3517		; visa id: 4567
  %3519 = inttoptr i64 %3518 to i16 addrspace(4)*		; visa id: 4568
  %3520 = addrspacecast i16 addrspace(4)* %3519 to i16 addrspace(1)*		; visa id: 4568
  %3521 = load i16, i16 addrspace(1)* %3520, align 2		; visa id: 4569
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4571
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4571
  %3522 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4571
  %3523 = insertelement <2 x i32> %3522, i32 %3278, i64 1		; visa id: 4572
  %3524 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4573
  store <2 x i32> %3523, <2 x i32>* %3524, align 4, !noalias !635		; visa id: 4573
  br label %._crit_edge276, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4575

._crit_edge276:                                   ; preds = %._crit_edge276.._crit_edge276_crit_edge, %3492
; BB318 :
  %3525 = phi i32 [ 0, %3492 ], [ %3534, %._crit_edge276.._crit_edge276_crit_edge ]
  %3526 = zext i32 %3525 to i64		; visa id: 4576
  %3527 = shl nuw nsw i64 %3526, 2		; visa id: 4577
  %3528 = add i64 %124, %3527		; visa id: 4578
  %3529 = inttoptr i64 %3528 to i32*		; visa id: 4579
  %3530 = load i32, i32* %3529, align 4, !noalias !635		; visa id: 4579
  %3531 = add i64 %119, %3527		; visa id: 4580
  %3532 = inttoptr i64 %3531 to i32*		; visa id: 4581
  store i32 %3530, i32* %3532, align 4, !alias.scope !635		; visa id: 4581
  %3533 = icmp eq i32 %3525, 0		; visa id: 4582
  br i1 %3533, label %._crit_edge276.._crit_edge276_crit_edge, label %3535, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4583

._crit_edge276.._crit_edge276_crit_edge:          ; preds = %._crit_edge276
; BB319 :
  %3534 = add nuw nsw i32 %3525, 1, !spirv.Decorations !631		; visa id: 4585
  br label %._crit_edge276, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4586

3535:                                             ; preds = %._crit_edge276
; BB320 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4588
  %3536 = load i64, i64* %120, align 8		; visa id: 4588
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4589
  %3537 = bitcast i64 %3536 to <2 x i32>		; visa id: 4589
  %3538 = extractelement <2 x i32> %3537, i32 0		; visa id: 4591
  %3539 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3538, i32 1
  %3540 = bitcast <2 x i32> %3539 to i64		; visa id: 4591
  %3541 = ashr exact i64 %3540, 32		; visa id: 4592
  %3542 = bitcast i64 %3541 to <2 x i32>		; visa id: 4593
  %3543 = extractelement <2 x i32> %3542, i32 0		; visa id: 4597
  %3544 = extractelement <2 x i32> %3542, i32 1		; visa id: 4597
  %3545 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3543, i32 %3544, i32 %44, i32 %45)
  %3546 = extractvalue { i32, i32 } %3545, 0		; visa id: 4597
  %3547 = extractvalue { i32, i32 } %3545, 1		; visa id: 4597
  %3548 = insertelement <2 x i32> undef, i32 %3546, i32 0		; visa id: 4604
  %3549 = insertelement <2 x i32> %3548, i32 %3547, i32 1		; visa id: 4605
  %3550 = bitcast <2 x i32> %3549 to i64		; visa id: 4606
  %3551 = shl i64 %3550, 1		; visa id: 4610
  %3552 = add i64 %.in400, %3551		; visa id: 4611
  %3553 = ashr i64 %3536, 31		; visa id: 4612
  %3554 = bitcast i64 %3553 to <2 x i32>		; visa id: 4613
  %3555 = extractelement <2 x i32> %3554, i32 0		; visa id: 4617
  %3556 = extractelement <2 x i32> %3554, i32 1		; visa id: 4617
  %3557 = and i32 %3555, -2		; visa id: 4617
  %3558 = insertelement <2 x i32> undef, i32 %3557, i32 0		; visa id: 4618
  %3559 = insertelement <2 x i32> %3558, i32 %3556, i32 1		; visa id: 4619
  %3560 = bitcast <2 x i32> %3559 to i64		; visa id: 4620
  %3561 = add i64 %3552, %3560		; visa id: 4624
  %3562 = inttoptr i64 %3561 to i16 addrspace(4)*		; visa id: 4625
  %3563 = addrspacecast i16 addrspace(4)* %3562 to i16 addrspace(1)*		; visa id: 4625
  %3564 = load i16, i16 addrspace(1)* %3563, align 2		; visa id: 4626
  %3565 = zext i16 %3521 to i32		; visa id: 4628
  %3566 = shl nuw i32 %3565, 16, !spirv.Decorations !639		; visa id: 4629
  %3567 = bitcast i32 %3566 to float
  %3568 = zext i16 %3564 to i32		; visa id: 4630
  %3569 = shl nuw i32 %3568, 16, !spirv.Decorations !639		; visa id: 4631
  %3570 = bitcast i32 %3569 to float
  %3571 = fmul reassoc nsz arcp contract float %3567, %3570, !spirv.Decorations !618
  %3572 = fadd reassoc nsz arcp contract float %3571, %.sroa.162.1, !spirv.Decorations !618		; visa id: 4632
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4633

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %3535
; BB321 :
  %.sroa.162.2 = phi float [ %3572, %3535 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  %3573 = icmp slt i32 %428, %const_reg_dword
  %3574 = icmp slt i32 %3278, %const_reg_dword1		; visa id: 4634
  %3575 = and i1 %3573, %3574		; visa id: 4635
  br i1 %3575, label %3576, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4637

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
; BB:
  br label %.preheader.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3576:                                             ; preds = %._crit_edge.2.8
; BB323 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4639
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4639
  %3577 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 4639
  %3578 = insertelement <2 x i32> %3577, i32 %113, i64 1		; visa id: 4640
  %3579 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4641
  store <2 x i32> %3578, <2 x i32>* %3579, align 4, !noalias !625		; visa id: 4641
  br label %._crit_edge277, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4643

._crit_edge277:                                   ; preds = %._crit_edge277.._crit_edge277_crit_edge, %3576
; BB324 :
  %3580 = phi i32 [ 0, %3576 ], [ %3589, %._crit_edge277.._crit_edge277_crit_edge ]
  %3581 = zext i32 %3580 to i64		; visa id: 4644
  %3582 = shl nuw nsw i64 %3581, 2		; visa id: 4645
  %3583 = add i64 %133, %3582		; visa id: 4646
  %3584 = inttoptr i64 %3583 to i32*		; visa id: 4647
  %3585 = load i32, i32* %3584, align 4, !noalias !625		; visa id: 4647
  %3586 = add i64 %128, %3582		; visa id: 4648
  %3587 = inttoptr i64 %3586 to i32*		; visa id: 4649
  store i32 %3585, i32* %3587, align 4, !alias.scope !625		; visa id: 4649
  %3588 = icmp eq i32 %3580, 0		; visa id: 4650
  br i1 %3588, label %._crit_edge277.._crit_edge277_crit_edge, label %3590, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4651

._crit_edge277.._crit_edge277_crit_edge:          ; preds = %._crit_edge277
; BB325 :
  %3589 = add nuw nsw i32 %3580, 1, !spirv.Decorations !631		; visa id: 4653
  br label %._crit_edge277, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4654

3590:                                             ; preds = %._crit_edge277
; BB326 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4656
  %3591 = load i64, i64* %129, align 8		; visa id: 4656
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4657
  %3592 = bitcast i64 %3591 to <2 x i32>		; visa id: 4657
  %3593 = extractelement <2 x i32> %3592, i32 0		; visa id: 4659
  %3594 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3593, i32 1
  %3595 = bitcast <2 x i32> %3594 to i64		; visa id: 4659
  %3596 = ashr exact i64 %3595, 32		; visa id: 4660
  %3597 = bitcast i64 %3596 to <2 x i32>		; visa id: 4661
  %3598 = extractelement <2 x i32> %3597, i32 0		; visa id: 4665
  %3599 = extractelement <2 x i32> %3597, i32 1		; visa id: 4665
  %3600 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3598, i32 %3599, i32 %41, i32 %42)
  %3601 = extractvalue { i32, i32 } %3600, 0		; visa id: 4665
  %3602 = extractvalue { i32, i32 } %3600, 1		; visa id: 4665
  %3603 = insertelement <2 x i32> undef, i32 %3601, i32 0		; visa id: 4672
  %3604 = insertelement <2 x i32> %3603, i32 %3602, i32 1		; visa id: 4673
  %3605 = bitcast <2 x i32> %3604 to i64		; visa id: 4674
  %3606 = shl i64 %3605, 1		; visa id: 4678
  %3607 = add i64 %.in401, %3606		; visa id: 4679
  %3608 = ashr i64 %3591, 31		; visa id: 4680
  %3609 = bitcast i64 %3608 to <2 x i32>		; visa id: 4681
  %3610 = extractelement <2 x i32> %3609, i32 0		; visa id: 4685
  %3611 = extractelement <2 x i32> %3609, i32 1		; visa id: 4685
  %3612 = and i32 %3610, -2		; visa id: 4685
  %3613 = insertelement <2 x i32> undef, i32 %3612, i32 0		; visa id: 4686
  %3614 = insertelement <2 x i32> %3613, i32 %3611, i32 1		; visa id: 4687
  %3615 = bitcast <2 x i32> %3614 to i64		; visa id: 4688
  %3616 = add i64 %3607, %3615		; visa id: 4692
  %3617 = inttoptr i64 %3616 to i16 addrspace(4)*		; visa id: 4693
  %3618 = addrspacecast i16 addrspace(4)* %3617 to i16 addrspace(1)*		; visa id: 4693
  %3619 = load i16, i16 addrspace(1)* %3618, align 2		; visa id: 4694
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4696
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4696
  %3620 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4696
  %3621 = insertelement <2 x i32> %3620, i32 %3278, i64 1		; visa id: 4697
  %3622 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4698
  store <2 x i32> %3621, <2 x i32>* %3622, align 4, !noalias !635		; visa id: 4698
  br label %._crit_edge278, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4700

._crit_edge278:                                   ; preds = %._crit_edge278.._crit_edge278_crit_edge, %3590
; BB327 :
  %3623 = phi i32 [ 0, %3590 ], [ %3632, %._crit_edge278.._crit_edge278_crit_edge ]
  %3624 = zext i32 %3623 to i64		; visa id: 4701
  %3625 = shl nuw nsw i64 %3624, 2		; visa id: 4702
  %3626 = add i64 %124, %3625		; visa id: 4703
  %3627 = inttoptr i64 %3626 to i32*		; visa id: 4704
  %3628 = load i32, i32* %3627, align 4, !noalias !635		; visa id: 4704
  %3629 = add i64 %119, %3625		; visa id: 4705
  %3630 = inttoptr i64 %3629 to i32*		; visa id: 4706
  store i32 %3628, i32* %3630, align 4, !alias.scope !635		; visa id: 4706
  %3631 = icmp eq i32 %3623, 0		; visa id: 4707
  br i1 %3631, label %._crit_edge278.._crit_edge278_crit_edge, label %3633, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4708

._crit_edge278.._crit_edge278_crit_edge:          ; preds = %._crit_edge278
; BB328 :
  %3632 = add nuw nsw i32 %3623, 1, !spirv.Decorations !631		; visa id: 4710
  br label %._crit_edge278, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4711

3633:                                             ; preds = %._crit_edge278
; BB329 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4713
  %3634 = load i64, i64* %120, align 8		; visa id: 4713
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4714
  %3635 = bitcast i64 %3634 to <2 x i32>		; visa id: 4714
  %3636 = extractelement <2 x i32> %3635, i32 0		; visa id: 4716
  %3637 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3636, i32 1
  %3638 = bitcast <2 x i32> %3637 to i64		; visa id: 4716
  %3639 = ashr exact i64 %3638, 32		; visa id: 4717
  %3640 = bitcast i64 %3639 to <2 x i32>		; visa id: 4718
  %3641 = extractelement <2 x i32> %3640, i32 0		; visa id: 4722
  %3642 = extractelement <2 x i32> %3640, i32 1		; visa id: 4722
  %3643 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3641, i32 %3642, i32 %44, i32 %45)
  %3644 = extractvalue { i32, i32 } %3643, 0		; visa id: 4722
  %3645 = extractvalue { i32, i32 } %3643, 1		; visa id: 4722
  %3646 = insertelement <2 x i32> undef, i32 %3644, i32 0		; visa id: 4729
  %3647 = insertelement <2 x i32> %3646, i32 %3645, i32 1		; visa id: 4730
  %3648 = bitcast <2 x i32> %3647 to i64		; visa id: 4731
  %3649 = shl i64 %3648, 1		; visa id: 4735
  %3650 = add i64 %.in400, %3649		; visa id: 4736
  %3651 = ashr i64 %3634, 31		; visa id: 4737
  %3652 = bitcast i64 %3651 to <2 x i32>		; visa id: 4738
  %3653 = extractelement <2 x i32> %3652, i32 0		; visa id: 4742
  %3654 = extractelement <2 x i32> %3652, i32 1		; visa id: 4742
  %3655 = and i32 %3653, -2		; visa id: 4742
  %3656 = insertelement <2 x i32> undef, i32 %3655, i32 0		; visa id: 4743
  %3657 = insertelement <2 x i32> %3656, i32 %3654, i32 1		; visa id: 4744
  %3658 = bitcast <2 x i32> %3657 to i64		; visa id: 4745
  %3659 = add i64 %3650, %3658		; visa id: 4749
  %3660 = inttoptr i64 %3659 to i16 addrspace(4)*		; visa id: 4750
  %3661 = addrspacecast i16 addrspace(4)* %3660 to i16 addrspace(1)*		; visa id: 4750
  %3662 = load i16, i16 addrspace(1)* %3661, align 2		; visa id: 4751
  %3663 = zext i16 %3619 to i32		; visa id: 4753
  %3664 = shl nuw i32 %3663, 16, !spirv.Decorations !639		; visa id: 4754
  %3665 = bitcast i32 %3664 to float
  %3666 = zext i16 %3662 to i32		; visa id: 4755
  %3667 = shl nuw i32 %3666, 16, !spirv.Decorations !639		; visa id: 4756
  %3668 = bitcast i32 %3667 to float
  %3669 = fmul reassoc nsz arcp contract float %3665, %3668, !spirv.Decorations !618
  %3670 = fadd reassoc nsz arcp contract float %3669, %.sroa.226.1, !spirv.Decorations !618		; visa id: 4757
  br label %.preheader.8, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4758

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %3633
; BB330 :
  %.sroa.226.2 = phi float [ %3670, %3633 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %3671 = add i32 %69, 9		; visa id: 4759
  %3672 = icmp slt i32 %3671, %const_reg_dword1		; visa id: 4760
  %3673 = icmp slt i32 %65, %const_reg_dword
  %3674 = and i1 %3673, %3672		; visa id: 4761
  br i1 %3674, label %3675, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4763

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
; BB:
  br label %._crit_edge.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3675:                                             ; preds = %.preheader.8
; BB332 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4765
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4765
  %3676 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 4765
  %3677 = insertelement <2 x i32> %3676, i32 %113, i64 1		; visa id: 4766
  %3678 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4767
  store <2 x i32> %3677, <2 x i32>* %3678, align 4, !noalias !625		; visa id: 4767
  br label %._crit_edge279, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4769

._crit_edge279:                                   ; preds = %._crit_edge279.._crit_edge279_crit_edge, %3675
; BB333 :
  %3679 = phi i32 [ 0, %3675 ], [ %3688, %._crit_edge279.._crit_edge279_crit_edge ]
  %3680 = zext i32 %3679 to i64		; visa id: 4770
  %3681 = shl nuw nsw i64 %3680, 2		; visa id: 4771
  %3682 = add i64 %133, %3681		; visa id: 4772
  %3683 = inttoptr i64 %3682 to i32*		; visa id: 4773
  %3684 = load i32, i32* %3683, align 4, !noalias !625		; visa id: 4773
  %3685 = add i64 %128, %3681		; visa id: 4774
  %3686 = inttoptr i64 %3685 to i32*		; visa id: 4775
  store i32 %3684, i32* %3686, align 4, !alias.scope !625		; visa id: 4775
  %3687 = icmp eq i32 %3679, 0		; visa id: 4776
  br i1 %3687, label %._crit_edge279.._crit_edge279_crit_edge, label %3689, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4777

._crit_edge279.._crit_edge279_crit_edge:          ; preds = %._crit_edge279
; BB334 :
  %3688 = add nuw nsw i32 %3679, 1, !spirv.Decorations !631		; visa id: 4779
  br label %._crit_edge279, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4780

3689:                                             ; preds = %._crit_edge279
; BB335 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4782
  %3690 = load i64, i64* %129, align 8		; visa id: 4782
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4783
  %3691 = bitcast i64 %3690 to <2 x i32>		; visa id: 4783
  %3692 = extractelement <2 x i32> %3691, i32 0		; visa id: 4785
  %3693 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3692, i32 1
  %3694 = bitcast <2 x i32> %3693 to i64		; visa id: 4785
  %3695 = ashr exact i64 %3694, 32		; visa id: 4786
  %3696 = bitcast i64 %3695 to <2 x i32>		; visa id: 4787
  %3697 = extractelement <2 x i32> %3696, i32 0		; visa id: 4791
  %3698 = extractelement <2 x i32> %3696, i32 1		; visa id: 4791
  %3699 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3697, i32 %3698, i32 %41, i32 %42)
  %3700 = extractvalue { i32, i32 } %3699, 0		; visa id: 4791
  %3701 = extractvalue { i32, i32 } %3699, 1		; visa id: 4791
  %3702 = insertelement <2 x i32> undef, i32 %3700, i32 0		; visa id: 4798
  %3703 = insertelement <2 x i32> %3702, i32 %3701, i32 1		; visa id: 4799
  %3704 = bitcast <2 x i32> %3703 to i64		; visa id: 4800
  %3705 = shl i64 %3704, 1		; visa id: 4804
  %3706 = add i64 %.in401, %3705		; visa id: 4805
  %3707 = ashr i64 %3690, 31		; visa id: 4806
  %3708 = bitcast i64 %3707 to <2 x i32>		; visa id: 4807
  %3709 = extractelement <2 x i32> %3708, i32 0		; visa id: 4811
  %3710 = extractelement <2 x i32> %3708, i32 1		; visa id: 4811
  %3711 = and i32 %3709, -2		; visa id: 4811
  %3712 = insertelement <2 x i32> undef, i32 %3711, i32 0		; visa id: 4812
  %3713 = insertelement <2 x i32> %3712, i32 %3710, i32 1		; visa id: 4813
  %3714 = bitcast <2 x i32> %3713 to i64		; visa id: 4814
  %3715 = add i64 %3706, %3714		; visa id: 4818
  %3716 = inttoptr i64 %3715 to i16 addrspace(4)*		; visa id: 4819
  %3717 = addrspacecast i16 addrspace(4)* %3716 to i16 addrspace(1)*		; visa id: 4819
  %3718 = load i16, i16 addrspace(1)* %3717, align 2		; visa id: 4820
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4822
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4822
  %3719 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4822
  %3720 = insertelement <2 x i32> %3719, i32 %3671, i64 1		; visa id: 4823
  %3721 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4824
  store <2 x i32> %3720, <2 x i32>* %3721, align 4, !noalias !635		; visa id: 4824
  br label %._crit_edge280, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4826

._crit_edge280:                                   ; preds = %._crit_edge280.._crit_edge280_crit_edge, %3689
; BB336 :
  %3722 = phi i32 [ 0, %3689 ], [ %3731, %._crit_edge280.._crit_edge280_crit_edge ]
  %3723 = zext i32 %3722 to i64		; visa id: 4827
  %3724 = shl nuw nsw i64 %3723, 2		; visa id: 4828
  %3725 = add i64 %124, %3724		; visa id: 4829
  %3726 = inttoptr i64 %3725 to i32*		; visa id: 4830
  %3727 = load i32, i32* %3726, align 4, !noalias !635		; visa id: 4830
  %3728 = add i64 %119, %3724		; visa id: 4831
  %3729 = inttoptr i64 %3728 to i32*		; visa id: 4832
  store i32 %3727, i32* %3729, align 4, !alias.scope !635		; visa id: 4832
  %3730 = icmp eq i32 %3722, 0		; visa id: 4833
  br i1 %3730, label %._crit_edge280.._crit_edge280_crit_edge, label %3732, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4834

._crit_edge280.._crit_edge280_crit_edge:          ; preds = %._crit_edge280
; BB337 :
  %3731 = add nuw nsw i32 %3722, 1, !spirv.Decorations !631		; visa id: 4836
  br label %._crit_edge280, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4837

3732:                                             ; preds = %._crit_edge280
; BB338 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4839
  %3733 = load i64, i64* %120, align 8		; visa id: 4839
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4840
  %3734 = bitcast i64 %3733 to <2 x i32>		; visa id: 4840
  %3735 = extractelement <2 x i32> %3734, i32 0		; visa id: 4842
  %3736 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3735, i32 1
  %3737 = bitcast <2 x i32> %3736 to i64		; visa id: 4842
  %3738 = ashr exact i64 %3737, 32		; visa id: 4843
  %3739 = bitcast i64 %3738 to <2 x i32>		; visa id: 4844
  %3740 = extractelement <2 x i32> %3739, i32 0		; visa id: 4848
  %3741 = extractelement <2 x i32> %3739, i32 1		; visa id: 4848
  %3742 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3740, i32 %3741, i32 %44, i32 %45)
  %3743 = extractvalue { i32, i32 } %3742, 0		; visa id: 4848
  %3744 = extractvalue { i32, i32 } %3742, 1		; visa id: 4848
  %3745 = insertelement <2 x i32> undef, i32 %3743, i32 0		; visa id: 4855
  %3746 = insertelement <2 x i32> %3745, i32 %3744, i32 1		; visa id: 4856
  %3747 = bitcast <2 x i32> %3746 to i64		; visa id: 4857
  %3748 = shl i64 %3747, 1		; visa id: 4861
  %3749 = add i64 %.in400, %3748		; visa id: 4862
  %3750 = ashr i64 %3733, 31		; visa id: 4863
  %3751 = bitcast i64 %3750 to <2 x i32>		; visa id: 4864
  %3752 = extractelement <2 x i32> %3751, i32 0		; visa id: 4868
  %3753 = extractelement <2 x i32> %3751, i32 1		; visa id: 4868
  %3754 = and i32 %3752, -2		; visa id: 4868
  %3755 = insertelement <2 x i32> undef, i32 %3754, i32 0		; visa id: 4869
  %3756 = insertelement <2 x i32> %3755, i32 %3753, i32 1		; visa id: 4870
  %3757 = bitcast <2 x i32> %3756 to i64		; visa id: 4871
  %3758 = add i64 %3749, %3757		; visa id: 4875
  %3759 = inttoptr i64 %3758 to i16 addrspace(4)*		; visa id: 4876
  %3760 = addrspacecast i16 addrspace(4)* %3759 to i16 addrspace(1)*		; visa id: 4876
  %3761 = load i16, i16 addrspace(1)* %3760, align 2		; visa id: 4877
  %3762 = zext i16 %3718 to i32		; visa id: 4879
  %3763 = shl nuw i32 %3762, 16, !spirv.Decorations !639		; visa id: 4880
  %3764 = bitcast i32 %3763 to float
  %3765 = zext i16 %3761 to i32		; visa id: 4881
  %3766 = shl nuw i32 %3765, 16, !spirv.Decorations !639		; visa id: 4882
  %3767 = bitcast i32 %3766 to float
  %3768 = fmul reassoc nsz arcp contract float %3764, %3767, !spirv.Decorations !618
  %3769 = fadd reassoc nsz arcp contract float %3768, %.sroa.38.1, !spirv.Decorations !618		; visa id: 4883
  br label %._crit_edge.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4884

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %3732
; BB339 :
  %.sroa.38.2 = phi float [ %3769, %3732 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  %3770 = icmp slt i32 %230, %const_reg_dword
  %3771 = icmp slt i32 %3671, %const_reg_dword1		; visa id: 4885
  %3772 = and i1 %3770, %3771		; visa id: 4886
  br i1 %3772, label %3773, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 4888

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
; BB:
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3773:                                             ; preds = %._crit_edge.9
; BB341 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 4890
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 4890
  %3774 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 4890
  %3775 = insertelement <2 x i32> %3774, i32 %113, i64 1		; visa id: 4891
  %3776 = inttoptr i64 %133 to <2 x i32>*		; visa id: 4892
  store <2 x i32> %3775, <2 x i32>* %3776, align 4, !noalias !625		; visa id: 4892
  br label %._crit_edge281, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4894

._crit_edge281:                                   ; preds = %._crit_edge281.._crit_edge281_crit_edge, %3773
; BB342 :
  %3777 = phi i32 [ 0, %3773 ], [ %3786, %._crit_edge281.._crit_edge281_crit_edge ]
  %3778 = zext i32 %3777 to i64		; visa id: 4895
  %3779 = shl nuw nsw i64 %3778, 2		; visa id: 4896
  %3780 = add i64 %133, %3779		; visa id: 4897
  %3781 = inttoptr i64 %3780 to i32*		; visa id: 4898
  %3782 = load i32, i32* %3781, align 4, !noalias !625		; visa id: 4898
  %3783 = add i64 %128, %3779		; visa id: 4899
  %3784 = inttoptr i64 %3783 to i32*		; visa id: 4900
  store i32 %3782, i32* %3784, align 4, !alias.scope !625		; visa id: 4900
  %3785 = icmp eq i32 %3777, 0		; visa id: 4901
  br i1 %3785, label %._crit_edge281.._crit_edge281_crit_edge, label %3787, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4902

._crit_edge281.._crit_edge281_crit_edge:          ; preds = %._crit_edge281
; BB343 :
  %3786 = add nuw nsw i32 %3777, 1, !spirv.Decorations !631		; visa id: 4904
  br label %._crit_edge281, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4905

3787:                                             ; preds = %._crit_edge281
; BB344 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 4907
  %3788 = load i64, i64* %129, align 8		; visa id: 4907
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 4908
  %3789 = bitcast i64 %3788 to <2 x i32>		; visa id: 4908
  %3790 = extractelement <2 x i32> %3789, i32 0		; visa id: 4910
  %3791 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3790, i32 1
  %3792 = bitcast <2 x i32> %3791 to i64		; visa id: 4910
  %3793 = ashr exact i64 %3792, 32		; visa id: 4911
  %3794 = bitcast i64 %3793 to <2 x i32>		; visa id: 4912
  %3795 = extractelement <2 x i32> %3794, i32 0		; visa id: 4916
  %3796 = extractelement <2 x i32> %3794, i32 1		; visa id: 4916
  %3797 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3795, i32 %3796, i32 %41, i32 %42)
  %3798 = extractvalue { i32, i32 } %3797, 0		; visa id: 4916
  %3799 = extractvalue { i32, i32 } %3797, 1		; visa id: 4916
  %3800 = insertelement <2 x i32> undef, i32 %3798, i32 0		; visa id: 4923
  %3801 = insertelement <2 x i32> %3800, i32 %3799, i32 1		; visa id: 4924
  %3802 = bitcast <2 x i32> %3801 to i64		; visa id: 4925
  %3803 = shl i64 %3802, 1		; visa id: 4929
  %3804 = add i64 %.in401, %3803		; visa id: 4930
  %3805 = ashr i64 %3788, 31		; visa id: 4931
  %3806 = bitcast i64 %3805 to <2 x i32>		; visa id: 4932
  %3807 = extractelement <2 x i32> %3806, i32 0		; visa id: 4936
  %3808 = extractelement <2 x i32> %3806, i32 1		; visa id: 4936
  %3809 = and i32 %3807, -2		; visa id: 4936
  %3810 = insertelement <2 x i32> undef, i32 %3809, i32 0		; visa id: 4937
  %3811 = insertelement <2 x i32> %3810, i32 %3808, i32 1		; visa id: 4938
  %3812 = bitcast <2 x i32> %3811 to i64		; visa id: 4939
  %3813 = add i64 %3804, %3812		; visa id: 4943
  %3814 = inttoptr i64 %3813 to i16 addrspace(4)*		; visa id: 4944
  %3815 = addrspacecast i16 addrspace(4)* %3814 to i16 addrspace(1)*		; visa id: 4944
  %3816 = load i16, i16 addrspace(1)* %3815, align 2		; visa id: 4945
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 4947
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 4947
  %3817 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 4947
  %3818 = insertelement <2 x i32> %3817, i32 %3671, i64 1		; visa id: 4948
  %3819 = inttoptr i64 %124 to <2 x i32>*		; visa id: 4949
  store <2 x i32> %3818, <2 x i32>* %3819, align 4, !noalias !635		; visa id: 4949
  br label %._crit_edge282, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 4951

._crit_edge282:                                   ; preds = %._crit_edge282.._crit_edge282_crit_edge, %3787
; BB345 :
  %3820 = phi i32 [ 0, %3787 ], [ %3829, %._crit_edge282.._crit_edge282_crit_edge ]
  %3821 = zext i32 %3820 to i64		; visa id: 4952
  %3822 = shl nuw nsw i64 %3821, 2		; visa id: 4953
  %3823 = add i64 %124, %3822		; visa id: 4954
  %3824 = inttoptr i64 %3823 to i32*		; visa id: 4955
  %3825 = load i32, i32* %3824, align 4, !noalias !635		; visa id: 4955
  %3826 = add i64 %119, %3822		; visa id: 4956
  %3827 = inttoptr i64 %3826 to i32*		; visa id: 4957
  store i32 %3825, i32* %3827, align 4, !alias.scope !635		; visa id: 4957
  %3828 = icmp eq i32 %3820, 0		; visa id: 4958
  br i1 %3828, label %._crit_edge282.._crit_edge282_crit_edge, label %3830, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 4959

._crit_edge282.._crit_edge282_crit_edge:          ; preds = %._crit_edge282
; BB346 :
  %3829 = add nuw nsw i32 %3820, 1, !spirv.Decorations !631		; visa id: 4961
  br label %._crit_edge282, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 4962

3830:                                             ; preds = %._crit_edge282
; BB347 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 4964
  %3831 = load i64, i64* %120, align 8		; visa id: 4964
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 4965
  %3832 = bitcast i64 %3831 to <2 x i32>		; visa id: 4965
  %3833 = extractelement <2 x i32> %3832, i32 0		; visa id: 4967
  %3834 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3833, i32 1
  %3835 = bitcast <2 x i32> %3834 to i64		; visa id: 4967
  %3836 = ashr exact i64 %3835, 32		; visa id: 4968
  %3837 = bitcast i64 %3836 to <2 x i32>		; visa id: 4969
  %3838 = extractelement <2 x i32> %3837, i32 0		; visa id: 4973
  %3839 = extractelement <2 x i32> %3837, i32 1		; visa id: 4973
  %3840 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3838, i32 %3839, i32 %44, i32 %45)
  %3841 = extractvalue { i32, i32 } %3840, 0		; visa id: 4973
  %3842 = extractvalue { i32, i32 } %3840, 1		; visa id: 4973
  %3843 = insertelement <2 x i32> undef, i32 %3841, i32 0		; visa id: 4980
  %3844 = insertelement <2 x i32> %3843, i32 %3842, i32 1		; visa id: 4981
  %3845 = bitcast <2 x i32> %3844 to i64		; visa id: 4982
  %3846 = shl i64 %3845, 1		; visa id: 4986
  %3847 = add i64 %.in400, %3846		; visa id: 4987
  %3848 = ashr i64 %3831, 31		; visa id: 4988
  %3849 = bitcast i64 %3848 to <2 x i32>		; visa id: 4989
  %3850 = extractelement <2 x i32> %3849, i32 0		; visa id: 4993
  %3851 = extractelement <2 x i32> %3849, i32 1		; visa id: 4993
  %3852 = and i32 %3850, -2		; visa id: 4993
  %3853 = insertelement <2 x i32> undef, i32 %3852, i32 0		; visa id: 4994
  %3854 = insertelement <2 x i32> %3853, i32 %3851, i32 1		; visa id: 4995
  %3855 = bitcast <2 x i32> %3854 to i64		; visa id: 4996
  %3856 = add i64 %3847, %3855		; visa id: 5000
  %3857 = inttoptr i64 %3856 to i16 addrspace(4)*		; visa id: 5001
  %3858 = addrspacecast i16 addrspace(4)* %3857 to i16 addrspace(1)*		; visa id: 5001
  %3859 = load i16, i16 addrspace(1)* %3858, align 2		; visa id: 5002
  %3860 = zext i16 %3816 to i32		; visa id: 5004
  %3861 = shl nuw i32 %3860, 16, !spirv.Decorations !639		; visa id: 5005
  %3862 = bitcast i32 %3861 to float
  %3863 = zext i16 %3859 to i32		; visa id: 5006
  %3864 = shl nuw i32 %3863, 16, !spirv.Decorations !639		; visa id: 5007
  %3865 = bitcast i32 %3864 to float
  %3866 = fmul reassoc nsz arcp contract float %3862, %3865, !spirv.Decorations !618
  %3867 = fadd reassoc nsz arcp contract float %3866, %.sroa.102.1, !spirv.Decorations !618		; visa id: 5008
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5009

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %3830
; BB348 :
  %.sroa.102.2 = phi float [ %3867, %3830 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  %3868 = icmp slt i32 %329, %const_reg_dword
  %3869 = icmp slt i32 %3671, %const_reg_dword1		; visa id: 5010
  %3870 = and i1 %3868, %3869		; visa id: 5011
  br i1 %3870, label %3871, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5013

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
; BB:
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3871:                                             ; preds = %._crit_edge.1.9
; BB350 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5015
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5015
  %3872 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 5015
  %3873 = insertelement <2 x i32> %3872, i32 %113, i64 1		; visa id: 5016
  %3874 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5017
  store <2 x i32> %3873, <2 x i32>* %3874, align 4, !noalias !625		; visa id: 5017
  br label %._crit_edge283, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5019

._crit_edge283:                                   ; preds = %._crit_edge283.._crit_edge283_crit_edge, %3871
; BB351 :
  %3875 = phi i32 [ 0, %3871 ], [ %3884, %._crit_edge283.._crit_edge283_crit_edge ]
  %3876 = zext i32 %3875 to i64		; visa id: 5020
  %3877 = shl nuw nsw i64 %3876, 2		; visa id: 5021
  %3878 = add i64 %133, %3877		; visa id: 5022
  %3879 = inttoptr i64 %3878 to i32*		; visa id: 5023
  %3880 = load i32, i32* %3879, align 4, !noalias !625		; visa id: 5023
  %3881 = add i64 %128, %3877		; visa id: 5024
  %3882 = inttoptr i64 %3881 to i32*		; visa id: 5025
  store i32 %3880, i32* %3882, align 4, !alias.scope !625		; visa id: 5025
  %3883 = icmp eq i32 %3875, 0		; visa id: 5026
  br i1 %3883, label %._crit_edge283.._crit_edge283_crit_edge, label %3885, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5027

._crit_edge283.._crit_edge283_crit_edge:          ; preds = %._crit_edge283
; BB352 :
  %3884 = add nuw nsw i32 %3875, 1, !spirv.Decorations !631		; visa id: 5029
  br label %._crit_edge283, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5030

3885:                                             ; preds = %._crit_edge283
; BB353 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5032
  %3886 = load i64, i64* %129, align 8		; visa id: 5032
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5033
  %3887 = bitcast i64 %3886 to <2 x i32>		; visa id: 5033
  %3888 = extractelement <2 x i32> %3887, i32 0		; visa id: 5035
  %3889 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3888, i32 1
  %3890 = bitcast <2 x i32> %3889 to i64		; visa id: 5035
  %3891 = ashr exact i64 %3890, 32		; visa id: 5036
  %3892 = bitcast i64 %3891 to <2 x i32>		; visa id: 5037
  %3893 = extractelement <2 x i32> %3892, i32 0		; visa id: 5041
  %3894 = extractelement <2 x i32> %3892, i32 1		; visa id: 5041
  %3895 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3893, i32 %3894, i32 %41, i32 %42)
  %3896 = extractvalue { i32, i32 } %3895, 0		; visa id: 5041
  %3897 = extractvalue { i32, i32 } %3895, 1		; visa id: 5041
  %3898 = insertelement <2 x i32> undef, i32 %3896, i32 0		; visa id: 5048
  %3899 = insertelement <2 x i32> %3898, i32 %3897, i32 1		; visa id: 5049
  %3900 = bitcast <2 x i32> %3899 to i64		; visa id: 5050
  %3901 = shl i64 %3900, 1		; visa id: 5054
  %3902 = add i64 %.in401, %3901		; visa id: 5055
  %3903 = ashr i64 %3886, 31		; visa id: 5056
  %3904 = bitcast i64 %3903 to <2 x i32>		; visa id: 5057
  %3905 = extractelement <2 x i32> %3904, i32 0		; visa id: 5061
  %3906 = extractelement <2 x i32> %3904, i32 1		; visa id: 5061
  %3907 = and i32 %3905, -2		; visa id: 5061
  %3908 = insertelement <2 x i32> undef, i32 %3907, i32 0		; visa id: 5062
  %3909 = insertelement <2 x i32> %3908, i32 %3906, i32 1		; visa id: 5063
  %3910 = bitcast <2 x i32> %3909 to i64		; visa id: 5064
  %3911 = add i64 %3902, %3910		; visa id: 5068
  %3912 = inttoptr i64 %3911 to i16 addrspace(4)*		; visa id: 5069
  %3913 = addrspacecast i16 addrspace(4)* %3912 to i16 addrspace(1)*		; visa id: 5069
  %3914 = load i16, i16 addrspace(1)* %3913, align 2		; visa id: 5070
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5072
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5072
  %3915 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5072
  %3916 = insertelement <2 x i32> %3915, i32 %3671, i64 1		; visa id: 5073
  %3917 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5074
  store <2 x i32> %3916, <2 x i32>* %3917, align 4, !noalias !635		; visa id: 5074
  br label %._crit_edge284, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5076

._crit_edge284:                                   ; preds = %._crit_edge284.._crit_edge284_crit_edge, %3885
; BB354 :
  %3918 = phi i32 [ 0, %3885 ], [ %3927, %._crit_edge284.._crit_edge284_crit_edge ]
  %3919 = zext i32 %3918 to i64		; visa id: 5077
  %3920 = shl nuw nsw i64 %3919, 2		; visa id: 5078
  %3921 = add i64 %124, %3920		; visa id: 5079
  %3922 = inttoptr i64 %3921 to i32*		; visa id: 5080
  %3923 = load i32, i32* %3922, align 4, !noalias !635		; visa id: 5080
  %3924 = add i64 %119, %3920		; visa id: 5081
  %3925 = inttoptr i64 %3924 to i32*		; visa id: 5082
  store i32 %3923, i32* %3925, align 4, !alias.scope !635		; visa id: 5082
  %3926 = icmp eq i32 %3918, 0		; visa id: 5083
  br i1 %3926, label %._crit_edge284.._crit_edge284_crit_edge, label %3928, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5084

._crit_edge284.._crit_edge284_crit_edge:          ; preds = %._crit_edge284
; BB355 :
  %3927 = add nuw nsw i32 %3918, 1, !spirv.Decorations !631		; visa id: 5086
  br label %._crit_edge284, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5087

3928:                                             ; preds = %._crit_edge284
; BB356 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5089
  %3929 = load i64, i64* %120, align 8		; visa id: 5089
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5090
  %3930 = bitcast i64 %3929 to <2 x i32>		; visa id: 5090
  %3931 = extractelement <2 x i32> %3930, i32 0		; visa id: 5092
  %3932 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3931, i32 1
  %3933 = bitcast <2 x i32> %3932 to i64		; visa id: 5092
  %3934 = ashr exact i64 %3933, 32		; visa id: 5093
  %3935 = bitcast i64 %3934 to <2 x i32>		; visa id: 5094
  %3936 = extractelement <2 x i32> %3935, i32 0		; visa id: 5098
  %3937 = extractelement <2 x i32> %3935, i32 1		; visa id: 5098
  %3938 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3936, i32 %3937, i32 %44, i32 %45)
  %3939 = extractvalue { i32, i32 } %3938, 0		; visa id: 5098
  %3940 = extractvalue { i32, i32 } %3938, 1		; visa id: 5098
  %3941 = insertelement <2 x i32> undef, i32 %3939, i32 0		; visa id: 5105
  %3942 = insertelement <2 x i32> %3941, i32 %3940, i32 1		; visa id: 5106
  %3943 = bitcast <2 x i32> %3942 to i64		; visa id: 5107
  %3944 = shl i64 %3943, 1		; visa id: 5111
  %3945 = add i64 %.in400, %3944		; visa id: 5112
  %3946 = ashr i64 %3929, 31		; visa id: 5113
  %3947 = bitcast i64 %3946 to <2 x i32>		; visa id: 5114
  %3948 = extractelement <2 x i32> %3947, i32 0		; visa id: 5118
  %3949 = extractelement <2 x i32> %3947, i32 1		; visa id: 5118
  %3950 = and i32 %3948, -2		; visa id: 5118
  %3951 = insertelement <2 x i32> undef, i32 %3950, i32 0		; visa id: 5119
  %3952 = insertelement <2 x i32> %3951, i32 %3949, i32 1		; visa id: 5120
  %3953 = bitcast <2 x i32> %3952 to i64		; visa id: 5121
  %3954 = add i64 %3945, %3953		; visa id: 5125
  %3955 = inttoptr i64 %3954 to i16 addrspace(4)*		; visa id: 5126
  %3956 = addrspacecast i16 addrspace(4)* %3955 to i16 addrspace(1)*		; visa id: 5126
  %3957 = load i16, i16 addrspace(1)* %3956, align 2		; visa id: 5127
  %3958 = zext i16 %3914 to i32		; visa id: 5129
  %3959 = shl nuw i32 %3958, 16, !spirv.Decorations !639		; visa id: 5130
  %3960 = bitcast i32 %3959 to float
  %3961 = zext i16 %3957 to i32		; visa id: 5131
  %3962 = shl nuw i32 %3961, 16, !spirv.Decorations !639		; visa id: 5132
  %3963 = bitcast i32 %3962 to float
  %3964 = fmul reassoc nsz arcp contract float %3960, %3963, !spirv.Decorations !618
  %3965 = fadd reassoc nsz arcp contract float %3964, %.sroa.166.1, !spirv.Decorations !618		; visa id: 5133
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5134

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %3928
; BB357 :
  %.sroa.166.2 = phi float [ %3965, %3928 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  %3966 = icmp slt i32 %428, %const_reg_dword
  %3967 = icmp slt i32 %3671, %const_reg_dword1		; visa id: 5135
  %3968 = and i1 %3966, %3967		; visa id: 5136
  br i1 %3968, label %3969, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5138

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
; BB:
  br label %.preheader.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

3969:                                             ; preds = %._crit_edge.2.9
; BB359 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5140
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5140
  %3970 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 5140
  %3971 = insertelement <2 x i32> %3970, i32 %113, i64 1		; visa id: 5141
  %3972 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5142
  store <2 x i32> %3971, <2 x i32>* %3972, align 4, !noalias !625		; visa id: 5142
  br label %._crit_edge285, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5144

._crit_edge285:                                   ; preds = %._crit_edge285.._crit_edge285_crit_edge, %3969
; BB360 :
  %3973 = phi i32 [ 0, %3969 ], [ %3982, %._crit_edge285.._crit_edge285_crit_edge ]
  %3974 = zext i32 %3973 to i64		; visa id: 5145
  %3975 = shl nuw nsw i64 %3974, 2		; visa id: 5146
  %3976 = add i64 %133, %3975		; visa id: 5147
  %3977 = inttoptr i64 %3976 to i32*		; visa id: 5148
  %3978 = load i32, i32* %3977, align 4, !noalias !625		; visa id: 5148
  %3979 = add i64 %128, %3975		; visa id: 5149
  %3980 = inttoptr i64 %3979 to i32*		; visa id: 5150
  store i32 %3978, i32* %3980, align 4, !alias.scope !625		; visa id: 5150
  %3981 = icmp eq i32 %3973, 0		; visa id: 5151
  br i1 %3981, label %._crit_edge285.._crit_edge285_crit_edge, label %3983, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5152

._crit_edge285.._crit_edge285_crit_edge:          ; preds = %._crit_edge285
; BB361 :
  %3982 = add nuw nsw i32 %3973, 1, !spirv.Decorations !631		; visa id: 5154
  br label %._crit_edge285, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5155

3983:                                             ; preds = %._crit_edge285
; BB362 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5157
  %3984 = load i64, i64* %129, align 8		; visa id: 5157
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5158
  %3985 = bitcast i64 %3984 to <2 x i32>		; visa id: 5158
  %3986 = extractelement <2 x i32> %3985, i32 0		; visa id: 5160
  %3987 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %3986, i32 1
  %3988 = bitcast <2 x i32> %3987 to i64		; visa id: 5160
  %3989 = ashr exact i64 %3988, 32		; visa id: 5161
  %3990 = bitcast i64 %3989 to <2 x i32>		; visa id: 5162
  %3991 = extractelement <2 x i32> %3990, i32 0		; visa id: 5166
  %3992 = extractelement <2 x i32> %3990, i32 1		; visa id: 5166
  %3993 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %3991, i32 %3992, i32 %41, i32 %42)
  %3994 = extractvalue { i32, i32 } %3993, 0		; visa id: 5166
  %3995 = extractvalue { i32, i32 } %3993, 1		; visa id: 5166
  %3996 = insertelement <2 x i32> undef, i32 %3994, i32 0		; visa id: 5173
  %3997 = insertelement <2 x i32> %3996, i32 %3995, i32 1		; visa id: 5174
  %3998 = bitcast <2 x i32> %3997 to i64		; visa id: 5175
  %3999 = shl i64 %3998, 1		; visa id: 5179
  %4000 = add i64 %.in401, %3999		; visa id: 5180
  %4001 = ashr i64 %3984, 31		; visa id: 5181
  %4002 = bitcast i64 %4001 to <2 x i32>		; visa id: 5182
  %4003 = extractelement <2 x i32> %4002, i32 0		; visa id: 5186
  %4004 = extractelement <2 x i32> %4002, i32 1		; visa id: 5186
  %4005 = and i32 %4003, -2		; visa id: 5186
  %4006 = insertelement <2 x i32> undef, i32 %4005, i32 0		; visa id: 5187
  %4007 = insertelement <2 x i32> %4006, i32 %4004, i32 1		; visa id: 5188
  %4008 = bitcast <2 x i32> %4007 to i64		; visa id: 5189
  %4009 = add i64 %4000, %4008		; visa id: 5193
  %4010 = inttoptr i64 %4009 to i16 addrspace(4)*		; visa id: 5194
  %4011 = addrspacecast i16 addrspace(4)* %4010 to i16 addrspace(1)*		; visa id: 5194
  %4012 = load i16, i16 addrspace(1)* %4011, align 2		; visa id: 5195
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5197
  %4013 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5197
  %4014 = insertelement <2 x i32> %4013, i32 %3671, i64 1		; visa id: 5198
  %4015 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5199
  store <2 x i32> %4014, <2 x i32>* %4015, align 4, !noalias !635		; visa id: 5199
  br label %._crit_edge286, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5201

._crit_edge286:                                   ; preds = %._crit_edge286.._crit_edge286_crit_edge, %3983
; BB363 :
  %4016 = phi i32 [ 0, %3983 ], [ %4025, %._crit_edge286.._crit_edge286_crit_edge ]
  %4017 = zext i32 %4016 to i64		; visa id: 5202
  %4018 = shl nuw nsw i64 %4017, 2		; visa id: 5203
  %4019 = add i64 %124, %4018		; visa id: 5204
  %4020 = inttoptr i64 %4019 to i32*		; visa id: 5205
  %4021 = load i32, i32* %4020, align 4, !noalias !635		; visa id: 5205
  %4022 = add i64 %119, %4018		; visa id: 5206
  %4023 = inttoptr i64 %4022 to i32*		; visa id: 5207
  store i32 %4021, i32* %4023, align 4, !alias.scope !635		; visa id: 5207
  %4024 = icmp eq i32 %4016, 0		; visa id: 5208
  br i1 %4024, label %._crit_edge286.._crit_edge286_crit_edge, label %4026, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5209

._crit_edge286.._crit_edge286_crit_edge:          ; preds = %._crit_edge286
; BB364 :
  %4025 = add nuw nsw i32 %4016, 1, !spirv.Decorations !631		; visa id: 5211
  br label %._crit_edge286, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5212

4026:                                             ; preds = %._crit_edge286
; BB365 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5214
  %4027 = load i64, i64* %120, align 8		; visa id: 5214
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5215
  %4028 = bitcast i64 %4027 to <2 x i32>		; visa id: 5215
  %4029 = extractelement <2 x i32> %4028, i32 0		; visa id: 5217
  %4030 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4029, i32 1
  %4031 = bitcast <2 x i32> %4030 to i64		; visa id: 5217
  %4032 = ashr exact i64 %4031, 32		; visa id: 5218
  %4033 = bitcast i64 %4032 to <2 x i32>		; visa id: 5219
  %4034 = extractelement <2 x i32> %4033, i32 0		; visa id: 5223
  %4035 = extractelement <2 x i32> %4033, i32 1		; visa id: 5223
  %4036 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4034, i32 %4035, i32 %44, i32 %45)
  %4037 = extractvalue { i32, i32 } %4036, 0		; visa id: 5223
  %4038 = extractvalue { i32, i32 } %4036, 1		; visa id: 5223
  %4039 = insertelement <2 x i32> undef, i32 %4037, i32 0		; visa id: 5230
  %4040 = insertelement <2 x i32> %4039, i32 %4038, i32 1		; visa id: 5231
  %4041 = bitcast <2 x i32> %4040 to i64		; visa id: 5232
  %4042 = shl i64 %4041, 1		; visa id: 5236
  %4043 = add i64 %.in400, %4042		; visa id: 5237
  %4044 = ashr i64 %4027, 31		; visa id: 5238
  %4045 = bitcast i64 %4044 to <2 x i32>		; visa id: 5239
  %4046 = extractelement <2 x i32> %4045, i32 0		; visa id: 5243
  %4047 = extractelement <2 x i32> %4045, i32 1		; visa id: 5243
  %4048 = and i32 %4046, -2		; visa id: 5243
  %4049 = insertelement <2 x i32> undef, i32 %4048, i32 0		; visa id: 5244
  %4050 = insertelement <2 x i32> %4049, i32 %4047, i32 1		; visa id: 5245
  %4051 = bitcast <2 x i32> %4050 to i64		; visa id: 5246
  %4052 = add i64 %4043, %4051		; visa id: 5250
  %4053 = inttoptr i64 %4052 to i16 addrspace(4)*		; visa id: 5251
  %4054 = addrspacecast i16 addrspace(4)* %4053 to i16 addrspace(1)*		; visa id: 5251
  %4055 = load i16, i16 addrspace(1)* %4054, align 2		; visa id: 5252
  %4056 = zext i16 %4012 to i32		; visa id: 5254
  %4057 = shl nuw i32 %4056, 16, !spirv.Decorations !639		; visa id: 5255
  %4058 = bitcast i32 %4057 to float
  %4059 = zext i16 %4055 to i32		; visa id: 5256
  %4060 = shl nuw i32 %4059, 16, !spirv.Decorations !639		; visa id: 5257
  %4061 = bitcast i32 %4060 to float
  %4062 = fmul reassoc nsz arcp contract float %4058, %4061, !spirv.Decorations !618
  %4063 = fadd reassoc nsz arcp contract float %4062, %.sroa.230.1, !spirv.Decorations !618		; visa id: 5258
  br label %.preheader.9, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5259

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %4026
; BB366 :
  %.sroa.230.2 = phi float [ %4063, %4026 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %4064 = add i32 %69, 10		; visa id: 5260
  %4065 = icmp slt i32 %4064, %const_reg_dword1		; visa id: 5261
  %4066 = icmp slt i32 %65, %const_reg_dword
  %4067 = and i1 %4066, %4065		; visa id: 5262
  br i1 %4067, label %4068, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5264

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
; BB:
  br label %._crit_edge.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4068:                                             ; preds = %.preheader.9
; BB368 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5266
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5266
  %4069 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 5266
  %4070 = insertelement <2 x i32> %4069, i32 %113, i64 1		; visa id: 5267
  %4071 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5268
  store <2 x i32> %4070, <2 x i32>* %4071, align 4, !noalias !625		; visa id: 5268
  br label %._crit_edge287, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5270

._crit_edge287:                                   ; preds = %._crit_edge287.._crit_edge287_crit_edge, %4068
; BB369 :
  %4072 = phi i32 [ 0, %4068 ], [ %4081, %._crit_edge287.._crit_edge287_crit_edge ]
  %4073 = zext i32 %4072 to i64		; visa id: 5271
  %4074 = shl nuw nsw i64 %4073, 2		; visa id: 5272
  %4075 = add i64 %133, %4074		; visa id: 5273
  %4076 = inttoptr i64 %4075 to i32*		; visa id: 5274
  %4077 = load i32, i32* %4076, align 4, !noalias !625		; visa id: 5274
  %4078 = add i64 %128, %4074		; visa id: 5275
  %4079 = inttoptr i64 %4078 to i32*		; visa id: 5276
  store i32 %4077, i32* %4079, align 4, !alias.scope !625		; visa id: 5276
  %4080 = icmp eq i32 %4072, 0		; visa id: 5277
  br i1 %4080, label %._crit_edge287.._crit_edge287_crit_edge, label %4082, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5278

._crit_edge287.._crit_edge287_crit_edge:          ; preds = %._crit_edge287
; BB370 :
  %4081 = add nuw nsw i32 %4072, 1, !spirv.Decorations !631		; visa id: 5280
  br label %._crit_edge287, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5281

4082:                                             ; preds = %._crit_edge287
; BB371 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5283
  %4083 = load i64, i64* %129, align 8		; visa id: 5283
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5284
  %4084 = bitcast i64 %4083 to <2 x i32>		; visa id: 5284
  %4085 = extractelement <2 x i32> %4084, i32 0		; visa id: 5286
  %4086 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4085, i32 1
  %4087 = bitcast <2 x i32> %4086 to i64		; visa id: 5286
  %4088 = ashr exact i64 %4087, 32		; visa id: 5287
  %4089 = bitcast i64 %4088 to <2 x i32>		; visa id: 5288
  %4090 = extractelement <2 x i32> %4089, i32 0		; visa id: 5292
  %4091 = extractelement <2 x i32> %4089, i32 1		; visa id: 5292
  %4092 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4090, i32 %4091, i32 %41, i32 %42)
  %4093 = extractvalue { i32, i32 } %4092, 0		; visa id: 5292
  %4094 = extractvalue { i32, i32 } %4092, 1		; visa id: 5292
  %4095 = insertelement <2 x i32> undef, i32 %4093, i32 0		; visa id: 5299
  %4096 = insertelement <2 x i32> %4095, i32 %4094, i32 1		; visa id: 5300
  %4097 = bitcast <2 x i32> %4096 to i64		; visa id: 5301
  %4098 = shl i64 %4097, 1		; visa id: 5305
  %4099 = add i64 %.in401, %4098		; visa id: 5306
  %4100 = ashr i64 %4083, 31		; visa id: 5307
  %4101 = bitcast i64 %4100 to <2 x i32>		; visa id: 5308
  %4102 = extractelement <2 x i32> %4101, i32 0		; visa id: 5312
  %4103 = extractelement <2 x i32> %4101, i32 1		; visa id: 5312
  %4104 = and i32 %4102, -2		; visa id: 5312
  %4105 = insertelement <2 x i32> undef, i32 %4104, i32 0		; visa id: 5313
  %4106 = insertelement <2 x i32> %4105, i32 %4103, i32 1		; visa id: 5314
  %4107 = bitcast <2 x i32> %4106 to i64		; visa id: 5315
  %4108 = add i64 %4099, %4107		; visa id: 5319
  %4109 = inttoptr i64 %4108 to i16 addrspace(4)*		; visa id: 5320
  %4110 = addrspacecast i16 addrspace(4)* %4109 to i16 addrspace(1)*		; visa id: 5320
  %4111 = load i16, i16 addrspace(1)* %4110, align 2		; visa id: 5321
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5323
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5323
  %4112 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5323
  %4113 = insertelement <2 x i32> %4112, i32 %4064, i64 1		; visa id: 5324
  %4114 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5325
  store <2 x i32> %4113, <2 x i32>* %4114, align 4, !noalias !635		; visa id: 5325
  br label %._crit_edge288, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5327

._crit_edge288:                                   ; preds = %._crit_edge288.._crit_edge288_crit_edge, %4082
; BB372 :
  %4115 = phi i32 [ 0, %4082 ], [ %4124, %._crit_edge288.._crit_edge288_crit_edge ]
  %4116 = zext i32 %4115 to i64		; visa id: 5328
  %4117 = shl nuw nsw i64 %4116, 2		; visa id: 5329
  %4118 = add i64 %124, %4117		; visa id: 5330
  %4119 = inttoptr i64 %4118 to i32*		; visa id: 5331
  %4120 = load i32, i32* %4119, align 4, !noalias !635		; visa id: 5331
  %4121 = add i64 %119, %4117		; visa id: 5332
  %4122 = inttoptr i64 %4121 to i32*		; visa id: 5333
  store i32 %4120, i32* %4122, align 4, !alias.scope !635		; visa id: 5333
  %4123 = icmp eq i32 %4115, 0		; visa id: 5334
  br i1 %4123, label %._crit_edge288.._crit_edge288_crit_edge, label %4125, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5335

._crit_edge288.._crit_edge288_crit_edge:          ; preds = %._crit_edge288
; BB373 :
  %4124 = add nuw nsw i32 %4115, 1, !spirv.Decorations !631		; visa id: 5337
  br label %._crit_edge288, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5338

4125:                                             ; preds = %._crit_edge288
; BB374 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5340
  %4126 = load i64, i64* %120, align 8		; visa id: 5340
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5341
  %4127 = bitcast i64 %4126 to <2 x i32>		; visa id: 5341
  %4128 = extractelement <2 x i32> %4127, i32 0		; visa id: 5343
  %4129 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4128, i32 1
  %4130 = bitcast <2 x i32> %4129 to i64		; visa id: 5343
  %4131 = ashr exact i64 %4130, 32		; visa id: 5344
  %4132 = bitcast i64 %4131 to <2 x i32>		; visa id: 5345
  %4133 = extractelement <2 x i32> %4132, i32 0		; visa id: 5349
  %4134 = extractelement <2 x i32> %4132, i32 1		; visa id: 5349
  %4135 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4133, i32 %4134, i32 %44, i32 %45)
  %4136 = extractvalue { i32, i32 } %4135, 0		; visa id: 5349
  %4137 = extractvalue { i32, i32 } %4135, 1		; visa id: 5349
  %4138 = insertelement <2 x i32> undef, i32 %4136, i32 0		; visa id: 5356
  %4139 = insertelement <2 x i32> %4138, i32 %4137, i32 1		; visa id: 5357
  %4140 = bitcast <2 x i32> %4139 to i64		; visa id: 5358
  %4141 = shl i64 %4140, 1		; visa id: 5362
  %4142 = add i64 %.in400, %4141		; visa id: 5363
  %4143 = ashr i64 %4126, 31		; visa id: 5364
  %4144 = bitcast i64 %4143 to <2 x i32>		; visa id: 5365
  %4145 = extractelement <2 x i32> %4144, i32 0		; visa id: 5369
  %4146 = extractelement <2 x i32> %4144, i32 1		; visa id: 5369
  %4147 = and i32 %4145, -2		; visa id: 5369
  %4148 = insertelement <2 x i32> undef, i32 %4147, i32 0		; visa id: 5370
  %4149 = insertelement <2 x i32> %4148, i32 %4146, i32 1		; visa id: 5371
  %4150 = bitcast <2 x i32> %4149 to i64		; visa id: 5372
  %4151 = add i64 %4142, %4150		; visa id: 5376
  %4152 = inttoptr i64 %4151 to i16 addrspace(4)*		; visa id: 5377
  %4153 = addrspacecast i16 addrspace(4)* %4152 to i16 addrspace(1)*		; visa id: 5377
  %4154 = load i16, i16 addrspace(1)* %4153, align 2		; visa id: 5378
  %4155 = zext i16 %4111 to i32		; visa id: 5380
  %4156 = shl nuw i32 %4155, 16, !spirv.Decorations !639		; visa id: 5381
  %4157 = bitcast i32 %4156 to float
  %4158 = zext i16 %4154 to i32		; visa id: 5382
  %4159 = shl nuw i32 %4158, 16, !spirv.Decorations !639		; visa id: 5383
  %4160 = bitcast i32 %4159 to float
  %4161 = fmul reassoc nsz arcp contract float %4157, %4160, !spirv.Decorations !618
  %4162 = fadd reassoc nsz arcp contract float %4161, %.sroa.42.1, !spirv.Decorations !618		; visa id: 5384
  br label %._crit_edge.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5385

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %4125
; BB375 :
  %.sroa.42.2 = phi float [ %4162, %4125 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  %4163 = icmp slt i32 %230, %const_reg_dword
  %4164 = icmp slt i32 %4064, %const_reg_dword1		; visa id: 5386
  %4165 = and i1 %4163, %4164		; visa id: 5387
  br i1 %4165, label %4166, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5389

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
; BB:
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4166:                                             ; preds = %._crit_edge.10
; BB377 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5391
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5391
  %4167 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 5391
  %4168 = insertelement <2 x i32> %4167, i32 %113, i64 1		; visa id: 5392
  %4169 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5393
  store <2 x i32> %4168, <2 x i32>* %4169, align 4, !noalias !625		; visa id: 5393
  br label %._crit_edge289, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5395

._crit_edge289:                                   ; preds = %._crit_edge289.._crit_edge289_crit_edge, %4166
; BB378 :
  %4170 = phi i32 [ 0, %4166 ], [ %4179, %._crit_edge289.._crit_edge289_crit_edge ]
  %4171 = zext i32 %4170 to i64		; visa id: 5396
  %4172 = shl nuw nsw i64 %4171, 2		; visa id: 5397
  %4173 = add i64 %133, %4172		; visa id: 5398
  %4174 = inttoptr i64 %4173 to i32*		; visa id: 5399
  %4175 = load i32, i32* %4174, align 4, !noalias !625		; visa id: 5399
  %4176 = add i64 %128, %4172		; visa id: 5400
  %4177 = inttoptr i64 %4176 to i32*		; visa id: 5401
  store i32 %4175, i32* %4177, align 4, !alias.scope !625		; visa id: 5401
  %4178 = icmp eq i32 %4170, 0		; visa id: 5402
  br i1 %4178, label %._crit_edge289.._crit_edge289_crit_edge, label %4180, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5403

._crit_edge289.._crit_edge289_crit_edge:          ; preds = %._crit_edge289
; BB379 :
  %4179 = add nuw nsw i32 %4170, 1, !spirv.Decorations !631		; visa id: 5405
  br label %._crit_edge289, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5406

4180:                                             ; preds = %._crit_edge289
; BB380 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5408
  %4181 = load i64, i64* %129, align 8		; visa id: 5408
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5409
  %4182 = bitcast i64 %4181 to <2 x i32>		; visa id: 5409
  %4183 = extractelement <2 x i32> %4182, i32 0		; visa id: 5411
  %4184 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4183, i32 1
  %4185 = bitcast <2 x i32> %4184 to i64		; visa id: 5411
  %4186 = ashr exact i64 %4185, 32		; visa id: 5412
  %4187 = bitcast i64 %4186 to <2 x i32>		; visa id: 5413
  %4188 = extractelement <2 x i32> %4187, i32 0		; visa id: 5417
  %4189 = extractelement <2 x i32> %4187, i32 1		; visa id: 5417
  %4190 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4188, i32 %4189, i32 %41, i32 %42)
  %4191 = extractvalue { i32, i32 } %4190, 0		; visa id: 5417
  %4192 = extractvalue { i32, i32 } %4190, 1		; visa id: 5417
  %4193 = insertelement <2 x i32> undef, i32 %4191, i32 0		; visa id: 5424
  %4194 = insertelement <2 x i32> %4193, i32 %4192, i32 1		; visa id: 5425
  %4195 = bitcast <2 x i32> %4194 to i64		; visa id: 5426
  %4196 = shl i64 %4195, 1		; visa id: 5430
  %4197 = add i64 %.in401, %4196		; visa id: 5431
  %4198 = ashr i64 %4181, 31		; visa id: 5432
  %4199 = bitcast i64 %4198 to <2 x i32>		; visa id: 5433
  %4200 = extractelement <2 x i32> %4199, i32 0		; visa id: 5437
  %4201 = extractelement <2 x i32> %4199, i32 1		; visa id: 5437
  %4202 = and i32 %4200, -2		; visa id: 5437
  %4203 = insertelement <2 x i32> undef, i32 %4202, i32 0		; visa id: 5438
  %4204 = insertelement <2 x i32> %4203, i32 %4201, i32 1		; visa id: 5439
  %4205 = bitcast <2 x i32> %4204 to i64		; visa id: 5440
  %4206 = add i64 %4197, %4205		; visa id: 5444
  %4207 = inttoptr i64 %4206 to i16 addrspace(4)*		; visa id: 5445
  %4208 = addrspacecast i16 addrspace(4)* %4207 to i16 addrspace(1)*		; visa id: 5445
  %4209 = load i16, i16 addrspace(1)* %4208, align 2		; visa id: 5446
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5448
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5448
  %4210 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5448
  %4211 = insertelement <2 x i32> %4210, i32 %4064, i64 1		; visa id: 5449
  %4212 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5450
  store <2 x i32> %4211, <2 x i32>* %4212, align 4, !noalias !635		; visa id: 5450
  br label %._crit_edge290, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5452

._crit_edge290:                                   ; preds = %._crit_edge290.._crit_edge290_crit_edge, %4180
; BB381 :
  %4213 = phi i32 [ 0, %4180 ], [ %4222, %._crit_edge290.._crit_edge290_crit_edge ]
  %4214 = zext i32 %4213 to i64		; visa id: 5453
  %4215 = shl nuw nsw i64 %4214, 2		; visa id: 5454
  %4216 = add i64 %124, %4215		; visa id: 5455
  %4217 = inttoptr i64 %4216 to i32*		; visa id: 5456
  %4218 = load i32, i32* %4217, align 4, !noalias !635		; visa id: 5456
  %4219 = add i64 %119, %4215		; visa id: 5457
  %4220 = inttoptr i64 %4219 to i32*		; visa id: 5458
  store i32 %4218, i32* %4220, align 4, !alias.scope !635		; visa id: 5458
  %4221 = icmp eq i32 %4213, 0		; visa id: 5459
  br i1 %4221, label %._crit_edge290.._crit_edge290_crit_edge, label %4223, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5460

._crit_edge290.._crit_edge290_crit_edge:          ; preds = %._crit_edge290
; BB382 :
  %4222 = add nuw nsw i32 %4213, 1, !spirv.Decorations !631		; visa id: 5462
  br label %._crit_edge290, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5463

4223:                                             ; preds = %._crit_edge290
; BB383 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5465
  %4224 = load i64, i64* %120, align 8		; visa id: 5465
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5466
  %4225 = bitcast i64 %4224 to <2 x i32>		; visa id: 5466
  %4226 = extractelement <2 x i32> %4225, i32 0		; visa id: 5468
  %4227 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4226, i32 1
  %4228 = bitcast <2 x i32> %4227 to i64		; visa id: 5468
  %4229 = ashr exact i64 %4228, 32		; visa id: 5469
  %4230 = bitcast i64 %4229 to <2 x i32>		; visa id: 5470
  %4231 = extractelement <2 x i32> %4230, i32 0		; visa id: 5474
  %4232 = extractelement <2 x i32> %4230, i32 1		; visa id: 5474
  %4233 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4231, i32 %4232, i32 %44, i32 %45)
  %4234 = extractvalue { i32, i32 } %4233, 0		; visa id: 5474
  %4235 = extractvalue { i32, i32 } %4233, 1		; visa id: 5474
  %4236 = insertelement <2 x i32> undef, i32 %4234, i32 0		; visa id: 5481
  %4237 = insertelement <2 x i32> %4236, i32 %4235, i32 1		; visa id: 5482
  %4238 = bitcast <2 x i32> %4237 to i64		; visa id: 5483
  %4239 = shl i64 %4238, 1		; visa id: 5487
  %4240 = add i64 %.in400, %4239		; visa id: 5488
  %4241 = ashr i64 %4224, 31		; visa id: 5489
  %4242 = bitcast i64 %4241 to <2 x i32>		; visa id: 5490
  %4243 = extractelement <2 x i32> %4242, i32 0		; visa id: 5494
  %4244 = extractelement <2 x i32> %4242, i32 1		; visa id: 5494
  %4245 = and i32 %4243, -2		; visa id: 5494
  %4246 = insertelement <2 x i32> undef, i32 %4245, i32 0		; visa id: 5495
  %4247 = insertelement <2 x i32> %4246, i32 %4244, i32 1		; visa id: 5496
  %4248 = bitcast <2 x i32> %4247 to i64		; visa id: 5497
  %4249 = add i64 %4240, %4248		; visa id: 5501
  %4250 = inttoptr i64 %4249 to i16 addrspace(4)*		; visa id: 5502
  %4251 = addrspacecast i16 addrspace(4)* %4250 to i16 addrspace(1)*		; visa id: 5502
  %4252 = load i16, i16 addrspace(1)* %4251, align 2		; visa id: 5503
  %4253 = zext i16 %4209 to i32		; visa id: 5505
  %4254 = shl nuw i32 %4253, 16, !spirv.Decorations !639		; visa id: 5506
  %4255 = bitcast i32 %4254 to float
  %4256 = zext i16 %4252 to i32		; visa id: 5507
  %4257 = shl nuw i32 %4256, 16, !spirv.Decorations !639		; visa id: 5508
  %4258 = bitcast i32 %4257 to float
  %4259 = fmul reassoc nsz arcp contract float %4255, %4258, !spirv.Decorations !618
  %4260 = fadd reassoc nsz arcp contract float %4259, %.sroa.106.1, !spirv.Decorations !618		; visa id: 5509
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5510

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %4223
; BB384 :
  %.sroa.106.2 = phi float [ %4260, %4223 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  %4261 = icmp slt i32 %329, %const_reg_dword
  %4262 = icmp slt i32 %4064, %const_reg_dword1		; visa id: 5511
  %4263 = and i1 %4261, %4262		; visa id: 5512
  br i1 %4263, label %4264, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5514

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
; BB:
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4264:                                             ; preds = %._crit_edge.1.10
; BB386 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5516
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5516
  %4265 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 5516
  %4266 = insertelement <2 x i32> %4265, i32 %113, i64 1		; visa id: 5517
  %4267 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5518
  store <2 x i32> %4266, <2 x i32>* %4267, align 4, !noalias !625		; visa id: 5518
  br label %._crit_edge291, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5520

._crit_edge291:                                   ; preds = %._crit_edge291.._crit_edge291_crit_edge, %4264
; BB387 :
  %4268 = phi i32 [ 0, %4264 ], [ %4277, %._crit_edge291.._crit_edge291_crit_edge ]
  %4269 = zext i32 %4268 to i64		; visa id: 5521
  %4270 = shl nuw nsw i64 %4269, 2		; visa id: 5522
  %4271 = add i64 %133, %4270		; visa id: 5523
  %4272 = inttoptr i64 %4271 to i32*		; visa id: 5524
  %4273 = load i32, i32* %4272, align 4, !noalias !625		; visa id: 5524
  %4274 = add i64 %128, %4270		; visa id: 5525
  %4275 = inttoptr i64 %4274 to i32*		; visa id: 5526
  store i32 %4273, i32* %4275, align 4, !alias.scope !625		; visa id: 5526
  %4276 = icmp eq i32 %4268, 0		; visa id: 5527
  br i1 %4276, label %._crit_edge291.._crit_edge291_crit_edge, label %4278, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5528

._crit_edge291.._crit_edge291_crit_edge:          ; preds = %._crit_edge291
; BB388 :
  %4277 = add nuw nsw i32 %4268, 1, !spirv.Decorations !631		; visa id: 5530
  br label %._crit_edge291, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5531

4278:                                             ; preds = %._crit_edge291
; BB389 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5533
  %4279 = load i64, i64* %129, align 8		; visa id: 5533
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5534
  %4280 = bitcast i64 %4279 to <2 x i32>		; visa id: 5534
  %4281 = extractelement <2 x i32> %4280, i32 0		; visa id: 5536
  %4282 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4281, i32 1
  %4283 = bitcast <2 x i32> %4282 to i64		; visa id: 5536
  %4284 = ashr exact i64 %4283, 32		; visa id: 5537
  %4285 = bitcast i64 %4284 to <2 x i32>		; visa id: 5538
  %4286 = extractelement <2 x i32> %4285, i32 0		; visa id: 5542
  %4287 = extractelement <2 x i32> %4285, i32 1		; visa id: 5542
  %4288 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4286, i32 %4287, i32 %41, i32 %42)
  %4289 = extractvalue { i32, i32 } %4288, 0		; visa id: 5542
  %4290 = extractvalue { i32, i32 } %4288, 1		; visa id: 5542
  %4291 = insertelement <2 x i32> undef, i32 %4289, i32 0		; visa id: 5549
  %4292 = insertelement <2 x i32> %4291, i32 %4290, i32 1		; visa id: 5550
  %4293 = bitcast <2 x i32> %4292 to i64		; visa id: 5551
  %4294 = shl i64 %4293, 1		; visa id: 5555
  %4295 = add i64 %.in401, %4294		; visa id: 5556
  %4296 = ashr i64 %4279, 31		; visa id: 5557
  %4297 = bitcast i64 %4296 to <2 x i32>		; visa id: 5558
  %4298 = extractelement <2 x i32> %4297, i32 0		; visa id: 5562
  %4299 = extractelement <2 x i32> %4297, i32 1		; visa id: 5562
  %4300 = and i32 %4298, -2		; visa id: 5562
  %4301 = insertelement <2 x i32> undef, i32 %4300, i32 0		; visa id: 5563
  %4302 = insertelement <2 x i32> %4301, i32 %4299, i32 1		; visa id: 5564
  %4303 = bitcast <2 x i32> %4302 to i64		; visa id: 5565
  %4304 = add i64 %4295, %4303		; visa id: 5569
  %4305 = inttoptr i64 %4304 to i16 addrspace(4)*		; visa id: 5570
  %4306 = addrspacecast i16 addrspace(4)* %4305 to i16 addrspace(1)*		; visa id: 5570
  %4307 = load i16, i16 addrspace(1)* %4306, align 2		; visa id: 5571
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5573
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5573
  %4308 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5573
  %4309 = insertelement <2 x i32> %4308, i32 %4064, i64 1		; visa id: 5574
  %4310 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5575
  store <2 x i32> %4309, <2 x i32>* %4310, align 4, !noalias !635		; visa id: 5575
  br label %._crit_edge292, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5577

._crit_edge292:                                   ; preds = %._crit_edge292.._crit_edge292_crit_edge, %4278
; BB390 :
  %4311 = phi i32 [ 0, %4278 ], [ %4320, %._crit_edge292.._crit_edge292_crit_edge ]
  %4312 = zext i32 %4311 to i64		; visa id: 5578
  %4313 = shl nuw nsw i64 %4312, 2		; visa id: 5579
  %4314 = add i64 %124, %4313		; visa id: 5580
  %4315 = inttoptr i64 %4314 to i32*		; visa id: 5581
  %4316 = load i32, i32* %4315, align 4, !noalias !635		; visa id: 5581
  %4317 = add i64 %119, %4313		; visa id: 5582
  %4318 = inttoptr i64 %4317 to i32*		; visa id: 5583
  store i32 %4316, i32* %4318, align 4, !alias.scope !635		; visa id: 5583
  %4319 = icmp eq i32 %4311, 0		; visa id: 5584
  br i1 %4319, label %._crit_edge292.._crit_edge292_crit_edge, label %4321, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5585

._crit_edge292.._crit_edge292_crit_edge:          ; preds = %._crit_edge292
; BB391 :
  %4320 = add nuw nsw i32 %4311, 1, !spirv.Decorations !631		; visa id: 5587
  br label %._crit_edge292, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5588

4321:                                             ; preds = %._crit_edge292
; BB392 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5590
  %4322 = load i64, i64* %120, align 8		; visa id: 5590
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5591
  %4323 = bitcast i64 %4322 to <2 x i32>		; visa id: 5591
  %4324 = extractelement <2 x i32> %4323, i32 0		; visa id: 5593
  %4325 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4324, i32 1
  %4326 = bitcast <2 x i32> %4325 to i64		; visa id: 5593
  %4327 = ashr exact i64 %4326, 32		; visa id: 5594
  %4328 = bitcast i64 %4327 to <2 x i32>		; visa id: 5595
  %4329 = extractelement <2 x i32> %4328, i32 0		; visa id: 5599
  %4330 = extractelement <2 x i32> %4328, i32 1		; visa id: 5599
  %4331 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4329, i32 %4330, i32 %44, i32 %45)
  %4332 = extractvalue { i32, i32 } %4331, 0		; visa id: 5599
  %4333 = extractvalue { i32, i32 } %4331, 1		; visa id: 5599
  %4334 = insertelement <2 x i32> undef, i32 %4332, i32 0		; visa id: 5606
  %4335 = insertelement <2 x i32> %4334, i32 %4333, i32 1		; visa id: 5607
  %4336 = bitcast <2 x i32> %4335 to i64		; visa id: 5608
  %4337 = shl i64 %4336, 1		; visa id: 5612
  %4338 = add i64 %.in400, %4337		; visa id: 5613
  %4339 = ashr i64 %4322, 31		; visa id: 5614
  %4340 = bitcast i64 %4339 to <2 x i32>		; visa id: 5615
  %4341 = extractelement <2 x i32> %4340, i32 0		; visa id: 5619
  %4342 = extractelement <2 x i32> %4340, i32 1		; visa id: 5619
  %4343 = and i32 %4341, -2		; visa id: 5619
  %4344 = insertelement <2 x i32> undef, i32 %4343, i32 0		; visa id: 5620
  %4345 = insertelement <2 x i32> %4344, i32 %4342, i32 1		; visa id: 5621
  %4346 = bitcast <2 x i32> %4345 to i64		; visa id: 5622
  %4347 = add i64 %4338, %4346		; visa id: 5626
  %4348 = inttoptr i64 %4347 to i16 addrspace(4)*		; visa id: 5627
  %4349 = addrspacecast i16 addrspace(4)* %4348 to i16 addrspace(1)*		; visa id: 5627
  %4350 = load i16, i16 addrspace(1)* %4349, align 2		; visa id: 5628
  %4351 = zext i16 %4307 to i32		; visa id: 5630
  %4352 = shl nuw i32 %4351, 16, !spirv.Decorations !639		; visa id: 5631
  %4353 = bitcast i32 %4352 to float
  %4354 = zext i16 %4350 to i32		; visa id: 5632
  %4355 = shl nuw i32 %4354, 16, !spirv.Decorations !639		; visa id: 5633
  %4356 = bitcast i32 %4355 to float
  %4357 = fmul reassoc nsz arcp contract float %4353, %4356, !spirv.Decorations !618
  %4358 = fadd reassoc nsz arcp contract float %4357, %.sroa.170.1, !spirv.Decorations !618		; visa id: 5634
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5635

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %4321
; BB393 :
  %.sroa.170.2 = phi float [ %4358, %4321 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  %4359 = icmp slt i32 %428, %const_reg_dword
  %4360 = icmp slt i32 %4064, %const_reg_dword1		; visa id: 5636
  %4361 = and i1 %4359, %4360		; visa id: 5637
  br i1 %4361, label %4362, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5639

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
; BB:
  br label %.preheader.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4362:                                             ; preds = %._crit_edge.2.10
; BB395 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5641
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5641
  %4363 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 5641
  %4364 = insertelement <2 x i32> %4363, i32 %113, i64 1		; visa id: 5642
  %4365 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5643
  store <2 x i32> %4364, <2 x i32>* %4365, align 4, !noalias !625		; visa id: 5643
  br label %._crit_edge293, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5645

._crit_edge293:                                   ; preds = %._crit_edge293.._crit_edge293_crit_edge, %4362
; BB396 :
  %4366 = phi i32 [ 0, %4362 ], [ %4375, %._crit_edge293.._crit_edge293_crit_edge ]
  %4367 = zext i32 %4366 to i64		; visa id: 5646
  %4368 = shl nuw nsw i64 %4367, 2		; visa id: 5647
  %4369 = add i64 %133, %4368		; visa id: 5648
  %4370 = inttoptr i64 %4369 to i32*		; visa id: 5649
  %4371 = load i32, i32* %4370, align 4, !noalias !625		; visa id: 5649
  %4372 = add i64 %128, %4368		; visa id: 5650
  %4373 = inttoptr i64 %4372 to i32*		; visa id: 5651
  store i32 %4371, i32* %4373, align 4, !alias.scope !625		; visa id: 5651
  %4374 = icmp eq i32 %4366, 0		; visa id: 5652
  br i1 %4374, label %._crit_edge293.._crit_edge293_crit_edge, label %4376, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5653

._crit_edge293.._crit_edge293_crit_edge:          ; preds = %._crit_edge293
; BB397 :
  %4375 = add nuw nsw i32 %4366, 1, !spirv.Decorations !631		; visa id: 5655
  br label %._crit_edge293, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5656

4376:                                             ; preds = %._crit_edge293
; BB398 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5658
  %4377 = load i64, i64* %129, align 8		; visa id: 5658
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5659
  %4378 = bitcast i64 %4377 to <2 x i32>		; visa id: 5659
  %4379 = extractelement <2 x i32> %4378, i32 0		; visa id: 5661
  %4380 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4379, i32 1
  %4381 = bitcast <2 x i32> %4380 to i64		; visa id: 5661
  %4382 = ashr exact i64 %4381, 32		; visa id: 5662
  %4383 = bitcast i64 %4382 to <2 x i32>		; visa id: 5663
  %4384 = extractelement <2 x i32> %4383, i32 0		; visa id: 5667
  %4385 = extractelement <2 x i32> %4383, i32 1		; visa id: 5667
  %4386 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4384, i32 %4385, i32 %41, i32 %42)
  %4387 = extractvalue { i32, i32 } %4386, 0		; visa id: 5667
  %4388 = extractvalue { i32, i32 } %4386, 1		; visa id: 5667
  %4389 = insertelement <2 x i32> undef, i32 %4387, i32 0		; visa id: 5674
  %4390 = insertelement <2 x i32> %4389, i32 %4388, i32 1		; visa id: 5675
  %4391 = bitcast <2 x i32> %4390 to i64		; visa id: 5676
  %4392 = shl i64 %4391, 1		; visa id: 5680
  %4393 = add i64 %.in401, %4392		; visa id: 5681
  %4394 = ashr i64 %4377, 31		; visa id: 5682
  %4395 = bitcast i64 %4394 to <2 x i32>		; visa id: 5683
  %4396 = extractelement <2 x i32> %4395, i32 0		; visa id: 5687
  %4397 = extractelement <2 x i32> %4395, i32 1		; visa id: 5687
  %4398 = and i32 %4396, -2		; visa id: 5687
  %4399 = insertelement <2 x i32> undef, i32 %4398, i32 0		; visa id: 5688
  %4400 = insertelement <2 x i32> %4399, i32 %4397, i32 1		; visa id: 5689
  %4401 = bitcast <2 x i32> %4400 to i64		; visa id: 5690
  %4402 = add i64 %4393, %4401		; visa id: 5694
  %4403 = inttoptr i64 %4402 to i16 addrspace(4)*		; visa id: 5695
  %4404 = addrspacecast i16 addrspace(4)* %4403 to i16 addrspace(1)*		; visa id: 5695
  %4405 = load i16, i16 addrspace(1)* %4404, align 2		; visa id: 5696
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5698
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5698
  %4406 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5698
  %4407 = insertelement <2 x i32> %4406, i32 %4064, i64 1		; visa id: 5699
  %4408 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5700
  store <2 x i32> %4407, <2 x i32>* %4408, align 4, !noalias !635		; visa id: 5700
  br label %._crit_edge294, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5702

._crit_edge294:                                   ; preds = %._crit_edge294.._crit_edge294_crit_edge, %4376
; BB399 :
  %4409 = phi i32 [ 0, %4376 ], [ %4418, %._crit_edge294.._crit_edge294_crit_edge ]
  %4410 = zext i32 %4409 to i64		; visa id: 5703
  %4411 = shl nuw nsw i64 %4410, 2		; visa id: 5704
  %4412 = add i64 %124, %4411		; visa id: 5705
  %4413 = inttoptr i64 %4412 to i32*		; visa id: 5706
  %4414 = load i32, i32* %4413, align 4, !noalias !635		; visa id: 5706
  %4415 = add i64 %119, %4411		; visa id: 5707
  %4416 = inttoptr i64 %4415 to i32*		; visa id: 5708
  store i32 %4414, i32* %4416, align 4, !alias.scope !635		; visa id: 5708
  %4417 = icmp eq i32 %4409, 0		; visa id: 5709
  br i1 %4417, label %._crit_edge294.._crit_edge294_crit_edge, label %4419, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5710

._crit_edge294.._crit_edge294_crit_edge:          ; preds = %._crit_edge294
; BB400 :
  %4418 = add nuw nsw i32 %4409, 1, !spirv.Decorations !631		; visa id: 5712
  br label %._crit_edge294, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5713

4419:                                             ; preds = %._crit_edge294
; BB401 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5715
  %4420 = load i64, i64* %120, align 8		; visa id: 5715
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5716
  %4421 = bitcast i64 %4420 to <2 x i32>		; visa id: 5716
  %4422 = extractelement <2 x i32> %4421, i32 0		; visa id: 5718
  %4423 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4422, i32 1
  %4424 = bitcast <2 x i32> %4423 to i64		; visa id: 5718
  %4425 = ashr exact i64 %4424, 32		; visa id: 5719
  %4426 = bitcast i64 %4425 to <2 x i32>		; visa id: 5720
  %4427 = extractelement <2 x i32> %4426, i32 0		; visa id: 5724
  %4428 = extractelement <2 x i32> %4426, i32 1		; visa id: 5724
  %4429 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4427, i32 %4428, i32 %44, i32 %45)
  %4430 = extractvalue { i32, i32 } %4429, 0		; visa id: 5724
  %4431 = extractvalue { i32, i32 } %4429, 1		; visa id: 5724
  %4432 = insertelement <2 x i32> undef, i32 %4430, i32 0		; visa id: 5731
  %4433 = insertelement <2 x i32> %4432, i32 %4431, i32 1		; visa id: 5732
  %4434 = bitcast <2 x i32> %4433 to i64		; visa id: 5733
  %4435 = shl i64 %4434, 1		; visa id: 5737
  %4436 = add i64 %.in400, %4435		; visa id: 5738
  %4437 = ashr i64 %4420, 31		; visa id: 5739
  %4438 = bitcast i64 %4437 to <2 x i32>		; visa id: 5740
  %4439 = extractelement <2 x i32> %4438, i32 0		; visa id: 5744
  %4440 = extractelement <2 x i32> %4438, i32 1		; visa id: 5744
  %4441 = and i32 %4439, -2		; visa id: 5744
  %4442 = insertelement <2 x i32> undef, i32 %4441, i32 0		; visa id: 5745
  %4443 = insertelement <2 x i32> %4442, i32 %4440, i32 1		; visa id: 5746
  %4444 = bitcast <2 x i32> %4443 to i64		; visa id: 5747
  %4445 = add i64 %4436, %4444		; visa id: 5751
  %4446 = inttoptr i64 %4445 to i16 addrspace(4)*		; visa id: 5752
  %4447 = addrspacecast i16 addrspace(4)* %4446 to i16 addrspace(1)*		; visa id: 5752
  %4448 = load i16, i16 addrspace(1)* %4447, align 2		; visa id: 5753
  %4449 = zext i16 %4405 to i32		; visa id: 5755
  %4450 = shl nuw i32 %4449, 16, !spirv.Decorations !639		; visa id: 5756
  %4451 = bitcast i32 %4450 to float
  %4452 = zext i16 %4448 to i32		; visa id: 5757
  %4453 = shl nuw i32 %4452, 16, !spirv.Decorations !639		; visa id: 5758
  %4454 = bitcast i32 %4453 to float
  %4455 = fmul reassoc nsz arcp contract float %4451, %4454, !spirv.Decorations !618
  %4456 = fadd reassoc nsz arcp contract float %4455, %.sroa.234.1, !spirv.Decorations !618		; visa id: 5759
  br label %.preheader.10, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5760

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %4419
; BB402 :
  %.sroa.234.2 = phi float [ %4456, %4419 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %4457 = add i32 %69, 11		; visa id: 5761
  %4458 = icmp slt i32 %4457, %const_reg_dword1		; visa id: 5762
  %4459 = icmp slt i32 %65, %const_reg_dword
  %4460 = and i1 %4459, %4458		; visa id: 5763
  br i1 %4460, label %4461, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5765

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
; BB:
  br label %._crit_edge.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4461:                                             ; preds = %.preheader.10
; BB404 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5767
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5767
  %4462 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 5767
  %4463 = insertelement <2 x i32> %4462, i32 %113, i64 1		; visa id: 5768
  %4464 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5769
  store <2 x i32> %4463, <2 x i32>* %4464, align 4, !noalias !625		; visa id: 5769
  br label %._crit_edge295, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5771

._crit_edge295:                                   ; preds = %._crit_edge295.._crit_edge295_crit_edge, %4461
; BB405 :
  %4465 = phi i32 [ 0, %4461 ], [ %4474, %._crit_edge295.._crit_edge295_crit_edge ]
  %4466 = zext i32 %4465 to i64		; visa id: 5772
  %4467 = shl nuw nsw i64 %4466, 2		; visa id: 5773
  %4468 = add i64 %133, %4467		; visa id: 5774
  %4469 = inttoptr i64 %4468 to i32*		; visa id: 5775
  %4470 = load i32, i32* %4469, align 4, !noalias !625		; visa id: 5775
  %4471 = add i64 %128, %4467		; visa id: 5776
  %4472 = inttoptr i64 %4471 to i32*		; visa id: 5777
  store i32 %4470, i32* %4472, align 4, !alias.scope !625		; visa id: 5777
  %4473 = icmp eq i32 %4465, 0		; visa id: 5778
  br i1 %4473, label %._crit_edge295.._crit_edge295_crit_edge, label %4475, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5779

._crit_edge295.._crit_edge295_crit_edge:          ; preds = %._crit_edge295
; BB406 :
  %4474 = add nuw nsw i32 %4465, 1, !spirv.Decorations !631		; visa id: 5781
  br label %._crit_edge295, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5782

4475:                                             ; preds = %._crit_edge295
; BB407 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5784
  %4476 = load i64, i64* %129, align 8		; visa id: 5784
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5785
  %4477 = bitcast i64 %4476 to <2 x i32>		; visa id: 5785
  %4478 = extractelement <2 x i32> %4477, i32 0		; visa id: 5787
  %4479 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4478, i32 1
  %4480 = bitcast <2 x i32> %4479 to i64		; visa id: 5787
  %4481 = ashr exact i64 %4480, 32		; visa id: 5788
  %4482 = bitcast i64 %4481 to <2 x i32>		; visa id: 5789
  %4483 = extractelement <2 x i32> %4482, i32 0		; visa id: 5793
  %4484 = extractelement <2 x i32> %4482, i32 1		; visa id: 5793
  %4485 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4483, i32 %4484, i32 %41, i32 %42)
  %4486 = extractvalue { i32, i32 } %4485, 0		; visa id: 5793
  %4487 = extractvalue { i32, i32 } %4485, 1		; visa id: 5793
  %4488 = insertelement <2 x i32> undef, i32 %4486, i32 0		; visa id: 5800
  %4489 = insertelement <2 x i32> %4488, i32 %4487, i32 1		; visa id: 5801
  %4490 = bitcast <2 x i32> %4489 to i64		; visa id: 5802
  %4491 = shl i64 %4490, 1		; visa id: 5806
  %4492 = add i64 %.in401, %4491		; visa id: 5807
  %4493 = ashr i64 %4476, 31		; visa id: 5808
  %4494 = bitcast i64 %4493 to <2 x i32>		; visa id: 5809
  %4495 = extractelement <2 x i32> %4494, i32 0		; visa id: 5813
  %4496 = extractelement <2 x i32> %4494, i32 1		; visa id: 5813
  %4497 = and i32 %4495, -2		; visa id: 5813
  %4498 = insertelement <2 x i32> undef, i32 %4497, i32 0		; visa id: 5814
  %4499 = insertelement <2 x i32> %4498, i32 %4496, i32 1		; visa id: 5815
  %4500 = bitcast <2 x i32> %4499 to i64		; visa id: 5816
  %4501 = add i64 %4492, %4500		; visa id: 5820
  %4502 = inttoptr i64 %4501 to i16 addrspace(4)*		; visa id: 5821
  %4503 = addrspacecast i16 addrspace(4)* %4502 to i16 addrspace(1)*		; visa id: 5821
  %4504 = load i16, i16 addrspace(1)* %4503, align 2		; visa id: 5822
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5824
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5824
  %4505 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5824
  %4506 = insertelement <2 x i32> %4505, i32 %4457, i64 1		; visa id: 5825
  %4507 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5826
  store <2 x i32> %4506, <2 x i32>* %4507, align 4, !noalias !635		; visa id: 5826
  br label %._crit_edge296, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5828

._crit_edge296:                                   ; preds = %._crit_edge296.._crit_edge296_crit_edge, %4475
; BB408 :
  %4508 = phi i32 [ 0, %4475 ], [ %4517, %._crit_edge296.._crit_edge296_crit_edge ]
  %4509 = zext i32 %4508 to i64		; visa id: 5829
  %4510 = shl nuw nsw i64 %4509, 2		; visa id: 5830
  %4511 = add i64 %124, %4510		; visa id: 5831
  %4512 = inttoptr i64 %4511 to i32*		; visa id: 5832
  %4513 = load i32, i32* %4512, align 4, !noalias !635		; visa id: 5832
  %4514 = add i64 %119, %4510		; visa id: 5833
  %4515 = inttoptr i64 %4514 to i32*		; visa id: 5834
  store i32 %4513, i32* %4515, align 4, !alias.scope !635		; visa id: 5834
  %4516 = icmp eq i32 %4508, 0		; visa id: 5835
  br i1 %4516, label %._crit_edge296.._crit_edge296_crit_edge, label %4518, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5836

._crit_edge296.._crit_edge296_crit_edge:          ; preds = %._crit_edge296
; BB409 :
  %4517 = add nuw nsw i32 %4508, 1, !spirv.Decorations !631		; visa id: 5838
  br label %._crit_edge296, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5839

4518:                                             ; preds = %._crit_edge296
; BB410 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5841
  %4519 = load i64, i64* %120, align 8		; visa id: 5841
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5842
  %4520 = bitcast i64 %4519 to <2 x i32>		; visa id: 5842
  %4521 = extractelement <2 x i32> %4520, i32 0		; visa id: 5844
  %4522 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4521, i32 1
  %4523 = bitcast <2 x i32> %4522 to i64		; visa id: 5844
  %4524 = ashr exact i64 %4523, 32		; visa id: 5845
  %4525 = bitcast i64 %4524 to <2 x i32>		; visa id: 5846
  %4526 = extractelement <2 x i32> %4525, i32 0		; visa id: 5850
  %4527 = extractelement <2 x i32> %4525, i32 1		; visa id: 5850
  %4528 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4526, i32 %4527, i32 %44, i32 %45)
  %4529 = extractvalue { i32, i32 } %4528, 0		; visa id: 5850
  %4530 = extractvalue { i32, i32 } %4528, 1		; visa id: 5850
  %4531 = insertelement <2 x i32> undef, i32 %4529, i32 0		; visa id: 5857
  %4532 = insertelement <2 x i32> %4531, i32 %4530, i32 1		; visa id: 5858
  %4533 = bitcast <2 x i32> %4532 to i64		; visa id: 5859
  %4534 = shl i64 %4533, 1		; visa id: 5863
  %4535 = add i64 %.in400, %4534		; visa id: 5864
  %4536 = ashr i64 %4519, 31		; visa id: 5865
  %4537 = bitcast i64 %4536 to <2 x i32>		; visa id: 5866
  %4538 = extractelement <2 x i32> %4537, i32 0		; visa id: 5870
  %4539 = extractelement <2 x i32> %4537, i32 1		; visa id: 5870
  %4540 = and i32 %4538, -2		; visa id: 5870
  %4541 = insertelement <2 x i32> undef, i32 %4540, i32 0		; visa id: 5871
  %4542 = insertelement <2 x i32> %4541, i32 %4539, i32 1		; visa id: 5872
  %4543 = bitcast <2 x i32> %4542 to i64		; visa id: 5873
  %4544 = add i64 %4535, %4543		; visa id: 5877
  %4545 = inttoptr i64 %4544 to i16 addrspace(4)*		; visa id: 5878
  %4546 = addrspacecast i16 addrspace(4)* %4545 to i16 addrspace(1)*		; visa id: 5878
  %4547 = load i16, i16 addrspace(1)* %4546, align 2		; visa id: 5879
  %4548 = zext i16 %4504 to i32		; visa id: 5881
  %4549 = shl nuw i32 %4548, 16, !spirv.Decorations !639		; visa id: 5882
  %4550 = bitcast i32 %4549 to float
  %4551 = zext i16 %4547 to i32		; visa id: 5883
  %4552 = shl nuw i32 %4551, 16, !spirv.Decorations !639		; visa id: 5884
  %4553 = bitcast i32 %4552 to float
  %4554 = fmul reassoc nsz arcp contract float %4550, %4553, !spirv.Decorations !618
  %4555 = fadd reassoc nsz arcp contract float %4554, %.sroa.46.1, !spirv.Decorations !618		; visa id: 5885
  br label %._crit_edge.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5886

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %4518
; BB411 :
  %.sroa.46.2 = phi float [ %4555, %4518 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  %4556 = icmp slt i32 %230, %const_reg_dword
  %4557 = icmp slt i32 %4457, %const_reg_dword1		; visa id: 5887
  %4558 = and i1 %4556, %4557		; visa id: 5888
  br i1 %4558, label %4559, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 5890

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
; BB:
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4559:                                             ; preds = %._crit_edge.11
; BB413 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 5892
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 5892
  %4560 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 5892
  %4561 = insertelement <2 x i32> %4560, i32 %113, i64 1		; visa id: 5893
  %4562 = inttoptr i64 %133 to <2 x i32>*		; visa id: 5894
  store <2 x i32> %4561, <2 x i32>* %4562, align 4, !noalias !625		; visa id: 5894
  br label %._crit_edge297, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5896

._crit_edge297:                                   ; preds = %._crit_edge297.._crit_edge297_crit_edge, %4559
; BB414 :
  %4563 = phi i32 [ 0, %4559 ], [ %4572, %._crit_edge297.._crit_edge297_crit_edge ]
  %4564 = zext i32 %4563 to i64		; visa id: 5897
  %4565 = shl nuw nsw i64 %4564, 2		; visa id: 5898
  %4566 = add i64 %133, %4565		; visa id: 5899
  %4567 = inttoptr i64 %4566 to i32*		; visa id: 5900
  %4568 = load i32, i32* %4567, align 4, !noalias !625		; visa id: 5900
  %4569 = add i64 %128, %4565		; visa id: 5901
  %4570 = inttoptr i64 %4569 to i32*		; visa id: 5902
  store i32 %4568, i32* %4570, align 4, !alias.scope !625		; visa id: 5902
  %4571 = icmp eq i32 %4563, 0		; visa id: 5903
  br i1 %4571, label %._crit_edge297.._crit_edge297_crit_edge, label %4573, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5904

._crit_edge297.._crit_edge297_crit_edge:          ; preds = %._crit_edge297
; BB415 :
  %4572 = add nuw nsw i32 %4563, 1, !spirv.Decorations !631		; visa id: 5906
  br label %._crit_edge297, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5907

4573:                                             ; preds = %._crit_edge297
; BB416 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 5909
  %4574 = load i64, i64* %129, align 8		; visa id: 5909
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 5910
  %4575 = bitcast i64 %4574 to <2 x i32>		; visa id: 5910
  %4576 = extractelement <2 x i32> %4575, i32 0		; visa id: 5912
  %4577 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4576, i32 1
  %4578 = bitcast <2 x i32> %4577 to i64		; visa id: 5912
  %4579 = ashr exact i64 %4578, 32		; visa id: 5913
  %4580 = bitcast i64 %4579 to <2 x i32>		; visa id: 5914
  %4581 = extractelement <2 x i32> %4580, i32 0		; visa id: 5918
  %4582 = extractelement <2 x i32> %4580, i32 1		; visa id: 5918
  %4583 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4581, i32 %4582, i32 %41, i32 %42)
  %4584 = extractvalue { i32, i32 } %4583, 0		; visa id: 5918
  %4585 = extractvalue { i32, i32 } %4583, 1		; visa id: 5918
  %4586 = insertelement <2 x i32> undef, i32 %4584, i32 0		; visa id: 5925
  %4587 = insertelement <2 x i32> %4586, i32 %4585, i32 1		; visa id: 5926
  %4588 = bitcast <2 x i32> %4587 to i64		; visa id: 5927
  %4589 = shl i64 %4588, 1		; visa id: 5931
  %4590 = add i64 %.in401, %4589		; visa id: 5932
  %4591 = ashr i64 %4574, 31		; visa id: 5933
  %4592 = bitcast i64 %4591 to <2 x i32>		; visa id: 5934
  %4593 = extractelement <2 x i32> %4592, i32 0		; visa id: 5938
  %4594 = extractelement <2 x i32> %4592, i32 1		; visa id: 5938
  %4595 = and i32 %4593, -2		; visa id: 5938
  %4596 = insertelement <2 x i32> undef, i32 %4595, i32 0		; visa id: 5939
  %4597 = insertelement <2 x i32> %4596, i32 %4594, i32 1		; visa id: 5940
  %4598 = bitcast <2 x i32> %4597 to i64		; visa id: 5941
  %4599 = add i64 %4590, %4598		; visa id: 5945
  %4600 = inttoptr i64 %4599 to i16 addrspace(4)*		; visa id: 5946
  %4601 = addrspacecast i16 addrspace(4)* %4600 to i16 addrspace(1)*		; visa id: 5946
  %4602 = load i16, i16 addrspace(1)* %4601, align 2		; visa id: 5947
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 5949
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 5949
  %4603 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 5949
  %4604 = insertelement <2 x i32> %4603, i32 %4457, i64 1		; visa id: 5950
  %4605 = inttoptr i64 %124 to <2 x i32>*		; visa id: 5951
  store <2 x i32> %4604, <2 x i32>* %4605, align 4, !noalias !635		; visa id: 5951
  br label %._crit_edge298, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 5953

._crit_edge298:                                   ; preds = %._crit_edge298.._crit_edge298_crit_edge, %4573
; BB417 :
  %4606 = phi i32 [ 0, %4573 ], [ %4615, %._crit_edge298.._crit_edge298_crit_edge ]
  %4607 = zext i32 %4606 to i64		; visa id: 5954
  %4608 = shl nuw nsw i64 %4607, 2		; visa id: 5955
  %4609 = add i64 %124, %4608		; visa id: 5956
  %4610 = inttoptr i64 %4609 to i32*		; visa id: 5957
  %4611 = load i32, i32* %4610, align 4, !noalias !635		; visa id: 5957
  %4612 = add i64 %119, %4608		; visa id: 5958
  %4613 = inttoptr i64 %4612 to i32*		; visa id: 5959
  store i32 %4611, i32* %4613, align 4, !alias.scope !635		; visa id: 5959
  %4614 = icmp eq i32 %4606, 0		; visa id: 5960
  br i1 %4614, label %._crit_edge298.._crit_edge298_crit_edge, label %4616, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 5961

._crit_edge298.._crit_edge298_crit_edge:          ; preds = %._crit_edge298
; BB418 :
  %4615 = add nuw nsw i32 %4606, 1, !spirv.Decorations !631		; visa id: 5963
  br label %._crit_edge298, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 5964

4616:                                             ; preds = %._crit_edge298
; BB419 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 5966
  %4617 = load i64, i64* %120, align 8		; visa id: 5966
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 5967
  %4618 = bitcast i64 %4617 to <2 x i32>		; visa id: 5967
  %4619 = extractelement <2 x i32> %4618, i32 0		; visa id: 5969
  %4620 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4619, i32 1
  %4621 = bitcast <2 x i32> %4620 to i64		; visa id: 5969
  %4622 = ashr exact i64 %4621, 32		; visa id: 5970
  %4623 = bitcast i64 %4622 to <2 x i32>		; visa id: 5971
  %4624 = extractelement <2 x i32> %4623, i32 0		; visa id: 5975
  %4625 = extractelement <2 x i32> %4623, i32 1		; visa id: 5975
  %4626 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4624, i32 %4625, i32 %44, i32 %45)
  %4627 = extractvalue { i32, i32 } %4626, 0		; visa id: 5975
  %4628 = extractvalue { i32, i32 } %4626, 1		; visa id: 5975
  %4629 = insertelement <2 x i32> undef, i32 %4627, i32 0		; visa id: 5982
  %4630 = insertelement <2 x i32> %4629, i32 %4628, i32 1		; visa id: 5983
  %4631 = bitcast <2 x i32> %4630 to i64		; visa id: 5984
  %4632 = shl i64 %4631, 1		; visa id: 5988
  %4633 = add i64 %.in400, %4632		; visa id: 5989
  %4634 = ashr i64 %4617, 31		; visa id: 5990
  %4635 = bitcast i64 %4634 to <2 x i32>		; visa id: 5991
  %4636 = extractelement <2 x i32> %4635, i32 0		; visa id: 5995
  %4637 = extractelement <2 x i32> %4635, i32 1		; visa id: 5995
  %4638 = and i32 %4636, -2		; visa id: 5995
  %4639 = insertelement <2 x i32> undef, i32 %4638, i32 0		; visa id: 5996
  %4640 = insertelement <2 x i32> %4639, i32 %4637, i32 1		; visa id: 5997
  %4641 = bitcast <2 x i32> %4640 to i64		; visa id: 5998
  %4642 = add i64 %4633, %4641		; visa id: 6002
  %4643 = inttoptr i64 %4642 to i16 addrspace(4)*		; visa id: 6003
  %4644 = addrspacecast i16 addrspace(4)* %4643 to i16 addrspace(1)*		; visa id: 6003
  %4645 = load i16, i16 addrspace(1)* %4644, align 2		; visa id: 6004
  %4646 = zext i16 %4602 to i32		; visa id: 6006
  %4647 = shl nuw i32 %4646, 16, !spirv.Decorations !639		; visa id: 6007
  %4648 = bitcast i32 %4647 to float
  %4649 = zext i16 %4645 to i32		; visa id: 6008
  %4650 = shl nuw i32 %4649, 16, !spirv.Decorations !639		; visa id: 6009
  %4651 = bitcast i32 %4650 to float
  %4652 = fmul reassoc nsz arcp contract float %4648, %4651, !spirv.Decorations !618
  %4653 = fadd reassoc nsz arcp contract float %4652, %.sroa.110.1, !spirv.Decorations !618		; visa id: 6010
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6011

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %4616
; BB420 :
  %.sroa.110.2 = phi float [ %4653, %4616 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  %4654 = icmp slt i32 %329, %const_reg_dword
  %4655 = icmp slt i32 %4457, %const_reg_dword1		; visa id: 6012
  %4656 = and i1 %4654, %4655		; visa id: 6013
  br i1 %4656, label %4657, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6015

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
; BB:
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4657:                                             ; preds = %._crit_edge.1.11
; BB422 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6017
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6017
  %4658 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 6017
  %4659 = insertelement <2 x i32> %4658, i32 %113, i64 1		; visa id: 6018
  %4660 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6019
  store <2 x i32> %4659, <2 x i32>* %4660, align 4, !noalias !625		; visa id: 6019
  br label %._crit_edge299, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6021

._crit_edge299:                                   ; preds = %._crit_edge299.._crit_edge299_crit_edge, %4657
; BB423 :
  %4661 = phi i32 [ 0, %4657 ], [ %4670, %._crit_edge299.._crit_edge299_crit_edge ]
  %4662 = zext i32 %4661 to i64		; visa id: 6022
  %4663 = shl nuw nsw i64 %4662, 2		; visa id: 6023
  %4664 = add i64 %133, %4663		; visa id: 6024
  %4665 = inttoptr i64 %4664 to i32*		; visa id: 6025
  %4666 = load i32, i32* %4665, align 4, !noalias !625		; visa id: 6025
  %4667 = add i64 %128, %4663		; visa id: 6026
  %4668 = inttoptr i64 %4667 to i32*		; visa id: 6027
  store i32 %4666, i32* %4668, align 4, !alias.scope !625		; visa id: 6027
  %4669 = icmp eq i32 %4661, 0		; visa id: 6028
  br i1 %4669, label %._crit_edge299.._crit_edge299_crit_edge, label %4671, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6029

._crit_edge299.._crit_edge299_crit_edge:          ; preds = %._crit_edge299
; BB424 :
  %4670 = add nuw nsw i32 %4661, 1, !spirv.Decorations !631		; visa id: 6031
  br label %._crit_edge299, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6032

4671:                                             ; preds = %._crit_edge299
; BB425 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6034
  %4672 = load i64, i64* %129, align 8		; visa id: 6034
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6035
  %4673 = bitcast i64 %4672 to <2 x i32>		; visa id: 6035
  %4674 = extractelement <2 x i32> %4673, i32 0		; visa id: 6037
  %4675 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4674, i32 1
  %4676 = bitcast <2 x i32> %4675 to i64		; visa id: 6037
  %4677 = ashr exact i64 %4676, 32		; visa id: 6038
  %4678 = bitcast i64 %4677 to <2 x i32>		; visa id: 6039
  %4679 = extractelement <2 x i32> %4678, i32 0		; visa id: 6043
  %4680 = extractelement <2 x i32> %4678, i32 1		; visa id: 6043
  %4681 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4679, i32 %4680, i32 %41, i32 %42)
  %4682 = extractvalue { i32, i32 } %4681, 0		; visa id: 6043
  %4683 = extractvalue { i32, i32 } %4681, 1		; visa id: 6043
  %4684 = insertelement <2 x i32> undef, i32 %4682, i32 0		; visa id: 6050
  %4685 = insertelement <2 x i32> %4684, i32 %4683, i32 1		; visa id: 6051
  %4686 = bitcast <2 x i32> %4685 to i64		; visa id: 6052
  %4687 = shl i64 %4686, 1		; visa id: 6056
  %4688 = add i64 %.in401, %4687		; visa id: 6057
  %4689 = ashr i64 %4672, 31		; visa id: 6058
  %4690 = bitcast i64 %4689 to <2 x i32>		; visa id: 6059
  %4691 = extractelement <2 x i32> %4690, i32 0		; visa id: 6063
  %4692 = extractelement <2 x i32> %4690, i32 1		; visa id: 6063
  %4693 = and i32 %4691, -2		; visa id: 6063
  %4694 = insertelement <2 x i32> undef, i32 %4693, i32 0		; visa id: 6064
  %4695 = insertelement <2 x i32> %4694, i32 %4692, i32 1		; visa id: 6065
  %4696 = bitcast <2 x i32> %4695 to i64		; visa id: 6066
  %4697 = add i64 %4688, %4696		; visa id: 6070
  %4698 = inttoptr i64 %4697 to i16 addrspace(4)*		; visa id: 6071
  %4699 = addrspacecast i16 addrspace(4)* %4698 to i16 addrspace(1)*		; visa id: 6071
  %4700 = load i16, i16 addrspace(1)* %4699, align 2		; visa id: 6072
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6074
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6074
  %4701 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6074
  %4702 = insertelement <2 x i32> %4701, i32 %4457, i64 1		; visa id: 6075
  %4703 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6076
  store <2 x i32> %4702, <2 x i32>* %4703, align 4, !noalias !635		; visa id: 6076
  br label %._crit_edge300, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6078

._crit_edge300:                                   ; preds = %._crit_edge300.._crit_edge300_crit_edge, %4671
; BB426 :
  %4704 = phi i32 [ 0, %4671 ], [ %4713, %._crit_edge300.._crit_edge300_crit_edge ]
  %4705 = zext i32 %4704 to i64		; visa id: 6079
  %4706 = shl nuw nsw i64 %4705, 2		; visa id: 6080
  %4707 = add i64 %124, %4706		; visa id: 6081
  %4708 = inttoptr i64 %4707 to i32*		; visa id: 6082
  %4709 = load i32, i32* %4708, align 4, !noalias !635		; visa id: 6082
  %4710 = add i64 %119, %4706		; visa id: 6083
  %4711 = inttoptr i64 %4710 to i32*		; visa id: 6084
  store i32 %4709, i32* %4711, align 4, !alias.scope !635		; visa id: 6084
  %4712 = icmp eq i32 %4704, 0		; visa id: 6085
  br i1 %4712, label %._crit_edge300.._crit_edge300_crit_edge, label %4714, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6086

._crit_edge300.._crit_edge300_crit_edge:          ; preds = %._crit_edge300
; BB427 :
  %4713 = add nuw nsw i32 %4704, 1, !spirv.Decorations !631		; visa id: 6088
  br label %._crit_edge300, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6089

4714:                                             ; preds = %._crit_edge300
; BB428 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6091
  %4715 = load i64, i64* %120, align 8		; visa id: 6091
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6092
  %4716 = bitcast i64 %4715 to <2 x i32>		; visa id: 6092
  %4717 = extractelement <2 x i32> %4716, i32 0		; visa id: 6094
  %4718 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4717, i32 1
  %4719 = bitcast <2 x i32> %4718 to i64		; visa id: 6094
  %4720 = ashr exact i64 %4719, 32		; visa id: 6095
  %4721 = bitcast i64 %4720 to <2 x i32>		; visa id: 6096
  %4722 = extractelement <2 x i32> %4721, i32 0		; visa id: 6100
  %4723 = extractelement <2 x i32> %4721, i32 1		; visa id: 6100
  %4724 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4722, i32 %4723, i32 %44, i32 %45)
  %4725 = extractvalue { i32, i32 } %4724, 0		; visa id: 6100
  %4726 = extractvalue { i32, i32 } %4724, 1		; visa id: 6100
  %4727 = insertelement <2 x i32> undef, i32 %4725, i32 0		; visa id: 6107
  %4728 = insertelement <2 x i32> %4727, i32 %4726, i32 1		; visa id: 6108
  %4729 = bitcast <2 x i32> %4728 to i64		; visa id: 6109
  %4730 = shl i64 %4729, 1		; visa id: 6113
  %4731 = add i64 %.in400, %4730		; visa id: 6114
  %4732 = ashr i64 %4715, 31		; visa id: 6115
  %4733 = bitcast i64 %4732 to <2 x i32>		; visa id: 6116
  %4734 = extractelement <2 x i32> %4733, i32 0		; visa id: 6120
  %4735 = extractelement <2 x i32> %4733, i32 1		; visa id: 6120
  %4736 = and i32 %4734, -2		; visa id: 6120
  %4737 = insertelement <2 x i32> undef, i32 %4736, i32 0		; visa id: 6121
  %4738 = insertelement <2 x i32> %4737, i32 %4735, i32 1		; visa id: 6122
  %4739 = bitcast <2 x i32> %4738 to i64		; visa id: 6123
  %4740 = add i64 %4731, %4739		; visa id: 6127
  %4741 = inttoptr i64 %4740 to i16 addrspace(4)*		; visa id: 6128
  %4742 = addrspacecast i16 addrspace(4)* %4741 to i16 addrspace(1)*		; visa id: 6128
  %4743 = load i16, i16 addrspace(1)* %4742, align 2		; visa id: 6129
  %4744 = zext i16 %4700 to i32		; visa id: 6131
  %4745 = shl nuw i32 %4744, 16, !spirv.Decorations !639		; visa id: 6132
  %4746 = bitcast i32 %4745 to float
  %4747 = zext i16 %4743 to i32		; visa id: 6133
  %4748 = shl nuw i32 %4747, 16, !spirv.Decorations !639		; visa id: 6134
  %4749 = bitcast i32 %4748 to float
  %4750 = fmul reassoc nsz arcp contract float %4746, %4749, !spirv.Decorations !618
  %4751 = fadd reassoc nsz arcp contract float %4750, %.sroa.174.1, !spirv.Decorations !618		; visa id: 6135
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6136

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %4714
; BB429 :
  %.sroa.174.2 = phi float [ %4751, %4714 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  %4752 = icmp slt i32 %428, %const_reg_dword
  %4753 = icmp slt i32 %4457, %const_reg_dword1		; visa id: 6137
  %4754 = and i1 %4752, %4753		; visa id: 6138
  br i1 %4754, label %4755, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6140

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
; BB:
  br label %.preheader.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4755:                                             ; preds = %._crit_edge.2.11
; BB431 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6142
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6142
  %4756 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 6142
  %4757 = insertelement <2 x i32> %4756, i32 %113, i64 1		; visa id: 6143
  %4758 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6144
  store <2 x i32> %4757, <2 x i32>* %4758, align 4, !noalias !625		; visa id: 6144
  br label %._crit_edge301, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6146

._crit_edge301:                                   ; preds = %._crit_edge301.._crit_edge301_crit_edge, %4755
; BB432 :
  %4759 = phi i32 [ 0, %4755 ], [ %4768, %._crit_edge301.._crit_edge301_crit_edge ]
  %4760 = zext i32 %4759 to i64		; visa id: 6147
  %4761 = shl nuw nsw i64 %4760, 2		; visa id: 6148
  %4762 = add i64 %133, %4761		; visa id: 6149
  %4763 = inttoptr i64 %4762 to i32*		; visa id: 6150
  %4764 = load i32, i32* %4763, align 4, !noalias !625		; visa id: 6150
  %4765 = add i64 %128, %4761		; visa id: 6151
  %4766 = inttoptr i64 %4765 to i32*		; visa id: 6152
  store i32 %4764, i32* %4766, align 4, !alias.scope !625		; visa id: 6152
  %4767 = icmp eq i32 %4759, 0		; visa id: 6153
  br i1 %4767, label %._crit_edge301.._crit_edge301_crit_edge, label %4769, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6154

._crit_edge301.._crit_edge301_crit_edge:          ; preds = %._crit_edge301
; BB433 :
  %4768 = add nuw nsw i32 %4759, 1, !spirv.Decorations !631		; visa id: 6156
  br label %._crit_edge301, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6157

4769:                                             ; preds = %._crit_edge301
; BB434 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6159
  %4770 = load i64, i64* %129, align 8		; visa id: 6159
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6160
  %4771 = bitcast i64 %4770 to <2 x i32>		; visa id: 6160
  %4772 = extractelement <2 x i32> %4771, i32 0		; visa id: 6162
  %4773 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4772, i32 1
  %4774 = bitcast <2 x i32> %4773 to i64		; visa id: 6162
  %4775 = ashr exact i64 %4774, 32		; visa id: 6163
  %4776 = bitcast i64 %4775 to <2 x i32>		; visa id: 6164
  %4777 = extractelement <2 x i32> %4776, i32 0		; visa id: 6168
  %4778 = extractelement <2 x i32> %4776, i32 1		; visa id: 6168
  %4779 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4777, i32 %4778, i32 %41, i32 %42)
  %4780 = extractvalue { i32, i32 } %4779, 0		; visa id: 6168
  %4781 = extractvalue { i32, i32 } %4779, 1		; visa id: 6168
  %4782 = insertelement <2 x i32> undef, i32 %4780, i32 0		; visa id: 6175
  %4783 = insertelement <2 x i32> %4782, i32 %4781, i32 1		; visa id: 6176
  %4784 = bitcast <2 x i32> %4783 to i64		; visa id: 6177
  %4785 = shl i64 %4784, 1		; visa id: 6181
  %4786 = add i64 %.in401, %4785		; visa id: 6182
  %4787 = ashr i64 %4770, 31		; visa id: 6183
  %4788 = bitcast i64 %4787 to <2 x i32>		; visa id: 6184
  %4789 = extractelement <2 x i32> %4788, i32 0		; visa id: 6188
  %4790 = extractelement <2 x i32> %4788, i32 1		; visa id: 6188
  %4791 = and i32 %4789, -2		; visa id: 6188
  %4792 = insertelement <2 x i32> undef, i32 %4791, i32 0		; visa id: 6189
  %4793 = insertelement <2 x i32> %4792, i32 %4790, i32 1		; visa id: 6190
  %4794 = bitcast <2 x i32> %4793 to i64		; visa id: 6191
  %4795 = add i64 %4786, %4794		; visa id: 6195
  %4796 = inttoptr i64 %4795 to i16 addrspace(4)*		; visa id: 6196
  %4797 = addrspacecast i16 addrspace(4)* %4796 to i16 addrspace(1)*		; visa id: 6196
  %4798 = load i16, i16 addrspace(1)* %4797, align 2		; visa id: 6197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6199
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6199
  %4799 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6199
  %4800 = insertelement <2 x i32> %4799, i32 %4457, i64 1		; visa id: 6200
  %4801 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6201
  store <2 x i32> %4800, <2 x i32>* %4801, align 4, !noalias !635		; visa id: 6201
  br label %._crit_edge302, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6203

._crit_edge302:                                   ; preds = %._crit_edge302.._crit_edge302_crit_edge, %4769
; BB435 :
  %4802 = phi i32 [ 0, %4769 ], [ %4811, %._crit_edge302.._crit_edge302_crit_edge ]
  %4803 = zext i32 %4802 to i64		; visa id: 6204
  %4804 = shl nuw nsw i64 %4803, 2		; visa id: 6205
  %4805 = add i64 %124, %4804		; visa id: 6206
  %4806 = inttoptr i64 %4805 to i32*		; visa id: 6207
  %4807 = load i32, i32* %4806, align 4, !noalias !635		; visa id: 6207
  %4808 = add i64 %119, %4804		; visa id: 6208
  %4809 = inttoptr i64 %4808 to i32*		; visa id: 6209
  store i32 %4807, i32* %4809, align 4, !alias.scope !635		; visa id: 6209
  %4810 = icmp eq i32 %4802, 0		; visa id: 6210
  br i1 %4810, label %._crit_edge302.._crit_edge302_crit_edge, label %4812, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6211

._crit_edge302.._crit_edge302_crit_edge:          ; preds = %._crit_edge302
; BB436 :
  %4811 = add nuw nsw i32 %4802, 1, !spirv.Decorations !631		; visa id: 6213
  br label %._crit_edge302, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6214

4812:                                             ; preds = %._crit_edge302
; BB437 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6216
  %4813 = load i64, i64* %120, align 8		; visa id: 6216
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6217
  %4814 = bitcast i64 %4813 to <2 x i32>		; visa id: 6217
  %4815 = extractelement <2 x i32> %4814, i32 0		; visa id: 6219
  %4816 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4815, i32 1
  %4817 = bitcast <2 x i32> %4816 to i64		; visa id: 6219
  %4818 = ashr exact i64 %4817, 32		; visa id: 6220
  %4819 = bitcast i64 %4818 to <2 x i32>		; visa id: 6221
  %4820 = extractelement <2 x i32> %4819, i32 0		; visa id: 6225
  %4821 = extractelement <2 x i32> %4819, i32 1		; visa id: 6225
  %4822 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4820, i32 %4821, i32 %44, i32 %45)
  %4823 = extractvalue { i32, i32 } %4822, 0		; visa id: 6225
  %4824 = extractvalue { i32, i32 } %4822, 1		; visa id: 6225
  %4825 = insertelement <2 x i32> undef, i32 %4823, i32 0		; visa id: 6232
  %4826 = insertelement <2 x i32> %4825, i32 %4824, i32 1		; visa id: 6233
  %4827 = bitcast <2 x i32> %4826 to i64		; visa id: 6234
  %4828 = shl i64 %4827, 1		; visa id: 6238
  %4829 = add i64 %.in400, %4828		; visa id: 6239
  %4830 = ashr i64 %4813, 31		; visa id: 6240
  %4831 = bitcast i64 %4830 to <2 x i32>		; visa id: 6241
  %4832 = extractelement <2 x i32> %4831, i32 0		; visa id: 6245
  %4833 = extractelement <2 x i32> %4831, i32 1		; visa id: 6245
  %4834 = and i32 %4832, -2		; visa id: 6245
  %4835 = insertelement <2 x i32> undef, i32 %4834, i32 0		; visa id: 6246
  %4836 = insertelement <2 x i32> %4835, i32 %4833, i32 1		; visa id: 6247
  %4837 = bitcast <2 x i32> %4836 to i64		; visa id: 6248
  %4838 = add i64 %4829, %4837		; visa id: 6252
  %4839 = inttoptr i64 %4838 to i16 addrspace(4)*		; visa id: 6253
  %4840 = addrspacecast i16 addrspace(4)* %4839 to i16 addrspace(1)*		; visa id: 6253
  %4841 = load i16, i16 addrspace(1)* %4840, align 2		; visa id: 6254
  %4842 = zext i16 %4798 to i32		; visa id: 6256
  %4843 = shl nuw i32 %4842, 16, !spirv.Decorations !639		; visa id: 6257
  %4844 = bitcast i32 %4843 to float
  %4845 = zext i16 %4841 to i32		; visa id: 6258
  %4846 = shl nuw i32 %4845, 16, !spirv.Decorations !639		; visa id: 6259
  %4847 = bitcast i32 %4846 to float
  %4848 = fmul reassoc nsz arcp contract float %4844, %4847, !spirv.Decorations !618
  %4849 = fadd reassoc nsz arcp contract float %4848, %.sroa.238.1, !spirv.Decorations !618		; visa id: 6260
  br label %.preheader.11, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6261

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %4812
; BB438 :
  %.sroa.238.2 = phi float [ %4849, %4812 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %4850 = add i32 %69, 12		; visa id: 6262
  %4851 = icmp slt i32 %4850, %const_reg_dword1		; visa id: 6263
  %4852 = icmp slt i32 %65, %const_reg_dword
  %4853 = and i1 %4852, %4851		; visa id: 6264
  br i1 %4853, label %4854, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6266

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
; BB:
  br label %._crit_edge.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4854:                                             ; preds = %.preheader.11
; BB440 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6268
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6268
  %4855 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 6268
  %4856 = insertelement <2 x i32> %4855, i32 %113, i64 1		; visa id: 6269
  %4857 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6270
  store <2 x i32> %4856, <2 x i32>* %4857, align 4, !noalias !625		; visa id: 6270
  br label %._crit_edge303, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6272

._crit_edge303:                                   ; preds = %._crit_edge303.._crit_edge303_crit_edge, %4854
; BB441 :
  %4858 = phi i32 [ 0, %4854 ], [ %4867, %._crit_edge303.._crit_edge303_crit_edge ]
  %4859 = zext i32 %4858 to i64		; visa id: 6273
  %4860 = shl nuw nsw i64 %4859, 2		; visa id: 6274
  %4861 = add i64 %133, %4860		; visa id: 6275
  %4862 = inttoptr i64 %4861 to i32*		; visa id: 6276
  %4863 = load i32, i32* %4862, align 4, !noalias !625		; visa id: 6276
  %4864 = add i64 %128, %4860		; visa id: 6277
  %4865 = inttoptr i64 %4864 to i32*		; visa id: 6278
  store i32 %4863, i32* %4865, align 4, !alias.scope !625		; visa id: 6278
  %4866 = icmp eq i32 %4858, 0		; visa id: 6279
  br i1 %4866, label %._crit_edge303.._crit_edge303_crit_edge, label %4868, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6280

._crit_edge303.._crit_edge303_crit_edge:          ; preds = %._crit_edge303
; BB442 :
  %4867 = add nuw nsw i32 %4858, 1, !spirv.Decorations !631		; visa id: 6282
  br label %._crit_edge303, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6283

4868:                                             ; preds = %._crit_edge303
; BB443 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6285
  %4869 = load i64, i64* %129, align 8		; visa id: 6285
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6286
  %4870 = bitcast i64 %4869 to <2 x i32>		; visa id: 6286
  %4871 = extractelement <2 x i32> %4870, i32 0		; visa id: 6288
  %4872 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4871, i32 1
  %4873 = bitcast <2 x i32> %4872 to i64		; visa id: 6288
  %4874 = ashr exact i64 %4873, 32		; visa id: 6289
  %4875 = bitcast i64 %4874 to <2 x i32>		; visa id: 6290
  %4876 = extractelement <2 x i32> %4875, i32 0		; visa id: 6294
  %4877 = extractelement <2 x i32> %4875, i32 1		; visa id: 6294
  %4878 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4876, i32 %4877, i32 %41, i32 %42)
  %4879 = extractvalue { i32, i32 } %4878, 0		; visa id: 6294
  %4880 = extractvalue { i32, i32 } %4878, 1		; visa id: 6294
  %4881 = insertelement <2 x i32> undef, i32 %4879, i32 0		; visa id: 6301
  %4882 = insertelement <2 x i32> %4881, i32 %4880, i32 1		; visa id: 6302
  %4883 = bitcast <2 x i32> %4882 to i64		; visa id: 6303
  %4884 = shl i64 %4883, 1		; visa id: 6307
  %4885 = add i64 %.in401, %4884		; visa id: 6308
  %4886 = ashr i64 %4869, 31		; visa id: 6309
  %4887 = bitcast i64 %4886 to <2 x i32>		; visa id: 6310
  %4888 = extractelement <2 x i32> %4887, i32 0		; visa id: 6314
  %4889 = extractelement <2 x i32> %4887, i32 1		; visa id: 6314
  %4890 = and i32 %4888, -2		; visa id: 6314
  %4891 = insertelement <2 x i32> undef, i32 %4890, i32 0		; visa id: 6315
  %4892 = insertelement <2 x i32> %4891, i32 %4889, i32 1		; visa id: 6316
  %4893 = bitcast <2 x i32> %4892 to i64		; visa id: 6317
  %4894 = add i64 %4885, %4893		; visa id: 6321
  %4895 = inttoptr i64 %4894 to i16 addrspace(4)*		; visa id: 6322
  %4896 = addrspacecast i16 addrspace(4)* %4895 to i16 addrspace(1)*		; visa id: 6322
  %4897 = load i16, i16 addrspace(1)* %4896, align 2		; visa id: 6323
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6325
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6325
  %4898 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6325
  %4899 = insertelement <2 x i32> %4898, i32 %4850, i64 1		; visa id: 6326
  %4900 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6327
  store <2 x i32> %4899, <2 x i32>* %4900, align 4, !noalias !635		; visa id: 6327
  br label %._crit_edge304, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6329

._crit_edge304:                                   ; preds = %._crit_edge304.._crit_edge304_crit_edge, %4868
; BB444 :
  %4901 = phi i32 [ 0, %4868 ], [ %4910, %._crit_edge304.._crit_edge304_crit_edge ]
  %4902 = zext i32 %4901 to i64		; visa id: 6330
  %4903 = shl nuw nsw i64 %4902, 2		; visa id: 6331
  %4904 = add i64 %124, %4903		; visa id: 6332
  %4905 = inttoptr i64 %4904 to i32*		; visa id: 6333
  %4906 = load i32, i32* %4905, align 4, !noalias !635		; visa id: 6333
  %4907 = add i64 %119, %4903		; visa id: 6334
  %4908 = inttoptr i64 %4907 to i32*		; visa id: 6335
  store i32 %4906, i32* %4908, align 4, !alias.scope !635		; visa id: 6335
  %4909 = icmp eq i32 %4901, 0		; visa id: 6336
  br i1 %4909, label %._crit_edge304.._crit_edge304_crit_edge, label %4911, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6337

._crit_edge304.._crit_edge304_crit_edge:          ; preds = %._crit_edge304
; BB445 :
  %4910 = add nuw nsw i32 %4901, 1, !spirv.Decorations !631		; visa id: 6339
  br label %._crit_edge304, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6340

4911:                                             ; preds = %._crit_edge304
; BB446 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6342
  %4912 = load i64, i64* %120, align 8		; visa id: 6342
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6343
  %4913 = bitcast i64 %4912 to <2 x i32>		; visa id: 6343
  %4914 = extractelement <2 x i32> %4913, i32 0		; visa id: 6345
  %4915 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4914, i32 1
  %4916 = bitcast <2 x i32> %4915 to i64		; visa id: 6345
  %4917 = ashr exact i64 %4916, 32		; visa id: 6346
  %4918 = bitcast i64 %4917 to <2 x i32>		; visa id: 6347
  %4919 = extractelement <2 x i32> %4918, i32 0		; visa id: 6351
  %4920 = extractelement <2 x i32> %4918, i32 1		; visa id: 6351
  %4921 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4919, i32 %4920, i32 %44, i32 %45)
  %4922 = extractvalue { i32, i32 } %4921, 0		; visa id: 6351
  %4923 = extractvalue { i32, i32 } %4921, 1		; visa id: 6351
  %4924 = insertelement <2 x i32> undef, i32 %4922, i32 0		; visa id: 6358
  %4925 = insertelement <2 x i32> %4924, i32 %4923, i32 1		; visa id: 6359
  %4926 = bitcast <2 x i32> %4925 to i64		; visa id: 6360
  %4927 = shl i64 %4926, 1		; visa id: 6364
  %4928 = add i64 %.in400, %4927		; visa id: 6365
  %4929 = ashr i64 %4912, 31		; visa id: 6366
  %4930 = bitcast i64 %4929 to <2 x i32>		; visa id: 6367
  %4931 = extractelement <2 x i32> %4930, i32 0		; visa id: 6371
  %4932 = extractelement <2 x i32> %4930, i32 1		; visa id: 6371
  %4933 = and i32 %4931, -2		; visa id: 6371
  %4934 = insertelement <2 x i32> undef, i32 %4933, i32 0		; visa id: 6372
  %4935 = insertelement <2 x i32> %4934, i32 %4932, i32 1		; visa id: 6373
  %4936 = bitcast <2 x i32> %4935 to i64		; visa id: 6374
  %4937 = add i64 %4928, %4936		; visa id: 6378
  %4938 = inttoptr i64 %4937 to i16 addrspace(4)*		; visa id: 6379
  %4939 = addrspacecast i16 addrspace(4)* %4938 to i16 addrspace(1)*		; visa id: 6379
  %4940 = load i16, i16 addrspace(1)* %4939, align 2		; visa id: 6380
  %4941 = zext i16 %4897 to i32		; visa id: 6382
  %4942 = shl nuw i32 %4941, 16, !spirv.Decorations !639		; visa id: 6383
  %4943 = bitcast i32 %4942 to float
  %4944 = zext i16 %4940 to i32		; visa id: 6384
  %4945 = shl nuw i32 %4944, 16, !spirv.Decorations !639		; visa id: 6385
  %4946 = bitcast i32 %4945 to float
  %4947 = fmul reassoc nsz arcp contract float %4943, %4946, !spirv.Decorations !618
  %4948 = fadd reassoc nsz arcp contract float %4947, %.sroa.50.1, !spirv.Decorations !618		; visa id: 6386
  br label %._crit_edge.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6387

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %4911
; BB447 :
  %.sroa.50.2 = phi float [ %4948, %4911 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  %4949 = icmp slt i32 %230, %const_reg_dword
  %4950 = icmp slt i32 %4850, %const_reg_dword1		; visa id: 6388
  %4951 = and i1 %4949, %4950		; visa id: 6389
  br i1 %4951, label %4952, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6391

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
; BB:
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

4952:                                             ; preds = %._crit_edge.12
; BB449 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6393
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6393
  %4953 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 6393
  %4954 = insertelement <2 x i32> %4953, i32 %113, i64 1		; visa id: 6394
  %4955 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6395
  store <2 x i32> %4954, <2 x i32>* %4955, align 4, !noalias !625		; visa id: 6395
  br label %._crit_edge305, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6397

._crit_edge305:                                   ; preds = %._crit_edge305.._crit_edge305_crit_edge, %4952
; BB450 :
  %4956 = phi i32 [ 0, %4952 ], [ %4965, %._crit_edge305.._crit_edge305_crit_edge ]
  %4957 = zext i32 %4956 to i64		; visa id: 6398
  %4958 = shl nuw nsw i64 %4957, 2		; visa id: 6399
  %4959 = add i64 %133, %4958		; visa id: 6400
  %4960 = inttoptr i64 %4959 to i32*		; visa id: 6401
  %4961 = load i32, i32* %4960, align 4, !noalias !625		; visa id: 6401
  %4962 = add i64 %128, %4958		; visa id: 6402
  %4963 = inttoptr i64 %4962 to i32*		; visa id: 6403
  store i32 %4961, i32* %4963, align 4, !alias.scope !625		; visa id: 6403
  %4964 = icmp eq i32 %4956, 0		; visa id: 6404
  br i1 %4964, label %._crit_edge305.._crit_edge305_crit_edge, label %4966, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6405

._crit_edge305.._crit_edge305_crit_edge:          ; preds = %._crit_edge305
; BB451 :
  %4965 = add nuw nsw i32 %4956, 1, !spirv.Decorations !631		; visa id: 6407
  br label %._crit_edge305, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6408

4966:                                             ; preds = %._crit_edge305
; BB452 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6410
  %4967 = load i64, i64* %129, align 8		; visa id: 6410
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6411
  %4968 = bitcast i64 %4967 to <2 x i32>		; visa id: 6411
  %4969 = extractelement <2 x i32> %4968, i32 0		; visa id: 6413
  %4970 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %4969, i32 1
  %4971 = bitcast <2 x i32> %4970 to i64		; visa id: 6413
  %4972 = ashr exact i64 %4971, 32		; visa id: 6414
  %4973 = bitcast i64 %4972 to <2 x i32>		; visa id: 6415
  %4974 = extractelement <2 x i32> %4973, i32 0		; visa id: 6419
  %4975 = extractelement <2 x i32> %4973, i32 1		; visa id: 6419
  %4976 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %4974, i32 %4975, i32 %41, i32 %42)
  %4977 = extractvalue { i32, i32 } %4976, 0		; visa id: 6419
  %4978 = extractvalue { i32, i32 } %4976, 1		; visa id: 6419
  %4979 = insertelement <2 x i32> undef, i32 %4977, i32 0		; visa id: 6426
  %4980 = insertelement <2 x i32> %4979, i32 %4978, i32 1		; visa id: 6427
  %4981 = bitcast <2 x i32> %4980 to i64		; visa id: 6428
  %4982 = shl i64 %4981, 1		; visa id: 6432
  %4983 = add i64 %.in401, %4982		; visa id: 6433
  %4984 = ashr i64 %4967, 31		; visa id: 6434
  %4985 = bitcast i64 %4984 to <2 x i32>		; visa id: 6435
  %4986 = extractelement <2 x i32> %4985, i32 0		; visa id: 6439
  %4987 = extractelement <2 x i32> %4985, i32 1		; visa id: 6439
  %4988 = and i32 %4986, -2		; visa id: 6439
  %4989 = insertelement <2 x i32> undef, i32 %4988, i32 0		; visa id: 6440
  %4990 = insertelement <2 x i32> %4989, i32 %4987, i32 1		; visa id: 6441
  %4991 = bitcast <2 x i32> %4990 to i64		; visa id: 6442
  %4992 = add i64 %4983, %4991		; visa id: 6446
  %4993 = inttoptr i64 %4992 to i16 addrspace(4)*		; visa id: 6447
  %4994 = addrspacecast i16 addrspace(4)* %4993 to i16 addrspace(1)*		; visa id: 6447
  %4995 = load i16, i16 addrspace(1)* %4994, align 2		; visa id: 6448
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6450
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6450
  %4996 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6450
  %4997 = insertelement <2 x i32> %4996, i32 %4850, i64 1		; visa id: 6451
  %4998 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6452
  store <2 x i32> %4997, <2 x i32>* %4998, align 4, !noalias !635		; visa id: 6452
  br label %._crit_edge306, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6454

._crit_edge306:                                   ; preds = %._crit_edge306.._crit_edge306_crit_edge, %4966
; BB453 :
  %4999 = phi i32 [ 0, %4966 ], [ %5008, %._crit_edge306.._crit_edge306_crit_edge ]
  %5000 = zext i32 %4999 to i64		; visa id: 6455
  %5001 = shl nuw nsw i64 %5000, 2		; visa id: 6456
  %5002 = add i64 %124, %5001		; visa id: 6457
  %5003 = inttoptr i64 %5002 to i32*		; visa id: 6458
  %5004 = load i32, i32* %5003, align 4, !noalias !635		; visa id: 6458
  %5005 = add i64 %119, %5001		; visa id: 6459
  %5006 = inttoptr i64 %5005 to i32*		; visa id: 6460
  store i32 %5004, i32* %5006, align 4, !alias.scope !635		; visa id: 6460
  %5007 = icmp eq i32 %4999, 0		; visa id: 6461
  br i1 %5007, label %._crit_edge306.._crit_edge306_crit_edge, label %5009, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6462

._crit_edge306.._crit_edge306_crit_edge:          ; preds = %._crit_edge306
; BB454 :
  %5008 = add nuw nsw i32 %4999, 1, !spirv.Decorations !631		; visa id: 6464
  br label %._crit_edge306, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6465

5009:                                             ; preds = %._crit_edge306
; BB455 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6467
  %5010 = load i64, i64* %120, align 8		; visa id: 6467
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6468
  %5011 = bitcast i64 %5010 to <2 x i32>		; visa id: 6468
  %5012 = extractelement <2 x i32> %5011, i32 0		; visa id: 6470
  %5013 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5012, i32 1
  %5014 = bitcast <2 x i32> %5013 to i64		; visa id: 6470
  %5015 = ashr exact i64 %5014, 32		; visa id: 6471
  %5016 = bitcast i64 %5015 to <2 x i32>		; visa id: 6472
  %5017 = extractelement <2 x i32> %5016, i32 0		; visa id: 6476
  %5018 = extractelement <2 x i32> %5016, i32 1		; visa id: 6476
  %5019 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5017, i32 %5018, i32 %44, i32 %45)
  %5020 = extractvalue { i32, i32 } %5019, 0		; visa id: 6476
  %5021 = extractvalue { i32, i32 } %5019, 1		; visa id: 6476
  %5022 = insertelement <2 x i32> undef, i32 %5020, i32 0		; visa id: 6483
  %5023 = insertelement <2 x i32> %5022, i32 %5021, i32 1		; visa id: 6484
  %5024 = bitcast <2 x i32> %5023 to i64		; visa id: 6485
  %5025 = shl i64 %5024, 1		; visa id: 6489
  %5026 = add i64 %.in400, %5025		; visa id: 6490
  %5027 = ashr i64 %5010, 31		; visa id: 6491
  %5028 = bitcast i64 %5027 to <2 x i32>		; visa id: 6492
  %5029 = extractelement <2 x i32> %5028, i32 0		; visa id: 6496
  %5030 = extractelement <2 x i32> %5028, i32 1		; visa id: 6496
  %5031 = and i32 %5029, -2		; visa id: 6496
  %5032 = insertelement <2 x i32> undef, i32 %5031, i32 0		; visa id: 6497
  %5033 = insertelement <2 x i32> %5032, i32 %5030, i32 1		; visa id: 6498
  %5034 = bitcast <2 x i32> %5033 to i64		; visa id: 6499
  %5035 = add i64 %5026, %5034		; visa id: 6503
  %5036 = inttoptr i64 %5035 to i16 addrspace(4)*		; visa id: 6504
  %5037 = addrspacecast i16 addrspace(4)* %5036 to i16 addrspace(1)*		; visa id: 6504
  %5038 = load i16, i16 addrspace(1)* %5037, align 2		; visa id: 6505
  %5039 = zext i16 %4995 to i32		; visa id: 6507
  %5040 = shl nuw i32 %5039, 16, !spirv.Decorations !639		; visa id: 6508
  %5041 = bitcast i32 %5040 to float
  %5042 = zext i16 %5038 to i32		; visa id: 6509
  %5043 = shl nuw i32 %5042, 16, !spirv.Decorations !639		; visa id: 6510
  %5044 = bitcast i32 %5043 to float
  %5045 = fmul reassoc nsz arcp contract float %5041, %5044, !spirv.Decorations !618
  %5046 = fadd reassoc nsz arcp contract float %5045, %.sroa.114.1, !spirv.Decorations !618		; visa id: 6511
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6512

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %5009
; BB456 :
  %.sroa.114.2 = phi float [ %5046, %5009 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  %5047 = icmp slt i32 %329, %const_reg_dword
  %5048 = icmp slt i32 %4850, %const_reg_dword1		; visa id: 6513
  %5049 = and i1 %5047, %5048		; visa id: 6514
  br i1 %5049, label %5050, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6516

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
; BB:
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5050:                                             ; preds = %._crit_edge.1.12
; BB458 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6518
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6518
  %5051 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 6518
  %5052 = insertelement <2 x i32> %5051, i32 %113, i64 1		; visa id: 6519
  %5053 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6520
  store <2 x i32> %5052, <2 x i32>* %5053, align 4, !noalias !625		; visa id: 6520
  br label %._crit_edge307, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6522

._crit_edge307:                                   ; preds = %._crit_edge307.._crit_edge307_crit_edge, %5050
; BB459 :
  %5054 = phi i32 [ 0, %5050 ], [ %5063, %._crit_edge307.._crit_edge307_crit_edge ]
  %5055 = zext i32 %5054 to i64		; visa id: 6523
  %5056 = shl nuw nsw i64 %5055, 2		; visa id: 6524
  %5057 = add i64 %133, %5056		; visa id: 6525
  %5058 = inttoptr i64 %5057 to i32*		; visa id: 6526
  %5059 = load i32, i32* %5058, align 4, !noalias !625		; visa id: 6526
  %5060 = add i64 %128, %5056		; visa id: 6527
  %5061 = inttoptr i64 %5060 to i32*		; visa id: 6528
  store i32 %5059, i32* %5061, align 4, !alias.scope !625		; visa id: 6528
  %5062 = icmp eq i32 %5054, 0		; visa id: 6529
  br i1 %5062, label %._crit_edge307.._crit_edge307_crit_edge, label %5064, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6530

._crit_edge307.._crit_edge307_crit_edge:          ; preds = %._crit_edge307
; BB460 :
  %5063 = add nuw nsw i32 %5054, 1, !spirv.Decorations !631		; visa id: 6532
  br label %._crit_edge307, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6533

5064:                                             ; preds = %._crit_edge307
; BB461 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6535
  %5065 = load i64, i64* %129, align 8		; visa id: 6535
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6536
  %5066 = bitcast i64 %5065 to <2 x i32>		; visa id: 6536
  %5067 = extractelement <2 x i32> %5066, i32 0		; visa id: 6538
  %5068 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5067, i32 1
  %5069 = bitcast <2 x i32> %5068 to i64		; visa id: 6538
  %5070 = ashr exact i64 %5069, 32		; visa id: 6539
  %5071 = bitcast i64 %5070 to <2 x i32>		; visa id: 6540
  %5072 = extractelement <2 x i32> %5071, i32 0		; visa id: 6544
  %5073 = extractelement <2 x i32> %5071, i32 1		; visa id: 6544
  %5074 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5072, i32 %5073, i32 %41, i32 %42)
  %5075 = extractvalue { i32, i32 } %5074, 0		; visa id: 6544
  %5076 = extractvalue { i32, i32 } %5074, 1		; visa id: 6544
  %5077 = insertelement <2 x i32> undef, i32 %5075, i32 0		; visa id: 6551
  %5078 = insertelement <2 x i32> %5077, i32 %5076, i32 1		; visa id: 6552
  %5079 = bitcast <2 x i32> %5078 to i64		; visa id: 6553
  %5080 = shl i64 %5079, 1		; visa id: 6557
  %5081 = add i64 %.in401, %5080		; visa id: 6558
  %5082 = ashr i64 %5065, 31		; visa id: 6559
  %5083 = bitcast i64 %5082 to <2 x i32>		; visa id: 6560
  %5084 = extractelement <2 x i32> %5083, i32 0		; visa id: 6564
  %5085 = extractelement <2 x i32> %5083, i32 1		; visa id: 6564
  %5086 = and i32 %5084, -2		; visa id: 6564
  %5087 = insertelement <2 x i32> undef, i32 %5086, i32 0		; visa id: 6565
  %5088 = insertelement <2 x i32> %5087, i32 %5085, i32 1		; visa id: 6566
  %5089 = bitcast <2 x i32> %5088 to i64		; visa id: 6567
  %5090 = add i64 %5081, %5089		; visa id: 6571
  %5091 = inttoptr i64 %5090 to i16 addrspace(4)*		; visa id: 6572
  %5092 = addrspacecast i16 addrspace(4)* %5091 to i16 addrspace(1)*		; visa id: 6572
  %5093 = load i16, i16 addrspace(1)* %5092, align 2		; visa id: 6573
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6575
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6575
  %5094 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6575
  %5095 = insertelement <2 x i32> %5094, i32 %4850, i64 1		; visa id: 6576
  %5096 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6577
  store <2 x i32> %5095, <2 x i32>* %5096, align 4, !noalias !635		; visa id: 6577
  br label %._crit_edge308, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6579

._crit_edge308:                                   ; preds = %._crit_edge308.._crit_edge308_crit_edge, %5064
; BB462 :
  %5097 = phi i32 [ 0, %5064 ], [ %5106, %._crit_edge308.._crit_edge308_crit_edge ]
  %5098 = zext i32 %5097 to i64		; visa id: 6580
  %5099 = shl nuw nsw i64 %5098, 2		; visa id: 6581
  %5100 = add i64 %124, %5099		; visa id: 6582
  %5101 = inttoptr i64 %5100 to i32*		; visa id: 6583
  %5102 = load i32, i32* %5101, align 4, !noalias !635		; visa id: 6583
  %5103 = add i64 %119, %5099		; visa id: 6584
  %5104 = inttoptr i64 %5103 to i32*		; visa id: 6585
  store i32 %5102, i32* %5104, align 4, !alias.scope !635		; visa id: 6585
  %5105 = icmp eq i32 %5097, 0		; visa id: 6586
  br i1 %5105, label %._crit_edge308.._crit_edge308_crit_edge, label %5107, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6587

._crit_edge308.._crit_edge308_crit_edge:          ; preds = %._crit_edge308
; BB463 :
  %5106 = add nuw nsw i32 %5097, 1, !spirv.Decorations !631		; visa id: 6589
  br label %._crit_edge308, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6590

5107:                                             ; preds = %._crit_edge308
; BB464 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6592
  %5108 = load i64, i64* %120, align 8		; visa id: 6592
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6593
  %5109 = bitcast i64 %5108 to <2 x i32>		; visa id: 6593
  %5110 = extractelement <2 x i32> %5109, i32 0		; visa id: 6595
  %5111 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5110, i32 1
  %5112 = bitcast <2 x i32> %5111 to i64		; visa id: 6595
  %5113 = ashr exact i64 %5112, 32		; visa id: 6596
  %5114 = bitcast i64 %5113 to <2 x i32>		; visa id: 6597
  %5115 = extractelement <2 x i32> %5114, i32 0		; visa id: 6601
  %5116 = extractelement <2 x i32> %5114, i32 1		; visa id: 6601
  %5117 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5115, i32 %5116, i32 %44, i32 %45)
  %5118 = extractvalue { i32, i32 } %5117, 0		; visa id: 6601
  %5119 = extractvalue { i32, i32 } %5117, 1		; visa id: 6601
  %5120 = insertelement <2 x i32> undef, i32 %5118, i32 0		; visa id: 6608
  %5121 = insertelement <2 x i32> %5120, i32 %5119, i32 1		; visa id: 6609
  %5122 = bitcast <2 x i32> %5121 to i64		; visa id: 6610
  %5123 = shl i64 %5122, 1		; visa id: 6614
  %5124 = add i64 %.in400, %5123		; visa id: 6615
  %5125 = ashr i64 %5108, 31		; visa id: 6616
  %5126 = bitcast i64 %5125 to <2 x i32>		; visa id: 6617
  %5127 = extractelement <2 x i32> %5126, i32 0		; visa id: 6621
  %5128 = extractelement <2 x i32> %5126, i32 1		; visa id: 6621
  %5129 = and i32 %5127, -2		; visa id: 6621
  %5130 = insertelement <2 x i32> undef, i32 %5129, i32 0		; visa id: 6622
  %5131 = insertelement <2 x i32> %5130, i32 %5128, i32 1		; visa id: 6623
  %5132 = bitcast <2 x i32> %5131 to i64		; visa id: 6624
  %5133 = add i64 %5124, %5132		; visa id: 6628
  %5134 = inttoptr i64 %5133 to i16 addrspace(4)*		; visa id: 6629
  %5135 = addrspacecast i16 addrspace(4)* %5134 to i16 addrspace(1)*		; visa id: 6629
  %5136 = load i16, i16 addrspace(1)* %5135, align 2		; visa id: 6630
  %5137 = zext i16 %5093 to i32		; visa id: 6632
  %5138 = shl nuw i32 %5137, 16, !spirv.Decorations !639		; visa id: 6633
  %5139 = bitcast i32 %5138 to float
  %5140 = zext i16 %5136 to i32		; visa id: 6634
  %5141 = shl nuw i32 %5140, 16, !spirv.Decorations !639		; visa id: 6635
  %5142 = bitcast i32 %5141 to float
  %5143 = fmul reassoc nsz arcp contract float %5139, %5142, !spirv.Decorations !618
  %5144 = fadd reassoc nsz arcp contract float %5143, %.sroa.178.1, !spirv.Decorations !618		; visa id: 6636
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6637

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %5107
; BB465 :
  %.sroa.178.2 = phi float [ %5144, %5107 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  %5145 = icmp slt i32 %428, %const_reg_dword
  %5146 = icmp slt i32 %4850, %const_reg_dword1		; visa id: 6638
  %5147 = and i1 %5145, %5146		; visa id: 6639
  br i1 %5147, label %5148, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6641

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
; BB:
  br label %.preheader.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5148:                                             ; preds = %._crit_edge.2.12
; BB467 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6643
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6643
  %5149 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 6643
  %5150 = insertelement <2 x i32> %5149, i32 %113, i64 1		; visa id: 6644
  %5151 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6645
  store <2 x i32> %5150, <2 x i32>* %5151, align 4, !noalias !625		; visa id: 6645
  br label %._crit_edge309, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6647

._crit_edge309:                                   ; preds = %._crit_edge309.._crit_edge309_crit_edge, %5148
; BB468 :
  %5152 = phi i32 [ 0, %5148 ], [ %5161, %._crit_edge309.._crit_edge309_crit_edge ]
  %5153 = zext i32 %5152 to i64		; visa id: 6648
  %5154 = shl nuw nsw i64 %5153, 2		; visa id: 6649
  %5155 = add i64 %133, %5154		; visa id: 6650
  %5156 = inttoptr i64 %5155 to i32*		; visa id: 6651
  %5157 = load i32, i32* %5156, align 4, !noalias !625		; visa id: 6651
  %5158 = add i64 %128, %5154		; visa id: 6652
  %5159 = inttoptr i64 %5158 to i32*		; visa id: 6653
  store i32 %5157, i32* %5159, align 4, !alias.scope !625		; visa id: 6653
  %5160 = icmp eq i32 %5152, 0		; visa id: 6654
  br i1 %5160, label %._crit_edge309.._crit_edge309_crit_edge, label %5162, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6655

._crit_edge309.._crit_edge309_crit_edge:          ; preds = %._crit_edge309
; BB469 :
  %5161 = add nuw nsw i32 %5152, 1, !spirv.Decorations !631		; visa id: 6657
  br label %._crit_edge309, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6658

5162:                                             ; preds = %._crit_edge309
; BB470 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6660
  %5163 = load i64, i64* %129, align 8		; visa id: 6660
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6661
  %5164 = bitcast i64 %5163 to <2 x i32>		; visa id: 6661
  %5165 = extractelement <2 x i32> %5164, i32 0		; visa id: 6663
  %5166 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5165, i32 1
  %5167 = bitcast <2 x i32> %5166 to i64		; visa id: 6663
  %5168 = ashr exact i64 %5167, 32		; visa id: 6664
  %5169 = bitcast i64 %5168 to <2 x i32>		; visa id: 6665
  %5170 = extractelement <2 x i32> %5169, i32 0		; visa id: 6669
  %5171 = extractelement <2 x i32> %5169, i32 1		; visa id: 6669
  %5172 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5170, i32 %5171, i32 %41, i32 %42)
  %5173 = extractvalue { i32, i32 } %5172, 0		; visa id: 6669
  %5174 = extractvalue { i32, i32 } %5172, 1		; visa id: 6669
  %5175 = insertelement <2 x i32> undef, i32 %5173, i32 0		; visa id: 6676
  %5176 = insertelement <2 x i32> %5175, i32 %5174, i32 1		; visa id: 6677
  %5177 = bitcast <2 x i32> %5176 to i64		; visa id: 6678
  %5178 = shl i64 %5177, 1		; visa id: 6682
  %5179 = add i64 %.in401, %5178		; visa id: 6683
  %5180 = ashr i64 %5163, 31		; visa id: 6684
  %5181 = bitcast i64 %5180 to <2 x i32>		; visa id: 6685
  %5182 = extractelement <2 x i32> %5181, i32 0		; visa id: 6689
  %5183 = extractelement <2 x i32> %5181, i32 1		; visa id: 6689
  %5184 = and i32 %5182, -2		; visa id: 6689
  %5185 = insertelement <2 x i32> undef, i32 %5184, i32 0		; visa id: 6690
  %5186 = insertelement <2 x i32> %5185, i32 %5183, i32 1		; visa id: 6691
  %5187 = bitcast <2 x i32> %5186 to i64		; visa id: 6692
  %5188 = add i64 %5179, %5187		; visa id: 6696
  %5189 = inttoptr i64 %5188 to i16 addrspace(4)*		; visa id: 6697
  %5190 = addrspacecast i16 addrspace(4)* %5189 to i16 addrspace(1)*		; visa id: 6697
  %5191 = load i16, i16 addrspace(1)* %5190, align 2		; visa id: 6698
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6700
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6700
  %5192 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6700
  %5193 = insertelement <2 x i32> %5192, i32 %4850, i64 1		; visa id: 6701
  %5194 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6702
  store <2 x i32> %5193, <2 x i32>* %5194, align 4, !noalias !635		; visa id: 6702
  br label %._crit_edge310, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6704

._crit_edge310:                                   ; preds = %._crit_edge310.._crit_edge310_crit_edge, %5162
; BB471 :
  %5195 = phi i32 [ 0, %5162 ], [ %5204, %._crit_edge310.._crit_edge310_crit_edge ]
  %5196 = zext i32 %5195 to i64		; visa id: 6705
  %5197 = shl nuw nsw i64 %5196, 2		; visa id: 6706
  %5198 = add i64 %124, %5197		; visa id: 6707
  %5199 = inttoptr i64 %5198 to i32*		; visa id: 6708
  %5200 = load i32, i32* %5199, align 4, !noalias !635		; visa id: 6708
  %5201 = add i64 %119, %5197		; visa id: 6709
  %5202 = inttoptr i64 %5201 to i32*		; visa id: 6710
  store i32 %5200, i32* %5202, align 4, !alias.scope !635		; visa id: 6710
  %5203 = icmp eq i32 %5195, 0		; visa id: 6711
  br i1 %5203, label %._crit_edge310.._crit_edge310_crit_edge, label %5205, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6712

._crit_edge310.._crit_edge310_crit_edge:          ; preds = %._crit_edge310
; BB472 :
  %5204 = add nuw nsw i32 %5195, 1, !spirv.Decorations !631		; visa id: 6714
  br label %._crit_edge310, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6715

5205:                                             ; preds = %._crit_edge310
; BB473 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6717
  %5206 = load i64, i64* %120, align 8		; visa id: 6717
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6718
  %5207 = bitcast i64 %5206 to <2 x i32>		; visa id: 6718
  %5208 = extractelement <2 x i32> %5207, i32 0		; visa id: 6720
  %5209 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5208, i32 1
  %5210 = bitcast <2 x i32> %5209 to i64		; visa id: 6720
  %5211 = ashr exact i64 %5210, 32		; visa id: 6721
  %5212 = bitcast i64 %5211 to <2 x i32>		; visa id: 6722
  %5213 = extractelement <2 x i32> %5212, i32 0		; visa id: 6726
  %5214 = extractelement <2 x i32> %5212, i32 1		; visa id: 6726
  %5215 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5213, i32 %5214, i32 %44, i32 %45)
  %5216 = extractvalue { i32, i32 } %5215, 0		; visa id: 6726
  %5217 = extractvalue { i32, i32 } %5215, 1		; visa id: 6726
  %5218 = insertelement <2 x i32> undef, i32 %5216, i32 0		; visa id: 6733
  %5219 = insertelement <2 x i32> %5218, i32 %5217, i32 1		; visa id: 6734
  %5220 = bitcast <2 x i32> %5219 to i64		; visa id: 6735
  %5221 = shl i64 %5220, 1		; visa id: 6739
  %5222 = add i64 %.in400, %5221		; visa id: 6740
  %5223 = ashr i64 %5206, 31		; visa id: 6741
  %5224 = bitcast i64 %5223 to <2 x i32>		; visa id: 6742
  %5225 = extractelement <2 x i32> %5224, i32 0		; visa id: 6746
  %5226 = extractelement <2 x i32> %5224, i32 1		; visa id: 6746
  %5227 = and i32 %5225, -2		; visa id: 6746
  %5228 = insertelement <2 x i32> undef, i32 %5227, i32 0		; visa id: 6747
  %5229 = insertelement <2 x i32> %5228, i32 %5226, i32 1		; visa id: 6748
  %5230 = bitcast <2 x i32> %5229 to i64		; visa id: 6749
  %5231 = add i64 %5222, %5230		; visa id: 6753
  %5232 = inttoptr i64 %5231 to i16 addrspace(4)*		; visa id: 6754
  %5233 = addrspacecast i16 addrspace(4)* %5232 to i16 addrspace(1)*		; visa id: 6754
  %5234 = load i16, i16 addrspace(1)* %5233, align 2		; visa id: 6755
  %5235 = zext i16 %5191 to i32		; visa id: 6757
  %5236 = shl nuw i32 %5235, 16, !spirv.Decorations !639		; visa id: 6758
  %5237 = bitcast i32 %5236 to float
  %5238 = zext i16 %5234 to i32		; visa id: 6759
  %5239 = shl nuw i32 %5238, 16, !spirv.Decorations !639		; visa id: 6760
  %5240 = bitcast i32 %5239 to float
  %5241 = fmul reassoc nsz arcp contract float %5237, %5240, !spirv.Decorations !618
  %5242 = fadd reassoc nsz arcp contract float %5241, %.sroa.242.1, !spirv.Decorations !618		; visa id: 6761
  br label %.preheader.12, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6762

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %5205
; BB474 :
  %.sroa.242.2 = phi float [ %5242, %5205 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %5243 = add i32 %69, 13		; visa id: 6763
  %5244 = icmp slt i32 %5243, %const_reg_dword1		; visa id: 6764
  %5245 = icmp slt i32 %65, %const_reg_dword
  %5246 = and i1 %5245, %5244		; visa id: 6765
  br i1 %5246, label %5247, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6767

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
; BB:
  br label %._crit_edge.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5247:                                             ; preds = %.preheader.12
; BB476 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6769
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6769
  %5248 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 6769
  %5249 = insertelement <2 x i32> %5248, i32 %113, i64 1		; visa id: 6770
  %5250 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6771
  store <2 x i32> %5249, <2 x i32>* %5250, align 4, !noalias !625		; visa id: 6771
  br label %._crit_edge311, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6773

._crit_edge311:                                   ; preds = %._crit_edge311.._crit_edge311_crit_edge, %5247
; BB477 :
  %5251 = phi i32 [ 0, %5247 ], [ %5260, %._crit_edge311.._crit_edge311_crit_edge ]
  %5252 = zext i32 %5251 to i64		; visa id: 6774
  %5253 = shl nuw nsw i64 %5252, 2		; visa id: 6775
  %5254 = add i64 %133, %5253		; visa id: 6776
  %5255 = inttoptr i64 %5254 to i32*		; visa id: 6777
  %5256 = load i32, i32* %5255, align 4, !noalias !625		; visa id: 6777
  %5257 = add i64 %128, %5253		; visa id: 6778
  %5258 = inttoptr i64 %5257 to i32*		; visa id: 6779
  store i32 %5256, i32* %5258, align 4, !alias.scope !625		; visa id: 6779
  %5259 = icmp eq i32 %5251, 0		; visa id: 6780
  br i1 %5259, label %._crit_edge311.._crit_edge311_crit_edge, label %5261, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6781

._crit_edge311.._crit_edge311_crit_edge:          ; preds = %._crit_edge311
; BB478 :
  %5260 = add nuw nsw i32 %5251, 1, !spirv.Decorations !631		; visa id: 6783
  br label %._crit_edge311, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6784

5261:                                             ; preds = %._crit_edge311
; BB479 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6786
  %5262 = load i64, i64* %129, align 8		; visa id: 6786
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6787
  %5263 = bitcast i64 %5262 to <2 x i32>		; visa id: 6787
  %5264 = extractelement <2 x i32> %5263, i32 0		; visa id: 6789
  %5265 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5264, i32 1
  %5266 = bitcast <2 x i32> %5265 to i64		; visa id: 6789
  %5267 = ashr exact i64 %5266, 32		; visa id: 6790
  %5268 = bitcast i64 %5267 to <2 x i32>		; visa id: 6791
  %5269 = extractelement <2 x i32> %5268, i32 0		; visa id: 6795
  %5270 = extractelement <2 x i32> %5268, i32 1		; visa id: 6795
  %5271 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5269, i32 %5270, i32 %41, i32 %42)
  %5272 = extractvalue { i32, i32 } %5271, 0		; visa id: 6795
  %5273 = extractvalue { i32, i32 } %5271, 1		; visa id: 6795
  %5274 = insertelement <2 x i32> undef, i32 %5272, i32 0		; visa id: 6802
  %5275 = insertelement <2 x i32> %5274, i32 %5273, i32 1		; visa id: 6803
  %5276 = bitcast <2 x i32> %5275 to i64		; visa id: 6804
  %5277 = shl i64 %5276, 1		; visa id: 6808
  %5278 = add i64 %.in401, %5277		; visa id: 6809
  %5279 = ashr i64 %5262, 31		; visa id: 6810
  %5280 = bitcast i64 %5279 to <2 x i32>		; visa id: 6811
  %5281 = extractelement <2 x i32> %5280, i32 0		; visa id: 6815
  %5282 = extractelement <2 x i32> %5280, i32 1		; visa id: 6815
  %5283 = and i32 %5281, -2		; visa id: 6815
  %5284 = insertelement <2 x i32> undef, i32 %5283, i32 0		; visa id: 6816
  %5285 = insertelement <2 x i32> %5284, i32 %5282, i32 1		; visa id: 6817
  %5286 = bitcast <2 x i32> %5285 to i64		; visa id: 6818
  %5287 = add i64 %5278, %5286		; visa id: 6822
  %5288 = inttoptr i64 %5287 to i16 addrspace(4)*		; visa id: 6823
  %5289 = addrspacecast i16 addrspace(4)* %5288 to i16 addrspace(1)*		; visa id: 6823
  %5290 = load i16, i16 addrspace(1)* %5289, align 2		; visa id: 6824
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6826
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6826
  %5291 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6826
  %5292 = insertelement <2 x i32> %5291, i32 %5243, i64 1		; visa id: 6827
  %5293 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6828
  store <2 x i32> %5292, <2 x i32>* %5293, align 4, !noalias !635		; visa id: 6828
  br label %._crit_edge312, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6830

._crit_edge312:                                   ; preds = %._crit_edge312.._crit_edge312_crit_edge, %5261
; BB480 :
  %5294 = phi i32 [ 0, %5261 ], [ %5303, %._crit_edge312.._crit_edge312_crit_edge ]
  %5295 = zext i32 %5294 to i64		; visa id: 6831
  %5296 = shl nuw nsw i64 %5295, 2		; visa id: 6832
  %5297 = add i64 %124, %5296		; visa id: 6833
  %5298 = inttoptr i64 %5297 to i32*		; visa id: 6834
  %5299 = load i32, i32* %5298, align 4, !noalias !635		; visa id: 6834
  %5300 = add i64 %119, %5296		; visa id: 6835
  %5301 = inttoptr i64 %5300 to i32*		; visa id: 6836
  store i32 %5299, i32* %5301, align 4, !alias.scope !635		; visa id: 6836
  %5302 = icmp eq i32 %5294, 0		; visa id: 6837
  br i1 %5302, label %._crit_edge312.._crit_edge312_crit_edge, label %5304, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6838

._crit_edge312.._crit_edge312_crit_edge:          ; preds = %._crit_edge312
; BB481 :
  %5303 = add nuw nsw i32 %5294, 1, !spirv.Decorations !631		; visa id: 6840
  br label %._crit_edge312, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6841

5304:                                             ; preds = %._crit_edge312
; BB482 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6843
  %5305 = load i64, i64* %120, align 8		; visa id: 6843
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6844
  %5306 = bitcast i64 %5305 to <2 x i32>		; visa id: 6844
  %5307 = extractelement <2 x i32> %5306, i32 0		; visa id: 6846
  %5308 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5307, i32 1
  %5309 = bitcast <2 x i32> %5308 to i64		; visa id: 6846
  %5310 = ashr exact i64 %5309, 32		; visa id: 6847
  %5311 = bitcast i64 %5310 to <2 x i32>		; visa id: 6848
  %5312 = extractelement <2 x i32> %5311, i32 0		; visa id: 6852
  %5313 = extractelement <2 x i32> %5311, i32 1		; visa id: 6852
  %5314 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5312, i32 %5313, i32 %44, i32 %45)
  %5315 = extractvalue { i32, i32 } %5314, 0		; visa id: 6852
  %5316 = extractvalue { i32, i32 } %5314, 1		; visa id: 6852
  %5317 = insertelement <2 x i32> undef, i32 %5315, i32 0		; visa id: 6859
  %5318 = insertelement <2 x i32> %5317, i32 %5316, i32 1		; visa id: 6860
  %5319 = bitcast <2 x i32> %5318 to i64		; visa id: 6861
  %5320 = shl i64 %5319, 1		; visa id: 6865
  %5321 = add i64 %.in400, %5320		; visa id: 6866
  %5322 = ashr i64 %5305, 31		; visa id: 6867
  %5323 = bitcast i64 %5322 to <2 x i32>		; visa id: 6868
  %5324 = extractelement <2 x i32> %5323, i32 0		; visa id: 6872
  %5325 = extractelement <2 x i32> %5323, i32 1		; visa id: 6872
  %5326 = and i32 %5324, -2		; visa id: 6872
  %5327 = insertelement <2 x i32> undef, i32 %5326, i32 0		; visa id: 6873
  %5328 = insertelement <2 x i32> %5327, i32 %5325, i32 1		; visa id: 6874
  %5329 = bitcast <2 x i32> %5328 to i64		; visa id: 6875
  %5330 = add i64 %5321, %5329		; visa id: 6879
  %5331 = inttoptr i64 %5330 to i16 addrspace(4)*		; visa id: 6880
  %5332 = addrspacecast i16 addrspace(4)* %5331 to i16 addrspace(1)*		; visa id: 6880
  %5333 = load i16, i16 addrspace(1)* %5332, align 2		; visa id: 6881
  %5334 = zext i16 %5290 to i32		; visa id: 6883
  %5335 = shl nuw i32 %5334, 16, !spirv.Decorations !639		; visa id: 6884
  %5336 = bitcast i32 %5335 to float
  %5337 = zext i16 %5333 to i32		; visa id: 6885
  %5338 = shl nuw i32 %5337, 16, !spirv.Decorations !639		; visa id: 6886
  %5339 = bitcast i32 %5338 to float
  %5340 = fmul reassoc nsz arcp contract float %5336, %5339, !spirv.Decorations !618
  %5341 = fadd reassoc nsz arcp contract float %5340, %.sroa.54.1, !spirv.Decorations !618		; visa id: 6887
  br label %._crit_edge.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6888

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %5304
; BB483 :
  %.sroa.54.2 = phi float [ %5341, %5304 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  %5342 = icmp slt i32 %230, %const_reg_dword
  %5343 = icmp slt i32 %5243, %const_reg_dword1		; visa id: 6889
  %5344 = and i1 %5342, %5343		; visa id: 6890
  br i1 %5344, label %5345, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 6892

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
; BB:
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5345:                                             ; preds = %._crit_edge.13
; BB485 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 6894
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 6894
  %5346 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 6894
  %5347 = insertelement <2 x i32> %5346, i32 %113, i64 1		; visa id: 6895
  %5348 = inttoptr i64 %133 to <2 x i32>*		; visa id: 6896
  store <2 x i32> %5347, <2 x i32>* %5348, align 4, !noalias !625		; visa id: 6896
  br label %._crit_edge313, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6898

._crit_edge313:                                   ; preds = %._crit_edge313.._crit_edge313_crit_edge, %5345
; BB486 :
  %5349 = phi i32 [ 0, %5345 ], [ %5358, %._crit_edge313.._crit_edge313_crit_edge ]
  %5350 = zext i32 %5349 to i64		; visa id: 6899
  %5351 = shl nuw nsw i64 %5350, 2		; visa id: 6900
  %5352 = add i64 %133, %5351		; visa id: 6901
  %5353 = inttoptr i64 %5352 to i32*		; visa id: 6902
  %5354 = load i32, i32* %5353, align 4, !noalias !625		; visa id: 6902
  %5355 = add i64 %128, %5351		; visa id: 6903
  %5356 = inttoptr i64 %5355 to i32*		; visa id: 6904
  store i32 %5354, i32* %5356, align 4, !alias.scope !625		; visa id: 6904
  %5357 = icmp eq i32 %5349, 0		; visa id: 6905
  br i1 %5357, label %._crit_edge313.._crit_edge313_crit_edge, label %5359, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6906

._crit_edge313.._crit_edge313_crit_edge:          ; preds = %._crit_edge313
; BB487 :
  %5358 = add nuw nsw i32 %5349, 1, !spirv.Decorations !631		; visa id: 6908
  br label %._crit_edge313, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6909

5359:                                             ; preds = %._crit_edge313
; BB488 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 6911
  %5360 = load i64, i64* %129, align 8		; visa id: 6911
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 6912
  %5361 = bitcast i64 %5360 to <2 x i32>		; visa id: 6912
  %5362 = extractelement <2 x i32> %5361, i32 0		; visa id: 6914
  %5363 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5362, i32 1
  %5364 = bitcast <2 x i32> %5363 to i64		; visa id: 6914
  %5365 = ashr exact i64 %5364, 32		; visa id: 6915
  %5366 = bitcast i64 %5365 to <2 x i32>		; visa id: 6916
  %5367 = extractelement <2 x i32> %5366, i32 0		; visa id: 6920
  %5368 = extractelement <2 x i32> %5366, i32 1		; visa id: 6920
  %5369 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5367, i32 %5368, i32 %41, i32 %42)
  %5370 = extractvalue { i32, i32 } %5369, 0		; visa id: 6920
  %5371 = extractvalue { i32, i32 } %5369, 1		; visa id: 6920
  %5372 = insertelement <2 x i32> undef, i32 %5370, i32 0		; visa id: 6927
  %5373 = insertelement <2 x i32> %5372, i32 %5371, i32 1		; visa id: 6928
  %5374 = bitcast <2 x i32> %5373 to i64		; visa id: 6929
  %5375 = shl i64 %5374, 1		; visa id: 6933
  %5376 = add i64 %.in401, %5375		; visa id: 6934
  %5377 = ashr i64 %5360, 31		; visa id: 6935
  %5378 = bitcast i64 %5377 to <2 x i32>		; visa id: 6936
  %5379 = extractelement <2 x i32> %5378, i32 0		; visa id: 6940
  %5380 = extractelement <2 x i32> %5378, i32 1		; visa id: 6940
  %5381 = and i32 %5379, -2		; visa id: 6940
  %5382 = insertelement <2 x i32> undef, i32 %5381, i32 0		; visa id: 6941
  %5383 = insertelement <2 x i32> %5382, i32 %5380, i32 1		; visa id: 6942
  %5384 = bitcast <2 x i32> %5383 to i64		; visa id: 6943
  %5385 = add i64 %5376, %5384		; visa id: 6947
  %5386 = inttoptr i64 %5385 to i16 addrspace(4)*		; visa id: 6948
  %5387 = addrspacecast i16 addrspace(4)* %5386 to i16 addrspace(1)*		; visa id: 6948
  %5388 = load i16, i16 addrspace(1)* %5387, align 2		; visa id: 6949
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 6951
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 6951
  %5389 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 6951
  %5390 = insertelement <2 x i32> %5389, i32 %5243, i64 1		; visa id: 6952
  %5391 = inttoptr i64 %124 to <2 x i32>*		; visa id: 6953
  store <2 x i32> %5390, <2 x i32>* %5391, align 4, !noalias !635		; visa id: 6953
  br label %._crit_edge314, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 6955

._crit_edge314:                                   ; preds = %._crit_edge314.._crit_edge314_crit_edge, %5359
; BB489 :
  %5392 = phi i32 [ 0, %5359 ], [ %5401, %._crit_edge314.._crit_edge314_crit_edge ]
  %5393 = zext i32 %5392 to i64		; visa id: 6956
  %5394 = shl nuw nsw i64 %5393, 2		; visa id: 6957
  %5395 = add i64 %124, %5394		; visa id: 6958
  %5396 = inttoptr i64 %5395 to i32*		; visa id: 6959
  %5397 = load i32, i32* %5396, align 4, !noalias !635		; visa id: 6959
  %5398 = add i64 %119, %5394		; visa id: 6960
  %5399 = inttoptr i64 %5398 to i32*		; visa id: 6961
  store i32 %5397, i32* %5399, align 4, !alias.scope !635		; visa id: 6961
  %5400 = icmp eq i32 %5392, 0		; visa id: 6962
  br i1 %5400, label %._crit_edge314.._crit_edge314_crit_edge, label %5402, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 6963

._crit_edge314.._crit_edge314_crit_edge:          ; preds = %._crit_edge314
; BB490 :
  %5401 = add nuw nsw i32 %5392, 1, !spirv.Decorations !631		; visa id: 6965
  br label %._crit_edge314, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 6966

5402:                                             ; preds = %._crit_edge314
; BB491 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 6968
  %5403 = load i64, i64* %120, align 8		; visa id: 6968
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 6969
  %5404 = bitcast i64 %5403 to <2 x i32>		; visa id: 6969
  %5405 = extractelement <2 x i32> %5404, i32 0		; visa id: 6971
  %5406 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5405, i32 1
  %5407 = bitcast <2 x i32> %5406 to i64		; visa id: 6971
  %5408 = ashr exact i64 %5407, 32		; visa id: 6972
  %5409 = bitcast i64 %5408 to <2 x i32>		; visa id: 6973
  %5410 = extractelement <2 x i32> %5409, i32 0		; visa id: 6977
  %5411 = extractelement <2 x i32> %5409, i32 1		; visa id: 6977
  %5412 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5410, i32 %5411, i32 %44, i32 %45)
  %5413 = extractvalue { i32, i32 } %5412, 0		; visa id: 6977
  %5414 = extractvalue { i32, i32 } %5412, 1		; visa id: 6977
  %5415 = insertelement <2 x i32> undef, i32 %5413, i32 0		; visa id: 6984
  %5416 = insertelement <2 x i32> %5415, i32 %5414, i32 1		; visa id: 6985
  %5417 = bitcast <2 x i32> %5416 to i64		; visa id: 6986
  %5418 = shl i64 %5417, 1		; visa id: 6990
  %5419 = add i64 %.in400, %5418		; visa id: 6991
  %5420 = ashr i64 %5403, 31		; visa id: 6992
  %5421 = bitcast i64 %5420 to <2 x i32>		; visa id: 6993
  %5422 = extractelement <2 x i32> %5421, i32 0		; visa id: 6997
  %5423 = extractelement <2 x i32> %5421, i32 1		; visa id: 6997
  %5424 = and i32 %5422, -2		; visa id: 6997
  %5425 = insertelement <2 x i32> undef, i32 %5424, i32 0		; visa id: 6998
  %5426 = insertelement <2 x i32> %5425, i32 %5423, i32 1		; visa id: 6999
  %5427 = bitcast <2 x i32> %5426 to i64		; visa id: 7000
  %5428 = add i64 %5419, %5427		; visa id: 7004
  %5429 = inttoptr i64 %5428 to i16 addrspace(4)*		; visa id: 7005
  %5430 = addrspacecast i16 addrspace(4)* %5429 to i16 addrspace(1)*		; visa id: 7005
  %5431 = load i16, i16 addrspace(1)* %5430, align 2		; visa id: 7006
  %5432 = zext i16 %5388 to i32		; visa id: 7008
  %5433 = shl nuw i32 %5432, 16, !spirv.Decorations !639		; visa id: 7009
  %5434 = bitcast i32 %5433 to float
  %5435 = zext i16 %5431 to i32		; visa id: 7010
  %5436 = shl nuw i32 %5435, 16, !spirv.Decorations !639		; visa id: 7011
  %5437 = bitcast i32 %5436 to float
  %5438 = fmul reassoc nsz arcp contract float %5434, %5437, !spirv.Decorations !618
  %5439 = fadd reassoc nsz arcp contract float %5438, %.sroa.118.1, !spirv.Decorations !618		; visa id: 7012
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7013

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %5402
; BB492 :
  %.sroa.118.2 = phi float [ %5439, %5402 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  %5440 = icmp slt i32 %329, %const_reg_dword
  %5441 = icmp slt i32 %5243, %const_reg_dword1		; visa id: 7014
  %5442 = and i1 %5440, %5441		; visa id: 7015
  br i1 %5442, label %5443, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7017

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
; BB:
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5443:                                             ; preds = %._crit_edge.1.13
; BB494 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7019
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7019
  %5444 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 7019
  %5445 = insertelement <2 x i32> %5444, i32 %113, i64 1		; visa id: 7020
  %5446 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7021
  store <2 x i32> %5445, <2 x i32>* %5446, align 4, !noalias !625		; visa id: 7021
  br label %._crit_edge315, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7023

._crit_edge315:                                   ; preds = %._crit_edge315.._crit_edge315_crit_edge, %5443
; BB495 :
  %5447 = phi i32 [ 0, %5443 ], [ %5456, %._crit_edge315.._crit_edge315_crit_edge ]
  %5448 = zext i32 %5447 to i64		; visa id: 7024
  %5449 = shl nuw nsw i64 %5448, 2		; visa id: 7025
  %5450 = add i64 %133, %5449		; visa id: 7026
  %5451 = inttoptr i64 %5450 to i32*		; visa id: 7027
  %5452 = load i32, i32* %5451, align 4, !noalias !625		; visa id: 7027
  %5453 = add i64 %128, %5449		; visa id: 7028
  %5454 = inttoptr i64 %5453 to i32*		; visa id: 7029
  store i32 %5452, i32* %5454, align 4, !alias.scope !625		; visa id: 7029
  %5455 = icmp eq i32 %5447, 0		; visa id: 7030
  br i1 %5455, label %._crit_edge315.._crit_edge315_crit_edge, label %5457, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7031

._crit_edge315.._crit_edge315_crit_edge:          ; preds = %._crit_edge315
; BB496 :
  %5456 = add nuw nsw i32 %5447, 1, !spirv.Decorations !631		; visa id: 7033
  br label %._crit_edge315, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7034

5457:                                             ; preds = %._crit_edge315
; BB497 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7036
  %5458 = load i64, i64* %129, align 8		; visa id: 7036
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7037
  %5459 = bitcast i64 %5458 to <2 x i32>		; visa id: 7037
  %5460 = extractelement <2 x i32> %5459, i32 0		; visa id: 7039
  %5461 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5460, i32 1
  %5462 = bitcast <2 x i32> %5461 to i64		; visa id: 7039
  %5463 = ashr exact i64 %5462, 32		; visa id: 7040
  %5464 = bitcast i64 %5463 to <2 x i32>		; visa id: 7041
  %5465 = extractelement <2 x i32> %5464, i32 0		; visa id: 7045
  %5466 = extractelement <2 x i32> %5464, i32 1		; visa id: 7045
  %5467 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5465, i32 %5466, i32 %41, i32 %42)
  %5468 = extractvalue { i32, i32 } %5467, 0		; visa id: 7045
  %5469 = extractvalue { i32, i32 } %5467, 1		; visa id: 7045
  %5470 = insertelement <2 x i32> undef, i32 %5468, i32 0		; visa id: 7052
  %5471 = insertelement <2 x i32> %5470, i32 %5469, i32 1		; visa id: 7053
  %5472 = bitcast <2 x i32> %5471 to i64		; visa id: 7054
  %5473 = shl i64 %5472, 1		; visa id: 7058
  %5474 = add i64 %.in401, %5473		; visa id: 7059
  %5475 = ashr i64 %5458, 31		; visa id: 7060
  %5476 = bitcast i64 %5475 to <2 x i32>		; visa id: 7061
  %5477 = extractelement <2 x i32> %5476, i32 0		; visa id: 7065
  %5478 = extractelement <2 x i32> %5476, i32 1		; visa id: 7065
  %5479 = and i32 %5477, -2		; visa id: 7065
  %5480 = insertelement <2 x i32> undef, i32 %5479, i32 0		; visa id: 7066
  %5481 = insertelement <2 x i32> %5480, i32 %5478, i32 1		; visa id: 7067
  %5482 = bitcast <2 x i32> %5481 to i64		; visa id: 7068
  %5483 = add i64 %5474, %5482		; visa id: 7072
  %5484 = inttoptr i64 %5483 to i16 addrspace(4)*		; visa id: 7073
  %5485 = addrspacecast i16 addrspace(4)* %5484 to i16 addrspace(1)*		; visa id: 7073
  %5486 = load i16, i16 addrspace(1)* %5485, align 2		; visa id: 7074
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7076
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7076
  %5487 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7076
  %5488 = insertelement <2 x i32> %5487, i32 %5243, i64 1		; visa id: 7077
  %5489 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7078
  store <2 x i32> %5488, <2 x i32>* %5489, align 4, !noalias !635		; visa id: 7078
  br label %._crit_edge316, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7080

._crit_edge316:                                   ; preds = %._crit_edge316.._crit_edge316_crit_edge, %5457
; BB498 :
  %5490 = phi i32 [ 0, %5457 ], [ %5499, %._crit_edge316.._crit_edge316_crit_edge ]
  %5491 = zext i32 %5490 to i64		; visa id: 7081
  %5492 = shl nuw nsw i64 %5491, 2		; visa id: 7082
  %5493 = add i64 %124, %5492		; visa id: 7083
  %5494 = inttoptr i64 %5493 to i32*		; visa id: 7084
  %5495 = load i32, i32* %5494, align 4, !noalias !635		; visa id: 7084
  %5496 = add i64 %119, %5492		; visa id: 7085
  %5497 = inttoptr i64 %5496 to i32*		; visa id: 7086
  store i32 %5495, i32* %5497, align 4, !alias.scope !635		; visa id: 7086
  %5498 = icmp eq i32 %5490, 0		; visa id: 7087
  br i1 %5498, label %._crit_edge316.._crit_edge316_crit_edge, label %5500, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7088

._crit_edge316.._crit_edge316_crit_edge:          ; preds = %._crit_edge316
; BB499 :
  %5499 = add nuw nsw i32 %5490, 1, !spirv.Decorations !631		; visa id: 7090
  br label %._crit_edge316, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7091

5500:                                             ; preds = %._crit_edge316
; BB500 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7093
  %5501 = load i64, i64* %120, align 8		; visa id: 7093
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7094
  %5502 = bitcast i64 %5501 to <2 x i32>		; visa id: 7094
  %5503 = extractelement <2 x i32> %5502, i32 0		; visa id: 7096
  %5504 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5503, i32 1
  %5505 = bitcast <2 x i32> %5504 to i64		; visa id: 7096
  %5506 = ashr exact i64 %5505, 32		; visa id: 7097
  %5507 = bitcast i64 %5506 to <2 x i32>		; visa id: 7098
  %5508 = extractelement <2 x i32> %5507, i32 0		; visa id: 7102
  %5509 = extractelement <2 x i32> %5507, i32 1		; visa id: 7102
  %5510 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5508, i32 %5509, i32 %44, i32 %45)
  %5511 = extractvalue { i32, i32 } %5510, 0		; visa id: 7102
  %5512 = extractvalue { i32, i32 } %5510, 1		; visa id: 7102
  %5513 = insertelement <2 x i32> undef, i32 %5511, i32 0		; visa id: 7109
  %5514 = insertelement <2 x i32> %5513, i32 %5512, i32 1		; visa id: 7110
  %5515 = bitcast <2 x i32> %5514 to i64		; visa id: 7111
  %5516 = shl i64 %5515, 1		; visa id: 7115
  %5517 = add i64 %.in400, %5516		; visa id: 7116
  %5518 = ashr i64 %5501, 31		; visa id: 7117
  %5519 = bitcast i64 %5518 to <2 x i32>		; visa id: 7118
  %5520 = extractelement <2 x i32> %5519, i32 0		; visa id: 7122
  %5521 = extractelement <2 x i32> %5519, i32 1		; visa id: 7122
  %5522 = and i32 %5520, -2		; visa id: 7122
  %5523 = insertelement <2 x i32> undef, i32 %5522, i32 0		; visa id: 7123
  %5524 = insertelement <2 x i32> %5523, i32 %5521, i32 1		; visa id: 7124
  %5525 = bitcast <2 x i32> %5524 to i64		; visa id: 7125
  %5526 = add i64 %5517, %5525		; visa id: 7129
  %5527 = inttoptr i64 %5526 to i16 addrspace(4)*		; visa id: 7130
  %5528 = addrspacecast i16 addrspace(4)* %5527 to i16 addrspace(1)*		; visa id: 7130
  %5529 = load i16, i16 addrspace(1)* %5528, align 2		; visa id: 7131
  %5530 = zext i16 %5486 to i32		; visa id: 7133
  %5531 = shl nuw i32 %5530, 16, !spirv.Decorations !639		; visa id: 7134
  %5532 = bitcast i32 %5531 to float
  %5533 = zext i16 %5529 to i32		; visa id: 7135
  %5534 = shl nuw i32 %5533, 16, !spirv.Decorations !639		; visa id: 7136
  %5535 = bitcast i32 %5534 to float
  %5536 = fmul reassoc nsz arcp contract float %5532, %5535, !spirv.Decorations !618
  %5537 = fadd reassoc nsz arcp contract float %5536, %.sroa.182.1, !spirv.Decorations !618		; visa id: 7137
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7138

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %5500
; BB501 :
  %.sroa.182.2 = phi float [ %5537, %5500 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  %5538 = icmp slt i32 %428, %const_reg_dword
  %5539 = icmp slt i32 %5243, %const_reg_dword1		; visa id: 7139
  %5540 = and i1 %5538, %5539		; visa id: 7140
  br i1 %5540, label %5541, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7142

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
; BB:
  br label %.preheader.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5541:                                             ; preds = %._crit_edge.2.13
; BB503 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7144
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7144
  %5542 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 7144
  %5543 = insertelement <2 x i32> %5542, i32 %113, i64 1		; visa id: 7145
  %5544 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7146
  store <2 x i32> %5543, <2 x i32>* %5544, align 4, !noalias !625		; visa id: 7146
  br label %._crit_edge317, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7148

._crit_edge317:                                   ; preds = %._crit_edge317.._crit_edge317_crit_edge, %5541
; BB504 :
  %5545 = phi i32 [ 0, %5541 ], [ %5554, %._crit_edge317.._crit_edge317_crit_edge ]
  %5546 = zext i32 %5545 to i64		; visa id: 7149
  %5547 = shl nuw nsw i64 %5546, 2		; visa id: 7150
  %5548 = add i64 %133, %5547		; visa id: 7151
  %5549 = inttoptr i64 %5548 to i32*		; visa id: 7152
  %5550 = load i32, i32* %5549, align 4, !noalias !625		; visa id: 7152
  %5551 = add i64 %128, %5547		; visa id: 7153
  %5552 = inttoptr i64 %5551 to i32*		; visa id: 7154
  store i32 %5550, i32* %5552, align 4, !alias.scope !625		; visa id: 7154
  %5553 = icmp eq i32 %5545, 0		; visa id: 7155
  br i1 %5553, label %._crit_edge317.._crit_edge317_crit_edge, label %5555, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7156

._crit_edge317.._crit_edge317_crit_edge:          ; preds = %._crit_edge317
; BB505 :
  %5554 = add nuw nsw i32 %5545, 1, !spirv.Decorations !631		; visa id: 7158
  br label %._crit_edge317, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7159

5555:                                             ; preds = %._crit_edge317
; BB506 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7161
  %5556 = load i64, i64* %129, align 8		; visa id: 7161
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7162
  %5557 = bitcast i64 %5556 to <2 x i32>		; visa id: 7162
  %5558 = extractelement <2 x i32> %5557, i32 0		; visa id: 7164
  %5559 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5558, i32 1
  %5560 = bitcast <2 x i32> %5559 to i64		; visa id: 7164
  %5561 = ashr exact i64 %5560, 32		; visa id: 7165
  %5562 = bitcast i64 %5561 to <2 x i32>		; visa id: 7166
  %5563 = extractelement <2 x i32> %5562, i32 0		; visa id: 7170
  %5564 = extractelement <2 x i32> %5562, i32 1		; visa id: 7170
  %5565 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5563, i32 %5564, i32 %41, i32 %42)
  %5566 = extractvalue { i32, i32 } %5565, 0		; visa id: 7170
  %5567 = extractvalue { i32, i32 } %5565, 1		; visa id: 7170
  %5568 = insertelement <2 x i32> undef, i32 %5566, i32 0		; visa id: 7177
  %5569 = insertelement <2 x i32> %5568, i32 %5567, i32 1		; visa id: 7178
  %5570 = bitcast <2 x i32> %5569 to i64		; visa id: 7179
  %5571 = shl i64 %5570, 1		; visa id: 7183
  %5572 = add i64 %.in401, %5571		; visa id: 7184
  %5573 = ashr i64 %5556, 31		; visa id: 7185
  %5574 = bitcast i64 %5573 to <2 x i32>		; visa id: 7186
  %5575 = extractelement <2 x i32> %5574, i32 0		; visa id: 7190
  %5576 = extractelement <2 x i32> %5574, i32 1		; visa id: 7190
  %5577 = and i32 %5575, -2		; visa id: 7190
  %5578 = insertelement <2 x i32> undef, i32 %5577, i32 0		; visa id: 7191
  %5579 = insertelement <2 x i32> %5578, i32 %5576, i32 1		; visa id: 7192
  %5580 = bitcast <2 x i32> %5579 to i64		; visa id: 7193
  %5581 = add i64 %5572, %5580		; visa id: 7197
  %5582 = inttoptr i64 %5581 to i16 addrspace(4)*		; visa id: 7198
  %5583 = addrspacecast i16 addrspace(4)* %5582 to i16 addrspace(1)*		; visa id: 7198
  %5584 = load i16, i16 addrspace(1)* %5583, align 2		; visa id: 7199
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7201
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7201
  %5585 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7201
  %5586 = insertelement <2 x i32> %5585, i32 %5243, i64 1		; visa id: 7202
  %5587 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7203
  store <2 x i32> %5586, <2 x i32>* %5587, align 4, !noalias !635		; visa id: 7203
  br label %._crit_edge318, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7205

._crit_edge318:                                   ; preds = %._crit_edge318.._crit_edge318_crit_edge, %5555
; BB507 :
  %5588 = phi i32 [ 0, %5555 ], [ %5597, %._crit_edge318.._crit_edge318_crit_edge ]
  %5589 = zext i32 %5588 to i64		; visa id: 7206
  %5590 = shl nuw nsw i64 %5589, 2		; visa id: 7207
  %5591 = add i64 %124, %5590		; visa id: 7208
  %5592 = inttoptr i64 %5591 to i32*		; visa id: 7209
  %5593 = load i32, i32* %5592, align 4, !noalias !635		; visa id: 7209
  %5594 = add i64 %119, %5590		; visa id: 7210
  %5595 = inttoptr i64 %5594 to i32*		; visa id: 7211
  store i32 %5593, i32* %5595, align 4, !alias.scope !635		; visa id: 7211
  %5596 = icmp eq i32 %5588, 0		; visa id: 7212
  br i1 %5596, label %._crit_edge318.._crit_edge318_crit_edge, label %5598, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7213

._crit_edge318.._crit_edge318_crit_edge:          ; preds = %._crit_edge318
; BB508 :
  %5597 = add nuw nsw i32 %5588, 1, !spirv.Decorations !631		; visa id: 7215
  br label %._crit_edge318, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7216

5598:                                             ; preds = %._crit_edge318
; BB509 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7218
  %5599 = load i64, i64* %120, align 8		; visa id: 7218
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7219
  %5600 = bitcast i64 %5599 to <2 x i32>		; visa id: 7219
  %5601 = extractelement <2 x i32> %5600, i32 0		; visa id: 7221
  %5602 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5601, i32 1
  %5603 = bitcast <2 x i32> %5602 to i64		; visa id: 7221
  %5604 = ashr exact i64 %5603, 32		; visa id: 7222
  %5605 = bitcast i64 %5604 to <2 x i32>		; visa id: 7223
  %5606 = extractelement <2 x i32> %5605, i32 0		; visa id: 7227
  %5607 = extractelement <2 x i32> %5605, i32 1		; visa id: 7227
  %5608 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5606, i32 %5607, i32 %44, i32 %45)
  %5609 = extractvalue { i32, i32 } %5608, 0		; visa id: 7227
  %5610 = extractvalue { i32, i32 } %5608, 1		; visa id: 7227
  %5611 = insertelement <2 x i32> undef, i32 %5609, i32 0		; visa id: 7234
  %5612 = insertelement <2 x i32> %5611, i32 %5610, i32 1		; visa id: 7235
  %5613 = bitcast <2 x i32> %5612 to i64		; visa id: 7236
  %5614 = shl i64 %5613, 1		; visa id: 7240
  %5615 = add i64 %.in400, %5614		; visa id: 7241
  %5616 = ashr i64 %5599, 31		; visa id: 7242
  %5617 = bitcast i64 %5616 to <2 x i32>		; visa id: 7243
  %5618 = extractelement <2 x i32> %5617, i32 0		; visa id: 7247
  %5619 = extractelement <2 x i32> %5617, i32 1		; visa id: 7247
  %5620 = and i32 %5618, -2		; visa id: 7247
  %5621 = insertelement <2 x i32> undef, i32 %5620, i32 0		; visa id: 7248
  %5622 = insertelement <2 x i32> %5621, i32 %5619, i32 1		; visa id: 7249
  %5623 = bitcast <2 x i32> %5622 to i64		; visa id: 7250
  %5624 = add i64 %5615, %5623		; visa id: 7254
  %5625 = inttoptr i64 %5624 to i16 addrspace(4)*		; visa id: 7255
  %5626 = addrspacecast i16 addrspace(4)* %5625 to i16 addrspace(1)*		; visa id: 7255
  %5627 = load i16, i16 addrspace(1)* %5626, align 2		; visa id: 7256
  %5628 = zext i16 %5584 to i32		; visa id: 7258
  %5629 = shl nuw i32 %5628, 16, !spirv.Decorations !639		; visa id: 7259
  %5630 = bitcast i32 %5629 to float
  %5631 = zext i16 %5627 to i32		; visa id: 7260
  %5632 = shl nuw i32 %5631, 16, !spirv.Decorations !639		; visa id: 7261
  %5633 = bitcast i32 %5632 to float
  %5634 = fmul reassoc nsz arcp contract float %5630, %5633, !spirv.Decorations !618
  %5635 = fadd reassoc nsz arcp contract float %5634, %.sroa.246.1, !spirv.Decorations !618		; visa id: 7262
  br label %.preheader.13, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7263

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %5598
; BB510 :
  %.sroa.246.2 = phi float [ %5635, %5598 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %5636 = add i32 %69, 14		; visa id: 7264
  %5637 = icmp slt i32 %5636, %const_reg_dword1		; visa id: 7265
  %5638 = icmp slt i32 %65, %const_reg_dword
  %5639 = and i1 %5638, %5637		; visa id: 7266
  br i1 %5639, label %5640, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7268

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
; BB:
  br label %._crit_edge.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5640:                                             ; preds = %.preheader.13
; BB512 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7270
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7270
  %5641 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 7270
  %5642 = insertelement <2 x i32> %5641, i32 %113, i64 1		; visa id: 7271
  %5643 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7272
  store <2 x i32> %5642, <2 x i32>* %5643, align 4, !noalias !625		; visa id: 7272
  br label %._crit_edge319, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7274

._crit_edge319:                                   ; preds = %._crit_edge319.._crit_edge319_crit_edge, %5640
; BB513 :
  %5644 = phi i32 [ 0, %5640 ], [ %5653, %._crit_edge319.._crit_edge319_crit_edge ]
  %5645 = zext i32 %5644 to i64		; visa id: 7275
  %5646 = shl nuw nsw i64 %5645, 2		; visa id: 7276
  %5647 = add i64 %133, %5646		; visa id: 7277
  %5648 = inttoptr i64 %5647 to i32*		; visa id: 7278
  %5649 = load i32, i32* %5648, align 4, !noalias !625		; visa id: 7278
  %5650 = add i64 %128, %5646		; visa id: 7279
  %5651 = inttoptr i64 %5650 to i32*		; visa id: 7280
  store i32 %5649, i32* %5651, align 4, !alias.scope !625		; visa id: 7280
  %5652 = icmp eq i32 %5644, 0		; visa id: 7281
  br i1 %5652, label %._crit_edge319.._crit_edge319_crit_edge, label %5654, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7282

._crit_edge319.._crit_edge319_crit_edge:          ; preds = %._crit_edge319
; BB514 :
  %5653 = add nuw nsw i32 %5644, 1, !spirv.Decorations !631		; visa id: 7284
  br label %._crit_edge319, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7285

5654:                                             ; preds = %._crit_edge319
; BB515 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7287
  %5655 = load i64, i64* %129, align 8		; visa id: 7287
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7288
  %5656 = bitcast i64 %5655 to <2 x i32>		; visa id: 7288
  %5657 = extractelement <2 x i32> %5656, i32 0		; visa id: 7290
  %5658 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5657, i32 1
  %5659 = bitcast <2 x i32> %5658 to i64		; visa id: 7290
  %5660 = ashr exact i64 %5659, 32		; visa id: 7291
  %5661 = bitcast i64 %5660 to <2 x i32>		; visa id: 7292
  %5662 = extractelement <2 x i32> %5661, i32 0		; visa id: 7296
  %5663 = extractelement <2 x i32> %5661, i32 1		; visa id: 7296
  %5664 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5662, i32 %5663, i32 %41, i32 %42)
  %5665 = extractvalue { i32, i32 } %5664, 0		; visa id: 7296
  %5666 = extractvalue { i32, i32 } %5664, 1		; visa id: 7296
  %5667 = insertelement <2 x i32> undef, i32 %5665, i32 0		; visa id: 7303
  %5668 = insertelement <2 x i32> %5667, i32 %5666, i32 1		; visa id: 7304
  %5669 = bitcast <2 x i32> %5668 to i64		; visa id: 7305
  %5670 = shl i64 %5669, 1		; visa id: 7309
  %5671 = add i64 %.in401, %5670		; visa id: 7310
  %5672 = ashr i64 %5655, 31		; visa id: 7311
  %5673 = bitcast i64 %5672 to <2 x i32>		; visa id: 7312
  %5674 = extractelement <2 x i32> %5673, i32 0		; visa id: 7316
  %5675 = extractelement <2 x i32> %5673, i32 1		; visa id: 7316
  %5676 = and i32 %5674, -2		; visa id: 7316
  %5677 = insertelement <2 x i32> undef, i32 %5676, i32 0		; visa id: 7317
  %5678 = insertelement <2 x i32> %5677, i32 %5675, i32 1		; visa id: 7318
  %5679 = bitcast <2 x i32> %5678 to i64		; visa id: 7319
  %5680 = add i64 %5671, %5679		; visa id: 7323
  %5681 = inttoptr i64 %5680 to i16 addrspace(4)*		; visa id: 7324
  %5682 = addrspacecast i16 addrspace(4)* %5681 to i16 addrspace(1)*		; visa id: 7324
  %5683 = load i16, i16 addrspace(1)* %5682, align 2		; visa id: 7325
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7327
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7327
  %5684 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7327
  %5685 = insertelement <2 x i32> %5684, i32 %5636, i64 1		; visa id: 7328
  %5686 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7329
  store <2 x i32> %5685, <2 x i32>* %5686, align 4, !noalias !635		; visa id: 7329
  br label %._crit_edge320, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7331

._crit_edge320:                                   ; preds = %._crit_edge320.._crit_edge320_crit_edge, %5654
; BB516 :
  %5687 = phi i32 [ 0, %5654 ], [ %5696, %._crit_edge320.._crit_edge320_crit_edge ]
  %5688 = zext i32 %5687 to i64		; visa id: 7332
  %5689 = shl nuw nsw i64 %5688, 2		; visa id: 7333
  %5690 = add i64 %124, %5689		; visa id: 7334
  %5691 = inttoptr i64 %5690 to i32*		; visa id: 7335
  %5692 = load i32, i32* %5691, align 4, !noalias !635		; visa id: 7335
  %5693 = add i64 %119, %5689		; visa id: 7336
  %5694 = inttoptr i64 %5693 to i32*		; visa id: 7337
  store i32 %5692, i32* %5694, align 4, !alias.scope !635		; visa id: 7337
  %5695 = icmp eq i32 %5687, 0		; visa id: 7338
  br i1 %5695, label %._crit_edge320.._crit_edge320_crit_edge, label %5697, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7339

._crit_edge320.._crit_edge320_crit_edge:          ; preds = %._crit_edge320
; BB517 :
  %5696 = add nuw nsw i32 %5687, 1, !spirv.Decorations !631		; visa id: 7341
  br label %._crit_edge320, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7342

5697:                                             ; preds = %._crit_edge320
; BB518 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7344
  %5698 = load i64, i64* %120, align 8		; visa id: 7344
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7345
  %5699 = bitcast i64 %5698 to <2 x i32>		; visa id: 7345
  %5700 = extractelement <2 x i32> %5699, i32 0		; visa id: 7347
  %5701 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5700, i32 1
  %5702 = bitcast <2 x i32> %5701 to i64		; visa id: 7347
  %5703 = ashr exact i64 %5702, 32		; visa id: 7348
  %5704 = bitcast i64 %5703 to <2 x i32>		; visa id: 7349
  %5705 = extractelement <2 x i32> %5704, i32 0		; visa id: 7353
  %5706 = extractelement <2 x i32> %5704, i32 1		; visa id: 7353
  %5707 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5705, i32 %5706, i32 %44, i32 %45)
  %5708 = extractvalue { i32, i32 } %5707, 0		; visa id: 7353
  %5709 = extractvalue { i32, i32 } %5707, 1		; visa id: 7353
  %5710 = insertelement <2 x i32> undef, i32 %5708, i32 0		; visa id: 7360
  %5711 = insertelement <2 x i32> %5710, i32 %5709, i32 1		; visa id: 7361
  %5712 = bitcast <2 x i32> %5711 to i64		; visa id: 7362
  %5713 = shl i64 %5712, 1		; visa id: 7366
  %5714 = add i64 %.in400, %5713		; visa id: 7367
  %5715 = ashr i64 %5698, 31		; visa id: 7368
  %5716 = bitcast i64 %5715 to <2 x i32>		; visa id: 7369
  %5717 = extractelement <2 x i32> %5716, i32 0		; visa id: 7373
  %5718 = extractelement <2 x i32> %5716, i32 1		; visa id: 7373
  %5719 = and i32 %5717, -2		; visa id: 7373
  %5720 = insertelement <2 x i32> undef, i32 %5719, i32 0		; visa id: 7374
  %5721 = insertelement <2 x i32> %5720, i32 %5718, i32 1		; visa id: 7375
  %5722 = bitcast <2 x i32> %5721 to i64		; visa id: 7376
  %5723 = add i64 %5714, %5722		; visa id: 7380
  %5724 = inttoptr i64 %5723 to i16 addrspace(4)*		; visa id: 7381
  %5725 = addrspacecast i16 addrspace(4)* %5724 to i16 addrspace(1)*		; visa id: 7381
  %5726 = load i16, i16 addrspace(1)* %5725, align 2		; visa id: 7382
  %5727 = zext i16 %5683 to i32		; visa id: 7384
  %5728 = shl nuw i32 %5727, 16, !spirv.Decorations !639		; visa id: 7385
  %5729 = bitcast i32 %5728 to float
  %5730 = zext i16 %5726 to i32		; visa id: 7386
  %5731 = shl nuw i32 %5730, 16, !spirv.Decorations !639		; visa id: 7387
  %5732 = bitcast i32 %5731 to float
  %5733 = fmul reassoc nsz arcp contract float %5729, %5732, !spirv.Decorations !618
  %5734 = fadd reassoc nsz arcp contract float %5733, %.sroa.58.1, !spirv.Decorations !618		; visa id: 7388
  br label %._crit_edge.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7389

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %5697
; BB519 :
  %.sroa.58.2 = phi float [ %5734, %5697 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  %5735 = icmp slt i32 %230, %const_reg_dword
  %5736 = icmp slt i32 %5636, %const_reg_dword1		; visa id: 7390
  %5737 = and i1 %5735, %5736		; visa id: 7391
  br i1 %5737, label %5738, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7393

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
; BB:
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5738:                                             ; preds = %._crit_edge.14
; BB521 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7395
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7395
  %5739 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 7395
  %5740 = insertelement <2 x i32> %5739, i32 %113, i64 1		; visa id: 7396
  %5741 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7397
  store <2 x i32> %5740, <2 x i32>* %5741, align 4, !noalias !625		; visa id: 7397
  br label %._crit_edge321, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7399

._crit_edge321:                                   ; preds = %._crit_edge321.._crit_edge321_crit_edge, %5738
; BB522 :
  %5742 = phi i32 [ 0, %5738 ], [ %5751, %._crit_edge321.._crit_edge321_crit_edge ]
  %5743 = zext i32 %5742 to i64		; visa id: 7400
  %5744 = shl nuw nsw i64 %5743, 2		; visa id: 7401
  %5745 = add i64 %133, %5744		; visa id: 7402
  %5746 = inttoptr i64 %5745 to i32*		; visa id: 7403
  %5747 = load i32, i32* %5746, align 4, !noalias !625		; visa id: 7403
  %5748 = add i64 %128, %5744		; visa id: 7404
  %5749 = inttoptr i64 %5748 to i32*		; visa id: 7405
  store i32 %5747, i32* %5749, align 4, !alias.scope !625		; visa id: 7405
  %5750 = icmp eq i32 %5742, 0		; visa id: 7406
  br i1 %5750, label %._crit_edge321.._crit_edge321_crit_edge, label %5752, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7407

._crit_edge321.._crit_edge321_crit_edge:          ; preds = %._crit_edge321
; BB523 :
  %5751 = add nuw nsw i32 %5742, 1, !spirv.Decorations !631		; visa id: 7409
  br label %._crit_edge321, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7410

5752:                                             ; preds = %._crit_edge321
; BB524 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7412
  %5753 = load i64, i64* %129, align 8		; visa id: 7412
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7413
  %5754 = bitcast i64 %5753 to <2 x i32>		; visa id: 7413
  %5755 = extractelement <2 x i32> %5754, i32 0		; visa id: 7415
  %5756 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5755, i32 1
  %5757 = bitcast <2 x i32> %5756 to i64		; visa id: 7415
  %5758 = ashr exact i64 %5757, 32		; visa id: 7416
  %5759 = bitcast i64 %5758 to <2 x i32>		; visa id: 7417
  %5760 = extractelement <2 x i32> %5759, i32 0		; visa id: 7421
  %5761 = extractelement <2 x i32> %5759, i32 1		; visa id: 7421
  %5762 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5760, i32 %5761, i32 %41, i32 %42)
  %5763 = extractvalue { i32, i32 } %5762, 0		; visa id: 7421
  %5764 = extractvalue { i32, i32 } %5762, 1		; visa id: 7421
  %5765 = insertelement <2 x i32> undef, i32 %5763, i32 0		; visa id: 7428
  %5766 = insertelement <2 x i32> %5765, i32 %5764, i32 1		; visa id: 7429
  %5767 = bitcast <2 x i32> %5766 to i64		; visa id: 7430
  %5768 = shl i64 %5767, 1		; visa id: 7434
  %5769 = add i64 %.in401, %5768		; visa id: 7435
  %5770 = ashr i64 %5753, 31		; visa id: 7436
  %5771 = bitcast i64 %5770 to <2 x i32>		; visa id: 7437
  %5772 = extractelement <2 x i32> %5771, i32 0		; visa id: 7441
  %5773 = extractelement <2 x i32> %5771, i32 1		; visa id: 7441
  %5774 = and i32 %5772, -2		; visa id: 7441
  %5775 = insertelement <2 x i32> undef, i32 %5774, i32 0		; visa id: 7442
  %5776 = insertelement <2 x i32> %5775, i32 %5773, i32 1		; visa id: 7443
  %5777 = bitcast <2 x i32> %5776 to i64		; visa id: 7444
  %5778 = add i64 %5769, %5777		; visa id: 7448
  %5779 = inttoptr i64 %5778 to i16 addrspace(4)*		; visa id: 7449
  %5780 = addrspacecast i16 addrspace(4)* %5779 to i16 addrspace(1)*		; visa id: 7449
  %5781 = load i16, i16 addrspace(1)* %5780, align 2		; visa id: 7450
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7452
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7452
  %5782 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7452
  %5783 = insertelement <2 x i32> %5782, i32 %5636, i64 1		; visa id: 7453
  %5784 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7454
  store <2 x i32> %5783, <2 x i32>* %5784, align 4, !noalias !635		; visa id: 7454
  br label %._crit_edge322, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7456

._crit_edge322:                                   ; preds = %._crit_edge322.._crit_edge322_crit_edge, %5752
; BB525 :
  %5785 = phi i32 [ 0, %5752 ], [ %5794, %._crit_edge322.._crit_edge322_crit_edge ]
  %5786 = zext i32 %5785 to i64		; visa id: 7457
  %5787 = shl nuw nsw i64 %5786, 2		; visa id: 7458
  %5788 = add i64 %124, %5787		; visa id: 7459
  %5789 = inttoptr i64 %5788 to i32*		; visa id: 7460
  %5790 = load i32, i32* %5789, align 4, !noalias !635		; visa id: 7460
  %5791 = add i64 %119, %5787		; visa id: 7461
  %5792 = inttoptr i64 %5791 to i32*		; visa id: 7462
  store i32 %5790, i32* %5792, align 4, !alias.scope !635		; visa id: 7462
  %5793 = icmp eq i32 %5785, 0		; visa id: 7463
  br i1 %5793, label %._crit_edge322.._crit_edge322_crit_edge, label %5795, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7464

._crit_edge322.._crit_edge322_crit_edge:          ; preds = %._crit_edge322
; BB526 :
  %5794 = add nuw nsw i32 %5785, 1, !spirv.Decorations !631		; visa id: 7466
  br label %._crit_edge322, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7467

5795:                                             ; preds = %._crit_edge322
; BB527 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7469
  %5796 = load i64, i64* %120, align 8		; visa id: 7469
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7470
  %5797 = bitcast i64 %5796 to <2 x i32>		; visa id: 7470
  %5798 = extractelement <2 x i32> %5797, i32 0		; visa id: 7472
  %5799 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5798, i32 1
  %5800 = bitcast <2 x i32> %5799 to i64		; visa id: 7472
  %5801 = ashr exact i64 %5800, 32		; visa id: 7473
  %5802 = bitcast i64 %5801 to <2 x i32>		; visa id: 7474
  %5803 = extractelement <2 x i32> %5802, i32 0		; visa id: 7478
  %5804 = extractelement <2 x i32> %5802, i32 1		; visa id: 7478
  %5805 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5803, i32 %5804, i32 %44, i32 %45)
  %5806 = extractvalue { i32, i32 } %5805, 0		; visa id: 7478
  %5807 = extractvalue { i32, i32 } %5805, 1		; visa id: 7478
  %5808 = insertelement <2 x i32> undef, i32 %5806, i32 0		; visa id: 7485
  %5809 = insertelement <2 x i32> %5808, i32 %5807, i32 1		; visa id: 7486
  %5810 = bitcast <2 x i32> %5809 to i64		; visa id: 7487
  %5811 = shl i64 %5810, 1		; visa id: 7491
  %5812 = add i64 %.in400, %5811		; visa id: 7492
  %5813 = ashr i64 %5796, 31		; visa id: 7493
  %5814 = bitcast i64 %5813 to <2 x i32>		; visa id: 7494
  %5815 = extractelement <2 x i32> %5814, i32 0		; visa id: 7498
  %5816 = extractelement <2 x i32> %5814, i32 1		; visa id: 7498
  %5817 = and i32 %5815, -2		; visa id: 7498
  %5818 = insertelement <2 x i32> undef, i32 %5817, i32 0		; visa id: 7499
  %5819 = insertelement <2 x i32> %5818, i32 %5816, i32 1		; visa id: 7500
  %5820 = bitcast <2 x i32> %5819 to i64		; visa id: 7501
  %5821 = add i64 %5812, %5820		; visa id: 7505
  %5822 = inttoptr i64 %5821 to i16 addrspace(4)*		; visa id: 7506
  %5823 = addrspacecast i16 addrspace(4)* %5822 to i16 addrspace(1)*		; visa id: 7506
  %5824 = load i16, i16 addrspace(1)* %5823, align 2		; visa id: 7507
  %5825 = zext i16 %5781 to i32		; visa id: 7509
  %5826 = shl nuw i32 %5825, 16, !spirv.Decorations !639		; visa id: 7510
  %5827 = bitcast i32 %5826 to float
  %5828 = zext i16 %5824 to i32		; visa id: 7511
  %5829 = shl nuw i32 %5828, 16, !spirv.Decorations !639		; visa id: 7512
  %5830 = bitcast i32 %5829 to float
  %5831 = fmul reassoc nsz arcp contract float %5827, %5830, !spirv.Decorations !618
  %5832 = fadd reassoc nsz arcp contract float %5831, %.sroa.122.1, !spirv.Decorations !618		; visa id: 7513
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7514

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %5795
; BB528 :
  %.sroa.122.2 = phi float [ %5832, %5795 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  %5833 = icmp slt i32 %329, %const_reg_dword
  %5834 = icmp slt i32 %5636, %const_reg_dword1		; visa id: 7515
  %5835 = and i1 %5833, %5834		; visa id: 7516
  br i1 %5835, label %5836, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7518

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
; BB:
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5836:                                             ; preds = %._crit_edge.1.14
; BB530 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7520
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7520
  %5837 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 7520
  %5838 = insertelement <2 x i32> %5837, i32 %113, i64 1		; visa id: 7521
  %5839 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7522
  store <2 x i32> %5838, <2 x i32>* %5839, align 4, !noalias !625		; visa id: 7522
  br label %._crit_edge323, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7524

._crit_edge323:                                   ; preds = %._crit_edge323.._crit_edge323_crit_edge, %5836
; BB531 :
  %5840 = phi i32 [ 0, %5836 ], [ %5849, %._crit_edge323.._crit_edge323_crit_edge ]
  %5841 = zext i32 %5840 to i64		; visa id: 7525
  %5842 = shl nuw nsw i64 %5841, 2		; visa id: 7526
  %5843 = add i64 %133, %5842		; visa id: 7527
  %5844 = inttoptr i64 %5843 to i32*		; visa id: 7528
  %5845 = load i32, i32* %5844, align 4, !noalias !625		; visa id: 7528
  %5846 = add i64 %128, %5842		; visa id: 7529
  %5847 = inttoptr i64 %5846 to i32*		; visa id: 7530
  store i32 %5845, i32* %5847, align 4, !alias.scope !625		; visa id: 7530
  %5848 = icmp eq i32 %5840, 0		; visa id: 7531
  br i1 %5848, label %._crit_edge323.._crit_edge323_crit_edge, label %5850, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7532

._crit_edge323.._crit_edge323_crit_edge:          ; preds = %._crit_edge323
; BB532 :
  %5849 = add nuw nsw i32 %5840, 1, !spirv.Decorations !631		; visa id: 7534
  br label %._crit_edge323, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7535

5850:                                             ; preds = %._crit_edge323
; BB533 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7537
  %5851 = load i64, i64* %129, align 8		; visa id: 7537
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7538
  %5852 = bitcast i64 %5851 to <2 x i32>		; visa id: 7538
  %5853 = extractelement <2 x i32> %5852, i32 0		; visa id: 7540
  %5854 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5853, i32 1
  %5855 = bitcast <2 x i32> %5854 to i64		; visa id: 7540
  %5856 = ashr exact i64 %5855, 32		; visa id: 7541
  %5857 = bitcast i64 %5856 to <2 x i32>		; visa id: 7542
  %5858 = extractelement <2 x i32> %5857, i32 0		; visa id: 7546
  %5859 = extractelement <2 x i32> %5857, i32 1		; visa id: 7546
  %5860 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5858, i32 %5859, i32 %41, i32 %42)
  %5861 = extractvalue { i32, i32 } %5860, 0		; visa id: 7546
  %5862 = extractvalue { i32, i32 } %5860, 1		; visa id: 7546
  %5863 = insertelement <2 x i32> undef, i32 %5861, i32 0		; visa id: 7553
  %5864 = insertelement <2 x i32> %5863, i32 %5862, i32 1		; visa id: 7554
  %5865 = bitcast <2 x i32> %5864 to i64		; visa id: 7555
  %5866 = shl i64 %5865, 1		; visa id: 7559
  %5867 = add i64 %.in401, %5866		; visa id: 7560
  %5868 = ashr i64 %5851, 31		; visa id: 7561
  %5869 = bitcast i64 %5868 to <2 x i32>		; visa id: 7562
  %5870 = extractelement <2 x i32> %5869, i32 0		; visa id: 7566
  %5871 = extractelement <2 x i32> %5869, i32 1		; visa id: 7566
  %5872 = and i32 %5870, -2		; visa id: 7566
  %5873 = insertelement <2 x i32> undef, i32 %5872, i32 0		; visa id: 7567
  %5874 = insertelement <2 x i32> %5873, i32 %5871, i32 1		; visa id: 7568
  %5875 = bitcast <2 x i32> %5874 to i64		; visa id: 7569
  %5876 = add i64 %5867, %5875		; visa id: 7573
  %5877 = inttoptr i64 %5876 to i16 addrspace(4)*		; visa id: 7574
  %5878 = addrspacecast i16 addrspace(4)* %5877 to i16 addrspace(1)*		; visa id: 7574
  %5879 = load i16, i16 addrspace(1)* %5878, align 2		; visa id: 7575
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7577
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7577
  %5880 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7577
  %5881 = insertelement <2 x i32> %5880, i32 %5636, i64 1		; visa id: 7578
  %5882 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7579
  store <2 x i32> %5881, <2 x i32>* %5882, align 4, !noalias !635		; visa id: 7579
  br label %._crit_edge324, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7581

._crit_edge324:                                   ; preds = %._crit_edge324.._crit_edge324_crit_edge, %5850
; BB534 :
  %5883 = phi i32 [ 0, %5850 ], [ %5892, %._crit_edge324.._crit_edge324_crit_edge ]
  %5884 = zext i32 %5883 to i64		; visa id: 7582
  %5885 = shl nuw nsw i64 %5884, 2		; visa id: 7583
  %5886 = add i64 %124, %5885		; visa id: 7584
  %5887 = inttoptr i64 %5886 to i32*		; visa id: 7585
  %5888 = load i32, i32* %5887, align 4, !noalias !635		; visa id: 7585
  %5889 = add i64 %119, %5885		; visa id: 7586
  %5890 = inttoptr i64 %5889 to i32*		; visa id: 7587
  store i32 %5888, i32* %5890, align 4, !alias.scope !635		; visa id: 7587
  %5891 = icmp eq i32 %5883, 0		; visa id: 7588
  br i1 %5891, label %._crit_edge324.._crit_edge324_crit_edge, label %5893, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7589

._crit_edge324.._crit_edge324_crit_edge:          ; preds = %._crit_edge324
; BB535 :
  %5892 = add nuw nsw i32 %5883, 1, !spirv.Decorations !631		; visa id: 7591
  br label %._crit_edge324, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7592

5893:                                             ; preds = %._crit_edge324
; BB536 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7594
  %5894 = load i64, i64* %120, align 8		; visa id: 7594
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7595
  %5895 = bitcast i64 %5894 to <2 x i32>		; visa id: 7595
  %5896 = extractelement <2 x i32> %5895, i32 0		; visa id: 7597
  %5897 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5896, i32 1
  %5898 = bitcast <2 x i32> %5897 to i64		; visa id: 7597
  %5899 = ashr exact i64 %5898, 32		; visa id: 7598
  %5900 = bitcast i64 %5899 to <2 x i32>		; visa id: 7599
  %5901 = extractelement <2 x i32> %5900, i32 0		; visa id: 7603
  %5902 = extractelement <2 x i32> %5900, i32 1		; visa id: 7603
  %5903 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5901, i32 %5902, i32 %44, i32 %45)
  %5904 = extractvalue { i32, i32 } %5903, 0		; visa id: 7603
  %5905 = extractvalue { i32, i32 } %5903, 1		; visa id: 7603
  %5906 = insertelement <2 x i32> undef, i32 %5904, i32 0		; visa id: 7610
  %5907 = insertelement <2 x i32> %5906, i32 %5905, i32 1		; visa id: 7611
  %5908 = bitcast <2 x i32> %5907 to i64		; visa id: 7612
  %5909 = shl i64 %5908, 1		; visa id: 7616
  %5910 = add i64 %.in400, %5909		; visa id: 7617
  %5911 = ashr i64 %5894, 31		; visa id: 7618
  %5912 = bitcast i64 %5911 to <2 x i32>		; visa id: 7619
  %5913 = extractelement <2 x i32> %5912, i32 0		; visa id: 7623
  %5914 = extractelement <2 x i32> %5912, i32 1		; visa id: 7623
  %5915 = and i32 %5913, -2		; visa id: 7623
  %5916 = insertelement <2 x i32> undef, i32 %5915, i32 0		; visa id: 7624
  %5917 = insertelement <2 x i32> %5916, i32 %5914, i32 1		; visa id: 7625
  %5918 = bitcast <2 x i32> %5917 to i64		; visa id: 7626
  %5919 = add i64 %5910, %5918		; visa id: 7630
  %5920 = inttoptr i64 %5919 to i16 addrspace(4)*		; visa id: 7631
  %5921 = addrspacecast i16 addrspace(4)* %5920 to i16 addrspace(1)*		; visa id: 7631
  %5922 = load i16, i16 addrspace(1)* %5921, align 2		; visa id: 7632
  %5923 = zext i16 %5879 to i32		; visa id: 7634
  %5924 = shl nuw i32 %5923, 16, !spirv.Decorations !639		; visa id: 7635
  %5925 = bitcast i32 %5924 to float
  %5926 = zext i16 %5922 to i32		; visa id: 7636
  %5927 = shl nuw i32 %5926, 16, !spirv.Decorations !639		; visa id: 7637
  %5928 = bitcast i32 %5927 to float
  %5929 = fmul reassoc nsz arcp contract float %5925, %5928, !spirv.Decorations !618
  %5930 = fadd reassoc nsz arcp contract float %5929, %.sroa.186.1, !spirv.Decorations !618		; visa id: 7638
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7639

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %5893
; BB537 :
  %.sroa.186.2 = phi float [ %5930, %5893 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  %5931 = icmp slt i32 %428, %const_reg_dword
  %5932 = icmp slt i32 %5636, %const_reg_dword1		; visa id: 7640
  %5933 = and i1 %5931, %5932		; visa id: 7641
  br i1 %5933, label %5934, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7643

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
; BB:
  br label %.preheader.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

5934:                                             ; preds = %._crit_edge.2.14
; BB539 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7645
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7645
  %5935 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 7645
  %5936 = insertelement <2 x i32> %5935, i32 %113, i64 1		; visa id: 7646
  %5937 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7647
  store <2 x i32> %5936, <2 x i32>* %5937, align 4, !noalias !625		; visa id: 7647
  br label %._crit_edge325, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7649

._crit_edge325:                                   ; preds = %._crit_edge325.._crit_edge325_crit_edge, %5934
; BB540 :
  %5938 = phi i32 [ 0, %5934 ], [ %5947, %._crit_edge325.._crit_edge325_crit_edge ]
  %5939 = zext i32 %5938 to i64		; visa id: 7650
  %5940 = shl nuw nsw i64 %5939, 2		; visa id: 7651
  %5941 = add i64 %133, %5940		; visa id: 7652
  %5942 = inttoptr i64 %5941 to i32*		; visa id: 7653
  %5943 = load i32, i32* %5942, align 4, !noalias !625		; visa id: 7653
  %5944 = add i64 %128, %5940		; visa id: 7654
  %5945 = inttoptr i64 %5944 to i32*		; visa id: 7655
  store i32 %5943, i32* %5945, align 4, !alias.scope !625		; visa id: 7655
  %5946 = icmp eq i32 %5938, 0		; visa id: 7656
  br i1 %5946, label %._crit_edge325.._crit_edge325_crit_edge, label %5948, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7657

._crit_edge325.._crit_edge325_crit_edge:          ; preds = %._crit_edge325
; BB541 :
  %5947 = add nuw nsw i32 %5938, 1, !spirv.Decorations !631		; visa id: 7659
  br label %._crit_edge325, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7660

5948:                                             ; preds = %._crit_edge325
; BB542 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7662
  %5949 = load i64, i64* %129, align 8		; visa id: 7662
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7663
  %5950 = bitcast i64 %5949 to <2 x i32>		; visa id: 7663
  %5951 = extractelement <2 x i32> %5950, i32 0		; visa id: 7665
  %5952 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5951, i32 1
  %5953 = bitcast <2 x i32> %5952 to i64		; visa id: 7665
  %5954 = ashr exact i64 %5953, 32		; visa id: 7666
  %5955 = bitcast i64 %5954 to <2 x i32>		; visa id: 7667
  %5956 = extractelement <2 x i32> %5955, i32 0		; visa id: 7671
  %5957 = extractelement <2 x i32> %5955, i32 1		; visa id: 7671
  %5958 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5956, i32 %5957, i32 %41, i32 %42)
  %5959 = extractvalue { i32, i32 } %5958, 0		; visa id: 7671
  %5960 = extractvalue { i32, i32 } %5958, 1		; visa id: 7671
  %5961 = insertelement <2 x i32> undef, i32 %5959, i32 0		; visa id: 7678
  %5962 = insertelement <2 x i32> %5961, i32 %5960, i32 1		; visa id: 7679
  %5963 = bitcast <2 x i32> %5962 to i64		; visa id: 7680
  %5964 = shl i64 %5963, 1		; visa id: 7684
  %5965 = add i64 %.in401, %5964		; visa id: 7685
  %5966 = ashr i64 %5949, 31		; visa id: 7686
  %5967 = bitcast i64 %5966 to <2 x i32>		; visa id: 7687
  %5968 = extractelement <2 x i32> %5967, i32 0		; visa id: 7691
  %5969 = extractelement <2 x i32> %5967, i32 1		; visa id: 7691
  %5970 = and i32 %5968, -2		; visa id: 7691
  %5971 = insertelement <2 x i32> undef, i32 %5970, i32 0		; visa id: 7692
  %5972 = insertelement <2 x i32> %5971, i32 %5969, i32 1		; visa id: 7693
  %5973 = bitcast <2 x i32> %5972 to i64		; visa id: 7694
  %5974 = add i64 %5965, %5973		; visa id: 7698
  %5975 = inttoptr i64 %5974 to i16 addrspace(4)*		; visa id: 7699
  %5976 = addrspacecast i16 addrspace(4)* %5975 to i16 addrspace(1)*		; visa id: 7699
  %5977 = load i16, i16 addrspace(1)* %5976, align 2		; visa id: 7700
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7702
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7702
  %5978 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7702
  %5979 = insertelement <2 x i32> %5978, i32 %5636, i64 1		; visa id: 7703
  %5980 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7704
  store <2 x i32> %5979, <2 x i32>* %5980, align 4, !noalias !635		; visa id: 7704
  br label %._crit_edge326, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7706

._crit_edge326:                                   ; preds = %._crit_edge326.._crit_edge326_crit_edge, %5948
; BB543 :
  %5981 = phi i32 [ 0, %5948 ], [ %5990, %._crit_edge326.._crit_edge326_crit_edge ]
  %5982 = zext i32 %5981 to i64		; visa id: 7707
  %5983 = shl nuw nsw i64 %5982, 2		; visa id: 7708
  %5984 = add i64 %124, %5983		; visa id: 7709
  %5985 = inttoptr i64 %5984 to i32*		; visa id: 7710
  %5986 = load i32, i32* %5985, align 4, !noalias !635		; visa id: 7710
  %5987 = add i64 %119, %5983		; visa id: 7711
  %5988 = inttoptr i64 %5987 to i32*		; visa id: 7712
  store i32 %5986, i32* %5988, align 4, !alias.scope !635		; visa id: 7712
  %5989 = icmp eq i32 %5981, 0		; visa id: 7713
  br i1 %5989, label %._crit_edge326.._crit_edge326_crit_edge, label %5991, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7714

._crit_edge326.._crit_edge326_crit_edge:          ; preds = %._crit_edge326
; BB544 :
  %5990 = add nuw nsw i32 %5981, 1, !spirv.Decorations !631		; visa id: 7716
  br label %._crit_edge326, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7717

5991:                                             ; preds = %._crit_edge326
; BB545 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7719
  %5992 = load i64, i64* %120, align 8		; visa id: 7719
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7720
  %5993 = bitcast i64 %5992 to <2 x i32>		; visa id: 7720
  %5994 = extractelement <2 x i32> %5993, i32 0		; visa id: 7722
  %5995 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %5994, i32 1
  %5996 = bitcast <2 x i32> %5995 to i64		; visa id: 7722
  %5997 = ashr exact i64 %5996, 32		; visa id: 7723
  %5998 = bitcast i64 %5997 to <2 x i32>		; visa id: 7724
  %5999 = extractelement <2 x i32> %5998, i32 0		; visa id: 7728
  %6000 = extractelement <2 x i32> %5998, i32 1		; visa id: 7728
  %6001 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %5999, i32 %6000, i32 %44, i32 %45)
  %6002 = extractvalue { i32, i32 } %6001, 0		; visa id: 7728
  %6003 = extractvalue { i32, i32 } %6001, 1		; visa id: 7728
  %6004 = insertelement <2 x i32> undef, i32 %6002, i32 0		; visa id: 7735
  %6005 = insertelement <2 x i32> %6004, i32 %6003, i32 1		; visa id: 7736
  %6006 = bitcast <2 x i32> %6005 to i64		; visa id: 7737
  %6007 = shl i64 %6006, 1		; visa id: 7741
  %6008 = add i64 %.in400, %6007		; visa id: 7742
  %6009 = ashr i64 %5992, 31		; visa id: 7743
  %6010 = bitcast i64 %6009 to <2 x i32>		; visa id: 7744
  %6011 = extractelement <2 x i32> %6010, i32 0		; visa id: 7748
  %6012 = extractelement <2 x i32> %6010, i32 1		; visa id: 7748
  %6013 = and i32 %6011, -2		; visa id: 7748
  %6014 = insertelement <2 x i32> undef, i32 %6013, i32 0		; visa id: 7749
  %6015 = insertelement <2 x i32> %6014, i32 %6012, i32 1		; visa id: 7750
  %6016 = bitcast <2 x i32> %6015 to i64		; visa id: 7751
  %6017 = add i64 %6008, %6016		; visa id: 7755
  %6018 = inttoptr i64 %6017 to i16 addrspace(4)*		; visa id: 7756
  %6019 = addrspacecast i16 addrspace(4)* %6018 to i16 addrspace(1)*		; visa id: 7756
  %6020 = load i16, i16 addrspace(1)* %6019, align 2		; visa id: 7757
  %6021 = zext i16 %5977 to i32		; visa id: 7759
  %6022 = shl nuw i32 %6021, 16, !spirv.Decorations !639		; visa id: 7760
  %6023 = bitcast i32 %6022 to float
  %6024 = zext i16 %6020 to i32		; visa id: 7761
  %6025 = shl nuw i32 %6024, 16, !spirv.Decorations !639		; visa id: 7762
  %6026 = bitcast i32 %6025 to float
  %6027 = fmul reassoc nsz arcp contract float %6023, %6026, !spirv.Decorations !618
  %6028 = fadd reassoc nsz arcp contract float %6027, %.sroa.250.1, !spirv.Decorations !618		; visa id: 7763
  br label %.preheader.14, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7764

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %5991
; BB546 :
  %.sroa.250.2 = phi float [ %6028, %5991 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %6029 = add i32 %69, 15		; visa id: 7765
  %6030 = icmp slt i32 %6029, %const_reg_dword1		; visa id: 7766
  %6031 = icmp slt i32 %65, %const_reg_dword
  %6032 = and i1 %6031, %6030		; visa id: 7767
  br i1 %6032, label %6033, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7769

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
; BB:
  br label %._crit_edge.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

6033:                                             ; preds = %.preheader.14
; BB548 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7771
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7771
  %6034 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 7771
  %6035 = insertelement <2 x i32> %6034, i32 %113, i64 1		; visa id: 7772
  %6036 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7773
  store <2 x i32> %6035, <2 x i32>* %6036, align 4, !noalias !625		; visa id: 7773
  br label %._crit_edge327, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7775

._crit_edge327:                                   ; preds = %._crit_edge327.._crit_edge327_crit_edge, %6033
; BB549 :
  %6037 = phi i32 [ 0, %6033 ], [ %6046, %._crit_edge327.._crit_edge327_crit_edge ]
  %6038 = zext i32 %6037 to i64		; visa id: 7776
  %6039 = shl nuw nsw i64 %6038, 2		; visa id: 7777
  %6040 = add i64 %133, %6039		; visa id: 7778
  %6041 = inttoptr i64 %6040 to i32*		; visa id: 7779
  %6042 = load i32, i32* %6041, align 4, !noalias !625		; visa id: 7779
  %6043 = add i64 %128, %6039		; visa id: 7780
  %6044 = inttoptr i64 %6043 to i32*		; visa id: 7781
  store i32 %6042, i32* %6044, align 4, !alias.scope !625		; visa id: 7781
  %6045 = icmp eq i32 %6037, 0		; visa id: 7782
  br i1 %6045, label %._crit_edge327.._crit_edge327_crit_edge, label %6047, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7783

._crit_edge327.._crit_edge327_crit_edge:          ; preds = %._crit_edge327
; BB550 :
  %6046 = add nuw nsw i32 %6037, 1, !spirv.Decorations !631		; visa id: 7785
  br label %._crit_edge327, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7786

6047:                                             ; preds = %._crit_edge327
; BB551 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7788
  %6048 = load i64, i64* %129, align 8		; visa id: 7788
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7789
  %6049 = bitcast i64 %6048 to <2 x i32>		; visa id: 7789
  %6050 = extractelement <2 x i32> %6049, i32 0		; visa id: 7791
  %6051 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6050, i32 1
  %6052 = bitcast <2 x i32> %6051 to i64		; visa id: 7791
  %6053 = ashr exact i64 %6052, 32		; visa id: 7792
  %6054 = bitcast i64 %6053 to <2 x i32>		; visa id: 7793
  %6055 = extractelement <2 x i32> %6054, i32 0		; visa id: 7797
  %6056 = extractelement <2 x i32> %6054, i32 1		; visa id: 7797
  %6057 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6055, i32 %6056, i32 %41, i32 %42)
  %6058 = extractvalue { i32, i32 } %6057, 0		; visa id: 7797
  %6059 = extractvalue { i32, i32 } %6057, 1		; visa id: 7797
  %6060 = insertelement <2 x i32> undef, i32 %6058, i32 0		; visa id: 7804
  %6061 = insertelement <2 x i32> %6060, i32 %6059, i32 1		; visa id: 7805
  %6062 = bitcast <2 x i32> %6061 to i64		; visa id: 7806
  %6063 = shl i64 %6062, 1		; visa id: 7810
  %6064 = add i64 %.in401, %6063		; visa id: 7811
  %6065 = ashr i64 %6048, 31		; visa id: 7812
  %6066 = bitcast i64 %6065 to <2 x i32>		; visa id: 7813
  %6067 = extractelement <2 x i32> %6066, i32 0		; visa id: 7817
  %6068 = extractelement <2 x i32> %6066, i32 1		; visa id: 7817
  %6069 = and i32 %6067, -2		; visa id: 7817
  %6070 = insertelement <2 x i32> undef, i32 %6069, i32 0		; visa id: 7818
  %6071 = insertelement <2 x i32> %6070, i32 %6068, i32 1		; visa id: 7819
  %6072 = bitcast <2 x i32> %6071 to i64		; visa id: 7820
  %6073 = add i64 %6064, %6072		; visa id: 7824
  %6074 = inttoptr i64 %6073 to i16 addrspace(4)*		; visa id: 7825
  %6075 = addrspacecast i16 addrspace(4)* %6074 to i16 addrspace(1)*		; visa id: 7825
  %6076 = load i16, i16 addrspace(1)* %6075, align 2		; visa id: 7826
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7828
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7828
  %6077 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7828
  %6078 = insertelement <2 x i32> %6077, i32 %6029, i64 1		; visa id: 7829
  %6079 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7830
  store <2 x i32> %6078, <2 x i32>* %6079, align 4, !noalias !635		; visa id: 7830
  br label %._crit_edge328, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7832

._crit_edge328:                                   ; preds = %._crit_edge328.._crit_edge328_crit_edge, %6047
; BB552 :
  %6080 = phi i32 [ 0, %6047 ], [ %6089, %._crit_edge328.._crit_edge328_crit_edge ]
  %6081 = zext i32 %6080 to i64		; visa id: 7833
  %6082 = shl nuw nsw i64 %6081, 2		; visa id: 7834
  %6083 = add i64 %124, %6082		; visa id: 7835
  %6084 = inttoptr i64 %6083 to i32*		; visa id: 7836
  %6085 = load i32, i32* %6084, align 4, !noalias !635		; visa id: 7836
  %6086 = add i64 %119, %6082		; visa id: 7837
  %6087 = inttoptr i64 %6086 to i32*		; visa id: 7838
  store i32 %6085, i32* %6087, align 4, !alias.scope !635		; visa id: 7838
  %6088 = icmp eq i32 %6080, 0		; visa id: 7839
  br i1 %6088, label %._crit_edge328.._crit_edge328_crit_edge, label %6090, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7840

._crit_edge328.._crit_edge328_crit_edge:          ; preds = %._crit_edge328
; BB553 :
  %6089 = add nuw nsw i32 %6080, 1, !spirv.Decorations !631		; visa id: 7842
  br label %._crit_edge328, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7843

6090:                                             ; preds = %._crit_edge328
; BB554 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7845
  %6091 = load i64, i64* %120, align 8		; visa id: 7845
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7846
  %6092 = bitcast i64 %6091 to <2 x i32>		; visa id: 7846
  %6093 = extractelement <2 x i32> %6092, i32 0		; visa id: 7848
  %6094 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6093, i32 1
  %6095 = bitcast <2 x i32> %6094 to i64		; visa id: 7848
  %6096 = ashr exact i64 %6095, 32		; visa id: 7849
  %6097 = bitcast i64 %6096 to <2 x i32>		; visa id: 7850
  %6098 = extractelement <2 x i32> %6097, i32 0		; visa id: 7854
  %6099 = extractelement <2 x i32> %6097, i32 1		; visa id: 7854
  %6100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6098, i32 %6099, i32 %44, i32 %45)
  %6101 = extractvalue { i32, i32 } %6100, 0		; visa id: 7854
  %6102 = extractvalue { i32, i32 } %6100, 1		; visa id: 7854
  %6103 = insertelement <2 x i32> undef, i32 %6101, i32 0		; visa id: 7861
  %6104 = insertelement <2 x i32> %6103, i32 %6102, i32 1		; visa id: 7862
  %6105 = bitcast <2 x i32> %6104 to i64		; visa id: 7863
  %6106 = shl i64 %6105, 1		; visa id: 7867
  %6107 = add i64 %.in400, %6106		; visa id: 7868
  %6108 = ashr i64 %6091, 31		; visa id: 7869
  %6109 = bitcast i64 %6108 to <2 x i32>		; visa id: 7870
  %6110 = extractelement <2 x i32> %6109, i32 0		; visa id: 7874
  %6111 = extractelement <2 x i32> %6109, i32 1		; visa id: 7874
  %6112 = and i32 %6110, -2		; visa id: 7874
  %6113 = insertelement <2 x i32> undef, i32 %6112, i32 0		; visa id: 7875
  %6114 = insertelement <2 x i32> %6113, i32 %6111, i32 1		; visa id: 7876
  %6115 = bitcast <2 x i32> %6114 to i64		; visa id: 7877
  %6116 = add i64 %6107, %6115		; visa id: 7881
  %6117 = inttoptr i64 %6116 to i16 addrspace(4)*		; visa id: 7882
  %6118 = addrspacecast i16 addrspace(4)* %6117 to i16 addrspace(1)*		; visa id: 7882
  %6119 = load i16, i16 addrspace(1)* %6118, align 2		; visa id: 7883
  %6120 = zext i16 %6076 to i32		; visa id: 7885
  %6121 = shl nuw i32 %6120, 16, !spirv.Decorations !639		; visa id: 7886
  %6122 = bitcast i32 %6121 to float
  %6123 = zext i16 %6119 to i32		; visa id: 7887
  %6124 = shl nuw i32 %6123, 16, !spirv.Decorations !639		; visa id: 7888
  %6125 = bitcast i32 %6124 to float
  %6126 = fmul reassoc nsz arcp contract float %6122, %6125, !spirv.Decorations !618
  %6127 = fadd reassoc nsz arcp contract float %6126, %.sroa.62.1, !spirv.Decorations !618		; visa id: 7889
  br label %._crit_edge.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7890

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %6090
; BB555 :
  %.sroa.62.2 = phi float [ %6127, %6090 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  %6128 = icmp slt i32 %230, %const_reg_dword
  %6129 = icmp slt i32 %6029, %const_reg_dword1		; visa id: 7891
  %6130 = and i1 %6128, %6129		; visa id: 7892
  br i1 %6130, label %6131, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 7894

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
; BB:
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

6131:                                             ; preds = %._crit_edge.15
; BB557 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 7896
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 7896
  %6132 = insertelement <2 x i32> undef, i32 %230, i64 0		; visa id: 7896
  %6133 = insertelement <2 x i32> %6132, i32 %113, i64 1		; visa id: 7897
  %6134 = inttoptr i64 %133 to <2 x i32>*		; visa id: 7898
  store <2 x i32> %6133, <2 x i32>* %6134, align 4, !noalias !625		; visa id: 7898
  br label %._crit_edge329, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7900

._crit_edge329:                                   ; preds = %._crit_edge329.._crit_edge329_crit_edge, %6131
; BB558 :
  %6135 = phi i32 [ 0, %6131 ], [ %6144, %._crit_edge329.._crit_edge329_crit_edge ]
  %6136 = zext i32 %6135 to i64		; visa id: 7901
  %6137 = shl nuw nsw i64 %6136, 2		; visa id: 7902
  %6138 = add i64 %133, %6137		; visa id: 7903
  %6139 = inttoptr i64 %6138 to i32*		; visa id: 7904
  %6140 = load i32, i32* %6139, align 4, !noalias !625		; visa id: 7904
  %6141 = add i64 %128, %6137		; visa id: 7905
  %6142 = inttoptr i64 %6141 to i32*		; visa id: 7906
  store i32 %6140, i32* %6142, align 4, !alias.scope !625		; visa id: 7906
  %6143 = icmp eq i32 %6135, 0		; visa id: 7907
  br i1 %6143, label %._crit_edge329.._crit_edge329_crit_edge, label %6145, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7908

._crit_edge329.._crit_edge329_crit_edge:          ; preds = %._crit_edge329
; BB559 :
  %6144 = add nuw nsw i32 %6135, 1, !spirv.Decorations !631		; visa id: 7910
  br label %._crit_edge329, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7911

6145:                                             ; preds = %._crit_edge329
; BB560 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 7913
  %6146 = load i64, i64* %129, align 8		; visa id: 7913
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 7914
  %6147 = bitcast i64 %6146 to <2 x i32>		; visa id: 7914
  %6148 = extractelement <2 x i32> %6147, i32 0		; visa id: 7916
  %6149 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6148, i32 1
  %6150 = bitcast <2 x i32> %6149 to i64		; visa id: 7916
  %6151 = ashr exact i64 %6150, 32		; visa id: 7917
  %6152 = bitcast i64 %6151 to <2 x i32>		; visa id: 7918
  %6153 = extractelement <2 x i32> %6152, i32 0		; visa id: 7922
  %6154 = extractelement <2 x i32> %6152, i32 1		; visa id: 7922
  %6155 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6153, i32 %6154, i32 %41, i32 %42)
  %6156 = extractvalue { i32, i32 } %6155, 0		; visa id: 7922
  %6157 = extractvalue { i32, i32 } %6155, 1		; visa id: 7922
  %6158 = insertelement <2 x i32> undef, i32 %6156, i32 0		; visa id: 7929
  %6159 = insertelement <2 x i32> %6158, i32 %6157, i32 1		; visa id: 7930
  %6160 = bitcast <2 x i32> %6159 to i64		; visa id: 7931
  %6161 = shl i64 %6160, 1		; visa id: 7935
  %6162 = add i64 %.in401, %6161		; visa id: 7936
  %6163 = ashr i64 %6146, 31		; visa id: 7937
  %6164 = bitcast i64 %6163 to <2 x i32>		; visa id: 7938
  %6165 = extractelement <2 x i32> %6164, i32 0		; visa id: 7942
  %6166 = extractelement <2 x i32> %6164, i32 1		; visa id: 7942
  %6167 = and i32 %6165, -2		; visa id: 7942
  %6168 = insertelement <2 x i32> undef, i32 %6167, i32 0		; visa id: 7943
  %6169 = insertelement <2 x i32> %6168, i32 %6166, i32 1		; visa id: 7944
  %6170 = bitcast <2 x i32> %6169 to i64		; visa id: 7945
  %6171 = add i64 %6162, %6170		; visa id: 7949
  %6172 = inttoptr i64 %6171 to i16 addrspace(4)*		; visa id: 7950
  %6173 = addrspacecast i16 addrspace(4)* %6172 to i16 addrspace(1)*		; visa id: 7950
  %6174 = load i16, i16 addrspace(1)* %6173, align 2		; visa id: 7951
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 7953
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 7953
  %6175 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 7953
  %6176 = insertelement <2 x i32> %6175, i32 %6029, i64 1		; visa id: 7954
  %6177 = inttoptr i64 %124 to <2 x i32>*		; visa id: 7955
  store <2 x i32> %6176, <2 x i32>* %6177, align 4, !noalias !635		; visa id: 7955
  br label %._crit_edge330, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 7957

._crit_edge330:                                   ; preds = %._crit_edge330.._crit_edge330_crit_edge, %6145
; BB561 :
  %6178 = phi i32 [ 0, %6145 ], [ %6187, %._crit_edge330.._crit_edge330_crit_edge ]
  %6179 = zext i32 %6178 to i64		; visa id: 7958
  %6180 = shl nuw nsw i64 %6179, 2		; visa id: 7959
  %6181 = add i64 %124, %6180		; visa id: 7960
  %6182 = inttoptr i64 %6181 to i32*		; visa id: 7961
  %6183 = load i32, i32* %6182, align 4, !noalias !635		; visa id: 7961
  %6184 = add i64 %119, %6180		; visa id: 7962
  %6185 = inttoptr i64 %6184 to i32*		; visa id: 7963
  store i32 %6183, i32* %6185, align 4, !alias.scope !635		; visa id: 7963
  %6186 = icmp eq i32 %6178, 0		; visa id: 7964
  br i1 %6186, label %._crit_edge330.._crit_edge330_crit_edge, label %6188, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 7965

._crit_edge330.._crit_edge330_crit_edge:          ; preds = %._crit_edge330
; BB562 :
  %6187 = add nuw nsw i32 %6178, 1, !spirv.Decorations !631		; visa id: 7967
  br label %._crit_edge330, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 7968

6188:                                             ; preds = %._crit_edge330
; BB563 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 7970
  %6189 = load i64, i64* %120, align 8		; visa id: 7970
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 7971
  %6190 = bitcast i64 %6189 to <2 x i32>		; visa id: 7971
  %6191 = extractelement <2 x i32> %6190, i32 0		; visa id: 7973
  %6192 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6191, i32 1
  %6193 = bitcast <2 x i32> %6192 to i64		; visa id: 7973
  %6194 = ashr exact i64 %6193, 32		; visa id: 7974
  %6195 = bitcast i64 %6194 to <2 x i32>		; visa id: 7975
  %6196 = extractelement <2 x i32> %6195, i32 0		; visa id: 7979
  %6197 = extractelement <2 x i32> %6195, i32 1		; visa id: 7979
  %6198 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6196, i32 %6197, i32 %44, i32 %45)
  %6199 = extractvalue { i32, i32 } %6198, 0		; visa id: 7979
  %6200 = extractvalue { i32, i32 } %6198, 1		; visa id: 7979
  %6201 = insertelement <2 x i32> undef, i32 %6199, i32 0		; visa id: 7986
  %6202 = insertelement <2 x i32> %6201, i32 %6200, i32 1		; visa id: 7987
  %6203 = bitcast <2 x i32> %6202 to i64		; visa id: 7988
  %6204 = shl i64 %6203, 1		; visa id: 7992
  %6205 = add i64 %.in400, %6204		; visa id: 7993
  %6206 = ashr i64 %6189, 31		; visa id: 7994
  %6207 = bitcast i64 %6206 to <2 x i32>		; visa id: 7995
  %6208 = extractelement <2 x i32> %6207, i32 0		; visa id: 7999
  %6209 = extractelement <2 x i32> %6207, i32 1		; visa id: 7999
  %6210 = and i32 %6208, -2		; visa id: 7999
  %6211 = insertelement <2 x i32> undef, i32 %6210, i32 0		; visa id: 8000
  %6212 = insertelement <2 x i32> %6211, i32 %6209, i32 1		; visa id: 8001
  %6213 = bitcast <2 x i32> %6212 to i64		; visa id: 8002
  %6214 = add i64 %6205, %6213		; visa id: 8006
  %6215 = inttoptr i64 %6214 to i16 addrspace(4)*		; visa id: 8007
  %6216 = addrspacecast i16 addrspace(4)* %6215 to i16 addrspace(1)*		; visa id: 8007
  %6217 = load i16, i16 addrspace(1)* %6216, align 2		; visa id: 8008
  %6218 = zext i16 %6174 to i32		; visa id: 8010
  %6219 = shl nuw i32 %6218, 16, !spirv.Decorations !639		; visa id: 8011
  %6220 = bitcast i32 %6219 to float
  %6221 = zext i16 %6217 to i32		; visa id: 8012
  %6222 = shl nuw i32 %6221, 16, !spirv.Decorations !639		; visa id: 8013
  %6223 = bitcast i32 %6222 to float
  %6224 = fmul reassoc nsz arcp contract float %6220, %6223, !spirv.Decorations !618
  %6225 = fadd reassoc nsz arcp contract float %6224, %.sroa.126.1, !spirv.Decorations !618		; visa id: 8014
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8015

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %6188
; BB564 :
  %.sroa.126.2 = phi float [ %6225, %6188 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  %6226 = icmp slt i32 %329, %const_reg_dword
  %6227 = icmp slt i32 %6029, %const_reg_dword1		; visa id: 8016
  %6228 = and i1 %6226, %6227		; visa id: 8017
  br i1 %6228, label %6229, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 8019

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
; BB:
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

6229:                                             ; preds = %._crit_edge.1.15
; BB566 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 8021
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 8021
  %6230 = insertelement <2 x i32> undef, i32 %329, i64 0		; visa id: 8021
  %6231 = insertelement <2 x i32> %6230, i32 %113, i64 1		; visa id: 8022
  %6232 = inttoptr i64 %133 to <2 x i32>*		; visa id: 8023
  store <2 x i32> %6231, <2 x i32>* %6232, align 4, !noalias !625		; visa id: 8023
  br label %._crit_edge331, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8025

._crit_edge331:                                   ; preds = %._crit_edge331.._crit_edge331_crit_edge, %6229
; BB567 :
  %6233 = phi i32 [ 0, %6229 ], [ %6242, %._crit_edge331.._crit_edge331_crit_edge ]
  %6234 = zext i32 %6233 to i64		; visa id: 8026
  %6235 = shl nuw nsw i64 %6234, 2		; visa id: 8027
  %6236 = add i64 %133, %6235		; visa id: 8028
  %6237 = inttoptr i64 %6236 to i32*		; visa id: 8029
  %6238 = load i32, i32* %6237, align 4, !noalias !625		; visa id: 8029
  %6239 = add i64 %128, %6235		; visa id: 8030
  %6240 = inttoptr i64 %6239 to i32*		; visa id: 8031
  store i32 %6238, i32* %6240, align 4, !alias.scope !625		; visa id: 8031
  %6241 = icmp eq i32 %6233, 0		; visa id: 8032
  br i1 %6241, label %._crit_edge331.._crit_edge331_crit_edge, label %6243, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 8033

._crit_edge331.._crit_edge331_crit_edge:          ; preds = %._crit_edge331
; BB568 :
  %6242 = add nuw nsw i32 %6233, 1, !spirv.Decorations !631		; visa id: 8035
  br label %._crit_edge331, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 8036

6243:                                             ; preds = %._crit_edge331
; BB569 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 8038
  %6244 = load i64, i64* %129, align 8		; visa id: 8038
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 8039
  %6245 = bitcast i64 %6244 to <2 x i32>		; visa id: 8039
  %6246 = extractelement <2 x i32> %6245, i32 0		; visa id: 8041
  %6247 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6246, i32 1
  %6248 = bitcast <2 x i32> %6247 to i64		; visa id: 8041
  %6249 = ashr exact i64 %6248, 32		; visa id: 8042
  %6250 = bitcast i64 %6249 to <2 x i32>		; visa id: 8043
  %6251 = extractelement <2 x i32> %6250, i32 0		; visa id: 8047
  %6252 = extractelement <2 x i32> %6250, i32 1		; visa id: 8047
  %6253 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6251, i32 %6252, i32 %41, i32 %42)
  %6254 = extractvalue { i32, i32 } %6253, 0		; visa id: 8047
  %6255 = extractvalue { i32, i32 } %6253, 1		; visa id: 8047
  %6256 = insertelement <2 x i32> undef, i32 %6254, i32 0		; visa id: 8054
  %6257 = insertelement <2 x i32> %6256, i32 %6255, i32 1		; visa id: 8055
  %6258 = bitcast <2 x i32> %6257 to i64		; visa id: 8056
  %6259 = shl i64 %6258, 1		; visa id: 8060
  %6260 = add i64 %.in401, %6259		; visa id: 8061
  %6261 = ashr i64 %6244, 31		; visa id: 8062
  %6262 = bitcast i64 %6261 to <2 x i32>		; visa id: 8063
  %6263 = extractelement <2 x i32> %6262, i32 0		; visa id: 8067
  %6264 = extractelement <2 x i32> %6262, i32 1		; visa id: 8067
  %6265 = and i32 %6263, -2		; visa id: 8067
  %6266 = insertelement <2 x i32> undef, i32 %6265, i32 0		; visa id: 8068
  %6267 = insertelement <2 x i32> %6266, i32 %6264, i32 1		; visa id: 8069
  %6268 = bitcast <2 x i32> %6267 to i64		; visa id: 8070
  %6269 = add i64 %6260, %6268		; visa id: 8074
  %6270 = inttoptr i64 %6269 to i16 addrspace(4)*		; visa id: 8075
  %6271 = addrspacecast i16 addrspace(4)* %6270 to i16 addrspace(1)*		; visa id: 8075
  %6272 = load i16, i16 addrspace(1)* %6271, align 2		; visa id: 8076
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 8078
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 8078
  %6273 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 8078
  %6274 = insertelement <2 x i32> %6273, i32 %6029, i64 1		; visa id: 8079
  %6275 = inttoptr i64 %124 to <2 x i32>*		; visa id: 8080
  store <2 x i32> %6274, <2 x i32>* %6275, align 4, !noalias !635		; visa id: 8080
  br label %._crit_edge332, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8082

._crit_edge332:                                   ; preds = %._crit_edge332.._crit_edge332_crit_edge, %6243
; BB570 :
  %6276 = phi i32 [ 0, %6243 ], [ %6285, %._crit_edge332.._crit_edge332_crit_edge ]
  %6277 = zext i32 %6276 to i64		; visa id: 8083
  %6278 = shl nuw nsw i64 %6277, 2		; visa id: 8084
  %6279 = add i64 %124, %6278		; visa id: 8085
  %6280 = inttoptr i64 %6279 to i32*		; visa id: 8086
  %6281 = load i32, i32* %6280, align 4, !noalias !635		; visa id: 8086
  %6282 = add i64 %119, %6278		; visa id: 8087
  %6283 = inttoptr i64 %6282 to i32*		; visa id: 8088
  store i32 %6281, i32* %6283, align 4, !alias.scope !635		; visa id: 8088
  %6284 = icmp eq i32 %6276, 0		; visa id: 8089
  br i1 %6284, label %._crit_edge332.._crit_edge332_crit_edge, label %6286, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 8090

._crit_edge332.._crit_edge332_crit_edge:          ; preds = %._crit_edge332
; BB571 :
  %6285 = add nuw nsw i32 %6276, 1, !spirv.Decorations !631		; visa id: 8092
  br label %._crit_edge332, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 8093

6286:                                             ; preds = %._crit_edge332
; BB572 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 8095
  %6287 = load i64, i64* %120, align 8		; visa id: 8095
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 8096
  %6288 = bitcast i64 %6287 to <2 x i32>		; visa id: 8096
  %6289 = extractelement <2 x i32> %6288, i32 0		; visa id: 8098
  %6290 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6289, i32 1
  %6291 = bitcast <2 x i32> %6290 to i64		; visa id: 8098
  %6292 = ashr exact i64 %6291, 32		; visa id: 8099
  %6293 = bitcast i64 %6292 to <2 x i32>		; visa id: 8100
  %6294 = extractelement <2 x i32> %6293, i32 0		; visa id: 8104
  %6295 = extractelement <2 x i32> %6293, i32 1		; visa id: 8104
  %6296 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6294, i32 %6295, i32 %44, i32 %45)
  %6297 = extractvalue { i32, i32 } %6296, 0		; visa id: 8104
  %6298 = extractvalue { i32, i32 } %6296, 1		; visa id: 8104
  %6299 = insertelement <2 x i32> undef, i32 %6297, i32 0		; visa id: 8111
  %6300 = insertelement <2 x i32> %6299, i32 %6298, i32 1		; visa id: 8112
  %6301 = bitcast <2 x i32> %6300 to i64		; visa id: 8113
  %6302 = shl i64 %6301, 1		; visa id: 8117
  %6303 = add i64 %.in400, %6302		; visa id: 8118
  %6304 = ashr i64 %6287, 31		; visa id: 8119
  %6305 = bitcast i64 %6304 to <2 x i32>		; visa id: 8120
  %6306 = extractelement <2 x i32> %6305, i32 0		; visa id: 8124
  %6307 = extractelement <2 x i32> %6305, i32 1		; visa id: 8124
  %6308 = and i32 %6306, -2		; visa id: 8124
  %6309 = insertelement <2 x i32> undef, i32 %6308, i32 0		; visa id: 8125
  %6310 = insertelement <2 x i32> %6309, i32 %6307, i32 1		; visa id: 8126
  %6311 = bitcast <2 x i32> %6310 to i64		; visa id: 8127
  %6312 = add i64 %6303, %6311		; visa id: 8131
  %6313 = inttoptr i64 %6312 to i16 addrspace(4)*		; visa id: 8132
  %6314 = addrspacecast i16 addrspace(4)* %6313 to i16 addrspace(1)*		; visa id: 8132
  %6315 = load i16, i16 addrspace(1)* %6314, align 2		; visa id: 8133
  %6316 = zext i16 %6272 to i32		; visa id: 8135
  %6317 = shl nuw i32 %6316, 16, !spirv.Decorations !639		; visa id: 8136
  %6318 = bitcast i32 %6317 to float
  %6319 = zext i16 %6315 to i32		; visa id: 8137
  %6320 = shl nuw i32 %6319, 16, !spirv.Decorations !639		; visa id: 8138
  %6321 = bitcast i32 %6320 to float
  %6322 = fmul reassoc nsz arcp contract float %6318, %6321, !spirv.Decorations !618
  %6323 = fadd reassoc nsz arcp contract float %6322, %.sroa.190.1, !spirv.Decorations !618		; visa id: 8139
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8140

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %6286
; BB573 :
  %.sroa.190.2 = phi float [ %6323, %6286 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  %6324 = icmp slt i32 %428, %const_reg_dword
  %6325 = icmp slt i32 %6029, %const_reg_dword1		; visa id: 8141
  %6326 = and i1 %6324, %6325		; visa id: 8142
  br i1 %6326, label %6327, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 8144

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
; BB:
  br label %.preheader.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615

6327:                                             ; preds = %._crit_edge.2.15
; BB575 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %130)		; visa id: 8146
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %134)		; visa id: 8146
  %6328 = insertelement <2 x i32> undef, i32 %428, i64 0		; visa id: 8146
  %6329 = insertelement <2 x i32> %6328, i32 %113, i64 1		; visa id: 8147
  %6330 = inttoptr i64 %133 to <2 x i32>*		; visa id: 8148
  store <2 x i32> %6329, <2 x i32>* %6330, align 4, !noalias !625		; visa id: 8148
  br label %._crit_edge333, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8150

._crit_edge333:                                   ; preds = %._crit_edge333.._crit_edge333_crit_edge, %6327
; BB576 :
  %6331 = phi i32 [ 0, %6327 ], [ %6340, %._crit_edge333.._crit_edge333_crit_edge ]
  %6332 = zext i32 %6331 to i64		; visa id: 8151
  %6333 = shl nuw nsw i64 %6332, 2		; visa id: 8152
  %6334 = add i64 %133, %6333		; visa id: 8153
  %6335 = inttoptr i64 %6334 to i32*		; visa id: 8154
  %6336 = load i32, i32* %6335, align 4, !noalias !625		; visa id: 8154
  %6337 = add i64 %128, %6333		; visa id: 8155
  %6338 = inttoptr i64 %6337 to i32*		; visa id: 8156
  store i32 %6336, i32* %6338, align 4, !alias.scope !625		; visa id: 8156
  %6339 = icmp eq i32 %6331, 0		; visa id: 8157
  br i1 %6339, label %._crit_edge333.._crit_edge333_crit_edge, label %6341, !llvm.loop !628, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 8158

._crit_edge333.._crit_edge333_crit_edge:          ; preds = %._crit_edge333
; BB577 :
  %6340 = add nuw nsw i32 %6331, 1, !spirv.Decorations !631		; visa id: 8160
  br label %._crit_edge333, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 8161

6341:                                             ; preds = %._crit_edge333
; BB578 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %134)		; visa id: 8163
  %6342 = load i64, i64* %129, align 8		; visa id: 8163
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %130)		; visa id: 8164
  %6343 = bitcast i64 %6342 to <2 x i32>		; visa id: 8164
  %6344 = extractelement <2 x i32> %6343, i32 0		; visa id: 8166
  %6345 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6344, i32 1
  %6346 = bitcast <2 x i32> %6345 to i64		; visa id: 8166
  %6347 = ashr exact i64 %6346, 32		; visa id: 8167
  %6348 = bitcast i64 %6347 to <2 x i32>		; visa id: 8168
  %6349 = extractelement <2 x i32> %6348, i32 0		; visa id: 8172
  %6350 = extractelement <2 x i32> %6348, i32 1		; visa id: 8172
  %6351 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6349, i32 %6350, i32 %41, i32 %42)
  %6352 = extractvalue { i32, i32 } %6351, 0		; visa id: 8172
  %6353 = extractvalue { i32, i32 } %6351, 1		; visa id: 8172
  %6354 = insertelement <2 x i32> undef, i32 %6352, i32 0		; visa id: 8179
  %6355 = insertelement <2 x i32> %6354, i32 %6353, i32 1		; visa id: 8180
  %6356 = bitcast <2 x i32> %6355 to i64		; visa id: 8181
  %6357 = shl i64 %6356, 1		; visa id: 8185
  %6358 = add i64 %.in401, %6357		; visa id: 8186
  %6359 = ashr i64 %6342, 31		; visa id: 8187
  %6360 = bitcast i64 %6359 to <2 x i32>		; visa id: 8188
  %6361 = extractelement <2 x i32> %6360, i32 0		; visa id: 8192
  %6362 = extractelement <2 x i32> %6360, i32 1		; visa id: 8192
  %6363 = and i32 %6361, -2		; visa id: 8192
  %6364 = insertelement <2 x i32> undef, i32 %6363, i32 0		; visa id: 8193
  %6365 = insertelement <2 x i32> %6364, i32 %6362, i32 1		; visa id: 8194
  %6366 = bitcast <2 x i32> %6365 to i64		; visa id: 8195
  %6367 = add i64 %6358, %6366		; visa id: 8199
  %6368 = inttoptr i64 %6367 to i16 addrspace(4)*		; visa id: 8200
  %6369 = addrspacecast i16 addrspace(4)* %6368 to i16 addrspace(1)*		; visa id: 8200
  %6370 = load i16, i16 addrspace(1)* %6369, align 2		; visa id: 8201
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %121)		; visa id: 8203
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %125)		; visa id: 8203
  %6371 = insertelement <2 x i32> undef, i32 %113, i64 0		; visa id: 8203
  %6372 = insertelement <2 x i32> %6371, i32 %6029, i64 1		; visa id: 8204
  %6373 = inttoptr i64 %124 to <2 x i32>*		; visa id: 8205
  store <2 x i32> %6372, <2 x i32>* %6373, align 4, !noalias !635		; visa id: 8205
  br label %._crit_edge334, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8207

._crit_edge334:                                   ; preds = %._crit_edge334.._crit_edge334_crit_edge, %6341
; BB579 :
  %6374 = phi i32 [ 0, %6341 ], [ %6383, %._crit_edge334.._crit_edge334_crit_edge ]
  %6375 = zext i32 %6374 to i64		; visa id: 8208
  %6376 = shl nuw nsw i64 %6375, 2		; visa id: 8209
  %6377 = add i64 %124, %6376		; visa id: 8210
  %6378 = inttoptr i64 %6377 to i32*		; visa id: 8211
  %6379 = load i32, i32* %6378, align 4, !noalias !635		; visa id: 8211
  %6380 = add i64 %119, %6376		; visa id: 8212
  %6381 = inttoptr i64 %6380 to i32*		; visa id: 8213
  store i32 %6379, i32* %6381, align 4, !alias.scope !635		; visa id: 8213
  %6382 = icmp eq i32 %6374, 0		; visa id: 8214
  br i1 %6382, label %._crit_edge334.._crit_edge334_crit_edge, label %6384, !llvm.loop !638, !stats.blockFrequency.digits !630, !stats.blockFrequency.scale !615		; visa id: 8215

._crit_edge334.._crit_edge334_crit_edge:          ; preds = %._crit_edge334
; BB580 :
  %6383 = add nuw nsw i32 %6374, 1, !spirv.Decorations !631		; visa id: 8217
  br label %._crit_edge334, !stats.blockFrequency.digits !634, !stats.blockFrequency.scale !615		; visa id: 8218

6384:                                             ; preds = %._crit_edge334
; BB581 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %125)		; visa id: 8220
  %6385 = load i64, i64* %120, align 8		; visa id: 8220
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %121)		; visa id: 8221
  %6386 = bitcast i64 %6385 to <2 x i32>		; visa id: 8221
  %6387 = extractelement <2 x i32> %6386, i32 0		; visa id: 8223
  %6388 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6387, i32 1
  %6389 = bitcast <2 x i32> %6388 to i64		; visa id: 8223
  %6390 = ashr exact i64 %6389, 32		; visa id: 8224
  %6391 = bitcast i64 %6390 to <2 x i32>		; visa id: 8225
  %6392 = extractelement <2 x i32> %6391, i32 0		; visa id: 8229
  %6393 = extractelement <2 x i32> %6391, i32 1		; visa id: 8229
  %6394 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6392, i32 %6393, i32 %44, i32 %45)
  %6395 = extractvalue { i32, i32 } %6394, 0		; visa id: 8229
  %6396 = extractvalue { i32, i32 } %6394, 1		; visa id: 8229
  %6397 = insertelement <2 x i32> undef, i32 %6395, i32 0		; visa id: 8236
  %6398 = insertelement <2 x i32> %6397, i32 %6396, i32 1		; visa id: 8237
  %6399 = bitcast <2 x i32> %6398 to i64		; visa id: 8238
  %6400 = shl i64 %6399, 1		; visa id: 8242
  %6401 = add i64 %.in400, %6400		; visa id: 8243
  %6402 = ashr i64 %6385, 31		; visa id: 8244
  %6403 = bitcast i64 %6402 to <2 x i32>		; visa id: 8245
  %6404 = extractelement <2 x i32> %6403, i32 0		; visa id: 8249
  %6405 = extractelement <2 x i32> %6403, i32 1		; visa id: 8249
  %6406 = and i32 %6404, -2		; visa id: 8249
  %6407 = insertelement <2 x i32> undef, i32 %6406, i32 0		; visa id: 8250
  %6408 = insertelement <2 x i32> %6407, i32 %6405, i32 1		; visa id: 8251
  %6409 = bitcast <2 x i32> %6408 to i64		; visa id: 8252
  %6410 = add i64 %6401, %6409		; visa id: 8256
  %6411 = inttoptr i64 %6410 to i16 addrspace(4)*		; visa id: 8257
  %6412 = addrspacecast i16 addrspace(4)* %6411 to i16 addrspace(1)*		; visa id: 8257
  %6413 = load i16, i16 addrspace(1)* %6412, align 2		; visa id: 8258
  %6414 = zext i16 %6370 to i32		; visa id: 8260
  %6415 = shl nuw i32 %6414, 16, !spirv.Decorations !639		; visa id: 8261
  %6416 = bitcast i32 %6415 to float
  %6417 = zext i16 %6413 to i32		; visa id: 8262
  %6418 = shl nuw i32 %6417, 16, !spirv.Decorations !639		; visa id: 8263
  %6419 = bitcast i32 %6418 to float
  %6420 = fmul reassoc nsz arcp contract float %6416, %6419, !spirv.Decorations !618
  %6421 = fadd reassoc nsz arcp contract float %6420, %.sroa.254.1, !spirv.Decorations !618		; visa id: 8264
  br label %.preheader.15, !stats.blockFrequency.digits !624, !stats.blockFrequency.scale !615		; visa id: 8265

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %6384
; BB582 :
  %.sroa.254.2 = phi float [ %6421, %6384 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %6422 = add nuw nsw i32 %113, 1, !spirv.Decorations !631		; visa id: 8266
  %6423 = icmp slt i32 %6422, %const_reg_dword2		; visa id: 8267
  br i1 %6423, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !llvm.loop !640, !stats.blockFrequency.digits !623, !stats.blockFrequency.scale !615		; visa id: 8268

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
  %6424 = add nuw nsw i32 %56, %109		; visa id: 8270
  %6425 = zext i32 %6424 to i64		; visa id: 8271
  %6426 = add i64 %111, %6425		; visa id: 8272
  %6427 = inttoptr i64 %6426 to i8*		; visa id: 8273
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8273
  %6428 = add nuw nsw i32 %56, %102		; visa id: 8273
  %6429 = zext i32 %6428 to i64		; visa id: 8274
  %6430 = add i64 %111, %6429		; visa id: 8275
  %6431 = inttoptr i64 %6430 to i8*		; visa id: 8276
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8276
  %6432 = insertelement <2 x i32> undef, i32 %65, i64 0		; visa id: 8276
  %6433 = insertelement <2 x i32> %6432, i32 %69, i64 1		; visa id: 8277
  %6434 = inttoptr i64 %6430 to <2 x i32>*		; visa id: 8280
  store <2 x i32> %6433, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8280
  br label %._crit_edge335, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8282

._crit_edge335:                                   ; preds = %._crit_edge335.._crit_edge335_crit_edge, %.preheader1.preheader
; BB586 :
  %6435 = phi i32 [ 0, %.preheader1.preheader ], [ %6444, %._crit_edge335.._crit_edge335_crit_edge ]
  %6436 = zext i32 %6435 to i64		; visa id: 8283
  %6437 = shl nuw nsw i64 %6436, 2		; visa id: 8284
  %6438 = add i64 %6430, %6437		; visa id: 8285
  %6439 = inttoptr i64 %6438 to i32*		; visa id: 8286
  %6440 = load i32, i32* %6439, align 4, !noalias !642		; visa id: 8286
  %6441 = add i64 %6426, %6437		; visa id: 8287
  %6442 = inttoptr i64 %6441 to i32*		; visa id: 8288
  store i32 %6440, i32* %6442, align 4, !alias.scope !642		; visa id: 8288
  %6443 = icmp eq i32 %6435, 0		; visa id: 8289
  br i1 %6443, label %._crit_edge335.._crit_edge335_crit_edge, label %6445, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8290

._crit_edge335.._crit_edge335_crit_edge:          ; preds = %._crit_edge335
; BB587 :
  %6444 = add nuw nsw i32 %6435, 1, !spirv.Decorations !631		; visa id: 8292
  br label %._crit_edge335, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8293

6445:                                             ; preds = %._crit_edge335
; BB588 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8295
  %6446 = inttoptr i64 %6426 to i64*		; visa id: 8295
  %6447 = load i64, i64* %6446, align 8		; visa id: 8295
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8296
  %6448 = icmp slt i32 %65, %const_reg_dword
  %6449 = icmp slt i32 %69, %const_reg_dword1		; visa id: 8296
  %6450 = and i1 %6448, %6449		; visa id: 8297
  br i1 %6450, label %6451, label %.._crit_edge70_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8299

.._crit_edge70_crit_edge:                         ; preds = %6445
; BB:
  br label %._crit_edge70, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6451:                                             ; preds = %6445
; BB590 :
  %6452 = bitcast i64 %6447 to <2 x i32>		; visa id: 8301
  %6453 = extractelement <2 x i32> %6452, i32 0		; visa id: 8303
  %6454 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6453, i32 1
  %6455 = bitcast <2 x i32> %6454 to i64		; visa id: 8303
  %6456 = ashr exact i64 %6455, 32		; visa id: 8304
  %6457 = bitcast i64 %6456 to <2 x i32>		; visa id: 8305
  %6458 = extractelement <2 x i32> %6457, i32 0		; visa id: 8309
  %6459 = extractelement <2 x i32> %6457, i32 1		; visa id: 8309
  %6460 = ashr i64 %6447, 32		; visa id: 8309
  %6461 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6458, i32 %6459, i32 %50, i32 %51)
  %6462 = extractvalue { i32, i32 } %6461, 0		; visa id: 8310
  %6463 = extractvalue { i32, i32 } %6461, 1		; visa id: 8310
  %6464 = insertelement <2 x i32> undef, i32 %6462, i32 0		; visa id: 8317
  %6465 = insertelement <2 x i32> %6464, i32 %6463, i32 1		; visa id: 8318
  %6466 = bitcast <2 x i32> %6465 to i64		; visa id: 8319
  %6467 = add nsw i64 %6466, %6460, !spirv.Decorations !649		; visa id: 8323
  %6468 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !618		; visa id: 8324
  br i1 %86, label %6474, label %6469, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8325

6469:                                             ; preds = %6451
; BB591 :
  %6470 = shl i64 %6467, 2		; visa id: 8327
  %6471 = add i64 %.in, %6470		; visa id: 8328
  %6472 = inttoptr i64 %6471 to float addrspace(4)*		; visa id: 8329
  %6473 = addrspacecast float addrspace(4)* %6472 to float addrspace(1)*		; visa id: 8329
  store float %6468, float addrspace(1)* %6473, align 4		; visa id: 8330
  br label %._crit_edge70, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8331

6474:                                             ; preds = %6451
; BB592 :
  %6475 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6458, i32 %6459, i32 %47, i32 %48)
  %6476 = extractvalue { i32, i32 } %6475, 0		; visa id: 8333
  %6477 = extractvalue { i32, i32 } %6475, 1		; visa id: 8333
  %6478 = insertelement <2 x i32> undef, i32 %6476, i32 0		; visa id: 8340
  %6479 = insertelement <2 x i32> %6478, i32 %6477, i32 1		; visa id: 8341
  %6480 = bitcast <2 x i32> %6479 to i64		; visa id: 8342
  %6481 = shl i64 %6480, 2		; visa id: 8346
  %6482 = add i64 %.in399, %6481		; visa id: 8347
  %6483 = shl nsw i64 %6460, 2		; visa id: 8348
  %6484 = add i64 %6482, %6483		; visa id: 8349
  %6485 = inttoptr i64 %6484 to float addrspace(4)*		; visa id: 8350
  %6486 = addrspacecast float addrspace(4)* %6485 to float addrspace(1)*		; visa id: 8350
  %6487 = load float, float addrspace(1)* %6486, align 4		; visa id: 8351
  %6488 = fmul reassoc nsz arcp contract float %6487, %4, !spirv.Decorations !618		; visa id: 8352
  %6489 = fadd reassoc nsz arcp contract float %6468, %6488, !spirv.Decorations !618		; visa id: 8353
  %6490 = shl i64 %6467, 2		; visa id: 8354
  %6491 = add i64 %.in, %6490		; visa id: 8355
  %6492 = inttoptr i64 %6491 to float addrspace(4)*		; visa id: 8356
  %6493 = addrspacecast float addrspace(4)* %6492 to float addrspace(1)*		; visa id: 8356
  store float %6489, float addrspace(1)* %6493, align 4		; visa id: 8357
  br label %._crit_edge70, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8358

._crit_edge70:                                    ; preds = %.._crit_edge70_crit_edge, %6469, %6474
; BB593 :
  %6494 = add i32 %65, 1		; visa id: 8359
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8360
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8360
  %6495 = insertelement <2 x i32> undef, i32 %6494, i64 0		; visa id: 8360
  %6496 = insertelement <2 x i32> %6495, i32 %69, i64 1		; visa id: 8361
  store <2 x i32> %6496, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8364
  br label %._crit_edge336, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8366

._crit_edge336:                                   ; preds = %._crit_edge336.._crit_edge336_crit_edge, %._crit_edge70
; BB594 :
  %6497 = phi i32 [ 0, %._crit_edge70 ], [ %6506, %._crit_edge336.._crit_edge336_crit_edge ]
  %6498 = zext i32 %6497 to i64		; visa id: 8367
  %6499 = shl nuw nsw i64 %6498, 2		; visa id: 8368
  %6500 = add i64 %6430, %6499		; visa id: 8369
  %6501 = inttoptr i64 %6500 to i32*		; visa id: 8370
  %6502 = load i32, i32* %6501, align 4, !noalias !642		; visa id: 8370
  %6503 = add i64 %6426, %6499		; visa id: 8371
  %6504 = inttoptr i64 %6503 to i32*		; visa id: 8372
  store i32 %6502, i32* %6504, align 4, !alias.scope !642		; visa id: 8372
  %6505 = icmp eq i32 %6497, 0		; visa id: 8373
  br i1 %6505, label %._crit_edge336.._crit_edge336_crit_edge, label %6507, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8374

._crit_edge336.._crit_edge336_crit_edge:          ; preds = %._crit_edge336
; BB595 :
  %6506 = add nuw nsw i32 %6497, 1, !spirv.Decorations !631		; visa id: 8376
  br label %._crit_edge336, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8377

6507:                                             ; preds = %._crit_edge336
; BB596 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8379
  %6508 = load i64, i64* %6446, align 8		; visa id: 8379
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8380
  %6509 = icmp slt i32 %6494, %const_reg_dword
  %6510 = icmp slt i32 %69, %const_reg_dword1		; visa id: 8380
  %6511 = and i1 %6509, %6510		; visa id: 8381
  br i1 %6511, label %6512, label %.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8383

.._crit_edge70.1_crit_edge:                       ; preds = %6507
; BB:
  br label %._crit_edge70.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6512:                                             ; preds = %6507
; BB598 :
  %6513 = bitcast i64 %6508 to <2 x i32>		; visa id: 8385
  %6514 = extractelement <2 x i32> %6513, i32 0		; visa id: 8387
  %6515 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6514, i32 1
  %6516 = bitcast <2 x i32> %6515 to i64		; visa id: 8387
  %6517 = ashr exact i64 %6516, 32		; visa id: 8388
  %6518 = bitcast i64 %6517 to <2 x i32>		; visa id: 8389
  %6519 = extractelement <2 x i32> %6518, i32 0		; visa id: 8393
  %6520 = extractelement <2 x i32> %6518, i32 1		; visa id: 8393
  %6521 = ashr i64 %6508, 32		; visa id: 8393
  %6522 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6519, i32 %6520, i32 %50, i32 %51)
  %6523 = extractvalue { i32, i32 } %6522, 0		; visa id: 8394
  %6524 = extractvalue { i32, i32 } %6522, 1		; visa id: 8394
  %6525 = insertelement <2 x i32> undef, i32 %6523, i32 0		; visa id: 8401
  %6526 = insertelement <2 x i32> %6525, i32 %6524, i32 1		; visa id: 8402
  %6527 = bitcast <2 x i32> %6526 to i64		; visa id: 8403
  %6528 = add nsw i64 %6527, %6521, !spirv.Decorations !649		; visa id: 8407
  %6529 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !618		; visa id: 8408
  br i1 %86, label %6535, label %6530, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8409

6530:                                             ; preds = %6512
; BB599 :
  %6531 = shl i64 %6528, 2		; visa id: 8411
  %6532 = add i64 %.in, %6531		; visa id: 8412
  %6533 = inttoptr i64 %6532 to float addrspace(4)*		; visa id: 8413
  %6534 = addrspacecast float addrspace(4)* %6533 to float addrspace(1)*		; visa id: 8413
  store float %6529, float addrspace(1)* %6534, align 4		; visa id: 8414
  br label %._crit_edge70.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8415

6535:                                             ; preds = %6512
; BB600 :
  %6536 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6519, i32 %6520, i32 %47, i32 %48)
  %6537 = extractvalue { i32, i32 } %6536, 0		; visa id: 8417
  %6538 = extractvalue { i32, i32 } %6536, 1		; visa id: 8417
  %6539 = insertelement <2 x i32> undef, i32 %6537, i32 0		; visa id: 8424
  %6540 = insertelement <2 x i32> %6539, i32 %6538, i32 1		; visa id: 8425
  %6541 = bitcast <2 x i32> %6540 to i64		; visa id: 8426
  %6542 = shl i64 %6541, 2		; visa id: 8430
  %6543 = add i64 %.in399, %6542		; visa id: 8431
  %6544 = shl nsw i64 %6521, 2		; visa id: 8432
  %6545 = add i64 %6543, %6544		; visa id: 8433
  %6546 = inttoptr i64 %6545 to float addrspace(4)*		; visa id: 8434
  %6547 = addrspacecast float addrspace(4)* %6546 to float addrspace(1)*		; visa id: 8434
  %6548 = load float, float addrspace(1)* %6547, align 4		; visa id: 8435
  %6549 = fmul reassoc nsz arcp contract float %6548, %4, !spirv.Decorations !618		; visa id: 8436
  %6550 = fadd reassoc nsz arcp contract float %6529, %6549, !spirv.Decorations !618		; visa id: 8437
  %6551 = shl i64 %6528, 2		; visa id: 8438
  %6552 = add i64 %.in, %6551		; visa id: 8439
  %6553 = inttoptr i64 %6552 to float addrspace(4)*		; visa id: 8440
  %6554 = addrspacecast float addrspace(4)* %6553 to float addrspace(1)*		; visa id: 8440
  store float %6550, float addrspace(1)* %6554, align 4		; visa id: 8441
  br label %._crit_edge70.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8442

._crit_edge70.1:                                  ; preds = %.._crit_edge70.1_crit_edge, %6535, %6530
; BB601 :
  %6555 = add i32 %65, 2		; visa id: 8443
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8444
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8444
  %6556 = insertelement <2 x i32> undef, i32 %6555, i64 0		; visa id: 8444
  %6557 = insertelement <2 x i32> %6556, i32 %69, i64 1		; visa id: 8445
  store <2 x i32> %6557, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8448
  br label %._crit_edge337, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8450

._crit_edge337:                                   ; preds = %._crit_edge337.._crit_edge337_crit_edge, %._crit_edge70.1
; BB602 :
  %6558 = phi i32 [ 0, %._crit_edge70.1 ], [ %6567, %._crit_edge337.._crit_edge337_crit_edge ]
  %6559 = zext i32 %6558 to i64		; visa id: 8451
  %6560 = shl nuw nsw i64 %6559, 2		; visa id: 8452
  %6561 = add i64 %6430, %6560		; visa id: 8453
  %6562 = inttoptr i64 %6561 to i32*		; visa id: 8454
  %6563 = load i32, i32* %6562, align 4, !noalias !642		; visa id: 8454
  %6564 = add i64 %6426, %6560		; visa id: 8455
  %6565 = inttoptr i64 %6564 to i32*		; visa id: 8456
  store i32 %6563, i32* %6565, align 4, !alias.scope !642		; visa id: 8456
  %6566 = icmp eq i32 %6558, 0		; visa id: 8457
  br i1 %6566, label %._crit_edge337.._crit_edge337_crit_edge, label %6568, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8458

._crit_edge337.._crit_edge337_crit_edge:          ; preds = %._crit_edge337
; BB603 :
  %6567 = add nuw nsw i32 %6558, 1, !spirv.Decorations !631		; visa id: 8460
  br label %._crit_edge337, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8461

6568:                                             ; preds = %._crit_edge337
; BB604 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8463
  %6569 = load i64, i64* %6446, align 8		; visa id: 8463
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8464
  %6570 = icmp slt i32 %6555, %const_reg_dword
  %6571 = icmp slt i32 %69, %const_reg_dword1		; visa id: 8464
  %6572 = and i1 %6570, %6571		; visa id: 8465
  br i1 %6572, label %6573, label %.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8467

.._crit_edge70.2_crit_edge:                       ; preds = %6568
; BB:
  br label %._crit_edge70.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6573:                                             ; preds = %6568
; BB606 :
  %6574 = bitcast i64 %6569 to <2 x i32>		; visa id: 8469
  %6575 = extractelement <2 x i32> %6574, i32 0		; visa id: 8471
  %6576 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6575, i32 1
  %6577 = bitcast <2 x i32> %6576 to i64		; visa id: 8471
  %6578 = ashr exact i64 %6577, 32		; visa id: 8472
  %6579 = bitcast i64 %6578 to <2 x i32>		; visa id: 8473
  %6580 = extractelement <2 x i32> %6579, i32 0		; visa id: 8477
  %6581 = extractelement <2 x i32> %6579, i32 1		; visa id: 8477
  %6582 = ashr i64 %6569, 32		; visa id: 8477
  %6583 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6580, i32 %6581, i32 %50, i32 %51)
  %6584 = extractvalue { i32, i32 } %6583, 0		; visa id: 8478
  %6585 = extractvalue { i32, i32 } %6583, 1		; visa id: 8478
  %6586 = insertelement <2 x i32> undef, i32 %6584, i32 0		; visa id: 8485
  %6587 = insertelement <2 x i32> %6586, i32 %6585, i32 1		; visa id: 8486
  %6588 = bitcast <2 x i32> %6587 to i64		; visa id: 8487
  %6589 = add nsw i64 %6588, %6582, !spirv.Decorations !649		; visa id: 8491
  %6590 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !618		; visa id: 8492
  br i1 %86, label %6596, label %6591, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8493

6591:                                             ; preds = %6573
; BB607 :
  %6592 = shl i64 %6589, 2		; visa id: 8495
  %6593 = add i64 %.in, %6592		; visa id: 8496
  %6594 = inttoptr i64 %6593 to float addrspace(4)*		; visa id: 8497
  %6595 = addrspacecast float addrspace(4)* %6594 to float addrspace(1)*		; visa id: 8497
  store float %6590, float addrspace(1)* %6595, align 4		; visa id: 8498
  br label %._crit_edge70.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8499

6596:                                             ; preds = %6573
; BB608 :
  %6597 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6580, i32 %6581, i32 %47, i32 %48)
  %6598 = extractvalue { i32, i32 } %6597, 0		; visa id: 8501
  %6599 = extractvalue { i32, i32 } %6597, 1		; visa id: 8501
  %6600 = insertelement <2 x i32> undef, i32 %6598, i32 0		; visa id: 8508
  %6601 = insertelement <2 x i32> %6600, i32 %6599, i32 1		; visa id: 8509
  %6602 = bitcast <2 x i32> %6601 to i64		; visa id: 8510
  %6603 = shl i64 %6602, 2		; visa id: 8514
  %6604 = add i64 %.in399, %6603		; visa id: 8515
  %6605 = shl nsw i64 %6582, 2		; visa id: 8516
  %6606 = add i64 %6604, %6605		; visa id: 8517
  %6607 = inttoptr i64 %6606 to float addrspace(4)*		; visa id: 8518
  %6608 = addrspacecast float addrspace(4)* %6607 to float addrspace(1)*		; visa id: 8518
  %6609 = load float, float addrspace(1)* %6608, align 4		; visa id: 8519
  %6610 = fmul reassoc nsz arcp contract float %6609, %4, !spirv.Decorations !618		; visa id: 8520
  %6611 = fadd reassoc nsz arcp contract float %6590, %6610, !spirv.Decorations !618		; visa id: 8521
  %6612 = shl i64 %6589, 2		; visa id: 8522
  %6613 = add i64 %.in, %6612		; visa id: 8523
  %6614 = inttoptr i64 %6613 to float addrspace(4)*		; visa id: 8524
  %6615 = addrspacecast float addrspace(4)* %6614 to float addrspace(1)*		; visa id: 8524
  store float %6611, float addrspace(1)* %6615, align 4		; visa id: 8525
  br label %._crit_edge70.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8526

._crit_edge70.2:                                  ; preds = %.._crit_edge70.2_crit_edge, %6596, %6591
; BB609 :
  %6616 = add i32 %65, 3		; visa id: 8527
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8528
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8528
  %6617 = insertelement <2 x i32> undef, i32 %6616, i64 0		; visa id: 8528
  %6618 = insertelement <2 x i32> %6617, i32 %69, i64 1		; visa id: 8529
  store <2 x i32> %6618, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8532
  br label %._crit_edge338, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8534

._crit_edge338:                                   ; preds = %._crit_edge338.._crit_edge338_crit_edge, %._crit_edge70.2
; BB610 :
  %6619 = phi i32 [ 0, %._crit_edge70.2 ], [ %6628, %._crit_edge338.._crit_edge338_crit_edge ]
  %6620 = zext i32 %6619 to i64		; visa id: 8535
  %6621 = shl nuw nsw i64 %6620, 2		; visa id: 8536
  %6622 = add i64 %6430, %6621		; visa id: 8537
  %6623 = inttoptr i64 %6622 to i32*		; visa id: 8538
  %6624 = load i32, i32* %6623, align 4, !noalias !642		; visa id: 8538
  %6625 = add i64 %6426, %6621		; visa id: 8539
  %6626 = inttoptr i64 %6625 to i32*		; visa id: 8540
  store i32 %6624, i32* %6626, align 4, !alias.scope !642		; visa id: 8540
  %6627 = icmp eq i32 %6619, 0		; visa id: 8541
  br i1 %6627, label %._crit_edge338.._crit_edge338_crit_edge, label %6629, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8542

._crit_edge338.._crit_edge338_crit_edge:          ; preds = %._crit_edge338
; BB611 :
  %6628 = add nuw nsw i32 %6619, 1, !spirv.Decorations !631		; visa id: 8544
  br label %._crit_edge338, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8545

6629:                                             ; preds = %._crit_edge338
; BB612 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8547
  %6630 = load i64, i64* %6446, align 8		; visa id: 8547
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8548
  %6631 = icmp slt i32 %6616, %const_reg_dword
  %6632 = icmp slt i32 %69, %const_reg_dword1		; visa id: 8548
  %6633 = and i1 %6631, %6632		; visa id: 8549
  br i1 %6633, label %6634, label %..preheader1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8551

..preheader1_crit_edge:                           ; preds = %6629
; BB:
  br label %.preheader1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6634:                                             ; preds = %6629
; BB614 :
  %6635 = bitcast i64 %6630 to <2 x i32>		; visa id: 8553
  %6636 = extractelement <2 x i32> %6635, i32 0		; visa id: 8555
  %6637 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6636, i32 1
  %6638 = bitcast <2 x i32> %6637 to i64		; visa id: 8555
  %6639 = ashr exact i64 %6638, 32		; visa id: 8556
  %6640 = bitcast i64 %6639 to <2 x i32>		; visa id: 8557
  %6641 = extractelement <2 x i32> %6640, i32 0		; visa id: 8561
  %6642 = extractelement <2 x i32> %6640, i32 1		; visa id: 8561
  %6643 = ashr i64 %6630, 32		; visa id: 8561
  %6644 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6641, i32 %6642, i32 %50, i32 %51)
  %6645 = extractvalue { i32, i32 } %6644, 0		; visa id: 8562
  %6646 = extractvalue { i32, i32 } %6644, 1		; visa id: 8562
  %6647 = insertelement <2 x i32> undef, i32 %6645, i32 0		; visa id: 8569
  %6648 = insertelement <2 x i32> %6647, i32 %6646, i32 1		; visa id: 8570
  %6649 = bitcast <2 x i32> %6648 to i64		; visa id: 8571
  %6650 = add nsw i64 %6649, %6643, !spirv.Decorations !649		; visa id: 8575
  %6651 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !618		; visa id: 8576
  br i1 %86, label %6657, label %6652, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8577

6652:                                             ; preds = %6634
; BB615 :
  %6653 = shl i64 %6650, 2		; visa id: 8579
  %6654 = add i64 %.in, %6653		; visa id: 8580
  %6655 = inttoptr i64 %6654 to float addrspace(4)*		; visa id: 8581
  %6656 = addrspacecast float addrspace(4)* %6655 to float addrspace(1)*		; visa id: 8581
  store float %6651, float addrspace(1)* %6656, align 4		; visa id: 8582
  br label %.preheader1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8583

6657:                                             ; preds = %6634
; BB616 :
  %6658 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6641, i32 %6642, i32 %47, i32 %48)
  %6659 = extractvalue { i32, i32 } %6658, 0		; visa id: 8585
  %6660 = extractvalue { i32, i32 } %6658, 1		; visa id: 8585
  %6661 = insertelement <2 x i32> undef, i32 %6659, i32 0		; visa id: 8592
  %6662 = insertelement <2 x i32> %6661, i32 %6660, i32 1		; visa id: 8593
  %6663 = bitcast <2 x i32> %6662 to i64		; visa id: 8594
  %6664 = shl i64 %6663, 2		; visa id: 8598
  %6665 = add i64 %.in399, %6664		; visa id: 8599
  %6666 = shl nsw i64 %6643, 2		; visa id: 8600
  %6667 = add i64 %6665, %6666		; visa id: 8601
  %6668 = inttoptr i64 %6667 to float addrspace(4)*		; visa id: 8602
  %6669 = addrspacecast float addrspace(4)* %6668 to float addrspace(1)*		; visa id: 8602
  %6670 = load float, float addrspace(1)* %6669, align 4		; visa id: 8603
  %6671 = fmul reassoc nsz arcp contract float %6670, %4, !spirv.Decorations !618		; visa id: 8604
  %6672 = fadd reassoc nsz arcp contract float %6651, %6671, !spirv.Decorations !618		; visa id: 8605
  %6673 = shl i64 %6650, 2		; visa id: 8606
  %6674 = add i64 %.in, %6673		; visa id: 8607
  %6675 = inttoptr i64 %6674 to float addrspace(4)*		; visa id: 8608
  %6676 = addrspacecast float addrspace(4)* %6675 to float addrspace(1)*		; visa id: 8608
  store float %6672, float addrspace(1)* %6676, align 4		; visa id: 8609
  br label %.preheader1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8610

.preheader1:                                      ; preds = %..preheader1_crit_edge, %6657, %6652
; BB617 :
  %6677 = add i32 %69, 1		; visa id: 8611
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8612
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8612
  %6678 = insertelement <2 x i32> %6432, i32 %6677, i64 1		; visa id: 8612
  store <2 x i32> %6678, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8615
  br label %._crit_edge339, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8617

._crit_edge339:                                   ; preds = %._crit_edge339.._crit_edge339_crit_edge, %.preheader1
; BB618 :
  %6679 = phi i32 [ 0, %.preheader1 ], [ %6688, %._crit_edge339.._crit_edge339_crit_edge ]
  %6680 = zext i32 %6679 to i64		; visa id: 8618
  %6681 = shl nuw nsw i64 %6680, 2		; visa id: 8619
  %6682 = add i64 %6430, %6681		; visa id: 8620
  %6683 = inttoptr i64 %6682 to i32*		; visa id: 8621
  %6684 = load i32, i32* %6683, align 4, !noalias !642		; visa id: 8621
  %6685 = add i64 %6426, %6681		; visa id: 8622
  %6686 = inttoptr i64 %6685 to i32*		; visa id: 8623
  store i32 %6684, i32* %6686, align 4, !alias.scope !642		; visa id: 8623
  %6687 = icmp eq i32 %6679, 0		; visa id: 8624
  br i1 %6687, label %._crit_edge339.._crit_edge339_crit_edge, label %6689, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8625

._crit_edge339.._crit_edge339_crit_edge:          ; preds = %._crit_edge339
; BB619 :
  %6688 = add nuw nsw i32 %6679, 1, !spirv.Decorations !631		; visa id: 8627
  br label %._crit_edge339, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8628

6689:                                             ; preds = %._crit_edge339
; BB620 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8630
  %6690 = load i64, i64* %6446, align 8		; visa id: 8630
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8631
  %6691 = icmp slt i32 %6677, %const_reg_dword1		; visa id: 8631
  %6692 = icmp slt i32 %65, %const_reg_dword
  %6693 = and i1 %6692, %6691		; visa id: 8632
  br i1 %6693, label %6694, label %.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8634

.._crit_edge70.176_crit_edge:                     ; preds = %6689
; BB:
  br label %._crit_edge70.176, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6694:                                             ; preds = %6689
; BB622 :
  %6695 = bitcast i64 %6690 to <2 x i32>		; visa id: 8636
  %6696 = extractelement <2 x i32> %6695, i32 0		; visa id: 8638
  %6697 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6696, i32 1
  %6698 = bitcast <2 x i32> %6697 to i64		; visa id: 8638
  %6699 = ashr exact i64 %6698, 32		; visa id: 8639
  %6700 = bitcast i64 %6699 to <2 x i32>		; visa id: 8640
  %6701 = extractelement <2 x i32> %6700, i32 0		; visa id: 8644
  %6702 = extractelement <2 x i32> %6700, i32 1		; visa id: 8644
  %6703 = ashr i64 %6690, 32		; visa id: 8644
  %6704 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6701, i32 %6702, i32 %50, i32 %51)
  %6705 = extractvalue { i32, i32 } %6704, 0		; visa id: 8645
  %6706 = extractvalue { i32, i32 } %6704, 1		; visa id: 8645
  %6707 = insertelement <2 x i32> undef, i32 %6705, i32 0		; visa id: 8652
  %6708 = insertelement <2 x i32> %6707, i32 %6706, i32 1		; visa id: 8653
  %6709 = bitcast <2 x i32> %6708 to i64		; visa id: 8654
  %6710 = add nsw i64 %6709, %6703, !spirv.Decorations !649		; visa id: 8658
  %6711 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !618		; visa id: 8659
  br i1 %86, label %6717, label %6712, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8660

6712:                                             ; preds = %6694
; BB623 :
  %6713 = shl i64 %6710, 2		; visa id: 8662
  %6714 = add i64 %.in, %6713		; visa id: 8663
  %6715 = inttoptr i64 %6714 to float addrspace(4)*		; visa id: 8664
  %6716 = addrspacecast float addrspace(4)* %6715 to float addrspace(1)*		; visa id: 8664
  store float %6711, float addrspace(1)* %6716, align 4		; visa id: 8665
  br label %._crit_edge70.176, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8666

6717:                                             ; preds = %6694
; BB624 :
  %6718 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6701, i32 %6702, i32 %47, i32 %48)
  %6719 = extractvalue { i32, i32 } %6718, 0		; visa id: 8668
  %6720 = extractvalue { i32, i32 } %6718, 1		; visa id: 8668
  %6721 = insertelement <2 x i32> undef, i32 %6719, i32 0		; visa id: 8675
  %6722 = insertelement <2 x i32> %6721, i32 %6720, i32 1		; visa id: 8676
  %6723 = bitcast <2 x i32> %6722 to i64		; visa id: 8677
  %6724 = shl i64 %6723, 2		; visa id: 8681
  %6725 = add i64 %.in399, %6724		; visa id: 8682
  %6726 = shl nsw i64 %6703, 2		; visa id: 8683
  %6727 = add i64 %6725, %6726		; visa id: 8684
  %6728 = inttoptr i64 %6727 to float addrspace(4)*		; visa id: 8685
  %6729 = addrspacecast float addrspace(4)* %6728 to float addrspace(1)*		; visa id: 8685
  %6730 = load float, float addrspace(1)* %6729, align 4		; visa id: 8686
  %6731 = fmul reassoc nsz arcp contract float %6730, %4, !spirv.Decorations !618		; visa id: 8687
  %6732 = fadd reassoc nsz arcp contract float %6711, %6731, !spirv.Decorations !618		; visa id: 8688
  %6733 = shl i64 %6710, 2		; visa id: 8689
  %6734 = add i64 %.in, %6733		; visa id: 8690
  %6735 = inttoptr i64 %6734 to float addrspace(4)*		; visa id: 8691
  %6736 = addrspacecast float addrspace(4)* %6735 to float addrspace(1)*		; visa id: 8691
  store float %6732, float addrspace(1)* %6736, align 4		; visa id: 8692
  br label %._crit_edge70.176, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8693

._crit_edge70.176:                                ; preds = %.._crit_edge70.176_crit_edge, %6717, %6712
; BB625 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8694
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8694
  %6737 = insertelement <2 x i32> %6495, i32 %6677, i64 1		; visa id: 8694
  store <2 x i32> %6737, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8697
  br label %._crit_edge340, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8699

._crit_edge340:                                   ; preds = %._crit_edge340.._crit_edge340_crit_edge, %._crit_edge70.176
; BB626 :
  %6738 = phi i32 [ 0, %._crit_edge70.176 ], [ %6747, %._crit_edge340.._crit_edge340_crit_edge ]
  %6739 = zext i32 %6738 to i64		; visa id: 8700
  %6740 = shl nuw nsw i64 %6739, 2		; visa id: 8701
  %6741 = add i64 %6430, %6740		; visa id: 8702
  %6742 = inttoptr i64 %6741 to i32*		; visa id: 8703
  %6743 = load i32, i32* %6742, align 4, !noalias !642		; visa id: 8703
  %6744 = add i64 %6426, %6740		; visa id: 8704
  %6745 = inttoptr i64 %6744 to i32*		; visa id: 8705
  store i32 %6743, i32* %6745, align 4, !alias.scope !642		; visa id: 8705
  %6746 = icmp eq i32 %6738, 0		; visa id: 8706
  br i1 %6746, label %._crit_edge340.._crit_edge340_crit_edge, label %6748, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8707

._crit_edge340.._crit_edge340_crit_edge:          ; preds = %._crit_edge340
; BB627 :
  %6747 = add nuw nsw i32 %6738, 1, !spirv.Decorations !631		; visa id: 8709
  br label %._crit_edge340, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8710

6748:                                             ; preds = %._crit_edge340
; BB628 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8712
  %6749 = load i64, i64* %6446, align 8		; visa id: 8712
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8713
  %6750 = icmp slt i32 %6494, %const_reg_dword
  %6751 = icmp slt i32 %6677, %const_reg_dword1		; visa id: 8713
  %6752 = and i1 %6750, %6751		; visa id: 8714
  br i1 %6752, label %6753, label %.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8716

.._crit_edge70.1.1_crit_edge:                     ; preds = %6748
; BB:
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6753:                                             ; preds = %6748
; BB630 :
  %6754 = bitcast i64 %6749 to <2 x i32>		; visa id: 8718
  %6755 = extractelement <2 x i32> %6754, i32 0		; visa id: 8720
  %6756 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6755, i32 1
  %6757 = bitcast <2 x i32> %6756 to i64		; visa id: 8720
  %6758 = ashr exact i64 %6757, 32		; visa id: 8721
  %6759 = bitcast i64 %6758 to <2 x i32>		; visa id: 8722
  %6760 = extractelement <2 x i32> %6759, i32 0		; visa id: 8726
  %6761 = extractelement <2 x i32> %6759, i32 1		; visa id: 8726
  %6762 = ashr i64 %6749, 32		; visa id: 8726
  %6763 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6760, i32 %6761, i32 %50, i32 %51)
  %6764 = extractvalue { i32, i32 } %6763, 0		; visa id: 8727
  %6765 = extractvalue { i32, i32 } %6763, 1		; visa id: 8727
  %6766 = insertelement <2 x i32> undef, i32 %6764, i32 0		; visa id: 8734
  %6767 = insertelement <2 x i32> %6766, i32 %6765, i32 1		; visa id: 8735
  %6768 = bitcast <2 x i32> %6767 to i64		; visa id: 8736
  %6769 = add nsw i64 %6768, %6762, !spirv.Decorations !649		; visa id: 8740
  %6770 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !618		; visa id: 8741
  br i1 %86, label %6776, label %6771, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8742

6771:                                             ; preds = %6753
; BB631 :
  %6772 = shl i64 %6769, 2		; visa id: 8744
  %6773 = add i64 %.in, %6772		; visa id: 8745
  %6774 = inttoptr i64 %6773 to float addrspace(4)*		; visa id: 8746
  %6775 = addrspacecast float addrspace(4)* %6774 to float addrspace(1)*		; visa id: 8746
  store float %6770, float addrspace(1)* %6775, align 4		; visa id: 8747
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8748

6776:                                             ; preds = %6753
; BB632 :
  %6777 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6760, i32 %6761, i32 %47, i32 %48)
  %6778 = extractvalue { i32, i32 } %6777, 0		; visa id: 8750
  %6779 = extractvalue { i32, i32 } %6777, 1		; visa id: 8750
  %6780 = insertelement <2 x i32> undef, i32 %6778, i32 0		; visa id: 8757
  %6781 = insertelement <2 x i32> %6780, i32 %6779, i32 1		; visa id: 8758
  %6782 = bitcast <2 x i32> %6781 to i64		; visa id: 8759
  %6783 = shl i64 %6782, 2		; visa id: 8763
  %6784 = add i64 %.in399, %6783		; visa id: 8764
  %6785 = shl nsw i64 %6762, 2		; visa id: 8765
  %6786 = add i64 %6784, %6785		; visa id: 8766
  %6787 = inttoptr i64 %6786 to float addrspace(4)*		; visa id: 8767
  %6788 = addrspacecast float addrspace(4)* %6787 to float addrspace(1)*		; visa id: 8767
  %6789 = load float, float addrspace(1)* %6788, align 4		; visa id: 8768
  %6790 = fmul reassoc nsz arcp contract float %6789, %4, !spirv.Decorations !618		; visa id: 8769
  %6791 = fadd reassoc nsz arcp contract float %6770, %6790, !spirv.Decorations !618		; visa id: 8770
  %6792 = shl i64 %6769, 2		; visa id: 8771
  %6793 = add i64 %.in, %6792		; visa id: 8772
  %6794 = inttoptr i64 %6793 to float addrspace(4)*		; visa id: 8773
  %6795 = addrspacecast float addrspace(4)* %6794 to float addrspace(1)*		; visa id: 8773
  store float %6791, float addrspace(1)* %6795, align 4		; visa id: 8774
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8775

._crit_edge70.1.1:                                ; preds = %.._crit_edge70.1.1_crit_edge, %6776, %6771
; BB633 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8776
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8776
  %6796 = insertelement <2 x i32> %6556, i32 %6677, i64 1		; visa id: 8776
  store <2 x i32> %6796, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8779
  br label %._crit_edge341, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8781

._crit_edge341:                                   ; preds = %._crit_edge341.._crit_edge341_crit_edge, %._crit_edge70.1.1
; BB634 :
  %6797 = phi i32 [ 0, %._crit_edge70.1.1 ], [ %6806, %._crit_edge341.._crit_edge341_crit_edge ]
  %6798 = zext i32 %6797 to i64		; visa id: 8782
  %6799 = shl nuw nsw i64 %6798, 2		; visa id: 8783
  %6800 = add i64 %6430, %6799		; visa id: 8784
  %6801 = inttoptr i64 %6800 to i32*		; visa id: 8785
  %6802 = load i32, i32* %6801, align 4, !noalias !642		; visa id: 8785
  %6803 = add i64 %6426, %6799		; visa id: 8786
  %6804 = inttoptr i64 %6803 to i32*		; visa id: 8787
  store i32 %6802, i32* %6804, align 4, !alias.scope !642		; visa id: 8787
  %6805 = icmp eq i32 %6797, 0		; visa id: 8788
  br i1 %6805, label %._crit_edge341.._crit_edge341_crit_edge, label %6807, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8789

._crit_edge341.._crit_edge341_crit_edge:          ; preds = %._crit_edge341
; BB635 :
  %6806 = add nuw nsw i32 %6797, 1, !spirv.Decorations !631		; visa id: 8791
  br label %._crit_edge341, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8792

6807:                                             ; preds = %._crit_edge341
; BB636 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8794
  %6808 = load i64, i64* %6446, align 8		; visa id: 8794
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8795
  %6809 = icmp slt i32 %6555, %const_reg_dword
  %6810 = icmp slt i32 %6677, %const_reg_dword1		; visa id: 8795
  %6811 = and i1 %6809, %6810		; visa id: 8796
  br i1 %6811, label %6812, label %.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8798

.._crit_edge70.2.1_crit_edge:                     ; preds = %6807
; BB:
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6812:                                             ; preds = %6807
; BB638 :
  %6813 = bitcast i64 %6808 to <2 x i32>		; visa id: 8800
  %6814 = extractelement <2 x i32> %6813, i32 0		; visa id: 8802
  %6815 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6814, i32 1
  %6816 = bitcast <2 x i32> %6815 to i64		; visa id: 8802
  %6817 = ashr exact i64 %6816, 32		; visa id: 8803
  %6818 = bitcast i64 %6817 to <2 x i32>		; visa id: 8804
  %6819 = extractelement <2 x i32> %6818, i32 0		; visa id: 8808
  %6820 = extractelement <2 x i32> %6818, i32 1		; visa id: 8808
  %6821 = ashr i64 %6808, 32		; visa id: 8808
  %6822 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6819, i32 %6820, i32 %50, i32 %51)
  %6823 = extractvalue { i32, i32 } %6822, 0		; visa id: 8809
  %6824 = extractvalue { i32, i32 } %6822, 1		; visa id: 8809
  %6825 = insertelement <2 x i32> undef, i32 %6823, i32 0		; visa id: 8816
  %6826 = insertelement <2 x i32> %6825, i32 %6824, i32 1		; visa id: 8817
  %6827 = bitcast <2 x i32> %6826 to i64		; visa id: 8818
  %6828 = add nsw i64 %6827, %6821, !spirv.Decorations !649		; visa id: 8822
  %6829 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !618		; visa id: 8823
  br i1 %86, label %6835, label %6830, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8824

6830:                                             ; preds = %6812
; BB639 :
  %6831 = shl i64 %6828, 2		; visa id: 8826
  %6832 = add i64 %.in, %6831		; visa id: 8827
  %6833 = inttoptr i64 %6832 to float addrspace(4)*		; visa id: 8828
  %6834 = addrspacecast float addrspace(4)* %6833 to float addrspace(1)*		; visa id: 8828
  store float %6829, float addrspace(1)* %6834, align 4		; visa id: 8829
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8830

6835:                                             ; preds = %6812
; BB640 :
  %6836 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6819, i32 %6820, i32 %47, i32 %48)
  %6837 = extractvalue { i32, i32 } %6836, 0		; visa id: 8832
  %6838 = extractvalue { i32, i32 } %6836, 1		; visa id: 8832
  %6839 = insertelement <2 x i32> undef, i32 %6837, i32 0		; visa id: 8839
  %6840 = insertelement <2 x i32> %6839, i32 %6838, i32 1		; visa id: 8840
  %6841 = bitcast <2 x i32> %6840 to i64		; visa id: 8841
  %6842 = shl i64 %6841, 2		; visa id: 8845
  %6843 = add i64 %.in399, %6842		; visa id: 8846
  %6844 = shl nsw i64 %6821, 2		; visa id: 8847
  %6845 = add i64 %6843, %6844		; visa id: 8848
  %6846 = inttoptr i64 %6845 to float addrspace(4)*		; visa id: 8849
  %6847 = addrspacecast float addrspace(4)* %6846 to float addrspace(1)*		; visa id: 8849
  %6848 = load float, float addrspace(1)* %6847, align 4		; visa id: 8850
  %6849 = fmul reassoc nsz arcp contract float %6848, %4, !spirv.Decorations !618		; visa id: 8851
  %6850 = fadd reassoc nsz arcp contract float %6829, %6849, !spirv.Decorations !618		; visa id: 8852
  %6851 = shl i64 %6828, 2		; visa id: 8853
  %6852 = add i64 %.in, %6851		; visa id: 8854
  %6853 = inttoptr i64 %6852 to float addrspace(4)*		; visa id: 8855
  %6854 = addrspacecast float addrspace(4)* %6853 to float addrspace(1)*		; visa id: 8855
  store float %6850, float addrspace(1)* %6854, align 4		; visa id: 8856
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8857

._crit_edge70.2.1:                                ; preds = %.._crit_edge70.2.1_crit_edge, %6835, %6830
; BB641 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8858
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8858
  %6855 = insertelement <2 x i32> %6617, i32 %6677, i64 1		; visa id: 8858
  store <2 x i32> %6855, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8861
  br label %._crit_edge342, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8863

._crit_edge342:                                   ; preds = %._crit_edge342.._crit_edge342_crit_edge, %._crit_edge70.2.1
; BB642 :
  %6856 = phi i32 [ 0, %._crit_edge70.2.1 ], [ %6865, %._crit_edge342.._crit_edge342_crit_edge ]
  %6857 = zext i32 %6856 to i64		; visa id: 8864
  %6858 = shl nuw nsw i64 %6857, 2		; visa id: 8865
  %6859 = add i64 %6430, %6858		; visa id: 8866
  %6860 = inttoptr i64 %6859 to i32*		; visa id: 8867
  %6861 = load i32, i32* %6860, align 4, !noalias !642		; visa id: 8867
  %6862 = add i64 %6426, %6858		; visa id: 8868
  %6863 = inttoptr i64 %6862 to i32*		; visa id: 8869
  store i32 %6861, i32* %6863, align 4, !alias.scope !642		; visa id: 8869
  %6864 = icmp eq i32 %6856, 0		; visa id: 8870
  br i1 %6864, label %._crit_edge342.._crit_edge342_crit_edge, label %6866, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8871

._crit_edge342.._crit_edge342_crit_edge:          ; preds = %._crit_edge342
; BB643 :
  %6865 = add nuw nsw i32 %6856, 1, !spirv.Decorations !631		; visa id: 8873
  br label %._crit_edge342, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8874

6866:                                             ; preds = %._crit_edge342
; BB644 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8876
  %6867 = load i64, i64* %6446, align 8		; visa id: 8876
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8877
  %6868 = icmp slt i32 %6616, %const_reg_dword
  %6869 = icmp slt i32 %6677, %const_reg_dword1		; visa id: 8877
  %6870 = and i1 %6868, %6869		; visa id: 8878
  br i1 %6870, label %6871, label %..preheader1.1_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8880

..preheader1.1_crit_edge:                         ; preds = %6866
; BB:
  br label %.preheader1.1, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6871:                                             ; preds = %6866
; BB646 :
  %6872 = bitcast i64 %6867 to <2 x i32>		; visa id: 8882
  %6873 = extractelement <2 x i32> %6872, i32 0		; visa id: 8884
  %6874 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6873, i32 1
  %6875 = bitcast <2 x i32> %6874 to i64		; visa id: 8884
  %6876 = ashr exact i64 %6875, 32		; visa id: 8885
  %6877 = bitcast i64 %6876 to <2 x i32>		; visa id: 8886
  %6878 = extractelement <2 x i32> %6877, i32 0		; visa id: 8890
  %6879 = extractelement <2 x i32> %6877, i32 1		; visa id: 8890
  %6880 = ashr i64 %6867, 32		; visa id: 8890
  %6881 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6878, i32 %6879, i32 %50, i32 %51)
  %6882 = extractvalue { i32, i32 } %6881, 0		; visa id: 8891
  %6883 = extractvalue { i32, i32 } %6881, 1		; visa id: 8891
  %6884 = insertelement <2 x i32> undef, i32 %6882, i32 0		; visa id: 8898
  %6885 = insertelement <2 x i32> %6884, i32 %6883, i32 1		; visa id: 8899
  %6886 = bitcast <2 x i32> %6885 to i64		; visa id: 8900
  %6887 = add nsw i64 %6886, %6880, !spirv.Decorations !649		; visa id: 8904
  %6888 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !618		; visa id: 8905
  br i1 %86, label %6894, label %6889, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8906

6889:                                             ; preds = %6871
; BB647 :
  %6890 = shl i64 %6887, 2		; visa id: 8908
  %6891 = add i64 %.in, %6890		; visa id: 8909
  %6892 = inttoptr i64 %6891 to float addrspace(4)*		; visa id: 8910
  %6893 = addrspacecast float addrspace(4)* %6892 to float addrspace(1)*		; visa id: 8910
  store float %6888, float addrspace(1)* %6893, align 4		; visa id: 8911
  br label %.preheader1.1, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8912

6894:                                             ; preds = %6871
; BB648 :
  %6895 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6878, i32 %6879, i32 %47, i32 %48)
  %6896 = extractvalue { i32, i32 } %6895, 0		; visa id: 8914
  %6897 = extractvalue { i32, i32 } %6895, 1		; visa id: 8914
  %6898 = insertelement <2 x i32> undef, i32 %6896, i32 0		; visa id: 8921
  %6899 = insertelement <2 x i32> %6898, i32 %6897, i32 1		; visa id: 8922
  %6900 = bitcast <2 x i32> %6899 to i64		; visa id: 8923
  %6901 = shl i64 %6900, 2		; visa id: 8927
  %6902 = add i64 %.in399, %6901		; visa id: 8928
  %6903 = shl nsw i64 %6880, 2		; visa id: 8929
  %6904 = add i64 %6902, %6903		; visa id: 8930
  %6905 = inttoptr i64 %6904 to float addrspace(4)*		; visa id: 8931
  %6906 = addrspacecast float addrspace(4)* %6905 to float addrspace(1)*		; visa id: 8931
  %6907 = load float, float addrspace(1)* %6906, align 4		; visa id: 8932
  %6908 = fmul reassoc nsz arcp contract float %6907, %4, !spirv.Decorations !618		; visa id: 8933
  %6909 = fadd reassoc nsz arcp contract float %6888, %6908, !spirv.Decorations !618		; visa id: 8934
  %6910 = shl i64 %6887, 2		; visa id: 8935
  %6911 = add i64 %.in, %6910		; visa id: 8936
  %6912 = inttoptr i64 %6911 to float addrspace(4)*		; visa id: 8937
  %6913 = addrspacecast float addrspace(4)* %6912 to float addrspace(1)*		; visa id: 8937
  store float %6909, float addrspace(1)* %6913, align 4		; visa id: 8938
  br label %.preheader1.1, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 8939

.preheader1.1:                                    ; preds = %..preheader1.1_crit_edge, %6894, %6889
; BB649 :
  %6914 = add i32 %69, 2		; visa id: 8940
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8941
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8941
  %6915 = insertelement <2 x i32> %6432, i32 %6914, i64 1		; visa id: 8941
  store <2 x i32> %6915, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 8944
  br label %._crit_edge343, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8946

._crit_edge343:                                   ; preds = %._crit_edge343.._crit_edge343_crit_edge, %.preheader1.1
; BB650 :
  %6916 = phi i32 [ 0, %.preheader1.1 ], [ %6925, %._crit_edge343.._crit_edge343_crit_edge ]
  %6917 = zext i32 %6916 to i64		; visa id: 8947
  %6918 = shl nuw nsw i64 %6917, 2		; visa id: 8948
  %6919 = add i64 %6430, %6918		; visa id: 8949
  %6920 = inttoptr i64 %6919 to i32*		; visa id: 8950
  %6921 = load i32, i32* %6920, align 4, !noalias !642		; visa id: 8950
  %6922 = add i64 %6426, %6918		; visa id: 8951
  %6923 = inttoptr i64 %6922 to i32*		; visa id: 8952
  store i32 %6921, i32* %6923, align 4, !alias.scope !642		; visa id: 8952
  %6924 = icmp eq i32 %6916, 0		; visa id: 8953
  br i1 %6924, label %._crit_edge343.._crit_edge343_crit_edge, label %6926, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 8954

._crit_edge343.._crit_edge343_crit_edge:          ; preds = %._crit_edge343
; BB651 :
  %6925 = add nuw nsw i32 %6916, 1, !spirv.Decorations !631		; visa id: 8956
  br label %._crit_edge343, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 8957

6926:                                             ; preds = %._crit_edge343
; BB652 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 8959
  %6927 = load i64, i64* %6446, align 8		; visa id: 8959
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 8960
  %6928 = icmp slt i32 %6914, %const_reg_dword1		; visa id: 8960
  %6929 = icmp slt i32 %65, %const_reg_dword
  %6930 = and i1 %6929, %6928		; visa id: 8961
  br i1 %6930, label %6931, label %.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 8963

.._crit_edge70.277_crit_edge:                     ; preds = %6926
; BB:
  br label %._crit_edge70.277, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6931:                                             ; preds = %6926
; BB654 :
  %6932 = bitcast i64 %6927 to <2 x i32>		; visa id: 8965
  %6933 = extractelement <2 x i32> %6932, i32 0		; visa id: 8967
  %6934 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6933, i32 1
  %6935 = bitcast <2 x i32> %6934 to i64		; visa id: 8967
  %6936 = ashr exact i64 %6935, 32		; visa id: 8968
  %6937 = bitcast i64 %6936 to <2 x i32>		; visa id: 8969
  %6938 = extractelement <2 x i32> %6937, i32 0		; visa id: 8973
  %6939 = extractelement <2 x i32> %6937, i32 1		; visa id: 8973
  %6940 = ashr i64 %6927, 32		; visa id: 8973
  %6941 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6938, i32 %6939, i32 %50, i32 %51)
  %6942 = extractvalue { i32, i32 } %6941, 0		; visa id: 8974
  %6943 = extractvalue { i32, i32 } %6941, 1		; visa id: 8974
  %6944 = insertelement <2 x i32> undef, i32 %6942, i32 0		; visa id: 8981
  %6945 = insertelement <2 x i32> %6944, i32 %6943, i32 1		; visa id: 8982
  %6946 = bitcast <2 x i32> %6945 to i64		; visa id: 8983
  %6947 = add nsw i64 %6946, %6940, !spirv.Decorations !649		; visa id: 8987
  %6948 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !618		; visa id: 8988
  br i1 %86, label %6954, label %6949, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 8989

6949:                                             ; preds = %6931
; BB655 :
  %6950 = shl i64 %6947, 2		; visa id: 8991
  %6951 = add i64 %.in, %6950		; visa id: 8992
  %6952 = inttoptr i64 %6951 to float addrspace(4)*		; visa id: 8993
  %6953 = addrspacecast float addrspace(4)* %6952 to float addrspace(1)*		; visa id: 8993
  store float %6948, float addrspace(1)* %6953, align 4		; visa id: 8994
  br label %._crit_edge70.277, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 8995

6954:                                             ; preds = %6931
; BB656 :
  %6955 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6938, i32 %6939, i32 %47, i32 %48)
  %6956 = extractvalue { i32, i32 } %6955, 0		; visa id: 8997
  %6957 = extractvalue { i32, i32 } %6955, 1		; visa id: 8997
  %6958 = insertelement <2 x i32> undef, i32 %6956, i32 0		; visa id: 9004
  %6959 = insertelement <2 x i32> %6958, i32 %6957, i32 1		; visa id: 9005
  %6960 = bitcast <2 x i32> %6959 to i64		; visa id: 9006
  %6961 = shl i64 %6960, 2		; visa id: 9010
  %6962 = add i64 %.in399, %6961		; visa id: 9011
  %6963 = shl nsw i64 %6940, 2		; visa id: 9012
  %6964 = add i64 %6962, %6963		; visa id: 9013
  %6965 = inttoptr i64 %6964 to float addrspace(4)*		; visa id: 9014
  %6966 = addrspacecast float addrspace(4)* %6965 to float addrspace(1)*		; visa id: 9014
  %6967 = load float, float addrspace(1)* %6966, align 4		; visa id: 9015
  %6968 = fmul reassoc nsz arcp contract float %6967, %4, !spirv.Decorations !618		; visa id: 9016
  %6969 = fadd reassoc nsz arcp contract float %6948, %6968, !spirv.Decorations !618		; visa id: 9017
  %6970 = shl i64 %6947, 2		; visa id: 9018
  %6971 = add i64 %.in, %6970		; visa id: 9019
  %6972 = inttoptr i64 %6971 to float addrspace(4)*		; visa id: 9020
  %6973 = addrspacecast float addrspace(4)* %6972 to float addrspace(1)*		; visa id: 9020
  store float %6969, float addrspace(1)* %6973, align 4		; visa id: 9021
  br label %._crit_edge70.277, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9022

._crit_edge70.277:                                ; preds = %.._crit_edge70.277_crit_edge, %6954, %6949
; BB657 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9023
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9023
  %6974 = insertelement <2 x i32> %6495, i32 %6914, i64 1		; visa id: 9023
  store <2 x i32> %6974, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9026
  br label %._crit_edge344, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9028

._crit_edge344:                                   ; preds = %._crit_edge344.._crit_edge344_crit_edge, %._crit_edge70.277
; BB658 :
  %6975 = phi i32 [ 0, %._crit_edge70.277 ], [ %6984, %._crit_edge344.._crit_edge344_crit_edge ]
  %6976 = zext i32 %6975 to i64		; visa id: 9029
  %6977 = shl nuw nsw i64 %6976, 2		; visa id: 9030
  %6978 = add i64 %6430, %6977		; visa id: 9031
  %6979 = inttoptr i64 %6978 to i32*		; visa id: 9032
  %6980 = load i32, i32* %6979, align 4, !noalias !642		; visa id: 9032
  %6981 = add i64 %6426, %6977		; visa id: 9033
  %6982 = inttoptr i64 %6981 to i32*		; visa id: 9034
  store i32 %6980, i32* %6982, align 4, !alias.scope !642		; visa id: 9034
  %6983 = icmp eq i32 %6975, 0		; visa id: 9035
  br i1 %6983, label %._crit_edge344.._crit_edge344_crit_edge, label %6985, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9036

._crit_edge344.._crit_edge344_crit_edge:          ; preds = %._crit_edge344
; BB659 :
  %6984 = add nuw nsw i32 %6975, 1, !spirv.Decorations !631		; visa id: 9038
  br label %._crit_edge344, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9039

6985:                                             ; preds = %._crit_edge344
; BB660 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9041
  %6986 = load i64, i64* %6446, align 8		; visa id: 9041
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9042
  %6987 = icmp slt i32 %6494, %const_reg_dword
  %6988 = icmp slt i32 %6914, %const_reg_dword1		; visa id: 9042
  %6989 = and i1 %6987, %6988		; visa id: 9043
  br i1 %6989, label %6990, label %.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9045

.._crit_edge70.1.2_crit_edge:                     ; preds = %6985
; BB:
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

6990:                                             ; preds = %6985
; BB662 :
  %6991 = bitcast i64 %6986 to <2 x i32>		; visa id: 9047
  %6992 = extractelement <2 x i32> %6991, i32 0		; visa id: 9049
  %6993 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %6992, i32 1
  %6994 = bitcast <2 x i32> %6993 to i64		; visa id: 9049
  %6995 = ashr exact i64 %6994, 32		; visa id: 9050
  %6996 = bitcast i64 %6995 to <2 x i32>		; visa id: 9051
  %6997 = extractelement <2 x i32> %6996, i32 0		; visa id: 9055
  %6998 = extractelement <2 x i32> %6996, i32 1		; visa id: 9055
  %6999 = ashr i64 %6986, 32		; visa id: 9055
  %7000 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6997, i32 %6998, i32 %50, i32 %51)
  %7001 = extractvalue { i32, i32 } %7000, 0		; visa id: 9056
  %7002 = extractvalue { i32, i32 } %7000, 1		; visa id: 9056
  %7003 = insertelement <2 x i32> undef, i32 %7001, i32 0		; visa id: 9063
  %7004 = insertelement <2 x i32> %7003, i32 %7002, i32 1		; visa id: 9064
  %7005 = bitcast <2 x i32> %7004 to i64		; visa id: 9065
  %7006 = add nsw i64 %7005, %6999, !spirv.Decorations !649		; visa id: 9069
  %7007 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !618		; visa id: 9070
  br i1 %86, label %7013, label %7008, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9071

7008:                                             ; preds = %6990
; BB663 :
  %7009 = shl i64 %7006, 2		; visa id: 9073
  %7010 = add i64 %.in, %7009		; visa id: 9074
  %7011 = inttoptr i64 %7010 to float addrspace(4)*		; visa id: 9075
  %7012 = addrspacecast float addrspace(4)* %7011 to float addrspace(1)*		; visa id: 9075
  store float %7007, float addrspace(1)* %7012, align 4		; visa id: 9076
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9077

7013:                                             ; preds = %6990
; BB664 :
  %7014 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %6997, i32 %6998, i32 %47, i32 %48)
  %7015 = extractvalue { i32, i32 } %7014, 0		; visa id: 9079
  %7016 = extractvalue { i32, i32 } %7014, 1		; visa id: 9079
  %7017 = insertelement <2 x i32> undef, i32 %7015, i32 0		; visa id: 9086
  %7018 = insertelement <2 x i32> %7017, i32 %7016, i32 1		; visa id: 9087
  %7019 = bitcast <2 x i32> %7018 to i64		; visa id: 9088
  %7020 = shl i64 %7019, 2		; visa id: 9092
  %7021 = add i64 %.in399, %7020		; visa id: 9093
  %7022 = shl nsw i64 %6999, 2		; visa id: 9094
  %7023 = add i64 %7021, %7022		; visa id: 9095
  %7024 = inttoptr i64 %7023 to float addrspace(4)*		; visa id: 9096
  %7025 = addrspacecast float addrspace(4)* %7024 to float addrspace(1)*		; visa id: 9096
  %7026 = load float, float addrspace(1)* %7025, align 4		; visa id: 9097
  %7027 = fmul reassoc nsz arcp contract float %7026, %4, !spirv.Decorations !618		; visa id: 9098
  %7028 = fadd reassoc nsz arcp contract float %7007, %7027, !spirv.Decorations !618		; visa id: 9099
  %7029 = shl i64 %7006, 2		; visa id: 9100
  %7030 = add i64 %.in, %7029		; visa id: 9101
  %7031 = inttoptr i64 %7030 to float addrspace(4)*		; visa id: 9102
  %7032 = addrspacecast float addrspace(4)* %7031 to float addrspace(1)*		; visa id: 9102
  store float %7028, float addrspace(1)* %7032, align 4		; visa id: 9103
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9104

._crit_edge70.1.2:                                ; preds = %.._crit_edge70.1.2_crit_edge, %7013, %7008
; BB665 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9105
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9105
  %7033 = insertelement <2 x i32> %6556, i32 %6914, i64 1		; visa id: 9105
  store <2 x i32> %7033, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9108
  br label %._crit_edge345, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9110

._crit_edge345:                                   ; preds = %._crit_edge345.._crit_edge345_crit_edge, %._crit_edge70.1.2
; BB666 :
  %7034 = phi i32 [ 0, %._crit_edge70.1.2 ], [ %7043, %._crit_edge345.._crit_edge345_crit_edge ]
  %7035 = zext i32 %7034 to i64		; visa id: 9111
  %7036 = shl nuw nsw i64 %7035, 2		; visa id: 9112
  %7037 = add i64 %6430, %7036		; visa id: 9113
  %7038 = inttoptr i64 %7037 to i32*		; visa id: 9114
  %7039 = load i32, i32* %7038, align 4, !noalias !642		; visa id: 9114
  %7040 = add i64 %6426, %7036		; visa id: 9115
  %7041 = inttoptr i64 %7040 to i32*		; visa id: 9116
  store i32 %7039, i32* %7041, align 4, !alias.scope !642		; visa id: 9116
  %7042 = icmp eq i32 %7034, 0		; visa id: 9117
  br i1 %7042, label %._crit_edge345.._crit_edge345_crit_edge, label %7044, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9118

._crit_edge345.._crit_edge345_crit_edge:          ; preds = %._crit_edge345
; BB667 :
  %7043 = add nuw nsw i32 %7034, 1, !spirv.Decorations !631		; visa id: 9120
  br label %._crit_edge345, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9121

7044:                                             ; preds = %._crit_edge345
; BB668 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9123
  %7045 = load i64, i64* %6446, align 8		; visa id: 9123
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9124
  %7046 = icmp slt i32 %6555, %const_reg_dword
  %7047 = icmp slt i32 %6914, %const_reg_dword1		; visa id: 9124
  %7048 = and i1 %7046, %7047		; visa id: 9125
  br i1 %7048, label %7049, label %.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9127

.._crit_edge70.2.2_crit_edge:                     ; preds = %7044
; BB:
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7049:                                             ; preds = %7044
; BB670 :
  %7050 = bitcast i64 %7045 to <2 x i32>		; visa id: 9129
  %7051 = extractelement <2 x i32> %7050, i32 0		; visa id: 9131
  %7052 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7051, i32 1
  %7053 = bitcast <2 x i32> %7052 to i64		; visa id: 9131
  %7054 = ashr exact i64 %7053, 32		; visa id: 9132
  %7055 = bitcast i64 %7054 to <2 x i32>		; visa id: 9133
  %7056 = extractelement <2 x i32> %7055, i32 0		; visa id: 9137
  %7057 = extractelement <2 x i32> %7055, i32 1		; visa id: 9137
  %7058 = ashr i64 %7045, 32		; visa id: 9137
  %7059 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7056, i32 %7057, i32 %50, i32 %51)
  %7060 = extractvalue { i32, i32 } %7059, 0		; visa id: 9138
  %7061 = extractvalue { i32, i32 } %7059, 1		; visa id: 9138
  %7062 = insertelement <2 x i32> undef, i32 %7060, i32 0		; visa id: 9145
  %7063 = insertelement <2 x i32> %7062, i32 %7061, i32 1		; visa id: 9146
  %7064 = bitcast <2 x i32> %7063 to i64		; visa id: 9147
  %7065 = add nsw i64 %7064, %7058, !spirv.Decorations !649		; visa id: 9151
  %7066 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !618		; visa id: 9152
  br i1 %86, label %7072, label %7067, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9153

7067:                                             ; preds = %7049
; BB671 :
  %7068 = shl i64 %7065, 2		; visa id: 9155
  %7069 = add i64 %.in, %7068		; visa id: 9156
  %7070 = inttoptr i64 %7069 to float addrspace(4)*		; visa id: 9157
  %7071 = addrspacecast float addrspace(4)* %7070 to float addrspace(1)*		; visa id: 9157
  store float %7066, float addrspace(1)* %7071, align 4		; visa id: 9158
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9159

7072:                                             ; preds = %7049
; BB672 :
  %7073 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7056, i32 %7057, i32 %47, i32 %48)
  %7074 = extractvalue { i32, i32 } %7073, 0		; visa id: 9161
  %7075 = extractvalue { i32, i32 } %7073, 1		; visa id: 9161
  %7076 = insertelement <2 x i32> undef, i32 %7074, i32 0		; visa id: 9168
  %7077 = insertelement <2 x i32> %7076, i32 %7075, i32 1		; visa id: 9169
  %7078 = bitcast <2 x i32> %7077 to i64		; visa id: 9170
  %7079 = shl i64 %7078, 2		; visa id: 9174
  %7080 = add i64 %.in399, %7079		; visa id: 9175
  %7081 = shl nsw i64 %7058, 2		; visa id: 9176
  %7082 = add i64 %7080, %7081		; visa id: 9177
  %7083 = inttoptr i64 %7082 to float addrspace(4)*		; visa id: 9178
  %7084 = addrspacecast float addrspace(4)* %7083 to float addrspace(1)*		; visa id: 9178
  %7085 = load float, float addrspace(1)* %7084, align 4		; visa id: 9179
  %7086 = fmul reassoc nsz arcp contract float %7085, %4, !spirv.Decorations !618		; visa id: 9180
  %7087 = fadd reassoc nsz arcp contract float %7066, %7086, !spirv.Decorations !618		; visa id: 9181
  %7088 = shl i64 %7065, 2		; visa id: 9182
  %7089 = add i64 %.in, %7088		; visa id: 9183
  %7090 = inttoptr i64 %7089 to float addrspace(4)*		; visa id: 9184
  %7091 = addrspacecast float addrspace(4)* %7090 to float addrspace(1)*		; visa id: 9184
  store float %7087, float addrspace(1)* %7091, align 4		; visa id: 9185
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9186

._crit_edge70.2.2:                                ; preds = %.._crit_edge70.2.2_crit_edge, %7072, %7067
; BB673 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9187
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9187
  %7092 = insertelement <2 x i32> %6617, i32 %6914, i64 1		; visa id: 9187
  store <2 x i32> %7092, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9190
  br label %._crit_edge346, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9192

._crit_edge346:                                   ; preds = %._crit_edge346.._crit_edge346_crit_edge, %._crit_edge70.2.2
; BB674 :
  %7093 = phi i32 [ 0, %._crit_edge70.2.2 ], [ %7102, %._crit_edge346.._crit_edge346_crit_edge ]
  %7094 = zext i32 %7093 to i64		; visa id: 9193
  %7095 = shl nuw nsw i64 %7094, 2		; visa id: 9194
  %7096 = add i64 %6430, %7095		; visa id: 9195
  %7097 = inttoptr i64 %7096 to i32*		; visa id: 9196
  %7098 = load i32, i32* %7097, align 4, !noalias !642		; visa id: 9196
  %7099 = add i64 %6426, %7095		; visa id: 9197
  %7100 = inttoptr i64 %7099 to i32*		; visa id: 9198
  store i32 %7098, i32* %7100, align 4, !alias.scope !642		; visa id: 9198
  %7101 = icmp eq i32 %7093, 0		; visa id: 9199
  br i1 %7101, label %._crit_edge346.._crit_edge346_crit_edge, label %7103, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9200

._crit_edge346.._crit_edge346_crit_edge:          ; preds = %._crit_edge346
; BB675 :
  %7102 = add nuw nsw i32 %7093, 1, !spirv.Decorations !631		; visa id: 9202
  br label %._crit_edge346, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9203

7103:                                             ; preds = %._crit_edge346
; BB676 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9205
  %7104 = load i64, i64* %6446, align 8		; visa id: 9205
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9206
  %7105 = icmp slt i32 %6616, %const_reg_dword
  %7106 = icmp slt i32 %6914, %const_reg_dword1		; visa id: 9206
  %7107 = and i1 %7105, %7106		; visa id: 9207
  br i1 %7107, label %7108, label %..preheader1.2_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9209

..preheader1.2_crit_edge:                         ; preds = %7103
; BB:
  br label %.preheader1.2, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7108:                                             ; preds = %7103
; BB678 :
  %7109 = bitcast i64 %7104 to <2 x i32>		; visa id: 9211
  %7110 = extractelement <2 x i32> %7109, i32 0		; visa id: 9213
  %7111 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7110, i32 1
  %7112 = bitcast <2 x i32> %7111 to i64		; visa id: 9213
  %7113 = ashr exact i64 %7112, 32		; visa id: 9214
  %7114 = bitcast i64 %7113 to <2 x i32>		; visa id: 9215
  %7115 = extractelement <2 x i32> %7114, i32 0		; visa id: 9219
  %7116 = extractelement <2 x i32> %7114, i32 1		; visa id: 9219
  %7117 = ashr i64 %7104, 32		; visa id: 9219
  %7118 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7115, i32 %7116, i32 %50, i32 %51)
  %7119 = extractvalue { i32, i32 } %7118, 0		; visa id: 9220
  %7120 = extractvalue { i32, i32 } %7118, 1		; visa id: 9220
  %7121 = insertelement <2 x i32> undef, i32 %7119, i32 0		; visa id: 9227
  %7122 = insertelement <2 x i32> %7121, i32 %7120, i32 1		; visa id: 9228
  %7123 = bitcast <2 x i32> %7122 to i64		; visa id: 9229
  %7124 = add nsw i64 %7123, %7117, !spirv.Decorations !649		; visa id: 9233
  %7125 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !618		; visa id: 9234
  br i1 %86, label %7131, label %7126, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9235

7126:                                             ; preds = %7108
; BB679 :
  %7127 = shl i64 %7124, 2		; visa id: 9237
  %7128 = add i64 %.in, %7127		; visa id: 9238
  %7129 = inttoptr i64 %7128 to float addrspace(4)*		; visa id: 9239
  %7130 = addrspacecast float addrspace(4)* %7129 to float addrspace(1)*		; visa id: 9239
  store float %7125, float addrspace(1)* %7130, align 4		; visa id: 9240
  br label %.preheader1.2, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9241

7131:                                             ; preds = %7108
; BB680 :
  %7132 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7115, i32 %7116, i32 %47, i32 %48)
  %7133 = extractvalue { i32, i32 } %7132, 0		; visa id: 9243
  %7134 = extractvalue { i32, i32 } %7132, 1		; visa id: 9243
  %7135 = insertelement <2 x i32> undef, i32 %7133, i32 0		; visa id: 9250
  %7136 = insertelement <2 x i32> %7135, i32 %7134, i32 1		; visa id: 9251
  %7137 = bitcast <2 x i32> %7136 to i64		; visa id: 9252
  %7138 = shl i64 %7137, 2		; visa id: 9256
  %7139 = add i64 %.in399, %7138		; visa id: 9257
  %7140 = shl nsw i64 %7117, 2		; visa id: 9258
  %7141 = add i64 %7139, %7140		; visa id: 9259
  %7142 = inttoptr i64 %7141 to float addrspace(4)*		; visa id: 9260
  %7143 = addrspacecast float addrspace(4)* %7142 to float addrspace(1)*		; visa id: 9260
  %7144 = load float, float addrspace(1)* %7143, align 4		; visa id: 9261
  %7145 = fmul reassoc nsz arcp contract float %7144, %4, !spirv.Decorations !618		; visa id: 9262
  %7146 = fadd reassoc nsz arcp contract float %7125, %7145, !spirv.Decorations !618		; visa id: 9263
  %7147 = shl i64 %7124, 2		; visa id: 9264
  %7148 = add i64 %.in, %7147		; visa id: 9265
  %7149 = inttoptr i64 %7148 to float addrspace(4)*		; visa id: 9266
  %7150 = addrspacecast float addrspace(4)* %7149 to float addrspace(1)*		; visa id: 9266
  store float %7146, float addrspace(1)* %7150, align 4		; visa id: 9267
  br label %.preheader1.2, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9268

.preheader1.2:                                    ; preds = %..preheader1.2_crit_edge, %7131, %7126
; BB681 :
  %7151 = add i32 %69, 3		; visa id: 9269
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9270
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9270
  %7152 = insertelement <2 x i32> %6432, i32 %7151, i64 1		; visa id: 9270
  store <2 x i32> %7152, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9273
  br label %._crit_edge347, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9275

._crit_edge347:                                   ; preds = %._crit_edge347.._crit_edge347_crit_edge, %.preheader1.2
; BB682 :
  %7153 = phi i32 [ 0, %.preheader1.2 ], [ %7162, %._crit_edge347.._crit_edge347_crit_edge ]
  %7154 = zext i32 %7153 to i64		; visa id: 9276
  %7155 = shl nuw nsw i64 %7154, 2		; visa id: 9277
  %7156 = add i64 %6430, %7155		; visa id: 9278
  %7157 = inttoptr i64 %7156 to i32*		; visa id: 9279
  %7158 = load i32, i32* %7157, align 4, !noalias !642		; visa id: 9279
  %7159 = add i64 %6426, %7155		; visa id: 9280
  %7160 = inttoptr i64 %7159 to i32*		; visa id: 9281
  store i32 %7158, i32* %7160, align 4, !alias.scope !642		; visa id: 9281
  %7161 = icmp eq i32 %7153, 0		; visa id: 9282
  br i1 %7161, label %._crit_edge347.._crit_edge347_crit_edge, label %7163, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9283

._crit_edge347.._crit_edge347_crit_edge:          ; preds = %._crit_edge347
; BB683 :
  %7162 = add nuw nsw i32 %7153, 1, !spirv.Decorations !631		; visa id: 9285
  br label %._crit_edge347, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9286

7163:                                             ; preds = %._crit_edge347
; BB684 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9288
  %7164 = load i64, i64* %6446, align 8		; visa id: 9288
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9289
  %7165 = icmp slt i32 %7151, %const_reg_dword1		; visa id: 9289
  %7166 = icmp slt i32 %65, %const_reg_dword
  %7167 = and i1 %7166, %7165		; visa id: 9290
  br i1 %7167, label %7168, label %.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9292

.._crit_edge70.378_crit_edge:                     ; preds = %7163
; BB:
  br label %._crit_edge70.378, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7168:                                             ; preds = %7163
; BB686 :
  %7169 = bitcast i64 %7164 to <2 x i32>		; visa id: 9294
  %7170 = extractelement <2 x i32> %7169, i32 0		; visa id: 9296
  %7171 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7170, i32 1
  %7172 = bitcast <2 x i32> %7171 to i64		; visa id: 9296
  %7173 = ashr exact i64 %7172, 32		; visa id: 9297
  %7174 = bitcast i64 %7173 to <2 x i32>		; visa id: 9298
  %7175 = extractelement <2 x i32> %7174, i32 0		; visa id: 9302
  %7176 = extractelement <2 x i32> %7174, i32 1		; visa id: 9302
  %7177 = ashr i64 %7164, 32		; visa id: 9302
  %7178 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7175, i32 %7176, i32 %50, i32 %51)
  %7179 = extractvalue { i32, i32 } %7178, 0		; visa id: 9303
  %7180 = extractvalue { i32, i32 } %7178, 1		; visa id: 9303
  %7181 = insertelement <2 x i32> undef, i32 %7179, i32 0		; visa id: 9310
  %7182 = insertelement <2 x i32> %7181, i32 %7180, i32 1		; visa id: 9311
  %7183 = bitcast <2 x i32> %7182 to i64		; visa id: 9312
  %7184 = add nsw i64 %7183, %7177, !spirv.Decorations !649		; visa id: 9316
  %7185 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !618		; visa id: 9317
  br i1 %86, label %7191, label %7186, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9318

7186:                                             ; preds = %7168
; BB687 :
  %7187 = shl i64 %7184, 2		; visa id: 9320
  %7188 = add i64 %.in, %7187		; visa id: 9321
  %7189 = inttoptr i64 %7188 to float addrspace(4)*		; visa id: 9322
  %7190 = addrspacecast float addrspace(4)* %7189 to float addrspace(1)*		; visa id: 9322
  store float %7185, float addrspace(1)* %7190, align 4		; visa id: 9323
  br label %._crit_edge70.378, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9324

7191:                                             ; preds = %7168
; BB688 :
  %7192 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7175, i32 %7176, i32 %47, i32 %48)
  %7193 = extractvalue { i32, i32 } %7192, 0		; visa id: 9326
  %7194 = extractvalue { i32, i32 } %7192, 1		; visa id: 9326
  %7195 = insertelement <2 x i32> undef, i32 %7193, i32 0		; visa id: 9333
  %7196 = insertelement <2 x i32> %7195, i32 %7194, i32 1		; visa id: 9334
  %7197 = bitcast <2 x i32> %7196 to i64		; visa id: 9335
  %7198 = shl i64 %7197, 2		; visa id: 9339
  %7199 = add i64 %.in399, %7198		; visa id: 9340
  %7200 = shl nsw i64 %7177, 2		; visa id: 9341
  %7201 = add i64 %7199, %7200		; visa id: 9342
  %7202 = inttoptr i64 %7201 to float addrspace(4)*		; visa id: 9343
  %7203 = addrspacecast float addrspace(4)* %7202 to float addrspace(1)*		; visa id: 9343
  %7204 = load float, float addrspace(1)* %7203, align 4		; visa id: 9344
  %7205 = fmul reassoc nsz arcp contract float %7204, %4, !spirv.Decorations !618		; visa id: 9345
  %7206 = fadd reassoc nsz arcp contract float %7185, %7205, !spirv.Decorations !618		; visa id: 9346
  %7207 = shl i64 %7184, 2		; visa id: 9347
  %7208 = add i64 %.in, %7207		; visa id: 9348
  %7209 = inttoptr i64 %7208 to float addrspace(4)*		; visa id: 9349
  %7210 = addrspacecast float addrspace(4)* %7209 to float addrspace(1)*		; visa id: 9349
  store float %7206, float addrspace(1)* %7210, align 4		; visa id: 9350
  br label %._crit_edge70.378, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9351

._crit_edge70.378:                                ; preds = %.._crit_edge70.378_crit_edge, %7191, %7186
; BB689 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9352
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9352
  %7211 = insertelement <2 x i32> %6495, i32 %7151, i64 1		; visa id: 9352
  store <2 x i32> %7211, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9355
  br label %._crit_edge348, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9357

._crit_edge348:                                   ; preds = %._crit_edge348.._crit_edge348_crit_edge, %._crit_edge70.378
; BB690 :
  %7212 = phi i32 [ 0, %._crit_edge70.378 ], [ %7221, %._crit_edge348.._crit_edge348_crit_edge ]
  %7213 = zext i32 %7212 to i64		; visa id: 9358
  %7214 = shl nuw nsw i64 %7213, 2		; visa id: 9359
  %7215 = add i64 %6430, %7214		; visa id: 9360
  %7216 = inttoptr i64 %7215 to i32*		; visa id: 9361
  %7217 = load i32, i32* %7216, align 4, !noalias !642		; visa id: 9361
  %7218 = add i64 %6426, %7214		; visa id: 9362
  %7219 = inttoptr i64 %7218 to i32*		; visa id: 9363
  store i32 %7217, i32* %7219, align 4, !alias.scope !642		; visa id: 9363
  %7220 = icmp eq i32 %7212, 0		; visa id: 9364
  br i1 %7220, label %._crit_edge348.._crit_edge348_crit_edge, label %7222, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9365

._crit_edge348.._crit_edge348_crit_edge:          ; preds = %._crit_edge348
; BB691 :
  %7221 = add nuw nsw i32 %7212, 1, !spirv.Decorations !631		; visa id: 9367
  br label %._crit_edge348, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9368

7222:                                             ; preds = %._crit_edge348
; BB692 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9370
  %7223 = load i64, i64* %6446, align 8		; visa id: 9370
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9371
  %7224 = icmp slt i32 %6494, %const_reg_dword
  %7225 = icmp slt i32 %7151, %const_reg_dword1		; visa id: 9371
  %7226 = and i1 %7224, %7225		; visa id: 9372
  br i1 %7226, label %7227, label %.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9374

.._crit_edge70.1.3_crit_edge:                     ; preds = %7222
; BB:
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7227:                                             ; preds = %7222
; BB694 :
  %7228 = bitcast i64 %7223 to <2 x i32>		; visa id: 9376
  %7229 = extractelement <2 x i32> %7228, i32 0		; visa id: 9378
  %7230 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7229, i32 1
  %7231 = bitcast <2 x i32> %7230 to i64		; visa id: 9378
  %7232 = ashr exact i64 %7231, 32		; visa id: 9379
  %7233 = bitcast i64 %7232 to <2 x i32>		; visa id: 9380
  %7234 = extractelement <2 x i32> %7233, i32 0		; visa id: 9384
  %7235 = extractelement <2 x i32> %7233, i32 1		; visa id: 9384
  %7236 = ashr i64 %7223, 32		; visa id: 9384
  %7237 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7234, i32 %7235, i32 %50, i32 %51)
  %7238 = extractvalue { i32, i32 } %7237, 0		; visa id: 9385
  %7239 = extractvalue { i32, i32 } %7237, 1		; visa id: 9385
  %7240 = insertelement <2 x i32> undef, i32 %7238, i32 0		; visa id: 9392
  %7241 = insertelement <2 x i32> %7240, i32 %7239, i32 1		; visa id: 9393
  %7242 = bitcast <2 x i32> %7241 to i64		; visa id: 9394
  %7243 = add nsw i64 %7242, %7236, !spirv.Decorations !649		; visa id: 9398
  %7244 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !618		; visa id: 9399
  br i1 %86, label %7250, label %7245, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9400

7245:                                             ; preds = %7227
; BB695 :
  %7246 = shl i64 %7243, 2		; visa id: 9402
  %7247 = add i64 %.in, %7246		; visa id: 9403
  %7248 = inttoptr i64 %7247 to float addrspace(4)*		; visa id: 9404
  %7249 = addrspacecast float addrspace(4)* %7248 to float addrspace(1)*		; visa id: 9404
  store float %7244, float addrspace(1)* %7249, align 4		; visa id: 9405
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9406

7250:                                             ; preds = %7227
; BB696 :
  %7251 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7234, i32 %7235, i32 %47, i32 %48)
  %7252 = extractvalue { i32, i32 } %7251, 0		; visa id: 9408
  %7253 = extractvalue { i32, i32 } %7251, 1		; visa id: 9408
  %7254 = insertelement <2 x i32> undef, i32 %7252, i32 0		; visa id: 9415
  %7255 = insertelement <2 x i32> %7254, i32 %7253, i32 1		; visa id: 9416
  %7256 = bitcast <2 x i32> %7255 to i64		; visa id: 9417
  %7257 = shl i64 %7256, 2		; visa id: 9421
  %7258 = add i64 %.in399, %7257		; visa id: 9422
  %7259 = shl nsw i64 %7236, 2		; visa id: 9423
  %7260 = add i64 %7258, %7259		; visa id: 9424
  %7261 = inttoptr i64 %7260 to float addrspace(4)*		; visa id: 9425
  %7262 = addrspacecast float addrspace(4)* %7261 to float addrspace(1)*		; visa id: 9425
  %7263 = load float, float addrspace(1)* %7262, align 4		; visa id: 9426
  %7264 = fmul reassoc nsz arcp contract float %7263, %4, !spirv.Decorations !618		; visa id: 9427
  %7265 = fadd reassoc nsz arcp contract float %7244, %7264, !spirv.Decorations !618		; visa id: 9428
  %7266 = shl i64 %7243, 2		; visa id: 9429
  %7267 = add i64 %.in, %7266		; visa id: 9430
  %7268 = inttoptr i64 %7267 to float addrspace(4)*		; visa id: 9431
  %7269 = addrspacecast float addrspace(4)* %7268 to float addrspace(1)*		; visa id: 9431
  store float %7265, float addrspace(1)* %7269, align 4		; visa id: 9432
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9433

._crit_edge70.1.3:                                ; preds = %.._crit_edge70.1.3_crit_edge, %7250, %7245
; BB697 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9434
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9434
  %7270 = insertelement <2 x i32> %6556, i32 %7151, i64 1		; visa id: 9434
  store <2 x i32> %7270, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9437
  br label %._crit_edge349, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9439

._crit_edge349:                                   ; preds = %._crit_edge349.._crit_edge349_crit_edge, %._crit_edge70.1.3
; BB698 :
  %7271 = phi i32 [ 0, %._crit_edge70.1.3 ], [ %7280, %._crit_edge349.._crit_edge349_crit_edge ]
  %7272 = zext i32 %7271 to i64		; visa id: 9440
  %7273 = shl nuw nsw i64 %7272, 2		; visa id: 9441
  %7274 = add i64 %6430, %7273		; visa id: 9442
  %7275 = inttoptr i64 %7274 to i32*		; visa id: 9443
  %7276 = load i32, i32* %7275, align 4, !noalias !642		; visa id: 9443
  %7277 = add i64 %6426, %7273		; visa id: 9444
  %7278 = inttoptr i64 %7277 to i32*		; visa id: 9445
  store i32 %7276, i32* %7278, align 4, !alias.scope !642		; visa id: 9445
  %7279 = icmp eq i32 %7271, 0		; visa id: 9446
  br i1 %7279, label %._crit_edge349.._crit_edge349_crit_edge, label %7281, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9447

._crit_edge349.._crit_edge349_crit_edge:          ; preds = %._crit_edge349
; BB699 :
  %7280 = add nuw nsw i32 %7271, 1, !spirv.Decorations !631		; visa id: 9449
  br label %._crit_edge349, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9450

7281:                                             ; preds = %._crit_edge349
; BB700 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9452
  %7282 = load i64, i64* %6446, align 8		; visa id: 9452
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9453
  %7283 = icmp slt i32 %6555, %const_reg_dword
  %7284 = icmp slt i32 %7151, %const_reg_dword1		; visa id: 9453
  %7285 = and i1 %7283, %7284		; visa id: 9454
  br i1 %7285, label %7286, label %.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9456

.._crit_edge70.2.3_crit_edge:                     ; preds = %7281
; BB:
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7286:                                             ; preds = %7281
; BB702 :
  %7287 = bitcast i64 %7282 to <2 x i32>		; visa id: 9458
  %7288 = extractelement <2 x i32> %7287, i32 0		; visa id: 9460
  %7289 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7288, i32 1
  %7290 = bitcast <2 x i32> %7289 to i64		; visa id: 9460
  %7291 = ashr exact i64 %7290, 32		; visa id: 9461
  %7292 = bitcast i64 %7291 to <2 x i32>		; visa id: 9462
  %7293 = extractelement <2 x i32> %7292, i32 0		; visa id: 9466
  %7294 = extractelement <2 x i32> %7292, i32 1		; visa id: 9466
  %7295 = ashr i64 %7282, 32		; visa id: 9466
  %7296 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7293, i32 %7294, i32 %50, i32 %51)
  %7297 = extractvalue { i32, i32 } %7296, 0		; visa id: 9467
  %7298 = extractvalue { i32, i32 } %7296, 1		; visa id: 9467
  %7299 = insertelement <2 x i32> undef, i32 %7297, i32 0		; visa id: 9474
  %7300 = insertelement <2 x i32> %7299, i32 %7298, i32 1		; visa id: 9475
  %7301 = bitcast <2 x i32> %7300 to i64		; visa id: 9476
  %7302 = add nsw i64 %7301, %7295, !spirv.Decorations !649		; visa id: 9480
  %7303 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !618		; visa id: 9481
  br i1 %86, label %7309, label %7304, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9482

7304:                                             ; preds = %7286
; BB703 :
  %7305 = shl i64 %7302, 2		; visa id: 9484
  %7306 = add i64 %.in, %7305		; visa id: 9485
  %7307 = inttoptr i64 %7306 to float addrspace(4)*		; visa id: 9486
  %7308 = addrspacecast float addrspace(4)* %7307 to float addrspace(1)*		; visa id: 9486
  store float %7303, float addrspace(1)* %7308, align 4		; visa id: 9487
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9488

7309:                                             ; preds = %7286
; BB704 :
  %7310 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7293, i32 %7294, i32 %47, i32 %48)
  %7311 = extractvalue { i32, i32 } %7310, 0		; visa id: 9490
  %7312 = extractvalue { i32, i32 } %7310, 1		; visa id: 9490
  %7313 = insertelement <2 x i32> undef, i32 %7311, i32 0		; visa id: 9497
  %7314 = insertelement <2 x i32> %7313, i32 %7312, i32 1		; visa id: 9498
  %7315 = bitcast <2 x i32> %7314 to i64		; visa id: 9499
  %7316 = shl i64 %7315, 2		; visa id: 9503
  %7317 = add i64 %.in399, %7316		; visa id: 9504
  %7318 = shl nsw i64 %7295, 2		; visa id: 9505
  %7319 = add i64 %7317, %7318		; visa id: 9506
  %7320 = inttoptr i64 %7319 to float addrspace(4)*		; visa id: 9507
  %7321 = addrspacecast float addrspace(4)* %7320 to float addrspace(1)*		; visa id: 9507
  %7322 = load float, float addrspace(1)* %7321, align 4		; visa id: 9508
  %7323 = fmul reassoc nsz arcp contract float %7322, %4, !spirv.Decorations !618		; visa id: 9509
  %7324 = fadd reassoc nsz arcp contract float %7303, %7323, !spirv.Decorations !618		; visa id: 9510
  %7325 = shl i64 %7302, 2		; visa id: 9511
  %7326 = add i64 %.in, %7325		; visa id: 9512
  %7327 = inttoptr i64 %7326 to float addrspace(4)*		; visa id: 9513
  %7328 = addrspacecast float addrspace(4)* %7327 to float addrspace(1)*		; visa id: 9513
  store float %7324, float addrspace(1)* %7328, align 4		; visa id: 9514
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9515

._crit_edge70.2.3:                                ; preds = %.._crit_edge70.2.3_crit_edge, %7309, %7304
; BB705 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9516
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9516
  %7329 = insertelement <2 x i32> %6617, i32 %7151, i64 1		; visa id: 9516
  store <2 x i32> %7329, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9519
  br label %._crit_edge350, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9521

._crit_edge350:                                   ; preds = %._crit_edge350.._crit_edge350_crit_edge, %._crit_edge70.2.3
; BB706 :
  %7330 = phi i32 [ 0, %._crit_edge70.2.3 ], [ %7339, %._crit_edge350.._crit_edge350_crit_edge ]
  %7331 = zext i32 %7330 to i64		; visa id: 9522
  %7332 = shl nuw nsw i64 %7331, 2		; visa id: 9523
  %7333 = add i64 %6430, %7332		; visa id: 9524
  %7334 = inttoptr i64 %7333 to i32*		; visa id: 9525
  %7335 = load i32, i32* %7334, align 4, !noalias !642		; visa id: 9525
  %7336 = add i64 %6426, %7332		; visa id: 9526
  %7337 = inttoptr i64 %7336 to i32*		; visa id: 9527
  store i32 %7335, i32* %7337, align 4, !alias.scope !642		; visa id: 9527
  %7338 = icmp eq i32 %7330, 0		; visa id: 9528
  br i1 %7338, label %._crit_edge350.._crit_edge350_crit_edge, label %7340, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9529

._crit_edge350.._crit_edge350_crit_edge:          ; preds = %._crit_edge350
; BB707 :
  %7339 = add nuw nsw i32 %7330, 1, !spirv.Decorations !631		; visa id: 9531
  br label %._crit_edge350, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9532

7340:                                             ; preds = %._crit_edge350
; BB708 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9534
  %7341 = load i64, i64* %6446, align 8		; visa id: 9534
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9535
  %7342 = icmp slt i32 %6616, %const_reg_dword
  %7343 = icmp slt i32 %7151, %const_reg_dword1		; visa id: 9535
  %7344 = and i1 %7342, %7343		; visa id: 9536
  br i1 %7344, label %7345, label %..preheader1.3_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9538

..preheader1.3_crit_edge:                         ; preds = %7340
; BB:
  br label %.preheader1.3, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7345:                                             ; preds = %7340
; BB710 :
  %7346 = bitcast i64 %7341 to <2 x i32>		; visa id: 9540
  %7347 = extractelement <2 x i32> %7346, i32 0		; visa id: 9542
  %7348 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7347, i32 1
  %7349 = bitcast <2 x i32> %7348 to i64		; visa id: 9542
  %7350 = ashr exact i64 %7349, 32		; visa id: 9543
  %7351 = bitcast i64 %7350 to <2 x i32>		; visa id: 9544
  %7352 = extractelement <2 x i32> %7351, i32 0		; visa id: 9548
  %7353 = extractelement <2 x i32> %7351, i32 1		; visa id: 9548
  %7354 = ashr i64 %7341, 32		; visa id: 9548
  %7355 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7352, i32 %7353, i32 %50, i32 %51)
  %7356 = extractvalue { i32, i32 } %7355, 0		; visa id: 9549
  %7357 = extractvalue { i32, i32 } %7355, 1		; visa id: 9549
  %7358 = insertelement <2 x i32> undef, i32 %7356, i32 0		; visa id: 9556
  %7359 = insertelement <2 x i32> %7358, i32 %7357, i32 1		; visa id: 9557
  %7360 = bitcast <2 x i32> %7359 to i64		; visa id: 9558
  %7361 = add nsw i64 %7360, %7354, !spirv.Decorations !649		; visa id: 9562
  %7362 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !618		; visa id: 9563
  br i1 %86, label %7368, label %7363, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9564

7363:                                             ; preds = %7345
; BB711 :
  %7364 = shl i64 %7361, 2		; visa id: 9566
  %7365 = add i64 %.in, %7364		; visa id: 9567
  %7366 = inttoptr i64 %7365 to float addrspace(4)*		; visa id: 9568
  %7367 = addrspacecast float addrspace(4)* %7366 to float addrspace(1)*		; visa id: 9568
  store float %7362, float addrspace(1)* %7367, align 4		; visa id: 9569
  br label %.preheader1.3, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9570

7368:                                             ; preds = %7345
; BB712 :
  %7369 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7352, i32 %7353, i32 %47, i32 %48)
  %7370 = extractvalue { i32, i32 } %7369, 0		; visa id: 9572
  %7371 = extractvalue { i32, i32 } %7369, 1		; visa id: 9572
  %7372 = insertelement <2 x i32> undef, i32 %7370, i32 0		; visa id: 9579
  %7373 = insertelement <2 x i32> %7372, i32 %7371, i32 1		; visa id: 9580
  %7374 = bitcast <2 x i32> %7373 to i64		; visa id: 9581
  %7375 = shl i64 %7374, 2		; visa id: 9585
  %7376 = add i64 %.in399, %7375		; visa id: 9586
  %7377 = shl nsw i64 %7354, 2		; visa id: 9587
  %7378 = add i64 %7376, %7377		; visa id: 9588
  %7379 = inttoptr i64 %7378 to float addrspace(4)*		; visa id: 9589
  %7380 = addrspacecast float addrspace(4)* %7379 to float addrspace(1)*		; visa id: 9589
  %7381 = load float, float addrspace(1)* %7380, align 4		; visa id: 9590
  %7382 = fmul reassoc nsz arcp contract float %7381, %4, !spirv.Decorations !618		; visa id: 9591
  %7383 = fadd reassoc nsz arcp contract float %7362, %7382, !spirv.Decorations !618		; visa id: 9592
  %7384 = shl i64 %7361, 2		; visa id: 9593
  %7385 = add i64 %.in, %7384		; visa id: 9594
  %7386 = inttoptr i64 %7385 to float addrspace(4)*		; visa id: 9595
  %7387 = addrspacecast float addrspace(4)* %7386 to float addrspace(1)*		; visa id: 9595
  store float %7383, float addrspace(1)* %7387, align 4		; visa id: 9596
  br label %.preheader1.3, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9597

.preheader1.3:                                    ; preds = %..preheader1.3_crit_edge, %7368, %7363
; BB713 :
  %7388 = add i32 %69, 4		; visa id: 9598
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9599
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9599
  %7389 = insertelement <2 x i32> %6432, i32 %7388, i64 1		; visa id: 9599
  store <2 x i32> %7389, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9602
  br label %._crit_edge351, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9604

._crit_edge351:                                   ; preds = %._crit_edge351.._crit_edge351_crit_edge, %.preheader1.3
; BB714 :
  %7390 = phi i32 [ 0, %.preheader1.3 ], [ %7399, %._crit_edge351.._crit_edge351_crit_edge ]
  %7391 = zext i32 %7390 to i64		; visa id: 9605
  %7392 = shl nuw nsw i64 %7391, 2		; visa id: 9606
  %7393 = add i64 %6430, %7392		; visa id: 9607
  %7394 = inttoptr i64 %7393 to i32*		; visa id: 9608
  %7395 = load i32, i32* %7394, align 4, !noalias !642		; visa id: 9608
  %7396 = add i64 %6426, %7392		; visa id: 9609
  %7397 = inttoptr i64 %7396 to i32*		; visa id: 9610
  store i32 %7395, i32* %7397, align 4, !alias.scope !642		; visa id: 9610
  %7398 = icmp eq i32 %7390, 0		; visa id: 9611
  br i1 %7398, label %._crit_edge351.._crit_edge351_crit_edge, label %7400, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9612

._crit_edge351.._crit_edge351_crit_edge:          ; preds = %._crit_edge351
; BB715 :
  %7399 = add nuw nsw i32 %7390, 1, !spirv.Decorations !631		; visa id: 9614
  br label %._crit_edge351, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9615

7400:                                             ; preds = %._crit_edge351
; BB716 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9617
  %7401 = load i64, i64* %6446, align 8		; visa id: 9617
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9618
  %7402 = icmp slt i32 %7388, %const_reg_dword1		; visa id: 9618
  %7403 = icmp slt i32 %65, %const_reg_dword
  %7404 = and i1 %7403, %7402		; visa id: 9619
  br i1 %7404, label %7405, label %.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9621

.._crit_edge70.4_crit_edge:                       ; preds = %7400
; BB:
  br label %._crit_edge70.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7405:                                             ; preds = %7400
; BB718 :
  %7406 = bitcast i64 %7401 to <2 x i32>		; visa id: 9623
  %7407 = extractelement <2 x i32> %7406, i32 0		; visa id: 9625
  %7408 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7407, i32 1
  %7409 = bitcast <2 x i32> %7408 to i64		; visa id: 9625
  %7410 = ashr exact i64 %7409, 32		; visa id: 9626
  %7411 = bitcast i64 %7410 to <2 x i32>		; visa id: 9627
  %7412 = extractelement <2 x i32> %7411, i32 0		; visa id: 9631
  %7413 = extractelement <2 x i32> %7411, i32 1		; visa id: 9631
  %7414 = ashr i64 %7401, 32		; visa id: 9631
  %7415 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7412, i32 %7413, i32 %50, i32 %51)
  %7416 = extractvalue { i32, i32 } %7415, 0		; visa id: 9632
  %7417 = extractvalue { i32, i32 } %7415, 1		; visa id: 9632
  %7418 = insertelement <2 x i32> undef, i32 %7416, i32 0		; visa id: 9639
  %7419 = insertelement <2 x i32> %7418, i32 %7417, i32 1		; visa id: 9640
  %7420 = bitcast <2 x i32> %7419 to i64		; visa id: 9641
  %7421 = add nsw i64 %7420, %7414, !spirv.Decorations !649		; visa id: 9645
  %7422 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !618		; visa id: 9646
  br i1 %86, label %7428, label %7423, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9647

7423:                                             ; preds = %7405
; BB719 :
  %7424 = shl i64 %7421, 2		; visa id: 9649
  %7425 = add i64 %.in, %7424		; visa id: 9650
  %7426 = inttoptr i64 %7425 to float addrspace(4)*		; visa id: 9651
  %7427 = addrspacecast float addrspace(4)* %7426 to float addrspace(1)*		; visa id: 9651
  store float %7422, float addrspace(1)* %7427, align 4		; visa id: 9652
  br label %._crit_edge70.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9653

7428:                                             ; preds = %7405
; BB720 :
  %7429 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7412, i32 %7413, i32 %47, i32 %48)
  %7430 = extractvalue { i32, i32 } %7429, 0		; visa id: 9655
  %7431 = extractvalue { i32, i32 } %7429, 1		; visa id: 9655
  %7432 = insertelement <2 x i32> undef, i32 %7430, i32 0		; visa id: 9662
  %7433 = insertelement <2 x i32> %7432, i32 %7431, i32 1		; visa id: 9663
  %7434 = bitcast <2 x i32> %7433 to i64		; visa id: 9664
  %7435 = shl i64 %7434, 2		; visa id: 9668
  %7436 = add i64 %.in399, %7435		; visa id: 9669
  %7437 = shl nsw i64 %7414, 2		; visa id: 9670
  %7438 = add i64 %7436, %7437		; visa id: 9671
  %7439 = inttoptr i64 %7438 to float addrspace(4)*		; visa id: 9672
  %7440 = addrspacecast float addrspace(4)* %7439 to float addrspace(1)*		; visa id: 9672
  %7441 = load float, float addrspace(1)* %7440, align 4		; visa id: 9673
  %7442 = fmul reassoc nsz arcp contract float %7441, %4, !spirv.Decorations !618		; visa id: 9674
  %7443 = fadd reassoc nsz arcp contract float %7422, %7442, !spirv.Decorations !618		; visa id: 9675
  %7444 = shl i64 %7421, 2		; visa id: 9676
  %7445 = add i64 %.in, %7444		; visa id: 9677
  %7446 = inttoptr i64 %7445 to float addrspace(4)*		; visa id: 9678
  %7447 = addrspacecast float addrspace(4)* %7446 to float addrspace(1)*		; visa id: 9678
  store float %7443, float addrspace(1)* %7447, align 4		; visa id: 9679
  br label %._crit_edge70.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9680

._crit_edge70.4:                                  ; preds = %.._crit_edge70.4_crit_edge, %7428, %7423
; BB721 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9681
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9681
  %7448 = insertelement <2 x i32> %6495, i32 %7388, i64 1		; visa id: 9681
  store <2 x i32> %7448, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9684
  br label %._crit_edge352, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9686

._crit_edge352:                                   ; preds = %._crit_edge352.._crit_edge352_crit_edge, %._crit_edge70.4
; BB722 :
  %7449 = phi i32 [ 0, %._crit_edge70.4 ], [ %7458, %._crit_edge352.._crit_edge352_crit_edge ]
  %7450 = zext i32 %7449 to i64		; visa id: 9687
  %7451 = shl nuw nsw i64 %7450, 2		; visa id: 9688
  %7452 = add i64 %6430, %7451		; visa id: 9689
  %7453 = inttoptr i64 %7452 to i32*		; visa id: 9690
  %7454 = load i32, i32* %7453, align 4, !noalias !642		; visa id: 9690
  %7455 = add i64 %6426, %7451		; visa id: 9691
  %7456 = inttoptr i64 %7455 to i32*		; visa id: 9692
  store i32 %7454, i32* %7456, align 4, !alias.scope !642		; visa id: 9692
  %7457 = icmp eq i32 %7449, 0		; visa id: 9693
  br i1 %7457, label %._crit_edge352.._crit_edge352_crit_edge, label %7459, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9694

._crit_edge352.._crit_edge352_crit_edge:          ; preds = %._crit_edge352
; BB723 :
  %7458 = add nuw nsw i32 %7449, 1, !spirv.Decorations !631		; visa id: 9696
  br label %._crit_edge352, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9697

7459:                                             ; preds = %._crit_edge352
; BB724 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9699
  %7460 = load i64, i64* %6446, align 8		; visa id: 9699
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9700
  %7461 = icmp slt i32 %6494, %const_reg_dword
  %7462 = icmp slt i32 %7388, %const_reg_dword1		; visa id: 9700
  %7463 = and i1 %7461, %7462		; visa id: 9701
  br i1 %7463, label %7464, label %.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9703

.._crit_edge70.1.4_crit_edge:                     ; preds = %7459
; BB:
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7464:                                             ; preds = %7459
; BB726 :
  %7465 = bitcast i64 %7460 to <2 x i32>		; visa id: 9705
  %7466 = extractelement <2 x i32> %7465, i32 0		; visa id: 9707
  %7467 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7466, i32 1
  %7468 = bitcast <2 x i32> %7467 to i64		; visa id: 9707
  %7469 = ashr exact i64 %7468, 32		; visa id: 9708
  %7470 = bitcast i64 %7469 to <2 x i32>		; visa id: 9709
  %7471 = extractelement <2 x i32> %7470, i32 0		; visa id: 9713
  %7472 = extractelement <2 x i32> %7470, i32 1		; visa id: 9713
  %7473 = ashr i64 %7460, 32		; visa id: 9713
  %7474 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7471, i32 %7472, i32 %50, i32 %51)
  %7475 = extractvalue { i32, i32 } %7474, 0		; visa id: 9714
  %7476 = extractvalue { i32, i32 } %7474, 1		; visa id: 9714
  %7477 = insertelement <2 x i32> undef, i32 %7475, i32 0		; visa id: 9721
  %7478 = insertelement <2 x i32> %7477, i32 %7476, i32 1		; visa id: 9722
  %7479 = bitcast <2 x i32> %7478 to i64		; visa id: 9723
  %7480 = add nsw i64 %7479, %7473, !spirv.Decorations !649		; visa id: 9727
  %7481 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !618		; visa id: 9728
  br i1 %86, label %7487, label %7482, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9729

7482:                                             ; preds = %7464
; BB727 :
  %7483 = shl i64 %7480, 2		; visa id: 9731
  %7484 = add i64 %.in, %7483		; visa id: 9732
  %7485 = inttoptr i64 %7484 to float addrspace(4)*		; visa id: 9733
  %7486 = addrspacecast float addrspace(4)* %7485 to float addrspace(1)*		; visa id: 9733
  store float %7481, float addrspace(1)* %7486, align 4		; visa id: 9734
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9735

7487:                                             ; preds = %7464
; BB728 :
  %7488 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7471, i32 %7472, i32 %47, i32 %48)
  %7489 = extractvalue { i32, i32 } %7488, 0		; visa id: 9737
  %7490 = extractvalue { i32, i32 } %7488, 1		; visa id: 9737
  %7491 = insertelement <2 x i32> undef, i32 %7489, i32 0		; visa id: 9744
  %7492 = insertelement <2 x i32> %7491, i32 %7490, i32 1		; visa id: 9745
  %7493 = bitcast <2 x i32> %7492 to i64		; visa id: 9746
  %7494 = shl i64 %7493, 2		; visa id: 9750
  %7495 = add i64 %.in399, %7494		; visa id: 9751
  %7496 = shl nsw i64 %7473, 2		; visa id: 9752
  %7497 = add i64 %7495, %7496		; visa id: 9753
  %7498 = inttoptr i64 %7497 to float addrspace(4)*		; visa id: 9754
  %7499 = addrspacecast float addrspace(4)* %7498 to float addrspace(1)*		; visa id: 9754
  %7500 = load float, float addrspace(1)* %7499, align 4		; visa id: 9755
  %7501 = fmul reassoc nsz arcp contract float %7500, %4, !spirv.Decorations !618		; visa id: 9756
  %7502 = fadd reassoc nsz arcp contract float %7481, %7501, !spirv.Decorations !618		; visa id: 9757
  %7503 = shl i64 %7480, 2		; visa id: 9758
  %7504 = add i64 %.in, %7503		; visa id: 9759
  %7505 = inttoptr i64 %7504 to float addrspace(4)*		; visa id: 9760
  %7506 = addrspacecast float addrspace(4)* %7505 to float addrspace(1)*		; visa id: 9760
  store float %7502, float addrspace(1)* %7506, align 4		; visa id: 9761
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9762

._crit_edge70.1.4:                                ; preds = %.._crit_edge70.1.4_crit_edge, %7487, %7482
; BB729 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9763
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9763
  %7507 = insertelement <2 x i32> %6556, i32 %7388, i64 1		; visa id: 9763
  store <2 x i32> %7507, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9766
  br label %._crit_edge353, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9768

._crit_edge353:                                   ; preds = %._crit_edge353.._crit_edge353_crit_edge, %._crit_edge70.1.4
; BB730 :
  %7508 = phi i32 [ 0, %._crit_edge70.1.4 ], [ %7517, %._crit_edge353.._crit_edge353_crit_edge ]
  %7509 = zext i32 %7508 to i64		; visa id: 9769
  %7510 = shl nuw nsw i64 %7509, 2		; visa id: 9770
  %7511 = add i64 %6430, %7510		; visa id: 9771
  %7512 = inttoptr i64 %7511 to i32*		; visa id: 9772
  %7513 = load i32, i32* %7512, align 4, !noalias !642		; visa id: 9772
  %7514 = add i64 %6426, %7510		; visa id: 9773
  %7515 = inttoptr i64 %7514 to i32*		; visa id: 9774
  store i32 %7513, i32* %7515, align 4, !alias.scope !642		; visa id: 9774
  %7516 = icmp eq i32 %7508, 0		; visa id: 9775
  br i1 %7516, label %._crit_edge353.._crit_edge353_crit_edge, label %7518, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9776

._crit_edge353.._crit_edge353_crit_edge:          ; preds = %._crit_edge353
; BB731 :
  %7517 = add nuw nsw i32 %7508, 1, !spirv.Decorations !631		; visa id: 9778
  br label %._crit_edge353, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9779

7518:                                             ; preds = %._crit_edge353
; BB732 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9781
  %7519 = load i64, i64* %6446, align 8		; visa id: 9781
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9782
  %7520 = icmp slt i32 %6555, %const_reg_dword
  %7521 = icmp slt i32 %7388, %const_reg_dword1		; visa id: 9782
  %7522 = and i1 %7520, %7521		; visa id: 9783
  br i1 %7522, label %7523, label %.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9785

.._crit_edge70.2.4_crit_edge:                     ; preds = %7518
; BB:
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7523:                                             ; preds = %7518
; BB734 :
  %7524 = bitcast i64 %7519 to <2 x i32>		; visa id: 9787
  %7525 = extractelement <2 x i32> %7524, i32 0		; visa id: 9789
  %7526 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7525, i32 1
  %7527 = bitcast <2 x i32> %7526 to i64		; visa id: 9789
  %7528 = ashr exact i64 %7527, 32		; visa id: 9790
  %7529 = bitcast i64 %7528 to <2 x i32>		; visa id: 9791
  %7530 = extractelement <2 x i32> %7529, i32 0		; visa id: 9795
  %7531 = extractelement <2 x i32> %7529, i32 1		; visa id: 9795
  %7532 = ashr i64 %7519, 32		; visa id: 9795
  %7533 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7530, i32 %7531, i32 %50, i32 %51)
  %7534 = extractvalue { i32, i32 } %7533, 0		; visa id: 9796
  %7535 = extractvalue { i32, i32 } %7533, 1		; visa id: 9796
  %7536 = insertelement <2 x i32> undef, i32 %7534, i32 0		; visa id: 9803
  %7537 = insertelement <2 x i32> %7536, i32 %7535, i32 1		; visa id: 9804
  %7538 = bitcast <2 x i32> %7537 to i64		; visa id: 9805
  %7539 = add nsw i64 %7538, %7532, !spirv.Decorations !649		; visa id: 9809
  %7540 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !618		; visa id: 9810
  br i1 %86, label %7546, label %7541, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9811

7541:                                             ; preds = %7523
; BB735 :
  %7542 = shl i64 %7539, 2		; visa id: 9813
  %7543 = add i64 %.in, %7542		; visa id: 9814
  %7544 = inttoptr i64 %7543 to float addrspace(4)*		; visa id: 9815
  %7545 = addrspacecast float addrspace(4)* %7544 to float addrspace(1)*		; visa id: 9815
  store float %7540, float addrspace(1)* %7545, align 4		; visa id: 9816
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9817

7546:                                             ; preds = %7523
; BB736 :
  %7547 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7530, i32 %7531, i32 %47, i32 %48)
  %7548 = extractvalue { i32, i32 } %7547, 0		; visa id: 9819
  %7549 = extractvalue { i32, i32 } %7547, 1		; visa id: 9819
  %7550 = insertelement <2 x i32> undef, i32 %7548, i32 0		; visa id: 9826
  %7551 = insertelement <2 x i32> %7550, i32 %7549, i32 1		; visa id: 9827
  %7552 = bitcast <2 x i32> %7551 to i64		; visa id: 9828
  %7553 = shl i64 %7552, 2		; visa id: 9832
  %7554 = add i64 %.in399, %7553		; visa id: 9833
  %7555 = shl nsw i64 %7532, 2		; visa id: 9834
  %7556 = add i64 %7554, %7555		; visa id: 9835
  %7557 = inttoptr i64 %7556 to float addrspace(4)*		; visa id: 9836
  %7558 = addrspacecast float addrspace(4)* %7557 to float addrspace(1)*		; visa id: 9836
  %7559 = load float, float addrspace(1)* %7558, align 4		; visa id: 9837
  %7560 = fmul reassoc nsz arcp contract float %7559, %4, !spirv.Decorations !618		; visa id: 9838
  %7561 = fadd reassoc nsz arcp contract float %7540, %7560, !spirv.Decorations !618		; visa id: 9839
  %7562 = shl i64 %7539, 2		; visa id: 9840
  %7563 = add i64 %.in, %7562		; visa id: 9841
  %7564 = inttoptr i64 %7563 to float addrspace(4)*		; visa id: 9842
  %7565 = addrspacecast float addrspace(4)* %7564 to float addrspace(1)*		; visa id: 9842
  store float %7561, float addrspace(1)* %7565, align 4		; visa id: 9843
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9844

._crit_edge70.2.4:                                ; preds = %.._crit_edge70.2.4_crit_edge, %7546, %7541
; BB737 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9845
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9845
  %7566 = insertelement <2 x i32> %6617, i32 %7388, i64 1		; visa id: 9845
  store <2 x i32> %7566, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9848
  br label %._crit_edge354, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9850

._crit_edge354:                                   ; preds = %._crit_edge354.._crit_edge354_crit_edge, %._crit_edge70.2.4
; BB738 :
  %7567 = phi i32 [ 0, %._crit_edge70.2.4 ], [ %7576, %._crit_edge354.._crit_edge354_crit_edge ]
  %7568 = zext i32 %7567 to i64		; visa id: 9851
  %7569 = shl nuw nsw i64 %7568, 2		; visa id: 9852
  %7570 = add i64 %6430, %7569		; visa id: 9853
  %7571 = inttoptr i64 %7570 to i32*		; visa id: 9854
  %7572 = load i32, i32* %7571, align 4, !noalias !642		; visa id: 9854
  %7573 = add i64 %6426, %7569		; visa id: 9855
  %7574 = inttoptr i64 %7573 to i32*		; visa id: 9856
  store i32 %7572, i32* %7574, align 4, !alias.scope !642		; visa id: 9856
  %7575 = icmp eq i32 %7567, 0		; visa id: 9857
  br i1 %7575, label %._crit_edge354.._crit_edge354_crit_edge, label %7577, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9858

._crit_edge354.._crit_edge354_crit_edge:          ; preds = %._crit_edge354
; BB739 :
  %7576 = add nuw nsw i32 %7567, 1, !spirv.Decorations !631		; visa id: 9860
  br label %._crit_edge354, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9861

7577:                                             ; preds = %._crit_edge354
; BB740 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9863
  %7578 = load i64, i64* %6446, align 8		; visa id: 9863
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9864
  %7579 = icmp slt i32 %6616, %const_reg_dword
  %7580 = icmp slt i32 %7388, %const_reg_dword1		; visa id: 9864
  %7581 = and i1 %7579, %7580		; visa id: 9865
  br i1 %7581, label %7582, label %..preheader1.4_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9867

..preheader1.4_crit_edge:                         ; preds = %7577
; BB:
  br label %.preheader1.4, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7582:                                             ; preds = %7577
; BB742 :
  %7583 = bitcast i64 %7578 to <2 x i32>		; visa id: 9869
  %7584 = extractelement <2 x i32> %7583, i32 0		; visa id: 9871
  %7585 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7584, i32 1
  %7586 = bitcast <2 x i32> %7585 to i64		; visa id: 9871
  %7587 = ashr exact i64 %7586, 32		; visa id: 9872
  %7588 = bitcast i64 %7587 to <2 x i32>		; visa id: 9873
  %7589 = extractelement <2 x i32> %7588, i32 0		; visa id: 9877
  %7590 = extractelement <2 x i32> %7588, i32 1		; visa id: 9877
  %7591 = ashr i64 %7578, 32		; visa id: 9877
  %7592 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7589, i32 %7590, i32 %50, i32 %51)
  %7593 = extractvalue { i32, i32 } %7592, 0		; visa id: 9878
  %7594 = extractvalue { i32, i32 } %7592, 1		; visa id: 9878
  %7595 = insertelement <2 x i32> undef, i32 %7593, i32 0		; visa id: 9885
  %7596 = insertelement <2 x i32> %7595, i32 %7594, i32 1		; visa id: 9886
  %7597 = bitcast <2 x i32> %7596 to i64		; visa id: 9887
  %7598 = add nsw i64 %7597, %7591, !spirv.Decorations !649		; visa id: 9891
  %7599 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !618		; visa id: 9892
  br i1 %86, label %7605, label %7600, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9893

7600:                                             ; preds = %7582
; BB743 :
  %7601 = shl i64 %7598, 2		; visa id: 9895
  %7602 = add i64 %.in, %7601		; visa id: 9896
  %7603 = inttoptr i64 %7602 to float addrspace(4)*		; visa id: 9897
  %7604 = addrspacecast float addrspace(4)* %7603 to float addrspace(1)*		; visa id: 9897
  store float %7599, float addrspace(1)* %7604, align 4		; visa id: 9898
  br label %.preheader1.4, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9899

7605:                                             ; preds = %7582
; BB744 :
  %7606 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7589, i32 %7590, i32 %47, i32 %48)
  %7607 = extractvalue { i32, i32 } %7606, 0		; visa id: 9901
  %7608 = extractvalue { i32, i32 } %7606, 1		; visa id: 9901
  %7609 = insertelement <2 x i32> undef, i32 %7607, i32 0		; visa id: 9908
  %7610 = insertelement <2 x i32> %7609, i32 %7608, i32 1		; visa id: 9909
  %7611 = bitcast <2 x i32> %7610 to i64		; visa id: 9910
  %7612 = shl i64 %7611, 2		; visa id: 9914
  %7613 = add i64 %.in399, %7612		; visa id: 9915
  %7614 = shl nsw i64 %7591, 2		; visa id: 9916
  %7615 = add i64 %7613, %7614		; visa id: 9917
  %7616 = inttoptr i64 %7615 to float addrspace(4)*		; visa id: 9918
  %7617 = addrspacecast float addrspace(4)* %7616 to float addrspace(1)*		; visa id: 9918
  %7618 = load float, float addrspace(1)* %7617, align 4		; visa id: 9919
  %7619 = fmul reassoc nsz arcp contract float %7618, %4, !spirv.Decorations !618		; visa id: 9920
  %7620 = fadd reassoc nsz arcp contract float %7599, %7619, !spirv.Decorations !618		; visa id: 9921
  %7621 = shl i64 %7598, 2		; visa id: 9922
  %7622 = add i64 %.in, %7621		; visa id: 9923
  %7623 = inttoptr i64 %7622 to float addrspace(4)*		; visa id: 9924
  %7624 = addrspacecast float addrspace(4)* %7623 to float addrspace(1)*		; visa id: 9924
  store float %7620, float addrspace(1)* %7624, align 4		; visa id: 9925
  br label %.preheader1.4, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 9926

.preheader1.4:                                    ; preds = %..preheader1.4_crit_edge, %7605, %7600
; BB745 :
  %7625 = add i32 %69, 5		; visa id: 9927
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9928
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9928
  %7626 = insertelement <2 x i32> %6432, i32 %7625, i64 1		; visa id: 9928
  store <2 x i32> %7626, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 9931
  br label %._crit_edge355, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9933

._crit_edge355:                                   ; preds = %._crit_edge355.._crit_edge355_crit_edge, %.preheader1.4
; BB746 :
  %7627 = phi i32 [ 0, %.preheader1.4 ], [ %7636, %._crit_edge355.._crit_edge355_crit_edge ]
  %7628 = zext i32 %7627 to i64		; visa id: 9934
  %7629 = shl nuw nsw i64 %7628, 2		; visa id: 9935
  %7630 = add i64 %6430, %7629		; visa id: 9936
  %7631 = inttoptr i64 %7630 to i32*		; visa id: 9937
  %7632 = load i32, i32* %7631, align 4, !noalias !642		; visa id: 9937
  %7633 = add i64 %6426, %7629		; visa id: 9938
  %7634 = inttoptr i64 %7633 to i32*		; visa id: 9939
  store i32 %7632, i32* %7634, align 4, !alias.scope !642		; visa id: 9939
  %7635 = icmp eq i32 %7627, 0		; visa id: 9940
  br i1 %7635, label %._crit_edge355.._crit_edge355_crit_edge, label %7637, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 9941

._crit_edge355.._crit_edge355_crit_edge:          ; preds = %._crit_edge355
; BB747 :
  %7636 = add nuw nsw i32 %7627, 1, !spirv.Decorations !631		; visa id: 9943
  br label %._crit_edge355, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 9944

7637:                                             ; preds = %._crit_edge355
; BB748 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 9946
  %7638 = load i64, i64* %6446, align 8		; visa id: 9946
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 9947
  %7639 = icmp slt i32 %7625, %const_reg_dword1		; visa id: 9947
  %7640 = icmp slt i32 %65, %const_reg_dword
  %7641 = and i1 %7640, %7639		; visa id: 9948
  br i1 %7641, label %7642, label %.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 9950

.._crit_edge70.5_crit_edge:                       ; preds = %7637
; BB:
  br label %._crit_edge70.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7642:                                             ; preds = %7637
; BB750 :
  %7643 = bitcast i64 %7638 to <2 x i32>		; visa id: 9952
  %7644 = extractelement <2 x i32> %7643, i32 0		; visa id: 9954
  %7645 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7644, i32 1
  %7646 = bitcast <2 x i32> %7645 to i64		; visa id: 9954
  %7647 = ashr exact i64 %7646, 32		; visa id: 9955
  %7648 = bitcast i64 %7647 to <2 x i32>		; visa id: 9956
  %7649 = extractelement <2 x i32> %7648, i32 0		; visa id: 9960
  %7650 = extractelement <2 x i32> %7648, i32 1		; visa id: 9960
  %7651 = ashr i64 %7638, 32		; visa id: 9960
  %7652 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7649, i32 %7650, i32 %50, i32 %51)
  %7653 = extractvalue { i32, i32 } %7652, 0		; visa id: 9961
  %7654 = extractvalue { i32, i32 } %7652, 1		; visa id: 9961
  %7655 = insertelement <2 x i32> undef, i32 %7653, i32 0		; visa id: 9968
  %7656 = insertelement <2 x i32> %7655, i32 %7654, i32 1		; visa id: 9969
  %7657 = bitcast <2 x i32> %7656 to i64		; visa id: 9970
  %7658 = add nsw i64 %7657, %7651, !spirv.Decorations !649		; visa id: 9974
  %7659 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !618		; visa id: 9975
  br i1 %86, label %7665, label %7660, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 9976

7660:                                             ; preds = %7642
; BB751 :
  %7661 = shl i64 %7658, 2		; visa id: 9978
  %7662 = add i64 %.in, %7661		; visa id: 9979
  %7663 = inttoptr i64 %7662 to float addrspace(4)*		; visa id: 9980
  %7664 = addrspacecast float addrspace(4)* %7663 to float addrspace(1)*		; visa id: 9980
  store float %7659, float addrspace(1)* %7664, align 4		; visa id: 9981
  br label %._crit_edge70.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 9982

7665:                                             ; preds = %7642
; BB752 :
  %7666 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7649, i32 %7650, i32 %47, i32 %48)
  %7667 = extractvalue { i32, i32 } %7666, 0		; visa id: 9984
  %7668 = extractvalue { i32, i32 } %7666, 1		; visa id: 9984
  %7669 = insertelement <2 x i32> undef, i32 %7667, i32 0		; visa id: 9991
  %7670 = insertelement <2 x i32> %7669, i32 %7668, i32 1		; visa id: 9992
  %7671 = bitcast <2 x i32> %7670 to i64		; visa id: 9993
  %7672 = shl i64 %7671, 2		; visa id: 9997
  %7673 = add i64 %.in399, %7672		; visa id: 9998
  %7674 = shl nsw i64 %7651, 2		; visa id: 9999
  %7675 = add i64 %7673, %7674		; visa id: 10000
  %7676 = inttoptr i64 %7675 to float addrspace(4)*		; visa id: 10001
  %7677 = addrspacecast float addrspace(4)* %7676 to float addrspace(1)*		; visa id: 10001
  %7678 = load float, float addrspace(1)* %7677, align 4		; visa id: 10002
  %7679 = fmul reassoc nsz arcp contract float %7678, %4, !spirv.Decorations !618		; visa id: 10003
  %7680 = fadd reassoc nsz arcp contract float %7659, %7679, !spirv.Decorations !618		; visa id: 10004
  %7681 = shl i64 %7658, 2		; visa id: 10005
  %7682 = add i64 %.in, %7681		; visa id: 10006
  %7683 = inttoptr i64 %7682 to float addrspace(4)*		; visa id: 10007
  %7684 = addrspacecast float addrspace(4)* %7683 to float addrspace(1)*		; visa id: 10007
  store float %7680, float addrspace(1)* %7684, align 4		; visa id: 10008
  br label %._crit_edge70.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10009

._crit_edge70.5:                                  ; preds = %.._crit_edge70.5_crit_edge, %7665, %7660
; BB753 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10010
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10010
  %7685 = insertelement <2 x i32> %6495, i32 %7625, i64 1		; visa id: 10010
  store <2 x i32> %7685, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10013
  br label %._crit_edge356, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10015

._crit_edge356:                                   ; preds = %._crit_edge356.._crit_edge356_crit_edge, %._crit_edge70.5
; BB754 :
  %7686 = phi i32 [ 0, %._crit_edge70.5 ], [ %7695, %._crit_edge356.._crit_edge356_crit_edge ]
  %7687 = zext i32 %7686 to i64		; visa id: 10016
  %7688 = shl nuw nsw i64 %7687, 2		; visa id: 10017
  %7689 = add i64 %6430, %7688		; visa id: 10018
  %7690 = inttoptr i64 %7689 to i32*		; visa id: 10019
  %7691 = load i32, i32* %7690, align 4, !noalias !642		; visa id: 10019
  %7692 = add i64 %6426, %7688		; visa id: 10020
  %7693 = inttoptr i64 %7692 to i32*		; visa id: 10021
  store i32 %7691, i32* %7693, align 4, !alias.scope !642		; visa id: 10021
  %7694 = icmp eq i32 %7686, 0		; visa id: 10022
  br i1 %7694, label %._crit_edge356.._crit_edge356_crit_edge, label %7696, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10023

._crit_edge356.._crit_edge356_crit_edge:          ; preds = %._crit_edge356
; BB755 :
  %7695 = add nuw nsw i32 %7686, 1, !spirv.Decorations !631		; visa id: 10025
  br label %._crit_edge356, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10026

7696:                                             ; preds = %._crit_edge356
; BB756 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10028
  %7697 = load i64, i64* %6446, align 8		; visa id: 10028
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10029
  %7698 = icmp slt i32 %6494, %const_reg_dword
  %7699 = icmp slt i32 %7625, %const_reg_dword1		; visa id: 10029
  %7700 = and i1 %7698, %7699		; visa id: 10030
  br i1 %7700, label %7701, label %.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10032

.._crit_edge70.1.5_crit_edge:                     ; preds = %7696
; BB:
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7701:                                             ; preds = %7696
; BB758 :
  %7702 = bitcast i64 %7697 to <2 x i32>		; visa id: 10034
  %7703 = extractelement <2 x i32> %7702, i32 0		; visa id: 10036
  %7704 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7703, i32 1
  %7705 = bitcast <2 x i32> %7704 to i64		; visa id: 10036
  %7706 = ashr exact i64 %7705, 32		; visa id: 10037
  %7707 = bitcast i64 %7706 to <2 x i32>		; visa id: 10038
  %7708 = extractelement <2 x i32> %7707, i32 0		; visa id: 10042
  %7709 = extractelement <2 x i32> %7707, i32 1		; visa id: 10042
  %7710 = ashr i64 %7697, 32		; visa id: 10042
  %7711 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7708, i32 %7709, i32 %50, i32 %51)
  %7712 = extractvalue { i32, i32 } %7711, 0		; visa id: 10043
  %7713 = extractvalue { i32, i32 } %7711, 1		; visa id: 10043
  %7714 = insertelement <2 x i32> undef, i32 %7712, i32 0		; visa id: 10050
  %7715 = insertelement <2 x i32> %7714, i32 %7713, i32 1		; visa id: 10051
  %7716 = bitcast <2 x i32> %7715 to i64		; visa id: 10052
  %7717 = add nsw i64 %7716, %7710, !spirv.Decorations !649		; visa id: 10056
  %7718 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !618		; visa id: 10057
  br i1 %86, label %7724, label %7719, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10058

7719:                                             ; preds = %7701
; BB759 :
  %7720 = shl i64 %7717, 2		; visa id: 10060
  %7721 = add i64 %.in, %7720		; visa id: 10061
  %7722 = inttoptr i64 %7721 to float addrspace(4)*		; visa id: 10062
  %7723 = addrspacecast float addrspace(4)* %7722 to float addrspace(1)*		; visa id: 10062
  store float %7718, float addrspace(1)* %7723, align 4		; visa id: 10063
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10064

7724:                                             ; preds = %7701
; BB760 :
  %7725 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7708, i32 %7709, i32 %47, i32 %48)
  %7726 = extractvalue { i32, i32 } %7725, 0		; visa id: 10066
  %7727 = extractvalue { i32, i32 } %7725, 1		; visa id: 10066
  %7728 = insertelement <2 x i32> undef, i32 %7726, i32 0		; visa id: 10073
  %7729 = insertelement <2 x i32> %7728, i32 %7727, i32 1		; visa id: 10074
  %7730 = bitcast <2 x i32> %7729 to i64		; visa id: 10075
  %7731 = shl i64 %7730, 2		; visa id: 10079
  %7732 = add i64 %.in399, %7731		; visa id: 10080
  %7733 = shl nsw i64 %7710, 2		; visa id: 10081
  %7734 = add i64 %7732, %7733		; visa id: 10082
  %7735 = inttoptr i64 %7734 to float addrspace(4)*		; visa id: 10083
  %7736 = addrspacecast float addrspace(4)* %7735 to float addrspace(1)*		; visa id: 10083
  %7737 = load float, float addrspace(1)* %7736, align 4		; visa id: 10084
  %7738 = fmul reassoc nsz arcp contract float %7737, %4, !spirv.Decorations !618		; visa id: 10085
  %7739 = fadd reassoc nsz arcp contract float %7718, %7738, !spirv.Decorations !618		; visa id: 10086
  %7740 = shl i64 %7717, 2		; visa id: 10087
  %7741 = add i64 %.in, %7740		; visa id: 10088
  %7742 = inttoptr i64 %7741 to float addrspace(4)*		; visa id: 10089
  %7743 = addrspacecast float addrspace(4)* %7742 to float addrspace(1)*		; visa id: 10089
  store float %7739, float addrspace(1)* %7743, align 4		; visa id: 10090
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10091

._crit_edge70.1.5:                                ; preds = %.._crit_edge70.1.5_crit_edge, %7724, %7719
; BB761 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10092
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10092
  %7744 = insertelement <2 x i32> %6556, i32 %7625, i64 1		; visa id: 10092
  store <2 x i32> %7744, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10095
  br label %._crit_edge357, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10097

._crit_edge357:                                   ; preds = %._crit_edge357.._crit_edge357_crit_edge, %._crit_edge70.1.5
; BB762 :
  %7745 = phi i32 [ 0, %._crit_edge70.1.5 ], [ %7754, %._crit_edge357.._crit_edge357_crit_edge ]
  %7746 = zext i32 %7745 to i64		; visa id: 10098
  %7747 = shl nuw nsw i64 %7746, 2		; visa id: 10099
  %7748 = add i64 %6430, %7747		; visa id: 10100
  %7749 = inttoptr i64 %7748 to i32*		; visa id: 10101
  %7750 = load i32, i32* %7749, align 4, !noalias !642		; visa id: 10101
  %7751 = add i64 %6426, %7747		; visa id: 10102
  %7752 = inttoptr i64 %7751 to i32*		; visa id: 10103
  store i32 %7750, i32* %7752, align 4, !alias.scope !642		; visa id: 10103
  %7753 = icmp eq i32 %7745, 0		; visa id: 10104
  br i1 %7753, label %._crit_edge357.._crit_edge357_crit_edge, label %7755, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10105

._crit_edge357.._crit_edge357_crit_edge:          ; preds = %._crit_edge357
; BB763 :
  %7754 = add nuw nsw i32 %7745, 1, !spirv.Decorations !631		; visa id: 10107
  br label %._crit_edge357, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10108

7755:                                             ; preds = %._crit_edge357
; BB764 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10110
  %7756 = load i64, i64* %6446, align 8		; visa id: 10110
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10111
  %7757 = icmp slt i32 %6555, %const_reg_dword
  %7758 = icmp slt i32 %7625, %const_reg_dword1		; visa id: 10111
  %7759 = and i1 %7757, %7758		; visa id: 10112
  br i1 %7759, label %7760, label %.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10114

.._crit_edge70.2.5_crit_edge:                     ; preds = %7755
; BB:
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7760:                                             ; preds = %7755
; BB766 :
  %7761 = bitcast i64 %7756 to <2 x i32>		; visa id: 10116
  %7762 = extractelement <2 x i32> %7761, i32 0		; visa id: 10118
  %7763 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7762, i32 1
  %7764 = bitcast <2 x i32> %7763 to i64		; visa id: 10118
  %7765 = ashr exact i64 %7764, 32		; visa id: 10119
  %7766 = bitcast i64 %7765 to <2 x i32>		; visa id: 10120
  %7767 = extractelement <2 x i32> %7766, i32 0		; visa id: 10124
  %7768 = extractelement <2 x i32> %7766, i32 1		; visa id: 10124
  %7769 = ashr i64 %7756, 32		; visa id: 10124
  %7770 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7767, i32 %7768, i32 %50, i32 %51)
  %7771 = extractvalue { i32, i32 } %7770, 0		; visa id: 10125
  %7772 = extractvalue { i32, i32 } %7770, 1		; visa id: 10125
  %7773 = insertelement <2 x i32> undef, i32 %7771, i32 0		; visa id: 10132
  %7774 = insertelement <2 x i32> %7773, i32 %7772, i32 1		; visa id: 10133
  %7775 = bitcast <2 x i32> %7774 to i64		; visa id: 10134
  %7776 = add nsw i64 %7775, %7769, !spirv.Decorations !649		; visa id: 10138
  %7777 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !618		; visa id: 10139
  br i1 %86, label %7783, label %7778, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10140

7778:                                             ; preds = %7760
; BB767 :
  %7779 = shl i64 %7776, 2		; visa id: 10142
  %7780 = add i64 %.in, %7779		; visa id: 10143
  %7781 = inttoptr i64 %7780 to float addrspace(4)*		; visa id: 10144
  %7782 = addrspacecast float addrspace(4)* %7781 to float addrspace(1)*		; visa id: 10144
  store float %7777, float addrspace(1)* %7782, align 4		; visa id: 10145
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10146

7783:                                             ; preds = %7760
; BB768 :
  %7784 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7767, i32 %7768, i32 %47, i32 %48)
  %7785 = extractvalue { i32, i32 } %7784, 0		; visa id: 10148
  %7786 = extractvalue { i32, i32 } %7784, 1		; visa id: 10148
  %7787 = insertelement <2 x i32> undef, i32 %7785, i32 0		; visa id: 10155
  %7788 = insertelement <2 x i32> %7787, i32 %7786, i32 1		; visa id: 10156
  %7789 = bitcast <2 x i32> %7788 to i64		; visa id: 10157
  %7790 = shl i64 %7789, 2		; visa id: 10161
  %7791 = add i64 %.in399, %7790		; visa id: 10162
  %7792 = shl nsw i64 %7769, 2		; visa id: 10163
  %7793 = add i64 %7791, %7792		; visa id: 10164
  %7794 = inttoptr i64 %7793 to float addrspace(4)*		; visa id: 10165
  %7795 = addrspacecast float addrspace(4)* %7794 to float addrspace(1)*		; visa id: 10165
  %7796 = load float, float addrspace(1)* %7795, align 4		; visa id: 10166
  %7797 = fmul reassoc nsz arcp contract float %7796, %4, !spirv.Decorations !618		; visa id: 10167
  %7798 = fadd reassoc nsz arcp contract float %7777, %7797, !spirv.Decorations !618		; visa id: 10168
  %7799 = shl i64 %7776, 2		; visa id: 10169
  %7800 = add i64 %.in, %7799		; visa id: 10170
  %7801 = inttoptr i64 %7800 to float addrspace(4)*		; visa id: 10171
  %7802 = addrspacecast float addrspace(4)* %7801 to float addrspace(1)*		; visa id: 10171
  store float %7798, float addrspace(1)* %7802, align 4		; visa id: 10172
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10173

._crit_edge70.2.5:                                ; preds = %.._crit_edge70.2.5_crit_edge, %7783, %7778
; BB769 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10174
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10174
  %7803 = insertelement <2 x i32> %6617, i32 %7625, i64 1		; visa id: 10174
  store <2 x i32> %7803, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10177
  br label %._crit_edge358, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10179

._crit_edge358:                                   ; preds = %._crit_edge358.._crit_edge358_crit_edge, %._crit_edge70.2.5
; BB770 :
  %7804 = phi i32 [ 0, %._crit_edge70.2.5 ], [ %7813, %._crit_edge358.._crit_edge358_crit_edge ]
  %7805 = zext i32 %7804 to i64		; visa id: 10180
  %7806 = shl nuw nsw i64 %7805, 2		; visa id: 10181
  %7807 = add i64 %6430, %7806		; visa id: 10182
  %7808 = inttoptr i64 %7807 to i32*		; visa id: 10183
  %7809 = load i32, i32* %7808, align 4, !noalias !642		; visa id: 10183
  %7810 = add i64 %6426, %7806		; visa id: 10184
  %7811 = inttoptr i64 %7810 to i32*		; visa id: 10185
  store i32 %7809, i32* %7811, align 4, !alias.scope !642		; visa id: 10185
  %7812 = icmp eq i32 %7804, 0		; visa id: 10186
  br i1 %7812, label %._crit_edge358.._crit_edge358_crit_edge, label %7814, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10187

._crit_edge358.._crit_edge358_crit_edge:          ; preds = %._crit_edge358
; BB771 :
  %7813 = add nuw nsw i32 %7804, 1, !spirv.Decorations !631		; visa id: 10189
  br label %._crit_edge358, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10190

7814:                                             ; preds = %._crit_edge358
; BB772 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10192
  %7815 = load i64, i64* %6446, align 8		; visa id: 10192
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10193
  %7816 = icmp slt i32 %6616, %const_reg_dword
  %7817 = icmp slt i32 %7625, %const_reg_dword1		; visa id: 10193
  %7818 = and i1 %7816, %7817		; visa id: 10194
  br i1 %7818, label %7819, label %..preheader1.5_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10196

..preheader1.5_crit_edge:                         ; preds = %7814
; BB:
  br label %.preheader1.5, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7819:                                             ; preds = %7814
; BB774 :
  %7820 = bitcast i64 %7815 to <2 x i32>		; visa id: 10198
  %7821 = extractelement <2 x i32> %7820, i32 0		; visa id: 10200
  %7822 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7821, i32 1
  %7823 = bitcast <2 x i32> %7822 to i64		; visa id: 10200
  %7824 = ashr exact i64 %7823, 32		; visa id: 10201
  %7825 = bitcast i64 %7824 to <2 x i32>		; visa id: 10202
  %7826 = extractelement <2 x i32> %7825, i32 0		; visa id: 10206
  %7827 = extractelement <2 x i32> %7825, i32 1		; visa id: 10206
  %7828 = ashr i64 %7815, 32		; visa id: 10206
  %7829 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7826, i32 %7827, i32 %50, i32 %51)
  %7830 = extractvalue { i32, i32 } %7829, 0		; visa id: 10207
  %7831 = extractvalue { i32, i32 } %7829, 1		; visa id: 10207
  %7832 = insertelement <2 x i32> undef, i32 %7830, i32 0		; visa id: 10214
  %7833 = insertelement <2 x i32> %7832, i32 %7831, i32 1		; visa id: 10215
  %7834 = bitcast <2 x i32> %7833 to i64		; visa id: 10216
  %7835 = add nsw i64 %7834, %7828, !spirv.Decorations !649		; visa id: 10220
  %7836 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !618		; visa id: 10221
  br i1 %86, label %7842, label %7837, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10222

7837:                                             ; preds = %7819
; BB775 :
  %7838 = shl i64 %7835, 2		; visa id: 10224
  %7839 = add i64 %.in, %7838		; visa id: 10225
  %7840 = inttoptr i64 %7839 to float addrspace(4)*		; visa id: 10226
  %7841 = addrspacecast float addrspace(4)* %7840 to float addrspace(1)*		; visa id: 10226
  store float %7836, float addrspace(1)* %7841, align 4		; visa id: 10227
  br label %.preheader1.5, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10228

7842:                                             ; preds = %7819
; BB776 :
  %7843 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7826, i32 %7827, i32 %47, i32 %48)
  %7844 = extractvalue { i32, i32 } %7843, 0		; visa id: 10230
  %7845 = extractvalue { i32, i32 } %7843, 1		; visa id: 10230
  %7846 = insertelement <2 x i32> undef, i32 %7844, i32 0		; visa id: 10237
  %7847 = insertelement <2 x i32> %7846, i32 %7845, i32 1		; visa id: 10238
  %7848 = bitcast <2 x i32> %7847 to i64		; visa id: 10239
  %7849 = shl i64 %7848, 2		; visa id: 10243
  %7850 = add i64 %.in399, %7849		; visa id: 10244
  %7851 = shl nsw i64 %7828, 2		; visa id: 10245
  %7852 = add i64 %7850, %7851		; visa id: 10246
  %7853 = inttoptr i64 %7852 to float addrspace(4)*		; visa id: 10247
  %7854 = addrspacecast float addrspace(4)* %7853 to float addrspace(1)*		; visa id: 10247
  %7855 = load float, float addrspace(1)* %7854, align 4		; visa id: 10248
  %7856 = fmul reassoc nsz arcp contract float %7855, %4, !spirv.Decorations !618		; visa id: 10249
  %7857 = fadd reassoc nsz arcp contract float %7836, %7856, !spirv.Decorations !618		; visa id: 10250
  %7858 = shl i64 %7835, 2		; visa id: 10251
  %7859 = add i64 %.in, %7858		; visa id: 10252
  %7860 = inttoptr i64 %7859 to float addrspace(4)*		; visa id: 10253
  %7861 = addrspacecast float addrspace(4)* %7860 to float addrspace(1)*		; visa id: 10253
  store float %7857, float addrspace(1)* %7861, align 4		; visa id: 10254
  br label %.preheader1.5, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10255

.preheader1.5:                                    ; preds = %..preheader1.5_crit_edge, %7842, %7837
; BB777 :
  %7862 = add i32 %69, 6		; visa id: 10256
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10257
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10257
  %7863 = insertelement <2 x i32> %6432, i32 %7862, i64 1		; visa id: 10257
  store <2 x i32> %7863, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10260
  br label %._crit_edge359, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10262

._crit_edge359:                                   ; preds = %._crit_edge359.._crit_edge359_crit_edge, %.preheader1.5
; BB778 :
  %7864 = phi i32 [ 0, %.preheader1.5 ], [ %7873, %._crit_edge359.._crit_edge359_crit_edge ]
  %7865 = zext i32 %7864 to i64		; visa id: 10263
  %7866 = shl nuw nsw i64 %7865, 2		; visa id: 10264
  %7867 = add i64 %6430, %7866		; visa id: 10265
  %7868 = inttoptr i64 %7867 to i32*		; visa id: 10266
  %7869 = load i32, i32* %7868, align 4, !noalias !642		; visa id: 10266
  %7870 = add i64 %6426, %7866		; visa id: 10267
  %7871 = inttoptr i64 %7870 to i32*		; visa id: 10268
  store i32 %7869, i32* %7871, align 4, !alias.scope !642		; visa id: 10268
  %7872 = icmp eq i32 %7864, 0		; visa id: 10269
  br i1 %7872, label %._crit_edge359.._crit_edge359_crit_edge, label %7874, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10270

._crit_edge359.._crit_edge359_crit_edge:          ; preds = %._crit_edge359
; BB779 :
  %7873 = add nuw nsw i32 %7864, 1, !spirv.Decorations !631		; visa id: 10272
  br label %._crit_edge359, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10273

7874:                                             ; preds = %._crit_edge359
; BB780 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10275
  %7875 = load i64, i64* %6446, align 8		; visa id: 10275
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10276
  %7876 = icmp slt i32 %7862, %const_reg_dword1		; visa id: 10276
  %7877 = icmp slt i32 %65, %const_reg_dword
  %7878 = and i1 %7877, %7876		; visa id: 10277
  br i1 %7878, label %7879, label %.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10279

.._crit_edge70.6_crit_edge:                       ; preds = %7874
; BB:
  br label %._crit_edge70.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7879:                                             ; preds = %7874
; BB782 :
  %7880 = bitcast i64 %7875 to <2 x i32>		; visa id: 10281
  %7881 = extractelement <2 x i32> %7880, i32 0		; visa id: 10283
  %7882 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7881, i32 1
  %7883 = bitcast <2 x i32> %7882 to i64		; visa id: 10283
  %7884 = ashr exact i64 %7883, 32		; visa id: 10284
  %7885 = bitcast i64 %7884 to <2 x i32>		; visa id: 10285
  %7886 = extractelement <2 x i32> %7885, i32 0		; visa id: 10289
  %7887 = extractelement <2 x i32> %7885, i32 1		; visa id: 10289
  %7888 = ashr i64 %7875, 32		; visa id: 10289
  %7889 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7886, i32 %7887, i32 %50, i32 %51)
  %7890 = extractvalue { i32, i32 } %7889, 0		; visa id: 10290
  %7891 = extractvalue { i32, i32 } %7889, 1		; visa id: 10290
  %7892 = insertelement <2 x i32> undef, i32 %7890, i32 0		; visa id: 10297
  %7893 = insertelement <2 x i32> %7892, i32 %7891, i32 1		; visa id: 10298
  %7894 = bitcast <2 x i32> %7893 to i64		; visa id: 10299
  %7895 = add nsw i64 %7894, %7888, !spirv.Decorations !649		; visa id: 10303
  %7896 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !618		; visa id: 10304
  br i1 %86, label %7902, label %7897, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10305

7897:                                             ; preds = %7879
; BB783 :
  %7898 = shl i64 %7895, 2		; visa id: 10307
  %7899 = add i64 %.in, %7898		; visa id: 10308
  %7900 = inttoptr i64 %7899 to float addrspace(4)*		; visa id: 10309
  %7901 = addrspacecast float addrspace(4)* %7900 to float addrspace(1)*		; visa id: 10309
  store float %7896, float addrspace(1)* %7901, align 4		; visa id: 10310
  br label %._crit_edge70.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10311

7902:                                             ; preds = %7879
; BB784 :
  %7903 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7886, i32 %7887, i32 %47, i32 %48)
  %7904 = extractvalue { i32, i32 } %7903, 0		; visa id: 10313
  %7905 = extractvalue { i32, i32 } %7903, 1		; visa id: 10313
  %7906 = insertelement <2 x i32> undef, i32 %7904, i32 0		; visa id: 10320
  %7907 = insertelement <2 x i32> %7906, i32 %7905, i32 1		; visa id: 10321
  %7908 = bitcast <2 x i32> %7907 to i64		; visa id: 10322
  %7909 = shl i64 %7908, 2		; visa id: 10326
  %7910 = add i64 %.in399, %7909		; visa id: 10327
  %7911 = shl nsw i64 %7888, 2		; visa id: 10328
  %7912 = add i64 %7910, %7911		; visa id: 10329
  %7913 = inttoptr i64 %7912 to float addrspace(4)*		; visa id: 10330
  %7914 = addrspacecast float addrspace(4)* %7913 to float addrspace(1)*		; visa id: 10330
  %7915 = load float, float addrspace(1)* %7914, align 4		; visa id: 10331
  %7916 = fmul reassoc nsz arcp contract float %7915, %4, !spirv.Decorations !618		; visa id: 10332
  %7917 = fadd reassoc nsz arcp contract float %7896, %7916, !spirv.Decorations !618		; visa id: 10333
  %7918 = shl i64 %7895, 2		; visa id: 10334
  %7919 = add i64 %.in, %7918		; visa id: 10335
  %7920 = inttoptr i64 %7919 to float addrspace(4)*		; visa id: 10336
  %7921 = addrspacecast float addrspace(4)* %7920 to float addrspace(1)*		; visa id: 10336
  store float %7917, float addrspace(1)* %7921, align 4		; visa id: 10337
  br label %._crit_edge70.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10338

._crit_edge70.6:                                  ; preds = %.._crit_edge70.6_crit_edge, %7902, %7897
; BB785 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10339
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10339
  %7922 = insertelement <2 x i32> %6495, i32 %7862, i64 1		; visa id: 10339
  store <2 x i32> %7922, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10342
  br label %._crit_edge360, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10344

._crit_edge360:                                   ; preds = %._crit_edge360.._crit_edge360_crit_edge, %._crit_edge70.6
; BB786 :
  %7923 = phi i32 [ 0, %._crit_edge70.6 ], [ %7932, %._crit_edge360.._crit_edge360_crit_edge ]
  %7924 = zext i32 %7923 to i64		; visa id: 10345
  %7925 = shl nuw nsw i64 %7924, 2		; visa id: 10346
  %7926 = add i64 %6430, %7925		; visa id: 10347
  %7927 = inttoptr i64 %7926 to i32*		; visa id: 10348
  %7928 = load i32, i32* %7927, align 4, !noalias !642		; visa id: 10348
  %7929 = add i64 %6426, %7925		; visa id: 10349
  %7930 = inttoptr i64 %7929 to i32*		; visa id: 10350
  store i32 %7928, i32* %7930, align 4, !alias.scope !642		; visa id: 10350
  %7931 = icmp eq i32 %7923, 0		; visa id: 10351
  br i1 %7931, label %._crit_edge360.._crit_edge360_crit_edge, label %7933, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10352

._crit_edge360.._crit_edge360_crit_edge:          ; preds = %._crit_edge360
; BB787 :
  %7932 = add nuw nsw i32 %7923, 1, !spirv.Decorations !631		; visa id: 10354
  br label %._crit_edge360, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10355

7933:                                             ; preds = %._crit_edge360
; BB788 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10357
  %7934 = load i64, i64* %6446, align 8		; visa id: 10357
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10358
  %7935 = icmp slt i32 %6494, %const_reg_dword
  %7936 = icmp slt i32 %7862, %const_reg_dword1		; visa id: 10358
  %7937 = and i1 %7935, %7936		; visa id: 10359
  br i1 %7937, label %7938, label %.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10361

.._crit_edge70.1.6_crit_edge:                     ; preds = %7933
; BB:
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7938:                                             ; preds = %7933
; BB790 :
  %7939 = bitcast i64 %7934 to <2 x i32>		; visa id: 10363
  %7940 = extractelement <2 x i32> %7939, i32 0		; visa id: 10365
  %7941 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7940, i32 1
  %7942 = bitcast <2 x i32> %7941 to i64		; visa id: 10365
  %7943 = ashr exact i64 %7942, 32		; visa id: 10366
  %7944 = bitcast i64 %7943 to <2 x i32>		; visa id: 10367
  %7945 = extractelement <2 x i32> %7944, i32 0		; visa id: 10371
  %7946 = extractelement <2 x i32> %7944, i32 1		; visa id: 10371
  %7947 = ashr i64 %7934, 32		; visa id: 10371
  %7948 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7945, i32 %7946, i32 %50, i32 %51)
  %7949 = extractvalue { i32, i32 } %7948, 0		; visa id: 10372
  %7950 = extractvalue { i32, i32 } %7948, 1		; visa id: 10372
  %7951 = insertelement <2 x i32> undef, i32 %7949, i32 0		; visa id: 10379
  %7952 = insertelement <2 x i32> %7951, i32 %7950, i32 1		; visa id: 10380
  %7953 = bitcast <2 x i32> %7952 to i64		; visa id: 10381
  %7954 = add nsw i64 %7953, %7947, !spirv.Decorations !649		; visa id: 10385
  %7955 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !618		; visa id: 10386
  br i1 %86, label %7961, label %7956, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10387

7956:                                             ; preds = %7938
; BB791 :
  %7957 = shl i64 %7954, 2		; visa id: 10389
  %7958 = add i64 %.in, %7957		; visa id: 10390
  %7959 = inttoptr i64 %7958 to float addrspace(4)*		; visa id: 10391
  %7960 = addrspacecast float addrspace(4)* %7959 to float addrspace(1)*		; visa id: 10391
  store float %7955, float addrspace(1)* %7960, align 4		; visa id: 10392
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10393

7961:                                             ; preds = %7938
; BB792 :
  %7962 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7945, i32 %7946, i32 %47, i32 %48)
  %7963 = extractvalue { i32, i32 } %7962, 0		; visa id: 10395
  %7964 = extractvalue { i32, i32 } %7962, 1		; visa id: 10395
  %7965 = insertelement <2 x i32> undef, i32 %7963, i32 0		; visa id: 10402
  %7966 = insertelement <2 x i32> %7965, i32 %7964, i32 1		; visa id: 10403
  %7967 = bitcast <2 x i32> %7966 to i64		; visa id: 10404
  %7968 = shl i64 %7967, 2		; visa id: 10408
  %7969 = add i64 %.in399, %7968		; visa id: 10409
  %7970 = shl nsw i64 %7947, 2		; visa id: 10410
  %7971 = add i64 %7969, %7970		; visa id: 10411
  %7972 = inttoptr i64 %7971 to float addrspace(4)*		; visa id: 10412
  %7973 = addrspacecast float addrspace(4)* %7972 to float addrspace(1)*		; visa id: 10412
  %7974 = load float, float addrspace(1)* %7973, align 4		; visa id: 10413
  %7975 = fmul reassoc nsz arcp contract float %7974, %4, !spirv.Decorations !618		; visa id: 10414
  %7976 = fadd reassoc nsz arcp contract float %7955, %7975, !spirv.Decorations !618		; visa id: 10415
  %7977 = shl i64 %7954, 2		; visa id: 10416
  %7978 = add i64 %.in, %7977		; visa id: 10417
  %7979 = inttoptr i64 %7978 to float addrspace(4)*		; visa id: 10418
  %7980 = addrspacecast float addrspace(4)* %7979 to float addrspace(1)*		; visa id: 10418
  store float %7976, float addrspace(1)* %7980, align 4		; visa id: 10419
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10420

._crit_edge70.1.6:                                ; preds = %.._crit_edge70.1.6_crit_edge, %7961, %7956
; BB793 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10421
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10421
  %7981 = insertelement <2 x i32> %6556, i32 %7862, i64 1		; visa id: 10421
  store <2 x i32> %7981, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10424
  br label %._crit_edge361, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10426

._crit_edge361:                                   ; preds = %._crit_edge361.._crit_edge361_crit_edge, %._crit_edge70.1.6
; BB794 :
  %7982 = phi i32 [ 0, %._crit_edge70.1.6 ], [ %7991, %._crit_edge361.._crit_edge361_crit_edge ]
  %7983 = zext i32 %7982 to i64		; visa id: 10427
  %7984 = shl nuw nsw i64 %7983, 2		; visa id: 10428
  %7985 = add i64 %6430, %7984		; visa id: 10429
  %7986 = inttoptr i64 %7985 to i32*		; visa id: 10430
  %7987 = load i32, i32* %7986, align 4, !noalias !642		; visa id: 10430
  %7988 = add i64 %6426, %7984		; visa id: 10431
  %7989 = inttoptr i64 %7988 to i32*		; visa id: 10432
  store i32 %7987, i32* %7989, align 4, !alias.scope !642		; visa id: 10432
  %7990 = icmp eq i32 %7982, 0		; visa id: 10433
  br i1 %7990, label %._crit_edge361.._crit_edge361_crit_edge, label %7992, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10434

._crit_edge361.._crit_edge361_crit_edge:          ; preds = %._crit_edge361
; BB795 :
  %7991 = add nuw nsw i32 %7982, 1, !spirv.Decorations !631		; visa id: 10436
  br label %._crit_edge361, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10437

7992:                                             ; preds = %._crit_edge361
; BB796 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10439
  %7993 = load i64, i64* %6446, align 8		; visa id: 10439
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10440
  %7994 = icmp slt i32 %6555, %const_reg_dword
  %7995 = icmp slt i32 %7862, %const_reg_dword1		; visa id: 10440
  %7996 = and i1 %7994, %7995		; visa id: 10441
  br i1 %7996, label %7997, label %.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10443

.._crit_edge70.2.6_crit_edge:                     ; preds = %7992
; BB:
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

7997:                                             ; preds = %7992
; BB798 :
  %7998 = bitcast i64 %7993 to <2 x i32>		; visa id: 10445
  %7999 = extractelement <2 x i32> %7998, i32 0		; visa id: 10447
  %8000 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %7999, i32 1
  %8001 = bitcast <2 x i32> %8000 to i64		; visa id: 10447
  %8002 = ashr exact i64 %8001, 32		; visa id: 10448
  %8003 = bitcast i64 %8002 to <2 x i32>		; visa id: 10449
  %8004 = extractelement <2 x i32> %8003, i32 0		; visa id: 10453
  %8005 = extractelement <2 x i32> %8003, i32 1		; visa id: 10453
  %8006 = ashr i64 %7993, 32		; visa id: 10453
  %8007 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8004, i32 %8005, i32 %50, i32 %51)
  %8008 = extractvalue { i32, i32 } %8007, 0		; visa id: 10454
  %8009 = extractvalue { i32, i32 } %8007, 1		; visa id: 10454
  %8010 = insertelement <2 x i32> undef, i32 %8008, i32 0		; visa id: 10461
  %8011 = insertelement <2 x i32> %8010, i32 %8009, i32 1		; visa id: 10462
  %8012 = bitcast <2 x i32> %8011 to i64		; visa id: 10463
  %8013 = add nsw i64 %8012, %8006, !spirv.Decorations !649		; visa id: 10467
  %8014 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !618		; visa id: 10468
  br i1 %86, label %8020, label %8015, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10469

8015:                                             ; preds = %7997
; BB799 :
  %8016 = shl i64 %8013, 2		; visa id: 10471
  %8017 = add i64 %.in, %8016		; visa id: 10472
  %8018 = inttoptr i64 %8017 to float addrspace(4)*		; visa id: 10473
  %8019 = addrspacecast float addrspace(4)* %8018 to float addrspace(1)*		; visa id: 10473
  store float %8014, float addrspace(1)* %8019, align 4		; visa id: 10474
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10475

8020:                                             ; preds = %7997
; BB800 :
  %8021 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8004, i32 %8005, i32 %47, i32 %48)
  %8022 = extractvalue { i32, i32 } %8021, 0		; visa id: 10477
  %8023 = extractvalue { i32, i32 } %8021, 1		; visa id: 10477
  %8024 = insertelement <2 x i32> undef, i32 %8022, i32 0		; visa id: 10484
  %8025 = insertelement <2 x i32> %8024, i32 %8023, i32 1		; visa id: 10485
  %8026 = bitcast <2 x i32> %8025 to i64		; visa id: 10486
  %8027 = shl i64 %8026, 2		; visa id: 10490
  %8028 = add i64 %.in399, %8027		; visa id: 10491
  %8029 = shl nsw i64 %8006, 2		; visa id: 10492
  %8030 = add i64 %8028, %8029		; visa id: 10493
  %8031 = inttoptr i64 %8030 to float addrspace(4)*		; visa id: 10494
  %8032 = addrspacecast float addrspace(4)* %8031 to float addrspace(1)*		; visa id: 10494
  %8033 = load float, float addrspace(1)* %8032, align 4		; visa id: 10495
  %8034 = fmul reassoc nsz arcp contract float %8033, %4, !spirv.Decorations !618		; visa id: 10496
  %8035 = fadd reassoc nsz arcp contract float %8014, %8034, !spirv.Decorations !618		; visa id: 10497
  %8036 = shl i64 %8013, 2		; visa id: 10498
  %8037 = add i64 %.in, %8036		; visa id: 10499
  %8038 = inttoptr i64 %8037 to float addrspace(4)*		; visa id: 10500
  %8039 = addrspacecast float addrspace(4)* %8038 to float addrspace(1)*		; visa id: 10500
  store float %8035, float addrspace(1)* %8039, align 4		; visa id: 10501
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10502

._crit_edge70.2.6:                                ; preds = %.._crit_edge70.2.6_crit_edge, %8020, %8015
; BB801 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10503
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10503
  %8040 = insertelement <2 x i32> %6617, i32 %7862, i64 1		; visa id: 10503
  store <2 x i32> %8040, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10506
  br label %._crit_edge362, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10508

._crit_edge362:                                   ; preds = %._crit_edge362.._crit_edge362_crit_edge, %._crit_edge70.2.6
; BB802 :
  %8041 = phi i32 [ 0, %._crit_edge70.2.6 ], [ %8050, %._crit_edge362.._crit_edge362_crit_edge ]
  %8042 = zext i32 %8041 to i64		; visa id: 10509
  %8043 = shl nuw nsw i64 %8042, 2		; visa id: 10510
  %8044 = add i64 %6430, %8043		; visa id: 10511
  %8045 = inttoptr i64 %8044 to i32*		; visa id: 10512
  %8046 = load i32, i32* %8045, align 4, !noalias !642		; visa id: 10512
  %8047 = add i64 %6426, %8043		; visa id: 10513
  %8048 = inttoptr i64 %8047 to i32*		; visa id: 10514
  store i32 %8046, i32* %8048, align 4, !alias.scope !642		; visa id: 10514
  %8049 = icmp eq i32 %8041, 0		; visa id: 10515
  br i1 %8049, label %._crit_edge362.._crit_edge362_crit_edge, label %8051, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10516

._crit_edge362.._crit_edge362_crit_edge:          ; preds = %._crit_edge362
; BB803 :
  %8050 = add nuw nsw i32 %8041, 1, !spirv.Decorations !631		; visa id: 10518
  br label %._crit_edge362, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10519

8051:                                             ; preds = %._crit_edge362
; BB804 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10521
  %8052 = load i64, i64* %6446, align 8		; visa id: 10521
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10522
  %8053 = icmp slt i32 %6616, %const_reg_dword
  %8054 = icmp slt i32 %7862, %const_reg_dword1		; visa id: 10522
  %8055 = and i1 %8053, %8054		; visa id: 10523
  br i1 %8055, label %8056, label %..preheader1.6_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10525

..preheader1.6_crit_edge:                         ; preds = %8051
; BB:
  br label %.preheader1.6, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8056:                                             ; preds = %8051
; BB806 :
  %8057 = bitcast i64 %8052 to <2 x i32>		; visa id: 10527
  %8058 = extractelement <2 x i32> %8057, i32 0		; visa id: 10529
  %8059 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8058, i32 1
  %8060 = bitcast <2 x i32> %8059 to i64		; visa id: 10529
  %8061 = ashr exact i64 %8060, 32		; visa id: 10530
  %8062 = bitcast i64 %8061 to <2 x i32>		; visa id: 10531
  %8063 = extractelement <2 x i32> %8062, i32 0		; visa id: 10535
  %8064 = extractelement <2 x i32> %8062, i32 1		; visa id: 10535
  %8065 = ashr i64 %8052, 32		; visa id: 10535
  %8066 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8063, i32 %8064, i32 %50, i32 %51)
  %8067 = extractvalue { i32, i32 } %8066, 0		; visa id: 10536
  %8068 = extractvalue { i32, i32 } %8066, 1		; visa id: 10536
  %8069 = insertelement <2 x i32> undef, i32 %8067, i32 0		; visa id: 10543
  %8070 = insertelement <2 x i32> %8069, i32 %8068, i32 1		; visa id: 10544
  %8071 = bitcast <2 x i32> %8070 to i64		; visa id: 10545
  %8072 = add nsw i64 %8071, %8065, !spirv.Decorations !649		; visa id: 10549
  %8073 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !618		; visa id: 10550
  br i1 %86, label %8079, label %8074, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10551

8074:                                             ; preds = %8056
; BB807 :
  %8075 = shl i64 %8072, 2		; visa id: 10553
  %8076 = add i64 %.in, %8075		; visa id: 10554
  %8077 = inttoptr i64 %8076 to float addrspace(4)*		; visa id: 10555
  %8078 = addrspacecast float addrspace(4)* %8077 to float addrspace(1)*		; visa id: 10555
  store float %8073, float addrspace(1)* %8078, align 4		; visa id: 10556
  br label %.preheader1.6, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10557

8079:                                             ; preds = %8056
; BB808 :
  %8080 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8063, i32 %8064, i32 %47, i32 %48)
  %8081 = extractvalue { i32, i32 } %8080, 0		; visa id: 10559
  %8082 = extractvalue { i32, i32 } %8080, 1		; visa id: 10559
  %8083 = insertelement <2 x i32> undef, i32 %8081, i32 0		; visa id: 10566
  %8084 = insertelement <2 x i32> %8083, i32 %8082, i32 1		; visa id: 10567
  %8085 = bitcast <2 x i32> %8084 to i64		; visa id: 10568
  %8086 = shl i64 %8085, 2		; visa id: 10572
  %8087 = add i64 %.in399, %8086		; visa id: 10573
  %8088 = shl nsw i64 %8065, 2		; visa id: 10574
  %8089 = add i64 %8087, %8088		; visa id: 10575
  %8090 = inttoptr i64 %8089 to float addrspace(4)*		; visa id: 10576
  %8091 = addrspacecast float addrspace(4)* %8090 to float addrspace(1)*		; visa id: 10576
  %8092 = load float, float addrspace(1)* %8091, align 4		; visa id: 10577
  %8093 = fmul reassoc nsz arcp contract float %8092, %4, !spirv.Decorations !618		; visa id: 10578
  %8094 = fadd reassoc nsz arcp contract float %8073, %8093, !spirv.Decorations !618		; visa id: 10579
  %8095 = shl i64 %8072, 2		; visa id: 10580
  %8096 = add i64 %.in, %8095		; visa id: 10581
  %8097 = inttoptr i64 %8096 to float addrspace(4)*		; visa id: 10582
  %8098 = addrspacecast float addrspace(4)* %8097 to float addrspace(1)*		; visa id: 10582
  store float %8094, float addrspace(1)* %8098, align 4		; visa id: 10583
  br label %.preheader1.6, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10584

.preheader1.6:                                    ; preds = %..preheader1.6_crit_edge, %8079, %8074
; BB809 :
  %8099 = add i32 %69, 7		; visa id: 10585
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10586
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10586
  %8100 = insertelement <2 x i32> %6432, i32 %8099, i64 1		; visa id: 10586
  store <2 x i32> %8100, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10589
  br label %._crit_edge363, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10591

._crit_edge363:                                   ; preds = %._crit_edge363.._crit_edge363_crit_edge, %.preheader1.6
; BB810 :
  %8101 = phi i32 [ 0, %.preheader1.6 ], [ %8110, %._crit_edge363.._crit_edge363_crit_edge ]
  %8102 = zext i32 %8101 to i64		; visa id: 10592
  %8103 = shl nuw nsw i64 %8102, 2		; visa id: 10593
  %8104 = add i64 %6430, %8103		; visa id: 10594
  %8105 = inttoptr i64 %8104 to i32*		; visa id: 10595
  %8106 = load i32, i32* %8105, align 4, !noalias !642		; visa id: 10595
  %8107 = add i64 %6426, %8103		; visa id: 10596
  %8108 = inttoptr i64 %8107 to i32*		; visa id: 10597
  store i32 %8106, i32* %8108, align 4, !alias.scope !642		; visa id: 10597
  %8109 = icmp eq i32 %8101, 0		; visa id: 10598
  br i1 %8109, label %._crit_edge363.._crit_edge363_crit_edge, label %8111, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10599

._crit_edge363.._crit_edge363_crit_edge:          ; preds = %._crit_edge363
; BB811 :
  %8110 = add nuw nsw i32 %8101, 1, !spirv.Decorations !631		; visa id: 10601
  br label %._crit_edge363, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10602

8111:                                             ; preds = %._crit_edge363
; BB812 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10604
  %8112 = load i64, i64* %6446, align 8		; visa id: 10604
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10605
  %8113 = icmp slt i32 %8099, %const_reg_dword1		; visa id: 10605
  %8114 = icmp slt i32 %65, %const_reg_dword
  %8115 = and i1 %8114, %8113		; visa id: 10606
  br i1 %8115, label %8116, label %.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10608

.._crit_edge70.7_crit_edge:                       ; preds = %8111
; BB:
  br label %._crit_edge70.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8116:                                             ; preds = %8111
; BB814 :
  %8117 = bitcast i64 %8112 to <2 x i32>		; visa id: 10610
  %8118 = extractelement <2 x i32> %8117, i32 0		; visa id: 10612
  %8119 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8118, i32 1
  %8120 = bitcast <2 x i32> %8119 to i64		; visa id: 10612
  %8121 = ashr exact i64 %8120, 32		; visa id: 10613
  %8122 = bitcast i64 %8121 to <2 x i32>		; visa id: 10614
  %8123 = extractelement <2 x i32> %8122, i32 0		; visa id: 10618
  %8124 = extractelement <2 x i32> %8122, i32 1		; visa id: 10618
  %8125 = ashr i64 %8112, 32		; visa id: 10618
  %8126 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8123, i32 %8124, i32 %50, i32 %51)
  %8127 = extractvalue { i32, i32 } %8126, 0		; visa id: 10619
  %8128 = extractvalue { i32, i32 } %8126, 1		; visa id: 10619
  %8129 = insertelement <2 x i32> undef, i32 %8127, i32 0		; visa id: 10626
  %8130 = insertelement <2 x i32> %8129, i32 %8128, i32 1		; visa id: 10627
  %8131 = bitcast <2 x i32> %8130 to i64		; visa id: 10628
  %8132 = add nsw i64 %8131, %8125, !spirv.Decorations !649		; visa id: 10632
  %8133 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !618		; visa id: 10633
  br i1 %86, label %8139, label %8134, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10634

8134:                                             ; preds = %8116
; BB815 :
  %8135 = shl i64 %8132, 2		; visa id: 10636
  %8136 = add i64 %.in, %8135		; visa id: 10637
  %8137 = inttoptr i64 %8136 to float addrspace(4)*		; visa id: 10638
  %8138 = addrspacecast float addrspace(4)* %8137 to float addrspace(1)*		; visa id: 10638
  store float %8133, float addrspace(1)* %8138, align 4		; visa id: 10639
  br label %._crit_edge70.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10640

8139:                                             ; preds = %8116
; BB816 :
  %8140 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8123, i32 %8124, i32 %47, i32 %48)
  %8141 = extractvalue { i32, i32 } %8140, 0		; visa id: 10642
  %8142 = extractvalue { i32, i32 } %8140, 1		; visa id: 10642
  %8143 = insertelement <2 x i32> undef, i32 %8141, i32 0		; visa id: 10649
  %8144 = insertelement <2 x i32> %8143, i32 %8142, i32 1		; visa id: 10650
  %8145 = bitcast <2 x i32> %8144 to i64		; visa id: 10651
  %8146 = shl i64 %8145, 2		; visa id: 10655
  %8147 = add i64 %.in399, %8146		; visa id: 10656
  %8148 = shl nsw i64 %8125, 2		; visa id: 10657
  %8149 = add i64 %8147, %8148		; visa id: 10658
  %8150 = inttoptr i64 %8149 to float addrspace(4)*		; visa id: 10659
  %8151 = addrspacecast float addrspace(4)* %8150 to float addrspace(1)*		; visa id: 10659
  %8152 = load float, float addrspace(1)* %8151, align 4		; visa id: 10660
  %8153 = fmul reassoc nsz arcp contract float %8152, %4, !spirv.Decorations !618		; visa id: 10661
  %8154 = fadd reassoc nsz arcp contract float %8133, %8153, !spirv.Decorations !618		; visa id: 10662
  %8155 = shl i64 %8132, 2		; visa id: 10663
  %8156 = add i64 %.in, %8155		; visa id: 10664
  %8157 = inttoptr i64 %8156 to float addrspace(4)*		; visa id: 10665
  %8158 = addrspacecast float addrspace(4)* %8157 to float addrspace(1)*		; visa id: 10665
  store float %8154, float addrspace(1)* %8158, align 4		; visa id: 10666
  br label %._crit_edge70.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10667

._crit_edge70.7:                                  ; preds = %.._crit_edge70.7_crit_edge, %8139, %8134
; BB817 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10668
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10668
  %8159 = insertelement <2 x i32> %6495, i32 %8099, i64 1		; visa id: 10668
  store <2 x i32> %8159, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10671
  br label %._crit_edge364, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10673

._crit_edge364:                                   ; preds = %._crit_edge364.._crit_edge364_crit_edge, %._crit_edge70.7
; BB818 :
  %8160 = phi i32 [ 0, %._crit_edge70.7 ], [ %8169, %._crit_edge364.._crit_edge364_crit_edge ]
  %8161 = zext i32 %8160 to i64		; visa id: 10674
  %8162 = shl nuw nsw i64 %8161, 2		; visa id: 10675
  %8163 = add i64 %6430, %8162		; visa id: 10676
  %8164 = inttoptr i64 %8163 to i32*		; visa id: 10677
  %8165 = load i32, i32* %8164, align 4, !noalias !642		; visa id: 10677
  %8166 = add i64 %6426, %8162		; visa id: 10678
  %8167 = inttoptr i64 %8166 to i32*		; visa id: 10679
  store i32 %8165, i32* %8167, align 4, !alias.scope !642		; visa id: 10679
  %8168 = icmp eq i32 %8160, 0		; visa id: 10680
  br i1 %8168, label %._crit_edge364.._crit_edge364_crit_edge, label %8170, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10681

._crit_edge364.._crit_edge364_crit_edge:          ; preds = %._crit_edge364
; BB819 :
  %8169 = add nuw nsw i32 %8160, 1, !spirv.Decorations !631		; visa id: 10683
  br label %._crit_edge364, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10684

8170:                                             ; preds = %._crit_edge364
; BB820 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10686
  %8171 = load i64, i64* %6446, align 8		; visa id: 10686
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10687
  %8172 = icmp slt i32 %6494, %const_reg_dword
  %8173 = icmp slt i32 %8099, %const_reg_dword1		; visa id: 10687
  %8174 = and i1 %8172, %8173		; visa id: 10688
  br i1 %8174, label %8175, label %.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10690

.._crit_edge70.1.7_crit_edge:                     ; preds = %8170
; BB:
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8175:                                             ; preds = %8170
; BB822 :
  %8176 = bitcast i64 %8171 to <2 x i32>		; visa id: 10692
  %8177 = extractelement <2 x i32> %8176, i32 0		; visa id: 10694
  %8178 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8177, i32 1
  %8179 = bitcast <2 x i32> %8178 to i64		; visa id: 10694
  %8180 = ashr exact i64 %8179, 32		; visa id: 10695
  %8181 = bitcast i64 %8180 to <2 x i32>		; visa id: 10696
  %8182 = extractelement <2 x i32> %8181, i32 0		; visa id: 10700
  %8183 = extractelement <2 x i32> %8181, i32 1		; visa id: 10700
  %8184 = ashr i64 %8171, 32		; visa id: 10700
  %8185 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8182, i32 %8183, i32 %50, i32 %51)
  %8186 = extractvalue { i32, i32 } %8185, 0		; visa id: 10701
  %8187 = extractvalue { i32, i32 } %8185, 1		; visa id: 10701
  %8188 = insertelement <2 x i32> undef, i32 %8186, i32 0		; visa id: 10708
  %8189 = insertelement <2 x i32> %8188, i32 %8187, i32 1		; visa id: 10709
  %8190 = bitcast <2 x i32> %8189 to i64		; visa id: 10710
  %8191 = add nsw i64 %8190, %8184, !spirv.Decorations !649		; visa id: 10714
  %8192 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !618		; visa id: 10715
  br i1 %86, label %8198, label %8193, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10716

8193:                                             ; preds = %8175
; BB823 :
  %8194 = shl i64 %8191, 2		; visa id: 10718
  %8195 = add i64 %.in, %8194		; visa id: 10719
  %8196 = inttoptr i64 %8195 to float addrspace(4)*		; visa id: 10720
  %8197 = addrspacecast float addrspace(4)* %8196 to float addrspace(1)*		; visa id: 10720
  store float %8192, float addrspace(1)* %8197, align 4		; visa id: 10721
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10722

8198:                                             ; preds = %8175
; BB824 :
  %8199 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8182, i32 %8183, i32 %47, i32 %48)
  %8200 = extractvalue { i32, i32 } %8199, 0		; visa id: 10724
  %8201 = extractvalue { i32, i32 } %8199, 1		; visa id: 10724
  %8202 = insertelement <2 x i32> undef, i32 %8200, i32 0		; visa id: 10731
  %8203 = insertelement <2 x i32> %8202, i32 %8201, i32 1		; visa id: 10732
  %8204 = bitcast <2 x i32> %8203 to i64		; visa id: 10733
  %8205 = shl i64 %8204, 2		; visa id: 10737
  %8206 = add i64 %.in399, %8205		; visa id: 10738
  %8207 = shl nsw i64 %8184, 2		; visa id: 10739
  %8208 = add i64 %8206, %8207		; visa id: 10740
  %8209 = inttoptr i64 %8208 to float addrspace(4)*		; visa id: 10741
  %8210 = addrspacecast float addrspace(4)* %8209 to float addrspace(1)*		; visa id: 10741
  %8211 = load float, float addrspace(1)* %8210, align 4		; visa id: 10742
  %8212 = fmul reassoc nsz arcp contract float %8211, %4, !spirv.Decorations !618		; visa id: 10743
  %8213 = fadd reassoc nsz arcp contract float %8192, %8212, !spirv.Decorations !618		; visa id: 10744
  %8214 = shl i64 %8191, 2		; visa id: 10745
  %8215 = add i64 %.in, %8214		; visa id: 10746
  %8216 = inttoptr i64 %8215 to float addrspace(4)*		; visa id: 10747
  %8217 = addrspacecast float addrspace(4)* %8216 to float addrspace(1)*		; visa id: 10747
  store float %8213, float addrspace(1)* %8217, align 4		; visa id: 10748
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10749

._crit_edge70.1.7:                                ; preds = %.._crit_edge70.1.7_crit_edge, %8198, %8193
; BB825 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10750
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10750
  %8218 = insertelement <2 x i32> %6556, i32 %8099, i64 1		; visa id: 10750
  store <2 x i32> %8218, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10753
  br label %._crit_edge365, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10755

._crit_edge365:                                   ; preds = %._crit_edge365.._crit_edge365_crit_edge, %._crit_edge70.1.7
; BB826 :
  %8219 = phi i32 [ 0, %._crit_edge70.1.7 ], [ %8228, %._crit_edge365.._crit_edge365_crit_edge ]
  %8220 = zext i32 %8219 to i64		; visa id: 10756
  %8221 = shl nuw nsw i64 %8220, 2		; visa id: 10757
  %8222 = add i64 %6430, %8221		; visa id: 10758
  %8223 = inttoptr i64 %8222 to i32*		; visa id: 10759
  %8224 = load i32, i32* %8223, align 4, !noalias !642		; visa id: 10759
  %8225 = add i64 %6426, %8221		; visa id: 10760
  %8226 = inttoptr i64 %8225 to i32*		; visa id: 10761
  store i32 %8224, i32* %8226, align 4, !alias.scope !642		; visa id: 10761
  %8227 = icmp eq i32 %8219, 0		; visa id: 10762
  br i1 %8227, label %._crit_edge365.._crit_edge365_crit_edge, label %8229, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10763

._crit_edge365.._crit_edge365_crit_edge:          ; preds = %._crit_edge365
; BB827 :
  %8228 = add nuw nsw i32 %8219, 1, !spirv.Decorations !631		; visa id: 10765
  br label %._crit_edge365, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10766

8229:                                             ; preds = %._crit_edge365
; BB828 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10768
  %8230 = load i64, i64* %6446, align 8		; visa id: 10768
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10769
  %8231 = icmp slt i32 %6555, %const_reg_dword
  %8232 = icmp slt i32 %8099, %const_reg_dword1		; visa id: 10769
  %8233 = and i1 %8231, %8232		; visa id: 10770
  br i1 %8233, label %8234, label %.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10772

.._crit_edge70.2.7_crit_edge:                     ; preds = %8229
; BB:
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8234:                                             ; preds = %8229
; BB830 :
  %8235 = bitcast i64 %8230 to <2 x i32>		; visa id: 10774
  %8236 = extractelement <2 x i32> %8235, i32 0		; visa id: 10776
  %8237 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8236, i32 1
  %8238 = bitcast <2 x i32> %8237 to i64		; visa id: 10776
  %8239 = ashr exact i64 %8238, 32		; visa id: 10777
  %8240 = bitcast i64 %8239 to <2 x i32>		; visa id: 10778
  %8241 = extractelement <2 x i32> %8240, i32 0		; visa id: 10782
  %8242 = extractelement <2 x i32> %8240, i32 1		; visa id: 10782
  %8243 = ashr i64 %8230, 32		; visa id: 10782
  %8244 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8241, i32 %8242, i32 %50, i32 %51)
  %8245 = extractvalue { i32, i32 } %8244, 0		; visa id: 10783
  %8246 = extractvalue { i32, i32 } %8244, 1		; visa id: 10783
  %8247 = insertelement <2 x i32> undef, i32 %8245, i32 0		; visa id: 10790
  %8248 = insertelement <2 x i32> %8247, i32 %8246, i32 1		; visa id: 10791
  %8249 = bitcast <2 x i32> %8248 to i64		; visa id: 10792
  %8250 = add nsw i64 %8249, %8243, !spirv.Decorations !649		; visa id: 10796
  %8251 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !618		; visa id: 10797
  br i1 %86, label %8257, label %8252, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10798

8252:                                             ; preds = %8234
; BB831 :
  %8253 = shl i64 %8250, 2		; visa id: 10800
  %8254 = add i64 %.in, %8253		; visa id: 10801
  %8255 = inttoptr i64 %8254 to float addrspace(4)*		; visa id: 10802
  %8256 = addrspacecast float addrspace(4)* %8255 to float addrspace(1)*		; visa id: 10802
  store float %8251, float addrspace(1)* %8256, align 4		; visa id: 10803
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10804

8257:                                             ; preds = %8234
; BB832 :
  %8258 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8241, i32 %8242, i32 %47, i32 %48)
  %8259 = extractvalue { i32, i32 } %8258, 0		; visa id: 10806
  %8260 = extractvalue { i32, i32 } %8258, 1		; visa id: 10806
  %8261 = insertelement <2 x i32> undef, i32 %8259, i32 0		; visa id: 10813
  %8262 = insertelement <2 x i32> %8261, i32 %8260, i32 1		; visa id: 10814
  %8263 = bitcast <2 x i32> %8262 to i64		; visa id: 10815
  %8264 = shl i64 %8263, 2		; visa id: 10819
  %8265 = add i64 %.in399, %8264		; visa id: 10820
  %8266 = shl nsw i64 %8243, 2		; visa id: 10821
  %8267 = add i64 %8265, %8266		; visa id: 10822
  %8268 = inttoptr i64 %8267 to float addrspace(4)*		; visa id: 10823
  %8269 = addrspacecast float addrspace(4)* %8268 to float addrspace(1)*		; visa id: 10823
  %8270 = load float, float addrspace(1)* %8269, align 4		; visa id: 10824
  %8271 = fmul reassoc nsz arcp contract float %8270, %4, !spirv.Decorations !618		; visa id: 10825
  %8272 = fadd reassoc nsz arcp contract float %8251, %8271, !spirv.Decorations !618		; visa id: 10826
  %8273 = shl i64 %8250, 2		; visa id: 10827
  %8274 = add i64 %.in, %8273		; visa id: 10828
  %8275 = inttoptr i64 %8274 to float addrspace(4)*		; visa id: 10829
  %8276 = addrspacecast float addrspace(4)* %8275 to float addrspace(1)*		; visa id: 10829
  store float %8272, float addrspace(1)* %8276, align 4		; visa id: 10830
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10831

._crit_edge70.2.7:                                ; preds = %.._crit_edge70.2.7_crit_edge, %8257, %8252
; BB833 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10832
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10832
  %8277 = insertelement <2 x i32> %6617, i32 %8099, i64 1		; visa id: 10832
  store <2 x i32> %8277, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10835
  br label %._crit_edge366, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10837

._crit_edge366:                                   ; preds = %._crit_edge366.._crit_edge366_crit_edge, %._crit_edge70.2.7
; BB834 :
  %8278 = phi i32 [ 0, %._crit_edge70.2.7 ], [ %8287, %._crit_edge366.._crit_edge366_crit_edge ]
  %8279 = zext i32 %8278 to i64		; visa id: 10838
  %8280 = shl nuw nsw i64 %8279, 2		; visa id: 10839
  %8281 = add i64 %6430, %8280		; visa id: 10840
  %8282 = inttoptr i64 %8281 to i32*		; visa id: 10841
  %8283 = load i32, i32* %8282, align 4, !noalias !642		; visa id: 10841
  %8284 = add i64 %6426, %8280		; visa id: 10842
  %8285 = inttoptr i64 %8284 to i32*		; visa id: 10843
  store i32 %8283, i32* %8285, align 4, !alias.scope !642		; visa id: 10843
  %8286 = icmp eq i32 %8278, 0		; visa id: 10844
  br i1 %8286, label %._crit_edge366.._crit_edge366_crit_edge, label %8288, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10845

._crit_edge366.._crit_edge366_crit_edge:          ; preds = %._crit_edge366
; BB835 :
  %8287 = add nuw nsw i32 %8278, 1, !spirv.Decorations !631		; visa id: 10847
  br label %._crit_edge366, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10848

8288:                                             ; preds = %._crit_edge366
; BB836 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10850
  %8289 = load i64, i64* %6446, align 8		; visa id: 10850
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10851
  %8290 = icmp slt i32 %6616, %const_reg_dword
  %8291 = icmp slt i32 %8099, %const_reg_dword1		; visa id: 10851
  %8292 = and i1 %8290, %8291		; visa id: 10852
  br i1 %8292, label %8293, label %..preheader1.7_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10854

..preheader1.7_crit_edge:                         ; preds = %8288
; BB:
  br label %.preheader1.7, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8293:                                             ; preds = %8288
; BB838 :
  %8294 = bitcast i64 %8289 to <2 x i32>		; visa id: 10856
  %8295 = extractelement <2 x i32> %8294, i32 0		; visa id: 10858
  %8296 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8295, i32 1
  %8297 = bitcast <2 x i32> %8296 to i64		; visa id: 10858
  %8298 = ashr exact i64 %8297, 32		; visa id: 10859
  %8299 = bitcast i64 %8298 to <2 x i32>		; visa id: 10860
  %8300 = extractelement <2 x i32> %8299, i32 0		; visa id: 10864
  %8301 = extractelement <2 x i32> %8299, i32 1		; visa id: 10864
  %8302 = ashr i64 %8289, 32		; visa id: 10864
  %8303 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8300, i32 %8301, i32 %50, i32 %51)
  %8304 = extractvalue { i32, i32 } %8303, 0		; visa id: 10865
  %8305 = extractvalue { i32, i32 } %8303, 1		; visa id: 10865
  %8306 = insertelement <2 x i32> undef, i32 %8304, i32 0		; visa id: 10872
  %8307 = insertelement <2 x i32> %8306, i32 %8305, i32 1		; visa id: 10873
  %8308 = bitcast <2 x i32> %8307 to i64		; visa id: 10874
  %8309 = add nsw i64 %8308, %8302, !spirv.Decorations !649		; visa id: 10878
  %8310 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !618		; visa id: 10879
  br i1 %86, label %8316, label %8311, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10880

8311:                                             ; preds = %8293
; BB839 :
  %8312 = shl i64 %8309, 2		; visa id: 10882
  %8313 = add i64 %.in, %8312		; visa id: 10883
  %8314 = inttoptr i64 %8313 to float addrspace(4)*		; visa id: 10884
  %8315 = addrspacecast float addrspace(4)* %8314 to float addrspace(1)*		; visa id: 10884
  store float %8310, float addrspace(1)* %8315, align 4		; visa id: 10885
  br label %.preheader1.7, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10886

8316:                                             ; preds = %8293
; BB840 :
  %8317 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8300, i32 %8301, i32 %47, i32 %48)
  %8318 = extractvalue { i32, i32 } %8317, 0		; visa id: 10888
  %8319 = extractvalue { i32, i32 } %8317, 1		; visa id: 10888
  %8320 = insertelement <2 x i32> undef, i32 %8318, i32 0		; visa id: 10895
  %8321 = insertelement <2 x i32> %8320, i32 %8319, i32 1		; visa id: 10896
  %8322 = bitcast <2 x i32> %8321 to i64		; visa id: 10897
  %8323 = shl i64 %8322, 2		; visa id: 10901
  %8324 = add i64 %.in399, %8323		; visa id: 10902
  %8325 = shl nsw i64 %8302, 2		; visa id: 10903
  %8326 = add i64 %8324, %8325		; visa id: 10904
  %8327 = inttoptr i64 %8326 to float addrspace(4)*		; visa id: 10905
  %8328 = addrspacecast float addrspace(4)* %8327 to float addrspace(1)*		; visa id: 10905
  %8329 = load float, float addrspace(1)* %8328, align 4		; visa id: 10906
  %8330 = fmul reassoc nsz arcp contract float %8329, %4, !spirv.Decorations !618		; visa id: 10907
  %8331 = fadd reassoc nsz arcp contract float %8310, %8330, !spirv.Decorations !618		; visa id: 10908
  %8332 = shl i64 %8309, 2		; visa id: 10909
  %8333 = add i64 %.in, %8332		; visa id: 10910
  %8334 = inttoptr i64 %8333 to float addrspace(4)*		; visa id: 10911
  %8335 = addrspacecast float addrspace(4)* %8334 to float addrspace(1)*		; visa id: 10911
  store float %8331, float addrspace(1)* %8335, align 4		; visa id: 10912
  br label %.preheader1.7, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10913

.preheader1.7:                                    ; preds = %..preheader1.7_crit_edge, %8316, %8311
; BB841 :
  %8336 = add i32 %69, 8		; visa id: 10914
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10915
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10915
  %8337 = insertelement <2 x i32> %6432, i32 %8336, i64 1		; visa id: 10915
  store <2 x i32> %8337, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 10918
  br label %._crit_edge367, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10920

._crit_edge367:                                   ; preds = %._crit_edge367.._crit_edge367_crit_edge, %.preheader1.7
; BB842 :
  %8338 = phi i32 [ 0, %.preheader1.7 ], [ %8347, %._crit_edge367.._crit_edge367_crit_edge ]
  %8339 = zext i32 %8338 to i64		; visa id: 10921
  %8340 = shl nuw nsw i64 %8339, 2		; visa id: 10922
  %8341 = add i64 %6430, %8340		; visa id: 10923
  %8342 = inttoptr i64 %8341 to i32*		; visa id: 10924
  %8343 = load i32, i32* %8342, align 4, !noalias !642		; visa id: 10924
  %8344 = add i64 %6426, %8340		; visa id: 10925
  %8345 = inttoptr i64 %8344 to i32*		; visa id: 10926
  store i32 %8343, i32* %8345, align 4, !alias.scope !642		; visa id: 10926
  %8346 = icmp eq i32 %8338, 0		; visa id: 10927
  br i1 %8346, label %._crit_edge367.._crit_edge367_crit_edge, label %8348, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 10928

._crit_edge367.._crit_edge367_crit_edge:          ; preds = %._crit_edge367
; BB843 :
  %8347 = add nuw nsw i32 %8338, 1, !spirv.Decorations !631		; visa id: 10930
  br label %._crit_edge367, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 10931

8348:                                             ; preds = %._crit_edge367
; BB844 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10933
  %8349 = load i64, i64* %6446, align 8		; visa id: 10933
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10934
  %8350 = icmp slt i32 %8336, %const_reg_dword1		; visa id: 10934
  %8351 = icmp slt i32 %65, %const_reg_dword
  %8352 = and i1 %8351, %8350		; visa id: 10935
  br i1 %8352, label %8353, label %.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 10937

.._crit_edge70.8_crit_edge:                       ; preds = %8348
; BB:
  br label %._crit_edge70.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8353:                                             ; preds = %8348
; BB846 :
  %8354 = bitcast i64 %8349 to <2 x i32>		; visa id: 10939
  %8355 = extractelement <2 x i32> %8354, i32 0		; visa id: 10941
  %8356 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8355, i32 1
  %8357 = bitcast <2 x i32> %8356 to i64		; visa id: 10941
  %8358 = ashr exact i64 %8357, 32		; visa id: 10942
  %8359 = bitcast i64 %8358 to <2 x i32>		; visa id: 10943
  %8360 = extractelement <2 x i32> %8359, i32 0		; visa id: 10947
  %8361 = extractelement <2 x i32> %8359, i32 1		; visa id: 10947
  %8362 = ashr i64 %8349, 32		; visa id: 10947
  %8363 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8360, i32 %8361, i32 %50, i32 %51)
  %8364 = extractvalue { i32, i32 } %8363, 0		; visa id: 10948
  %8365 = extractvalue { i32, i32 } %8363, 1		; visa id: 10948
  %8366 = insertelement <2 x i32> undef, i32 %8364, i32 0		; visa id: 10955
  %8367 = insertelement <2 x i32> %8366, i32 %8365, i32 1		; visa id: 10956
  %8368 = bitcast <2 x i32> %8367 to i64		; visa id: 10957
  %8369 = add nsw i64 %8368, %8362, !spirv.Decorations !649		; visa id: 10961
  %8370 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !618		; visa id: 10962
  br i1 %86, label %8376, label %8371, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 10963

8371:                                             ; preds = %8353
; BB847 :
  %8372 = shl i64 %8369, 2		; visa id: 10965
  %8373 = add i64 %.in, %8372		; visa id: 10966
  %8374 = inttoptr i64 %8373 to float addrspace(4)*		; visa id: 10967
  %8375 = addrspacecast float addrspace(4)* %8374 to float addrspace(1)*		; visa id: 10967
  store float %8370, float addrspace(1)* %8375, align 4		; visa id: 10968
  br label %._crit_edge70.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 10969

8376:                                             ; preds = %8353
; BB848 :
  %8377 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8360, i32 %8361, i32 %47, i32 %48)
  %8378 = extractvalue { i32, i32 } %8377, 0		; visa id: 10971
  %8379 = extractvalue { i32, i32 } %8377, 1		; visa id: 10971
  %8380 = insertelement <2 x i32> undef, i32 %8378, i32 0		; visa id: 10978
  %8381 = insertelement <2 x i32> %8380, i32 %8379, i32 1		; visa id: 10979
  %8382 = bitcast <2 x i32> %8381 to i64		; visa id: 10980
  %8383 = shl i64 %8382, 2		; visa id: 10984
  %8384 = add i64 %.in399, %8383		; visa id: 10985
  %8385 = shl nsw i64 %8362, 2		; visa id: 10986
  %8386 = add i64 %8384, %8385		; visa id: 10987
  %8387 = inttoptr i64 %8386 to float addrspace(4)*		; visa id: 10988
  %8388 = addrspacecast float addrspace(4)* %8387 to float addrspace(1)*		; visa id: 10988
  %8389 = load float, float addrspace(1)* %8388, align 4		; visa id: 10989
  %8390 = fmul reassoc nsz arcp contract float %8389, %4, !spirv.Decorations !618		; visa id: 10990
  %8391 = fadd reassoc nsz arcp contract float %8370, %8390, !spirv.Decorations !618		; visa id: 10991
  %8392 = shl i64 %8369, 2		; visa id: 10992
  %8393 = add i64 %.in, %8392		; visa id: 10993
  %8394 = inttoptr i64 %8393 to float addrspace(4)*		; visa id: 10994
  %8395 = addrspacecast float addrspace(4)* %8394 to float addrspace(1)*		; visa id: 10994
  store float %8391, float addrspace(1)* %8395, align 4		; visa id: 10995
  br label %._crit_edge70.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 10996

._crit_edge70.8:                                  ; preds = %.._crit_edge70.8_crit_edge, %8376, %8371
; BB849 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 10997
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 10997
  %8396 = insertelement <2 x i32> %6495, i32 %8336, i64 1		; visa id: 10997
  store <2 x i32> %8396, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11000
  br label %._crit_edge368, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11002

._crit_edge368:                                   ; preds = %._crit_edge368.._crit_edge368_crit_edge, %._crit_edge70.8
; BB850 :
  %8397 = phi i32 [ 0, %._crit_edge70.8 ], [ %8406, %._crit_edge368.._crit_edge368_crit_edge ]
  %8398 = zext i32 %8397 to i64		; visa id: 11003
  %8399 = shl nuw nsw i64 %8398, 2		; visa id: 11004
  %8400 = add i64 %6430, %8399		; visa id: 11005
  %8401 = inttoptr i64 %8400 to i32*		; visa id: 11006
  %8402 = load i32, i32* %8401, align 4, !noalias !642		; visa id: 11006
  %8403 = add i64 %6426, %8399		; visa id: 11007
  %8404 = inttoptr i64 %8403 to i32*		; visa id: 11008
  store i32 %8402, i32* %8404, align 4, !alias.scope !642		; visa id: 11008
  %8405 = icmp eq i32 %8397, 0		; visa id: 11009
  br i1 %8405, label %._crit_edge368.._crit_edge368_crit_edge, label %8407, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11010

._crit_edge368.._crit_edge368_crit_edge:          ; preds = %._crit_edge368
; BB851 :
  %8406 = add nuw nsw i32 %8397, 1, !spirv.Decorations !631		; visa id: 11012
  br label %._crit_edge368, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11013

8407:                                             ; preds = %._crit_edge368
; BB852 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11015
  %8408 = load i64, i64* %6446, align 8		; visa id: 11015
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11016
  %8409 = icmp slt i32 %6494, %const_reg_dword
  %8410 = icmp slt i32 %8336, %const_reg_dword1		; visa id: 11016
  %8411 = and i1 %8409, %8410		; visa id: 11017
  br i1 %8411, label %8412, label %.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11019

.._crit_edge70.1.8_crit_edge:                     ; preds = %8407
; BB:
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8412:                                             ; preds = %8407
; BB854 :
  %8413 = bitcast i64 %8408 to <2 x i32>		; visa id: 11021
  %8414 = extractelement <2 x i32> %8413, i32 0		; visa id: 11023
  %8415 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8414, i32 1
  %8416 = bitcast <2 x i32> %8415 to i64		; visa id: 11023
  %8417 = ashr exact i64 %8416, 32		; visa id: 11024
  %8418 = bitcast i64 %8417 to <2 x i32>		; visa id: 11025
  %8419 = extractelement <2 x i32> %8418, i32 0		; visa id: 11029
  %8420 = extractelement <2 x i32> %8418, i32 1		; visa id: 11029
  %8421 = ashr i64 %8408, 32		; visa id: 11029
  %8422 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8419, i32 %8420, i32 %50, i32 %51)
  %8423 = extractvalue { i32, i32 } %8422, 0		; visa id: 11030
  %8424 = extractvalue { i32, i32 } %8422, 1		; visa id: 11030
  %8425 = insertelement <2 x i32> undef, i32 %8423, i32 0		; visa id: 11037
  %8426 = insertelement <2 x i32> %8425, i32 %8424, i32 1		; visa id: 11038
  %8427 = bitcast <2 x i32> %8426 to i64		; visa id: 11039
  %8428 = add nsw i64 %8427, %8421, !spirv.Decorations !649		; visa id: 11043
  %8429 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !618		; visa id: 11044
  br i1 %86, label %8435, label %8430, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11045

8430:                                             ; preds = %8412
; BB855 :
  %8431 = shl i64 %8428, 2		; visa id: 11047
  %8432 = add i64 %.in, %8431		; visa id: 11048
  %8433 = inttoptr i64 %8432 to float addrspace(4)*		; visa id: 11049
  %8434 = addrspacecast float addrspace(4)* %8433 to float addrspace(1)*		; visa id: 11049
  store float %8429, float addrspace(1)* %8434, align 4		; visa id: 11050
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11051

8435:                                             ; preds = %8412
; BB856 :
  %8436 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8419, i32 %8420, i32 %47, i32 %48)
  %8437 = extractvalue { i32, i32 } %8436, 0		; visa id: 11053
  %8438 = extractvalue { i32, i32 } %8436, 1		; visa id: 11053
  %8439 = insertelement <2 x i32> undef, i32 %8437, i32 0		; visa id: 11060
  %8440 = insertelement <2 x i32> %8439, i32 %8438, i32 1		; visa id: 11061
  %8441 = bitcast <2 x i32> %8440 to i64		; visa id: 11062
  %8442 = shl i64 %8441, 2		; visa id: 11066
  %8443 = add i64 %.in399, %8442		; visa id: 11067
  %8444 = shl nsw i64 %8421, 2		; visa id: 11068
  %8445 = add i64 %8443, %8444		; visa id: 11069
  %8446 = inttoptr i64 %8445 to float addrspace(4)*		; visa id: 11070
  %8447 = addrspacecast float addrspace(4)* %8446 to float addrspace(1)*		; visa id: 11070
  %8448 = load float, float addrspace(1)* %8447, align 4		; visa id: 11071
  %8449 = fmul reassoc nsz arcp contract float %8448, %4, !spirv.Decorations !618		; visa id: 11072
  %8450 = fadd reassoc nsz arcp contract float %8429, %8449, !spirv.Decorations !618		; visa id: 11073
  %8451 = shl i64 %8428, 2		; visa id: 11074
  %8452 = add i64 %.in, %8451		; visa id: 11075
  %8453 = inttoptr i64 %8452 to float addrspace(4)*		; visa id: 11076
  %8454 = addrspacecast float addrspace(4)* %8453 to float addrspace(1)*		; visa id: 11076
  store float %8450, float addrspace(1)* %8454, align 4		; visa id: 11077
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11078

._crit_edge70.1.8:                                ; preds = %.._crit_edge70.1.8_crit_edge, %8435, %8430
; BB857 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11079
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11079
  %8455 = insertelement <2 x i32> %6556, i32 %8336, i64 1		; visa id: 11079
  store <2 x i32> %8455, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11082
  br label %._crit_edge369, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11084

._crit_edge369:                                   ; preds = %._crit_edge369.._crit_edge369_crit_edge, %._crit_edge70.1.8
; BB858 :
  %8456 = phi i32 [ 0, %._crit_edge70.1.8 ], [ %8465, %._crit_edge369.._crit_edge369_crit_edge ]
  %8457 = zext i32 %8456 to i64		; visa id: 11085
  %8458 = shl nuw nsw i64 %8457, 2		; visa id: 11086
  %8459 = add i64 %6430, %8458		; visa id: 11087
  %8460 = inttoptr i64 %8459 to i32*		; visa id: 11088
  %8461 = load i32, i32* %8460, align 4, !noalias !642		; visa id: 11088
  %8462 = add i64 %6426, %8458		; visa id: 11089
  %8463 = inttoptr i64 %8462 to i32*		; visa id: 11090
  store i32 %8461, i32* %8463, align 4, !alias.scope !642		; visa id: 11090
  %8464 = icmp eq i32 %8456, 0		; visa id: 11091
  br i1 %8464, label %._crit_edge369.._crit_edge369_crit_edge, label %8466, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11092

._crit_edge369.._crit_edge369_crit_edge:          ; preds = %._crit_edge369
; BB859 :
  %8465 = add nuw nsw i32 %8456, 1, !spirv.Decorations !631		; visa id: 11094
  br label %._crit_edge369, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11095

8466:                                             ; preds = %._crit_edge369
; BB860 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11097
  %8467 = load i64, i64* %6446, align 8		; visa id: 11097
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11098
  %8468 = icmp slt i32 %6555, %const_reg_dword
  %8469 = icmp slt i32 %8336, %const_reg_dword1		; visa id: 11098
  %8470 = and i1 %8468, %8469		; visa id: 11099
  br i1 %8470, label %8471, label %.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11101

.._crit_edge70.2.8_crit_edge:                     ; preds = %8466
; BB:
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8471:                                             ; preds = %8466
; BB862 :
  %8472 = bitcast i64 %8467 to <2 x i32>		; visa id: 11103
  %8473 = extractelement <2 x i32> %8472, i32 0		; visa id: 11105
  %8474 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8473, i32 1
  %8475 = bitcast <2 x i32> %8474 to i64		; visa id: 11105
  %8476 = ashr exact i64 %8475, 32		; visa id: 11106
  %8477 = bitcast i64 %8476 to <2 x i32>		; visa id: 11107
  %8478 = extractelement <2 x i32> %8477, i32 0		; visa id: 11111
  %8479 = extractelement <2 x i32> %8477, i32 1		; visa id: 11111
  %8480 = ashr i64 %8467, 32		; visa id: 11111
  %8481 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8478, i32 %8479, i32 %50, i32 %51)
  %8482 = extractvalue { i32, i32 } %8481, 0		; visa id: 11112
  %8483 = extractvalue { i32, i32 } %8481, 1		; visa id: 11112
  %8484 = insertelement <2 x i32> undef, i32 %8482, i32 0		; visa id: 11119
  %8485 = insertelement <2 x i32> %8484, i32 %8483, i32 1		; visa id: 11120
  %8486 = bitcast <2 x i32> %8485 to i64		; visa id: 11121
  %8487 = add nsw i64 %8486, %8480, !spirv.Decorations !649		; visa id: 11125
  %8488 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !618		; visa id: 11126
  br i1 %86, label %8494, label %8489, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11127

8489:                                             ; preds = %8471
; BB863 :
  %8490 = shl i64 %8487, 2		; visa id: 11129
  %8491 = add i64 %.in, %8490		; visa id: 11130
  %8492 = inttoptr i64 %8491 to float addrspace(4)*		; visa id: 11131
  %8493 = addrspacecast float addrspace(4)* %8492 to float addrspace(1)*		; visa id: 11131
  store float %8488, float addrspace(1)* %8493, align 4		; visa id: 11132
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11133

8494:                                             ; preds = %8471
; BB864 :
  %8495 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8478, i32 %8479, i32 %47, i32 %48)
  %8496 = extractvalue { i32, i32 } %8495, 0		; visa id: 11135
  %8497 = extractvalue { i32, i32 } %8495, 1		; visa id: 11135
  %8498 = insertelement <2 x i32> undef, i32 %8496, i32 0		; visa id: 11142
  %8499 = insertelement <2 x i32> %8498, i32 %8497, i32 1		; visa id: 11143
  %8500 = bitcast <2 x i32> %8499 to i64		; visa id: 11144
  %8501 = shl i64 %8500, 2		; visa id: 11148
  %8502 = add i64 %.in399, %8501		; visa id: 11149
  %8503 = shl nsw i64 %8480, 2		; visa id: 11150
  %8504 = add i64 %8502, %8503		; visa id: 11151
  %8505 = inttoptr i64 %8504 to float addrspace(4)*		; visa id: 11152
  %8506 = addrspacecast float addrspace(4)* %8505 to float addrspace(1)*		; visa id: 11152
  %8507 = load float, float addrspace(1)* %8506, align 4		; visa id: 11153
  %8508 = fmul reassoc nsz arcp contract float %8507, %4, !spirv.Decorations !618		; visa id: 11154
  %8509 = fadd reassoc nsz arcp contract float %8488, %8508, !spirv.Decorations !618		; visa id: 11155
  %8510 = shl i64 %8487, 2		; visa id: 11156
  %8511 = add i64 %.in, %8510		; visa id: 11157
  %8512 = inttoptr i64 %8511 to float addrspace(4)*		; visa id: 11158
  %8513 = addrspacecast float addrspace(4)* %8512 to float addrspace(1)*		; visa id: 11158
  store float %8509, float addrspace(1)* %8513, align 4		; visa id: 11159
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11160

._crit_edge70.2.8:                                ; preds = %.._crit_edge70.2.8_crit_edge, %8494, %8489
; BB865 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11161
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11161
  %8514 = insertelement <2 x i32> %6617, i32 %8336, i64 1		; visa id: 11161
  store <2 x i32> %8514, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11164
  br label %._crit_edge370, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11166

._crit_edge370:                                   ; preds = %._crit_edge370.._crit_edge370_crit_edge, %._crit_edge70.2.8
; BB866 :
  %8515 = phi i32 [ 0, %._crit_edge70.2.8 ], [ %8524, %._crit_edge370.._crit_edge370_crit_edge ]
  %8516 = zext i32 %8515 to i64		; visa id: 11167
  %8517 = shl nuw nsw i64 %8516, 2		; visa id: 11168
  %8518 = add i64 %6430, %8517		; visa id: 11169
  %8519 = inttoptr i64 %8518 to i32*		; visa id: 11170
  %8520 = load i32, i32* %8519, align 4, !noalias !642		; visa id: 11170
  %8521 = add i64 %6426, %8517		; visa id: 11171
  %8522 = inttoptr i64 %8521 to i32*		; visa id: 11172
  store i32 %8520, i32* %8522, align 4, !alias.scope !642		; visa id: 11172
  %8523 = icmp eq i32 %8515, 0		; visa id: 11173
  br i1 %8523, label %._crit_edge370.._crit_edge370_crit_edge, label %8525, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11174

._crit_edge370.._crit_edge370_crit_edge:          ; preds = %._crit_edge370
; BB867 :
  %8524 = add nuw nsw i32 %8515, 1, !spirv.Decorations !631		; visa id: 11176
  br label %._crit_edge370, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11177

8525:                                             ; preds = %._crit_edge370
; BB868 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11179
  %8526 = load i64, i64* %6446, align 8		; visa id: 11179
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11180
  %8527 = icmp slt i32 %6616, %const_reg_dword
  %8528 = icmp slt i32 %8336, %const_reg_dword1		; visa id: 11180
  %8529 = and i1 %8527, %8528		; visa id: 11181
  br i1 %8529, label %8530, label %..preheader1.8_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11183

..preheader1.8_crit_edge:                         ; preds = %8525
; BB:
  br label %.preheader1.8, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8530:                                             ; preds = %8525
; BB870 :
  %8531 = bitcast i64 %8526 to <2 x i32>		; visa id: 11185
  %8532 = extractelement <2 x i32> %8531, i32 0		; visa id: 11187
  %8533 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8532, i32 1
  %8534 = bitcast <2 x i32> %8533 to i64		; visa id: 11187
  %8535 = ashr exact i64 %8534, 32		; visa id: 11188
  %8536 = bitcast i64 %8535 to <2 x i32>		; visa id: 11189
  %8537 = extractelement <2 x i32> %8536, i32 0		; visa id: 11193
  %8538 = extractelement <2 x i32> %8536, i32 1		; visa id: 11193
  %8539 = ashr i64 %8526, 32		; visa id: 11193
  %8540 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8537, i32 %8538, i32 %50, i32 %51)
  %8541 = extractvalue { i32, i32 } %8540, 0		; visa id: 11194
  %8542 = extractvalue { i32, i32 } %8540, 1		; visa id: 11194
  %8543 = insertelement <2 x i32> undef, i32 %8541, i32 0		; visa id: 11201
  %8544 = insertelement <2 x i32> %8543, i32 %8542, i32 1		; visa id: 11202
  %8545 = bitcast <2 x i32> %8544 to i64		; visa id: 11203
  %8546 = add nsw i64 %8545, %8539, !spirv.Decorations !649		; visa id: 11207
  %8547 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !618		; visa id: 11208
  br i1 %86, label %8553, label %8548, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11209

8548:                                             ; preds = %8530
; BB871 :
  %8549 = shl i64 %8546, 2		; visa id: 11211
  %8550 = add i64 %.in, %8549		; visa id: 11212
  %8551 = inttoptr i64 %8550 to float addrspace(4)*		; visa id: 11213
  %8552 = addrspacecast float addrspace(4)* %8551 to float addrspace(1)*		; visa id: 11213
  store float %8547, float addrspace(1)* %8552, align 4		; visa id: 11214
  br label %.preheader1.8, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11215

8553:                                             ; preds = %8530
; BB872 :
  %8554 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8537, i32 %8538, i32 %47, i32 %48)
  %8555 = extractvalue { i32, i32 } %8554, 0		; visa id: 11217
  %8556 = extractvalue { i32, i32 } %8554, 1		; visa id: 11217
  %8557 = insertelement <2 x i32> undef, i32 %8555, i32 0		; visa id: 11224
  %8558 = insertelement <2 x i32> %8557, i32 %8556, i32 1		; visa id: 11225
  %8559 = bitcast <2 x i32> %8558 to i64		; visa id: 11226
  %8560 = shl i64 %8559, 2		; visa id: 11230
  %8561 = add i64 %.in399, %8560		; visa id: 11231
  %8562 = shl nsw i64 %8539, 2		; visa id: 11232
  %8563 = add i64 %8561, %8562		; visa id: 11233
  %8564 = inttoptr i64 %8563 to float addrspace(4)*		; visa id: 11234
  %8565 = addrspacecast float addrspace(4)* %8564 to float addrspace(1)*		; visa id: 11234
  %8566 = load float, float addrspace(1)* %8565, align 4		; visa id: 11235
  %8567 = fmul reassoc nsz arcp contract float %8566, %4, !spirv.Decorations !618		; visa id: 11236
  %8568 = fadd reassoc nsz arcp contract float %8547, %8567, !spirv.Decorations !618		; visa id: 11237
  %8569 = shl i64 %8546, 2		; visa id: 11238
  %8570 = add i64 %.in, %8569		; visa id: 11239
  %8571 = inttoptr i64 %8570 to float addrspace(4)*		; visa id: 11240
  %8572 = addrspacecast float addrspace(4)* %8571 to float addrspace(1)*		; visa id: 11240
  store float %8568, float addrspace(1)* %8572, align 4		; visa id: 11241
  br label %.preheader1.8, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11242

.preheader1.8:                                    ; preds = %..preheader1.8_crit_edge, %8553, %8548
; BB873 :
  %8573 = add i32 %69, 9		; visa id: 11243
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11244
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11244
  %8574 = insertelement <2 x i32> %6432, i32 %8573, i64 1		; visa id: 11244
  store <2 x i32> %8574, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11247
  br label %._crit_edge371, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11249

._crit_edge371:                                   ; preds = %._crit_edge371.._crit_edge371_crit_edge, %.preheader1.8
; BB874 :
  %8575 = phi i32 [ 0, %.preheader1.8 ], [ %8584, %._crit_edge371.._crit_edge371_crit_edge ]
  %8576 = zext i32 %8575 to i64		; visa id: 11250
  %8577 = shl nuw nsw i64 %8576, 2		; visa id: 11251
  %8578 = add i64 %6430, %8577		; visa id: 11252
  %8579 = inttoptr i64 %8578 to i32*		; visa id: 11253
  %8580 = load i32, i32* %8579, align 4, !noalias !642		; visa id: 11253
  %8581 = add i64 %6426, %8577		; visa id: 11254
  %8582 = inttoptr i64 %8581 to i32*		; visa id: 11255
  store i32 %8580, i32* %8582, align 4, !alias.scope !642		; visa id: 11255
  %8583 = icmp eq i32 %8575, 0		; visa id: 11256
  br i1 %8583, label %._crit_edge371.._crit_edge371_crit_edge, label %8585, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11257

._crit_edge371.._crit_edge371_crit_edge:          ; preds = %._crit_edge371
; BB875 :
  %8584 = add nuw nsw i32 %8575, 1, !spirv.Decorations !631		; visa id: 11259
  br label %._crit_edge371, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11260

8585:                                             ; preds = %._crit_edge371
; BB876 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11262
  %8586 = load i64, i64* %6446, align 8		; visa id: 11262
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11263
  %8587 = icmp slt i32 %8573, %const_reg_dword1		; visa id: 11263
  %8588 = icmp slt i32 %65, %const_reg_dword
  %8589 = and i1 %8588, %8587		; visa id: 11264
  br i1 %8589, label %8590, label %.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11266

.._crit_edge70.9_crit_edge:                       ; preds = %8585
; BB:
  br label %._crit_edge70.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8590:                                             ; preds = %8585
; BB878 :
  %8591 = bitcast i64 %8586 to <2 x i32>		; visa id: 11268
  %8592 = extractelement <2 x i32> %8591, i32 0		; visa id: 11270
  %8593 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8592, i32 1
  %8594 = bitcast <2 x i32> %8593 to i64		; visa id: 11270
  %8595 = ashr exact i64 %8594, 32		; visa id: 11271
  %8596 = bitcast i64 %8595 to <2 x i32>		; visa id: 11272
  %8597 = extractelement <2 x i32> %8596, i32 0		; visa id: 11276
  %8598 = extractelement <2 x i32> %8596, i32 1		; visa id: 11276
  %8599 = ashr i64 %8586, 32		; visa id: 11276
  %8600 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8597, i32 %8598, i32 %50, i32 %51)
  %8601 = extractvalue { i32, i32 } %8600, 0		; visa id: 11277
  %8602 = extractvalue { i32, i32 } %8600, 1		; visa id: 11277
  %8603 = insertelement <2 x i32> undef, i32 %8601, i32 0		; visa id: 11284
  %8604 = insertelement <2 x i32> %8603, i32 %8602, i32 1		; visa id: 11285
  %8605 = bitcast <2 x i32> %8604 to i64		; visa id: 11286
  %8606 = add nsw i64 %8605, %8599, !spirv.Decorations !649		; visa id: 11290
  %8607 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !618		; visa id: 11291
  br i1 %86, label %8613, label %8608, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11292

8608:                                             ; preds = %8590
; BB879 :
  %8609 = shl i64 %8606, 2		; visa id: 11294
  %8610 = add i64 %.in, %8609		; visa id: 11295
  %8611 = inttoptr i64 %8610 to float addrspace(4)*		; visa id: 11296
  %8612 = addrspacecast float addrspace(4)* %8611 to float addrspace(1)*		; visa id: 11296
  store float %8607, float addrspace(1)* %8612, align 4		; visa id: 11297
  br label %._crit_edge70.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11298

8613:                                             ; preds = %8590
; BB880 :
  %8614 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8597, i32 %8598, i32 %47, i32 %48)
  %8615 = extractvalue { i32, i32 } %8614, 0		; visa id: 11300
  %8616 = extractvalue { i32, i32 } %8614, 1		; visa id: 11300
  %8617 = insertelement <2 x i32> undef, i32 %8615, i32 0		; visa id: 11307
  %8618 = insertelement <2 x i32> %8617, i32 %8616, i32 1		; visa id: 11308
  %8619 = bitcast <2 x i32> %8618 to i64		; visa id: 11309
  %8620 = shl i64 %8619, 2		; visa id: 11313
  %8621 = add i64 %.in399, %8620		; visa id: 11314
  %8622 = shl nsw i64 %8599, 2		; visa id: 11315
  %8623 = add i64 %8621, %8622		; visa id: 11316
  %8624 = inttoptr i64 %8623 to float addrspace(4)*		; visa id: 11317
  %8625 = addrspacecast float addrspace(4)* %8624 to float addrspace(1)*		; visa id: 11317
  %8626 = load float, float addrspace(1)* %8625, align 4		; visa id: 11318
  %8627 = fmul reassoc nsz arcp contract float %8626, %4, !spirv.Decorations !618		; visa id: 11319
  %8628 = fadd reassoc nsz arcp contract float %8607, %8627, !spirv.Decorations !618		; visa id: 11320
  %8629 = shl i64 %8606, 2		; visa id: 11321
  %8630 = add i64 %.in, %8629		; visa id: 11322
  %8631 = inttoptr i64 %8630 to float addrspace(4)*		; visa id: 11323
  %8632 = addrspacecast float addrspace(4)* %8631 to float addrspace(1)*		; visa id: 11323
  store float %8628, float addrspace(1)* %8632, align 4		; visa id: 11324
  br label %._crit_edge70.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11325

._crit_edge70.9:                                  ; preds = %.._crit_edge70.9_crit_edge, %8613, %8608
; BB881 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11326
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11326
  %8633 = insertelement <2 x i32> %6495, i32 %8573, i64 1		; visa id: 11326
  store <2 x i32> %8633, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11329
  br label %._crit_edge372, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11331

._crit_edge372:                                   ; preds = %._crit_edge372.._crit_edge372_crit_edge, %._crit_edge70.9
; BB882 :
  %8634 = phi i32 [ 0, %._crit_edge70.9 ], [ %8643, %._crit_edge372.._crit_edge372_crit_edge ]
  %8635 = zext i32 %8634 to i64		; visa id: 11332
  %8636 = shl nuw nsw i64 %8635, 2		; visa id: 11333
  %8637 = add i64 %6430, %8636		; visa id: 11334
  %8638 = inttoptr i64 %8637 to i32*		; visa id: 11335
  %8639 = load i32, i32* %8638, align 4, !noalias !642		; visa id: 11335
  %8640 = add i64 %6426, %8636		; visa id: 11336
  %8641 = inttoptr i64 %8640 to i32*		; visa id: 11337
  store i32 %8639, i32* %8641, align 4, !alias.scope !642		; visa id: 11337
  %8642 = icmp eq i32 %8634, 0		; visa id: 11338
  br i1 %8642, label %._crit_edge372.._crit_edge372_crit_edge, label %8644, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11339

._crit_edge372.._crit_edge372_crit_edge:          ; preds = %._crit_edge372
; BB883 :
  %8643 = add nuw nsw i32 %8634, 1, !spirv.Decorations !631		; visa id: 11341
  br label %._crit_edge372, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11342

8644:                                             ; preds = %._crit_edge372
; BB884 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11344
  %8645 = load i64, i64* %6446, align 8		; visa id: 11344
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11345
  %8646 = icmp slt i32 %6494, %const_reg_dword
  %8647 = icmp slt i32 %8573, %const_reg_dword1		; visa id: 11345
  %8648 = and i1 %8646, %8647		; visa id: 11346
  br i1 %8648, label %8649, label %.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11348

.._crit_edge70.1.9_crit_edge:                     ; preds = %8644
; BB:
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8649:                                             ; preds = %8644
; BB886 :
  %8650 = bitcast i64 %8645 to <2 x i32>		; visa id: 11350
  %8651 = extractelement <2 x i32> %8650, i32 0		; visa id: 11352
  %8652 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8651, i32 1
  %8653 = bitcast <2 x i32> %8652 to i64		; visa id: 11352
  %8654 = ashr exact i64 %8653, 32		; visa id: 11353
  %8655 = bitcast i64 %8654 to <2 x i32>		; visa id: 11354
  %8656 = extractelement <2 x i32> %8655, i32 0		; visa id: 11358
  %8657 = extractelement <2 x i32> %8655, i32 1		; visa id: 11358
  %8658 = ashr i64 %8645, 32		; visa id: 11358
  %8659 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8656, i32 %8657, i32 %50, i32 %51)
  %8660 = extractvalue { i32, i32 } %8659, 0		; visa id: 11359
  %8661 = extractvalue { i32, i32 } %8659, 1		; visa id: 11359
  %8662 = insertelement <2 x i32> undef, i32 %8660, i32 0		; visa id: 11366
  %8663 = insertelement <2 x i32> %8662, i32 %8661, i32 1		; visa id: 11367
  %8664 = bitcast <2 x i32> %8663 to i64		; visa id: 11368
  %8665 = add nsw i64 %8664, %8658, !spirv.Decorations !649		; visa id: 11372
  %8666 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !618		; visa id: 11373
  br i1 %86, label %8672, label %8667, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11374

8667:                                             ; preds = %8649
; BB887 :
  %8668 = shl i64 %8665, 2		; visa id: 11376
  %8669 = add i64 %.in, %8668		; visa id: 11377
  %8670 = inttoptr i64 %8669 to float addrspace(4)*		; visa id: 11378
  %8671 = addrspacecast float addrspace(4)* %8670 to float addrspace(1)*		; visa id: 11378
  store float %8666, float addrspace(1)* %8671, align 4		; visa id: 11379
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11380

8672:                                             ; preds = %8649
; BB888 :
  %8673 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8656, i32 %8657, i32 %47, i32 %48)
  %8674 = extractvalue { i32, i32 } %8673, 0		; visa id: 11382
  %8675 = extractvalue { i32, i32 } %8673, 1		; visa id: 11382
  %8676 = insertelement <2 x i32> undef, i32 %8674, i32 0		; visa id: 11389
  %8677 = insertelement <2 x i32> %8676, i32 %8675, i32 1		; visa id: 11390
  %8678 = bitcast <2 x i32> %8677 to i64		; visa id: 11391
  %8679 = shl i64 %8678, 2		; visa id: 11395
  %8680 = add i64 %.in399, %8679		; visa id: 11396
  %8681 = shl nsw i64 %8658, 2		; visa id: 11397
  %8682 = add i64 %8680, %8681		; visa id: 11398
  %8683 = inttoptr i64 %8682 to float addrspace(4)*		; visa id: 11399
  %8684 = addrspacecast float addrspace(4)* %8683 to float addrspace(1)*		; visa id: 11399
  %8685 = load float, float addrspace(1)* %8684, align 4		; visa id: 11400
  %8686 = fmul reassoc nsz arcp contract float %8685, %4, !spirv.Decorations !618		; visa id: 11401
  %8687 = fadd reassoc nsz arcp contract float %8666, %8686, !spirv.Decorations !618		; visa id: 11402
  %8688 = shl i64 %8665, 2		; visa id: 11403
  %8689 = add i64 %.in, %8688		; visa id: 11404
  %8690 = inttoptr i64 %8689 to float addrspace(4)*		; visa id: 11405
  %8691 = addrspacecast float addrspace(4)* %8690 to float addrspace(1)*		; visa id: 11405
  store float %8687, float addrspace(1)* %8691, align 4		; visa id: 11406
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11407

._crit_edge70.1.9:                                ; preds = %.._crit_edge70.1.9_crit_edge, %8672, %8667
; BB889 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11408
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11408
  %8692 = insertelement <2 x i32> %6556, i32 %8573, i64 1		; visa id: 11408
  store <2 x i32> %8692, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11411
  br label %._crit_edge373, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11413

._crit_edge373:                                   ; preds = %._crit_edge373.._crit_edge373_crit_edge, %._crit_edge70.1.9
; BB890 :
  %8693 = phi i32 [ 0, %._crit_edge70.1.9 ], [ %8702, %._crit_edge373.._crit_edge373_crit_edge ]
  %8694 = zext i32 %8693 to i64		; visa id: 11414
  %8695 = shl nuw nsw i64 %8694, 2		; visa id: 11415
  %8696 = add i64 %6430, %8695		; visa id: 11416
  %8697 = inttoptr i64 %8696 to i32*		; visa id: 11417
  %8698 = load i32, i32* %8697, align 4, !noalias !642		; visa id: 11417
  %8699 = add i64 %6426, %8695		; visa id: 11418
  %8700 = inttoptr i64 %8699 to i32*		; visa id: 11419
  store i32 %8698, i32* %8700, align 4, !alias.scope !642		; visa id: 11419
  %8701 = icmp eq i32 %8693, 0		; visa id: 11420
  br i1 %8701, label %._crit_edge373.._crit_edge373_crit_edge, label %8703, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11421

._crit_edge373.._crit_edge373_crit_edge:          ; preds = %._crit_edge373
; BB891 :
  %8702 = add nuw nsw i32 %8693, 1, !spirv.Decorations !631		; visa id: 11423
  br label %._crit_edge373, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11424

8703:                                             ; preds = %._crit_edge373
; BB892 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11426
  %8704 = load i64, i64* %6446, align 8		; visa id: 11426
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11427
  %8705 = icmp slt i32 %6555, %const_reg_dword
  %8706 = icmp slt i32 %8573, %const_reg_dword1		; visa id: 11427
  %8707 = and i1 %8705, %8706		; visa id: 11428
  br i1 %8707, label %8708, label %.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11430

.._crit_edge70.2.9_crit_edge:                     ; preds = %8703
; BB:
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8708:                                             ; preds = %8703
; BB894 :
  %8709 = bitcast i64 %8704 to <2 x i32>		; visa id: 11432
  %8710 = extractelement <2 x i32> %8709, i32 0		; visa id: 11434
  %8711 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8710, i32 1
  %8712 = bitcast <2 x i32> %8711 to i64		; visa id: 11434
  %8713 = ashr exact i64 %8712, 32		; visa id: 11435
  %8714 = bitcast i64 %8713 to <2 x i32>		; visa id: 11436
  %8715 = extractelement <2 x i32> %8714, i32 0		; visa id: 11440
  %8716 = extractelement <2 x i32> %8714, i32 1		; visa id: 11440
  %8717 = ashr i64 %8704, 32		; visa id: 11440
  %8718 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8715, i32 %8716, i32 %50, i32 %51)
  %8719 = extractvalue { i32, i32 } %8718, 0		; visa id: 11441
  %8720 = extractvalue { i32, i32 } %8718, 1		; visa id: 11441
  %8721 = insertelement <2 x i32> undef, i32 %8719, i32 0		; visa id: 11448
  %8722 = insertelement <2 x i32> %8721, i32 %8720, i32 1		; visa id: 11449
  %8723 = bitcast <2 x i32> %8722 to i64		; visa id: 11450
  %8724 = add nsw i64 %8723, %8717, !spirv.Decorations !649		; visa id: 11454
  %8725 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !618		; visa id: 11455
  br i1 %86, label %8731, label %8726, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11456

8726:                                             ; preds = %8708
; BB895 :
  %8727 = shl i64 %8724, 2		; visa id: 11458
  %8728 = add i64 %.in, %8727		; visa id: 11459
  %8729 = inttoptr i64 %8728 to float addrspace(4)*		; visa id: 11460
  %8730 = addrspacecast float addrspace(4)* %8729 to float addrspace(1)*		; visa id: 11460
  store float %8725, float addrspace(1)* %8730, align 4		; visa id: 11461
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11462

8731:                                             ; preds = %8708
; BB896 :
  %8732 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8715, i32 %8716, i32 %47, i32 %48)
  %8733 = extractvalue { i32, i32 } %8732, 0		; visa id: 11464
  %8734 = extractvalue { i32, i32 } %8732, 1		; visa id: 11464
  %8735 = insertelement <2 x i32> undef, i32 %8733, i32 0		; visa id: 11471
  %8736 = insertelement <2 x i32> %8735, i32 %8734, i32 1		; visa id: 11472
  %8737 = bitcast <2 x i32> %8736 to i64		; visa id: 11473
  %8738 = shl i64 %8737, 2		; visa id: 11477
  %8739 = add i64 %.in399, %8738		; visa id: 11478
  %8740 = shl nsw i64 %8717, 2		; visa id: 11479
  %8741 = add i64 %8739, %8740		; visa id: 11480
  %8742 = inttoptr i64 %8741 to float addrspace(4)*		; visa id: 11481
  %8743 = addrspacecast float addrspace(4)* %8742 to float addrspace(1)*		; visa id: 11481
  %8744 = load float, float addrspace(1)* %8743, align 4		; visa id: 11482
  %8745 = fmul reassoc nsz arcp contract float %8744, %4, !spirv.Decorations !618		; visa id: 11483
  %8746 = fadd reassoc nsz arcp contract float %8725, %8745, !spirv.Decorations !618		; visa id: 11484
  %8747 = shl i64 %8724, 2		; visa id: 11485
  %8748 = add i64 %.in, %8747		; visa id: 11486
  %8749 = inttoptr i64 %8748 to float addrspace(4)*		; visa id: 11487
  %8750 = addrspacecast float addrspace(4)* %8749 to float addrspace(1)*		; visa id: 11487
  store float %8746, float addrspace(1)* %8750, align 4		; visa id: 11488
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11489

._crit_edge70.2.9:                                ; preds = %.._crit_edge70.2.9_crit_edge, %8731, %8726
; BB897 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11490
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11490
  %8751 = insertelement <2 x i32> %6617, i32 %8573, i64 1		; visa id: 11490
  store <2 x i32> %8751, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11493
  br label %._crit_edge374, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11495

._crit_edge374:                                   ; preds = %._crit_edge374.._crit_edge374_crit_edge, %._crit_edge70.2.9
; BB898 :
  %8752 = phi i32 [ 0, %._crit_edge70.2.9 ], [ %8761, %._crit_edge374.._crit_edge374_crit_edge ]
  %8753 = zext i32 %8752 to i64		; visa id: 11496
  %8754 = shl nuw nsw i64 %8753, 2		; visa id: 11497
  %8755 = add i64 %6430, %8754		; visa id: 11498
  %8756 = inttoptr i64 %8755 to i32*		; visa id: 11499
  %8757 = load i32, i32* %8756, align 4, !noalias !642		; visa id: 11499
  %8758 = add i64 %6426, %8754		; visa id: 11500
  %8759 = inttoptr i64 %8758 to i32*		; visa id: 11501
  store i32 %8757, i32* %8759, align 4, !alias.scope !642		; visa id: 11501
  %8760 = icmp eq i32 %8752, 0		; visa id: 11502
  br i1 %8760, label %._crit_edge374.._crit_edge374_crit_edge, label %8762, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11503

._crit_edge374.._crit_edge374_crit_edge:          ; preds = %._crit_edge374
; BB899 :
  %8761 = add nuw nsw i32 %8752, 1, !spirv.Decorations !631		; visa id: 11505
  br label %._crit_edge374, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11506

8762:                                             ; preds = %._crit_edge374
; BB900 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11508
  %8763 = load i64, i64* %6446, align 8		; visa id: 11508
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11509
  %8764 = icmp slt i32 %6616, %const_reg_dword
  %8765 = icmp slt i32 %8573, %const_reg_dword1		; visa id: 11509
  %8766 = and i1 %8764, %8765		; visa id: 11510
  br i1 %8766, label %8767, label %..preheader1.9_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11512

..preheader1.9_crit_edge:                         ; preds = %8762
; BB:
  br label %.preheader1.9, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8767:                                             ; preds = %8762
; BB902 :
  %8768 = bitcast i64 %8763 to <2 x i32>		; visa id: 11514
  %8769 = extractelement <2 x i32> %8768, i32 0		; visa id: 11516
  %8770 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8769, i32 1
  %8771 = bitcast <2 x i32> %8770 to i64		; visa id: 11516
  %8772 = ashr exact i64 %8771, 32		; visa id: 11517
  %8773 = bitcast i64 %8772 to <2 x i32>		; visa id: 11518
  %8774 = extractelement <2 x i32> %8773, i32 0		; visa id: 11522
  %8775 = extractelement <2 x i32> %8773, i32 1		; visa id: 11522
  %8776 = ashr i64 %8763, 32		; visa id: 11522
  %8777 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8774, i32 %8775, i32 %50, i32 %51)
  %8778 = extractvalue { i32, i32 } %8777, 0		; visa id: 11523
  %8779 = extractvalue { i32, i32 } %8777, 1		; visa id: 11523
  %8780 = insertelement <2 x i32> undef, i32 %8778, i32 0		; visa id: 11530
  %8781 = insertelement <2 x i32> %8780, i32 %8779, i32 1		; visa id: 11531
  %8782 = bitcast <2 x i32> %8781 to i64		; visa id: 11532
  %8783 = add nsw i64 %8782, %8776, !spirv.Decorations !649		; visa id: 11536
  %8784 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !618		; visa id: 11537
  br i1 %86, label %8790, label %8785, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11538

8785:                                             ; preds = %8767
; BB903 :
  %8786 = shl i64 %8783, 2		; visa id: 11540
  %8787 = add i64 %.in, %8786		; visa id: 11541
  %8788 = inttoptr i64 %8787 to float addrspace(4)*		; visa id: 11542
  %8789 = addrspacecast float addrspace(4)* %8788 to float addrspace(1)*		; visa id: 11542
  store float %8784, float addrspace(1)* %8789, align 4		; visa id: 11543
  br label %.preheader1.9, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11544

8790:                                             ; preds = %8767
; BB904 :
  %8791 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8774, i32 %8775, i32 %47, i32 %48)
  %8792 = extractvalue { i32, i32 } %8791, 0		; visa id: 11546
  %8793 = extractvalue { i32, i32 } %8791, 1		; visa id: 11546
  %8794 = insertelement <2 x i32> undef, i32 %8792, i32 0		; visa id: 11553
  %8795 = insertelement <2 x i32> %8794, i32 %8793, i32 1		; visa id: 11554
  %8796 = bitcast <2 x i32> %8795 to i64		; visa id: 11555
  %8797 = shl i64 %8796, 2		; visa id: 11559
  %8798 = add i64 %.in399, %8797		; visa id: 11560
  %8799 = shl nsw i64 %8776, 2		; visa id: 11561
  %8800 = add i64 %8798, %8799		; visa id: 11562
  %8801 = inttoptr i64 %8800 to float addrspace(4)*		; visa id: 11563
  %8802 = addrspacecast float addrspace(4)* %8801 to float addrspace(1)*		; visa id: 11563
  %8803 = load float, float addrspace(1)* %8802, align 4		; visa id: 11564
  %8804 = fmul reassoc nsz arcp contract float %8803, %4, !spirv.Decorations !618		; visa id: 11565
  %8805 = fadd reassoc nsz arcp contract float %8784, %8804, !spirv.Decorations !618		; visa id: 11566
  %8806 = shl i64 %8783, 2		; visa id: 11567
  %8807 = add i64 %.in, %8806		; visa id: 11568
  %8808 = inttoptr i64 %8807 to float addrspace(4)*		; visa id: 11569
  %8809 = addrspacecast float addrspace(4)* %8808 to float addrspace(1)*		; visa id: 11569
  store float %8805, float addrspace(1)* %8809, align 4		; visa id: 11570
  br label %.preheader1.9, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11571

.preheader1.9:                                    ; preds = %..preheader1.9_crit_edge, %8790, %8785
; BB905 :
  %8810 = add i32 %69, 10		; visa id: 11572
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11573
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11573
  %8811 = insertelement <2 x i32> %6432, i32 %8810, i64 1		; visa id: 11573
  store <2 x i32> %8811, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11576
  br label %._crit_edge375, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11578

._crit_edge375:                                   ; preds = %._crit_edge375.._crit_edge375_crit_edge, %.preheader1.9
; BB906 :
  %8812 = phi i32 [ 0, %.preheader1.9 ], [ %8821, %._crit_edge375.._crit_edge375_crit_edge ]
  %8813 = zext i32 %8812 to i64		; visa id: 11579
  %8814 = shl nuw nsw i64 %8813, 2		; visa id: 11580
  %8815 = add i64 %6430, %8814		; visa id: 11581
  %8816 = inttoptr i64 %8815 to i32*		; visa id: 11582
  %8817 = load i32, i32* %8816, align 4, !noalias !642		; visa id: 11582
  %8818 = add i64 %6426, %8814		; visa id: 11583
  %8819 = inttoptr i64 %8818 to i32*		; visa id: 11584
  store i32 %8817, i32* %8819, align 4, !alias.scope !642		; visa id: 11584
  %8820 = icmp eq i32 %8812, 0		; visa id: 11585
  br i1 %8820, label %._crit_edge375.._crit_edge375_crit_edge, label %8822, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11586

._crit_edge375.._crit_edge375_crit_edge:          ; preds = %._crit_edge375
; BB907 :
  %8821 = add nuw nsw i32 %8812, 1, !spirv.Decorations !631		; visa id: 11588
  br label %._crit_edge375, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11589

8822:                                             ; preds = %._crit_edge375
; BB908 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11591
  %8823 = load i64, i64* %6446, align 8		; visa id: 11591
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11592
  %8824 = icmp slt i32 %8810, %const_reg_dword1		; visa id: 11592
  %8825 = icmp slt i32 %65, %const_reg_dword
  %8826 = and i1 %8825, %8824		; visa id: 11593
  br i1 %8826, label %8827, label %.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11595

.._crit_edge70.10_crit_edge:                      ; preds = %8822
; BB:
  br label %._crit_edge70.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8827:                                             ; preds = %8822
; BB910 :
  %8828 = bitcast i64 %8823 to <2 x i32>		; visa id: 11597
  %8829 = extractelement <2 x i32> %8828, i32 0		; visa id: 11599
  %8830 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8829, i32 1
  %8831 = bitcast <2 x i32> %8830 to i64		; visa id: 11599
  %8832 = ashr exact i64 %8831, 32		; visa id: 11600
  %8833 = bitcast i64 %8832 to <2 x i32>		; visa id: 11601
  %8834 = extractelement <2 x i32> %8833, i32 0		; visa id: 11605
  %8835 = extractelement <2 x i32> %8833, i32 1		; visa id: 11605
  %8836 = ashr i64 %8823, 32		; visa id: 11605
  %8837 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8834, i32 %8835, i32 %50, i32 %51)
  %8838 = extractvalue { i32, i32 } %8837, 0		; visa id: 11606
  %8839 = extractvalue { i32, i32 } %8837, 1		; visa id: 11606
  %8840 = insertelement <2 x i32> undef, i32 %8838, i32 0		; visa id: 11613
  %8841 = insertelement <2 x i32> %8840, i32 %8839, i32 1		; visa id: 11614
  %8842 = bitcast <2 x i32> %8841 to i64		; visa id: 11615
  %8843 = add nsw i64 %8842, %8836, !spirv.Decorations !649		; visa id: 11619
  %8844 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !618		; visa id: 11620
  br i1 %86, label %8850, label %8845, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11621

8845:                                             ; preds = %8827
; BB911 :
  %8846 = shl i64 %8843, 2		; visa id: 11623
  %8847 = add i64 %.in, %8846		; visa id: 11624
  %8848 = inttoptr i64 %8847 to float addrspace(4)*		; visa id: 11625
  %8849 = addrspacecast float addrspace(4)* %8848 to float addrspace(1)*		; visa id: 11625
  store float %8844, float addrspace(1)* %8849, align 4		; visa id: 11626
  br label %._crit_edge70.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11627

8850:                                             ; preds = %8827
; BB912 :
  %8851 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8834, i32 %8835, i32 %47, i32 %48)
  %8852 = extractvalue { i32, i32 } %8851, 0		; visa id: 11629
  %8853 = extractvalue { i32, i32 } %8851, 1		; visa id: 11629
  %8854 = insertelement <2 x i32> undef, i32 %8852, i32 0		; visa id: 11636
  %8855 = insertelement <2 x i32> %8854, i32 %8853, i32 1		; visa id: 11637
  %8856 = bitcast <2 x i32> %8855 to i64		; visa id: 11638
  %8857 = shl i64 %8856, 2		; visa id: 11642
  %8858 = add i64 %.in399, %8857		; visa id: 11643
  %8859 = shl nsw i64 %8836, 2		; visa id: 11644
  %8860 = add i64 %8858, %8859		; visa id: 11645
  %8861 = inttoptr i64 %8860 to float addrspace(4)*		; visa id: 11646
  %8862 = addrspacecast float addrspace(4)* %8861 to float addrspace(1)*		; visa id: 11646
  %8863 = load float, float addrspace(1)* %8862, align 4		; visa id: 11647
  %8864 = fmul reassoc nsz arcp contract float %8863, %4, !spirv.Decorations !618		; visa id: 11648
  %8865 = fadd reassoc nsz arcp contract float %8844, %8864, !spirv.Decorations !618		; visa id: 11649
  %8866 = shl i64 %8843, 2		; visa id: 11650
  %8867 = add i64 %.in, %8866		; visa id: 11651
  %8868 = inttoptr i64 %8867 to float addrspace(4)*		; visa id: 11652
  %8869 = addrspacecast float addrspace(4)* %8868 to float addrspace(1)*		; visa id: 11652
  store float %8865, float addrspace(1)* %8869, align 4		; visa id: 11653
  br label %._crit_edge70.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11654

._crit_edge70.10:                                 ; preds = %.._crit_edge70.10_crit_edge, %8850, %8845
; BB913 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11655
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11655
  %8870 = insertelement <2 x i32> %6495, i32 %8810, i64 1		; visa id: 11655
  store <2 x i32> %8870, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11658
  br label %._crit_edge376, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11660

._crit_edge376:                                   ; preds = %._crit_edge376.._crit_edge376_crit_edge, %._crit_edge70.10
; BB914 :
  %8871 = phi i32 [ 0, %._crit_edge70.10 ], [ %8880, %._crit_edge376.._crit_edge376_crit_edge ]
  %8872 = zext i32 %8871 to i64		; visa id: 11661
  %8873 = shl nuw nsw i64 %8872, 2		; visa id: 11662
  %8874 = add i64 %6430, %8873		; visa id: 11663
  %8875 = inttoptr i64 %8874 to i32*		; visa id: 11664
  %8876 = load i32, i32* %8875, align 4, !noalias !642		; visa id: 11664
  %8877 = add i64 %6426, %8873		; visa id: 11665
  %8878 = inttoptr i64 %8877 to i32*		; visa id: 11666
  store i32 %8876, i32* %8878, align 4, !alias.scope !642		; visa id: 11666
  %8879 = icmp eq i32 %8871, 0		; visa id: 11667
  br i1 %8879, label %._crit_edge376.._crit_edge376_crit_edge, label %8881, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11668

._crit_edge376.._crit_edge376_crit_edge:          ; preds = %._crit_edge376
; BB915 :
  %8880 = add nuw nsw i32 %8871, 1, !spirv.Decorations !631		; visa id: 11670
  br label %._crit_edge376, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11671

8881:                                             ; preds = %._crit_edge376
; BB916 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11673
  %8882 = load i64, i64* %6446, align 8		; visa id: 11673
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11674
  %8883 = icmp slt i32 %6494, %const_reg_dword
  %8884 = icmp slt i32 %8810, %const_reg_dword1		; visa id: 11674
  %8885 = and i1 %8883, %8884		; visa id: 11675
  br i1 %8885, label %8886, label %.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11677

.._crit_edge70.1.10_crit_edge:                    ; preds = %8881
; BB:
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8886:                                             ; preds = %8881
; BB918 :
  %8887 = bitcast i64 %8882 to <2 x i32>		; visa id: 11679
  %8888 = extractelement <2 x i32> %8887, i32 0		; visa id: 11681
  %8889 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8888, i32 1
  %8890 = bitcast <2 x i32> %8889 to i64		; visa id: 11681
  %8891 = ashr exact i64 %8890, 32		; visa id: 11682
  %8892 = bitcast i64 %8891 to <2 x i32>		; visa id: 11683
  %8893 = extractelement <2 x i32> %8892, i32 0		; visa id: 11687
  %8894 = extractelement <2 x i32> %8892, i32 1		; visa id: 11687
  %8895 = ashr i64 %8882, 32		; visa id: 11687
  %8896 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8893, i32 %8894, i32 %50, i32 %51)
  %8897 = extractvalue { i32, i32 } %8896, 0		; visa id: 11688
  %8898 = extractvalue { i32, i32 } %8896, 1		; visa id: 11688
  %8899 = insertelement <2 x i32> undef, i32 %8897, i32 0		; visa id: 11695
  %8900 = insertelement <2 x i32> %8899, i32 %8898, i32 1		; visa id: 11696
  %8901 = bitcast <2 x i32> %8900 to i64		; visa id: 11697
  %8902 = add nsw i64 %8901, %8895, !spirv.Decorations !649		; visa id: 11701
  %8903 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !618		; visa id: 11702
  br i1 %86, label %8909, label %8904, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11703

8904:                                             ; preds = %8886
; BB919 :
  %8905 = shl i64 %8902, 2		; visa id: 11705
  %8906 = add i64 %.in, %8905		; visa id: 11706
  %8907 = inttoptr i64 %8906 to float addrspace(4)*		; visa id: 11707
  %8908 = addrspacecast float addrspace(4)* %8907 to float addrspace(1)*		; visa id: 11707
  store float %8903, float addrspace(1)* %8908, align 4		; visa id: 11708
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11709

8909:                                             ; preds = %8886
; BB920 :
  %8910 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8893, i32 %8894, i32 %47, i32 %48)
  %8911 = extractvalue { i32, i32 } %8910, 0		; visa id: 11711
  %8912 = extractvalue { i32, i32 } %8910, 1		; visa id: 11711
  %8913 = insertelement <2 x i32> undef, i32 %8911, i32 0		; visa id: 11718
  %8914 = insertelement <2 x i32> %8913, i32 %8912, i32 1		; visa id: 11719
  %8915 = bitcast <2 x i32> %8914 to i64		; visa id: 11720
  %8916 = shl i64 %8915, 2		; visa id: 11724
  %8917 = add i64 %.in399, %8916		; visa id: 11725
  %8918 = shl nsw i64 %8895, 2		; visa id: 11726
  %8919 = add i64 %8917, %8918		; visa id: 11727
  %8920 = inttoptr i64 %8919 to float addrspace(4)*		; visa id: 11728
  %8921 = addrspacecast float addrspace(4)* %8920 to float addrspace(1)*		; visa id: 11728
  %8922 = load float, float addrspace(1)* %8921, align 4		; visa id: 11729
  %8923 = fmul reassoc nsz arcp contract float %8922, %4, !spirv.Decorations !618		; visa id: 11730
  %8924 = fadd reassoc nsz arcp contract float %8903, %8923, !spirv.Decorations !618		; visa id: 11731
  %8925 = shl i64 %8902, 2		; visa id: 11732
  %8926 = add i64 %.in, %8925		; visa id: 11733
  %8927 = inttoptr i64 %8926 to float addrspace(4)*		; visa id: 11734
  %8928 = addrspacecast float addrspace(4)* %8927 to float addrspace(1)*		; visa id: 11734
  store float %8924, float addrspace(1)* %8928, align 4		; visa id: 11735
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11736

._crit_edge70.1.10:                               ; preds = %.._crit_edge70.1.10_crit_edge, %8909, %8904
; BB921 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11737
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11737
  %8929 = insertelement <2 x i32> %6556, i32 %8810, i64 1		; visa id: 11737
  store <2 x i32> %8929, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11740
  br label %._crit_edge377, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11742

._crit_edge377:                                   ; preds = %._crit_edge377.._crit_edge377_crit_edge, %._crit_edge70.1.10
; BB922 :
  %8930 = phi i32 [ 0, %._crit_edge70.1.10 ], [ %8939, %._crit_edge377.._crit_edge377_crit_edge ]
  %8931 = zext i32 %8930 to i64		; visa id: 11743
  %8932 = shl nuw nsw i64 %8931, 2		; visa id: 11744
  %8933 = add i64 %6430, %8932		; visa id: 11745
  %8934 = inttoptr i64 %8933 to i32*		; visa id: 11746
  %8935 = load i32, i32* %8934, align 4, !noalias !642		; visa id: 11746
  %8936 = add i64 %6426, %8932		; visa id: 11747
  %8937 = inttoptr i64 %8936 to i32*		; visa id: 11748
  store i32 %8935, i32* %8937, align 4, !alias.scope !642		; visa id: 11748
  %8938 = icmp eq i32 %8930, 0		; visa id: 11749
  br i1 %8938, label %._crit_edge377.._crit_edge377_crit_edge, label %8940, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11750

._crit_edge377.._crit_edge377_crit_edge:          ; preds = %._crit_edge377
; BB923 :
  %8939 = add nuw nsw i32 %8930, 1, !spirv.Decorations !631		; visa id: 11752
  br label %._crit_edge377, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11753

8940:                                             ; preds = %._crit_edge377
; BB924 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11755
  %8941 = load i64, i64* %6446, align 8		; visa id: 11755
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11756
  %8942 = icmp slt i32 %6555, %const_reg_dword
  %8943 = icmp slt i32 %8810, %const_reg_dword1		; visa id: 11756
  %8944 = and i1 %8942, %8943		; visa id: 11757
  br i1 %8944, label %8945, label %.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11759

.._crit_edge70.2.10_crit_edge:                    ; preds = %8940
; BB:
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

8945:                                             ; preds = %8940
; BB926 :
  %8946 = bitcast i64 %8941 to <2 x i32>		; visa id: 11761
  %8947 = extractelement <2 x i32> %8946, i32 0		; visa id: 11763
  %8948 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %8947, i32 1
  %8949 = bitcast <2 x i32> %8948 to i64		; visa id: 11763
  %8950 = ashr exact i64 %8949, 32		; visa id: 11764
  %8951 = bitcast i64 %8950 to <2 x i32>		; visa id: 11765
  %8952 = extractelement <2 x i32> %8951, i32 0		; visa id: 11769
  %8953 = extractelement <2 x i32> %8951, i32 1		; visa id: 11769
  %8954 = ashr i64 %8941, 32		; visa id: 11769
  %8955 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8952, i32 %8953, i32 %50, i32 %51)
  %8956 = extractvalue { i32, i32 } %8955, 0		; visa id: 11770
  %8957 = extractvalue { i32, i32 } %8955, 1		; visa id: 11770
  %8958 = insertelement <2 x i32> undef, i32 %8956, i32 0		; visa id: 11777
  %8959 = insertelement <2 x i32> %8958, i32 %8957, i32 1		; visa id: 11778
  %8960 = bitcast <2 x i32> %8959 to i64		; visa id: 11779
  %8961 = add nsw i64 %8960, %8954, !spirv.Decorations !649		; visa id: 11783
  %8962 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !618		; visa id: 11784
  br i1 %86, label %8968, label %8963, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11785

8963:                                             ; preds = %8945
; BB927 :
  %8964 = shl i64 %8961, 2		; visa id: 11787
  %8965 = add i64 %.in, %8964		; visa id: 11788
  %8966 = inttoptr i64 %8965 to float addrspace(4)*		; visa id: 11789
  %8967 = addrspacecast float addrspace(4)* %8966 to float addrspace(1)*		; visa id: 11789
  store float %8962, float addrspace(1)* %8967, align 4		; visa id: 11790
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11791

8968:                                             ; preds = %8945
; BB928 :
  %8969 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %8952, i32 %8953, i32 %47, i32 %48)
  %8970 = extractvalue { i32, i32 } %8969, 0		; visa id: 11793
  %8971 = extractvalue { i32, i32 } %8969, 1		; visa id: 11793
  %8972 = insertelement <2 x i32> undef, i32 %8970, i32 0		; visa id: 11800
  %8973 = insertelement <2 x i32> %8972, i32 %8971, i32 1		; visa id: 11801
  %8974 = bitcast <2 x i32> %8973 to i64		; visa id: 11802
  %8975 = shl i64 %8974, 2		; visa id: 11806
  %8976 = add i64 %.in399, %8975		; visa id: 11807
  %8977 = shl nsw i64 %8954, 2		; visa id: 11808
  %8978 = add i64 %8976, %8977		; visa id: 11809
  %8979 = inttoptr i64 %8978 to float addrspace(4)*		; visa id: 11810
  %8980 = addrspacecast float addrspace(4)* %8979 to float addrspace(1)*		; visa id: 11810
  %8981 = load float, float addrspace(1)* %8980, align 4		; visa id: 11811
  %8982 = fmul reassoc nsz arcp contract float %8981, %4, !spirv.Decorations !618		; visa id: 11812
  %8983 = fadd reassoc nsz arcp contract float %8962, %8982, !spirv.Decorations !618		; visa id: 11813
  %8984 = shl i64 %8961, 2		; visa id: 11814
  %8985 = add i64 %.in, %8984		; visa id: 11815
  %8986 = inttoptr i64 %8985 to float addrspace(4)*		; visa id: 11816
  %8987 = addrspacecast float addrspace(4)* %8986 to float addrspace(1)*		; visa id: 11816
  store float %8983, float addrspace(1)* %8987, align 4		; visa id: 11817
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11818

._crit_edge70.2.10:                               ; preds = %.._crit_edge70.2.10_crit_edge, %8968, %8963
; BB929 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11819
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11819
  %8988 = insertelement <2 x i32> %6617, i32 %8810, i64 1		; visa id: 11819
  store <2 x i32> %8988, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11822
  br label %._crit_edge378, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11824

._crit_edge378:                                   ; preds = %._crit_edge378.._crit_edge378_crit_edge, %._crit_edge70.2.10
; BB930 :
  %8989 = phi i32 [ 0, %._crit_edge70.2.10 ], [ %8998, %._crit_edge378.._crit_edge378_crit_edge ]
  %8990 = zext i32 %8989 to i64		; visa id: 11825
  %8991 = shl nuw nsw i64 %8990, 2		; visa id: 11826
  %8992 = add i64 %6430, %8991		; visa id: 11827
  %8993 = inttoptr i64 %8992 to i32*		; visa id: 11828
  %8994 = load i32, i32* %8993, align 4, !noalias !642		; visa id: 11828
  %8995 = add i64 %6426, %8991		; visa id: 11829
  %8996 = inttoptr i64 %8995 to i32*		; visa id: 11830
  store i32 %8994, i32* %8996, align 4, !alias.scope !642		; visa id: 11830
  %8997 = icmp eq i32 %8989, 0		; visa id: 11831
  br i1 %8997, label %._crit_edge378.._crit_edge378_crit_edge, label %8999, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11832

._crit_edge378.._crit_edge378_crit_edge:          ; preds = %._crit_edge378
; BB931 :
  %8998 = add nuw nsw i32 %8989, 1, !spirv.Decorations !631		; visa id: 11834
  br label %._crit_edge378, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11835

8999:                                             ; preds = %._crit_edge378
; BB932 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11837
  %9000 = load i64, i64* %6446, align 8		; visa id: 11837
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11838
  %9001 = icmp slt i32 %6616, %const_reg_dword
  %9002 = icmp slt i32 %8810, %const_reg_dword1		; visa id: 11838
  %9003 = and i1 %9001, %9002		; visa id: 11839
  br i1 %9003, label %9004, label %..preheader1.10_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11841

..preheader1.10_crit_edge:                        ; preds = %8999
; BB:
  br label %.preheader1.10, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9004:                                             ; preds = %8999
; BB934 :
  %9005 = bitcast i64 %9000 to <2 x i32>		; visa id: 11843
  %9006 = extractelement <2 x i32> %9005, i32 0		; visa id: 11845
  %9007 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9006, i32 1
  %9008 = bitcast <2 x i32> %9007 to i64		; visa id: 11845
  %9009 = ashr exact i64 %9008, 32		; visa id: 11846
  %9010 = bitcast i64 %9009 to <2 x i32>		; visa id: 11847
  %9011 = extractelement <2 x i32> %9010, i32 0		; visa id: 11851
  %9012 = extractelement <2 x i32> %9010, i32 1		; visa id: 11851
  %9013 = ashr i64 %9000, 32		; visa id: 11851
  %9014 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9011, i32 %9012, i32 %50, i32 %51)
  %9015 = extractvalue { i32, i32 } %9014, 0		; visa id: 11852
  %9016 = extractvalue { i32, i32 } %9014, 1		; visa id: 11852
  %9017 = insertelement <2 x i32> undef, i32 %9015, i32 0		; visa id: 11859
  %9018 = insertelement <2 x i32> %9017, i32 %9016, i32 1		; visa id: 11860
  %9019 = bitcast <2 x i32> %9018 to i64		; visa id: 11861
  %9020 = add nsw i64 %9019, %9013, !spirv.Decorations !649		; visa id: 11865
  %9021 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !618		; visa id: 11866
  br i1 %86, label %9027, label %9022, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11867

9022:                                             ; preds = %9004
; BB935 :
  %9023 = shl i64 %9020, 2		; visa id: 11869
  %9024 = add i64 %.in, %9023		; visa id: 11870
  %9025 = inttoptr i64 %9024 to float addrspace(4)*		; visa id: 11871
  %9026 = addrspacecast float addrspace(4)* %9025 to float addrspace(1)*		; visa id: 11871
  store float %9021, float addrspace(1)* %9026, align 4		; visa id: 11872
  br label %.preheader1.10, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11873

9027:                                             ; preds = %9004
; BB936 :
  %9028 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9011, i32 %9012, i32 %47, i32 %48)
  %9029 = extractvalue { i32, i32 } %9028, 0		; visa id: 11875
  %9030 = extractvalue { i32, i32 } %9028, 1		; visa id: 11875
  %9031 = insertelement <2 x i32> undef, i32 %9029, i32 0		; visa id: 11882
  %9032 = insertelement <2 x i32> %9031, i32 %9030, i32 1		; visa id: 11883
  %9033 = bitcast <2 x i32> %9032 to i64		; visa id: 11884
  %9034 = shl i64 %9033, 2		; visa id: 11888
  %9035 = add i64 %.in399, %9034		; visa id: 11889
  %9036 = shl nsw i64 %9013, 2		; visa id: 11890
  %9037 = add i64 %9035, %9036		; visa id: 11891
  %9038 = inttoptr i64 %9037 to float addrspace(4)*		; visa id: 11892
  %9039 = addrspacecast float addrspace(4)* %9038 to float addrspace(1)*		; visa id: 11892
  %9040 = load float, float addrspace(1)* %9039, align 4		; visa id: 11893
  %9041 = fmul reassoc nsz arcp contract float %9040, %4, !spirv.Decorations !618		; visa id: 11894
  %9042 = fadd reassoc nsz arcp contract float %9021, %9041, !spirv.Decorations !618		; visa id: 11895
  %9043 = shl i64 %9020, 2		; visa id: 11896
  %9044 = add i64 %.in, %9043		; visa id: 11897
  %9045 = inttoptr i64 %9044 to float addrspace(4)*		; visa id: 11898
  %9046 = addrspacecast float addrspace(4)* %9045 to float addrspace(1)*		; visa id: 11898
  store float %9042, float addrspace(1)* %9046, align 4		; visa id: 11899
  br label %.preheader1.10, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11900

.preheader1.10:                                   ; preds = %..preheader1.10_crit_edge, %9027, %9022
; BB937 :
  %9047 = add i32 %69, 11		; visa id: 11901
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11902
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11902
  %9048 = insertelement <2 x i32> %6432, i32 %9047, i64 1		; visa id: 11902
  store <2 x i32> %9048, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11905
  br label %._crit_edge379, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11907

._crit_edge379:                                   ; preds = %._crit_edge379.._crit_edge379_crit_edge, %.preheader1.10
; BB938 :
  %9049 = phi i32 [ 0, %.preheader1.10 ], [ %9058, %._crit_edge379.._crit_edge379_crit_edge ]
  %9050 = zext i32 %9049 to i64		; visa id: 11908
  %9051 = shl nuw nsw i64 %9050, 2		; visa id: 11909
  %9052 = add i64 %6430, %9051		; visa id: 11910
  %9053 = inttoptr i64 %9052 to i32*		; visa id: 11911
  %9054 = load i32, i32* %9053, align 4, !noalias !642		; visa id: 11911
  %9055 = add i64 %6426, %9051		; visa id: 11912
  %9056 = inttoptr i64 %9055 to i32*		; visa id: 11913
  store i32 %9054, i32* %9056, align 4, !alias.scope !642		; visa id: 11913
  %9057 = icmp eq i32 %9049, 0		; visa id: 11914
  br i1 %9057, label %._crit_edge379.._crit_edge379_crit_edge, label %9059, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11915

._crit_edge379.._crit_edge379_crit_edge:          ; preds = %._crit_edge379
; BB939 :
  %9058 = add nuw nsw i32 %9049, 1, !spirv.Decorations !631		; visa id: 11917
  br label %._crit_edge379, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 11918

9059:                                             ; preds = %._crit_edge379
; BB940 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11920
  %9060 = load i64, i64* %6446, align 8		; visa id: 11920
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11921
  %9061 = icmp slt i32 %9047, %const_reg_dword1		; visa id: 11921
  %9062 = icmp slt i32 %65, %const_reg_dword
  %9063 = and i1 %9062, %9061		; visa id: 11922
  br i1 %9063, label %9064, label %.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11924

.._crit_edge70.11_crit_edge:                      ; preds = %9059
; BB:
  br label %._crit_edge70.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9064:                                             ; preds = %9059
; BB942 :
  %9065 = bitcast i64 %9060 to <2 x i32>		; visa id: 11926
  %9066 = extractelement <2 x i32> %9065, i32 0		; visa id: 11928
  %9067 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9066, i32 1
  %9068 = bitcast <2 x i32> %9067 to i64		; visa id: 11928
  %9069 = ashr exact i64 %9068, 32		; visa id: 11929
  %9070 = bitcast i64 %9069 to <2 x i32>		; visa id: 11930
  %9071 = extractelement <2 x i32> %9070, i32 0		; visa id: 11934
  %9072 = extractelement <2 x i32> %9070, i32 1		; visa id: 11934
  %9073 = ashr i64 %9060, 32		; visa id: 11934
  %9074 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9071, i32 %9072, i32 %50, i32 %51)
  %9075 = extractvalue { i32, i32 } %9074, 0		; visa id: 11935
  %9076 = extractvalue { i32, i32 } %9074, 1		; visa id: 11935
  %9077 = insertelement <2 x i32> undef, i32 %9075, i32 0		; visa id: 11942
  %9078 = insertelement <2 x i32> %9077, i32 %9076, i32 1		; visa id: 11943
  %9079 = bitcast <2 x i32> %9078 to i64		; visa id: 11944
  %9080 = add nsw i64 %9079, %9073, !spirv.Decorations !649		; visa id: 11948
  %9081 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !618		; visa id: 11949
  br i1 %86, label %9087, label %9082, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 11950

9082:                                             ; preds = %9064
; BB943 :
  %9083 = shl i64 %9080, 2		; visa id: 11952
  %9084 = add i64 %.in, %9083		; visa id: 11953
  %9085 = inttoptr i64 %9084 to float addrspace(4)*		; visa id: 11954
  %9086 = addrspacecast float addrspace(4)* %9085 to float addrspace(1)*		; visa id: 11954
  store float %9081, float addrspace(1)* %9086, align 4		; visa id: 11955
  br label %._crit_edge70.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 11956

9087:                                             ; preds = %9064
; BB944 :
  %9088 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9071, i32 %9072, i32 %47, i32 %48)
  %9089 = extractvalue { i32, i32 } %9088, 0		; visa id: 11958
  %9090 = extractvalue { i32, i32 } %9088, 1		; visa id: 11958
  %9091 = insertelement <2 x i32> undef, i32 %9089, i32 0		; visa id: 11965
  %9092 = insertelement <2 x i32> %9091, i32 %9090, i32 1		; visa id: 11966
  %9093 = bitcast <2 x i32> %9092 to i64		; visa id: 11967
  %9094 = shl i64 %9093, 2		; visa id: 11971
  %9095 = add i64 %.in399, %9094		; visa id: 11972
  %9096 = shl nsw i64 %9073, 2		; visa id: 11973
  %9097 = add i64 %9095, %9096		; visa id: 11974
  %9098 = inttoptr i64 %9097 to float addrspace(4)*		; visa id: 11975
  %9099 = addrspacecast float addrspace(4)* %9098 to float addrspace(1)*		; visa id: 11975
  %9100 = load float, float addrspace(1)* %9099, align 4		; visa id: 11976
  %9101 = fmul reassoc nsz arcp contract float %9100, %4, !spirv.Decorations !618		; visa id: 11977
  %9102 = fadd reassoc nsz arcp contract float %9081, %9101, !spirv.Decorations !618		; visa id: 11978
  %9103 = shl i64 %9080, 2		; visa id: 11979
  %9104 = add i64 %.in, %9103		; visa id: 11980
  %9105 = inttoptr i64 %9104 to float addrspace(4)*		; visa id: 11981
  %9106 = addrspacecast float addrspace(4)* %9105 to float addrspace(1)*		; visa id: 11981
  store float %9102, float addrspace(1)* %9106, align 4		; visa id: 11982
  br label %._crit_edge70.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 11983

._crit_edge70.11:                                 ; preds = %.._crit_edge70.11_crit_edge, %9087, %9082
; BB945 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 11984
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 11984
  %9107 = insertelement <2 x i32> %6495, i32 %9047, i64 1		; visa id: 11984
  store <2 x i32> %9107, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 11987
  br label %._crit_edge380, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 11989

._crit_edge380:                                   ; preds = %._crit_edge380.._crit_edge380_crit_edge, %._crit_edge70.11
; BB946 :
  %9108 = phi i32 [ 0, %._crit_edge70.11 ], [ %9117, %._crit_edge380.._crit_edge380_crit_edge ]
  %9109 = zext i32 %9108 to i64		; visa id: 11990
  %9110 = shl nuw nsw i64 %9109, 2		; visa id: 11991
  %9111 = add i64 %6430, %9110		; visa id: 11992
  %9112 = inttoptr i64 %9111 to i32*		; visa id: 11993
  %9113 = load i32, i32* %9112, align 4, !noalias !642		; visa id: 11993
  %9114 = add i64 %6426, %9110		; visa id: 11994
  %9115 = inttoptr i64 %9114 to i32*		; visa id: 11995
  store i32 %9113, i32* %9115, align 4, !alias.scope !642		; visa id: 11995
  %9116 = icmp eq i32 %9108, 0		; visa id: 11996
  br i1 %9116, label %._crit_edge380.._crit_edge380_crit_edge, label %9118, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 11997

._crit_edge380.._crit_edge380_crit_edge:          ; preds = %._crit_edge380
; BB947 :
  %9117 = add nuw nsw i32 %9108, 1, !spirv.Decorations !631		; visa id: 11999
  br label %._crit_edge380, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12000

9118:                                             ; preds = %._crit_edge380
; BB948 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12002
  %9119 = load i64, i64* %6446, align 8		; visa id: 12002
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12003
  %9120 = icmp slt i32 %6494, %const_reg_dword
  %9121 = icmp slt i32 %9047, %const_reg_dword1		; visa id: 12003
  %9122 = and i1 %9120, %9121		; visa id: 12004
  br i1 %9122, label %9123, label %.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12006

.._crit_edge70.1.11_crit_edge:                    ; preds = %9118
; BB:
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9123:                                             ; preds = %9118
; BB950 :
  %9124 = bitcast i64 %9119 to <2 x i32>		; visa id: 12008
  %9125 = extractelement <2 x i32> %9124, i32 0		; visa id: 12010
  %9126 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9125, i32 1
  %9127 = bitcast <2 x i32> %9126 to i64		; visa id: 12010
  %9128 = ashr exact i64 %9127, 32		; visa id: 12011
  %9129 = bitcast i64 %9128 to <2 x i32>		; visa id: 12012
  %9130 = extractelement <2 x i32> %9129, i32 0		; visa id: 12016
  %9131 = extractelement <2 x i32> %9129, i32 1		; visa id: 12016
  %9132 = ashr i64 %9119, 32		; visa id: 12016
  %9133 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9130, i32 %9131, i32 %50, i32 %51)
  %9134 = extractvalue { i32, i32 } %9133, 0		; visa id: 12017
  %9135 = extractvalue { i32, i32 } %9133, 1		; visa id: 12017
  %9136 = insertelement <2 x i32> undef, i32 %9134, i32 0		; visa id: 12024
  %9137 = insertelement <2 x i32> %9136, i32 %9135, i32 1		; visa id: 12025
  %9138 = bitcast <2 x i32> %9137 to i64		; visa id: 12026
  %9139 = add nsw i64 %9138, %9132, !spirv.Decorations !649		; visa id: 12030
  %9140 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !618		; visa id: 12031
  br i1 %86, label %9146, label %9141, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12032

9141:                                             ; preds = %9123
; BB951 :
  %9142 = shl i64 %9139, 2		; visa id: 12034
  %9143 = add i64 %.in, %9142		; visa id: 12035
  %9144 = inttoptr i64 %9143 to float addrspace(4)*		; visa id: 12036
  %9145 = addrspacecast float addrspace(4)* %9144 to float addrspace(1)*		; visa id: 12036
  store float %9140, float addrspace(1)* %9145, align 4		; visa id: 12037
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12038

9146:                                             ; preds = %9123
; BB952 :
  %9147 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9130, i32 %9131, i32 %47, i32 %48)
  %9148 = extractvalue { i32, i32 } %9147, 0		; visa id: 12040
  %9149 = extractvalue { i32, i32 } %9147, 1		; visa id: 12040
  %9150 = insertelement <2 x i32> undef, i32 %9148, i32 0		; visa id: 12047
  %9151 = insertelement <2 x i32> %9150, i32 %9149, i32 1		; visa id: 12048
  %9152 = bitcast <2 x i32> %9151 to i64		; visa id: 12049
  %9153 = shl i64 %9152, 2		; visa id: 12053
  %9154 = add i64 %.in399, %9153		; visa id: 12054
  %9155 = shl nsw i64 %9132, 2		; visa id: 12055
  %9156 = add i64 %9154, %9155		; visa id: 12056
  %9157 = inttoptr i64 %9156 to float addrspace(4)*		; visa id: 12057
  %9158 = addrspacecast float addrspace(4)* %9157 to float addrspace(1)*		; visa id: 12057
  %9159 = load float, float addrspace(1)* %9158, align 4		; visa id: 12058
  %9160 = fmul reassoc nsz arcp contract float %9159, %4, !spirv.Decorations !618		; visa id: 12059
  %9161 = fadd reassoc nsz arcp contract float %9140, %9160, !spirv.Decorations !618		; visa id: 12060
  %9162 = shl i64 %9139, 2		; visa id: 12061
  %9163 = add i64 %.in, %9162		; visa id: 12062
  %9164 = inttoptr i64 %9163 to float addrspace(4)*		; visa id: 12063
  %9165 = addrspacecast float addrspace(4)* %9164 to float addrspace(1)*		; visa id: 12063
  store float %9161, float addrspace(1)* %9165, align 4		; visa id: 12064
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12065

._crit_edge70.1.11:                               ; preds = %.._crit_edge70.1.11_crit_edge, %9146, %9141
; BB953 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12066
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12066
  %9166 = insertelement <2 x i32> %6556, i32 %9047, i64 1		; visa id: 12066
  store <2 x i32> %9166, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12069
  br label %._crit_edge381, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12071

._crit_edge381:                                   ; preds = %._crit_edge381.._crit_edge381_crit_edge, %._crit_edge70.1.11
; BB954 :
  %9167 = phi i32 [ 0, %._crit_edge70.1.11 ], [ %9176, %._crit_edge381.._crit_edge381_crit_edge ]
  %9168 = zext i32 %9167 to i64		; visa id: 12072
  %9169 = shl nuw nsw i64 %9168, 2		; visa id: 12073
  %9170 = add i64 %6430, %9169		; visa id: 12074
  %9171 = inttoptr i64 %9170 to i32*		; visa id: 12075
  %9172 = load i32, i32* %9171, align 4, !noalias !642		; visa id: 12075
  %9173 = add i64 %6426, %9169		; visa id: 12076
  %9174 = inttoptr i64 %9173 to i32*		; visa id: 12077
  store i32 %9172, i32* %9174, align 4, !alias.scope !642		; visa id: 12077
  %9175 = icmp eq i32 %9167, 0		; visa id: 12078
  br i1 %9175, label %._crit_edge381.._crit_edge381_crit_edge, label %9177, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12079

._crit_edge381.._crit_edge381_crit_edge:          ; preds = %._crit_edge381
; BB955 :
  %9176 = add nuw nsw i32 %9167, 1, !spirv.Decorations !631		; visa id: 12081
  br label %._crit_edge381, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12082

9177:                                             ; preds = %._crit_edge381
; BB956 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12084
  %9178 = load i64, i64* %6446, align 8		; visa id: 12084
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12085
  %9179 = icmp slt i32 %6555, %const_reg_dword
  %9180 = icmp slt i32 %9047, %const_reg_dword1		; visa id: 12085
  %9181 = and i1 %9179, %9180		; visa id: 12086
  br i1 %9181, label %9182, label %.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12088

.._crit_edge70.2.11_crit_edge:                    ; preds = %9177
; BB:
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9182:                                             ; preds = %9177
; BB958 :
  %9183 = bitcast i64 %9178 to <2 x i32>		; visa id: 12090
  %9184 = extractelement <2 x i32> %9183, i32 0		; visa id: 12092
  %9185 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9184, i32 1
  %9186 = bitcast <2 x i32> %9185 to i64		; visa id: 12092
  %9187 = ashr exact i64 %9186, 32		; visa id: 12093
  %9188 = bitcast i64 %9187 to <2 x i32>		; visa id: 12094
  %9189 = extractelement <2 x i32> %9188, i32 0		; visa id: 12098
  %9190 = extractelement <2 x i32> %9188, i32 1		; visa id: 12098
  %9191 = ashr i64 %9178, 32		; visa id: 12098
  %9192 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9189, i32 %9190, i32 %50, i32 %51)
  %9193 = extractvalue { i32, i32 } %9192, 0		; visa id: 12099
  %9194 = extractvalue { i32, i32 } %9192, 1		; visa id: 12099
  %9195 = insertelement <2 x i32> undef, i32 %9193, i32 0		; visa id: 12106
  %9196 = insertelement <2 x i32> %9195, i32 %9194, i32 1		; visa id: 12107
  %9197 = bitcast <2 x i32> %9196 to i64		; visa id: 12108
  %9198 = add nsw i64 %9197, %9191, !spirv.Decorations !649		; visa id: 12112
  %9199 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !618		; visa id: 12113
  br i1 %86, label %9205, label %9200, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12114

9200:                                             ; preds = %9182
; BB959 :
  %9201 = shl i64 %9198, 2		; visa id: 12116
  %9202 = add i64 %.in, %9201		; visa id: 12117
  %9203 = inttoptr i64 %9202 to float addrspace(4)*		; visa id: 12118
  %9204 = addrspacecast float addrspace(4)* %9203 to float addrspace(1)*		; visa id: 12118
  store float %9199, float addrspace(1)* %9204, align 4		; visa id: 12119
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12120

9205:                                             ; preds = %9182
; BB960 :
  %9206 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9189, i32 %9190, i32 %47, i32 %48)
  %9207 = extractvalue { i32, i32 } %9206, 0		; visa id: 12122
  %9208 = extractvalue { i32, i32 } %9206, 1		; visa id: 12122
  %9209 = insertelement <2 x i32> undef, i32 %9207, i32 0		; visa id: 12129
  %9210 = insertelement <2 x i32> %9209, i32 %9208, i32 1		; visa id: 12130
  %9211 = bitcast <2 x i32> %9210 to i64		; visa id: 12131
  %9212 = shl i64 %9211, 2		; visa id: 12135
  %9213 = add i64 %.in399, %9212		; visa id: 12136
  %9214 = shl nsw i64 %9191, 2		; visa id: 12137
  %9215 = add i64 %9213, %9214		; visa id: 12138
  %9216 = inttoptr i64 %9215 to float addrspace(4)*		; visa id: 12139
  %9217 = addrspacecast float addrspace(4)* %9216 to float addrspace(1)*		; visa id: 12139
  %9218 = load float, float addrspace(1)* %9217, align 4		; visa id: 12140
  %9219 = fmul reassoc nsz arcp contract float %9218, %4, !spirv.Decorations !618		; visa id: 12141
  %9220 = fadd reassoc nsz arcp contract float %9199, %9219, !spirv.Decorations !618		; visa id: 12142
  %9221 = shl i64 %9198, 2		; visa id: 12143
  %9222 = add i64 %.in, %9221		; visa id: 12144
  %9223 = inttoptr i64 %9222 to float addrspace(4)*		; visa id: 12145
  %9224 = addrspacecast float addrspace(4)* %9223 to float addrspace(1)*		; visa id: 12145
  store float %9220, float addrspace(1)* %9224, align 4		; visa id: 12146
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12147

._crit_edge70.2.11:                               ; preds = %.._crit_edge70.2.11_crit_edge, %9205, %9200
; BB961 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12148
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12148
  %9225 = insertelement <2 x i32> %6617, i32 %9047, i64 1		; visa id: 12148
  store <2 x i32> %9225, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12151
  br label %._crit_edge382, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12153

._crit_edge382:                                   ; preds = %._crit_edge382.._crit_edge382_crit_edge, %._crit_edge70.2.11
; BB962 :
  %9226 = phi i32 [ 0, %._crit_edge70.2.11 ], [ %9235, %._crit_edge382.._crit_edge382_crit_edge ]
  %9227 = zext i32 %9226 to i64		; visa id: 12154
  %9228 = shl nuw nsw i64 %9227, 2		; visa id: 12155
  %9229 = add i64 %6430, %9228		; visa id: 12156
  %9230 = inttoptr i64 %9229 to i32*		; visa id: 12157
  %9231 = load i32, i32* %9230, align 4, !noalias !642		; visa id: 12157
  %9232 = add i64 %6426, %9228		; visa id: 12158
  %9233 = inttoptr i64 %9232 to i32*		; visa id: 12159
  store i32 %9231, i32* %9233, align 4, !alias.scope !642		; visa id: 12159
  %9234 = icmp eq i32 %9226, 0		; visa id: 12160
  br i1 %9234, label %._crit_edge382.._crit_edge382_crit_edge, label %9236, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12161

._crit_edge382.._crit_edge382_crit_edge:          ; preds = %._crit_edge382
; BB963 :
  %9235 = add nuw nsw i32 %9226, 1, !spirv.Decorations !631		; visa id: 12163
  br label %._crit_edge382, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12164

9236:                                             ; preds = %._crit_edge382
; BB964 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12166
  %9237 = load i64, i64* %6446, align 8		; visa id: 12166
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12167
  %9238 = icmp slt i32 %6616, %const_reg_dword
  %9239 = icmp slt i32 %9047, %const_reg_dword1		; visa id: 12167
  %9240 = and i1 %9238, %9239		; visa id: 12168
  br i1 %9240, label %9241, label %..preheader1.11_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12170

..preheader1.11_crit_edge:                        ; preds = %9236
; BB:
  br label %.preheader1.11, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9241:                                             ; preds = %9236
; BB966 :
  %9242 = bitcast i64 %9237 to <2 x i32>		; visa id: 12172
  %9243 = extractelement <2 x i32> %9242, i32 0		; visa id: 12174
  %9244 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9243, i32 1
  %9245 = bitcast <2 x i32> %9244 to i64		; visa id: 12174
  %9246 = ashr exact i64 %9245, 32		; visa id: 12175
  %9247 = bitcast i64 %9246 to <2 x i32>		; visa id: 12176
  %9248 = extractelement <2 x i32> %9247, i32 0		; visa id: 12180
  %9249 = extractelement <2 x i32> %9247, i32 1		; visa id: 12180
  %9250 = ashr i64 %9237, 32		; visa id: 12180
  %9251 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9248, i32 %9249, i32 %50, i32 %51)
  %9252 = extractvalue { i32, i32 } %9251, 0		; visa id: 12181
  %9253 = extractvalue { i32, i32 } %9251, 1		; visa id: 12181
  %9254 = insertelement <2 x i32> undef, i32 %9252, i32 0		; visa id: 12188
  %9255 = insertelement <2 x i32> %9254, i32 %9253, i32 1		; visa id: 12189
  %9256 = bitcast <2 x i32> %9255 to i64		; visa id: 12190
  %9257 = add nsw i64 %9256, %9250, !spirv.Decorations !649		; visa id: 12194
  %9258 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !618		; visa id: 12195
  br i1 %86, label %9264, label %9259, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12196

9259:                                             ; preds = %9241
; BB967 :
  %9260 = shl i64 %9257, 2		; visa id: 12198
  %9261 = add i64 %.in, %9260		; visa id: 12199
  %9262 = inttoptr i64 %9261 to float addrspace(4)*		; visa id: 12200
  %9263 = addrspacecast float addrspace(4)* %9262 to float addrspace(1)*		; visa id: 12200
  store float %9258, float addrspace(1)* %9263, align 4		; visa id: 12201
  br label %.preheader1.11, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12202

9264:                                             ; preds = %9241
; BB968 :
  %9265 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9248, i32 %9249, i32 %47, i32 %48)
  %9266 = extractvalue { i32, i32 } %9265, 0		; visa id: 12204
  %9267 = extractvalue { i32, i32 } %9265, 1		; visa id: 12204
  %9268 = insertelement <2 x i32> undef, i32 %9266, i32 0		; visa id: 12211
  %9269 = insertelement <2 x i32> %9268, i32 %9267, i32 1		; visa id: 12212
  %9270 = bitcast <2 x i32> %9269 to i64		; visa id: 12213
  %9271 = shl i64 %9270, 2		; visa id: 12217
  %9272 = add i64 %.in399, %9271		; visa id: 12218
  %9273 = shl nsw i64 %9250, 2		; visa id: 12219
  %9274 = add i64 %9272, %9273		; visa id: 12220
  %9275 = inttoptr i64 %9274 to float addrspace(4)*		; visa id: 12221
  %9276 = addrspacecast float addrspace(4)* %9275 to float addrspace(1)*		; visa id: 12221
  %9277 = load float, float addrspace(1)* %9276, align 4		; visa id: 12222
  %9278 = fmul reassoc nsz arcp contract float %9277, %4, !spirv.Decorations !618		; visa id: 12223
  %9279 = fadd reassoc nsz arcp contract float %9258, %9278, !spirv.Decorations !618		; visa id: 12224
  %9280 = shl i64 %9257, 2		; visa id: 12225
  %9281 = add i64 %.in, %9280		; visa id: 12226
  %9282 = inttoptr i64 %9281 to float addrspace(4)*		; visa id: 12227
  %9283 = addrspacecast float addrspace(4)* %9282 to float addrspace(1)*		; visa id: 12227
  store float %9279, float addrspace(1)* %9283, align 4		; visa id: 12228
  br label %.preheader1.11, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12229

.preheader1.11:                                   ; preds = %..preheader1.11_crit_edge, %9264, %9259
; BB969 :
  %9284 = add i32 %69, 12		; visa id: 12230
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12231
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12231
  %9285 = insertelement <2 x i32> %6432, i32 %9284, i64 1		; visa id: 12231
  store <2 x i32> %9285, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12234
  br label %._crit_edge383, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12236

._crit_edge383:                                   ; preds = %._crit_edge383.._crit_edge383_crit_edge, %.preheader1.11
; BB970 :
  %9286 = phi i32 [ 0, %.preheader1.11 ], [ %9295, %._crit_edge383.._crit_edge383_crit_edge ]
  %9287 = zext i32 %9286 to i64		; visa id: 12237
  %9288 = shl nuw nsw i64 %9287, 2		; visa id: 12238
  %9289 = add i64 %6430, %9288		; visa id: 12239
  %9290 = inttoptr i64 %9289 to i32*		; visa id: 12240
  %9291 = load i32, i32* %9290, align 4, !noalias !642		; visa id: 12240
  %9292 = add i64 %6426, %9288		; visa id: 12241
  %9293 = inttoptr i64 %9292 to i32*		; visa id: 12242
  store i32 %9291, i32* %9293, align 4, !alias.scope !642		; visa id: 12242
  %9294 = icmp eq i32 %9286, 0		; visa id: 12243
  br i1 %9294, label %._crit_edge383.._crit_edge383_crit_edge, label %9296, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12244

._crit_edge383.._crit_edge383_crit_edge:          ; preds = %._crit_edge383
; BB971 :
  %9295 = add nuw nsw i32 %9286, 1, !spirv.Decorations !631		; visa id: 12246
  br label %._crit_edge383, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12247

9296:                                             ; preds = %._crit_edge383
; BB972 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12249
  %9297 = load i64, i64* %6446, align 8		; visa id: 12249
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12250
  %9298 = icmp slt i32 %9284, %const_reg_dword1		; visa id: 12250
  %9299 = icmp slt i32 %65, %const_reg_dword
  %9300 = and i1 %9299, %9298		; visa id: 12251
  br i1 %9300, label %9301, label %.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12253

.._crit_edge70.12_crit_edge:                      ; preds = %9296
; BB:
  br label %._crit_edge70.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9301:                                             ; preds = %9296
; BB974 :
  %9302 = bitcast i64 %9297 to <2 x i32>		; visa id: 12255
  %9303 = extractelement <2 x i32> %9302, i32 0		; visa id: 12257
  %9304 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9303, i32 1
  %9305 = bitcast <2 x i32> %9304 to i64		; visa id: 12257
  %9306 = ashr exact i64 %9305, 32		; visa id: 12258
  %9307 = bitcast i64 %9306 to <2 x i32>		; visa id: 12259
  %9308 = extractelement <2 x i32> %9307, i32 0		; visa id: 12263
  %9309 = extractelement <2 x i32> %9307, i32 1		; visa id: 12263
  %9310 = ashr i64 %9297, 32		; visa id: 12263
  %9311 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9308, i32 %9309, i32 %50, i32 %51)
  %9312 = extractvalue { i32, i32 } %9311, 0		; visa id: 12264
  %9313 = extractvalue { i32, i32 } %9311, 1		; visa id: 12264
  %9314 = insertelement <2 x i32> undef, i32 %9312, i32 0		; visa id: 12271
  %9315 = insertelement <2 x i32> %9314, i32 %9313, i32 1		; visa id: 12272
  %9316 = bitcast <2 x i32> %9315 to i64		; visa id: 12273
  %9317 = add nsw i64 %9316, %9310, !spirv.Decorations !649		; visa id: 12277
  %9318 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !618		; visa id: 12278
  br i1 %86, label %9324, label %9319, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12279

9319:                                             ; preds = %9301
; BB975 :
  %9320 = shl i64 %9317, 2		; visa id: 12281
  %9321 = add i64 %.in, %9320		; visa id: 12282
  %9322 = inttoptr i64 %9321 to float addrspace(4)*		; visa id: 12283
  %9323 = addrspacecast float addrspace(4)* %9322 to float addrspace(1)*		; visa id: 12283
  store float %9318, float addrspace(1)* %9323, align 4		; visa id: 12284
  br label %._crit_edge70.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12285

9324:                                             ; preds = %9301
; BB976 :
  %9325 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9308, i32 %9309, i32 %47, i32 %48)
  %9326 = extractvalue { i32, i32 } %9325, 0		; visa id: 12287
  %9327 = extractvalue { i32, i32 } %9325, 1		; visa id: 12287
  %9328 = insertelement <2 x i32> undef, i32 %9326, i32 0		; visa id: 12294
  %9329 = insertelement <2 x i32> %9328, i32 %9327, i32 1		; visa id: 12295
  %9330 = bitcast <2 x i32> %9329 to i64		; visa id: 12296
  %9331 = shl i64 %9330, 2		; visa id: 12300
  %9332 = add i64 %.in399, %9331		; visa id: 12301
  %9333 = shl nsw i64 %9310, 2		; visa id: 12302
  %9334 = add i64 %9332, %9333		; visa id: 12303
  %9335 = inttoptr i64 %9334 to float addrspace(4)*		; visa id: 12304
  %9336 = addrspacecast float addrspace(4)* %9335 to float addrspace(1)*		; visa id: 12304
  %9337 = load float, float addrspace(1)* %9336, align 4		; visa id: 12305
  %9338 = fmul reassoc nsz arcp contract float %9337, %4, !spirv.Decorations !618		; visa id: 12306
  %9339 = fadd reassoc nsz arcp contract float %9318, %9338, !spirv.Decorations !618		; visa id: 12307
  %9340 = shl i64 %9317, 2		; visa id: 12308
  %9341 = add i64 %.in, %9340		; visa id: 12309
  %9342 = inttoptr i64 %9341 to float addrspace(4)*		; visa id: 12310
  %9343 = addrspacecast float addrspace(4)* %9342 to float addrspace(1)*		; visa id: 12310
  store float %9339, float addrspace(1)* %9343, align 4		; visa id: 12311
  br label %._crit_edge70.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12312

._crit_edge70.12:                                 ; preds = %.._crit_edge70.12_crit_edge, %9324, %9319
; BB977 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12313
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12313
  %9344 = insertelement <2 x i32> %6495, i32 %9284, i64 1		; visa id: 12313
  store <2 x i32> %9344, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12316
  br label %._crit_edge384, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12318

._crit_edge384:                                   ; preds = %._crit_edge384.._crit_edge384_crit_edge, %._crit_edge70.12
; BB978 :
  %9345 = phi i32 [ 0, %._crit_edge70.12 ], [ %9354, %._crit_edge384.._crit_edge384_crit_edge ]
  %9346 = zext i32 %9345 to i64		; visa id: 12319
  %9347 = shl nuw nsw i64 %9346, 2		; visa id: 12320
  %9348 = add i64 %6430, %9347		; visa id: 12321
  %9349 = inttoptr i64 %9348 to i32*		; visa id: 12322
  %9350 = load i32, i32* %9349, align 4, !noalias !642		; visa id: 12322
  %9351 = add i64 %6426, %9347		; visa id: 12323
  %9352 = inttoptr i64 %9351 to i32*		; visa id: 12324
  store i32 %9350, i32* %9352, align 4, !alias.scope !642		; visa id: 12324
  %9353 = icmp eq i32 %9345, 0		; visa id: 12325
  br i1 %9353, label %._crit_edge384.._crit_edge384_crit_edge, label %9355, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12326

._crit_edge384.._crit_edge384_crit_edge:          ; preds = %._crit_edge384
; BB979 :
  %9354 = add nuw nsw i32 %9345, 1, !spirv.Decorations !631		; visa id: 12328
  br label %._crit_edge384, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12329

9355:                                             ; preds = %._crit_edge384
; BB980 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12331
  %9356 = load i64, i64* %6446, align 8		; visa id: 12331
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12332
  %9357 = icmp slt i32 %6494, %const_reg_dword
  %9358 = icmp slt i32 %9284, %const_reg_dword1		; visa id: 12332
  %9359 = and i1 %9357, %9358		; visa id: 12333
  br i1 %9359, label %9360, label %.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12335

.._crit_edge70.1.12_crit_edge:                    ; preds = %9355
; BB:
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9360:                                             ; preds = %9355
; BB982 :
  %9361 = bitcast i64 %9356 to <2 x i32>		; visa id: 12337
  %9362 = extractelement <2 x i32> %9361, i32 0		; visa id: 12339
  %9363 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9362, i32 1
  %9364 = bitcast <2 x i32> %9363 to i64		; visa id: 12339
  %9365 = ashr exact i64 %9364, 32		; visa id: 12340
  %9366 = bitcast i64 %9365 to <2 x i32>		; visa id: 12341
  %9367 = extractelement <2 x i32> %9366, i32 0		; visa id: 12345
  %9368 = extractelement <2 x i32> %9366, i32 1		; visa id: 12345
  %9369 = ashr i64 %9356, 32		; visa id: 12345
  %9370 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9367, i32 %9368, i32 %50, i32 %51)
  %9371 = extractvalue { i32, i32 } %9370, 0		; visa id: 12346
  %9372 = extractvalue { i32, i32 } %9370, 1		; visa id: 12346
  %9373 = insertelement <2 x i32> undef, i32 %9371, i32 0		; visa id: 12353
  %9374 = insertelement <2 x i32> %9373, i32 %9372, i32 1		; visa id: 12354
  %9375 = bitcast <2 x i32> %9374 to i64		; visa id: 12355
  %9376 = add nsw i64 %9375, %9369, !spirv.Decorations !649		; visa id: 12359
  %9377 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !618		; visa id: 12360
  br i1 %86, label %9383, label %9378, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12361

9378:                                             ; preds = %9360
; BB983 :
  %9379 = shl i64 %9376, 2		; visa id: 12363
  %9380 = add i64 %.in, %9379		; visa id: 12364
  %9381 = inttoptr i64 %9380 to float addrspace(4)*		; visa id: 12365
  %9382 = addrspacecast float addrspace(4)* %9381 to float addrspace(1)*		; visa id: 12365
  store float %9377, float addrspace(1)* %9382, align 4		; visa id: 12366
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12367

9383:                                             ; preds = %9360
; BB984 :
  %9384 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9367, i32 %9368, i32 %47, i32 %48)
  %9385 = extractvalue { i32, i32 } %9384, 0		; visa id: 12369
  %9386 = extractvalue { i32, i32 } %9384, 1		; visa id: 12369
  %9387 = insertelement <2 x i32> undef, i32 %9385, i32 0		; visa id: 12376
  %9388 = insertelement <2 x i32> %9387, i32 %9386, i32 1		; visa id: 12377
  %9389 = bitcast <2 x i32> %9388 to i64		; visa id: 12378
  %9390 = shl i64 %9389, 2		; visa id: 12382
  %9391 = add i64 %.in399, %9390		; visa id: 12383
  %9392 = shl nsw i64 %9369, 2		; visa id: 12384
  %9393 = add i64 %9391, %9392		; visa id: 12385
  %9394 = inttoptr i64 %9393 to float addrspace(4)*		; visa id: 12386
  %9395 = addrspacecast float addrspace(4)* %9394 to float addrspace(1)*		; visa id: 12386
  %9396 = load float, float addrspace(1)* %9395, align 4		; visa id: 12387
  %9397 = fmul reassoc nsz arcp contract float %9396, %4, !spirv.Decorations !618		; visa id: 12388
  %9398 = fadd reassoc nsz arcp contract float %9377, %9397, !spirv.Decorations !618		; visa id: 12389
  %9399 = shl i64 %9376, 2		; visa id: 12390
  %9400 = add i64 %.in, %9399		; visa id: 12391
  %9401 = inttoptr i64 %9400 to float addrspace(4)*		; visa id: 12392
  %9402 = addrspacecast float addrspace(4)* %9401 to float addrspace(1)*		; visa id: 12392
  store float %9398, float addrspace(1)* %9402, align 4		; visa id: 12393
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12394

._crit_edge70.1.12:                               ; preds = %.._crit_edge70.1.12_crit_edge, %9383, %9378
; BB985 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12395
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12395
  %9403 = insertelement <2 x i32> %6556, i32 %9284, i64 1		; visa id: 12395
  store <2 x i32> %9403, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12398
  br label %._crit_edge385, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12400

._crit_edge385:                                   ; preds = %._crit_edge385.._crit_edge385_crit_edge, %._crit_edge70.1.12
; BB986 :
  %9404 = phi i32 [ 0, %._crit_edge70.1.12 ], [ %9413, %._crit_edge385.._crit_edge385_crit_edge ]
  %9405 = zext i32 %9404 to i64		; visa id: 12401
  %9406 = shl nuw nsw i64 %9405, 2		; visa id: 12402
  %9407 = add i64 %6430, %9406		; visa id: 12403
  %9408 = inttoptr i64 %9407 to i32*		; visa id: 12404
  %9409 = load i32, i32* %9408, align 4, !noalias !642		; visa id: 12404
  %9410 = add i64 %6426, %9406		; visa id: 12405
  %9411 = inttoptr i64 %9410 to i32*		; visa id: 12406
  store i32 %9409, i32* %9411, align 4, !alias.scope !642		; visa id: 12406
  %9412 = icmp eq i32 %9404, 0		; visa id: 12407
  br i1 %9412, label %._crit_edge385.._crit_edge385_crit_edge, label %9414, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12408

._crit_edge385.._crit_edge385_crit_edge:          ; preds = %._crit_edge385
; BB987 :
  %9413 = add nuw nsw i32 %9404, 1, !spirv.Decorations !631		; visa id: 12410
  br label %._crit_edge385, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12411

9414:                                             ; preds = %._crit_edge385
; BB988 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12413
  %9415 = load i64, i64* %6446, align 8		; visa id: 12413
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12414
  %9416 = icmp slt i32 %6555, %const_reg_dword
  %9417 = icmp slt i32 %9284, %const_reg_dword1		; visa id: 12414
  %9418 = and i1 %9416, %9417		; visa id: 12415
  br i1 %9418, label %9419, label %.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12417

.._crit_edge70.2.12_crit_edge:                    ; preds = %9414
; BB:
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9419:                                             ; preds = %9414
; BB990 :
  %9420 = bitcast i64 %9415 to <2 x i32>		; visa id: 12419
  %9421 = extractelement <2 x i32> %9420, i32 0		; visa id: 12421
  %9422 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9421, i32 1
  %9423 = bitcast <2 x i32> %9422 to i64		; visa id: 12421
  %9424 = ashr exact i64 %9423, 32		; visa id: 12422
  %9425 = bitcast i64 %9424 to <2 x i32>		; visa id: 12423
  %9426 = extractelement <2 x i32> %9425, i32 0		; visa id: 12427
  %9427 = extractelement <2 x i32> %9425, i32 1		; visa id: 12427
  %9428 = ashr i64 %9415, 32		; visa id: 12427
  %9429 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9426, i32 %9427, i32 %50, i32 %51)
  %9430 = extractvalue { i32, i32 } %9429, 0		; visa id: 12428
  %9431 = extractvalue { i32, i32 } %9429, 1		; visa id: 12428
  %9432 = insertelement <2 x i32> undef, i32 %9430, i32 0		; visa id: 12435
  %9433 = insertelement <2 x i32> %9432, i32 %9431, i32 1		; visa id: 12436
  %9434 = bitcast <2 x i32> %9433 to i64		; visa id: 12437
  %9435 = add nsw i64 %9434, %9428, !spirv.Decorations !649		; visa id: 12441
  %9436 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !618		; visa id: 12442
  br i1 %86, label %9442, label %9437, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12443

9437:                                             ; preds = %9419
; BB991 :
  %9438 = shl i64 %9435, 2		; visa id: 12445
  %9439 = add i64 %.in, %9438		; visa id: 12446
  %9440 = inttoptr i64 %9439 to float addrspace(4)*		; visa id: 12447
  %9441 = addrspacecast float addrspace(4)* %9440 to float addrspace(1)*		; visa id: 12447
  store float %9436, float addrspace(1)* %9441, align 4		; visa id: 12448
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12449

9442:                                             ; preds = %9419
; BB992 :
  %9443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9426, i32 %9427, i32 %47, i32 %48)
  %9444 = extractvalue { i32, i32 } %9443, 0		; visa id: 12451
  %9445 = extractvalue { i32, i32 } %9443, 1		; visa id: 12451
  %9446 = insertelement <2 x i32> undef, i32 %9444, i32 0		; visa id: 12458
  %9447 = insertelement <2 x i32> %9446, i32 %9445, i32 1		; visa id: 12459
  %9448 = bitcast <2 x i32> %9447 to i64		; visa id: 12460
  %9449 = shl i64 %9448, 2		; visa id: 12464
  %9450 = add i64 %.in399, %9449		; visa id: 12465
  %9451 = shl nsw i64 %9428, 2		; visa id: 12466
  %9452 = add i64 %9450, %9451		; visa id: 12467
  %9453 = inttoptr i64 %9452 to float addrspace(4)*		; visa id: 12468
  %9454 = addrspacecast float addrspace(4)* %9453 to float addrspace(1)*		; visa id: 12468
  %9455 = load float, float addrspace(1)* %9454, align 4		; visa id: 12469
  %9456 = fmul reassoc nsz arcp contract float %9455, %4, !spirv.Decorations !618		; visa id: 12470
  %9457 = fadd reassoc nsz arcp contract float %9436, %9456, !spirv.Decorations !618		; visa id: 12471
  %9458 = shl i64 %9435, 2		; visa id: 12472
  %9459 = add i64 %.in, %9458		; visa id: 12473
  %9460 = inttoptr i64 %9459 to float addrspace(4)*		; visa id: 12474
  %9461 = addrspacecast float addrspace(4)* %9460 to float addrspace(1)*		; visa id: 12474
  store float %9457, float addrspace(1)* %9461, align 4		; visa id: 12475
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12476

._crit_edge70.2.12:                               ; preds = %.._crit_edge70.2.12_crit_edge, %9442, %9437
; BB993 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12477
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12477
  %9462 = insertelement <2 x i32> %6617, i32 %9284, i64 1		; visa id: 12477
  store <2 x i32> %9462, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12480
  br label %._crit_edge386, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12482

._crit_edge386:                                   ; preds = %._crit_edge386.._crit_edge386_crit_edge, %._crit_edge70.2.12
; BB994 :
  %9463 = phi i32 [ 0, %._crit_edge70.2.12 ], [ %9472, %._crit_edge386.._crit_edge386_crit_edge ]
  %9464 = zext i32 %9463 to i64		; visa id: 12483
  %9465 = shl nuw nsw i64 %9464, 2		; visa id: 12484
  %9466 = add i64 %6430, %9465		; visa id: 12485
  %9467 = inttoptr i64 %9466 to i32*		; visa id: 12486
  %9468 = load i32, i32* %9467, align 4, !noalias !642		; visa id: 12486
  %9469 = add i64 %6426, %9465		; visa id: 12487
  %9470 = inttoptr i64 %9469 to i32*		; visa id: 12488
  store i32 %9468, i32* %9470, align 4, !alias.scope !642		; visa id: 12488
  %9471 = icmp eq i32 %9463, 0		; visa id: 12489
  br i1 %9471, label %._crit_edge386.._crit_edge386_crit_edge, label %9473, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12490

._crit_edge386.._crit_edge386_crit_edge:          ; preds = %._crit_edge386
; BB995 :
  %9472 = add nuw nsw i32 %9463, 1, !spirv.Decorations !631		; visa id: 12492
  br label %._crit_edge386, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12493

9473:                                             ; preds = %._crit_edge386
; BB996 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12495
  %9474 = load i64, i64* %6446, align 8		; visa id: 12495
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12496
  %9475 = icmp slt i32 %6616, %const_reg_dword
  %9476 = icmp slt i32 %9284, %const_reg_dword1		; visa id: 12496
  %9477 = and i1 %9475, %9476		; visa id: 12497
  br i1 %9477, label %9478, label %..preheader1.12_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12499

..preheader1.12_crit_edge:                        ; preds = %9473
; BB:
  br label %.preheader1.12, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9478:                                             ; preds = %9473
; BB998 :
  %9479 = bitcast i64 %9474 to <2 x i32>		; visa id: 12501
  %9480 = extractelement <2 x i32> %9479, i32 0		; visa id: 12503
  %9481 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9480, i32 1
  %9482 = bitcast <2 x i32> %9481 to i64		; visa id: 12503
  %9483 = ashr exact i64 %9482, 32		; visa id: 12504
  %9484 = bitcast i64 %9483 to <2 x i32>		; visa id: 12505
  %9485 = extractelement <2 x i32> %9484, i32 0		; visa id: 12509
  %9486 = extractelement <2 x i32> %9484, i32 1		; visa id: 12509
  %9487 = ashr i64 %9474, 32		; visa id: 12509
  %9488 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9485, i32 %9486, i32 %50, i32 %51)
  %9489 = extractvalue { i32, i32 } %9488, 0		; visa id: 12510
  %9490 = extractvalue { i32, i32 } %9488, 1		; visa id: 12510
  %9491 = insertelement <2 x i32> undef, i32 %9489, i32 0		; visa id: 12517
  %9492 = insertelement <2 x i32> %9491, i32 %9490, i32 1		; visa id: 12518
  %9493 = bitcast <2 x i32> %9492 to i64		; visa id: 12519
  %9494 = add nsw i64 %9493, %9487, !spirv.Decorations !649		; visa id: 12523
  %9495 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !618		; visa id: 12524
  br i1 %86, label %9501, label %9496, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12525

9496:                                             ; preds = %9478
; BB999 :
  %9497 = shl i64 %9494, 2		; visa id: 12527
  %9498 = add i64 %.in, %9497		; visa id: 12528
  %9499 = inttoptr i64 %9498 to float addrspace(4)*		; visa id: 12529
  %9500 = addrspacecast float addrspace(4)* %9499 to float addrspace(1)*		; visa id: 12529
  store float %9495, float addrspace(1)* %9500, align 4		; visa id: 12530
  br label %.preheader1.12, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12531

9501:                                             ; preds = %9478
; BB1000 :
  %9502 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9485, i32 %9486, i32 %47, i32 %48)
  %9503 = extractvalue { i32, i32 } %9502, 0		; visa id: 12533
  %9504 = extractvalue { i32, i32 } %9502, 1		; visa id: 12533
  %9505 = insertelement <2 x i32> undef, i32 %9503, i32 0		; visa id: 12540
  %9506 = insertelement <2 x i32> %9505, i32 %9504, i32 1		; visa id: 12541
  %9507 = bitcast <2 x i32> %9506 to i64		; visa id: 12542
  %9508 = shl i64 %9507, 2		; visa id: 12546
  %9509 = add i64 %.in399, %9508		; visa id: 12547
  %9510 = shl nsw i64 %9487, 2		; visa id: 12548
  %9511 = add i64 %9509, %9510		; visa id: 12549
  %9512 = inttoptr i64 %9511 to float addrspace(4)*		; visa id: 12550
  %9513 = addrspacecast float addrspace(4)* %9512 to float addrspace(1)*		; visa id: 12550
  %9514 = load float, float addrspace(1)* %9513, align 4		; visa id: 12551
  %9515 = fmul reassoc nsz arcp contract float %9514, %4, !spirv.Decorations !618		; visa id: 12552
  %9516 = fadd reassoc nsz arcp contract float %9495, %9515, !spirv.Decorations !618		; visa id: 12553
  %9517 = shl i64 %9494, 2		; visa id: 12554
  %9518 = add i64 %.in, %9517		; visa id: 12555
  %9519 = inttoptr i64 %9518 to float addrspace(4)*		; visa id: 12556
  %9520 = addrspacecast float addrspace(4)* %9519 to float addrspace(1)*		; visa id: 12556
  store float %9516, float addrspace(1)* %9520, align 4		; visa id: 12557
  br label %.preheader1.12, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12558

.preheader1.12:                                   ; preds = %..preheader1.12_crit_edge, %9501, %9496
; BB1001 :
  %9521 = add i32 %69, 13		; visa id: 12559
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12560
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12560
  %9522 = insertelement <2 x i32> %6432, i32 %9521, i64 1		; visa id: 12560
  store <2 x i32> %9522, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12563
  br label %._crit_edge387, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12565

._crit_edge387:                                   ; preds = %._crit_edge387.._crit_edge387_crit_edge, %.preheader1.12
; BB1002 :
  %9523 = phi i32 [ 0, %.preheader1.12 ], [ %9532, %._crit_edge387.._crit_edge387_crit_edge ]
  %9524 = zext i32 %9523 to i64		; visa id: 12566
  %9525 = shl nuw nsw i64 %9524, 2		; visa id: 12567
  %9526 = add i64 %6430, %9525		; visa id: 12568
  %9527 = inttoptr i64 %9526 to i32*		; visa id: 12569
  %9528 = load i32, i32* %9527, align 4, !noalias !642		; visa id: 12569
  %9529 = add i64 %6426, %9525		; visa id: 12570
  %9530 = inttoptr i64 %9529 to i32*		; visa id: 12571
  store i32 %9528, i32* %9530, align 4, !alias.scope !642		; visa id: 12571
  %9531 = icmp eq i32 %9523, 0		; visa id: 12572
  br i1 %9531, label %._crit_edge387.._crit_edge387_crit_edge, label %9533, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12573

._crit_edge387.._crit_edge387_crit_edge:          ; preds = %._crit_edge387
; BB1003 :
  %9532 = add nuw nsw i32 %9523, 1, !spirv.Decorations !631		; visa id: 12575
  br label %._crit_edge387, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12576

9533:                                             ; preds = %._crit_edge387
; BB1004 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12578
  %9534 = load i64, i64* %6446, align 8		; visa id: 12578
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12579
  %9535 = icmp slt i32 %9521, %const_reg_dword1		; visa id: 12579
  %9536 = icmp slt i32 %65, %const_reg_dword
  %9537 = and i1 %9536, %9535		; visa id: 12580
  br i1 %9537, label %9538, label %.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12582

.._crit_edge70.13_crit_edge:                      ; preds = %9533
; BB:
  br label %._crit_edge70.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9538:                                             ; preds = %9533
; BB1006 :
  %9539 = bitcast i64 %9534 to <2 x i32>		; visa id: 12584
  %9540 = extractelement <2 x i32> %9539, i32 0		; visa id: 12586
  %9541 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9540, i32 1
  %9542 = bitcast <2 x i32> %9541 to i64		; visa id: 12586
  %9543 = ashr exact i64 %9542, 32		; visa id: 12587
  %9544 = bitcast i64 %9543 to <2 x i32>		; visa id: 12588
  %9545 = extractelement <2 x i32> %9544, i32 0		; visa id: 12592
  %9546 = extractelement <2 x i32> %9544, i32 1		; visa id: 12592
  %9547 = ashr i64 %9534, 32		; visa id: 12592
  %9548 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9545, i32 %9546, i32 %50, i32 %51)
  %9549 = extractvalue { i32, i32 } %9548, 0		; visa id: 12593
  %9550 = extractvalue { i32, i32 } %9548, 1		; visa id: 12593
  %9551 = insertelement <2 x i32> undef, i32 %9549, i32 0		; visa id: 12600
  %9552 = insertelement <2 x i32> %9551, i32 %9550, i32 1		; visa id: 12601
  %9553 = bitcast <2 x i32> %9552 to i64		; visa id: 12602
  %9554 = add nsw i64 %9553, %9547, !spirv.Decorations !649		; visa id: 12606
  %9555 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !618		; visa id: 12607
  br i1 %86, label %9561, label %9556, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12608

9556:                                             ; preds = %9538
; BB1007 :
  %9557 = shl i64 %9554, 2		; visa id: 12610
  %9558 = add i64 %.in, %9557		; visa id: 12611
  %9559 = inttoptr i64 %9558 to float addrspace(4)*		; visa id: 12612
  %9560 = addrspacecast float addrspace(4)* %9559 to float addrspace(1)*		; visa id: 12612
  store float %9555, float addrspace(1)* %9560, align 4		; visa id: 12613
  br label %._crit_edge70.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12614

9561:                                             ; preds = %9538
; BB1008 :
  %9562 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9545, i32 %9546, i32 %47, i32 %48)
  %9563 = extractvalue { i32, i32 } %9562, 0		; visa id: 12616
  %9564 = extractvalue { i32, i32 } %9562, 1		; visa id: 12616
  %9565 = insertelement <2 x i32> undef, i32 %9563, i32 0		; visa id: 12623
  %9566 = insertelement <2 x i32> %9565, i32 %9564, i32 1		; visa id: 12624
  %9567 = bitcast <2 x i32> %9566 to i64		; visa id: 12625
  %9568 = shl i64 %9567, 2		; visa id: 12629
  %9569 = add i64 %.in399, %9568		; visa id: 12630
  %9570 = shl nsw i64 %9547, 2		; visa id: 12631
  %9571 = add i64 %9569, %9570		; visa id: 12632
  %9572 = inttoptr i64 %9571 to float addrspace(4)*		; visa id: 12633
  %9573 = addrspacecast float addrspace(4)* %9572 to float addrspace(1)*		; visa id: 12633
  %9574 = load float, float addrspace(1)* %9573, align 4		; visa id: 12634
  %9575 = fmul reassoc nsz arcp contract float %9574, %4, !spirv.Decorations !618		; visa id: 12635
  %9576 = fadd reassoc nsz arcp contract float %9555, %9575, !spirv.Decorations !618		; visa id: 12636
  %9577 = shl i64 %9554, 2		; visa id: 12637
  %9578 = add i64 %.in, %9577		; visa id: 12638
  %9579 = inttoptr i64 %9578 to float addrspace(4)*		; visa id: 12639
  %9580 = addrspacecast float addrspace(4)* %9579 to float addrspace(1)*		; visa id: 12639
  store float %9576, float addrspace(1)* %9580, align 4		; visa id: 12640
  br label %._crit_edge70.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12641

._crit_edge70.13:                                 ; preds = %.._crit_edge70.13_crit_edge, %9561, %9556
; BB1009 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12642
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12642
  %9581 = insertelement <2 x i32> %6495, i32 %9521, i64 1		; visa id: 12642
  store <2 x i32> %9581, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12645
  br label %._crit_edge388, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12647

._crit_edge388:                                   ; preds = %._crit_edge388.._crit_edge388_crit_edge, %._crit_edge70.13
; BB1010 :
  %9582 = phi i32 [ 0, %._crit_edge70.13 ], [ %9591, %._crit_edge388.._crit_edge388_crit_edge ]
  %9583 = zext i32 %9582 to i64		; visa id: 12648
  %9584 = shl nuw nsw i64 %9583, 2		; visa id: 12649
  %9585 = add i64 %6430, %9584		; visa id: 12650
  %9586 = inttoptr i64 %9585 to i32*		; visa id: 12651
  %9587 = load i32, i32* %9586, align 4, !noalias !642		; visa id: 12651
  %9588 = add i64 %6426, %9584		; visa id: 12652
  %9589 = inttoptr i64 %9588 to i32*		; visa id: 12653
  store i32 %9587, i32* %9589, align 4, !alias.scope !642		; visa id: 12653
  %9590 = icmp eq i32 %9582, 0		; visa id: 12654
  br i1 %9590, label %._crit_edge388.._crit_edge388_crit_edge, label %9592, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12655

._crit_edge388.._crit_edge388_crit_edge:          ; preds = %._crit_edge388
; BB1011 :
  %9591 = add nuw nsw i32 %9582, 1, !spirv.Decorations !631		; visa id: 12657
  br label %._crit_edge388, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12658

9592:                                             ; preds = %._crit_edge388
; BB1012 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12660
  %9593 = load i64, i64* %6446, align 8		; visa id: 12660
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12661
  %9594 = icmp slt i32 %6494, %const_reg_dword
  %9595 = icmp slt i32 %9521, %const_reg_dword1		; visa id: 12661
  %9596 = and i1 %9594, %9595		; visa id: 12662
  br i1 %9596, label %9597, label %.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12664

.._crit_edge70.1.13_crit_edge:                    ; preds = %9592
; BB:
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9597:                                             ; preds = %9592
; BB1014 :
  %9598 = bitcast i64 %9593 to <2 x i32>		; visa id: 12666
  %9599 = extractelement <2 x i32> %9598, i32 0		; visa id: 12668
  %9600 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9599, i32 1
  %9601 = bitcast <2 x i32> %9600 to i64		; visa id: 12668
  %9602 = ashr exact i64 %9601, 32		; visa id: 12669
  %9603 = bitcast i64 %9602 to <2 x i32>		; visa id: 12670
  %9604 = extractelement <2 x i32> %9603, i32 0		; visa id: 12674
  %9605 = extractelement <2 x i32> %9603, i32 1		; visa id: 12674
  %9606 = ashr i64 %9593, 32		; visa id: 12674
  %9607 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9604, i32 %9605, i32 %50, i32 %51)
  %9608 = extractvalue { i32, i32 } %9607, 0		; visa id: 12675
  %9609 = extractvalue { i32, i32 } %9607, 1		; visa id: 12675
  %9610 = insertelement <2 x i32> undef, i32 %9608, i32 0		; visa id: 12682
  %9611 = insertelement <2 x i32> %9610, i32 %9609, i32 1		; visa id: 12683
  %9612 = bitcast <2 x i32> %9611 to i64		; visa id: 12684
  %9613 = add nsw i64 %9612, %9606, !spirv.Decorations !649		; visa id: 12688
  %9614 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !618		; visa id: 12689
  br i1 %86, label %9620, label %9615, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12690

9615:                                             ; preds = %9597
; BB1015 :
  %9616 = shl i64 %9613, 2		; visa id: 12692
  %9617 = add i64 %.in, %9616		; visa id: 12693
  %9618 = inttoptr i64 %9617 to float addrspace(4)*		; visa id: 12694
  %9619 = addrspacecast float addrspace(4)* %9618 to float addrspace(1)*		; visa id: 12694
  store float %9614, float addrspace(1)* %9619, align 4		; visa id: 12695
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12696

9620:                                             ; preds = %9597
; BB1016 :
  %9621 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9604, i32 %9605, i32 %47, i32 %48)
  %9622 = extractvalue { i32, i32 } %9621, 0		; visa id: 12698
  %9623 = extractvalue { i32, i32 } %9621, 1		; visa id: 12698
  %9624 = insertelement <2 x i32> undef, i32 %9622, i32 0		; visa id: 12705
  %9625 = insertelement <2 x i32> %9624, i32 %9623, i32 1		; visa id: 12706
  %9626 = bitcast <2 x i32> %9625 to i64		; visa id: 12707
  %9627 = shl i64 %9626, 2		; visa id: 12711
  %9628 = add i64 %.in399, %9627		; visa id: 12712
  %9629 = shl nsw i64 %9606, 2		; visa id: 12713
  %9630 = add i64 %9628, %9629		; visa id: 12714
  %9631 = inttoptr i64 %9630 to float addrspace(4)*		; visa id: 12715
  %9632 = addrspacecast float addrspace(4)* %9631 to float addrspace(1)*		; visa id: 12715
  %9633 = load float, float addrspace(1)* %9632, align 4		; visa id: 12716
  %9634 = fmul reassoc nsz arcp contract float %9633, %4, !spirv.Decorations !618		; visa id: 12717
  %9635 = fadd reassoc nsz arcp contract float %9614, %9634, !spirv.Decorations !618		; visa id: 12718
  %9636 = shl i64 %9613, 2		; visa id: 12719
  %9637 = add i64 %.in, %9636		; visa id: 12720
  %9638 = inttoptr i64 %9637 to float addrspace(4)*		; visa id: 12721
  %9639 = addrspacecast float addrspace(4)* %9638 to float addrspace(1)*		; visa id: 12721
  store float %9635, float addrspace(1)* %9639, align 4		; visa id: 12722
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12723

._crit_edge70.1.13:                               ; preds = %.._crit_edge70.1.13_crit_edge, %9620, %9615
; BB1017 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12724
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12724
  %9640 = insertelement <2 x i32> %6556, i32 %9521, i64 1		; visa id: 12724
  store <2 x i32> %9640, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12727
  br label %._crit_edge389, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12729

._crit_edge389:                                   ; preds = %._crit_edge389.._crit_edge389_crit_edge, %._crit_edge70.1.13
; BB1018 :
  %9641 = phi i32 [ 0, %._crit_edge70.1.13 ], [ %9650, %._crit_edge389.._crit_edge389_crit_edge ]
  %9642 = zext i32 %9641 to i64		; visa id: 12730
  %9643 = shl nuw nsw i64 %9642, 2		; visa id: 12731
  %9644 = add i64 %6430, %9643		; visa id: 12732
  %9645 = inttoptr i64 %9644 to i32*		; visa id: 12733
  %9646 = load i32, i32* %9645, align 4, !noalias !642		; visa id: 12733
  %9647 = add i64 %6426, %9643		; visa id: 12734
  %9648 = inttoptr i64 %9647 to i32*		; visa id: 12735
  store i32 %9646, i32* %9648, align 4, !alias.scope !642		; visa id: 12735
  %9649 = icmp eq i32 %9641, 0		; visa id: 12736
  br i1 %9649, label %._crit_edge389.._crit_edge389_crit_edge, label %9651, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12737

._crit_edge389.._crit_edge389_crit_edge:          ; preds = %._crit_edge389
; BB1019 :
  %9650 = add nuw nsw i32 %9641, 1, !spirv.Decorations !631		; visa id: 12739
  br label %._crit_edge389, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12740

9651:                                             ; preds = %._crit_edge389
; BB1020 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12742
  %9652 = load i64, i64* %6446, align 8		; visa id: 12742
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12743
  %9653 = icmp slt i32 %6555, %const_reg_dword
  %9654 = icmp slt i32 %9521, %const_reg_dword1		; visa id: 12743
  %9655 = and i1 %9653, %9654		; visa id: 12744
  br i1 %9655, label %9656, label %.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12746

.._crit_edge70.2.13_crit_edge:                    ; preds = %9651
; BB:
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9656:                                             ; preds = %9651
; BB1022 :
  %9657 = bitcast i64 %9652 to <2 x i32>		; visa id: 12748
  %9658 = extractelement <2 x i32> %9657, i32 0		; visa id: 12750
  %9659 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9658, i32 1
  %9660 = bitcast <2 x i32> %9659 to i64		; visa id: 12750
  %9661 = ashr exact i64 %9660, 32		; visa id: 12751
  %9662 = bitcast i64 %9661 to <2 x i32>		; visa id: 12752
  %9663 = extractelement <2 x i32> %9662, i32 0		; visa id: 12756
  %9664 = extractelement <2 x i32> %9662, i32 1		; visa id: 12756
  %9665 = ashr i64 %9652, 32		; visa id: 12756
  %9666 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9663, i32 %9664, i32 %50, i32 %51)
  %9667 = extractvalue { i32, i32 } %9666, 0		; visa id: 12757
  %9668 = extractvalue { i32, i32 } %9666, 1		; visa id: 12757
  %9669 = insertelement <2 x i32> undef, i32 %9667, i32 0		; visa id: 12764
  %9670 = insertelement <2 x i32> %9669, i32 %9668, i32 1		; visa id: 12765
  %9671 = bitcast <2 x i32> %9670 to i64		; visa id: 12766
  %9672 = add nsw i64 %9671, %9665, !spirv.Decorations !649		; visa id: 12770
  %9673 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !618		; visa id: 12771
  br i1 %86, label %9679, label %9674, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12772

9674:                                             ; preds = %9656
; BB1023 :
  %9675 = shl i64 %9672, 2		; visa id: 12774
  %9676 = add i64 %.in, %9675		; visa id: 12775
  %9677 = inttoptr i64 %9676 to float addrspace(4)*		; visa id: 12776
  %9678 = addrspacecast float addrspace(4)* %9677 to float addrspace(1)*		; visa id: 12776
  store float %9673, float addrspace(1)* %9678, align 4		; visa id: 12777
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12778

9679:                                             ; preds = %9656
; BB1024 :
  %9680 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9663, i32 %9664, i32 %47, i32 %48)
  %9681 = extractvalue { i32, i32 } %9680, 0		; visa id: 12780
  %9682 = extractvalue { i32, i32 } %9680, 1		; visa id: 12780
  %9683 = insertelement <2 x i32> undef, i32 %9681, i32 0		; visa id: 12787
  %9684 = insertelement <2 x i32> %9683, i32 %9682, i32 1		; visa id: 12788
  %9685 = bitcast <2 x i32> %9684 to i64		; visa id: 12789
  %9686 = shl i64 %9685, 2		; visa id: 12793
  %9687 = add i64 %.in399, %9686		; visa id: 12794
  %9688 = shl nsw i64 %9665, 2		; visa id: 12795
  %9689 = add i64 %9687, %9688		; visa id: 12796
  %9690 = inttoptr i64 %9689 to float addrspace(4)*		; visa id: 12797
  %9691 = addrspacecast float addrspace(4)* %9690 to float addrspace(1)*		; visa id: 12797
  %9692 = load float, float addrspace(1)* %9691, align 4		; visa id: 12798
  %9693 = fmul reassoc nsz arcp contract float %9692, %4, !spirv.Decorations !618		; visa id: 12799
  %9694 = fadd reassoc nsz arcp contract float %9673, %9693, !spirv.Decorations !618		; visa id: 12800
  %9695 = shl i64 %9672, 2		; visa id: 12801
  %9696 = add i64 %.in, %9695		; visa id: 12802
  %9697 = inttoptr i64 %9696 to float addrspace(4)*		; visa id: 12803
  %9698 = addrspacecast float addrspace(4)* %9697 to float addrspace(1)*		; visa id: 12803
  store float %9694, float addrspace(1)* %9698, align 4		; visa id: 12804
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12805

._crit_edge70.2.13:                               ; preds = %.._crit_edge70.2.13_crit_edge, %9679, %9674
; BB1025 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12806
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12806
  %9699 = insertelement <2 x i32> %6617, i32 %9521, i64 1		; visa id: 12806
  store <2 x i32> %9699, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12809
  br label %._crit_edge390, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12811

._crit_edge390:                                   ; preds = %._crit_edge390.._crit_edge390_crit_edge, %._crit_edge70.2.13
; BB1026 :
  %9700 = phi i32 [ 0, %._crit_edge70.2.13 ], [ %9709, %._crit_edge390.._crit_edge390_crit_edge ]
  %9701 = zext i32 %9700 to i64		; visa id: 12812
  %9702 = shl nuw nsw i64 %9701, 2		; visa id: 12813
  %9703 = add i64 %6430, %9702		; visa id: 12814
  %9704 = inttoptr i64 %9703 to i32*		; visa id: 12815
  %9705 = load i32, i32* %9704, align 4, !noalias !642		; visa id: 12815
  %9706 = add i64 %6426, %9702		; visa id: 12816
  %9707 = inttoptr i64 %9706 to i32*		; visa id: 12817
  store i32 %9705, i32* %9707, align 4, !alias.scope !642		; visa id: 12817
  %9708 = icmp eq i32 %9700, 0		; visa id: 12818
  br i1 %9708, label %._crit_edge390.._crit_edge390_crit_edge, label %9710, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12819

._crit_edge390.._crit_edge390_crit_edge:          ; preds = %._crit_edge390
; BB1027 :
  %9709 = add nuw nsw i32 %9700, 1, !spirv.Decorations !631		; visa id: 12821
  br label %._crit_edge390, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12822

9710:                                             ; preds = %._crit_edge390
; BB1028 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12824
  %9711 = load i64, i64* %6446, align 8		; visa id: 12824
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12825
  %9712 = icmp slt i32 %6616, %const_reg_dword
  %9713 = icmp slt i32 %9521, %const_reg_dword1		; visa id: 12825
  %9714 = and i1 %9712, %9713		; visa id: 12826
  br i1 %9714, label %9715, label %..preheader1.13_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12828

..preheader1.13_crit_edge:                        ; preds = %9710
; BB:
  br label %.preheader1.13, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9715:                                             ; preds = %9710
; BB1030 :
  %9716 = bitcast i64 %9711 to <2 x i32>		; visa id: 12830
  %9717 = extractelement <2 x i32> %9716, i32 0		; visa id: 12832
  %9718 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9717, i32 1
  %9719 = bitcast <2 x i32> %9718 to i64		; visa id: 12832
  %9720 = ashr exact i64 %9719, 32		; visa id: 12833
  %9721 = bitcast i64 %9720 to <2 x i32>		; visa id: 12834
  %9722 = extractelement <2 x i32> %9721, i32 0		; visa id: 12838
  %9723 = extractelement <2 x i32> %9721, i32 1		; visa id: 12838
  %9724 = ashr i64 %9711, 32		; visa id: 12838
  %9725 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9722, i32 %9723, i32 %50, i32 %51)
  %9726 = extractvalue { i32, i32 } %9725, 0		; visa id: 12839
  %9727 = extractvalue { i32, i32 } %9725, 1		; visa id: 12839
  %9728 = insertelement <2 x i32> undef, i32 %9726, i32 0		; visa id: 12846
  %9729 = insertelement <2 x i32> %9728, i32 %9727, i32 1		; visa id: 12847
  %9730 = bitcast <2 x i32> %9729 to i64		; visa id: 12848
  %9731 = add nsw i64 %9730, %9724, !spirv.Decorations !649		; visa id: 12852
  %9732 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !618		; visa id: 12853
  br i1 %86, label %9738, label %9733, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12854

9733:                                             ; preds = %9715
; BB1031 :
  %9734 = shl i64 %9731, 2		; visa id: 12856
  %9735 = add i64 %.in, %9734		; visa id: 12857
  %9736 = inttoptr i64 %9735 to float addrspace(4)*		; visa id: 12858
  %9737 = addrspacecast float addrspace(4)* %9736 to float addrspace(1)*		; visa id: 12858
  store float %9732, float addrspace(1)* %9737, align 4		; visa id: 12859
  br label %.preheader1.13, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12860

9738:                                             ; preds = %9715
; BB1032 :
  %9739 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9722, i32 %9723, i32 %47, i32 %48)
  %9740 = extractvalue { i32, i32 } %9739, 0		; visa id: 12862
  %9741 = extractvalue { i32, i32 } %9739, 1		; visa id: 12862
  %9742 = insertelement <2 x i32> undef, i32 %9740, i32 0		; visa id: 12869
  %9743 = insertelement <2 x i32> %9742, i32 %9741, i32 1		; visa id: 12870
  %9744 = bitcast <2 x i32> %9743 to i64		; visa id: 12871
  %9745 = shl i64 %9744, 2		; visa id: 12875
  %9746 = add i64 %.in399, %9745		; visa id: 12876
  %9747 = shl nsw i64 %9724, 2		; visa id: 12877
  %9748 = add i64 %9746, %9747		; visa id: 12878
  %9749 = inttoptr i64 %9748 to float addrspace(4)*		; visa id: 12879
  %9750 = addrspacecast float addrspace(4)* %9749 to float addrspace(1)*		; visa id: 12879
  %9751 = load float, float addrspace(1)* %9750, align 4		; visa id: 12880
  %9752 = fmul reassoc nsz arcp contract float %9751, %4, !spirv.Decorations !618		; visa id: 12881
  %9753 = fadd reassoc nsz arcp contract float %9732, %9752, !spirv.Decorations !618		; visa id: 12882
  %9754 = shl i64 %9731, 2		; visa id: 12883
  %9755 = add i64 %.in, %9754		; visa id: 12884
  %9756 = inttoptr i64 %9755 to float addrspace(4)*		; visa id: 12885
  %9757 = addrspacecast float addrspace(4)* %9756 to float addrspace(1)*		; visa id: 12885
  store float %9753, float addrspace(1)* %9757, align 4		; visa id: 12886
  br label %.preheader1.13, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12887

.preheader1.13:                                   ; preds = %..preheader1.13_crit_edge, %9738, %9733
; BB1033 :
  %9758 = add i32 %69, 14		; visa id: 12888
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12889
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12889
  %9759 = insertelement <2 x i32> %6432, i32 %9758, i64 1		; visa id: 12889
  store <2 x i32> %9759, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12892
  br label %._crit_edge391, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12894

._crit_edge391:                                   ; preds = %._crit_edge391.._crit_edge391_crit_edge, %.preheader1.13
; BB1034 :
  %9760 = phi i32 [ 0, %.preheader1.13 ], [ %9769, %._crit_edge391.._crit_edge391_crit_edge ]
  %9761 = zext i32 %9760 to i64		; visa id: 12895
  %9762 = shl nuw nsw i64 %9761, 2		; visa id: 12896
  %9763 = add i64 %6430, %9762		; visa id: 12897
  %9764 = inttoptr i64 %9763 to i32*		; visa id: 12898
  %9765 = load i32, i32* %9764, align 4, !noalias !642		; visa id: 12898
  %9766 = add i64 %6426, %9762		; visa id: 12899
  %9767 = inttoptr i64 %9766 to i32*		; visa id: 12900
  store i32 %9765, i32* %9767, align 4, !alias.scope !642		; visa id: 12900
  %9768 = icmp eq i32 %9760, 0		; visa id: 12901
  br i1 %9768, label %._crit_edge391.._crit_edge391_crit_edge, label %9770, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12902

._crit_edge391.._crit_edge391_crit_edge:          ; preds = %._crit_edge391
; BB1035 :
  %9769 = add nuw nsw i32 %9760, 1, !spirv.Decorations !631		; visa id: 12904
  br label %._crit_edge391, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12905

9770:                                             ; preds = %._crit_edge391
; BB1036 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12907
  %9771 = load i64, i64* %6446, align 8		; visa id: 12907
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12908
  %9772 = icmp slt i32 %9758, %const_reg_dword1		; visa id: 12908
  %9773 = icmp slt i32 %65, %const_reg_dword
  %9774 = and i1 %9773, %9772		; visa id: 12909
  br i1 %9774, label %9775, label %.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12911

.._crit_edge70.14_crit_edge:                      ; preds = %9770
; BB:
  br label %._crit_edge70.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9775:                                             ; preds = %9770
; BB1038 :
  %9776 = bitcast i64 %9771 to <2 x i32>		; visa id: 12913
  %9777 = extractelement <2 x i32> %9776, i32 0		; visa id: 12915
  %9778 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9777, i32 1
  %9779 = bitcast <2 x i32> %9778 to i64		; visa id: 12915
  %9780 = ashr exact i64 %9779, 32		; visa id: 12916
  %9781 = bitcast i64 %9780 to <2 x i32>		; visa id: 12917
  %9782 = extractelement <2 x i32> %9781, i32 0		; visa id: 12921
  %9783 = extractelement <2 x i32> %9781, i32 1		; visa id: 12921
  %9784 = ashr i64 %9771, 32		; visa id: 12921
  %9785 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9782, i32 %9783, i32 %50, i32 %51)
  %9786 = extractvalue { i32, i32 } %9785, 0		; visa id: 12922
  %9787 = extractvalue { i32, i32 } %9785, 1		; visa id: 12922
  %9788 = insertelement <2 x i32> undef, i32 %9786, i32 0		; visa id: 12929
  %9789 = insertelement <2 x i32> %9788, i32 %9787, i32 1		; visa id: 12930
  %9790 = bitcast <2 x i32> %9789 to i64		; visa id: 12931
  %9791 = add nsw i64 %9790, %9784, !spirv.Decorations !649		; visa id: 12935
  %9792 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !618		; visa id: 12936
  br i1 %86, label %9798, label %9793, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 12937

9793:                                             ; preds = %9775
; BB1039 :
  %9794 = shl i64 %9791, 2		; visa id: 12939
  %9795 = add i64 %.in, %9794		; visa id: 12940
  %9796 = inttoptr i64 %9795 to float addrspace(4)*		; visa id: 12941
  %9797 = addrspacecast float addrspace(4)* %9796 to float addrspace(1)*		; visa id: 12941
  store float %9792, float addrspace(1)* %9797, align 4		; visa id: 12942
  br label %._crit_edge70.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 12943

9798:                                             ; preds = %9775
; BB1040 :
  %9799 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9782, i32 %9783, i32 %47, i32 %48)
  %9800 = extractvalue { i32, i32 } %9799, 0		; visa id: 12945
  %9801 = extractvalue { i32, i32 } %9799, 1		; visa id: 12945
  %9802 = insertelement <2 x i32> undef, i32 %9800, i32 0		; visa id: 12952
  %9803 = insertelement <2 x i32> %9802, i32 %9801, i32 1		; visa id: 12953
  %9804 = bitcast <2 x i32> %9803 to i64		; visa id: 12954
  %9805 = shl i64 %9804, 2		; visa id: 12958
  %9806 = add i64 %.in399, %9805		; visa id: 12959
  %9807 = shl nsw i64 %9784, 2		; visa id: 12960
  %9808 = add i64 %9806, %9807		; visa id: 12961
  %9809 = inttoptr i64 %9808 to float addrspace(4)*		; visa id: 12962
  %9810 = addrspacecast float addrspace(4)* %9809 to float addrspace(1)*		; visa id: 12962
  %9811 = load float, float addrspace(1)* %9810, align 4		; visa id: 12963
  %9812 = fmul reassoc nsz arcp contract float %9811, %4, !spirv.Decorations !618		; visa id: 12964
  %9813 = fadd reassoc nsz arcp contract float %9792, %9812, !spirv.Decorations !618		; visa id: 12965
  %9814 = shl i64 %9791, 2		; visa id: 12966
  %9815 = add i64 %.in, %9814		; visa id: 12967
  %9816 = inttoptr i64 %9815 to float addrspace(4)*		; visa id: 12968
  %9817 = addrspacecast float addrspace(4)* %9816 to float addrspace(1)*		; visa id: 12968
  store float %9813, float addrspace(1)* %9817, align 4		; visa id: 12969
  br label %._crit_edge70.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 12970

._crit_edge70.14:                                 ; preds = %.._crit_edge70.14_crit_edge, %9798, %9793
; BB1041 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12971
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12971
  %9818 = insertelement <2 x i32> %6495, i32 %9758, i64 1		; visa id: 12971
  store <2 x i32> %9818, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 12974
  br label %._crit_edge392, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12976

._crit_edge392:                                   ; preds = %._crit_edge392.._crit_edge392_crit_edge, %._crit_edge70.14
; BB1042 :
  %9819 = phi i32 [ 0, %._crit_edge70.14 ], [ %9828, %._crit_edge392.._crit_edge392_crit_edge ]
  %9820 = zext i32 %9819 to i64		; visa id: 12977
  %9821 = shl nuw nsw i64 %9820, 2		; visa id: 12978
  %9822 = add i64 %6430, %9821		; visa id: 12979
  %9823 = inttoptr i64 %9822 to i32*		; visa id: 12980
  %9824 = load i32, i32* %9823, align 4, !noalias !642		; visa id: 12980
  %9825 = add i64 %6426, %9821		; visa id: 12981
  %9826 = inttoptr i64 %9825 to i32*		; visa id: 12982
  store i32 %9824, i32* %9826, align 4, !alias.scope !642		; visa id: 12982
  %9827 = icmp eq i32 %9819, 0		; visa id: 12983
  br i1 %9827, label %._crit_edge392.._crit_edge392_crit_edge, label %9829, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 12984

._crit_edge392.._crit_edge392_crit_edge:          ; preds = %._crit_edge392
; BB1043 :
  %9828 = add nuw nsw i32 %9819, 1, !spirv.Decorations !631		; visa id: 12986
  br label %._crit_edge392, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 12987

9829:                                             ; preds = %._crit_edge392
; BB1044 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 12989
  %9830 = load i64, i64* %6446, align 8		; visa id: 12989
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 12990
  %9831 = icmp slt i32 %6494, %const_reg_dword
  %9832 = icmp slt i32 %9758, %const_reg_dword1		; visa id: 12990
  %9833 = and i1 %9831, %9832		; visa id: 12991
  br i1 %9833, label %9834, label %.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 12993

.._crit_edge70.1.14_crit_edge:                    ; preds = %9829
; BB:
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9834:                                             ; preds = %9829
; BB1046 :
  %9835 = bitcast i64 %9830 to <2 x i32>		; visa id: 12995
  %9836 = extractelement <2 x i32> %9835, i32 0		; visa id: 12997
  %9837 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9836, i32 1
  %9838 = bitcast <2 x i32> %9837 to i64		; visa id: 12997
  %9839 = ashr exact i64 %9838, 32		; visa id: 12998
  %9840 = bitcast i64 %9839 to <2 x i32>		; visa id: 12999
  %9841 = extractelement <2 x i32> %9840, i32 0		; visa id: 13003
  %9842 = extractelement <2 x i32> %9840, i32 1		; visa id: 13003
  %9843 = ashr i64 %9830, 32		; visa id: 13003
  %9844 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9841, i32 %9842, i32 %50, i32 %51)
  %9845 = extractvalue { i32, i32 } %9844, 0		; visa id: 13004
  %9846 = extractvalue { i32, i32 } %9844, 1		; visa id: 13004
  %9847 = insertelement <2 x i32> undef, i32 %9845, i32 0		; visa id: 13011
  %9848 = insertelement <2 x i32> %9847, i32 %9846, i32 1		; visa id: 13012
  %9849 = bitcast <2 x i32> %9848 to i64		; visa id: 13013
  %9850 = add nsw i64 %9849, %9843, !spirv.Decorations !649		; visa id: 13017
  %9851 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !618		; visa id: 13018
  br i1 %86, label %9857, label %9852, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13019

9852:                                             ; preds = %9834
; BB1047 :
  %9853 = shl i64 %9850, 2		; visa id: 13021
  %9854 = add i64 %.in, %9853		; visa id: 13022
  %9855 = inttoptr i64 %9854 to float addrspace(4)*		; visa id: 13023
  %9856 = addrspacecast float addrspace(4)* %9855 to float addrspace(1)*		; visa id: 13023
  store float %9851, float addrspace(1)* %9856, align 4		; visa id: 13024
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13025

9857:                                             ; preds = %9834
; BB1048 :
  %9858 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9841, i32 %9842, i32 %47, i32 %48)
  %9859 = extractvalue { i32, i32 } %9858, 0		; visa id: 13027
  %9860 = extractvalue { i32, i32 } %9858, 1		; visa id: 13027
  %9861 = insertelement <2 x i32> undef, i32 %9859, i32 0		; visa id: 13034
  %9862 = insertelement <2 x i32> %9861, i32 %9860, i32 1		; visa id: 13035
  %9863 = bitcast <2 x i32> %9862 to i64		; visa id: 13036
  %9864 = shl i64 %9863, 2		; visa id: 13040
  %9865 = add i64 %.in399, %9864		; visa id: 13041
  %9866 = shl nsw i64 %9843, 2		; visa id: 13042
  %9867 = add i64 %9865, %9866		; visa id: 13043
  %9868 = inttoptr i64 %9867 to float addrspace(4)*		; visa id: 13044
  %9869 = addrspacecast float addrspace(4)* %9868 to float addrspace(1)*		; visa id: 13044
  %9870 = load float, float addrspace(1)* %9869, align 4		; visa id: 13045
  %9871 = fmul reassoc nsz arcp contract float %9870, %4, !spirv.Decorations !618		; visa id: 13046
  %9872 = fadd reassoc nsz arcp contract float %9851, %9871, !spirv.Decorations !618		; visa id: 13047
  %9873 = shl i64 %9850, 2		; visa id: 13048
  %9874 = add i64 %.in, %9873		; visa id: 13049
  %9875 = inttoptr i64 %9874 to float addrspace(4)*		; visa id: 13050
  %9876 = addrspacecast float addrspace(4)* %9875 to float addrspace(1)*		; visa id: 13050
  store float %9872, float addrspace(1)* %9876, align 4		; visa id: 13051
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13052

._crit_edge70.1.14:                               ; preds = %.._crit_edge70.1.14_crit_edge, %9857, %9852
; BB1049 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13053
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13053
  %9877 = insertelement <2 x i32> %6556, i32 %9758, i64 1		; visa id: 13053
  store <2 x i32> %9877, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13056
  br label %._crit_edge393, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13058

._crit_edge393:                                   ; preds = %._crit_edge393.._crit_edge393_crit_edge, %._crit_edge70.1.14
; BB1050 :
  %9878 = phi i32 [ 0, %._crit_edge70.1.14 ], [ %9887, %._crit_edge393.._crit_edge393_crit_edge ]
  %9879 = zext i32 %9878 to i64		; visa id: 13059
  %9880 = shl nuw nsw i64 %9879, 2		; visa id: 13060
  %9881 = add i64 %6430, %9880		; visa id: 13061
  %9882 = inttoptr i64 %9881 to i32*		; visa id: 13062
  %9883 = load i32, i32* %9882, align 4, !noalias !642		; visa id: 13062
  %9884 = add i64 %6426, %9880		; visa id: 13063
  %9885 = inttoptr i64 %9884 to i32*		; visa id: 13064
  store i32 %9883, i32* %9885, align 4, !alias.scope !642		; visa id: 13064
  %9886 = icmp eq i32 %9878, 0		; visa id: 13065
  br i1 %9886, label %._crit_edge393.._crit_edge393_crit_edge, label %9888, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13066

._crit_edge393.._crit_edge393_crit_edge:          ; preds = %._crit_edge393
; BB1051 :
  %9887 = add nuw nsw i32 %9878, 1, !spirv.Decorations !631		; visa id: 13068
  br label %._crit_edge393, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13069

9888:                                             ; preds = %._crit_edge393
; BB1052 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13071
  %9889 = load i64, i64* %6446, align 8		; visa id: 13071
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13072
  %9890 = icmp slt i32 %6555, %const_reg_dword
  %9891 = icmp slt i32 %9758, %const_reg_dword1		; visa id: 13072
  %9892 = and i1 %9890, %9891		; visa id: 13073
  br i1 %9892, label %9893, label %.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13075

.._crit_edge70.2.14_crit_edge:                    ; preds = %9888
; BB:
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9893:                                             ; preds = %9888
; BB1054 :
  %9894 = bitcast i64 %9889 to <2 x i32>		; visa id: 13077
  %9895 = extractelement <2 x i32> %9894, i32 0		; visa id: 13079
  %9896 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9895, i32 1
  %9897 = bitcast <2 x i32> %9896 to i64		; visa id: 13079
  %9898 = ashr exact i64 %9897, 32		; visa id: 13080
  %9899 = bitcast i64 %9898 to <2 x i32>		; visa id: 13081
  %9900 = extractelement <2 x i32> %9899, i32 0		; visa id: 13085
  %9901 = extractelement <2 x i32> %9899, i32 1		; visa id: 13085
  %9902 = ashr i64 %9889, 32		; visa id: 13085
  %9903 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9900, i32 %9901, i32 %50, i32 %51)
  %9904 = extractvalue { i32, i32 } %9903, 0		; visa id: 13086
  %9905 = extractvalue { i32, i32 } %9903, 1		; visa id: 13086
  %9906 = insertelement <2 x i32> undef, i32 %9904, i32 0		; visa id: 13093
  %9907 = insertelement <2 x i32> %9906, i32 %9905, i32 1		; visa id: 13094
  %9908 = bitcast <2 x i32> %9907 to i64		; visa id: 13095
  %9909 = add nsw i64 %9908, %9902, !spirv.Decorations !649		; visa id: 13099
  %9910 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !618		; visa id: 13100
  br i1 %86, label %9916, label %9911, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13101

9911:                                             ; preds = %9893
; BB1055 :
  %9912 = shl i64 %9909, 2		; visa id: 13103
  %9913 = add i64 %.in, %9912		; visa id: 13104
  %9914 = inttoptr i64 %9913 to float addrspace(4)*		; visa id: 13105
  %9915 = addrspacecast float addrspace(4)* %9914 to float addrspace(1)*		; visa id: 13105
  store float %9910, float addrspace(1)* %9915, align 4		; visa id: 13106
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13107

9916:                                             ; preds = %9893
; BB1056 :
  %9917 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9900, i32 %9901, i32 %47, i32 %48)
  %9918 = extractvalue { i32, i32 } %9917, 0		; visa id: 13109
  %9919 = extractvalue { i32, i32 } %9917, 1		; visa id: 13109
  %9920 = insertelement <2 x i32> undef, i32 %9918, i32 0		; visa id: 13116
  %9921 = insertelement <2 x i32> %9920, i32 %9919, i32 1		; visa id: 13117
  %9922 = bitcast <2 x i32> %9921 to i64		; visa id: 13118
  %9923 = shl i64 %9922, 2		; visa id: 13122
  %9924 = add i64 %.in399, %9923		; visa id: 13123
  %9925 = shl nsw i64 %9902, 2		; visa id: 13124
  %9926 = add i64 %9924, %9925		; visa id: 13125
  %9927 = inttoptr i64 %9926 to float addrspace(4)*		; visa id: 13126
  %9928 = addrspacecast float addrspace(4)* %9927 to float addrspace(1)*		; visa id: 13126
  %9929 = load float, float addrspace(1)* %9928, align 4		; visa id: 13127
  %9930 = fmul reassoc nsz arcp contract float %9929, %4, !spirv.Decorations !618		; visa id: 13128
  %9931 = fadd reassoc nsz arcp contract float %9910, %9930, !spirv.Decorations !618		; visa id: 13129
  %9932 = shl i64 %9909, 2		; visa id: 13130
  %9933 = add i64 %.in, %9932		; visa id: 13131
  %9934 = inttoptr i64 %9933 to float addrspace(4)*		; visa id: 13132
  %9935 = addrspacecast float addrspace(4)* %9934 to float addrspace(1)*		; visa id: 13132
  store float %9931, float addrspace(1)* %9935, align 4		; visa id: 13133
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13134

._crit_edge70.2.14:                               ; preds = %.._crit_edge70.2.14_crit_edge, %9916, %9911
; BB1057 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13135
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13135
  %9936 = insertelement <2 x i32> %6617, i32 %9758, i64 1		; visa id: 13135
  store <2 x i32> %9936, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13138
  br label %._crit_edge394, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13140

._crit_edge394:                                   ; preds = %._crit_edge394.._crit_edge394_crit_edge, %._crit_edge70.2.14
; BB1058 :
  %9937 = phi i32 [ 0, %._crit_edge70.2.14 ], [ %9946, %._crit_edge394.._crit_edge394_crit_edge ]
  %9938 = zext i32 %9937 to i64		; visa id: 13141
  %9939 = shl nuw nsw i64 %9938, 2		; visa id: 13142
  %9940 = add i64 %6430, %9939		; visa id: 13143
  %9941 = inttoptr i64 %9940 to i32*		; visa id: 13144
  %9942 = load i32, i32* %9941, align 4, !noalias !642		; visa id: 13144
  %9943 = add i64 %6426, %9939		; visa id: 13145
  %9944 = inttoptr i64 %9943 to i32*		; visa id: 13146
  store i32 %9942, i32* %9944, align 4, !alias.scope !642		; visa id: 13146
  %9945 = icmp eq i32 %9937, 0		; visa id: 13147
  br i1 %9945, label %._crit_edge394.._crit_edge394_crit_edge, label %9947, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13148

._crit_edge394.._crit_edge394_crit_edge:          ; preds = %._crit_edge394
; BB1059 :
  %9946 = add nuw nsw i32 %9937, 1, !spirv.Decorations !631		; visa id: 13150
  br label %._crit_edge394, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13151

9947:                                             ; preds = %._crit_edge394
; BB1060 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13153
  %9948 = load i64, i64* %6446, align 8		; visa id: 13153
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13154
  %9949 = icmp slt i32 %6616, %const_reg_dword
  %9950 = icmp slt i32 %9758, %const_reg_dword1		; visa id: 13154
  %9951 = and i1 %9949, %9950		; visa id: 13155
  br i1 %9951, label %9952, label %..preheader1.14_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13157

..preheader1.14_crit_edge:                        ; preds = %9947
; BB:
  br label %.preheader1.14, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

9952:                                             ; preds = %9947
; BB1062 :
  %9953 = bitcast i64 %9948 to <2 x i32>		; visa id: 13159
  %9954 = extractelement <2 x i32> %9953, i32 0		; visa id: 13161
  %9955 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %9954, i32 1
  %9956 = bitcast <2 x i32> %9955 to i64		; visa id: 13161
  %9957 = ashr exact i64 %9956, 32		; visa id: 13162
  %9958 = bitcast i64 %9957 to <2 x i32>		; visa id: 13163
  %9959 = extractelement <2 x i32> %9958, i32 0		; visa id: 13167
  %9960 = extractelement <2 x i32> %9958, i32 1		; visa id: 13167
  %9961 = ashr i64 %9948, 32		; visa id: 13167
  %9962 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9959, i32 %9960, i32 %50, i32 %51)
  %9963 = extractvalue { i32, i32 } %9962, 0		; visa id: 13168
  %9964 = extractvalue { i32, i32 } %9962, 1		; visa id: 13168
  %9965 = insertelement <2 x i32> undef, i32 %9963, i32 0		; visa id: 13175
  %9966 = insertelement <2 x i32> %9965, i32 %9964, i32 1		; visa id: 13176
  %9967 = bitcast <2 x i32> %9966 to i64		; visa id: 13177
  %9968 = add nsw i64 %9967, %9961, !spirv.Decorations !649		; visa id: 13181
  %9969 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !618		; visa id: 13182
  br i1 %86, label %9975, label %9970, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13183

9970:                                             ; preds = %9952
; BB1063 :
  %9971 = shl i64 %9968, 2		; visa id: 13185
  %9972 = add i64 %.in, %9971		; visa id: 13186
  %9973 = inttoptr i64 %9972 to float addrspace(4)*		; visa id: 13187
  %9974 = addrspacecast float addrspace(4)* %9973 to float addrspace(1)*		; visa id: 13187
  store float %9969, float addrspace(1)* %9974, align 4		; visa id: 13188
  br label %.preheader1.14, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13189

9975:                                             ; preds = %9952
; BB1064 :
  %9976 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %9959, i32 %9960, i32 %47, i32 %48)
  %9977 = extractvalue { i32, i32 } %9976, 0		; visa id: 13191
  %9978 = extractvalue { i32, i32 } %9976, 1		; visa id: 13191
  %9979 = insertelement <2 x i32> undef, i32 %9977, i32 0		; visa id: 13198
  %9980 = insertelement <2 x i32> %9979, i32 %9978, i32 1		; visa id: 13199
  %9981 = bitcast <2 x i32> %9980 to i64		; visa id: 13200
  %9982 = shl i64 %9981, 2		; visa id: 13204
  %9983 = add i64 %.in399, %9982		; visa id: 13205
  %9984 = shl nsw i64 %9961, 2		; visa id: 13206
  %9985 = add i64 %9983, %9984		; visa id: 13207
  %9986 = inttoptr i64 %9985 to float addrspace(4)*		; visa id: 13208
  %9987 = addrspacecast float addrspace(4)* %9986 to float addrspace(1)*		; visa id: 13208
  %9988 = load float, float addrspace(1)* %9987, align 4		; visa id: 13209
  %9989 = fmul reassoc nsz arcp contract float %9988, %4, !spirv.Decorations !618		; visa id: 13210
  %9990 = fadd reassoc nsz arcp contract float %9969, %9989, !spirv.Decorations !618		; visa id: 13211
  %9991 = shl i64 %9968, 2		; visa id: 13212
  %9992 = add i64 %.in, %9991		; visa id: 13213
  %9993 = inttoptr i64 %9992 to float addrspace(4)*		; visa id: 13214
  %9994 = addrspacecast float addrspace(4)* %9993 to float addrspace(1)*		; visa id: 13214
  store float %9990, float addrspace(1)* %9994, align 4		; visa id: 13215
  br label %.preheader1.14, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13216

.preheader1.14:                                   ; preds = %..preheader1.14_crit_edge, %9975, %9970
; BB1065 :
  %9995 = add i32 %69, 15		; visa id: 13217
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13218
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13218
  %9996 = insertelement <2 x i32> %6432, i32 %9995, i64 1		; visa id: 13218
  store <2 x i32> %9996, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13219
  br label %._crit_edge395, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13221

._crit_edge395:                                   ; preds = %._crit_edge395.._crit_edge395_crit_edge, %.preheader1.14
; BB1066 :
  %9997 = phi i32 [ 0, %.preheader1.14 ], [ %10006, %._crit_edge395.._crit_edge395_crit_edge ]
  %9998 = zext i32 %9997 to i64		; visa id: 13222
  %9999 = shl nuw nsw i64 %9998, 2		; visa id: 13223
  %10000 = add i64 %6430, %9999		; visa id: 13224
  %10001 = inttoptr i64 %10000 to i32*		; visa id: 13225
  %10002 = load i32, i32* %10001, align 4, !noalias !642		; visa id: 13225
  %10003 = add i64 %6426, %9999		; visa id: 13226
  %10004 = inttoptr i64 %10003 to i32*		; visa id: 13227
  store i32 %10002, i32* %10004, align 4, !alias.scope !642		; visa id: 13227
  %10005 = icmp eq i32 %9997, 0		; visa id: 13228
  br i1 %10005, label %._crit_edge395.._crit_edge395_crit_edge, label %10007, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13229

._crit_edge395.._crit_edge395_crit_edge:          ; preds = %._crit_edge395
; BB1067 :
  %10006 = add nuw nsw i32 %9997, 1, !spirv.Decorations !631		; visa id: 13231
  br label %._crit_edge395, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13232

10007:                                            ; preds = %._crit_edge395
; BB1068 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13234
  %10008 = load i64, i64* %6446, align 8		; visa id: 13234
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13235
  %10009 = icmp slt i32 %9995, %const_reg_dword1		; visa id: 13235
  %10010 = icmp slt i32 %65, %const_reg_dword
  %10011 = and i1 %10010, %10009		; visa id: 13236
  br i1 %10011, label %10012, label %.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13238

.._crit_edge70.15_crit_edge:                      ; preds = %10007
; BB:
  br label %._crit_edge70.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

10012:                                            ; preds = %10007
; BB1070 :
  %10013 = bitcast i64 %10008 to <2 x i32>		; visa id: 13240
  %10014 = extractelement <2 x i32> %10013, i32 0		; visa id: 13242
  %10015 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %10014, i32 1
  %10016 = bitcast <2 x i32> %10015 to i64		; visa id: 13242
  %10017 = ashr exact i64 %10016, 32		; visa id: 13243
  %10018 = bitcast i64 %10017 to <2 x i32>		; visa id: 13244
  %10019 = extractelement <2 x i32> %10018, i32 0		; visa id: 13248
  %10020 = extractelement <2 x i32> %10018, i32 1		; visa id: 13248
  %10021 = ashr i64 %10008, 32		; visa id: 13248
  %10022 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10019, i32 %10020, i32 %50, i32 %51)
  %10023 = extractvalue { i32, i32 } %10022, 0		; visa id: 13249
  %10024 = extractvalue { i32, i32 } %10022, 1		; visa id: 13249
  %10025 = insertelement <2 x i32> undef, i32 %10023, i32 0		; visa id: 13256
  %10026 = insertelement <2 x i32> %10025, i32 %10024, i32 1		; visa id: 13257
  %10027 = bitcast <2 x i32> %10026 to i64		; visa id: 13258
  %10028 = add nsw i64 %10027, %10021, !spirv.Decorations !649		; visa id: 13262
  %10029 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !618		; visa id: 13263
  br i1 %86, label %10035, label %10030, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13264

10030:                                            ; preds = %10012
; BB1071 :
  %10031 = shl i64 %10028, 2		; visa id: 13266
  %10032 = add i64 %.in, %10031		; visa id: 13267
  %10033 = inttoptr i64 %10032 to float addrspace(4)*		; visa id: 13268
  %10034 = addrspacecast float addrspace(4)* %10033 to float addrspace(1)*		; visa id: 13268
  store float %10029, float addrspace(1)* %10034, align 4		; visa id: 13269
  br label %._crit_edge70.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13270

10035:                                            ; preds = %10012
; BB1072 :
  %10036 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10019, i32 %10020, i32 %47, i32 %48)
  %10037 = extractvalue { i32, i32 } %10036, 0		; visa id: 13272
  %10038 = extractvalue { i32, i32 } %10036, 1		; visa id: 13272
  %10039 = insertelement <2 x i32> undef, i32 %10037, i32 0		; visa id: 13279
  %10040 = insertelement <2 x i32> %10039, i32 %10038, i32 1		; visa id: 13280
  %10041 = bitcast <2 x i32> %10040 to i64		; visa id: 13281
  %10042 = shl i64 %10041, 2		; visa id: 13285
  %10043 = add i64 %.in399, %10042		; visa id: 13286
  %10044 = shl nsw i64 %10021, 2		; visa id: 13287
  %10045 = add i64 %10043, %10044		; visa id: 13288
  %10046 = inttoptr i64 %10045 to float addrspace(4)*		; visa id: 13289
  %10047 = addrspacecast float addrspace(4)* %10046 to float addrspace(1)*		; visa id: 13289
  %10048 = load float, float addrspace(1)* %10047, align 4		; visa id: 13290
  %10049 = fmul reassoc nsz arcp contract float %10048, %4, !spirv.Decorations !618		; visa id: 13291
  %10050 = fadd reassoc nsz arcp contract float %10029, %10049, !spirv.Decorations !618		; visa id: 13292
  %10051 = shl i64 %10028, 2		; visa id: 13293
  %10052 = add i64 %.in, %10051		; visa id: 13294
  %10053 = inttoptr i64 %10052 to float addrspace(4)*		; visa id: 13295
  %10054 = addrspacecast float addrspace(4)* %10053 to float addrspace(1)*		; visa id: 13295
  store float %10050, float addrspace(1)* %10054, align 4		; visa id: 13296
  br label %._crit_edge70.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13297

._crit_edge70.15:                                 ; preds = %.._crit_edge70.15_crit_edge, %10035, %10030
; BB1073 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13298
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13298
  %10055 = insertelement <2 x i32> %6495, i32 %9995, i64 1		; visa id: 13298
  store <2 x i32> %10055, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13299
  br label %._crit_edge396, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13301

._crit_edge396:                                   ; preds = %._crit_edge396.._crit_edge396_crit_edge, %._crit_edge70.15
; BB1074 :
  %10056 = phi i32 [ 0, %._crit_edge70.15 ], [ %10065, %._crit_edge396.._crit_edge396_crit_edge ]
  %10057 = zext i32 %10056 to i64		; visa id: 13302
  %10058 = shl nuw nsw i64 %10057, 2		; visa id: 13303
  %10059 = add i64 %6430, %10058		; visa id: 13304
  %10060 = inttoptr i64 %10059 to i32*		; visa id: 13305
  %10061 = load i32, i32* %10060, align 4, !noalias !642		; visa id: 13305
  %10062 = add i64 %6426, %10058		; visa id: 13306
  %10063 = inttoptr i64 %10062 to i32*		; visa id: 13307
  store i32 %10061, i32* %10063, align 4, !alias.scope !642		; visa id: 13307
  %10064 = icmp eq i32 %10056, 0		; visa id: 13308
  br i1 %10064, label %._crit_edge396.._crit_edge396_crit_edge, label %10066, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13309

._crit_edge396.._crit_edge396_crit_edge:          ; preds = %._crit_edge396
; BB1075 :
  %10065 = add nuw nsw i32 %10056, 1, !spirv.Decorations !631		; visa id: 13311
  br label %._crit_edge396, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13312

10066:                                            ; preds = %._crit_edge396
; BB1076 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13314
  %10067 = load i64, i64* %6446, align 8		; visa id: 13314
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13315
  %10068 = icmp slt i32 %6494, %const_reg_dword
  %10069 = icmp slt i32 %9995, %const_reg_dword1		; visa id: 13315
  %10070 = and i1 %10068, %10069		; visa id: 13316
  br i1 %10070, label %10071, label %.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13318

.._crit_edge70.1.15_crit_edge:                    ; preds = %10066
; BB:
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

10071:                                            ; preds = %10066
; BB1078 :
  %10072 = bitcast i64 %10067 to <2 x i32>		; visa id: 13320
  %10073 = extractelement <2 x i32> %10072, i32 0		; visa id: 13322
  %10074 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %10073, i32 1
  %10075 = bitcast <2 x i32> %10074 to i64		; visa id: 13322
  %10076 = ashr exact i64 %10075, 32		; visa id: 13323
  %10077 = bitcast i64 %10076 to <2 x i32>		; visa id: 13324
  %10078 = extractelement <2 x i32> %10077, i32 0		; visa id: 13328
  %10079 = extractelement <2 x i32> %10077, i32 1		; visa id: 13328
  %10080 = ashr i64 %10067, 32		; visa id: 13328
  %10081 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10078, i32 %10079, i32 %50, i32 %51)
  %10082 = extractvalue { i32, i32 } %10081, 0		; visa id: 13329
  %10083 = extractvalue { i32, i32 } %10081, 1		; visa id: 13329
  %10084 = insertelement <2 x i32> undef, i32 %10082, i32 0		; visa id: 13336
  %10085 = insertelement <2 x i32> %10084, i32 %10083, i32 1		; visa id: 13337
  %10086 = bitcast <2 x i32> %10085 to i64		; visa id: 13338
  %10087 = add nsw i64 %10086, %10080, !spirv.Decorations !649		; visa id: 13342
  %10088 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !618		; visa id: 13343
  br i1 %86, label %10094, label %10089, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13344

10089:                                            ; preds = %10071
; BB1079 :
  %10090 = shl i64 %10087, 2		; visa id: 13346
  %10091 = add i64 %.in, %10090		; visa id: 13347
  %10092 = inttoptr i64 %10091 to float addrspace(4)*		; visa id: 13348
  %10093 = addrspacecast float addrspace(4)* %10092 to float addrspace(1)*		; visa id: 13348
  store float %10088, float addrspace(1)* %10093, align 4		; visa id: 13349
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13350

10094:                                            ; preds = %10071
; BB1080 :
  %10095 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10078, i32 %10079, i32 %47, i32 %48)
  %10096 = extractvalue { i32, i32 } %10095, 0		; visa id: 13352
  %10097 = extractvalue { i32, i32 } %10095, 1		; visa id: 13352
  %10098 = insertelement <2 x i32> undef, i32 %10096, i32 0		; visa id: 13359
  %10099 = insertelement <2 x i32> %10098, i32 %10097, i32 1		; visa id: 13360
  %10100 = bitcast <2 x i32> %10099 to i64		; visa id: 13361
  %10101 = shl i64 %10100, 2		; visa id: 13365
  %10102 = add i64 %.in399, %10101		; visa id: 13366
  %10103 = shl nsw i64 %10080, 2		; visa id: 13367
  %10104 = add i64 %10102, %10103		; visa id: 13368
  %10105 = inttoptr i64 %10104 to float addrspace(4)*		; visa id: 13369
  %10106 = addrspacecast float addrspace(4)* %10105 to float addrspace(1)*		; visa id: 13369
  %10107 = load float, float addrspace(1)* %10106, align 4		; visa id: 13370
  %10108 = fmul reassoc nsz arcp contract float %10107, %4, !spirv.Decorations !618		; visa id: 13371
  %10109 = fadd reassoc nsz arcp contract float %10088, %10108, !spirv.Decorations !618		; visa id: 13372
  %10110 = shl i64 %10087, 2		; visa id: 13373
  %10111 = add i64 %.in, %10110		; visa id: 13374
  %10112 = inttoptr i64 %10111 to float addrspace(4)*		; visa id: 13375
  %10113 = addrspacecast float addrspace(4)* %10112 to float addrspace(1)*		; visa id: 13375
  store float %10109, float addrspace(1)* %10113, align 4		; visa id: 13376
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13377

._crit_edge70.1.15:                               ; preds = %.._crit_edge70.1.15_crit_edge, %10094, %10089
; BB1081 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13378
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13378
  %10114 = insertelement <2 x i32> %6556, i32 %9995, i64 1		; visa id: 13378
  store <2 x i32> %10114, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13379
  br label %._crit_edge397, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13381

._crit_edge397:                                   ; preds = %._crit_edge397.._crit_edge397_crit_edge, %._crit_edge70.1.15
; BB1082 :
  %10115 = phi i32 [ 0, %._crit_edge70.1.15 ], [ %10124, %._crit_edge397.._crit_edge397_crit_edge ]
  %10116 = zext i32 %10115 to i64		; visa id: 13382
  %10117 = shl nuw nsw i64 %10116, 2		; visa id: 13383
  %10118 = add i64 %6430, %10117		; visa id: 13384
  %10119 = inttoptr i64 %10118 to i32*		; visa id: 13385
  %10120 = load i32, i32* %10119, align 4, !noalias !642		; visa id: 13385
  %10121 = add i64 %6426, %10117		; visa id: 13386
  %10122 = inttoptr i64 %10121 to i32*		; visa id: 13387
  store i32 %10120, i32* %10122, align 4, !alias.scope !642		; visa id: 13387
  %10123 = icmp eq i32 %10115, 0		; visa id: 13388
  br i1 %10123, label %._crit_edge397.._crit_edge397_crit_edge, label %10125, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13389

._crit_edge397.._crit_edge397_crit_edge:          ; preds = %._crit_edge397
; BB1083 :
  %10124 = add nuw nsw i32 %10115, 1, !spirv.Decorations !631		; visa id: 13391
  br label %._crit_edge397, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13392

10125:                                            ; preds = %._crit_edge397
; BB1084 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13394
  %10126 = load i64, i64* %6446, align 8		; visa id: 13394
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13395
  %10127 = icmp slt i32 %6555, %const_reg_dword
  %10128 = icmp slt i32 %9995, %const_reg_dword1		; visa id: 13395
  %10129 = and i1 %10127, %10128		; visa id: 13396
  br i1 %10129, label %10130, label %.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13398

.._crit_edge70.2.15_crit_edge:                    ; preds = %10125
; BB:
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

10130:                                            ; preds = %10125
; BB1086 :
  %10131 = bitcast i64 %10126 to <2 x i32>		; visa id: 13400
  %10132 = extractelement <2 x i32> %10131, i32 0		; visa id: 13402
  %10133 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %10132, i32 1
  %10134 = bitcast <2 x i32> %10133 to i64		; visa id: 13402
  %10135 = ashr exact i64 %10134, 32		; visa id: 13403
  %10136 = bitcast i64 %10135 to <2 x i32>		; visa id: 13404
  %10137 = extractelement <2 x i32> %10136, i32 0		; visa id: 13408
  %10138 = extractelement <2 x i32> %10136, i32 1		; visa id: 13408
  %10139 = ashr i64 %10126, 32		; visa id: 13408
  %10140 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10137, i32 %10138, i32 %50, i32 %51)
  %10141 = extractvalue { i32, i32 } %10140, 0		; visa id: 13409
  %10142 = extractvalue { i32, i32 } %10140, 1		; visa id: 13409
  %10143 = insertelement <2 x i32> undef, i32 %10141, i32 0		; visa id: 13416
  %10144 = insertelement <2 x i32> %10143, i32 %10142, i32 1		; visa id: 13417
  %10145 = bitcast <2 x i32> %10144 to i64		; visa id: 13418
  %10146 = add nsw i64 %10145, %10139, !spirv.Decorations !649		; visa id: 13422
  %10147 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !618		; visa id: 13423
  br i1 %86, label %10153, label %10148, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13424

10148:                                            ; preds = %10130
; BB1087 :
  %10149 = shl i64 %10146, 2		; visa id: 13426
  %10150 = add i64 %.in, %10149		; visa id: 13427
  %10151 = inttoptr i64 %10150 to float addrspace(4)*		; visa id: 13428
  %10152 = addrspacecast float addrspace(4)* %10151 to float addrspace(1)*		; visa id: 13428
  store float %10147, float addrspace(1)* %10152, align 4		; visa id: 13429
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13430

10153:                                            ; preds = %10130
; BB1088 :
  %10154 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10137, i32 %10138, i32 %47, i32 %48)
  %10155 = extractvalue { i32, i32 } %10154, 0		; visa id: 13432
  %10156 = extractvalue { i32, i32 } %10154, 1		; visa id: 13432
  %10157 = insertelement <2 x i32> undef, i32 %10155, i32 0		; visa id: 13439
  %10158 = insertelement <2 x i32> %10157, i32 %10156, i32 1		; visa id: 13440
  %10159 = bitcast <2 x i32> %10158 to i64		; visa id: 13441
  %10160 = shl i64 %10159, 2		; visa id: 13445
  %10161 = add i64 %.in399, %10160		; visa id: 13446
  %10162 = shl nsw i64 %10139, 2		; visa id: 13447
  %10163 = add i64 %10161, %10162		; visa id: 13448
  %10164 = inttoptr i64 %10163 to float addrspace(4)*		; visa id: 13449
  %10165 = addrspacecast float addrspace(4)* %10164 to float addrspace(1)*		; visa id: 13449
  %10166 = load float, float addrspace(1)* %10165, align 4		; visa id: 13450
  %10167 = fmul reassoc nsz arcp contract float %10166, %4, !spirv.Decorations !618		; visa id: 13451
  %10168 = fadd reassoc nsz arcp contract float %10147, %10167, !spirv.Decorations !618		; visa id: 13452
  %10169 = shl i64 %10146, 2		; visa id: 13453
  %10170 = add i64 %.in, %10169		; visa id: 13454
  %10171 = inttoptr i64 %10170 to float addrspace(4)*		; visa id: 13455
  %10172 = addrspacecast float addrspace(4)* %10171 to float addrspace(1)*		; visa id: 13455
  store float %10168, float addrspace(1)* %10172, align 4		; visa id: 13456
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13457

._crit_edge70.2.15:                               ; preds = %.._crit_edge70.2.15_crit_edge, %10153, %10148
; BB1089 :
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13458
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13458
  %10173 = insertelement <2 x i32> %6617, i32 %9995, i64 1		; visa id: 13458
  store <2 x i32> %10173, <2 x i32>* %6434, align 4, !noalias !642		; visa id: 13459
  br label %._crit_edge398, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13461

._crit_edge398:                                   ; preds = %._crit_edge398.._crit_edge398_crit_edge, %._crit_edge70.2.15
; BB1090 :
  %10174 = phi i32 [ 0, %._crit_edge70.2.15 ], [ %10183, %._crit_edge398.._crit_edge398_crit_edge ]
  %10175 = zext i32 %10174 to i64		; visa id: 13462
  %10176 = shl nuw nsw i64 %10175, 2		; visa id: 13463
  %10177 = add i64 %6430, %10176		; visa id: 13464
  %10178 = inttoptr i64 %10177 to i32*		; visa id: 13465
  %10179 = load i32, i32* %10178, align 4, !noalias !642		; visa id: 13465
  %10180 = add i64 %6426, %10176		; visa id: 13466
  %10181 = inttoptr i64 %10180 to i32*		; visa id: 13467
  store i32 %10179, i32* %10181, align 4, !alias.scope !642		; visa id: 13467
  %10182 = icmp eq i32 %10174, 0		; visa id: 13468
  br i1 %10182, label %._crit_edge398.._crit_edge398_crit_edge, label %10184, !llvm.loop !645, !stats.blockFrequency.digits !646, !stats.blockFrequency.scale !615		; visa id: 13469

._crit_edge398.._crit_edge398_crit_edge:          ; preds = %._crit_edge398
; BB1091 :
  %10183 = add nuw nsw i32 %10174, 1, !spirv.Decorations !631		; visa id: 13471
  br label %._crit_edge398, !stats.blockFrequency.digits !647, !stats.blockFrequency.scale !615		; visa id: 13472

10184:                                            ; preds = %._crit_edge398
; BB1092 :
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6431)		; visa id: 13474
  %10185 = load i64, i64* %6446, align 8		; visa id: 13474
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6427)		; visa id: 13475
  %10186 = icmp slt i32 %6616, %const_reg_dword
  %10187 = icmp slt i32 %9995, %const_reg_dword1		; visa id: 13475
  %10188 = and i1 %10186, %10187		; visa id: 13476
  br i1 %10188, label %10189, label %..preheader1.15_crit_edge, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13478

..preheader1.15_crit_edge:                        ; preds = %10184
; BB:
  br label %.preheader1.15, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615

10189:                                            ; preds = %10184
; BB1094 :
  %10190 = bitcast i64 %10185 to <2 x i32>		; visa id: 13480
  %10191 = extractelement <2 x i32> %10190, i32 0		; visa id: 13482
  %10192 = insertelement <2 x i32> <i32 0, i32 undef>, i32 %10191, i32 1
  %10193 = bitcast <2 x i32> %10192 to i64		; visa id: 13482
  %10194 = ashr exact i64 %10193, 32		; visa id: 13483
  %10195 = bitcast i64 %10194 to <2 x i32>		; visa id: 13484
  %10196 = extractelement <2 x i32> %10195, i32 0		; visa id: 13488
  %10197 = extractelement <2 x i32> %10195, i32 1		; visa id: 13488
  %10198 = ashr i64 %10185, 32		; visa id: 13488
  %10199 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10196, i32 %10197, i32 %50, i32 %51)
  %10200 = extractvalue { i32, i32 } %10199, 0		; visa id: 13489
  %10201 = extractvalue { i32, i32 } %10199, 1		; visa id: 13489
  %10202 = insertelement <2 x i32> undef, i32 %10200, i32 0		; visa id: 13496
  %10203 = insertelement <2 x i32> %10202, i32 %10201, i32 1		; visa id: 13497
  %10204 = bitcast <2 x i32> %10203 to i64		; visa id: 13498
  %10205 = add nsw i64 %10204, %10198, !spirv.Decorations !649		; visa id: 13502
  %10206 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !618		; visa id: 13503
  br i1 %86, label %10212, label %10207, !stats.blockFrequency.digits !648, !stats.blockFrequency.scale !615		; visa id: 13504

10207:                                            ; preds = %10189
; BB1095 :
  %10208 = shl i64 %10205, 2		; visa id: 13506
  %10209 = add i64 %.in, %10208		; visa id: 13507
  %10210 = inttoptr i64 %10209 to float addrspace(4)*		; visa id: 13508
  %10211 = addrspacecast float addrspace(4)* %10210 to float addrspace(1)*		; visa id: 13508
  store float %10206, float addrspace(1)* %10211, align 4		; visa id: 13509
  br label %.preheader1.15, !stats.blockFrequency.digits !650, !stats.blockFrequency.scale !615		; visa id: 13510

10212:                                            ; preds = %10189
; BB1096 :
  %10213 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %10196, i32 %10197, i32 %47, i32 %48)
  %10214 = extractvalue { i32, i32 } %10213, 0		; visa id: 13512
  %10215 = extractvalue { i32, i32 } %10213, 1		; visa id: 13512
  %10216 = insertelement <2 x i32> undef, i32 %10214, i32 0		; visa id: 13519
  %10217 = insertelement <2 x i32> %10216, i32 %10215, i32 1		; visa id: 13520
  %10218 = bitcast <2 x i32> %10217 to i64		; visa id: 13521
  %10219 = shl i64 %10218, 2		; visa id: 13525
  %10220 = add i64 %.in399, %10219		; visa id: 13526
  %10221 = shl nsw i64 %10198, 2		; visa id: 13527
  %10222 = add i64 %10220, %10221		; visa id: 13528
  %10223 = inttoptr i64 %10222 to float addrspace(4)*		; visa id: 13529
  %10224 = addrspacecast float addrspace(4)* %10223 to float addrspace(1)*		; visa id: 13529
  %10225 = load float, float addrspace(1)* %10224, align 4		; visa id: 13530
  %10226 = fmul reassoc nsz arcp contract float %10225, %4, !spirv.Decorations !618		; visa id: 13531
  %10227 = fadd reassoc nsz arcp contract float %10206, %10226, !spirv.Decorations !618		; visa id: 13532
  %10228 = shl i64 %10205, 2		; visa id: 13533
  %10229 = add i64 %.in, %10228		; visa id: 13534
  %10230 = inttoptr i64 %10229 to float addrspace(4)*		; visa id: 13535
  %10231 = addrspacecast float addrspace(4)* %10230 to float addrspace(1)*		; visa id: 13535
  store float %10227, float addrspace(1)* %10231, align 4		; visa id: 13536
  br label %.preheader1.15, !stats.blockFrequency.digits !651, !stats.blockFrequency.scale !615		; visa id: 13537

.preheader1.15:                                   ; preds = %..preheader1.15_crit_edge, %10212, %10207
; BB1097 :
  %10232 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %15, i32 %16)
  %10233 = extractvalue { i32, i32 } %10232, 0		; visa id: 13538
  %10234 = extractvalue { i32, i32 } %10232, 1		; visa id: 13538
  %10235 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %18, i32 %19)
  %10236 = extractvalue { i32, i32 } %10235, 0		; visa id: 13545
  %10237 = extractvalue { i32, i32 } %10235, 1		; visa id: 13545
  %10238 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %21, i32 %22)
  %10239 = extractvalue { i32, i32 } %10238, 0		; visa id: 13552
  %10240 = extractvalue { i32, i32 } %10238, 1		; visa id: 13552
  %10241 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 0, i32 %24, i32 %25)
  %10242 = extractvalue { i32, i32 } %10241, 0		; visa id: 13559
  %10243 = extractvalue { i32, i32 } %10241, 1		; visa id: 13559
  %10244 = add i32 %110, %57		; visa id: 13566
  %10245 = icmp slt i32 %10244, %8		; visa id: 13567
  br i1 %10245, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !llvm.loop !652, !stats.blockFrequency.digits !620, !stats.blockFrequency.scale !615		; visa id: 13568

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
; BB:
  br label %._crit_edge72, !stats.blockFrequency.digits !616, !stats.blockFrequency.scale !615

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
; BB1099 :
  %10246 = insertelement <2 x i32> undef, i32 %10233, i32 0		; visa id: 13570
  %10247 = insertelement <2 x i32> %10246, i32 %10234, i32 1		; visa id: 13571
  %10248 = bitcast <2 x i32> %10247 to i64		; visa id: 13572
  %10249 = shl i64 %10248, 1		; visa id: 13574
  %10250 = add i64 %.in401, %10249		; visa id: 13575
  %10251 = insertelement <2 x i32> undef, i32 %10236, i32 0		; visa id: 13576
  %10252 = insertelement <2 x i32> %10251, i32 %10237, i32 1		; visa id: 13577
  %10253 = bitcast <2 x i32> %10252 to i64		; visa id: 13578
  %10254 = shl i64 %10253, 1		; visa id: 13580
  %10255 = add i64 %.in400, %10254		; visa id: 13581
  %10256 = insertelement <2 x i32> undef, i32 %10239, i32 0		; visa id: 13582
  %10257 = insertelement <2 x i32> %10256, i32 %10240, i32 1		; visa id: 13583
  %10258 = bitcast <2 x i32> %10257 to i64		; visa id: 13584
  %.op402 = shl i64 %10258, 2		; visa id: 13586
  %10259 = bitcast i64 %.op402 to <2 x i32>		; visa id: 13587
  %10260 = extractelement <2 x i32> %10259, i32 0		; visa id: 13588
  %10261 = extractelement <2 x i32> %10259, i32 1		; visa id: 13588
  %10262 = select i1 %86, i32 %10260, i32 0		; visa id: 13588
  %10263 = select i1 %86, i32 %10261, i32 0		; visa id: 13589
  %10264 = insertelement <2 x i32> undef, i32 %10262, i32 0		; visa id: 13590
  %10265 = insertelement <2 x i32> %10264, i32 %10263, i32 1		; visa id: 13591
  %10266 = bitcast <2 x i32> %10265 to i64		; visa id: 13592
  %10267 = add i64 %.in399, %10266		; visa id: 13594
  %10268 = insertelement <2 x i32> undef, i32 %10242, i32 0		; visa id: 13595
  %10269 = insertelement <2 x i32> %10268, i32 %10243, i32 1		; visa id: 13596
  %10270 = bitcast <2 x i32> %10269 to i64		; visa id: 13597
  %10271 = shl i64 %10270, 2		; visa id: 13599
  %10272 = add i64 %.in, %10271		; visa id: 13600
  br label %.preheader2.preheader, !stats.blockFrequency.digits !653, !stats.blockFrequency.scale !615		; visa id: 13601

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
; BB1100 :
  ret void, !stats.blockFrequency.digits !614, !stats.blockFrequency.scale !615		; visa id: 13603
}

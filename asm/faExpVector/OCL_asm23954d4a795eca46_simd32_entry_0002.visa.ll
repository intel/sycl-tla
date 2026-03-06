; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0002.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail(i8 addrspace(1)* align 1 %0, i64 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i8 addrspace(1)* align 1 %3, i64 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i32 %bufferOffset, i32 %bufferOffset6, i32 %bindlessOffset, i32 %bindlessOffset7, i32 %bindlessOffset8) #0 {
; BB0 :
  %7 = bitcast i64 %1 to <2 x i32>		; visa id: 2
  %8 = extractelement <2 x i32> %7, i32 0		; visa id: 3
  %9 = extractelement <2 x i32> %7, i32 1		; visa id: 3
  %10 = bitcast i64 %4 to <2 x i32>		; visa id: 3
  %11 = extractelement <2 x i32> %10, i32 0		; visa id: 4
  %12 = extractelement <2 x i32> %10, i32 1		; visa id: 4
  %13 = bitcast i64 %const_reg_qword to <2 x i32>		; visa id: 4
  %14 = extractelement <2 x i32> %13, i32 0		; visa id: 5
  %15 = extractelement <2 x i32> %13, i32 1		; visa id: 5
  %16 = bitcast i64 %const_reg_qword3 to <2 x i32>		; visa id: 5
  %17 = extractelement <2 x i32> %16, i32 0		; visa id: 6
  %18 = extractelement <2 x i32> %16, i32 1		; visa id: 6
  %19 = extractelement <3 x i32> %globalOffset, i32 0		; visa id: 6
  %20 = extractelement <3 x i32> %globalOffset, i32 1		; visa id: 6
  %21 = extractelement <3 x i32> %globalOffset, i32 2		; visa id: 6
  %22 = extractelement <3 x i32> %enqueuedLocalSize, i32 0		; visa id: 6
  %23 = extractelement <3 x i32> %enqueuedLocalSize, i32 1		; visa id: 6
  %24 = extractelement <3 x i32> %enqueuedLocalSize, i32 2		; visa id: 6
  %25 = extractelement <8 x i32> %r0, i32 1		; visa id: 6
  %26 = extractelement <8 x i32> %r0, i32 6		; visa id: 6
  %27 = extractelement <8 x i32> %r0, i32 7		; visa id: 6
  %28 = mul i32 %24, %27		; visa id: 6
  %29 = zext i16 %localIdZ to i32		; visa id: 7
  %30 = add i32 %28, %29
  %31 = add i32 %30, %21		; visa id: 8
  %32 = mul i32 %23, %26		; visa id: 9
  %33 = zext i16 %localIdY to i32		; visa id: 10
  %34 = add i32 %32, %33
  %35 = add i32 %34, %20		; visa id: 11
  %36 = mul i32 %22, %25		; visa id: 12
  %37 = zext i16 %localIdX to i32		; visa id: 13
  %38 = add i32 %36, %37
  %39 = add i32 %38, %19		; visa id: 14
  %40 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %39, i32 0, i32 %11, i32 %12)
  %41 = extractvalue { i32, i32 } %40, 0		; visa id: 15
  %42 = extractvalue { i32, i32 } %40, 1		; visa id: 15
  %43 = insertelement <2 x i32> undef, i32 %41, i32 0		; visa id: 22
  %44 = insertelement <2 x i32> %43, i32 %42, i32 1		; visa id: 23
  %45 = bitcast <2 x i32> %44 to i64		; visa id: 24
  %46 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %35, i32 0, i32 %17, i32 %18)
  %47 = extractvalue { i32, i32 } %46, 0		; visa id: 28
  %48 = extractvalue { i32, i32 } %46, 1		; visa id: 28
  %49 = insertelement <2 x i32> undef, i32 %47, i32 0		; visa id: 35
  %50 = insertelement <2 x i32> %49, i32 %48, i32 1		; visa id: 36
  %51 = bitcast <2 x i32> %50 to i64		; visa id: 37
  %52 = ptrtoint i8 addrspace(1)* %3 to i64		; visa id: 41
  %53 = add i64 %45, %52		; visa id: 41
  %54 = add i64 %53, %51		; visa id: 42
  %55 = zext i32 %31 to i64		; visa id: 43
  %56 = add i64 %54, %55		; visa id: 44
  %57 = inttoptr i64 %56 to i8 addrspace(1)*		; visa id: 45
  %58 = load i8, i8 addrspace(1)* %57, align 1		; visa id: 45
  %59 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %39, i32 0, i32 %8, i32 %9)
  %60 = extractvalue { i32, i32 } %59, 0		; visa id: 47
  %61 = extractvalue { i32, i32 } %59, 1		; visa id: 47
  %62 = insertelement <2 x i32> undef, i32 %60, i32 0		; visa id: 54
  %63 = insertelement <2 x i32> %62, i32 %61, i32 1		; visa id: 55
  %64 = bitcast <2 x i32> %63 to i64		; visa id: 56
  %65 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %35, i32 0, i32 %14, i32 %15)
  %66 = extractvalue { i32, i32 } %65, 0		; visa id: 60
  %67 = extractvalue { i32, i32 } %65, 1		; visa id: 60
  %68 = insertelement <2 x i32> undef, i32 %66, i32 0		; visa id: 67
  %69 = insertelement <2 x i32> %68, i32 %67, i32 1		; visa id: 68
  %70 = bitcast <2 x i32> %69 to i64		; visa id: 69
  %71 = ptrtoint i8 addrspace(1)* %0 to i64		; visa id: 73
  %72 = add i64 %64, %71		; visa id: 73
  %73 = add i64 %72, %70		; visa id: 74
  %74 = add i64 %73, %55		; visa id: 75
  %75 = inttoptr i64 %74 to i8 addrspace(1)*		; visa id: 76
  store i8 %58, i8 addrspace(1)* %75, align 1		; visa id: 76
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 78
}

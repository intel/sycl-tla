; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0005.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.9* byval(%class.__generated_.9) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i32 %const_reg_dword, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i32 %bindlessOffset) #0 {
; BB0 :
  %3 = bitcast i64 %const_reg_qword to <2 x i32>		; visa id: 2
  %4 = extractelement <2 x i32> %3, i32 0		; visa id: 3
  %5 = extractelement <2 x i32> %3, i32 1		; visa id: 3
  %6 = extractelement <3 x i32> %globalOffset, i32 0		; visa id: 3
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 0		; visa id: 3
  %8 = extractelement <8 x i32> %r0, i32 1		; visa id: 3
  %9 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7, i32 0, i32 %8, i32 0)
  %10 = extractvalue { i32, i32 } %9, 0		; visa id: 3
  %11 = extractvalue { i32, i32 } %9, 1		; visa id: 3
  %12 = insertelement <2 x i32> undef, i32 %10, i32 0		; visa id: 10
  %13 = insertelement <2 x i32> %12, i32 %11, i32 1		; visa id: 11
  %14 = bitcast <2 x i32> %13 to i64		; visa id: 12
  %15 = zext i16 %localIdX to i64		; visa id: 14
  %16 = add nuw i64 %14, %15		; visa id: 15
  %17 = zext i32 %6 to i64		; visa id: 16
  %18 = add nuw i64 %16, %17		; visa id: 17
  %19 = bitcast i64 %18 to <2 x i32>		; visa id: 18
  %20 = extractelement <2 x i32> %19, i32 0		; visa id: 22
  %21 = extractelement <2 x i32> %19, i32 1		; visa id: 22
  %22 = icmp eq i32 %21, %5
  %23 = icmp ult i32 %20, %4		; visa id: 22
  %24 = and i1 %22, %23		; visa id: 23
  %25 = icmp ult i32 %21, %5
  %26 = or i1 %24, %25		; visa id: 25
  br i1 %26, label %.lr.ph.preheader, label %.._crit_edge_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 27

.._crit_edge_crit_edge:                           ; preds = %2
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
; BB2 :
  %27 = extractelement <3 x i32> %globalSize, i32 0		; visa id: 29
  %28 = zext i32 %27 to i64		; visa id: 29
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879		; visa id: 30

.lr.ph:                                           ; preds = %.lr.ph..lr.ph_crit_edge, %.lr.ph.preheader
; BB3 :
  %29 = phi i64 [ %47, %.lr.ph..lr.ph_crit_edge ], [ %18, %.lr.ph.preheader ]
  %30 = shl i64 %29, 2		; visa id: 31
  %31 = add i64 %30, %const_reg_qword1		; visa id: 32
  %32 = inttoptr i64 %31 to i32 addrspace(4)*		; visa id: 33
  %33 = addrspacecast i32 addrspace(4)* %32 to i32 addrspace(1)*		; visa id: 33
  store i32 %const_reg_dword, i32 addrspace(1)* %33, align 4		; visa id: 34
  %34 = add nuw nsw i64 %29, %28		; visa id: 36
  %35 = bitcast i64 %34 to <2 x i32>		; visa id: 37
  %36 = extractelement <2 x i32> %35, i32 0		; visa id: 41
  %37 = extractelement <2 x i32> %35, i32 1		; visa id: 41
  %38 = icmp eq i32 %37, %5
  %39 = icmp ult i32 %36, %4		; visa id: 41
  %40 = and i1 %38, %39		; visa id: 42
  %41 = icmp ult i32 %37, %5
  %42 = or i1 %40, %41		; visa id: 44
  br i1 %42, label %.lr.ph..lr.ph_crit_edge, label %._crit_edge.loopexit, !stats.blockFrequency.digits !881, !stats.blockFrequency.scale !879		; visa id: 46

._crit_edge.loopexit:                             ; preds = %.lr.ph
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph..lr.ph_crit_edge:                          ; preds = %.lr.ph
; BB5 :
  %43 = select i1 %42, i32 %36, i32 %20		; visa id: 48
  %44 = select i1 %42, i32 %37, i32 %21		; visa id: 49
  %45 = insertelement <2 x i32> undef, i32 %43, i32 0		; visa id: 50
  %46 = insertelement <2 x i32> %45, i32 %44, i32 1		; visa id: 51
  %47 = bitcast <2 x i32> %46 to i64		; visa id: 52
  br label %.lr.ph, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879		; visa id: 56

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
; BB6 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 58
}

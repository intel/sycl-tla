; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0007.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.12* byval(%class.__generated_.12) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i8 %const_reg_byte7, i8 %const_reg_byte8, i32 %bindlessOffset) #0 {
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
  %29 = phi i64 [ %46, %.lr.ph..lr.ph_crit_edge ], [ %18, %.lr.ph.preheader ]
  %30 = add i64 %29, %const_reg_qword1		; visa id: 31
  %31 = inttoptr i64 %30 to i8 addrspace(4)*		; visa id: 32
  %32 = addrspacecast i8 addrspace(4)* %31 to i8 addrspace(1)*		; visa id: 32
  store i8 %const_reg_byte, i8 addrspace(1)* %32, align 1		; visa id: 33
  %33 = add nuw nsw i64 %29, %28		; visa id: 35
  %34 = bitcast i64 %33 to <2 x i32>		; visa id: 36
  %35 = extractelement <2 x i32> %34, i32 0		; visa id: 40
  %36 = extractelement <2 x i32> %34, i32 1		; visa id: 40
  %37 = icmp eq i32 %36, %5
  %38 = icmp ult i32 %35, %4		; visa id: 40
  %39 = and i1 %37, %38		; visa id: 41
  %40 = icmp ult i32 %36, %5
  %41 = or i1 %39, %40		; visa id: 43
  br i1 %41, label %.lr.ph..lr.ph_crit_edge, label %._crit_edge.loopexit, !stats.blockFrequency.digits !881, !stats.blockFrequency.scale !879		; visa id: 45

._crit_edge.loopexit:                             ; preds = %.lr.ph
; BB:
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph..lr.ph_crit_edge:                          ; preds = %.lr.ph
; BB5 :
  %42 = select i1 %41, i32 %35, i32 %20		; visa id: 47
  %43 = select i1 %41, i32 %36, i32 %21		; visa id: 48
  %44 = insertelement <2 x i32> undef, i32 %42, i32 0		; visa id: 49
  %45 = insertelement <2 x i32> %44, i32 %43, i32 1		; visa id: 50
  %46 = bitcast <2 x i32> %45 to i64		; visa id: 51
  br label %.lr.ph, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879		; visa id: 55

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
; BB6 :
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 57
}

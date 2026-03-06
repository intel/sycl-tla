; ------------------------------------------------
; OCL_asm23954d4a795eca46_simd32_entry_0008.visa.ll
; LLVM major version: 16
; ------------------------------------------------
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_(i8 addrspace(1)* align 1 %0, i8 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
; BB0 :
  %3 = extractelement <3 x i32> %globalOffset, i32 0		; visa id: 2
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i32 0		; visa id: 2
  %5 = extractelement <8 x i32> %r0, i32 1		; visa id: 2
  %6 = mul i32 %4, %5		; visa id: 2
  %7 = zext i16 %localIdX to i32		; visa id: 3
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3		; visa id: 4
  %10 = zext i32 %9 to i64		; visa id: 5
  %11 = ptrtoint i8 addrspace(1)* %0 to i64		; visa id: 6
  %12 = add i64 %11, %10		; visa id: 6
  %13 = inttoptr i64 %12 to i8 addrspace(1)*		; visa id: 7
  store i8 %1, i8 addrspace(1)* %13, align 1		; visa id: 7
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879		; visa id: 9
}

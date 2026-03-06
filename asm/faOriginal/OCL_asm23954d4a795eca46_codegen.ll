; ------------------------------------------------
; OCL_asm23954d4a795eca46_codegen.ll
; LLVM major version: 16
; ------------------------------------------------
; ModuleID = '<origin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [3 x i64] }
%class.__generated_ = type { i8 addrspace(1)*, i64, %"class.sycl::_V1::range", i8 addrspace(1)*, i64, %"class.sycl::_V1::range" }
%"class.sycl::_V1::range.0" = type { %"class.sycl::_V1::detail::array.1" }
%"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
%class.__generated_.2 = type <{ i8 addrspace(1)*, i16, [6 x i8] }>
%class.__generated_.9 = type <{ i8 addrspace(1)*, i32, [4 x i8] }>
%class.__generated_.12 = type <{ i8 addrspace(1)*, i8, [7 x i8] }>
%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params" = type <{ i64, float, float, i32, float, float, [4 x i8] }>
%__StructSOALayout_ = type <{ i32, i32, float, float }>
%__StructSOALayout_.7 = type <{ i32, float, float }>
%"struct.cutlass::gemm::GemmCoord" = type { %"struct.cutlass::Coord" }
%"struct.cutlass::Coord" = type { [3 x i32] }
%"class.cutlass::__generated_TensorRef" = type { i8 addrspace(1)*, %"class.sycl::_V1::range.0" }

@gVar = internal global [36 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@gVar.61 = internal global [24 x i8] zeroinitializer, align 8, !spirv.Decorations !0
@llvm.used = appending global [2 x i8*] [i8* getelementptr inbounds ([36 x i8], [36 x i8]* @gVar, i32 0, i32 0), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @gVar.61, i32 0, i32 0)], section "llvm.metadata"

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE(%"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %0, %class.__generated_* byval(%class.__generated_) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i64 %const_reg_qword12, i32 %bindlessOffset) #0 {
  %3 = bitcast i64 %const_reg_qword to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = extractelement <2 x i32> %3, i32 1
  %6 = bitcast i64 %const_reg_qword1 to <2 x i32>
  %7 = extractelement <2 x i32> %6, i32 0
  %8 = extractelement <2 x i32> %6, i32 1
  %9 = bitcast i64 %const_reg_qword2 to <2 x i32>
  %10 = extractelement <2 x i32> %9, i32 0
  %11 = extractelement <2 x i32> %9, i32 1
  %12 = extractelement <3 x i32> %globalOffset, i32 0
  %13 = extractelement <3 x i32> %globalOffset, i32 1
  %14 = extractelement <3 x i32> %globalOffset, i32 2
  %15 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %16 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %17 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %18 = extractelement <8 x i32> %r0, i32 1
  %19 = extractelement <8 x i32> %r0, i32 6
  %20 = extractelement <8 x i32> %r0, i32 7
  %21 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %17, i32 0, i32 %20, i32 0)
  %22 = extractvalue { i32, i32 } %21, 0
  %23 = extractvalue { i32, i32 } %21, 1
  %24 = insertelement <2 x i32> undef, i32 %22, i32 0
  %25 = insertelement <2 x i32> %24, i32 %23, i32 1
  %26 = bitcast <2 x i32> %25 to i64
  %27 = zext i16 %localIdZ to i64
  %28 = add nuw i64 %26, %27
  %29 = zext i32 %14 to i64
  %30 = add nuw i64 %28, %29
  %31 = bitcast i64 %30 to <2 x i32>
  %32 = extractelement <2 x i32> %31, i32 0
  %33 = extractelement <2 x i32> %31, i32 1
  %34 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %16, i32 0, i32 %19, i32 0)
  %35 = extractvalue { i32, i32 } %34, 0
  %36 = extractvalue { i32, i32 } %34, 1
  %37 = insertelement <2 x i32> undef, i32 %35, i32 0
  %38 = insertelement <2 x i32> %37, i32 %36, i32 1
  %39 = bitcast <2 x i32> %38 to i64
  %40 = zext i16 %localIdY to i64
  %41 = add nuw i64 %39, %40
  %42 = zext i32 %13 to i64
  %43 = add nuw i64 %41, %42
  %44 = bitcast i64 %43 to <2 x i32>
  %45 = extractelement <2 x i32> %44, i32 0
  %46 = extractelement <2 x i32> %44, i32 1
  %47 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %15, i32 0, i32 %18, i32 0)
  %48 = extractvalue { i32, i32 } %47, 0
  %49 = extractvalue { i32, i32 } %47, 1
  %50 = insertelement <2 x i32> undef, i32 %48, i32 0
  %51 = insertelement <2 x i32> %50, i32 %49, i32 1
  %52 = bitcast <2 x i32> %51 to i64
  %53 = zext i16 %localIdX to i64
  %54 = add nuw i64 %52, %53
  %55 = zext i32 %12 to i64
  %56 = add nuw i64 %54, %55
  %57 = bitcast i64 %56 to <2 x i32>
  %58 = extractelement <2 x i32> %57, i32 0
  %59 = extractelement <2 x i32> %57, i32 1
  %60 = icmp eq i32 %33, %5
  %61 = icmp ult i32 %32, %4
  %62 = and i1 %60, %61
  %63 = icmp ult i32 %33, %5
  %64 = or i1 %62, %63
  %65 = icmp eq i32 %46, %8
  %66 = icmp ult i32 %45, %7
  %67 = and i1 %65, %66
  %68 = icmp ult i32 %46, %8
  %69 = or i1 %67, %68
  %70 = icmp eq i32 %59, %11
  %71 = icmp ult i32 %58, %10
  %72 = and i1 %70, %71
  %73 = icmp ult i32 %59, %11
  %74 = or i1 %72, %73
  %75 = and i1 %74, %69
  %76 = and i1 %75, %64
  br i1 %76, label %.lr.ph.preheader, label %.._crit_edge101_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge101_crit_edge:                        ; preds = %2
  br label %._crit_edge101, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
  %77 = bitcast i64 %const_reg_qword4 to <2 x i32>
  %78 = extractelement <2 x i32> %77, i32 0
  %79 = extractelement <2 x i32> %77, i32 1
  %80 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %81 = extractelement <2 x i32> %80, i32 0
  %82 = extractelement <2 x i32> %80, i32 1
  %83 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %84 = extractelement <2 x i32> %83, i32 0
  %85 = extractelement <2 x i32> %83, i32 1
  %86 = bitcast i64 %const_reg_qword10 to <2 x i32>
  %87 = extractelement <2 x i32> %86, i32 0
  %88 = extractelement <2 x i32> %86, i32 1
  %89 = extractelement <3 x i32> %globalSize, i32 0
  %90 = extractelement <3 x i32> %globalSize, i32 1
  %91 = extractelement <3 x i32> %globalSize, i32 2
  %92 = zext i32 %89 to i64
  %93 = zext i32 %90 to i64
  %94 = zext i32 %91 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %.lr.ph.backedge, %.lr.ph.preheader
  %95 = phi i64 [ %56, %.lr.ph.preheader ], [ %.be, %.lr.ph.backedge ]
  %96 = phi i64 [ %43, %.lr.ph.preheader ], [ %.be151, %.lr.ph.backedge ]
  %97 = phi i64 [ %30, %.lr.ph.preheader ], [ %.be152, %.lr.ph.backedge ]
  %98 = bitcast i64 %96 to <2 x i32>
  %99 = extractelement <2 x i32> %98, i32 0
  %100 = extractelement <2 x i32> %98, i32 1
  %101 = bitcast i64 %95 to <2 x i32>
  %102 = extractelement <2 x i32> %101, i32 0
  %103 = extractelement <2 x i32> %101, i32 1
  %104 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %102, i32 %103, i32 %84, i32 %85)
  %105 = extractvalue { i32, i32 } %104, 0
  %106 = extractvalue { i32, i32 } %104, 1
  %107 = insertelement <2 x i32> undef, i32 %105, i32 0
  %108 = insertelement <2 x i32> %107, i32 %106, i32 1
  %109 = bitcast <2 x i32> %108 to i64
  %110 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %99, i32 %100, i32 %87, i32 %88)
  %111 = extractvalue { i32, i32 } %110, 0
  %112 = extractvalue { i32, i32 } %110, 1
  %113 = insertelement <2 x i32> undef, i32 %111, i32 0
  %114 = insertelement <2 x i32> %113, i32 %112, i32 1
  %115 = bitcast <2 x i32> %114 to i64
  %116 = add i64 %109, %const_reg_qword8
  %117 = add i64 %116, %115
  %118 = add i64 %117, %97
  %119 = inttoptr i64 %118 to i8 addrspace(4)*
  %120 = addrspacecast i8 addrspace(4)* %119 to i8 addrspace(1)*
  %121 = load i8, i8 addrspace(1)* %120, align 1
  %122 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %102, i32 %103, i32 %78, i32 %79)
  %123 = extractvalue { i32, i32 } %122, 0
  %124 = extractvalue { i32, i32 } %122, 1
  %125 = insertelement <2 x i32> undef, i32 %123, i32 0
  %126 = insertelement <2 x i32> %125, i32 %124, i32 1
  %127 = bitcast <2 x i32> %126 to i64
  %128 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %99, i32 %100, i32 %81, i32 %82)
  %129 = extractvalue { i32, i32 } %128, 0
  %130 = extractvalue { i32, i32 } %128, 1
  %131 = insertelement <2 x i32> undef, i32 %129, i32 0
  %132 = insertelement <2 x i32> %131, i32 %130, i32 1
  %133 = bitcast <2 x i32> %132 to i64
  %134 = add i64 %127, %const_reg_qword3
  %135 = add i64 %134, %133
  %136 = add i64 %135, %97
  %137 = inttoptr i64 %136 to i8 addrspace(4)*
  %138 = addrspacecast i8 addrspace(4)* %137 to i8 addrspace(1)*
  store i8 %121, i8 addrspace(1)* %138, align 1
  %139 = add nuw nsw i64 %97, %94
  %140 = bitcast i64 %139 to <2 x i32>
  %141 = extractelement <2 x i32> %140, i32 0
  %142 = extractelement <2 x i32> %140, i32 1
  %143 = icmp eq i32 %142, %5
  %144 = icmp ult i32 %141, %4
  %145 = and i1 %143, %144
  %146 = icmp ult i32 %142, %5
  %147 = or i1 %145, %146
  br i1 %147, label %.lr.ph.._crit_edge99_crit_edge, label %148, !stats.blockFrequency.digits !881, !stats.blockFrequency.scale !879

.lr.ph.._crit_edge99_crit_edge:                   ; preds = %.lr.ph
  br label %._crit_edge99, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879

148:                                              ; preds = %.lr.ph
  %149 = add nuw nsw i64 %96, %93
  %150 = bitcast i64 %149 to <2 x i32>
  %151 = extractelement <2 x i32> %150, i32 0
  %152 = extractelement <2 x i32> %150, i32 1
  %153 = icmp eq i32 %152, %8
  %154 = icmp ult i32 %151, %7
  %155 = and i1 %153, %154
  %156 = icmp ult i32 %152, %8
  %157 = or i1 %155, %156
  br i1 %157, label %.._crit_edge99_crit_edge, label %160, !stats.blockFrequency.digits !882, !stats.blockFrequency.scale !879

.._crit_edge99_crit_edge:                         ; preds = %148
  br label %._crit_edge99, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge99:                                    ; preds = %.._crit_edge99_crit_edge, %.lr.ph.._crit_edge99_crit_edge
  %158 = phi i64 [ %96, %.lr.ph.._crit_edge99_crit_edge ], [ %149, %.._crit_edge99_crit_edge ]
  %159 = phi i64 [ %139, %.lr.ph.._crit_edge99_crit_edge ], [ %30, %.._crit_edge99_crit_edge ]
  br label %.lr.ph.backedge, !stats.blockFrequency.digits !884, !stats.blockFrequency.scale !879

160:                                              ; preds = %148
  %161 = add nuw nsw i64 %95, %92
  %162 = bitcast i64 %161 to <2 x i32>
  %163 = extractelement <2 x i32> %162, i32 0
  %164 = extractelement <2 x i32> %162, i32 1
  %165 = icmp eq i32 %164, %11
  %166 = icmp ult i32 %163, %10
  %167 = and i1 %165, %166
  %168 = icmp ult i32 %164, %11
  %169 = or i1 %167, %168
  br i1 %169, label %..lr.ph.backedge_crit_edge, label %._crit_edge101.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge101.loopexit:                          ; preds = %160
  br label %._crit_edge101, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

..lr.ph.backedge_crit_edge:                       ; preds = %160
  %170 = select i1 %169, i32 %163, i32 %58
  %171 = select i1 %169, i32 %164, i32 %59
  %172 = insertelement <2 x i32> undef, i32 %170, i32 0
  %173 = insertelement <2 x i32> %172, i32 %171, i32 1
  %174 = bitcast <2 x i32> %173 to i64
  br label %.lr.ph.backedge, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

.lr.ph.backedge:                                  ; preds = %..lr.ph.backedge_crit_edge, %._crit_edge99
  %.be = phi i64 [ %95, %._crit_edge99 ], [ %174, %..lr.ph.backedge_crit_edge ]
  %.be151 = phi i64 [ %158, %._crit_edge99 ], [ %43, %..lr.ph.backedge_crit_edge ]
  %.be152 = phi i64 [ %159, %._crit_edge99 ], [ %30, %..lr.ph.backedge_crit_edge ]
  br label %.lr.ph, !stats.blockFrequency.digits !886, !stats.blockFrequency.scale !879

._crit_edge101:                                   ; preds = %.._crit_edge101_crit_edge, %._crit_edge101.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail(i8 addrspace(1)* align 1 %0, i64 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i8 addrspace(1)* align 1 %3, i64 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i32 %bufferOffset, i32 %bufferOffset6, i32 %bindlessOffset, i32 %bindlessOffset7, i32 %bindlessOffset8) #0 {
  %7 = bitcast i64 %1 to <2 x i32>
  %8 = extractelement <2 x i32> %7, i32 0
  %9 = extractelement <2 x i32> %7, i32 1
  %10 = bitcast i64 %4 to <2 x i32>
  %11 = extractelement <2 x i32> %10, i32 0
  %12 = extractelement <2 x i32> %10, i32 1
  %13 = bitcast i64 %const_reg_qword to <2 x i32>
  %14 = extractelement <2 x i32> %13, i32 0
  %15 = extractelement <2 x i32> %13, i32 1
  %16 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %17 = extractelement <2 x i32> %16, i32 0
  %18 = extractelement <2 x i32> %16, i32 1
  %19 = extractelement <3 x i32> %globalOffset, i32 0
  %20 = extractelement <3 x i32> %globalOffset, i32 1
  %21 = extractelement <3 x i32> %globalOffset, i32 2
  %22 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %23 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %24 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %25 = extractelement <8 x i32> %r0, i32 1
  %26 = extractelement <8 x i32> %r0, i32 6
  %27 = extractelement <8 x i32> %r0, i32 7
  %28 = mul i32 %24, %27
  %29 = zext i16 %localIdZ to i32
  %30 = add i32 %28, %29
  %31 = add i32 %30, %21
  %32 = mul i32 %23, %26
  %33 = zext i16 %localIdY to i32
  %34 = add i32 %32, %33
  %35 = add i32 %34, %20
  %36 = mul i32 %22, %25
  %37 = zext i16 %localIdX to i32
  %38 = add i32 %36, %37
  %39 = add i32 %38, %19
  %40 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %39, i32 0, i32 %11, i32 %12)
  %41 = extractvalue { i32, i32 } %40, 0
  %42 = extractvalue { i32, i32 } %40, 1
  %43 = insertelement <2 x i32> undef, i32 %41, i32 0
  %44 = insertelement <2 x i32> %43, i32 %42, i32 1
  %45 = bitcast <2 x i32> %44 to i64
  %46 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %35, i32 0, i32 %17, i32 %18)
  %47 = extractvalue { i32, i32 } %46, 0
  %48 = extractvalue { i32, i32 } %46, 1
  %49 = insertelement <2 x i32> undef, i32 %47, i32 0
  %50 = insertelement <2 x i32> %49, i32 %48, i32 1
  %51 = bitcast <2 x i32> %50 to i64
  %52 = ptrtoint i8 addrspace(1)* %3 to i64
  %53 = add i64 %45, %52
  %54 = add i64 %53, %51
  %55 = zext i32 %31 to i64
  %56 = add i64 %54, %55
  %57 = inttoptr i64 %56 to i8 addrspace(1)*
  %58 = load i8, i8 addrspace(1)* %57, align 1
  %59 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %39, i32 0, i32 %8, i32 %9)
  %60 = extractvalue { i32, i32 } %59, 0
  %61 = extractvalue { i32, i32 } %59, 1
  %62 = insertelement <2 x i32> undef, i32 %60, i32 0
  %63 = insertelement <2 x i32> %62, i32 %61, i32 1
  %64 = bitcast <2 x i32> %63 to i64
  %65 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %35, i32 0, i32 %14, i32 %15)
  %66 = extractvalue { i32, i32 } %65, 0
  %67 = extractvalue { i32, i32 } %65, 1
  %68 = insertelement <2 x i32> undef, i32 %66, i32 0
  %69 = insertelement <2 x i32> %68, i32 %67, i32 1
  %70 = bitcast <2 x i32> %69 to i64
  %71 = ptrtoint i8 addrspace(1)* %0 to i64
  %72 = add i64 %64, %71
  %73 = add i64 %72, %70
  %74 = add i64 %73, %55
  %75 = inttoptr i64 %74 to i8 addrspace(1)*
  store i8 %58, i8 addrspace(1)* %75, align 1
  ret void, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !887
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.2* byval(%class.__generated_.2) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i16 %const_reg_word, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bindlessOffset) #0 {
  %3 = bitcast i64 %const_reg_qword to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = extractelement <2 x i32> %3, i32 1
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %8 = extractelement <8 x i32> %r0, i32 1
  %9 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7, i32 0, i32 %8, i32 0)
  %10 = extractvalue { i32, i32 } %9, 0
  %11 = extractvalue { i32, i32 } %9, 1
  %12 = insertelement <2 x i32> undef, i32 %10, i32 0
  %13 = insertelement <2 x i32> %12, i32 %11, i32 1
  %14 = bitcast <2 x i32> %13 to i64
  %15 = zext i16 %localIdX to i64
  %16 = add nuw i64 %14, %15
  %17 = zext i32 %6 to i64
  %18 = add nuw i64 %16, %17
  %19 = bitcast i64 %18 to <2 x i32>
  %20 = extractelement <2 x i32> %19, i32 0
  %21 = extractelement <2 x i32> %19, i32 1
  %22 = icmp eq i32 %21, %5
  %23 = icmp ult i32 %20, %4
  %24 = and i1 %22, %23
  %25 = icmp ult i32 %21, %5
  %26 = or i1 %24, %25
  br i1 %26, label %.lr.ph.preheader, label %.._crit_edge_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge_crit_edge:                           ; preds = %2
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
  %27 = extractelement <3 x i32> %globalSize, i32 0
  %28 = zext i32 %27 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %.lr.ph..lr.ph_crit_edge, %.lr.ph.preheader
  %29 = phi i64 [ %47, %.lr.ph..lr.ph_crit_edge ], [ %18, %.lr.ph.preheader ]
  %30 = shl i64 %29, 1
  %31 = add i64 %30, %const_reg_qword1
  %32 = inttoptr i64 %31 to i16 addrspace(4)*
  %33 = addrspacecast i16 addrspace(4)* %32 to i16 addrspace(1)*
  store i16 %const_reg_word, i16 addrspace(1)* %33, align 2
  %34 = add nuw nsw i64 %29, %28
  %35 = bitcast i64 %34 to <2 x i32>
  %36 = extractelement <2 x i32> %35, i32 0
  %37 = extractelement <2 x i32> %35, i32 1
  %38 = icmp eq i32 %37, %5
  %39 = icmp ult i32 %36, %4
  %40 = and i1 %38, %39
  %41 = icmp ult i32 %37, %5
  %42 = or i1 %40, %41
  br i1 %42, label %.lr.ph..lr.ph_crit_edge, label %._crit_edge.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph..lr.ph_crit_edge:                          ; preds = %.lr.ph
  %43 = select i1 %42, i32 %36, i32 %20
  %44 = select i1 %42, i32 %37, i32 %21
  %45 = insertelement <2 x i32> undef, i32 %43, i32 0
  %46 = insertelement <2 x i32> %45, i32 %44, i32 1
  %47 = bitcast <2 x i32> %46 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_(i16 addrspace(1)* align 2 %0, i16 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %5 = extractelement <8 x i32> %r0, i32 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = ptrtoint i16 addrspace(1)* %0 to i64
  %12 = shl nuw nsw i64 %10, 1
  %13 = add i64 %12, %11
  %14 = inttoptr i64 %13 to i16 addrspace(1)*
  store i16 %1, i16 addrspace(1)* %14, align 2
  ret void, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !887
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.9* byval(%class.__generated_.9) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i32 %const_reg_dword, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i32 %bindlessOffset) #0 {
  %3 = bitcast i64 %const_reg_qword to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = extractelement <2 x i32> %3, i32 1
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %8 = extractelement <8 x i32> %r0, i32 1
  %9 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7, i32 0, i32 %8, i32 0)
  %10 = extractvalue { i32, i32 } %9, 0
  %11 = extractvalue { i32, i32 } %9, 1
  %12 = insertelement <2 x i32> undef, i32 %10, i32 0
  %13 = insertelement <2 x i32> %12, i32 %11, i32 1
  %14 = bitcast <2 x i32> %13 to i64
  %15 = zext i16 %localIdX to i64
  %16 = add nuw i64 %14, %15
  %17 = zext i32 %6 to i64
  %18 = add nuw i64 %16, %17
  %19 = bitcast i64 %18 to <2 x i32>
  %20 = extractelement <2 x i32> %19, i32 0
  %21 = extractelement <2 x i32> %19, i32 1
  %22 = icmp eq i32 %21, %5
  %23 = icmp ult i32 %20, %4
  %24 = and i1 %22, %23
  %25 = icmp ult i32 %21, %5
  %26 = or i1 %24, %25
  br i1 %26, label %.lr.ph.preheader, label %.._crit_edge_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge_crit_edge:                           ; preds = %2
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
  %27 = extractelement <3 x i32> %globalSize, i32 0
  %28 = zext i32 %27 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %.lr.ph..lr.ph_crit_edge, %.lr.ph.preheader
  %29 = phi i64 [ %47, %.lr.ph..lr.ph_crit_edge ], [ %18, %.lr.ph.preheader ]
  %30 = shl i64 %29, 2
  %31 = add i64 %30, %const_reg_qword1
  %32 = inttoptr i64 %31 to i32 addrspace(4)*
  %33 = addrspacecast i32 addrspace(4)* %32 to i32 addrspace(1)*
  store i32 %const_reg_dword, i32 addrspace(1)* %33, align 4
  %34 = add nuw nsw i64 %29, %28
  %35 = bitcast i64 %34 to <2 x i32>
  %36 = extractelement <2 x i32> %35, i32 0
  %37 = extractelement <2 x i32> %35, i32 1
  %38 = icmp eq i32 %37, %5
  %39 = icmp ult i32 %36, %4
  %40 = and i1 %38, %39
  %41 = icmp ult i32 %37, %5
  %42 = or i1 %40, %41
  br i1 %42, label %.lr.ph..lr.ph_crit_edge, label %._crit_edge.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph..lr.ph_crit_edge:                          ; preds = %.lr.ph
  %43 = select i1 %42, i32 %36, i32 %20
  %44 = select i1 %42, i32 %37, i32 %21
  %45 = insertelement <2 x i32> undef, i32 %43, i32 0
  %46 = insertelement <2 x i32> %45, i32 %44, i32 1
  %47 = bitcast <2 x i32> %46 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_(i32 addrspace(1)* align 4 %0, i32 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %5 = extractelement <8 x i32> %r0, i32 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = ptrtoint i32 addrspace(1)* %0 to i64
  %12 = shl nuw nsw i64 %10, 2
  %13 = add i64 %12, %11
  %14 = inttoptr i64 %13 to i32 addrspace(1)*
  store i32 %1, i32 addrspace(1)* %14, align 4
  ret void, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !887
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE(%"class.sycl::_V1::range.0"* byval(%"class.sycl::_V1::range.0") align 8 %0, %class.__generated_.12* byval(%class.__generated_.12) align 8 %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %globalSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i8 %const_reg_byte, i8 %const_reg_byte2, i8 %const_reg_byte3, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i8 %const_reg_byte7, i8 %const_reg_byte8, i32 %bindlessOffset) #0 {
  %3 = bitcast i64 %const_reg_qword to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = extractelement <2 x i32> %3, i32 1
  %6 = extractelement <3 x i32> %globalOffset, i32 0
  %7 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %8 = extractelement <8 x i32> %r0, i32 1
  %9 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %7, i32 0, i32 %8, i32 0)
  %10 = extractvalue { i32, i32 } %9, 0
  %11 = extractvalue { i32, i32 } %9, 1
  %12 = insertelement <2 x i32> undef, i32 %10, i32 0
  %13 = insertelement <2 x i32> %12, i32 %11, i32 1
  %14 = bitcast <2 x i32> %13 to i64
  %15 = zext i16 %localIdX to i64
  %16 = add nuw i64 %14, %15
  %17 = zext i32 %6 to i64
  %18 = add nuw i64 %16, %17
  %19 = bitcast i64 %18 to <2 x i32>
  %20 = extractelement <2 x i32> %19, i32 0
  %21 = extractelement <2 x i32> %19, i32 1
  %22 = icmp eq i32 %21, %5
  %23 = icmp ult i32 %20, %4
  %24 = and i1 %22, %23
  %25 = icmp ult i32 %21, %5
  %26 = or i1 %24, %25
  br i1 %26, label %.lr.ph.preheader, label %.._crit_edge_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge_crit_edge:                           ; preds = %2
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph.preheader:                                 ; preds = %2
  %27 = extractelement <3 x i32> %globalSize, i32 0
  %28 = zext i32 %27 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %.lr.ph..lr.ph_crit_edge, %.lr.ph.preheader
  %29 = phi i64 [ %46, %.lr.ph..lr.ph_crit_edge ], [ %18, %.lr.ph.preheader ]
  %30 = add i64 %29, %const_reg_qword1
  %31 = inttoptr i64 %30 to i8 addrspace(4)*
  %32 = addrspacecast i8 addrspace(4)* %31 to i8 addrspace(1)*
  store i8 %const_reg_byte, i8 addrspace(1)* %32, align 1
  %33 = add nuw nsw i64 %29, %28
  %34 = bitcast i64 %33 to <2 x i32>
  %35 = extractelement <2 x i32> %34, i32 0
  %36 = extractelement <2 x i32> %34, i32 1
  %37 = icmp eq i32 %36, %5
  %38 = icmp ult i32 %35, %4
  %39 = and i1 %37, %38
  %40 = icmp ult i32 %36, %5
  %41 = or i1 %39, %40
  br i1 %41, label %.lr.ph..lr.ph_crit_edge, label %._crit_edge.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph..lr.ph_crit_edge:                          ; preds = %.lr.ph
  %42 = select i1 %41, i32 %35, i32 %20
  %43 = select i1 %41, i32 %36, i32 %21
  %44 = insertelement <2 x i32> undef, i32 %42, i32 0
  %45 = insertelement <2 x i32> %44, i32 %43, i32 1
  %46 = bitcast <2 x i32> %45 to i64
  br label %.lr.ph, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.._crit_edge_crit_edge, %._crit_edge.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_(i8 addrspace(1)* align 1 %0, i8 zeroext %1, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset1) #0 {
  %3 = extractelement <3 x i32> %globalOffset, i32 0
  %4 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %5 = extractelement <8 x i32> %r0, i32 1
  %6 = mul i32 %4, %5
  %7 = zext i16 %localIdX to i32
  %8 = add i32 %6, %7
  %9 = add i32 %8, %3
  %10 = zext i32 %9 to i64
  %11 = ptrtoint i8 addrspace(1)* %0 to i64
  %12 = add i64 %11, %10
  %13 = inttoptr i64 %12 to i8 addrspace(1)*
  store i8 %1, i8 addrspace(1)* %13, align 1
  ret void, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !887
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE(i16 addrspace(1)* align 2 %0, i64 %1, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"* byval(%"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params") align 8 %2, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i64 %const_reg_qword, float %const_reg_fp32, float %const_reg_fp321, i32 %const_reg_dword, float %const_reg_fp322, float %const_reg_fp323, i8 %const_reg_byte, i8 %const_reg_byte4, i8 %const_reg_byte5, i8 %const_reg_byte6, i32 %bufferOffset, i32 %bindlessOffset, i32 %bindlessOffset7) #0 {
._crit_edge:
  %3 = bitcast i64 %1 to <2 x i32>
  %4 = extractelement <2 x i32> %3, i32 0
  %5 = extractelement <2 x i32> %3, i32 1
  %6 = call i16 @llvm.genx.GenISA.simdLaneId()
  %7 = call i32 @llvm.genx.GenISA.simdSize()
  %8 = call i32 @llvm.genx.GenISA.hw.thread.id.alloca.i32()
  %9 = mul i32 %7, 112
  %10 = mul i32 %8, %9, !perThreadOffset !888
  %11 = ptrtoint i8 addrspace(2)* %constBase to i64
  %12 = add i64 %11, 40
  %13 = extractelement <3 x i32> %localSize, i32 0
  %14 = extractelement <8 x i32> %r0, i32 1
  %15 = mul i32 %7, 24
  %16 = zext i16 %6 to i32
  %17 = mul nuw nsw i32 %16, 88
  %18 = add i32 %15, %17
  %19 = add nuw nsw i32 %10, %18
  %20 = ptrtoint i8* %privateBase to i64
  %21 = zext i32 %19 to i64
  %22 = add i64 %20, %21
  %23 = inttoptr i64 %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 88, i8* nonnull %23)
  %24 = bitcast i64 %const_reg_qword to <2 x i32>
  %25 = extractelement <2 x i32> %24, i32 0
  %26 = extractelement <2 x i32> %24, i32 1
  %27 = insertvalue %__StructSOALayout_ undef, i32 %25, 0
  %28 = insertvalue %__StructSOALayout_ %27, i32 %26, 1
  %29 = insertvalue %__StructSOALayout_ %28, float %const_reg_fp32, 2
  %30 = insertvalue %__StructSOALayout_ %29, float %const_reg_fp321, 3
  %31 = call <4 x i32> @llvm.genx.GenISA.bitcastfromstruct.v4i32.__StructSOALayout_(%__StructSOALayout_ %30)
  %32 = inttoptr i64 %22 to <4 x i32>*
  store <4 x i32> %31, <4 x i32>* %32, align 8, !user_as_priv !889
  %33 = add i64 %22, 16
  %34 = insertvalue %__StructSOALayout_.7 undef, i32 %const_reg_dword, 0
  %35 = insertvalue %__StructSOALayout_.7 %34, float %const_reg_fp322, 1
  %36 = insertvalue %__StructSOALayout_.7 %35, float %const_reg_fp323, 2
  %37 = call <3 x i32> @llvm.genx.GenISA.bitcastfromstruct.v3i32.__StructSOALayout_.7(%__StructSOALayout_.7 %36)
  %38 = inttoptr i64 %33 to <3 x i32>*
  store <3 x i32> %37, <3 x i32>* %38, align 8, !user_as_priv !889
  %39 = add i64 %22, 32
  %40 = insertelement <2 x float> undef, float %const_reg_fp321, i64 0
  %41 = insertelement <2 x float> %40, float %const_reg_fp32, i64 1
  %42 = inttoptr i64 %39 to <2 x float>*
  store <2 x float> %41, <2 x float>* %42, align 8, !user_as_priv !889
  %43 = mul i32 %14, %13
  %44 = zext i16 %localIdX to i32
  %45 = add i32 %43, %44
  %46 = add i64 %22, 40
  %47 = bitcast i64 %const_reg_qword to <2 x i32>
  %48 = inttoptr i64 %46 to <2 x i32>*
  store <2 x i32> %47, <2 x i32>* %48, align 8
  %49 = add i64 %22, 48
  %50 = bitcast i8 addrspace(2)* %constBase to <8 x i32> addrspace(2)*
  %memcpy_vsrc = addrspacecast <8 x i32> addrspace(2)* %50 to <8 x i32>*
  %memcpy_vdst = inttoptr i64 %49 to <8 x i32>*
  %51 = load <8 x i32>, <8 x i32>* %memcpy_vsrc, align 8
  store <8 x i32> %51, <8 x i32>* %memcpy_vdst, align 8
  %52 = add i64 %11, 32
  %53 = add i64 %22, 80
  %54 = inttoptr i64 %52 to i32 addrspace(2)*
  %memcpy_rem = addrspacecast i32 addrspace(2)* %54 to i32*
  %memcpy_rem66 = inttoptr i64 %53 to i32*
  %55 = load i32, i32* %memcpy_rem, align 4
  store i32 %55, i32* %memcpy_rem66, align 4
  %56 = mul nuw nsw i32 %16, 24
  %57 = add nuw nsw i32 %10, %56
  %58 = zext i32 %57 to i64
  %59 = add i64 %20, %58
  %60 = inttoptr i64 %59 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %60)
  %memcpy_rem69 = inttoptr i64 %59 to <4 x i32>*
  %61 = inttoptr i64 %12 to <6 x i32> addrspace(2)*
  %62 = addrspacecast <6 x i32> addrspace(2)* %61 to <6 x i32>*
  %63 = inttoptr i64 %12 to <4 x i32> addrspace(2)*
  %64 = addrspacecast <4 x i32> addrspace(2)* %63 to <4 x i32>*
  %65 = load <4 x i32>, <4 x i32>* %64, align 8
  %66 = ptrtoint <6 x i32>* %62 to i64
  %67 = add i64 %66, 16
  %68 = inttoptr i64 %67 to <2 x i32>*
  %69 = load <2 x i32>, <2 x i32>* %68, align 8
  store <4 x i32> %65, <4 x i32>* %memcpy_rem69, align 8
  %70 = add i64 %59, 16
  %memcpy_rem71 = inttoptr i64 %70 to <2 x i32>*
  store <2 x i32> %69, <2 x i32>* %memcpy_rem71, align 8
  %71 = add i64 %59, 8
  %72 = zext i32 %45 to i64
  %73 = bitcast i64 %72 to <2 x i32>
  %74 = inttoptr i64 %71 to <2 x i32>*
  store <2 x i32> %73, <2 x i32>* %74, align 8, !user_as_priv !889
  %75 = icmp eq i32 %45, 0
  br i1 %75, label %._crit_edge._ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit_crit_edge, label %LeafBlock35._crit_edge, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !891

._crit_edge._ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit_crit_edge: ; preds = %._crit_edge
  br label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit, !stats.blockFrequency.digits !892, !stats.blockFrequency.scale !893

LeafBlock35._crit_edge:                           ; preds = %._crit_edge
  %76 = bitcast i64 %const_reg_qword to <2 x i32>
  %77 = extractelement <2 x i32> %76, i32 0
  %78 = extractelement <2 x i32> %76, i32 1
  %79 = trunc i64 %const_reg_qword to i32
  %80 = extractelement <2 x i32> %24, i32 1
  %b2s = select i1 %75, i16 0, i16 2
  %81 = inttoptr i64 %59 to <2 x i32>*
  store <2 x i32> zeroinitializer, <2 x i32>* %81, align 8, !user_as_priv !889
  %82 = trunc i16 %b2s to i8
  %.demoted.zext = zext i8 %82 to i32
  %83 = add nsw i32 %.demoted.zext, -1, !spirv.Decorations !894
  %84 = zext i32 %83 to i64
  %85 = shl nuw nsw i64 %84, 3
  %86 = add i64 %59, %85
  %87 = add i64 %86, -8
  %88 = inttoptr i64 %87 to <2 x i64>*
  %89 = load <2 x i64>, <2 x i64>* %88, align 8
  %90 = extractelement <2 x i64> %89, i32 0
  %91 = extractelement <2 x i64> %89, i32 1
  %92 = lshr i64 %91, 2
  %93 = add nsw i32 %.demoted.zext, -2, !spirv.Decorations !894
  %94 = zext i32 %93 to i64
  %95 = shl nuw nsw i64 %94, 3
  %96 = add i64 %59, %95
  %97 = shl i64 %91, 62
  %98 = bitcast i64 %97 to <2 x i32>
  %99 = extractelement <2 x i32> %98, i32 0
  %100 = extractelement <2 x i32> %98, i32 1
  %101 = lshr i64 %90, 2
  %102 = bitcast i64 %101 to <2 x i32>
  %103 = extractelement <2 x i32> %102, i32 0
  %104 = extractelement <2 x i32> %102, i32 1
  %105 = or i32 %99, %103
  %106 = or i32 %100, %104
  %107 = insertelement <2 x i32> undef, i32 %105, i32 0
  %108 = insertelement <2 x i32> %107, i32 %106, i32 1
  %109 = bitcast <2 x i32> %108 to i64
  %110 = insertelement <2 x i64> undef, i64 %109, i64 0
  %111 = insertelement <2 x i64> %110, i64 %92, i64 1
  %112 = inttoptr i64 %96 to <2 x i64>*
  store <2 x i64> %111, <2 x i64>* %112, align 8, !user_as_priv !889
  %113 = inttoptr i64 %59 to <2 x i64>*
  %114 = load <2 x i64>, <2 x i64>* %113, align 8
  %115 = extractelement <2 x i64> %114, i32 0
  %116 = bitcast i64 %115 to <2 x i32>
  %117 = extractelement <2 x i32> %116, i32 0
  %118 = extractelement <2 x i32> %116, i32 1
  %119 = extractelement <2 x i64> %114, i32 1
  %120 = bitcast i64 %119 to <2 x i32>
  %121 = extractelement <2 x i32> %120, i32 0
  %122 = extractelement <2 x i32> %120, i32 1
  %123 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %117, i32 0, i32 -766435501, i32 0)
  %124 = extractvalue { i32, i32 } %123, 0
  %125 = extractvalue { i32, i32 } %123, 1
  %126 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 0, i32 -845247145, i32 0)
  %127 = extractvalue { i32, i32 } %126, 0
  %128 = extractvalue { i32, i32 } %126, 1
  %129 = xor i32 %117, %127
  %130 = xor i32 %118, %128
  %131 = insertelement <2 x i32> undef, i32 %129, i32 0
  %132 = insertelement <2 x i32> %131, i32 %130, i32 1
  %133 = bitcast <2 x i32> %132 to i64
  %134 = lshr i64 %133, 32
  %135 = bitcast i64 %134 to <2 x i32>
  %136 = extractelement <2 x i32> %135, i32 0
  %137 = extractelement <2 x i32> %135, i32 1
  %138 = xor i32 %121, %124
  %139 = xor i32 %122, %125
  %140 = xor i32 %138, %77
  %141 = xor i32 %139, %78
  %142 = insertelement <2 x i32> undef, i32 %140, i32 0
  %143 = insertelement <2 x i32> %142, i32 %141, i32 1
  %144 = bitcast <2 x i32> %143 to i64
  %145 = lshr i64 %144, 32
  %146 = bitcast i64 %145 to <2 x i32>
  %147 = extractelement <2 x i32> %146, i32 0
  %148 = extractelement <2 x i32> %146, i32 1
  %149 = add i32 %79, -1640531527
  %150 = add i32 %80, -1150833019
  %151 = xor i32 %136, %77
  %152 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %151, i32 %137, i32 -766435501, i32 0)
  %153 = extractvalue { i32, i32 } %152, 0
  %154 = extractvalue { i32, i32 } %152, 1
  %155 = insertelement <2 x i32> undef, i32 %153, i32 0
  %156 = insertelement <2 x i32> %155, i32 %154, i32 1
  %157 = bitcast <2 x i32> %156 to i64
  %158 = lshr i64 %157, 32
  %159 = bitcast i64 %158 to <2 x i32>
  %160 = extractelement <2 x i32> %159, i32 0
  %161 = extractelement <2 x i32> %159, i32 1
  %162 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %147, i32 %148, i32 -845247145, i32 0)
  %163 = extractvalue { i32, i32 } %162, 0
  %164 = extractvalue { i32, i32 } %162, 1
  %165 = insertelement <2 x i32> undef, i32 %163, i32 0
  %166 = insertelement <2 x i32> %165, i32 %164, i32 1
  %167 = bitcast <2 x i32> %166 to i64
  %168 = lshr i64 %167, 32
  %169 = bitcast i64 %168 to <2 x i32>
  %170 = extractelement <2 x i32> %169, i32 0
  %171 = extractelement <2 x i32> %169, i32 1
  %172 = xor i32 %127, %170
  %173 = xor i32 %128, %171
  %174 = insertelement <2 x i32> undef, i32 %172, i32 0
  %175 = insertelement <2 x i32> %174, i32 %173, i32 1
  %176 = bitcast <2 x i32> %175 to i64
  %177 = trunc i64 %176 to i32
  %178 = xor i32 %149, %177
  %179 = xor i32 %124, %160
  %180 = xor i32 %125, %161
  %181 = insertelement <2 x i32> undef, i32 %179, i32 0
  %182 = insertelement <2 x i32> %181, i32 %180, i32 1
  %183 = bitcast <2 x i32> %182 to i64
  %184 = trunc i64 %183 to i32
  %185 = xor i32 %150, %184
  %186 = add i32 %79, 1013904242
  %187 = add i32 %80, 1993301258
  %188 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %178, i32 0, i32 -766435501, i32 0)
  %189 = extractvalue { i32, i32 } %188, 0
  %190 = extractvalue { i32, i32 } %188, 1
  %191 = insertelement <2 x i32> undef, i32 %189, i32 0
  %192 = insertelement <2 x i32> %191, i32 %190, i32 1
  %193 = bitcast <2 x i32> %192 to i64
  %194 = lshr i64 %193, 32
  %195 = bitcast i64 %194 to <2 x i32>
  %196 = extractelement <2 x i32> %195, i32 0
  %197 = extractelement <2 x i32> %195, i32 1
  %198 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %185, i32 0, i32 -845247145, i32 0)
  %199 = extractvalue { i32, i32 } %198, 0
  %200 = extractvalue { i32, i32 } %198, 1
  %201 = insertelement <2 x i32> undef, i32 %199, i32 0
  %202 = insertelement <2 x i32> %201, i32 %200, i32 1
  %203 = bitcast <2 x i32> %202 to i64
  %204 = lshr i64 %203, 32
  %205 = bitcast i64 %204 to <2 x i32>
  %206 = extractelement <2 x i32> %205, i32 0
  %207 = extractelement <2 x i32> %205, i32 1
  %208 = xor i32 %163, %206
  %209 = xor i32 %164, %207
  %210 = insertelement <2 x i32> undef, i32 %208, i32 0
  %211 = insertelement <2 x i32> %210, i32 %209, i32 1
  %212 = bitcast <2 x i32> %211 to i64
  %213 = trunc i64 %212 to i32
  %214 = xor i32 %186, %213
  %215 = xor i32 %153, %196
  %216 = xor i32 %154, %197
  %217 = insertelement <2 x i32> undef, i32 %215, i32 0
  %218 = insertelement <2 x i32> %217, i32 %216, i32 1
  %219 = bitcast <2 x i32> %218 to i64
  %220 = trunc i64 %219 to i32
  %221 = xor i32 %187, %220
  %222 = add i32 %79, -626627285
  %223 = add i32 %80, 842468239
  %224 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %214, i32 0, i32 -766435501, i32 0)
  %225 = extractvalue { i32, i32 } %224, 0
  %226 = extractvalue { i32, i32 } %224, 1
  %227 = insertelement <2 x i32> undef, i32 %225, i32 0
  %228 = insertelement <2 x i32> %227, i32 %226, i32 1
  %229 = bitcast <2 x i32> %228 to i64
  %230 = lshr i64 %229, 32
  %231 = bitcast i64 %230 to <2 x i32>
  %232 = extractelement <2 x i32> %231, i32 0
  %233 = extractelement <2 x i32> %231, i32 1
  %234 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %221, i32 0, i32 -845247145, i32 0)
  %235 = extractvalue { i32, i32 } %234, 0
  %236 = extractvalue { i32, i32 } %234, 1
  %237 = insertelement <2 x i32> undef, i32 %235, i32 0
  %238 = insertelement <2 x i32> %237, i32 %236, i32 1
  %239 = bitcast <2 x i32> %238 to i64
  %240 = lshr i64 %239, 32
  %241 = bitcast i64 %240 to <2 x i32>
  %242 = extractelement <2 x i32> %241, i32 0
  %243 = extractelement <2 x i32> %241, i32 1
  %244 = xor i32 %199, %242
  %245 = xor i32 %200, %243
  %246 = insertelement <2 x i32> undef, i32 %244, i32 0
  %247 = insertelement <2 x i32> %246, i32 %245, i32 1
  %248 = bitcast <2 x i32> %247 to i64
  %249 = trunc i64 %248 to i32
  %250 = xor i32 %222, %249
  %251 = xor i32 %189, %232
  %252 = xor i32 %190, %233
  %253 = insertelement <2 x i32> undef, i32 %251, i32 0
  %254 = insertelement <2 x i32> %253, i32 %252, i32 1
  %255 = bitcast <2 x i32> %254 to i64
  %256 = trunc i64 %255 to i32
  %257 = xor i32 %223, %256
  %258 = add i32 %79, 2027808484
  %259 = add i32 %80, -308364780
  %260 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %250, i32 0, i32 -766435501, i32 0)
  %261 = extractvalue { i32, i32 } %260, 0
  %262 = extractvalue { i32, i32 } %260, 1
  %263 = insertelement <2 x i32> undef, i32 %261, i32 0
  %264 = insertelement <2 x i32> %263, i32 %262, i32 1
  %265 = bitcast <2 x i32> %264 to i64
  %266 = lshr i64 %265, 32
  %267 = bitcast i64 %266 to <2 x i32>
  %268 = extractelement <2 x i32> %267, i32 0
  %269 = extractelement <2 x i32> %267, i32 1
  %270 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %257, i32 0, i32 -845247145, i32 0)
  %271 = extractvalue { i32, i32 } %270, 0
  %272 = extractvalue { i32, i32 } %270, 1
  %273 = insertelement <2 x i32> undef, i32 %271, i32 0
  %274 = insertelement <2 x i32> %273, i32 %272, i32 1
  %275 = bitcast <2 x i32> %274 to i64
  %276 = lshr i64 %275, 32
  %277 = bitcast i64 %276 to <2 x i32>
  %278 = extractelement <2 x i32> %277, i32 0
  %279 = extractelement <2 x i32> %277, i32 1
  %280 = xor i32 %235, %278
  %281 = xor i32 %236, %279
  %282 = insertelement <2 x i32> undef, i32 %280, i32 0
  %283 = insertelement <2 x i32> %282, i32 %281, i32 1
  %284 = bitcast <2 x i32> %283 to i64
  %285 = trunc i64 %284 to i32
  %286 = xor i32 %258, %285
  %287 = xor i32 %225, %268
  %288 = xor i32 %226, %269
  %289 = insertelement <2 x i32> undef, i32 %287, i32 0
  %290 = insertelement <2 x i32> %289, i32 %288, i32 1
  %291 = bitcast <2 x i32> %290 to i64
  %292 = trunc i64 %291 to i32
  %293 = xor i32 %259, %292
  %294 = add i32 %79, 387276957
  %295 = add i32 %80, -1459197799
  %296 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %286, i32 0, i32 -766435501, i32 0)
  %297 = extractvalue { i32, i32 } %296, 0
  %298 = extractvalue { i32, i32 } %296, 1
  %299 = insertelement <2 x i32> undef, i32 %297, i32 0
  %300 = insertelement <2 x i32> %299, i32 %298, i32 1
  %301 = bitcast <2 x i32> %300 to i64
  %302 = lshr i64 %301, 32
  %303 = bitcast i64 %302 to <2 x i32>
  %304 = extractelement <2 x i32> %303, i32 0
  %305 = extractelement <2 x i32> %303, i32 1
  %306 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %293, i32 0, i32 -845247145, i32 0)
  %307 = extractvalue { i32, i32 } %306, 0
  %308 = extractvalue { i32, i32 } %306, 1
  %309 = insertelement <2 x i32> undef, i32 %307, i32 0
  %310 = insertelement <2 x i32> %309, i32 %308, i32 1
  %311 = bitcast <2 x i32> %310 to i64
  %312 = lshr i64 %311, 32
  %313 = bitcast i64 %312 to <2 x i32>
  %314 = extractelement <2 x i32> %313, i32 0
  %315 = extractelement <2 x i32> %313, i32 1
  %316 = xor i32 %271, %314
  %317 = xor i32 %272, %315
  %318 = insertelement <2 x i32> undef, i32 %316, i32 0
  %319 = insertelement <2 x i32> %318, i32 %317, i32 1
  %320 = bitcast <2 x i32> %319 to i64
  %321 = trunc i64 %320 to i32
  %322 = xor i32 %294, %321
  %323 = xor i32 %261, %304
  %324 = xor i32 %262, %305
  %325 = insertelement <2 x i32> undef, i32 %323, i32 0
  %326 = insertelement <2 x i32> %325, i32 %324, i32 1
  %327 = bitcast <2 x i32> %326 to i64
  %328 = trunc i64 %327 to i32
  %329 = xor i32 %295, %328
  %330 = add i32 %79, -1253254570
  %331 = add i32 %80, 1684936478
  %332 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %322, i32 0, i32 -766435501, i32 0)
  %333 = extractvalue { i32, i32 } %332, 0
  %334 = extractvalue { i32, i32 } %332, 1
  %335 = insertelement <2 x i32> undef, i32 %333, i32 0
  %336 = insertelement <2 x i32> %335, i32 %334, i32 1
  %337 = bitcast <2 x i32> %336 to i64
  %338 = lshr i64 %337, 32
  %339 = bitcast i64 %338 to <2 x i32>
  %340 = extractelement <2 x i32> %339, i32 0
  %341 = extractelement <2 x i32> %339, i32 1
  %342 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %329, i32 0, i32 -845247145, i32 0)
  %343 = extractvalue { i32, i32 } %342, 0
  %344 = extractvalue { i32, i32 } %342, 1
  %345 = insertelement <2 x i32> undef, i32 %343, i32 0
  %346 = insertelement <2 x i32> %345, i32 %344, i32 1
  %347 = bitcast <2 x i32> %346 to i64
  %348 = lshr i64 %347, 32
  %349 = bitcast i64 %348 to <2 x i32>
  %350 = extractelement <2 x i32> %349, i32 0
  %351 = extractelement <2 x i32> %349, i32 1
  %352 = xor i32 %307, %350
  %353 = xor i32 %308, %351
  %354 = insertelement <2 x i32> undef, i32 %352, i32 0
  %355 = insertelement <2 x i32> %354, i32 %353, i32 1
  %356 = bitcast <2 x i32> %355 to i64
  %357 = trunc i64 %356 to i32
  %358 = xor i32 %330, %357
  %359 = xor i32 %297, %340
  %360 = xor i32 %298, %341
  %361 = insertelement <2 x i32> undef, i32 %359, i32 0
  %362 = insertelement <2 x i32> %361, i32 %360, i32 1
  %363 = bitcast <2 x i32> %362 to i64
  %364 = trunc i64 %363 to i32
  %365 = xor i32 %331, %364
  %366 = add i32 %79, 1401181199
  %367 = add i32 %80, 534103459
  %368 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %358, i32 0, i32 -766435501, i32 0)
  %369 = extractvalue { i32, i32 } %368, 0
  %370 = extractvalue { i32, i32 } %368, 1
  %371 = insertelement <2 x i32> undef, i32 %369, i32 0
  %372 = insertelement <2 x i32> %371, i32 %370, i32 1
  %373 = bitcast <2 x i32> %372 to i64
  %374 = lshr i64 %373, 32
  %375 = bitcast i64 %374 to <2 x i32>
  %376 = extractelement <2 x i32> %375, i32 0
  %377 = extractelement <2 x i32> %375, i32 1
  %378 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %365, i32 0, i32 -845247145, i32 0)
  %379 = extractvalue { i32, i32 } %378, 0
  %380 = extractvalue { i32, i32 } %378, 1
  %381 = insertelement <2 x i32> undef, i32 %379, i32 0
  %382 = insertelement <2 x i32> %381, i32 %380, i32 1
  %383 = bitcast <2 x i32> %382 to i64
  %384 = lshr i64 %383, 32
  %385 = bitcast i64 %384 to <2 x i32>
  %386 = extractelement <2 x i32> %385, i32 0
  %387 = extractelement <2 x i32> %385, i32 1
  %388 = xor i32 %343, %386
  %389 = xor i32 %344, %387
  %390 = insertelement <2 x i32> undef, i32 %388, i32 0
  %391 = insertelement <2 x i32> %390, i32 %389, i32 1
  %392 = bitcast <2 x i32> %391 to i64
  %393 = trunc i64 %392 to i32
  %394 = xor i32 %366, %393
  %395 = xor i32 %333, %376
  %396 = xor i32 %334, %377
  %397 = insertelement <2 x i32> undef, i32 %395, i32 0
  %398 = insertelement <2 x i32> %397, i32 %396, i32 1
  %399 = bitcast <2 x i32> %398 to i64
  %400 = trunc i64 %399 to i32
  %401 = xor i32 %367, %400
  %402 = add i32 %79, -239350328
  %403 = add i32 %80, -616729560
  %404 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %394, i32 0, i32 -766435501, i32 0)
  %405 = extractvalue { i32, i32 } %404, 0
  %406 = extractvalue { i32, i32 } %404, 1
  %407 = insertelement <2 x i32> undef, i32 %405, i32 0
  %408 = insertelement <2 x i32> %407, i32 %406, i32 1
  %409 = bitcast <2 x i32> %408 to i64
  %410 = lshr i64 %409, 32
  %411 = bitcast i64 %410 to <2 x i32>
  %412 = extractelement <2 x i32> %411, i32 0
  %413 = extractelement <2 x i32> %411, i32 1
  %414 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %401, i32 0, i32 -845247145, i32 0)
  %415 = extractvalue { i32, i32 } %414, 0
  %416 = extractvalue { i32, i32 } %414, 1
  %417 = insertelement <2 x i32> undef, i32 %415, i32 0
  %418 = insertelement <2 x i32> %417, i32 %416, i32 1
  %419 = bitcast <2 x i32> %418 to i64
  %420 = lshr i64 %419, 32
  %421 = bitcast i64 %420 to <2 x i32>
  %422 = extractelement <2 x i32> %421, i32 0
  %423 = extractelement <2 x i32> %421, i32 1
  %424 = xor i32 %379, %422
  %425 = xor i32 %380, %423
  %426 = insertelement <2 x i32> undef, i32 %424, i32 0
  %427 = insertelement <2 x i32> %426, i32 %425, i32 1
  %428 = bitcast <2 x i32> %427 to i64
  %429 = trunc i64 %428 to i32
  %430 = xor i32 %402, %429
  %431 = xor i32 %369, %412
  %432 = xor i32 %370, %413
  %433 = insertelement <2 x i32> undef, i32 %431, i32 0
  %434 = insertelement <2 x i32> %433, i32 %432, i32 1
  %435 = bitcast <2 x i32> %434 to i64
  %436 = trunc i64 %435 to i32
  %437 = xor i32 %403, %436
  %438 = add i32 %79, -1879881855
  %439 = add i32 %80, -1767562579
  %440 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %430, i32 0, i32 -766435501, i32 0)
  %441 = extractvalue { i32, i32 } %440, 0
  %442 = extractvalue { i32, i32 } %440, 1
  %443 = insertelement <2 x i32> undef, i32 %441, i32 0
  %444 = insertelement <2 x i32> %443, i32 %442, i32 1
  %445 = bitcast <2 x i32> %444 to i64
  %446 = trunc i64 %445 to i32
  %447 = lshr i64 %445, 32
  %448 = bitcast i64 %447 to <2 x i32>
  %449 = extractelement <2 x i32> %448, i32 0
  %450 = extractelement <2 x i32> %448, i32 1
  %451 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %437, i32 0, i32 -845247145, i32 0)
  %452 = extractvalue { i32, i32 } %451, 0
  %453 = extractvalue { i32, i32 } %451, 1
  %454 = insertelement <2 x i32> undef, i32 %452, i32 0
  %455 = insertelement <2 x i32> %454, i32 %453, i32 1
  %456 = bitcast <2 x i32> %455 to i64
  %457 = trunc i64 %456 to i32
  %458 = lshr i64 %456, 32
  %459 = bitcast i64 %458 to <2 x i32>
  %460 = extractelement <2 x i32> %459, i32 0
  %461 = extractelement <2 x i32> %459, i32 1
  %462 = xor i32 %415, %460
  %463 = xor i32 %416, %461
  %464 = insertelement <2 x i32> undef, i32 %462, i32 0
  %465 = insertelement <2 x i32> %464, i32 %463, i32 1
  %466 = bitcast <2 x i32> %465 to i64
  %467 = trunc i64 %466 to i32
  %468 = xor i32 %438, %467
  %469 = xor i32 %405, %449
  %470 = xor i32 %406, %450
  %471 = insertelement <2 x i32> undef, i32 %469, i32 0
  %472 = insertelement <2 x i32> %471, i32 %470, i32 1
  %473 = bitcast <2 x i32> %472 to i64
  %474 = trunc i64 %473 to i32
  %475 = xor i32 %439, %474
  %476 = insertelement <4 x i32> <i32 4, i32 undef, i32 undef, i32 undef>, i32 %468, i64 1
  %477 = insertelement <4 x i32> %476, i32 %457, i64 2
  %478 = insertelement <4 x i32> %477, i32 %475, i64 3
  %479 = add i64 %22, 64
  %480 = inttoptr i64 %479 to <4 x i32>*
  store <4 x i32> %478, <4 x i32>* %480, align 8, !user_as_priv !889
  store i32 %446, i32* %memcpy_rem66, align 8, !user_as_priv !889
  %481 = add i64 %115, 1
  %482 = bitcast i64 %481 to <2 x i32>
  %483 = extractelement <2 x i32> %482, i32 0
  %484 = extractelement <2 x i32> %482, i32 1
  %485 = icmp eq i32 %484, 0
  %486 = icmp eq i32 %483, 0
  %487 = and i1 %485, %486
  %488 = sext i1 %487 to i64
  %489 = sub i64 0, %488
  %490 = add i64 %119, %489
  %491 = trunc i64 %490 to i32
  %492 = bitcast i64 %490 to <2 x i32>
  %493 = extractelement <2 x i32> %492, i32 1
  %494 = insertelement <2 x i64> undef, i64 %481, i64 0
  %495 = bitcast <2 x i64> %494 to <4 x i32>
  %496 = insertelement <4 x i32> %495, i32 %491, i64 2
  %497 = insertelement <4 x i32> %496, i32 %493, i64 3
  %498 = add i64 %22, 48
  %499 = inttoptr i64 %498 to <4 x i32>*
  store <4 x i32> %497, <4 x i32>* %499, align 8, !user_as_priv !889
  br label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit, !stats.blockFrequency.digits !896, !stats.blockFrequency.scale !897

_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit: ; preds = %._crit_edge._ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit_crit_edge, %LeafBlock35._crit_edge
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %60)
  %500 = icmp eq i32 0, %5
  %501 = icmp ult i32 %45, %4
  %502 = and i1 %500, %501
  %503 = icmp ult i32 0, %5
  %504 = or i1 %502, %503
  br i1 %504, label %.lr.ph48, label %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit.._crit_edge49_crit_edge, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !891

_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit.._crit_edge49_crit_edge: ; preds = %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit
  br label %._crit_edge49, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !897

.lr.ph48:                                         ; preds = %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit
  %505 = extractelement <3 x i32> %numWorkGroups, i32 0
  %506 = add i64 %22, 52
  %507 = inttoptr i64 %506 to i32*
  %508 = add i64 %22, 56
  %509 = inttoptr i64 %508 to i32*
  %510 = add i64 %22, 60
  %511 = inttoptr i64 %510 to i32*
  %512 = add i64 %22, 64
  %513 = inttoptr i64 %512 to i32*
  %514 = add i64 %22, 32
  %515 = inttoptr i64 %514 to <4 x float>*
  %516 = load <4 x float>, <4 x float>* %515, align 8
  %517 = extractelement <4 x float> %516, i32 0
  %518 = extractelement <4 x float> %516, i32 1
  %519 = extractelement <4 x float> %516, i32 2
  %520 = bitcast float %519 to i32
  %521 = extractelement <4 x float> %516, i32 3
  %522 = bitcast float %521 to i32
  %523 = add i64 %22, 48
  %524 = inttoptr i64 %523 to i32*
  %525 = add i32 %520, -1640531527
  %526 = add i32 %522, -1150833019
  %527 = add i32 %520, 1013904242
  %528 = add i32 %522, 1993301258
  %529 = add i32 %520, -626627285
  %530 = add i32 %522, 842468239
  %531 = add i32 %520, 2027808484
  %532 = add i32 %522, -308364780
  %533 = add i32 %520, 387276957
  %534 = add i32 %522, -1459197799
  %535 = add i32 %520, -1253254570
  %536 = add i32 %522, 1684936478
  %537 = add i32 %520, 1401181199
  %538 = add i32 %522, 534103459
  %539 = add i32 %522, -616729560
  %540 = add i32 %520, -1879881855
  %541 = add i32 %520, -239350328
  %542 = add i32 %522, -1767562579
  %543 = fadd reassoc nsz arcp contract float %518, %517, !spirv.Decorations !898
  %544 = fmul reassoc nsz arcp contract float %543, 5.000000e-01
  %545 = fsub reassoc nsz arcp contract float %518, %517, !spirv.Decorations !898
  %546 = fmul reassoc nsz arcp contract float %545, 0x3DF0000000000000
  %547 = load <3 x i32>, <3 x i32>* %38, align 8
  %548 = extractelement <3 x i32> %547, i32 0
  %549 = extractelement <3 x i32> %547, i32 1
  %550 = bitcast i32 %549 to float
  %551 = extractelement <3 x i32> %547, i32 2
  %552 = bitcast i32 %551 to float
  %553 = icmp sgt i32 %548, -1
  %.narrow = mul i32 %13, %505
  %554 = zext i32 %.narrow to i64
  %555 = add i64 %22, 52
  %556 = inttoptr i64 %555 to <4 x i32>*
  %557 = load <4 x i32>, <4 x i32>* %556, align 4
  %558 = extractelement <4 x i32> %557, i32 0
  %559 = extractelement <4 x i32> %557, i32 1
  %560 = extractelement <4 x i32> %557, i32 2
  %561 = extractelement <4 x i32> %557, i32 3
  %.promoted57 = load i32, i32* %524, align 8, !noalias !900, !user_as_priv !889
  %562 = add i64 %22, 68
  %563 = add i64 %22, 64
  %564 = inttoptr i64 %563 to <4 x i32>*
  %565 = ptrtoint i16 addrspace(1)* %0 to i64
  br label %._crit_edge74, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !897

._crit_edge74:                                    ; preds = %.._crit_edge74_crit_edge, %.lr.ph48
  %566 = phi i32 [ %560, %.lr.ph48 ], [ %915, %.._crit_edge74_crit_edge ]
  %567 = phi i32 [ %559, %.lr.ph48 ], [ %916, %.._crit_edge74_crit_edge ]
  %568 = phi i32 [ %558, %.lr.ph48 ], [ %917, %.._crit_edge74_crit_edge ]
  %569 = phi i32 [ %.promoted57, %.lr.ph48 ], [ %918, %.._crit_edge74_crit_edge ]
  %570 = phi i32 [ %561, %.lr.ph48 ], [ %919, %.._crit_edge74_crit_edge ]
  %571 = phi i64 [ %72, %.lr.ph48 ], [ %944, %.._crit_edge74_crit_edge ]
  %.not.not = icmp eq i32 %570, 0
  br i1 %.not.not, label %580, label %572, !stats.blockFrequency.digits !903, !stats.blockFrequency.scale !904

572:                                              ; preds = %._crit_edge74
  %573 = sub nsw i32 4, %570, !spirv.Decorations !894
  %574 = sext i32 %573 to i64
  %575 = shl nsw i64 %574, 2
  %576 = add i64 %562, %575
  %577 = inttoptr i64 %576 to i32*
  %578 = load i32, i32* %577, align 4, !noalias !905, !user_as_priv !889
  %579 = add i32 %570, -1
  store i32 %579, i32* %513, align 8, !noalias !900, !user_as_priv !889
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit, !stats.blockFrequency.digits !908, !stats.blockFrequency.scale !909

580:                                              ; preds = %._crit_edge74
  %581 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %569, i32 0, i32 -766435501, i32 0)
  %582 = extractvalue { i32, i32 } %581, 0
  %583 = extractvalue { i32, i32 } %581, 1
  %584 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %567, i32 0, i32 -845247145, i32 0)
  %585 = extractvalue { i32, i32 } %584, 0
  %586 = extractvalue { i32, i32 } %584, 1
  %587 = xor i32 %568, %586
  %588 = xor i32 %587, %520
  %589 = xor i32 %566, %583
  %590 = xor i32 %589, %522
  %591 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %588, i32 0, i32 -766435501, i32 0)
  %592 = extractvalue { i32, i32 } %591, 0
  %593 = extractvalue { i32, i32 } %591, 1
  %594 = insertelement <2 x i32> undef, i32 %592, i32 0
  %595 = insertelement <2 x i32> %594, i32 %593, i32 1
  %596 = bitcast <2 x i32> %595 to i64
  %597 = lshr i64 %596, 32
  %598 = bitcast i64 %597 to <2 x i32>
  %599 = extractelement <2 x i32> %598, i32 0
  %600 = extractelement <2 x i32> %598, i32 1
  %601 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %590, i32 0, i32 -845247145, i32 0)
  %602 = extractvalue { i32, i32 } %601, 0
  %603 = extractvalue { i32, i32 } %601, 1
  %604 = insertelement <2 x i32> undef, i32 %602, i32 0
  %605 = insertelement <2 x i32> %604, i32 %603, i32 1
  %606 = bitcast <2 x i32> %605 to i64
  %607 = lshr i64 %606, 32
  %608 = bitcast i64 %607 to <2 x i32>
  %609 = extractelement <2 x i32> %608, i32 0
  %610 = extractelement <2 x i32> %608, i32 1
  %611 = xor i32 %585, %609
  %612 = xor i32 %586, %610
  %613 = insertelement <2 x i32> undef, i32 %611, i32 0
  %614 = insertelement <2 x i32> %613, i32 %612, i32 1
  %615 = bitcast <2 x i32> %614 to i64
  %616 = trunc i64 %615 to i32
  %617 = xor i32 %525, %616
  %618 = xor i32 %582, %599
  %619 = xor i32 %583, %600
  %620 = insertelement <2 x i32> undef, i32 %618, i32 0
  %621 = insertelement <2 x i32> %620, i32 %619, i32 1
  %622 = bitcast <2 x i32> %621 to i64
  %623 = trunc i64 %622 to i32
  %624 = xor i32 %526, %623
  %625 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %617, i32 0, i32 -766435501, i32 0)
  %626 = extractvalue { i32, i32 } %625, 0
  %627 = extractvalue { i32, i32 } %625, 1
  %628 = insertelement <2 x i32> undef, i32 %626, i32 0
  %629 = insertelement <2 x i32> %628, i32 %627, i32 1
  %630 = bitcast <2 x i32> %629 to i64
  %631 = lshr i64 %630, 32
  %632 = bitcast i64 %631 to <2 x i32>
  %633 = extractelement <2 x i32> %632, i32 0
  %634 = extractelement <2 x i32> %632, i32 1
  %635 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %624, i32 0, i32 -845247145, i32 0)
  %636 = extractvalue { i32, i32 } %635, 0
  %637 = extractvalue { i32, i32 } %635, 1
  %638 = insertelement <2 x i32> undef, i32 %636, i32 0
  %639 = insertelement <2 x i32> %638, i32 %637, i32 1
  %640 = bitcast <2 x i32> %639 to i64
  %641 = lshr i64 %640, 32
  %642 = bitcast i64 %641 to <2 x i32>
  %643 = extractelement <2 x i32> %642, i32 0
  %644 = extractelement <2 x i32> %642, i32 1
  %645 = xor i32 %602, %643
  %646 = xor i32 %603, %644
  %647 = insertelement <2 x i32> undef, i32 %645, i32 0
  %648 = insertelement <2 x i32> %647, i32 %646, i32 1
  %649 = bitcast <2 x i32> %648 to i64
  %650 = trunc i64 %649 to i32
  %651 = xor i32 %527, %650
  %652 = xor i32 %592, %633
  %653 = xor i32 %593, %634
  %654 = insertelement <2 x i32> undef, i32 %652, i32 0
  %655 = insertelement <2 x i32> %654, i32 %653, i32 1
  %656 = bitcast <2 x i32> %655 to i64
  %657 = trunc i64 %656 to i32
  %658 = xor i32 %528, %657
  %659 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %651, i32 0, i32 -766435501, i32 0)
  %660 = extractvalue { i32, i32 } %659, 0
  %661 = extractvalue { i32, i32 } %659, 1
  %662 = insertelement <2 x i32> undef, i32 %660, i32 0
  %663 = insertelement <2 x i32> %662, i32 %661, i32 1
  %664 = bitcast <2 x i32> %663 to i64
  %665 = lshr i64 %664, 32
  %666 = bitcast i64 %665 to <2 x i32>
  %667 = extractelement <2 x i32> %666, i32 0
  %668 = extractelement <2 x i32> %666, i32 1
  %669 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %658, i32 0, i32 -845247145, i32 0)
  %670 = extractvalue { i32, i32 } %669, 0
  %671 = extractvalue { i32, i32 } %669, 1
  %672 = insertelement <2 x i32> undef, i32 %670, i32 0
  %673 = insertelement <2 x i32> %672, i32 %671, i32 1
  %674 = bitcast <2 x i32> %673 to i64
  %675 = lshr i64 %674, 32
  %676 = bitcast i64 %675 to <2 x i32>
  %677 = extractelement <2 x i32> %676, i32 0
  %678 = extractelement <2 x i32> %676, i32 1
  %679 = xor i32 %636, %677
  %680 = xor i32 %637, %678
  %681 = insertelement <2 x i32> undef, i32 %679, i32 0
  %682 = insertelement <2 x i32> %681, i32 %680, i32 1
  %683 = bitcast <2 x i32> %682 to i64
  %684 = trunc i64 %683 to i32
  %685 = xor i32 %529, %684
  %686 = xor i32 %626, %667
  %687 = xor i32 %627, %668
  %688 = insertelement <2 x i32> undef, i32 %686, i32 0
  %689 = insertelement <2 x i32> %688, i32 %687, i32 1
  %690 = bitcast <2 x i32> %689 to i64
  %691 = trunc i64 %690 to i32
  %692 = xor i32 %530, %691
  %693 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %685, i32 0, i32 -766435501, i32 0)
  %694 = extractvalue { i32, i32 } %693, 0
  %695 = extractvalue { i32, i32 } %693, 1
  %696 = insertelement <2 x i32> undef, i32 %694, i32 0
  %697 = insertelement <2 x i32> %696, i32 %695, i32 1
  %698 = bitcast <2 x i32> %697 to i64
  %699 = lshr i64 %698, 32
  %700 = bitcast i64 %699 to <2 x i32>
  %701 = extractelement <2 x i32> %700, i32 0
  %702 = extractelement <2 x i32> %700, i32 1
  %703 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %692, i32 0, i32 -845247145, i32 0)
  %704 = extractvalue { i32, i32 } %703, 0
  %705 = extractvalue { i32, i32 } %703, 1
  %706 = insertelement <2 x i32> undef, i32 %704, i32 0
  %707 = insertelement <2 x i32> %706, i32 %705, i32 1
  %708 = bitcast <2 x i32> %707 to i64
  %709 = lshr i64 %708, 32
  %710 = bitcast i64 %709 to <2 x i32>
  %711 = extractelement <2 x i32> %710, i32 0
  %712 = extractelement <2 x i32> %710, i32 1
  %713 = xor i32 %670, %711
  %714 = xor i32 %671, %712
  %715 = insertelement <2 x i32> undef, i32 %713, i32 0
  %716 = insertelement <2 x i32> %715, i32 %714, i32 1
  %717 = bitcast <2 x i32> %716 to i64
  %718 = trunc i64 %717 to i32
  %719 = xor i32 %531, %718
  %720 = xor i32 %660, %701
  %721 = xor i32 %661, %702
  %722 = insertelement <2 x i32> undef, i32 %720, i32 0
  %723 = insertelement <2 x i32> %722, i32 %721, i32 1
  %724 = bitcast <2 x i32> %723 to i64
  %725 = trunc i64 %724 to i32
  %726 = xor i32 %532, %725
  %727 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %719, i32 0, i32 -766435501, i32 0)
  %728 = extractvalue { i32, i32 } %727, 0
  %729 = extractvalue { i32, i32 } %727, 1
  %730 = insertelement <2 x i32> undef, i32 %728, i32 0
  %731 = insertelement <2 x i32> %730, i32 %729, i32 1
  %732 = bitcast <2 x i32> %731 to i64
  %733 = lshr i64 %732, 32
  %734 = bitcast i64 %733 to <2 x i32>
  %735 = extractelement <2 x i32> %734, i32 0
  %736 = extractelement <2 x i32> %734, i32 1
  %737 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %726, i32 0, i32 -845247145, i32 0)
  %738 = extractvalue { i32, i32 } %737, 0
  %739 = extractvalue { i32, i32 } %737, 1
  %740 = insertelement <2 x i32> undef, i32 %738, i32 0
  %741 = insertelement <2 x i32> %740, i32 %739, i32 1
  %742 = bitcast <2 x i32> %741 to i64
  %743 = lshr i64 %742, 32
  %744 = bitcast i64 %743 to <2 x i32>
  %745 = extractelement <2 x i32> %744, i32 0
  %746 = extractelement <2 x i32> %744, i32 1
  %747 = xor i32 %704, %745
  %748 = xor i32 %705, %746
  %749 = insertelement <2 x i32> undef, i32 %747, i32 0
  %750 = insertelement <2 x i32> %749, i32 %748, i32 1
  %751 = bitcast <2 x i32> %750 to i64
  %752 = trunc i64 %751 to i32
  %753 = xor i32 %533, %752
  %754 = xor i32 %694, %735
  %755 = xor i32 %695, %736
  %756 = insertelement <2 x i32> undef, i32 %754, i32 0
  %757 = insertelement <2 x i32> %756, i32 %755, i32 1
  %758 = bitcast <2 x i32> %757 to i64
  %759 = trunc i64 %758 to i32
  %760 = xor i32 %534, %759
  %761 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %753, i32 0, i32 -766435501, i32 0)
  %762 = extractvalue { i32, i32 } %761, 0
  %763 = extractvalue { i32, i32 } %761, 1
  %764 = insertelement <2 x i32> undef, i32 %762, i32 0
  %765 = insertelement <2 x i32> %764, i32 %763, i32 1
  %766 = bitcast <2 x i32> %765 to i64
  %767 = lshr i64 %766, 32
  %768 = bitcast i64 %767 to <2 x i32>
  %769 = extractelement <2 x i32> %768, i32 0
  %770 = extractelement <2 x i32> %768, i32 1
  %771 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %760, i32 0, i32 -845247145, i32 0)
  %772 = extractvalue { i32, i32 } %771, 0
  %773 = extractvalue { i32, i32 } %771, 1
  %774 = insertelement <2 x i32> undef, i32 %772, i32 0
  %775 = insertelement <2 x i32> %774, i32 %773, i32 1
  %776 = bitcast <2 x i32> %775 to i64
  %777 = lshr i64 %776, 32
  %778 = bitcast i64 %777 to <2 x i32>
  %779 = extractelement <2 x i32> %778, i32 0
  %780 = extractelement <2 x i32> %778, i32 1
  %781 = xor i32 %738, %779
  %782 = xor i32 %739, %780
  %783 = insertelement <2 x i32> undef, i32 %781, i32 0
  %784 = insertelement <2 x i32> %783, i32 %782, i32 1
  %785 = bitcast <2 x i32> %784 to i64
  %786 = trunc i64 %785 to i32
  %787 = xor i32 %535, %786
  %788 = xor i32 %728, %769
  %789 = xor i32 %729, %770
  %790 = insertelement <2 x i32> undef, i32 %788, i32 0
  %791 = insertelement <2 x i32> %790, i32 %789, i32 1
  %792 = bitcast <2 x i32> %791 to i64
  %793 = trunc i64 %792 to i32
  %794 = xor i32 %536, %793
  %795 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %787, i32 0, i32 -766435501, i32 0)
  %796 = extractvalue { i32, i32 } %795, 0
  %797 = extractvalue { i32, i32 } %795, 1
  %798 = insertelement <2 x i32> undef, i32 %796, i32 0
  %799 = insertelement <2 x i32> %798, i32 %797, i32 1
  %800 = bitcast <2 x i32> %799 to i64
  %801 = lshr i64 %800, 32
  %802 = bitcast i64 %801 to <2 x i32>
  %803 = extractelement <2 x i32> %802, i32 0
  %804 = extractelement <2 x i32> %802, i32 1
  %805 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %794, i32 0, i32 -845247145, i32 0)
  %806 = extractvalue { i32, i32 } %805, 0
  %807 = extractvalue { i32, i32 } %805, 1
  %808 = insertelement <2 x i32> undef, i32 %806, i32 0
  %809 = insertelement <2 x i32> %808, i32 %807, i32 1
  %810 = bitcast <2 x i32> %809 to i64
  %811 = lshr i64 %810, 32
  %812 = bitcast i64 %811 to <2 x i32>
  %813 = extractelement <2 x i32> %812, i32 0
  %814 = extractelement <2 x i32> %812, i32 1
  %815 = xor i32 %772, %813
  %816 = xor i32 %773, %814
  %817 = insertelement <2 x i32> undef, i32 %815, i32 0
  %818 = insertelement <2 x i32> %817, i32 %816, i32 1
  %819 = bitcast <2 x i32> %818 to i64
  %820 = trunc i64 %819 to i32
  %821 = xor i32 %537, %820
  %822 = xor i32 %762, %803
  %823 = xor i32 %763, %804
  %824 = insertelement <2 x i32> undef, i32 %822, i32 0
  %825 = insertelement <2 x i32> %824, i32 %823, i32 1
  %826 = bitcast <2 x i32> %825 to i64
  %827 = trunc i64 %826 to i32
  %828 = xor i32 %538, %827
  %829 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %821, i32 0, i32 -766435501, i32 0)
  %830 = extractvalue { i32, i32 } %829, 0
  %831 = extractvalue { i32, i32 } %829, 1
  %832 = insertelement <2 x i32> undef, i32 %830, i32 0
  %833 = insertelement <2 x i32> %832, i32 %831, i32 1
  %834 = bitcast <2 x i32> %833 to i64
  %835 = lshr i64 %834, 32
  %836 = bitcast i64 %835 to <2 x i32>
  %837 = extractelement <2 x i32> %836, i32 0
  %838 = extractelement <2 x i32> %836, i32 1
  %839 = mul i32 %828, -845247145
  %840 = xor i32 %796, %837
  %841 = xor i32 %797, %838
  %842 = insertelement <2 x i32> undef, i32 %840, i32 0
  %843 = insertelement <2 x i32> %842, i32 %841, i32 1
  %844 = bitcast <2 x i32> %843 to i64
  %845 = trunc i64 %844 to i32
  %846 = xor i32 %539, %845
  %847 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %846, i32 0, i32 -845247145, i32 0)
  %848 = extractvalue { i32, i32 } %847, 0
  %849 = extractvalue { i32, i32 } %847, 1
  %850 = insertelement <2 x i32> undef, i32 %848, i32 0
  %851 = insertelement <2 x i32> %850, i32 %849, i32 1
  %852 = bitcast <2 x i32> %851 to i64
  %853 = lshr i64 %852, 32
  %854 = bitcast i64 %853 to <2 x i32>
  %855 = extractelement <2 x i32> %854, i32 0
  %856 = extractelement <2 x i32> %854, i32 1
  %857 = xor i32 %839, %849
  %858 = xor i32 %857, %540
  %859 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %828, i32 0, i32 -845247145, i32 0)
  %860 = extractvalue { i32, i32 } %859, 0
  %861 = extractvalue { i32, i32 } %859, 1
  %862 = insertelement <2 x i32> undef, i32 %860, i32 0
  %863 = insertelement <2 x i32> %862, i32 %861, i32 1
  %864 = bitcast <2 x i32> %863 to i64
  %865 = lshr i64 %864, 32
  %866 = bitcast i64 %865 to <2 x i32>
  %867 = extractelement <2 x i32> %866, i32 0
  %868 = extractelement <2 x i32> %866, i32 1
  %869 = xor i32 %806, %867
  %870 = xor i32 %807, %868
  %871 = insertelement <2 x i32> undef, i32 %869, i32 0
  %872 = insertelement <2 x i32> %871, i32 %870, i32 1
  %873 = bitcast <2 x i32> %872 to i64
  %874 = trunc i64 %873 to i32
  %875 = xor i32 %541, %874
  %876 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %875, i32 0, i32 -766435501, i32 0)
  %877 = extractvalue { i32, i32 } %876, 0
  %878 = extractvalue { i32, i32 } %876, 1
  %879 = insertelement <2 x i32> undef, i32 %877, i32 0
  %880 = insertelement <2 x i32> %879, i32 %878, i32 1
  %881 = bitcast <2 x i32> %880 to i64
  %882 = trunc i64 %881 to i32
  %883 = lshr i64 %881, 32
  %884 = bitcast i64 %883 to <2 x i32>
  %885 = extractelement <2 x i32> %884, i32 0
  %886 = extractelement <2 x i32> %884, i32 1
  %887 = trunc i64 %852 to i32
  %888 = xor i32 %860, %855
  %889 = xor i32 %861, %856
  %890 = insertelement <2 x i32> undef, i32 %888, i32 0
  %891 = insertelement <2 x i32> %890, i32 %889, i32 1
  %892 = bitcast <2 x i32> %891 to i64
  %893 = trunc i64 %892 to i32
  %894 = xor i32 %540, %893
  %895 = xor i32 %830, %885
  %896 = xor i32 %831, %886
  %897 = insertelement <2 x i32> undef, i32 %895, i32 0
  %898 = insertelement <2 x i32> %897, i32 %896, i32 1
  %899 = bitcast <2 x i32> %898 to i64
  %900 = trunc i64 %899 to i32
  %901 = xor i32 %542, %900
  %902 = insertelement <4 x i32> <i32 3, i32 undef, i32 undef, i32 undef>, i32 %894, i64 1
  %903 = insertelement <4 x i32> %902, i32 %887, i64 2
  %904 = insertelement <4 x i32> %903, i32 %901, i64 3
  store <4 x i32> %904, <4 x i32>* %564, align 8, !noalias !900, !user_as_priv !889
  store i32 %882, i32* %memcpy_rem66, align 8, !noalias !900, !user_as_priv !889
  %905 = add i32 %569, 1
  store i32 %905, i32* %524, align 8, !noalias !900, !user_as_priv !889
  %906 = icmp eq i32 %905, 0
  br i1 %906, label %907, label %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge, !stats.blockFrequency.digits !910, !stats.blockFrequency.scale !911

._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge: ; preds = %580
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit, !stats.blockFrequency.digits !912, !stats.blockFrequency.scale !911

907:                                              ; preds = %580
  %908 = add i32 %568, 1
  store i32 %908, i32* %507, align 4, !noalias !900, !user_as_priv !889
  %909 = icmp eq i32 %908, 0
  br i1 %909, label %910, label %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88, !stats.blockFrequency.digits !913, !stats.blockFrequency.scale !914

._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88: ; preds = %907
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit, !stats.blockFrequency.digits !892, !stats.blockFrequency.scale !891

910:                                              ; preds = %907
  %911 = add i32 %567, 1
  store i32 %911, i32* %509, align 8, !noalias !900, !user_as_priv !889
  %912 = icmp eq i32 %911, 0
  br i1 %912, label %913, label %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89, !stats.blockFrequency.digits !915, !stats.blockFrequency.scale !891

._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89: ; preds = %910
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit, !stats.blockFrequency.digits !913, !stats.blockFrequency.scale !897

913:                                              ; preds = %910
  %914 = add i32 %566, 1
  store i32 %914, i32* %511, align 4, !noalias !900, !user_as_priv !889
  br label %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit, !stats.blockFrequency.digits !913, !stats.blockFrequency.scale !893

_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit: ; preds = %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge, %913, %572
  %915 = phi i32 [ %566, %572 ], [ %914, %913 ], [ %566, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ %566, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ %566, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %916 = phi i32 [ %567, %572 ], [ 0, %913 ], [ %567, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ %567, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ %911, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %917 = phi i32 [ %568, %572 ], [ 0, %913 ], [ %568, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ %908, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ 0, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %918 = phi i32 [ %569, %572 ], [ 0, %913 ], [ %905, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ 0, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ 0, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %919 = phi i32 [ %579, %572 ], [ 3, %913 ], [ 3, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ 3, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ 3, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %920 = phi i32 [ %578, %572 ], [ %858, %913 ], [ %858, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge ], [ %858, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge88 ], [ %858, %._ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit_crit_edge89 ]
  %921 = sitofp i32 %920 to float
  %922 = fmul reassoc nsz arcp contract float %546, %921, !spirv.Decorations !898
  %923 = fadd reassoc nsz arcp contract float %922, %544, !spirv.Decorations !898
  br i1 %553, label %925, label %924, !stats.blockFrequency.digits !903, !stats.blockFrequency.scale !904

924:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %bf_cvt = call i16 @llvm.genx.GenISA.ftobf.i16.f32(float %923, i32 0)
  br label %939, !stats.blockFrequency.digits !910, !stats.blockFrequency.scale !911

925:                                              ; preds = %_ZN6oneapi3mkl3rng6device8generateINS2_7uniformIfNS2_14uniform_method8standardEEENS2_13philox4x32x10ILi1EEEEENSt11conditionalIXeqsrT0_8vec_sizeLi1EENT_11result_typeEN4sycl3_V13vecISD_XsrSB_8vec_sizeEEEE4typeERSC_RSB_.exit
  %926 = fmul reassoc nsz arcp contract float %923, %550, !spirv.Decorations !898
  %927 = fcmp olt float %926, 0.000000e+00
  %928 = select i1 %927, float 0xBFDFFFFFE0000000, float 0x3FDFFFFFE0000000
  %929 = fadd float %928, %926
  %930 = call float @llvm.trunc.f32(float %929)
  %931 = fptosi float %930 to i32
  %932 = sitofp i32 %931 to float
  %933 = fmul reassoc nsz arcp contract float %552, %932, !spirv.Decorations !898
  %934 = fptosi float %933 to i32
  %935 = sitofp i32 %934 to float
  %936 = bitcast float %935 to i32
  %937 = lshr i32 %936, 16
  %938 = trunc i32 %937 to i16
  br label %939, !stats.blockFrequency.digits !908, !stats.blockFrequency.scale !909

939:                                              ; preds = %924, %925
  %940 = phi i16 [ %938, %925 ], [ %bf_cvt, %924 ]
  %941 = shl i64 %571, 1
  %942 = add i64 %941, %565
  %943 = inttoptr i64 %942 to i16 addrspace(1)*
  store i16 %940, i16 addrspace(1)* %943, align 2
  %944 = add i64 %571, %554
  %945 = bitcast i64 %944 to <2 x i32>
  %946 = extractelement <2 x i32> %945, i32 0
  %947 = extractelement <2 x i32> %945, i32 1
  %948 = icmp eq i32 %947, %5
  %949 = icmp ult i32 %946, %4
  %950 = and i1 %948, %949
  %951 = icmp ult i32 %947, %5
  %952 = or i1 %950, %951
  br i1 %952, label %.._crit_edge74_crit_edge, label %._crit_edge49.loopexit, !stats.blockFrequency.digits !903, !stats.blockFrequency.scale !904

._crit_edge49.loopexit:                           ; preds = %939
  br label %._crit_edge49, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !897

.._crit_edge74_crit_edge:                         ; preds = %939
  br label %._crit_edge74, !stats.blockFrequency.digits !916, !stats.blockFrequency.scale !904

._crit_edge49:                                    ; preds = %_ZN7cutlass9reference6device6detail17RandomUniformFuncINS_10bfloat16_tEEC2ERKNS5_6ParamsE.exit.._crit_edge49_crit_edge, %._crit_edge49.loopexit
  call void @llvm.lifetime.end.p0i8(i64 88, i8* nonnull %23)
  ret void, !stats.blockFrequency.digits !890, !stats.blockFrequency.scale !891
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = bitcast i64 %9 to <2 x i32>
  %15 = extractelement <2 x i32> %14, i32 0
  %16 = extractelement <2 x i32> %14, i32 1
  %17 = bitcast i64 %10 to <2 x i32>
  %18 = extractelement <2 x i32> %17, i32 0
  %19 = extractelement <2 x i32> %17, i32 1
  %20 = bitcast i64 %11 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = extractelement <2 x i32> %20, i32 1
  %23 = bitcast i64 %12 to <2 x i32>
  %24 = extractelement <2 x i32> %23, i32 0
  %25 = extractelement <2 x i32> %23, i32 1
  %26 = extractelement <8 x i32> %r0, i32 7
  %27 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %15, i32 %16)
  %28 = extractvalue { i32, i32 } %27, 0
  %29 = extractvalue { i32, i32 } %27, 1
  %30 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %18, i32 %19)
  %31 = extractvalue { i32, i32 } %30, 0
  %32 = extractvalue { i32, i32 } %30, 1
  %33 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %21, i32 %22)
  %34 = extractvalue { i32, i32 } %33, 0
  %35 = extractvalue { i32, i32 } %33, 1
  %36 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %24, i32 %25)
  %37 = extractvalue { i32, i32 } %36, 0
  %38 = extractvalue { i32, i32 } %36, 1
  %39 = icmp slt i32 %26, %8
  br i1 %39, label %.lr.ph, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge72_crit_edge:                         ; preds = %13
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %13
  %40 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = extractelement <2 x i32> %40, i32 1
  %43 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = extractelement <2 x i32> %43, i32 1
  %46 = bitcast i64 %const_reg_qword7 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i32 0
  %48 = extractelement <2 x i32> %46, i32 1
  %49 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i32 0
  %51 = extractelement <2 x i32> %49, i32 1
  %52 = extractelement <3 x i32> %numWorkGroups, i32 2
  %53 = extractelement <3 x i32> %localSize, i32 0
  %54 = extractelement <3 x i32> %localSize, i32 1
  %55 = extractelement <8 x i32> %r0, i32 1
  %56 = extractelement <8 x i32> %r0, i32 6
  %57 = mul i32 %55, %53
  %58 = zext i16 %localIdX to i32
  %59 = add i32 %57, %58
  %60 = shl i32 %59, 2
  %61 = mul i32 %56, %54
  %62 = zext i16 %localIdY to i32
  %63 = add i32 %61, %62
  %64 = shl i32 %63, 2
  %65 = insertelement <2 x i32> undef, i32 %28, i32 0
  %66 = insertelement <2 x i32> %65, i32 %29, i32 1
  %67 = bitcast <2 x i32> %66 to i64
  %68 = shl i64 %67, 1
  %69 = add i64 %68, %const_reg_qword
  %70 = insertelement <2 x i32> undef, i32 %31, i32 0
  %71 = insertelement <2 x i32> %70, i32 %32, i32 1
  %72 = bitcast <2 x i32> %71 to i64
  %73 = shl i64 %72, 1
  %74 = add i64 %73, %const_reg_qword4
  %75 = insertelement <2 x i32> undef, i32 %34, i32 0
  %76 = insertelement <2 x i32> %75, i32 %35, i32 1
  %77 = bitcast <2 x i32> %76 to i64
  %.op = shl i64 %77, 2
  %78 = bitcast i64 %.op to <2 x i32>
  %79 = extractelement <2 x i32> %78, i32 0
  %80 = extractelement <2 x i32> %78, i32 1
  %81 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !898
  %82 = select i1 %81, i32 %79, i32 0
  %83 = select i1 %81, i32 %80, i32 0
  %84 = insertelement <2 x i32> undef, i32 %82, i32 0
  %85 = insertelement <2 x i32> %84, i32 %83, i32 1
  %86 = bitcast <2 x i32> %85 to i64
  %87 = add i64 %86, %const_reg_qword6
  %88 = insertelement <2 x i32> undef, i32 %37, i32 0
  %89 = insertelement <2 x i32> %88, i32 %38, i32 1
  %90 = bitcast <2 x i32> %89 to i64
  %91 = shl i64 %90, 2
  %92 = add i64 %91, %const_reg_qword8
  %93 = icmp sgt i32 %const_reg_dword2, 0
  %94 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %15, i32 %16)
  %95 = extractvalue { i32, i32 } %94, 0
  %96 = extractvalue { i32, i32 } %94, 1
  %97 = insertelement <2 x i32> undef, i32 %95, i32 0
  %98 = insertelement <2 x i32> %97, i32 %96, i32 1
  %99 = bitcast <2 x i32> %98 to i64
  %100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %18, i32 %19)
  %101 = extractvalue { i32, i32 } %100, 0
  %102 = extractvalue { i32, i32 } %100, 1
  %103 = insertelement <2 x i32> undef, i32 %101, i32 0
  %104 = insertelement <2 x i32> %103, i32 %102, i32 1
  %105 = bitcast <2 x i32> %104 to i64
  %106 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %21, i32 %22)
  %107 = extractvalue { i32, i32 } %106, 0
  %108 = extractvalue { i32, i32 } %106, 1
  %109 = insertelement <2 x i32> undef, i32 %107, i32 0
  %110 = insertelement <2 x i32> %109, i32 %108, i32 1
  %111 = bitcast <2 x i32> %110 to i64
  %112 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %24, i32 %25)
  %113 = extractvalue { i32, i32 } %112, 0
  %114 = extractvalue { i32, i32 } %112, 1
  %115 = insertelement <2 x i32> undef, i32 %113, i32 0
  %116 = insertelement <2 x i32> %115, i32 %114, i32 1
  %117 = bitcast <2 x i32> %116 to i64
  %118 = icmp slt i32 %60, %const_reg_dword
  %119 = icmp slt i32 %64, %const_reg_dword1
  %120 = and i1 %118, %119
  %121 = add i32 %60, 1
  %122 = icmp slt i32 %121, %const_reg_dword
  %123 = icmp slt i32 %64, %const_reg_dword1
  %124 = and i1 %122, %123
  %125 = add i32 %60, 2
  %126 = icmp slt i32 %125, %const_reg_dword
  %127 = icmp slt i32 %64, %const_reg_dword1
  %128 = and i1 %126, %127
  %129 = add i32 %60, 3
  %130 = icmp slt i32 %129, %const_reg_dword
  %131 = icmp slt i32 %64, %const_reg_dword1
  %132 = and i1 %130, %131
  %133 = add i32 %64, 1
  %134 = icmp slt i32 %133, %const_reg_dword1
  %135 = icmp slt i32 %60, %const_reg_dword
  %136 = and i1 %135, %134
  %137 = icmp slt i32 %121, %const_reg_dword
  %138 = icmp slt i32 %133, %const_reg_dword1
  %139 = and i1 %137, %138
  %140 = icmp slt i32 %125, %const_reg_dword
  %141 = icmp slt i32 %133, %const_reg_dword1
  %142 = and i1 %140, %141
  %143 = icmp slt i32 %129, %const_reg_dword
  %144 = icmp slt i32 %133, %const_reg_dword1
  %145 = and i1 %143, %144
  %146 = add i32 %64, 2
  %147 = icmp slt i32 %146, %const_reg_dword1
  %148 = icmp slt i32 %60, %const_reg_dword
  %149 = and i1 %148, %147
  %150 = icmp slt i32 %121, %const_reg_dword
  %151 = icmp slt i32 %146, %const_reg_dword1
  %152 = and i1 %150, %151
  %153 = icmp slt i32 %125, %const_reg_dword
  %154 = icmp slt i32 %146, %const_reg_dword1
  %155 = and i1 %153, %154
  %156 = icmp slt i32 %129, %const_reg_dword
  %157 = icmp slt i32 %146, %const_reg_dword1
  %158 = and i1 %156, %157
  %159 = add i32 %64, 3
  %160 = icmp slt i32 %159, %const_reg_dword1
  %161 = icmp slt i32 %60, %const_reg_dword
  %162 = and i1 %161, %160
  %163 = icmp slt i32 %121, %const_reg_dword
  %164 = icmp slt i32 %159, %const_reg_dword1
  %165 = and i1 %163, %164
  %166 = icmp slt i32 %125, %const_reg_dword
  %167 = icmp slt i32 %159, %const_reg_dword1
  %168 = and i1 %166, %167
  %169 = icmp slt i32 %129, %const_reg_dword
  %170 = icmp slt i32 %159, %const_reg_dword1
  %171 = and i1 %169, %170
  %172 = ashr i32 %60, 31
  %173 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %41, i32 %42)
  %174 = extractvalue { i32, i32 } %173, 0
  %175 = extractvalue { i32, i32 } %173, 1
  %176 = insertelement <2 x i32> undef, i32 %174, i32 0
  %177 = insertelement <2 x i32> %176, i32 %175, i32 1
  %178 = bitcast <2 x i32> %177 to i64
  %179 = shl i64 %178, 1
  %180 = ashr i32 %64, 31
  %181 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %64, i32 %180, i32 %44, i32 %45)
  %182 = extractvalue { i32, i32 } %181, 0
  %183 = extractvalue { i32, i32 } %181, 1
  %184 = insertelement <2 x i32> undef, i32 %182, i32 0
  %185 = insertelement <2 x i32> %184, i32 %183, i32 1
  %186 = bitcast <2 x i32> %185 to i64
  %187 = shl i64 %186, 1
  %188 = ashr i32 %121, 31
  %189 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %188, i32 %41, i32 %42)
  %190 = extractvalue { i32, i32 } %189, 0
  %191 = extractvalue { i32, i32 } %189, 1
  %192 = insertelement <2 x i32> undef, i32 %190, i32 0
  %193 = insertelement <2 x i32> %192, i32 %191, i32 1
  %194 = bitcast <2 x i32> %193 to i64
  %195 = shl i64 %194, 1
  %196 = ashr i32 %125, 31
  %197 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %196, i32 %41, i32 %42)
  %198 = extractvalue { i32, i32 } %197, 0
  %199 = extractvalue { i32, i32 } %197, 1
  %200 = insertelement <2 x i32> undef, i32 %198, i32 0
  %201 = insertelement <2 x i32> %200, i32 %199, i32 1
  %202 = bitcast <2 x i32> %201 to i64
  %203 = shl i64 %202, 1
  %204 = ashr i32 %129, 31
  %205 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %204, i32 %41, i32 %42)
  %206 = extractvalue { i32, i32 } %205, 0
  %207 = extractvalue { i32, i32 } %205, 1
  %208 = insertelement <2 x i32> undef, i32 %206, i32 0
  %209 = insertelement <2 x i32> %208, i32 %207, i32 1
  %210 = bitcast <2 x i32> %209 to i64
  %211 = shl i64 %210, 1
  %212 = ashr i32 %133, 31
  %213 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %133, i32 %212, i32 %44, i32 %45)
  %214 = extractvalue { i32, i32 } %213, 0
  %215 = extractvalue { i32, i32 } %213, 1
  %216 = insertelement <2 x i32> undef, i32 %214, i32 0
  %217 = insertelement <2 x i32> %216, i32 %215, i32 1
  %218 = bitcast <2 x i32> %217 to i64
  %219 = shl i64 %218, 1
  %220 = ashr i32 %146, 31
  %221 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %146, i32 %220, i32 %44, i32 %45)
  %222 = extractvalue { i32, i32 } %221, 0
  %223 = extractvalue { i32, i32 } %221, 1
  %224 = insertelement <2 x i32> undef, i32 %222, i32 0
  %225 = insertelement <2 x i32> %224, i32 %223, i32 1
  %226 = bitcast <2 x i32> %225 to i64
  %227 = shl i64 %226, 1
  %228 = ashr i32 %159, 31
  %229 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %159, i32 %228, i32 %44, i32 %45)
  %230 = extractvalue { i32, i32 } %229, 0
  %231 = extractvalue { i32, i32 } %229, 1
  %232 = insertelement <2 x i32> undef, i32 %230, i32 0
  %233 = insertelement <2 x i32> %232, i32 %231, i32 1
  %234 = bitcast <2 x i32> %233 to i64
  %235 = shl i64 %234, 1
  %236 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %50, i32 %51)
  %237 = extractvalue { i32, i32 } %236, 0
  %238 = extractvalue { i32, i32 } %236, 1
  %239 = insertelement <2 x i32> undef, i32 %237, i32 0
  %240 = insertelement <2 x i32> %239, i32 %238, i32 1
  %241 = bitcast <2 x i32> %240 to i64
  %242 = sext i32 %64 to i64
  %243 = add nsw i64 %241, %242
  %244 = shl i64 %243, 2
  %245 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %47, i32 %48)
  %246 = extractvalue { i32, i32 } %245, 0
  %247 = extractvalue { i32, i32 } %245, 1
  %248 = insertelement <2 x i32> undef, i32 %246, i32 0
  %249 = insertelement <2 x i32> %248, i32 %247, i32 1
  %250 = bitcast <2 x i32> %249 to i64
  %251 = shl i64 %250, 2
  %252 = shl nsw i64 %242, 2
  %253 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %188, i32 %50, i32 %51)
  %254 = extractvalue { i32, i32 } %253, 0
  %255 = extractvalue { i32, i32 } %253, 1
  %256 = insertelement <2 x i32> undef, i32 %254, i32 0
  %257 = insertelement <2 x i32> %256, i32 %255, i32 1
  %258 = bitcast <2 x i32> %257 to i64
  %259 = add nsw i64 %258, %242
  %260 = shl i64 %259, 2
  %261 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %188, i32 %47, i32 %48)
  %262 = extractvalue { i32, i32 } %261, 0
  %263 = extractvalue { i32, i32 } %261, 1
  %264 = insertelement <2 x i32> undef, i32 %262, i32 0
  %265 = insertelement <2 x i32> %264, i32 %263, i32 1
  %266 = bitcast <2 x i32> %265 to i64
  %267 = shl i64 %266, 2
  %268 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %196, i32 %50, i32 %51)
  %269 = extractvalue { i32, i32 } %268, 0
  %270 = extractvalue { i32, i32 } %268, 1
  %271 = insertelement <2 x i32> undef, i32 %269, i32 0
  %272 = insertelement <2 x i32> %271, i32 %270, i32 1
  %273 = bitcast <2 x i32> %272 to i64
  %274 = add nsw i64 %273, %242
  %275 = shl i64 %274, 2
  %276 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %196, i32 %47, i32 %48)
  %277 = extractvalue { i32, i32 } %276, 0
  %278 = extractvalue { i32, i32 } %276, 1
  %279 = insertelement <2 x i32> undef, i32 %277, i32 0
  %280 = insertelement <2 x i32> %279, i32 %278, i32 1
  %281 = bitcast <2 x i32> %280 to i64
  %282 = shl i64 %281, 2
  %283 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %204, i32 %50, i32 %51)
  %284 = extractvalue { i32, i32 } %283, 0
  %285 = extractvalue { i32, i32 } %283, 1
  %286 = insertelement <2 x i32> undef, i32 %284, i32 0
  %287 = insertelement <2 x i32> %286, i32 %285, i32 1
  %288 = bitcast <2 x i32> %287 to i64
  %289 = add nsw i64 %288, %242
  %290 = shl i64 %289, 2
  %291 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %204, i32 %47, i32 %48)
  %292 = extractvalue { i32, i32 } %291, 0
  %293 = extractvalue { i32, i32 } %291, 1
  %294 = insertelement <2 x i32> undef, i32 %292, i32 0
  %295 = insertelement <2 x i32> %294, i32 %293, i32 1
  %296 = bitcast <2 x i32> %295 to i64
  %297 = shl i64 %296, 2
  %298 = sext i32 %133 to i64
  %299 = add nsw i64 %241, %298
  %300 = shl i64 %299, 2
  %301 = shl nsw i64 %298, 2
  %302 = add nsw i64 %258, %298
  %303 = shl i64 %302, 2
  %304 = add nsw i64 %273, %298
  %305 = shl i64 %304, 2
  %306 = add nsw i64 %288, %298
  %307 = shl i64 %306, 2
  %308 = sext i32 %146 to i64
  %309 = add nsw i64 %241, %308
  %310 = shl i64 %309, 2
  %311 = shl nsw i64 %308, 2
  %312 = add nsw i64 %258, %308
  %313 = shl i64 %312, 2
  %314 = add nsw i64 %273, %308
  %315 = shl i64 %314, 2
  %316 = add nsw i64 %288, %308
  %317 = shl i64 %316, 2
  %318 = sext i32 %159 to i64
  %319 = add nsw i64 %241, %318
  %320 = shl i64 %319, 2
  %321 = shl nsw i64 %318, 2
  %322 = add nsw i64 %258, %318
  %323 = shl i64 %322, 2
  %324 = add nsw i64 %273, %318
  %325 = shl i64 %324, 2
  %326 = add nsw i64 %288, %318
  %327 = shl i64 %326, 2
  %328 = shl i64 %99, 1
  %329 = shl i64 %105, 1
  %.op991 = shl i64 %111, 2
  %330 = bitcast i64 %.op991 to <2 x i32>
  %331 = extractelement <2 x i32> %330, i32 0
  %332 = extractelement <2 x i32> %330, i32 1
  %333 = select i1 %81, i32 %331, i32 0
  %334 = select i1 %81, i32 %332, i32 0
  %335 = insertelement <2 x i32> undef, i32 %333, i32 0
  %336 = insertelement <2 x i32> %335, i32 %334, i32 1
  %337 = bitcast <2 x i32> %336 to i64
  %338 = shl i64 %117, 2
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader2.preheader:                            ; preds = %.preheader1.3..preheader2.preheader_crit_edge, %.lr.ph
  %339 = phi i32 [ %26, %.lr.ph ], [ %943, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %92, %.lr.ph ], [ %948, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in988 = phi i64 [ %87, %.lr.ph ], [ %947, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in989 = phi i64 [ %74, %.lr.ph ], [ %946, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in990 = phi i64 [ %69, %.lr.ph ], [ %945, %.preheader1.3..preheader2.preheader_crit_edge ]
  br i1 %93, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
  br label %.preheader1.preheader, !stats.blockFrequency.digits !917, !stats.blockFrequency.scale !879

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
  %340 = add i64 %.in990, %179
  %341 = add i64 %.in989, %187
  %342 = add i64 %.in990, %195
  %343 = add i64 %.in990, %203
  %344 = add i64 %.in990, %211
  %345 = add i64 %.in989, %219
  %346 = add i64 %.in989, %227
  %347 = add i64 %.in989, %235
  br label %.preheader.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader.preheader:                             ; preds = %.preheader.3..preheader.preheader_crit_edge, %.preheader.preheader.preheader
  %348 = phi float [ %668, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %349 = phi float [ %649, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %350 = phi float [ %630, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %351 = phi float [ %611, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %352 = phi float [ %592, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %353 = phi float [ %573, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %354 = phi float [ %554, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %355 = phi float [ %535, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %356 = phi float [ %516, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %357 = phi float [ %497, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %358 = phi float [ %478, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %359 = phi float [ %459, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %360 = phi float [ %440, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %361 = phi float [ %421, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %362 = phi float [ %402, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %363 = phi float [ %383, %.preheader.3..preheader.preheader_crit_edge ], [ %7, %.preheader.preheader.preheader ]
  %364 = phi i32 [ %669, %.preheader.3..preheader.preheader_crit_edge ], [ 0, %.preheader.preheader.preheader ]
  br i1 %120, label %365, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

365:                                              ; preds = %.preheader.preheader
  %.sroa.64400.0.insert.ext = zext i32 %364 to i64
  %366 = shl nuw nsw i64 %.sroa.64400.0.insert.ext, 1
  %367 = add i64 %340, %366
  %368 = inttoptr i64 %367 to i16 addrspace(4)*
  %369 = addrspacecast i16 addrspace(4)* %368 to i16 addrspace(1)*
  %370 = load i16, i16 addrspace(1)* %369, align 2
  %371 = add i64 %341, %366
  %372 = inttoptr i64 %371 to i16 addrspace(4)*
  %373 = addrspacecast i16 addrspace(4)* %372 to i16 addrspace(1)*
  %374 = load i16, i16 addrspace(1)* %373, align 2
  %375 = zext i16 %370 to i32
  %376 = shl nuw i32 %375, 16, !spirv.Decorations !921
  %377 = bitcast i32 %376 to float
  %378 = zext i16 %374 to i32
  %379 = shl nuw i32 %378, 16, !spirv.Decorations !921
  %380 = bitcast i32 %379 to float
  %381 = fmul reassoc nsz arcp contract float %377, %380, !spirv.Decorations !898
  %382 = fadd reassoc nsz arcp contract float %381, %363, !spirv.Decorations !898
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %365
  %383 = phi float [ %382, %365 ], [ %363, %.preheader.preheader.._crit_edge_crit_edge ]
  br i1 %124, label %384, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

384:                                              ; preds = %._crit_edge
  %.sroa.64400.0.insert.ext402 = zext i32 %364 to i64
  %385 = shl nuw nsw i64 %.sroa.64400.0.insert.ext402, 1
  %386 = add i64 %342, %385
  %387 = inttoptr i64 %386 to i16 addrspace(4)*
  %388 = addrspacecast i16 addrspace(4)* %387 to i16 addrspace(1)*
  %389 = load i16, i16 addrspace(1)* %388, align 2
  %390 = add i64 %341, %385
  %391 = inttoptr i64 %390 to i16 addrspace(4)*
  %392 = addrspacecast i16 addrspace(4)* %391 to i16 addrspace(1)*
  %393 = load i16, i16 addrspace(1)* %392, align 2
  %394 = zext i16 %389 to i32
  %395 = shl nuw i32 %394, 16, !spirv.Decorations !921
  %396 = bitcast i32 %395 to float
  %397 = zext i16 %393 to i32
  %398 = shl nuw i32 %397, 16, !spirv.Decorations !921
  %399 = bitcast i32 %398 to float
  %400 = fmul reassoc nsz arcp contract float %396, %399, !spirv.Decorations !898
  %401 = fadd reassoc nsz arcp contract float %400, %362, !spirv.Decorations !898
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %384
  %402 = phi float [ %401, %384 ], [ %362, %._crit_edge.._crit_edge.1_crit_edge ]
  br i1 %128, label %403, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

403:                                              ; preds = %._crit_edge.1
  %.sroa.64400.0.insert.ext407 = zext i32 %364 to i64
  %404 = shl nuw nsw i64 %.sroa.64400.0.insert.ext407, 1
  %405 = add i64 %343, %404
  %406 = inttoptr i64 %405 to i16 addrspace(4)*
  %407 = addrspacecast i16 addrspace(4)* %406 to i16 addrspace(1)*
  %408 = load i16, i16 addrspace(1)* %407, align 2
  %409 = add i64 %341, %404
  %410 = inttoptr i64 %409 to i16 addrspace(4)*
  %411 = addrspacecast i16 addrspace(4)* %410 to i16 addrspace(1)*
  %412 = load i16, i16 addrspace(1)* %411, align 2
  %413 = zext i16 %408 to i32
  %414 = shl nuw i32 %413, 16, !spirv.Decorations !921
  %415 = bitcast i32 %414 to float
  %416 = zext i16 %412 to i32
  %417 = shl nuw i32 %416, 16, !spirv.Decorations !921
  %418 = bitcast i32 %417 to float
  %419 = fmul reassoc nsz arcp contract float %415, %418, !spirv.Decorations !898
  %420 = fadd reassoc nsz arcp contract float %419, %361, !spirv.Decorations !898
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %403
  %421 = phi float [ %420, %403 ], [ %361, %._crit_edge.1.._crit_edge.2_crit_edge ]
  br i1 %132, label %422, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

422:                                              ; preds = %._crit_edge.2
  %.sroa.64400.0.insert.ext412 = zext i32 %364 to i64
  %423 = shl nuw nsw i64 %.sroa.64400.0.insert.ext412, 1
  %424 = add i64 %344, %423
  %425 = inttoptr i64 %424 to i16 addrspace(4)*
  %426 = addrspacecast i16 addrspace(4)* %425 to i16 addrspace(1)*
  %427 = load i16, i16 addrspace(1)* %426, align 2
  %428 = add i64 %341, %423
  %429 = inttoptr i64 %428 to i16 addrspace(4)*
  %430 = addrspacecast i16 addrspace(4)* %429 to i16 addrspace(1)*
  %431 = load i16, i16 addrspace(1)* %430, align 2
  %432 = zext i16 %427 to i32
  %433 = shl nuw i32 %432, 16, !spirv.Decorations !921
  %434 = bitcast i32 %433 to float
  %435 = zext i16 %431 to i32
  %436 = shl nuw i32 %435, 16, !spirv.Decorations !921
  %437 = bitcast i32 %436 to float
  %438 = fmul reassoc nsz arcp contract float %434, %437, !spirv.Decorations !898
  %439 = fadd reassoc nsz arcp contract float %438, %360, !spirv.Decorations !898
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %422
  %440 = phi float [ %439, %422 ], [ %360, %._crit_edge.2..preheader_crit_edge ]
  br i1 %136, label %441, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

441:                                              ; preds = %.preheader
  %.sroa.64400.0.insert.ext417 = zext i32 %364 to i64
  %442 = shl nuw nsw i64 %.sroa.64400.0.insert.ext417, 1
  %443 = add i64 %340, %442
  %444 = inttoptr i64 %443 to i16 addrspace(4)*
  %445 = addrspacecast i16 addrspace(4)* %444 to i16 addrspace(1)*
  %446 = load i16, i16 addrspace(1)* %445, align 2
  %447 = add i64 %345, %442
  %448 = inttoptr i64 %447 to i16 addrspace(4)*
  %449 = addrspacecast i16 addrspace(4)* %448 to i16 addrspace(1)*
  %450 = load i16, i16 addrspace(1)* %449, align 2
  %451 = zext i16 %446 to i32
  %452 = shl nuw i32 %451, 16, !spirv.Decorations !921
  %453 = bitcast i32 %452 to float
  %454 = zext i16 %450 to i32
  %455 = shl nuw i32 %454, 16, !spirv.Decorations !921
  %456 = bitcast i32 %455 to float
  %457 = fmul reassoc nsz arcp contract float %453, %456, !spirv.Decorations !898
  %458 = fadd reassoc nsz arcp contract float %457, %359, !spirv.Decorations !898
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %441
  %459 = phi float [ %458, %441 ], [ %359, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %139, label %460, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

460:                                              ; preds = %._crit_edge.173
  %.sroa.64400.0.insert.ext422 = zext i32 %364 to i64
  %461 = shl nuw nsw i64 %.sroa.64400.0.insert.ext422, 1
  %462 = add i64 %342, %461
  %463 = inttoptr i64 %462 to i16 addrspace(4)*
  %464 = addrspacecast i16 addrspace(4)* %463 to i16 addrspace(1)*
  %465 = load i16, i16 addrspace(1)* %464, align 2
  %466 = add i64 %345, %461
  %467 = inttoptr i64 %466 to i16 addrspace(4)*
  %468 = addrspacecast i16 addrspace(4)* %467 to i16 addrspace(1)*
  %469 = load i16, i16 addrspace(1)* %468, align 2
  %470 = zext i16 %465 to i32
  %471 = shl nuw i32 %470, 16, !spirv.Decorations !921
  %472 = bitcast i32 %471 to float
  %473 = zext i16 %469 to i32
  %474 = shl nuw i32 %473, 16, !spirv.Decorations !921
  %475 = bitcast i32 %474 to float
  %476 = fmul reassoc nsz arcp contract float %472, %475, !spirv.Decorations !898
  %477 = fadd reassoc nsz arcp contract float %476, %358, !spirv.Decorations !898
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %460
  %478 = phi float [ %477, %460 ], [ %358, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %142, label %479, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

479:                                              ; preds = %._crit_edge.1.1
  %.sroa.64400.0.insert.ext427 = zext i32 %364 to i64
  %480 = shl nuw nsw i64 %.sroa.64400.0.insert.ext427, 1
  %481 = add i64 %343, %480
  %482 = inttoptr i64 %481 to i16 addrspace(4)*
  %483 = addrspacecast i16 addrspace(4)* %482 to i16 addrspace(1)*
  %484 = load i16, i16 addrspace(1)* %483, align 2
  %485 = add i64 %345, %480
  %486 = inttoptr i64 %485 to i16 addrspace(4)*
  %487 = addrspacecast i16 addrspace(4)* %486 to i16 addrspace(1)*
  %488 = load i16, i16 addrspace(1)* %487, align 2
  %489 = zext i16 %484 to i32
  %490 = shl nuw i32 %489, 16, !spirv.Decorations !921
  %491 = bitcast i32 %490 to float
  %492 = zext i16 %488 to i32
  %493 = shl nuw i32 %492, 16, !spirv.Decorations !921
  %494 = bitcast i32 %493 to float
  %495 = fmul reassoc nsz arcp contract float %491, %494, !spirv.Decorations !898
  %496 = fadd reassoc nsz arcp contract float %495, %357, !spirv.Decorations !898
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %479
  %497 = phi float [ %496, %479 ], [ %357, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %145, label %498, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

498:                                              ; preds = %._crit_edge.2.1
  %.sroa.64400.0.insert.ext432 = zext i32 %364 to i64
  %499 = shl nuw nsw i64 %.sroa.64400.0.insert.ext432, 1
  %500 = add i64 %344, %499
  %501 = inttoptr i64 %500 to i16 addrspace(4)*
  %502 = addrspacecast i16 addrspace(4)* %501 to i16 addrspace(1)*
  %503 = load i16, i16 addrspace(1)* %502, align 2
  %504 = add i64 %345, %499
  %505 = inttoptr i64 %504 to i16 addrspace(4)*
  %506 = addrspacecast i16 addrspace(4)* %505 to i16 addrspace(1)*
  %507 = load i16, i16 addrspace(1)* %506, align 2
  %508 = zext i16 %503 to i32
  %509 = shl nuw i32 %508, 16, !spirv.Decorations !921
  %510 = bitcast i32 %509 to float
  %511 = zext i16 %507 to i32
  %512 = shl nuw i32 %511, 16, !spirv.Decorations !921
  %513 = bitcast i32 %512 to float
  %514 = fmul reassoc nsz arcp contract float %510, %513, !spirv.Decorations !898
  %515 = fadd reassoc nsz arcp contract float %514, %356, !spirv.Decorations !898
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %498
  %516 = phi float [ %515, %498 ], [ %356, %._crit_edge.2.1..preheader.1_crit_edge ]
  br i1 %149, label %517, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

517:                                              ; preds = %.preheader.1
  %.sroa.64400.0.insert.ext437 = zext i32 %364 to i64
  %518 = shl nuw nsw i64 %.sroa.64400.0.insert.ext437, 1
  %519 = add i64 %340, %518
  %520 = inttoptr i64 %519 to i16 addrspace(4)*
  %521 = addrspacecast i16 addrspace(4)* %520 to i16 addrspace(1)*
  %522 = load i16, i16 addrspace(1)* %521, align 2
  %523 = add i64 %346, %518
  %524 = inttoptr i64 %523 to i16 addrspace(4)*
  %525 = addrspacecast i16 addrspace(4)* %524 to i16 addrspace(1)*
  %526 = load i16, i16 addrspace(1)* %525, align 2
  %527 = zext i16 %522 to i32
  %528 = shl nuw i32 %527, 16, !spirv.Decorations !921
  %529 = bitcast i32 %528 to float
  %530 = zext i16 %526 to i32
  %531 = shl nuw i32 %530, 16, !spirv.Decorations !921
  %532 = bitcast i32 %531 to float
  %533 = fmul reassoc nsz arcp contract float %529, %532, !spirv.Decorations !898
  %534 = fadd reassoc nsz arcp contract float %533, %355, !spirv.Decorations !898
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %517
  %535 = phi float [ %534, %517 ], [ %355, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %152, label %536, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

536:                                              ; preds = %._crit_edge.274
  %.sroa.64400.0.insert.ext442 = zext i32 %364 to i64
  %537 = shl nuw nsw i64 %.sroa.64400.0.insert.ext442, 1
  %538 = add i64 %342, %537
  %539 = inttoptr i64 %538 to i16 addrspace(4)*
  %540 = addrspacecast i16 addrspace(4)* %539 to i16 addrspace(1)*
  %541 = load i16, i16 addrspace(1)* %540, align 2
  %542 = add i64 %346, %537
  %543 = inttoptr i64 %542 to i16 addrspace(4)*
  %544 = addrspacecast i16 addrspace(4)* %543 to i16 addrspace(1)*
  %545 = load i16, i16 addrspace(1)* %544, align 2
  %546 = zext i16 %541 to i32
  %547 = shl nuw i32 %546, 16, !spirv.Decorations !921
  %548 = bitcast i32 %547 to float
  %549 = zext i16 %545 to i32
  %550 = shl nuw i32 %549, 16, !spirv.Decorations !921
  %551 = bitcast i32 %550 to float
  %552 = fmul reassoc nsz arcp contract float %548, %551, !spirv.Decorations !898
  %553 = fadd reassoc nsz arcp contract float %552, %354, !spirv.Decorations !898
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %536
  %554 = phi float [ %553, %536 ], [ %354, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %155, label %555, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

555:                                              ; preds = %._crit_edge.1.2
  %.sroa.64400.0.insert.ext447 = zext i32 %364 to i64
  %556 = shl nuw nsw i64 %.sroa.64400.0.insert.ext447, 1
  %557 = add i64 %343, %556
  %558 = inttoptr i64 %557 to i16 addrspace(4)*
  %559 = addrspacecast i16 addrspace(4)* %558 to i16 addrspace(1)*
  %560 = load i16, i16 addrspace(1)* %559, align 2
  %561 = add i64 %346, %556
  %562 = inttoptr i64 %561 to i16 addrspace(4)*
  %563 = addrspacecast i16 addrspace(4)* %562 to i16 addrspace(1)*
  %564 = load i16, i16 addrspace(1)* %563, align 2
  %565 = zext i16 %560 to i32
  %566 = shl nuw i32 %565, 16, !spirv.Decorations !921
  %567 = bitcast i32 %566 to float
  %568 = zext i16 %564 to i32
  %569 = shl nuw i32 %568, 16, !spirv.Decorations !921
  %570 = bitcast i32 %569 to float
  %571 = fmul reassoc nsz arcp contract float %567, %570, !spirv.Decorations !898
  %572 = fadd reassoc nsz arcp contract float %571, %353, !spirv.Decorations !898
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %555
  %573 = phi float [ %572, %555 ], [ %353, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %158, label %574, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

574:                                              ; preds = %._crit_edge.2.2
  %.sroa.64400.0.insert.ext452 = zext i32 %364 to i64
  %575 = shl nuw nsw i64 %.sroa.64400.0.insert.ext452, 1
  %576 = add i64 %344, %575
  %577 = inttoptr i64 %576 to i16 addrspace(4)*
  %578 = addrspacecast i16 addrspace(4)* %577 to i16 addrspace(1)*
  %579 = load i16, i16 addrspace(1)* %578, align 2
  %580 = add i64 %346, %575
  %581 = inttoptr i64 %580 to i16 addrspace(4)*
  %582 = addrspacecast i16 addrspace(4)* %581 to i16 addrspace(1)*
  %583 = load i16, i16 addrspace(1)* %582, align 2
  %584 = zext i16 %579 to i32
  %585 = shl nuw i32 %584, 16, !spirv.Decorations !921
  %586 = bitcast i32 %585 to float
  %587 = zext i16 %583 to i32
  %588 = shl nuw i32 %587, 16, !spirv.Decorations !921
  %589 = bitcast i32 %588 to float
  %590 = fmul reassoc nsz arcp contract float %586, %589, !spirv.Decorations !898
  %591 = fadd reassoc nsz arcp contract float %590, %352, !spirv.Decorations !898
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %574
  %592 = phi float [ %591, %574 ], [ %352, %._crit_edge.2.2..preheader.2_crit_edge ]
  br i1 %162, label %593, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

593:                                              ; preds = %.preheader.2
  %.sroa.64400.0.insert.ext457 = zext i32 %364 to i64
  %594 = shl nuw nsw i64 %.sroa.64400.0.insert.ext457, 1
  %595 = add i64 %340, %594
  %596 = inttoptr i64 %595 to i16 addrspace(4)*
  %597 = addrspacecast i16 addrspace(4)* %596 to i16 addrspace(1)*
  %598 = load i16, i16 addrspace(1)* %597, align 2
  %599 = add i64 %347, %594
  %600 = inttoptr i64 %599 to i16 addrspace(4)*
  %601 = addrspacecast i16 addrspace(4)* %600 to i16 addrspace(1)*
  %602 = load i16, i16 addrspace(1)* %601, align 2
  %603 = zext i16 %598 to i32
  %604 = shl nuw i32 %603, 16, !spirv.Decorations !921
  %605 = bitcast i32 %604 to float
  %606 = zext i16 %602 to i32
  %607 = shl nuw i32 %606, 16, !spirv.Decorations !921
  %608 = bitcast i32 %607 to float
  %609 = fmul reassoc nsz arcp contract float %605, %608, !spirv.Decorations !898
  %610 = fadd reassoc nsz arcp contract float %609, %351, !spirv.Decorations !898
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %593
  %611 = phi float [ %610, %593 ], [ %351, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %165, label %612, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

612:                                              ; preds = %._crit_edge.375
  %.sroa.64400.0.insert.ext462 = zext i32 %364 to i64
  %613 = shl nuw nsw i64 %.sroa.64400.0.insert.ext462, 1
  %614 = add i64 %342, %613
  %615 = inttoptr i64 %614 to i16 addrspace(4)*
  %616 = addrspacecast i16 addrspace(4)* %615 to i16 addrspace(1)*
  %617 = load i16, i16 addrspace(1)* %616, align 2
  %618 = add i64 %347, %613
  %619 = inttoptr i64 %618 to i16 addrspace(4)*
  %620 = addrspacecast i16 addrspace(4)* %619 to i16 addrspace(1)*
  %621 = load i16, i16 addrspace(1)* %620, align 2
  %622 = zext i16 %617 to i32
  %623 = shl nuw i32 %622, 16, !spirv.Decorations !921
  %624 = bitcast i32 %623 to float
  %625 = zext i16 %621 to i32
  %626 = shl nuw i32 %625, 16, !spirv.Decorations !921
  %627 = bitcast i32 %626 to float
  %628 = fmul reassoc nsz arcp contract float %624, %627, !spirv.Decorations !898
  %629 = fadd reassoc nsz arcp contract float %628, %350, !spirv.Decorations !898
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %612
  %630 = phi float [ %629, %612 ], [ %350, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %168, label %631, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

631:                                              ; preds = %._crit_edge.1.3
  %.sroa.64400.0.insert.ext467 = zext i32 %364 to i64
  %632 = shl nuw nsw i64 %.sroa.64400.0.insert.ext467, 1
  %633 = add i64 %343, %632
  %634 = inttoptr i64 %633 to i16 addrspace(4)*
  %635 = addrspacecast i16 addrspace(4)* %634 to i16 addrspace(1)*
  %636 = load i16, i16 addrspace(1)* %635, align 2
  %637 = add i64 %347, %632
  %638 = inttoptr i64 %637 to i16 addrspace(4)*
  %639 = addrspacecast i16 addrspace(4)* %638 to i16 addrspace(1)*
  %640 = load i16, i16 addrspace(1)* %639, align 2
  %641 = zext i16 %636 to i32
  %642 = shl nuw i32 %641, 16, !spirv.Decorations !921
  %643 = bitcast i32 %642 to float
  %644 = zext i16 %640 to i32
  %645 = shl nuw i32 %644, 16, !spirv.Decorations !921
  %646 = bitcast i32 %645 to float
  %647 = fmul reassoc nsz arcp contract float %643, %646, !spirv.Decorations !898
  %648 = fadd reassoc nsz arcp contract float %647, %349, !spirv.Decorations !898
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %631
  %649 = phi float [ %648, %631 ], [ %349, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %171, label %650, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

650:                                              ; preds = %._crit_edge.2.3
  %.sroa.64400.0.insert.ext472 = zext i32 %364 to i64
  %651 = shl nuw nsw i64 %.sroa.64400.0.insert.ext472, 1
  %652 = add i64 %344, %651
  %653 = inttoptr i64 %652 to i16 addrspace(4)*
  %654 = addrspacecast i16 addrspace(4)* %653 to i16 addrspace(1)*
  %655 = load i16, i16 addrspace(1)* %654, align 2
  %656 = add i64 %347, %651
  %657 = inttoptr i64 %656 to i16 addrspace(4)*
  %658 = addrspacecast i16 addrspace(4)* %657 to i16 addrspace(1)*
  %659 = load i16, i16 addrspace(1)* %658, align 2
  %660 = zext i16 %655 to i32
  %661 = shl nuw i32 %660, 16, !spirv.Decorations !921
  %662 = bitcast i32 %661 to float
  %663 = zext i16 %659 to i32
  %664 = shl nuw i32 %663, 16, !spirv.Decorations !921
  %665 = bitcast i32 %664 to float
  %666 = fmul reassoc nsz arcp contract float %662, %665, !spirv.Decorations !898
  %667 = fadd reassoc nsz arcp contract float %666, %348, !spirv.Decorations !898
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %650
  %668 = phi float [ %667, %650 ], [ %348, %._crit_edge.2.3..preheader.3_crit_edge ]
  %669 = add nuw nsw i32 %364, 1, !spirv.Decorations !923
  %670 = icmp slt i32 %669, %const_reg_dword2
  br i1 %670, label %.preheader.3..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.3..preheader.preheader_crit_edge:      ; preds = %.preheader.3
  br label %.preheader.preheader, !stats.blockFrequency.digits !924, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.3
  %.lcssa1021 = phi float [ %668, %.preheader.3 ]
  %.lcssa1020 = phi float [ %649, %.preheader.3 ]
  %.lcssa1019 = phi float [ %630, %.preheader.3 ]
  %.lcssa1018 = phi float [ %611, %.preheader.3 ]
  %.lcssa1017 = phi float [ %592, %.preheader.3 ]
  %.lcssa1016 = phi float [ %573, %.preheader.3 ]
  %.lcssa1015 = phi float [ %554, %.preheader.3 ]
  %.lcssa1014 = phi float [ %535, %.preheader.3 ]
  %.lcssa1013 = phi float [ %516, %.preheader.3 ]
  %.lcssa1012 = phi float [ %497, %.preheader.3 ]
  %.lcssa1011 = phi float [ %478, %.preheader.3 ]
  %.lcssa1010 = phi float [ %459, %.preheader.3 ]
  %.lcssa1009 = phi float [ %440, %.preheader.3 ]
  %.lcssa1008 = phi float [ %421, %.preheader.3 ]
  %.lcssa1007 = phi float [ %402, %.preheader.3 ]
  %.lcssa = phi float [ %383, %.preheader.3 ]
  br label %.preheader1.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
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
  br i1 %120, label %671, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
  br label %._crit_edge70, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

671:                                              ; preds = %.preheader1.preheader
  %672 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !898
  br i1 %81, label %677, label %673, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

673:                                              ; preds = %671
  %674 = add i64 %.in, %244
  %675 = inttoptr i64 %674 to float addrspace(4)*
  %676 = addrspacecast float addrspace(4)* %675 to float addrspace(1)*
  store float %672, float addrspace(1)* %676, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

677:                                              ; preds = %671
  %678 = add i64 %.in988, %251
  %679 = add i64 %678, %252
  %680 = inttoptr i64 %679 to float addrspace(4)*
  %681 = addrspacecast float addrspace(4)* %680 to float addrspace(1)*
  %682 = load float, float addrspace(1)* %681, align 4
  %683 = fmul reassoc nsz arcp contract float %682, %4, !spirv.Decorations !898
  %684 = fadd reassoc nsz arcp contract float %672, %683, !spirv.Decorations !898
  %685 = add i64 %.in, %244
  %686 = inttoptr i64 %685 to float addrspace(4)*
  %687 = addrspacecast float addrspace(4)* %686 to float addrspace(1)*
  store float %684, float addrspace(1)* %687, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %673, %677
  br i1 %124, label %688, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
  br label %._crit_edge70.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

688:                                              ; preds = %._crit_edge70
  %689 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !898
  br i1 %81, label %694, label %690, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

690:                                              ; preds = %688
  %691 = add i64 %.in, %260
  %692 = inttoptr i64 %691 to float addrspace(4)*
  %693 = addrspacecast float addrspace(4)* %692 to float addrspace(1)*
  store float %689, float addrspace(1)* %693, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

694:                                              ; preds = %688
  %695 = add i64 %.in988, %267
  %696 = add i64 %695, %252
  %697 = inttoptr i64 %696 to float addrspace(4)*
  %698 = addrspacecast float addrspace(4)* %697 to float addrspace(1)*
  %699 = load float, float addrspace(1)* %698, align 4
  %700 = fmul reassoc nsz arcp contract float %699, %4, !spirv.Decorations !898
  %701 = fadd reassoc nsz arcp contract float %689, %700, !spirv.Decorations !898
  %702 = add i64 %.in, %260
  %703 = inttoptr i64 %702 to float addrspace(4)*
  %704 = addrspacecast float addrspace(4)* %703 to float addrspace(1)*
  store float %701, float addrspace(1)* %704, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %694, %690
  br i1 %128, label %705, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
  br label %._crit_edge70.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

705:                                              ; preds = %._crit_edge70.1
  %706 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !898
  br i1 %81, label %711, label %707, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

707:                                              ; preds = %705
  %708 = add i64 %.in, %275
  %709 = inttoptr i64 %708 to float addrspace(4)*
  %710 = addrspacecast float addrspace(4)* %709 to float addrspace(1)*
  store float %706, float addrspace(1)* %710, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

711:                                              ; preds = %705
  %712 = add i64 %.in988, %282
  %713 = add i64 %712, %252
  %714 = inttoptr i64 %713 to float addrspace(4)*
  %715 = addrspacecast float addrspace(4)* %714 to float addrspace(1)*
  %716 = load float, float addrspace(1)* %715, align 4
  %717 = fmul reassoc nsz arcp contract float %716, %4, !spirv.Decorations !898
  %718 = fadd reassoc nsz arcp contract float %706, %717, !spirv.Decorations !898
  %719 = add i64 %.in, %275
  %720 = inttoptr i64 %719 to float addrspace(4)*
  %721 = addrspacecast float addrspace(4)* %720 to float addrspace(1)*
  store float %718, float addrspace(1)* %721, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %711, %707
  br i1 %132, label %722, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
  br label %.preheader1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

722:                                              ; preds = %._crit_edge70.2
  %723 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !898
  br i1 %81, label %728, label %724, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

724:                                              ; preds = %722
  %725 = add i64 %.in, %290
  %726 = inttoptr i64 %725 to float addrspace(4)*
  %727 = addrspacecast float addrspace(4)* %726 to float addrspace(1)*
  store float %723, float addrspace(1)* %727, align 4
  br label %.preheader1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

728:                                              ; preds = %722
  %729 = add i64 %.in988, %297
  %730 = add i64 %729, %252
  %731 = inttoptr i64 %730 to float addrspace(4)*
  %732 = addrspacecast float addrspace(4)* %731 to float addrspace(1)*
  %733 = load float, float addrspace(1)* %732, align 4
  %734 = fmul reassoc nsz arcp contract float %733, %4, !spirv.Decorations !898
  %735 = fadd reassoc nsz arcp contract float %723, %734, !spirv.Decorations !898
  %736 = add i64 %.in, %290
  %737 = inttoptr i64 %736 to float addrspace(4)*
  %738 = addrspacecast float addrspace(4)* %737 to float addrspace(1)*
  store float %735, float addrspace(1)* %738, align 4
  br label %.preheader1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %728, %724
  br i1 %136, label %739, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
  br label %._crit_edge70.176, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

739:                                              ; preds = %.preheader1
  %740 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !898
  br i1 %81, label %745, label %741, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

741:                                              ; preds = %739
  %742 = add i64 %.in, %300
  %743 = inttoptr i64 %742 to float addrspace(4)*
  %744 = addrspacecast float addrspace(4)* %743 to float addrspace(1)*
  store float %740, float addrspace(1)* %744, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

745:                                              ; preds = %739
  %746 = add i64 %.in988, %251
  %747 = add i64 %746, %301
  %748 = inttoptr i64 %747 to float addrspace(4)*
  %749 = addrspacecast float addrspace(4)* %748 to float addrspace(1)*
  %750 = load float, float addrspace(1)* %749, align 4
  %751 = fmul reassoc nsz arcp contract float %750, %4, !spirv.Decorations !898
  %752 = fadd reassoc nsz arcp contract float %740, %751, !spirv.Decorations !898
  %753 = add i64 %.in, %300
  %754 = inttoptr i64 %753 to float addrspace(4)*
  %755 = addrspacecast float addrspace(4)* %754 to float addrspace(1)*
  store float %752, float addrspace(1)* %755, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %745, %741
  br i1 %139, label %756, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

756:                                              ; preds = %._crit_edge70.176
  %757 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !898
  br i1 %81, label %762, label %758, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

758:                                              ; preds = %756
  %759 = add i64 %.in, %303
  %760 = inttoptr i64 %759 to float addrspace(4)*
  %761 = addrspacecast float addrspace(4)* %760 to float addrspace(1)*
  store float %757, float addrspace(1)* %761, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

762:                                              ; preds = %756
  %763 = add i64 %.in988, %267
  %764 = add i64 %763, %301
  %765 = inttoptr i64 %764 to float addrspace(4)*
  %766 = addrspacecast float addrspace(4)* %765 to float addrspace(1)*
  %767 = load float, float addrspace(1)* %766, align 4
  %768 = fmul reassoc nsz arcp contract float %767, %4, !spirv.Decorations !898
  %769 = fadd reassoc nsz arcp contract float %757, %768, !spirv.Decorations !898
  %770 = add i64 %.in, %303
  %771 = inttoptr i64 %770 to float addrspace(4)*
  %772 = addrspacecast float addrspace(4)* %771 to float addrspace(1)*
  store float %769, float addrspace(1)* %772, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %762, %758
  br i1 %142, label %773, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

773:                                              ; preds = %._crit_edge70.1.1
  %774 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !898
  br i1 %81, label %779, label %775, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

775:                                              ; preds = %773
  %776 = add i64 %.in, %305
  %777 = inttoptr i64 %776 to float addrspace(4)*
  %778 = addrspacecast float addrspace(4)* %777 to float addrspace(1)*
  store float %774, float addrspace(1)* %778, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

779:                                              ; preds = %773
  %780 = add i64 %.in988, %282
  %781 = add i64 %780, %301
  %782 = inttoptr i64 %781 to float addrspace(4)*
  %783 = addrspacecast float addrspace(4)* %782 to float addrspace(1)*
  %784 = load float, float addrspace(1)* %783, align 4
  %785 = fmul reassoc nsz arcp contract float %784, %4, !spirv.Decorations !898
  %786 = fadd reassoc nsz arcp contract float %774, %785, !spirv.Decorations !898
  %787 = add i64 %.in, %305
  %788 = inttoptr i64 %787 to float addrspace(4)*
  %789 = addrspacecast float addrspace(4)* %788 to float addrspace(1)*
  store float %786, float addrspace(1)* %789, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %779, %775
  br i1 %145, label %790, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
  br label %.preheader1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

790:                                              ; preds = %._crit_edge70.2.1
  %791 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !898
  br i1 %81, label %796, label %792, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

792:                                              ; preds = %790
  %793 = add i64 %.in, %307
  %794 = inttoptr i64 %793 to float addrspace(4)*
  %795 = addrspacecast float addrspace(4)* %794 to float addrspace(1)*
  store float %791, float addrspace(1)* %795, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

796:                                              ; preds = %790
  %797 = add i64 %.in988, %297
  %798 = add i64 %797, %301
  %799 = inttoptr i64 %798 to float addrspace(4)*
  %800 = addrspacecast float addrspace(4)* %799 to float addrspace(1)*
  %801 = load float, float addrspace(1)* %800, align 4
  %802 = fmul reassoc nsz arcp contract float %801, %4, !spirv.Decorations !898
  %803 = fadd reassoc nsz arcp contract float %791, %802, !spirv.Decorations !898
  %804 = add i64 %.in, %307
  %805 = inttoptr i64 %804 to float addrspace(4)*
  %806 = addrspacecast float addrspace(4)* %805 to float addrspace(1)*
  store float %803, float addrspace(1)* %806, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %796, %792
  br i1 %149, label %807, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
  br label %._crit_edge70.277, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

807:                                              ; preds = %.preheader1.1
  %808 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !898
  br i1 %81, label %813, label %809, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

809:                                              ; preds = %807
  %810 = add i64 %.in, %310
  %811 = inttoptr i64 %810 to float addrspace(4)*
  %812 = addrspacecast float addrspace(4)* %811 to float addrspace(1)*
  store float %808, float addrspace(1)* %812, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

813:                                              ; preds = %807
  %814 = add i64 %.in988, %251
  %815 = add i64 %814, %311
  %816 = inttoptr i64 %815 to float addrspace(4)*
  %817 = addrspacecast float addrspace(4)* %816 to float addrspace(1)*
  %818 = load float, float addrspace(1)* %817, align 4
  %819 = fmul reassoc nsz arcp contract float %818, %4, !spirv.Decorations !898
  %820 = fadd reassoc nsz arcp contract float %808, %819, !spirv.Decorations !898
  %821 = add i64 %.in, %310
  %822 = inttoptr i64 %821 to float addrspace(4)*
  %823 = addrspacecast float addrspace(4)* %822 to float addrspace(1)*
  store float %820, float addrspace(1)* %823, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %813, %809
  br i1 %152, label %824, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

824:                                              ; preds = %._crit_edge70.277
  %825 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !898
  br i1 %81, label %830, label %826, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

826:                                              ; preds = %824
  %827 = add i64 %.in, %313
  %828 = inttoptr i64 %827 to float addrspace(4)*
  %829 = addrspacecast float addrspace(4)* %828 to float addrspace(1)*
  store float %825, float addrspace(1)* %829, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

830:                                              ; preds = %824
  %831 = add i64 %.in988, %267
  %832 = add i64 %831, %311
  %833 = inttoptr i64 %832 to float addrspace(4)*
  %834 = addrspacecast float addrspace(4)* %833 to float addrspace(1)*
  %835 = load float, float addrspace(1)* %834, align 4
  %836 = fmul reassoc nsz arcp contract float %835, %4, !spirv.Decorations !898
  %837 = fadd reassoc nsz arcp contract float %825, %836, !spirv.Decorations !898
  %838 = add i64 %.in, %313
  %839 = inttoptr i64 %838 to float addrspace(4)*
  %840 = addrspacecast float addrspace(4)* %839 to float addrspace(1)*
  store float %837, float addrspace(1)* %840, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %830, %826
  br i1 %155, label %841, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

841:                                              ; preds = %._crit_edge70.1.2
  %842 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !898
  br i1 %81, label %847, label %843, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

843:                                              ; preds = %841
  %844 = add i64 %.in, %315
  %845 = inttoptr i64 %844 to float addrspace(4)*
  %846 = addrspacecast float addrspace(4)* %845 to float addrspace(1)*
  store float %842, float addrspace(1)* %846, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

847:                                              ; preds = %841
  %848 = add i64 %.in988, %282
  %849 = add i64 %848, %311
  %850 = inttoptr i64 %849 to float addrspace(4)*
  %851 = addrspacecast float addrspace(4)* %850 to float addrspace(1)*
  %852 = load float, float addrspace(1)* %851, align 4
  %853 = fmul reassoc nsz arcp contract float %852, %4, !spirv.Decorations !898
  %854 = fadd reassoc nsz arcp contract float %842, %853, !spirv.Decorations !898
  %855 = add i64 %.in, %315
  %856 = inttoptr i64 %855 to float addrspace(4)*
  %857 = addrspacecast float addrspace(4)* %856 to float addrspace(1)*
  store float %854, float addrspace(1)* %857, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %847, %843
  br i1 %158, label %858, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
  br label %.preheader1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

858:                                              ; preds = %._crit_edge70.2.2
  %859 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !898
  br i1 %81, label %864, label %860, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

860:                                              ; preds = %858
  %861 = add i64 %.in, %317
  %862 = inttoptr i64 %861 to float addrspace(4)*
  %863 = addrspacecast float addrspace(4)* %862 to float addrspace(1)*
  store float %859, float addrspace(1)* %863, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

864:                                              ; preds = %858
  %865 = add i64 %.in988, %297
  %866 = add i64 %865, %311
  %867 = inttoptr i64 %866 to float addrspace(4)*
  %868 = addrspacecast float addrspace(4)* %867 to float addrspace(1)*
  %869 = load float, float addrspace(1)* %868, align 4
  %870 = fmul reassoc nsz arcp contract float %869, %4, !spirv.Decorations !898
  %871 = fadd reassoc nsz arcp contract float %859, %870, !spirv.Decorations !898
  %872 = add i64 %.in, %317
  %873 = inttoptr i64 %872 to float addrspace(4)*
  %874 = addrspacecast float addrspace(4)* %873 to float addrspace(1)*
  store float %871, float addrspace(1)* %874, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %864, %860
  br i1 %162, label %875, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
  br label %._crit_edge70.378, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

875:                                              ; preds = %.preheader1.2
  %876 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !898
  br i1 %81, label %881, label %877, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

877:                                              ; preds = %875
  %878 = add i64 %.in, %320
  %879 = inttoptr i64 %878 to float addrspace(4)*
  %880 = addrspacecast float addrspace(4)* %879 to float addrspace(1)*
  store float %876, float addrspace(1)* %880, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

881:                                              ; preds = %875
  %882 = add i64 %.in988, %251
  %883 = add i64 %882, %321
  %884 = inttoptr i64 %883 to float addrspace(4)*
  %885 = addrspacecast float addrspace(4)* %884 to float addrspace(1)*
  %886 = load float, float addrspace(1)* %885, align 4
  %887 = fmul reassoc nsz arcp contract float %886, %4, !spirv.Decorations !898
  %888 = fadd reassoc nsz arcp contract float %876, %887, !spirv.Decorations !898
  %889 = add i64 %.in, %320
  %890 = inttoptr i64 %889 to float addrspace(4)*
  %891 = addrspacecast float addrspace(4)* %890 to float addrspace(1)*
  store float %888, float addrspace(1)* %891, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %881, %877
  br i1 %165, label %892, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

892:                                              ; preds = %._crit_edge70.378
  %893 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !898
  br i1 %81, label %898, label %894, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

894:                                              ; preds = %892
  %895 = add i64 %.in, %323
  %896 = inttoptr i64 %895 to float addrspace(4)*
  %897 = addrspacecast float addrspace(4)* %896 to float addrspace(1)*
  store float %893, float addrspace(1)* %897, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

898:                                              ; preds = %892
  %899 = add i64 %.in988, %267
  %900 = add i64 %899, %321
  %901 = inttoptr i64 %900 to float addrspace(4)*
  %902 = addrspacecast float addrspace(4)* %901 to float addrspace(1)*
  %903 = load float, float addrspace(1)* %902, align 4
  %904 = fmul reassoc nsz arcp contract float %903, %4, !spirv.Decorations !898
  %905 = fadd reassoc nsz arcp contract float %893, %904, !spirv.Decorations !898
  %906 = add i64 %.in, %323
  %907 = inttoptr i64 %906 to float addrspace(4)*
  %908 = addrspacecast float addrspace(4)* %907 to float addrspace(1)*
  store float %905, float addrspace(1)* %908, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %898, %894
  br i1 %168, label %909, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

909:                                              ; preds = %._crit_edge70.1.3
  %910 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !898
  br i1 %81, label %915, label %911, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

911:                                              ; preds = %909
  %912 = add i64 %.in, %325
  %913 = inttoptr i64 %912 to float addrspace(4)*
  %914 = addrspacecast float addrspace(4)* %913 to float addrspace(1)*
  store float %910, float addrspace(1)* %914, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

915:                                              ; preds = %909
  %916 = add i64 %.in988, %282
  %917 = add i64 %916, %321
  %918 = inttoptr i64 %917 to float addrspace(4)*
  %919 = addrspacecast float addrspace(4)* %918 to float addrspace(1)*
  %920 = load float, float addrspace(1)* %919, align 4
  %921 = fmul reassoc nsz arcp contract float %920, %4, !spirv.Decorations !898
  %922 = fadd reassoc nsz arcp contract float %910, %921, !spirv.Decorations !898
  %923 = add i64 %.in, %325
  %924 = inttoptr i64 %923 to float addrspace(4)*
  %925 = addrspacecast float addrspace(4)* %924 to float addrspace(1)*
  store float %922, float addrspace(1)* %925, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %915, %911
  br i1 %171, label %926, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
  br label %.preheader1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

926:                                              ; preds = %._crit_edge70.2.3
  %927 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !898
  br i1 %81, label %932, label %928, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

928:                                              ; preds = %926
  %929 = add i64 %.in, %327
  %930 = inttoptr i64 %929 to float addrspace(4)*
  %931 = addrspacecast float addrspace(4)* %930 to float addrspace(1)*
  store float %927, float addrspace(1)* %931, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

932:                                              ; preds = %926
  %933 = add i64 %.in988, %297
  %934 = add i64 %933, %321
  %935 = inttoptr i64 %934 to float addrspace(4)*
  %936 = addrspacecast float addrspace(4)* %935 to float addrspace(1)*
  %937 = load float, float addrspace(1)* %936, align 4
  %938 = fmul reassoc nsz arcp contract float %937, %4, !spirv.Decorations !898
  %939 = fadd reassoc nsz arcp contract float %927, %938, !spirv.Decorations !898
  %940 = add i64 %.in, %327
  %941 = inttoptr i64 %940 to float addrspace(4)*
  %942 = addrspacecast float addrspace(4)* %941 to float addrspace(1)*
  store float %939, float addrspace(1)* %942, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %932, %928
  %943 = add i32 %339, %52
  %944 = icmp slt i32 %943, %8
  br i1 %944, label %.preheader1.3..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge72.loopexit:                           ; preds = %.preheader1.3
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.3..preheader2.preheader_crit_edge:    ; preds = %.preheader1.3
  %945 = add i64 %.in990, %328
  %946 = add i64 %.in989, %329
  %947 = add i64 %.in988, %337
  %948 = add i64 %.in, %338
  br label %.preheader2.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = bitcast i64 %9 to <2 x i32>
  %15 = extractelement <2 x i32> %14, i32 0
  %16 = extractelement <2 x i32> %14, i32 1
  %17 = bitcast i64 %10 to <2 x i32>
  %18 = extractelement <2 x i32> %17, i32 0
  %19 = extractelement <2 x i32> %17, i32 1
  %20 = bitcast i64 %11 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = extractelement <2 x i32> %20, i32 1
  %23 = bitcast i64 %12 to <2 x i32>
  %24 = extractelement <2 x i32> %23, i32 0
  %25 = extractelement <2 x i32> %23, i32 1
  %26 = extractelement <8 x i32> %r0, i32 7
  %27 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %15, i32 %16)
  %28 = extractvalue { i32, i32 } %27, 0
  %29 = extractvalue { i32, i32 } %27, 1
  %30 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %18, i32 %19)
  %31 = extractvalue { i32, i32 } %30, 0
  %32 = extractvalue { i32, i32 } %30, 1
  %33 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %21, i32 %22)
  %34 = extractvalue { i32, i32 } %33, 0
  %35 = extractvalue { i32, i32 } %33, 1
  %36 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %24, i32 %25)
  %37 = extractvalue { i32, i32 } %36, 0
  %38 = extractvalue { i32, i32 } %36, 1
  %39 = icmp slt i32 %26, %8
  br i1 %39, label %.lr.ph, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge72_crit_edge:                         ; preds = %13
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %13
  %40 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = extractelement <2 x i32> %40, i32 1
  %43 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = extractelement <2 x i32> %43, i32 1
  %46 = bitcast i64 %const_reg_qword7 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i32 0
  %48 = extractelement <2 x i32> %46, i32 1
  %49 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i32 0
  %51 = extractelement <2 x i32> %49, i32 1
  %52 = extractelement <3 x i32> %numWorkGroups, i32 2
  %53 = extractelement <3 x i32> %localSize, i32 0
  %54 = extractelement <3 x i32> %localSize, i32 1
  %55 = extractelement <8 x i32> %r0, i32 1
  %56 = extractelement <8 x i32> %r0, i32 6
  %57 = mul i32 %55, %53
  %58 = zext i16 %localIdX to i32
  %59 = add i32 %57, %58
  %60 = shl i32 %59, 2
  %61 = mul i32 %56, %54
  %62 = zext i16 %localIdY to i32
  %63 = add i32 %61, %62
  %64 = shl i32 %63, 4
  %65 = insertelement <2 x i32> undef, i32 %28, i32 0
  %66 = insertelement <2 x i32> %65, i32 %29, i32 1
  %67 = bitcast <2 x i32> %66 to i64
  %68 = shl i64 %67, 1
  %69 = add i64 %68, %const_reg_qword
  %70 = insertelement <2 x i32> undef, i32 %31, i32 0
  %71 = insertelement <2 x i32> %70, i32 %32, i32 1
  %72 = bitcast <2 x i32> %71 to i64
  %73 = shl i64 %72, 1
  %74 = add i64 %73, %const_reg_qword4
  %75 = insertelement <2 x i32> undef, i32 %34, i32 0
  %76 = insertelement <2 x i32> %75, i32 %35, i32 1
  %77 = bitcast <2 x i32> %76 to i64
  %.op = shl i64 %77, 2
  %78 = bitcast i64 %.op to <2 x i32>
  %79 = extractelement <2 x i32> %78, i32 0
  %80 = extractelement <2 x i32> %78, i32 1
  %81 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !898
  %82 = select i1 %81, i32 %79, i32 0
  %83 = select i1 %81, i32 %80, i32 0
  %84 = insertelement <2 x i32> undef, i32 %82, i32 0
  %85 = insertelement <2 x i32> %84, i32 %83, i32 1
  %86 = bitcast <2 x i32> %85 to i64
  %87 = add i64 %86, %const_reg_qword6
  %88 = insertelement <2 x i32> undef, i32 %37, i32 0
  %89 = insertelement <2 x i32> %88, i32 %38, i32 1
  %90 = bitcast <2 x i32> %89 to i64
  %91 = shl i64 %90, 2
  %92 = add i64 %91, %const_reg_qword8
  %93 = icmp sgt i32 %const_reg_dword2, 0
  %94 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %15, i32 %16)
  %95 = extractvalue { i32, i32 } %94, 0
  %96 = extractvalue { i32, i32 } %94, 1
  %97 = insertelement <2 x i32> undef, i32 %95, i32 0
  %98 = insertelement <2 x i32> %97, i32 %96, i32 1
  %99 = bitcast <2 x i32> %98 to i64
  %100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %18, i32 %19)
  %101 = extractvalue { i32, i32 } %100, 0
  %102 = extractvalue { i32, i32 } %100, 1
  %103 = insertelement <2 x i32> undef, i32 %101, i32 0
  %104 = insertelement <2 x i32> %103, i32 %102, i32 1
  %105 = bitcast <2 x i32> %104 to i64
  %106 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %21, i32 %22)
  %107 = extractvalue { i32, i32 } %106, 0
  %108 = extractvalue { i32, i32 } %106, 1
  %109 = insertelement <2 x i32> undef, i32 %107, i32 0
  %110 = insertelement <2 x i32> %109, i32 %108, i32 1
  %111 = bitcast <2 x i32> %110 to i64
  %112 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %24, i32 %25)
  %113 = extractvalue { i32, i32 } %112, 0
  %114 = extractvalue { i32, i32 } %112, 1
  %115 = insertelement <2 x i32> undef, i32 %113, i32 0
  %116 = insertelement <2 x i32> %115, i32 %114, i32 1
  %117 = bitcast <2 x i32> %116 to i64
  %118 = icmp slt i32 %60, %const_reg_dword
  %119 = icmp slt i32 %64, %const_reg_dword1
  %120 = and i1 %118, %119
  %121 = add i32 %60, 1
  %122 = icmp slt i32 %121, %const_reg_dword
  %123 = icmp slt i32 %64, %const_reg_dword1
  %124 = and i1 %122, %123
  %125 = add i32 %60, 2
  %126 = icmp slt i32 %125, %const_reg_dword
  %127 = icmp slt i32 %64, %const_reg_dword1
  %128 = and i1 %126, %127
  %129 = add i32 %60, 3
  %130 = icmp slt i32 %129, %const_reg_dword
  %131 = icmp slt i32 %64, %const_reg_dword1
  %132 = and i1 %130, %131
  %133 = add i32 %64, 1
  %134 = icmp slt i32 %133, %const_reg_dword1
  %135 = icmp slt i32 %60, %const_reg_dword
  %136 = and i1 %135, %134
  %137 = icmp slt i32 %121, %const_reg_dword
  %138 = icmp slt i32 %133, %const_reg_dword1
  %139 = and i1 %137, %138
  %140 = icmp slt i32 %125, %const_reg_dword
  %141 = icmp slt i32 %133, %const_reg_dword1
  %142 = and i1 %140, %141
  %143 = icmp slt i32 %129, %const_reg_dword
  %144 = icmp slt i32 %133, %const_reg_dword1
  %145 = and i1 %143, %144
  %146 = add i32 %64, 2
  %147 = icmp slt i32 %146, %const_reg_dword1
  %148 = icmp slt i32 %60, %const_reg_dword
  %149 = and i1 %148, %147
  %150 = icmp slt i32 %121, %const_reg_dword
  %151 = icmp slt i32 %146, %const_reg_dword1
  %152 = and i1 %150, %151
  %153 = icmp slt i32 %125, %const_reg_dword
  %154 = icmp slt i32 %146, %const_reg_dword1
  %155 = and i1 %153, %154
  %156 = icmp slt i32 %129, %const_reg_dword
  %157 = icmp slt i32 %146, %const_reg_dword1
  %158 = and i1 %156, %157
  %159 = add i32 %64, 3
  %160 = icmp slt i32 %159, %const_reg_dword1
  %161 = icmp slt i32 %60, %const_reg_dword
  %162 = and i1 %161, %160
  %163 = icmp slt i32 %121, %const_reg_dword
  %164 = icmp slt i32 %159, %const_reg_dword1
  %165 = and i1 %163, %164
  %166 = icmp slt i32 %125, %const_reg_dword
  %167 = icmp slt i32 %159, %const_reg_dword1
  %168 = and i1 %166, %167
  %169 = icmp slt i32 %129, %const_reg_dword
  %170 = icmp slt i32 %159, %const_reg_dword1
  %171 = and i1 %169, %170
  %172 = add i32 %64, 4
  %173 = icmp slt i32 %172, %const_reg_dword1
  %174 = icmp slt i32 %60, %const_reg_dword
  %175 = and i1 %174, %173
  %176 = icmp slt i32 %121, %const_reg_dword
  %177 = icmp slt i32 %172, %const_reg_dword1
  %178 = and i1 %176, %177
  %179 = icmp slt i32 %125, %const_reg_dword
  %180 = icmp slt i32 %172, %const_reg_dword1
  %181 = and i1 %179, %180
  %182 = icmp slt i32 %129, %const_reg_dword
  %183 = icmp slt i32 %172, %const_reg_dword1
  %184 = and i1 %182, %183
  %185 = add i32 %64, 5
  %186 = icmp slt i32 %185, %const_reg_dword1
  %187 = icmp slt i32 %60, %const_reg_dword
  %188 = and i1 %187, %186
  %189 = icmp slt i32 %121, %const_reg_dword
  %190 = icmp slt i32 %185, %const_reg_dword1
  %191 = and i1 %189, %190
  %192 = icmp slt i32 %125, %const_reg_dword
  %193 = icmp slt i32 %185, %const_reg_dword1
  %194 = and i1 %192, %193
  %195 = icmp slt i32 %129, %const_reg_dword
  %196 = icmp slt i32 %185, %const_reg_dword1
  %197 = and i1 %195, %196
  %198 = add i32 %64, 6
  %199 = icmp slt i32 %198, %const_reg_dword1
  %200 = icmp slt i32 %60, %const_reg_dword
  %201 = and i1 %200, %199
  %202 = icmp slt i32 %121, %const_reg_dword
  %203 = icmp slt i32 %198, %const_reg_dword1
  %204 = and i1 %202, %203
  %205 = icmp slt i32 %125, %const_reg_dword
  %206 = icmp slt i32 %198, %const_reg_dword1
  %207 = and i1 %205, %206
  %208 = icmp slt i32 %129, %const_reg_dword
  %209 = icmp slt i32 %198, %const_reg_dword1
  %210 = and i1 %208, %209
  %211 = add i32 %64, 7
  %212 = icmp slt i32 %211, %const_reg_dword1
  %213 = icmp slt i32 %60, %const_reg_dword
  %214 = and i1 %213, %212
  %215 = icmp slt i32 %121, %const_reg_dword
  %216 = icmp slt i32 %211, %const_reg_dword1
  %217 = and i1 %215, %216
  %218 = icmp slt i32 %125, %const_reg_dword
  %219 = icmp slt i32 %211, %const_reg_dword1
  %220 = and i1 %218, %219
  %221 = icmp slt i32 %129, %const_reg_dword
  %222 = icmp slt i32 %211, %const_reg_dword1
  %223 = and i1 %221, %222
  %224 = add i32 %64, 8
  %225 = icmp slt i32 %224, %const_reg_dword1
  %226 = icmp slt i32 %60, %const_reg_dword
  %227 = and i1 %226, %225
  %228 = icmp slt i32 %121, %const_reg_dword
  %229 = icmp slt i32 %224, %const_reg_dword1
  %230 = and i1 %228, %229
  %231 = icmp slt i32 %125, %const_reg_dword
  %232 = icmp slt i32 %224, %const_reg_dword1
  %233 = and i1 %231, %232
  %234 = icmp slt i32 %129, %const_reg_dword
  %235 = icmp slt i32 %224, %const_reg_dword1
  %236 = and i1 %234, %235
  %237 = add i32 %64, 9
  %238 = icmp slt i32 %237, %const_reg_dword1
  %239 = icmp slt i32 %60, %const_reg_dword
  %240 = and i1 %239, %238
  %241 = icmp slt i32 %121, %const_reg_dword
  %242 = icmp slt i32 %237, %const_reg_dword1
  %243 = and i1 %241, %242
  %244 = icmp slt i32 %125, %const_reg_dword
  %245 = icmp slt i32 %237, %const_reg_dword1
  %246 = and i1 %244, %245
  %247 = icmp slt i32 %129, %const_reg_dword
  %248 = icmp slt i32 %237, %const_reg_dword1
  %249 = and i1 %247, %248
  %250 = add i32 %64, 10
  %251 = icmp slt i32 %250, %const_reg_dword1
  %252 = icmp slt i32 %60, %const_reg_dword
  %253 = and i1 %252, %251
  %254 = icmp slt i32 %121, %const_reg_dword
  %255 = icmp slt i32 %250, %const_reg_dword1
  %256 = and i1 %254, %255
  %257 = icmp slt i32 %125, %const_reg_dword
  %258 = icmp slt i32 %250, %const_reg_dword1
  %259 = and i1 %257, %258
  %260 = icmp slt i32 %129, %const_reg_dword
  %261 = icmp slt i32 %250, %const_reg_dword1
  %262 = and i1 %260, %261
  %263 = add i32 %64, 11
  %264 = icmp slt i32 %263, %const_reg_dword1
  %265 = icmp slt i32 %60, %const_reg_dword
  %266 = and i1 %265, %264
  %267 = icmp slt i32 %121, %const_reg_dword
  %268 = icmp slt i32 %263, %const_reg_dword1
  %269 = and i1 %267, %268
  %270 = icmp slt i32 %125, %const_reg_dword
  %271 = icmp slt i32 %263, %const_reg_dword1
  %272 = and i1 %270, %271
  %273 = icmp slt i32 %129, %const_reg_dword
  %274 = icmp slt i32 %263, %const_reg_dword1
  %275 = and i1 %273, %274
  %276 = add i32 %64, 12
  %277 = icmp slt i32 %276, %const_reg_dword1
  %278 = icmp slt i32 %60, %const_reg_dword
  %279 = and i1 %278, %277
  %280 = icmp slt i32 %121, %const_reg_dword
  %281 = icmp slt i32 %276, %const_reg_dword1
  %282 = and i1 %280, %281
  %283 = icmp slt i32 %125, %const_reg_dword
  %284 = icmp slt i32 %276, %const_reg_dword1
  %285 = and i1 %283, %284
  %286 = icmp slt i32 %129, %const_reg_dword
  %287 = icmp slt i32 %276, %const_reg_dword1
  %288 = and i1 %286, %287
  %289 = add i32 %64, 13
  %290 = icmp slt i32 %289, %const_reg_dword1
  %291 = icmp slt i32 %60, %const_reg_dword
  %292 = and i1 %291, %290
  %293 = icmp slt i32 %121, %const_reg_dword
  %294 = icmp slt i32 %289, %const_reg_dword1
  %295 = and i1 %293, %294
  %296 = icmp slt i32 %125, %const_reg_dword
  %297 = icmp slt i32 %289, %const_reg_dword1
  %298 = and i1 %296, %297
  %299 = icmp slt i32 %129, %const_reg_dword
  %300 = icmp slt i32 %289, %const_reg_dword1
  %301 = and i1 %299, %300
  %302 = add i32 %64, 14
  %303 = icmp slt i32 %302, %const_reg_dword1
  %304 = icmp slt i32 %60, %const_reg_dword
  %305 = and i1 %304, %303
  %306 = icmp slt i32 %121, %const_reg_dword
  %307 = icmp slt i32 %302, %const_reg_dword1
  %308 = and i1 %306, %307
  %309 = icmp slt i32 %125, %const_reg_dword
  %310 = icmp slt i32 %302, %const_reg_dword1
  %311 = and i1 %309, %310
  %312 = icmp slt i32 %129, %const_reg_dword
  %313 = icmp slt i32 %302, %const_reg_dword1
  %314 = and i1 %312, %313
  %315 = add i32 %64, 15
  %316 = icmp slt i32 %315, %const_reg_dword1
  %317 = icmp slt i32 %60, %const_reg_dword
  %318 = and i1 %317, %316
  %319 = icmp slt i32 %121, %const_reg_dword
  %320 = icmp slt i32 %315, %const_reg_dword1
  %321 = and i1 %319, %320
  %322 = icmp slt i32 %125, %const_reg_dword
  %323 = icmp slt i32 %315, %const_reg_dword1
  %324 = and i1 %322, %323
  %325 = icmp slt i32 %129, %const_reg_dword
  %326 = icmp slt i32 %315, %const_reg_dword1
  %327 = and i1 %325, %326
  %328 = ashr i32 %60, 31
  %329 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %41, i32 %42)
  %330 = extractvalue { i32, i32 } %329, 0
  %331 = extractvalue { i32, i32 } %329, 1
  %332 = insertelement <2 x i32> undef, i32 %330, i32 0
  %333 = insertelement <2 x i32> %332, i32 %331, i32 1
  %334 = ashr i32 %64, 31
  %335 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %64, i32 %334, i32 %44, i32 %45)
  %336 = extractvalue { i32, i32 } %335, 0
  %337 = extractvalue { i32, i32 } %335, 1
  %338 = insertelement <2 x i32> undef, i32 %336, i32 0
  %339 = insertelement <2 x i32> %338, i32 %337, i32 1
  %340 = ashr i32 %121, 31
  %341 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %41, i32 %42)
  %342 = extractvalue { i32, i32 } %341, 0
  %343 = extractvalue { i32, i32 } %341, 1
  %344 = insertelement <2 x i32> undef, i32 %342, i32 0
  %345 = insertelement <2 x i32> %344, i32 %343, i32 1
  %346 = ashr i32 %125, 31
  %347 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %41, i32 %42)
  %348 = extractvalue { i32, i32 } %347, 0
  %349 = extractvalue { i32, i32 } %347, 1
  %350 = insertelement <2 x i32> undef, i32 %348, i32 0
  %351 = insertelement <2 x i32> %350, i32 %349, i32 1
  %352 = ashr i32 %129, 31
  %353 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %41, i32 %42)
  %354 = extractvalue { i32, i32 } %353, 0
  %355 = extractvalue { i32, i32 } %353, 1
  %356 = insertelement <2 x i32> undef, i32 %354, i32 0
  %357 = insertelement <2 x i32> %356, i32 %355, i32 1
  %358 = ashr i32 %133, 31
  %359 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %133, i32 %358, i32 %44, i32 %45)
  %360 = extractvalue { i32, i32 } %359, 0
  %361 = extractvalue { i32, i32 } %359, 1
  %362 = insertelement <2 x i32> undef, i32 %360, i32 0
  %363 = insertelement <2 x i32> %362, i32 %361, i32 1
  %364 = ashr i32 %146, 31
  %365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %146, i32 %364, i32 %44, i32 %45)
  %366 = extractvalue { i32, i32 } %365, 0
  %367 = extractvalue { i32, i32 } %365, 1
  %368 = insertelement <2 x i32> undef, i32 %366, i32 0
  %369 = insertelement <2 x i32> %368, i32 %367, i32 1
  %370 = ashr i32 %159, 31
  %371 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %159, i32 %370, i32 %44, i32 %45)
  %372 = extractvalue { i32, i32 } %371, 0
  %373 = extractvalue { i32, i32 } %371, 1
  %374 = insertelement <2 x i32> undef, i32 %372, i32 0
  %375 = insertelement <2 x i32> %374, i32 %373, i32 1
  %376 = ashr i32 %172, 31
  %377 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %172, i32 %376, i32 %44, i32 %45)
  %378 = extractvalue { i32, i32 } %377, 0
  %379 = extractvalue { i32, i32 } %377, 1
  %380 = insertelement <2 x i32> undef, i32 %378, i32 0
  %381 = insertelement <2 x i32> %380, i32 %379, i32 1
  %382 = ashr i32 %185, 31
  %383 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %185, i32 %382, i32 %44, i32 %45)
  %384 = extractvalue { i32, i32 } %383, 0
  %385 = extractvalue { i32, i32 } %383, 1
  %386 = insertelement <2 x i32> undef, i32 %384, i32 0
  %387 = insertelement <2 x i32> %386, i32 %385, i32 1
  %388 = ashr i32 %198, 31
  %389 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %198, i32 %388, i32 %44, i32 %45)
  %390 = extractvalue { i32, i32 } %389, 0
  %391 = extractvalue { i32, i32 } %389, 1
  %392 = insertelement <2 x i32> undef, i32 %390, i32 0
  %393 = insertelement <2 x i32> %392, i32 %391, i32 1
  %394 = ashr i32 %211, 31
  %395 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %211, i32 %394, i32 %44, i32 %45)
  %396 = extractvalue { i32, i32 } %395, 0
  %397 = extractvalue { i32, i32 } %395, 1
  %398 = insertelement <2 x i32> undef, i32 %396, i32 0
  %399 = insertelement <2 x i32> %398, i32 %397, i32 1
  %400 = ashr i32 %224, 31
  %401 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %224, i32 %400, i32 %44, i32 %45)
  %402 = extractvalue { i32, i32 } %401, 0
  %403 = extractvalue { i32, i32 } %401, 1
  %404 = insertelement <2 x i32> undef, i32 %402, i32 0
  %405 = insertelement <2 x i32> %404, i32 %403, i32 1
  %406 = ashr i32 %237, 31
  %407 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %237, i32 %406, i32 %44, i32 %45)
  %408 = extractvalue { i32, i32 } %407, 0
  %409 = extractvalue { i32, i32 } %407, 1
  %410 = insertelement <2 x i32> undef, i32 %408, i32 0
  %411 = insertelement <2 x i32> %410, i32 %409, i32 1
  %412 = ashr i32 %250, 31
  %413 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %250, i32 %412, i32 %44, i32 %45)
  %414 = extractvalue { i32, i32 } %413, 0
  %415 = extractvalue { i32, i32 } %413, 1
  %416 = insertelement <2 x i32> undef, i32 %414, i32 0
  %417 = insertelement <2 x i32> %416, i32 %415, i32 1
  %418 = ashr i32 %263, 31
  %419 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %263, i32 %418, i32 %44, i32 %45)
  %420 = extractvalue { i32, i32 } %419, 0
  %421 = extractvalue { i32, i32 } %419, 1
  %422 = insertelement <2 x i32> undef, i32 %420, i32 0
  %423 = insertelement <2 x i32> %422, i32 %421, i32 1
  %424 = ashr i32 %276, 31
  %425 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %276, i32 %424, i32 %44, i32 %45)
  %426 = extractvalue { i32, i32 } %425, 0
  %427 = extractvalue { i32, i32 } %425, 1
  %428 = insertelement <2 x i32> undef, i32 %426, i32 0
  %429 = insertelement <2 x i32> %428, i32 %427, i32 1
  %430 = ashr i32 %289, 31
  %431 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %289, i32 %430, i32 %44, i32 %45)
  %432 = extractvalue { i32, i32 } %431, 0
  %433 = extractvalue { i32, i32 } %431, 1
  %434 = insertelement <2 x i32> undef, i32 %432, i32 0
  %435 = insertelement <2 x i32> %434, i32 %433, i32 1
  %436 = ashr i32 %302, 31
  %437 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %302, i32 %436, i32 %44, i32 %45)
  %438 = extractvalue { i32, i32 } %437, 0
  %439 = extractvalue { i32, i32 } %437, 1
  %440 = insertelement <2 x i32> undef, i32 %438, i32 0
  %441 = insertelement <2 x i32> %440, i32 %439, i32 1
  %442 = ashr i32 %315, 31
  %443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %315, i32 %442, i32 %44, i32 %45)
  %444 = extractvalue { i32, i32 } %443, 0
  %445 = extractvalue { i32, i32 } %443, 1
  %446 = insertelement <2 x i32> undef, i32 %444, i32 0
  %447 = insertelement <2 x i32> %446, i32 %445, i32 1
  %448 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %50, i32 %51)
  %449 = extractvalue { i32, i32 } %448, 0
  %450 = extractvalue { i32, i32 } %448, 1
  %451 = insertelement <2 x i32> undef, i32 %449, i32 0
  %452 = insertelement <2 x i32> %451, i32 %450, i32 1
  %453 = bitcast <2 x i32> %452 to i64
  %454 = sext i32 %64 to i64
  %455 = add nsw i64 %453, %454
  %456 = shl i64 %455, 2
  %457 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %328, i32 %47, i32 %48)
  %458 = extractvalue { i32, i32 } %457, 0
  %459 = extractvalue { i32, i32 } %457, 1
  %460 = insertelement <2 x i32> undef, i32 %458, i32 0
  %461 = insertelement <2 x i32> %460, i32 %459, i32 1
  %462 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %50, i32 %51)
  %463 = extractvalue { i32, i32 } %462, 0
  %464 = extractvalue { i32, i32 } %462, 1
  %465 = insertelement <2 x i32> undef, i32 %463, i32 0
  %466 = insertelement <2 x i32> %465, i32 %464, i32 1
  %467 = bitcast <2 x i32> %466 to i64
  %468 = add nsw i64 %467, %454
  %469 = shl i64 %468, 2
  %470 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %340, i32 %47, i32 %48)
  %471 = extractvalue { i32, i32 } %470, 0
  %472 = extractvalue { i32, i32 } %470, 1
  %473 = insertelement <2 x i32> undef, i32 %471, i32 0
  %474 = insertelement <2 x i32> %473, i32 %472, i32 1
  %475 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %50, i32 %51)
  %476 = extractvalue { i32, i32 } %475, 0
  %477 = extractvalue { i32, i32 } %475, 1
  %478 = insertelement <2 x i32> undef, i32 %476, i32 0
  %479 = insertelement <2 x i32> %478, i32 %477, i32 1
  %480 = bitcast <2 x i32> %479 to i64
  %481 = add nsw i64 %480, %454
  %482 = shl i64 %481, 2
  %483 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %346, i32 %47, i32 %48)
  %484 = extractvalue { i32, i32 } %483, 0
  %485 = extractvalue { i32, i32 } %483, 1
  %486 = insertelement <2 x i32> undef, i32 %484, i32 0
  %487 = insertelement <2 x i32> %486, i32 %485, i32 1
  %488 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %50, i32 %51)
  %489 = extractvalue { i32, i32 } %488, 0
  %490 = extractvalue { i32, i32 } %488, 1
  %491 = insertelement <2 x i32> undef, i32 %489, i32 0
  %492 = insertelement <2 x i32> %491, i32 %490, i32 1
  %493 = bitcast <2 x i32> %492 to i64
  %494 = add nsw i64 %493, %454
  %495 = shl i64 %494, 2
  %496 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %352, i32 %47, i32 %48)
  %497 = extractvalue { i32, i32 } %496, 0
  %498 = extractvalue { i32, i32 } %496, 1
  %499 = insertelement <2 x i32> undef, i32 %497, i32 0
  %500 = insertelement <2 x i32> %499, i32 %498, i32 1
  %501 = sext i32 %133 to i64
  %502 = add nsw i64 %453, %501
  %503 = shl i64 %502, 2
  %504 = add nsw i64 %467, %501
  %505 = shl i64 %504, 2
  %506 = add nsw i64 %480, %501
  %507 = shl i64 %506, 2
  %508 = add nsw i64 %493, %501
  %509 = shl i64 %508, 2
  %510 = sext i32 %146 to i64
  %511 = add nsw i64 %453, %510
  %512 = shl i64 %511, 2
  %513 = add nsw i64 %467, %510
  %514 = shl i64 %513, 2
  %515 = add nsw i64 %480, %510
  %516 = shl i64 %515, 2
  %517 = add nsw i64 %493, %510
  %518 = shl i64 %517, 2
  %519 = sext i32 %159 to i64
  %520 = add nsw i64 %453, %519
  %521 = shl i64 %520, 2
  %522 = add nsw i64 %467, %519
  %523 = shl i64 %522, 2
  %524 = add nsw i64 %480, %519
  %525 = shl i64 %524, 2
  %526 = add nsw i64 %493, %519
  %527 = shl i64 %526, 2
  %528 = sext i32 %172 to i64
  %529 = add nsw i64 %453, %528
  %530 = shl i64 %529, 2
  %531 = add nsw i64 %467, %528
  %532 = shl i64 %531, 2
  %533 = add nsw i64 %480, %528
  %534 = shl i64 %533, 2
  %535 = add nsw i64 %493, %528
  %536 = shl i64 %535, 2
  %537 = sext i32 %185 to i64
  %538 = add nsw i64 %453, %537
  %539 = shl i64 %538, 2
  %540 = add nsw i64 %467, %537
  %541 = shl i64 %540, 2
  %542 = add nsw i64 %480, %537
  %543 = shl i64 %542, 2
  %544 = add nsw i64 %493, %537
  %545 = shl i64 %544, 2
  %546 = sext i32 %198 to i64
  %547 = add nsw i64 %453, %546
  %548 = shl i64 %547, 2
  %549 = add nsw i64 %467, %546
  %550 = shl i64 %549, 2
  %551 = add nsw i64 %480, %546
  %552 = shl i64 %551, 2
  %553 = add nsw i64 %493, %546
  %554 = shl i64 %553, 2
  %555 = sext i32 %211 to i64
  %556 = add nsw i64 %453, %555
  %557 = shl i64 %556, 2
  %558 = add nsw i64 %467, %555
  %559 = shl i64 %558, 2
  %560 = add nsw i64 %480, %555
  %561 = shl i64 %560, 2
  %562 = add nsw i64 %493, %555
  %563 = shl i64 %562, 2
  %564 = sext i32 %224 to i64
  %565 = add nsw i64 %453, %564
  %566 = shl i64 %565, 2
  %567 = add nsw i64 %467, %564
  %568 = shl i64 %567, 2
  %569 = add nsw i64 %480, %564
  %570 = shl i64 %569, 2
  %571 = add nsw i64 %493, %564
  %572 = shl i64 %571, 2
  %573 = sext i32 %237 to i64
  %574 = add nsw i64 %453, %573
  %575 = shl i64 %574, 2
  %576 = add nsw i64 %467, %573
  %577 = shl i64 %576, 2
  %578 = add nsw i64 %480, %573
  %579 = shl i64 %578, 2
  %580 = add nsw i64 %493, %573
  %581 = shl i64 %580, 2
  %582 = sext i32 %250 to i64
  %583 = add nsw i64 %453, %582
  %584 = shl i64 %583, 2
  %585 = add nsw i64 %467, %582
  %586 = shl i64 %585, 2
  %587 = add nsw i64 %480, %582
  %588 = shl i64 %587, 2
  %589 = add nsw i64 %493, %582
  %590 = shl i64 %589, 2
  %591 = sext i32 %263 to i64
  %592 = add nsw i64 %453, %591
  %593 = shl i64 %592, 2
  %594 = add nsw i64 %467, %591
  %595 = shl i64 %594, 2
  %596 = add nsw i64 %480, %591
  %597 = shl i64 %596, 2
  %598 = add nsw i64 %493, %591
  %599 = shl i64 %598, 2
  %600 = sext i32 %276 to i64
  %601 = add nsw i64 %453, %600
  %602 = shl i64 %601, 2
  %603 = add nsw i64 %467, %600
  %604 = shl i64 %603, 2
  %605 = add nsw i64 %480, %600
  %606 = shl i64 %605, 2
  %607 = add nsw i64 %493, %600
  %608 = shl i64 %607, 2
  %609 = sext i32 %289 to i64
  %610 = add nsw i64 %453, %609
  %611 = shl i64 %610, 2
  %612 = add nsw i64 %467, %609
  %613 = shl i64 %612, 2
  %614 = add nsw i64 %480, %609
  %615 = shl i64 %614, 2
  %616 = add nsw i64 %493, %609
  %617 = shl i64 %616, 2
  %618 = sext i32 %302 to i64
  %619 = add nsw i64 %453, %618
  %620 = shl i64 %619, 2
  %621 = add nsw i64 %467, %618
  %622 = shl i64 %621, 2
  %623 = add nsw i64 %480, %618
  %624 = shl i64 %623, 2
  %625 = add nsw i64 %493, %618
  %626 = shl i64 %625, 2
  %627 = sext i32 %315 to i64
  %628 = add nsw i64 %453, %627
  %629 = shl i64 %628, 2
  %630 = add nsw i64 %467, %627
  %631 = shl i64 %630, 2
  %632 = add nsw i64 %480, %627
  %633 = shl i64 %632, 2
  %634 = add nsw i64 %493, %627
  %635 = shl i64 %634, 2
  %636 = shl i64 %99, 1
  %637 = shl i64 %105, 1
  %.op3824 = shl i64 %111, 2
  %638 = bitcast i64 %.op3824 to <2 x i32>
  %639 = extractelement <2 x i32> %638, i32 0
  %640 = extractelement <2 x i32> %638, i32 1
  %641 = select i1 %81, i32 %639, i32 0
  %642 = select i1 %81, i32 %640, i32 0
  %643 = insertelement <2 x i32> undef, i32 %641, i32 0
  %644 = insertelement <2 x i32> %643, i32 %642, i32 1
  %645 = shl i64 %117, 2
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader2.preheader:                            ; preds = %.preheader1.15..preheader2.preheader_crit_edge, %.lr.ph
  %646 = phi i32 [ %26, %.lr.ph ], [ %2890, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %92, %.lr.ph ], [ %2895, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3821 = phi i64 [ %87, %.lr.ph ], [ %2894, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3822 = phi i64 [ %74, %.lr.ph ], [ %2893, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3823 = phi i64 [ %69, %.lr.ph ], [ %2892, %.preheader1.15..preheader2.preheader_crit_edge ]
  br i1 %93, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
  br label %.preheader1.preheader, !stats.blockFrequency.digits !917, !stats.blockFrequency.scale !879

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
  br label %.preheader.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader.preheader:                             ; preds = %.preheader.15..preheader.preheader_crit_edge, %.preheader.preheader.preheader
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
  %sink_sink_3888 = bitcast <2 x i32> %333 to i64
  %sink_sink_3864 = shl i64 %sink_sink_3888, 1
  %sink_3908 = add i64 %.in3823, %sink_sink_3864
  %sink_sink_3887 = bitcast <2 x i32> %339 to i64
  %sink_sink_3863 = shl i64 %sink_sink_3887, 1
  %sink_3907 = add i64 %.in3822, %sink_sink_3863
  br i1 %120, label %648, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

648:                                              ; preds = %.preheader.preheader
  %.sroa.256.0.insert.ext = zext i32 %647 to i64
  %649 = shl nuw nsw i64 %.sroa.256.0.insert.ext, 1
  %650 = add i64 %sink_3908, %649
  %651 = inttoptr i64 %650 to i16 addrspace(4)*
  %652 = addrspacecast i16 addrspace(4)* %651 to i16 addrspace(1)*
  %653 = load i16, i16 addrspace(1)* %652, align 2
  %654 = add i64 %sink_3907, %649
  %655 = inttoptr i64 %654 to i16 addrspace(4)*
  %656 = addrspacecast i16 addrspace(4)* %655 to i16 addrspace(1)*
  %657 = load i16, i16 addrspace(1)* %656, align 2
  %658 = zext i16 %653 to i32
  %659 = shl nuw i32 %658, 16, !spirv.Decorations !921
  %660 = bitcast i32 %659 to float
  %661 = zext i16 %657 to i32
  %662 = shl nuw i32 %661, 16, !spirv.Decorations !921
  %663 = bitcast i32 %662 to float
  %664 = fmul reassoc nsz arcp contract float %660, %663, !spirv.Decorations !898
  %665 = fadd reassoc nsz arcp contract float %664, %.sroa.0.1, !spirv.Decorations !898
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %648
  %.sroa.0.2 = phi float [ %665, %648 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  %sink_sink_3886 = bitcast <2 x i32> %345 to i64
  %sink_sink_3862 = shl i64 %sink_sink_3886, 1
  %sink_3906 = add i64 %.in3823, %sink_sink_3862
  br i1 %124, label %666, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

666:                                              ; preds = %._crit_edge
  %.sroa.256.0.insert.ext588 = zext i32 %647 to i64
  %667 = shl nuw nsw i64 %.sroa.256.0.insert.ext588, 1
  %668 = add i64 %sink_3906, %667
  %669 = inttoptr i64 %668 to i16 addrspace(4)*
  %670 = addrspacecast i16 addrspace(4)* %669 to i16 addrspace(1)*
  %671 = load i16, i16 addrspace(1)* %670, align 2
  %672 = add i64 %sink_3907, %667
  %673 = inttoptr i64 %672 to i16 addrspace(4)*
  %674 = addrspacecast i16 addrspace(4)* %673 to i16 addrspace(1)*
  %675 = load i16, i16 addrspace(1)* %674, align 2
  %676 = zext i16 %671 to i32
  %677 = shl nuw i32 %676, 16, !spirv.Decorations !921
  %678 = bitcast i32 %677 to float
  %679 = zext i16 %675 to i32
  %680 = shl nuw i32 %679, 16, !spirv.Decorations !921
  %681 = bitcast i32 %680 to float
  %682 = fmul reassoc nsz arcp contract float %678, %681, !spirv.Decorations !898
  %683 = fadd reassoc nsz arcp contract float %682, %.sroa.66.1, !spirv.Decorations !898
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %666
  %.sroa.66.2 = phi float [ %683, %666 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  %sink_sink_3885 = bitcast <2 x i32> %351 to i64
  %sink_sink_3861 = shl i64 %sink_sink_3885, 1
  %sink_3905 = add i64 %.in3823, %sink_sink_3861
  br i1 %128, label %684, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

684:                                              ; preds = %._crit_edge.1
  %.sroa.256.0.insert.ext593 = zext i32 %647 to i64
  %685 = shl nuw nsw i64 %.sroa.256.0.insert.ext593, 1
  %686 = add i64 %sink_3905, %685
  %687 = inttoptr i64 %686 to i16 addrspace(4)*
  %688 = addrspacecast i16 addrspace(4)* %687 to i16 addrspace(1)*
  %689 = load i16, i16 addrspace(1)* %688, align 2
  %690 = add i64 %sink_3907, %685
  %691 = inttoptr i64 %690 to i16 addrspace(4)*
  %692 = addrspacecast i16 addrspace(4)* %691 to i16 addrspace(1)*
  %693 = load i16, i16 addrspace(1)* %692, align 2
  %694 = zext i16 %689 to i32
  %695 = shl nuw i32 %694, 16, !spirv.Decorations !921
  %696 = bitcast i32 %695 to float
  %697 = zext i16 %693 to i32
  %698 = shl nuw i32 %697, 16, !spirv.Decorations !921
  %699 = bitcast i32 %698 to float
  %700 = fmul reassoc nsz arcp contract float %696, %699, !spirv.Decorations !898
  %701 = fadd reassoc nsz arcp contract float %700, %.sroa.130.1, !spirv.Decorations !898
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %684
  %.sroa.130.2 = phi float [ %701, %684 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  %sink_sink_3884 = bitcast <2 x i32> %357 to i64
  %sink_sink_3860 = shl i64 %sink_sink_3884, 1
  %sink_3904 = add i64 %.in3823, %sink_sink_3860
  br i1 %132, label %702, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

702:                                              ; preds = %._crit_edge.2
  %.sroa.256.0.insert.ext598 = zext i32 %647 to i64
  %703 = shl nuw nsw i64 %.sroa.256.0.insert.ext598, 1
  %704 = add i64 %sink_3904, %703
  %705 = inttoptr i64 %704 to i16 addrspace(4)*
  %706 = addrspacecast i16 addrspace(4)* %705 to i16 addrspace(1)*
  %707 = load i16, i16 addrspace(1)* %706, align 2
  %708 = add i64 %sink_3907, %703
  %709 = inttoptr i64 %708 to i16 addrspace(4)*
  %710 = addrspacecast i16 addrspace(4)* %709 to i16 addrspace(1)*
  %711 = load i16, i16 addrspace(1)* %710, align 2
  %712 = zext i16 %707 to i32
  %713 = shl nuw i32 %712, 16, !spirv.Decorations !921
  %714 = bitcast i32 %713 to float
  %715 = zext i16 %711 to i32
  %716 = shl nuw i32 %715, 16, !spirv.Decorations !921
  %717 = bitcast i32 %716 to float
  %718 = fmul reassoc nsz arcp contract float %714, %717, !spirv.Decorations !898
  %719 = fadd reassoc nsz arcp contract float %718, %.sroa.194.1, !spirv.Decorations !898
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %702
  %.sroa.194.2 = phi float [ %719, %702 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %sink_sink_3883 = bitcast <2 x i32> %363 to i64
  %sink_sink_3859 = shl i64 %sink_sink_3883, 1
  %sink_3903 = add i64 %.in3822, %sink_sink_3859
  br i1 %136, label %720, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

720:                                              ; preds = %.preheader
  %.sroa.256.0.insert.ext603 = zext i32 %647 to i64
  %721 = shl nuw nsw i64 %.sroa.256.0.insert.ext603, 1
  %722 = add i64 %sink_3908, %721
  %723 = inttoptr i64 %722 to i16 addrspace(4)*
  %724 = addrspacecast i16 addrspace(4)* %723 to i16 addrspace(1)*
  %725 = load i16, i16 addrspace(1)* %724, align 2
  %726 = add i64 %sink_3903, %721
  %727 = inttoptr i64 %726 to i16 addrspace(4)*
  %728 = addrspacecast i16 addrspace(4)* %727 to i16 addrspace(1)*
  %729 = load i16, i16 addrspace(1)* %728, align 2
  %730 = zext i16 %725 to i32
  %731 = shl nuw i32 %730, 16, !spirv.Decorations !921
  %732 = bitcast i32 %731 to float
  %733 = zext i16 %729 to i32
  %734 = shl nuw i32 %733, 16, !spirv.Decorations !921
  %735 = bitcast i32 %734 to float
  %736 = fmul reassoc nsz arcp contract float %732, %735, !spirv.Decorations !898
  %737 = fadd reassoc nsz arcp contract float %736, %.sroa.6.1, !spirv.Decorations !898
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %720
  %.sroa.6.2 = phi float [ %737, %720 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %139, label %738, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

738:                                              ; preds = %._crit_edge.173
  %.sroa.256.0.insert.ext608 = zext i32 %647 to i64
  %739 = shl nuw nsw i64 %.sroa.256.0.insert.ext608, 1
  %740 = add i64 %sink_3906, %739
  %741 = inttoptr i64 %740 to i16 addrspace(4)*
  %742 = addrspacecast i16 addrspace(4)* %741 to i16 addrspace(1)*
  %743 = load i16, i16 addrspace(1)* %742, align 2
  %744 = add i64 %sink_3903, %739
  %745 = inttoptr i64 %744 to i16 addrspace(4)*
  %746 = addrspacecast i16 addrspace(4)* %745 to i16 addrspace(1)*
  %747 = load i16, i16 addrspace(1)* %746, align 2
  %748 = zext i16 %743 to i32
  %749 = shl nuw i32 %748, 16, !spirv.Decorations !921
  %750 = bitcast i32 %749 to float
  %751 = zext i16 %747 to i32
  %752 = shl nuw i32 %751, 16, !spirv.Decorations !921
  %753 = bitcast i32 %752 to float
  %754 = fmul reassoc nsz arcp contract float %750, %753, !spirv.Decorations !898
  %755 = fadd reassoc nsz arcp contract float %754, %.sroa.70.1, !spirv.Decorations !898
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %738
  %.sroa.70.2 = phi float [ %755, %738 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %142, label %756, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

756:                                              ; preds = %._crit_edge.1.1
  %.sroa.256.0.insert.ext613 = zext i32 %647 to i64
  %757 = shl nuw nsw i64 %.sroa.256.0.insert.ext613, 1
  %758 = add i64 %sink_3905, %757
  %759 = inttoptr i64 %758 to i16 addrspace(4)*
  %760 = addrspacecast i16 addrspace(4)* %759 to i16 addrspace(1)*
  %761 = load i16, i16 addrspace(1)* %760, align 2
  %762 = add i64 %sink_3903, %757
  %763 = inttoptr i64 %762 to i16 addrspace(4)*
  %764 = addrspacecast i16 addrspace(4)* %763 to i16 addrspace(1)*
  %765 = load i16, i16 addrspace(1)* %764, align 2
  %766 = zext i16 %761 to i32
  %767 = shl nuw i32 %766, 16, !spirv.Decorations !921
  %768 = bitcast i32 %767 to float
  %769 = zext i16 %765 to i32
  %770 = shl nuw i32 %769, 16, !spirv.Decorations !921
  %771 = bitcast i32 %770 to float
  %772 = fmul reassoc nsz arcp contract float %768, %771, !spirv.Decorations !898
  %773 = fadd reassoc nsz arcp contract float %772, %.sroa.134.1, !spirv.Decorations !898
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %756
  %.sroa.134.2 = phi float [ %773, %756 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %145, label %774, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

774:                                              ; preds = %._crit_edge.2.1
  %.sroa.256.0.insert.ext618 = zext i32 %647 to i64
  %775 = shl nuw nsw i64 %.sroa.256.0.insert.ext618, 1
  %776 = add i64 %sink_3904, %775
  %777 = inttoptr i64 %776 to i16 addrspace(4)*
  %778 = addrspacecast i16 addrspace(4)* %777 to i16 addrspace(1)*
  %779 = load i16, i16 addrspace(1)* %778, align 2
  %780 = add i64 %sink_3903, %775
  %781 = inttoptr i64 %780 to i16 addrspace(4)*
  %782 = addrspacecast i16 addrspace(4)* %781 to i16 addrspace(1)*
  %783 = load i16, i16 addrspace(1)* %782, align 2
  %784 = zext i16 %779 to i32
  %785 = shl nuw i32 %784, 16, !spirv.Decorations !921
  %786 = bitcast i32 %785 to float
  %787 = zext i16 %783 to i32
  %788 = shl nuw i32 %787, 16, !spirv.Decorations !921
  %789 = bitcast i32 %788 to float
  %790 = fmul reassoc nsz arcp contract float %786, %789, !spirv.Decorations !898
  %791 = fadd reassoc nsz arcp contract float %790, %.sroa.198.1, !spirv.Decorations !898
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %774
  %.sroa.198.2 = phi float [ %791, %774 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %sink_sink_3882 = bitcast <2 x i32> %369 to i64
  %sink_sink_3858 = shl i64 %sink_sink_3882, 1
  %sink_3902 = add i64 %.in3822, %sink_sink_3858
  br i1 %149, label %792, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

792:                                              ; preds = %.preheader.1
  %.sroa.256.0.insert.ext623 = zext i32 %647 to i64
  %793 = shl nuw nsw i64 %.sroa.256.0.insert.ext623, 1
  %794 = add i64 %sink_3908, %793
  %795 = inttoptr i64 %794 to i16 addrspace(4)*
  %796 = addrspacecast i16 addrspace(4)* %795 to i16 addrspace(1)*
  %797 = load i16, i16 addrspace(1)* %796, align 2
  %798 = add i64 %sink_3902, %793
  %799 = inttoptr i64 %798 to i16 addrspace(4)*
  %800 = addrspacecast i16 addrspace(4)* %799 to i16 addrspace(1)*
  %801 = load i16, i16 addrspace(1)* %800, align 2
  %802 = zext i16 %797 to i32
  %803 = shl nuw i32 %802, 16, !spirv.Decorations !921
  %804 = bitcast i32 %803 to float
  %805 = zext i16 %801 to i32
  %806 = shl nuw i32 %805, 16, !spirv.Decorations !921
  %807 = bitcast i32 %806 to float
  %808 = fmul reassoc nsz arcp contract float %804, %807, !spirv.Decorations !898
  %809 = fadd reassoc nsz arcp contract float %808, %.sroa.10.1, !spirv.Decorations !898
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %792
  %.sroa.10.2 = phi float [ %809, %792 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %152, label %810, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

810:                                              ; preds = %._crit_edge.274
  %.sroa.256.0.insert.ext628 = zext i32 %647 to i64
  %811 = shl nuw nsw i64 %.sroa.256.0.insert.ext628, 1
  %812 = add i64 %sink_3906, %811
  %813 = inttoptr i64 %812 to i16 addrspace(4)*
  %814 = addrspacecast i16 addrspace(4)* %813 to i16 addrspace(1)*
  %815 = load i16, i16 addrspace(1)* %814, align 2
  %816 = add i64 %sink_3902, %811
  %817 = inttoptr i64 %816 to i16 addrspace(4)*
  %818 = addrspacecast i16 addrspace(4)* %817 to i16 addrspace(1)*
  %819 = load i16, i16 addrspace(1)* %818, align 2
  %820 = zext i16 %815 to i32
  %821 = shl nuw i32 %820, 16, !spirv.Decorations !921
  %822 = bitcast i32 %821 to float
  %823 = zext i16 %819 to i32
  %824 = shl nuw i32 %823, 16, !spirv.Decorations !921
  %825 = bitcast i32 %824 to float
  %826 = fmul reassoc nsz arcp contract float %822, %825, !spirv.Decorations !898
  %827 = fadd reassoc nsz arcp contract float %826, %.sroa.74.1, !spirv.Decorations !898
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %810
  %.sroa.74.2 = phi float [ %827, %810 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %155, label %828, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

828:                                              ; preds = %._crit_edge.1.2
  %.sroa.256.0.insert.ext633 = zext i32 %647 to i64
  %829 = shl nuw nsw i64 %.sroa.256.0.insert.ext633, 1
  %830 = add i64 %sink_3905, %829
  %831 = inttoptr i64 %830 to i16 addrspace(4)*
  %832 = addrspacecast i16 addrspace(4)* %831 to i16 addrspace(1)*
  %833 = load i16, i16 addrspace(1)* %832, align 2
  %834 = add i64 %sink_3902, %829
  %835 = inttoptr i64 %834 to i16 addrspace(4)*
  %836 = addrspacecast i16 addrspace(4)* %835 to i16 addrspace(1)*
  %837 = load i16, i16 addrspace(1)* %836, align 2
  %838 = zext i16 %833 to i32
  %839 = shl nuw i32 %838, 16, !spirv.Decorations !921
  %840 = bitcast i32 %839 to float
  %841 = zext i16 %837 to i32
  %842 = shl nuw i32 %841, 16, !spirv.Decorations !921
  %843 = bitcast i32 %842 to float
  %844 = fmul reassoc nsz arcp contract float %840, %843, !spirv.Decorations !898
  %845 = fadd reassoc nsz arcp contract float %844, %.sroa.138.1, !spirv.Decorations !898
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %828
  %.sroa.138.2 = phi float [ %845, %828 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %158, label %846, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

846:                                              ; preds = %._crit_edge.2.2
  %.sroa.256.0.insert.ext638 = zext i32 %647 to i64
  %847 = shl nuw nsw i64 %.sroa.256.0.insert.ext638, 1
  %848 = add i64 %sink_3904, %847
  %849 = inttoptr i64 %848 to i16 addrspace(4)*
  %850 = addrspacecast i16 addrspace(4)* %849 to i16 addrspace(1)*
  %851 = load i16, i16 addrspace(1)* %850, align 2
  %852 = add i64 %sink_3902, %847
  %853 = inttoptr i64 %852 to i16 addrspace(4)*
  %854 = addrspacecast i16 addrspace(4)* %853 to i16 addrspace(1)*
  %855 = load i16, i16 addrspace(1)* %854, align 2
  %856 = zext i16 %851 to i32
  %857 = shl nuw i32 %856, 16, !spirv.Decorations !921
  %858 = bitcast i32 %857 to float
  %859 = zext i16 %855 to i32
  %860 = shl nuw i32 %859, 16, !spirv.Decorations !921
  %861 = bitcast i32 %860 to float
  %862 = fmul reassoc nsz arcp contract float %858, %861, !spirv.Decorations !898
  %863 = fadd reassoc nsz arcp contract float %862, %.sroa.202.1, !spirv.Decorations !898
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %846
  %.sroa.202.2 = phi float [ %863, %846 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %sink_sink_3881 = bitcast <2 x i32> %375 to i64
  %sink_sink_3857 = shl i64 %sink_sink_3881, 1
  %sink_3901 = add i64 %.in3822, %sink_sink_3857
  br i1 %162, label %864, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

864:                                              ; preds = %.preheader.2
  %.sroa.256.0.insert.ext643 = zext i32 %647 to i64
  %865 = shl nuw nsw i64 %.sroa.256.0.insert.ext643, 1
  %866 = add i64 %sink_3908, %865
  %867 = inttoptr i64 %866 to i16 addrspace(4)*
  %868 = addrspacecast i16 addrspace(4)* %867 to i16 addrspace(1)*
  %869 = load i16, i16 addrspace(1)* %868, align 2
  %870 = add i64 %sink_3901, %865
  %871 = inttoptr i64 %870 to i16 addrspace(4)*
  %872 = addrspacecast i16 addrspace(4)* %871 to i16 addrspace(1)*
  %873 = load i16, i16 addrspace(1)* %872, align 2
  %874 = zext i16 %869 to i32
  %875 = shl nuw i32 %874, 16, !spirv.Decorations !921
  %876 = bitcast i32 %875 to float
  %877 = zext i16 %873 to i32
  %878 = shl nuw i32 %877, 16, !spirv.Decorations !921
  %879 = bitcast i32 %878 to float
  %880 = fmul reassoc nsz arcp contract float %876, %879, !spirv.Decorations !898
  %881 = fadd reassoc nsz arcp contract float %880, %.sroa.14.1, !spirv.Decorations !898
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %864
  %.sroa.14.2 = phi float [ %881, %864 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %165, label %882, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

882:                                              ; preds = %._crit_edge.375
  %.sroa.256.0.insert.ext648 = zext i32 %647 to i64
  %883 = shl nuw nsw i64 %.sroa.256.0.insert.ext648, 1
  %884 = add i64 %sink_3906, %883
  %885 = inttoptr i64 %884 to i16 addrspace(4)*
  %886 = addrspacecast i16 addrspace(4)* %885 to i16 addrspace(1)*
  %887 = load i16, i16 addrspace(1)* %886, align 2
  %888 = add i64 %sink_3901, %883
  %889 = inttoptr i64 %888 to i16 addrspace(4)*
  %890 = addrspacecast i16 addrspace(4)* %889 to i16 addrspace(1)*
  %891 = load i16, i16 addrspace(1)* %890, align 2
  %892 = zext i16 %887 to i32
  %893 = shl nuw i32 %892, 16, !spirv.Decorations !921
  %894 = bitcast i32 %893 to float
  %895 = zext i16 %891 to i32
  %896 = shl nuw i32 %895, 16, !spirv.Decorations !921
  %897 = bitcast i32 %896 to float
  %898 = fmul reassoc nsz arcp contract float %894, %897, !spirv.Decorations !898
  %899 = fadd reassoc nsz arcp contract float %898, %.sroa.78.1, !spirv.Decorations !898
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %882
  %.sroa.78.2 = phi float [ %899, %882 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %168, label %900, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

900:                                              ; preds = %._crit_edge.1.3
  %.sroa.256.0.insert.ext653 = zext i32 %647 to i64
  %901 = shl nuw nsw i64 %.sroa.256.0.insert.ext653, 1
  %902 = add i64 %sink_3905, %901
  %903 = inttoptr i64 %902 to i16 addrspace(4)*
  %904 = addrspacecast i16 addrspace(4)* %903 to i16 addrspace(1)*
  %905 = load i16, i16 addrspace(1)* %904, align 2
  %906 = add i64 %sink_3901, %901
  %907 = inttoptr i64 %906 to i16 addrspace(4)*
  %908 = addrspacecast i16 addrspace(4)* %907 to i16 addrspace(1)*
  %909 = load i16, i16 addrspace(1)* %908, align 2
  %910 = zext i16 %905 to i32
  %911 = shl nuw i32 %910, 16, !spirv.Decorations !921
  %912 = bitcast i32 %911 to float
  %913 = zext i16 %909 to i32
  %914 = shl nuw i32 %913, 16, !spirv.Decorations !921
  %915 = bitcast i32 %914 to float
  %916 = fmul reassoc nsz arcp contract float %912, %915, !spirv.Decorations !898
  %917 = fadd reassoc nsz arcp contract float %916, %.sroa.142.1, !spirv.Decorations !898
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %900
  %.sroa.142.2 = phi float [ %917, %900 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %171, label %918, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

918:                                              ; preds = %._crit_edge.2.3
  %.sroa.256.0.insert.ext658 = zext i32 %647 to i64
  %919 = shl nuw nsw i64 %.sroa.256.0.insert.ext658, 1
  %920 = add i64 %sink_3904, %919
  %921 = inttoptr i64 %920 to i16 addrspace(4)*
  %922 = addrspacecast i16 addrspace(4)* %921 to i16 addrspace(1)*
  %923 = load i16, i16 addrspace(1)* %922, align 2
  %924 = add i64 %sink_3901, %919
  %925 = inttoptr i64 %924 to i16 addrspace(4)*
  %926 = addrspacecast i16 addrspace(4)* %925 to i16 addrspace(1)*
  %927 = load i16, i16 addrspace(1)* %926, align 2
  %928 = zext i16 %923 to i32
  %929 = shl nuw i32 %928, 16, !spirv.Decorations !921
  %930 = bitcast i32 %929 to float
  %931 = zext i16 %927 to i32
  %932 = shl nuw i32 %931, 16, !spirv.Decorations !921
  %933 = bitcast i32 %932 to float
  %934 = fmul reassoc nsz arcp contract float %930, %933, !spirv.Decorations !898
  %935 = fadd reassoc nsz arcp contract float %934, %.sroa.206.1, !spirv.Decorations !898
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %918
  %.sroa.206.2 = phi float [ %935, %918 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %sink_sink_3880 = bitcast <2 x i32> %381 to i64
  %sink_sink_3856 = shl i64 %sink_sink_3880, 1
  %sink_3900 = add i64 %.in3822, %sink_sink_3856
  br i1 %175, label %936, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
  br label %._crit_edge.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

936:                                              ; preds = %.preheader.3
  %.sroa.256.0.insert.ext663 = zext i32 %647 to i64
  %937 = shl nuw nsw i64 %.sroa.256.0.insert.ext663, 1
  %938 = add i64 %sink_3908, %937
  %939 = inttoptr i64 %938 to i16 addrspace(4)*
  %940 = addrspacecast i16 addrspace(4)* %939 to i16 addrspace(1)*
  %941 = load i16, i16 addrspace(1)* %940, align 2
  %942 = add i64 %sink_3900, %937
  %943 = inttoptr i64 %942 to i16 addrspace(4)*
  %944 = addrspacecast i16 addrspace(4)* %943 to i16 addrspace(1)*
  %945 = load i16, i16 addrspace(1)* %944, align 2
  %946 = zext i16 %941 to i32
  %947 = shl nuw i32 %946, 16, !spirv.Decorations !921
  %948 = bitcast i32 %947 to float
  %949 = zext i16 %945 to i32
  %950 = shl nuw i32 %949, 16, !spirv.Decorations !921
  %951 = bitcast i32 %950 to float
  %952 = fmul reassoc nsz arcp contract float %948, %951, !spirv.Decorations !898
  %953 = fadd reassoc nsz arcp contract float %952, %.sroa.18.1, !spirv.Decorations !898
  br label %._crit_edge.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %936
  %.sroa.18.2 = phi float [ %953, %936 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  br i1 %178, label %954, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

954:                                              ; preds = %._crit_edge.4
  %.sroa.256.0.insert.ext668 = zext i32 %647 to i64
  %955 = shl nuw nsw i64 %.sroa.256.0.insert.ext668, 1
  %956 = add i64 %sink_3906, %955
  %957 = inttoptr i64 %956 to i16 addrspace(4)*
  %958 = addrspacecast i16 addrspace(4)* %957 to i16 addrspace(1)*
  %959 = load i16, i16 addrspace(1)* %958, align 2
  %960 = add i64 %sink_3900, %955
  %961 = inttoptr i64 %960 to i16 addrspace(4)*
  %962 = addrspacecast i16 addrspace(4)* %961 to i16 addrspace(1)*
  %963 = load i16, i16 addrspace(1)* %962, align 2
  %964 = zext i16 %959 to i32
  %965 = shl nuw i32 %964, 16, !spirv.Decorations !921
  %966 = bitcast i32 %965 to float
  %967 = zext i16 %963 to i32
  %968 = shl nuw i32 %967, 16, !spirv.Decorations !921
  %969 = bitcast i32 %968 to float
  %970 = fmul reassoc nsz arcp contract float %966, %969, !spirv.Decorations !898
  %971 = fadd reassoc nsz arcp contract float %970, %.sroa.82.1, !spirv.Decorations !898
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %954
  %.sroa.82.2 = phi float [ %971, %954 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  br i1 %181, label %972, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

972:                                              ; preds = %._crit_edge.1.4
  %.sroa.256.0.insert.ext673 = zext i32 %647 to i64
  %973 = shl nuw nsw i64 %.sroa.256.0.insert.ext673, 1
  %974 = add i64 %sink_3905, %973
  %975 = inttoptr i64 %974 to i16 addrspace(4)*
  %976 = addrspacecast i16 addrspace(4)* %975 to i16 addrspace(1)*
  %977 = load i16, i16 addrspace(1)* %976, align 2
  %978 = add i64 %sink_3900, %973
  %979 = inttoptr i64 %978 to i16 addrspace(4)*
  %980 = addrspacecast i16 addrspace(4)* %979 to i16 addrspace(1)*
  %981 = load i16, i16 addrspace(1)* %980, align 2
  %982 = zext i16 %977 to i32
  %983 = shl nuw i32 %982, 16, !spirv.Decorations !921
  %984 = bitcast i32 %983 to float
  %985 = zext i16 %981 to i32
  %986 = shl nuw i32 %985, 16, !spirv.Decorations !921
  %987 = bitcast i32 %986 to float
  %988 = fmul reassoc nsz arcp contract float %984, %987, !spirv.Decorations !898
  %989 = fadd reassoc nsz arcp contract float %988, %.sroa.146.1, !spirv.Decorations !898
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %972
  %.sroa.146.2 = phi float [ %989, %972 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  br i1 %184, label %990, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
  br label %.preheader.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

990:                                              ; preds = %._crit_edge.2.4
  %.sroa.256.0.insert.ext678 = zext i32 %647 to i64
  %991 = shl nuw nsw i64 %.sroa.256.0.insert.ext678, 1
  %992 = add i64 %sink_3904, %991
  %993 = inttoptr i64 %992 to i16 addrspace(4)*
  %994 = addrspacecast i16 addrspace(4)* %993 to i16 addrspace(1)*
  %995 = load i16, i16 addrspace(1)* %994, align 2
  %996 = add i64 %sink_3900, %991
  %997 = inttoptr i64 %996 to i16 addrspace(4)*
  %998 = addrspacecast i16 addrspace(4)* %997 to i16 addrspace(1)*
  %999 = load i16, i16 addrspace(1)* %998, align 2
  %1000 = zext i16 %995 to i32
  %1001 = shl nuw i32 %1000, 16, !spirv.Decorations !921
  %1002 = bitcast i32 %1001 to float
  %1003 = zext i16 %999 to i32
  %1004 = shl nuw i32 %1003, 16, !spirv.Decorations !921
  %1005 = bitcast i32 %1004 to float
  %1006 = fmul reassoc nsz arcp contract float %1002, %1005, !spirv.Decorations !898
  %1007 = fadd reassoc nsz arcp contract float %1006, %.sroa.210.1, !spirv.Decorations !898
  br label %.preheader.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %990
  %.sroa.210.2 = phi float [ %1007, %990 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %sink_sink_3879 = bitcast <2 x i32> %387 to i64
  %sink_sink_3855 = shl i64 %sink_sink_3879, 1
  %sink_3899 = add i64 %.in3822, %sink_sink_3855
  br i1 %188, label %1008, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
  br label %._crit_edge.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1008:                                             ; preds = %.preheader.4
  %.sroa.256.0.insert.ext683 = zext i32 %647 to i64
  %1009 = shl nuw nsw i64 %.sroa.256.0.insert.ext683, 1
  %1010 = add i64 %sink_3908, %1009
  %1011 = inttoptr i64 %1010 to i16 addrspace(4)*
  %1012 = addrspacecast i16 addrspace(4)* %1011 to i16 addrspace(1)*
  %1013 = load i16, i16 addrspace(1)* %1012, align 2
  %1014 = add i64 %sink_3899, %1009
  %1015 = inttoptr i64 %1014 to i16 addrspace(4)*
  %1016 = addrspacecast i16 addrspace(4)* %1015 to i16 addrspace(1)*
  %1017 = load i16, i16 addrspace(1)* %1016, align 2
  %1018 = zext i16 %1013 to i32
  %1019 = shl nuw i32 %1018, 16, !spirv.Decorations !921
  %1020 = bitcast i32 %1019 to float
  %1021 = zext i16 %1017 to i32
  %1022 = shl nuw i32 %1021, 16, !spirv.Decorations !921
  %1023 = bitcast i32 %1022 to float
  %1024 = fmul reassoc nsz arcp contract float %1020, %1023, !spirv.Decorations !898
  %1025 = fadd reassoc nsz arcp contract float %1024, %.sroa.22.1, !spirv.Decorations !898
  br label %._crit_edge.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %1008
  %.sroa.22.2 = phi float [ %1025, %1008 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  br i1 %191, label %1026, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1026:                                             ; preds = %._crit_edge.5
  %.sroa.256.0.insert.ext688 = zext i32 %647 to i64
  %1027 = shl nuw nsw i64 %.sroa.256.0.insert.ext688, 1
  %1028 = add i64 %sink_3906, %1027
  %1029 = inttoptr i64 %1028 to i16 addrspace(4)*
  %1030 = addrspacecast i16 addrspace(4)* %1029 to i16 addrspace(1)*
  %1031 = load i16, i16 addrspace(1)* %1030, align 2
  %1032 = add i64 %sink_3899, %1027
  %1033 = inttoptr i64 %1032 to i16 addrspace(4)*
  %1034 = addrspacecast i16 addrspace(4)* %1033 to i16 addrspace(1)*
  %1035 = load i16, i16 addrspace(1)* %1034, align 2
  %1036 = zext i16 %1031 to i32
  %1037 = shl nuw i32 %1036, 16, !spirv.Decorations !921
  %1038 = bitcast i32 %1037 to float
  %1039 = zext i16 %1035 to i32
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !921
  %1041 = bitcast i32 %1040 to float
  %1042 = fmul reassoc nsz arcp contract float %1038, %1041, !spirv.Decorations !898
  %1043 = fadd reassoc nsz arcp contract float %1042, %.sroa.86.1, !spirv.Decorations !898
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %1026
  %.sroa.86.2 = phi float [ %1043, %1026 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  br i1 %194, label %1044, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1044:                                             ; preds = %._crit_edge.1.5
  %.sroa.256.0.insert.ext693 = zext i32 %647 to i64
  %1045 = shl nuw nsw i64 %.sroa.256.0.insert.ext693, 1
  %1046 = add i64 %sink_3905, %1045
  %1047 = inttoptr i64 %1046 to i16 addrspace(4)*
  %1048 = addrspacecast i16 addrspace(4)* %1047 to i16 addrspace(1)*
  %1049 = load i16, i16 addrspace(1)* %1048, align 2
  %1050 = add i64 %sink_3899, %1045
  %1051 = inttoptr i64 %1050 to i16 addrspace(4)*
  %1052 = addrspacecast i16 addrspace(4)* %1051 to i16 addrspace(1)*
  %1053 = load i16, i16 addrspace(1)* %1052, align 2
  %1054 = zext i16 %1049 to i32
  %1055 = shl nuw i32 %1054, 16, !spirv.Decorations !921
  %1056 = bitcast i32 %1055 to float
  %1057 = zext i16 %1053 to i32
  %1058 = shl nuw i32 %1057, 16, !spirv.Decorations !921
  %1059 = bitcast i32 %1058 to float
  %1060 = fmul reassoc nsz arcp contract float %1056, %1059, !spirv.Decorations !898
  %1061 = fadd reassoc nsz arcp contract float %1060, %.sroa.150.1, !spirv.Decorations !898
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %1044
  %.sroa.150.2 = phi float [ %1061, %1044 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  br i1 %197, label %1062, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
  br label %.preheader.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1062:                                             ; preds = %._crit_edge.2.5
  %.sroa.256.0.insert.ext698 = zext i32 %647 to i64
  %1063 = shl nuw nsw i64 %.sroa.256.0.insert.ext698, 1
  %1064 = add i64 %sink_3904, %1063
  %1065 = inttoptr i64 %1064 to i16 addrspace(4)*
  %1066 = addrspacecast i16 addrspace(4)* %1065 to i16 addrspace(1)*
  %1067 = load i16, i16 addrspace(1)* %1066, align 2
  %1068 = add i64 %sink_3899, %1063
  %1069 = inttoptr i64 %1068 to i16 addrspace(4)*
  %1070 = addrspacecast i16 addrspace(4)* %1069 to i16 addrspace(1)*
  %1071 = load i16, i16 addrspace(1)* %1070, align 2
  %1072 = zext i16 %1067 to i32
  %1073 = shl nuw i32 %1072, 16, !spirv.Decorations !921
  %1074 = bitcast i32 %1073 to float
  %1075 = zext i16 %1071 to i32
  %1076 = shl nuw i32 %1075, 16, !spirv.Decorations !921
  %1077 = bitcast i32 %1076 to float
  %1078 = fmul reassoc nsz arcp contract float %1074, %1077, !spirv.Decorations !898
  %1079 = fadd reassoc nsz arcp contract float %1078, %.sroa.214.1, !spirv.Decorations !898
  br label %.preheader.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %1062
  %.sroa.214.2 = phi float [ %1079, %1062 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %sink_sink_3878 = bitcast <2 x i32> %393 to i64
  %sink_sink_3854 = shl i64 %sink_sink_3878, 1
  %sink_3898 = add i64 %.in3822, %sink_sink_3854
  br i1 %201, label %1080, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
  br label %._crit_edge.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1080:                                             ; preds = %.preheader.5
  %.sroa.256.0.insert.ext703 = zext i32 %647 to i64
  %1081 = shl nuw nsw i64 %.sroa.256.0.insert.ext703, 1
  %1082 = add i64 %sink_3908, %1081
  %1083 = inttoptr i64 %1082 to i16 addrspace(4)*
  %1084 = addrspacecast i16 addrspace(4)* %1083 to i16 addrspace(1)*
  %1085 = load i16, i16 addrspace(1)* %1084, align 2
  %1086 = add i64 %sink_3898, %1081
  %1087 = inttoptr i64 %1086 to i16 addrspace(4)*
  %1088 = addrspacecast i16 addrspace(4)* %1087 to i16 addrspace(1)*
  %1089 = load i16, i16 addrspace(1)* %1088, align 2
  %1090 = zext i16 %1085 to i32
  %1091 = shl nuw i32 %1090, 16, !spirv.Decorations !921
  %1092 = bitcast i32 %1091 to float
  %1093 = zext i16 %1089 to i32
  %1094 = shl nuw i32 %1093, 16, !spirv.Decorations !921
  %1095 = bitcast i32 %1094 to float
  %1096 = fmul reassoc nsz arcp contract float %1092, %1095, !spirv.Decorations !898
  %1097 = fadd reassoc nsz arcp contract float %1096, %.sroa.26.1, !spirv.Decorations !898
  br label %._crit_edge.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %1080
  %.sroa.26.2 = phi float [ %1097, %1080 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  br i1 %204, label %1098, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1098:                                             ; preds = %._crit_edge.6
  %.sroa.256.0.insert.ext708 = zext i32 %647 to i64
  %1099 = shl nuw nsw i64 %.sroa.256.0.insert.ext708, 1
  %1100 = add i64 %sink_3906, %1099
  %1101 = inttoptr i64 %1100 to i16 addrspace(4)*
  %1102 = addrspacecast i16 addrspace(4)* %1101 to i16 addrspace(1)*
  %1103 = load i16, i16 addrspace(1)* %1102, align 2
  %1104 = add i64 %sink_3898, %1099
  %1105 = inttoptr i64 %1104 to i16 addrspace(4)*
  %1106 = addrspacecast i16 addrspace(4)* %1105 to i16 addrspace(1)*
  %1107 = load i16, i16 addrspace(1)* %1106, align 2
  %1108 = zext i16 %1103 to i32
  %1109 = shl nuw i32 %1108, 16, !spirv.Decorations !921
  %1110 = bitcast i32 %1109 to float
  %1111 = zext i16 %1107 to i32
  %1112 = shl nuw i32 %1111, 16, !spirv.Decorations !921
  %1113 = bitcast i32 %1112 to float
  %1114 = fmul reassoc nsz arcp contract float %1110, %1113, !spirv.Decorations !898
  %1115 = fadd reassoc nsz arcp contract float %1114, %.sroa.90.1, !spirv.Decorations !898
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %1098
  %.sroa.90.2 = phi float [ %1115, %1098 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  br i1 %207, label %1116, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1116:                                             ; preds = %._crit_edge.1.6
  %.sroa.256.0.insert.ext713 = zext i32 %647 to i64
  %1117 = shl nuw nsw i64 %.sroa.256.0.insert.ext713, 1
  %1118 = add i64 %sink_3905, %1117
  %1119 = inttoptr i64 %1118 to i16 addrspace(4)*
  %1120 = addrspacecast i16 addrspace(4)* %1119 to i16 addrspace(1)*
  %1121 = load i16, i16 addrspace(1)* %1120, align 2
  %1122 = add i64 %sink_3898, %1117
  %1123 = inttoptr i64 %1122 to i16 addrspace(4)*
  %1124 = addrspacecast i16 addrspace(4)* %1123 to i16 addrspace(1)*
  %1125 = load i16, i16 addrspace(1)* %1124, align 2
  %1126 = zext i16 %1121 to i32
  %1127 = shl nuw i32 %1126, 16, !spirv.Decorations !921
  %1128 = bitcast i32 %1127 to float
  %1129 = zext i16 %1125 to i32
  %1130 = shl nuw i32 %1129, 16, !spirv.Decorations !921
  %1131 = bitcast i32 %1130 to float
  %1132 = fmul reassoc nsz arcp contract float %1128, %1131, !spirv.Decorations !898
  %1133 = fadd reassoc nsz arcp contract float %1132, %.sroa.154.1, !spirv.Decorations !898
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %1116
  %.sroa.154.2 = phi float [ %1133, %1116 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  br i1 %210, label %1134, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
  br label %.preheader.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1134:                                             ; preds = %._crit_edge.2.6
  %.sroa.256.0.insert.ext718 = zext i32 %647 to i64
  %1135 = shl nuw nsw i64 %.sroa.256.0.insert.ext718, 1
  %1136 = add i64 %sink_3904, %1135
  %1137 = inttoptr i64 %1136 to i16 addrspace(4)*
  %1138 = addrspacecast i16 addrspace(4)* %1137 to i16 addrspace(1)*
  %1139 = load i16, i16 addrspace(1)* %1138, align 2
  %1140 = add i64 %sink_3898, %1135
  %1141 = inttoptr i64 %1140 to i16 addrspace(4)*
  %1142 = addrspacecast i16 addrspace(4)* %1141 to i16 addrspace(1)*
  %1143 = load i16, i16 addrspace(1)* %1142, align 2
  %1144 = zext i16 %1139 to i32
  %1145 = shl nuw i32 %1144, 16, !spirv.Decorations !921
  %1146 = bitcast i32 %1145 to float
  %1147 = zext i16 %1143 to i32
  %1148 = shl nuw i32 %1147, 16, !spirv.Decorations !921
  %1149 = bitcast i32 %1148 to float
  %1150 = fmul reassoc nsz arcp contract float %1146, %1149, !spirv.Decorations !898
  %1151 = fadd reassoc nsz arcp contract float %1150, %.sroa.218.1, !spirv.Decorations !898
  br label %.preheader.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %1134
  %.sroa.218.2 = phi float [ %1151, %1134 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %sink_sink_3877 = bitcast <2 x i32> %399 to i64
  %sink_sink_3853 = shl i64 %sink_sink_3877, 1
  %sink_3897 = add i64 %.in3822, %sink_sink_3853
  br i1 %214, label %1152, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
  br label %._crit_edge.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1152:                                             ; preds = %.preheader.6
  %.sroa.256.0.insert.ext723 = zext i32 %647 to i64
  %1153 = shl nuw nsw i64 %.sroa.256.0.insert.ext723, 1
  %1154 = add i64 %sink_3908, %1153
  %1155 = inttoptr i64 %1154 to i16 addrspace(4)*
  %1156 = addrspacecast i16 addrspace(4)* %1155 to i16 addrspace(1)*
  %1157 = load i16, i16 addrspace(1)* %1156, align 2
  %1158 = add i64 %sink_3897, %1153
  %1159 = inttoptr i64 %1158 to i16 addrspace(4)*
  %1160 = addrspacecast i16 addrspace(4)* %1159 to i16 addrspace(1)*
  %1161 = load i16, i16 addrspace(1)* %1160, align 2
  %1162 = zext i16 %1157 to i32
  %1163 = shl nuw i32 %1162, 16, !spirv.Decorations !921
  %1164 = bitcast i32 %1163 to float
  %1165 = zext i16 %1161 to i32
  %1166 = shl nuw i32 %1165, 16, !spirv.Decorations !921
  %1167 = bitcast i32 %1166 to float
  %1168 = fmul reassoc nsz arcp contract float %1164, %1167, !spirv.Decorations !898
  %1169 = fadd reassoc nsz arcp contract float %1168, %.sroa.30.1, !spirv.Decorations !898
  br label %._crit_edge.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %1152
  %.sroa.30.2 = phi float [ %1169, %1152 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  br i1 %217, label %1170, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1170:                                             ; preds = %._crit_edge.7
  %.sroa.256.0.insert.ext728 = zext i32 %647 to i64
  %1171 = shl nuw nsw i64 %.sroa.256.0.insert.ext728, 1
  %1172 = add i64 %sink_3906, %1171
  %1173 = inttoptr i64 %1172 to i16 addrspace(4)*
  %1174 = addrspacecast i16 addrspace(4)* %1173 to i16 addrspace(1)*
  %1175 = load i16, i16 addrspace(1)* %1174, align 2
  %1176 = add i64 %sink_3897, %1171
  %1177 = inttoptr i64 %1176 to i16 addrspace(4)*
  %1178 = addrspacecast i16 addrspace(4)* %1177 to i16 addrspace(1)*
  %1179 = load i16, i16 addrspace(1)* %1178, align 2
  %1180 = zext i16 %1175 to i32
  %1181 = shl nuw i32 %1180, 16, !spirv.Decorations !921
  %1182 = bitcast i32 %1181 to float
  %1183 = zext i16 %1179 to i32
  %1184 = shl nuw i32 %1183, 16, !spirv.Decorations !921
  %1185 = bitcast i32 %1184 to float
  %1186 = fmul reassoc nsz arcp contract float %1182, %1185, !spirv.Decorations !898
  %1187 = fadd reassoc nsz arcp contract float %1186, %.sroa.94.1, !spirv.Decorations !898
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %1170
  %.sroa.94.2 = phi float [ %1187, %1170 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  br i1 %220, label %1188, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1188:                                             ; preds = %._crit_edge.1.7
  %.sroa.256.0.insert.ext733 = zext i32 %647 to i64
  %1189 = shl nuw nsw i64 %.sroa.256.0.insert.ext733, 1
  %1190 = add i64 %sink_3905, %1189
  %1191 = inttoptr i64 %1190 to i16 addrspace(4)*
  %1192 = addrspacecast i16 addrspace(4)* %1191 to i16 addrspace(1)*
  %1193 = load i16, i16 addrspace(1)* %1192, align 2
  %1194 = add i64 %sink_3897, %1189
  %1195 = inttoptr i64 %1194 to i16 addrspace(4)*
  %1196 = addrspacecast i16 addrspace(4)* %1195 to i16 addrspace(1)*
  %1197 = load i16, i16 addrspace(1)* %1196, align 2
  %1198 = zext i16 %1193 to i32
  %1199 = shl nuw i32 %1198, 16, !spirv.Decorations !921
  %1200 = bitcast i32 %1199 to float
  %1201 = zext i16 %1197 to i32
  %1202 = shl nuw i32 %1201, 16, !spirv.Decorations !921
  %1203 = bitcast i32 %1202 to float
  %1204 = fmul reassoc nsz arcp contract float %1200, %1203, !spirv.Decorations !898
  %1205 = fadd reassoc nsz arcp contract float %1204, %.sroa.158.1, !spirv.Decorations !898
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %1188
  %.sroa.158.2 = phi float [ %1205, %1188 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  br i1 %223, label %1206, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
  br label %.preheader.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1206:                                             ; preds = %._crit_edge.2.7
  %.sroa.256.0.insert.ext738 = zext i32 %647 to i64
  %1207 = shl nuw nsw i64 %.sroa.256.0.insert.ext738, 1
  %1208 = add i64 %sink_3904, %1207
  %1209 = inttoptr i64 %1208 to i16 addrspace(4)*
  %1210 = addrspacecast i16 addrspace(4)* %1209 to i16 addrspace(1)*
  %1211 = load i16, i16 addrspace(1)* %1210, align 2
  %1212 = add i64 %sink_3897, %1207
  %1213 = inttoptr i64 %1212 to i16 addrspace(4)*
  %1214 = addrspacecast i16 addrspace(4)* %1213 to i16 addrspace(1)*
  %1215 = load i16, i16 addrspace(1)* %1214, align 2
  %1216 = zext i16 %1211 to i32
  %1217 = shl nuw i32 %1216, 16, !spirv.Decorations !921
  %1218 = bitcast i32 %1217 to float
  %1219 = zext i16 %1215 to i32
  %1220 = shl nuw i32 %1219, 16, !spirv.Decorations !921
  %1221 = bitcast i32 %1220 to float
  %1222 = fmul reassoc nsz arcp contract float %1218, %1221, !spirv.Decorations !898
  %1223 = fadd reassoc nsz arcp contract float %1222, %.sroa.222.1, !spirv.Decorations !898
  br label %.preheader.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %1206
  %.sroa.222.2 = phi float [ %1223, %1206 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %sink_sink_3876 = bitcast <2 x i32> %405 to i64
  %sink_sink_3852 = shl i64 %sink_sink_3876, 1
  %sink_3896 = add i64 %.in3822, %sink_sink_3852
  br i1 %227, label %1224, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
  br label %._crit_edge.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1224:                                             ; preds = %.preheader.7
  %.sroa.256.0.insert.ext743 = zext i32 %647 to i64
  %1225 = shl nuw nsw i64 %.sroa.256.0.insert.ext743, 1
  %1226 = add i64 %sink_3908, %1225
  %1227 = inttoptr i64 %1226 to i16 addrspace(4)*
  %1228 = addrspacecast i16 addrspace(4)* %1227 to i16 addrspace(1)*
  %1229 = load i16, i16 addrspace(1)* %1228, align 2
  %1230 = add i64 %sink_3896, %1225
  %1231 = inttoptr i64 %1230 to i16 addrspace(4)*
  %1232 = addrspacecast i16 addrspace(4)* %1231 to i16 addrspace(1)*
  %1233 = load i16, i16 addrspace(1)* %1232, align 2
  %1234 = zext i16 %1229 to i32
  %1235 = shl nuw i32 %1234, 16, !spirv.Decorations !921
  %1236 = bitcast i32 %1235 to float
  %1237 = zext i16 %1233 to i32
  %1238 = shl nuw i32 %1237, 16, !spirv.Decorations !921
  %1239 = bitcast i32 %1238 to float
  %1240 = fmul reassoc nsz arcp contract float %1236, %1239, !spirv.Decorations !898
  %1241 = fadd reassoc nsz arcp contract float %1240, %.sroa.34.1, !spirv.Decorations !898
  br label %._crit_edge.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %1224
  %.sroa.34.2 = phi float [ %1241, %1224 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  br i1 %230, label %1242, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1242:                                             ; preds = %._crit_edge.8
  %.sroa.256.0.insert.ext748 = zext i32 %647 to i64
  %1243 = shl nuw nsw i64 %.sroa.256.0.insert.ext748, 1
  %1244 = add i64 %sink_3906, %1243
  %1245 = inttoptr i64 %1244 to i16 addrspace(4)*
  %1246 = addrspacecast i16 addrspace(4)* %1245 to i16 addrspace(1)*
  %1247 = load i16, i16 addrspace(1)* %1246, align 2
  %1248 = add i64 %sink_3896, %1243
  %1249 = inttoptr i64 %1248 to i16 addrspace(4)*
  %1250 = addrspacecast i16 addrspace(4)* %1249 to i16 addrspace(1)*
  %1251 = load i16, i16 addrspace(1)* %1250, align 2
  %1252 = zext i16 %1247 to i32
  %1253 = shl nuw i32 %1252, 16, !spirv.Decorations !921
  %1254 = bitcast i32 %1253 to float
  %1255 = zext i16 %1251 to i32
  %1256 = shl nuw i32 %1255, 16, !spirv.Decorations !921
  %1257 = bitcast i32 %1256 to float
  %1258 = fmul reassoc nsz arcp contract float %1254, %1257, !spirv.Decorations !898
  %1259 = fadd reassoc nsz arcp contract float %1258, %.sroa.98.1, !spirv.Decorations !898
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %1242
  %.sroa.98.2 = phi float [ %1259, %1242 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  br i1 %233, label %1260, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1260:                                             ; preds = %._crit_edge.1.8
  %.sroa.256.0.insert.ext753 = zext i32 %647 to i64
  %1261 = shl nuw nsw i64 %.sroa.256.0.insert.ext753, 1
  %1262 = add i64 %sink_3905, %1261
  %1263 = inttoptr i64 %1262 to i16 addrspace(4)*
  %1264 = addrspacecast i16 addrspace(4)* %1263 to i16 addrspace(1)*
  %1265 = load i16, i16 addrspace(1)* %1264, align 2
  %1266 = add i64 %sink_3896, %1261
  %1267 = inttoptr i64 %1266 to i16 addrspace(4)*
  %1268 = addrspacecast i16 addrspace(4)* %1267 to i16 addrspace(1)*
  %1269 = load i16, i16 addrspace(1)* %1268, align 2
  %1270 = zext i16 %1265 to i32
  %1271 = shl nuw i32 %1270, 16, !spirv.Decorations !921
  %1272 = bitcast i32 %1271 to float
  %1273 = zext i16 %1269 to i32
  %1274 = shl nuw i32 %1273, 16, !spirv.Decorations !921
  %1275 = bitcast i32 %1274 to float
  %1276 = fmul reassoc nsz arcp contract float %1272, %1275, !spirv.Decorations !898
  %1277 = fadd reassoc nsz arcp contract float %1276, %.sroa.162.1, !spirv.Decorations !898
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %1260
  %.sroa.162.2 = phi float [ %1277, %1260 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  br i1 %236, label %1278, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
  br label %.preheader.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1278:                                             ; preds = %._crit_edge.2.8
  %.sroa.256.0.insert.ext758 = zext i32 %647 to i64
  %1279 = shl nuw nsw i64 %.sroa.256.0.insert.ext758, 1
  %1280 = add i64 %sink_3904, %1279
  %1281 = inttoptr i64 %1280 to i16 addrspace(4)*
  %1282 = addrspacecast i16 addrspace(4)* %1281 to i16 addrspace(1)*
  %1283 = load i16, i16 addrspace(1)* %1282, align 2
  %1284 = add i64 %sink_3896, %1279
  %1285 = inttoptr i64 %1284 to i16 addrspace(4)*
  %1286 = addrspacecast i16 addrspace(4)* %1285 to i16 addrspace(1)*
  %1287 = load i16, i16 addrspace(1)* %1286, align 2
  %1288 = zext i16 %1283 to i32
  %1289 = shl nuw i32 %1288, 16, !spirv.Decorations !921
  %1290 = bitcast i32 %1289 to float
  %1291 = zext i16 %1287 to i32
  %1292 = shl nuw i32 %1291, 16, !spirv.Decorations !921
  %1293 = bitcast i32 %1292 to float
  %1294 = fmul reassoc nsz arcp contract float %1290, %1293, !spirv.Decorations !898
  %1295 = fadd reassoc nsz arcp contract float %1294, %.sroa.226.1, !spirv.Decorations !898
  br label %.preheader.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %1278
  %.sroa.226.2 = phi float [ %1295, %1278 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %sink_sink_3875 = bitcast <2 x i32> %411 to i64
  %sink_sink_3851 = shl i64 %sink_sink_3875, 1
  %sink_3895 = add i64 %.in3822, %sink_sink_3851
  br i1 %240, label %1296, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
  br label %._crit_edge.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1296:                                             ; preds = %.preheader.8
  %.sroa.256.0.insert.ext763 = zext i32 %647 to i64
  %1297 = shl nuw nsw i64 %.sroa.256.0.insert.ext763, 1
  %1298 = add i64 %sink_3908, %1297
  %1299 = inttoptr i64 %1298 to i16 addrspace(4)*
  %1300 = addrspacecast i16 addrspace(4)* %1299 to i16 addrspace(1)*
  %1301 = load i16, i16 addrspace(1)* %1300, align 2
  %1302 = add i64 %sink_3895, %1297
  %1303 = inttoptr i64 %1302 to i16 addrspace(4)*
  %1304 = addrspacecast i16 addrspace(4)* %1303 to i16 addrspace(1)*
  %1305 = load i16, i16 addrspace(1)* %1304, align 2
  %1306 = zext i16 %1301 to i32
  %1307 = shl nuw i32 %1306, 16, !spirv.Decorations !921
  %1308 = bitcast i32 %1307 to float
  %1309 = zext i16 %1305 to i32
  %1310 = shl nuw i32 %1309, 16, !spirv.Decorations !921
  %1311 = bitcast i32 %1310 to float
  %1312 = fmul reassoc nsz arcp contract float %1308, %1311, !spirv.Decorations !898
  %1313 = fadd reassoc nsz arcp contract float %1312, %.sroa.38.1, !spirv.Decorations !898
  br label %._crit_edge.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %1296
  %.sroa.38.2 = phi float [ %1313, %1296 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  br i1 %243, label %1314, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1314:                                             ; preds = %._crit_edge.9
  %.sroa.256.0.insert.ext768 = zext i32 %647 to i64
  %1315 = shl nuw nsw i64 %.sroa.256.0.insert.ext768, 1
  %1316 = add i64 %sink_3906, %1315
  %1317 = inttoptr i64 %1316 to i16 addrspace(4)*
  %1318 = addrspacecast i16 addrspace(4)* %1317 to i16 addrspace(1)*
  %1319 = load i16, i16 addrspace(1)* %1318, align 2
  %1320 = add i64 %sink_3895, %1315
  %1321 = inttoptr i64 %1320 to i16 addrspace(4)*
  %1322 = addrspacecast i16 addrspace(4)* %1321 to i16 addrspace(1)*
  %1323 = load i16, i16 addrspace(1)* %1322, align 2
  %1324 = zext i16 %1319 to i32
  %1325 = shl nuw i32 %1324, 16, !spirv.Decorations !921
  %1326 = bitcast i32 %1325 to float
  %1327 = zext i16 %1323 to i32
  %1328 = shl nuw i32 %1327, 16, !spirv.Decorations !921
  %1329 = bitcast i32 %1328 to float
  %1330 = fmul reassoc nsz arcp contract float %1326, %1329, !spirv.Decorations !898
  %1331 = fadd reassoc nsz arcp contract float %1330, %.sroa.102.1, !spirv.Decorations !898
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %1314
  %.sroa.102.2 = phi float [ %1331, %1314 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  br i1 %246, label %1332, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1332:                                             ; preds = %._crit_edge.1.9
  %.sroa.256.0.insert.ext773 = zext i32 %647 to i64
  %1333 = shl nuw nsw i64 %.sroa.256.0.insert.ext773, 1
  %1334 = add i64 %sink_3905, %1333
  %1335 = inttoptr i64 %1334 to i16 addrspace(4)*
  %1336 = addrspacecast i16 addrspace(4)* %1335 to i16 addrspace(1)*
  %1337 = load i16, i16 addrspace(1)* %1336, align 2
  %1338 = add i64 %sink_3895, %1333
  %1339 = inttoptr i64 %1338 to i16 addrspace(4)*
  %1340 = addrspacecast i16 addrspace(4)* %1339 to i16 addrspace(1)*
  %1341 = load i16, i16 addrspace(1)* %1340, align 2
  %1342 = zext i16 %1337 to i32
  %1343 = shl nuw i32 %1342, 16, !spirv.Decorations !921
  %1344 = bitcast i32 %1343 to float
  %1345 = zext i16 %1341 to i32
  %1346 = shl nuw i32 %1345, 16, !spirv.Decorations !921
  %1347 = bitcast i32 %1346 to float
  %1348 = fmul reassoc nsz arcp contract float %1344, %1347, !spirv.Decorations !898
  %1349 = fadd reassoc nsz arcp contract float %1348, %.sroa.166.1, !spirv.Decorations !898
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %1332
  %.sroa.166.2 = phi float [ %1349, %1332 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  br i1 %249, label %1350, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
  br label %.preheader.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1350:                                             ; preds = %._crit_edge.2.9
  %.sroa.256.0.insert.ext778 = zext i32 %647 to i64
  %1351 = shl nuw nsw i64 %.sroa.256.0.insert.ext778, 1
  %1352 = add i64 %sink_3904, %1351
  %1353 = inttoptr i64 %1352 to i16 addrspace(4)*
  %1354 = addrspacecast i16 addrspace(4)* %1353 to i16 addrspace(1)*
  %1355 = load i16, i16 addrspace(1)* %1354, align 2
  %1356 = add i64 %sink_3895, %1351
  %1357 = inttoptr i64 %1356 to i16 addrspace(4)*
  %1358 = addrspacecast i16 addrspace(4)* %1357 to i16 addrspace(1)*
  %1359 = load i16, i16 addrspace(1)* %1358, align 2
  %1360 = zext i16 %1355 to i32
  %1361 = shl nuw i32 %1360, 16, !spirv.Decorations !921
  %1362 = bitcast i32 %1361 to float
  %1363 = zext i16 %1359 to i32
  %1364 = shl nuw i32 %1363, 16, !spirv.Decorations !921
  %1365 = bitcast i32 %1364 to float
  %1366 = fmul reassoc nsz arcp contract float %1362, %1365, !spirv.Decorations !898
  %1367 = fadd reassoc nsz arcp contract float %1366, %.sroa.230.1, !spirv.Decorations !898
  br label %.preheader.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %1350
  %.sroa.230.2 = phi float [ %1367, %1350 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %sink_sink_3874 = bitcast <2 x i32> %417 to i64
  %sink_sink_3850 = shl i64 %sink_sink_3874, 1
  %sink_3894 = add i64 %.in3822, %sink_sink_3850
  br i1 %253, label %1368, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
  br label %._crit_edge.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1368:                                             ; preds = %.preheader.9
  %.sroa.256.0.insert.ext783 = zext i32 %647 to i64
  %1369 = shl nuw nsw i64 %.sroa.256.0.insert.ext783, 1
  %1370 = add i64 %sink_3908, %1369
  %1371 = inttoptr i64 %1370 to i16 addrspace(4)*
  %1372 = addrspacecast i16 addrspace(4)* %1371 to i16 addrspace(1)*
  %1373 = load i16, i16 addrspace(1)* %1372, align 2
  %1374 = add i64 %sink_3894, %1369
  %1375 = inttoptr i64 %1374 to i16 addrspace(4)*
  %1376 = addrspacecast i16 addrspace(4)* %1375 to i16 addrspace(1)*
  %1377 = load i16, i16 addrspace(1)* %1376, align 2
  %1378 = zext i16 %1373 to i32
  %1379 = shl nuw i32 %1378, 16, !spirv.Decorations !921
  %1380 = bitcast i32 %1379 to float
  %1381 = zext i16 %1377 to i32
  %1382 = shl nuw i32 %1381, 16, !spirv.Decorations !921
  %1383 = bitcast i32 %1382 to float
  %1384 = fmul reassoc nsz arcp contract float %1380, %1383, !spirv.Decorations !898
  %1385 = fadd reassoc nsz arcp contract float %1384, %.sroa.42.1, !spirv.Decorations !898
  br label %._crit_edge.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %1368
  %.sroa.42.2 = phi float [ %1385, %1368 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  br i1 %256, label %1386, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1386:                                             ; preds = %._crit_edge.10
  %.sroa.256.0.insert.ext788 = zext i32 %647 to i64
  %1387 = shl nuw nsw i64 %.sroa.256.0.insert.ext788, 1
  %1388 = add i64 %sink_3906, %1387
  %1389 = inttoptr i64 %1388 to i16 addrspace(4)*
  %1390 = addrspacecast i16 addrspace(4)* %1389 to i16 addrspace(1)*
  %1391 = load i16, i16 addrspace(1)* %1390, align 2
  %1392 = add i64 %sink_3894, %1387
  %1393 = inttoptr i64 %1392 to i16 addrspace(4)*
  %1394 = addrspacecast i16 addrspace(4)* %1393 to i16 addrspace(1)*
  %1395 = load i16, i16 addrspace(1)* %1394, align 2
  %1396 = zext i16 %1391 to i32
  %1397 = shl nuw i32 %1396, 16, !spirv.Decorations !921
  %1398 = bitcast i32 %1397 to float
  %1399 = zext i16 %1395 to i32
  %1400 = shl nuw i32 %1399, 16, !spirv.Decorations !921
  %1401 = bitcast i32 %1400 to float
  %1402 = fmul reassoc nsz arcp contract float %1398, %1401, !spirv.Decorations !898
  %1403 = fadd reassoc nsz arcp contract float %1402, %.sroa.106.1, !spirv.Decorations !898
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %1386
  %.sroa.106.2 = phi float [ %1403, %1386 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  br i1 %259, label %1404, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1404:                                             ; preds = %._crit_edge.1.10
  %.sroa.256.0.insert.ext793 = zext i32 %647 to i64
  %1405 = shl nuw nsw i64 %.sroa.256.0.insert.ext793, 1
  %1406 = add i64 %sink_3905, %1405
  %1407 = inttoptr i64 %1406 to i16 addrspace(4)*
  %1408 = addrspacecast i16 addrspace(4)* %1407 to i16 addrspace(1)*
  %1409 = load i16, i16 addrspace(1)* %1408, align 2
  %1410 = add i64 %sink_3894, %1405
  %1411 = inttoptr i64 %1410 to i16 addrspace(4)*
  %1412 = addrspacecast i16 addrspace(4)* %1411 to i16 addrspace(1)*
  %1413 = load i16, i16 addrspace(1)* %1412, align 2
  %1414 = zext i16 %1409 to i32
  %1415 = shl nuw i32 %1414, 16, !spirv.Decorations !921
  %1416 = bitcast i32 %1415 to float
  %1417 = zext i16 %1413 to i32
  %1418 = shl nuw i32 %1417, 16, !spirv.Decorations !921
  %1419 = bitcast i32 %1418 to float
  %1420 = fmul reassoc nsz arcp contract float %1416, %1419, !spirv.Decorations !898
  %1421 = fadd reassoc nsz arcp contract float %1420, %.sroa.170.1, !spirv.Decorations !898
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %1404
  %.sroa.170.2 = phi float [ %1421, %1404 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  br i1 %262, label %1422, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
  br label %.preheader.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1422:                                             ; preds = %._crit_edge.2.10
  %.sroa.256.0.insert.ext798 = zext i32 %647 to i64
  %1423 = shl nuw nsw i64 %.sroa.256.0.insert.ext798, 1
  %1424 = add i64 %sink_3904, %1423
  %1425 = inttoptr i64 %1424 to i16 addrspace(4)*
  %1426 = addrspacecast i16 addrspace(4)* %1425 to i16 addrspace(1)*
  %1427 = load i16, i16 addrspace(1)* %1426, align 2
  %1428 = add i64 %sink_3894, %1423
  %1429 = inttoptr i64 %1428 to i16 addrspace(4)*
  %1430 = addrspacecast i16 addrspace(4)* %1429 to i16 addrspace(1)*
  %1431 = load i16, i16 addrspace(1)* %1430, align 2
  %1432 = zext i16 %1427 to i32
  %1433 = shl nuw i32 %1432, 16, !spirv.Decorations !921
  %1434 = bitcast i32 %1433 to float
  %1435 = zext i16 %1431 to i32
  %1436 = shl nuw i32 %1435, 16, !spirv.Decorations !921
  %1437 = bitcast i32 %1436 to float
  %1438 = fmul reassoc nsz arcp contract float %1434, %1437, !spirv.Decorations !898
  %1439 = fadd reassoc nsz arcp contract float %1438, %.sroa.234.1, !spirv.Decorations !898
  br label %.preheader.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %1422
  %.sroa.234.2 = phi float [ %1439, %1422 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %sink_sink_3873 = bitcast <2 x i32> %423 to i64
  %sink_sink_3849 = shl i64 %sink_sink_3873, 1
  %sink_3893 = add i64 %.in3822, %sink_sink_3849
  br i1 %266, label %1440, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
  br label %._crit_edge.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1440:                                             ; preds = %.preheader.10
  %.sroa.256.0.insert.ext803 = zext i32 %647 to i64
  %1441 = shl nuw nsw i64 %.sroa.256.0.insert.ext803, 1
  %1442 = add i64 %sink_3908, %1441
  %1443 = inttoptr i64 %1442 to i16 addrspace(4)*
  %1444 = addrspacecast i16 addrspace(4)* %1443 to i16 addrspace(1)*
  %1445 = load i16, i16 addrspace(1)* %1444, align 2
  %1446 = add i64 %sink_3893, %1441
  %1447 = inttoptr i64 %1446 to i16 addrspace(4)*
  %1448 = addrspacecast i16 addrspace(4)* %1447 to i16 addrspace(1)*
  %1449 = load i16, i16 addrspace(1)* %1448, align 2
  %1450 = zext i16 %1445 to i32
  %1451 = shl nuw i32 %1450, 16, !spirv.Decorations !921
  %1452 = bitcast i32 %1451 to float
  %1453 = zext i16 %1449 to i32
  %1454 = shl nuw i32 %1453, 16, !spirv.Decorations !921
  %1455 = bitcast i32 %1454 to float
  %1456 = fmul reassoc nsz arcp contract float %1452, %1455, !spirv.Decorations !898
  %1457 = fadd reassoc nsz arcp contract float %1456, %.sroa.46.1, !spirv.Decorations !898
  br label %._crit_edge.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %1440
  %.sroa.46.2 = phi float [ %1457, %1440 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  br i1 %269, label %1458, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1458:                                             ; preds = %._crit_edge.11
  %.sroa.256.0.insert.ext808 = zext i32 %647 to i64
  %1459 = shl nuw nsw i64 %.sroa.256.0.insert.ext808, 1
  %1460 = add i64 %sink_3906, %1459
  %1461 = inttoptr i64 %1460 to i16 addrspace(4)*
  %1462 = addrspacecast i16 addrspace(4)* %1461 to i16 addrspace(1)*
  %1463 = load i16, i16 addrspace(1)* %1462, align 2
  %1464 = add i64 %sink_3893, %1459
  %1465 = inttoptr i64 %1464 to i16 addrspace(4)*
  %1466 = addrspacecast i16 addrspace(4)* %1465 to i16 addrspace(1)*
  %1467 = load i16, i16 addrspace(1)* %1466, align 2
  %1468 = zext i16 %1463 to i32
  %1469 = shl nuw i32 %1468, 16, !spirv.Decorations !921
  %1470 = bitcast i32 %1469 to float
  %1471 = zext i16 %1467 to i32
  %1472 = shl nuw i32 %1471, 16, !spirv.Decorations !921
  %1473 = bitcast i32 %1472 to float
  %1474 = fmul reassoc nsz arcp contract float %1470, %1473, !spirv.Decorations !898
  %1475 = fadd reassoc nsz arcp contract float %1474, %.sroa.110.1, !spirv.Decorations !898
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %1458
  %.sroa.110.2 = phi float [ %1475, %1458 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  br i1 %272, label %1476, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1476:                                             ; preds = %._crit_edge.1.11
  %.sroa.256.0.insert.ext813 = zext i32 %647 to i64
  %1477 = shl nuw nsw i64 %.sroa.256.0.insert.ext813, 1
  %1478 = add i64 %sink_3905, %1477
  %1479 = inttoptr i64 %1478 to i16 addrspace(4)*
  %1480 = addrspacecast i16 addrspace(4)* %1479 to i16 addrspace(1)*
  %1481 = load i16, i16 addrspace(1)* %1480, align 2
  %1482 = add i64 %sink_3893, %1477
  %1483 = inttoptr i64 %1482 to i16 addrspace(4)*
  %1484 = addrspacecast i16 addrspace(4)* %1483 to i16 addrspace(1)*
  %1485 = load i16, i16 addrspace(1)* %1484, align 2
  %1486 = zext i16 %1481 to i32
  %1487 = shl nuw i32 %1486, 16, !spirv.Decorations !921
  %1488 = bitcast i32 %1487 to float
  %1489 = zext i16 %1485 to i32
  %1490 = shl nuw i32 %1489, 16, !spirv.Decorations !921
  %1491 = bitcast i32 %1490 to float
  %1492 = fmul reassoc nsz arcp contract float %1488, %1491, !spirv.Decorations !898
  %1493 = fadd reassoc nsz arcp contract float %1492, %.sroa.174.1, !spirv.Decorations !898
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %1476
  %.sroa.174.2 = phi float [ %1493, %1476 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  br i1 %275, label %1494, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
  br label %.preheader.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1494:                                             ; preds = %._crit_edge.2.11
  %.sroa.256.0.insert.ext818 = zext i32 %647 to i64
  %1495 = shl nuw nsw i64 %.sroa.256.0.insert.ext818, 1
  %1496 = add i64 %sink_3904, %1495
  %1497 = inttoptr i64 %1496 to i16 addrspace(4)*
  %1498 = addrspacecast i16 addrspace(4)* %1497 to i16 addrspace(1)*
  %1499 = load i16, i16 addrspace(1)* %1498, align 2
  %1500 = add i64 %sink_3893, %1495
  %1501 = inttoptr i64 %1500 to i16 addrspace(4)*
  %1502 = addrspacecast i16 addrspace(4)* %1501 to i16 addrspace(1)*
  %1503 = load i16, i16 addrspace(1)* %1502, align 2
  %1504 = zext i16 %1499 to i32
  %1505 = shl nuw i32 %1504, 16, !spirv.Decorations !921
  %1506 = bitcast i32 %1505 to float
  %1507 = zext i16 %1503 to i32
  %1508 = shl nuw i32 %1507, 16, !spirv.Decorations !921
  %1509 = bitcast i32 %1508 to float
  %1510 = fmul reassoc nsz arcp contract float %1506, %1509, !spirv.Decorations !898
  %1511 = fadd reassoc nsz arcp contract float %1510, %.sroa.238.1, !spirv.Decorations !898
  br label %.preheader.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %1494
  %.sroa.238.2 = phi float [ %1511, %1494 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %sink_sink_3872 = bitcast <2 x i32> %429 to i64
  %sink_sink_3848 = shl i64 %sink_sink_3872, 1
  %sink_3892 = add i64 %.in3822, %sink_sink_3848
  br i1 %279, label %1512, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
  br label %._crit_edge.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1512:                                             ; preds = %.preheader.11
  %.sroa.256.0.insert.ext823 = zext i32 %647 to i64
  %1513 = shl nuw nsw i64 %.sroa.256.0.insert.ext823, 1
  %1514 = add i64 %sink_3908, %1513
  %1515 = inttoptr i64 %1514 to i16 addrspace(4)*
  %1516 = addrspacecast i16 addrspace(4)* %1515 to i16 addrspace(1)*
  %1517 = load i16, i16 addrspace(1)* %1516, align 2
  %1518 = add i64 %sink_3892, %1513
  %1519 = inttoptr i64 %1518 to i16 addrspace(4)*
  %1520 = addrspacecast i16 addrspace(4)* %1519 to i16 addrspace(1)*
  %1521 = load i16, i16 addrspace(1)* %1520, align 2
  %1522 = zext i16 %1517 to i32
  %1523 = shl nuw i32 %1522, 16, !spirv.Decorations !921
  %1524 = bitcast i32 %1523 to float
  %1525 = zext i16 %1521 to i32
  %1526 = shl nuw i32 %1525, 16, !spirv.Decorations !921
  %1527 = bitcast i32 %1526 to float
  %1528 = fmul reassoc nsz arcp contract float %1524, %1527, !spirv.Decorations !898
  %1529 = fadd reassoc nsz arcp contract float %1528, %.sroa.50.1, !spirv.Decorations !898
  br label %._crit_edge.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %1512
  %.sroa.50.2 = phi float [ %1529, %1512 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  br i1 %282, label %1530, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1530:                                             ; preds = %._crit_edge.12
  %.sroa.256.0.insert.ext828 = zext i32 %647 to i64
  %1531 = shl nuw nsw i64 %.sroa.256.0.insert.ext828, 1
  %1532 = add i64 %sink_3906, %1531
  %1533 = inttoptr i64 %1532 to i16 addrspace(4)*
  %1534 = addrspacecast i16 addrspace(4)* %1533 to i16 addrspace(1)*
  %1535 = load i16, i16 addrspace(1)* %1534, align 2
  %1536 = add i64 %sink_3892, %1531
  %1537 = inttoptr i64 %1536 to i16 addrspace(4)*
  %1538 = addrspacecast i16 addrspace(4)* %1537 to i16 addrspace(1)*
  %1539 = load i16, i16 addrspace(1)* %1538, align 2
  %1540 = zext i16 %1535 to i32
  %1541 = shl nuw i32 %1540, 16, !spirv.Decorations !921
  %1542 = bitcast i32 %1541 to float
  %1543 = zext i16 %1539 to i32
  %1544 = shl nuw i32 %1543, 16, !spirv.Decorations !921
  %1545 = bitcast i32 %1544 to float
  %1546 = fmul reassoc nsz arcp contract float %1542, %1545, !spirv.Decorations !898
  %1547 = fadd reassoc nsz arcp contract float %1546, %.sroa.114.1, !spirv.Decorations !898
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %1530
  %.sroa.114.2 = phi float [ %1547, %1530 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  br i1 %285, label %1548, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1548:                                             ; preds = %._crit_edge.1.12
  %.sroa.256.0.insert.ext833 = zext i32 %647 to i64
  %1549 = shl nuw nsw i64 %.sroa.256.0.insert.ext833, 1
  %1550 = add i64 %sink_3905, %1549
  %1551 = inttoptr i64 %1550 to i16 addrspace(4)*
  %1552 = addrspacecast i16 addrspace(4)* %1551 to i16 addrspace(1)*
  %1553 = load i16, i16 addrspace(1)* %1552, align 2
  %1554 = add i64 %sink_3892, %1549
  %1555 = inttoptr i64 %1554 to i16 addrspace(4)*
  %1556 = addrspacecast i16 addrspace(4)* %1555 to i16 addrspace(1)*
  %1557 = load i16, i16 addrspace(1)* %1556, align 2
  %1558 = zext i16 %1553 to i32
  %1559 = shl nuw i32 %1558, 16, !spirv.Decorations !921
  %1560 = bitcast i32 %1559 to float
  %1561 = zext i16 %1557 to i32
  %1562 = shl nuw i32 %1561, 16, !spirv.Decorations !921
  %1563 = bitcast i32 %1562 to float
  %1564 = fmul reassoc nsz arcp contract float %1560, %1563, !spirv.Decorations !898
  %1565 = fadd reassoc nsz arcp contract float %1564, %.sroa.178.1, !spirv.Decorations !898
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %1548
  %.sroa.178.2 = phi float [ %1565, %1548 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  br i1 %288, label %1566, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
  br label %.preheader.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1566:                                             ; preds = %._crit_edge.2.12
  %.sroa.256.0.insert.ext838 = zext i32 %647 to i64
  %1567 = shl nuw nsw i64 %.sroa.256.0.insert.ext838, 1
  %1568 = add i64 %sink_3904, %1567
  %1569 = inttoptr i64 %1568 to i16 addrspace(4)*
  %1570 = addrspacecast i16 addrspace(4)* %1569 to i16 addrspace(1)*
  %1571 = load i16, i16 addrspace(1)* %1570, align 2
  %1572 = add i64 %sink_3892, %1567
  %1573 = inttoptr i64 %1572 to i16 addrspace(4)*
  %1574 = addrspacecast i16 addrspace(4)* %1573 to i16 addrspace(1)*
  %1575 = load i16, i16 addrspace(1)* %1574, align 2
  %1576 = zext i16 %1571 to i32
  %1577 = shl nuw i32 %1576, 16, !spirv.Decorations !921
  %1578 = bitcast i32 %1577 to float
  %1579 = zext i16 %1575 to i32
  %1580 = shl nuw i32 %1579, 16, !spirv.Decorations !921
  %1581 = bitcast i32 %1580 to float
  %1582 = fmul reassoc nsz arcp contract float %1578, %1581, !spirv.Decorations !898
  %1583 = fadd reassoc nsz arcp contract float %1582, %.sroa.242.1, !spirv.Decorations !898
  br label %.preheader.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %1566
  %.sroa.242.2 = phi float [ %1583, %1566 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %sink_sink_3871 = bitcast <2 x i32> %435 to i64
  %sink_sink_3847 = shl i64 %sink_sink_3871, 1
  %sink_3891 = add i64 %.in3822, %sink_sink_3847
  br i1 %292, label %1584, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
  br label %._crit_edge.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1584:                                             ; preds = %.preheader.12
  %.sroa.256.0.insert.ext843 = zext i32 %647 to i64
  %1585 = shl nuw nsw i64 %.sroa.256.0.insert.ext843, 1
  %1586 = add i64 %sink_3908, %1585
  %1587 = inttoptr i64 %1586 to i16 addrspace(4)*
  %1588 = addrspacecast i16 addrspace(4)* %1587 to i16 addrspace(1)*
  %1589 = load i16, i16 addrspace(1)* %1588, align 2
  %1590 = add i64 %sink_3891, %1585
  %1591 = inttoptr i64 %1590 to i16 addrspace(4)*
  %1592 = addrspacecast i16 addrspace(4)* %1591 to i16 addrspace(1)*
  %1593 = load i16, i16 addrspace(1)* %1592, align 2
  %1594 = zext i16 %1589 to i32
  %1595 = shl nuw i32 %1594, 16, !spirv.Decorations !921
  %1596 = bitcast i32 %1595 to float
  %1597 = zext i16 %1593 to i32
  %1598 = shl nuw i32 %1597, 16, !spirv.Decorations !921
  %1599 = bitcast i32 %1598 to float
  %1600 = fmul reassoc nsz arcp contract float %1596, %1599, !spirv.Decorations !898
  %1601 = fadd reassoc nsz arcp contract float %1600, %.sroa.54.1, !spirv.Decorations !898
  br label %._crit_edge.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %1584
  %.sroa.54.2 = phi float [ %1601, %1584 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  br i1 %295, label %1602, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1602:                                             ; preds = %._crit_edge.13
  %.sroa.256.0.insert.ext848 = zext i32 %647 to i64
  %1603 = shl nuw nsw i64 %.sroa.256.0.insert.ext848, 1
  %1604 = add i64 %sink_3906, %1603
  %1605 = inttoptr i64 %1604 to i16 addrspace(4)*
  %1606 = addrspacecast i16 addrspace(4)* %1605 to i16 addrspace(1)*
  %1607 = load i16, i16 addrspace(1)* %1606, align 2
  %1608 = add i64 %sink_3891, %1603
  %1609 = inttoptr i64 %1608 to i16 addrspace(4)*
  %1610 = addrspacecast i16 addrspace(4)* %1609 to i16 addrspace(1)*
  %1611 = load i16, i16 addrspace(1)* %1610, align 2
  %1612 = zext i16 %1607 to i32
  %1613 = shl nuw i32 %1612, 16, !spirv.Decorations !921
  %1614 = bitcast i32 %1613 to float
  %1615 = zext i16 %1611 to i32
  %1616 = shl nuw i32 %1615, 16, !spirv.Decorations !921
  %1617 = bitcast i32 %1616 to float
  %1618 = fmul reassoc nsz arcp contract float %1614, %1617, !spirv.Decorations !898
  %1619 = fadd reassoc nsz arcp contract float %1618, %.sroa.118.1, !spirv.Decorations !898
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %1602
  %.sroa.118.2 = phi float [ %1619, %1602 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  br i1 %298, label %1620, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1620:                                             ; preds = %._crit_edge.1.13
  %.sroa.256.0.insert.ext853 = zext i32 %647 to i64
  %1621 = shl nuw nsw i64 %.sroa.256.0.insert.ext853, 1
  %1622 = add i64 %sink_3905, %1621
  %1623 = inttoptr i64 %1622 to i16 addrspace(4)*
  %1624 = addrspacecast i16 addrspace(4)* %1623 to i16 addrspace(1)*
  %1625 = load i16, i16 addrspace(1)* %1624, align 2
  %1626 = add i64 %sink_3891, %1621
  %1627 = inttoptr i64 %1626 to i16 addrspace(4)*
  %1628 = addrspacecast i16 addrspace(4)* %1627 to i16 addrspace(1)*
  %1629 = load i16, i16 addrspace(1)* %1628, align 2
  %1630 = zext i16 %1625 to i32
  %1631 = shl nuw i32 %1630, 16, !spirv.Decorations !921
  %1632 = bitcast i32 %1631 to float
  %1633 = zext i16 %1629 to i32
  %1634 = shl nuw i32 %1633, 16, !spirv.Decorations !921
  %1635 = bitcast i32 %1634 to float
  %1636 = fmul reassoc nsz arcp contract float %1632, %1635, !spirv.Decorations !898
  %1637 = fadd reassoc nsz arcp contract float %1636, %.sroa.182.1, !spirv.Decorations !898
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %1620
  %.sroa.182.2 = phi float [ %1637, %1620 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  br i1 %301, label %1638, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
  br label %.preheader.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1638:                                             ; preds = %._crit_edge.2.13
  %.sroa.256.0.insert.ext858 = zext i32 %647 to i64
  %1639 = shl nuw nsw i64 %.sroa.256.0.insert.ext858, 1
  %1640 = add i64 %sink_3904, %1639
  %1641 = inttoptr i64 %1640 to i16 addrspace(4)*
  %1642 = addrspacecast i16 addrspace(4)* %1641 to i16 addrspace(1)*
  %1643 = load i16, i16 addrspace(1)* %1642, align 2
  %1644 = add i64 %sink_3891, %1639
  %1645 = inttoptr i64 %1644 to i16 addrspace(4)*
  %1646 = addrspacecast i16 addrspace(4)* %1645 to i16 addrspace(1)*
  %1647 = load i16, i16 addrspace(1)* %1646, align 2
  %1648 = zext i16 %1643 to i32
  %1649 = shl nuw i32 %1648, 16, !spirv.Decorations !921
  %1650 = bitcast i32 %1649 to float
  %1651 = zext i16 %1647 to i32
  %1652 = shl nuw i32 %1651, 16, !spirv.Decorations !921
  %1653 = bitcast i32 %1652 to float
  %1654 = fmul reassoc nsz arcp contract float %1650, %1653, !spirv.Decorations !898
  %1655 = fadd reassoc nsz arcp contract float %1654, %.sroa.246.1, !spirv.Decorations !898
  br label %.preheader.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %1638
  %.sroa.246.2 = phi float [ %1655, %1638 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %sink_sink_3870 = bitcast <2 x i32> %441 to i64
  %sink_sink_3846 = shl i64 %sink_sink_3870, 1
  %sink_3890 = add i64 %.in3822, %sink_sink_3846
  br i1 %305, label %1656, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
  br label %._crit_edge.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1656:                                             ; preds = %.preheader.13
  %.sroa.256.0.insert.ext863 = zext i32 %647 to i64
  %1657 = shl nuw nsw i64 %.sroa.256.0.insert.ext863, 1
  %1658 = add i64 %sink_3908, %1657
  %1659 = inttoptr i64 %1658 to i16 addrspace(4)*
  %1660 = addrspacecast i16 addrspace(4)* %1659 to i16 addrspace(1)*
  %1661 = load i16, i16 addrspace(1)* %1660, align 2
  %1662 = add i64 %sink_3890, %1657
  %1663 = inttoptr i64 %1662 to i16 addrspace(4)*
  %1664 = addrspacecast i16 addrspace(4)* %1663 to i16 addrspace(1)*
  %1665 = load i16, i16 addrspace(1)* %1664, align 2
  %1666 = zext i16 %1661 to i32
  %1667 = shl nuw i32 %1666, 16, !spirv.Decorations !921
  %1668 = bitcast i32 %1667 to float
  %1669 = zext i16 %1665 to i32
  %1670 = shl nuw i32 %1669, 16, !spirv.Decorations !921
  %1671 = bitcast i32 %1670 to float
  %1672 = fmul reassoc nsz arcp contract float %1668, %1671, !spirv.Decorations !898
  %1673 = fadd reassoc nsz arcp contract float %1672, %.sroa.58.1, !spirv.Decorations !898
  br label %._crit_edge.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %1656
  %.sroa.58.2 = phi float [ %1673, %1656 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  br i1 %308, label %1674, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1674:                                             ; preds = %._crit_edge.14
  %.sroa.256.0.insert.ext868 = zext i32 %647 to i64
  %1675 = shl nuw nsw i64 %.sroa.256.0.insert.ext868, 1
  %1676 = add i64 %sink_3906, %1675
  %1677 = inttoptr i64 %1676 to i16 addrspace(4)*
  %1678 = addrspacecast i16 addrspace(4)* %1677 to i16 addrspace(1)*
  %1679 = load i16, i16 addrspace(1)* %1678, align 2
  %1680 = add i64 %sink_3890, %1675
  %1681 = inttoptr i64 %1680 to i16 addrspace(4)*
  %1682 = addrspacecast i16 addrspace(4)* %1681 to i16 addrspace(1)*
  %1683 = load i16, i16 addrspace(1)* %1682, align 2
  %1684 = zext i16 %1679 to i32
  %1685 = shl nuw i32 %1684, 16, !spirv.Decorations !921
  %1686 = bitcast i32 %1685 to float
  %1687 = zext i16 %1683 to i32
  %1688 = shl nuw i32 %1687, 16, !spirv.Decorations !921
  %1689 = bitcast i32 %1688 to float
  %1690 = fmul reassoc nsz arcp contract float %1686, %1689, !spirv.Decorations !898
  %1691 = fadd reassoc nsz arcp contract float %1690, %.sroa.122.1, !spirv.Decorations !898
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %1674
  %.sroa.122.2 = phi float [ %1691, %1674 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  br i1 %311, label %1692, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1692:                                             ; preds = %._crit_edge.1.14
  %.sroa.256.0.insert.ext873 = zext i32 %647 to i64
  %1693 = shl nuw nsw i64 %.sroa.256.0.insert.ext873, 1
  %1694 = add i64 %sink_3905, %1693
  %1695 = inttoptr i64 %1694 to i16 addrspace(4)*
  %1696 = addrspacecast i16 addrspace(4)* %1695 to i16 addrspace(1)*
  %1697 = load i16, i16 addrspace(1)* %1696, align 2
  %1698 = add i64 %sink_3890, %1693
  %1699 = inttoptr i64 %1698 to i16 addrspace(4)*
  %1700 = addrspacecast i16 addrspace(4)* %1699 to i16 addrspace(1)*
  %1701 = load i16, i16 addrspace(1)* %1700, align 2
  %1702 = zext i16 %1697 to i32
  %1703 = shl nuw i32 %1702, 16, !spirv.Decorations !921
  %1704 = bitcast i32 %1703 to float
  %1705 = zext i16 %1701 to i32
  %1706 = shl nuw i32 %1705, 16, !spirv.Decorations !921
  %1707 = bitcast i32 %1706 to float
  %1708 = fmul reassoc nsz arcp contract float %1704, %1707, !spirv.Decorations !898
  %1709 = fadd reassoc nsz arcp contract float %1708, %.sroa.186.1, !spirv.Decorations !898
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %1692
  %.sroa.186.2 = phi float [ %1709, %1692 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  br i1 %314, label %1710, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
  br label %.preheader.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1710:                                             ; preds = %._crit_edge.2.14
  %.sroa.256.0.insert.ext878 = zext i32 %647 to i64
  %1711 = shl nuw nsw i64 %.sroa.256.0.insert.ext878, 1
  %1712 = add i64 %sink_3904, %1711
  %1713 = inttoptr i64 %1712 to i16 addrspace(4)*
  %1714 = addrspacecast i16 addrspace(4)* %1713 to i16 addrspace(1)*
  %1715 = load i16, i16 addrspace(1)* %1714, align 2
  %1716 = add i64 %sink_3890, %1711
  %1717 = inttoptr i64 %1716 to i16 addrspace(4)*
  %1718 = addrspacecast i16 addrspace(4)* %1717 to i16 addrspace(1)*
  %1719 = load i16, i16 addrspace(1)* %1718, align 2
  %1720 = zext i16 %1715 to i32
  %1721 = shl nuw i32 %1720, 16, !spirv.Decorations !921
  %1722 = bitcast i32 %1721 to float
  %1723 = zext i16 %1719 to i32
  %1724 = shl nuw i32 %1723, 16, !spirv.Decorations !921
  %1725 = bitcast i32 %1724 to float
  %1726 = fmul reassoc nsz arcp contract float %1722, %1725, !spirv.Decorations !898
  %1727 = fadd reassoc nsz arcp contract float %1726, %.sroa.250.1, !spirv.Decorations !898
  br label %.preheader.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %1710
  %.sroa.250.2 = phi float [ %1727, %1710 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %sink_sink_3869 = bitcast <2 x i32> %447 to i64
  %sink_sink_3845 = shl i64 %sink_sink_3869, 1
  %sink_3889 = add i64 %.in3822, %sink_sink_3845
  br i1 %318, label %1728, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
  br label %._crit_edge.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1728:                                             ; preds = %.preheader.14
  %.sroa.256.0.insert.ext883 = zext i32 %647 to i64
  %1729 = shl nuw nsw i64 %.sroa.256.0.insert.ext883, 1
  %1730 = add i64 %sink_3908, %1729
  %1731 = inttoptr i64 %1730 to i16 addrspace(4)*
  %1732 = addrspacecast i16 addrspace(4)* %1731 to i16 addrspace(1)*
  %1733 = load i16, i16 addrspace(1)* %1732, align 2
  %1734 = add i64 %sink_3889, %1729
  %1735 = inttoptr i64 %1734 to i16 addrspace(4)*
  %1736 = addrspacecast i16 addrspace(4)* %1735 to i16 addrspace(1)*
  %1737 = load i16, i16 addrspace(1)* %1736, align 2
  %1738 = zext i16 %1733 to i32
  %1739 = shl nuw i32 %1738, 16, !spirv.Decorations !921
  %1740 = bitcast i32 %1739 to float
  %1741 = zext i16 %1737 to i32
  %1742 = shl nuw i32 %1741, 16, !spirv.Decorations !921
  %1743 = bitcast i32 %1742 to float
  %1744 = fmul reassoc nsz arcp contract float %1740, %1743, !spirv.Decorations !898
  %1745 = fadd reassoc nsz arcp contract float %1744, %.sroa.62.1, !spirv.Decorations !898
  br label %._crit_edge.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %1728
  %.sroa.62.2 = phi float [ %1745, %1728 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  br i1 %321, label %1746, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1746:                                             ; preds = %._crit_edge.15
  %.sroa.256.0.insert.ext888 = zext i32 %647 to i64
  %1747 = shl nuw nsw i64 %.sroa.256.0.insert.ext888, 1
  %1748 = add i64 %sink_3906, %1747
  %1749 = inttoptr i64 %1748 to i16 addrspace(4)*
  %1750 = addrspacecast i16 addrspace(4)* %1749 to i16 addrspace(1)*
  %1751 = load i16, i16 addrspace(1)* %1750, align 2
  %1752 = add i64 %sink_3889, %1747
  %1753 = inttoptr i64 %1752 to i16 addrspace(4)*
  %1754 = addrspacecast i16 addrspace(4)* %1753 to i16 addrspace(1)*
  %1755 = load i16, i16 addrspace(1)* %1754, align 2
  %1756 = zext i16 %1751 to i32
  %1757 = shl nuw i32 %1756, 16, !spirv.Decorations !921
  %1758 = bitcast i32 %1757 to float
  %1759 = zext i16 %1755 to i32
  %1760 = shl nuw i32 %1759, 16, !spirv.Decorations !921
  %1761 = bitcast i32 %1760 to float
  %1762 = fmul reassoc nsz arcp contract float %1758, %1761, !spirv.Decorations !898
  %1763 = fadd reassoc nsz arcp contract float %1762, %.sroa.126.1, !spirv.Decorations !898
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %1746
  %.sroa.126.2 = phi float [ %1763, %1746 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  br i1 %324, label %1764, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1764:                                             ; preds = %._crit_edge.1.15
  %.sroa.256.0.insert.ext893 = zext i32 %647 to i64
  %1765 = shl nuw nsw i64 %.sroa.256.0.insert.ext893, 1
  %1766 = add i64 %sink_3905, %1765
  %1767 = inttoptr i64 %1766 to i16 addrspace(4)*
  %1768 = addrspacecast i16 addrspace(4)* %1767 to i16 addrspace(1)*
  %1769 = load i16, i16 addrspace(1)* %1768, align 2
  %1770 = add i64 %sink_3889, %1765
  %1771 = inttoptr i64 %1770 to i16 addrspace(4)*
  %1772 = addrspacecast i16 addrspace(4)* %1771 to i16 addrspace(1)*
  %1773 = load i16, i16 addrspace(1)* %1772, align 2
  %1774 = zext i16 %1769 to i32
  %1775 = shl nuw i32 %1774, 16, !spirv.Decorations !921
  %1776 = bitcast i32 %1775 to float
  %1777 = zext i16 %1773 to i32
  %1778 = shl nuw i32 %1777, 16, !spirv.Decorations !921
  %1779 = bitcast i32 %1778 to float
  %1780 = fmul reassoc nsz arcp contract float %1776, %1779, !spirv.Decorations !898
  %1781 = fadd reassoc nsz arcp contract float %1780, %.sroa.190.1, !spirv.Decorations !898
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %1764
  %.sroa.190.2 = phi float [ %1781, %1764 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  br i1 %327, label %1782, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
  br label %.preheader.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1782:                                             ; preds = %._crit_edge.2.15
  %.sroa.256.0.insert.ext898 = zext i32 %647 to i64
  %1783 = shl nuw nsw i64 %.sroa.256.0.insert.ext898, 1
  %1784 = add i64 %sink_3904, %1783
  %1785 = inttoptr i64 %1784 to i16 addrspace(4)*
  %1786 = addrspacecast i16 addrspace(4)* %1785 to i16 addrspace(1)*
  %1787 = load i16, i16 addrspace(1)* %1786, align 2
  %1788 = add i64 %sink_3889, %1783
  %1789 = inttoptr i64 %1788 to i16 addrspace(4)*
  %1790 = addrspacecast i16 addrspace(4)* %1789 to i16 addrspace(1)*
  %1791 = load i16, i16 addrspace(1)* %1790, align 2
  %1792 = zext i16 %1787 to i32
  %1793 = shl nuw i32 %1792, 16, !spirv.Decorations !921
  %1794 = bitcast i32 %1793 to float
  %1795 = zext i16 %1791 to i32
  %1796 = shl nuw i32 %1795, 16, !spirv.Decorations !921
  %1797 = bitcast i32 %1796 to float
  %1798 = fmul reassoc nsz arcp contract float %1794, %1797, !spirv.Decorations !898
  %1799 = fadd reassoc nsz arcp contract float %1798, %.sroa.254.1, !spirv.Decorations !898
  br label %.preheader.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %1782
  %.sroa.254.2 = phi float [ %1799, %1782 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %1800 = add nuw nsw i32 %647, 1, !spirv.Decorations !923
  %1801 = icmp slt i32 %1800, %const_reg_dword2
  br i1 %1801, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.15..preheader.preheader_crit_edge:     ; preds = %.preheader.15
  br label %.preheader.preheader, !stats.blockFrequency.digits !924, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.15
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
  br label %.preheader1.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
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
  %sink_3868 = bitcast <2 x i32> %461 to i64
  %sink_3844 = shl i64 %sink_3868, 2
  %sink_3843 = shl nsw i64 %454, 2
  br i1 %120, label %1802, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
  br label %._crit_edge70, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1802:                                             ; preds = %.preheader1.preheader
  %1803 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !898
  br i1 %81, label %1808, label %1804, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1804:                                             ; preds = %1802
  %1805 = add i64 %.in, %456
  %1806 = inttoptr i64 %1805 to float addrspace(4)*
  %1807 = addrspacecast float addrspace(4)* %1806 to float addrspace(1)*
  store float %1803, float addrspace(1)* %1807, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1808:                                             ; preds = %1802
  %1809 = add i64 %.in3821, %sink_3844
  %1810 = add i64 %1809, %sink_3843
  %1811 = inttoptr i64 %1810 to float addrspace(4)*
  %1812 = addrspacecast float addrspace(4)* %1811 to float addrspace(1)*
  %1813 = load float, float addrspace(1)* %1812, align 4
  %1814 = fmul reassoc nsz arcp contract float %1813, %4, !spirv.Decorations !898
  %1815 = fadd reassoc nsz arcp contract float %1803, %1814, !spirv.Decorations !898
  %1816 = add i64 %.in, %456
  %1817 = inttoptr i64 %1816 to float addrspace(4)*
  %1818 = addrspacecast float addrspace(4)* %1817 to float addrspace(1)*
  store float %1815, float addrspace(1)* %1818, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %1804, %1808
  %sink_3867 = bitcast <2 x i32> %474 to i64
  %sink_3842 = shl i64 %sink_3867, 2
  br i1 %124, label %1819, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
  br label %._crit_edge70.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1819:                                             ; preds = %._crit_edge70
  %1820 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !898
  br i1 %81, label %1825, label %1821, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1821:                                             ; preds = %1819
  %1822 = add i64 %.in, %469
  %1823 = inttoptr i64 %1822 to float addrspace(4)*
  %1824 = addrspacecast float addrspace(4)* %1823 to float addrspace(1)*
  store float %1820, float addrspace(1)* %1824, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1825:                                             ; preds = %1819
  %1826 = add i64 %.in3821, %sink_3842
  %1827 = add i64 %1826, %sink_3843
  %1828 = inttoptr i64 %1827 to float addrspace(4)*
  %1829 = addrspacecast float addrspace(4)* %1828 to float addrspace(1)*
  %1830 = load float, float addrspace(1)* %1829, align 4
  %1831 = fmul reassoc nsz arcp contract float %1830, %4, !spirv.Decorations !898
  %1832 = fadd reassoc nsz arcp contract float %1820, %1831, !spirv.Decorations !898
  %1833 = add i64 %.in, %469
  %1834 = inttoptr i64 %1833 to float addrspace(4)*
  %1835 = addrspacecast float addrspace(4)* %1834 to float addrspace(1)*
  store float %1832, float addrspace(1)* %1835, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %1825, %1821
  %sink_3866 = bitcast <2 x i32> %487 to i64
  %sink_3841 = shl i64 %sink_3866, 2
  br i1 %128, label %1836, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
  br label %._crit_edge70.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1836:                                             ; preds = %._crit_edge70.1
  %1837 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !898
  br i1 %81, label %1842, label %1838, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1838:                                             ; preds = %1836
  %1839 = add i64 %.in, %482
  %1840 = inttoptr i64 %1839 to float addrspace(4)*
  %1841 = addrspacecast float addrspace(4)* %1840 to float addrspace(1)*
  store float %1837, float addrspace(1)* %1841, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1842:                                             ; preds = %1836
  %1843 = add i64 %.in3821, %sink_3841
  %1844 = add i64 %1843, %sink_3843
  %1845 = inttoptr i64 %1844 to float addrspace(4)*
  %1846 = addrspacecast float addrspace(4)* %1845 to float addrspace(1)*
  %1847 = load float, float addrspace(1)* %1846, align 4
  %1848 = fmul reassoc nsz arcp contract float %1847, %4, !spirv.Decorations !898
  %1849 = fadd reassoc nsz arcp contract float %1837, %1848, !spirv.Decorations !898
  %1850 = add i64 %.in, %482
  %1851 = inttoptr i64 %1850 to float addrspace(4)*
  %1852 = addrspacecast float addrspace(4)* %1851 to float addrspace(1)*
  store float %1849, float addrspace(1)* %1852, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %1842, %1838
  %sink_3865 = bitcast <2 x i32> %500 to i64
  %sink_3840 = shl i64 %sink_3865, 2
  br i1 %132, label %1853, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
  br label %.preheader1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1853:                                             ; preds = %._crit_edge70.2
  %1854 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !898
  br i1 %81, label %1859, label %1855, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1855:                                             ; preds = %1853
  %1856 = add i64 %.in, %495
  %1857 = inttoptr i64 %1856 to float addrspace(4)*
  %1858 = addrspacecast float addrspace(4)* %1857 to float addrspace(1)*
  store float %1854, float addrspace(1)* %1858, align 4
  br label %.preheader1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1859:                                             ; preds = %1853
  %1860 = add i64 %.in3821, %sink_3840
  %1861 = add i64 %1860, %sink_3843
  %1862 = inttoptr i64 %1861 to float addrspace(4)*
  %1863 = addrspacecast float addrspace(4)* %1862 to float addrspace(1)*
  %1864 = load float, float addrspace(1)* %1863, align 4
  %1865 = fmul reassoc nsz arcp contract float %1864, %4, !spirv.Decorations !898
  %1866 = fadd reassoc nsz arcp contract float %1854, %1865, !spirv.Decorations !898
  %1867 = add i64 %.in, %495
  %1868 = inttoptr i64 %1867 to float addrspace(4)*
  %1869 = addrspacecast float addrspace(4)* %1868 to float addrspace(1)*
  store float %1866, float addrspace(1)* %1869, align 4
  br label %.preheader1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %1859, %1855
  %sink_3839 = shl nsw i64 %501, 2
  br i1 %136, label %1870, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
  br label %._crit_edge70.176, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1870:                                             ; preds = %.preheader1
  %1871 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !898
  br i1 %81, label %1876, label %1872, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1872:                                             ; preds = %1870
  %1873 = add i64 %.in, %503
  %1874 = inttoptr i64 %1873 to float addrspace(4)*
  %1875 = addrspacecast float addrspace(4)* %1874 to float addrspace(1)*
  store float %1871, float addrspace(1)* %1875, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1876:                                             ; preds = %1870
  %1877 = add i64 %.in3821, %sink_3844
  %1878 = add i64 %1877, %sink_3839
  %1879 = inttoptr i64 %1878 to float addrspace(4)*
  %1880 = addrspacecast float addrspace(4)* %1879 to float addrspace(1)*
  %1881 = load float, float addrspace(1)* %1880, align 4
  %1882 = fmul reassoc nsz arcp contract float %1881, %4, !spirv.Decorations !898
  %1883 = fadd reassoc nsz arcp contract float %1871, %1882, !spirv.Decorations !898
  %1884 = add i64 %.in, %503
  %1885 = inttoptr i64 %1884 to float addrspace(4)*
  %1886 = addrspacecast float addrspace(4)* %1885 to float addrspace(1)*
  store float %1883, float addrspace(1)* %1886, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %1876, %1872
  br i1 %139, label %1887, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1887:                                             ; preds = %._crit_edge70.176
  %1888 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !898
  br i1 %81, label %1893, label %1889, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1889:                                             ; preds = %1887
  %1890 = add i64 %.in, %505
  %1891 = inttoptr i64 %1890 to float addrspace(4)*
  %1892 = addrspacecast float addrspace(4)* %1891 to float addrspace(1)*
  store float %1888, float addrspace(1)* %1892, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1893:                                             ; preds = %1887
  %1894 = add i64 %.in3821, %sink_3842
  %1895 = add i64 %1894, %sink_3839
  %1896 = inttoptr i64 %1895 to float addrspace(4)*
  %1897 = addrspacecast float addrspace(4)* %1896 to float addrspace(1)*
  %1898 = load float, float addrspace(1)* %1897, align 4
  %1899 = fmul reassoc nsz arcp contract float %1898, %4, !spirv.Decorations !898
  %1900 = fadd reassoc nsz arcp contract float %1888, %1899, !spirv.Decorations !898
  %1901 = add i64 %.in, %505
  %1902 = inttoptr i64 %1901 to float addrspace(4)*
  %1903 = addrspacecast float addrspace(4)* %1902 to float addrspace(1)*
  store float %1900, float addrspace(1)* %1903, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %1893, %1889
  br i1 %142, label %1904, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1904:                                             ; preds = %._crit_edge70.1.1
  %1905 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !898
  br i1 %81, label %1910, label %1906, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1906:                                             ; preds = %1904
  %1907 = add i64 %.in, %507
  %1908 = inttoptr i64 %1907 to float addrspace(4)*
  %1909 = addrspacecast float addrspace(4)* %1908 to float addrspace(1)*
  store float %1905, float addrspace(1)* %1909, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1910:                                             ; preds = %1904
  %1911 = add i64 %.in3821, %sink_3841
  %1912 = add i64 %1911, %sink_3839
  %1913 = inttoptr i64 %1912 to float addrspace(4)*
  %1914 = addrspacecast float addrspace(4)* %1913 to float addrspace(1)*
  %1915 = load float, float addrspace(1)* %1914, align 4
  %1916 = fmul reassoc nsz arcp contract float %1915, %4, !spirv.Decorations !898
  %1917 = fadd reassoc nsz arcp contract float %1905, %1916, !spirv.Decorations !898
  %1918 = add i64 %.in, %507
  %1919 = inttoptr i64 %1918 to float addrspace(4)*
  %1920 = addrspacecast float addrspace(4)* %1919 to float addrspace(1)*
  store float %1917, float addrspace(1)* %1920, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %1910, %1906
  br i1 %145, label %1921, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
  br label %.preheader1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1921:                                             ; preds = %._crit_edge70.2.1
  %1922 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !898
  br i1 %81, label %1927, label %1923, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1923:                                             ; preds = %1921
  %1924 = add i64 %.in, %509
  %1925 = inttoptr i64 %1924 to float addrspace(4)*
  %1926 = addrspacecast float addrspace(4)* %1925 to float addrspace(1)*
  store float %1922, float addrspace(1)* %1926, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1927:                                             ; preds = %1921
  %1928 = add i64 %.in3821, %sink_3840
  %1929 = add i64 %1928, %sink_3839
  %1930 = inttoptr i64 %1929 to float addrspace(4)*
  %1931 = addrspacecast float addrspace(4)* %1930 to float addrspace(1)*
  %1932 = load float, float addrspace(1)* %1931, align 4
  %1933 = fmul reassoc nsz arcp contract float %1932, %4, !spirv.Decorations !898
  %1934 = fadd reassoc nsz arcp contract float %1922, %1933, !spirv.Decorations !898
  %1935 = add i64 %.in, %509
  %1936 = inttoptr i64 %1935 to float addrspace(4)*
  %1937 = addrspacecast float addrspace(4)* %1936 to float addrspace(1)*
  store float %1934, float addrspace(1)* %1937, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %1927, %1923
  %sink_3838 = shl nsw i64 %510, 2
  br i1 %149, label %1938, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
  br label %._crit_edge70.277, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1938:                                             ; preds = %.preheader1.1
  %1939 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !898
  br i1 %81, label %1944, label %1940, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1940:                                             ; preds = %1938
  %1941 = add i64 %.in, %512
  %1942 = inttoptr i64 %1941 to float addrspace(4)*
  %1943 = addrspacecast float addrspace(4)* %1942 to float addrspace(1)*
  store float %1939, float addrspace(1)* %1943, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1944:                                             ; preds = %1938
  %1945 = add i64 %.in3821, %sink_3844
  %1946 = add i64 %1945, %sink_3838
  %1947 = inttoptr i64 %1946 to float addrspace(4)*
  %1948 = addrspacecast float addrspace(4)* %1947 to float addrspace(1)*
  %1949 = load float, float addrspace(1)* %1948, align 4
  %1950 = fmul reassoc nsz arcp contract float %1949, %4, !spirv.Decorations !898
  %1951 = fadd reassoc nsz arcp contract float %1939, %1950, !spirv.Decorations !898
  %1952 = add i64 %.in, %512
  %1953 = inttoptr i64 %1952 to float addrspace(4)*
  %1954 = addrspacecast float addrspace(4)* %1953 to float addrspace(1)*
  store float %1951, float addrspace(1)* %1954, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %1944, %1940
  br i1 %152, label %1955, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1955:                                             ; preds = %._crit_edge70.277
  %1956 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !898
  br i1 %81, label %1961, label %1957, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1957:                                             ; preds = %1955
  %1958 = add i64 %.in, %514
  %1959 = inttoptr i64 %1958 to float addrspace(4)*
  %1960 = addrspacecast float addrspace(4)* %1959 to float addrspace(1)*
  store float %1956, float addrspace(1)* %1960, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1961:                                             ; preds = %1955
  %1962 = add i64 %.in3821, %sink_3842
  %1963 = add i64 %1962, %sink_3838
  %1964 = inttoptr i64 %1963 to float addrspace(4)*
  %1965 = addrspacecast float addrspace(4)* %1964 to float addrspace(1)*
  %1966 = load float, float addrspace(1)* %1965, align 4
  %1967 = fmul reassoc nsz arcp contract float %1966, %4, !spirv.Decorations !898
  %1968 = fadd reassoc nsz arcp contract float %1956, %1967, !spirv.Decorations !898
  %1969 = add i64 %.in, %514
  %1970 = inttoptr i64 %1969 to float addrspace(4)*
  %1971 = addrspacecast float addrspace(4)* %1970 to float addrspace(1)*
  store float %1968, float addrspace(1)* %1971, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %1961, %1957
  br i1 %155, label %1972, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1972:                                             ; preds = %._crit_edge70.1.2
  %1973 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !898
  br i1 %81, label %1978, label %1974, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1974:                                             ; preds = %1972
  %1975 = add i64 %.in, %516
  %1976 = inttoptr i64 %1975 to float addrspace(4)*
  %1977 = addrspacecast float addrspace(4)* %1976 to float addrspace(1)*
  store float %1973, float addrspace(1)* %1977, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1978:                                             ; preds = %1972
  %1979 = add i64 %.in3821, %sink_3841
  %1980 = add i64 %1979, %sink_3838
  %1981 = inttoptr i64 %1980 to float addrspace(4)*
  %1982 = addrspacecast float addrspace(4)* %1981 to float addrspace(1)*
  %1983 = load float, float addrspace(1)* %1982, align 4
  %1984 = fmul reassoc nsz arcp contract float %1983, %4, !spirv.Decorations !898
  %1985 = fadd reassoc nsz arcp contract float %1973, %1984, !spirv.Decorations !898
  %1986 = add i64 %.in, %516
  %1987 = inttoptr i64 %1986 to float addrspace(4)*
  %1988 = addrspacecast float addrspace(4)* %1987 to float addrspace(1)*
  store float %1985, float addrspace(1)* %1988, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %1978, %1974
  br i1 %158, label %1989, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
  br label %.preheader1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1989:                                             ; preds = %._crit_edge70.2.2
  %1990 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !898
  br i1 %81, label %1995, label %1991, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1991:                                             ; preds = %1989
  %1992 = add i64 %.in, %518
  %1993 = inttoptr i64 %1992 to float addrspace(4)*
  %1994 = addrspacecast float addrspace(4)* %1993 to float addrspace(1)*
  store float %1990, float addrspace(1)* %1994, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1995:                                             ; preds = %1989
  %1996 = add i64 %.in3821, %sink_3840
  %1997 = add i64 %1996, %sink_3838
  %1998 = inttoptr i64 %1997 to float addrspace(4)*
  %1999 = addrspacecast float addrspace(4)* %1998 to float addrspace(1)*
  %2000 = load float, float addrspace(1)* %1999, align 4
  %2001 = fmul reassoc nsz arcp contract float %2000, %4, !spirv.Decorations !898
  %2002 = fadd reassoc nsz arcp contract float %1990, %2001, !spirv.Decorations !898
  %2003 = add i64 %.in, %518
  %2004 = inttoptr i64 %2003 to float addrspace(4)*
  %2005 = addrspacecast float addrspace(4)* %2004 to float addrspace(1)*
  store float %2002, float addrspace(1)* %2005, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %1995, %1991
  %sink_3837 = shl nsw i64 %519, 2
  br i1 %162, label %2006, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
  br label %._crit_edge70.378, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2006:                                             ; preds = %.preheader1.2
  %2007 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !898
  br i1 %81, label %2012, label %2008, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2008:                                             ; preds = %2006
  %2009 = add i64 %.in, %521
  %2010 = inttoptr i64 %2009 to float addrspace(4)*
  %2011 = addrspacecast float addrspace(4)* %2010 to float addrspace(1)*
  store float %2007, float addrspace(1)* %2011, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2012:                                             ; preds = %2006
  %2013 = add i64 %.in3821, %sink_3844
  %2014 = add i64 %2013, %sink_3837
  %2015 = inttoptr i64 %2014 to float addrspace(4)*
  %2016 = addrspacecast float addrspace(4)* %2015 to float addrspace(1)*
  %2017 = load float, float addrspace(1)* %2016, align 4
  %2018 = fmul reassoc nsz arcp contract float %2017, %4, !spirv.Decorations !898
  %2019 = fadd reassoc nsz arcp contract float %2007, %2018, !spirv.Decorations !898
  %2020 = add i64 %.in, %521
  %2021 = inttoptr i64 %2020 to float addrspace(4)*
  %2022 = addrspacecast float addrspace(4)* %2021 to float addrspace(1)*
  store float %2019, float addrspace(1)* %2022, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %2012, %2008
  br i1 %165, label %2023, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2023:                                             ; preds = %._crit_edge70.378
  %2024 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !898
  br i1 %81, label %2029, label %2025, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2025:                                             ; preds = %2023
  %2026 = add i64 %.in, %523
  %2027 = inttoptr i64 %2026 to float addrspace(4)*
  %2028 = addrspacecast float addrspace(4)* %2027 to float addrspace(1)*
  store float %2024, float addrspace(1)* %2028, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2029:                                             ; preds = %2023
  %2030 = add i64 %.in3821, %sink_3842
  %2031 = add i64 %2030, %sink_3837
  %2032 = inttoptr i64 %2031 to float addrspace(4)*
  %2033 = addrspacecast float addrspace(4)* %2032 to float addrspace(1)*
  %2034 = load float, float addrspace(1)* %2033, align 4
  %2035 = fmul reassoc nsz arcp contract float %2034, %4, !spirv.Decorations !898
  %2036 = fadd reassoc nsz arcp contract float %2024, %2035, !spirv.Decorations !898
  %2037 = add i64 %.in, %523
  %2038 = inttoptr i64 %2037 to float addrspace(4)*
  %2039 = addrspacecast float addrspace(4)* %2038 to float addrspace(1)*
  store float %2036, float addrspace(1)* %2039, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %2029, %2025
  br i1 %168, label %2040, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2040:                                             ; preds = %._crit_edge70.1.3
  %2041 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !898
  br i1 %81, label %2046, label %2042, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2042:                                             ; preds = %2040
  %2043 = add i64 %.in, %525
  %2044 = inttoptr i64 %2043 to float addrspace(4)*
  %2045 = addrspacecast float addrspace(4)* %2044 to float addrspace(1)*
  store float %2041, float addrspace(1)* %2045, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2046:                                             ; preds = %2040
  %2047 = add i64 %.in3821, %sink_3841
  %2048 = add i64 %2047, %sink_3837
  %2049 = inttoptr i64 %2048 to float addrspace(4)*
  %2050 = addrspacecast float addrspace(4)* %2049 to float addrspace(1)*
  %2051 = load float, float addrspace(1)* %2050, align 4
  %2052 = fmul reassoc nsz arcp contract float %2051, %4, !spirv.Decorations !898
  %2053 = fadd reassoc nsz arcp contract float %2041, %2052, !spirv.Decorations !898
  %2054 = add i64 %.in, %525
  %2055 = inttoptr i64 %2054 to float addrspace(4)*
  %2056 = addrspacecast float addrspace(4)* %2055 to float addrspace(1)*
  store float %2053, float addrspace(1)* %2056, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %2046, %2042
  br i1 %171, label %2057, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
  br label %.preheader1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2057:                                             ; preds = %._crit_edge70.2.3
  %2058 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !898
  br i1 %81, label %2063, label %2059, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2059:                                             ; preds = %2057
  %2060 = add i64 %.in, %527
  %2061 = inttoptr i64 %2060 to float addrspace(4)*
  %2062 = addrspacecast float addrspace(4)* %2061 to float addrspace(1)*
  store float %2058, float addrspace(1)* %2062, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2063:                                             ; preds = %2057
  %2064 = add i64 %.in3821, %sink_3840
  %2065 = add i64 %2064, %sink_3837
  %2066 = inttoptr i64 %2065 to float addrspace(4)*
  %2067 = addrspacecast float addrspace(4)* %2066 to float addrspace(1)*
  %2068 = load float, float addrspace(1)* %2067, align 4
  %2069 = fmul reassoc nsz arcp contract float %2068, %4, !spirv.Decorations !898
  %2070 = fadd reassoc nsz arcp contract float %2058, %2069, !spirv.Decorations !898
  %2071 = add i64 %.in, %527
  %2072 = inttoptr i64 %2071 to float addrspace(4)*
  %2073 = addrspacecast float addrspace(4)* %2072 to float addrspace(1)*
  store float %2070, float addrspace(1)* %2073, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %2063, %2059
  %sink_3836 = shl nsw i64 %528, 2
  br i1 %175, label %2074, label %.preheader1.3.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.3.._crit_edge70.4_crit_edge:          ; preds = %.preheader1.3
  br label %._crit_edge70.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2074:                                             ; preds = %.preheader1.3
  %2075 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !898
  br i1 %81, label %2080, label %2076, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2076:                                             ; preds = %2074
  %2077 = add i64 %.in, %530
  %2078 = inttoptr i64 %2077 to float addrspace(4)*
  %2079 = addrspacecast float addrspace(4)* %2078 to float addrspace(1)*
  store float %2075, float addrspace(1)* %2079, align 4
  br label %._crit_edge70.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2080:                                             ; preds = %2074
  %2081 = add i64 %.in3821, %sink_3844
  %2082 = add i64 %2081, %sink_3836
  %2083 = inttoptr i64 %2082 to float addrspace(4)*
  %2084 = addrspacecast float addrspace(4)* %2083 to float addrspace(1)*
  %2085 = load float, float addrspace(1)* %2084, align 4
  %2086 = fmul reassoc nsz arcp contract float %2085, %4, !spirv.Decorations !898
  %2087 = fadd reassoc nsz arcp contract float %2075, %2086, !spirv.Decorations !898
  %2088 = add i64 %.in, %530
  %2089 = inttoptr i64 %2088 to float addrspace(4)*
  %2090 = addrspacecast float addrspace(4)* %2089 to float addrspace(1)*
  store float %2087, float addrspace(1)* %2090, align 4
  br label %._crit_edge70.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.4:                                  ; preds = %.preheader1.3.._crit_edge70.4_crit_edge, %2080, %2076
  br i1 %178, label %2091, label %._crit_edge70.4.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.4.._crit_edge70.1.4_crit_edge:      ; preds = %._crit_edge70.4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2091:                                             ; preds = %._crit_edge70.4
  %2092 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !898
  br i1 %81, label %2097, label %2093, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2093:                                             ; preds = %2091
  %2094 = add i64 %.in, %532
  %2095 = inttoptr i64 %2094 to float addrspace(4)*
  %2096 = addrspacecast float addrspace(4)* %2095 to float addrspace(1)*
  store float %2092, float addrspace(1)* %2096, align 4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2097:                                             ; preds = %2091
  %2098 = add i64 %.in3821, %sink_3842
  %2099 = add i64 %2098, %sink_3836
  %2100 = inttoptr i64 %2099 to float addrspace(4)*
  %2101 = addrspacecast float addrspace(4)* %2100 to float addrspace(1)*
  %2102 = load float, float addrspace(1)* %2101, align 4
  %2103 = fmul reassoc nsz arcp contract float %2102, %4, !spirv.Decorations !898
  %2104 = fadd reassoc nsz arcp contract float %2092, %2103, !spirv.Decorations !898
  %2105 = add i64 %.in, %532
  %2106 = inttoptr i64 %2105 to float addrspace(4)*
  %2107 = addrspacecast float addrspace(4)* %2106 to float addrspace(1)*
  store float %2104, float addrspace(1)* %2107, align 4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4.._crit_edge70.1.4_crit_edge, %2097, %2093
  br i1 %181, label %2108, label %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.4.._crit_edge70.2.4_crit_edge:    ; preds = %._crit_edge70.1.4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2108:                                             ; preds = %._crit_edge70.1.4
  %2109 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !898
  br i1 %81, label %2114, label %2110, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2110:                                             ; preds = %2108
  %2111 = add i64 %.in, %534
  %2112 = inttoptr i64 %2111 to float addrspace(4)*
  %2113 = addrspacecast float addrspace(4)* %2112 to float addrspace(1)*
  store float %2109, float addrspace(1)* %2113, align 4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2114:                                             ; preds = %2108
  %2115 = add i64 %.in3821, %sink_3841
  %2116 = add i64 %2115, %sink_3836
  %2117 = inttoptr i64 %2116 to float addrspace(4)*
  %2118 = addrspacecast float addrspace(4)* %2117 to float addrspace(1)*
  %2119 = load float, float addrspace(1)* %2118, align 4
  %2120 = fmul reassoc nsz arcp contract float %2119, %4, !spirv.Decorations !898
  %2121 = fadd reassoc nsz arcp contract float %2109, %2120, !spirv.Decorations !898
  %2122 = add i64 %.in, %534
  %2123 = inttoptr i64 %2122 to float addrspace(4)*
  %2124 = addrspacecast float addrspace(4)* %2123 to float addrspace(1)*
  store float %2121, float addrspace(1)* %2124, align 4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, %2114, %2110
  br i1 %184, label %2125, label %._crit_edge70.2.4..preheader1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.4..preheader1.4_crit_edge:        ; preds = %._crit_edge70.2.4
  br label %.preheader1.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2125:                                             ; preds = %._crit_edge70.2.4
  %2126 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !898
  br i1 %81, label %2131, label %2127, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2127:                                             ; preds = %2125
  %2128 = add i64 %.in, %536
  %2129 = inttoptr i64 %2128 to float addrspace(4)*
  %2130 = addrspacecast float addrspace(4)* %2129 to float addrspace(1)*
  store float %2126, float addrspace(1)* %2130, align 4
  br label %.preheader1.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2131:                                             ; preds = %2125
  %2132 = add i64 %.in3821, %sink_3840
  %2133 = add i64 %2132, %sink_3836
  %2134 = inttoptr i64 %2133 to float addrspace(4)*
  %2135 = addrspacecast float addrspace(4)* %2134 to float addrspace(1)*
  %2136 = load float, float addrspace(1)* %2135, align 4
  %2137 = fmul reassoc nsz arcp contract float %2136, %4, !spirv.Decorations !898
  %2138 = fadd reassoc nsz arcp contract float %2126, %2137, !spirv.Decorations !898
  %2139 = add i64 %.in, %536
  %2140 = inttoptr i64 %2139 to float addrspace(4)*
  %2141 = addrspacecast float addrspace(4)* %2140 to float addrspace(1)*
  store float %2138, float addrspace(1)* %2141, align 4
  br label %.preheader1.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.4:                                    ; preds = %._crit_edge70.2.4..preheader1.4_crit_edge, %2131, %2127
  %sink_3835 = shl nsw i64 %537, 2
  br i1 %188, label %2142, label %.preheader1.4.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.4.._crit_edge70.5_crit_edge:          ; preds = %.preheader1.4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2142:                                             ; preds = %.preheader1.4
  %2143 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !898
  br i1 %81, label %2148, label %2144, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2144:                                             ; preds = %2142
  %2145 = add i64 %.in, %539
  %2146 = inttoptr i64 %2145 to float addrspace(4)*
  %2147 = addrspacecast float addrspace(4)* %2146 to float addrspace(1)*
  store float %2143, float addrspace(1)* %2147, align 4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2148:                                             ; preds = %2142
  %2149 = add i64 %.in3821, %sink_3844
  %2150 = add i64 %2149, %sink_3835
  %2151 = inttoptr i64 %2150 to float addrspace(4)*
  %2152 = addrspacecast float addrspace(4)* %2151 to float addrspace(1)*
  %2153 = load float, float addrspace(1)* %2152, align 4
  %2154 = fmul reassoc nsz arcp contract float %2153, %4, !spirv.Decorations !898
  %2155 = fadd reassoc nsz arcp contract float %2143, %2154, !spirv.Decorations !898
  %2156 = add i64 %.in, %539
  %2157 = inttoptr i64 %2156 to float addrspace(4)*
  %2158 = addrspacecast float addrspace(4)* %2157 to float addrspace(1)*
  store float %2155, float addrspace(1)* %2158, align 4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.5:                                  ; preds = %.preheader1.4.._crit_edge70.5_crit_edge, %2148, %2144
  br i1 %191, label %2159, label %._crit_edge70.5.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.5.._crit_edge70.1.5_crit_edge:      ; preds = %._crit_edge70.5
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2159:                                             ; preds = %._crit_edge70.5
  %2160 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !898
  br i1 %81, label %2165, label %2161, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2161:                                             ; preds = %2159
  %2162 = add i64 %.in, %541
  %2163 = inttoptr i64 %2162 to float addrspace(4)*
  %2164 = addrspacecast float addrspace(4)* %2163 to float addrspace(1)*
  store float %2160, float addrspace(1)* %2164, align 4
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2165:                                             ; preds = %2159
  %2166 = add i64 %.in3821, %sink_3842
  %2167 = add i64 %2166, %sink_3835
  %2168 = inttoptr i64 %2167 to float addrspace(4)*
  %2169 = addrspacecast float addrspace(4)* %2168 to float addrspace(1)*
  %2170 = load float, float addrspace(1)* %2169, align 4
  %2171 = fmul reassoc nsz arcp contract float %2170, %4, !spirv.Decorations !898
  %2172 = fadd reassoc nsz arcp contract float %2160, %2171, !spirv.Decorations !898
  %2173 = add i64 %.in, %541
  %2174 = inttoptr i64 %2173 to float addrspace(4)*
  %2175 = addrspacecast float addrspace(4)* %2174 to float addrspace(1)*
  store float %2172, float addrspace(1)* %2175, align 4
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5.._crit_edge70.1.5_crit_edge, %2165, %2161
  br i1 %194, label %2176, label %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.5.._crit_edge70.2.5_crit_edge:    ; preds = %._crit_edge70.1.5
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2176:                                             ; preds = %._crit_edge70.1.5
  %2177 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !898
  br i1 %81, label %2182, label %2178, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2178:                                             ; preds = %2176
  %2179 = add i64 %.in, %543
  %2180 = inttoptr i64 %2179 to float addrspace(4)*
  %2181 = addrspacecast float addrspace(4)* %2180 to float addrspace(1)*
  store float %2177, float addrspace(1)* %2181, align 4
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2182:                                             ; preds = %2176
  %2183 = add i64 %.in3821, %sink_3841
  %2184 = add i64 %2183, %sink_3835
  %2185 = inttoptr i64 %2184 to float addrspace(4)*
  %2186 = addrspacecast float addrspace(4)* %2185 to float addrspace(1)*
  %2187 = load float, float addrspace(1)* %2186, align 4
  %2188 = fmul reassoc nsz arcp contract float %2187, %4, !spirv.Decorations !898
  %2189 = fadd reassoc nsz arcp contract float %2177, %2188, !spirv.Decorations !898
  %2190 = add i64 %.in, %543
  %2191 = inttoptr i64 %2190 to float addrspace(4)*
  %2192 = addrspacecast float addrspace(4)* %2191 to float addrspace(1)*
  store float %2189, float addrspace(1)* %2192, align 4
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, %2182, %2178
  br i1 %197, label %2193, label %._crit_edge70.2.5..preheader1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.5..preheader1.5_crit_edge:        ; preds = %._crit_edge70.2.5
  br label %.preheader1.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2193:                                             ; preds = %._crit_edge70.2.5
  %2194 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !898
  br i1 %81, label %2199, label %2195, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2195:                                             ; preds = %2193
  %2196 = add i64 %.in, %545
  %2197 = inttoptr i64 %2196 to float addrspace(4)*
  %2198 = addrspacecast float addrspace(4)* %2197 to float addrspace(1)*
  store float %2194, float addrspace(1)* %2198, align 4
  br label %.preheader1.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2199:                                             ; preds = %2193
  %2200 = add i64 %.in3821, %sink_3840
  %2201 = add i64 %2200, %sink_3835
  %2202 = inttoptr i64 %2201 to float addrspace(4)*
  %2203 = addrspacecast float addrspace(4)* %2202 to float addrspace(1)*
  %2204 = load float, float addrspace(1)* %2203, align 4
  %2205 = fmul reassoc nsz arcp contract float %2204, %4, !spirv.Decorations !898
  %2206 = fadd reassoc nsz arcp contract float %2194, %2205, !spirv.Decorations !898
  %2207 = add i64 %.in, %545
  %2208 = inttoptr i64 %2207 to float addrspace(4)*
  %2209 = addrspacecast float addrspace(4)* %2208 to float addrspace(1)*
  store float %2206, float addrspace(1)* %2209, align 4
  br label %.preheader1.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.5:                                    ; preds = %._crit_edge70.2.5..preheader1.5_crit_edge, %2199, %2195
  %sink_3834 = shl nsw i64 %546, 2
  br i1 %201, label %2210, label %.preheader1.5.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.5.._crit_edge70.6_crit_edge:          ; preds = %.preheader1.5
  br label %._crit_edge70.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2210:                                             ; preds = %.preheader1.5
  %2211 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !898
  br i1 %81, label %2216, label %2212, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2212:                                             ; preds = %2210
  %2213 = add i64 %.in, %548
  %2214 = inttoptr i64 %2213 to float addrspace(4)*
  %2215 = addrspacecast float addrspace(4)* %2214 to float addrspace(1)*
  store float %2211, float addrspace(1)* %2215, align 4
  br label %._crit_edge70.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2216:                                             ; preds = %2210
  %2217 = add i64 %.in3821, %sink_3844
  %2218 = add i64 %2217, %sink_3834
  %2219 = inttoptr i64 %2218 to float addrspace(4)*
  %2220 = addrspacecast float addrspace(4)* %2219 to float addrspace(1)*
  %2221 = load float, float addrspace(1)* %2220, align 4
  %2222 = fmul reassoc nsz arcp contract float %2221, %4, !spirv.Decorations !898
  %2223 = fadd reassoc nsz arcp contract float %2211, %2222, !spirv.Decorations !898
  %2224 = add i64 %.in, %548
  %2225 = inttoptr i64 %2224 to float addrspace(4)*
  %2226 = addrspacecast float addrspace(4)* %2225 to float addrspace(1)*
  store float %2223, float addrspace(1)* %2226, align 4
  br label %._crit_edge70.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.6:                                  ; preds = %.preheader1.5.._crit_edge70.6_crit_edge, %2216, %2212
  br i1 %204, label %2227, label %._crit_edge70.6.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.6.._crit_edge70.1.6_crit_edge:      ; preds = %._crit_edge70.6
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2227:                                             ; preds = %._crit_edge70.6
  %2228 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !898
  br i1 %81, label %2233, label %2229, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2229:                                             ; preds = %2227
  %2230 = add i64 %.in, %550
  %2231 = inttoptr i64 %2230 to float addrspace(4)*
  %2232 = addrspacecast float addrspace(4)* %2231 to float addrspace(1)*
  store float %2228, float addrspace(1)* %2232, align 4
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2233:                                             ; preds = %2227
  %2234 = add i64 %.in3821, %sink_3842
  %2235 = add i64 %2234, %sink_3834
  %2236 = inttoptr i64 %2235 to float addrspace(4)*
  %2237 = addrspacecast float addrspace(4)* %2236 to float addrspace(1)*
  %2238 = load float, float addrspace(1)* %2237, align 4
  %2239 = fmul reassoc nsz arcp contract float %2238, %4, !spirv.Decorations !898
  %2240 = fadd reassoc nsz arcp contract float %2228, %2239, !spirv.Decorations !898
  %2241 = add i64 %.in, %550
  %2242 = inttoptr i64 %2241 to float addrspace(4)*
  %2243 = addrspacecast float addrspace(4)* %2242 to float addrspace(1)*
  store float %2240, float addrspace(1)* %2243, align 4
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6.._crit_edge70.1.6_crit_edge, %2233, %2229
  br i1 %207, label %2244, label %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.6.._crit_edge70.2.6_crit_edge:    ; preds = %._crit_edge70.1.6
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2244:                                             ; preds = %._crit_edge70.1.6
  %2245 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !898
  br i1 %81, label %2250, label %2246, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2246:                                             ; preds = %2244
  %2247 = add i64 %.in, %552
  %2248 = inttoptr i64 %2247 to float addrspace(4)*
  %2249 = addrspacecast float addrspace(4)* %2248 to float addrspace(1)*
  store float %2245, float addrspace(1)* %2249, align 4
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2250:                                             ; preds = %2244
  %2251 = add i64 %.in3821, %sink_3841
  %2252 = add i64 %2251, %sink_3834
  %2253 = inttoptr i64 %2252 to float addrspace(4)*
  %2254 = addrspacecast float addrspace(4)* %2253 to float addrspace(1)*
  %2255 = load float, float addrspace(1)* %2254, align 4
  %2256 = fmul reassoc nsz arcp contract float %2255, %4, !spirv.Decorations !898
  %2257 = fadd reassoc nsz arcp contract float %2245, %2256, !spirv.Decorations !898
  %2258 = add i64 %.in, %552
  %2259 = inttoptr i64 %2258 to float addrspace(4)*
  %2260 = addrspacecast float addrspace(4)* %2259 to float addrspace(1)*
  store float %2257, float addrspace(1)* %2260, align 4
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, %2250, %2246
  br i1 %210, label %2261, label %._crit_edge70.2.6..preheader1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.6..preheader1.6_crit_edge:        ; preds = %._crit_edge70.2.6
  br label %.preheader1.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2261:                                             ; preds = %._crit_edge70.2.6
  %2262 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !898
  br i1 %81, label %2267, label %2263, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2263:                                             ; preds = %2261
  %2264 = add i64 %.in, %554
  %2265 = inttoptr i64 %2264 to float addrspace(4)*
  %2266 = addrspacecast float addrspace(4)* %2265 to float addrspace(1)*
  store float %2262, float addrspace(1)* %2266, align 4
  br label %.preheader1.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2267:                                             ; preds = %2261
  %2268 = add i64 %.in3821, %sink_3840
  %2269 = add i64 %2268, %sink_3834
  %2270 = inttoptr i64 %2269 to float addrspace(4)*
  %2271 = addrspacecast float addrspace(4)* %2270 to float addrspace(1)*
  %2272 = load float, float addrspace(1)* %2271, align 4
  %2273 = fmul reassoc nsz arcp contract float %2272, %4, !spirv.Decorations !898
  %2274 = fadd reassoc nsz arcp contract float %2262, %2273, !spirv.Decorations !898
  %2275 = add i64 %.in, %554
  %2276 = inttoptr i64 %2275 to float addrspace(4)*
  %2277 = addrspacecast float addrspace(4)* %2276 to float addrspace(1)*
  store float %2274, float addrspace(1)* %2277, align 4
  br label %.preheader1.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.6:                                    ; preds = %._crit_edge70.2.6..preheader1.6_crit_edge, %2267, %2263
  %sink_3833 = shl nsw i64 %555, 2
  br i1 %214, label %2278, label %.preheader1.6.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.6.._crit_edge70.7_crit_edge:          ; preds = %.preheader1.6
  br label %._crit_edge70.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2278:                                             ; preds = %.preheader1.6
  %2279 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !898
  br i1 %81, label %2284, label %2280, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2280:                                             ; preds = %2278
  %2281 = add i64 %.in, %557
  %2282 = inttoptr i64 %2281 to float addrspace(4)*
  %2283 = addrspacecast float addrspace(4)* %2282 to float addrspace(1)*
  store float %2279, float addrspace(1)* %2283, align 4
  br label %._crit_edge70.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2284:                                             ; preds = %2278
  %2285 = add i64 %.in3821, %sink_3844
  %2286 = add i64 %2285, %sink_3833
  %2287 = inttoptr i64 %2286 to float addrspace(4)*
  %2288 = addrspacecast float addrspace(4)* %2287 to float addrspace(1)*
  %2289 = load float, float addrspace(1)* %2288, align 4
  %2290 = fmul reassoc nsz arcp contract float %2289, %4, !spirv.Decorations !898
  %2291 = fadd reassoc nsz arcp contract float %2279, %2290, !spirv.Decorations !898
  %2292 = add i64 %.in, %557
  %2293 = inttoptr i64 %2292 to float addrspace(4)*
  %2294 = addrspacecast float addrspace(4)* %2293 to float addrspace(1)*
  store float %2291, float addrspace(1)* %2294, align 4
  br label %._crit_edge70.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.7:                                  ; preds = %.preheader1.6.._crit_edge70.7_crit_edge, %2284, %2280
  br i1 %217, label %2295, label %._crit_edge70.7.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.7.._crit_edge70.1.7_crit_edge:      ; preds = %._crit_edge70.7
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2295:                                             ; preds = %._crit_edge70.7
  %2296 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !898
  br i1 %81, label %2301, label %2297, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2297:                                             ; preds = %2295
  %2298 = add i64 %.in, %559
  %2299 = inttoptr i64 %2298 to float addrspace(4)*
  %2300 = addrspacecast float addrspace(4)* %2299 to float addrspace(1)*
  store float %2296, float addrspace(1)* %2300, align 4
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2301:                                             ; preds = %2295
  %2302 = add i64 %.in3821, %sink_3842
  %2303 = add i64 %2302, %sink_3833
  %2304 = inttoptr i64 %2303 to float addrspace(4)*
  %2305 = addrspacecast float addrspace(4)* %2304 to float addrspace(1)*
  %2306 = load float, float addrspace(1)* %2305, align 4
  %2307 = fmul reassoc nsz arcp contract float %2306, %4, !spirv.Decorations !898
  %2308 = fadd reassoc nsz arcp contract float %2296, %2307, !spirv.Decorations !898
  %2309 = add i64 %.in, %559
  %2310 = inttoptr i64 %2309 to float addrspace(4)*
  %2311 = addrspacecast float addrspace(4)* %2310 to float addrspace(1)*
  store float %2308, float addrspace(1)* %2311, align 4
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7.._crit_edge70.1.7_crit_edge, %2301, %2297
  br i1 %220, label %2312, label %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.7.._crit_edge70.2.7_crit_edge:    ; preds = %._crit_edge70.1.7
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2312:                                             ; preds = %._crit_edge70.1.7
  %2313 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !898
  br i1 %81, label %2318, label %2314, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2314:                                             ; preds = %2312
  %2315 = add i64 %.in, %561
  %2316 = inttoptr i64 %2315 to float addrspace(4)*
  %2317 = addrspacecast float addrspace(4)* %2316 to float addrspace(1)*
  store float %2313, float addrspace(1)* %2317, align 4
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2318:                                             ; preds = %2312
  %2319 = add i64 %.in3821, %sink_3841
  %2320 = add i64 %2319, %sink_3833
  %2321 = inttoptr i64 %2320 to float addrspace(4)*
  %2322 = addrspacecast float addrspace(4)* %2321 to float addrspace(1)*
  %2323 = load float, float addrspace(1)* %2322, align 4
  %2324 = fmul reassoc nsz arcp contract float %2323, %4, !spirv.Decorations !898
  %2325 = fadd reassoc nsz arcp contract float %2313, %2324, !spirv.Decorations !898
  %2326 = add i64 %.in, %561
  %2327 = inttoptr i64 %2326 to float addrspace(4)*
  %2328 = addrspacecast float addrspace(4)* %2327 to float addrspace(1)*
  store float %2325, float addrspace(1)* %2328, align 4
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, %2318, %2314
  br i1 %223, label %2329, label %._crit_edge70.2.7..preheader1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.7..preheader1.7_crit_edge:        ; preds = %._crit_edge70.2.7
  br label %.preheader1.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2329:                                             ; preds = %._crit_edge70.2.7
  %2330 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !898
  br i1 %81, label %2335, label %2331, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2331:                                             ; preds = %2329
  %2332 = add i64 %.in, %563
  %2333 = inttoptr i64 %2332 to float addrspace(4)*
  %2334 = addrspacecast float addrspace(4)* %2333 to float addrspace(1)*
  store float %2330, float addrspace(1)* %2334, align 4
  br label %.preheader1.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2335:                                             ; preds = %2329
  %2336 = add i64 %.in3821, %sink_3840
  %2337 = add i64 %2336, %sink_3833
  %2338 = inttoptr i64 %2337 to float addrspace(4)*
  %2339 = addrspacecast float addrspace(4)* %2338 to float addrspace(1)*
  %2340 = load float, float addrspace(1)* %2339, align 4
  %2341 = fmul reassoc nsz arcp contract float %2340, %4, !spirv.Decorations !898
  %2342 = fadd reassoc nsz arcp contract float %2330, %2341, !spirv.Decorations !898
  %2343 = add i64 %.in, %563
  %2344 = inttoptr i64 %2343 to float addrspace(4)*
  %2345 = addrspacecast float addrspace(4)* %2344 to float addrspace(1)*
  store float %2342, float addrspace(1)* %2345, align 4
  br label %.preheader1.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.7:                                    ; preds = %._crit_edge70.2.7..preheader1.7_crit_edge, %2335, %2331
  %sink_3832 = shl nsw i64 %564, 2
  br i1 %227, label %2346, label %.preheader1.7.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.7.._crit_edge70.8_crit_edge:          ; preds = %.preheader1.7
  br label %._crit_edge70.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2346:                                             ; preds = %.preheader1.7
  %2347 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !898
  br i1 %81, label %2352, label %2348, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2348:                                             ; preds = %2346
  %2349 = add i64 %.in, %566
  %2350 = inttoptr i64 %2349 to float addrspace(4)*
  %2351 = addrspacecast float addrspace(4)* %2350 to float addrspace(1)*
  store float %2347, float addrspace(1)* %2351, align 4
  br label %._crit_edge70.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2352:                                             ; preds = %2346
  %2353 = add i64 %.in3821, %sink_3844
  %2354 = add i64 %2353, %sink_3832
  %2355 = inttoptr i64 %2354 to float addrspace(4)*
  %2356 = addrspacecast float addrspace(4)* %2355 to float addrspace(1)*
  %2357 = load float, float addrspace(1)* %2356, align 4
  %2358 = fmul reassoc nsz arcp contract float %2357, %4, !spirv.Decorations !898
  %2359 = fadd reassoc nsz arcp contract float %2347, %2358, !spirv.Decorations !898
  %2360 = add i64 %.in, %566
  %2361 = inttoptr i64 %2360 to float addrspace(4)*
  %2362 = addrspacecast float addrspace(4)* %2361 to float addrspace(1)*
  store float %2359, float addrspace(1)* %2362, align 4
  br label %._crit_edge70.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.8:                                  ; preds = %.preheader1.7.._crit_edge70.8_crit_edge, %2352, %2348
  br i1 %230, label %2363, label %._crit_edge70.8.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.8.._crit_edge70.1.8_crit_edge:      ; preds = %._crit_edge70.8
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2363:                                             ; preds = %._crit_edge70.8
  %2364 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !898
  br i1 %81, label %2369, label %2365, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2365:                                             ; preds = %2363
  %2366 = add i64 %.in, %568
  %2367 = inttoptr i64 %2366 to float addrspace(4)*
  %2368 = addrspacecast float addrspace(4)* %2367 to float addrspace(1)*
  store float %2364, float addrspace(1)* %2368, align 4
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2369:                                             ; preds = %2363
  %2370 = add i64 %.in3821, %sink_3842
  %2371 = add i64 %2370, %sink_3832
  %2372 = inttoptr i64 %2371 to float addrspace(4)*
  %2373 = addrspacecast float addrspace(4)* %2372 to float addrspace(1)*
  %2374 = load float, float addrspace(1)* %2373, align 4
  %2375 = fmul reassoc nsz arcp contract float %2374, %4, !spirv.Decorations !898
  %2376 = fadd reassoc nsz arcp contract float %2364, %2375, !spirv.Decorations !898
  %2377 = add i64 %.in, %568
  %2378 = inttoptr i64 %2377 to float addrspace(4)*
  %2379 = addrspacecast float addrspace(4)* %2378 to float addrspace(1)*
  store float %2376, float addrspace(1)* %2379, align 4
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8.._crit_edge70.1.8_crit_edge, %2369, %2365
  br i1 %233, label %2380, label %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.8.._crit_edge70.2.8_crit_edge:    ; preds = %._crit_edge70.1.8
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2380:                                             ; preds = %._crit_edge70.1.8
  %2381 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !898
  br i1 %81, label %2386, label %2382, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2382:                                             ; preds = %2380
  %2383 = add i64 %.in, %570
  %2384 = inttoptr i64 %2383 to float addrspace(4)*
  %2385 = addrspacecast float addrspace(4)* %2384 to float addrspace(1)*
  store float %2381, float addrspace(1)* %2385, align 4
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2386:                                             ; preds = %2380
  %2387 = add i64 %.in3821, %sink_3841
  %2388 = add i64 %2387, %sink_3832
  %2389 = inttoptr i64 %2388 to float addrspace(4)*
  %2390 = addrspacecast float addrspace(4)* %2389 to float addrspace(1)*
  %2391 = load float, float addrspace(1)* %2390, align 4
  %2392 = fmul reassoc nsz arcp contract float %2391, %4, !spirv.Decorations !898
  %2393 = fadd reassoc nsz arcp contract float %2381, %2392, !spirv.Decorations !898
  %2394 = add i64 %.in, %570
  %2395 = inttoptr i64 %2394 to float addrspace(4)*
  %2396 = addrspacecast float addrspace(4)* %2395 to float addrspace(1)*
  store float %2393, float addrspace(1)* %2396, align 4
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, %2386, %2382
  br i1 %236, label %2397, label %._crit_edge70.2.8..preheader1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.8..preheader1.8_crit_edge:        ; preds = %._crit_edge70.2.8
  br label %.preheader1.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2397:                                             ; preds = %._crit_edge70.2.8
  %2398 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !898
  br i1 %81, label %2403, label %2399, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2399:                                             ; preds = %2397
  %2400 = add i64 %.in, %572
  %2401 = inttoptr i64 %2400 to float addrspace(4)*
  %2402 = addrspacecast float addrspace(4)* %2401 to float addrspace(1)*
  store float %2398, float addrspace(1)* %2402, align 4
  br label %.preheader1.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2403:                                             ; preds = %2397
  %2404 = add i64 %.in3821, %sink_3840
  %2405 = add i64 %2404, %sink_3832
  %2406 = inttoptr i64 %2405 to float addrspace(4)*
  %2407 = addrspacecast float addrspace(4)* %2406 to float addrspace(1)*
  %2408 = load float, float addrspace(1)* %2407, align 4
  %2409 = fmul reassoc nsz arcp contract float %2408, %4, !spirv.Decorations !898
  %2410 = fadd reassoc nsz arcp contract float %2398, %2409, !spirv.Decorations !898
  %2411 = add i64 %.in, %572
  %2412 = inttoptr i64 %2411 to float addrspace(4)*
  %2413 = addrspacecast float addrspace(4)* %2412 to float addrspace(1)*
  store float %2410, float addrspace(1)* %2413, align 4
  br label %.preheader1.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.8:                                    ; preds = %._crit_edge70.2.8..preheader1.8_crit_edge, %2403, %2399
  %sink_3831 = shl nsw i64 %573, 2
  br i1 %240, label %2414, label %.preheader1.8.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.8.._crit_edge70.9_crit_edge:          ; preds = %.preheader1.8
  br label %._crit_edge70.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2414:                                             ; preds = %.preheader1.8
  %2415 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !898
  br i1 %81, label %2420, label %2416, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2416:                                             ; preds = %2414
  %2417 = add i64 %.in, %575
  %2418 = inttoptr i64 %2417 to float addrspace(4)*
  %2419 = addrspacecast float addrspace(4)* %2418 to float addrspace(1)*
  store float %2415, float addrspace(1)* %2419, align 4
  br label %._crit_edge70.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2420:                                             ; preds = %2414
  %2421 = add i64 %.in3821, %sink_3844
  %2422 = add i64 %2421, %sink_3831
  %2423 = inttoptr i64 %2422 to float addrspace(4)*
  %2424 = addrspacecast float addrspace(4)* %2423 to float addrspace(1)*
  %2425 = load float, float addrspace(1)* %2424, align 4
  %2426 = fmul reassoc nsz arcp contract float %2425, %4, !spirv.Decorations !898
  %2427 = fadd reassoc nsz arcp contract float %2415, %2426, !spirv.Decorations !898
  %2428 = add i64 %.in, %575
  %2429 = inttoptr i64 %2428 to float addrspace(4)*
  %2430 = addrspacecast float addrspace(4)* %2429 to float addrspace(1)*
  store float %2427, float addrspace(1)* %2430, align 4
  br label %._crit_edge70.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.9:                                  ; preds = %.preheader1.8.._crit_edge70.9_crit_edge, %2420, %2416
  br i1 %243, label %2431, label %._crit_edge70.9.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.9.._crit_edge70.1.9_crit_edge:      ; preds = %._crit_edge70.9
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2431:                                             ; preds = %._crit_edge70.9
  %2432 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !898
  br i1 %81, label %2437, label %2433, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2433:                                             ; preds = %2431
  %2434 = add i64 %.in, %577
  %2435 = inttoptr i64 %2434 to float addrspace(4)*
  %2436 = addrspacecast float addrspace(4)* %2435 to float addrspace(1)*
  store float %2432, float addrspace(1)* %2436, align 4
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2437:                                             ; preds = %2431
  %2438 = add i64 %.in3821, %sink_3842
  %2439 = add i64 %2438, %sink_3831
  %2440 = inttoptr i64 %2439 to float addrspace(4)*
  %2441 = addrspacecast float addrspace(4)* %2440 to float addrspace(1)*
  %2442 = load float, float addrspace(1)* %2441, align 4
  %2443 = fmul reassoc nsz arcp contract float %2442, %4, !spirv.Decorations !898
  %2444 = fadd reassoc nsz arcp contract float %2432, %2443, !spirv.Decorations !898
  %2445 = add i64 %.in, %577
  %2446 = inttoptr i64 %2445 to float addrspace(4)*
  %2447 = addrspacecast float addrspace(4)* %2446 to float addrspace(1)*
  store float %2444, float addrspace(1)* %2447, align 4
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9.._crit_edge70.1.9_crit_edge, %2437, %2433
  br i1 %246, label %2448, label %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.9.._crit_edge70.2.9_crit_edge:    ; preds = %._crit_edge70.1.9
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2448:                                             ; preds = %._crit_edge70.1.9
  %2449 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !898
  br i1 %81, label %2454, label %2450, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2450:                                             ; preds = %2448
  %2451 = add i64 %.in, %579
  %2452 = inttoptr i64 %2451 to float addrspace(4)*
  %2453 = addrspacecast float addrspace(4)* %2452 to float addrspace(1)*
  store float %2449, float addrspace(1)* %2453, align 4
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2454:                                             ; preds = %2448
  %2455 = add i64 %.in3821, %sink_3841
  %2456 = add i64 %2455, %sink_3831
  %2457 = inttoptr i64 %2456 to float addrspace(4)*
  %2458 = addrspacecast float addrspace(4)* %2457 to float addrspace(1)*
  %2459 = load float, float addrspace(1)* %2458, align 4
  %2460 = fmul reassoc nsz arcp contract float %2459, %4, !spirv.Decorations !898
  %2461 = fadd reassoc nsz arcp contract float %2449, %2460, !spirv.Decorations !898
  %2462 = add i64 %.in, %579
  %2463 = inttoptr i64 %2462 to float addrspace(4)*
  %2464 = addrspacecast float addrspace(4)* %2463 to float addrspace(1)*
  store float %2461, float addrspace(1)* %2464, align 4
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, %2454, %2450
  br i1 %249, label %2465, label %._crit_edge70.2.9..preheader1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.9..preheader1.9_crit_edge:        ; preds = %._crit_edge70.2.9
  br label %.preheader1.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2465:                                             ; preds = %._crit_edge70.2.9
  %2466 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !898
  br i1 %81, label %2471, label %2467, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2467:                                             ; preds = %2465
  %2468 = add i64 %.in, %581
  %2469 = inttoptr i64 %2468 to float addrspace(4)*
  %2470 = addrspacecast float addrspace(4)* %2469 to float addrspace(1)*
  store float %2466, float addrspace(1)* %2470, align 4
  br label %.preheader1.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2471:                                             ; preds = %2465
  %2472 = add i64 %.in3821, %sink_3840
  %2473 = add i64 %2472, %sink_3831
  %2474 = inttoptr i64 %2473 to float addrspace(4)*
  %2475 = addrspacecast float addrspace(4)* %2474 to float addrspace(1)*
  %2476 = load float, float addrspace(1)* %2475, align 4
  %2477 = fmul reassoc nsz arcp contract float %2476, %4, !spirv.Decorations !898
  %2478 = fadd reassoc nsz arcp contract float %2466, %2477, !spirv.Decorations !898
  %2479 = add i64 %.in, %581
  %2480 = inttoptr i64 %2479 to float addrspace(4)*
  %2481 = addrspacecast float addrspace(4)* %2480 to float addrspace(1)*
  store float %2478, float addrspace(1)* %2481, align 4
  br label %.preheader1.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.9:                                    ; preds = %._crit_edge70.2.9..preheader1.9_crit_edge, %2471, %2467
  %sink_3830 = shl nsw i64 %582, 2
  br i1 %253, label %2482, label %.preheader1.9.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.9.._crit_edge70.10_crit_edge:         ; preds = %.preheader1.9
  br label %._crit_edge70.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2482:                                             ; preds = %.preheader1.9
  %2483 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !898
  br i1 %81, label %2488, label %2484, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2484:                                             ; preds = %2482
  %2485 = add i64 %.in, %584
  %2486 = inttoptr i64 %2485 to float addrspace(4)*
  %2487 = addrspacecast float addrspace(4)* %2486 to float addrspace(1)*
  store float %2483, float addrspace(1)* %2487, align 4
  br label %._crit_edge70.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2488:                                             ; preds = %2482
  %2489 = add i64 %.in3821, %sink_3844
  %2490 = add i64 %2489, %sink_3830
  %2491 = inttoptr i64 %2490 to float addrspace(4)*
  %2492 = addrspacecast float addrspace(4)* %2491 to float addrspace(1)*
  %2493 = load float, float addrspace(1)* %2492, align 4
  %2494 = fmul reassoc nsz arcp contract float %2493, %4, !spirv.Decorations !898
  %2495 = fadd reassoc nsz arcp contract float %2483, %2494, !spirv.Decorations !898
  %2496 = add i64 %.in, %584
  %2497 = inttoptr i64 %2496 to float addrspace(4)*
  %2498 = addrspacecast float addrspace(4)* %2497 to float addrspace(1)*
  store float %2495, float addrspace(1)* %2498, align 4
  br label %._crit_edge70.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.10:                                 ; preds = %.preheader1.9.._crit_edge70.10_crit_edge, %2488, %2484
  br i1 %256, label %2499, label %._crit_edge70.10.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.10.._crit_edge70.1.10_crit_edge:    ; preds = %._crit_edge70.10
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2499:                                             ; preds = %._crit_edge70.10
  %2500 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !898
  br i1 %81, label %2505, label %2501, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2501:                                             ; preds = %2499
  %2502 = add i64 %.in, %586
  %2503 = inttoptr i64 %2502 to float addrspace(4)*
  %2504 = addrspacecast float addrspace(4)* %2503 to float addrspace(1)*
  store float %2500, float addrspace(1)* %2504, align 4
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2505:                                             ; preds = %2499
  %2506 = add i64 %.in3821, %sink_3842
  %2507 = add i64 %2506, %sink_3830
  %2508 = inttoptr i64 %2507 to float addrspace(4)*
  %2509 = addrspacecast float addrspace(4)* %2508 to float addrspace(1)*
  %2510 = load float, float addrspace(1)* %2509, align 4
  %2511 = fmul reassoc nsz arcp contract float %2510, %4, !spirv.Decorations !898
  %2512 = fadd reassoc nsz arcp contract float %2500, %2511, !spirv.Decorations !898
  %2513 = add i64 %.in, %586
  %2514 = inttoptr i64 %2513 to float addrspace(4)*
  %2515 = addrspacecast float addrspace(4)* %2514 to float addrspace(1)*
  store float %2512, float addrspace(1)* %2515, align 4
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10.._crit_edge70.1.10_crit_edge, %2505, %2501
  br i1 %259, label %2516, label %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.10.._crit_edge70.2.10_crit_edge:  ; preds = %._crit_edge70.1.10
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2516:                                             ; preds = %._crit_edge70.1.10
  %2517 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !898
  br i1 %81, label %2522, label %2518, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2518:                                             ; preds = %2516
  %2519 = add i64 %.in, %588
  %2520 = inttoptr i64 %2519 to float addrspace(4)*
  %2521 = addrspacecast float addrspace(4)* %2520 to float addrspace(1)*
  store float %2517, float addrspace(1)* %2521, align 4
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2522:                                             ; preds = %2516
  %2523 = add i64 %.in3821, %sink_3841
  %2524 = add i64 %2523, %sink_3830
  %2525 = inttoptr i64 %2524 to float addrspace(4)*
  %2526 = addrspacecast float addrspace(4)* %2525 to float addrspace(1)*
  %2527 = load float, float addrspace(1)* %2526, align 4
  %2528 = fmul reassoc nsz arcp contract float %2527, %4, !spirv.Decorations !898
  %2529 = fadd reassoc nsz arcp contract float %2517, %2528, !spirv.Decorations !898
  %2530 = add i64 %.in, %588
  %2531 = inttoptr i64 %2530 to float addrspace(4)*
  %2532 = addrspacecast float addrspace(4)* %2531 to float addrspace(1)*
  store float %2529, float addrspace(1)* %2532, align 4
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, %2522, %2518
  br i1 %262, label %2533, label %._crit_edge70.2.10..preheader1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.10..preheader1.10_crit_edge:      ; preds = %._crit_edge70.2.10
  br label %.preheader1.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2533:                                             ; preds = %._crit_edge70.2.10
  %2534 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !898
  br i1 %81, label %2539, label %2535, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2535:                                             ; preds = %2533
  %2536 = add i64 %.in, %590
  %2537 = inttoptr i64 %2536 to float addrspace(4)*
  %2538 = addrspacecast float addrspace(4)* %2537 to float addrspace(1)*
  store float %2534, float addrspace(1)* %2538, align 4
  br label %.preheader1.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2539:                                             ; preds = %2533
  %2540 = add i64 %.in3821, %sink_3840
  %2541 = add i64 %2540, %sink_3830
  %2542 = inttoptr i64 %2541 to float addrspace(4)*
  %2543 = addrspacecast float addrspace(4)* %2542 to float addrspace(1)*
  %2544 = load float, float addrspace(1)* %2543, align 4
  %2545 = fmul reassoc nsz arcp contract float %2544, %4, !spirv.Decorations !898
  %2546 = fadd reassoc nsz arcp contract float %2534, %2545, !spirv.Decorations !898
  %2547 = add i64 %.in, %590
  %2548 = inttoptr i64 %2547 to float addrspace(4)*
  %2549 = addrspacecast float addrspace(4)* %2548 to float addrspace(1)*
  store float %2546, float addrspace(1)* %2549, align 4
  br label %.preheader1.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.10:                                   ; preds = %._crit_edge70.2.10..preheader1.10_crit_edge, %2539, %2535
  %sink_3829 = shl nsw i64 %591, 2
  br i1 %266, label %2550, label %.preheader1.10.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.10.._crit_edge70.11_crit_edge:        ; preds = %.preheader1.10
  br label %._crit_edge70.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2550:                                             ; preds = %.preheader1.10
  %2551 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !898
  br i1 %81, label %2556, label %2552, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2552:                                             ; preds = %2550
  %2553 = add i64 %.in, %593
  %2554 = inttoptr i64 %2553 to float addrspace(4)*
  %2555 = addrspacecast float addrspace(4)* %2554 to float addrspace(1)*
  store float %2551, float addrspace(1)* %2555, align 4
  br label %._crit_edge70.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2556:                                             ; preds = %2550
  %2557 = add i64 %.in3821, %sink_3844
  %2558 = add i64 %2557, %sink_3829
  %2559 = inttoptr i64 %2558 to float addrspace(4)*
  %2560 = addrspacecast float addrspace(4)* %2559 to float addrspace(1)*
  %2561 = load float, float addrspace(1)* %2560, align 4
  %2562 = fmul reassoc nsz arcp contract float %2561, %4, !spirv.Decorations !898
  %2563 = fadd reassoc nsz arcp contract float %2551, %2562, !spirv.Decorations !898
  %2564 = add i64 %.in, %593
  %2565 = inttoptr i64 %2564 to float addrspace(4)*
  %2566 = addrspacecast float addrspace(4)* %2565 to float addrspace(1)*
  store float %2563, float addrspace(1)* %2566, align 4
  br label %._crit_edge70.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.11:                                 ; preds = %.preheader1.10.._crit_edge70.11_crit_edge, %2556, %2552
  br i1 %269, label %2567, label %._crit_edge70.11.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.11.._crit_edge70.1.11_crit_edge:    ; preds = %._crit_edge70.11
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2567:                                             ; preds = %._crit_edge70.11
  %2568 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !898
  br i1 %81, label %2573, label %2569, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2569:                                             ; preds = %2567
  %2570 = add i64 %.in, %595
  %2571 = inttoptr i64 %2570 to float addrspace(4)*
  %2572 = addrspacecast float addrspace(4)* %2571 to float addrspace(1)*
  store float %2568, float addrspace(1)* %2572, align 4
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2573:                                             ; preds = %2567
  %2574 = add i64 %.in3821, %sink_3842
  %2575 = add i64 %2574, %sink_3829
  %2576 = inttoptr i64 %2575 to float addrspace(4)*
  %2577 = addrspacecast float addrspace(4)* %2576 to float addrspace(1)*
  %2578 = load float, float addrspace(1)* %2577, align 4
  %2579 = fmul reassoc nsz arcp contract float %2578, %4, !spirv.Decorations !898
  %2580 = fadd reassoc nsz arcp contract float %2568, %2579, !spirv.Decorations !898
  %2581 = add i64 %.in, %595
  %2582 = inttoptr i64 %2581 to float addrspace(4)*
  %2583 = addrspacecast float addrspace(4)* %2582 to float addrspace(1)*
  store float %2580, float addrspace(1)* %2583, align 4
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11.._crit_edge70.1.11_crit_edge, %2573, %2569
  br i1 %272, label %2584, label %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.11.._crit_edge70.2.11_crit_edge:  ; preds = %._crit_edge70.1.11
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2584:                                             ; preds = %._crit_edge70.1.11
  %2585 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !898
  br i1 %81, label %2590, label %2586, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2586:                                             ; preds = %2584
  %2587 = add i64 %.in, %597
  %2588 = inttoptr i64 %2587 to float addrspace(4)*
  %2589 = addrspacecast float addrspace(4)* %2588 to float addrspace(1)*
  store float %2585, float addrspace(1)* %2589, align 4
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2590:                                             ; preds = %2584
  %2591 = add i64 %.in3821, %sink_3841
  %2592 = add i64 %2591, %sink_3829
  %2593 = inttoptr i64 %2592 to float addrspace(4)*
  %2594 = addrspacecast float addrspace(4)* %2593 to float addrspace(1)*
  %2595 = load float, float addrspace(1)* %2594, align 4
  %2596 = fmul reassoc nsz arcp contract float %2595, %4, !spirv.Decorations !898
  %2597 = fadd reassoc nsz arcp contract float %2585, %2596, !spirv.Decorations !898
  %2598 = add i64 %.in, %597
  %2599 = inttoptr i64 %2598 to float addrspace(4)*
  %2600 = addrspacecast float addrspace(4)* %2599 to float addrspace(1)*
  store float %2597, float addrspace(1)* %2600, align 4
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, %2590, %2586
  br i1 %275, label %2601, label %._crit_edge70.2.11..preheader1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.11..preheader1.11_crit_edge:      ; preds = %._crit_edge70.2.11
  br label %.preheader1.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2601:                                             ; preds = %._crit_edge70.2.11
  %2602 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !898
  br i1 %81, label %2607, label %2603, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2603:                                             ; preds = %2601
  %2604 = add i64 %.in, %599
  %2605 = inttoptr i64 %2604 to float addrspace(4)*
  %2606 = addrspacecast float addrspace(4)* %2605 to float addrspace(1)*
  store float %2602, float addrspace(1)* %2606, align 4
  br label %.preheader1.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2607:                                             ; preds = %2601
  %2608 = add i64 %.in3821, %sink_3840
  %2609 = add i64 %2608, %sink_3829
  %2610 = inttoptr i64 %2609 to float addrspace(4)*
  %2611 = addrspacecast float addrspace(4)* %2610 to float addrspace(1)*
  %2612 = load float, float addrspace(1)* %2611, align 4
  %2613 = fmul reassoc nsz arcp contract float %2612, %4, !spirv.Decorations !898
  %2614 = fadd reassoc nsz arcp contract float %2602, %2613, !spirv.Decorations !898
  %2615 = add i64 %.in, %599
  %2616 = inttoptr i64 %2615 to float addrspace(4)*
  %2617 = addrspacecast float addrspace(4)* %2616 to float addrspace(1)*
  store float %2614, float addrspace(1)* %2617, align 4
  br label %.preheader1.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.11:                                   ; preds = %._crit_edge70.2.11..preheader1.11_crit_edge, %2607, %2603
  %sink_3828 = shl nsw i64 %600, 2
  br i1 %279, label %2618, label %.preheader1.11.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.11.._crit_edge70.12_crit_edge:        ; preds = %.preheader1.11
  br label %._crit_edge70.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2618:                                             ; preds = %.preheader1.11
  %2619 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !898
  br i1 %81, label %2624, label %2620, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2620:                                             ; preds = %2618
  %2621 = add i64 %.in, %602
  %2622 = inttoptr i64 %2621 to float addrspace(4)*
  %2623 = addrspacecast float addrspace(4)* %2622 to float addrspace(1)*
  store float %2619, float addrspace(1)* %2623, align 4
  br label %._crit_edge70.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2624:                                             ; preds = %2618
  %2625 = add i64 %.in3821, %sink_3844
  %2626 = add i64 %2625, %sink_3828
  %2627 = inttoptr i64 %2626 to float addrspace(4)*
  %2628 = addrspacecast float addrspace(4)* %2627 to float addrspace(1)*
  %2629 = load float, float addrspace(1)* %2628, align 4
  %2630 = fmul reassoc nsz arcp contract float %2629, %4, !spirv.Decorations !898
  %2631 = fadd reassoc nsz arcp contract float %2619, %2630, !spirv.Decorations !898
  %2632 = add i64 %.in, %602
  %2633 = inttoptr i64 %2632 to float addrspace(4)*
  %2634 = addrspacecast float addrspace(4)* %2633 to float addrspace(1)*
  store float %2631, float addrspace(1)* %2634, align 4
  br label %._crit_edge70.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.12:                                 ; preds = %.preheader1.11.._crit_edge70.12_crit_edge, %2624, %2620
  br i1 %282, label %2635, label %._crit_edge70.12.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.12.._crit_edge70.1.12_crit_edge:    ; preds = %._crit_edge70.12
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2635:                                             ; preds = %._crit_edge70.12
  %2636 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !898
  br i1 %81, label %2641, label %2637, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2637:                                             ; preds = %2635
  %2638 = add i64 %.in, %604
  %2639 = inttoptr i64 %2638 to float addrspace(4)*
  %2640 = addrspacecast float addrspace(4)* %2639 to float addrspace(1)*
  store float %2636, float addrspace(1)* %2640, align 4
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2641:                                             ; preds = %2635
  %2642 = add i64 %.in3821, %sink_3842
  %2643 = add i64 %2642, %sink_3828
  %2644 = inttoptr i64 %2643 to float addrspace(4)*
  %2645 = addrspacecast float addrspace(4)* %2644 to float addrspace(1)*
  %2646 = load float, float addrspace(1)* %2645, align 4
  %2647 = fmul reassoc nsz arcp contract float %2646, %4, !spirv.Decorations !898
  %2648 = fadd reassoc nsz arcp contract float %2636, %2647, !spirv.Decorations !898
  %2649 = add i64 %.in, %604
  %2650 = inttoptr i64 %2649 to float addrspace(4)*
  %2651 = addrspacecast float addrspace(4)* %2650 to float addrspace(1)*
  store float %2648, float addrspace(1)* %2651, align 4
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12.._crit_edge70.1.12_crit_edge, %2641, %2637
  br i1 %285, label %2652, label %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.12.._crit_edge70.2.12_crit_edge:  ; preds = %._crit_edge70.1.12
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2652:                                             ; preds = %._crit_edge70.1.12
  %2653 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !898
  br i1 %81, label %2658, label %2654, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2654:                                             ; preds = %2652
  %2655 = add i64 %.in, %606
  %2656 = inttoptr i64 %2655 to float addrspace(4)*
  %2657 = addrspacecast float addrspace(4)* %2656 to float addrspace(1)*
  store float %2653, float addrspace(1)* %2657, align 4
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2658:                                             ; preds = %2652
  %2659 = add i64 %.in3821, %sink_3841
  %2660 = add i64 %2659, %sink_3828
  %2661 = inttoptr i64 %2660 to float addrspace(4)*
  %2662 = addrspacecast float addrspace(4)* %2661 to float addrspace(1)*
  %2663 = load float, float addrspace(1)* %2662, align 4
  %2664 = fmul reassoc nsz arcp contract float %2663, %4, !spirv.Decorations !898
  %2665 = fadd reassoc nsz arcp contract float %2653, %2664, !spirv.Decorations !898
  %2666 = add i64 %.in, %606
  %2667 = inttoptr i64 %2666 to float addrspace(4)*
  %2668 = addrspacecast float addrspace(4)* %2667 to float addrspace(1)*
  store float %2665, float addrspace(1)* %2668, align 4
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, %2658, %2654
  br i1 %288, label %2669, label %._crit_edge70.2.12..preheader1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.12..preheader1.12_crit_edge:      ; preds = %._crit_edge70.2.12
  br label %.preheader1.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2669:                                             ; preds = %._crit_edge70.2.12
  %2670 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !898
  br i1 %81, label %2675, label %2671, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2671:                                             ; preds = %2669
  %2672 = add i64 %.in, %608
  %2673 = inttoptr i64 %2672 to float addrspace(4)*
  %2674 = addrspacecast float addrspace(4)* %2673 to float addrspace(1)*
  store float %2670, float addrspace(1)* %2674, align 4
  br label %.preheader1.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2675:                                             ; preds = %2669
  %2676 = add i64 %.in3821, %sink_3840
  %2677 = add i64 %2676, %sink_3828
  %2678 = inttoptr i64 %2677 to float addrspace(4)*
  %2679 = addrspacecast float addrspace(4)* %2678 to float addrspace(1)*
  %2680 = load float, float addrspace(1)* %2679, align 4
  %2681 = fmul reassoc nsz arcp contract float %2680, %4, !spirv.Decorations !898
  %2682 = fadd reassoc nsz arcp contract float %2670, %2681, !spirv.Decorations !898
  %2683 = add i64 %.in, %608
  %2684 = inttoptr i64 %2683 to float addrspace(4)*
  %2685 = addrspacecast float addrspace(4)* %2684 to float addrspace(1)*
  store float %2682, float addrspace(1)* %2685, align 4
  br label %.preheader1.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.12:                                   ; preds = %._crit_edge70.2.12..preheader1.12_crit_edge, %2675, %2671
  %sink_3827 = shl nsw i64 %609, 2
  br i1 %292, label %2686, label %.preheader1.12.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.12.._crit_edge70.13_crit_edge:        ; preds = %.preheader1.12
  br label %._crit_edge70.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2686:                                             ; preds = %.preheader1.12
  %2687 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !898
  br i1 %81, label %2692, label %2688, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2688:                                             ; preds = %2686
  %2689 = add i64 %.in, %611
  %2690 = inttoptr i64 %2689 to float addrspace(4)*
  %2691 = addrspacecast float addrspace(4)* %2690 to float addrspace(1)*
  store float %2687, float addrspace(1)* %2691, align 4
  br label %._crit_edge70.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2692:                                             ; preds = %2686
  %2693 = add i64 %.in3821, %sink_3844
  %2694 = add i64 %2693, %sink_3827
  %2695 = inttoptr i64 %2694 to float addrspace(4)*
  %2696 = addrspacecast float addrspace(4)* %2695 to float addrspace(1)*
  %2697 = load float, float addrspace(1)* %2696, align 4
  %2698 = fmul reassoc nsz arcp contract float %2697, %4, !spirv.Decorations !898
  %2699 = fadd reassoc nsz arcp contract float %2687, %2698, !spirv.Decorations !898
  %2700 = add i64 %.in, %611
  %2701 = inttoptr i64 %2700 to float addrspace(4)*
  %2702 = addrspacecast float addrspace(4)* %2701 to float addrspace(1)*
  store float %2699, float addrspace(1)* %2702, align 4
  br label %._crit_edge70.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.13:                                 ; preds = %.preheader1.12.._crit_edge70.13_crit_edge, %2692, %2688
  br i1 %295, label %2703, label %._crit_edge70.13.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.13.._crit_edge70.1.13_crit_edge:    ; preds = %._crit_edge70.13
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2703:                                             ; preds = %._crit_edge70.13
  %2704 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !898
  br i1 %81, label %2709, label %2705, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2705:                                             ; preds = %2703
  %2706 = add i64 %.in, %613
  %2707 = inttoptr i64 %2706 to float addrspace(4)*
  %2708 = addrspacecast float addrspace(4)* %2707 to float addrspace(1)*
  store float %2704, float addrspace(1)* %2708, align 4
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2709:                                             ; preds = %2703
  %2710 = add i64 %.in3821, %sink_3842
  %2711 = add i64 %2710, %sink_3827
  %2712 = inttoptr i64 %2711 to float addrspace(4)*
  %2713 = addrspacecast float addrspace(4)* %2712 to float addrspace(1)*
  %2714 = load float, float addrspace(1)* %2713, align 4
  %2715 = fmul reassoc nsz arcp contract float %2714, %4, !spirv.Decorations !898
  %2716 = fadd reassoc nsz arcp contract float %2704, %2715, !spirv.Decorations !898
  %2717 = add i64 %.in, %613
  %2718 = inttoptr i64 %2717 to float addrspace(4)*
  %2719 = addrspacecast float addrspace(4)* %2718 to float addrspace(1)*
  store float %2716, float addrspace(1)* %2719, align 4
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13.._crit_edge70.1.13_crit_edge, %2709, %2705
  br i1 %298, label %2720, label %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.13.._crit_edge70.2.13_crit_edge:  ; preds = %._crit_edge70.1.13
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2720:                                             ; preds = %._crit_edge70.1.13
  %2721 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !898
  br i1 %81, label %2726, label %2722, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2722:                                             ; preds = %2720
  %2723 = add i64 %.in, %615
  %2724 = inttoptr i64 %2723 to float addrspace(4)*
  %2725 = addrspacecast float addrspace(4)* %2724 to float addrspace(1)*
  store float %2721, float addrspace(1)* %2725, align 4
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2726:                                             ; preds = %2720
  %2727 = add i64 %.in3821, %sink_3841
  %2728 = add i64 %2727, %sink_3827
  %2729 = inttoptr i64 %2728 to float addrspace(4)*
  %2730 = addrspacecast float addrspace(4)* %2729 to float addrspace(1)*
  %2731 = load float, float addrspace(1)* %2730, align 4
  %2732 = fmul reassoc nsz arcp contract float %2731, %4, !spirv.Decorations !898
  %2733 = fadd reassoc nsz arcp contract float %2721, %2732, !spirv.Decorations !898
  %2734 = add i64 %.in, %615
  %2735 = inttoptr i64 %2734 to float addrspace(4)*
  %2736 = addrspacecast float addrspace(4)* %2735 to float addrspace(1)*
  store float %2733, float addrspace(1)* %2736, align 4
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, %2726, %2722
  br i1 %301, label %2737, label %._crit_edge70.2.13..preheader1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.13..preheader1.13_crit_edge:      ; preds = %._crit_edge70.2.13
  br label %.preheader1.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2737:                                             ; preds = %._crit_edge70.2.13
  %2738 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !898
  br i1 %81, label %2743, label %2739, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2739:                                             ; preds = %2737
  %2740 = add i64 %.in, %617
  %2741 = inttoptr i64 %2740 to float addrspace(4)*
  %2742 = addrspacecast float addrspace(4)* %2741 to float addrspace(1)*
  store float %2738, float addrspace(1)* %2742, align 4
  br label %.preheader1.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2743:                                             ; preds = %2737
  %2744 = add i64 %.in3821, %sink_3840
  %2745 = add i64 %2744, %sink_3827
  %2746 = inttoptr i64 %2745 to float addrspace(4)*
  %2747 = addrspacecast float addrspace(4)* %2746 to float addrspace(1)*
  %2748 = load float, float addrspace(1)* %2747, align 4
  %2749 = fmul reassoc nsz arcp contract float %2748, %4, !spirv.Decorations !898
  %2750 = fadd reassoc nsz arcp contract float %2738, %2749, !spirv.Decorations !898
  %2751 = add i64 %.in, %617
  %2752 = inttoptr i64 %2751 to float addrspace(4)*
  %2753 = addrspacecast float addrspace(4)* %2752 to float addrspace(1)*
  store float %2750, float addrspace(1)* %2753, align 4
  br label %.preheader1.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.13:                                   ; preds = %._crit_edge70.2.13..preheader1.13_crit_edge, %2743, %2739
  %sink_3826 = shl nsw i64 %618, 2
  br i1 %305, label %2754, label %.preheader1.13.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.13.._crit_edge70.14_crit_edge:        ; preds = %.preheader1.13
  br label %._crit_edge70.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2754:                                             ; preds = %.preheader1.13
  %2755 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !898
  br i1 %81, label %2760, label %2756, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2756:                                             ; preds = %2754
  %2757 = add i64 %.in, %620
  %2758 = inttoptr i64 %2757 to float addrspace(4)*
  %2759 = addrspacecast float addrspace(4)* %2758 to float addrspace(1)*
  store float %2755, float addrspace(1)* %2759, align 4
  br label %._crit_edge70.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2760:                                             ; preds = %2754
  %2761 = add i64 %.in3821, %sink_3844
  %2762 = add i64 %2761, %sink_3826
  %2763 = inttoptr i64 %2762 to float addrspace(4)*
  %2764 = addrspacecast float addrspace(4)* %2763 to float addrspace(1)*
  %2765 = load float, float addrspace(1)* %2764, align 4
  %2766 = fmul reassoc nsz arcp contract float %2765, %4, !spirv.Decorations !898
  %2767 = fadd reassoc nsz arcp contract float %2755, %2766, !spirv.Decorations !898
  %2768 = add i64 %.in, %620
  %2769 = inttoptr i64 %2768 to float addrspace(4)*
  %2770 = addrspacecast float addrspace(4)* %2769 to float addrspace(1)*
  store float %2767, float addrspace(1)* %2770, align 4
  br label %._crit_edge70.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.14:                                 ; preds = %.preheader1.13.._crit_edge70.14_crit_edge, %2760, %2756
  br i1 %308, label %2771, label %._crit_edge70.14.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.14.._crit_edge70.1.14_crit_edge:    ; preds = %._crit_edge70.14
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2771:                                             ; preds = %._crit_edge70.14
  %2772 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !898
  br i1 %81, label %2777, label %2773, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2773:                                             ; preds = %2771
  %2774 = add i64 %.in, %622
  %2775 = inttoptr i64 %2774 to float addrspace(4)*
  %2776 = addrspacecast float addrspace(4)* %2775 to float addrspace(1)*
  store float %2772, float addrspace(1)* %2776, align 4
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2777:                                             ; preds = %2771
  %2778 = add i64 %.in3821, %sink_3842
  %2779 = add i64 %2778, %sink_3826
  %2780 = inttoptr i64 %2779 to float addrspace(4)*
  %2781 = addrspacecast float addrspace(4)* %2780 to float addrspace(1)*
  %2782 = load float, float addrspace(1)* %2781, align 4
  %2783 = fmul reassoc nsz arcp contract float %2782, %4, !spirv.Decorations !898
  %2784 = fadd reassoc nsz arcp contract float %2772, %2783, !spirv.Decorations !898
  %2785 = add i64 %.in, %622
  %2786 = inttoptr i64 %2785 to float addrspace(4)*
  %2787 = addrspacecast float addrspace(4)* %2786 to float addrspace(1)*
  store float %2784, float addrspace(1)* %2787, align 4
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14.._crit_edge70.1.14_crit_edge, %2777, %2773
  br i1 %311, label %2788, label %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.14.._crit_edge70.2.14_crit_edge:  ; preds = %._crit_edge70.1.14
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2788:                                             ; preds = %._crit_edge70.1.14
  %2789 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !898
  br i1 %81, label %2794, label %2790, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2790:                                             ; preds = %2788
  %2791 = add i64 %.in, %624
  %2792 = inttoptr i64 %2791 to float addrspace(4)*
  %2793 = addrspacecast float addrspace(4)* %2792 to float addrspace(1)*
  store float %2789, float addrspace(1)* %2793, align 4
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2794:                                             ; preds = %2788
  %2795 = add i64 %.in3821, %sink_3841
  %2796 = add i64 %2795, %sink_3826
  %2797 = inttoptr i64 %2796 to float addrspace(4)*
  %2798 = addrspacecast float addrspace(4)* %2797 to float addrspace(1)*
  %2799 = load float, float addrspace(1)* %2798, align 4
  %2800 = fmul reassoc nsz arcp contract float %2799, %4, !spirv.Decorations !898
  %2801 = fadd reassoc nsz arcp contract float %2789, %2800, !spirv.Decorations !898
  %2802 = add i64 %.in, %624
  %2803 = inttoptr i64 %2802 to float addrspace(4)*
  %2804 = addrspacecast float addrspace(4)* %2803 to float addrspace(1)*
  store float %2801, float addrspace(1)* %2804, align 4
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, %2794, %2790
  br i1 %314, label %2805, label %._crit_edge70.2.14..preheader1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.14..preheader1.14_crit_edge:      ; preds = %._crit_edge70.2.14
  br label %.preheader1.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2805:                                             ; preds = %._crit_edge70.2.14
  %2806 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !898
  br i1 %81, label %2811, label %2807, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2807:                                             ; preds = %2805
  %2808 = add i64 %.in, %626
  %2809 = inttoptr i64 %2808 to float addrspace(4)*
  %2810 = addrspacecast float addrspace(4)* %2809 to float addrspace(1)*
  store float %2806, float addrspace(1)* %2810, align 4
  br label %.preheader1.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2811:                                             ; preds = %2805
  %2812 = add i64 %.in3821, %sink_3840
  %2813 = add i64 %2812, %sink_3826
  %2814 = inttoptr i64 %2813 to float addrspace(4)*
  %2815 = addrspacecast float addrspace(4)* %2814 to float addrspace(1)*
  %2816 = load float, float addrspace(1)* %2815, align 4
  %2817 = fmul reassoc nsz arcp contract float %2816, %4, !spirv.Decorations !898
  %2818 = fadd reassoc nsz arcp contract float %2806, %2817, !spirv.Decorations !898
  %2819 = add i64 %.in, %626
  %2820 = inttoptr i64 %2819 to float addrspace(4)*
  %2821 = addrspacecast float addrspace(4)* %2820 to float addrspace(1)*
  store float %2818, float addrspace(1)* %2821, align 4
  br label %.preheader1.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.14:                                   ; preds = %._crit_edge70.2.14..preheader1.14_crit_edge, %2811, %2807
  %sink_3825 = shl nsw i64 %627, 2
  br i1 %318, label %2822, label %.preheader1.14.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.14.._crit_edge70.15_crit_edge:        ; preds = %.preheader1.14
  br label %._crit_edge70.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2822:                                             ; preds = %.preheader1.14
  %2823 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !898
  br i1 %81, label %2828, label %2824, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2824:                                             ; preds = %2822
  %2825 = add i64 %.in, %629
  %2826 = inttoptr i64 %2825 to float addrspace(4)*
  %2827 = addrspacecast float addrspace(4)* %2826 to float addrspace(1)*
  store float %2823, float addrspace(1)* %2827, align 4
  br label %._crit_edge70.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2828:                                             ; preds = %2822
  %2829 = add i64 %.in3821, %sink_3844
  %2830 = add i64 %2829, %sink_3825
  %2831 = inttoptr i64 %2830 to float addrspace(4)*
  %2832 = addrspacecast float addrspace(4)* %2831 to float addrspace(1)*
  %2833 = load float, float addrspace(1)* %2832, align 4
  %2834 = fmul reassoc nsz arcp contract float %2833, %4, !spirv.Decorations !898
  %2835 = fadd reassoc nsz arcp contract float %2823, %2834, !spirv.Decorations !898
  %2836 = add i64 %.in, %629
  %2837 = inttoptr i64 %2836 to float addrspace(4)*
  %2838 = addrspacecast float addrspace(4)* %2837 to float addrspace(1)*
  store float %2835, float addrspace(1)* %2838, align 4
  br label %._crit_edge70.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.15:                                 ; preds = %.preheader1.14.._crit_edge70.15_crit_edge, %2828, %2824
  br i1 %321, label %2839, label %._crit_edge70.15.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.15.._crit_edge70.1.15_crit_edge:    ; preds = %._crit_edge70.15
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2839:                                             ; preds = %._crit_edge70.15
  %2840 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !898
  br i1 %81, label %2845, label %2841, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2841:                                             ; preds = %2839
  %2842 = add i64 %.in, %631
  %2843 = inttoptr i64 %2842 to float addrspace(4)*
  %2844 = addrspacecast float addrspace(4)* %2843 to float addrspace(1)*
  store float %2840, float addrspace(1)* %2844, align 4
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2845:                                             ; preds = %2839
  %2846 = add i64 %.in3821, %sink_3842
  %2847 = add i64 %2846, %sink_3825
  %2848 = inttoptr i64 %2847 to float addrspace(4)*
  %2849 = addrspacecast float addrspace(4)* %2848 to float addrspace(1)*
  %2850 = load float, float addrspace(1)* %2849, align 4
  %2851 = fmul reassoc nsz arcp contract float %2850, %4, !spirv.Decorations !898
  %2852 = fadd reassoc nsz arcp contract float %2840, %2851, !spirv.Decorations !898
  %2853 = add i64 %.in, %631
  %2854 = inttoptr i64 %2853 to float addrspace(4)*
  %2855 = addrspacecast float addrspace(4)* %2854 to float addrspace(1)*
  store float %2852, float addrspace(1)* %2855, align 4
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15.._crit_edge70.1.15_crit_edge, %2845, %2841
  br i1 %324, label %2856, label %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.15.._crit_edge70.2.15_crit_edge:  ; preds = %._crit_edge70.1.15
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2856:                                             ; preds = %._crit_edge70.1.15
  %2857 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !898
  br i1 %81, label %2862, label %2858, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2858:                                             ; preds = %2856
  %2859 = add i64 %.in, %633
  %2860 = inttoptr i64 %2859 to float addrspace(4)*
  %2861 = addrspacecast float addrspace(4)* %2860 to float addrspace(1)*
  store float %2857, float addrspace(1)* %2861, align 4
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2862:                                             ; preds = %2856
  %2863 = add i64 %.in3821, %sink_3841
  %2864 = add i64 %2863, %sink_3825
  %2865 = inttoptr i64 %2864 to float addrspace(4)*
  %2866 = addrspacecast float addrspace(4)* %2865 to float addrspace(1)*
  %2867 = load float, float addrspace(1)* %2866, align 4
  %2868 = fmul reassoc nsz arcp contract float %2867, %4, !spirv.Decorations !898
  %2869 = fadd reassoc nsz arcp contract float %2857, %2868, !spirv.Decorations !898
  %2870 = add i64 %.in, %633
  %2871 = inttoptr i64 %2870 to float addrspace(4)*
  %2872 = addrspacecast float addrspace(4)* %2871 to float addrspace(1)*
  store float %2869, float addrspace(1)* %2872, align 4
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, %2862, %2858
  br i1 %327, label %2873, label %._crit_edge70.2.15..preheader1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.15..preheader1.15_crit_edge:      ; preds = %._crit_edge70.2.15
  br label %.preheader1.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2873:                                             ; preds = %._crit_edge70.2.15
  %2874 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !898
  br i1 %81, label %2879, label %2875, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2875:                                             ; preds = %2873
  %2876 = add i64 %.in, %635
  %2877 = inttoptr i64 %2876 to float addrspace(4)*
  %2878 = addrspacecast float addrspace(4)* %2877 to float addrspace(1)*
  store float %2874, float addrspace(1)* %2878, align 4
  br label %.preheader1.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2879:                                             ; preds = %2873
  %2880 = add i64 %.in3821, %sink_3840
  %2881 = add i64 %2880, %sink_3825
  %2882 = inttoptr i64 %2881 to float addrspace(4)*
  %2883 = addrspacecast float addrspace(4)* %2882 to float addrspace(1)*
  %2884 = load float, float addrspace(1)* %2883, align 4
  %2885 = fmul reassoc nsz arcp contract float %2884, %4, !spirv.Decorations !898
  %2886 = fadd reassoc nsz arcp contract float %2874, %2885, !spirv.Decorations !898
  %2887 = add i64 %.in, %635
  %2888 = inttoptr i64 %2887 to float addrspace(4)*
  %2889 = addrspacecast float addrspace(4)* %2888 to float addrspace(1)*
  store float %2886, float addrspace(1)* %2889, align 4
  br label %.preheader1.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.15:                                   ; preds = %._crit_edge70.2.15..preheader1.15_crit_edge, %2879, %2875
  %2890 = add i32 %646, %52
  %2891 = icmp slt i32 %2890, %8
  br i1 %2891, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
  %2892 = add i64 %.in3823, %636
  %2893 = add i64 %.in3822, %637
  %sink_ = bitcast <2 x i32> %644 to i64
  %2894 = add i64 %.in3821, %sink_
  %2895 = add i64 %.in, %645
  br label %.preheader2.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = bitcast i64 %9 to <2 x i32>
  %15 = extractelement <2 x i32> %14, i32 0
  %16 = extractelement <2 x i32> %14, i32 1
  %17 = bitcast i64 %10 to <2 x i32>
  %18 = extractelement <2 x i32> %17, i32 0
  %19 = extractelement <2 x i32> %17, i32 1
  %20 = bitcast i64 %11 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = extractelement <2 x i32> %20, i32 1
  %23 = bitcast i64 %12 to <2 x i32>
  %24 = extractelement <2 x i32> %23, i32 0
  %25 = extractelement <2 x i32> %23, i32 1
  %26 = extractelement <8 x i32> %r0, i32 7
  %27 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %15, i32 %16)
  %28 = extractvalue { i32, i32 } %27, 0
  %29 = extractvalue { i32, i32 } %27, 1
  %30 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %18, i32 %19)
  %31 = extractvalue { i32, i32 } %30, 0
  %32 = extractvalue { i32, i32 } %30, 1
  %33 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %21, i32 %22)
  %34 = extractvalue { i32, i32 } %33, 0
  %35 = extractvalue { i32, i32 } %33, 1
  %36 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %24, i32 %25)
  %37 = extractvalue { i32, i32 } %36, 0
  %38 = extractvalue { i32, i32 } %36, 1
  %39 = icmp slt i32 %26, %8
  br i1 %39, label %.lr.ph, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge72_crit_edge:                         ; preds = %13
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %13
  %40 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = extractelement <2 x i32> %40, i32 1
  %43 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = extractelement <2 x i32> %43, i32 1
  %46 = bitcast i64 %const_reg_qword7 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i32 0
  %48 = extractelement <2 x i32> %46, i32 1
  %49 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %50 = extractelement <2 x i32> %49, i32 0
  %51 = extractelement <2 x i32> %49, i32 1
  %52 = extractelement <3 x i32> %numWorkGroups, i32 2
  %53 = extractelement <3 x i32> %localSize, i32 0
  %54 = extractelement <3 x i32> %localSize, i32 1
  %55 = extractelement <8 x i32> %r0, i32 1
  %56 = extractelement <8 x i32> %r0, i32 6
  %57 = mul i32 %55, %53
  %58 = zext i16 %localIdX to i32
  %59 = add i32 %57, %58
  %60 = shl i32 %59, 2
  %61 = mul i32 %56, %54
  %62 = zext i16 %localIdY to i32
  %63 = add i32 %61, %62
  %64 = shl i32 %63, 2
  %65 = insertelement <2 x i32> undef, i32 %28, i32 0
  %66 = insertelement <2 x i32> %65, i32 %29, i32 1
  %67 = bitcast <2 x i32> %66 to i64
  %68 = shl i64 %67, 1
  %69 = add i64 %68, %const_reg_qword
  %70 = insertelement <2 x i32> undef, i32 %31, i32 0
  %71 = insertelement <2 x i32> %70, i32 %32, i32 1
  %72 = bitcast <2 x i32> %71 to i64
  %73 = shl i64 %72, 1
  %74 = add i64 %73, %const_reg_qword4
  %75 = insertelement <2 x i32> undef, i32 %34, i32 0
  %76 = insertelement <2 x i32> %75, i32 %35, i32 1
  %77 = bitcast <2 x i32> %76 to i64
  %.op = shl i64 %77, 2
  %78 = bitcast i64 %.op to <2 x i32>
  %79 = extractelement <2 x i32> %78, i32 0
  %80 = extractelement <2 x i32> %78, i32 1
  %81 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !898
  %82 = select i1 %81, i32 %79, i32 0
  %83 = select i1 %81, i32 %80, i32 0
  %84 = insertelement <2 x i32> undef, i32 %82, i32 0
  %85 = insertelement <2 x i32> %84, i32 %83, i32 1
  %86 = bitcast <2 x i32> %85 to i64
  %87 = add i64 %86, %const_reg_qword6
  %88 = insertelement <2 x i32> undef, i32 %37, i32 0
  %89 = insertelement <2 x i32> %88, i32 %38, i32 1
  %90 = bitcast <2 x i32> %89 to i64
  %91 = shl i64 %90, 2
  %92 = add i64 %91, %const_reg_qword8
  %93 = icmp sgt i32 %const_reg_dword2, 0
  %94 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %15, i32 %16)
  %95 = extractvalue { i32, i32 } %94, 0
  %96 = extractvalue { i32, i32 } %94, 1
  %97 = insertelement <2 x i32> undef, i32 %95, i32 0
  %98 = insertelement <2 x i32> %97, i32 %96, i32 1
  %99 = bitcast <2 x i32> %98 to i64
  %100 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %18, i32 %19)
  %101 = extractvalue { i32, i32 } %100, 0
  %102 = extractvalue { i32, i32 } %100, 1
  %103 = insertelement <2 x i32> undef, i32 %101, i32 0
  %104 = insertelement <2 x i32> %103, i32 %102, i32 1
  %105 = bitcast <2 x i32> %104 to i64
  %106 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %21, i32 %22)
  %107 = extractvalue { i32, i32 } %106, 0
  %108 = extractvalue { i32, i32 } %106, 1
  %109 = insertelement <2 x i32> undef, i32 %107, i32 0
  %110 = insertelement <2 x i32> %109, i32 %108, i32 1
  %111 = bitcast <2 x i32> %110 to i64
  %112 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %52, i32 0, i32 %24, i32 %25)
  %113 = extractvalue { i32, i32 } %112, 0
  %114 = extractvalue { i32, i32 } %112, 1
  %115 = insertelement <2 x i32> undef, i32 %113, i32 0
  %116 = insertelement <2 x i32> %115, i32 %114, i32 1
  %117 = bitcast <2 x i32> %116 to i64
  %118 = icmp slt i32 %60, %const_reg_dword
  %119 = icmp slt i32 %64, %const_reg_dword1
  %120 = and i1 %118, %119
  %121 = add i32 %60, 1
  %122 = icmp slt i32 %121, %const_reg_dword
  %123 = icmp slt i32 %64, %const_reg_dword1
  %124 = and i1 %122, %123
  %125 = add i32 %60, 2
  %126 = icmp slt i32 %125, %const_reg_dword
  %127 = icmp slt i32 %64, %const_reg_dword1
  %128 = and i1 %126, %127
  %129 = add i32 %60, 3
  %130 = icmp slt i32 %129, %const_reg_dword
  %131 = icmp slt i32 %64, %const_reg_dword1
  %132 = and i1 %130, %131
  %133 = add i32 %64, 1
  %134 = icmp slt i32 %133, %const_reg_dword1
  %135 = icmp slt i32 %60, %const_reg_dword
  %136 = and i1 %135, %134
  %137 = icmp slt i32 %121, %const_reg_dword
  %138 = icmp slt i32 %133, %const_reg_dword1
  %139 = and i1 %137, %138
  %140 = icmp slt i32 %125, %const_reg_dword
  %141 = icmp slt i32 %133, %const_reg_dword1
  %142 = and i1 %140, %141
  %143 = icmp slt i32 %129, %const_reg_dword
  %144 = icmp slt i32 %133, %const_reg_dword1
  %145 = and i1 %143, %144
  %146 = add i32 %64, 2
  %147 = icmp slt i32 %146, %const_reg_dword1
  %148 = icmp slt i32 %60, %const_reg_dword
  %149 = and i1 %148, %147
  %150 = icmp slt i32 %121, %const_reg_dword
  %151 = icmp slt i32 %146, %const_reg_dword1
  %152 = and i1 %150, %151
  %153 = icmp slt i32 %125, %const_reg_dword
  %154 = icmp slt i32 %146, %const_reg_dword1
  %155 = and i1 %153, %154
  %156 = icmp slt i32 %129, %const_reg_dword
  %157 = icmp slt i32 %146, %const_reg_dword1
  %158 = and i1 %156, %157
  %159 = add i32 %64, 3
  %160 = icmp slt i32 %159, %const_reg_dword1
  %161 = icmp slt i32 %60, %const_reg_dword
  %162 = and i1 %161, %160
  %163 = icmp slt i32 %121, %const_reg_dword
  %164 = icmp slt i32 %159, %const_reg_dword1
  %165 = and i1 %163, %164
  %166 = icmp slt i32 %125, %const_reg_dword
  %167 = icmp slt i32 %159, %const_reg_dword1
  %168 = and i1 %166, %167
  %169 = icmp slt i32 %129, %const_reg_dword
  %170 = icmp slt i32 %159, %const_reg_dword1
  %171 = and i1 %169, %170
  %172 = ashr i32 %60, 31
  %173 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %41, i32 %42)
  %174 = extractvalue { i32, i32 } %173, 0
  %175 = extractvalue { i32, i32 } %173, 1
  %176 = insertelement <2 x i32> undef, i32 %174, i32 0
  %177 = insertelement <2 x i32> %176, i32 %175, i32 1
  %178 = bitcast <2 x i32> %177 to i64
  %179 = shl i64 %178, 1
  %180 = sext i32 %64 to i64
  %181 = shl nsw i64 %180, 1
  %182 = ashr i32 %121, 31
  %183 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %41, i32 %42)
  %184 = extractvalue { i32, i32 } %183, 0
  %185 = extractvalue { i32, i32 } %183, 1
  %186 = insertelement <2 x i32> undef, i32 %184, i32 0
  %187 = insertelement <2 x i32> %186, i32 %185, i32 1
  %188 = bitcast <2 x i32> %187 to i64
  %189 = shl i64 %188, 1
  %190 = ashr i32 %125, 31
  %191 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %41, i32 %42)
  %192 = extractvalue { i32, i32 } %191, 0
  %193 = extractvalue { i32, i32 } %191, 1
  %194 = insertelement <2 x i32> undef, i32 %192, i32 0
  %195 = insertelement <2 x i32> %194, i32 %193, i32 1
  %196 = bitcast <2 x i32> %195 to i64
  %197 = shl i64 %196, 1
  %198 = ashr i32 %129, 31
  %199 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %41, i32 %42)
  %200 = extractvalue { i32, i32 } %199, 0
  %201 = extractvalue { i32, i32 } %199, 1
  %202 = insertelement <2 x i32> undef, i32 %200, i32 0
  %203 = insertelement <2 x i32> %202, i32 %201, i32 1
  %204 = bitcast <2 x i32> %203 to i64
  %205 = shl i64 %204, 1
  %206 = sext i32 %133 to i64
  %207 = shl nsw i64 %206, 1
  %208 = sext i32 %146 to i64
  %209 = shl nsw i64 %208, 1
  %210 = sext i32 %159 to i64
  %211 = shl nsw i64 %210, 1
  %212 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %50, i32 %51)
  %213 = extractvalue { i32, i32 } %212, 0
  %214 = extractvalue { i32, i32 } %212, 1
  %215 = insertelement <2 x i32> undef, i32 %213, i32 0
  %216 = insertelement <2 x i32> %215, i32 %214, i32 1
  %217 = bitcast <2 x i32> %216 to i64
  %218 = add nsw i64 %217, %180
  %219 = shl i64 %218, 2
  %220 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %60, i32 %172, i32 %47, i32 %48)
  %221 = extractvalue { i32, i32 } %220, 0
  %222 = extractvalue { i32, i32 } %220, 1
  %223 = insertelement <2 x i32> undef, i32 %221, i32 0
  %224 = insertelement <2 x i32> %223, i32 %222, i32 1
  %225 = bitcast <2 x i32> %224 to i64
  %226 = shl i64 %225, 2
  %227 = shl nsw i64 %180, 2
  %228 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %50, i32 %51)
  %229 = extractvalue { i32, i32 } %228, 0
  %230 = extractvalue { i32, i32 } %228, 1
  %231 = insertelement <2 x i32> undef, i32 %229, i32 0
  %232 = insertelement <2 x i32> %231, i32 %230, i32 1
  %233 = bitcast <2 x i32> %232 to i64
  %234 = add nsw i64 %233, %180
  %235 = shl i64 %234, 2
  %236 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %121, i32 %182, i32 %47, i32 %48)
  %237 = extractvalue { i32, i32 } %236, 0
  %238 = extractvalue { i32, i32 } %236, 1
  %239 = insertelement <2 x i32> undef, i32 %237, i32 0
  %240 = insertelement <2 x i32> %239, i32 %238, i32 1
  %241 = bitcast <2 x i32> %240 to i64
  %242 = shl i64 %241, 2
  %243 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %50, i32 %51)
  %244 = extractvalue { i32, i32 } %243, 0
  %245 = extractvalue { i32, i32 } %243, 1
  %246 = insertelement <2 x i32> undef, i32 %244, i32 0
  %247 = insertelement <2 x i32> %246, i32 %245, i32 1
  %248 = bitcast <2 x i32> %247 to i64
  %249 = add nsw i64 %248, %180
  %250 = shl i64 %249, 2
  %251 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %125, i32 %190, i32 %47, i32 %48)
  %252 = extractvalue { i32, i32 } %251, 0
  %253 = extractvalue { i32, i32 } %251, 1
  %254 = insertelement <2 x i32> undef, i32 %252, i32 0
  %255 = insertelement <2 x i32> %254, i32 %253, i32 1
  %256 = bitcast <2 x i32> %255 to i64
  %257 = shl i64 %256, 2
  %258 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %50, i32 %51)
  %259 = extractvalue { i32, i32 } %258, 0
  %260 = extractvalue { i32, i32 } %258, 1
  %261 = insertelement <2 x i32> undef, i32 %259, i32 0
  %262 = insertelement <2 x i32> %261, i32 %260, i32 1
  %263 = bitcast <2 x i32> %262 to i64
  %264 = add nsw i64 %263, %180
  %265 = shl i64 %264, 2
  %266 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %129, i32 %198, i32 %47, i32 %48)
  %267 = extractvalue { i32, i32 } %266, 0
  %268 = extractvalue { i32, i32 } %266, 1
  %269 = insertelement <2 x i32> undef, i32 %267, i32 0
  %270 = insertelement <2 x i32> %269, i32 %268, i32 1
  %271 = bitcast <2 x i32> %270 to i64
  %272 = shl i64 %271, 2
  %273 = add nsw i64 %217, %206
  %274 = shl i64 %273, 2
  %275 = shl nsw i64 %206, 2
  %276 = add nsw i64 %233, %206
  %277 = shl i64 %276, 2
  %278 = add nsw i64 %248, %206
  %279 = shl i64 %278, 2
  %280 = add nsw i64 %263, %206
  %281 = shl i64 %280, 2
  %282 = add nsw i64 %217, %208
  %283 = shl i64 %282, 2
  %284 = shl nsw i64 %208, 2
  %285 = add nsw i64 %233, %208
  %286 = shl i64 %285, 2
  %287 = add nsw i64 %248, %208
  %288 = shl i64 %287, 2
  %289 = add nsw i64 %263, %208
  %290 = shl i64 %289, 2
  %291 = add nsw i64 %217, %210
  %292 = shl i64 %291, 2
  %293 = shl nsw i64 %210, 2
  %294 = add nsw i64 %233, %210
  %295 = shl i64 %294, 2
  %296 = add nsw i64 %248, %210
  %297 = shl i64 %296, 2
  %298 = add nsw i64 %263, %210
  %299 = shl i64 %298, 2
  %300 = shl i64 %99, 1
  %301 = shl i64 %105, 1
  %.op991 = shl i64 %111, 2
  %302 = bitcast i64 %.op991 to <2 x i32>
  %303 = extractelement <2 x i32> %302, i32 0
  %304 = extractelement <2 x i32> %302, i32 1
  %305 = select i1 %81, i32 %303, i32 0
  %306 = select i1 %81, i32 %304, i32 0
  %307 = insertelement <2 x i32> undef, i32 %305, i32 0
  %308 = insertelement <2 x i32> %307, i32 %306, i32 1
  %309 = bitcast <2 x i32> %308 to i64
  %310 = shl i64 %117, 2
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader2.preheader:                            ; preds = %.preheader1.3..preheader2.preheader_crit_edge, %.lr.ph
  %311 = phi i32 [ %26, %.lr.ph ], [ %1039, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %92, %.lr.ph ], [ %1044, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in988 = phi i64 [ %87, %.lr.ph ], [ %1043, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in989 = phi i64 [ %74, %.lr.ph ], [ %1042, %.preheader1.3..preheader2.preheader_crit_edge ]
  %.in990 = phi i64 [ %69, %.lr.ph ], [ %1041, %.preheader1.3..preheader2.preheader_crit_edge ]
  br i1 %93, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
  br label %.preheader1.preheader, !stats.blockFrequency.digits !917, !stats.blockFrequency.scale !879

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
  %312 = add i64 %.in990, %179
  %313 = add i64 %.in990, %189
  %314 = add i64 %.in990, %197
  %315 = add i64 %.in990, %205
  br label %.preheader.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader.preheader:                             ; preds = %.preheader.3..preheader.preheader_crit_edge, %.preheader.preheader.preheader
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
  br i1 %120, label %333, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

333:                                              ; preds = %.preheader.preheader
  %.sroa.64.0.insert.ext = zext i32 %332 to i64
  %334 = shl nuw nsw i64 %.sroa.64.0.insert.ext, 1
  %335 = add i64 %312, %334
  %336 = inttoptr i64 %335 to i16 addrspace(4)*
  %337 = addrspacecast i16 addrspace(4)* %336 to i16 addrspace(1)*
  %338 = load i16, i16 addrspace(1)* %337, align 2
  %339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %340 = extractvalue { i32, i32 } %339, 0
  %341 = extractvalue { i32, i32 } %339, 1
  %342 = insertelement <2 x i32> undef, i32 %340, i32 0
  %343 = insertelement <2 x i32> %342, i32 %341, i32 1
  %344 = bitcast <2 x i32> %343 to i64
  %345 = shl i64 %344, 1
  %346 = add i64 %.in989, %345
  %347 = add i64 %346, %181
  %348 = inttoptr i64 %347 to i16 addrspace(4)*
  %349 = addrspacecast i16 addrspace(4)* %348 to i16 addrspace(1)*
  %350 = load i16, i16 addrspace(1)* %349, align 2
  %351 = zext i16 %338 to i32
  %352 = shl nuw i32 %351, 16, !spirv.Decorations !921
  %353 = bitcast i32 %352 to float
  %354 = zext i16 %350 to i32
  %355 = shl nuw i32 %354, 16, !spirv.Decorations !921
  %356 = bitcast i32 %355 to float
  %357 = fmul reassoc nsz arcp contract float %353, %356, !spirv.Decorations !898
  %358 = fadd reassoc nsz arcp contract float %357, %331, !spirv.Decorations !898
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %333
  %359 = phi float [ %358, %333 ], [ %331, %.preheader.preheader.._crit_edge_crit_edge ]
  br i1 %124, label %360, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

360:                                              ; preds = %._crit_edge
  %.sroa.64.0.insert.ext203 = zext i32 %332 to i64
  %361 = shl nuw nsw i64 %.sroa.64.0.insert.ext203, 1
  %362 = add i64 %313, %361
  %363 = inttoptr i64 %362 to i16 addrspace(4)*
  %364 = addrspacecast i16 addrspace(4)* %363 to i16 addrspace(1)*
  %365 = load i16, i16 addrspace(1)* %364, align 2
  %366 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %367 = extractvalue { i32, i32 } %366, 0
  %368 = extractvalue { i32, i32 } %366, 1
  %369 = insertelement <2 x i32> undef, i32 %367, i32 0
  %370 = insertelement <2 x i32> %369, i32 %368, i32 1
  %371 = bitcast <2 x i32> %370 to i64
  %372 = shl i64 %371, 1
  %373 = add i64 %.in989, %372
  %374 = add i64 %373, %181
  %375 = inttoptr i64 %374 to i16 addrspace(4)*
  %376 = addrspacecast i16 addrspace(4)* %375 to i16 addrspace(1)*
  %377 = load i16, i16 addrspace(1)* %376, align 2
  %378 = zext i16 %365 to i32
  %379 = shl nuw i32 %378, 16, !spirv.Decorations !921
  %380 = bitcast i32 %379 to float
  %381 = zext i16 %377 to i32
  %382 = shl nuw i32 %381, 16, !spirv.Decorations !921
  %383 = bitcast i32 %382 to float
  %384 = fmul reassoc nsz arcp contract float %380, %383, !spirv.Decorations !898
  %385 = fadd reassoc nsz arcp contract float %384, %330, !spirv.Decorations !898
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %360
  %386 = phi float [ %385, %360 ], [ %330, %._crit_edge.._crit_edge.1_crit_edge ]
  br i1 %128, label %387, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

387:                                              ; preds = %._crit_edge.1
  %.sroa.64.0.insert.ext208 = zext i32 %332 to i64
  %388 = shl nuw nsw i64 %.sroa.64.0.insert.ext208, 1
  %389 = add i64 %314, %388
  %390 = inttoptr i64 %389 to i16 addrspace(4)*
  %391 = addrspacecast i16 addrspace(4)* %390 to i16 addrspace(1)*
  %392 = load i16, i16 addrspace(1)* %391, align 2
  %393 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %394 = extractvalue { i32, i32 } %393, 0
  %395 = extractvalue { i32, i32 } %393, 1
  %396 = insertelement <2 x i32> undef, i32 %394, i32 0
  %397 = insertelement <2 x i32> %396, i32 %395, i32 1
  %398 = bitcast <2 x i32> %397 to i64
  %399 = shl i64 %398, 1
  %400 = add i64 %.in989, %399
  %401 = add i64 %400, %181
  %402 = inttoptr i64 %401 to i16 addrspace(4)*
  %403 = addrspacecast i16 addrspace(4)* %402 to i16 addrspace(1)*
  %404 = load i16, i16 addrspace(1)* %403, align 2
  %405 = zext i16 %392 to i32
  %406 = shl nuw i32 %405, 16, !spirv.Decorations !921
  %407 = bitcast i32 %406 to float
  %408 = zext i16 %404 to i32
  %409 = shl nuw i32 %408, 16, !spirv.Decorations !921
  %410 = bitcast i32 %409 to float
  %411 = fmul reassoc nsz arcp contract float %407, %410, !spirv.Decorations !898
  %412 = fadd reassoc nsz arcp contract float %411, %329, !spirv.Decorations !898
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %387
  %413 = phi float [ %412, %387 ], [ %329, %._crit_edge.1.._crit_edge.2_crit_edge ]
  br i1 %132, label %414, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

414:                                              ; preds = %._crit_edge.2
  %.sroa.64.0.insert.ext213 = zext i32 %332 to i64
  %415 = shl nuw nsw i64 %.sroa.64.0.insert.ext213, 1
  %416 = add i64 %315, %415
  %417 = inttoptr i64 %416 to i16 addrspace(4)*
  %418 = addrspacecast i16 addrspace(4)* %417 to i16 addrspace(1)*
  %419 = load i16, i16 addrspace(1)* %418, align 2
  %420 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %421 = extractvalue { i32, i32 } %420, 0
  %422 = extractvalue { i32, i32 } %420, 1
  %423 = insertelement <2 x i32> undef, i32 %421, i32 0
  %424 = insertelement <2 x i32> %423, i32 %422, i32 1
  %425 = bitcast <2 x i32> %424 to i64
  %426 = shl i64 %425, 1
  %427 = add i64 %.in989, %426
  %428 = add i64 %427, %181
  %429 = inttoptr i64 %428 to i16 addrspace(4)*
  %430 = addrspacecast i16 addrspace(4)* %429 to i16 addrspace(1)*
  %431 = load i16, i16 addrspace(1)* %430, align 2
  %432 = zext i16 %419 to i32
  %433 = shl nuw i32 %432, 16, !spirv.Decorations !921
  %434 = bitcast i32 %433 to float
  %435 = zext i16 %431 to i32
  %436 = shl nuw i32 %435, 16, !spirv.Decorations !921
  %437 = bitcast i32 %436 to float
  %438 = fmul reassoc nsz arcp contract float %434, %437, !spirv.Decorations !898
  %439 = fadd reassoc nsz arcp contract float %438, %328, !spirv.Decorations !898
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %414
  %440 = phi float [ %439, %414 ], [ %328, %._crit_edge.2..preheader_crit_edge ]
  br i1 %136, label %441, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

441:                                              ; preds = %.preheader
  %.sroa.64.0.insert.ext218 = zext i32 %332 to i64
  %442 = shl nuw nsw i64 %.sroa.64.0.insert.ext218, 1
  %443 = add i64 %312, %442
  %444 = inttoptr i64 %443 to i16 addrspace(4)*
  %445 = addrspacecast i16 addrspace(4)* %444 to i16 addrspace(1)*
  %446 = load i16, i16 addrspace(1)* %445, align 2
  %447 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %448 = extractvalue { i32, i32 } %447, 0
  %449 = extractvalue { i32, i32 } %447, 1
  %450 = insertelement <2 x i32> undef, i32 %448, i32 0
  %451 = insertelement <2 x i32> %450, i32 %449, i32 1
  %452 = bitcast <2 x i32> %451 to i64
  %453 = shl i64 %452, 1
  %454 = add i64 %.in989, %453
  %455 = add i64 %454, %207
  %456 = inttoptr i64 %455 to i16 addrspace(4)*
  %457 = addrspacecast i16 addrspace(4)* %456 to i16 addrspace(1)*
  %458 = load i16, i16 addrspace(1)* %457, align 2
  %459 = zext i16 %446 to i32
  %460 = shl nuw i32 %459, 16, !spirv.Decorations !921
  %461 = bitcast i32 %460 to float
  %462 = zext i16 %458 to i32
  %463 = shl nuw i32 %462, 16, !spirv.Decorations !921
  %464 = bitcast i32 %463 to float
  %465 = fmul reassoc nsz arcp contract float %461, %464, !spirv.Decorations !898
  %466 = fadd reassoc nsz arcp contract float %465, %327, !spirv.Decorations !898
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %441
  %467 = phi float [ %466, %441 ], [ %327, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %139, label %468, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

468:                                              ; preds = %._crit_edge.173
  %.sroa.64.0.insert.ext223 = zext i32 %332 to i64
  %469 = shl nuw nsw i64 %.sroa.64.0.insert.ext223, 1
  %470 = add i64 %313, %469
  %471 = inttoptr i64 %470 to i16 addrspace(4)*
  %472 = addrspacecast i16 addrspace(4)* %471 to i16 addrspace(1)*
  %473 = load i16, i16 addrspace(1)* %472, align 2
  %474 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %475 = extractvalue { i32, i32 } %474, 0
  %476 = extractvalue { i32, i32 } %474, 1
  %477 = insertelement <2 x i32> undef, i32 %475, i32 0
  %478 = insertelement <2 x i32> %477, i32 %476, i32 1
  %479 = bitcast <2 x i32> %478 to i64
  %480 = shl i64 %479, 1
  %481 = add i64 %.in989, %480
  %482 = add i64 %481, %207
  %483 = inttoptr i64 %482 to i16 addrspace(4)*
  %484 = addrspacecast i16 addrspace(4)* %483 to i16 addrspace(1)*
  %485 = load i16, i16 addrspace(1)* %484, align 2
  %486 = zext i16 %473 to i32
  %487 = shl nuw i32 %486, 16, !spirv.Decorations !921
  %488 = bitcast i32 %487 to float
  %489 = zext i16 %485 to i32
  %490 = shl nuw i32 %489, 16, !spirv.Decorations !921
  %491 = bitcast i32 %490 to float
  %492 = fmul reassoc nsz arcp contract float %488, %491, !spirv.Decorations !898
  %493 = fadd reassoc nsz arcp contract float %492, %326, !spirv.Decorations !898
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %468
  %494 = phi float [ %493, %468 ], [ %326, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %142, label %495, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

495:                                              ; preds = %._crit_edge.1.1
  %.sroa.64.0.insert.ext228 = zext i32 %332 to i64
  %496 = shl nuw nsw i64 %.sroa.64.0.insert.ext228, 1
  %497 = add i64 %314, %496
  %498 = inttoptr i64 %497 to i16 addrspace(4)*
  %499 = addrspacecast i16 addrspace(4)* %498 to i16 addrspace(1)*
  %500 = load i16, i16 addrspace(1)* %499, align 2
  %501 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %502 = extractvalue { i32, i32 } %501, 0
  %503 = extractvalue { i32, i32 } %501, 1
  %504 = insertelement <2 x i32> undef, i32 %502, i32 0
  %505 = insertelement <2 x i32> %504, i32 %503, i32 1
  %506 = bitcast <2 x i32> %505 to i64
  %507 = shl i64 %506, 1
  %508 = add i64 %.in989, %507
  %509 = add i64 %508, %207
  %510 = inttoptr i64 %509 to i16 addrspace(4)*
  %511 = addrspacecast i16 addrspace(4)* %510 to i16 addrspace(1)*
  %512 = load i16, i16 addrspace(1)* %511, align 2
  %513 = zext i16 %500 to i32
  %514 = shl nuw i32 %513, 16, !spirv.Decorations !921
  %515 = bitcast i32 %514 to float
  %516 = zext i16 %512 to i32
  %517 = shl nuw i32 %516, 16, !spirv.Decorations !921
  %518 = bitcast i32 %517 to float
  %519 = fmul reassoc nsz arcp contract float %515, %518, !spirv.Decorations !898
  %520 = fadd reassoc nsz arcp contract float %519, %325, !spirv.Decorations !898
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %495
  %521 = phi float [ %520, %495 ], [ %325, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %145, label %522, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

522:                                              ; preds = %._crit_edge.2.1
  %.sroa.64.0.insert.ext233 = zext i32 %332 to i64
  %523 = shl nuw nsw i64 %.sroa.64.0.insert.ext233, 1
  %524 = add i64 %315, %523
  %525 = inttoptr i64 %524 to i16 addrspace(4)*
  %526 = addrspacecast i16 addrspace(4)* %525 to i16 addrspace(1)*
  %527 = load i16, i16 addrspace(1)* %526, align 2
  %528 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %529 = extractvalue { i32, i32 } %528, 0
  %530 = extractvalue { i32, i32 } %528, 1
  %531 = insertelement <2 x i32> undef, i32 %529, i32 0
  %532 = insertelement <2 x i32> %531, i32 %530, i32 1
  %533 = bitcast <2 x i32> %532 to i64
  %534 = shl i64 %533, 1
  %535 = add i64 %.in989, %534
  %536 = add i64 %535, %207
  %537 = inttoptr i64 %536 to i16 addrspace(4)*
  %538 = addrspacecast i16 addrspace(4)* %537 to i16 addrspace(1)*
  %539 = load i16, i16 addrspace(1)* %538, align 2
  %540 = zext i16 %527 to i32
  %541 = shl nuw i32 %540, 16, !spirv.Decorations !921
  %542 = bitcast i32 %541 to float
  %543 = zext i16 %539 to i32
  %544 = shl nuw i32 %543, 16, !spirv.Decorations !921
  %545 = bitcast i32 %544 to float
  %546 = fmul reassoc nsz arcp contract float %542, %545, !spirv.Decorations !898
  %547 = fadd reassoc nsz arcp contract float %546, %324, !spirv.Decorations !898
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %522
  %548 = phi float [ %547, %522 ], [ %324, %._crit_edge.2.1..preheader.1_crit_edge ]
  br i1 %149, label %549, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

549:                                              ; preds = %.preheader.1
  %.sroa.64.0.insert.ext238 = zext i32 %332 to i64
  %550 = shl nuw nsw i64 %.sroa.64.0.insert.ext238, 1
  %551 = add i64 %312, %550
  %552 = inttoptr i64 %551 to i16 addrspace(4)*
  %553 = addrspacecast i16 addrspace(4)* %552 to i16 addrspace(1)*
  %554 = load i16, i16 addrspace(1)* %553, align 2
  %555 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %556 = extractvalue { i32, i32 } %555, 0
  %557 = extractvalue { i32, i32 } %555, 1
  %558 = insertelement <2 x i32> undef, i32 %556, i32 0
  %559 = insertelement <2 x i32> %558, i32 %557, i32 1
  %560 = bitcast <2 x i32> %559 to i64
  %561 = shl i64 %560, 1
  %562 = add i64 %.in989, %561
  %563 = add i64 %562, %209
  %564 = inttoptr i64 %563 to i16 addrspace(4)*
  %565 = addrspacecast i16 addrspace(4)* %564 to i16 addrspace(1)*
  %566 = load i16, i16 addrspace(1)* %565, align 2
  %567 = zext i16 %554 to i32
  %568 = shl nuw i32 %567, 16, !spirv.Decorations !921
  %569 = bitcast i32 %568 to float
  %570 = zext i16 %566 to i32
  %571 = shl nuw i32 %570, 16, !spirv.Decorations !921
  %572 = bitcast i32 %571 to float
  %573 = fmul reassoc nsz arcp contract float %569, %572, !spirv.Decorations !898
  %574 = fadd reassoc nsz arcp contract float %573, %323, !spirv.Decorations !898
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %549
  %575 = phi float [ %574, %549 ], [ %323, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %152, label %576, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

576:                                              ; preds = %._crit_edge.274
  %.sroa.64.0.insert.ext243 = zext i32 %332 to i64
  %577 = shl nuw nsw i64 %.sroa.64.0.insert.ext243, 1
  %578 = add i64 %313, %577
  %579 = inttoptr i64 %578 to i16 addrspace(4)*
  %580 = addrspacecast i16 addrspace(4)* %579 to i16 addrspace(1)*
  %581 = load i16, i16 addrspace(1)* %580, align 2
  %582 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %583 = extractvalue { i32, i32 } %582, 0
  %584 = extractvalue { i32, i32 } %582, 1
  %585 = insertelement <2 x i32> undef, i32 %583, i32 0
  %586 = insertelement <2 x i32> %585, i32 %584, i32 1
  %587 = bitcast <2 x i32> %586 to i64
  %588 = shl i64 %587, 1
  %589 = add i64 %.in989, %588
  %590 = add i64 %589, %209
  %591 = inttoptr i64 %590 to i16 addrspace(4)*
  %592 = addrspacecast i16 addrspace(4)* %591 to i16 addrspace(1)*
  %593 = load i16, i16 addrspace(1)* %592, align 2
  %594 = zext i16 %581 to i32
  %595 = shl nuw i32 %594, 16, !spirv.Decorations !921
  %596 = bitcast i32 %595 to float
  %597 = zext i16 %593 to i32
  %598 = shl nuw i32 %597, 16, !spirv.Decorations !921
  %599 = bitcast i32 %598 to float
  %600 = fmul reassoc nsz arcp contract float %596, %599, !spirv.Decorations !898
  %601 = fadd reassoc nsz arcp contract float %600, %322, !spirv.Decorations !898
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %576
  %602 = phi float [ %601, %576 ], [ %322, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %155, label %603, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

603:                                              ; preds = %._crit_edge.1.2
  %.sroa.64.0.insert.ext248 = zext i32 %332 to i64
  %604 = shl nuw nsw i64 %.sroa.64.0.insert.ext248, 1
  %605 = add i64 %314, %604
  %606 = inttoptr i64 %605 to i16 addrspace(4)*
  %607 = addrspacecast i16 addrspace(4)* %606 to i16 addrspace(1)*
  %608 = load i16, i16 addrspace(1)* %607, align 2
  %609 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %610 = extractvalue { i32, i32 } %609, 0
  %611 = extractvalue { i32, i32 } %609, 1
  %612 = insertelement <2 x i32> undef, i32 %610, i32 0
  %613 = insertelement <2 x i32> %612, i32 %611, i32 1
  %614 = bitcast <2 x i32> %613 to i64
  %615 = shl i64 %614, 1
  %616 = add i64 %.in989, %615
  %617 = add i64 %616, %209
  %618 = inttoptr i64 %617 to i16 addrspace(4)*
  %619 = addrspacecast i16 addrspace(4)* %618 to i16 addrspace(1)*
  %620 = load i16, i16 addrspace(1)* %619, align 2
  %621 = zext i16 %608 to i32
  %622 = shl nuw i32 %621, 16, !spirv.Decorations !921
  %623 = bitcast i32 %622 to float
  %624 = zext i16 %620 to i32
  %625 = shl nuw i32 %624, 16, !spirv.Decorations !921
  %626 = bitcast i32 %625 to float
  %627 = fmul reassoc nsz arcp contract float %623, %626, !spirv.Decorations !898
  %628 = fadd reassoc nsz arcp contract float %627, %321, !spirv.Decorations !898
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %603
  %629 = phi float [ %628, %603 ], [ %321, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %158, label %630, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

630:                                              ; preds = %._crit_edge.2.2
  %.sroa.64.0.insert.ext253 = zext i32 %332 to i64
  %631 = shl nuw nsw i64 %.sroa.64.0.insert.ext253, 1
  %632 = add i64 %315, %631
  %633 = inttoptr i64 %632 to i16 addrspace(4)*
  %634 = addrspacecast i16 addrspace(4)* %633 to i16 addrspace(1)*
  %635 = load i16, i16 addrspace(1)* %634, align 2
  %636 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %637 = extractvalue { i32, i32 } %636, 0
  %638 = extractvalue { i32, i32 } %636, 1
  %639 = insertelement <2 x i32> undef, i32 %637, i32 0
  %640 = insertelement <2 x i32> %639, i32 %638, i32 1
  %641 = bitcast <2 x i32> %640 to i64
  %642 = shl i64 %641, 1
  %643 = add i64 %.in989, %642
  %644 = add i64 %643, %209
  %645 = inttoptr i64 %644 to i16 addrspace(4)*
  %646 = addrspacecast i16 addrspace(4)* %645 to i16 addrspace(1)*
  %647 = load i16, i16 addrspace(1)* %646, align 2
  %648 = zext i16 %635 to i32
  %649 = shl nuw i32 %648, 16, !spirv.Decorations !921
  %650 = bitcast i32 %649 to float
  %651 = zext i16 %647 to i32
  %652 = shl nuw i32 %651, 16, !spirv.Decorations !921
  %653 = bitcast i32 %652 to float
  %654 = fmul reassoc nsz arcp contract float %650, %653, !spirv.Decorations !898
  %655 = fadd reassoc nsz arcp contract float %654, %320, !spirv.Decorations !898
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %630
  %656 = phi float [ %655, %630 ], [ %320, %._crit_edge.2.2..preheader.2_crit_edge ]
  br i1 %162, label %657, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

657:                                              ; preds = %.preheader.2
  %.sroa.64.0.insert.ext258 = zext i32 %332 to i64
  %658 = shl nuw nsw i64 %.sroa.64.0.insert.ext258, 1
  %659 = add i64 %312, %658
  %660 = inttoptr i64 %659 to i16 addrspace(4)*
  %661 = addrspacecast i16 addrspace(4)* %660 to i16 addrspace(1)*
  %662 = load i16, i16 addrspace(1)* %661, align 2
  %663 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %664 = extractvalue { i32, i32 } %663, 0
  %665 = extractvalue { i32, i32 } %663, 1
  %666 = insertelement <2 x i32> undef, i32 %664, i32 0
  %667 = insertelement <2 x i32> %666, i32 %665, i32 1
  %668 = bitcast <2 x i32> %667 to i64
  %669 = shl i64 %668, 1
  %670 = add i64 %.in989, %669
  %671 = add i64 %670, %211
  %672 = inttoptr i64 %671 to i16 addrspace(4)*
  %673 = addrspacecast i16 addrspace(4)* %672 to i16 addrspace(1)*
  %674 = load i16, i16 addrspace(1)* %673, align 2
  %675 = zext i16 %662 to i32
  %676 = shl nuw i32 %675, 16, !spirv.Decorations !921
  %677 = bitcast i32 %676 to float
  %678 = zext i16 %674 to i32
  %679 = shl nuw i32 %678, 16, !spirv.Decorations !921
  %680 = bitcast i32 %679 to float
  %681 = fmul reassoc nsz arcp contract float %677, %680, !spirv.Decorations !898
  %682 = fadd reassoc nsz arcp contract float %681, %319, !spirv.Decorations !898
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %657
  %683 = phi float [ %682, %657 ], [ %319, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %165, label %684, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

684:                                              ; preds = %._crit_edge.375
  %.sroa.64.0.insert.ext263 = zext i32 %332 to i64
  %685 = shl nuw nsw i64 %.sroa.64.0.insert.ext263, 1
  %686 = add i64 %313, %685
  %687 = inttoptr i64 %686 to i16 addrspace(4)*
  %688 = addrspacecast i16 addrspace(4)* %687 to i16 addrspace(1)*
  %689 = load i16, i16 addrspace(1)* %688, align 2
  %690 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %691 = extractvalue { i32, i32 } %690, 0
  %692 = extractvalue { i32, i32 } %690, 1
  %693 = insertelement <2 x i32> undef, i32 %691, i32 0
  %694 = insertelement <2 x i32> %693, i32 %692, i32 1
  %695 = bitcast <2 x i32> %694 to i64
  %696 = shl i64 %695, 1
  %697 = add i64 %.in989, %696
  %698 = add i64 %697, %211
  %699 = inttoptr i64 %698 to i16 addrspace(4)*
  %700 = addrspacecast i16 addrspace(4)* %699 to i16 addrspace(1)*
  %701 = load i16, i16 addrspace(1)* %700, align 2
  %702 = zext i16 %689 to i32
  %703 = shl nuw i32 %702, 16, !spirv.Decorations !921
  %704 = bitcast i32 %703 to float
  %705 = zext i16 %701 to i32
  %706 = shl nuw i32 %705, 16, !spirv.Decorations !921
  %707 = bitcast i32 %706 to float
  %708 = fmul reassoc nsz arcp contract float %704, %707, !spirv.Decorations !898
  %709 = fadd reassoc nsz arcp contract float %708, %318, !spirv.Decorations !898
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %684
  %710 = phi float [ %709, %684 ], [ %318, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %168, label %711, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

711:                                              ; preds = %._crit_edge.1.3
  %.sroa.64.0.insert.ext268 = zext i32 %332 to i64
  %712 = shl nuw nsw i64 %.sroa.64.0.insert.ext268, 1
  %713 = add i64 %314, %712
  %714 = inttoptr i64 %713 to i16 addrspace(4)*
  %715 = addrspacecast i16 addrspace(4)* %714 to i16 addrspace(1)*
  %716 = load i16, i16 addrspace(1)* %715, align 2
  %717 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %718 = extractvalue { i32, i32 } %717, 0
  %719 = extractvalue { i32, i32 } %717, 1
  %720 = insertelement <2 x i32> undef, i32 %718, i32 0
  %721 = insertelement <2 x i32> %720, i32 %719, i32 1
  %722 = bitcast <2 x i32> %721 to i64
  %723 = shl i64 %722, 1
  %724 = add i64 %.in989, %723
  %725 = add i64 %724, %211
  %726 = inttoptr i64 %725 to i16 addrspace(4)*
  %727 = addrspacecast i16 addrspace(4)* %726 to i16 addrspace(1)*
  %728 = load i16, i16 addrspace(1)* %727, align 2
  %729 = zext i16 %716 to i32
  %730 = shl nuw i32 %729, 16, !spirv.Decorations !921
  %731 = bitcast i32 %730 to float
  %732 = zext i16 %728 to i32
  %733 = shl nuw i32 %732, 16, !spirv.Decorations !921
  %734 = bitcast i32 %733 to float
  %735 = fmul reassoc nsz arcp contract float %731, %734, !spirv.Decorations !898
  %736 = fadd reassoc nsz arcp contract float %735, %317, !spirv.Decorations !898
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %711
  %737 = phi float [ %736, %711 ], [ %317, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %171, label %738, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

738:                                              ; preds = %._crit_edge.2.3
  %.sroa.64.0.insert.ext273 = zext i32 %332 to i64
  %739 = shl nuw nsw i64 %.sroa.64.0.insert.ext273, 1
  %740 = add i64 %315, %739
  %741 = inttoptr i64 %740 to i16 addrspace(4)*
  %742 = addrspacecast i16 addrspace(4)* %741 to i16 addrspace(1)*
  %743 = load i16, i16 addrspace(1)* %742, align 2
  %744 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %332, i32 0, i32 %44, i32 %45)
  %745 = extractvalue { i32, i32 } %744, 0
  %746 = extractvalue { i32, i32 } %744, 1
  %747 = insertelement <2 x i32> undef, i32 %745, i32 0
  %748 = insertelement <2 x i32> %747, i32 %746, i32 1
  %749 = bitcast <2 x i32> %748 to i64
  %750 = shl i64 %749, 1
  %751 = add i64 %.in989, %750
  %752 = add i64 %751, %211
  %753 = inttoptr i64 %752 to i16 addrspace(4)*
  %754 = addrspacecast i16 addrspace(4)* %753 to i16 addrspace(1)*
  %755 = load i16, i16 addrspace(1)* %754, align 2
  %756 = zext i16 %743 to i32
  %757 = shl nuw i32 %756, 16, !spirv.Decorations !921
  %758 = bitcast i32 %757 to float
  %759 = zext i16 %755 to i32
  %760 = shl nuw i32 %759, 16, !spirv.Decorations !921
  %761 = bitcast i32 %760 to float
  %762 = fmul reassoc nsz arcp contract float %758, %761, !spirv.Decorations !898
  %763 = fadd reassoc nsz arcp contract float %762, %316, !spirv.Decorations !898
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %738
  %764 = phi float [ %763, %738 ], [ %316, %._crit_edge.2.3..preheader.3_crit_edge ]
  %765 = add nuw nsw i32 %332, 1, !spirv.Decorations !923
  %766 = icmp slt i32 %765, %const_reg_dword2
  br i1 %766, label %.preheader.3..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.3..preheader.preheader_crit_edge:      ; preds = %.preheader.3
  br label %.preheader.preheader, !stats.blockFrequency.digits !924, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.3
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
  br label %.preheader1.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
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
  br i1 %120, label %767, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
  br label %._crit_edge70, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

767:                                              ; preds = %.preheader1.preheader
  %768 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !898
  br i1 %81, label %773, label %769, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

769:                                              ; preds = %767
  %770 = add i64 %.in, %219
  %771 = inttoptr i64 %770 to float addrspace(4)*
  %772 = addrspacecast float addrspace(4)* %771 to float addrspace(1)*
  store float %768, float addrspace(1)* %772, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

773:                                              ; preds = %767
  %774 = add i64 %.in988, %226
  %775 = add i64 %774, %227
  %776 = inttoptr i64 %775 to float addrspace(4)*
  %777 = addrspacecast float addrspace(4)* %776 to float addrspace(1)*
  %778 = load float, float addrspace(1)* %777, align 4
  %779 = fmul reassoc nsz arcp contract float %778, %4, !spirv.Decorations !898
  %780 = fadd reassoc nsz arcp contract float %768, %779, !spirv.Decorations !898
  %781 = add i64 %.in, %219
  %782 = inttoptr i64 %781 to float addrspace(4)*
  %783 = addrspacecast float addrspace(4)* %782 to float addrspace(1)*
  store float %780, float addrspace(1)* %783, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %769, %773
  br i1 %124, label %784, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
  br label %._crit_edge70.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

784:                                              ; preds = %._crit_edge70
  %785 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !898
  br i1 %81, label %790, label %786, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

786:                                              ; preds = %784
  %787 = add i64 %.in, %235
  %788 = inttoptr i64 %787 to float addrspace(4)*
  %789 = addrspacecast float addrspace(4)* %788 to float addrspace(1)*
  store float %785, float addrspace(1)* %789, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

790:                                              ; preds = %784
  %791 = add i64 %.in988, %242
  %792 = add i64 %791, %227
  %793 = inttoptr i64 %792 to float addrspace(4)*
  %794 = addrspacecast float addrspace(4)* %793 to float addrspace(1)*
  %795 = load float, float addrspace(1)* %794, align 4
  %796 = fmul reassoc nsz arcp contract float %795, %4, !spirv.Decorations !898
  %797 = fadd reassoc nsz arcp contract float %785, %796, !spirv.Decorations !898
  %798 = add i64 %.in, %235
  %799 = inttoptr i64 %798 to float addrspace(4)*
  %800 = addrspacecast float addrspace(4)* %799 to float addrspace(1)*
  store float %797, float addrspace(1)* %800, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %790, %786
  br i1 %128, label %801, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
  br label %._crit_edge70.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

801:                                              ; preds = %._crit_edge70.1
  %802 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !898
  br i1 %81, label %807, label %803, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

803:                                              ; preds = %801
  %804 = add i64 %.in, %250
  %805 = inttoptr i64 %804 to float addrspace(4)*
  %806 = addrspacecast float addrspace(4)* %805 to float addrspace(1)*
  store float %802, float addrspace(1)* %806, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

807:                                              ; preds = %801
  %808 = add i64 %.in988, %257
  %809 = add i64 %808, %227
  %810 = inttoptr i64 %809 to float addrspace(4)*
  %811 = addrspacecast float addrspace(4)* %810 to float addrspace(1)*
  %812 = load float, float addrspace(1)* %811, align 4
  %813 = fmul reassoc nsz arcp contract float %812, %4, !spirv.Decorations !898
  %814 = fadd reassoc nsz arcp contract float %802, %813, !spirv.Decorations !898
  %815 = add i64 %.in, %250
  %816 = inttoptr i64 %815 to float addrspace(4)*
  %817 = addrspacecast float addrspace(4)* %816 to float addrspace(1)*
  store float %814, float addrspace(1)* %817, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %807, %803
  br i1 %132, label %818, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
  br label %.preheader1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

818:                                              ; preds = %._crit_edge70.2
  %819 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !898
  br i1 %81, label %824, label %820, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

820:                                              ; preds = %818
  %821 = add i64 %.in, %265
  %822 = inttoptr i64 %821 to float addrspace(4)*
  %823 = addrspacecast float addrspace(4)* %822 to float addrspace(1)*
  store float %819, float addrspace(1)* %823, align 4
  br label %.preheader1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

824:                                              ; preds = %818
  %825 = add i64 %.in988, %272
  %826 = add i64 %825, %227
  %827 = inttoptr i64 %826 to float addrspace(4)*
  %828 = addrspacecast float addrspace(4)* %827 to float addrspace(1)*
  %829 = load float, float addrspace(1)* %828, align 4
  %830 = fmul reassoc nsz arcp contract float %829, %4, !spirv.Decorations !898
  %831 = fadd reassoc nsz arcp contract float %819, %830, !spirv.Decorations !898
  %832 = add i64 %.in, %265
  %833 = inttoptr i64 %832 to float addrspace(4)*
  %834 = addrspacecast float addrspace(4)* %833 to float addrspace(1)*
  store float %831, float addrspace(1)* %834, align 4
  br label %.preheader1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %824, %820
  br i1 %136, label %835, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
  br label %._crit_edge70.176, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

835:                                              ; preds = %.preheader1
  %836 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !898
  br i1 %81, label %841, label %837, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

837:                                              ; preds = %835
  %838 = add i64 %.in, %274
  %839 = inttoptr i64 %838 to float addrspace(4)*
  %840 = addrspacecast float addrspace(4)* %839 to float addrspace(1)*
  store float %836, float addrspace(1)* %840, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

841:                                              ; preds = %835
  %842 = add i64 %.in988, %226
  %843 = add i64 %842, %275
  %844 = inttoptr i64 %843 to float addrspace(4)*
  %845 = addrspacecast float addrspace(4)* %844 to float addrspace(1)*
  %846 = load float, float addrspace(1)* %845, align 4
  %847 = fmul reassoc nsz arcp contract float %846, %4, !spirv.Decorations !898
  %848 = fadd reassoc nsz arcp contract float %836, %847, !spirv.Decorations !898
  %849 = add i64 %.in, %274
  %850 = inttoptr i64 %849 to float addrspace(4)*
  %851 = addrspacecast float addrspace(4)* %850 to float addrspace(1)*
  store float %848, float addrspace(1)* %851, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %841, %837
  br i1 %139, label %852, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

852:                                              ; preds = %._crit_edge70.176
  %853 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !898
  br i1 %81, label %858, label %854, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

854:                                              ; preds = %852
  %855 = add i64 %.in, %277
  %856 = inttoptr i64 %855 to float addrspace(4)*
  %857 = addrspacecast float addrspace(4)* %856 to float addrspace(1)*
  store float %853, float addrspace(1)* %857, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

858:                                              ; preds = %852
  %859 = add i64 %.in988, %242
  %860 = add i64 %859, %275
  %861 = inttoptr i64 %860 to float addrspace(4)*
  %862 = addrspacecast float addrspace(4)* %861 to float addrspace(1)*
  %863 = load float, float addrspace(1)* %862, align 4
  %864 = fmul reassoc nsz arcp contract float %863, %4, !spirv.Decorations !898
  %865 = fadd reassoc nsz arcp contract float %853, %864, !spirv.Decorations !898
  %866 = add i64 %.in, %277
  %867 = inttoptr i64 %866 to float addrspace(4)*
  %868 = addrspacecast float addrspace(4)* %867 to float addrspace(1)*
  store float %865, float addrspace(1)* %868, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %858, %854
  br i1 %142, label %869, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

869:                                              ; preds = %._crit_edge70.1.1
  %870 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !898
  br i1 %81, label %875, label %871, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

871:                                              ; preds = %869
  %872 = add i64 %.in, %279
  %873 = inttoptr i64 %872 to float addrspace(4)*
  %874 = addrspacecast float addrspace(4)* %873 to float addrspace(1)*
  store float %870, float addrspace(1)* %874, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

875:                                              ; preds = %869
  %876 = add i64 %.in988, %257
  %877 = add i64 %876, %275
  %878 = inttoptr i64 %877 to float addrspace(4)*
  %879 = addrspacecast float addrspace(4)* %878 to float addrspace(1)*
  %880 = load float, float addrspace(1)* %879, align 4
  %881 = fmul reassoc nsz arcp contract float %880, %4, !spirv.Decorations !898
  %882 = fadd reassoc nsz arcp contract float %870, %881, !spirv.Decorations !898
  %883 = add i64 %.in, %279
  %884 = inttoptr i64 %883 to float addrspace(4)*
  %885 = addrspacecast float addrspace(4)* %884 to float addrspace(1)*
  store float %882, float addrspace(1)* %885, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %875, %871
  br i1 %145, label %886, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
  br label %.preheader1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

886:                                              ; preds = %._crit_edge70.2.1
  %887 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !898
  br i1 %81, label %892, label %888, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

888:                                              ; preds = %886
  %889 = add i64 %.in, %281
  %890 = inttoptr i64 %889 to float addrspace(4)*
  %891 = addrspacecast float addrspace(4)* %890 to float addrspace(1)*
  store float %887, float addrspace(1)* %891, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

892:                                              ; preds = %886
  %893 = add i64 %.in988, %272
  %894 = add i64 %893, %275
  %895 = inttoptr i64 %894 to float addrspace(4)*
  %896 = addrspacecast float addrspace(4)* %895 to float addrspace(1)*
  %897 = load float, float addrspace(1)* %896, align 4
  %898 = fmul reassoc nsz arcp contract float %897, %4, !spirv.Decorations !898
  %899 = fadd reassoc nsz arcp contract float %887, %898, !spirv.Decorations !898
  %900 = add i64 %.in, %281
  %901 = inttoptr i64 %900 to float addrspace(4)*
  %902 = addrspacecast float addrspace(4)* %901 to float addrspace(1)*
  store float %899, float addrspace(1)* %902, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %892, %888
  br i1 %149, label %903, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
  br label %._crit_edge70.277, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

903:                                              ; preds = %.preheader1.1
  %904 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !898
  br i1 %81, label %909, label %905, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

905:                                              ; preds = %903
  %906 = add i64 %.in, %283
  %907 = inttoptr i64 %906 to float addrspace(4)*
  %908 = addrspacecast float addrspace(4)* %907 to float addrspace(1)*
  store float %904, float addrspace(1)* %908, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

909:                                              ; preds = %903
  %910 = add i64 %.in988, %226
  %911 = add i64 %910, %284
  %912 = inttoptr i64 %911 to float addrspace(4)*
  %913 = addrspacecast float addrspace(4)* %912 to float addrspace(1)*
  %914 = load float, float addrspace(1)* %913, align 4
  %915 = fmul reassoc nsz arcp contract float %914, %4, !spirv.Decorations !898
  %916 = fadd reassoc nsz arcp contract float %904, %915, !spirv.Decorations !898
  %917 = add i64 %.in, %283
  %918 = inttoptr i64 %917 to float addrspace(4)*
  %919 = addrspacecast float addrspace(4)* %918 to float addrspace(1)*
  store float %916, float addrspace(1)* %919, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %909, %905
  br i1 %152, label %920, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

920:                                              ; preds = %._crit_edge70.277
  %921 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !898
  br i1 %81, label %926, label %922, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

922:                                              ; preds = %920
  %923 = add i64 %.in, %286
  %924 = inttoptr i64 %923 to float addrspace(4)*
  %925 = addrspacecast float addrspace(4)* %924 to float addrspace(1)*
  store float %921, float addrspace(1)* %925, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

926:                                              ; preds = %920
  %927 = add i64 %.in988, %242
  %928 = add i64 %927, %284
  %929 = inttoptr i64 %928 to float addrspace(4)*
  %930 = addrspacecast float addrspace(4)* %929 to float addrspace(1)*
  %931 = load float, float addrspace(1)* %930, align 4
  %932 = fmul reassoc nsz arcp contract float %931, %4, !spirv.Decorations !898
  %933 = fadd reassoc nsz arcp contract float %921, %932, !spirv.Decorations !898
  %934 = add i64 %.in, %286
  %935 = inttoptr i64 %934 to float addrspace(4)*
  %936 = addrspacecast float addrspace(4)* %935 to float addrspace(1)*
  store float %933, float addrspace(1)* %936, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %926, %922
  br i1 %155, label %937, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

937:                                              ; preds = %._crit_edge70.1.2
  %938 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !898
  br i1 %81, label %943, label %939, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

939:                                              ; preds = %937
  %940 = add i64 %.in, %288
  %941 = inttoptr i64 %940 to float addrspace(4)*
  %942 = addrspacecast float addrspace(4)* %941 to float addrspace(1)*
  store float %938, float addrspace(1)* %942, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

943:                                              ; preds = %937
  %944 = add i64 %.in988, %257
  %945 = add i64 %944, %284
  %946 = inttoptr i64 %945 to float addrspace(4)*
  %947 = addrspacecast float addrspace(4)* %946 to float addrspace(1)*
  %948 = load float, float addrspace(1)* %947, align 4
  %949 = fmul reassoc nsz arcp contract float %948, %4, !spirv.Decorations !898
  %950 = fadd reassoc nsz arcp contract float %938, %949, !spirv.Decorations !898
  %951 = add i64 %.in, %288
  %952 = inttoptr i64 %951 to float addrspace(4)*
  %953 = addrspacecast float addrspace(4)* %952 to float addrspace(1)*
  store float %950, float addrspace(1)* %953, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %943, %939
  br i1 %158, label %954, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
  br label %.preheader1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

954:                                              ; preds = %._crit_edge70.2.2
  %955 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !898
  br i1 %81, label %960, label %956, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

956:                                              ; preds = %954
  %957 = add i64 %.in, %290
  %958 = inttoptr i64 %957 to float addrspace(4)*
  %959 = addrspacecast float addrspace(4)* %958 to float addrspace(1)*
  store float %955, float addrspace(1)* %959, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

960:                                              ; preds = %954
  %961 = add i64 %.in988, %272
  %962 = add i64 %961, %284
  %963 = inttoptr i64 %962 to float addrspace(4)*
  %964 = addrspacecast float addrspace(4)* %963 to float addrspace(1)*
  %965 = load float, float addrspace(1)* %964, align 4
  %966 = fmul reassoc nsz arcp contract float %965, %4, !spirv.Decorations !898
  %967 = fadd reassoc nsz arcp contract float %955, %966, !spirv.Decorations !898
  %968 = add i64 %.in, %290
  %969 = inttoptr i64 %968 to float addrspace(4)*
  %970 = addrspacecast float addrspace(4)* %969 to float addrspace(1)*
  store float %967, float addrspace(1)* %970, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %960, %956
  br i1 %162, label %971, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
  br label %._crit_edge70.378, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

971:                                              ; preds = %.preheader1.2
  %972 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !898
  br i1 %81, label %977, label %973, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

973:                                              ; preds = %971
  %974 = add i64 %.in, %292
  %975 = inttoptr i64 %974 to float addrspace(4)*
  %976 = addrspacecast float addrspace(4)* %975 to float addrspace(1)*
  store float %972, float addrspace(1)* %976, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

977:                                              ; preds = %971
  %978 = add i64 %.in988, %226
  %979 = add i64 %978, %293
  %980 = inttoptr i64 %979 to float addrspace(4)*
  %981 = addrspacecast float addrspace(4)* %980 to float addrspace(1)*
  %982 = load float, float addrspace(1)* %981, align 4
  %983 = fmul reassoc nsz arcp contract float %982, %4, !spirv.Decorations !898
  %984 = fadd reassoc nsz arcp contract float %972, %983, !spirv.Decorations !898
  %985 = add i64 %.in, %292
  %986 = inttoptr i64 %985 to float addrspace(4)*
  %987 = addrspacecast float addrspace(4)* %986 to float addrspace(1)*
  store float %984, float addrspace(1)* %987, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %977, %973
  br i1 %165, label %988, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

988:                                              ; preds = %._crit_edge70.378
  %989 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !898
  br i1 %81, label %994, label %990, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

990:                                              ; preds = %988
  %991 = add i64 %.in, %295
  %992 = inttoptr i64 %991 to float addrspace(4)*
  %993 = addrspacecast float addrspace(4)* %992 to float addrspace(1)*
  store float %989, float addrspace(1)* %993, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

994:                                              ; preds = %988
  %995 = add i64 %.in988, %242
  %996 = add i64 %995, %293
  %997 = inttoptr i64 %996 to float addrspace(4)*
  %998 = addrspacecast float addrspace(4)* %997 to float addrspace(1)*
  %999 = load float, float addrspace(1)* %998, align 4
  %1000 = fmul reassoc nsz arcp contract float %999, %4, !spirv.Decorations !898
  %1001 = fadd reassoc nsz arcp contract float %989, %1000, !spirv.Decorations !898
  %1002 = add i64 %.in, %295
  %1003 = inttoptr i64 %1002 to float addrspace(4)*
  %1004 = addrspacecast float addrspace(4)* %1003 to float addrspace(1)*
  store float %1001, float addrspace(1)* %1004, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %994, %990
  br i1 %168, label %1005, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1005:                                             ; preds = %._crit_edge70.1.3
  %1006 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !898
  br i1 %81, label %1011, label %1007, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1007:                                             ; preds = %1005
  %1008 = add i64 %.in, %297
  %1009 = inttoptr i64 %1008 to float addrspace(4)*
  %1010 = addrspacecast float addrspace(4)* %1009 to float addrspace(1)*
  store float %1006, float addrspace(1)* %1010, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1011:                                             ; preds = %1005
  %1012 = add i64 %.in988, %257
  %1013 = add i64 %1012, %293
  %1014 = inttoptr i64 %1013 to float addrspace(4)*
  %1015 = addrspacecast float addrspace(4)* %1014 to float addrspace(1)*
  %1016 = load float, float addrspace(1)* %1015, align 4
  %1017 = fmul reassoc nsz arcp contract float %1016, %4, !spirv.Decorations !898
  %1018 = fadd reassoc nsz arcp contract float %1006, %1017, !spirv.Decorations !898
  %1019 = add i64 %.in, %297
  %1020 = inttoptr i64 %1019 to float addrspace(4)*
  %1021 = addrspacecast float addrspace(4)* %1020 to float addrspace(1)*
  store float %1018, float addrspace(1)* %1021, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %1011, %1007
  br i1 %171, label %1022, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
  br label %.preheader1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1022:                                             ; preds = %._crit_edge70.2.3
  %1023 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !898
  br i1 %81, label %1028, label %1024, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

1024:                                             ; preds = %1022
  %1025 = add i64 %.in, %299
  %1026 = inttoptr i64 %1025 to float addrspace(4)*
  %1027 = addrspacecast float addrspace(4)* %1026 to float addrspace(1)*
  store float %1023, float addrspace(1)* %1027, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

1028:                                             ; preds = %1022
  %1029 = add i64 %.in988, %272
  %1030 = add i64 %1029, %293
  %1031 = inttoptr i64 %1030 to float addrspace(4)*
  %1032 = addrspacecast float addrspace(4)* %1031 to float addrspace(1)*
  %1033 = load float, float addrspace(1)* %1032, align 4
  %1034 = fmul reassoc nsz arcp contract float %1033, %4, !spirv.Decorations !898
  %1035 = fadd reassoc nsz arcp contract float %1023, %1034, !spirv.Decorations !898
  %1036 = add i64 %.in, %299
  %1037 = inttoptr i64 %1036 to float addrspace(4)*
  %1038 = addrspacecast float addrspace(4)* %1037 to float addrspace(1)*
  store float %1035, float addrspace(1)* %1038, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %1028, %1024
  %1039 = add i32 %311, %52
  %1040 = icmp slt i32 %1039, %8
  br i1 %1040, label %.preheader1.3..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge72.loopexit:                           ; preds = %.preheader1.3
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.3..preheader2.preheader_crit_edge:    ; preds = %.preheader1.3
  %1041 = add i64 %.in990, %300
  %1042 = add i64 %.in989, %301
  %1043 = add i64 %.in988, %309
  %1044 = add i64 %.in, %310
  br label %.preheader2.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE(%"struct.cutlass::gemm::GemmCoord"* byval(%"struct.cutlass::gemm::GemmCoord") align 4 %0, float %1, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %2, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %3, float %4, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %5, %"class.cutlass::__generated_TensorRef"* byval(%"class.cutlass::__generated_TensorRef") align 8 %6, float %7, i32 %8, i64 %9, i64 %10, i64 %11, i64 %12, <8 x i32> %r0, <3 x i32> %globalOffset, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8 addrspace(2)* %constBase, i8* %privateBase, i32 %const_reg_dword, i32 %const_reg_dword1, i32 %const_reg_dword2, i64 %const_reg_qword, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i32 %bindlessOffset) #0 {
  %14 = bitcast i64 %9 to <2 x i32>
  %15 = extractelement <2 x i32> %14, i32 0
  %16 = extractelement <2 x i32> %14, i32 1
  %17 = bitcast i64 %10 to <2 x i32>
  %18 = extractelement <2 x i32> %17, i32 0
  %19 = extractelement <2 x i32> %17, i32 1
  %20 = bitcast i64 %11 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i32 0
  %22 = extractelement <2 x i32> %20, i32 1
  %23 = bitcast i64 %12 to <2 x i32>
  %24 = extractelement <2 x i32> %23, i32 0
  %25 = extractelement <2 x i32> %23, i32 1
  %26 = extractelement <8 x i32> %r0, i32 7
  %27 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %15, i32 %16)
  %28 = extractvalue { i32, i32 } %27, 0
  %29 = extractvalue { i32, i32 } %27, 1
  %30 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %18, i32 %19)
  %31 = extractvalue { i32, i32 } %30, 0
  %32 = extractvalue { i32, i32 } %30, 1
  %33 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %21, i32 %22)
  %34 = extractvalue { i32, i32 } %33, 0
  %35 = extractvalue { i32, i32 } %33, 1
  %36 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 0, i32 %24, i32 %25)
  %37 = extractvalue { i32, i32 } %36, 0
  %38 = extractvalue { i32, i32 } %36, 1
  %39 = icmp slt i32 %26, %8
  br i1 %39, label %.lr.ph, label %.._crit_edge72_crit_edge, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879

.._crit_edge72_crit_edge:                         ; preds = %13
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.lr.ph:                                           ; preds = %13
  %40 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %41 = extractelement <2 x i32> %40, i32 0
  %42 = extractelement <2 x i32> %40, i32 1
  %43 = bitcast i64 %const_reg_qword7 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i32 0
  %45 = extractelement <2 x i32> %43, i32 1
  %46 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %47 = extractelement <2 x i32> %46, i32 0
  %48 = extractelement <2 x i32> %46, i32 1
  %49 = extractelement <3 x i32> %numWorkGroups, i32 2
  %50 = extractelement <3 x i32> %localSize, i32 0
  %51 = extractelement <3 x i32> %localSize, i32 1
  %52 = extractelement <8 x i32> %r0, i32 1
  %53 = extractelement <8 x i32> %r0, i32 6
  %54 = mul i32 %52, %50
  %55 = zext i16 %localIdX to i32
  %56 = add i32 %54, %55
  %57 = shl i32 %56, 2
  %58 = mul i32 %53, %51
  %59 = zext i16 %localIdY to i32
  %60 = add i32 %58, %59
  %61 = shl i32 %60, 4
  %62 = insertelement <2 x i32> undef, i32 %28, i32 0
  %63 = insertelement <2 x i32> %62, i32 %29, i32 1
  %64 = bitcast <2 x i32> %63 to i64
  %65 = shl i64 %64, 1
  %66 = add i64 %65, %const_reg_qword
  %67 = insertelement <2 x i32> undef, i32 %31, i32 0
  %68 = insertelement <2 x i32> %67, i32 %32, i32 1
  %69 = bitcast <2 x i32> %68 to i64
  %70 = shl i64 %69, 1
  %71 = add i64 %70, %const_reg_qword4
  %72 = insertelement <2 x i32> undef, i32 %34, i32 0
  %73 = insertelement <2 x i32> %72, i32 %35, i32 1
  %74 = bitcast <2 x i32> %73 to i64
  %.op = shl i64 %74, 2
  %75 = bitcast i64 %.op to <2 x i32>
  %76 = extractelement <2 x i32> %75, i32 0
  %77 = extractelement <2 x i32> %75, i32 1
  %78 = fcmp reassoc nsz arcp contract une float %4, 0.000000e+00, !spirv.Decorations !898
  %79 = select i1 %78, i32 %76, i32 0
  %80 = select i1 %78, i32 %77, i32 0
  %81 = insertelement <2 x i32> undef, i32 %79, i32 0
  %82 = insertelement <2 x i32> %81, i32 %80, i32 1
  %83 = bitcast <2 x i32> %82 to i64
  %84 = add i64 %83, %const_reg_qword6
  %85 = insertelement <2 x i32> undef, i32 %37, i32 0
  %86 = insertelement <2 x i32> %85, i32 %38, i32 1
  %87 = bitcast <2 x i32> %86 to i64
  %88 = shl i64 %87, 2
  %89 = add i64 %88, %const_reg_qword8
  %90 = icmp sgt i32 %const_reg_dword2, 0
  %91 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %15, i32 %16)
  %92 = extractvalue { i32, i32 } %91, 0
  %93 = extractvalue { i32, i32 } %91, 1
  %94 = insertelement <2 x i32> undef, i32 %92, i32 0
  %95 = insertelement <2 x i32> %94, i32 %93, i32 1
  %96 = bitcast <2 x i32> %95 to i64
  %97 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %18, i32 %19)
  %98 = extractvalue { i32, i32 } %97, 0
  %99 = extractvalue { i32, i32 } %97, 1
  %100 = insertelement <2 x i32> undef, i32 %98, i32 0
  %101 = insertelement <2 x i32> %100, i32 %99, i32 1
  %102 = bitcast <2 x i32> %101 to i64
  %103 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %21, i32 %22)
  %104 = extractvalue { i32, i32 } %103, 0
  %105 = extractvalue { i32, i32 } %103, 1
  %106 = insertelement <2 x i32> undef, i32 %104, i32 0
  %107 = insertelement <2 x i32> %106, i32 %105, i32 1
  %108 = bitcast <2 x i32> %107 to i64
  %109 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %49, i32 0, i32 %24, i32 %25)
  %110 = extractvalue { i32, i32 } %109, 0
  %111 = extractvalue { i32, i32 } %109, 1
  %112 = insertelement <2 x i32> undef, i32 %110, i32 0
  %113 = insertelement <2 x i32> %112, i32 %111, i32 1
  %114 = bitcast <2 x i32> %113 to i64
  %115 = icmp slt i32 %57, %const_reg_dword
  %116 = icmp slt i32 %61, %const_reg_dword1
  %117 = and i1 %115, %116
  %118 = add i32 %57, 1
  %119 = icmp slt i32 %118, %const_reg_dword
  %120 = icmp slt i32 %61, %const_reg_dword1
  %121 = and i1 %119, %120
  %122 = add i32 %57, 2
  %123 = icmp slt i32 %122, %const_reg_dword
  %124 = icmp slt i32 %61, %const_reg_dword1
  %125 = and i1 %123, %124
  %126 = add i32 %57, 3
  %127 = icmp slt i32 %126, %const_reg_dword
  %128 = icmp slt i32 %61, %const_reg_dword1
  %129 = and i1 %127, %128
  %130 = add i32 %61, 1
  %131 = icmp slt i32 %130, %const_reg_dword1
  %132 = icmp slt i32 %57, %const_reg_dword
  %133 = and i1 %132, %131
  %134 = icmp slt i32 %118, %const_reg_dword
  %135 = icmp slt i32 %130, %const_reg_dword1
  %136 = and i1 %134, %135
  %137 = icmp slt i32 %122, %const_reg_dword
  %138 = icmp slt i32 %130, %const_reg_dword1
  %139 = and i1 %137, %138
  %140 = icmp slt i32 %126, %const_reg_dword
  %141 = icmp slt i32 %130, %const_reg_dword1
  %142 = and i1 %140, %141
  %143 = add i32 %61, 2
  %144 = icmp slt i32 %143, %const_reg_dword1
  %145 = icmp slt i32 %57, %const_reg_dword
  %146 = and i1 %145, %144
  %147 = icmp slt i32 %118, %const_reg_dword
  %148 = icmp slt i32 %143, %const_reg_dword1
  %149 = and i1 %147, %148
  %150 = icmp slt i32 %122, %const_reg_dword
  %151 = icmp slt i32 %143, %const_reg_dword1
  %152 = and i1 %150, %151
  %153 = icmp slt i32 %126, %const_reg_dword
  %154 = icmp slt i32 %143, %const_reg_dword1
  %155 = and i1 %153, %154
  %156 = add i32 %61, 3
  %157 = icmp slt i32 %156, %const_reg_dword1
  %158 = icmp slt i32 %57, %const_reg_dword
  %159 = and i1 %158, %157
  %160 = icmp slt i32 %118, %const_reg_dword
  %161 = icmp slt i32 %156, %const_reg_dword1
  %162 = and i1 %160, %161
  %163 = icmp slt i32 %122, %const_reg_dword
  %164 = icmp slt i32 %156, %const_reg_dword1
  %165 = and i1 %163, %164
  %166 = icmp slt i32 %126, %const_reg_dword
  %167 = icmp slt i32 %156, %const_reg_dword1
  %168 = and i1 %166, %167
  %169 = add i32 %61, 4
  %170 = icmp slt i32 %169, %const_reg_dword1
  %171 = icmp slt i32 %57, %const_reg_dword
  %172 = and i1 %171, %170
  %173 = icmp slt i32 %118, %const_reg_dword
  %174 = icmp slt i32 %169, %const_reg_dword1
  %175 = and i1 %173, %174
  %176 = icmp slt i32 %122, %const_reg_dword
  %177 = icmp slt i32 %169, %const_reg_dword1
  %178 = and i1 %176, %177
  %179 = icmp slt i32 %126, %const_reg_dword
  %180 = icmp slt i32 %169, %const_reg_dword1
  %181 = and i1 %179, %180
  %182 = add i32 %61, 5
  %183 = icmp slt i32 %182, %const_reg_dword1
  %184 = icmp slt i32 %57, %const_reg_dword
  %185 = and i1 %184, %183
  %186 = icmp slt i32 %118, %const_reg_dword
  %187 = icmp slt i32 %182, %const_reg_dword1
  %188 = and i1 %186, %187
  %189 = icmp slt i32 %122, %const_reg_dword
  %190 = icmp slt i32 %182, %const_reg_dword1
  %191 = and i1 %189, %190
  %192 = icmp slt i32 %126, %const_reg_dword
  %193 = icmp slt i32 %182, %const_reg_dword1
  %194 = and i1 %192, %193
  %195 = add i32 %61, 6
  %196 = icmp slt i32 %195, %const_reg_dword1
  %197 = icmp slt i32 %57, %const_reg_dword
  %198 = and i1 %197, %196
  %199 = icmp slt i32 %118, %const_reg_dword
  %200 = icmp slt i32 %195, %const_reg_dword1
  %201 = and i1 %199, %200
  %202 = icmp slt i32 %122, %const_reg_dword
  %203 = icmp slt i32 %195, %const_reg_dword1
  %204 = and i1 %202, %203
  %205 = icmp slt i32 %126, %const_reg_dword
  %206 = icmp slt i32 %195, %const_reg_dword1
  %207 = and i1 %205, %206
  %208 = add i32 %61, 7
  %209 = icmp slt i32 %208, %const_reg_dword1
  %210 = icmp slt i32 %57, %const_reg_dword
  %211 = and i1 %210, %209
  %212 = icmp slt i32 %118, %const_reg_dword
  %213 = icmp slt i32 %208, %const_reg_dword1
  %214 = and i1 %212, %213
  %215 = icmp slt i32 %122, %const_reg_dword
  %216 = icmp slt i32 %208, %const_reg_dword1
  %217 = and i1 %215, %216
  %218 = icmp slt i32 %126, %const_reg_dword
  %219 = icmp slt i32 %208, %const_reg_dword1
  %220 = and i1 %218, %219
  %221 = add i32 %61, 8
  %222 = icmp slt i32 %221, %const_reg_dword1
  %223 = icmp slt i32 %57, %const_reg_dword
  %224 = and i1 %223, %222
  %225 = icmp slt i32 %118, %const_reg_dword
  %226 = icmp slt i32 %221, %const_reg_dword1
  %227 = and i1 %225, %226
  %228 = icmp slt i32 %122, %const_reg_dword
  %229 = icmp slt i32 %221, %const_reg_dword1
  %230 = and i1 %228, %229
  %231 = icmp slt i32 %126, %const_reg_dword
  %232 = icmp slt i32 %221, %const_reg_dword1
  %233 = and i1 %231, %232
  %234 = add i32 %61, 9
  %235 = icmp slt i32 %234, %const_reg_dword1
  %236 = icmp slt i32 %57, %const_reg_dword
  %237 = and i1 %236, %235
  %238 = icmp slt i32 %118, %const_reg_dword
  %239 = icmp slt i32 %234, %const_reg_dword1
  %240 = and i1 %238, %239
  %241 = icmp slt i32 %122, %const_reg_dword
  %242 = icmp slt i32 %234, %const_reg_dword1
  %243 = and i1 %241, %242
  %244 = icmp slt i32 %126, %const_reg_dword
  %245 = icmp slt i32 %234, %const_reg_dword1
  %246 = and i1 %244, %245
  %247 = add i32 %61, 10
  %248 = icmp slt i32 %247, %const_reg_dword1
  %249 = icmp slt i32 %57, %const_reg_dword
  %250 = and i1 %249, %248
  %251 = icmp slt i32 %118, %const_reg_dword
  %252 = icmp slt i32 %247, %const_reg_dword1
  %253 = and i1 %251, %252
  %254 = icmp slt i32 %122, %const_reg_dword
  %255 = icmp slt i32 %247, %const_reg_dword1
  %256 = and i1 %254, %255
  %257 = icmp slt i32 %126, %const_reg_dword
  %258 = icmp slt i32 %247, %const_reg_dword1
  %259 = and i1 %257, %258
  %260 = add i32 %61, 11
  %261 = icmp slt i32 %260, %const_reg_dword1
  %262 = icmp slt i32 %57, %const_reg_dword
  %263 = and i1 %262, %261
  %264 = icmp slt i32 %118, %const_reg_dword
  %265 = icmp slt i32 %260, %const_reg_dword1
  %266 = and i1 %264, %265
  %267 = icmp slt i32 %122, %const_reg_dword
  %268 = icmp slt i32 %260, %const_reg_dword1
  %269 = and i1 %267, %268
  %270 = icmp slt i32 %126, %const_reg_dword
  %271 = icmp slt i32 %260, %const_reg_dword1
  %272 = and i1 %270, %271
  %273 = add i32 %61, 12
  %274 = icmp slt i32 %273, %const_reg_dword1
  %275 = icmp slt i32 %57, %const_reg_dword
  %276 = and i1 %275, %274
  %277 = icmp slt i32 %118, %const_reg_dword
  %278 = icmp slt i32 %273, %const_reg_dword1
  %279 = and i1 %277, %278
  %280 = icmp slt i32 %122, %const_reg_dword
  %281 = icmp slt i32 %273, %const_reg_dword1
  %282 = and i1 %280, %281
  %283 = icmp slt i32 %126, %const_reg_dword
  %284 = icmp slt i32 %273, %const_reg_dword1
  %285 = and i1 %283, %284
  %286 = add i32 %61, 13
  %287 = icmp slt i32 %286, %const_reg_dword1
  %288 = icmp slt i32 %57, %const_reg_dword
  %289 = and i1 %288, %287
  %290 = icmp slt i32 %118, %const_reg_dword
  %291 = icmp slt i32 %286, %const_reg_dword1
  %292 = and i1 %290, %291
  %293 = icmp slt i32 %122, %const_reg_dword
  %294 = icmp slt i32 %286, %const_reg_dword1
  %295 = and i1 %293, %294
  %296 = icmp slt i32 %126, %const_reg_dword
  %297 = icmp slt i32 %286, %const_reg_dword1
  %298 = and i1 %296, %297
  %299 = add i32 %61, 14
  %300 = icmp slt i32 %299, %const_reg_dword1
  %301 = icmp slt i32 %57, %const_reg_dword
  %302 = and i1 %301, %300
  %303 = icmp slt i32 %118, %const_reg_dword
  %304 = icmp slt i32 %299, %const_reg_dword1
  %305 = and i1 %303, %304
  %306 = icmp slt i32 %122, %const_reg_dword
  %307 = icmp slt i32 %299, %const_reg_dword1
  %308 = and i1 %306, %307
  %309 = icmp slt i32 %126, %const_reg_dword
  %310 = icmp slt i32 %299, %const_reg_dword1
  %311 = and i1 %309, %310
  %312 = add i32 %61, 15
  %313 = icmp slt i32 %312, %const_reg_dword1
  %314 = icmp slt i32 %57, %const_reg_dword
  %315 = and i1 %314, %313
  %316 = icmp slt i32 %118, %const_reg_dword
  %317 = icmp slt i32 %312, %const_reg_dword1
  %318 = and i1 %316, %317
  %319 = icmp slt i32 %122, %const_reg_dword
  %320 = icmp slt i32 %312, %const_reg_dword1
  %321 = and i1 %319, %320
  %322 = icmp slt i32 %126, %const_reg_dword
  %323 = icmp slt i32 %312, %const_reg_dword1
  %324 = and i1 %322, %323
  %325 = ashr i32 %57, 31
  %326 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %41, i32 %42)
  %327 = extractvalue { i32, i32 } %326, 0
  %328 = extractvalue { i32, i32 } %326, 1
  %329 = insertelement <2 x i32> undef, i32 %327, i32 0
  %330 = insertelement <2 x i32> %329, i32 %328, i32 1
  %331 = sext i32 %61 to i64
  %332 = ashr i32 %118, 31
  %333 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %41, i32 %42)
  %334 = extractvalue { i32, i32 } %333, 0
  %335 = extractvalue { i32, i32 } %333, 1
  %336 = insertelement <2 x i32> undef, i32 %334, i32 0
  %337 = insertelement <2 x i32> %336, i32 %335, i32 1
  %338 = ashr i32 %122, 31
  %339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %41, i32 %42)
  %340 = extractvalue { i32, i32 } %339, 0
  %341 = extractvalue { i32, i32 } %339, 1
  %342 = insertelement <2 x i32> undef, i32 %340, i32 0
  %343 = insertelement <2 x i32> %342, i32 %341, i32 1
  %344 = ashr i32 %126, 31
  %345 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %41, i32 %42)
  %346 = extractvalue { i32, i32 } %345, 0
  %347 = extractvalue { i32, i32 } %345, 1
  %348 = insertelement <2 x i32> undef, i32 %346, i32 0
  %349 = insertelement <2 x i32> %348, i32 %347, i32 1
  %350 = sext i32 %130 to i64
  %351 = sext i32 %143 to i64
  %352 = sext i32 %156 to i64
  %353 = sext i32 %169 to i64
  %354 = sext i32 %182 to i64
  %355 = sext i32 %195 to i64
  %356 = sext i32 %208 to i64
  %357 = sext i32 %221 to i64
  %358 = sext i32 %234 to i64
  %359 = sext i32 %247 to i64
  %360 = sext i32 %260 to i64
  %361 = sext i32 %273 to i64
  %362 = sext i32 %286 to i64
  %363 = sext i32 %299 to i64
  %364 = sext i32 %312 to i64
  %365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %47, i32 %48)
  %366 = extractvalue { i32, i32 } %365, 0
  %367 = extractvalue { i32, i32 } %365, 1
  %368 = insertelement <2 x i32> undef, i32 %366, i32 0
  %369 = insertelement <2 x i32> %368, i32 %367, i32 1
  %370 = bitcast <2 x i32> %369 to i64
  %371 = add nsw i64 %370, %331
  %372 = shl i64 %371, 2
  %373 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %57, i32 %325, i32 %44, i32 %45)
  %374 = extractvalue { i32, i32 } %373, 0
  %375 = extractvalue { i32, i32 } %373, 1
  %376 = insertelement <2 x i32> undef, i32 %374, i32 0
  %377 = insertelement <2 x i32> %376, i32 %375, i32 1
  %378 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %47, i32 %48)
  %379 = extractvalue { i32, i32 } %378, 0
  %380 = extractvalue { i32, i32 } %378, 1
  %381 = insertelement <2 x i32> undef, i32 %379, i32 0
  %382 = insertelement <2 x i32> %381, i32 %380, i32 1
  %383 = bitcast <2 x i32> %382 to i64
  %384 = add nsw i64 %383, %331
  %385 = shl i64 %384, 2
  %386 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %118, i32 %332, i32 %44, i32 %45)
  %387 = extractvalue { i32, i32 } %386, 0
  %388 = extractvalue { i32, i32 } %386, 1
  %389 = insertelement <2 x i32> undef, i32 %387, i32 0
  %390 = insertelement <2 x i32> %389, i32 %388, i32 1
  %391 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %47, i32 %48)
  %392 = extractvalue { i32, i32 } %391, 0
  %393 = extractvalue { i32, i32 } %391, 1
  %394 = insertelement <2 x i32> undef, i32 %392, i32 0
  %395 = insertelement <2 x i32> %394, i32 %393, i32 1
  %396 = bitcast <2 x i32> %395 to i64
  %397 = add nsw i64 %396, %331
  %398 = shl i64 %397, 2
  %399 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %122, i32 %338, i32 %44, i32 %45)
  %400 = extractvalue { i32, i32 } %399, 0
  %401 = extractvalue { i32, i32 } %399, 1
  %402 = insertelement <2 x i32> undef, i32 %400, i32 0
  %403 = insertelement <2 x i32> %402, i32 %401, i32 1
  %404 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %47, i32 %48)
  %405 = extractvalue { i32, i32 } %404, 0
  %406 = extractvalue { i32, i32 } %404, 1
  %407 = insertelement <2 x i32> undef, i32 %405, i32 0
  %408 = insertelement <2 x i32> %407, i32 %406, i32 1
  %409 = bitcast <2 x i32> %408 to i64
  %410 = add nsw i64 %409, %331
  %411 = shl i64 %410, 2
  %412 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %126, i32 %344, i32 %44, i32 %45)
  %413 = extractvalue { i32, i32 } %412, 0
  %414 = extractvalue { i32, i32 } %412, 1
  %415 = insertelement <2 x i32> undef, i32 %413, i32 0
  %416 = insertelement <2 x i32> %415, i32 %414, i32 1
  %417 = add nsw i64 %370, %350
  %418 = shl i64 %417, 2
  %419 = add nsw i64 %383, %350
  %420 = shl i64 %419, 2
  %421 = add nsw i64 %396, %350
  %422 = shl i64 %421, 2
  %423 = add nsw i64 %409, %350
  %424 = shl i64 %423, 2
  %425 = add nsw i64 %370, %351
  %426 = shl i64 %425, 2
  %427 = add nsw i64 %383, %351
  %428 = shl i64 %427, 2
  %429 = add nsw i64 %396, %351
  %430 = shl i64 %429, 2
  %431 = add nsw i64 %409, %351
  %432 = shl i64 %431, 2
  %433 = add nsw i64 %370, %352
  %434 = shl i64 %433, 2
  %435 = add nsw i64 %383, %352
  %436 = shl i64 %435, 2
  %437 = add nsw i64 %396, %352
  %438 = shl i64 %437, 2
  %439 = add nsw i64 %409, %352
  %440 = shl i64 %439, 2
  %441 = add nsw i64 %370, %353
  %442 = shl i64 %441, 2
  %443 = add nsw i64 %383, %353
  %444 = shl i64 %443, 2
  %445 = add nsw i64 %396, %353
  %446 = shl i64 %445, 2
  %447 = add nsw i64 %409, %353
  %448 = shl i64 %447, 2
  %449 = add nsw i64 %370, %354
  %450 = shl i64 %449, 2
  %451 = add nsw i64 %383, %354
  %452 = shl i64 %451, 2
  %453 = add nsw i64 %396, %354
  %454 = shl i64 %453, 2
  %455 = add nsw i64 %409, %354
  %456 = shl i64 %455, 2
  %457 = add nsw i64 %370, %355
  %458 = shl i64 %457, 2
  %459 = add nsw i64 %383, %355
  %460 = shl i64 %459, 2
  %461 = add nsw i64 %396, %355
  %462 = shl i64 %461, 2
  %463 = add nsw i64 %409, %355
  %464 = shl i64 %463, 2
  %465 = add nsw i64 %370, %356
  %466 = shl i64 %465, 2
  %467 = add nsw i64 %383, %356
  %468 = shl i64 %467, 2
  %469 = add nsw i64 %396, %356
  %470 = shl i64 %469, 2
  %471 = add nsw i64 %409, %356
  %472 = shl i64 %471, 2
  %473 = add nsw i64 %370, %357
  %474 = shl i64 %473, 2
  %475 = add nsw i64 %383, %357
  %476 = shl i64 %475, 2
  %477 = add nsw i64 %396, %357
  %478 = shl i64 %477, 2
  %479 = add nsw i64 %409, %357
  %480 = shl i64 %479, 2
  %481 = add nsw i64 %370, %358
  %482 = shl i64 %481, 2
  %483 = add nsw i64 %383, %358
  %484 = shl i64 %483, 2
  %485 = add nsw i64 %396, %358
  %486 = shl i64 %485, 2
  %487 = add nsw i64 %409, %358
  %488 = shl i64 %487, 2
  %489 = add nsw i64 %370, %359
  %490 = shl i64 %489, 2
  %491 = add nsw i64 %383, %359
  %492 = shl i64 %491, 2
  %493 = add nsw i64 %396, %359
  %494 = shl i64 %493, 2
  %495 = add nsw i64 %409, %359
  %496 = shl i64 %495, 2
  %497 = add nsw i64 %370, %360
  %498 = shl i64 %497, 2
  %499 = add nsw i64 %383, %360
  %500 = shl i64 %499, 2
  %501 = add nsw i64 %396, %360
  %502 = shl i64 %501, 2
  %503 = add nsw i64 %409, %360
  %504 = shl i64 %503, 2
  %505 = add nsw i64 %370, %361
  %506 = shl i64 %505, 2
  %507 = add nsw i64 %383, %361
  %508 = shl i64 %507, 2
  %509 = add nsw i64 %396, %361
  %510 = shl i64 %509, 2
  %511 = add nsw i64 %409, %361
  %512 = shl i64 %511, 2
  %513 = add nsw i64 %370, %362
  %514 = shl i64 %513, 2
  %515 = add nsw i64 %383, %362
  %516 = shl i64 %515, 2
  %517 = add nsw i64 %396, %362
  %518 = shl i64 %517, 2
  %519 = add nsw i64 %409, %362
  %520 = shl i64 %519, 2
  %521 = add nsw i64 %370, %363
  %522 = shl i64 %521, 2
  %523 = add nsw i64 %383, %363
  %524 = shl i64 %523, 2
  %525 = add nsw i64 %396, %363
  %526 = shl i64 %525, 2
  %527 = add nsw i64 %409, %363
  %528 = shl i64 %527, 2
  %529 = add nsw i64 %370, %364
  %530 = shl i64 %529, 2
  %531 = add nsw i64 %383, %364
  %532 = shl i64 %531, 2
  %533 = add nsw i64 %396, %364
  %534 = shl i64 %533, 2
  %535 = add nsw i64 %409, %364
  %536 = shl i64 %535, 2
  %537 = shl i64 %96, 1
  %538 = shl i64 %102, 1
  %.op3824 = shl i64 %108, 2
  %539 = bitcast i64 %.op3824 to <2 x i32>
  %540 = extractelement <2 x i32> %539, i32 0
  %541 = extractelement <2 x i32> %539, i32 1
  %542 = select i1 %78, i32 %540, i32 0
  %543 = select i1 %78, i32 %541, i32 0
  %544 = insertelement <2 x i32> undef, i32 %542, i32 0
  %545 = insertelement <2 x i32> %544, i32 %543, i32 1
  %546 = shl i64 %114, 2
  br label %.preheader2.preheader, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader2.preheader:                            ; preds = %.preheader1.15..preheader2.preheader_crit_edge, %.lr.ph
  %547 = phi i32 [ %26, %.lr.ph ], [ %3307, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in = phi i64 [ %89, %.lr.ph ], [ %3312, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3821 = phi i64 [ %84, %.lr.ph ], [ %3311, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3822 = phi i64 [ %71, %.lr.ph ], [ %3310, %.preheader1.15..preheader2.preheader_crit_edge ]
  %.in3823 = phi i64 [ %66, %.lr.ph ], [ %3309, %.preheader1.15..preheader2.preheader_crit_edge ]
  br i1 %90, label %.preheader.preheader.preheader, label %.preheader2.preheader..preheader1.preheader_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader2.preheader..preheader1.preheader_crit_edge: ; preds = %.preheader2.preheader
  br label %.preheader1.preheader, !stats.blockFrequency.digits !917, !stats.blockFrequency.scale !879

.preheader.preheader.preheader:                   ; preds = %.preheader2.preheader
  %sink_3874 = bitcast <2 x i32> %330 to i64
  %sink_3866 = shl i64 %sink_3874, 1
  %548 = add i64 %.in3823, %sink_3866
  %sink_3873 = bitcast <2 x i32> %337 to i64
  %sink_3865 = shl i64 %sink_3873, 1
  %549 = add i64 %.in3823, %sink_3865
  %sink_3872 = bitcast <2 x i32> %343 to i64
  %sink_3864 = shl i64 %sink_3872, 1
  %550 = add i64 %.in3823, %sink_3864
  %sink_3871 = bitcast <2 x i32> %349 to i64
  %sink_3863 = shl i64 %sink_3871, 1
  %551 = add i64 %.in3823, %sink_3863
  br label %.preheader.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader.preheader:                             ; preds = %.preheader.15..preheader.preheader_crit_edge, %.preheader.preheader.preheader
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
  %sink_3861 = shl nsw i64 %331, 1
  %sink_3875 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %sink_3826 = extractelement <2 x i32> %sink_3875, i32 0
  %sink_3825 = extractelement <2 x i32> %sink_3875, i32 1
  br i1 %117, label %553, label %.preheader.preheader.._crit_edge_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.preheader.._crit_edge_crit_edge:       ; preds = %.preheader.preheader
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

553:                                              ; preds = %.preheader.preheader
  %.sroa.256.0.insert.ext = zext i32 %552 to i64
  %554 = shl nuw nsw i64 %.sroa.256.0.insert.ext, 1
  %555 = add i64 %548, %554
  %556 = inttoptr i64 %555 to i16 addrspace(4)*
  %557 = addrspacecast i16 addrspace(4)* %556 to i16 addrspace(1)*
  %558 = load i16, i16 addrspace(1)* %557, align 2
  %559 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %560 = extractvalue { i32, i32 } %559, 0
  %561 = extractvalue { i32, i32 } %559, 1
  %562 = insertelement <2 x i32> undef, i32 %560, i32 0
  %563 = insertelement <2 x i32> %562, i32 %561, i32 1
  %564 = bitcast <2 x i32> %563 to i64
  %565 = shl i64 %564, 1
  %566 = add i64 %.in3822, %565
  %567 = add i64 %566, %sink_3861
  %568 = inttoptr i64 %567 to i16 addrspace(4)*
  %569 = addrspacecast i16 addrspace(4)* %568 to i16 addrspace(1)*
  %570 = load i16, i16 addrspace(1)* %569, align 2
  %571 = zext i16 %558 to i32
  %572 = shl nuw i32 %571, 16, !spirv.Decorations !921
  %573 = bitcast i32 %572 to float
  %574 = zext i16 %570 to i32
  %575 = shl nuw i32 %574, 16, !spirv.Decorations !921
  %576 = bitcast i32 %575 to float
  %577 = fmul reassoc nsz arcp contract float %573, %576, !spirv.Decorations !898
  %578 = fadd reassoc nsz arcp contract float %577, %.sroa.0.1, !spirv.Decorations !898
  br label %._crit_edge, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge:                                      ; preds = %.preheader.preheader.._crit_edge_crit_edge, %553
  %.sroa.0.2 = phi float [ %578, %553 ], [ %.sroa.0.1, %.preheader.preheader.._crit_edge_crit_edge ]
  br i1 %121, label %579, label %._crit_edge.._crit_edge.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.._crit_edge.1_crit_edge:              ; preds = %._crit_edge
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

579:                                              ; preds = %._crit_edge
  %.sroa.256.0.insert.ext588 = zext i32 %552 to i64
  %580 = shl nuw nsw i64 %.sroa.256.0.insert.ext588, 1
  %581 = add i64 %549, %580
  %582 = inttoptr i64 %581 to i16 addrspace(4)*
  %583 = addrspacecast i16 addrspace(4)* %582 to i16 addrspace(1)*
  %584 = load i16, i16 addrspace(1)* %583, align 2
  %585 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %586 = extractvalue { i32, i32 } %585, 0
  %587 = extractvalue { i32, i32 } %585, 1
  %588 = insertelement <2 x i32> undef, i32 %586, i32 0
  %589 = insertelement <2 x i32> %588, i32 %587, i32 1
  %590 = bitcast <2 x i32> %589 to i64
  %591 = shl i64 %590, 1
  %592 = add i64 %.in3822, %591
  %593 = add i64 %592, %sink_3861
  %594 = inttoptr i64 %593 to i16 addrspace(4)*
  %595 = addrspacecast i16 addrspace(4)* %594 to i16 addrspace(1)*
  %596 = load i16, i16 addrspace(1)* %595, align 2
  %597 = zext i16 %584 to i32
  %598 = shl nuw i32 %597, 16, !spirv.Decorations !921
  %599 = bitcast i32 %598 to float
  %600 = zext i16 %596 to i32
  %601 = shl nuw i32 %600, 16, !spirv.Decorations !921
  %602 = bitcast i32 %601 to float
  %603 = fmul reassoc nsz arcp contract float %599, %602, !spirv.Decorations !898
  %604 = fadd reassoc nsz arcp contract float %603, %.sroa.66.1, !spirv.Decorations !898
  br label %._crit_edge.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1:                                    ; preds = %._crit_edge.._crit_edge.1_crit_edge, %579
  %.sroa.66.2 = phi float [ %604, %579 ], [ %.sroa.66.1, %._crit_edge.._crit_edge.1_crit_edge ]
  br i1 %125, label %605, label %._crit_edge.1.._crit_edge.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.._crit_edge.2_crit_edge:            ; preds = %._crit_edge.1
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

605:                                              ; preds = %._crit_edge.1
  %.sroa.256.0.insert.ext593 = zext i32 %552 to i64
  %606 = shl nuw nsw i64 %.sroa.256.0.insert.ext593, 1
  %607 = add i64 %550, %606
  %608 = inttoptr i64 %607 to i16 addrspace(4)*
  %609 = addrspacecast i16 addrspace(4)* %608 to i16 addrspace(1)*
  %610 = load i16, i16 addrspace(1)* %609, align 2
  %611 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %612 = extractvalue { i32, i32 } %611, 0
  %613 = extractvalue { i32, i32 } %611, 1
  %614 = insertelement <2 x i32> undef, i32 %612, i32 0
  %615 = insertelement <2 x i32> %614, i32 %613, i32 1
  %616 = bitcast <2 x i32> %615 to i64
  %617 = shl i64 %616, 1
  %618 = add i64 %.in3822, %617
  %619 = add i64 %618, %sink_3861
  %620 = inttoptr i64 %619 to i16 addrspace(4)*
  %621 = addrspacecast i16 addrspace(4)* %620 to i16 addrspace(1)*
  %622 = load i16, i16 addrspace(1)* %621, align 2
  %623 = zext i16 %610 to i32
  %624 = shl nuw i32 %623, 16, !spirv.Decorations !921
  %625 = bitcast i32 %624 to float
  %626 = zext i16 %622 to i32
  %627 = shl nuw i32 %626, 16, !spirv.Decorations !921
  %628 = bitcast i32 %627 to float
  %629 = fmul reassoc nsz arcp contract float %625, %628, !spirv.Decorations !898
  %630 = fadd reassoc nsz arcp contract float %629, %.sroa.130.1, !spirv.Decorations !898
  br label %._crit_edge.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2:                                    ; preds = %._crit_edge.1.._crit_edge.2_crit_edge, %605
  %.sroa.130.2 = phi float [ %630, %605 ], [ %.sroa.130.1, %._crit_edge.1.._crit_edge.2_crit_edge ]
  br i1 %129, label %631, label %._crit_edge.2..preheader_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2..preheader_crit_edge:               ; preds = %._crit_edge.2
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

631:                                              ; preds = %._crit_edge.2
  %.sroa.256.0.insert.ext598 = zext i32 %552 to i64
  %632 = shl nuw nsw i64 %.sroa.256.0.insert.ext598, 1
  %633 = add i64 %551, %632
  %634 = inttoptr i64 %633 to i16 addrspace(4)*
  %635 = addrspacecast i16 addrspace(4)* %634 to i16 addrspace(1)*
  %636 = load i16, i16 addrspace(1)* %635, align 2
  %637 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %638 = extractvalue { i32, i32 } %637, 0
  %639 = extractvalue { i32, i32 } %637, 1
  %640 = insertelement <2 x i32> undef, i32 %638, i32 0
  %641 = insertelement <2 x i32> %640, i32 %639, i32 1
  %642 = bitcast <2 x i32> %641 to i64
  %643 = shl i64 %642, 1
  %644 = add i64 %.in3822, %643
  %645 = add i64 %644, %sink_3861
  %646 = inttoptr i64 %645 to i16 addrspace(4)*
  %647 = addrspacecast i16 addrspace(4)* %646 to i16 addrspace(1)*
  %648 = load i16, i16 addrspace(1)* %647, align 2
  %649 = zext i16 %636 to i32
  %650 = shl nuw i32 %649, 16, !spirv.Decorations !921
  %651 = bitcast i32 %650 to float
  %652 = zext i16 %648 to i32
  %653 = shl nuw i32 %652, 16, !spirv.Decorations !921
  %654 = bitcast i32 %653 to float
  %655 = fmul reassoc nsz arcp contract float %651, %654, !spirv.Decorations !898
  %656 = fadd reassoc nsz arcp contract float %655, %.sroa.194.1, !spirv.Decorations !898
  br label %.preheader, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader:                                       ; preds = %._crit_edge.2..preheader_crit_edge, %631
  %.sroa.194.2 = phi float [ %656, %631 ], [ %.sroa.194.1, %._crit_edge.2..preheader_crit_edge ]
  %sink_3856 = shl nsw i64 %350, 1
  br i1 %133, label %657, label %.preheader.._crit_edge.173_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.._crit_edge.173_crit_edge:             ; preds = %.preheader
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

657:                                              ; preds = %.preheader
  %.sroa.256.0.insert.ext603 = zext i32 %552 to i64
  %658 = shl nuw nsw i64 %.sroa.256.0.insert.ext603, 1
  %659 = add i64 %548, %658
  %660 = inttoptr i64 %659 to i16 addrspace(4)*
  %661 = addrspacecast i16 addrspace(4)* %660 to i16 addrspace(1)*
  %662 = load i16, i16 addrspace(1)* %661, align 2
  %663 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %664 = extractvalue { i32, i32 } %663, 0
  %665 = extractvalue { i32, i32 } %663, 1
  %666 = insertelement <2 x i32> undef, i32 %664, i32 0
  %667 = insertelement <2 x i32> %666, i32 %665, i32 1
  %668 = bitcast <2 x i32> %667 to i64
  %669 = shl i64 %668, 1
  %670 = add i64 %.in3822, %669
  %671 = add i64 %670, %sink_3856
  %672 = inttoptr i64 %671 to i16 addrspace(4)*
  %673 = addrspacecast i16 addrspace(4)* %672 to i16 addrspace(1)*
  %674 = load i16, i16 addrspace(1)* %673, align 2
  %675 = zext i16 %662 to i32
  %676 = shl nuw i32 %675, 16, !spirv.Decorations !921
  %677 = bitcast i32 %676 to float
  %678 = zext i16 %674 to i32
  %679 = shl nuw i32 %678, 16, !spirv.Decorations !921
  %680 = bitcast i32 %679 to float
  %681 = fmul reassoc nsz arcp contract float %677, %680, !spirv.Decorations !898
  %682 = fadd reassoc nsz arcp contract float %681, %.sroa.6.1, !spirv.Decorations !898
  br label %._crit_edge.173, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.173:                                  ; preds = %.preheader.._crit_edge.173_crit_edge, %657
  %.sroa.6.2 = phi float [ %682, %657 ], [ %.sroa.6.1, %.preheader.._crit_edge.173_crit_edge ]
  br i1 %136, label %683, label %._crit_edge.173.._crit_edge.1.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.173.._crit_edge.1.1_crit_edge:        ; preds = %._crit_edge.173
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

683:                                              ; preds = %._crit_edge.173
  %.sroa.256.0.insert.ext608 = zext i32 %552 to i64
  %684 = shl nuw nsw i64 %.sroa.256.0.insert.ext608, 1
  %685 = add i64 %549, %684
  %686 = inttoptr i64 %685 to i16 addrspace(4)*
  %687 = addrspacecast i16 addrspace(4)* %686 to i16 addrspace(1)*
  %688 = load i16, i16 addrspace(1)* %687, align 2
  %689 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %690 = extractvalue { i32, i32 } %689, 0
  %691 = extractvalue { i32, i32 } %689, 1
  %692 = insertelement <2 x i32> undef, i32 %690, i32 0
  %693 = insertelement <2 x i32> %692, i32 %691, i32 1
  %694 = bitcast <2 x i32> %693 to i64
  %695 = shl i64 %694, 1
  %696 = add i64 %.in3822, %695
  %697 = add i64 %696, %sink_3856
  %698 = inttoptr i64 %697 to i16 addrspace(4)*
  %699 = addrspacecast i16 addrspace(4)* %698 to i16 addrspace(1)*
  %700 = load i16, i16 addrspace(1)* %699, align 2
  %701 = zext i16 %688 to i32
  %702 = shl nuw i32 %701, 16, !spirv.Decorations !921
  %703 = bitcast i32 %702 to float
  %704 = zext i16 %700 to i32
  %705 = shl nuw i32 %704, 16, !spirv.Decorations !921
  %706 = bitcast i32 %705 to float
  %707 = fmul reassoc nsz arcp contract float %703, %706, !spirv.Decorations !898
  %708 = fadd reassoc nsz arcp contract float %707, %.sroa.70.1, !spirv.Decorations !898
  br label %._crit_edge.1.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.1:                                  ; preds = %._crit_edge.173.._crit_edge.1.1_crit_edge, %683
  %.sroa.70.2 = phi float [ %708, %683 ], [ %.sroa.70.1, %._crit_edge.173.._crit_edge.1.1_crit_edge ]
  br i1 %139, label %709, label %._crit_edge.1.1.._crit_edge.2.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.1.._crit_edge.2.1_crit_edge:        ; preds = %._crit_edge.1.1
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

709:                                              ; preds = %._crit_edge.1.1
  %.sroa.256.0.insert.ext613 = zext i32 %552 to i64
  %710 = shl nuw nsw i64 %.sroa.256.0.insert.ext613, 1
  %711 = add i64 %550, %710
  %712 = inttoptr i64 %711 to i16 addrspace(4)*
  %713 = addrspacecast i16 addrspace(4)* %712 to i16 addrspace(1)*
  %714 = load i16, i16 addrspace(1)* %713, align 2
  %715 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %716 = extractvalue { i32, i32 } %715, 0
  %717 = extractvalue { i32, i32 } %715, 1
  %718 = insertelement <2 x i32> undef, i32 %716, i32 0
  %719 = insertelement <2 x i32> %718, i32 %717, i32 1
  %720 = bitcast <2 x i32> %719 to i64
  %721 = shl i64 %720, 1
  %722 = add i64 %.in3822, %721
  %723 = add i64 %722, %sink_3856
  %724 = inttoptr i64 %723 to i16 addrspace(4)*
  %725 = addrspacecast i16 addrspace(4)* %724 to i16 addrspace(1)*
  %726 = load i16, i16 addrspace(1)* %725, align 2
  %727 = zext i16 %714 to i32
  %728 = shl nuw i32 %727, 16, !spirv.Decorations !921
  %729 = bitcast i32 %728 to float
  %730 = zext i16 %726 to i32
  %731 = shl nuw i32 %730, 16, !spirv.Decorations !921
  %732 = bitcast i32 %731 to float
  %733 = fmul reassoc nsz arcp contract float %729, %732, !spirv.Decorations !898
  %734 = fadd reassoc nsz arcp contract float %733, %.sroa.134.1, !spirv.Decorations !898
  br label %._crit_edge.2.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.1:                                  ; preds = %._crit_edge.1.1.._crit_edge.2.1_crit_edge, %709
  %.sroa.134.2 = phi float [ %734, %709 ], [ %.sroa.134.1, %._crit_edge.1.1.._crit_edge.2.1_crit_edge ]
  br i1 %142, label %735, label %._crit_edge.2.1..preheader.1_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.1..preheader.1_crit_edge:           ; preds = %._crit_edge.2.1
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

735:                                              ; preds = %._crit_edge.2.1
  %.sroa.256.0.insert.ext618 = zext i32 %552 to i64
  %736 = shl nuw nsw i64 %.sroa.256.0.insert.ext618, 1
  %737 = add i64 %551, %736
  %738 = inttoptr i64 %737 to i16 addrspace(4)*
  %739 = addrspacecast i16 addrspace(4)* %738 to i16 addrspace(1)*
  %740 = load i16, i16 addrspace(1)* %739, align 2
  %741 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %742 = extractvalue { i32, i32 } %741, 0
  %743 = extractvalue { i32, i32 } %741, 1
  %744 = insertelement <2 x i32> undef, i32 %742, i32 0
  %745 = insertelement <2 x i32> %744, i32 %743, i32 1
  %746 = bitcast <2 x i32> %745 to i64
  %747 = shl i64 %746, 1
  %748 = add i64 %.in3822, %747
  %749 = add i64 %748, %sink_3856
  %750 = inttoptr i64 %749 to i16 addrspace(4)*
  %751 = addrspacecast i16 addrspace(4)* %750 to i16 addrspace(1)*
  %752 = load i16, i16 addrspace(1)* %751, align 2
  %753 = zext i16 %740 to i32
  %754 = shl nuw i32 %753, 16, !spirv.Decorations !921
  %755 = bitcast i32 %754 to float
  %756 = zext i16 %752 to i32
  %757 = shl nuw i32 %756, 16, !spirv.Decorations !921
  %758 = bitcast i32 %757 to float
  %759 = fmul reassoc nsz arcp contract float %755, %758, !spirv.Decorations !898
  %760 = fadd reassoc nsz arcp contract float %759, %.sroa.198.1, !spirv.Decorations !898
  br label %.preheader.1, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.1:                                     ; preds = %._crit_edge.2.1..preheader.1_crit_edge, %735
  %.sroa.198.2 = phi float [ %760, %735 ], [ %.sroa.198.1, %._crit_edge.2.1..preheader.1_crit_edge ]
  %sink_3854 = shl nsw i64 %351, 1
  br i1 %146, label %761, label %.preheader.1.._crit_edge.274_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.1.._crit_edge.274_crit_edge:           ; preds = %.preheader.1
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

761:                                              ; preds = %.preheader.1
  %.sroa.256.0.insert.ext623 = zext i32 %552 to i64
  %762 = shl nuw nsw i64 %.sroa.256.0.insert.ext623, 1
  %763 = add i64 %548, %762
  %764 = inttoptr i64 %763 to i16 addrspace(4)*
  %765 = addrspacecast i16 addrspace(4)* %764 to i16 addrspace(1)*
  %766 = load i16, i16 addrspace(1)* %765, align 2
  %767 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %768 = extractvalue { i32, i32 } %767, 0
  %769 = extractvalue { i32, i32 } %767, 1
  %770 = insertelement <2 x i32> undef, i32 %768, i32 0
  %771 = insertelement <2 x i32> %770, i32 %769, i32 1
  %772 = bitcast <2 x i32> %771 to i64
  %773 = shl i64 %772, 1
  %774 = add i64 %.in3822, %773
  %775 = add i64 %774, %sink_3854
  %776 = inttoptr i64 %775 to i16 addrspace(4)*
  %777 = addrspacecast i16 addrspace(4)* %776 to i16 addrspace(1)*
  %778 = load i16, i16 addrspace(1)* %777, align 2
  %779 = zext i16 %766 to i32
  %780 = shl nuw i32 %779, 16, !spirv.Decorations !921
  %781 = bitcast i32 %780 to float
  %782 = zext i16 %778 to i32
  %783 = shl nuw i32 %782, 16, !spirv.Decorations !921
  %784 = bitcast i32 %783 to float
  %785 = fmul reassoc nsz arcp contract float %781, %784, !spirv.Decorations !898
  %786 = fadd reassoc nsz arcp contract float %785, %.sroa.10.1, !spirv.Decorations !898
  br label %._crit_edge.274, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.274:                                  ; preds = %.preheader.1.._crit_edge.274_crit_edge, %761
  %.sroa.10.2 = phi float [ %786, %761 ], [ %.sroa.10.1, %.preheader.1.._crit_edge.274_crit_edge ]
  br i1 %149, label %787, label %._crit_edge.274.._crit_edge.1.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.274.._crit_edge.1.2_crit_edge:        ; preds = %._crit_edge.274
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

787:                                              ; preds = %._crit_edge.274
  %.sroa.256.0.insert.ext628 = zext i32 %552 to i64
  %788 = shl nuw nsw i64 %.sroa.256.0.insert.ext628, 1
  %789 = add i64 %549, %788
  %790 = inttoptr i64 %789 to i16 addrspace(4)*
  %791 = addrspacecast i16 addrspace(4)* %790 to i16 addrspace(1)*
  %792 = load i16, i16 addrspace(1)* %791, align 2
  %793 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %794 = extractvalue { i32, i32 } %793, 0
  %795 = extractvalue { i32, i32 } %793, 1
  %796 = insertelement <2 x i32> undef, i32 %794, i32 0
  %797 = insertelement <2 x i32> %796, i32 %795, i32 1
  %798 = bitcast <2 x i32> %797 to i64
  %799 = shl i64 %798, 1
  %800 = add i64 %.in3822, %799
  %801 = add i64 %800, %sink_3854
  %802 = inttoptr i64 %801 to i16 addrspace(4)*
  %803 = addrspacecast i16 addrspace(4)* %802 to i16 addrspace(1)*
  %804 = load i16, i16 addrspace(1)* %803, align 2
  %805 = zext i16 %792 to i32
  %806 = shl nuw i32 %805, 16, !spirv.Decorations !921
  %807 = bitcast i32 %806 to float
  %808 = zext i16 %804 to i32
  %809 = shl nuw i32 %808, 16, !spirv.Decorations !921
  %810 = bitcast i32 %809 to float
  %811 = fmul reassoc nsz arcp contract float %807, %810, !spirv.Decorations !898
  %812 = fadd reassoc nsz arcp contract float %811, %.sroa.74.1, !spirv.Decorations !898
  br label %._crit_edge.1.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.2:                                  ; preds = %._crit_edge.274.._crit_edge.1.2_crit_edge, %787
  %.sroa.74.2 = phi float [ %812, %787 ], [ %.sroa.74.1, %._crit_edge.274.._crit_edge.1.2_crit_edge ]
  br i1 %152, label %813, label %._crit_edge.1.2.._crit_edge.2.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.2.._crit_edge.2.2_crit_edge:        ; preds = %._crit_edge.1.2
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

813:                                              ; preds = %._crit_edge.1.2
  %.sroa.256.0.insert.ext633 = zext i32 %552 to i64
  %814 = shl nuw nsw i64 %.sroa.256.0.insert.ext633, 1
  %815 = add i64 %550, %814
  %816 = inttoptr i64 %815 to i16 addrspace(4)*
  %817 = addrspacecast i16 addrspace(4)* %816 to i16 addrspace(1)*
  %818 = load i16, i16 addrspace(1)* %817, align 2
  %819 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %820 = extractvalue { i32, i32 } %819, 0
  %821 = extractvalue { i32, i32 } %819, 1
  %822 = insertelement <2 x i32> undef, i32 %820, i32 0
  %823 = insertelement <2 x i32> %822, i32 %821, i32 1
  %824 = bitcast <2 x i32> %823 to i64
  %825 = shl i64 %824, 1
  %826 = add i64 %.in3822, %825
  %827 = add i64 %826, %sink_3854
  %828 = inttoptr i64 %827 to i16 addrspace(4)*
  %829 = addrspacecast i16 addrspace(4)* %828 to i16 addrspace(1)*
  %830 = load i16, i16 addrspace(1)* %829, align 2
  %831 = zext i16 %818 to i32
  %832 = shl nuw i32 %831, 16, !spirv.Decorations !921
  %833 = bitcast i32 %832 to float
  %834 = zext i16 %830 to i32
  %835 = shl nuw i32 %834, 16, !spirv.Decorations !921
  %836 = bitcast i32 %835 to float
  %837 = fmul reassoc nsz arcp contract float %833, %836, !spirv.Decorations !898
  %838 = fadd reassoc nsz arcp contract float %837, %.sroa.138.1, !spirv.Decorations !898
  br label %._crit_edge.2.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.2:                                  ; preds = %._crit_edge.1.2.._crit_edge.2.2_crit_edge, %813
  %.sroa.138.2 = phi float [ %838, %813 ], [ %.sroa.138.1, %._crit_edge.1.2.._crit_edge.2.2_crit_edge ]
  br i1 %155, label %839, label %._crit_edge.2.2..preheader.2_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.2..preheader.2_crit_edge:           ; preds = %._crit_edge.2.2
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

839:                                              ; preds = %._crit_edge.2.2
  %.sroa.256.0.insert.ext638 = zext i32 %552 to i64
  %840 = shl nuw nsw i64 %.sroa.256.0.insert.ext638, 1
  %841 = add i64 %551, %840
  %842 = inttoptr i64 %841 to i16 addrspace(4)*
  %843 = addrspacecast i16 addrspace(4)* %842 to i16 addrspace(1)*
  %844 = load i16, i16 addrspace(1)* %843, align 2
  %845 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %846 = extractvalue { i32, i32 } %845, 0
  %847 = extractvalue { i32, i32 } %845, 1
  %848 = insertelement <2 x i32> undef, i32 %846, i32 0
  %849 = insertelement <2 x i32> %848, i32 %847, i32 1
  %850 = bitcast <2 x i32> %849 to i64
  %851 = shl i64 %850, 1
  %852 = add i64 %.in3822, %851
  %853 = add i64 %852, %sink_3854
  %854 = inttoptr i64 %853 to i16 addrspace(4)*
  %855 = addrspacecast i16 addrspace(4)* %854 to i16 addrspace(1)*
  %856 = load i16, i16 addrspace(1)* %855, align 2
  %857 = zext i16 %844 to i32
  %858 = shl nuw i32 %857, 16, !spirv.Decorations !921
  %859 = bitcast i32 %858 to float
  %860 = zext i16 %856 to i32
  %861 = shl nuw i32 %860, 16, !spirv.Decorations !921
  %862 = bitcast i32 %861 to float
  %863 = fmul reassoc nsz arcp contract float %859, %862, !spirv.Decorations !898
  %864 = fadd reassoc nsz arcp contract float %863, %.sroa.202.1, !spirv.Decorations !898
  br label %.preheader.2, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.2:                                     ; preds = %._crit_edge.2.2..preheader.2_crit_edge, %839
  %.sroa.202.2 = phi float [ %864, %839 ], [ %.sroa.202.1, %._crit_edge.2.2..preheader.2_crit_edge ]
  %sink_3852 = shl nsw i64 %352, 1
  br i1 %159, label %865, label %.preheader.2.._crit_edge.375_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.2.._crit_edge.375_crit_edge:           ; preds = %.preheader.2
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

865:                                              ; preds = %.preheader.2
  %.sroa.256.0.insert.ext643 = zext i32 %552 to i64
  %866 = shl nuw nsw i64 %.sroa.256.0.insert.ext643, 1
  %867 = add i64 %548, %866
  %868 = inttoptr i64 %867 to i16 addrspace(4)*
  %869 = addrspacecast i16 addrspace(4)* %868 to i16 addrspace(1)*
  %870 = load i16, i16 addrspace(1)* %869, align 2
  %871 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %872 = extractvalue { i32, i32 } %871, 0
  %873 = extractvalue { i32, i32 } %871, 1
  %874 = insertelement <2 x i32> undef, i32 %872, i32 0
  %875 = insertelement <2 x i32> %874, i32 %873, i32 1
  %876 = bitcast <2 x i32> %875 to i64
  %877 = shl i64 %876, 1
  %878 = add i64 %.in3822, %877
  %879 = add i64 %878, %sink_3852
  %880 = inttoptr i64 %879 to i16 addrspace(4)*
  %881 = addrspacecast i16 addrspace(4)* %880 to i16 addrspace(1)*
  %882 = load i16, i16 addrspace(1)* %881, align 2
  %883 = zext i16 %870 to i32
  %884 = shl nuw i32 %883, 16, !spirv.Decorations !921
  %885 = bitcast i32 %884 to float
  %886 = zext i16 %882 to i32
  %887 = shl nuw i32 %886, 16, !spirv.Decorations !921
  %888 = bitcast i32 %887 to float
  %889 = fmul reassoc nsz arcp contract float %885, %888, !spirv.Decorations !898
  %890 = fadd reassoc nsz arcp contract float %889, %.sroa.14.1, !spirv.Decorations !898
  br label %._crit_edge.375, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.375:                                  ; preds = %.preheader.2.._crit_edge.375_crit_edge, %865
  %.sroa.14.2 = phi float [ %890, %865 ], [ %.sroa.14.1, %.preheader.2.._crit_edge.375_crit_edge ]
  br i1 %162, label %891, label %._crit_edge.375.._crit_edge.1.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.375.._crit_edge.1.3_crit_edge:        ; preds = %._crit_edge.375
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

891:                                              ; preds = %._crit_edge.375
  %.sroa.256.0.insert.ext648 = zext i32 %552 to i64
  %892 = shl nuw nsw i64 %.sroa.256.0.insert.ext648, 1
  %893 = add i64 %549, %892
  %894 = inttoptr i64 %893 to i16 addrspace(4)*
  %895 = addrspacecast i16 addrspace(4)* %894 to i16 addrspace(1)*
  %896 = load i16, i16 addrspace(1)* %895, align 2
  %897 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %898 = extractvalue { i32, i32 } %897, 0
  %899 = extractvalue { i32, i32 } %897, 1
  %900 = insertelement <2 x i32> undef, i32 %898, i32 0
  %901 = insertelement <2 x i32> %900, i32 %899, i32 1
  %902 = bitcast <2 x i32> %901 to i64
  %903 = shl i64 %902, 1
  %904 = add i64 %.in3822, %903
  %905 = add i64 %904, %sink_3852
  %906 = inttoptr i64 %905 to i16 addrspace(4)*
  %907 = addrspacecast i16 addrspace(4)* %906 to i16 addrspace(1)*
  %908 = load i16, i16 addrspace(1)* %907, align 2
  %909 = zext i16 %896 to i32
  %910 = shl nuw i32 %909, 16, !spirv.Decorations !921
  %911 = bitcast i32 %910 to float
  %912 = zext i16 %908 to i32
  %913 = shl nuw i32 %912, 16, !spirv.Decorations !921
  %914 = bitcast i32 %913 to float
  %915 = fmul reassoc nsz arcp contract float %911, %914, !spirv.Decorations !898
  %916 = fadd reassoc nsz arcp contract float %915, %.sroa.78.1, !spirv.Decorations !898
  br label %._crit_edge.1.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.3:                                  ; preds = %._crit_edge.375.._crit_edge.1.3_crit_edge, %891
  %.sroa.78.2 = phi float [ %916, %891 ], [ %.sroa.78.1, %._crit_edge.375.._crit_edge.1.3_crit_edge ]
  br i1 %165, label %917, label %._crit_edge.1.3.._crit_edge.2.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.3.._crit_edge.2.3_crit_edge:        ; preds = %._crit_edge.1.3
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

917:                                              ; preds = %._crit_edge.1.3
  %.sroa.256.0.insert.ext653 = zext i32 %552 to i64
  %918 = shl nuw nsw i64 %.sroa.256.0.insert.ext653, 1
  %919 = add i64 %550, %918
  %920 = inttoptr i64 %919 to i16 addrspace(4)*
  %921 = addrspacecast i16 addrspace(4)* %920 to i16 addrspace(1)*
  %922 = load i16, i16 addrspace(1)* %921, align 2
  %923 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %924 = extractvalue { i32, i32 } %923, 0
  %925 = extractvalue { i32, i32 } %923, 1
  %926 = insertelement <2 x i32> undef, i32 %924, i32 0
  %927 = insertelement <2 x i32> %926, i32 %925, i32 1
  %928 = bitcast <2 x i32> %927 to i64
  %929 = shl i64 %928, 1
  %930 = add i64 %.in3822, %929
  %931 = add i64 %930, %sink_3852
  %932 = inttoptr i64 %931 to i16 addrspace(4)*
  %933 = addrspacecast i16 addrspace(4)* %932 to i16 addrspace(1)*
  %934 = load i16, i16 addrspace(1)* %933, align 2
  %935 = zext i16 %922 to i32
  %936 = shl nuw i32 %935, 16, !spirv.Decorations !921
  %937 = bitcast i32 %936 to float
  %938 = zext i16 %934 to i32
  %939 = shl nuw i32 %938, 16, !spirv.Decorations !921
  %940 = bitcast i32 %939 to float
  %941 = fmul reassoc nsz arcp contract float %937, %940, !spirv.Decorations !898
  %942 = fadd reassoc nsz arcp contract float %941, %.sroa.142.1, !spirv.Decorations !898
  br label %._crit_edge.2.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.3:                                  ; preds = %._crit_edge.1.3.._crit_edge.2.3_crit_edge, %917
  %.sroa.142.2 = phi float [ %942, %917 ], [ %.sroa.142.1, %._crit_edge.1.3.._crit_edge.2.3_crit_edge ]
  br i1 %168, label %943, label %._crit_edge.2.3..preheader.3_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.3..preheader.3_crit_edge:           ; preds = %._crit_edge.2.3
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

943:                                              ; preds = %._crit_edge.2.3
  %.sroa.256.0.insert.ext658 = zext i32 %552 to i64
  %944 = shl nuw nsw i64 %.sroa.256.0.insert.ext658, 1
  %945 = add i64 %551, %944
  %946 = inttoptr i64 %945 to i16 addrspace(4)*
  %947 = addrspacecast i16 addrspace(4)* %946 to i16 addrspace(1)*
  %948 = load i16, i16 addrspace(1)* %947, align 2
  %949 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %950 = extractvalue { i32, i32 } %949, 0
  %951 = extractvalue { i32, i32 } %949, 1
  %952 = insertelement <2 x i32> undef, i32 %950, i32 0
  %953 = insertelement <2 x i32> %952, i32 %951, i32 1
  %954 = bitcast <2 x i32> %953 to i64
  %955 = shl i64 %954, 1
  %956 = add i64 %.in3822, %955
  %957 = add i64 %956, %sink_3852
  %958 = inttoptr i64 %957 to i16 addrspace(4)*
  %959 = addrspacecast i16 addrspace(4)* %958 to i16 addrspace(1)*
  %960 = load i16, i16 addrspace(1)* %959, align 2
  %961 = zext i16 %948 to i32
  %962 = shl nuw i32 %961, 16, !spirv.Decorations !921
  %963 = bitcast i32 %962 to float
  %964 = zext i16 %960 to i32
  %965 = shl nuw i32 %964, 16, !spirv.Decorations !921
  %966 = bitcast i32 %965 to float
  %967 = fmul reassoc nsz arcp contract float %963, %966, !spirv.Decorations !898
  %968 = fadd reassoc nsz arcp contract float %967, %.sroa.206.1, !spirv.Decorations !898
  br label %.preheader.3, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.3:                                     ; preds = %._crit_edge.2.3..preheader.3_crit_edge, %943
  %.sroa.206.2 = phi float [ %968, %943 ], [ %.sroa.206.1, %._crit_edge.2.3..preheader.3_crit_edge ]
  %sink_3850 = shl nsw i64 %353, 1
  br i1 %172, label %969, label %.preheader.3.._crit_edge.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.3.._crit_edge.4_crit_edge:             ; preds = %.preheader.3
  br label %._crit_edge.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

969:                                              ; preds = %.preheader.3
  %.sroa.256.0.insert.ext663 = zext i32 %552 to i64
  %970 = shl nuw nsw i64 %.sroa.256.0.insert.ext663, 1
  %971 = add i64 %548, %970
  %972 = inttoptr i64 %971 to i16 addrspace(4)*
  %973 = addrspacecast i16 addrspace(4)* %972 to i16 addrspace(1)*
  %974 = load i16, i16 addrspace(1)* %973, align 2
  %975 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %976 = extractvalue { i32, i32 } %975, 0
  %977 = extractvalue { i32, i32 } %975, 1
  %978 = insertelement <2 x i32> undef, i32 %976, i32 0
  %979 = insertelement <2 x i32> %978, i32 %977, i32 1
  %980 = bitcast <2 x i32> %979 to i64
  %981 = shl i64 %980, 1
  %982 = add i64 %.in3822, %981
  %983 = add i64 %982, %sink_3850
  %984 = inttoptr i64 %983 to i16 addrspace(4)*
  %985 = addrspacecast i16 addrspace(4)* %984 to i16 addrspace(1)*
  %986 = load i16, i16 addrspace(1)* %985, align 2
  %987 = zext i16 %974 to i32
  %988 = shl nuw i32 %987, 16, !spirv.Decorations !921
  %989 = bitcast i32 %988 to float
  %990 = zext i16 %986 to i32
  %991 = shl nuw i32 %990, 16, !spirv.Decorations !921
  %992 = bitcast i32 %991 to float
  %993 = fmul reassoc nsz arcp contract float %989, %992, !spirv.Decorations !898
  %994 = fadd reassoc nsz arcp contract float %993, %.sroa.18.1, !spirv.Decorations !898
  br label %._crit_edge.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.4:                                    ; preds = %.preheader.3.._crit_edge.4_crit_edge, %969
  %.sroa.18.2 = phi float [ %994, %969 ], [ %.sroa.18.1, %.preheader.3.._crit_edge.4_crit_edge ]
  br i1 %175, label %995, label %._crit_edge.4.._crit_edge.1.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.4.._crit_edge.1.4_crit_edge:          ; preds = %._crit_edge.4
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

995:                                              ; preds = %._crit_edge.4
  %.sroa.256.0.insert.ext668 = zext i32 %552 to i64
  %996 = shl nuw nsw i64 %.sroa.256.0.insert.ext668, 1
  %997 = add i64 %549, %996
  %998 = inttoptr i64 %997 to i16 addrspace(4)*
  %999 = addrspacecast i16 addrspace(4)* %998 to i16 addrspace(1)*
  %1000 = load i16, i16 addrspace(1)* %999, align 2
  %1001 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1002 = extractvalue { i32, i32 } %1001, 0
  %1003 = extractvalue { i32, i32 } %1001, 1
  %1004 = insertelement <2 x i32> undef, i32 %1002, i32 0
  %1005 = insertelement <2 x i32> %1004, i32 %1003, i32 1
  %1006 = bitcast <2 x i32> %1005 to i64
  %1007 = shl i64 %1006, 1
  %1008 = add i64 %.in3822, %1007
  %1009 = add i64 %1008, %sink_3850
  %1010 = inttoptr i64 %1009 to i16 addrspace(4)*
  %1011 = addrspacecast i16 addrspace(4)* %1010 to i16 addrspace(1)*
  %1012 = load i16, i16 addrspace(1)* %1011, align 2
  %1013 = zext i16 %1000 to i32
  %1014 = shl nuw i32 %1013, 16, !spirv.Decorations !921
  %1015 = bitcast i32 %1014 to float
  %1016 = zext i16 %1012 to i32
  %1017 = shl nuw i32 %1016, 16, !spirv.Decorations !921
  %1018 = bitcast i32 %1017 to float
  %1019 = fmul reassoc nsz arcp contract float %1015, %1018, !spirv.Decorations !898
  %1020 = fadd reassoc nsz arcp contract float %1019, %.sroa.82.1, !spirv.Decorations !898
  br label %._crit_edge.1.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.4:                                  ; preds = %._crit_edge.4.._crit_edge.1.4_crit_edge, %995
  %.sroa.82.2 = phi float [ %1020, %995 ], [ %.sroa.82.1, %._crit_edge.4.._crit_edge.1.4_crit_edge ]
  br i1 %178, label %1021, label %._crit_edge.1.4.._crit_edge.2.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.4.._crit_edge.2.4_crit_edge:        ; preds = %._crit_edge.1.4
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1021:                                             ; preds = %._crit_edge.1.4
  %.sroa.256.0.insert.ext673 = zext i32 %552 to i64
  %1022 = shl nuw nsw i64 %.sroa.256.0.insert.ext673, 1
  %1023 = add i64 %550, %1022
  %1024 = inttoptr i64 %1023 to i16 addrspace(4)*
  %1025 = addrspacecast i16 addrspace(4)* %1024 to i16 addrspace(1)*
  %1026 = load i16, i16 addrspace(1)* %1025, align 2
  %1027 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1028 = extractvalue { i32, i32 } %1027, 0
  %1029 = extractvalue { i32, i32 } %1027, 1
  %1030 = insertelement <2 x i32> undef, i32 %1028, i32 0
  %1031 = insertelement <2 x i32> %1030, i32 %1029, i32 1
  %1032 = bitcast <2 x i32> %1031 to i64
  %1033 = shl i64 %1032, 1
  %1034 = add i64 %.in3822, %1033
  %1035 = add i64 %1034, %sink_3850
  %1036 = inttoptr i64 %1035 to i16 addrspace(4)*
  %1037 = addrspacecast i16 addrspace(4)* %1036 to i16 addrspace(1)*
  %1038 = load i16, i16 addrspace(1)* %1037, align 2
  %1039 = zext i16 %1026 to i32
  %1040 = shl nuw i32 %1039, 16, !spirv.Decorations !921
  %1041 = bitcast i32 %1040 to float
  %1042 = zext i16 %1038 to i32
  %1043 = shl nuw i32 %1042, 16, !spirv.Decorations !921
  %1044 = bitcast i32 %1043 to float
  %1045 = fmul reassoc nsz arcp contract float %1041, %1044, !spirv.Decorations !898
  %1046 = fadd reassoc nsz arcp contract float %1045, %.sroa.146.1, !spirv.Decorations !898
  br label %._crit_edge.2.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.4:                                  ; preds = %._crit_edge.1.4.._crit_edge.2.4_crit_edge, %1021
  %.sroa.146.2 = phi float [ %1046, %1021 ], [ %.sroa.146.1, %._crit_edge.1.4.._crit_edge.2.4_crit_edge ]
  br i1 %181, label %1047, label %._crit_edge.2.4..preheader.4_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.4..preheader.4_crit_edge:           ; preds = %._crit_edge.2.4
  br label %.preheader.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1047:                                             ; preds = %._crit_edge.2.4
  %.sroa.256.0.insert.ext678 = zext i32 %552 to i64
  %1048 = shl nuw nsw i64 %.sroa.256.0.insert.ext678, 1
  %1049 = add i64 %551, %1048
  %1050 = inttoptr i64 %1049 to i16 addrspace(4)*
  %1051 = addrspacecast i16 addrspace(4)* %1050 to i16 addrspace(1)*
  %1052 = load i16, i16 addrspace(1)* %1051, align 2
  %1053 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1054 = extractvalue { i32, i32 } %1053, 0
  %1055 = extractvalue { i32, i32 } %1053, 1
  %1056 = insertelement <2 x i32> undef, i32 %1054, i32 0
  %1057 = insertelement <2 x i32> %1056, i32 %1055, i32 1
  %1058 = bitcast <2 x i32> %1057 to i64
  %1059 = shl i64 %1058, 1
  %1060 = add i64 %.in3822, %1059
  %1061 = add i64 %1060, %sink_3850
  %1062 = inttoptr i64 %1061 to i16 addrspace(4)*
  %1063 = addrspacecast i16 addrspace(4)* %1062 to i16 addrspace(1)*
  %1064 = load i16, i16 addrspace(1)* %1063, align 2
  %1065 = zext i16 %1052 to i32
  %1066 = shl nuw i32 %1065, 16, !spirv.Decorations !921
  %1067 = bitcast i32 %1066 to float
  %1068 = zext i16 %1064 to i32
  %1069 = shl nuw i32 %1068, 16, !spirv.Decorations !921
  %1070 = bitcast i32 %1069 to float
  %1071 = fmul reassoc nsz arcp contract float %1067, %1070, !spirv.Decorations !898
  %1072 = fadd reassoc nsz arcp contract float %1071, %.sroa.210.1, !spirv.Decorations !898
  br label %.preheader.4, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.4:                                     ; preds = %._crit_edge.2.4..preheader.4_crit_edge, %1047
  %.sroa.210.2 = phi float [ %1072, %1047 ], [ %.sroa.210.1, %._crit_edge.2.4..preheader.4_crit_edge ]
  %sink_3848 = shl nsw i64 %354, 1
  br i1 %185, label %1073, label %.preheader.4.._crit_edge.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.4.._crit_edge.5_crit_edge:             ; preds = %.preheader.4
  br label %._crit_edge.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1073:                                             ; preds = %.preheader.4
  %.sroa.256.0.insert.ext683 = zext i32 %552 to i64
  %1074 = shl nuw nsw i64 %.sroa.256.0.insert.ext683, 1
  %1075 = add i64 %548, %1074
  %1076 = inttoptr i64 %1075 to i16 addrspace(4)*
  %1077 = addrspacecast i16 addrspace(4)* %1076 to i16 addrspace(1)*
  %1078 = load i16, i16 addrspace(1)* %1077, align 2
  %1079 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1080 = extractvalue { i32, i32 } %1079, 0
  %1081 = extractvalue { i32, i32 } %1079, 1
  %1082 = insertelement <2 x i32> undef, i32 %1080, i32 0
  %1083 = insertelement <2 x i32> %1082, i32 %1081, i32 1
  %1084 = bitcast <2 x i32> %1083 to i64
  %1085 = shl i64 %1084, 1
  %1086 = add i64 %.in3822, %1085
  %1087 = add i64 %1086, %sink_3848
  %1088 = inttoptr i64 %1087 to i16 addrspace(4)*
  %1089 = addrspacecast i16 addrspace(4)* %1088 to i16 addrspace(1)*
  %1090 = load i16, i16 addrspace(1)* %1089, align 2
  %1091 = zext i16 %1078 to i32
  %1092 = shl nuw i32 %1091, 16, !spirv.Decorations !921
  %1093 = bitcast i32 %1092 to float
  %1094 = zext i16 %1090 to i32
  %1095 = shl nuw i32 %1094, 16, !spirv.Decorations !921
  %1096 = bitcast i32 %1095 to float
  %1097 = fmul reassoc nsz arcp contract float %1093, %1096, !spirv.Decorations !898
  %1098 = fadd reassoc nsz arcp contract float %1097, %.sroa.22.1, !spirv.Decorations !898
  br label %._crit_edge.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.5:                                    ; preds = %.preheader.4.._crit_edge.5_crit_edge, %1073
  %.sroa.22.2 = phi float [ %1098, %1073 ], [ %.sroa.22.1, %.preheader.4.._crit_edge.5_crit_edge ]
  br i1 %188, label %1099, label %._crit_edge.5.._crit_edge.1.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.5.._crit_edge.1.5_crit_edge:          ; preds = %._crit_edge.5
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1099:                                             ; preds = %._crit_edge.5
  %.sroa.256.0.insert.ext688 = zext i32 %552 to i64
  %1100 = shl nuw nsw i64 %.sroa.256.0.insert.ext688, 1
  %1101 = add i64 %549, %1100
  %1102 = inttoptr i64 %1101 to i16 addrspace(4)*
  %1103 = addrspacecast i16 addrspace(4)* %1102 to i16 addrspace(1)*
  %1104 = load i16, i16 addrspace(1)* %1103, align 2
  %1105 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1106 = extractvalue { i32, i32 } %1105, 0
  %1107 = extractvalue { i32, i32 } %1105, 1
  %1108 = insertelement <2 x i32> undef, i32 %1106, i32 0
  %1109 = insertelement <2 x i32> %1108, i32 %1107, i32 1
  %1110 = bitcast <2 x i32> %1109 to i64
  %1111 = shl i64 %1110, 1
  %1112 = add i64 %.in3822, %1111
  %1113 = add i64 %1112, %sink_3848
  %1114 = inttoptr i64 %1113 to i16 addrspace(4)*
  %1115 = addrspacecast i16 addrspace(4)* %1114 to i16 addrspace(1)*
  %1116 = load i16, i16 addrspace(1)* %1115, align 2
  %1117 = zext i16 %1104 to i32
  %1118 = shl nuw i32 %1117, 16, !spirv.Decorations !921
  %1119 = bitcast i32 %1118 to float
  %1120 = zext i16 %1116 to i32
  %1121 = shl nuw i32 %1120, 16, !spirv.Decorations !921
  %1122 = bitcast i32 %1121 to float
  %1123 = fmul reassoc nsz arcp contract float %1119, %1122, !spirv.Decorations !898
  %1124 = fadd reassoc nsz arcp contract float %1123, %.sroa.86.1, !spirv.Decorations !898
  br label %._crit_edge.1.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.5:                                  ; preds = %._crit_edge.5.._crit_edge.1.5_crit_edge, %1099
  %.sroa.86.2 = phi float [ %1124, %1099 ], [ %.sroa.86.1, %._crit_edge.5.._crit_edge.1.5_crit_edge ]
  br i1 %191, label %1125, label %._crit_edge.1.5.._crit_edge.2.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.5.._crit_edge.2.5_crit_edge:        ; preds = %._crit_edge.1.5
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1125:                                             ; preds = %._crit_edge.1.5
  %.sroa.256.0.insert.ext693 = zext i32 %552 to i64
  %1126 = shl nuw nsw i64 %.sroa.256.0.insert.ext693, 1
  %1127 = add i64 %550, %1126
  %1128 = inttoptr i64 %1127 to i16 addrspace(4)*
  %1129 = addrspacecast i16 addrspace(4)* %1128 to i16 addrspace(1)*
  %1130 = load i16, i16 addrspace(1)* %1129, align 2
  %1131 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1132 = extractvalue { i32, i32 } %1131, 0
  %1133 = extractvalue { i32, i32 } %1131, 1
  %1134 = insertelement <2 x i32> undef, i32 %1132, i32 0
  %1135 = insertelement <2 x i32> %1134, i32 %1133, i32 1
  %1136 = bitcast <2 x i32> %1135 to i64
  %1137 = shl i64 %1136, 1
  %1138 = add i64 %.in3822, %1137
  %1139 = add i64 %1138, %sink_3848
  %1140 = inttoptr i64 %1139 to i16 addrspace(4)*
  %1141 = addrspacecast i16 addrspace(4)* %1140 to i16 addrspace(1)*
  %1142 = load i16, i16 addrspace(1)* %1141, align 2
  %1143 = zext i16 %1130 to i32
  %1144 = shl nuw i32 %1143, 16, !spirv.Decorations !921
  %1145 = bitcast i32 %1144 to float
  %1146 = zext i16 %1142 to i32
  %1147 = shl nuw i32 %1146, 16, !spirv.Decorations !921
  %1148 = bitcast i32 %1147 to float
  %1149 = fmul reassoc nsz arcp contract float %1145, %1148, !spirv.Decorations !898
  %1150 = fadd reassoc nsz arcp contract float %1149, %.sroa.150.1, !spirv.Decorations !898
  br label %._crit_edge.2.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.5:                                  ; preds = %._crit_edge.1.5.._crit_edge.2.5_crit_edge, %1125
  %.sroa.150.2 = phi float [ %1150, %1125 ], [ %.sroa.150.1, %._crit_edge.1.5.._crit_edge.2.5_crit_edge ]
  br i1 %194, label %1151, label %._crit_edge.2.5..preheader.5_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.5..preheader.5_crit_edge:           ; preds = %._crit_edge.2.5
  br label %.preheader.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1151:                                             ; preds = %._crit_edge.2.5
  %.sroa.256.0.insert.ext698 = zext i32 %552 to i64
  %1152 = shl nuw nsw i64 %.sroa.256.0.insert.ext698, 1
  %1153 = add i64 %551, %1152
  %1154 = inttoptr i64 %1153 to i16 addrspace(4)*
  %1155 = addrspacecast i16 addrspace(4)* %1154 to i16 addrspace(1)*
  %1156 = load i16, i16 addrspace(1)* %1155, align 2
  %1157 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1158 = extractvalue { i32, i32 } %1157, 0
  %1159 = extractvalue { i32, i32 } %1157, 1
  %1160 = insertelement <2 x i32> undef, i32 %1158, i32 0
  %1161 = insertelement <2 x i32> %1160, i32 %1159, i32 1
  %1162 = bitcast <2 x i32> %1161 to i64
  %1163 = shl i64 %1162, 1
  %1164 = add i64 %.in3822, %1163
  %1165 = add i64 %1164, %sink_3848
  %1166 = inttoptr i64 %1165 to i16 addrspace(4)*
  %1167 = addrspacecast i16 addrspace(4)* %1166 to i16 addrspace(1)*
  %1168 = load i16, i16 addrspace(1)* %1167, align 2
  %1169 = zext i16 %1156 to i32
  %1170 = shl nuw i32 %1169, 16, !spirv.Decorations !921
  %1171 = bitcast i32 %1170 to float
  %1172 = zext i16 %1168 to i32
  %1173 = shl nuw i32 %1172, 16, !spirv.Decorations !921
  %1174 = bitcast i32 %1173 to float
  %1175 = fmul reassoc nsz arcp contract float %1171, %1174, !spirv.Decorations !898
  %1176 = fadd reassoc nsz arcp contract float %1175, %.sroa.214.1, !spirv.Decorations !898
  br label %.preheader.5, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.5:                                     ; preds = %._crit_edge.2.5..preheader.5_crit_edge, %1151
  %.sroa.214.2 = phi float [ %1176, %1151 ], [ %.sroa.214.1, %._crit_edge.2.5..preheader.5_crit_edge ]
  %sink_3846 = shl nsw i64 %355, 1
  br i1 %198, label %1177, label %.preheader.5.._crit_edge.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.5.._crit_edge.6_crit_edge:             ; preds = %.preheader.5
  br label %._crit_edge.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1177:                                             ; preds = %.preheader.5
  %.sroa.256.0.insert.ext703 = zext i32 %552 to i64
  %1178 = shl nuw nsw i64 %.sroa.256.0.insert.ext703, 1
  %1179 = add i64 %548, %1178
  %1180 = inttoptr i64 %1179 to i16 addrspace(4)*
  %1181 = addrspacecast i16 addrspace(4)* %1180 to i16 addrspace(1)*
  %1182 = load i16, i16 addrspace(1)* %1181, align 2
  %1183 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1184 = extractvalue { i32, i32 } %1183, 0
  %1185 = extractvalue { i32, i32 } %1183, 1
  %1186 = insertelement <2 x i32> undef, i32 %1184, i32 0
  %1187 = insertelement <2 x i32> %1186, i32 %1185, i32 1
  %1188 = bitcast <2 x i32> %1187 to i64
  %1189 = shl i64 %1188, 1
  %1190 = add i64 %.in3822, %1189
  %1191 = add i64 %1190, %sink_3846
  %1192 = inttoptr i64 %1191 to i16 addrspace(4)*
  %1193 = addrspacecast i16 addrspace(4)* %1192 to i16 addrspace(1)*
  %1194 = load i16, i16 addrspace(1)* %1193, align 2
  %1195 = zext i16 %1182 to i32
  %1196 = shl nuw i32 %1195, 16, !spirv.Decorations !921
  %1197 = bitcast i32 %1196 to float
  %1198 = zext i16 %1194 to i32
  %1199 = shl nuw i32 %1198, 16, !spirv.Decorations !921
  %1200 = bitcast i32 %1199 to float
  %1201 = fmul reassoc nsz arcp contract float %1197, %1200, !spirv.Decorations !898
  %1202 = fadd reassoc nsz arcp contract float %1201, %.sroa.26.1, !spirv.Decorations !898
  br label %._crit_edge.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.6:                                    ; preds = %.preheader.5.._crit_edge.6_crit_edge, %1177
  %.sroa.26.2 = phi float [ %1202, %1177 ], [ %.sroa.26.1, %.preheader.5.._crit_edge.6_crit_edge ]
  br i1 %201, label %1203, label %._crit_edge.6.._crit_edge.1.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.6.._crit_edge.1.6_crit_edge:          ; preds = %._crit_edge.6
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1203:                                             ; preds = %._crit_edge.6
  %.sroa.256.0.insert.ext708 = zext i32 %552 to i64
  %1204 = shl nuw nsw i64 %.sroa.256.0.insert.ext708, 1
  %1205 = add i64 %549, %1204
  %1206 = inttoptr i64 %1205 to i16 addrspace(4)*
  %1207 = addrspacecast i16 addrspace(4)* %1206 to i16 addrspace(1)*
  %1208 = load i16, i16 addrspace(1)* %1207, align 2
  %1209 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1210 = extractvalue { i32, i32 } %1209, 0
  %1211 = extractvalue { i32, i32 } %1209, 1
  %1212 = insertelement <2 x i32> undef, i32 %1210, i32 0
  %1213 = insertelement <2 x i32> %1212, i32 %1211, i32 1
  %1214 = bitcast <2 x i32> %1213 to i64
  %1215 = shl i64 %1214, 1
  %1216 = add i64 %.in3822, %1215
  %1217 = add i64 %1216, %sink_3846
  %1218 = inttoptr i64 %1217 to i16 addrspace(4)*
  %1219 = addrspacecast i16 addrspace(4)* %1218 to i16 addrspace(1)*
  %1220 = load i16, i16 addrspace(1)* %1219, align 2
  %1221 = zext i16 %1208 to i32
  %1222 = shl nuw i32 %1221, 16, !spirv.Decorations !921
  %1223 = bitcast i32 %1222 to float
  %1224 = zext i16 %1220 to i32
  %1225 = shl nuw i32 %1224, 16, !spirv.Decorations !921
  %1226 = bitcast i32 %1225 to float
  %1227 = fmul reassoc nsz arcp contract float %1223, %1226, !spirv.Decorations !898
  %1228 = fadd reassoc nsz arcp contract float %1227, %.sroa.90.1, !spirv.Decorations !898
  br label %._crit_edge.1.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.6:                                  ; preds = %._crit_edge.6.._crit_edge.1.6_crit_edge, %1203
  %.sroa.90.2 = phi float [ %1228, %1203 ], [ %.sroa.90.1, %._crit_edge.6.._crit_edge.1.6_crit_edge ]
  br i1 %204, label %1229, label %._crit_edge.1.6.._crit_edge.2.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.6.._crit_edge.2.6_crit_edge:        ; preds = %._crit_edge.1.6
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1229:                                             ; preds = %._crit_edge.1.6
  %.sroa.256.0.insert.ext713 = zext i32 %552 to i64
  %1230 = shl nuw nsw i64 %.sroa.256.0.insert.ext713, 1
  %1231 = add i64 %550, %1230
  %1232 = inttoptr i64 %1231 to i16 addrspace(4)*
  %1233 = addrspacecast i16 addrspace(4)* %1232 to i16 addrspace(1)*
  %1234 = load i16, i16 addrspace(1)* %1233, align 2
  %1235 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1236 = extractvalue { i32, i32 } %1235, 0
  %1237 = extractvalue { i32, i32 } %1235, 1
  %1238 = insertelement <2 x i32> undef, i32 %1236, i32 0
  %1239 = insertelement <2 x i32> %1238, i32 %1237, i32 1
  %1240 = bitcast <2 x i32> %1239 to i64
  %1241 = shl i64 %1240, 1
  %1242 = add i64 %.in3822, %1241
  %1243 = add i64 %1242, %sink_3846
  %1244 = inttoptr i64 %1243 to i16 addrspace(4)*
  %1245 = addrspacecast i16 addrspace(4)* %1244 to i16 addrspace(1)*
  %1246 = load i16, i16 addrspace(1)* %1245, align 2
  %1247 = zext i16 %1234 to i32
  %1248 = shl nuw i32 %1247, 16, !spirv.Decorations !921
  %1249 = bitcast i32 %1248 to float
  %1250 = zext i16 %1246 to i32
  %1251 = shl nuw i32 %1250, 16, !spirv.Decorations !921
  %1252 = bitcast i32 %1251 to float
  %1253 = fmul reassoc nsz arcp contract float %1249, %1252, !spirv.Decorations !898
  %1254 = fadd reassoc nsz arcp contract float %1253, %.sroa.154.1, !spirv.Decorations !898
  br label %._crit_edge.2.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.6:                                  ; preds = %._crit_edge.1.6.._crit_edge.2.6_crit_edge, %1229
  %.sroa.154.2 = phi float [ %1254, %1229 ], [ %.sroa.154.1, %._crit_edge.1.6.._crit_edge.2.6_crit_edge ]
  br i1 %207, label %1255, label %._crit_edge.2.6..preheader.6_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.6..preheader.6_crit_edge:           ; preds = %._crit_edge.2.6
  br label %.preheader.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1255:                                             ; preds = %._crit_edge.2.6
  %.sroa.256.0.insert.ext718 = zext i32 %552 to i64
  %1256 = shl nuw nsw i64 %.sroa.256.0.insert.ext718, 1
  %1257 = add i64 %551, %1256
  %1258 = inttoptr i64 %1257 to i16 addrspace(4)*
  %1259 = addrspacecast i16 addrspace(4)* %1258 to i16 addrspace(1)*
  %1260 = load i16, i16 addrspace(1)* %1259, align 2
  %1261 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1262 = extractvalue { i32, i32 } %1261, 0
  %1263 = extractvalue { i32, i32 } %1261, 1
  %1264 = insertelement <2 x i32> undef, i32 %1262, i32 0
  %1265 = insertelement <2 x i32> %1264, i32 %1263, i32 1
  %1266 = bitcast <2 x i32> %1265 to i64
  %1267 = shl i64 %1266, 1
  %1268 = add i64 %.in3822, %1267
  %1269 = add i64 %1268, %sink_3846
  %1270 = inttoptr i64 %1269 to i16 addrspace(4)*
  %1271 = addrspacecast i16 addrspace(4)* %1270 to i16 addrspace(1)*
  %1272 = load i16, i16 addrspace(1)* %1271, align 2
  %1273 = zext i16 %1260 to i32
  %1274 = shl nuw i32 %1273, 16, !spirv.Decorations !921
  %1275 = bitcast i32 %1274 to float
  %1276 = zext i16 %1272 to i32
  %1277 = shl nuw i32 %1276, 16, !spirv.Decorations !921
  %1278 = bitcast i32 %1277 to float
  %1279 = fmul reassoc nsz arcp contract float %1275, %1278, !spirv.Decorations !898
  %1280 = fadd reassoc nsz arcp contract float %1279, %.sroa.218.1, !spirv.Decorations !898
  br label %.preheader.6, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.6:                                     ; preds = %._crit_edge.2.6..preheader.6_crit_edge, %1255
  %.sroa.218.2 = phi float [ %1280, %1255 ], [ %.sroa.218.1, %._crit_edge.2.6..preheader.6_crit_edge ]
  %sink_3844 = shl nsw i64 %356, 1
  br i1 %211, label %1281, label %.preheader.6.._crit_edge.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.6.._crit_edge.7_crit_edge:             ; preds = %.preheader.6
  br label %._crit_edge.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1281:                                             ; preds = %.preheader.6
  %.sroa.256.0.insert.ext723 = zext i32 %552 to i64
  %1282 = shl nuw nsw i64 %.sroa.256.0.insert.ext723, 1
  %1283 = add i64 %548, %1282
  %1284 = inttoptr i64 %1283 to i16 addrspace(4)*
  %1285 = addrspacecast i16 addrspace(4)* %1284 to i16 addrspace(1)*
  %1286 = load i16, i16 addrspace(1)* %1285, align 2
  %1287 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1288 = extractvalue { i32, i32 } %1287, 0
  %1289 = extractvalue { i32, i32 } %1287, 1
  %1290 = insertelement <2 x i32> undef, i32 %1288, i32 0
  %1291 = insertelement <2 x i32> %1290, i32 %1289, i32 1
  %1292 = bitcast <2 x i32> %1291 to i64
  %1293 = shl i64 %1292, 1
  %1294 = add i64 %.in3822, %1293
  %1295 = add i64 %1294, %sink_3844
  %1296 = inttoptr i64 %1295 to i16 addrspace(4)*
  %1297 = addrspacecast i16 addrspace(4)* %1296 to i16 addrspace(1)*
  %1298 = load i16, i16 addrspace(1)* %1297, align 2
  %1299 = zext i16 %1286 to i32
  %1300 = shl nuw i32 %1299, 16, !spirv.Decorations !921
  %1301 = bitcast i32 %1300 to float
  %1302 = zext i16 %1298 to i32
  %1303 = shl nuw i32 %1302, 16, !spirv.Decorations !921
  %1304 = bitcast i32 %1303 to float
  %1305 = fmul reassoc nsz arcp contract float %1301, %1304, !spirv.Decorations !898
  %1306 = fadd reassoc nsz arcp contract float %1305, %.sroa.30.1, !spirv.Decorations !898
  br label %._crit_edge.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.7:                                    ; preds = %.preheader.6.._crit_edge.7_crit_edge, %1281
  %.sroa.30.2 = phi float [ %1306, %1281 ], [ %.sroa.30.1, %.preheader.6.._crit_edge.7_crit_edge ]
  br i1 %214, label %1307, label %._crit_edge.7.._crit_edge.1.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.7.._crit_edge.1.7_crit_edge:          ; preds = %._crit_edge.7
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1307:                                             ; preds = %._crit_edge.7
  %.sroa.256.0.insert.ext728 = zext i32 %552 to i64
  %1308 = shl nuw nsw i64 %.sroa.256.0.insert.ext728, 1
  %1309 = add i64 %549, %1308
  %1310 = inttoptr i64 %1309 to i16 addrspace(4)*
  %1311 = addrspacecast i16 addrspace(4)* %1310 to i16 addrspace(1)*
  %1312 = load i16, i16 addrspace(1)* %1311, align 2
  %1313 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1314 = extractvalue { i32, i32 } %1313, 0
  %1315 = extractvalue { i32, i32 } %1313, 1
  %1316 = insertelement <2 x i32> undef, i32 %1314, i32 0
  %1317 = insertelement <2 x i32> %1316, i32 %1315, i32 1
  %1318 = bitcast <2 x i32> %1317 to i64
  %1319 = shl i64 %1318, 1
  %1320 = add i64 %.in3822, %1319
  %1321 = add i64 %1320, %sink_3844
  %1322 = inttoptr i64 %1321 to i16 addrspace(4)*
  %1323 = addrspacecast i16 addrspace(4)* %1322 to i16 addrspace(1)*
  %1324 = load i16, i16 addrspace(1)* %1323, align 2
  %1325 = zext i16 %1312 to i32
  %1326 = shl nuw i32 %1325, 16, !spirv.Decorations !921
  %1327 = bitcast i32 %1326 to float
  %1328 = zext i16 %1324 to i32
  %1329 = shl nuw i32 %1328, 16, !spirv.Decorations !921
  %1330 = bitcast i32 %1329 to float
  %1331 = fmul reassoc nsz arcp contract float %1327, %1330, !spirv.Decorations !898
  %1332 = fadd reassoc nsz arcp contract float %1331, %.sroa.94.1, !spirv.Decorations !898
  br label %._crit_edge.1.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.7:                                  ; preds = %._crit_edge.7.._crit_edge.1.7_crit_edge, %1307
  %.sroa.94.2 = phi float [ %1332, %1307 ], [ %.sroa.94.1, %._crit_edge.7.._crit_edge.1.7_crit_edge ]
  br i1 %217, label %1333, label %._crit_edge.1.7.._crit_edge.2.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.7.._crit_edge.2.7_crit_edge:        ; preds = %._crit_edge.1.7
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1333:                                             ; preds = %._crit_edge.1.7
  %.sroa.256.0.insert.ext733 = zext i32 %552 to i64
  %1334 = shl nuw nsw i64 %.sroa.256.0.insert.ext733, 1
  %1335 = add i64 %550, %1334
  %1336 = inttoptr i64 %1335 to i16 addrspace(4)*
  %1337 = addrspacecast i16 addrspace(4)* %1336 to i16 addrspace(1)*
  %1338 = load i16, i16 addrspace(1)* %1337, align 2
  %1339 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1340 = extractvalue { i32, i32 } %1339, 0
  %1341 = extractvalue { i32, i32 } %1339, 1
  %1342 = insertelement <2 x i32> undef, i32 %1340, i32 0
  %1343 = insertelement <2 x i32> %1342, i32 %1341, i32 1
  %1344 = bitcast <2 x i32> %1343 to i64
  %1345 = shl i64 %1344, 1
  %1346 = add i64 %.in3822, %1345
  %1347 = add i64 %1346, %sink_3844
  %1348 = inttoptr i64 %1347 to i16 addrspace(4)*
  %1349 = addrspacecast i16 addrspace(4)* %1348 to i16 addrspace(1)*
  %1350 = load i16, i16 addrspace(1)* %1349, align 2
  %1351 = zext i16 %1338 to i32
  %1352 = shl nuw i32 %1351, 16, !spirv.Decorations !921
  %1353 = bitcast i32 %1352 to float
  %1354 = zext i16 %1350 to i32
  %1355 = shl nuw i32 %1354, 16, !spirv.Decorations !921
  %1356 = bitcast i32 %1355 to float
  %1357 = fmul reassoc nsz arcp contract float %1353, %1356, !spirv.Decorations !898
  %1358 = fadd reassoc nsz arcp contract float %1357, %.sroa.158.1, !spirv.Decorations !898
  br label %._crit_edge.2.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.7:                                  ; preds = %._crit_edge.1.7.._crit_edge.2.7_crit_edge, %1333
  %.sroa.158.2 = phi float [ %1358, %1333 ], [ %.sroa.158.1, %._crit_edge.1.7.._crit_edge.2.7_crit_edge ]
  br i1 %220, label %1359, label %._crit_edge.2.7..preheader.7_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.7..preheader.7_crit_edge:           ; preds = %._crit_edge.2.7
  br label %.preheader.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1359:                                             ; preds = %._crit_edge.2.7
  %.sroa.256.0.insert.ext738 = zext i32 %552 to i64
  %1360 = shl nuw nsw i64 %.sroa.256.0.insert.ext738, 1
  %1361 = add i64 %551, %1360
  %1362 = inttoptr i64 %1361 to i16 addrspace(4)*
  %1363 = addrspacecast i16 addrspace(4)* %1362 to i16 addrspace(1)*
  %1364 = load i16, i16 addrspace(1)* %1363, align 2
  %1365 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1366 = extractvalue { i32, i32 } %1365, 0
  %1367 = extractvalue { i32, i32 } %1365, 1
  %1368 = insertelement <2 x i32> undef, i32 %1366, i32 0
  %1369 = insertelement <2 x i32> %1368, i32 %1367, i32 1
  %1370 = bitcast <2 x i32> %1369 to i64
  %1371 = shl i64 %1370, 1
  %1372 = add i64 %.in3822, %1371
  %1373 = add i64 %1372, %sink_3844
  %1374 = inttoptr i64 %1373 to i16 addrspace(4)*
  %1375 = addrspacecast i16 addrspace(4)* %1374 to i16 addrspace(1)*
  %1376 = load i16, i16 addrspace(1)* %1375, align 2
  %1377 = zext i16 %1364 to i32
  %1378 = shl nuw i32 %1377, 16, !spirv.Decorations !921
  %1379 = bitcast i32 %1378 to float
  %1380 = zext i16 %1376 to i32
  %1381 = shl nuw i32 %1380, 16, !spirv.Decorations !921
  %1382 = bitcast i32 %1381 to float
  %1383 = fmul reassoc nsz arcp contract float %1379, %1382, !spirv.Decorations !898
  %1384 = fadd reassoc nsz arcp contract float %1383, %.sroa.222.1, !spirv.Decorations !898
  br label %.preheader.7, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.7:                                     ; preds = %._crit_edge.2.7..preheader.7_crit_edge, %1359
  %.sroa.222.2 = phi float [ %1384, %1359 ], [ %.sroa.222.1, %._crit_edge.2.7..preheader.7_crit_edge ]
  %sink_3842 = shl nsw i64 %357, 1
  br i1 %224, label %1385, label %.preheader.7.._crit_edge.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.7.._crit_edge.8_crit_edge:             ; preds = %.preheader.7
  br label %._crit_edge.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1385:                                             ; preds = %.preheader.7
  %.sroa.256.0.insert.ext743 = zext i32 %552 to i64
  %1386 = shl nuw nsw i64 %.sroa.256.0.insert.ext743, 1
  %1387 = add i64 %548, %1386
  %1388 = inttoptr i64 %1387 to i16 addrspace(4)*
  %1389 = addrspacecast i16 addrspace(4)* %1388 to i16 addrspace(1)*
  %1390 = load i16, i16 addrspace(1)* %1389, align 2
  %1391 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1392 = extractvalue { i32, i32 } %1391, 0
  %1393 = extractvalue { i32, i32 } %1391, 1
  %1394 = insertelement <2 x i32> undef, i32 %1392, i32 0
  %1395 = insertelement <2 x i32> %1394, i32 %1393, i32 1
  %1396 = bitcast <2 x i32> %1395 to i64
  %1397 = shl i64 %1396, 1
  %1398 = add i64 %.in3822, %1397
  %1399 = add i64 %1398, %sink_3842
  %1400 = inttoptr i64 %1399 to i16 addrspace(4)*
  %1401 = addrspacecast i16 addrspace(4)* %1400 to i16 addrspace(1)*
  %1402 = load i16, i16 addrspace(1)* %1401, align 2
  %1403 = zext i16 %1390 to i32
  %1404 = shl nuw i32 %1403, 16, !spirv.Decorations !921
  %1405 = bitcast i32 %1404 to float
  %1406 = zext i16 %1402 to i32
  %1407 = shl nuw i32 %1406, 16, !spirv.Decorations !921
  %1408 = bitcast i32 %1407 to float
  %1409 = fmul reassoc nsz arcp contract float %1405, %1408, !spirv.Decorations !898
  %1410 = fadd reassoc nsz arcp contract float %1409, %.sroa.34.1, !spirv.Decorations !898
  br label %._crit_edge.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.8:                                    ; preds = %.preheader.7.._crit_edge.8_crit_edge, %1385
  %.sroa.34.2 = phi float [ %1410, %1385 ], [ %.sroa.34.1, %.preheader.7.._crit_edge.8_crit_edge ]
  br i1 %227, label %1411, label %._crit_edge.8.._crit_edge.1.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.8.._crit_edge.1.8_crit_edge:          ; preds = %._crit_edge.8
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1411:                                             ; preds = %._crit_edge.8
  %.sroa.256.0.insert.ext748 = zext i32 %552 to i64
  %1412 = shl nuw nsw i64 %.sroa.256.0.insert.ext748, 1
  %1413 = add i64 %549, %1412
  %1414 = inttoptr i64 %1413 to i16 addrspace(4)*
  %1415 = addrspacecast i16 addrspace(4)* %1414 to i16 addrspace(1)*
  %1416 = load i16, i16 addrspace(1)* %1415, align 2
  %1417 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1418 = extractvalue { i32, i32 } %1417, 0
  %1419 = extractvalue { i32, i32 } %1417, 1
  %1420 = insertelement <2 x i32> undef, i32 %1418, i32 0
  %1421 = insertelement <2 x i32> %1420, i32 %1419, i32 1
  %1422 = bitcast <2 x i32> %1421 to i64
  %1423 = shl i64 %1422, 1
  %1424 = add i64 %.in3822, %1423
  %1425 = add i64 %1424, %sink_3842
  %1426 = inttoptr i64 %1425 to i16 addrspace(4)*
  %1427 = addrspacecast i16 addrspace(4)* %1426 to i16 addrspace(1)*
  %1428 = load i16, i16 addrspace(1)* %1427, align 2
  %1429 = zext i16 %1416 to i32
  %1430 = shl nuw i32 %1429, 16, !spirv.Decorations !921
  %1431 = bitcast i32 %1430 to float
  %1432 = zext i16 %1428 to i32
  %1433 = shl nuw i32 %1432, 16, !spirv.Decorations !921
  %1434 = bitcast i32 %1433 to float
  %1435 = fmul reassoc nsz arcp contract float %1431, %1434, !spirv.Decorations !898
  %1436 = fadd reassoc nsz arcp contract float %1435, %.sroa.98.1, !spirv.Decorations !898
  br label %._crit_edge.1.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.8:                                  ; preds = %._crit_edge.8.._crit_edge.1.8_crit_edge, %1411
  %.sroa.98.2 = phi float [ %1436, %1411 ], [ %.sroa.98.1, %._crit_edge.8.._crit_edge.1.8_crit_edge ]
  br i1 %230, label %1437, label %._crit_edge.1.8.._crit_edge.2.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.8.._crit_edge.2.8_crit_edge:        ; preds = %._crit_edge.1.8
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1437:                                             ; preds = %._crit_edge.1.8
  %.sroa.256.0.insert.ext753 = zext i32 %552 to i64
  %1438 = shl nuw nsw i64 %.sroa.256.0.insert.ext753, 1
  %1439 = add i64 %550, %1438
  %1440 = inttoptr i64 %1439 to i16 addrspace(4)*
  %1441 = addrspacecast i16 addrspace(4)* %1440 to i16 addrspace(1)*
  %1442 = load i16, i16 addrspace(1)* %1441, align 2
  %1443 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1444 = extractvalue { i32, i32 } %1443, 0
  %1445 = extractvalue { i32, i32 } %1443, 1
  %1446 = insertelement <2 x i32> undef, i32 %1444, i32 0
  %1447 = insertelement <2 x i32> %1446, i32 %1445, i32 1
  %1448 = bitcast <2 x i32> %1447 to i64
  %1449 = shl i64 %1448, 1
  %1450 = add i64 %.in3822, %1449
  %1451 = add i64 %1450, %sink_3842
  %1452 = inttoptr i64 %1451 to i16 addrspace(4)*
  %1453 = addrspacecast i16 addrspace(4)* %1452 to i16 addrspace(1)*
  %1454 = load i16, i16 addrspace(1)* %1453, align 2
  %1455 = zext i16 %1442 to i32
  %1456 = shl nuw i32 %1455, 16, !spirv.Decorations !921
  %1457 = bitcast i32 %1456 to float
  %1458 = zext i16 %1454 to i32
  %1459 = shl nuw i32 %1458, 16, !spirv.Decorations !921
  %1460 = bitcast i32 %1459 to float
  %1461 = fmul reassoc nsz arcp contract float %1457, %1460, !spirv.Decorations !898
  %1462 = fadd reassoc nsz arcp contract float %1461, %.sroa.162.1, !spirv.Decorations !898
  br label %._crit_edge.2.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.8:                                  ; preds = %._crit_edge.1.8.._crit_edge.2.8_crit_edge, %1437
  %.sroa.162.2 = phi float [ %1462, %1437 ], [ %.sroa.162.1, %._crit_edge.1.8.._crit_edge.2.8_crit_edge ]
  br i1 %233, label %1463, label %._crit_edge.2.8..preheader.8_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.8..preheader.8_crit_edge:           ; preds = %._crit_edge.2.8
  br label %.preheader.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1463:                                             ; preds = %._crit_edge.2.8
  %.sroa.256.0.insert.ext758 = zext i32 %552 to i64
  %1464 = shl nuw nsw i64 %.sroa.256.0.insert.ext758, 1
  %1465 = add i64 %551, %1464
  %1466 = inttoptr i64 %1465 to i16 addrspace(4)*
  %1467 = addrspacecast i16 addrspace(4)* %1466 to i16 addrspace(1)*
  %1468 = load i16, i16 addrspace(1)* %1467, align 2
  %1469 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1470 = extractvalue { i32, i32 } %1469, 0
  %1471 = extractvalue { i32, i32 } %1469, 1
  %1472 = insertelement <2 x i32> undef, i32 %1470, i32 0
  %1473 = insertelement <2 x i32> %1472, i32 %1471, i32 1
  %1474 = bitcast <2 x i32> %1473 to i64
  %1475 = shl i64 %1474, 1
  %1476 = add i64 %.in3822, %1475
  %1477 = add i64 %1476, %sink_3842
  %1478 = inttoptr i64 %1477 to i16 addrspace(4)*
  %1479 = addrspacecast i16 addrspace(4)* %1478 to i16 addrspace(1)*
  %1480 = load i16, i16 addrspace(1)* %1479, align 2
  %1481 = zext i16 %1468 to i32
  %1482 = shl nuw i32 %1481, 16, !spirv.Decorations !921
  %1483 = bitcast i32 %1482 to float
  %1484 = zext i16 %1480 to i32
  %1485 = shl nuw i32 %1484, 16, !spirv.Decorations !921
  %1486 = bitcast i32 %1485 to float
  %1487 = fmul reassoc nsz arcp contract float %1483, %1486, !spirv.Decorations !898
  %1488 = fadd reassoc nsz arcp contract float %1487, %.sroa.226.1, !spirv.Decorations !898
  br label %.preheader.8, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.8:                                     ; preds = %._crit_edge.2.8..preheader.8_crit_edge, %1463
  %.sroa.226.2 = phi float [ %1488, %1463 ], [ %.sroa.226.1, %._crit_edge.2.8..preheader.8_crit_edge ]
  %sink_3840 = shl nsw i64 %358, 1
  br i1 %237, label %1489, label %.preheader.8.._crit_edge.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.8.._crit_edge.9_crit_edge:             ; preds = %.preheader.8
  br label %._crit_edge.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1489:                                             ; preds = %.preheader.8
  %.sroa.256.0.insert.ext763 = zext i32 %552 to i64
  %1490 = shl nuw nsw i64 %.sroa.256.0.insert.ext763, 1
  %1491 = add i64 %548, %1490
  %1492 = inttoptr i64 %1491 to i16 addrspace(4)*
  %1493 = addrspacecast i16 addrspace(4)* %1492 to i16 addrspace(1)*
  %1494 = load i16, i16 addrspace(1)* %1493, align 2
  %1495 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1496 = extractvalue { i32, i32 } %1495, 0
  %1497 = extractvalue { i32, i32 } %1495, 1
  %1498 = insertelement <2 x i32> undef, i32 %1496, i32 0
  %1499 = insertelement <2 x i32> %1498, i32 %1497, i32 1
  %1500 = bitcast <2 x i32> %1499 to i64
  %1501 = shl i64 %1500, 1
  %1502 = add i64 %.in3822, %1501
  %1503 = add i64 %1502, %sink_3840
  %1504 = inttoptr i64 %1503 to i16 addrspace(4)*
  %1505 = addrspacecast i16 addrspace(4)* %1504 to i16 addrspace(1)*
  %1506 = load i16, i16 addrspace(1)* %1505, align 2
  %1507 = zext i16 %1494 to i32
  %1508 = shl nuw i32 %1507, 16, !spirv.Decorations !921
  %1509 = bitcast i32 %1508 to float
  %1510 = zext i16 %1506 to i32
  %1511 = shl nuw i32 %1510, 16, !spirv.Decorations !921
  %1512 = bitcast i32 %1511 to float
  %1513 = fmul reassoc nsz arcp contract float %1509, %1512, !spirv.Decorations !898
  %1514 = fadd reassoc nsz arcp contract float %1513, %.sroa.38.1, !spirv.Decorations !898
  br label %._crit_edge.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.9:                                    ; preds = %.preheader.8.._crit_edge.9_crit_edge, %1489
  %.sroa.38.2 = phi float [ %1514, %1489 ], [ %.sroa.38.1, %.preheader.8.._crit_edge.9_crit_edge ]
  br i1 %240, label %1515, label %._crit_edge.9.._crit_edge.1.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.9.._crit_edge.1.9_crit_edge:          ; preds = %._crit_edge.9
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1515:                                             ; preds = %._crit_edge.9
  %.sroa.256.0.insert.ext768 = zext i32 %552 to i64
  %1516 = shl nuw nsw i64 %.sroa.256.0.insert.ext768, 1
  %1517 = add i64 %549, %1516
  %1518 = inttoptr i64 %1517 to i16 addrspace(4)*
  %1519 = addrspacecast i16 addrspace(4)* %1518 to i16 addrspace(1)*
  %1520 = load i16, i16 addrspace(1)* %1519, align 2
  %1521 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1522 = extractvalue { i32, i32 } %1521, 0
  %1523 = extractvalue { i32, i32 } %1521, 1
  %1524 = insertelement <2 x i32> undef, i32 %1522, i32 0
  %1525 = insertelement <2 x i32> %1524, i32 %1523, i32 1
  %1526 = bitcast <2 x i32> %1525 to i64
  %1527 = shl i64 %1526, 1
  %1528 = add i64 %.in3822, %1527
  %1529 = add i64 %1528, %sink_3840
  %1530 = inttoptr i64 %1529 to i16 addrspace(4)*
  %1531 = addrspacecast i16 addrspace(4)* %1530 to i16 addrspace(1)*
  %1532 = load i16, i16 addrspace(1)* %1531, align 2
  %1533 = zext i16 %1520 to i32
  %1534 = shl nuw i32 %1533, 16, !spirv.Decorations !921
  %1535 = bitcast i32 %1534 to float
  %1536 = zext i16 %1532 to i32
  %1537 = shl nuw i32 %1536, 16, !spirv.Decorations !921
  %1538 = bitcast i32 %1537 to float
  %1539 = fmul reassoc nsz arcp contract float %1535, %1538, !spirv.Decorations !898
  %1540 = fadd reassoc nsz arcp contract float %1539, %.sroa.102.1, !spirv.Decorations !898
  br label %._crit_edge.1.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.9:                                  ; preds = %._crit_edge.9.._crit_edge.1.9_crit_edge, %1515
  %.sroa.102.2 = phi float [ %1540, %1515 ], [ %.sroa.102.1, %._crit_edge.9.._crit_edge.1.9_crit_edge ]
  br i1 %243, label %1541, label %._crit_edge.1.9.._crit_edge.2.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.9.._crit_edge.2.9_crit_edge:        ; preds = %._crit_edge.1.9
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1541:                                             ; preds = %._crit_edge.1.9
  %.sroa.256.0.insert.ext773 = zext i32 %552 to i64
  %1542 = shl nuw nsw i64 %.sroa.256.0.insert.ext773, 1
  %1543 = add i64 %550, %1542
  %1544 = inttoptr i64 %1543 to i16 addrspace(4)*
  %1545 = addrspacecast i16 addrspace(4)* %1544 to i16 addrspace(1)*
  %1546 = load i16, i16 addrspace(1)* %1545, align 2
  %1547 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1548 = extractvalue { i32, i32 } %1547, 0
  %1549 = extractvalue { i32, i32 } %1547, 1
  %1550 = insertelement <2 x i32> undef, i32 %1548, i32 0
  %1551 = insertelement <2 x i32> %1550, i32 %1549, i32 1
  %1552 = bitcast <2 x i32> %1551 to i64
  %1553 = shl i64 %1552, 1
  %1554 = add i64 %.in3822, %1553
  %1555 = add i64 %1554, %sink_3840
  %1556 = inttoptr i64 %1555 to i16 addrspace(4)*
  %1557 = addrspacecast i16 addrspace(4)* %1556 to i16 addrspace(1)*
  %1558 = load i16, i16 addrspace(1)* %1557, align 2
  %1559 = zext i16 %1546 to i32
  %1560 = shl nuw i32 %1559, 16, !spirv.Decorations !921
  %1561 = bitcast i32 %1560 to float
  %1562 = zext i16 %1558 to i32
  %1563 = shl nuw i32 %1562, 16, !spirv.Decorations !921
  %1564 = bitcast i32 %1563 to float
  %1565 = fmul reassoc nsz arcp contract float %1561, %1564, !spirv.Decorations !898
  %1566 = fadd reassoc nsz arcp contract float %1565, %.sroa.166.1, !spirv.Decorations !898
  br label %._crit_edge.2.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.9:                                  ; preds = %._crit_edge.1.9.._crit_edge.2.9_crit_edge, %1541
  %.sroa.166.2 = phi float [ %1566, %1541 ], [ %.sroa.166.1, %._crit_edge.1.9.._crit_edge.2.9_crit_edge ]
  br i1 %246, label %1567, label %._crit_edge.2.9..preheader.9_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.9..preheader.9_crit_edge:           ; preds = %._crit_edge.2.9
  br label %.preheader.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1567:                                             ; preds = %._crit_edge.2.9
  %.sroa.256.0.insert.ext778 = zext i32 %552 to i64
  %1568 = shl nuw nsw i64 %.sroa.256.0.insert.ext778, 1
  %1569 = add i64 %551, %1568
  %1570 = inttoptr i64 %1569 to i16 addrspace(4)*
  %1571 = addrspacecast i16 addrspace(4)* %1570 to i16 addrspace(1)*
  %1572 = load i16, i16 addrspace(1)* %1571, align 2
  %1573 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1574 = extractvalue { i32, i32 } %1573, 0
  %1575 = extractvalue { i32, i32 } %1573, 1
  %1576 = insertelement <2 x i32> undef, i32 %1574, i32 0
  %1577 = insertelement <2 x i32> %1576, i32 %1575, i32 1
  %1578 = bitcast <2 x i32> %1577 to i64
  %1579 = shl i64 %1578, 1
  %1580 = add i64 %.in3822, %1579
  %1581 = add i64 %1580, %sink_3840
  %1582 = inttoptr i64 %1581 to i16 addrspace(4)*
  %1583 = addrspacecast i16 addrspace(4)* %1582 to i16 addrspace(1)*
  %1584 = load i16, i16 addrspace(1)* %1583, align 2
  %1585 = zext i16 %1572 to i32
  %1586 = shl nuw i32 %1585, 16, !spirv.Decorations !921
  %1587 = bitcast i32 %1586 to float
  %1588 = zext i16 %1584 to i32
  %1589 = shl nuw i32 %1588, 16, !spirv.Decorations !921
  %1590 = bitcast i32 %1589 to float
  %1591 = fmul reassoc nsz arcp contract float %1587, %1590, !spirv.Decorations !898
  %1592 = fadd reassoc nsz arcp contract float %1591, %.sroa.230.1, !spirv.Decorations !898
  br label %.preheader.9, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.9:                                     ; preds = %._crit_edge.2.9..preheader.9_crit_edge, %1567
  %.sroa.230.2 = phi float [ %1592, %1567 ], [ %.sroa.230.1, %._crit_edge.2.9..preheader.9_crit_edge ]
  %sink_3838 = shl nsw i64 %359, 1
  br i1 %250, label %1593, label %.preheader.9.._crit_edge.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.9.._crit_edge.10_crit_edge:            ; preds = %.preheader.9
  br label %._crit_edge.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1593:                                             ; preds = %.preheader.9
  %.sroa.256.0.insert.ext783 = zext i32 %552 to i64
  %1594 = shl nuw nsw i64 %.sroa.256.0.insert.ext783, 1
  %1595 = add i64 %548, %1594
  %1596 = inttoptr i64 %1595 to i16 addrspace(4)*
  %1597 = addrspacecast i16 addrspace(4)* %1596 to i16 addrspace(1)*
  %1598 = load i16, i16 addrspace(1)* %1597, align 2
  %1599 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1600 = extractvalue { i32, i32 } %1599, 0
  %1601 = extractvalue { i32, i32 } %1599, 1
  %1602 = insertelement <2 x i32> undef, i32 %1600, i32 0
  %1603 = insertelement <2 x i32> %1602, i32 %1601, i32 1
  %1604 = bitcast <2 x i32> %1603 to i64
  %1605 = shl i64 %1604, 1
  %1606 = add i64 %.in3822, %1605
  %1607 = add i64 %1606, %sink_3838
  %1608 = inttoptr i64 %1607 to i16 addrspace(4)*
  %1609 = addrspacecast i16 addrspace(4)* %1608 to i16 addrspace(1)*
  %1610 = load i16, i16 addrspace(1)* %1609, align 2
  %1611 = zext i16 %1598 to i32
  %1612 = shl nuw i32 %1611, 16, !spirv.Decorations !921
  %1613 = bitcast i32 %1612 to float
  %1614 = zext i16 %1610 to i32
  %1615 = shl nuw i32 %1614, 16, !spirv.Decorations !921
  %1616 = bitcast i32 %1615 to float
  %1617 = fmul reassoc nsz arcp contract float %1613, %1616, !spirv.Decorations !898
  %1618 = fadd reassoc nsz arcp contract float %1617, %.sroa.42.1, !spirv.Decorations !898
  br label %._crit_edge.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.10:                                   ; preds = %.preheader.9.._crit_edge.10_crit_edge, %1593
  %.sroa.42.2 = phi float [ %1618, %1593 ], [ %.sroa.42.1, %.preheader.9.._crit_edge.10_crit_edge ]
  br i1 %253, label %1619, label %._crit_edge.10.._crit_edge.1.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.10.._crit_edge.1.10_crit_edge:        ; preds = %._crit_edge.10
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1619:                                             ; preds = %._crit_edge.10
  %.sroa.256.0.insert.ext788 = zext i32 %552 to i64
  %1620 = shl nuw nsw i64 %.sroa.256.0.insert.ext788, 1
  %1621 = add i64 %549, %1620
  %1622 = inttoptr i64 %1621 to i16 addrspace(4)*
  %1623 = addrspacecast i16 addrspace(4)* %1622 to i16 addrspace(1)*
  %1624 = load i16, i16 addrspace(1)* %1623, align 2
  %1625 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1626 = extractvalue { i32, i32 } %1625, 0
  %1627 = extractvalue { i32, i32 } %1625, 1
  %1628 = insertelement <2 x i32> undef, i32 %1626, i32 0
  %1629 = insertelement <2 x i32> %1628, i32 %1627, i32 1
  %1630 = bitcast <2 x i32> %1629 to i64
  %1631 = shl i64 %1630, 1
  %1632 = add i64 %.in3822, %1631
  %1633 = add i64 %1632, %sink_3838
  %1634 = inttoptr i64 %1633 to i16 addrspace(4)*
  %1635 = addrspacecast i16 addrspace(4)* %1634 to i16 addrspace(1)*
  %1636 = load i16, i16 addrspace(1)* %1635, align 2
  %1637 = zext i16 %1624 to i32
  %1638 = shl nuw i32 %1637, 16, !spirv.Decorations !921
  %1639 = bitcast i32 %1638 to float
  %1640 = zext i16 %1636 to i32
  %1641 = shl nuw i32 %1640, 16, !spirv.Decorations !921
  %1642 = bitcast i32 %1641 to float
  %1643 = fmul reassoc nsz arcp contract float %1639, %1642, !spirv.Decorations !898
  %1644 = fadd reassoc nsz arcp contract float %1643, %.sroa.106.1, !spirv.Decorations !898
  br label %._crit_edge.1.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.10:                                 ; preds = %._crit_edge.10.._crit_edge.1.10_crit_edge, %1619
  %.sroa.106.2 = phi float [ %1644, %1619 ], [ %.sroa.106.1, %._crit_edge.10.._crit_edge.1.10_crit_edge ]
  br i1 %256, label %1645, label %._crit_edge.1.10.._crit_edge.2.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.10.._crit_edge.2.10_crit_edge:      ; preds = %._crit_edge.1.10
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1645:                                             ; preds = %._crit_edge.1.10
  %.sroa.256.0.insert.ext793 = zext i32 %552 to i64
  %1646 = shl nuw nsw i64 %.sroa.256.0.insert.ext793, 1
  %1647 = add i64 %550, %1646
  %1648 = inttoptr i64 %1647 to i16 addrspace(4)*
  %1649 = addrspacecast i16 addrspace(4)* %1648 to i16 addrspace(1)*
  %1650 = load i16, i16 addrspace(1)* %1649, align 2
  %1651 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1652 = extractvalue { i32, i32 } %1651, 0
  %1653 = extractvalue { i32, i32 } %1651, 1
  %1654 = insertelement <2 x i32> undef, i32 %1652, i32 0
  %1655 = insertelement <2 x i32> %1654, i32 %1653, i32 1
  %1656 = bitcast <2 x i32> %1655 to i64
  %1657 = shl i64 %1656, 1
  %1658 = add i64 %.in3822, %1657
  %1659 = add i64 %1658, %sink_3838
  %1660 = inttoptr i64 %1659 to i16 addrspace(4)*
  %1661 = addrspacecast i16 addrspace(4)* %1660 to i16 addrspace(1)*
  %1662 = load i16, i16 addrspace(1)* %1661, align 2
  %1663 = zext i16 %1650 to i32
  %1664 = shl nuw i32 %1663, 16, !spirv.Decorations !921
  %1665 = bitcast i32 %1664 to float
  %1666 = zext i16 %1662 to i32
  %1667 = shl nuw i32 %1666, 16, !spirv.Decorations !921
  %1668 = bitcast i32 %1667 to float
  %1669 = fmul reassoc nsz arcp contract float %1665, %1668, !spirv.Decorations !898
  %1670 = fadd reassoc nsz arcp contract float %1669, %.sroa.170.1, !spirv.Decorations !898
  br label %._crit_edge.2.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.10:                                 ; preds = %._crit_edge.1.10.._crit_edge.2.10_crit_edge, %1645
  %.sroa.170.2 = phi float [ %1670, %1645 ], [ %.sroa.170.1, %._crit_edge.1.10.._crit_edge.2.10_crit_edge ]
  br i1 %259, label %1671, label %._crit_edge.2.10..preheader.10_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.10..preheader.10_crit_edge:         ; preds = %._crit_edge.2.10
  br label %.preheader.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1671:                                             ; preds = %._crit_edge.2.10
  %.sroa.256.0.insert.ext798 = zext i32 %552 to i64
  %1672 = shl nuw nsw i64 %.sroa.256.0.insert.ext798, 1
  %1673 = add i64 %551, %1672
  %1674 = inttoptr i64 %1673 to i16 addrspace(4)*
  %1675 = addrspacecast i16 addrspace(4)* %1674 to i16 addrspace(1)*
  %1676 = load i16, i16 addrspace(1)* %1675, align 2
  %1677 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1678 = extractvalue { i32, i32 } %1677, 0
  %1679 = extractvalue { i32, i32 } %1677, 1
  %1680 = insertelement <2 x i32> undef, i32 %1678, i32 0
  %1681 = insertelement <2 x i32> %1680, i32 %1679, i32 1
  %1682 = bitcast <2 x i32> %1681 to i64
  %1683 = shl i64 %1682, 1
  %1684 = add i64 %.in3822, %1683
  %1685 = add i64 %1684, %sink_3838
  %1686 = inttoptr i64 %1685 to i16 addrspace(4)*
  %1687 = addrspacecast i16 addrspace(4)* %1686 to i16 addrspace(1)*
  %1688 = load i16, i16 addrspace(1)* %1687, align 2
  %1689 = zext i16 %1676 to i32
  %1690 = shl nuw i32 %1689, 16, !spirv.Decorations !921
  %1691 = bitcast i32 %1690 to float
  %1692 = zext i16 %1688 to i32
  %1693 = shl nuw i32 %1692, 16, !spirv.Decorations !921
  %1694 = bitcast i32 %1693 to float
  %1695 = fmul reassoc nsz arcp contract float %1691, %1694, !spirv.Decorations !898
  %1696 = fadd reassoc nsz arcp contract float %1695, %.sroa.234.1, !spirv.Decorations !898
  br label %.preheader.10, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.10:                                    ; preds = %._crit_edge.2.10..preheader.10_crit_edge, %1671
  %.sroa.234.2 = phi float [ %1696, %1671 ], [ %.sroa.234.1, %._crit_edge.2.10..preheader.10_crit_edge ]
  %sink_3836 = shl nsw i64 %360, 1
  br i1 %263, label %1697, label %.preheader.10.._crit_edge.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.10.._crit_edge.11_crit_edge:           ; preds = %.preheader.10
  br label %._crit_edge.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1697:                                             ; preds = %.preheader.10
  %.sroa.256.0.insert.ext803 = zext i32 %552 to i64
  %1698 = shl nuw nsw i64 %.sroa.256.0.insert.ext803, 1
  %1699 = add i64 %548, %1698
  %1700 = inttoptr i64 %1699 to i16 addrspace(4)*
  %1701 = addrspacecast i16 addrspace(4)* %1700 to i16 addrspace(1)*
  %1702 = load i16, i16 addrspace(1)* %1701, align 2
  %1703 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1704 = extractvalue { i32, i32 } %1703, 0
  %1705 = extractvalue { i32, i32 } %1703, 1
  %1706 = insertelement <2 x i32> undef, i32 %1704, i32 0
  %1707 = insertelement <2 x i32> %1706, i32 %1705, i32 1
  %1708 = bitcast <2 x i32> %1707 to i64
  %1709 = shl i64 %1708, 1
  %1710 = add i64 %.in3822, %1709
  %1711 = add i64 %1710, %sink_3836
  %1712 = inttoptr i64 %1711 to i16 addrspace(4)*
  %1713 = addrspacecast i16 addrspace(4)* %1712 to i16 addrspace(1)*
  %1714 = load i16, i16 addrspace(1)* %1713, align 2
  %1715 = zext i16 %1702 to i32
  %1716 = shl nuw i32 %1715, 16, !spirv.Decorations !921
  %1717 = bitcast i32 %1716 to float
  %1718 = zext i16 %1714 to i32
  %1719 = shl nuw i32 %1718, 16, !spirv.Decorations !921
  %1720 = bitcast i32 %1719 to float
  %1721 = fmul reassoc nsz arcp contract float %1717, %1720, !spirv.Decorations !898
  %1722 = fadd reassoc nsz arcp contract float %1721, %.sroa.46.1, !spirv.Decorations !898
  br label %._crit_edge.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.11:                                   ; preds = %.preheader.10.._crit_edge.11_crit_edge, %1697
  %.sroa.46.2 = phi float [ %1722, %1697 ], [ %.sroa.46.1, %.preheader.10.._crit_edge.11_crit_edge ]
  br i1 %266, label %1723, label %._crit_edge.11.._crit_edge.1.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.11.._crit_edge.1.11_crit_edge:        ; preds = %._crit_edge.11
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1723:                                             ; preds = %._crit_edge.11
  %.sroa.256.0.insert.ext808 = zext i32 %552 to i64
  %1724 = shl nuw nsw i64 %.sroa.256.0.insert.ext808, 1
  %1725 = add i64 %549, %1724
  %1726 = inttoptr i64 %1725 to i16 addrspace(4)*
  %1727 = addrspacecast i16 addrspace(4)* %1726 to i16 addrspace(1)*
  %1728 = load i16, i16 addrspace(1)* %1727, align 2
  %1729 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1730 = extractvalue { i32, i32 } %1729, 0
  %1731 = extractvalue { i32, i32 } %1729, 1
  %1732 = insertelement <2 x i32> undef, i32 %1730, i32 0
  %1733 = insertelement <2 x i32> %1732, i32 %1731, i32 1
  %1734 = bitcast <2 x i32> %1733 to i64
  %1735 = shl i64 %1734, 1
  %1736 = add i64 %.in3822, %1735
  %1737 = add i64 %1736, %sink_3836
  %1738 = inttoptr i64 %1737 to i16 addrspace(4)*
  %1739 = addrspacecast i16 addrspace(4)* %1738 to i16 addrspace(1)*
  %1740 = load i16, i16 addrspace(1)* %1739, align 2
  %1741 = zext i16 %1728 to i32
  %1742 = shl nuw i32 %1741, 16, !spirv.Decorations !921
  %1743 = bitcast i32 %1742 to float
  %1744 = zext i16 %1740 to i32
  %1745 = shl nuw i32 %1744, 16, !spirv.Decorations !921
  %1746 = bitcast i32 %1745 to float
  %1747 = fmul reassoc nsz arcp contract float %1743, %1746, !spirv.Decorations !898
  %1748 = fadd reassoc nsz arcp contract float %1747, %.sroa.110.1, !spirv.Decorations !898
  br label %._crit_edge.1.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.11:                                 ; preds = %._crit_edge.11.._crit_edge.1.11_crit_edge, %1723
  %.sroa.110.2 = phi float [ %1748, %1723 ], [ %.sroa.110.1, %._crit_edge.11.._crit_edge.1.11_crit_edge ]
  br i1 %269, label %1749, label %._crit_edge.1.11.._crit_edge.2.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.11.._crit_edge.2.11_crit_edge:      ; preds = %._crit_edge.1.11
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1749:                                             ; preds = %._crit_edge.1.11
  %.sroa.256.0.insert.ext813 = zext i32 %552 to i64
  %1750 = shl nuw nsw i64 %.sroa.256.0.insert.ext813, 1
  %1751 = add i64 %550, %1750
  %1752 = inttoptr i64 %1751 to i16 addrspace(4)*
  %1753 = addrspacecast i16 addrspace(4)* %1752 to i16 addrspace(1)*
  %1754 = load i16, i16 addrspace(1)* %1753, align 2
  %1755 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1756 = extractvalue { i32, i32 } %1755, 0
  %1757 = extractvalue { i32, i32 } %1755, 1
  %1758 = insertelement <2 x i32> undef, i32 %1756, i32 0
  %1759 = insertelement <2 x i32> %1758, i32 %1757, i32 1
  %1760 = bitcast <2 x i32> %1759 to i64
  %1761 = shl i64 %1760, 1
  %1762 = add i64 %.in3822, %1761
  %1763 = add i64 %1762, %sink_3836
  %1764 = inttoptr i64 %1763 to i16 addrspace(4)*
  %1765 = addrspacecast i16 addrspace(4)* %1764 to i16 addrspace(1)*
  %1766 = load i16, i16 addrspace(1)* %1765, align 2
  %1767 = zext i16 %1754 to i32
  %1768 = shl nuw i32 %1767, 16, !spirv.Decorations !921
  %1769 = bitcast i32 %1768 to float
  %1770 = zext i16 %1766 to i32
  %1771 = shl nuw i32 %1770, 16, !spirv.Decorations !921
  %1772 = bitcast i32 %1771 to float
  %1773 = fmul reassoc nsz arcp contract float %1769, %1772, !spirv.Decorations !898
  %1774 = fadd reassoc nsz arcp contract float %1773, %.sroa.174.1, !spirv.Decorations !898
  br label %._crit_edge.2.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.11:                                 ; preds = %._crit_edge.1.11.._crit_edge.2.11_crit_edge, %1749
  %.sroa.174.2 = phi float [ %1774, %1749 ], [ %.sroa.174.1, %._crit_edge.1.11.._crit_edge.2.11_crit_edge ]
  br i1 %272, label %1775, label %._crit_edge.2.11..preheader.11_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.11..preheader.11_crit_edge:         ; preds = %._crit_edge.2.11
  br label %.preheader.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1775:                                             ; preds = %._crit_edge.2.11
  %.sroa.256.0.insert.ext818 = zext i32 %552 to i64
  %1776 = shl nuw nsw i64 %.sroa.256.0.insert.ext818, 1
  %1777 = add i64 %551, %1776
  %1778 = inttoptr i64 %1777 to i16 addrspace(4)*
  %1779 = addrspacecast i16 addrspace(4)* %1778 to i16 addrspace(1)*
  %1780 = load i16, i16 addrspace(1)* %1779, align 2
  %1781 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1782 = extractvalue { i32, i32 } %1781, 0
  %1783 = extractvalue { i32, i32 } %1781, 1
  %1784 = insertelement <2 x i32> undef, i32 %1782, i32 0
  %1785 = insertelement <2 x i32> %1784, i32 %1783, i32 1
  %1786 = bitcast <2 x i32> %1785 to i64
  %1787 = shl i64 %1786, 1
  %1788 = add i64 %.in3822, %1787
  %1789 = add i64 %1788, %sink_3836
  %1790 = inttoptr i64 %1789 to i16 addrspace(4)*
  %1791 = addrspacecast i16 addrspace(4)* %1790 to i16 addrspace(1)*
  %1792 = load i16, i16 addrspace(1)* %1791, align 2
  %1793 = zext i16 %1780 to i32
  %1794 = shl nuw i32 %1793, 16, !spirv.Decorations !921
  %1795 = bitcast i32 %1794 to float
  %1796 = zext i16 %1792 to i32
  %1797 = shl nuw i32 %1796, 16, !spirv.Decorations !921
  %1798 = bitcast i32 %1797 to float
  %1799 = fmul reassoc nsz arcp contract float %1795, %1798, !spirv.Decorations !898
  %1800 = fadd reassoc nsz arcp contract float %1799, %.sroa.238.1, !spirv.Decorations !898
  br label %.preheader.11, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.11:                                    ; preds = %._crit_edge.2.11..preheader.11_crit_edge, %1775
  %.sroa.238.2 = phi float [ %1800, %1775 ], [ %.sroa.238.1, %._crit_edge.2.11..preheader.11_crit_edge ]
  %sink_3834 = shl nsw i64 %361, 1
  br i1 %276, label %1801, label %.preheader.11.._crit_edge.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.11.._crit_edge.12_crit_edge:           ; preds = %.preheader.11
  br label %._crit_edge.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1801:                                             ; preds = %.preheader.11
  %.sroa.256.0.insert.ext823 = zext i32 %552 to i64
  %1802 = shl nuw nsw i64 %.sroa.256.0.insert.ext823, 1
  %1803 = add i64 %548, %1802
  %1804 = inttoptr i64 %1803 to i16 addrspace(4)*
  %1805 = addrspacecast i16 addrspace(4)* %1804 to i16 addrspace(1)*
  %1806 = load i16, i16 addrspace(1)* %1805, align 2
  %1807 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1808 = extractvalue { i32, i32 } %1807, 0
  %1809 = extractvalue { i32, i32 } %1807, 1
  %1810 = insertelement <2 x i32> undef, i32 %1808, i32 0
  %1811 = insertelement <2 x i32> %1810, i32 %1809, i32 1
  %1812 = bitcast <2 x i32> %1811 to i64
  %1813 = shl i64 %1812, 1
  %1814 = add i64 %.in3822, %1813
  %1815 = add i64 %1814, %sink_3834
  %1816 = inttoptr i64 %1815 to i16 addrspace(4)*
  %1817 = addrspacecast i16 addrspace(4)* %1816 to i16 addrspace(1)*
  %1818 = load i16, i16 addrspace(1)* %1817, align 2
  %1819 = zext i16 %1806 to i32
  %1820 = shl nuw i32 %1819, 16, !spirv.Decorations !921
  %1821 = bitcast i32 %1820 to float
  %1822 = zext i16 %1818 to i32
  %1823 = shl nuw i32 %1822, 16, !spirv.Decorations !921
  %1824 = bitcast i32 %1823 to float
  %1825 = fmul reassoc nsz arcp contract float %1821, %1824, !spirv.Decorations !898
  %1826 = fadd reassoc nsz arcp contract float %1825, %.sroa.50.1, !spirv.Decorations !898
  br label %._crit_edge.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.12:                                   ; preds = %.preheader.11.._crit_edge.12_crit_edge, %1801
  %.sroa.50.2 = phi float [ %1826, %1801 ], [ %.sroa.50.1, %.preheader.11.._crit_edge.12_crit_edge ]
  br i1 %279, label %1827, label %._crit_edge.12.._crit_edge.1.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.12.._crit_edge.1.12_crit_edge:        ; preds = %._crit_edge.12
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1827:                                             ; preds = %._crit_edge.12
  %.sroa.256.0.insert.ext828 = zext i32 %552 to i64
  %1828 = shl nuw nsw i64 %.sroa.256.0.insert.ext828, 1
  %1829 = add i64 %549, %1828
  %1830 = inttoptr i64 %1829 to i16 addrspace(4)*
  %1831 = addrspacecast i16 addrspace(4)* %1830 to i16 addrspace(1)*
  %1832 = load i16, i16 addrspace(1)* %1831, align 2
  %1833 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1834 = extractvalue { i32, i32 } %1833, 0
  %1835 = extractvalue { i32, i32 } %1833, 1
  %1836 = insertelement <2 x i32> undef, i32 %1834, i32 0
  %1837 = insertelement <2 x i32> %1836, i32 %1835, i32 1
  %1838 = bitcast <2 x i32> %1837 to i64
  %1839 = shl i64 %1838, 1
  %1840 = add i64 %.in3822, %1839
  %1841 = add i64 %1840, %sink_3834
  %1842 = inttoptr i64 %1841 to i16 addrspace(4)*
  %1843 = addrspacecast i16 addrspace(4)* %1842 to i16 addrspace(1)*
  %1844 = load i16, i16 addrspace(1)* %1843, align 2
  %1845 = zext i16 %1832 to i32
  %1846 = shl nuw i32 %1845, 16, !spirv.Decorations !921
  %1847 = bitcast i32 %1846 to float
  %1848 = zext i16 %1844 to i32
  %1849 = shl nuw i32 %1848, 16, !spirv.Decorations !921
  %1850 = bitcast i32 %1849 to float
  %1851 = fmul reassoc nsz arcp contract float %1847, %1850, !spirv.Decorations !898
  %1852 = fadd reassoc nsz arcp contract float %1851, %.sroa.114.1, !spirv.Decorations !898
  br label %._crit_edge.1.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.12:                                 ; preds = %._crit_edge.12.._crit_edge.1.12_crit_edge, %1827
  %.sroa.114.2 = phi float [ %1852, %1827 ], [ %.sroa.114.1, %._crit_edge.12.._crit_edge.1.12_crit_edge ]
  br i1 %282, label %1853, label %._crit_edge.1.12.._crit_edge.2.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.12.._crit_edge.2.12_crit_edge:      ; preds = %._crit_edge.1.12
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1853:                                             ; preds = %._crit_edge.1.12
  %.sroa.256.0.insert.ext833 = zext i32 %552 to i64
  %1854 = shl nuw nsw i64 %.sroa.256.0.insert.ext833, 1
  %1855 = add i64 %550, %1854
  %1856 = inttoptr i64 %1855 to i16 addrspace(4)*
  %1857 = addrspacecast i16 addrspace(4)* %1856 to i16 addrspace(1)*
  %1858 = load i16, i16 addrspace(1)* %1857, align 2
  %1859 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1860 = extractvalue { i32, i32 } %1859, 0
  %1861 = extractvalue { i32, i32 } %1859, 1
  %1862 = insertelement <2 x i32> undef, i32 %1860, i32 0
  %1863 = insertelement <2 x i32> %1862, i32 %1861, i32 1
  %1864 = bitcast <2 x i32> %1863 to i64
  %1865 = shl i64 %1864, 1
  %1866 = add i64 %.in3822, %1865
  %1867 = add i64 %1866, %sink_3834
  %1868 = inttoptr i64 %1867 to i16 addrspace(4)*
  %1869 = addrspacecast i16 addrspace(4)* %1868 to i16 addrspace(1)*
  %1870 = load i16, i16 addrspace(1)* %1869, align 2
  %1871 = zext i16 %1858 to i32
  %1872 = shl nuw i32 %1871, 16, !spirv.Decorations !921
  %1873 = bitcast i32 %1872 to float
  %1874 = zext i16 %1870 to i32
  %1875 = shl nuw i32 %1874, 16, !spirv.Decorations !921
  %1876 = bitcast i32 %1875 to float
  %1877 = fmul reassoc nsz arcp contract float %1873, %1876, !spirv.Decorations !898
  %1878 = fadd reassoc nsz arcp contract float %1877, %.sroa.178.1, !spirv.Decorations !898
  br label %._crit_edge.2.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.12:                                 ; preds = %._crit_edge.1.12.._crit_edge.2.12_crit_edge, %1853
  %.sroa.178.2 = phi float [ %1878, %1853 ], [ %.sroa.178.1, %._crit_edge.1.12.._crit_edge.2.12_crit_edge ]
  br i1 %285, label %1879, label %._crit_edge.2.12..preheader.12_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.12..preheader.12_crit_edge:         ; preds = %._crit_edge.2.12
  br label %.preheader.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1879:                                             ; preds = %._crit_edge.2.12
  %.sroa.256.0.insert.ext838 = zext i32 %552 to i64
  %1880 = shl nuw nsw i64 %.sroa.256.0.insert.ext838, 1
  %1881 = add i64 %551, %1880
  %1882 = inttoptr i64 %1881 to i16 addrspace(4)*
  %1883 = addrspacecast i16 addrspace(4)* %1882 to i16 addrspace(1)*
  %1884 = load i16, i16 addrspace(1)* %1883, align 2
  %1885 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1886 = extractvalue { i32, i32 } %1885, 0
  %1887 = extractvalue { i32, i32 } %1885, 1
  %1888 = insertelement <2 x i32> undef, i32 %1886, i32 0
  %1889 = insertelement <2 x i32> %1888, i32 %1887, i32 1
  %1890 = bitcast <2 x i32> %1889 to i64
  %1891 = shl i64 %1890, 1
  %1892 = add i64 %.in3822, %1891
  %1893 = add i64 %1892, %sink_3834
  %1894 = inttoptr i64 %1893 to i16 addrspace(4)*
  %1895 = addrspacecast i16 addrspace(4)* %1894 to i16 addrspace(1)*
  %1896 = load i16, i16 addrspace(1)* %1895, align 2
  %1897 = zext i16 %1884 to i32
  %1898 = shl nuw i32 %1897, 16, !spirv.Decorations !921
  %1899 = bitcast i32 %1898 to float
  %1900 = zext i16 %1896 to i32
  %1901 = shl nuw i32 %1900, 16, !spirv.Decorations !921
  %1902 = bitcast i32 %1901 to float
  %1903 = fmul reassoc nsz arcp contract float %1899, %1902, !spirv.Decorations !898
  %1904 = fadd reassoc nsz arcp contract float %1903, %.sroa.242.1, !spirv.Decorations !898
  br label %.preheader.12, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.12:                                    ; preds = %._crit_edge.2.12..preheader.12_crit_edge, %1879
  %.sroa.242.2 = phi float [ %1904, %1879 ], [ %.sroa.242.1, %._crit_edge.2.12..preheader.12_crit_edge ]
  %sink_3832 = shl nsw i64 %362, 1
  br i1 %289, label %1905, label %.preheader.12.._crit_edge.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.12.._crit_edge.13_crit_edge:           ; preds = %.preheader.12
  br label %._crit_edge.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1905:                                             ; preds = %.preheader.12
  %.sroa.256.0.insert.ext843 = zext i32 %552 to i64
  %1906 = shl nuw nsw i64 %.sroa.256.0.insert.ext843, 1
  %1907 = add i64 %548, %1906
  %1908 = inttoptr i64 %1907 to i16 addrspace(4)*
  %1909 = addrspacecast i16 addrspace(4)* %1908 to i16 addrspace(1)*
  %1910 = load i16, i16 addrspace(1)* %1909, align 2
  %1911 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1912 = extractvalue { i32, i32 } %1911, 0
  %1913 = extractvalue { i32, i32 } %1911, 1
  %1914 = insertelement <2 x i32> undef, i32 %1912, i32 0
  %1915 = insertelement <2 x i32> %1914, i32 %1913, i32 1
  %1916 = bitcast <2 x i32> %1915 to i64
  %1917 = shl i64 %1916, 1
  %1918 = add i64 %.in3822, %1917
  %1919 = add i64 %1918, %sink_3832
  %1920 = inttoptr i64 %1919 to i16 addrspace(4)*
  %1921 = addrspacecast i16 addrspace(4)* %1920 to i16 addrspace(1)*
  %1922 = load i16, i16 addrspace(1)* %1921, align 2
  %1923 = zext i16 %1910 to i32
  %1924 = shl nuw i32 %1923, 16, !spirv.Decorations !921
  %1925 = bitcast i32 %1924 to float
  %1926 = zext i16 %1922 to i32
  %1927 = shl nuw i32 %1926, 16, !spirv.Decorations !921
  %1928 = bitcast i32 %1927 to float
  %1929 = fmul reassoc nsz arcp contract float %1925, %1928, !spirv.Decorations !898
  %1930 = fadd reassoc nsz arcp contract float %1929, %.sroa.54.1, !spirv.Decorations !898
  br label %._crit_edge.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.13:                                   ; preds = %.preheader.12.._crit_edge.13_crit_edge, %1905
  %.sroa.54.2 = phi float [ %1930, %1905 ], [ %.sroa.54.1, %.preheader.12.._crit_edge.13_crit_edge ]
  br i1 %292, label %1931, label %._crit_edge.13.._crit_edge.1.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.13.._crit_edge.1.13_crit_edge:        ; preds = %._crit_edge.13
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1931:                                             ; preds = %._crit_edge.13
  %.sroa.256.0.insert.ext848 = zext i32 %552 to i64
  %1932 = shl nuw nsw i64 %.sroa.256.0.insert.ext848, 1
  %1933 = add i64 %549, %1932
  %1934 = inttoptr i64 %1933 to i16 addrspace(4)*
  %1935 = addrspacecast i16 addrspace(4)* %1934 to i16 addrspace(1)*
  %1936 = load i16, i16 addrspace(1)* %1935, align 2
  %1937 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1938 = extractvalue { i32, i32 } %1937, 0
  %1939 = extractvalue { i32, i32 } %1937, 1
  %1940 = insertelement <2 x i32> undef, i32 %1938, i32 0
  %1941 = insertelement <2 x i32> %1940, i32 %1939, i32 1
  %1942 = bitcast <2 x i32> %1941 to i64
  %1943 = shl i64 %1942, 1
  %1944 = add i64 %.in3822, %1943
  %1945 = add i64 %1944, %sink_3832
  %1946 = inttoptr i64 %1945 to i16 addrspace(4)*
  %1947 = addrspacecast i16 addrspace(4)* %1946 to i16 addrspace(1)*
  %1948 = load i16, i16 addrspace(1)* %1947, align 2
  %1949 = zext i16 %1936 to i32
  %1950 = shl nuw i32 %1949, 16, !spirv.Decorations !921
  %1951 = bitcast i32 %1950 to float
  %1952 = zext i16 %1948 to i32
  %1953 = shl nuw i32 %1952, 16, !spirv.Decorations !921
  %1954 = bitcast i32 %1953 to float
  %1955 = fmul reassoc nsz arcp contract float %1951, %1954, !spirv.Decorations !898
  %1956 = fadd reassoc nsz arcp contract float %1955, %.sroa.118.1, !spirv.Decorations !898
  br label %._crit_edge.1.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.13:                                 ; preds = %._crit_edge.13.._crit_edge.1.13_crit_edge, %1931
  %.sroa.118.2 = phi float [ %1956, %1931 ], [ %.sroa.118.1, %._crit_edge.13.._crit_edge.1.13_crit_edge ]
  br i1 %295, label %1957, label %._crit_edge.1.13.._crit_edge.2.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.13.._crit_edge.2.13_crit_edge:      ; preds = %._crit_edge.1.13
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1957:                                             ; preds = %._crit_edge.1.13
  %.sroa.256.0.insert.ext853 = zext i32 %552 to i64
  %1958 = shl nuw nsw i64 %.sroa.256.0.insert.ext853, 1
  %1959 = add i64 %550, %1958
  %1960 = inttoptr i64 %1959 to i16 addrspace(4)*
  %1961 = addrspacecast i16 addrspace(4)* %1960 to i16 addrspace(1)*
  %1962 = load i16, i16 addrspace(1)* %1961, align 2
  %1963 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1964 = extractvalue { i32, i32 } %1963, 0
  %1965 = extractvalue { i32, i32 } %1963, 1
  %1966 = insertelement <2 x i32> undef, i32 %1964, i32 0
  %1967 = insertelement <2 x i32> %1966, i32 %1965, i32 1
  %1968 = bitcast <2 x i32> %1967 to i64
  %1969 = shl i64 %1968, 1
  %1970 = add i64 %.in3822, %1969
  %1971 = add i64 %1970, %sink_3832
  %1972 = inttoptr i64 %1971 to i16 addrspace(4)*
  %1973 = addrspacecast i16 addrspace(4)* %1972 to i16 addrspace(1)*
  %1974 = load i16, i16 addrspace(1)* %1973, align 2
  %1975 = zext i16 %1962 to i32
  %1976 = shl nuw i32 %1975, 16, !spirv.Decorations !921
  %1977 = bitcast i32 %1976 to float
  %1978 = zext i16 %1974 to i32
  %1979 = shl nuw i32 %1978, 16, !spirv.Decorations !921
  %1980 = bitcast i32 %1979 to float
  %1981 = fmul reassoc nsz arcp contract float %1977, %1980, !spirv.Decorations !898
  %1982 = fadd reassoc nsz arcp contract float %1981, %.sroa.182.1, !spirv.Decorations !898
  br label %._crit_edge.2.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.13:                                 ; preds = %._crit_edge.1.13.._crit_edge.2.13_crit_edge, %1957
  %.sroa.182.2 = phi float [ %1982, %1957 ], [ %.sroa.182.1, %._crit_edge.1.13.._crit_edge.2.13_crit_edge ]
  br i1 %298, label %1983, label %._crit_edge.2.13..preheader.13_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.13..preheader.13_crit_edge:         ; preds = %._crit_edge.2.13
  br label %.preheader.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

1983:                                             ; preds = %._crit_edge.2.13
  %.sroa.256.0.insert.ext858 = zext i32 %552 to i64
  %1984 = shl nuw nsw i64 %.sroa.256.0.insert.ext858, 1
  %1985 = add i64 %551, %1984
  %1986 = inttoptr i64 %1985 to i16 addrspace(4)*
  %1987 = addrspacecast i16 addrspace(4)* %1986 to i16 addrspace(1)*
  %1988 = load i16, i16 addrspace(1)* %1987, align 2
  %1989 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %1990 = extractvalue { i32, i32 } %1989, 0
  %1991 = extractvalue { i32, i32 } %1989, 1
  %1992 = insertelement <2 x i32> undef, i32 %1990, i32 0
  %1993 = insertelement <2 x i32> %1992, i32 %1991, i32 1
  %1994 = bitcast <2 x i32> %1993 to i64
  %1995 = shl i64 %1994, 1
  %1996 = add i64 %.in3822, %1995
  %1997 = add i64 %1996, %sink_3832
  %1998 = inttoptr i64 %1997 to i16 addrspace(4)*
  %1999 = addrspacecast i16 addrspace(4)* %1998 to i16 addrspace(1)*
  %2000 = load i16, i16 addrspace(1)* %1999, align 2
  %2001 = zext i16 %1988 to i32
  %2002 = shl nuw i32 %2001, 16, !spirv.Decorations !921
  %2003 = bitcast i32 %2002 to float
  %2004 = zext i16 %2000 to i32
  %2005 = shl nuw i32 %2004, 16, !spirv.Decorations !921
  %2006 = bitcast i32 %2005 to float
  %2007 = fmul reassoc nsz arcp contract float %2003, %2006, !spirv.Decorations !898
  %2008 = fadd reassoc nsz arcp contract float %2007, %.sroa.246.1, !spirv.Decorations !898
  br label %.preheader.13, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.13:                                    ; preds = %._crit_edge.2.13..preheader.13_crit_edge, %1983
  %.sroa.246.2 = phi float [ %2008, %1983 ], [ %.sroa.246.1, %._crit_edge.2.13..preheader.13_crit_edge ]
  %sink_3830 = shl nsw i64 %363, 1
  br i1 %302, label %2009, label %.preheader.13.._crit_edge.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.13.._crit_edge.14_crit_edge:           ; preds = %.preheader.13
  br label %._crit_edge.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2009:                                             ; preds = %.preheader.13
  %.sroa.256.0.insert.ext863 = zext i32 %552 to i64
  %2010 = shl nuw nsw i64 %.sroa.256.0.insert.ext863, 1
  %2011 = add i64 %548, %2010
  %2012 = inttoptr i64 %2011 to i16 addrspace(4)*
  %2013 = addrspacecast i16 addrspace(4)* %2012 to i16 addrspace(1)*
  %2014 = load i16, i16 addrspace(1)* %2013, align 2
  %2015 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2016 = extractvalue { i32, i32 } %2015, 0
  %2017 = extractvalue { i32, i32 } %2015, 1
  %2018 = insertelement <2 x i32> undef, i32 %2016, i32 0
  %2019 = insertelement <2 x i32> %2018, i32 %2017, i32 1
  %2020 = bitcast <2 x i32> %2019 to i64
  %2021 = shl i64 %2020, 1
  %2022 = add i64 %.in3822, %2021
  %2023 = add i64 %2022, %sink_3830
  %2024 = inttoptr i64 %2023 to i16 addrspace(4)*
  %2025 = addrspacecast i16 addrspace(4)* %2024 to i16 addrspace(1)*
  %2026 = load i16, i16 addrspace(1)* %2025, align 2
  %2027 = zext i16 %2014 to i32
  %2028 = shl nuw i32 %2027, 16, !spirv.Decorations !921
  %2029 = bitcast i32 %2028 to float
  %2030 = zext i16 %2026 to i32
  %2031 = shl nuw i32 %2030, 16, !spirv.Decorations !921
  %2032 = bitcast i32 %2031 to float
  %2033 = fmul reassoc nsz arcp contract float %2029, %2032, !spirv.Decorations !898
  %2034 = fadd reassoc nsz arcp contract float %2033, %.sroa.58.1, !spirv.Decorations !898
  br label %._crit_edge.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.14:                                   ; preds = %.preheader.13.._crit_edge.14_crit_edge, %2009
  %.sroa.58.2 = phi float [ %2034, %2009 ], [ %.sroa.58.1, %.preheader.13.._crit_edge.14_crit_edge ]
  br i1 %305, label %2035, label %._crit_edge.14.._crit_edge.1.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.14.._crit_edge.1.14_crit_edge:        ; preds = %._crit_edge.14
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2035:                                             ; preds = %._crit_edge.14
  %.sroa.256.0.insert.ext868 = zext i32 %552 to i64
  %2036 = shl nuw nsw i64 %.sroa.256.0.insert.ext868, 1
  %2037 = add i64 %549, %2036
  %2038 = inttoptr i64 %2037 to i16 addrspace(4)*
  %2039 = addrspacecast i16 addrspace(4)* %2038 to i16 addrspace(1)*
  %2040 = load i16, i16 addrspace(1)* %2039, align 2
  %2041 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2042 = extractvalue { i32, i32 } %2041, 0
  %2043 = extractvalue { i32, i32 } %2041, 1
  %2044 = insertelement <2 x i32> undef, i32 %2042, i32 0
  %2045 = insertelement <2 x i32> %2044, i32 %2043, i32 1
  %2046 = bitcast <2 x i32> %2045 to i64
  %2047 = shl i64 %2046, 1
  %2048 = add i64 %.in3822, %2047
  %2049 = add i64 %2048, %sink_3830
  %2050 = inttoptr i64 %2049 to i16 addrspace(4)*
  %2051 = addrspacecast i16 addrspace(4)* %2050 to i16 addrspace(1)*
  %2052 = load i16, i16 addrspace(1)* %2051, align 2
  %2053 = zext i16 %2040 to i32
  %2054 = shl nuw i32 %2053, 16, !spirv.Decorations !921
  %2055 = bitcast i32 %2054 to float
  %2056 = zext i16 %2052 to i32
  %2057 = shl nuw i32 %2056, 16, !spirv.Decorations !921
  %2058 = bitcast i32 %2057 to float
  %2059 = fmul reassoc nsz arcp contract float %2055, %2058, !spirv.Decorations !898
  %2060 = fadd reassoc nsz arcp contract float %2059, %.sroa.122.1, !spirv.Decorations !898
  br label %._crit_edge.1.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.14:                                 ; preds = %._crit_edge.14.._crit_edge.1.14_crit_edge, %2035
  %.sroa.122.2 = phi float [ %2060, %2035 ], [ %.sroa.122.1, %._crit_edge.14.._crit_edge.1.14_crit_edge ]
  br i1 %308, label %2061, label %._crit_edge.1.14.._crit_edge.2.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.14.._crit_edge.2.14_crit_edge:      ; preds = %._crit_edge.1.14
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2061:                                             ; preds = %._crit_edge.1.14
  %.sroa.256.0.insert.ext873 = zext i32 %552 to i64
  %2062 = shl nuw nsw i64 %.sroa.256.0.insert.ext873, 1
  %2063 = add i64 %550, %2062
  %2064 = inttoptr i64 %2063 to i16 addrspace(4)*
  %2065 = addrspacecast i16 addrspace(4)* %2064 to i16 addrspace(1)*
  %2066 = load i16, i16 addrspace(1)* %2065, align 2
  %2067 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2068 = extractvalue { i32, i32 } %2067, 0
  %2069 = extractvalue { i32, i32 } %2067, 1
  %2070 = insertelement <2 x i32> undef, i32 %2068, i32 0
  %2071 = insertelement <2 x i32> %2070, i32 %2069, i32 1
  %2072 = bitcast <2 x i32> %2071 to i64
  %2073 = shl i64 %2072, 1
  %2074 = add i64 %.in3822, %2073
  %2075 = add i64 %2074, %sink_3830
  %2076 = inttoptr i64 %2075 to i16 addrspace(4)*
  %2077 = addrspacecast i16 addrspace(4)* %2076 to i16 addrspace(1)*
  %2078 = load i16, i16 addrspace(1)* %2077, align 2
  %2079 = zext i16 %2066 to i32
  %2080 = shl nuw i32 %2079, 16, !spirv.Decorations !921
  %2081 = bitcast i32 %2080 to float
  %2082 = zext i16 %2078 to i32
  %2083 = shl nuw i32 %2082, 16, !spirv.Decorations !921
  %2084 = bitcast i32 %2083 to float
  %2085 = fmul reassoc nsz arcp contract float %2081, %2084, !spirv.Decorations !898
  %2086 = fadd reassoc nsz arcp contract float %2085, %.sroa.186.1, !spirv.Decorations !898
  br label %._crit_edge.2.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.14:                                 ; preds = %._crit_edge.1.14.._crit_edge.2.14_crit_edge, %2061
  %.sroa.186.2 = phi float [ %2086, %2061 ], [ %.sroa.186.1, %._crit_edge.1.14.._crit_edge.2.14_crit_edge ]
  br i1 %311, label %2087, label %._crit_edge.2.14..preheader.14_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.14..preheader.14_crit_edge:         ; preds = %._crit_edge.2.14
  br label %.preheader.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2087:                                             ; preds = %._crit_edge.2.14
  %.sroa.256.0.insert.ext878 = zext i32 %552 to i64
  %2088 = shl nuw nsw i64 %.sroa.256.0.insert.ext878, 1
  %2089 = add i64 %551, %2088
  %2090 = inttoptr i64 %2089 to i16 addrspace(4)*
  %2091 = addrspacecast i16 addrspace(4)* %2090 to i16 addrspace(1)*
  %2092 = load i16, i16 addrspace(1)* %2091, align 2
  %2093 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2094 = extractvalue { i32, i32 } %2093, 0
  %2095 = extractvalue { i32, i32 } %2093, 1
  %2096 = insertelement <2 x i32> undef, i32 %2094, i32 0
  %2097 = insertelement <2 x i32> %2096, i32 %2095, i32 1
  %2098 = bitcast <2 x i32> %2097 to i64
  %2099 = shl i64 %2098, 1
  %2100 = add i64 %.in3822, %2099
  %2101 = add i64 %2100, %sink_3830
  %2102 = inttoptr i64 %2101 to i16 addrspace(4)*
  %2103 = addrspacecast i16 addrspace(4)* %2102 to i16 addrspace(1)*
  %2104 = load i16, i16 addrspace(1)* %2103, align 2
  %2105 = zext i16 %2092 to i32
  %2106 = shl nuw i32 %2105, 16, !spirv.Decorations !921
  %2107 = bitcast i32 %2106 to float
  %2108 = zext i16 %2104 to i32
  %2109 = shl nuw i32 %2108, 16, !spirv.Decorations !921
  %2110 = bitcast i32 %2109 to float
  %2111 = fmul reassoc nsz arcp contract float %2107, %2110, !spirv.Decorations !898
  %2112 = fadd reassoc nsz arcp contract float %2111, %.sroa.250.1, !spirv.Decorations !898
  br label %.preheader.14, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.14:                                    ; preds = %._crit_edge.2.14..preheader.14_crit_edge, %2087
  %.sroa.250.2 = phi float [ %2112, %2087 ], [ %.sroa.250.1, %._crit_edge.2.14..preheader.14_crit_edge ]
  %sink_3828 = shl nsw i64 %364, 1
  br i1 %315, label %2113, label %.preheader.14.._crit_edge.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.14.._crit_edge.15_crit_edge:           ; preds = %.preheader.14
  br label %._crit_edge.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2113:                                             ; preds = %.preheader.14
  %.sroa.256.0.insert.ext883 = zext i32 %552 to i64
  %2114 = shl nuw nsw i64 %.sroa.256.0.insert.ext883, 1
  %2115 = add i64 %548, %2114
  %2116 = inttoptr i64 %2115 to i16 addrspace(4)*
  %2117 = addrspacecast i16 addrspace(4)* %2116 to i16 addrspace(1)*
  %2118 = load i16, i16 addrspace(1)* %2117, align 2
  %2119 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2120 = extractvalue { i32, i32 } %2119, 0
  %2121 = extractvalue { i32, i32 } %2119, 1
  %2122 = insertelement <2 x i32> undef, i32 %2120, i32 0
  %2123 = insertelement <2 x i32> %2122, i32 %2121, i32 1
  %2124 = bitcast <2 x i32> %2123 to i64
  %2125 = shl i64 %2124, 1
  %2126 = add i64 %.in3822, %2125
  %2127 = add i64 %2126, %sink_3828
  %2128 = inttoptr i64 %2127 to i16 addrspace(4)*
  %2129 = addrspacecast i16 addrspace(4)* %2128 to i16 addrspace(1)*
  %2130 = load i16, i16 addrspace(1)* %2129, align 2
  %2131 = zext i16 %2118 to i32
  %2132 = shl nuw i32 %2131, 16, !spirv.Decorations !921
  %2133 = bitcast i32 %2132 to float
  %2134 = zext i16 %2130 to i32
  %2135 = shl nuw i32 %2134, 16, !spirv.Decorations !921
  %2136 = bitcast i32 %2135 to float
  %2137 = fmul reassoc nsz arcp contract float %2133, %2136, !spirv.Decorations !898
  %2138 = fadd reassoc nsz arcp contract float %2137, %.sroa.62.1, !spirv.Decorations !898
  br label %._crit_edge.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.15:                                   ; preds = %.preheader.14.._crit_edge.15_crit_edge, %2113
  %.sroa.62.2 = phi float [ %2138, %2113 ], [ %.sroa.62.1, %.preheader.14.._crit_edge.15_crit_edge ]
  br i1 %318, label %2139, label %._crit_edge.15.._crit_edge.1.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.15.._crit_edge.1.15_crit_edge:        ; preds = %._crit_edge.15
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2139:                                             ; preds = %._crit_edge.15
  %.sroa.256.0.insert.ext888 = zext i32 %552 to i64
  %2140 = shl nuw nsw i64 %.sroa.256.0.insert.ext888, 1
  %2141 = add i64 %549, %2140
  %2142 = inttoptr i64 %2141 to i16 addrspace(4)*
  %2143 = addrspacecast i16 addrspace(4)* %2142 to i16 addrspace(1)*
  %2144 = load i16, i16 addrspace(1)* %2143, align 2
  %2145 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2146 = extractvalue { i32, i32 } %2145, 0
  %2147 = extractvalue { i32, i32 } %2145, 1
  %2148 = insertelement <2 x i32> undef, i32 %2146, i32 0
  %2149 = insertelement <2 x i32> %2148, i32 %2147, i32 1
  %2150 = bitcast <2 x i32> %2149 to i64
  %2151 = shl i64 %2150, 1
  %2152 = add i64 %.in3822, %2151
  %2153 = add i64 %2152, %sink_3828
  %2154 = inttoptr i64 %2153 to i16 addrspace(4)*
  %2155 = addrspacecast i16 addrspace(4)* %2154 to i16 addrspace(1)*
  %2156 = load i16, i16 addrspace(1)* %2155, align 2
  %2157 = zext i16 %2144 to i32
  %2158 = shl nuw i32 %2157, 16, !spirv.Decorations !921
  %2159 = bitcast i32 %2158 to float
  %2160 = zext i16 %2156 to i32
  %2161 = shl nuw i32 %2160, 16, !spirv.Decorations !921
  %2162 = bitcast i32 %2161 to float
  %2163 = fmul reassoc nsz arcp contract float %2159, %2162, !spirv.Decorations !898
  %2164 = fadd reassoc nsz arcp contract float %2163, %.sroa.126.1, !spirv.Decorations !898
  br label %._crit_edge.1.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.1.15:                                 ; preds = %._crit_edge.15.._crit_edge.1.15_crit_edge, %2139
  %.sroa.126.2 = phi float [ %2164, %2139 ], [ %.sroa.126.1, %._crit_edge.15.._crit_edge.1.15_crit_edge ]
  br i1 %321, label %2165, label %._crit_edge.1.15.._crit_edge.2.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.1.15.._crit_edge.2.15_crit_edge:      ; preds = %._crit_edge.1.15
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2165:                                             ; preds = %._crit_edge.1.15
  %.sroa.256.0.insert.ext893 = zext i32 %552 to i64
  %2166 = shl nuw nsw i64 %.sroa.256.0.insert.ext893, 1
  %2167 = add i64 %550, %2166
  %2168 = inttoptr i64 %2167 to i16 addrspace(4)*
  %2169 = addrspacecast i16 addrspace(4)* %2168 to i16 addrspace(1)*
  %2170 = load i16, i16 addrspace(1)* %2169, align 2
  %2171 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2172 = extractvalue { i32, i32 } %2171, 0
  %2173 = extractvalue { i32, i32 } %2171, 1
  %2174 = insertelement <2 x i32> undef, i32 %2172, i32 0
  %2175 = insertelement <2 x i32> %2174, i32 %2173, i32 1
  %2176 = bitcast <2 x i32> %2175 to i64
  %2177 = shl i64 %2176, 1
  %2178 = add i64 %.in3822, %2177
  %2179 = add i64 %2178, %sink_3828
  %2180 = inttoptr i64 %2179 to i16 addrspace(4)*
  %2181 = addrspacecast i16 addrspace(4)* %2180 to i16 addrspace(1)*
  %2182 = load i16, i16 addrspace(1)* %2181, align 2
  %2183 = zext i16 %2170 to i32
  %2184 = shl nuw i32 %2183, 16, !spirv.Decorations !921
  %2185 = bitcast i32 %2184 to float
  %2186 = zext i16 %2182 to i32
  %2187 = shl nuw i32 %2186, 16, !spirv.Decorations !921
  %2188 = bitcast i32 %2187 to float
  %2189 = fmul reassoc nsz arcp contract float %2185, %2188, !spirv.Decorations !898
  %2190 = fadd reassoc nsz arcp contract float %2189, %.sroa.190.1, !spirv.Decorations !898
  br label %._crit_edge.2.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

._crit_edge.2.15:                                 ; preds = %._crit_edge.1.15.._crit_edge.2.15_crit_edge, %2165
  %.sroa.190.2 = phi float [ %2190, %2165 ], [ %.sroa.190.1, %._crit_edge.1.15.._crit_edge.2.15_crit_edge ]
  br i1 %324, label %2191, label %._crit_edge.2.15..preheader.15_crit_edge, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

._crit_edge.2.15..preheader.15_crit_edge:         ; preds = %._crit_edge.2.15
  br label %.preheader.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

2191:                                             ; preds = %._crit_edge.2.15
  %.sroa.256.0.insert.ext898 = zext i32 %552 to i64
  %2192 = shl nuw nsw i64 %.sroa.256.0.insert.ext898, 1
  %2193 = add i64 %551, %2192
  %2194 = inttoptr i64 %2193 to i16 addrspace(4)*
  %2195 = addrspacecast i16 addrspace(4)* %2194 to i16 addrspace(1)*
  %2196 = load i16, i16 addrspace(1)* %2195, align 2
  %2197 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %552, i32 0, i32 %sink_3826, i32 %sink_3825)
  %2198 = extractvalue { i32, i32 } %2197, 0
  %2199 = extractvalue { i32, i32 } %2197, 1
  %2200 = insertelement <2 x i32> undef, i32 %2198, i32 0
  %2201 = insertelement <2 x i32> %2200, i32 %2199, i32 1
  %2202 = bitcast <2 x i32> %2201 to i64
  %2203 = shl i64 %2202, 1
  %2204 = add i64 %.in3822, %2203
  %2205 = add i64 %2204, %sink_3828
  %2206 = inttoptr i64 %2205 to i16 addrspace(4)*
  %2207 = addrspacecast i16 addrspace(4)* %2206 to i16 addrspace(1)*
  %2208 = load i16, i16 addrspace(1)* %2207, align 2
  %2209 = zext i16 %2196 to i32
  %2210 = shl nuw i32 %2209, 16, !spirv.Decorations !921
  %2211 = bitcast i32 %2210 to float
  %2212 = zext i16 %2208 to i32
  %2213 = shl nuw i32 %2212, 16, !spirv.Decorations !921
  %2214 = bitcast i32 %2213 to float
  %2215 = fmul reassoc nsz arcp contract float %2211, %2214, !spirv.Decorations !898
  %2216 = fadd reassoc nsz arcp contract float %2215, %.sroa.254.1, !spirv.Decorations !898
  br label %.preheader.15, !stats.blockFrequency.digits !920, !stats.blockFrequency.scale !879

.preheader.15:                                    ; preds = %._crit_edge.2.15..preheader.15_crit_edge, %2191
  %.sroa.254.2 = phi float [ %2216, %2191 ], [ %.sroa.254.1, %._crit_edge.2.15..preheader.15_crit_edge ]
  %2217 = add nuw nsw i32 %552, 1, !spirv.Decorations !923
  %2218 = icmp slt i32 %2217, %const_reg_dword2
  br i1 %2218, label %.preheader.15..preheader.preheader_crit_edge, label %.preheader1.preheader.loopexit, !stats.blockFrequency.digits !919, !stats.blockFrequency.scale !879

.preheader.15..preheader.preheader_crit_edge:     ; preds = %.preheader.15
  br label %.preheader.preheader, !stats.blockFrequency.digits !924, !stats.blockFrequency.scale !879

.preheader1.preheader.loopexit:                   ; preds = %.preheader.15
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
  br label %.preheader1.preheader, !stats.blockFrequency.digits !918, !stats.blockFrequency.scale !879

.preheader1.preheader:                            ; preds = %.preheader2.preheader..preheader1.preheader_crit_edge, %.preheader1.preheader.loopexit
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
  %sink_3870 = bitcast <2 x i32> %377 to i64
  %sink_3862 = shl i64 %sink_3870, 2
  %sink_3860 = shl nsw i64 %331, 2
  br i1 %117, label %2219, label %.preheader1.preheader.._crit_edge70_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.preheader.._crit_edge70_crit_edge:    ; preds = %.preheader1.preheader
  br label %._crit_edge70, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2219:                                             ; preds = %.preheader1.preheader
  %2220 = fmul reassoc nsz arcp contract float %.sroa.0.0, %1, !spirv.Decorations !898
  br i1 %78, label %2225, label %2221, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2221:                                             ; preds = %2219
  %2222 = add i64 %.in, %372
  %2223 = inttoptr i64 %2222 to float addrspace(4)*
  %2224 = addrspacecast float addrspace(4)* %2223 to float addrspace(1)*
  store float %2220, float addrspace(1)* %2224, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2225:                                             ; preds = %2219
  %2226 = add i64 %.in3821, %sink_3862
  %2227 = add i64 %2226, %sink_3860
  %2228 = inttoptr i64 %2227 to float addrspace(4)*
  %2229 = addrspacecast float addrspace(4)* %2228 to float addrspace(1)*
  %2230 = load float, float addrspace(1)* %2229, align 4
  %2231 = fmul reassoc nsz arcp contract float %2230, %4, !spirv.Decorations !898
  %2232 = fadd reassoc nsz arcp contract float %2220, %2231, !spirv.Decorations !898
  %2233 = add i64 %.in, %372
  %2234 = inttoptr i64 %2233 to float addrspace(4)*
  %2235 = addrspacecast float addrspace(4)* %2234 to float addrspace(1)*
  store float %2232, float addrspace(1)* %2235, align 4
  br label %._crit_edge70, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70:                                    ; preds = %.preheader1.preheader.._crit_edge70_crit_edge, %2221, %2225
  %sink_3869 = bitcast <2 x i32> %390 to i64
  %sink_3859 = shl i64 %sink_3869, 2
  br i1 %121, label %2236, label %._crit_edge70.._crit_edge70.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.._crit_edge70.1_crit_edge:          ; preds = %._crit_edge70
  br label %._crit_edge70.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2236:                                             ; preds = %._crit_edge70
  %2237 = fmul reassoc nsz arcp contract float %.sroa.66.0, %1, !spirv.Decorations !898
  br i1 %78, label %2242, label %2238, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2238:                                             ; preds = %2236
  %2239 = add i64 %.in, %385
  %2240 = inttoptr i64 %2239 to float addrspace(4)*
  %2241 = addrspacecast float addrspace(4)* %2240 to float addrspace(1)*
  store float %2237, float addrspace(1)* %2241, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2242:                                             ; preds = %2236
  %2243 = add i64 %.in3821, %sink_3859
  %2244 = add i64 %2243, %sink_3860
  %2245 = inttoptr i64 %2244 to float addrspace(4)*
  %2246 = addrspacecast float addrspace(4)* %2245 to float addrspace(1)*
  %2247 = load float, float addrspace(1)* %2246, align 4
  %2248 = fmul reassoc nsz arcp contract float %2247, %4, !spirv.Decorations !898
  %2249 = fadd reassoc nsz arcp contract float %2237, %2248, !spirv.Decorations !898
  %2250 = add i64 %.in, %385
  %2251 = inttoptr i64 %2250 to float addrspace(4)*
  %2252 = addrspacecast float addrspace(4)* %2251 to float addrspace(1)*
  store float %2249, float addrspace(1)* %2252, align 4
  br label %._crit_edge70.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1:                                  ; preds = %._crit_edge70.._crit_edge70.1_crit_edge, %2242, %2238
  %sink_3868 = bitcast <2 x i32> %403 to i64
  %sink_3858 = shl i64 %sink_3868, 2
  br i1 %125, label %2253, label %._crit_edge70.1.._crit_edge70.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.._crit_edge70.2_crit_edge:        ; preds = %._crit_edge70.1
  br label %._crit_edge70.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2253:                                             ; preds = %._crit_edge70.1
  %2254 = fmul reassoc nsz arcp contract float %.sroa.130.0, %1, !spirv.Decorations !898
  br i1 %78, label %2259, label %2255, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2255:                                             ; preds = %2253
  %2256 = add i64 %.in, %398
  %2257 = inttoptr i64 %2256 to float addrspace(4)*
  %2258 = addrspacecast float addrspace(4)* %2257 to float addrspace(1)*
  store float %2254, float addrspace(1)* %2258, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2259:                                             ; preds = %2253
  %2260 = add i64 %.in3821, %sink_3858
  %2261 = add i64 %2260, %sink_3860
  %2262 = inttoptr i64 %2261 to float addrspace(4)*
  %2263 = addrspacecast float addrspace(4)* %2262 to float addrspace(1)*
  %2264 = load float, float addrspace(1)* %2263, align 4
  %2265 = fmul reassoc nsz arcp contract float %2264, %4, !spirv.Decorations !898
  %2266 = fadd reassoc nsz arcp contract float %2254, %2265, !spirv.Decorations !898
  %2267 = add i64 %.in, %398
  %2268 = inttoptr i64 %2267 to float addrspace(4)*
  %2269 = addrspacecast float addrspace(4)* %2268 to float addrspace(1)*
  store float %2266, float addrspace(1)* %2269, align 4
  br label %._crit_edge70.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2:                                  ; preds = %._crit_edge70.1.._crit_edge70.2_crit_edge, %2259, %2255
  %sink_3867 = bitcast <2 x i32> %416 to i64
  %sink_3857 = shl i64 %sink_3867, 2
  br i1 %129, label %2270, label %._crit_edge70.2..preheader1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2..preheader1_crit_edge:            ; preds = %._crit_edge70.2
  br label %.preheader1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2270:                                             ; preds = %._crit_edge70.2
  %2271 = fmul reassoc nsz arcp contract float %.sroa.194.0, %1, !spirv.Decorations !898
  br i1 %78, label %2276, label %2272, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2272:                                             ; preds = %2270
  %2273 = add i64 %.in, %411
  %2274 = inttoptr i64 %2273 to float addrspace(4)*
  %2275 = addrspacecast float addrspace(4)* %2274 to float addrspace(1)*
  store float %2271, float addrspace(1)* %2275, align 4
  br label %.preheader1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2276:                                             ; preds = %2270
  %2277 = add i64 %.in3821, %sink_3857
  %2278 = add i64 %2277, %sink_3860
  %2279 = inttoptr i64 %2278 to float addrspace(4)*
  %2280 = addrspacecast float addrspace(4)* %2279 to float addrspace(1)*
  %2281 = load float, float addrspace(1)* %2280, align 4
  %2282 = fmul reassoc nsz arcp contract float %2281, %4, !spirv.Decorations !898
  %2283 = fadd reassoc nsz arcp contract float %2271, %2282, !spirv.Decorations !898
  %2284 = add i64 %.in, %411
  %2285 = inttoptr i64 %2284 to float addrspace(4)*
  %2286 = addrspacecast float addrspace(4)* %2285 to float addrspace(1)*
  store float %2283, float addrspace(1)* %2286, align 4
  br label %.preheader1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1:                                      ; preds = %._crit_edge70.2..preheader1_crit_edge, %2276, %2272
  %sink_3855 = shl nsw i64 %350, 2
  br i1 %133, label %2287, label %.preheader1.._crit_edge70.176_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.._crit_edge70.176_crit_edge:          ; preds = %.preheader1
  br label %._crit_edge70.176, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2287:                                             ; preds = %.preheader1
  %2288 = fmul reassoc nsz arcp contract float %.sroa.6.0, %1, !spirv.Decorations !898
  br i1 %78, label %2293, label %2289, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2289:                                             ; preds = %2287
  %2290 = add i64 %.in, %418
  %2291 = inttoptr i64 %2290 to float addrspace(4)*
  %2292 = addrspacecast float addrspace(4)* %2291 to float addrspace(1)*
  store float %2288, float addrspace(1)* %2292, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2293:                                             ; preds = %2287
  %2294 = add i64 %.in3821, %sink_3862
  %2295 = add i64 %2294, %sink_3855
  %2296 = inttoptr i64 %2295 to float addrspace(4)*
  %2297 = addrspacecast float addrspace(4)* %2296 to float addrspace(1)*
  %2298 = load float, float addrspace(1)* %2297, align 4
  %2299 = fmul reassoc nsz arcp contract float %2298, %4, !spirv.Decorations !898
  %2300 = fadd reassoc nsz arcp contract float %2288, %2299, !spirv.Decorations !898
  %2301 = add i64 %.in, %418
  %2302 = inttoptr i64 %2301 to float addrspace(4)*
  %2303 = addrspacecast float addrspace(4)* %2302 to float addrspace(1)*
  store float %2300, float addrspace(1)* %2303, align 4
  br label %._crit_edge70.176, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.176:                                ; preds = %.preheader1.._crit_edge70.176_crit_edge, %2293, %2289
  br i1 %136, label %2304, label %._crit_edge70.176.._crit_edge70.1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.176.._crit_edge70.1.1_crit_edge:    ; preds = %._crit_edge70.176
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2304:                                             ; preds = %._crit_edge70.176
  %2305 = fmul reassoc nsz arcp contract float %.sroa.70.0, %1, !spirv.Decorations !898
  br i1 %78, label %2310, label %2306, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2306:                                             ; preds = %2304
  %2307 = add i64 %.in, %420
  %2308 = inttoptr i64 %2307 to float addrspace(4)*
  %2309 = addrspacecast float addrspace(4)* %2308 to float addrspace(1)*
  store float %2305, float addrspace(1)* %2309, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2310:                                             ; preds = %2304
  %2311 = add i64 %.in3821, %sink_3859
  %2312 = add i64 %2311, %sink_3855
  %2313 = inttoptr i64 %2312 to float addrspace(4)*
  %2314 = addrspacecast float addrspace(4)* %2313 to float addrspace(1)*
  %2315 = load float, float addrspace(1)* %2314, align 4
  %2316 = fmul reassoc nsz arcp contract float %2315, %4, !spirv.Decorations !898
  %2317 = fadd reassoc nsz arcp contract float %2305, %2316, !spirv.Decorations !898
  %2318 = add i64 %.in, %420
  %2319 = inttoptr i64 %2318 to float addrspace(4)*
  %2320 = addrspacecast float addrspace(4)* %2319 to float addrspace(1)*
  store float %2317, float addrspace(1)* %2320, align 4
  br label %._crit_edge70.1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.1:                                ; preds = %._crit_edge70.176.._crit_edge70.1.1_crit_edge, %2310, %2306
  br i1 %139, label %2321, label %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.1.._crit_edge70.2.1_crit_edge:    ; preds = %._crit_edge70.1.1
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2321:                                             ; preds = %._crit_edge70.1.1
  %2322 = fmul reassoc nsz arcp contract float %.sroa.134.0, %1, !spirv.Decorations !898
  br i1 %78, label %2327, label %2323, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2323:                                             ; preds = %2321
  %2324 = add i64 %.in, %422
  %2325 = inttoptr i64 %2324 to float addrspace(4)*
  %2326 = addrspacecast float addrspace(4)* %2325 to float addrspace(1)*
  store float %2322, float addrspace(1)* %2326, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2327:                                             ; preds = %2321
  %2328 = add i64 %.in3821, %sink_3858
  %2329 = add i64 %2328, %sink_3855
  %2330 = inttoptr i64 %2329 to float addrspace(4)*
  %2331 = addrspacecast float addrspace(4)* %2330 to float addrspace(1)*
  %2332 = load float, float addrspace(1)* %2331, align 4
  %2333 = fmul reassoc nsz arcp contract float %2332, %4, !spirv.Decorations !898
  %2334 = fadd reassoc nsz arcp contract float %2322, %2333, !spirv.Decorations !898
  %2335 = add i64 %.in, %422
  %2336 = inttoptr i64 %2335 to float addrspace(4)*
  %2337 = addrspacecast float addrspace(4)* %2336 to float addrspace(1)*
  store float %2334, float addrspace(1)* %2337, align 4
  br label %._crit_edge70.2.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.1:                                ; preds = %._crit_edge70.1.1.._crit_edge70.2.1_crit_edge, %2327, %2323
  br i1 %142, label %2338, label %._crit_edge70.2.1..preheader1.1_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.1..preheader1.1_crit_edge:        ; preds = %._crit_edge70.2.1
  br label %.preheader1.1, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2338:                                             ; preds = %._crit_edge70.2.1
  %2339 = fmul reassoc nsz arcp contract float %.sroa.198.0, %1, !spirv.Decorations !898
  br i1 %78, label %2344, label %2340, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2340:                                             ; preds = %2338
  %2341 = add i64 %.in, %424
  %2342 = inttoptr i64 %2341 to float addrspace(4)*
  %2343 = addrspacecast float addrspace(4)* %2342 to float addrspace(1)*
  store float %2339, float addrspace(1)* %2343, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2344:                                             ; preds = %2338
  %2345 = add i64 %.in3821, %sink_3857
  %2346 = add i64 %2345, %sink_3855
  %2347 = inttoptr i64 %2346 to float addrspace(4)*
  %2348 = addrspacecast float addrspace(4)* %2347 to float addrspace(1)*
  %2349 = load float, float addrspace(1)* %2348, align 4
  %2350 = fmul reassoc nsz arcp contract float %2349, %4, !spirv.Decorations !898
  %2351 = fadd reassoc nsz arcp contract float %2339, %2350, !spirv.Decorations !898
  %2352 = add i64 %.in, %424
  %2353 = inttoptr i64 %2352 to float addrspace(4)*
  %2354 = addrspacecast float addrspace(4)* %2353 to float addrspace(1)*
  store float %2351, float addrspace(1)* %2354, align 4
  br label %.preheader1.1, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.1:                                    ; preds = %._crit_edge70.2.1..preheader1.1_crit_edge, %2344, %2340
  %sink_3853 = shl nsw i64 %351, 2
  br i1 %146, label %2355, label %.preheader1.1.._crit_edge70.277_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.1.._crit_edge70.277_crit_edge:        ; preds = %.preheader1.1
  br label %._crit_edge70.277, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2355:                                             ; preds = %.preheader1.1
  %2356 = fmul reassoc nsz arcp contract float %.sroa.10.0, %1, !spirv.Decorations !898
  br i1 %78, label %2361, label %2357, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2357:                                             ; preds = %2355
  %2358 = add i64 %.in, %426
  %2359 = inttoptr i64 %2358 to float addrspace(4)*
  %2360 = addrspacecast float addrspace(4)* %2359 to float addrspace(1)*
  store float %2356, float addrspace(1)* %2360, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2361:                                             ; preds = %2355
  %2362 = add i64 %.in3821, %sink_3862
  %2363 = add i64 %2362, %sink_3853
  %2364 = inttoptr i64 %2363 to float addrspace(4)*
  %2365 = addrspacecast float addrspace(4)* %2364 to float addrspace(1)*
  %2366 = load float, float addrspace(1)* %2365, align 4
  %2367 = fmul reassoc nsz arcp contract float %2366, %4, !spirv.Decorations !898
  %2368 = fadd reassoc nsz arcp contract float %2356, %2367, !spirv.Decorations !898
  %2369 = add i64 %.in, %426
  %2370 = inttoptr i64 %2369 to float addrspace(4)*
  %2371 = addrspacecast float addrspace(4)* %2370 to float addrspace(1)*
  store float %2368, float addrspace(1)* %2371, align 4
  br label %._crit_edge70.277, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.277:                                ; preds = %.preheader1.1.._crit_edge70.277_crit_edge, %2361, %2357
  br i1 %149, label %2372, label %._crit_edge70.277.._crit_edge70.1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.277.._crit_edge70.1.2_crit_edge:    ; preds = %._crit_edge70.277
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2372:                                             ; preds = %._crit_edge70.277
  %2373 = fmul reassoc nsz arcp contract float %.sroa.74.0, %1, !spirv.Decorations !898
  br i1 %78, label %2378, label %2374, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2374:                                             ; preds = %2372
  %2375 = add i64 %.in, %428
  %2376 = inttoptr i64 %2375 to float addrspace(4)*
  %2377 = addrspacecast float addrspace(4)* %2376 to float addrspace(1)*
  store float %2373, float addrspace(1)* %2377, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2378:                                             ; preds = %2372
  %2379 = add i64 %.in3821, %sink_3859
  %2380 = add i64 %2379, %sink_3853
  %2381 = inttoptr i64 %2380 to float addrspace(4)*
  %2382 = addrspacecast float addrspace(4)* %2381 to float addrspace(1)*
  %2383 = load float, float addrspace(1)* %2382, align 4
  %2384 = fmul reassoc nsz arcp contract float %2383, %4, !spirv.Decorations !898
  %2385 = fadd reassoc nsz arcp contract float %2373, %2384, !spirv.Decorations !898
  %2386 = add i64 %.in, %428
  %2387 = inttoptr i64 %2386 to float addrspace(4)*
  %2388 = addrspacecast float addrspace(4)* %2387 to float addrspace(1)*
  store float %2385, float addrspace(1)* %2388, align 4
  br label %._crit_edge70.1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.2:                                ; preds = %._crit_edge70.277.._crit_edge70.1.2_crit_edge, %2378, %2374
  br i1 %152, label %2389, label %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.2.._crit_edge70.2.2_crit_edge:    ; preds = %._crit_edge70.1.2
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2389:                                             ; preds = %._crit_edge70.1.2
  %2390 = fmul reassoc nsz arcp contract float %.sroa.138.0, %1, !spirv.Decorations !898
  br i1 %78, label %2395, label %2391, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2391:                                             ; preds = %2389
  %2392 = add i64 %.in, %430
  %2393 = inttoptr i64 %2392 to float addrspace(4)*
  %2394 = addrspacecast float addrspace(4)* %2393 to float addrspace(1)*
  store float %2390, float addrspace(1)* %2394, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2395:                                             ; preds = %2389
  %2396 = add i64 %.in3821, %sink_3858
  %2397 = add i64 %2396, %sink_3853
  %2398 = inttoptr i64 %2397 to float addrspace(4)*
  %2399 = addrspacecast float addrspace(4)* %2398 to float addrspace(1)*
  %2400 = load float, float addrspace(1)* %2399, align 4
  %2401 = fmul reassoc nsz arcp contract float %2400, %4, !spirv.Decorations !898
  %2402 = fadd reassoc nsz arcp contract float %2390, %2401, !spirv.Decorations !898
  %2403 = add i64 %.in, %430
  %2404 = inttoptr i64 %2403 to float addrspace(4)*
  %2405 = addrspacecast float addrspace(4)* %2404 to float addrspace(1)*
  store float %2402, float addrspace(1)* %2405, align 4
  br label %._crit_edge70.2.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.2:                                ; preds = %._crit_edge70.1.2.._crit_edge70.2.2_crit_edge, %2395, %2391
  br i1 %155, label %2406, label %._crit_edge70.2.2..preheader1.2_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.2..preheader1.2_crit_edge:        ; preds = %._crit_edge70.2.2
  br label %.preheader1.2, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2406:                                             ; preds = %._crit_edge70.2.2
  %2407 = fmul reassoc nsz arcp contract float %.sroa.202.0, %1, !spirv.Decorations !898
  br i1 %78, label %2412, label %2408, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2408:                                             ; preds = %2406
  %2409 = add i64 %.in, %432
  %2410 = inttoptr i64 %2409 to float addrspace(4)*
  %2411 = addrspacecast float addrspace(4)* %2410 to float addrspace(1)*
  store float %2407, float addrspace(1)* %2411, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2412:                                             ; preds = %2406
  %2413 = add i64 %.in3821, %sink_3857
  %2414 = add i64 %2413, %sink_3853
  %2415 = inttoptr i64 %2414 to float addrspace(4)*
  %2416 = addrspacecast float addrspace(4)* %2415 to float addrspace(1)*
  %2417 = load float, float addrspace(1)* %2416, align 4
  %2418 = fmul reassoc nsz arcp contract float %2417, %4, !spirv.Decorations !898
  %2419 = fadd reassoc nsz arcp contract float %2407, %2418, !spirv.Decorations !898
  %2420 = add i64 %.in, %432
  %2421 = inttoptr i64 %2420 to float addrspace(4)*
  %2422 = addrspacecast float addrspace(4)* %2421 to float addrspace(1)*
  store float %2419, float addrspace(1)* %2422, align 4
  br label %.preheader1.2, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.2:                                    ; preds = %._crit_edge70.2.2..preheader1.2_crit_edge, %2412, %2408
  %sink_3851 = shl nsw i64 %352, 2
  br i1 %159, label %2423, label %.preheader1.2.._crit_edge70.378_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.2.._crit_edge70.378_crit_edge:        ; preds = %.preheader1.2
  br label %._crit_edge70.378, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2423:                                             ; preds = %.preheader1.2
  %2424 = fmul reassoc nsz arcp contract float %.sroa.14.0, %1, !spirv.Decorations !898
  br i1 %78, label %2429, label %2425, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2425:                                             ; preds = %2423
  %2426 = add i64 %.in, %434
  %2427 = inttoptr i64 %2426 to float addrspace(4)*
  %2428 = addrspacecast float addrspace(4)* %2427 to float addrspace(1)*
  store float %2424, float addrspace(1)* %2428, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2429:                                             ; preds = %2423
  %2430 = add i64 %.in3821, %sink_3862
  %2431 = add i64 %2430, %sink_3851
  %2432 = inttoptr i64 %2431 to float addrspace(4)*
  %2433 = addrspacecast float addrspace(4)* %2432 to float addrspace(1)*
  %2434 = load float, float addrspace(1)* %2433, align 4
  %2435 = fmul reassoc nsz arcp contract float %2434, %4, !spirv.Decorations !898
  %2436 = fadd reassoc nsz arcp contract float %2424, %2435, !spirv.Decorations !898
  %2437 = add i64 %.in, %434
  %2438 = inttoptr i64 %2437 to float addrspace(4)*
  %2439 = addrspacecast float addrspace(4)* %2438 to float addrspace(1)*
  store float %2436, float addrspace(1)* %2439, align 4
  br label %._crit_edge70.378, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.378:                                ; preds = %.preheader1.2.._crit_edge70.378_crit_edge, %2429, %2425
  br i1 %162, label %2440, label %._crit_edge70.378.._crit_edge70.1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.378.._crit_edge70.1.3_crit_edge:    ; preds = %._crit_edge70.378
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2440:                                             ; preds = %._crit_edge70.378
  %2441 = fmul reassoc nsz arcp contract float %.sroa.78.0, %1, !spirv.Decorations !898
  br i1 %78, label %2446, label %2442, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2442:                                             ; preds = %2440
  %2443 = add i64 %.in, %436
  %2444 = inttoptr i64 %2443 to float addrspace(4)*
  %2445 = addrspacecast float addrspace(4)* %2444 to float addrspace(1)*
  store float %2441, float addrspace(1)* %2445, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2446:                                             ; preds = %2440
  %2447 = add i64 %.in3821, %sink_3859
  %2448 = add i64 %2447, %sink_3851
  %2449 = inttoptr i64 %2448 to float addrspace(4)*
  %2450 = addrspacecast float addrspace(4)* %2449 to float addrspace(1)*
  %2451 = load float, float addrspace(1)* %2450, align 4
  %2452 = fmul reassoc nsz arcp contract float %2451, %4, !spirv.Decorations !898
  %2453 = fadd reassoc nsz arcp contract float %2441, %2452, !spirv.Decorations !898
  %2454 = add i64 %.in, %436
  %2455 = inttoptr i64 %2454 to float addrspace(4)*
  %2456 = addrspacecast float addrspace(4)* %2455 to float addrspace(1)*
  store float %2453, float addrspace(1)* %2456, align 4
  br label %._crit_edge70.1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.3:                                ; preds = %._crit_edge70.378.._crit_edge70.1.3_crit_edge, %2446, %2442
  br i1 %165, label %2457, label %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.3.._crit_edge70.2.3_crit_edge:    ; preds = %._crit_edge70.1.3
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2457:                                             ; preds = %._crit_edge70.1.3
  %2458 = fmul reassoc nsz arcp contract float %.sroa.142.0, %1, !spirv.Decorations !898
  br i1 %78, label %2463, label %2459, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2459:                                             ; preds = %2457
  %2460 = add i64 %.in, %438
  %2461 = inttoptr i64 %2460 to float addrspace(4)*
  %2462 = addrspacecast float addrspace(4)* %2461 to float addrspace(1)*
  store float %2458, float addrspace(1)* %2462, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2463:                                             ; preds = %2457
  %2464 = add i64 %.in3821, %sink_3858
  %2465 = add i64 %2464, %sink_3851
  %2466 = inttoptr i64 %2465 to float addrspace(4)*
  %2467 = addrspacecast float addrspace(4)* %2466 to float addrspace(1)*
  %2468 = load float, float addrspace(1)* %2467, align 4
  %2469 = fmul reassoc nsz arcp contract float %2468, %4, !spirv.Decorations !898
  %2470 = fadd reassoc nsz arcp contract float %2458, %2469, !spirv.Decorations !898
  %2471 = add i64 %.in, %438
  %2472 = inttoptr i64 %2471 to float addrspace(4)*
  %2473 = addrspacecast float addrspace(4)* %2472 to float addrspace(1)*
  store float %2470, float addrspace(1)* %2473, align 4
  br label %._crit_edge70.2.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.3:                                ; preds = %._crit_edge70.1.3.._crit_edge70.2.3_crit_edge, %2463, %2459
  br i1 %168, label %2474, label %._crit_edge70.2.3..preheader1.3_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.3..preheader1.3_crit_edge:        ; preds = %._crit_edge70.2.3
  br label %.preheader1.3, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2474:                                             ; preds = %._crit_edge70.2.3
  %2475 = fmul reassoc nsz arcp contract float %.sroa.206.0, %1, !spirv.Decorations !898
  br i1 %78, label %2480, label %2476, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2476:                                             ; preds = %2474
  %2477 = add i64 %.in, %440
  %2478 = inttoptr i64 %2477 to float addrspace(4)*
  %2479 = addrspacecast float addrspace(4)* %2478 to float addrspace(1)*
  store float %2475, float addrspace(1)* %2479, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2480:                                             ; preds = %2474
  %2481 = add i64 %.in3821, %sink_3857
  %2482 = add i64 %2481, %sink_3851
  %2483 = inttoptr i64 %2482 to float addrspace(4)*
  %2484 = addrspacecast float addrspace(4)* %2483 to float addrspace(1)*
  %2485 = load float, float addrspace(1)* %2484, align 4
  %2486 = fmul reassoc nsz arcp contract float %2485, %4, !spirv.Decorations !898
  %2487 = fadd reassoc nsz arcp contract float %2475, %2486, !spirv.Decorations !898
  %2488 = add i64 %.in, %440
  %2489 = inttoptr i64 %2488 to float addrspace(4)*
  %2490 = addrspacecast float addrspace(4)* %2489 to float addrspace(1)*
  store float %2487, float addrspace(1)* %2490, align 4
  br label %.preheader1.3, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.3:                                    ; preds = %._crit_edge70.2.3..preheader1.3_crit_edge, %2480, %2476
  %sink_3849 = shl nsw i64 %353, 2
  br i1 %172, label %2491, label %.preheader1.3.._crit_edge70.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.3.._crit_edge70.4_crit_edge:          ; preds = %.preheader1.3
  br label %._crit_edge70.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2491:                                             ; preds = %.preheader1.3
  %2492 = fmul reassoc nsz arcp contract float %.sroa.18.0, %1, !spirv.Decorations !898
  br i1 %78, label %2497, label %2493, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2493:                                             ; preds = %2491
  %2494 = add i64 %.in, %442
  %2495 = inttoptr i64 %2494 to float addrspace(4)*
  %2496 = addrspacecast float addrspace(4)* %2495 to float addrspace(1)*
  store float %2492, float addrspace(1)* %2496, align 4
  br label %._crit_edge70.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2497:                                             ; preds = %2491
  %2498 = add i64 %.in3821, %sink_3862
  %2499 = add i64 %2498, %sink_3849
  %2500 = inttoptr i64 %2499 to float addrspace(4)*
  %2501 = addrspacecast float addrspace(4)* %2500 to float addrspace(1)*
  %2502 = load float, float addrspace(1)* %2501, align 4
  %2503 = fmul reassoc nsz arcp contract float %2502, %4, !spirv.Decorations !898
  %2504 = fadd reassoc nsz arcp contract float %2492, %2503, !spirv.Decorations !898
  %2505 = add i64 %.in, %442
  %2506 = inttoptr i64 %2505 to float addrspace(4)*
  %2507 = addrspacecast float addrspace(4)* %2506 to float addrspace(1)*
  store float %2504, float addrspace(1)* %2507, align 4
  br label %._crit_edge70.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.4:                                  ; preds = %.preheader1.3.._crit_edge70.4_crit_edge, %2497, %2493
  br i1 %175, label %2508, label %._crit_edge70.4.._crit_edge70.1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.4.._crit_edge70.1.4_crit_edge:      ; preds = %._crit_edge70.4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2508:                                             ; preds = %._crit_edge70.4
  %2509 = fmul reassoc nsz arcp contract float %.sroa.82.0, %1, !spirv.Decorations !898
  br i1 %78, label %2514, label %2510, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2510:                                             ; preds = %2508
  %2511 = add i64 %.in, %444
  %2512 = inttoptr i64 %2511 to float addrspace(4)*
  %2513 = addrspacecast float addrspace(4)* %2512 to float addrspace(1)*
  store float %2509, float addrspace(1)* %2513, align 4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2514:                                             ; preds = %2508
  %2515 = add i64 %.in3821, %sink_3859
  %2516 = add i64 %2515, %sink_3849
  %2517 = inttoptr i64 %2516 to float addrspace(4)*
  %2518 = addrspacecast float addrspace(4)* %2517 to float addrspace(1)*
  %2519 = load float, float addrspace(1)* %2518, align 4
  %2520 = fmul reassoc nsz arcp contract float %2519, %4, !spirv.Decorations !898
  %2521 = fadd reassoc nsz arcp contract float %2509, %2520, !spirv.Decorations !898
  %2522 = add i64 %.in, %444
  %2523 = inttoptr i64 %2522 to float addrspace(4)*
  %2524 = addrspacecast float addrspace(4)* %2523 to float addrspace(1)*
  store float %2521, float addrspace(1)* %2524, align 4
  br label %._crit_edge70.1.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.4:                                ; preds = %._crit_edge70.4.._crit_edge70.1.4_crit_edge, %2514, %2510
  br i1 %178, label %2525, label %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.4.._crit_edge70.2.4_crit_edge:    ; preds = %._crit_edge70.1.4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2525:                                             ; preds = %._crit_edge70.1.4
  %2526 = fmul reassoc nsz arcp contract float %.sroa.146.0, %1, !spirv.Decorations !898
  br i1 %78, label %2531, label %2527, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2527:                                             ; preds = %2525
  %2528 = add i64 %.in, %446
  %2529 = inttoptr i64 %2528 to float addrspace(4)*
  %2530 = addrspacecast float addrspace(4)* %2529 to float addrspace(1)*
  store float %2526, float addrspace(1)* %2530, align 4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2531:                                             ; preds = %2525
  %2532 = add i64 %.in3821, %sink_3858
  %2533 = add i64 %2532, %sink_3849
  %2534 = inttoptr i64 %2533 to float addrspace(4)*
  %2535 = addrspacecast float addrspace(4)* %2534 to float addrspace(1)*
  %2536 = load float, float addrspace(1)* %2535, align 4
  %2537 = fmul reassoc nsz arcp contract float %2536, %4, !spirv.Decorations !898
  %2538 = fadd reassoc nsz arcp contract float %2526, %2537, !spirv.Decorations !898
  %2539 = add i64 %.in, %446
  %2540 = inttoptr i64 %2539 to float addrspace(4)*
  %2541 = addrspacecast float addrspace(4)* %2540 to float addrspace(1)*
  store float %2538, float addrspace(1)* %2541, align 4
  br label %._crit_edge70.2.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.4:                                ; preds = %._crit_edge70.1.4.._crit_edge70.2.4_crit_edge, %2531, %2527
  br i1 %181, label %2542, label %._crit_edge70.2.4..preheader1.4_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.4..preheader1.4_crit_edge:        ; preds = %._crit_edge70.2.4
  br label %.preheader1.4, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2542:                                             ; preds = %._crit_edge70.2.4
  %2543 = fmul reassoc nsz arcp contract float %.sroa.210.0, %1, !spirv.Decorations !898
  br i1 %78, label %2548, label %2544, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2544:                                             ; preds = %2542
  %2545 = add i64 %.in, %448
  %2546 = inttoptr i64 %2545 to float addrspace(4)*
  %2547 = addrspacecast float addrspace(4)* %2546 to float addrspace(1)*
  store float %2543, float addrspace(1)* %2547, align 4
  br label %.preheader1.4, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2548:                                             ; preds = %2542
  %2549 = add i64 %.in3821, %sink_3857
  %2550 = add i64 %2549, %sink_3849
  %2551 = inttoptr i64 %2550 to float addrspace(4)*
  %2552 = addrspacecast float addrspace(4)* %2551 to float addrspace(1)*
  %2553 = load float, float addrspace(1)* %2552, align 4
  %2554 = fmul reassoc nsz arcp contract float %2553, %4, !spirv.Decorations !898
  %2555 = fadd reassoc nsz arcp contract float %2543, %2554, !spirv.Decorations !898
  %2556 = add i64 %.in, %448
  %2557 = inttoptr i64 %2556 to float addrspace(4)*
  %2558 = addrspacecast float addrspace(4)* %2557 to float addrspace(1)*
  store float %2555, float addrspace(1)* %2558, align 4
  br label %.preheader1.4, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.4:                                    ; preds = %._crit_edge70.2.4..preheader1.4_crit_edge, %2548, %2544
  %sink_3847 = shl nsw i64 %354, 2
  br i1 %185, label %2559, label %.preheader1.4.._crit_edge70.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.4.._crit_edge70.5_crit_edge:          ; preds = %.preheader1.4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2559:                                             ; preds = %.preheader1.4
  %2560 = fmul reassoc nsz arcp contract float %.sroa.22.0, %1, !spirv.Decorations !898
  br i1 %78, label %2565, label %2561, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2561:                                             ; preds = %2559
  %2562 = add i64 %.in, %450
  %2563 = inttoptr i64 %2562 to float addrspace(4)*
  %2564 = addrspacecast float addrspace(4)* %2563 to float addrspace(1)*
  store float %2560, float addrspace(1)* %2564, align 4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2565:                                             ; preds = %2559
  %2566 = add i64 %.in3821, %sink_3862
  %2567 = add i64 %2566, %sink_3847
  %2568 = inttoptr i64 %2567 to float addrspace(4)*
  %2569 = addrspacecast float addrspace(4)* %2568 to float addrspace(1)*
  %2570 = load float, float addrspace(1)* %2569, align 4
  %2571 = fmul reassoc nsz arcp contract float %2570, %4, !spirv.Decorations !898
  %2572 = fadd reassoc nsz arcp contract float %2560, %2571, !spirv.Decorations !898
  %2573 = add i64 %.in, %450
  %2574 = inttoptr i64 %2573 to float addrspace(4)*
  %2575 = addrspacecast float addrspace(4)* %2574 to float addrspace(1)*
  store float %2572, float addrspace(1)* %2575, align 4
  br label %._crit_edge70.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.5:                                  ; preds = %.preheader1.4.._crit_edge70.5_crit_edge, %2565, %2561
  br i1 %188, label %2576, label %._crit_edge70.5.._crit_edge70.1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.5.._crit_edge70.1.5_crit_edge:      ; preds = %._crit_edge70.5
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2576:                                             ; preds = %._crit_edge70.5
  %2577 = fmul reassoc nsz arcp contract float %.sroa.86.0, %1, !spirv.Decorations !898
  br i1 %78, label %2582, label %2578, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2578:                                             ; preds = %2576
  %2579 = add i64 %.in, %452
  %2580 = inttoptr i64 %2579 to float addrspace(4)*
  %2581 = addrspacecast float addrspace(4)* %2580 to float addrspace(1)*
  store float %2577, float addrspace(1)* %2581, align 4
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2582:                                             ; preds = %2576
  %2583 = add i64 %.in3821, %sink_3859
  %2584 = add i64 %2583, %sink_3847
  %2585 = inttoptr i64 %2584 to float addrspace(4)*
  %2586 = addrspacecast float addrspace(4)* %2585 to float addrspace(1)*
  %2587 = load float, float addrspace(1)* %2586, align 4
  %2588 = fmul reassoc nsz arcp contract float %2587, %4, !spirv.Decorations !898
  %2589 = fadd reassoc nsz arcp contract float %2577, %2588, !spirv.Decorations !898
  %2590 = add i64 %.in, %452
  %2591 = inttoptr i64 %2590 to float addrspace(4)*
  %2592 = addrspacecast float addrspace(4)* %2591 to float addrspace(1)*
  store float %2589, float addrspace(1)* %2592, align 4
  br label %._crit_edge70.1.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.5:                                ; preds = %._crit_edge70.5.._crit_edge70.1.5_crit_edge, %2582, %2578
  br i1 %191, label %2593, label %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.5.._crit_edge70.2.5_crit_edge:    ; preds = %._crit_edge70.1.5
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2593:                                             ; preds = %._crit_edge70.1.5
  %2594 = fmul reassoc nsz arcp contract float %.sroa.150.0, %1, !spirv.Decorations !898
  br i1 %78, label %2599, label %2595, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2595:                                             ; preds = %2593
  %2596 = add i64 %.in, %454
  %2597 = inttoptr i64 %2596 to float addrspace(4)*
  %2598 = addrspacecast float addrspace(4)* %2597 to float addrspace(1)*
  store float %2594, float addrspace(1)* %2598, align 4
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2599:                                             ; preds = %2593
  %2600 = add i64 %.in3821, %sink_3858
  %2601 = add i64 %2600, %sink_3847
  %2602 = inttoptr i64 %2601 to float addrspace(4)*
  %2603 = addrspacecast float addrspace(4)* %2602 to float addrspace(1)*
  %2604 = load float, float addrspace(1)* %2603, align 4
  %2605 = fmul reassoc nsz arcp contract float %2604, %4, !spirv.Decorations !898
  %2606 = fadd reassoc nsz arcp contract float %2594, %2605, !spirv.Decorations !898
  %2607 = add i64 %.in, %454
  %2608 = inttoptr i64 %2607 to float addrspace(4)*
  %2609 = addrspacecast float addrspace(4)* %2608 to float addrspace(1)*
  store float %2606, float addrspace(1)* %2609, align 4
  br label %._crit_edge70.2.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.5:                                ; preds = %._crit_edge70.1.5.._crit_edge70.2.5_crit_edge, %2599, %2595
  br i1 %194, label %2610, label %._crit_edge70.2.5..preheader1.5_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.5..preheader1.5_crit_edge:        ; preds = %._crit_edge70.2.5
  br label %.preheader1.5, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2610:                                             ; preds = %._crit_edge70.2.5
  %2611 = fmul reassoc nsz arcp contract float %.sroa.214.0, %1, !spirv.Decorations !898
  br i1 %78, label %2616, label %2612, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2612:                                             ; preds = %2610
  %2613 = add i64 %.in, %456
  %2614 = inttoptr i64 %2613 to float addrspace(4)*
  %2615 = addrspacecast float addrspace(4)* %2614 to float addrspace(1)*
  store float %2611, float addrspace(1)* %2615, align 4
  br label %.preheader1.5, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2616:                                             ; preds = %2610
  %2617 = add i64 %.in3821, %sink_3857
  %2618 = add i64 %2617, %sink_3847
  %2619 = inttoptr i64 %2618 to float addrspace(4)*
  %2620 = addrspacecast float addrspace(4)* %2619 to float addrspace(1)*
  %2621 = load float, float addrspace(1)* %2620, align 4
  %2622 = fmul reassoc nsz arcp contract float %2621, %4, !spirv.Decorations !898
  %2623 = fadd reassoc nsz arcp contract float %2611, %2622, !spirv.Decorations !898
  %2624 = add i64 %.in, %456
  %2625 = inttoptr i64 %2624 to float addrspace(4)*
  %2626 = addrspacecast float addrspace(4)* %2625 to float addrspace(1)*
  store float %2623, float addrspace(1)* %2626, align 4
  br label %.preheader1.5, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.5:                                    ; preds = %._crit_edge70.2.5..preheader1.5_crit_edge, %2616, %2612
  %sink_3845 = shl nsw i64 %355, 2
  br i1 %198, label %2627, label %.preheader1.5.._crit_edge70.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.5.._crit_edge70.6_crit_edge:          ; preds = %.preheader1.5
  br label %._crit_edge70.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2627:                                             ; preds = %.preheader1.5
  %2628 = fmul reassoc nsz arcp contract float %.sroa.26.0, %1, !spirv.Decorations !898
  br i1 %78, label %2633, label %2629, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2629:                                             ; preds = %2627
  %2630 = add i64 %.in, %458
  %2631 = inttoptr i64 %2630 to float addrspace(4)*
  %2632 = addrspacecast float addrspace(4)* %2631 to float addrspace(1)*
  store float %2628, float addrspace(1)* %2632, align 4
  br label %._crit_edge70.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2633:                                             ; preds = %2627
  %2634 = add i64 %.in3821, %sink_3862
  %2635 = add i64 %2634, %sink_3845
  %2636 = inttoptr i64 %2635 to float addrspace(4)*
  %2637 = addrspacecast float addrspace(4)* %2636 to float addrspace(1)*
  %2638 = load float, float addrspace(1)* %2637, align 4
  %2639 = fmul reassoc nsz arcp contract float %2638, %4, !spirv.Decorations !898
  %2640 = fadd reassoc nsz arcp contract float %2628, %2639, !spirv.Decorations !898
  %2641 = add i64 %.in, %458
  %2642 = inttoptr i64 %2641 to float addrspace(4)*
  %2643 = addrspacecast float addrspace(4)* %2642 to float addrspace(1)*
  store float %2640, float addrspace(1)* %2643, align 4
  br label %._crit_edge70.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.6:                                  ; preds = %.preheader1.5.._crit_edge70.6_crit_edge, %2633, %2629
  br i1 %201, label %2644, label %._crit_edge70.6.._crit_edge70.1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.6.._crit_edge70.1.6_crit_edge:      ; preds = %._crit_edge70.6
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2644:                                             ; preds = %._crit_edge70.6
  %2645 = fmul reassoc nsz arcp contract float %.sroa.90.0, %1, !spirv.Decorations !898
  br i1 %78, label %2650, label %2646, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2646:                                             ; preds = %2644
  %2647 = add i64 %.in, %460
  %2648 = inttoptr i64 %2647 to float addrspace(4)*
  %2649 = addrspacecast float addrspace(4)* %2648 to float addrspace(1)*
  store float %2645, float addrspace(1)* %2649, align 4
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2650:                                             ; preds = %2644
  %2651 = add i64 %.in3821, %sink_3859
  %2652 = add i64 %2651, %sink_3845
  %2653 = inttoptr i64 %2652 to float addrspace(4)*
  %2654 = addrspacecast float addrspace(4)* %2653 to float addrspace(1)*
  %2655 = load float, float addrspace(1)* %2654, align 4
  %2656 = fmul reassoc nsz arcp contract float %2655, %4, !spirv.Decorations !898
  %2657 = fadd reassoc nsz arcp contract float %2645, %2656, !spirv.Decorations !898
  %2658 = add i64 %.in, %460
  %2659 = inttoptr i64 %2658 to float addrspace(4)*
  %2660 = addrspacecast float addrspace(4)* %2659 to float addrspace(1)*
  store float %2657, float addrspace(1)* %2660, align 4
  br label %._crit_edge70.1.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.6:                                ; preds = %._crit_edge70.6.._crit_edge70.1.6_crit_edge, %2650, %2646
  br i1 %204, label %2661, label %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.6.._crit_edge70.2.6_crit_edge:    ; preds = %._crit_edge70.1.6
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2661:                                             ; preds = %._crit_edge70.1.6
  %2662 = fmul reassoc nsz arcp contract float %.sroa.154.0, %1, !spirv.Decorations !898
  br i1 %78, label %2667, label %2663, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2663:                                             ; preds = %2661
  %2664 = add i64 %.in, %462
  %2665 = inttoptr i64 %2664 to float addrspace(4)*
  %2666 = addrspacecast float addrspace(4)* %2665 to float addrspace(1)*
  store float %2662, float addrspace(1)* %2666, align 4
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2667:                                             ; preds = %2661
  %2668 = add i64 %.in3821, %sink_3858
  %2669 = add i64 %2668, %sink_3845
  %2670 = inttoptr i64 %2669 to float addrspace(4)*
  %2671 = addrspacecast float addrspace(4)* %2670 to float addrspace(1)*
  %2672 = load float, float addrspace(1)* %2671, align 4
  %2673 = fmul reassoc nsz arcp contract float %2672, %4, !spirv.Decorations !898
  %2674 = fadd reassoc nsz arcp contract float %2662, %2673, !spirv.Decorations !898
  %2675 = add i64 %.in, %462
  %2676 = inttoptr i64 %2675 to float addrspace(4)*
  %2677 = addrspacecast float addrspace(4)* %2676 to float addrspace(1)*
  store float %2674, float addrspace(1)* %2677, align 4
  br label %._crit_edge70.2.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.6:                                ; preds = %._crit_edge70.1.6.._crit_edge70.2.6_crit_edge, %2667, %2663
  br i1 %207, label %2678, label %._crit_edge70.2.6..preheader1.6_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.6..preheader1.6_crit_edge:        ; preds = %._crit_edge70.2.6
  br label %.preheader1.6, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2678:                                             ; preds = %._crit_edge70.2.6
  %2679 = fmul reassoc nsz arcp contract float %.sroa.218.0, %1, !spirv.Decorations !898
  br i1 %78, label %2684, label %2680, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2680:                                             ; preds = %2678
  %2681 = add i64 %.in, %464
  %2682 = inttoptr i64 %2681 to float addrspace(4)*
  %2683 = addrspacecast float addrspace(4)* %2682 to float addrspace(1)*
  store float %2679, float addrspace(1)* %2683, align 4
  br label %.preheader1.6, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2684:                                             ; preds = %2678
  %2685 = add i64 %.in3821, %sink_3857
  %2686 = add i64 %2685, %sink_3845
  %2687 = inttoptr i64 %2686 to float addrspace(4)*
  %2688 = addrspacecast float addrspace(4)* %2687 to float addrspace(1)*
  %2689 = load float, float addrspace(1)* %2688, align 4
  %2690 = fmul reassoc nsz arcp contract float %2689, %4, !spirv.Decorations !898
  %2691 = fadd reassoc nsz arcp contract float %2679, %2690, !spirv.Decorations !898
  %2692 = add i64 %.in, %464
  %2693 = inttoptr i64 %2692 to float addrspace(4)*
  %2694 = addrspacecast float addrspace(4)* %2693 to float addrspace(1)*
  store float %2691, float addrspace(1)* %2694, align 4
  br label %.preheader1.6, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.6:                                    ; preds = %._crit_edge70.2.6..preheader1.6_crit_edge, %2684, %2680
  %sink_3843 = shl nsw i64 %356, 2
  br i1 %211, label %2695, label %.preheader1.6.._crit_edge70.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.6.._crit_edge70.7_crit_edge:          ; preds = %.preheader1.6
  br label %._crit_edge70.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2695:                                             ; preds = %.preheader1.6
  %2696 = fmul reassoc nsz arcp contract float %.sroa.30.0, %1, !spirv.Decorations !898
  br i1 %78, label %2701, label %2697, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2697:                                             ; preds = %2695
  %2698 = add i64 %.in, %466
  %2699 = inttoptr i64 %2698 to float addrspace(4)*
  %2700 = addrspacecast float addrspace(4)* %2699 to float addrspace(1)*
  store float %2696, float addrspace(1)* %2700, align 4
  br label %._crit_edge70.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2701:                                             ; preds = %2695
  %2702 = add i64 %.in3821, %sink_3862
  %2703 = add i64 %2702, %sink_3843
  %2704 = inttoptr i64 %2703 to float addrspace(4)*
  %2705 = addrspacecast float addrspace(4)* %2704 to float addrspace(1)*
  %2706 = load float, float addrspace(1)* %2705, align 4
  %2707 = fmul reassoc nsz arcp contract float %2706, %4, !spirv.Decorations !898
  %2708 = fadd reassoc nsz arcp contract float %2696, %2707, !spirv.Decorations !898
  %2709 = add i64 %.in, %466
  %2710 = inttoptr i64 %2709 to float addrspace(4)*
  %2711 = addrspacecast float addrspace(4)* %2710 to float addrspace(1)*
  store float %2708, float addrspace(1)* %2711, align 4
  br label %._crit_edge70.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.7:                                  ; preds = %.preheader1.6.._crit_edge70.7_crit_edge, %2701, %2697
  br i1 %214, label %2712, label %._crit_edge70.7.._crit_edge70.1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.7.._crit_edge70.1.7_crit_edge:      ; preds = %._crit_edge70.7
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2712:                                             ; preds = %._crit_edge70.7
  %2713 = fmul reassoc nsz arcp contract float %.sroa.94.0, %1, !spirv.Decorations !898
  br i1 %78, label %2718, label %2714, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2714:                                             ; preds = %2712
  %2715 = add i64 %.in, %468
  %2716 = inttoptr i64 %2715 to float addrspace(4)*
  %2717 = addrspacecast float addrspace(4)* %2716 to float addrspace(1)*
  store float %2713, float addrspace(1)* %2717, align 4
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2718:                                             ; preds = %2712
  %2719 = add i64 %.in3821, %sink_3859
  %2720 = add i64 %2719, %sink_3843
  %2721 = inttoptr i64 %2720 to float addrspace(4)*
  %2722 = addrspacecast float addrspace(4)* %2721 to float addrspace(1)*
  %2723 = load float, float addrspace(1)* %2722, align 4
  %2724 = fmul reassoc nsz arcp contract float %2723, %4, !spirv.Decorations !898
  %2725 = fadd reassoc nsz arcp contract float %2713, %2724, !spirv.Decorations !898
  %2726 = add i64 %.in, %468
  %2727 = inttoptr i64 %2726 to float addrspace(4)*
  %2728 = addrspacecast float addrspace(4)* %2727 to float addrspace(1)*
  store float %2725, float addrspace(1)* %2728, align 4
  br label %._crit_edge70.1.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.7:                                ; preds = %._crit_edge70.7.._crit_edge70.1.7_crit_edge, %2718, %2714
  br i1 %217, label %2729, label %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.7.._crit_edge70.2.7_crit_edge:    ; preds = %._crit_edge70.1.7
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2729:                                             ; preds = %._crit_edge70.1.7
  %2730 = fmul reassoc nsz arcp contract float %.sroa.158.0, %1, !spirv.Decorations !898
  br i1 %78, label %2735, label %2731, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2731:                                             ; preds = %2729
  %2732 = add i64 %.in, %470
  %2733 = inttoptr i64 %2732 to float addrspace(4)*
  %2734 = addrspacecast float addrspace(4)* %2733 to float addrspace(1)*
  store float %2730, float addrspace(1)* %2734, align 4
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2735:                                             ; preds = %2729
  %2736 = add i64 %.in3821, %sink_3858
  %2737 = add i64 %2736, %sink_3843
  %2738 = inttoptr i64 %2737 to float addrspace(4)*
  %2739 = addrspacecast float addrspace(4)* %2738 to float addrspace(1)*
  %2740 = load float, float addrspace(1)* %2739, align 4
  %2741 = fmul reassoc nsz arcp contract float %2740, %4, !spirv.Decorations !898
  %2742 = fadd reassoc nsz arcp contract float %2730, %2741, !spirv.Decorations !898
  %2743 = add i64 %.in, %470
  %2744 = inttoptr i64 %2743 to float addrspace(4)*
  %2745 = addrspacecast float addrspace(4)* %2744 to float addrspace(1)*
  store float %2742, float addrspace(1)* %2745, align 4
  br label %._crit_edge70.2.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.7:                                ; preds = %._crit_edge70.1.7.._crit_edge70.2.7_crit_edge, %2735, %2731
  br i1 %220, label %2746, label %._crit_edge70.2.7..preheader1.7_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.7..preheader1.7_crit_edge:        ; preds = %._crit_edge70.2.7
  br label %.preheader1.7, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2746:                                             ; preds = %._crit_edge70.2.7
  %2747 = fmul reassoc nsz arcp contract float %.sroa.222.0, %1, !spirv.Decorations !898
  br i1 %78, label %2752, label %2748, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2748:                                             ; preds = %2746
  %2749 = add i64 %.in, %472
  %2750 = inttoptr i64 %2749 to float addrspace(4)*
  %2751 = addrspacecast float addrspace(4)* %2750 to float addrspace(1)*
  store float %2747, float addrspace(1)* %2751, align 4
  br label %.preheader1.7, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2752:                                             ; preds = %2746
  %2753 = add i64 %.in3821, %sink_3857
  %2754 = add i64 %2753, %sink_3843
  %2755 = inttoptr i64 %2754 to float addrspace(4)*
  %2756 = addrspacecast float addrspace(4)* %2755 to float addrspace(1)*
  %2757 = load float, float addrspace(1)* %2756, align 4
  %2758 = fmul reassoc nsz arcp contract float %2757, %4, !spirv.Decorations !898
  %2759 = fadd reassoc nsz arcp contract float %2747, %2758, !spirv.Decorations !898
  %2760 = add i64 %.in, %472
  %2761 = inttoptr i64 %2760 to float addrspace(4)*
  %2762 = addrspacecast float addrspace(4)* %2761 to float addrspace(1)*
  store float %2759, float addrspace(1)* %2762, align 4
  br label %.preheader1.7, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.7:                                    ; preds = %._crit_edge70.2.7..preheader1.7_crit_edge, %2752, %2748
  %sink_3841 = shl nsw i64 %357, 2
  br i1 %224, label %2763, label %.preheader1.7.._crit_edge70.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.7.._crit_edge70.8_crit_edge:          ; preds = %.preheader1.7
  br label %._crit_edge70.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2763:                                             ; preds = %.preheader1.7
  %2764 = fmul reassoc nsz arcp contract float %.sroa.34.0, %1, !spirv.Decorations !898
  br i1 %78, label %2769, label %2765, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2765:                                             ; preds = %2763
  %2766 = add i64 %.in, %474
  %2767 = inttoptr i64 %2766 to float addrspace(4)*
  %2768 = addrspacecast float addrspace(4)* %2767 to float addrspace(1)*
  store float %2764, float addrspace(1)* %2768, align 4
  br label %._crit_edge70.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2769:                                             ; preds = %2763
  %2770 = add i64 %.in3821, %sink_3862
  %2771 = add i64 %2770, %sink_3841
  %2772 = inttoptr i64 %2771 to float addrspace(4)*
  %2773 = addrspacecast float addrspace(4)* %2772 to float addrspace(1)*
  %2774 = load float, float addrspace(1)* %2773, align 4
  %2775 = fmul reassoc nsz arcp contract float %2774, %4, !spirv.Decorations !898
  %2776 = fadd reassoc nsz arcp contract float %2764, %2775, !spirv.Decorations !898
  %2777 = add i64 %.in, %474
  %2778 = inttoptr i64 %2777 to float addrspace(4)*
  %2779 = addrspacecast float addrspace(4)* %2778 to float addrspace(1)*
  store float %2776, float addrspace(1)* %2779, align 4
  br label %._crit_edge70.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.8:                                  ; preds = %.preheader1.7.._crit_edge70.8_crit_edge, %2769, %2765
  br i1 %227, label %2780, label %._crit_edge70.8.._crit_edge70.1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.8.._crit_edge70.1.8_crit_edge:      ; preds = %._crit_edge70.8
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2780:                                             ; preds = %._crit_edge70.8
  %2781 = fmul reassoc nsz arcp contract float %.sroa.98.0, %1, !spirv.Decorations !898
  br i1 %78, label %2786, label %2782, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2782:                                             ; preds = %2780
  %2783 = add i64 %.in, %476
  %2784 = inttoptr i64 %2783 to float addrspace(4)*
  %2785 = addrspacecast float addrspace(4)* %2784 to float addrspace(1)*
  store float %2781, float addrspace(1)* %2785, align 4
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2786:                                             ; preds = %2780
  %2787 = add i64 %.in3821, %sink_3859
  %2788 = add i64 %2787, %sink_3841
  %2789 = inttoptr i64 %2788 to float addrspace(4)*
  %2790 = addrspacecast float addrspace(4)* %2789 to float addrspace(1)*
  %2791 = load float, float addrspace(1)* %2790, align 4
  %2792 = fmul reassoc nsz arcp contract float %2791, %4, !spirv.Decorations !898
  %2793 = fadd reassoc nsz arcp contract float %2781, %2792, !spirv.Decorations !898
  %2794 = add i64 %.in, %476
  %2795 = inttoptr i64 %2794 to float addrspace(4)*
  %2796 = addrspacecast float addrspace(4)* %2795 to float addrspace(1)*
  store float %2793, float addrspace(1)* %2796, align 4
  br label %._crit_edge70.1.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.8:                                ; preds = %._crit_edge70.8.._crit_edge70.1.8_crit_edge, %2786, %2782
  br i1 %230, label %2797, label %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.8.._crit_edge70.2.8_crit_edge:    ; preds = %._crit_edge70.1.8
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2797:                                             ; preds = %._crit_edge70.1.8
  %2798 = fmul reassoc nsz arcp contract float %.sroa.162.0, %1, !spirv.Decorations !898
  br i1 %78, label %2803, label %2799, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2799:                                             ; preds = %2797
  %2800 = add i64 %.in, %478
  %2801 = inttoptr i64 %2800 to float addrspace(4)*
  %2802 = addrspacecast float addrspace(4)* %2801 to float addrspace(1)*
  store float %2798, float addrspace(1)* %2802, align 4
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2803:                                             ; preds = %2797
  %2804 = add i64 %.in3821, %sink_3858
  %2805 = add i64 %2804, %sink_3841
  %2806 = inttoptr i64 %2805 to float addrspace(4)*
  %2807 = addrspacecast float addrspace(4)* %2806 to float addrspace(1)*
  %2808 = load float, float addrspace(1)* %2807, align 4
  %2809 = fmul reassoc nsz arcp contract float %2808, %4, !spirv.Decorations !898
  %2810 = fadd reassoc nsz arcp contract float %2798, %2809, !spirv.Decorations !898
  %2811 = add i64 %.in, %478
  %2812 = inttoptr i64 %2811 to float addrspace(4)*
  %2813 = addrspacecast float addrspace(4)* %2812 to float addrspace(1)*
  store float %2810, float addrspace(1)* %2813, align 4
  br label %._crit_edge70.2.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.8:                                ; preds = %._crit_edge70.1.8.._crit_edge70.2.8_crit_edge, %2803, %2799
  br i1 %233, label %2814, label %._crit_edge70.2.8..preheader1.8_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.8..preheader1.8_crit_edge:        ; preds = %._crit_edge70.2.8
  br label %.preheader1.8, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2814:                                             ; preds = %._crit_edge70.2.8
  %2815 = fmul reassoc nsz arcp contract float %.sroa.226.0, %1, !spirv.Decorations !898
  br i1 %78, label %2820, label %2816, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2816:                                             ; preds = %2814
  %2817 = add i64 %.in, %480
  %2818 = inttoptr i64 %2817 to float addrspace(4)*
  %2819 = addrspacecast float addrspace(4)* %2818 to float addrspace(1)*
  store float %2815, float addrspace(1)* %2819, align 4
  br label %.preheader1.8, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2820:                                             ; preds = %2814
  %2821 = add i64 %.in3821, %sink_3857
  %2822 = add i64 %2821, %sink_3841
  %2823 = inttoptr i64 %2822 to float addrspace(4)*
  %2824 = addrspacecast float addrspace(4)* %2823 to float addrspace(1)*
  %2825 = load float, float addrspace(1)* %2824, align 4
  %2826 = fmul reassoc nsz arcp contract float %2825, %4, !spirv.Decorations !898
  %2827 = fadd reassoc nsz arcp contract float %2815, %2826, !spirv.Decorations !898
  %2828 = add i64 %.in, %480
  %2829 = inttoptr i64 %2828 to float addrspace(4)*
  %2830 = addrspacecast float addrspace(4)* %2829 to float addrspace(1)*
  store float %2827, float addrspace(1)* %2830, align 4
  br label %.preheader1.8, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.8:                                    ; preds = %._crit_edge70.2.8..preheader1.8_crit_edge, %2820, %2816
  %sink_3839 = shl nsw i64 %358, 2
  br i1 %237, label %2831, label %.preheader1.8.._crit_edge70.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.8.._crit_edge70.9_crit_edge:          ; preds = %.preheader1.8
  br label %._crit_edge70.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2831:                                             ; preds = %.preheader1.8
  %2832 = fmul reassoc nsz arcp contract float %.sroa.38.0, %1, !spirv.Decorations !898
  br i1 %78, label %2837, label %2833, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2833:                                             ; preds = %2831
  %2834 = add i64 %.in, %482
  %2835 = inttoptr i64 %2834 to float addrspace(4)*
  %2836 = addrspacecast float addrspace(4)* %2835 to float addrspace(1)*
  store float %2832, float addrspace(1)* %2836, align 4
  br label %._crit_edge70.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2837:                                             ; preds = %2831
  %2838 = add i64 %.in3821, %sink_3862
  %2839 = add i64 %2838, %sink_3839
  %2840 = inttoptr i64 %2839 to float addrspace(4)*
  %2841 = addrspacecast float addrspace(4)* %2840 to float addrspace(1)*
  %2842 = load float, float addrspace(1)* %2841, align 4
  %2843 = fmul reassoc nsz arcp contract float %2842, %4, !spirv.Decorations !898
  %2844 = fadd reassoc nsz arcp contract float %2832, %2843, !spirv.Decorations !898
  %2845 = add i64 %.in, %482
  %2846 = inttoptr i64 %2845 to float addrspace(4)*
  %2847 = addrspacecast float addrspace(4)* %2846 to float addrspace(1)*
  store float %2844, float addrspace(1)* %2847, align 4
  br label %._crit_edge70.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.9:                                  ; preds = %.preheader1.8.._crit_edge70.9_crit_edge, %2837, %2833
  br i1 %240, label %2848, label %._crit_edge70.9.._crit_edge70.1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.9.._crit_edge70.1.9_crit_edge:      ; preds = %._crit_edge70.9
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2848:                                             ; preds = %._crit_edge70.9
  %2849 = fmul reassoc nsz arcp contract float %.sroa.102.0, %1, !spirv.Decorations !898
  br i1 %78, label %2854, label %2850, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2850:                                             ; preds = %2848
  %2851 = add i64 %.in, %484
  %2852 = inttoptr i64 %2851 to float addrspace(4)*
  %2853 = addrspacecast float addrspace(4)* %2852 to float addrspace(1)*
  store float %2849, float addrspace(1)* %2853, align 4
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2854:                                             ; preds = %2848
  %2855 = add i64 %.in3821, %sink_3859
  %2856 = add i64 %2855, %sink_3839
  %2857 = inttoptr i64 %2856 to float addrspace(4)*
  %2858 = addrspacecast float addrspace(4)* %2857 to float addrspace(1)*
  %2859 = load float, float addrspace(1)* %2858, align 4
  %2860 = fmul reassoc nsz arcp contract float %2859, %4, !spirv.Decorations !898
  %2861 = fadd reassoc nsz arcp contract float %2849, %2860, !spirv.Decorations !898
  %2862 = add i64 %.in, %484
  %2863 = inttoptr i64 %2862 to float addrspace(4)*
  %2864 = addrspacecast float addrspace(4)* %2863 to float addrspace(1)*
  store float %2861, float addrspace(1)* %2864, align 4
  br label %._crit_edge70.1.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.9:                                ; preds = %._crit_edge70.9.._crit_edge70.1.9_crit_edge, %2854, %2850
  br i1 %243, label %2865, label %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.9.._crit_edge70.2.9_crit_edge:    ; preds = %._crit_edge70.1.9
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2865:                                             ; preds = %._crit_edge70.1.9
  %2866 = fmul reassoc nsz arcp contract float %.sroa.166.0, %1, !spirv.Decorations !898
  br i1 %78, label %2871, label %2867, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2867:                                             ; preds = %2865
  %2868 = add i64 %.in, %486
  %2869 = inttoptr i64 %2868 to float addrspace(4)*
  %2870 = addrspacecast float addrspace(4)* %2869 to float addrspace(1)*
  store float %2866, float addrspace(1)* %2870, align 4
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2871:                                             ; preds = %2865
  %2872 = add i64 %.in3821, %sink_3858
  %2873 = add i64 %2872, %sink_3839
  %2874 = inttoptr i64 %2873 to float addrspace(4)*
  %2875 = addrspacecast float addrspace(4)* %2874 to float addrspace(1)*
  %2876 = load float, float addrspace(1)* %2875, align 4
  %2877 = fmul reassoc nsz arcp contract float %2876, %4, !spirv.Decorations !898
  %2878 = fadd reassoc nsz arcp contract float %2866, %2877, !spirv.Decorations !898
  %2879 = add i64 %.in, %486
  %2880 = inttoptr i64 %2879 to float addrspace(4)*
  %2881 = addrspacecast float addrspace(4)* %2880 to float addrspace(1)*
  store float %2878, float addrspace(1)* %2881, align 4
  br label %._crit_edge70.2.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.9:                                ; preds = %._crit_edge70.1.9.._crit_edge70.2.9_crit_edge, %2871, %2867
  br i1 %246, label %2882, label %._crit_edge70.2.9..preheader1.9_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.9..preheader1.9_crit_edge:        ; preds = %._crit_edge70.2.9
  br label %.preheader1.9, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2882:                                             ; preds = %._crit_edge70.2.9
  %2883 = fmul reassoc nsz arcp contract float %.sroa.230.0, %1, !spirv.Decorations !898
  br i1 %78, label %2888, label %2884, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2884:                                             ; preds = %2882
  %2885 = add i64 %.in, %488
  %2886 = inttoptr i64 %2885 to float addrspace(4)*
  %2887 = addrspacecast float addrspace(4)* %2886 to float addrspace(1)*
  store float %2883, float addrspace(1)* %2887, align 4
  br label %.preheader1.9, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2888:                                             ; preds = %2882
  %2889 = add i64 %.in3821, %sink_3857
  %2890 = add i64 %2889, %sink_3839
  %2891 = inttoptr i64 %2890 to float addrspace(4)*
  %2892 = addrspacecast float addrspace(4)* %2891 to float addrspace(1)*
  %2893 = load float, float addrspace(1)* %2892, align 4
  %2894 = fmul reassoc nsz arcp contract float %2893, %4, !spirv.Decorations !898
  %2895 = fadd reassoc nsz arcp contract float %2883, %2894, !spirv.Decorations !898
  %2896 = add i64 %.in, %488
  %2897 = inttoptr i64 %2896 to float addrspace(4)*
  %2898 = addrspacecast float addrspace(4)* %2897 to float addrspace(1)*
  store float %2895, float addrspace(1)* %2898, align 4
  br label %.preheader1.9, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.9:                                    ; preds = %._crit_edge70.2.9..preheader1.9_crit_edge, %2888, %2884
  %sink_3837 = shl nsw i64 %359, 2
  br i1 %250, label %2899, label %.preheader1.9.._crit_edge70.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.9.._crit_edge70.10_crit_edge:         ; preds = %.preheader1.9
  br label %._crit_edge70.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2899:                                             ; preds = %.preheader1.9
  %2900 = fmul reassoc nsz arcp contract float %.sroa.42.0, %1, !spirv.Decorations !898
  br i1 %78, label %2905, label %2901, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2901:                                             ; preds = %2899
  %2902 = add i64 %.in, %490
  %2903 = inttoptr i64 %2902 to float addrspace(4)*
  %2904 = addrspacecast float addrspace(4)* %2903 to float addrspace(1)*
  store float %2900, float addrspace(1)* %2904, align 4
  br label %._crit_edge70.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2905:                                             ; preds = %2899
  %2906 = add i64 %.in3821, %sink_3862
  %2907 = add i64 %2906, %sink_3837
  %2908 = inttoptr i64 %2907 to float addrspace(4)*
  %2909 = addrspacecast float addrspace(4)* %2908 to float addrspace(1)*
  %2910 = load float, float addrspace(1)* %2909, align 4
  %2911 = fmul reassoc nsz arcp contract float %2910, %4, !spirv.Decorations !898
  %2912 = fadd reassoc nsz arcp contract float %2900, %2911, !spirv.Decorations !898
  %2913 = add i64 %.in, %490
  %2914 = inttoptr i64 %2913 to float addrspace(4)*
  %2915 = addrspacecast float addrspace(4)* %2914 to float addrspace(1)*
  store float %2912, float addrspace(1)* %2915, align 4
  br label %._crit_edge70.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.10:                                 ; preds = %.preheader1.9.._crit_edge70.10_crit_edge, %2905, %2901
  br i1 %253, label %2916, label %._crit_edge70.10.._crit_edge70.1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.10.._crit_edge70.1.10_crit_edge:    ; preds = %._crit_edge70.10
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2916:                                             ; preds = %._crit_edge70.10
  %2917 = fmul reassoc nsz arcp contract float %.sroa.106.0, %1, !spirv.Decorations !898
  br i1 %78, label %2922, label %2918, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2918:                                             ; preds = %2916
  %2919 = add i64 %.in, %492
  %2920 = inttoptr i64 %2919 to float addrspace(4)*
  %2921 = addrspacecast float addrspace(4)* %2920 to float addrspace(1)*
  store float %2917, float addrspace(1)* %2921, align 4
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2922:                                             ; preds = %2916
  %2923 = add i64 %.in3821, %sink_3859
  %2924 = add i64 %2923, %sink_3837
  %2925 = inttoptr i64 %2924 to float addrspace(4)*
  %2926 = addrspacecast float addrspace(4)* %2925 to float addrspace(1)*
  %2927 = load float, float addrspace(1)* %2926, align 4
  %2928 = fmul reassoc nsz arcp contract float %2927, %4, !spirv.Decorations !898
  %2929 = fadd reassoc nsz arcp contract float %2917, %2928, !spirv.Decorations !898
  %2930 = add i64 %.in, %492
  %2931 = inttoptr i64 %2930 to float addrspace(4)*
  %2932 = addrspacecast float addrspace(4)* %2931 to float addrspace(1)*
  store float %2929, float addrspace(1)* %2932, align 4
  br label %._crit_edge70.1.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.10:                               ; preds = %._crit_edge70.10.._crit_edge70.1.10_crit_edge, %2922, %2918
  br i1 %256, label %2933, label %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.10.._crit_edge70.2.10_crit_edge:  ; preds = %._crit_edge70.1.10
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2933:                                             ; preds = %._crit_edge70.1.10
  %2934 = fmul reassoc nsz arcp contract float %.sroa.170.0, %1, !spirv.Decorations !898
  br i1 %78, label %2939, label %2935, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2935:                                             ; preds = %2933
  %2936 = add i64 %.in, %494
  %2937 = inttoptr i64 %2936 to float addrspace(4)*
  %2938 = addrspacecast float addrspace(4)* %2937 to float addrspace(1)*
  store float %2934, float addrspace(1)* %2938, align 4
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2939:                                             ; preds = %2933
  %2940 = add i64 %.in3821, %sink_3858
  %2941 = add i64 %2940, %sink_3837
  %2942 = inttoptr i64 %2941 to float addrspace(4)*
  %2943 = addrspacecast float addrspace(4)* %2942 to float addrspace(1)*
  %2944 = load float, float addrspace(1)* %2943, align 4
  %2945 = fmul reassoc nsz arcp contract float %2944, %4, !spirv.Decorations !898
  %2946 = fadd reassoc nsz arcp contract float %2934, %2945, !spirv.Decorations !898
  %2947 = add i64 %.in, %494
  %2948 = inttoptr i64 %2947 to float addrspace(4)*
  %2949 = addrspacecast float addrspace(4)* %2948 to float addrspace(1)*
  store float %2946, float addrspace(1)* %2949, align 4
  br label %._crit_edge70.2.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.10:                               ; preds = %._crit_edge70.1.10.._crit_edge70.2.10_crit_edge, %2939, %2935
  br i1 %259, label %2950, label %._crit_edge70.2.10..preheader1.10_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.10..preheader1.10_crit_edge:      ; preds = %._crit_edge70.2.10
  br label %.preheader1.10, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2950:                                             ; preds = %._crit_edge70.2.10
  %2951 = fmul reassoc nsz arcp contract float %.sroa.234.0, %1, !spirv.Decorations !898
  br i1 %78, label %2956, label %2952, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2952:                                             ; preds = %2950
  %2953 = add i64 %.in, %496
  %2954 = inttoptr i64 %2953 to float addrspace(4)*
  %2955 = addrspacecast float addrspace(4)* %2954 to float addrspace(1)*
  store float %2951, float addrspace(1)* %2955, align 4
  br label %.preheader1.10, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2956:                                             ; preds = %2950
  %2957 = add i64 %.in3821, %sink_3857
  %2958 = add i64 %2957, %sink_3837
  %2959 = inttoptr i64 %2958 to float addrspace(4)*
  %2960 = addrspacecast float addrspace(4)* %2959 to float addrspace(1)*
  %2961 = load float, float addrspace(1)* %2960, align 4
  %2962 = fmul reassoc nsz arcp contract float %2961, %4, !spirv.Decorations !898
  %2963 = fadd reassoc nsz arcp contract float %2951, %2962, !spirv.Decorations !898
  %2964 = add i64 %.in, %496
  %2965 = inttoptr i64 %2964 to float addrspace(4)*
  %2966 = addrspacecast float addrspace(4)* %2965 to float addrspace(1)*
  store float %2963, float addrspace(1)* %2966, align 4
  br label %.preheader1.10, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.10:                                   ; preds = %._crit_edge70.2.10..preheader1.10_crit_edge, %2956, %2952
  %sink_3835 = shl nsw i64 %360, 2
  br i1 %263, label %2967, label %.preheader1.10.._crit_edge70.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.10.._crit_edge70.11_crit_edge:        ; preds = %.preheader1.10
  br label %._crit_edge70.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2967:                                             ; preds = %.preheader1.10
  %2968 = fmul reassoc nsz arcp contract float %.sroa.46.0, %1, !spirv.Decorations !898
  br i1 %78, label %2973, label %2969, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2969:                                             ; preds = %2967
  %2970 = add i64 %.in, %498
  %2971 = inttoptr i64 %2970 to float addrspace(4)*
  %2972 = addrspacecast float addrspace(4)* %2971 to float addrspace(1)*
  store float %2968, float addrspace(1)* %2972, align 4
  br label %._crit_edge70.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2973:                                             ; preds = %2967
  %2974 = add i64 %.in3821, %sink_3862
  %2975 = add i64 %2974, %sink_3835
  %2976 = inttoptr i64 %2975 to float addrspace(4)*
  %2977 = addrspacecast float addrspace(4)* %2976 to float addrspace(1)*
  %2978 = load float, float addrspace(1)* %2977, align 4
  %2979 = fmul reassoc nsz arcp contract float %2978, %4, !spirv.Decorations !898
  %2980 = fadd reassoc nsz arcp contract float %2968, %2979, !spirv.Decorations !898
  %2981 = add i64 %.in, %498
  %2982 = inttoptr i64 %2981 to float addrspace(4)*
  %2983 = addrspacecast float addrspace(4)* %2982 to float addrspace(1)*
  store float %2980, float addrspace(1)* %2983, align 4
  br label %._crit_edge70.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.11:                                 ; preds = %.preheader1.10.._crit_edge70.11_crit_edge, %2973, %2969
  br i1 %266, label %2984, label %._crit_edge70.11.._crit_edge70.1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.11.._crit_edge70.1.11_crit_edge:    ; preds = %._crit_edge70.11
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2984:                                             ; preds = %._crit_edge70.11
  %2985 = fmul reassoc nsz arcp contract float %.sroa.110.0, %1, !spirv.Decorations !898
  br i1 %78, label %2990, label %2986, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

2986:                                             ; preds = %2984
  %2987 = add i64 %.in, %500
  %2988 = inttoptr i64 %2987 to float addrspace(4)*
  %2989 = addrspacecast float addrspace(4)* %2988 to float addrspace(1)*
  store float %2985, float addrspace(1)* %2989, align 4
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

2990:                                             ; preds = %2984
  %2991 = add i64 %.in3821, %sink_3859
  %2992 = add i64 %2991, %sink_3835
  %2993 = inttoptr i64 %2992 to float addrspace(4)*
  %2994 = addrspacecast float addrspace(4)* %2993 to float addrspace(1)*
  %2995 = load float, float addrspace(1)* %2994, align 4
  %2996 = fmul reassoc nsz arcp contract float %2995, %4, !spirv.Decorations !898
  %2997 = fadd reassoc nsz arcp contract float %2985, %2996, !spirv.Decorations !898
  %2998 = add i64 %.in, %500
  %2999 = inttoptr i64 %2998 to float addrspace(4)*
  %3000 = addrspacecast float addrspace(4)* %2999 to float addrspace(1)*
  store float %2997, float addrspace(1)* %3000, align 4
  br label %._crit_edge70.1.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.11:                               ; preds = %._crit_edge70.11.._crit_edge70.1.11_crit_edge, %2990, %2986
  br i1 %269, label %3001, label %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.11.._crit_edge70.2.11_crit_edge:  ; preds = %._crit_edge70.1.11
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3001:                                             ; preds = %._crit_edge70.1.11
  %3002 = fmul reassoc nsz arcp contract float %.sroa.174.0, %1, !spirv.Decorations !898
  br i1 %78, label %3007, label %3003, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3003:                                             ; preds = %3001
  %3004 = add i64 %.in, %502
  %3005 = inttoptr i64 %3004 to float addrspace(4)*
  %3006 = addrspacecast float addrspace(4)* %3005 to float addrspace(1)*
  store float %3002, float addrspace(1)* %3006, align 4
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3007:                                             ; preds = %3001
  %3008 = add i64 %.in3821, %sink_3858
  %3009 = add i64 %3008, %sink_3835
  %3010 = inttoptr i64 %3009 to float addrspace(4)*
  %3011 = addrspacecast float addrspace(4)* %3010 to float addrspace(1)*
  %3012 = load float, float addrspace(1)* %3011, align 4
  %3013 = fmul reassoc nsz arcp contract float %3012, %4, !spirv.Decorations !898
  %3014 = fadd reassoc nsz arcp contract float %3002, %3013, !spirv.Decorations !898
  %3015 = add i64 %.in, %502
  %3016 = inttoptr i64 %3015 to float addrspace(4)*
  %3017 = addrspacecast float addrspace(4)* %3016 to float addrspace(1)*
  store float %3014, float addrspace(1)* %3017, align 4
  br label %._crit_edge70.2.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.11:                               ; preds = %._crit_edge70.1.11.._crit_edge70.2.11_crit_edge, %3007, %3003
  br i1 %272, label %3018, label %._crit_edge70.2.11..preheader1.11_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.11..preheader1.11_crit_edge:      ; preds = %._crit_edge70.2.11
  br label %.preheader1.11, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3018:                                             ; preds = %._crit_edge70.2.11
  %3019 = fmul reassoc nsz arcp contract float %.sroa.238.0, %1, !spirv.Decorations !898
  br i1 %78, label %3024, label %3020, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3020:                                             ; preds = %3018
  %3021 = add i64 %.in, %504
  %3022 = inttoptr i64 %3021 to float addrspace(4)*
  %3023 = addrspacecast float addrspace(4)* %3022 to float addrspace(1)*
  store float %3019, float addrspace(1)* %3023, align 4
  br label %.preheader1.11, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3024:                                             ; preds = %3018
  %3025 = add i64 %.in3821, %sink_3857
  %3026 = add i64 %3025, %sink_3835
  %3027 = inttoptr i64 %3026 to float addrspace(4)*
  %3028 = addrspacecast float addrspace(4)* %3027 to float addrspace(1)*
  %3029 = load float, float addrspace(1)* %3028, align 4
  %3030 = fmul reassoc nsz arcp contract float %3029, %4, !spirv.Decorations !898
  %3031 = fadd reassoc nsz arcp contract float %3019, %3030, !spirv.Decorations !898
  %3032 = add i64 %.in, %504
  %3033 = inttoptr i64 %3032 to float addrspace(4)*
  %3034 = addrspacecast float addrspace(4)* %3033 to float addrspace(1)*
  store float %3031, float addrspace(1)* %3034, align 4
  br label %.preheader1.11, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.11:                                   ; preds = %._crit_edge70.2.11..preheader1.11_crit_edge, %3024, %3020
  %sink_3833 = shl nsw i64 %361, 2
  br i1 %276, label %3035, label %.preheader1.11.._crit_edge70.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.11.._crit_edge70.12_crit_edge:        ; preds = %.preheader1.11
  br label %._crit_edge70.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3035:                                             ; preds = %.preheader1.11
  %3036 = fmul reassoc nsz arcp contract float %.sroa.50.0, %1, !spirv.Decorations !898
  br i1 %78, label %3041, label %3037, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3037:                                             ; preds = %3035
  %3038 = add i64 %.in, %506
  %3039 = inttoptr i64 %3038 to float addrspace(4)*
  %3040 = addrspacecast float addrspace(4)* %3039 to float addrspace(1)*
  store float %3036, float addrspace(1)* %3040, align 4
  br label %._crit_edge70.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3041:                                             ; preds = %3035
  %3042 = add i64 %.in3821, %sink_3862
  %3043 = add i64 %3042, %sink_3833
  %3044 = inttoptr i64 %3043 to float addrspace(4)*
  %3045 = addrspacecast float addrspace(4)* %3044 to float addrspace(1)*
  %3046 = load float, float addrspace(1)* %3045, align 4
  %3047 = fmul reassoc nsz arcp contract float %3046, %4, !spirv.Decorations !898
  %3048 = fadd reassoc nsz arcp contract float %3036, %3047, !spirv.Decorations !898
  %3049 = add i64 %.in, %506
  %3050 = inttoptr i64 %3049 to float addrspace(4)*
  %3051 = addrspacecast float addrspace(4)* %3050 to float addrspace(1)*
  store float %3048, float addrspace(1)* %3051, align 4
  br label %._crit_edge70.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.12:                                 ; preds = %.preheader1.11.._crit_edge70.12_crit_edge, %3041, %3037
  br i1 %279, label %3052, label %._crit_edge70.12.._crit_edge70.1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.12.._crit_edge70.1.12_crit_edge:    ; preds = %._crit_edge70.12
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3052:                                             ; preds = %._crit_edge70.12
  %3053 = fmul reassoc nsz arcp contract float %.sroa.114.0, %1, !spirv.Decorations !898
  br i1 %78, label %3058, label %3054, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3054:                                             ; preds = %3052
  %3055 = add i64 %.in, %508
  %3056 = inttoptr i64 %3055 to float addrspace(4)*
  %3057 = addrspacecast float addrspace(4)* %3056 to float addrspace(1)*
  store float %3053, float addrspace(1)* %3057, align 4
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3058:                                             ; preds = %3052
  %3059 = add i64 %.in3821, %sink_3859
  %3060 = add i64 %3059, %sink_3833
  %3061 = inttoptr i64 %3060 to float addrspace(4)*
  %3062 = addrspacecast float addrspace(4)* %3061 to float addrspace(1)*
  %3063 = load float, float addrspace(1)* %3062, align 4
  %3064 = fmul reassoc nsz arcp contract float %3063, %4, !spirv.Decorations !898
  %3065 = fadd reassoc nsz arcp contract float %3053, %3064, !spirv.Decorations !898
  %3066 = add i64 %.in, %508
  %3067 = inttoptr i64 %3066 to float addrspace(4)*
  %3068 = addrspacecast float addrspace(4)* %3067 to float addrspace(1)*
  store float %3065, float addrspace(1)* %3068, align 4
  br label %._crit_edge70.1.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.12:                               ; preds = %._crit_edge70.12.._crit_edge70.1.12_crit_edge, %3058, %3054
  br i1 %282, label %3069, label %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.12.._crit_edge70.2.12_crit_edge:  ; preds = %._crit_edge70.1.12
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3069:                                             ; preds = %._crit_edge70.1.12
  %3070 = fmul reassoc nsz arcp contract float %.sroa.178.0, %1, !spirv.Decorations !898
  br i1 %78, label %3075, label %3071, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3071:                                             ; preds = %3069
  %3072 = add i64 %.in, %510
  %3073 = inttoptr i64 %3072 to float addrspace(4)*
  %3074 = addrspacecast float addrspace(4)* %3073 to float addrspace(1)*
  store float %3070, float addrspace(1)* %3074, align 4
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3075:                                             ; preds = %3069
  %3076 = add i64 %.in3821, %sink_3858
  %3077 = add i64 %3076, %sink_3833
  %3078 = inttoptr i64 %3077 to float addrspace(4)*
  %3079 = addrspacecast float addrspace(4)* %3078 to float addrspace(1)*
  %3080 = load float, float addrspace(1)* %3079, align 4
  %3081 = fmul reassoc nsz arcp contract float %3080, %4, !spirv.Decorations !898
  %3082 = fadd reassoc nsz arcp contract float %3070, %3081, !spirv.Decorations !898
  %3083 = add i64 %.in, %510
  %3084 = inttoptr i64 %3083 to float addrspace(4)*
  %3085 = addrspacecast float addrspace(4)* %3084 to float addrspace(1)*
  store float %3082, float addrspace(1)* %3085, align 4
  br label %._crit_edge70.2.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.12:                               ; preds = %._crit_edge70.1.12.._crit_edge70.2.12_crit_edge, %3075, %3071
  br i1 %285, label %3086, label %._crit_edge70.2.12..preheader1.12_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.12..preheader1.12_crit_edge:      ; preds = %._crit_edge70.2.12
  br label %.preheader1.12, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3086:                                             ; preds = %._crit_edge70.2.12
  %3087 = fmul reassoc nsz arcp contract float %.sroa.242.0, %1, !spirv.Decorations !898
  br i1 %78, label %3092, label %3088, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3088:                                             ; preds = %3086
  %3089 = add i64 %.in, %512
  %3090 = inttoptr i64 %3089 to float addrspace(4)*
  %3091 = addrspacecast float addrspace(4)* %3090 to float addrspace(1)*
  store float %3087, float addrspace(1)* %3091, align 4
  br label %.preheader1.12, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3092:                                             ; preds = %3086
  %3093 = add i64 %.in3821, %sink_3857
  %3094 = add i64 %3093, %sink_3833
  %3095 = inttoptr i64 %3094 to float addrspace(4)*
  %3096 = addrspacecast float addrspace(4)* %3095 to float addrspace(1)*
  %3097 = load float, float addrspace(1)* %3096, align 4
  %3098 = fmul reassoc nsz arcp contract float %3097, %4, !spirv.Decorations !898
  %3099 = fadd reassoc nsz arcp contract float %3087, %3098, !spirv.Decorations !898
  %3100 = add i64 %.in, %512
  %3101 = inttoptr i64 %3100 to float addrspace(4)*
  %3102 = addrspacecast float addrspace(4)* %3101 to float addrspace(1)*
  store float %3099, float addrspace(1)* %3102, align 4
  br label %.preheader1.12, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.12:                                   ; preds = %._crit_edge70.2.12..preheader1.12_crit_edge, %3092, %3088
  %sink_3831 = shl nsw i64 %362, 2
  br i1 %289, label %3103, label %.preheader1.12.._crit_edge70.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.12.._crit_edge70.13_crit_edge:        ; preds = %.preheader1.12
  br label %._crit_edge70.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3103:                                             ; preds = %.preheader1.12
  %3104 = fmul reassoc nsz arcp contract float %.sroa.54.0, %1, !spirv.Decorations !898
  br i1 %78, label %3109, label %3105, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3105:                                             ; preds = %3103
  %3106 = add i64 %.in, %514
  %3107 = inttoptr i64 %3106 to float addrspace(4)*
  %3108 = addrspacecast float addrspace(4)* %3107 to float addrspace(1)*
  store float %3104, float addrspace(1)* %3108, align 4
  br label %._crit_edge70.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3109:                                             ; preds = %3103
  %3110 = add i64 %.in3821, %sink_3862
  %3111 = add i64 %3110, %sink_3831
  %3112 = inttoptr i64 %3111 to float addrspace(4)*
  %3113 = addrspacecast float addrspace(4)* %3112 to float addrspace(1)*
  %3114 = load float, float addrspace(1)* %3113, align 4
  %3115 = fmul reassoc nsz arcp contract float %3114, %4, !spirv.Decorations !898
  %3116 = fadd reassoc nsz arcp contract float %3104, %3115, !spirv.Decorations !898
  %3117 = add i64 %.in, %514
  %3118 = inttoptr i64 %3117 to float addrspace(4)*
  %3119 = addrspacecast float addrspace(4)* %3118 to float addrspace(1)*
  store float %3116, float addrspace(1)* %3119, align 4
  br label %._crit_edge70.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.13:                                 ; preds = %.preheader1.12.._crit_edge70.13_crit_edge, %3109, %3105
  br i1 %292, label %3120, label %._crit_edge70.13.._crit_edge70.1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.13.._crit_edge70.1.13_crit_edge:    ; preds = %._crit_edge70.13
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3120:                                             ; preds = %._crit_edge70.13
  %3121 = fmul reassoc nsz arcp contract float %.sroa.118.0, %1, !spirv.Decorations !898
  br i1 %78, label %3126, label %3122, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3122:                                             ; preds = %3120
  %3123 = add i64 %.in, %516
  %3124 = inttoptr i64 %3123 to float addrspace(4)*
  %3125 = addrspacecast float addrspace(4)* %3124 to float addrspace(1)*
  store float %3121, float addrspace(1)* %3125, align 4
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3126:                                             ; preds = %3120
  %3127 = add i64 %.in3821, %sink_3859
  %3128 = add i64 %3127, %sink_3831
  %3129 = inttoptr i64 %3128 to float addrspace(4)*
  %3130 = addrspacecast float addrspace(4)* %3129 to float addrspace(1)*
  %3131 = load float, float addrspace(1)* %3130, align 4
  %3132 = fmul reassoc nsz arcp contract float %3131, %4, !spirv.Decorations !898
  %3133 = fadd reassoc nsz arcp contract float %3121, %3132, !spirv.Decorations !898
  %3134 = add i64 %.in, %516
  %3135 = inttoptr i64 %3134 to float addrspace(4)*
  %3136 = addrspacecast float addrspace(4)* %3135 to float addrspace(1)*
  store float %3133, float addrspace(1)* %3136, align 4
  br label %._crit_edge70.1.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.13:                               ; preds = %._crit_edge70.13.._crit_edge70.1.13_crit_edge, %3126, %3122
  br i1 %295, label %3137, label %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.13.._crit_edge70.2.13_crit_edge:  ; preds = %._crit_edge70.1.13
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3137:                                             ; preds = %._crit_edge70.1.13
  %3138 = fmul reassoc nsz arcp contract float %.sroa.182.0, %1, !spirv.Decorations !898
  br i1 %78, label %3143, label %3139, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3139:                                             ; preds = %3137
  %3140 = add i64 %.in, %518
  %3141 = inttoptr i64 %3140 to float addrspace(4)*
  %3142 = addrspacecast float addrspace(4)* %3141 to float addrspace(1)*
  store float %3138, float addrspace(1)* %3142, align 4
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3143:                                             ; preds = %3137
  %3144 = add i64 %.in3821, %sink_3858
  %3145 = add i64 %3144, %sink_3831
  %3146 = inttoptr i64 %3145 to float addrspace(4)*
  %3147 = addrspacecast float addrspace(4)* %3146 to float addrspace(1)*
  %3148 = load float, float addrspace(1)* %3147, align 4
  %3149 = fmul reassoc nsz arcp contract float %3148, %4, !spirv.Decorations !898
  %3150 = fadd reassoc nsz arcp contract float %3138, %3149, !spirv.Decorations !898
  %3151 = add i64 %.in, %518
  %3152 = inttoptr i64 %3151 to float addrspace(4)*
  %3153 = addrspacecast float addrspace(4)* %3152 to float addrspace(1)*
  store float %3150, float addrspace(1)* %3153, align 4
  br label %._crit_edge70.2.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.13:                               ; preds = %._crit_edge70.1.13.._crit_edge70.2.13_crit_edge, %3143, %3139
  br i1 %298, label %3154, label %._crit_edge70.2.13..preheader1.13_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.13..preheader1.13_crit_edge:      ; preds = %._crit_edge70.2.13
  br label %.preheader1.13, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3154:                                             ; preds = %._crit_edge70.2.13
  %3155 = fmul reassoc nsz arcp contract float %.sroa.246.0, %1, !spirv.Decorations !898
  br i1 %78, label %3160, label %3156, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3156:                                             ; preds = %3154
  %3157 = add i64 %.in, %520
  %3158 = inttoptr i64 %3157 to float addrspace(4)*
  %3159 = addrspacecast float addrspace(4)* %3158 to float addrspace(1)*
  store float %3155, float addrspace(1)* %3159, align 4
  br label %.preheader1.13, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3160:                                             ; preds = %3154
  %3161 = add i64 %.in3821, %sink_3857
  %3162 = add i64 %3161, %sink_3831
  %3163 = inttoptr i64 %3162 to float addrspace(4)*
  %3164 = addrspacecast float addrspace(4)* %3163 to float addrspace(1)*
  %3165 = load float, float addrspace(1)* %3164, align 4
  %3166 = fmul reassoc nsz arcp contract float %3165, %4, !spirv.Decorations !898
  %3167 = fadd reassoc nsz arcp contract float %3155, %3166, !spirv.Decorations !898
  %3168 = add i64 %.in, %520
  %3169 = inttoptr i64 %3168 to float addrspace(4)*
  %3170 = addrspacecast float addrspace(4)* %3169 to float addrspace(1)*
  store float %3167, float addrspace(1)* %3170, align 4
  br label %.preheader1.13, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.13:                                   ; preds = %._crit_edge70.2.13..preheader1.13_crit_edge, %3160, %3156
  %sink_3829 = shl nsw i64 %363, 2
  br i1 %302, label %3171, label %.preheader1.13.._crit_edge70.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.13.._crit_edge70.14_crit_edge:        ; preds = %.preheader1.13
  br label %._crit_edge70.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3171:                                             ; preds = %.preheader1.13
  %3172 = fmul reassoc nsz arcp contract float %.sroa.58.0, %1, !spirv.Decorations !898
  br i1 %78, label %3177, label %3173, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3173:                                             ; preds = %3171
  %3174 = add i64 %.in, %522
  %3175 = inttoptr i64 %3174 to float addrspace(4)*
  %3176 = addrspacecast float addrspace(4)* %3175 to float addrspace(1)*
  store float %3172, float addrspace(1)* %3176, align 4
  br label %._crit_edge70.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3177:                                             ; preds = %3171
  %3178 = add i64 %.in3821, %sink_3862
  %3179 = add i64 %3178, %sink_3829
  %3180 = inttoptr i64 %3179 to float addrspace(4)*
  %3181 = addrspacecast float addrspace(4)* %3180 to float addrspace(1)*
  %3182 = load float, float addrspace(1)* %3181, align 4
  %3183 = fmul reassoc nsz arcp contract float %3182, %4, !spirv.Decorations !898
  %3184 = fadd reassoc nsz arcp contract float %3172, %3183, !spirv.Decorations !898
  %3185 = add i64 %.in, %522
  %3186 = inttoptr i64 %3185 to float addrspace(4)*
  %3187 = addrspacecast float addrspace(4)* %3186 to float addrspace(1)*
  store float %3184, float addrspace(1)* %3187, align 4
  br label %._crit_edge70.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.14:                                 ; preds = %.preheader1.13.._crit_edge70.14_crit_edge, %3177, %3173
  br i1 %305, label %3188, label %._crit_edge70.14.._crit_edge70.1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.14.._crit_edge70.1.14_crit_edge:    ; preds = %._crit_edge70.14
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3188:                                             ; preds = %._crit_edge70.14
  %3189 = fmul reassoc nsz arcp contract float %.sroa.122.0, %1, !spirv.Decorations !898
  br i1 %78, label %3194, label %3190, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3190:                                             ; preds = %3188
  %3191 = add i64 %.in, %524
  %3192 = inttoptr i64 %3191 to float addrspace(4)*
  %3193 = addrspacecast float addrspace(4)* %3192 to float addrspace(1)*
  store float %3189, float addrspace(1)* %3193, align 4
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3194:                                             ; preds = %3188
  %3195 = add i64 %.in3821, %sink_3859
  %3196 = add i64 %3195, %sink_3829
  %3197 = inttoptr i64 %3196 to float addrspace(4)*
  %3198 = addrspacecast float addrspace(4)* %3197 to float addrspace(1)*
  %3199 = load float, float addrspace(1)* %3198, align 4
  %3200 = fmul reassoc nsz arcp contract float %3199, %4, !spirv.Decorations !898
  %3201 = fadd reassoc nsz arcp contract float %3189, %3200, !spirv.Decorations !898
  %3202 = add i64 %.in, %524
  %3203 = inttoptr i64 %3202 to float addrspace(4)*
  %3204 = addrspacecast float addrspace(4)* %3203 to float addrspace(1)*
  store float %3201, float addrspace(1)* %3204, align 4
  br label %._crit_edge70.1.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.14:                               ; preds = %._crit_edge70.14.._crit_edge70.1.14_crit_edge, %3194, %3190
  br i1 %308, label %3205, label %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.14.._crit_edge70.2.14_crit_edge:  ; preds = %._crit_edge70.1.14
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3205:                                             ; preds = %._crit_edge70.1.14
  %3206 = fmul reassoc nsz arcp contract float %.sroa.186.0, %1, !spirv.Decorations !898
  br i1 %78, label %3211, label %3207, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3207:                                             ; preds = %3205
  %3208 = add i64 %.in, %526
  %3209 = inttoptr i64 %3208 to float addrspace(4)*
  %3210 = addrspacecast float addrspace(4)* %3209 to float addrspace(1)*
  store float %3206, float addrspace(1)* %3210, align 4
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3211:                                             ; preds = %3205
  %3212 = add i64 %.in3821, %sink_3858
  %3213 = add i64 %3212, %sink_3829
  %3214 = inttoptr i64 %3213 to float addrspace(4)*
  %3215 = addrspacecast float addrspace(4)* %3214 to float addrspace(1)*
  %3216 = load float, float addrspace(1)* %3215, align 4
  %3217 = fmul reassoc nsz arcp contract float %3216, %4, !spirv.Decorations !898
  %3218 = fadd reassoc nsz arcp contract float %3206, %3217, !spirv.Decorations !898
  %3219 = add i64 %.in, %526
  %3220 = inttoptr i64 %3219 to float addrspace(4)*
  %3221 = addrspacecast float addrspace(4)* %3220 to float addrspace(1)*
  store float %3218, float addrspace(1)* %3221, align 4
  br label %._crit_edge70.2.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.14:                               ; preds = %._crit_edge70.1.14.._crit_edge70.2.14_crit_edge, %3211, %3207
  br i1 %311, label %3222, label %._crit_edge70.2.14..preheader1.14_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.14..preheader1.14_crit_edge:      ; preds = %._crit_edge70.2.14
  br label %.preheader1.14, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3222:                                             ; preds = %._crit_edge70.2.14
  %3223 = fmul reassoc nsz arcp contract float %.sroa.250.0, %1, !spirv.Decorations !898
  br i1 %78, label %3228, label %3224, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3224:                                             ; preds = %3222
  %3225 = add i64 %.in, %528
  %3226 = inttoptr i64 %3225 to float addrspace(4)*
  %3227 = addrspacecast float addrspace(4)* %3226 to float addrspace(1)*
  store float %3223, float addrspace(1)* %3227, align 4
  br label %.preheader1.14, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3228:                                             ; preds = %3222
  %3229 = add i64 %.in3821, %sink_3857
  %3230 = add i64 %3229, %sink_3829
  %3231 = inttoptr i64 %3230 to float addrspace(4)*
  %3232 = addrspacecast float addrspace(4)* %3231 to float addrspace(1)*
  %3233 = load float, float addrspace(1)* %3232, align 4
  %3234 = fmul reassoc nsz arcp contract float %3233, %4, !spirv.Decorations !898
  %3235 = fadd reassoc nsz arcp contract float %3223, %3234, !spirv.Decorations !898
  %3236 = add i64 %.in, %528
  %3237 = inttoptr i64 %3236 to float addrspace(4)*
  %3238 = addrspacecast float addrspace(4)* %3237 to float addrspace(1)*
  store float %3235, float addrspace(1)* %3238, align 4
  br label %.preheader1.14, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.14:                                   ; preds = %._crit_edge70.2.14..preheader1.14_crit_edge, %3228, %3224
  %sink_3827 = shl nsw i64 %364, 2
  br i1 %315, label %3239, label %.preheader1.14.._crit_edge70.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

.preheader1.14.._crit_edge70.15_crit_edge:        ; preds = %.preheader1.14
  br label %._crit_edge70.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3239:                                             ; preds = %.preheader1.14
  %3240 = fmul reassoc nsz arcp contract float %.sroa.62.0, %1, !spirv.Decorations !898
  br i1 %78, label %3245, label %3241, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3241:                                             ; preds = %3239
  %3242 = add i64 %.in, %530
  %3243 = inttoptr i64 %3242 to float addrspace(4)*
  %3244 = addrspacecast float addrspace(4)* %3243 to float addrspace(1)*
  store float %3240, float addrspace(1)* %3244, align 4
  br label %._crit_edge70.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3245:                                             ; preds = %3239
  %3246 = add i64 %.in3821, %sink_3862
  %3247 = add i64 %3246, %sink_3827
  %3248 = inttoptr i64 %3247 to float addrspace(4)*
  %3249 = addrspacecast float addrspace(4)* %3248 to float addrspace(1)*
  %3250 = load float, float addrspace(1)* %3249, align 4
  %3251 = fmul reassoc nsz arcp contract float %3250, %4, !spirv.Decorations !898
  %3252 = fadd reassoc nsz arcp contract float %3240, %3251, !spirv.Decorations !898
  %3253 = add i64 %.in, %530
  %3254 = inttoptr i64 %3253 to float addrspace(4)*
  %3255 = addrspacecast float addrspace(4)* %3254 to float addrspace(1)*
  store float %3252, float addrspace(1)* %3255, align 4
  br label %._crit_edge70.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.15:                                 ; preds = %.preheader1.14.._crit_edge70.15_crit_edge, %3245, %3241
  br i1 %318, label %3256, label %._crit_edge70.15.._crit_edge70.1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.15.._crit_edge70.1.15_crit_edge:    ; preds = %._crit_edge70.15
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3256:                                             ; preds = %._crit_edge70.15
  %3257 = fmul reassoc nsz arcp contract float %.sroa.126.0, %1, !spirv.Decorations !898
  br i1 %78, label %3262, label %3258, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3258:                                             ; preds = %3256
  %3259 = add i64 %.in, %532
  %3260 = inttoptr i64 %3259 to float addrspace(4)*
  %3261 = addrspacecast float addrspace(4)* %3260 to float addrspace(1)*
  store float %3257, float addrspace(1)* %3261, align 4
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3262:                                             ; preds = %3256
  %3263 = add i64 %.in3821, %sink_3859
  %3264 = add i64 %3263, %sink_3827
  %3265 = inttoptr i64 %3264 to float addrspace(4)*
  %3266 = addrspacecast float addrspace(4)* %3265 to float addrspace(1)*
  %3267 = load float, float addrspace(1)* %3266, align 4
  %3268 = fmul reassoc nsz arcp contract float %3267, %4, !spirv.Decorations !898
  %3269 = fadd reassoc nsz arcp contract float %3257, %3268, !spirv.Decorations !898
  %3270 = add i64 %.in, %532
  %3271 = inttoptr i64 %3270 to float addrspace(4)*
  %3272 = addrspacecast float addrspace(4)* %3271 to float addrspace(1)*
  store float %3269, float addrspace(1)* %3272, align 4
  br label %._crit_edge70.1.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.1.15:                               ; preds = %._crit_edge70.15.._crit_edge70.1.15_crit_edge, %3262, %3258
  br i1 %321, label %3273, label %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.1.15.._crit_edge70.2.15_crit_edge:  ; preds = %._crit_edge70.1.15
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3273:                                             ; preds = %._crit_edge70.1.15
  %3274 = fmul reassoc nsz arcp contract float %.sroa.190.0, %1, !spirv.Decorations !898
  br i1 %78, label %3279, label %3275, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3275:                                             ; preds = %3273
  %3276 = add i64 %.in, %534
  %3277 = inttoptr i64 %3276 to float addrspace(4)*
  %3278 = addrspacecast float addrspace(4)* %3277 to float addrspace(1)*
  store float %3274, float addrspace(1)* %3278, align 4
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3279:                                             ; preds = %3273
  %3280 = add i64 %.in3821, %sink_3858
  %3281 = add i64 %3280, %sink_3827
  %3282 = inttoptr i64 %3281 to float addrspace(4)*
  %3283 = addrspacecast float addrspace(4)* %3282 to float addrspace(1)*
  %3284 = load float, float addrspace(1)* %3283, align 4
  %3285 = fmul reassoc nsz arcp contract float %3284, %4, !spirv.Decorations !898
  %3286 = fadd reassoc nsz arcp contract float %3274, %3285, !spirv.Decorations !898
  %3287 = add i64 %.in, %534
  %3288 = inttoptr i64 %3287 to float addrspace(4)*
  %3289 = addrspacecast float addrspace(4)* %3288 to float addrspace(1)*
  store float %3286, float addrspace(1)* %3289, align 4
  br label %._crit_edge70.2.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

._crit_edge70.2.15:                               ; preds = %._crit_edge70.1.15.._crit_edge70.2.15_crit_edge, %3279, %3275
  br i1 %324, label %3290, label %._crit_edge70.2.15..preheader1.15_crit_edge, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge70.2.15..preheader1.15_crit_edge:      ; preds = %._crit_edge70.2.15
  br label %.preheader1.15, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3290:                                             ; preds = %._crit_edge70.2.15
  %3291 = fmul reassoc nsz arcp contract float %.sroa.254.0, %1, !spirv.Decorations !898
  br i1 %78, label %3296, label %3292, !stats.blockFrequency.digits !925, !stats.blockFrequency.scale !879

3292:                                             ; preds = %3290
  %3293 = add i64 %.in, %536
  %3294 = inttoptr i64 %3293 to float addrspace(4)*
  %3295 = addrspacecast float addrspace(4)* %3294 to float addrspace(1)*
  store float %3291, float addrspace(1)* %3295, align 4
  br label %.preheader1.15, !stats.blockFrequency.digits !926, !stats.blockFrequency.scale !879

3296:                                             ; preds = %3290
  %3297 = add i64 %.in3821, %sink_3857
  %3298 = add i64 %3297, %sink_3827
  %3299 = inttoptr i64 %3298 to float addrspace(4)*
  %3300 = addrspacecast float addrspace(4)* %3299 to float addrspace(1)*
  %3301 = load float, float addrspace(1)* %3300, align 4
  %3302 = fmul reassoc nsz arcp contract float %3301, %4, !spirv.Decorations !898
  %3303 = fadd reassoc nsz arcp contract float %3291, %3302, !spirv.Decorations !898
  %3304 = add i64 %.in, %536
  %3305 = inttoptr i64 %3304 to float addrspace(4)*
  %3306 = addrspacecast float addrspace(4)* %3305 to float addrspace(1)*
  store float %3303, float addrspace(1)* %3306, align 4
  br label %.preheader1.15, !stats.blockFrequency.digits !927, !stats.blockFrequency.scale !879

.preheader1.15:                                   ; preds = %._crit_edge70.2.15..preheader1.15_crit_edge, %3296, %3292
  %3307 = add i32 %547, %49
  %3308 = icmp slt i32 %3307, %8
  br i1 %3308, label %.preheader1.15..preheader2.preheader_crit_edge, label %._crit_edge72.loopexit, !stats.blockFrequency.digits !883, !stats.blockFrequency.scale !879

._crit_edge72.loopexit:                           ; preds = %.preheader1.15
  br label %._crit_edge72, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !879

.preheader1.15..preheader2.preheader_crit_edge:   ; preds = %.preheader1.15
  %3309 = add i64 %.in3823, %537
  %3310 = add i64 %.in3822, %538
  %sink_ = bitcast <2 x i32> %545 to i64
  %3311 = add i64 %.in3821, %sink_
  %3312 = add i64 %.in, %546
  br label %.preheader2.preheader, !stats.blockFrequency.digits !885, !stats.blockFrequency.scale !879

._crit_edge72:                                    ; preds = %.._crit_edge72_crit_edge, %._crit_edge72.loopexit
  ret void, !stats.blockFrequency.digits !878, !stats.blockFrequency.scale !879
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func float @__builtin_IB_frnd_zi(float noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func signext i16 @__builtin_IB_ftobf_1(float noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_global_offset(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_num_groups(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_global_size(i32 noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #3

declare i32 @printf(i8 addrspace(2)*, ...)

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind willreturn memory(none)
declare i16 @llvm.genx.GenISA.ftobf.i16.f32(float, i32) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.trunc.f32(float) #6

define spir_kernel void @Intel_Symbol_Table_Void_Program() {
entry:
  ret void, !stats.blockFrequency.digits !880, !stats.blockFrequency.scale !887
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.fshl.i64(i64, i64, i64) #6

; Function Desc: 
; Output: 
; Function Attrs: nounwind willreturn memory(none)
declare i16 @llvm.genx.GenISA.simdLaneId() #5

; Function Desc: 
; Output: 
; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.genx.GenISA.simdSize() #5

; Function Desc: 
; Output: 
; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.genx.GenISA.hw.thread.id.alloca.i32() #5

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Function Attrs: nounwind willreturn memory(none)
declare { i32, i32 } @llvm.genx.GenISA.mul.pair(i32, i32, i32, i32) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind willreturn memory(none)
declare <4 x i32> @llvm.genx.GenISA.bitcastfromstruct.v4i32.__StructSOALayout_(%__StructSOALayout_) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind willreturn memory(none)
declare <3 x i32> @llvm.genx.GenISA.bitcastfromstruct.v3i32.__StructSOALayout_.7(%__StructSOALayout_.7) #5

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!spirv.MemoryModel = !{!2, !2}
!spirv.Source = !{!3, !3}
!spirv.Generator = !{!4, !4}
!igc.functions = !{!5, !46, !65, !82, !88, !92, !93, !99, !100, !122, !137, !140, !143, !146}
!IGCMetadata = !{!149}
!opencl.ocl.version = !{!875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875}
!opencl.spir.version = !{!875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875, !875}
!llvm.ident = !{!876, !876, !876, !876, !876, !876, !876, !876, !876, !876, !876, !876, !876}
!llvm.module.flags = !{!877}

!0 = !{!1}
!1 = !{i32 44, i32 8}
!2 = !{i32 2, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{i16 6, i16 14}
!5 = !{void (%"class.sycl::_V1::range"*, %class.__generated_*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE, !6}
!6 = !{!7, !8, !45}
!7 = !{!"function_type", i32 0}
!8 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !21, !23, !25, !27, !28, !29, !31, !33, !35, !37, !39, !41, !43}
!9 = !{i32 0}
!10 = !{i32 2}
!11 = !{i32 5}
!12 = !{i32 7}
!13 = !{i32 8}
!14 = !{i32 9}
!15 = !{i32 10}
!16 = !{i32 11}
!17 = !{i32 13}
!18 = !{i32 17, !19, !20}
!19 = !{!"explicit_arg_num", i32 0}
!20 = !{!"struct_arg_offset", i32 0}
!21 = !{i32 17, !19, !22}
!22 = !{!"struct_arg_offset", i32 8}
!23 = !{i32 17, !19, !24}
!24 = !{!"struct_arg_offset", i32 16}
!25 = !{i32 17, !26, !20}
!26 = !{!"explicit_arg_num", i32 1}
!27 = !{i32 17, !26, !22}
!28 = !{i32 17, !26, !24}
!29 = !{i32 17, !26, !30}
!30 = !{!"struct_arg_offset", i32 24}
!31 = !{i32 17, !26, !32}
!32 = !{!"struct_arg_offset", i32 32}
!33 = !{i32 17, !26, !34}
!34 = !{!"struct_arg_offset", i32 40}
!35 = !{i32 17, !26, !36}
!36 = !{!"struct_arg_offset", i32 48}
!37 = !{i32 17, !26, !38}
!38 = !{!"struct_arg_offset", i32 56}
!39 = !{i32 17, !26, !40}
!40 = !{!"struct_arg_offset", i32 64}
!41 = !{i32 17, !26, !42}
!42 = !{!"struct_arg_offset", i32 72}
!43 = !{i32 59, !44}
!44 = !{!"explicit_arg_num", i32 9}
!45 = !{!"max_reg_pressure", i32 23}
!46 = !{void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail, !47}
!47 = !{!7, !48, !64}
!48 = !{!"implicit_arg_desc", !9, !10, !12, !13, !14, !15, !16, !17, !49, !51, !52, !53, !55, !56, !57, !58, !60, !61, !62}
!49 = !{i32 17, !50, !20}
!50 = !{!"explicit_arg_num", i32 2}
!51 = !{i32 17, !50, !22}
!52 = !{i32 17, !50, !24}
!53 = !{i32 17, !54, !20}
!54 = !{!"explicit_arg_num", i32 5}
!55 = !{i32 17, !54, !22}
!56 = !{i32 17, !54, !24}
!57 = !{i32 15, !19}
!58 = !{i32 15, !59}
!59 = !{!"explicit_arg_num", i32 3}
!60 = !{i32 59, !19}
!61 = !{i32 59, !59}
!62 = !{i32 59, !63}
!63 = !{!"explicit_arg_num", i32 12}
!64 = !{!"max_reg_pressure", i32 9}
!65 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !66}
!66 = !{!7, !67, !81}
!67 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !68, !69, !71, !73, !75, !77, !79, !43}
!68 = !{i32 19, !26, !22}
!69 = !{i32 20, !26, !70}
!70 = !{!"struct_arg_offset", i32 10}
!71 = !{i32 20, !26, !72}
!72 = !{!"struct_arg_offset", i32 11}
!73 = !{i32 20, !26, !74}
!74 = !{!"struct_arg_offset", i32 12}
!75 = !{i32 20, !26, !76}
!76 = !{!"struct_arg_offset", i32 13}
!77 = !{i32 20, !26, !78}
!78 = !{!"struct_arg_offset", i32 14}
!79 = !{i32 20, !26, !80}
!80 = !{!"struct_arg_offset", i32 15}
!81 = !{!"max_reg_pressure", i32 7}
!82 = !{void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_, !83}
!83 = !{!7, !84, !87}
!84 = !{!"implicit_arg_desc", !9, !10, !12, !13, !14, !15, !16, !57, !60, !85}
!85 = !{i32 59, !86}
!86 = !{!"explicit_arg_num", i32 8}
!87 = !{!"max_reg_pressure", i32 3}
!88 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !89}
!89 = !{!7, !90, !81}
!90 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !91, !73, !75, !77, !79, !43}
!91 = !{i32 18, !26, !22}
!92 = !{void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_, !83}
!93 = !{void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE, !94}
!94 = !{!7, !95, !81}
!95 = !{!"implicit_arg_desc", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !25, !96, !97, !69, !71, !73, !75, !77, !79, !43}
!96 = !{i32 20, !26, !22}
!97 = !{i32 20, !26, !98}
!98 = !{!"struct_arg_offset", i32 9}
!99 = !{void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_, !83}
!100 = !{void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE, !101}
!101 = !{!7, !102, !121}
!102 = !{!"implicit_arg_desc", !9, !10, !103, !104, !13, !14, !15, !16, !17, !49, !105, !106, !107, !108, !110, !111, !113, !115, !117, !57, !60, !119}
!103 = !{i32 4}
!104 = !{i32 6}
!105 = !{i32 16, !50, !22}
!106 = !{i32 16, !50, !74}
!107 = !{i32 18, !50, !24}
!108 = !{i32 16, !50, !109}
!109 = !{!"struct_arg_offset", i32 20}
!110 = !{i32 16, !50, !30}
!111 = !{i32 20, !50, !112}
!112 = !{!"struct_arg_offset", i32 28}
!113 = !{i32 20, !50, !114}
!114 = !{!"struct_arg_offset", i32 29}
!115 = !{i32 20, !50, !116}
!116 = !{!"struct_arg_offset", i32 30}
!117 = !{i32 20, !50, !118}
!118 = !{!"struct_arg_offset", i32 31}
!119 = !{i32 59, !120}
!120 = !{!"explicit_arg_num", i32 10}
!121 = !{!"max_reg_pressure", i32 61}
!122 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE, !123}
!123 = !{!7, !124, !136}
!124 = !{!"implicit_arg_desc", !9, !10, !103, !104, !13, !14, !15, !16, !17, !125, !126, !128, !49, !51, !129, !130, !53, !55, !131, !133, !134}
!125 = !{i32 18, !19, !20}
!126 = !{i32 18, !19, !127}
!127 = !{!"struct_arg_offset", i32 4}
!128 = !{i32 18, !19, !22}
!129 = !{i32 17, !59, !20}
!130 = !{i32 17, !59, !22}
!131 = !{i32 17, !132, !20}
!132 = !{!"explicit_arg_num", i32 6}
!133 = !{i32 17, !132, !22}
!134 = !{i32 59, !135}
!135 = !{!"explicit_arg_num", i32 20}
!136 = !{!"max_reg_pressure", i32 101}
!137 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !138}
!138 = !{!7, !124, !139}
!139 = !{!"max_reg_pressure", i32 288}
!140 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE, !141}
!141 = !{!7, !124, !142}
!142 = !{!"max_reg_pressure", i32 93}
!143 = !{void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE, !144}
!144 = !{!7, !124, !145}
!145 = !{!"max_reg_pressure", i32 257}
!146 = !{void ()* @Intel_Symbol_Table_Void_Program, !147}
!147 = !{!7, !148}
!148 = !{!"max_reg_pressure", i32 0}
!149 = !{!"ModuleMD", !150, !151, !289, !606, !637, !659, !660, !664, !667, !668, !669, !708, !733, !747, !748, !749, !766, !767, !768, !769, !773, !774, !782, !783, !784, !785, !786, !787, !788, !789, !790, !791, !792, !797, !799, !803, !804, !805, !806, !807, !808, !809, !810, !811, !812, !813, !814, !815, !816, !817, !818, !819, !820, !821, !822, !369, !823, !852, !853, !855, !857, !860, !861, !862, !864, !865, !866, !871, !872, !873, !874}
!150 = !{!"isPrecise", i1 false}
!151 = !{!"compOpt", !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !249, !250, !251, !252, !253, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287, !288}
!152 = !{!"DenormsAreZero", i1 false}
!153 = !{!"BFTFDenormsAreZero", i1 false}
!154 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!155 = !{!"OptDisable", i1 false}
!156 = !{!"MadEnable", i1 true}
!157 = !{!"NoSignedZeros", i1 false}
!158 = !{!"NoNaNs", i1 false}
!159 = !{!"FloatDenormMode16", !"FLOAT_DENORM_RETAIN"}
!160 = !{!"FloatDenormMode32", !"FLOAT_DENORM_RETAIN"}
!161 = !{!"FloatDenormMode64", !"FLOAT_DENORM_RETAIN"}
!162 = !{!"FloatDenormModeBFTF", !"FLOAT_DENORM_RETAIN"}
!163 = !{!"FloatRoundingMode", i32 0}
!164 = !{!"FloatCvtIntRoundingMode", i32 3}
!165 = !{!"LoadCacheDefault", i32 4}
!166 = !{!"StoreCacheDefault", i32 2}
!167 = !{!"VISAPreSchedRPThreshold", i32 0}
!168 = !{!"VISAPreSchedCtrl", i32 0}
!169 = !{!"SetLoopUnrollThreshold", i32 0}
!170 = !{!"UnsafeMathOptimizations", i1 false}
!171 = !{!"disableCustomUnsafeOpts", i1 false}
!172 = !{!"disableReducePow", i1 false}
!173 = !{!"disableSqrtOpt", i1 false}
!174 = !{!"FiniteMathOnly", i1 false}
!175 = !{!"FastRelaxedMath", i1 false}
!176 = !{!"DashGSpecified", i1 false}
!177 = !{!"FastCompilation", i1 false}
!178 = !{!"UseScratchSpacePrivateMemory", i1 false}
!179 = !{!"RelaxedBuiltins", i1 false}
!180 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!181 = !{!"GreaterThan2GBBufferRequired", i1 true}
!182 = !{!"GreaterThan4GBBufferRequired", i1 true}
!183 = !{!"DisableA64WA", i1 false}
!184 = !{!"ForceEnableA64WA", i1 false}
!185 = !{!"PushConstantsEnable", i1 true}
!186 = !{!"HasPositivePointerOffset", i1 false}
!187 = !{!"HasBufferOffsetArg", i1 true}
!188 = !{!"BufferOffsetArgOptional", i1 true}
!189 = !{!"replaceGlobalOffsetsByZero", i1 false}
!190 = !{!"forcePixelShaderSIMDMode", i32 0}
!191 = !{!"forceTotalGRFNum", i32 0}
!192 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!193 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!194 = !{!"UniformWGS", i1 false}
!195 = !{!"disableVertexComponentPacking", i1 false}
!196 = !{!"disablePartialVertexComponentPacking", i1 false}
!197 = !{!"PreferBindlessImages", i1 true}
!198 = !{!"UseBindlessMode", i1 true}
!199 = !{!"UseLegacyBindlessMode", i1 false}
!200 = !{!"disableMathRefactoring", i1 false}
!201 = !{!"atomicBranch", i1 false}
!202 = !{!"spillCompression", i1 false}
!203 = !{!"AllowLICM", i1 true}
!204 = !{!"DisableEarlyOut", i1 false}
!205 = !{!"ForceInt32DivRemEmu", i1 false}
!206 = !{!"ForceInt32DivRemEmuSP", i1 false}
!207 = !{!"DisableIntDivRemIncrementReduction", i1 false}
!208 = !{!"WaveIntrinsicUsed", i1 false}
!209 = !{!"DisableMultiPolyPS", i1 false}
!210 = !{!"NeedTexture3DLODWA", i1 false}
!211 = !{!"UseLivePrologueKernelForRaytracingDispatch", i1 false}
!212 = !{!"DisableFastestSingleCSSIMD", i1 false}
!213 = !{!"DisableFastestLinearScan", i1 false}
!214 = !{!"UseStatelessforPrivateMemory", i1 false}
!215 = !{!"EnableTakeGlobalAddress", i1 false}
!216 = !{!"IsLibraryCompilation", i1 false}
!217 = !{!"LibraryCompileSIMDSize", i32 0}
!218 = !{!"FastVISACompile", i1 false}
!219 = !{!"MatchSinCosPi", i1 false}
!220 = !{!"ExcludeIRFromZEBinary", i1 false}
!221 = !{!"EmitZeBinVISASections", i1 false}
!222 = !{!"FP64GenEmulationEnabled", i1 false}
!223 = !{!"FP64GenConvEmulationEnabled", i1 false}
!224 = !{!"allowDisableRematforCS", i1 false}
!225 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!226 = !{!"DisableCPSOmaskWA", i1 false}
!227 = !{!"DisableFastestGopt", i1 false}
!228 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!229 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!230 = !{!"DisableConstantCoalescing", i1 false}
!231 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!232 = !{!"WaEnableALTModeVisaWA", i1 false}
!233 = !{!"EnableLdStCombineforLoad", i1 false}
!234 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!235 = !{!"WaEnableAtomicWaveFusion", i1 false}
!236 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!237 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!238 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!239 = !{!"WaEnableAtomicWaveFusionPartial", i1 false}
!240 = !{!"WaEnableAtomicWaveFusionMoreDimensions", i1 false}
!241 = !{!"WaEnableAtomicWaveFusionLoop", i1 false}
!242 = !{!"WaEnableAtomicWaveFusionReturnValuePolicy", i32 0}
!243 = !{!"ForceCBThroughSampler3D", i1 false}
!244 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!245 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!246 = !{!"WaTypedAtomicBinToRawAtomicBin", i1 false}
!247 = !{!"WaRawAtomicBinToTypedAtomicBin", i1 false}
!248 = !{!"WaSampleLoadToTypedRead", i1 false}
!249 = !{!"EnableTypedBufferStoreToUntypedStore", i1 false}
!250 = !{!"WaZeroSLMBeforeUse", i1 false}
!251 = !{!"EnableEmitMoreMoviCases", i1 false}
!252 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!253 = !{!"EnableFastSampleD", i1 false}
!254 = !{!"ForceUniformBuffer", i1 false}
!255 = !{!"ForceUniformSurfaceSampler", i1 false}
!256 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!257 = !{!"NewSpillCostFunction", i1 false}
!258 = !{!"EnableVRT", i1 false}
!259 = !{!"ForceLargeGRFNum4RQ", i1 false}
!260 = !{!"Enable2xGRFRetry", i1 false}
!261 = !{!"Detect2xGRFCandidate", i1 false}
!262 = !{!"EnableURBWritesMerging", i1 true}
!263 = !{!"ForceCacheLineAlignedURBWriteMerging", i1 false}
!264 = !{!"DisableURBLayoutAlignmentToCacheLine", i1 false}
!265 = !{!"DisableEUFusion", i1 false}
!266 = !{!"DisableFDivToFMulInvOpt", i1 false}
!267 = !{!"initializePhiSampleSourceWA", i1 false}
!268 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!269 = !{!"DisableLoosenSimd32Occu", i1 false}
!270 = !{!"FastestS1Options", i32 0}
!271 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!272 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!273 = !{!"LscSamplerRouting", i32 0}
!274 = !{!"UseBarrierControlFlowOptimization", i1 false}
!275 = !{!"EnableDynamicRQManagement", i1 false}
!276 = !{!"WaDisablePayloadCoalescing", i1 false}
!277 = !{!"Quad8InputThreshold", i32 0}
!278 = !{!"UseResourceLoopUnrollNested", i1 false}
!279 = !{!"DisableLoopUnroll", i1 false}
!280 = !{!"ForcePushConstantMode", i32 0}
!281 = !{!"UseInstructionHoistingOptimization", i1 false}
!282 = !{!"DisableResourceLoopDestLifeTimeStart", i1 false}
!283 = !{!"ForceVRTGRFCeiling", i32 0}
!284 = !{!"DisableSamplerBackingByLSC", i32 0}
!285 = !{!"UseLinearScanRA", i1 false}
!286 = !{!"DisableConvertingAtomicIAddToIncDec", i1 false}
!287 = !{!"EnableInlinedCrossThreadData", i1 false}
!288 = !{!"ZeroInitRegistersBeforeExecution", i1 false}
!289 = !{!"FuncMD", !290, !291, !406, !407, !444, !445, !456, !457, !467, !468, !475, !476, !483, !484, !491, !492, !497, !498, !508, !509, !587, !588, !589, !590, !591, !592, !593, !594}
!290 = !{!"FuncMDMap[0]", void (%"class.sycl::_V1::range"*, %class.__generated_*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE}
!291 = !{!"FuncMDValue[0]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !327, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !384, !387, !390, !393, !396, !399, !402}
!292 = !{!"localOffsets"}
!293 = !{!"workGroupWalkOrder", !294, !295, !296}
!294 = !{!"dim0", i32 0}
!295 = !{!"dim1", i32 1}
!296 = !{!"dim2", i32 2}
!297 = !{!"funcArgs"}
!298 = !{!"functionType", !"KernelFunction"}
!299 = !{!"inlineDynConstants"}
!300 = !{!"inlineDynRootConstant"}
!301 = !{!"inlineDynConstantDescTable"}
!302 = !{!"m_pInterestingConstants"}
!303 = !{!"rtInfo", !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314, !315, !316, !317, !318, !319, !320, !322, !323, !324, !325, !326}
!304 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!305 = !{!"isContinuation", i1 false}
!306 = !{!"isMonolithic", i1 false}
!307 = !{!"hasTraceRayPayload", i1 false}
!308 = !{!"hasHitAttributes", i1 false}
!309 = !{!"hasCallableData", i1 false}
!310 = !{!"ShaderStackSize", i32 0}
!311 = !{!"ShaderHash", i64 0}
!312 = !{!"ShaderName", !""}
!313 = !{!"ParentName", !""}
!314 = !{!"SlotNum", i1* null}
!315 = !{!"NOSSize", i32 0}
!316 = !{!"globalRootSignatureSize", i32 0}
!317 = !{!"Entries"}
!318 = !{!"SpillUnions"}
!319 = !{!"CustomHitAttrSizeInBytes", i32 0}
!320 = !{!"Types", !321}
!321 = !{!"FullFrameTys"}
!322 = !{!"Aliases"}
!323 = !{!"numSyncRTStacks", i32 0}
!324 = !{!"NumCoherenceHintBits", i32 0}
!325 = !{!"useSyncHWStack", i1 false}
!326 = !{!"OriginatingShaderName", !""}
!327 = !{!"resAllocMD", !328, !329, !330, !331, !360}
!328 = !{!"uavsNumType", i32 0}
!329 = !{!"srvsNumType", i32 0}
!330 = !{!"samplersNumType", i32 0}
!331 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359}
!332 = !{!"argAllocMDListVec[0]", !333, !334, !335}
!333 = !{!"type", i32 0}
!334 = !{!"extensionType", i32 -1}
!335 = !{!"indexType", i32 -1}
!336 = !{!"argAllocMDListVec[1]", !333, !334, !335}
!337 = !{!"argAllocMDListVec[2]", !333, !334, !335}
!338 = !{!"argAllocMDListVec[3]", !333, !334, !335}
!339 = !{!"argAllocMDListVec[4]", !333, !334, !335}
!340 = !{!"argAllocMDListVec[5]", !333, !334, !335}
!341 = !{!"argAllocMDListVec[6]", !333, !334, !335}
!342 = !{!"argAllocMDListVec[7]", !333, !334, !335}
!343 = !{!"argAllocMDListVec[8]", !333, !334, !335}
!344 = !{!"argAllocMDListVec[9]", !333, !334, !335}
!345 = !{!"argAllocMDListVec[10]", !333, !334, !335}
!346 = !{!"argAllocMDListVec[11]", !333, !334, !335}
!347 = !{!"argAllocMDListVec[12]", !333, !334, !335}
!348 = !{!"argAllocMDListVec[13]", !333, !334, !335}
!349 = !{!"argAllocMDListVec[14]", !333, !334, !335}
!350 = !{!"argAllocMDListVec[15]", !333, !334, !335}
!351 = !{!"argAllocMDListVec[16]", !333, !334, !335}
!352 = !{!"argAllocMDListVec[17]", !333, !334, !335}
!353 = !{!"argAllocMDListVec[18]", !333, !334, !335}
!354 = !{!"argAllocMDListVec[19]", !333, !334, !335}
!355 = !{!"argAllocMDListVec[20]", !333, !334, !335}
!356 = !{!"argAllocMDListVec[21]", !333, !334, !335}
!357 = !{!"argAllocMDListVec[22]", !333, !334, !335}
!358 = !{!"argAllocMDListVec[23]", !333, !334, !335}
!359 = !{!"argAllocMDListVec[24]", !333, !334, !335}
!360 = !{!"inlineSamplersMD"}
!361 = !{!"maxByteOffsets"}
!362 = !{!"IsInitializer", i1 false}
!363 = !{!"IsFinalizer", i1 false}
!364 = !{!"CompiledSubGroupsNumber", i32 0}
!365 = !{!"hasInlineVmeSamplers", i1 false}
!366 = !{!"localSize", i32 0}
!367 = !{!"localIDPresent", i1 false}
!368 = !{!"groupIDPresent", i1 false}
!369 = !{!"privateMemoryPerWI", i32 0}
!370 = !{!"prevFPOffset", i32 0}
!371 = !{!"globalIDPresent", i1 false}
!372 = !{!"hasSyncRTCalls", i1 false}
!373 = !{!"hasPrintfCalls", i1 false}
!374 = !{!"requireAssertBuffer", i1 false}
!375 = !{!"requireSyncBuffer", i1 false}
!376 = !{!"hasIndirectCalls", i1 false}
!377 = !{!"hasNonKernelArgLoad", i1 false}
!378 = !{!"hasNonKernelArgStore", i1 false}
!379 = !{!"hasNonKernelArgAtomic", i1 false}
!380 = !{!"UserAnnotations"}
!381 = !{!"m_OpenCLArgAddressSpaces", !382, !383}
!382 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 0}
!383 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!384 = !{!"m_OpenCLArgAccessQualifiers", !385, !386}
!385 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!386 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!387 = !{!"m_OpenCLArgTypes", !388, !389}
!388 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::range"}
!389 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_"}
!390 = !{!"m_OpenCLArgBaseTypes", !391, !392}
!391 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::range"}
!392 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_"}
!393 = !{!"m_OpenCLArgTypeQualifiers", !394, !395}
!394 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!395 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!396 = !{!"m_OpenCLArgNames", !397, !398}
!397 = !{!"m_OpenCLArgNamesVec[0]", !""}
!398 = !{!"m_OpenCLArgNamesVec[1]", !""}
!399 = !{!"m_OpenCLArgScalarAsPointers", !400, !401}
!400 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 14}
!401 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 19}
!402 = !{!"m_OptsToDisablePerFunc", !403, !404, !405}
!403 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!404 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!405 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!406 = !{!"FuncMDMap[1]", void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail}
!407 = !{!"FuncMDValue[1]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !327, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !408, !414, !419, !426, !433, !438, !443, !402}
!408 = !{!"m_OpenCLArgAddressSpaces", !409, !383, !410, !411, !412, !413}
!409 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!410 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!411 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!412 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 0}
!413 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!414 = !{!"m_OpenCLArgAccessQualifiers", !385, !386, !415, !416, !417, !418}
!415 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!416 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!417 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!418 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!419 = !{!"m_OpenCLArgTypes", !420, !421, !422, !423, !424, !425}
!420 = !{!"m_OpenCLArgTypesVec[0]", !"char*"}
!421 = !{!"m_OpenCLArgTypesVec[1]", !"long"}
!422 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!423 = !{!"m_OpenCLArgTypesVec[3]", !"char*"}
!424 = !{!"m_OpenCLArgTypesVec[4]", !"long"}
!425 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!426 = !{!"m_OpenCLArgBaseTypes", !427, !428, !429, !430, !431, !432}
!427 = !{!"m_OpenCLArgBaseTypesVec[0]", !"char*"}
!428 = !{!"m_OpenCLArgBaseTypesVec[1]", !"long"}
!429 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!430 = !{!"m_OpenCLArgBaseTypesVec[3]", !"char*"}
!431 = !{!"m_OpenCLArgBaseTypesVec[4]", !"long"}
!432 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!433 = !{!"m_OpenCLArgTypeQualifiers", !394, !395, !434, !435, !436, !437}
!434 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!435 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!436 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!437 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!438 = !{!"m_OpenCLArgNames", !397, !398, !439, !440, !441, !442}
!439 = !{!"m_OpenCLArgNamesVec[2]", !""}
!440 = !{!"m_OpenCLArgNamesVec[3]", !""}
!441 = !{!"m_OpenCLArgNamesVec[4]", !""}
!442 = !{!"m_OpenCLArgNamesVec[5]", !""}
!443 = !{!"m_OpenCLArgScalarAsPointers"}
!444 = !{!"FuncMDMap[2]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!445 = !{!"FuncMDValue[2]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !446, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !384, !448, !451, !393, !396, !454, !402}
!446 = !{!"resAllocMD", !328, !329, !330, !447, !360}
!447 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355}
!448 = !{!"m_OpenCLArgTypes", !449, !450}
!449 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::range.0"}
!450 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.2"}
!451 = !{!"m_OpenCLArgBaseTypes", !452, !453}
!452 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::range.0"}
!453 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.2"}
!454 = !{!"m_OpenCLArgScalarAsPointers", !455}
!455 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 12}
!456 = !{!"FuncMDMap[3]", void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!457 = !{!"FuncMDValue[3]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !458, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !460, !384, !461, !464, !393, !396, !443, !402}
!458 = !{!"resAllocMD", !328, !329, !330, !459, !360}
!459 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346}
!460 = !{!"m_OpenCLArgAddressSpaces", !409, !383}
!461 = !{!"m_OpenCLArgTypes", !462, !463}
!462 = !{!"m_OpenCLArgTypesVec[0]", !"short*"}
!463 = !{!"m_OpenCLArgTypesVec[1]", !"ushort"}
!464 = !{!"m_OpenCLArgBaseTypes", !465, !466}
!465 = !{!"m_OpenCLArgBaseTypesVec[0]", !"short*"}
!466 = !{!"m_OpenCLArgBaseTypesVec[1]", !"ushort"}
!467 = !{!"FuncMDMap[4]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!468 = !{!"FuncMDValue[4]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !469, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !384, !471, !473, !393, !396, !454, !402}
!469 = !{!"resAllocMD", !328, !329, !330, !470, !360}
!470 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353}
!471 = !{!"m_OpenCLArgTypes", !449, !472}
!472 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.9"}
!473 = !{!"m_OpenCLArgBaseTypes", !452, !474}
!474 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.9"}
!475 = !{!"FuncMDMap[5]", void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!476 = !{!"FuncMDValue[5]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !458, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !460, !384, !477, !480, !393, !396, !443, !402}
!477 = !{!"m_OpenCLArgTypes", !478, !479}
!478 = !{!"m_OpenCLArgTypesVec[0]", !"int*"}
!479 = !{!"m_OpenCLArgTypesVec[1]", !"int"}
!480 = !{!"m_OpenCLArgBaseTypes", !481, !482}
!481 = !{!"m_OpenCLArgBaseTypesVec[0]", !"int*"}
!482 = !{!"m_OpenCLArgBaseTypesVec[1]", !"int"}
!483 = !{!"FuncMDMap[6]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!484 = !{!"FuncMDValue[6]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !485, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !384, !487, !489, !393, !396, !454, !402}
!485 = !{!"resAllocMD", !328, !329, !330, !486, !360}
!486 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356}
!487 = !{!"m_OpenCLArgTypes", !449, !488}
!488 = !{!"m_OpenCLArgTypesVec[1]", !"class.__generated_.12"}
!489 = !{!"m_OpenCLArgBaseTypes", !452, !490}
!490 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.__generated_.12"}
!491 = !{!"FuncMDMap[7]", void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!492 = !{!"FuncMDValue[7]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !458, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !460, !384, !493, !495, !393, !396, !443, !402}
!493 = !{!"m_OpenCLArgTypes", !420, !494}
!494 = !{!"m_OpenCLArgTypesVec[1]", !"uchar"}
!495 = !{!"m_OpenCLArgBaseTypes", !427, !496}
!496 = !{!"m_OpenCLArgBaseTypesVec[1]", !"uchar"}
!497 = !{!"FuncMDMap[8]", void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE}
!498 = !{!"FuncMDValue[8]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !327, !361, !362, !363, !364, !365, !366, !367, !368, !499, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !500, !501, !502, !504, !506, !507, !443, !402}
!499 = !{!"privateMemoryPerWI", i32 112}
!500 = !{!"m_OpenCLArgAddressSpaces", !409, !383, !410}
!501 = !{!"m_OpenCLArgAccessQualifiers", !385, !386, !415}
!502 = !{!"m_OpenCLArgTypes", !462, !421, !503}
!503 = !{!"m_OpenCLArgTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!504 = !{!"m_OpenCLArgBaseTypes", !465, !428, !505}
!505 = !{!"m_OpenCLArgBaseTypesVec[2]", !"struct cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"}
!506 = !{!"m_OpenCLArgTypeQualifiers", !394, !395, !434}
!507 = !{!"m_OpenCLArgNames", !397, !398, !439}
!508 = !{!"FuncMDMap[9]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE}
!509 = !{!"FuncMDValue[9]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !510, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !521, !530, !538, !552, !566, !574, !582, !402}
!510 = !{!"resAllocMD", !328, !329, !330, !511, !360}
!511 = !{!"argAllocMDList", !332, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !512, !513, !514, !515, !516, !517, !518, !519, !520}
!512 = !{!"argAllocMDListVec[25]", !333, !334, !335}
!513 = !{!"argAllocMDListVec[26]", !333, !334, !335}
!514 = !{!"argAllocMDListVec[27]", !333, !334, !335}
!515 = !{!"argAllocMDListVec[28]", !333, !334, !335}
!516 = !{!"argAllocMDListVec[29]", !333, !334, !335}
!517 = !{!"argAllocMDListVec[30]", !333, !334, !335}
!518 = !{!"argAllocMDListVec[31]", !333, !334, !335}
!519 = !{!"argAllocMDListVec[32]", !333, !334, !335}
!520 = !{!"argAllocMDListVec[33]", !333, !334, !335}
!521 = !{!"m_OpenCLArgAddressSpaces", !382, !383, !410, !522, !412, !413, !523, !524, !525, !526, !527, !528, !529}
!522 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!523 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!524 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 0}
!525 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!526 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!527 = !{!"m_OpenCLArgAddressSpacesVec[10]", i32 0}
!528 = !{!"m_OpenCLArgAddressSpacesVec[11]", i32 0}
!529 = !{!"m_OpenCLArgAddressSpacesVec[12]", i32 0}
!530 = !{!"m_OpenCLArgAccessQualifiers", !385, !386, !415, !416, !417, !418, !531, !532, !533, !534, !535, !536, !537}
!531 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!532 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!533 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!534 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!535 = !{!"m_OpenCLArgAccessQualifiersVec[10]", !"none"}
!536 = !{!"m_OpenCLArgAccessQualifiersVec[11]", !"none"}
!537 = !{!"m_OpenCLArgAccessQualifiersVec[12]", !"none"}
!538 = !{!"m_OpenCLArgTypes", !539, !540, !541, !542, !543, !544, !545, !546, !547, !548, !549, !550, !551}
!539 = !{!"m_OpenCLArgTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!540 = !{!"m_OpenCLArgTypesVec[1]", !"float"}
!541 = !{!"m_OpenCLArgTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!542 = !{!"m_OpenCLArgTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!543 = !{!"m_OpenCLArgTypesVec[4]", !"float"}
!544 = !{!"m_OpenCLArgTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!545 = !{!"m_OpenCLArgTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!546 = !{!"m_OpenCLArgTypesVec[7]", !"float"}
!547 = !{!"m_OpenCLArgTypesVec[8]", !"int"}
!548 = !{!"m_OpenCLArgTypesVec[9]", !"long"}
!549 = !{!"m_OpenCLArgTypesVec[10]", !"long"}
!550 = !{!"m_OpenCLArgTypesVec[11]", !"long"}
!551 = !{!"m_OpenCLArgTypesVec[12]", !"long"}
!552 = !{!"m_OpenCLArgBaseTypes", !553, !554, !555, !556, !557, !558, !559, !560, !561, !562, !563, !564, !565}
!553 = !{!"m_OpenCLArgBaseTypesVec[0]", !"struct cutlass::gemm::GemmCoord"}
!554 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float"}
!555 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.cutlass::__generated_TensorRef"}
!556 = !{!"m_OpenCLArgBaseTypesVec[3]", !"class.cutlass::__generated_TensorRef"}
!557 = !{!"m_OpenCLArgBaseTypesVec[4]", !"float"}
!558 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.cutlass::__generated_TensorRef"}
!559 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.cutlass::__generated_TensorRef"}
!560 = !{!"m_OpenCLArgBaseTypesVec[7]", !"float"}
!561 = !{!"m_OpenCLArgBaseTypesVec[8]", !"int"}
!562 = !{!"m_OpenCLArgBaseTypesVec[9]", !"long"}
!563 = !{!"m_OpenCLArgBaseTypesVec[10]", !"long"}
!564 = !{!"m_OpenCLArgBaseTypesVec[11]", !"long"}
!565 = !{!"m_OpenCLArgBaseTypesVec[12]", !"long"}
!566 = !{!"m_OpenCLArgTypeQualifiers", !394, !395, !434, !435, !436, !437, !567, !568, !569, !570, !571, !572, !573}
!567 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!568 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!569 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!570 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!571 = !{!"m_OpenCLArgTypeQualifiersVec[10]", !""}
!572 = !{!"m_OpenCLArgTypeQualifiersVec[11]", !""}
!573 = !{!"m_OpenCLArgTypeQualifiersVec[12]", !""}
!574 = !{!"m_OpenCLArgNames", !397, !398, !439, !440, !441, !442, !575, !576, !577, !578, !579, !580, !581}
!575 = !{!"m_OpenCLArgNamesVec[6]", !""}
!576 = !{!"m_OpenCLArgNamesVec[7]", !""}
!577 = !{!"m_OpenCLArgNamesVec[8]", !""}
!578 = !{!"m_OpenCLArgNamesVec[9]", !""}
!579 = !{!"m_OpenCLArgNamesVec[10]", !""}
!580 = !{!"m_OpenCLArgNamesVec[11]", !""}
!581 = !{!"m_OpenCLArgNamesVec[12]", !""}
!582 = !{!"m_OpenCLArgScalarAsPointers", !583, !584, !585, !586}
!583 = !{!"m_OpenCLArgScalarAsPointersSet[0]", i32 25}
!584 = !{!"m_OpenCLArgScalarAsPointersSet[1]", i32 27}
!585 = !{!"m_OpenCLArgScalarAsPointersSet[2]", i32 29}
!586 = !{!"m_OpenCLArgScalarAsPointersSet[3]", i32 31}
!587 = !{!"FuncMDMap[10]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!588 = !{!"FuncMDValue[10]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !510, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !521, !530, !538, !552, !566, !574, !582, !402}
!589 = !{!"FuncMDMap[11]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE}
!590 = !{!"FuncMDValue[11]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !510, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !521, !530, !538, !552, !566, !574, !582, !402}
!591 = !{!"FuncMDMap[12]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!592 = !{!"FuncMDValue[12]", !292, !293, !297, !298, !299, !300, !301, !302, !303, !510, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !521, !530, !538, !552, !566, !574, !582, !402}
!593 = !{!"FuncMDMap[13]", void ()* @Intel_Symbol_Table_Void_Program}
!594 = !{!"FuncMDValue[13]", !292, !595, !297, !298, !299, !300, !301, !302, !303, !598, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !600, !601, !602, !603, !604, !605, !443, !402}
!595 = !{!"workGroupWalkOrder", !294, !596, !597}
!596 = !{!"dim1", i32 0}
!597 = !{!"dim2", i32 0}
!598 = !{!"resAllocMD", !328, !329, !330, !599, !360}
!599 = !{!"argAllocMDList"}
!600 = !{!"m_OpenCLArgAddressSpaces"}
!601 = !{!"m_OpenCLArgAccessQualifiers"}
!602 = !{!"m_OpenCLArgTypes"}
!603 = !{!"m_OpenCLArgBaseTypes"}
!604 = !{!"m_OpenCLArgTypeQualifiers"}
!605 = !{!"m_OpenCLArgNames"}
!606 = !{!"pushInfo", !607, !608, !609, !613, !614, !615, !616, !617, !618, !619, !620, !633, !634, !635, !636}
!607 = !{!"pushableAddresses"}
!608 = !{!"bindlessPushInfo"}
!609 = !{!"dynamicBufferInfo", !610, !611, !612}
!610 = !{!"firstIndex", i32 0}
!611 = !{!"numOffsets", i32 0}
!612 = !{!"forceDisabled", i1 false}
!613 = !{!"MaxNumberOfPushedBuffers", i32 0}
!614 = !{!"inlineConstantBufferSlot", i32 -1}
!615 = !{!"inlineConstantBufferOffset", i32 -1}
!616 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!617 = !{!"constants"}
!618 = !{!"inputs"}
!619 = !{!"constantReg"}
!620 = !{!"simplePushInfoArr", !621, !630, !631, !632}
!621 = !{!"simplePushInfoArrVec[0]", !622, !623, !624, !625, !626, !627, !628, !629}
!622 = !{!"cbIdx", i32 0}
!623 = !{!"pushableAddressGrfOffset", i32 -1}
!624 = !{!"pushableOffsetGrfOffset", i32 -1}
!625 = !{!"offset", i32 0}
!626 = !{!"size", i32 0}
!627 = !{!"isStateless", i1 false}
!628 = !{!"isBindless", i1 false}
!629 = !{!"simplePushLoads"}
!630 = !{!"simplePushInfoArrVec[1]", !622, !623, !624, !625, !626, !627, !628, !629}
!631 = !{!"simplePushInfoArrVec[2]", !622, !623, !624, !625, !626, !627, !628, !629}
!632 = !{!"simplePushInfoArrVec[3]", !622, !623, !624, !625, !626, !627, !628, !629}
!633 = !{!"simplePushBufferUsed", i32 0}
!634 = !{!"pushAnalysisWIInfos"}
!635 = !{!"inlineRTGlobalPtrOffset", i32 0}
!636 = !{!"rtSyncSurfPtrOffset", i32 0}
!637 = !{!"pISAInfo", !638, !639, !643, !644, !652, !656, !658}
!638 = !{!"shaderType", !"UNKNOWN"}
!639 = !{!"geometryInfo", !640, !641, !642}
!640 = !{!"needsVertexHandles", i1 false}
!641 = !{!"needsPrimitiveIDEnable", i1 false}
!642 = !{!"VertexCount", i32 0}
!643 = !{!"hullInfo", !640, !641}
!644 = !{!"pixelInfo", !645, !646, !647, !648, !649, !650, !651}
!645 = !{!"perPolyStartGrf", i32 0}
!646 = !{!"hasZWDeltaOrPerspBaryPlanes", i1 false}
!647 = !{!"hasNonPerspBaryPlanes", i1 false}
!648 = !{!"maxPerPrimConstDataId", i32 -1}
!649 = !{!"maxSetupId", i32 -1}
!650 = !{!"hasVMask", i1 false}
!651 = !{!"PixelGRFBitmask", i32 0}
!652 = !{!"domainInfo", !653, !654, !655}
!653 = !{!"DomainPointUArgIdx", i32 -1}
!654 = !{!"DomainPointVArgIdx", i32 -1}
!655 = !{!"DomainPointWArgIdx", i32 -1}
!656 = !{!"computeInfo", !657}
!657 = !{!"EnableHWGenerateLID", i1 true}
!658 = !{!"URBOutputLength", i32 0}
!659 = !{!"WaEnableICBPromotion", i1 false}
!660 = !{!"vsInfo", !661, !662, !663}
!661 = !{!"DrawIndirectBufferIndex", i32 -1}
!662 = !{!"vertexReordering", i32 -1}
!663 = !{!"MaxNumOfOutputs", i32 0}
!664 = !{!"hsInfo", !665, !666}
!665 = !{!"numPatchAttributesPatchBaseName", !""}
!666 = !{!"numVertexAttributesPatchBaseName", !""}
!667 = !{!"dsInfo", !663}
!668 = !{!"gsInfo", !663}
!669 = !{!"psInfo", !670, !671, !672, !673, !674, !675, !676, !677, !678, !679, !680, !681, !682, !683, !684, !685, !686, !687, !688, !689, !690, !691, !692, !693, !694, !695, !696, !697, !698, !699, !700, !701, !702, !703, !704, !705, !706, !707}
!670 = !{!"BlendStateDisabledMask", i8 0}
!671 = !{!"SkipSrc0Alpha", i1 false}
!672 = !{!"DualSourceBlendingDisabled", i1 false}
!673 = !{!"ForceEnableSimd32", i1 false}
!674 = !{!"DisableSimd32WithDiscard", i1 false}
!675 = !{!"outputDepth", i1 false}
!676 = !{!"outputStencil", i1 false}
!677 = !{!"outputMask", i1 false}
!678 = !{!"blendToFillEnabled", i1 false}
!679 = !{!"forceEarlyZ", i1 false}
!680 = !{!"hasVersionedLoop", i1 false}
!681 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!682 = !{!"requestCPSizeRelevant", i1 false}
!683 = !{!"requestCPSize", i1 false}
!684 = !{!"texelMaskFastClearMode", !"Disabled"}
!685 = !{!"NumSamples", i8 0}
!686 = !{!"blendOptimizationMode"}
!687 = !{!"colorOutputMask"}
!688 = !{!"ProvokingVertexModeNosIndex", i32 0}
!689 = !{!"ProvokingVertexModeNosPatch", !""}
!690 = !{!"ProvokingVertexModeLast", !"Negative"}
!691 = !{!"VertexAttributesBypass", i1 false}
!692 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!693 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!694 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!695 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!696 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!697 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!698 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!699 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!700 = !{!"generatePatchesForRTWriteSends", i1 false}
!701 = !{!"generatePatchesForRT_BTIndex", i1 false}
!702 = !{!"forceVMask", i1 false}
!703 = !{!"isNumPerPrimAttributesSet", i1 false}
!704 = !{!"numPerPrimAttributes", i32 0}
!705 = !{!"WaDisableVRS", i1 false}
!706 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!707 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!708 = !{!"csInfo", !709, !710, !711, !712, !191, !167, !168, !713, !169, !714, !715, !716, !717, !718, !719, !720, !721, !722, !723, !724, !202, !725, !726, !727, !728, !729, !730, !731, !732}
!709 = !{!"maxWorkGroupSize", i32 0}
!710 = !{!"waveSize", i32 0}
!711 = !{!"ComputeShaderSecondCompile"}
!712 = !{!"forcedSIMDSize", i8 0}
!713 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!714 = !{!"forceSpillCompression", i1 false}
!715 = !{!"allowLowerSimd", i1 false}
!716 = !{!"disableSimd32Slicing", i1 false}
!717 = !{!"disableSplitOnSpill", i1 false}
!718 = !{!"enableNewSpillCostFunction", i1 false}
!719 = !{!"forceVISAPreSched", i1 false}
!720 = !{!"disableLocalIdOrderOptimizations", i1 false}
!721 = !{!"disableDispatchAlongY", i1 false}
!722 = !{!"neededThreadIdLayout", i1* null}
!723 = !{!"forceTileYWalk", i1 false}
!724 = !{!"atomicBranch", i32 0}
!725 = !{!"disableEarlyOut", i1 false}
!726 = !{!"walkOrderEnabled", i1 false}
!727 = !{!"walkOrderOverride", i32 0}
!728 = !{!"ResForHfPacking"}
!729 = !{!"constantFoldSimdSize", i1 false}
!730 = !{!"isNodeShader", i1 false}
!731 = !{!"threadGroupMergeSize", i32 0}
!732 = !{!"threadGroupMergeOverY", i1 false}
!733 = !{!"msInfo", !734, !735, !736, !737, !738, !739, !740, !741, !742, !743, !744, !690, !688, !745, !746, !730}
!734 = !{!"PrimitiveTopology", i32 3}
!735 = !{!"MaxNumOfPrimitives", i32 0}
!736 = !{!"MaxNumOfVertices", i32 0}
!737 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!738 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!739 = !{!"WorkGroupSize", i32 0}
!740 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!741 = !{!"IndexFormat", i32 6}
!742 = !{!"SubgroupSize", i32 0}
!743 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!744 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!745 = !{!"Is16BMUEModeAllowed", i1 false}
!746 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!747 = !{!"taskInfo", !663, !739, !740, !742}
!748 = !{!"NBarrierCnt", i32 0}
!749 = !{!"rtInfo", !750, !751, !752, !753, !754, !755, !756, !757, !758, !759, !760, !761, !762, !763, !764, !765, !323}
!750 = !{!"RayQueryAllocSizeInBytes", i32 0}
!751 = !{!"NumContinuations", i32 0}
!752 = !{!"RTAsyncStackAddrspace", i32 -1}
!753 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!754 = !{!"SWHotZoneAddrspace", i32 -1}
!755 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!756 = !{!"SWStackAddrspace", i32 -1}
!757 = !{!"SWStackSurfaceStateOffset", i1* null}
!758 = !{!"RTSyncStackAddrspace", i32 -1}
!759 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!760 = !{!"doSyncDispatchRays", i1 false}
!761 = !{!"MemStyle", !"Xe"}
!762 = !{!"GlobalDataStyle", !"Xe"}
!763 = !{!"NeedsBTD", i1 true}
!764 = !{!"SERHitObjectFullType", i1* null}
!765 = !{!"uberTileDimensions", i1* null}
!766 = !{!"CurUniqueIndirectIdx", i32 0}
!767 = !{!"inlineDynTextures"}
!768 = !{!"inlineResInfoData"}
!769 = !{!"immConstant", !770, !771, !772}
!770 = !{!"data"}
!771 = !{!"sizes"}
!772 = !{!"zeroIdxs"}
!773 = !{!"stringConstants"}
!774 = !{!"inlineBuffers", !775, !779, !781}
!775 = !{!"inlineBuffersVec[0]", !776, !777, !778}
!776 = !{!"alignment", i32 0}
!777 = !{!"allocSize", i64 64}
!778 = !{!"Buffer"}
!779 = !{!"inlineBuffersVec[1]", !776, !780, !778}
!780 = !{!"allocSize", i64 0}
!781 = !{!"inlineBuffersVec[2]", !776, !780, !778}
!782 = !{!"GlobalPointerProgramBinaryInfos"}
!783 = !{!"ConstantPointerProgramBinaryInfos"}
!784 = !{!"GlobalBufferAddressRelocInfo"}
!785 = !{!"ConstantBufferAddressRelocInfo"}
!786 = !{!"forceLscCacheList"}
!787 = !{!"SrvMap"}
!788 = !{!"RootConstantBufferOffsetInBytes"}
!789 = !{!"RasterizerOrderedByteAddressBuffer"}
!790 = !{!"RasterizerOrderedViews"}
!791 = !{!"MinNOSPushConstantSize", i32 2}
!792 = !{!"inlineProgramScopeOffsets", !793, !794, !795, !796}
!793 = !{!"inlineProgramScopeOffsetsMap[0]", [36 x i8]* @gVar}
!794 = !{!"inlineProgramScopeOffsetsValue[0]", i64 0}
!795 = !{!"inlineProgramScopeOffsetsMap[1]", [24 x i8]* @gVar.61}
!796 = !{!"inlineProgramScopeOffsetsValue[1]", i64 40}
!797 = !{!"shaderData", !798}
!798 = !{!"numReplicas", i32 0}
!799 = !{!"URBInfo", !800, !801, !802}
!800 = !{!"has64BVertexHeaderInput", i1 false}
!801 = !{!"has64BVertexHeaderOutput", i1 false}
!802 = !{!"hasVertexHeader", i1 true}
!803 = !{!"m_ForcePullModel", i1 false}
!804 = !{!"UseBindlessImage", i1 true}
!805 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!806 = !{!"enableRangeReduce", i1 false}
!807 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!808 = !{!"enableFRemToSRemOpt", i1 false}
!809 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!810 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!811 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!812 = !{!"allowMatchMadOptimizationforVS", i1 false}
!813 = !{!"disableMatchMadOptimizationForCS", i1 false}
!814 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!815 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!816 = !{!"statefulResourcesNotAliased", i1 false}
!817 = !{!"disableMixMode", i1 false}
!818 = !{!"genericAccessesResolved", i1 false}
!819 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!820 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!821 = !{!"disableSeparateScratchWA", i1 false}
!822 = !{!"enableRemoveUnusedTGMFence", i1 false}
!823 = !{!"PrivateMemoryPerFG", !824, !825, !826, !827, !828, !829, !830, !831, !832, !833, !834, !835, !836, !837, !838, !839, !840, !841, !842, !843, !844, !845, !846, !847, !848, !849, !850, !851}
!824 = !{!"PrivateMemoryPerFGMap[0]", void (%"class.sycl::_V1::range"*, %class.__generated_*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE}
!825 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!826 = !{!"PrivateMemoryPerFGMap[1]", void (i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, i8 addrspace(1)*, i64, %"class.sycl::_V1::range"*, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i64, i64, i64, i64, i32, i32, i32, i32, i32)* @_ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail}
!827 = !{!"PrivateMemoryPerFGValue[1]", i32 0}
!828 = !{!"PrivateMemoryPerFGMap[2]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.2*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i16, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!829 = !{!"PrivateMemoryPerFGValue[2]", i32 0}
!830 = !{!"PrivateMemoryPerFGMap[3]", void (i16 addrspace(1)*, i16, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillItEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!831 = !{!"PrivateMemoryPerFGValue[3]", i32 0}
!832 = !{!"PrivateMemoryPerFGMap[4]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.9*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i32, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!833 = !{!"PrivateMemoryPerFGValue[4]", i32 0}
!834 = !{!"PrivateMemoryPerFGMap[5]", void (i32 addrspace(1)*, i32, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!835 = !{!"PrivateMemoryPerFGValue[5]", i32 0}
!836 = !{!"PrivateMemoryPerFGMap[6]", void (%"class.sycl::_V1::range.0"*, %class.__generated_.12*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8, i32)* @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_EE}
!837 = !{!"PrivateMemoryPerFGValue[6]", i32 0}
!838 = !{!"PrivateMemoryPerFGMap[7]", void (i8 addrspace(1)*, i8, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i32, i32, i32)* @_ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_}
!839 = !{!"PrivateMemoryPerFGValue[7]", i32 0}
!840 = !{!"PrivateMemoryPerFGMap[8]", void (i16 addrspace(1)*, i64, %"struct.cutlass::reference::device::detail::RandomUniformFunc<cutlass::bfloat16_t>::Params"*, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i64, float, float, i32, float, float, i8, i8, i8, i8, i32, i32, i32)* @_ZTSN7cutlass9reference6device22BlockForEachKernelNameINS_10bfloat16_tENS1_6detail17RandomUniformFuncIS3_EEEE}
!841 = !{!"PrivateMemoryPerFGValue[8]", i32 112}
!842 = !{!"PrivateMemoryPerFGMap[9]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSC_EEE}
!843 = !{!"PrivateMemoryPerFGValue[9]", i32 0}
!844 = !{!"PrivateMemoryPerFGMap[10]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_NS4_11ColumnMajorEfS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!845 = !{!"PrivateMemoryPerFGValue[10]", i32 0}
!846 = !{!"PrivateMemoryPerFGMap[11]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEKiSB_EEE}
!847 = !{!"PrivateMemoryPerFGValue[11]", i32 0}
!848 = !{!"PrivateMemoryPerFGMap[12]", void (%"struct.cutlass::gemm::GemmCoord"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, %"class.cutlass::__generated_TensorRef"*, %"class.cutlass::__generated_TensorRef"*, float, i32, i64, i64, i64, i64, <8 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8 addrspace(2)*, i8*, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i32)* @_ZTSN7cutlass9reference6device21GemmComplexKernelNameIJNS_10bfloat16_tENS_6layout8RowMajorES3_S5_fS5_fffS5_NS_16NumericConverterIffLNS_15FloatRoundStyleE2EEENS_12multiply_addIfffEEEEE}
!849 = !{!"PrivateMemoryPerFGValue[12]", i32 0}
!850 = !{!"PrivateMemoryPerFGMap[13]", void ()* @Intel_Symbol_Table_Void_Program}
!851 = !{!"PrivateMemoryPerFGValue[13]", i32 0}
!852 = !{!"m_OptsToDisable"}
!853 = !{!"capabilities", !854}
!854 = !{!"globalVariableDecorationsINTEL", i1 false}
!855 = !{!"extensions", !856}
!856 = !{!"spvINTELBindlessImages", i1 false}
!857 = !{!"m_ShaderResourceViewMcsMask", !858, !859}
!858 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!859 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!860 = !{!"computedDepthMode", i32 0}
!861 = !{!"isHDCFastClearShader", i1 false}
!862 = !{!"argRegisterReservations", !863}
!863 = !{!"argRegisterReservationsVec[0]", i32 0}
!864 = !{!"SIMD16_SpillThreshold", i8 0}
!865 = !{!"SIMD32_SpillThreshold", i8 0}
!866 = !{!"m_CacheControlOption", !867, !868, !869, !870}
!867 = !{!"LscLoadCacheControlOverride", i8 0}
!868 = !{!"LscStoreCacheControlOverride", i8 0}
!869 = !{!"TgmLoadCacheControlOverride", i8 0}
!870 = !{!"TgmStoreCacheControlOverride", i8 0}
!871 = !{!"ModuleUsesBindless", i1 false}
!872 = !{!"predicationMap"}
!873 = !{!"lifeTimeStartMap"}
!874 = !{!"HitGroups"}
!875 = !{i32 2, i32 0}
!876 = !{!"clang version 16.0.6"}
!877 = !{i32 1, !"wchar_size", i32 4}
!878 = !{!"160"}
!879 = !{!"-4"}
!880 = !{!"80"}
!881 = !{!"10240"}
!882 = !{!"5120"}
!883 = !{!"2560"}
!884 = !{!"7680"}
!885 = !{!"2480"}
!886 = !{!"10160"}
!887 = !{!"-3"}
!888 = !{null}
!889 = !{!"CannotUseSOALayout"}
!890 = !{!"11529215046068469760"}
!891 = !{!"-60"}
!892 = !{!"16470307208669242514"}
!893 = !{!"-62"}
!894 = !{!895}
!895 = !{i32 4469}
!896 = !{!"13999761127368856138"}
!897 = !{!"-61"}
!898 = !{!899}
!899 = !{i32 40, i32 196620}
!900 = !{!901}
!901 = distinct !{!901, !902}
!902 = distinct !{!902}
!903 = !{!"11709359031163289600"}
!904 = !{!"-56"}
!905 = !{!906, !901}
!906 = distinct !{!906, !907}
!907 = distinct !{!907}
!908 = !{!"14617397647693952731"}
!909 = !{!"-57"}
!910 = !{!"17499701409211070171"}
!911 = !{!"-58"}
!912 = !{!"10911578525743373166"}
!913 = !{!"13176245766935394011"}
!914 = !{!"-59"}
!915 = !{!"9882184325201545509"}
!916 = !{!"11323336205960104229"}
!917 = !{!"960"}
!918 = !{!"1600"}
!919 = !{!"51200"}
!920 = !{!"25600"}
!921 = !{!922}
!922 = !{i32 4470}
!923 = !{!895, !922}
!924 = !{!"49600"}
!925 = !{!"1280"}
!926 = !{!"480"}
!927 = !{!"800"}

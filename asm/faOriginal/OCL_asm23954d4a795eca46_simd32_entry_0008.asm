//.kernel _ZTSZN4sycl3_V17handler4fillIhEEvPvRKT_mEUlNS0_2idILi1EEEE_
//.platform XE2
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.options_string "-emitCrossThreadOffR0Reloc -perfmodel -hashmovs 596987210 2036255302 -hashmovs1 0 8 "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -TotalGRFNum 256 -abortOnSpill 4 -enableBCR -forceBCR -enableBundleCR 3 -perfmodel -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -hashmovs 596987210 2036255302 -hashmovs1 0 8 "
//.instCount 33
//.RA type	LOCAL_ROUND_ROBIN_BC_RA
//.git-hash 

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r2.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r1.8)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r1.9)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r1.4)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r1.5)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r1.6)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r1.7)
//.declare %tsc (22)  rf=r size=20 type=ud align=2 words
//.declare %arg (23)  rf=r size=0 type=ud align=32 words (r26.0)
//.declare %retval (24)  rf=r size=0 type=ud align=32 words (r26.0) Output
//.declare %sp (25)  rf=r size=8 type=uq align=4 words (r255.3)
//.declare %fp (26)  rf=r size=8 type=uq align=4 words (r255.2)
//.declare %sr0 (27)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (28)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (29)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (30)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (32)  rf=r size=8 type=uq align=4 words (r254.0)
//.declare localIdBufPtr (33)  rf=r size=8 type=uq align=4 words (r254.3)
//.declare %msg0 (34)  rf=r size=12 type=ud align=2 words
//.declare %scratchloc (35)  rf=r size=8 type=uq align=4 words (s0.7)
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r2.0)
//.declare V0034 (44)  rf=r size=8 type=uq align=4 words (r4.3)
//.declare V0035 (45)  rf=r size=1 type=b align=1 words (r5.0)
//.declare V0037 (47)  rf=r size=32 type=d alias=+0 align=32 words (r2.0)
//.declare V0038 (48)  rf=r size=12 type=d align=2 words (r4.0)
//.declare V0039 (49)  rf=r size=12 type=d align=2 words (r4.3)
//.declare V0040 (50)  rf=r size=64 type=w align=32 words (r1.0)
//.declare V0041 (51)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0042 (52)  rf=r size=64 type=w align=32 words (r3.0)
//.declare V0043 (53)  rf=r size=8 type=uq align=4 words (r5.1)
//.declare V0047 (57)  rf=r size=4 type=d align=32 words (r3.0)
//.declare V0049 (59)  rf=r size=64 type=uw alias=V0040+0 align=32 words (r1.0)
//.declare V0050 (60)  rf=r size=128 type=d align=32 words (r6.0)
//.declare V0052 (62)  rf=r size=128 type=ud alias=V0050+0 align=32 words (r6.0)
//.declare V0053 (63)  rf=r size=8 type=q alias=V0034+0 align=4 words (r4.3)
//.declare V0054 (64)  rf=r size=256 type=q align=32 words (r8.0)
//.declare V0055 (65)  rf=r size=256 type=uq alias=V0054+0 align=32 words (r8.0)
//.declare V0056 (66)  rf=r size=1 type=ub alias=V0035+0 align=1 words (r5.0)
//.declare V0057 (67)  rf=r size=128 type=ud align=32 words (r12.0)
//.declare  (68)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare  (69)  rf=r size=128 type=ud align=32 words (r14.0)
//.declare  (70)  rf=r size=128 type=ud align=32 words (r16.0)
//.declare r0 (73)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (74)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare inlineRegFromTDL (75)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (76)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (77)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (78)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (79)  rf=r size=32 type=ud align=2 words (r5.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0040    | :w x 32  |   0x40 | r1       | pti[tid]+0x0     |
// | V0041    | :w x 32  |   0x40 | r2       | pti[tid]+0x40    |
// | V0042    | :w x 32  |   0x40 | r3       | pti[tid]+0x80    |
// | V0038    | :d x 3   |    0xC | r4       | inline+0x0       |
// | V0039    | :d x 3   |    0xC | r4+0xC   | inline+0xC       |
// | V0034    | :uq      |    0x8 | r4+0x18  | inline+0x18      |
// | V0035    | :b       |    0x1 | r5       | cti+0x20         |
// | V0043    | :uq      |    0x8 | r5+0x8   | cti+0x28         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r255.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r255.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x20:ud              {I@2}          //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     mad (1|M0)               r255.0<1>:ud  r255.2<0;0>:ud    r255.0<0;0>:uw    0xC0:uw              {I@1} //  ALU pipe: int; 
(W)     mov (8|M0)               r4.0<1>:ud    r1.0<1;1,0>:ud                                        //  ALU pipe: int; 
(W)     load.ugm.d32x32t.a32.ca.cc (1|M0)  r1:2 bti[255][r255:1]   {A@1,$0} // ex_desc:0xFF000000; desc:0x6229E500 // 
(W)     load.ugm.d32x16t.a32.ca.cc (1|M0)  r3:1 bti[255][r255:1+0x80]  {$1} // ex_desc:0xFF080000; desc:0x6219D500 // 
        nop                                                                                          // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
        sync.nop                             null                             {Compacted,$1.src}     // 
(W)     and (1|M0)               r255.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud              {$0.src} //  ALU pipe: int; 
(W)     add (1|M0)               r255.0<1>:ud  r255.0<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
(W)     load.ugm.d32x8t.a32.ca.cc (1|M0)  r5:1  bti[255][r255:1]   {I@1,$2} // ex_desc:0xFF000000; desc:0x6219C500 // 
// B002: Preds:{B001},  Succs:{}
// _main:
(W)     mov (16|M0)              r2.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted,$0.dst}   //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mul (1|M0)               acc0.0<1>:d   r4.3<0;1,0>:d     r2.2<0;1,0>:uw   {A@1}              //  ALU pipe: int; $2
        mov (32|M0)              r12.0<1>:ud   r5.0<0;1,0>:ub                   {$2.dst}             //  ALU pipe: int; $7
(W)     macl (1|M0)              r3.0<1>:d     r4.3<0;1,0>:d     r2.1<0;1,0>:d    {Compacted,$1.dst} //  ALU pipe: int; $4
(W)     mov (16|M0)              r255.0<1>:f   r2.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $9
        add3 (32|M0)             r6.0<1>:d     r3.0<0;0>:d       r1.0<1;0>:uw      r4.0<0>:d        {I@1} //  ALU pipe: int; $4
        mov (16|M0)              r14.0<2>:ud   r6.0<1;1,0>:ud                   {Compacted,I@1}      //  ALU pipe: int; $6
        mov (16|M16)             r16.0<2>:ud   r7.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $6
        add (16|M0)              r8.0<1>:q     r4.3<0;1,0>:q     r14.0<2;1,0>:ud  {I@2}              //  ALU pipe: int; $6
        add (16|M16)             r10.0<1>:q    r4.3<0;1,0>:q     r16.0<2;1,0>:ud  {I@2}              //  ALU pipe: int; $6
        store.ugm.d8u32.a64 (32|M0)  [r8:4]     r12:2              {I@1,$3} // ex_desc:0x0; desc:0x8000984 // $8
(W)     send.gtwy (1|M0)         null     r255  null:0  0x0            0x02000010           {EOT,F@1,$4} // wr:1+0, rd:0; end of thread // $9
L416:
(W)     mov (16|M0)              null<1>:ud    0x23954D4A:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x795ECA46:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x8:ud                                                // 


//.BankConflicts: 0
//.ByteRMWs: 0
//


//.numALUInst: 25
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 8
//.allAtOneDistNum: 3
//.syncInstCount: 1
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 3
//.AfterReadTokenDepCount: 2

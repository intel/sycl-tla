//.kernel _ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZNS0_7handler4fillIjEEvPvRKT_mEUlNS0_2idILi1EEEE_EE
//.platform XE2
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.options_string "-emitCrossThreadOffR0Reloc -perfmodel -hashmovs 596987210 2036255302 -hashmovs1 0 5 "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -TotalGRFNum 256 -abortOnSpill 4 -enableBCR -enableBundleCR 3 -perfmodel -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -hashmovs 596987210 2036255302 -hashmovs1 0 5 "
//.instCount 69
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r2.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r2.3)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r2.4)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r1.15)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r2.0)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r2.1)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r2.2)
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
//.declare V0035 (45)  rf=r size=32 type=d alias=+0 align=32 words (r2.0)
//.declare V0036 (46)  rf=r size=12 type=d align=2 words (r4.0)
//.declare V0037 (47)  rf=r size=12 type=d align=2 words (r5.12)
//.declare V0038 (48)  rf=r size=12 type=d align=2 words (r4.3)
//.declare V0039 (49)  rf=r size=64 type=w align=32 words (r1.0)
//.declare V0040 (50)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0041 (51)  rf=r size=64 type=w align=32 words (r3.0)
//.declare V0042 (52)  rf=r size=8 type=uq align=4 words (r5.4)
//.declare V0043 (53)  rf=r size=8 type=uq align=4 words (r5.5)
//.declare V0044 (54)  rf=r size=8 type=q align=4 words (r4.3)
//.declare V0045 (55)  rf=r size=8 type=q align=4 words (r5.0)
//.declare V0046 (56)  rf=r size=4 type=d align=2 words (r5.2)
//.declare V0047 (57)  rf=r size=1 type=b align=2 words (r5.12)
//.declare V0048 (58)  rf=r size=1 type=b align=2 words (r5.16)
//.declare V0049 (59)  rf=r size=1 type=b align=2 words (r5.20)
//.declare V0050 (60)  rf=r size=1 type=b align=2 words (r5.24)
//.declare V0052 (62)  rf=r size=8 type=d align=2 words (r3.0)
//.declare V0053 (63)  rf=r size=8 type=d alias=V0044+0 align=4 words (r4.6)
//.declare V0057 (67)  rf=r size=12 type=ud alias=V0038+0 align=2 words (r4.3)
//.declare V0058 (68)  rf=r size=32 type=ud alias=V0035+0 align=16 words (r2.0)
//.declare V0059 (69)  rf=r size=4 type=d align=32 words (r6.0)
//.declare V0063 (73)  rf=r size=8 type=q align=32 words (r7.0)
//.declare V0064 (74)  rf=r size=8 type=d alias=V0063+0 align=4 words (r7.0)
//.declare V0066 (76)  rf=r size=64 type=uw alias=V0039+0 align=32 words (r1.0)
//.declare V0069 (79)  rf=r size=12 type=ud alias=V0036+0 align=2 words (r4.0)
//.declare V0073 (83)  rf=r size=128 type=d align=32 words (r8.0)
//.declare V0074 (84)  rf=r size=128 type=d align=32 words (r10.0)
//.declare P01 (85)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0075 (86)  rf=r size=128 type=ud alias=V0073+0 align=32 words (r8.0)
//.declare V0076 (87)  rf=r size=8 type=ud alias=V0052+0 align=2 words (r3.0)
//.declare P02 (88)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare P03 (89)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0077 (90)  rf=r size=128 type=ud alias=V0074+0 align=32 words (r10.0)
//.declare V0078 (91)  rf=r size=8 type=q align=4 words (r3.1)
//.declare V0079 (92)  rf=r size=12 type=ud alias=V0037+0 align=2 words (r5.12)
//.declare V0083 (96)  rf=r size=256 type=uq align=32 words (r12.0)
//.declare V0084 (97)  rf=r size=128 type=d align=32 words (r16.0)
//.declare V0088 (101)  rf=r size=128 type=d align=32 words (r18.0)
//.declare V0089 (102)  rf=r size=128 type=d align=32 words (r20.0)
//.declare P04 (103)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0090 (104)  rf=r size=128 type=ud alias=V0088+0 align=32 words (r18.0)
//.declare P05 (105)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare P06 (106)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare V0091 (107)  rf=r size=128 type=ud alias=V0089+0 align=32 words (r20.0)
//.declare  (111)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare  (112)  rf=r size=128 type=uw align=32 words (r22.0)
//.declare  (113)  rf=r size=128 type=uw align=32 words (r24.0)
//.declare  (116)  rf=r size=128 type=q align=32 words (r26.0)
//.declare  (117)  rf=r size=128 type=q align=32 words (r28.0)
//.declare  (118)  rf=r size=128 type=q align=32 words (r30.0)
//.declare  (119)  rf=r size=128 type=q align=32 words (r32.0)
//.declare  (122)  rf=r size=128 type=q align=32 words (r34.0)
//.declare  (123)  rf=r size=128 type=q align=32 words (r36.0)
//.declare  (126)  rf=r size=128 type=q align=32 words (r38.0)
//.declare  (127)  rf=r size=128 type=q align=32 words (r40.0)
//.declare  (130)  rf=r size=128 type=d align=32 words (r42.0)
//.declare  (131)  rf=r size=128 type=d align=32 words (r44.0)
//.declare  (132)  rf=r size=128 type=d alias=+0 align=32 words (r30.0)
//.declare  (133)  rf=r size=128 type=d alias=+0 align=32 words (r32.0)
//.declare  (134)  rf=r size=128 type=d alias=+0 align=32 words (r38.0)
//.declare  (135)  rf=r size=128 type=d alias=+0 align=32 words (r40.0)
//.declare r0 (136)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (137)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare inlineRegFromTDL (138)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (139)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (140)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (141)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (142)  rf=r size=64 type=ud align=32 words (r5.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0039    | :w x 32  |   0x40 | r1       | pti[tid]+0x0     |
// | V0040    | :w x 32  |   0x40 | r2       | pti[tid]+0x40    |
// | V0041    | :w x 32  |   0x40 | r3       | pti[tid]+0x80    |
// | V0036    | :d x 3   |    0xC | r4       | inline+0x0       |
// | V0038    | :d x 3   |    0xC | r4+0xC   | inline+0xC       |
// | V0044    | :q       |    0x8 | r4+0x18  | inline+0x18      |
// | V0045    | :q       |    0x8 | r5       | cti+0x20         |
// | V0046    | :d       |    0x4 | r5+0x8   | cti+0x28         |
// | V0047    | :b       |    0x1 | r5+0xC   | cti+0x2C         |
// | V0048    | :b       |    0x1 | r5+0x10  | cti+0x30         |
// | V0049    | :b       |    0x1 | r5+0x14  | cti+0x34         |
// | V0050    | :b       |    0x1 | r5+0x18  | cti+0x38         |
// | V0042    | :uq      |    0x8 | r5+0x20  | cti+0x40         |
// | V0043    | :uq      |    0x8 | r5+0x28  | cti+0x48         |
// | V0037    | :d x 3   |    0xC | r5+0x30  | cti+0x50         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r255.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r255.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x40:ud              {I@2}          //  ALU pipe: int; 
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
(W)     load.ugm.d32x16t.a32.ca.cc (1|M0)  r5:1 bti[255][r255:1]   {I@1,$2} // ex_desc:0xFF000000; desc:0x6219D500 // 
// B002: Preds:{B001},  Succs:{B003, B006}
// _main:
(W)     mov (16|M0)              r2.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted,$0.dst}   //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mul (1|M0)               acc0.0<1>:ud  r4.3<0;1,0>:ud    r2.2<0;1,0>:uw   {A@1}              //  ALU pipe: int; $3
        mov (16|M0)              r22.0<4>:uw   r1.0<1;1,0>:uw                                        //  ALU pipe: int; $15
(W)     macl (1|M0)              r7.0<1>:ud    r4.3<0;1,0>:ud    r2.1<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $4
(W)     mul (1|M0)               acc0.0<1>:ud  r4.3<0;1,0>:ud    r2.2<0;1,0>:uw                      //  ALU pipe: int; $4
        mov (16|M16)             r24.0<4>:uw   r1.16<1;1,0>:uw                                       //  ALU pipe: int; $15
(W)     mach (1|M0)              r6.0<1>:d     r4.3<0;1,0>:ud    r2.1<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mov (2|M0)               r3.0<1>:d     r4.6<1;1,0>:d                    {Compacted,$1.dst}   //  ALU pipe: int; $2
(W)     mov (1|M0)               r7.1<1>:d     r6.0<0;1,0>:d                    {Compacted,I@2}      //  ALU pipe: int; $9
        add (16|M0)              r26.0<1>:q    r7.0<0;1,0>:q     r22.0<4;1,0>:uw  {I@1}              //  ALU pipe: int; $15
        add (16|M16)             r28.0<1>:q    r7.0<0;1,0>:q     r24.0<4;1,0>:uw                     //  ALU pipe: int; $15
        add (16|M0)              r30.0<1>:q    r26.0<1;1,0>:q    r4.0<0;1,0>:ud   {I@2}              //  ALU pipe: int; $17
        add (16|M16)             r32.0<1>:q    r28.0<1;1,0>:q    r4.0<0;1,0>:ud   {I@2}              //  ALU pipe: int; $17
        mov (16|M0)              r8.0<1>:d     r30.0<2;1,0>:d                   {Compacted,I@2}      //  ALU pipe: int; $18
        mov (16|M16)             r9.0<1>:d     r32.0<2;1,0>:d                   {Compacted,I@2}      //  ALU pipe: int; $19
        mov (16|M0)              r10.0<1>:d    r30.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $20
        mov (16|M16)             r11.0<1>:d    r32.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $21
        cmp (32|M0)   (lt)f1.0   null<1>:ud    r8.0<1;1,0>:ud    r3.0<0;1,0>:ud   {I@3}              //  ALU pipe: int; $22
(f1.0)  cmp (32|M0)   (eq)f1.0   null<1>:d     r10.0<1;1,0>:d    r3.1<0;1,0>:d    {I@2}              //  ALU pipe: int; $23
(~f1.0) cmp (32|M0)   (lt)f1.0   null<1>:ud    r10.0<1;1,0>:ud   r3.1<0;1,0>:ud                      //  ALU pipe: int; $25
(~f1.0) goto (32|M0)                         _0_007            _0_007                                //  ALU pipe: int; $27
// B003: [inDivergent],  Preds:{B002},  Succs:{B004}
_0_008:
(W)     mov (1|M0)               r3.1<1>:q     r5.12<0;1,0>:ud                  {$2.dst}             //  ALU pipe: int; $29
// B004: [inDivergent],  Preds:{B005, B003},  Succs:{B005, B006}
_0_009:
        add (16|M0)              r38.0<1>:q    r30.0<1;1,0>:q    r3.1<0;1,0>:q    {Compacted,I@1}    //  ALU pipe: int; $36
        add (16|M16)             r40.0<1>:q    r32.0<1;1,0>:q    r3.1<0;1,0>:q    {Compacted}        //  ALU pipe: int; $36
        shl (16|M0)              r34.0<1>:q    r30.0<1;1,0>:q    2:w               {Compacted}       //  ALU pipe: int; $31
        mov (16|M0)              r18.0<1>:d    r38.0<2;1,0>:d                   {Compacted,I@3}      //  ALU pipe: int; $37
        mov (16|M16)             r19.0<1>:d    r40.0<2;1,0>:d                   {Compacted,I@3}      //  ALU pipe: int; $38
        mov (16|M0)              r20.0<1>:d    r38.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $39
        mov (16|M16)             r21.0<1>:d    r40.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $40
        cmp (32|M0)   (lt)f0.0   null<1>:ud    r18.0<1;1,0>:ud   r3.0<0;1,0>:ud   {I@3}              //  ALU pipe: int; $41
        shl (16|M16)             r36.0<1>:q    r32.0<1;1,0>:q    2:w               {Compacted}       //  ALU pipe: int; $31
        mov (32|M0)              r16.0<1>:d    r5.2<0;1,0>:d                    {Compacted,$3.src}   //  ALU pipe: int; $34
(f0.0)  cmp (32|M0)   (eq)f0.0   null<1>:d     r20.0<1;1,0>:d    r3.1<0;1,0>:d    {I@4}              //  ALU pipe: int; $42
        add (16|M0)              r12.0<1>:q    r34.0<1;1,0>:q    r5.0<0;1,0>:q    {Compacted}        //  ALU pipe: int; $32
        add (16|M16)             r14.0<1>:q    r36.0<1;1,0>:q    r5.0<0;1,0>:q    {Compacted,I@4}    //  ALU pipe: int; $32
(~f0.0) cmp (32|M0)   (lt)f0.0   null<1>:ud    r20.0<1;1,0>:ud   r3.1<0;1,0>:ud                      //  ALU pipe: int; $44
        store.ugm.d32.a64 (32|M0)  [r12:4]      r16:2              {I@2,$3} // ex_desc:0x0; desc:0x8000584 // $35
(~f0.0) goto (32|M0)                         _0_007            _0_007                                //  ALU pipe: int; $46
// B005: [inDivergent],  Preds:{B004},  Succs:{B004}
_0_010:
(f0.0)  sel (32|M0)              r42.0<1>:d    r18.0<1;1,0>:d    r8.0<1;1,0>:d                       //  ALU pipe: int; $48
(f0.0)  sel (32|M0)              r44.0<1>:d    r20.0<1;1,0>:d    r10.0<1;1,0>:d                      //  ALU pipe: int; $49
        mov (16|M0)              r30.0<2>:d    r42.0<1;1,0>:d                   {I@2}                //  ALU pipe: int; $52
        mov (16|M16)             r32.0<2>:d    r43.0<1;1,0>:d                                        //  ALU pipe: int; $53
        mov (16|M0)              r30.1<2>:d    r44.0<1;1,0>:d                   {I@3}                //  ALU pipe: int; $54
        mov (16|M16)             r32.1<2>:d    r45.0<1;1,0>:d                                        //  ALU pipe: int; $55
(W)     jmpi                                 _0_009                                                  // $56
// B006: Preds:{B004, B002},  Succs:{}
_0_007:
        join (32|M0)                         L848                                                    // 
L848:
(W)     mov (16|M0)              r255.0<1>:f   r2.0<1;1,0>:f                    {Compacted,$2.src}   //  ALU pipe: float; $58
(W)     send.gtwy (1|M0)         null     r255  null:0  0x0            0x02000010           {EOT,F@1,$4} // wr:1+0, rd:0; end of thread // $58
L872:
(W)     mov (16|M0)              null<1>:ud    0x23954D4A:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x795ECA46:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x5:ud                                                // 


//.BankConflicts: 0
//.ByteRMWs: 0
//


//.numALUInst: 57
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 7
//.allAtOneDistNum: 3
//.syncInstCount: 1
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 3
//.AfterReadTokenDepCount: 4

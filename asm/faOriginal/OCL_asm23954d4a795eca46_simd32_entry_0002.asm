//.kernel _ZTSZZN6compat6detailL6memcpyEN4sycl3_V15queueEPvPKvNS2_5rangeILi3EEES8_NS2_2idILi3EEESA_S8_RKSt6vectorINS2_5eventESaISC_EEENKUlRNS2_7handlerEE_clESI_E16memcpy_3d_detail
//.platform XE2
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.options_string "-emitCrossThreadOffR0Reloc -perfmodel -hashmovs 596987210 2036255302 -hashmovs1 0 2 "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -TotalGRFNum 256 -abortOnSpill 4 -enableBCR -forceBCR -enableBundleCR 3 -perfmodel -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -hashmovs 596987210 2036255302 -hashmovs1 0 2 "
//.instCount 128
//.RA type	LOCAL_ROUND_ROBIN_BC_RA
//.git-hash 

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r6.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r2.10)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r2.11)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r2.6)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r2.7)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r2.8)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r2.9)
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
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r6.0)
//.declare V0034 (44)  rf=r size=8 type=uq align=4 words (r4.3)
//.declare V0035 (45)  rf=r size=8 type=q align=4 words (r5.1)
//.declare V0036 (46)  rf=r size=8 type=uq align=4 words (r5.0)
//.declare V0037 (47)  rf=r size=8 type=q align=4 words (r5.2)
//.declare V0039 (49)  rf=r size=32 type=d alias=+0 align=32 words (r6.0)
//.declare V0040 (50)  rf=r size=12 type=d align=2 words (r4.0)
//.declare V0041 (51)  rf=r size=12 type=d align=2 words (r4.3)
//.declare V0042 (52)  rf=r size=64 type=w align=32 words (r1.0)
//.declare V0043 (53)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0044 (54)  rf=r size=64 type=w align=32 words (r3.0)
//.declare V0045 (55)  rf=r size=8 type=uq align=4 words (r6.1)
//.declare V0046 (56)  rf=r size=8 type=uq align=4 words (r6.2)
//.declare V0047 (57)  rf=r size=8 type=q align=4 words (r5.3)
//.declare V0048 (58)  rf=r size=8 type=q align=4 words (r5.4)
//.declare V0049 (59)  rf=r size=8 type=q align=4 words (r5.5)
//.declare V0050 (60)  rf=r size=8 type=q align=4 words (r5.6)
//.declare V0051 (61)  rf=r size=8 type=q align=4 words (r5.7)
//.declare V0052 (62)  rf=r size=8 type=q align=4 words (r6.0)
//.declare V0058 (68)  rf=r size=8 type=d align=2 words (r4.8)
//.declare V0059 (69)  rf=r size=8 type=d alias=V0035+0 align=4 words (r5.2)
//.declare V0060 (70)  rf=r size=8 type=d align=2 words (r4.10)
//.declare V0061 (71)  rf=r size=8 type=d alias=V0037+0 align=4 words (r5.4)
//.declare V0062 (72)  rf=r size=8 type=d align=2 words (r4.12)
//.declare V0063 (73)  rf=r size=8 type=d alias=V0047+0 align=4 words (r5.6)
//.declare V0064 (74)  rf=r size=8 type=d align=2 words (r4.14)
//.declare V0065 (75)  rf=r size=8 type=d alias=V0050+0 align=4 words (r5.12)
//.declare V0066 (76)  rf=r size=4 type=d align=32 words (r7.0)
//.declare V0068 (78)  rf=r size=64 type=uw alias=V0044+0 align=32 words (r3.0)
//.declare V0069 (79)  rf=r size=128 type=d align=32 words (r8.0)
//.declare V0070 (80)  rf=r size=4 type=d align=32 words (r10.0)
//.declare V0072 (82)  rf=r size=64 type=uw alias=V0043+0 align=32 words (r2.0)
//.declare V0073 (83)  rf=r size=128 type=d align=32 words (r12.0)
//.declare V0074 (84)  rf=r size=4 type=d align=32 words (r11.0)
//.declare V0076 (86)  rf=r size=64 type=uw alias=V0042+0 align=32 words (r1.0)
//.declare V0077 (87)  rf=r size=128 type=d align=32 words (r14.0)
//.declare V0081 (91)  rf=r size=128 type=ud alias=V0077+0 align=32 words (r14.0)
//.declare V0082 (92)  rf=r size=8 type=ud alias=V0060+0 align=2 words (r4.10)
//.declare V0083 (93)  rf=r size=128 type=d align=32 words (r17.0)
//.declare V0085 (95)  rf=r size=128 type=d align=32 words (r20.0)
//.declare V0092 (102)  rf=r size=128 type=ud alias=V0073+0 align=32 words (r12.0)
//.declare V0093 (103)  rf=r size=8 type=ud alias=V0064+0 align=2 words (r4.14)
//.declare V0094 (104)  rf=r size=128 type=d align=32 words (r23.0)
//.declare V0096 (106)  rf=r size=128 type=d align=32 words (r26.0)
//.declare V0100 (110)  rf=r size=8 type=q alias=V0036+0 align=4 words (r5.0)
//.declare V0104 (114)  rf=r size=128 type=ud alias=V0069+0 align=32 words (r8.0)
//.declare V0105 (115)  rf=r size=256 type=q align=32 words (r28.0)
//.declare V0106 (116)  rf=r size=256 type=uq alias=V0105+0 align=32 words (r28.0)
//.declare V0107 (117)  rf=r size=32 type=b align=32 words (r16.0)
//.declare V0108 (118)  rf=r size=128 type=ud align=32 words (r32.0)
//.declare V0109 (119)  rf=r size=128 type=b alias=V0108+0 align=32 words (r32.0)
//.declare V0113 (123)  rf=r size=8 type=ud alias=V0058+0 align=2 words (r4.8)
//.declare V0114 (124)  rf=r size=128 type=d align=32 words (r35.0)
//.declare V0116 (126)  rf=r size=128 type=d align=32 words (r38.0)
//.declare V0123 (133)  rf=r size=8 type=ud alias=V0062+0 align=2 words (r4.12)
//.declare V0124 (134)  rf=r size=128 type=d align=32 words (r41.0)
//.declare V0126 (136)  rf=r size=128 type=d align=32 words (r44.0)
//.declare V0130 (140)  rf=r size=8 type=q alias=V0034+0 align=4 words (r4.3)
//.declare V0133 (143)  rf=r size=256 type=q align=32 words (r46.0)
//.declare V0134 (144)  rf=r size=256 type=uq alias=V0133+0 align=32 words (r46.0)
//.declare V0135 (145)  rf=r size=32 type=ub alias=V0107+0 align=32 words (r16.0)
//.declare V0136 (146)  rf=r size=128 type=ud align=32 words (r50.0)
//.declare  (147)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare  (148)  rf=r size=128 type=ud align=32 words (r52.0)
//.declare  (149)  rf=r size=128 type=ud align=32 words (r54.0)
//.declare  (152)  rf=r size=128 type=d align=32 words (r56.0)
//.declare  (153)  rf=r size=128 type=d align=32 words (r58.0)
//.declare  (154)  rf=r size=128 type=q align=32 words (r61.0)
//.declare  (155)  rf=r size=128 type=q align=32 words (r63.0)
//.declare  (156)  rf=r size=128 type=d align=32 words (r65.0)
//.declare  (157)  rf=r size=128 type=d align=32 words (r67.0)
//.declare  (158)  rf=r size=128 type=q align=32 words (r70.0)
//.declare  (159)  rf=r size=128 type=q align=32 words (r72.0)
//.declare  (160)  rf=r size=128 type=q align=32 words (r75.0)
//.declare  (161)  rf=r size=128 type=q align=32 words (r77.0)
//.declare  (162)  rf=r size=128 type=q align=32 words (r79.0)
//.declare  (163)  rf=r size=128 type=q align=32 words (r81.0)
//.declare  (166)  rf=r size=128 type=d align=32 words (r83.0)
//.declare  (167)  rf=r size=128 type=d align=32 words (r85.0)
//.declare  (168)  rf=r size=128 type=q align=32 words (r87.0)
//.declare  (169)  rf=r size=128 type=q align=32 words (r89.0)
//.declare  (170)  rf=r size=128 type=d align=32 words (r91.0)
//.declare  (171)  rf=r size=128 type=d align=32 words (r93.0)
//.declare  (172)  rf=r size=128 type=q align=32 words (r96.0)
//.declare  (173)  rf=r size=128 type=q align=32 words (r98.0)
//.declare  (174)  rf=r size=128 type=q align=32 words (r101.0)
//.declare  (175)  rf=r size=128 type=q align=32 words (r103.0)
//.declare  (176)  rf=r size=128 type=q align=32 words (r105.0)
//.declare  (177)  rf=r size=128 type=q align=32 words (r107.0)
//.declare  (178)  rf=r size=128 type=ud alias=+0 align=32 words (r56.0)
//.declare  (179)  rf=r size=128 type=d alias=+0 align=32 words (r61.0)
//.declare  (180)  rf=r size=128 type=d alias=+0 align=32 words (r63.0)
//.declare  (181)  rf=r size=128 type=ud alias=+0 align=32 words (r65.0)
//.declare  (182)  rf=r size=128 type=d alias=+0 align=32 words (r70.0)
//.declare  (183)  rf=r size=128 type=d alias=+0 align=32 words (r72.0)
//.declare  (184)  rf=r size=128 type=ud alias=+0 align=32 words (r83.0)
//.declare  (185)  rf=r size=128 type=d alias=+0 align=32 words (r87.0)
//.declare  (186)  rf=r size=128 type=d alias=+0 align=32 words (r89.0)
//.declare  (187)  rf=r size=128 type=ud alias=+0 align=32 words (r91.0)
//.declare  (188)  rf=r size=128 type=d alias=+0 align=32 words (r96.0)
//.declare  (189)  rf=r size=128 type=d alias=+0 align=32 words (r98.0)
//.declare r0 (190)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (191)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare inlineRegFromTDL (192)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (193)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (194)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (195)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (196)  rf=r size=64 type=ud align=32 words (r5.0)
//.declare  (197)  rf=r size=32 type=ud align=2 words (r6.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// | V0042    | :w x 32  |   0x40 | r1       | pti[tid]+0x0     |
// | V0043    | :w x 32  |   0x40 | r2       | pti[tid]+0x40    |
// | V0044    | :w x 32  |   0x40 | r3       | pti[tid]+0x80    |
// | V0040    | :d x 3   |    0xC | r4       | inline+0x0       |
// | V0041    | :d x 3   |    0xC | r4+0xC   | inline+0xC       |
// | V0034    | :uq      |    0x8 | r4+0x18  | inline+0x18      |
// | V0036    | :uq      |    0x8 | r5       | cti+0x20         |
// | V0035    | :q       |    0x8 | r5+0x8   | cti+0x28         |
// | V0037    | :q       |    0x8 | r5+0x10  | cti+0x30         |
// | V0047    | :q       |    0x8 | r5+0x18  | cti+0x38         |
// | V0048    | :q       |    0x8 | r5+0x20  | cti+0x40         |
// | V0049    | :q       |    0x8 | r5+0x28  | cti+0x48         |
// | V0050    | :q       |    0x8 | r5+0x30  | cti+0x50         |
// | V0051    | :q       |    0x8 | r5+0x38  | cti+0x58         |
// | V0052    | :q       |    0x8 | r6       | cti+0x60         |
// | V0045    | :uq      |    0x8 | r6+0x8   | cti+0x68         |
// | V0046    | :uq      |    0x8 | r6+0x10  | cti+0x70         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r255.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r255.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x60:ud              {I@2}          //  ALU pipe: int; 
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
(W)     load.ugm.d32x8t.a32.ca.cc (1|M0)  r6:1  bti[255][r255:1+0x40]  {$3} // ex_desc:0xFF040000; desc:0x6219C500 // 
// B002: Preds:{B001},  Succs:{}
// _main:
(W)     mov (16|M0)              r6.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted,$3.dst}   //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mul (1|M0)               acc0.0<1>:d   r4.5<0;1,0>:d     r6.14<0;1,0>:uw  {A@1}              //  ALU pipe: int; $6
(W)     mov (2|M0)               r4.10<1>:d    r5.4<1;1,0>:d                    {$2.dst}             //  ALU pipe: int; $3
(W)     mov (2|M0)               r4.14<1>:d    r5.12<1;1,0>:d                                        //  ALU pipe: int; $5
(W)     macl (1|M0)              r7.0<1>:d     r4.5<0;1,0>:d     r6.7<0;1,0>:d                       //  ALU pipe: int; $8
(W)     mul (1|M0)               acc0.0<1>:d   r4.4<0;1,0>:d     r6.12<0;1,0>:uw                     //  ALU pipe: int; $9
(W)     mov (2|M0)               r4.8<1>:d     r5.2<1;1,0>:d                                         //  ALU pipe: int; $2
(W)     mov (2|M0)               r4.12<1>:d    r5.6<1;1,0>:d                                         //  ALU pipe: int; $4
(W)     macl (1|M0)              r10.0<1>:d    r4.4<0;1,0>:d     r6.6<0;1,0>:d    {Compacted}        //  ALU pipe: int; $11
(W)     mul (1|M0)               acc0.0<1>:d   r4.3<0;1,0>:d     r6.2<0;1,0>:uw                      //  ALU pipe: int; $12
        add3 (32|M0)             r8.0<1>:d     r7.0<0;0>:d       r3.0<1;0>:uw      r4.2<0>:d        {@6,$1.dst} //  ALU pipe: int; $8
(W)     macl (1|M0)              r11.0<1>:d    r4.3<0;1,0>:d     r6.1<0;1,0>:d    {Compacted}        //  ALU pipe: int; $14
        add3 (32|M0)             r12.0<1>:d    r10.0<0;0>:d      r2.0<1;0>:uw      r4.1<0>:d        {@4,$0.dst} //  ALU pipe: int; $11 R{} IR{}{E:5,E:1,E:2,},  R{r10,r2,r4,} IR{} {BC=1}
        add3 (32|M0)             r14.0<1>:d    r11.0<0;0>:d      r1.0<1;0>:uw      r4.0<0>:d        {I@2} //  ALU pipe: int; $14
        mov (16|M0)              r52.0<2>:ud   r8.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $44
(W)     mul (16|M0)              acc0.0<1>:ud  r14.0<1;1,0>:ud   r4.20<0;1,0>:uw  {I@2}              //  ALU pipe: int; $15
        mov (16|M16)             r54.0<2>:ud   r9.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $44
        macl (16|M0)             r56.0<1>:ud   r14.0<1;1,0>:ud   r4.10<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $15
(W)     mul (16|M16)             acc0.0<1>:ud  r15.0<1;1,0>:ud   r4.20<0;1,0>:uw                     //  ALU pipe: int; $15
(W)     mov (16|M0)              r255.0<1>:f   r6.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $78
        macl (16|M16)            r57.0<1>:ud   r15.0<1;1,0>:ud   r4.10<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $16
(W)     mul (16|M0)              acc0.0<1>:ud  r14.0<1;1,0>:ud   r4.20<0;1,0>:uw                     //  ALU pipe: int; $16
        mov (16|M0)              r61.0<2>:d    r56.0<1;1,0>:d                   {I@4}                //  ALU pipe: int; $24
        mach (16|M0)             r17.0<1>:d    r14.0<1;1,0>:ud   r4.10<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r15.0<1;1,0>:ud   r4.20<0;1,0>:uw                     //  ALU pipe: int; $16
        mov (16|M16)             r63.0<2>:d    r57.0<1;1,0>:d                   {I@5}                //  ALU pipe: int; $25
        mach (16|M16)            r18.0<1>:d    r15.0<1;1,0>:ud   r4.10<0;1,0>:ud                     //  ALU pipe: int; $17
(W)     mul (16|M0)              acc0.0<1>:d   r14.0<1;1,0>:ud   r4.22<0;1,0>:uw                     //  ALU pipe: int; $17
        macl (16|M0)             r20.0<1>:d    r14.0<1;1,0>:ud   r4.11<0;1,0>:d                      //  ALU pipe: int; $17
(W)     mul (16|M16)             acc0.0<1>:d   r15.0<1;1,0>:ud   r4.22<0;1,0>:uw                     //  ALU pipe: int; $17
        macl (16|M16)            r21.0<1>:d    r15.0<1;1,0>:ud   r4.11<0;1,0>:d                      //  ALU pipe: int; $18
(W)     mul (16|M0)              acc0.0<1>:ud  r12.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $28
        macl (16|M0)             r65.0<1>:ud   r12.0<1;1,0>:ud   r4.14<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $28
(W)     mul (16|M16)             acc0.0<1>:ud  r13.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $28
        add (32|M0)              r17.0<1>:d    r17.0<1;1,0>:d    r20.0<1;1,0>:d   {Compacted,I@4}    //  ALU pipe: int; $18
        macl (16|M16)            r66.0<1>:ud   r13.0<1;1,0>:ud   r4.14<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $29
(W)     mul (16|M0)              acc0.0<1>:ud  r12.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $29
        mov (32|M0)              r58.0<1>:f    r17.0<1;1,0>:f                   {Compacted,I@3}      //  ALU pipe: float; $21
        mach (16|M0)             r23.0<1>:d    r12.0<1;1,0>:ud   r4.14<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r13.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $29
        mov (16|M0)              r61.1<2>:d    r58.0<1;1,0>:d                   {F@1}                //  ALU pipe: int; $26
        mach (16|M16)            r24.0<1>:d    r13.0<1;1,0>:ud   r4.14<0;1,0>:ud                     //  ALU pipe: int; $30
(W)     mul (16|M0)              acc0.0<1>:d   r12.0<1;1,0>:ud   r4.30<0;1,0>:uw                     //  ALU pipe: int; $30
        mov (16|M16)             r63.1<2>:d    r59.0<1;1,0>:d                                        //  ALU pipe: int; $27
        macl (16|M0)             r26.0<1>:d    r12.0<1;1,0>:ud   r4.15<0;1,0>:d                      //  ALU pipe: int; $30
(W)     mul (16|M16)             acc0.0<1>:d   r13.0<1;1,0>:ud   r4.30<0;1,0>:uw                     //  ALU pipe: int; $30
        mov (16|M0)              r70.0<2>:d    r65.0<1;1,0>:d                                        //  ALU pipe: int; $37
        macl (16|M16)            r27.0<1>:d    r13.0<1;1,0>:ud   r4.15<0;1,0>:d                      //  ALU pipe: int; $31
        mov (16|M16)             r72.0<2>:d    r66.0<1;1,0>:d                                        //  ALU pipe: int; $38
        add (16|M0)              r75.0<1>:q    r61.0<1;1,0>:q    r5.0<0;1,0>:q    {Compacted,I@7}    //  ALU pipe: int; $41
        add (32|M0)              r23.0<1>:d    r23.0<1;1,0>:d    r26.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $31
        add (16|M16)             r77.0<1>:q    r63.0<1;1,0>:q    r5.0<0;1,0>:q    {Compacted}        //  ALU pipe: int; $41
        mov (32|M0)              r67.0<1>:f    r23.0<1;1,0>:f                   {Compacted,I@2}      //  ALU pipe: float; $34
(W)     mul (16|M0)              acc0.0<1>:ud  r14.0<1;1,0>:ud   r4.16<0;1,0>:uw                     //  ALU pipe: int; $47
        mov (16|M0)              r70.1<2>:d    r67.0<1;1,0>:d                   {F@1}                //  ALU pipe: int; $39
        mov (16|M16)             r72.1<2>:d    r68.0<1;1,0>:d                                        //  ALU pipe: int; $40
        macl (16|M0)             r83.0<1>:ud   r14.0<1;1,0>:ud   r4.8<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $47
        add (16|M0)              r79.0<1>:q    r75.0<1;1,0>:q    r70.0<1;1,0>:q   {Compacted,I@3}    //  ALU pipe: int; $42
        add (16|M16)             r81.0<1>:q    r77.0<1;1,0>:q    r72.0<1;1,0>:q   {Compacted,I@3}    //  ALU pipe: int; $42
(W)     mul (16|M16)             acc0.0<1>:ud  r15.0<1;1,0>:ud   r4.16<0;1,0>:uw                     //  ALU pipe: int; $47
        add (16|M0)              r28.0<1>:q    r79.0<1;1,0>:q    r52.0<2;1,0>:ud  {I@3}              //  ALU pipe: int; $44
        add (16|M16)             r30.0<1>:q    r81.0<1;1,0>:q    r54.0<2;1,0>:ud  {I@3}              //  ALU pipe: int; $44
        macl (16|M16)            r84.0<1>:ud   r15.0<1;1,0>:ud   r4.8<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $48
(W)     mul (16|M0)              acc0.0<1>:ud  r14.0<1;1,0>:ud   r4.16<0;1,0>:uw                     //  ALU pipe: int; $48
        load.ugm.d8u32.a64 (32|M0)  r32:2       [r28:4]            {I@3,$4} // ex_desc:0x0; desc:0x8200980 // $45
        mach (16|M0)             r35.0<1>:d    r14.0<1;1,0>:ud   r4.8<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r15.0<1;1,0>:ud   r4.16<0;1,0>:uw                     //  ALU pipe: int; $48
        mov (16|M0)              r87.0<2>:d    r83.0<1;1,0>:d                                        //  ALU pipe: int; $56
        mach (16|M16)            r36.0<1>:d    r15.0<1;1,0>:ud   r4.8<0;1,0>:ud                      //  ALU pipe: int; $49
(W)     mul (16|M0)              acc0.0<1>:d   r14.0<1;1,0>:ud   r4.18<0;1,0>:uw                     //  ALU pipe: int; $49
        mov (16|M16)             r89.0<2>:d    r84.0<1;1,0>:d                   {I@7}                //  ALU pipe: int; $57
        macl (16|M0)             r38.0<1>:d    r14.0<1;1,0>:ud   r4.9<0;1,0>:d                       //  ALU pipe: int; $49
(W)     mul (16|M16)             acc0.0<1>:d   r15.0<1;1,0>:ud   r4.18<0;1,0>:uw                     //  ALU pipe: int; $49
        macl (16|M16)            r39.0<1>:d    r15.0<1;1,0>:ud   r4.9<0;1,0>:d                       //  ALU pipe: int; $50
(W)     mul (16|M0)              acc0.0<1>:ud  r12.0<1;1,0>:ud   r4.24<0;1,0>:uw                     //  ALU pipe: int; $60
        macl (16|M0)             r91.0<1>:ud   r12.0<1;1,0>:ud   r4.12<0;1,0>:ud                     //  ALU pipe: int; $60
(W)     mul (16|M16)             acc0.0<1>:ud  r13.0<1;1,0>:ud   r4.24<0;1,0>:uw                     //  ALU pipe: int; $60
        add (32|M0)              r35.0<1>:d    r35.0<1;1,0>:d    r38.0<1;1,0>:d   {Compacted,I@4}    //  ALU pipe: int; $50
        macl (16|M16)            r92.0<1>:ud   r13.0<1;1,0>:ud   r4.12<0;1,0>:ud                     //  ALU pipe: int; $61
(W)     mul (16|M0)              acc0.0<1>:ud  r12.0<1;1,0>:ud   r4.24<0;1,0>:uw                     //  ALU pipe: int; $61
        mov (32|M0)              r85.0<1>:f    r35.0<1;1,0>:f                   {Compacted,I@3}      //  ALU pipe: float; $53
        mach (16|M0)             r41.0<1>:d    r12.0<1;1,0>:ud   r4.12<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r13.0<1;1,0>:ud   r4.24<0;1,0>:uw                     //  ALU pipe: int; $61
        mov (16|M0)              r87.1<2>:d    r85.0<1;1,0>:d                   {F@1}                //  ALU pipe: int; $58
        mach (16|M16)            r42.0<1>:d    r13.0<1;1,0>:ud   r4.12<0;1,0>:ud                     //  ALU pipe: int; $62
(W)     mul (16|M0)              acc0.0<1>:d   r12.0<1;1,0>:ud   r4.26<0;1,0>:uw                     //  ALU pipe: int; $62
        mov (16|M16)             r89.1<2>:d    r86.0<1;1,0>:d                                        //  ALU pipe: int; $59
        macl (16|M0)             r44.0<1>:d    r12.0<1;1,0>:ud   r4.13<0;1,0>:d                      //  ALU pipe: int; $62
(W)     mul (16|M16)             acc0.0<1>:d   r13.0<1;1,0>:ud   r4.26<0;1,0>:uw                     //  ALU pipe: int; $62
        mov (16|M0)              r96.0<2>:d    r91.0<1;1,0>:d                                        //  ALU pipe: int; $69
        macl (16|M16)            r45.0<1>:d    r13.0<1;1,0>:ud   r4.13<0;1,0>:d                      //  ALU pipe: int; $63
        mov (16|M16)             r98.0<2>:d    r92.0<1;1,0>:d                                        //  ALU pipe: int; $70
        add (16|M0)              r101.0<1>:q   r87.0<1;1,0>:q    r4.3<0;1,0>:q    {Compacted,I@7}    //  ALU pipe: int; $73
        add (32|M0)              r41.0<1>:d    r41.0<1;1,0>:d    r44.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $63
        add (16|M16)             r103.0<1>:q   r89.0<1;1,0>:q    r4.3<0;1,0>:q    {Compacted}        //  ALU pipe: int; $73
        mov (32|M0)              r93.0<1>:f    r41.0<1;1,0>:f                   {Compacted,I@2}      //  ALU pipe: float; $66
        mov (16|M0)              r96.1<2>:d    r93.0<1;1,0>:d                   {F@1}                //  ALU pipe: int; $71
        mov (16|M16)             r98.1<2>:d    r94.0<1;1,0>:d                                        //  ALU pipe: int; $72
        add (16|M0)              r105.0<1>:q   r101.0<1;1,0>:q   r96.0<1;1,0>:q   {Compacted,I@2}    //  ALU pipe: int; $74
        add (16|M16)             r107.0<1>:q   r103.0<1;1,0>:q   r98.0<1;1,0>:q   {Compacted,I@2}    //  ALU pipe: int; $74
        add (16|M0)              r46.0<1>:q    r105.0<1;1,0>:q   r52.0<2;1,0>:ud  {I@2}              //  ALU pipe: int; $75
        add (16|M16)             r48.0<1>:q    r107.0<1;1,0>:q   r54.0<2;1,0>:ud  {I@2}              //  ALU pipe: int; $75
        mov (32|M0)              r16.0<1>:b    r32.0<4;1,0>:b                   {$4.dst}             //  ALU pipe: int; $46
        mov (32|M0)              r50.0<1>:ud   r16.0<1;1,0>:ub                  {I@1}                //  ALU pipe: int; $76
        store.ugm.d8u32.a64 (32|M0)  [r46:4]    r50:2              {I@1,$5} // ex_desc:0x0; desc:0x8000984 // $77
(W)     send.gtwy (1|M0)         null     r255  null:0  0x0            0x02000010           {EOT,$6} // wr:1+0, rd:0; end of thread // $78
L1752:
(W)     mov (16|M0)              null<1>:ud    0x23954D4A:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x795ECA46:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x2:ud                                                // 


//.BankConflicts: 1
//.ByteRMWs: 0
//


//.numALUInst: 118
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 10
//.allAtOneDistNum: 3
//.syncInstCount: 1
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 5
//.AfterReadTokenDepCount: 2

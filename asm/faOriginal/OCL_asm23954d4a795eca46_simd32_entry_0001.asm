//.kernel _ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZN6compat6detailL6memcpyENS0_5queueEPvPKvNS0_5rangeILi3EEESA_NS0_2idILi3EEESC_SA_RKSt6vectorINS0_5eventESaISE_EEENKUlRNS0_7handlerEE_clESK_E16memcpy_3d_detailEE
//.platform XE2
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.options_string "-emitCrossThreadOffR0Reloc -perfmodel -hashmovs 596987210 2036255302 -hashmovs1 0 1 "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -TotalGRFNum 256 -abortOnSpill 4 -enableBCR -enableBundleCR 3 -perfmodel -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -hashmovs 596987210 2036255302 -hashmovs1 0 1 "
//.instCount 269
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r7.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r3.3)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r3.4)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r2.15)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r3.0)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r3.1)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r3.2)
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
//.declare V0033 (43)  rf=r size=64 type=d alias=+0 align=32 words (r7.0)
//.declare V0035 (45)  rf=r size=32 type=d alias=+0 align=32 words (r7.0)
//.declare V0036 (46)  rf=r size=12 type=d align=2 words (r4.0)
//.declare V0037 (47)  rf=r size=12 type=d align=2 words (r6.12)
//.declare V0038 (48)  rf=r size=12 type=d align=2 words (r4.3)
//.declare V0039 (49)  rf=r size=64 type=w align=32 words (r1.0)
//.declare V0040 (50)  rf=r size=64 type=w align=32 words (r2.0)
//.declare V0041 (51)  rf=r size=64 type=w align=32 words (r3.0)
//.declare V0042 (52)  rf=r size=8 type=uq align=4 words (r6.4)
//.declare V0043 (53)  rf=r size=8 type=uq align=4 words (r6.5)
//.declare V0044 (54)  rf=r size=8 type=q align=4 words (r4.3)
//.declare V0045 (55)  rf=r size=8 type=q align=4 words (r5.0)
//.declare V0046 (56)  rf=r size=8 type=q align=4 words (r5.1)
//.declare V0047 (57)  rf=r size=8 type=q align=4 words (r5.2)
//.declare V0048 (58)  rf=r size=8 type=q align=4 words (r5.3)
//.declare V0049 (59)  rf=r size=8 type=q align=4 words (r5.4)
//.declare V0050 (60)  rf=r size=8 type=q align=4 words (r5.5)
//.declare V0051 (61)  rf=r size=8 type=q align=4 words (r5.6)
//.declare V0052 (62)  rf=r size=8 type=q align=4 words (r5.7)
//.declare V0053 (63)  rf=r size=8 type=q align=4 words (r6.0)
//.declare V0054 (64)  rf=r size=8 type=q align=4 words (r6.1)
//.declare V0055 (65)  rf=r size=8 type=q align=4 words (r6.2)
//.declare V0056 (66)  rf=r size=8 type=q align=4 words (r6.3)
//.declare V0058 (68)  rf=r size=8 type=d align=2 words (r4.8)
//.declare V0059 (69)  rf=r size=8 type=d alias=V0044+0 align=4 words (r4.6)
//.declare V0060 (70)  rf=r size=8 type=d align=2 words (r4.10)
//.declare V0061 (71)  rf=r size=8 type=d alias=V0045+0 align=4 words (r5.0)
//.declare V0062 (72)  rf=r size=8 type=d align=2 words (r4.12)
//.declare V0063 (73)  rf=r size=8 type=d alias=V0046+0 align=4 words (r5.2)
//.declare V0067 (77)  rf=r size=12 type=ud alias=V0038+0 align=2 words (r4.3)
//.declare V0068 (78)  rf=r size=32 type=ud alias=V0035+0 align=16 words (r7.0)
//.declare V0069 (79)  rf=r size=4 type=d align=32 words (r8.0)
//.declare V0073 (83)  rf=r size=8 type=q align=32 words (r9.0)
//.declare V0074 (84)  rf=r size=8 type=d alias=V0073+0 align=4 words (r9.0)
//.declare V0076 (86)  rf=r size=64 type=uw alias=V0041+0 align=32 words (r3.0)
//.declare V0079 (89)  rf=r size=12 type=ud alias=V0036+0 align=2 words (r4.0)
//.declare V0083 (93)  rf=r size=128 type=d align=32 words (r10.0)
//.declare V0084 (94)  rf=r size=128 type=d align=32 words (r12.0)
//.declare V0088 (98)  rf=r size=4 type=d align=32 words (r14.0)
//.declare V0092 (102)  rf=r size=8 type=q align=32 words (r15.0)
//.declare V0093 (103)  rf=r size=8 type=d alias=V0092+0 align=4 words (r15.0)
//.declare V0095 (105)  rf=r size=64 type=uw alias=V0040+0 align=32 words (r2.0)
//.declare V0101 (111)  rf=r size=128 type=d align=32 words (r16.0)
//.declare V0102 (112)  rf=r size=128 type=d align=32 words (r18.0)
//.declare V0106 (116)  rf=r size=4 type=d align=32 words (r20.0)
//.declare V0110 (120)  rf=r size=8 type=q align=32 words (r21.0)
//.declare V0111 (121)  rf=r size=8 type=d alias=V0110+0 align=4 words (r21.0)
//.declare V0113 (123)  rf=r size=64 type=uw alias=V0039+0 align=32 words (r1.0)
//.declare V0119 (129)  rf=r size=128 type=d align=32 words (r22.0)
//.declare V0120 (130)  rf=r size=128 type=d align=32 words (r24.0)
//.declare P01 (131)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0121 (132)  rf=r size=128 type=ud alias=V0083+0 align=32 words (r10.0)
//.declare V0122 (133)  rf=r size=8 type=ud alias=V0058+0 align=2 words (r4.8)
//.declare P02 (134)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare P03 (135)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0123 (136)  rf=r size=128 type=ud alias=V0084+0 align=32 words (r12.0)
//.declare P04 (137)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0124 (138)  rf=r size=128 type=ud alias=V0101+0 align=32 words (r16.0)
//.declare V0125 (139)  rf=r size=8 type=ud alias=V0060+0 align=2 words (r4.10)
//.declare P05 (140)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare P06 (141)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare V0126 (142)  rf=r size=128 type=ud alias=V0102+0 align=32 words (r18.0)
//.declare P07 (143)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0127 (144)  rf=r size=128 type=ud alias=V0119+0 align=32 words (r22.0)
//.declare V0128 (145)  rf=r size=8 type=ud alias=V0062+0 align=2 words (r4.12)
//.declare P08 (146)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare P09 (147)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0129 (148)  rf=r size=128 type=ud alias=V0120+0 align=32 words (r24.0)
//.declare P10 (149)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare P11 (150)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare V0130 (151)  rf=r size=8 type=d align=2 words (r4.14)
//.declare V0131 (152)  rf=r size=8 type=d alias=V0048+0 align=4 words (r5.6)
//.declare V0132 (153)  rf=r size=8 type=d align=2 words (r5.10)
//.declare V0133 (154)  rf=r size=8 type=d alias=V0049+0 align=4 words (r5.8)
//.declare V0134 (155)  rf=r size=8 type=d align=2 words (r5.12)
//.declare V0135 (156)  rf=r size=8 type=d alias=V0053+0 align=4 words (r6.0)
//.declare V0136 (157)  rf=r size=8 type=d align=2 words (r6.4)
//.declare V0137 (158)  rf=r size=8 type=d alias=V0054+0 align=4 words (r6.2)
//.declare V0138 (159)  rf=r size=8 type=q alias=+0 align=4 words (r6.4)
//.declare V0139 (160)  rf=r size=12 type=ud alias=V0037+0 align=2 words (r6.12)
//.declare V0140 (161)  rf=r size=8 type=q alias=+8 align=4 words (r6.5)
//.declare V0141 (162)  rf=r size=8 type=q align=4 words (r6.3)
//.declare V0146 (167)  rf=r size=128 type=d align=32 words (r26.0)
//.declare V0147 (168)  rf=r size=128 type=d align=32 words (r28.0)
//.declare V0149 (170)  rf=r size=128 type=d align=32 words (r30.0)
//.declare V0150 (171)  rf=r size=128 type=d align=32 words (r32.0)
//.declare V0154 (175)  rf=r size=128 type=ud alias=V0149+0 align=32 words (r30.0)
//.declare V0155 (176)  rf=r size=8 type=ud alias=V0134+0 align=2 words (r5.12)
//.declare V0156 (177)  rf=r size=128 type=d align=32 words (r34.0)
//.declare V0158 (179)  rf=r size=128 type=d align=32 words (r36.0)
//.declare V0165 (186)  rf=r size=128 type=ud alias=V0146+0 align=32 words (r26.0)
//.declare V0166 (187)  rf=r size=8 type=ud alias=V0136+0 align=2 words (r6.4)
//.declare V0167 (188)  rf=r size=128 type=d align=32 words (r38.0)
//.declare V0169 (190)  rf=r size=128 type=d align=32 words (r40.0)
//.declare V0177 (198)  rf=r size=256 type=uq align=32 words (r42.0)
//.declare V0178 (199)  rf=r size=32 type=b align=32 words (r46.0)
//.declare V0179 (200)  rf=r size=128 type=ud align=32 words (r48.0)
//.declare V0180 (201)  rf=r size=128 type=b alias=V0179+0 align=32 words (r48.0)
//.declare V0184 (205)  rf=r size=8 type=ud alias=V0130+0 align=2 words (r4.14)
//.declare V0185 (206)  rf=r size=128 type=d align=32 words (r50.0)
//.declare V0187 (208)  rf=r size=128 type=d align=32 words (r52.0)
//.declare V0194 (215)  rf=r size=8 type=ud alias=V0132+0 align=2 words (r5.10)
//.declare V0195 (216)  rf=r size=128 type=d align=32 words (r54.0)
//.declare V0197 (218)  rf=r size=128 type=d align=32 words (r56.0)
//.declare V0205 (226)  rf=r size=256 type=uq align=32 words (r58.0)
//.declare V0206 (227)  rf=r size=32 type=ub alias=V0178+0 align=32 words (r46.0)
//.declare V0207 (228)  rf=r size=128 type=ud align=32 words (r62.0)
//.declare V0211 (232)  rf=r size=128 type=d align=32 words (r64.0)
//.declare V0212 (233)  rf=r size=128 type=d align=32 words (r66.0)
//.declare P12 (234)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0213 (235)  rf=r size=128 type=ud alias=V0211+0 align=32 words (r64.0)
//.declare P13 (236)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare P14 (237)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare V0214 (238)  rf=r size=128 type=ud alias=V0212+0 align=32 words (r66.0)
//.declare V0219 (243)  rf=r size=128 type=d align=32 words (r68.0)
//.declare V0220 (244)  rf=r size=128 type=d align=32 words (r70.0)
//.declare P15 (245)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0221 (246)  rf=r size=128 type=ud alias=V0219+0 align=32 words (r68.0)
//.declare P16 (247)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare P17 (248)  rf=f32  size=4 type=uw align=2 words (f3.0)
//.declare V0222 (249)  rf=r size=128 type=ud alias=V0220+0 align=32 words (r70.0)
//.declare V0226 (253)  rf=r size=128 type=d align=32 words (r72.0)
//.declare V0227 (254)  rf=r size=128 type=d align=32 words (r74.0)
//.declare P18 (255)  rf=f32  size=4 type=uw align=2 words (f1.0)
//.declare V0228 (256)  rf=r size=128 type=ud alias=V0226+0 align=32 words (r72.0)
//.declare P19 (257)  rf=f32  size=4 type=uw align=2 words (f0.0)
//.declare P20 (258)  rf=f32  size=4 type=uw align=2 words (f2.0)
//.declare V0229 (259)  rf=r size=128 type=ud alias=V0227+0 align=32 words (r74.0)
//.declare  (263)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare  (264)  rf=r size=16 type=q align=8 words (r6.4)
//.declare  (265)  rf=r size=64 type=uw align=32 words (r47.0)
//.declare  (266)  rf=r size=64 type=uw align=32 words (r76.0)
//.declare  (267)  rf=r size=64 type=uw align=32 words (r77.0)
//.declare  (268)  rf=r size=64 type=uw align=32 words (r78.0)
//.declare  (269)  rf=r size=2 type=uw align=1 words (r6.30)
//.declare  (273)  rf=r size=128 type=uw align=32 words (r79.0)
//.declare  (274)  rf=r size=128 type=uw align=32 words (r81.0)
//.declare  (275)  rf=r size=128 type=uw align=32 words (r83.0)
//.declare  (276)  rf=r size=128 type=uw align=32 words (r85.0)
//.declare  (277)  rf=r size=128 type=uw align=32 words (r87.0)
//.declare  (278)  rf=r size=128 type=uw align=32 words (r89.0)
//.declare  (281)  rf=r size=128 type=q align=32 words (r91.0)
//.declare  (282)  rf=r size=128 type=q align=32 words (r93.0)
//.declare  (283)  rf=r size=128 type=q align=32 words (r95.0)
//.declare  (284)  rf=r size=128 type=q align=32 words (r97.0)
//.declare  (289)  rf=r size=128 type=q align=32 words (r99.0)
//.declare  (290)  rf=r size=128 type=q align=32 words (r101.0)
//.declare  (291)  rf=r size=128 type=q align=32 words (r103.0)
//.declare  (292)  rf=r size=128 type=q align=32 words (r105.0)
//.declare  (297)  rf=r size=128 type=q align=32 words (r107.0)
//.declare  (298)  rf=r size=128 type=q align=32 words (r109.0)
//.declare  (299)  rf=r size=128 type=q align=32 words (r111.0)
//.declare  (300)  rf=r size=128 type=q align=32 words (r113.0)
//.declare  (303)  rf=r size=128 type=q align=32 words (r115.0)
//.declare  (304)  rf=r size=128 type=q align=32 words (r117.0)
//.declare  (305)  rf=r size=128 type=q align=32 words (r119.0)
//.declare  (306)  rf=r size=128 type=q align=32 words (r121.0)
//.declare  (311)  rf=r size=128 type=d align=32 words (r123.0)
//.declare  (312)  rf=r size=128 type=d align=32 words (r125.0)
//.declare  (313)  rf=r size=128 type=q align=32 words (r127.0)
//.declare  (314)  rf=r size=128 type=q align=32 words (r129.0)
//.declare  (315)  rf=r size=128 type=d align=32 words (r131.0)
//.declare  (316)  rf=r size=128 type=d align=32 words (r133.0)
//.declare  (317)  rf=r size=128 type=q align=32 words (r135.0)
//.declare  (318)  rf=r size=128 type=q align=32 words (r137.0)
//.declare  (319)  rf=r size=128 type=q align=32 words (r139.0)
//.declare  (320)  rf=r size=128 type=q align=32 words (r141.0)
//.declare  (321)  rf=r size=128 type=q align=32 words (r143.0)
//.declare  (322)  rf=r size=128 type=q align=32 words (r145.0)
//.declare  (325)  rf=r size=128 type=d align=32 words (r147.0)
//.declare  (326)  rf=r size=128 type=d align=32 words (r149.0)
//.declare  (327)  rf=r size=128 type=q align=32 words (r151.0)
//.declare  (328)  rf=r size=128 type=q align=32 words (r153.0)
//.declare  (329)  rf=r size=128 type=d align=32 words (r155.0)
//.declare  (330)  rf=r size=128 type=d align=32 words (r157.0)
//.declare  (331)  rf=r size=128 type=q align=32 words (r159.0)
//.declare  (332)  rf=r size=128 type=q align=32 words (r161.0)
//.declare  (333)  rf=r size=128 type=q align=32 words (r163.0)
//.declare  (334)  rf=r size=128 type=q align=32 words (r165.0)
//.declare  (335)  rf=r size=128 type=q align=32 words (r167.0)
//.declare  (336)  rf=r size=128 type=q align=32 words (r169.0)
//.declare  (339)  rf=r size=128 type=q align=32 words (r171.0)
//.declare  (340)  rf=r size=128 type=q align=32 words (r173.0)
//.declare  (343)  rf=r size=128 type=q align=32 words (r175.0)
//.declare  (344)  rf=r size=128 type=q align=32 words (r177.0)
//.declare  (345)  rf=r size=128 type=q align=32 words (r179.0)
//.declare  (346)  rf=r size=128 type=q align=32 words (r181.0)
//.declare  (349)  rf=r size=128 type=q align=32 words (r183.0)
//.declare  (350)  rf=r size=128 type=q align=32 words (r185.0)
//.declare  (353)  rf=r size=128 type=d align=32 words (r187.0)
//.declare  (354)  rf=r size=128 type=d align=32 words (r189.0)
//.declare  (355)  rf=r size=128 type=d alias=+0 align=32 words (r95.0)
//.declare  (356)  rf=r size=128 type=d alias=+0 align=32 words (r97.0)
//.declare  (357)  rf=r size=128 type=d alias=+0 align=32 words (r103.0)
//.declare  (358)  rf=r size=128 type=d alias=+0 align=32 words (r105.0)
//.declare  (359)  rf=r size=128 type=d alias=+0 align=32 words (r111.0)
//.declare  (360)  rf=r size=128 type=d alias=+0 align=32 words (r113.0)
//.declare  (361)  rf=r size=128 type=d alias=+0 align=32 words (r115.0)
//.declare  (362)  rf=r size=128 type=d alias=+0 align=32 words (r117.0)
//.declare  (363)  rf=r size=128 type=ud alias=+0 align=32 words (r123.0)
//.declare  (364)  rf=r size=128 type=d alias=+0 align=32 words (r127.0)
//.declare  (365)  rf=r size=128 type=d alias=+0 align=32 words (r129.0)
//.declare  (366)  rf=r size=128 type=ud alias=+0 align=32 words (r131.0)
//.declare  (367)  rf=r size=128 type=d alias=+0 align=32 words (r135.0)
//.declare  (368)  rf=r size=128 type=d alias=+0 align=32 words (r137.0)
//.declare  (369)  rf=r size=128 type=ud alias=+0 align=32 words (r147.0)
//.declare  (370)  rf=r size=128 type=d alias=+0 align=32 words (r151.0)
//.declare  (371)  rf=r size=128 type=d alias=+0 align=32 words (r153.0)
//.declare  (372)  rf=r size=128 type=ud alias=+0 align=32 words (r155.0)
//.declare  (373)  rf=r size=128 type=d alias=+0 align=32 words (r159.0)
//.declare  (374)  rf=r size=128 type=d alias=+0 align=32 words (r161.0)
//.declare  (375)  rf=r size=128 type=d alias=+0 align=32 words (r171.0)
//.declare  (376)  rf=r size=128 type=d alias=+0 align=32 words (r173.0)
//.declare  (377)  rf=r size=128 type=d alias=+0 align=32 words (r175.0)
//.declare  (378)  rf=r size=128 type=d alias=+0 align=32 words (r177.0)
//.declare  (379)  rf=r size=128 type=d alias=+0 align=32 words (r183.0)
//.declare  (380)  rf=r size=128 type=d alias=+0 align=32 words (r185.0)
//.declare r0 (381)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (382)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare inlineRegFromTDL (383)  rf=r size=32 type=ud align=2 words (r1.0)
//.declare inlineRegExpectedLocation (384)  rf=r size=32 type=ud align=2 words (r4.0)
//.declare  (385)  rf=r size=128 type=ud align=32 words (r1.0)
//.declare  (386)  rf=r size=64 type=ud align=32 words (r3.0)
//.declare  (387)  rf=r size=128 type=ud align=32 words (r5.0)

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
// | V0046    | :q       |    0x8 | r5+0x8   | cti+0x28         |
// | V0047    | :q       |    0x8 | r5+0x10  | cti+0x30         |
// | V0048    | :q       |    0x8 | r5+0x18  | cti+0x38         |
// | V0049    | :q       |    0x8 | r5+0x20  | cti+0x40         |
// | V0050    | :q       |    0x8 | r5+0x28  | cti+0x48         |
// | V0051    | :q       |    0x8 | r5+0x30  | cti+0x50         |
// | V0052    | :q       |    0x8 | r5+0x38  | cti+0x58         |
// | V0053    | :q       |    0x8 | r6       | cti+0x60         |
// | V0054    | :q       |    0x8 | r6+0x8   | cti+0x68         |
// | V0055    | :q       |    0x8 | r6+0x10  | cti+0x70         |
// | V0056    | :q       |    0x8 | r6+0x18  | cti+0x78         |
// | V0042    | :uq      |    0x8 | r6+0x20  | cti+0x80         |
// | V0043    | :uq      |    0x8 | r6+0x28  | cti+0x88         |
// | V0037    | :d x 3   |    0xC | r6+0x30  | cti+0x90         |
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (16|M0)              r255.0<1>:ud  0x0:ud                                                //  ALU pipe: int; 
(W)     and (1|M0)               r255.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     and (1|M0)               r255.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             //  ALU pipe: int; 
(W)     add (1|M0)               r255.2<1>:ud  r255.2<0;1,0>:ud  0x80:ud              {I@2}          //  ALU pipe: int; 
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
(W)     load.ugm.d32x32t.a32.ca.cc (1|M0)  r5:2 bti[255][r255:1]   {I@1,$2} // ex_desc:0xFF000000; desc:0x6229E500 // 
// B002: Preds:{B001},  Succs:{B003, B011}
// _main:
(W)     mov (16|M0)              r7.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mul (1|M0)               acc0.0<1>:ud  r4.5<0;1,0>:ud    r7.14<0;1,0>:uw  {A@1}              //  ALU pipe: int; $5
        mov (16|M0)              r83.0<4>:uw   r2.0<1;1,0>:uw                   {$0.dst}             //  ALU pipe: int; $36
(W)     macl (1|M0)              r9.0<1>:ud    r4.5<0;1,0>:ud    r7.7<0;1,0>:ud                      //  ALU pipe: int; $6
(W)     mul (1|M0)               acc0.0<1>:ud  r4.5<0;1,0>:ud    r7.14<0;1,0>:uw                     //  ALU pipe: int; $6
        mov (16|M16)             r85.0<4>:uw   r2.16<1;1,0>:uw                                       //  ALU pipe: int; $36
(W)     mach (1|M0)              r8.0<1>:d     r4.5<0;1,0>:ud    r7.7<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mul (1|M0)               acc0.0<1>:ud  r4.4<0;1,0>:ud    r7.12<0;1,0>:uw                     //  ALU pipe: int; $24
        mov (16|M0)              r87.0<4>:uw   r1.0<1;1,0>:uw                                        //  ALU pipe: int; $55
(W)     macl (1|M0)              r15.0<1>:ud   r4.4<0;1,0>:ud    r7.6<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $25
(W)     mul (1|M0)               acc0.0<1>:ud  r4.4<0;1,0>:ud    r7.12<0;1,0>:uw                     //  ALU pipe: int; $25
        mov (16|M16)             r89.0<4>:uw   r1.16<1;1,0>:uw                                       //  ALU pipe: int; $55
(W)     mach (1|M0)              r14.0<1>:d    r4.4<0;1,0>:ud    r7.6<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mul (1|M0)               acc0.0<1>:ud  r4.3<0;1,0>:ud    r7.2<0;1,0>:uw                      //  ALU pipe: int; $43
        mov (16|M0)              r79.0<4>:uw   r3.0<1;1,0>:uw                   {$1.dst}             //  ALU pipe: int; $17
(W)     macl (1|M0)              r21.0<1>:ud   r4.3<0;1,0>:ud    r7.1<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $44
(W)     mul (1|M0)               acc0.0<1>:ud  r4.3<0;1,0>:ud    r7.2<0;1,0>:uw                      //  ALU pipe: int; $44
(W)     mov (1|M0)               r15.1<1>:d    r14.0<0;1,0>:d                   {Compacted,I@5}      //  ALU pipe: int; $30
        mov (16|M16)             r81.0<4>:uw   r3.16<1;1,0>:uw                                       //  ALU pipe: int; $17
(W)     mach (1|M0)              r20.0<1>:d    r4.3<0;1,0>:ud    r7.1<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mov (1|M0)               r9.1<1>:d     r8.0<0;1,0>:d                    {Compacted}          //  ALU pipe: int; $11
        add (16|M0)              r99.0<1>:q    r15.0<0;1,0>:q    r83.0<4;1,0>:uw  {I@4}              //  ALU pipe: int; $36
        add (16|M16)             r101.0<1>:q   r15.0<0;1,0>:q    r85.0<4;1,0>:uw                     //  ALU pipe: int; $36
(W)     mov (1|M0)               r21.1<1>:d    r20.0<0;1,0>:d                   {Compacted,I@4}      //  ALU pipe: int; $49
        add (16|M0)              r103.0<1>:q   r99.0<1;1,0>:q    r4.1<0;1,0>:ud   {I@3}              //  ALU pipe: int; $38
        add (16|M16)             r105.0<1>:q   r101.0<1;1,0>:q   r4.1<0;1,0>:ud   {I@3}              //  ALU pipe: int; $38
        add (16|M0)              r107.0<1>:q   r21.0<0;1,0>:q    r87.0<4;1,0>:uw  {I@3}              //  ALU pipe: int; $55
        add (16|M16)             r109.0<1>:q   r21.0<0;1,0>:q    r89.0<4;1,0>:uw                     //  ALU pipe: int; $55
(W)     mov (2|M0)               r4.10<1>:d    r5.0<1;1,0>:d                    {$2.dst}             //  ALU pipe: int; $3
        add (16|M0)              r111.0<1>:q   r107.0<1;1,0>:q   r4.0<0;1,0>:ud   {I@3}              //  ALU pipe: int; $57
        add (16|M16)             r113.0<1>:q   r109.0<1;1,0>:q   r4.0<0;1,0>:ud   {I@3}              //  ALU pipe: int; $57
(W)     mov (2|M0)               r4.12<1>:d    r5.2<1;1,0>:d                                         //  ALU pipe: int; $4
        mov (16|M0)              r16.0<1>:d    r103.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $39
        mov (16|M16)             r17.0<1>:d    r105.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $40
        mov (16|M0)              r22.0<1>:d    r111.0<2;1,0>:d                  {Compacted,I@5}      //  ALU pipe: int; $58
        mov (16|M16)             r23.0<1>:d    r113.0<2;1,0>:d                  {Compacted,I@5}      //  ALU pipe: int; $59
        add (16|M0)              r91.0<1>:q    r9.0<0;1,0>:q     r79.0<4;1,0>:uw                     //  ALU pipe: int; $17
        add (16|M16)             r93.0<1>:q    r9.0<0;1,0>:q     r81.0<4;1,0>:uw                     //  ALU pipe: int; $17
        cmp (32|M0)   (lt)f2.0   null<1>:ud    r16.0<1;1,0>:ud   r4.10<0;1,0>:ud  {I@5}              //  ALU pipe: int; $67
        cmp (32|M0)   (lt)f1.0   null<1>:ud    r22.0<1;1,0>:ud   r4.12<0;1,0>:ud  {I@4}              //  ALU pipe: int; $72
        mov (16|M0)              r18.0<1>:d    r103.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $41
        mov (16|M16)             r19.0<1>:d    r105.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $42
        mov (16|M0)              r24.0<1>:d    r111.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $60
        mov (16|M16)             r25.0<1>:d    r113.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $61
        add (16|M0)              r95.0<1>:q    r91.0<1;1,0>:q    r4.2<0;1,0>:ud   {I@7}              //  ALU pipe: int; $19
        add (16|M16)             r97.0<1>:q    r93.0<1;1,0>:q    r4.2<0;1,0>:ud   {I@7}              //  ALU pipe: int; $19
(W)     mov (2|M0)               r4.8<1>:d     r4.6<1;1,0>:d                                         //  ALU pipe: int; $2
(f2.0)  cmp (32|M0)   (eq)f2.0   null<1>:d     r18.0<1;1,0>:d    r4.11<0;1,0>:d   {I@6}              //  ALU pipe: int; $68
(f1.0)  cmp (32|M0)   (eq)f1.0   null<1>:d     r24.0<1;1,0>:d    r4.13<0;1,0>:d   {I@5}              //  ALU pipe: int; $73
        mov (16|M0)              r10.0<1>:d    r95.0<2;1,0>:d                   {Compacted,I@5}      //  ALU pipe: int; $20
        mov (16|M16)             r11.0<1>:d    r97.0<2;1,0>:d                   {Compacted,I@5}      //  ALU pipe: int; $21
        mov (16|M0)              r12.0<1>:d    r95.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $22
        mov (16|M16)             r13.0<1>:d    r97.1<2;1,0>:d                   {Compacted}          //  ALU pipe: int; $23
        cmp (32|M0)   (lt)f3.0   null<1>:ud    r10.0<1;1,0>:ud   r4.8<0;1,0>:ud   {I@3}              //  ALU pipe: int; $62
(~f2.0) cmp (32|M0)   (lt)f2.0   null<1>:ud    r18.0<1;1,0>:ud   r4.11<0;1,0>:ud                     //  ALU pipe: int; $70
(~f1.0) cmp (32|M0)   (lt)f1.0   null<1>:ud    r24.0<1;1,0>:ud   r4.13<0;1,0>:ud                     //  ALU pipe: int; $75
(W)     mov (1|M0)               r6.30<1>:hf   0x1:hf                                                //  ALU pipe: float; $77
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:d     r12.0<1;1,0>:d    r4.9<0;1,0>:d    {I@4}              //  ALU pipe: int; $63
(f1.0)  sel (32|M0)              r77.0<1>:uw   r6.30<0;1,0>:uw   0x0:uw              {F@1}           //  ALU pipe: int; $77
(f2.0)  sel (32|M0)              r78.0<1>:uw   r6.30<0;1,0>:uw   0x0:uw                              //  ALU pipe: int; $77
(~f3.0) cmp (32|M0)   (lt)f3.0   null<1>:ud    r12.0<1;1,0>:ud   r4.9<0;1,0>:ud                      //  ALU pipe: int; $65
        and (32|M0)   (ne)f1.0   null<2>:uw    r77.0<1;1,0>:uw   r78.0<1;1,0>:uw  {I@2}              //  ALU pipe: int; $77
(f3.0)  sel (32|M0)              r76.0<1>:uw   r6.30<0;1,0>:uw   0x0:uw                              //  ALU pipe: int; $78
(f1.0)  sel (32|M0)              r47.0<1>:uw   r6.30<0;1,0>:uw   0x0:uw                              //  ALU pipe: int; $78
        and (32|M0)   (ne)f0.0   null<2>:uw    r47.0<1;1,0>:uw   r76.0<1;1,0>:uw  {I@1}              //  ALU pipe: int; $78
(~f0.0) goto (32|M0)                         _0_013            _0_013                                //  ALU pipe: int; $79
// B003: [inDivergent],  Preds:{B002},  Succs:{B004}
_0_014:
        mov (16|M0)              r115.0<1>:q   r103.0<1;1,0>:q                                       //  ALU pipe: int; $88
        mov (16|M16)             r117.0<1>:q   r105.0<1;1,0>:q                                       //  ALU pipe: int; $88
        mov (16|M0)              r119.0<1>:q   r95.0<1;1,0>:q                                        //  ALU pipe: int; $89
        mov (16|M16)             r121.0<1>:q   r97.0<1;1,0>:q                                        //  ALU pipe: int; $89
(W)     mov (2|M0)               r4.14<1>:d    r5.6<1;1,0>:d                                         //  ALU pipe: int; $81
(W)     mov (2|M0)               r5.10<1>:d    r5.8<1;1,0>:d                                         //  ALU pipe: int; $82
(W)     mov (2|M0)               r5.12<1>:d    r6.0<1;1,0>:d                                         //  ALU pipe: int; $83
(W)     mov (2|M0)               r6.4<1>:d     r6.2<1;1,0>:d                                         //  ALU pipe: int; $84
(W)     mov (1|M0)               r6.4<1>:q     r6.12<0;1,0>:ud                                       //  ALU pipe: int; $85
(W)     mov (1|M0)               r6.5<1>:q     r6.13<0;1,0>:ud                                       //  ALU pipe: int; $85
(W)     mov (1|M0)               r6.3<1>:q     r6.14<0;1,0>:ud                                       //  ALU pipe: int; $87
// B004: [inDivergent],  Preds:{B010, B003},  Succs:{B005, B006}
_0_015:
        mov (16|M0)              r30.0<1>:d    r111.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $95
        mov (16|M16)             r31.0<1>:d    r113.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $96
        mov (16|M0)              r32.0<1>:d    r111.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $97
(W)     mul (16|M0)              acc0.0<1>:ud  r30.0<1;1,0>:ud   r5.24<0;1,0>:uw  {I@3}              //  ALU pipe: int; $99
        mov (16|M16)             r33.0<1>:d    r113.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $98
        macl (16|M0)             r123.0<1>:ud  r30.0<1;1,0>:ud   r5.12<0;1,0>:ud                     //  ALU pipe: int; $99
(W)     mul (16|M16)             acc0.0<1>:ud  r31.0<1;1,0>:ud   r5.24<0;1,0>:uw  {I@5}              //  ALU pipe: int; $99
        mov (16|M0)              r26.0<1>:d    r115.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $91
        macl (16|M16)            r124.0<1>:ud  r31.0<1;1,0>:ud   r5.12<0;1,0>:ud                     //  ALU pipe: int; $100
(W)     mul (16|M0)              acc0.0<1>:ud  r30.0<1;1,0>:ud   r5.24<0;1,0>:uw                     //  ALU pipe: int; $100
        mov (16|M16)             r27.0<1>:d    r117.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $92
        mach (16|M0)             r34.0<1>:d    r30.0<1;1,0>:ud   r5.12<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r31.0<1;1,0>:ud   r5.24<0;1,0>:uw                     //  ALU pipe: int; $100
        mov (16|M0)              r28.0<1>:d    r115.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $93
        mach (16|M16)            r35.0<1>:d    r31.0<1;1,0>:ud   r5.12<0;1,0>:ud                     //  ALU pipe: int; $101
(W)     mul (16|M0)              acc0.0<1>:d   r30.0<1;1,0>:ud   r5.26<0;1,0>:uw                     //  ALU pipe: int; $101
        mov (16|M16)             r29.0<1>:d    r117.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $94
        macl (16|M0)             r36.0<1>:d    r30.0<1;1,0>:ud   r5.13<0;1,0>:d                      //  ALU pipe: int; $101
(W)     mul (16|M16)             acc0.0<1>:d   r31.0<1;1,0>:ud   r5.26<0;1,0>:uw                     //  ALU pipe: int; $101
        mov (16|M0)              r127.0<2>:d   r123.0<1;1,0>:d                                       //  ALU pipe: int; $108
        macl (16|M16)            r37.0<1>:d    r31.0<1;1,0>:ud   r5.13<0;1,0>:d                      //  ALU pipe: int; $102
(W)     mul (16|M0)              acc0.0<1>:d   r5.12<0;1,0>:ud   r32.0<2;1,0>:uw                     //  ALU pipe: int; $103
        mov (16|M16)             r129.0<2>:d   r124.0<1;1,0>:d                                       //  ALU pipe: int; $109
        add (32|M0)              r34.0<1>:d    r34.0<1;1,0>:d    r36.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $102
        macl (16|M0)             r36.0<1>:d    r5.12<0;1,0>:ud   r32.0<1;1,0>:d                      //  ALU pipe: int; $103
(W)     mul (16|M16)             acc0.0<1>:d   r5.12<0;1,0>:ud   r33.0<2;1,0>:uw                     //  ALU pipe: int; $103
        add (16|M0)              r171.0<1>:q   r119.0<1;1,0>:q   r6.3<0;1,0>:q    {Compacted}        //  ALU pipe: int; $163
        macl (16|M16)            r37.0<1>:d    r5.12<0;1,0>:ud   r33.0<1;1,0>:d                      //  ALU pipe: int; $105
(W)     mul (16|M0)              acc0.0<1>:ud  r26.0<1;1,0>:ud   r6.8<0;1,0>:uw                      //  ALU pipe: int; $112
        add (16|M16)             r173.0<1>:q   r121.0<1;1,0>:q   r6.3<0;1,0>:q    {Compacted}        //  ALU pipe: int; $163
        macl (16|M0)             r131.0<1>:ud  r26.0<1;1,0>:ud   r6.4<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $112
(W)     mul (16|M16)             acc0.0<1>:ud  r27.0<1;1,0>:ud   r6.8<0;1,0>:uw                      //  ALU pipe: int; $112
        add (32|M0)              r125.0<1>:d   r34.0<1;1,0>:d    r36.0<1;1,0>:d   {Compacted,I@5}    //  ALU pipe: int; $105
        macl (16|M16)            r132.0<1>:ud  r27.0<1;1,0>:ud   r6.4<0;1,0>:ud   {Compacted}        //  ALU pipe: int; $113
(W)     mul (16|M0)              acc0.0<1>:ud  r26.0<1;1,0>:ud   r6.8<0;1,0>:uw                      //  ALU pipe: int; $113
        mov (16|M0)              r127.1<2>:d   r125.0<1;1,0>:d                  {I@3}                //  ALU pipe: int; $110
        mach (16|M0)             r38.0<1>:d    r26.0<1;1,0>:ud   r6.4<0;1,0>:ud                      //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r27.0<1;1,0>:ud   r6.8<0;1,0>:uw                      //  ALU pipe: int; $113
        mov (16|M16)             r129.1<2>:d   r126.0<1;1,0>:d                                       //  ALU pipe: int; $111
        mach (16|M16)            r39.0<1>:d    r27.0<1;1,0>:ud   r6.4<0;1,0>:ud                      //  ALU pipe: int; $114
(W)     mul (16|M0)              acc0.0<1>:d   r26.0<1;1,0>:ud   r6.10<0;1,0>:uw                     //  ALU pipe: int; $114
        mov (16|M0)              r135.0<2>:d   r131.0<1;1,0>:d                                       //  ALU pipe: int; $121
        macl (16|M0)             r40.0<1>:d    r26.0<1;1,0>:ud   r6.5<0;1,0>:d                       //  ALU pipe: int; $114
(W)     mul (16|M16)             acc0.0<1>:d   r27.0<1;1,0>:ud   r6.10<0;1,0>:uw                     //  ALU pipe: int; $114
        mov (16|M16)             r137.0<2>:d   r132.0<1;1,0>:d                  {I@7}                //  ALU pipe: int; $122
        macl (16|M16)            r41.0<1>:d    r27.0<1;1,0>:ud   r6.5<0;1,0>:d                       //  ALU pipe: int; $115
(W)     mul (16|M0)              acc0.0<1>:d   r6.4<0;1,0>:ud    r28.0<2;1,0>:uw                     //  ALU pipe: int; $116
        add (16|M0)              r139.0<1>:q   r127.0<1;1,0>:q   r5.7<0;1,0>:q    {Compacted}        //  ALU pipe: int; $125
        add (32|M0)              r38.0<1>:d    r38.0<1;1,0>:d    r40.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $115
        macl (16|M0)             r40.0<1>:d    r6.4<0;1,0>:ud    r28.0<1;1,0>:d                      //  ALU pipe: int; $116
(W)     mul (16|M16)             acc0.0<1>:d   r6.4<0;1,0>:ud    r29.0<2;1,0>:uw                     //  ALU pipe: int; $116
        add (16|M16)             r141.0<1>:q   r129.0<1;1,0>:q   r5.7<0;1,0>:q    {Compacted}        //  ALU pipe: int; $125
        macl (16|M16)            r41.0<1>:d    r6.4<0;1,0>:ud    r29.0<1;1,0>:d                      //  ALU pipe: int; $118
(W)     mul (16|M0)              acc0.0<1>:ud  r30.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $131
        mov (16|M0)              r64.0<1>:d    r171.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $164
        add (32|M0)              r133.0<1>:d   r38.0<1;1,0>:d    r40.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $118
        macl (16|M0)             r147.0<1>:ud  r30.0<1;1,0>:ud   r4.14<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $131
        mov (16|M0)              r135.1<2>:d   r133.0<1;1,0>:d                  {I@2}                //  ALU pipe: int; $123
        mov (16|M16)             r137.1<2>:d   r134.0<1;1,0>:d                                       //  ALU pipe: int; $124
(W)     mul (16|M16)             acc0.0<1>:ud  r31.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $131
        add (16|M0)              r143.0<1>:q   r139.0<1;1,0>:q   r135.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $126
        add (16|M16)             r145.0<1>:q   r141.0<1;1,0>:q   r137.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $126
        macl (16|M16)            r148.0<1>:ud  r31.0<1;1,0>:ud   r4.14<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $132
        add (16|M0)              r42.0<1>:q    r143.0<1;1,0>:q   r119.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $127
        add (16|M16)             r44.0<1>:q    r145.0<1;1,0>:q   r121.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $127
(W)     mul (16|M0)              acc0.0<1>:ud  r30.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $132
        mov (16|M16)             r65.0<1>:d    r173.0<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $165
        load.ugm.d8u32.a64 (32|M0)  r48:2       [r42:4]            {I@3,$4} // ex_desc:0x0; desc:0x8200980 // $129
        mach (16|M0)             r50.0<1>:d    r30.0<1;1,0>:ud   r4.14<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r31.0<1;1,0>:ud   r4.28<0;1,0>:uw                     //  ALU pipe: int; $132
        cmp (32|M0)   (lt)f0.0   null<1>:ud    r64.0<1;1,0>:ud   r4.8<0;1,0>:ud   {I@3}              //  ALU pipe: int; $168
        mach (16|M16)            r51.0<1>:d    r31.0<1;1,0>:ud   r4.14<0;1,0>:ud                     //  ALU pipe: int; $133
(W)     mul (16|M0)              acc0.0<1>:d   r30.0<1;1,0>:ud   r4.30<0;1,0>:uw                     //  ALU pipe: int; $133
        mov (16|M0)              r151.0<2>:d   r147.0<1;1,0>:d                                       //  ALU pipe: int; $140
        macl (16|M0)             r52.0<1>:d    r30.0<1;1,0>:ud   r4.15<0;1,0>:d                      //  ALU pipe: int; $133
(W)     mul (16|M16)             acc0.0<1>:d   r31.0<1;1,0>:ud   r4.30<0;1,0>:uw                     //  ALU pipe: int; $133
        mov (16|M16)             r153.0<2>:d   r148.0<1;1,0>:d                                       //  ALU pipe: int; $141
        macl (16|M16)            r53.0<1>:d    r31.0<1;1,0>:ud   r4.15<0;1,0>:d                      //  ALU pipe: int; $134
(W)     mul (16|M0)              acc0.0<1>:d   r4.14<0;1,0>:ud   r32.0<2;1,0>:uw                     //  ALU pipe: int; $135
        mov (16|M0)              r66.0<1>:d    r171.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $166
        add (32|M0)              r50.0<1>:d    r50.0<1;1,0>:d    r52.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $134
        macl (16|M0)             r52.0<1>:d    r4.14<0;1,0>:ud   r32.0<1;1,0>:d                      //  ALU pipe: int; $135
(W)     mul (16|M16)             acc0.0<1>:d   r4.14<0;1,0>:ud   r33.0<2;1,0>:uw                     //  ALU pipe: int; $135
        mov (16|M16)             r67.0<1>:d    r173.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $167
        macl (16|M16)            r53.0<1>:d    r4.14<0;1,0>:ud   r33.0<1;1,0>:d                      //  ALU pipe: int; $137
(W)     mul (16|M0)              acc0.0<1>:ud  r26.0<1;1,0>:ud   r5.20<0;1,0>:uw                     //  ALU pipe: int; $144
(f0.0)  cmp (32|M0)   (eq)f0.0   null<1>:d     r66.0<1;1,0>:d    r4.9<0;1,0>:d    {I@3}              //  ALU pipe: int; $169
        macl (16|M0)             r155.0<1>:ud  r26.0<1;1,0>:ud   r5.10<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $144
(W)     mul (16|M16)             acc0.0<1>:ud  r27.0<1;1,0>:ud   r5.20<0;1,0>:uw                     //  ALU pipe: int; $144
        add (32|M0)              r149.0<1>:d   r50.0<1;1,0>:d    r52.0<1;1,0>:d   {Compacted,I@5}    //  ALU pipe: int; $137
        macl (16|M16)            r156.0<1>:ud  r27.0<1;1,0>:ud   r5.10<0;1,0>:ud  {Compacted}        //  ALU pipe: int; $145
(W)     mul (16|M0)              acc0.0<1>:ud  r26.0<1;1,0>:ud   r5.20<0;1,0>:uw                     //  ALU pipe: int; $145
        mov (16|M0)              r151.1<2>:d   r149.0<1;1,0>:d                  {I@3}                //  ALU pipe: int; $142
        mach (16|M0)             r54.0<1>:d    r26.0<1;1,0>:ud   r5.10<0;1,0>:ud                     //  ALU pipe: int; 
(W)     mul (16|M0)              acc0.0<1>:ud  r27.0<1;1,0>:ud   r5.20<0;1,0>:uw                     //  ALU pipe: int; $145
        mov (16|M16)             r153.1<2>:d   r150.0<1;1,0>:d                                       //  ALU pipe: int; $143
        mach (16|M16)            r55.0<1>:d    r27.0<1;1,0>:ud   r5.10<0;1,0>:ud                     //  ALU pipe: int; $146
(W)     mul (16|M0)              acc0.0<1>:d   r26.0<1;1,0>:ud   r5.22<0;1,0>:uw                     //  ALU pipe: int; $146
        mov (16|M0)              r159.0<2>:d   r155.0<1;1,0>:d                                       //  ALU pipe: int; $153
        macl (16|M0)             r56.0<1>:d    r26.0<1;1,0>:ud   r5.11<0;1,0>:d                      //  ALU pipe: int; $146
(W)     mul (16|M16)             acc0.0<1>:d   r27.0<1;1,0>:ud   r5.22<0;1,0>:uw                     //  ALU pipe: int; $146
        mov (16|M16)             r161.0<2>:d   r156.0<1;1,0>:d                  {I@7}                //  ALU pipe: int; $154
        macl (16|M16)            r57.0<1>:d    r27.0<1;1,0>:ud   r5.11<0;1,0>:d                      //  ALU pipe: int; $147
(W)     mul (16|M0)              acc0.0<1>:d   r5.10<0;1,0>:ud   r28.0<2;1,0>:uw                     //  ALU pipe: int; $148
        add (16|M0)              r163.0<1>:q   r151.0<1;1,0>:q   r5.2<0;1,0>:q    {Compacted}        //  ALU pipe: int; $157
        add (32|M0)              r54.0<1>:d    r54.0<1;1,0>:d    r56.0<1;1,0>:d   {Compacted,I@3}    //  ALU pipe: int; $147
        macl (16|M0)             r56.0<1>:d    r5.10<0;1,0>:ud   r28.0<1;1,0>:d                      //  ALU pipe: int; $148
(W)     mul (16|M16)             acc0.0<1>:d   r5.10<0;1,0>:ud   r29.0<2;1,0>:uw                     //  ALU pipe: int; $148
        add (16|M16)             r165.0<1>:q   r153.0<1;1,0>:q   r5.2<0;1,0>:q    {Compacted}        //  ALU pipe: int; $157
        macl (16|M16)            r57.0<1>:d    r5.10<0;1,0>:ud   r29.0<1;1,0>:d                      //  ALU pipe: int; $150
(~f0.0) cmp (32|M0)   (lt)f0.0   null<1>:ud    r66.0<1;1,0>:ud   r4.9<0;1,0>:ud                      //  ALU pipe: int; $171
        add (32|M0)              r157.0<1>:d   r54.0<1;1,0>:d    r56.0<1;1,0>:d   {Compacted,I@2}    //  ALU pipe: int; $150
        mov (32|M0)              r46.0<1>:b    r48.0<4;1,0>:b                   {$4.dst}             //  ALU pipe: int; $130
        mov (16|M0)              r159.1<2>:d   r157.0<1;1,0>:d                  {I@2}                //  ALU pipe: int; $155
        mov (16|M16)             r161.1<2>:d   r158.0<1;1,0>:d                                       //  ALU pipe: int; $156
        mov (32|M0)              r62.0<1>:ud   r46.0<1;1,0>:ub                  {@3,$3.src}          //  ALU pipe: int; $161
        add (16|M0)              r167.0<1>:q   r163.0<1;1,0>:q   r159.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $158
        add (16|M16)             r169.0<1>:q   r165.0<1;1,0>:q   r161.0<1;1,0>:q  {Compacted,I@3}    //  ALU pipe: int; $158
        add (16|M0)              r58.0<1>:q    r167.0<1;1,0>:q   r119.0<1;1,0>:q  {Compacted,I@2}    //  ALU pipe: int; $159 R{} IR{}{O:3,O:3,},  R{} IR{}{E:4,E:12,},  {BC=1}
        add (16|M16)             r60.0<1>:q    r169.0<1;1,0>:q   r121.0<1;1,0>:q  {Compacted,I@2}    //  ALU pipe: int; $159 R{} IR{}{O:4,O:4,},  R{} IR{}{E:5,E:13,},  {BC=1}
        store.ugm.d8u32.a64 (32|M0)  [r58:4]    r62:2              {I@1,$3} // ex_desc:0x0; desc:0x8000984 // $162
(~f0.0) goto (32|M0)                         _0_016            _0_016                                //  ALU pipe: int; $173
// B005: [inDivergent],  Preds:{B004},  Succs:{B010}
_0_017:
        mov (16|M0)              r175.0<1>:q   r115.0<1;1,0>:q                                       //  ALU pipe: int; $175
        mov (16|M16)             r177.0<1>:q   r117.0<1;1,0>:q                                       //  ALU pipe: int; $175
        mov (16|M0)              r179.0<1>:q   r171.0<1;1,0>:q                                       //  ALU pipe: int; $176
        mov (16|M16)             r181.0<1>:q   r173.0<1;1,0>:q                                       //  ALU pipe: int; $176
        goto (32|M0)                         _0_016            _0_018                                // $177
// B006: [inDivergent],  Preds:{B004},  Succs:{B007, B008}
_0_016:
        join (32|M0)                         _0_018                                                  // 
L3080:
        add (16|M0)              r175.0<1>:q   r115.0<1;1,0>:q   r6.5<0;1,0>:q    {Compacted}        //  ALU pipe: int; $179
        add (16|M16)             r177.0<1>:q   r117.0<1;1,0>:q   r6.5<0;1,0>:q    {Compacted}        //  ALU pipe: int; $179
        mov (16|M0)              r68.0<1>:d    r175.0<2;1,0>:d                  {Compacted,I@2}      //  ALU pipe: int; $180
        mov (16|M16)             r69.0<1>:d    r177.0<2;1,0>:d                  {Compacted,I@2}      //  ALU pipe: int; $181
        mov (16|M0)              r70.0<1>:d    r175.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $182
        mov (16|M16)             r71.0<1>:d    r177.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $183
        cmp (32|M0)   (lt)f3.0   null<1>:ud    r68.0<1;1,0>:ud   r4.10<0;1,0>:ud  {I@3}              //  ALU pipe: int; $184 R{} IR{}{E:2,E:2,},  R{r4,} IR{} {BC=1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:d     r70.0<1;1,0>:d    r4.11<0;1,0>:d   {I@2}              //  ALU pipe: int; $185
(~f3.0) cmp (32|M0)   (lt)f3.0   null<1>:ud    r70.0<1;1,0>:ud   r4.11<0;1,0>:ud                     //  ALU pipe: int; $187
(~f3.0) goto (32|M0)                         _0_019            _0_019                                //  ALU pipe: int; $189
// B007: [inDivergent],  Preds:{B006},  Succs:{B010}
_0_020:
        mov (16|M0)              r179.0<1>:q   r95.0<1;1,0>:q                                        //  ALU pipe: int; $191
        mov (16|M16)             r181.0<1>:q   r97.0<1;1,0>:q                                        //  ALU pipe: int; $191
        goto (32|M0)                         _0_019            _0_018                                // $192
// B008: [inDivergent],  Preds:{B006},  Succs:{B009, B011}
_0_019:
        join (32|M0)                         _0_018                                                  // 
L3256:
        add (16|M0)              r183.0<1>:q   r111.0<1;1,0>:q   r6.4<0;1,0>:q    {Compacted}        //  ALU pipe: int; $194
        add (16|M16)             r185.0<1>:q   r113.0<1;1,0>:q   r6.4<0;1,0>:q    {Compacted}        //  ALU pipe: int; $194
        mov (16|M0)              r72.0<1>:d    r183.0<2;1,0>:d                  {Compacted,I@2}      //  ALU pipe: int; $195
        mov (16|M16)             r73.0<1>:d    r185.0<2;1,0>:d                  {Compacted,I@2}      //  ALU pipe: int; $196
        mov (16|M0)              r74.0<1>:d    r183.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $197
        mov (16|M16)             r75.0<1>:d    r185.1<2;1,0>:d                  {Compacted}          //  ALU pipe: int; $198
        cmp (32|M0)   (lt)f2.0   null<1>:ud    r72.0<1;1,0>:ud   r4.12<0;1,0>:ud  {I@3}              //  ALU pipe: int; $199
(f2.0)  cmp (32|M0)   (eq)f2.0   null<1>:d     r74.0<1;1,0>:d    r4.13<0;1,0>:d   {I@2}              //  ALU pipe: int; $200
(~f2.0) cmp (32|M0)   (lt)f2.0   null<1>:ud    r74.0<1;1,0>:ud   r4.13<0;1,0>:ud                     //  ALU pipe: int; $202
(~f2.0) goto (32|M0)                         _0_018            _0_013                                //  ALU pipe: int; $204
// B009: [inDivergent],  Preds:{B008},  Succs:{B010}
_0_021:
(f2.0)  sel (32|M0)              r187.0<1>:d   r72.0<1;1,0>:d    r22.0<1;1,0>:d                      //  ALU pipe: int; $206
(f2.0)  sel (32|M0)              r189.0<1>:d   r74.0<1;1,0>:d    r24.0<1;1,0>:d                      //  ALU pipe: int; $207
        mov (16|M0)              r111.0<2>:d   r187.0<1;1,0>:d                  {I@2}                //  ALU pipe: int; $210
        mov (16|M16)             r113.0<2>:d   r188.0<1;1,0>:d                                       //  ALU pipe: int; $211
        mov (16|M0)              r175.0<1>:q   r103.0<1;1,0>:q                                       //  ALU pipe: int; $214
        mov (16|M16)             r177.0<1>:q   r105.0<1;1,0>:q                                       //  ALU pipe: int; $214
        mov (16|M0)              r179.0<1>:q   r95.0<1;1,0>:q                                        //  ALU pipe: int; $215
        mov (16|M16)             r181.0<1>:q   r97.0<1;1,0>:q                                        //  ALU pipe: int; $215
        mov (16|M0)              r111.1<2>:d   r189.0<1;1,0>:d                  {I@7}                //  ALU pipe: int; $212
        mov (16|M16)             r113.1<2>:d   r190.0<1;1,0>:d                                       //  ALU pipe: int; $213
// B010: [inDivergent],  Preds:{B009, B007, B005},  Succs:{B004}
_0_018:
        join (32|M0)                         _0_013                                                  // 
L3544:
        mov (16|M0)              r115.0<1>:q   r175.0<1;1,0>:q                  {I@7}                //  ALU pipe: int; $217
        mov (16|M16)             r117.0<1>:q   r177.0<1;1,0>:q                  {I@7}                //  ALU pipe: int; $217
        mov (16|M0)              r119.0<1>:q   r179.0<1;1,0>:q                  {I@7}                //  ALU pipe: int; $218
        mov (16|M16)             r121.0<1>:q   r181.0<1;1,0>:q                  {I@7}                //  ALU pipe: int; $218
(W)     jmpi                                 _0_015                                                  // $219
// B011: Preds:{B008, B002},  Succs:{}
_0_013:
        join (32|M0)                         L3640                                                   // 
L3640:
(W)     mov (16|M0)              r255.0<1>:f   r7.0<1;1,0>:f                    {Compacted}          //  ALU pipe: float; $221
(W)     send.gtwy (1|M0)         null     r255  null:0  0x0            0x02000010           {EOT,F@1,$5} // wr:1+0, rd:0; end of thread // $221
L3664:
(W)     mov (16|M0)              null<1>:ud    0x23954D4A:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x795ECA46:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0x1:ud                                                // 


//.BankConflicts: 3
//.ByteRMWs: 0
//


//.numALUInst: 249
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
//.AfterWriteTokenDepCount: 4
//.AfterReadTokenDepCount: 3

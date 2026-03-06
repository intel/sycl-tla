//.kernel Intel_Symbol_Table_Void_Program
//.platform XE2
//.thread_config numGRF=256, numAcc=8, numSWSB=32
//.options_string "-emitCrossThreadOffR0Reloc -perfmodel -hashmovs 4208518013 2685686631 -hashmovs1 0 13 "
//.full_options "-emitLocation -wideMulMadOpsEn -enableCoalesceScalarMoves -supportLSCImmScale 0 -samplerHeaderWA -enablePreemptionR0Only -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -TotalGRFNum 256 -abortOnSpill 4 -enableBundleCR 3 -perfmodel -freqBasedSpillCost 8 -freqBasedSpillCostFunc 1 -boundsChecking -presched-ctrl 6 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -PVCSendWARWA -output -binary -dumpcommonisa -dumpcombinedcisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -partialInt64 -activeThreadsOnlyBarrier -hashmovs 4208518013 2685686631 -hashmovs1 0 13 "
//.instCount 10
//.RA type	LOCAL_ROUND_ROBIN_RA
//.git-hash 

//.declare BuiltInR0 (0)  rf=r size=64 type=ud align=32 words (r0.0) IsBuiltin
//.declare  (1)  rf=r size=64 type=ud align=32 words (r1.0) IsBuiltin
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0) IsBuiltin
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2) IsBuiltin
//.declare BuiltinSR0Dot1 (5)  rf=r size=4 type=ud align=2 words IsBuiltin
//.declare %null (10)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (13)  rf=r size=4 type=ud align=2 words (r1.4)
//.declare %local_id_y (14)  rf=r size=4 type=ud align=2 words (r1.5)
//.declare %local_size_x (15)  rf=r size=4 type=ud align=2 words (r1.0)
//.declare %local_size_y (16)  rf=r size=4 type=ud align=2 words (r1.1)
//.declare %group_id_x (17)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (18)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (19)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (20)  rf=r size=4 type=ud align=2 words (r1.2)
//.declare %group_count_y (21)  rf=r size=4 type=ud align=2 words (r1.3)
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
//.declare  (42)  rf=r size=64 type=ud align=32 words (r255.0)
//.declare r0 (43)  rf=r size=64 type=ud align=32 words (r0.0)
//.declare rtmp (44)  rf=r size=64 type=ud align=32 words (r255.0)

// .inputs
// +----------+----------+--------+----------+------------------+
// | id       | type     |  bytes | at       | from             |
// +----------+----------+--------+----------+------------------+
// +----------+----------+--------+----------+------------------+


// B000: Preds:{},  Succs:{B001}
cross_thread_prolog:
(W)     and (1|M0)               r255.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       //  ALU pipe: int; 
(W)     add (1|M0)               r255.0<1>:ud  r255.0<0;1,0>:ud  0x0:ud              {I@1}           //  R_SYM_ADDR_32: __INTEL_PATCH_CROSS_THREAD_OFFSET_OFF_R0; ALU pipe: int; 
// B001: Preds:{B000},  Succs:{}
// _main:
(W)     mov (16|M0)              r1.0<1>:ud    r0.0<1;1,0>:ud                   {Compacted}          //  ALU pipe: int; $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}    // $1
(W)     mov (16|M0)              r255.0<1>:f   r1.0<1;1,0>:f                    {Compacted,A@1}      //  ALU pipe: float; $2
(W)     send.gtwy (1|M0)         null     r255  null:0  0x0            0x02000010           {EOT,A@1,$0} // wr:1+0, rd:0; end of thread // $2
L80:
(W)     mov (16|M0)              null<1>:ud    0xFAD8E37D:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0xA0145367:ud                                         // 
(W)     mov (16|M0)              null<1>:ud    0x0:ud                                                // 
(W)     mov (16|M0)              null<1>:ud    0xD:ud                                                // 


//.BankConflicts: 0
//.ByteRMWs: 0
//


//.numALUInst: 9
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 1
//.allAtOneDistNum: 3
//.syncInstCount: 0
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 0
//.AfterReadTokenDepCount: 0

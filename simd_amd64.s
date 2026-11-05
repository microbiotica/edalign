//go:build amd64 && !purego

#include "textflag.h"

// The column kernels compute a run of n Myers blocks as one long bit-vector:
//
//	Xv  = Eq | Mv
//	Xh  = (((Eq & Pv) + Pv) ^ Pv) | Eq      (multi-word add, carry-in = hin < 0)
//	Ph  = Mv | ^(Xh | Pv)
//	Mh  = Pv & Xh
//	Score += Ph>>63 - Mh>>63                (per block)
//	Ph <<= 1, shifting in hin > 0           (multi-word shift)
//	Mh <<= 1, shifting in hin < 0           (multi-word shift)
//	Pv' = Mh | ^(Xv | Ph)
//	Mv' = Ph & Xv
//
// This is exactly calculateBlock applied block by block: the carry out of a block's add, and the bits
// shifted out of its Ph and Mh, are its hout < 0, hout < 0 and hout > 0 respectively, which calculateBlock
// feeds into the next block as hin.
//
// The carry between the 64-bit lanes of a vector is resolved with a carry-lookahead on the lane masks:
// g = lanes that generated a carry, p = lanes that propagate an incoming carry (sum is all ones), and
// the carries into each lane are ((g << 1 | cin) + p) ^ p, the bit above the top lane being the carry out.

// func myersColumnAVX512(p, m *word, score *int, eq *word, n int, hin int)
// Requires AVX512F.
TEXT ·myersColumnAVX512(SB), NOSPLIT, $0-48
	MOVQ p+0(FP), DI
	MOVQ m+8(FP), SI
	MOVQ score+16(FP), DX
	MOVQ eq+24(FP), R9
	MOVQ n+32(FP), BX
	MOVQ hin+40(FP), AX

	// R10 = carry in (hin < 0), R11 = Ph shift in (hin > 0)
	XORL R10, R10
	XORL R11, R11
	TESTQ AX, AX
	SETLT R10B
	SETGT R11B

	// Z14, Z15: Ph>>63 and Mh>>63 of the previous vector, only the top lane is used.
	VPBROADCASTQ R11, Z14
	VPBROADCASTQ R10, Z15
	// Z13: all ones
	VPTERNLOGQ $0xff, Z13, Z13, Z13

avx512Loop:
	// K1: lanes of this vector that are in range
	MOVL $0xff, R8
	CMPQ BX, $8
	JAE  avx512Full
	MOVQ BX, CX
	MOVL $1, R8
	SHLL CX, R8
	DECL R8

avx512Full:
	KMOVW R8, K1

	VMOVDQU64.Z (DI), K1, Z0 // Pv
	VMOVDQU64.Z (SI), K1, Z1 // Mv
	VMOVDQU64.Z (R9), K1, Z2 // Eq

	// Z3 = (Eq & Pv) + Pv, lane-wise
	VPANDQ Z2, Z0, Z3
	VPADDQ Z0, Z3, Z3

	// Propagate carries between lanes
	VPCMPUQ  $1, Z0, Z3, K2 // generate: sum < Pv
	VPCMPEQQ Z13, Z3, K3    // propagate: sum == all ones
	KMOVW    K2, R12
	KMOVW    K3, R13
	SHLQ     $1, R12
	ORQ      R10, R12
	ADDQ     R13, R12
	XORQ     R13, R12
	MOVQ     R12, R10
	SHRQ     $8, R10        // carry out of the top lane
	KMOVW    R12, K4
	VPSUBQ   Z13, Z3, K4, Z3 // +1 in lanes receiving a carry

	VPTERNLOGQ $0xbe, Z2, Z0, Z3 // Xh = (sum ^ Pv) | Eq
	VPORQ      Z1, Z2, Z4        // Xv = Eq | Mv
	VMOVDQA64  Z1, Z5
	VPTERNLOGQ $0xf1, Z0, Z3, Z5 // Ph = Mv | ^(Xh | Pv)
	VPANDQ     Z3, Z0, Z6        // Mh = Pv & Xh

	// Score += hout
	VPSRLQ          $63, Z5, Z7
	VPSRLQ          $63, Z6, Z8
	VMOVDQU64.Z     (DX), K1, Z9
	VPADDQ          Z7, Z9, Z9
	VPSUBQ          Z8, Z9, Z9
	VMOVDQU64       Z9, K1, (DX)

	// Ph <<= 1, Mh <<= 1 across lanes
	VALIGNQ   $7, Z14, Z7, Z10
	VALIGNQ   $7, Z15, Z8, Z11
	VMOVDQA64 Z7, Z14
	VMOVDQA64 Z8, Z15
	VPSLLQ    $1, Z5, Z5
	VPORQ     Z10, Z5, Z5
	VPSLLQ    $1, Z6, Z6
	VPORQ     Z11, Z6, Z6

	VMOVDQA64  Z6, Z0
	VPTERNLOGQ $0xf1, Z5, Z4, Z0 // Pv' = Mh | ^(Xv | Ph)
	VPANDQ     Z4, Z5, Z1        // Mv' = Ph & Xv

	VMOVDQU64 Z0, K1, (DI)
	VMOVDQU64 Z1, K1, (SI)

	ADDQ $64, DI
	ADDQ $64, SI
	ADDQ $64, DX
	ADDQ $64, R9
	SUBQ $8, BX
	JG   avx512Loop

	VZEROUPPER
	RET

// func myersColumnAVX2(p, m *word, score *int, eq *word, n int, hin int)
// Requires AVX2.
TEXT ·myersColumnAVX2(SB), NOSPLIT, $0-48
	MOVQ p+0(FP), DI
	MOVQ m+8(FP), SI
	MOVQ score+16(FP), DX
	MOVQ eq+24(FP), R9
	MOVQ n+32(FP), BX
	MOVQ hin+40(FP), AX

	// R10 = carry in (hin < 0), R11 = Ph shift in (hin > 0)
	XORL R10, R10
	XORL R11, R11
	TESTQ AX, AX
	SETLT R10B
	SETGT R11B

	// Y12, Y13: Ph>>63 and Mh>>63 of the previous vector, rotated up a lane, so only the bottom lane is used.
	VMOVQ R11, X12
	VMOVQ R10, X13
	// Y15: all ones, Y14: sign bits
	VPCMPEQQ Y15, Y15, Y15
	VPSLLQ   $63, Y15, Y14

	LEAQ carryMasks<>(SB), AX
	LEAQ laneMasks<>(SB), R14

	CMPQ BX, $4
	JL   avx2Tail

avx2Loop:
	VMOVDQU    (DI), Y0 // Pv
	VMOVDQU    (SI), Y1 // Mv
	VMOVDQU    (R9), Y2 // Eq

	// Y3 = (Eq & Pv) + Pv, lane-wise
	VPAND  Y2, Y0, Y3
	VPADDQ Y0, Y3, Y3

	// Propagate carries between lanes
	VPXOR     Y14, Y3, Y4
	VPXOR     Y14, Y0, Y5
	VPCMPGTQ  Y4, Y5, Y4 // generate: sum < Pv (unsigned)
	VPCMPEQQ  Y15, Y3, Y5 // propagate: sum == all ones
	VMOVMSKPD Y4, R12
	VMOVMSKPD Y5, R13
	SHLQ      $1, R12
	ORQ       R10, R12
	ADDQ      R13, R12
	XORQ      R13, R12
	MOVQ      R12, R10
	SHRQ      $4, R10 // carry out of the top lane
	ANDQ      $15, R12
	SHLQ      $5, R12
	VPSUBQ    (AX)(R12*1), Y3, Y3 // +1 in lanes receiving a carry

	VPXOR Y0, Y3, Y3  // sum ^ Pv
	VPOR  Y2, Y3, Y3  // Xh = (sum ^ Pv) | Eq
	VPOR  Y1, Y2, Y4  // Xv = Eq | Mv
	VPOR  Y0, Y3, Y5
	VPXOR Y15, Y5, Y5
	VPOR  Y1, Y5, Y5  // Ph = Mv | ^(Xh | Pv)
	VPAND Y3, Y0, Y6  // Mh = Pv & Xh

	// Score += hout
	VPSRLQ     $63, Y5, Y7
	VPSRLQ     $63, Y6, Y8
	VMOVDQU    (DX), Y9
	VPADDQ     Y7, Y9, Y9
	VPSUBQ     Y8, Y9, Y9
	VMOVDQU    Y9, (DX)

	// Ph <<= 1, Mh <<= 1 across lanes
	VPERMQ   $0x93, Y7, Y7
	VPERMQ   $0x93, Y8, Y8
	VPBLENDD $0x03, Y12, Y7, Y9
	VPBLENDD $0x03, Y13, Y8, Y10
	VMOVDQA  Y7, Y12
	VMOVDQA  Y8, Y13
	VPSLLQ   $1, Y5, Y5
	VPOR     Y9, Y5, Y5
	VPSLLQ   $1, Y6, Y6
	VPOR     Y10, Y6, Y6

	VPOR  Y5, Y4, Y0
	VPXOR Y15, Y0, Y0
	VPOR  Y6, Y0, Y0  // Pv' = Mh | ^(Xv | Ph)
	VPAND Y4, Y5, Y1  // Mv' = Ph & Xv

	VMOVDQU    Y0, (DI)
	VMOVDQU    Y1, (SI)

	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, DX
	ADDQ $32, R9
	SUBQ $4, BX
	CMPQ BX, $4
	JGE  avx2Loop

avx2Tail:
	// The remaining 0-3 blocks, with loads and stores masked to the blocks in range.
	TESTQ BX, BX
	JZ    avx2Done
	SHLQ    $5, BX
	VMOVDQU (R14)(BX*1), Y11

	VPMASKMOVQ (DI), Y11, Y0 // Pv
	VPMASKMOVQ (SI), Y11, Y1 // Mv
	VPMASKMOVQ (R9), Y11, Y2 // Eq

	// Y3 = (Eq & Pv) + Pv, lane-wise
	VPAND  Y2, Y0, Y3
	VPADDQ Y0, Y3, Y3

	// Propagate carries between lanes
	VPXOR     Y14, Y3, Y4
	VPXOR     Y14, Y0, Y5
	VPCMPGTQ  Y4, Y5, Y4 // generate: sum < Pv (unsigned)
	VPCMPEQQ  Y15, Y3, Y5 // propagate: sum == all ones
	VMOVMSKPD Y4, R12
	VMOVMSKPD Y5, R13
	SHLQ      $1, R12
	ORQ       R10, R12
	ADDQ      R13, R12
	XORQ      R13, R12
	MOVQ      R12, R10
	SHRQ      $4, R10 // carry out of the top lane
	ANDQ      $15, R12
	SHLQ      $5, R12
	VPSUBQ    (AX)(R12*1), Y3, Y3 // +1 in lanes receiving a carry

	VPXOR Y0, Y3, Y3  // sum ^ Pv
	VPOR  Y2, Y3, Y3  // Xh = (sum ^ Pv) | Eq
	VPOR  Y1, Y2, Y4  // Xv = Eq | Mv
	VPOR  Y0, Y3, Y5
	VPXOR Y15, Y5, Y5
	VPOR  Y1, Y5, Y5  // Ph = Mv | ^(Xh | Pv)
	VPAND Y3, Y0, Y6  // Mh = Pv & Xh

	// Score += hout
	VPSRLQ     $63, Y5, Y7
	VPSRLQ     $63, Y6, Y8
	VPMASKMOVQ (DX), Y11, Y9
	VPADDQ     Y7, Y9, Y9
	VPSUBQ     Y8, Y9, Y9
	VPMASKMOVQ Y9, Y11, (DX)

	// Ph <<= 1, Mh <<= 1 across lanes
	VPERMQ   $0x93, Y7, Y7
	VPERMQ   $0x93, Y8, Y8
	VPBLENDD $0x03, Y12, Y7, Y9
	VPBLENDD $0x03, Y13, Y8, Y10
	VMOVDQA  Y7, Y12
	VMOVDQA  Y8, Y13
	VPSLLQ   $1, Y5, Y5
	VPOR     Y9, Y5, Y5
	VPSLLQ   $1, Y6, Y6
	VPOR     Y10, Y6, Y6

	VPOR  Y5, Y4, Y0
	VPXOR Y15, Y0, Y0
	VPOR  Y6, Y0, Y0  // Pv' = Mh | ^(Xv | Ph)
	VPAND Y4, Y5, Y1  // Mv' = Ph & Xv

	VPMASKMOVQ Y0, Y11, (DI)
	VPMASKMOVQ Y1, Y11, (SI)


avx2Done:
	VZEROUPPER
	RET

// carryMasks[i] has all bits set in lane j when bit j of i is set.
DATA carryMasks<>+0(SB)/8, $0
DATA carryMasks<>+8(SB)/8, $0
DATA carryMasks<>+16(SB)/8, $0
DATA carryMasks<>+24(SB)/8, $0
DATA carryMasks<>+32(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+40(SB)/8, $0
DATA carryMasks<>+48(SB)/8, $0
DATA carryMasks<>+56(SB)/8, $0
DATA carryMasks<>+64(SB)/8, $0
DATA carryMasks<>+72(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+80(SB)/8, $0
DATA carryMasks<>+88(SB)/8, $0
DATA carryMasks<>+96(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+104(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+112(SB)/8, $0
DATA carryMasks<>+120(SB)/8, $0
DATA carryMasks<>+128(SB)/8, $0
DATA carryMasks<>+136(SB)/8, $0
DATA carryMasks<>+144(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+152(SB)/8, $0
DATA carryMasks<>+160(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+168(SB)/8, $0
DATA carryMasks<>+176(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+184(SB)/8, $0
DATA carryMasks<>+192(SB)/8, $0
DATA carryMasks<>+200(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+208(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+216(SB)/8, $0
DATA carryMasks<>+224(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+232(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+240(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+248(SB)/8, $0
DATA carryMasks<>+256(SB)/8, $0
DATA carryMasks<>+264(SB)/8, $0
DATA carryMasks<>+272(SB)/8, $0
DATA carryMasks<>+280(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+288(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+296(SB)/8, $0
DATA carryMasks<>+304(SB)/8, $0
DATA carryMasks<>+312(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+320(SB)/8, $0
DATA carryMasks<>+328(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+336(SB)/8, $0
DATA carryMasks<>+344(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+352(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+360(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+368(SB)/8, $0
DATA carryMasks<>+376(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+384(SB)/8, $0
DATA carryMasks<>+392(SB)/8, $0
DATA carryMasks<>+400(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+408(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+416(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+424(SB)/8, $0
DATA carryMasks<>+432(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+440(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+448(SB)/8, $0
DATA carryMasks<>+456(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+464(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+472(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+480(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+488(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+496(SB)/8, $0xffffffffffffffff
DATA carryMasks<>+504(SB)/8, $0xffffffffffffffff
GLOBL carryMasks<>(SB), RODATA|NOPTR, $512

// laneMasks[i] has all bits set in the first i lanes.
DATA laneMasks<>+0(SB)/8, $0
DATA laneMasks<>+8(SB)/8, $0
DATA laneMasks<>+16(SB)/8, $0
DATA laneMasks<>+24(SB)/8, $0
DATA laneMasks<>+32(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+40(SB)/8, $0
DATA laneMasks<>+48(SB)/8, $0
DATA laneMasks<>+56(SB)/8, $0
DATA laneMasks<>+64(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+72(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+80(SB)/8, $0
DATA laneMasks<>+88(SB)/8, $0
DATA laneMasks<>+96(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+104(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+112(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+120(SB)/8, $0
DATA laneMasks<>+128(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+136(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+144(SB)/8, $0xffffffffffffffff
DATA laneMasks<>+152(SB)/8, $0xffffffffffffffff
GLOBL laneMasks<>(SB), RODATA|NOPTR, $160

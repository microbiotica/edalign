//go:build arm64 && !purego

#include "textflag.h"

// myersColumnARM64 computes a run of n Myers blocks as one long bit-vector, as the amd64 kernels do
// (see simd_amd64.s):
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
// The add carries between blocks in the carry flag (ADCS), which is never touched by anything else in the
// loop. The shift takes the top bit of the previous block's Ph and Mh with EXTR. The only dependency chain
// between blocks is therefore one ADCS, so blocks are computed at the core's integer throughput.

// BLOCK computes Xh, Xv, Ph and Mh of one block, and adds its hout to sc.
// Consumes the carry flag as the carry in, and sets it to the carry out.
#define BLOCK(pv, mv, eq, sc, xh, xv, ph, mh) \
	AND  eq, pv, xh       \
	ADCS pv, xh, xh       \ // sum = (Eq & Pv) + Pv + carry
	EOR  pv, xh, xh       \
	ORR  eq, xh, xh       \ // Xh = (sum ^ Pv) | Eq
	ORR  eq, mv, xv       \ // Xv = Eq | Mv
	ORR  pv, xh, ph       \
	ORN  ph, mv, ph       \ // Ph = Mv | ^(Xh | Pv)
	AND  xh, pv, mh       \ // Mh = Pv & Xh
	ADD  ph>>63, sc, sc   \
	SUB  mh>>63, sc, sc     // Score += hout

// SHIFT shifts Ph and Mh up one bit, shifting in the top bits of the previous block's Ph and Mh,
// and computes the new Pv and Mv into pv and mv. Overwrites xh.
#define SHIFT(ph, mh, prevPh, prevMh, xv, xh, pv, mv) \
	EXTR $63, prevPh, ph, xh \ // Ph << 1
	EXTR $63, prevMh, mh, mv \ // Mh << 1
	ORR  xh, xv, pv          \
	ORN  pv, mv, pv          \ // Pv' = Mh | ^(Xv | Ph)
	AND  xv, xh, mv            // Mv' = Ph & Xv

// func myersColumnARM64(p, m *word, score *int, eq *word, n int, hin int)
TEXT ·myersColumnARM64(SB), NOSPLIT, $0-48
	MOVD p+0(FP), R0
	MOVD m+8(FP), R1
	MOVD score+16(FP), R2
	MOVD eq+24(FP), R3
	MOVD n+32(FP), R4
	MOVD hin+40(FP), R5

	// R20, R21: Ph and Mh of the previous block; only the top bit is used.
	// Initially the top bits are hin > 0 and hin < 0.
	CMP  $0, R5
	CSET GT, R20
	LSL  $63, R20, R20
	AND  $0x8000000000000000, R5, R21

	// Carry flag = hin < 0
	LSR $63, R5, R6
	CMP $1, R6

	// R5: number of block pairs
	LSR $1, R4, R5
	CBZ R5, tail

pairs:
	LDP (R0), (R8, R12)  // Pv
	LDP (R1), (R9, R13)  // Mv
	LDP.P 16(R3), (R10, R14) // Eq
	LDP (R2), (R11, R15) // Score

	BLOCK(R8, R9, R10, R11, R16, R17, R19, R22)
	SHIFT(R19, R22, R20, R21, R17, R16, R8, R9)
	BLOCK(R12, R13, R14, R15, R16, R17, R20, R21)
	SHIFT(R20, R21, R19, R22, R17, R16, R12, R13)

	STP.P (R11, R15), 16(R2)
	STP.P (R8, R12), 16(R0)
	STP.P (R9, R13), 16(R1)

	// SUB and CBNZ leave the carry flag intact for the next block.
	SUB  $1, R5, R5
	CBNZ R5, pairs

tail:
	TBZ $0, R4, done

	MOVD (R0), R8
	MOVD (R1), R9
	MOVD (R3), R10
	MOVD (R2), R11

	BLOCK(R8, R9, R10, R11, R16, R17, R19, R22)
	SHIFT(R19, R22, R20, R21, R17, R16, R8, R9)

	MOVD R11, (R2)
	MOVD R8, (R0)
	MOVD R9, (R1)

done:
	RET

//go:build arm64 && !purego

package edalign

//go:noescape
func myersColumnARM64(p, m *word, score *int, eq *word, n int, hin int)

func columnARM64(p, m []word, score []int, eq []word, hin int) {
	var n = len(p)
	if n == 0 {
		return
	}

	_, _, _ = m[n-1], score[n-1], eq[n-1]
	myersColumnARM64(&p[0], &m[0], &score[0], &eq[0], n, hin)
}

func init() {
	// The kernel only uses base ARMv8-A instructions, so it is supported on every arm64 CPU.
	//
	// A NEON kernel (two blocks per vector, with a carry-lookahead between lanes) was also tried. On Apple M4
	// it was 15-20% slower than this kernel wherever the kernel dominates: the lookahead adds enough vector
	// operations that the four SIMD pipes become the bottleneck, while here the carry is free in ADCS.
	simdKernels = append(simdKernels, namedKernel{"arm64", columnARM64})
	simdKernel = columnARM64
}

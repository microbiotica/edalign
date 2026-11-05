//go:build amd64 && !purego

package edalign

import "golang.org/x/sys/cpu"

//go:noescape
func myersColumnAVX512(p, m *word, score *int, eq *word, n int, hin int)

//go:noescape
func myersColumnAVX2(p, m *word, score *int, eq *word, n int, hin int)

func columnAVX512(p, m []word, score []int, eq []word, hin int) {
	var n = len(p)
	if n == 0 {
		return
	}

	_, _, _ = m[n-1], score[n-1], eq[n-1]
	myersColumnAVX512(&p[0], &m[0], &score[0], &eq[0], n, hin)
}

func columnAVX2(p, m []word, score []int, eq []word, hin int) {
	var n = len(p)
	if n == 0 {
		return
	}

	_, _, _ = m[n-1], score[n-1], eq[n-1]
	myersColumnAVX2(&p[0], &m[0], &score[0], &eq[0], n, hin)
}

func init() {
	if cpu.X86.HasAVX2 {
		simdKernels = append(simdKernels, namedKernel{"avx2", columnAVX2})
	}
	if cpu.X86.HasAVX512F {
		simdKernels = append(simdKernels, namedKernel{"avx512", columnAVX512})
	}

	// Prefer the widest supported kernel
	if len(simdKernels) > 0 {
		simdKernel = simdKernels[len(simdKernels)-1].kernel
	}
}

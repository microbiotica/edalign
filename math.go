package edalign

// word represents an architecture-specific CPU word
type word uint

const (
	wordSize        = 32 << (^word(0) >> 63)
	wordHighBitMask = word(1) << (wordSize - 1)
	wordAllOnes     = ^word(0)
)

func ceilDiv[T ~int](x, y T) T {
	if x%y > 0 {
		return x/y + 1
	}

	return x / y
}

func abs[T ~int](a T) T {
	if a >= 0 {
		return a
	}

	return -a
}

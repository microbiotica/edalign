package edalign

import (
	"unsafe"
)

type alignmentCode uint8

const (
	// Match indicates a match
	Match alignmentCode = iota
	// Insert indicates an insertion to target = deletion from query
	Insert
	// Delete indicates a deletion from target = insertion to query
	Delete
	// Mismatch indicates a mismatch
	Mismatch

	maxAlignmentCode
)

// cigarExtendedFormat represents the CIGAR extended format indices for an alignment type
var cigarExtendedFormat = [maxAlignmentCode]byte{'=', 'I', 'D', 'X'}

type Alignment []alignmentCode

// AppendCIGARString appends the alignment in CIGAR extended format to the input byte slice and returns the result.
func (aln Alignment) AppendCIGARString(cigar []byte) []byte {
	var (
		lastMove       byte // character of previous move, 0 if no previous move
		numOfSameMoves int
	)
	for i := 0; i <= len(aln); i++ {
		// If new sequence of same moves started
		if i == len(aln) || (cigarExtendedFormat[aln[i]] != lastMove && lastMove != 0) {
			// Write number of moves to cigar string
			if numOfSameMoves > 1 {
				var numDigits = 0
				for ; numOfSameMoves > 0; numOfSameMoves /= 10 {
					cigar = append(cigar, '0'+byte(numOfSameMoves%10))
					numDigits++
				}

				reverse(cigar[len(cigar)-numDigits:])
			}

			// Write code of move to cigar string
			cigar = append(cigar, lastMove)

			// If not at the end, start a new sequence of moves.
			if i < len(aln) {
				numOfSameMoves = 0
			}
		}

		if i < len(aln) {
			lastMove = cigarExtendedFormat[aln[i]]
			numOfSameMoves++
		}
	}

	return cigar
}

// String returns the alignment as string in CIGAR extended format.
func (aln Alignment) String() string {
	var cigar = aln.AppendCIGARString(nil)

	// nocopy string cast from rune slice
	return *(*string)(unsafe.Pointer(&cigar))
}

type Result struct {
	EditDistance int
	EndLocations []int

	// PercentIdentity is (n - EditDistance) / n * 100, where n is the length of the shorter of
	// query and target, clamped to [0, 100]. It is 0 if no alignment was found (EditDistance == -1).
	PercentIdentity float64

	// StartLocations reports the start locations of the alignment.
	// Only set when TaskLocation is enabled in the alignment Config.
	StartLocations []int

	// Alignment reports the alignment path.
	// Only set when TaskPath is enabled in the alignment Config.
	Alignment Alignment
}

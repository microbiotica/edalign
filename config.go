package edalign

type alignMode uint8

const (
	// ModeGlobal uses Needleman-Wunsch global alignment.
	// Equivalent to the NW mode of the original edlib.
	ModeGlobal alignMode = iota
	// ModeInfix uses hybrid Needleman-Wunsch infix alignment. Gaps before and after query are not penalized.
	// Equivalent to the HW mode of the original edlib.
	ModeInfix
	// ModePrefix uses semi-hybrid Needleman-Wunsch prefix alignment. Gap after query is not penalized.
	// Equivalent to the SHW mode of the original edlib.
	ModePrefix
)

type alignTask uint8

const (
	// TaskDistance finds edit distance and end locations of optimal alignment paths in target.
	TaskDistance alignTask = 0x00
	// TaskLocation finds edit distance and start and end locations of optimal alignment paths in target.
	TaskLocation = TaskDistance | 0x01
	// TaskPath finds edit distance, alignment path (and start and end locations of it in target).
	TaskPath = TaskDistance | TaskLocation | 0x10
)

type Config struct {
	// Set k to a non-negative value to tell alignment that edit distance is not larger than k.
	// Smaller k can significantly improve speed of computation.
	// If edit distance is larger than k, Align will set edit distance to -1.
	// Set k to negative value and Align will internally auto-adjust k until score is found.
	K int

	// Mode is the alignment mode.
	Mode alignMode

	// Task tells alignment what to calculate.
	// The less to calculate the faster it is.
	Task alignTask

	// Alphabet describes the available character alphabet of any alignment sequences.
	Alphabet *Alphabet
}

var (
	// IUPAC is an Alphabet that contains IUPAC (DNA) upper-case nucleotides.
	IUPAC *Alphabet
	PROT  *Alphabet
)

func init() {
	IUPAC, _ = NewAlphabet("ATCGRYSWKMBDHVN", true,
		// 2-base ambiguous
		AdditionalEquality{What: 'R', Equals: 'A'}, AdditionalEquality{What: 'R', Equals: 'G'},
		AdditionalEquality{What: 'Y', Equals: 'C'}, AdditionalEquality{What: 'Y', Equals: 'T'},
		AdditionalEquality{What: 'S', Equals: 'G'}, AdditionalEquality{What: 'S', Equals: 'C'},
		AdditionalEquality{What: 'W', Equals: 'A'}, AdditionalEquality{What: 'W', Equals: 'T'},
		AdditionalEquality{What: 'K', Equals: 'G'}, AdditionalEquality{What: 'K', Equals: 'T'},
		AdditionalEquality{What: 'M', Equals: 'A'}, AdditionalEquality{What: 'M', Equals: 'C'},

		// 3-base ambiguous
		AdditionalEquality{What: 'B', Equals: 'C'}, AdditionalEquality{What: 'B', Equals: 'G'}, AdditionalEquality{What: 'B', Equals: 'T'},
		AdditionalEquality{What: 'D', Equals: 'A'}, AdditionalEquality{What: 'D', Equals: 'G'}, AdditionalEquality{What: 'D', Equals: 'T'},
		AdditionalEquality{What: 'H', Equals: 'A'}, AdditionalEquality{What: 'H', Equals: 'C'}, AdditionalEquality{What: 'H', Equals: 'T'},
		AdditionalEquality{What: 'V', Equals: 'A'}, AdditionalEquality{What: 'V', Equals: 'C'}, AdditionalEquality{What: 'V', Equals: 'G'},

		// 3-base ambiguous also fit into 2-base
		AdditionalEquality{What: 'B', Equals: 'S'}, AdditionalEquality{What: 'B', Equals: 'K'}, AdditionalEquality{What: 'B', Equals: 'Y'},
		AdditionalEquality{What: 'D', Equals: 'R'}, AdditionalEquality{What: 'D', Equals: 'K'}, AdditionalEquality{What: 'D', Equals: 'W'},
		AdditionalEquality{What: 'H', Equals: 'M'}, AdditionalEquality{What: 'H', Equals: 'Y'}, AdditionalEquality{What: 'H', Equals: 'W'},
		AdditionalEquality{What: 'V', Equals: 'M'}, AdditionalEquality{What: 'V', Equals: 'S'}, AdditionalEquality{What: 'V', Equals: 'R'},

		// N ambiguous matches everything
		AdditionalEquality{What: 'N', Equals: 'A'},
		AdditionalEquality{What: 'N', Equals: 'T'},
		AdditionalEquality{What: 'N', Equals: 'C'},
		AdditionalEquality{What: 'N', Equals: 'G'},

		AdditionalEquality{What: 'N', Equals: 'R'},
		AdditionalEquality{What: 'N', Equals: 'Y'},
		AdditionalEquality{What: 'N', Equals: 'S'},
		AdditionalEquality{What: 'N', Equals: 'W'},
		AdditionalEquality{What: 'N', Equals: 'K'},
		AdditionalEquality{What: 'N', Equals: 'M'},

		AdditionalEquality{What: 'N', Equals: 'B'},
		AdditionalEquality{What: 'N', Equals: 'D'},
		AdditionalEquality{What: 'N', Equals: 'H'},
		AdditionalEquality{What: 'N', Equals: 'V'},
	)
	// Characters outside the alphabet (gaps, RNA 'U', etc.) are treated as 'N', which matches everything.
	IUPAC.unknown = IUPAC.index['N']

	PROT, _ = NewAlphabet("*LMIVKRFYCAGSTPWEDNQH", true)
}

// DefaultConfig is the default configuration for alignment.
// It uses Infix alignment of IUPAC ambiguous sequences, extracting only the edit distance.
func DefaultConfig() *Config {
	return &Config{
		K:        -1,
		Mode:     ModeInfix,
		Task:     TaskDistance,
		Alphabet: IUPAC,
	}
}

// ProteinConfig is the config for 21-character protein alphabet (20AAs + '*' as stop)
// It uses Infix alignment, extracting only the edit distance.
func ProteinConfig() *Config {
	return &Config{
		K:        -1,
		Mode:     ModeInfix,
		Task:     TaskDistance,
		Alphabet: PROT,
	}
}

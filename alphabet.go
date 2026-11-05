package edalign

import (
	"errors"
	"fmt"
	"strings"
)

var (
	ErrCharacterNotInRange   = errors.New("edalign: character not in ASCII range (0-255)")
	ErrEqualityNotInAlphabet = errors.New("edalign: additional equality not in alphabet")
)

// A Sequence is an input sequence transformed into the target Alphabet.
type Sequence []uint8

func toUpper(c byte) byte {
	if 'a' <= c && c <= 'z' {
		c -= 'a' - 'A'
	}
	return c
}

// AdditionalEquality describes additional equalities between characters in an Alphabet.
// Both What and Equals must be present in the Alphabet.
type AdditionalEquality struct {
	// What is the source character of the sequence.
	What byte
	// Equals is the character that is considered equivalent.
	Equals byte
}

func NewAlphabet(characters string, forceUpperCase bool, additional ...AdditionalEquality) (ab *Alphabet, err error) {
	ab = &Alphabet{
		forceUpperCase: forceUpperCase,
	}

	for _, c := range ([]byte)(characters) {
		if int(c) >= len(ab.contains) {
			// character is in the invalid ASCII character range (0-255)
			err = ErrCharacterNotInRange
			return
		}

		if forceUpperCase {
			c = toUpper(c)
		}

		// Alphabet doesn't currently contain character c, so add it
		if !ab.contains[c] {
			ab.contains[c] = true
			ab.index[c] = uint8(len(ab.characters))
			ab.characters = append(ab.characters, c)
		}
	}

	// Build the equality definition
	for i := 0; i < len(ab.characters); i++ {
		for j := 0; j < len(ab.characters); j++ {
			ab.equalityDefinition[i][j] = i == j
		}
	}

	// Assign additional equalities
	for _, eq := range additional {
		if !ab.contains[eq.What] || !ab.contains[eq.Equals] {
			err = ErrEqualityNotInAlphabet
			return
		}

		var (
			ft = ab.index[eq.What]
			st = ab.index[eq.Equals]
		)

		ab.equalityDefinition[ft][st] = true
		ab.equalityDefinition[st][ft] = true
	}

	ab.equalTo = ab.buildEqualTo()

	return
}

// buildEqualTo lists, for each character in the Alphabet, every character that is Equal to it.
func (ab *Alphabet) buildEqualTo() [][]uint8 {
	var equalTo = make([][]uint8, len(ab.characters))

	for symbol := range equalTo {
		for c := range ab.characters {
			if ab.Equal(uint8(c), uint8(symbol)) {
				equalTo[symbol] = append(equalTo[symbol], uint8(c))
			}
		}
	}

	return equalTo
}

type Alphabet struct {
	forceUpperCase bool
	// characters maps the Alphabet positions back to their characters.
	characters []byte
	// contains tracks if ASCII character by index exists within the alphabet.
	contains [256]bool
	// index is the index of character in the alphabet
	index [256]uint8

	// equalityDefinition describes additional equalities of characters in the Alphabet
	equalityDefinition [256][256]bool

	// equalTo[s] lists the characters c for which Equal(c, s) is true.
	equalTo [][]uint8

	// unknown is the index that characters not in the Alphabet are transformed to.
	// It is 0 (the first character) unless set otherwise, e.g. 'N' for IUPAC.
	unknown uint8
}

func (ab *Alphabet) Len() int {
	return len(ab.characters)
}

// Equal returns true if two characters in the Alphabet are equal according to the Alphabet's equality definition
func (ab *Alphabet) Equal(a, b uint8) bool {
	return ab.equalityDefinition[a][b]
}

// TransformBytes transforms input into a Sequence of this Alphabet.
//
// Characters that are not in the Alphabet are mapped to the Alphabet's unknown character: 'N' for IUPAC,
// which matches every base, and the first character of the Alphabet otherwise (e.g. '*' for PROT).
// Use TransformBytesStrict to reject such input instead.
func (ab *Alphabet) TransformBytes(input []byte) (transformed Sequence) {
	transformed = make(Sequence, len(input))

	for i, c := range input {
		if ab.forceUpperCase {
			c = toUpper(c)
		}

		if !ab.contains[c] {
			transformed[i] = ab.unknown
			continue
		}

		// Set the transformed sequence to the mapped alphabet character
		transformed[i] = ab.index[c]
	}

	return
}

// Transform transforms input into a Sequence of this Alphabet.
// Characters not in the Alphabet are handled as described for TransformBytes.
func (ab *Alphabet) Transform(input string) (transformed Sequence) {
	return ab.TransformBytes(([]byte)(input))
}

// UnknownCharacterError reports a character in the input that is not in the Alphabet.
type UnknownCharacterError struct {
	// Character is the input character that was not found.
	Character byte
	// Position is the index of Character in the input.
	Position int
}

func (e *UnknownCharacterError) Error() string {
	return fmt.Sprintf("edalign: character %q at position %d is not in the alphabet", e.Character, e.Position)
}

// TransformBytesStrict is like TransformBytes, but returns an *UnknownCharacterError for the first
// character in input that is not in the Alphabet.
func (ab *Alphabet) TransformBytesStrict(input []byte) (Sequence, error) {
	var transformed = make(Sequence, len(input))

	for i, c := range input {
		if ab.forceUpperCase {
			c = toUpper(c)
		}

		if !ab.contains[c] {
			return nil, &UnknownCharacterError{Character: input[i], Position: i}
		}

		transformed[i] = ab.index[c]
	}

	return transformed, nil
}

// TransformStrict is like Transform, but returns an *UnknownCharacterError for the first
// character in input that is not in the Alphabet.
func (ab *Alphabet) TransformStrict(input string) (Sequence, error) {
	return ab.TransformBytesStrict(([]byte)(input))
}

func (ab *Alphabet) Character(i uint8) byte {
	return ab.characters[i]
}

// Decode reverses an input Sequence built with this alphabet back to the original string.
// If forceUpperCase is enabled in this alphabet then the returned string may not be directly equal to the original input.
func (ab *Alphabet) Decode(seq Sequence) string {
	var b strings.Builder
	b.Grow(len(seq))

	for _, c := range seq {
		b.WriteByte(ab.Character(c))
	}

	return b.String()
}

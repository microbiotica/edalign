package edalign

import (
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// validAlignment reports why r.Alignment is not a valid alignment of query against
// target[StartLocations[0]:EndLocations[0]+1] with r.EditDistance, or "" if it is valid.
func validAlignment(query, target Sequence, ab *Alphabet, r *Result) string {
	var qi, ti, edits = 0, r.StartLocations[0], 0

	for _, op := range r.Alignment {
		switch op {
		case Match:
			if qi >= len(query) || ti >= len(target) || !ab.Equal(query[qi], target[ti]) {
				return "bad match"
			}
			qi++
			ti++
		case Mismatch:
			if qi >= len(query) || ti >= len(target) || ab.Equal(query[qi], target[ti]) {
				return "bad mismatch"
			}
			qi++
			ti++
			edits++
		case Insert:
			qi++
			edits++
		case Delete:
			ti++
			edits++
		}
	}

	switch {
	case qi != len(query):
		return "alignment does not consume the whole query"
	case ti != r.EndLocations[0]+1:
		return "alignment does not end at EndLocations[0]"
	case edits != r.EditDistance:
		return "alignment edit count differs from EditDistance"
	}

	return ""
}

// TestAlignPathValid checks TaskPath alignments are internally consistent, including sizes that use
// Hirschberg's algorithm (BUGS.md #1, #2).
func TestAlignPathValid(t *testing.T) {
	var (
		rng   = rand.New(rand.NewSource(1))
		cases []alignCase
	)

	// An identical copy just above the Hirschberg threshold.
	var identical = randomSequence(rng, IUPAC, 3000, 4)
	cases = append(cases, alignCase{"identical3000", identical, append(Sequence(nil), identical...), nil})

	for i := range 24 {
		var (
			length = []int{500, 1500, 3000, 4500, 6500, 10000}[i%6]
			query  = randomSequence(rng, IUPAC, length, 4)
			target = mutate(rng, IUPAC, query, []float64{0.001, 0.01, 0.05, 0.3}[i%4], 4)
		)
		if i%3 == 0 {
			target = append(append(randomSequence(rng, IUPAC, 300, 4), target...), randomSequence(rng, IUPAC, 300, 4)...)
		}
		cases = append(cases, alignCase{fmt.Sprintf("random%d/q%d/t%d", i, len(query), len(target)), query, target, nil})
	}

	if testing.Short() {
		cases = cases[:7]
	}

	for _, impl := range benchImplementations() {
		t.Run(impl.name, func(t *testing.T) {
			for _, tc := range cases {
				for _, mode := range []alignMode{ModeGlobal, ModeInfix, ModePrefix} {
					var (
						config = &Config{K: -1, Mode: mode, Task: TaskPath, Alphabet: IUPAC}
						out    alignOutcome
					)
					withKernel(impl.kernel, func() { out = alignCapture(tc.query, tc.target, config) })

					if out.Panic != "" || out.Err != nil {
						t.Fatalf("%s/mode%d: panic %q, err %v", tc.name, mode, out.Panic, out.Err)
					}
					if reason := validAlignment(tc.query, tc.target, IUPAC, out.Result); reason != "" {
						t.Fatalf("%s/mode%d: %s (distance %d, alignment length %d)", tc.name, mode, reason, out.Result.EditDistance, len(out.Result.Alignment))
					}
				}
			}
		})
	}
}

// TestAlignPathNoAlignmentWithinK checks TaskPath with a fixed K does not panic when there is no
// alignment within K (BUGS.md #4).
func TestAlignPathNoAlignmentWithinK(t *testing.T) {
	var cfg = &Config{K: 1, Mode: ModeInfix, Task: TaskPath, Alphabet: IUPAC}

	result, err := Align(IUPAC.Transform("AAAAAA"), IUPAC.Transform("CCCCCC"), cfg)
	assert.NoError(t, err)
	assert.Equal(t, -1, result.EditDistance)
	assert.Nil(t, result.Alignment)
	assert.Equal(t, float64(0), result.PercentIdentity)
}

// TestPercentIdentityRange checks PercentIdentity is within [0, 100] (BUGS.md #3).
func TestPercentIdentityRange(t *testing.T) {
	t.Run("NoAlignment", func(t *testing.T) {
		var cfg = &Config{K: 1, Mode: ModeInfix, Task: TaskDistance, Alphabet: IUPAC}

		result, err := Align(IUPAC.Transform("AAAAAA"), IUPAC.Transform("CCCCCC"), cfg)
		assert.NoError(t, err)
		assert.Equal(t, -1, result.EditDistance)
		assert.Equal(t, float64(0), result.PercentIdentity)
	})

	t.Run("DistanceAboveShorterLength", func(t *testing.T) {
		var cfg = &Config{K: -1, Mode: ModeGlobal, Task: TaskDistance, Alphabet: IUPAC}

		result, err := Align(IUPAC.Transform("A"), IUPAC.Transform("CCCC"), cfg)
		assert.NoError(t, err)
		assert.Equal(t, 4, result.EditDistance)
		assert.Equal(t, float64(0), result.PercentIdentity)
	})

	t.Run("Partial", func(t *testing.T) {
		var cfg = &Config{K: -1, Mode: ModeGlobal, Task: TaskDistance, Alphabet: IUPAC}

		result, err := Align(IUPAC.Transform("ACGT"), IUPAC.Transform("ACGA"), cfg)
		assert.NoError(t, err)
		assert.Equal(t, 1, result.EditDistance)
		assert.Equal(t, float64(75), result.PercentIdentity)
	})
}

// TestTransformUnknown checks IUPAC maps unknown characters to 'N' rather than 'A' (BUGS.md #6).
func TestTransformUnknown(t *testing.T) {
	assert.Equal(t, "GNNNNNN", IUPAC.Decode(IUPAC.Transform("G-X.U*u")))

	result, err := Align(IUPAC.Transform("CC"), IUPAC.Transform("G-X"), DefaultConfig())
	assert.NoError(t, err)
	assert.Equal(t, 0, result.EditDistance) // N matches C

	result, err = Align(IUPAC.Transform("AA"), IUPAC.Transform("GU"), &Config{K: -1, Mode: ModeGlobal, Alphabet: IUPAC})
	assert.NoError(t, err)
	assert.Equal(t, 1, result.EditDistance)
}

// TestTransformStrict checks unknown characters are reported rather than silently mapped (BUGS.md #6).
func TestTransformStrict(t *testing.T) {
	seq, err := IUPAC.TransformStrict("acgtn")
	assert.NoError(t, err)
	assert.Equal(t, IUPAC.Transform("ACGTN"), seq)

	_, err = IUPAC.TransformStrict("AC-GT")
	var unknown *UnknownCharacterError
	if assert.True(t, errors.As(err, &unknown)) {
		assert.Equal(t, byte('-'), unknown.Character)
		assert.Equal(t, 2, unknown.Position)
	}

	_, err = PROT.TransformStrict("MKXX")
	if assert.True(t, errors.As(err, &unknown)) {
		assert.Equal(t, byte('X'), unknown.Character)
		assert.Equal(t, 2, unknown.Position)
	}
}

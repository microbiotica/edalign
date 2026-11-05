package edalign

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

// withKernel runs fn with kernel selected as the SIMD kernel (nil selects the portable implementation).
func withKernel(kernel columnKernel, fn func()) {
	var previous = simdKernel
	simdKernel = kernel
	defer func() { simdKernel = previous }()

	fn()
}

func requireSIMD(t testing.TB) {
	if len(simdKernels) == 0 {
		t.Skip("no SIMD kernel supported on this platform")
	}
}

func TestColumnKernels(t *testing.T) {
	requireSIMD(t)

	var rng = rand.New(rand.NewSource(1))

	for _, nk := range simdKernels {
		t.Run(nk.name, func(t *testing.T) {
			for iter := range 20000 {
				var (
					n   = 1 + rng.Intn(40)
					hin = rng.Intn(3) - 1

					p, m, eq = make([]word, n), make([]word, n), make([]word, n)
					score    = make([]int, n)
				)

				for i := range n {
					// Bias towards runs of set/unset bits so carries propagate across lanes.
					switch rng.Intn(4) {
					case 0:
						p[i] = wordAllOnes
					case 1:
						p[i] = 0
					default:
						p[i] = word(rng.Uint64())
					}
					m[i] = word(rng.Uint64()) &^ p[i] // vertical deltas are never both +1 and -1
					switch rng.Intn(3) {
					case 0:
						eq[i] = word(rng.Uint64())
					default:
						eq[i] = word(rng.Uint64())
					}
					score[i] = rng.Intn(1000)
				}

				var (
					wantP, wantM = append([]word(nil), p...), append([]word(nil), m...)
					wantScore    = append([]int(nil), score...)
					h            = hin
				)
				for i := range n {
					wantP[i], wantM[i], h = calculateBlock(wantP[i], wantM[i], eq[i], h)
					wantScore[i] += h
				}

				// Guard values just outside the run must be untouched.
				var (
					pg, mg = append(append([]word{7}, p...), 7), append(append([]word{7}, m...), 7)
					sg     = append(append([]int{7}, score...), 7)
					eqg    = append(append([]word{7}, eq...), 7)
				)
				nk.kernel(pg[1:n+1], mg[1:n+1], sg[1:n+1], eqg[1:n+1], hin)

				if !reflect.DeepEqual(pg[1:n+1], wantP) || !reflect.DeepEqual(mg[1:n+1], wantM) || !reflect.DeepEqual(sg[1:n+1], wantScore) {
					t.Fatalf("iteration %d (n=%d, hin=%d): kernel output differs from calculateBlock", iter, n, hin)
				}
				if pg[0] != 7 || pg[n+1] != 7 || mg[0] != 7 || mg[n+1] != 7 || sg[0] != 7 || sg[n+1] != 7 {
					t.Fatalf("iteration %d (n=%d): kernel wrote outside of its blocks", iter, n)
				}
			}
		})
	}
}

func randomSequence(rng *rand.Rand, alphabet *Alphabet, length int, symbols int) Sequence {
	var seq = make(Sequence, length)
	for i := range seq {
		seq[i] = uint8(rng.Intn(min(symbols, alphabet.Len())))
	}

	return seq
}

// mutate returns a copy of seq with roughly rate*len(seq) random substitutions, insertions and deletions.
func mutate(rng *rand.Rand, alphabet *Alphabet, seq Sequence, rate float64, symbols int) Sequence {
	var out = make(Sequence, 0, len(seq)+len(seq)/4)
	for _, c := range seq {
		if rng.Float64() >= rate {
			out = append(out, c)
			continue
		}

		switch rng.Intn(3) {
		case 0: // substitution
			out = append(out, uint8(rng.Intn(min(symbols, alphabet.Len()))))
		case 1: // insertion
			out = append(out, c, uint8(rng.Intn(min(symbols, alphabet.Len()))))
		case 2: // deletion
		}
	}

	return out
}

func TestBuildPeqFast(t *testing.T) {
	var rng = rand.New(rand.NewSource(2))

	for _, alphabet := range []*Alphabet{IUPAC, PROT} {
		for length := 0; length <= 300; length++ {
			var query = randomSequence(rng, alphabet, length, alphabet.Len())

			var want Peq
			withKernel(nil, func() { want = BuildPeq(query, alphabet) })

			if got := buildPeqFast(query, alphabet); !reflect.DeepEqual(got, want) {
				t.Fatalf("length %d: buildPeqFast differs from BuildPeq", length)
			}
		}
	}
}

// alignOutcome captures the result of Align, including any panic, so implementations can be compared exactly.
type alignOutcome struct {
	Result *Result
	Err    error
	Panic  string
}

func alignCapture(query, target Sequence, config *Config) (outcome alignOutcome) {
	defer func() {
		if r := recover(); r != nil {
			outcome.Panic = fmt.Sprint(r)
		}
	}()

	outcome.Result, outcome.Err = Align(query, target, config)
	return
}

type alignCase struct {
	name          string
	query, target Sequence
	config        *Config
}

func generateAlignCases(rng *rand.Rand, count int) []alignCase {
	var (
		modes  = []alignMode{ModeGlobal, ModeInfix, ModePrefix}
		tasks  = []alignTask{TaskDistance, TaskLocation, TaskPath}
		cases  []alignCase
		ab     *Alphabet
		symbol int
	)

	for i := range count {
		if rng.Intn(4) == 0 {
			ab, symbol = PROT, PROT.Len()
		} else {
			// Mostly unambiguous bases, sometimes with ambiguity codes.
			ab, symbol = IUPAC, 4
			if rng.Intn(3) == 0 {
				symbol = IUPAC.Len()
			}
		}

		var queryLength int
		switch rng.Intn(4) {
		case 0:
			queryLength = 1 + rng.Intn(64)
		case 1:
			queryLength = 64 * (1 + rng.Intn(12)) // exact multiples of the word size, no padding
		default:
			queryLength = 1 + rng.Intn(900)
		}

		var (
			query  = randomSequence(rng, ab, queryLength, symbol)
			target Sequence
		)

		switch rng.Intn(5) {
		case 0: // unrelated
			target = randomSequence(rng, ab, 1+rng.Intn(1200), symbol)
		case 1: // query embedded in a longer target
			var rate = []float64{0, 0.01, 0.05, 0.2}[rng.Intn(4)]
			target = append(randomSequence(rng, ab, rng.Intn(200), symbol), mutate(rng, ab, query, rate, symbol)...)
			target = append(target, randomSequence(rng, ab, rng.Intn(200), symbol)...)
		default: // mutated copy of query
			var rate = []float64{0, 0.01, 0.05, 0.2, 0.5}[rng.Intn(5)]
			target = mutate(rng, ab, query, rate, symbol)
		}

		if len(target) == 0 {
			target = randomSequence(rng, ab, 1, symbol)
		}

		var config = &Config{
			K:        -1,
			Mode:     modes[rng.Intn(len(modes))],
			Task:     tasks[rng.Intn(len(tasks))],
			Alphabet: ab,
		}
		if rng.Intn(3) == 0 {
			config.K = rng.Intn(queryLength + 10)
		}

		cases = append(cases, alignCase{
			name:   fmt.Sprintf("case%d/q%d/t%d/mode%d/task%d/k%d", i, len(query), len(target), config.Mode, config.Task, config.K),
			query:  query,
			target: target,
			config: config,
		})
	}

	return cases
}

func TestAlignSIMDMatchesGeneric(t *testing.T) {
	requireSIMD(t)

	var (
		rng   = rand.New(rand.NewSource(3))
		cases = generateAlignCases(rng, 3000)
	)

	// Long sequences so alignment uses Hirschberg's algorithm, and strong band reduction kicks in.
	for i := range 12 {
		var (
			query  = randomSequence(rng, IUPAC, 1000+rng.Intn(3000), 4)
			target = mutate(rng, IUPAC, query, []float64{0.001, 0.02, 0.1}[i%3], 4)
		)
		if i%2 == 0 {
			target = append(append(randomSequence(rng, IUPAC, 500, 4), target...), randomSequence(rng, IUPAC, 500, 4)...)
		}

		for _, mode := range []alignMode{ModeGlobal, ModeInfix, ModePrefix} {
			cases = append(cases, alignCase{
				name:   fmt.Sprintf("long%d/q%d/t%d/mode%d", i, len(query), len(target), mode),
				query:  query,
				target: target,
				config: &Config{K: -1, Mode: mode, Task: TaskPath, Alphabet: IUPAC},
			})
		}
	}

	for _, nk := range simdKernels {
		t.Run(nk.name, func(t *testing.T) {
			for _, tc := range cases {
				var want, got alignOutcome
				withKernel(nil, func() { want = alignCapture(tc.query, tc.target, tc.config) })
				withKernel(nk.kernel, func() { got = alignCapture(tc.query, tc.target, tc.config) })

				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s: SIMD result differs\n got: %+v %+v\nwant: %+v %+v", tc.name, got, got.Result, want, want.Result)
				}
			}
		})
	}
}

func FuzzAlignSIMDMatchesGeneric(f *testing.F) {
	requireSIMD(f)

	f.Add([]byte("ACT"), []byte("CGACTGAC"), uint8(1), uint8(2), int16(-1))
	f.Add([]byte("NNNACGT"), []byte("ATCGATTACA"), uint8(0), uint8(1), int16(3))

	f.Fuzz(func(t *testing.T, query, target []byte, mode, task uint8, k int16) {
		var config = &Config{
			K:        int(k),
			Mode:     alignMode(mode % 3),
			Task:     []alignTask{TaskDistance, TaskLocation, TaskPath}[task%3],
			Alphabet: IUPAC,
		}
		var q, tg = IUPAC.TransformBytes(query), IUPAC.TransformBytes(target)

		var want alignOutcome
		withKernel(nil, func() { want = alignCapture(q, tg, config) })

		for _, nk := range simdKernels {
			var got alignOutcome
			withKernel(nk.kernel, func() { got = alignCapture(q, tg, config) })

			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s: SIMD result differs\n got: %+v %+v\nwant: %+v %+v", nk.name, got, got.Result, want, want.Result)
			}
		}
	})
}

type benchScenario struct {
	name          string
	query, target Sequence
	config        *Config
}

func benchScenarios() []benchScenario {
	var rng = rand.New(rand.NewSource(4))

	var (
		// ~16S rRNA gene: 1.5kb, 0.5% divergence, with flanking sequence
		rRNA       = randomSequence(rng, IUPAC, 1500, 4)
		rRNATarget = append(append(randomSequence(rng, IUPAC, 20, 4), mutate(rng, IUPAC, rRNA, 0.005, 4)...), randomSequence(rng, IUPAC, 20, 4)...)
		// Divergent sequences: 1.5kb, 15% divergence
		divergentTarget = mutate(rng, IUPAC, rRNA, 0.15, 4)
		// Long sequences: 10kb, 1% divergence
		long       = randomSequence(rng, IUPAC, 10000, 4)
		longTarget = mutate(rng, IUPAC, long, 0.01, 4)
		// Protein: 500aa, 5% divergence
		protein       = randomSequence(rng, PROT, 500, PROT.Len())
		proteinTarget = mutate(rng, PROT, protein, 0.05, PROT.Len())
		// Short read: 150bp within a 1.5kb reference
		read = mutate(rng, IUPAC, rRNA[700:850], 0.01, 4)
	)

	var config = func(mode alignMode, task alignTask, ab *Alphabet) *Config {
		return &Config{K: -1, Mode: mode, Task: task, Alphabet: ab}
	}

	return []benchScenario{
		{"16S/infix/distance", rRNA, rRNATarget, config(ModeInfix, TaskDistance, IUPAC)},
		{"16S/infix/path", rRNA, rRNATarget, config(ModeInfix, TaskPath, IUPAC)},
		{"16S/global/distance", rRNA, rRNATarget, config(ModeGlobal, TaskDistance, IUPAC)},
		{"16S-divergent/infix/distance", rRNA, divergentTarget, config(ModeInfix, TaskDistance, IUPAC)},
		{"16S-divergent/global/path", rRNA, divergentTarget, config(ModeGlobal, TaskPath, IUPAC)},
		{"10kb/global/distance", long, longTarget, config(ModeGlobal, TaskDistance, IUPAC)},
		{"10kb/infix/distance", long, longTarget, config(ModeInfix, TaskDistance, IUPAC)},
		{"protein/infix/distance", protein, proteinTarget, config(ModeInfix, TaskDistance, PROT)},
		{"read150/infix/location", read, rRNA, config(ModeInfix, TaskLocation, IUPAC)},
	}
}

func benchImplementations() []namedKernel {
	return append([]namedKernel{{"generic", nil}}, simdKernels...)
}

// BenchmarkAlignImplementations compares Align across the portable implementation and each supported SIMD kernel.
func BenchmarkAlignImplementations(b *testing.B) {
	for _, sc := range benchScenarios() {
		for _, impl := range benchImplementations() {
			b.Run(sc.name+"/"+impl.name, func(b *testing.B) {
				withKernel(impl.kernel, func() {
					b.SetBytes(int64(len(sc.target)))
					for i := 0; i < b.N; i++ {
						_, _ = Align(sc.query, sc.target, sc.config)
					}
				})
			})
		}
	}
}

// BenchmarkAlignPEQImplementations is BenchmarkAlignImplementations with the query peq built ahead of time.
func BenchmarkAlignPEQImplementations(b *testing.B) {
	for _, sc := range benchScenarios() {
		var peq Peq
		withKernel(nil, func() { peq = BuildPeq(sc.query, sc.config.Alphabet) })

		for _, impl := range benchImplementations() {
			b.Run(sc.name+"/"+impl.name, func(b *testing.B) {
				withKernel(impl.kernel, func() {
					b.SetBytes(int64(len(sc.target)))
					for i := 0; i < b.N; i++ {
						_, _ = AlignPEQ(sc.query, sc.target, peq, sc.config)
					}
				})
			})
		}
	}
}

func BenchmarkBuildPeqImplementations(b *testing.B) {
	var rng = rand.New(rand.NewSource(5))

	for _, ab := range []*Alphabet{IUPAC, PROT} {
		var query = randomSequence(rng, ab, 1500, ab.Len())

		b.Run(fmt.Sprintf("%d-symbols/generic", ab.Len()), func(b *testing.B) {
			withKernel(nil, func() {
				for i := 0; i < b.N; i++ {
					_ = BuildPeq(query, ab)
				}
			})
		})
		b.Run(fmt.Sprintf("%d-symbols/fast", ab.Len()), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = buildPeqFast(query, ab)
			}
		})
	}
}

// TestBandCellChecks compares the early-exit band checks with the loops in myersCalcEditDistanceGlobal.
func TestBandCellChecks(t *testing.T) {
	var rng = rand.New(rand.NewSource(6))

	for range 200000 {
		var (
			bc       = newBlockColumn(4)
			b        = rng.Intn(4)
			W        = rng.Intn(wordSize)
			numCells = numCellsInBlock(b, 4, W)
			k        = rng.Intn(200)
			bound    = rng.Intn(600) - 300
		)

		bc.P[b] = word(rng.Uint64())
		bc.M[b] = word(rng.Uint64()) &^ bc.P[b]
		bc.Score[b] = rng.Intn(300)

		var (
			scores               = getBlockCellValues(bc.block(b))
			wantAbove, wantBelow bool
			rAbove, rBelow       = b*wordSize + numCells - 1, b*wordSize + numCells - 1
		)
		for i := wordSize - numCells; i < wordSize; i++ {
			// Conditions as written in myersCalcEditDistanceGlobal, with the bound folded in.
			if scores[i] <= k && rAbove <= bound-scores[i] {
				wantAbove = true
			}
			if scores[i] <= k && rBelow >= scores[i]+bound {
				wantBelow = true
			}
			rAbove--
			rBelow--
		}

		if got := bc.hasCellAboveDiagonal(b, numCells, k, bound); got != wantAbove {
			t.Fatalf("hasCellAboveDiagonal(b=%d, numCells=%d, k=%d, bound=%d) = %v, want %v", b, numCells, k, bound, got, wantAbove)
		}
		if got := bc.hasCellBelowDiagonal(b, numCells, k, bound); got != wantBelow {
			t.Fatalf("hasCellBelowDiagonal(b=%d, numCells=%d, k=%d, bound=%d) = %v, want %v", b, numCells, k, bound, got, wantBelow)
		}
	}
}

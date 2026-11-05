package edalign

import "math/bits"

// columnKernel advances a contiguous run of Myers blocks by one column (one target character).
//
// p, m and score hold the vertical positive/negative delta vectors and bottom-cell score of each block
// and are updated in place. eq holds the peq word of each block for the current target character.
// hin is the horizontal delta (-1, 0 or 1) entering the first block.
//
// A kernel must produce exactly the same values as calling calculateBlock on each block in turn,
// feeding each block's hout into the next block's hin.
// Only the len(p) blocks given are read or written.
type columnKernel func(p, m []word, score []int, eq []word, hin int)

// simdKernel is the SIMD column kernel selected for the running CPU at init.
// It is nil when no SIMD implementation is supported, in which case the portable implementation is used.
var simdKernel columnKernel

type namedKernel struct {
	name   string
	kernel columnKernel
}

// simdKernels lists the column kernels supported by the running CPU, in increasing order of preference.
var simdKernels []namedKernel

// scalarBandBlocks is the widest band that is computed without the kernel.
const scalarBandBlocks = 3

// blockColumn holds the blocks of a column as separate arrays, so that runs of blocks can be handed to a columnKernel.
type blockColumn struct {
	P     []word
	M     []word
	Score []int
}

func newBlockColumn(maxNumBlocks int) blockColumn {
	var pm = make([]word, 2*maxNumBlocks)

	return blockColumn{
		P:     pm[:maxNumBlocks:maxNumBlocks],
		M:     pm[maxNumBlocks:],
		Score: make([]int, maxNumBlocks),
	}
}

// numCellsInBlock is the number of cells in block b that are not padding.
func numCellsInBlock(b, maxNumBlocks, W int) int {
	if b == maxNumBlocks-1 {
		return wordSize - W
	}

	return wordSize
}

// hasCellAboveDiagonal reports whether any of the top numCells cells of block b has a score s <= k and
// row r with r <= bound-s. This is the condition myersCalcEditDistanceGlobal checks for keeping the last block.
//
// Vertical deltas are -1, 0 or +1, so s+r never increases moving up a block: the cells with s+r <= bound
// are a run starting at the top. Scanning down from the top can stop at the first cell outside that run.
func (bc *blockColumn) hasCellAboveDiagonal(b, numCells, k, bound int) bool {
	var (
		p, m = bc.P[b] >> 1, bc.M[b] >> 1
		// score and row of the top cell
		s = bc.Score[b] - bits.OnesCount64(uint64(p)) + bits.OnesCount64(uint64(m))
		r = b * wordSize
	)

	for n := 0; n < numCells && s+r <= bound; n++ {
		if s <= k {
			return true
		}

		s += int(p&1) - int(m&1)
		p >>= 1
		m >>= 1
		r++
	}

	return false
}

// hasCellBelowDiagonal reports whether any of the top numCells cells of block b has a score s <= k and
// row r with r >= s+bound. This is the condition myersCalcEditDistanceGlobal checks for keeping the first block.
//
// Vertical deltas are -1, 0 or +1, so r-s never decreases moving down a block: the cells with r-s >= bound
// are a run ending at the bottom (non-padding) cell. Scanning up from there can stop at the first cell outside that run.
func (bc *blockColumn) hasCellBelowDiagonal(b, numCells, k, bound int) bool {
	var (
		skip = wordSize - numCells
		p, m = bc.P[b] << uint(skip), bc.M[b] << uint(skip)
		// score and row of the bottom non-padding cell
		s = bc.Score[b] - bits.OnesCount64(uint64(bc.P[b]>>uint(wordSize-skip))) + bits.OnesCount64(uint64(bc.M[b]>>uint(wordSize-skip)))
		r = b*wordSize + numCells - 1
	)

	for n := 0; n < numCells && r-s >= bound; n++ {
		if s <= k {
			return true
		}

		s -= int(p>>(wordSize-1)) - int(m>>(wordSize-1))
		p <<= 1
		m <<= 1
		r--
	}

	return false
}

func (bc *blockColumn) block(b int) *myersBlock {
	return &myersBlock{P: bc.P[b], M: bc.M[b], Score: bc.Score[b]}
}

// advance runs kernel over blocks [firstBlock, lastBlock] and returns hout of the last block.
func (bc *blockColumn) advance(kernel columnKernel, eq []word, firstBlock, lastBlock int, hin int) (hout int) {
	// Narrow bands are faster to compute directly than to hand to a kernel.
	if lastBlock-firstBlock < scalarBandBlocks {
		hout = hin
		for b := firstBlock; b <= lastBlock; b++ {
			bc.P[b], bc.M[b], hout = calculateBlock(bc.P[b], bc.M[b], eq[b], hout)
			bc.Score[b] += hout
		}

		return hout
	}

	var lastScore = bc.Score[lastBlock]

	kernel(
		bc.P[firstBlock:lastBlock+1],
		bc.M[firstBlock:lastBlock+1],
		bc.Score[firstBlock:lastBlock+1],
		eq[firstBlock:lastBlock+1],
		hin,
	)

	return bc.Score[lastBlock] - lastScore
}

// buildPeqFast builds the same peq table as BuildPeq.
// Rather than testing every (query position, symbol) pair for equality, it records the positions of each character
// within a block once, then combines them for every symbol using the alphabet's equality lists.
func buildPeqFast(query Sequence, alphabet *Alphabet) Peq {
	var (
		alphabetLength = alphabet.Len()
		maxNumBlocks   = ceilDiv(len(query), wordSize)
		peq            = make([]word, (alphabetLength+1)*maxNumBlocks)
		equalTo        = alphabet.equalTo
		positions      [256]word
	)

	if equalTo == nil {
		equalTo = alphabet.buildEqualTo()
	}

	for b := range maxNumBlocks {
		var (
			blockStart = b * wordSize
			blockQuery = query[blockStart:min(blockStart+wordSize, len(query))]
			// NOTE: We pretend like query is padded at the end with W wildcard symbols
			padding word
		)

		if len(blockQuery) < wordSize {
			padding = wordAllOnes << uint(len(blockQuery))
		}

		for r, c := range blockQuery {
			positions[c] |= word(1) << uint(r)
		}

		for symbol := range alphabetLength {
			var bits = padding
			for _, c := range equalTo[symbol] {
				bits |= positions[c]
			}

			peq[symbol*maxNumBlocks+b] = bits
		}

		// Last symbol is wildcard, so it is all 1s
		peq[alphabetLength*maxNumBlocks+b] = wordAllOnes

		for _, c := range blockQuery {
			positions[c] = 0
		}
	}

	return peq
}

// myersCalcEditDistanceSemiGlobalSIMD is myersCalcEditDistanceSemiGlobal with each column computed by kernel.
// Banding decisions are identical to myersCalcEditDistanceSemiGlobal.
func myersCalcEditDistanceSemiGlobalSIMD(kernel columnKernel, peq []word, W int, maxNumBlocks int, queryLength int, target Sequence, k int, mode alignMode, result *Result) {
	var (
		// firstBlock is 0-based index of the first block in Ukkonen band
		firstBlock = 0
		// lastBlock is 0-based index of last block in Ukkonen band
		lastBlock = min(ceilDiv(k+1, wordSize), maxNumBlocks) - 1 // y in Myers

		blocks    = newBlockColumn(maxNumBlocks)
		positions []int
	)

	// solution will never be larger than queryLength
	if mode == ModeInfix {
		k = min(queryLength, k)
	}

	// Each strongReduceNum column is reduced in more expensive way.
	// This gives speed up of about 2 times for small k
	const strongReduceNum = 2048

	// Initialize P, M and score
	for b := 0; b <= lastBlock; b++ {
		blocks.Score[b] = (b + 1) * wordSize
		blocks.P[b] = wordAllOnes // All 1s
		blocks.M[b] = 0
	}

	var bestScore = -1

	var startHout int
	if mode != ModeInfix {
		// If infix then gap before query is not penalized
		startHout = 1
	}

	for c := range target { // for each column
		var eq = peq[int(target[c])*maxNumBlocks:][:maxNumBlocks]

		// Calculate column
		var hout = blocks.advance(kernel, eq, firstBlock, lastBlock, startHout)

		// Adjust number of blocks according to Ukkonen
		if (lastBlock < maxNumBlocks-1) &&
			(blocks.Score[lastBlock]-hout <= k) &&
			((eq[lastBlock+1]&word(1) > 0) || hout < 0) {
			// If score of left block is not too big, calculate one more block
			lastBlock++

			var blHout int
			blocks.P[lastBlock], blocks.M[lastBlock], blHout = calculateBlock(wordAllOnes, 0, eq[lastBlock], hout)
			blocks.Score[lastBlock] = blocks.Score[lastBlock-1] - hout + wordSize + blHout
		} else {
			for lastBlock >= firstBlock && blocks.Score[lastBlock] >= k+wordSize {
				lastBlock--
			}
		}

		// Every some columns, do some expensive but also more efficient block reducing
		// This is important!
		//
		// Reduce the band be decreasing last block if possible
		if c%strongReduceNum == 0 {
			for lastBlock >= 0 && lastBlock >= firstBlock && allBlockCellsLarger(blocks.block(lastBlock), k) {
				lastBlock--
			}
		}

		// Even if all cells are > k, there still my be solution in next column
		// because starting conditions at upper boundary are 0.
		// That means that first block is always candidate for solution,
		// and we can never end calculation before last column
		if mode == ModeInfix && lastBlock == -1 {
			lastBlock++
		}

		// reduce band by increasing first block if possible, Not applicable to HW.
		if mode != ModeInfix {
			for firstBlock <= lastBlock && blocks.Score[firstBlock] >= k+wordSize {
				firstBlock++
			}

			if c%strongReduceNum == 0 { // Do strong reduction every some blocks
				for firstBlock <= lastBlock && allBlockCellsLarger(blocks.block(firstBlock), k) {
					firstBlock++
				}
			}
		}

		// If band stops to exist finish
		if lastBlock < firstBlock {
			result.EditDistance = bestScore
			if bestScore != -1 {
				result.EndLocations = positions
			}

			return
		}

		// Update best score
		if lastBlock == maxNumBlocks-1 {
			var colScore = blocks.Score[lastBlock]
			if colScore <= k { // Scores > k don't have correct values (so we cannot use them), but are certainly > k.
				// NOTE: Score that I find in column c is actually score from column c-W
				if bestScore == -1 || colScore <= bestScore {
					if colScore != bestScore {
						positions = positions[:0] // clear positions
						bestScore = colScore
						// Change k so we will look only for equal or better
						// scores then the best found so far.
						k = bestScore
					}

					positions = append(positions, c-W)
				}
			}
		}
	}

	// Obtain results for last W columns from last column.
	if lastBlock == maxNumBlocks-1 {
		var blockScores = getBlockCellValues(blocks.block(lastBlock))

		for i := range W {
			var colScore = blockScores[i+1]
			if colScore <= k && (bestScore == -1 || colScore <= bestScore) {
				if colScore != bestScore {
					positions = positions[:0]
					bestScore = colScore
					k = colScore
				}

				positions = append(positions, len(target)-W+i)
			}
		}
	}

	result.EditDistance = bestScore
	if bestScore != -1 {
		result.EndLocations = positions
	}
}

// myersCalcEditDistanceGlobalSIMD is myersCalcEditDistanceGlobal with each column computed by kernel.
// Banding decisions are identical to myersCalcEditDistanceGlobal.
func myersCalcEditDistanceGlobalSIMD(kernel columnKernel, peq []word, W int, maxNumBlocks int, queryLength int, target Sequence, k int, findAlignment bool, targetStopPosition int) (alignment *alignmentData, bestScore, position int) {
	if targetStopPosition > -1 && findAlignment {
		// myersCalcEditDistanceGlobal is only called by Align
		panic("cannot set both targetStopPosition and findAlignment")
	}

	// Each strongReduceNum column is reduced in more expensive way
	const strongReduceNum = 2048

	if k < abs(len(target)-queryLength) {
		bestScore = -1
		position = -1
		return
	}

	k = min(k, max(queryLength, len(target))) // Upper bound for k

	var (
		// firstBlock is 0-based index of first block in Ukkonen band.
		firstBlock int
		// lastBlock is 0-based index of last block in Ukkonen band.
		lastBlock = min(maxNumBlocks, ceilDiv(min(k, (k+queryLength-len(target))/2)+1, wordSize)) - 1
		blocks    = newBlockColumn(maxNumBlocks)
	)

	// Initialize P, M and score
	for b := 0; b <= lastBlock; b++ {
		blocks.Score[b] = (b + 1) * wordSize
		blocks.P[b] = wordAllOnes
		blocks.M[b] = 0
	}

	// If we want to find alignment, we have to store needed data.
	switch {
	case findAlignment:
		alignment = newAlignmentData(maxNumBlocks, len(target))
	case targetStopPosition > -1:
		alignment = newAlignmentData(maxNumBlocks, 1)
	}

	for c := range target {
		var eq = peq[int(target[c])*maxNumBlocks:][:maxNumBlocks]

		// Calculate column
		var hout = blocks.advance(kernel, eq, firstBlock, lastBlock, 1)

		// Update k. I do it only on end of column because it would slow calculation too much otherwise
		// NOTICE: I add W when in last block because it is actually result from W cells to the left and W cells up.
		var q int
		if lastBlock == maxNumBlocks-1 {
			q = W
		}
		k = min(
			k,
			blocks.Score[lastBlock]+max(len(target)-c-1, queryLength-((1+lastBlock)*wordSize-1)-1)+q,
		)

		// Adjust number of blocks according to Ukkonen
		// Adjust last block
		// If block is not beneath band, calculate next block. Only next because others are certainly beneath band.
		if lastBlock+1 < maxNumBlocks && (lastBlock+1)*wordSize-1 <= k-blocks.Score[lastBlock]+2*wordSize-2-len(target)+c+queryLength {
			lastBlock++

			var newHout int
			blocks.P[lastBlock], blocks.M[lastBlock], newHout = calculateBlock(wordAllOnes, 0, eq[lastBlock], hout)
			blocks.Score[lastBlock] = blocks.Score[lastBlock-1] - hout + wordSize + newHout
		}

		// While block is out of band, move one block up.
		// NOTE: Condition used here is more loose than the one from the article, since I simplified the max() part of it.
		// I could consider adding that max part, for optimal performance
		for lastBlock >= firstBlock && (blocks.Score[lastBlock] >= k+wordSize || (lastBlock+1)*wordSize-1 > k-blocks.Score[lastBlock]+2*wordSize-2-len(target)+c+queryLength+1) {
			lastBlock--
		}

		// Adjust first block
		// While outside of band, advance block
		for firstBlock <= lastBlock && (blocks.Score[firstBlock] >= k+wordSize || (firstBlock+1)*wordSize-1 < blocks.Score[firstBlock]-k-len(target)+queryLength+c) {
			firstBlock++
		}

		// NOTE: this matches myersCalcEditDistanceGlobal exactly, as the band it produces must be identical.
		if c%strongReduceNum == 0 { // Every some columns do more expensive but more efficient reduction
			for lastBlock >= firstBlock && !blocks.hasCellAboveDiagonal(lastBlock, numCellsInBlock(lastBlock, maxNumBlocks, W), k, k-len(target)+c+queryLength+1) {
				lastBlock--
			}

			// If all cells outside of band, remove block
			for firstBlock <= lastBlock && !blocks.hasCellBelowDiagonal(firstBlock, numCellsInBlock(firstBlock, maxNumBlocks, W), k, -k-len(target)+c+queryLength) {
				firstBlock++
			}
		}

		// If band stops to exist finish
		if lastBlock < firstBlock {
			bestScore = -1
			position = -1
			return
		}

		// Save column so it can be used for reconstruction
		if findAlignment {
			var offset = maxNumBlocks * c
			copy(alignment.Ps[offset+firstBlock:offset+lastBlock+1], blocks.P[firstBlock:lastBlock+1])
			copy(alignment.Ms[offset+firstBlock:offset+lastBlock+1], blocks.M[firstBlock:lastBlock+1])
			copy(alignment.Scores[offset+firstBlock:offset+lastBlock+1], blocks.Score[firstBlock:lastBlock+1])
			alignment.FirstBlocks[c] = firstBlock
			alignment.LastBlocks[c] = lastBlock
		}

		// If this is stop column, save it and finish
		if c == targetStopPosition {
			copy(alignment.Ps[firstBlock:lastBlock+1], blocks.P[firstBlock:lastBlock+1])
			copy(alignment.Ms[firstBlock:lastBlock+1], blocks.M[firstBlock:lastBlock+1])
			copy(alignment.Scores[firstBlock:lastBlock+1], blocks.Score[firstBlock:lastBlock+1])
			alignment.FirstBlocks[0] = firstBlock
			alignment.LastBlocks[0] = lastBlock

			bestScore = -1
			position = targetStopPosition

			return
		}
	}

	if lastBlock == maxNumBlocks-1 { // If last block of last column was calculated
		// Obtain best score from block -> it is complicated because query is padded with W cells
		var bestScoreW = getBlockCellValues(blocks.block(lastBlock))[W]
		if bestScoreW <= k {
			bestScore = bestScoreW
			position = len(target) - 1
			return
		}
	}

	bestScore = -1
	position = -1
	return
}

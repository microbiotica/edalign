package edalign

import (
	"errors"
	"unsafe"
)

func newAlignmentData(maxNumBlocks, targetLength int) *alignmentData {
	return &alignmentData{
		Ps:          make([]word, maxNumBlocks*targetLength),
		Ms:          make([]word, maxNumBlocks*targetLength),
		Scores:      make([]int, maxNumBlocks*targetLength),
		FirstBlocks: make([]int, targetLength),
		LastBlocks:  make([]int, targetLength),
	}
}

type alignmentData struct {
	Ps          []word
	Ms          []word
	Scores      []int
	FirstBlocks []int
	LastBlocks  []int
}

type Peq []word

// BuildPeq builds a peq table for given query and alphabet.
// peq is a table of dimensions alphabetLength+1 x maxNumBlocks.
// Bit i of peq[s * maxNumBlocks + b] is 1 if the i-th symbol from block b of query equals symbol s, otherwise it is 0
func BuildPeq(query Sequence, alphabet *Alphabet) Peq {
	if simdKernel != nil {
		return buildPeqFast(query, alphabet)
	}

	var (
		maxNumBlocks = ceilDiv(len(query), wordSize)
		peq          = make([]word, (alphabet.Len()+1)*maxNumBlocks)
	)

	// Build peq (1 is match, 0 is mismatch)
	// NOTE: last column is wildcard(symbol that matches anything) with just 1s
	for symbol := 0; symbol <= alphabet.Len(); symbol++ {
		for b := range maxNumBlocks {
			switch {
			case symbol < alphabet.Len():
				for r := (b+1)*wordSize - 1; r >= b*wordSize; r-- {
					peq[symbol*maxNumBlocks+b] <<= 1

					// NOTE: We pretend like query is padded at the end with W wildcard symbols
					if r >= len(query) || alphabet.Equal(query[r], uint8(symbol)) {
						peq[symbol*maxNumBlocks+b] += 1
					}
				}
			default:
				// Last symbol is wildcard, so it is all 1s
				peq[symbol*maxNumBlocks+b] = wordAllOnes
			}
		}
	}

	return peq
}

func myersCalcEditDistanceSemiGlobal(peq []word, W int, maxNumBlocks int, queryLength int, target Sequence, k int, mode alignMode, result *Result) {
	if simdKernel != nil {
		myersCalcEditDistanceSemiGlobalSIMD(simdKernel, peq, W, maxNumBlocks, queryLength, target, k, mode, result)
		return
	}

	var (
		// firstBlock is 0-based index of the first block in Ukkonen band
		firstBlock = 0
		// lastBlock is 0-based index of last block in Ukkonen band
		lastBlock = min(ceilDiv(k+1, wordSize), maxNumBlocks) - 1 // y in Myers

		blocks    = make([]*myersBlock, maxNumBlocks)
		bl        int
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
		blocks[bl] = &myersBlock{
			Score: (b + 1) * wordSize,
			P:     wordAllOnes, // All 1s
			M:     0,
		}
		bl++
	}

	var bestScore = -1

	var startHout int
	if mode != ModeInfix {
		// If infix then gap before query is not penalized
		startHout = 1
	}

	for c := range target { // for each column
		var peq_c = int(target[c]) * maxNumBlocks

		// Calculate column
		var hout = startHout
		bl = firstBlock
		peq_c += firstBlock

		for i := firstBlock; i <= lastBlock; i++ {
			var b = blocks[bl]
			b.P, b.M, hout = calculateBlock(b.P, b.M, peq[peq_c], hout)
			b.Score += hout

			bl++
			peq_c++
		}

		bl--
		peq_c--

		// Adjust number of blocks according to Ukkonen
		if (lastBlock < maxNumBlocks-1) &&
			(blocks[bl].Score-hout <= k) && // bl is pointing to last block
			((peq[peq_c+1]&word(1) > 0) || hout < 0) { // peq_c is pointing to last block
			// If score of left block is not too big, calculate one more block
			lastBlock++
			bl++
			peq_c++

			var b = myersBlock{
				P: wordAllOnes, // all 1s
				M: 0,
			}

			var bl_hout int
			b.P, b.M, bl_hout = calculateBlock(b.P, b.M, peq[peq_c], hout)
			b.Score = blocks[bl-1].Score - hout + wordSize + bl_hout

			blocks[bl] = &b
		} else {
			for lastBlock >= firstBlock && blocks[bl].Score >= k+wordSize {
				lastBlock--
				bl--
				peq_c--
			}
		}

		// Every some columns, do some expensive but also more efficient block reducing
		// This is important!
		//
		// Reduce the band be decreasing last block if possible
		if c%strongReduceNum == 0 {
			for lastBlock >= 0 && lastBlock >= firstBlock && allBlockCellsLarger(blocks[bl], k) {
				lastBlock--
				bl--
				peq_c--
			}
		}

		// Even if all cells are > k, there still my be solution in next column
		// because starting conditions at upper boundary are 0.
		// That means that first block is always candidate for solution,
		// and we can never end calculation before last column
		if mode == ModeInfix && lastBlock == -1 {
			lastBlock++
			bl++
			peq_c++
		}

		// reduce band by increasing first block if possible, Not applicable to HW.
		if mode != ModeInfix {
			for firstBlock <= lastBlock && blocks[firstBlock].Score >= k+wordSize {
				firstBlock++
			}

			if c%strongReduceNum == 0 { // Do strong reduction every some blocks
				for firstBlock <= lastBlock && allBlockCellsLarger(blocks[firstBlock], k) {
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
			var colScore = blocks[bl].Score
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
		var blockScores = getBlockCellValues(blocks[bl])

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

	return
}

func myersCalcEditDistanceGlobal(peq []word, W int, maxNumBlocks int, queryLength int, target Sequence, k int, findAlignment bool, targetStopPosition int) (alignment *alignmentData, bestScore, position int) {
	if simdKernel != nil {
		return myersCalcEditDistanceGlobalSIMD(simdKernel, peq, W, maxNumBlocks, queryLength, target, k, findAlignment, targetStopPosition)
	}

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
		// Current block
		bl     int
		blocks = make([]*myersBlock, maxNumBlocks)
	)

	// Initialize P, M and score
	for ; bl <= lastBlock; bl++ {
		blocks[bl] = &myersBlock{
			Score: (bl + 1) * wordSize,
			P:     wordAllOnes,
			M:     0,
		}
	}

	// If we want to find alignment, we have to store needed data.
	switch {
	case findAlignment:
		alignment = newAlignmentData(maxNumBlocks, len(target))
	case targetStopPosition > -1:
		alignment = newAlignmentData(maxNumBlocks, 1)
	}

	var targetChar int
	for c := range target {
		var peq_c = peq[int(target[targetChar])*maxNumBlocks:]

		// Calculate column
		var hout = 1
		bl = firstBlock

		for b := firstBlock; b <= lastBlock; b++ {
			block := blocks[bl]
			block.P, block.M, hout = calculateBlock(block.P, block.M, peq_c[b], hout)
			block.Score += hout

			bl++
		}

		bl--
		// bl now points to last block

		// Update k. I do it only on end of column because it would slow calculation too much otherwise
		// NOTICE: I add W when in last block because it is actually result from W cells to the left and W cells up.
		var q int
		if lastBlock == maxNumBlocks-1 {
			q = W
		}
		k = min(
			k,
			blocks[bl].Score+max(len(target)-c-1, queryLength-((1+lastBlock)*wordSize-1)-1)+q,
		)

		// Adjust number of blocks according to Ukkonen
		// Adjust last block
		// If block is not beneath band, calculate next block. Only next because others are certainly beneath band.
		if lastBlock+1 < maxNumBlocks && !(
		//score[lastBlock] >= k + WORD_SIZE ||  // NOTICE: this condition could be satisfied if above block also!
		(lastBlock+1)*wordSize-1 > k-blocks[bl].Score+2*wordSize-2-len(target)+c+queryLength) {

			lastBlock++
			bl++

			var block = &myersBlock{
				P: wordAllOnes, // All 1s
				M: 0,
			}

			var newHout int
			block.P, block.M, newHout = calculateBlock(block.P, block.M, peq_c[lastBlock], hout)
			block.Score = blocks[bl-1].Score - hout + wordSize + newHout
			hout = newHout

			blocks[bl] = block
		}

		// While block is out of band, move one block up.
		// NOTE: Condition used here is more loose than the one from the article, since I simplified the max() part of it.
		// I could consider adding that max part, for optimal performance
		for lastBlock >= firstBlock && (blocks[bl].Score >= k+wordSize || (lastBlock+1)*wordSize-1 > k-blocks[bl].Score+2*wordSize-2-len(target)+c+queryLength+1) {
			lastBlock--
			bl--
		}

		// Adjust first block
		// While outside of band, advance block
		for firstBlock <= lastBlock && (blocks[firstBlock].Score >= k+wordSize || (firstBlock+1)*wordSize-1 < blocks[firstBlock].Score-k-len(target)+queryLength+c) {
			firstBlock++
		}

		// TODO: consider if this part is useful, it does not seem to help much
		if c%strongReduceNum == 0 { // Every some columns do more expensive but more efficient reduction
			for lastBlock >= firstBlock {
				var scores = getBlockCellValues(blocks[bl])

				var numCells = wordSize
				if lastBlock == maxNumBlocks-1 {
					numCells = wordSize - W
				}

				var r = lastBlock*wordSize + numCells - 1
				var reduce = true

				for i := wordSize - numCells; i < wordSize; i++ {
					if scores[i] <= k && r <= k-scores[i]-len(target)+c+queryLength+1 {
						reduce = false
						break
					}

					r--
				}

				if !reduce {
					break
				}

				lastBlock--
				bl--
			}

			for firstBlock <= lastBlock {
				// If all cells outside of band, remove block
				var scores = getBlockCellValues(blocks[firstBlock])

				var numCells = wordSize
				if firstBlock == maxNumBlocks-1 {
					numCells = wordSize - W
				}

				var r = firstBlock*wordSize + numCells - 1
				var reduce = true

				for i := wordSize - numCells; i < wordSize; i++ {
					if scores[i] <= k && r >= scores[i]-k-len(target)+c+queryLength {
						reduce = false
						break
					}

					r--
				}

				if !reduce {
					break
				}

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
		if findAlignment && c < len(target) {
			bl = firstBlock
			for b := firstBlock; b <= lastBlock; b++ {
				var block = blocks[bl]

				alignment.Ps[maxNumBlocks*c+b] = block.P
				alignment.Ms[maxNumBlocks*c+b] = block.M
				alignment.Scores[maxNumBlocks*c+b] = block.Score
				alignment.FirstBlocks[c] = firstBlock
				alignment.LastBlocks[c] = lastBlock

				bl++
			}
		}

		// If this is stop column, save it and finish
		if c == targetStopPosition {
			for b := firstBlock; b <= lastBlock; b++ {
				var block = blocks[b]

				alignment.Ps[b] = block.P
				alignment.Ms[b] = block.M
				alignment.Scores[b] = block.Score
				alignment.FirstBlocks[0] = firstBlock
				alignment.LastBlocks[0] = lastBlock
			}

			bestScore = -1
			position = targetStopPosition

			return
		}

		targetChar++
	}

	if lastBlock == maxNumBlocks-1 { // If last block of last column was calculated
		// Obtain best score from block -> it is complicated because query is padded with W cells
		var bestScoreW = getBlockCellValues(blocks[lastBlock])[W]
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

func createReverseCopy(seq Sequence) Sequence {
	reversed := make(Sequence, len(seq))

	for i := range seq {
		reversed[i] = seq[len(seq)-i-1]
	}

	return reversed
}

// reverses a slice with elements of type E
func reverse[S ~[]E, E any](s S) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
}

func obtainAlignmentTraceback(queryLength, targetLength int, bestScore int, alignData *alignmentData) Alignment {
	var (
		maxNumBlocks = ceilDiv(queryLength, wordSize)
		W            = maxNumBlocks*wordSize - queryLength
		alignment    = make(Alignment, 0, queryLength+targetLength-1)

		c         = targetLength - 1 // index of column
		b         = maxNumBlocks - 1 // index of block in column
		currScore = bestScore        // score of current cell
		lScore    = -1               // score of left cell
		uScore    = -1               // score of upper cell
		ulScore   = -1               // score of upper left cell

		currP = alignData.Ps[c*maxNumBlocks+b] // P of current block
		currM = alignData.Ms[c*maxNumBlocks+b] // M of current block

		thereIsLeftBlock = c > 0 && b >= alignData.FirstBlocks[c-1] && b <= alignData.LastBlocks[c-1]
	)

	var (
		lP word
		lM word
	)

	if thereIsLeftBlock {
		lP = alignData.Ps[(c-1)*maxNumBlocks+b] // P of block to the left
		lM = alignData.Ms[(c-1)*maxNumBlocks+b] // M of block to the left
	}

	currP <<= W
	currM <<= W

	var blockPos = wordSize - W - 1 // 0 based index of current cell in blockPos

scan:
	for {
		if c == 0 {
			thereIsLeftBlock = true
			lScore = b*wordSize + blockPos + 1
			ulScore = lScore - 1
		}

		// calculate scores
		if lScore == -1 && thereIsLeftBlock {
			lScore = alignData.Scores[(c-1)*maxNumBlocks+b] // score of block to the left

			for i := 0; i < wordSize-blockPos-1; i++ {
				if lP&wordHighBitMask > 0 {
					lScore--
				}
				if lM&wordHighBitMask > 0 {
					lScore++
				}

				lP <<= 1
				lM <<= 1
			}
		}
		if ulScore == -1 {
			switch {
			case lScore != -1:
				ulScore = lScore
				if lP&wordHighBitMask > 0 {
					ulScore--
				}
				if lM&wordHighBitMask > 0 {
					ulScore++
				}
			case c > 0 && b-1 >= alignData.FirstBlocks[c-1] && b-1 <= alignData.LastBlocks[c-1]:
				// This is the case when upper left cell is last cell in block,
				// and block to left if not in band so lScore is -1.
				ulScore = alignData.Scores[(c-1)*maxNumBlocks+b-1]
			}
		}
		if uScore == -1 {
			uScore = currScore
			if currP&wordHighBitMask > 0 {
				uScore--
			}
			if currM&wordHighBitMask > 0 {
				uScore++
			}

			currP <<= 1
			currM <<= 1
		}

		// Move
		switch {
		case uScore != -1 && uScore+1 == currScore: // Move up - insertion to target - deletion from query
			currScore = uScore
			lScore = ulScore
			uScore = -1
			ulScore = -1

			if blockPos == 0 { // If entering new (upper) block
				if b == 0 { // If there are no cells above (only boundary cells)
					alignment = append(alignment, Insert) // Move up
					for i := 0; i < c+1; i++ {            // Move left until end
						alignment = append(alignment, Delete)
					}
					break scan
				} else {
					blockPos = wordSize - 1
					b--
					currP = alignData.Ps[c*maxNumBlocks+b]
					currM = alignData.Ms[c*maxNumBlocks+b]

					if c > 0 && b >= alignData.FirstBlocks[c-1] && b <= alignData.LastBlocks[c-1] {
						thereIsLeftBlock = true
						lP = alignData.Ps[(c-1)*maxNumBlocks+b]
						lM = alignData.Ms[(c-1)*maxNumBlocks+b]
					} else {
						thereIsLeftBlock = false
						// TODO(martin): There may not be left block, but there can be left boundary - do we
						// handle this correctly then? Are l and ul score set correctly? I should check that / refactor this.
					}
				}
			} else {
				blockPos--
				lP <<= 1
				lM <<= 1
			}

			// Mark move
			alignment = append(alignment, Insert)

		case lScore != -1 && lScore+1 == currScore: // Move left - deletion from target - insertion to query
			currScore = lScore
			uScore = ulScore
			lScore = -1
			ulScore = -1
			c--

			if c == -1 { // If there are no cells to the left (only boundary cells)
				alignment = append(alignment, Delete)
				var numUp = b*wordSize + blockPos + 1
				for range numUp { // Move up until end
					alignment = append(alignment, Insert)
				}
				break scan
			}

			currP = lP
			currM = lM

			if c > 0 && b >= alignData.FirstBlocks[c-1] && b <= alignData.LastBlocks[c-1] {
				thereIsLeftBlock = true
				lP = alignData.Ps[(c-1)*maxNumBlocks+b]
				lM = alignData.Ms[(c-1)*maxNumBlocks+b]
			} else {
				if c == 0 { // If there are no cells to the left (only boundary cells)
					thereIsLeftBlock = true
					lScore = b*wordSize + blockPos + 1
					ulScore = lScore - 1
				} else {
					thereIsLeftBlock = false
				}
			}

			// Mark move
			alignment = append(alignment, Delete)

		case ulScore != -1: // Move up left - (mis)match
			var moveCode = Mismatch
			if ulScore == currScore {
				moveCode = Match
			}

			currScore = ulScore
			uScore = -1
			lScore = -1
			ulScore = -1
			c--

			if c == -1 { // If there are no cells to the left (only boundary cells)
				alignment = append(alignment, moveCode) // Move left
				var numUp = b*wordSize + blockPos
				for range numUp { // Move up until end
					alignment = append(alignment, Insert)
				}
				break scan
			}
			if blockPos == 0 { // If entering upper left block
				if b == 0 { // If there are no more cells above (only boundary cells)
					alignment = append(alignment, moveCode) // Move up left
					for i := 0; i < c+1; i++ {              // Move left until end
						alignment = append(alignment, Delete)
					}
					break scan
				}

				blockPos = wordSize - 1
				b--
				currP = alignData.Ps[c*maxNumBlocks+b]
				currM = alignData.Ms[c*maxNumBlocks+b]
			} else { // If entering left block
				blockPos--
				currP = lP
				currM = lM
				currP <<= 1
				currM <<= 1
			}

			// Set new left block
			if c > 0 && b >= alignData.FirstBlocks[c-1] && b <= alignData.LastBlocks[c-1] {
				thereIsLeftBlock = true
				lP = alignData.Ps[(c-1)*maxNumBlocks+b]
				lM = alignData.Ms[(c-1)*maxNumBlocks+b]
			} else {
				if c == 0 { // If there are no cells to the left (only boundary cells)
					thereIsLeftBlock = true
					lScore = b*wordSize + blockPos - 1
					ulScore = lScore - 1
				} else {
					thereIsLeftBlock = false
				}
			}

			// Mark move
			alignment = append(alignment, moveCode)

		default:
			// Reached end - finished!
			break scan
		}
	}

	reverse(alignment)

	return alignment
}

func obtainAlignmentHirschbergPeq(query, rQuery Sequence, target, rTarget Sequence, peq Peq, alphabet *Alphabet, bestScore int) Alignment {
	var (
		maxNumBlocks = ceilDiv(len(query), wordSize)
		W            = maxNumBlocks*wordSize - len(query)

		rPeq = BuildPeq(rQuery, alphabet)
	)

	// Divide dynamic matrix into two halves, left and right
	var (
		leftHalfWidth  = len(target) / 2
		rightHalfWidth = len(target) - leftHalfWidth
	)

	// Calculate left half
	alignDataLeftHalf, _, _ := myersCalcEditDistanceGlobal(peq, W, maxNumBlocks, len(query), target, bestScore, false, leftHalfWidth-1)
	// Calculate right half
	alignDataRightHalf, _, _ := myersCalcEditDistanceGlobal(rPeq, W, maxNumBlocks, len(query), rTarget, bestScore, false, rightHalfWidth-1)

	// Unwrap the left half.
	var (
		firstBlockIdxLeft = alignDataLeftHalf.FirstBlocks[0]
		lastBlockIdxLeft  = alignDataLeftHalf.LastBlocks[0]

		// scoresLeft contains scores from left column, starting with scoresLeftStartIDx row (query index)
		// and ending with scoresLeftEndIdx row (0-indexed).
		scoresLeft = make([]int, (lastBlockIdxLeft-firstBlockIdxLeft+1)*wordSize)
	)

	for blockIdx := firstBlockIdxLeft; blockIdx <= lastBlockIdxLeft; blockIdx++ {
		readBlock(
			myersBlock{
				P:     alignDataLeftHalf.Ps[blockIdx],
				M:     alignDataLeftHalf.Ms[blockIdx],
				Score: alignDataLeftHalf.Scores[blockIdx],
			},
			scoresLeft[(blockIdx-firstBlockIdxLeft)*wordSize:],
		)
	}

	var scoresLeftStartIdx = firstBlockIdxLeft * wordSize
	// If last block contains padding, shorten the length of scores for the length of padding
	if lastBlockIdxLeft == maxNumBlocks-1 {
		scoresLeft = scoresLeft[:len(scoresLeft)-W]
	}

	// Unwrap the right half (I also reverse it while unwrapping).
	var (
		firstBlockIdxRight = alignDataRightHalf.FirstBlocks[0]
		lastBlockIdxRight  = alignDataRightHalf.LastBlocks[0]
		scoresRight        = make([]int, (lastBlockIdxRight-firstBlockIdxRight+1)*wordSize)
	)

	for blockIdx := firstBlockIdxRight; blockIdx <= lastBlockIdxRight; blockIdx++ {
		readBlockReverse(
			myersBlock{
				P:     alignDataRightHalf.Ps[blockIdx],
				M:     alignDataRightHalf.Ms[blockIdx],
				Score: alignDataRightHalf.Scores[blockIdx],
			},
			scoresRight[(lastBlockIdxRight-blockIdx)*wordSize:],
		)
	}

	var scoresRightStartIdx = len(query) - (lastBlockIdxRight+1)*wordSize
	// If there is padding at the beginning of scoresRight (that can happen because of reversing that we do),
	// move pointer forward to remove the padding (that is why we remember originalStart).
	if scoresRightStartIdx < 0 {
		scoresRight = scoresRight[W:]
		scoresRightStartIdx += W
	}

	//--------------------- Find the best move ----------------//
	// Find the query/row index of cell in left column which together with its lower right neighbour
	// from right column gives the best score (when summed). We also have to consider boundary cells
	// (those cells at -1 indexes).
	//  x|
	//  -+-
	//   |x

	var (
		queryIdxLeftStart = max(scoresLeftStartIdx, scoresRightStartIdx-1)
		queryIdxLeftEnd   = min(scoresLeftStartIdx+len(scoresLeft)-1, scoresRightStartIdx+len(scoresRight)-2)

		leftScore                  = -1
		rightScore                 = -1
		queryIdxLeftAlignment      = -1 // Query/row index of cell in left column where alignment is passing through.
		queryIdxLeftAlignmentFound = false
	)

	for queryIdx := queryIdxLeftStart; queryIdx <= queryIdxLeftEnd; queryIdx++ {
		leftScore = scoresLeft[queryIdx-scoresLeftStartIdx]
		rightScore = scoresRight[queryIdx+1-scoresRightStartIdx]

		if leftScore+rightScore == bestScore {
			queryIdxLeftAlignment = queryIdx
			queryIdxLeftAlignmentFound = true
			break
		}
	}

	// Check boundary cells
	if !queryIdxLeftAlignmentFound && scoresLeftStartIdx == 0 && scoresRightStartIdx == 0 {
		leftScore = leftHalfWidth
		rightScore = scoresRight[0]

		if leftScore+rightScore == bestScore {
			queryIdxLeftAlignment = -1
			queryIdxLeftAlignmentFound = true
		}
	}

	if !queryIdxLeftAlignmentFound && scoresLeftStartIdx+len(scoresLeft) == len(query) && scoresRightStartIdx+len(scoresRight) == len(query) {
		leftScore = scoresLeft[len(scoresLeft)-1]
		rightScore = rightHalfWidth

		if leftScore+rightScore == bestScore {
			queryIdxLeftAlignment = len(query) - 1
			queryIdxLeftAlignmentFound = true
		}
	}

	if !queryIdxLeftAlignmentFound {
		panic("edalign: there was no move that is part of the optimal alignment; there is no such alignment or given bestScore is not correct")
	}

	// Calculate alignments for upper half of left half (upper left - ul)
	// and lower half of right half (lower right - lr)
	var (
		ulHeight = queryIdxLeftAlignment + 1
		lrHeight = len(query) - ulHeight
		ulWidth  = leftHalfWidth
		lrWidth  = rightHalfWidth
	)

	var ulAlignment = obtainAlignmentPeq(
		query[:ulHeight], rQuery[lrHeight:],
		target[:ulWidth], rTarget[lrWidth:],
		BuildPeq(query[:ulHeight], alphabet), alphabet, leftScore,
	)
	var lrAlignment = obtainAlignmentPeq(
		query[ulHeight:ulHeight+lrHeight], rQuery[:lrHeight],
		target[ulWidth:ulWidth+lrWidth], rTarget[:lrWidth],
		BuildPeq(query[ulHeight:ulHeight+lrHeight], alphabet), alphabet, rightScore,
	)

	var alignment = make(Alignment, len(ulAlignment)+len(lrAlignment))
	copy(alignment, ulAlignment)
	copy(alignment[len(ulAlignment):], lrAlignment)

	return alignment
}

func obtainAlignmentPeq(query, rQuery Sequence, target, rTarget Sequence, peq Peq, alphabet *Alphabet, bestScore int) Alignment {
	// Handle special case when one of sequences has length of 0
	if len(query) == 0 || len(target) == 0 {
		var alignment = make(Alignment, len(target)+len(query))
		var t = Insert
		if len(query) == 0 {
			t = Delete
		}
		for i := range alignment {
			alignment[i] = t
		}

		return alignment
	}

	var (
		maxNumBlocks = ceilDiv(len(query), wordSize)
		W            = maxNumBlocks*wordSize - len(query)
	)

	// TODO: think about reducing number of memory allocations in alignment functions, probably
	// by sharing some memory that is allocated only once. That refers to: Peq, columns in Hirschberg,
	// and it could also be done for alignments - we could have one big array for alignment that would be
	// sparsely populated by each of steps in recursion, and at the end we would just consolidate those results.

	// If estimated memory consumption for traceback algorithm is smaller than 1MB use it,
	// otherwise use Hirschberg's algorithm. By running few tests I choose boundary of 1MB as optimal.
	var alignmentDataSize = (2*int(unsafe.Sizeof(word(0)))+int(unsafe.Sizeof(int(0))))*maxNumBlocks*len(target) + 2*int(unsafe.Sizeof(int(0)))*len(target)

	if alignmentDataSize < 1024*1024 {
		var alignment, _, _ = myersCalcEditDistanceGlobal(peq, W, maxNumBlocks, len(query), target, bestScore, true, -1)

		return obtainAlignmentTraceback(len(query), len(target), bestScore, alignment)
	} else {
		return obtainAlignmentHirschbergPeq(query, rQuery, target, rTarget, peq, alphabet, bestScore)
	}
}

// AlignPEQ implements Align for a precalculated Peq of the query sequence.
// peq must come from the same Alphabet used in config.
func AlignPEQ(query, target Sequence, peq Peq, config *Config) (*Result, error) {
	// Handle special situation when at least one of the sequences has length 0.
	if len(query) == 0 || len(target) == 0 {
		switch config.Mode {
		case ModeGlobal:
			return &Result{
				EditDistance: max(len(query), len(target)),
				EndLocations: []int{len(target) - 1},
			}, nil
		case ModeInfix, ModePrefix:
			return &Result{
				EditDistance: len(query),
				EndLocations: []int{-1},
			}, nil
		default:
			// Modes are declared internally by this package.
			panic(config.Mode)
		}
	}

	// Initialisation
	var (
		maxNumBlocks = ceilDiv(len(query), wordSize)
		W            = maxNumBlocks*wordSize - len(query)
	)

	// Main Calculation
	var (
		result = Result{
			EditDistance: -1,
		}
		dynamicK bool
		k        = config.K
	)

	if k < 0 { // if valid k is not given, auto-adjust k until solution is found
		dynamicK = true
		k = wordSize
	}

	for ; dynamicK && result.EditDistance == -1; k *= 2 {
		switch config.Mode {
		case ModeInfix, ModePrefix:
			myersCalcEditDistanceSemiGlobal(peq, W, maxNumBlocks, len(query), target, k, config.Mode, &result)
		case ModeGlobal:
			var err error
			_, result.EditDistance, _ = myersCalcEditDistanceGlobal(peq, W, maxNumBlocks, len(query), target, k, false, -1)
			if err != nil {
				return nil, err
			}
		default:
			return nil, errors.New("invalid alignment mode for dynamic K")
		}
	}

	// If there is a solution
	if result.EditDistance >= 0 {
		// If ModeGlobal, set end location explicitly.
		if config.Mode == ModeGlobal {
			result.EndLocations = make([]int, 1)
			result.EndLocations[0] = len(target) - 1
		}

		// Find starting locations
		switch {
		case config.Task&TaskLocation == TaskLocation:
			result.StartLocations = make([]int, len(result.EndLocations))

			switch config.Mode {
			case ModeInfix: // If HW, I need to calculate start locations
				var (
					rTarget = createReverseCopy(target)
					rQuery  = createReverseCopy(query)
					// peq for reversed query
					rPeq = BuildPeq(rQuery, config.Alphabet)
				)

				for i := 0; i < len(result.EndLocations); i++ {
					var endLocation = result.EndLocations[i]

					if endLocation == -1 {
						// NOTE: Sometimes one of optimal solutions is that query starts before target, like this:
						//                       AAGG <- target
						//                   CCTT     <- query
						//   It will never be only optimal solution and it does not happen often, however it is
						//   possible and in that case end location will be -1. What should we do with that?
						//   Should we just skip reporting such end location, although it is a solution?
						//   If we do report it, what is the start location? -4? -1? Nothing?
						// TODO: Figure this out. This has to do in general with how we think about start
						//   and end locations.
						//   Also, we have alignment later relying on this locations to limit the space of it's
						//   search -> how can it do it right if these locations are negative or incorrect?
						result.StartLocations[i] = 0 // I put 0 for now, but it does not make much sense.
					} else {
						var resultsSHW Result

						targetSection := rTarget[len(target)-endLocation-1:]
						targetSection = targetSection[:endLocation+1]

						myersCalcEditDistanceSemiGlobal(rPeq, W, maxNumBlocks, len(query), targetSection, result.EditDistance, ModePrefix, &resultsSHW)

						// Taking last location as start ensures that alignment will not start with insertions
						// if it can start with mismatches instead.
						result.StartLocations[i] = endLocation - resultsSHW.EndLocations[len(resultsSHW.EndLocations)-1]
					}
				}
			default:
				// If mode is SHW or NW
				for i := 0; i < len(result.EndLocations); i++ {
					result.StartLocations[i] = 0
				}
			}
		}
	}

	if config.Task&TaskPath == TaskPath && result.EditDistance >= 0 {
		var (
			alnStartLocation = result.StartLocations[0]
			alnEndLocation   = result.EndLocations[0]
			alnTarget        = target[alnStartLocation : alnEndLocation+1]
			rAlnTarget       = createReverseCopy(alnTarget)
			rQuery           = createReverseCopy(query)
		)

		result.Alignment = obtainAlignmentPeq(query, rQuery, alnTarget, rAlnTarget, peq, config.Alphabet, result.EditDistance)
	}

	result.PercentIdentity = percentIdentity(len(query), len(target), result.EditDistance)

	return &result, nil
}

// percentIdentity returns the identity of an alignment with editDistance relative to the shorter
// of the two sequences, clamped to [0, 100]. It returns 0 if no alignment was found (editDistance < 0).
func percentIdentity(queryLength, targetLength, editDistance int) float64 {
	var minimal = min(queryLength, targetLength)
	if editDistance < 0 || minimal == 0 {
		return 0
	}

	return float64(max(minimal-editDistance, 0)) / float64(minimal) * 100.0
}

func Align(query, target Sequence, config *Config) (*Result, error) {
	return AlignPEQ(query, target, BuildPeq(query, config.Alphabet), config)
}

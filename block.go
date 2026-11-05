package edalign

type myersBlock struct {
	P     word // Pvin
	M     word // Mvin
	Score int  // score of last cell in block
}

// calculateBlock corresponds to the Advanced_Block function from Myers.
// Calculates one word(block), which is part of a column.
// Highest bit of word (one most to the left) is most bottom cell of block from column.
func calculateBlock(pv, mv, eq word, hin int) (pvOut, mvOut word, hout int) {
	// hin can be 1, -1 or 0.
	// 1  -> 00...01
	// 0  -> 00...00
	// -1 -> 11...11 (2-complement)

	// 00...001 if hin is -1, 00...000 if 0 or 1
	var hinIsNeg = word(hin>>2) & word(1)

	var xv = eq | mv

	eq |= hinIsNeg

	var xh = (((eq & pv) + pv) ^ pv) | eq

	var ph = mv | ^(xh | pv)

	var mh = pv & xh

	hout = int((ph & wordHighBitMask) >> (wordSize - 1))
	hout -= int((mh & wordHighBitMask) >> (wordSize - 1))

	ph <<= 1
	mh <<= 1

	mh |= hinIsNeg
	ph |= word((hin + 1) >> 1)

	pvOut = mh | ^(xv | ph)
	mvOut = ph & xv

	return
}

// getBlockCellValues gets values of cells in a myersBlock, starting with the bottom cell in myersBlock.
func getBlockCellValues(block *myersBlock) (scores [wordSize]int) {
	var score = block.Score
	var mask = wordHighBitMask
	for i := range wordSize - 1 {
		scores[i] = score
		if block.P&mask > 0 {
			score--
		}
		if block.M&mask > 0 {
			score++
		}
		mask >>= 1
	}

	scores[wordSize-1] = score
	return
}

// allBlockCellsLarger returns true if all cells in block have value larger than k.
func allBlockCellsLarger(block *myersBlock, k int) bool {
	var scores = getBlockCellValues(block)

	for i := range wordSize {
		if scores[i] <= k {
			return false
		}
	}

	return true
}

// readBlock appends write of cells in block onto given slice, starting with first/top cell.
// dest must have len of at least wordSize.
func readBlock(block myersBlock, dest []int) {
	var score = block.Score
	var mask = wordHighBitMask

	for i := range wordSize - 1 {
		dest[wordSize-1-i] = score
		if block.P&mask > 0 {
			score--
		}
		if block.M&mask > 0 {
			score++
		}
		mask >>= 1
	}

	dest[0] = score
}

// readBlockReverse writes values of cells in block into given slice, starting with last/bottom cell.
// dest must have a len of at least wordSize
func readBlockReverse(block myersBlock, dest []int) {
	var score = block.Score
	var mask = wordHighBitMask

	for i := range wordSize - 1 {
		dest[i] = score
		if block.P&mask > 0 {
			score--
		}
		if block.M&mask > 0 {
			score++
		}
		mask >>= 1
	}

	dest[wordSize-1] = score
}

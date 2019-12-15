package quantile

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

var _ = fmt.Println

type tuple struct {
	v     float64
	g     int
	delta int
}

type gksummary []tuple

func (gk *gksummary) Len() int           { return len(*gk) }
func (gk *gksummary) Less(i, j int) bool { return (*gk)[i].v < (*gk)[j].v }
func (gk *gksummary) Swap(i, j int)      { (*gk)[i], (*gk)[j] = (*gk)[j], (*gk)[i] }

func (gk *gksummary) Size() int {

	l := len(*gk)

	if l == 0 {
		return 0
	}

	var n int
	for _, t := range *gk {
		n += t.g
	}

	return n + (*gk)[l-1].delta

}

// reduces the number of elements but doesn't lose precision.
// Algorithm "value merging" in Appendix A of
// "Power-Conserving Computation of Order-Statistics over Sensor Networks" (Greenwald, Khanna 2004)
// http://www.cis.upenn.edu/~mbgreen/papers/pods04.pdf
func (gk *gksummary) mergeValues() {

	return

	var missing int

	var dst int

	for src := 1; src < len(*gk); src++ {
		if (*gk)[dst].v == (*gk)[src].v {
			(*gk)[dst].delta += (*gk)[src].g + (*gk)[src].delta
			missing += (*gk)[src].g
			continue
		}

		dst++
		// add in the extra 'g' for the elements we removed
		(*gk)[src].g += missing
		missing = 0
		(*gk)[dst] = (*gk)[src]
	}

	(*gk) = (*gk)[:dst+1]
}

type Stream struct {
	summary []gksummary
	epsilon float64
	n       int
	b       int // block size
}

func New(epsilon float64, n int) (*Stream, error) {
	epsN := epsilon * float64(n)
	b := int(math.Floor(math.Log(epsN) / epsilon))
	if b < 0 {
		return nil, errors.New("epsilon too accurate for n")
	}
	return &Stream{summary: make([]gksummary, 1, 1), epsilon: epsilon, n: n, b: b}, nil
}

func (s *Stream) Dump() {
	for _, ss := range s.summary {
		fmt.Println(ss)
	}
}

func (s *Stream) Update(e float64) {

	// From http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/08-Quantile/Zhang.html

	s.summary[0] = append(s.summary[0], tuple{e, 1, 0}) // insert unsorted

	if len(s.summary[0]) < s.b {
		return // all done
	}

	/* -----------------------------------
	   Level 0 is full... PACK IT UP !!!
	   ----------------------------------- */

	sort.Sort(&s.summary[0])

	s.summary[0].mergeValues()

	sc := prune(s.summary[0], (s.b+1)/2+1, 0, 0)
	s.summary[0] = s.summary[0][:0] // empty

	for k := 1; k < len(s.summary); k++ {
		if len(s.summary[k]) == 0 {
			/* --------------------------------------
			   Empty: put compressed summary in sk
			   -------------------------------------- */

			s.summary[k] = sc // Store it
			s.Dump()
			return // Done
		}

		/* --------------------------------------
		   sk contained a compressed summary
		   -------------------------------------- */

		tmp := merge(s.summary[k], sc) // here we're merging two summaries with s.b * 2^k entries each
		sc = prune(tmp, (s.b+1)/2+1, float64(k)/float64(s.b), k)
		// NOTE: sc is used in next iteration
		// -  it is passed to the next level !

		s.summary[k] = s.summary[k][:0] // Re-initialize
	}

	// fell off the end of our loop -- no more s.summary entries
	s.summary = append(s.summary, sc)
	s.Dump()

}

// From http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/08-Quantile/Greenwald-D.html "Prune"
func prune(sc gksummary, b int, epsilon float64, level int) gksummary {

	r := gksummary{sc[0]}

	rmin := sc[0].g
	size := sc.Size()

	for i := 1; i <= b; i++ {

		rank := int(float64(size) * float64(i) / float64(b))
		v := lookupRank(sc, rank, epsilon+float64(level)/float64(b), size)

		elt := tuple{v: v.v}

		elt.g = v.rmin - rmin
		rmin += elt.g

		elt.delta = v.rmax - rmin
		/*
			if r[len(r)-1].v == elt.v {
				// ignore if we've already seen it
				continue
			}
		*/
		r = append(r, elt)
	}

	return r
}

type lookupResult struct {
	v    float64
	rmin int
	rmax int
}

// return the tuple containing rank 'r' in summary
// combine this inline with prune(), otherwise we're O(n^2)
// or over a channel?
func lookupRank(summary gksummary, r int, epsilon float64, n int) lookupResult {

	var rmin int

	if r == 1 {
		return lookupResult{v: summary[0].v, rmin: 1, rmax: summary[0].delta + 1}
	}

	epsN := int(epsilon * float64(n))

	for _, t := range summary {

		rmin += t.g
		rmax := rmin + t.delta

		if r-rmin <= epsN && rmax-r <= epsN {
			return lookupResult{v: t.v, rmin: rmin, rmax: rmax}
		}
	}

	return lookupResult{v: summary[len(summary)-1].v, rmin: rmin, rmax: rmin + summary[len(summary)-1].delta}

}

// Other 'merge' algorithms:
// http://www.cs.ubc.ca/~xujian/paper/quant.pdf .
// http://www.mathcs.emory.edu/~cheung/Courses/584-StreamDB/Syllabus/08-Quantile/Greenwald-D.html
// or "COMBINE" in http://www.cis.upenn.edu/~mbgreen/papers/chapter.pdf
// "Quantiles and Equidepth Histograms over Streams" (Greenwald, Khanna 2005)

// This paper points out it's a merge, sort, and I *believe* that the new
// rmin/rmax definitions just work out from the existing g/delta combinations.
// "Power-conserving Computation of Order-Statistics over Sensor Networks"
// http://www.cis.upenn.edu/~mbgreen/papers/pods04.pdf
func merge(s1, s2 gksummary) gksummary {

	if len(s1) == 0 {
		return s2
	}
	if len(s2) == 0 {
		return s1
	}

	var smerge gksummary

	var i1, i2 int

	for i1 < len(s1) || i2 < len(s2) {

		var t tuple

		var from int

		if i1 < len(s1) && i2 < len(s2) {

			if s1[i1].v <= s2[i2].v {
				t = s1[i1]
				from = 1
				i1++
			} else {
				t = s2[i2]
				from = 2
				i2++
			}
		} else if i1 < len(s1) && i2 >= len(s2) {
			t = s1[i1]
			from = 1
			i1++
		} else if i1 >= len(s1) && i2 < len(s2) {
			t = s2[i2]
			from = 2
			i2++
		} else {
			panic("invariant violated")
		}

		if from == 1 {
			if i2 < len(s2) {
				t.delta += s2[i2].g + s2[i2].delta - 1
			} else {
				t.delta += s2[i2-1].g + s2[i2-1].delta
			}
		}

		if from == 2 {
			if i1 < len(s1) {
				t.delta += s1[i1].g + s1[i1].delta - 1
			} else {
				t.delta += s1[i1-1].g + s1[i1-1].delta
			}
		}

		smerge = append(smerge, t)
	}

	// all done

	// The merged list might have duplicate elements -- merge them.
	smerge.mergeValues()

	return smerge
}

// !! Must call Finish to allow processing queries
func (s *Stream) Finish() {
	sort.Sort(&s.summary[0])
	s.summary[0].mergeValues()

	size := len(s.summary[0])

	for i := 1; i < len(s.summary); i++ {
		s.summary[0] = merge(s.summary[0], s.summary[i])
		size += s.b * 1 << uint(i)
	}
}

// GK query
func (s *Stream) Query(q float64) float64 {

	// convert quantile to rank

	r := int(q * float64(s.n))

	if r == 1 {
		return s.summary[0][0].v
	}
	epsN := int(s.epsilon * float64(s.n))

	var rmin int
	for _, t := range s.summary[0] {
		rmin += t.g
		rmax := rmin + t.delta

		if r-rmin <= epsN && rmax-r <= epsN {
			return t.v
		}
	}

	return s.summary[0][len(s.summary[0])-1].v
}

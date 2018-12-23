package data

type MnistSample struct {
	dataVec []uint8
	label   int
}

type FuncFeature struct {
	XDIndex    int
	XDValue    uint8
	LabelIndex int
	FeatureKey string
	Count      int
	Weight     float64
	Prob       float64
}

func (f *FuncFeature) IncCount() {
	f.Count += 1
}

func (s *MnistSample) GetDataVectorLen() int {
	return len(s.dataVec)
}

func (s *MnistSample) GetDataVec() []uint8 {
	return s.dataVec
}

func (s *MnistSample) GetDataByIndex(index int) uint8 {
	return s.dataVec[index]
}

func (s *MnistSample) GetLabel() int {
	return s.label
}

package data

type MnistSample struct {
	dataVec []uint8
	label   int
}

func (s *MnistSample) GetDataVectorLen() int {
	return len(s.dataVec)
}

func (s *MnistSample) GetDataVec() []uint8 {
	return s.dataVec
}

func (s *MnistSample) GetLabel() int {
	return s.label
}

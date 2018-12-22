package IIS

import (
	"fmt"
	"math"
	"math/rand"
	"maxent/dataformat"
	"sync"
	"time"
)

type MaxEntIIS struct {
	test  []*data.MnistSample
	train []*data.MnistSample

	XFeatureNum  int
	w            [][]float64
	probSampleX  map[int]float64
	probSampleXY map[string]float64

	labelYCount int
	M           float64
	N           float64

	featureRecord map[int]bool
}

func (m *MaxEntIIS) LoadData(trainPath, testPath string) {
	var yCount int
	m.test, m.labelYCount = data.ReadMnistCsv(trainPath)
	m.train, yCount = data.ReadMnistCsv(testPath)
	if yCount != m.labelYCount {
		panic("output label not equal between training and test set")
	}

	m.XFeatureNum = m.train[0].GetDataVectorLen()
	m.N = 1 / float64(len(m.train))
	m.w = make([][]float64, m.labelYCount)
	rand.Seed(88888)
	for i := range m.w {
		m.w[i] = make([]float64, m.XFeatureNum)
		for j := 0; j < m.XFeatureNum; j++ {
			m.w[i][j] = rand.Float64()
		}
	}

	fmt.Println("load data done")

	m.initProbSample()

	return
}

func (m *MaxEntIIS) initProbSample() {
	m.probSampleX = make(map[int]float64)
	m.probSampleXY = make(map[string]float64)
	m.featureRecord = make(map[int]bool)
	globalCounter := 0
	m.M = 0.0002
	for _, X := range m.train {
		label := X.GetLabel()
		for fi, xi := range X.GetDataVec() {
			if xi == 1 {
				globalCounter += 1
				m.probSampleX[fi] += 1
				m.featureRecord[fi] = true
				xyKey := fmt.Sprintf("%d_%d", fi, label)
				m.probSampleXY[xyKey] += 1
			}
		}
	}
	fmt.Println(len(m.featureRecord))

	for key, count := range m.probSampleX {
		m.probSampleX[key] = count / float64(globalCounter)
	}

	for key, count := range m.probSampleXY {
		m.probSampleXY[key] = count / float64(globalCounter)
	}

	fmt.Println("init prob done")

}

func (m *MaxEntIIS) computeExw() {
	wg := sync.WaitGroup{}
	wg.Add(m.labelYCount)
	for li := 0; li < m.labelYCount; li += 1 {
		go func(label int, w *sync.WaitGroup) {
			defer w.Done()
			for j := 0; j < m.XFeatureNum; j++ {
				if m.w[label][j] > 0 {
					m.w[label][j] = math.Exp(m.w[label][j])
				}
			}
		}(li, &wg)
	}
	wg.Wait()

}

func (m *MaxEntIIS) modelCompute(sample *data.MnistSample) float64 {
	x := sample.GetDataVec()
	normalization := 0.0
	numerator := 0.0
	for yi := range m.w {
		tmp := 0.0
		wi := m.w[yi]
		for i, value := range x {
			if value == 1 {
				tmp += wi[i]
			}
		}
		expTmp := math.Exp(tmp)
		if yi == sample.GetLabel() {
			numerator = expTmp
		}
		normalization += expTmp
	}
	return numerator / normalization
}

func (m *MaxEntIIS) solveBetaDelta(fi, label int) float64 {
	xyKey := fmt.Sprintf("%d_%d", fi, label)
	epobf := m.probSampleXY[xyKey]

	result := 0.0
	for _, item := range m.train {
		result += m.N * m.modelCompute(item)
	}

	return math.Log(epobf/result) * m.M
}

func (m *MaxEntIIS) trainLabels(label int, group *sync.WaitGroup) {
	defer group.Done()

	subWg := &sync.WaitGroup{}
	subWg.Add(4)
	for i := 0; i < 4; i++ {
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := i * 196; fi < (i+1)*196; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.w[label][fi] += m.solveBetaDelta(fi, label)
				}
			}
		}(subWg)
	}
	subWg.Wait()
}

func (m *MaxEntIIS) StartTraining(iter int) {
	for i := 0; i < iter; i++ {
		fmt.Println("iter ", i, " start")
		wg := sync.WaitGroup{}
		start := time.Now()
		for li := 0; li < m.labelYCount; li += 1 {
			wg.Add(1)
			go m.trainLabels(li, &wg)
		}
		wg.Wait()
		fmt.Println("iter ", i, "time cost: ", time.Now().Sub(start))

		m.Test()
	}
}

func (m *MaxEntIIS) Predict(item *data.MnistSample) bool {
	x := item.GetDataVec()
	max := 0.0
	predictLabel := -1
	for yi := range m.w {
		tmp := 0.0
		wi := m.w[yi]
		for i, value := range x {
			if value == 1 {
				tmp += wi[i]
			}
		}
		tmp = math.Exp(tmp)
		if tmp > max {
			max = tmp
			predictLabel = yi
		}
	}
	return predictLabel == item.GetLabel()
}

func (m *MaxEntIIS) Test() {
	top1Hit := 0
	testCount := len(m.test)
	for i := 0; i < testCount; i++ {
		if m.Predict(m.test[i]) {
			top1Hit += 1
		}
	}
	fmt.Println("test accuracy：", float64(top1Hit)/float64(testCount))
}

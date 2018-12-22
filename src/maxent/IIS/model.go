package IIS

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"maxent/dataformat"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type MaxEntIIS struct {
	test  []*data.MnistSample
	train []*data.MnistSample

	XFeatureNum  int
	w            [][]float64
	expW         [][]float64
	probSampleX  map[int]float64
	probSampleXY map[string]float64

	labelYCount int
	M           float64
	N           float64

	featureRecord map[int]bool
}

const SplitCount = 3

func (m *MaxEntIIS) LoadData(trainPath, testPath string) {
	var yCount int
	m.test, m.labelYCount = data.ReadMnistCsv(trainPath)
	m.train, yCount = data.ReadMnistCsv(testPath)
	if yCount != m.labelYCount {
		panic("output label not equal between training and test set")
	}

	m.XFeatureNum = m.train[0].GetDataVectorLen()
	m.N = 1 / float64(len(m.train))
	rand.Seed(88888)
	m.w = make([][]float64, m.labelYCount)
	for i := 0; i < m.labelYCount; i++ {
		m.w[i] = make([]float64, m.XFeatureNum)
		for j := 0; j < m.XFeatureNum; j++ {
			m.w[i][j] = rand.Float64()
		}
	}
	m.LoadModel()

	m.expW = make([][]float64, m.labelYCount)
	for i := 0; i < m.labelYCount; i++ {
		m.expW[i] = make([]float64, m.XFeatureNum)
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
	m.M = 0.0005
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

func (m *MaxEntIIS) computeExpW(label int, group *sync.WaitGroup) {

	defer group.Done()

	subWg := &sync.WaitGroup{}
	subWg.Add(SplitCount)
	if SplitCount == 3 {
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 0; fi < 262; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.expW[label][fi] = math.Exp(m.w[label][fi])
				}
			}
		}(subWg)
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 262; fi < 522; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.expW[label][fi] = math.Exp(m.w[label][fi])
				}
			}
		}(subWg)
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 522; fi < m.XFeatureNum; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.expW[label][fi] = math.Exp(m.w[label][fi])
				}
			}
		}(subWg)

	} else if SplitCount == 4 {
		for i := 0; i < SplitCount; i++ {
			go func(start int, wg *sync.WaitGroup) {
				defer wg.Done()
				for fi := start * 196; fi < (start+1)*196; fi++ {
					if _, ok := m.featureRecord[fi]; ok {
						m.expW[label][fi] = math.Exp(m.w[label][fi])
					}
				}
			}(i, subWg)
		}
	}
	subWg.Wait()
}

func (m *MaxEntIIS) modelCompute(sample *data.MnistSample) float64 {
	x := sample.GetDataVec()
	normalization := 0.0
	numerator := 0.0
	for yi := range m.w {
		tmp := 1.0
		wi := m.expW[yi]
		for i, value := range x {
			if value == 1 {
				tmp *= wi[i]
			}
		}
		if yi == sample.GetLabel() {
			numerator = tmp
		}
		normalization += tmp
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
	subWg.Add(SplitCount)
	if SplitCount == 3 {
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 0; fi < 262; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.w[label][fi] += m.solveBetaDelta(fi, label)
				}
			}
		}(subWg)
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 262; fi < 522; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.w[label][fi] += m.solveBetaDelta(fi, label)
				}
			}
		}(subWg)
		go func(wg *sync.WaitGroup) {
			defer wg.Done()
			for fi := 522; fi < m.XFeatureNum; fi++ {
				if _, ok := m.featureRecord[fi]; ok {
					m.w[label][fi] += m.solveBetaDelta(fi, label)
				}
			}
		}(subWg)
	} else if SplitCount == 4 {
		for i := 0; i < 4; i++ {
			go func(start int, wg *sync.WaitGroup) {
				defer wg.Done()
				for fi := start * 196; fi < (start+1)*196; fi++ {
					if _, ok := m.featureRecord[fi]; ok {
						m.w[label][fi] += m.solveBetaDelta(fi, label)
					}
				}
			}(i, subWg)
		}
	}
	subWg.Wait()
}

func (m *MaxEntIIS) StartTraining(iter int) {
	for i := 0; i < iter; i++ {
		start := time.Now()
		fmt.Println(" ------ ", start, " ------ iter", i)
		wg := sync.WaitGroup{}
		wg.Add(m.labelYCount)
		for li := 0; li < m.labelYCount; li += 1 {
			m.computeExpW(li, &wg)
		}
		wg.Wait()

		wg.Add(m.labelYCount)
		for li := 0; li < m.labelYCount; li += 1 {
			go m.trainLabels(li, &wg)
		}
		wg.Wait()
		fmt.Println("iter", i, "time cost: ", time.Now().Sub(start))

		m.Test()
	}
	m.SaveModel()
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

func (m *MaxEntIIS) SaveModel() {
	fileName := fmt.Sprintf("./last_model.dat")
	if f, err := os.Create(fileName); err == nil {
		defer f.Close()
		for li := 0; li < m.labelYCount; li += 1 {
			for fi := 0; fi < m.XFeatureNum; fi += 1 {
				if fi == m.XFeatureNum-1 {
					if _, err := f.WriteString(
						fmt.Sprintf("%f", m.w[li][fi])); err != nil {

					}
				} else {
					if _, err := f.WriteString(
						fmt.Sprintf("%f,", m.w[li][fi])); err != nil {

					}
				}
			}
			if _, err := f.WriteString("\n"); err != nil {
				break
			}
		}
	}
}

func (m *MaxEntIIS) LoadModel() bool {
	fileName := fmt.Sprintf("./last_model.dat")
	if file, err := os.Open(fileName); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		label := 0
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			for fi := 0; fi < len(items); fi++ {
				m.w[label][fi], _ = strconv.ParseFloat(items[fi], 64)
			}
			if label < m.labelYCount {
				label++
			} else {
				break
			}
		}
		return true
	}
	return false
}

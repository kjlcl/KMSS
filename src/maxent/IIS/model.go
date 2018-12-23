package IIS

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"maxent/dataformat"
	"os"
	"strings"
	"time"
)

type MaxEntIIS struct {
	test  []*data.MnistSample
	train []*data.MnistSample

	xDimension     int
	featureMap     map[string]*data.FuncFeature
	featureArray   []*data.FuncFeature
	featureFuncLen int
	allPwXy        []float64

	labelYCount int
	M           float64
	N           int
	probX       float64
}

func (m *MaxEntIIS) LoadData(trainPath, testPath string) {
	var yCount int
	m.test, m.labelYCount = data.ReadMnistCsv(trainPath)
	m.train, yCount = data.ReadMnistCsv(testPath)
	if yCount != m.labelYCount {
		panic("output label not equal between training and test set")
	}

	m.xDimension = m.train[0].GetDataVectorLen()
	m.featureMap = make(map[string]*data.FuncFeature)
	m.N = len(m.train)
	m.probX = float64(m.N)
	m.allPwXy = make([]float64, m.N)
	m.M = 1.0 / float64(m.xDimension)
	for i := 0; i < m.N; i++ {
		item := m.train[i]
		for j := 0; j < m.xDimension; j++ {
			key := fmt.Sprintf(
				"%d_%d_%d", item.GetLabel(), j, item.GetDataByIndex(j))
			if feature, ok := m.featureMap[key]; ok {
				feature.IncCount()
			} else {
				m.featureMap[key] = &data.FuncFeature{
					LabelIndex: item.GetLabel(),
					XDIndex:    j,
					XDValue:    item.GetDataByIndex(j),
					FeatureKey: key,
					Count:      1,
				}
			}
		}
	}

	totalCount := 0
	for _, item := range m.featureMap {
		totalCount += item.Count
	}
	m.featureArray = make([]*data.FuncFeature, len(m.featureMap))

	arrayIndex := 0
	rand.Seed(345345)
	for _, item := range m.featureMap {
		item.Prob = float64(item.Count) / float64(totalCount)
		//item.Weight = rand.Float64() / 10
		m.featureArray[arrayIndex] = item
		arrayIndex += 1
	}

	//m.LoadModel()

	fmt.Println("load data done")

	return
}

func (m *MaxEntIIS) PwXY(sample *data.MnistSample) float64 {
	Zw := 0.0
	tmpSum := make([]float64, m.labelYCount)
	fiY := sample.GetLabel()
	for i := 0; i < m.labelYCount; i++ {
		tmpSum[i] = 0.0
	}
	for _, feature := range m.featureArray {
		if feature.XDValue == sample.GetDataByIndex(feature.XDIndex) {
			tmpSum[feature.LabelIndex] += feature.Weight
		}
	}

	for i := 0; i < m.labelYCount; i++ {
		tmpSum[i] = math.Exp(tmpSum[i])
		Zw += tmpSum[i]
	}
	return tmpSum[fiY] / Zw

}

func (m *MaxEntIIS) StartTraining(iter int, coreNum int) {
	for i := 0; i < iter; i++ {
		start := time.Now()
		fmt.Println(" ------ ", start, " ------ iter", i)
		m.calcAllPwXY()

		for _, feature := range m.featureArray {
			epobf := feature.Prob
			result := 0.0
			for ti, item := range m.train {
				if item.GetLabel() != feature.LabelIndex ||
					item.GetDataByIndex(feature.XDIndex) != feature.XDValue {
					continue
				}

				result += m.allPwXy[ti]
			}
			feature.Weight += math.Log((m.probX*epobf)/result) * m.M
		}

		fmt.Println("iter", i, "time cost: ", time.Now().Sub(start))
		m.Validation()
	}
}

func (m *MaxEntIIS) Predict(item *data.MnistSample) bool {
	tmpSum := make([]float64, m.labelYCount)
	for _, feature := range m.featureArray {
		if feature.XDValue == item.GetDataByIndex(feature.XDIndex) {
			tmpSum[feature.LabelIndex] += feature.Weight
		}
	}
	if tmpSum[0] > tmpSum[1] {
		return 0 == item.GetLabel()
	} else {
		return 1 == item.GetLabel()
	}
}

func (m *MaxEntIIS) Validation() {
	top1Hit := 0
	vCount := len(m.train[:10000])
	for i := 0; i < vCount; i++ {
		if m.Predict(m.test[i]) {
			top1Hit += 1
		}
	}
	fmt.Println("test accuracy：", float64(top1Hit)/float64(vCount))
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
		for fi := 0; fi < len(m.featureArray); fi += 1 {
			feature := m.featureArray[fi]
			f.WriteString(
				fmt.Sprintf(
					"%d,%d,%d,%f",
					feature.LabelIndex, feature.XDIndex, feature.XDValue, feature.Weight))
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

func (m *MaxEntIIS) TestAllPwXY() {
	m.calcAllPwXY()
}

func (m *MaxEntIIS) calcAllPwXY() {
	for i, item := range m.train {
		m.allPwXy[i] = m.PwXY(item)
	}
}

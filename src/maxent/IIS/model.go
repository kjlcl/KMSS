package IIS

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"maxent/dataformat"
	"os"
	"sort"
	"sync"
	"time"
)

type FeatureList []*data.FuncFeature

func (fl FeatureList) Len() int { return len(fl) }
func (fl FeatureList) Less(i, j int) bool {
	if fl[i].XDIndex == fl[j].XDIndex {
		if fl[i].XDValue == fl[j].XDValue {
			return fl[i].LabelIndex < fl[j].LabelIndex
		}
		return fl[i].XDValue < fl[j].XDValue
	}
	return fl[i].XDIndex < fl[j].XDIndex
}
func (fl FeatureList) Swap(i, j int) { fl[i], fl[j] = fl[j], fl[i] }

type MaxEntIIS struct {
	test  []*data.MnistSample
	train []*data.MnistSample

	xDimension     int
	featureArray   FeatureList
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
	featureMap := make(map[string]*data.FuncFeature)
	m.N = len(m.train)
	m.probX = float64(m.N)
	m.allPwXy = make([]float64, m.N)
	m.M = 1.0 / float64(m.xDimension)
	for i := 0; i < m.N; i++ {
		item := m.train[i]
		for j := 0; j < m.xDimension; j++ {
			key := fmt.Sprintf(
				"%d_%d_%d", item.GetLabel(), j, item.GetDataByIndex(j))
			if feature, ok := featureMap[key]; ok {
				feature.IncCount()
			} else {
				featureMap[key] = &data.FuncFeature{
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
	for _, item := range featureMap {
		totalCount += item.Count
	}
	m.featureFuncLen = len(featureMap)
	m.featureArray = make(FeatureList, m.featureFuncLen)

	arrayIndex := 0
	rand.Seed(time.Now().Unix())
	for _, item := range featureMap {
		item.Prob = float64(item.Count) / float64(totalCount)
		item.Weight = rand.Float64() / 74
		m.featureArray[arrayIndex] = item
		arrayIndex++
	}

	sort.Sort(m.featureArray)

	fmt.Println("load data done")

	return
}

func (m *MaxEntIIS) ComputePwXY(sample *data.MnistSample) float64 {
	Zw := 0.0
	tmpSum := make([]float64, m.labelYCount)
	fiY := sample.GetLabel()
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

func (m *MaxEntIIS) ComputePwXYV2(dataVec []uint8, label int) float64 {
	Zw := 0.0
	tmpSum := make([]float64, m.labelYCount)
	for _, feature := range m.featureArray {
		if feature.XDValue == dataVec[feature.XDIndex] {
			tmpSum[feature.LabelIndex] += feature.Weight
		}
	}

	for i := 0; i < m.labelYCount; i++ {
		tmpSum[i] = math.Exp(tmpSum[i])
		Zw += tmpSum[i]
	}
	return tmpSum[label] / Zw
}

func (m *MaxEntIIS) StartTraining(iter int, coreNum int) {
	for i := 0; i < iter; i++ {
		start := time.Now()
		fmt.Println(" ------ ", start, " ------ iter", i)
		m.calcAllPwXYV2()

		deltaList := make([]float64, m.featureFuncLen)
		for fi, feature := range m.featureArray {
			/**
			将calcAllPwXYV2中计算的 m.allPwXy[fi] * 1 / m.probX提出放到这里
			*/
			//deltaList[fi] = math.Log((feature.Prob)/m.allPwXy[fi]) * m.M
			deltaList[fi] = math.Log((feature.Prob*m.probX)/m.allPwXy[fi]) * m.M
		}
		for fi := 0; fi < m.featureFuncLen; fi++ {
			m.featureArray[fi].Weight += deltaList[fi]
		}
		m.Test()
		if i%10 == 0 {

		}
	}
}

func (m *MaxEntIIS) Predict(item *data.MnistSample) bool {
	tmpSum := make([]float64, m.labelYCount)
	for _, feature := range m.featureArray {
		if feature.XDValue == item.GetDataByIndex(feature.XDIndex) {
			tmpSum[feature.LabelIndex] += feature.Weight
		}
	}

	//if tmpSum[1] >= tmpSum[0] {
	//	return 1 == item.GetLabel()
	//} else {
	//	return 0 == item.GetLabel()
	//}

	max := -1000000000000.0
	maxLabel := 0
	for li := 0; li < m.labelYCount; li++ {
		if tmpSum[li] >= max {
			max = tmpSum[li]
			maxLabel = li
		}
	}
	return item.GetLabel() == maxLabel
}

func (m *MaxEntIIS) Validation() {
	top1Hit := 0
	vCount := len(m.train[:10000])
	for i := 0; i < vCount; i++ {
		if m.Predict(m.test[i]) {
			top1Hit += 1
		}
	}
	fmt.Println("validation accuracy：", float64(top1Hit)/float64(vCount))
}

func (m *MaxEntIIS) ComputeLH() {
	result := 0.0
	for i := 0; i < len(m.train); i++ {
		result += math.Log(m.allPwXy[i])
	}
	result = result / m.probX
	fmt.Println("likehood ", result)

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
		if content, err := json.Marshal(m.featureArray); err == nil {
			if _, err := f.Write(content); err != nil {
				fmt.Println(err.Error())
			}
		}
	}
}

func (m *MaxEntIIS) LoadModel() bool {
	fileName := fmt.Sprintf("./last_model.dat")
	if dat, err := ioutil.ReadFile(fileName); err == nil {
		if err := json.Unmarshal(dat, &m.featureArray); err == nil {
			return true
		} else {
			fmt.Println(err.Error())
		}
	} else {
		fmt.Println(err.Error())
	}
	return false
}

func (m *MaxEntIIS) calcAllPwXY() {

	for i, item := range m.train {
		m.allPwXy[i] = m.ComputePwXY(item)
	}
}

func (m *MaxEntIIS) calcAllPwXYParallel(coreNum int) {
	trainLen := len(m.train)
	batchSize := trainLen / coreNum
	wg := sync.WaitGroup{}
	wg.Add(coreNum)
	for i := 0; i < coreNum; i++ {
		indEnd := (i + 1) * batchSize
		if i == coreNum-1 {
			indEnd = trainLen
		}
		go func(start, end int, notify *sync.WaitGroup) {
			defer notify.Done()
			for i, item := range m.train[start:end] {
				m.allPwXy[i] = m.ComputePwXY(item)
			}
		}(i*batchSize, indEnd, &wg)
	}
}

func (m *MaxEntIIS) calcAllPwXYV2() {

	pwTmp := make([]float64, m.labelYCount)
	for _, item := range m.train {
		/**
		相比Version 1 只遍历样本中出现的x，y对，
		version 2 最大的修改为遍历训练数据的所有x，并配对所有的y进行计算
		*/
		for li := 0; li < m.labelYCount; li++ {
			pwTmp[li] = m.ComputePwXYV2(item.GetDataVec(), li)
		}

		for fi, feature := range m.featureArray {
			for i, pw := range pwTmp {
				if feature.LabelIndex == i && feature.XDValue == item.GetDataByIndex(feature.XDIndex) {
					/**
					  将(1 / m.probX)相乘提出放到外面，可以节省计算
					*/
					//m.allPwXy[fi] += (1 / m.probX) * pw
					m.allPwXy[fi] += pw
				}
			}
		}
	}
}

func (m *MaxEntIIS) TestIter() {
	m.calcAllPwXYV2()

	deltaList := make([]float64, m.featureFuncLen)
	for fi, feature := range m.featureArray {
		/**
		将calcAllPwXYV2中计算的 m.allPwXy[fi] * 1 / m.probX提出放到这里
		*/
		//deltaList[fi] = math.Log((feature.Prob)/m.allPwXy[fi]) * m.M
		deltaList[fi] = math.Log((feature.Prob*m.probX)/m.allPwXy[fi]) * m.M
	}
	for fi := 0; fi < m.featureFuncLen; fi++ {
		m.featureArray[fi].Weight += deltaList[fi]
	}
}

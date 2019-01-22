package LR

import (
	"bufio"
	"config"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

const LoadDataInMem bool = true

type LogisticRegression struct {
	weights    []float64
	bias       float64
	featureLen int
	trainPath  string
	testPath   string
}

type SoftMaxRegression struct {
	weights         [][]float64
	featureLen      int
	labelCount      int
	lossGradient    []float64
	softmax         []float64
	bias            []float64
	softmaxGradient [][]float64
	trainPath       string
	testPath        string
}

type SparseTrainItem struct {
	Label    int
	Features map[int]float64
}

type IndexTrainItem struct {
	Label    int
	Features []float64
}

func Shuffle(vals []int) {
	r := rand.New(rand.NewSource(time.Now().Unix()))
	for len(vals) > 0 {
		n := len(vals)
		randIndex := r.Intn(n)
		vals[n-1], vals[randIndex] = vals[randIndex], vals[n-1]
		vals = vals[:n-1]
	}
}

func (lr *LogisticRegression) init() {
	lr.trainPath = config.GetLRConf().TrainPath
	lr.testPath = config.GetLRConf().TestPath
}

func (lr *LogisticRegression) sigmoid(z float64) float64 {
	return 1.0 / (1 + math.Exp(-z))
}

func (lr *LogisticRegression) Train(iter int) {
	var training []IndexTrainItem
	var testing []IndexTrainItem
	if file, err := os.Open(lr.trainPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				fs := make([]float64, lr.featureLen)
				for i, item := range items[1:] {
					if pixel, err := strconv.ParseFloat(item, 64); err == nil {
						fs[i] = float64(pixel) / 255
					}
				}
				training = append(training, IndexTrainItem{Label: label, Features: fs})
			}
		}
	}
	if file, err := os.Open(lr.testPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				fs := make([]float64, lr.featureLen)
				for i, item := range items[1:] {
					if pixel, err := strconv.ParseFloat(item, 64); err == nil {
						fs[i] = float64(pixel) / 255
					}
				}
				testing = append(testing, IndexTrainItem{Label: label, Features: fs})
			}
		}
	}

	trainCount := len(training)
	batchCount := config.GetLRConf().BatchCount
	gradients := make([]float64, batchCount)
	for i := 0; i < iter; i++ {
		end := trainCount / batchCount

		for batchIndex := 0; batchIndex < end; batchIndex++ {
			start := batchIndex * batchCount
			end := start + batchCount
			if end > trainCount {
				end = trainCount
			}
			for j := 0; j < batchCount; j++ {
				gradients[j] = 0.0
			}
			for j := start; j < end; j++ {
				item := training[j]
				tmp := 0.0
				for k := 0; k < lr.featureLen; k++ {
					tmp += item.Features[k] * lr.weights[k]
				}
				gradients[j] = float64(item.Label) - tmp - lr.bias
			}
			for f := 0; f < lr.featureLen; f++ {
				dwf := 0.0
				for j := start; j < end; j++ {
					dwf += gradients[j] * training[j].Features[f]
				}
				lr.weights[f] += 0.01 * dwf
			}
		}
	}
}

func (smr *SoftMaxRegression) init() {
	smr.featureLen = 784
	smr.labelCount = 10

	smr.softmax = make([]float64, smr.labelCount)
	smr.bias = make([]float64, smr.labelCount)
	smr.lossGradient = make([]float64, smr.labelCount)
	smr.softmaxGradient = make([][]float64, smr.labelCount)
	for i := 0; i < smr.labelCount; i++ {
		smr.softmaxGradient[i] = make([]float64, smr.labelCount)
	}

	smr.weights = make([][]float64, smr.labelCount)
	for i := 0; i < smr.labelCount; i++ {
		smr.weights[i] = make([]float64, smr.featureLen)
	}
}

func (smr *SoftMaxRegression) forward(x []float64) {

	max := 0.0
	for i := 0; i < smr.labelCount; i++ {
		// TODO 检查weights 和x 维度是否一致
		weights := smr.weights[i]
		tmp := 0.0
		for j := 0; j < len(weights); j++ {
			tmp += weights[j] * x[j]
		}
		smr.softmax[i] = tmp + smr.bias[i]
		if smr.softmax[i] > max {
			max = smr.softmax[i]
		}
	}
	for i := 0; i < smr.labelCount; i++ {
		smr.softmax[i] = math.Exp(smr.softmax[i] - max)
	}

	for i := 0; i < smr.labelCount; i++ {
		for j := 0; j < smr.labelCount; j++ {
			if i == j {
				smr.softmaxGradient[i][j] = smr.softmax[i] * (1.0 - smr.softmax[j])
			} else {
				smr.softmaxGradient[i][j] = -smr.softmax[i] * smr.softmax[j]
			}
		}
	}
}

func (lr *LogisticRegression) RandomWriteTest() {
	length := len(lr.weights)

	for i := 0; i < length; i++ {
		index := rand.Float64() * float64(length)
		lr.weights[int(index)] = rand.Float64()
	}
}

func (smr *SoftMaxRegression) Train(iter int) {

	trainCount := 0
	var training []IndexTrainItem
	var testing []IndexTrainItem
	if file, err := os.Open(smr.trainPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				trainCount += 1
				fs := make([]float64, smr.featureLen)
				for i, item := range items[1:] {
					if pixel, err := strconv.ParseFloat(item, 64); err == nil {
						fs[i] = float64(pixel) / 255
					}
				}
				training = append(training, IndexTrainItem{Label: label, Features: fs})
			}
		}
	}
	if file, err := os.Open(smr.testPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				fs := make([]float64, smr.featureLen)
				for i, item := range items[1:] {
					if pixel, err := strconv.ParseFloat(item, 64); err == nil {
						fs[i] = float64(pixel) / 255
					}
				}
				testing = append(testing, IndexTrainItem{Label: label, Features: fs})
			}
		}
	}

	randArray := make([]int, trainCount)
	for i := 0; i < trainCount; i++ {
		randArray[i] = i
	}

	batchCount := config.GetSoftmaxConf().BatchCount
	for i := 0; i < iter; i++ {
		end := trainCount / batchCount

		for batchIndex := 0; batchIndex < end; batchIndex++ {
			start := batchIndex * batchCount
			end := start + batchCount
			if end > trainCount {
				end = trainCount
			}
			for _, index := range randArray[start:end] {
				item := training[randArray[index]]
				smr.forward(item.Features)
				for i := 0; i < smr.labelCount; i++ {
					for j := 0; j < smr.featureLen; j++ {
						dw := -1.0 / smr.softmax[i] * smr.softmaxGradient[item.Label][i] * item.Features[j]
						smr.weights[i][j] += 0.01 * dw
					}

				}
			}
		}

		Shuffle(randArray)
	}
}

package LR

import (
	"bufio"
	"config"
	"fmt"
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
}

type SoftMaxRegression struct {
	weights         [][]float64
	featureLen      int
	labelCount      int
	lossGradient    [][]float64
	softmax         [][]float64
	bias            []float64
	softmaxGradient [][][]float64
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

func (lr *LogisticRegression) sigmoid(z float64) float64 {
	return 1.0 / (1 + math.Exp(-z))
}

func (lr *LogisticRegression) init() {
	lr.featureLen = config.GetLRConf().FeatureLen
	lr.weights = make([]float64, lr.featureLen)
}

func (lr *LogisticRegression) predict(item *SparseTrainItem) bool {
	sum := 0.0
	for k, score := range item.Features {
		sum += lr.weights[k] * score
	}
	p := lr.sigmoid(sum + lr.bias)
	y := item.Label
	if y == 1 && p >= 0.5 || y == 0 && p < 0.5 {
		return true
	}
	return false
}

func (lr *LogisticRegression) Train(iter int) {
	lr.init()

	learningRate := config.GetLRConf().LearningRate
	var training []SparseTrainItem
	var testing []SparseTrainItem
	trainPath := config.GetLRConf().TrainPath
	testPath := config.GetLRConf().TestPath
	if file, err := os.Open(trainPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				fs := make(map[int]float64)
				for _, item := range items[1:] {
					pair := strings.Split(item, ":")
					if len(pair) != 2 {
						fmt.Println("error format ", line)
						continue
					}
					if score, err := strconv.ParseFloat(pair[1], 64); err == nil {
						if index, err := strconv.Atoi(pair[0]); err == nil {
							fs[index] = score
						}
					}
				}
				training = append(training, SparseTrainItem{Label: label, Features: fs})
			}
		}
	}
	if file, err := os.Open(testPath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			if label, err := strconv.Atoi(items[0]); err == nil {
				fs := make(map[int]float64)
				for _, item := range items[1:] {
					pair := strings.Split(item, ":")
					if len(pair) != 2 {
						fmt.Println("error format ", line)
						continue
					}
					if score, err := strconv.ParseFloat(pair[1], 64); err == nil {
						if index, err := strconv.Atoi(pair[0]); err == nil {
							fs[index] = score
						}
					}
				}
				testing = append(testing, SparseTrainItem{Label: label, Features: fs})
			}
		}
	}

	trainCount := len(training)
	batchCount := config.GetLRConf().OneBatch
	residual := make([]float64, batchCount)
	for it := 0; it < iter; it++ {
		end := trainCount / batchCount

		for batchIndex := 0; batchIndex < end; batchIndex++ {
			start := batchIndex * batchCount
			end := start + batchCount
			if end > trainCount {
				end = trainCount
			}
			updateIndex := make(map[int]int)
			db := 0.0
			for bi, item := range training[start:end] {
				tmp := 0.0
				for k, score := range item.Features {
					tmp += score * lr.weights[k]
					updateIndex[k] = 1
				}
				residual[bi] = float64(item.Label) - lr.sigmoid(tmp+lr.bias)
				db += residual[bi]
			}
			lr.bias += learningRate * (db / float64(end-start))

			for fi := range updateIndex {
				dwf := 0.0
				for bi, item := range training[start:end] {
					if score, ok := item.Features[fi]; ok {
						dwf += residual[bi] * score
					}
				}
				lr.weights[fi] += learningRate * dwf / float64(end-start)
			}
		}

		correctCount := 0
		testCount := float64(len(testing))
		for i := 0; i < len(testing); i++ {
			item := testing[i]

		}
		fmt.Println("iter ", it, " test ac ", float64(correctCount)/testCount, "correct count: ", correctCount)

	}
}

func (smr *SoftMaxRegression) init() {
	smr.featureLen = 784
	smr.labelCount = 10
	oneBatch := config.GetSoftmaxConf().OneBatch

	smr.softmax = make([][]float64, oneBatch)
	for i := 0; i < oneBatch; i++ {
		smr.softmax[i] = make([]float64, smr.labelCount)
	}
	smr.bias = make([]float64, smr.labelCount)
	smr.lossGradient = make([][]float64, oneBatch)
	for i := 0; i < oneBatch; i++ {
		smr.lossGradient[i] = make([]float64, smr.labelCount)
	}
	smr.softmaxGradient = make([][][]float64, oneBatch)
	for i := 0; i < oneBatch; i++ {
		smr.softmaxGradient[i] = make([][]float64, smr.labelCount)
		for j := 0; j < smr.labelCount; j++ {
			smr.softmaxGradient[i][j] = make([]float64, smr.labelCount)
		}
	}

	smr.weights = make([][]float64, smr.labelCount)
	for i := 0; i < smr.labelCount; i++ {
		smr.weights[i] = make([]float64, smr.featureLen)
	}
}

func (smr *SoftMaxRegression) forward(batchIndex int, x []float64) {

	max := -1000000000.0
	for i := 0; i < smr.labelCount; i++ {
		// TODO 检查weights 和x 维度是否一致
		weights := smr.weights[i]
		tmp := 0.0
		for j := 0; j < len(weights); j++ {
			tmp += weights[j] * x[j]
		}
		smr.softmax[batchIndex][i] = tmp + smr.bias[i]
		if smr.softmax[batchIndex][i] > max {
			max = smr.softmax[batchIndex][i]
		}
	}
	sum := 0.0
	for i := 0; i < smr.labelCount; i++ {
		smr.softmax[batchIndex][i] = math.Exp(smr.softmax[batchIndex][i] - max)
		sum += smr.softmax[batchIndex][i]
	}
	for i := 0; i < smr.labelCount; i++ {
		smr.softmax[batchIndex][i] = smr.softmax[batchIndex][i] / sum
	}

	for i := 0; i < smr.labelCount; i++ {
		for j := 0; j < smr.labelCount; j++ {
			if i == j {
				smr.softmaxGradient[batchIndex][i][j] = smr.softmax[batchIndex][i] * (1.0 - smr.softmax[batchIndex][j])
			} else {
				smr.softmaxGradient[batchIndex][i][j] = -smr.softmax[batchIndex][i] * smr.softmax[batchIndex][j]
			}
		}
	}
}

func (smr *SoftMaxRegression) predict(item *IndexTrainItem) bool {

	max := -10000000000.0
	predictLabel := 0
	weightCount := len(smr.weights[0])
	softmax := make([]float64, smr.labelCount)
	for i := 0; i < smr.labelCount; i++ {
		tmp := 0.0
		for j := 0; j < weightCount; j++ {
			tmp += smr.weights[i][j] * item.Features[j]
		}
		softmax[i] = tmp + smr.bias[i]
		if softmax[i] > max {
			max = softmax[i]
			predictLabel = i
		}
	}
	return predictLabel == item.Label
}

func (lr *LogisticRegression) RandomWriteTest() {
	length := len(lr.weights)

	for i := 0; i < length; i++ {
		index := rand.Float64() * float64(length)
		lr.weights[int(index)] = rand.Float64()
	}
}

func (smr *SoftMaxRegression) Train(iter int) {

	smr.init()

	trainCount := 0
	var training []IndexTrainItem
	var testing []IndexTrainItem
	lr := config.GetSoftmaxConf().LearningRate
	trainPath := config.GetSoftmaxConf().TrainPath
	testPath := config.GetSoftmaxConf().TestPath
	if file, err := os.Open(trainPath); err == nil {
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
	if file, err := os.Open(testPath); err == nil {
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

	oneBatch := config.GetSoftmaxConf().OneBatch
	batchSize := trainCount / oneBatch
	for it := 0; it < iter; it++ {
		for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
			start := batchIndex * oneBatch
			end := start + oneBatch
			if end > trainCount {
				end = trainCount
			}
			for batchIndex, index := range randArray[start:end] {
				item := training[randArray[index]]
				smr.forward(batchIndex, item.Features)
			}
			for i := 0; i < smr.labelCount; i++ {
				for j := 0; j < smr.featureLen; j++ {
					dw := 0.0
					for bi, index := range randArray[start:end] {
						item := training[randArray[index]]
						if item.Features[j] != 0 {
							dw += -1.0 / smr.softmax[bi][item.Label] * smr.softmaxGradient[bi][item.Label][i] * item.Features[j]
						}
					}
					dw /= float64(oneBatch)
					if !math.IsNaN(dw) {
						smr.weights[i][j] -= lr * dw
					}
				}
				db := 0.0
				for bi, index := range randArray[start:end] {
					item := training[randArray[index]]
					db += -1.0 / smr.softmax[bi][item.Label] * smr.softmaxGradient[bi][item.Label][i]
				}
				db /= float64(oneBatch)
				if !math.IsNaN(db) {
					smr.bias[i] -= lr * db
				}
			}
		}

		//correctCount := 0
		//trainCount := float64(len(training))
		//for i := 0; i < len(training); i++ {
		//	item := training[i]
		//	if smr.predict(&item) {
		//		correctCount += 1
		//	}
		//}
		//fmt.Println("------------------- iter ", it, " ------------------ ac ", float64(correctCount)/trainCount)

		correctCount := 0
		testCount := float64(len(testing))
		for i := 0; i < len(testing); i++ {
			item := testing[i]
			if smr.predict(&item) {
				correctCount += 1
			}
		}
		fmt.Println("------------------- iter ", it, " ------------------ ac ", float64(correctCount)/testCount)

		Shuffle(randArray)
	}
}

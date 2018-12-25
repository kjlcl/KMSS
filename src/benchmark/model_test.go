package benchmark

import (
	"fmt"
	"math"
	"maxent/IIS"
	"testing"
)

var model = IIS.MaxEntIIS{}

func init() {
	fmt.Println("test start")
	model.LoadData(
		"/Users/liang/Works/KMSS/resource/Mnist/mnist_test.csv",
		"/Users/liang/Works/KMSS/resource/Mnist/mnist_train.csv")
}

func BenchmarkExpTest(b *testing.B) {
	for i := 0; i < b.N; i++ {
		model.TestIter()
	}
}

func ExpTest(b *testing.B) {
	a := []float64{0.04, 0, 3, 0.5, 0.14, 0.66, 0.17, 0.32, 0.31, 0.444}
	for i := 0; i < b.N; i++ {
		result := 0.0
		for _, item := range a {
			result += item
		}
		math.Exp(result)
	}
}

func MultiplyTest(b *testing.B) {
	a := []float64{0.04, 0, 3, 0.5, 0.14, 0.66, 0.17, 0.32, 0.31, 0.444}
	for i := 0; i < b.N; i++ {
		result := 1.0
		for _, item := range a {
			result *= item
		}
	}
}

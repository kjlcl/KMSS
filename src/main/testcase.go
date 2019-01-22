package main

import (
	"LR"
	"fmt"
	"math"
	"maxent/IIS"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func maxent() {

	model := IIS.MaxEntIIS{}
	model.LoadData(
		"./resource/Mnist/mnist_test.csv",
		"./resource/Mnist/mnist_train.csv")

	model.StartTraining(1500, 1)
	//model.StartTraining(200)
	//start := time.Now()
	//model.TestAllPwXY()
	//fmt.Println(time.Now().Sub(start))

	//fmt.Println(math.Exp(0.9))
}

func _main() {
	start := time.Now()
	array := []float64{0.8, 0.7, 0.55, 0.12321, 0.99, 1.8}
	for i := 0; i < 100000; i++ {
		for _, v := range array {
			math.Log(v)
		}
	}
	fmt.Println(time.Now().Sub(start))
}

func main() {
	a := LR.SoftMaxRegression{}
	a.InitWeights(300000000)

	a.RandomWriteTest()
	fmt.Println("done")
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	<-signalChan
}

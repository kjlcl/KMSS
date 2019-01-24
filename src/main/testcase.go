package main

import (
	"LR"
	"fmt"
	"math"
	"maxent/IIS"
	"os"
	"os/signal"
	"syscall"
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

func main_() {
	a := 0.2
	fmt.Println(a + math.NaN())
}

func waitSignal() {

	fmt.Println("done")
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)

	<-signalChan
}

func main() {
	a := LR.SoftMaxRegression{}
	a.Train(100)
}

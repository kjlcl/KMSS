package main

import "maxent/IIS"

func main() {

	//testSet, yCount := data.ReadMnistCsv("./resource/Mnist/mnist_train.csv")
	//fmt.Println(len(testSet), yCount)

	model := IIS.MaxEntIIS{}
	model.LoadData(
		"./resource/Mnist/mnist_test.csv",
		"./resource/Mnist/mnist_train.csv")

	model.StartTraining(200)
}

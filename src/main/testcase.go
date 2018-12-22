package main

import "maxent/IIS"

func main() {

	model := IIS.MaxEntIIS{}
	model.LoadData(
		"./resource/Mnist/mnist_test.csv",
		"./resource/Mnist/mnist_train.csv")

	model.StartTraining(500)
}

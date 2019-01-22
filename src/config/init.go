package config

import (
	"fmt"
	"os"
	"strings"
)

var (
	config = Config{}
)

type Config struct {
	LogConf     LogConf   `yaml:"log"`
	SoftmaxConf TrainConf `yaml:"softmax"`
	LRConf      TrainConf `yaml:"lr"`
}

type LogConf struct {
	LogLevel string `yaml:"level"`
	LogPath  string `yaml:"path"`
}

type TrainConf struct {
	TrainPath    string  `yaml:"train"`
	TestPath     string  `yaml:"test"`
	FeatureLen   int     `yaml:"featureLen"`
	BatchCount   int     `yaml:"batchCount"`
	LearningRate float32 `yaml:"learningRate"`
	Normal       string  `yaml:"normal"`
	NormalRate   float32 `yaml:"normalRate"`
}

func (logConf *LogConf) updateFileName(logName string) {
	if strings.Index(logConf.LogPath, "%s") != -1 {
		logConf.LogPath = fmt.Sprintf(logConf.LogPath, logName)
	}
}

func init() {
	// Default path
	confPath := "./config/settings_dev_1.yml"
	for i := 1; i < len(os.Args); i++ {
		if hit := strings.HasPrefix(os.Args[i], "--conf="); hit {
			confPath = os.Args[i][7:]
			break
		}
	}
	err := config.LoadConfig(confPath)
	if err != nil {
		panic(err.Error())
	}

	logName := "app"
	for i := 1; i < len(os.Args); i++ {
		if hit := strings.HasPrefix(os.Args[i], "--log="); hit {
			logName = os.Args[i][6:]
			break
		}
	}
	config.LogConf.updateFileName(logName)
}

func GetLogConf() LogConf {
	return config.LogConf
}

func GetLRConf() TrainConf {
	return config.LRConf
}

func GetSoftmaxConf() TrainConf {
	return config.SoftmaxConf
}

package config

import (
	"fmt"
	"gopkg.in/yaml.v2"
	"io/ioutil"
)

func (c *Config) LoadConfig(filePath string) error {
	fmt.Println("load config " + filePath)
	data, err := ioutil.ReadFile(filePath)
	if err == nil {
		err = yaml.Unmarshal(data, c)
	}

	return err
}

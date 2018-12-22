package data

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func ReadMnistCsv(filePath string) (result []*MnistSample, yCount int) {

	yMap := make(map[int]uint8)
	if file, err := os.Open(filePath); err == nil {
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			items := strings.Split(line, ",")
			sample := &MnistSample{}
			if label, err := strconv.Atoi(items[0]); err == nil {
				sample.label = label
				yMap[sample.label] = 1
			}

			sample.dataVec = make([]uint8, len(items)-1)
			for i, value := range items[1:] {
				if di, err := strconv.Atoi(value); err == nil {
					//sample.dataVec[i] = uint8(di)

					// 简化模型，像素点取值映射为0，1
					if di > 128 {
						sample.dataVec[i] = 1
					} else {
						sample.dataVec[i] = 0
					}
				}
			}
			result = append(result, sample)
		}
	}
	yCount = len(yMap)
	return
}

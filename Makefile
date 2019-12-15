
WORKING_DIR := $(shell pwd)
GOPATH=/Users/liang/Works/gopath:${WORKING_DIR}
Flag=CGO_ENABLED=0 GOOS=linux GOARCH=amd64

dep:
	go get github.com/golang/protobuf/proto

build-train-lr:
	@GOPATH=${GOPATH} ${Flag} go build -o train_lr src/main/testcase.go

install:
	scp train_lr works@172.17.104.250:trainning_data/

build: build-train-lr

clean:
	rm train_lr
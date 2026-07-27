.PHONY: build test vet lint check docker

build:
	go build -o fakerock ./cmd/fakerock

test:
	go test ./...

vet:
	go vet ./...

lint:
	golangci-lint run ./...

check: vet lint test

docker:
	docker build -t fakerock .

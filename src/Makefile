
all: attention

attention: 
	nvcc attention.cu -o /app/attention/bin/attention -lcudart -lcublas -lcudnn -diag-suppress 549 -diag-suppress 550 --std c++17

clean:
	rm -rf bin/attention
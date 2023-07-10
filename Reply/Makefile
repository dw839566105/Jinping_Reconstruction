fileno:=$(shell seq 257 289)
dst:$(fileno:%=%.h5)
.PHONY: all

all: dst

%.h5:
	python3 check2.py $*

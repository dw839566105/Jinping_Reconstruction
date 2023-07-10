srcL:=$(wildcard /mnt/stage/Jinping_validation/electron/sphere/dn0/*.root)
dstL:=$(srcL:%.root=%_old.h5)

.PHONY: all
all: $(dstL)

%_old.h5: %.root
	nice -n 19 python3 ConvertTruthMerge.py $^ $@

.DELETE_ON_ERROR:
.SECONDARY:

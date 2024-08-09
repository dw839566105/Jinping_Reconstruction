# probe 源自 douwei 
coeff_PE_temp:=/JNE/coeff/Legendre/Gather/PE/2/80/40.h5
coeff_time_temp:=/JNE/coeff/Legendre/Gather/Time/2/80/10.h5
TimeCalib:=/JNE/Jinping_1ton_Data/CalibData/TimeCalibData/PMTTimeCalib_Run257toRun262.txt
PMTCalib:=/JNE/Jinping_1ton_Data/CalibData/GainCalibData/PMTGainCalib_Run0257toRun0271.txt
PMT:=PMT.txt
MCstep:=2000
reconfiles:=$(patsubst fsmp/%.pq, tvE/%.h5, $(wildcard fsmp/BiPo/run00000257/*.pq))
BlockNum:=3
SRUN:=sudo -u\#35905 srun -g 1 -c 2

all: Fig/BiPo.pdf

# 临时计算暗噪声率，后续再仔细考虑。MonitorRun0257_Run0290.root 来自于
# online@jinping.g.airelinux.org:~/ZLWork/Calibration/OutputFile/MonitorFile/MonitorRun0257_Run0290.root
darknoise.txt: /JNE/calibration/ZLwork/MonitorRun0257_Run0290.root
	time python3 genDark.py -i $< -o $@ -d darkrate.pdf

# 事例重建
tvE/%.h5: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT) darknoise.txt $(TimeCalib)
	mkdir -p $(dir $@)
	$(SRUN) python3 main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) --dark $(word 6, $^) --timecalib $(word 7, $^) -n $(BlockNum) -m $(MCstep) -o $@

tvE_Burn/%.h5: tvE/%.h5
	mkdir -p $(dir $@)
	python3 Burn.py -i $< -o $@ -r 0.5

# 生成 run0257 的 BiPo 事例列表和已有重建结果图
BiPo0257:=/JNE/eternity/Reconstruction/00000257.root
Bi214_0257.txt: $(BiPo0257)
	python3 pick.py -i $^ -o $@ -r run0000257.pdf -c run0000257_r3cut.pdf

# BiPo 能谱和顶点分布
Fig/BiPo.h5: $(BiPo0257) $(reconfiles) Bi214_0257.txt
	mkdir -p $(dir $@)
	python3 Fig/pre_bipo.py -b $(BiPo0257) -r $(reconfiles) -s $(MCstep) -e Bi214_0257.txt -o $@

Fig/BiPo.pdf: Fig/BiPo.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_bipo.py $^ -o $@

# 单事例不同步骤重建结果分布图
Fig/steps/sim/%.pdf: tvE/%.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_step.py $< -o $@ -n 5 -s $(MCstep) --switch OFF --mode sim -t /JNE/resolution/$*.root

Fig/steps/raw/%.pdf: tvE/%.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_step.py $< -o $@ -n 10 -s $(MCstep) --switch ON --mode raw

## 模拟数据：真值与重建对比图 (已经不再兼容，待整理)
# 球内均匀
Fig/sim/ball.h5: $(simreconball) $(simrootball)
	mkdir -p $(dir $@)
	python3 Fig/pre_sim.py -r $(simreconball) -t $(simrootball) -s $(MCstep) -o $@
# z 轴均匀
Fig/sim/pointz.h5: $(simreconz) $(simrootz)
	mkdir -p $(dir $@)
	python3 Fig/pre_sim.py -r $(simreconz) -t $(simrootz) -s $(MCstep) -o $@

Fig/sim/ball.pdf: Fig/sim/ball.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_sim.py $^ -o $@

Fig/sim/pointz.pdf: Fig/sim/pointz.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_pointz.py $^ -o $@

# time profile
profile/%.stat: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT) darknoise.txt $(TimeCalib)
	mkdir -p $(dir $@)
	python3 -m cProfile -o $@ main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) --dark $(word 6, $^) --timecalib $(word 7, $^) -n $(BlockNum) -m $(MCstep) -o $@.h5

line:=$(shell grep -n "def genR(self, vertex, PEt, sum_mode = True)" Detector.py | cut -d ":" -f 1)
lineprofile/%.lprof: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT) darknoise.txt $(TimeCalib) Detector.py
	mkdir -p $(dir $@)
	sed -i '$(line) i\    from line_profiler import LineProfiler\n    @profile' $(word 8, $^)
	kernprof -o $@ -l main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) --dark $(word 6, $^) --timecalib $(word 7, $^) -n $(BlockNum) -m $(MCstep) -o $@.h5
	python -m line_profiler $@ > $@.log
	sed -i '$(line)d' $(word 8, $^)
	sed -i '$(line)d' $(word 8, $^)

rocprofile/%: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT) darknoise.txt $(TimeCalib)
	mkdir -p $@
	rocprof -d rocprofile/$* --hip-trace main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) --dark $(word 6, $^) --timecalib $(word 7, $^) -n $(BlockNum) -m $(MCstep) -o $@.h5
	mv results.* $@/

profile/%.svg: profile/%.stat
	gprof2dot -f pstats $^ | dot -Tsvg -o $@

.DELETE_ON_ERROR:

.SECONDARY:

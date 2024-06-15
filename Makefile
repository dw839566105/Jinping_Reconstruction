evtmax:=1000
energy:=2
Radius:=0.65

scan:=$(shell seq 0.01 0.01 $(Radius)) # different radius
scan_compact:=$(shell seq 0.01 0.01 0.55) $(shell seq 0.55 0.002 0.644) # compact radius for calibration
duplicate:=$(shell seq -w 1 30) # for spherical simulation data
duplicate_v:=$(shell seq -w 60 99) # for validate data
path:=JP_1t
geo:=$(G__JSAPSYS)/Simulation/DetectorStructure/1t

sim: shell point/x point/y point/z ball
recon: recon_shell recon_x recon_y recon_z recon_ball

shell: $(scan_compact:%=$(path)/root/shell/$(energy)/%.root)
point/x: $(scan:%=$(path)/root/point/x/$(energy)/%.root)
point/y: $(scan:%=$(path)/root/point/y/$(energy)/%.root)
point/z: $(scan:%=$(path)/root/point/z/$(energy)/%.root)
ball: $(duplicate:%=$(path)/root/ball/$(energy)/%.root)

ball_v: $(duplicate_v:%=$(path)/root/ball/$(energy)/%.root)
vset:=$(duplicate_v:%=$(path)/concat/ball/$(energy)/%.h5)

recon_shell: $(scan_compact:%=$(path)/recon/shell/$(energy)/%.h5)
recon_x: $(scan:%=$(path)/recon/point/x/$(energy)/%.h5)
recon_y: $(scan:%=$(path)/recon/point/y/$(energy)/%.h5)
recon_z: $(scan:%=Reconresult/root/point/z/$(energy)/%.h5)
recon_ball: $(duplicate:%=$(path)/recon/ball/$(energy)/%.h5)


coeff_PE:=$(path)/coeff/Legendre/Gather/PE/$(energy)/80/40.h5
coeff_time:=$(path)/coeff/Legendre/Gather/Time/$(energy)/80/10.h5
coeff: $(coeff_PE) $(coeff_time)

order_Z = $(shell seq -w 25 5 40)
order_dLeg = $(shell seq -w 20 5 30)
order1 = $(shell seq -w 10 10 90)
order2 = $(shell seq -w 10 5 40)

coeff_Z: $(order_Z:%=$(path)/coeff/Zernike/PE/$(energy)/shell/%.csv)
coeff_dLeg: $(path)/coeff/dLegendre/PE/$(energy)/shell/40/30.h5 $(path)/coeff/dLegendre/PE/$(energy)/shell/60/20.h5
coeff_dLeg_pdf: $(path)/coeff/dLegendre/PE/$(energy)/shell/40/30.pdf $(path)/coeff/dLegendre/PE/$(energy)/shell/60/20.pdf
coeff_dLeg_csv: $(path)/coeff/dLegendre/PE/$(energy)/shell/40/30.csv $(path)/coeff/dLegendre/PE/$(energy)/shell/60/20.csv
coeff_Leg:  $(foreach o1,$(order1),$(foreach o2,$(order2),$(path)/coeff/Legendre/Gather/PE/$(energy)/$(o1)/$(o2).csv $(path)/coeff/Legendre/Gather/Time/$(energy)/$(o1)/$(o2).h5))
coeff_Leg_pdf:  $(foreach o1,$(order1),$(foreach o2,$(order2),$(path)/coeff/Legendre/Gather/PE/$(energy)/$(o1)/$(o2).pdf))

################# Reconstruction ############################
# probe 源自 douwei
coeff_PE_temp:=/JNE/coeff/Legendre/Gather/PE/2/80/40.h5
coeff_time_temp:=/JNE/coeff/Legendre/Gather/Time/2/80/10.h5
TimeCalib:=/JNE/Jinping_1ton_Data/CalibData/TimeCalibData/PMTTimeCalib_Run257toRun262.txt
PMTCalib:=/JNE/Jinping_1ton_Data/CalibData/GainCalibData/PMTGainCalib_Run0257toRun0271.txt
PMT:=PMT.txt
MCstep:=10000
reconfiles:=$(patsubst fsmp/%.pq, tvE/%.h5, $(wildcard fsmp/BiPo/run00000257/*.pq))
simrootball:=$(patsubst fsmp/%.pq, /JNE/resolution/%.root, $(wildcard fsmp/ball/2/*.pq))
simrootz:=$(patsubst fsmp/%.pq, /JNE/resolution/%.root, $(wildcard fsmp/point/z/2/*.pq))
simreconball:=$(patsubst fsmp/%.pq, tvE/%.h5, $(wildcard fsmp/ball/2/*.pq))
simreconz:=$(patsubst fsmp/%.pq, tvE/%.h5, $(wildcard fsmp/point/z/2/*.pq))

# 03 05 文件异常, 暂时去除观察其他事例
simreconball:=$(filter-out tvE/ball/2/03.h5 tvE/ball/2/05.h5, $(simreconball))
simrootball:=$(filter-out /JNE/resolution/ball/2/03.root /JNE/resolution/ball/2/05.root, $(simrootball))
all: Fig/BiPo.pdf

# 事例重建
tvE/%.h5: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT) $(PMTCalib) $(TimeCalib)
	mkdir -p $(dir $@)
	time python3 main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) --dark $(word 6, $^) --timecalib $(word 7, $^) -n 0 -m $(MCstep) -o $@ --record OFF

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
	python3 Fig/plot_step.py $< -o $@ -n 5 -s $(MCstep) --switch OFF --mode sim -t /JNE/resolution/$*.root --record OFF

Fig/steps/raw/%.pdf: tvE/%.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_step.py $< -o $@ -n 10 -s $(MCstep) --switch ON --mode raw --record OFF

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
profile/%.stat: fsmp/%.pq sparsify/%.h5 $(coeff_PE_temp) $(coeff_time_temp) $(PMT)
	mkdir -p $(dir $@)
	python3 -m cProfile -o $@ main.py -f $< --sparsify $(word 2, $^) --pe $(word 3, $^) --time $(word 4, $^) --PMT $(word 5, $^) -n 10 -m $(MCstep) -o profile/$*.h5 --record OFF

profile/%.svg: profile/%.stat
	gprof2dot -f pstats $^ | dot -Tsvg -o $@

################# generate macro files ######################
$(path)/mac/shell/$(energy)/%.mac: Simulation/macro/example_shell.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@inner/$*/; s/@outer/$*/' $^ > $@

$(path)/mac/ball/$(energy)/%.mac: Simulation/macro/example_ball.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@Radius/$(Radius)/' $^ > $@

$(path)/mac/point/x/$(energy)/%.mac: Simulation/macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/$* 0 0/;' $^ > $@

$(path)/mac/point/y/$(energy)/%.mac: Simulation/macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/0 $* 0/;' $^ > $@

$(path)/mac/point/z/$(energy)/%.mac: Simulation/macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/0 0 $*/;' $^ > $@

################# Simulate root files #######################
$(path)/root/%.root: $(path)/mac/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -o $@ > $@.log 2>&1

################# Convert .root to .h5 ######################
$(path)/h5/%.h5: $(path)/root/%.root
	mkdir -p $(dir $@)
	/home/douwei/root2hdf5/build/ConvertSimData $^ $@ > $@.log 2>&1

################# Calculate r, theta basis ##################
$(path)/concat/%.h5: $(path)/h5/%.h5
	mkdir -p $(dir $@)
	python3 Simulation/concat.py $^ -o $@ --pmt PMT.txt > $@.log

############## Different Calib models  ######################
## varying-coefficient method 
# 1st-step fit theta
Lo_PE :=
Lo_Time :=
define Leg_rule1
$(path)/coeff/Legendre/PE/$(energy)/$(1)/$(2).h5: $(path)/concat/shell/$(energy)/$(1).h5
	mkdir -p $$(dir $$@)
	python3 Regression/main_sLG.py -f $$< --order $(2) -o $$@ > $$@.log

$(path)/coeff/Legendre/Time/$(energy)/$(1)/$(2).h5: $(path)/concat/shell/$(energy)/$(1).h5
	mkdir -p $$(dir $$@)
	python3 Regression/main_sLG.py -f $$< --order $(2) --mode time -o $$@ > $$@.log

Lo_PE += $(path)/coeff/Legendre/PE/$(energy)/$(1)/$(2).h5
Lo_Time += $(path)/coeff/Legendre/Time/$(energy)/$(1)/$(2).h5
endef

$(foreach r,$(scan_compact), $(foreach o2,$(order2), $(eval $(call Leg_rule1,$(r),$(o2)))))
# 2nd-step fit r
Lo_PE2 :=
Lo_Time2 :=
define Leg_rule2
$(path)/coeff/Legendre/Gather/PE/$(energy)/$(1)/$(2).h5: $(Lo_PE)
	mkdir -p $$(dir $$@)
	python3 Regression/Gather.py -p $(path)/coeff/Legendre/PE/$(energy)/ -o $$@ --o1 $(2) --o2 $(1)

$(path)/coeff/Legendre/Gather/Time/$(energy)/$(1)/$(2).h5: $(Lo_Time)
	mkdir -p $$(dir $$@)
	python3 Regression/Gather.py -p $(path)/coeff/Legendre/Time/$(energy)/ -o $$@ --o1 $(2) --o2 $(1)

Lo_PE2 := $(path)/coeff/Legendre/Gather/PE/$(energy)/$(1)/$(2).h5
Lo_Time2 := $(path)/coeff/Legendre/Gather/Time/$(energy)/$(1)/$(2).h5
endef

$(foreach o1,$(order1), $(foreach o2,$(order2), $(eval $(call Leg_rule2,$(o1),$(o2)))))

## Zernike basis
$(path)/coeff/Zernike/PE/$(energy)/shell/%.h5:
	mkdir -p $(dir $@)
	python3 Regression/main_Zernike.py -f $(path)/concat/shell --order $* --r_max 0.638 -o $@

$(path)/coeff/Zernike/Time/$(energy)/shell/%.h5:
	mkdir -p $(dir $@)
	python3 Regression/main_Zernike.py -f $(path)/concat/shell --mode time --order $* --r_max 0.638 -o $@

## double Legendre basis
$(path)/coeff/dLegendre/PE/$(energy)/shell/40/30.h5:
	mkdir -p $(dir $@)
	python3 Regression/main_dLeg.py -f $(path)/concat/shell/$(energy)/ --order 40 30 --r_max 0.638 -o $@ > $@.log

$(path)/coeff/dLegendre/PE/$(energy)/shell/60/20.h5:
	mkdir -p $(dir $@)
	python3 Regression/main_dLeg.py -f $(path)/concat/shell/$(energy)/ --order 60 20 --r_max 0.638 -o $@ > $@.log

############## Validate  ######################
%.csv: %.h5 $(vset)
	python3 Draw/validate.py --pe $< --vset $(wordlist 2, 100, $^) -o $@ 

%.pdf: %.h5 $(vset)
	python3 Draw/plot_probe.py --pe $< --time $(path)/coeff/Legendre/Gather/Time/$(energy)/80/10.h5 -o $@ --vset $(wordlist 2, 100, $^)

.DELETE_ON_ERROR:

.SECONDARY:

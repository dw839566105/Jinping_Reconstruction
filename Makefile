evtmax:=1000
energy:=2
Radius:=0.65

scan:=$(shell seq 0.01 0.01 $(Radius)) # different radius
scan_compact:=$(shell seq 0.01 0.01 0.55) $(shell seq 0.55 0.002 0.644) # compact radius for calibration
duplicate:=$(shell seq -w 1 30) # for spherical simulation data
duplicate_v:=$(shell seq -w 60 99) # for validate data
path:=/mnt/stage/douwei/JP_1t_github
geo:=Simulation/DetectorStructure/1t

.PHONY: all
all: recon

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
recon_z: $(scan:%=$(path)/recon/point/z/$(energy)/%.h5)
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
coeff_PE_temp:=coeff/Legendre/Gather/PE/2/80/40.h5
coeff_time_temp:=coeff/Legendre/Gather/Time/2/80/10.h5
Reconresult/%.h5: charge/%.parquet $(coeff_PE_temp) $(coeff_time_temp)
	mkdir -p $(dir $@)
	time python3 Reconstruction/main.py -f $< --pe $(word 2, $^) --time $(word 3, $^) -o $@ > $@.log

Reconresult/%_stack.h5: charge/%.parquet $(coeff_PE_temp) $(coeff_time_temp)
	mkdir -p $(dir $@)
	time python3 Reconstruction/main_stack.py -f $< --pe $(word 2, $^) --time $(word 3, $^) -o $@ > $@.log

# 生成 run0257 的 BiPo 事例列表和已有重建结果图
collect/Bi214_0257.csv: collect/00000257.root
	ln -s ../Fig/fit.py .
	python3 collect/pick.py -i $^ -o $@ -r collect/run0000257.pdf -c collect/run0000257_r3cut.pdf

# gelman-rubin 收敛检验
MutiRecon/%.h5: charge/%.parquet $(coeff_PE_temp) $(coeff_time_temp)
	mkdir -p $(dir $@)
	time python3 Reconstruction/MutiRecon.py -f $< --pe $(word 2, $^) --time $(word 3, $^) -o $@ --num 10 --event 598515

Gelman/%.h5: MutiRecon/%*.h5
	mkdir -p $(dir $@)
	python3 Gelman/gelman.py -i $^ -o $@

Gelman/%.parquet: Gelman/%.h5
	mkdir -p $(dir $@)
	./Gelman/gelman.R -i $< -o $@

Gelmanget: MutiRecon/0/BiPo/run00002000/1.h5 Gelman/0/BiPo/run00002000/1.h5
recon257get: Reconresult/0/BiPo/run00000257/0.h5 Reconresult/0/BiPo/run00000257/1.h5 Reconresult/0/BiPo/run00000257/2.h5

# 能谱和顶点分布
Fig/events/%.h5: Reconresult/%/*.h5
	mkdir -p $(dir $@)
	python3 Fig/pre_plot.py $^ -o $@ -e ../Bi214_0257.txt

Fig/events/%.pdf: Fig/events/%.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_recon.py $^ -o $@

# 单事例不同步骤重建结果分布图
Fig/steps/%.pdf: Reconresult/%.h5
	mkdir -p $(dir $@)
	python3 Fig/plot_recon_singleEvent.py $^ -o $@ -n 50

Fig/gelman/%.pdf: Gelman/%.parquet
	mkdir -p $(dir $@)
	python3 Fig/plot_gelman.py $^ -o $@ -n 50

# time profile
profile/%.stat: charge/%.parquet $(coeff_PE_temp) $(coeff_time_temp)
	mkdir -p $(dir $@)
	python3 -m cProfile -o foo_profile.stat Reconstruction/main.py -f $< --pe $(word 2, $^) --time $(word 3, $^) -o profile/$*.h5

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

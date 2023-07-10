evtmax:=1000
energy:=2
Radius:=0.65
duplicate:=$(shell seq -w 1 30)
order1 = $(shell seq -w 25 5 40)
o1 = $(shell seq -w 20 5 35)
o2 = $(shell seq -w 20 5 30)
scan:=$(shell seq 0 0.01 $(Radius))
scan1:=$(shell seq 0.01 0.01 0.55) $(shell seq 0.55 0.002 0.644)
v:=$(shell seq 50 1 60)
path:=/mnt/stage/douwei/JP_1t_paper
geo:=DetectorStructure/1t

.PHONY: all
all: coeff

targets:=shell point/x point/y point/z
coef:=PE Time

# Legendre fit
# $(1) is the radius
# $(2) is the order

Lo_PE :=
Lo_Time :=
define Legendre_rule
$(path)/coeff/Legendre/PE/$(energy)/$(1)/$(2).h5: $(path)/concat/shell/$(energy)/$(1).h5
	mkdir -p $$(dir $$@)
	python3 calib/main_sLG.py -f $$< --order $(2) -o $$@ > $$@.log

$(path)/coeff/Legendre/Time/$(energy)/$(1)/$(2).h5: $(path)/concat/shell/$(energy)/$(1).h5
	mkdir -p $$(dir $$@)
	python3 calib/main_sLG.py -f $$< --order $(2) --mode time -o $$@ > $$@.log

Lo_PE += $(path)/coeff/Legendre/PE/$(energy)/$(1)/$(2).h5
Lo_Time += $(path)/coeff/Legendre/Time/$(energy)/$(1)/$(2).h5
endef

define fn
$(foreach o, $(order1), $(eval $(call Legendre_rule,$(1),$(o))))
endef

$(foreach ra, $(scan1), $(eval $(call fn,$(ra))))
Ld: $(Lo_PE) $(Lo_Time)

$(path)/coeff/Legendre/Gather/PE/$(energy)/$(go)/%.h5: $(Lo_PE)
	mkdir -p $(dir $@)
	python3 calib/Gather.py -p $(path)/coeff/Legendre/PE/$(energy)/ -o $@ --o1 $* --o2 $(go)

$(path)/coeff/Legendre/Gather/Time/$(energy)/$(go)/%.h5: $(Lo_Time)
	mkdir -p $(dir $@)
	python3 calib/Gather.py -p $(path)/coeff/Legendre/Time/$(energy)/ -o $@ --o1 $* --o2 $(go)

# double Legendre fit
# $(1) is the 1st order
# $(2) is the 2nd order

ifeq ("x","y")
dLo_PE:=
dLo_Time:=
define dLegendre_rule
$(path)/coeff/dLegendre/PE/$(energy)/$(1)/$(2).h5: ball
	mkdir -p $$(dir $$@)
	python3 calib/main_dLG.py -f $(path)/concat/ball/$(energy)/ --order $(1) $(2) -o $$@ > $$@.log

$(path)/coeff/dLegendre/Time/$(energy)/$(1)/$(2).h5: ball
	mkdir -p $$(dir $$@)
	python3 calib/main_dLG.py -f $(path)/concat/ball/$(energy)/ --order $(1) $(2) --mode time -o $$@ > $$@.log
dLo_PE += coeff/dLegendre/PE/$(energy)/$(1)/$(2).h5
dLo_Time += coeff/dLegendre/Time/$(energy)/$(1)/$(2).h5
endef

define dfn
$(foreach o, $(o1), $(eval $(call dLegendre_rule,$(o),$(1))))
endef

$(foreach o, $(o1), $(eval $(call dfn,$(o))))
dLd: $(dLo_PE) $(dLo_Time)
endif

dLo_PE:=
dLo_Time:=
define dLegendre_rule
$(path)/coeff/dLegendre/PE/$(energy)/shell/$(1)/$(2).h5:
	mkdir -p $$(dir $$@)
	python3 calib/main_dLG_shell_new.py -f $(path)/concat/shell/$(energy)/ --order $(1) $(2) --r_max 0.638 -o $$@ > $$@.log

$(path)/coeff/dLegendre/Time/$(energy)/shell/$(1)/$(2).h5:
	mkdir -p $$(dir $$@)
	python3 calib/main_dLG_shell.py -f $(path)/concat/shell/$(energy)/ --order $(1) $(2) --mode time --r_max 0.638 -o $$@ > $$@.log
dLo_PE += $(path)/coeff/dLegendre/PE/$(energy)/shell/$(1)/$(2).h5
dLo_Time += $(path)/coeff/dLegendre/Time/$(energy)/shell/$(1)/$(2).h5
endef

define dfn
$(foreach o, $(o1), $(eval $(call dLegendre_rule,$(o),$(1))))
endef

$(foreach o, $(o2), $(eval $(call dfn,$(o))))
dLd: $(dLo_PE) $(dLo_Time)

recon: $(foreach t, $(targets), $(scan:%=$(path)/recon/$(t)/$(energy)/%.h5))
ball: $(duplicate:%=$(path)/concat/ball/$(energy)/%.h5)
sim_0.26: $(duplicate:%=$(path)/root/shell/0.26/$(energy)/%.root)
recon_0.26: $(duplicate:%=$(path)/recon/shell/0.26/$(energy)/%.h5)
recon_close_0.26: $(duplicate:%=$(path)/recon_close/shell/0.26/$(energy)/%.h5)
coeff: $(order1:%=coeff/Zernike/PE/$(energy)/%.h5) $(order1:%=coeff/Zernike/Time/$(energy)/%.h5)
coeff_Zs: $(order1:%=$(path)/coeff/Zernike/PE/$(energy)/shell/%.h5)
coeff_Ls: $(o1:%=$(path)/coeff/dLegendre/PE/$(energy)/shell/%/30.h5)
coeff_Ld: $(order1:%=$(path)/coeff/Legendre/Gather/PE/$(energy)/$(go)/%.h5) $(order1:%=$(path)/coeff/Legendre/Gather/Time/$(energy)/$(go)/%.h5)
track: $(path)/track/shell/$(energy)/0.26.h5 $(path)/track/shell/$(energy)/0.62.h5 $(path)/track/point/z/$(energy)/0.62.h5

rep:=$(shell seq -w 1 1 300)
add1: $(rep:%=$(path)/add/shell/$(energy)/0.26/%.h5)
add2: $(rep:%=$(path)/add/shell/$(energy)/0.60/%.concat)

vset: $(v:%=$(path)/concat/ball/$(energy)/%.h5)
# validate: $(order1:%=coeff/Zernike/PE/$(energy)/%.csv)
validate1: $(order1:%=$(path)/coeff/Legendre/Gather/PE/$(energy)/$(go)/%.csv)
validateZ: $(order1:%=$(path)/coeff/Zernike/PE/$(energy)/shell/%.csv)
validate2: $(dLo_PE:%.h5=%.csv)

#############################################################
################# generate macro files ######################
#############################################################

$(path)/mac/ball/$(energy)/%.mac: macro/example_ball.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@Radius/$(Radius)/' $^ > $@

$(path)/mac/shell/$(energy)/%.mac: macro/example_shell.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@inner/$*/; s/@outer/$*/' $^ > $@

$(path)/mac/point/x/$(energy)/%.mac: macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/$* 0 0/;' $^ > $@

$(path)/mac/point/y/$(energy)/%.mac: macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/0 $* 0/;' $^ > $@

$(path)/mac/point/z/$(energy)/%.mac: macro/example_point.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@x @y @z/0 0 $*/;' $^ > $@

#############################################################
############### Generate h5 files in sim ####################
#############################################################

$(path)/root/%.root: $(path)/mac/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -o $@ > $@.log 2>&1

$(path)/mac/shell/0.26/$(energy)/%.mac: macro/example_shell_026.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@inner/0.26/; s/@outer/0.26/' $^ > $@

$(path)/root/shell/0.26/$(energy)/%.root: $(path)/mac/shell/0.26/energy/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -o $@ > $@.log 2>&1

######################################################################
$(path)/add/shell/$(energy)/0.26/%.mac: macro/example_shell_026.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@inner/0.26/; s/@outer/0.26/' $^ > $@

$(path)/add/shell/$(energy)/0.26/%.root: $(path)/add/shell/$(energy)/0.26/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -o $@ -t ON -tV ON > $@.log 2>&1

$(path)/add/shell/$(energy)/0.26/%.h5: $(path)/add/shell/$(energy)/0.26/%.root
	mkdir -p $(dir $@)
	/usr/bin/ConvertSimData $^ $@ > $@.log 2>&1


$(path)/add/shell/$(energy)/0.60/%.mac: macro/example_shell_026.mac
	mkdir -p $(dir $@)
	sed -e 's/@seed/$*/; s/@particle/e-/; s/@evtmax/$(evtmax)/; s/@energy/$(energy)/; s/@inner/0.60/; s/@outer/0.60/' $^ > $@

$(path)/add/shell/$(energy)/0.60/%.root: $(path)/add/shell/$(energy)/0.60/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -o $@ > $@.log 2>&1

$(path)/add/shell/$(energy)/0.60/%.h5: $(path)/add/shell/$(energy)/0.60/%.root
	mkdir -p $(dir $@)
	/usr/bin/ConvertSimData $^ $@ > $@.log 2>&1
	
$(path)/add/shell/$(energy)/0.60/%.concat: $(path)/add/shell/$(energy)/0.60/%.h5
	mkdir -p $(dir $@)
	python3 concat.py $^ -o $@ --pmt PMT.txt > $@.log

##############################################################
$(path)/recon/shell/0.26/$(energy)/%.h5: $(path)/root/shell/0.26/$(energy)/%.root
	mkdir -p $(dir $@)
	python3 Recon/main.py -f $< --pe $(path)/coeff/Legendre/Gather/PE/$(energy)/40/25.h5 --time $(path)/coeff/Legendre/Gather/Time/$(energy)/40/20.h5 -o $@ > $@.log



$(path)/recon/shell/0.26/$(energy)/%.h5: $(path)/root/shell/0.26/$(energy)/%.root
	mkdir -p $(dir $@)
	python3 Recon/main.py -f $< --pe $(path)/coeff/Legendre/Gather/PE/$(energy)/40/25.h5 --time $(path)/coeff/Legendre/Gather/Time/$(energy)/40/20.h5 -o $@ > $@.log

$(path)/recon_close/shell/0.26/$(energy)/%.h5: $(path)/root/shell/0.26/$(energy)/%.root
	mkdir -p $(dir $@)
	python3 Recon/main_close.py -f $< --pe $(path)/coeff/Legendre/Gather/PE/$(energy)/40/25.h5 --time $(path)/coeff/Legendre/Gather/Time/$(energy)/40/20.h5 -o $@ > $@.log

$(path)/track/%.root: $(path)/mac/%.mac
	mkdir -p $(dir $@)
	JPSim -n OFF -e ON -g $(geo) -m $^ -dn 0 -gpsFixTime ON -t ON -o $@ > $@.log 2>&1

$(path)/track/%.h5: $(path)/track/%.root
	mkdir -p $(dir $@)
	/usr/bin/ConvertSimData $^ $@ > $@.log 2>&1

$(path)/h5/%.h5: $(path)/root/%.root
	mkdir -p $(dir $@)
	/usr/bin/ConvertSimData $^ $@ > $@.log 2>&1

$(path)/concat/%.h5: $(path)/h5/%.h5
	mkdir -p $(dir $@)
	python3 concat.py $^ -o $@ --pmt PMT.txt > $@.log

coeff/Zernike/PE/$(energy)/%.h5: ball
	mkdir -p $(dir $@)
	python3 calib/main.py -f $(path)/concat/ball/ --order $* --r_max $(Radius) -o $@

$(path)/coeff/Zernike/PE/$(energy)/shell/%.h5:
	mkdir -p $(dir $@)
	python3 calib/main_shell.py -f $(path)/concat/shell --order $* --r_max 0.638 -o $@

coeff/Zernike/Time/$(energy)/%.h5: ball
	mkdir -p $(dir $@)
	python3.8 calib/main.py -f $(path)/concat/ball/ --mode time --order $* --r_max $(Radius) -o $@

$(path)/recon/%.h5: $(path)/root/%.root coeff_Ld
	mkdir -p $(dir $@)
	python3 Recon/main.py -f $< --pe $(path)/coeff/Legendre/Gather/PE/$(energy)/40/25.h5 --time $(path)/coeff/Legendre/Gather/Time/$(energy)/40/20.h5 -o $@ > $@.log

%.csv: %.h5 vset
	python3 draw/validate.py --pe $< --time coeff/Zernike/Time/15.h5 -o $@

.DELETE_ON_ERROR:
.SECONDARY:

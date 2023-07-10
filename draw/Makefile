o1 = $(shell seq -w 1.01 0.01 2.00)
o2 = $(shell seq -w 5 1 50)

Ls:=
define LOG_rule
log/$(1)/$(2).log:
	mkdir -p $$(dir $$@)
	python3.8 cos_dist.py --R $(1) --N $(2) > $$@

Ls += log/$(1)/$(2).log
endef

define fn
$(foreach o, $(o2), $(eval $(call LOG_rule,$(1),$(o))))
endef

$(foreach o, $(o1), $(eval $(call fn,$(o))))

Ld: $(Ls)

.DELETE_ON_ERROR:
.SECONDARY:
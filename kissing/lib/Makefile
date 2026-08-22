# Build the C tools.  The Riesz optimiser needs OpenBLAS + OpenMP; the rest
# match the historical gcc -O2 -lm command line.  Prefer a system OpenBLAS,
# then fall back to scipy-openblas32 (whose public symbols are prefixed).
#
# The BLAS probe below must never abort at parse time: mis8, clique, gclique,
# addable, opt, opt2 and signmis do not link BLAS, and neither does `clean`.
# Missing OpenBLAS is reported from the Riesz recipes only.
CC = gcc
OMPFLAGS ?= -O3 -fopenmp
CFLAGS ?= -O2
LDLIBS ?= -lm

OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)

ifeq ($(strip $(OPENBLAS_LIBS)),)
SCIPY_OPENBLAS_INC := $(shell python3 -c "import scipy_openblas32 as s; print(s.get_include_dir())" 2>/dev/null)
SCIPY_OPENBLAS_LIB := $(shell python3 -c "import scipy_openblas32 as s; print(s.get_lib_dir())" 2>/dev/null)
ifeq ($(strip $(SCIPY_OPENBLAS_LIB)),)
BLAS_MISSING := 1
else
RIESZ_CFLAGS := -I$(SCIPY_OPENBLAS_INC) -include cblas.h \
	-Dcblas_dgemm=scipy_cblas_dgemm -Dcblas_dscal=scipy_cblas_dscal \
	-Dcblas_ddot=scipy_cblas_ddot -Dcblas_daxpy=scipy_cblas_daxpy \
	-Dcblas_dcopy=scipy_cblas_dcopy -Dcblas_dnrm2=scipy_cblas_dnrm2 \
	-Dopenblas_set_num_threads=scipy_openblas_set_num_threads
LIBS_RIESZ := -L$(SCIPY_OPENBLAS_LIB) -Wl,-rpath,$(SCIPY_OPENBLAS_LIB) -lscipy_openblas -lm
endif
else
RIESZ_CFLAGS := $(OPENBLAS_CFLAGS)
LIBS_RIESZ := $(OPENBLAS_LIBS) -lm
endif

define require_blas
	@if [ -n "$(BLAS_MISSING)" ]; then \
		echo "OpenBLAS not found.  Install a system OpenBLAS (pkg-config openblas)" >&2; \
		echo "or run: python3 -m pip install scipy-openblas32" >&2; \
		echo "The BLAS-free verification tools still build: make verify-tools" >&2; \
		exit 1; \
	fi
endef

.PHONY: all riesz-tools verify-tools blas-check clean

all: riesz-tools verify-tools

riesz-tools: riesz riesz2

# Reports whether the Riesz targets can be built, without building anything.
blas-check:
	$(require_blas)
	@echo "OpenBLAS OK"

riesz: riesz.c
	$(require_blas)
	$(CC) $(OMPFLAGS) $(RIESZ_CFLAGS) -o $@ $< $(LIBS_RIESZ)

# riesz2 is the name every legacy script invokes; it is the same program.
riesz2: riesz
	cp -f riesz riesz2

verify-tools: mis8 clique gclique addable opt opt2 signmis

mis8: mis8.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
clique: clique.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
gclique: gclique.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
addable: addable.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
opt: opt.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
opt2: opt
	cp -f opt opt2
signmis: signmis.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f riesz riesz2

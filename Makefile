CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=

BUILD_DIR := build
SRC_DIR := src

ARCH := $(shell uname -m)
ASM_SOURCES :=
ifeq ($(ARCH),arm64)
ASM_SOURCES += $(SRC_DIR)/simple_blas_arm64_kernel.S
endif
ifeq ($(ARCH),aarch64)
ASM_SOURCES += $(SRC_DIR)/simple_blas_arm64_kernel.S
endif

C_SOURCES := $(SRC_DIR)/simple_blas.c $(SRC_DIR)/benchmark.c
SOURCES := $(C_SOURCES) $(ASM_SOURCES)
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o))) \
           $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.S=.o)))

.PHONY: all clean run

all: bench

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSIMPLE_BLAS_NO_ALIAS -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.S | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

bench: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

run: bench
	./bench

clean:
	rm -rf $(BUILD_DIR) bench

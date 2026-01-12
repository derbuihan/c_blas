CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=

BUILD_DIR := build
SRC_DIR := src

SOURCES := $(SRC_DIR)/simple_blas.c $(SRC_DIR)/benchmark.c
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(SOURCES:.c=.o)))

.PHONY: all clean run

all: bench

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSIMPLE_BLAS_NO_ALIAS -c $< -o $@

bench: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

run: bench
	./bench

clean:
	rm -rf $(BUILD_DIR) bench

CXX ?= g++
MPICXX ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17
FEDERATED_DEFINES ?=

SRC_DIR := src
CENTRALISED_BIN := $(SRC_DIR)/centralised
FEDERATED_BIN := $(SRC_DIR)/federated

.PHONY: all centralised federated clean

all: centralised federated

centralised: $(CENTRALISED_BIN)

federated: $(FEDERATED_BIN)

$(CENTRALISED_BIN): $(SRC_DIR)/centralised_main.cpp $(SRC_DIR)/federated_model.h $(SRC_DIR)/csv_utils.h
	$(CXX) $(CXXFLAGS) $< -o $@

$(FEDERATED_BIN): $(SRC_DIR)/federated_main.cpp $(SRC_DIR)/federated_model.h $(SRC_DIR)/data_distribution.h $(SRC_DIR)/csv_utils.h
	$(MPICXX) $(CXXFLAGS) $(FEDERATED_DEFINES) $< -o $@

clean:
	rm -f $(CENTRALISED_BIN) $(FEDERATED_BIN)

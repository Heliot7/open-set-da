#LIBMR_DIR should be absolute path, not relative (no ..)                                                                                                          
LIBMR_DIR = D:/PhD/Code/Research/TwoStepDetection/classifiers/SVM/libsvm-openset/libMR

ifeq ($(OS),Windows_NT)
    CFLAGS += -D WIN32
    ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
        CFLAGS += -D AMD64
	LIBSUFFIX = .dll
    endif
    ifeq ($(PROCESSOR_ARCHITECTURE),x86)
        CFLAGS += -D IA32
	LIBSUFFIX = .dll
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        CFLAGS += -D LINUX -Wl,-rpath=$(LIBMR_DIR)/libMR/build/libMR/
	LIBSUFFIX = .so
    endif
    ifeq ($(UNAME_S),Darwin)
        CFLAGS += -D OSX
	LIBSUFFIX = .dylib
    endif
endif

CXX ?= g++ -g
#CFLAGS = -Wall -Wconversion -g   

LIBMR_LIB = $(LIBMR_DIR)/libMR/build/libMR/libMR$(LIBSUFFIX)
#LIBMR_LIB = build/libMR/libMR.so

CFLAGS += -Wall  -g   -I $(LIBMR_DIR)/libMR 
CFLAGS += -O2 -fPIC
SHVER = 2

all: svm-train svm-predict svm-scale $(LIBMR_LIB)

$(LIBMR_LIB): $(LIBMR_DIR)/libMR/MetaRecognition.h $(LIBMR_DIR)/libMR/MetaRecognition.cpp
	mkdir -p $(LIBMR_DIR)/libMR/build
	cd  $(LIBMR_DIR)/libMR/build; cmake -DCMAKE_BUILD_TYPE=Debug $(LIBMR_DIR); make



lib: svm.o $(LIBMR_LIB)
	$(CXX) -shared -dynamiclib svm.o $(LIBMR_LIB) -o libsvm.so.$(SHVER) 

svm-predict: svm-predict.cpp svm.o $(LIBMR_LIB)
	$(CXX) $(CFLAGS) svm-predict.cpp svm.o $(LIBMR_LIB)  -o svm-predict -lm
svm-train: svm-train.cpp svm.o $(LIBMR_LIB)
	$(CXX) $(CFLAGS) svm-train.cpp svm.o $(LIBMR_LIB) -o svm-train -lm
svm-scale: svm-scale.c $(LIBMR_LIB)
	$(CXX) $(CFLAGS) svm-scale.c $(LIBMR_LIB) -o svm-scale
svm.o: svm.cpp svm.h 
	$(CXX) $(CFLAGS) -c svm.cpp 
clean:
	rm -fr *~ svm.o svm-train svm-predict svm-scale libsvm.so.$(SHVER) $(LIBMR_DIR)/libMR/build

% This make.m is used under Windows

% add -largeArrayDims on 64-bit machines

mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims -I..\ -L..\libMR\ -lMR -c svm_model_matlab.cpp
mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims -I..\ -c ..\svm.cpp
mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims -I..\ -L..\libMR\ -lMR svmtrain_open.cpp svm.obj svm_model_matlab.obj
mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims -I..\ -L..\libMR\ -lMR svmpredict_open.cpp svm.obj svm_model_matlab.obj
mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims libsvmread.cpp
mex -O CFLAGS="\$CFLAGS -std=c99" COMPFLAGS="/D CV_OMP /openmp $COMPFLAGS" -largeArrayDims libsvmwrite.cpp

all : src/Main.cpp
	gcc -O3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_filesystem -lboost_system -o DetectorDeMonumentos src/oi.cpp
	
debug : src/Main.cpp
	gcc -O3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_filesystem -lboost_system -o DetectorDeMonumentos -g src/oi.cpp 

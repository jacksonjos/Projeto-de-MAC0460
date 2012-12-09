all : src/oi.cpp
	gcc -O3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_filesystem -lboost_system -o oi src/oi.cpp
	
debug : src/oi.cpp
	gcc -O3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_filesystem -lboost_system -o oi -g src/oi.cpp 

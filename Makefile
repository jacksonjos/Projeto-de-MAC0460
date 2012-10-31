bowdemo : src/Main.cpp
	gcc -O3 `pkg-config --cflags opencv` `pkg-config --libs opencv` -lboost_filesystem -lboost_system -o DetectorDeMonumetos src/Main.cpp

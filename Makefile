CC:=g++
LDFLAGS:=$(shell pkg-config --cflags --libs /usr/local/Cellar/opencv/4.2.0_1/lib/pkgconfig/opencv4.pc)
CFLAGS:=--std=c++14
SRC_FILES:=$(wildcard ./*.cpp)
OBJ_FILES:=$(patsubst %.cpp,obj/%.o,$(SRC_FILES))

steganography: $(OBJ_FILES)
	$(CC) $(LDFLAGS) -o $@ $^

obj/%.o: %.cpp
	$(CC) $(LDFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm steganography
	rm obj/*.o

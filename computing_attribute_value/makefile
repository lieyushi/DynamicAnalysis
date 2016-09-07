CC = g++

CFLAGS = -g -Wall

TARGET = FTLE

RM = rm

RM_F = rm -f

STD = -std=c++11

MP = -fopenmp

DICTORY = -I/usr/local/include

SOURCE = Block.cpp main.cpp

HEADER = Block.h

OBJECT = $(SOURCE:.cpp = .o)

all: $(SOURCE) $(TARGET)

clean: 
	$(RM) $(TARGET)
	$(RM_F) Attribute *.txt
	$(RM_F) FTLE\ *.vtk

$(TARGET): $(OBJECT)
	$(CC) $(OBJECT) -o $@ $(MP)

.cpp.o: $(HEADER)
	$(CC) $(STD) $(CFLAGS) $(DICTORY) $(MP) $< -c -o $@



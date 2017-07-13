CC = g++

CFLAGS = -Wall

TARGET = ftle

RM = rm

RM_F = rm -f

STD = -std=c++11

MP = -fopenmp

DICTORY = -I/usr/local/include

SOURCE = main.cpp Block.cpp FTLE_Calculator.cpp ABC_flow.cpp

HEADER = Block.h FTLE_Calculator.h ABC_flow.h 

OBJECT = $(SOURCE:.cpp = .o)

all: $(SOURCE) $(TARGET)

clean: 
	$(RM) $(TARGET)
	$(RM_F) FTLE *.txt
	$(RM_F) frame *.vtk
	$(RM_F) README

$(TARGET): $(OBJECT)
	$(CC) $(OBJECT) -o $@ $(MP)

.cpp.o: $(HEADER)
	$(CC) $(STD) $(CFLAGS) $(DICTORY) $(MP) $< -c -o $@



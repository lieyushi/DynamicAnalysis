CC = g++

CFLAGS = -g -Wall

TARGET = pbf

RM = rm

RM-F = rm -f

STD = -std=c++98

MP = -fopenmp

DICTORY = -I/usr/local/include

SOURCE = Block.cpp Grid.cpp Particle.cpp SPH.cpp main.cpp Kernel.cpp SurfaceTension.cpp

HEADER = Block.h Grid.h Particle.h SPH.h Kernel.h SurfaceTension.h

OBJECT = $(SOURCE:.cpp = .o)

all: $(SOURCE) $(TARGET)

clean: 
	$(RM) $(TARGET) 
	$(RM-F) Frame *.vtk frame *.txt

$(TARGET): $(OBJECT)
	$(CC) $(OBJECT) -o $@ $(MP)

.cpp.o: $(HEADER)
	$(CC) $(CFLAGS) $(DICTORY) $(MP) $< -c -o $@



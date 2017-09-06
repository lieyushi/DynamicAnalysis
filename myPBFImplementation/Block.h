#pragma once
#include <cassert>
class Block
{
public:
	int* inside;
	static int max_size;
	int size;

	Block();
	Block(const Block&);
	~Block();
	void add(const int& index);
};


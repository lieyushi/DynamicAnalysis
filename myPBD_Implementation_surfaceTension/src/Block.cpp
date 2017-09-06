#include "Block.h"

int Block::max_size = 40;

Block::Block()
{
	inside = new int[max_size];
	size = 0;
}

Block::Block(const Block&)
{
	inside = new int[max_size];
	size = 0;
}

Block::~Block()
{
	delete[] inside;
	size = 0;
}

void Block::add(const int& index)
{
	assert(size < max_size);
	inside[size++] = index;
}

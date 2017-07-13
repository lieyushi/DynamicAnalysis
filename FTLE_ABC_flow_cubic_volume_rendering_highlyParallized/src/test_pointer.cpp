#include<iostream>
#include<cassert>
#include<vector>
using namespace std;

class Block
{
public:
	int *inside;
	const static int max = 100;
	int size;

	Block();
	~Block();
	Block(const Block&);
	void add(const int& index);
};


Block::Block()
{
	inside = new int[max];
	cout << "Created!" << endl;
	size = 0;
}

Block::~Block()
{
	delete[] inside;
	cout << "Deleted!" << endl;
	size = 0;
}

Block::Block(const Block&)
{
	inside = new int[max];
	cout << "Created!" << endl;
	size = 0;
}

void Block::add(const int& index)
{
	assert(size < max);
	inside[size++] = index;
}

void change(float *temp, const int& size)
{
	for (int i = 0; i < size; ++i)
	{
		temp[i] += i;
	}
}

void display(float *temp, const int& size)
{
	for (int i = 0; i < size; ++i)
	{
		cout << temp[i] << " ";
	}
	cout << endl;
}

int main()
{
	vector<Block> av(10);

	float a[4] = {0};
	display(a, 4);
	change(a, 4);
	display(a, 4);
	float *b = &a[1];
	cout << b[0] << " " << b[1] << " " << b[2] << endl;

	return 0;
}
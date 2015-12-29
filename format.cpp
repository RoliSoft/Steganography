#include "format.h"

using namespace std;

bool Format::Data::istty = false;

#ifdef _WIN32
HANDLE Format::Data::stdHwd;
CONSOLE_SCREEN_BUFFER_INFO Format::Data::bufInf;
WORD Format::Data::defColor;
WORD Format::Data::curColor;
WORD Format::Data::curStyle;
#endif

void Format::Init()
{
	Data::istty = isatty(fileno(stdout)) != 0;

	if (!Data::istty)
	{
		return;
	}

#ifdef _WIN32

	Data::stdHwd = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(Data::stdHwd, &Data::bufInf);
	Data::defColor = Data::bufInf.wAttributes;

#endif
}

ostream& Format::operator<<(ostream& os, ColorCode code)
{
	if (!Data::istty)
	{
		return os;
	}

#ifdef _WIN32

	if (code == ColorCode::Default)
	{
		Data::curColor = Data::defColor;
	}
	else
	{
		Data::curColor = WORD(code);
	}

	SetConsoleTextAttribute(Data::stdHwd, Data::curColor | Data::curStyle);

#elif __unix__

	os << "\033[" << static_cast<int>(code) << "m";

#endif

	return os;
}

ostream& Format::operator<<(ostream& os, StyleCode code)
{
	if (!Data::istty)
	{
		return os;
	}

#ifdef _WIN32

	if (code == StyleCode::Normal)
	{
		Data::curStyle = 0;
	}
	else
	{
		Data::curStyle |= WORD(code);
	}

	SetConsoleTextAttribute(Data::stdHwd, Data::curColor | Data::curStyle);

#elif __unix__

	os << "\033[" << static_cast<int>(code) << "m";

#endif

	return os;
}

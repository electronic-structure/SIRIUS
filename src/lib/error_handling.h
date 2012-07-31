#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__

void error(const char* file_name, int line_number, const char* message)
{
    printf("Fatal error at line %i of file %s \n", line_number, file_name);
    printf("%s\n", message);
    exit(0);
}

void error(const char* file_name, int line_number, const std::string& message)
{
    error(file_name, line_number, message.c_str());
}

void error(const char* file_name, int line_number, const std::stringstream& message)
{
    error(file_name, line_number, message.str().c_str());
}

#endif // __ERROR_HANDLING_H__


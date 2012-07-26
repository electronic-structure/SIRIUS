#include "error_handling.h"

void error(const char* file_name, int line_number, const char* message)
{
    printf("Fatal error at line %i of file %s \n", line_number, file_name);
    printf("%s\n", message);
    exit(0);
}

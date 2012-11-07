#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__

/*
    There are two types of errors: fatal and non-fatal. A non-fatal error is a warining. Errors
    can be local (happened on a single MPI rank, for example matrix diagonalization failed on some node) 
    and global (happened globally, for example charge density is incorrect after k-point summation).
    Global errors are reported by a single (root) MPI rank. Local errors are reported by each failed MPI rank.

    Examples:

        error(__FILE__, __LINE__, message, global_error | fatal_error); // fatal global error (default) 

        error(__FILE__, __LINE__, message, global_error); // global non-fatal (warning) message 

*/

const int global_error = 1 << 0;
const int fatal_error = 1 << 1;

const int default_error_flags = global_error | fatal_error;
const int default_warning_flags = global_error;

void error(const char* file_name, int line_number, const char* message, int flags = default_error_flags)
{
    bool verbose = (flags & global_error) ? (sirius::mpi_world.rank() == 0) : true;
    char header[1024];

    if (flags & fatal_error)
        sprintf(header, "\n=== Fatal error at line %i of file %s", line_number, file_name);
    else
        sprintf(header, "\n=== Warning at line %i of file %s", line_number, file_name);
    
    if (verbose)
        printf("%s\n%s\n\n", header, message);

    if (flags & fatal_error) 
    {
        sirius::mpi_world.abort();
        // raise(SIGTERM);
        // exit(0);
    }
}

void error(const char* file_name, int line_number, const std::string& message, int flags = default_error_flags)
{
    error(file_name, line_number, message.c_str(), flags);
}

void error(const char* file_name, int line_number, const std::stringstream& message, int flags = default_error_flags)
{
    error(file_name, line_number, message.str().c_str(), flags);
}

void warning(const char* file_name, int line_number, const std::stringstream& message, 
             int flags = default_warning_flags)
{
    error(file_name, line_number, message.str().c_str(), flags);
}

#endif // __ERROR_HANDLING_H__


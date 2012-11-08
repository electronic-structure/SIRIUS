#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__

/*
    There are two types of errors: fatal and non-fatal. A non-fatal error is a warining. Errors
    can be local (happened on a single MPI rank, for example matrix diagonalization failed on some node) 
    and global (happened globally, for example charge density is incorrect after k-point summation).
    Global errors are reported by a single (root) MPI rank. Local errors are reported by each failed MPI rank.

    Examples:

        error(__FILE__, __LINE__, message, global_msg | fatal_err); // fatal global error (default) 

        error(__FILE__, __LINE__, message, global_msg); // global non-fatal (warning) message 

*/

const int global_msg = 1 << 0;
const int fatal_err = 1 << 1;

const int default_error_flags = global_msg | fatal_err;
const int default_warning_flags = global_msg;

void error(const char* file_name, int line_number, const char* message, int flags = default_error_flags)
{
    bool verbose = (flags & global_msg) ? (MPIWorld::verbose()) : true;
    char header[1024];

    if (flags & fatal_err)
        sprintf(header, "\n=== Fatal error at line %i of file %s\n=== MPI rank: %i", 
                line_number, file_name, MPIWorld::rank());
    else
        sprintf(header, "\n=== Warning at line %i of file %s\n=== MPI rank: %i", 
                line_number, file_name, MPIWorld::rank());
    
    if (verbose)
        printf("%s\n%s\n\n", header, message);

    if (flags & fatal_err) 
    {
        // give writing ranks some time to flush the output buffer 
        double delay_time = 0.3;
        timeval t1;
        timeval t2;
        double d;

        gettimeofday(&t1, NULL);
        do
        {
          gettimeofday(&t2, NULL);
          d = double(t2.tv_sec - t1.tv_sec) + double(t2.tv_usec - t1.tv_usec) / 1e6;
        } while (d < delay_time);
 
        MPIWorld::abort();
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


// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__

/** \file error_handling.h
    
    \brief Simple error handling.
    
    There are two types of errors: fatal and non-fatal. A non-fatal error is a warining. Errors can be 
    local (happened on a single MPI rank, for example matrix diagonalization failed on some node) 
    and global (happened globally, for example charge density is incorrect after k-point summation).
    Global errors are reported by a single (root) MPI rank. Local errors are reported by each failed MPI rank.

    Examples:

        error_local(__FILE__, __LINE__, "Every MPI rank will print this message");
        
        std::stringstream s;
        s << "This is global warning message";
        warning_global(__FILE__, __LINE__, s);
*/

const int _global_message_ = 1 << 0;
const int _fatal_error_ = 1 << 1;

/// General error report
void error_message(const char* file_name, int line_number, const std::string& message, int flags)
{
    bool verbose = (flags & _global_message_) ? (Platform::mpi_rank() == 0) : true;
    if (verbosity_level >= 10) verbose = true;
    
    std::vector<char> buffer(message.size() + 1000);

    int n;
    if (flags & _fatal_error_)
    {
        n = sprintf(&buffer[0], "\n=== Fatal error at line %i of file %s ===", line_number, file_name);
    }
    else
    {
        n = sprintf(&buffer[0], "\n=== Warning at line %i of file %s ===", line_number, file_name);
    }
    int n1 = n - 1;

    if (!(flags & _global_message_)) n += sprintf(&buffer[n], "\n=== MPI rank: %i ===", Platform::mpi_rank());
    
    if (verbose) 
    {
        n += sprintf(&buffer[n], "\n%s\n", message.c_str());
        for (int i = 0; i < n1; i++) n += sprintf(&buffer[n], "-");
        printf("%s\n", &buffer[0]);
    }
    
    if (flags & _fatal_error_) 
    {
        // give writing ranks some time to flush the output buffer 
        double delay_time = 3.5;
        timeval t1;
        timeval t2;
        double d;

        gettimeofday(&t1, NULL);
        do
        {
            gettimeofday(&t2, NULL);
            d = double(t2.tv_sec - t1.tv_sec) + double(t2.tv_usec - t1.tv_usec) / 1e6;
        } while (d < delay_time);

        Platform::abort();
    }
}

/// Global error report for string message
void error_global(const char* file_name, int line_number, const std::string& message)
{
    error_message(file_name, line_number, message, _global_message_ | _fatal_error_);
}

/// Global error report for stringstream message
void error_global(const char* file_name, int line_number, const std::stringstream& message)
{
    error_message(file_name, line_number, message.str(), _global_message_ | _fatal_error_);
}

/// Loal error report for string message
void error_local(const char* file_name, int line_number, const std::string& message)
{
    error_message(file_name, line_number, message, _fatal_error_);
}

/// Local error report for stringstream message
void error_local(const char* file_name, int line_number, const std::stringstream& message)
{
    error_message(file_name, line_number, message.str(), _fatal_error_);
}
/// Global warning report for string message
void warning_global(const char* file_name, int line_number, const std::string& message)
{
    error_message(file_name, line_number, message, _global_message_);
}

/// Global warning report for stringstream message
void warning_global(const char* file_name, int line_number, const std::stringstream& message)
{
    error_message(file_name, line_number, message.str(), _global_message_);
}

/// Loal warning report for string message
void warning_local(const char* file_name, int line_number, const std::string& message)
{
    error_message(file_name, line_number, message, 0);
}

/// Local warning report for stringstream message
void warning_local(const char* file_name, int line_number, const std::stringstream& message)
{
    error_message(file_name, line_number, message.str(), 0);
}

#endif // __ERROR_HANDLING_H__


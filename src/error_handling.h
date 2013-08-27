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

//const int default_error_flags = fatal_err;
//const int default_warning_flags = 0;

/// General error report
void error_message(const char* file_name, int line_number, const std::string& message, int flags)
{
    bool verbose = (flags & _global_message_) ? (Platform::mpi_rank() == 0) : true;
    char header[1024];

    if (flags & _fatal_error_)
    {
        sprintf(header, "\n=== Fatal error at line %i of file %s\n=== MPI rank: %i", 
                line_number, file_name, Platform::mpi_rank());
    }
    else
    {
        sprintf(header, "\n=== Warning at line %i of file %s\n=== MPI rank: %i", 
                line_number, file_name, Platform::mpi_rank());
    }
    
    if (verbose) printf("%s\n%s\n\n", header, message.c_str());

    if (flags & _fatal_error_) 
    {
        // give writing ranks some time to flush the output buffer 
        double delay_time = 0.5;
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

//** 
//** 
//** 
//** 
//** 
//** 
//** void error(const char* file_name, int line_number, const char* message, int flags = default_error_flags)
//** {
//**     bool verbose = (flags & global_msg) ? (Platform::verbose()) : true;
//**     char header[1024];
//** 
//**     if (flags & fatal_err)
//**     {
//**         sprintf(header, "\n=== Fatal error at line %i of file %s\n=== MPI rank: %i", 
//**                 line_number, file_name, Platform::mpi_rank());
//**     }
//**     else
//**     {
//**         sprintf(header, "\n=== Warning at line %i of file %s\n=== MPI rank: %i", 
//**                 line_number, file_name, Platform::mpi_rank());
//**     }
//**     
//**     if (verbose) printf("%s\n%s\n\n", header, message);
//** 
//**     if (flags & fatal_err) 
//**     {
//**         // give writing ranks some time to flush the output buffer 
//**         double delay_time = 0.5;
//**         timeval t1;
//**         timeval t2;
//**         double d;
//** 
//**         gettimeofday(&t1, NULL);
//**         do
//**         {
//**           gettimeofday(&t2, NULL);
//**           d = double(t2.tv_sec - t1.tv_sec) + double(t2.tv_usec - t1.tv_usec) / 1e6;
//**         } while (d < delay_time);
//**  
//**         Platform::abort();
//**     }
//** }
//** 
//** void error(const char* file_name, int line_number, const std::string& message, int flags = default_error_flags)
//** {
//**     error(file_name, line_number, message.c_str(), flags);
//** }
//** 
//** void error(const char* file_name, int line_number, const std::stringstream& message, int flags = default_error_flags)
//** {
//**     error(file_name, line_number, message.str().c_str(), flags);
//** }
//** 
//** void warning(const char* file_name, int line_number, const std::stringstream& message, 
//**              int flags = default_warning_flags)
//** {
//**     error(file_name, line_number, message.str().c_str(), flags);
//** }
//** 
#endif // __ERROR_HANDLING_H__


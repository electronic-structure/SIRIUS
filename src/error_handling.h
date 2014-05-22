// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file error_handling.h
 *  
 *  \brief Simple error handling.
 *  
 *  There are two types of errors: fatal and non-fatal. A non-fatal error is a warining. Errors can be 
 *  local (happened on a single MPI rank, for example matrix diagonalization failed on some node) 
 *  and global (happened globally, for example charge density is incorrect after k-point summation).
 *  Global errors are reported by a single (root) MPI rank. Local errors are reported by each failed MPI rank.
 *
 *  Examples:
 *
 *      error_local(__FILE__, __LINE__, "Every MPI rank will print this message");
 *      
 *      std::stringstream s;
 *      s << "This is global warning message";
 *      warning_global(__FILE__, __LINE__, s);
 */

#ifndef __ERROR_HANDLING_H__
#define __ERROR_HANDLING_H__

#include <sys/time.h>
#include <sstream>
#include <string>
#include "config.h"
#include "platform.h"
#include "timer.h"

const int _global_message_ = 1 << 0;
const int _fatal_error_ = 1 << 1;

void terminate(const char* file_name, int line_number, const std::string& message);

void terminate(const char* file_name, int line_number, const std::stringstream& message);

/// General error report
void error_message(const char* file_name, int line_number, const std::string& message, int flags);

/// Global error report for string message
void error_global(const char* file_name, int line_number, const std::string& message);

/// Global error report for stringstream message
void error_global(const char* file_name, int line_number, const std::stringstream& message);

/// Loal error report for string message
void error_local(const char* file_name, int line_number, const std::string& message);

/// Local error report for stringstream message
void error_local(const char* file_name, int line_number, const std::stringstream& message);

/// Global warning report for string message
void warning_global(const char* file_name, int line_number, const std::string& message);

/// Global warning report for stringstream message
void warning_global(const char* file_name, int line_number, const std::stringstream& message);

/// Loal warning report for string message
void warning_local(const char* file_name, int line_number, const std::string& message);

/// Local warning report for stringstream message
void warning_local(const char* file_name, int line_number, const std::stringstream& message);

void log_function_enter(const char* func_name);

void log_function_exit(const char* func_name);

#define stop_here Timer::print(); error_local(__FILE__, __LINE__, "stop_here macros is called");

#define TERMINATE_NO_GPU terminate(__FILE__, __LINE__, "not compiled with GPU support");

#define INFO std::cout << "[" << __func__ << ":" << Platform::mpi_rank() << "] "

#endif // __ERROR_HANDLING_H__


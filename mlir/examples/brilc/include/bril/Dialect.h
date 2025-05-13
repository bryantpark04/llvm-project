//===- Dialect.h - Dialect definition for the Bril IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Bril language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_BRIL_DIALECT_H_
#define MLIR_TUTORIAL_BRIL_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the bril
/// dialect.
#include "bril/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// bril operations.
#define GET_OP_CLASSES
#include "bril/Ops.h.inc"

#endif // MLIR_TUTORIAL_BRIL_DIALECT_H_

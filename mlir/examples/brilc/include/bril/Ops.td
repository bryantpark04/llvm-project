//===- Ops.td - Bril dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Bril dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BRIL_OPS
#define BRIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'bril' dialect in the ODS framework so that we
// can define our operations.
def Bril_Dialect : Dialect {
  let name = "bril";
  let cppNamespace = "::mlir::bril";
}

// Base class for bril dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Bril_Op<string mnemonic, list<Trait> traits = []> :
    Op<Bril_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Bril Operations
//===----------------------------------------------------------------------===//

def IDOp : Bril_Op<"id", [Pure]> {
  let summary = "A type-insensitive identity.";
  let description = [{
    Takes one argument, which is a variable of any type, and produces the same value.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  
  let builders = [
    OpBuilder<(ins "Value":$input), [{
      build($_builder, $_state, input.getType(), input);
    }]>
  ];
}

def Nop : Bril_Op<"nop", []> {
  let summary = "No-op.";
  let description = [{
    A no-op.
  }];
  
  let arguments = (ins);
  let results = (outs);
  
  let assemblyFormat = "attr-dict";
}

def PrintOp : Bril_Op<"print", [MemoryEffects<[MemWrite]>]> {
  let summary = "Print operation";
  let description = [{
    Prints a value to standard output.
    ```
  }];
  
  let arguments = (ins AnyType:$value);
  let results = (outs);
  
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // BRIL_OPS

//===- brilc.cpp - The Bril Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Bril compiler.
//
//===----------------------------------------------------------------------===//

#include "bril/Dialect.h"
#include "bril/MLIRGen.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <iostream>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "bril/MLIR2JSON.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"


#include <nlohmann/json.hpp>

using json = nlohmann::json;


using namespace bril;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input bril file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Bril, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Bril), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Bril, "bril", "load the input file as a Bril source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpMLIR, convertMLIR };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(convertMLIR, "convert", "convert the MLIR to json")));


int dumpAST(json &j) {
  std::cout << j.dump(2) << std::endl;
  return 0;
}

int dumpMLIR(json& j) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::bril::BrilDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

  // dumpAST(j);

  mlir::OwningOpRef<mlir::ModuleOp> module = ::mlirGen(context, j);
  if (!module)
    return 1;

  module->print(llvm::outs());

  return 0;
}

int emitJSON() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::bril::BrilDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> memBuffer = llvm::MemoryBuffer::getSTDIN();
  if (std::error_code ec = memBuffer.getError()) {
    llvm::errs() << "Could not read STDIN: " << ec.message() << "\n";
    return -1;
  }
  
  llvm::SourceMgr sourceMgr;
  
  sourceMgr.AddNewSourceBuffer(std::move(*memBuffer), llvm::SMLoc());
  
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    std::cerr << "Error parsing MLIR input from stdin\n";
    return 1;
  }

  json j = bril::mlirToJson(*module);

  std::cout << j.dump() << std::endl;

  return 0;
}

int main(int argc, char **argv) {

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "bril compiler\n");

  json j;

  switch (emitAction) {
  case Action::DumpAST:
    std::cin >> j;
    return dumpAST(j);
  case Action::DumpMLIR:
    std::cin >> j;
    return dumpMLIR(j);
  case Action::convertMLIR:
    return emitJSON();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}

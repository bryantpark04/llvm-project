#include "bril/MLIRGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "bril/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

using namespace mlir::bril;
using namespace bril;

using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::cf;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

using json = nlohmann::json;

// print json for debugging
void dbg(json& j) {
  if (true) {
    std::cout << j.dump(2) << std::endl;
  }
}

// define DenseMapInfo for strings so we can use DenseMap
namespace llvm {
  template <>
  struct DenseMapInfo<std::string> {
    static inline std::string getEmptyKey() {
      return "";
    }

    static inline std::string getTombstoneKey() {
      return "\0";
    }

    static unsigned getHashValue(const std::string &Val) {
      return std::hash<std::string>()(Val);
    }

    static bool isEqual(const std::string &LHS, const std::string &RHS) {
      return LHS == RHS;
    }
  };
}


namespace bril {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(json &j) {
    brilModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    // generate MLIR for each function
    for (auto &f : j["functions"]) {
      if (failed(mlirGenFunction(f))) {
        brilModule.emitError("error generating function");
        return nullptr;
      }
    }

    // verify program
    if (failed(mlir::verify(brilModule))) {
      brilModule.emitError("module verification error");
      return nullptr;
    }

    return brilModule;
  }

private:
  mlir::ModuleOp brilModule;
  mlir::OpBuilder builder;

  // convert bril types into MLIR types
  mlir::Type getBrilType(const std::string &typeStr) {
    if (typeStr == "int") return builder.getI32Type();
    if (typeStr == "bool") return builder.getI1Type();
    if (typeStr == "float") return builder.getF32Type();
    return nullptr;
  }

  // store phi node data
  struct PhiInfo {
    std::string dest;
    std::vector<std::string> args;
    std::vector<std::string> labels;
    std::string type;
  };

  // generate MLIR for a function and add to module
  llvm::LogicalResult mlirGenFunction(json &func) {
    // reset builder
    builder.clearInsertionPoint();
    auto loc = builder.getUnknownLoc();
    
    std::string name = func["name"];

    // get function arg types
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for(auto& arg : func["args"]){
      std::string typeStr = arg["type"];
      auto type = getBrilType(typeStr);
      if(!type){
        // std::cout << "no MLIR type for bril type: " << typeStr << std::endl;
        return mlir::failure();
      }
      argTypes.push_back(type);
    }

    // get return type
    std::optional<llvm::SmallVector<mlir::Type, 1>> returnType;
    if(func.contains("type")){
      std::string typeStr = func["type"];
      auto type = getBrilType(typeStr);
      returnType->push_back(type);
    }

    // create function type
    auto funcType = builder.getFunctionType(argTypes, *returnType);
    auto function = builder.create<mlir::func::FuncOp>(loc, name, funcType);
    brilModule.push_back(function);
    
    // create blocks and collect phis
    llvm::DenseMap<std::string, mlir::Block*> labelToBlock;
    llvm::DenseMap<std::string, std::vector<PhiInfo>> blockPhis;

    // set entry block
    mlir::Block *entry = function.addEntryBlock();
    labelToBlock[name + "_entry"] = entry;

    // collect phis
    for (auto &instr : func["instrs"]) {
      if (instr.contains("label")) {
        std::string label = instr["label"];
        if (labelToBlock.count(label) == 0) {
          labelToBlock[label] = function.addBlock();
        }
      } else if (instr.contains("op") && instr["op"] == "phi") {
        std::string blockLabel = name + "_entry";
        
        // find block label for this phi
        for (auto it = &instr-1; it >= &func["instrs"][0]; it--) {
          if (it->contains("label")) {
            blockLabel = (*it)["label"];
            break;
          }
        }
        
        // store in PhiInfo
        PhiInfo phi;
        phi.dest = instr["dest"];
        phi.type = instr["type"];
        for (const auto& arg : instr["args"]) {
          phi.args.push_back(arg);
        }
        for (const auto& label : instr["labels"]) {
          phi.labels.push_back(label);
        }
        
        // store in blockPhis
        blockPhis[blockLabel].push_back(phi);
      }
    }

    // get block args
    for (const auto& pair : blockPhis) {
      // get block and phi node data
      std::string blockLabel = pair.first;
      const std::vector<PhiInfo>& phis = pair.second;
      mlir::Block* block = labelToBlock[blockLabel];
      
      // add args
      for (const auto& phi : phis) {
        mlir::Type argType = getBrilType(phi.type);
        block->addArgument(argType, loc);
      }
    }

    // initial symbol table
    llvm::DenseMap<std::string, mlir::Value> symbolTable;
    for (size_t i = 0; i < func["args"].size(); ++i) {
      std::string argName = func["args"][i]["name"];
      symbolTable[argName] = entry->getArgument(i);
    }

    // start at entry
    mlir::Block *curBlock = entry;
    builder.setInsertionPointToStart(entry);

    // map phi dests to block args
    llvm::DenseMap<std::string, mlir::BlockArgument> phiDestToArg;
    
    // iterate over and process instructions
    int blockArgIndex = 0;
    for (auto &instr : func["instrs"]) {
      // dbg(instr);
      if (instr.contains("label")) {
        // find new block
        std::string label = instr["label"];
        curBlock = labelToBlock[label];
        builder.setInsertionPointToStart(curBlock);
        
        // make sure block does not end up empty
        if (curBlock->empty()) {
          builder.create<Nop>(loc);
        }
        
        // map block args
        blockArgIndex = 0;
        if (blockPhis.count(label) > 0) {
          for (const auto& phi : blockPhis[label]) {
            mlir::BlockArgument arg = curBlock->getArgument(blockArgIndex++);
            symbolTable[phi.dest] = arg;
            phiDestToArg[phi.dest] = arg;
          }
        }
      } else if (instr.contains("op")) {
        std::string op = instr["op"];

        // phis handled above
        if (op == "phi") {
          continue;
        }
        
        if (op == "jmp") {
          std::string target = instr["labels"][0];
          mlir::Block* targetBlock = labelToBlock[target];
          
          // check block args
          if (blockPhis.count(target) > 0) {
            llvm::SmallVector<mlir::Value, 4> blockArgs;
            
            for (const auto& phi : blockPhis[target]) {
              // get label
              int labelIndex = -1;
              std::string currentLabel;
              for (const auto& [label, block] : labelToBlock) {
                if (block == curBlock) {
                  currentLabel = label;
                  break;
                }
              }
              
              // find index in phi labels
              for (size_t i = 0; i < phi.labels.size(); i++) {
                if (phi.labels[i] == currentLabel) {
                  labelIndex = i;
                  break;
                }
              }
              
              if (labelIndex >= 0) {
                // pass in phi arg
                std::string argName = phi.args[labelIndex];
                mlir::Value argValue = symbolTable[argName];
                blockArgs.push_back(argValue);
              } else {
                std::cout << "Error: Could not find matching predecessor in phi" << std::endl;
                return mlir::failure();
              }
            }
            
            // create jump with phi args
            builder.create<BranchOp>(loc, targetBlock, blockArgs);
          } else {
            // create jump with no phi args
            builder.create<BranchOp>(loc, targetBlock);
          }
        } else if (op == "br") {
          // get fields
          auto cond = symbolTable[instr["args"][0]];
          std::string trueTarget = instr["labels"][0];
          std::string falseTarget = instr["labels"][1];
          
          // find target blocks
          mlir::Block* trueBlock = labelToBlock[trueTarget];
          mlir::Block* falseBlock = labelToBlock[falseTarget];

          // TODO: dedup
          // get args for true
          llvm::SmallVector<mlir::Value, 4> trueBlockArgs;
          if (blockPhis.count(trueTarget) > 0){
            std::string currentLabel;
            for (const auto& [label, block] : labelToBlock){
              if (block == curBlock) {
                currentLabel = label;
                break;
              }
            }
            
            for (const auto& phi : blockPhis[trueTarget]){
              int labelIndex = -1;
              for (size_t i = 0; i < phi.labels.size(); i++) {
                if (phi.labels[i] == currentLabel) {
                  labelIndex = i;
                  break;
                }
              }
              
              if (labelIndex >= 0){
                std::string argName = phi.args[labelIndex];
                trueBlockArgs.push_back(symbolTable[argName]);
              } else {
                std::cerr << "Error: Could not find matching predecessor for true branch in phi" << std::endl;
                return mlir::failure();
              }
            }
          }
          
          // get args for false
          llvm::SmallVector<mlir::Value, 4> falseBlockArgs;
          if (blockPhis.count(falseTarget) > 0) {
            std::string currentLabel;
            for (const auto& [label, block] : labelToBlock) {
              if (block == curBlock) {
                currentLabel = label;
                break;
              }
            }
            
            for (const auto& phi : blockPhis[falseTarget]) {
              int labelIndex = -1;
              for (size_t i = 0; i < phi.labels.size(); i++) {
                if (phi.labels[i] == currentLabel) {
                  labelIndex = i;
                  break;
                }
              }
              
              if (labelIndex >= 0) {
                std::string argName = phi.args[labelIndex];
                falseBlockArgs.push_back(symbolTable[argName]);
              } else {
                std::cerr << "Error: Could not find matching predecessor for false branch in phi" << std::endl;
                return mlir::failure();
              }
            }
          }
          
          // create branch with block args
          builder.create<CondBranchOp>(loc, cond, trueBlock, trueBlockArgs, falseBlock, falseBlockArgs);
        } else if (failed(mlirGenInstr(instr, symbolTable, loc))) {
          std::cout << "failed on instruction " << instr.dump(2) << std::endl;
          return mlir::failure();
        }
      }
    }

    // make sure all blocks have terminator
    for (auto &block : function.getBody()) {
      // skip if already has terminator
      if (!block.empty() && block.back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
        continue;
      }
      
      // insert terminator at end
      builder.setInsertionPointToEnd(&block);
      
      if (&block == &function.getBody().back()) {
        // return if last block
        builder.create<mlir::func::ReturnOp>(loc);
      } else {
        // branch to next block
        auto nextBlockIt = std::next(block.getIterator());
        if (nextBlockIt != function.getRegion().end()){
          mlir::Block *nextBlock = &(*nextBlockIt);
          
          // get next block
          std::string nextBlockLabel;
          for (const auto& [label, block] : labelToBlock) {
            if (block == nextBlock) {
              nextBlockLabel = label;
              break;
            }
          }
          
          // FIXME: does this even make sense
          if (blockPhis.count(nextBlockLabel) > 0) {
            builder.create<mlir::func::ReturnOp>(loc);
          } else {
            builder.create<BranchOp>(loc, nextBlock);
          }
        } else {
          builder.create<mlir::func::ReturnOp>(loc);
        }
      }
    }
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrNop(json& instr, mlir::Location loc){
    builder.create<Nop>(loc);
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrPrint(json& instr, mlir::Location loc, mlir::Value value){
    // TODO: support printing multiple values???
    if (!value)
      return mlir::failure();
    
    builder.create<mlir::bril::PrintOp>(loc, value);
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrConst(json& instr, mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, std::string type){
    if (type == "int") {
        int value = instr["value"];
        auto constOp = builder.create<ConstantIntOp>(loc, value, 32);
        symTab[dest] = constOp;
        return mlir::success();
      }

      // bool
      if(type == "bool"){
        bool value = instr["value"].get<bool>();
        auto constOp = builder.create<ConstantIntOp>(loc, value, 1);
        symTab[dest] = constOp;
        return mlir::success();
      }

      return mlir::failure();
  }

  llvm::LogicalResult mlirGenInstrArithBinop(std::string op, mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, mlir::Value lhs, mlir::Value rhs){
    if (!lhs || !rhs)
      return mlir::failure();

    mlir::Value result;
    if (op == "add") result = builder.create<AddIOp>(loc, lhs, rhs);
    else if (op == "mul") result = builder.create<MulIOp>(loc, lhs, rhs);
    else if (op == "sub") result = builder.create<SubIOp>(loc, lhs, rhs);
    else if (op == "div") result = builder.create<DivSIOp>(loc, lhs, rhs);

    symTab[dest] = result;
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrComp(std::string op, mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, mlir::Value lhs, mlir::Value rhs){
      if (!lhs || !rhs)
        return mlir::failure();

      CmpIPredicate predicate;
      if (op == "eq") predicate = CmpIPredicate::eq;
      else if (op == "lt") predicate = CmpIPredicate::slt;
      else if (op == "gt") predicate = CmpIPredicate::sgt;
      else if (op == "le") predicate = CmpIPredicate::sle;
      else if (op == "ge") predicate = CmpIPredicate::sge;
      else return mlir::failure(); 

      auto cmp = builder.create<CmpIOp>(loc, predicate, lhs, rhs);
      symTab[dest] = cmp;
      return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrLogicBinop(std::string op, mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, mlir::Value lhs, mlir::Value rhs){
    if (!lhs || !rhs)
        return mlir::failure();

    mlir::Value result;
    if (op == "and") result = builder.create<AndIOp>(loc, lhs, rhs);
    else if (op == "or") result = builder.create<OrIOp>(loc, lhs, rhs);

    symTab[dest] = result;
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrNot(mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, mlir::Value arg){
    if (!arg)
        return mlir::failure();

    auto one = builder.create<ConstantIntOp>(loc, 1, 1);
    auto result = builder.create<XOrIOp>(loc, arg, one);
    symTab[dest] = result;
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrId(mlir::Location loc, llvm::DenseMap<std::string, mlir::Value> &symTab, std::string dest, mlir::Value input){
    if (!input)
        return mlir::failure();

    auto idOp = builder.create<mlir::bril::IDOp>(loc, input.getType(), input);
    symTab[dest] = idOp.getResult();
    return mlir::success();
  }

  llvm::LogicalResult mlirGenInstrRet(mlir::Location loc, llvm::SmallVector<mlir::Value> returnVals){
    builder.create<mlir::func::ReturnOp>(loc, returnVals);
    return mlir::success(); 
  }

  llvm::LogicalResult mlirGenInstr(json &instr, llvm::DenseMap<std::string, mlir::Value> &symTab, mlir::Location loc){
    // get fields
    std::string op = instr["op"];
    std::vector<std::string> args = instr.value("args", std::vector<std::string>());
    std::string dest = instr.value("dest", "");
    std::string type = instr.value("type", "");

    // lookup for var names
    auto get = [&](const std::string &name) -> mlir::Value {
      auto it = symTab.find(name);
      return it != symTab.end() ? it->second : nullptr;
    };

    // nop
    if (op == "nop") {
      return mlirGenInstrNop(instr, loc);
    }

    // print
    if (op == "print") {
      auto value = get(instr["args"][0]);
      return mlirGenInstrPrint(instr, loc, value);
    }

    // const
    if (op == "const") {
      return mlirGenInstrConst(instr, loc, symTab, dest, type);
    }

    // arithmetic binops
    if (op == "add" || op == "mul" || op == "sub" || op == "div") {
       auto lhs = get(args[0]);
       auto rhs = get(args[1]);
      return mlirGenInstrArithBinop(op, loc, symTab, dest, lhs, rhs);
    }

    if (op == "eq" || op == "lt" || op == "gt" || op == "le" || op == "ge") {
      auto lhs = get(args[0]);
      auto rhs = get(args[1]);
      return mlirGenInstrComp(op, loc, symTab, dest, lhs, rhs);
    }

    // logical binops
    if (op == "and" || op == "or") {
      auto lhs = get(args[0]);
      auto rhs = get(args[1]);
      return mlirGenInstrLogicBinop(op, loc, symTab, dest, lhs, rhs);
    }
    
    // not
    if (op == "not") {
      auto arg = get(args[0]);
      return mlirGenInstrNot(loc, symTab, dest, arg);
    }
    
    // id
    if (op == "id") {
      auto input = get(args[0]);
      return mlirGenInstrId(loc, symTab, dest, input);
    }

    // function call
    if(op == "call"){
      // get args
      llvm::SmallVector<mlir::Value, 4> operands;
      for(const auto& arg: args){
        operands.push_back(get(arg));
      }

      std::string funcName = instr["funcs"][0];

      // get function return type
      llvm::SmallVector<mlir::Type, 1> resultTypes;
      if (instr.contains("dest")) {
        std::string typeName = instr["type"];
        mlir::Type resultType = getBrilType(typeName);
        resultTypes.push_back(resultType);
      }

      // create CallOp
      auto callOp = builder.create<mlir::func::CallOp>(loc, resultTypes, funcName, operands);
      
      // assign if non-void
      if (instr.contains("dest")) {
        std::string destName = instr["dest"];
        mlir::Value result = callOp.getResult(0);
        symTab[destName] = result;
      }
      return mlir::success(); 
    }

    // ret
    if(op == "ret"){
      llvm::SmallVector<mlir::Value> returnVals;
      for(const auto& arg: args){
        returnVals.push_back(get(arg));
      }
      return mlirGenInstrRet(loc, returnVals);
    }

    return mlir::failure();
  }
};

// entry point
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, json &j) {
  return mlir::OwningOpRef<mlir::ModuleOp>(
      MLIRGenImpl(context).mlirGen(j));
}

}
#include "bril/MLIR2JSON.h"
#include "bril/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using json = nlohmann::json;
using namespace mlir;

namespace {

class MLIRToJSONConverter {
public:
  MLIRToJSONConverter() = default;
  
  json convertModule(mlir::ModuleOp module);

private:
  json convertFunction(mlir::func::FuncOp funcOp);
  json convertBlock(mlir::Block &block, llvm::DenseMap<mlir::Block*, std::string> &blockNames);
  json convertOperation(mlir::Operation &op, llvm::DenseMap<mlir::Block*, std::string> &blockNames);
  
  void analyzeControlFlow(mlir::func::FuncOp funcOp);
  void gatherPhiValues(mlir::func::FuncOp funcOp);
  void generatePhiNodes(mlir::Block &block, json &instrs, llvm::DenseMap<mlir::Block*, std::string> &blockNames);

  llvm::DenseMap<mlir::Value, std::string> valueToName;
  int nextVarId = 0;
  std::string getValueName(mlir::Value value);
  std::string getTypeString(mlir::Type type);
  
  struct CFGNode {
    mlir::Block* block = nullptr;
    llvm::SmallVector<mlir::Block*, 4> predecessors;
    llvm::SmallVector<mlir::Block*, 4> successors;
  };
  
  llvm::DenseMap<mlir::Block*, CFGNode> cfgNodes;
  
  // BlockArg -> (Predecessor -> Value)
  llvm::DenseMap<mlir::Value, llvm::DenseMap<mlir::Block*, mlir::Value>> phiValues;
  
  llvm::DenseMap<mlir::Block*, std::string> currentBlockNames;
};

std::string MLIRToJSONConverter::getValueName(mlir::Value value) {
  auto it = valueToName.find(value);
  if (it != valueToName.end()) {
    return it->second;
  }
  
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner()->isEntryBlock() && 
        isa<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      std::string name = "arg" + std::to_string(blockArg.getArgNumber());
      valueToName[value] = name;
      return name;
    }
  }
  
  std::string name = "v" + std::to_string(nextVarId++);
  valueToName[value] = name;
  return name;
}

std::string MLIRToJSONConverter::getTypeString(mlir::Type type) {
  if (type.isIntOrIndex()) {
    if (type.isInteger(1)) {
      return "bool";
    }
    return "int";
  }
  if (type.isF32()) {
    return "float";
  }
  return "int";
}

void MLIRToJSONConverter::analyzeControlFlow(mlir::func::FuncOp funcOp) {
  cfgNodes.clear();
  
  for (auto &block : funcOp.getBody()) {
    CFGNode &node = cfgNodes[&block];
    node.block = &block;
  }
  
  for (auto &block : funcOp.getBody()) {
    if (block.empty()) continue;
    
    Operation &terminator = block.back();
    
    if (auto brOp = dyn_cast<mlir::cf::BranchOp>(&terminator)) {
      mlir::Block *dest = brOp.getDest();
      cfgNodes[&block].successors.push_back(dest);
      cfgNodes[dest].predecessors.push_back(&block);
    }
    else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(&terminator)) {
      mlir::Block *trueDest = condBrOp.getTrueDest();
      mlir::Block *falseDest = condBrOp.getFalseDest();
      
      cfgNodes[&block].successors.push_back(trueDest);
      cfgNodes[&block].successors.push_back(falseDest);
      
      cfgNodes[trueDest].predecessors.push_back(&block);
      cfgNodes[falseDest].predecessors.push_back(&block);
    }
  }
}

void MLIRToJSONConverter::gatherPhiValues(mlir::func::FuncOp funcOp) {
  phiValues.clear();
  
  for (auto &block : funcOp.getBody()) {
    if (block.empty()) continue;
    
    Operation &terminator = block.back();
    
    if (auto brOp = dyn_cast<mlir::cf::BranchOp>(&terminator)) {
      mlir::Block *dest = brOp.getDest();
      
      for (unsigned i = 0; i < brOp.getNumOperands(); ++i) {
        if (i < dest->getNumArguments()) {
          mlir::Value destArg = dest->getArgument(i);
          mlir::Value sourceVal = brOp.getOperand(i);
          phiValues[destArg][&block] = sourceVal;
        }
      }
    }
    else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(&terminator)) {
      mlir::Block *trueDest = condBrOp.getTrueDest();
      for (unsigned i = 0; i < condBrOp.getNumTrueOperands(); ++i) {
        if (i < trueDest->getNumArguments()) {
          mlir::Value destArg = trueDest->getArgument(i);
          mlir::Value sourceVal = condBrOp.getTrueOperand(i);
          phiValues[destArg][&block] = sourceVal;
        }
      }
      
      mlir::Block *falseDest = condBrOp.getFalseDest();
      for (unsigned i = 0; i < condBrOp.getNumFalseOperands(); ++i) {
        if (i < falseDest->getNumArguments()) {
          mlir::Value destArg = falseDest->getArgument(i);
          mlir::Value sourceVal = condBrOp.getFalseOperand(i);
          phiValues[destArg][&block] = sourceVal;
        }
      }
    }
  }
}

void MLIRToJSONConverter::generatePhiNodes(mlir::Block &block, json &instrs, 
                                           llvm::DenseMap<mlir::Block*, std::string> &blockNames) {
  for (auto arg : block.getArguments()) {
    json phiInstr;
    phiInstr["op"] = "phi";
    phiInstr["dest"] = getValueName(arg);
    phiInstr["type"] = getTypeString(arg.getType());
    
    auto it = phiValues.find(arg);
    if (it != phiValues.end()) {
      std::vector<std::string> argNames;
      std::vector<std::string> labelNames;

      auto &node = cfgNodes[&block];
      for (auto *pred : node.predecessors) {
        std::string predName = blockNames[pred];
        
        auto predValIt = it->second.find(pred);
        if (predValIt != it->second.end()) {
          argNames.push_back(getValueName(predValIt->second));
          labelNames.push_back(predName);
        } else {
          if (arg.getArgNumber() < block.getParent()->getArguments().size()) {
            mlir::Value funcArg = block.getParent()->getArgument(arg.getArgNumber());
            argNames.push_back(getValueName(funcArg));
          } else {
            argNames.push_back("__undefined");
          }
          labelNames.push_back(predName);
        }
      }
      
      if (!argNames.empty()) {
        phiInstr["args"] = argNames;
        phiInstr["labels"] = labelNames;
        instrs.push_back(phiInstr);
      } else if (block.isEntryBlock()) {
        phiInstr["args"] = json::array();
        phiInstr["labels"] = json::array();
        instrs.push_back(phiInstr);
      }
    } else if (block.isEntryBlock()) {
      phiInstr["args"] = json::array();
      phiInstr["labels"] = json::array();
      instrs.push_back(phiInstr);
    }
  }
}

json MLIRToJSONConverter::convertOperation(mlir::Operation &op, llvm::DenseMap<mlir::Block*, std::string> &blockNames) {
  json instr;
  
  if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(&op)) {
    instr["op"] = "const";
    instr["dest"] = getValueName(constOp->getResult(0));
    instr["type"] = getTypeString(constOp->getResult(0).getType());
    
    if (constOp->getResult(0).getType().isInteger(1)) {
      auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (intAttr) {
        bool boolValue = intAttr.getInt() != 0;
        instr["value"] = boolValue;
      } else {
        instr["value"] = false;
      }
    } else {
      auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (intAttr) {
        int64_t intValue = intAttr.getInt();
        instr["value"] = intValue;
      } else {
        instr["value"] = 0;
      }
    }
  }
  else if (auto addOp = dyn_cast<mlir::arith::AddIOp>(&op)) {
    instr["op"] = "add";
    instr["dest"] = getValueName(addOp->getResult(0));
    instr["type"] = getTypeString(addOp->getResult(0).getType());
    instr["args"] = {
      getValueName(addOp.getLhs()),
      getValueName(addOp.getRhs())
    };
  }
  else if (auto mulOp = dyn_cast<mlir::arith::MulIOp>(&op)) {
    instr["op"] = "mul";
    instr["dest"] = getValueName(mulOp->getResult(0));
    instr["type"] = getTypeString(mulOp->getResult(0).getType());
    instr["args"] = {
      getValueName(mulOp.getLhs()),
      getValueName(mulOp.getRhs())
    };
  }
  else if (auto subOp = dyn_cast<mlir::arith::SubIOp>(&op)) {
    instr["op"] = "sub";
    instr["dest"] = getValueName(subOp->getResult(0));
    instr["type"] = getTypeString(subOp->getResult(0).getType());
    instr["args"] = {
      getValueName(subOp.getLhs()),
      getValueName(subOp.getRhs())
    };
  }
  else if (auto divOp = dyn_cast<mlir::arith::DivSIOp>(&op)) {
    instr["op"] = "div";
    instr["dest"] = getValueName(divOp->getResult(0));
    instr["type"] = getTypeString(divOp->getResult(0).getType());
    instr["args"] = {
      getValueName(divOp.getLhs()),
      getValueName(divOp.getRhs())
    };
  }
  else if (auto cmpOp = dyn_cast<mlir::arith::CmpIOp>(&op)) {
    std::string brilOp;
    switch (cmpOp.getPredicate()) {
      case mlir::arith::CmpIPredicate::eq: brilOp = "eq"; break;
      case mlir::arith::CmpIPredicate::ne: brilOp = "ne"; break;
      case mlir::arith::CmpIPredicate::slt: brilOp = "lt"; break;
      case mlir::arith::CmpIPredicate::sle: brilOp = "le"; break;
      case mlir::arith::CmpIPredicate::sgt: brilOp = "gt"; break;
      case mlir::arith::CmpIPredicate::sge: brilOp = "ge"; break;
      default: brilOp = "eq"; 
    }
    
    instr["op"] = brilOp;
    instr["dest"] = getValueName(cmpOp->getResult(0));
    instr["type"] = "bool";
    instr["args"] = {
      getValueName(cmpOp.getLhs()),
      getValueName(cmpOp.getRhs())
    };
  }
  else if (auto andOp = dyn_cast<mlir::arith::AndIOp>(&op)) {
    instr["op"] = "and";
    instr["dest"] = getValueName(andOp->getResult(0));
    instr["type"] = "bool";
    instr["args"] = {
      getValueName(andOp.getLhs()),
      getValueName(andOp.getRhs())
    };
  }
  else if (auto orOp = dyn_cast<mlir::arith::OrIOp>(&op)) {
    instr["op"] = "or";
    instr["dest"] = getValueName(orOp->getResult(0));
    instr["type"] = "bool";
    instr["args"] = {
      getValueName(orOp.getLhs()),
      getValueName(orOp.getRhs())
    };
  }
  else if (auto xorOp = dyn_cast<mlir::arith::XOrIOp>(&op)) {
    if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(xorOp.getRhs().getDefiningOp())) {
      auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (intAttr && intAttr.getInt() == 1) {
        instr["op"] = "not";
        instr["dest"] = getValueName(xorOp->getResult(0));
        instr["type"] = "bool";
        instr["args"] = {getValueName(xorOp.getLhs())};
        return instr;
      }
    }
    
    instr["op"] = "not";
    instr["dest"] = getValueName(xorOp->getResult(0));
    instr["type"] = "bool";
    instr["args"] = {getValueName(xorOp.getLhs())};
  }
  else if (auto idOp = dyn_cast<mlir::bril::IDOp>(&op)) {
    instr["op"] = "id";
    instr["dest"] = getValueName(idOp->getResult(0));
    instr["type"] = getTypeString(idOp->getResult(0).getType());
    instr["args"] = {getValueName(idOp.getInput())};
  }
  else if (auto printOp = dyn_cast<mlir::bril::PrintOp>(&op)) {
    instr["op"] = "print";
    instr["args"] = json::array();
    for (auto value : printOp.getOperands()) {
      instr["args"].push_back(getValueName(value));
    }
  }
  else if (auto nopOp = dyn_cast<mlir::bril::Nop>(&op)) {
    instr["op"] = "nop";
  }
  else if (auto brOp = dyn_cast<mlir::cf::BranchOp>(&op)) {
    instr["op"] = "jmp";
    mlir::Block *destBlock = brOp.getDest();
    instr["labels"] = { blockNames[destBlock] };
  }
  else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(&op)) {
    instr["op"] = "br";
    instr["args"] = {getValueName(condBrOp.getCondition())};
    
    mlir::Block *trueBlock = condBrOp.getTrueDest();
    mlir::Block *falseBlock = condBrOp.getFalseDest();
    
    instr["labels"] = {
      blockNames[trueBlock],
      blockNames[falseBlock]
    };
  }
  else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(&op)) {
    instr["op"] = "ret";
    
    if (returnOp.getNumOperands() > 0) {
      std::vector<std::string> args;
      for (auto operand : returnOp.getOperands()) {
        args.push_back(getValueName(operand));
      }
      instr["args"] = args;
    }
  }
  else if (auto callOp = dyn_cast<mlir::func::CallOp>(&op)) {
    instr["op"] = "call";
    instr["funcs"] = {callOp.getCallee().str()};
    
    std::vector<std::string> args;
    for (auto operand : callOp.getOperands()) {
      args.push_back(getValueName(operand));
    }
    instr["args"] = args;
    
    if (callOp.getNumResults() > 0) {
      instr["dest"] = getValueName(callOp.getResult(0));
      instr["type"] = getTypeString(callOp.getResult(0).getType());
    }
  }
  else {
    instr["op"] = "nop";
    instr["comment"] = "Unknown MLIR operation: " + op.getName().getStringRef().str();
  }
  
  return instr;
}

json MLIRToJSONConverter::convertBlock(mlir::Block &block, llvm::DenseMap<mlir::Block*, std::string> &blockNames) {
  json instrs = json::array();
  
  if (!block.isEntryBlock()) {
    json labelInstr;
    labelInstr["label"] = blockNames[&block];
    instrs.push_back(labelInstr);
  }

  generatePhiNodes(block, instrs, blockNames);
  
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (!isa<mlir::cf::BranchOp>(&op) && !isa<mlir::cf::CondBranchOp>(&op)) {
        instrs.push_back(convertOperation(op, blockNames));
      }
      continue;
    }

    instrs.push_back(convertOperation(op, blockNames));
  }
  
  if (!block.empty()) {
    Operation &terminator = block.back();
    if (terminator.hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (isa<mlir::cf::BranchOp>(&terminator) || isa<mlir::cf::CondBranchOp>(&terminator)) {
        instrs.push_back(convertOperation(terminator, blockNames));
      }
    }
  }
  
  return instrs;
}

json MLIRToJSONConverter::convertFunction(mlir::func::FuncOp funcOp) {
  json func;
  func["name"] = funcOp.getName().str();
  
  valueToName.clear();
  nextVarId = 0;
  
  analyzeControlFlow(funcOp);
  
  gatherPhiValues(funcOp);
  
  llvm::DenseMap<mlir::Block*, std::string> blockNames;
  
  blockNames[&funcOp.getBody().front()] = ".b1";
  
  int blockId = 0;
  for (auto &block : funcOp.getBody()) {
    if (&block != &funcOp.getBody().front()) {
      blockNames[&block] = ".block" + std::to_string(blockId++);
    }
  }
  
  currentBlockNames = blockNames;
  
  json args = json::array();
  for (auto arg : funcOp.getArguments()) {
    json argObj;
    argObj["name"] = getValueName(arg);
    argObj["type"] = getTypeString(arg.getType());
    args.push_back(argObj);
  }
  func["args"] = args;
  
  json allInstrs = json::array();
  
  for (auto &block : funcOp.getBody()) {
    json blockInstrs = convertBlock(block, blockNames);
    for (auto &instr : blockInstrs) {
      allInstrs.push_back(instr);
    }
  }

  mlir::FunctionType fnType = funcOp.getFunctionType();
  if (fnType.getNumResults() > 0) {
    func["type"] = getTypeString(fnType.getResult(0));
  }
  
  func["instrs"] = allInstrs;
  return func;
}

json MLIRToJSONConverter::convertModule(mlir::ModuleOp module) {
  json j;
  json functions = json::array();
  
  for (auto funcOp : module.getOps<mlir::func::FuncOp>()) {
    functions.push_back(convertFunction(funcOp));
  }
  
  j["functions"] = functions;
  return j;
}

} // namespace

namespace bril {

nlohmann::json mlirToJson(mlir::ModuleOp module) {
  MLIRToJSONConverter converter;
  return converter.convertModule(module);
}

} // namespace bril